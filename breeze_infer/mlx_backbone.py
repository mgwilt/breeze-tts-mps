"""Isolated Qwen3 backbone continuation for Breeze; not a serving backend.

Keep Torch text preparation/prefill, then transfer already-rotated branch KV once.
Logical positions remain per-row while left padding gives common physical cache
storage. State is explicit and immutable; no depth sampler or EOS rules are reused.
"""

import mlx.core as mx
import numpy as np

from breeze_infer.mlx_depth import (
    from_torch,
    rms_norm,
    scale_attention_scores,
    silu_product,
)


class MLXBackbone:
    def __init__(
        self,
        reference,
        *,
        head_weight=None,
        max_positions=4096,
        attention_kind="sdpa",
        quant_bits=None,
        cache_chunk=128,
    ):
        config = reference.layers[0].self_attn.config
        if reference.training or config.model_type != "qwen3":
            raise ValueError("Require an evaluation-mode Qwen3 backbone")
        if (
            config.attention_bias
            or config.hidden_act != "silu"
            or any(layer.self_attn.sliding_window is not None for layer in reference.layers)
        ):
            raise ValueError("Only bias-free, full-attention SiLU Qwen3 is supported")
        if attention_kind not in ("eager", "sdpa") or quant_bits not in (None, 8):
            raise ValueError("Unsupported backbone candidate")
        if max_positions < 1 or max_positions > config.max_position_embeddings:
            raise ValueError("Invalid rotary table bound")
        if type(cache_chunk) is not int or cache_chunk < 1:
            raise ValueError("Invalid cache chunk")
        if reference.embed_tokens.audio_embeds_projector is not None:
            raise ValueError("Audio embedding projection is not implemented")
        self.hidden, self.heads, self.kv_heads = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        self.head_dim, self.layers = config.head_dim, config.num_hidden_layers
        self.epsilon, self.max_positions = config.rms_norm_eps, max_positions
        self.cache_chunk = cache_chunk
        self.books, self.vocab = (
            reference.config.num_codebooks,
            reference.config.vocab_size,
        )
        self.attention_kind, self.quant_bits, self.quant_group = (
            attention_kind,
            quant_bits,
            64,
        )
        self.weights = {
            name: from_torch(value)
            for name, value in reference.state_dict().items()
            if name.startswith("layers.") or name == "norm.weight"
        }
        self.dtype = self.weights["norm.weight"].dtype
        self.embedding = from_torch(reference.embed_tokens.embed_audio_tokens.weight)
        self.head = from_torch(head_weight) if head_weight is not None else None
        if self.embedding.shape != (self.books * self.vocab, self.hidden):
            raise ValueError("Unexpected shared audio embedding shape")
        if self.head is not None and self.head.shape != (self.vocab + 1, self.hidden):
            raise ValueError("Backbone head must include the separate utterance EOS row")
        if (
            self.heads % self.kv_heads
            or any(w.dtype != self.dtype for w in self.weights.values())
            or self.embedding.dtype != self.dtype
            or (self.head is not None and self.head.dtype != self.dtype)
        ):
            raise ValueError("Unsupported GQA or mixed precision")
        self.quantized, self.quantization = {}, None
        if quant_bits:
            self.quantization = self._quantize()
        # Capture the actual nested Qwen3 reference's rotary coefficients. Do not
        # inherit top-level Breeze/MLXDepth theta, scaling, epsilon or positions.
        import torch

        parameter = reference.norm.weight
        with torch.inference_mode():
            positions = torch.arange(max_positions, device=parameter.device)[None]
            cosine, sine = reference.rotary_emb(parameter, positions)
        self.cosine, self.sine = from_torch(cosine[0]), from_torch(sine[0])
        half = self.head_dim // 2
        self.rotate_indices = mx.array(
            list(range(half, self.head_dim)) + list(range(half)), dtype=mx.int32
        )
        self.rotate_signs = mx.array([-1] * half + [1] * half, dtype=self.dtype)
        self.query_to_kv = mx.array(
            [head // (self.heads // self.kv_heads) for head in range(self.heads)],
            dtype=mx.int32,
        )
        mx.eval(self.weights, self.quantized, self.embedding, self.cosine, self.sine)
        if self.head is not None:
            mx.eval(self.head)

    def _quantize(self):
        names = {
            f"layers.{i}.{module}.{name}_proj.weight"
            for i in range(self.layers)
            for module, projections in (
                ("self_attn", ("q", "k", "v", "o")),
                ("mlp", ("gate", "up", "down")),
            )
            for name in projections
        }
        if not names <= self.weights.keys():
            raise ValueError("Missing Qwen3 projection weights")
        records = []
        original_bytes = sum(w.nbytes for w in self.weights.values())
        for name in sorted(names):
            weight = self.weights[name]
            if weight.ndim != 2 or weight.shape[-1] % self.quant_group:
                raise ValueError(f"Unsupported quantized shape: {name}")
            packed = mx.quantize(
                weight, group_size=self.quant_group, bits=self.quant_bits, mode="affine"
            )
            mx.eval(packed)
            records.append(
                dict(
                    name=name,
                    shape=list(weight.shape),
                    original_bytes=weight.nbytes,
                    packed_bytes=sum(x.nbytes for x in packed),
                )
            )
            self.quantized[name] = packed
            del self.weights[name]
        return dict(
            bits=self.quant_bits,
            mode="affine",
            group_size=self.quant_group,
            records=records,
            covered_original_bytes=sum(r["original_bytes"] for r in records),
            backbone_original_bytes=original_bytes,
            packed_bytes=sum(r["packed_bytes"] for r in records),
        )

    def _linear(self, x, name):
        if name + ".weight" in self.quantized:
            packed, scales, biases = self.quantized[name + ".weight"]
            return mx.quantized_matmul(
                x,
                packed,
                scales,
                biases,
                transpose=True,
                group_size=self.quant_group,
                bits=self.quant_bits,
                mode="affine",
            )
        return x @ self.weights[name + ".weight"].T

    def pair_torch_caches(self, caches, masks):
        """Return (KV buffers, validity mask, logical positions, physical offset)."""
        if len(caches) != 2 or len(masks) != 2:
            raise ValueError("Require one conditional/unconditional cache pair")
        lengths = [cache.get_seq_length() for cache in caches]
        if not all(lengths) or max(lengths) >= self.max_positions:
            raise ValueError("Invalid transferred cache lengths")
        width, pairs, valid = max(lengths), [], []
        capacity = min(self.max_positions, ((width // self.cache_chunk) + 1) * self.cache_chunk)
        for index, (mask, length) in enumerate(zip(masks, lengths)):
            if len(caches[index].layers) != self.layers or tuple(mask.shape) != (
                1,
                length,
            ):
                raise ValueError("Transferred cache/mask shape mismatch")
            host = mask.detach().float().cpu().numpy()
            if not np.all(np.isfinite(host) & ((host == 0) | (host == 1))):
                raise ValueError("Transferred mask must contain finite binary values")
            host = host.astype(np.bool_)
            if not host.any():
                raise ValueError("Empty logical prefix")
            valid.append(mx.array(np.pad(host, ((0, 0), (width - length, capacity - width)))))
        for layer in range(self.layers):
            branches = []
            for cache, length in zip(caches, lengths):
                entry = cache.layers[layer]
                expected = (1, self.kv_heads, length, self.head_dim)
                if tuple(entry.keys.shape) != expected or tuple(entry.values.shape) != expected:
                    raise ValueError("Transferred KV shape mismatch")
                key, value = (
                    from_torch(entry.keys.detach().clone()),
                    from_torch(entry.values.detach().clone()),
                )
                if key.dtype != self.dtype or value.dtype != self.dtype:
                    raise ValueError("Transferred KV precision mismatch")
                pad = ((0, 0), (0, 0), (width - length, capacity - width), (0, 0))
                branches.append((mx.pad(key, pad), mx.pad(value, pad)))
            pairs.append(
                tuple(
                    mx.concatenate([row[column] for row in branches], axis=0) for column in (0, 1)
                )
            )
        mask = mx.concatenate(valid)
        positions = mx.sum(mask.astype(mx.int32), axis=-1, keepdims=True)
        state = tuple(pairs), mask, positions, mx.array(width, dtype=mx.int32)
        mx.eval(state)
        return state

    def _step(self, x, cache, mask, positions, offset):
        batch = x.shape[0]
        valid = mx.slice_update(mask, mx.ones((batch, 1), dtype=mx.bool_), offset, (1,))
        cosine = mx.expand_dims(mx.take(self.cosine, positions, axis=0), axis=1)
        sine = mx.expand_dims(mx.take(self.sine, positions, axis=0), axis=1)
        attention_mask = mx.expand_dims(mx.expand_dims(valid, axis=1), axis=1)

        def rotate(value):
            swapped = mx.take(value, self.rotate_indices, axis=-1) * self.rotate_signs
            return value * cosine + swapped * sine

        updated = []
        for index in range(self.layers):
            prefix = f"layers.{index}."
            norm = rms_norm(x, self.weights[prefix + "input_layernorm.weight"], self.epsilon)
            q, k, v = [
                self._linear(norm, prefix + f"self_attn.{name}_proj").reshape(
                    batch, 1, heads, self.head_dim
                )
                for name, heads in (
                    ("q", self.heads),
                    ("k", self.kv_heads),
                    ("v", self.kv_heads),
                )
            ]
            q = rms_norm(
                q, self.weights[prefix + "self_attn.q_norm.weight"], self.epsilon
            ).transpose(0, 2, 1, 3)
            k = rms_norm(
                k, self.weights[prefix + "self_attn.k_norm.weight"], self.epsilon
            ).transpose(0, 2, 1, 3)
            q, k = rotate(q), rotate(k)
            k = mx.slice_update(cache[index][0], k, offset, (2,))
            v = mx.slice_update(cache[index][1], v.transpose(0, 2, 1, 3), offset, (2,))
            updated.append((k, v))
            if self.attention_kind == "sdpa":
                attended = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.head_dim**-0.5, mask=attention_mask
                )
            else:
                kr, vr = (
                    mx.take(k, self.query_to_kv, axis=1),
                    mx.take(v, self.query_to_kv, axis=1),
                )
                scores = scale_attention_scores(q @ mx.swapaxes(kr, -1, -2), self.head_dim**-0.5)
                scores = mx.where(attention_mask, scores, -float("inf"))
                probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(self.dtype)
                attended = probs @ vr
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, 1, self.heads * self.head_dim)
            x = x + self._linear(attended, prefix + "self_attn.o_proj")
            norm = rms_norm(
                x,
                self.weights[prefix + "post_attention_layernorm.weight"],
                self.epsilon,
            )
            gate, up = [self._linear(norm, prefix + f"mlp.{name}_proj") for name in ("gate", "up")]
            x = x + self._linear(silu_product(gate, up), prefix + "mlp.down_proj")
        return rms_norm(x, self.weights["norm.weight"], self.epsilon), (
            tuple(updated),
            valid,
            positions + 1,
            offset + 1,
        )

    def step_runner(self, *, compiled=False):
        # Fixed-capacity chunks avoid unsupported/stale shape inference in MLX
        # 0.32 shapeless attention. Only a capacity expansion needs a new graph.
        inner = mx.compile(self._step) if compiled else self._step

        def step(x, state):
            cache, mask, positions, offset = state
            if x.shape != (2, 1, self.hidden) or x.dtype != self.dtype:
                raise ValueError("Require paired one-token embeddings in model precision")
            if (
                len(cache) != self.layers
                or mask.ndim != 2
                or mask.shape[0] != 2
                or mask.dtype != mx.bool_
            ):
                raise ValueError("Invalid paired cache mask")
            if positions.shape != (2, 1) or positions.dtype not in (
                mx.int32,
                mx.uint32,
            ):
                raise ValueError("Invalid logical positions")
            if offset.shape != () or offset.dtype != mx.int32:
                raise ValueError("Invalid physical offset")
            physical = offset.item()
            if (
                bool(mx.any(positions < 0))
                or bool(mx.any(positions >= self.max_positions))
                or physical < 0
                or physical >= self.max_positions
                or physical > mask.shape[-1]
            ):
                raise ValueError("Backbone position bound exceeded")
            expected = (2, self.kv_heads, mask.shape[-1], self.head_dim)
            if any(
                k.shape != expected
                or v.shape != expected
                or k.dtype != self.dtype
                or v.dtype != self.dtype
                for k, v in cache
            ):
                raise ValueError("Invalid KV shape or precision")
            if bool(mx.any(mask[:, physical:])):
                raise ValueError("Future cache slots must not be valid")
            if physical == mask.shape[-1]:
                growth = min(self.cache_chunk, self.max_positions - physical)
                cache = tuple(
                    tuple(mx.pad(value, ((0, 0), (0, 0), (0, growth), (0, 0))) for value in pair)
                    for pair in cache
                )
                mask = mx.pad(mask, ((0, 0), (0, growth)))
            return inner(x, cache, mask, positions, offset)

        return step

    def audio_embeddings(self, tokens):
        if (
            tokens.ndim != 2
            or tokens.shape[1] != self.books
            or tokens.dtype not in (mx.int32, mx.uint32, mx.int64)
        ):
            raise ValueError("Invalid codebook frame")
        if bool(mx.any(tokens < 0)) or bool(mx.any(tokens >= self.vocab)):
            raise ValueError("Codebook ID outside embedding vocabulary")
        offsets = mx.arange(self.books) * self.vocab
        # Torch sum accumulates BF16 input in F32 before returning BF16.
        return mx.sum(
            self.embedding[tokens + offsets].astype(mx.float32), axis=1, keepdims=True
        ).astype(self.dtype)

    def logits(self, hidden):
        if self.head is None:
            raise ValueError("No backbone output head loaded")
        return hidden[:, -1, :] @ self.head.T
