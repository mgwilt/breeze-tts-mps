"""Experimental Breeze depth decoder in MLX; not wired into serving.

Weights and rotary tables are copied from the locked Torch reference. No model
conversion is persisted and no MLX-LM/Transformers dependency migration is needed.
Each frame owns its functional KV state; conditional rows precede unconditional
rows and sampled codebook tokens are shared between the two branches.
"""

from dataclasses import dataclass
import math

import mlx.core as mx


def from_torch(tensor):
    """Explicit host bridge, measured separately from MLX execution by probes."""
    dtype = str(tensor.dtype)
    if dtype not in ("torch.float32", "torch.bfloat16"):
        raise ValueError(f"Unsupported reference precision: {dtype}")
    result = mx.array(tensor.detach().float().cpu().numpy())
    return result.astype(mx.bfloat16) if dtype == "torch.bfloat16" else result


def rms_norm(x, weight, epsilon):
    full = x.astype(mx.float32)
    normalized = full * mx.rsqrt(mx.mean(full * full, axis=-1, keepdims=True) + epsilon)
    return normalized.astype(x.dtype) * weight


def silu_product(gate, up):
    # Torch SiLU calculates internally in F32 but returns model precision before
    # multiplying the up projection. Compilation is a separate numeric candidate.
    full = gate.astype(mx.float32)
    return (full * mx.sigmoid(full)).astype(gate.dtype) * up


def rotary(x, cosine, sine):
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cosine + rotated * sine


def scale_attention_scores(scores, scale):
    # Torch's wrapped Python scalar is applied in F32 before the BF16 result.
    # MLX's BF16 * Python float otherwise rounds the scalar to BF16 first.
    return (scores.astype(mx.float32) * scale).astype(scores.dtype)


def attention(q, k, v, start, kind):
    length, total = q.shape[-2], k.shape[-2]
    mask = mx.arange(total)[None, :] <= (start + mx.arange(length))[:, None]
    scale = q.shape[-1] ** -0.5
    if kind == "sdpa":
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    groups = q.shape[1] // k.shape[1]
    k, v = mx.repeat(k, groups, axis=1), mx.repeat(v, groups, axis=1)
    scores = scale_attention_scores(q @ mx.swapaxes(k, -1, -2), scale)
    scores = mx.where(mask, scores, mx.array(-float("inf"), dtype=scores.dtype))
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return probs @ v


@dataclass(frozen=True)
class Sampling:
    cfg: float = 4.0
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        if not math.isfinite(self.cfg):
            raise ValueError("CFG must be finite")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if not isinstance(self.top_k, int) or self.top_k < 0 or not 0 <= self.top_p <= 1:
            raise ValueError("Invalid top-k/top-p")


def guided_logits(logits, cfg, valid_size):
    cond, uncond = mx.split(logits.astype(mx.float32), 2, axis=0)
    guided = uncond + cfg * (cond - uncond)
    return mx.where(mx.arange(guided.shape[-1]) < valid_size, guided, -float("inf"))


def filter_probabilities(probs, top_k, top_p):
    """Breeze's top-k renormalization followed by shifted, crossing-inclusive p."""
    order = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    if top_k > 0:
        sorted_probs = mx.where(mx.arange(probs.shape[-1]) < top_k, sorted_probs, 0)
        sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
    if top_p < 1:
        remove = mx.cumsum(sorted_probs, axis=-1) > top_p
        remove = mx.concatenate([mx.zeros_like(remove[..., :1]), remove[..., :-1]], axis=-1)
        sorted_probs = mx.where(remove, 0, sorted_probs)
        sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
    return mx.take_along_axis(sorted_probs, mx.argsort(order, axis=-1), axis=-1)


def sampling_probabilities(logits, settings, valid_size):
    scores = guided_logits(logits, settings.cfg, valid_size) / settings.temperature
    return filter_probabilities(mx.softmax(scores, axis=-1), settings.top_k, settings.top_p)


class MLXDepth:
    """Small functional implementation, deliberately limited to Breeze's shapes."""

    def __init__(self, reference, *, valid_size, attention_kind="eager", quant_bits=None):
        config = reference.config
        if reference.training:
            raise ValueError("Reference must be in evaluation mode")
        if attention_kind not in ("eager", "sdpa"):
            raise ValueError("Unknown attention candidate")
        if quant_bits not in (None, 8, 4):
            raise ValueError("Only isolated 8-bit/4-bit affine candidates are supported")
        if config.attention_bias or config.mlp_bias or config.hidden_act != "silu":
            raise ValueError("Only bias-free SiLU Breeze depth is supported")
        self.books = config.num_codebooks
        self.vocab = config.vocab_size
        self.valid_size = valid_size
        self.hidden = config.hidden_size
        self.backbone_hidden = config.backbone_hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.epsilon = config.rms_norm_eps
        self.layers = config.num_hidden_layers
        self.attention_kind = attention_kind
        self.quant_bits = quant_bits
        self.quant_group = 64
        self.quantized = {}
        if not (1 < self.books and 0 < valid_size <= self.vocab):
            raise ValueError("Invalid codebook dimensions")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Invalid grouped-query/rotary dimensions")
        self.weights = {name: from_torch(value) for name, value in reference.state_dict().items()}
        self.dtype = self.weights["model.embed_tokens.weight"].dtype
        if any(value.dtype != self.dtype for value in self.weights.values()):
            raise ValueError("Mixed reference precision is not supported")
        expected = (self.books - 1, self.hidden, self.vocab)
        if self.weights["codebooks_head.weight"].shape != expected:
            raise ValueError("Breeze codebook heads must be [books-1,hidden,vocab]")
        self.quantization = self._quantize_linears() if quant_bits else None
        # Preserve the locked reference's configured RoPE, including its frequency
        # scaling and device-specific trig rounding. Report this prototype bridge.
        import torch

        parameter = next(reference.parameters())
        with torch.inference_mode():
            positions = torch.arange(self.books, device=parameter.device)[None]
            cosine, sine = reference.model.rotary_emb(parameter, positions)
        self.cosine, self.sine = from_torch(cosine), from_torch(sine)
        mx.eval(self.weights, self.cosine, self.sine)

    def _quantize_linears(self):
        """Only named attention/MLP matrices; exclude all custom/output heads."""
        original_bytes = sum(value.nbytes for value in self.weights.values())
        records, covered_bytes = [], 0
        eligible = {
            f"model.layers.{index}.{module}.{name}_proj.weight"
            for index in range(self.layers)
            for module, names in (
                ("self_attn", ("q", "k", "v", "o")),
                ("mlp", ("gate", "up", "down")),
            )
            for name in names
        }
        if not eligible <= self.weights.keys():
            raise ValueError("Missing expected depth quantization weights")
        for name, weight in list(self.weights.items()):
            if name not in eligible:
                continue
            if weight.ndim != 2 or weight.shape[-1] % self.quant_group:
                raise ValueError(f"Unsupported quantization shape: {name} {weight.shape}")
            packed, scales, biases = mx.quantize(
                weight, group_size=self.quant_group, bits=self.quant_bits, mode="affine"
            )
            mx.eval(packed, scales, biases)
            records.append(
                dict(
                    name=name,
                    shape=list(weight.shape),
                    original_bytes=weight.nbytes,
                    packed_bytes=packed.nbytes + scales.nbytes + biases.nbytes,
                    scale_dtype=str(scales.dtype),
                )
            )
            covered_bytes += weight.nbytes
            self.quantized[name] = (packed, scales, biases)
            del self.weights[name]
        if self.quantized.keys() != eligible:
            raise ValueError("Expected exactly seven eligible linears per depth layer")
        return dict(
            mode="affine",
            bits=self.quant_bits,
            group_size=self.quant_group,
            layers=len(records),
            covered_original_bytes=covered_bytes,
            depth_original_bytes=original_bytes,
            coverage=covered_bytes / original_bytes,
            packed_bytes=sum(r["packed_bytes"] for r in records),
            records=records,
        )

    def _linear(self, x, name):
        if name + ".weight" in self.quantized:
            weight, scales, biases = self.quantized[name + ".weight"]
            return mx.quantized_matmul(
                x,
                weight,
                scales,
                biases,
                transpose=True,
                group_size=self.quant_group,
                bits=self.quant_bits,
                mode="affine",
            )
        return x @ self.weights[name + ".weight"].T

    def forward(self, ids, hidden=None, cache=()):
        """One prefix/continuation; cache contains only populated positions."""
        self._validate_ids(ids)
        return self._forward(ids, hidden, cache)

    def _validate_ids(self, ids):
        if ids.ndim != 2 or not ids.size or ids.dtype not in (mx.int32, mx.uint32, mx.int64):
            raise ValueError("Token IDs must be a nonempty integer matrix")
        if bool(mx.any(ids < 0)) or bool(mx.any(ids >= self.vocab)):
            raise ValueError("Token ID outside its codebook vocabulary")

    def _forward(self, ids, hidden=None, cache=()):
        """Trusted-token core: value validation stays outside compiled graphs."""
        start = cache[0][0].shape[-2] if cache else 0
        batch, length = ids.shape
        if ids.dtype not in (mx.int32, mx.uint32, mx.int64):
            raise ValueError("Token IDs must be integer")
        if length < 1 or start + length > self.books or (not cache and length < 2):
            raise ValueError("Invalid depth positions")
        if cache and (len(cache) != self.layers or hidden is not None):
            raise ValueError("Invalid continuation cache/hidden state")
        if not cache and (hidden is None or hidden.shape != (batch, self.backbone_hidden)):
            raise ValueError("Prefill requires matching backbone hidden rows")
        if hidden is not None and hidden.dtype != self.dtype:
            raise ValueError("Backbone hidden precision differs from depth weights")
        positions = mx.arange(start, start + length)
        offsets = mx.maximum(positions - 1, 0) * self.vocab
        x = self.weights["model.embed_tokens.weight"][ids + offsets]
        if hidden is not None:
            if "model.backbone_hidden_state_projector.weight" in self.weights:
                hidden = self._linear(hidden, "model.backbone_hidden_state_projector")
            x = mx.concatenate([hidden[:, None, :], x[:, 1:, :]], axis=1)
        x = self._linear(x, "model.inputs_embeds_projector")
        cosine = self.cosine[:, None, start : start + length, :]
        sine = self.sine[:, None, start : start + length, :]
        updated = []
        for index in range(self.layers):
            prefix = f"model.layers.{index}."
            norm = rms_norm(x, self.weights[prefix + "input_layernorm.weight"], self.epsilon)
            q, k, v = [
                self._linear(norm, prefix + f"self_attn.{name}_proj")
                .reshape(batch, length, heads, self.head_dim)
                .transpose(0, 2, 1, 3)
                for name, heads in (
                    ("q", self.heads),
                    ("k", self.kv_heads),
                    ("v", self.kv_heads),
                )
            ]
            q, k = rotary(q, cosine, sine), rotary(k, cosine, sine)
            if cache:
                old_k, old_v = cache[index]
                if (
                    old_k.shape != (batch, self.kv_heads, start, self.head_dim)
                    or old_v.shape != old_k.shape
                ):
                    raise ValueError("Inconsistent KV cache shape")
                if old_k.dtype != self.dtype or old_v.dtype != self.dtype:
                    raise ValueError("KV cache precision differs from depth weights")
                k, v = (
                    mx.concatenate([old_k, k], axis=2),
                    mx.concatenate([old_v, v], axis=2),
                )
            updated.append((k, v))
            attended = attention(q, k, v, start, self.attention_kind)
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, length, self.heads * self.head_dim
            )
            x = x + self._linear(attended, prefix + "self_attn.o_proj")
            norm = rms_norm(
                x,
                self.weights[prefix + "post_attention_layernorm.weight"],
                self.epsilon,
            )
            gate, up = [self._linear(norm, prefix + f"mlp.{name}_proj") for name in ("gate", "up")]
            x = x + self._linear(silu_product(gate, up), prefix + "mlp.down_proj")
        x = rms_norm(x, self.weights["model.norm.weight"], self.epsilon)
        head = start + length - 2
        logits = x[:, -1, :] @ self.weights["codebooks_head.weight"][head]
        return logits, tuple(updated)

    def generator(self, settings=Sampling(), *, compiled=False):
        """Return a whole-frame function; explicit key in/out, fresh cache per call.

        Inputs: initial [B,2] placeholder+c0, hidden [2B,H], PRNG key. Outputs:
        [B,books+1] including placeholder and a new key. No per-step host bridge.
        MLX/Torch random streams are NOT claimed to be seed-equivalent.
        """

        def generate(initial, hidden, key):
            if (
                initial.ndim != 2
                or initial.shape[1] != 2
                or hidden.shape[0] != 2 * initial.shape[0]
            ):
                raise ValueError("Generation requires paired CFG rows and placeholder+c0")
            sequence, cache = initial, ()
            for step in range(self.books - 1):
                ids = sequence if step == 0 else sequence[:, -1:]
                logits, cache = self._forward(
                    mx.concatenate([ids, ids]), hidden if step == 0 else None, cache
                )
                if settings.do_sample:
                    key, sample_key = mx.random.split(key)
                    probs = sampling_probabilities(logits, settings, self.valid_size)
                    selected = mx.random.categorical(mx.log(probs), key=sample_key)
                else:
                    selected = mx.argmax(
                        guided_logits(logits, settings.cfg, self.valid_size), axis=-1
                    )
                sequence = mx.concatenate(
                    [sequence, selected.astype(initial.dtype)[:, None]], axis=1
                )
            return sequence, key

        inner = mx.compile(generate) if compiled else generate

        def validated(initial, hidden, key):
            # Initial values come from the host boundary; subsequent samples are
            # reserved-token masked. Construct a new generator after weight reload.
            self._validate_ids(initial)
            return inner(initial, hidden, key)

        return validated
