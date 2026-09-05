"""Isolated complete-utterance MLX candidate; never selected by live serving.

Torch retains text conditioning/prefill; MLX owns paired backbone continuation,
codebook0 and depth generation. The existing portable runtime owns codec state,
bounded delivery, cancellation and inference ownership. No weight export occurs.
"""

from copy import deepcopy
import math
import time

import mlx.core as mx
import numpy as np
import torch
from transformers import GenerationConfig

from breeze_infer.mlx_backbone import MLXBackbone
from breeze_infer.mlx_depth import MLXDepth, Sampling, from_torch
from breeze_infer.portable_runtime import GenerationCancelled

_INHERIT_QUANTIZATION = object()


def component_quantization(legacy, backbone, depth):
    """Resolve weight-only overrides before constructing either component."""

    def valid(value):
        return value is None or (type(value) is int and value == 8)

    if not valid(legacy):
        raise ValueError("Speech quantization requires None or integer8")
    selected = tuple(
        legacy if value is _INHERIT_QUANTIZATION else value for value in (backbone, depth)
    )
    if not all(valid(value) for value in selected):
        raise ValueError("Component quantization requires None or integer8")
    return selected


def sampling_settings(config, cfg, *, depth=False, books=16):
    """Fail closed on nondefault, unimplemented processors or decoding modes."""
    defaults = GenerationConfig().to_dict()
    supported = {
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "max_new_tokens",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "transformers_version",
        "_from_model_config",
        "depth_decoder_do_sample",
        "depth_decoder_temperature",
        "depth_decoder_top_k",
        "depth_decoder_top_p",
    }
    if depth:
        supported.add("min_new_tokens")
        if config.min_new_tokens not in (
            None,
            books - 1,
        ) or config.max_new_tokens not in (None, books - 1):
            raise ValueError("Unsupported depth generation length")
    for name, value in config.to_dict().items():
        if name not in supported and value != defaults.get(name):
            raise ValueError(f"Unsupported generation setting: {name}={value!r}")
    if type(config.do_sample) is not bool:
        raise ValueError("do_sample must be explicit boolean")
    if config.top_k is not None and type(config.top_k) is not int:
        raise ValueError("top_k must be an explicit integer")
    return Sampling(
        cfg=cfg,
        temperature=config.temperature if config.temperature is not None else 1.0,
        top_k=config.top_k if config.top_k is not None else 0,
        top_p=config.top_p if config.top_p is not None else 1.0,
        do_sample=config.do_sample,
    )


def backbone_scores(logits, settings, valid_size):
    """HF warpers then Breeze reserved masking; retain the separate EOS row."""
    conditional, unconditional = mx.split(logits.astype(mx.float32), 2, axis=0)
    scores = unconditional + settings.cfg * (conditional - unconditional)
    if settings.do_sample:
        scores = scores / settings.temperature
        if settings.top_k:
            k = min(settings.top_k, scores.shape[-1])
            threshold = mx.sort(scores, axis=-1)[..., -k, None]
            scores = mx.where(scores < threshold, -float("inf"), scores)
        if settings.top_p < 1:
            order = mx.argsort(scores, axis=-1)
            sorted_scores = mx.take_along_axis(scores, order, axis=-1)
            remove = mx.cumsum(mx.softmax(sorted_scores, axis=-1), axis=-1) <= (1 - settings.top_p)
            remove = mx.concatenate([remove[..., :-1], mx.zeros_like(remove[..., -1:])], axis=-1)
            remove = mx.take_along_axis(remove, mx.argsort(order, axis=-1), axis=-1)
            scores = mx.where(remove, -float("inf"), scores)
    index = mx.arange(scores.shape[-1])
    return mx.where((index < valid_size) | (index == scores.shape[-1] - 1), scores, -float("inf"))


def sample_backbone(logits, settings, valid_size, key):
    scores = backbone_scores(logits, settings, valid_size)
    if bool(mx.any(mx.isnan(scores) | mx.isposinf(scores))) or not bool(
        mx.all(mx.any(mx.isfinite(scores), axis=-1))
    ):
        raise ValueError("Invalid or fully filtered backbone distribution")
    if settings.do_sample:
        key, sample_key = mx.random.split(key)
        token = mx.random.categorical(scores, key=sample_key)
    else:
        token = mx.argmax(scores, axis=-1)
    return token.astype(mx.int32), key


class MLXSpeechModel:
    """Single paired-CFG streaming adapter for PortableBreezeStreamingRuntime."""

    def __init__(
        self,
        reference,
        *,
        cfg=4.0,
        quant_bits=None,
        compiled=True,
        backbone_quant_bits=_INHERIT_QUANTIZATION,
        depth_quant_bits=_INHERIT_QUANTIZATION,
    ):
        backbone_bits, depth_bits = component_quantization(
            quant_bits, backbone_quant_bits, depth_quant_bits
        )
        if not math.isfinite(cfg) or cfg <= 0 or cfg == 1:
            raise ValueError("Prototype requires explicit paired CFG, not CFG1")
        self.reference, self.config = reference, deepcopy(reference.config)
        if (
            self.config.num_codebooks != 16
            or self.config.vocab_size != 2051
            or self.config.codebook_pad_token_id != 2050
            or self.config.codec_config.codebook_size != 2048
        ):
            raise ValueError("Unsupported Breeze codec/EOS layout")
        self.generation_config = deepcopy(reference.generation_config)
        for name in ("do_sample", "temperature", "top_k", "top_p"):
            duplicate = "depth_decoder_" + name
            if hasattr(self.generation_config, duplicate) and getattr(
                self.generation_config, duplicate
            ) != getattr(reference.depth_decoder.generation_config, name):
                raise ValueError("Conflicting effective depth sampling settings")
        self.backbone_settings = sampling_settings(self.generation_config, cfg)
        self.depth_settings = sampling_settings(
            reference.depth_decoder.generation_config,
            cfg,
            depth=True,
            books=self.config.num_codebooks,
        )
        self.limit = self.generation_config.max_new_tokens
        if type(self.limit) is not int or not 1 <= self.limit <= 750:
            raise ValueError("Require explicit generation limit in1..750")
        self.valid_size = self.config.codec_config.codebook_size
        self.backbone = MLXBackbone(
            reference.backbone_model,
            head_weight=reference.lm_head.weight,
            quant_bits=backbone_bits,
        )
        self.depth = MLXDepth(
            reference.depth_decoder,
            valid_size=self.valid_size,
            attention_kind="sdpa",
            quant_bits=depth_bits,
        )
        self.step = self.backbone.step_runner(compiled=compiled)
        self.depth_generate = self.depth.generator(self.depth_settings, compiled=compiled)
        self.last_stages = {}

    def validate_inputs(self, inputs):
        """Validate both prepared branches without running model or codec work."""
        names = {
            "input_ids",
            "attention_mask",
            "text_ids_mask",
            "text_ids_len",
            "input_values",
        }
        negative_names = {
            "cfg_negative_prompt_ids": "input_ids",
            "cfg_negative_prompt_attention_mask": "attention_mask",
            "cfg_negative_text_ids_mask": "text_ids_mask",
            "cfg_negative_text_ids_len": "text_ids_len",
            "cfg_negative_input_values": "input_values",
        }
        unknown = inputs.keys() - names - negative_names.keys() - {"cfg_scale"}
        if unknown or inputs.get("cfg_scale") != self.backbone_settings.cfg:
            raise ValueError(f"Unsupported inputs or CFG recipe: {sorted(unknown)}")
        if (
            inputs.get("input_values") is not None
            or inputs.get("cfg_negative_input_values") is not None
        ):
            raise ValueError("Reference audio is not supported by this prototype")
        positive = {k: v for k, v in inputs.items() if k in names}
        negative = {
            target: inputs[source] for source, target in negative_names.items() if source in inputs
        }
        # Validate both branches before either runs; only prepared text prefixes
        # are accepted, never3D generated codes or partially specified text data.
        for branch in (positive, negative):
            if not {
                "input_ids",
                "attention_mask",
                "text_ids_mask",
                "text_ids_len",
            } <= branch.keys() or any(
                not isinstance(branch[name], torch.Tensor)
                for name in (
                    "input_ids",
                    "attention_mask",
                    "text_ids_mask",
                    "text_ids_len",
                )
            ):
                raise ValueError("Require complete prepared text prefix tensors")
            ids, mask, text_mask, lengths = (
                branch[name]
                for name in (
                    "input_ids",
                    "attention_mask",
                    "text_ids_mask",
                    "text_ids_len",
                )
            )
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype != torch.long:
                raise ValueError("Require one2D integer text prefix")
            if (
                mask.shape != ids.shape
                or text_mask.shape != ids.shape
                or text_mask.dtype != torch.bool
            ):
                raise ValueError("Invalid prepared text masks")
            if (
                not bool(torch.all((mask == 0) | (mask == 1)))
                or not bool(mask[0, -1])
                or not torch.equal(text_mask, mask.bool())
            ):
                raise ValueError("Require valid text-only masks with final token unpadded")
            if (
                lengths.ndim != 1
                or lengths.dtype != torch.long
                or not bool(torch.all(lengths > 0))
                or int(lengths.sum()) != int(text_mask.sum())
            ):
                raise ValueError("Invalid text segment lengths")
            if ids.shape[1] + self.limit > self.backbone.max_positions:
                raise ValueError("Prefix plus generation exceeds positional limit")
        return positive, negative

    def _prefill(self, inputs, check_stop=lambda: None):
        positive, negative = self.validate_inputs(inputs)
        outputs, masks = [], []
        start = time.perf_counter()
        for branch in (positive, negative):
            check_stop()
            mask = branch["attention_mask"]
            outputs.append(
                self.reference(
                    **branch,
                    position_ids=(mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1),
                    use_cache=True,
                    output_hidden_states=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
            )
            masks.append(mask)
        torch.mps.synchronize()
        prefill_s = time.perf_counter() - start
        start = time.perf_counter()
        state = self.backbone.pair_torch_caches([o.past_key_values for o in outputs], masks)
        hidden = from_torch(torch.cat([o.hidden_states[-1][:, -1] for o in outputs]))
        # Preserve actual reference first head logits; later heads run in MLX.
        logits = from_torch(torch.cat([o.logits[:, -1].float() for o in outputs]))
        mx.eval(hidden, logits, state)
        self.last_stages = dict(prefill_s=prefill_s, prefill_bridge_s=time.perf_counter() - start)
        return hidden, logits, state

    @torch.inference_mode()
    def generate(self, *, streamer, output_audio=False, mlx_seed=42, **inputs):
        if output_audio or type(mlx_seed) is not int or not 0 <= mlx_seed <= 2**32 - 1:
            raise ValueError("Require streaming codes and an explicit uint32 seed")

        def check_stop():
            if getattr(streamer, "stopped", None) is not None and streamer.stopped.is_set():
                raise GenerationCancelled()

        check_stop()
        self.last_stages = {}
        # Preserve existing streamer's initial-prompt handshake and cancel check.
        streamer.put(inputs["input_ids"])
        hidden, logits, state = self._prefill(inputs, check_stop)
        check_stop()
        key = mx.random.key(mlx_seed)
        for index in range(self.limit):
            check_stop()
            first, key = sample_backbone(logits, self.backbone_settings, self.valid_size, key)
            token = first.item()
            if token == self.config.vocab_size:
                streamer.put(
                    torch.full(
                        (1, self.config.num_codebooks),
                        self.config.codebook_pad_token_id,
                        dtype=torch.long,
                    )
                )
                streamer.end()
                return
            if not 0 <= token < self.valid_size:
                raise ValueError("Invalid first codebook token")
            check_stop()
            initial = mx.concatenate([mx.zeros((1, 1), dtype=mx.int32), first[:, None]], axis=-1)
            sequence, key = self.depth_generate(initial, hidden, key)
            mx.eval(sequence, key)
            frame = np.array(sequence[:, 1:]).astype(np.int64)
            if (
                frame.shape != (1, self.config.num_codebooks)
                or np.any(frame < 0)
                or np.any(frame >= self.valid_size)
            ):
                raise ValueError("Invalid generated depth frame")
            check_stop()
            streamer.put(torch.from_numpy(frame))
            check_stop()
            if index + 1 < self.limit:
                embeddings = self.backbone.audio_embeddings(sequence[:, 1:])
                hidden3, state = self.step(mx.concatenate([embeddings, embeddings]), state)
                logits = self.backbone.logits(hidden3).astype(mx.float32)
                hidden = hidden3[:, -1]
                mx.eval(hidden, logits, state)
        raise RuntimeError("Breeze stopped without EOS; output may be truncated")
