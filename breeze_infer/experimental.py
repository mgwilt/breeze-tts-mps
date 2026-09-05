"""Explicit, unaccepted local experiments; importing this module never loads MLX."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json

RECIPE = "mlx-int8-v1"
DEPENDENCIES = {
    "torch": "2.9.1",
    "transformers": "4.57.3",
    "qwen-tts": "0.1.1",
    "mlx": "0.32.0",
    "mlx-metal": "0.32.0",
}
METAL_ARTIFACTS = {
    "mlx/lib/libmlx.dylib": "1876795e05b3434925e745fbf6e9f0c8c0446b666224c9d881609ab353e94e51",
    "mlx/lib/mlx.metallib": "1518c08860738b08dc4563ddcf380a08dec4e6ad146c0d54888790e80656e9e3",
}


def validate_settings(settings, device):
    if settings.experimental_recipe != RECIPE:
        raise ValueError("Unknown experimental recipe")
    if (
        device != "mps"
        or settings.engine != "streaming"
        or settings.attention != "eager"
        or settings.quantization != "none"
        or settings.depth_cache != "dynamic"
        or settings.fast_all not in (None, False)
        or any(
            getattr(settings, name)
            for name in (
                "fast_text_encoder",
                "fast_backbone_prefill",
                "fast_backbone_decode",
                "fast_depth_decoder",
                "fast_codec",
            )
        )
    ):
        raise ValueError("mlx-int8-v1 requires MPS streaming with unchanged reference settings")


def validate_request(cfg_scale, seed, *, has_reference):
    if cfg_scale != 4.0 or has_reference:
        raise ValueError("mlx-int8-v1 supports explicit CFG4 instruction-only requests")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("mlx-int8-v1 requires a uint32 seed")


def dependency_identity():
    actual = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    if actual != DEPENDENCIES:
        raise ValueError(
            f"Experimental recipe requires its tested dependency versions: {DEPENDENCIES}"
        )
    return actual


def kernel_identity():
    distribution = importlib.metadata.distribution("mlx-metal")
    result = {}
    for path in distribution.files or ():
        if str(path).endswith(("mlx.metallib", "libmlx.dylib")):
            with distribution.locate_file(path).open("rb") as source:
                result[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    if result != METAL_ARTIFACTS:
        raise ValueError("Metal kernel artifacts do not match the tested recipe")
    return result


def _coverage(inventory):
    records = inventory["records"]
    return {
        **{key: value for key, value in inventory.items() if key != "records"},
        "linear_count": len(records),
        "inventory_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()
        ).hexdigest(),
    }


def load_candidate(reference, audio_tokenizer):
    dependencies, kernels = dependency_identity(), kernel_identity()
    from breeze_infer.mlx_speech import MLXSpeechModel

    candidate = MLXSpeechModel(reference, cfg=4, quant_bits=8, compiled=True)
    metadata = dict(
        experimental_recipe=RECIPE,
        performance_mode="experimental",
        release_accepted=False,
        engine="torch-prefill-mlx-decode",
        attention={"prefill": "eager", "backbone": "sdpa", "depth": "sdpa"},
        quantization="mlx-affine-int8-group64",
        depth_cache="mlx-functional",
        cached_depth_cfg=False,
        cfg_policy="explicit-4-instruction-only",
        sampling={
            "backbone": asdict(candidate.backbone_settings),
            "depth": asdict(candidate.depth_settings),
            "backbone_filter": "HF-temperature-threshold-topk-ascending-topp-then-reserved-mask",
            "depth_filter": "reserved-mask-exact-topk-shifted-descending-topp",
            "prng": "explicit-uint32-MLX-key-not-Torch-seed-equivalent",
            "max_new_tokens": candidate.limit,
        },
        runtime_settings={
            "text_prefill": "torch-eager-bfloat16",
            "backbone_dtype": str(candidate.backbone.dtype),
            "depth_dtype": str(candidate.depth.dtype),
            "codec_dtype": str(next(audio_tokenizer.model.parameters()).dtype),
            "compiled_backbone_depth": True,
            "backbone_cache_chunk": candidate.backbone.cache_chunk,
            "max_positions": candidate.backbone.max_positions,
            "backbone_quantization": _coverage(candidate.backbone.quantization),
            "depth_quantization": _coverage(candidate.depth.quantization),
            "quantization_exclusions": "embeddings,norms,projectors,custom/output heads,codec",
        },
        dependencies=dependencies,
        metal_artifacts=kernels,
    )
    return candidate, metadata


def resolved_identity(base, metadata):
    identity = {**base, **metadata}
    identity.pop("runtime_fingerprint", None)
    identity["runtime_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return identity
