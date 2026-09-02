from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from models.breeze import BreezeForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def load_model_config(ckpt_dir: Path, *, attn_implementation: str) -> Any:
    """Load Breeze config while applying attention choice to the nested encoder."""
    config = AutoConfig.from_pretrained(ckpt_dir)
    text_encoder_config = getattr(config, "text_encoder_config", None)
    if text_encoder_config is not None:
        text_encoder_config.preferred_attn_implementation = attn_implementation
    return config


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    config = load_model_config(ckpt_dir, attn_implementation=attn_implementation)
    model = BreezeForConditionalGeneration.from_pretrained(
        ckpt_dir,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model.to(device).eval()

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )
    return tokenizer, model, audio_tokenizer
