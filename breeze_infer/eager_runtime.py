"""Portable eager audio generation for non-CUDA Breeze devices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EagerStreamingChunk:
    """One decoded mono floating-point audio chunk."""

    audio: np.ndarray


class EagerBreezeStreamingRuntime:
    """Preserve the streaming API contract using Breeze's eager generation path.

    Non-CUDA devices cannot use the CUDA graph streaming runtime. Generation is
    therefore completed before decoded audio is divided into response chunks.
    """

    fast_enabled = False

    def __init__(
        self,
        model: Any,
        audio_tokenizer: Any,
        config: Any,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self.tokenizer = tokenizer
        self.config = config
        self.sample_rate = int(model.config.codec_config.sampling_rate)

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> Iterator[EagerStreamingChunk]:
        del request_id
        generated = self.model.generate(
            **inputs,
            output_audio=True,
            audio_tokenizer=self.audio_tokenizer,
        )
        audio = generated.audio if hasattr(generated, "audio") else generated
        if not audio:
            return
        tensor = audio[0]
        while tensor.dim() > 1:
            tensor = tensor[0]
        samples = tensor.detach().float().cpu().numpy()
        chunk_samples = self.sample_rate // 10
        for start in range(0, len(samples), chunk_samples):
            yield EagerStreamingChunk(samples[start : start + chunk_samples])
