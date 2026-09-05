from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from breeze_infer.eager_runtime import EagerBreezeStreamingRuntime
from breeze_infer.runtime import load_model_config, resolve_device


class _Model:
    config = SimpleNamespace(codec_config=SimpleNamespace(sampling_rate=24_000))

    def generate(self, **kwargs):
        assert kwargs["output_audio"] is True
        assert kwargs["audio_tokenizer"] == "codec"
        return [torch.arange(3_000, dtype=torch.float32)]


def test_eager_runtime_generates_then_yields_bounded_audio_chunks() -> None:
    runtime = EagerBreezeStreamingRuntime(
        _Model(),
        "codec",
        SimpleNamespace(),
    )

    chunks = list(runtime.iter_audio_chunks({"input_ids": torch.tensor([[1]])}))

    assert runtime.fast_enabled is False
    assert runtime.sample_rate == 24_000
    assert [len(chunk.audio) for chunk in chunks] == [2_400, 600]
    assert np.concatenate([chunk.audio for chunk in chunks]).tolist() == list(
        np.arange(3_000, dtype=np.float32)
    )


def test_resolve_device_prefers_mps_after_cuda() -> None:
    with (
        patch("breeze_infer.runtime.torch.cuda.is_available", return_value=False),
        patch(
            "breeze_infer.runtime.torch.backends.mps.is_available", return_value=True
        ),
    ):
        assert resolve_device() == "mps"


def test_model_config_applies_attention_choice_to_nested_encoder() -> None:
    config = SimpleNamespace(text_encoder_config=SimpleNamespace())
    with patch("breeze_infer.runtime.AutoConfig.from_pretrained", return_value=config):
        loaded = load_model_config("model", attn_implementation="eager")

    assert loaded is config
    assert config.text_encoder_config.preferred_attn_implementation == "eager"
