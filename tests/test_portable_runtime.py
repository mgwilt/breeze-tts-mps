import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime


class Codec:
    def __init__(self):
        self.active = set()
        self.frames = []

    def open_request(self, request_id, **kwargs):
        assert not self.active
        self.active.add(request_id)

    def decode_request_chunk(self, request_id, codes, **kwargs):
        assert request_id in self.active
        assert codes.shape == (1, 3, 1)
        self.frames.append(codes.clone())
        return torch.full((1, 1, 1920), float(codes[0, 0, 0]) / 16)

    def close_request(self, request_id):
        self.active.remove(request_id)


class Model:
    config = SimpleNamespace(
        num_codebooks=3,
        codebook_pad_token_id=18,
        codec_config=SimpleNamespace(sampling_rate=24000, codebook_size=16),
    )
    generation_config = SimpleNamespace(pad_token_id=0)

    def __init__(self, count=3, fail=False):
        self.count, self.fail = count, fail
        self.finished = threading.Event()

    def generate(self, **kwargs):
        assert kwargs["output_audio"] is False
        assert "audio_tokenizer" not in kwargs
        streamer = kwargs["streamer"]
        try:
            streamer.put(torch.tensor([[101, 102]]))
            for _ in range(self.count):
                streamer.put(torch.tensor([[2, 3, 4]]))
            if self.fail:
                raise RuntimeError("generation failed")
            streamer.put(torch.tensor([[18, 18, 18]]))
            streamer.end()
        finally:
            self.finished.set()


def runtime(model):
    codec = Codec()
    return PortableBreezeStreamingRuntime(model, None, None, codec=codec, queue_capacity=1), codec


def test_streaming_frames_skip_prompt_and_codec_pad_not_text_pad():
    engine, codec = runtime(Model())
    chunks = list(engine.iter_audio_chunks({}))
    assert len(chunks) == 3
    assert np.concatenate([chunk.audio for chunk in chunks]).shape == (5760,)
    assert not codec.active
    assert engine.last_metrics["codec_frames"] == 3
    assert engine.last_metrics["completed"] is True


def test_first_chunk_before_generation_finishes_and_cancel_releases_full_queue():
    model = Model(100)
    engine, codec = runtime(model)
    iterator = engine.iter_audio_chunks({})
    next(iterator)
    assert not model.finished.is_set()
    iterator.close()
    assert model.finished.is_set()
    assert not codec.active
    assert engine.last_metrics["cancelled"] is True
    model.count = 1
    assert len(list(engine.iter_audio_chunks({}))) == 1


@pytest.mark.parametrize(
    "count,fail,match", [(0, False, "no audio"), (2, True, "generation failed")]
)
def test_errors_release_codec_and_never_mark_complete(count, fail, match):
    engine, codec = runtime(Model(count, fail))
    with pytest.raises(RuntimeError, match=match):
        list(engine.iter_audio_chunks({}))
    assert not codec.active
    assert engine.last_metrics["completed"] is False


def test_token_limit_without_eos_is_not_complete():
    class TruncatedModel(Model):
        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put(torch.tensor([[101]]))
            streamer.put(torch.tensor([[2, 3, 4]]))
            streamer.end()

    engine, codec = runtime(TruncatedModel())
    with pytest.raises(RuntimeError, match="without EOS"):
        list(engine.iter_audio_chunks({}))
    assert not engine.last_metrics["completed"] and not codec.active
