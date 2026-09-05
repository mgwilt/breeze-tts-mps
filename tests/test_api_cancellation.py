import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch
from starlette.requests import Request

import breeze_infer.api as api
from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime
from test_portable_runtime import Codec, Model


class BlockingModel(Model):
    def __init__(self, hold_first=False):
        super().__init__(100)
        self.entered = threading.Event()
        self.hold_first = hold_first

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        try:
            streamer.put(torch.tensor([[101]]))
            self.entered.set()
            if self.hold_first:
                streamer.stopped.wait(timeout=2)
            for _ in range(100):
                streamer.put(torch.tensor([[2, 3, 4]]))
            streamer.end()
        finally:
            self.finished.set()


def configure(monkeypatch, model, codec):
    engine = PortableBreezeStreamingRuntime(model, None, None, codec=codec, queue_capacity=1)
    lock = threading.Lock()
    monkeypatch.setattr(api, "_request_lock", lock)
    monkeypatch.setattr(api, "_settings", SimpleNamespace(runtime_fingerprint="test"))
    monkeypatch.setattr(
        api, "prepare_inputs", lambda *args, **kwargs: {"input_ids": torch.tensor([[1]])}
    )
    for name in ("tokenizer", "audio_tokenizer", "model"):
        monkeypatch.setattr(api.app.state, name, object(), raising=False)
    monkeypatch.setattr(api.app.state, "runtime", engine, raising=False)
    return engine, lock


async def make_response(spec):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/speech",
        "headers": [],
        "asgi": {"spec_version": spec},
    }
    response = await api.speech(
        Request(scope), text="x", instruction="y", cfg_scale=4, ref_audio=None, ref_text="", seed=42
    )
    return scope, response


@pytest.mark.parametrize("fail_at", ["http.response.start", "http.response.body"])
def test_send_disconnect_always_releases_ownership(monkeypatch, fail_at):
    model, codec = BlockingModel(), Codec()
    engine, lock = configure(monkeypatch, model, codec)

    async def run():
        scope, response = await make_response("2.4")

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == fail_at:
                raise OSError("simulated disconnect")

        with pytest.raises(Exception):
            await response(scope, receive, send)

    asyncio.run(run())
    assert not lock.locked() and not codec.active and not engine._active.locked()
    if fail_at == "http.response.body":
        assert model.finished.is_set()
    else:
        assert not model.entered.is_set()


def test_disconnect_while_waiting_for_first_pcm(monkeypatch):
    model, codec = BlockingModel(hold_first=True), Codec()
    engine, lock = configure(monkeypatch, model, codec)

    async def run():
        scope, response = await make_response("2.3")

        async def receive():
            assert await asyncio.to_thread(model.entered.wait, 1)
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        await asyncio.wait_for(response(scope, receive, send), 3)

    asyncio.run(run())
    assert model.finished.is_set()
    assert not lock.locked() and not codec.active and not engine._active.locked()
    assert engine.last_metrics["cancelled"]


def test_codec_cleanup_failure_is_not_success_and_prevents_reuse():
    class BadCodec(Codec):
        def close_request(self, request_id):
            raise RuntimeError("cleanup failed")

    engine = PortableBreezeStreamingRuntime(
        Model(1), None, None, codec=BadCodec(), queue_capacity=1
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        list(engine.iter_audio_chunks({}))
    assert not engine.last_metrics["completed"]
    with pytest.raises(RuntimeError, match="restart"):
        list(engine.iter_audio_chunks({}))


def test_poisoned_runtime_rejects_before_success_headers(monkeypatch):
    engine, lock = configure(monkeypatch, Model(1), Codec())
    engine._poisoned = True
    with pytest.raises(api.HTTPException) as caught:
        asyncio.run(make_response("2.4"))
    assert caught.value.status_code == 503
    assert not lock.locked()


@pytest.mark.parametrize(
    "text,instruction,cfg",
    [("", "y", 4), ("x" * 4001, "y", 4), ("x", "y" * 2001, 4), ("x", "y", float("nan"))],
)
def test_invalid_requests_release_lock(monkeypatch, text, instruction, cfg):
    _, lock = configure(monkeypatch, Model(1), Codec())

    async def run():
        with pytest.raises(api.HTTPException) as error:
            await api.speech(
                Request({"type": "http", "headers": []}),
                text=text,
                instruction=instruction,
                cfg_scale=cfg,
                ref_audio=None,
                ref_text="",
                seed=42,
            )
        assert error.value.status_code == 400

    asyncio.run(run())
    assert not lock.locked()


def test_reference_unlink_failure_still_releases_lock(monkeypatch):
    _, lock = configure(monkeypatch, Model(1), Codec())

    class BadPath:
        def unlink(self, **kwargs):
            raise OSError("unlink failed")

    async def save(upload):
        return BadPath()

    monkeypatch.setattr(api, "_save_upload", save)
    monkeypatch.setattr(
        api, "prepare_inputs", lambda *a, **kw: (_ for _ in ()).throw(ValueError("prepare failed"))
    )

    async def run():
        with pytest.raises(OSError, match="unlink failed"):
            await api.speech(
                Request({"type": "http", "headers": []}),
                text="x",
                instruction="y",
                cfg_scale=4,
                ref_audio=SimpleNamespace(filename="ref.wav"),
                ref_text="ref",
                seed=42,
            )

    asyncio.run(run())
    assert not lock.locked()
