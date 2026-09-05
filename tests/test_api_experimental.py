import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from starlette.requests import Request

import breeze_infer.api as api
import breeze_infer.experimental as experimental
from test_api_cancellation import configure
from test_portable_runtime import Codec, Model


def settings(**changes):
    values = dict(
        model=Path("unused"),
        fast_all=False,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        device="mps",
    )
    return api.ApiSettings(**{**values, **changes})


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine", "reference"),
        ("attention", "sdpa"),
        ("quantization", "int8"),
        ("depth_cache", "compiled"),
        ("fast_codec", True),
        ("experimental_recipe", "unknown"),
    ],
)
def test_recipe_conflicts_reject(field, value):
    selected = settings(**{**dict(experimental_recipe=experimental.RECIPE), field: value})
    with pytest.raises(ValueError):
        experimental.validate_settings(selected, "mps")
    with pytest.raises(ValueError):
        experimental.validate_settings(settings(experimental_recipe=experimental.RECIPE), "cpu")


def test_actual_metadata_changes_fingerprint_not_request_metrics():
    base = dict(source_digest="source", model_digest="weights", runtime_fingerprint="old")
    metadata = dict(dependencies={"mlx": "0.32.0"}, runtime_settings={"bits": 8})
    first = experimental.resolved_identity(base, metadata)
    second = experimental.resolved_identity(base, {**metadata, "runtime_settings": {"bits": 4}})
    assert first["runtime_fingerprint"] != "old"
    assert first["runtime_fingerprint"] != second["runtime_fingerprint"]
    assert first == experimental.resolved_identity(base, metadata)
    assert "last_request" not in first and "busy" not in first
    assert len(json.dumps(first)) < 65536


def test_default_load_does_not_import_mlx(monkeypatch):
    reference = Model(1)
    monkeypatch.setitem(sys.modules, "breeze_infer.mlx_speech", None)
    monkeypatch.setattr(api, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(api, "load_runtime", lambda *a, **kw: (None, reference, None))
    monkeypatch.setattr(api, "update_generation_config_for_breeze", lambda _: None)
    monkeypatch.setattr(
        api,
        "PortableBreezeStreamingRuntime",
        lambda *a, **kw: SimpleNamespace(fast_enabled=False),
    )
    app = FastAPI()
    api._load_app(app, settings(device="cpu"))
    assert app.state.model is reference
    assert app.state.runtime_identity is None


def test_loaded_experiment_preserves_reference_and_resolves_identity(monkeypatch):
    reference, candidate = Model(1), Model(1)
    monkeypatch.setattr(api, "resolve_device", lambda _: "mps")
    monkeypatch.setattr(api, "load_runtime", lambda *a, **kw: (None, reference, None))
    monkeypatch.setattr(api, "update_generation_config_for_breeze", lambda _: None)
    monkeypatch.setattr(experimental, "dependency_identity", lambda: {})
    monkeypatch.setattr(experimental, "load_candidate", lambda *a: (candidate, {"actual_bits": 8}))
    monkeypatch.setattr(
        api,
        "PortableBreezeStreamingRuntime",
        lambda model, *a, **kw: SimpleNamespace(model=model, fast_enabled=False),
    )
    app = FastAPI()
    api._load_app(
        app,
        settings(
            experimental_recipe=experimental.RECIPE,
            runtime_identity={"source": "s"},
            runtime_fingerprint="preload",
        ),
    )
    assert app.state.model is reference and app.state.runtime.model is candidate
    assert app.state.runtime_identity["actual_bits"] == 8
    assert (
        app.state.runtime_fingerprint
        == app.state.runtime_identity["runtime_fingerprint"]
        != "preload"
    )


def configure_candidate(monkeypatch, *, bad_prefix=False):
    class Candidate(Model):
        def __init__(self):
            super().__init__(1)
            self.seeds = []

        def validate_inputs(self, inputs):
            assert "mlx_seed" not in inputs
            if bad_prefix:
                raise ValueError("bad prepared prefix")

        def generate(self, **kwargs):
            self.seeds.append(kwargs["mlx_seed"])
            super().generate(**kwargs)

    model, codec = Candidate(), Codec()
    runtime, lock = configure(monkeypatch, model, codec)
    monkeypatch.setattr(
        api,
        "_settings",
        settings(experimental_recipe=experimental.RECIPE, runtime_fingerprint="preload"),
    )
    monkeypatch.setattr(api.app.state, "runtime_fingerprint", "loaded", raising=False)
    return model, codec, runtime, lock


async def request(**changes):
    values = dict(
        text="hello",
        instruction="calm",
        cfg_scale=4.0,
        seed=17,
        ref_audio=None,
        ref_text="",
    )
    fingerprint = changes.pop("fingerprint", "loaded")
    scope = {"type": "http", "headers": [(b"x-breeze-runtime", fingerprint.encode())]}
    return await api.speech(Request(scope), **{**values, **changes})


@pytest.mark.parametrize(
    "changes",
    [
        dict(cfg_scale=1),
        dict(seed=-1),
        dict(seed=2**32),
        dict(seed=True),
        dict(ref_text="reference"),
        dict(ref_audio=UploadFile(io.BytesIO(b"audio"), filename="")),
        dict(fingerprint="preload"),
    ],
)
def test_unsupported_requests_fail_before_model_and_headers(monkeypatch, changes):
    model, codec, runtime, lock = configure_candidate(monkeypatch)
    monkeypatch.setattr(
        api, "prepare_inputs", lambda *a, **kw: pytest.fail("prepared rejected request")
    )
    with pytest.raises(HTTPException) as failure:
        asyncio.run(request(**changes))
    assert failure.value.status_code in (400, 409)
    assert not model.seeds and not codec.active and not lock.locked()


def test_invalid_prepared_prefix_rejects_before_codec(monkeypatch):
    model, codec, _, lock = configure_candidate(monkeypatch, bad_prefix=True)
    with pytest.raises(HTTPException, match="bad prepared prefix"):
        asyncio.run(request())
    assert not model.seeds and not codec.active and not lock.locked()


def test_request_seed_forwarding_and_loaded_header_are_consistent(monkeypatch):
    model, codec, _, lock = configure_candidate(monkeypatch)

    async def run():
        for seed in (17, 29, 17, 2**32 - 1):
            response = await request(seed=seed)
            assert response.headers["X-Breeze-Runtime"] == "loaded"
            assert [chunk async for chunk in response.body_iterator]
            assert not lock.locked() and not codec.active

    asyncio.run(run())
    assert model.seeds == [17, 29, 17, 2**32 - 1]
