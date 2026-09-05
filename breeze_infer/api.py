"""Thin streaming API over the PyTorch Breeze inference runtime."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

from breeze_infer.eager_runtime import EagerBreezeStreamingRuntime
from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime
from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
OPTIONAL_AUDIO_FILE = File(None)


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool
    device: str | None = None
    engine: str = "streaming"
    attention: str = "eager"
    quantization: str = "none"
    runtime_fingerprint: str = ""
    depth_cache: str = "dynamic"


_settings: ApiSettings | None = None
_request_lock = threading.Lock()


class _OwnedStreamingResponse(StreamingResponse):
    """Release ownership even when sending headers fails before iteration."""

    def __init__(self, body, cleanup, **kwargs):
        self._owned_body, self._cleanup = body, cleanup
        super().__init__(body, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self._owned_body.aclose()
                finally:
                    self._cleanup()


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="breeze_ref_", suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            payload = await upload.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            temporary.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    started = time.perf_counter()
    device = resolve_device(settings.device)
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model,
        device=device,
        attn_implementation=settings.attention,
    )
    update_generation_config_for_breeze(model)
    model._cached_depth_cfg = settings.engine == "streaming"
    if settings.depth_cache != "dynamic":
        if settings.engine != "streaming" or settings.quantization != "none":
            raise ValueError("Static/compiled depth candidates require unquantized streaming")
        from models.static_depth import StaticDepthRunner
        model._static_depth_runner = StaticDepthRunner(model.depth_decoder.model, compile_decode=settings.depth_cache == "compiled")
        compile_started = time.perf_counter()
        model._static_depth_runner.warmup()
        app.state.depth_warmup_s = time.perf_counter() - compile_started
    if settings.quantization != "none":
        from breeze_infer.quantization import quantize_model
        app.state.quantization = quantize_model(model, bits=int(settings.quantization[-1]) if settings.quantization == "int8" else 4)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime_type = (
        FastBreezeStreamingRuntime
        if device.startswith("cuda")
        else (PortableBreezeStreamingRuntime if settings.engine == "streaming" else EagerBreezeStreamingRuntime)
    )
    runtime = runtime_type(model, audio_tokenizer, config, tokenizer=tokenizer)
    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.audio_tokenizer = audio_tokenizer
    app.state.runtime = runtime
    app.state.load_s = time.perf_counter() - started


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    _load_app(app, _settings)
    yield


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not hasattr(app.state, "runtime"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "sample_rate": app.state.runtime.sample_rate})


@app.post("/v1/audio/speech")
async def speech(
    http_request: Request,
    text: str = Form(...),
    instruction: str = Form("Speak clearly and naturally."),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
) -> StreamingResponse:
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An inference request is already running."
        )

    reference_path: Path | None = None
    cancelled = threading.Event()
    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if not cleaned:
            cleaned = True
            try:
                if reference_path is not None:
                    reference_path.unlink(missing_ok=True)
            finally:
                _request_lock.release()

    try:
        if getattr(app.state.runtime, "_poisoned", False):
            raise HTTPException(status_code=503, detail="Codec cleanup failed; restart Breeze.")
        expected = http_request.headers.get("X-Breeze-Runtime")
        fingerprint = _settings.runtime_fingerprint if _settings is not None else ""
        if expected and expected != fingerprint:
            raise HTTPException(status_code=409, detail="Breeze runtime changed; refresh health.")
        if not text.strip() or not instruction.strip() or len(text) > 4000 or len(instruction) > 2000:
            raise HTTPException(status_code=400, detail="Text/instruction must be non-empty and within input bounds.")
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)

        request_id = f"api-{uuid.uuid4().hex}"
        request = {
            "id": request_id,
            "text": text,
            "instruction": instruction,
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = ref_text
            template_name = "ref_edit_tata"

        prepared_at = time.perf_counter()
        inputs = prepare_inputs(
            app.state.tokenizer,
            app.state.audio_tokenizer,
            app.state.model,
            [request],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        preparation_s = time.perf_counter() - prepared_at
        if inputs["input_ids"].shape[1] > MAX_SEQ_LEN - 750:
            raise HTTPException(status_code=400, detail="Prompt exceeds supported sequence capacity.")
    except BaseException:
        cleanup()
        raise

    async def body() -> AsyncIterator[bytes]:
        runtime = app.state.runtime
        kwargs = {"request_id": request_id}
        if isinstance(runtime, PortableBreezeStreamingRuntime):
            kwargs["cancelled"] = cancelled
        iterator = runtime.iter_audio_chunks(inputs, **kwargs)
        pending = None

        def pull():
            try:
                return next(iterator)
            except StopIteration:
                return None

        try:
            set_all_seeds(seed)
            while True:
                pending = asyncio.create_task(asyncio.to_thread(pull))
                chunk = await asyncio.shield(pending)
                if chunk is None:
                    break
                pcm = _pcm16(chunk.audio)
                if pcm:
                    yield pcm
        finally:
            cancelled.set()
            # ASGI disconnect cancels the response task. Ownership is retained
            # until its outstanding pull and model/codec worker have stopped.
            with anyio.CancelScope(shield=True):
                try:
                    if pending is not None and not pending.done():
                        await asyncio.shield(pending)
                finally:
                    try:
                        await asyncio.to_thread(iterator.close)
                        if hasattr(runtime, "last_metrics"):
                            runtime.last_metrics["preparation_s"] = preparation_s
                    finally:
                        cleanup()

    return _OwnedStreamingResponse(
        body(),
        cleanup,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(app.state.runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "X-Breeze-Runtime": fingerprint,
            "X-Breeze-Request-ID": request_id,
            "Cache-Control": "no-store",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("mps", "cpu"))
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    global _settings
    _settings = ApiSettings(
        model=args.model,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        device=args.device,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
