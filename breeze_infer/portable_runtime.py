"""Incremental eager Breeze generation without CUDA graphs or full-audio buffering."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from breeze_infer.eager_runtime import EagerStreamingChunk


class GenerationCancelled(Exception):
    """Cooperative cancellation at a generated-frame boundary."""


class _FrameStreamer:
    def __init__(
        self,
        model: Any,
        codec: Any,
        request_id: str,
        emit: Any,
        stopped: threading.Event,
        metrics: dict[str, Any],
    ) -> None:
        self.model, self.codec, self.request_id = model, codec, request_id
        self.emit, self.stopped, self.metrics = emit, stopped, metrics
        self.prompt_pending = True
        self.ended = False
        self.saw_eos = False

    def put(self, value: torch.Tensor) -> None:
        if self.stopped.is_set():
            raise GenerationCancelled()
        if self.prompt_pending:
            self.prompt_pending = False
            return  # Transformers emits the input prompt before generated frames.
        expected = (1, self.model.config.num_codebooks)
        if tuple(value.shape) != expected or value.dtype != torch.long:
            raise ValueError(f"Expected one integer codec frame {expected}, got {value.shape}")
        pad = self.model.config.codebook_pad_token_id
        if pad is not None and bool(torch.all(value == pad)):
            self.ended = True
            self.saw_eos = True
            return
        if self.ended:
            raise ValueError("Received codec frames after EOS")
        if bool(torch.any(value < 0)) or bool(
            torch.any(value >= self.model.config.codec_config.codebook_size)
        ):
            raise ValueError("Generated frame contains invalid codec tokens")
        now = time.perf_counter()
        if self.metrics["first_codes_s"] is None:
            self.metrics["first_codes_s"] = now - self.metrics["started"]
        decoded = self.codec.decode_request_chunk(
            self.request_id, value.reshape(1, expected[1], 1), reset=False
        )
        audio = decoded.detach().float().cpu().numpy().reshape(-1)
        self.metrics["codec_s"] += time.perf_counter() - now
        self.metrics["codec_frames"] += 1
        if not np.isfinite(audio).all():
            raise ValueError("Codec returned non-finite audio")
        if audio.size:
            if self.metrics["first_pcm_s"] is None:
                self.metrics["first_pcm_s"] = time.perf_counter() - self.metrics["started"]
            self.metrics["audio_samples"] += int(audio.size)
            self.emit(EagerStreamingChunk(audio))

    def end(self) -> None:
        self.ended = True


class PortableBreezeStreamingRuntime:
    """One model/codec worker, bounded delivery, and request-scoped codec state."""

    fast_enabled = False

    def __init__(
        self,
        model: Any,
        audio_tokenizer: Any,
        config: Any,
        *,
        tokenizer: Any = None,
        codec: Any = None,
        queue_capacity: int = 4,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.model, self.audio_tokenizer, self.config = model, audio_tokenizer, config
        self.tokenizer = tokenizer
        self.sample_rate = int(model.config.codec_config.sampling_rate)
        self.queue_capacity = queue_capacity
        self.last_metrics: dict[str, Any] = {}
        self._active = threading.Lock()
        self._poisoned = False
        if codec is None:
            from models.stream_runtime.stream.runtime import (
                MultiRequestStreamRuntime,
                QwenStreamRuntimeConfig,
            )

            parameter = next(audio_tokenizer.model.parameters())
            codec = MultiRequestStreamRuntime(
                audio_tokenizer,
                QwenStreamRuntimeConfig(
                    chunk_frames=1,
                    num_lanes=1,
                    max_active_reqs=1,
                    fast=False,
                    lifecycle_assert_mode="raise",
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            )
        self.codec = codec

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
        cancelled: threading.Event | None = None,
    ) -> Iterator[EagerStreamingChunk]:
        if self._poisoned:
            raise RuntimeError("Codec cleanup failed; restart the runtime before reuse")
        if not self._active.acquire(blocking=False):
            raise RuntimeError("An inference request is already running")
        stopped = cancelled if cancelled is not None else threading.Event()
        done = threading.Event()
        chunks: queue.Queue = queue.Queue(self.queue_capacity)
        selected_id = request_id or f"portable-{uuid.uuid4().hex}"
        metrics = dict(
            request_id=selected_id,
            started=time.perf_counter(),
            first_codes_s=None,
            first_pcm_s=None,
            codec_s=0.0,
            codec_frames=0,
            audio_samples=0,
            queue_wait_s=0.0,
            completed=False,
            cancelled=False,
        )

        def emit(value: Any) -> None:
            started = time.perf_counter()
            while not stopped.is_set():
                try:
                    chunks.put(value, timeout=0.05)
                    metrics["queue_wait_s"] += time.perf_counter() - started
                    return
                except queue.Full:
                    continue
            raise GenerationCancelled()

        def produce() -> None:
            opened = False
            try:
                if stopped.is_set():
                    raise GenerationCancelled()
                with torch.inference_mode():
                    self.codec.open_request(selected_id, reset=True, is_first_decode=True)
                    opened = True
                    streamer = _FrameStreamer(
                        self.model, self.codec, selected_id, emit, stopped, metrics
                    )
                    self.model.generate(**inputs, output_audio=False, streamer=streamer)
                if metrics["audio_samples"] == 0:
                    raise RuntimeError("Breeze generated no audio frames")
                metrics["eos_reached"] = streamer.saw_eos
                if not streamer.saw_eos:
                    raise RuntimeError("Breeze stopped without EOS; output may be truncated")
                metrics["completed"] = not stopped.is_set()
            except GenerationCancelled:
                metrics["cancelled"] = True
            except Exception as error:
                if not opened:
                    # An interrupted/failed open may have partially allocated state.
                    self._poisoned = True
                try:
                    emit(error)
                except GenerationCancelled:
                    metrics["cancelled"] = True
            finally:
                try:
                    if opened:
                        self.codec.close_request(selected_id)
                except Exception as error:
                    metrics["completed"] = False
                    self._poisoned = True
                    try:
                        emit(error)
                    except GenerationCancelled:
                        metrics["cancelled"] = True
                finally:
                    metrics["generation_including_codec_s"] = time.perf_counter() - metrics.pop(
                        "started"
                    )
                    metrics["audio_s"] = metrics["audio_samples"] / self.sample_rate
                    self.last_metrics = metrics
                    done.set()

        worker = threading.Thread(target=produce, name="breeze-generation", daemon=True)
        try:
            worker.start()
            while not done.is_set() or not chunks.empty():
                if stopped.is_set():
                    break
                try:
                    item = chunks.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stopped.set()
            if worker.ident is not None:
                worker.join()  # Never release model ownership while it is still running.
            self._active.release()
