"""Retain real MLX speech/PCM arrival evidence; not LAN or listening acceptance."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time
import wave

import mlx.core as mx
import numpy as np
import torch

with redirect_stdout(sys.stderr):
    from breeze_infer.mlx_speech import MLXSpeechModel
    from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime
    from breeze_infer.probe_mlx_depth import kernel_identity
    from breeze_infer.profile_stages import source_identity
    from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs


def save_audio(path, audio, rate):
    pcm = (np.clip(audio, -1, 1) * 32767).round().astype("<i2").tobytes()
    # Evidence destinations must be fresh; never overwrite existing audio.
    with path.open("xb") as target, wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)
    return dict(
        path=str(path.resolve()),
        pcm_sha256=hashlib.sha256(pcm).hexdigest(),
        samples=audio.size,
        sample_rate=rate,
    )


def load_corpus(path):
    content = path.read_bytes()
    rows = json.loads(content)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("Corpus must contain1..12 prompt/instruction pairs")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"text", "instruction"}
            or any(
                not isinstance(row[name], str) or not row[name].strip() or len(row[name]) > 2000
                for name in row
            )
        ):
            raise ValueError("Invalid bounded corpus entry")
    return [(row["text"], row["instruction"]) for row in rows], dict(
        path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest()
    )


def lifecycle_state(runtime):
    return dict(
        active_requests=list(runtime.codec.request_pool.active_req_ids()),
        inference_locked=runtime._active.locked(),
        poisoned=runtime._poisoned,
        generation_workers=[
            worker.ident for worker in threading.enumerate() if worker.name == "breeze-generation"
        ],
    )


def interrupted_request(runtime, inputs, mode):
    """Exercise real runtime cleanup; injection affects only this probe request."""
    if mode not in {"close", "event", "consumer-error", "codec-error"}:
        raise ValueError("Unknown lifecycle probe mode")
    chunks, failure, busy_error = [], None, None
    cancelled = threading.Event()
    original_decode = runtime.codec.decode_request_chunk
    calls = 0
    interrupted = None

    def fail_second_decode(*args, **kwargs):
        nonlocal calls, interrupted
        calls += 1
        if calls == 2:
            interrupted = time.perf_counter()
            raise RuntimeError("Injected codec decode failure")
        return original_decode(*args, **kwargs)

    if mode == "codec-error":
        runtime.codec.decode_request_chunk = fail_second_decode
    iterator = runtime.iter_audio_chunks(inputs, cancelled=cancelled)
    started = time.perf_counter()
    try:
        for chunk in iterator:
            chunks.append(chunk.audio.copy())
            if len(chunks) == 1:
                # The first generator retains its inference lock across yield.
                concurrent = runtime.iter_audio_chunks(inputs)
                try:
                    next(concurrent)
                except RuntimeError as error:
                    busy_error = str(error)
                finally:
                    concurrent.close()
                if mode != "codec-error":
                    interrupted = time.perf_counter()
                    if mode == "event":
                        cancelled.set()
                    elif mode == "consumer-error":
                        raise RuntimeError("Injected consumer failure")
                    else:
                        break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        if interrupted is None:
            interrupted = time.perf_counter()
    finally:
        if interrupted is None:
            interrupted = time.perf_counter()
        iterator.close()  # Join worker before restoring codec or measuring idle.
        cleanup_s = time.perf_counter() - interrupted
        runtime.codec.decode_request_chunk = original_decode
    state = lifecycle_state(runtime)
    expected_failure = {
        "close": None,
        "event": None,
        "consumer-error": "RuntimeError: Injected consumer failure",
        "codec-error": "RuntimeError: Injected codec decode failure",
    }[mode]
    metrics = dict(runtime.last_metrics)
    passed = (
        bool(chunks)
        and failure == expected_failure
        and busy_error == "An inference request is already running"
        and not metrics.get("completed")
        and (mode == "codec-error" or metrics.get("cancelled") is True)
        and not any(state.values())
    )
    return dict(
        mode=mode,
        passed=bool(passed),
        failure=failure,
        overlap_rejection=busy_error,
        cleanup_s=cleanup_s,
        total_s=time.perf_counter() - started,
        producer=metrics,
        after_cleanup=state,
    ), np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--quant-bits", type=int, choices=(8,))
    parser.add_argument("--backbone-quant-bits", choices=("none", "8"))
    parser.add_argument("--depth-quant-bits", choices=("none", "8"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--uncompiled", action="store_true")
    parser.add_argument("--lifecycle-checks", action="store_true")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Bounded JSON prompt/instruction pairs; overrides the3-prompt screen",
    )
    args = parser.parse_args(argv)
    if args.warmups < 0 or any(not 0 <= seed <= 2**32 - 1 for seed in args.seeds):
        parser.error("Require nonnegative warmups and uint32 seeds")
    if args.audio_dir.exists():
        parser.error("Use a new audio evidence directory")
    return args


def quantization_options(args):
    """Omitted selectors inherit legacy; explicit none disables weight quantization."""
    options = {"quant_bits": args.quant_bits}
    for name in ("backbone_quant_bits", "depth_quant_bits"):
        value = getattr(args, name)
        if value is not None:
            options[name] = None if value == "none" else int(value)
    return options


def main():
    args = parse_args()
    if args.corpus:
        prompts, corpus_identity = load_corpus(args.corpus)
    else:
        corpus_identity = None
        prompts = [
            ("Hello, I am ready to help.", "Speak calmly and clearly."),
            ("Wait! The train is leaving now.", "Speak urgently with excitement."),
            ("The rain sounds peaceful tonight.", "Speak softly and slowly."),
        ][: args.limit]
    args.audio_dir.mkdir(parents=True)
    started_unix_ns = time.time_ns()
    identity, kernels = source_identity(), kernel_identity()
    with redirect_stdout(sys.stderr):
        start = time.perf_counter()
        tokenizer, model, audio_tokenizer = load_runtime(
            args.model_path, device="mps", attn_implementation="eager"
        )
        update_generation_config_for_breeze(model)
        torch.mps.synchronize()
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        candidate = MLXSpeechModel(
            model, cfg=4, compiled=not args.uncompiled, **quantization_options(args)
        )
        conversion_s = time.perf_counter() - start
        runtime = PortableBreezeStreamingRuntime(
            candidate, audio_tokenizer, None, tokenizer=tokenizer
        )

    def run(index, seed, label):
        text, instruction = prompts[index]
        print(f"Speech {label}: prompt{index}, seed{seed}", file=sys.stderr, flush=True)
        chunks, arrivals, failure = [], [], None
        request_started_unix_ns = time.time_ns()
        start = time.perf_counter()
        with redirect_stdout(sys.stderr):
            inputs = prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [dict(id=label, text=text, instruction=instruction, speaker="S0")],
                get_template("tts_instruction"),
                guidance_scale=4,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            prepared_s = time.perf_counter() - start
            try:
                for chunk in runtime.iter_audio_chunks({**inputs, "mlx_seed": seed}):
                    arrivals.append(
                        dict(
                            seconds=time.perf_counter() - start,
                            samples=int(chunk.audio.size),
                        )
                    )
                    chunks.append(chunk.audio.copy())
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
        total_s = time.perf_counter() - start
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        duration = audio.size / runtime.sample_rate
        first = arrivals[0]["seconds"] if arrivals else None
        first_audio = arrivals[0]["samples"] / runtime.sample_rate if arrivals else 0
        steady = (
            (arrivals[-1]["seconds"] - first) / (duration - first_audio)
            if len(arrivals) > 1
            else None
        )
        artifact = save_audio(args.audio_dir / f"{label}.wav", audio, runtime.sample_rate)
        record = dict(
            prompt=text,
            instruction=instruction,
            seed=seed,
            started_unix_ns=request_started_unix_ns,
            ended_unix_ns=time.time_ns(),
            total_s=total_s,
            preparation_s=prepared_s,
            first_pcm_s=first,
            audio_s=duration,
            total_rtf=total_s / duration if duration else None,
            steady_rtf=steady,
            arrivals=arrivals,
            failure=failure,
            stages=dict(candidate.last_stages),
            producer=dict(runtime.last_metrics),
        )
        print(
            f"Speech {label}: {duration:.2f}s audio, totalRTF={record['total_rtf']}, steadyRTF={steady}, failure={failure}",
            file=sys.stderr,
            flush=True,
        )
        return record, artifact

    warmups = [run(0, args.seeds[0], f"warmup-{i}") for i in range(args.warmups)]
    samples, artifacts = [], []
    for index in range(len(prompts)):
        for repetition, seed in enumerate(args.seeds):
            result, artifact = run(index, seed, f"sample-{index}-{repetition}")
            samples.append(result)
            artifacts.append(artifact)
    lifecycle = []
    if args.lifecycle_checks:
        text, instruction = prompts[0]
        with redirect_stdout(sys.stderr):
            inputs = prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [dict(id="lifecycle", text=text, instruction=instruction, speaker="S0")],
                get_template("tts_instruction"),
                guidance_scale=4,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
        for mode in ("close", "event", "consumer-error", "codec-error"):
            with redirect_stdout(sys.stderr):
                trial, audio = interrupted_request(
                    runtime, {**inputs, "mlx_seed": args.seeds[0]}, mode
                )
            trial["partial_artifact"] = save_audio(
                args.audio_dir / f"lifecycle-{mode}-partial.wav",
                audio,
                runtime.sample_rate,
            )
            retry, artifact = run(0, args.seeds[0], f"lifecycle-{mode}-retry")
            trial.update(
                retry=retry,
                retry_artifact=artifact,
                retry_state=lifecycle_state(runtime),
                retry_pcm_matches_baseline=artifact["pcm_sha256"] == artifacts[0]["pcm_sha256"],
            )
            trial["passed"] = bool(
                trial["passed"]
                and not retry["failure"]
                and retry["producer"].get("completed")
                and trial["retry_pcm_matches_baseline"]
                and not any(trial["retry_state"].values())
            )
            lifecycle.append(trial)
    report = dict(
        schema_version=1,
        source=identity,
        started_unix_ns=started_unix_ns,
        ended_unix_ns=time.time_ns(),
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "qwen-tts", "mlx", "mlx-metal")
        },
        metal_artifacts=kernels,
        metal_device=mx.device_info(),
        load_s=load_s,
        conversion_s=conversion_s,
        compiled=not args.uncompiled,
        corpus=corpus_identity,
        effective_settings=dict(
            backbone=asdict(candidate.backbone_settings),
            depth=asdict(candidate.depth_settings),
            max_new_tokens=candidate.limit,
            backbone_dtype=str(candidate.backbone.dtype),
            depth_dtype=str(candidate.depth.dtype),
            codec_dtype=str(next(audio_tokenizer.model.parameters()).dtype),
            backbone_quantization=candidate.backbone.quantization,
            depth_quantization=candidate.depth.quantization,
        ),
        warmups=[dict(result=result, artifact=artifact) for result, artifact in warmups],
        samples=samples,
        audio_artifacts=artifacts,
        lifecycle=lifecycle,
        quality_acceptance=False,
        limits=[
            "Isolated local PCM producer; no HTTP/LAN or browser playback timing",
            "Corpus and seeds identify the measured producer subset; no full-system release follows from this report",
            "Same-seed MLX/Torch sampling streams differ; reference equivalence remains unaccepted",
            "Warmups retain cold compilation; each timed request is uncached and includes preparation/prefill",
            "Failed/partial WAVs are retained evidence, never completed-preview cache entries",
            "Lifecycle checks cover the isolated real model/codec runtime, not HTTP Stop/disconnect",
        ],
    )
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if all(
            not row["failure"] and row["producer"].get("completed")
            for row in samples + [result for result, _ in warmups]
        )
        and all(trial["passed"] for trial in lifecycle)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
