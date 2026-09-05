"""Synchronized MPS block diagnostics, deliberately not a release benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
from pathlib import Path


class StageRecorder:
    """Scope temporary wrappers and retain inclusive timings even on failure."""

    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.samples = defaultdict(list)
        self.originals = []

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)
        was_local = attribute in vars(owner)

        def measured(*args, **kwargs):
            name = label(args, kwargs) if callable(label) else label
            self.synchronize()
            start = self.clock()
            try:
                return original(*args, **kwargs)
            finally:
                self.synchronize()
                self.samples[name].append(self.clock() - start)

        self.originals.append((owner, attribute, original, was_local))
        setattr(owner, attribute, measured)

    def restore(self):
        for owner, attribute, original, was_local in reversed(self.originals):
            if was_local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.originals.clear()

    @contextmanager
    def installed(self):
        try:
            yield self
        finally:
            self.restore()

    def summary(self):
        result = {}
        for name, values in self.samples.items():
            ordered = sorted(values)
            result[name] = dict(
                calls=len(values),
                total_s=sum(values),
                mean_s=sum(values) / len(values),
                p95_s=ordered[max(0, (len(values) * 95 + 99) // 100 - 1)],
                samples_s=values,
            )
        return result


def backbone_stage(_args, kwargs):
    # Audio decode carries input_ids, not inputs_embeds. Cache occupancy also
    # distinguishes a one-token prefill without inspecting or copying tensors.
    cache = kwargs.get("past_key_values")
    return (
        "backbone_decode"
        if cache is not None and cache.get_seq_length() > 0
        else "backbone_prefill"
    )


def source_identity():
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (source / "models", source / "breeze_infer"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return dict(
        revision=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_digest=digest.hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--depth-cache", choices=("dynamic", "static", "compiled"), default="dynamic"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")
    identity = source_identity()
    with redirect_stdout(sys.stderr):
        import torch
        from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime
        from breeze_infer.runtime import (
            load_runtime,
            set_all_seeds,
            update_generation_config_for_breeze,
        )
        from breeze_infer.templates import get_template, prepare_inputs
        from models.static_depth import StaticDepthRunner

        load_start = time.perf_counter()
        tokenizer, model, codec = load_runtime(
            args.model_path, device="mps", attn_implementation=args.attention
        )
        update_generation_config_for_breeze(model)
        model._cached_depth_cfg = True
        if args.depth_cache != "dynamic":
            model._static_depth_runner = StaticDepthRunner(
                model.depth_decoder.model, compile_decode=args.depth_cache == "compiled"
            )
            model._static_depth_runner.warmup()
        runtime = PortableBreezeStreamingRuntime(model, codec, None, tokenizer=tokenizer)
        torch.mps.synchronize()
        load_s = time.perf_counter() - load_start
        prompts = [
            ("Hello, I am ready to help.", "Speak calmly and clearly."),
            ("Wait! The train is leaving now.", "Speak urgently with excitement."),
            ("The rain sounds peaceful tonight.", "Speak softly and slowly."),
        ]

        def run(index):
            text, instruction = prompts[index]
            set_all_seeds(42)
            inputs = prepare_inputs(
                tokenizer,
                codec,
                model,
                [
                    dict(
                        id=f"profile-{index}",
                        text=text,
                        instruction=instruction,
                        speaker="S0",
                    )
                ],
                get_template("tts_instruction"),
                guidance_scale=4,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            start = time.perf_counter()
            count = 0
            audio_hash = hashlib.sha256()
            for chunk in runtime.iter_audio_chunks(inputs):
                count += chunk.audio.size
                audio_hash.update(chunk.audio.tobytes())
            torch.mps.synchronize()
            if count != runtime.last_metrics["audio_samples"]:
                raise RuntimeError("Delivered audio differs from producer metrics")
            return dict(
                wall_s=time.perf_counter() - start,
                audio_float32_sha256=audio_hash.hexdigest(),
                **runtime.last_metrics,
            )

        for _ in range(args.warmups):
            run(0)
        records = []
        for index in range(args.limit):
            # Uninstrumented control distinguishes synchronization overhead from
            # actual end-to-end throughput. Both runs reset the same seed.
            control = run(index)
            recorder = StageRecorder(torch.mps.synchronize)
            with recorder.installed():
                recorder.wrap(model.backbone_model, "forward", backbone_stage)
                recorder.wrap(model.text_encoder, "forward", "text_encoder")
                recorder.wrap(
                    model,
                    "_depth_decoder_generate_with_cfg",
                    "depth_frame_including_sampling",
                )
                recorder.wrap(runtime.codec, "decode_request_chunk", "codec")
                profiled = run(index)
            records.append(
                dict(
                    prompt_index=index,
                    text=prompts[index][0],
                    instruction=prompts[index][1],
                    control=control,
                    synchronized=profiled,
                    stages=recorder.summary(),
                )
            )
            print(f"Profiled prompt {index + 1}/{args.limit}", file=sys.stderr, flush=True)
    report = dict(
        schema_version=1,
        **identity,
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        model_config_sha256=hashlib.sha256(
            (args.model_path / "config.json").read_bytes()
        ).hexdigest(),
        dependencies={
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "qwen-tts")
        },
        platform=platform.platform(),
        device="mps",
        dtype="bfloat16",
        codec_dtype=str(next(codec.model.parameters()).dtype),
        attention=args.attention,
        depth_cache=args.depth_cache,
        cfg=4,
        seed=42,
        sampling=dict(temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.0),
        warmups=args.warmups,
        load_and_compile_s=load_s,
        records=records,
        proof="Inclusive synchronized block diagnostics plus uninstrumented controls; not release p95, physical playback, or quality acceptance",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
