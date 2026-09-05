"""Production-weight MLX depth numerics and whole-depth-frame microbenchmark.

This diagnostic does not synthesize utterances or establish perceptual/RTF gates.
It measures heads, guidance and sampling with explicit evaluation and separately
reports host bridges. Original weights, serving runtime and locks stay untouched.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import mlx.core as mx
import numpy as np
import torch

with redirect_stdout(sys.stderr):
    from breeze_infer.mlx_depth import MLXDepth, Sampling, from_torch, guided_logits
    from breeze_infer.probe_correctness import compare, guided
    from breeze_infer.profile_stages import source_identity
    from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs


def to_torch(value):
    return torch.from_numpy(np.array(value.astype(mx.float32)))


def timed(call, synchronize):
    synchronize()
    start = time.perf_counter()
    result = call()
    synchronize()
    return result, time.perf_counter() - start


def summary(values):
    return dict(
        samples_s=values,
        mean_s=float(np.mean(values)),
        p95_s=float(np.quantile(values, 0.95, method="higher")),
    )


def kernel_identity():
    distribution = importlib.metadata.distribution("mlx-metal")
    result = {}
    for path in distribution.files:
        if str(path).endswith(("mlx.metallib", "libmlx.dylib")):
            with distribution.locate_file(path).open("rb") as source:
                result[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument("--quant-bits", type=int, choices=(8, 4))
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("Require nonnegative warmups and positive repeats")
    identity = source_identity()
    kernels = kernel_identity()
    torch.manual_seed(42)
    with redirect_stdout(sys.stderr):
        load_start = time.perf_counter()
        tokenizer, model, codec = load_runtime(
            args.model_path, device="mps", attn_implementation="eager"
        )
        update_generation_config_for_breeze(model)
        torch.mps.synchronize()
        load_s = time.perf_counter() - load_start
    depth = model.depth_decoder
    cfg = depth.generation_config
    settings = Sampling(
        cfg=4,
        temperature=cfg.temperature or 1.0,
        top_k=cfg.top_k or 0,
        top_p=cfg.top_p if cfg.top_p is not None else 1.0,
        do_sample=cfg.do_sample,
    )
    size = model.config.codec_config.codebook_size
    conversion_start = time.perf_counter()
    candidate = MLXDepth(
        depth,
        valid_size=size,
        attention_kind=args.attention,
        quant_bits=args.quant_bits,
    )
    conversion_s = time.perf_counter() - conversion_start
    model._cached_depth_cfg = True
    candidates = {"mlx": candidate.generator(settings)}
    if not args.skip_compiled:
        candidates["mlx_compiled"] = candidate.generator(settings, compiled=True)
    greedy_settings = Sampling(cfg=4, do_sample=False)
    greedy = candidate.generator(greedy_settings)
    greedy_compiled = (
        candidate.generator(greedy_settings, compiled=True) if not args.skip_compiled else None
    )
    prompts = [
        ("Hello, I am ready to help.", "Speak calmly and clearly."),
        ("Wait! The train is leaving now.", "Speak urgently with excitement."),
        ("The rain sounds peaceful tonight.", "Speak softly and slowly."),
    ][: args.limit]

    def teacher_forced(tokens, hidden):
        cache, logits = (), []
        for step in range(candidate.books - 1):
            ids = tokens[:, :2] if step == 0 else tokens[:, step + 1 : step + 2]
            value, cache = candidate._forward(
                mx.concatenate([ids, ids]), hidden if step == 0 else None, cache
            )
            logits.append(value)
        return mx.stack(logits, axis=1)

    compiled_teacher = mx.compile(teacher_forced) if not args.skip_compiled else None
    numeric, benchmarks, greedy_records = [], [], []
    for index, (text, instruction) in enumerate(prompts):
        print(
            f"MLX probe {index + 1}/{len(prompts)}: prefill and teacher-forced heads",
            file=sys.stderr,
            flush=True,
        )
        with redirect_stdout(sys.stderr):
            inputs = prepare_inputs(
                tokenizer,
                codec,
                model,
                [
                    dict(
                        id=f"mlx-probe-{index}",
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
        positive = {k: v for k, v in inputs.items() if not k.startswith("cfg_")}
        negative = dict(
            input_ids=inputs["cfg_negative_prompt_ids"],
            attention_mask=inputs["cfg_negative_prompt_attention_mask"],
            text_ids_mask=inputs["cfg_negative_text_ids_mask"],
            text_ids_len=inputs["cfg_negative_text_ids_len"],
        )
        outputs = [
            model(
                **branch,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=1,
                return_dict=True,
            )
            for branch in (positive, negative)
        ]
        hidden = torch.cat([o.hidden_states[-1][:, -1, :] for o in outputs])
        first = guided(torch.cat([o.logits[:, -1, :].float() for o in outputs]))[:, :size].argmax(
            -1, keepdim=True
        )
        initial = torch.cat([torch.zeros_like(first), first], dim=1)
        sequence = initial

        def bridge_inputs():
            result = (
                from_torch(hidden),
                mx.array(initial.cpu().numpy().astype(np.int32)),
            )
            mx.eval(result)
            return result

        def sync_both():
            torch.mps.synchronize()
            mx.synchronize()

        (mlx_hidden, mlx_initial), bridge_s = timed(bridge_inputs, sync_both)
        cache, references = (), []
        for step in range(candidate.books - 1):
            reference = torch.cat(
                [
                    depth(
                        input_ids=sequence,
                        backbone_last_hidden_state=row[None],
                        use_cache=False,
                    ).logits[:, -1, :]
                    for row in hidden
                ]
            ).float()
            references.append(reference.cpu())
            ids = sequence if step == 0 else sequence[:, -1:]
            logits, cache = candidate.forward(
                mx.array(ids.repeat(2, 1).cpu().numpy().astype(np.int32)),
                mlx_hidden if step == 0 else None,
                cache,
            )
            mx.eval(logits, cache)
            reference_cfg = guided(reference)
            candidate_cfg = to_torch(guided_logits(logits, 4, size)[:, :size])
            margin = reference_cfg[:, :size].topk(2).values.diff(dim=-1).abs()
            numeric.append(
                dict(
                    prompt=index,
                    head=step,
                    candidate="mlx",
                    branch=compare(reference, to_torch(logits), size),
                    cfg=compare(reference_cfg[:, :size], candidate_cfg, size),
                    reference_top1_margin=float(margin.min()),
                )
            )
            sequence = torch.cat(
                [sequence, reference_cfg[:, :size].argmax(-1, keepdim=True)], dim=1
            )
        if compiled_teacher is not None:
            compiled_heads = compiled_teacher(
                mx.array(sequence[:, : candidate.books].cpu().numpy().astype(np.int32)),
                mlx_hidden,
            )
            mx.eval(compiled_heads)
            for step, reference in enumerate(references):
                logits = to_torch(compiled_heads[:, step])
                reference_cfg = guided(reference)[:, :size]
                numeric.append(
                    dict(
                        prompt=index,
                        head=step,
                        candidate="mlx_compiled_teacher",
                        branch=compare(reference, logits, size),
                        cfg=compare(reference_cfg, guided(logits)[:, :size], size),
                        reference_top1_margin=float(
                            reference_cfg.topk(2).values.diff(dim=-1).abs().min()
                        ),
                    )
                )
        for name, generator in (("mlx", greedy), ("mlx_compiled", greedy_compiled)):
            if generator is None:
                continue
            print(
                f"MLX probe {index + 1}: {name} full greedy frame",
                file=sys.stderr,
                flush=True,
            )

            def call_greedy():
                result = generator(mlx_initial, mlx_hidden, mx.random.key(42))
                mx.eval(result)
                return result[0]

            result, seconds = timed(call_greedy, mx.synchronize)
            actual = np.array(result)
            expected = sequence.cpu().numpy()
            greedy_records.append(
                dict(
                    prompt=index,
                    candidate=name,
                    first_call_s=seconds,
                    equal=bool(np.array_equal(actual, expected)),
                    equal_codebooks=int((actual[:, 1:] == expected[:, 1:]).sum()),
                    reference=expected.tolist(),
                    actual=actual.tolist(),
                )
            )
        for name in ("torch_cached_eager", *candidates):
            print(
                f"MLX probe {index + 1}: {name} sampled timing",
                file=sys.stderr,
                flush=True,
            )
            if name == "torch_cached_eager":

                def call(_key):
                    return model._depth_decoder_generate_with_cfg(
                        initial, hidden[:1], hidden[1:], 4
                    )

                sync = torch.mps.synchronize
                setup = torch.manual_seed
            else:
                generator = candidates[name]

                def call(key):
                    result = generator(mlx_initial, mlx_hidden, key)
                    mx.eval(result)
                    return result

                sync = mx.synchronize
                setup = mx.random.key

            def trial(seed):
                key = setup(seed)
                if name != "torch_cached_eager":
                    mx.eval(key)
                return timed(lambda: call(key), sync)

            last_result, cold = trial(42)
            for warmup in range(args.warmups):
                trial(100 + warmup)
            samples = []
            for repeat in range(args.repeats):
                last_result, seconds = trial(1000 + repeat)
                samples.append(seconds)
            output_bridge_s = None
            if name != "torch_cached_eager":
                _, output_bridge_s = timed(
                    lambda: torch.from_numpy(np.array(last_result[0])).to(
                        device="mps", dtype=initial.dtype
                    ),
                    sync_both,
                )
            benchmarks.append(
                dict(
                    prompt=index,
                    candidate=name,
                    first_call_s=cold,
                    input_bridge_ready_s=bridge_s if name != "torch_cached_eager" else None,
                    output_bridge_ready_s=output_bridge_s,
                    **summary(samples),
                )
            )
    report = dict(
        schema_version=2,
        scope="first frame from the recorded real prefills; no utterance, perceptual, LAN or release acceptance",
        source=identity,
        metal_artifacts=kernels,
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "mlx", "mlx-metal")
        },
        hardware=platform.platform(),
        metal_device=mx.device_info(),
        reference=dict(attention="eager", dtype=str(next(model.parameters()).dtype)),
        candidate=dict(
            attention=args.attention,
            dtype=str(candidate.dtype),
            rope="copied from locked Torch MPS reference",
            cache="functional, populated positions only",
            compilation="whole depth frame including heads/CFG/filter/sampling",
            quantization=candidate.quantization,
        ),
        settings=asdict(settings),
        load_s=load_s,
        conversion_s=conversion_s,
        warmups=args.warmups,
        repeats=args.repeats,
        prompts=prompts,
        numerics=numeric,
        greedy=greedy_records,
        timing=benchmarks,
        caveats=[
            "Different RNG algorithms: same seed is not a matched Torch/MLX sample",
            "First call includes graph compilation where enabled; later prompts reuse compiled shapes",
            "Fully evaluated input/output bridges measured separately; utterance scheduling not included",
            "Per-trial seed/key setup is outside timing; sampling itself is included",
            "Compiled teacher forcing is its own whole-frame graph, not the sampled generation graph",
            "Tiny precision tests do not establish production equivalence",
        ],
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(r["branch"]["finite"] and r["cfg"]["finite"] for r in numeric) else 1


if __name__ == "__main__":
    raise SystemExit(main())
