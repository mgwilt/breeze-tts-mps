"""Production Qwen3 continuation numerics and paired-backbone stage timings.

Keeps original Torch text preparation/prefill and transfers KV once. Repeats a
real first codec frame to exercise growing state; this is not generated speech,
an utterance quality test, or an end-to-end Fast benchmark.
"""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
import torch
from transformers.cache_utils import DynamicCache

with redirect_stdout(sys.stderr):
    from breeze_infer.mlx_backbone import MLXBackbone
    from breeze_infer.probe_correctness import compare, guided
    from breeze_infer.probe_mlx_depth import kernel_identity, summary, to_torch
    from breeze_infer.profile_stages import source_identity
    from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs


def checked_logits(logits, valid):
    # Include utterance EOS (last row), excluding the three reserved codec IDs.
    return torch.cat([logits[:, :valid], logits[:, -1:]], dim=-1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--quant-bits", type=int, choices=(8,))
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if not 2 <= args.steps <= 128 or args.warmups < 0 or args.repeats < 1:
        parser.error("Require2..128 steps, nonnegative warmups and positive repeats")
    identity, kernels = source_identity(), kernel_identity()
    with redirect_stdout(sys.stderr):
        start = time.perf_counter()
        tokenizer, model, codec = load_runtime(
            args.model_path, device="mps", attn_implementation="eager"
        )
        update_generation_config_for_breeze(model)
        torch.mps.synchronize()
        load_s = time.perf_counter() - start
    backbone, size = model.backbone_model, model.config.codec_config.codebook_size
    model._cached_depth_cfg = True
    model.depth_decoder.generation_config.do_sample = False
    start = time.perf_counter()
    candidate = MLXBackbone(
        backbone,
        head_weight=model.lm_head.weight,
        attention_kind=args.attention,
        quant_bits=args.quant_bits,
    )
    conversion_s = time.perf_counter() - start
    runners = {
        "mlx": candidate.step_runner(),
        "mlx_compiled": candidate.step_runner(compiled=True),
    }
    prompts = [
        ("Hello, I am ready to help.", "Speak calmly and clearly."),
        ("Wait! The train is leaving now.", "Speak urgently with excitement."),
        ("The rain sounds peaceful tonight.", "Speak softly and slowly."),
    ][: args.limit]
    numerics, timing, prefill_records, initial_evaluations = [], [], [], []
    for index, (text, instruction) in enumerate(prompts):
        print(
            f"Backbone probe {index + 1}/{len(prompts)}: Torch text/prefill and KV transfer",
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
                        id=f"backbone-{index}",
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
        branches = [positive, negative]
        masks = [branch["attention_mask"] for branch in branches]
        start = time.perf_counter()
        outputs = [
            model(
                **branch,
                position_ids=(mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1),
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
                return_dict=True,
            )
            for branch, mask in zip(branches, masks)
        ]
        torch.mps.synchronize()
        prefill_s = time.perf_counter() - start
        caches = [o.past_key_values for o in outputs]
        lengths = [cache.get_seq_length() for cache in caches]
        logical_lengths = [int(mask.sum()) for mask in masks]
        start = time.perf_counter()
        initial = candidate.pair_torch_caches(caches, masks)
        transfer_s = time.perf_counter() - start
        states = {name: initial for name in runners}
        hidden = torch.cat([o.hidden_states[-1][:, -1] for o in outputs])
        first = guided(torch.cat([o.logits[:, -1].float() for o in outputs]))[:, :size].argmax(
            -1, keepdim=True
        )
        sequence = model._depth_decoder_generate_with_cfg(
            torch.cat([torch.zeros_like(first), first], dim=1),
            hidden[:1],
            hidden[1:],
            4,
        )
        frame = sequence[:, 1:]
        mlx_frame = mx.array(frame.repeat(2, 1).cpu().numpy().astype(np.int32))
        mx.eval(mlx_frame)
        prefill_records.append(
            dict(
                prompt=index,
                physical_lengths=lengths,
                logical_lengths=logical_lengths,
                prefill_s=prefill_s,
                cache_transfer_ready_s=transfer_s,
                repeated_codec_frame=frame.cpu().tolist(),
                initial_capacity=initial[1].shape[-1],
            )
        )
        for step in range(args.steps):
            snapshots = [
                tuple((layer.keys, layer.values) for layer in cache.layers) for cache in caches
            ]
            current_masks = [
                torch.cat(
                    [
                        mask,
                        torch.ones((1, step + 1), dtype=mask.dtype, device=mask.device),
                    ],
                    dim=-1,
                )
                for mask in masks
            ]
            positions = [
                torch.tensor([[length + step]], device="mps") for length in logical_lengths
            ]
            physical = [torch.tensor([length + step], device="mps") for length in lengths]

            def torch_step(branch_caches):
                rows = [
                    backbone(
                        input_ids=frame[:, None],
                        attention_mask=current_masks[row],
                        position_ids=positions[row],
                        cache_position=physical[row],
                        past_key_values=cache,
                        use_cache=True,
                    ).last_hidden_state
                    for row, cache in enumerate(branch_caches)
                ]
                state = torch.cat(rows)[:, -1]
                logits = model.lm_head(state).float()
                return state, logits, guided(logits)

            reference_hidden, reference_logits, reference_cfg = torch_step(caches)
            for name, runner in runners.items():
                before = states[name]

                def mlx_step():
                    hidden, state = runner(candidate.audio_embeddings(mlx_frame), before)
                    logits = candidate.logits(hidden).astype(mx.float32)
                    conditional, unconditional = mx.split(logits, 2, axis=0)
                    guided_scores = unconditional + 4 * (conditional - unconditional)
                    mx.eval(hidden, state, logits, guided_scores)
                    return hidden, state, logits, guided_scores

                mx.synchronize()
                start = time.perf_counter()
                result = mlx_step()
                elapsed = time.perf_counter() - start
                initial_evaluations.append(
                    dict(
                        prompt=index,
                        step=step,
                        candidate=name,
                        seconds=elapsed,
                        capacity=before[1].shape[-1],
                        next_capacity=result[1][1].shape[-1],
                    )
                )
                states[name] = result[1]
                numeric_logits = to_torch(result[2])
                numerics.append(
                    dict(
                        prompt=index,
                        step=step,
                        candidate=name,
                        hidden=compare(
                            reference_hidden,
                            to_torch(result[0][:, -1]),
                            candidate.hidden,
                        ),
                        branch=compare(
                            checked_logits(reference_logits, size),
                            checked_logits(numeric_logits, size),
                            size + 1,
                        ),
                        cfg=compare(
                            checked_logits(reference_cfg, size),
                            checked_logits(to_torch(result[3]), size),
                            size + 1,
                        ),
                    )
                )
                if step in (0, args.steps - 1):
                    for _ in range(args.warmups):
                        mlx_step()
                    samples = []
                    for _ in range(args.repeats):
                        mx.synchronize()
                        start = time.perf_counter()
                        mlx_step()
                        samples.append(time.perf_counter() - start)
                    timing.append(dict(prompt=index, step=step, candidate=name, **summary(samples)))
            if step in (0, args.steps - 1):

                def fresh_caches():
                    # DynamicCache(ddp_cache_data=...) copies through update/cat
                    # in Transformers 4.57.3. Reset outside the timed interval.
                    return [
                        DynamicCache(ddp_cache_data=snapshot, config=backbone.config)
                        for snapshot in snapshots
                    ]

                for _ in range(args.warmups):
                    torch_step(fresh_caches())
                samples = []
                for _ in range(args.repeats):
                    fresh = fresh_caches()
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    torch_step(fresh)
                    torch.mps.synchronize()
                    samples.append(time.perf_counter() - start)
                timing.append(
                    dict(
                        prompt=index,
                        step=step,
                        candidate="torch_separate_eager",
                        **summary(samples),
                    )
                )
        print(
            f"Backbone probe {index + 1}: {args.steps} continuation steps complete",
            file=sys.stderr,
            flush=True,
        )
    report = dict(
        schema_version=1,
        scope="paired backbone audio embedding, layers, output head and F32 CFG; no sampling/depth/codec or generated utterance",
        source=identity,
        metal_artifacts=kernels,
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "mlx", "mlx-metal")
        },
        metal_device=mx.device_info(),
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        reference_attention="eager",
        candidate_attention=args.attention,
        dtype=str(candidate.dtype),
        quantization=candidate.quantization,
        load_s=load_s,
        conversion_s=conversion_s,
        steps=args.steps,
        warmups=args.warmups,
        repeats=args.repeats,
        prefill=prefill_records,
        numerics=numerics,
        initial_evaluations=initial_evaluations,
        timing=timing,
        limitations=[
            "Fixed repeated real first-frame codes exercise cache state, not natural speech",
            "Only Torch text/prefill is implemented; candidate does not handle initial EOS or complete generation",
            "Torch cache reconstruction and its GPU copies complete before each timed control; normal continuation cache updates remain timed",
            "No numerical, quality, LAN, resident or Fast acceptance follows from finite results",
        ],
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(r["branch"]["finite"] and r["cfg"]["finite"] for r in numerics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
