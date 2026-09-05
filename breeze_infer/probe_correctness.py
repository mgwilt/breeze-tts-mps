"""Production-weight teacher-forced depth and incremental codec diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

with redirect_stdout(sys.stderr):
    from breeze_infer.runtime import load_runtime
    from breeze_infer.templates import get_template, prepare_inputs
    from models.generation_breeze import _extract_decoded_audio_tensor
    from models.static_depth import StaticDepthRunner
    from models.stream_runtime.stream.runtime import (
        MultiRequestStreamRuntime,
        QwenStreamRuntimeConfig,
    )


def compare(reference, actual, valid_size):
    r, a = reference.float().cpu(), actual.float().cpu()
    error = a - r
    return dict(
        max_abs=float(error.abs().max()),
        rmse=float(error.square().mean().sqrt()),
        relative_l2=float(error.norm() / r.norm().clamp_min(1e-12)),
        finite=bool(torch.isfinite(a).all()),
        vectors=r.shape[0],
        valid_top1_equal=int((r[:, :valid_size].argmax(-1) == a[:, :valid_size].argmax(-1)).sum()),
    )


def guided(logits):
    return logits[1:] + 4 * (logits[:1] - logits[1:])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    with redirect_stdout(sys.stderr):
        tokenizer, model, audio_tokenizer = load_runtime(
            args.model_path, device="mps", attn_implementation="eager"
        )
    depth, books = model.depth_decoder, model.config.num_codebooks
    size = model.config.codec_config.codebook_size
    positions = torch.arange(books, device="mps")
    runners = {"static": StaticDepthRunner(depth.model, compile_decode=False)}
    if args.compiled:
        runners["compiled"] = StaticDepthRunner(depth.model, compile_decode=True)
        runners["compiled"].warmup()
    prompts = [
        ("Hello, I am ready to help.", "Speak calmly and clearly."),
        ("Wait! The train is leaving now.", "Speak urgently with excitement."),
        ("The rain sounds peaceful tonight.", "Speak softly and slowly."),
    ]
    records, frames = [], []
    for index, (text, instruction) in enumerate(prompts):
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [dict(id=f"probe-{index}", text=text, instruction=instruction, speaker="S0")],
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
        sequence = torch.cat([torch.zeros_like(first), first], dim=1)
        past = None
        for step in range(books - 1):
            reference = torch.cat(
                [
                    depth(
                        input_ids=sequence, backbone_last_hidden_state=row[None], use_cache=False
                    ).logits[:, -1, :]
                    for row in hidden
                ]
            ).float()
            ids = (sequence if step == 0 else sequence[:, -1:]).repeat(2, 1)
            dynamic = depth(
                input_ids=ids,
                backbone_last_hidden_state=hidden if step == 0 else None,
                past_key_values=past,
                cache_position=positions[:2] if step == 0 else positions[step + 1 : step + 2],
                use_cache=True,
                logits_to_keep=1,
                codebook_index=step,
            )
            past = dynamic.past_key_values
            candidates = {"dynamic": dynamic.logits[:, -1, :].float()}
            for name, runner in runners.items():
                state = runner.begin(ids, hidden) if step == 0 else runner.step(ids, step)
                candidates[name] = torch.nn.functional.linear(
                    state, depth.codebooks_head.weight[step].T
                )[:, -1, :].float()
            for name, logits in candidates.items():
                records.append(
                    dict(
                        prompt=index,
                        head=step,
                        candidate=name,
                        branch=compare(reference, logits, size),
                        cfg=compare(guided(reference), guided(logits), size),
                    )
                )
            sequence = torch.cat(
                [sequence, guided(reference)[:, :size].argmax(-1, keepdim=True)], dim=1
            )
        frames.append(sequence[0, 1:])
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
    fixed = torch.stack(frames).repeat(4, 1)
    codec_records, previous = [], None
    for run, length in enumerate((1, 3, 12, 12)):
        codes = fixed[:length]
        offline = (
            _extract_decoded_audio_tensor(audio_tokenizer.decode({"audio_codes": codes}))
            .float()
            .cpu()
            .reshape(-1)
        )
        request = f"correctness-{run}"
        codec.open_request(request, reset=True, is_first_decode=True)
        try:
            streamed = torch.cat(
                [
                    codec.decode_request_chunk(request, frame.reshape(1, books, 1), reset=False)
                    .float()
                    .cpu()
                    .reshape(-1)
                    for frame in codes
                ]
            )
        finally:
            codec.close_request(request)
        equal_length = streamed.shape == offline.shape
        codec_records.append(
            dict(
                frames=length,
                samples=streamed.numel(),
                equal_length=equal_length,
                finite=bool(torch.isfinite(streamed).all()),
                max_abs=float((streamed - offline).abs().max()) if equal_length else None,
                close=equal_length
                and bool(torch.allclose(streamed, offline, atol=1e-4, rtol=1e-3)),
                repeat_exact=bool(torch.equal(previous, streamed))
                if previous is not None and length == 12
                else None,
            )
        )
        previous = streamed if length == 12 else None
    report = dict(
        schema_version=1,
        torch=torch.__version__,
        device="mps",
        dtype=str(next(model.parameters()).dtype),
        codec_dtype=str(parameter.dtype),
        source_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        prompts=prompts,
        depth=records,
        codec=codec_records,
        proves="bounded numerical consistency; no perceptual or performance acceptance",
    )
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if all(r["branch"]["finite"] and r["cfg"]["finite"] for r in records)
        and all(r["close"] for r in codec_records)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
