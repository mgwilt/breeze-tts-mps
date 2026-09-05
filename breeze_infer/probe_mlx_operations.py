"""Localize BF16 backend drift by evaluating MLX on exact Torch module inputs.

Diagnostic hooks do not replace or modify Torch outputs. Measurements are not
timings; every comparison synchronizes and copies data. Original weights remain
untouched and this module is never imported by serving.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import torch

with redirect_stdout(sys.stderr):
    from breeze_infer.mlx_depth import MLXDepth, from_torch, rms_norm, rotary
    from models.breeze import apply_rotary_pos_emb, repeat_kv
    from breeze_infer.probe_correctness import guided
    from breeze_infer.profile_stages import source_identity
    from breeze_infer.runtime import load_runtime
    from breeze_infer.templates import get_template, prepare_inputs


def error(reference, actual):
    ref = reference.detach().float().cpu().numpy()
    other = (
        np.array(actual.astype(mx.float32))
        if isinstance(actual, mx.array)
        else actual.detach().float().cpu().numpy()
    )
    delta = other - ref
    return dict(
        shape=list(ref.shape),
        max_abs=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-12)),
        mismatch_fraction=float(np.mean(other != ref)),
        finite=bool(np.isfinite(other).all()),
    )


@contextmanager
def compare_operations(depth, candidate, context, records):
    handles = []

    def hook(name, kind):
        def compare(module, args, output):
            source = args[0]
            value = from_torch(source)
            if kind == "linear":
                result = candidate._linear(value, name)
            elif kind == "rms_norm":
                result = rms_norm(
                    value, candidate.weights[name + ".weight"], module.variance_epsilon
                )
            else:
                full = value.astype(mx.float32)
                result = (full * mx.sigmoid(full)).astype(value.dtype)
            mx.eval(result)
            record = dict(**context, name=name, operation=kind, mlx=error(output, result))
            if kind == "silu":
                # Distinguish framework/device activation arithmetic from an
                # accumulated transformer difference. Same input, no weight copy.
                cpu = torch.nn.functional.silu(source.float().cpu()).to(source.dtype)
                record["cpu_float32_then_cast"] = error(output, cpu)
                native = value * mx.sigmoid(value)
                mx.eval(native)
                record["mlx_bf16_intermediates"] = error(output, native)
            records.append(record)

        return compare

    def attention_hook(name):
        def compare(module, _args, kwargs, _output):
            source = kwargs["hidden_states"]
            shape = (*source.shape[:-1], -1, module.head_dim)
            q, k, v = [
                torch.nn.functional.linear(source, getattr(module, p + "_proj").weight)
                .view(shape)
                .transpose(1, 2)
                for p in ("q", "k", "v")
            ]
            cosine, sine = kwargs["position_embeddings"]
            qr, kr = apply_rotary_pos_emb(q, k, cosine, sine)
            cosine_mx, sine_mx = from_torch(cosine[:, None]), from_torch(sine[:, None])

            def record(operation, reference, actual):
                mx.eval(actual)
                records.append(
                    dict(
                        **context,
                        name=name,
                        operation=operation,
                        mlx=error(reference, actual),
                    )
                )

            record("rotary_q", qr, rotary(from_torch(q), cosine_mx, sine_mx))
            record("rotary_k", kr, rotary(from_torch(k), cosine_mx, sine_mx))
            keys, values = (
                repeat_kv(kr, module.num_key_value_groups),
                repeat_kv(v, module.num_key_value_groups),
            )
            scores = qr @ keys.transpose(-1, -2)
            record(
                "attention_dot",
                scores,
                from_torch(qr) @ mx.swapaxes(from_torch(keys), -1, -2),
            )
            scaled = scores * module.scaling
            record("attention_scale", scaled, from_torch(scores) * module.scaling)
            record(
                "attention_scale_float32",
                scaled,
                (from_torch(scores).astype(mx.float32) * module.scaling).astype(mx.bfloat16),
            )
            mask = kwargs.get("attention_mask")
            masked = scaled + mask if mask is not None else scaled
            probabilities = torch.softmax(masked, dim=-1, dtype=torch.float32).to(source.dtype)
            record(
                "attention_softmax",
                probabilities,
                mx.softmax(from_torch(masked).astype(mx.float32), axis=-1).astype(mx.bfloat16),
            )
            context_output = probabilities @ values
            record(
                "attention_value",
                context_output,
                from_torch(probabilities) @ from_torch(values),
            )

        return compare

    try:
        for name, module in depth.named_modules():
            kind = None
            if isinstance(module, torch.nn.Linear):
                kind = "linear"
            elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                kind = "rms_norm"
            elif name.endswith(".act_fn"):
                kind = "silu"
            if kind:
                handles.append(module.register_forward_hook(hook(name, kind)))
            elif name.endswith(".self_attn"):
                handles.append(module.register_forward_hook(attention_hook(name), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    identity = source_identity()
    with redirect_stdout(sys.stderr):
        tokenizer, model, codec = load_runtime(
            args.model_path, device="mps", attn_implementation="eager"
        )
    depth = model.depth_decoder
    candidate = MLXDepth(depth, valid_size=model.config.codec_config.codebook_size)
    prompts = [
        ("Wait! The train is leaving now.", "Speak urgently with excitement.", {0}),
        ("The rain sounds peaceful tonight.", "Speak softly and slowly.", {9, 10}),
    ]
    records, checked = [], []
    for index, (text, instruction, heads) in enumerate(prompts):
        with redirect_stdout(sys.stderr):
            inputs = prepare_inputs(
                tokenizer,
                codec,
                model,
                [
                    dict(
                        id=f"operation-{index}",
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
        first = guided(torch.cat([o.logits[:, -1, :].float() for o in outputs]))[
            :, : candidate.valid_size
        ].argmax(-1, keepdim=True)
        sequence = torch.cat([torch.zeros_like(first), first], dim=1)
        for head in range(max(heads) + 1):
            reference = torch.cat(
                [
                    depth(
                        input_ids=sequence,
                        backbone_last_hidden_state=row[None],
                        use_cache=False,
                    ).logits[:, -1]
                    for row in hidden
                ]
            )
            if head in heads:
                print(
                    f"Same-input operation probe prompt={index}, head={head}",
                    file=sys.stderr,
                    flush=True,
                )
                for branch, row in enumerate(hidden):
                    with compare_operations(
                        depth,
                        candidate,
                        dict(prompt=index, head=head, branch=branch),
                        records,
                    ):
                        observed = depth(
                            input_ids=sequence,
                            backbone_last_hidden_state=row[None],
                            use_cache=False,
                        ).logits[:, -1]
                    checked.append(bool(torch.equal(observed, reference[branch : branch + 1])))
            sequence = torch.cat(
                [
                    sequence,
                    guided(reference.float())[:, : candidate.valid_size].argmax(-1, keepdim=True),
                ],
                dim=1,
            )
    print(
        json.dumps(
            dict(
                schema_version=1,
                source=identity,
                prompts=[dict(text=p[0], instruction=p[1], heads=sorted(p[2])) for p in prompts],
                torch=torch.__version__,
                model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
                dtype=str(next(depth.parameters()).dtype),
                observed_reference_unchanged=checked,
                operations=records,
                scope="same-input linears, norms, SiLU, rotary and attention stages; not accumulated-error attribution or performance",
                timing_claim=False,
            ),
            sort_keys=True,
        )
    )
    return 0 if all(checked) and all(r["mlx"]["finite"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
