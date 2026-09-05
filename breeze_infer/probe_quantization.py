"""Correctness-first native MPS quantization probe; never modifies model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from breeze_infer.quantization import MPSWeightOnlyLinear, quantize_weight


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    rows = (a - b).square().mean(-1).sqrt() / b.square().mean(-1).sqrt().clamp_min(1e-8)
    return dict(
        relative_rms=float(rows.max()),
        max_abs=float((a - b).abs().max()),
        finite=bool(torch.isfinite(a).all()),
    )


def timings(call, repeats):
    values = []
    for _ in range(5):
        call()
    torch.mps.synchronize()
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        torch.mps.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return dict(
        median_ms=statistics.median(values),
        p95_ms=sorted(values)[math.ceil(len(values) * 0.95) - 1],
        samples_ms=values,
    )


def paired_blocks(calls, repeats, block_size=100):
    values = {name: [] for name in calls}
    for block in range(repeats):
        for name in list(calls) if block % 2 == 0 else list(reversed(calls)):
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(block_size):
                calls[name]()
            torch.mps.synchronize()
            values[name].append((time.perf_counter() - start) * 1000 / block_size)
    return {
        name: dict(median_ms=statistics.median(samples), samples_ms=samples, block_size=block_size)
        for name, samples in values.items()
    }


def probe(weight, bits, dtype, repeats):
    torch.manual_seed(42)
    weight = weight.to(device="mps", dtype=dtype)
    adapter = MPSWeightOnlyLinear(weight, bits=bits)
    q, parameters = quantize_weight(weight, bits)
    # Compare with the same stored, rounded quantization parameters, separately
    # from the original-weight distortion introduced by quantization itself.
    if bits == 8:
        reference = q.float().to("mps") * adapter.scales.float()[:, None]
    else:
        scales, offsets = adapter.scales.transpose(0, 1).unbind(-1)
        zero = offsets - scales * 8
        reference = (
            q.reshape(weight.shape[0], -1, 64).to("mps").float() * scales.float()[..., None]
            + zero.float()[..., None]
        ).reshape_as(weight)
    n, k = weight.shape
    checks = []
    for shape in [(k,), (2, k), (1, 16, k), (2, 1, k), (2, 4, k)]:
        for sliced in (False, True):
            x = torch.randn(*shape[:-1], k * (2 if sliced else 1), device="mps", dtype=dtype)
            if sliced:
                x = x[..., ::2]
            actual = adapter(x)
            check = errors(actual, F.linear(x.float(), reference).to(dtype))
            check.update(
                shape=list(shape),
                noncontiguous=sliced,
                distortion=errors(actual, F.linear(x, weight)),
            )
            if x.ndim > 1:
                separate = torch.cat(
                    [adapter(row[None]) for row in x.reshape(-1, k)], dim=0
                ).reshape_as(actual)
                check["row_isolation"] = errors(actual, separate)
            checks.append(check)
    for name, x in (
        ("zero", torch.zeros(2, k, device="mps", dtype=dtype)),
        ("duplicate", torch.randn(1, k, device="mps", dtype=dtype).expand(2, -1)),
    ):
        check = errors(adapter(x), F.linear(x.float(), reference).to(dtype))
        check["case"] = name
        checks.append(check)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    passed = all(
        item["finite"]
        and item["relative_rms"] <= tolerance
        and item.get("row_isolation", {}).get("relative_rms", 0) <= tolerance
        for item in checks
    )
    timing = []
    if passed:
        for rows in (1, 2, 64, 256):
            x = torch.randn(rows, k, device="mps", dtype=dtype)
            # Alternating which recipe runs first limits order bias; raw samples
            # are retained and microbenchmarks never establish audio acceptance.
            calls = {"original": lambda: F.linear(x, weight), "quantized": lambda: adapter(x)}
            order = ("original", "quantized") if rows in (1, 64) else ("quantized", "original")
            timing.append(
                dict(
                    rows=rows,
                    isolated={name: timings(calls[name], repeats) for name in order},
                    amortized=paired_blocks(calls, repeats),
                )
            )
    return dict(
        bits=bits,
        dtype=str(dtype),
        shape=[n, k],
        checks=checks,
        correctness_pass=passed,
        timings=timing,
        original_bytes=weight.numel() * weight.element_size(),
        group_size=64 if bits == 4 else None,
        packed_bytes=sum(b.numel() * b.element_size() for b in adapter.buffers()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--bits", type=int, nargs="+", choices=(8, 4), default=[8, 4])
    parser.add_argument(
        "--dtype", nargs="+", choices=("bfloat16", "float16"), default=["bfloat16", "float16"]
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or args.repeats < 1:
        raise RuntimeError("Probe requires MPS and positive repeats")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    index = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = ("backbone_model", "depth_decoder.model")
    suffixes = ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj")
    results = []
    for prefix in prefixes:
        for suffix in suffixes[:1] if args.quick else suffixes:
            key = f"{prefix}.layers.0.{suffix}.weight"
            with safe_open(args.model_path / index[key], framework="pt", device="cpu") as source:
                weight = source.get_tensor(key)
            weight_digest = hashlib.sha256(
                weight.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for dtype in args.dtype:
                for bits in args.bits:
                    try:
                        with torch.inference_mode():
                            result = probe(weight, bits, getattr(torch, dtype), args.repeats)
                        result["key"] = key
                    except Exception as error:
                        result = dict(
                            key=key,
                            bits=bits,
                            dtype=dtype,
                            correctness_pass=False,
                            error=str(error),
                        )
                    results.append(result)
                    result["weight_digest"] = weight_digest
    payload = dict(
        schema_version=2,
        torch_version=torch.__version__,
        os=platform.platform(),
        backend="pytorch-native-mps",
        seed=42,
        threads=torch.get_num_threads(),
        settings={
            "bits": args.bits,
            "dtype": args.dtype,
            "quick": args.quick,
            "repeats": args.repeats,
        },
        source_digest=hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("quantization.py").read_bytes()
        ).hexdigest(),
        model_marker=json.loads((args.model_path / ".simo-model.json").read_text()),
        model_index_digest=hashlib.sha256(
            (args.model_path / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        results=results,
        proves="matrix correctness and isolated latency only",
        audio_acceptance=False,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if all(row["correctness_pass"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
