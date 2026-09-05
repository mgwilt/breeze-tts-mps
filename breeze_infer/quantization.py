"""Experimental weight-only MPS linear adapters pinned to native PyTorch ops.

Original checkpoints remain untouched; packed weights are rebuilt at startup.
Only explicitly selected backbone/depth layer linears are eligible.
"""

from __future__ import annotations

import torch
from torch import nn


def quantize_weight(weight: torch.Tensor, bits: int, group_size: int = 64):
    if bits not in (4, 8) or group_size != 64:
        raise ValueError("Only 8-bit or 4-bit with group size 64 is supported")
    if weight.ndim != 2:
        raise ValueError("Expected a matrix")
    n, k = weight.shape
    if n % 32 or k % 128:
        raise ValueError(f"Unsupported MPS quantization shape {(n, k)}")
    w = weight.detach().float().cpu().contiguous()
    if not torch.isfinite(w).all():
        raise ValueError("Weights must be finite")
    if bits == 8:
        scale = (w.abs().amax(dim=1) / 127.5).clamp_min(torch.finfo(torch.float32).eps)
        q = (w / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        return q, scale
    groups = w.reshape(n, k // group_size, group_size)
    minimum, maximum = groups.amin(dim=-1), groups.amax(dim=-1)
    scale = (maximum - minimum).clamp_min(1e-6) / 15
    offset = minimum + 8 * scale
    q = ((groups - minimum[..., None]) / scale[..., None]).round().clamp(0, 15)
    return q.to(torch.uint8).reshape(n, k), torch.stack((scale, offset), dim=-1)


class MPSWeightOnlyLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, *, bits: int, group_size: int = 64):
        super().__init__()
        if torch.__version__.split("+")[0] != "2.9.1":
            raise RuntimeError("Native quantization recipe is validated only with pinned torch2.9.1")
        if weight.device.type != "mps" or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Native MPS quantization requires FP16/BF16 MPS weights")
        q, parameters = quantize_weight(weight, bits, group_size)
        self.out_features, self.in_features = weight.shape
        self.bits, self.group_size = bits, group_size
        if bits == 8:
            self.register_buffer("packed", q.to("mps"))
            self.register_buffer("scales", parameters.to(device="mps", dtype=weight.dtype))
        else:
            packed_bytes = ((q[:, 0::2] << 4) | q[:, 1::2]).contiguous().to("mps")
            self.register_buffer("packed", torch._convert_weight_to_int4pack(packed_bytes, 2))
            self.register_buffer(
                "scales",
                parameters.transpose(0, 1).contiguous().to(device="mps", dtype=weight.dtype),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type != "mps"
            or x.dtype != self.scales.dtype
            or x.shape[-1] != self.in_features
        ):
            raise ValueError("Activation shape/device/dtype differs from the quantization recipe")
        matrix = x.reshape(-1, self.in_features).contiguous()
        if self.bits == 8:
            result = torch._weight_int8pack_mm(matrix, self.packed, self.scales)
        else:
            result = torch._weight_int4pack_mm(matrix, self.packed, self.group_size, self.scales)
        return result.reshape(*x.shape[:-1], self.out_features)


def quantize_model(model: nn.Module, *, bits: int) -> dict:
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith(("backbone_model.layers.", "depth_decoder.model.layers."))
    ]
    inventory = []
    total_model_parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    original_bytes = packed_bytes = 0
    replacements = []
    for name, module in selected:
        if module.bias is not None:
            raise ValueError(f"Candidate does not support bias: {name}")
        replacement = MPSWeightOnlyLinear(module.weight, bits=bits)
        replacements.append((name, replacement))
        inventory.append(dict(module=name, shape=list(module.weight.shape)))
        original_bytes += module.weight.numel() * module.weight.element_size()
        packed_bytes += sum(item.numel() * item.element_size() for item in replacement.buffers())
    if not inventory:
        raise ValueError("No eligible Breeze layers found")
    # All adapters must construct successfully before mutating the model.
    for name, replacement in replacements:
        parent, _, child = name.rpartition(".")
        model.get_submodule(parent).__setattr__(child, replacement)
    return dict(
        backend="pytorch-native-mps",
        torch_version=torch.__version__,
        bits=bits,
        group_size=64 if bits == 4 else None,
        modules=inventory,
        original_bytes=original_bytes,
        packed_bytes=packed_bytes,
        total_model_parameter_bytes=total_model_parameter_bytes,
        selected_parameter_fraction=original_bytes / total_model_parameter_bytes,
    )
