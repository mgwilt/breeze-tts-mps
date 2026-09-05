import pytest
import torch
from breeze_infer.quantization import quantize_weight


@pytest.mark.parametrize("bits", [4, 8])
def test_zero_constant_and_random_quantization(bits):
    for weight in (torch.zeros(32, 128), torch.full((32, 128), 0.3), torch.randn(32, 128)):
        original = weight.clone()
        q, parameters = quantize_weight(weight, bits)
        assert torch.equal(original, weight)
        assert torch.isfinite(parameters).all()
        if bits == 8:
            decoded = q.float() * parameters[:, None]
            tolerance = parameters.max().item() * 1.01
        else:
            groups = q.float().reshape(32, 2, 64)
            decoded = ((groups - 8) * parameters[..., 0, None] + parameters[..., 1, None]).reshape(
                32, 128
            )
            tolerance = parameters[..., 0].max().item() * 0.51 + 1e-6
        assert (decoded - weight).abs().max().item() <= tolerance


def test_quantization_rejects_unsupported_shapes_and_recipe():
    with pytest.raises(ValueError):
        quantize_weight(torch.zeros(31, 128), 4)
    with pytest.raises(ValueError):
        quantize_weight(torch.zeros(32, 128), 2)
