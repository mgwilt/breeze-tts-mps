"""Optional MLX candidate tests; the locked serving environment need not install MLX."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from breeze_infer.mlx_depth import (  # noqa: E402
    MLXDepth,
    Sampling,
    attention,
    filter_probabilities,
    from_torch,
    rms_norm,
    scale_attention_scores,
    rotary,
    sampling_probabilities,
    silu_product,
)
from test_cached_depth import tiny_depth  # noqa: E402


def array(value):
    return np.array(value.astype(mx.float32))


def test_bf16_precision_boundaries():
    x = mx.array([0.125, 1, 2, 3], dtype=mx.bfloat16)
    w = mx.array([1.1, 0.7, 1.3, -0.9], dtype=mx.bfloat16)
    np.testing.assert_array_equal(
        array(rms_norm(x, w, 1e-5)), [0.07373046875, 0.375, 1.390625, -1.4375]
    )
    product = silu_product(
        mx.array(-0.375, dtype=mx.bfloat16), mx.array(-0.8984375, dtype=mx.bfloat16)
    )
    assert product.item() == 0.13671875
    angles = torch.tensor([0.2, 1.0, 0.2, 1.0])
    q = mx.array([1.25, -0.375, 0.75, -2.5], dtype=mx.bfloat16)
    result = rotary(q, from_torch(angles.cos().bfloat16()), from_torch(angles.sin().bfloat16()))
    np.testing.assert_array_equal(array(result), [1.078125, 1.890625, 0.984375, -1.65625])


def test_attention_scale_preserves_torch_scalar_precision():
    scores = torch.arange(-1000, 1001).bfloat16()
    scale = 128**-0.5
    expected = (scores * scale).float().numpy()
    inputs = from_torch(scores)
    for operation in (
        lambda x: scale_attention_scores(x, scale),
        mx.compile(lambda x: scale_attention_scores(x, scale)),
    ):
        np.testing.assert_array_equal(array(operation(inputs)), expected)
    # Negative control: the former implementation loses scalar precision.
    assert not np.array_equal(array(inputs * scale), expected)


@pytest.mark.parametrize(
    "probs,k,p,expected",
    [
        ([0.4, 0.3, 0.2, 0.1], 2, 0.55, [1, 0, 0, 0]),
        ([0.5, 0.3, 0.2], 0, 0.6, [0.625, 0.375, 0]),
        ([0.5, 0.25, 0.125, 0.125], 0, 0.5, [2 / 3, 1 / 3, 0, 0]),
        ([0.5, 0.3, 0.2], 99, 1, [0.5, 0.3, 0.2]),
        ([0.5, 0.3, 0.2], 0, 0, [1, 0, 0]),
    ],
)
def test_filter_order_and_crossing(probs, k, p, expected):
    result = filter_probabilities(mx.array([probs]), k, p)
    np.testing.assert_allclose(array(result)[0], expected, atol=1e-6)


def test_temperature_and_reserved_mask():
    logits = mx.log(mx.array([[0.45, 0.35, 0.20, 1000], [0.45, 0.35, 0.20, 1000]]))
    probs = sampling_probabilities(logits, Sampling(temperature=0.5, top_k=0, top_p=0.5), 3)
    np.testing.assert_array_equal(array(probs), [[1, 0, 0, 0]])


@pytest.mark.parametrize("kind", ["eager", "sdpa"])
def test_gqa_branch_isolation(kind):
    q, k = mx.zeros((2, 8, 1, 8)), mx.zeros((2, 2, 1, 8))
    v = mx.broadcast_to(mx.array([10, 20, -10, -20]).reshape(2, 2, 1, 1), (2, 2, 1, 8)).astype(
        mx.float32
    )
    result = attention(q, k, v, 0, kind)
    expected = np.repeat(array(v), 4, axis=1)
    np.testing.assert_array_equal(array(result), expected)


@pytest.mark.parametrize("kind", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_teacher_forced_and_frame_isolation(kind, dtype):
    torch.manual_seed(42)
    reference = tiny_depth(4).to(dtype)
    candidate = MLXDepth(reference, valid_size=16, attention_kind=kind)
    hidden = torch.randn(2, 24).to(dtype)
    tokens = torch.tensor([[0, 3, 8, 4], [0, 3, 8, 4]])
    cache = ()
    with torch.inference_mode():
        for step in range(3):
            prefix = tokens[:, : step + 2]
            expected = reference(
                input_ids=prefix, backbone_last_hidden_state=hidden, use_cache=False
            ).logits[:, -1]
            full, _ = candidate.forward(mx.array(prefix.numpy()), from_torch(hidden))
            ids = prefix if step == 0 else prefix[:, -1:]
            result, cache = candidate.forward(
                mx.array(ids.numpy()), from_torch(hidden) if step == 0 else None, cache
            )
            tolerance = 0.003 if dtype == torch.bfloat16 else 2e-6
            np.testing.assert_allclose(
                array(result), expected.float().numpy(), atol=tolerance, rtol=tolerance
            )
            np.testing.assert_allclose(array(full), array(result), atol=tolerance, rtol=tolerance)
    initial = mx.array([[0, 3]], dtype=mx.int32)
    generate = candidate.generator(Sampling(do_sample=False))
    first, key = generate(initial, from_torch(hidden), mx.random.key(42))
    mx.eval(first, key)
    mx.eval(generate(initial, from_torch(-hidden), key))
    repeated, _ = generate(initial, from_torch(hidden), mx.random.key(7))
    np.testing.assert_array_equal(np.array(first), np.array(repeated))
    alternate, _ = generate(mx.array([[15, 3]], dtype=mx.int32), from_torch(hidden), key)
    np.testing.assert_array_equal(np.array(first)[:, 1:], np.array(alternate)[:, 1:])


def test_compiled_sampling_has_explicit_key_and_fresh_cache():
    torch.manual_seed(11)
    candidate = MLXDepth(tiny_depth(4), valid_size=16)
    hidden = from_torch(torch.randn(2, 24))
    initial, key = mx.array([[0, 3]], dtype=mx.int32), mx.random.key(42)
    generate = candidate.generator(compiled=True)
    first, next_key = generate(initial, hidden, key)
    mx.eval(first, next_key)
    alternate = generate(mx.array([[0, 9]], dtype=mx.int32), -hidden, mx.random.key(7))
    mx.eval(alternate)
    oracle = candidate.generator()(mx.array([[0, 9]], dtype=mx.int32), -hidden, mx.random.key(7))
    mx.eval(oracle)
    np.testing.assert_array_equal(np.array(alternate[0]), np.array(oracle[0]))
    repeated, repeated_key = generate(initial, hidden, key)
    np.testing.assert_array_equal(np.array(first), np.array(repeated))
    np.testing.assert_array_equal(np.array(next_key), np.array(repeated_key))
    assert not np.array_equal(np.array(key), np.array(next_key))
    expected_key = key
    for _ in range(candidate.books - 1):
        expected_key = mx.random.split(expected_key)[0]
    np.testing.assert_array_equal(np.array(next_key), np.array(expected_key))
    assert np.array(first).shape == (1, 5)
    assert (np.array(first) < 16).all()


def test_input_validation():
    candidate = MLXDepth(tiny_depth(4), valid_size=16)
    with pytest.raises(ValueError, match="positions"):
        candidate.forward(mx.array([[2]]))
    with pytest.raises(ValueError, match="hidden"):
        candidate.forward(mx.array([[0, 2]]))
    with pytest.raises(ValueError, match="paired"):
        candidate.generator()(mx.array([[0, 2]]), mx.zeros((1, 24)), mx.random.key(0))
    with pytest.raises(ValueError, match="Temperature"):
        Sampling(temperature=0)
    with pytest.raises(ValueError, match="top-k"):
        Sampling(top_k=2.5)
    with pytest.raises(ValueError, match="vocabulary"):
        candidate.generator()(mx.array([[0, 19]]), mx.zeros((2, 24)), mx.random.key(0))
    with pytest.raises(ValueError, match="vocabulary"):
        candidate.forward(mx.array([[0, -1]]), mx.zeros((1, 24)))
    with pytest.raises(ValueError, match="evaluation"):
        MLXDepth(tiny_depth(4).train(), valid_size=16)
    bf16 = MLXDepth(tiny_depth(4).bfloat16(), valid_size=16)
    _, cache = bf16.forward(mx.array([[0, 1]]), mx.zeros((1, 24), dtype=mx.bfloat16))
    wrong = tuple((k.astype(mx.float32), v.astype(mx.float32)) for k, v in cache)
    with pytest.raises(ValueError, match="precision"):
        bf16.forward(mx.array([[2]]), cache=wrong)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize(
    "out_size,in_size", [(1024, 1024), (256, 1024), (8192, 1024), (1024, 8192)]
)
def test_quantized_actual_shapes_and_paired_cfg(bits, out_size, in_size):
    # Every distinct eligible production matrix shape, both prefill and decode.
    # Compare the kernel to its dequantized weights, not the original model: this
    # is a kernel/shape screen; quantization quality needs production evidence.
    mx.random.seed(42)
    weight = (mx.random.normal((out_size, in_size)) * 0.02).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits, mode="affine")
    restored = mx.dequantize(packed, scales, biases, group_size=64, bits=bits, mode="affine")
    for length in (1, 2):
        x = mx.random.normal((2, length, in_size)).astype(mx.bfloat16)
        actual = mx.quantized_matmul(
            x,
            packed,
            scales,
            biases,
            transpose=True,
            group_size=64,
            bits=bits,
            mode="affine",
        )
        expected = x @ restored.T
        mx.eval(actual, expected)
        assert actual.dtype == mx.bfloat16
        assert actual.shape == (2, length, out_size)
        relative_error = mx.sqrt(
            mx.sum((actual.astype(mx.float32) - expected.astype(mx.float32)) ** 2)
            / mx.sum(expected.astype(mx.float32) ** 2)
        ).item()
        assert relative_error < 0.01
        # Keep the matrix batch shape constant while replacing the other branch.
        # Batch1 vs batch2 can dispatch different quantized reduction kernels and
        # does not isolate cross-row contamination from arithmetic differences.
        isolated = mx.quantized_matmul(
            mx.concatenate([x[:1], mx.zeros_like(x[1:])]),
            packed,
            scales,
            biases,
            transpose=True,
            group_size=64,
            bits=bits,
            mode="affine",
        )
        np.testing.assert_array_equal(array(actual[:1]), array(isolated[:1]))


@pytest.mark.parametrize("bits", [8, 4])
def test_quantization_excludes_custom_modules(bits):
    from models.breeze import BreezeDepthDecoderForCausalLM
    from models.breeze_config import BreezeDepthDecoderConfig

    config = BreezeDepthDecoderConfig(
        vocab_size=19,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_codebooks=4,
        backbone_hidden_size=64,
    )
    model = BreezeDepthDecoderForCausalLM(config).eval().bfloat16()
    candidate = MLXDepth(model, valid_size=16, quant_bits=bits)
    assert len(candidate.quantized) == 7
    for name in (
        "model.embed_tokens.weight",
        "codebooks_head.weight",
        "model.inputs_embeds_projector.weight",
        "model.norm.weight",
    ):
        assert name in candidate.weights
        assert name not in candidate.quantized
        np.testing.assert_array_equal(
            array(candidate.weights[name]), model.state_dict()[name].float().numpy()
        )
    assert 0 < candidate.quantization["coverage"] < 1
    generator = candidate.generator(compiled=True)
    result = generator(mx.array([[0, 2]]), mx.zeros((2, 64), dtype=mx.bfloat16), mx.random.key(1))
    mx.eval(result)
    assert np.array(result[0]).shape == (1, 5)
