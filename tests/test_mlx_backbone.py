"""Optional MLX backbone continuation and actual-shape kernel tests."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from breeze_infer.mlx_backbone import MLXBackbone  # noqa: E402
from breeze_infer.mlx_depth import from_torch  # noqa: E402
from models.breeze_backbone_factory import BreezeBackboneAdapter  # noqa: E402
from models.breeze_config import BreezeConfig  # noqa: E402
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model  # noqa: E402


def host(value):
    return np.array(value.astype(mx.float32))


def fixture(hidden=32, dtype=torch.float32):
    torch.manual_seed(42)
    qcfg = Qwen3Config(
        vocab_size=23,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        rope_scaling=None,
        max_position_embeddings=128,
        use_sliding_window=False,
        layer_types=["full_attention"] * 2,
        attention_bias=False,
        attention_dropout=0.0,
    )
    qcfg._attn_implementation = "eager"
    qwen = Qwen3Model(qcfg).eval()
    # Deliberately different outer parameters detect accidentally using Breeze's
    # top-level configuration instead of the actual nested Qwen3 layer settings.
    bcfg = BreezeConfig(
        num_codebooks=4,
        vocab_size=19,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        audio_embed_size=hidden,
        backbone_model_type="qwen3",
        backbone_config=qcfg.to_dict(),
        rms_norm_eps=0.1,
        rope_theta=50.0,
    )
    bcfg._attn_implementation = "eager"
    adapter = BreezeBackboneAdapter(bcfg, qwen.layers, qwen.norm, qwen.rotary_emb).eval().to(dtype)
    with torch.no_grad():
        for layer in adapter.layers:
            layer.self_attn.q_norm.weight.copy_(torch.linspace(0.5, 1.5, hidden // 4))
            layer.self_attn.k_norm.weight.copy_(torch.linspace(1.5, 0.5, hidden // 4))
    return adapter


def prefills(adapter):
    dtype = adapter.norm.weight.dtype
    outputs, masks = [], []
    with torch.inference_mode():
        for length in (3, 5):
            mask = torch.ones((1, length), dtype=torch.long)
            outputs.append(
                adapter(
                    inputs_embeds=torch.randn(1, length, adapter.norm.weight.numel()).to(dtype),
                    attention_mask=mask,
                    position_ids=torch.arange(length)[None],
                    cache_position=torch.arange(length),
                    use_cache=True,
                )
            )
            masks.append(mask)
    return [o.past_key_values for o in outputs], masks


@pytest.mark.parametrize("kind", ["eager", "sdpa"])
@pytest.mark.parametrize("compiled", [False, True])
def test_unequal_prefixes_positions_and_growing_compiled_cache(kind, compiled):
    adapter = fixture()
    candidate = MLXBackbone(adapter, max_positions=128, attention_kind=kind, cache_chunk=8)
    assert candidate.epsilon == 1e-6
    caches, masks = prefills(adapter)
    state = candidate.pair_torch_caches(caches, masks)
    np.testing.assert_array_equal(
        np.array(state[1]),
        [
            [False, False, True, True, True, False, False, False],
            [True] * 5 + [False] * 3,
        ],
    )
    np.testing.assert_array_equal(np.array(state[2]), [[3], [5]])
    initial_state = state
    runner = candidate.step_runner(compiled=compiled)
    first_input, first_result = None, None
    with torch.inference_mode():
        for step in range(5):
            inputs = torch.randn(1, 1, 32).repeat(2, 1, 1)
            expected = torch.cat(
                [
                    adapter(
                        inputs_embeds=inputs[row : row + 1],
                        attention_mask=torch.ones(1, length + step + 1, dtype=torch.long),
                        position_ids=torch.tensor([[length + step]]),
                        cache_position=torch.tensor([length + step]),
                        past_key_values=caches[row],
                        use_cache=True,
                    ).last_hidden_state
                    for row, length in enumerate((3, 5))
                ]
            )
            previous = state
            snapshot = (
                tuple(tuple(host(value).copy() for value in pair) for pair in state[0]),
                np.array(state[1]).copy(),
            )
            actual, state = runner(from_torch(inputs), state)
            mx.eval(actual, state)
            np.testing.assert_allclose(host(actual), expected.numpy(), atol=3e-6, rtol=3e-5)
            np.testing.assert_array_equal(np.array(previous[1]), snapshot[1])
            for old_pair, saved_pair in zip(previous[0], snapshot[0]):
                for old_value, saved_value in zip(old_pair, saved_pair):
                    np.testing.assert_array_equal(host(old_value), saved_value)
            for old, new in zip(previous[0], state[0]):
                for before, after in zip(old, new):
                    np.testing.assert_array_equal(
                        host(before[..., : previous[3].item(), :]),
                        host(after[..., : previous[3].item(), :]),
                    )
            if step == 0:
                first_input, first_result = from_torch(inputs), actual
    # A→B→A after evaluated later steps, using an immutable transferred snapshot.
    repeated, _ = runner(first_input, initial_state)
    np.testing.assert_array_equal(host(repeated), host(first_result))
    bad_positions = mx.array([[5], [5]], dtype=mx.int32)
    wrong, _ = runner(
        first_input,
        (initial_state[0], initial_state[1], bad_positions, initial_state[3]),
    )
    assert np.max(np.abs(host(wrong[0]) - host(first_result[0]))) > 1e-5
    # Same input shape with changed positions must not reuse captured values.
    repeated, _ = runner(first_input, initial_state)
    np.testing.assert_array_equal(host(repeated), host(first_result))
    cache, mask, positions, offset = initial_state
    adversarial = tuple(
        tuple(mx.where(mask[:, None, :, None], value, 10000) for value in pair) for pair in cache
    )
    protected, _ = runner(first_input, (adversarial, mask, positions, offset))
    np.testing.assert_allclose(host(protected), host(first_result), atol=1e-6, rtol=1e-6)
    swapped = (
        tuple(tuple(value[::-1] for value in pair) for pair in cache),
        mask[::-1],
        positions[::-1],
        offset,
    )
    output, _ = runner(first_input[::-1], swapped)
    np.testing.assert_allclose(host(output), host(first_result[::-1]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_audio_embedding_sum_and_output_head(dtype):
    adapter = fixture(dtype=dtype)
    head = torch.nn.Linear(32, 20, bias=False).eval().to(dtype)
    candidate = MLXBackbone(adapter, head_weight=head.weight, max_positions=128)
    tokens = torch.tensor([[0, 5, 17, 3], [1, 2, 3, 4]])
    with torch.inference_mode():
        expected = adapter.embed_tokens(tokens[:, None])
        actual = candidate.audio_embeddings(mx.array(tokens.numpy()))
        np.testing.assert_allclose(host(actual), expected.float().numpy(), atol=2e-6, rtol=1e-6)
        logits = candidate.logits(actual)
        np.testing.assert_allclose(
            host(logits),
            head(expected)[:, -1].float().numpy(),
            atol=0.01 if dtype == torch.bfloat16 else 2e-6,
            rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
        )
        assert logits.shape == (2, 20)  # includes utterance EOS, unlike depth head


def test_state_validation():
    adapter = fixture()
    candidate = MLXBackbone(adapter, max_positions=128)
    caches, masks = prefills(adapter)
    state = candidate.pair_torch_caches(caches, masks)
    with pytest.raises(ValueError, match="mask"):
        candidate.pair_torch_caches(caches, [torch.ones(1, 2), masks[1]])
    runner = candidate.step_runner()
    with pytest.raises(ValueError, match="paired one-token"):
        runner(mx.zeros((1, 1, 32)), state)
    with pytest.raises(ValueError, match="bound"):
        runner(mx.zeros((2, 1, 32)), (state[0], state[1], mx.array([[128], [5]]), state[3]))
    with pytest.raises(ValueError, match="vocabulary"):
        candidate.audio_embeddings(mx.array([[0, 0, 0, 19]]))
    future = mx.slice_update(state[1], mx.ones((2, 1), dtype=mx.bool_), state[3] + 1, (1,))
    with pytest.raises(ValueError, match="Future cache"):
        runner(mx.zeros((2, 1, 32)), (state[0], future, state[2], state[3]))


@pytest.mark.parametrize("value", [-1.0, 0.5, 2.0, float("nan"), float("inf")])
def test_transferred_mask_rejects_nonbinary_values(value):
    adapter = fixture()
    candidate = MLXBackbone(adapter, max_positions=128)
    caches, masks = prefills(adapter)
    masks[0] = masks[0].float()
    masks[0][0, 1] = value
    with pytest.raises(ValueError, match="finite binary"):
        candidate.pair_torch_caches(caches, masks)


@pytest.mark.parametrize("component", ["embedding", "head", "projector"])
def test_rejects_unimplemented_precision_or_projection(component):
    adapter = fixture()
    head = torch.nn.Linear(32, 20, bias=False).eval()
    if component == "embedding":
        adapter.embed_tokens.embed_audio_tokens.to(torch.bfloat16)
    elif component == "head":
        head.to(torch.bfloat16)
    else:
        adapter.embed_tokens.audio_embeds_projector = torch.nn.Linear(32, 32)
    with pytest.raises(ValueError, match="precision|projection"):
        MLXBackbone(adapter, head_weight=head.weight, max_positions=128)


@pytest.mark.parametrize("compiled", [False, True])
def test_cache_expansion_at_nonmultiple_position_limit(compiled):
    adapter = fixture()
    candidate = MLXBackbone(adapter, max_positions=10, cache_chunk=8)
    state = candidate.pair_torch_caches(*prefills(adapter))
    runner = candidate.step_runner(compiled=compiled)
    for offset in range(5, 10):
        output, state = runner(mx.zeros((2, 1, 32)), state)
        mx.eval(output, state)
        assert state[3].item() == offset + 1
        assert state[1].shape[-1] == (8 if offset < 8 else 10)
    with pytest.raises(ValueError, match="bound"):
        runner(mx.zeros((2, 1, 32)), state)


@pytest.mark.parametrize(
    "out_size,in_size", [(2048, 2048), (1024, 2048), (6144, 2048), (2048, 6144)]
)
def test_backbone_int8_actual_shapes(out_size, in_size):
    weight = (mx.random.normal((out_size, in_size), key=mx.random.key(42)) * 0.02).astype(
        mx.bfloat16
    )
    packed, scales, biases = mx.quantize(weight, group_size=64, bits=8, mode="affine")
    restored = mx.dequantize(packed, scales, biases, group_size=64, bits=8, mode="affine")
    x = mx.random.normal((2, 1, in_size), key=mx.random.key(7)).astype(mx.bfloat16)
    actual = mx.quantized_matmul(x, packed, scales, biases, group_size=64, bits=8, mode="affine")
    expected = x @ restored.T
    mx.eval(actual, expected)
    error = np.linalg.norm(host(actual) - host(expected)) / np.linalg.norm(host(expected))
    assert error < 0.01
    assert actual.dtype == mx.bfloat16


def test_backbone_quantization_scope_and_execution():
    adapter = fixture(hidden=64, dtype=torch.bfloat16)
    candidate = MLXBackbone(adapter, max_positions=128, quant_bits=8)
    assert len(candidate.quantized) == 14
    for name in (
        "norm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.input_layernorm.weight",
    ):
        np.testing.assert_array_equal(
            host(candidate.weights[name]), adapter.state_dict()[name].float().numpy()
        )
    caches, masks = prefills(adapter)
    state = candidate.pair_torch_caches(caches, masks)
    result = candidate.step_runner(compiled=True)(mx.zeros((2, 1, 64), dtype=mx.bfloat16), state)
    mx.eval(result)
    assert result[0].shape == (2, 1, 64)
