import pytest
import torch
from models.static_depth import StaticDepthRunner
from test_cached_depth import tiny_depth


@pytest.mark.parametrize("books", [4, 16])
def test_static_depth_logits_and_cache_reset(books):
    torch.manual_seed(72)
    outer = tiny_depth(books)
    runner = StaticDepthRunner(outer.model, compile_decode=False)
    pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in runner.cache.layers]
    with torch.inference_mode():
        for _ in range(2):
            sequence = torch.randint(0, 16, (1, books + 1))
            branch_hidden = torch.randn(2, 24)
            for step in range(books - 1):
                prefix = sequence[:, : step + 2].repeat(2, 1)
                expected = outer(
                    input_ids=prefix, backbone_last_hidden_state=branch_hidden, use_cache=False
                ).logits[:, -1:, :]
                hidden = (
                    runner.begin(prefix, branch_hidden)
                    if step == 0
                    else runner.step(prefix[:, -1:], step)
                )
                actual = torch.nn.functional.linear(hidden, outer.codebooks_head.weight[step].T)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert pointers == [
        (layer.keys.data_ptr(), layer.values.data_ptr()) for layer in runner.cache.layers
    ]
    with pytest.raises(ValueError, match="CFG pair"):
        runner.begin(torch.zeros(4, 2, dtype=torch.long), torch.zeros(4, 24))
    with pytest.raises(ValueError, match="codebook order"):
        runner.step(torch.zeros(2, 1, dtype=torch.long), 1)
