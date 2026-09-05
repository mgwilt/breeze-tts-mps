from types import SimpleNamespace

import pytest
import torch
from models.breeze import BreezeDepthDecoderForCausalLM
from models.breeze_config import BreezeDepthDecoderConfig
from models.generation_breeze import BreezeGenerationMixin


def tiny_depth(codebooks):
    config = BreezeDepthDecoderConfig(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_codebooks=codebooks,
        backbone_hidden_size=24,
        audio_embed_dim=12,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return BreezeDepthDecoderForCausalLM(config).eval()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("codebooks", [4, 16])
@pytest.mark.parametrize("single_head", [False, True])
def test_cached_cfg_teacher_forced_logits(batch, codebooks, single_head):
    torch.manual_seed(42)
    model = tiny_depth(codebooks)
    tokens = torch.randint(0, 16, (batch, codebooks))
    cond, uncond = torch.randn(batch, 24), torch.randn(batch, 24)
    past = None
    with torch.inference_mode():
        for step in range(codebooks - 1):
            prefix = tokens[:, : step + 2]
            reference = torch.cat(
                [
                    model(
                        input_ids=prefix, backbone_last_hidden_state=hidden, use_cache=False
                    ).logits[:, -1:, :]
                    for hidden in (cond, uncond)
                ]
            )
            new = prefix if step == 0 else prefix[:, -1:]
            positions = torch.arange(2) if step == 0 else torch.tensor([step + 1])
            result = model(
                input_ids=torch.cat([new, new]),
                backbone_last_hidden_state=torch.cat([cond, uncond]) if step == 0 else None,
                past_key_values=past,
                cache_position=positions,
                use_cache=True,
                logits_to_keep=1,
                codebook_index=step if single_head else None,
            )
            past = result.past_key_values
            torch.testing.assert_close(result.logits, reference, atol=1e-6, rtol=1e-5)
        assert past.get_seq_length() == codebooks


@pytest.mark.parametrize("cfg", [1.0, 4.0])
def test_cached_cfg_preserves_greedy_tokens(cfg):
    class Harness(BreezeGenerationMixin):
        pass

    torch.manual_seed(42)
    model = tiny_depth(4)
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    runner = Harness()
    runner.depth_decoder = model
    runner.config = SimpleNamespace(
        num_codebooks=4, vocab_size=19, codec_config=SimpleNamespace(codebook_size=16)
    )
    inputs = torch.tensor([[0, 2], [0, 7]])
    cond, uncond = torch.randn(2, 24), torch.randn(2, 24)
    with torch.inference_mode():
        reference = runner._depth_decoder_generate_with_cfg(inputs, cond, uncond, cfg)
        runner._cached_depth_cfg = True
        cached = runner._depth_decoder_generate_with_cfg(inputs, cond, uncond, cfg)
    assert torch.equal(cached, reference)
