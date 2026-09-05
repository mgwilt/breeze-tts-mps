"""Optional standalone MLX sampling and portable streaming integration tests."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

mx = pytest.importorskip("mlx.core")
from breeze_infer.mlx_depth import Sampling  # noqa: E402
from breeze_infer.mlx_speech import (  # noqa: E402
    MLXSpeechModel,
    backbone_scores,
    sample_backbone,
    sampling_settings,
)
from breeze_infer.portable_runtime import PortableBreezeStreamingRuntime  # noqa: E402


@pytest.mark.parametrize(
    "top_k,top_p,sample",
    [
        (0, 1.0, True),
        (2, 1.0, True),
        (3, 0.5, True),
        (0, 0.5, True),
        (0, 0.0, True),
        (50, 0.9, False),
    ],
)
def test_hf_backbone_filter_parity_and_reserved_order(top_k, top_p, sample):
    # Reserved indices4..6 deliberately score highly, and ties straddle top-k.
    logits = torch.tensor([[1.0, 2.0, 2.0, 2.0, 8.0, 1.0, 1.0, 3.0], [0.0] * 8])
    settings = Sampling(cfg=1, temperature=0.9, top_k=top_k, top_p=top_p, do_sample=sample)
    processors = LogitsProcessorList()
    if sample:
        processors.append(TemperatureLogitsWarper(0.9))
        if top_k:
            processors.append(TopKLogitsWarper(top_k))
        if top_p < 1:
            processors.append(TopPLogitsWarper(top_p))
    expected = processors(torch.zeros((1, 1), dtype=torch.long), logits[:1].clone())
    expected[:, 4:7] = -float("inf")
    actual = np.array(backbone_scores(mx.array(logits.numpy()), settings, 4))
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-6)


def test_hf_topk_ties_topp_boundary_and_eos():
    logits = mx.array([[2.0, 2.0, 2.0, 1.0, -9.0, -9.0, -9.0, 0.0], [0.0] * 8])
    scores = np.array(backbone_scores(logits, Sampling(cfg=1, temperature=1, top_k=2), 4))
    assert np.isfinite(scores).sum() == 3
    logits = mx.array([np.log([0.5, 0.25, 0.125, 0.125]).tolist(), [0.0] * 4])
    scores = np.array(
        backbone_scores(logits, Sampling(cfg=1, temperature=1, top_k=0, top_p=0.5), 3)
    )
    assert np.isfinite(scores).sum() == 1 and np.isfinite(scores[0, 0])
    logits = mx.array([[0.0, 0.0, 0.0, 9.0], [0.0] * 4])
    token, _ = sample_backbone(logits, Sampling(cfg=1, do_sample=False), 2, mx.random.key(3))
    assert token.item() == 3


def test_fully_filtered_distribution_fails_and_seed_is_explicit():
    settings = Sampling(cfg=1, top_k=1)
    logits = mx.array([[0.0, 0.0, 99.0, 0.0], [0.0] * 4])
    with pytest.raises(ValueError, match="filtered"):
        sample_backbone(logits, settings, 2, mx.random.key(42))
    logits = mx.zeros((2, 8))
    settings = Sampling(cfg=4, top_k=0)
    first = sample_backbone(logits, settings, 4, mx.random.key(42))
    second = sample_backbone(logits, settings, 4, mx.random.key(42))
    mx.eval(first, second)
    np.testing.assert_array_equal(np.array(first[0]), np.array(second[0]))
    np.testing.assert_array_equal(np.array(first[1]), np.array(second[1]))
    assert not np.array_equal(np.array(first[1]), np.array(mx.random.key(42)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("repetition_penalty", 1.1),
        ("num_beams", 2),
        ("min_new_tokens", 1),
        ("min_p", 0.1),
        ("bad_words_ids", [[2]]),
        ("use_cache", False),
        ("renormalize_logits", True),
        ("sequence_bias", {(2,): 1.0}),
    ],
)
def test_rejects_unsupported_generation_settings(field, value):
    config = GenerationConfig(do_sample=True, max_new_tokens=750)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="Unsupported generation setting"):
        sampling_settings(config, 4)


def test_supported_explicit_config_and_depth_bound():
    cfg = GenerationConfig(do_sample=True, temperature=0.9, top_k=50, top_p=1.0, max_new_tokens=750)
    assert sampling_settings(cfg, 4) == Sampling()
    cfg.min_new_tokens, cfg.max_new_tokens = 15, 15
    assert sampling_settings(cfg, 4, depth=True) == Sampling()
    cfg.max_new_tokens = 14
    with pytest.raises(ValueError, match="depth generation length"):
        sampling_settings(cfg, 4, depth=True)


def reference_config():
    cfg = GenerationConfig(do_sample=True, temperature=0.9, top_k=50, top_p=1.0, max_new_tokens=750)
    for name in ("do_sample", "temperature", "top_k", "top_p"):
        setattr(cfg, "depth_decoder_" + name, getattr(cfg, name))
    return SimpleNamespace(
        config=SimpleNamespace(
            num_codebooks=16,
            vocab_size=2051,
            codebook_pad_token_id=2050,
            codec_config=SimpleNamespace(codebook_size=2048),
        ),
        generation_config=cfg,
        depth_decoder=SimpleNamespace(
            generation_config=GenerationConfig(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
        ),
        backbone_model=None,
        lm_head=SimpleNamespace(weight=None),
    )


def test_constructor_freezes_recipe_and_rejects_conflicting_metadata(monkeypatch):
    import breeze_infer.mlx_speech as module

    monkeypatch.setattr(
        module,
        "MLXBackbone",
        lambda *args, **kwargs: SimpleNamespace(step_runner=lambda **kw: None),
    )
    monkeypatch.setattr(
        module,
        "MLXDepth",
        lambda *args, **kwargs: SimpleNamespace(generator=lambda *args, **kw: None),
    )
    reference = reference_config()
    model = MLXSpeechModel(reference)
    reference.generation_config.temperature = 4.0
    reference.config.vocab_size = 99
    assert model.backbone_settings.temperature == 0.9 and model.config.vocab_size == 2051
    reference = reference_config()
    reference.generation_config.depth_decoder_top_p = 0.5
    with pytest.raises(ValueError, match="Conflicting"):
        MLXSpeechModel(reference)
    reference = reference_config()
    reference.config.codebook_pad_token_id = 0
    with pytest.raises(ValueError, match="codec/EOS"):
        MLXSpeechModel(reference)
    for cfg in (0, -1, 1, float("nan")):
        with pytest.raises(ValueError, match="paired CFG"):
            MLXSpeechModel(reference_config(), cfg=cfg)
    config = reference_config().generation_config
    config.top_k = True
    with pytest.raises(ValueError, match="integer"):
        sampling_settings(config, 4)


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, (None, None)),
        ({"quant_bits": 8}, (8, 8)),
        ({"depth_quant_bits": 8}, (None, 8)),
        ({"quant_bits": 8, "depth_quant_bits": None}, (8, None)),
        ({"quant_bits": 8, "backbone_quant_bits": None}, (None, 8)),
        ({"backbone_quant_bits": 8, "depth_quant_bits": None}, (8, None)),
        (
            {"quant_bits": 8, "backbone_quant_bits": None, "depth_quant_bits": None},
            (None, None),
        ),
    ],
)
def test_component_weight_selection_preserves_legacy_and_fresh_runners(
    monkeypatch, options, expected
):
    import breeze_infer.mlx_speech as module

    calls, runners = [], []

    def backbone(reference, *, head_weight, quant_bits):
        calls.append(("backbone", quant_bits))

        def runner(*, compiled):
            assert compiled is True
            value = object()
            runners.append(value)
            return value

        return SimpleNamespace(step_runner=runner)

    def depth(reference, *, valid_size, attention_kind, quant_bits):
        assert valid_size == 2048 and attention_kind == "sdpa"
        calls.append(("depth", quant_bits))

        def generator(settings, *, compiled):
            assert settings == Sampling() and compiled is True
            value = object()
            runners.append(value)
            return value

        return SimpleNamespace(generator=generator)

    monkeypatch.setattr(module, "MLXBackbone", backbone)
    monkeypatch.setattr(module, "MLXDepth", depth)
    first = MLXSpeechModel(reference_config(), **options)
    second = MLXSpeechModel(reference_config(), **options)
    assert calls == [("backbone", expected[0]), ("depth", expected[1])] * 2
    assert first.step is not second.step and first.depth_generate is not second.depth_generate
    assert len({id(value) for value in runners}) == 4


@pytest.mark.parametrize("field", ["quant_bits", "backbone_quant_bits", "depth_quant_bits"])
@pytest.mark.parametrize("value", [False, True, 0, 4, 8.0, "8", object()])
def test_invalid_precision_rejected_before_reference_or_component_access(field, value):
    with pytest.raises(ValueError, match="quantization"):
        MLXSpeechModel(None, **{field: value})


def test_probe_precision_arguments_preserve_unset_and_explicit_none(tmp_path):
    from breeze_infer.probe_mlx_speech import parse_args, quantization_options

    base = ["--model-path", str(tmp_path), "--audio-dir", str(tmp_path / "new")]
    assert quantization_options(parse_args(base)) == {"quant_bits": None}
    assert quantization_options(
        parse_args(base + ["--quant-bits", "8", "--depth-quant-bits", "none"])
    ) == {"quant_bits": 8, "depth_quant_bits": None}
    assert quantization_options(
        parse_args(base + ["--backbone-quant-bits", "8", "--depth-quant-bits", "none"])
    ) == {"quant_bits": None, "backbone_quant_bits": 8, "depth_quant_bits": None}
    for flag in ("--quant-bits", "--backbone-quant-bits", "--depth-quant-bits"):
        with pytest.raises(SystemExit) as error:
            parse_args(base + [flag, "4"])
        assert error.value.code == 2
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "extra",
    [
        {"logits_processor": []},
        {"generation_config": None},
        {"cfg_scale_ref": 4.0},
        {"input_values": torch.zeros(1)},
        {"cfg_scale": 1.0},
    ],
)
def test_unsupported_request_rejected_before_prefill(extra):
    model = MLXSpeechModel.__new__(MLXSpeechModel)
    model.backbone_settings = Sampling()
    with pytest.raises(ValueError, match="Unsupported|Reference audio"):
        model._prefill({"input_ids": torch.tensor([[1]]), "cfg_scale": 4.0, **extra})


@pytest.mark.parametrize("defect", ["rank", "missing", "mask", "lengths", "right_pad", "limit"])
def test_complete_prefix_validation_precedes_model_execution(defect):
    model = MLXSpeechModel.__new__(MLXSpeechModel)
    model.backbone_settings, model.limit = Sampling(), 750
    model.backbone = SimpleNamespace(max_positions=4096)
    branch = dict(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        text_ids_mask=torch.ones((1, 2), dtype=torch.bool),
        text_ids_len=torch.tensor([1, 1]),
    )
    negative = {
        "cfg_negative_prompt_ids": branch["input_ids"],
        "cfg_negative_prompt_attention_mask": branch["attention_mask"],
        "cfg_negative_text_ids_mask": branch["text_ids_mask"],
        "cfg_negative_text_ids_len": branch["text_ids_len"],
    }
    if defect == "rank":
        branch["input_ids"] = torch.zeros((1, 2, 16), dtype=torch.long)
    elif defect == "missing":
        del negative["cfg_negative_text_ids_len"]
    elif defect == "mask":
        branch["text_ids_mask"] = torch.zeros((1, 2), dtype=torch.bool)
    elif defect == "lengths":
        branch["text_ids_len"] = torch.tensor([3])
    elif defect == "right_pad":
        branch["attention_mask"] = torch.tensor([[1, 0]])
    else:
        model.backbone.max_positions = 751
    with pytest.raises(ValueError, match="prefix|mask|length|limit"):
        model._prefill({**branch, **negative, "cfg_scale": 4.0})


def test_probe_corpus_bounds_and_exclusive_audio_evidence(tmp_path):
    import json
    from breeze_infer.probe_mlx_speech import load_corpus, save_audio

    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([dict(text="Hello.", instruction="Speak clearly.")]))
    prompts, identity = load_corpus(corpus)
    assert prompts == [("Hello.", "Speak clearly.")]
    import hashlib

    assert identity["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    corpus.write_text(json.dumps([dict(text="", instruction="Speak clearly.")]))
    assert identity["sha256"] != hashlib.sha256(corpus.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="entry"):
        load_corpus(corpus)
    path = tmp_path / "proof.wav"
    save_audio(path, np.zeros(10), 24000)
    with pytest.raises(FileExistsError):
        save_audio(path, np.zeros(10), 24000)


class Codec:
    def __init__(self):
        self.active, self.frames = set(), []
        self.request_pool = SimpleNamespace(active_req_ids=lambda: tuple(self.active))

    def open_request(self, request_id, **kwargs):
        assert not self.active
        self.active.add(request_id)

    def decode_request_chunk(self, request_id, codes, **kwargs):
        assert request_id in self.active and codes.shape == (1, 3, 1)
        self.frames.append(codes.clone())
        return torch.zeros((1, 1, 1920))

    def close_request(self, request_id):
        self.active.remove(request_id)


def scripted(tokens, *, limit=8, depth_bad=False, event=None):
    model = MLXSpeechModel.__new__(MLXSpeechModel)
    model.config = SimpleNamespace(
        num_codebooks=3,
        vocab_size=19,
        codebook_pad_token_id=18,
        codec_config=SimpleNamespace(sampling_rate=24000, codebook_size=16),
    )
    model.limit, model.valid_size, model.calls = limit, 16, 0
    model.backbone_settings = Sampling(do_sample=False)
    model.keys = []
    cursor = 0

    def logits():
        values = np.zeros((2, 20), dtype=np.float32)
        values[0, tokens[min(cursor, len(tokens) - 1)]] = 10
        return mx.array(values)

    def prefill(inputs, check_stop):
        nonlocal cursor
        cursor = 0
        if event is not None:
            event.set()
        return mx.zeros((2, 4)), logits(), ()

    def depth(initial, hidden, key):
        model.calls += 1
        model.keys.append(np.array(key).copy())
        key, _ = mx.random.split(key)
        return mx.concatenate(
            [initial, mx.array([[1, 99 if depth_bad else 2]], dtype=mx.int32)], axis=1
        ), key

    def step(embedding, state):
        nonlocal cursor
        cursor += 1
        return mx.zeros((2, 1, 4)), ()

    model._prefill, model.depth_generate, model.step = prefill, depth, step
    model.backbone = SimpleNamespace(
        audio_embeddings=lambda frame: mx.zeros((1, 1, 4)),
        logits=lambda hidden: logits(),
    )
    codec = Codec()
    runtime = PortableBreezeStreamingRuntime(model, None, None, codec=codec, queue_capacity=1)
    return model, runtime, codec


def inputs():
    return dict(input_ids=torch.tensor([[101]]), cfg_scale=4.0, mlx_seed=42)


def test_full_loop_zero_is_audio_eos_skips_depth_and_replay_resets_state():
    model, runtime, codec = scripted([0, 3, 19])
    assert len(list(runtime.iter_audio_chunks(inputs()))) == 2
    assert model.calls == 2 and len(codec.frames) == 2 and not codec.active
    assert codec.frames[0][0, 0, 0] == 0
    assert runtime.last_metrics["completed"] and runtime.last_metrics["eos_reached"]
    saved = [frame.clone() for frame in codec.frames]
    keys = [key.copy() for key in model.keys]
    assert len(list(runtime.iter_audio_chunks(inputs()))) == 2
    for a, b in zip(saved, codec.frames[2:]):
        assert torch.equal(a, b)
    for a, b in zip(keys, model.keys[2:]):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(keys[0], keys[1])


@pytest.mark.parametrize(
    "tokens,limit,bad,error",
    [
        ([19], 8, False, "no audio"),
        ([0], 2, False, "without EOS"),
        ([0, 19], 8, True, "depth frame"),
    ],
)
def test_failed_utterances_release_codec_without_completion(tokens, limit, bad, error):
    model, runtime, codec = scripted(tokens, limit=limit, depth_bad=bad)
    with pytest.raises((ValueError, RuntimeError), match=error):
        list(runtime.iter_audio_chunks(inputs()))
    assert not runtime.last_metrics["completed"] and not codec.active
    if tokens == [19]:
        assert model.calls == 0 and not codec.frames


def test_stop_during_prefill_and_bounded_delivery_then_retry():
    cancelled = threading.Event()
    model, runtime, codec = scripted([0, 19], event=cancelled)
    assert not list(runtime.iter_audio_chunks(inputs(), cancelled=cancelled))
    assert model.calls == 0 and not codec.active and runtime.last_metrics["cancelled"]
    cancelled.clear()
    model, runtime, codec = scripted([0], limit=750)
    iterator = runtime.iter_audio_chunks(inputs())
    next(iterator)
    iterator.close()
    assert runtime.last_metrics["cancelled"] and not codec.active
    # Replace only the scripted prefill source, keeping the same runtime/codec.
    original = model._prefill

    def eos(inputs, check_stop):
        hidden, logits, state = original(inputs, check_stop)
        return hidden, mx.array([[0.0] * 19 + [10.0], [0.0] * 20]), state

    model._prefill = eos
    with pytest.raises(RuntimeError, match="no audio"):
        list(runtime.iter_audio_chunks(inputs()))
    assert not codec.active


@pytest.mark.parametrize("mode", ["close", "event", "consumer-error", "codec-error"])
def test_probe_real_lifecycle_checks_cleanup_and_retry(mode):
    from breeze_infer.probe_mlx_speech import interrupted_request, lifecycle_state

    model, runtime, codec = scripted([0, 1, 2, 19])
    baseline = list(runtime.iter_audio_chunks(inputs()))
    result, partial = interrupted_request(runtime, inputs(), mode)
    assert result["passed"] and partial.size > 0
    assert not any(lifecycle_state(runtime).values())
    retry = list(runtime.iter_audio_chunks(inputs()))
    assert len(retry) == len(baseline) and runtime.last_metrics["completed"]
    for a, b in zip(baseline, retry):
        np.testing.assert_array_equal(a.audio, b.audio)


def test_probe_rejects_unknown_lifecycle_mode():
    from breeze_infer.probe_mlx_speech import interrupted_request

    _, runtime, _ = scripted([0, 19])
    with pytest.raises(ValueError, match="mode"):
        interrupted_request(runtime, inputs(), "unknown")
