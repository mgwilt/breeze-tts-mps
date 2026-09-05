from types import SimpleNamespace

import pytest

from breeze_infer.profile_stages import StageRecorder, backbone_stage


def test_stage_recorder_times_and_restores_inherited_method_on_failure():
    class Example:
        def method(self, value):
            if value < 0:
                raise ValueError("negative")
            return value + 1

    times = iter((1, 3, 4, 7))
    syncs = []
    owner = Example()
    recorder = StageRecorder(lambda: syncs.append(True), lambda: next(times))
    with recorder.installed():
        recorder.wrap(owner, "method", "call")
        assert owner.method(2) == 3
        with pytest.raises(ValueError, match="negative"):
            owner.method(-1)
    assert "method" not in vars(owner)
    assert owner.method(3) == 4
    assert recorder.summary()["call"] == dict(
        calls=2, total_s=5, mean_s=2.5, p95_s=3, samples_s=[2, 3]
    )
    assert len(syncs) == 4


def test_stage_recorder_preserves_instance_override_and_dynamic_label():
    def original(value):
        return value

    owner = SimpleNamespace(method=original)
    times = iter((0, 1))
    recorder = StageRecorder(lambda: None, lambda: next(times))
    with pytest.raises(RuntimeError):
        with recorder.installed():
            recorder.wrap(
                owner,
                "method",
                lambda args, kwargs: "prefill" if args[0] > 1 else "decode",
            )
            assert owner.method(2) == 2
            raise RuntimeError("cleanup")
    assert owner.method is original
    assert recorder.summary()["prefill"]["total_s"] == 1


@pytest.mark.parametrize("length,expected", [(0, "backbone_prefill"), (10, "backbone_decode")])
def test_backbone_label_uses_cache_not_optional_embeddings(length, expected):
    cache = SimpleNamespace(get_seq_length=lambda: length)
    assert backbone_stage((), {"inputs_embeds": None, "past_key_values": cache}) == expected
    assert backbone_stage((), {"inputs_embeds": None}) == "backbone_prefill"
