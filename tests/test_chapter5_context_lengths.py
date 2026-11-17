"""Input-length checks use each domain's actual native-tokenizer mean."""
import pytest

from experiments.chapter5.live import _validate_context_lengths


CONFIG = {"model": {"name": "local-tokenizer"}, "inference": {"generation_tokens": 4}}
DOMAINS = ("medical", "financial", "legal", "code")


class Service:
    def __init__(self):
        self.calls = []

    def describe(self, prompt, generation_tokens):
        self.calls.append((prompt, generation_tokens))
        return {"token_count": int(prompt)}


def inputs(target=1000, *, declared=True):
    return {split: [dict(domain=domain, prompt=str(target),
                         **({"target_tokens": target, "token_count": target} if declared else {}))
                    for domain in DOMAINS] for split in ("train", "test")}


def test_domain_length_errors_cannot_cancel_in_a_global_mean():
    rows = inputs(declared=False)
    rows["train"][0]["prompt"] = "600"
    rows["train"][1]["prompt"] = "1400"
    # The previous aggregate check accepted this exactly-1000 overall mean.
    assert sum(int(row["prompt"]) for row in rows["train"]) / 4 == 1000
    with pytest.raises(ValueError, match="train/medical.*600.0"):
        _validate_context_lengths({}, {"setting": "unstructured_short"}, rows, CONFIG, Service())


@pytest.mark.parametrize("setting,target", [("templated_short", 1000), ("templated_long", 3000),
    ("unstructured_short", 1000), ("unstructured_long", 3000), ("cross_template", 1000)])
def test_every_domain_and_split_is_measured_for_all_five_settings(setting, target):
    service = Service()
    summary = _validate_context_lengths({}, {"setting": setting}, inputs(target), CONFIG, service)
    assert len(service.calls) == 8 and all(tokens == 4 for _, tokens in service.calls)
    for split in ("train", "test"):
        assert set(summary["splits"][split]) == set(DOMAINS)
        assert all(row["mean_tokens"] == target and row["count"] == 1
                   for row in summary["splits"][split].values())
    assert summary["token_count_scope"] == "rendered_prompt_with_tokenizer_special_tokens"


@pytest.mark.parametrize("project,condition,declared,origin", [
    ({}, {"setting": "cross_template", "target_tokens": 1800}, False, "declared_condition"),
    ({"cross_template_target_tokens": 1800}, {"setting": "cross_template"}, False, "declared_project"),
    ({}, {"setting": "cross_template"}, True, "prepared_rows"),
])
def test_cross_template_uses_its_declared_target(project, condition, declared, origin):
    summary = _validate_context_lengths(project, condition, inputs(1800, declared=declared), CONFIG, Service())
    assert summary["target_tokens"] == 1800 and summary["target_source"] == origin


def test_undeclared_cross_template_target_is_an_artifact_choice():
    summary = _validate_context_lengths({}, {"setting": "cross_template"},
                                        inputs(declared=False), CONFIG, Service())
    assert summary["target_tokens"] == 1000 and summary["target_source"] == "artifact_default"
    rows = inputs(declared=False)
    rows["test"][3]["prompt"] = "700"
    with pytest.raises(ValueError, match="test/code.*700.0"):
        _validate_context_lengths({}, {"setting": "cross_template"}, rows, CONFIG, Service())


def test_conflicting_prepared_targets_fail_before_describe():
    rows = inputs()
    rows["test"][2]["target_tokens"] = 3000
    service = Service()
    with pytest.raises(ValueError, match="target_tokens differs"):
        _validate_context_lengths({}, {"setting": "cross_template"}, rows, CONFIG, service)
    assert service.calls == []


def test_native_special_token_count_must_equal_prepared_count():
    rows = inputs()
    rows["train"][0]["token_count"] = 999
    with pytest.raises(ValueError, match="token_count differs from the native"):
        _validate_context_lengths({}, {"setting": "templated_short"}, rows, CONFIG, Service())


def test_context_sweep_uses_its_exact_target_instead_of_the_short_long_alias():
    condition = {"setting": "templated_short", "factor": "context_tokens", "value": 2000}
    summary = _validate_context_lengths({}, condition, inputs(2000), CONFIG, Service())
    assert summary["target_tokens"] == 2000 and summary["target_source"] == "paper_context_sweep"


@pytest.mark.parametrize("tolerance", [True, -0.1, 1.0, float("nan"), "0.1"])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="context_mean_tolerance_fraction"):
        _validate_context_lengths({"context_mean_tolerance_fraction": tolerance},
                                  {"setting": "templated_short"}, inputs(), CONFIG, Service())


def test_auxiliary_datasets_do_not_invent_a_paper_length():
    service = Service()
    assert _validate_context_lengths({}, {"setting": "auxiliary"}, inputs(), CONFIG, service) is None
    assert service.calls == []
