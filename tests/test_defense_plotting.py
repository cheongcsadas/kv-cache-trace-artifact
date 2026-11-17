"""Figure 7 axes and Table 8 inputs checked with explicit numerical fixtures."""
from copy import deepcopy

import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from experiments.chapter6.plotting import figure7, table8
from experiments.chapter6.specification import SYSTEMS, DEFENSES


def fixture_latencies():
    return [dict(system=system, defense=defense, mean_seconds=4.25+system_index*.15+defense_index*.4)
            for system_index, system in enumerate(SYSTEMS)
            for defense_index, defense in enumerate(DEFENSES)]


def fixture_accuracies():
    modes = (("none", "none"), ("fixed_budget", "aware"), ("fixed_budget", "oblivious"),
             ("dynamic_timing", "aware"), ("dynamic_timing", "oblivious"))
    return [dict(system=system, defense=defense, attacker_setting=mode, attack="MAA",
                 metrics=dict(ai_asr_percent=40+index))
            for system in SYSTEMS for index, (defense, mode) in enumerate(modes)]


def test_figure7_preserves_paper_axes_system_and_defense_order(tmp_path, monkeypatch):
    observed = []
    original = Figure.savefig
    def save(figure, *args, **kwargs):
        observed.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", save)
    figure7(fixture_latencies(), tmp_path/"figure")
    ax = observed[0].axes[0]
    assert ax.get_ylim() == (4, 7)
    assert list(ax.get_yticks()) == [4, 5, 6, 7]
    assert ax.get_ylabel() == "Inference Latency (s)"
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "Quest-Llama", "InfiniGen\nMistral", "RetrievalAttention\nQwen"]
    assert [label.get_text() for label in ax.get_legend().texts] == ["No Defense", "Fixed-Budget", "Dynamic Timing"]
    for index, container in enumerate(ax.containers):
        assert [patch.get_height() for patch in container] == pytest.approx(
            [4.25+system_index*.15+index*.4 for system_index in range(3)])
        assert {patch.get_hatch() for patch in container} == {("||", "--", "++")[index]}
    assert all(line.get_visible() for line in ax.get_ygridlines())
    assert all(spine.get_visible() for spine in ax.spines.values())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "below_axis", "above_axis", "none", "boolean"])
def test_figure7_does_not_replace_or_silently_clip_invalid_input(tmp_path, kind):
    records = fixture_latencies()
    if kind == "missing": records.pop()
    if kind == "duplicate": records[-1] = deepcopy(records[0])
    if kind == "below_axis": records[0]["mean_seconds"] = 3.2
    if kind == "above_axis": records[0]["mean_seconds"] = 7.1
    if kind == "none": records[0]["mean_seconds"] = None
    if kind == "boolean": records[0]["mean_seconds"] = True
    with pytest.raises(ValueError):
        figure7(records, tmp_path/"invalid")


@pytest.mark.parametrize("kind", ["attack", "boolean"])
def test_table8_rejects_non_maa_or_boolean_accuracy(tmp_path, kind):
    records = fixture_accuracies()
    if kind == "attack": records[0]["attack"] = "SAA"
    else: records[0]["metrics"]["ai_asr_percent"] = True
    with pytest.raises(ValueError):
        table8(records, tmp_path/"invalid.csv")
