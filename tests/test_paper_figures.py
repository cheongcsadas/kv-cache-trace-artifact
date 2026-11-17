"""Verify Figure 4 and Table 6 geometry using explicit plotting fixtures."""
import copy
import csv

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.plotting import (SYSTEMS, FIGURE4_ATTACKS, FIGURE4_VARIANTS, FIGURE4_LABELS,
                                  TABLE6_LEVELS, ablation_figure, sensitivity_rows,
                                  sensitivity_figure, plot_summary)


def ablation_results():
    return [dict(system=system, attack=attack, ablation=variant, setting="templated_short",
                 metrics={"sr_asr_percent" if attack in {"MSA", "SSA"} else "ai_asr_percent":
                          10 + 20*attack_index + 3*system_index + variant_index})
            for system_index, system in enumerate(SYSTEMS)
            for attack_index, attack in enumerate(FIGURE4_ATTACKS)
            for variant_index, variant in enumerate(FIGURE4_VARIANTS)]


def sensitivity_results():
    return [dict(system=system, attack="MAA", factor=factor, value=value,
                 metrics=dict(ai_asr_percent=61.25+10*system_index+level_index))
            for system_index, system in enumerate(SYSTEMS)
            for factor, levels in TABLE6_LEVELS.items()
            for level_index, value in enumerate(levels)]


def test_figure4_matches_paper_panel_order_axes_and_group_order():
    fig = ablation_figure(ablation_results())
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 4
        bounds = [ax.get_position().bounds for ax in fig.axes]
        np.testing.assert_allclose([box[1:] for box in bounds], [bounds[0][1:]]*4, atol=1e-10)
        assert [text.get_text() for text in fig.legends[0].texts] == list(FIGURE4_LABELS)
        for panel, (attack, ax) in enumerate(zip(FIGURE4_ATTACKS, fig.axes)):
            assert ax.texts[0].get_text() == f"({chr(97+panel)}) {attack}"
            assert ax.get_ylim() == (0, 100)
            assert list(ax.get_yticks()) == [0, 20, 40, 60, 80, 100]
            assert [tick.get_text() for tick in ax.get_xticklabels()] == [
                "Quest-Llama", "InfiniGen\nMistral", "RetrievalAttention\nQwen"]
            assert len(ax.containers) == 5
            for variant_index, bars in enumerate(ax.containers):
                assert [bar.get_height() for bar in bars] == [10+20*panel+3*system_index+variant_index for system_index in range(3)]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_setting"])
def test_figure4_rejects_incomplete_or_nonpaper_conditions(kind):
    rows = ablation_results()
    if kind == "missing": rows.pop()
    if kind == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    if kind == "wrong_setting": rows[0]["setting"] = "templated_long"
    with pytest.raises(ValueError):
        ablation_figure(rows)


def test_table6_values_order_and_trend_ticks():
    rows = sensitivity_rows(sensitivity_results())
    assert [row["setting"] for row in rows] == ["20%", "40%", "60%", "8 tokens", "16 tokens", "32 tokens", "1K", "2K", "3K", "4K"]
    assert rows[0]["quest"] == 61.25 and rows[0]["retrievalattention"] == 81.25
    fig = sensitivity_figure(rows)
    try:
        for ax, ticks, labels in zip(fig.axes, ([20, 40, 60], [8, 16, 32], [1000, 2000, 3000, 4000]),
                                    (["20%", "40%", "60%"], ["8", "16", "32"], ["1K", "2K", "3K", "4K"])):
            assert list(ax.get_xticks()) == ticks
            assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
            assert list(ax.get_yticks()) == [0, 20, 40, 60, 80, 100]
            assert ax.get_ylim() == (0, 100)
            assert len(ax.lines) == 3
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_level", "wrong_attack"])
def test_table6_requires_all_and_only_the_30_paper_conditions(kind):
    rows = sensitivity_results()
    if kind == "missing": rows.pop()
    if kind == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    if kind == "wrong_level": rows[0]["value"] = .3
    if kind == "wrong_attack": rows[0]["attack"] = "SAA"
    with pytest.raises(ValueError):
        sensitivity_rows(rows)


def test_computed_table6_export_retains_numeric_source_values(tmp_path):
    output = tmp_path/"table6"
    names = plot_summary(dict(experiment="e09_cache_sensitivity", protocol="pilot", runs=sensitivity_results()), output)
    assert len(names) == 6
    with (output/"table6_cache_sensitivity.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert float(rows[0]["quest"]) == 61.25
    assert rows[0]["setting"] == "20%"
    for name in names:
        assert (output/name).stat().st_size > 100
