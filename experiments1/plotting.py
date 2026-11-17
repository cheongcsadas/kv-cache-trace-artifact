"""Plot Chapter 5 experiment summaries using only computed results."""
import argparse
import csv
import json
import math
from pathlib import Path

SYSTEMS = ("quest", "infinigen", "retrievalattention")
ATTACKS = ("MAA", "MSA", "SAA", "SSA")
COLORS = ("#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214")
FIGURE4_ATTACKS = ("MAA", "SAA", "SSA", "MSA")
FIGURE4_VARIANTS = ("full", "remove_load", "remove_hit", "remove_evict", "remove_all")
FIGURE4_LABELS = ("All Features", "Remove LOAD", "Remove HIT", "Remove EVICT", "Remove All")
FIGURE4_COLORS = ("#a98b35", "#ab5f48", "#329c94", "#ef8a34", "#255f7e")
SYSTEM_LABELS = ("Quest-Llama", "InfiniGen\nMistral", "RetrievalAttention\nQwen")
TABLE6_LEVELS = {"residency_fraction": (.2, .4, .6), "block_tokens": (8, 16, 32),
                 "context_tokens": (1000, 2000, 3000, 4000)}
TABLE6_FACTORS = {"residency_fraction": "Residency budget", "block_tokens": "Block size",
                  "context_tokens": "Context length"}


def _score(record):
    metrics = record["metrics"]
    key = "sr_asr_percent" if record["attack"] in {"MSA", "SSA"} else "ai_asr_percent"
    value = float(metrics[key])
    if not math.isfinite(value) or not 0 <= value <= 100: raise ValueError("Invalid measured accuracy")
    return value


def _axes_style(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)


def _condition_index(records, fields, expected):
    index = {}
    for row in records:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError("Duplicate plot condition")
        _score(row)
        index[key] = row
    if set(index) != expected:
        raise ValueError("Plot needs exactly the complete paper condition grid")
    return index


def ablation_figure(records):
    """Figure 4: four attack panels, three systems, five grouped bars each."""
    import matplotlib.pyplot as plt
    import numpy as np
    expected = {(system, attack, variant) for system in SYSTEMS
                for attack in FIGURE4_ATTACKS for variant in FIGURE4_VARIANTS}
    index = _condition_index(records, ("system", "attack", "ablation"), expected)
    if any(row.get("setting", "templated_short") != "templated_short" for row in records):
        raise ValueError("Figure 4 uses short templated prompts")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.45))
    fig.subplots_adjust(left=.045, right=.995, bottom=.27, top=.84, wspace=.28)
    x, width = np.arange(len(SYSTEMS)), .15
    for panel, (ax, attack) in enumerate(zip(axes, FIGURE4_ATTACKS)):
        for position, variant in enumerate(FIGURE4_VARIANTS):
            values = [_score(index[system, attack, variant]) for system in SYSTEMS]
            ax.bar(x+(position-2)*width, values, width, label=FIGURE4_LABELS[position],
                   color=FIGURE4_COLORS[position], hatch=("////", "\\\\\\\\", "||||", "xxxx", "....")[position],
                   edgecolor="white", linewidth=.35)
        ax.set_xticks(x, SYSTEM_LABELS, fontsize=8)
        _axes_style(ax, "")
        ax.set_ylabel("Attack Success Rate (%)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(False)
        ax.text(.5, -.34, f"({chr(97+panel)}) {attack}", ha="center", transform=ax.transAxes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .99), fontsize=9, handlelength=1.4)
    return fig


def sensitivity_rows(records):
    """Table 6 rows in paper order, populated only from computed MAA results."""
    expected = {(system, factor, value) for system in SYSTEMS
                for factor, levels in TABLE6_LEVELS.items() for value in levels}
    index = _condition_index(records, ("system", "factor", "value"), expected)
    if any(row["attack"] != "MAA" for row in records):
        raise ValueError("Table 6 evaluates MAA only")
    rows = []
    for factor, levels in TABLE6_LEVELS.items():
        for value in levels:
            setting = (f"{value:.0%}" if factor == "residency_fraction" else
                       f"{value} tokens" if factor == "block_tokens" else f"{value//1000}K")
            rows.append(dict(factor=factor, setting=setting, value=value,
                             **{system: _score(index[system, factor, value]) for system in SYSTEMS}))
    return rows


def sensitivity_figure(rows):
    """An additional trend view of Table 6, with the exact evaluated x ticks."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (factor, levels) in zip(axes, TABLE6_LEVELS.items()):
        subset = [row for row in rows if row["factor"] == factor]
        ticks = [100*value for value in levels] if factor == "residency_fraction" else list(levels)
        labels = ([f"{value:g}%" for value in ticks] if factor == "residency_fraction" else
                  [str(value) for value in levels] if factor == "block_tokens" else
                  [f"{value//1000}K" for value in levels])
        for position, system in enumerate(SYSTEMS):
            ax.plot(ticks, [row[system] for row in subset], marker="o", label=SYSTEM_LABELS[position].replace("\n", "-"),
                    color=COLORS[position])
        ax.set_xticks(ticks, labels)
        _axes_style(ax, TABLE6_FACTORS[factor] + (" (tokens)" if factor in {"block_tokens", "context_tokens"} else ""))
        ax.set_ylabel("AI-ASR (%)")
    axes[0].legend(fontsize=8)
    return fig


def sensitivity_table(rows):
    """Render the paper's Table 6 layout with measured values, not copied scores."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.axis("off")
    body = []
    for row in rows:
        body.append([TABLE6_FACTORS[row["factor"]], row["setting"]] + [f"{row[system]:.1f}" for system in SYSTEMS])
    table = ax.table(cellText=body, colLabels=["Factor", "Setting", "Quest", "InfiniGen", "Retrieval"],
                     colWidths=[.27, .18, .17, .19, .19], cellLoc="center", bbox=[0, .02, 1, .88])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(.25)
        cell.set_edgecolor("#888888")
        cell.set_facecolor("#e9e9e9" if row == 0 or row % 2 == 0 else "white")
    ax.set_title("Table 6: Sensitivity of MAA to KV-cache configurations (AI-ASR, %)", fontsize=11, pad=4)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.88)
    return fig


def _write_sensitivity_table(rows, output):
    fields = ["factor", "setting", "value", *SYSTEMS]
    with (output/"table6_cache_sensitivity.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Factor | Setting | Quest | InfiniGen | Retrieval |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([TABLE6_FACTORS[row["factor"]], row["setting"]] +
                                        [f"{row[system]:.1f}" for system in SYSTEMS]) + " |")
    (output/"table6_cache_sensitivity.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    table = sensitivity_table(rows)
    table.savefig(output/"table6_cache_sensitivity.png", dpi=300)
    table.savefig(output/"table6_cache_sensitivity.pdf", metadata={"Title":"Table 6", "Author":""})
    import matplotlib.pyplot as plt
    plt.close(table)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = summary["runs"]
    experiment = summary["experiment"]
    if not records: raise ValueError("No experiment results")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42})
    if experiment == "e08_event_ablation":
        fig = ablation_figure(records)
    elif experiment in {"e10_concurrency", "e12_profiling_scale"}:
        field = "concurrent_workloads" if experiment == "e10_concurrency" else "train_per_domain"
        settings = (None,) if field == "concurrent_workloads" else ("templated_short", "templated_long")
        levels = [1,4,8,16] if field == "concurrent_workloads" else [400,800,1600,3200]
        fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax, system in zip(axes,SYSTEMS):
            for index, attack in enumerate(ATTACKS):
                for setting in settings:
                    subset=[r for r in records if r["system"] == system and r["attack"] == attack
                            and (setting is None or r["setting"] == setting)]
                    if sorted(r[field] for r in subset) != levels:
                        raise ValueError("Plot needs one result at every declared paper sweep level")
                    subset.sort(key=lambda r:r[field])
                    label=attack+(" TL" if setting == "templated_long" else " TS" if setting else "")
                    ax.plot(levels,[_score(r) for r in subset],label=label,color=COLORS[index],
                            linestyle="--" if setting == "templated_long" else "-",marker="o",markersize=4)
            ax.set_title(system)
            ax.set_xticks(levels)
            _axes_style(ax,"Asynchronous workloads" if field == "concurrent_workloads" else "Training prompts per domain")
        axes[0].legend(fontsize=8,ncol=2)
    elif experiment == "e09_cache_sensitivity":
        rows = sensitivity_rows(records)
        fig = sensitivity_figure(rows)
        _write_sensitivity_table(rows, output)
    else:
        raise ValueError("Plotting supports event ablation, sensitivity, concurrency and profiling scale summaries")
    stem=experiment.removeprefix("e")
    fig.savefig(output/(stem+".png"),dpi=300)
    fig.savefig(output/(stem+".pdf"),metadata={"Title":experiment,"Author":""})
    plt.close(fig)
    return sorted(p.name for p in output.iterdir())


def main(expected_experiment=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    summary=json.loads(Path(a.input).read_text(encoding="utf-8"))
    if expected_experiment is not None and summary.get("experiment") != expected_experiment:
        p.error(f"This plot entry point requires {expected_experiment} results")
    for name in plot_summary(summary,a.output): print(name)


if __name__ == "__main__": main()
