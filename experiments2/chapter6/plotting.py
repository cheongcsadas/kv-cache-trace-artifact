"""Render Table 8 and Figure 7 exclusively from computed defense results."""
import csv
import math
from pathlib import Path
from .specification import SYSTEMS, DEFENSES


def table8(records, destination):
    if any(record.get("attack") != "MAA" for record in records):
        raise ValueError("Table 8 evaluates MAA only")
    indexed = {(r["system"], r["defense"], r["attacker_setting"]): r for r in records}
    columns = (("none", "none"), ("fixed_budget", "aware"), ("fixed_budget", "oblivious"),
               ("dynamic_timing", "aware"), ("dynamic_timing", "oblivious"))
    expected = {(system, defense, mode) for system in SYSTEMS for defense, mode in columns}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Table 8 requires all 15 distinct MAA conditions")
    with Path(destination).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["system", "no_defense", "fixed_aware", "fixed_oblivious", "timing_aware", "timing_oblivious"])
        for system in SYSTEMS:
            values = [indexed[system, defense, mode]["metrics"]["ai_asr_percent"] for defense, mode in columns]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100 for v in values):
                raise ValueError("Table 8 needs finite computed percentages")
            writer.writerow([system, *values])


def figure7(records, output, *, auto_y=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    indexed = {(r["system"], r["defense"]): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(s, d) for s in SYSTEMS for d in DEFENSES}:
        raise ValueError("Figure 7 requires three systems and three conditions")
    values = [r["mean_seconds"] for r in records]
    if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Figure 7 requires paired completed inference measurements")
    if not auto_y and any(v < 4 or v > 7 for v in values):
        raise ValueError("Measured latency is outside the paper's 4-7 s axis; select --auto-y to show every value")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    x, width = np.arange(3), .24
    labels = ("No Defense", "Fixed-Budget", "Dynamic Timing")
    for i, (defense, color, hatch, label) in enumerate(zip(DEFENSES,
            ("#f5bc82", "#99cf9c", "#bfadd0"), ("||", "--", "++"), labels)):
        ax.bar(x+(i-1)*width, [indexed[s, defense]["mean_seconds"] for s in SYSTEMS], width,
               label=label, color=color, edgecolor="black", linewidth=.6, hatch=hatch)
    ax.set_xticks(x, ["Quest-Llama", "InfiniGen\nMistral", "RetrievalAttention\nQwen"])
    ax.set_ylabel("Inference Latency (s)")
    if auto_y:
        upper = max(7, math.ceil(max(values)))
        lower = min(4, max(0, math.floor(min(values))-1))
        ax.set_ylim(lower, upper)
        ax.set_yticks(range(lower, upper+1))
    else:
        ax.set_ylim(4, 7)
        ax.set_yticks([4, 5, 6, 7])
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=3, frameon=False, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="black", alpha=.65, linewidth=.6, linestyle=(0, (1, 3)))
    fig.savefig(output/"figure7.png", dpi=300)
    fig.savefig(output/"figure7.pdf", metadata={"Title": "Inference latency under KV-loading defenses", "Author": ""})
    plt.close(fig)
    return ["figure7.png", "figure7.pdf"]
