"""Executable Table 8 and Figure 7 conditions."""
SYSTEMS = ("quest", "infinigen", "retrievalattention")
DEFENSES = ("none", "fixed_budget", "dynamic_timing")
EXPERIMENTS = ("e13_defense_accuracy", "e14_defense_latency")


def specification(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown Chapter 6 experiment")
    return dict(experiment=experiment, paper_section="6.3", systems=list(SYSTEMS),
        defenses=list(DEFENSES), setting="templated_short", channel="meta", attack="MAA",
        train_per_domain=3000, test_per_domain=1000, block_tokens=16, residency_fraction=.2,
        fixed_budget_pages=65, maximum_delay_ms=5,
        attacker_settings=["none", "aware", "oblivious"] if experiment == EXPERIMENTS[0] else [],
        output="table8" if experiment == EXPERIMENTS[0] else "figure7")
