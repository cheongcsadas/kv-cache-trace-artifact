"""Executable per-experiment parameter specifications for Chapter 5."""
import json
from experiments.paper_protocol import DOMAINS, MODEL_FAMILIES, SEMANTIC_CLASSES, SWEEP_LEVELS


def canonical_spec(experiment):
    systems=list(MODEL_FAMILIES)
    attacks=["MAA","MSA","SAA","SSA"]
    settings=["templated_short","templated_long","unstructured_short","unstructured_long","cross_template"]
    short=["templated_short"]
    inference=dict(train_per_domain=3000,test_per_domain=1000)
    specifications={
        "e01_metadata_recovery":dict(paper_section="5.2.1",systems=systems,sources=["logs","telemetry","shm","combined"],auxiliary_unique_source_records=1000),
        "e02_hardware_recovery":dict(paper_section="5.2.2",systems=systems,activities=["HIT","LOAD","EVICT"],auxiliary_unique_source_records=1000,reference_merge_gap_ms=50),
        "e03_attribute_inference":dict(paper_section="5.3.1",systems=systems,attacks=["MAA","SAA"],settings=settings,**inference),
        "e04_semantic_recovery":dict(paper_section="5.3.2",systems=["quest"],attacks=["MSA","SSA"],settings=short,semantic_classes=SEMANTIC_CLASSES,**inference),
        "e05_end_to_end":dict(paper_section="5.3.3",localization_failures_in_denominator=True,
            condition_groups=[dict(systems=systems,attacks=["MAA","SAA"],settings=settings),
                              dict(systems=["quest"],attacks=["MSA","SSA"],settings=short)],**inference),
        "e06_within_domain_intent":dict(paper_section="5.3.3",systems=["quest"],domains=list(DOMAINS),intent_labels_per_run=2),
        "e07_transfer_contribution":dict(paper_section="5.4",systems=["quest"],attacks=["SAA"],settings=short,
            ablations=["page_only","h2d_only","fusion"],train_per_domain=3000,test_total=1000),
        "e08_event_ablation":dict(paper_section="5.4",systems=systems,attacks=attacks,settings=short,
            ablations=["full","remove_load","remove_hit","remove_evict","remove_all"],**inference),
        "e09_cache_sensitivity":dict(paper_section="5.5.1",systems=systems,attacks=["MAA"],
            factors={k:list(SWEEP_LEVELS[k]) for k in ("residency_fraction","block_tokens","context_tokens")},
            fixed_selected_token_budget_for_block_sweep=True,**inference),
        "e10_concurrency":dict(paper_section="5.5.2",systems=systems,attacks=attacks,settings=short,
            concurrent_workloads=list(SWEEP_LEVELS["concurrent_workloads"]),arrival_mode="asynchronous",**inference),
        "e11_profiling_cost":dict(paper_section="5.6",systems=systems,channels=["meta","hw"],
            auxiliary_per_domain=3000,auxiliary_total=12000,timing_scope="observation_collection"),
        "e12_profiling_scale":dict(paper_section="5.6.1",systems=systems,attacks=attacks,
            settings=["templated_short","templated_long"],train_per_domain=list(SWEEP_LEVELS["training_per_domain"]),
            test_per_domain=1000,fixed_test_set=True),
    }
    if experiment not in specifications: raise ValueError("Unknown Chapter 5 experiment")
    return dict(experiment=experiment,**specifications[experiment],window_ms=10,max_windows=512,
                random_forest=dict(n_estimators=100,max_depth=10))


def validate_spec(parameters):
    """Reject declared YAML parameters that disagree with the executed method."""
    expected=canonical_spec(parameters.get("experiment"))
    for key,value in expected.items():
        if json.dumps(parameters.get(key),sort_keys=True) != json.dumps(value,sort_keys=True):
            raise ValueError(f"Experiment parameter {key} must equal {value!r}")
    if set(parameters)-set(expected): raise ValueError("Unknown experiment parameter keys")
    return parameters
