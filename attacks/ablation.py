"""Dependency-aware event-group ablations and probe-source comparisons.

Every ablated setting requires retraining, including its semantic domain model.
Original feature-group index lists were not published; these explicit groups are
the artifact's auditable definitions, not an assertion of identical original code.
"""
from .metadata import encode_metadata, OPS
from .hardware import encode_hardware

ABLATIONS = ("full", "remove_hit", "remove_load", "remove_evict", "remove_all",
             "h2d_only", "page_only", "fusion")
METADATA_ABLATIONS = ABLATIONS[:5]


def encode_ablation(observation, channel, ablation="full", calibration=None, **kwargs):
    """Return an EncodedTrace using only the selected feature dependencies.

    Metadata event removal precedes all counts, histograms, and time aggregates.
    Its origin and duration are recomputed from retained evidence, so removed
    event times cannot influence masks or offsets. Hardware comparisons retain a
    common H2D-localized request interval: they compare inference features under
    that conditional alignment, not independently localizing page-only attacks.
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    if channel == "meta":
        if ablation not in METADATA_ABLATIONS:
            raise ValueError("probe-source comparisons apply only to hardware")
        if ablation == "full":
            return encode_metadata(observation, **kwargs)
        retained = () if ablation == "remove_all" else tuple(op for op in OPS if op.lower() != ablation.removeprefix("remove_"))
        return encode_metadata(observation, retained_ops=retained, rebase_retained=True, **kwargs)
    if channel != "hw":
        raise ValueError("channel must be 'meta' or 'hw'")
    if calibration is None:
        raise ValueError("hardware ablations require frozen calibration")
    if "channels" in kwargs:
        raise ValueError("choose ablation or explicit channels, not both")
    channels = {
        "full": ("h2d", "compute", "page", "pressure"),
        "remove_hit": ("h2d", "page", "pressure"),
        "remove_load": ("compute", "page"),
        "remove_evict": ("h2d", "compute", "page"),
        "remove_all": (),
        "h2d_only": ("h2d",),
        "page_only": ("page",),
        "fusion": ("h2d", "page"),
    }[ablation]
    result = encode_hardware(observation, calibration, channels=channels, **kwargs)
    if ablation == "remove_load":
        # Candidate-HIT explicitly uses the absence of an H2D segment. Keeping it
        # after LOAD removal would retain a transformed transfer feature.
        result.values[:, result.feature_names.index("candidate_hit_fraction")] = 0
    if ablation == "remove_all":
        # Anonymous page signals and localization duration also carry management
        # information. A strict no-management control exposes none of them.
        result.values.fill(0)
        result.mask.fill(False)
    return result
