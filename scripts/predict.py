"""Predict from observations only, using a trusted locally trained checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from attacks.hardware import encode_hardware, LocalizationError
from attacks.metadata import encode_metadata
from attacks.ablation import encode_ablation
from experiments.paper_protocol import validate_hardware_observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    bundle = joblib.load(a.model)
    observation = json.loads(Path(a.observation).read_text(encoding="utf-8"))
    config = bundle["config"]
    params = dict(window_ms=config["observation"]["window_ms"], max_windows=config["observation"]["max_windows"])
    try:
        hardware = bundle["attack"] in {"SAA", "SSA"}
        if hardware:
            validate_hardware_observation(observation, config)
        if bundle.get("ablation", "full") != "full":
            options = {} if hardware else dict(spatial_bins=config["implementation_choices"]["metadata_spatial_bins"])
            encoded = encode_ablation(observation, "hw" if hardware else "meta", bundle["ablation"],
                                      bundle["calibration"], **params, **options)
        elif hardware:
            encoded = encode_hardware(observation, bundle["calibration"], **params)
        else:
            encoded = encode_metadata(observation,
                spatial_bins=config["implementation_choices"]["metadata_spatial_bins"], **params)
        prediction = bundle["model"].predict(encoded.flatten()[None, :])[0]
        result = dict(attack=bundle["attack"], prediction=prediction, status="predicted")
    except LocalizationError as exc:
        result = dict(attack=bundle["attack"], prediction=None, status="localization_failure", reason=str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)


if __name__ == "__main__": main()
