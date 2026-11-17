"""Independent Chapter 6 experiment commands."""
import argparse
from pathlib import Path
import yaml

from experiments.chapter5.common import read_json, write_json, write_summary
from .specification import EXPERIMENTS, specification
from .evaluate import accuracy, latency


def execute(experiment, manifest_path, output, *, pilot=False, allow_synthetic=False, plot=False, auto_y=False):
    if allow_synthetic and not pilot:
        raise ValueError("Synthetic fixtures require explicit pilot mode")
    expected = specification(experiment)
    spec_path = Path(__file__).resolve().parents[1]/experiment/"parameters.yaml"
    if yaml.safe_load(spec_path.read_text(encoding="utf-8")) != expected:
        raise ValueError("Chapter 6 parameters differ from the executed paper conditions")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Accuracy creates one independent fitted forest per attacker condition.
    output.mkdir(parents=True)
    if experiment == EXPERIMENTS[0]:
        records = accuracy(manifest, manifest_path.parent, output, pilot=pilot, allow_synthetic=allow_synthetic)
        from .plotting import table8
        table8(records, output/"table8.csv")
    else:
        records = latency(manifest, manifest_path.parent, pilot=pilot, allow_synthetic=allow_synthetic)
    write_json(output/"parameters.json", expected)
    result = write_summary(output, experiment, records, "pilot" if pilot else "paper")
    if plot and experiment == EXPERIMENTS[1]:
        from .plotting import figure7
        figure7(records, output/"figure7", auto_y=auto_y)
    return result


def main(experiment):
    parser = argparse.ArgumentParser(description=f"Execute {experiment} using actual native KV defenses")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Evaluate captured Chapter 6 conditions; e14 can reuse e13 captures")
    source.add_argument("--native-manifest", help="Native project with the three configured systems and prepared domain datasets")
    parser.add_argument("--capture-output", help="Native observation directory, default OUTPUT-native")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="Only for explicitly marked pilot fixtures")
    parser.add_argument("--plot", action="store_true", help="Render measured Figure 7 latency")
    parser.add_argument("--auto-y", action="store_true", help="Expand Figure 7 y-axis when measurements exceed the paper's 4-7 s range")
    args = parser.parse_args()
    manifest = args.manifest
    if args.native_manifest:
        if args.allow_synthetic:
            parser.error("Native captures always use measured data")
        from .live import prepare_native
        manifest = prepare_native(experiment, args.native_manifest,
            args.capture_output or str(Path(args.output))+"-native", pilot=args.pilot)
    result = execute(experiment, manifest, args.output, pilot=args.pilot,
        allow_synthetic=args.allow_synthetic, plot=args.plot, auto_y=args.auto_y)
    print(f"{experiment}: {len(result['runs'])} result rows saved")
