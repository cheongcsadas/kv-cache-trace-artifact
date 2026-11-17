"""Command-line dispatch for individually executable Chapter 5 experiments."""
import argparse
import itertools
from pathlib import Path
import re
import yaml
from .common import read_json, write_summary, write_json
from .specification import validate_spec


def expected_grid(experiment, parameters):
    if experiment == "e05_end_to_end":
        return {(s,a,t) for group in parameters["condition_groups"]
                for s,a,t in itertools.product(group["systems"],group["attacks"],group["settings"])}
    systems=parameters.get("systems",[])
    attacks=parameters.get("attacks",[])
    settings=parameters.get("settings",[])
    if experiment in {"e01_metadata_recovery","e02_hardware_recovery"}:
        return {("system", s) for s in systems}
    if experiment == "e11_profiling_cost":
        return {(s,c) for s,c in itertools.product(systems,parameters["channels"])}
    if experiment == "e06_within_domain_intent":
        return {("quest",d) for d in parameters["domains"]}
    if experiment == "e09_cache_sensitivity":
        return {(s,f,v) for s in systems for f,values in parameters["factors"].items() for v in values}
    if experiment == "e10_concurrency":
        return {(s,a,n) for s,a,n in itertools.product(systems,attacks,parameters["concurrent_workloads"])}
    if experiment == "e12_profiling_scale":
        return {(s,a,t,n) for s,a,t,n in itertools.product(systems,attacks,settings,parameters["train_per_domain"])}
    if experiment in {"e07_transfer_contribution","e08_event_ablation"}:
        return {(s,a,b) for s,a,b in itertools.product(systems,attacks,parameters["ablations"])}
    return {(s,a,t) for s,a,t in itertools.product(systems,attacks,settings)}


def check_complete(experiment, manifest, parameters, base):
    from experiments.config import load_config
    actual=set()
    for run in manifest["runs"]:
        system=run.get("system")
        if system is None: system=load_config(base/run["config"])["system"]
        if experiment in {"e01_metadata_recovery","e02_hardware_recovery"}: key=("system",system)
        elif experiment == "e11_profiling_cost": key=(system,run["channel"])
        elif experiment == "e06_within_domain_intent": key=(system,run["domain"])
        elif experiment == "e09_cache_sensitivity": key=(system,run["factor"],run["value"])
        elif experiment == "e10_concurrency": key=(system,run["attack"],run["concurrent_workloads"])
        elif experiment == "e12_profiling_scale": key=(system,run["attack"],run["setting"],run["train_per_domain"])
        elif experiment in {"e07_transfer_contribution","e08_event_ablation"}: key=(system,run["attack"],run["ablation"])
        else: key=(system,run["attack"],run["setting"])
        if key in actual: raise ValueError(f"Repeated experiment condition: {key}")
        actual.add(key)
    expected=expected_grid(experiment,parameters)
    if actual != expected:
        raise ValueError(f"Full experiment grid differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def execute(experiment, manifest_path, output, *, pilot=False, allow_synthetic=False, complete=False):
    manifest_path=Path(manifest_path).resolve()
    manifest=read_json(manifest_path)
    runs=manifest.get("runs",[])
    if not isinstance(runs,list) or not runs: raise ValueError("Manifest requires a nonempty runs array")
    ids=[r.get("id") for r in runs]
    if any(not isinstance(i,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Run IDs must be unique safe filename components")
    if allow_synthetic and not pilot: raise ValueError("Synthetic inputs require explicit --pilot mode")
    spec_path=Path(__file__).resolve().parents[1]/experiment/"parameters.yaml"
    parameters=yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    validate_spec(parameters)
    if parameters["experiment"] != experiment: raise ValueError("Experiment specification mismatch")
    if complete: check_complete(experiment,manifest,parameters,manifest_path.parent)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if experiment in {"e01_metadata_recovery","e02_hardware_recovery"}:
        from .recovery import execute_recovery
        records=execute_recovery(manifest,manifest_path.parent,experiment == "e02_hardware_recovery",pilot,allow_synthetic)
    elif experiment == "e11_profiling_cost":
        from .cost import execute_cost
        records=execute_cost(manifest,manifest_path.parent,pilot,allow_synthetic)
    elif experiment == "e06_within_domain_intent":
        from .inference import execute_intent
        records=execute_intent(manifest,manifest_path.parent,output,pilot,allow_synthetic)
    else:
        from .inference import execute_inference
        records=execute_inference(experiment,manifest,manifest_path.parent,output,pilot,allow_synthetic)
    write_json(output/"parameters.json",parameters)
    return write_summary(output,experiment,records,"pilot" if pilot else "paper")


def main(experiment):
    parser=argparse.ArgumentParser(description=f"Execute {experiment} from local observation records")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest",help="Evaluate previously collected observation records")
    source.add_argument("--native-manifest",help="Load native systems and collect actual rendered prompts before evaluation")
    parser.add_argument("--capture-output",help="Native observations directory; defaults to OUTPUT-native")
    parser.add_argument("--output",required=True)
    parser.add_argument("--complete",action="store_true",help="Require every condition in parameters.yaml")
    parser.add_argument("--plot",action="store_true",help="Export paper-style figures/tables for e08/e09/e10/e12")
    parser.add_argument("--pilot",action="store_true",help="Use a smaller dataset with the same method parameters")
    parser.add_argument("--allow-synthetic",action="store_true",help="Permit marked fixture input in pilot mode")
    args=parser.parse_args()
    if args.plot and experiment not in {"e08_event_ablation","e09_cache_sensitivity","e10_concurrency","e12_profiling_scale"}:
        parser.error("--plot is supported by e08, e09, e10 and e12")
    manifest=args.manifest
    if args.native_manifest:
        if args.allow_synthetic:
            parser.error("Native collection produces measured data; --allow-synthetic is incompatible")
        from .live import prepare_native
        manifest=prepare_native(experiment,args.native_manifest,
                                args.capture_output or str(Path(args.output))+"-native",pilot=args.pilot)
    result=execute(experiment,manifest,args.output,pilot=args.pilot,
                   allow_synthetic=args.allow_synthetic,complete=args.complete)
    if args.plot:
        from experiments.plotting import plot_summary
        plot_summary(result,Path(args.output)/"figures")
    print(f"{experiment}: {len(result['runs'])} result rows saved")
