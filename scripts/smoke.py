"""Exercise both observation paths and all classifiers on explicitly synthetic inputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run import run


def write_fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    domains = ["medical", "financial", "legal", "code"]
    for channel in ("meta", "hw"):
        for split, count in (("train", 12), ("test", 4)):
            rows = []
            for d, domain in enumerate(domains):
                for j in range(count):
                    sid = f"synthetic-{split}-{domain}-{j}"
                    if channel == "meta":
                        obs = dict(request_id=sid, start_ms=0, end_ms=100, snapshots=[], events=[
                            dict(time_ms=float(1+i*2), layer=0, block_id=d*32+i%8,
                                 op=("LOAD" if i < 8 else "HIT"), kind="management")
                            for i in range(12+d*3)])
                    else:
                        obs = dict(submission_ms=0, capacity_slots=2, ambiguous=False,
                            h2d=[dict(time_ms=float(t), latency_us=float(230+d*20 if t < 5+d*2 else 150))
                                 for t in range(100)],
                            compute=[dict(time_ms=float(t)+.1, latency_us=float(40+d*10)) for t in range(100)],
                            page=[dict(time_ms=float(t)+.2, latency_us=float(80+d*15)) for t in range(100)])
                    rows.append(dict(sample_id=sid, source_id=sid, provenance="synthetic", domain=domain,
                        labels=dict(domain=domain, intent=f"intent-{j%3}", attribute=f"attribute-{j%4}",
                                    keyword=f"keyword-{j%4}"), observation=obs))
            (root/f"{channel}-{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    (root/"calibration.json").write_text(json.dumps(dict(baseline_h2d_us=150,
        compute_threshold_us=45, page_threshold_us=100)), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/smoke")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    fixture = output/"fixture"
    write_fixture(fixture)
    config = Path(__file__).resolve().parents[1]/"configs/quest.yaml"
    for attack in ("MAA", "MSA", "SAA", "SSA"):
        channel = "hw" if attack in ("SAA", "SSA") else "meta"
        report = run(config, fixture/f"{channel}-train.jsonl", fixture/f"{channel}-test.jsonl",
                     output/attack, attack, fixture/"calibration.json", allow_synthetic=True)
        print(f"{attack}: synthetic pipeline completed, {report['test_count']} attempts")
    print("Synthetic checks only. No paper results or GPU measurements were reproduced.")


if __name__ == "__main__": main()
