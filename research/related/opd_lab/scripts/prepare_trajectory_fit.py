"""Freeze phase10 sparse/dense supervision and its common weight warm start."""
import argparse
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--attempt", default="001")
    args = parser.parse_args()
    old = ROOT / "runs/le9_requery_001"
    campaign = ROOT / "experiments/local_expert_v10"
    previous, results = read_json(old / "probe_plan.json"), read_json(ROOT / "experiments/local_expert_v9/results.json")
    joined = {(e["row"]["id"], e["row"]["candidate"]): e for e in results["joined_review"]}
    inputs = [old / name for name in ("result.json", "probe_plan.json", "candidates.jsonl")]
    inputs += [ROOT / "experiments/local_expert_v9" / name for name in ("results.json", "NEXT.md", "observations.json", "observations_frozen.json")]
    inputs += [ROOT / "runs/le9_review_001" / f"{name}.json" for name in ("key", "inherited", "observations", "observations_frozen")]
    inputs += [Path(previous[name]) for name in ("student", "frozen_queries")]
    selected = []
    for q in previous["queries"]:
        candidate = "geneval@1" if q["case"] == 19 else "base"
        base, target = joined[(q["id"], "base")], joined[(q["id"], candidate)]
        if candidate != "base" and not target["comparison"]["accepted_encoded_repair"]:
            raise ValueError("The selected fresh target is no longer a protected repair")
        branch = old / f"branches-{q['case']:02d}.pt"
        if sha256(branch) != results["gpu_result"]["branches_sha256"][branch.name]:
            raise ValueError("Saved branch changed")
        selected.append({**q, "candidate": candidate, "branch_file": str(branch),
                         "base_image_sha256": base["row"]["image_sha256"],
                         "target_image_sha256": target["row"]["image_sha256"],
                         "base_observation": base["observation"], "target_observation": target["observation"]})
        inputs.append(branch)
        for e in (base, target):
            path = old / e["row"]["image"]
            if sha256(path) != e["row"]["image_sha256"]:
                raise ValueError("Selected image changed")
            inputs.append(path)
    if [s["case"] for s in selected] != [9, 11, 12, 17, 19]:
        raise ValueError("Wrong selected-case list")
    config = Experiment.load(old / "config.json")
    run = prepare(config, ROOT / f"runs/le10_fit_{args.attempt}", read_json(args.site))
    write_json(run / "probe_plan.json", {"version": 1, "selected": selected, "student": previous["student"],
               "frozen_queries": previous["frozen_queries"], "optimizer_policy": "common reset, weight warm start",
               "arms": {"sparse": {"queries_per_update": 5, "updates": 320, "audits": [0, 32, 320]},
                        "dense": {"queries_per_update": 50, "updates": 32, "audits": [0, 32]}}})
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.append(run / name)
    inputs.append(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "source_phase9": str(old),
               "plan_sha256": sha256(run / "probe_plan.json"), "protocol_sha256": sha256(run / "PROTOCOL.md")})
    print(run)


if __name__ == "__main__":
    main()
