"""Freeze phase11 checkpoints, paired noise seeds, anchors and development split."""
import argparse
import json
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from opdlab.transfer_eval import make_pairs


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    old, campaign = ROOT / "runs/le10_fit_001", ROOT / "experiments/local_expert_v11"
    previous, prior_result = read_json(old / "probe_plan.json"), read_json(old / "result.json")
    config = Experiment.load(old / "config.json")
    train, validation = load_records(config.dataset, "train"), load_records(config.dataset, "validation")
    pairs = make_pairs(previous["selected"], train, validation)
    apple = next(s for s in previous["selected"] if s["case"] == 19)
    endpoints = [json.loads(s) for s in (old / "endpoints.jsonl").read_text().splitlines()]
    references = [json.loads(s) for s in (old / "references.jsonl").read_text().splitlines()]
    models = {}
    inputs = [old / n for n in ("probe_plan.json", "result.json", "endpoints.jsonl", "references.jsonl")]
    inputs += [ROOT / "experiments/local_expert_v10" / n for n in ("NEXT.md", "STATUS.md", "results.json", "observations.json", "observations_frozen.json")]
    for name, arm, step in (("joint128", None, None), ("sparse32", "sparse", 32), ("sparse320", "sparse", 320), ("dense32", "dense", 32)):
        weights = Path(previous["student"]) if arm is None else old / "checkpoints" / arm / f"step{step:03d}.pt"
        row = next(r for r in references if r["id"] == apple["id"] and r["mode"] == "base") if arm is None else next(
            r for r in endpoints if (r["arm"], r["optimizer_step"], r["mode"], r["id"]) == (arm, step, "full", apple["id"]))
        if arm is not None and sha256(weights) != prior_result["checkpoint_hashes"][str(weights.relative_to(old))]:
            raise ValueError("Checkpoint changed")
        image = old / row["image"]
        if sha256(image) != row["image_sha256"]:
            raise ValueError("Anchor image changed")
        inputs += [weights, image]
        models[name] = {"weights": str(weights), "kind": "raw_tensor_state",
                        "anchor": {"id": apple["id"], "seed": apple["seed"], "image_sha256": row["image_sha256"]}}
    run = prepare(config, ROOT / f"runs/le11_transfer_{args.attempt}", read_json(args.site))
    write_json(run / "probe_plan.json", {"version": 1, "selected": previous["selected"], "pairs": pairs, "models": models})
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.append(run / name)
    inputs.append(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md"), "transfer_images": 168, "additional_anchor_images": 4})
    print(run)


if __name__ == "__main__":
    main()
