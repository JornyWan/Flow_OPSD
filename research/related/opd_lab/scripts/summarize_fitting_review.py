"""Unblind the frozen descriptive observations and audit v1 measurement coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

from opdlab.io import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--review", required=True)
    p.add_argument("--dataset", required=True, help="Frozen run data: map every scene clause explicitly")
    args = p.parse_args()
    root = Path(args.review)
    freeze = read_json(root / "observations_frozen.json")
    if freeze["observations_sha256"] != sha256(root / "observations.json"):
        raise ValueError("Observations changed after identity reveal")
    observations = read_json(root / "observations.json")
    keys = read_json(root / "key.json")
    records = {r["id"]: r for r in map(json.loads, Path(args.dataset).read_text().splitlines())}
    mapping = {(k["case"], label): {"id": k["id"], **value} for k in keys for label, value in k["mapping"].items()}
    if len(observations) != 40 or {(r["case"], r["label"]) for r in observations} != set(mapping):
        raise ValueError("Missing, duplicated or unknown observations")
    rows, by_clause, by_arm = [], {k: Counter() for k in ("text", "scene")}, {}
    for o in observations:
        for field in ("text_exact_no_extra", "scene_explicit", "full_visibility"):
            if o[field] is not None and type(o[field]) is not bool:
                raise ValueError("Annotations must be bool or null")
        key = mapping[o["case"], o["label"]]
        arm, verdict = key["arm"], key["row"]
        row = {**o, "id": key["id"], "arm": arm, "primary": verdict}
        constraints = records[key["id"]]["constraints"]
        text_indices = [i for i, c in enumerate(constraints) if c["kind"] == "ocr"]
        scene_indices = [i for i, c in enumerate(constraints) if c["kind"] == "geneval"]
        if len(constraints) != len(verdict["satisfied"]) or len(text_indices) != 1 or not scene_indices:
            raise ValueError("Unexpected constraint schema")
        for field, label, indices in (("text_exact_no_extra", "text", text_indices), ("scene_explicit", "scene", scene_indices)):
            manual, primary = o[field], all(verdict["satisfied"][i] for i in indices)
            name = "uncertain" if manual is None else ("agreement" if manual == primary else
                   ("primary_pass_visual_fail" if primary else "primary_fail_visual_pass"))
            by_clause[label][name] += 1
        if arm not in by_arm:
            by_arm[arm] = Counter()
        counts = by_arm[arm]
        counts["images"] += 1
        counts["v1_joint_pass"] += bool(verdict["joint_success"])
        for field in ("text_exact_no_extra", "scene_explicit", "full_visibility"):
            counts[f"{field}_{'uncertain' if o[field] is None else 'pass' if o[field] else 'fail'}"] += 1
        counts["v1_joint_pass_visibility_fail"] += bool(verdict["joint_success"] and o["full_visibility"] is False)
        counts["v1_joint_pass_text_fail"] += bool(verdict["joint_success"] and o["text_exact_no_extra"] is False)
        rows.append(row)
    summary = {"scope": "40 preselected training images; implementing-assistant observation, not independent human truth or a new efficacy metric.",
               "observation_freeze": freeze, "dataset_sha256": sha256(args.dataset), "images": 40,
               "by_clause": {k: dict(v) for k, v in by_clause.items()},
               "by_arm": {k: dict(v) for k, v in by_arm.items()}, "rows": rows}
    if (root / "summary.json").exists():
        raise FileExistsError("Review already unblinded and summarized")
    write_json(root / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
