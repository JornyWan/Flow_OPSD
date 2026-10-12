"""Join frozen identity-hidden assistant observations with the unchanged automated scores."""
import argparse
from collections import Counter
from pathlib import Path

from opdlab.io import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--review", required=True)
    p.add_argument("--evaluation", required=True)
    args = p.parse_args()
    review, evaluation = Path(args.review).resolve(), Path(args.evaluation).resolve()
    if (review / "comparison.json").exists():
        raise FileExistsError("Review comparison already exists")
    source = review / "assistant_observations.json"
    observations = {r["case"]: r for r in read_json(source)["rows"]}
    key = read_json(review / "key.json")
    if set(observations) != {r["case"] for r in key} or len(key) != 16:
        raise ValueError("Incomplete predeclared review")
    secondary = {}
    for arm in next(iter(key))["mapping"].values():
        name = arm["arm"]
        secondary[name] = {(r["id"], r["replicate"]): r for r in read_json(evaluation / name / "secondary_ocr.json")["rows"]}
    rows, counts = [], {metric: Counter() for metric in ("primary_text", "secondary_text", "primary_scene", "primary_joint")}
    for case in key:
        obs = observations[case["case"]]
        for i, letter in enumerate("ABCDE"):
            item = case["mapping"][letter]
            arm, primary = item["arm"], item["row"]
            other = secondary[arm][primary["id"], primary["replicate"]]
            if primary["image_sha256"] != other["image_sha256"]:
                raise ValueError("Reader image identities differ")
            text, scene = obs["text_exact"][i], obs["scene_satisfied"][i]
            joint = False if False in (text, scene) else (True if text is True and scene is True else None)
            j = primary["constraint_ids"].index("text")
            row = {"case": case["case"], "letter": letter, "id": case["id"], "arm": arm,
                   "image": str(evaluation / arm / primary["image"]), "image_sha256": primary["image_sha256"],
                   "assistant_reading": obs["readings"][i], "assistant_text": text, "assistant_scene": scene,
                   "assistant_joint": joint, "notes": obs["notes"],
                   "primary_reading": primary["details"]["ocr"]["recognized_text"], "secondary_reading": other["text"],
                   "primary_text": primary["satisfied"][j], "secondary_text": other["satisfied"][j],
                   "primary_scene": all(v for index, v in enumerate(primary["satisfied"]) if index != j),
                   "primary_joint": primary["joint_success"]}
            for name, reference in (("primary_text", text), ("secondary_text", text),
                                    ("primary_scene", scene), ("primary_joint", joint)):
                if reference is None:
                    counts[name]["uncertain"] += 1
                else:
                    counts[name]["assessable"] += 1
                    counts[name]["agreement" if row[name] == reference else ("false_positive" if row[name] else "false_negative")] += 1
            rows.append(row)
    result = {"scope": "Preselected 16 prompts, one seed, all five methods; implementing-assistant observations are provisional, not independent human ground truth. Do not rank methods with unequal assessable subsets.",
              "observations_sha256": sha256(source), "key_sha256": sha256(review / "key.json"),
              "counts_relative_to_provisional_observations": {k: dict(v) for k, v in counts.items()}, "rows": rows}
    write_json(review / "comparison.json", result)
    lines = ["# Validation image review", "", result["scope"], "",
             "| Reader/metric | Assessable | Agreement | False positives | False negatives | Uncertain |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, c in counts.items():
        lines.append(f"| {name} | {c['assessable']} | {c['agreement']} | {c['false_positive']} | {c['false_negative']} | {c['uncertain']} |")
    lines += ["", "These counts are measurement diagnostics, not a replacement efficacy benchmark.",
              "Cropping and typography requirements beyond explicit metric clauses are recorded in the observations.",
              "Full per-image joins and disagreements are in comparison.json."]
    (review / "report.md").write_text("\n".join(lines) + "\n")
    print(result["counts_relative_to_provisional_observations"])


if __name__ == "__main__":
    main()
