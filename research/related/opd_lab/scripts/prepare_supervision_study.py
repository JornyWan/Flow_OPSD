"""Phase18: compile existing certified actions into one inventory and matched schedules."""
import argparse
from pathlib import Path
import shutil

import torch

from opdlab.config import Experiment
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, digest, read_json, sha256, stable_seed, write_json
from opdlab.runtime import prepare
from opdlab.supervision_study import VERSION, schedules
from prepare_coverage_probe import seed_values
from review_prospective import lines


def seed_inventory():
    patterns = ("*plan*.json", "*protocol*.json", "*manifest*.json", "config.json", "endpoints.jsonl",
                "candidates.jsonl", "results.jsonl", "references.jsonl", "queries.jsonl", "results.json", "supervision_inventory.json")
    paths = set()
    for pattern in patterns:
        paths.update((ROOT / "runs").glob(f"*/{pattern}"))
        paths.update((ROOT / "experiments").glob(f"local_expert_v*/{pattern}"))
    files, excluded = {}, set()
    for p in sorted(paths):
        if "le18_" in str(p) or "local_expert_v18" in str(p):
            continue
        seeds = seed_values(lines(p) if p.suffix == ".jsonl" else read_json(p))
        files[str(p)] = {"sha256": sha256(p), "seeds": sorted(seeds)}
        excluded.update(seeds)
    return {"files": files, "excluded_seeds": sorted(excluded), "scope": "Conservative prior lab top-level seed-field integer inventory, including phase17 and failed runs; not unrelated projects."}


def compile_points(old):
    inventory = read_json(ROOT / "experiments/local_expert_v17/supervision_inventory.json")["supervision_inventory"]["fixed_i3"]["states"]
    points = []
    for p in old["points"]:
        points.append({**p, "saved_query": str(Path(old["source"]) / p["saved_query"]),
                       "images": {n: {**v, "image": str(Path(old["source"]) / v["image"])} for n, v in p["images"].items()},
                       "target": p["label_target"], "original": True, "origin": "phase12"})
    for target in inventory:
        if target["origin"].startswith("phase12"):
            continue
        phase = 16 if target["origin"] == "phase16_fresh" else 17
        run = ROOT / ("runs/le16_transfer_001" if phase == 16 else "runs/le17_coverage_001")
        rows = lines(run / "endpoints.jsonl") + (lines(run / "references.jsonl") if phase == 16 else [])
        names = {"noop": "joint128" if phase == 16 else "noop", "ocr@1": "ocr@1" if phase == 16 else "i3_ocr@1", "geneval@1": "geneval@1" if phase == 16 else "i3_geneval@1"}
        images = {}
        for n, mode in names.items():
            matches = [r for r in rows if r["id"] == target["id"] and r["seed"] == target["seed"] and r["mode"] == mode]
            assert len(matches) == 1
            row = matches[0]
            images[n] = {"image": str(run / row["image"]), "sha256": row["image_sha256"]}
        points.append({**target, "group": target["id"].rsplit("-", 1)[0], "original": False,
                       "target": target["mode"].removeprefix("i3_"), "images": images})
    for p in points:
        assert p["index"] == 3 and sha256(p["saved_query"]) == p["saved_query_sha256"]
        q = torch.load(p["saved_query"], map_location="cpu", weights_only=False)
        p["action_names"] = {n: n if n in q["actions"] else f"i3_{n}" for n in ("noop", "ocr@1", "geneval@1")}
        p["action_hashes"] = {n: tensor_digest({"action": q["actions"][source]}) for n, source in p["action_names"].items()}
        assert (q["record"]["id"], q["seed"], q["index"]) == (p["id"], p["seed"], p["index"])
        for v in p["images"].values():
            assert sha256(v["image"]) == v["sha256"]
        p["key"] = digest([p["id"], p["seed"], p["index"]])[:24]
        if p["target"] != "noop":
            certified = next(t for t in inventory if (t["id"], t["seed"]) == (p["id"], p["seed"]))
            assert certified["action_sha256"] == p["action_hashes"][p["target"]]
    assert len(points) == len({p["key"] for p in points}) == 15
    return points


def evaluation_pairs(old, points, excluded):
    added = {p["id"] for p in points if not p["original"]}
    pairs = []
    for source in old["evaluation"]:
        if source["cohort"] == "fitting_seed_diagnostic":
            continue
        p = {k: source[k] for k in ("id", "replicate", "cohort")}
        p["seed"] = stable_seed(18018, VERSION, "evaluation", p["id"], p["replicate"])
        p["cohorts"] = [p["cohort"], "all_fresh"]
        p["single_action"] = p["cohort"] == "positive_fresh_seed"
        if p["single_action"]:
            p["cohorts"].append("added_prompts_fresh" if p["id"] in added else "other_positive_prompts_fresh")
        pairs.append(p)
    assert len(pairs) == len({p["seed"] for p in pairs}) == 32
    assert not {p["seed"] for p in pairs} & set(excluded)
    for p in points:
        if not p["original"]:
            pairs.append({"id": p["id"], "seed": p["seed"], "replicate": 0, "cohort": "added_fitting_states",
                          "cohorts": ["added_fitting_states"], "single_action": True})
    for p in pairs:
        p["pair_id"] = digest([VERSION, p["id"], p["seed"]])[:24]
    assert len({p["pair_id"] for p in pairs}) == 35
    return pairs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--site", default="site.local.json")
    args = parser.parse_args()
    campaign = ROOT / "experiments/local_expert_v18"
    source = ROOT / "runs/le15_annotation_001"
    old = read_json(source / "probe_plan.json")
    points, inventory = compile_points(old), seed_inventory()
    pairs = evaluation_pairs(old, points, inventory["excluded_seeds"])
    config = Experiment.load(source / "config.json")
    run = prepare(config, ROOT / f"runs/le18_supervision_{args.attempt}", read_json(args.site))
    old_model = source / "checkpoints/label_selected/step128.pt"
    plan = {"version": VERSION, "student": old["student"], "points": points, "updates": 128, "audits": [0, 32, 128],
            "index": 3, "arms": ["single", "multi", "random_matched"], "schedules": schedules(points, 128), "evaluation": pairs,
            "reference_model_hashes": {"single": tensor_digest(torch.load(old_model, map_location="cpu", weights_only=True))},
            "reference_model": str(old_model), "reference_images": 24, "evaluation_images": 185,
            "expected_counts": {"forward_student": 6338, "forward_behavior": 339, "forward_examples": 13354,
                                "backward_queries": 4608, "forward_ocr": 0, "forward_geneval": 0,
                                "decode_images": 209, "verifier_images": 209, "text_encode_prompts": 40},
            "scope": "Fixed-index multistate supervision mechanism test with provisional assistant labels; no automatic verifier or native OPD claim"}
    write_json(run / "probe_plan.json", plan)
    write_json(run / "seed_inventory.json", inventory)
    paths = {Path(p) for p in inventory["files"]}
    paths.update([Path(plan["student"]), old_model, Path(__file__).resolve(), ROOT / "scripts/review_supervision.py"])
    paths.update(ROOT / "experiments/local_expert_v17" / n for n in ("NEXT.md", "report_index.json", "supervision_inventory.json"))
    for p in points:
        paths.add(Path(p["saved_query"]))
        paths.update(Path(v["image"]) for v in p["images"].values())
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        paths.add(run / name)
    paths.update([run / "probe_plan.json", run / "seed_inventory.json"])
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in sorted(paths)}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "seed_source_files": len(inventory["files"]), "excluded_integer_count": len(inventory["excluded_seeds"]),
               "new_evaluation_seeds": 32, "training_query_states": 15, "backward_queries": 4608,
               "evaluation_images": 185, "new_teacher_queries": 0})
    print({"run": str(run), "points": 15, "pairs": 35, "images": 185, "counts": plan["expected_counts"]})


if __name__ == "__main__":
    main()
