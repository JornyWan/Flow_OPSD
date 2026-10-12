"""Prepare separate fitting/evaluation attempts after the frozen collection gate."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from opdlab.acceptance_learning import ARMS, VERSION, schedules
from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from review_acceptance_collection import build_report, rows


def points_from_collection(collection, targets, old):
    endpoints = {(r["pair_id"], r["mode"]): r for r in rows(collection / "endpoints.jsonl")}
    points = []
    for target in targets:
        images = {}
        for mode in ("noop", "geneval@1"):
            r = endpoints[target["pair_id"], mode]
            images[mode] = {"image": str(collection / r["image"]), "sha256": r["image_sha256"]}
        points.append({**target, "key": target["pair_id"], "role": "positive", "images": images,
                       "action_names": {m: m for m in ("noop", "geneval@1")}})
    for p in old["points"]:
        if p["target"] == "noop":
            points.append({**p, "role": "preservation", "replicate": 0,
                           "images": {"noop": p["images"]["noop"]},
                           "action_names": {"noop": p["action_names"]["noop"]},
                           "action_hashes": {"noop": p["action_hashes"]["noop"]}})
    assert len(points) == 54 and len({p["key"] for p in points}) == 54
    assert Counter(p["role"] for p in points) == {"positive": 48, "preservation": 6}
    return points


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--fit-run", default="runs/le20_fit_001")
    args = parser.parse_args()
    campaign = ROOT / "experiments/local_expert_v20"
    collection = ROOT / "runs/le20_collection_001"
    report = build_report(collection, ROOT / "runs/le20_review_001")
    assert report == read_json(campaign / "results.json")
    manifest = read_json(campaign / "target_manifest.json")
    assert manifest["targets"] == report["targets"] and report["coverage_gate"]["passed"]
    assert manifest["training_allowed_by_coverage_gate"] and manifest["source_results_sha256"] == sha256(campaign / "results.json")
    qualified = ROOT / "runs/le18_supervision_003/probe_plan.json"
    old = read_json(qualified)
    points = points_from_collection(collection, manifest["targets"], old)
    schedule, randomization = schedules(points, 128)
    collection_plan = read_json(collection / "probe_plan.json")
    evaluation = [{**p, "single_action": True, "cohorts": [p["cohort"], "all_fresh"]} for p in collection_plan["evaluation"]]
    assert len(evaluation) == 32 and all(p["single_action"] for p in evaluation)
    accepted = sum(p["target"] != "noop" for p in points)
    references = points if args.stage == "fit" else [
        {**p, "target": "noop", "action_names": {"noop": "noop"}, "action_hashes": {"noop": p["action_hashes"]["noop"]}}
        for p in points if p["role"] == "positive" and p["replicate"] == 0]
    nref = len(references) + (accepted if args.stage == "fit" else 0)
    config = Experiment.load(collection / "config.json")
    run = prepare(config, ROOT / f"runs/le20_{'fit' if args.stage == 'fit' else 'eval'}_{args.attempt}", read_json(ROOT / "site.local.json"))
    plan = {"version": VERSION, "stage": args.stage, "student": collection_plan["student"],
            "points": points, "schedules": schedule, "randomization": randomization, "arms": ARMS,
            "index": 3, "updates": 128, "audits": [0, 32, 128], "evaluation": evaluation,
            "evaluation_images": 224 if args.stage == "evaluate" else 0,
            "reference_images": nref, "references": references,
            "collection_targets": str(campaign / "target_manifest.json"), "qualification_plan": str(qualified),
            "scope": "Provisional assistant acceptance labels; new noise seeds on familiar prompts; fixed-query OPD transition regression"}
    inputs = {campaign / n for n in ("target_manifest.json", "results.json", "observations_frozen.json", "verification.json")}
    inputs.update([qualified, Path(old["training_execution"]["qualification_result"]), Path(plan["student"]),
                   collection / "probe_plan.json", collection / "result.json", Path(__file__).resolve()])
    if args.stage == "fit":
        plan["expected_counts"] = {"forward_student": 540 + 3 * (1536 + 54 * 3), "forward_behavior": 54 + accepted * 6,
                                   "forward_examples": 2 * (540 + 3 * (1536 + 54 * 3) + 54 + accepted * 6),
                                   "backward_queries": 4608, "forward_geneval": 0, "forward_ocr": 0,
                                   "decode_images": nref, "verifier_images": nref, "text_encode_prompts": 12}
    else:
        fit_run = Path(args.fit_run).resolve()
        fit_result, fit_plan = read_json(fit_run / "result.json"), read_json(fit_run / "probe_plan.json")
        assert fit_result["status"] == "complete" and fit_result["optimizer_updates"] == 384
        for p, h in fit_result["output_hashes"].items():
            assert sha256(fit_run / p) == h
        assert all(plan[k] == fit_plan[k] for k in ("points", "schedules", "randomization", "arms", "evaluation", "student", "updates", "audits"))
        (run / "checkpoints").symlink_to(fit_run / "checkpoints", target_is_directory=True)
        plan["fit_run"] = str(fit_run)
        # Six baseline references, then 32 initial/full + 96 trained/full + 96 local.
        plan["expected_counts"] = {"forward_student": 60 + 1280 + 96, "forward_behavior": 6 + 96 * 6,
                                   "forward_examples": 2 * (60 + 1280 + 96 + 6 + 96 * 6),
                                   "backward_queries": 0, "forward_geneval": 0, "forward_ocr": 0,
                                   "decode_images": 230, "verifier_images": 230, "text_encode_prompts": len({p["id"] for p in evaluation})}
        inputs.update([fit_run / "result.json", fit_run / "probe_plan.json", fit_run / "fit_reports.json", fit_run / "metrics.jsonl"])
        inputs.update(fit_run / "checkpoints" / arm / "step128.pt" for arm in ARMS)
    for p in points:
        inputs.add(Path(p["saved_query"]))
        inputs.update(Path(v["image"]) for v in p["images"].values())
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.add(run / name)
    write_json(run / "probe_plan.json", plan)
    inputs.add(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in sorted(inputs)}})
    write_json(campaign / f"{args.stage}_attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "counts": plan["expected_counts"], "training_states": len(points), "reference_images": nref,
               "all_noop_batches": {a: sum(all(e["target"] == "noop" for e in row) for row in schedule[a]) for a in ARMS}})
    print({"run": str(run), "counts": plan["expected_counts"], "references": nref})


if __name__ == "__main__":
    main()
