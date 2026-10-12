"""Freeze the complete phase16 diagnostic before allocating a GPU."""
import argparse
from pathlib import Path
import shutil

from opdlab.action_transfer import VERSION, select_pairs
from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from review_prospective import lines


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--site", default="site.local.json")
    args = parser.parse_args()
    campaign, previous = ROOT / "experiments/local_expert_v16", ROOT / "runs/le15_annotation_001"
    old = read_json(previous / "probe_plan.json")
    points, pairs = select_pairs(old)
    endpoints = lines(previous / "endpoints.jsonl")
    refs = {p["pair_id"]: {n: next(r for r in endpoints if (r["pair_id"], r["model"]) == (p["pair_id"], n))
                          for n in ("joint128", "label_selected")} for p in pairs}
    config = Experiment.load(previous / "config.json")
    run = prepare(config, ROOT / f"runs/le16_transfer_{args.attempt}", read_json(args.site))
    plan = {"version": VERSION, "previous": str(previous), "query_source": old["source"], "points": points,
            "pairs": pairs, "references": refs, "index": 3, "alpha": 1,
            "checkpoints": {"joint128": old["student"], "label_selected": str(previous / "checkpoints/label_selected/step128.pt")},
            "expected_counts": {"forward_student": 378, "forward_behavior": 306, "forward_ocr": 12,
                                "forward_geneval": 12, "forward_examples": 1404, "decode_images": 84,
                                "verifier_images": 84, "backward_queries": 0, "text_encode_prompts": 12},
            "scope": "Development-only frozen-checkpoint mechanism diagnosis; no fitting, no seed/checkpoint selection"}
    write_json(run / "probe_plan.json", plan)
    inputs = [previous / n for n in ("probe_plan.json", "result.json", "endpoints.jsonl", "probe_attestation.json")]
    inputs += [ROOT / "experiments/local_expert_v15" / n for n in
               ("results.json", "observations.json", "observations_frozen.json", "report_index.json", "NEXT.md")]
    inputs += [ROOT / "experiments/local_expert_v12/results.json"] + [Path(p) for p in plan["checkpoints"].values()]
    for p in points:
        inputs.append(Path(old["source"]) / p["saved_query"])
        for im in p["images"].values():
            inputs.append(Path(old["source"]) / im["image"])
    for group in refs.values():
        inputs.extend(previous / r["image"] for r in group.values())
    for n in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / n, run / n)
        inputs.append(run / n)
    inputs += [run / "probe_plan.json", Path(__file__).resolve()]
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md"), "new_images": 42, "new_teacher_queries": 24, "updates": 0})
    print({"run": str(run), "pairs": len(pairs), "counts": plan["expected_counts"]})


if __name__ == "__main__":
    main()
