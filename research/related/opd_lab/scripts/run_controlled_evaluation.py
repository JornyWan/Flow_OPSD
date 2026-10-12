"""Evaluate every predeclared phase-2 arm in one sampler/step cell on one GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from opdlab.audit_ocr import AuditOCR
from opdlab.evaluate import compare, read_evaluation
from opdlab.io import read_json, sha256, write_json
from opdlab.report import prompt_summary
from opdlab.verifiers import text_progress


def secondary_scores(path, records, reader, config):
    _, rows = read_evaluation(path)
    scored = {}
    for key, row in rows.items():
        # Preserve all object clauses; only replace the OCR clause with another proxy.
        result = reader(Path(path) / row["image"], row["image_sha256"])
        satisfied, progress = list(row["satisfied"]), list(row["progress"])
        for i, clause in enumerate(records[row["id"]]["constraints"]):
            if clause["kind"] == "ocr":
                satisfied[i], progress[i] = text_progress(result["text"], clause["text"])
        joint = all(satisfied)
        eta = config["utility_joint_weight"]
        scored[key] = {"id": row["id"], "replicate": row["replicate"], **result,
                       "utility": eta * joint + (1 - eta) * sum(progress) / len(progress),
                       "joint_success": joint, "satisfied": satisfied, "progress": progress}
    write_json(Path(path) / "secondary_ocr.json", {"reader": reader.version,
               "interpretation": "Audit-only OCR proxy; shares GenEval clauses and is not ground truth.",
               "rows": list(scored.values())})
    return scored


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--context", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--ocr-config", required=True)
    p.add_argument("--sampler", required=True, choices=["flow", "dpm2"])
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--out", required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--defer-ocr", action="store_true", help="Generate on GPU; score saved images later on the calibrated OCR host")
    modes.add_argument("--score-existing", action="store_true", help="Finish deferred audit scoring without loading the diffusion model")
    args = p.parse_args()
    context, out = Path(args.context).resolve(), Path(args.out).resolve()
    if not args.score_existing:
        out.mkdir(parents=True, exist_ok=False)
    plan, site, config = read_json(args.runs), read_json(context / "site.json"), read_json(context / "config.json")
    protocol = plan["evaluation"]
    if [args.sampler, args.steps] not in protocol["cells"]:
        raise ValueError("Cell was not predeclared")
    env = dict(os.environ, PYTHONPATH=str(context / "framework/python"))
    records = {r["id"]: r for r in map(json.loads, (context / "data.jsonl").read_text().splitlines())}
    reader = None if args.defer_ocr else AuditOCR(args.ocr_config)
    secondary, results = {}, {"status": "running", "runs_manifest_sha256": sha256(args.runs),
                             "ocr_config_sha256": sha256(args.ocr_config), "arms": {}}
    if args.score_existing:
        prior = read_json(out / "result.json")
        if (prior["status"] != "awaiting_secondary_ocr"
                or prior["runs_manifest_sha256"] != results["runs_manifest_sha256"]
                or prior["ocr_config_sha256"] != results["ocr_config_sha256"]):
            raise ValueError("Only the matching deferred evaluation can be finalized")
    for name, run in [("base", None), *plan["training_runs"].items()]:
        target = out / name
        command = [site["python"], "-m", "opdlab", "eval", "--run", str(context), "--out", str(target),
                   "--sampler", args.sampler, "--steps", str(args.steps), "--split", protocol["split"],
                   "--seeds", *map(str, protocol["seeds"]), "--save-images"]
        if run:
            if read_json(Path(run) / "result.json")["status"] != "complete":
                raise ValueError("Training arm incomplete")
            checkpoint = Path(run) / f"checkpoints/epoch-{protocol['checkpoint_epoch']:06d}"
            if not read_json(checkpoint / "metadata.json")["complete"]:
                raise ValueError("Checkpoint incomplete")
            command += ["--adapter", str(checkpoint / "raw_lora"), "--adapter-kind", protocol["adapter_kind"]]
        print(json.dumps({"event": "evaluate_arm", "name": name, "sampler": args.sampler, "steps": args.steps}), flush=True)
        if not args.score_existing:
            subprocess.run(command, env=env, check=True)
        if reader is not None:
            secondary[name] = secondary_scores(target, records, reader, config)
        results["arms"][name] = read_json(target / "summary.json")
        write_json(out / "progress.json", results)
    contrasts = [(name, "base") for name in plan["training_runs"]] + [
                 ("verified_fixed", "fixed_geneval"), ("verified_multi", "verified_fixed"),
                 ("verified_multi", "utility_only")]
    results["contrasts"] = {}
    for method, baseline in contrasts:
        results["contrasts"][f"{method}_minus_{baseline}"] = {
            "primary": compare(out / baseline, out / method)}
        if reader is not None:
            a, b = secondary[baseline], secondary[method]
            if a.keys() != b.keys():
                raise ValueError("Secondary evaluation sample mismatch")
            results["contrasts"][f"{method}_minus_{baseline}"]["secondary_ocr"] = {
                metric: prompt_summary([(key[0], float(b[key][metric]) - float(a[key][metric])) for key in a])
                for metric in ("utility", "joint_success")}
    results["status"] = "awaiting_secondary_ocr" if args.defer_ocr else "complete"
    write_json(out / "result.json", results)
    print(json.dumps({"event": "cell_complete", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
