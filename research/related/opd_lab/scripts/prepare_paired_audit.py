"""Freeze all128 phase12 expert pairs, reversed pairs and64 self controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.paired_audit import SPEC, requirements


def main(attempt):
    campaign = ROOT / "experiments/local_expert_v14"
    source = ROOT / "runs/le12_collect_002"
    rows = [json.loads(s) for s in (source / "candidates.jsonl").read_text().splitlines()]
    records = {r["id"]: r for r in map(json.loads, (source / "data.jsonl").read_text().splitlines())}
    assert len(rows) == 64 and sha256(source / "candidates.jsonl") == read_json(source / "result.json")["candidates_sha256"]
    images, pairs, key, tasks = {}, [], [], []
    for row in rows:
        record = records[row["id"]]
        assert record["split"] == "train"
        candidates = row["candidates"]
        assert [c["name"] for c in candidates] == ["noop", "ocr@1", "geneval@1"]
        uid = digest([SPEC["seed"], row["id"], row["seed"]])[:24]
        names = {}
        for c in candidates:
            image_id = digest([uid, c["name"]])[:24]
            path = source / c["image"]
            assert sha256(path) == c["image_sha256"]
            images[image_id] = {"uid": image_id, "image": str(path), "image_sha256": c["image_sha256"]}
            names[c["name"]] = image_id
        key.append({"uid": uid, "id": row["id"], "seed": row["seed"], "images": names,
                    "distances": {c["name"]: c["distance"] for c in candidates[1:]}})
        for name in ("ocr@1", "geneval@1", "noop"):
            pid = digest([uid, name, "pair"])[:24]
            pair = {"uid": pid, "query_uid": uid, "images": [names["noop"], names[name]],
                    "clauses": requirements(record), "self_control": name == "noop"}
            pairs.append(pair)
            for reverse in ([False] if name == "noop" else [False, True]):
                tasks.append({"uid": digest([pid, reverse])[:24], "pair_uid": pid, "reverse": reverse,
                              "images": list(reversed(pair["images"])) if reverse else pair["images"]})
    assert len(images) == 192 and len(pairs) == 192 and len(tasks) == 320
    run = ROOT / f"runs/le14_paired_{attempt}"
    run.mkdir(exist_ok=False)
    framework = run / "framework/python/opdlab"
    framework.mkdir(parents=True)
    for f in ("__init__.py", "io.py", "paired_audit.py", "evidence_dominance.py", "vlm_audit.py"):
        shutil.copyfile(ROOT / "python/opdlab" / f, framework / f)
    for f in ("run_paired_audit.py", "run_vlm_audit.py"):
        shutil.copyfile(ROOT / "scripts" / f, run / f)
    shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
    old = ROOT / "experiments/local_expert_v6"
    inputs = [source / n for n in ("candidates.jsonl", "result.json", "data.jsonl")]
    inputs += [ROOT / "experiments/local_expert_v12" / n for n in ("observations.json", "observations_frozen.json", "results.json", "report_index.json")]
    inputs += [ROOT / "experiments/local_expert_v13" / n for n in ("results.json", "posthoc_diagnosis.json", "report_index.json")]
    inputs += [old / n for n in ("vlm_assets.json", "vlm_environment.json", "model_index_audit.json")]
    plan = {"spec": SPEC, "scope": "exposed phase12 development images; no training or new validation",
            "images": sorted(images.values(), key=lambda r: digest([r["uid"], "observation_order"])),
            "pairs": pairs, "tasks": sorted(tasks, key=lambda r: digest([r["uid"], "task_order"])),
            "assets": read_json(old / "vlm_assets.json"), "environment": read_json(old / "vlm_environment.json"),
            "input_files": {str(p): sha256(p) for p in inputs},
            "framework_files": {str(p.relative_to(run)): sha256(p) for p in [*run.glob("*.py"), *framework.glob("*.py")]}}
    write_json(run / "private_key.json", key)
    write_json(run / "paired_plan.json", plan)
    write_json(run / "attestation.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "plan_sha256": sha256(run / "paired_plan.json"), "protocol_sha256": sha256(run / "PROTOCOL.md"),
               "private_key_sha256": sha256(run / "private_key.json")})
    write_json(run / "site.json", {"python": plan["environment"]["executable"],
               "environment": {"HF_HUB_OFFLINE": "1", "HF_HOME": "/midtier/minilab/scratch/chw4026/.cache/huggingface"}})
    write_json(campaign / f"attempt_{attempt}.json", {"run": str(run), **read_json(run / "attestation.json"),
               "calls": 512, "generated_images": 0, "optimizer_updates": 0})
    print(run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--attempt", default="001")
    main(p.parse_args().attempt)
