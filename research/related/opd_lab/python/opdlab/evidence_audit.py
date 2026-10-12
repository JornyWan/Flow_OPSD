"""Replay frozen endpoint images through an independently versioned verifier."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

from PIL import Image

from .backends import activate_upstream
from .data import load_records
from .engine import seed_all
from .evidence_verifier import EvidenceVerifier, SPEC, choose
from .io import append_jsonl, assert_environment, read_json, sha256, write_json
from .runtime import verify_run


def run_audit(run):
    run = Path(run).resolve()
    assert verify_run(run) == []
    assert_environment(read_json(run / "environment.json"))
    plan = read_json(run / "audit_plan.json")
    assert plan["verifier_spec"] == SPEC
    for path,h in read_json(run / "audit_attestation.json")["inputs"].items():
        assert sha256(path) == h, path
    if any((run/n).exists() for n in ("RUNNING","result.json","failure.json","evidence.jsonl")):
        raise FileExistsError("Use a fresh attempt")
    with (run / "RUNNING").open("x") as f:
        f.write(f"pid={os.getpid()}\n")
    start = time.perf_counter()
    try:
        seed_all(13013)
        activate_upstream(run / "upstream")
        from flow_grpo.ocr import OcrScorer
        from flow_grpo.gen_eval import load_geneval
        verifier = EvidenceVerifier(OcrScorer(use_gpu=False).ocr,load_geneval("cuda"))
        records = {r["id"]:r for r in load_records(run / "data.jsonl","train")}
        totals = Counter()
        for ordinal,query in enumerate(plan["queries"]):
            evidence,distances,parity = {}, {}, {}
            before = verifier.counters.copy()
            for candidate in query["candidates"]:
                path = Path(candidate["image"])
                assert sha256(path) == candidate["image_sha256"]
                with Image.open(path) as source:
                    e = verifier(source.convert("RGB"),records[query["id"]])
                evidence[candidate["name"]],distances[candidate["name"]] = e,candidate["distance"]
                raw_text = " ".join(x["text"] for x in e["evidence"]["ocr"]["raw"])
                old = candidate["v1"]
                differences = []
                if raw_text != old["details"]["ocr"]["recognized_text"]:
                    differences.append("ocr_raw")
                objects = {c["id"]:c for c in e["evidence"]["objects"][0]}
                for clause,satisfied,progress in zip(old["details"]["clauses"],old["satisfied"],old["progress"]):
                    if clause["id"] in objects:
                        new = objects[clause["id"]]
                        if new["strict"] != satisfied or abs(max(0,min(1,new["score"]))-progress)>1e-7:
                            differences.append(clause["id"])
                parity[candidate["name"]] = differences
                totals["raw_replay_mismatches"] += bool(differences)
            decision = choose(evidence,distances)
            row = {"ordinal":ordinal,"id":query["id"],"seed":query["seed"],"v1_selected":query["v1_selected"],
                   "evidence":evidence,"decision":decision,"raw_replay_differences":parity,
                   "counts":dict(verifier.counters-before)}
            append_jsonl(run / "evidence.jsonl",row)
            totals["queries"] += 1
            print({"event":"query_complete","completed":ordinal+1,"total":len(plan["queries"])},flush=True)
        elapsed = time.perf_counter()-start
        write_json(run / "result.json",{"status":"complete","verifier_spec":SPEC,"queries":totals["queries"],
                   "raw_replay_mismatches":totals["raw_replay_mismatches"],"counts":dict(verifier.counters),
                   "optimizer_updates":0,"generated_images":0,"elapsed_seconds":elapsed,"worker_gpu_hours":elapsed/3600,
                   "evidence_sha256":sha256(run / "evidence.jsonl")})
    except Exception as e:
        write_json(run / "failure.json",{"error":repr(e),"elapsed_seconds":time.perf_counter()-start})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    run_audit(parser.parse_args().run)
