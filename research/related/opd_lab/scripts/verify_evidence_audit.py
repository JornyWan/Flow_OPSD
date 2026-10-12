"""Verify measurement replay, decisions, costs and historical immutability."""
from collections import Counter
from pathlib import Path
import subprocess

from opdlab.evidence_verifier import SPEC,choose
from opdlab.evidence_dominance import choose_dominance
from opdlab.io import ROOT,read_json,sha256,write_json
from opdlab.runtime import verify_run
from review_prospective import compare,lines,visual
from verify_state_probes import historical_index


def main():
    campaign = ROOT / "experiments/local_expert_v13"
    report = read_json(campaign / "results.json")
    run = Path(report["run"])
    assert verify_run(run) == []
    plan,gpu = read_json(run / "audit_plan.json"),read_json(run / "result.json")
    assert gpu == report["gpu_result"] and plan["verifier_spec"] == SPEC
    assert gpu["optimizer_updates"] == gpu["generated_images"] == 0
    assert report["script_sha256"] == sha256(ROOT / "scripts/summarize_evidence.py")
    assert report["source_results_sha256"] == sha256(ROOT / "experiments/local_expert_v12/results.json")
    for path,h in read_json(run / "audit_attestation.json")["inputs"].items():
        assert sha256(path) == h,path
    assert sha256(run / "evidence.jsonl") == gpu["evidence_sha256"]
    evidence = lines(run / "evidence.jsonl")
    counts,images,parity = Counter(),set(),0
    assert len(evidence) == len(report["queries"]) == len(plan["queries"]) == 64
    for spec,row,q in zip(plan["queries"],evidence,report["queries"]):
        assert (spec["id"],spec["seed"]) == (row["id"],row["seed"]) == (q["id"],q["seed"])
        distances = {c["name"]:c["distance"] for c in spec["candidates"]}
        assert choose(row["evidence"],distances) == row["decision"] == q["v2_decision"]
        # In this cohort, the declared repair-count ordering also matches adding
        # only a discrete-event filter while retaining the original v1 utility.
        base = spec["candidates"][0]["v1"]
        eligible = []
        for i,c in enumerate(spec["candidates"][1:],1):
            v = c["v1"]
            repaired = any(not a and b for a,b in zip(base["satisfied"],v["satisfied"]))
            if v["admissible"] and repaired and v["utility"] > base["utility"]:
                eligible.append(((v["utility"],-c["distance"],-i),c["name"]))
        assert (max(eligible)[1] if eligible else "noop") == q["selections"]["v1_discrete_event"]
        for c in spec["candidates"]:
            assert sha256(c["image"]) == c["image_sha256"]
            images.add(c["image_sha256"])
        counts.update(row["counts"])
        parity += sum(bool(v) for v in row["raw_replay_differences"].values())
        for policy,name in q["selections"].items():
            assert compare(q["observations"]["noop"],q["observations"][name]) == q["comparisons"][policy]
    assert dict(counts) == gpu["counts"]
    assert counts["evidence_images"] == 192 and counts["ocr_detection_views"] == 384
    assert counts["geneval_clause_views"] == 480 and len(images) == 192
    assert parity == gpu["raw_replay_mismatches"]
    for policy,metrics in report["policies"].items():
        accepted = [q for q in report["queries"] if q["selections"][policy] != "noop"]
        assert dict(Counter(q["comparisons"][policy]["status"] for q in accepted)) == metrics["accepted_classes"]
        for metric in ("complete_scene","joint_clauses","full_visibility","has_unknown"):
            total = sum(visual(q["observations"][q["selections"][policy]])[metric] for q in report["queries"])
            assert total == metrics["endpoints"][metric]["sum"]
    old = read_json(ROOT / "experiments/local_expert_v12/results.json")
    for new_key,old_key in (("acceptance","acceptance"),("safe_precision","conservative_precision"),
                            ("known_harm_rate","accepted_harm"),("unresolved_rate","accepted_unresolved"),
                            ("safe_recall","recall_including_rejections")):
        assert report["policies"]["v1"][new_key] == old["calibration"][old_key]
    assert report["policies"]["v1"]["protected_transitions"] == old["protected_transitions"]["selected"]
    posthoc = read_json(campaign / "posthoc_diagnosis.json")
    assert posthoc["results_sha256"] == sha256(campaign / "results.json")
    assert posthoc["selector_sha256"] == sha256(ROOT / "python/opdlab/evidence_dominance.py")
    assert posthoc["script_sha256"] == sha256(ROOT / "scripts/explain_evidence_audit.py")
    post_by_id = {q["id"]:q for q in posthoc["queries"]}
    for spec,row,q in zip(plan["queries"],evidence,report["queries"]):
        selected = choose_dominance(row["evidence"],{c["name"]:c["distance"] for c in spec["candidates"]})
        assert selected == post_by_id[q["id"]]["selected"]
        assert compare(q["observations"]["noop"],q["observations"][selected]) == post_by_id[q["id"]]["comparison"]
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1 and status["jobs"][0]["state"] == "COMPLETED" and status["jobs"][0]["exit_code"] == "0:0"
    upstream = {}
    for name,spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git","status","--porcelain"],cwd=path).strip() == b""
        head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    historical = {str(p):historical_index(p,revision) for p,revision in ((6,"e384a09"),(7,"8c1e160"),(8,"d9c392c"),(9,"d9c392c"),(10,"9dcc18b"),(11,"3923b6a"),(12,"8e9c655"))}
    old_worker = subprocess.check_output(["git","show","8e9c655:python/opdlab/measurement_audit.py"],cwd=ROOT)
    assert (ROOT / "python/opdlab/measurement_audit.py").read_bytes() == old_worker
    write_json(campaign / "verification.json",{"input_images_verified":192,"queries_and_decisions_recomputed":64,
               "raw_replay_mismatches":parity,"counts_verified":dict(counts),"generated_images":0,"optimizer_updates":0,
               "frozen_labels_unchanged":True,"old_measurement_worker_unchanged":True,"historical_artifacts":historical,
               "original_v1_statistics_reproduced":True,"separate_posthoc_decisions_recomputed":64,
               "discrete_filter_original_utility_ranking_matches":64,
               "upstream":upstream,"allocated_gpu_hours":status["allocated_gpu_hours_including_failed_attempts_so_far"],
               "limitations":"Development images and implementing-assistant labels; correlated detector/OCR views; large shared assets use recorded inventories."})
    print({"verified_images":len(images),"queries":len(evidence),"raw_replay_mismatches":parity,"historical_phases_unchanged":list(historical)})


if __name__ == "__main__":
    main()
