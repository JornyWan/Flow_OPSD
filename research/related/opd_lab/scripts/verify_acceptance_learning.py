"""Audit exact schedules, optimizer budgets, immutable collection and paired evaluation."""
import argparse
import json
from pathlib import Path

import torch

from opdlab.acceptance_learning import ARMS, schedules
from opdlab.fitting_probe import tensor_digest
from opdlab.io import assert_environment, read_json, sha256, write_json
from opdlab.runtime import verify_run
from review_acceptance_learning import build_report
from review_prospective import lines
from verify_state_probes import historical_index


def audit_run(run):
    assets=verify_run(run)
    assert_environment(read_json(run/"manifest.json")["environment"])
    for p,h in read_json(run/"probe_attestation.json")["inputs"].items():
        assert sha256(p)==h,p
    plan,result=read_json(run/"probe_plan.json"),read_json(run/"result.json")
    assert result["status"]=="complete" and not any((run/n).exists() for n in ("RUNNING","failure.json"))
    assert (plan["schedules"],plan["randomization"])==schedules(plan["points"],128)
    for p,h in result["output_hashes"].items():
        assert sha256(run/p)==h,p
    for k,v in plan["expected_counts"].items():
        assert result["counts"].get(k,0)==v,(k,v)
    assert result["new_teacher_queries"]==0
    for row in lines(run/"references.jsonl"):
        assert row["exact"] and row["valid"] and sha256(run/row["image"])==row["image_sha256"]
    assert len(lines(run/"references.jsonl"))==plan["reference_images"]
    return plan,result,assets


def verify_fit(run):
    plan,result,assets=audit_run(run)
    assert plan["stage"]=="fit" and result["optimizer_updates"]==384 and result["evaluation_images"]==0
    assert read_json(run/"fit_reports.json")==result["arms"]
    assert not (run/"endpoints.jsonl").exists() and not (run/"eval_queries").exists()
    initial=tensor_digest(torch.load(plan["student"],map_location="cpu",weights_only=True))
    updates=lines(run/"updates.jsonl")
    metrics=lines(run/"metrics.jsonl")
    assert len(updates)==384 and len(metrics)==486
    assert {(m["arm"],m["step"],m["key"]) for m in metrics}=={(a,s,p["key"]) for a in ARMS for s in (0,32,128) for p in plan["points"]}
    for arm in ARMS:
        actual=[r for r in updates if r["arm"]==arm]
        assert [r["step"] for r in actual]==list(range(1,129))
        assert [r["queries"] for r in actual]==plan["schedules"][arm]
        r=result["arms"][arm]
        assert r["optimizer_updates"]==128 and r["counts"]["backward_queries"]==1536 and r["initial_sha256"]==initial
        assert tensor_digest(torch.load(run/f"checkpoints/{arm}/step000.pt",map_location="cpu",weights_only=True))==initial
        final=torch.load(run/f"checkpoints/{arm}/step128.pt",map_location="cpu",weights_only=True)
        saved=torch.load(run/f"checkpoints/{arm}/training.pt",map_location="cpu",weights_only=False)
        assert tensor_digest(final)==tensor_digest(saved["student"])==r["final_sha256"]
        assert saved["optimizer_step"]==128 and saved["optimizer"]["state"]
        assert all(v["step"].item()==128 for v in saved["optimizer"]["state"].values())
    execution=result["training_execution"]
    assert execution["deterministic_algorithms"] and not execution["warn_only"] and execution["cublas_workspace_config"]==":4096:8"
    refs={r["key"]+"/"+r["mode"]:r for r in lines(run/"references.jsonl")}
    for p in plan["points"]:
        assert sha256(p["saved_query"])==p["saved_query_sha256"]
        for mode in ("noop",p["target"]):
            assert refs[p["key"]+"/"+mode]["image_sha256"]==p["images"][mode]["sha256"]
    return {"assets":assets,"reference_images_exact":63,"updates":384,"backward_queries":4608,
            "training_states":54,"all_schedules_reconstructed":True,"complete_optimizer_snapshots":3,
            "evaluation_images_generated":0,"deterministic_fitting":True}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("fit","outputs","evaluate"))
    parser.add_argument("--fit-run",default="runs/le20_fit_002")
    parser.add_argument("--run",default="runs/le20_eval_001")
    parser.add_argument("--review",default="runs/le20_eval_review_001")
    parser.add_argument("--campaign",default="experiments/local_expert_v20")
    args=parser.parse_args()
    campaign=Path(args.campaign).resolve()
    report={"fit":verify_fit(Path(args.fit_run).resolve())}
    if args.stage in ("outputs","evaluate"):
        run,review=Path(args.run).resolve(),Path(args.review).resolve()
        plan,result,assets=audit_run(run)
        assert result["optimizer_updates"]==0 and result["evaluation_images"]==224
        assert plan["fit_run"]==str(Path(args.fit_run).resolve())
        rows=lines(run/"endpoints.jsonl")
        assert len(rows)==224 and len(list((run/"eval_queries").glob("*.pt")))==32
        initial=tensor_digest(torch.load(plan["student"],map_location="cpu",weights_only=True))
        collection=read_json(Path(plan["collection_targets"]).parent/"target_manifest.json")
        assert not {r["seed"] for r in rows}&{p["seed"] for p in collection["targets"]}
        for pair in plan["evaluation"]:
            q=torch.load(run/"eval_queries"/f"{pair['pair_id']}.pt",map_location="cpu",weights_only=False)
            assert (q["record"]["id"],q["seed"],q["index"])==(pair["id"],pair["seed"],3)
            noise=torch.randn(q["state"].shape,generator=torch.Generator().manual_seed(q["seed"]),dtype=torch.float32)
            expected=tensor_digest({"noise":noise})
            matched=[r for r in rows if r["pair_id"]==pair["pair_id"]]
            assert len(matched)==7 and all(r["noise_sha256"]==expected for r in matched)
            for r in matched:
                assert r["valid"] and sha256(run/r["image"])==r["image_sha256"]
                assert r["student_sha256"]==result["model_hashes"][r["model"]]
            assert result["model_hashes"]["joint128"]==initial
        report["evaluation"]={"assets":assets,"new_states":32,"images":224,"reference_images_exact":6,"review_complete":False}
        if args.stage=="evaluate":
            rebuilt=json.loads(json.dumps(build_report(run,review)))
            assert rebuilt==read_json(campaign/"learning_results.json")
            items={i["case"]:i for i in read_json(review/"items.json")}
            for item in read_json(review/"key.json"):
                for letter,r in item["mapping"].items():
                    assert sha256(review/items[item["case"]]["images"][letter])==r["image_sha256"]
            report["evaluation"].update(review_complete=True,review_slots=256,statistics_rebuilt=True,pilot=rebuilt["promising_pilot"])
    report["phase19_archive"]=historical_index(19,"9cbbf01")
    report["stage"]=args.stage
    write_json(campaign/f"{args.stage}_verification.json",report)
    print({"verified":args.stage,"fit_updates":384,"backward_queries":4608})


if __name__=="__main__":
    main()
