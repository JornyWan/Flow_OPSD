"""Complete interrupted evaluation from verified final models, without more fitting."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import seed_all
from .fitting_probe import tensor_digest
from .io import ROOT, append_jsonl, assert_environment, read_json, sha256, write_json
from .learnability import save_verdict, single_endpoint
from .runtime import prepare, verify_run
from .supervision_study import VERSION, load_queries, schedules
from .training_replay import qualification
from .verifiers import build_verifier


def rows(path):
    import json
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def key(row):
    return (row["pair_id"], row["model"], row["path"])


def tasks(plan, existing):
    expected = [{**p, "model": m, "path": path} for m in ["joint128", *plan["arms"]] for p in plan["evaluation"]
                for path in (["full", "single"] if p["single_action"] and m != "joint128" else ["full"])]
    seen = {key(r): r for r in existing}
    assert len(seen) == len(existing) and set(seen) <= {key(r) for r in expected}
    for p in expected:
        if key(p) in seen:
            assert all(seen[key(p)][k] == v for k, v in p.items())
    # First available row per model/path, selected only by frozen order.
    replay = {}
    for r in existing:
        replay.setdefault((r["model"], r["path"]), r)
    return list(replay.values()), [p for p in expected if key(p) not in seen]


def inspect_training(source):
    verify_run(source)
    plan = read_json(source / "probe_plan.json")
    qualification(plan)
    assert plan["schedules"] == schedules(plan["points"], plan["updates"])
    updates, metrics = rows(source / "updates.jsonl"), rows(source / "metrics.jsonl")
    assert len(updates) == 384 and len(metrics) == 135
    assert Counter((r["arm"], r["step"]) for r in metrics) == {(a,s):15 for a in plan["arms"] for s in plan["audits"]}
    initial = tensor_digest(torch.load(plan["student"], map_location="cpu", weights_only=True))
    reports = {}
    for arm in plan["arms"]:
        actual = [r for r in updates if r["arm"] == arm]
        assert [r["step"] for r in actual] == list(range(1,129))
        assert [r["queries"] for r in actual] == plan["schedules"][arm]
        directory = source / "checkpoints" / arm
        assert tensor_digest(torch.load(directory / "step000.pt", map_location="cpu", weights_only=True)) == initial
        final = torch.load(directory / "step128.pt", map_location="cpu", weights_only=True)
        training = torch.load(directory / "training.pt", map_location="cpu", weights_only=False)
        assert training["optimizer_step"] == 128 and tensor_digest(training["student"]) == tensor_digest(final)
        assert training["optimizer"]["state"] and all(int(v["step"]) == 128 for v in training["optimizer"]["state"].values())
        reports[arm] = {"optimizer_updates":128, "initial_sha256":initial, "final_sha256":tensor_digest(final),
                        "counts":{"backward_queries":1536,"forward_student":1581,"forward_examples":3162},
                        "counts_basis":"Reconstructed from complete update/audit records and frozen loop.",
                        "historical_replay_exact":None, "elapsed_seconds":None,
                        "elapsed_note":"Per-arm timer was lost on timeout; do not infer wall time from file mtimes."}
    # Reaching evaluation proves the training context's final deterministic assertion passed.
    assert "'evaluation_complete', 'model': 'multi'" in (source / "slurm-1050296.log").read_text()
    return reports


def prepare_completion(source, run):
    source, run = Path(source).resolve(), Path(run).resolve()
    reports = inspect_training(source)
    plan = read_json(source / "probe_plan.json")
    existing = rows(source / "endpoints.jsonl")
    replay, missing = tasks(plan, existing)
    assert len(existing) == 158 and len(replay) == 7 and len(missing) == 27
    inputs = read_json(source / "probe_attestation.json")["inputs"]
    paths = [source / n for n in ("manifest.json","config.json","probe_plan.json","updates.jsonl","metrics.jsonl","references.jsonl","endpoints.jsonl","slurm-1050296.log")]
    paths += list((source / "checkpoints").rglob("*.pt")) + list((source / "eval_queries").glob("*.pt"))
    paths += [source / r["image"] for r in existing + rows(source / "references.jsonl")]
    for r in existing:
        assert r["valid"] and sha256(source / r["image"]) == r["image_sha256"]
        expected_hash = reports[r["model"]]["final_sha256"] if r["model"] != "joint128" else reports["single"]["initial_sha256"]
        assert r["student_sha256"] == expected_hash
    inputs.update({str(p):sha256(p) for p in paths})
    for p,h in inputs.items():
        assert sha256(p) == h,p
    prepare(Experiment.load(source / "config.json"),run,read_json(source / "site.json"))
    for directory in ("checkpoints","eval_queries"):
        shutil.copytree(source / directory, run / directory)
    for name in ("updates.jsonl","metrics.jsonl","endpoints.jsonl","seed_inventory.json","REVIEW_RUBRIC.md"):
        shutil.copyfile(source / name, run / name)
    for r in existing:
        target = run / r["image"]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source / r["image"],target)
    plan["completion"] = {"source":str(source),"training_reports":reports,"replay":replay,"missing":missing,
                           "source_endpoints":158,"new_endpoints":27,"replay_images":31,
                           "expected_runtime_counts":{"forward_student":400,"forward_behavior":129,"forward_examples":1058,
                                                      "backward_queries":0,"forward_ocr":0,"forward_geneval":0,"decode_images":58,"verifier_images":58},
                           "scope":"Reuse all final training states and158 validated outputs. Reproduce24 old references and7 existing model/path outputs before filling27 missing outputs. No optimizer update, seed change, or label access."}
    write_json(run / "probe_plan.json",plan)
    amendment = (ROOT / "experiments/local_expert_v18/PROTOCOL_COMPLETION.md").read_text()
    (run / "PROTOCOL.md").write_text((source / "PROTOCOL.md").read_text()+"\n\n"+amendment)
    paths = [run / n for n in ("probe_plan.json","PROTOCOL.md","REVIEW_RUBRIC.md","seed_inventory.json")]
    paths += [ROOT / "experiments/local_expert_v18/PROTOCOL_COMPLETION.md"]
    inputs.update({str(p):sha256(p) for p in paths})
    write_json(run / "probe_attestation.json",{"inputs":inputs})
    print({"run":str(run),"preserved":158,"missing":27,"replays":31,"updates":0})


@torch.no_grad()
def complete(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p,h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p)==h,p
    if any((run / n).exists() for n in ("RUNNING","result.json","failure.json","references.jsonl")):
        raise FileExistsError("Use a fresh completion attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started=time.perf_counter()
    try:
        plan=read_json(run / "probe_plan.json"); config=Experiment.load(run / "config.json")
        recovery=plan["completion"]
        evidence=qualification(plan)
        assert torch.__version__==evidence["torch"] and torch.cuda.get_device_name()==evidence["gpu"]
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8" and not torch.are_deterministic_algorithms_enabled()
        seed_all(config.seed)
        backend=build_backend(config,read_json(run / "site.json"),run / "upstream","cuda",training=True)
        initial=torch.load(plan["student"],map_location="cpu",weights_only=True)
        backend.load_student(initial,refresh_behavior=True)
        records={r["id"]:r for r in load_records(config.dataset,"train")}
        verifier=build_verifier(config,read_json(run / "site.json"),backend.device,list(records.values()))
        load_queries(backend,verifier,config,plan,run)
        models={"joint128":plan["student"], **{a:str(run / f"checkpoints/{a}/step128.pt") for a in plan["arms"]}}
        hashes={m:tensor_digest(torch.load(p,map_location="cpu",weights_only=True)) for m,p in models.items()}
        for phase in ("replay","missing"):
            for spec in recovery[phase]:
                model,pid,mode=spec["model"],spec["pair_id"],spec["path"]
                backend.load_student(torch.load(models[model],map_location="cpu",weights_only=True),refresh_behavior=False)
                noise=tensor_digest({"noise":backend.initial(spec["seed"])})
                if mode=="full":
                    image=backend.generate(records[spec["id"]],spec["seed"],config.steps)
                else:
                    q=torch.load(run / f"eval_queries/{pid}.pt",map_location="cpu",weights_only=False)
                    assert (q["record"]["id"],q["seed"],q["index"],q["noise_sha256"])==(spec["id"],spec["seed"],plan["index"],noise)
                    q["state"]=q["state"].to(backend.device)
                    backend.schedule(config.steps)
                    action=backend.predict(q["state"],q["index"],q["context"],"student")
                    image=single_endpoint(backend,q,action,config.steps)
                identity=[VERSION,pid,model,mode] if phase=="missing" else ["completion_replay",pid,model,mode]
                verdict=save_verdict(run,backend,verifier,records[spec["id"]],image,identity,config.utility_joint_weight)
                if phase=="replay":
                    assert verdict["image_sha256"]==spec["image_sha256"] and hashes[model]==spec["student_sha256"] and noise==spec["noise_sha256"]
                pair=next(p for p in plan["evaluation"] if p["pair_id"]==pid)
                append_jsonl(run / ("replays.jsonl" if phase=="replay" else "endpoints.jsonl"),
                             {**pair,"model":model,"path":mode,"student_sha256":hashes[model],"noise_sha256":noise,**verdict})
            print({"event":"completion_stage_done","phase":phase,"images":len(recovery[phase])},flush=True)
        assert tensor_digest(backend.behavior_state())==hashes["joint128"]
        runtime_counts=dict(backend.counters+verifier.counters)
        for k,v in recovery["expected_runtime_counts"].items():
            assert runtime_counts.get(k,0)==v,(k,runtime_counts.get(k),v)
        endpoints=rows(run / "endpoints.jsonl")
        assert len(endpoints)==185 and len({key(r) for r in endpoints})==185
        noises={r["pair_id"]:r["noise_sha256"] for r in endpoints}
        assert all(noises[r["pair_id"]]==r["noise_sha256"] for r in endpoints)
        evidence=qualification(plan)
        execution={**plan["training_execution"],"torch":evidence["torch"],"gpu":evidence["gpu"],
                   "deterministic_algorithms":True,"warn_only":False,"cublas_workspace_config":":4096:8",
                   "evidence_basis":"Recovered from attested source worker control flow: completed fitting and entered evaluation after the deterministic context exit assertion."}
        write_json(run / "result.json",{"status":"complete","arms":recovery["training_reports"],"model_hashes":hashes,
                   "noise_hashes":noises,"counts":plan["expected_counts"],
                   "counts_basis":"Unique scientific work reconstructed from frozen loops and complete saved records. Source cumulative runtime counters were lost on timeout; these are not measured combined job counters.",
                   "completion":{"source":recovery["source"],"source_endpoints":158,"new_endpoints":27,"exact_replays":31,
                                 "new_optimizer_updates":0,"runtime_counts":runtime_counts,"elapsed_seconds":time.perf_counter()-started},
                   "training_execution":execution,"historical_model_comparison":{"single":hashes["single"]==plan["historical_reference_model_hashes"]["single"]},
                   "evaluation_images":185,"reference_images":24,"new_teacher_queries":0,
                   "elapsed_seconds":time.perf_counter()-started,"elapsed_scope":"Evaluation completion job only; total allocated cost must include source timeout and other attempts.",
                   "output_hashes":{str(p.relative_to(run)):sha256(p) for p in run.rglob("*") if p.is_file() and
                                    (p.suffix==".pt" and "upstream" not in p.parts or p.parent==run and p.suffix==".jsonl")}})
    except Exception as exc:
        write_json(run / "failure.json",{"error":repr(exc),"elapsed_seconds":time.perf_counter()-started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--run",required=True)
    parser.add_argument("--source")
    args=parser.parse_args()
    prepare_completion(args.source,args.run) if args.stage=="prepare" else complete(args.run)
