"""Bounded continuation: successful fit -> audited evaluation -> unlabelled review pages."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from opdlab.io import ROOT, append_jsonl, read_json, sha256, write_json

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def completed_result(run, state, exit_code):
    """A failed allocation or missing worker result never authorizes the next stage."""
    state = state.split()[0].rstrip("+")
    if state not in TERMINAL:
        return False
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(f"{run.name}: allocation {state} {exit_code}; no automatic retry")
    result=read_json(run/"result.json")
    if result.get("status") != "complete" or any((run/n).exists() for n in ("RUNNING","failure.json")):
        raise RuntimeError(f"{run.name}: allocation completed without a clean worker result")
    return True


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--control",required=True)
    args=parser.parse_args()
    control=Path(args.control).resolve()
    plan=read_json(control/"plan.json")
    campaign, fit=Path(plan["campaign"]),Path(plan["fit_run"])
    evaluation, review=Path(plan["evaluation_run"]),Path(plan["review"])
    deadline=time.monotonic()+plan["deadline_minutes"]*60
    env={**os.environ,"PYTHONPATH":str(ROOT/"python")+":"+str(ROOT/"scripts"),"PYTHONNOUSERSITE":"1"}
    with (control/"RUNNING").open("x") as f:f.write(str(os.getpid())+"\n")

    def state(status,**details):
        now=datetime.now(timezone.utc).isoformat()
        row={"time_utc":now,"event":"acceptance_continuation","status":status,**details}
        append_jsonl(campaign/"ledger.jsonl",row)
        write_json(control/"progress.json",row)
        progress=read_json(campaign/"progress.json")
        progress.update(updated_utc=now,status=status,**details)
        write_json(campaign/"progress.json",progress)
        write_json(control/"latest.json",row)
        (campaign/"LIVE.md").write_text(f"# Phase20实时进度\n\n更新时间：{now}\n\n状态：`{status}`\n\n"+"\n".join(f"- {k}: {v}" for k,v in details.items())+"\n\n收集结论见COLLECTION.md。当前没有经过冻结图像复核的学生有效性结论；不能从训练loss或作业完成状态推断成功。\n")
        print(row,flush=True)

    def check_sources():
        if (control/"STOP").exists():raise RuntimeError("Explicit STOP requested")
        for p,h in plan["inputs"].items():
            if sha256(p)!=h:raise RuntimeError(f"Continuation source changed: {p}")

    def command(script,*args):
        check_sources()
        subprocess.run([sys.executable,str(ROOT/"scripts"/script),*map(str,args)],cwd=ROOT,env=env,check=True)

    def wait(run):
        job=read_json(run/"submission.json")["job_id"]
        while time.monotonic()<deadline:
            check_sources()
            rows=subprocess.check_output(["sacct","-X","-j",job,"--noheader","--parsable2","--format=JobIDRaw,State,ExitCode"],text=True)
            for line in rows.splitlines():
                fields=line.split("|")
                if fields[0].strip()==job and completed_result(run,fields[1],fields[2]):return job
            time.sleep(30)
        raise TimeoutError("Bounded continuation deadline reached; inspect existing jobs, do not automatically resubmit")

    try:
        check_sources()
        state("fitting",fit_run=str(fit),fit_job_id=read_json(fit/"submission.json")["job_id"])
        wait(fit)
        state("verifying_fit",fit_run=str(fit))
        command("verify_acceptance_learning.py","fit","--fit-run",fit,"--campaign",campaign)
        assert not evaluation.exists() and not review.exists()
        command("prepare_acceptance_learning.py","evaluate","--fit-run",fit,"--attempt",plan["evaluation_attempt"])
        assert evaluation.exists()
        ep=read_json(evaluation/"probe_plan.json")
        assert ep["fit_run"]==str(fit) and ep["evaluation_images"]==224
        paths={Path(p) for p in read_json(evaluation/"probe_attestation.json")["inputs"]}
        paths.update(Path(p) for p in plan["inputs"])
        paths.update(p for p in evaluation.rglob("*") if p.is_file() and "upstream" not in p.parts and "__pycache__" not in p.parts and not p.is_symlink() and p.suffix in (".py",".json",".md"))
        write_json(campaign/"evaluation_preparation_index.json",{"scope":"Frozen evaluation assets, exact fitted models, review/analysis code before new evaluation images.","artifacts_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}})
        command("submit_slurm.py","--run",evaluation,"--stage","le20_acceptance_evaluate","--gres","gpu:l40:1","--time","00:20:00","--ledger",campaign/"ledger.jsonl","--","env","CUBLAS_WORKSPACE_CONFIG=:4096:8",sys.executable,"-m","opdlab.acceptance_learning","--run",evaluation)
        job=read_json(evaluation/"submission.json")["job_id"]
        state("evaluating",fit_complete=True,fit_optimizer_updates=384,evaluation_run=str(evaluation),evaluation_job_id=job,evaluation_images_planned=224)
        wait(evaluation)
        command("refresh_campaign.py","--campaign",campaign)
        command("verify_acceptance_learning.py","outputs","--fit-run",fit,"--run",evaluation,"--campaign",campaign)
        command("review_acceptance_learning.py","prepare","--run",evaluation,"--review",review,"--campaign",campaign)
        state("awaiting_blind_review",fit_complete=True,fit_optimizer_updates=384,evaluation_images_generated=224,review=str(review),review_pages=64,review_slots=256,student_results_available=False)
        write_json(control/"result.json",{"status":"complete","scope":"Fit and evaluation verified; unlabelled review prepared. No annotations, identity join or scientific pass decision.","evaluation_job_id":job})
    except Exception as exc:
        state("continuation_stopped",error=repr(exc),automatic_retry=False)
        write_json(control/"failure.json",{"error":repr(exc),"automatic_retry":False})
        raise
    finally:
        (control/"RUNNING").unlink()


if __name__=="__main__":main()
