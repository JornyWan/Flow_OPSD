"""After the GPU job terminates, prepare blinded review or record its actual failure."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from opdlab.io import append_jsonl, read_json, sha256, write_json
from opdlab.training_replay import validate_training_replay
from review_prospective import lines
from review_supervision import prepare, validate
from verify_supervision_completion import validate_completion


def finish(run, review, campaign):
    now = datetime.now(timezone.utc).isoformat()
    plan = read_json(run / "probe_plan.json")
    progress = {"updated_utc": now, "active_run": str(run), "review": str(review),
                "training_states": len(plan["points"]), "evaluation_images_planned": plan["evaluation_images"],
                "results_available": False, "scientific_analysis_complete": False}
    if not (run / "result.json").exists():
        failure = read_json(run / "failure.json") if (run / "failure.json").exists() else {"error": "GPU job ended without a complete result; inspect Slurm accounting/log."}
        progress.update(status="gpu_attempt_incomplete", failure=failure)
    else:
        result = read_json(run / "result.json")
        validate(run)
        if "completion" in plan:
            progress["completion"] = validate_completion(run)
        refs = lines(run / "references.jsonl")
        assert len(refs) == plan["reference_images"] and all(r["exact"] for r in refs)
        for key, value in plan["expected_counts"].items():
            assert result["counts"].get(key, 0) == value
        progress["training_replay"] = validate_training_replay(plan, result)
        if review.exists():
            assert read_json(review / "inputs.json")["result_sha256"] == sha256(run / "result.json")
        else:
            prepare(run, review, campaign)
        progress.update(status="ready_for_image_review", reference_images_exact=len(refs),
                        training_completed=True, optimizer_updates={a:r["optimizer_updates"] for a,r in result["arms"].items()},
                        evaluation_images_generated=result["evaluation_images"], review_sheets=read_json(review / "inputs.json")["cases"],
                        observations_frozen=(review / "observations_frozen.json").exists())
    # This stage never invents labels or treats raw verifier scores as findings.
    write_json(campaign / "progress.json", progress)
    append_jsonl(campaign / "ledger.jsonl", {"time_utc":now,"event":"postprocess_finished","run":str(run),"status":progress["status"],"scientific_analysis_complete":False})
    print(progress["status"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--review",required=True)
    parser.add_argument("--campaign",required=True)
    args=parser.parse_args()
    run,review,campaign=(Path(p).resolve() for p in (args.run,args.review,args.campaign))
    try:
        finish(run,review,campaign)
    except Exception as exc:
        write_json(campaign/"postprocess_failure.json",{"error":repr(exc),"run":str(run),"scientific_analysis_complete":False})
        raise
