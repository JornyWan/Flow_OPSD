"""New Phase18 attempt with qualified deterministic fitting and unchanged scientific design."""
import argparse
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare, verify_run
from opdlab.training_replay import MODE, qualification


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", default="003")
    args = parser.parse_args()
    campaign = ROOT / "experiments/local_expert_v18"
    source = ROOT / "runs/le18_supervision_001"
    diagnostic = ROOT / "runs/le18_repeatability_001"
    verify_run(source)
    verify_run(diagnostic)
    inputs = {**read_json(source / "probe_attestation.json")["inputs"],
              **read_json(diagnostic / "probe_attestation.json")["inputs"]}
    for p, h in inputs.items():
        assert sha256(p) == h, p
    plan = read_json(source / "probe_plan.json")
    plan["historical_reference_model_hashes"] = plan.pop("reference_model_hashes")
    plan["training_execution"] = {"mode": MODE, "qualification_result": str(diagnostic / "result.json"),
                                  "qualification_sha256": sha256(diagnostic / "result.json")}
    qualification(plan)
    run = prepare(Experiment.load(source / "config.json"), ROOT / f"runs/le18_supervision_{args.attempt}",
                  read_json(source / "site.json"))
    write_json(run / "probe_plan.json", plan)
    shutil.copyfile(source / "seed_inventory.json", run / "seed_inventory.json")
    shutil.copyfile(source / "REVIEW_RUBRIC.md", run / "REVIEW_RUBRIC.md")
    (run / "PROTOCOL.md").write_text((source / "PROTOCOL.md").read_text() + "\n\n" +
                                      (campaign / "PROTOCOL_REPLAY.md").read_text())
    paths = [Path(__file__).resolve(), diagnostic / "result.json", diagnostic / "probe_attestation.json",
             source / "probe_plan.json", campaign / "PROTOCOL_REPLAY.md"]
    paths.extend(run / n for n in ("probe_plan.json", "seed_inventory.json", "PROTOCOL.md", "REVIEW_RUBRIC.md"))
    # Preserve the qualifying code and measured updates as well as its result.
    paths.extend(p for p in diagnostic.rglob("*") if p.is_file() and "upstream" not in p.parts and
                 ("framework" in p.parts and p.suffix == ".py" or p.name in {"updates.jsonl", "step002.pt"}))
    inputs.update({str(p): sha256(p) for p in paths})
    write_json(run / "probe_attestation.json", {"inputs": inputs})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "source_attempt": str(source),
               "plan_sha256": sha256(run / "probe_plan.json"), "numerical_protocol": plan["training_execution"],
               "unchanged_scientific_fields": [k for k in plan if k not in {"historical_reference_model_hashes", "training_execution"}],
               "training_query_states": 15, "backward_queries": 4608, "evaluation_images": 185,
               "new_evaluation_seeds": 32, "new_teacher_queries": 0})
    verify_run(run)
    print(run)


if __name__ == "__main__":
    main()
