"""Join logged targets and post-update audits; these are training-state diagnostics."""
import argparse
import json
from pathlib import Path

from opdlab.io import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if read_json(run / "result.json")["status"] != "complete":
        raise ValueError("A completed training run is required")
    inputs = sorted(run.glob("queries-rank-*.jsonl")) + sorted(run.glob("postfit-rank-*.jsonl"))
    key = lambda r: (r["outer_epoch"], r["id"], r["seed"], r["query_index"])
    queries, post = {}, {}
    for path in inputs:
        target = queries if path.name.startswith("queries-") else post
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if key(row) in target:
                raise ValueError("Duplicate query identity")
            target[key(row)] = row
    if not queries or queries.keys() != post.keys():
        raise ValueError("Missing query or post-update evidence")
    rows = []
    for identity in sorted(queries):
        before, after = queries[identity], post[identity]
        base = before["candidates"][0]
        selected = next(c for c in before["candidates"] if c["name"] == before["selected"])
        if (any(not c["valid"] for c in before["candidates"])
                or not after["postfit_single_intervention"]["valid"]
                or not after["postfit_full_policy_train_prompt"]["valid"]):
            raise ValueError("Invalid candidate or post-update verdict")
        # At collection, the student's weights equal the frozen behavior's.
        # Candidate distance is mean((target_velocity - behavior_velocity)^2).
        pre_loss = 0.5 * before["dt"] ** 2 * selected["distance"]
        rows.append({"id": before["id"], "outer_epoch": before["outer_epoch"],
                     "seed": before["seed"], "query_index": before["query_index"],
                     "accepted": before["accepted"], "selected": before["selected"],
                     "pre_update_target_loss": pre_loss,
                     "post_update_target_loss": after["local_target_loss"],
                     "baseline_utility": base["utility"], "target_utility": selected["utility"],
                     "postfit_local_utility": after["postfit_single_intervention"]["utility"],
                     "postfit_full_train_utility": after["postfit_full_policy_train_prompt"]["utility"]})
    epochs = {}
    for epoch in sorted({r["outer_epoch"] for r in rows}):
        accepted = [r for r in rows if r["outer_epoch"] == epoch and r["accepted"]]
        pre = sum(r["pre_update_target_loss"] for r in accepted)
        after = sum(r["post_update_target_loss"] for r in accepted)
        epochs[str(epoch)] = {"accepted": len(accepted), "accepted_loss_sum_before": pre,
                              "accepted_loss_sum_after": after,
                              "after_over_before": after / pre if pre else None}
    result = {"run": str(run), "input_sha256": {p.name: sha256(p) for p in inputs},
              "interpretation": "Training-state fitting audit; repeated states and selecting verifier. No held-out efficacy claim or causal diagnosis of learning rate.",
              "loss_definition": "0.5 * dt^2 * velocity MSE, before versus after that outer epoch's updates",
              "epochs": epochs, "rows": rows}
    write_json(out, result)
    print(json.dumps({"run": str(run), "epochs": epochs}))


if __name__ == "__main__":
    main()
