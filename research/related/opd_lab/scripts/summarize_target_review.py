"""Reveal the frozen all-accepted-pairs audit without changing observations or v1."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from opdlab.action_probe import verify_plan
from opdlab.io import read_json, sha256, write_json


def summarize(run, review):
    plan = verify_plan(run)
    frozen = read_json(review / "observations_frozen.json")
    if frozen["observations_sha256"] != sha256(review / "observations.json"):
        raise ValueError("Anonymous observations changed after freeze")
    if read_json(review / "provenance.json")["plan_sha256"] != sha256(run / "action_plan.json"):
        raise ValueError("Review plan changed")
    if frozen["key_opened"] or frozen["new_ray_results_inspected"]:
        raise ValueError("Observations were not frozen before reveal")
    obs, key = read_json(review / "observations.json"), read_json(review / "key.json")
    if [r["case"] for r in obs] != list(range(1,23)) or [r["id"] for r in key] != plan["review_ids"]:
        raise ValueError("Review coverage differs from predeclared queries")
    rows, preferences, transitions, disagreements = [], Counter(), Counter(), Counter()
    for o, k in zip(obs, key):
        if o["case"] != k["case"] or {m["action"] for m in k["mapping"].values()} != {"noop", "target"}:
            raise ValueError("Invalid review mapping")
        cids = [c["id"] for c in k["record"]["constraints"]]
        mapped = {}
        for side, m in k["mapping"].items():
            v = o["observations"][side]
            if set(v["clauses"]) != set(cids):
                raise ValueError("Constraint coverage changed")
            image = m["row"]["image"]
            if sha256(Path(plan["parent_run"]) / image) != plan["parent_files"][image]:
                raise ValueError("Reviewed parent image changed")
            mapped[m["action"]] = {"observation": v, "proxy": m["row"]}
            for cid, proxy in zip(cids, m["row"]["satisfied"]):
                seen = v["clauses"][cid]
                if seen is not None and seen != proxy:
                    disagreements[f"{cid}:proxy_{proxy}_visual_{seen}"] += 1
        preference = k["mapping"][o["better"]]["action"] + "_better" if o["better"] in ("A", "B") else o["better"]
        preferences[preference] += 1
        before, after = [mapped[a]["observation"] for a in ("noop", "target")]
        improved, regressed = [], []
        for cid in cids:
            a, b = before["clauses"][cid], after["clauses"][cid]
            transitions[f"{cid}:{a}->{b}"] += 1
            if a is False and b is True:
                improved.append(cid)
            if a is True and b is False:
                regressed.append(cid)
        rows.append({"case": o["case"], "id": k["id"], "preference": preference,
                     "proxy_utility_gain": mapped["target"]["proxy"]["utility"] - mapped["noop"]["proxy"]["utility"],
                     "target_proxy_joint_success": mapped["target"]["proxy"]["joint_success"],
                     "target_false_clauses": [c for c in cids if after["clauses"][c] is False],
                     "target_unknown_clauses": [c for c in cids if after["clauses"][c] is None],
                     "target_all_visual_clauses_true": all(after["clauses"][c] is True for c in cids),
                     "target_full_visibility": after["full_visibility"],
                     "improved_clauses": improved, "regressed_clauses": regressed,
                     "comparison_notes": o["comparison_notes"], "noop": before, "target": after})
    return {"scope": "Descriptive implementing-assistant review, not independent human truth. All 22 selected TRAIN pairs, with possible prior phase3 exposure. Known visual disagreements require adjudication; unchanged binary clauses alone do not invalidate progress gains.",
            "observations_frozen": frozen, "revealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preferences": dict(preferences), "clause_transitions": dict(transitions),
            "proxy_visual_clause_disagreements_44_images": dict(disagreements),
            "target_proxy_joint_success": sum(r["target_proxy_joint_success"] for r in rows),
            "target_visual_all_clauses_true": sum(r["target_all_visual_clauses_true"] for r in rows),
            "target_visibility_false": sum(r["target_full_visibility"] is False for r in rows),
            "target_with_unknown_clause": sum(bool(r["target_unknown_clauses"]) for r in rows),
            "target_proxy_joint_with_known_visual_clause_failure_cases": [r["case"] for r in rows if r["target_proxy_joint_success"] and r["target_false_clauses"]],
            "target_proxy_joint_with_visibility_failure_cases": [r["case"] for r in rows if r["target_proxy_joint_success"] and r["target_full_visibility"] is False],
            "cases_with_known_clause_improvement": [r["case"] for r in rows if r["improved_clauses"]],
            "cases_with_known_clause_regression": [r["case"] for r in rows if r["regressed_clauses"]],
            "rows": rows, "source_sha256": sha256(__file__), "key_sha256": sha256(review / "key.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True); p.add_argument("--review", required=True)
    args = p.parse_args()
    review = Path(args.review).resolve()
    if (review / "summary.json").exists():
        raise FileExistsError("Use an explicit new report version; do not overwrite a revealed audit")
    summary = summarize(Path(args.run).resolve(), review)
    write_json(review / "summary.json", summary)
    report = ["# Accepted-target image review", "", summary["scope"], "",
              f"Preferences: {summary['preferences']}", "",
              "| Case | Target/noop utility delta (v1) | Visual preference | Target known failed clauses | Target full visibility |",
              "|---|---:|---|---|---|"]
    for row in summary["rows"]:
        report.append(f"| {row['case']:02d} | {row['proxy_utility_gain']:.6f} | {row['preference']} | {', '.join(row['target_false_clauses']) or 'none recorded'} | {row['target_full_visibility']} |")
    report += ["", "An empty failure list may contain unknown clauses; inspect summary.json. Full visibility, text color and sign placement are not fully represented in v1. This review does not alter training selection or labels.", "",
               f"Known visual-clause failure despite target v1 joint success: cases {summary['target_proxy_joint_with_known_visual_clause_failure_cases']}.",
               f"Visibility failure despite target v1 joint success: cases {summary['target_proxy_joint_with_visibility_failure_cases']}."]
    (review / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows", "clause_transitions", "observations_frozen")}))
