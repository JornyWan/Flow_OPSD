"""Post-review measurement audit; never changes candidates, labels or selection."""
from collections import Counter, defaultdict

from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import visual


def main():
    campaign = ROOT / "experiments/local_expert_v12"
    result = read_json(campaign / "results.json")
    frozen = read_json(campaign / "observations_frozen.json")
    assert frozen["frozen_before_key_join"]
    assert sha256(campaign / "observations.json") == frozen["observations_sha256"]
    candidates = {(e["case"], e["candidate"]["name"]): e["candidate"] for e in result["joined_review"]}
    confusion, harm_witnesses, accepted, groups = {}, [], [], {}
    for policy in ("noop", "ocr@1", "geneval@1", "selected"):
        counts = defaultdict(Counter)
        for q in result["queries"]:
            name = q["selected"] if policy == "selected" else policy
            c, obs = candidates[q["case"], name], q["observations"][name]
            assert len(c["details"]["clauses"]) == len(c["satisfied"])
            assert {x["id"] for x in c["details"]["clauses"]} == set(obs["clauses"])
            for clause, prediction in zip(c["details"]["clauses"], c["satisfied"]):
                label = obs["clauses"][clause["id"]]
                key = f"review_{str(label).lower()}_proxy_{str(prediction).lower()}"
                counts[clause["id"]][key] += 1
        confusion[policy] = {k: dict(v) for k, v in counts.items()}
    for q in result["queries"]:
        g = groups.setdefault(q["group"], Counter())
        g["queries"] += 1
        g["candidate_headroom"] += q["any_safe_repair"]
        if not q["accepted"]:
            continue
        comparison = q["selected_comparison"]
        g["accepted"] += 1
        g[comparison["status"]] += 1
        before, after = candidates[q["case"], "noop"], candidates[q["case"], q["selected"]]
        row = {"case": q["case"], "id": q["id"], "selected": q["selected"], "classification": comparison,
               "before": {"utility": before["utility"], "proxy_clauses": dict(zip(
                   [c["id"] for c in before["details"]["clauses"]], before["satisfied"])),
                   "ocr": before["details"]["ocr"]["recognized_text"], "review": q["observations"]["noop"]},
               "after": {"utility": after["utility"], "proxy_clauses": dict(zip(
                   [c["id"] for c in after["details"]["clauses"]], after["satisfied"])),
                   "ocr": after["details"]["ocr"]["recognized_text"], "review": q["observations"][q["selected"]]}}
        accepted.append(row)
        for clause in comparison["harmed_clauses"]:
            harm_witnesses.append({"case": q["case"], "clause": clause,
                "baseline_proxy_satisfied": row["before"]["proxy_clauses"][clause],
                "selected_proxy_satisfied": row["after"]["proxy_clauses"][clause],
                "baseline_review": True, "selected_review": False})
    complete_changes = []
    visibility_only = {name: [] for name in ("ocr@1", "geneval@1", "selected")}
    for q in result["queries"]:
        for policy in visibility_only:
            name = q["selected"] if policy == "selected" else policy
            if name == "noop":
                continue
            comparison = q["comparisons"][name]
            if comparison["safe_visibility_gain"] and not comparison["repaired_clauses"]:
                visibility_only[policy].append(q["case"])
        a = visual(q["observations"]["noop"])["complete_scene"]
        b = visual(q["observations"][q["selected"]])["complete_scene"]
        if a != b:
            complete_changes.append({"case": q["case"], "before": a, "after": b,
                                     "classification": q["selected_comparison"]["status"]})
    report = {"scope": "Post-unblind descriptive diagnosis against frozen implementing-assistant labels; no rule tuning or new validation",
              "results_sha256": sha256(campaign / "results.json"), "script_sha256": sha256(__file__),
              "proxy_review_confusion": confusion, "accepted_harmed_clause_witnesses": harm_witnesses,
              "accepted_harmed_clauses_outside_proxy_protection": sum(not r["baseline_proxy_satisfied"] for r in harm_witnesses),
              "accepted_queries": accepted, "groups": {k: dict(v) for k, v in groups.items()},
              "safe_visibility_only_gain_cases": visibility_only,
              "selected_complete_scene_changes": complete_changes,
              "limits": "Unknown review labels stay separate. Candidate rows share prompts; confusion counts are descriptive, not independent sample inference. Visibility and sign layout are absent from the frozen proxy clauses."}
    write_json(campaign / "measurement_diagnosis.json", report)
    print({"accepted_queries": len(accepted), "accepted_harmed_clauses": len(harm_witnesses),
           "outside_proxy_protection": report["accepted_harmed_clauses_outside_proxy_protection"],
           "complete_scene_changes": complete_changes})


if __name__ == "__main__":
    main()
