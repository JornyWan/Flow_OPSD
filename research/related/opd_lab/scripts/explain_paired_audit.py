"""Explain frozen decisions without proposing or selecting a new policy."""
import argparse
from collections import Counter, defaultdict

from opdlab.evidence_dominance import admissible_repair
from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import compare, visual
from summarize_paired_audit import POLICIES


def reasons(pair):
    a, b = pair
    av, bv = {**a["clauses"], "full_visibility": a["full_visibility"]}, {**b["clauses"], "full_visibility": b["full_visibility"]}
    repaired = [k for k in a["clauses"] if av[k] is False and bv[k] is True]
    unknown = [k for k, v in bv.items() if v is None]
    unprotected = [k for k in av if av[k] is not False and bv[k] is not True]
    tags = (["no_confirmed_encoded_repair"] if not repaired else [])
    tags += ["candidate_unknown:" + k for k in unknown]
    tags += ["baseline_success_not_preserved:" + k for k in unprotected]
    assert (not tags) == admissible_repair(a, b)
    return {"repaired": repaired, "rejection_reasons": tags}


def main(array_diagnostic=False):
    campaign = ROOT / "experiments/local_expert_v14"
    source = campaign / ("array_diagnostic.json" if array_diagnostic else "results.json")
    result = read_json(source)
    queries = {q["id"]: q for q in result["queries"]}
    confusion = {p: defaultdict(Counter) for p in POLICIES}
    reason_counts = {p: Counter() for p in POLICIES}
    rows = []
    for pair in result["pairs"]:
        q = queries[pair["id"]]
        explanation = {}
        for policy in POLICIES:
            evidence = pair["evidence"][policy]
            explanation[policy] = reasons(evidence)
            reason_counts[policy].update(explanation[policy]["rejection_reasons"])
            for name, e in zip(("noop", pair["name"]), evidence):
                reference = q["observations"][name]
                for k, label in {**reference["clauses"], "full_visibility": reference["full_visibility"]}.items():
                    predicted = e["full_visibility"] if k == "full_visibility" else e["clauses"][k]
                    confusion[policy][k][f"review_{label}_prediction_{predicted}"] += 1
        rows.append({"case": q["case"], "id": q["id"], "name": pair["name"],
                     "reference": pair["reference"], "eligible": pair["eligible"], "explanation": explanation,
                     "selected_primary": q["selections"]["primary"] == pair["name"],
                     "literal_labels": pair["literal_labels"], "literal_observations":
                     [q["literal_observations"][n] for n in ("noop", pair["name"])],
                     "reference_observations": [q["observations"][n] for n in ("noop", pair["name"])],
                     "forward_response": pair["forward_response"], "reverse_response": pair["reverse_response"]})
    rows.sort(key=lambda r: (r["case"], r["name"]))
    groups = {}
    for group in sorted({q["group"] for q in queries.values()}):
        qs = [q for q in queries.values() if q["group"] == group]
        ps = [r for r in rows if queries[r["id"]]["group"] == group]
        groups[group] = {"states": len(qs), "states_with_safe_candidate": sum(q["any_safe_repair"] for q in qs),
                         "safe_candidates": sum(p["reference"]["safe_repair"] for p in ps),
                         "primary_selected_classes": dict(Counter(p["reference"]["status"] for p in ps if p["selected_primary"]))}
    out = {"scope": "descriptive error attribution for frozen policies; no posthoc replacement selector",
           "results_source": source.name, "results_sha256": sha256(source), "script_sha256": sha256(__file__),
           "endpoint_confusion_denominator": "256 pair-context endpoint instances per policy; the64 noops occur twice; not256 unique images",
           "endpoint_confusion": {p: {k: dict(c) for k, c in v.items()} for p, v in confusion.items()},
           "rejection_reason_counts": {p: dict(c) for p, c in reason_counts.items()}, "by_group": groups,
           "frozen_review_endpoint_headroom": {
               "states": len(queries),
               "baseline_complete": sum(visual(q["observations"]["noop"])["complete_scene"] for q in queries.values()),
               "best_available_complete_including_unknown_baselines": sum(any(visual(o)["complete_scene"] for o in q["observations"].values()) for q in queries.values()),
               "complete_with_only_known_safe_repairs": sum(visual(q["observations"]["noop"])["complete_scene"] or
                   any(compare(q["observations"]["noop"], o)["safe_repair"] and visual(o)["complete_scene"] for o in q["observations"].values()) for q in queries.values()),
               "meaning": "Label-based ceiling among these fixed endpoints; not an executable policy or trained model performance"},
           "all_pairs": rows,
           "safe_candidate_cases": [r for r in rows if r["reference"]["safe_repair"]],
           "primary_false_or_unknown_accepts": [r for r in rows if r["selected_primary"] and not r["reference"]["safe_repair"]]}
    write_json(campaign / ("array_failure_analysis.json" if array_diagnostic else "failure_analysis.json"), out)
    print({"known_safe_candidates": len(out["safe_candidate_cases"]),
           "primary_false_or_unknown_accepts": len(out["primary_false_or_unknown_accepts"]), "groups": groups})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--array-diagnostic", action="store_true")
    main(parser.parse_args().array_diagnostic)
