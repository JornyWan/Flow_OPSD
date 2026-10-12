"""Descriptive, post-join diagnostic breakdown; does not change labels or selection."""
from statistics import median

from opdlab.action_transfer import OLD, FRESH
from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import visual


def build():
    campaign = ROOT / "experiments/local_expert_v16"
    report = read_json(campaign / "results.json")
    geometry, availability = {}, {}
    for cohort in (OLD, FRESH):
        cases = [r for r in report["cases"] if r["query"]["cohort"] == cohort]
        values = [r["query"]["action_metrics"][r["query"]["selected_teacher"]] for r in cases]
        geometry[cohort] = {"states": len(values), "metric_reference": "Same expert type selected for this prompt in phase15; fresh actions are not all safe targets",
                            **{k: {"median": median(v[k] for v in values), "min": min(v[k] for v in values), "max": max(v[k] for v in values)}
                               for k in ("relative_target_mse", "gamma", "orthogonal_mse_over_initial")}}
        bins = {"known_safe_available": [], "no_known_safe_but_unresolved": [], "all_teacher_comparisons_known_no_safe": []}
        for c in cases:
            teachers = [v for m, v in c["comparisons"].items() if m in ("selected_teacher", "ocr@1", "geneval@1")]
            name = "known_safe_available" if any(t["safe_repair"] for t in teachers) else "no_known_safe_but_unresolved" if any(t["has_unknown"] for t in teachers) else "all_teacher_comparisons_known_no_safe"
            bins[name].append(c["case"])
        availability[cohort] = {k: {"count": len(v), "cases": v} for k, v in bins.items()}
    case_table = []
    for c in sorted(report["cases"], key=lambda c: c["case"]):
        q = c["query"]
        case_table.append({"case": c["case"], "id": q["id"], "seed": q["seed"], "cohort": q["cohort"],
                           "complete": {m: visual(o)["complete_scene"] for m, o in c["observations"].items()},
                           "status": {m: v["status"] for m, v in c["comparisons"].items()},
                           "safe_teachers": c["safe_teachers"], "same_type_teacher_alignment": q["action_metrics"][q["selected_teacher"]]})
    return {"source_results_sha256": sha256(campaign / "results.json"), "analysis_script_sha256": sha256(__file__),
            "stage": "post-join descriptive mechanism analysis, no selection or new training",
            "action_geometry": geometry, "teacher_availability": availability, "cases": case_table,
            "conclusions": [
                "Fresh single-student and full-student have exactly the same complete-scene indicator on every state (2/12 each), not only equal aggregate counts.",
                "Fresh expert availability: 2 known-safe states, 4 unresolved, 6 with all teacher comparisons known and no safe repair. Two of the six already have complete initial scenes.",
                "Case5: GenEval fixes four cups to two, text remains wrong; learned single action keeps four cups, full learned image also loses visibility. Determinate local transfer failure with an available teacher action.",
                "Case18: both experts and local/full student repair PAC to PEACH, but all retain cropped chair legs. This is a clause repair, not a complete scene.",
                "Case8: single learned, full learned and GenEval damage correct BLOOM to BLOM; OCR preserves the complete baseline. Acceptance/preservation remains necessary.",
                "Old case12: learned single action repairs the cup/bottle relation, whereas full learned trajectory loses it; trajectory interference exists on this fitting seed.",
                "Old single-student has four determinate safe repairs and two uncertain texts. The two uncertain old targets are not proof of terminal repair failure despite low MSE.",
                "Fresh same-expert action alignment degrades, but matching all fresh expert actions would be inappropriate because many are not safe targets. Improve coverage and certify targets before fitting."
            ]}


if __name__ == "__main__":
    result = build()
    write_json(ROOT / "experiments/local_expert_v16/failure_analysis.json", result)
    print(result["action_geometry"])
    print(result["teacher_availability"])
