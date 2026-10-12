"""Use the existing hidden-identity review layout for the fixed acceptance comparison."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from opdlab.io import read_json, sha256, write_json
from opdlab.report import prompt_summary
import review_supervision as shared
from review_action_transfer import check_labels
from review_prospective import compare, lines, visual


def prepare(run, review, campaign):
    shared.prepare(run, review, campaign)
    inputs = read_json(review / "inputs.json")
    assert (inputs["cases"], inputs["views"], inputs["generated_endpoints"]) == (64, 256, 224)
    write_json(review / "analysis_attestation.json", {"script_sha256": sha256(__file__),
               "inputs_sha256": sha256(review / "inputs.json"), "primary_path": "single",
               "primary_cohort": "positive_fresh_seed", "primary_arm": "certified"})


def freeze(run, review, campaign):
    shared.freeze(run, review, campaign)
    assert read_json(review / "analysis_attestation.json")["script_sha256"] == sha256(__file__)
    write_json(review / "analysis_frozen.json", {"analysis_attestation_sha256": sha256(review / "analysis_attestation.json"),
               "observations_frozen_sha256": sha256(review / "observations_frozen.json"), "before_identity_join": True})


def pilot_checks(scores, contrasts):
    primary = contrasts["positive_fresh_seed"]["single"]
    return {"at_least2_more_complete_than_initial": primary["joint128"]["sum"] >= 2,
            "at_least1_more_complete_than_all_teacher": primary["all_teacher"]["sum"] >= 1,
            "at_least1_more_complete_than_random": primary["random_matched"]["sum"] >= 1,
            "zero_known_harm_all_fresh_local": scores["all_fresh"]["single"]["certified"]["known_harm"]["sum"] == 0,
            "preservation_local_not_worse": contrasts["preservation_fresh_seed"]["single"]["joint128"]["sum"] >= 0,
            "unfitted_local_not_worse": contrasts["unfitted_prompt_fresh_seed"]["single"]["joint128"]["sum"] >= 0}


def build_report(run, review):
    shared.validate(run)
    plan = read_json(run / "probe_plan.json")
    assert plan["stage"] == "evaluate"
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    analysis = read_json(review / "analysis_attestation.json")
    assert analysis["script_sha256"] == sha256(__file__) and analysis["inputs_sha256"] == sha256(review / "inputs.json")
    freeze = read_json(review / "analysis_frozen.json")
    assert freeze == {"analysis_attestation_sha256": sha256(review / "analysis_attestation.json"),
                      "observations_frozen_sha256": sha256(review / "observations_frozen.json"), "before_identity_join": True}
    assert frozen["frozen_before_key_join"] and inputs["result_sha256"] == sha256(run / "result.json")
    assert sha256(shared.__file__) == inputs["review_script_sha256"] == frozen["review_script_sha256"]
    assert frozen["rubric_sha256"] == sha256(run / "REVIEW_RUBRIC.md")
    for n in ("observations", "items", "key", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    observations = check_labels(review)
    assert observations == read_json(review / "observations.json")
    obs = {o["case"]: o["views"] for o in observations}
    values, direct, joined = defaultdict(list), defaultdict(list), []
    for item in read_json(review / "key.json"):
        models = {r["model"]: (r, obs[item["case"]][letter]) for letter, r in item["mapping"].items()}
        base, path = models["joint128"][1], item["path"]
        for model, (row, o) in models.items():
            comparison = compare(base, o)
            joined.append({"case": item["case"], "evaluated_path": path, "row": row, "observation": o, "comparison": comparison})
            metrics = {**visual(o), "safe_repair": comparison["safe_repair"], "known_harm": comparison["known_harm"], "comparison_unknown": comparison["has_unknown"]}
            for cohort in row["cohorts"]:
                for metric, v in metrics.items():
                    values[cohort, path, model, metric].append((row["id"], v))
                for clause, before in {**base["clauses"], "visibility": base["full_visibility"]}.items():
                    if before is True:
                        after = o["full_visibility"] if clause == "visibility" else o["clauses"][clause]
                        values[cohort,path,model,clause+"_true_to_false"].append((row["id"], after is False))
                        values[cohort,path,model,clause+"_true_to_unknown"].append((row["id"], after is None))
        row, selected = models["certified"]
        for other in ("joint128", "all_teacher", "random_matched"):
            for cohort in row["cohorts"]:
                direct[cohort,path,other].append((row["id"], int(visual(selected)["complete_scene"]) - int(visual(models[other][1])["complete_scene"])))
    def packed(data):
        result = {}
        for keys, rows in sorted(data.items()):
            target = result
            for k in keys[:-1]:
                target = target.setdefault(k,{})
            target[keys[-1]] = {**prompt_summary(rows), "sum": sum(float(v) for _,v in rows)}
        return result
    scores, contrasts = packed(values), packed(direct)
    fit_run = Path(plan["fit_run"])
    points = {p["key"]:p for p in plan["points"]}
    aligned, drift = defaultdict(list), defaultdict(list)
    for r in lines(fit_run / "metrics.jsonl"):
        p = points[r["key"]]
        if "alignment" in r:
            aligned[r["arm"],str(r["step"])].append(r["alignment"]["relative_target_mse"])
        else:
            drift[r["arm"],str(r["step"]),p["role"]].append(r["noop_drift_mse"])
    def medians(data):
        result = {}
        for keys, vals in sorted(data.items()):
            target = result
            for key in keys[:-1]:
                target=target.setdefault(key,{})
            target[keys[-1]]={"states":len(vals),"median":median(vals)}
        return result
    checks = pilot_checks(scores,contrasts)
    return {"scores":scores,"certified_vs_other_complete":contrasts,"accepted_alignment":medians(aligned),
            "noop_target_drift":medians(drift),"promising_pilot":{"checks":checks,"passed":all(checks.values())},
            "joined_review":joined,"review_frozen":frozen,"review_inputs":inputs,
            "fit_result":read_json(fit_run / "result.json"),"evaluation_result":read_json(run / "result.json"),
            "limitations":"Provisional implementing-assistant labels, familiar development prompts, one training seed. Fixed-query OPD-style comparison; no native-online, automatic verifier or sampler-robustness claim."}


def summarize(run, review, campaign):
    report=build_report(run,review)
    write_json(campaign / "learning_results.json",report)
    print({c:{path:{m:{k:s[k]["sum"] for k in ("complete_scene","known_harm","has_unknown")} for m,s in models.items()} for path,models in paths.items()} for c,paths in report["scores"].items()})
    print(report["promising_pilot"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("prepare","freeze","summarize"))
    parser.add_argument("--run",default="runs/le20_eval_001")
    parser.add_argument("--review",default="runs/le20_eval_review_001")
    parser.add_argument("--campaign",default="experiments/local_expert_v20")
    args=parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(),Path(args.review).resolve(),Path(args.campaign).resolve())
