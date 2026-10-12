"""Evaluate a frozen measurement prototype against immutable development labels."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

from opdlab.evidence_verifier import choose
from opdlab.io import ROOT,read_json,sha256,write_json
from opdlab.report import prompt_summary
from review_prospective import compare,lines,ratio,visual


def summarize(run):
    campaign = ROOT / "experiments/local_expert_v13"
    old = read_json(ROOT / "experiments/local_expert_v12/results.json")
    plan,gpu = read_json(run / "audit_plan.json"),read_json(run / "result.json")
    assert gpu["status"] == "complete" and sha256(run / "evidence.jsonl") == gpu["evidence_sha256"]
    for path,h in read_json(run / "audit_attestation.json")["inputs"].items():
        assert sha256(path) == h,path
    rows = lines(run / "evidence.jsonl")
    assert len(rows) == len(plan["queries"]) == len(old["queries"]) == 64
    prior = {(q["id"],q["seed"]):q for q in old["queries"]}
    queries,confusion = [],defaultdict(lambda:defaultdict(Counter))
    for spec,row in zip(plan["queries"],rows):
        assert (row["id"],row["seed"]) == (spec["id"],spec["seed"])
        q = prior[row["id"],row["seed"]]
        distances = {c["name"]:c["distance"] for c in spec["candidates"]}
        assert choose(row["evidence"],distances) == row["decision"]
        v1 = {c["name"]:{"clauses":dict(zip([x["id"] for x in c["v1"]["details"]["clauses"]],c["v1"]["satisfied"])),
                           "full_visibility":True} for c in spec["candidates"]}
        discrete = choose(v1,distances)
        assert row["v1_selected"] == spec["v1_selected"] == q["selected"]
        selections = {"noop":"noop","v1":q["selected"],"v1_discrete_event":discrete["selected"],"v2":row["decision"]["selected"]}
        comparisons = {p:compare(q["observations"]["noop"],q["observations"][n]) for p,n in selections.items()}
        queries.append({"case":q["case"],"id":q["id"],"seed":q["seed"],"group":q["group"],
                        "selections":selections,"comparisons":comparisons,"any_safe_repair":q["any_safe_repair"],
                        "v2_decision":row["decision"],"discrete_decision":discrete,
                        "observations":q["observations"]})
        for name,e in row["evidence"].items():
            obs = q["observations"][name]
            for k,label in {**obs["clauses"],"visibility":obs["full_visibility"]}.items():
                prediction = e["full_visibility"] if k == "visibility" else e["clauses"][k]
                confusion["v2"][k][f"review_{str(label).lower()}_prediction_{str(prediction).lower()}"] += 1
                if k != "visibility":
                    prediction = v1[name]["clauses"][k]
                    confusion["v1"][k][f"review_{str(label).lower()}_prediction_{str(prediction).lower()}"] += 1
    policies = {}
    # Preserve phase12's input order for identical finite-resample v1 intervals.
    stats_queries = sorted(queries,key=lambda q:q["case"])
    for p in ("noop","v1","v1_discrete_event","v2"):
        accepted = [q["selections"][p] != "noop" for q in stats_queries]
        safe = [a and q["comparisons"][p]["safe_repair"] for a,q in zip(accepted,stats_queries)]
        harm = [a and q["comparisons"][p]["known_harm"] for a,q in zip(accepted,stats_queries)]
        unknown = [a and q["comparisons"][p]["status"] == "unknown" for a,q in zip(accepted,stats_queries)]
        classes = Counter(q["comparisons"][p]["status"] for a,q in zip(accepted,stats_queries) if a)
        values,changes = defaultdict(list),defaultdict(Counter)
        for q in queries:
            obs,base = q["observations"][q["selections"][p]],q["observations"]["noop"]
            vm,bm = visual(obs),visual(base)
            for k,v in vm.items():
                values[k].append((q["id"],v))
            values["complete_scene_delta"].append((q["id"],int(vm["complete_scene"])-int(bm["complete_scene"])))
            for k,before in {**base["clauses"],"visibility":base["full_visibility"]}.items():
                if before is True:
                    after = obs["full_visibility"] if k == "visibility" else obs["clauses"][k]
                    changes[k]["denominator"] += 1
                    changes[k]["false"] += after is False
                    changes[k]["unknown"] += after is None
        policies[p] = {"accepted_classes":dict(classes),"acceptance":ratio(accepted,[True]*64),
                       "safe_precision":ratio(safe,accepted),"known_harm_rate":ratio(harm,accepted),
                       "unresolved_rate":ratio(unknown,accepted),"safe_recall":ratio(safe,[q["any_safe_repair"] for q in stats_queries]),
                       "endpoints":{k:{**prompt_summary(v),"sum":sum(float(x) for _,x in v)} for k,v in values.items()},
                       "protected_transitions":{k:dict(v) for k,v in changes.items()}}
    p = policies["v2"]
    gate = {"at_least4_safe_repairs_retained":p["safe_precision"]["numerator"]>=4,
            "precision_at_least80pct":p["safe_precision"]["mean"] is not None and p["safe_precision"]["mean"]>=.8,
            "zero_known_harms":p["known_harm_rate"]["numerator"]==0,
            "unresolved_at_most10pct":p["unresolved_rate"]["mean"] is not None and p["unresolved_rate"]["mean"]<=.1,
            "zero_raw_replay_mismatches":gpu["raw_replay_mismatches"]==0}
    reasons = Counter(reason for q in queries for d in q["v2_decision"]["candidates"].values() for reason in d["reasons"])
    result = {"scope":"development audit on all192 previously reviewed phase12 images; no new-seed validation or policy training",
              "run":str(run),"gpu_result":gpu,"policies":policies,"queries":queries,"confusion":{p:{k:dict(c) for k,c in v.items()} for p,v in confusion.items()},
              "candidate_rejection_reason_counts":dict(reasons),"development_readiness":{"checks":gate,"passed":all(gate.values())},
              "source_results_sha256":sha256(ROOT / "experiments/local_expert_v12/results.json"),"script_sha256":sha256(__file__)}
    write_json(campaign / "results.json",result)
    print({p:{"accepted":v["acceptance"]["numerator"],"classes":v["accepted_classes"],
              "complete_scene":v["endpoints"]["complete_scene"]["sum"]} for p,v in policies.items()})
    print({"development_readiness":gate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    summarize(Path(parser.parse_args().run).resolve())
