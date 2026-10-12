"""Posthoc ablation and retained evidence, explicitly separate from frozen v2."""
from collections import Counter
from pathlib import Path

from opdlab.evidence_dominance import VERSION,choose_dominance
from opdlab.io import ROOT,read_json,sha256,write_json
from review_prospective import compare,lines,ratio,visual


def main():
    campaign = ROOT / "experiments/local_expert_v13"
    result = read_json(campaign / "results.json")
    run = Path(result["run"])
    plan = read_json(run / "audit_plan.json")
    evidence = {(r["id"],r["seed"]):r for r in lines(run / "evidence.jsonl")}
    specs = {(r["id"],r["seed"]):r for r in plan["queries"]}
    records,examples = [],[]
    for q in sorted(result["queries"],key=lambda q:q["case"]):
        row,spec = evidence[q["id"],q["seed"]],specs[q["id"],q["seed"]]
        name = choose_dominance(row["evidence"],{c["name"]:c["distance"] for c in spec["candidates"]})
        c = compare(q["observations"]["noop"],q["observations"][name])
        records.append({"case":q["case"],"id":q["id"],"selected":name,"accepted":name!='noop',"comparison":c,
                        "has_safe_candidate":q["any_safe_repair"],"complete_scene":visual(q["observations"][name])["complete_scene"],
                        "frozen_v2_selected":q["selections"]["v2"]})
        if q["any_safe_repair"] or q["selections"]["v2"] != "noop" or q["comparisons"]["v1"]["known_harm"]:
            details = {}
            for n,e in row["evidence"].items():
                o = e["evidence"]["ocr"]
                details[n] = {"clauses":e["clauses"],"full_visibility":e["full_visibility"],
                              "ocr":{k:[{"text":x["text"],"confidence":x["confidence"]} for x in o[k]]
                                     for k in ("rows","scaled_rows","reread","scaled_reread")},
                              "object_reasons":[[{"id":d["id"],"strict":d["strict"],"reason":d["reason"]} for d in view]
                                                for view in e["evidence"]["objects"]],"review":q["observations"][n]}
            examples.append({"case":q["case"],"id":q["id"],"selections":q["selections"],"details":details})
    accepted = [r["accepted"] for r in records]
    safe = [r["accepted"] and r["comparison"]["safe_repair"] for r in records]
    report = {"scope":"posthoc development ablation after frozen v2 failed readiness; no new GPU inference or validation",
              "version":VERSION,"results_sha256":sha256(campaign / "results.json"),"script_sha256":sha256(__file__),
              "selector_sha256":sha256(ROOT / "python/opdlab/evidence_dominance.py"),"queries":records,"examples":examples,
              "accepted_classes":dict(Counter(r["comparison"]["status"] for r in records if r["accepted"])),
              "safe_precision":ratio(safe,accepted),"safe_recall":ratio(safe,[r["has_safe_candidate"] for r in records]),
              "complete_scene":sum(r["complete_scene"] for r in records),
              "changed_selection_cases":[r["case"] for r in records if r["selected"]!=r["frozen_v2_selected"]],
              "interpretation_limit":"The logical guarantee assumes the evidence sets contain the truth; correlated confidently wrong readings violate that premise."}
    write_json(campaign / "posthoc_diagnosis.json",report)
    print({k:report[k] for k in ("accepted_classes","safe_precision","safe_recall","complete_scene","changed_selection_cases")})


if __name__ == "__main__":
    main()
