"""Replay all frozen phase14 responses and compare with unchanged development labels."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from opdlab.evidence_dominance import admissible_repair
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.paired_audit import (OBSERVATION_PROMPT, SPEC, agreed_endpoints, choose_pairs,
                                 endpoints, guarded_endpoints, literal_label, pair_request,
                                 parse_observation, parse_pair)
from opdlab.report import prompt_summary
from review_prospective import compare, lines, ratio, visual


POLICIES = ("forward", "reverse", "pair_consensus", "primary")


def replay(response, stage, ids=()):
    parsed, errors = parse_observation(response["raw"]) if stage == "observation" else parse_pair(response["raw"], ids)
    if response["truncated"]:
        parsed, _ = parse_observation("null") if stage == "observation" else parse_pair("null", ids)
        errors.append("Generation reached token cap")
    assert parsed == response["parsed"] and errors == response["errors"]
    assert response["truncated"] == (response["output_tokens"] >= SPEC[stage + "_max_new_tokens"])
    return parsed


def policy_metrics(queries, policy):
    accepted = [q["selections"][policy] != "noop" for q in queries]
    comparisons = [compare(q["observations"]["noop"], q["observations"][q["selections"][policy]]) for q in queries]
    classes = Counter(c["status"] for a, c in zip(accepted, comparisons) if a)
    values, transitions = defaultdict(list), defaultdict(Counter)
    for q in queries:
        base, chosen = q["observations"]["noop"], q["observations"][q["selections"][policy]]
        vm, bm = visual(chosen), visual(base)
        for k, v in vm.items():
            values[k].append((q["id"], v))
        values["complete_scene_delta"].append((q["id"], int(vm["complete_scene"]) - int(bm["complete_scene"])))
        for k, before in {**base["clauses"], "visibility": base["full_visibility"]}.items():
            if before is True:
                after = chosen["full_visibility"] if k == "visibility" else chosen["clauses"][k]
                transitions[k]["denominator"] += 1
                transitions[k]["false"] += after is False
                transitions[k]["unknown"] += after is None
    return {"accepted_classes": dict(classes), "acceptance": ratio(accepted, [True] * 64),
            "safe_precision": ratio([a and c["safe_repair"] for a, c in zip(accepted, comparisons)], accepted),
            "safe_recall": ratio([a and c["safe_repair"] for a, c in zip(accepted, comparisons)], [q["any_safe_repair"] for q in queries]),
            "known_harm_rate": ratio([a and c["known_harm"] for a, c in zip(accepted, comparisons)], accepted),
            "unresolved_rate": ratio([a and c["status"] == "unknown" for a, c in zip(accepted, comparisons)], accepted),
            "endpoints": {k: {**prompt_summary(v), "sum": sum(float(x) for _, x in v)} for k, v in values.items()},
            "protected_transitions": {k: dict(v) for k, v in transitions.items()}}


def build_report(run, *, diagnostic_pair_parser=None):
    run = Path(run).resolve()
    plan, result, a = (read_json(run / n) for n in ("paired_plan.json", "result.json", "attestation.json"))
    assert result["status"] == "complete" and plan["spec"] == SPEC
    assert sha256(run / "paired_plan.json") == a["plan_sha256"] == result["plan_sha256"]
    assert sha256(run / "PROTOCOL.md") == a["protocol_sha256"]
    assert sha256(run / "private_key.json") == a["private_key_sha256"]
    for p, h in plan["input_files"].items():
        assert sha256(p) == h, p
    for p, h in plan["framework_files"].items():
        assert sha256(run / p) == h, p
    for r in plan["images"]:
        assert sha256(r["image"]) == r["image_sha256"]
    assert len({r["image_sha256"] for r in plan["images"]}) == 192
    observations, pairs = lines(run / "observations.jsonl"), lines(run / "pairs.jsonl")
    assert sha256(run / "observations.jsonl") == result["observations_sha256"] == read_json(run / "observations_frozen.json")["sha256"]
    assert read_json(run / "observations_frozen.json")["before_any_pair_requests"] is True
    assert sha256(run / "pairs.jsonl") == result["pairs_sha256"]
    assert len(observations) == 192 and len(pairs) == 320
    counts = Counter({k: 0 for k in result["counts"]})
    byimage, bypair = {}, defaultdict(dict)
    all_fields, errors, grids, parsing = Counter(), Counter(), Counter(), Counter()
    specs = {p["uid"]: p for p in plan["pairs"]}
    for stage, rows, expected in (("observation", observations, plan["images"]), ("pair", pairs, plan["tasks"])):
        for row, task in zip(rows, expected):
            assert row["uid"] == task["uid"]
            response = row["response"]
            ids = [] if stage == "observation" else [c["id"] for c in specs[task["pair_uid"]]["clauses"]]
            parsed = replay(response, stage, ids)
            analysis_errors = response["errors"]
            if stage == "pair" and diagnostic_pair_parser is not None:
                parsed, analysis_errors = diagnostic_pair_parser(response["raw"], ids)
                if response["truncated"]:
                    parsed, _ = parse_pair("null", ids)
                    analysis_errors.append("Generation reached token cap")
            if stage == "observation":
                assert response["request"] == OBSERVATION_PROMPT
                byimage[row["uid"]] = parsed
            else:
                assert {k: row[k] for k in task} == task
                assert response["request"] == pair_request(specs[task["pair_uid"]]["clauses"], [byimage[i] for i in task["images"]])
                bypair[task["pair_uid"]][task["reverse"]] = parsed
                for k, field in parsed.items():
                    all_fields[k + ":total"] += 1
                    all_fields[k + ":unknown"] += field["A"] is None or field["B"] is None
            counts["calls"] += 1
            counts[stage + "_calls"] += 1
            counts["image_presentations"] += 1 if stage == "observation" else 2
            for k in ("input_tokens", "output_tokens"):
                counts[k] += response[k]
            counts["responses_with_errors"] += bool(response["errors"])
            counts["truncated_outputs"] += response["truncated"]
            errors.update(analysis_errors)
            parsing[stage + "_responses_with_errors"] += bool(analysis_errors)
            parsing[stage + "_responses"] += 1
            grids.update(str(g) for g in response["image_grid_thw"])
    assert dict(counts) == result["counts"] and counts["image_presentations"] == 832
    prior = {q["id"]: q for q in read_json(ROOT / "experiments/local_expert_v12/results.json")["queries"]}
    old = {q["id"]: q for q in read_json(ROOT / "experiments/local_expert_v13/results.json")["queries"]}
    posthoc = {q["id"]: q for q in read_json(ROOT / "experiments/local_expert_v13/posthoc_diagnosis.json")["queries"]}
    source_records = {r["id"]: r for r in lines(ROOT / "runs/le12_collect_002/data.jsonl")}
    spec_by_images = {tuple(p["images"]): p for p in plan["pairs"]}
    queries, pair_rows, self_rows = [], [], []
    order, self_counts, literal_confusion = Counter(), Counter(), Counter()
    for key in read_json(run / "private_key.json"):
        q = prior[key["id"]]
        assert q["seed"] == key["seed"]
        model_pairs, literals = {p: {} for p in POLICIES}, {}
        for name, image_id in key["images"].items():
            literals[name] = byimage[image_id]
        word = next(c["text"] for c in source_records[q["id"]]["constraints"] if c["kind"] == "ocr")
        literal = {n: literal_label(o, word) for n, o in literals.items()}
        for name, label in literal.items():
            literal_confusion[f'review_{q["observations"][name]["clauses"]["text"]}_prediction_{label}'] += 1
        for name in ("ocr@1", "geneval@1"):
            spec = spec_by_images[key["images"]["noop"], key["images"][name]]
            forward, reverse = bypair[spec["uid"]][False], bypair[spec["uid"]][True]
            model_pairs["forward"][name] = endpoints(forward)
            model_pairs["reverse"][name] = endpoints(reverse, True)
            model_pairs["pair_consensus"][name] = agreed_endpoints(forward, reverse)
            model_pairs["primary"][name] = guarded_endpoints(model_pairs["pair_consensus"][name], [literal["noop"], literal[name]])
            eligible = {p: admissible_repair(*v[name]) for p, v in model_pairs.items()}
            field_order = {}
            for k in forward:
                x, y = (forward[k]["A"], forward[k]["B"]), (reverse[k]["B"], reverse[k]["A"])
                known = all(v is not None for v in (*x, *y))
                agrees = x == y
                order["fields"] += 1
                order["shared_determinate_fields"] += known
                order["shared_determinate_agreements"] += known and agrees
                order["agreements_including_unknown"] += agrees
                order["definite_conflicts"] += any(a is not None and b is not None and a != b for a, b in zip(x, y))
                field_order[k] = {"forward": x, "reverse_canonical": y, "known": known, "agrees": agrees}
            order["pairs"] += 1
            order["eligibility_agreements"] += eligible["forward"] == eligible["reverse"]
            pair_rows.append({"case": q["case"], "id": q["id"], "name": name, "pair_uid": spec["uid"],
                              "reference": compare(q["observations"]["noop"], q["observations"][name]),
                              "eligible": eligible, "evidence": {p: v[name] for p, v in model_pairs.items()},
                              "literal_labels": [literal["noop"], literal[name]], "order": field_order,
                              "forward_response": forward, "reverse_response": reverse})
        spec = spec_by_images[key["images"]["noop"], key["images"]["noop"]]
        parsed = bypair[spec["uid"]][False]
        for f in parsed.values():
            self_counts["fields"] += 1
            self_counts["determinate_fields"] += f["A"] is not None and f["B"] is not None
            self_counts["repair_fields"] += f["change"] == "repair"
            self_counts["damage_fields"] += f["change"] == "damage"
        raw_eligible = admissible_repair(*endpoints(parsed))
        self_counts["pairs"] += 1
        self_counts["eligible_pairs"] += raw_eligible
        self_rows.append({"case": q["case"], "id": q["id"], "pair_uid": spec["uid"], "raw_eligible": raw_eligible, "response": parsed})
        selections = {p: choose_pairs(v, key["distances"]) for p, v in model_pairs.items()}
        selections.update(old[q["id"]]["selections"])
        selections["phase13_posthoc_dominance"] = posthoc[q["id"]]["selected"]
        queries.append({"case": q["case"], "id": q["id"], "seed": q["seed"], "group": q["group"],
                        "observations": q["observations"], "literal_observations": literals,
                        "literal_labels": literal, "selections": selections, "any_safe_repair": q["any_safe_repair"]})
    queries.sort(key=lambda q: q["case"])
    policies = {p: policy_metrics(queries, p) for p in queries[0]["selections"]}
    candidate_metrics = {}
    for p in POLICIES:
        arrays = defaultdict(list)
        confusion = Counter()
        for q in queries:
            rows = [r for r in pair_rows if r["id"] == q["id"]]
            assert len(rows) == 2
            arrays["eligible"].append(sum(r["eligible"][p] for r in rows))
            arrays["safe"].append(sum(r["eligible"][p] and r["reference"]["safe_repair"] for r in rows))
            arrays["harm"].append(sum(r["eligible"][p] and r["reference"]["known_harm"] for r in rows))
            arrays["unknown"].append(sum(r["eligible"][p] and r["reference"]["status"] == "unknown" for r in rows))
            arrays["headroom"].append(sum(r["reference"]["safe_repair"] for r in rows))
            for r in rows:
                predicted = compare(*r["evidence"][p])
                confusion[f'review_{r["reference"]["status"]}_prediction_{predicted["status"]}'] += 1
        candidate_metrics[p] = {"eligible": ratio(arrays["eligible"], [2] * 64), "safe_precision": ratio(arrays["safe"], arrays["eligible"]),
                                "safe_recall": ratio(arrays["safe"], arrays["headroom"]), "known_harm_rate": ratio(arrays["harm"], arrays["eligible"]),
                                "unknown_rate": ratio(arrays["unknown"], arrays["eligible"]), "status_confusion": dict(confusion)}
    primary = policies["primary"]
    gate = {"at_least4_safe_states": primary["safe_precision"]["numerator"] >= 4,
            "safe_precision_at_least80pct": (primary["safe_precision"]["mean"] or 0) >= .8,
            "zero_selected_known_harm": primary["known_harm_rate"]["numerator"] == 0,
            "selected_unknown_at_most10pct": primary["unresolved_rate"]["mean"] is not None and primary["unresolved_rate"]["mean"] <= .1,
            "zero_self_changes": self_counts["repair_fields"] + self_counts["damage_fields"] == 0,
            "self_determinate_coverage_at_least80pct": self_counts["determinate_fields"] / self_counts["fields"] >= .8,
            "order_shared_determinate_coverage_at_least80pct": order["shared_determinate_fields"] / order["fields"] >= .8,
            "order_agreement_at_least90pct": order["shared_determinate_fields"] > 0 and order["shared_determinate_agreements"] / order["shared_determinate_fields"] >= .9,
            "complete_attested_coverage": True}
    return {"scope": plan["scope"] if diagnostic_pair_parser is None else "posthoc container-only parsing diagnostic on the same exposed development responses; not the frozen primary analysis",
            "run": str(run), "gpu_result": result, "policies": policies,
            "candidate_metrics": candidate_metrics, "queries": queries, "pairs": pair_rows, "self_pairs": self_rows,
            "order_control": dict(order), "self_control": dict(self_counts), "literal_confusion": dict(literal_confusion),
            "pair_field_unknown_counts": dict(all_fields), "parse_errors": dict(errors), "analysis_parsing": dict(parsing), "image_grids": dict(grids),
            "development_readiness": {"checks": gate, "passed": all(gate.values())},
            "forward_reverse_choice_agreements": sum(q["selections"]["forward"] == q["selections"]["reverse"] for q in queries),
            "script_sha256": sha256(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    report = build_report(parser.parse_args().run)
    write_json(ROOT / "experiments/local_expert_v14/results.json", report)
    print({p: {"accepted": m["acceptance"]["numerator"], "classes": m["accepted_classes"],
               "complete": m["endpoints"]["complete_scene"]["sum"]} for p, m in report["policies"].items()})
    print(report["development_readiness"])
