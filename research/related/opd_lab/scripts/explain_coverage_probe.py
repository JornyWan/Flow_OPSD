"""Post-join diagnosis and cross-phase supervision inventory; no labels or runs change."""
from collections import Counter

import torch

from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import lines, visual


def build():
    out = ROOT / "experiments/local_expert_v17"
    current = read_json(out / "results.json")
    previous_path = ROOT / "experiments/local_expert_v16/results.json"
    previous = read_json(previous_path)
    bins = {k: [] for k in ("complete", "unknown", "known_encoded_failure", "visibility_only_failure")}
    for c in current["cases"]:
        base = c["observations"]["noop"]
        v = visual(base)
        name = "unknown" if v["has_unknown"] else "complete" if v["complete_scene"] else "known_encoded_failure" if not v["joint_clauses"] else "visibility_only_failure"
        bins[name].append(c["case"])

    def target(query_path, case, mode, comparison, source):
        q = torch.load(query_path, map_location="cpu", weights_only=False)
        assert comparison["safe_repair"] and not comparison["has_unknown"] and not comparison["known_harm"]
        return {"id": q["record"]["id"], "seed": q["seed"], "index": q["index"],
                "mode": mode, "source_case": case, "source_results": str(source), "source_results_sha256": sha256(source),
                "saved_query": str(query_path), "saved_query_sha256": sha256(query_path),
                "action_sha256": tensor_digest({"action": q["actions"][mode]}),
                "repaired_clauses": comparison["repaired_clauses"]}

    historical = []
    for c in previous["cases"]:
        q = c["query"]
        saved = ROOT / "runs/le16_transfer_001" / q["saved_query"]
        assert sha256(saved) == q["saved_query_sha256"]
        if q["cohort"] == "positive_fresh_seed":
            options = c["safe_teachers"]
            if not options:
                continue
            payload = torch.load(saved, map_location="cpu", weights_only=False)
            def rank(mode):
                displacement = (payload["actions"][mode] - payload["actions"]["noop"]).double().square().mean().item() * q["dt"] ** 2
                return -len(c["comparisons"][mode]["repaired_clauses"]), displacement, mode
            mode = min(options, key=rank)
            comparison = c["comparisons"][mode]
        else:
            mode, comparison = q["selected_teacher"], c["comparisons"]["selected_teacher"]
        entry = target(saved, c["case"], mode, comparison, previous_path)
        assert entry["action_sha256"] == q["action_hashes"][mode]
        assert q["prefix_sha256"] == q["suffix_sha256"] == current["gpu_result"]["student_sha256"]
        entry["origin"] = "phase16_fresh" if q["cohort"] == "positive_fresh_seed" else "phase12_original_replayed_in_phase16"
        historical.append(entry)
    assert len(historical) == 8
    inventories = {}
    new_queries = lines(ROOT / "runs/le17_coverage_001/queries.jsonl")
    for pool, label in (("expanded", "multi_time"), ("fixed_i3_a1", "fixed_i3")):
        entries = list(historical)
        for c in current["cases"]:
            mode = c["pools"][pool]["selected"]
            if mode == "noop":
                continue
            # The same state may select a different time in the fixed pool.
            index = int(mode.split("_")[0][1:])
            queries = [r for r in new_queries if r["pair_id"] == c["pair_id"] and r["index"] == index]
            assert len(queries) == 1
            q = queries[0]
            saved = ROOT / "runs/le17_coverage_001" / q["saved_query"]
            assert sha256(saved) == q["saved_query_sha256"]
            entry = target(saved, c["case"], mode, c["comparisons"][mode], out / "results.json")
            assert entry["action_sha256"] == q["action_hashes"][mode]
            entry["origin"] = "phase17_collection"
            entries.append(entry)
        assert len({(e["id"], e["seed"]) for e in entries}) == len(entries)
        counts = dict(sorted(Counter(e["id"] for e in entries).items()))
        inventories[label] = {"states": entries, "count": len(entries), "per_prompt_states": counts,
                              "prompts_with_multiple_states": [p for p, n in counts.items() if n >= 2]}
    assert inventories["multi_time"]["count"] == 11 and inventories["fixed_i3"]["count"] == 9
    assert all(e["index"] == 3 for e in inventories["fixed_i3"]["states"])
    return {"stage": "post-join descriptive inventory, not a frozen training protocol or automatic acceptance",
            "analysis_script_sha256": sha256(__file__), "source_results_sha256": sha256(out / "results.json"),
            "baseline_partition": {k: {"count": len(v), "cases": v} for k, v in bins.items()},
            "known_encoded_failure_safe_covered": [c["case"] for c in current["cases"] if c["case"] in bins["known_encoded_failure"] and c["pools"]["expanded"]["safe_available"]],
            "supervision_inventory": inventories,
            "selection_notes": "Keep original six targets; phase16 fresh candidates use repaired clauses, then transition MSE, then name. Phase17 uses its frozen pool choices. No observations changed. Historical states are development data, not independent tests.",
            "limitations": "Assistant labels; 11 mixed-time states cannot isolate a pure state-coverage effect. The fixed-index inventory has multiple states for only two prompts and changes teacher/task mix. Neither inventory includes preservation states or authorizes a method-effectiveness claim."}


if __name__ == "__main__":
    result = build()
    write_json(ROOT / "experiments/local_expert_v17/supervision_inventory.json", result)
    print(result["baseline_partition"])
    print({k: v["per_prompt_states"] for k, v in result["supervision_inventory"].items()})
