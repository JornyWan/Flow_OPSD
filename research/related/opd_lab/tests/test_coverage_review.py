from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_coverage_probe import choose_target, pools
from review_prospective import compare


def observation(a, b, visible=True):
    return {"clauses": {"text": a, "count": b}, "full_visibility": visible}


def test_target_selection_rejects_unknown_and_harm_then_ranks_terminal_repairs():
    base = observation(False, False)
    values = {"unclear": observation(True, None), "one": observation(True, False),
              "two": observation(True, True), "smaller": observation(True, False)}
    comparisons = {n: compare(base, v) for n, v in values.items()}
    candidates = [{"mode": n, "transition_distance": d} for n, d in (("unclear", 0), ("one", 1), ("two", 2), ("smaller", .5))]
    assert choose_target(candidates, comparisons)["mode"] == "two"
    assert choose_target([c for c in candidates if c["mode"] != "two"], comparisons)["mode"] == "smaller"
    assert choose_target(candidates[:1], comparisons) is None
    harm = compare(observation(False, True), observation(True, False))
    assert choose_target([{"mode": "harm", "transition_distance": 0}], {"harm": harm}) is None


def test_fixed_and_expanded_pool_accounting():
    sets = pools()
    assert list(map(len, sets.values())) == [2, 2, 2, 4, 6, 12]
    assert set(sets["fixed_i3_a1"]) <= set(sets["all_times_a1"]) <= set(sets["expanded"])
    assert set(sets["i3_all_strengths"]) <= set(sets["expanded"])
    assert len(set(sets["i1_a1"]) | set(sets["fixed_i3_a1"]) | set(sets["i5_a1"])) == 6


def test_report_clusters_seeds_and_exports_only_certified_targets(tmp_path, monkeypatch):
    import json
    import review_coverage_probe as review
    from opdlab.io import read_json, write_json, sha256
    run, out = tmp_path / "run", tmp_path / "review"
    run.mkdir(); out.mkdir()
    (run / "REVIEW_RUBRIC.md").write_text("fixture")
    write_json(run / "result.json", {})
    items, key, observations, queries = [], [], [], []
    for case in range(12):
        views, mapping, images = {}, {}, {}
        for letter, mode in zip("ABCDEFGHIJKLM", ["noop", *pools()["expanded"]]):
            name = f"{case}-{letter}.png"
            (out / name).write_bytes(name.encode())
            images[letter] = name
            o = observation(False, None if case % 2 else True)
            if mode == "i1_ocr@0.5":
                o["clauses"]["text"] = True
            o.update(visible_text="synthetic", notes="fixture")
            views[letter] = o
            index = int(mode[1]) if mode != "noop" else None
            mapping[letter] = {"mode": mode, "id": str(case // 2), "seed": case, "replicate": case % 2,
                               "pair_id": str(case), "index": index, "action_distance": .1, "image": name, "image_sha256": sha256(out / name)}
        for i in (1,3,5):
            queries.append({"pair_id": str(case), "index": i, "dt": -.1, "saved_query": "fixture.pt", "saved_query_sha256": "synthetic",
                            "action_hashes": {m: "synthetic" for m in pools()["expanded"]}})
        items.append({"case": case, "images": images, "constraints": [{"id":"text"},{"id":"count"}]})
        key.append({"case": case, "pair_id": str(case), "mapping": mapping})
        observations.append({"case": case, "views": views})
    for n, value in (("items",items),("key",key),("inherited",{})):
        write_json(out / f"{n}.json",value)
    write_json(out / "inputs.json", {"result_sha256":sha256(run / "result.json"),"review_script_sha256":sha256(review.__file__)})
    (out / "observations_draft.jsonl").write_text("\n".join(json.dumps(o) for o in observations)+"\n")
    (run / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries)+"\n")
    review.freeze(run,out)
    monkeypatch.setattr(review,"validate",lambda run: [])
    result=review.build_report(run,out)
    assert result["pool_scores"]["fixed_i3_a1"]["safe_available"]["sum"] == 0
    assert result["pool_scores"]["expanded"]["safe_available"]["sum"] == 6
    assert result["pool_scores"]["expanded"]["unresolved_without_safe"]["sum"] == 6
    assert result["expanded_minus_fixed_safe_coverage"]["mean"] == .5
    assert result["expanded_minus_fixed_safe_coverage"]["prompts"] == 6
    assert all(v==1 for v in result["per_prompt_safe_states"]["expanded"].values())
    assert sum(t["accepted"] for t in result["target_manifest"]) == 6
    assert all(t["mode"]=="i1_ocr@0.5" for t in result["target_manifest"] if t["accepted"])
