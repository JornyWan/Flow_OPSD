"""Freeze all phase17 labels before evaluating candidate pools or saving targets."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.coverage_probe import candidate_specs
from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from review_action_transfer import check_labels
from review_prospective import compare, lines, visual


def pools():
    specs = candidate_specs()
    return {"fixed_i3_a1": [s["mode"] for s in specs if s["index"] == 3 and s["alpha"] == 1],
            "i1_a1": [s["mode"] for s in specs if s["index"] == 1 and s["alpha"] == 1],
            "i5_a1": [s["mode"] for s in specs if s["index"] == 5 and s["alpha"] == 1],
            "i3_all_strengths": [s["mode"] for s in specs if s["index"] == 3],
            "all_times_a1": [s["mode"] for s in specs if s["alpha"] == 1],
            "expanded": [s["mode"] for s in specs]}


def choose_target(candidates, comparisons):
    safe = [c for c in candidates if comparisons[c["mode"]]["safe_repair"]]
    return min(safe, key=lambda c: (-len(comparisons[c["mode"]]["repaired_clauses"]),
               c["transition_distance"], c["mode"])) if safe else None


def validate(run):
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert result["status"] == "complete" and plan["candidates"] == candidate_specs()
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    for p, h in result["output_hashes"].items():
        assert sha256(run / p) == h, p
    rows = lines(run / "endpoints.jsonl")
    expected = {(p["pair_id"], m) for p in plan["pairs"] for m in ["noop", *pools()["expanded"]]}
    assert len(rows) == len(expected) == 156
    assert {(r["pair_id"], r["mode"]) for r in rows} == expected
    for r in rows + lines(run / "references.jsonl"):
        assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
    return rows


def prepare(run, review):
    rows = validate(run)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    review.mkdir(parents=True, exist_ok=False)
    items, key = [], []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pid in enumerate(sorted(grouped, key=lambda p: digest(["phase17_case", p])), 1):
        group = sorted(grouped[pid], key=lambda r: digest(["phase17_slot", pid, r["mode"]]))
        record = records[group[0]["id"]]
        mapping, images = {}, {}
        for i, (letter, row) in enumerate(zip("ABCDEFGHIJKLM", group)):
            if i % 4 == 0:
                sheet = Image.new("RGB", (1024, 1200 if i < 12 else 690), "white")
                draw = ImageDraw.Draw(sheet)
                draw.text((6, 4), f"Case {case:02d}, page {i // 4 + 1}/4\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], review / name)
            x, y = (i % 2) * 512, 128 + ((i % 4) // 2) * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(review / name), (x, y + 24))
            mapping[letter], images[letter] = row, name
            if i % 4 == 3 or i == 12:
                sheet.save(review / f"case-{case:02d}-page{i // 4 + 1}.png")
        key.append({"case": case, "pair_id": pid, "mapping": mapping})
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
    write_json(review / "items.json", items)
    write_json(review / "key.json", key)
    write_json(review / "inherited.json", {})
    write_json(review / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("items", "key", "inherited")},
               "review_script_sha256": sha256(__file__), "cases": 12, "new_images": 156,
               "scope": "Assistant annotation, path/baseline/time/strength/seed/score hidden; known prompt family, not independent human review"})
    print("Prepared12 cases,48 native-resolution sheets,156 randomized views including baselines.")


def freeze(run, review):
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    observations = check_labels(review)
    assert len(observations) == 12 and all(len(o["views"]) == 13 for o in observations)
    write_json(review / "observations.json", observations)
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant, not independent human ground truth",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "items", "key", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"), "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md"),
               "review_script_sha256": sha256(__file__)})
    print("All156 observations frozen before key and score join.")


def build_report(run, review):
    rows = validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] and sha256(run / "result.json") == inputs["result_sha256"]
    assert sha256(__file__) == inputs["review_script_sha256"] == frozen["review_script_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    for n in ("observations", "items", "key", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    obs = {r["case"]: r["views"] for r in read_json(review / "observations.json")}
    qs = {(q["pair_id"], q["index"]): q for q in lines(run / "queries.jsonl")}
    cases, joined, targets = [], [], []
    metrics, candidate_metrics = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for entry in read_json(review / "key.json"):
        pid = entry["pair_id"]
        bymode = {r["mode"]: (r, obs[entry["case"]][letter]) for letter, r in entry["mapping"].items()}
        base = bymode["noop"][1]
        comparisons = {m: compare(base, o) for m, (_, o) in bymode.items()}
        candidates = [{**r, "transition_distance": r["action_distance"] * qs[(pid, r["index"])]["dt"]**2}
                      for m, (r, _) in bymode.items() if m != "noop"]
        identity = {k: bymode["noop"][0][k] for k in ("id", "seed", "replicate", "pair_id")}
        for letter, r in entry["mapping"].items():
            joined.append({"case": entry["case"], "view": letter, "row": r, "observation": obs[entry["case"]][letter]})
            values = {**visual(obs[entry["case"]][letter]), "safe_repair": comparisons[r["mode"]]["safe_repair"],
                      "known_harm": comparisons[r["mode"]]["known_harm"], "comparison_unknown": comparisons[r["mode"]]["has_unknown"]}
            for k, v in values.items():
                candidate_metrics[r["mode"]][k].append((identity["id"], v))
        summaries = {}
        for name, modes in pools().items():
            available = [c for c in candidates if c["mode"] in modes]
            selected = choose_target(available, comparisons)
            chosen = selected["mode"] if selected else "noop"
            values = {"safe_available": selected is not None,
                      "unresolved_without_safe": selected is None and any(comparisons[m]["has_unknown"] for m in modes),
                      "any_known_harm": any(comparisons[m]["known_harm"] for m in modes),
                      "oracle_complete_including_noop": any(visual(bymode[m][1])["complete_scene"] for m in ["noop", *modes]),
                      "safe_selected_complete": visual(bymode[chosen][1])["complete_scene"],
                      "safe_selected_complete_delta": int(visual(bymode[chosen][1])["complete_scene"]) - int(visual(base)["complete_scene"])}
            summaries[name] = {**values, "selected": chosen, "safe_modes": [m for m in modes if comparisons[m]["safe_repair"]]}
            for k, v in values.items():
                metrics[name][k].append((identity["id"], v))
        chosen = summaries["expanded"]["selected"]
        target = {**identity, "case": entry["case"], "accepted": chosen != "noop", "mode": chosen,
                  "supervision": "provisional assistant-certified development target; not automatic verifier selection"}
        if chosen != "noop":
            row = bymode[chosen][0]
            q = qs[(pid, row["index"])]
            target.update(index=row["index"], saved_query=str(run / q["saved_query"]), saved_query_sha256=q["saved_query_sha256"],
                          action_sha256=q["action_hashes"][chosen], endpoint_image=str(run / row["image"]),
                          endpoint_image_sha256=row["image_sha256"], repaired_clauses=comparisons[chosen]["repaired_clauses"])
        targets.append(target)
        cases.append({"case": entry["case"], **identity, "observations": {m: o for m, (_, o) in bymode.items()},
                      "comparisons": comparisons, "pools": summaries})
    scores = {p: {m: {**prompt_summary(v), "sum": sum(float(x) for _, x in v)} for m, v in ms.items()} for p, ms in metrics.items()}
    candidate_scores = {p: {m: {**prompt_summary(v), "sum": sum(float(x) for _, x in v)} for m, v in ms.items()} for p, ms in candidate_metrics.items()}
    coverage = {name: {pid: sum(c["pools"][name]["safe_available"] for c in cases if c["id"] == pid)
                       for pid in sorted({c["id"] for c in cases})} for name in pools()}
    delta = prompt_summary([(c["id"], int(c["pools"]["expanded"]["safe_available"]) - int(c["pools"]["fixed_i3_a1"]["safe_available"])) for c in cases])
    return {"cases": cases, "joined_review": joined, "pool_scores": scores, "candidate_scores": candidate_scores,
            "per_prompt_safe_states": coverage, "expanded_minus_fixed_safe_coverage": delta, "target_manifest": targets,
            "gpu_result": read_json(run / "result.json"), "review_frozen": frozen, "review_inputs": inputs,
            "limitations": "Six selected prompts, two collection seeds each, provisional assistant labels; expanded pool uses more compute; oracle availability/target selection is not a deployable verifier or trained method."}


def summarize(run, review):
    campaign = ROOT / "experiments/local_expert_v17"
    result = build_report(run, review)
    write_json(campaign / "results.json", result)
    write_json(campaign / "target_manifest.json", {"stage": "untrained provisional target pool", "source_results_sha256": sha256(campaign / "results.json"), "targets": result["target_manifest"]})
    for n in ("observations.json", "observations_frozen.json"):
        shutil.copyfile(review / n, campaign / n)
    print({p: {k: v[k]["sum"] for k in ("safe_available", "unresolved_without_safe", "safe_selected_complete")} for p, v in result["pool_scores"].items()})
    print(result["per_prompt_safe_states"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", default="runs/le17_coverage_001")
    parser.add_argument("--review", default="runs/le17_review_001")
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve())
