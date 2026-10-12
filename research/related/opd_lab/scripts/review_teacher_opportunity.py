"""Freeze new teacher-image observations, then compare opportunity with the learned action."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
from statistics import median
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from opdlab.teacher_opportunity import pairs_from_plan, rows
from review_action_transfer import check_labels
from review_prospective import compare, visual

MODES = ("noop", "student_single", "ocr@1", "geneval@1")


def validate(run):
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    assert result["status"] == "complete" and result["optimizer_updates"] == 0
    assert not any((run / n).exists() for n in ("RUNNING", "failure.json"))
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    for p, h in result["output_hashes"].items():
        assert sha256(run / p) == h, p
    assert plan["pairs"] == pairs_from_plan(read_json(Path(plan["source"]) / "probe_plan.json"))
    for k, v in plan["expected_counts"].items():
        assert result["counts"].get(k, 0) == v
    refs, endpoints = rows(run / "references.jsonl"), rows(run / "endpoints.jsonl")
    for entries, modes in ((refs, MODES[:2]), (endpoints, MODES[2:])):
        assert len(entries) == 24
        assert {(r["pair_id"], r["mode"]) for r in entries} == {(p["pair_id"], m) for p in plan["pairs"] for m in modes}
        for r in entries:
            assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
            assert r["noise_sha256"] == plan["references"][r["pair_id"]]["noop"]["noise_sha256"]
    for r in refs:
        assert r["exact"] and r["image_sha256"] == plan["references"][r["pair_id"]][r["mode"]]["image_sha256"]
    gate = read_json(run / "replay_gate.json")
    assert gate["all_exact"] and gate["reference_images"] == 24 and gate["states"] == 12
    assert all(gate["counts_before_teachers"].get("forward_" + t, 0) == 0 for t in ("ocr", "geneval"))
    assert len(rows(run / "queries.jsonl")) == 12
    return refs + endpoints


def prepare(run, review, campaign):
    entries = validate(run)
    records = {r["id"]: r for r in rows(run / "data.jsonl")}
    known = {e["row"]["image_sha256"]: e["observation"] for e in
             read_json(ROOT / "experiments/local_expert_v18/results.json")["joined_review"]}
    grouped = defaultdict(list)
    for r in entries:
        grouped[r["pair_id"]].append(r)
    review.mkdir(parents=True, exist_ok=False)
    items, key, inherited = [], [], {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pid in enumerate(sorted(grouped, key=lambda p: digest(["opportunity_case", p])), 1):
        group = sorted(grouped[pid], key=lambda r: digest(["opportunity_slot", pid, r["mode"]]))
        record = records[group[0]["id"]]
        sheet = Image.new("RGB", (1024, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
        mapping, images = {}, {}
        for i, (letter, r) in enumerate(zip("ABCD", group)):
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / r["image"], review / name)
            x, y = i % 2 * 512, 128 + i // 2 * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(review / name), (x, y + 24))
            mapping[letter], images[letter] = r, name
            if r["image_sha256"] in known:
                inherited.setdefault(str(case), {})[letter] = known[r["image_sha256"]]
        sheet.save(review / f"case-{case:02d}.png")
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
        key.append({"case": case, "pair_id": pid, "mapping": mapping})
    for n, obj in (("items", items), ("key", key), ("inherited", inherited)):
        write_json(review / f"{n}.json", obj)
    write_json(review / "inputs.json", {"run": str(run), "cases": 12, "views": 48,
               "new_teacher_images": 24, "inherited_views": sum(map(len, inherited.values())),
               "result_sha256": sha256(run / "result.json"), "review_script_sha256": sha256(__file__),
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("items", "key", "inherited")},
               "scope": "Teacher identity and scores hidden for24 new images;24 reference images already exposed and frozen. Implementing assistant, not independent annotation."})
    print({"cases": 12, "views": 48, "inherited": sum(map(len, inherited.values()))})


def freeze(run, review, campaign):
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    obs = check_labels(review)
    assert len(obs) == 12 and sum(len(o["views"]) for o in obs) == 48
    write_json(review / "observations.json", obs)
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant; provisional development labels",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "items", "key", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"),
               "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md"), "review_script_sha256": sha256(__file__)})
    print("Frozen48 observations before new teacher identity and action-metric joins.")


def diagnose(observations):
    base, local = observations["noop"], observations["student_single"]
    comparisons = {m: compare(base, o) for m, o in observations.items()}
    complete = {m: visual(o)["complete_scene"] for m, o in observations.items()}
    safe = [m for m in MODES[2:] if comparisons[m]["safe_repair"]]
    safe_complete = [m for m in safe if complete[m]]
    local_known = not comparisons["student_single"]["has_unknown"]
    return comparisons, {"any_teacher_safe": bool(safe), "any_teacher_safe_complete": bool(safe_complete),
        "teacher_safe_local_determinate_not_safe": bool(safe) and local_known and not comparisons["student_single"]["safe_repair"],
        "teacher_safe_complete_local_determinate_incomplete": bool(safe_complete) and local_known and not complete["student_single"],
        "teacher_safe_only_partial": bool(safe) and not safe_complete,
        "teacher_unresolved_without_known_safe": not safe and any(comparisons[m]["has_unknown"] for m in MODES[2:]),
        "oracle_complete_including_noop": any(complete[m] for m in ("noop", *MODES[2:])),
        "safe_complete_or_noop": complete["noop"] or bool(safe_complete)}


def build_report(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] and inputs["result_sha256"] == sha256(run / "result.json")
    assert inputs["review_script_sha256"] == frozen["review_script_sha256"] == sha256(__file__)
    assert frozen["rubric_sha256"] == sha256(run / "REVIEW_RUBRIC.md")
    for n in ("observations", "items", "key", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
        if n != "observations":
            assert inputs[f"{n}_sha256"] == frozen[f"{n}_sha256"]
    assert check_labels(review) == read_json(review / "observations.json")
    obs = {o["case"]: o["views"] for o in read_json(review / "observations.json")}
    queries = {q["pair_id"]: q for q in rows(run / "queries.jsonl")}
    buckets, flags, cases, joined = defaultdict(list), defaultdict(list), [], []
    for item in read_json(review / "key.json"):
        bymode = {r["mode"]: obs[item["case"]][letter] for letter, r in item["mapping"].items()}
        comparisons, signals = diagnose(bymode)
        q = queries[item["pair_id"]]
        for letter, r in item["mapping"].items():
            o, c = bymode[r["mode"]], comparisons[r["mode"]]
            joined.append({"case": item["case"], "view": letter, "row": r, "observation": o, "comparison": c})
            metrics = {**visual(o), "safe_repair": c["safe_repair"], "known_harm": c["known_harm"], "comparison_unknown": c["has_unknown"]}
            for k, v in metrics.items():
                buckets[r["mode"], k].append((r["id"], v))
        for k, v in signals.items():
            flags[k].append((q["id"], v))
        cases.append({"case": item["case"], "query": q, "observations": bymode, "comparisons": comparisons, "signals": signals})
    scores = {}
    for (mode, metric), values in buckets.items():
        scores.setdefault(mode, {})[metric] = {**prompt_summary(values), "sum": sum(v for _, v in values)}
    return {"scores": scores, "signals": {k: {**prompt_summary(v), "sum": sum(x for _, x in v)} for k, v in flags.items()},
            "action_mse_medians": {m: median(q["action_metrics"][m]["relative_target_mse"] for q in queries.values()) for m in MODES[2:]},
            "cases": cases, "joined_review": joined, "gpu_result": read_json(run / "result.json"), "review_frozen": frozen,
            "limitations": "Exposed development seeds, six prompts, provisional assistant labels. Teacher oracle opportunity is not learned performance or an automatic selector. Unknown is not evidence of teacher failure."}


def summarize(run, review, campaign):
    report = build_report(run, review)
    write_json(campaign / "results.json", report)
    for n in ("observations.json", "observations_frozen.json"):
        shutil.copyfile(review / n, campaign / n)
    print({m: {k: v["sum"] for k, v in s.items() if k in ("complete_scene", "safe_repair", "known_harm", "has_unknown")} for m, s in report["scores"].items()})
    print({k: v["sum"] for k, v in report["signals"].items()})
    print(report["action_mse_medians"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", default="runs/le19_opportunity_001")
    parser.add_argument("--review", default="runs/le19_review_001")
    parser.add_argument("--campaign", default="experiments/local_expert_v19")
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve(), Path(args.campaign).resolve())
