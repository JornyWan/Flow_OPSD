"""Path-hidden phase16 observations, then explicit joins to inherited references."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.action_transfer import OLD, FRESH
from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from review_prospective import compare, lines, visual


def validate(run):
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert result["status"] == "complete"
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    for p, h in result["output_hashes"].items():
        assert sha256(run / p) == h, p
    rows = lines(run / "endpoints.jsonl")
    expected = {(p["pair_id"], n) for p in plan["pairs"] for n in
                (("student_single",) if p["cohort"] == OLD else ("student_single", "ocr@1", "geneval@1"))}
    assert len(rows) == len(expected) == 42
    assert {(r["pair_id"], r["mode"]) for r in rows} == expected
    for r in rows + lines(run / "references.jsonl"):
        assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
    return rows


def inherited_labels():
    known = {}
    for phase in (12, 15):
        for entry in read_json(ROOT / f"experiments/local_expert_v{phase}/results.json")["joined_review"]:
            row = entry["candidate"] if phase == 12 else entry["row"]
            h, obs = row["image_sha256"], entry["observation"]
            if h in known:
                assert (known[h]["clauses"], known[h]["full_visibility"]) == (obs["clauses"], obs["full_visibility"])
            known[h] = obs
    return known


def prepare(run, review):
    rows, known = validate(run), inherited_labels()
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    review.mkdir(parents=True, exist_ok=False)
    items, key, inherited = [], [], {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pid in enumerate(sorted(grouped, key=lambda p: digest(["phase16_case", p])), 1):
        group = sorted(grouped[pid], key=lambda r: digest(["phase16_slot", pid, r["mode"]]))
        record = records[group[0]["id"]]
        sheet = Image.new("RGB", (1024, 1200 if len(group) == 3 else 690), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
        mapping, images = {}, {}
        for i, (letter, row) in enumerate(zip("ABC", group)):
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], review / name)
            x, y = (i % 2) * 512, 128 + (i // 2) * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(review / name), (x, y + 24))
            mapping[letter], images[letter] = row, name
            if row["image_sha256"] in known:
                inherited.setdefault(str(case), {})[letter] = known[row["image_sha256"]]
        sheet.save(review / f"case-{case:02d}.png")
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
        key.append({"case": case, "pair_id": pid, "mapping": mapping})
    write_json(review / "items.json", items)
    write_json(review / "key.json", key)
    write_json(review / "inherited.json", inherited)
    write_json(review / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("key", "items", "inherited")},
               "cases": len(items), "new_images": len(rows), "inherited_new_pixel_matches": sum(len(v) for v in inherited.values()),
               "caveat": "New path identities hidden; prior phase15 images exposed; assistant review, not independent human annotation."})
    print({"cases": len(items), "new_images": len(rows), "inherited_matches": sum(len(v) for v in inherited.values())})


def check_labels(review):
    items, inherited = read_json(review / "items.json"), read_json(review / "inherited.json")
    observations = lines(review / "observations_draft.jsonl")
    labels = {o["case"]: o for o in observations}
    assert len(labels) == len(observations) == len(items)
    assert set(labels) == {i["case"] for i in items}
    seen = {}
    for item in items:
        views = labels[item["case"]]["views"]
        assert set(views) == set(item["images"])
        for letter, obs in views.items():
            assert set(obs) == {"clauses", "full_visibility", "visible_text", "notes"}
            assert set(obs["clauses"]) == {c["id"] for c in item["constraints"]}
            assert all(type(v) is bool or v is None for v in [*obs["clauses"].values(), obs["full_visibility"]])
            assert isinstance(obs["visible_text"], str) and isinstance(obs["notes"], str)
            semantic = (obs["clauses"], obs["full_visibility"])
            old = inherited.get(str(item["case"]), {}).get(letter)
            if old:
                assert semantic == (old["clauses"], old["full_visibility"]), "Changed inherited pixels"
            h = sha256(review / item["images"][letter])
            if h in seen:
                assert semantic == seen[h], "Inconsistent duplicate pixels"
            seen[h] = semantic
    return sorted(observations, key=lambda o: o["case"])


def freeze(run, review):
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    observations = check_labels(review)
    assert len(observations) == 18 and sum(len(o["views"]) for o in observations) == 42
    write_json(review / "observations.json", observations)
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant; partially blinded, provisional",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "key", "items", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"), "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md")})
    print("All42 new observations frozen before identity join.")


def build_report(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] and sha256(run / "result.json") == inputs["result_sha256"]
    for n in ("observations", "key", "items", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
        if n != "observations":
            assert frozen[f"{n}_sha256"] == inputs[f"{n}_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    labels = {o["case"]: o for o in read_json(review / "observations.json")}
    known, queries = inherited_labels(), {r["pair_id"]: r for r in lines(run / "queries.jsonl")}
    plan = read_json(run / "probe_plan.json")
    refs = defaultdict(dict)
    for r in lines(run / "references.jsonl"):
        refs[r["pair_id"]][r["mode"]] = r
    joined, cases, buckets = [], [], defaultdict(lambda: defaultdict(list))
    for item in read_json(review / "key.json"):
        pid = item["pair_id"]
        observations = {m: known[r["image_sha256"]] for m, r in refs[pid].items()}
        for m, r in refs[pid].items():
            joined.append({"case": item["case"], "row": r, "observation": observations[m], "inherited": True})
        for letter, row in item["mapping"].items():
            observations[row["mode"]] = labels[item["case"]]["views"][letter]
            joined.append({"case": item["case"], "view": letter, "row": row, "observation": observations[row["mode"]], "inherited": False})
        q = queries[pid]
        baseline = observations["joint128"]
        comparisons = {m: compare(baseline, obs) for m, obs in observations.items()}
        for m, obs in observations.items():
            c = comparisons[m]
            values = {**visual(obs), "safe_repair": c["safe_repair"], "known_harm": c["known_harm"],
                      "comparison_unknown": c["has_unknown"],
                      "complete_scene_delta": int(visual(obs)["complete_scene"]) - int(visual(baseline)["complete_scene"])}
            for name, value in values.items():
                buckets[(q["cohort"], m)][name].append((q["id"], value))
        teachers = [m for m in observations if m in ("selected_teacher", "ocr@1", "geneval@1")]
        tsafe = [m for m in teachers if comparisons[m]["safe_repair"]]
        local, full = comparisons["student_single"], comparisons["label_selected"]
        flags = {"any_teacher_safe": bool(tsafe),
                 "teacher_safe_local_not_safe": bool(tsafe) and not local["safe_repair"],
                 "teacher_safe_local_determinate_not_safe": bool(tsafe) and not local["safe_repair"] and not local["has_unknown"],
                 "local_safe_full_not_safe": local["safe_repair"] and not full["safe_repair"],
                 "local_safe_full_harm": local["safe_repair"] and full["known_harm"],
                 "local_complete_full_incomplete": visual(observations["student_single"])["complete_scene"] and not visual(observations["label_selected"])["complete_scene"],
                 "full_complete_local_incomplete": visual(observations["label_selected"])["complete_scene"] and not visual(observations["student_single"])["complete_scene"]}
        cases.append({"case": item["case"], "query": q, "observations": observations, "comparisons": comparisons,
                      "safe_teachers": tsafe, "diagnostic_flags": flags})
    scores = {}
    for (cohort, mode), metrics in buckets.items():
        scores.setdefault(cohort, {})[mode] = {k: {**prompt_summary(v), "sum": sum(float(x) for _, x in v)} for k, v in metrics.items()}
    signals, contrasts = {}, {}
    for cohort in (OLD, FRESH):
        rows = [r for r in cases if r["query"]["cohort"] == cohort]
        signals[cohort] = {k: {"count": sum(r["diagnostic_flags"][k] for r in rows), "states": len(rows),
                              "cases": [r["case"] for r in rows if r["diagnostic_flags"][k]]} for k in rows[0]["diagnostic_flags"]}
        contrasts[cohort] = prompt_summary([(r["query"]["id"],
                            int(visual(r["observations"]["student_single"])["complete_scene"]) -
                            int(visual(r["observations"]["label_selected"])["complete_scene"])) for r in rows])
    return {"scores": scores, "cases": cases, "joined_review": joined, "diagnostic_signals": signals,
            "local_minus_full_complete": contrasts, "gpu_result": read_json(run / "result.json"),
            "review_frozen": frozen, "review_inputs": inputs, "script_sha256": sha256(__file__),
            "scope": plan["scope"], "limitations": "Six selected prompts, exposed development seeds, provisional inherited/new assistant labels; oracle teacher availability, no deployable selector or causal prefix/suffix separation."}


def summarize(run, review):
    campaign = ROOT / "experiments/local_expert_v16"
    report = build_report(run, review)
    write_json(campaign / "results.json", report)
    for n in ("observations.json", "observations_frozen.json"):
        shutil.copyfile(review / n, campaign / n)
    print({c: {m: {k: v[k]["sum"] for k in ("complete_scene", "safe_repair", "known_harm", "has_unknown")}
               for m, v in rows.items()} for c, rows in report["scores"].items()})
    print(report["diagnostic_signals"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", default="runs/le16_transfer_001")
    parser.add_argument("--review", default="runs/le16_review_001")
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve())
