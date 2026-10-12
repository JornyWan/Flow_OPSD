"""Blind paired collection review, then apply the predeclared training-coverage gate."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.acceptance_collection import GATE, coverage_gate, rows
from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from review_action_transfer import check_labels
from review_prospective import compare, visual


def validate(run):
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    assert result["status"] == "complete" and result["optimizer_updates"] == result["evaluation_images_generated"] == 0
    assert plan["coverage_gate"] == GATE and not any((run / n).exists() for n in ("RUNNING", "failure.json"))
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    for p, h in result["output_hashes"].items():
        assert sha256(run / p) == h, p
    for k, v in plan["expected_counts"].items():
        assert result["counts"].get(k, 0) == v
    entries = rows(run / "endpoints.jsonl")
    assert len(entries) == 96
    assert {(r["pair_id"], r["mode"]) for r in entries} == {(p["pair_id"], m) for p in plan["collection"] for m in ("noop", "geneval@1")}
    assert not {r["seed"] for r in entries} & {p["seed"] for p in plan["evaluation"]}
    for r in entries + rows(run / "references.jsonl"):
        assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
    return entries


def prepare(run, review, campaign):
    entries = validate(run)
    records = {r["id"]: r for r in rows(run / "data.jsonl")}
    known = {}
    for phase in (18, 19):
        for e in read_json(ROOT / f"experiments/local_expert_v{phase}/results.json")["joined_review"]:
            known[e["row"]["image_sha256"]] = e["observation"]
    grouped = defaultdict(list)
    for r in entries:
        grouped[r["pair_id"]].append(r)
    review.mkdir(parents=True, exist_ok=False)
    items, key, inherited = [], [], {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pid in enumerate(sorted(grouped, key=lambda p: digest(["acceptance_case", p])), 1):
        group = sorted(grouped[pid], key=lambda r: digest(["acceptance_slot", pid, r["mode"]]))
        record = records[group[0]["id"]]
        if case % 2:
            sheet = Image.new("RGB", (1024, 1280), "white")
            draw = ImageDraw.Draw(sheet)
        y = ((case - 1) % 2) * 640
        draw.text((6, y + 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
        mapping, images = {}, {}
        for i, (letter, row) in enumerate(zip("AB", group)):
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], review / name)
            draw.text((i * 512 + 8, y + 106), letter, font=font, fill="black")
            sheet.paste(Image.open(review / name), (i * 512, y + 128))
            mapping[letter], images[letter] = row, name
            if row["image_sha256"] in known:
                inherited.setdefault(str(case), {})[letter] = known[row["image_sha256"]]
        if case % 2 == 0:
            sheet.save(review / f"page-{case // 2:02d}.png")
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
        key.append({"case": case, "pair_id": pid, "mapping": mapping})
    for n, obj in (("items", items), ("key", key), ("inherited", inherited)):
        write_json(review / f"{n}.json", obj)
    write_json(review / "inputs.json", {"run": str(run), "cases": 48, "views": 96, "pages": 24,
               "inherited_views": sum(map(len, inherited.values())), "result_sha256": sha256(run / "result.json"),
               "review_script_sha256": sha256(__file__),
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("items", "key", "inherited")},
               "scope": "New collection images with baseline/teacher/seed/score identity hidden until freeze. Provisional implementing-assistant labels, familiar prompts."})
    print({"cases": 48, "views": 96, "pages": 24, "inherited": sum(map(len, inherited.values()))})


def freeze(run, review, campaign):
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    obs = check_labels(review)
    assert len(obs) == 48 and sum(len(o["views"]) for o in obs) == 96
    write_json(review / "observations.json", obs)
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant; provisional development labels",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "items", "key", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"), "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md"),
               "review_script_sha256": sha256(__file__)})
    print("Frozen96 observations before identity join or target selection.")


def build_report(run, review):
    validate(run)
    inputs, frozen = read_json(review / "inputs.json"), read_json(review / "observations_frozen.json")
    assert inputs["result_sha256"] == sha256(run / "result.json") and frozen["frozen_before_key_join"]
    assert inputs["review_script_sha256"] == frozen["review_script_sha256"] == sha256(__file__)
    assert frozen["rubric_sha256"] == sha256(run / "REVIEW_RUBRIC.md")
    for n in ("observations", "items", "key", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
        if n != "observations":
            assert inputs[f"{n}_sha256"] == frozen[f"{n}_sha256"]
    assert check_labels(review) == read_json(review / "observations.json")
    observations = {o["case"]: o["views"] for o in read_json(review / "observations.json")}
    queries = {q["pair_id"]: q for q in rows(run / "queries.jsonl")}
    buckets, cases, joined, targets = defaultdict(list), [], [], []
    for item in read_json(review / "key.json"):
        obs = {r["mode"]: observations[item["case"]][letter] for letter, r in item["mapping"].items()}
        q = queries[item["pair_id"]]
        comparison = compare(obs["noop"], obs["geneval@1"])
        case = {"case": item["case"], "id": q["id"], "seed": q["seed"], "pair_id": q["pair_id"],
                "observations": obs, "comparison": comparison, "safe_repair": comparison["safe_repair"],
                "complete_scene": visual(obs["geneval@1"])["complete_scene"]}
        cases.append(case)
        for letter, r in item["mapping"].items():
            o, c = obs[r["mode"]], compare(obs["noop"], obs[r["mode"]])
            joined.append({"case": item["case"], "view": letter, "row": r, "observation": o, "comparison": c})
            for k, v in {**visual(o), "safe_repair": c["safe_repair"], "known_harm": c["known_harm"], "comparison_unknown": c["has_unknown"]}.items():
                buckets[r["mode"], k].append((r["id"], v))
        target = "geneval@1" if comparison["safe_repair"] else "noop"
        targets.append({**{k: q[k] for k in ("id", "seed", "pair_id", "replicate", "index", "dt", "action_hashes", "saved_query_sha256")},
                        "saved_query": str(run / q["saved_query"]), "target": target, "accepted": target != "noop",
                        "reason": comparison["status"], "case": item["case"], "scope": "Provisional assistant label, not automatic verifier acceptance"})
    scores = {}
    for (mode, metric), values in buckets.items():
        scores.setdefault(mode, {})[metric] = {**prompt_summary(values), "sum": sum(v for _, v in values)}
    return {"scores": scores, "coverage_gate": coverage_gate(cases), "cases": cases, "joined_review": joined,
            "targets": targets, "gpu_result": read_json(run / "result.json"), "review_frozen": frozen,
            "limitations": "Development collection on six familiar prompts; provisional assistant labels. No student training and no independent evaluation images generated. Gate is a preregistered engineering decision, not a statistical test."}


def summarize(run, review, campaign):
    report = build_report(run, review)
    write_json(campaign / "results.json", report)
    write_json(campaign / "target_manifest.json", {"training_allowed_by_coverage_gate": report["coverage_gate"]["passed"],
               "source_results_sha256": sha256(campaign / "results.json"), "targets": report["targets"]})
    for n in ("observations.json", "observations_frozen.json"):
        shutil.copyfile(review / n, campaign / n)
    print(report["coverage_gate"])
    print({m: {k: v["sum"] for k, v in s.items() if k in ("complete_scene", "safe_repair", "known_harm", "has_unknown")} for m, s in report["scores"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", default="runs/le20_collection_001")
    parser.add_argument("--review", default="runs/le20_review_001")
    parser.add_argument("--campaign", default="experiments/local_expert_v20")
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve(), Path(args.campaign).resolve())
