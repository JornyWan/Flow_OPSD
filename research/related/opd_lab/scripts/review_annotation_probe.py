"""Review every phase15 model output before joining arm identities or scores."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from opdlab.annotation_probe import ARMS
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
    expected = {(m, p["pair_id"]) for m in ("joint128", *ARMS) for p in plan["evaluation"]}
    assert len(rows) == len(expected) == 152
    assert {(r["model"], r["pair_id"]) for r in rows} == expected
    for r in rows:
        assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
    return rows


def prepare(run, out):
    rows = validate(run)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    old = read_json(ROOT / "experiments/local_expert_v12/results.json")
    known = {e["candidate"]["image_sha256"]: e["observation"] for e in old["joined_review"]}
    out.mkdir(parents=True, exist_ok=False)
    key, items, inherited = [], [], {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pid in enumerate(sorted(grouped, key=lambda p: digest(["phase15_review_case", p])), 1):
        group = sorted(grouped[pid], key=lambda r: digest(["phase15_review_slot", pid, r["model"]]))
        record = records[group[0]["id"]]
        sheet = Image.new("RGB", (1024, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
        mapping, images, previous = {}, {}, {}
        for i, (letter, row) in enumerate(zip("ABCD", group)):
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], out / name)
            x, y = (i % 2) * 512, 128 + (i // 2) * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(out / name), (x, y + 24))
            mapping[letter], images[letter] = row, name
            if row["image_sha256"] in known:
                previous[letter] = known[row["image_sha256"]]
        sheet.save(out / f"case-{case:02d}.png")
        key.append({"case": case, "record": record, "mapping": mapping, "pair_id": pid})
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
        if previous:
            inherited[str(case)] = previous
    write_json(out / "key.json", key)
    write_json(out / "items.json", items)
    write_json(out / "inherited.json", inherited)
    write_json(out / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "key_sha256": sha256(out / "key.json"), "items_sha256": sha256(out / "items.json"),
               "inherited_sha256": sha256(out / "inherited.json"), "cases": len(items), "images": len(rows),
               "inherited_views": sum(len(v) for v in inherited.values()),
               "blinding": "Fresh outputs have randomized arm slots; exact archived pixels inherit old labels and are not newly blinded observations."})
    print({"cases": len(items), "images": len(rows), "inherited": sum(len(v) for v in inherited.values())})


def freeze(run, review):
    items, inherited = read_json(review / "items.json"), read_json(review / "inherited.json")
    draft = lines(review / "observations_draft.jsonl")
    labels = {r["case"]: r for r in draft}
    assert len(labels) == len(draft) == len(items) == 38
    assert set(labels) == {i["case"] for i in items}
    seen = {}
    for item in items:
        views = labels[item["case"]]["views"]
        assert set(views) == set("ABCD")
        for letter, obs in views.items():
            assert set(obs) == {"clauses", "full_visibility", "visible_text", "notes"}
            assert set(obs["clauses"]) == {c["id"] for c in item["constraints"]}
            assert all(type(v) is bool or v is None for v in [*obs["clauses"].values(), obs["full_visibility"]])
            assert isinstance(obs["visible_text"], str) and isinstance(obs["notes"], str)
            semantic = (obs["clauses"], obs["full_visibility"])
            old = inherited.get(str(item["case"]), {}).get(letter)
            if old:
                assert semantic == (old["clauses"], old["full_visibility"]), (item["case"], letter, "old pixels require old labels")
            h = sha256(review / item["images"][letter])
            if h in seen:
                assert seen[h] == semantic, (item["case"], letter, "identical pixels")
            seen[h] = semantic
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    write_json(review / "observations.json", sorted(draft, key=lambda r: r["case"]))
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant, not independent human annotation",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "key", "items", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"), "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md")})
    print("All152 views frozen before model-key join.")


def build_report(run, review):
    validate(run)
    frozen = read_json(review / "observations_frozen.json")
    assert frozen["frozen_before_key_join"]
    for n in ("observations", "key", "items", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    observations = {o["case"]: o for o in read_json(review / "observations.json")}
    scores, paired, joined = defaultdict(list), [], []
    for item in read_json(review / "key.json"):
        bymodel = {r["model"]: (r, observations[item["case"]]["views"][letter]) for letter, r in item["mapping"].items()}
        base = bymodel["joint128"][1]
        for model, (r, obs) in bymodel.items():
            cohorts = [r["cohort"]] + (["all_fresh"] if r["cohort"] != "fitting_seed_diagnostic" else [])
            comparison = compare(base, obs)
            vm, bm = visual(obs), visual(base)
            joined.append({"case": item["case"], "row": r, "observation": obs})
            for cohort in cohorts:
                values = {**vm, **{f"clause_{k}": v is True for k, v in obs["clauses"].items()},
                          "complete_scene_delta": int(vm["complete_scene"]) - int(bm["complete_scene"]),
                          "safe_repair": comparison["safe_repair"], "known_harm": comparison["known_harm"],
                          "comparison_unknown": comparison["has_unknown"], "proxy_utility": r["utility"]}
                for k, value in values.items():
                    scores[cohort, model, k].append((r["id"], value))
                for k, before in {**base["clauses"], "visibility": base["full_visibility"]}.items():
                    if before is True:
                        after = obs["full_visibility"] if k == "visibility" else obs["clauses"][k]
                        scores[cohort, model, k + "_true_to_false"].append((r["id"], after is False))
                        scores[cohort, model, k + "_true_to_unknown"].append((r["id"], after is None))
            paired.append({"case": item["case"], "id": r["id"], "seed": r["seed"], "cohort": r["cohort"],
                           "model": model, "comparison": comparison, "complete_scene_delta": int(vm["complete_scene"]) - int(bm["complete_scene"])})
    packed = {}
    for (cohort, model, metric), values in sorted(scores.items()):
        packed.setdefault(cohort, {}).setdefault(model, {})[metric] = {**prompt_summary(values), "sum": sum(float(v) for _, v in values)}
    # Direct paired model comparison, keeping replicate seeds inside each prompt.
    contrasts = {}
    for cohort in sorted(packed):
        rows = [e for e in joined if e["row"]["cohort"] == cohort or cohort == "all_fresh" and e["row"]["cohort"] != "fitting_seed_diagnostic"]
        indexed = {(e["row"]["pair_id"], e["row"]["model"]): e for e in rows}
        for other in ("joint128", "random_matched", "all_teacher"):
            values = []
            for e in rows:
                if e["row"]["model"] == "label_selected":
                    b = indexed[e["row"]["pair_id"], other]
                    values.append((e["row"]["id"], int(visual(e["observation"])["complete_scene"]) - int(visual(b["observation"])["complete_scene"])))
            contrasts.setdefault(cohort, {})[other] = {**prompt_summary(values), "sum": sum(v for _, v in values)}
    numeric = lines(run / "metrics.jsonl")
    alignment = [r["label_target_alignment"]["relative_target_mse"] for r in numeric if r["arm"] == "label_selected" and r["step"] == 128 and "label_target_alignment" in r]
    assert len(alignment) == 6
    primary = contrasts["positive_fresh_seed"]
    gate = {"median_relative_target_mse_at_most_half": float(np.median(alignment)) <= .5,
            "primary_at_least2_more_complete_than_initial": primary["joint128"]["sum"] >= 2,
            "primary_at_least2_more_complete_than_random": primary["random_matched"]["sum"] >= 2,
            "zero_known_harm_all_fresh": packed["all_fresh"]["label_selected"]["known_harm"]["sum"] == 0,
            "preservation_complete_not_worse": contrasts["preservation_fresh_seed"]["joint128"]["sum"] >= 0,
            "unfitted_complete_not_worse": contrasts["unfitted_prompt_fresh_seed"]["joint128"]["sum"] >= 0}
    return {"scope": "small selected-cohort label-assisted learning diagnostic; no automatic or native OPD method claim",
               "scores": packed, "label_vs_other_complete": contrasts, "paired_rows": paired, "joined_review": joined,
               "median_label_target_relative_mse": float(np.median(alignment)), "promising_pilot": {"checks": gate, "passed": all(gate.values())},
               "gpu_result": read_json(run / "result.json"), "review_inputs": read_json(review / "inputs.json"),
               "review_frozen": frozen, "script_sha256": sha256(__file__)}


def summarize(run, review):
    report = build_report(run, review)
    campaign = ROOT / "experiments/local_expert_v15"
    write_json(campaign / "results.json", report)
    for n in ("observations", "observations_frozen"):
        shutil.copyfile(review / f"{n}.json", campaign / f"{n}.json")
    print({c: {m: {k: v["sum"] for k, v in metrics.items() if k in ("complete_scene", "known_harm", "has_unknown")}
               for m, metrics in models.items()} for c, models in report["scores"].items()})
    print({"promising_pilot": report["promising_pilot"], "median_relative_mse": report["median_label_target_relative_mse"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve())
