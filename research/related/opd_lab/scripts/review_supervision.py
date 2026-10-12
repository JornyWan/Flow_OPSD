"""Review full and single-action paths through the same frozen, model-hidden protocol."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
from statistics import median
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from review_action_transfer import check_labels
from review_prospective import compare, lines, visual


def validate(run):
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert result["status"] == "complete"
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    for p, h in result["output_hashes"].items():
        assert sha256(run / p) == h, p
    rows = lines(run / "endpoints.jsonl")
    expected = {(p["pair_id"], m, path) for p in plan["evaluation"] for m in ["joint128", *plan["arms"]]
                for path in (["full", "single"] if p["single_action"] and m != "joint128" else ["full"])}
    assert len(rows) == len(expected) == plan["evaluation_images"]
    assert {(r["pair_id"], r["model"], r["path"]) for r in rows} == expected
    for r in rows:
        assert r["valid"] and sha256(run / r["image"]) == r["image_sha256"]
    return rows


def prepare(run, review, campaign):
    rows = validate(run)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"], r["path"]].append(r)
    for (pid, path), group in list(grouped.items()):
        if path == "single":
            group.append(next(r for r in grouped[pid, "full"] if r["model"] == "joint128"))
    relevant = {r["image_sha256"] for r in rows}
    known = {}
    for phase in (12, 15, 16, 17):
        for e in read_json(campaign.parent / f"local_expert_v{phase}/results.json")["joined_review"]:
            row, obs = e.get("row", e.get("candidate")), e["observation"]
            h = row["image_sha256"]
            if h not in relevant:
                continue
            if h in known:
                assert (known[h]["clauses"], known[h]["full_visibility"]) == (obs["clauses"], obs["full_visibility"])
            known[h] = obs
    review.mkdir(parents=True, exist_ok=False)
    items, keys, inherited = [], [], {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, (pid, path) in enumerate(sorted(grouped, key=lambda k: digest(["supervision_case", k])), 1):
        group = sorted(grouped[pid, path], key=lambda r: digest(["supervision_slot", pid, path, r["model"]]))
        assert len(group) == 4
        record = records[group[0]["id"]]
        sheet = Image.new("RGB", (1024, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)), font=font, fill="black", spacing=3)
        images, mapping = {}, {}
        for i, (letter, row) in enumerate(zip("ABCD", group)):
            name = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], review / name)
            x, y = i % 2 * 512, 128 + i // 2 * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(review / name), (x, y + 24))
            images[letter], mapping[letter] = name, row
            if row["image_sha256"] in known:
                inherited.setdefault(str(case), {})[letter] = known[row["image_sha256"]]
        sheet.save(review / f"case-{case:02d}.png")
        items.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
        keys.append({"case": case, "pair_id": pid, "path": path, "mapping": mapping})
    for name, value in (("items", items), ("key", keys), ("inherited", inherited)):
        write_json(review / f"{name}.json", value)
    write_json(review / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "review_script_sha256": sha256(__file__), "cases": len(items), "views": 4 * len(items),
               "generated_endpoints": len(rows), "inherited_views": sum(map(len, inherited.values())),
               "scope": "Fresh model/path/seed/score identities hidden; fitting diagnostic pixels may inherit prior labels. Assistant development review, not independent annotation."})
    print({"cases": len(items), "views": 4 * len(items), "inherited": sum(map(len, inherited.values()))})


def freeze(run, review, campaign):
    if (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    observations = check_labels(review)
    inputs = read_json(review / "inputs.json")
    assert len(observations) == inputs["cases"] and sum(len(o["views"]) for o in observations) == inputs["views"]
    write_json(review / "observations.json", observations)
    write_json(review / "observations_frozen.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join": True, "reviewer": "implementing assistant; provisional labels",
               **{f"{n}_sha256": sha256(review / f"{n}.json") for n in ("observations", "items", "key", "inherited")},
               "draft_sha256": sha256(review / "observations_draft.jsonl"), "rubric_sha256": sha256(run / "REVIEW_RUBRIC.md"),
               "review_script_sha256": sha256(__file__)})
    print(f"All{inputs['views']} view observations frozen before identity join.")


def build_report(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] and inputs["result_sha256"] == sha256(run / "result.json")
    assert sha256(__file__) == inputs["review_script_sha256"] == frozen["review_script_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    for n in ("observations", "items", "key", "inherited"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    obs = {o["case"]: o["views"] for o in read_json(review / "observations.json")}
    values, direct, joined = defaultdict(list), defaultdict(list), []
    for item in read_json(review / "key.json"):
        models = {r["model"]: (r, obs[item["case"]][letter]) for letter, r in item["mapping"].items()}
        base = models["joint128"][1]
        path = item["path"]
        for model, (row, o) in models.items():
            comparison = compare(base, o)
            metrics = {**visual(o), "safe_repair": comparison["safe_repair"], "known_harm": comparison["known_harm"],
                       "comparison_unknown": comparison["has_unknown"]}
            joined.append({"case": item["case"], "evaluated_path": path, "row": row, "observation": o, "comparison": comparison})
            for cohort in row["cohorts"]:
                for metric, value in metrics.items():
                    values[cohort, path, model, metric].append((row["id"], value))
                for clause, before in {**base["clauses"], "visibility": base["full_visibility"]}.items():
                    if before is True:
                        after = o["full_visibility"] if clause == "visibility" else o["clauses"][clause]
                        values[cohort, path, model, clause + "_true_to_false"].append((row["id"], after is False))
                        values[cohort, path, model, clause + "_true_to_unknown"].append((row["id"], after is None))
        for other in ("joint128", "single", "random_matched"):
            row, multi = models["multi"]
            for cohort in row["cohorts"]:
                direct[cohort, path, other].append((row["id"], int(visual(multi)["complete_scene"]) - int(visual(models[other][1])["complete_scene"])))
    def packed(data):
        result = {}
        for keys, rows in sorted(data.items()):
            target = result
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = {**prompt_summary(rows), "sum": sum(float(v) for _, v in rows)}
        return result
    scores, contrasts = packed(values), packed(direct)
    plan = read_json(run / "probe_plan.json")
    points = {p["key"]: p for p in plan["points"]}
    alignment = defaultdict(list)
    for r in lines(run / "metrics.jsonl"):
        if r["step"] == plan["updates"] and "alignment" in r:
            subset = "original" if points[r["key"]]["original"] else "added"
            alignment[r["arm"], subset].append(r["alignment"]["relative_target_mse"])
    medians = {}
    for (arm, subset), rows in alignment.items():
        medians.setdefault(arm, {})[subset] = {"states": len(rows), "median": median(rows)}
    primary = contrasts["positive_fresh_seed"]["full"]
    checks = {"added_target_median_relative_mse_at_most_half": medians["multi"]["added"]["median"] <= .5,
              "at_least2_more_complete_than_initial": primary["joint128"]["sum"] >= 2,
              "at_least1_more_complete_than_single": primary["single"]["sum"] >= 1,
              "at_least2_more_complete_than_random": primary["random_matched"]["sum"] >= 2,
              "zero_known_harm_all_fresh": scores["all_fresh"]["full"]["multi"]["known_harm"]["sum"] == 0,
              "preservation_not_worse": contrasts["preservation_fresh_seed"]["full"]["joint128"]["sum"] >= 0,
              "unfitted_not_worse": contrasts["unfitted_prompt_fresh_seed"]["full"]["joint128"]["sum"] >= 0}
    return {"scores": scores, "multi_vs_other_complete": contrasts, "alignment": medians,
            "promising_pilot": {"checks": checks, "passed": all(checks.values())}, "joined_review": joined,
            "gpu_result": read_json(run / "result.json"), "review_frozen": frozen, "review_inputs": inputs,
            "limitations": "Only two prompts gain training states; selected development cohort, provisional assistant labels, fixed-query learning not native online OPD, no automatic-verifier or sampler-robustness claim. Random arm permutes expert/noop roles within tasks, including protection slots."}


def summarize(run, review, campaign):
    report = build_report(run, review)
    write_json(campaign / "results.json", report)
    for n in ("observations", "observations_frozen"):
        shutil.copyfile(review / f"{n}.json", campaign / f"{n}.json")
    print({c: {m: {k: v["sum"] for k, v in ms.items() if k in ("complete_scene", "known_harm", "has_unknown")} for m, ms in paths["full"].items()} for c, paths in report["scores"].items()})
    print(report["alignment"])
    print(report["promising_pilot"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "summarize"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    globals()[args.stage](Path(args.run).resolve(), Path(args.review).resolve(), Path(args.campaign).resolve())
