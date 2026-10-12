"""Complete paired review and prompt-clustered transfer statistics for phase11."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.report import prompt_summary
from review_state_requery import compare_observations


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def validate(run):
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    if result["status"] != "complete":
        raise ValueError("Incomplete evaluation")
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError("Changed frozen input")
    for name, expected in result["output_hashes"].items():
        if sha256(run / name) != expected:
            raise ValueError("Changed output rows")
    rows = lines(run / "endpoints.jsonl")
    expected = {(name, p["pair_id"]) for name in plan["models"] for p in plan["pairs"]}
    if len(rows) != len(expected) or {(r["model"], r["pair_id"]) for r in rows} != expected:
        raise ValueError("Incomplete/duplicate model-pair matrix")
    pairs = {p["pair_id"]: p for p in plan["pairs"]}
    for row in rows:
        if any(row[k] != v for k, v in pairs[row["pair_id"]].items()):
            raise ValueError("Output pair identity/seed differs from plan")
        if sha256(run / row["image"]) != row["image_sha256"] or not row["valid"]:
            raise ValueError("Changed image or invalid verifier")
    return rows


def prepare(run, out):
    rows = validate(run)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    out.mkdir(parents=True, exist_ok=False)
    key, public = [], []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, pair in enumerate(sorted(grouped, key=lambda p: digest(["phase11_blind_pair", p])), 1):
        group = sorted(grouped[pair], key=lambda r: digest(["phase11_blind_model", pair, r["model"]]))
        record = records[group[0]["id"]]
        sheet = Image.new("RGB", (1024, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"], 125)),
                  font=font, fill="black", spacing=3)
        mapping, images = {}, {}
        for i, (letter, row) in enumerate(zip("ABCD", group)):
            filename = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / row["image"], out / filename)
            x, y = (i % 2) * 512, 128 + (i // 2) * 536
            draw.text((x + 8, y), letter, font=font, fill="black")
            sheet.paste(Image.open(out / filename), (x, y + 24))
            mapping[letter], images[letter] = row, filename
        sheet.save(out / f"case-{case:02d}.png")
        key.append({"case": case, "pair_id": pair, "record": record, "mapping": mapping})
        public.append({"case": case, "prompt": record["prompt"], "constraints": record["constraints"], "images": images})
    write_json(out / "key.json", key)
    write_json(out / "items.json", public)
    write_json(out / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "key_sha256": sha256(out / "key.json"), "items_sha256": sha256(out / "items.json"),
               "pairs": len(key), "images": len(rows), "layout": "independently permuted2x2 model panels per pair"})
    print(f"Prepared all{len(rows)} images in{len(key)} blinded paired sheets.")


def visual_metrics(observation):
    values = list(observation["clauses"].values())
    return {"joint_clauses": all(v is True for v in values),
            "complete_scene": all(v is True for v in values) and observation["full_visibility"] is True,
            "full_visibility": observation["full_visibility"] is True,
            "has_unknown": any(v is None for v in [*values, observation["full_visibility"]]),
            **{f"clause_{k}": v is True for k, v in observation["clauses"].items()}}


def summarize(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    for name in ("observations", "key", "items"):
        if sha256(review / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Frozen review changed")
    if sha256(run / "result.json") != inputs["result_sha256"] or sha256(run / "REVIEW_RUBRIC.md") != frozen["rubric_sha256"]:
        raise ValueError("Result or rubric changed")
    observations = read_json(review / "observations.json")
    observed = {r["case"]: r for r in observations}
    if len(observed) != len(observations) or set(observed) != set(range(1, 43)):
        raise ValueError("Missing or duplicate reviewed case")
    joined, paired, scores, comparisons = [], [], defaultdict(list), defaultdict(list)
    for entry in read_json(review / "key.json"):
        views = observed[entry["case"]]["views"]
        if set(views) != set("ABCD"):
            raise ValueError("Incomplete four-view review")
        by_model = {}
        for letter, row in entry["mapping"].items():
            obs = views[letter]
            if set(obs["clauses"]) != set(row["constraint_ids"]) or any(type(v) is not bool and v is not None for v in [*obs["clauses"].values(), obs["full_visibility"]]):
                raise ValueError("Invalid clause/visibility labels")
            by_model[row["model"]] = (row, obs)
            joined.append({"case": entry["case"], "view": letter, "row": row, "observation": obs})
            for metric, value in {**visual_metrics(obs), "proxy_utility": row["utility"], "proxy_joint": row["joint_success"]}.items():
                scores[(row["cohort"], row["model"], metric)].append((row["id"], value))
        base_row, base = by_model["joint128"]
        for model, (row, obs) in by_model.items():
            if model == "joint128":
                continue
            comparison = compare_observations(base, obs)
            transitions = {k: {"before": base["clauses"][k], "after": obs["clauses"][k]} for k in obs["clauses"]}
            paired.append({"case": entry["case"], "pair_id": row["pair_id"], "id": row["id"],
                           "cohort": row["cohort"], "model": model, "seed": row["seed"],
                           "comparison": comparison, "transitions": transitions,
                           "base_visibility": base["full_visibility"], "new_visibility": obs["full_visibility"]})
            a, b = visual_metrics(base), visual_metrics(obs)
            deltas = {f"{m}_delta": int(b[m]) - int(a[m]) for m in a if m != "has_unknown"}
            deltas.update(proxy_utility_delta=row["utility"] - base_row["utility"],
                          proxy_joint_delta=int(row["joint_success"]) - int(base_row["joint_success"]),
                          **{k: comparison[k] for k in ("accepted_encoded_repair", "known_harm", "has_unknown")})
            for k, value in deltas.items():
                comparisons[(row["cohort"], model, k)].append((row["id"], value))
            for clause, transition in transitions.items():
                if transition["before"] is True:
                    for label, value in (("false", transition["after"] is False), ("unknown", transition["after"] is None)):
                        comparisons[(row["cohort"], model, f"{clause}_true_to_{label}")].append((row["id"], value))
            if base["full_visibility"] is True:
                for label, value in (("false", obs["full_visibility"] is False), ("unknown", obs["full_visibility"] is None)):
                    comparisons[(row["cohort"], model, f"visibility_true_to_{label}")].append((row["id"], value))
    def pack(values):
        result = {}
        for (cohort, model, metric), entries in sorted(values.items()):
            result.setdefault(cohort, {}).setdefault(model, {})[metric] = {**prompt_summary(entries), "sum": sum(float(v) for _, v in entries)}
        return result
    campaign = ROOT / "experiments/local_expert_v11"
    result = {"scope": "fresh-seed and existing development-set transfer; no checkpoint selection or sampler claim",
              "statistics": "equal prompt weighting;2000 paired prompt bootstrap resamples,seed0; unknown reported separately and not strict success",
              "scores": pack(scores), "paired_summary": pack(comparisons), "paired_rows": paired, "joined_review": joined,
              "gpu_result": read_json(run / "result.json"), "review_inputs": inputs}
    write_json(campaign / "results.json", result)
    for name in ("observations", "observations_frozen"):
        shutil.copyfile(review / f"{name}.json", campaign / f"{name}.json")
    print(json.dumps({c: {m: {k: v for k, v in metrics.items() if k in ("complete_scene_delta", "known_harm", "accepted_encoded_repair")}
                         for m, metrics in models.items()} for c, models in result["paired_summary"].items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=["prepare", "summarize"])
    p.add_argument("--run", required=True)
    p.add_argument("--review", required=True)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else summarize)(Path(args.run).resolve(), Path(args.review).resolve())
