"""Review every phase10 endpoint without exposing arm, checkpoint or score."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import ROOT, digest, read_json, sha256, write_json
from review_state_requery import compare_observations


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def validate(run):
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    if result["status"] != "complete":
        raise ValueError("Fitting is incomplete")
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError("Changed frozen input")
    for name, expected in result["output_hashes"].items():
        if sha256(run / name) != expected:
            raise ValueError("Changed GPU output")
    rows = lines(run / "endpoints.jsonl")
    expected = {(arm, step, spec["id"], mode) for arm, a in plan["arms"].items()
                for step in a["audits"] if step for spec in plan["selected"] for mode in ("local", "full")}
    actual = {(r["arm"], r["optimizer_step"], r["id"], r["mode"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Incomplete endpoint matrix")
    for row in rows:
        if sha256(run / row["image"]) != row["image_sha256"]:
            raise ValueError("Changed endpoint pixels")
    return rows


def prepare(run, out):
    rows = validate(run)
    known = {}
    for entry in read_json(ROOT / "experiments/local_expert_v9/results.json")["joined_review"]:
        h, obs = entry["row"]["image_sha256"], entry["observation"]
        if h in known and any(known[h][k] != obs[k] for k in ("clauses", "full_visibility")):
            raise ValueError("Conflicting inherited labels")
        known[h] = obs
    inherited, unseen = [], {}
    for row in rows:
        h = row["image_sha256"]
        if h in known:
            inherited.append({"row": row, "observation": known[h], "source": "phase9_exact_pixels"})
        else:
            unseen.setdefault(h, []).append(row)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    out.mkdir(parents=True, exist_ok=False)
    public, key = [], []
    for case, (h, group) in enumerate(sorted(unseen.items(), key=lambda x: digest(["phase10_blind", x[0]])), 1):
        if len({r["id"] for r in group}) != 1:
            raise ValueError("Image shared across different prompts")
        image = f"image-{case:02d}.png"
        shutil.copyfile(run / group[0]["image"], out / image)
        public.append({"case": case, "image": image, "prompt": records[group[0]["id"]]["prompt"]})
        key.append({"case": case, "image_sha256": h, "rows": group})
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    for start in range(0, len(public), 4):
        sheet = Image.new("RGB", (1024, 1216), "white")
        draw = ImageDraw.Draw(sheet)
        for i, item in enumerate(public[start:start + 4]):
            x, y = (i % 2) * 512, (i // 2) * 608
            caption = f"Image {item['case']:02d}\n" + "\n".join(textwrap.wrap(item["prompt"], 68))
            draw.text((x + 6, y + 3), caption, font=font, fill="black", spacing=2)
            sheet.paste(Image.open(out / item["image"]), (x, y + 96))
        sheet.save(out / f"sheet-{start // 4 + 1:02d}.png")
    write_json(out / "items.json", public)
    write_json(out / "key.json", key)
    write_json(out / "inherited.json", inherited)
    write_json(out / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "new_unique_images": len(public), "inherited_rows": len(inherited), "endpoint_rows": len(rows),
               "key_sha256": sha256(out / "key.json"), "inherited_sha256": sha256(out / "inherited.json")})
    print(f"Prepared {len(public)} new unique images and {len(inherited)} inherited rows.")


def summarize(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    for name in ("observations", "key", "inherited"):
        if sha256(review / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Frozen review changed")
    if sha256(run / "result.json") != inputs["result_sha256"] or sha256(run / "REVIEW_RUBRIC.md") != frozen["rubric_sha256"]:
        raise ValueError("GPU result or rubric changed")
    observed = read_json(review / "observations.json")
    labels = {r["case"]: r for r in observed}
    if len(labels) != len(observed) or set(labels) != set(range(1, inputs["new_unique_images"] + 1)):
        raise ValueError("Review coverage incomplete")
    joined = read_json(review / "inherited.json")
    for entry in read_json(review / "key.json"):
        joined.extend({"row": row, "observation": labels[entry["case"]], "source": "phase10_blinded_unique_image"} for row in entry["rows"])
    if len(joined) != inputs["endpoint_rows"]:
        raise ValueError("Incomplete joined matrix")
    plan = read_json(run / "probe_plan.json")
    selected = {s["id"]: s for s in plan["selected"]}
    for entry in joined:
        entry["comparison"] = compare_observations(selected[entry["row"]["id"]]["base_observation"], entry["observation"])
    counts = {}
    for arm, a in plan["arms"].items():
        for step in a["audits"]:
            if not step:
                continue
            for mode in ("local", "full"):
                group = [e for e in joined if (e["row"]["arm"], e["row"]["optimizer_step"], e["row"]["mode"]) == (arm, step, mode)]
                apple = [e for e in group if selected[e["row"]["id"]]["candidate"] != "base"]
                controls = [e for e in group if selected[e["row"]["id"]]["candidate"] == "base"]
                assert len(apple) == 1 and len(controls) == 4
                counts[f"{arm}{step}_{mode}"] = {"apple_repaired": apple[0]["comparison"]["accepted_encoded_repair"],
                    "controls_preserved": sum(not e["comparison"]["known_harm"] and not e["comparison"]["has_unknown"] for e in controls),
                    "controls_known_harm": sum(e["comparison"]["known_harm"] for e in controls),
                    "controls_unknown": sum(e["comparison"]["has_unknown"] for e in controls),
                    "fully_correct": sum(all(v is True for v in e["observation"]["clauses"].values()) and e["observation"]["full_visibility"] is True for e in group)}
    metrics = lines(run / "metrics.jsonl")
    fitting = []
    for arm, a in plan["arms"].items():
        for step in a["audits"]:
            group = [r for r in metrics if r["arm"] == arm and r["optimizer_step"] == step]
            initial = [r for r in metrics if r["arm"] == arm and r["optimizer_step"] == 0]
            repair = [r for r in group if r["origin"] == "external_expert"]
            fitting.append({"arm": arm, "optimizer_step": step, "apple_query": repair[0],
                "all50_target_loss": sum(r["target_loss"] for r in group) / len(group),
                "all50_loss_over_initial": sum(r["target_loss"] for r in group) / sum(r["target_loss"] for r in initial),
                "preservation_mean_mse": sum(r["target_mse"] for r in group if r["origin"] != "external_expert") / (len(group) - 1)})
    campaign = ROOT / "experiments/local_expert_v10"
    write_json(campaign / "results.json", {"scope": "one selected repair learning diagnostic, not generalization", "counts": counts,
               "fitting": fitting, "joined_review": joined, "gpu_result": read_json(run / "result.json"), "review_inputs": inputs})
    for name in ("observations", "observations_frozen"):
        shutil.copyfile(review / f"{name}.json", campaign / f"{name}.json")
    print(json.dumps({"counts": counts, "fitting": fitting}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=["prepare", "summarize"])
    p.add_argument("--run", required=True)
    p.add_argument("--review", required=True)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else summarize)(Path(args.run).resolve(), Path(args.review).resolve())
