"""Prepare phase8 unique-image review, then join only frozen observations."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.trajectory_probe import MODES


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def prepare(run, out):
    result = read_json(run / "result.json")
    if result["status"] != "complete" or sha256(run / "endpoints.jsonl") != result["endpoints_sha256"]:
        raise ValueError("Incomplete or changed GPU results")
    rows = lines(run / "endpoints.jsonl")
    if len(rows) != 35 or len({(r["id"], r["mode"]) for r in rows}) != 35:
        raise ValueError("Incomplete matrix")
    old = ROOT / "runs/le7_review_001"
    frozen = read_json(old / "observations_frozen.json")
    for name in ("key", "observations"):
        if sha256(old / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Old frozen review changed")
    old_labels = {r["case"]: r for r in read_json(old / "observations.json")}
    known = {}
    for entry in read_json(old / "key.json"):
        h, obs = entry["row"]["image_sha256"], old_labels[entry["case"]]
        if h in known and any(known[h][k] != obs[k] for k in ("clauses", "full_visibility")):
            raise ValueError("Conflicting old labels on identical pixels")
        known[h] = obs
    unseen, inherited = {}, []
    for row in rows:
        if sha256(run / row["image"]) != row["image_sha256"]:
            raise ValueError("Image changed")
        if row["mode"] == "base":
            continue
        h = row["image_sha256"]
        if h in known:
            inherited.append({"row": row, "observation": known[h], "source": "frozen_phase7_exact_pixels"})
        else:
            unseen.setdefault(h, []).append(row)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    out.mkdir(parents=True, exist_ok=False)
    key, public = [], []
    for case, (h, group) in enumerate(sorted(unseen.items(), key=lambda item: digest(["phase8_blind", item[0]])), 1):
        if len({r["id"] for r in group}) != 1:
            raise ValueError("Same pixels have different prompts")
        name = f"image-{case:02d}.png"
        shutil.copyfile(run / group[0]["image"], out / name)
        key.append({"case": case, "image_sha256": h, "rows": group})
        public.append({"case": case, "image": name, "prompt": records[group[0]["id"]]["prompt"]})
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
    write_json(out / "key.json", key)
    write_json(out / "items.json", public)
    write_json(out / "inherited.json", inherited)
    write_json(out / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "key_sha256": sha256(out / "key.json"), "inherited_sha256": sha256(out / "inherited.json"),
               "old_observations_sha256": frozen["observations_sha256"], "new_unique_images": len(public),
               "mode_rows_inherited_from_phase7": len(inherited), "total_mode_rows": 30})
    print(f"Prepared {len(public)} unseen images; {len(inherited)} rows inherit exact frozen phase7 labels.")


def summarize(run, review):
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    for name in ("observations", "key", "inherited"):
        if sha256(review / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Frozen visual evidence changed")
    if sha256(run / "result.json") != inputs["result_sha256"]:
        raise ValueError("GPU result changed")
    observed = read_json(review / "observations.json")
    labels = {r["case"]: r for r in observed}
    if len(labels) != len(observed) or set(labels) != set(range(1, inputs["new_unique_images"] + 1)):
        raise ValueError("Missing or duplicate image observations")
    joined = read_json(review / "inherited.json")
    for entry in read_json(review / "key.json"):
        for row in entry["rows"]:
            joined.append({"row": row, "observation": labels[entry["case"]], "source": "phase8_blinded_unique_image"})
    specs = {s["id"]: s for s in read_json(ROOT / "experiments/local_expert_v7/selection.json")}
    for entry in joined:
        spec, obs = specs[entry["row"]["id"]], entry["observation"]
        if (set(obs["clauses"]) != set(spec["old_observations"]["target"]["clauses"]) or
                any(type(v) is not bool and v is not None for v in [*obs["clauses"].values(), obs["full_visibility"]])):
            raise ValueError("Inconsistent clause/visibility labels")
        retained = all(obs["clauses"][k] is True for k in set(spec["repaired"] + spec["preserved"]))
        if spec["old_observations"]["target"]["full_visibility"]:
            retained = retained and obs["full_visibility"] is True
        entry.update(repair_retained=retained, complete_visibility_subset=spec["complete_visibility_subset"])
    counts = {}
    for mode in MODES:
        group = [r for r in joined if r["row"]["mode"] == mode]
        if len(group) != 5 or len({r["row"]["id"] for r in group}) != 5:
            raise ValueError("Incomplete joined mode")
        clean = [r for r in group if r["complete_visibility_subset"]]
        counts[mode] = {"retained": sum(r["repair_retained"] for r in group), "n": 5,
                        "complete_visibility_subset_retained": sum(r["repair_retained"] for r in clean),
                        "complete_visibility_subset_n": len(clean),
                        "mean_proxy_utility": sum(r["row"]["utility"] for r in group) / 5}
    matrix = {ident: {r["row"]["mode"]: {"retained": r["repair_retained"],
              "query_state_mse_vs_frozen": r["row"]["query_state_mse_vs_frozen"]}
              for r in joined if r["row"]["id"] == ident} for ident in specs}
    campaign = ROOT / "experiments/local_expert_v8"
    write_json(campaign / "results.json", {"scope": "selected-training-query mechanism diagnostic; provisional review",
               "counts": counts, "matrix": matrix, "joined_review": joined, "review_inputs": inputs,
               "gpu_result": read_json(run / "result.json")})
    write_json(campaign / "observations.json", observed)
    write_json(campaign / "observations_frozen.json", frozen)
    print(json.dumps({"counts": counts, "matrix": matrix}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=["prepare", "summarize"])
    p.add_argument("--run", required=True)
    p.add_argument("--review", required=True)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else summarize)(Path(args.run).resolve(), Path(args.review).resolve())
