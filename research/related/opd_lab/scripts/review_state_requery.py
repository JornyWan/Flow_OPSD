"""Blind review and current-baseline headroom audit for the phase9 candidates."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import ROOT, digest, read_json, sha256, write_json


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def prepare(run, out):
    for path, h in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != h:
            raise ValueError("Frozen phase9 input changed")
    result = read_json(run / "result.json")
    if result["status"] != "complete" or sha256(run / "candidates.jsonl") != result["candidates_sha256"]:
        raise ValueError("Incomplete or changed candidates")
    rows = lines(run / "candidates.jsonl")
    if len(rows) != 25 or len({(r["id"], r["candidate"]) for r in rows}) != 25:
        raise ValueError("Incomplete candidate matrix")
    known = {}
    for entry in read_json(ROOT / "experiments/local_expert_v8/results.json")["joined_review"]:
        h, obs = entry["row"]["image_sha256"], entry["observation"]
        if h in known and any(known[h][k] != obs[k] for k in ("clauses", "full_visibility")):
            raise ValueError("Conflicting old labels")
        known[h] = obs
    inherited, unseen = [], {}
    for row in rows:
        h = row["image_sha256"]
        if sha256(run / row["image"]) != h:
            raise ValueError("Candidate pixels changed")
        if h in known:
            inherited.append({"row": row, "observation": known[h], "source": "phase8_exact_pixels"})
        else:
            if row["candidate"] == "base":
                raise ValueError("Baseline lacks a frozen visual observation")
            unseen.setdefault(h, []).append(row)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    out.mkdir(parents=True, exist_ok=False)
    public, key = [], []
    for case, (h, group) in enumerate(sorted(unseen.items(), key=lambda x: digest(["phase9_blind", x[0]])), 1):
        if len({r["id"] for r in group}) != 1:
            raise ValueError("One image with different prompts")
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
               "new_unique_images": len(public), "inherited_rows": len(inherited),
               "key_sha256": sha256(out / "key.json"), "inherited_sha256": sha256(out / "inherited.json")})
    print(f"Prepared {len(public)} unique unseen images; {len(inherited)} inherited rows.")


def compare_observations(base, candidate):
    a, b = base["clauses"], candidate["clauses"]
    if set(a) != set(b) or any(type(v) is not bool and v is not None for v in [*a.values(), *b.values(),
                                      base["full_visibility"], candidate["full_visibility"]]):
        raise ValueError("Clause/visibility label contract differs")
    repairs = [k for k in a if a[k] is False and b[k] is True]
    harms = [k for k in a if a[k] is True and b[k] is False]
    visibility_harm = base["full_visibility"] is True and candidate["full_visibility"] is False
    unknown = any(v is None for v in b.values()) or candidate["full_visibility"] is None
    protected = not harms and not visibility_harm and not unknown
    return {"repaired_clauses": repairs, "harmed_clauses": harms, "visibility_harm": visibility_harm,
            "has_unknown": unknown, "accepted_encoded_repair": bool(repairs) and protected,
            "safe_visibility_gain": base["full_visibility"] is False and candidate["full_visibility"] is True and protected,
            "known_harm": bool(harms) or visibility_harm}


def summarize(run, review):
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    for name in ("observations", "key", "inherited"):
        if sha256(review / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Frozen review changed")
    if sha256(run / "result.json") != inputs["result_sha256"]:
        raise ValueError("GPU result changed")
    observed = read_json(review / "observations.json")
    labels = {r["case"]: r for r in observed}
    if len(labels) != len(observed) or set(labels) != set(range(1, inputs["new_unique_images"] + 1)):
        raise ValueError("Review coverage incomplete")
    joined = read_json(review / "inherited.json")
    for entry in read_json(review / "key.json"):
        joined.extend({"row": r, "observation": labels[entry["case"]], "source": "phase9_blinded_unique_image"} for r in entry["rows"])
    if len(joined) != 25:
        raise ValueError("Incomplete join")
    baselines = {r["row"]["id"]: r["observation"] for r in joined if r["row"]["candidate"] == "base"}
    per_query, per_candidate = {}, {}
    for entry in joined:
        row = entry["row"]
        entry["comparison"] = compare_observations(baselines[row["id"]], entry["observation"])
    for ident in baselines:
        group = [r for r in joined if r["row"]["id"] == ident and r["row"]["candidate"] != "base"]
        per_query[ident] = {"baseline": baselines[ident],
             "encoded_repair_candidates": [r["row"]["candidate"] for r in group if r["comparison"]["accepted_encoded_repair"]],
             "visibility_gain_candidates": [r["row"]["candidate"] for r in group if r["comparison"]["safe_visibility_gain"]],
             "known_harm_candidates": [r["row"]["candidate"] for r in group if r["comparison"]["known_harm"]],
             "unknown_candidates": [r["row"]["candidate"] for r in group if r["comparison"]["has_unknown"]]}
    for name in ("ocr@0.5", "ocr@1", "geneval@0.5", "geneval@1"):
        group = [r for r in joined if r["row"]["candidate"] == name]
        if len(group) != 5 or len({r["row"]["id"] for r in group}) != 5:
            raise ValueError("Missing candidate rows")
        per_candidate[name] = {key: sum(r["comparison"][key] for r in group) for key in
                              ("accepted_encoded_repair", "safe_visibility_gain", "known_harm", "has_unknown")}
    campaign = ROOT / "experiments/local_expert_v9"
    result = {"scope": "fresh teacher action headroom on selected current-policy states, not a policy gain",
              "per_query": per_query, "per_candidate": per_candidate, "joined_review": joined,
              "gpu_result": read_json(run / "result.json"), "review_inputs": inputs}
    write_json(campaign / "results.json", result)
    write_json(campaign / "observations.json", observed)
    write_json(campaign / "observations_frozen.json", frozen)
    print(json.dumps({"per_query": per_query, "per_candidate": per_candidate}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=["prepare", "summarize"])
    p.add_argument("--run", required=True)
    p.add_argument("--review", required=True)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else summarize)(Path(args.run).resolve(), Path(args.review).resolve())
