"""Descriptive phase15 failure analysis; never change frozen labels or gates."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from opdlab.annotation_probe import ARMS
from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import lines, visual


def build():
    campaign, run = ROOT / "experiments/local_expert_v15", ROOT / "runs/le15_annotation_001"
    report = read_json(campaign / "results.json")
    metrics = lines(run / "metrics.jsonl")
    numerical = {}
    for arm in ARMS:
        numerical[arm] = {}
        for step in (0, 32, 128):
            rows = [r for r in metrics if (r["arm"], r["step"]) == (arm, step)]
            align = [r["label_target_alignment"]["relative_target_mse"] for r in rows if "label_target_alignment" in r]
            numerical[arm][str(step)] = {"own_fitting_loss_mean": float(np.mean([r["fitting_loss"] for r in rows])),
                                        "safe_target_relative_mse_median": float(np.median(align)),
                                        "safe_target_relative_mse_range": [min(align), max(align)]}
        numerical[arm]["own_objective_final_over_initial"] = numerical[arm]["128"]["own_fitting_loss_mean"] / numerical[arm]["0"]["own_fitting_loss_mean"]
    joined = report["joined_review"]
    bycase = defaultdict(dict)
    byprompt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in joined:
        row = entry["row"]
        bycase[entry["case"]][row["model"]] = entry
        byprompt[row["id"]][row["cohort"]][row["model"]].append({"case": entry["case"], "seed": row["seed"],
                                                                  "complete_scene": visual(entry["observation"])["complete_scene"]})
    harmed, repaired = [], []
    for p in report["paired_rows"]:
        if p["model"] != "label_selected" or p["cohort"] == "fitting_seed_diagnostic":
            continue
        detail = {**p, "initial": bycase[p["case"]]["joint128"], "fitted": bycase[p["case"]]["label_selected"]}
        if p["comparison"]["known_harm"]:
            harmed.append(detail)
        if p["comparison"]["safe_repair"]:
            repaired.append(detail)
    causes = Counter(c for p in harmed for c in p["comparison"]["harmed_clauses"])
    causes["visibility"] = sum(p["comparison"]["visibility_harm"] for p in harmed)
    status = read_json(campaign / "slurm_status.json")
    return {"scope": "post-evaluation descriptive analysis, no new GPU calls or changed decisions",
            "results_sha256": sha256(campaign / "results.json"), "metrics_sha256": sha256(run / "metrics.jsonl"),
            "script_sha256": sha256(__file__), "numerical_fitting": numerical, "per_prompt": byprompt,
            "all_fresh_label_harms": harmed, "all_fresh_label_safe_repairs": repaired,
            "harm_fields": dict(causes), "safe_repair_prompt_count": len({r["id"] for r in repaired}),
            "new_allocated_gpu_hours": status["allocated_gpu_hours_including_failed_attempts_so_far"],
            "inherited_phase12_collection_gpu_hours": 428 / 3600,
            "cost_exclusions": "Earlier common checkpoint/expert training and incompletely timed annotation labor; not a cumulative project cost."}


def illustrate():
    campaign, run = ROOT / "experiments/local_expert_v15", ROOT / "runs/le15_annotation_001"
    report = read_json(campaign / "results.json")
    indexed = {(e["case"], e["row"]["model"]): e for e in report["joined_review"]}
    sheet = Image.new("RGB", (1024, 1200), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    draw.text((12, 8), 'Same prompt: one bowl, only the word "BLOOM"\nIllustration selected after aggregate analysis; no new evaluation.', font=font, fill="black", spacing=5)
    sources = []
    for row, (case, title) in enumerate(((31, "Fitting seed: repair"), (16, "Fresh seed: regression"))):
        top = 78 + row * 560
        for col, model in enumerate(("joint128", "label_selected")):
            entry = indexed[case, model]
            r, obs = entry["row"], entry["observation"]
            assert r["id"] == "jp1-train-count-12"
            draw.text((col * 512 + 8, top), f"{title} | {'Initial' if col == 0 else 'Label-selected'}", font=font, fill="black")
            draw.text((col * 512 + 8, top + 23), f"Case {case}, seed {r['seed']}, text: {obs['visible_text']}", font=font, fill="black")
            sheet.paste(Image.open(run / r["image"]), (col * 512, top + 48))
            sources.append({"case": case, "model": model, "image": r["image"], "image_sha256": r["image_sha256"]})
    sheet.save(campaign / "fitting_vs_fresh.png")
    write_json(campaign / "example_sources.json", {"selection": "posthoc illustration of a same-prompt seed reversal; not an aggregate estimate",
               "results_sha256": sha256(campaign / "results.json"), "script_sha256": sha256(__file__),
               "sources": sources, "figure_sha256": sha256(campaign / "fitting_vs_fresh.png")})


if __name__ == "__main__":
    write_json(ROOT / "experiments/local_expert_v15/failure_analysis.json", build())
    illustrate()
