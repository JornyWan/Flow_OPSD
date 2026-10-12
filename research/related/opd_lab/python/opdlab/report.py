"""Audited frozen-intervention summaries and an identity-hidden image review sheet."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path

import numpy as np

from .diagnostic import validate_manifest, validate_saved
from .config import Experiment
from .data import load_records
from .io import digest, read_json, write_json


def prompt_summary(values):
    """Equal prompt weighting; seeds and intervention positions are repeated measures."""
    grouped = defaultdict(list)
    for prompt, value in values:
        grouped[prompt].append(float(value))
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    interval = None
    if len(means) > 1:
        rng = np.random.default_rng(0)
        samples = rng.choice(means, (2000, len(means)), replace=True).mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"mean": float(means.mean()) if len(means) else None, "ci95": interval,
            "prompts": len(means), "observations": len(values)}


def arm_summary(rows, name, record_map=None):
    metrics = defaultdict(list)
    for row in rows:
        base = row["candidates"][0]
        candidate = next(c for c in row["candidates"] if c["name"] == name)
        gain = candidate["utility"] - base["utility"]
        values = {"utility_delta": gain, "joint_success_delta": int(candidate["joint_success"]) - int(base["joint_success"]),
                  "negative_gain_rate": gain < 0, "positive_gain_rate": gain > 0,
                  "collateral_damage_rate": candidate["collateral_damage"],
                  "admissible_positive_rate": candidate["admissible"] and gain > 0}
        if any(base["satisfied"]):
            values["damage_given_existing_success"] = candidate["collateral_damage"]
        if record_map is not None:
            kinds = [c["kind"] for c in record_map[row["id"]]["constraints"]]
            for kind in sorted(set(kinds)):
                indices = [i for i, value in enumerate(kinds) if value == kind]
                values[f"{kind}_clause_success_delta"] = np.mean([
                    int(candidate["satisfied"][i]) - int(base["satisfied"][i]) for i in indices])
        for metric, value in values.items():
            metrics[metric].append((row["id"], value))
    return {metric: prompt_summary(values) for metric, values in metrics.items()}


def diagnostic_report(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    from .runtime import verify_run
    asset_audit = verify_run(run)
    config = Experiment.load(run / "config.json")
    records = load_records(config.dataset, config.split)
    record_map = {r["id"]: r for r in records}
    queries = read_json(config.diagnostic_manifest)
    validate_manifest(queries, config, records)
    completion = read_json(run / "result.json")
    if completion["status"] != "diagnosed" or completion["invalid_candidates"]:
        raise ValueError("Invalid/incomplete diagnostic cannot be promoted into a scientific comparison")
    signature = digest({"manifest": read_json(run / "manifest.json"), "queries": queries})
    rows = []
    for spec in queries["queries"]:
        row = read_json(run / "query_results" / spec["query_id"] / "result.json")
        validate_saved(row, run, signature, spec)
        if any(not c["valid"] for c in row["candidates"]):
            raise ValueError("Invalid candidate")
        rows.append(row)
    if "query_rows_digest" in completion and digest(rows) != completion["query_rows_digest"]:
        raise ValueError("Diagnostic query scores/records changed after completion")
    arms = [c["name"] for c in rows[0]["candidates"]]
    if any([c["name"] for c in r["candidates"]] != arms for r in rows):
        raise ValueError("Candidate sets differ between queries")
    summary = {"run": str(run), "queries": len(rows), "prompts": len(queries["record_ids"]),
               "asset_validation_notes": asset_audit,
               "query_rows_integrity": "completion_digest_verified" if "query_rows_digest" in completion else "legacy_image_and_input_checks_only",
               "trajectories": completion["unique_noise_trajectories"], "optimizer_updates": 0,
               "statistical_unit": "prompt; average seeds/positions within prompt; percentile bootstrap 2000 resamples, seed 0",
               "interpretation": "Development-set, frozen one-step interventions. Selected/oracle gains reuse the selecting verifier and do not establish trained-policy improvement or cross-sampler robustness.",
               "fixed_arms_all_positions": {a: arm_summary(rows, a, record_map) for a in arms[1:]},
               "by_position": {}, "by_group": {}, "counts": completion["counts"],
               "query_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows)}
    for index in queries["indices"]:
        subset = [r for r in rows if r["query_index"] == index]
        summary["by_position"][str(index)] = {
            "sigma": subset[0]["sigma"], "sigma_next": subset[0]["sigma_next"],
            "fixed_arms": {a: arm_summary(subset, a, record_map) for a in arms[1:]}}
    for group in sorted({record_map[r["id"]]["group"] for r in rows}):
        subset = [r for r in rows if record_map[r["id"]]["group"] == group]
        summary["by_group"][group] = {a: arm_summary(subset, a, record_map) for a in arms[1:]}
    selected, headroom, accepted = [], [], []
    for row in rows:
        base = row["candidates"][0]
        chosen = next(c for c in row["candidates"] if c["name"] == row["selected"])
        selected.append((row["id"], chosen["utility"] - base["utility"]))
        headroom.append((row["id"], max([0.0] + [c["utility"] - base["utility"] for c in row["candidates"] if c["admissible"]])))
        accepted.append((row["id"], row["accepted"]))
    summary["selected_local_gain_same_verifier"] = prompt_summary(selected)
    summary["oracle_admissible_headroom_same_verifier"] = prompt_summary(headroom)
    summary["acceptance"] = prompt_summary(accepted)
    out.mkdir(parents=True)
    write_json(out / "summary.json", summary)
    lines = ["# Frozen local expert diagnostic", "", summary["interpretation"], "",
             f"Queries: {len(rows)}; prompts: {summary['prompts']}; unique trajectories: {summary['trajectories']}; optimizer updates: 0.",
             "", "Intervals resample prompts, not individual seeds or query positions. These are exploratory intervals without multiplicity correction.", "",
             "| Fixed action, averaged over positions | Mean utility delta | Prompt bootstrap 95% interval | Negative gain | Existing constraint damaged |",
             "|---|---:|---|---:|---:|"]
    for name, values in summary["fixed_arms_all_positions"].items():
        ci = values["utility_delta"]["ci95"]
        lines.append(f"| {name} | {values['utility_delta']['mean']:.6f} | {ci} | {values['negative_gain_rate']['mean']:.3f} | {values['collateral_damage_rate']['mean']:.3f} |")
    lines += ["", f"Verifier-selected local gain: {summary['selected_local_gain_same_verifier']['mean']:.6f}; acceptance: {summary['acceptance']['mean']:.3f}.",
              "This selection statistic is descriptive and is not independent validation.", "",
              "Position-specific results and actual sigma values are in `summary.json`.",
              "`review.html` hides teacher identity, acceptance and scores. `review_key.json` reveals the mapping; inspect the images before consulting it.",
              "A review by the implementing assistant is a qualitative audit, not an independent human study."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    gallery, key = [], []
    record_map = {r["id"]: r for r in records}
    for number, row in enumerate(rows, 1):
        qid = row["query"]["query_id"]
        candidates = sorted(row["candidates"], key=lambda c: digest(["review-v1", qid, c["name"]]))
        cards = []
        mapping = {}
        for letter, candidate in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", candidates):
            relative = os.path.relpath(run / candidate["image"], out)
            cards.append(f'<figure><figcaption>{letter}</figcaption><a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}" alt="candidate {letter}"></a></figure>')
            mapping[letter] = {"name": candidate["name"], "image": candidate["image"], "verdict": candidate}
        prompt = html.escape(record_map[row["id"]]["prompt"])
        gallery.append(f'<section><h2>Case {number:03d}</h2><p>{prompt}</p><div class="grid">{"".join(cards)}</div></section>')
        key.append({"case": number, "query_id": qid, "query": row["query"], "mapping": mapping, "selected": row["selected"]})
    document = '<!doctype html><meta charset="utf-8"><title>Local expert candidate review</title><style>body{font:16px system-ui;margin:24px;background:#f4f4f4;color:#222}.grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:280px}img{width:100%}section{background:white;padding:20px;margin-bottom:24px}figcaption{font-weight:bold}</style><h1>Candidate review</h1><p>Check text, object count, color, spatial relations and visual quality before reading the key. Candidate identities and verifier scores are hidden.</p>' + "".join(gallery)
    (out / "review.html").write_text(document)
    write_json(out / "review_key.json", key)
    return summary
