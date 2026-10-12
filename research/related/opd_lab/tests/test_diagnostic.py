from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.diagnostic import make_manifest, run_diagnostic, validate_manifest
from opdlab.io import ROOT, read_json, write_json
from opdlab.method import Candidate, Verdict, select
from opdlab.runtime import prepare, verify_run


def test_round_robin_uses_query_ordinal_not_reward_or_rank():
    config = Experiment(method="round_robin")
    candidates = [Candidate("noop", None, 0, torch.zeros(1), 0, Verdict((True,), (1,))),
                  Candidate("ocr@1", "ocr", 1, torch.ones(1), 1, Verdict((False,), (0,))),
                  Candidate("geneval@1", "geneval", 1, torch.ones(1), 1, Verdict((False,), (0,)))]
    assert [select(candidates, config, seed=91, query_ordinal=i) for i in range(6)] == [1, 2, 1, 2, 1, 2]
    with pytest.raises(ValueError, match="ordinal"):
        select(candidates, config, 0)


def test_manifest_shared_noise_and_incomplete_grid_rejection(tmp_path):
    dataset = ROOT / "data/demo/toy.jsonl"
    config = Experiment(backend="toy", dataset=str(dataset), steps=4)
    manifest = make_manifest(dataset, "train", [42, 43], [0, 1, 3], 4, tmp_path / "queries.json")
    validate_manifest(manifest, config, load_records(dataset, "train"))
    assert len(manifest["queries"]) == 12
    assert len({r["seed"] for r in manifest["queries"][:3]}) == 1
    manifest["queries"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_manifest(manifest, config, load_records(dataset, "train"))


def test_frozen_diagnostic_saves_complete_evidence_and_resumes(tmp_path):
    dataset = ROOT / "data/demo/toy.jsonl"
    query_path = tmp_path / "queries.json"
    make_manifest(dataset, "train", [42], [0, 1, 3], 4, query_path)
    config = Experiment(backend="toy", dataset=str(dataset), steps=4, diagnostic_manifest=str(query_path))
    run = prepare(config, tmp_path / "run", None)
    verify_run(run)
    run_diagnostic(config, None, None, run)
    result = read_json(run / "result.json")
    assert result["queries"] == 6 and result["candidates"] == 30
    assert result["noop_identical_across_positions"] and result["optimizer_updates"] == 0
    from opdlab.report import diagnostic_report
    report = diagnostic_report(run, tmp_path / "report")
    assert report["prompts"] == 2 and report["queries"] == 6
    assert report["fixed_arms_all_positions"]["ocr@1"]["utility_delta"]["prompts"] == 2
    paths = sorted((run / "query_results").glob("*/result.json"))
    timestamps = [p.stat().st_mtime_ns for p in paths]
    run_diagnostic(config, None, None, run)
    assert [p.stat().st_mtime_ns for p in paths] == timestamps
    # A changed image must not be accepted as a completed query on resumption.
    row = read_json(paths[0])
    torch.save(torch.ones(1, 2), run / row["candidates"][0]["image"])
    with pytest.raises(ValueError, match="evidence changed"):
        run_diagnostic(config, None, None, run)


def test_constructed_pilot_has_balanced_families_and_disjoint_text():
    path = ROOT / "data/joint_pilot_v1/records.jsonl"
    records = {split: load_records(path, split) for split in ["train", "validation", "test"]}
    assert [len(records[s]) for s in records] == [64, 32, 64]
    words = {split: {r["constraints"][0]["text"] for r in rows} for split, rows in records.items()}
    assert not words["train"] & words["validation"]
    assert not words["train"] & words["test"]
    assert not words["test"] & words["validation"]


def test_clustered_interval_does_not_count_repeated_states_as_new_prompts():
    from opdlab.report import prompt_summary
    once = prompt_summary([("a", 0), ("b", 1)])
    repeated = prompt_summary([("a", 0)] * 100 + [("b", 1)] * 100)
    assert repeated["prompts"] == 2 and repeated["observations"] == 200
    assert once["ci95"] == repeated["ci95"] and once["mean"] == repeated["mean"] == 0.5
