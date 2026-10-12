from collections import Counter
import json

import pytest
import torch

from opdlab.annotation_probe import ARMS, cohort, control_targets, evaluation_pairs, fit_models
from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.engine import seed_all
from opdlab.fitting_probe import tensor_digest


def fixture():
    rows, review, records = [], [], []
    positive_cases = iter((2, 6, 17, 34, 42, 58))
    for group in ("color", "count", "relation", "count_color"):
        for i in range(16):
            ident = f"train-{group}-{i:02d}"
            positive = group != "count_color" and i < 2
            teacher = "ocr@1" if group == "color" else "geneval@1"
            candidates = [{"name": n, "distance": j, "image": "fixture.png", "image_sha256": "image-hash"}
                          for j, n in enumerate(("noop", "ocr@1", "geneval@1"))]
            rows.append({"id": ident, "seed": 1000 + len(rows), "query_index": 3,
                         "saved_query": "fixture.pt", "saved_query_sha256": "query-hash", "action_hashes": {}, "candidates": candidates})
            review.append({"id": ident, "group": group, "case": next(positive_cases) if positive else 100 + len(rows),
                           "comparisons": {n: {"safe_repair": positive and n == teacher, "repaired_clauses": ["text"]}
                                           for n in ("ocr@1", "geneval@1")}})
            records.append({"id": ident, "group": group, "split": "train"})
    return cohort(rows, review), records


def test_matching_coverage_and_nested_unfiltered_control():
    specs, _ = fixture()
    assert len(specs) == 12 and sum(s["label_target"] != "noop" for s in specs) == 6
    assert {s["source_case"] for s in specs if s["label_target"] != "noop"} == {2, 6, 17, 34, 42, 58}
    targets = control_targets(specs)
    assert targets == control_targets(json.loads(json.dumps(specs, sort_keys=True)))
    for arm, multiplier in (("label_selected", 1), ("random_matched", 1), ("all_teacher", 2)):
        counts = Counter((s["group"], targets[arm][s["id"]]) for s in specs if targets[arm][s["id"]] != "noop")
        expected = Counter((s["group"], s["label_target"]) for s in specs if s["label_target"] != "noop")
        assert counts == {k: multiplier * v for k, v in expected.items()}
    for ident, name in targets["random_matched"].items():
        if name != "noop":
            assert targets["all_teacher"][ident] == name


def test_fresh_evaluation_disjoint_from_saved_targets_and_complete():
    specs, records = fixture()
    excluded = [s["seed"] for s in specs]
    pairs = evaluation_pairs(specs, records, excluded)
    assert Counter(p["cohort"] for p in pairs) == {"positive_fresh_seed": 12, "preservation_fresh_seed": 12,
                                                 "unfitted_prompt_fresh_seed": 8, "fitting_seed_diagnostic": 6}
    assert len({p["pair_id"] for p in pairs}) == 38
    with pytest.raises(ValueError):
        evaluation_pairs(specs, records, excluded + [pairs[0]["seed"]])


def test_toy_arms_reset_optimizer_rng_and_keep_behavior_frozen(tmp_path):
    seed_all(42)
    config = Experiment(backend="toy", precision="no", steps=4, learning_rate=.03)
    backend = ToyBackend(config)
    initial = tensor_digest(backend.student_state())
    points = []
    for i in range(4):
        state = torch.zeros(1, 2)
        points.append({"record": {"id": str(i)}, "state": state, "index": 1, "context": "",
                       "dt": -.25, "label_target": "ocr@1" if i == 0 else "noop",
                       "actions": {"noop": state.clone(), "ocr@1": torch.ones_like(state) * .1,
                                   "geneval@1": torch.ones_like(state) * -.1}})
    targets = {a: {str(i): "ocr@1" if (i == 0 if a == "label_selected" else i == 1 if a == "random_matched" else True) else "noop" for i in range(4)} for a in ARMS}
    reports = fit_models(backend, points, config, {"targets": targets, "updates": 2, "audits": [0, 2]}, tmp_path)
    assert all(r["initial_student_sha256"] == initial and r["counts"]["backward_queries"] == 8 for r in reports.values())
    assert tensor_digest(backend.behavior_state()) == initial
    for a in ARMS:
        saved = torch.load(tmp_path / "checkpoints" / a / "training.pt", weights_only=False)
        assert saved["optimizer_step"] == 2 and saved["optimizer"]["state"]
