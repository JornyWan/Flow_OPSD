import pytest
from opdlab.acceptance_collection import coverage_gate, new_pairs


def test_collection_evaluation_and_history_are_disjoint():
    source = {"evaluation": [{"id": str(i), "replicate": r, "cohort": "positive_fresh_seed" if i < 6 else "preservation_fresh_seed",
                              "cohorts": ["all_fresh"]} for i in range(16) for r in range(2)]}
    collection, evaluation = new_pairs(source, [])
    assert len(collection) == 48 and len(evaluation) == 32
    assert not {p["seed"] for p in collection} & {p["seed"] for p in evaluation}
    assert len({p["pair_id"] for p in collection + evaluation}) == 80
    with pytest.raises(AssertionError):
        new_pairs(source, [collection[0]["seed"]])
    with pytest.raises(AssertionError):
        new_pairs(source, [evaluation[0]["seed"]])


def test_gate_requires_breadth_and_complete_not_only_many_partial_repairs():
    cases = [{"id": str(i), "safe_repair": j < 2 or i == 0 and j < 4, "complete_scene": i < 2 and j < 2}
             for i in range(6) for j in range(8)]
    assert coverage_gate(cases)["passed"]
    # Concentration on one prompt must not pass even with enough total complete repairs.
    concentrated = [{"id": str(i), "safe_repair": i == 0, "complete_scene": True} for i in range(6) for j in range(8)]
    assert not coverage_gate(concentrated)["passed"]
    assert not coverage_gate([{**c, "complete_scene": False} for c in cases])["passed"]


def test_collection_waits_for_replay_and_uses_one_teacher_per_new_state(tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    import torch
    from opdlab import acceptance_collection as study
    from opdlab.action_transfer import trace_baseline
    from opdlab.backends import ToyBackend
    from opdlab.config import Experiment
    from opdlab.fitting_probe import tensor_digest
    from opdlab.learnability import single_endpoint

    class Backend(ToyBackend):
        def encode(self, prompt):
            return (torch.zeros(1), torch.ones(1))

    config, record = Experiment(backend="toy", steps=10), {"id": "x", "prompt": "x"}
    monkeypatch.setattr(study, "load_records", lambda *args: [record])
    b = Backend(config)
    initial = b.student_state()
    torch.save(initial, tmp_path / "initial.pt")
    q, image = trace_baseline(b, record, 42, 10, 3)
    teacher = b.predict(q["state"], 3, q["context"], "geneval")
    q["actions"]["geneval@1"] = teacher
    torch.save(q, tmp_path / "query.pt")
    refs = {"noop": {"image_sha256": tensor_digest({"image": image})},
            "geneval@1": {"image_sha256": tensor_digest({"image": single_endpoint(b, q, teacher, 10)})}}
    plan = {"student": str(tmp_path / "initial.pt"), "references": [{"pair": {"pair_id": "ref", "id": "x", "seed": 42},
            "saved_query": str(tmp_path / "query.pt"), "images": refs}],
            "collection": [{"pair_id": "new", "id": "x", "seed": 43}],
            "expected_counts": {"forward_student": 20, "forward_behavior": 13, "forward_geneval": 2,
                                "forward_ocr": 0, "backward_queries": 0}}
    calls = []
    def save(run, backend, verifier, record, image, identity, weight):
        calls.append((identity[1], identity[-1]))
        return {"valid": True, "image_sha256": tensor_digest({"image": image})}
    monkeypatch.setattr(study, "save_verdict", save)
    run = tmp_path / "good"
    run.mkdir()
    result = study.execute(Backend(config), SimpleNamespace(counters=Counter()), config, plan, run)
    assert calls == [("reference", "noop"), ("reference", "geneval@1"), ("collection", "noop"), ("collection", "geneval@1")]
    saved = torch.load(run / "queries/new.pt", weights_only=False)
    assert set(saved["actions"]) == {"noop", "geneval@1"}
    assert result["counts"].get("backward_queries", 0) == 0
    refs["geneval@1"]["image_sha256"] = "wrong"
    failed = tmp_path / "failed"
    failed.mkdir()
    with pytest.raises(RuntimeError, match="exact replay"):
        study.execute(Backend(config), SimpleNamespace(counters=Counter()), config, plan, failed)
    assert not (failed / "endpoints.jsonl").exists() and not (failed / "queries").exists()
