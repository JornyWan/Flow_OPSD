import copy
import pytest

from opdlab.teacher_opportunity import pairs_from_plan
from review_teacher_opportunity import diagnose


def test_complete_cohort_selection_and_no_dropping_unknown():
    pairs = [{"pair_id": f"{i}-{j}", "id": str(i), "single_action": True,
              "cohorts": ["positive_fresh_seed"], "outcome": None if j else False}
             for i in range(6) for j in range(2)]
    assert pairs_from_plan({"evaluation": pairs + [{"cohorts": ["added_fitting_states"]}]}) == pairs
    with pytest.raises(AssertionError):
        pairs_from_plan({"evaluation": pairs[:-1]})
    with pytest.raises(AssertionError):
        pairs_from_plan({"evaluation": pairs[:-1] + [pairs[0]]})


def test_partial_repair_unknown_and_determinate_learning_gap_are_separate():
    def obs(text, count, visibility=True):
        return {"clauses": {"text": text, "count": count}, "full_visibility": visibility}
    sample = {"noop": obs(False, False), "student_single": obs(False, False),
              "ocr@1": obs(True, False), "geneval@1": obs(False, True)}
    _, flags = diagnose(sample)
    assert flags["teacher_safe_only_partial"] and flags["teacher_safe_local_determinate_not_safe"]
    assert not flags["any_teacher_safe_complete"] and not flags["oracle_complete_including_noop"]
    sample["ocr@1"] = obs(True, True)
    assert diagnose(sample)[1]["teacher_safe_complete_local_determinate_incomplete"]
    sample["student_single"] = obs(None, False)
    assert not diagnose(sample)[1]["teacher_safe_complete_local_determinate_incomplete"]
    ambiguous = copy.deepcopy(sample)
    ambiguous.update({"ocr@1": obs(None, True), "geneval@1": obs(None, False)})
    assert diagnose(ambiguous)[1]["teacher_unresolved_without_known_safe"]


def test_replays_precede_teacher_queries_and_keep_frozen_suffix(tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    import torch
    from opdlab import teacher_opportunity as study
    from opdlab.action_transfer import trace_baseline
    from opdlab.backends import ToyBackend
    from opdlab.config import Experiment
    from opdlab.fitting_probe import tensor_digest
    from opdlab.learnability import single_endpoint

    class Backend(ToyBackend):
        def encode(self, prompt):
            return (torch.zeros(1), torch.ones(1))

    config = Experiment(backend="toy", steps=10)
    record = {"id": "a", "prompt": "x"}
    monkeypatch.setattr(study, "load_records", lambda *args: [record])
    source = tmp_path / "source"
    (source / "eval_queries").mkdir(parents=True)
    initial, student = {"weight": torch.zeros(2)}, {"weight": torch.tensor([.2, -.1])}
    for name, state in (("initial", initial), ("student", student)):
        torch.save(state, source / f"{name}.pt")
    backend = Backend(config)
    q, image = trace_baseline(backend, record, 42, 10, 3)
    torch.save(q, source / "eval_queries/p.pt")
    ref = {"noop": {"image_sha256": tensor_digest({"image": image}), "noise_sha256": q["noise_sha256"]}}
    backend.load_student(student, refresh_behavior=False)
    action = backend.predict(q["state"], 3, q["context"], "student")
    ref["student_single"] = {"image_sha256": tensor_digest({"image": single_endpoint(backend, q, action, 10)})}
    plan = {"source": str(source), "index": 3, "pairs": [{"pair_id": "p", "id": "a", "seed": 42}],
            "references": {"p": ref}, "checkpoints": {n: str(source / f"{n}.pt") for n in ("initial", "student")},
            "expected_counts": {"backward_queries": 0, "forward_ocr": 1, "forward_geneval": 1}}
    calls = []
    def save(run, b, verifier, record, image, identity, weight):
        calls.append((identity[-1], b.counters["forward_ocr"] + b.counters["forward_geneval"]))
        verifier.counters["verifier_images"] += 1
        return {"valid": True, "image_sha256": tensor_digest({"image": image})}
    monkeypatch.setattr(study, "save_verdict", save)
    for corrupt in (False, True):
        run = tmp_path / str(corrupt)
        run.mkdir()
        b, v = Backend(config), SimpleNamespace(counters=Counter())
        if corrupt:
            plan["references"]["p"]["student_single"]["image_sha256"] = "bad"
            with pytest.raises(RuntimeError, match="exact replay"):
                study.execute(b, v, config, plan, run)
            assert b.counters["forward_ocr"] == b.counters["forward_geneval"] == 0
        else:
            result = study.execute(b, v, config, plan, run)
            assert calls[:2] == [("noop", 0), ("student_single", 0)]
            assert [name for name, _ in calls] == ["noop", "student_single", "ocr@1", "geneval@1"]
            assert result["counts"]["forward_behavior"] == 19  # one noop check plus three6-step suffixes
            assert tensor_digest(b.behavior_state()) == tensor_digest(initial)
            assert tensor_digest(b.student_state()) == tensor_digest(student)
