from dataclasses import replace

import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.method import Candidate, Verdict, admissible, collect_query, select, transition_mean_loss
from opdlab.verifiers import ToyVerifier, text_progress


def candidate(name, sat, progress, distance=1.0, teacher="ocr", alpha=1.0, **kwargs):
    return Candidate(name, teacher, alpha, torch.tensor([float(distance)]), distance,
                     Verdict(tuple(sat), tuple(progress), **kwargs))


def test_collateral_damage_is_rejected_even_when_utility_improves():
    config = Experiment()
    base = candidate("noop", [True, False, False], [1, 0, 0], teacher=None, alpha=0)
    harm = candidate("ocr@1", [False, True, True], [0.9, 1, 1])
    assert harm.verdict.utility(0.5) > base.verdict.utility(0.5)
    assert select([base, harm], config, 0) == 0
    assert select([base, harm], replace(config, protect_satisfied=False), 0) == 1


def test_verified_fixed_does_not_select_a_different_expert():
    c = Experiment(method="verified_fixed", fixed_teacher="geneval")
    base = candidate("noop", [False, False], [.2, .2], teacher=None, alpha=0)
    other = candidate("ocr@1", [True, True], [1, 1])
    fixed = candidate("geneval@1", [False, False], [.1, .1], teacher="geneval")
    assert select([base, other, fixed], c, 0) == 0
    assert select([base, other, fixed], replace(c, method="verified"), 0) == 1


def test_noop_wins_ties_and_invalid_baseline_fails_closed():
    base = candidate("noop", [True, False], [1, 0], teacher=None, alpha=0)
    tied = candidate("ocr@1", [True, False], [1, 0])
    assert select([base, tied], Experiment(), 0) == 0
    invalid = replace(base, verdict=replace(base.verdict, valid=False))
    perfect = candidate("ocr@1", [True, True], [1, 1])
    assert select([invalid, perfect], Experiment(), 0) == 0


def test_valid_benefit_tiebreaks_to_smallest_action():
    base = candidate("noop", [False], [0], teacher=None, alpha=0)
    far = candidate("ocr@1", [True], [1], distance=4)
    near = candidate("ocr@0.5", [True], [1], distance=1, alpha=0.5)
    assert select([base, far, near], Experiment(), 0) == 2


def test_quality_margin_and_verifier_validity():
    c = Experiment(quality_floor=4, quality_drop_tolerance=0.1)
    base = Verdict((False,), (0,), quality=5)
    assert not admissible(base, Verdict((True,), (1,), quality=4.5), c)
    assert not admissible(base, Verdict((True,), (1,)), c)
    assert not admissible(base, Verdict((True,), (1,), valid=False, quality=5), c)
    assert admissible(base, Verdict((True,), (1,), quality=5), c)


def test_random_and_disagreement_do_not_use_terminal_verdict():
    base = candidate("noop", [True], [1], teacher=None, alpha=0)
    harm = candidate("ocr@1", [False], [0], distance=2)
    assert select([base, harm], Experiment(method="fixed"), 0) == 1
    assert select([base, harm], Experiment(method="random", accept_probability=1), 0) == 1
    assert select([base, harm], Experiment(method="random", accept_probability=0), 0) == 0
    assert select([base, harm], Experiment(method="disagreement", disagreement_min=3), 0) == 0


def test_detached_targets_and_exact_deterministic_opd_loss():
    student = torch.tensor([[2.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 3.0]], requires_grad=True)
    state, dt = torch.tensor([[0.2, -0.5]]), -0.2
    native = 0.5 * ((state + dt * student) - (state + dt * teacher.detach())).square().mean()
    local = transition_mean_loss(student, teacher, dt)
    torch.testing.assert_close(local, native)
    local.backward()
    assert teacher.grad is None
    torch.testing.assert_close(student.grad, dt**2 * (student.detach() - teacher.detach()) / 2)


def toy_record():
    return {"id": "x", "prompt": "toy", "constraints": [
        {"id": "x", "kind": "toy", "axis": 0, "target": 0.0, "tolerance": 0.15},
        {"id": "y", "kind": "toy", "axis": 1, "target": 0.0, "tolerance": 0.15}]}


def test_noop_matches_full_rollout_and_teachers_only_predict_once():
    config = Experiment(backend="toy", steps=4)
    backend, verifier = ToyBackend(config), ToyVerifier(config)
    q = collect_query(backend, verifier, toy_record(), 123, 1, config)
    reference = verifier(backend.generate(toy_record(), 123, 4, adapter="behavior"), toy_record())
    assert q["candidates"][0].verdict == reference
    assert backend.counters["forward_ocr"] == backend.counters["forward_geneval"] == 1
    assert not q["target"].requires_grad and not q["state"].requires_grad
    assert len(q["candidates"]) == 5
    # Collection must not mutate the student or frozen behavior.
    torch.testing.assert_close(backend.weight, torch.zeros(2))
    torch.testing.assert_close(backend.behavior, torch.zeros(2))


def test_ocr_exact_match_is_not_substring_reward():
    assert text_progress("  FRESH   Tea ", "fresh tea") == (True, 1.0)
    exact, progress = text_progress("fresh tea sale", "fresh tea")
    assert not exact and 0 < progress < 1
    assert text_progress("", "required") == (False, 0.0)


@pytest.mark.parametrize("updates", [dict(accumulation=5), dict(alphas=[0]), dict(ema_decay=1),
                                   dict(method="typo"), dict(quality_drop_tolerance=-1)])
def test_invalid_configs_are_rejected(updates):
    with pytest.raises(ValueError):
        Experiment(**updates).validate()
