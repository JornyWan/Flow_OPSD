"""Pure terminal verification and detached one-step distillation primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch


@dataclass(frozen=True)
class Verdict:
    satisfied: tuple[bool, ...]
    progress: tuple[float, ...]
    valid: bool = True
    quality: float | None = None
    error: str | None = None
    details: dict | None = None

    def __post_init__(self):
        if len(self.satisfied) != len(self.progress) or not self.progress:
            raise ValueError("Verdicts need equally sized nonempty constraint vectors")
        if any(type(v) is not bool for v in self.satisfied):
            raise ValueError("Constraint satisfaction must be boolean")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.progress):
            raise ValueError("Progress must be finite in [0,1]")
        if self.quality is not None and not math.isfinite(self.quality):
            raise ValueError("Nonfinite quality")

    def utility(self, joint_weight):
        return joint_weight * float(all(self.satisfied)) + (1 - joint_weight) * sum(self.progress) / len(self.progress)

    def as_dict(self, joint_weight):
        return {**asdict(self), "joint_success": self.valid and all(self.satisfied),
                "utility": self.utility(joint_weight) if self.valid else None}


@dataclass
class Candidate:
    name: str
    teacher: str | None
    alpha: float
    action: torch.Tensor
    distance: float
    verdict: Verdict
    image_path: str | None = None
    image_sha256: str | None = None


def admissible(base, candidate, config):
    if not base.valid or not candidate.valid:
        return False
    if len(base.satisfied) != len(candidate.satisfied):
        raise ValueError("Cannot compare different constraint sets")
    if config.protect_satisfied and any(old and not new for old, new in zip(base.satisfied, candidate.satisfied)):
        return False
    if config.quality_floor is not None:
        if candidate.quality is None or candidate.quality < config.quality_floor:
            return False
    if config.quality_drop_tolerance is not None:
        if base.quality is None or candidate.quality is None or candidate.quality < base.quality - config.quality_drop_tolerance:
            return False
    return True


def select(candidates, config, seed, query_ordinal=None):
    if not candidates or candidates[0].name != "noop":
        raise ValueError("Candidate zero must be the behavior/noop action")
    base = candidates[0]
    fixed = next(c for c in candidates[1:] if c.teacher == config.fixed_teacher and c.alpha == config.fixed_alpha)
    if config.method == "round_robin":
        if query_ordinal is None:
            raise ValueError("Round-robin routing requires a global query ordinal")
        teacher = config.teachers[query_ordinal % len(config.teachers)]
        return next(i for i, c in enumerate(candidates)
                    if c.teacher == teacher and c.alpha == config.fixed_alpha)
    if config.method in {"fixed", "single_query"}:
        return candidates.index(fixed)
    if config.method == "random":
        # Random acceptance of the SAME fixed action controls screening, not an oracle proposal policy.
        return candidates.index(fixed) if random.Random(seed).random() < config.accept_probability else 0
    if config.method == "disagreement":
        return candidates.index(fixed) if fixed.distance > config.disagreement_min else 0
    winner = 0
    base_utility = base.verdict.utility(config.utility_joint_weight)
    best = None
    for i, c in enumerate(candidates[1:], 1):
        if config.method == "verified_fixed" and c is not fixed:
            continue
        if not admissible(base.verdict, c.verdict, config):
            continue
        utility = c.verdict.utility(config.utility_joint_weight)
        if utility <= base_utility + config.min_gain:
            continue
        key = (utility, -c.distance, -i)
        if best is None or key > best:
            winner, best = i, key
    return winner


def transition_mean_loss(prediction, target, dt):
    """Original deterministic flow OPD loss for one teacher/query: .5*mean((dt*dv)^2).

    No division by accepted count, time normalization, reward weighting, or
    gradient through the detached target. dt is the actual scheduler sigma step.
    """
    return 0.5 * ((prediction.float() - target.detach().float()) * dt).square().mean()


@torch.no_grad()
def collect_query(backend, verifier, record, seed, index, config, query_ordinal=None, save_image=None,
                  defer_selection=False):
    context = backend.encode(record["prompt"])
    schedule = backend.schedule(config.steps)
    state = backend.initial(seed)
    for j in range(index):
        state = backend.step(state, backend.predict(state, j, context, "behavior"), j)
    base_action = backend.predict(state, index, context, "behavior").detach()
    if not torch.isfinite(base_action).all():
        raise FloatingPointError("Nonfinite behavior prediction")
    actions = [("noop", None, 0.0, base_action)]
    for teacher in config.teachers:
        expert = backend.predict(state, index, context, teacher).detach()
        if not torch.isfinite(expert).all():
            raise FloatingPointError(f"Nonfinite teacher prediction: {teacher}")
        for alpha in config.alphas:
            actions.append((f"{teacher}@{alpha:g}", teacher, alpha,
                            (base_action + alpha * (expert - base_action)).detach()))
    candidates = []
    for name, teacher, alpha, action in actions:
        # Stateless deterministic Euler: no solver history, no branch-specific RNG.
        endpoint = backend.step(state.clone(), action, index)
        for j in range(index + 1, config.steps):
            endpoint = backend.step(endpoint, backend.predict(endpoint, j, context, "behavior"), j)
        image = backend.decode(endpoint)
        verdict = verifier(image, record)
        distance = float((action.float() - base_action.float()).square().mean())
        saved = save_image(name, image) if save_image is not None else (None, None)
        candidates.append(Candidate(name, teacher, alpha, action.detach(), distance, verdict, *saved))
    # Batch protocols decide only after all candidate evidence is collected.
    winner = 0 if defer_selection else select(candidates, config, seed, query_ordinal)
    return {"record": record, "seed": seed, "index": index,
            "state": state.detach(), "context": context,
            "dt": float(schedule[index + 1] - schedule[index]),
            "target": candidates[winner].action.detach(), "accepted": winner != 0,
            "candidates": candidates, "winner": winner}


def query_log(query, config):
    base = query["candidates"][0].verdict
    rows = []
    for c in query["candidates"]:
        value = c.verdict.as_dict(config.utility_joint_weight)
        value.update(name=c.name, teacher=c.teacher, alpha=c.alpha, distance=c.distance,
                     image=c.image_path, image_sha256=c.image_sha256,
                     admissible=admissible(base, c.verdict, config),
                     collateral_damage=base.valid and c.verdict.valid and any(
                         a and not b for a, b in zip(base.satisfied, c.verdict.satisfied)))
        rows.append(value)
    return {"id": query["record"]["id"], "seed": query["seed"], "query_index": query["index"],
            "dt": query["dt"], "accepted": query["accepted"],
            "selected": query["candidates"][query["winner"]].name, "candidates": rows}
