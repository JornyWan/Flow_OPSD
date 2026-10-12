"""CPU-only invariants for CODEX-0002 constrained retention.

This does not test SD3 execution.  It isolates the masking and hinge equations so
an implementation error cannot hide behind a successful distributed launch.
"""

import torch


def constrained_retention(current_kl, reference_kl, teacher_idx, rel_tol=0.02, abs_tol=1e-8):
    retain_mask = ((teacher_idx == 1) | (teacher_idx == 2)).to(current_kl.dtype)
    threshold = (1.0 + rel_tol) * reference_kl + abs_tol
    violation = current_kl - threshold
    loss = (retain_mask * torch.relu(violation)).sum() / retain_mask.sum().clamp(min=1.0)
    return loss, retain_mask, violation


def main():
    teacher_idx = torch.tensor([0, 1, 2])
    reference_kl = torch.tensor([0.50, 0.50, 0.50])

    # The direct-reward slot is never part of retention, and an unchanged
    # Student is strictly feasible because of the 2% slack.
    current_kl = reference_kl.clone().requires_grad_(True)
    loss, mask, violation = constrained_retention(current_kl, reference_kl, teacher_idx)
    assert mask.tolist() == [0.0, 1.0, 1.0]
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(current_kl.grad, torch.zeros_like(current_kl))

    # Degrade OCR only: the hinge activates only for slot 1 and sends a nonzero
    # gradient there; PickScore (slot 0) and still-feasible GenEval remain zero.
    current_kl = torch.tensor([100.0, 0.75, 0.50], requires_grad=True)
    loss, mask, violation = constrained_retention(current_kl, reference_kl, teacher_idx)
    assert float(loss) > 0.0
    loss.backward()
    assert current_kl.grad[0].item() == 0.0
    assert current_kl.grad[1].item() > 0.0
    assert current_kl.grad[2].item() == 0.0

    # RTC direct-reward routing is exactly slot 0 at the query step.
    rtc_jq = torch.tensor([7, 7, 7])
    assert ((teacher_idx == 0) & (rtc_jq == 7)).tolist() == [True, False, False]
    assert not bool(((teacher_idx == 0) & (rtc_jq == 6)).any())
    print("CODEX-0002 constraint/mask invariants: PASS")


if __name__ == "__main__":
    main()
