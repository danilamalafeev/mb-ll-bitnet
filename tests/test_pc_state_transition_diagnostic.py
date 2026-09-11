from __future__ import annotations

import pytest
import torch

from scripts import pc_state_transition_diagnostic as diagnostic


def _metadata(identifier: str, program: tuple[str, ...], state: tuple[int, int]) -> dict[str, object]:
    return {
        "id": identifier,
        "program": list(program),
        "final_semantic_state": list(state),
        "final_opcode": diagnostic.FINAL_OPCODE,
    }


def test_registered_pair_and_budget_are_exact() -> None:
    assert diagnostic.GOOD_PROGRAM[:2] == diagnostic.BAD_PROGRAM[:2] == diagnostic.PREFIX
    assert diagnostic.GOOD_PROGRAM[-1] == diagnostic.BAD_PROGRAM[-1] == diagnostic.FINAL_OPCODE
    assert len(diagnostic.GOOD_PROGRAM) == 12
    assert len(diagnostic.BAD_PROGRAM) == 24
    assert diagnostic.DIAGNOSTIC_BUDGET == {
        "path_forwards": 2,
        "path_cases": 4,
        "path_positions": 72,
        "path_native_steps": 576,
        "substitution_forwards": 1,
        "substitution_cases": 8,
        "substitution_positions": 8,
        "substitution_native_steps": 64,
        "optimizer_updates": 0,
    }


def test_pair_metadata_requires_same_semantic_state() -> None:
    good = _metadata(diagnostic.GOOD_ID, diagnostic.GOOD_PROGRAM, diagnostic.PAIR_STATES[0])
    bad = _metadata(diagnostic.BAD_ID, diagnostic.BAD_PROGRAM, diagnostic.PAIR_STATES[0])
    diagnostic.validate_pair_metadata(good, bad)
    broken = dict(bad, final_semantic_state=[0, 0])
    with pytest.raises(ValueError, match="semantic states"):
        diagnostic.validate_pair_metadata(good, broken)


def test_pair_distance_and_attention_summary_are_finite() -> None:
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    summary = diagnostic.pair_distance(left, right)
    assert summary["shape"] == [2, 2]
    assert summary["all_finite"] is True
    weights = torch.tensor(
        [
            [[[0.5, 0.5]], [[0.9, 0.1]]],
            [[[0.4, 0.6]], [[0.6, 0.4]]],
        ],
        dtype=torch.float32,
    )
    attention = diagnostic.attention_summary(weights)
    assert attention["shape"] == [2, 2, 1, 2]
    assert attention["near_uniform_fraction"] == pytest.approx(0.25)


def test_attention_rows_and_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        diagnostic.attention_summary(torch.tensor([[[[0.2, 0.2]]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        diagnostic.attention_summary(torch.zeros(2, 2, 2, 3))


def test_projection_summary_uses_centered_heads_and_writer_difference() -> None:
    phi_good = torch.zeros(2, diagnostic.HIDDEN_WIDTH)
    phi_bad = torch.ones_like(phi_good)
    x_head = torch.zeros(16, diagnostic.HIDDEN_WIDTH)
    y_head = torch.zeros(16, diagnostic.HIDDEN_WIDTH)
    writer_weight = torch.zeros(32, diagnostic.HIDDEN_WIDTH)
    writer_weight[0, 0] = 1.0
    writer_bias = torch.zeros(32)
    result = diagnostic.projection_summary(
        phi_good,
        phi_bad,
        x_head=x_head,
        y_head=y_head,
        writer_weight=writer_weight,
        writer_bias=writer_bias,
    )
    assert result["writer_bias_cancels_in_difference"] is True
    assert result["writer"]["delta_norm"]["max"] == pytest.approx(1.0)
    assert result["x_head"]["centered_delta_norm"]["max"] == pytest.approx(0.0)


def test_substitution_outcomes_require_exact_coverage() -> None:
    outcomes = [
        {
            "state": list(state),
            "label": label,
            "predicted": [0, 0],
            "target": [0, 0],
            "correct": True,
            "final_logits_finite": True,
        }
        for state in diagnostic.PAIR_STATES
        for label in diagnostic.SUBSTITUTION_LABELS
    ]
    assert len(diagnostic.validate_substitution_outcomes(outcomes)) == 8
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.validate_substitution_outcomes(outcomes[:-1])


def test_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.pair_distance(torch.tensor([[float("nan")]]), torch.zeros(1, 1))
