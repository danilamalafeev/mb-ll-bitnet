from __future__ import annotations

import copy

import pytest

from scripts.pc_explicit_registers import (
    OPS,
    QA_BUDGET_MAX,
    SCIENCE_BUDGET,
    budget,
    canonical_digest,
    compose_transition_tables,
    dsl_step,
    dsl_trace,
    paired_metrics,
    trace_diagnostic,
    validate_frozen_identity,
    validate_transition_tables,
)


SMALL_STATES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _tables(*, wrong_add: bool = False) -> dict[str, list[dict[str, list[int]]]]:
    tables = {}
    for opcode in OPS:
        rows = []
        for x, y in SMALL_STATES:
            if opcode == "ADD":
                output = [x + 1 if x < 1 else 0, y]
            elif opcode == "XOR":
                output = [x ^ 1, y ^ 1]
            else:
                output = [y, x]
            rows.append({"state": [x, y], "output": output})
        tables[opcode] = rows
    if wrong_add:
        tables["ADD"][0]["output"] = [0, 1]
        # This row makes teacher-forced and own-predicted composition diverge.
        tables["SWAP"][1]["output"] = [0, 0]
    return tables


def test_composition_feeds_own_predicted_registers_without_teacher_forcing() -> None:
    tables = _tables(wrong_add=True)
    rows = compose_transition_tables(tables, ["ADD", "SWAP"], SMALL_STATES)
    row = next(item for item in rows if item["state"] == [0, 0])
    assert row["predicted_trace"] == [[0, 1], [0, 0]]
    # Oracle/teacher-forced ADD would have supplied [1, 0] to SWAP instead.
    assert row["predicted_trace"][-1] != [1, 0]


def test_register_order_and_noncommuting_opcode_order_are_preserved() -> None:
    rows = compose_transition_tables(_tables(), ["ADD", "SWAP"], SMALL_STATES)
    reverse = compose_transition_tables(_tables(), ["SWAP", "ADD"], SMALL_STATES)
    left = next(item for item in rows if item["state"] == [0, 1])
    right = next(item for item in reverse if item["state"] == [0, 1])
    assert left["predicted_trace"] == [[1, 1], [1, 1]]
    assert right["predicted_trace"] == [[1, 0], [0, 0]]


def test_real_dsl_uses_modulo16_xor_swap_and_noncommuting_order() -> None:
    assert dsl_step("ADD", (15, 3)) == (2, 3)
    assert dsl_step("XOR", (12, 10)) == (6, 10)
    assert dsl_step("SWAP", (3, 12)) == (12, 3)
    assert dsl_trace(["ADD", "SWAP"], (3, 5)) == [[8, 5], [5, 8]]
    assert dsl_trace(["SWAP", "ADD"], (3, 5)) == [[5, 3], [8, 3]]


def test_trace_diagnostic_reports_first_error_and_later_recovery() -> None:
    diagnostic = trace_diagnostic(
        [[9, 9], [0, 1], [7, 6]],
        [[0, 0], [0, 1], [7, 6]],
    )
    assert diagnostic == {
        "positions": 3,
        "position_correct": [False, True, True],
        "first_error_position": 1,
        "first_recovery_position": 2,
        "final_correct": True,
        "full_trace_correct": False,
    }


@pytest.mark.parametrize(
    "mutator,pattern",
    [
        (lambda t: t["XOR"].pop(), "state coverage changed"),
        (lambda t: t["ADD"].append(copy.deepcopy(t["ADD"][0])), "duplicate state"),
        (lambda t: t["SWAP"].__setitem__(0, {"state": [0, 0], "output": [16, 0]}), "0..15"),
    ],
)
def test_malformed_missing_duplicate_and_out_of_range_tables_fail(mutator, pattern: str) -> None:
    tables = _tables()
    mutator(tables)
    with pytest.raises(ValueError, match=pattern):
        validate_transition_tables(tables, SMALL_STATES)


def test_unknown_opcode_and_unknown_program_operation_fail() -> None:
    tables = _tables()
    with pytest.raises(ValueError, match="exactly ADD"):
        validate_transition_tables({**tables, "MUL": tables["ADD"]}, SMALL_STATES)
    with pytest.raises(ValueError, match="unknown opcode"):
        compose_transition_tables(tables, ["ADD", "MUL"], SMALL_STATES)


def _prediction_rows() -> tuple[list[dict], list[dict]]:
    baseline, composed = [], []
    targets = {
        (0, 0): [[1, 0], [0, 1]],
        (0, 1): [[1, 1], [1, 1]],
        (1, 0): [[0, 0], [0, 0]],
        (1, 1): [[0, 1], [1, 0]],
    }
    baseline_predictions = {
        (0, 0): [[1, 0], [0, 1]],       # both correct
        (0, 1): [[1, 1], [1, 1]],       # final regression only
        (1, 0): [[1, 1], [0, 1]],       # final benefit only
        (1, 1): [[1, 1], [1, 0]],       # earlier error recovers at final position
    }
    composed_predictions = {
        (0, 0): [[1, 0], [0, 1]],
        (0, 1): [[1, 1], [0, 0]],
        (1, 0): [[0, 0], [0, 0]],
        (1, 1): [[0, 1], [1, 0]],
    }
    for state in SMALL_STATES:
        baseline.append({"state": list(state), "target_trace": targets[state],
                         "predicted_trace": baseline_predictions[state],
                         "joint_final_correct": "ignored"})
        composed.append({"state": list(state), "predicted_trace": composed_predictions[state]})
    return baseline, composed


def test_paired_metrics_recompute_benefit_regression_and_ties_from_traces() -> None:
    baseline, composed = _prediction_rows()
    metrics = paired_metrics(baseline, composed, ["ADD", "SWAP"], SMALL_STATES)
    assert metrics["final"] == {
        "both_correct": 2,
        "baseline_correct_composed_wrong": 1,
        "baseline_wrong_composed_correct": 1,
        "both_wrong": 0,
    }
    assert metrics["full_trace"] == {
        "both_correct": 1,
        "baseline_correct_composed_wrong": 1,
        "baseline_wrong_composed_correct": 2,
        "both_wrong": 0,
    }
    assert metrics["final_benefit"] == 1
    assert metrics["final_regression"] == 1
    assert metrics["final_ties"] == 2
    assert metrics["full_trace_benefit"] == 2


def test_paired_metrics_reject_duplicate_state_and_changed_target() -> None:
    baseline, composed = _prediction_rows()
    duplicate = copy.deepcopy(baseline)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate state"):
        paired_metrics(duplicate, composed, ["ADD", "SWAP"], SMALL_STATES)
    changed = copy.deepcopy(composed)
    changed[0]["target_trace"] = [[9, 9], [9, 9]]
    with pytest.raises(ValueError, match="target trace differs"):
        paired_metrics(baseline, changed, ["ADD", "SWAP"], SMALL_STATES)


def test_prediction_mapping_rejects_nested_state_that_disagrees_with_key() -> None:
    baseline, composed = _prediction_rows()
    keyed = {tuple(row["state"]): row for row in baseline}
    keyed[(0, 0)] = {**keyed[(0, 0)], "state": [0, 1]}
    with pytest.raises(ValueError, match="key and nested state disagree"):
        paired_metrics(keyed, composed, ["ADD", "SWAP"], SMALL_STATES)


def test_frozen_identity_and_digest_mutations_are_rejected() -> None:
    frozen = {
        "endpoint": "float128_seed0/B",
        "checkpoint": "checkpoint.pt",
        "program": ["ADD", "XOR", "SWAP"],
        "payload_sha256": "abc123",
    }
    digest = canonical_digest(frozen)
    assert validate_frozen_identity(frozen, frozen, expected_digest=digest) == digest

    changed = copy.deepcopy(frozen)
    changed["program"][0] = "SWAP"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_frozen_identity(changed, frozen, expected_digest=digest)
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_frozen_identity(frozen, frozen, expected_digest="0" * 64)


def test_registered_science_and_qa_budgets_are_exact() -> None:
    assert SCIENCE_BUDGET == {
        "updates": 0,
        "attempted_forwards": 18,
        "completed_forwards": 18,
        "attempted_cases": 4608,
        "completed_cases": 4608,
        "attempted_readout_positions": 4608,
        "completed_readout_positions": 4608,
        "attempted_native_steps": 36864,
        "completed_native_steps": 36864,
    }
    assert QA_BUDGET_MAX["attempted_forwards"] == 2
    assert QA_BUDGET_MAX["attempted_cases"] == 4
    assert QA_BUDGET_MAX["attempted_native_steps"] == 32
    partial = budget(2, 1, cases_per_forward=2)
    assert partial["attempted_cases"] == 4
    assert partial["completed_cases"] == 2
    assert partial["attempted_native_steps"] == 32
    assert partial["completed_native_steps"] == 16
    with pytest.raises(ValueError, match="cannot exceed"):
        budget(1, 2)
