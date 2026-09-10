"""Pure contract helpers for the explicit-register PC follow-up.

This module intentionally contains no model imports, checkpoint loading, or
forward calls.  A later runner will supply one-step prediction tables and use
these helpers to compose the model's own discrete outputs.  In particular, a
predicted register pair is the lookup key for the next opcode; DSL targets are
never fed back into a composition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


OPS = ("ADD", "XOR", "SWAP")
ENDPOINTS = (
    "float128_seed0/B",
    "w4128_seed0/B",
    "float128_seed1/B",
    "w4128_seed1/B",
    "float128_seed2/B",
    "w4128_seed2/B",
)
STATE_COUNT = 256
STATES = tuple((x, y) for x in range(16) for y in range(16))
NATIVE_STEPS_PER_POSITION = 8
ONE_OP_LENGTH = 1
SCIENCE_FORWARDS = len(ENDPOINTS) * len(OPS)
SCIENCE_CASES = SCIENCE_FORWARDS * STATE_COUNT
SCIENCE_NATIVE_STEPS = SCIENCE_CASES * ONE_OP_LENGTH * NATIVE_STEPS_PER_POSITION
QA_MAX_FORWARDS = 2
QA_STATES_PER_FORWARD = 2
QA_CASES = QA_MAX_FORWARDS * QA_STATES_PER_FORWARD
QA_NATIVE_STEPS = QA_CASES * ONE_OP_LENGTH * NATIVE_STEPS_PER_POSITION

PAIR_CATEGORIES = (
    "both_correct",
    "baseline_correct_composed_wrong",
    "baseline_wrong_composed_correct",
    "both_wrong",
)

State = tuple[int, int]
Pair = tuple[int, int]


def dsl_step(opcode: str, state: Sequence[int]) -> State:
    """Apply one real register-DSL operation using modulo-16 registers."""

    if opcode not in OPS:
        raise ValueError(f"unknown opcode: {opcode!r}")
    x, y = _state(state, label="DSL state")
    if opcode == "ADD":
        return ((x + y) % 16, y)
    if opcode == "XOR":
        return (x ^ y, y)
    return (y, x)


def dsl_trace(program: Sequence[str], state: Sequence[int]) -> list[list[int]]:
    """Return the independent real-DSL trace for one initial register pair."""

    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence) or not program:
        raise ValueError("program must be a non-empty opcode sequence")
    if any(opcode not in OPS for opcode in program):
        raise ValueError("program contains an unknown opcode")
    current = _state(state, label="DSL state")
    trace: list[list[int]] = []
    for opcode in program:
        current = dsl_step(opcode, current)
        trace.append([current[0], current[1]])
    return trace


def _pair(value: Any, *, label: str) -> Pair:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be a length-2 sequence")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} must contain plain integers")
    result = (int(value[0]), int(value[1]))
    if any(item < 0 or item > 15 for item in result):
        raise ValueError(f"{label} register values must be in 0..15")
    return result


def _state(value: Any, *, label: str) -> State:
    return _pair(value, label=label)


def _states(state_order: Sequence[Sequence[int]]) -> tuple[State, ...]:
    result = tuple(_state(value, label="state") for value in state_order)
    if len(set(result)) != len(result):
        raise ValueError("state order contains duplicates")
    return result


def trace_diagnostic(predicted_trace: Sequence[Sequence[int]], target_trace: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Report first error, first later recovery, final, and full-trace status."""

    if isinstance(predicted_trace, (str, bytes)) or not isinstance(predicted_trace, Sequence):
        raise ValueError("predicted trace must be a sequence")
    if isinstance(target_trace, (str, bytes)) or not isinstance(target_trace, Sequence):
        raise ValueError("target trace must be a sequence")
    if not predicted_trace or len(predicted_trace) != len(target_trace):
        raise ValueError("predicted and target traces must be non-empty and equal length")
    predicted = [_pair(value, label=f"predicted trace step {index}") for index, value in enumerate(predicted_trace)]
    target = [_pair(value, label=f"target trace step {index}") for index, value in enumerate(target_trace)]
    position_correct = [left == right for left, right in zip(predicted, target)]
    first_error_index = next((index for index, correct in enumerate(position_correct) if not correct), None)
    first_recovery_index = None
    if first_error_index is not None:
        first_recovery_index = next(
            (index for index in range(first_error_index + 1, len(position_correct)) if position_correct[index]),
            None,
        )
    return {
        "positions": len(position_correct),
        "position_correct": position_correct,
        "first_error_position": None if first_error_index is None else first_error_index + 1,
        "first_recovery_position": None if first_recovery_index is None else first_recovery_index + 1,
        "final_correct": position_correct[-1],
        "full_trace_correct": all(position_correct),
    }


def canonical_digest(value: Any) -> str:
    """Hash a deterministic JSON representation for frozen run identities."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value cannot be represented by the canonical digest") from exc
    return hashlib.sha256(encoded).hexdigest()


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        for key, value in left.items():
            if key not in right or not _typed_equal(value, right[key]):
                return False
        return True
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_frozen_identity(
    actual: Any,
    expected: Any,
    *,
    expected_digest: str | None = None,
    label: str = "frozen identity",
) -> str:
    """Reject any changed frozen value and, when supplied, its expected digest."""

    if not _typed_equal(actual, expected):
        raise ValueError(f"{label} identity mismatch")
    digest = canonical_digest(actual)
    if expected_digest is not None and (type(expected_digest) is not str or digest != expected_digest):
        raise ValueError(f"{label} digest mismatch")
    return digest


def _table_rows(table: Any, *, opcode: str) -> list[tuple[Any, Any]]:
    if isinstance(table, Mapping):
        return list(table.items())
    if isinstance(table, (str, bytes)) or not isinstance(table, Sequence):
        raise ValueError(f"{opcode} transition table must be a mapping or row sequence")
    rows: list[tuple[Any, Any]] = []
    for index, row in enumerate(table):
        if not isinstance(row, Mapping) or "state" not in row or "output" not in row:
            raise ValueError(f"{opcode} transition row {index} requires state and output")
        rows.append((row["state"], row["output"]))
    return rows


def validate_transition_tables(
    tables: Mapping[str, Any],
    state_order: Sequence[Sequence[int]] = STATES,
) -> dict[str, dict[State, Pair]]:
    """Normalize and validate three complete one-op prediction tables.

    Tables may be mappings keyed by ``(x, y)`` or sequences of
    ``{"state": [x, y], "output": [x, y]}`` rows.  A duplicate normalized
    state, missing state, unknown opcode, or malformed register is rejected.
    """

    if not isinstance(tables, Mapping) or set(tables) != set(OPS):
        raise ValueError("transition tables must contain exactly ADD, XOR, and SWAP")
    expected = _states(state_order)
    expected_set = set(expected)
    normalized: dict[str, dict[State, Pair]] = {}
    for opcode in OPS:
        rows = _table_rows(tables[opcode], opcode=opcode)
        current: dict[State, Pair] = {}
        for index, (raw_state, raw_output) in enumerate(rows):
            state = _state(raw_state, label=f"{opcode} row {index} state")
            if state in current:
                raise ValueError(f"{opcode} transition table contains duplicate state {state}")
            current[state] = _pair(raw_output, label=f"{opcode} row {index} output")
        if set(current) != expected_set:
            missing = sorted(expected_set - set(current))
            extra = sorted(set(current) - expected_set)
            raise ValueError(f"{opcode} transition table state coverage changed: missing={missing[:3]} extra={extra[:3]}")
        normalized[opcode] = current
    return normalized


def compose_transition_tables(
    tables: Mapping[str, Any],
    program: Sequence[str],
    state_order: Sequence[Sequence[int]] = STATES,
) -> list[dict[str, Any]]:
    """Compose one-step predictions using the model's own prior output.

    The returned rows contain a decoded trace for every initial state.  This
    is a discrete table composition fixture; it does not claim the wall time of
    a true sequential model forward, and it never substitutes an oracle target
    for the predicted intermediate state.
    """

    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence) or not program:
        raise ValueError("program must be a non-empty opcode sequence")
    if any(op not in OPS for op in program):
        raise ValueError("program contains an unknown opcode")
    states = _states(state_order)
    normalized = validate_transition_tables(tables, states)
    result = []
    for initial in states:
        current = initial
        trace: list[list[int]] = []
        for opcode in program:
            # `current` is the prior predicted pair.  This is the core
            # no-teacher-forcing invariant of the explicit-register contract.
            current = normalized[opcode][current]
            trace.append([current[0], current[1]])
        result.append({"state": [initial[0], initial[1]], "predicted_trace": trace})
    return result


def _prediction_rows(rows: Any, *, label: str, program_length: int) -> dict[State, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        normalized_rows = []
        for state, value in rows.items():
            key_state = _state(state, label=f"{label} mapping key")
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} mapping value for {key_state} must be a row mapping")
            row = dict(value)
            if "state" in row and _state(row["state"], label=f"{label} mapping state") != key_state:
                raise ValueError(f"{label} mapping key and nested state disagree for {key_state}")
            row["state"] = [key_state[0], key_state[1]]
            normalized_rows.append(row)
        rows = normalized_rows
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError(f"{label} predictions must be a row sequence or mapping")
    result: dict[State, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "state" not in row or "predicted_trace" not in row:
            raise ValueError(f"{label} row {index} requires state and predicted_trace")
        state = _state(row["state"], label=f"{label} row {index} state")
        if state in result:
            raise ValueError(f"{label} predictions contain duplicate state {state}")
        predicted = row["predicted_trace"]
        if isinstance(predicted, (str, bytes)) or not isinstance(predicted, Sequence) or len(predicted) != program_length:
            raise ValueError(f"{label} row {index} trace length changed")
        for step, value in enumerate(predicted):
            _pair(value, label=f"{label} row {index} step {step}")
        result[state] = row
    return result


def _category(left_correct: bool, right_correct: bool) -> str:
    if left_correct and right_correct:
        return "both_correct"
    if left_correct:
        return "baseline_correct_composed_wrong"
    if right_correct:
        return "baseline_wrong_composed_correct"
    return "both_wrong"


def _empty_pair_counts() -> dict[str, int]:
    return {key: 0 for key in PAIR_CATEGORIES}


def paired_metrics(
    baseline_rows: Any,
    composed_rows: Any,
    program: Sequence[str],
    state_order: Sequence[Sequence[int]] = STATES,
) -> dict[str, Any]:
    """Recompute paired final/full-trace outcomes from decoded rows.

    Correctness flags in either input are ignored.  Baseline targets are the
    saved target traces and are checked for shape; the composed rows supply
    only model-owned predictions.  The two independent category tables expose
    benefits, regressions, and ties without collapsing them into one score.
    """

    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence) or not program:
        raise ValueError("program must be a non-empty opcode sequence")
    if any(op not in OPS for op in program):
        raise ValueError("program contains an unknown opcode")
    states = _states(state_order)
    baseline = _prediction_rows(baseline_rows, label="baseline", program_length=len(program))
    composed = _prediction_rows(composed_rows, label="composed", program_length=len(program))
    expected = set(states)
    if set(baseline) != expected or set(composed) != expected:
        raise ValueError("baseline and composed predictions must cover the exact state order")

    final = _empty_pair_counts()
    full_trace = _empty_pair_counts()
    for state in states:
        left = baseline[state]
        right = composed[state]
        target = left.get("target_trace")
        if isinstance(target, (str, bytes)) or not isinstance(target, Sequence) or len(target) != len(program):
            raise ValueError(f"baseline target trace is malformed for {state}")
        target_pairs = [_pair(value, label=f"baseline target {state} step {index}") for index, value in enumerate(target)]
        left_pairs = [_pair(value, label=f"baseline prediction {state} step {index}") for index, value in enumerate(left["predicted_trace"])]
        right_pairs = [_pair(value, label=f"composed prediction {state} step {index}") for index, value in enumerate(right["predicted_trace"])]
        if "target_trace" in right and right["target_trace"] != left["target_trace"]:
            raise ValueError(f"baseline/composed target trace differs for {state}")
        left_diagnostic = trace_diagnostic(left_pairs, target_pairs)
        right_diagnostic = trace_diagnostic(right_pairs, target_pairs)
        left_final = left_diagnostic["final_correct"]
        right_final = right_diagnostic["final_correct"]
        left_full = left_diagnostic["full_trace_correct"]
        right_full = right_diagnostic["full_trace_correct"]
        final[_category(left_final, right_final)] += 1
        full_trace[_category(left_full, right_full)] += 1
    return {
        "schema": "pc_explicit_registers_paired_metrics_v1",
        "program": list(program),
        "states": len(states),
        "final": final,
        "full_trace": full_trace,
        "final_benefit": final["baseline_wrong_composed_correct"],
        "final_regression": final["baseline_correct_composed_wrong"],
        "final_ties": final["both_correct"] + final["both_wrong"],
        "full_trace_benefit": full_trace["baseline_wrong_composed_correct"],
        "full_trace_regression": full_trace["baseline_correct_composed_wrong"],
        "full_trace_ties": full_trace["both_correct"] + full_trace["both_wrong"],
    }


def budget(
    attempted_forwards: int,
    completed_forwards: int,
    *,
    cases_per_forward: int = STATE_COUNT,
    program_length: int = ONE_OP_LENGTH,
    native_steps_per_position: int = NATIVE_STEPS_PER_POSITION,
    updates: int = 0,
) -> dict[str, int]:
    """Return exact attempted/completed accounting for a fixed forward scope."""

    values = (attempted_forwards, completed_forwards, cases_per_forward, program_length,
              native_steps_per_position, updates)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("budget values must be non-negative plain integers")
    if completed_forwards > attempted_forwards:
        raise ValueError("completed forwards cannot exceed attempted forwards")
    attempted_cases = attempted_forwards * cases_per_forward
    completed_cases = completed_forwards * cases_per_forward
    attempted_positions = attempted_cases * program_length
    completed_positions = completed_cases * program_length
    return {
        "updates": updates,
        "attempted_forwards": attempted_forwards,
        "completed_forwards": completed_forwards,
        "attempted_cases": attempted_cases,
        "completed_cases": completed_cases,
        "attempted_readout_positions": attempted_positions,
        "completed_readout_positions": completed_positions,
        "attempted_native_steps": attempted_positions * native_steps_per_position,
        "completed_native_steps": completed_positions * native_steps_per_position,
    }


SCIENCE_BUDGET = budget(SCIENCE_FORWARDS, SCIENCE_FORWARDS)
QA_BUDGET_MAX = budget(QA_MAX_FORWARDS, QA_MAX_FORWARDS, cases_per_forward=QA_STATES_PER_FORWARD)


__all__ = [
    "ENDPOINTS", "OPS", "PAIR_CATEGORIES", "QA_BUDGET_MAX", "SCIENCE_BUDGET", "STATES",
    "budget", "canonical_digest", "compose_transition_tables", "dsl_step", "dsl_trace",
    "paired_metrics", "trace_diagnostic", "validate_frozen_identity", "validate_transition_tables",
]
