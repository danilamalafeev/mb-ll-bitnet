"""Pure trace-level accounting for the later latent padding-drift diagnostic.

This module deliberately has no model, checkpoint, CUDA, or filesystem work.
The future diagnostic runner will feed it decoded traces and small per-position
summary values captured from the accepted final latent checkpoint.
"""

from __future__ import annotations

import math
import hashlib
import statistics
from typing import Any, Mapping, Sequence


SLOT_COUNT = 2
SLOT_WIDTH = 16
FOCUS_LENGTHS = (24, 32)
NATIVE_STEPS = 8
CONCENTRATION_FRACTION = 0.75
THRESHOLD_MULTIPLIER = 4.0
STATE_ORDER = tuple((x, y) for x in range(16) for y in range(16))
# This is the accepted E15 state split source and algorithm.  STATE_ORDER is
# still the saved lexicographic iteration order; strata use the hash ranking.
E15_STATE_SPLIT_SOURCE = "looped_bitnet/register_e15.py"
E15_STATE_SPLIT_SOURCE_SHA256 = "bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac"
E15_STATE_SPLIT_ALGORITHM = "sha256('E15-state-v1:x:y') hex ranking; 192 train, 32 validation, 32 test"


_FOCUS_SPECS = (
    ("padding_ADDADD_ADD_k10", ("ADD", "ADD"), "ADD", 24),
    ("padding_ADDADD_ADD_k14", ("ADD", "ADD"), "ADD", 32),
    ("padding_ADDADD_XOR_k10", ("ADD", "ADD"), "XOR", 24),
    ("padding_ADDADD_XOR_k14", ("ADD", "ADD"), "XOR", 32),
    ("padding_ADDADD_SWAP_k10", ("ADD", "ADD"), "SWAP", 24),
    ("padding_ADDADD_SWAP_k14", ("ADD", "ADD"), "SWAP", 32),
    ("padding_XORSWAP_ADD_k10", ("XOR", "SWAP"), "ADD", 24),
    ("padding_XORSWAP_ADD_k14", ("XOR", "SWAP"), "ADD", 32),
    ("padding_XORSWAP_XOR_k10", ("XOR", "SWAP"), "XOR", 24),
    ("padding_XORSWAP_XOR_k14", ("XOR", "SWAP"), "XOR", 32),
    ("padding_XORSWAP_SWAP_k10", ("XOR", "SWAP"), "SWAP", 24),
    ("padding_XORSWAP_SWAP_k14", ("XOR", "SWAP"), "SWAP", 32),
    ("padding_SWAPXOR_ADD_k10", ("SWAP", "XOR"), "ADD", 24),
    ("padding_SWAPXOR_ADD_k14", ("SWAP", "XOR"), "ADD", 32),
    ("padding_SWAPXOR_XOR_k10", ("SWAP", "XOR"), "XOR", 24),
    ("padding_SWAPXOR_XOR_k14", ("SWAP", "XOR"), "XOR", 32),
    ("padding_SWAPXOR_SWAP_k10", ("SWAP", "XOR"), "SWAP", 24),
    ("padding_SWAPXOR_SWAP_k14", ("SWAP", "XOR"), "SWAP", 32),
)


def _focus_entry(focus_id: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    for identifier, prefix, final_opcode, length in _FOCUS_SPECS:
        if identifier == focus_id:
            program = prefix + ("SWAP",) * (length - len(prefix) - 1) + (final_opcode,)
            roles = ("useful_op",) * len(prefix) + ("padding_extension",) * (length - len(prefix) - 1) + ("useful_op",)
            return identifier, program, roles
    raise ValueError(f"unknown frozen padding focus ID: {focus_id}")


# A tuple of tuples keeps the focus IDs, opcode sequences, and role sequences
# immutable while remaining easy to serialize into a later input manifest.
FOCUS_PROGRAM_MAP = tuple(_focus_entry(identifier) for identifier, _prefix, _final, _length in _FOCUS_SPECS)


_E15_HASHED_STATE_ORDER = tuple(
    sorted(
        STATE_ORDER,
        key=lambda state: hashlib.sha256(f"E15-state-v1:{state[0]}:{state[1]}".encode("ascii")).hexdigest(),
    )
)
_E15_STATE_STRATUM = {
    state: ("train" if index < 192 else "validation" if index < 224 else "test")
    for index, state in enumerate(_E15_HASHED_STATE_ORDER)
}


def _state_stratum(state: tuple[int, int]) -> str:
    """Return the accepted E15 hash-ranked stratum for one state."""

    if state not in _E15_STATE_STRATUM:
        raise ValueError(f"unknown E15 state: {state!r}")
    return _E15_STATE_STRATUM[state]


def validate_focus_steps(focus_id: str, steps: Sequence[Mapping[str, Any]]) -> None:
    """Require every recorded position to match the frozen program/opcode/role map."""

    _identifier, program, roles = _focus_entry(focus_id)
    if len(steps) != len(program):
        raise ValueError("diagnostic focus step count changed")
    for expected_position, (opcode, role, raw) in enumerate(zip(program, roles, steps), start=1):
        if int(raw.get("position", -1)) != expected_position or str(raw.get("opcode", "")) != opcode or str(raw.get("role", "")) != role:
            raise ValueError("diagnostic focus opcode/role alignment changed")


def validate_focus_scope(rows: Sequence[Mapping[str, Any]], *, require_all_states: bool = True) -> None:
    """Validate the exact 18 padding rows and their state/position joins."""

    expected_ids = {identifier for identifier, _prefix, _final, _length in _FOCUS_SPECS}
    if len(rows) != len(expected_ids) or {str(row.get("id", "")) for row in rows} != expected_ids:
        raise ValueError("diagnostic focus program inventory changed")
    for row in rows:
        identifier = str(row["id"])
        _identifier, program, roles = _focus_entry(identifier)
        if row.get("suite") != "padding" or int(row.get("length", -1)) != len(program) or tuple(row.get("program", ())) != program:
            raise ValueError("diagnostic focus program identity changed")
        position_map = row.get("position_map")
        expected_map = [{"position": index, "opcode": opcode, "role": role} for index, (opcode, role) in enumerate(zip(program, roles), start=1)]
        if position_map != expected_map:
            raise ValueError("diagnostic focus position map changed")
        states = row.get("states")
        if not isinstance(states, Sequence) or len(states) != (len(STATE_ORDER) if require_all_states else len(states)):
            raise ValueError("diagnostic focus state count changed")
        seen: list[tuple[int, int]] = []
        for state_row in states:
            if not isinstance(state_row, Mapping):
                raise ValueError("diagnostic focus state row is malformed")
            state = tuple(int(value) for value in state_row.get("state", ()))
            if state not in STATE_ORDER or state in seen or str(state_row.get("stratum", "")) != _state_stratum(state):
                raise ValueError("diagnostic focus state/stratum join changed")
            seen.append(state)
        if require_all_states and tuple(seen) != STATE_ORDER:
            raise ValueError("diagnostic focus state order changed")


def _as_finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _matrix(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != SLOT_COUNT:
        raise ValueError(f"{label} must have two slots")
    rows: list[list[float]] = []
    for slot, raw_row in enumerate(value):
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)) or len(raw_row) != SLOT_WIDTH:
            raise ValueError(f"{label}[{slot}] must have {SLOT_WIDTH} channels")
        rows.append([_as_finite_float(item, label=f"{label}[{slot}]") for item in raw_row])
    return rows


def _l2(row: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in row))


def _abs_delta(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [abs(float(a) - float(b)) for a, b in zip(left, right)]


def _same_matrix(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> bool:
    return all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-7) for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row))


def summarize_step_metrics(steps: Sequence[Mapping[str, Any]], *, focus_id: str | None = None) -> list[dict[str, Any]]:
    """Validate aligned per-position summaries and return compact diagnostics.

    ``z`` and ``writer`` are the two-by-sixteen tensors at one instruction.
    They are reduced immediately to norms and per-channel absolute deltas, so
    callers do not need to persist hidden states or full trajectories.
    """

    if focus_id is not None:
        validate_focus_steps(focus_id, steps)
    output: list[dict[str, Any]] = []
    previous_z: list[list[float]] | None = None
    previous_writer: list[list[float]] | None = None
    for expected_position, raw in enumerate(steps, start=1):
        if int(raw.get("position", -1)) != expected_position:
            raise ValueError("diagnostic step positions are not contiguous")
        opcode = str(raw.get("opcode", ""))
        if not opcode:
            raise ValueError("diagnostic step opcode is missing")
        z = _matrix(raw.get("z"), label=f"z at position {expected_position}")
        writer = _matrix(raw.get("writer"), label=f"writer at position {expected_position}")
        role = str(raw.get("role", ""))
        if focus_id is not None and role not in {"padding_extension", "useful_op"}:
            raise ValueError("diagnostic focus role is missing")
        hidden_finite = bool(raw.get("hidden_finite", False))
        output_finite = bool(raw.get("output_finite", False))
        hidden_norm = _as_finite_float(raw.get("hidden_norm", 0.0), label="hidden_norm")
        output_norm = _as_finite_float(raw.get("output_norm", 0.0), label="output_norm")
        row: dict[str, Any] = {
            "position": expected_position,
            "opcode": opcode,
            "role": role or None,
            "z_l2_by_slot": [_l2(slot) for slot in z],
            "writer_l2_by_slot": [_l2(slot) for slot in writer],
            "z_delta_l2_by_slot": None,
            "writer_delta_l2_by_slot": None,
            "z_delta_abs_by_slot_channel": None,
            "writer_delta_abs_by_slot_channel": None,
            "hidden_norm": hidden_norm,
            "output_norm": output_norm,
            "hidden_finite": hidden_finite,
            "output_finite": output_finite,
            "finite": bool(hidden_finite and output_finite and all(math.isfinite(value) for slot in z for value in slot) and all(math.isfinite(value) for slot in writer for value in slot)),
            "writer_matches_next_z": None,
        }
        if previous_z is not None and previous_writer is not None and not _same_matrix(previous_writer, z):
            raise ValueError("writer output must equal the next instruction z input")
        if previous_z is not None:
            output[-1]["writer_matches_next_z"] = True
        if previous_z is not None and previous_writer is not None:
            z_delta = [_abs_delta(left, right) for left, right in zip(z, previous_z)]
            writer_delta = [_abs_delta(left, right) for left, right in zip(writer, previous_writer)]
            row["z_delta_l2_by_slot"] = [_l2(delta) for delta in z_delta]
            row["writer_delta_l2_by_slot"] = [_l2(delta) for delta in writer_delta]
            row["z_delta_abs_by_slot_channel"] = z_delta
            row["writer_delta_abs_by_slot_channel"] = writer_delta
        output.append(row)
        previous_z = z
        previous_writer = writer
    if not output:
        raise ValueError("diagnostic trace has no positions")
    return output


def trace_diagnostics(
    program: Sequence[str],
    target_trace: Sequence[Sequence[int]],
    predicted_trace: Sequence[Sequence[int]],
    *,
    padding_extension_positions: Sequence[int] = (),
    focus_id: str | None = None,
) -> dict[str, Any]:
    """Join decoded output to the true DSL trace with explicit step roles.

    A role marked ``padding_extension`` is an observational segment from the
    frozen padding program definition.  It is never treated as a semantic
    no-op by this helper; the role only separates extension drift from useful
    opcode errors in the later report.
    """

    length = len(program)
    if length == 0 or len(target_trace) != length or len(predicted_trace) != length:
        raise ValueError("program and traces must have equal nonzero length")
    padding = {int(position) for position in padding_extension_positions}
    if any(position < 1 or position > length for position in padding):
        raise ValueError("padding extension position is outside the program")
    if focus_id is not None:
        _identifier, expected_program, expected_roles = _focus_entry(focus_id)
        expected_padding = {position for position, role in enumerate(expected_roles, start=1) if role == "padding_extension"}
        if tuple(program) != expected_program or padding != expected_padding:
            raise ValueError("diagnostic focus program/role map changed")
    positions: list[dict[str, Any]] = []
    first_error: int | None = None
    first_recovery: int | None = None
    for index, (opcode, target, predicted) in enumerate(zip(program, target_trace, predicted_trace), start=1):
        if len(target) != 2 or len(predicted) != 2:
            raise ValueError("DSL trace states must contain two registers")
        target_pair = [int(target[0]), int(target[1])]
        predicted_pair = [int(predicted[0]), int(predicted[1])]
        wrong = target_pair != predicted_pair
        if wrong and first_error is None:
            first_error = index
        if first_error is not None and index > first_error and not wrong and first_recovery is None:
            first_recovery = index
        positions.append({
            "position": index,
            "opcode": str(opcode),
            "role": "padding_extension" if index in padding else "useful_op",
            "target": target_pair,
            "decoded": predicted_pair,
            "wrong": wrong,
            "x_wrong": target_pair[0] != predicted_pair[0],
            "y_wrong": target_pair[1] != predicted_pair[1],
        })
    final_correct = not positions[-1]["wrong"]
    return {
        "length": length,
        "positions": positions,
        "first_error": first_error,
        "first_subsequent_recovery": first_recovery,
        "recovered_final": bool(first_error is not None and final_correct),
        "final_correct": final_correct,
        "full_trace_correct": first_error is None,
        "error_by_role": {
            role: sum(int(item["wrong"] and item["role"] == role) for item in positions)
            for role in ("padding_extension", "useful_op")
        },
    }


def cumulative_drift_summary(
    step_metrics: Sequence[Mapping[str, Any]],
    *,
    first_error: int | None = None,
    first_error_role: str | None = None,
    error_positions: Sequence[int] = (),
) -> dict[str, Any]:
    """Summarize registered threshold and L1 concentration signs.

    The threshold is the z-delta total across all 32 channels at the first
    error position versus four times the median of all valid preceding z-delta
    totals.  It is unavailable until four preceding deltas exist.  Writer
    values are intentionally excluded from concentration because writer[t] is
    the next instruction's z[t+1], not an independent same-step signal.
    """

    if not step_metrics:
        raise ValueError("drift summary requires at least one step")
    positions = [int(row["position"]) for row in step_metrics]
    if positions != list(range(1, len(step_metrics) + 1)):
        raise ValueError("drift summary positions are not contiguous")
    slot_norms = [[float(row["z_l2_by_slot"][slot]) for row in step_metrics] for slot in range(SLOT_COUNT)]
    channel_delta = [[0.0] * SLOT_WIDTH for _ in range(SLOT_COUNT)]
    delta_l1_by_position: list[float | None] = []
    role_delta_l1: dict[str, float] = {"padding_extension": 0.0, "useful_op": 0.0}
    role_slot_delta_l1: dict[str, list[float]] = {"padding_extension": [0.0, 0.0], "useful_op": [0.0, 0.0]}
    error_set = {int(position) for position in error_positions}
    if any(position < 1 or position > len(step_metrics) for position in error_set):
        raise ValueError("error position is outside the diagnostic trace")
    errored_role_delta_l1: dict[str, float] = {"padding_extension": 0.0, "useful_op": 0.0}
    correct_role_delta_l1: dict[str, float] = {"padding_extension": 0.0, "useful_op": 0.0}
    errored_role_slot_delta_l1: dict[str, list[float]] = {"padding_extension": [0.0, 0.0], "useful_op": [0.0, 0.0]}
    correct_role_slot_delta_l1: dict[str, list[float]] = {"padding_extension": [0.0, 0.0], "useful_op": [0.0, 0.0]}
    for row in step_metrics:
        raw_delta = row.get("z_delta_abs_by_slot_channel")
        if raw_delta is None:
            delta_l1_by_position.append(None)
            continue
        role = str(row.get("role", ""))
        if role not in role_delta_l1:
            raise ValueError("drift summary requires explicit frozen position roles")
        slot_values = [sum(float(value) for value in raw_delta[slot]) for slot in range(SLOT_COUNT)]
        total = sum(slot_values)
        delta_l1_by_position.append(total)
        for slot in range(SLOT_COUNT):
            role_slot_delta_l1[role][slot] += slot_values[slot]
            for channel in range(SLOT_WIDTH):
                channel_delta[slot][channel] += float(raw_delta[slot][channel])
        role_delta_l1[role] += total
        if int(row["position"]) in error_set:
            errored_role_delta_l1[role] += total
            for slot in range(SLOT_COUNT):
                errored_role_slot_delta_l1[role][slot] += slot_values[slot]
        else:
            correct_role_delta_l1[role] += total
            for slot in range(SLOT_COUNT):
                correct_role_slot_delta_l1[role][slot] += slot_values[slot]
    slot_delta_l1 = [sum(channel_delta[slot]) for slot in range(SLOT_COUNT)]
    total_delta_l1 = sum(slot_delta_l1)
    slot_fractions = [value / total_delta_l1 for value in slot_delta_l1] if total_delta_l1 else [None, None]
    dominant_slot = max(range(SLOT_COUNT), key=lambda slot: slot_delta_l1[slot]) if total_delta_l1 else None
    channel_fractions = [
        [value / slot_delta_l1[slot] for value in channel_delta[slot]] if slot_delta_l1[slot] else [None] * SLOT_WIDTH
        for slot in range(SLOT_COUNT)
    ]
    dominant_channel = [
        max(range(SLOT_WIDTH), key=lambda channel: channel_delta[slot][channel]) if slot_delta_l1[slot] else None
        for slot in range(SLOT_COUNT)
    ]
    role_error_relation = {}
    for role in role_delta_l1:
        total = role_delta_l1[role]
        role_error_relation[role] = {
            "total_l1": total,
            "errored_position_l1": errored_role_delta_l1[role],
            "correct_position_l1": correct_role_delta_l1[role],
            "errored_fraction_of_role": errored_role_delta_l1[role] / total if total else None,
            "slot_total_l1": role_slot_delta_l1[role],
            "errored_slot_l1": errored_role_slot_delta_l1[role],
            "correct_slot_l1": correct_role_slot_delta_l1[role],
        }

    if first_error is not None and (first_error < 1 or first_error > len(step_metrics)):
        raise ValueError("first error is outside the diagnostic trace")
    if first_error is not None:
        actual_role = str(step_metrics[first_error - 1].get("role", ""))
        if first_error_role is not None and first_error_role != actual_role:
            raise ValueError("first error role does not match the frozen position map")
        first_error_role = actual_role
    prior_positions = list(range(2, first_error)) if first_error is not None else []
    preceding = [delta_l1_by_position[position - 1] for position in prior_positions]
    preceding = [float(value) for value in preceding if value is not None]
    first_delta = delta_l1_by_position[first_error - 1] if first_error is not None else None
    threshold_available = bool(first_error is not None and first_delta is not None and len(preceding) >= 4)
    prior_median = statistics.median(preceding) if threshold_available else None
    if threshold_available and prior_median == 0:
        qualifies = bool(float(first_delta) > 0.0)
        denominator_case = "positive_jump_qualifies_zero_to_zero_does_not"
    elif threshold_available:
        qualifies = bool(float(first_delta) >= THRESHOLD_MULTIPLIER * float(prior_median))
        denominator_case = "positive_median"
    else:
        qualifies = None
        denominator_case = "unavailable"
    threshold = {
        "available": threshold_available,
        "delta_measure": "total_z_delta_l1_abs_all_slots_channels",
        "multiplier": THRESHOLD_MULTIPLIER,
        "first_error": first_error,
        "first_error_role": first_error_role,
        "first_error_side_delta": first_delta,
        "preceding_positions": prior_positions,
        "preceding_count": len(preceding),
        "preceding_median": prior_median,
        "qualifies": qualifies,
        "denominator_case": denominator_case,
    }
    return {
        "positions": len(step_metrics),
        "z_norm_by_slot": slot_norms,
        "cumulative_z_delta_l1_by_slot": slot_delta_l1,
        "cumulative_z_delta_l1_total": total_delta_l1,
        "slot_l1_fraction": slot_fractions,
        "slot_75pct_concentration": None if not total_delta_l1 else bool(max(slot_fractions) >= CONCENTRATION_FRACTION),
        "dominant_slot": dominant_slot,
        "cumulative_z_delta_abs_by_slot_channel": channel_delta,
        "channel_l1_fraction_within_slot": channel_fractions,
        "channel_75pct_concentration_by_slot": [
            None if not slot_delta_l1[slot] else bool(max(channel_fractions[slot]) >= CONCENTRATION_FRACTION)
            for slot in range(SLOT_COUNT)
        ],
        "dominant_channel_by_slot": dominant_channel,
        "role_error_relation": role_error_relation,
        "monotonic_non_decreasing_norm_by_slot": [all(values[index] >= values[index - 1] for index in range(1, len(values))) for values in slot_norms],
        "threshold": threshold,
        "writer_signal_independent": False,
        "finite": all(bool(row.get("finite", False)) for row in step_metrics),
    }


def aggregate_trace_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate trace outcomes without pooling away length/role identity."""

    if not records:
        raise ValueError("diagnostic aggregation requires records")

    def empty() -> dict[str, Any]:
        return {"cases": 0, "final_correct": 0, "full_trace_correct": 0, "recovered_final": 0, "first_error": {}, "first_subsequent_recovery": {}, "error_by_role": {"padding_extension": 0, "useful_op": 0}, "finite_failures": 0}

    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        trace = record.get("trace")
        if not isinstance(trace, Mapping):
            raise ValueError("diagnostic record trace is missing")
        suite = str(record.get("suite", ""))
        length = int(record.get("length", trace.get("length", -1)))
        stratum = str(record.get("stratum", ""))
        program_id = str(record.get("id", ""))
        if suite != "padding" or not stratum or not program_id or length not in FOCUS_LENGTHS:
            raise ValueError("diagnostic record identity is incomplete")
        _identifier, expected_program, _roles = _focus_entry(program_id)
        if tuple(record.get("program", ())) != expected_program or int(trace.get("length", -1)) != length:
            raise ValueError("diagnostic record program/trace length changed")
        state = tuple(int(value) for value in record.get("state", ()))
        if state not in STATE_ORDER or stratum != _state_stratum(state):
            raise ValueError("diagnostic record state/stratum join changed")
        validate_focus_steps(program_id, trace.get("positions", ()))
        keys = ["all", f"length:{length}", f"stratum:{stratum}", f"length:{length}|stratum:{stratum}", f"program:{program_id}"]
        for key in keys:
            group = groups.setdefault(key, empty())
            group["cases"] += 1
            group["final_correct"] += int(bool(trace["final_correct"]))
            group["full_trace_correct"] += int(bool(trace["full_trace_correct"]))
            group["recovered_final"] += int(bool(trace["recovered_final"]))
            first = "none" if trace["first_error"] is None else str(trace["first_error"])
            group["first_error"][first] = group["first_error"].get(first, 0) + 1
            recovery = "none" if trace["first_subsequent_recovery"] is None else str(trace["first_subsequent_recovery"])
            group["first_subsequent_recovery"][recovery] = group["first_subsequent_recovery"].get(recovery, 0) + 1
            for role, count in trace["error_by_role"].items():
                group["error_by_role"][str(role)] += int(count)
            group["finite_failures"] += int(not bool(record.get("finite", True)))
    return {"groups": groups, "denominators": {key: value["cases"] for key, value in groups.items()}}


__all__ = [
    "CONCENTRATION_FRACTION", "E15_STATE_SPLIT_ALGORITHM", "E15_STATE_SPLIT_SOURCE", "E15_STATE_SPLIT_SOURCE_SHA256", "FOCUS_LENGTHS", "FOCUS_PROGRAM_MAP", "NATIVE_STEPS", "SLOT_COUNT", "SLOT_WIDTH", "STATE_ORDER", "THRESHOLD_MULTIPLIER",
    "aggregate_trace_records", "cumulative_drift_summary", "summarize_step_metrics", "trace_diagnostics", "validate_focus_scope", "validate_focus_steps",
]
