from __future__ import annotations

from copy import deepcopy

import pytest

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots_padding_diagnostic as diagnostic


def _matrix(value: float = 0.0) -> list[list[float]]:
    return [[value] * diagnostic.SLOT_WIDTH for _ in range(diagnostic.SLOT_COUNT)]


def _step(position: int, z_value: float, writer_value: float = 0.0) -> dict[str, object]:
    z = _matrix()
    z[0][0] = z_value
    writer = _matrix()
    writer[0][0] = writer_value
    return {
        "position": position,
        "opcode": "SWAP",
        "role": "padding_extension",
        "z": z,
        "writer": writer,
        "hidden_norm": 1.0,
        "output_norm": 2.0,
        "hidden_finite": True,
        "output_finite": True,
    }


def _steps(values: list[float]) -> list[dict[str, object]]:
    return [_step(index, value, values[index] if index < len(values) else 0.0) for index, value in enumerate(values, start=1)]


def _focus_rows() -> list[dict[str, object]]:
    rows = []
    for identifier, program, roles in diagnostic.FOCUS_PROGRAM_MAP:
        rows.append({
            "id": identifier,
            "suite": "padding",
            "length": len(program),
            "program": list(program),
            "position_map": [{"position": index, "opcode": opcode, "role": role} for index, (opcode, role) in enumerate(zip(program, roles), start=1)],
            "states": [{"state": list(state), "stratum": diagnostic._state_stratum(state)} for state in diagnostic.STATE_ORDER],
        })
    return rows


def test_e15_hash_ranked_state_split_matches_accepted_source() -> None:
    accepted = dsl.state_split()
    expected = {tuple(state): stratum for stratum, states in accepted.items() for state in states}
    actual = {state: diagnostic._state_stratum(state) for state in diagnostic.STATE_ORDER}
    assert actual == expected
    assert diagnostic.E15_STATE_SPLIT_SOURCE == "looped_bitnet/register_e15.py"
    assert diagnostic.E15_STATE_SPLIT_SOURCE_SHA256 == "bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac"
    assert diagnostic._state_stratum((0, 0)) == "test"
    assert {stratum: sum(value == stratum for value in actual.values()) for stratum in ("train", "validation", "test")} == {
        "train": 192,
        "validation": 32,
        "test": 32,
    }


def test_step_alignment_and_compact_slot_writer_metrics() -> None:
    metrics = diagnostic.summarize_step_metrics(_steps([0.0, 1.0, 2.0]))
    assert len(metrics) == 3
    assert metrics[0]["z_l2_by_slot"] == [0.0, 0.0]
    assert metrics[1]["z_delta_abs_by_slot_channel"][0][0] == 1.0  # type: ignore[index]
    assert metrics[2]["writer_delta_l2_by_slot"][0] == 2.0  # type: ignore[index]
    assert metrics[0]["writer_matches_next_z"] is True
    assert metrics[2]["writer_matches_next_z"] is None
    assert all(row["finite"] for row in metrics)
    with pytest.raises(ValueError, match="positions"):
        diagnostic.summarize_step_metrics([_step(1, 0.0), _step(3, 1.0)])


def test_finite_flags_are_preserved_and_nonfinite_channels_reject() -> None:
    flagged = _step(1, 0.0)
    flagged["hidden_finite"] = False
    assert diagnostic.summarize_step_metrics([flagged])[0]["finite"] is False
    broken = _step(1, 0.0)
    broken["z"] = [[float("nan")] + [0.0] * 15, [0.0] * 16]
    with pytest.raises(ValueError, match="finite"):
        diagnostic.summarize_step_metrics([broken])


def test_true_dsl_trace_errors_keep_first_recovery_and_roles() -> None:
    program = ("ADD", "SWAP", "XOR")
    example = dsl.RegisterExample(3, 5, program)
    target = [list(pair) for pair in example.targets]
    predicted = deepcopy(target)
    predicted[0] = [(target[0][0] + 1) % 16, target[0][1]]
    predicted[1] = [(target[1][0] + 1) % 16, target[1][1]]
    trace = diagnostic.trace_diagnostics(program, target, predicted, padding_extension_positions=(2,))
    assert trace["first_error"] == 1
    assert trace["first_subsequent_recovery"] == 3
    assert trace["recovered_final"] is True
    assert trace["error_by_role"] == {"padding_extension": 1, "useful_op": 1}
    assert trace["positions"][1]["role"] == "padding_extension"  # type: ignore[index]


def test_frozen_focus_scope_and_opcode_role_alignment_reject_mutation() -> None:
    rows = _focus_rows()
    diagnostic.validate_focus_scope(rows)
    changed = deepcopy(rows)
    changed[0]["position_map"][2]["role"] = "useful_op"  # type: ignore[index]
    with pytest.raises(ValueError, match="position map"):
        diagnostic.validate_focus_scope(changed)
    changed = deepcopy(rows)
    changed[1]["program"][5] = "XOR"  # type: ignore[index]
    with pytest.raises(ValueError, match="program identity"):
        diagnostic.validate_focus_scope(changed)
    changed = deepcopy(rows)
    changed[2]["states"][0]["stratum"] = "train"  # type: ignore[index]  # (0, 0) is E15 test
    with pytest.raises(ValueError, match="state/stratum"):
        diagnostic.validate_focus_scope(changed)


def test_writer_output_is_explicitly_the_next_z_and_shifted_rows_reject() -> None:
    focus_id, program, roles = diagnostic.FOCUS_PROGRAM_MAP[0]
    padding = tuple(index for index, role in enumerate(roles, start=1) if role == "padding_extension")
    target = [[0, 0] for _ in program]
    trace = diagnostic.trace_diagnostics(program, target, target, padding_extension_positions=padding, focus_id=focus_id)
    raw_steps = []
    for index in range(len(program)):
        z = _matrix(float(index))
        writer = _matrix(float(index + 1 if index + 1 < len(program) else 0))
        raw_steps.append({"position": index + 1, "opcode": program[index], "role": roles[index], "z": z, "writer": writer, "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True})
    metrics = diagnostic.summarize_step_metrics(raw_steps, focus_id=focus_id)
    assert all(row["writer_matches_next_z"] is True for row in metrics[:-1])
    assert trace["positions"][2]["opcode"] == program[2]  # type: ignore[index]
    broken = deepcopy(raw_steps)
    broken[5]["opcode"] = "XOR"
    with pytest.raises(ValueError, match="opcode/role"):
        diagnostic.summarize_step_metrics(broken, focus_id=focus_id)
    broken = deepcopy(raw_steps)
    broken[0]["writer"] = _matrix(99.0)
    with pytest.raises(ValueError, match="next instruction z"):
        diagnostic.summarize_step_metrics(broken, focus_id=focus_id)
    with pytest.raises(ValueError, match="program/role map"):
        diagnostic.trace_diagnostics(program, target, target, padding_extension_positions=padding[:-1], focus_id=focus_id)


def _record(identifier: str, stratum: str, wrong_position: int) -> dict[str, object]:
    _, program, roles = next(item for item in diagnostic.FOCUS_PROGRAM_MAP if item[0] == identifier)
    target = [[0, 0] for _ in range(len(program))]
    predicted = deepcopy(target)
    predicted[wrong_position - 1] = [1, 0]
    padding = tuple(index for index, role in enumerate(roles, start=1) if role == "padding_extension")
    trace = diagnostic.trace_diagnostics(program, target, predicted, padding_extension_positions=padding, focus_id=identifier)
    state = list(next(state for state in diagnostic.STATE_ORDER if diagnostic._state_stratum(state) == stratum))
    return {"id": identifier, "suite": "padding", "length": len(program), "program": list(program), "state": state, "stratum": stratum, "trace": trace, "finite": True}


def test_aggregate_keeps_length_stratum_program_and_error_roles() -> None:
    aggregate = diagnostic.aggregate_trace_records([
        _record("padding_ADDADD_ADD_k10", "train", 10),
        _record("padding_ADDADD_ADD_k14", "test", 2),
    ])
    assert aggregate["groups"]["length:24"]["cases"] == 1  # type: ignore[index]
    assert aggregate["groups"]["length:32|stratum:test"]["cases"] == 1  # type: ignore[index]
    assert aggregate["groups"]["program:padding_ADDADD_ADD_k10"]["error_by_role"]["padding_extension"] == 1  # type: ignore[index]
    assert aggregate["groups"]["program:padding_ADDADD_ADD_k14"]["error_by_role"]["useful_op"] == 1  # type: ignore[index]


def test_synthetic_cumulative_drift_oracle_uses_median_first_error_delta_threshold() -> None:
    smooth = diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(_steps([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])), first_error=7, first_error_role="padding_extension")
    abrupt = diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(_steps([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0])), first_error=7, first_error_role="padding_extension")
    insufficient = diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(_steps([0.0, 1.0, 2.0, 3.0, 4.0])), first_error=5, first_error_role="padding_extension")
    zero_jump = diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(_steps([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0])), first_error=6, first_error_role="padding_extension")
    zero_flat = diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(_steps([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])), first_error=6, first_error_role="padding_extension")
    assert smooth["threshold"]["preceding_median"] == 1.0  # type: ignore[index]
    assert smooth["threshold"]["qualifies"] is False  # type: ignore[index]
    assert abrupt["threshold"]["first_error_side_delta"] == 5.0  # type: ignore[index]
    assert abrupt["threshold"]["qualifies"] is True  # type: ignore[index]
    assert insufficient["threshold"]["available"] is False  # type: ignore[index]
    assert insufficient["threshold"]["qualifies"] is None  # type: ignore[index]
    assert zero_jump["threshold"]["denominator_case"] == "positive_jump_qualifies_zero_to_zero_does_not" and zero_jump["threshold"]["qualifies"] is True  # type: ignore[index]
    assert zero_flat["threshold"]["qualifies"] is False  # type: ignore[index]
    assert smooth["threshold"]["delta_measure"] == "total_z_delta_l1_abs_all_slots_channels"  # type: ignore[index]


def test_l1_concentration_rank_75_boundary_zero_case_and_error_role_relation() -> None:
    def matrix_with(first: float, second: float = 0.0) -> list[list[float]]:
        value = _matrix()
        value[0][0], value[0][1] = first, second
        return value

    second_z = [matrix_with(3.0, 4.0)[0], matrix_with(6.0)[0]]
    first = {"position": 1, "opcode": "SWAP", "role": "useful_op", "z": _matrix(), "writer": second_z, "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True}
    second = {"position": 2, "opcode": "SWAP", "role": "padding_extension", "z": second_z, "writer": _matrix(), "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True}
    metrics = diagnostic.summarize_step_metrics([first, second])
    summary = diagnostic.cumulative_drift_summary(metrics, first_error=2, first_error_role="padding_extension", error_positions=(2,))
    assert summary["cumulative_z_delta_l1_by_slot"] == [7.0, 6.0]
    assert summary["dominant_slot"] == 0  # L1 ranks slot 0; L2 would rank slot 1.
    assert summary["role_error_relation"]["padding_extension"]["errored_position_l1"] == 13.0  # type: ignore[index]
    def concentration(slot0: float, slot1: float, *, slot0_channel1: float = 0.0) -> dict[str, object]:
        next_z = matrix_with(slot0, slot0_channel1)
        next_z[1] = matrix_with(slot1)[0]
        steps = [
            {"position": 1, "opcode": "SWAP", "role": "padding_extension", "z": _matrix(), "writer": next_z, "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True},
            {"position": 2, "opcode": "SWAP", "role": "padding_extension", "z": next_z, "writer": _matrix(), "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True},
        ]
        return diagnostic.cumulative_drift_summary(diagnostic.summarize_step_metrics(steps))

    below = concentration(2.0, 1.0)
    exact = concentration(3.0, 1.0)
    above = concentration(4.0, 1.0)
    assert below["slot_75pct_concentration"] is False
    assert exact["slot_75pct_concentration"] is True
    assert above["slot_75pct_concentration"] is True
    within_below = concentration(2.0, 0.0, slot0_channel1=1.0)
    within_exact = concentration(3.0, 0.0, slot0_channel1=1.0)
    assert within_below["channel_75pct_concentration_by_slot"] == [False, None]
    assert within_exact["channel_75pct_concentration_by_slot"] == [True, None]
    zero = diagnostic.summarize_step_metrics([{"position": 1, "opcode": "SWAP", "role": "padding_extension", "z": _matrix(), "writer": _matrix(), "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True}, {"position": 2, "opcode": "SWAP", "role": "padding_extension", "z": _matrix(), "writer": _matrix(), "hidden_norm": 1.0, "output_norm": 1.0, "hidden_finite": True, "output_finite": True}])
    zero_summary = diagnostic.cumulative_drift_summary(zero)
    assert zero_summary["dominant_slot"] is None and zero_summary["slot_l1_fraction"] == [None, None]
    assert zero_summary["slot_75pct_concentration"] is None
    assert zero_summary["channel_75pct_concentration_by_slot"] == [None, None]
