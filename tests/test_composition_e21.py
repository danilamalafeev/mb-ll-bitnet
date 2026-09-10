"""Focused E21 symbolic, scope, counting, and legal seen-program QA controls."""
from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch
from torch import nn

from looped_bitnet import longer_native8_e20 as e20
from scripts import composition_e21 as e


class SeenToy(nn.Module):
    """Forward-only legal-seen toy; never receives an E21 primary/control program."""

    def forward(self, x, y, op_ids):
        return torch.zeros((x.shape[0], op_ids.shape[1], 16)), torch.zeros((x.shape[0], op_ids.shape[1], 16))


def _toy_prediction(state, length, *, final_joint, final_x, final_y, prefix):
    target = [[0, 0] for _ in range(length)]
    predicted = [[0, 0] for _ in range(length)]
    if not final_x:
        predicted[-1][0] = 1
    if not final_y:
        predicted[-1][1] = 1
    return {"state": list(state), "target_trace": target, "predicted_trace": predicted,
            "prefix_joint_correct": [bool(value) for value in prefix], "joint_final_correct": bool(final_joint)}


def _toy_rows(first_correct=244, second_correct=243):
    states = e._ordered_states(); strata = e._strata(); membership = {state: name for name, values in strata.items() for state in values}
    rows = []
    for index, program in enumerate(e.ALL_PROGRAMS):
        predictions = []
        for position, state in enumerate(states):
            if index == 0:
                # x=250, y=246, joint=244; two prefixes are 252 and 244,
                # with a 240 full trace through non-nested prefix sets.
                final_x = position < 250
                final_y = position < 244 or position in (250, 251)
                final_joint = position < 244
                prefix = (position >= 4, position < 244)
            elif index == 1:
                final_x = final_y = final_joint = position < second_correct
                prefix = (final_joint,) * len(program)
            else:
                final_x = final_y = final_joint = True
                prefix = (True,) * len(program)
            predictions.append(_toy_prediction(state, len(program), final_joint=final_joint,
                                               final_x=final_x, final_y=final_y, prefix=prefix))
        for prediction in predictions:
            prediction["stratum"] = membership[tuple(prediction["state"])]
        rows.append({"program": list(program), "role": "control" if program == e.CONTROL_PROGRAM else "primary", "predictions": predictions})
    return rows, strata


def test_symbolic_audit_freezes_order_novelty_and_control():
    symbolic = e.symbolic_audit()
    assert symbolic["states"] == [[x, y] for x in range(16) for y in range(16)]
    assert [item["program"] for item in symbolic["programs"]] == [list(program) for program in e.ALL_PROGRAMS]
    assert symbolic["reference"]["primary_programs"] == 6
    assert symbolic["reference"]["control"]["equivalent_to"] == ["ADD"]
    assert symbolic["reference"]["evaluation_cost"]["native8_substeps"] == 40960


def test_asymmetric_243_244_metrics_and_control_exclusion():
    rows, strata = _toy_rows()
    scored = e.score_rows(rows, strata)
    assert scored[e.PRIMARY_PROGRAMS[0]]["metrics"]["all"]["final_joint"] == 244
    assert scored[e.PRIMARY_PROGRAMS[0]]["metrics"]["all"]["final_x"] == 250
    assert scored[e.PRIMARY_PROGRAMS[0]]["metrics"]["all"]["final_y"] == 246
    assert scored[e.PRIMARY_PROGRAMS[0]]["metrics"]["all"]["prefix_joint"] == [252, 244]
    assert scored[e.PRIMARY_PROGRAMS[0]]["metrics"]["all"]["full_trace"] == 240
    assert scored[e.PRIMARY_PROGRAMS[1]]["metrics"]["all"]["final_joint"] == 243
    predicates = {" ".join(program): scored[program]["metrics"]["all"]["final_joint"] >= e.PRIMARY_THRESHOLD
                  for program in e.PRIMARY_PROGRAMS}
    assert predicates["ADD XOR"] and not predicates["ADD ADD XOR"]
    assert all(predicates.values()) is False
    assert scored[e.CONTROL_PROGRAM]["role"] == "control"
    assert scored[e.CONTROL_PROGRAM]["metrics"]["all"]["final_joint"] == 256
    report = e._report({"scored": scored, "primary_predicates": predicates,
                        "primary_conjunction": all(predicates.values()), "control": scored[e.CONTROL_PROGRAM],
                        "forward_passes": 7, "rows": rows})
    assert report["primary_conjunction"] is False
    assert "ADD XOR XOR" not in report["primary_predicates"]
    assert report["control_row"]["metrics"]["all"]["final_joint"] == 256


def test_preflight_freezes_manifest_without_e21_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "evaluate_model", lambda *args, **kwargs: pytest.fail("preflight predicted a model"))
    path = e.preflight(tmp_path / "preflight")
    manifest = e.load_manifest(path.parent)
    assert manifest["schema"] == e.E21_SCHEMA
    assert manifest["cost"] == e.EXPECTED_COST
    assert manifest["scope"]["states"] == 256
    with pytest.raises(FileExistsError):
        e.preflight(path.parent)


def test_manifest_reference_tamper_is_rejected_before_evaluation(tmp_path):
    path = e.preflight(tmp_path / "preflight")
    manifest = e._read_json(path)
    manifest["e20_model_digest"] = "tampered"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen manifest"):
        e.load_manifest(path.parent)


def test_actual_report_path_uses_seven_legal_seen_forwards_only(tmp_path):
    preflight = e.preflight(tmp_path / "preflight")
    manifest = e.load_manifest(preflight.parent)
    programs = (("ADD",), ("XOR",), ("SWAP",), ("XOR", "XOR"), ("SWAP", "SWAP"),
                ("XOR", "SWAP"), ("SWAP", "XOR"))
    assert all(program not in e.ALL_PROGRAMS and not e.forbidden(program) for program in programs)
    report = e.qa_seen_report(SeenToy(), out=tmp_path / "qa", manifest=manifest, programs=programs)
    assert report["status"] == "qa_complete"
    assert report["forward_passes"] == 7
    assert report["cost"] == {"training_updates": 0, "program_state_cases": 1792,
                               "readout_positions": 2816, "internal_state_substeps": 22528}
    assert len(report["rows"]) == 7
    assert all(tuple(row["program"]) not in e.ALL_PROGRAMS for row in report["rows"])


def test_metric_invariants_reject_bad_denominators():
    rows, strata = _toy_rows()
    rows[0]["predictions"].pop()
    with pytest.raises(ValueError, match="256 predictions"):
        e.score_rows(rows, strata)


def test_real_native8_seen_tiny_has_one_boundary_readout_and_eight_substeps():
    model, _, _ = e20.build_initial_model()
    calls = {"blocks": 0, "opcode": 0, "reader": 0, "output_norm": 0}
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(lambda *args: calls.__setitem__("blocks", calls["blocks"] + 1)))
    hooks.append(model.opcode_embedding.register_forward_hook(lambda *args: calls.__setitem__("opcode", calls["opcode"] + 1)))
    hooks.append(model.reader.register_forward_hook(lambda *args: calls.__setitem__("reader", calls["reader"] + 1)))
    hooks.append(model.output_norm.register_forward_hook(lambda *args: calls.__setitem__("output_norm", calls["output_norm"] + 1)))
    try:
        from looped_bitnet.register_e15 import evaluate_program
        evaluate_program(model, ("ADD",), [(0, 0), (1, 2)], include_predictions=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == {"blocks": 8, "opcode": 8, "reader": 8, "output_norm": 1}


def test_protected_snapshot_and_production_scope_constants():
    hashes = e.verify_protected_hashes()
    assert len(hashes) == 195
    assert e.EXPECTED_COST["training_updates"] == 0
    assert e.EXPECTED_COST["program_state_cases"] == 1792
    assert e.PRIMARY_THRESHOLD == 244
    assert e.CONTROL_PROGRAM not in e.PRIMARY_PROGRAMS
