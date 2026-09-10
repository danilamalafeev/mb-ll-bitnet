"""Focused checks for the bounded E15 train/validation diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looped_bitnet.register_e15 import protocol_manifest
from scripts import e15_train_validation_diagnostic as diagnostic


def _synthetic_row(program: list[str], n: int, count: int) -> dict:
    counts = {metric: count for metric in diagnostic.METRICS}
    prefix = [count] * len(program)
    return {
        "program": program,
        "n": n,
        "counts": {**counts, "prefix_joint": prefix},
        "prefix_joint": prefix,
        "rates": {**{metric: count / n for metric in diagnostic.METRICS}, "prefix_joint": [count / n] * len(program)},
        "prefix_rates": [count / n] * len(program),
    }


def test_scope_rejects_reserved_split_and_wrong_program_before_model_call():
    manifest = protocol_manifest()

    class ExplodingModel:
        def eval(self):
            raise AssertionError("model should not be touched")

    with pytest.raises(ValueError, match="train or validation"):
        diagnostic.evaluate_split(ExplodingModel(), manifest, "test")

    wrong = json.loads(json.dumps(manifest))
    wrong["programs"]["seen"][0] = ["ADD", "XOR"]
    with pytest.raises(ValueError, match="frozen seen32"):
        diagnostic.evaluate_split(ExplodingModel(), wrong, "train")


def test_macro_rates_are_unweighted_and_counts_remain_visible():
    manifest = protocol_manifest()
    train_rows = []
    validation_rows = []
    for program in manifest["programs"]["seen"]:
        length = len(program)
        train_count = 192 if length == 1 else 96 if length == 2 else 0
        validation_count = 32 if length < 3 else 0
        train_rows.append(_synthetic_row(program, 192, train_count))
        validation_rows.append(_synthetic_row(program, 32, validation_count))

    result = diagnostic.aggregate_metrics(train_rows, validation_rows)
    length2 = result["macros"]["length"]["2"]
    length3 = result["macros"]["length"]["3"]
    compositions = result["macros"]["seen_compositions"]
    assert length2["program_count"] == 8
    assert length2["counts"]["final_joint"]["mean"] == 96
    assert length2["rates"]["final_joint"] == pytest.approx(0.5)
    assert length3["rates"]["final_joint"] == pytest.approx(0.0)
    assert compositions["rates"]["final_joint"] == pytest.approx(0.25)
    assert compositions["gaps"]["final_joint"] == pytest.approx(-25.0)


def test_validation_replay_mismatch_stops_before_train_inference(monkeypatch, tmp_path):
    manifest = protocol_manifest()
    programs = [tuple(program) for program in manifest["programs"]["seen"]]
    zero_rows = [diagnostic._normalise_row({
        "program": list(program), "joint_final": 0, "final_x_correct": 0,
        "final_y_correct": 0, "prefix_joint": [0] * len(program), "full_trace": 0,
    }, 32, program) for program in programs]
    # The frozen selected validation evidence is deliberately inconsistent with
    # the replay.  This must fail while every model has only seen validation.
    listed = [{"program": list(row["program"]), "states": 32, "joint_final": 0,
               "final_x_correct": 0, "final_y_correct": 0, "prefix_joint": list(row["prefix_joint"]),
               "full_trace": 0} for row in zero_rows]
    listed[0]["joint_final"] = 1
    record = {"selected": {"validation": {"rows": listed}}}
    calls: list[str] = []

    monkeypatch.setattr(diagnostic, "_refuse_output", lambda path: None)
    monkeypatch.setattr(diagnostic, "_verify_protected_hashes", lambda root, snapshot: {"frozen": "ok"})
    monkeypatch.setattr(diagnostic, "_load_manifest", lambda root, preflight: (manifest, {"manifest_file_sha256": "x"}))
    monkeypatch.setattr(diagnostic, "reconstruct_training_coverage", lambda frozen: {"draws": 128000})
    monkeypatch.setattr(diagnostic, "_load_model_and_checkpoint", lambda root, frozen, arm: (object(), record, "hash"))

    # Keep the real replay comparison but make its model evaluation deterministic.
    monkeypatch.setattr(diagnostic, "evaluate_split", lambda model, frozen, split: (calls.append(split) or zero_rows))
    with pytest.raises(ValueError, match="validation replay mismatch"):
        diagnostic.run_diagnostic(report_path=tmp_path / "new" / "report.json")
    assert calls == ["validation"]


def test_output_refuses_existing_artifact_or_nonempty_catalog(tmp_path):
    report = tmp_path / "diagnostic" / "report.json"
    report.parent.mkdir()
    report.write_text("old\n")
    with pytest.raises(FileExistsError):
        diagnostic._refuse_output(report)

    other = tmp_path / "other" / "report.json"
    other.parent.mkdir()
    (other.parent / "unrelated.txt").write_text("keep\n")
    with pytest.raises(FileExistsError):
        diagnostic._refuse_output(other)


def test_success_rechecks_protected_snapshot_before_report(monkeypatch, tmp_path):
    manifest = protocol_manifest()
    manifest["source_hashes"] = {}
    programs = [tuple(program) for program in manifest["programs"]["seen"]]
    zero_rows = [diagnostic._normalise_row({
        "program": list(program), "joint_final": 0, "final_x_correct": 0,
        "final_y_correct": 0, "prefix_joint": [0] * len(program), "full_trace": 0,
    }, 32, program) for program in programs]
    train_rows = [diagnostic._normalise_row({
        "program": list(program), "joint_final": 0, "final_x_correct": 0,
        "final_y_correct": 0, "prefix_joint": [0] * len(program), "full_trace": 0,
    }, 192, program) for program in programs]
    listed = [{"program": list(row["program"]), "states": 32, "joint_final": 0,
               "final_x_correct": 0, "final_y_correct": 0, "prefix_joint": list(row["prefix_joint"]),
               "full_trace": 0} for row in zero_rows]
    record = {"selected": {"validation": {"rows": listed}}}
    protected_checks = []
    monkeypatch.setattr(diagnostic, "_verify_protected_hashes", lambda root, snapshot: (protected_checks.append(1) or {"frozen": "ok"}))
    monkeypatch.setattr(diagnostic, "_load_manifest", lambda root, preflight: (manifest, {"manifest_file_sha256": "x"}))
    monkeypatch.setattr(diagnostic, "reconstruct_training_coverage", lambda frozen: {"draws": 128000})
    monkeypatch.setattr(diagnostic, "_load_model_and_checkpoint", lambda root, frozen, arm: (object(), record, "hash"))
    monkeypatch.setattr(diagnostic, "evaluate_split", lambda model, frozen, split: train_rows if split == "train" else zero_rows)
    report = diagnostic.run_diagnostic(report_path=tmp_path / "new" / "report.json")
    assert protected_checks == [1, 1]
    assert report["provenance"]["evaluator_sha256"] == diagnostic.sha256_file(diagnostic.PROJECT_ROOT / diagnostic.DIAGNOSTIC_SCRIPT)
