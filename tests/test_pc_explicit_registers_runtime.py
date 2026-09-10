from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts import pc_explicit_registers_runtime as runtime


class _DeterministicOneOp(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor, op_ids: torch.Tensor):
        assert tuple(op_ids.shape) == (x.shape[0], 1)
        out_x = (x + y) % 16
        out_y = y
        logits_x = torch.full((x.shape[0], 1, 16), -100.0)
        logits_y = torch.full((x.shape[0], 1, 16), -100.0)
        logits_x[torch.arange(x.shape[0]), 0, out_x] = 100.0
        logits_y[torch.arange(x.shape[0]), 0, out_y] = 100.0
        return logits_x, logits_y


def test_one_op_forward_decodes_fresh_batch_and_accounts_native_steps() -> None:
    counter = runtime._new_counter()
    output = runtime._one_op_forward(
        _DeterministicOneOp(), ((0, 0), (15, 1)), "ADD", torch.device("cpu"), counter
    )
    assert output == [[0, 0], [0, 1]]
    assert counter["attempted_forwards"] == 1
    assert counter["completed_forwards"] == 1
    assert counter["attempted_cases"] == 2
    assert counter["completed_cases"] == 2
    assert counter["attempted_native_steps"] == 16
    assert counter["completed_native_steps"] == 16


def test_qa_binding_is_exactly_first_float_seed0_b_and_two_states() -> None:
    endpoint = next(item for item in runtime.migration.ENDPOINTS if item[0] == runtime.QA_LABEL)
    binding = runtime._program_binding("qa", (endpoint,), runtime.QA_STATES, (runtime.QA_PROGRAM,))
    assert binding["mode"] == "qa"
    assert binding["state_order"] == [[0, 0], [15, 1]]
    assert binding["primitive_ops"] == ["ADD", "XOR", "SWAP"]
    assert binding["programs"] == [["ADD", "XOR"]]
    assert binding["endpoint_rows"][0]["label"] == "float128_seed0/B"


def test_saved_program_pool_is_frozen_to_45_padding_plus_24_compositions() -> None:
    rows, digest = runtime._load_saved_programs()
    assert len(rows) == 69
    assert len(digest) == 64
    assert len({str(row["id"]) for row in rows}) == 69


def test_carried_wrong_local_correct_and_local_wrong_recovery_are_distinct() -> None:
    program = ("SWAP", "XOR")
    target = [[0, 1], [1, 1]]
    locally_correct_but_global_wrong = runtime._carried_local_observations(
        program, (1, 0), [[2, 1], [3, 1]], target
    )
    locally_wrong_but_global_recovered = runtime._carried_local_observations(
        program, (1, 0), [[2, 1], [1, 1]], target
    )
    assert locally_correct_but_global_wrong[1] == {
        "position": 2,
        "input_matches_true_prestate": False,
        "local_step_correct": True,
        "output_matches_true_poststate": False,
    }
    assert locally_wrong_but_global_recovered[1] == {
        "position": 2,
        "input_matches_true_prestate": False,
        "local_step_correct": False,
        "output_matches_true_poststate": True,
    }


def test_frozen_file_hash_rejects_source_baseline_or_checkpoint_mutation(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("stable\n", encoding="utf-8")
    expected = runtime.migration.sha256_file(source)
    assert runtime._require_file_hash(source, expected, "source") == expected
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        runtime._require_file_hash(source, expected, "source")
    for label in ("baseline endpoint", "checkpoint"):
        with pytest.raises(ValueError, match=f"{label} hash mismatch"):
            runtime._require_file_hash(source, expected, label)


def test_accepted_qa_binding_mutation_is_rejected_before_science_loader() -> None:
    binding = runtime._qa_bindings()
    changed = copy.deepcopy(binding)
    changed["accept"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="QA acceptance hash mismatch"):
        runtime._validate_qa_bindings(changed)


def test_accepted_qa_artifact_mutation_is_rejected_against_qa_acceptance(monkeypatch, tmp_path) -> None:
    changed = tmp_path / "accounting.json"
    changed.write_bytes((runtime.QA_ARTIFACTS["accounting"]).read_bytes() + b"changed")
    monkeypatch.setitem(runtime.QA_ARTIFACTS, "accounting", changed)
    with pytest.raises(ValueError, match="accepted QA artifact hash mismatch: accounting.json"):
        runtime._qa_bindings()


def test_reviewed_model_inventory_uses_accepted_absolute_paths_directly(tmp_path) -> None:
    reviewed_file = tmp_path / "__main__.py"
    reviewed_file.write_text("accepted\n", encoding="utf-8")
    review = tmp_path / "qa_source_review.json"
    review.write_text(json.dumps({
        "additional_model_source_inventory": {
            str(reviewed_file): runtime.migration.sha256_file(reviewed_file),
        },
    }), encoding="utf-8")
    runtime._validate_reviewed_model_sources(review)
    reviewed_file.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted model source hash mismatch"):
        runtime._validate_reviewed_model_sources(review)


def test_frozen_manifest_bytes_remain_unchanged_through_validation_failure(tmp_path) -> None:
    path = tmp_path / "science_manifest.json"
    value = {"schema": "pc_explicit_registers_science_manifest_v1", "status": "frozen", "n": 1}
    runtime._write_json(path, value)
    before = runtime.migration.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == value
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime._require_file_hash(path, "0" * 64, "frozen manifest")
    assert runtime.migration.sha256_file(path) == before


def test_partial_forward_counters_survive_failing_stub() -> None:
    class Failing(torch.nn.Module):
        def forward(self, *_args):
            raise RuntimeError("stub failure")

    counter = runtime._new_counter()
    with pytest.raises(RuntimeError, match="stub failure"):
        runtime._one_op_forward(Failing(), ((0, 0), (15, 1)), "ADD", torch.device("cpu"), counter)
    snapshot = json.loads(json.dumps(counter))
    assert snapshot["attempted_forwards"] == 1
    assert snapshot["completed_forwards"] == 0
    assert snapshot["attempted_cases"] == 2
    assert snapshot["completed_cases"] == 0
    assert snapshot["attempted_native_steps"] == 16
    assert snapshot["completed_native_steps"] == 0
    assert snapshot["failures"][0]["kind"] == "forward"


def test_forward_attempt_is_flushed_before_model_invocation() -> None:
    snapshots = []

    class Observing(torch.nn.Module):
        def forward(self, *_args):
            snapshots.append("model_called")
            raise RuntimeError("stub failure")

    counter = runtime._new_counter()

    def sink(value):
        snapshots.append(json.loads(json.dumps(value)))

    with pytest.raises(RuntimeError, match="stub failure"):
        runtime._one_op_forward(Observing(), ((0, 0),), "ADD", torch.device("cpu"), counter, sink)
    assert snapshots[0]["attempted_forwards"] == 1
    assert snapshots[0]["completed_forwards"] == 0
    assert snapshots[0]["attempted_cases"] == 1
    assert snapshots[0]["completed_cases"] == 0
    assert snapshots[1] == "model_called"
    assert snapshots[2]["failures"][0]["kind"] == "forward"
