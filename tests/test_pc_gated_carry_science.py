"""Focused, model-free tests for the gated-carry science boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_gated_carry_science as science


def test_registered_science_budget_and_stream_prefix() -> None:
    assert gated.TRAINING_BATCH_LENGTHS[:6] == (1, 2, 3, 4, 5, 6)
    assert len(gated.TRAINING_BATCH_LENGTHS) == science.TRAINING_UPDATES
    assert sum(gated.TRAINING_BATCH_LENGTHS) == 3496
    assert science.TRAINING_POSITIONS_PER_ARM == 223744
    assert science.TRAINING_NATIVE_STEPS_PER_ARM == 1789952
    assert science.SCIENCE_CALLS == 2531
    assert science.SCIENCE_CASES == 263936
    assert science.SCIENCE_POSITIONS == 3239936
    assert science.SCIENCE_NATIVE_STEPS == 25919488
    assert science.SCIENCE_DESERIALIZATIONS == 6


def test_new_control_scope_is_exact_identity_coverage() -> None:
    specs = science._new_specs()
    assert len(specs) == 81
    assert {spec["length"] for spec in specs} == {11, 19, 35}
    assert {length: sum(spec["length"] == length for spec in specs) for length in (11, 19, 35)} == {11: 27, 19: 27, 35: 27}
    assert all(len(spec["states"]) == 256 and len(spec["strata"]) == 256 for spec in specs)
    assert all(dsl.execute_program(tuple(spec["program"][2:-1]), state) == state for spec in specs for state in dsl.STATE_ORDER)


def test_program_to_ops_binding_rejects_wrong_opcode_or_shape() -> None:
    program = ("ADD", "SWAP", "XOR")
    correct = torch.tensor([[dsl.OP_TO_ID[op] for op in program]] * 2, dtype=torch.long)
    assert science._validate_program_ops(program, correct) == program
    wrong = correct.clone()
    wrong[1, 1] = dsl.OP_TO_ID["ADD"]
    with pytest.raises(ValueError, match="program-to-ops binding"):
        science._validate_program_ops(program, wrong)
    with pytest.raises(ValueError, match="shape"):
        science._validate_program_ops(program, correct[:, :2])


def test_checkpoint_metadata_has_absolute_and_resume_boundaries() -> None:
    record = science._checkpoint_metadata(750, arm=science.ARM_B, source_binding_digest="src", input_freeze_digest="freeze", parent_identity={"update": 42000})
    assert record["schema"] == science.CHECKPOINT_SCHEMA
    assert record["committed"] is True and record["complete"] is True
    assert record["local_update"] == 750
    assert record["absolute_update"] == 42750
    assert record["next_batch_index"] == 750
    with pytest.raises(ValueError, match="boundary"):
        science._checkpoint_metadata(251, arm=science.ARM_A, source_binding_digest="src", input_freeze_digest="freeze", parent_identity={})
    with pytest.raises(ValueError, match="arm"):
        science._checkpoint_metadata(0, arm="sham", source_binding_digest="src", input_freeze_digest="freeze", parent_identity={})


def test_reused_initial_a_row_is_explicitly_non_replayed() -> None:
    spec = {"id": "p", "suite": "padding", "length": 1, "program": ("ADD",), "states": tuple(dsl.STATE_ORDER), "strata": tuple("train" for _ in dsl.STATE_ORDER)}
    baseline = {"predictions": {"p": [{"state": [0, 0], "stratum": "train", "target_trace": [[0, 0]], "predicted_trace": [[0, 0]]}]}}
    row = science._reused_row(spec, baseline)
    assert row["arm"] == science.ARM_A
    assert row["phase"] == "initial_reused_saved"
    assert row["provenance"] == {"source": "accepted_final_latent", "replayed": False}


def test_pair_oracle_distinguishes_repairs_regressions_and_ties() -> None:
    def row(arm: str, full: bool) -> dict:
        return {"id": "p", "arm": arm, "predictions": [{"state": [0, 0], "full_trace_correct": full}]}
    assert science._paired([row("B", True)], [row("A", False)], label="repair")["repair"] == 1
    assert science._paired([row("B", False)], [row("A", True)], label="regression")["regression"] == 1
    tie = science._paired([row("B", True)], [row("A", True)], label="tie")
    assert tie["both_correct"] == 1 and tie["repair"] == 0 and tie["regression"] == 0


def test_accounting_rejects_budget_mismatch_without_model_work() -> None:
    counter = science._new_counter()
    with pytest.raises(ValueError, match="accounting mismatch"):
        science._validate_accounting(counter)
    assert counter["attempted_forwards"] == 0
    assert counter["attempted_underlying_deserializations"] == 0


def test_valid_accounting_fixture_covers_training_evaluation_and_checkpoints() -> None:
    counter = science._new_counter()
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2,
        "attempted_parent_loads": 2, "completed_parent_loads": 2,
        "attempted_endpoint_restores": 2, "completed_endpoint_restores": 2,
        "attempted_underlying_deserializations": 6, "completed_underlying_deserializations": 6,
        "attempted_forwards": 2531, "completed_forwards": 2531,
        "attempted_cases": 263936, "completed_cases": 263936,
        "attempted_readout_positions": 3239936, "completed_readout_positions": 3239936,
        "attempted_native_steps": 25919488, "completed_native_steps": 25919488,
        "attempted_updates": 2000, "completed_updates": 2000,
        "attempted_backwards": 2000, "completed_backwards": 2000,
        "attempted_optimizer_steps": 2000, "completed_optimizer_steps": 2000,
        "attempted_checkpoints": 10, "completed_checkpoints": 10,
    }
    counter.update(expected)
    counter["phase_counts"]["training"]["completed_updates"] = 2000
    counter["phase_counts"]["evaluation_initial"]["completed_forwards"] = 231
    counter["phase_counts"]["evaluation_final"]["completed_forwards"] = 300
    science._validate_accounting(counter)


def test_mixed_program_batch_ops_are_validated_per_example() -> None:
    programs = [("ADD", "SWAP"), ("XOR", "SWAP")]
    ops = torch.tensor([[dsl.OP_TO_ID[op] for op in program] for program in programs], dtype=torch.long)
    assert science._validate_batch_program_ops(programs, ops) == tuple(programs)
    with pytest.raises(ValueError, match="binding"):
        science._validate_batch_program_ops(programs, ops.flip(0))


def test_qa_step_overrides_are_restored_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = [dsl.RegisterExample(index % 16, (index // 16) % 16, ("ADD",)) for index in range(64)]
    original_batch = science.qa_runtime._batch_tensors
    original_validator = science.qa_runtime._validate_program_ops

    def fail(**_kwargs: object) -> float:
        raise RuntimeError("synthetic update failure")

    monkeypatch.setattr(science.qa_runtime, "_run_update", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        science._run_update(model=None, adapter=None, optimizer=None, batch=batch, arm=science.ARM_A, device=torch.device("cpu"), counter=science._new_counter(), sink=None)  # type: ignore[arg-type]
    assert science.qa_runtime._batch_tensors is original_batch
    assert science.qa_runtime._validate_program_ops is original_validator


def test_checkpoint_write_updates_real_accounting_and_is_atomic(tmp_path: Path) -> None:
    counter = science._new_counter()
    payload = {"arm": science.ARM_A, "local_update": 0, "absolute_update": 42000, "next_batch_index": 0, "checkpoint_digest": "digest"}
    result = science._write_checkpoint(arm_out=tmp_path / "A", payload=payload, counter=counter, sink=None, root=tmp_path)
    assert result["committed"] is True and result["complete"] is True
    assert (tmp_path / "A" / "checkpoints" / "local0000.pt").is_file()
    assert counter["attempted_checkpoints"] == 1 and counter["completed_checkpoints"] == 1


def test_partial_row_is_durable_before_later_work(tmp_path: Path) -> None:
    row_index = [0]
    row = {"id": "p", "arm": science.ARM_B, "phase": "initial", "predictions": []}
    science._persist_row(tmp_path, row, row_index)
    assert row_index == [1]
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["id"] == "p"


def test_gate_summary_keeps_per_slot_ranges() -> None:
    values = [torch.tensor([[[0.2], [0.8]], [[0.4], [0.6]]], dtype=torch.float32)]
    summary = science._gate_summary({"gated_carry": {"reader_calls": 1, "writer_calls": 1, "gate_values": values}}, 1)
    assert summary is not None
    assert summary[0]["per_slot"] == [
        {"slot": 0, "mean": pytest.approx(0.3), "min": pytest.approx(0.2), "max": pytest.approx(0.4)},
        {"slot": 1, "mean": pytest.approx(0.7), "min": pytest.approx(0.6), "max": pytest.approx(0.8)},
    ]


def test_expected_program_row_count_includes_reused_old_a_rows() -> None:
    assert science._expected_program_row_count() == 600


def test_qa_provenance_gate_rejects_mutated_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record_path = tmp_path / "qa_accept.json"
    record_path.write_text(json.dumps({"status": "QA_ACCEPT", "qa_origin_absolute_update": science.ENDPOINT_ABSOLUTE_UPDATE, "files": {}, "source_files": {}}), encoding="utf-8")
    monkeypatch.setattr(science, "QA_ACCEPT", record_path)
    monkeypatch.setattr(science, "QA_ACCEPT_SHA256", science._sha256(record_path))
    with pytest.raises(ValueError, match="files are missing"):
        science._validate_qa_accept(tmp_path)


def test_finite_trace_metrics_and_first_recovery_are_saved_shape() -> None:
    target = [[1, 2], [3, 4], [5, 6]]
    predicted = [[0, 2], [3, 4], [5, 6]]
    metrics = science._trace_metrics(target, predicted)
    assert metrics["full_trace_correct"] is False
    assert metrics["joint_final_correct"] is True
    assert metrics["first_subsequent_recovery"] == 2


@pytest.mark.parametrize("arm", ["A", "B"])
def test_real_checkpoint_payload_metadata_and_gate_state(arm: str) -> None:
    model = torch.nn.Linear(2, 2)
    adapter = science.latent.LatentSlotAdapter()
    optimizer = torch.optim.AdamW([
        {"params": list(model.parameters()), "param_names": [name for name, _ in model.named_parameters()]},
        {"params": list(adapter.parameters()), "param_names": list(science.latent.ADAPTER_PARAMETER_NAMES)},
    ], lr=0.001)
    if arm == "B":
        gated.attach_carry_gate(adapter)
        gated.append_carry_gate_optimizer_group(optimizer, adapter)
    payload = science._checkpoint_payload(model=model, adapter=adapter, optimizer=optimizer,
        arm=arm, local_update=250, manifest_sha256="manifest", source_binding_digest="source",
        input_freeze_digest="freeze", parent_identity={"update": 42000},
        endpoint_sha256="endpoint", adapter_initialization={})
    assert payload["optimizer_metadata"] == science.qa_runtime.accepted_qa_runtime._optimizer_metadata(optimizer)
    assert payload["absolute_update"] == 42250 and payload["next_batch_index"] == 250
    assert ("carry_gate.weight" in payload["adapter_state_dict"]) == (arm == "B")
    digest = payload.pop("checkpoint_digest")
    assert digest == science._digest(payload)


def test_preparation_failure_keeps_original_error_and_saved_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise ValueError("synthetic input mismatch")
    monkeypatch.setattr(science.diagnostic_runtime, "_validate_evidence", fail)
    with pytest.raises(ValueError, match="synthetic input mismatch"):
        science.run_science(root=tmp_path, manifest_path=tmp_path / "manifest.json", out=tmp_path / "science")
    report = json.loads((tmp_path / "science/report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["message"] == "synthetic input mismatch"
    assert report["accounting"]["attempted_forwards"] == 0
    assert report["accounting"]["failures"][-1]["phase"] == "load"


def test_evaluation_persists_first_row_before_second_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    monkeypatch.setattr(science.science, "_preserve_evaluation_state", lambda *a, **k: nullcontext())
    def run_program(**kwargs: object) -> dict:
        if kwargs["spec"]["id"] == "second":
            raise RuntimeError("synthetic second evaluation failure")
        return {"id": "first", "arm": "B", "phase": "initial", "predictions": []}
    monkeypatch.setattr(science, "_run_program", run_program)
    index = [0]
    with pytest.raises(RuntimeError, match="synthetic second"):
        science._evaluate_phase(model=None, adapter=None, optimizer=None,
            specs=[{"id": "first"}, {"id": "second"}], arm="B", phase="initial",
            device=torch.device("cpu"), counter=science._new_counter(), sink=None,
            row_sink=lambda row: science._persist_row(tmp_path, row, index))
    assert index == [1]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["id"] == "first"
