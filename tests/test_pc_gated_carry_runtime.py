from __future__ import annotations

import copy

import pytest
import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_gated_carry_runtime as runtime
from scripts import pc_latent_slots as latent


def _valid_arm_identity(arm: str) -> dict[str, object]:
    names = list(latent.ADAPTER_PARAMETER_NAMES)
    groups = [list(latent.ADAPTER_PARAMETER_NAMES)]
    if arm == runtime.ARM_A:
        groups = [["parent.weight"], list(latent.ADAPTER_PARAMETER_NAMES)]
        architecture = latent.ARCHITECTURE_ID
    else:
        names += list(gated.CARRY_GATE_PARAMETER_NAMES)
        groups = [["parent.weight"], list(latent.ADAPTER_PARAMETER_NAMES), list(gated.CARRY_GATE_PARAMETER_NAMES)]
        architecture = gated.GATED_ARCHITECTURE_ID
    return {
        "arm": arm,
        "label": runtime.QA_ARM_LABELS[arm],
        "architecture_id": architecture,
        "model_parameter_names": ["parent.weight"],
        "adapter": {
            "architecture_id": architecture,
            "parameter_names": names,
            "parameter_shapes": [],
            "parameter_count": len(names),
            "state_digest": "digest",
        },
        "optimizer_group_count": len(groups),
        "optimizer_group_param_names": groups,
    }


def _valid_snapshot(*, arm: str = runtime.ARM_B, include_gate: bool = True) -> tuple[dict[str, object], dict[str, object]]:
    identity = _valid_arm_identity(arm)
    names = identity["adapter"]["parameter_names"]  # type: ignore[index]
    adapter_state: dict[str, object] = {name: torch.zeros(1) for name in names}
    if arm == runtime.ARM_B and not include_gate:
        adapter_state.pop("carry_gate.weight", None)
        adapter_state.pop("carry_gate.bias", None)
    empty_tensor = torch.zeros(1, dtype=torch.uint8)
    payload: dict[str, object] = {
        "schema": runtime.SNAPSHOT_SCHEMA,
        "committed": True,
        "complete": True,
        "arm": arm,
        "architecture_id": identity["architecture_id"],
        "manifest_sha256": "manifest",
        "endpoint_sha256": "endpoint",
        "source_binding_digest": "source",
        "fixture_digest": "fixture",
        "parent_identity": {"parent": "accepted"},
        "local_update": 1,
        "absolute_update": 42001,
        "next_batch_index": 1,
        "arm_identity": identity,
        "adapter_identity": {
            "architecture_id": identity["architecture_id"],
            "parameter_names": names,
            "parameter_shapes": [],
            "parameter_count": len(names),
            "state_digest": "snapshot",
        },
        "adapter_initialization": {},
        "model_state_dict": {},
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": {},
        "optimizer_metadata": [],
        "cpu_rng_state": empty_tensor,
        "cuda_rng_state": [],
        "training_mode": True,
        "adapter_training_mode": True,
        "model_parameter_names": ["parent.weight"],
        "adapter_parameter_names": names,
        "optimizer_group_param_names": identity["optimizer_group_param_names"],
        "model_digest": runtime.science._digest({}),
        "adapter_digest": runtime.science._digest(adapter_state),
        "optimizer_digest": runtime.science._digest({}),
        "cpu_rng_digest": runtime.science._digest(empty_tensor),
        "cuda_rng_digest": runtime.science._digest([]),
    }
    payload["snapshot_digest"] = runtime.science._digest(runtime._snapshot_core(payload))
    return payload, identity


def test_fixed_qa_budget_and_actual_attempt_completion_ledger() -> None:
    assert runtime.QA_CALLS == 6
    assert runtime.QA_CASES == 12
    assert runtime.QA_POSITIONS == 20
    assert runtime.QA_NATIVE_STEPS == 160
    assert runtime.QA_UPDATES == 6
    assert runtime.QA_DESERIALIZATIONS == 8
    counter = runtime._new_counter()
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2,
        "attempted_parent_loads": 2, "completed_parent_loads": 2,
        "attempted_endpoint_restores": 2, "completed_endpoint_restores": 2,
        "attempted_snapshot_loads": 2, "completed_snapshot_loads": 2,
        "attempted_underlying_deserializations": 8, "completed_underlying_deserializations": 8,
        "attempted_updates": 6, "completed_updates": 6,
        "attempted_forwards": 6, "completed_forwards": 6,
        "attempted_cases": 12, "completed_cases": 12,
        "attempted_readout_positions": 20, "completed_readout_positions": 20,
        "attempted_native_steps": 160, "completed_native_steps": 160,
        "attempted_backwards": 6, "completed_backwards": 6,
        "attempted_optimizer_steps": 6, "completed_optimizer_steps": 6,
    }
    counter.update(expected)
    runtime._validate_accounting(counter)
    counter["completed_updates"] -= 1
    with pytest.raises(ValueError, match="accounting mismatch"):
        runtime._validate_accounting(counter)


def test_program_to_ops_binding_uses_each_saved_example_and_rejects_mutation() -> None:
    programs = (("XOR",), ("XOR",))
    ops = torch.tensor([[dsl.OP_TO_ID["XOR"]], [dsl.OP_TO_ID["XOR"]]], dtype=torch.long)
    assert runtime._validate_program_ops(programs, ops) == programs
    mutated = ops.clone()
    mutated[1, 0] = dsl.OP_TO_ID["SWAP"]
    with pytest.raises(ValueError, match="program-to-ops"):
        runtime._validate_program_ops(programs, mutated)
    with pytest.raises(ValueError, match="homogeneous"):
        runtime._validate_program_ops((("XOR",), ("XOR", "SWAP")), torch.zeros(2, 1, dtype=torch.long))


def test_arm_identity_requires_the_gate_inventory_and_named_group() -> None:
    runtime._validate_arm_identity(_valid_arm_identity(runtime.ARM_A), arm=runtime.ARM_A)
    runtime._validate_arm_identity(_valid_arm_identity(runtime.ARM_B), arm=runtime.ARM_B)
    wrong_gate = copy.deepcopy(_valid_arm_identity(runtime.ARM_B))
    wrong_gate["adapter"]["parameter_names"] = list(latent.ADAPTER_PARAMETER_NAMES)  # type: ignore[index]
    with pytest.raises(ValueError, match="gate inventory"):
        runtime._validate_arm_identity(wrong_gate, arm=runtime.ARM_B)
    wrong_group = copy.deepcopy(_valid_arm_identity(runtime.ARM_B))
    wrong_group["optimizer_group_param_names"][-1] = ["carry_gate.bias", "carry_gate.weight"]  # type: ignore[index]
    with pytest.raises(ValueError, match="optimizer name"):
        runtime._validate_arm_identity(wrong_group, arm=runtime.ARM_B)


def test_snapshot_missing_gate_is_rejected_before_restore() -> None:
    payload, identity = _valid_snapshot(include_gate=False)
    with pytest.raises(ValueError, match="missing carry_gate"):
        runtime._validate_snapshot_payload(
            payload,
            arm=runtime.ARM_B,
            parent_identity={"parent": "accepted"},
            source_binding_digest="source",
            fixture_digest="fixture",
            manifest_sha256="manifest",
            endpoint_sha256="endpoint",
            expected_arm_identity=identity,
        )


def test_snapshot_provenance_mismatch_is_rejected_before_state_restore() -> None:
    payload, identity = _valid_snapshot()
    with pytest.raises(ValueError, match="source_binding_digest"):
        runtime._validate_snapshot_payload(
            payload,
            arm=runtime.ARM_B,
            parent_identity={"parent": "accepted"},
            source_binding_digest="different-source",
            fixture_digest="fixture",
            manifest_sha256="manifest",
            endpoint_sha256="endpoint",
            expected_arm_identity=identity,
        )


def test_fixture_records_reject_target_or_digest_mutation_without_model_work() -> None:
    examples = [
        dsl.RegisterExample(0, 1, ("XOR",)),
        dsl.RegisterExample(2, 3, ("XOR",)),
    ]
    raw = [[runtime._example_record(example) for example in examples], [runtime._example_record(example) for example in [dsl.RegisterExample(4, 5, ("XOR", "SWAP")), dsl.RegisterExample(6, 7, ("XOR", "SWAP"))]]]
    fixture: dict[str, object] = {
        "batches": raw,
    }
    fixture["fixture_digest"] = runtime.science._digest({"batches": raw})
    assert len(runtime._fixture_from_records(fixture)) == 2
    changed = copy.deepcopy(fixture)
    changed["batches"][0][0]["target_trace"][0] = [15, 15]  # type: ignore[index]
    with pytest.raises(ValueError, match="target trace"):
        runtime._fixture_from_records(changed)



def test_snapshot_absolute_update_uses_parent_absolute_count():
    payload, identity = _valid_snapshot()
    kwargs = dict(arm=runtime.ARM_B, parent_identity={"parent": "accepted"},
                  source_binding_digest="source", fixture_digest="fixture",
                  manifest_sha256="manifest", endpoint_sha256="endpoint",
                  expected_arm_identity=identity)
    assert payload["absolute_update"] == 42001
    runtime._validate_snapshot_payload(payload, **kwargs)
    payload["absolute_update"] = 2001
    payload["snapshot_digest"] = runtime.science._digest(runtime._snapshot_core(payload))
    with pytest.raises(ValueError, match="absolute_update"):
        runtime._validate_snapshot_payload(payload, **kwargs)
