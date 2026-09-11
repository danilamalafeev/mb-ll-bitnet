from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_state_aware_carry as aware
from scripts import pc_state_aware_carry_runtime as runtime


def _example(opcodes: tuple[str, ...], x: int = 1, y: int = 3) -> dsl.RegisterExample:
    return dsl.RegisterExample(x, y, opcodes)


def _wrapped_state() -> tuple[nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer]:
    model = nn.Linear(2, 2)
    adapter, _ = latent.make_adapter()
    parent = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": parent, "param_names": ["weight", "bias"], "lr": 0.01}])
    optimizer.add_param_group({"params": list(adapter.parameters()), "param_names": list(latent.ADAPTER_PARAMETER_NAMES), "lr": 0.01})
    aware.install_state_aware_writer(adapter, optimizer)
    return model, adapter, optimizer


def test_qa_budget_and_counter_schema_are_fixed() -> None:
    counter = runtime._new_counter()
    assert counter["schema"] == runtime.ACCOUNTING_SCHEMA
    assert runtime.QA_ARMS == (runtime.ARM_A, runtime.ARM_C)
    assert runtime.QA_CALLS == 6
    assert runtime.QA_CASES == 12
    assert runtime.QA_POSITIONS == 20
    assert runtime.QA_NATIVE_STEPS == 160
    assert runtime.QA_DESERIALIZATIONS == 8
    assert "qa" in counter["phase_counts"]
    assert counter["attempted_snapshot_loads"] == 0
    assert counter["completed_snapshot_loads"] == 0


def test_batch_tensor_contract_binds_targets_and_opcode_ids() -> None:
    batch = [_example(("ADD",)), _example(("ADD",), x=2, y=4)]
    x_bits, y_bits, ops, targets_x, targets_y = runtime._batch_tensors(batch, torch.device("cpu"))
    assert x_bits.shape == y_bits.shape == (2, 4)
    assert ops.tolist() == [[dsl.OP_TO_ID["ADD"]], [dsl.OP_TO_ID["ADD"]]]
    assert targets_x.shape == targets_y.shape == (2, 1)
    with pytest.raises(ValueError, match="exactly two"):
        runtime._batch_tensors(batch[:1], torch.device("cpu"))
    with pytest.raises(ValueError, match="homogeneous"):
        runtime._batch_tensors([batch[0], _example(("ADD", "XOR"))], torch.device("cpu"))


def test_snapshot_roundtrip_restores_wrapped_adapter_optimizer_and_rng() -> None:
    model, adapter, optimizer = _wrapped_state()
    model.train(True)
    adapter.train(True)
    torch.manual_seed(991)
    loss = model(torch.ones(3, 2)).square().mean() + adapter.writer(torch.zeros(3, aware.STATE_INPUT_WIDTH)).square().mean()
    loss.backward()
    optimizer.step()
    payload = runtime._snapshot_payload(model, adapter, optimizer)
    expected = runtime._state_identity(model, adapter, optimizer)
    with torch.no_grad():
        model.weight.add_(2.0)
        adapter.writer.correction.bias.add_(2.0)
    torch.manual_seed(123)
    runtime._restore_snapshot(payload, model, adapter, optimizer, torch.device("cpu"))
    restored = runtime._state_identity(model, adapter, optimizer)
    for key in ("model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest"):
        assert restored[key] == expected[key]
    assert restored["adapter_parameter_names"] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)
    assert restored["optimizer_group_param_names"][-1] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)


def test_snapshot_digest_mutation_is_rejected() -> None:
    model, adapter, optimizer = _wrapped_state()
    payload = runtime._snapshot_payload(model, adapter, optimizer)
    broken = copy.deepcopy(payload)
    broken["adapter_digest"] = "0" * 64
    with pytest.raises(ValueError, match="state digest"):
        runtime._restore_snapshot(broken, model, adapter, optimizer, torch.device("cpu"))


def test_state_identity_exposes_wrapped_parameter_inventory() -> None:
    model, adapter, optimizer = _wrapped_state()
    identity = runtime._state_identity(model, adapter, optimizer)
    assert identity["adapter_parameter_names"] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)
    assert identity["optimizer_group_count"] == 2
    assert identity["optimizer_group_param_names"][-1] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)


def test_accounting_validator_rejects_partial_run_and_accepts_exact_counts() -> None:
    counter = runtime._new_counter()
    with pytest.raises(ValueError, match="accounting mismatch"):
        runtime._validate_accounting(counter)
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2,
        "attempted_snapshot_loads": 2, "completed_snapshot_loads": 2,
        "attempted_forwards": 6, "completed_forwards": 6,
        "attempted_cases": 12, "completed_cases": 12,
        "attempted_readout_positions": 20, "completed_readout_positions": 20,
        "attempted_native_steps": 160, "completed_native_steps": 160,
        "attempted_backwards": 6, "completed_backwards": 6,
        "attempted_optimizer_steps": 6, "completed_optimizer_steps": 6,
        "completed_updates": 6,
        "attempted_underlying_deserializations": 8,
        "completed_underlying_deserializations": 8,
    }
    counter.update(expected)
    runtime._validate_accounting(counter)
