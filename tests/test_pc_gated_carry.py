from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_latent_slots as latent
from scripts import pc_learned_scratchpad as accepted
from scripts.pc_learned_scratchpad_runtime import (
    _paired_metrics as accepted_paired_metrics,
    _trace_metrics as accepted_trace_metrics,
)


class _ToyReader:
    def build_kv(self, key_memory: Tensor, value_memory: Tensor) -> tuple[Tensor, Tensor]:
        return key_memory, value_memory


class _ToyCore(nn.Module):
    native_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.role_keys = nn.Parameter(torch.linspace(-1.0, 1.0, 2 * 128).reshape(2, 128))
        self.memory_norm = nn.LayerNorm(128, eps=1e-5)
        with torch.no_grad():
            self.memory_norm.weight.copy_(torch.linspace(0.7, 1.3, 128))
            self.memory_norm.bias.copy_(torch.linspace(-0.2, 0.2, 128))
        self.reader = _ToyReader()
        self.output_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 16, bias=False)

    def step(self, cache: dict[str, object], opcode: Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, object]]:
        values = cache["kv"][1]  # type: ignore[index]
        h = cache["h"] + values.mean(dim=1) * 0.125  # type: ignore[operator]
        h = h + opcode.to(dtype=h.dtype).unsqueeze(1) * 0.01
        logits = self.head(h)
        return (logits, logits + 0.1), {"h": h + 0.2, "kv": cache["kv"], "substeps": cache["substeps"] + 8}  # type: ignore[operator]


def _bits(values: list[int]) -> Tensor:
    matrix = accepted.signed_bit_matrix()
    return matrix[torch.tensor(values, dtype=torch.long)]


def _inputs() -> tuple[Tensor, Tensor]:
    return _bits([1, 6]), _bits([3, 12])


def _gated_fixture(length: int = 3) -> tuple[_ToyCore, latent.LatentSlotAdapter, Tensor, Tensor, Tensor]:
    torch.manual_seed(1234)
    model = _ToyCore()
    adapter, _ = latent.make_adapter()
    with torch.no_grad():
        adapter.writer.weight.zero_()
        adapter.writer.bias.copy_(torch.arange(32, dtype=torch.float32) / 10.0)
    gate, _ = gated.attach_carry_gate(adapter)
    x_bits, y_bits = _inputs()
    ops = torch.tensor([[0, 1, 2][:length], [0, 1, 2][:length]], dtype=torch.long)
    return model, adapter, x_bits, y_bits, ops


def test_gate_count_initialization_and_global_rng_preservation() -> None:
    torch.manual_seed(2718)
    before = torch.get_rng_state().clone()
    gate, metadata = gated.make_carry_gate()
    assert torch.equal(torch.get_rng_state(), before)
    assert gate.parameter_count() == gated.GATE_PARAMETER_COUNT == 322
    assert tuple(parameter.shape for parameter in gate.parameters()) == ((2, 160), (2,))
    assert torch.count_nonzero(gate.weight) == 0
    assert torch.allclose(gate.bias, torch.full_like(gate.bias, gated.GATE_INITIAL_LOGIT))
    assert metadata["initial_gate"] == gated.GATE_INITIAL_VALUE
    output = gate(torch.zeros(4, 2, 16), torch.zeros(4, 128))
    assert output.shape == (4, 2, 1)
    assert torch.allclose(output, torch.full_like(output, 0.9))


def test_manual_slot_broadcast_blend_and_input_validation() -> None:
    z = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16)
    proposal = z + 10.0
    gates = torch.tensor([[[0.25], [0.75]], [[1.0], [0.5]]], dtype=torch.float32)
    expected = z + gates * 10.0
    assert torch.equal(gated.blend_carry_state(z, proposal, gates), expected)
    with pytest.raises(ValueError, match="shape"):
        gated.blend_carry_state(z, proposal, gates[:, :, 0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gated.blend_carry_state(z, proposal, torch.full_like(gates, 1.01))
    with pytest.raises(ValueError, match="nonfinite"):
        gated.blend_carry_state(z, proposal.clone().fill_(float("nan")), gates)


def test_gate_has_only_current_z_and_hidden_inputs() -> None:
    parameters = inspect.signature(gated.CarryGate.forward).parameters
    assert tuple(parameters) == ("self", "z", "normalized_hidden")
    forward_parameters = inspect.signature(gated.gated_latent_slots_forward).parameters
    assert "target" not in forward_parameters
    assert "opcode" not in parameters and "position" not in parameters and "pair" not in parameters


def test_current_logits_are_unchanged_but_future_state_uses_carried_value() -> None:
    model, adapter, x_bits, y_bits, ops = _gated_fixture()
    ordinary_model = copy.deepcopy(model)
    ordinary_adapter = copy.deepcopy(adapter)
    del ordinary_adapter.carry_gate
    ordinary, ordinary_diagnostics = latent.latent_slots_forward(
        ordinary_model, ordinary_adapter, x_bits, y_bits, ops, return_diagnostics=True
    )
    carried, diagnostics = gated.gated_latent_slots_forward(
        model, adapter, x_bits, y_bits, ops, return_diagnostics=True
    )
    assert torch.equal(carried[0][:, 0], ordinary[0][:, 0])
    assert torch.equal(carried[1][:, 0], ordinary[1][:, 0])
    assert not torch.allclose(carried[0][:, 1:], ordinary[0][:, 1:])
    assert diagnostics["gated_carry"]["reader_calls"] == diagnostics["gated_carry"]["writer_calls"] == 3
    assert len(diagnostics["gated_carry"]["gate_values"]) == 3
    assert torch.allclose(diagnostics["slot_inputs"][1], diagnostics["slot_writes"][0])
    assert not torch.allclose(diagnostics["slot_inputs"][1], ordinary_diagnostics["slot_inputs"][1])
    assert diagnostics["detached_writes"] is False
    assert not adapter.reader._forward_pre_hooks  # type: ignore[attr-defined]
    assert not adapter.writer._forward_hooks  # type: ignore[attr-defined]


def test_final_only_loss_reaches_earlier_state_and_gate_without_detach() -> None:
    model, adapter, x_bits, y_bits, ops = _gated_fixture()
    (logits_x, logits_y), diagnostics = gated.gated_latent_slots_forward(
        model, adapter, x_bits, y_bits, ops, return_diagnostics=True
    )
    first_state = diagnostics["slot_inputs"][0]
    first_state.retain_grad()
    loss = F.cross_entropy(logits_x[:, -1], torch.tensor([2, 3])) + F.cross_entropy(
        logits_y[:, -1], torch.tensor([4, 5])
    )
    assert torch.isfinite(loss) and loss.item() != 0.0
    loss.backward()
    assert first_state.grad is not None and torch.count_nonzero(first_state.grad) > 0
    assert torch.isfinite(first_state.grad).all()
    assert adapter.carry_gate.weight.grad is not None
    assert torch.count_nonzero(adapter.carry_gate.weight.grad) > 0
    assert torch.isfinite(adapter.carry_gate.weight.grad).all()
    assert all(value.grad_fn is not None for value in diagnostics["gated_carry"]["gate_values"])


def test_raw_nonfinite_proposal_is_rejected_before_blending_and_hooks_are_cleaned() -> None:
    model, adapter, x_bits, y_bits, ops = _gated_fixture(length=1)
    original_writer = adapter.writer.forward

    def bad_writer(value: Tensor) -> Tensor:
        return original_writer(value).fill_(float("nan"))

    adapter.writer.forward = bad_writer  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="raw writer proposal"):
        gated.gated_latent_slots_forward(model, adapter, x_bits, y_bits, ops)
    assert not adapter.reader._forward_pre_hooks  # type: ignore[attr-defined]
    assert not adapter.writer._forward_hooks  # type: ignore[attr-defined]

    model, adapter, x_bits, y_bits, ops = _gated_fixture(length=1)
    original_gate_forward = adapter.carry_gate.forward

    def bad_gate(z: Tensor, hidden: Tensor) -> Tensor:
        return original_gate_forward(z, hidden).fill_(float("nan"))

    adapter.carry_gate.forward = bad_gate  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="nonfinite"):
        gated.gated_latent_slots_forward(model, adapter, x_bits, y_bits, ops)
    assert not adapter.reader._forward_pre_hooks  # type: ignore[attr-defined]
    assert not adapter.writer._forward_hooks  # type: ignore[attr-defined]


def test_gate_only_optimizer_group_copies_adapter_hyperparameters_and_has_no_moments() -> None:
    adapter, _ = latent.make_adapter()
    parent = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [parent], "param_names": ["parent"], "lr": 0.03, "weight_decay": 0.07}])
    optimizer.add_param_group({"params": list(adapter.parameters()), "param_names": list(latent.ADAPTER_PARAMETER_NAMES), "lr": 0.011, "weight_decay": 0.02})
    gate, _ = gated.attach_carry_gate(adapter)
    names = gated.append_carry_gate_optimizer_group(optimizer, adapter)
    assert names == gated.CARRY_GATE_PARAMETER_NAMES
    assert optimizer.param_groups[-1]["lr"] == 0.011
    assert optimizer.param_groups[-1]["weight_decay"] == 0.02
    assert list(optimizer.param_groups[-1]["params"]) == list(gate.parameters())
    assert not optimizer.state
    assert gated.validate_optimizer_carry_gate_association(optimizer, adapter) == names
    optimizer.param_groups[-1]["params"] = list(reversed(optimizer.param_groups[-1]["params"]))
    with pytest.raises(ValueError, match="parameters/order"):
        gated.validate_optimizer_carry_gate_association(optimizer, adapter)


def test_gate_identity_and_adapter_parameter_count_include_only_new_gate() -> None:
    adapter, _ = latent.make_adapter()
    gate, _ = gated.attach_carry_gate(adapter)
    identity = gated.carry_gate_identity(adapter)
    assert adapter.parameter_count() == gated.GATED_ADAPTER_PARAMETER_COUNT
    assert identity["parameter_names"] == list(gated.CARRY_GATE_PARAMETER_NAMES)
    assert identity["parameter_shapes"] == [[2, 160], [2]]
    assert identity["parameter_count"] == 322
    gated.validate_carry_gate_identity(adapter, copy.deepcopy(identity))
    with torch.no_grad():
        gate.bias[0].add_(1.0)
    with pytest.raises(ValueError, match="identity mismatch"):
        gated.validate_carry_gate_identity(adapter, identity)


def test_identity_controls_cover_exact_lengths_and_all_dsl_states() -> None:
    specs = gated.validate_control_program_specs()
    assert len(specs) == 81
    assert {length: sum(spec["length"] == length for spec in specs) for length in (11, 19, 35)} == {11: 27, 19: 27, 35: 27}
    for spec in specs:
        assert len(spec["program"]) == spec["length"]
        cell = dict(gated.IDENTITY_CELLS)[spec["cell"]]
        assert tuple(spec["program"][2:-1]) == cell * spec["repeat"]
        for state in dsl.STATE_ORDER:
            assert dsl.execute_program(cell, state) == state
    changed = list(specs)
    changed[0] = {**changed[0], "program": tuple(changed[0]["program"][:-1]) + ("XOR",)}
    with pytest.raises(ValueError, match="identity control"):
        gated.validate_control_program_specs(changed)


def test_fixed_training_qa_and_science_budgets() -> None:
    accounting = gated.training_accounting()
    assert accounting == {
        "updates_per_arm": 1000,
        "arms": 2,
        "forwards": 2000,
        "cases": 128000,
        "positions": 447488,
        "native_steps": 3579904,
        "backwards": 2000,
        "optimizer_updates": 2000,
    }
    assert gated.QA_ACCOUNTING == {
        "forwards": 6, "cases": 12, "positions": 20, "native_steps": 160,
        "backwards": 6, "optimizer_updates": 6, "underlying_deserializations": 8,
    }
    assert gated.SCIENCE_ACCOUNTING == {
        "training_forwards": 2000, "evaluation_forwards": 531, "forwards": 2531,
        "cases": 263936, "positions": 3239936, "native_steps": 25919488,
        "backwards": 2000, "optimizer_updates": 2000, "underlying_deserializations": 6,
    }
    with pytest.raises(ValueError, match="stream length"):
        gated.training_accounting(gated.TRAINING_BATCH_LENGTHS[:-1] + (6,))


def test_reuses_registered_paired_repair_regression_and_tie_metrics() -> None:
    target = [[0, 0], [1, 1]]
    states = ((0, 0), (1, 1), (2, 2), (3, 3))

    def row(predictions: list[list[list[int]]]) -> dict[str, object]:
        return {
            "id": "hand",
            "suite": "identity_controls",
            "length": 2,
            "program": ["ADD", "ADD"],
            "predictions": [
                {
                    "state": list(state),
                    "stratum": "test",
                    "target_trace": target,
                    "predicted_trace": predicted,
                    **accepted_trace_metrics(target, predicted),
                }
                for state, predicted in zip(states, predictions)
            ],
        }

    candidate = [row([
        [[0, 1], [1, 1]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]],
    ])]
    reference = [row([
        [[0, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]],
    ])]
    paired = accepted_paired_metrics(candidate, reference, label="gated_candidate_reference")
    assert paired["aggregate"]["final"] == {
        "left_correct_right_wrong": 1, "left_wrong_right_correct": 1,
        "both_correct": 1, "both_wrong": 1,
    }
    assert paired["aggregate"]["full_trace"] == {
        "left_correct_right_wrong": 0, "left_wrong_right_correct": 1,
        "both_correct": 1, "both_wrong": 2,
    }
    assert paired["improvements"] == {"final": 1, "full_trace": 0}
    assert paired["regressions"] == {"final": 1, "full_trace": 1}
    assert paired["ties"] == {"final": 2, "full_trace": 3}
