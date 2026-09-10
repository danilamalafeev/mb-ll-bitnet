from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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


def test_adapter_shape_count_local_initialization_and_global_rng_preservation() -> None:
    torch.manual_seed(2718)
    before = torch.get_rng_state().clone()
    adapter, first_metadata = latent.make_adapter()
    after = torch.get_rng_state()
    assert torch.equal(after, before)
    assert adapter.parameter_count() == 6464
    assert tuple(name for name, _ in adapter.named_parameters()) == latent.ADAPTER_PARAMETER_NAMES
    assert [tuple(parameter.shape) for parameter in adapter.parameters()] == [
        (32, 8), (32,), (128, 16), (32, 128), (32,)
    ]
    assert first_metadata["order"] == ["initializer", "reader", "writer"]
    assert first_metadata["seed"] == 0
    assert torch.count_nonzero(adapter.initializer.bias) == 0
    assert torch.count_nonzero(adapter.writer.bias) == 0
    assert torch.count_nonzero(adapter.initializer.weight) > 0
    assert torch.count_nonzero(adapter.reader.weight) > 0
    assert torch.count_nonzero(adapter.writer.weight) > 0
    second, second_metadata = latent.make_adapter()
    assert second_metadata == first_metadata
    assert all(torch.equal(left, right) for left, right in zip(adapter.parameters(), second.parameters()))
    assert latent.device_cross_entropy is accepted.device_cross_entropy


def test_fresh_cache_is_projection_then_normalization_and_rebuilds_from_slots_only() -> None:
    adapter, _ = latent.make_adapter()
    model = _ToyCore()
    slots = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16) / 17.0
    cache = latent.fresh_cache_from_slots(model, adapter, slots)
    projected = adapter.reader(slots)
    def hand_layer_norm(value: Tensor) -> Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value - mean).square().mean(dim=-1, keepdim=True)
        return ((value - mean) / torch.sqrt(variance + model.memory_norm.eps)) * model.memory_norm.weight + model.memory_norm.bias

    expected_keys = hand_layer_norm(model.role_keys)
    expected_values = hand_layer_norm(projected)
    assert torch.allclose(cache["kv"][0], expected_keys.unsqueeze(0).expand(2, -1, -1), atol=1e-6)  # type: ignore[index]
    assert torch.allclose(cache["h"], projected[:, 0] + projected[:, 1])
    assert torch.allclose(cache["kv"][1], expected_values, atol=1e-6)  # type: ignore[index]
    normalized_slots = F.layer_norm(slots, (latent.SLOT_WIDTH,), eps=model.memory_norm.eps)
    wrong_order = adapter.reader(normalized_slots)
    assert not torch.allclose(cache["kv"][1], wrong_order)  # type: ignore[index]

    poisoned = {
        "h": cache["h"].detach().clone(),
        "kv": (cache["kv"][0].detach().clone(), cache["kv"][1].detach().clone()),  # type: ignore[index]
        "substeps": cache["substeps"],
    }
    poisoned["h"].add_(1000.0)  # type: ignore[union-attr]
    poisoned["kv"] = (poisoned["kv"][0] + 1000.0, poisoned["kv"][1] + 1000.0)  # type: ignore[index,operator]
    x_bits, y_bits = _inputs()
    x_bits.add_(7.0)
    y_bits.mul_(-3.0)
    rebuilt = latent.fresh_cache_from_slots(model, adapter, slots)
    assert torch.allclose(rebuilt["h"], cache["h"])
    assert torch.allclose(rebuilt["kv"][1], cache["kv"][1])  # type: ignore[index]
    assert poisoned["h"].abs().mean() > cache["h"].abs().mean()  # type: ignore[union-attr]


def test_latent_forward_uses_one_initializer_native8_and_consumes_wrong_writes() -> None:
    adapter, _ = latent.make_adapter()
    model = _ToyCore()
    x_bits, y_bits = _inputs()
    with torch.no_grad():
        adapter.writer.weight.zero_()
        adapter.writer.bias.fill_(7.0)
    calls = 0
    original = adapter.initializer.forward

    def counted(value: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        return original(value)

    adapter.initializer.forward = counted  # type: ignore[method-assign]
    original_step = model.step

    def poison_originals(cache: dict[str, object], opcode: Tensor):
        if calls == 1:
            x_bits.fill_(123.0)
            y_bits.fill_(-456.0)
        return original_step(cache, opcode)

    model.step = poison_originals  # type: ignore[method-assign]
    (logits_x, logits_y), diagnostics = latent.latent_slots_forward(
        model, adapter, x_bits, y_bits, [[0, 1], [0, 1]], return_diagnostics=True
    )
    assert calls == 1
    assert logits_x.shape == logits_y.shape == (2, 2, 16)
    assert diagnostics["initializer_calls"] == 1
    assert diagnostics["substeps"] == [0, 0]
    assert diagnostics["native_steps"] == [8, 8]
    assert diagnostics["targets_in_signature"] is False
    assert not diagnostics["detached_writes"]
    assert torch.allclose(diagnostics["slot_writes"][0], torch.full_like(diagnostics["slot_writes"][0], 7.0))
    assert torch.allclose(diagnostics["slot_inputs"][1], diagnostics["slot_writes"][0])
    assert not torch.allclose(diagnostics["slot_inputs"][1], diagnostics["slot_inputs"][0])
    assert "target" not in inspect.signature(latent.latent_slots_forward).parameters


def test_final_only_loss_reaches_first_write_and_writer_but_detached_control_does_not() -> None:
    adapter, _ = latent.make_adapter()
    model = _ToyCore()
    x_bits, y_bits = _inputs()
    (logits_x, logits_y), diagnostics = latent.latent_slots_forward(
        model, adapter, x_bits, y_bits, [[0, 1], [0, 1]], return_diagnostics=True
    )
    first_write = diagnostics["slot_writes"][0]
    first_write.retain_grad()
    loss = F.cross_entropy(logits_x[:, -1], torch.tensor([2, 3])) + F.cross_entropy(logits_y[:, -1], torch.tensor([4, 5]))
    assert torch.isfinite(loss) and loss.item() != 0.0
    loss.backward()
    assert first_write.grad is not None and torch.count_nonzero(first_write.grad) > 0
    assert adapter.writer.weight.grad is not None and torch.count_nonzero(adapter.writer.weight.grad) > 0
    assert torch.isfinite(first_write.grad).all()
    assert torch.isfinite(adapter.writer.weight.grad).all()

    detached_adapter, _ = latent.make_adapter()
    detached_model = _ToyCore()
    initial = latent.initialize_slots(detached_adapter, x_bits, y_bits)
    first_cache = latent.fresh_cache_from_slots(detached_model, detached_adapter, initial)
    (_first_x, _first_y), returned = detached_model.step(first_cache, torch.tensor([0, 0]))
    detached_slots = detached_adapter.writer(detached_model.output_norm(returned["h"])).reshape(2, 2, 16).detach()
    second_cache = latent.fresh_cache_from_slots(detached_model, detached_adapter, detached_slots)
    final_logits, _ = detached_model.step(second_cache, torch.tensor([1, 1]))
    detached_loss = F.cross_entropy(final_logits[0], torch.tensor([2, 3]))
    detached_loss.backward()
    assert detached_adapter.writer.weight.grad is None or torch.count_nonzero(detached_adapter.writer.weight.grad) == 0


def test_adapter_identity_and_optimizer_name_association_reject_mutation() -> None:
    adapter, _ = latent.make_adapter()
    identity = latent.adapter_identity(adapter)
    latent.validate_adapter_identity(adapter, identity)
    changed = copy.deepcopy(identity)
    changed["parameter_names"][0] = "wrong.weight"
    with pytest.raises(ValueError, match="identity mismatch|names/order"):
        latent.validate_adapter_identity(adapter, changed)
    changed_state = copy.deepcopy(identity)
    changed_state["state_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        latent.validate_adapter_identity(adapter, changed_state)
    original_weight = adapter.initializer.weight.detach().clone()
    with torch.no_grad():
        adapter.initializer.weight[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="identity mismatch"):
        latent.validate_adapter_identity(adapter, identity)
    with torch.no_grad():
        adapter.initializer.weight.copy_(original_weight)

    parent = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parent], lr=0.01)
    optimizer.add_param_group({"params": list(adapter.parameters()), "lr": 0.01})
    assert latent.validate_optimizer_adapter_association(optimizer, adapter) == latent.ADAPTER_PARAMETER_NAMES
    group = optimizer.param_groups[-1]
    group["params"] = list(reversed(group["params"]))
    with pytest.raises(ValueError, match="parameters/order"):
        latent.validate_optimizer_adapter_association(optimizer, adapter)


def test_reuses_registered_paired_metrics_with_recovery_and_hand_roles() -> None:
    target = [[0, 0], [1, 1]]
    states = ((0, 0), (1, 1), (2, 2), (3, 3))

    def row(predictions: list[list[list[int]]]) -> dict[str, object]:
        return {
            "id": "hand",
            "suite": "padding",
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
        [[0, 1], [1, 1]],  # repair after an earlier error
        [[0, 1], [0, 0]],  # candidate-only final error
        [[0, 0], [1, 1]],  # tie: both correct
        [[0, 1], [0, 0]],  # tie: both wrong
    ])]
    reference = [row([
        [[0, 1], [0, 0]],  # reference-only final error
        [[0, 0], [1, 1]],  # reference-only correct
        [[0, 0], [1, 1]],  # tie: both correct
        [[0, 1], [0, 0]],  # tie: both wrong
    ])]
    paired = accepted_paired_metrics(candidate, reference, label="hand_candidate_reference")
    assert paired["roles"] == {"left": "candidate", "right": "reference"}
    assert paired["aggregate"]["final"] == {
        "left_correct_right_wrong": 1,
        "left_wrong_right_correct": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }
    assert paired["aggregate"]["full_trace"] == {
        "left_correct_right_wrong": 0,
        "left_wrong_right_correct": 1,
        "both_correct": 1,
        "both_wrong": 2,
    }
    assert paired["improvements"] == {"final": 1, "full_trace": 0}
    assert paired["regressions"] == {"final": 1, "full_trace": 1}
    assert paired["ties"] == {"final": 2, "full_trace": 3}
    assert paired["programs"][0]["left_recovered_final"] == 1


def test_fixed_cyclic_accounting_and_checkpoint_boundary_gate() -> None:
    assert latent.FIXED_BATCH_LENGTHS[:12] == (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
    assert latent.FIXED_BATCH_LENGTHS[-2:] == (1, 2)
    assert latent.training_accounting(latent.FIXED_BATCH_LENGTHS) == latent.TRAINING_ACCOUNTING
    assert latent.SCIENCE_ACCOUNTING == {
        "calls": latent.TRAINING_ACCOUNTING["calls"] + 2 * 69,
        "cases": latent.TRAINING_ACCOUNTING["cases"] + 2 * 17664,
        "readout_positions": latent.TRAINING_ACCOUNTING["readout_positions"] + 2 * 331776,
        "native_steps": latent.TRAINING_ACCOUNTING["native_steps"] + 2 * 2654208,
        "optimizer_updates": latent.TRAINING_ACCOUNTING["optimizer_updates"],
    }
    assert latent.SCIENCE_ACCOUNTING["calls"] == 2000 + 2 * 69
    assert latent.SCIENCE_ACCOUNTING["cases"] == 128000 + 2 * 17664
    assert latent.SCIENCE_ACCOUNTING["readout_positions"] == 447744 + 2 * 331776
    assert latent.SCIENCE_ACCOUNTING["native_steps"] == 3581952 + 2 * 2654208
    assert latent.SCIENCE_ACCOUNTING["optimizer_updates"] == 2000
    assert latent.QA_ACCOUNTING["calls"] + latent.SCIENCE_ACCOUNTING["calls"] == latent.TOTAL_ACCOUNTING["calls"]
    with pytest.raises(ValueError, match="cycle or tail"):
        latent.training_accounting(latent.FIXED_BATCH_LENGTHS[:-1] + (3,))

    digest = "a" * 64
    candidates = [
        {"path": "u0500.PT.TMP", "committed": True, "manifest_digest": digest, "next_batch_index": 500},
        {"path": "u0750.pt", "committed": True, "manifest_digest": "b" * 64, "next_batch_index": 750},
        {"path": "u1000.pt", "committed": True, "manifest_digest": digest, "next_batch_index": 1000},
    ]
    selected = latent.select_committed_checkpoint(candidates, manifest_digest=digest, next_batch_index=1000)
    assert selected["path"] == "u1000.pt"
    with pytest.raises(ValueError, match="unique committed"):
        latent.select_committed_checkpoint(candidates[:1], manifest_digest=digest, next_batch_index=500)
