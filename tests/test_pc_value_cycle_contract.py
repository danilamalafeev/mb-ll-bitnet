from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from scripts import pc_latent_slots as latent
from scripts import pc_value_cycle_contract as contract


class _ToyReader(nn.Module):
    num_heads = 4
    head_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.v_proj = nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            self.v_proj.weight.copy_(torch.eye(128))

    def build_kv(self, key_memory: Tensor, value_memory: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.v_proj(value_memory).view(value_memory.shape[0], 2, self.num_heads, self.head_dim)
        return key_memory, projected.transpose(1, 2)


class _ToyCore(nn.Module):
    native_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.role_keys = nn.Parameter(torch.linspace(-1.0, 1.0, 2 * 128).reshape(2, 128))
        self.memory_norm = nn.LayerNorm(128, eps=1e-5)
        self.reader = _ToyReader()
        self.output_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 16, bias=False)

    def step(self, cache: dict[str, object], opcode: Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, object]]:
        h = cache["h"] + opcode.to(dtype=cache["h"].dtype).unsqueeze(1) * 0.01  # type: ignore[operator]
        logits = self.head(h)
        return (logits, logits + 0.1), {"h": h + 0.2, "kv": cache["kv"], "substeps": cache["substeps"] + 8}  # type: ignore[operator]


def _slots() -> tuple[latent.LatentSlotAdapter, Tensor]:
    adapter, _ = latent.make_adapter()
    slots = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16) / 17.0
    return adapter, slots


def test_projected_values_follow_exact_reader_value_projection() -> None:
    adapter, slots = _slots()
    model = _ToyCore()
    actual = contract.projected_values_from_slots(model, adapter, slots)
    normalized = model.memory_norm(adapter.reader(slots))
    expected = model.reader.v_proj(normalized).view(2, 2, 4, 32).transpose(1, 2)
    assert actual.shape == (2, 4, 2, 32)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)
    cache = latent.fresh_cache_from_slots(model, adapter, slots)
    assert torch.allclose(actual, cache["kv"][1], atol=1e-6, rtol=0.0)  # type: ignore[index]


def test_identity_cycle_windows_prefer_longest_and_do_not_overlap() -> None:
    ops = torch.tensor([
        [2, 2, 2, 2],  # two disjoint SWAP2 cycles
        [2, 1, 1, 2],  # one mixed identity cycle
        [0, 2, 0, 1],  # no complete identity cycle
    ], dtype=torch.long)
    windows = contract.find_identity_cycle_windows(ops)
    assert [(item["batch_index"], item["start"], item["cycle_id"]) for item in windows] == [
        (0, 0, "SWAP2"),
        (0, 2, "SWAP2"),
        (1, 0, "SWAP_XOR2_SWAP"),
    ]


def test_projected_value_cycle_loss_uses_post_cycle_gradient_and_metadata() -> None:
    adapter, _ = latent.make_adapter()
    model = _ToyCore()
    bits = torch.ones(2, 4)
    ops = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    (_x, _y), diagnostics = latent.latent_slots_forward(model, adapter, bits, bits, ops, return_diagnostics=True)
    loss, metadata = contract.projected_value_cycle_loss(model, adapter, diagnostics, ops)
    assert metadata["windows"] == 2
    assert metadata["by_cycle"] == {"SWAP2": 1, "XOR2": 1}
    assert metadata["target_detached"] is True
    assert metadata["projected_state"] == "per_slot_projected_V"
    assert metadata["model_forwards"] == 0
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert adapter.writer.weight.grad is not None
    assert torch.count_nonzero(adapter.writer.weight.grad) > 0
    assert torch.isfinite(adapter.writer.weight.grad).all()


def test_no_cycle_returns_differentiable_zero_without_extra_model_work() -> None:
    adapter, _ = latent.make_adapter()
    model = _ToyCore()
    bits = torch.ones(2, 4)
    ops = torch.tensor([[0, 2], [0, 1]], dtype=torch.long)
    (_x, _y), diagnostics = latent.latent_slots_forward(model, adapter, bits, bits, ops, return_diagnostics=True)
    loss, metadata = contract.projected_value_cycle_loss(model, adapter, diagnostics, ops)
    assert metadata["windows"] == 0
    assert metadata["raw_loss"] == 0.0
    assert metadata["weighted_loss"] == 0.0
    assert metadata["model_forwards"] == 0
    assert loss.requires_grad
    loss.backward()


def test_value_cycle_contract_rejects_malformed_inputs() -> None:
    adapter, slots = _slots()
    model = _ToyCore()
    with pytest.raises(ValueError, match="ops"):
        contract.find_identity_cycle_windows(torch.tensor([1, 1], dtype=torch.long))
    with pytest.raises(ValueError, match="invalid opcode"):
        contract.find_identity_cycle_windows(torch.tensor([[3, 1]], dtype=torch.long))
    with pytest.raises(ValueError, match="weight"):
        contract.projected_value_cycle_loss(model, adapter, {"slot_inputs": [slots], "slot_writes": [slots]}, torch.tensor([[0]], dtype=torch.long), weight=-1.0)
    with pytest.raises(ValueError, match="memory_norm or reader.v_proj"):
        contract.projected_values_from_slots(object(), adapter, slots)
