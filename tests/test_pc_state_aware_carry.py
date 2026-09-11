from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import Tensor, nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_state_aware_carry as aware


class _ToyReader:
    def build_kv(self, key_memory: Tensor, value_memory: Tensor) -> tuple[Tensor, Tensor]:
        return key_memory, value_memory


class _ToyCore(nn.Module):
    native_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.role_keys = nn.Parameter(torch.linspace(-1.0, 1.0, 2 * 128).reshape(2, 128))
        self.memory_norm = nn.LayerNorm(128)
        self.reader = _ToyReader()
        self.output_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 16, bias=False)

    def step(self, cache: dict[str, object], opcode: Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, object]]:
        values = cache["kv"][1]  # type: ignore[index]
        hidden = cache["h"] + values.mean(dim=1) * 0.125  # type: ignore[operator]
        hidden = hidden + opcode.to(dtype=hidden.dtype).unsqueeze(1) * 0.01
        logits = self.head(hidden)
        return (logits, logits + 0.1), {"h": hidden + 0.2, "kv": cache["kv"], "substeps": cache["substeps"] + 8}  # type: ignore[operator]


def _bits(values: list[int]) -> Tensor:
    matrix = torch.tensor(
        [[1.0 if (value >> bit) & 1 else -1.0 for bit in range(4)] for value in range(16)],
        dtype=torch.float32,
    )
    return matrix[torch.tensor(values, dtype=torch.long)]


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    return _bits([1, 6]), _bits([3, 12]), torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)


def test_wrap_is_zero_initialized_rng_safe_and_exactly_retains_base_writer() -> None:
    torch.manual_seed(2718)
    adapter, _ = latent.make_adapter()
    old_writer = adapter.writer
    old_weight = old_writer.weight
    old_bias = old_writer.bias
    before = torch.get_rng_state().clone()
    writer, metadata = aware.make_state_aware_writer(base_writer=old_writer)
    assert torch.equal(torch.get_rng_state(), before)
    assert writer.base_writer is old_writer
    assert writer.parameter_count() == latent.WRITER_PARAMETER_COUNT + aware.CORRECTION_PARAMETER_COUNT
    assert aware.CORRECTION_PARAMETER_COUNT == 5152
    assert writer.correction_is_zero()
    assert metadata["correction_initialization"] == "zero"
    assert writer.base_writer.weight is old_weight
    assert writer.base_writer.bias is old_bias
    hidden = torch.randn(5, latent.HIDDEN_WIDTH)
    features = torch.cat((torch.randn(5, latent.OUTPUT_WIDTH), hidden), dim=-1)
    assert torch.equal(writer(features), old_writer(hidden))


def test_state_aware_features_keep_both_slots_and_correction_can_use_them() -> None:
    adapter, _ = latent.make_adapter()
    writer, _ = aware.make_state_aware_writer(base_writer=adapter.writer)
    z = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16) / 10.0
    hidden = torch.randn(2, 128)
    features = aware.state_aware_features(z, hidden)
    assert features.shape == (2, 160)
    assert torch.equal(features[:, :32], z.reshape(2, 32))
    assert torch.equal(features[:, 32:], hidden)
    with torch.no_grad():
        writer.correction.weight.zero_()
        writer.correction.weight[:, :32].fill_(0.01)
    changed = writer(aware.state_aware_features(z + 1.0, hidden))
    baseline = writer(aware.state_aware_features(z, hidden))
    assert not torch.equal(changed, baseline)
    assert torch.isfinite(aware.state_aware_slot_write(z, hidden, writer)).all()


def test_state_aware_forward_preserves_accepted_logits_at_zero_correction_and_tracks_hooks() -> None:
    torch.manual_seed(1234)
    model = _ToyCore()
    adapter, _ = latent.make_adapter()
    old_adapter = copy.deepcopy(adapter)
    aware.install_state_aware_writer(adapter)
    x_bits, y_bits, ops = _inputs()
    ordinary = latent.latent_slots_forward(model, old_adapter, x_bits, y_bits, ops, return_diagnostics=True)
    carried = aware.state_aware_latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
    assert torch.equal(ordinary[0][0], carried[0][0])
    assert torch.equal(ordinary[0][1], carried[0][1])
    assert torch.equal(ordinary[1]["slot_writes"][0], carried[1]["slot_writes"][0])
    assert carried[1]["state_aware_carry"] == {"reader_calls": 3, "writer_calls": 3}
    assert not adapter.reader._forward_pre_hooks  # type: ignore[attr-defined]
    assert not adapter.writer._forward_pre_hooks  # type: ignore[attr-defined]
    assert carried[1]["writer_receives"] == "[current_slots, returned_hidden]"


def test_hooks_remove_on_failure_and_reject_uncoupled_writer() -> None:
    adapter, _ = latent.make_adapter()
    aware.install_state_aware_writer(adapter)
    with pytest.raises(RuntimeError, match="before its reader"):
        with aware.state_aware_hooks(adapter):
            adapter.writer(torch.zeros(2, aware.STATE_INPUT_WIDTH))
    assert not adapter.reader._forward_pre_hooks  # type: ignore[attr-defined]
    assert not adapter.writer._forward_pre_hooks  # type: ignore[attr-defined]
    state = aware.StateAwareHookState()
    with pytest.raises(ValueError, match="shape"):
        state._reader_pre_hook(adapter.reader, (torch.zeros(2, 16),))


def test_install_preserves_optimizer_parameter_objects_and_appends_only_new_state() -> None:
    adapter, _ = latent.make_adapter()
    parent = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [parent], "param_names": ["parent"], "lr": 0.03}])
    optimizer.add_param_group({"params": list(adapter.parameters()), "param_names": list(latent.ADAPTER_PARAMETER_NAMES), "lr": 0.011})
    loss = adapter.writer(torch.zeros(2, 128)).square().mean()
    loss.backward()
    optimizer.step()
    old_writer_weight = adapter.writer.weight
    old_writer_state = copy.deepcopy(optimizer.state[old_writer_weight])
    writer, metadata = aware.install_state_aware_writer(adapter, optimizer)
    assert writer.base_writer.weight is old_writer_weight
    assert optimizer.state[old_writer_weight]["step"].equal(old_writer_state["step"])
    assert optimizer.state[old_writer_weight]["exp_avg"].equal(old_writer_state["exp_avg"])
    assert writer.correction.weight not in optimizer.state
    assert list(optimizer.param_groups[-1]["params"]) == list(adapter.parameters())
    assert optimizer.param_groups[-1]["param_names"] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)
    assert aware.validate_optimizer_state_aware_association(optimizer, adapter) == aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES
    assert metadata["base_writer_parameter_ids_preserved"] is True


def test_state_aware_identity_and_signature_are_explicit() -> None:
    adapter, _ = latent.make_adapter()
    aware.install_state_aware_writer(adapter)
    identity = aware.state_aware_adapter_identity(adapter)
    assert identity["architecture_id"] == aware.STATE_AWARE_ARCHITECTURE_ID
    assert identity["parameter_count"] == aware.STATE_AWARE_ADAPTER_PARAMETER_COUNT
    assert identity["parameter_names"] == list(aware.STATE_AWARE_ADAPTER_PARAMETER_NAMES)
    aware.validate_state_aware_adapter_identity(adapter, copy.deepcopy(identity))
    with torch.no_grad():
        adapter.writer.correction.bias[0].add_(1.0)
    with pytest.raises(ValueError, match="identity mismatch"):
        aware.validate_state_aware_adapter_identity(adapter, identity)
    assert tuple(inspect.signature(aware.StateAwareWriter.forward).parameters) == ("self", "combined")
    assert "target" not in inspect.signature(aware.state_aware_latent_slots_forward).parameters


def test_malformed_inputs_are_rejected_without_silent_projection() -> None:
    adapter, _ = latent.make_adapter()
    writer, _ = aware.make_state_aware_writer(base_writer=adapter.writer)
    with pytest.raises(ValueError, match="shape"):
        writer(torch.zeros(2, 128))
    with pytest.raises(ValueError, match="nonfinite"):
        aware.state_aware_features(torch.zeros(1, 2, 16), torch.full((1, 128), float("nan")))
    with pytest.raises(ValueError, match="float32"):
        aware.state_aware_features(torch.zeros(1, 2, 16, dtype=torch.float64), torch.zeros(1, 128, dtype=torch.float64))
    aware.install_state_aware_writer(adapter)
    with pytest.raises(ValueError, match="already has"):
        aware.install_state_aware_writer(adapter)
