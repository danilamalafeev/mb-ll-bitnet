from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_runtime as runtime


class _TinyParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)


def _parent_with_adapter() -> tuple[_TinyParent, torch.optim.Optimizer, latent.LatentSlotAdapter, dict[str, object]]:
    model = _TinyParent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    adapter, initialization, identity, _names = runtime._append_adapter_group(
        model, optimizer, device=torch.device("cpu")
    )
    return model, optimizer, adapter, {
        "initialization": initialization,
        "identity": identity,
    }


def _batch() -> list[dsl.RegisterExample]:
    return [
        dsl.RegisterExample(0, 1, ("ADD",)),
        dsl.RegisterExample(2, 3, ("ADD",)),
    ]


def test_append_group_copies_parent_hyperparameters_and_rejects_multiple_groups() -> None:
    model, optimizer, adapter, _metadata = _parent_with_adapter()
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[1]["weight_decay"] == optimizer.param_groups[0]["weight_decay"]
    assert optimizer.param_groups[0]["param_names"] == ["linear.weight", "linear.bias"]
    assert optimizer.param_groups[1]["param_names"] == list(latent.ADAPTER_PARAMETER_NAMES)
    assert latent.validate_optimizer_adapter_association(optimizer, adapter) == latent.ADAPTER_PARAMETER_NAMES

    second_group_model = _TinyParent()
    multiple = torch.optim.AdamW([
        {"params": list(second_group_model.linear.parameters()), "lr": 0.01},
        {"params": [nn.Parameter(torch.ones(()))], "lr": 0.02},
    ])
    with pytest.raises(ValueError, match="exactly one"):
        runtime._append_adapter_group(second_group_model, multiple, device=torch.device("cpu"))


def test_update_persists_attempted_and_completed_phase_accounting(tmp_path: Path) -> None:
    model, optimizer, adapter, _metadata = _parent_with_adapter()
    counter = runtime._new_counter()
    accounting = tmp_path / "accounting.json"
    sink = runtime._counter_sink(accounting)

    def synthetic_forward(model_: nn.Module, adapter_: latent.LatentSlotAdapter, x: torch.Tensor, y: torch.Tensor, ops: torch.Tensor):
        seed = model_.linear.weight.sum() * 0.01 + adapter_.writer.weight.sum() * 0.001
        logits = seed.expand(x.shape[0], ops.shape[1], latent.SLOT_WIDTH)
        return logits, logits + 0.05

    loss = runtime._run_update(
        model, adapter, optimizer, _batch(), device=torch.device("cpu"), counter=counter, sink=sink,
        forward=synthetic_forward,
    )
    assert loss > 0.0 and torch.isfinite(torch.tensor(loss))
    for key in (
        "attempted_updates", "completed_updates", "attempted_forwards", "completed_forwards",
        "attempted_backwards", "completed_backwards", "attempted_optimizer_steps", "completed_optimizer_steps",
    ):
        assert counter[key] == 1
    assert counter["attempted_cases"] == counter["completed_cases"] == 2
    assert counter["attempted_readout_positions"] == counter["completed_readout_positions"] == 2
    assert counter["attempted_native_steps"] == counter["completed_native_steps"] == 16
    assert accounting.is_file()


def test_snapshot_round_trip_restores_adapter_parent_optimizer_rng_and_boundary(tmp_path: Path) -> None:
    torch.manual_seed(801)
    model, optimizer, adapter, metadata = _parent_with_adapter()
    for parameter in [*model.parameters(), *adapter.parameters()]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parent_identity = {
        "checkpoint_sha256": "a" * 64,
        "model_digest": "b" * 64,
        "optimizer_digest": "c" * 64,
        "rng_digest": "d" * 64,
        "manifest_digest": "e" * 64,
        "label": "fixture",
    }
    snapshot = runtime._snapshot_payload(
        model, adapter, optimizer,
        parent_identity=parent_identity,
        source_binding_digest="f" * 64,
        fixture_digest="1" * 64,
        manifest_digest="2" * 64,
        adapter_identity=metadata["identity"],  # type: ignore[arg-type]
        adapter_initialization=metadata["initialization"],  # type: ignore[arg-type]
        next_batch_index=1,
        local_update=1,
    )
    path = tmp_path / "snapshot_l1.pt"
    runtime._atomic_torch(path, snapshot, refuse=True)
    loaded = runtime._load_snapshot(path)
    assert loaded["complete"] is True
    assert loaded["committed"] is True
    assert not path.with_name(path.name + ".tmp").exists()

    expected = runtime._state_identity(model, adapter, optimizer)
    with torch.no_grad():
        for parameter in [*model.parameters(), *adapter.parameters()]:
            parameter.add_(3.0)
    torch.manual_seed(999)
    restored_digest = runtime._restore_snapshot(
        loaded, model, adapter, optimizer,
        parent_identity=parent_identity,
        source_binding_digest="f" * 64,
        fixture_digest="1" * 64,
        manifest_digest="2" * 64,
        expected_adapter_identity=metadata["identity"],  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    actual = runtime._state_identity(model, adapter, optimizer)
    assert restored_digest == snapshot["snapshot_digest"]
    assert actual == expected


def test_atomic_commit_preserves_previous_checkpoint_and_rejects_temp_or_incomplete(tmp_path: Path) -> None:
    model, optimizer, adapter, metadata = _parent_with_adapter()
    parent_identity = {"checkpoint_sha256": "a" * 64, "model_digest": "b" * 64, "optimizer_digest": "c" * 64, "rng_digest": "d" * 64, "manifest_digest": "e" * 64}
    snapshot = runtime._snapshot_payload(
        model, adapter, optimizer, parent_identity=parent_identity, source_binding_digest="f" * 64,
        fixture_digest="1" * 64, manifest_digest="2" * 64, adapter_identity=metadata["identity"],
        adapter_initialization=metadata["initialization"], next_batch_index=1, local_update=1,
    )
    path = tmp_path / "committed.pt"
    runtime._atomic_torch(path, snapshot, refuse=True)
    original = path.read_bytes()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="temporary collision"):
        runtime._atomic_torch(path, snapshot, refuse=True)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="temporary snapshot"):
        runtime._load_snapshot(temporary)
    temporary.unlink()

    incomplete = dict(snapshot)
    incomplete["complete"] = False
    incomplete_path = tmp_path / "incomplete.pt"
    runtime._atomic_torch(incomplete_path, incomplete, refuse=True)
    with pytest.raises(ValueError, match="incomplete"):
        runtime._load_snapshot(incomplete_path)
