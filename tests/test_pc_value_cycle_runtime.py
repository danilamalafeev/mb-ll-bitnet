from __future__ import annotations

import torch
from torch import Tensor, nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_value_cycle_contract as contract
from scripts import pc_value_cycle_runtime as runtime


class _ToyReader(nn.Module):
    num_heads = 4
    head_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.v_proj = nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            self.v_proj.weight.copy_(torch.eye(128))

    def build_kv(self, key_memory: Tensor, value_memory: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.v_proj(value_memory).view(value_memory.shape[0], 2, 4, 32)
        return key_memory, projected.transpose(1, 2)


class _ToyCore(nn.Module):
    native_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.role_keys = nn.Parameter(torch.zeros(2, 128))
        self.memory_norm = nn.LayerNorm(128)
        self.reader = _ToyReader()
        self.output_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 16)

    def step(self, cache: dict[str, object], opcode: Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, object]]:
        h = cache["h"] + opcode.float().unsqueeze(1) * 0.01  # type: ignore[operator]
        logits = self.head(h)
        return (logits, logits + 0.1), {"h": h + 0.2, "kv": cache["kv"], "substeps": cache["substeps"] + 8}  # type: ignore[operator]


def _batches() -> list[list[dsl.RegisterExample]]:
    return [
        [dsl.RegisterExample(1, 3, ("SWAP", "SWAP")), dsl.RegisterExample(2, 5, ("SWAP", "SWAP"))],
        [dsl.RegisterExample(1, 3, ("XOR", "XOR")), dsl.RegisterExample(2, 5, ("XOR", "XOR"))],
    ]


def test_fixture_is_immutable_and_round_trips() -> None:
    source = {"digest": "source"}
    batches, fixture = runtime._fixture(source_binding=source, settings={"device": "cpu"})
    assert [len(batch[0].program) for batch in batches] == [2, 2]
    assert runtime._fixture_batches(fixture)[1][0].program == ("XOR", "XOR")
    assert fixture["contract"]["projected_state"] == "per_slot_projected_V"
    assert fixture["budget"]["native_steps"] == runtime.QA_NATIVE_STEPS


def test_runtime_update_adds_cycle_term_without_extra_native_forward() -> None:
    model = _ToyCore()
    adapter, _ = __import__("scripts.pc_latent_slots", fromlist=["make_adapter"]).make_adapter()
    optimizer = torch.optim.AdamW([*model.parameters(), *adapter.parameters()], lr=0.01)
    counter = runtime._new_counter()
    loss, metadata = runtime._run_update(
        model=model,
        adapter=adapter,
        optimizer=optimizer,
        batch=_batches()[0],
        device=torch.device("cpu"),
        counter=counter,
        sink=None,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert metadata["cycle"]["windows"] == 2
    assert counter["completed_forwards"] == 1
    assert counter["completed_native_steps"] == 2 * 2 * 8
    assert counter["cycle_windows"] == 2
    assert counter["cycle_updates"] == 1


def test_cycle_contract_constants_bind_to_runtime_fixture() -> None:
    assert runtime.QA_CALLS == 3
    assert runtime.QA_UPDATES == 3
    assert runtime.QA_CASES == 6
    assert runtime.QA_POSITIONS == 12
    assert runtime.QA_NATIVE_STEPS == 96
    assert runtime.QA_DESERIALIZATIONS == 3
    assert runtime.QA_CYCLE_WINDOWS == 6
    assert runtime.contract.VALUE_CYCLE_ARCHITECTURE_ID == contract.VALUE_CYCLE_ARCHITECTURE_ID
    assert [item[0] for item in contract.IDENTITY_CYCLES] == ["SWAP2", "XOR2", "SWAP_XOR2_SWAP"]


def test_accounting_validator_requires_exact_cycle_qa_counts() -> None:
    counter = runtime._new_counter()
    try:
        runtime._validate_accounting(counter)
    except ValueError as exc:
        assert "accounting mismatch" in str(exc)
    else:
        raise AssertionError("partial accounting unexpectedly accepted")
    expected = {
        "attempted_endpoint_loads": 1,
        "completed_endpoint_loads": 1,
        "attempted_snapshot_loads": 1,
        "completed_snapshot_loads": 1,
        "attempted_forwards": 3,
        "completed_forwards": 3,
        "attempted_cases": 6,
        "completed_cases": 6,
        "attempted_readout_positions": 12,
        "completed_readout_positions": 12,
        "attempted_native_steps": 96,
        "completed_native_steps": 96,
        "attempted_backwards": 3,
        "completed_backwards": 3,
        "attempted_optimizer_steps": 3,
        "completed_optimizer_steps": 3,
        "completed_updates": 3,
        "attempted_underlying_deserializations": 3,
        "completed_underlying_deserializations": 3,
        "cycle_windows": 6,
        "cycle_updates": 3,
    }
    counter.update(expected)
    runtime._validate_accounting(counter)
