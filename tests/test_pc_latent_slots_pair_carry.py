from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import Tensor, nn

from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_pair_carry as pair
from scripts import pc_learned_scratchpad as accepted
from scripts.pc_learned_scratchpad_runtime import (
    _paired_metrics as accepted_paired_metrics,
    _trace_metrics as accepted_trace_metrics,
)


class _OracleReader:
    def build_kv(self, key_memory: Tensor, value_memory: Tensor) -> tuple[Tensor, Tensor]:
        return key_memory, value_memory


class _EmbedReader(nn.Module):
    """Toy adapter reader that keeps both 16-wide slots visible to the core."""

    def forward(self, slots: Tensor) -> Tensor:
        zeros = torch.zeros(
            (slots.shape[0], slots.shape[1], latent.HIDDEN_WIDTH - latent.SLOT_WIDTH),
            device=slots.device,
            dtype=slots.dtype,
        )
        return torch.cat((slots, zeros), dim=-1)


class _SwapOracle(nn.Module):
    """Independent CPU oracle: SWAP is nonidentity once, identity twice."""

    native_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.role_keys = nn.Parameter(torch.zeros(2, latent.HIDDEN_WIDTH))
        self.memory_norm = nn.Identity()
        self.reader = _OracleReader()
        self.output_norm = nn.Identity()

    def step(self, cache: dict[str, object], opcode: Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, object]]:
        values = cache["kv"][1]  # type: ignore[index]
        current = torch.cat((values[:, 0, :16], values[:, 1, :16]), dim=-1)  # type: ignore[index]
        swapped = torch.cat((current[:, 16:], current[:, :16]), dim=-1)
        is_swap = opcode.eq(2).unsqueeze(1)
        next_slots = torch.where(is_swap, swapped, current)
        h = torch.cat(
            (
                next_slots,
                torch.zeros((next_slots.shape[0], latent.HIDDEN_WIDTH - 32), device=next_slots.device, dtype=next_slots.dtype),
            ),
            dim=-1,
        )
        logits_x = next_slots[:, :16]
        logits_y = next_slots[:, 16:] + 0.25
        return (logits_x, logits_y), {"h": h, "kv": cache["kv"], "substeps": cache["substeps"] + 8}  # type: ignore[operator]


TOY_PROGRAM = ("ADD", "SWAP", "SWAP", "ADD", "SWAP", "SWAP", "ADD")
TOY_PAIRS = ((2, 3), (5, 6))
TOY_OPS = torch.tensor([[0, 2, 2, 0, 2, 2, 0], [0, 2, 2, 0, 2, 2, 0]], dtype=torch.long)


def _bits(values: list[int]) -> Tensor:
    return accepted.signed_bit_matrix()[torch.tensor(values, dtype=torch.long)]


def _fixture() -> tuple[_SwapOracle, latent.LatentSlotAdapter, Tensor, Tensor]:
    adapter, _ = latent.make_adapter()
    with torch.no_grad():
        adapter.initializer.weight.zero_()
        adapter.initializer.bias.copy_(torch.arange(32, dtype=torch.float32))
    # The accepted adapter's dimensions stay intact, while this independent
    # reader exposes both slots for the toy core's semantic SWAP oracle.
    adapter.reader = _EmbedReader()
    ordinary_writer = adapter.writer
    calls = {"count": 0}

    def drifting_writer(hidden: Tensor) -> Tensor:
        calls["count"] += 1
        drift = torch.full((hidden.shape[0], latent.OUTPUT_WIDTH), 0.1 * calls["count"], dtype=hidden.dtype, device=hidden.device)
        return hidden[:, :latent.OUTPUT_WIDTH] + drift

    ordinary_writer.forward = drifting_writer  # type: ignore[method-assign]
    model = _SwapOracle()
    return model, adapter, _bits([1, 6]), _bits([3, 12])


def test_exact_focus_schedule_preserves_odd_tail_and_final_instruction() -> None:
    l24 = pair.expected_pair_schedule(24)
    l32 = pair.expected_pair_schedule(32)
    assert l24 == tuple((start, start + 1) for start in range(3, 22, 2))
    assert l32 == tuple((start, start + 1) for start in range(3, 30, 2))
    assert len(l24) == 10 and len(l32) == 14
    program = tuple(["ADD", "XOR"] + ["SWAP"] * 21 + ["ADD"])
    assert pair.expected_pair_schedule(program) == l24
    paired_positions = {position for current in l24 for position in current}
    assert 23 not in paired_positions and 24 not in paired_positions
    broken_tail = program[:-2] + ("ADD", "ADD")
    with pytest.raises(ValueError, match="odd SWAP tail"):
        pair.expected_pair_schedule(broken_tail)


@pytest.mark.parametrize(
    ("program", "pairs", "message"),
    [
        (("ADD", "SWAP", "SWAP"), ((1, 2),), "both be SWAP"),
        (("SWAP", "SWAP", "SWAP", "SWAP"), ((1, 2), (2, 3)), "overlapping"),
        (("SWAP", "SWAP"), ((2, 3),), "out of range"),
        (("SWAP", "SWAP", "SWAP"), ((1, 3),), "adjacent"),
    ],
)
def test_pair_schedule_rejects_malformed_non_swap_overlap_and_range(
    program: tuple[str, ...], pairs: tuple[tuple[int, int], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pair.validate_pair_schedule(program, pairs)


def test_sham_is_exact_noop_and_pair_carry_restores_only_future_input() -> None:
    baseline_model, baseline_adapter, x_bits, y_bits = _fixture()
    baseline, baseline_diagnostics = latent.latent_slots_forward(
        baseline_model, baseline_adapter, x_bits, y_bits, TOY_OPS, return_diagnostics=True
    )

    sham_model, sham_adapter, sham_x, sham_y = _fixture()
    sham, sham_diagnostics = pair.pair_carry_forward(
        sham_model,
        sham_adapter,
        sham_x,
        sham_y,
        TOY_OPS,
        program=TOY_PROGRAM,
        pairs=TOY_PAIRS,
        arm="sham",
        return_diagnostics=True,
    )
    assert torch.equal(sham[0], baseline[0]) and torch.equal(sham[1], baseline[1])
    assert sham_diagnostics["pair_carry"]["arm"] == "sham"

    carry_model, carry_adapter, carry_x, carry_y = _fixture()
    carry, carry_diagnostics = pair.pair_carry_forward(
        carry_model,
        carry_adapter,
        carry_x,
        carry_y,
        TOY_OPS,
        program=TOY_PROGRAM,
        pairs=TOY_PAIRS,
        arm="pair_carry",
        return_diagnostics=True,
    )
    assert carry_diagnostics["pair_carry"]["arm"] == "pair_carry"
    assert torch.isfinite(carry[0]).all() and torch.isfinite(carry[1]).all()
    # The first SWAP readout is ordinary and visibly nonidentity.
    first_entry = carry_diagnostics["slot_inputs"][1]
    assert torch.equal(carry[0][:, 1], first_entry[:, 1, :])
    assert not torch.equal(carry[0][:, 1], first_entry[:, 0, :])
    # Both ordinary readouts inside the first pair are unchanged; restore is
    # applied only after the second instruction has emitted its logits.
    assert torch.equal(carry[0][:, 1:3], sham[0][:, 1:3])
    assert torch.equal(carry[1][:, 1:3], sham[1][:, 1:3])
    # The pair-end writer is replaced with the pair-entry clone, and only the
    # next instruction consumes that restored value.
    assert torch.equal(carry_diagnostics["slot_writes"][2], first_entry)
    assert torch.equal(carry_diagnostics["slot_inputs"][3], first_entry)
    assert not torch.equal(sham_diagnostics["slot_inputs"][3], first_entry)
    second_entry = carry_diagnostics["slot_inputs"][4]
    assert torch.equal(carry_diagnostics["slot_writes"][5], second_entry)
    assert torch.equal(carry_diagnostics["slot_inputs"][6], second_entry)
    assert all(
        any(float(value) > 0.0 for value in row["l2_by_case"])
        for row in carry_diagnostics["pair_carry"]["pairs"]
    )
    assert carry_diagnostics["initializer_calls"] == 1
    assert carry_diagnostics["detached_writes"] is False
    assert "target" not in inspect.signature(pair.pair_carry_forward).parameters
    assert baseline_diagnostics["native_steps"] == sham_diagnostics["native_steps"] == carry_diagnostics["native_steps"]


def test_raw_nonfinite_writer_is_rejected_even_when_pair_restore_would_hide_it() -> None:
    model, adapter, x_bits, y_bits = _fixture()
    original = adapter.writer.forward

    def nonfinite_writer(hidden: Tensor) -> Tensor:
        output = original(hidden)
        output = output.clone()
        output[:, 0] = float("nan")
        return output

    adapter.writer.forward = nonfinite_writer  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="writer output is nonfinite"):
        pair.pair_carry_forward(
            model, adapter, x_bits, y_bits, TOY_OPS,
            program=TOY_PROGRAM, pairs=TOY_PAIRS, arm="pair_carry",
        )


def test_hooks_are_removed_after_success_and_forward_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    model, adapter, x_bits, y_bits = _fixture()
    pair.pair_carry_forward(
        model, adapter, x_bits, y_bits, TOY_OPS,
        program=TOY_PROGRAM, pairs=TOY_PAIRS, arm="sham",
    )
    assert len(adapter.reader._forward_pre_hooks) == 0
    assert len(adapter.writer._forward_hooks) == 0

    failing_model, failing_adapter, failing_x, failing_y = _fixture()

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("fixture forward failure")

    monkeypatch.setattr(pair.latent, "latent_slots_forward", fail)
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        pair.pair_carry_forward(
            failing_model, failing_adapter, failing_x, failing_y, TOY_OPS,
            program=TOY_PROGRAM, pairs=TOY_PAIRS, arm="pair_carry",
        )
    assert len(failing_adapter.reader._forward_pre_hooks) == 0
    assert len(failing_adapter.writer._forward_hooks) == 0


def test_hand_paired_metrics_and_fixed_protocol_budgets() -> None:
    target = [[0, 0], [1, 1]]
    states = ((0, 0), (1, 1), (2, 2), (3, 3))

    def rows(predictions: list[list[list[int]]]) -> list[dict[str, object]]:
        return [{
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
        }]

    candidate = rows([
        [[0, 1], [1, 1]],  # repair: candidate correct, reference wrong
        [[0, 1], [0, 0]],  # regression: candidate wrong, reference correct
        [[0, 0], [1, 1]],  # tie: both correct
        [[0, 1], [0, 0]],  # tie: both wrong
    ])
    reference = rows([
        [[0, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]],
    ])
    metrics = accepted_paired_metrics(candidate, reference, label="hand")
    assert metrics["aggregate"]["final"] == {
        "left_correct_right_wrong": 1,
        "left_wrong_right_correct": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }
    assert metrics["improvements"] == {"final": 1, "full_trace": 0}
    assert metrics["regressions"] == {"final": 1, "full_trace": 1}
    assert metrics["ties"] == {"final": 2, "full_trace": 3}
    assert pair.QA_ACCOUNTING == {
        "forwards": 3, "cases": 6, "positions": 30,
        "native_steps": 240, "backwards": 0, "optimizer_updates": 0,
    }
    assert pair.SCIENCE_ACCOUNTING == {
        "forwards": 36, "cases": 9216, "positions": 258048,
        "native_steps": 2064384, "backwards": 0, "optimizer_updates": 0,
    }
