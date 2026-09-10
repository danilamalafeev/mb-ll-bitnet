"""Pure pair-carry hooks for the registered latent padding intervention.

The intervention is deliberately a thin wrapper around the accepted latent
forward.  A reader pre-hook clones the slot tensor at each registered pair
start.  A writer forward hook leaves the ordinary writer output alone for the
sham arm, and replaces only the carried slot tensor at a registered pair end
for the pair-carry arm.  The current instruction's logits are therefore
always produced by the ordinary model step.

This module does not load a checkpoint, construct a model, use CUDA, or run a
science/QA program.  It is intended to be imported by the later disposable
runner after the focused CPU fixtures pass.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from scripts import pc_latent_slots as latent


PAIR_CARRY_ARM = "pair_carry"
SHAM_ARM = "sham"
_ARM_ALIASES = {SHAM_ARM: SHAM_ARM, PAIR_CARRY_ARM: PAIR_CARRY_ARM, "restore": PAIR_CARRY_ARM}
FOCUS_LENGTHS = (24, 32)

# Fixed protocol budgets.  These are accounting fixtures for the later runner;
# this pure module never executes either budget.
QA_ACCOUNTING = {
    "forwards": 3,
    "cases": 6,
    "positions": 30,
    "native_steps": 240,
    "backwards": 0,
    "optimizer_updates": 0,
}
SCIENCE_ACCOUNTING = {
    "forwards": 36,
    "cases": 9216,
    "positions": 258048,
    "native_steps": 2064384,
    "backwards": 0,
    "optimizer_updates": 0,
}


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _program(program: Sequence[str]) -> tuple[str, ...]:
    values = _sequence(program, label="program")
    if not values or any(not isinstance(opcode, str) or not opcode for opcode in values):
        raise ValueError("program must contain nonempty opcode strings")
    return tuple(values)


def _position(value: Any, *, label: str) -> int:
    # bool is an int subclass but cannot be a meaningful one-based position.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def normalize_arm(arm: str) -> str:
    """Return the canonical arm name used in diagnostics and output rows."""

    if not isinstance(arm, str) or arm not in _ARM_ALIASES:
        raise ValueError("arm must be 'sham' or 'pair_carry'")
    return _ARM_ALIASES[arm]


def validate_pair_schedule(
    program: Sequence[str],
    pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Validate one-based adjacent SWAP pairs and return an immutable schedule.

    Pair order is part of the registration contract because hooks observe the
    model in forward order.  The checks intentionally do not infer identity
    from labels or permit a pair to contain a non-SWAP instruction.
    """

    opcodes = _program(program)
    raw_pairs = _sequence(pairs, label="pairs")
    normalized: list[tuple[int, int]] = []
    used: set[int] = set()
    previous_start = 0
    for index, raw_pair in enumerate(raw_pairs):
        values = _sequence(raw_pair, label=f"pair {index}")
        if len(values) != 2:
            raise ValueError(f"pair {index} must contain exactly two positions")
        start = _position(values[0], label=f"pair {index} start")
        end = _position(values[1], label=f"pair {index} end")
        if start < 1 or end < 1 or end > len(opcodes):
            raise ValueError(f"pair {index} is out of range")
        if end != start + 1:
            raise ValueError(f"pair {index} must contain adjacent positions")
        if start <= previous_start:
            raise ValueError("pair schedule must be in strictly increasing order")
        if start in used or end in used:
            raise ValueError("pair schedule contains overlapping pairs")
        if opcodes[start - 1] != "SWAP" or opcodes[end - 1] != "SWAP":
            raise ValueError("registered pair positions must both be SWAP")
        used.update((start, end))
        previous_start = start
        normalized.append((start, end))
    return tuple(normalized)


def expected_pair_schedule(length_or_program: int | Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return the frozen focus schedule for an L24 or L32 program.

    The schedule covers positions 3--22 at L24 or 3--30 at L32 in adjacent
    pairs.  Position 23/31 remains the odd SWAP tail and the final useful
    instruction remains outside the intervention.
    """

    if isinstance(length_or_program, bool):
        raise ValueError("focus program length must be 24 or 32")
    if isinstance(length_or_program, int):
        length = length_or_program
        opcodes: tuple[str, ...] | None = None
    else:
        opcodes = _program(length_or_program)
        length = len(opcodes)
    if length not in FOCUS_LENGTHS:
        raise ValueError("focus program length must be 24 or 32")
    pairs = tuple((start, start + 1) for start in range(3, length - 1, 2))
    if opcodes is not None:
        validate_pair_schedule(opcodes, pairs)
        if opcodes[-2] != "SWAP":
            raise ValueError("focus program must preserve the odd SWAP tail")
    return pairs


# A descriptive alias makes the frozen schedule name explicit to callers.
pair_schedule_for_program = expected_pair_schedule


@dataclass
class PairCarryHookState:
    """Ephemeral state owned by one scoped hook installation."""

    pairs: tuple[tuple[int, int], ...]
    arm: str
    reader_calls: int = 0
    writer_calls: int = 0
    _entries: dict[int, Tensor] = field(default_factory=dict)
    _displacements: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.arm = normalize_arm(self.arm)
        self._end_to_start = {end: start for start, end in self.pairs}

    def _reader_pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        position = self.reader_calls + 1
        self.reader_calls = position
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("latent reader hook received malformed input")
        if position in {start for start, _end in self.pairs}:
            entry = args[0]
            if entry.ndim != 3 or tuple(entry.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH):
                raise ValueError("pair-entry z has malformed slot shape")
            if not torch.isfinite(entry).all():
                raise ValueError("pair-entry z is nonfinite")
            self._entries[position] = entry.clone()

    def _writer_hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        position = self.writer_calls + 1
        self.writer_calls = position
        if not isinstance(output, Tensor) or output.ndim < 1:
            raise ValueError("latent writer hook received malformed output")
        # Validate the ordinary writer result before a restore can discard it;
        # otherwise a pair-end NaN could be hidden by the replacement tensor.
        if not torch.isfinite(output).all():
            raise ValueError("latent writer output is nonfinite")
        start = self._end_to_start.get(position)
        if start is None:
            return output
        entry = self._entries.pop(start, None)
        if entry is None:
            raise RuntimeError(f"pair-entry z was not captured for pair ending at position {position}")
        if (
            output.numel() != entry.numel()
            or output.shape[0] != entry.shape[0]
            or output.device != entry.device
            or output.dtype != entry.dtype
        ):
            raise ValueError("latent writer output cannot be reshaped to pair-entry z")
        entry_as_output = entry.reshape_as(output)
        displacement = torch.linalg.vector_norm(
            output.reshape_as(entry) - entry,
            dim=(-2, -1),
        )
        self._displacements.append(
            {
                "start": start,
                "end": position,
                "l2_by_case": displacement.detach().cpu().tolist(),
            }
        )
        if self.arm == PAIR_CARRY_ARM:
            # Return a fresh view so no downstream operation can mutate the
            # captured entry tensor held by the state before diagnostics end.
            return entry_as_output.clone()
        return output

    def finish(self, expected_calls: int) -> None:
        if self.reader_calls != expected_calls or self.writer_calls != expected_calls:
            raise ValueError(
                "pair-carry hook call count changed: "
                f"reader={self.reader_calls}, writer={self.writer_calls}, expected={expected_calls}"
            )
        if self._entries:
            raise ValueError("pair-carry schedule did not reach every registered pair end")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "pairs": [dict(row) for row in self._displacements],
            "reader_calls": self.reader_calls,
            "writer_calls": self.writer_calls,
        }


@contextmanager
def pair_carry_hooks(
    adapter: latent.LatentSlotAdapter,
    program: Sequence[str],
    pairs: Sequence[Sequence[int]],
    *,
    arm: str = SHAM_ARM,
) -> Iterator[PairCarryHookState]:
    """Install reader/writer hooks for one validated program and remove both.

    Cleanup runs when the wrapped forward succeeds, raises, or hook
    registration itself fails after the first handle has been installed.
    """

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    opcodes = _program(program)
    schedule = validate_pair_schedule(opcodes, pairs)
    state = PairCarryHookState(schedule, normalize_arm(arm))
    handles: list[Any] = []
    try:
        handles.append(adapter.reader.register_forward_pre_hook(state._reader_pre_hook))
        handles.append(adapter.writer.register_forward_hook(state._writer_hook))
        yield state
    finally:
        for handle in reversed(handles):
            handle.remove()


def pair_carry_forward(
    model: Any,
    adapter: latent.LatentSlotAdapter,
    x_bits: Tensor,
    y_bits: Tensor,
    ops: Any,
    *,
    program: Sequence[str],
    pairs: Sequence[Sequence[int]] | None = None,
    arm: str = SHAM_ARM,
    return_diagnostics: bool = False,
) -> Any:
    """Run accepted latent forward with one sham or pair-carry intervention.

    ``program`` is used solely to validate the external registration.  The
    model still receives ``ops`` and executes every ordinary instruction.  No
    target labels are accepted by this interface.
    """

    opcodes = _program(program)
    schedule = expected_pair_schedule(opcodes) if pairs is None else validate_pair_schedule(opcodes, pairs)
    canonical_arm = normalize_arm(arm)
    with pair_carry_hooks(adapter, opcodes, schedule, arm=canonical_arm) as state:
        result, accepted_diagnostics = latent.latent_slots_forward(
            model,
            adapter,
            x_bits,
            y_bits,
            ops,
            return_diagnostics=True,
        )
        state.finish(len(opcodes))
    accepted_diagnostics["pair_carry"] = state.diagnostics()
    if return_diagnostics:
        return result, accepted_diagnostics
    return result


__all__ = [
    "FOCUS_LENGTHS",
    "PAIR_CARRY_ARM",
    "QA_ACCOUNTING",
    "SCIENCE_ACCOUNTING",
    "SHAM_ARM",
    "PairCarryHookState",
    "expected_pair_schedule",
    "normalize_arm",
    "pair_carry_forward",
    "pair_carry_hooks",
    "pair_schedule_for_program",
    "validate_pair_schedule",
]
