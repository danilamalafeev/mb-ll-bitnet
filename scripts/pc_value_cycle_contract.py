"""Pure writer-to-value transition contract for identity cycles.

The transition diagnostic found that projected keys are fixed by the role
table while projected values vary with the carried slots.  This module adds no
new writer, cache, or state bypass.  It supplies a small auxiliary objective
that compares per-slot projected ``V`` before and after a semantic identity
cycle.  Ordinary operations still use the existing writer update; the
objective is active only when a registered identity cycle occurs in a traced
program.

There is no checkpoint loading, model construction, CUDA setup, or training
loop here.  The helpers accept an already constructed model/adapter so pure
tests can exercise the exact reader projection and gradient path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import Tensor

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent


VALUE_CYCLE_ARCHITECTURE_ID = "latent_slots_2x16_projected_v_cycle_contract_v1"
VALUE_CYCLE_LOSS_WEIGHT = 0.1
VALUE_CYCLE_EPS = 1.0e-6

# These are exact DSL identities.  The mixed cycle is included because the
# transfer audit showed that different operation realizations can end in
# different latent states even when the semantic state agrees.
IDENTITY_CYCLES = (
    ("SWAP2", ("SWAP", "SWAP")),
    ("XOR2", ("XOR", "XOR")),
    ("SWAP_XOR2_SWAP", ("SWAP", "XOR", "XOR", "SWAP")),
)
IDENTITY_CYCLE_IDS = tuple(
    (name, tuple(dsl.OP_TO_ID[opcode] for opcode in program))
    for name, program in IDENTITY_CYCLES
)


def _finite_float(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
        raise ValueError(f"{label} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _same_shape(left: Tensor, right: Tensor, *, label: str) -> tuple[Tensor, Tensor]:
    left = _finite_float(left, label=f"{label}.before")
    right = _finite_float(right, label=f"{label}.after")
    if left.shape != right.shape or left.device != right.device:
        raise ValueError(f"{label} tensors must have matching shape and device")
    return left, right


def _validate_projected_values(values: Tensor, *, label: str) -> Tensor:
    values = _finite_float(values, label=label)
    if values.ndim != 4 or values.shape[2] != latent.SLOT_COUNT or values.shape[3] < 1:
        raise ValueError(f"{label} must have shape [batch, heads, 2, head_dim]")
    return values


def projected_values_from_slots(
    model: Any,
    adapter: latent.LatentSlotAdapter,
    slots: Tensor,
) -> Tensor:
    """Build the exact projected V tensor used by ``fresh_cache_from_slots``.

    The role-key path is intentionally absent.  This helper follows only
    ``slots -> reader -> memory_norm -> reader.v_proj`` and reshapes the result
    with the native reader geometry, so the transition loss targets the state
    component identified by the diagnostic.
    """

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    device = latent._validate_adapter_device(adapter)
    latent._validate_slots(slots, device)
    memory_norm = getattr(model, "memory_norm", None)
    reader = getattr(model, "reader", None)
    v_proj = getattr(reader, "v_proj", None) if reader is not None else None
    if not callable(memory_norm) or not callable(v_proj):
        raise ValueError("model lacks memory_norm or reader.v_proj")
    heads = getattr(reader, "num_heads", None)
    head_dim = getattr(reader, "head_dim", None)
    if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0:
        raise ValueError("reader.num_heads must be a positive integer")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("reader.head_dim must be a positive integer")
    if heads * head_dim != latent.HIDDEN_WIDTH:
        raise ValueError("reader head geometry changed")
    values = adapter.reader(slots)
    value_memory = memory_norm(values)
    projected = v_proj(value_memory)
    if not isinstance(projected, Tensor) or projected.shape != (slots.shape[0], latent.SLOT_COUNT, latent.HIDDEN_WIDTH):
        raise ValueError("reader.v_proj returned an unexpected value shape")
    projected = projected.view(slots.shape[0], latent.SLOT_COUNT, heads, head_dim).transpose(1, 2)
    return _validate_projected_values(projected, label="projected_values")


def normalized_projected_value_loss(
    before: Tensor,
    after: Tensor,
    *,
    eps: float = VALUE_CYCLE_EPS,
) -> Tensor:
    """Return a directional, scale-normalized projected-V compatibility loss.

    ``before`` is a detached target.  This keeps the objective from pulling
    every earlier state toward a single vector while still backpropagating
    through the post-cycle transition.  The denominator is per example and
    uses the target energy, making the coefficient stable across states.
    """

    before, after = _same_shape(before, after, label="projected_value_cycle")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a positive finite number")
    if before.ndim < 2:
        raise ValueError("projected value tensors must include a batch dimension")
    reduce_dims = tuple(range(1, before.ndim))
    target = before.detach().float()
    prediction = after.float()
    delta_energy = (prediction - target).square().mean(dim=reduce_dims)
    target_energy = target.square().mean(dim=reduce_dims).clamp_min(float(eps))
    loss = (delta_energy / target_energy).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("projected value cycle loss is nonfinite")
    return loss


def _validate_ops(ops: Tensor) -> Tensor:
    if not isinstance(ops, Tensor) or ops.ndim != 2 or ops.dtype is not torch.long:
        raise ValueError("ops must be a [batch, length] torch.long tensor")
    if ops.shape[0] < 1 or ops.shape[1] < 1 or torch.any((ops < 0) | (ops >= len(dsl.OPS))):
        raise ValueError("ops contains an invalid opcode")
    return ops


def find_identity_cycle_windows(ops: Tensor) -> list[dict[str, Any]]:
    """Find disjoint registered identity cycles in each program row.

    The longest cycle is preferred at a position, then the scan resumes after
    the match.  Thus ``SWAP`` repeated four times contributes two ``SWAP2``
    contracts instead of overlapping duplicate windows.
    """

    ops = _validate_ops(ops)
    cycles = tuple(sorted(IDENTITY_CYCLE_IDS, key=lambda item: len(item[1]), reverse=True))
    windows: list[dict[str, Any]] = []
    for batch_index in range(ops.shape[0]):
        start = 0
        length = int(ops.shape[1])
        while start < length:
            matched: tuple[str, tuple[int, ...]] | None = None
            for name, cycle_ids in cycles:
                end = start + len(cycle_ids)
                if end <= length and tuple(int(value) for value in ops[batch_index, start:end].tolist()) == cycle_ids:
                    matched = (name, cycle_ids)
                    break
            if matched is None:
                start += 1
                continue
            name, cycle_ids = matched
            windows.append({
                "batch_index": batch_index,
                "start": start,
                "length": len(cycle_ids),
                "cycle_id": name,
                "opcode_ids": list(cycle_ids),
            })
            start += len(cycle_ids)
    return windows


def _diagnostic_slots(diagnostics: Mapping[str, Any], ops: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    if not isinstance(diagnostics, Mapping):
        raise ValueError("latent diagnostics must be a mapping")
    inputs = diagnostics.get("slot_inputs")
    writes = diagnostics.get("slot_writes")
    if not isinstance(inputs, list) or not isinstance(writes, list) or len(inputs) != ops.shape[1] or len(writes) != ops.shape[1]:
        raise ValueError("latent diagnostics slot lists do not match ops length")
    if not inputs or any(not isinstance(value, Tensor) for value in [*inputs, *writes]):
        raise ValueError("latent diagnostics slot lists are malformed")
    first = inputs[0]
    if not isinstance(first, Tensor) or first.ndim != 3 or tuple(first.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH):
        raise ValueError("latent diagnostics slots have an unexpected shape")
    for label, values in (("slot_inputs", inputs), ("slot_writes", writes)):
        for value in values:
            if value.shape != first.shape or value.device != first.device or value.dtype is not torch.float32:
                raise ValueError(f"latent diagnostics {label} are inconsistent")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"latent diagnostics {label} contain nonfinite values")
    return inputs, writes


def projected_value_cycle_loss(
    model: Any,
    adapter: latent.LatentSlotAdapter,
    diagnostics: Mapping[str, Any],
    ops: Tensor,
    *,
    weight: float = VALUE_CYCLE_LOSS_WEIGHT,
    eps: float = VALUE_CYCLE_EPS,
) -> tuple[Tensor, dict[str, Any]]:
    """Compute the weighted cycle contract from one traced latent forward.

    ``diagnostics`` must come from ``latent_slots_forward(...,
    return_diagnostics=True)``.  The helper performs no native model step; it
    only reuses the saved slot tensors to form projected V targets.
    """

    ops = _validate_ops(ops)
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) < 0.0:
        raise ValueError("weight must be a nonnegative finite number")
    inputs, writes = _diagnostic_slots(diagnostics, ops)
    windows = find_identity_cycle_windows(ops)
    by_cycle = Counter(str(window["cycle_id"]) for window in windows)
    if not windows:
        zero = inputs[0].sum() * 0.0
        return zero, {
            "schema": "pc_projected_value_cycle_loss_v1",
            "windows": 0,
            "by_cycle": {},
            "raw_loss": 0.0,
            "weighted_loss": 0.0,
            "weight": float(weight),
            "target_detached": True,
            "projected_state": "per_slot_projected_V",
            "model_forwards": 0,
        }

    before_rows = [inputs[int(window["start"])][int(window["batch_index"])] for window in windows]
    after_rows = [writes[int(window["start"]) + int(window["length"]) - 1][int(window["batch_index"])] for window in windows]
    before = torch.stack(before_rows, dim=0)
    after = torch.stack(after_rows, dim=0)
    projected_before = projected_values_from_slots(model, adapter, before)
    projected_after = projected_values_from_slots(model, adapter, after)
    raw_loss = normalized_projected_value_loss(projected_before, projected_after, eps=eps)
    weighted = raw_loss * float(weight)
    if not bool(torch.isfinite(weighted).item()):
        raise FloatingPointError("weighted projected value cycle loss is nonfinite")
    return weighted, {
        "schema": "pc_projected_value_cycle_loss_v1",
        "windows": len(windows),
        "by_cycle": dict(sorted(by_cycle.items())),
        "raw_loss": float(raw_loss.detach().item()),
        "weighted_loss": float(weighted.detach().item()),
        "weight": float(weight),
        "target_detached": True,
        "projected_state": "per_slot_projected_V",
        "model_forwards": 0,
    }


__all__ = [
    "IDENTITY_CYCLES",
    "IDENTITY_CYCLE_IDS",
    "VALUE_CYCLE_ARCHITECTURE_ID",
    "VALUE_CYCLE_EPS",
    "VALUE_CYCLE_LOSS_WEIGHT",
    "find_identity_cycle_windows",
    "normalized_projected_value_loss",
    "projected_values_from_slots",
    "projected_value_cycle_loss",
]
