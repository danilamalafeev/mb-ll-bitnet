"""Pure contract and reductions for the writer-to-cache diagnostic.

The diagnostic compares one known-good and one known-bad padding trajectory at
the accepted local-2000 endpoint.  It is inference-only: this module does not
load checkpoints, construct a model, or run a training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import Tensor


SLOT_COUNT = 2
SLOT_WIDTH = 16
HIDDEN_WIDTH = 128
NATIVE_STEPS = 8

PREFIX = ("ADD", "ADD")
FINAL_OPCODE = "ADD"
GOOD_ID = "padding_ADDADD_ADD_k4"
BAD_ID = "padding_ADDADD_ADD_k10"
GOOD_PROGRAM = PREFIX + ("SWAP",) * 9 + (FINAL_OPCODE,)
BAD_PROGRAM = PREFIX + ("SWAP",) * 21 + (FINAL_OPCODE,)
PAIR_STATES = ((6, 4), (8, 14))
SUBSTITUTION_LABELS = (
    "bad_h_bad_kv",
    "good_h_good_kv",
    "good_h_bad_kv",
    "bad_h_good_kv",
)
DIAGNOSTIC_BUDGET = {
    "path_forwards": 2,
    "path_cases": 4,
    "path_positions": len(PAIR_STATES) * (len(GOOD_PROGRAM) + len(BAD_PROGRAM)),
    "path_native_steps": len(PAIR_STATES) * (len(GOOD_PROGRAM) + len(BAD_PROGRAM)) * NATIVE_STEPS,
    "substitution_forwards": 1,
    "substitution_cases": len(PAIR_STATES) * len(SUBSTITUTION_LABELS),
    "substitution_positions": len(PAIR_STATES) * len(SUBSTITUTION_LABELS),
    "substitution_native_steps": len(PAIR_STATES) * len(SUBSTITUTION_LABELS) * NATIVE_STEPS,
    "optimizer_updates": 0,
}


def _finite_tensor(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
        raise ValueError(f"{label} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _same_shape(left: Tensor, right: Tensor, *, label: str) -> tuple[Tensor, Tensor]:
    left = _finite_tensor(left, label=f"{label}.left")
    right = _finite_tensor(right, label=f"{label}.right")
    if left.shape != right.shape or left.device != right.device:
        raise ValueError(f"{label} tensors must have matching shape and device")
    return left, right


def _reduce(values: Tensor) -> dict[str, float]:
    flattened = values.reshape(values.shape[0], -1).float()
    return {
        "mean": float(flattened.mean().item()),
        "max": float(flattened.max().item()),
        "min": float(flattened.min().item()),
    }


def _norm_summary(values: Tensor) -> dict[str, float]:
    flattened = values.reshape(values.shape[0], -1).float()
    norms = torch.linalg.vector_norm(flattened, dim=-1)
    return _reduce(norms)


def pair_distance(left: Tensor, right: Tensor, *, label: str = "pair") -> dict[str, Any]:
    """Reduce a batch of paired representations without retaining raw tensors."""

    left, right = _same_shape(left, right, label=label)
    delta = right.float() - left.float()
    left_flat = left.float().reshape(left.shape[0], -1)
    right_flat = right.float().reshape(right.shape[0], -1)
    delta_flat = delta.reshape(delta.shape[0], -1)
    left_norm = torch.linalg.vector_norm(left_flat, dim=-1)
    right_norm = torch.linalg.vector_norm(right_flat, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta_flat, dim=-1)
    denominator = left_norm.clamp_min(torch.finfo(left_norm.dtype).eps)
    cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=-1, eps=1e-12)
    return {
        "shape": list(left.shape),
        "delta": _reduce(delta_norm),
        "relative_delta": _reduce(delta_norm / denominator),
        "left_norm": _reduce(left_norm),
        "right_norm": _reduce(right_norm),
        "cosine": _reduce(cosine),
        "all_finite": bool(torch.isfinite(delta).all().item()),
    }


def centered_logit_distance(left: Tensor, right: Tensor, *, label: str = "logits") -> dict[str, Any]:
    """Compare output-head differences after removing each row's common shift."""

    left, right = _same_shape(left, right, label=label)
    if left.ndim < 2 or left.shape[-1] < 2:
        raise ValueError(f"{label} must have a class dimension")
    left_centered = left.float() - left.float().mean(dim=-1, keepdim=True)
    right_centered = right.float() - right.float().mean(dim=-1, keepdim=True)
    return pair_distance(left_centered, right_centered, label=f"{label}.centered")


def attention_summary(weights: Tensor, *, label: str = "attention_weights") -> dict[str, Any]:
    """Summarize [native-step, batch, head, 2] role weights."""

    weights = _finite_tensor(weights, label=label)
    if weights.ndim != 4 or weights.shape[-1] != SLOT_COUNT:
        raise ValueError(f"{label} must have shape [steps, batch, heads, 2]")
    if not bool(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5, rtol=0.0)):
        raise ValueError(f"{label} rows must sum to one")
    distance_from_half = (weights[..., 0] - 0.5).abs()
    first_role = weights[..., 0]
    return {
        "shape": list(weights.shape),
        "first_role": _reduce(first_role),
        "distance_from_half": _reduce(distance_from_half),
        "near_uniform_fraction": float((distance_from_half <= 0.05).float().mean().item()),
        "near_decisive_fraction": float((distance_from_half >= 0.45).float().mean().item()),
        "all_finite": bool(torch.isfinite(weights).all().item()),
    }


def projection_summary(phi_good: Tensor, phi_bad: Tensor, *, x_head: Tensor, y_head: Tensor, writer_weight: Tensor, writer_bias: Tensor | None = None) -> dict[str, Any]:
    """Compare head and writer sensitivity to one good/bad ``phi`` direction."""

    phi_good, phi_bad = _same_shape(phi_good, phi_bad, label="phi")
    if phi_good.ndim != 2 or phi_good.shape[-1] != HIDDEN_WIDTH:
        raise ValueError("phi must have shape [batch, 128]")
    for label, value, shape in (
        ("x_head", x_head, (16, HIDDEN_WIDTH)),
        ("y_head", y_head, (16, HIDDEN_WIDTH)),
        ("writer_weight", writer_weight, (SLOT_COUNT * SLOT_WIDTH, HIDDEN_WIDTH)),
    ):
        value = _finite_tensor(value, label=label)
        if tuple(value.shape) != shape:
            raise ValueError(f"{label} has an unexpected shape")
    if writer_bias is not None:
        writer_bias = _finite_tensor(writer_bias, label="writer_bias")
        if tuple(writer_bias.shape) != (SLOT_COUNT * SLOT_WIDTH,):
            raise ValueError("writer_bias has an unexpected shape")
    delta_phi = phi_bad - phi_good
    x_delta = delta_phi @ x_head.float().T
    y_delta = delta_phi @ y_head.float().T
    writer_delta = delta_phi @ writer_weight.float().T
    x_centered = x_delta - x_delta.mean(dim=-1, keepdim=True)
    y_centered = y_delta - y_delta.mean(dim=-1, keepdim=True)
    return {
        "phi": pair_distance(phi_good, phi_bad, label="phi"),
        "x_head": {"centered_delta_norm": _norm_summary(x_centered), "max_abs_centered_delta": float(x_centered.abs().max().item())},
        "y_head": {"centered_delta_norm": _norm_summary(y_centered), "max_abs_centered_delta": float(y_centered.abs().max().item())},
        "writer": {"delta_norm": _norm_summary(writer_delta), "max_abs_delta": float(writer_delta.abs().max().item())},
        "writer_bias_cancels_in_difference": writer_bias is not None,
    }


def validate_pair_metadata(good: Mapping[str, Any], bad: Mapping[str, Any]) -> None:
    """Require the frozen good/bad pair to share the final operation and state."""

    if str(good.get("id")) != GOOD_ID or str(bad.get("id")) != BAD_ID:
        raise ValueError("diagnostic pair IDs changed")
    if tuple(good.get("program", ())) != GOOD_PROGRAM or tuple(bad.get("program", ())) != BAD_PROGRAM:
        raise ValueError("diagnostic pair programs changed")
    if tuple(good.get("final_semantic_state", ())) != tuple(bad.get("final_semantic_state", ())):
        raise ValueError("diagnostic pair semantic states differ")
    if str(good.get("final_opcode")) != FINAL_OPCODE or str(bad.get("final_opcode")) != FINAL_OPCODE:
        raise ValueError("diagnostic pair final opcode changed")


def validate_substitution_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate one outcome for every state/combination without interpreting it."""

    expected = {(tuple(state), label) for state in PAIR_STATES for label in SUBSTITUTION_LABELS}
    actual: set[tuple[tuple[int, int], str]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in outcomes:
        if not isinstance(raw, Mapping):
            raise ValueError("substitution outcome is malformed")
        state = tuple(int(value) for value in raw.get("state", ()))
        label = str(raw.get("label", ""))
        if state not in PAIR_STATES or label not in SUBSTITUTION_LABELS or (state, label) in actual:
            raise ValueError("substitution outcome identity changed")
        actual.add((state, label))
        for key in ("predicted", "target", "correct", "final_logits_finite"):
            if key not in raw:
                raise ValueError(f"substitution outcome missing {key}")
        normalized.append(dict(raw))
    if actual != expected:
        raise ValueError("substitution outcome coverage changed")
    return normalized


__all__ = [
    "BAD_ID", "BAD_PROGRAM", "DIAGNOSTIC_BUDGET", "FINAL_OPCODE", "GOOD_ID", "GOOD_PROGRAM",
    "HIDDEN_WIDTH", "NATIVE_STEPS", "PAIR_STATES", "PREFIX", "SLOT_COUNT", "SLOT_WIDTH",
    "SUBSTITUTION_LABELS", "attention_summary", "centered_logit_distance", "pair_distance",
    "projection_summary", "validate_pair_metadata", "validate_substitution_outcomes",
]
