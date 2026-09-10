"""Pure learned-scratchpad interfaces and bounded accounting.

This module deliberately does not load checkpoints, construct a training
runner, or edit the existing model.  It supplies the signed-bit cache bridge,
the differentiable own-write forward, child identity checks, and exact
registered budget arithmetic for a later reviewed runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


REGISTER_VALUES = tuple(range(16))
REGISTER_BITS = 4
REGISTER_WIDTH = 16
NATIVE_STEPS = 8
PARENT_UPDATE = 40000
CHILD_UPDATES = 2000
BATCH_SIZE = 64
ARMS = ("continuous_control", "soft_register_reset")
ARCHITECTURE_ID = "width128_native8_signed_bits_scratchpad_v1"

FIXED_BATCH_LENGTHS = (1, 2, 3, 4, 5, 6) * 333 + (1, 2)
TRAINING_READOUT_POSITIONS = sum(FIXED_BATCH_LENGTHS) * BATCH_SIZE
TRAINING_CASES = len(FIXED_BATCH_LENGTHS) * BATCH_SIZE
TRAINING_NATIVE_STEPS = TRAINING_READOUT_POSITIONS * NATIVE_STEPS

TINY_QA_CALL_CASES = (2, 2, 2, 2, 2, 2)
TINY_QA_CALL_POSITIONS = (2, 4, 4, 2, 4, 4)
TINY_QA_CALLS = len(TINY_QA_CALL_CASES)
TINY_QA_CASES = sum(TINY_QA_CALL_CASES)
TINY_QA_READOUT_POSITIONS = sum(TINY_QA_CALL_POSITIONS)
TINY_QA_NATIVE_STEPS = TINY_QA_READOUT_POSITIONS * NATIVE_STEPS

EVALUATION_PROGRAM_LENGTHS = (4, 12, 16, 24, 32) * 9 + (12,) * 6 + (16,) * 6 + (24,) * 6 + (32,) * 6
EVALUATION_CALLS = len(EVALUATION_PROGRAM_LENGTHS)
EVALUATION_CASES = EVALUATION_CALLS * REGISTER_WIDTH * REGISTER_WIDTH
EVALUATION_READOUT_POSITIONS = sum(EVALUATION_PROGRAM_LENGTHS) * REGISTER_WIDTH * REGISTER_WIDTH
EVALUATION_NATIVE_STEPS = EVALUATION_READOUT_POSITIONS * NATIVE_STEPS

SCIENCE_ACCOUNTING = {
    "calls": 4207,
    "cases": 308992,
    "readout_positions": 1890816,
    "native_steps": 15126528,
    "optimizer_updates": 4000,
}
TOTAL_ACCOUNTING = {
    "calls": 4213,
    "cases": 309004,
    "readout_positions": 1890836,
    "native_steps": 15126688,
    "optimizer_updates": 4006,
}


def signed_bit_matrix(*, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the fixed 16-by-4 LSB-first signed-bit matrix."""

    values = torch.arange(REGISTER_WIDTH, dtype=torch.long, device=device).unsqueeze(1)
    shifts = torch.arange(REGISTER_BITS, dtype=torch.long, device=device).unsqueeze(0)
    return ((((values >> shifts) & 1) * 2) - 1).to(dtype=dtype)


def _probabilities(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != REGISTER_WIDTH:
        raise ValueError(f"{label} must have shape [batch, 16]")
    if value.dtype is not torch.float32:
        raise ValueError(f"{label} must be float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return value


def _projection(model: Any, name: str) -> Any:
    encoder = getattr(model, name, None)
    projection = getattr(encoder, "projection", None)
    weight = getattr(projection, "weight", None)
    if projection is None or weight is None or tuple(weight.shape)[1:] != (REGISTER_BITS,) or projection.bias is not None:
        raise ValueError(f"{name} is not the registered bias-free signed-bit projection")
    return projection


def cache_from_probabilities(model: Any, px: Tensor, py: Tensor) -> dict[str, Any]:
    """Build a fresh WidthRegisterModel-style cache from own probabilities."""

    px = _probabilities(px, label="px")
    py = _probabilities(py, label="py")
    if px.shape != py.shape or px.device != py.device:
        raise ValueError("px and py must have matching shape and device")
    x_projection = _projection(model, "x_embedding")
    y_projection = _projection(model, "y_embedding")
    if x_projection.weight.device != px.device or y_projection.weight.device != py.device:
        raise ValueError("probabilities and projections must share a device")
    matrix = signed_bit_matrix(device=px.device, dtype=px.dtype)
    expected_x_bits = px @ matrix
    expected_y_bits = py @ matrix
    xv = x_projection(expected_x_bits)
    yv = y_projection(expected_y_bits)
    batch = px.shape[0]
    role_keys = getattr(model, "role_keys", None)
    memory_norm = getattr(model, "memory_norm", None)
    reader = getattr(model, "reader", None)
    if role_keys is None or memory_norm is None or reader is None or not hasattr(reader, "build_kv"):
        raise ValueError("model lacks the registered memory reader")
    # Keep a fresh cache allocation so poisoning an old reader cache cannot
    # mutate the model's role-key parameter or a later cache.
    key_memory = memory_norm(role_keys).unsqueeze(0).expand(batch, -1, -1).clone()
    value_memory = memory_norm(torch.stack((xv, yv), dim=1))
    with torch.autocast(device_type=px.device.type, enabled=False):
        kv = reader.build_kv(key_memory, value_memory)
    return {"h": xv + yv, "x_initial": xv, "y_initial": yv, "kv": kv, "substeps": 0}


def _operation_ids(ops: Any, *, batch: int, device: torch.device) -> Tensor:
    if isinstance(ops, Tensor):
        result = ops
        if result.ndim != 2 or result.shape[0] != batch or result.dtype is not torch.long:
            raise ValueError("ops must have shape [batch, length] and torch.long dtype")
        if result.device != device:
            raise ValueError("ops and registers must share a device")
    elif isinstance(ops, Sequence) and not isinstance(ops, (str, bytes)):
        if len(ops) != batch:
            raise ValueError("ops must have one row per initial state")
        result = torch.tensor(ops, dtype=torch.long, device=device)
        if result.ndim != 2:
            raise ValueError("ops must be rectangular")
    else:
        raise ValueError("ops must be a [batch, length] tensor or nested integer sequence")
    if result.shape[1] < 1 or torch.any((result < 0) | (result >= 3)):
        raise ValueError("ops contains an invalid opcode")
    return result


def soft_register_forward(
    model: Any,
    x: Tensor,
    y: Tensor,
    ops: Any,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Run fresh-cache native steps while writing own soft probabilities."""

    if not isinstance(x, Tensor) or not isinstance(y, Tensor) or x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be matching rank-1 tensors")
    if x.device != y.device:
        raise ValueError("x and y must share a device")
    if x.dtype is not torch.long or y.dtype is not torch.long or torch.any((x < 0) | (x >= REGISTER_WIDTH)) or torch.any((y < 0) | (y >= REGISTER_WIDTH)):
        raise ValueError("x and y must be torch.long registers in 0..15")
    if getattr(model, "native_steps", NATIVE_STEPS) != NATIVE_STEPS:
        raise ValueError("model.step must retain native8 execution")
    op_ids = _operation_ids(ops, batch=x.shape[0], device=x.device)
    px = F.one_hot(x, num_classes=REGISTER_WIDTH).to(dtype=torch.float32)
    py = F.one_hot(y, num_classes=REGISTER_WIDTH).to(dtype=torch.float32)
    x_logits: list[Tensor] = []
    y_logits: list[Tensor] = []
    writes: list[dict[str, Tensor]] = []
    cache_inputs: list[dict[str, Any]] = []
    for position in range(op_ids.shape[1]):
        cache = cache_from_probabilities(model, px, py)
        if return_diagnostics:
            cache_inputs.append(
                {
                    "px": px,
                    "py": py,
                    "h": cache["h"].clone(),
                    "kv": cache["kv"],
                    "substeps": cache["substeps"],
                }
            )
        (logits_x, logits_y), _discarded_cache = model.step(cache, op_ids[:, position])
        if tuple(logits_x.shape) != (x.shape[0], REGISTER_WIDTH) or tuple(logits_y.shape) != (x.shape[0], REGISTER_WIDTH):
            raise ValueError("model.step logits must have shape [batch, 16]")
        x_logits.append(logits_x)
        y_logits.append(logits_y)
        px = F.softmax(logits_x, dim=-1)
        py = F.softmax(logits_y, dim=-1)
        if return_diagnostics:
            writes.append({"px": px, "py": py})
    output = (torch.stack(x_logits, dim=1), torch.stack(y_logits, dim=1))
    if return_diagnostics:
        return output, {"writes": writes, "cache_inputs": cache_inputs, "temperature": 1.0, "detached_writes": False}
    return output


def device_cross_entropy(logits_x: Tensor, logits_y: Tensor, targets_x: Tensor, targets_y: Tensor) -> Tensor:
    """Compute the accepted summed x/y CE without moving tensors to CPU."""

    tensors = (logits_x, logits_y, targets_x, targets_y)
    if any(not isinstance(value, Tensor) for value in tensors):
        raise ValueError("logits and targets must be tensors")
    if logits_x.ndim != 3 or logits_y.shape != logits_x.shape or targets_x.shape != logits_x.shape[:2] or targets_y.shape != targets_x.shape:
        raise ValueError("logits/targets have incompatible batch-position shapes")
    if logits_x.device != logits_y.device or targets_x.device != logits_x.device or targets_y.device != logits_x.device:
        raise ValueError("logits and targets must remain on one device")
    if logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32:
        raise ValueError("logits must remain float32")
    if targets_x.dtype is not torch.long or targets_y.dtype is not torch.long:
        raise ValueError("targets must be torch.long")
    with torch.autocast(device_type=logits_x.device.type, enabled=False):
        # Flatten B*T x 16 so CUDA deterministic mode uses the supported
        # 2-D cross-entropy kernel while preserving the summed x/y means.
        return F.cross_entropy(logits_x.reshape(-1, REGISTER_WIDTH), targets_x.reshape(-1)) + F.cross_entropy(
            logits_y.reshape(-1, REGISTER_WIDTH), targets_y.reshape(-1)
        )


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("identity value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _hex_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    return value.lower()


def child_identity(
    parent: Mapping[str, Any],
    *,
    arm: str,
    architecture_id: str = ARCHITECTURE_ID,
    local_update: int,
    stream_digest: str,
    target_digest: str,
    source_digest: str,
    config_digest: str,
) -> dict[str, Any]:
    """Create the fixed parent/arm/update identity for a disposable child."""

    if not isinstance(parent, Mapping):
        raise ValueError("parent identity must be a mapping")
    parent_keys = ("checkpoint_sha256", "model_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest")
    if any(key not in parent for key in parent_keys):
        raise ValueError("parent identity fields are incomplete")
    if arm not in ARMS or architecture_id != ARCHITECTURE_ID:
        raise ValueError("child architecture or arm is not registered")
    if type(local_update) is not int or local_update < 0 or local_update > CHILD_UPDATES:
        raise ValueError("local update is outside the registered child budget")
    parent_record = {key: _hex_digest(parent[key], label=f"parent {key}") for key in parent_keys}
    for label, value in (("stream", stream_digest), ("target", target_digest), ("source", source_digest), ("config", config_digest)):
        _hex_digest(value, label=label)
    return {
        "schema": "pc_learned_scratchpad_child_identity_v1",
        "parent": parent_record,
        "architecture_id": architecture_id,
        "arm": arm,
        "parent_update_absolute": PARENT_UPDATE,
        "local_update": local_update,
        "absolute_update": PARENT_UPDATE + local_update,
        "stream_digest": stream_digest.lower(),
        "target_digest": target_digest.lower(),
        "source_digest": source_digest.lower(),
        "config_digest": config_digest.lower(),
    }


def validate_child_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    """Reject parent, arm, update, configuration, or digest mutations."""

    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("child identity must be a mapping")
    if actual != expected:
        raise ValueError("child identity binding mismatch")
    return _digest(actual)


def _bounded_pair(attempted: int, completed: int, *, maximum: int, label: str) -> tuple[int, int]:
    if type(attempted) is not int or type(completed) is not int or attempted < 0 or completed < 0 or completed > attempted or attempted > maximum:
        raise ValueError(f"{label} accounting is outside the registered bound")
    return attempted, completed


def tiny_qa_accounting(attempted_calls: int, completed_calls: int) -> dict[str, int]:
    """Account the six-call disposable two-path QA, including partial prefixes."""

    attempted, completed = _bounded_pair(attempted_calls, completed_calls, maximum=TINY_QA_CALLS, label="tiny QA")
    return {
        "attempted_calls": attempted,
        "completed_calls": completed,
        "attempted_cases": sum(TINY_QA_CALL_CASES[:attempted]),
        "completed_cases": sum(TINY_QA_CALL_CASES[:completed]),
        "attempted_readout_positions": sum(TINY_QA_CALL_POSITIONS[:attempted]),
        "completed_readout_positions": sum(TINY_QA_CALL_POSITIONS[:completed]),
        "attempted_native_steps": sum(TINY_QA_CALL_POSITIONS[:attempted]) * NATIVE_STEPS,
        "completed_native_steps": sum(TINY_QA_CALL_POSITIONS[:completed]) * NATIVE_STEPS,
        "attempted_optimizer_updates": attempted,
        "completed_optimizer_updates": completed,
    }


def training_accounting(attempted_updates: int, completed_updates: int) -> dict[str, int]:
    """Account a prefix of the frozen 2000-batch continuation for one child."""

    attempted, completed = _bounded_pair(attempted_updates, completed_updates, maximum=CHILD_UPDATES, label="training")
    attempted_lengths = FIXED_BATCH_LENGTHS[:attempted]
    completed_lengths = FIXED_BATCH_LENGTHS[:completed]
    return {
        "attempted_optimizer_updates": attempted,
        "completed_optimizer_updates": completed,
        "attempted_cases": len(attempted_lengths) * BATCH_SIZE,
        "completed_cases": len(completed_lengths) * BATCH_SIZE,
        "attempted_readout_positions": sum(attempted_lengths) * BATCH_SIZE,
        "completed_readout_positions": sum(completed_lengths) * BATCH_SIZE,
        "attempted_native_steps": sum(attempted_lengths) * BATCH_SIZE * NATIVE_STEPS,
        "completed_native_steps": sum(completed_lengths) * BATCH_SIZE * NATIVE_STEPS,
    }


def evaluation_accounting(attempted_calls: int, completed_calls: int) -> dict[str, int]:
    """Account a prefix of the fixed 69-program, 256-state evaluation."""

    attempted, completed = _bounded_pair(attempted_calls, completed_calls, maximum=EVALUATION_CALLS, label="evaluation")
    return {
        "attempted_calls": attempted,
        "completed_calls": completed,
        "attempted_cases": attempted * REGISTER_WIDTH * REGISTER_WIDTH,
        "completed_cases": completed * REGISTER_WIDTH * REGISTER_WIDTH,
        "attempted_readout_positions": sum(EVALUATION_PROGRAM_LENGTHS[:attempted]) * REGISTER_WIDTH * REGISTER_WIDTH,
        "completed_readout_positions": sum(EVALUATION_PROGRAM_LENGTHS[:completed]) * REGISTER_WIDTH * REGISTER_WIDTH,
        "attempted_native_steps": sum(EVALUATION_PROGRAM_LENGTHS[:attempted]) * REGISTER_WIDTH * REGISTER_WIDTH * NATIVE_STEPS,
        "completed_native_steps": sum(EVALUATION_PROGRAM_LENGTHS[:completed]) * REGISTER_WIDTH * REGISTER_WIDTH * NATIVE_STEPS,
    }


EVALUATION_ACCOUNTING = evaluation_accounting(EVALUATION_CALLS, EVALUATION_CALLS)


__all__ = [
    "ARCHITECTURE_ID", "ARMS", "BATCH_SIZE", "CHILD_UPDATES", "EVALUATION_ACCOUNTING", "EVALUATION_CALLS",
    "EVALUATION_CASES", "EVALUATION_NATIVE_STEPS", "EVALUATION_PROGRAM_LENGTHS", "EVALUATION_READOUT_POSITIONS",
    "FIXED_BATCH_LENGTHS", "NATIVE_STEPS", "PARENT_UPDATE", "SCIENCE_ACCOUNTING", "TINY_QA_CALLS", "TINY_QA_CASES",
    "TINY_QA_NATIVE_STEPS", "TINY_QA_READOUT_POSITIONS", "TOTAL_ACCOUNTING", "TRAINING_CASES", "TRAINING_NATIVE_STEPS",
    "TRAINING_READOUT_POSITIONS", "cache_from_probabilities", "child_identity", "device_cross_entropy",
    "evaluation_accounting", "signed_bit_matrix", "soft_register_forward", "tiny_qa_accounting", "training_accounting",
    "validate_child_identity",
]
