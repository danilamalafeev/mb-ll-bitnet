"""Pure latent-slot interface and bounded pilot accounting.

This module contains only the prospective two-slot adapter and small identity,
accounting, and checkpoint-selection gates.  It does not construct the parent
interpreter, load a checkpoint, or run a training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import Tensor, nn

from scripts.pc_learned_scratchpad import device_cross_entropy


SLOT_COUNT = 2
SLOT_WIDTH = 16
INITIAL_INPUT_WIDTH = 8
HIDDEN_WIDTH = 128
OUTPUT_WIDTH = SLOT_COUNT * SLOT_WIDTH
NATIVE_STEPS = 8
INITIALIZER_PARAMETER_COUNT = INITIAL_INPUT_WIDTH * OUTPUT_WIDTH + OUTPUT_WIDTH
READER_PARAMETER_COUNT = SLOT_WIDTH * HIDDEN_WIDTH
WRITER_PARAMETER_COUNT = HIDDEN_WIDTH * OUTPUT_WIDTH + OUTPUT_WIDTH
ADAPTER_PARAMETER_COUNT = INITIALIZER_PARAMETER_COUNT + READER_PARAMETER_COUNT + WRITER_PARAMETER_COUNT
ARCHITECTURE_ID = "latent_slots_2x16_v1"
ADAPTER_PARAMETER_NAMES = (
    "initializer.weight",
    "initializer.bias",
    "reader.weight",
    "writer.weight",
    "writer.bias",
)
INITIALIZER_ORDER = ("initializer", "reader", "writer")
LOCAL_INITIALIZATION_SEED = 0
LOCAL_INITIALIZATION_STD = 0.02

FIXED_BATCH_LENGTHS = (1, 2, 3, 4, 5, 6) * 333 + (1, 2)
SCIENCE_TRAINING_UPDATES = 2000
SCIENCE_BATCH_SIZE = 64
SCIENCE_TRAINING_CASES = SCIENCE_TRAINING_UPDATES * SCIENCE_BATCH_SIZE
SCIENCE_TRAINING_POSITIONS = sum(FIXED_BATCH_LENGTHS) * SCIENCE_BATCH_SIZE
SCIENCE_TRAINING_NATIVE_STEPS = SCIENCE_TRAINING_POSITIONS * NATIVE_STEPS
SCIENCE_EVALUATION_CALLS = 69
SCIENCE_EVALUATION_CASES = SCIENCE_EVALUATION_CALLS * SLOT_WIDTH * SLOT_WIDTH
SCIENCE_EVALUATION_POSITIONS = 331776
SCIENCE_EVALUATION_NATIVE_STEPS = SCIENCE_EVALUATION_POSITIONS * NATIVE_STEPS
TRAINING_ACCOUNTING = {
    "calls": SCIENCE_TRAINING_UPDATES,
    "cases": SCIENCE_TRAINING_CASES,
    "readout_positions": SCIENCE_TRAINING_POSITIONS,
    "native_steps": SCIENCE_TRAINING_NATIVE_STEPS,
    "optimizer_updates": SCIENCE_TRAINING_UPDATES,
}
SCIENCE_ACCOUNTING = {
    "calls": 2138,
    "cases": 163328,
    "readout_positions": 1111296,
    "native_steps": 8890368,
    "optimizer_updates": 2000,
}
QA_ACCOUNTING = {
    "calls": 3,
    "cases": 6,
    "readout_positions": 10,
    "native_steps": 80,
    "optimizer_updates": 3,
}
TOTAL_ACCOUNTING = {
    "calls": 2141,
    "cases": 163334,
    "readout_positions": 1111306,
    "native_steps": 8890448,
    "optimizer_updates": 2003,
}


class LatentSlotAdapter(nn.Module):
    """The exactly three-module I/R/W adapter for two unconstrained slots."""

    def __init__(self) -> None:
        super().__init__()
        self.initializer = nn.Linear(INITIAL_INPUT_WIDTH, OUTPUT_WIDTH, bias=True)
        self.reader = nn.Linear(SLOT_WIDTH, HIDDEN_WIDTH, bias=False)
        self.writer = nn.Linear(HIDDEN_WIDTH, OUTPUT_WIDTH, bias=True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _cuda_rng_state() -> list[Tensor] | None:
    return [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None


@contextmanager
def _preserve_global_rng() -> Iterator[None]:
    cpu_state = torch.get_rng_state()
    cuda_state = _cuda_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _local_normal(shape: Sequence[int], generator: torch.Generator) -> Tensor:
    return torch.empty(tuple(shape), dtype=torch.float32, device="cpu").normal_(
        mean=0.0, std=LOCAL_INITIALIZATION_STD, generator=generator
    )


def initialize_adapter(
    adapter: LatentSlotAdapter,
    *,
    seed: int = LOCAL_INITIALIZATION_SEED,
) -> dict[str, Any]:
    """Initialize I, R, W from one local CPU generator without global RNG use."""

    if not isinstance(adapter, LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    if adapter.parameter_count() != ADAPTER_PARAMETER_COUNT:
        raise ValueError("latent adapter parameter count changed")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with _preserve_global_rng(), torch.no_grad():
        for module_name in INITIALIZER_ORDER:
            module = getattr(adapter, module_name)
            weight = _local_normal(tuple(module.weight.shape), generator)
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None:
                module.bias.zero_()
    return {
        "seed": int(seed),
        "generator_device": "cpu",
        "order": list(INITIALIZER_ORDER),
        "weight_distribution": {"mean": 0.0, "std": LOCAL_INITIALIZATION_STD},
        "bias_value": 0.0,
        "parameter_count": ADAPTER_PARAMETER_COUNT,
    }


def make_adapter(*, device: torch.device | str | None = None) -> tuple[LatentSlotAdapter, dict[str, Any]]:
    """Construct and locally initialize an adapter while preserving parent RNG."""

    with _preserve_global_rng():
        adapter = LatentSlotAdapter()
    if device is not None:
        adapter.to(torch.device(device))
    metadata = initialize_adapter(adapter)
    return adapter, metadata


def _validate_adapter_device(adapter: LatentSlotAdapter) -> torch.device:
    parameters = list(adapter.parameters())
    if not parameters:
        raise ValueError("adapter has no parameters")
    if any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise ValueError("adapter parameters must remain float32")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("adapter parameters must share one device")
    return parameters[0].device


def initialize_slots(adapter: LatentSlotAdapter, x_bits: Tensor, y_bits: Tensor) -> Tensor:
    """Run I once on the external eight signed initial-register bits."""

    device = _validate_adapter_device(adapter)
    if not isinstance(x_bits, Tensor) or not isinstance(y_bits, Tensor):
        raise ValueError("x_bits and y_bits must be tensors")
    if x_bits.ndim != 2 or y_bits.shape != x_bits.shape or tuple(x_bits.shape[1:]) != (4,):
        raise ValueError("x_bits and y_bits must have shape [batch, 4]")
    if x_bits.device != device or y_bits.device != device:
        raise ValueError("initial bits and adapter must share a device")
    if x_bits.dtype is not torch.float32 or y_bits.dtype is not torch.float32:
        raise ValueError("initial bits must remain float32")
    if not torch.isfinite(x_bits).all() or not torch.isfinite(y_bits).all():
        raise ValueError("initial bits must be finite")
    for label, value in (("x_bits", x_bits), ("y_bits", y_bits)):
        if not torch.all((value == -1.0) | (value == 1.0)):
            raise ValueError(f"{label} must contain signed bits only")
    flat = adapter.initializer(torch.cat((x_bits, y_bits), dim=-1))
    return flat.reshape(x_bits.shape[0], SLOT_COUNT, SLOT_WIDTH)


def _validate_slots(slots: Tensor, device: torch.device) -> None:
    if not isinstance(slots, Tensor) or slots.ndim != 3 or tuple(slots.shape[1:]) != (SLOT_COUNT, SLOT_WIDTH):
        raise ValueError("slots must have shape [batch, 2, 16]")
    if slots.device != device or slots.dtype is not torch.float32:
        raise ValueError("slots must be float32 on the adapter device")
    if not torch.isfinite(slots).all():
        raise ValueError("slots must be finite")


def fresh_cache_from_slots(model: Any, adapter: LatentSlotAdapter, slots: Tensor) -> dict[str, Any]:
    """Build a new native cache from slots; no prior cache or original inputs are read."""

    adapter_device = _validate_adapter_device(adapter)
    _validate_slots(slots, adapter_device)
    if getattr(model, "native_steps", NATIVE_STEPS) != NATIVE_STEPS:
        raise ValueError("model.step must retain native8 execution")
    role_keys = getattr(model, "role_keys", None)
    memory_norm = getattr(model, "memory_norm", None)
    core_reader = getattr(model, "reader", None)
    if not isinstance(role_keys, Tensor) or tuple(role_keys.shape) != (SLOT_COUNT, HIDDEN_WIDTH):
        raise ValueError("model role_keys must have shape [2, 128]")
    if role_keys.device != adapter_device:
        raise ValueError("model and adapter must share a device")
    if not callable(memory_norm) or core_reader is None or not hasattr(core_reader, "build_kv"):
        raise ValueError("model lacks the registered memory reader")
    values = adapter.reader(slots)
    key_memory = memory_norm(role_keys).unsqueeze(0).expand(slots.shape[0], -1, -1).clone()
    value_memory = memory_norm(values)
    with torch.autocast(device_type=adapter_device.type, enabled=False):
        kv = core_reader.build_kv(key_memory, value_memory)
    return {
        "h": values[:, 0] + values[:, 1],
        "x_initial": values[:, 0],
        "y_initial": values[:, 1],
        "kv": kv,
        "substeps": 0,
    }


def _operation_ids(ops: Any, *, batch: int, device: torch.device) -> Tensor:
    if isinstance(ops, Tensor):
        result = ops
        if result.ndim != 2 or result.shape[0] != batch or result.dtype is not torch.long or result.device != device:
            raise ValueError("ops must have shape [batch, length] and torch.long dtype on the adapter device")
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


def latent_slots_forward(
    model: Any,
    adapter: LatentSlotAdapter,
    x_bits: Tensor,
    y_bits: Tensor,
    ops: Any,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Run native8 steps with fresh caches and differentiable own slot writes."""

    device = _validate_adapter_device(adapter)
    slots = initialize_slots(adapter, x_bits, y_bits)
    op_ids = _operation_ids(ops, batch=x_bits.shape[0], device=device)
    if getattr(model, "native_steps", NATIVE_STEPS) != NATIVE_STEPS:
        raise ValueError("model.step must retain native8 execution")
    output_norm = getattr(model, "output_norm", None)
    if not callable(output_norm):
        raise ValueError("model lacks output_norm for the registered writer")
    x_logits: list[Tensor] = []
    y_logits: list[Tensor] = []
    slot_inputs: list[Tensor] = []
    slot_writes: list[Tensor] = []
    substeps: list[int] = []
    for position in range(op_ids.shape[1]):
        slot_inputs.append(slots)
        cache = fresh_cache_from_slots(model, adapter, slots)
        substeps.append(int(cache["substeps"]))
        result = model.step(cache, op_ids[:, position])
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("model.step must return ((x_logits, y_logits), cache)")
        logits, returned_cache = result
        if not isinstance(logits, tuple) or len(logits) != 2 or not isinstance(returned_cache, Mapping):
            raise ValueError("model.step returned malformed output")
        logits_x, logits_y = logits
        expected_shape = (x_bits.shape[0], SLOT_WIDTH)
        if tuple(logits_x.shape) != expected_shape or tuple(logits_y.shape) != expected_shape:
            raise ValueError("model.step logits must have shape [batch, 16]")
        returned_h = returned_cache.get("h")
        if not isinstance(returned_h, Tensor) or tuple(returned_h.shape) != (x_bits.shape[0], HIDDEN_WIDTH):
            raise ValueError("model.step returned cache h must have shape [batch, 128]")
        x_logits.append(logits_x)
        y_logits.append(logits_y)
        slots = adapter.writer(output_norm(returned_h)).reshape(x_bits.shape[0], SLOT_COUNT, SLOT_WIDTH)
        if not torch.isfinite(slots).all():
            raise ValueError("latent slot write is nonfinite")
        slot_writes.append(slots)
    output = (torch.stack(x_logits, dim=1), torch.stack(y_logits, dim=1))
    if return_diagnostics:
        return output, {
            "slot_inputs": slot_inputs,
            "slot_writes": slot_writes,
            "substeps": substeps,
            "native_steps": [NATIVE_STEPS] * op_ids.shape[1],
            "initializer_calls": 1,
            "detached_writes": False,
            "targets_in_signature": False,
        }
    return output


def _parameter_bytes(parameter: Tensor) -> bytes:
    return parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _adapter_state_digest(adapter: LatentSlotAdapter) -> str:
    digest = hashlib.sha256()
    for name, parameter in adapter.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(json.dumps(list(parameter.shape), separators=(",", ":")).encode("ascii"))
        digest.update(_parameter_bytes(parameter))
    return digest.hexdigest()


def adapter_identity(adapter: LatentSlotAdapter) -> dict[str, Any]:
    """Return the immutable parameter/name identity used by the later runtime."""

    _validate_adapter_device(adapter)
    names = tuple(name for name, _ in adapter.named_parameters())
    if names != ADAPTER_PARAMETER_NAMES:
        raise ValueError("latent adapter parameter names/order changed")
    return {
        "schema": "pc_latent_slots_adapter_identity_v1",
        "architecture_id": ARCHITECTURE_ID,
        "parameter_names": list(names),
        "parameter_shapes": [list(parameter.shape) for _, parameter in adapter.named_parameters()],
        "parameter_count": adapter.parameter_count(),
        "state_digest": _adapter_state_digest(adapter),
    }


def validate_adapter_identity(adapter: LatentSlotAdapter, expected: Mapping[str, Any]) -> None:
    current = adapter_identity(adapter)
    for key in ("schema", "architecture_id", "parameter_names", "parameter_shapes", "parameter_count", "state_digest"):
        if current.get(key) != expected.get(key):
            raise ValueError(f"latent adapter identity mismatch: {key}")


def validate_optimizer_adapter_association(
    optimizer: torch.optim.Optimizer,
    adapter: LatentSlotAdapter,
    *,
    group_index: int = -1,
) -> tuple[str, ...]:
    """Require one optimizer group to contain exactly I/R/W in named order."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer is not a torch optimizer")
    groups = optimizer.param_groups
    if not groups:
        raise ValueError("optimizer has no parameter groups")
    try:
        group = groups[group_index]
    except IndexError as exc:
        raise ValueError("adapter optimizer group is missing") from exc
    expected = list(adapter.parameters())
    actual = list(group.get("params", ()))
    if len(actual) != len(expected) or any(left is not right for left, right in zip(actual, expected)):
        raise ValueError("adapter optimizer parameters/order changed")
    return ADAPTER_PARAMETER_NAMES


def training_accounting(batch_lengths: Sequence[int]) -> dict[str, int]:
    """Validate and account the exact 2,000-call cyclic B prefix."""

    lengths = tuple(int(value) for value in batch_lengths)
    if lengths != FIXED_BATCH_LENGTHS:
        raise ValueError("latent science batch length cycle or tail changed")
    return {
        "calls": len(lengths),
        "cases": len(lengths) * SCIENCE_BATCH_SIZE,
        "readout_positions": sum(lengths) * SCIENCE_BATCH_SIZE,
        "native_steps": sum(lengths) * SCIENCE_BATCH_SIZE * NATIVE_STEPS,
        "optimizer_updates": len(lengths),
    }


def select_committed_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
    *,
    manifest_digest: str,
    next_batch_index: int,
) -> Mapping[str, Any]:
    """Select one committed metadata record matching lineage and stream index."""

    matches = []
    for candidate in candidates:
        path = str(candidate.get("path", ""))
        if path.lower().endswith(".tmp") or not bool(candidate.get("committed", False)):
            continue
        if candidate.get("manifest_digest") != manifest_digest:
            continue
        if candidate.get("next_batch_index") != next_batch_index:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("checkpoint set has no unique committed matching boundary")
    return matches[0]


def pure_digest(value: Any) -> str:
    """Small canonical digest helper for hand-built fixture bindings."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ADAPTER_PARAMETER_COUNT",
    "ADAPTER_PARAMETER_NAMES",
    "ARCHITECTURE_ID",
    "FIXED_BATCH_LENGTHS",
    "LatentSlotAdapter",
    "NATIVE_STEPS",
    "QA_ACCOUNTING",
    "SCIENCE_ACCOUNTING",
    "TOTAL_ACCOUNTING",
    "TRAINING_ACCOUNTING",
    "adapter_identity",
    "device_cross_entropy",
    "fresh_cache_from_slots",
    "initialize_adapter",
    "initialize_slots",
    "latent_slots_forward",
    "make_adapter",
    "pure_digest",
    "select_committed_checkpoint",
    "training_accounting",
    "validate_adapter_identity",
    "validate_optimizer_adapter_association",
]
