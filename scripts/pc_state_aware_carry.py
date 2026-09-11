"""Pure state-aware writer seam for the latent-slot carry experiment.

The accepted latent-slot path must keep writing a fresh state after every
opcode.  Its structural weakness is that the writer currently receives only
the returned hidden vector: the two slot vectors are summed before the writer
is called.  This module keeps the existing writer as an exact base path and
adds a zero-initialized correction that receives ``[z_t, h_t]``.  The scoped
hooks preserve the accepted forward implementation while making the state
transition Markov in the carried slots.

No checkpoint loading, CUDA setup, training loop, or scientific evaluation is
performed here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import Tensor, nn

from scripts import pc_latent_slots as latent


STATE_INPUT_WIDTH = latent.OUTPUT_WIDTH + latent.HIDDEN_WIDTH
CORRECTION_PARAMETER_COUNT = latent.OUTPUT_WIDTH * STATE_INPUT_WIDTH + latent.OUTPUT_WIDTH
STATE_AWARE_ARCHITECTURE_ID = "latent_slots_2x16_state_aware_carry_v1"
STATE_AWARE_ADAPTER_PARAMETER_COUNT = latent.ADAPTER_PARAMETER_COUNT + CORRECTION_PARAMETER_COUNT
STATE_AWARE_ADAPTER_PARAMETER_NAMES = (
    "initializer.weight",
    "initializer.bias",
    "reader.weight",
    "writer.base_writer.weight",
    "writer.base_writer.bias",
    "writer.correction.weight",
    "writer.correction.bias",
)
BASE_WRITER_PARAMETER_NAMES = (
    "writer.base_writer.weight",
    "writer.base_writer.bias",
)
CORRECTION_PARAMETER_NAMES = (
    "writer.correction.weight",
    "writer.correction.bias",
)


def _validate_base_writer(writer: nn.Module) -> nn.Linear:
    if not isinstance(writer, nn.Linear):
        raise ValueError("base writer must be nn.Linear")
    if writer.in_features != latent.HIDDEN_WIDTH or writer.out_features != latent.OUTPUT_WIDTH or writer.bias is None:
        raise ValueError("base writer must be Linear(128, 32, bias=True)")
    parameters = list(writer.parameters())
    if any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise ValueError("base writer parameters must remain float32")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("base writer parameters must share one device")
    return writer


def _validate_correction(correction: nn.Linear) -> torch.device:
    if not isinstance(correction, nn.Linear):
        raise ValueError("writer correction must be nn.Linear")
    if correction.in_features != STATE_INPUT_WIDTH or correction.out_features != latent.OUTPUT_WIDTH or correction.bias is None:
        raise ValueError("writer correction must be Linear(160, 32, bias=True)")
    parameters = list(correction.parameters())
    if any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise ValueError("writer correction parameters must remain float32")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("writer correction parameters must share one device")
    return parameters[0].device


def _validate_state_input(combined: Tensor, device: torch.device) -> None:
    if not isinstance(combined, Tensor) or combined.ndim != 2 or combined.shape[1] != STATE_INPUT_WIDTH:
        raise ValueError("state-aware writer input must have shape [batch, 160]")
    if combined.device != device or combined.dtype is not torch.float32:
        raise ValueError("state-aware writer input must be float32 on the writer device")
    if not torch.isfinite(combined).all():
        raise ValueError("state-aware writer input is nonfinite")


def _validate_state_and_hidden(z: Tensor, normalized_hidden: Tensor, device: torch.device) -> None:
    if not isinstance(z, Tensor) or z.ndim != 3 or tuple(z.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH):
        raise ValueError("state-aware z must have shape [batch, 2, 16]")
    if not isinstance(normalized_hidden, Tensor) or normalized_hidden.ndim != 2 or normalized_hidden.shape[0] != z.shape[0] or normalized_hidden.shape[1] != latent.HIDDEN_WIDTH:
        raise ValueError("state-aware hidden input must have shape [batch, 128]")
    if z.device != device or normalized_hidden.device != device or z.dtype is not torch.float32 or normalized_hidden.dtype is not torch.float32:
        raise ValueError("state-aware inputs must be float32 on one device")
    if not torch.isfinite(z).all() or not torch.isfinite(normalized_hidden).all():
        raise ValueError("state-aware inputs are nonfinite")


class StateAwareWriter(nn.Module):
    """Existing writer plus a zero-initialized correction from ``[z_t, h_t]``.

    ``base_writer`` is the original ``Linear(128, 32)`` object, so its
    parameter identities and optimizer moments survive the wrapper.  The
    correction starts exactly at zero and therefore leaves the accepted
    endpoint unchanged before training.
    """

    def __init__(self, base_writer: nn.Linear) -> None:
        super().__init__()
        self.base_writer = _validate_base_writer(base_writer)
        device = self.base_writer.weight.device
        dtype = self.base_writer.weight.dtype
        with latent._preserve_global_rng():
            self.correction = nn.Linear(
                STATE_INPUT_WIDTH,
                latent.OUTPUT_WIDTH,
                bias=True,
                device=device,
                dtype=dtype,
            )
        with torch.no_grad():
            self.correction.weight.zero_()
            self.correction.bias.zero_()
        _validate_correction(self.correction)

    @classmethod
    def from_existing(cls, writer: nn.Module) -> "StateAwareWriter":
        return cls(_validate_base_writer(writer))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def correction_is_zero(self) -> bool:
        return bool(torch.count_nonzero(self.correction.weight) == 0 and torch.count_nonzero(self.correction.bias) == 0)

    def forward(self, combined: Tensor) -> Tensor:
        device = _validate_base_writer(self.base_writer).weight.device
        _validate_correction(self.correction)
        _validate_state_input(combined, device)
        hidden = combined[:, latent.OUTPUT_WIDTH:]
        result = self.base_writer(hidden) + self.correction(combined)
        if not torch.isfinite(result).all():
            raise ValueError("state-aware writer output is nonfinite")
        return result


def make_state_aware_writer(*, base_writer: nn.Linear) -> tuple[StateAwareWriter, dict[str, Any]]:
    """Wrap one existing writer without changing global RNG state."""

    writer = StateAwareWriter.from_existing(base_writer)
    return writer, {
        "architecture_id": STATE_AWARE_ARCHITECTURE_ID,
        "base_writer_shape": [latent.HIDDEN_WIDTH, latent.OUTPUT_WIDTH],
        "correction_shape": [STATE_INPUT_WIDTH, latent.OUTPUT_WIDTH],
        "correction_parameter_count": CORRECTION_PARAMETER_COUNT,
        "correction_initialization": "zero",
        "correction_is_zero": writer.correction_is_zero(),
    }


def state_aware_features(z: Tensor, normalized_hidden: Tensor, *, device: torch.device | None = None) -> Tensor:
    """Build the only new writer input, preserving both slot identities."""

    target_device = z.device if device is None else device
    _validate_state_and_hidden(z, normalized_hidden, target_device)
    features = torch.cat((z.reshape(z.shape[0], -1), normalized_hidden), dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("state-aware features are nonfinite")
    return features


def state_aware_slot_write(z: Tensor, normalized_hidden: Tensor, writer: StateAwareWriter) -> Tensor:
    """Return the differentiable next-slot tensor from current state and h."""

    if not isinstance(writer, StateAwareWriter):
        raise ValueError("writer must be StateAwareWriter")
    device = _validate_base_writer(writer.base_writer).weight.device
    features = state_aware_features(z, normalized_hidden, device=device)
    result = writer(features).reshape(z.shape[0], latent.SLOT_COUNT, latent.SLOT_WIDTH)
    if not torch.isfinite(result).all():
        raise ValueError("state-aware slot write is nonfinite")
    return result


@dataclass
class StateAwareHookState:
    reader_calls: int = 0
    writer_calls: int = 0
    _current_z: Tensor | None = None

    def _reader_pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        self.reader_calls += 1
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("state-aware reader hook received malformed slots")
        z = args[0]
        if z.ndim != 3 or tuple(z.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH) or not torch.isfinite(z).all():
            raise ValueError("state-aware reader slots have malformed shape or are nonfinite")
        self._current_z = z

    def _writer_pre_hook(self, _module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        self.writer_calls += 1
        if self._current_z is None:
            raise RuntimeError("state-aware writer ran before its reader")
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("state-aware writer hook received malformed hidden input")
        features = state_aware_features(self._current_z, args[0])
        self._current_z = None
        return (features, *args[1:])

    def finish(self, expected_calls: int) -> None:
        if self.reader_calls != expected_calls or self.writer_calls != expected_calls:
            raise ValueError(
                "state-aware hook call count changed: "
                f"reader={self.reader_calls}, writer={self.writer_calls}, expected={expected_calls}"
            )
        if self._current_z is not None:
            raise ValueError("state-aware run ended with an uncoupled reader input")

    def diagnostics(self) -> dict[str, Any]:
        return {"reader_calls": self.reader_calls, "writer_calls": self.writer_calls}


@contextmanager
def state_aware_hooks(adapter: latent.LatentSlotAdapter) -> Iterator[StateAwareHookState]:
    """Install reader/writer argument hooks and remove them on every exit."""

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    if not isinstance(getattr(adapter, "writer", None), StateAwareWriter):
        raise ValueError("adapter.writer must be StateAwareWriter")
    state = StateAwareHookState()
    handles: list[Any] = []
    try:
        handles.append(adapter.reader.register_forward_pre_hook(state._reader_pre_hook))
        handles.append(adapter.writer.register_forward_pre_hook(state._writer_pre_hook))
        yield state
    finally:
        for handle in reversed(handles):
            handle.remove()


def state_aware_latent_slots_forward(
    model: Any,
    adapter: latent.LatentSlotAdapter,
    x_bits: Tensor,
    y_bits: Tensor,
    ops: Any,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Run the accepted latent forward with the state-aware writer seam."""

    if not isinstance(getattr(adapter, "writer", None), StateAwareWriter):
        raise ValueError("state-aware writer is required")
    with state_aware_hooks(adapter) as state:
        result, diagnostics = latent.latent_slots_forward(
            model, adapter, x_bits, y_bits, ops, return_diagnostics=True
        )
        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("slot_inputs"), list):
            raise ValueError("accepted latent diagnostics are malformed")
        state.finish(len(diagnostics["slot_inputs"]))
    diagnostics = dict(diagnostics)
    diagnostics["state_aware_carry"] = state.diagnostics()
    diagnostics["writer_receives"] = "[current_slots, returned_hidden]"
    if return_diagnostics:
        return result, diagnostics
    return result


def install_state_aware_writer(
    adapter: latent.LatentSlotAdapter,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    adapter_group_index: int = -1,
) -> tuple[StateAwareWriter, dict[str, Any]]:
    """Replace the writer and optionally extend its existing optimizer group.

    The old writer parameter objects stay inside ``base_writer``.  Therefore
    loaded optimizer moments remain attached to the same objects; only the
    new correction parameters are appended with empty state.
    """

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    if isinstance(getattr(adapter, "writer", None), StateAwareWriter):
        raise ValueError("adapter already has a state-aware writer")
    old_parameters = list(adapter.parameters())
    old_writer = _validate_base_writer(getattr(adapter, "writer", None))
    old_names = list(name for name, _ in adapter.named_parameters())
    if tuple(old_names) != latent.ADAPTER_PARAMETER_NAMES:
        raise ValueError("ordinary latent adapter parameter names/order changed")
    if optimizer is not None:
        try:
            group = optimizer.param_groups[adapter_group_index]
        except IndexError as exc:
            raise ValueError("adapter optimizer group is missing") from exc
        actual = list(group.get("params", ()))
        if len(actual) != len(old_parameters) or any(left is not right for left, right in zip(actual, old_parameters)):
            raise ValueError("adapter optimizer group does not contain exactly ordinary I/R/W")
    writer, metadata = make_state_aware_writer(base_writer=old_writer)
    adapter.writer = writer
    current_parameters = list(adapter.parameters())
    current_names = [name for name, _ in adapter.named_parameters()]
    if tuple(current_names) != STATE_AWARE_ADAPTER_PARAMETER_NAMES or sum(parameter.numel() for parameter in current_parameters) != STATE_AWARE_ADAPTER_PARAMETER_COUNT:
        raise ValueError("state-aware adapter parameter inventory changed")
    if optimizer is not None:
        group = optimizer.param_groups[adapter_group_index]
        group["params"] = old_parameters + list(writer.correction.parameters())
        group["param_names"] = list(current_names)
        validate_optimizer_state_aware_association(optimizer, adapter, group_index=adapter_group_index)
    metadata = {
        **metadata,
        "base_writer_parameter_ids_preserved": True,
        "base_writer_parameter_names": list(BASE_WRITER_PARAMETER_NAMES),
        "correction_parameter_names": list(CORRECTION_PARAMETER_NAMES),
        "adapter_parameter_count": len(current_parameters),
    }
    return writer, metadata


def validate_optimizer_state_aware_association(
    optimizer: torch.optim.Optimizer,
    adapter: latent.LatentSlotAdapter,
    *,
    group_index: int = -1,
) -> tuple[str, ...]:
    """Require one optimizer group to contain the complete wrapped adapter."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer is not a torch optimizer")
    if not isinstance(getattr(adapter, "writer", None), StateAwareWriter):
        raise ValueError("adapter.writer is not state-aware")
    try:
        group = optimizer.param_groups[group_index]
    except IndexError as exc:
        raise ValueError("adapter optimizer group is missing") from exc
    expected = list(adapter.parameters())
    actual = list(group.get("params", ()))
    if len(actual) != len(expected) or any(left is not right for left, right in zip(actual, expected)):
        raise ValueError("state-aware adapter optimizer parameters/order changed")
    names = tuple(name for name, _ in adapter.named_parameters())
    stored = group.get("param_names")
    if stored is not None and tuple(stored) != names:
        raise ValueError("state-aware optimizer parameter names/order changed")
    return names


def _parameter_bytes(parameter: Tensor) -> bytes:
    return parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _adapter_state_digest(adapter: latent.LatentSlotAdapter) -> str:
    digest = hashlib.sha256()
    for name, parameter in adapter.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(json.dumps(list(parameter.shape), separators=(",", ":")).encode("ascii"))
        digest.update(_parameter_bytes(parameter))
    return digest.hexdigest()


def state_aware_adapter_identity(adapter: latent.LatentSlotAdapter) -> dict[str, Any]:
    """Return immutable identity for the wrapped adapter checkpoint seam."""

    if not isinstance(adapter, latent.LatentSlotAdapter) or not isinstance(getattr(adapter, "writer", None), StateAwareWriter):
        raise ValueError("adapter.writer must be StateAwareWriter")
    parameters = list(adapter.parameters())
    if any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise ValueError("state-aware adapter parameters must remain float32")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("state-aware adapter parameters must share one device")
    names = tuple(name for name, _ in adapter.named_parameters())
    if names != STATE_AWARE_ADAPTER_PARAMETER_NAMES or sum(parameter.numel() for parameter in parameters) != STATE_AWARE_ADAPTER_PARAMETER_COUNT:
        raise ValueError("state-aware adapter parameter names/order changed")
    return {
        "schema": "pc_state_aware_carry_adapter_identity_v1",
        "architecture_id": STATE_AWARE_ARCHITECTURE_ID,
        "parameter_names": list(names),
        "parameter_shapes": [list(parameter.shape) for parameter in parameters],
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "state_digest": _adapter_state_digest(adapter),
        "base_writer_parameter_names": list(BASE_WRITER_PARAMETER_NAMES),
        "correction_parameter_names": list(CORRECTION_PARAMETER_NAMES),
    }


def validate_state_aware_adapter_identity(adapter: latent.LatentSlotAdapter, expected: Mapping[str, Any]) -> None:
    current = state_aware_adapter_identity(adapter)
    for key in (
        "schema", "architecture_id", "parameter_names", "parameter_shapes", "parameter_count",
        "state_digest", "base_writer_parameter_names", "correction_parameter_names",
    ):
        if current.get(key) != expected.get(key):
            raise ValueError(f"state-aware adapter identity mismatch: {key}")


__all__ = [
    "BASE_WRITER_PARAMETER_NAMES",
    "CORRECTION_PARAMETER_COUNT",
    "CORRECTION_PARAMETER_NAMES",
    "STATE_AWARE_ADAPTER_PARAMETER_COUNT",
    "STATE_AWARE_ADAPTER_PARAMETER_NAMES",
    "STATE_AWARE_ARCHITECTURE_ID",
    "STATE_INPUT_WIDTH",
    "StateAwareHookState",
    "StateAwareWriter",
    "install_state_aware_writer",
    "make_state_aware_writer",
    "state_aware_adapter_identity",
    "state_aware_features",
    "state_aware_hooks",
    "state_aware_latent_slots_forward",
    "state_aware_slot_write",
    "validate_optimizer_state_aware_association",
    "validate_state_aware_adapter_identity",
]
