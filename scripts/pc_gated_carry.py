"""Pure gated carry proposal and fixed control fixtures.

This module contains no checkpoint loading, model construction, CUDA setup, or
training loop.  It adds the prospective two-scalar carry gate around the
accepted latent adapter's ordinary writer output.  The runtime can attach the
gate after its strict endpoint and optimizer identity checks.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent


GATE_INPUT_WIDTH = latent.OUTPUT_WIDTH + latent.HIDDEN_WIDTH
GATE_OUTPUT_WIDTH = latent.SLOT_COUNT
GATE_PARAMETER_COUNT = GATE_OUTPUT_WIDTH * GATE_INPUT_WIDTH + GATE_OUTPUT_WIDTH
GATE_INITIAL_LOGIT = math.log(9.0)
GATE_INITIAL_VALUE = 0.9
GATED_ARCHITECTURE_ID = "latent_slots_2x16_gated_carry_v1"
CARRY_GATE_PARAMETER_NAMES = ("carry_gate.weight", "carry_gate.bias")
GATED_ADAPTER_PARAMETER_COUNT = latent.ADAPTER_PARAMETER_COUNT + GATE_PARAMETER_COUNT

TRAINING_BATCH_LENGTHS = (1, 2, 3, 4, 5, 6) * 166 + (1, 2, 3, 4)
TRAINING_UPDATES_PER_ARM = 1000
TRAINING_BATCH_SIZE = 64
TRAINING_CASES_PER_ARM = TRAINING_UPDATES_PER_ARM * TRAINING_BATCH_SIZE
TRAINING_POSITIONS_PER_ARM = sum(TRAINING_BATCH_LENGTHS) * TRAINING_BATCH_SIZE
TRAINING_NATIVE_STEPS_PER_ARM = TRAINING_POSITIONS_PER_ARM * latent.NATIVE_STEPS

QA_ACCOUNTING = {
    "forwards": 6,
    "cases": 12,
    "positions": 20,
    "native_steps": 160,
    "backwards": 6,
    "optimizer_updates": 6,
    "underlying_deserializations": 8,
}
SCIENCE_ACCOUNTING = {
    "training_forwards": 2000,
    "evaluation_forwards": 531,
    "forwards": 2531,
    "cases": 263936,
    "positions": 3239936,
    "native_steps": 25919488,
    "backwards": 2000,
    "optimizer_updates": 2000,
    "underlying_deserializations": 6,
}

CONTROL_PREFIXES = (
    ("ADDADD", ("ADD", "ADD")),
    ("XORSWAP", ("XOR", "SWAP")),
    ("SWAPXOR", ("SWAP", "XOR")),
)
IDENTITY_CELLS = (
    ("SSSS", ("SWAP", "SWAP", "SWAP", "SWAP")),
    ("XXXX", ("XOR", "XOR", "XOR", "XOR")),
    ("SXXS", ("SWAP", "XOR", "XOR", "SWAP")),
)
CONTROL_REPEATS = (2, 4, 8)
CONTROL_SUFFIXES = ("ADD", "XOR", "SWAP")
CONTROL_PROGRAM_COUNT = 81
CONTROL_LENGTHS = (11, 19, 35)
CONTROL_PROGRAMS_PER_LENGTH = 27


def _strict_program(program: Sequence[str]) -> tuple[str, ...]:
    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence) or not program:
        raise ValueError("program must be a nonempty opcode sequence")
    if any(not isinstance(opcode, str) or opcode not in dsl.OPS for opcode in program):
        raise ValueError("program contains an unknown opcode")
    return tuple(program)


def _validate_gate_device(gate: "CarryGate") -> torch.device:
    parameters = list(gate.parameters())
    if not parameters or any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise ValueError("carry gate parameters must be float32")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("carry gate parameters must share one device")
    if gate.parameter_count() != GATE_PARAMETER_COUNT:
        raise ValueError("carry gate parameter count changed")
    return parameters[0].device


class CarryGate(nn.Linear):
    """Two scalar gates from flattened current z and normalized returned h."""

    def __init__(self) -> None:
        super().__init__(GATE_INPUT_WIDTH, GATE_OUTPUT_WIDTH, bias=True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, z: Tensor, normalized_hidden: Tensor) -> Tensor:
        _validate_gate_inputs(z, normalized_hidden)
        features = torch.cat((z.reshape(z.shape[0], -1), normalized_hidden), dim=-1)
        logits = super().forward(features)
        if not torch.isfinite(logits).all():
            raise ValueError("carry gate logits are nonfinite")
        gates = torch.sigmoid(logits).unsqueeze(-1)
        if not torch.isfinite(gates).all():
            raise ValueError("carry gate values are nonfinite")
        return gates


def _validate_gate_inputs(z: Tensor, normalized_hidden: Tensor) -> None:
    if not isinstance(z, Tensor) or z.ndim != 3 or tuple(z.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH):
        raise ValueError("carry gate z must have shape [batch, 2, 16]")
    if not isinstance(normalized_hidden, Tensor) or normalized_hidden.ndim != 2 or normalized_hidden.shape[0] != z.shape[0] or normalized_hidden.shape[1] != latent.HIDDEN_WIDTH:
        raise ValueError("carry gate hidden input must have shape [batch, 128]")
    if z.device != normalized_hidden.device or z.dtype is not torch.float32 or normalized_hidden.dtype is not torch.float32:
        raise ValueError("carry gate inputs must be float32 on one device")
    if not torch.isfinite(z).all() or not torch.isfinite(normalized_hidden).all():
        raise ValueError("carry gate inputs are nonfinite")


def initialize_carry_gate(gate: CarryGate) -> dict[str, Any]:
    """Set exact zero weights and log(9) biases without using global RNG."""

    if not isinstance(gate, CarryGate):
        raise ValueError("gate must be CarryGate")
    _validate_gate_device(gate)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.fill_(GATE_INITIAL_LOGIT)
    return {
        "architecture_id": GATED_ARCHITECTURE_ID,
        "parameter_count": GATE_PARAMETER_COUNT,
        "weight_value": 0.0,
        "bias_value": GATE_INITIAL_LOGIT,
        "initial_gate": GATE_INITIAL_VALUE,
    }


def make_carry_gate(*, device: torch.device | str | None = None) -> tuple[CarryGate, dict[str, Any]]:
    """Construct and initialize a gate while preserving process RNG state."""

    with latent._preserve_global_rng():
        gate = CarryGate()
    if device is not None:
        gate.to(torch.device(device))
    metadata = initialize_carry_gate(gate)
    return gate, metadata


def attach_carry_gate(
    adapter: latent.LatentSlotAdapter,
    gate: CarryGate | None = None,
) -> tuple[CarryGate, dict[str, Any]]:
    """Attach one initialized gate as ``adapter.carry_gate`` exactly once."""

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    if hasattr(adapter, "carry_gate"):
        raise ValueError("adapter already has a carry gate")
    adapter_device = latent._validate_adapter_device(adapter)
    if gate is None:
        gate, metadata = make_carry_gate(device=adapter_device)
    else:
        if not isinstance(gate, CarryGate):
            raise ValueError("gate must be CarryGate")
        if _validate_gate_device(gate) != adapter_device:
            raise ValueError("adapter and carry gate must share a device")
        metadata = {
            "architecture_id": GATED_ARCHITECTURE_ID,
            "parameter_count": GATE_PARAMETER_COUNT,
            "weight_value": None,
            "bias_value": None,
            "initial_gate": None,
        }
    adapter.add_module("carry_gate", gate)
    return gate, metadata


def _gate_parameters(adapter: latent.LatentSlotAdapter) -> list[nn.Parameter]:
    gate = getattr(adapter, "carry_gate", None)
    if not isinstance(gate, CarryGate):
        raise ValueError("adapter.carry_gate is missing")
    _validate_gate_device(gate)
    return list(gate.parameters())


def append_carry_gate_optimizer_group(
    optimizer: torch.optim.Optimizer,
    adapter: latent.LatentSlotAdapter,
    *,
    adapter_group_index: int = -1,
) -> tuple[str, ...]:
    """Append a gate-only group copied from the existing adapter group."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer is not a torch optimizer")
    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    gate_parameters = _gate_parameters(adapter)
    try:
        template = optimizer.param_groups[adapter_group_index]
    except IndexError as exc:
        raise ValueError("adapter optimizer group is missing") from exc
    adapter_parameters = [parameter for name, parameter in adapter.named_parameters() if not name.startswith("carry_gate.")]
    actual = list(template.get("params", ()))
    if len(actual) != len(adapter_parameters) or any(left is not right for left, right in zip(actual, adapter_parameters)):
        raise ValueError("adapter optimizer group does not contain exactly I/R/W")
    if any(any(parameter is candidate for candidate in gate_parameters) for group in optimizer.param_groups for parameter in group.get("params", ())):
        raise ValueError("carry gate is already associated with optimizer")
    copied = {key: value for key, value in template.items() if key not in {"params", "param_names"}}
    copied["params"] = gate_parameters
    copied["param_names"] = list(CARRY_GATE_PARAMETER_NAMES)
    optimizer.add_param_group(copied)
    validate_optimizer_carry_gate_association(optimizer, adapter)
    return CARRY_GATE_PARAMETER_NAMES


def validate_optimizer_carry_gate_association(
    optimizer: torch.optim.Optimizer,
    adapter: latent.LatentSlotAdapter,
    *,
    group_index: int = -1,
) -> tuple[str, ...]:
    """Require one optimizer group to contain exactly the two gate params."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer is not a torch optimizer")
    gate_parameters = _gate_parameters(adapter)
    try:
        group = optimizer.param_groups[group_index]
    except IndexError as exc:
        raise ValueError("carry gate optimizer group is missing") from exc
    actual = list(group.get("params", ()))
    if len(actual) != len(gate_parameters) or any(left is not right for left, right in zip(actual, gate_parameters)):
        raise ValueError("carry gate optimizer parameters/order changed")
    names = group.get("param_names")
    if names is not None and list(names) != list(CARRY_GATE_PARAMETER_NAMES):
        raise ValueError("carry gate optimizer parameter names changed")
    return CARRY_GATE_PARAMETER_NAMES


def blend_carry_state(z: Tensor, proposal: Tensor, gates: Tensor) -> Tensor:
    """Apply broadcast per-slot gates to a raw writer proposal."""

    if not isinstance(z, Tensor) or not isinstance(proposal, Tensor) or z.ndim != 3 or proposal.ndim != 3 or proposal.shape != z.shape or tuple(z.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH):
        raise ValueError("z and proposal must both have shape [batch, 2, 16]")
    if not isinstance(gates, Tensor) or gates.shape != (z.shape[0], latent.SLOT_COUNT, 1):
        raise ValueError("gates must have shape [batch, 2, 1]")
    if z.device != proposal.device or proposal.device != gates.device or z.dtype is not torch.float32 or proposal.dtype is not torch.float32 or gates.dtype is not torch.float32:
        raise ValueError("carry blend tensors must be float32 on one device")
    if not torch.isfinite(proposal).all() or not torch.isfinite(gates).all() or not torch.isfinite(z).all():
        raise ValueError("carry blend input is nonfinite")
    if torch.any(gates < 0.0) or torch.any(gates > 1.0):
        raise ValueError("carry gates must lie in [0, 1]")
    result = (1.0 - gates) * z + gates * proposal
    if not torch.isfinite(result).all():
        raise ValueError("carried state is nonfinite")
    return result


def gated_carry_update(
    z: Tensor,
    proposal: Tensor,
    normalized_hidden: Tensor,
    gate: CarryGate,
) -> tuple[Tensor, Tensor]:
    """Return differentiable carried state and the scalar gate tensor."""

    _validate_gate_device(gate)
    _validate_gate_inputs(z, normalized_hidden)
    if not isinstance(proposal, Tensor) or proposal.shape != z.shape or proposal.device != z.device or proposal.dtype is not torch.float32:
        raise ValueError("writer proposal must have shape/device/dtype of z")
    if not torch.isfinite(proposal).all():
        raise ValueError("raw writer proposal is nonfinite")
    gates = gate(z, normalized_hidden)
    return blend_carry_state(z, proposal, gates), gates


# Descriptive aliases for the runtime seam and focused tests.
gated_slot_write = gated_carry_update
apply_gated_carry = blend_carry_state


@dataclass
class GatedCarryHookState:
    gate: CarryGate
    reader_calls: int = 0
    writer_calls: int = 0
    _current_z: Tensor | None = None
    gate_values: list[Tensor] = field(default_factory=list)

    def _reader_pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        self.reader_calls += 1
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("gated carry reader hook received malformed z")
        z = args[0]
        if z.ndim != 3 or tuple(z.shape[1:]) != (latent.SLOT_COUNT, latent.SLOT_WIDTH) or not torch.isfinite(z).all():
            raise ValueError("gated carry reader z is malformed or nonfinite")
        self._current_z = z.clone()

    def _writer_hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> Any:
        self.writer_calls += 1
        if self._current_z is None:
            raise RuntimeError("gated carry writer ran before its reader")
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("gated carry writer hook received malformed hidden input")
        if not isinstance(output, Tensor) or output.ndim < 1:
            raise ValueError("gated carry writer proposal is malformed")
        if not torch.isfinite(output).all():
            raise ValueError("raw writer proposal is nonfinite")
        proposal = output.reshape_as(self._current_z)
        carried, gates = gated_carry_update(self._current_z, proposal, args[0], self.gate)
        self.gate_values.append(gates)
        self._current_z = None
        return carried.reshape_as(output)

    def finish(self, expected_calls: int) -> None:
        if self.reader_calls != expected_calls or self.writer_calls != expected_calls:
            raise ValueError(f"gated carry hook call count changed: reader={self.reader_calls}, writer={self.writer_calls}, expected={expected_calls}")
        if self._current_z is not None:
            raise ValueError("gated carry ended with an uncoupled reader input")

    def diagnostics(self) -> dict[str, Any]:
        return {"reader_calls": self.reader_calls, "writer_calls": self.writer_calls, "gate_values": list(self.gate_values)}


@contextmanager
def gated_carry_hooks(adapter: latent.LatentSlotAdapter) -> Iterator[GatedCarryHookState]:
    """Install the scoped reader/writer hooks and always remove both handles."""

    if not isinstance(adapter, latent.LatentSlotAdapter):
        raise ValueError("adapter must be LatentSlotAdapter")
    gate = getattr(adapter, "carry_gate", None)
    if not isinstance(gate, CarryGate):
        raise ValueError("adapter.carry_gate is missing")
    _validate_gate_device(gate)
    state = GatedCarryHookState(gate)
    handles: list[Any] = []
    try:
        handles.append(adapter.reader.register_forward_pre_hook(state._reader_pre_hook))
        handles.append(adapter.writer.register_forward_hook(state._writer_hook))
        yield state
    finally:
        for handle in reversed(handles):
            handle.remove()


def gated_latent_slots_forward(
    model: Any,
    adapter: latent.LatentSlotAdapter,
    x_bits: Tensor,
    y_bits: Tensor,
    ops: Any,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Run accepted latent forward with a differentiable gated future state."""

    if not isinstance(adapter, latent.LatentSlotAdapter) or not isinstance(getattr(adapter, "carry_gate", None), CarryGate):
        raise ValueError("adapter.carry_gate is required")
    with gated_carry_hooks(adapter) as state:
        result, diagnostics = latent.latent_slots_forward(
            model, adapter, x_bits, y_bits, ops, return_diagnostics=True
        )
        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("slot_inputs"), list):
            raise ValueError("accepted latent diagnostics are malformed")
        state.finish(len(diagnostics["slot_inputs"]))
    diagnostics = dict(diagnostics)
    diagnostics["gated_carry"] = state.diagnostics()
    if return_diagnostics:
        return result, diagnostics
    return result


def _gate_digest(gate: CarryGate) -> str:
    digest = hashlib.sha256()
    for name, parameter in gate.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def carry_gate_identity(gate_or_adapter: CarryGate | latent.LatentSlotAdapter) -> dict[str, Any]:
    """Return immutable gate parameter identity for a later checkpoint schema."""

    gate = gate_or_adapter if isinstance(gate_or_adapter, CarryGate) else getattr(gate_or_adapter, "carry_gate", None)
    if not isinstance(gate, CarryGate):
        raise ValueError("carry gate is missing")
    _validate_gate_device(gate)
    return {
        "schema": "pc_gated_carry_gate_identity_v1",
        "architecture_id": GATED_ARCHITECTURE_ID,
        "parameter_names": list(CARRY_GATE_PARAMETER_NAMES[0:0]) + [f"carry_gate.{name}" for name, _ in gate.named_parameters()],
        "parameter_shapes": [list(parameter.shape) for parameter in gate.parameters()],
        "parameter_count": gate.parameter_count(),
        "state_digest": _gate_digest(gate),
    }


def validate_carry_gate_identity(gate_or_adapter: CarryGate | latent.LatentSlotAdapter, expected: Mapping[str, Any]) -> None:
    current = carry_gate_identity(gate_or_adapter)
    for key in ("schema", "architecture_id", "parameter_names", "parameter_shapes", "parameter_count", "state_digest"):
        if current.get(key) != expected.get(key):
            raise ValueError(f"carry gate identity mismatch: {key}")


def training_accounting(batch_lengths: Sequence[int] = TRAINING_BATCH_LENGTHS) -> dict[str, int]:
    lengths = tuple(int(value) for value in batch_lengths)
    if lengths != TRAINING_BATCH_LENGTHS:
        raise ValueError("gated carry stream length prefix changed")
    return {
        "updates_per_arm": TRAINING_UPDATES_PER_ARM,
        "arms": 2,
        "forwards": 2 * len(lengths),
        "cases": 2 * len(lengths) * TRAINING_BATCH_SIZE,
        "positions": 2 * sum(lengths) * TRAINING_BATCH_SIZE,
        "native_steps": 2 * sum(lengths) * TRAINING_BATCH_SIZE * latent.NATIVE_STEPS,
        "backwards": 2 * len(lengths),
        "optimizer_updates": 2 * len(lengths),
    }


def control_program_specs() -> tuple[dict[str, Any], ...]:
    """Return the fixed 81-program Cartesian control suite."""

    specs: list[dict[str, Any]] = []
    for prefix_name, prefix in CONTROL_PREFIXES:
        for cell_name, cell in IDENTITY_CELLS:
            for repeat in CONTROL_REPEATS:
                for suffix in CONTROL_SUFFIXES:
                    program = prefix + cell * repeat + (suffix,)
                    specs.append({
                        "id": f"control_{prefix_name}_{cell_name}_r{repeat}_{suffix}",
                        "suite": "identity_controls",
                        "prefix": prefix_name,
                        "cell": cell_name,
                        "repeat": repeat,
                        "suffix": suffix,
                        "length": len(program),
                        "program": program,
                    })
    return tuple(specs)


def validate_control_program_specs(specs: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    """Validate exact coverage and pure DSL identity of every repeated cell."""

    expected_specs = control_program_specs()
    normalized = tuple(expected_specs if specs is None else specs)
    if len(normalized) != CONTROL_PROGRAM_COUNT or len({str(spec.get("id", "")) for spec in normalized}) != CONTROL_PROGRAM_COUNT:
        raise ValueError("identity control program count or IDs changed")
    expected_lengths = {11: 0, 19: 0, 35: 0}
    for index, spec in enumerate(normalized):
        expected = expected_specs[index]
        for key in ("id", "suite", "prefix", "cell", "repeat", "suffix", "length"):
            if spec.get(key) != expected[key]:
                raise ValueError("identity control program binding changed")
        program = _strict_program(tuple(spec.get("program", ())))
        if program != tuple(expected["program"]):
            raise ValueError("identity control program binding changed")
        if len(program) not in expected_lengths or int(spec.get("length", -1)) != len(program):
            raise ValueError("identity control program length changed")
        expected_lengths[len(program)] += 1
        cell_name = str(spec.get("cell", ""))
        cell = dict(IDENTITY_CELLS).get(cell_name)
        if cell is None or tuple(program[2:-1]) != cell * int(spec.get("repeat", -1)):
            raise ValueError("identity control cell binding changed")
        for state in dsl.STATE_ORDER:
            current = state
            for opcode in cell:
                current = dsl.execute_program((opcode,), current)
            if current != state:
                raise ValueError("identity control cell is not semantically identity")
    if expected_lengths != {11: CONTROL_PROGRAMS_PER_LENGTH, 19: CONTROL_PROGRAMS_PER_LENGTH, 35: CONTROL_PROGRAMS_PER_LENGTH}:
        raise ValueError("identity control length coverage changed")
    return tuple(dict(spec) for spec in normalized)


__all__ = [
    "CARRY_GATE_PARAMETER_NAMES",
    "CONTROL_LENGTHS",
    "CONTROL_PREFIXES",
    "CONTROL_PROGRAM_COUNT",
    "CONTROL_REPEATS",
    "CONTROL_SUFFIXES",
    "CarryGate",
    "GATE_INITIAL_LOGIT",
    "GATE_INITIAL_VALUE",
    "GATE_PARAMETER_COUNT",
    "GATED_ADAPTER_PARAMETER_COUNT",
    "GATED_ARCHITECTURE_ID",
    "IDENTITY_CELLS",
    "QA_ACCOUNTING",
    "SCIENCE_ACCOUNTING",
    "TRAINING_BATCH_LENGTHS",
    "TRAINING_BATCH_SIZE",
    "TRAINING_NATIVE_STEPS_PER_ARM",
    "TRAINING_POSITIONS_PER_ARM",
    "TRAINING_UPDATES_PER_ARM",
    "append_carry_gate_optimizer_group",
    "apply_gated_carry",
    "attach_carry_gate",
    "blend_carry_state",
    "carry_gate_identity",
    "control_program_specs",
    "gated_carry_update",
    "gated_latent_slots_forward",
    "gated_slot_write",
    "gated_carry_hooks",
    "initialize_carry_gate",
    "make_carry_gate",
    "training_accounting",
    "validate_control_program_specs",
    "validate_carry_gate_identity",
    "validate_optimizer_carry_gate_association",
]
