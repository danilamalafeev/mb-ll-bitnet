"""Bounded QA runner for the registered gated carry package.

The ``--qa`` mode loads the accepted latent local-2000 endpoint independently
for arms A and B, performs the fixed two-example L1/L2 snapshot-resume test,
and stops.  Science preparation and execution are intentionally absent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted
from scripts import pc_latent_slots_runtime as accepted_qa_runtime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_gated_carry_v1" / "qa"
SCIENCE_MANIFEST = diagnostic_runtime.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic_runtime.SCIENCE_ENDPOINT
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_GATED_CARRY_PROTOCOL.md"
PURE_HELPER = ROOT / "scripts" / "pc_gated_carry.py"
PURE_TESTS = ROOT / "tests" / "test_pc_gated_carry.py"
DIAGNOSTIC_RUNTIME = Path(diagnostic_runtime.__file__).resolve()
RUNTIME = Path(__file__).resolve()

SCHEMA = "pc_gated_carry_qa_v1"
SOURCE_SCHEMA = "pc_gated_carry_qa_source_binding_v1"
INPUT_SCHEMA = "pc_gated_carry_qa_input_freeze_v1"
SNAPSHOT_SCHEMA = "pc_gated_carry_qa_snapshot_v1"
ARM_A = "A"
ARM_B = "B"
QA_ARMS = (ARM_A, ARM_B)
QA_ARM_LABELS = {ARM_A: "unhooked", ARM_B: "gated"}
QA_CASES_PER_CALL = 2
QA_BATCHES = 2
QA_CALLS_PER_ARM = 3
QA_CALLS = 6
QA_CASES = 12
QA_POSITIONS = 20
QA_NATIVE_STEPS = 160
QA_UPDATES = 6
QA_DESERIALIZATIONS = 8
QA_SNAPSHOT_LOADS = 2
PURE_HELPER_SHA256 = "4A35CB641D087E98531F3DF50A76B478395B06EBE75102592A2D7F0EAA9B677D".lower()
PURE_TESTS_SHA256 = "7DE5C0027721A5C78E7AE88AA7C4B90E605EF24404503B100413E8485D389962".lower()


def _resolve(path: Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(child) for child in value)
    return deepcopy(value)


def _atomic_json(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision {temporary}")
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision {temporary}")
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty gated QA output {path}")
    path.mkdir(parents=True, exist_ok=True)


def _new_counter() -> dict[str, Any]:
    phase = {
        "attempted_endpoint_loads": 0,
        "completed_endpoint_loads": 0,
        "attempted_parent_loads": 0,
        "completed_parent_loads": 0,
        "attempted_endpoint_restores": 0,
        "completed_endpoint_restores": 0,
        "attempted_snapshot_loads": 0,
        "completed_snapshot_loads": 0,
        "attempted_underlying_deserializations": 0,
        "completed_underlying_deserializations": 0,
        "attempted_updates": 0,
        "completed_updates": 0,
        "attempted_forwards": 0,
        "completed_forwards": 0,
        "attempted_cases": 0,
        "completed_cases": 0,
        "attempted_readout_positions": 0,
        "completed_readout_positions": 0,
        "attempted_native_steps": 0,
        "completed_native_steps": 0,
        "attempted_backwards": 0,
        "completed_backwards": 0,
        "attempted_optimizer_steps": 0,
        "completed_optimizer_steps": 0,
        "failures": 0,
    }
    return {
        "schema": "pc_gated_carry_qa_accounting_v1",
        **{key: value for key, value in phase.items() if key != "failures"},
        "failures": [],
        "timings_seconds": {},
        "phase_counts": {
            "load": {key: value for key, value in phase.items() if key in {
                "attempted_endpoint_loads", "completed_endpoint_loads", "attempted_parent_loads",
                "completed_parent_loads", "attempted_endpoint_restores", "completed_endpoint_restores",
                "attempted_snapshot_loads", "completed_snapshot_loads", "attempted_underlying_deserializations",
                "completed_underlying_deserializations", "failures",
            }},
            "training": {key: value for key, value in phase.items() if key in {
                "attempted_updates", "completed_updates", "attempted_forwards", "completed_forwards",
                "attempted_cases", "completed_cases", "attempted_readout_positions", "completed_readout_positions",
                "attempted_native_steps", "completed_native_steps", "attempted_backwards", "completed_backwards",
                "attempted_optimizer_steps", "completed_optimizer_steps", "failures",
            }},
        },
    }


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    counter["failures"].append({"kind": kind, "phase": phase, "type": type(exc).__name__, "message": str(exc)})
    counter["phase_counts"][phase]["failures"] += 1


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        counter["timings_seconds"][name] = float(counter["timings_seconds"].get(name, 0.0)) + time.perf_counter() - started
        _flush(counter, sink)


def _phase_add(counter: dict[str, Any], phase: str, key: str, amount: int = 1) -> None:
    counter[key] += amount
    counter["phase_counts"][phase][key] += amount


def _source_binding(*, root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "runtime": RUNTIME,
        "gated_helper": _resolve(PURE_HELPER, root),
        "gated_tests": _resolve(PURE_TESTS, root),
        "diagnostic_runtime": DIAGNOSTIC_RUNTIME,
        "protocol": _resolve(PROTOCOL, root),
        "register_e15": _resolve(Path("looped_bitnet/register_e15.py"), root),
    }
    files: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"source binding is missing: {path}")
        digest = _sha256(path)
        if name == "gated_helper" and digest != PURE_HELPER_SHA256:
            raise ValueError("pure gated helper bytes changed")
        if name == "gated_tests" and digest != PURE_TESTS_SHA256:
            raise ValueError("pure gated tests bytes changed")
        files[name] = {"path": _relative(path, root), "sha256": digest}
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, Mapping) or not inventory.get("digest"):
        raise ValueError("accepted science source inventory is missing")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "accepted_science_source_inventory": dict(inventory),
        "accepted_evidence": dict(evidence),
        "runtime_settings": dict(settings),
        "manifest_digest": _sha256(_resolve(SCIENCE_MANIFEST, root)),
        "endpoint": dict(evidence.get("endpoint", {})),
        "architecture_ids": {ARM_A: latent.ARCHITECTURE_ID, ARM_B: gated.GATED_ARCHITECTURE_ID},
    }
    binding["digest"] = science._digest(binding)
    return binding


def _example_record(example: dsl.RegisterExample) -> dict[str, Any]:
    return {
        "x": int(example.x),
        "y": int(example.y),
        "program": list(example.program),
        "target_trace": [list(pair) for pair in example.targets],
    }


def _fixture_batches(manifest: Mapping[str, Any], root: Path) -> tuple[list[list[dsl.RegisterExample]], Mapping[str, Any]]:
    _scope, _evidence, batches = science._load_frozen_data(manifest, root)
    if len(batches) < QA_BATCHES:
        raise ValueError("accepted science stream has fewer than two batches")
    selected = [list(batches[index][:QA_CASES_PER_CALL]) for index in range(QA_BATCHES)]
    if any(len(batch) != QA_CASES_PER_CALL for batch in selected):
        raise ValueError("accepted science QA prefix has fewer than two examples")
    lengths = [len(batch[0].program) for batch in selected]
    if lengths != [1, 2] or any(len({len(example.program) for example in batch}) != 1 for batch in selected):
        raise ValueError("accepted science QA prefix is not L1 then L2")
    stream = manifest.get("stream")
    if not isinstance(stream, Mapping):
        raise ValueError("accepted science stream binding is missing")
    return selected, {
        "path": str(stream.get("path", "")),
        "sha256": str(stream.get("sha256", "")),
        "stream_digest": str(stream.get("stream_digest", "")),
        "target_digest": str(stream.get("target_digest", "")),
        "batch_count": int(stream.get("batch_count", -1)),
        "batch_size": int(stream.get("batch_size", -1)),
    }


def _fixture(*, batches: Sequence[Sequence[dsl.RegisterExample]], source_binding: Mapping[str, Any], manifest: Mapping[str, Any], settings: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if len(batches) != QA_BATCHES or any(len(batch) != QA_CASES_PER_CALL for batch in batches):
        raise ValueError("gated QA fixture batch count changed")
    lengths = [len(batch[0].program) for batch in batches]
    if lengths != [1, 2] or any(len({len(example.program) for example in batch}) != 1 for batch in batches):
        raise ValueError("gated QA fixture must be homogeneous L1/L2")
    raw_batches = [[_example_record(example) for example in batch] for batch in batches]
    result: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "immutable": True,
        "programs": [[list(example.program) for example in batch] for batch in batches],
        "batch_lengths": lengths,
        "batch_size": QA_CASES_PER_CALL,
        "batches": raw_batches,
        "manifest": {"path": _relative(_resolve(SCIENCE_MANIFEST, root), root), "sha256": _sha256(_resolve(SCIENCE_MANIFEST, root))},
        "source_binding_digest": source_binding["digest"],
        "runtime": dict(settings),
        "budget": {
            "forwards": QA_CALLS,
            "cases": QA_CASES,
            "positions": QA_POSITIONS,
            "native_steps": QA_NATIVE_STEPS,
            "backwards": QA_UPDATES,
            "optimizer_updates": QA_UPDATES,
            "underlying_deserializations": QA_DESERIALIZATIONS,
        },
        "accepted_stream_digest": manifest.get("stream", {}).get("stream_digest"),
        "accepted_target_digest": manifest.get("stream", {}).get("target_digest"),
    }
    result["fixture_digest"] = science._digest(result)
    return result


def _fixture_from_records(fixture: Mapping[str, Any]) -> list[list[dsl.RegisterExample]]:
    raw_batches = fixture.get("batches")
    if not isinstance(raw_batches, list) or len(raw_batches) != QA_BATCHES:
        raise ValueError("gated QA fixture batch count changed")
    batches: list[list[dsl.RegisterExample]] = []
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, list) or len(raw_batch) != QA_CASES_PER_CALL:
            raise ValueError("gated QA fixture case count changed")
        batch: list[dsl.RegisterExample] = []
        for raw in raw_batch:
            if not isinstance(raw, Mapping):
                raise ValueError("gated QA fixture example malformed")
            program = tuple(str(opcode) for opcode in raw.get("program", ()))
            example = dsl.RegisterExample(int(raw["x"]), int(raw["y"]), program)
            if raw.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("gated QA fixture target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("gated QA fixture batch is not homogeneous")
        batches.append(batch)
    lengths = [len(batch[0].program) for batch in batches]
    if lengths != [1, 2]:
        raise ValueError("gated QA fixture program prefix changed")
    if science._digest({key: fixture.get(key) for key in fixture if key != "fixture_digest"}) != fixture.get("fixture_digest"):
        raise ValueError("gated QA fixture digest changed")
    return batches


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != QA_CASES_PER_CALL:
        raise ValueError("gated QA batch must contain exactly two examples")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("gated QA batch must be homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _validate_program_ops(programs: Sequence[Sequence[str]], ops: torch.Tensor) -> tuple[tuple[str, ...], ...]:
    expected_programs = tuple(tuple(str(opcode) for opcode in program) for program in programs)
    if len(expected_programs) != QA_CASES_PER_CALL or any(not program or any(opcode not in dsl.OPS for opcode in program) for program in expected_programs):
        raise ValueError("QA program contains an unknown opcode")
    if len({len(program) for program in expected_programs}) != 1:
        raise ValueError("QA programs are not homogeneous")
    if ops.ndim != 2 or tuple(ops.shape) != (QA_CASES_PER_CALL, len(expected_programs[0])) or ops.dtype is not torch.long:
        raise ValueError("QA program-to-ops tensor shape or dtype changed")
    expected_rows = [[dsl.OP_TO_ID[opcode] for opcode in program] for program in expected_programs]
    if ops.detach().cpu().tolist() != expected_rows:
        raise ValueError("QA program-to-ops binding changed")
    return expected_programs


def _adapter_identity(adapter: latent.LatentSlotAdapter, arm: str) -> dict[str, Any]:
    names = [name for name, _ in adapter.named_parameters()]
    expected = list(latent.ADAPTER_PARAMETER_NAMES) if arm == ARM_A else list(latent.ADAPTER_PARAMETER_NAMES) + list(gated.CARRY_GATE_PARAMETER_NAMES)
    if names != expected:
        raise ValueError(f"{arm} adapter parameter names/order changed")
    return {
        "schema": "pc_gated_carry_adapter_identity_v1",
        "architecture_id": latent.ARCHITECTURE_ID if arm == ARM_A else gated.GATED_ARCHITECTURE_ID,
        "parameter_names": names,
        "parameter_shapes": [list(parameter.shape) for parameter in adapter.parameters()],
        "parameter_count": adapter.parameter_count(),
        "state_digest": science._digest(adapter.state_dict()),
    }


def _arm_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str) -> dict[str, Any]:
    if arm not in QA_ARMS:
        raise ValueError("unknown gated QA arm")
    identity = {
        "arm": arm,
        "label": QA_ARM_LABELS[arm],
        "architecture_id": latent.ARCHITECTURE_ID if arm == ARM_A else gated.GATED_ARCHITECTURE_ID,
        "model_parameter_names": list(model.state_dict()),
        "adapter": _adapter_identity(adapter, arm),
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }
    expected_groups = 2 if arm == ARM_A else 3
    if identity["optimizer_group_count"] != expected_groups:
        raise ValueError(f"{arm} optimizer group count changed")
    if arm == ARM_B and identity["optimizer_group_param_names"][-1] != list(gated.CARRY_GATE_PARAMETER_NAMES):
        raise ValueError("gated optimizer group names changed")
    if arm == ARM_A and "carry_gate.weight" in identity["adapter"]["parameter_names"]:
        raise ValueError("unhooked arm unexpectedly has a gate")
    if arm == ARM_B and not isinstance(getattr(adapter, "carry_gate", None), gated.CarryGate):
        raise ValueError("gated arm is missing carry_gate")
    return identity


def _validate_arm_identity(identity: Mapping[str, Any], *, arm: str) -> None:
    if identity.get("arm") != arm or identity.get("label") != QA_ARM_LABELS[arm]:
        raise ValueError("QA arm identity changed")
    expected_architecture = latent.ARCHITECTURE_ID if arm == ARM_A else gated.GATED_ARCHITECTURE_ID
    if identity.get("architecture_id") != expected_architecture:
        raise ValueError("QA arm architecture identity changed")
    adapter = identity.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("QA arm adapter identity is missing")
    expected_names = list(latent.ADAPTER_PARAMETER_NAMES) if arm == ARM_A else list(latent.ADAPTER_PARAMETER_NAMES) + list(gated.CARRY_GATE_PARAMETER_NAMES)
    if adapter.get("parameter_names") != expected_names:
        raise ValueError("QA arm adapter gate inventory changed")
    if identity.get("optimizer_group_count") != (2 if arm == ARM_A else 3):
        raise ValueError("QA arm optimizer group count changed")
    groups = identity.get("optimizer_group_param_names")
    if not isinstance(groups, list) or (arm == ARM_B and groups[-1] != list(gated.CARRY_GATE_PARAMETER_NAMES)):
        raise ValueError("QA arm optimizer name association changed")


def _gradient_status(gate: gated.CarryGate | None) -> dict[str, Any]:
    if gate is None:
        return {"present": False, "finite": True, "nonzero": False}
    gradients = [parameter.grad for parameter in gate.parameters()]
    present = any(gradient is not None for gradient in gradients)
    if not present:
        return {"present": False, "finite": True, "nonzero": False}
    return {
        "present": True,
        "finite": all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None),
        "nonzero": any(bool(torch.count_nonzero(gradient) > 0) for gradient in gradients if gradient is not None),
    }


def _run_update(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> float:
    if arm not in QA_ARMS:
        raise ValueError("unknown gated QA arm")
    length = len(batch[0].program)
    cases = len(batch)
    phase = "training"
    for key, amount in (("attempted_updates", 1), ("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, phase, key, amount)
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with _timed(counter, f"forward_{arm}_L{length}", sink), torch.autocast(device_type=device.type, enabled=False):
            _validate_program_ops([example.program for example in batch], ops)
            if arm == ARM_A:
                logits_x, logits_y = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops)
            else:
                logits_x, logits_y = gated.gated_latent_slots_forward(model, adapter, x_bits, y_bits, ops)
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not bool(torch.isfinite(logits_x).all() and torch.isfinite(logits_y).all()):
            raise ValueError("gated QA logits are malformed or nonfinite")
    except BaseException as exc:
        _failure(counter, "forward", exc, phase=phase)
        _flush(counter, sink)
        raise
    for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, phase, key, amount)
    _flush(counter, sink)
    try:
        loss = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("gated QA loss is nonfinite")
    except BaseException as exc:
        _failure(counter, "loss", exc, phase=phase)
        _flush(counter, sink)
        raise
    _phase_add(counter, phase, "attempted_backwards")
    try:
        with _timed(counter, f"backward_{arm}_L{length}", sink):
            loss.backward()
    except BaseException as exc:
        _failure(counter, "backward", exc, phase=phase)
        _flush(counter, sink)
        raise
    _phase_add(counter, phase, "completed_backwards")
    try:
        with _timed(counter, f"gradient_clip_{arm}_L{length}", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _failure(counter, "gradient_clip", exc, phase=phase)
        _flush(counter, sink)
        raise
    _phase_add(counter, phase, "attempted_optimizer_steps")
    try:
        with _timed(counter, f"optimizer_step_{arm}_L{length}", sink):
            optimizer.step()
    except BaseException as exc:
        _failure(counter, "optimizer_step", exc, phase=phase)
        _flush(counter, sink)
        raise
    _phase_add(counter, phase, "completed_optimizer_steps")
    _phase_add(counter, phase, "completed_updates")
    _flush(counter, sink)
    return float(loss.detach().item())


def _state_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = accepted_qa_runtime._state_identity(model, adapter, optimizer)
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in [*model.named_parameters(), *[(f"adapter.{name}", value) for name, value in adapter.named_parameters()]]
    }
    state["gradient_digest"] = science._digest(gradients)
    return state


def _adapter_state_structure(adapter: latent.LatentSlotAdapter, arm: str) -> dict[str, Any]:
    return {
        "names": [name for name, _ in adapter.named_parameters()],
        "shapes": [list(parameter.shape) for parameter in adapter.parameters()],
        "count": adapter.parameter_count(),
        "architecture_id": latent.ARCHITECTURE_ID if arm == ARM_A else gated.GATED_ARCHITECTURE_ID,
    }


def _snapshot_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema", "committed", "complete", "arm", "architecture_id", "manifest_sha256", "endpoint_sha256",
        "source_binding_digest", "fixture_digest", "parent_identity", "local_update", "absolute_update", "next_batch_index",
        "arm_identity", "adapter_identity", "adapter_initialization", "model_state_dict", "adapter_state_dict",
        "optimizer_state_dict", "optimizer_metadata", "cpu_rng_state", "cuda_rng_state", "training_mode",
        "adapter_training_mode", "model_parameter_names", "adapter_parameter_names", "optimizer_group_param_names",
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
    )
    return {key: payload[key] for key in keys}


def _rng_states() -> tuple[torch.Tensor, list[torch.Tensor]]:
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda


def _snapshot_payload(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, parent_identity: Mapping[str, Any], source_binding_digest: str, fixture_digest: str, manifest_sha256: str, endpoint_sha256: str, adapter_initialization: Mapping[str, Any], local_update: int = 1, next_batch_index: int = 1) -> dict[str, Any]:
    if local_update != 1 or next_batch_index != 1:
        raise ValueError("gated QA snapshot boundary changed")
    adapter_identity = _adapter_identity(adapter, arm)
    arm_identity = _arm_identity(model, adapter, optimizer, arm)
    model_state = _cpu_copy(model.state_dict())
    adapter_state = _cpu_copy(adapter.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng, cuda_rng = _rng_states()
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "committed": True,
        "complete": True,
        "arm": arm,
        "architecture_id": arm_identity["architecture_id"],
        "manifest_sha256": manifest_sha256,
        "endpoint_sha256": endpoint_sha256,
        "source_binding_digest": source_binding_digest,
        "fixture_digest": fixture_digest,
        "parent_identity": dict(parent_identity),
        "local_update": local_update,
        "absolute_update": diagnostic_runtime.ENDPOINT_ABSOLUTE_UPDATE + local_update,
        "next_batch_index": next_batch_index,
        "arm_identity": arm_identity,
        "adapter_identity": adapter_identity,
        "adapter_initialization": dict(adapter_initialization),
        "model_state_dict": model_state,
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_metadata": accepted_qa_runtime._optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }
    payload["model_digest"] = science._digest(model_state)
    payload["adapter_digest"] = science._digest(adapter_state)
    payload["optimizer_digest"] = science._digest(optimizer_state)
    payload["cpu_rng_digest"] = science._digest(cpu_rng)
    payload["cuda_rng_digest"] = science._digest(cuda_rng)
    payload["snapshot_digest"] = science._digest(_snapshot_core(payload))
    return payload


def _validate_snapshot_payload(payload: Mapping[str, Any], *, arm: str, parent_identity: Mapping[str, Any], source_binding_digest: str, fixture_digest: str, manifest_sha256: str, endpoint_sha256: str, expected_arm_identity: Mapping[str, Any]) -> None:
    required = set(_snapshot_core({key: None for key in _snapshot_core_keys()})) | {"snapshot_digest"}
    if any(key not in payload for key in required):
        raise ValueError("gated QA snapshot is incomplete")
    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("committed") is not True or payload.get("complete") is not True:
        raise ValueError("gated QA snapshot is incomplete")
    for key, expected in {
        "arm": arm,
        "architecture_id": expected_arm_identity.get("architecture_id"),
        "manifest_sha256": manifest_sha256,
        "endpoint_sha256": endpoint_sha256,
        "source_binding_digest": source_binding_digest,
        "fixture_digest": fixture_digest,
        "parent_identity": dict(parent_identity),
        "local_update": 1,
        "absolute_update": diagnostic_runtime.ENDPOINT_ABSOLUTE_UPDATE + 1,
        "next_batch_index": 1,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"gated QA snapshot binding changed: {key}")
    arm_identity = payload.get("arm_identity")
    if not isinstance(arm_identity, Mapping):
        raise ValueError("gated QA snapshot arm identity is missing")
    _validate_arm_identity(arm_identity, arm=arm)
    if arm_identity.get("adapter", {}).get("parameter_names") != expected_arm_identity.get("adapter", {}).get("parameter_names"):
        raise ValueError("gated QA snapshot adapter gate inventory changed")
    adapter_identity = payload.get("adapter_identity")
    if not isinstance(adapter_identity, Mapping) or adapter_identity.get("architecture_id") != expected_arm_identity.get("architecture_id"):
        raise ValueError("gated QA snapshot adapter identity changed")
    expected_names = expected_arm_identity.get("adapter", {}).get("parameter_names")
    if adapter_identity.get("parameter_names") != expected_names:
        raise ValueError("gated QA snapshot adapter parameter names changed")
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("gated QA snapshot adapter state is missing")
    gate_keys = {"carry_gate.weight", "carry_gate.bias"}
    if arm == ARM_B and not gate_keys.issubset(adapter_state):
        raise ValueError("gated QA snapshot is missing carry_gate state")
    if arm == ARM_A and any(key in adapter_state for key in gate_keys):
        raise ValueError("unhooked QA snapshot unexpectedly contains carry_gate state")
    if science._digest(payload["model_state_dict"]) != payload["model_digest"] or science._digest(adapter_state) != payload["adapter_digest"] or science._digest(payload["optimizer_state_dict"]) != payload["optimizer_digest"]:
        raise ValueError("gated QA snapshot state digest changed")
    if science._digest(payload["cpu_rng_state"]) != payload["cpu_rng_digest"] or science._digest(payload["cuda_rng_state"]) != payload["cuda_rng_digest"]:
        raise ValueError("gated QA snapshot RNG digest changed")
    if science._digest(_snapshot_core(payload)) != payload["snapshot_digest"]:
        raise ValueError("gated QA snapshot digest changed")


def _snapshot_core_keys() -> tuple[str, ...]:
    return (
        "schema", "committed", "complete", "arm", "architecture_id", "manifest_sha256", "endpoint_sha256",
        "source_binding_digest", "fixture_digest", "parent_identity", "local_update", "absolute_update", "next_batch_index",
        "arm_identity", "adapter_identity", "adapter_initialization", "model_state_dict", "adapter_state_dict",
        "optimizer_state_dict", "optimizer_metadata", "cpu_rng_state", "cuda_rng_state", "training_mode",
        "adapter_training_mode", "model_parameter_names", "adapter_parameter_names", "optimizer_group_param_names",
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
    )


def _load_snapshot(path: Path, *, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> Mapping[str, Any]:
    if Path(path).name.lower().endswith(".tmp") or not Path(path).is_file():
        raise ValueError("gated QA snapshot is not a committed file")
    _phase_add(counter, "load", "attempted_snapshot_loads")
    _flush(counter, sink)
    try:
        with diagnostic_runtime._count_deserializations(counter, sink):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("gated QA snapshot is malformed")
        _phase_add(counter, "load", "completed_snapshot_loads")
        _flush(counter, sink)
        return payload
    except BaseException as exc:
        _failure(counter, "snapshot_load", exc, phase="load")
        _flush(counter, sink)
        raise


def _restore_snapshot(payload: Mapping[str, Any], *, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, parent_identity: Mapping[str, Any], source_binding_digest: str, fixture_digest: str, manifest_sha256: str, endpoint_sha256: str, expected_arm_identity: Mapping[str, Any], device: torch.device) -> None:
    _validate_snapshot_payload(payload, arm=arm, parent_identity=parent_identity, source_binding_digest=source_binding_digest, fixture_digest=fixture_digest, manifest_sha256=manifest_sha256, endpoint_sha256=endpoint_sha256, expected_arm_identity=expected_arm_identity)
    if payload.get("model_parameter_names") != list(model.state_dict()) or payload.get("adapter_parameter_names") != [name for name, _ in adapter.named_parameters()]:
        raise ValueError("gated QA snapshot parameter inventory changed")
    if payload.get("optimizer_group_param_names") != [list(group.get("param_names", [])) for group in optimizer.param_groups]:
        raise ValueError("gated QA snapshot optimizer name association changed")
    if payload.get("optimizer_metadata") != accepted_qa_runtime._optimizer_metadata(optimizer):
        raise ValueError("gated QA snapshot optimizer configuration changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    accepted_qa_runtime._optimizer_to_device(optimizer, device)
    torch.set_rng_state(payload["cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    model.train(bool(payload["training_mode"]))
    adapter.train(bool(payload["adapter_training_mode"]))
    actual = _state_identity(model, adapter, optimizer)
    for key in ("model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest"):
        if actual[key] != payload[key]:
            raise ValueError(f"gated QA snapshot restore mismatch: {key}")


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2,
        "attempted_parent_loads": 2, "completed_parent_loads": 2,
        "attempted_endpoint_restores": 2, "completed_endpoint_restores": 2,
        "attempted_snapshot_loads": QA_SNAPSHOT_LOADS, "completed_snapshot_loads": QA_SNAPSHOT_LOADS,
        "attempted_underlying_deserializations": QA_DESERIALIZATIONS,
        "completed_underlying_deserializations": QA_DESERIALIZATIONS,
        "attempted_updates": QA_UPDATES, "completed_updates": QA_UPDATES,
        "attempted_forwards": QA_CALLS, "completed_forwards": QA_CALLS,
        "attempted_cases": QA_CASES, "completed_cases": QA_CASES,
        "attempted_readout_positions": QA_POSITIONS, "completed_readout_positions": QA_POSITIONS,
        "attempted_native_steps": QA_NATIVE_STEPS, "completed_native_steps": QA_NATIVE_STEPS,
        "attempted_backwards": QA_UPDATES, "completed_backwards": QA_UPDATES,
        "attempted_optimizer_steps": QA_UPDATES, "completed_optimizer_steps": QA_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"gated QA accounting mismatch: {key}={counter.get(key)} expected {value}")
    if counter.get("failures"):
        raise ValueError("gated QA recorded failures")


def _run_arm(*, arm: str, batches: Sequence[Sequence[dsl.RegisterExample]], manifest: Mapping[str, Any], science_report: Mapping[str, Any], source_binding: Mapping[str, Any], fixture: Mapping[str, Any], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], out: Path, root: Path) -> dict[str, Any]:
    arm_out = out / QA_ARM_LABELS[arm]
    arm_out.mkdir(parents=True, exist_ok=False)
    model, adapter, optimizer, endpoint_payload = diagnostic_runtime._load_endpoint(
        manifest=manifest,
        report=science_report,
        endpoint=_resolve(SCIENCE_ENDPOINT, root),
        manifest_path=_resolve(SCIENCE_MANIFEST, root),
        source_binding_digest=str(manifest["source_inventory"]["digest"]),
        device=device,
        counter=counter,
        sink=sink,
        root=root,
    )
    adapter_initialization = dict(endpoint_payload["adapter_initialization"])
    if arm == ARM_B:
        gate, gate_metadata = gated.attach_carry_gate(adapter)
        gated.append_carry_gate_optimizer_group(optimizer, adapter)
        gated.validate_optimizer_carry_gate_association(optimizer, adapter)
    else:
        gate = None
        gate_metadata = None
    model.train(True)
    adapter.train(True)
    identity = _arm_identity(model, adapter, optimizer, arm)
    _validate_arm_identity(identity, arm=arm)
    parent_identity = endpoint_payload.get("parent_identity")
    if not isinstance(parent_identity, Mapping):
        raise ValueError("accepted endpoint parent identity is missing")
    manifest_sha256 = _sha256(_resolve(SCIENCE_MANIFEST, root))
    endpoint_sha256 = diagnostic_runtime.ENDPOINT_SHA256
    loss_l1 = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[0], arm=arm, device=device, counter=counter, sink=sink)
    gate_l1 = _gradient_status(gate)
    if arm == ARM_B and gate_l1["present"]:
        raise ValueError("gated arm received a gate gradient from L1-only readouts")
    snapshot = _snapshot_payload(
        model=model, adapter=adapter, optimizer=optimizer, arm=arm, parent_identity=parent_identity,
        source_binding_digest=str(source_binding["digest"]), fixture_digest=str(fixture["fixture_digest"]),
        manifest_sha256=manifest_sha256, endpoint_sha256=endpoint_sha256,
        adapter_initialization=adapter_initialization,
    )
    snapshot_path = arm_out / "snapshot_l1.pt"
    with _timed(counter, f"snapshot_save_{arm}", sink):
        _atomic_torch(snapshot_path, snapshot, refuse=True)
    loss_uninterrupted = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], arm=arm, device=device, counter=counter, sink=sink)
    gate_l2_uninterrupted = _gradient_status(gate)
    if arm == ARM_B and (not gate_l2_uninterrupted["present"] or not gate_l2_uninterrupted["finite"] or not gate_l2_uninterrupted["nonzero"]):
        raise ValueError("gated arm L2 gate gradient is absent, nonfinite, or zero")
    uninterrupted = _state_identity(model, adapter, optimizer)
    loaded = _load_snapshot(snapshot_path, counter=counter, sink=sink)
    _restore_snapshot(
        loaded, model=model, adapter=adapter, optimizer=optimizer, arm=arm,
        parent_identity=parent_identity, source_binding_digest=str(source_binding["digest"]),
        fixture_digest=str(fixture["fixture_digest"]), manifest_sha256=manifest_sha256,
        endpoint_sha256=endpoint_sha256, expected_arm_identity=identity, device=device,
    )
    loss_reloaded = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], arm=arm, device=device, counter=counter, sink=sink)
    gate_l2_reloaded = _gradient_status(gate)
    if arm == ARM_B and (not gate_l2_reloaded["present"] or not gate_l2_reloaded["finite"] or not gate_l2_reloaded["nonzero"]):
        raise ValueError("gated arm reloaded L2 gate gradient is absent, nonfinite, or zero")
    reloaded = _state_identity(model, adapter, optimizer)
    compare_keys = (
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names",
        "optimizer_group_count", "optimizer_group_param_names", "gradient_digest",
    )
    equality = {key: uninterrupted[key] == reloaded[key] for key in compare_keys}
    equality["all_exact"] = all(equality.values())
    if not equality["all_exact"]:
        raise ValueError("gated QA reload next-update mismatch")
    record = {
        "schema": "pc_gated_carry_qa_arm_v1",
        "arm": arm,
        "label": QA_ARM_LABELS[arm],
        "status": "complete",
        "architecture_id": identity["architecture_id"],
        "arm_identity": identity,
        "parent_identity": dict(parent_identity),
        "adapter_initialization": adapter_initialization,
        "gate_initialization": gate_metadata,
        "snapshot": {
            "path": _relative(snapshot_path, root),
            "sha256": _sha256(snapshot_path),
            "snapshot_digest": snapshot["snapshot_digest"],
            "next_batch_index": 1,
            "committed": True,
            "complete": True,
        },
        "losses": [loss_l1, loss_uninterrupted, loss_reloaded],
        "gate_gradients": {"after_l1": gate_l1, "after_l2_uninterrupted": gate_l2_uninterrupted, "after_l2_reloaded": gate_l2_reloaded},
        "equality": equality,
        "calls": QA_CALLS_PER_ARM,
        "updates": QA_CALLS_PER_ARM,
        "cases": QA_CASES_PER_CALL * QA_CALLS_PER_ARM,
        "positions": len(batches[0][0].program) * QA_CASES_PER_CALL + len(batches[1][0].program) * QA_CASES_PER_CALL * 2,
        "native_steps": (len(batches[0][0].program) * QA_CASES_PER_CALL + len(batches[1][0].program) * QA_CASES_PER_CALL * 2) * latent.NATIVE_STEPS,
    }
    _atomic_json(arm_out / "report.json", record, refuse=True)
    return record


def run_qa(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = lambda value: _atomic_json(accounting_path, value)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "manifest": {"path": _relative(manifest_path, root), "sha256": None}, "budget": {"forwards": QA_CALLS, "cases": QA_CASES, "positions": QA_POSITIONS, "native_steps": QA_NATIVE_STEPS, "backwards": QA_UPDATES, "optimizer_updates": QA_UPDATES, "underlying_deserializations": QA_DESERIALIZATIONS}, "accounting": dict(counter), "arms": {}}
    model_loaded = False
    try:
        manifest, science_report, _scope, _final, evidence = diagnostic_runtime._validate_evidence(root, manifest_path)
        device, settings = diagnostic_runtime._load_runtime_settings(manifest)
        source_binding = _source_binding(root=root, manifest=manifest, evidence=evidence, settings=settings)
        batches, stream_evidence = _fixture_batches(manifest, root)
        fixture = _fixture(batches=batches, source_binding=source_binding, manifest=manifest, settings=settings, root=root)
        # Validate the exact recorded input again before constructing or loading a model.
        _fixture_from_records(fixture)
        input_freeze = {
            **fixture,
            "source_binding": dict(source_binding),
            "accepted_stream_evidence": dict(stream_evidence),
            "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
        }
        input_freeze["digest"] = science._digest(input_freeze)
        _atomic_json(out / "source_binding.json", source_binding, refuse=True)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report.update({
            "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
            "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]},
            "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": input_freeze["digest"]},
            "runtime": settings,
            "endpoint": evidence["endpoint"],
        })
        for arm in QA_ARMS:
            report["arms"][arm] = _run_arm(
                arm=arm, batches=batches, manifest=manifest, science_report=science_report,
                source_binding=source_binding, fixture=fixture, device=device, counter=counter,
                sink=sink, out=out, root=root,
            )
            model_loaded = True
        _validate_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "source_inventory_digest": manifest["source_inventory"]["digest"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        _failure(counter, "qa", exc, phase="training" if model_loaded else "load")
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_qa(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
