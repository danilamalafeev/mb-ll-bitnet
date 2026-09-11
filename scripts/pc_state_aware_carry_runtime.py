"""Bounded two-arm QA for the state-aware latent-slot writer.

The runner loads the accepted latent local-2000 endpoint independently for an
ordinary arm (A) and a state-aware arm (C), then checks exact next-update
snapshot resume equality.  It does not train beyond the three QA updates per
arm and contains no science or evaluation sweep.
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
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted
from scripts import pc_latent_slots_runtime as accepted_qa
from scripts import pc_state_aware_carry as aware


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_state_aware_carry_v1" / "qa"
SCIENCE_MANIFEST = diagnostic.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic.SCIENCE_ENDPOINT
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_STATE_AWARE_CARRY_PROTOCOL.md"
PURE_HELPER = ROOT / "scripts" / "pc_state_aware_carry.py"
PURE_TESTS = ROOT / "tests" / "test_pc_state_aware_carry.py"
RUNTIME = Path(__file__).resolve()

SCHEMA = "pc_state_aware_carry_qa_v1"
ACCOUNTING_SCHEMA = "pc_state_aware_carry_qa_accounting_v1"
ARM_A = "A"
ARM_C = "C"
QA_ARMS = (ARM_A, ARM_C)
QA_ARM_LABELS = {ARM_A: "ordinary_writer", ARM_C: "state_aware_writer"}
QA_CASES_PER_CALL = 2
QA_CALLS_PER_ARM = 3
QA_CALLS = 6
QA_CASES = 12
QA_POSITIONS = 20
QA_NATIVE_STEPS = 160
QA_UPDATES = 6
QA_DESERIALIZATIONS = 8
QA_SNAPSHOT_LOADS = 2
PURE_HELPER_SHA256 = "79e38ffba3fbc472706da15887d1ce66968aa15db4a8a9fe3e41266b9d45fd04"
PURE_TESTS_SHA256 = "4e099b4ac02be9da24f5863ed0ac7f7a8bc3c9a344074c8f80cdcc2e71e3e40e"


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
        raise FileExistsError(f"temporary collision: {temporary}")
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
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
        raise FileExistsError(f"temporary collision: {temporary}")
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty state-aware QA output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _new_counter() -> dict[str, Any]:
    counter = diagnostic._new_counter()
    counter["schema"] = ACCOUNTING_SCHEMA
    for key in ("attempted_snapshot_loads", "completed_snapshot_loads"):
        counter[key] = 0
    counter["phase_counts"]["qa"] = {
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
    return counter


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    counter.setdefault("failures", []).append({"kind": kind, "phase": phase, "type": type(exc).__name__, "message": str(exc)})
    if phase in counter.get("phase_counts", {}):
        counter["phase_counts"][phase]["failures"] += 1


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = counter.setdefault("timings_seconds", {})
        timings[name] = float(timings.get(name, 0.0)) + time.perf_counter() - started
        _flush(counter, sink)


def _source_binding(*, root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, path in {
        "runtime": RUNTIME,
        "helper": PURE_HELPER,
        "tests": PURE_TESTS,
        "protocol": PROTOCOL,
        "register_e15": root / "looped_bitnet" / "register_e15.py",
        "manifest": _resolve(SCIENCE_MANIFEST, root),
    }.items():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        files[name] = {"path": _relative(Path(path), root), "sha256": _sha256(Path(path))}
    if files["helper"]["sha256"] != PURE_HELPER_SHA256 or files["tests"]["sha256"] != PURE_TESTS_SHA256:
        raise ValueError("state-aware pure source bytes changed")
    result: dict[str, Any] = {
        "schema": "pc_state_aware_carry_qa_source_binding_v1",
        "accepted_science_source_inventory": manifest.get("source_inventory"),
        "files": files,
        "evidence": dict(evidence),
        "runtime_settings": dict(settings),
        "endpoint_local_update": diagnostic.ENDPOINT_LOCAL_UPDATE,
        "architecture_id": aware.STATE_AWARE_ARCHITECTURE_ID,
    }
    result["digest"] = science._digest(result)
    return result


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != QA_CASES_PER_CALL:
        raise ValueError("state-aware QA batch must contain exactly two examples")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("state-aware QA batch must be homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _fixture_batches(manifest: Mapping[str, Any], root: Path) -> tuple[list[list[dsl.RegisterExample]], dict[str, Any]]:
    _scope, source, stream = science._load_frozen_data(manifest, root)
    if len(stream) < 2:
        raise ValueError("accepted stream is too short for QA")
    batches = [list(stream[0][:QA_CASES_PER_CALL]), list(stream[1][:QA_CASES_PER_CALL])]
    if any(len(batch) != QA_CASES_PER_CALL for batch in batches):
        raise ValueError("accepted stream QA prefix is incomplete")
    if len({len(example.program) for batch in batches for example in batch}) != 2:
        raise ValueError("QA fixture must contain the first two distinct lengths")
    fixture = {
        "schema": "pc_state_aware_carry_qa_fixture_v1",
        "source_stream_digest": source["stream"]["stream_digest"],
        "batches": [[{
            "x": int(example.x),
            "y": int(example.y),
            "program": list(example.program),
            "target_trace": [list(pair) for pair in example.targets],
        } for example in batch] for batch in batches],
    }
    fixture["digest"] = science._digest(fixture)
    return batches, fixture


def _rng_states() -> tuple[torch.Tensor, list[torch.Tensor]]:
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda


def _state_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    cpu_rng, cuda_rng = _rng_states()
    return {
        "model_digest": science._digest(model.state_dict()),
        "adapter_digest": science._digest(adapter.state_dict()),
        "optimizer_digest": science._digest(optimizer.state_dict()),
        "cpu_rng_digest": science._digest(cpu_rng),
        "cuda_rng_digest": science._digest(cuda_rng),
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }


def _run_update(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> float:
    length = len(batch[0].program)
    cases = len(batch)
    update_number = counter["phase_counts"]["qa"].get("attempted_updates", 0) + 1
    phase = "qa"
    for key, amount in (("attempted_updates", 1), ("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
        counter[key] += amount
        counter["phase_counts"][phase][key] += amount
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with _timed(counter, f"{arm}_forward", sink), torch.autocast(device_type=device.type, enabled=False):
            if arm == ARM_A:
                logits_x, logits_y = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops)
            else:
                logits_x, logits_y = aware.state_aware_latent_slots_forward(model, adapter, x_bits, y_bits, ops)
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
            raise ValueError("state-aware QA logits are malformed or nonfinite")
    except BaseException as exc:
        _failure(counter, "forward", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
    for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
        counter["phase_counts"][phase][key] += amount
    _flush(counter, sink)
    try:
        with _timed(counter, f"{arm}_loss", sink):
            loss = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("state-aware QA loss is nonfinite")
    except BaseException as exc:
        _failure(counter, "loss", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["attempted_backwards"] += 1
    counter["phase_counts"][phase]["attempted_backwards"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, f"{arm}_backward", sink):
            loss.backward()
    except BaseException as exc:
        _failure(counter, "backward", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_backwards"] += 1
    counter["phase_counts"][phase]["completed_backwards"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, f"{arm}_gradient_clip", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _failure(counter, "gradient_clip", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["attempted_optimizer_steps"] += 1
    counter["phase_counts"][phase]["attempted_optimizer_steps"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, f"{arm}_optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        _failure(counter, "optimizer_step", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    counter["phase_counts"][phase]["completed_optimizer_steps"] += 1
    counter["phase_counts"][phase]["completed_updates"] += 1
    _flush(counter, sink)
    return float(loss.detach().item())


def _snapshot_payload(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_state = _cpu_copy(model.state_dict())
    adapter_state = _cpu_copy(adapter.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng, cuda_rng = _rng_states()
    payload: dict[str, Any] = {
        "schema": "pc_state_aware_carry_qa_snapshot_v1",
        "committed": True,
        "complete": True,
        "model_state_dict": model_state,
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": optimizer_state,
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }
    payload.update({
        "model_digest": science._digest(model_state),
        "adapter_digest": science._digest(adapter_state),
        "optimizer_digest": science._digest(optimizer_state),
        "cpu_rng_digest": science._digest(cpu_rng),
        "cuda_rng_digest": science._digest(cuda_rng),
    })
    payload["snapshot_digest"] = science._digest({key: value for key, value in payload.items() if key != "snapshot_digest"})
    return payload


def _load_snapshot(path: Path, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> Mapping[str, Any]:
    counter["attempted_snapshot_loads"] += 1
    counter["attempted_underlying_deserializations"] += 1
    _flush(counter, sink)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != "pc_state_aware_carry_qa_snapshot_v1" or payload.get("committed") is not True or payload.get("complete") is not True:
            raise ValueError("state-aware snapshot is malformed")
        counter["completed_snapshot_loads"] += 1
        counter["completed_underlying_deserializations"] += 1
        _flush(counter, sink)
        return payload
    except BaseException as exc:
        _failure(counter, "snapshot_load", exc, phase="load")
        _flush(counter, sink)
        raise


def _restore_snapshot(payload: Mapping[str, Any], model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    required = ("model_state_dict", "adapter_state_dict", "optimizer_state_dict", "cpu_rng_state", "cuda_rng_state", "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest", "snapshot_digest")
    if any(key not in payload for key in required):
        raise ValueError("state-aware snapshot fields are incomplete")
    if science._digest(payload["model_state_dict"]) != payload["model_digest"] or science._digest(payload["adapter_state_dict"]) != payload["adapter_digest"] or science._digest(payload["optimizer_state_dict"]) != payload["optimizer_digest"]:
        raise ValueError("state-aware snapshot state digest changed")
    if science._digest(payload["cpu_rng_state"]) != payload["cpu_rng_digest"] or science._digest(payload["cuda_rng_state"]) != payload["cuda_rng_digest"]:
        raise ValueError("state-aware snapshot RNG digest changed")
    core = {key: value for key, value in payload.items() if key != "snapshot_digest"}
    if science._digest(core) != payload["snapshot_digest"]:
        raise ValueError("state-aware snapshot digest changed")
    if payload["model_parameter_names"] != list(model.state_dict()) or payload["adapter_parameter_names"] != [name for name, _ in adapter.named_parameters()] or payload["optimizer_group_param_names"] != [list(group.get("param_names", [])) for group in optimizer.param_groups]:
        raise ValueError("state-aware snapshot parameter association changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    accepted_qa._optimizer_to_device(optimizer, device)
    torch.set_rng_state(payload["cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    model.train(bool(payload["training_mode"]))
    adapter.train(bool(payload["adapter_training_mode"]))
    actual = _state_identity(model, adapter, optimizer)
    for key in ("model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest"):
        if actual[key] != payload[key]:
            raise ValueError(f"state-aware snapshot restore mismatch: {key}")


def _load_arm(*, arm: str, manifest: Mapping[str, Any], science_report: Mapping[str, Any], manifest_path: Path, source_binding_digest: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], root: Path) -> tuple[torch.nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    model, adapter, optimizer, endpoint = diagnostic._load_endpoint(
        manifest=manifest,
        report=science_report,
        endpoint=_resolve(SCIENCE_ENDPOINT, root),
        manifest_path=manifest_path,
        source_binding_digest=source_binding_digest,
        device=device,
        counter=counter,
        sink=sink,
        root=root,
    )
    if arm == ARM_A:
        latent.validate_optimizer_adapter_association(optimizer, adapter)
        identity = latent.adapter_identity(adapter)
    elif arm == ARM_C:
        writer, metadata = aware.install_state_aware_writer(adapter, optimizer)
        if not writer.correction_is_zero() or metadata["correction_parameter_count"] != aware.CORRECTION_PARAMETER_COUNT:
            raise ValueError("state-aware correction initialization changed")
        aware.validate_optimizer_state_aware_association(optimizer, adapter)
        identity = aware.state_aware_adapter_identity(adapter)
    else:
        raise ValueError(f"unknown QA arm {arm}")
    model.train(True)
    adapter.train(True)
    return model, adapter, optimizer, dict(endpoint), identity


def _run_arm(*, arm: str, batches: Sequence[Sequence[dsl.RegisterExample]], manifest: Mapping[str, Any], science_report: Mapping[str, Any], manifest_path: Path, source_binding_digest: str, fixture: Mapping[str, Any], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], out: Path, root: Path) -> dict[str, Any]:
    arm_out = out / QA_ARM_LABELS[arm]
    arm_out.mkdir(parents=True, exist_ok=False)
    model, adapter, optimizer, endpoint, identity = _load_arm(
        arm=arm, manifest=manifest, science_report=science_report, manifest_path=manifest_path,
        source_binding_digest=source_binding_digest, device=device, counter=counter, sink=sink, root=root,
    )
    initial_state = _state_identity(model, adapter, optimizer)
    loss_l1 = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[0], arm=arm, device=device, counter=counter, sink=sink)
    snapshot = _snapshot_payload(model, adapter, optimizer)
    snapshot_path = arm_out / "snapshot_l1.pt"
    _atomic_torch(snapshot_path, snapshot, refuse=True)
    loss_uninterrupted = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], arm=arm, device=device, counter=counter, sink=sink)
    uninterrupted = _state_identity(model, adapter, optimizer)
    loaded = _load_snapshot(snapshot_path, counter, sink)
    _restore_snapshot(loaded, model, adapter, optimizer, device)
    loss_reloaded = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], arm=arm, device=device, counter=counter, sink=sink)
    reloaded = _state_identity(model, adapter, optimizer)
    equality = {key: uninterrupted[key] == reloaded[key] for key in (
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names",
        "optimizer_group_count", "optimizer_group_param_names",
    )}
    equality["all_exact"] = all(equality.values())
    if not equality["all_exact"]:
        raise ValueError(f"state-aware QA reload mismatch for arm {arm}")
    record = {
        "schema": "pc_state_aware_carry_qa_arm_v1",
        "arm": arm,
        "label": QA_ARM_LABELS[arm],
        "status": "complete",
        "endpoint": {"path": _relative(_resolve(SCIENCE_ENDPOINT, root), root), "sha256": _sha256(_resolve(SCIENCE_ENDPOINT, root)), "local_update": diagnostic.ENDPOINT_LOCAL_UPDATE},
        "adapter_identity": identity,
        "initial_state": initial_state,
        "snapshot": {"path": _relative(snapshot_path, root), "sha256": _sha256(snapshot_path), "snapshot_digest": loaded["snapshot_digest"], "committed": True, "complete": True},
        "losses": [loss_l1, loss_uninterrupted, loss_reloaded],
        "equality": equality,
        "calls": QA_CALLS_PER_ARM,
        "updates": QA_CALLS_PER_ARM,
        "cases": QA_CASES_PER_CALL * QA_CALLS_PER_ARM,
        "readout_positions": sum(len(batch[0].program) * QA_CASES_PER_CALL for batch in (batches[0], batches[1], batches[1])),
        "native_steps": sum(len(batch[0].program) * QA_CASES_PER_CALL * latent.NATIVE_STEPS for batch in (batches[0], batches[1], batches[1])),
    }
    _atomic_json(arm_out / "report.json", record, refuse=True)
    return record


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 2,
        "completed_endpoint_loads": 2,
        "attempted_snapshot_loads": 2,
        "completed_snapshot_loads": 2,
        "attempted_forwards": QA_CALLS,
        "completed_forwards": QA_CALLS,
        "attempted_cases": QA_CASES,
        "completed_cases": QA_CASES,
        "attempted_readout_positions": QA_POSITIONS,
        "completed_readout_positions": QA_POSITIONS,
        "attempted_native_steps": QA_NATIVE_STEPS,
        "completed_native_steps": QA_NATIVE_STEPS,
        "attempted_backwards": QA_UPDATES,
        "completed_backwards": QA_UPDATES,
        "attempted_optimizer_steps": QA_UPDATES,
        "completed_optimizer_steps": QA_UPDATES,
        "completed_updates": QA_UPDATES,
        "attempted_underlying_deserializations": QA_DESERIALIZATIONS,
        "completed_underlying_deserializations": QA_DESERIALIZATIONS,
    }
    for key, expected_value in expected.items():
        if counter.get(key) != expected_value:
            raise ValueError(f"state-aware QA accounting mismatch: {key}={counter.get(key)} expected {expected_value}")
    if counter.get("failures"):
        raise ValueError("state-aware QA recorded failures")


def run_qa(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = lambda current: _atomic_json(accounting_path, current)
    sink(counter)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "arms": {}, "budget": {"forwards": QA_CALLS, "cases": QA_CASES, "positions": QA_POSITIONS, "native_steps": QA_NATIVE_STEPS, "backwards": QA_UPDATES, "optimizer_updates": QA_UPDATES, "underlying_deserializations": QA_DESERIALIZATIONS}}
    try:
        manifest, science_report, _scope, _final, evidence = diagnostic._validate_evidence(root, manifest_path)
        device, settings = diagnostic._load_runtime_settings(manifest)
        binding = _source_binding(root=root, manifest=manifest, evidence=evidence, settings=settings)
        batches, fixture = _fixture_batches(manifest, root)
        _atomic_json(out / "source_binding.json", binding, refuse=True)
        _atomic_json(out / "fixture.json", fixture, refuse=True)
        report.update({
            "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
            "source_binding_digest": binding["digest"],
            "fixture": {"path": _relative(out / "fixture.json", root), "digest": fixture["digest"]},
            "runtime": settings,
            "endpoint": evidence["endpoint"],
        })
        for arm in QA_ARMS:
            record = _run_arm(
                arm=arm, batches=batches, manifest=manifest, science_report=science_report,
                manifest_path=manifest_path, source_binding_digest=str(manifest["source_inventory"]["digest"]),
                fixture=fixture, device=device, counter=counter, sink=sink, out=out, root=root,
            )
            report["arms"][arm] = record
        _validate_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "limitations": ["two-example snapshot QA only", "no scientific evaluation or efficacy claim", "state-aware correction starts at zero and is not a trained result"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        _failure(counter, "qa", exc, phase="qa" if "qa" in counter.get("phase_counts", {}) else "load")
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="run the bounded state-aware snapshot QA")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if not args.qa:
        parser.error("only --qa is supported by this bounded entry point")
    run_qa(out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
