"""Bounded 250-update screening for the state-aware latent carry.

The runner starts ordinary arm A and state-aware arm C independently from the
accepted latent local-2000 endpoint.  It reuses the frozen training stream,
evaluates the registered old and identity-control programs once at the end,
and runs the existing 16-forward identity-path audit.  This is a screening
boundary; it does not resume a previous run or authorize a full pilot.
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
from scripts import pc_gated_carry_science as gated_science
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic
from scripts import pc_latent_slots_runtime as latent_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted
from scripts import pc_state_aware_carry as aware
from scripts import pc_state_aware_carry_runtime as qa_runtime
from scripts import pc_state_transfer_audit as transfer_audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_state_aware_carry_v1" / "screening"
SCIENCE_MANIFEST = diagnostic.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic.SCIENCE_ENDPOINT
QA_ACCEPT = ROOT / "runs" / "pc_state_aware_carry_v1" / "qa_accept.json"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_STATE_AWARE_CARRY_PROTOCOL.md"
SCREENING_TESTS = ROOT / "tests" / "test_pc_state_aware_carry_screening.py"

SCHEMA = "pc_state_aware_carry_screening_v1"
ACCOUNTING_SCHEMA = "pc_state_aware_carry_screening_accounting_v1"
SOURCE_SCHEMA = "pc_state_aware_carry_screening_source_binding_v1"
INPUT_SCHEMA = "pc_state_aware_carry_screening_input_freeze_v1"
CHECKPOINT_SCHEMA = "pc_state_aware_carry_screening_checkpoint_v1"
PROGRAM_SCHEMA = "pc_state_aware_carry_screening_program_v1"
ARM_A = "A"
ARM_C = "C"
ARMS = (ARM_A, ARM_C)
ARM_LABELS = {ARM_A: "ordinary_writer", ARM_C: "state_aware_writer"}
TRAINING_UPDATES = 250
TRAINING_BATCH_SIZE = 64
TRAINING_BATCH_LENGTHS = tuple(int(value) for value in latent.FIXED_BATCH_LENGTHS[:TRAINING_UPDATES])
TRAINING_SUM_LENGTHS = sum(TRAINING_BATCH_LENGTHS)
CHECKPOINT_INDICES = (0, 125, 250)
ENDPOINT_ABSOLUTE_UPDATE = 42_000
OLD_PROGRAMS = 69
NEW_PROGRAMS = gated.CONTROL_PROGRAM_COUNT
EVAL_FORWARDS_PER_ARM = OLD_PROGRAMS + NEW_PROGRAMS
EVAL_CASES_PER_ARM = EVAL_FORWARDS_PER_ARM * len(dsl.STATE_ORDER)
EVAL_POSITIONS_PER_ARM = 781_056
EVAL_NATIVE_PER_ARM = EVAL_POSITIONS_PER_ARM * latent.NATIVE_STEPS
IDENTITY_PAIRS = transfer_audit.IDENTITY_PAIRS
AUDIT_FORWARDS_PER_ARM = len(IDENTITY_PAIRS) * 2
AUDIT_CASES_PER_ARM = AUDIT_FORWARDS_PER_ARM * len(dsl.STATE_ORDER)
AUDIT_POSITIONS_PER_ARM = 9_216
AUDIT_NATIVE_PER_ARM = AUDIT_POSITIONS_PER_ARM * latent.NATIVE_STEPS
QA_ACCEPT_SHA256 = "eeae02889ddc2a72410389b0d38d8fbb5879fc7e3f795a6f0a95c4c00fdd99cc"


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


def _digest(value: Any) -> str:
    return science._digest(value)


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
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
        raise FileExistsError(f"refusing non-empty screening output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _new_counter() -> dict[str, Any]:
    counter = diagnostic._new_counter()
    counter["schema"] = ACCOUNTING_SCHEMA
    counter["active_phase"] = "load"
    counter["attempted_checkpoints"] = 0
    counter["completed_checkpoints"] = 0
    template = {
        "attempted_forwards": 0, "completed_forwards": 0,
        "attempted_cases": 0, "completed_cases": 0,
        "attempted_readout_positions": 0, "completed_readout_positions": 0,
        "attempted_native_steps": 0, "completed_native_steps": 0,
        "failures": 0,
    }
    counter["phase_counts"] = {
        "load": dict(counter["phase_counts"]["load"]),
        "training": {**template, "attempted_updates": 0, "completed_updates": 0,
                      "attempted_backwards": 0, "completed_backwards": 0,
                      "attempted_optimizer_steps": 0, "completed_optimizer_steps": 0},
        "evaluation_final": dict(template),
        "state_audit": dict(template),
        "checkpoint_save": {"attempted": 0, "completed": 0, "failures": 0},
    }
    return counter


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


def _phase_add(counter: dict[str, Any], phase: str, key: str, amount: int = 1) -> None:
    counter[key] += amount
    counter["phase_counts"][phase][key] = counter["phase_counts"][phase].get(key, 0) + amount


def _phase_failure(counter: dict[str, Any], phase: str, kind: str, exc: BaseException, sink: Callable[[dict[str, Any]], None] | None) -> None:
    diagnostic._failure(counter, kind, exc, phase=phase)
    _flush(counter, sink)


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = counter.setdefault("timings_seconds", {})
        timings[name] = float(timings.get(name, 0.0)) + time.perf_counter() - started
        _flush(counter, sink)


def _file_binding(path: Path, root: Path = ROOT) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _validate_qa_accept(root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(QA_ACCEPT, root)
    if not path.is_file() or _sha256(path) != QA_ACCEPT_SHA256:
        raise ValueError("state-aware QA acceptance bytes changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or record.get("status") != "ACCEPT":
        raise ValueError("state-aware QA acceptance record is malformed")
    if record.get("next_authorized_step") != "Prepare and launch one bounded 250-update-per-arm A/C screening only after its source gate; stop at the saved screening report.":
        raise ValueError("state-aware QA authorization changed")
    output = _resolve(Path(str(record.get("output", ""))), root)
    files: dict[str, Any] = {"qa_accept": _file_binding(path, root)}
    for name, field in (("report", "report_sha256"), ("accounting", "accounting_sha256"), ("source_binding", "source_binding_sha256"), ("fixture", "fixture_sha256")):
        expected = str(record.get(field, "")).lower()
        artifact = output / ("accounting.json" if name == "accounting" else f"{name}.json")
        if not expected or not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"state-aware QA artifact changed: {name}")
        files[f"qa:{name}"] = _file_binding(artifact, root)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    accounting = json.loads((output / "accounting.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or accounting.get("failures"):
        raise ValueError("state-aware QA report is not complete")
    if any(report.get("arms", {}).get(arm, {}).get("equality", {}).get("all_exact") is not True for arm in ARMS):
        raise ValueError("state-aware QA exact-resume evidence is missing")
    expected_accounting = {"forwards": 6, "cases": 12, "positions": 20, "native_steps": 160, "backwards": 6, "optimizer_updates": 6, "underlying_deserializations": 8}
    if any(report.get("budget", {}).get(key) != value for key, value in expected_accounting.items()):
        raise ValueError("state-aware QA accounting changed")
    return {"record": dict(record), "files": files, "report": report, "accounting": accounting, "sha256": QA_ACCEPT_SHA256}


def _specs(scope: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_specs = gated_science._old_specs(scope)
    new_specs = gated_science._new_specs()
    if len(old_specs) != OLD_PROGRAMS or len(new_specs) != NEW_PROGRAMS:
        raise ValueError("screening program count changed")
    return old_specs, new_specs


def _compact_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = {"id": str(spec["id"]), "suite": str(spec["suite"]), "length": int(spec["length"]), "program": list(spec["program"]), "states": [list(state) for state in spec["states"]], "strata": list(spec["strata"])}
    for key in ("cell", "prefix", "repeat", "suffix"):
        if key in spec:
            result[key] = spec[key]
    return result


def _prefix_digest(batches: Sequence[Sequence[dsl.RegisterExample]]) -> str:
    if len(batches) < TRAINING_UPDATES:
        raise ValueError("screening stream is shorter than 250 batches")
    records: list[list[dict[str, Any]]] = []
    lengths: list[int] = []
    for index, batch in enumerate(batches[:TRAINING_UPDATES]):
        if len(batch) != TRAINING_BATCH_SIZE or not batch or len({len(example.program) for example in batch}) != 1:
            raise ValueError(f"screening batch {index} is not homogeneous 64-case data")
        length = len(batch[0].program)
        if length != TRAINING_BATCH_LENGTHS[index]:
            raise ValueError(f"screening stream length changed at batch {index}")
        lengths.append(length)
        records.append([{"x": int(example.x), "y": int(example.y), "program": list(example.program), "target_trace": [list(pair) for pair in example.targets]} for example in batch])
    return _digest({"batch_count": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": lengths, "records": records})


def _source_binding(*, root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], qa_accept: Mapping[str, Any], baseline: Mapping[str, Any], prefix_digest: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "screening_runtime": Path(__file__).resolve(),
        "screening_tests": SCREENING_TESTS,
        "state_aware_helper": root / "scripts" / "pc_state_aware_carry.py",
        "state_aware_tests": root / "tests" / "test_pc_state_aware_carry.py",
        "qa_runtime": root / "scripts" / "pc_state_aware_carry_runtime.py",
        "qa_runtime_tests": root / "tests" / "test_pc_state_aware_carry_runtime.py",
        "diagnostic_runtime": Path(diagnostic.__file__).resolve(),
        "accepted_science": root / "scripts" / "pc_latent_slots_science.py",
        "latent_helper": root / "scripts" / "pc_latent_slots.py",
        "latent_runtime": root / "scripts" / "pc_latent_slots_runtime.py",
        "accepted_scratchpad": root / "scripts" / "pc_learned_scratchpad.py",
        "accepted_scratchpad_runtime": root / "scripts" / "pc_learned_scratchpad_runtime.py",
        "register_e15": root / "looped_bitnet" / "register_e15.py",
        "protocol": _resolve(PROTOCOL, root),
        "manifest": _resolve(SCIENCE_MANIFEST, root),
        "endpoint": _resolve(SCIENCE_ENDPOINT, root),
        "qa_accept": _resolve(QA_ACCEPT, root),
    }
    files = {name: _file_binding(path, root) for name, path in paths.items()}
    source_clear = qa_accept["record"].get("source_code_clear", {})
    if files["state_aware_helper"]["sha256"] != str(source_clear.get("helper_sha256", "")).lower() or files["qa_runtime"]["sha256"] != str(source_clear.get("runtime_sha256", "")).lower():
        raise ValueError("state-aware accepted source bytes changed")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "accepted_science_source_inventory": manifest.get("source_inventory"),
        "accepted_evidence": dict(evidence),
        "accepted_qa": {"record": qa_accept["record"], "files": qa_accept["files"], "sha256": qa_accept["sha256"]},
        "baseline_digest": baseline["digest"],
        "training_prefix_digest": prefix_digest,
        "runtime_settings": dict(settings),
        "endpoint_local_update": diagnostic.ENDPOINT_LOCAL_UPDATE,
        "endpoint_absolute_update": ENDPOINT_ABSOLUTE_UPDATE,
    }
    binding["digest"] = _digest(binding)
    return binding


def _input_freeze(*, old_specs: Sequence[Mapping[str, Any]], new_specs: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], binding: Mapping[str, Any], prefix_digest: str, manifest_path: Path, settings: Mapping[str, Any], evidence: Mapping[str, Any], root: Path) -> dict[str, Any]:
    freeze: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "immutable": True,
        "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
        "protocol": _file_binding(_resolve(PROTOCOL, root), root),
        "arms": list(ARMS),
        "architecture_ids": {ARM_A: latent.ARCHITECTURE_ID, ARM_C: aware.STATE_AWARE_ARCHITECTURE_ID},
        "training": {"updates_per_arm": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": list(TRAINING_BATCH_LENGTHS), "sum_lengths": TRAINING_SUM_LENGTHS, "prefix_digest": prefix_digest},
        "checkpoints": list(CHECKPOINT_INDICES),
        "endpoint": {"local_update": diagnostic.ENDPOINT_LOCAL_UPDATE, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE, "sha256": _sha256(_resolve(SCIENCE_ENDPOINT, root))},
        "old_programs": [_compact_spec(spec) for spec in old_specs],
        "new_programs": [_compact_spec(spec) for spec in new_specs],
        "source_binding_digest": binding["digest"],
        "qa_accept_sha256": QA_ACCEPT_SHA256,
        "baseline_digest": baseline["digest"],
        "evidence": dict(evidence),
        "runtime": dict(settings),
    }
    freeze["digest"] = _digest(freeze)
    return freeze


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != TRAINING_BATCH_SIZE or not batch:
        raise ValueError("screening batch size changed")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("screening batch is not homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _run_update(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> float:
    if arm not in ARMS:
        raise ValueError("unknown screening arm")
    length = len(batch[0].program)
    cases = len(batch)
    counter["active_phase"] = "training"
    for key, amount in (("attempted_updates", 1), ("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, "training", key, amount)
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    forward = latent.latent_slots_forward if arm == ARM_A else aware.state_aware_latent_slots_forward
    try:
        with _timed(counter, "training_forward", sink), torch.autocast(device_type=device.type, enabled=False):
            logits_x, logits_y = forward(model, adapter, x_bits, y_bits, ops)
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or not bool(torch.isfinite(logits_x).all() and torch.isfinite(logits_y).all()):
            raise ValueError("screening logits are malformed or nonfinite")
    except BaseException as exc:
        _phase_failure(counter, "training", "forward", exc, sink)
        raise
    for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, "training", key, amount)
    _flush(counter, sink)
    try:
        with _timed(counter, "training_loss", sink):
            loss = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("screening loss is nonfinite")
    except BaseException as exc:
        _phase_failure(counter, "training", "loss", exc, sink)
        raise
    _phase_add(counter, "training", "attempted_backwards")
    _flush(counter, sink)
    try:
        with _timed(counter, "training_backward", sink):
            loss.backward()
    except BaseException as exc:
        _phase_failure(counter, "training", "backward", exc, sink)
        raise
    _phase_add(counter, "training", "completed_backwards")
    try:
        with _timed(counter, "training_gradient_clip", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _phase_failure(counter, "training", "gradient_clip", exc, sink)
        raise
    _phase_add(counter, "training", "attempted_optimizer_steps")
    _flush(counter, sink)
    try:
        with _timed(counter, "training_optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        _phase_failure(counter, "training", "optimizer_step", exc, sink)
        raise
    _phase_add(counter, "training", "completed_optimizer_steps")
    _phase_add(counter, "training", "completed_updates")
    _flush(counter, sink)
    return float(loss.detach().item())


def _adapter_identity(adapter: latent.LatentSlotAdapter, arm: str) -> dict[str, Any]:
    if arm == ARM_A:
        latent.validate_optimizer_adapter_association  # keep the accepted seam visible to source review
        return latent.adapter_identity(adapter)
    if arm == ARM_C:
        return aware.state_aware_adapter_identity(adapter)
    raise ValueError("unknown screening arm")


def _arm_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str) -> dict[str, Any]:
    adapter_id = _adapter_identity(adapter, arm)
    groups = [list(group.get("param_names", [])) for group in optimizer.param_groups]
    if len(groups) != 2:
        raise ValueError("screening optimizer group count changed")
    return {"arm": arm, "label": ARM_LABELS[arm], "architecture_id": adapter_id["architecture_id"], "model_parameter_names": list(model.state_dict()), "adapter": adapter_id, "optimizer_group_count": len(groups), "optimizer_group_param_names": groups}


def _load_arm(*, arm: str, manifest: Mapping[str, Any], science_report: Mapping[str, Any], manifest_path: Path, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], root: Path) -> tuple[torch.nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    model, adapter, optimizer, endpoint = diagnostic._load_endpoint(manifest=manifest, report=science_report, endpoint=_resolve(SCIENCE_ENDPOINT, root), manifest_path=manifest_path, source_binding_digest=str(manifest["source_inventory"]["digest"]), device=device, counter=counter, sink=sink, root=root)
    if arm == ARM_A:
        latent.validate_optimizer_adapter_association(optimizer, adapter)
    else:
        writer, metadata = aware.install_state_aware_writer(adapter, optimizer)
        if not writer.correction_is_zero() or metadata.get("correction_parameter_count") != aware.CORRECTION_PARAMETER_COUNT:
            raise ValueError("state-aware correction initialization changed")
        aware.validate_optimizer_state_aware_association(optimizer, adapter)
    model.train(True)
    adapter.train(True)
    return model, adapter, optimizer, dict(endpoint), _arm_identity(model, adapter, optimizer, arm)


def _checkpoint_payload(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, local_update: int, binding: Mapping[str, Any], freeze: Mapping[str, Any], endpoint_payload: Mapping[str, Any]) -> dict[str, Any]:
    if local_update not in CHECKPOINT_INDICES:
        raise ValueError("screening checkpoint boundary changed")
    model_state = _cpu_copy(model.state_dict())
    adapter_state = _cpu_copy(adapter.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    identity = _arm_identity(model, adapter, optimizer, arm)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA, "committed": True, "complete": True,
        "arm": arm, "architecture_id": identity["architecture_id"],
        "manifest_sha256": _sha256(_resolve(SCIENCE_MANIFEST, ROOT)), "endpoint_sha256": _sha256(_resolve(SCIENCE_ENDPOINT, ROOT)),
        "source_binding_digest": binding["digest"], "input_freeze_digest": freeze["digest"],
        "parent_identity": dict(endpoint_payload.get("parent_identity", {})),
        "local_update": local_update, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE + local_update, "next_batch_index": local_update,
        "arm_identity": identity, "adapter_identity": identity["adapter"],
        "adapter_initialization": dict(endpoint_payload.get("adapter_initialization", {})),
        "model_state_dict": model_state, "adapter_state_dict": adapter_state, "optimizer_state_dict": optimizer_state,
        "optimizer_metadata": latent_runtime._optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng, "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training), "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()), "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
        "model_digest": _digest(model_state), "adapter_digest": _digest(adapter_state), "optimizer_digest": _digest(optimizer_state),
        "cpu_rng_digest": _digest(cpu_rng), "cuda_rng_digest": _digest(cuda_rng),
    }
    payload["checkpoint_digest"] = _digest(payload)
    return payload


def _save_checkpoint(*, out: Path, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, local_update: int, binding: Mapping[str, Any], freeze: Mapping[str, Any], endpoint_payload: Mapping[str, Any], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], root: Path) -> dict[str, Any]:
    counter["active_phase"] = "checkpoint_save"
    counter["attempted_checkpoints"] += 1
    counter["phase_counts"]["checkpoint_save"]["attempted"] += 1
    _flush(counter, sink)
    path = out / ARM_LABELS[arm] / "checkpoints" / f"local{local_update}.pt"
    try:
        payload = _checkpoint_payload(model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=local_update, binding=binding, freeze=freeze, endpoint_payload=endpoint_payload)
        _atomic_torch(path, payload, refuse=True)
    except BaseException as exc:
        _phase_failure(counter, "checkpoint_save", "checkpoint_save", exc, sink)
        raise
    counter["completed_checkpoints"] += 1
    counter["phase_counts"]["checkpoint_save"]["completed"] += 1
    _flush(counter, sink)
    return {"local_update": local_update, "absolute_update": payload["absolute_update"], "next_batch_index": payload["next_batch_index"], "path": _relative(path, root), "sha256": _sha256(path), "checkpoint_digest": payload["checkpoint_digest"], "committed": True, "complete": True}


def _validate_logits(logits: Any, cases: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, tuple) or len(logits) != 2:
        raise ValueError("screening evaluation logits are malformed")
    x_logits, y_logits = logits
    expected = (cases, length, latent.SLOT_WIDTH)
    if tuple(x_logits.shape) != expected or tuple(y_logits.shape) != expected or x_logits.dtype is not torch.float32 or y_logits.dtype is not torch.float32 or not bool(torch.isfinite(x_logits).all() and torch.isfinite(y_logits).all()):
        raise ValueError("screening evaluation logits are malformed or nonfinite")
    return x_logits, y_logits


def _diagnostic_summary(diagnostics: Mapping[str, Any], length: int) -> dict[str, Any]:
    slot_inputs, slot_writes = diagnostics.get("slot_inputs"), diagnostics.get("slot_writes")
    if not isinstance(slot_inputs, list) or not isinstance(slot_writes, list) or len(slot_inputs) != length or len(slot_writes) != length:
        raise ValueError("screening latent diagnostics are incomplete")
    positions = []
    for position, (slot_input, slot_write) in enumerate(zip(slot_inputs, slot_writes), start=1):
        if not isinstance(slot_input, torch.Tensor) or not isinstance(slot_write, torch.Tensor) or not bool(torch.isfinite(slot_input).all() and torch.isfinite(slot_write).all()):
            raise ValueError("screening latent diagnostics are nonfinite")
        positions.append({"position": position, "input_l2_mean": float(torch.linalg.vector_norm(slot_input, dim=-1).mean().detach().cpu().item()), "write_l2_mean": float(torch.linalg.vector_norm(slot_write, dim=-1).mean().detach().cpu().item()), "finite": True})
    return {"positions": positions, "native_steps": latent.NATIVE_STEPS}


def _run_program(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, spec: Mapping[str, Any], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    counter["active_phase"] = "evaluation_final"
    inputs = gated_science._build_inputs(spec, device)
    program = inputs["program"]
    length = len(program)
    cases = len(inputs["examples"])
    for key, amount in (("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, "evaluation_final", key, amount)
    _flush(counter, sink)
    forward = latent.latent_slots_forward if arm == ARM_A else aware.state_aware_latent_slots_forward
    try:
        with _timed(counter, f"evaluation_{arm}_{spec['id']}", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            logits, diagnostics = forward(model, adapter, inputs["x_bits"], inputs["y_bits"], inputs["ops"], return_diagnostics=True)
            logits_x, logits_y = _validate_logits(logits, cases, length)
            diagnostic_summary = _diagnostic_summary(diagnostics, length)
        if arm == ARM_C:
            carry = diagnostics.get("state_aware_carry")
            if not isinstance(carry, Mapping) or carry.get("reader_calls") != length or carry.get("writer_calls") != length:
                raise ValueError("state-aware hook diagnostics changed")
            if len(adapter.reader._forward_pre_hooks) or len(adapter.writer._forward_pre_hooks):
                raise ValueError("state-aware forward left hooks installed")
        decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
        decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
        predictions = []
        for index, (state, stratum, example) in enumerate(zip(dsl.STATE_ORDER, spec["strata"], inputs["examples"])):
            target = [list(item) for item in example.targets]
            predicted = [[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)]
            predictions.append({"state": list(state), "stratum": stratum, "target_trace": target, "predicted_trace": predicted, **gated_science._trace_metrics(target, predicted)})
        for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
            _phase_add(counter, "evaluation_final", key, amount)
        _flush(counter, sink)
        row: dict[str, Any] = {"schema": PROGRAM_SCHEMA, "id": str(spec["id"]), "suite": str(spec["suite"]), "length": length, "program": list(program), "arm": arm, "phase": "final", "predictions": predictions, "writing_diagnostics": diagnostic_summary, "native_steps": latent.NATIVE_STEPS}
        for key in ("cell", "prefix", "repeat", "suffix"):
            if key in spec:
                row[key] = spec[key]
        return row
    except BaseException as exc:
        _phase_failure(counter, "evaluation_final", "evaluation_forward", exc, sink)
        raise


def _evaluate(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, specs: Sequence[Mapping[str, Any]], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], rows_dir: Path, row_index: list[int], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
        for spec in specs:
            row = _run_program(model=model, adapter=adapter, spec=spec, arm=arm, device=device, counter=counter, sink=sink)
            path = rows_dir / f"{row_index[0]:04d}_{arm}_final_{row['id']}.json"
            _atomic_json(path, row, refuse=True)
            row_index[0] += 1
            rows.append(row)
    return rows


def _audit_path(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, arm: str, program: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    full_program = tuple(program) + (transfer_audit.PROBE,)
    x_bits, y_bits, ops = transfer_audit._inputs(full_program, device)
    forward = latent.latent_slots_forward if arm == ARM_A else aware.state_aware_latent_slots_forward
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        logits, diagnostics = forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
    if not isinstance(diagnostics, Mapping) or len(diagnostics.get("slot_inputs", [])) != len(full_program) or len(diagnostics.get("slot_writes", [])) != len(full_program):
        raise ValueError("state audit diagnostics changed")
    if arm == ARM_C and (len(adapter.reader._forward_pre_hooks) or len(adapter.writer._forward_pre_hooks)):
        raise ValueError("state audit left state-aware hooks installed")
    return {"before_probe": diagnostics["slot_inputs"][-1].detach().float(), "initial": diagnostics["slot_inputs"][0].detach().float(), "probe_logits": logits[0][:, -1, :].detach().float()}


def _state_audit(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    before_model = _digest(model.state_dict())
    before_adapter = _digest(adapter.state_dict())
    states = [tuple(int(value) for value in state) for state in dsl.STATE_ORDER]
    pairs = []
    counter["active_phase"] = "state_audit"
    for spec in IDENTITY_PAIRS:
        path_results: list[dict[str, torch.Tensor]] = []
        for _program in (spec["left"], spec["right"]):
            length = len(_program) + 1
            for key, amount in (("attempted_forwards", 1), ("attempted_cases", len(states)), ("attempted_readout_positions", len(states) * length), ("attempted_native_steps", len(states) * length * latent.NATIVE_STEPS)):
                _phase_add(counter, "state_audit", key, amount)
            _flush(counter, sink)
            try:
                result = _audit_path(model, adapter, arm, _program, device)
            except BaseException as exc:
                _phase_failure(counter, "state_audit", "audit_forward", exc, sink)
                raise
            for key, amount in (("completed_forwards", 1), ("completed_cases", len(states)), ("completed_readout_positions", len(states) * length), ("completed_native_steps", len(states) * length * latent.NATIVE_STEPS)):
                _phase_add(counter, "state_audit", key, amount)
            _flush(counter, sink)
            path_results.append(result)
        left_result, right_result = path_results
        pairs.append({"id": spec["id"], "left": list(spec["left"]), "right": list(spec["right"]), "left_length": len(spec["left"]), "right_length": len(spec["right"]), "state_pair": transfer_audit._tensor_metrics(left_result["before_probe"], right_result["before_probe"]), "left_identity_drift": transfer_audit._tensor_metrics(left_result["before_probe"], left_result["initial"]), "right_identity_drift": transfer_audit._tensor_metrics(right_result["before_probe"], right_result["initial"]), "probe": transfer_audit._probe_metrics(left_result["probe_logits"], right_result["probe_logits"], states)})
    if _digest(model.state_dict()) != before_model or _digest(adapter.state_dict()) != before_adapter:
        raise ValueError("state audit mutated model or adapter")
    return {"arm": arm, "forwards": AUDIT_FORWARDS_PER_ARM, "pairs": pairs, "model_digest_preserved": True, "adapter_digest_preserved": True}


def _run_arm(*, arm: str, batches: Sequence[Sequence[dsl.RegisterExample]], old_specs: Sequence[Mapping[str, Any]], new_specs: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], science_report: Mapping[str, Any], manifest_path: Path, binding: Mapping[str, Any], freeze: Mapping[str, Any], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], out: Path, rows_dir: Path, row_index: list[int], root: Path) -> dict[str, Any]:
    arm_out = out / ARM_LABELS[arm]
    arm_out.mkdir(parents=True, exist_ok=False)
    model, adapter, optimizer, endpoint_payload, initial_identity = _load_arm(arm=arm, manifest=manifest, science_report=science_report, manifest_path=manifest_path, device=device, counter=counter, sink=sink, root=root)
    checkpoints = [_save_checkpoint(out=out, model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=0, binding=binding, freeze=freeze, endpoint_payload=endpoint_payload, counter=counter, sink=sink, root=root)]
    losses: list[float] = []
    for local_update, batch in enumerate(batches[:TRAINING_UPDATES], start=1):
        losses.append(_run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batch, arm=arm, device=device, counter=counter, sink=sink))
        if local_update in CHECKPOINT_INDICES[1:]:
            checkpoints.append(_save_checkpoint(out=out, model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=local_update, binding=binding, freeze=freeze, endpoint_payload=endpoint_payload, counter=counter, sink=sink, root=root))
    rows = _evaluate(model=model, adapter=adapter, optimizer=optimizer, specs=[*old_specs, *new_specs], arm=arm, device=device, counter=counter, sink=sink, rows_dir=rows_dir, row_index=row_index, root=root)
    audit = _state_audit(model=model, adapter=adapter, arm=arm, device=device, counter=counter, sink=sink)
    final_identity = _arm_identity(model, adapter, optimizer, arm)
    record = {"schema": f"{SCHEMA}_arm_v1", "status": "complete", "arm": arm, "label": ARM_LABELS[arm], "initial_identity": initial_identity, "final_identity": final_identity, "training": {"updates": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "sum_lengths": TRAINING_SUM_LENGTHS, "losses": losses, "first_loss": losses[0], "last_loss": losses[-1]}, "evaluation": {"rows": len(rows), "forwards": EVAL_FORWARDS_PER_ARM, "cases": EVAL_CASES_PER_ARM, "positions": sum(int(row["length"]) for row in rows) * len(dsl.STATE_ORDER), "native_steps": sum(int(row["length"]) for row in rows) * len(dsl.STATE_ORDER) * latent.NATIVE_STEPS}, "state_audit": audit, "checkpoints": checkpoints, "endpoint": {"path": _relative(_resolve(SCIENCE_ENDPOINT, root), root), "sha256": _sha256(_resolve(SCIENCE_ENDPOINT, root)), "local_update": diagnostic.ENDPOINT_LOCAL_UPDATE}}
    _atomic_json(arm_out / "report.json", record, refuse=True)
    return record


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2, "attempted_parent_loads": 2, "completed_parent_loads": 2, "attempted_endpoint_restores": 2, "completed_endpoint_restores": 2,
        "attempted_underlying_deserializations": 6, "completed_underlying_deserializations": 6,
        "attempted_forwards": 816, "completed_forwards": 816, "attempted_cases": 112896, "completed_cases": 112896,
        "attempted_readout_positions": 1692032, "completed_readout_positions": 1692032, "attempted_native_steps": 13536256, "completed_native_steps": 13536256,
        "attempted_updates": 500, "completed_updates": 500, "attempted_backwards": 500, "completed_backwards": 500, "attempted_optimizer_steps": 500, "completed_optimizer_steps": 500,
        "attempted_checkpoints": 6, "completed_checkpoints": 6,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"screening accounting mismatch: {key}={counter.get(key)} expected {value}")
    phases = counter.get("phase_counts", {})
    if phases.get("training", {}).get("completed_updates") != 500 or phases.get("evaluation_final", {}).get("completed_forwards") != 300 or phases.get("state_audit", {}).get("completed_forwards") != 16 or phases.get("checkpoint_save", {}).get("completed") != 6:
        raise ValueError("screening phase accounting mismatch")
    if counter.get("failures"):
        raise ValueError("screening recorded failures")


def run_screening(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = lambda current: _atomic_json(accounting_path, current)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "arms": {}, "budget": {"updates": 500, "forwards": 816, "cases": 112896, "positions": 1692032, "native_steps": 13536256, "backwards": 500, "optimizer_updates": 500, "underlying_deserializations": 6, "checkpoints": 6}}
    try:
        manifest, science_report, scope, final, evidence = diagnostic._validate_evidence(root, manifest_path)
        device, settings = diagnostic._load_runtime_settings(manifest)
        qa_accept = _validate_qa_accept(root)
        old_specs, new_specs = _specs(scope)
        baseline = gated_science._validate_old_baseline(final, old_specs)
        _scope, source, all_batches = science._load_frozen_data(manifest, root)
        prefix_digest = _prefix_digest(all_batches)
        binding = _source_binding(root=root, manifest=manifest, evidence=evidence, qa_accept=qa_accept, baseline=baseline, prefix_digest=prefix_digest, settings=settings)
        freeze = _input_freeze(old_specs=old_specs, new_specs=new_specs, baseline=baseline, binding=binding, prefix_digest=prefix_digest, manifest_path=manifest_path, settings=settings, evidence=evidence, root=root)
        _atomic_json(out / "source_binding.json", binding, refuse=True)
        _atomic_json(out / "input_freeze.json", freeze, refuse=True)
        _atomic_json(out / "stream_prefix.json", {"schema": "pc_state_aware_carry_screening_stream_prefix_v1", "batch_count": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": list(TRAINING_BATCH_LENGTHS), "sum_lengths": TRAINING_SUM_LENGTHS, "digest": prefix_digest}, refuse=True)
        rows_dir = out / "program_rows"
        rows_dir.mkdir(parents=True, exist_ok=False)
        row_index = [0]
        arm_records = {}
        for arm in ARMS:
            arm_records[arm] = _run_arm(arm=arm, batches=all_batches[:TRAINING_UPDATES], old_specs=old_specs, new_specs=new_specs, manifest=manifest, science_report=science_report, manifest_path=manifest_path, binding=binding, freeze=freeze, device=device, counter=counter, sink=sink, out=out, rows_dir=rows_dir, row_index=row_index, root=root)
        rows: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            rows[arm] = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(rows_dir.glob(f"*_{arm}_final_*.json"))]
        final_a, final_c = rows[ARM_A], rows[ARM_C]
        paired = {"old_final_C_vs_A": gated_science._paired([row for row in final_c if row["suite"] != "identity_controls"], [row for row in final_a if row["suite"] != "identity_controls"], label="old_final_C_vs_A"), "new_final_C_vs_A": gated_science._paired([row for row in final_c if row["suite"] == "identity_controls"], [row for row in final_a if row["suite"] == "identity_controls"], label="new_final_C_vs_A")}
        _validate_accounting(counter)
        report.update({"status": "complete", "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": binding["digest"]}, "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": freeze["digest"]}, "stream_prefix": {"path": _relative(out / "stream_prefix.json", root), "sha256": _sha256(out / "stream_prefix.json"), "digest": prefix_digest}, "endpoint": evidence["endpoint"], "qa_accept": {"path": _relative(QA_ACCEPT, root), "sha256": QA_ACCEPT_SHA256}, "baseline": {"source": "accepted_saved_final_A", "replayed": False, "digest": baseline["digest"]}, "scope": {"old_programs": OLD_PROGRAMS, "new_programs": NEW_PROGRAMS, "new_lengths": {str(length): sum(spec["length"] == length for spec in new_specs) for length in gated.CONTROL_LENGTHS}}, "aggregates": {"final_A": gated_science._aggregate(final_a), "final_C": gated_science._aggregate(final_c)}, "paired": paired, "arms": arm_records, "rows": {"directory": _relative(rows_dir, root), "count": row_index[0]}, "accounting": dict(counter), "limitations": ["250 fixed stream updates per arm only", "screening boundary, not a full 2000-update pilot", "one endpoint and one seed", "identity-path consistency is diagnostic evidence, not proof of a usable latent representation"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        phase = str(counter.get("active_phase", "load"))
        if phase not in counter.get("phase_counts", {}):
            phase = "load"
        _phase_failure(counter, phase, "screening", exc, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_screening(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
