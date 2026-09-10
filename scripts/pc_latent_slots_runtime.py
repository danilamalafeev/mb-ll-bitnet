"""Bounded latent-slot QA serialization gate.

This entry point is intentionally limited to the three-call QA contract.  It
does not prepare a science run or contain a training loop beyond the two
two-example updates required to prove snapshot resume equivalence.
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

from looped_bitnet import longer_native8_e20 as old
from looped_bitnet import register_e15 as dsl
from looped_bitnet import width_e32 as width
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import pc_inference_migration as migration
from scripts import pc_latent_slots as latent
from scripts import pc_learned_scratchpad as accepted
from scripts import pc_learned_scratchpad_runtime as accepted_runtime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_latent_slots_v1" / "qa"
PARENT_LABEL = "float128_seed0"
PARENT_ARM = "float"
PARENT_BRANCH = "B"
PARENT_UPDATE = 40000
PARENT_CHECKPOINT_SHA256 = "59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9"
E36_MANIFEST_SHA256 = followup.E36_MANIFEST_SHA256
E36_PREFLIGHT = followup.E36_PREFLIGHT
E36_RUN = followup.E36_RUN
QA_ACCEPT = ROOT / "runs" / "pc_learned_scratchpad_v1" / "qa_v2_accept.json"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LATENT_SLOTS_PROTOCOL.md"
SCHEMA = "pc_latent_slots_qa_v1"
SNAPSHOT_SCHEMA = "pc_latent_slots_snapshot_v1"
ARCHITECTURE_ID = latent.ARCHITECTURE_ID
QA_ARM = "latent_slots"
QA_BATCHES = 2
QA_CASES_PER_CALL = 2
QA_CALLS = 3
QA_UPDATES = 3
QA_CASES = 6
QA_POSITIONS = 10
QA_NATIVE_STEPS = 80
PURE_SOURCE_SHA256 = "ca1572b54f2ee1d88d310389584fc0af423f04b9a1dcee23abd175bb16af4752"


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


def _move_mapping(value: Mapping[Any, Any], device: torch.device) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, child in value.items():
        if isinstance(child, torch.Tensor):
            result[key] = child.to(device=device)
        elif isinstance(child, Mapping):
            result[key] = _move_mapping(child, device)
        elif isinstance(child, list):
            result[key] = [item.to(device=device) if isinstance(item, torch.Tensor) else item for item in child]
        elif isinstance(child, tuple):
            result[key] = tuple(item.to(device=device) if isinstance(item, torch.Tensor) else item for item in child)
        else:
            result[key] = child
    return result


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move existing moments in place; do not reconstruct the parent optimizer."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
            elif isinstance(value, Mapping):
                state[key] = _move_mapping(value, device)


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
    """Write a complete snapshot beside its destination, then atomically commit it."""

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


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty QA output {path}")
    path.mkdir(parents=True, exist_ok=True)


def _new_counter() -> dict[str, Any]:
    return {
        "attempted_endpoint_loads": 0,
        "completed_endpoint_loads": 0,
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
        "failures": [],
        "timings_seconds": {},
    }


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str | None = None) -> None:
    entry: dict[str, Any] = {"kind": kind, "type": type(exc).__name__, "message": str(exc)}
    if phase is not None:
        entry["phase"] = phase
    counter["failures"].append(entry)


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = counter.setdefault("timings_seconds", {})
        timings[name] = float(timings.get(name, 0.0)) + (time.perf_counter() - started)
        _flush(counter, sink)


@contextmanager
def _count_deserializations(
    counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None = None
) -> Iterator[None]:
    original = torch.load

    def counted(*args: Any, **kwargs: Any) -> Any:
        counter["attempted_underlying_deserializations"] += 1
        _flush(counter, sink)
        try:
            value = original(*args, **kwargs)
        except BaseException as exc:
            _failure(counter, "underlying_deserialize", exc)
            _flush(counter, sink)
            raise
        counter["completed_underlying_deserializations"] += 1
        _flush(counter, sink)
        return value

    torch.load = counted  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _counter_sink(path: Path) -> Callable[[dict[str, Any]], None]:
    def sink(counter: dict[str, Any]) -> None:
        _atomic_json(path, counter)

    return sink


def _source_binding(root: Path = ROOT) -> dict[str, Any]:
    """Bind the actual reviewed loader sources before any parent model load."""

    accepted_binding = accepted_runtime._accepted_qa_binding(root)
    paths = {
        "latent_pure": Path(latent.__file__).resolve(),
        "accepted_loss": Path(accepted.__file__).resolve(),
        "e36_loader": Path(wave.__file__).resolve(),
        "e33_loader": Path(accepted_runtime.e33.__file__).resolve(),
        "windows_compatibility_adapter": Path(migration.__file__).resolve(),
        "optimizer_adapter": Path(old.__file__).resolve(),
        "width_model": Path(width.__file__).resolve(),
        "register_core": Path(dsl.__file__).resolve(),
        "protocol": _resolve(PROTOCOL, root),
        "runtime": Path(__file__).resolve(),
        "accepted_qa": _resolve(QA_ACCEPT, root),
    }
    expected_by_name = {
        "latent_pure": PURE_SOURCE_SHA256,
        "accepted_loss": accepted_runtime._ACCEPTED_SOURCE_HASHES["scratchpad_pc_learned_scratchpad.py"],
        "e36_loader": accepted_runtime._ACCEPTED_SOURCE_HASHES["e36_loader_length_wave_e35_e36.py"],
        "e33_loader": accepted_runtime._ACCEPTED_SOURCE_HASHES["e33_loader_continuation_e33.py"],
        "windows_compatibility_adapter": accepted_runtime._ACCEPTED_SOURCE_HASHES["windows_compatibility_adapter_pc_inference_migration.py"],
        "optimizer_adapter": accepted_runtime._ACCEPTED_SOURCE_HASHES["optimizer_adapter_longer_native8_e20.py"],
        "width_model": accepted_runtime._ACCEPTED_SOURCE_HASHES["width_model_width_e32.py"],
        "register_core": accepted_runtime._ACCEPTED_SOURCE_HASHES["register_core_register_e15.py"],
    }
    files: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"source binding is missing: {path}")
        digest = _sha256(path)
        if name in expected_by_name and digest != expected_by_name[name]:
            raise ValueError(f"reviewed source changed: {name}")
        files[name] = {"path": _relative(path, root), "sha256": digest}
    checkpoint = _resolve(E36_RUN / PARENT_LABEL / PARENT_BRANCH / "u40000.pt", root)
    if not checkpoint.is_file() or _sha256(checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("accepted E36 parent checkpoint hash changed")
    manifest = _resolve(E36_PREFLIGHT / "manifest.json", root)
    if not manifest.is_file() or _sha256(manifest) != E36_MANIFEST_SHA256:
        raise ValueError("accepted E36 manifest hash changed")
    binding = {
        "schema": "pc_latent_slots_qa_source_binding_v1",
        "files": files,
        "accepted_qa_binding": accepted_binding,
        "parent_checkpoint": {"path": _relative(checkpoint, root), "sha256": PARENT_CHECKPOINT_SHA256},
        "e36_manifest": {"path": _relative(manifest, root), "sha256": E36_MANIFEST_SHA256},
    }
    binding["digest"] = old.digest_object(binding)
    return binding


def _configure_runtime() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("latent-slot QA requires CUDA")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    settings = {
        "device": str(device),
        "dtype": "torch.float32",
        "autocast": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "native_steps": latent.NATIVE_STEPS,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    return device, settings


def _example_record(example: dsl.RegisterExample) -> dict[str, Any]:
    return {
        "x": int(example.x),
        "y": int(example.y),
        "program": list(example.program),
        "target_trace": [list(pair) for pair in example.targets],
    }


def _fixture(source: Mapping[str, Any], settings: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    _a_stream, b_stream = wave.build_streams()
    if len(b_stream) < 2 or [len(b_stream[index][0].program) for index in range(2)] != [1, 2]:
        raise ValueError("accepted B QA prefix is not L1 then L2")
    batches = [list(b_stream[index][:QA_CASES_PER_CALL]) for index in range(QA_BATCHES)]
    if any(len(batch) != QA_CASES_PER_CALL or len({len(example.program) for example in batch}) != 1 for batch in batches):
        raise ValueError("accepted B QA prefix is not homogeneous two-example batches")
    raw_batches = [[_example_record(example) for example in batch] for batch in batches]
    config = {
        "schema": "pc_latent_slots_qa_config_v1",
        "architecture_id": ARCHITECTURE_ID,
        "parent_label": PARENT_LABEL,
        "parent_arm": PARENT_ARM,
        "parent_branch": PARENT_BRANCH,
        "parent_update": PARENT_UPDATE,
        "batch_size": QA_CASES_PER_CALL,
        "batch_lengths": [len(batch[0].program) for batch in batches],
        "sequence": ["L1_update_save", "L2_uninterrupted", "L2_reloaded"],
        "calls": QA_CALLS,
        "updates": QA_UPDATES,
        "cases": QA_CASES,
        "readout_positions": QA_POSITIONS,
        "native_steps": QA_NATIVE_STEPS,
        "optimizer": "existing_e36_optimizer_plus_one_latent_group",
        "loss": "accepted_flattened_summed_x_y_cross_entropy_means",
        "adapter_initialization": {
            "seed": latent.LOCAL_INITIALIZATION_SEED,
            "generator_device": "cpu",
            "order": list(latent.INITIALIZER_ORDER),
            "std": latent.LOCAL_INITIALIZATION_STD,
        },
        "runtime_settings": dict(settings),
    }
    fixture: dict[str, Any] = {
        "schema": SCHEMA,
        "parent": {
            "label": PARENT_LABEL,
            "arm": PARENT_ARM,
            "branch": PARENT_BRANCH,
            "update": PARENT_UPDATE,
            "checkpoint": _relative(_resolve(E36_RUN / PARENT_LABEL / PARENT_BRANCH / "u40000.pt", root), root),
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "e36_manifest": _relative(_resolve(E36_PREFLIGHT / "manifest.json", root), root),
            "e36_manifest_sha256": E36_MANIFEST_SHA256,
        },
        "batches": raw_batches,
        "source_digest": source["digest"],
        "config": config,
    }
    fixture["input_digest"] = old.digest_object(raw_batches)
    fixture["stream_digest"] = old.batch_digest(batches)
    fixture["target_digest"] = old.target_digest(batches)
    fixture["fixture_digest"] = old.digest_object({
        key: fixture[key] for key in ("schema", "parent", "batches", "input_digest", "stream_digest", "target_digest", "source_digest", "config")
    })
    return fixture


def _fixture_batches(fixture: Mapping[str, Any]) -> list[list[dsl.RegisterExample]]:
    raw_batches = fixture.get("batches")
    if not isinstance(raw_batches, list) or len(raw_batches) != QA_BATCHES:
        raise ValueError("QA fixture batch count changed")
    result: list[list[dsl.RegisterExample]] = []
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, list) or len(raw_batch) != QA_CASES_PER_CALL:
            raise ValueError("QA fixture case count changed")
        batch: list[dsl.RegisterExample] = []
        for raw in raw_batch:
            if not isinstance(raw, Mapping):
                raise ValueError("QA fixture example malformed")
            example = dsl.RegisterExample(int(raw["x"]), int(raw["y"]), tuple(str(op) for op in raw["program"]))
            if raw.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("QA fixture target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("QA fixture batch is not homogeneous")
        result.append(batch)
    if old.batch_digest(result) != fixture.get("stream_digest") or old.target_digest(result) != fixture.get("target_digest"):
        raise ValueError("QA fixture stream or target digest changed")
    if old.digest_object(raw_batches) != fixture.get("input_digest"):
        raise ValueError("QA fixture input digest changed")
    return result


def _parent_checkpoint(root: Path = ROOT) -> Path:
    return _resolve(E36_RUN / PARENT_LABEL / PARENT_BRANCH / "u40000.pt", root)


def _validate_parent_identity(payload: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {"arm": PARENT_ARM, "branch": PARENT_BRANCH, "width": 128, "seed": 0, "update": PARENT_UPDATE}
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"accepted E36 parent identity changed: {key}")
    if payload.get("model_digest") != width.digest_state_dict(model):
        raise ValueError("accepted E36 parent model digest changed")
    if payload.get("optimizer_digest") != width.digest_object(optimizer.state_dict()):
        raise ValueError("accepted E36 parent optimizer digest changed")


def _load_parent(
    *, manifest: Mapping[str, Any], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None = None,
    root: Path = ROOT,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    path = _parent_checkpoint(root)
    if not path.is_file() or _sha256(path) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("accepted E36 parent checkpoint hash changed")
    counter["attempted_endpoint_loads"] += 1
    _flush(counter, sink)
    try:
        model, optimizer, payload = wave.load_checkpoint(
            path, manifest, arm=PARENT_ARM, branch=PARENT_BRANCH,
            expected_update=PARENT_UPDATE, qa=False, root=root,
        )
        _validate_parent_identity(payload, model, optimizer)
        counter["completed_endpoint_loads"] += 1
        _flush(counter, sink)
        return model, optimizer, dict(payload)
    except BaseException as exc:
        _failure(counter, "endpoint_load", exc)
        _flush(counter, sink)
        raise


def _append_adapter_group(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, device: torch.device,
) -> tuple[latent.LatentSlotAdapter, dict[str, Any], dict[str, Any], list[str]]:
    if len(optimizer.param_groups) != 1:
        raise ValueError("accepted parent optimizer must have exactly one original group")
    original_group = optimizer.param_groups[0]
    original_parameters = list(original_group.get("params", ()))
    parent_parameters = list(model.parameters())
    if len(original_parameters) != len(parent_parameters) or any(left is not right for left, right in zip(original_parameters, parent_parameters)):
        raise ValueError("accepted optimizer parent parameter order changed")
    parent_names = [name for name, _ in model.named_parameters()]
    existing_names = original_group.get("param_names")
    if existing_names is not None and list(existing_names) != parent_names:
        raise ValueError("accepted optimizer parent names/order changed")
    # The inherited E36 optimizer has no names field on older PyTorch builds.
    # Adding the validated names preserves its parameter ordering and lets the
    # appended I/R/W group use the native optimizer association metadata.
    original_group.setdefault("param_names", parent_names)
    adapter, initialization = latent.make_adapter(device=device)
    initial_identity = latent.adapter_identity(adapter)
    copied = {key: deepcopy(value) for key, value in original_group.items() if key not in ("params", "param_names")}
    copied["params"] = list(adapter.parameters())
    copied["param_names"] = list(latent.ADAPTER_PARAMETER_NAMES)
    optimizer.add_param_group(copied)
    latent.validate_optimizer_adapter_association(optimizer, adapter)
    return adapter, initialization, initial_identity, [name for name, _ in model.named_parameters()]


def _batch_tensors(
    batch: Sequence[dsl.RegisterExample], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != QA_CASES_PER_CALL:
        raise ValueError("QA batch must contain exactly two examples")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("QA batch must be homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits = bits[ids_x]
    y_bits = bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _run_update(
    model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer,
    batch: Sequence[dsl.RegisterExample], *, device: torch.device, counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None] | None = None,
    forward: Callable[[torch.nn.Module, latent.LatentSlotAdapter, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> float:
    length = len(batch[0].program)
    cases = len(batch)
    update_number = counter["attempted_updates"] + 1
    phase = f"{QA_ARM}:update{update_number}"
    counter["attempted_updates"] += 1
    counter["attempted_forwards"] += 1
    counter["attempted_cases"] += cases
    counter["attempted_readout_positions"] += cases * length
    counter["attempted_native_steps"] += cases * length * latent.NATIVE_STEPS
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with _timed(counter, "forward", sink):
            if forward is None:
                logits_x, logits_y = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops)
            else:
                logits_x, logits_y = forward(model, adapter, x_bits, y_bits, ops)
        expected_shape = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected_shape or tuple(logits_y.shape) != expected_shape:
            raise ValueError("latent QA logits shape changed")
        if not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
            raise FloatingPointError("latent QA logits are nonfinite")
    except BaseException as exc:
        _failure(counter, "forward", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
    _flush(counter, sink)
    try:
        with _timed(counter, "loss", sink):
            loss = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("latent QA loss is nonfinite")
    except BaseException as exc:
        _failure(counter, "loss", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["attempted_backwards"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, "backward", sink):
            loss.backward()
    except BaseException as exc:
        _failure(counter, "backward", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_backwards"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, "gradient_clip", sink):
            parameters = [*model.parameters(), *adapter.parameters()]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _failure(counter, "gradient_clip", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["attempted_optimizer_steps"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, "optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        _failure(counter, "optimizer_step", exc, phase=phase)
        _flush(counter, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    _flush(counter, sink)
    return float(loss.detach().item())


def _rng_states() -> tuple[torch.Tensor, list[torch.Tensor]]:
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda


def _state_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    cpu_rng, cuda_rng = _rng_states()
    return {
        "model_digest": old.digest_object(model.state_dict()),
        "adapter_digest": old.digest_object(adapter.state_dict()),
        "optimizer_digest": old.digest_object(optimizer.state_dict()),
        "cpu_rng_digest": old.digest_object(cpu_rng),
        "cuda_rng_digest": old.digest_object(cuda_rng),
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }


def _optimizer_metadata(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [{key: _cpu_copy(value) for key, value in group.items() if key != "params"} for group in optimizer.param_groups]


def _snapshot_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in (
        "schema", "committed", "complete", "manifest_digest", "source_binding_digest", "fixture_digest",
        "parent_identity", "architecture_id", "local_update", "absolute_update", "next_batch_index",
        "adapter_identity", "adapter_initialization", "model_state_dict", "adapter_state_dict",
        "optimizer_state_dict", "optimizer_metadata", "cpu_rng_state", "cuda_rng_state",
        "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names",
        "optimizer_group_param_names", "model_digest", "adapter_digest", "optimizer_digest",
        "cpu_rng_digest", "cuda_rng_digest",
    )}


def _snapshot_payload(
    model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, *,
    parent_identity: Mapping[str, Any], source_binding_digest: str, fixture_digest: str, manifest_digest: str,
    adapter_identity: Mapping[str, Any], adapter_initialization: Mapping[str, Any], next_batch_index: int,
    local_update: int,
) -> dict[str, Any]:
    if next_batch_index != 1 or local_update != 1:
        raise ValueError("latent QA snapshot boundary changed")
    model_state = _cpu_copy(model.state_dict())
    adapter_state = _cpu_copy(adapter.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng, cuda_rng = _rng_states()
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "committed": True,
        "complete": True,
        "manifest_digest": manifest_digest,
        "source_binding_digest": source_binding_digest,
        "fixture_digest": fixture_digest,
        "parent_identity": dict(parent_identity),
        "architecture_id": ARCHITECTURE_ID,
        "local_update": local_update,
        "absolute_update": PARENT_UPDATE + local_update,
        "next_batch_index": next_batch_index,
        "adapter_identity": dict(adapter_identity),
        "adapter_initialization": dict(adapter_initialization),
        "model_state_dict": model_state,
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_metadata": _optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }
    payload["model_digest"] = old.digest_object(model_state)
    payload["adapter_digest"] = old.digest_object(adapter_state)
    payload["optimizer_digest"] = old.digest_object(optimizer_state)
    payload["cpu_rng_digest"] = old.digest_object(cpu_rng)
    payload["cuda_rng_digest"] = old.digest_object(cuda_rng)
    payload["snapshot_digest"] = old.digest_object(_snapshot_core(payload))
    return payload


def _load_snapshot(path: Path) -> Mapping[str, Any]:
    if Path(path).name.lower().endswith(".tmp"):
        raise ValueError("temporary snapshot is not a committed checkpoint")
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("latent QA snapshot is malformed")
    if payload.get("committed") is not True or payload.get("complete") is not True:
        raise ValueError("latent QA snapshot is incomplete")
    return payload


def _same_adapter_structure(current: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(current.get(key) == expected.get(key) for key in (
        "schema", "architecture_id", "parameter_names", "parameter_shapes", "parameter_count"
    ))


def _restore_snapshot(
    payload: Mapping[str, Any], model: torch.nn.Module, adapter: latent.LatentSlotAdapter,
    optimizer: torch.optim.Optimizer, *, parent_identity: Mapping[str, Any], source_binding_digest: str,
    fixture_digest: str, manifest_digest: str, expected_adapter_identity: Mapping[str, Any], device: torch.device,
) -> str:
    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("committed") is not True or payload.get("complete") is not True:
        raise ValueError("latent QA snapshot is incomplete")
    for key, expected in {
        "manifest_digest": manifest_digest,
        "source_binding_digest": source_binding_digest,
        "fixture_digest": fixture_digest,
        "parent_identity": dict(parent_identity),
        "architecture_id": ARCHITECTURE_ID,
        "local_update": 1,
        "absolute_update": PARENT_UPDATE + 1,
        "next_batch_index": 1,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"latent QA snapshot binding changed: {key}")
    if not _same_adapter_structure(payload.get("adapter_identity", {}), expected_adapter_identity):
        raise ValueError("latent QA snapshot adapter identity changed")
    if payload.get("model_parameter_names") != list(model.state_dict()) or payload.get("adapter_parameter_names") != [name for name, _ in adapter.named_parameters()]:
        raise ValueError("latent QA snapshot parameter inventory changed")
    if payload.get("optimizer_metadata") != _optimizer_metadata(optimizer):
        raise ValueError("latent QA snapshot optimizer configuration changed")
    if payload.get("optimizer_group_param_names") != [list(group.get("param_names", [])) for group in optimizer.param_groups]:
        raise ValueError("latent QA snapshot optimizer name association changed")
    required = ("model_state_dict", "adapter_state_dict", "optimizer_state_dict", "cpu_rng_state", "cuda_rng_state", "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest", "snapshot_digest")
    if any(key not in payload for key in required):
        raise ValueError("latent QA snapshot fields are incomplete")
    if old.digest_object(payload["model_state_dict"]) != payload["model_digest"]:
        raise ValueError("latent QA snapshot model digest changed")
    if old.digest_object(payload["adapter_state_dict"]) != payload["adapter_digest"]:
        raise ValueError("latent QA snapshot adapter digest changed")
    if old.digest_object(payload["optimizer_state_dict"]) != payload["optimizer_digest"]:
        raise ValueError("latent QA snapshot optimizer digest changed")
    if old.digest_object(payload["cpu_rng_state"]) != payload["cpu_rng_digest"] or old.digest_object(payload["cuda_rng_state"]) != payload["cuda_rng_digest"]:
        raise ValueError("latent QA snapshot RNG digest changed")
    if old.digest_object(_snapshot_core(payload)) != payload["snapshot_digest"]:
        raise ValueError("latent QA snapshot digest changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    torch.set_rng_state(payload["cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    model.train(bool(payload["training_mode"]))
    adapter.train(bool(payload["adapter_training_mode"]))
    actual = _state_identity(model, adapter, optimizer)
    for key in ("model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest"):
        if actual[key] != payload[key]:
            raise ValueError(f"latent QA snapshot restore mismatch: {key}")
    return str(payload["snapshot_digest"])


def _parent_identity(payload: Mapping[str, Any], checkpoint: Path, manifest_digest: str, cuda_rng: Sequence[torch.Tensor]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "model_digest": str(payload["model_digest"]),
        "optimizer_digest": str(payload["optimizer_digest"]),
        "rng_digest": str(payload["rng_digest"]),
        "manifest_digest": manifest_digest,
        "cuda_rng_digest": old.digest_object(list(cuda_rng)),
        "label": PARENT_LABEL,
        "arm": PARENT_ARM,
        "branch": PARENT_BRANCH,
        "update": PARENT_UPDATE,
    }


def _run_arm(
    *, batches: Sequence[Sequence[dsl.RegisterExample]], manifest: Mapping[str, Any], fixture: Mapping[str, Any],
    source_binding: Mapping[str, Any], device: torch.device, common_cuda_rng: Sequence[torch.Tensor],
    counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], out: Path, root: Path = ROOT,
) -> dict[str, Any]:
    arm_out = out / QA_ARM
    arm_out.mkdir(parents=True, exist_ok=False)
    checkpoint = _parent_checkpoint(root)
    with _timed(counter, "endpoint_load", sink):
        model, optimizer, parent = _load_parent(manifest=manifest, counter=counter, sink=sink, root=root)
    model.to(device)
    _optimizer_to_device(optimizer, device)
    torch.set_rng_state(parent["rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
    model.train(True)
    manifest_digest = old.canonical_hash(manifest)
    parent_id = _parent_identity(parent, checkpoint, manifest_digest, common_cuda_rng)
    adapter, initialization, adapter_initial_identity, parent_parameter_names = _append_adapter_group(model, optimizer, device=device)
    if parent_parameter_names != [name for name, _ in model.named_parameters()]:
        raise ValueError("parent parameter name inventory changed")
    adapter.train(True)
    latent.validate_optimizer_adapter_association(optimizer, adapter)
    identity = {
        "parent": parent_id,
        "source_binding_digest": source_binding["digest"],
        "manifest_digest": manifest_digest,
        "fixture_digest": fixture["fixture_digest"],
        "architecture_id": ARCHITECTURE_ID,
        "adapter_identity_digest": old.digest_object(adapter_initial_identity),
        "local_update": 0,
    }
    loss_l1 = _run_update(model, adapter, optimizer, batches[0], device=device, counter=counter, sink=sink)
    snapshot = _snapshot_payload(
        model, adapter, optimizer, parent_identity=parent_id, source_binding_digest=str(source_binding["digest"]),
        fixture_digest=str(fixture["fixture_digest"]), manifest_digest=manifest_digest,
        adapter_identity=adapter_initial_identity, adapter_initialization=initialization,
        next_batch_index=1, local_update=1,
    )
    snapshot_path = arm_out / "snapshot_l1.pt"
    with _timed(counter, "snapshot_save", sink):
        _atomic_torch(snapshot_path, snapshot, refuse=True)
    loss_uninterrupted = _run_update(model, adapter, optimizer, batches[1], device=device, counter=counter, sink=sink)
    uninterrupted = _state_identity(model, adapter, optimizer)
    counter["attempted_snapshot_loads"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, "snapshot_load", sink):
            loaded = _load_snapshot(snapshot_path)
        counter["completed_snapshot_loads"] += 1
        _flush(counter, sink)
    except BaseException as exc:
        _failure(counter, "snapshot_load", exc, phase=QA_ARM)
        _flush(counter, sink)
        raise
    with _timed(counter, "snapshot_restore", sink):
        try:
            snapshot_digest = _restore_snapshot(
                loaded, model, adapter, optimizer, parent_identity=parent_id,
                source_binding_digest=str(source_binding["digest"]), fixture_digest=str(fixture["fixture_digest"]),
                manifest_digest=manifest_digest, expected_adapter_identity=adapter_initial_identity, device=device,
            )
        except BaseException as exc:
            _failure(counter, "snapshot_restore", exc, phase=QA_ARM)
            _flush(counter, sink)
            raise
    loss_reloaded = _run_update(model, adapter, optimizer, batches[1], device=device, counter=counter, sink=sink)
    reloaded = _state_identity(model, adapter, optimizer)
    equality = {key: uninterrupted[key] == reloaded[key] for key in (
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names",
        "optimizer_group_count", "optimizer_group_param_names",
    )}
    equality["all_exact"] = all(equality.values())
    if not equality["all_exact"]:
        raise ValueError("latent QA reload next-update mismatch")
    record = {
        "schema": "pc_latent_slots_qa_arm_v1",
        "arm": QA_ARM,
        "status": "complete",
        "parent_identity": parent_id,
        "adapter_initialization": initialization,
        "adapter_identity": adapter_initial_identity,
        "parameter_names": {"parent": parent_parameter_names, "adapter": list(latent.ADAPTER_PARAMETER_NAMES)},
        "snapshot": {
            "path": _relative(snapshot_path, root),
            "sha256": _sha256(snapshot_path),
            "snapshot_digest": snapshot_digest,
            "next_batch_index": 1,
            "committed": True,
            "complete": True,
        },
        "losses": [loss_l1, loss_uninterrupted, loss_reloaded],
        "equality": equality,
        "calls": QA_CALLS,
        "updates": QA_UPDATES,
        "cases": QA_CASES,
        "readout_positions": QA_POSITIONS,
        "native_steps": QA_NATIVE_STEPS,
    }
    _atomic_json(arm_out / "report.json", record, refuse=True)
    return record


def run_qa(*, out: Path = OUTPUT, root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    _refuse_nonempty(out)
    source = _source_binding(root)
    device, settings = _configure_runtime()
    fixture = _fixture(source, settings, root=root)
    batches = _fixture_batches(fixture)
    _atomic_json(out / "source_binding.json", source, refuse=True)
    _atomic_json(out / "fixture.json", fixture, refuse=True)
    counter = _new_counter()
    sink = _counter_sink(out / "accounting.json")
    sink(counter)
    manifest_path = _resolve(E36_PREFLIGHT / "manifest.json", root)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source": source,
        "fixture": {"path": _relative(out / "fixture.json", root), "sha256": _sha256(out / "fixture.json"), "fixture_digest": fixture["fixture_digest"]},
        "runtime": settings,
        "models": {},
    }
    try:
        with migration.windows_compatibility_adapter():
            manifest = followup._load_inherited(root)
            common_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
            with _count_deserializations(counter, sink):
                report["models"][QA_ARM] = _run_arm(
                    batches=batches, manifest=manifest, fixture=fixture, source_binding=source,
                    device=device, common_cuda_rng=common_cuda_rng, counter=counter, sink=sink, out=out, root=root,
                )
        expected = {
            "attempted_endpoint_loads": 1, "completed_endpoint_loads": 1,
            "attempted_snapshot_loads": 1, "completed_snapshot_loads": 1,
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
                raise ValueError(f"latent QA accounting mismatch: {key}={counter.get(key)} expected {value}")
        report.update({
            "status": "complete",
            "accounting": dict(counter),
            "budget": {"calls": QA_CALLS, "updates": QA_UPDATES, "cases": QA_CASES, "readout_positions": QA_POSITIONS, "native_steps": QA_NATIVE_STEPS},
            "manifest_digest": old.canonical_hash(manifest),
        })
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        _failure(counter, "qa", exc)
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="run the bounded latent-slot serialization QA")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if not args.qa:
        parser.error("only --qa is supported by this bounded entry point")
    run_qa(out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
