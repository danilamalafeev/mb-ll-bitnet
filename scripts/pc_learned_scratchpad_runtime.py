"""Bounded learned-scratchpad QA and registered science runtime.

``--qa`` runs the accepted six-update serialization gate. ``--prepare-science``
freezes the zero-load science manifest, and ``--science`` runs only the
registered two-arm continuation and its fixed evaluations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from looped_bitnet import width_e32 as width
from scripts import continuation_e33 as e33
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import pc_inference_migration as migration
from looped_bitnet import longer_native8_e20 as old
from scripts import pc_learned_scratchpad as pure


ROOT = Path(__file__).resolve().parents[1]
QA_OUTPUT = ROOT / "runs" / "pc_learned_scratchpad_v1" / "qa"
SCIENCE_OUTPUT = ROOT / "runs" / "pc_learned_scratchpad_v1" / "science"
SCIENCE_MANIFEST = ROOT / "runs" / "pc_learned_scratchpad_v1" / "science_manifest.json"
QA_ACCEPT = ROOT / "runs" / "pc_learned_scratchpad_v1" / "qa_v2_accept.json"
QA_ACCEPTED_SOURCE = ROOT / "runs" / "pc_learned_scratchpad_v1" / "qa_v2_accepted_source"
QA_ACCEPTED_OUTPUT = ROOT / "runs" / "pc_learned_scratchpad_v1" / "qa_v2"
BASELINE_MANIFEST = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "manifest.json"
BASELINE_PROOF = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "exact_map_proof.json"
BASELINE_ENDPOINT = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "endpoints" / "float128_seed0_B.json"
SCIENCE_BATCHES = 2000
SCIENCE_BATCH_SIZE = pure.BATCH_SIZE
SCIENCE_UPDATES = pure.CHILD_UPDATES
SCIENCE_EVAL_STATES = tuple(dsl.STATE_ORDER)
SCIENCE_STATE_STRATUM = {tuple(state): name for name, values in dsl.state_split().items() for state in values}
SCIENCE_ENDPOINT_LABEL = "float128_seed0/B"
SCIENCE_SCHEMA = "pc_learned_scratchpad_science_manifest_v1"
SCIENCE_RUN_SCHEMA = "pc_learned_scratchpad_science_run_v1"
SCIENCE_ACCEPT_HASH = "3fe54ede6a41bea52dbcf8b3f9e201b68573de94d9ee17cc9522c788698f8484"
BASELINE_MANIFEST_SHA256 = "767c2263f25916bbb636131328198e73d524896e8da74577ce751eab6f1c4722"
BASELINE_ENDPOINT_SHA256 = "b1d23d042f06e8b4b3ef9f8b3edcf436a618901364a28e1eba12ea54f20f5297"
SCHEMA = "pc_learned_scratchpad_qa_v1"
SNAPSHOT_SCHEMA = "pc_learned_scratchpad_snapshot_v1"
PARENT_LABEL = "float128_seed0"
PARENT_ARM = "float"
PARENT_BRANCH = "B"
PARENT_UPDATE = 40000
CHILD_ARMS = pure.ARMS
QA_BATCHES = 2
QA_CASES_PER_CALL = 2
QA_CALLS_PER_ARM = 3
QA_TOTAL_CALLS = 6
QA_TOTAL_UPDATES = 6
PARENT_CHECKPOINT_SHA256 = "59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9"
SCIENCE_PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
BASELINE_PROOF_SHA256 = "bbafd6f3baf122f7afab2edd115a911433fbdc8aa242c4266f260351e1c1cd94"
E36_MANIFEST_SHA256 = followup.E36_MANIFEST_SHA256
E36_PREFLIGHT = followup.E36_PREFLIGHT
E36_RUN = followup.E36_RUN
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LEARNED_SCRATCHPAD_PROTOCOL.md"


def _resolve(path: Path, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_torch(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision {temporary}")
    torch.save(value, temporary)
    temporary.replace(path)


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty QA output {path}")
    path.mkdir(parents=True, exist_ok=True)


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
    moved: dict[Any, Any] = {}
    for key, child in value.items():
        if isinstance(child, torch.Tensor):
            moved[key] = child.to(device=device)
        elif isinstance(child, Mapping):
            moved[key] = _move_mapping(child, device)
        elif isinstance(child, (list, tuple)):
            moved[key] = type(child)(
                item.to(device=device) if isinstance(item, torch.Tensor)
                else _move_mapping(item, device) if isinstance(item, Mapping)
                else item for item in child
            )
        else:
            moved[key] = child
    return moved


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move existing optimizer states without reconstruction or reinitialization."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
            elif isinstance(value, Mapping):
                state[key] = _move_mapping(value, device)


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
    }


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str | None = None) -> None:
    item: dict[str, Any] = {"kind": kind, "type": type(exc).__name__, "message": str(exc)}
    if phase is not None:
        item["phase"] = phase
    counter["failures"].append(item)


@contextmanager
def _count_deserializations(
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None] | None = None,
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


def _source_inventory(root: Path = ROOT) -> dict[str, dict[str, str]]:
    paths = {
        "scratchpad": Path(pure.__file__).resolve(),
        "runtime": Path(__file__).resolve(),
        "e36_loader": Path(wave.__file__).resolve(),
        "e33_loader": Path(e33.__file__).resolve(),
        "windows_compatibility_adapter": Path(migration.__file__).resolve(),
        "optimizer_adapter": Path(old.__file__).resolve(),
        "width_model": Path(width.__file__).resolve(),
        "register_core": Path(dsl.__file__).resolve(),
        "protocol": _resolve(PROTOCOL, root),
    }
    return {name: {"path": _relative(path, root), "sha256": _sha256(path)} for name, path in paths.items()}


def _configure_runtime() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("learned scratchpad QA requires CUDA")
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
        "native_steps": pure.NATIVE_STEPS,
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
    b_stream = wave.build_streams()[1]
    if len(b_stream) < QA_BATCHES or [len(b_stream[index][0].program) for index in range(QA_BATCHES)] != [1, 2]:
        raise ValueError("accepted B QA prefix is not L1 then L2")
    batches = [list(b_stream[index][:QA_CASES_PER_CALL]) for index in range(QA_BATCHES)]
    if any(
        len(batch) != QA_CASES_PER_CALL or len({len(example.program) for example in batch}) != 1
        for batch in batches
    ):
        raise ValueError("QA prefix batches are not homogeneous two-example batches")
    stream_digest = old.batch_digest(batches)
    target_digest = old.target_digest(batches)
    config = {
        "schema": "pc_learned_scratchpad_qa_config_v1",
        "parent_label": PARENT_LABEL,
        "parent_branch": PARENT_BRANCH,
        "parent_update": PARENT_UPDATE,
        "child_arms": list(CHILD_ARMS),
        "batch_size": QA_CASES_PER_CALL,
        "batch_lengths": [len(batch[0].program) for batch in batches],
        "calls_per_arm": QA_CALLS_PER_ARM,
        "total_calls": QA_TOTAL_CALLS,
        "total_updates": QA_TOTAL_UPDATES,
        "native_steps": pure.NATIVE_STEPS,
        "sequence": ["L1_update_save", "L2_uninterrupted", "L2_reloaded"],
        "optimizer": "existing_e36_optimizer",
        "loss": "summed_x_y_cross_entropy_means",
        "model_branches": {
            "continuous_control": "original_forward",
            "soft_register_reset": "soft_register_forward",
        },
        "runtime_settings": dict(settings),
    }
    fixture = {
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
        "batches": [[_example_record(example) for example in batch] for batch in batches],
        "stream_digest": stream_digest,
        "target_digest": target_digest,
        "source": dict(source),
        "config": config,
    }
    fixture["input_digest"] = _canonical_digest(fixture["batches"])
    fixture["source_digest"] = _canonical_digest(source)
    fixture["config_digest"] = _canonical_digest(config)
    fixture["fixture_digest"] = _canonical_digest({
        key: fixture[key] for key in (
            "schema", "parent", "batches", "input_digest", "stream_digest", "target_digest", "source", "config"
        )
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
                raise ValueError("QA fixture example is malformed")
            example = dsl.RegisterExample(int(raw["x"]), int(raw["y"]), tuple(str(op) for op in raw["program"]))
            if raw.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("QA fixture target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("QA fixture batch is not homogeneous")
        result.append(batch)
    if old.batch_digest(result) != fixture.get("stream_digest") or old.target_digest(result) != fixture.get("target_digest"):
        raise ValueError("QA fixture stream or target digest changed")
    if _canonical_digest(raw_batches) != fixture.get("input_digest"):
        raise ValueError("QA fixture input digest changed")
    return result


def _parent_checkpoint(root: Path = ROOT) -> Path:
    return _resolve(E36_RUN / PARENT_LABEL / PARENT_BRANCH / "u40000.pt", root)


def _validate_parent_identity(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Validate the accepted E36 parent fields emitted by its strict loader."""

    expected = {
        "arm": PARENT_ARM,
        "branch": PARENT_BRANCH,
        "width": 128,
        "seed": 0,
        "update": PARENT_UPDATE,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"accepted E36 parent identity changed: {key}")
    if payload.get("model_digest") != width.digest_state_dict(model):
        raise ValueError("accepted E36 parent model digest changed")
    if payload.get("optimizer_digest") != width.digest_object(optimizer.state_dict()):
        raise ValueError("accepted E36 parent optimizer digest changed")


def _load_parent(
    *,
    arm: str,
    manifest: Mapping[str, Any],
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None] | None = None,
    root: Path = ROOT,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    if arm not in CHILD_ARMS:
        raise ValueError("unknown child arm")
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
        return model, optimizer, payload
    except BaseException as exc:
        _failure(counter, "endpoint_load", exc, phase=arm)
        _flush(counter, sink)
        raise


def _batch_tensors(
    batch: Sequence[dsl.RegisterExample],
    device: torch.device,
    *,
    expected_cases: int = QA_CASES_PER_CALL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch or len(batch) != expected_cases:
        raise ValueError(f"batch must contain {expected_cases} examples")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("QA batch must be homogeneous")
    x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    ops = torch.tensor(
        [[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch],
        dtype=torch.long,
        device=device,
    )
    tx = torch.tensor(
        [[target[0] for target in example.targets] for example in batch],
        dtype=torch.long,
        device=device,
    )
    ty = torch.tensor(
        [[target[1] for target in example.targets] for example in batch],
        dtype=torch.long,
        device=device,
    )
    return x, y, ops, tx, ty


def _forward_logits(
    model: torch.nn.Module,
    batch: Sequence[dsl.RegisterExample],
    *,
    arm: str,
    device: torch.device,
    expected_cases: int = QA_CASES_PER_CALL,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, ops, _tx, _ty = _batch_tensors(batch, device, expected_cases=expected_cases)
    with torch.autocast(device_type=device.type, enabled=False):
        if arm == "continuous_control":
            logits_x, logits_y = model(x, y, ops)
        elif arm == "soft_register_reset":
            logits_x, logits_y = pure.soft_register_forward(model, x, y, ops)
        else:
            raise ValueError("unknown child arm")
    expected = (len(batch), len(batch[0].program), pure.REGISTER_WIDTH)
    if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected:
        raise ValueError("QA model logits shape changed")
    if logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32:
        raise ValueError("QA logits are not float32")
    if not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
        raise FloatingPointError("QA model logits are nonfinite")
    return logits_x, logits_y


def _run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[dsl.RegisterExample],
    *,
    arm: str,
    device: torch.device,
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None] | None = None,
    forward: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    expected_cases: int = QA_CASES_PER_CALL,
) -> float:
    length = len(batch[0].program)
    cases = len(batch)
    update_index = counter["attempted_updates"] + 1
    counter["attempted_updates"] += 1
    counter["attempted_forwards"] += 1
    counter["attempted_cases"] += cases
    counter["attempted_readout_positions"] += cases * length
    counter["attempted_native_steps"] += cases * length * pure.NATIVE_STEPS
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    try:
        logits_x, logits_y = (
            forward(model, batch, arm=arm, device=device)
            if forward is not None
            else _forward_logits(model, batch, arm=arm, device=device, expected_cases=expected_cases)
        )
    except BaseException as exc:
        _failure(counter, "forward", exc, phase=f"{arm}:update{update_index}")
        _flush(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * pure.NATIVE_STEPS
    _flush(counter, sink)
    try:
        _x, _y, _ops, targets_x, targets_y = _batch_tensors(batch, device, expected_cases=expected_cases)
        with torch.autocast(device_type=device.type, enabled=False):
            loss = pure.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("QA loss is nonfinite")
    except BaseException as exc:
        _failure(counter, "loss", exc, phase=f"{arm}:update{update_index}")
        _flush(counter, sink)
        raise
    counter["attempted_backwards"] += 1
    _flush(counter, sink)
    try:
        loss.backward()
    except BaseException as exc:
        _failure(counter, "backward", exc, phase=f"{arm}:update{update_index}")
        _flush(counter, sink)
        raise
    counter["completed_backwards"] += 1
    _flush(counter, sink)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _failure(counter, "gradient_clip", exc, phase=f"{arm}:update{update_index}")
        _flush(counter, sink)
        raise
    counter["attempted_optimizer_steps"] += 1
    _flush(counter, sink)
    try:
        optimizer.step()
    except BaseException as exc:
        _failure(counter, "optimizer_step", exc, phase=f"{arm}:update{update_index}")
        _flush(counter, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    _flush(counter, sink)
    return float(loss.detach().item())


def _state_identity(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "model_digest": old.digest_object(model.state_dict()),
        "optimizer_digest": old.digest_object(optimizer.state_dict()),
        "cpu_rng_digest": old.digest_object(torch.get_rng_state()),
        "cuda_rng_digest": old.digest_object(torch.cuda.get_rng_state_all()),
        "training_mode": bool(model.training),
        "parameter_names": list(model.state_dict()),
    }


def _optimizer_metadata(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {key: _cpu_copy(value) for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def _snapshot_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    identity: Mapping[str, Any],
    fixture_digest: str,
) -> dict[str, Any]:
    model_state = _cpu_copy(model.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "identity": dict(identity),
        "fixture_digest": fixture_digest,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_metadata": _optimizer_metadata(optimizer),
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "training_mode": bool(model.training),
        "parameter_names": list(model.state_dict()),
    }
    payload["model_digest"] = old.digest_object(model_state)
    payload["optimizer_digest"] = old.digest_object(optimizer_state)
    payload["cpu_rng_digest"] = old.digest_object(payload["cpu_rng_state"])
    payload["cuda_rng_digest"] = old.digest_object(payload["cuda_rng_state"])
    payload["snapshot_digest"] = old.digest_object({
        key: payload[key] for key in (
            "schema", "identity", "fixture_digest", "model_state_dict", "optimizer_state_dict",
            "optimizer_metadata", "cpu_rng_state", "cuda_rng_state", "training_mode",
            "parameter_names", "model_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        )
    })
    return payload


def _restore_snapshot(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_identity: Mapping[str, Any],
    expected_fixture_digest: str,
    device: torch.device,
) -> str:
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("QA snapshot schema changed")
    pure.validate_child_identity(payload.get("identity", {}), expected_identity)
    if payload.get("fixture_digest") != expected_fixture_digest:
        raise ValueError("QA snapshot fixture binding changed")
    if payload.get("parameter_names") != list(model.state_dict()):
        raise ValueError("QA snapshot parameter inventory changed")
    if payload.get("optimizer_metadata") != _optimizer_metadata(optimizer):
        raise ValueError("QA snapshot optimizer configuration changed")
    required = (
        "model_state_dict", "optimizer_state_dict", "cpu_rng_state", "cuda_rng_state",
        "model_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest", "snapshot_digest",
    )
    if any(key not in payload for key in required):
        raise ValueError("QA snapshot fields are incomplete")
    if old.digest_object(payload["model_state_dict"]) != payload["model_digest"]:
        raise ValueError("QA snapshot model digest changed")
    if old.digest_object(payload["optimizer_state_dict"]) != payload["optimizer_digest"]:
        raise ValueError("QA snapshot optimizer digest changed")
    if old.digest_object(payload["cpu_rng_state"]) != payload["cpu_rng_digest"]:
        raise ValueError("QA snapshot CPU RNG digest changed")
    if old.digest_object(payload["cuda_rng_state"]) != payload["cuda_rng_digest"]:
        raise ValueError("QA snapshot CUDA RNG digest changed")
    expected_snapshot_digest = old.digest_object({
        key: payload[key] for key in (
            "schema", "identity", "fixture_digest", "model_state_dict", "optimizer_state_dict",
            "optimizer_metadata", "cpu_rng_state", "cuda_rng_state", "training_mode",
            "parameter_names", "model_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        )
    })
    if expected_snapshot_digest != payload["snapshot_digest"]:
        raise ValueError("QA snapshot digest changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    torch.set_rng_state(payload["cpu_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    model.train(bool(payload["training_mode"]))
    actual = _state_identity(model, optimizer)
    if actual["model_digest"] != payload["model_digest"] or actual["optimizer_digest"] != payload["optimizer_digest"]:
        raise ValueError("QA snapshot restore state mismatch")
    if actual["cpu_rng_digest"] != payload["cpu_rng_digest"] or actual["cuda_rng_digest"] != payload["cuda_rng_digest"]:
        raise ValueError("QA snapshot restore RNG mismatch")
    if actual["training_mode"] is not bool(payload["training_mode"]):
        raise ValueError("QA snapshot restore mode mismatch")
    return str(payload["snapshot_digest"])


def _child_parent_identity(
    payload: Mapping[str, Any],
    checkpoint: Path,
    cuda_rng: Sequence[torch.Tensor],
) -> dict[str, str]:
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "model_digest": str(payload["model_digest"]),
        "optimizer_digest": str(payload["optimizer_digest"]),
        "cpu_rng_digest": str(payload["rng_digest"]),
        "cuda_rng_digest": old.digest_object(list(cuda_rng)),
    }


def _arm_identity(
    parent_identity: Mapping[str, Any],
    *,
    arm: str,
    local_update: int,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    return pure.child_identity(
        parent_identity,
        arm=arm,
        local_update=local_update,
        stream_digest=str(fixture["stream_digest"]),
        target_digest=str(fixture["target_digest"]),
        source_digest=str(fixture["source_digest"]),
        config_digest=str(fixture["config_digest"]),
    )


def _counter_sink(path: Path) -> Callable[[dict[str, Any]], None]:
    def sink(counter: dict[str, Any]) -> None:
        _write_json(path, counter)
    return sink


def _run_arm(
    *,
    arm: str,
    batches: Sequence[Sequence[dsl.RegisterExample]],
    manifest: Mapping[str, Any],
    fixture: Mapping[str, Any],
    device: torch.device,
    common_cuda_rng: Sequence[torch.Tensor],
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
    out: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    checkpoint = _parent_checkpoint(root)
    model, optimizer, parent = _load_parent(
        arm=arm, manifest=manifest, counter=counter, sink=sink, root=root
    )
    model.to(device)
    _optimizer_to_device(optimizer, device)
    torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
    torch.set_rng_state(parent["rng_state"])
    model.train(True)
    parent_identity = _child_parent_identity(parent, checkpoint, common_cuda_rng)
    identity_u0 = _arm_identity(parent_identity, arm=arm, local_update=0, fixture=fixture)
    parameter_names = list(model.state_dict())
    record: dict[str, Any] = {
        "arm": arm,
        "status": "running",
        "parent_identity": parent_identity,
        "identity_u0": identity_u0,
        "parameter_names": parameter_names,
    }
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=False)
    loss_u1 = _run_update(
        model, optimizer, batches[0], arm=arm, device=device, counter=counter, sink=sink
    )
    identity_u1 = _arm_identity(parent_identity, arm=arm, local_update=1, fixture=fixture)
    snapshot = _snapshot_payload(
        model, optimizer, identity=identity_u1, fixture_digest=str(fixture["fixture_digest"])
    )
    snapshot_path = arm_out / "snapshot_u1.pt"
    try:
        _write_torch(snapshot_path, snapshot, refuse=True)
    except BaseException as exc:
        _failure(counter, "snapshot_save", exc, phase=arm)
        _flush(counter, sink)
        raise
    record["snapshot"] = {
        "path": _relative(snapshot_path, root),
        "sha256": _sha256(snapshot_path),
        "snapshot_digest": snapshot["snapshot_digest"],
        "identity": identity_u1,
    }
    loss_uninterrupted = _run_update(
        model, optimizer, batches[1], arm=arm, device=device, counter=counter, sink=sink
    )
    uninterrupted = _state_identity(model, optimizer)
    counter["attempted_snapshot_loads"] += 1
    _flush(counter, sink)
    try:
        loaded_snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_snapshot, Mapping):
            raise ValueError("QA snapshot payload is not a mapping")
        counter["completed_snapshot_loads"] += 1
        _flush(counter, sink)
    except BaseException as exc:
        _failure(counter, "snapshot_load", exc, phase=arm)
        _flush(counter, sink)
        raise
    try:
        _restore_snapshot(
            loaded_snapshot, model, optimizer, expected_identity=identity_u1,
            expected_fixture_digest=str(fixture["fixture_digest"]), device=device,
        )
    except BaseException as exc:
        _failure(counter, "snapshot_restore", exc, phase=arm)
        _flush(counter, sink)
        raise
    loss_reloaded = _run_update(
        model, optimizer, batches[1], arm=arm, device=device, counter=counter, sink=sink
    )
    reloaded = _state_identity(model, optimizer)
    equality = {
        "model_exact": uninterrupted["model_digest"] == reloaded["model_digest"],
        "optimizer_exact": uninterrupted["optimizer_digest"] == reloaded["optimizer_digest"],
        "cpu_rng_exact": uninterrupted["cpu_rng_digest"] == reloaded["cpu_rng_digest"],
        "cuda_rng_exact": uninterrupted["cuda_rng_digest"] == reloaded["cuda_rng_digest"],
        "mode_exact": uninterrupted["training_mode"] == reloaded["training_mode"],
        "parameter_names_exact": uninterrupted["parameter_names"] == reloaded["parameter_names"],
        "uninterrupted": uninterrupted,
        "reloaded": reloaded,
    }
    if not all(equality[key] for key in (
        "model_exact", "optimizer_exact", "cpu_rng_exact", "cuda_rng_exact",
        "mode_exact", "parameter_names_exact"
    )):
        raise ValueError(f"QA reload next-update mismatch: {arm}")
    record.update({
        "status": "complete",
        "losses": [loss_u1, loss_uninterrupted, loss_reloaded],
        "equality": equality,
        "calls": QA_CALLS_PER_ARM,
        "updates_completed": QA_CALLS_PER_ARM,
        "positions": 2 + 4 + 4,
        "native_steps": (2 + 4 + 4) * pure.NATIVE_STEPS,
    })
    _write_json(arm_out / "report.json", record, refuse=True)
    return record


def run_qa(*, out: Path = QA_OUTPUT, root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    _refuse_nonempty(out)
    device, settings = _configure_runtime()
    source = _source_inventory(root)
    fixture = _fixture(source, settings, root=root)
    batches = _fixture_batches(fixture)
    _write_json(out / "fixture.json", fixture, refuse=True)
    counter = _new_counter()
    sink = _counter_sink(out / "accounting.json")
    sink(counter)
    manifest_path = _resolve(E36_PREFLIGHT / "manifest.json", root)
    if not manifest_path.is_file() or _sha256(manifest_path) != E36_MANIFEST_SHA256:
        raise ValueError("accepted E36 manifest hash changed")
    with migration.windows_compatibility_adapter():
        manifest = followup._load_inherited(root)
        common_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "running",
            "fixture": {
                "path": _relative(out / "fixture.json", root),
                "sha256": _sha256(out / "fixture.json"),
                "fixture_digest": fixture["fixture_digest"],
            },
            "source": source,
            "runtime": settings,
            "common_cuda_rng_digest": old.digest_object(common_cuda_rng),
            "models": {},
        }
        try:
            with _count_deserializations(counter, sink):
                for arm in CHILD_ARMS:
                    report["models"][arm] = _run_arm(
                        arm=arm, batches=batches, manifest=manifest, fixture=fixture,
                        device=device, common_cuda_rng=common_cuda_rng, counter=counter,
                        sink=sink, out=out, root=root,
                    )
            report["status"] = "complete"
            report["accounting"] = dict(counter)
            report["budget"] = {
                "calls": QA_TOTAL_CALLS,
                "updates": QA_TOTAL_UPDATES,
                "cases": pure.TINY_QA_CASES,
                "readout_positions": pure.TINY_QA_READOUT_POSITIONS,
                "native_steps": pure.TINY_QA_NATIVE_STEPS,
            }
            _write_json(out / "report.json", report, refuse=True)
            return report
        except BaseException as exc:
            _failure(counter, "qa", exc)
            sink(counter)
            report["status"] = "failed"
            report["accounting"] = dict(counter)
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(out / "report.json", report, refuse=True)
            raise


_ACCEPTED_SOURCE_HASHES = {
    "e33_loader_continuation_e33.py": "677f25bbd51165f720afc008dae657cadd9cbc8e16fee6746c23e68286b60e73",
    "e36_loader_length_wave_e35_e36.py": "383658faa81b82a068f17a223450453eb8c1d2e591a3add2c3d7139d9af35c0f",
    "optimizer_adapter_longer_native8_e20.py": "8f738b0c881eaa1f0488f44b46d63591ac77bc276d811b016f44fab8b06e8c12",
    "protocol_PC_LEARNED_SCRATCHPAD_PROTOCOL.md": "9d74deef6ae11555f4650a5074bbd8fb3637097f8618cb159b10b8f85c90fc31",
    "register_core_register_e15.py": "bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac",
    "runtime_pc_learned_scratchpad_runtime.py": "579609560f3f59e94a25f85bbb8338d67743b7f4c6f0364cae5fa4a8e22cacc2",
    "scratchpad_pc_learned_scratchpad.py": "9cab245d4145456a319aea55c93ccf9c7e02190b25003ac5e307de93b6fdc88b",
    "width_model_width_e32.py": "7f754f63613cdf9c90b54f8eb934e114f9f0a72aab0f61639fc02321fc95f5c2",
    "windows_compatibility_adapter_pc_inference_migration.py": "569570fcfe45eb4c706bc2410f0221e78240c1958fb667d6c99bd69b2f1a0f8d",
}


_SCIENCE_TRANSITIVE_PATHS = {
    "model_core": ROOT / "looped_bitnet" / "model.py",
    "model_quantization": ROOT / "looped_bitnet" / "quantization.py",
    "model_runtime": ROOT / "looped_bitnet" / "runtime.py",
    "model_step_budget": ROOT / "looped_bitnet" / "step_budget_e18.py",
    "model_float_qat": ROOT / "looped_bitnet" / "float_qat_e16.py",
    "model_w4": ROOT / "looped_bitnet" / "w4_e27.py",
}


def _science_source_inventory(root: Path = ROOT) -> dict[str, dict[str, str]]:
    paths = {
        "runtime": Path(__file__).resolve(),
        "scratchpad": Path(pure.__file__).resolve(),
        "e36_loader": Path(wave.__file__).resolve(),
        "e33_loader": Path(e33.__file__).resolve(),
        "windows_compatibility_adapter": Path(migration.__file__).resolve(),
        "optimizer_adapter": Path(old.__file__).resolve(),
        "width_model": Path(width.__file__).resolve(),
        "register_core": Path(dsl.__file__).resolve(),
        "protocol": _resolve(PROTOCOL, root),
        **_SCIENCE_TRANSITIVE_PATHS,
    }
    result = {}
    for name, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"science source is missing: {path}")
        result[name] = {"path": _relative(path, root), "sha256": _sha256(path)}
    return result


def _accepted_qa_binding(root: Path = ROOT) -> dict[str, Any]:
    accept_path = _resolve(QA_ACCEPT, root)
    if not accept_path.is_file() or _sha256(accept_path) != SCIENCE_ACCEPT_HASH:
        raise ValueError("accepted QA binding changed")
    accepted = json.loads(accept_path.read_text(encoding="utf-8"))
    if accepted.get("status") != "QA_ACCEPT" or not isinstance(accepted.get("qa_files"), Mapping):
        raise ValueError("accepted QA record is malformed")
    source_path = _resolve(QA_ACCEPTED_SOURCE, root)
    if not source_path.is_dir():
        raise ValueError("accepted QA source snapshot is missing")
    source_files = {}
    for name, expected in _ACCEPTED_SOURCE_HASHES.items():
        path = source_path / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"accepted QA source snapshot changed: {name}")
        source_files[name] = {"path": _relative(path, root), "sha256": expected}
    qa_files = {}
    qa_path = _resolve(QA_ACCEPTED_OUTPUT, root)
    for relative, expected in accepted["qa_files"].items():
        path = qa_path / Path(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"accepted QA artifact changed: {relative}")
        qa_files[str(Path(relative))] = {"path": _relative(path, root), "sha256": expected}
    return {
        "accept": {"path": _relative(accept_path, root), "sha256": SCIENCE_ACCEPT_HASH},
        "qa_files": qa_files,
        "accepted_source_files": source_files,
    }


def _validate_science_batch_shape(batches: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    if len(batches) != SCIENCE_BATCHES:
        raise ValueError("accepted B science update count changed")
    lengths = tuple(len(batch[0].program) for batch in batches)
    if any(len(batch) != SCIENCE_BATCH_SIZE for batch in batches):
        raise ValueError("accepted B science batch budget changed")
    if any(any(len(example.program) != length for example in batch) for batch, length in zip(batches, lengths)):
        raise ValueError("accepted B science batches are not homogeneous")
    return lengths


def _science_stream_batches(root: Path = ROOT) -> tuple[list[list[dsl.RegisterExample]], dict[str, Any]]:
    _a_stream, b_stream = wave.build_streams()
    batches = [list(batch) for batch in b_stream[:SCIENCE_BATCHES]]
    lengths = _validate_science_batch_shape(batches)
    if lengths != tuple(pure.FIXED_BATCH_LENGTHS):
        raise ValueError("accepted B science length cycle changed")
    if sum(lengths) != 6996:
        raise ValueError("accepted B science batch budget changed")
    raw = [[_example_record(example) for example in batch] for batch in batches]
    return batches, {
        "updates": SCIENCE_BATCHES,
        "batch_size": SCIENCE_BATCH_SIZE,
        "lengths": list(lengths),
        "length_sum": sum(lengths),
        "stream_digest": old.batch_digest(batches),
        "target_digest": old.target_digest(batches),
        "input_digest": _canonical_digest(raw),
        "batches": raw,
    }


def _science_batches_from_manifest(raw: Any) -> list[list[dsl.RegisterExample]]:
    if not isinstance(raw, list) or len(raw) != SCIENCE_BATCHES:
        raise ValueError("science batch count changed")
    batches: list[list[dsl.RegisterExample]] = []
    for raw_batch in raw:
        if not isinstance(raw_batch, list) or len(raw_batch) != SCIENCE_BATCH_SIZE:
            raise ValueError("science batch size changed")
        batch = []
        for item in raw_batch:
            if not isinstance(item, Mapping):
                raise ValueError("science example is malformed")
            example = dsl.RegisterExample(int(item["x"]), int(item["y"]), tuple(str(op) for op in item["program"]))
            if item.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("science target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("science batch is not homogeneous")
        batches.append(batch)
    return batches


def _baseline_program_specs(root: Path = ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_path = _resolve(BASELINE_ENDPOINT, root)
    proof_path = _resolve(BASELINE_PROOF, root)
    manifest_path = _resolve(BASELINE_MANIFEST, root)
    if not baseline_path.is_file() or _sha256(baseline_path) != BASELINE_ENDPOINT_SHA256:
        raise ValueError("saved baseline endpoint changed")
    if not proof_path.is_file() or _sha256(proof_path) != BASELINE_PROOF_SHA256:
        raise ValueError("saved exact-map proof changed")
    if not manifest_path.is_file() or _sha256(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise ValueError("saved baseline manifest changed")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("states") != [list(state) for state in SCIENCE_EVAL_STATES]:
        raise ValueError("saved exact-map state order changed")
    endpoint = json.loads(baseline_path.read_text(encoding="utf-8"))
    if endpoint.get("label") != SCIENCE_ENDPOINT_LABEL or endpoint.get("branch") != PARENT_BRANCH:
        raise ValueError("saved baseline endpoint identity changed")
    if endpoint.get("checkpoint_sha256") != SCIENCE_PARENT_CHECKPOINT_SHA256:
        raise ValueError("saved baseline checkpoint join changed")
    checkpoint = _resolve(Path(endpoint.get("checkpoint", "")), root)
    if not checkpoint.is_file() or _sha256(checkpoint) != SCIENCE_PARENT_CHECKPOINT_SHA256:
        raise ValueError("saved baseline checkpoint bytes changed")
    specs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for suite in ("padding", "compositions"):
        values = endpoint.get(suite)
        if not isinstance(values, list):
            raise ValueError(f"saved baseline {suite} rows are missing")
        for row in values:
            if not isinstance(row, Mapping) or not isinstance(row.get("program"), list):
                raise ValueError("saved baseline program row is malformed")
            program = tuple(str(op) for op in row["program"])
            if not program or any(op not in dsl.OPS for op in program) or int(row.get("length", -1)) != len(program):
                raise ValueError("saved baseline program identity changed")
            specs.append({"id": str(row["id"]), "suite": suite, "length": len(program), "program": list(program)})
            rows.append(dict(row) | {"suite": suite, "program": list(program), "length": len(program)})
    if len(specs) != pure.EVALUATION_CALLS:
        raise ValueError("saved baseline program count changed")
    if [item["length"] for item in specs] != list(pure.EVALUATION_PROGRAM_LENGTHS):
        raise ValueError("saved baseline evaluation length order changed")
    for row in rows:
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(SCIENCE_EVAL_STATES):
            raise ValueError("saved baseline state rows changed")
        for state, prediction in zip(SCIENCE_EVAL_STATES, predictions):
            if prediction.get("state") != list(state):
                raise ValueError("saved baseline state order changed")
            expected = [list(pair) for pair in dsl.RegisterExample(state[0], state[1], tuple(row["program"])).targets]
            if prediction.get("target_trace") != expected:
                raise ValueError("saved baseline target trace changed")
    binding = {
        "manifest": {"path": _relative(manifest_path, root), "sha256": BASELINE_MANIFEST_SHA256},
        "proof": {"path": _relative(proof_path, root), "sha256": BASELINE_PROOF_SHA256},
        "endpoint": {"path": _relative(baseline_path, root), "sha256": BASELINE_ENDPOINT_SHA256},
        "checkpoint": {"path": _relative(checkpoint, root), "sha256": SCIENCE_PARENT_CHECKPOINT_SHA256,
                       "model_digest": str(endpoint.get("model_digest"))},
        "program_ids": [item["id"] for item in specs],
        "programs": specs,
        "states": [list(state) for state in SCIENCE_EVAL_STATES],
    }
    return specs, rows, binding


def _science_config(source: Mapping[str, Any], settings: Mapping[str, Any], stream: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "pc_learned_scratchpad_science_config_v1",
        "endpoint": SCIENCE_ENDPOINT_LABEL,
        "arm_order": list(CHILD_ARMS),
        "parent_update": PARENT_UPDATE,
        "updates_per_arm": SCIENCE_UPDATES,
        "batch_size": SCIENCE_BATCH_SIZE,
        "batch_lengths": list(pure.FIXED_BATCH_LENGTHS),
        "stream_length_sum": stream["length_sum"],
        "eval_programs": pure.EVALUATION_CALLS,
        "eval_states": len(SCIENCE_EVAL_STATES),
        "eval_sequence": ["initial_soft", "final_continuous", "final_soft"],
        "loss": "summed_x_y_cross_entropy_means",
        "runtime_settings": dict(settings),
        "source_digest": _canonical_digest(source),
    }


def _verify_parent_static_binding(root: Path = ROOT) -> None:
    manifest_path = _resolve(E36_PREFLIGHT / "manifest.json", root)
    checkpoint = _parent_checkpoint(root)
    if not manifest_path.is_file() or _sha256(manifest_path) != E36_MANIFEST_SHA256:
        raise ValueError("accepted E36 manifest binding changed")
    if not checkpoint.is_file() or _sha256(checkpoint) != SCIENCE_PARENT_CHECKPOINT_SHA256:
        raise ValueError("accepted E36 parent checkpoint binding changed")


def prepare_science(*, out: Path = SCIENCE_MANIFEST, root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite science manifest {out}")
    _device, settings = _configure_runtime()
    _verify_parent_static_binding(root)
    source = _science_source_inventory(root)
    qa_binding = _accepted_qa_binding(root)
    batches, stream = _science_stream_batches(root)
    specs, _baseline_rows, baseline = _baseline_program_specs(root)
    config = _science_config(source, settings, stream)
    manifest: dict[str, Any] = {
        "schema": SCIENCE_SCHEMA,
        "parent": {
            "label": PARENT_LABEL,
            "arm": PARENT_ARM,
            "branch": PARENT_BRANCH,
            "seed": 0,
            "width": 128,
            "update": PARENT_UPDATE,
            "checkpoint": _relative(_parent_checkpoint(root), root),
            "checkpoint_sha256": SCIENCE_PARENT_CHECKPOINT_SHA256,
            "e36_manifest": _relative(_resolve(E36_PREFLIGHT / "manifest.json", root), root),
            "e36_manifest_sha256": E36_MANIFEST_SHA256,
        },
        "source": source,
        "accepted_qa": qa_binding,
        "baseline": baseline,
        "stream": stream,
        "programs": specs,
        "states": [list(state) for state in SCIENCE_EVAL_STATES],
        "strata": {f"{state[0]},{state[1]}": SCIENCE_STATE_STRATUM[state] for state in SCIENCE_EVAL_STATES},
        "config": config,
        "budget": dict(pure.SCIENCE_ACCOUNTING),
    }
    manifest["source_digest"] = _canonical_digest(source)
    manifest["config_digest"] = _canonical_digest(config)
    manifest["manifest_digest"] = _canonical_digest(manifest)
    _write_json(out, manifest, refuse=True)
    return manifest


def _validate_science_manifest(path: Path, *, root: Path, settings: Mapping[str, Any]) -> tuple[dict[str, Any], list[list[dsl.RegisterExample]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = _resolve(path, root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCIENCE_SCHEMA:
        raise ValueError("science manifest schema changed")
    digest_payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if manifest.get("manifest_digest") != _canonical_digest(digest_payload):
        raise ValueError("science manifest digest changed")
    source = _science_source_inventory(root)
    _verify_parent_static_binding(root)
    if manifest.get("source") != source or manifest.get("source_digest") != _canonical_digest(source):
        raise ValueError("science source binding changed")
    if manifest.get("accepted_qa") != _accepted_qa_binding(root):
        raise ValueError("accepted QA binding changed after preparation")
    batches, stream = _science_stream_batches(root)
    if manifest.get("stream") != stream:
        raise ValueError("science stream binding changed")
    batches_from_manifest = _science_batches_from_manifest(manifest.get("stream", {}).get("batches"))
    if old.batch_digest(batches_from_manifest) != stream["stream_digest"] or old.target_digest(batches_from_manifest) != stream["target_digest"]:
        raise ValueError("science stream digest changed")
    specs, baseline_rows, baseline = _baseline_program_specs(root)
    if manifest.get("programs") != specs or manifest.get("baseline") != baseline:
        raise ValueError("science baseline or program binding changed")
    if manifest.get("states") != [list(state) for state in SCIENCE_EVAL_STATES]:
        raise ValueError("science evaluation state order changed")
    if manifest.get("budget") != pure.SCIENCE_ACCOUNTING:
        raise ValueError("science budget binding changed")
    if manifest.get("config", {}).get("runtime_settings") != dict(settings):
        raise ValueError("science runtime settings changed")
    expected_parent = {
        "label": PARENT_LABEL, "arm": PARENT_ARM, "branch": PARENT_BRANCH, "seed": 0, "width": 128,
        "update": PARENT_UPDATE, "checkpoint": _relative(_parent_checkpoint(root), root),
        "checkpoint_sha256": SCIENCE_PARENT_CHECKPOINT_SHA256,
        "e36_manifest": _relative(_resolve(E36_PREFLIGHT / "manifest.json", root), root),
        "e36_manifest_sha256": E36_MANIFEST_SHA256,
    }
    if manifest.get("parent") != expected_parent:
        raise ValueError("science parent binding changed")
    return manifest, batches_from_manifest, specs, baseline_rows


def _counter_mark(counter: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.items() if key.startswith("attempted_") or key.startswith("completed_")}


def _phase_record(counter: Mapping[str, Any], before: Mapping[str, int], failure_index: int, failures: Sequence[Any], elapsed: float) -> dict[str, Any]:
    result = {key: int(counter.get(key, 0)) - int(before.get(key, 0)) for key in before}
    result["failures"] = list(failures[failure_index:])
    result["elapsed_seconds"] = elapsed
    return result


@contextmanager
def _preserve_evaluation_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Iterator[None]:
    before = _state_identity(model, optimizer)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    training_mode = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        model.train(training_mode)
        after = _state_identity(model, optimizer)
        for key in ("model_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest", "training_mode", "parameter_names"):
            if after[key] != before[key]:
                raise ValueError(f"evaluation changed preserved state: {key}")


def _trace_metrics(target: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> dict[str, Any]:
    prefix = [bool(tuple(a) == tuple(b)) for a, b in zip(target, predicted)]
    if len(prefix) != len(target):
        raise ValueError("trace length changed")
    first = next((index + 1 for index, correct in enumerate(prefix) if not correct), None)
    final = bool(prefix[-1]) if prefix else False
    return {
        "prefix_joint_correct": prefix,
        "joint_final_correct": final,
        "full_trace_correct": all(prefix),
        "first_error": first,
        "recovered_final": bool(first is not None and final),
    }


def _target_bit_deviation(
    probabilities_x: torch.Tensor,
    probabilities_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Return x+y mean absolute four-bit error against the true DSL targets."""
    target_bits_x = matrix[target_x]
    target_bits_y = matrix[target_y]
    return ((probabilities_x @ matrix - target_bits_x).abs().mean((0, 2))
            + (probabilities_y @ matrix - target_bits_y).abs().mean((0, 2)))


def _evaluate_science(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    mode: str,
    specs: Sequence[Mapping[str, Any]],
    states: Sequence[tuple[int, int]],
    device: torch.device,
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    if mode not in CHILD_ARMS:
        raise ValueError("unknown science evaluation mode")
    rows: list[dict[str, Any]] = []
    with _preserve_evaluation_state(model, optimizer):
        for spec in specs:
            program = tuple(str(op) for op in spec["program"])
            batch = [dsl.RegisterExample(state[0], state[1], program) for state in states]
            cases = len(batch)
            length = len(program)
            counter["attempted_forwards"] += 1
            counter["attempted_cases"] += cases
            counter["attempted_readout_positions"] += cases * length
            counter["attempted_native_steps"] += cases * length * pure.NATIVE_STEPS
            _flush(counter, sink)
            try:
                with torch.inference_mode():
                    logits_x, logits_y = _forward_logits(
                        model, batch, arm=mode, device=device, expected_cases=cases
                    )
                probabilities_x = torch.softmax(logits_x, dim=-1)
                probabilities_y = torch.softmax(logits_y, dim=-1)
                decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
                decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
                predicted = [
                    [[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)]
                    for index in range(cases)
                ]
                matrix = pure.signed_bit_matrix(device=device, dtype=probabilities_x.dtype)
                entropy = (-(probabilities_x.clamp_min(1e-12) * probabilities_x.clamp_min(1e-12).log()).sum(-1)
                            + -(probabilities_y.clamp_min(1e-12) * probabilities_y.clamp_min(1e-12).log()).sum(-1)).mean(0)
                target_x = torch.tensor(
                    [[target[0] for target in example.targets] for example in batch],
                    dtype=torch.long, device=device,
                )
                target_y = torch.tensor(
                    [[target[1] for target in example.targets] for example in batch],
                    dtype=torch.long, device=device,
                )
                bit_deviation = _target_bit_deviation(
                    probabilities_x, probabilities_y, target_x, target_y, matrix
                )
                counter["completed_forwards"] += 1
                counter["completed_cases"] += cases
                counter["completed_readout_positions"] += cases * length
                counter["completed_native_steps"] += cases * length * pure.NATIVE_STEPS
                _flush(counter, sink)
            except BaseException as exc:
                _failure(counter, "evaluation_forward", exc, phase=str(spec["id"]))
                _flush(counter, sink)
                raise
            predictions = []
            for state, example, trace in zip(states, batch, predicted):
                target = [list(pair) for pair in example.targets]
                metrics = _trace_metrics(target, trace)
                predictions.append({
                    "state": list(state),
                    "stratum": SCIENCE_STATE_STRATUM[state],
                    "target_trace": target,
                    "predicted_trace": trace,
                    **metrics,
                })
            rows.append({
                "id": str(spec["id"]),
                "suite": str(spec["suite"]),
                "length": length,
                "program": list(program),
                "predictions": predictions,
                "writing_diagnostics": {
                    "entropy_per_position": [float(value) for value in entropy.detach().cpu().tolist()],
                    "expected_bit_deviation_per_position": [float(value) for value in bit_deviation.detach().cpu().tolist()],
                    "expected_bit_deviation_definition": "x+y sum of per-register mean absolute four-bit errors against true DSL target bits",
                },
            })
    return rows


def _normalise_baseline_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised = []
    for row in rows:
        predictions = []
        for prediction in row["predictions"]:
            target = prediction["target_trace"]
            predicted = prediction["predicted_trace"]
            predictions.append({
                "state": list(prediction["state"]),
                "stratum": SCIENCE_STATE_STRATUM[tuple(prediction["state"])],
                "target_trace": [list(pair) for pair in target],
                "predicted_trace": [list(pair) for pair in predicted],
                **_trace_metrics(target, predicted),
            })
        normalised.append({
            "id": str(row["id"]), "suite": str(row["suite"]), "length": int(row["length"]),
            "program": list(row["program"]), "predictions": predictions,
        })
    return normalised


def _outcome(left: bool, right: bool) -> str:
    if left and right:
        return "both_correct"
    if left:
        return "left_correct_right_wrong"
    if right:
        return "left_wrong_right_correct"
    return "both_wrong"


def _empty_outcomes() -> dict[str, int]:
    return {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0}


def _paired_metrics(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    left_by_id = {str(row["id"]): row for row in left_rows}
    right_by_id = {str(row["id"]): row for row in right_rows}
    if set(left_by_id) != set(right_by_id):
        raise ValueError(f"paired program IDs differ: {label}")
    aggregate = {"final": _empty_outcomes(), "full_trace": _empty_outcomes()}
    groups: dict[str, dict[str, dict[str, int]]] = {}
    programs = []
    for identifier in sorted(left_by_id):
        left = left_by_id[identifier]
        right = right_by_id[identifier]
        if left["program"] != right["program"] or left["suite"] != right["suite"]:
            raise ValueError(f"paired program identity differs: {identifier}")
        right_states = {tuple(item["state"]): item for item in right["predictions"]}
        program_counts = {"final": _empty_outcomes(), "full_trace": _empty_outcomes()}
        left_first: dict[str, int] = {}
        right_first: dict[str, int] = {}
        left_recovery = right_recovery = 0
        for item in left["predictions"]:
            state = tuple(item["state"])
            other = right_states.get(state)
            if other is None:
                raise ValueError(f"paired state identity differs: {identifier}")
            final_key = _outcome(bool(item["joint_final_correct"]), bool(other["joint_final_correct"]))
            full_key = _outcome(bool(item["full_trace_correct"]), bool(other["full_trace_correct"]))
            program_counts["final"][final_key] += 1
            program_counts["full_trace"][full_key] += 1
            aggregate["final"][final_key] += 1
            aggregate["full_trace"][full_key] += 1
            first_left = "none" if item["first_error"] is None else str(item["first_error"])
            first_right = "none" if other["first_error"] is None else str(other["first_error"])
            left_first[first_left] = left_first.get(first_left, 0) + 1
            right_first[first_right] = right_first.get(first_right, 0) + 1
            left_recovery += int(bool(item["recovered_final"]))
            right_recovery += int(bool(other["recovered_final"]))
            keys = [str(item["stratum"]), "heldout64" if item["stratum"] in {"validation", "test"} else None]
            for key in keys:
                if key is None:
                    continue
                group = groups.setdefault(key, {"final": _empty_outcomes(), "full_trace": _empty_outcomes()})
                group["final"][final_key] += 1
                group["full_trace"][full_key] += 1
        programs.append({
            "id": identifier, "suite": left["suite"], "length": left["length"], "program": list(left["program"]),
            "final": program_counts["final"], "full_trace": program_counts["full_trace"],
            "left_first_error_histogram": left_first, "right_first_error_histogram": right_first,
            "left_recovered_final": left_recovery, "right_recovered_final": right_recovery,
        })
    by_suite: dict[str, dict[str, dict[str, int]]] = {}
    by_length: dict[str, dict[str, dict[str, int]]] = {}
    for program in programs:
        for target, key in ((by_suite, str(program["suite"])), (by_length, str(program["length"]))):
            group = target.setdefault(key, {"final": _empty_outcomes(), "full_trace": _empty_outcomes()})
            for metric in ("final", "full_trace"):
                for outcome, count in program[metric].items():
                    group[metric][outcome] += count
    return {
        "label": label,
        "endpoint": SCIENCE_ENDPOINT_LABEL,
        "roles": {"left": "candidate", "right": "reference"},
        "programs": programs,
        "aggregate": aggregate,
        "by_suite": by_suite,
        "by_length": by_length,
        "by_stratum": groups,
        "ties": {metric: aggregate[metric]["both_correct"] + aggregate[metric]["both_wrong"] for metric in aggregate},
        "improvements": {"final": aggregate["final"]["left_correct_right_wrong"],
                          "full_trace": aggregate["full_trace"]["left_correct_right_wrong"]},
        "regressions": {"final": aggregate["final"]["left_wrong_right_correct"],
                         "full_trace": aggregate["full_trace"]["left_wrong_right_correct"]},
    }


def _science_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    parent_identity: Mapping[str, Any],
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = _cpu_copy(model.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    payload: dict[str, Any] = {
        "schema": "pc_learned_scratchpad_science_checkpoint_v1",
        "arm": arm,
        "branch": PARENT_BRANCH,
        "width": 128,
        "seed": 0,
        "parent_update": PARENT_UPDATE,
        "added_updates": SCIENCE_UPDATES,
        "update": PARENT_UPDATE + SCIENCE_UPDATES,
        "training_mode": bool(model.training),
        "identity": dict(identity),
        "parent_identity": dict(parent_identity),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "parameter_names": list(model.state_dict()),
        "stream_digest": manifest["stream"]["stream_digest"],
        "target_digest": manifest["stream"]["target_digest"],
        "source_digest": manifest["source_digest"],
        "config_digest": manifest["config_digest"],
    }
    payload["model_digest"] = old.digest_object(model_state)
    payload["optimizer_digest"] = old.digest_object(optimizer_state)
    payload["cpu_rng_digest"] = old.digest_object(cpu_rng)
    payload["cuda_rng_digest"] = old.digest_object(cuda_rng)
    payload["payload_digest"] = old.digest_object({key: payload[key] for key in payload if key not in {"payload_digest"}})
    return payload


def _run_science_arm(
    *,
    arm: str,
    batches: Sequence[Sequence[dsl.RegisterExample]],
    specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    device: torch.device,
    common_cuda_rng: Sequence[torch.Tensor],
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
    out: Path,
    initial_soft: bool,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    checkpoint = _parent_checkpoint(root)
    load_before = _counter_mark(counter)
    load_failure_index = len(counter["failures"])
    load_started = time.monotonic()
    model, optimizer, parent = _load_parent(arm=arm, manifest=followup._load_inherited(root), counter=counter, sink=sink, root=root)
    load_phase = _phase_record(counter, load_before, load_failure_index, counter["failures"], time.monotonic() - load_started)
    model.to(device)
    _optimizer_to_device(optimizer, device)
    torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
    torch.set_rng_state(parent["rng_state"])
    model.train(True)
    parent_identity = _child_parent_identity(parent, checkpoint, common_cuda_rng)
    identity_fixture = {
        "stream_digest": manifest["stream"]["stream_digest"],
        "target_digest": manifest["stream"]["target_digest"],
        "source_digest": manifest["source_digest"],
        "config_digest": manifest["config_digest"],
    }
    identity_u0 = _arm_identity(parent_identity, arm=arm, local_update=0, fixture=identity_fixture)
    initial_rows = None
    evaluation_phases: dict[str, Any] = {"parent_load": load_phase}
    if initial_soft:
        phase_dir = out / "evaluations"
        phase_dir.mkdir(parents=True, exist_ok=True)
        eval_before = _counter_mark(counter)
        eval_failure_index = len(counter["failures"])
        eval_started = time.monotonic()
        initial_rows = _evaluate_science(
            model, optimizer, mode="soft_register_reset", specs=specs, states=SCIENCE_EVAL_STATES,
            device=device, counter=counter, sink=sink,
        )
        evaluation_phases["initial_soft"] = _phase_record(
            counter, eval_before, eval_failure_index, counter["failures"], time.monotonic() - eval_started
        )
        _write_json(phase_dir / "initial_soft.json", {"schema": "pc_learned_scratchpad_evaluation_v1", "phase": "initial_soft", "rows": initial_rows}, refuse=True)
    losses = []
    train_before = _counter_mark(counter)
    train_failure_index = len(counter["failures"])
    train_started = time.monotonic()
    for local_update, batch in enumerate(batches, 1):
        loss = _run_update(
            model, optimizer, batch, arm=arm, device=device, counter=counter, sink=sink,
            expected_cases=SCIENCE_BATCH_SIZE,
        )
        losses.append({
            "local_update": local_update,
            "absolute_update": PARENT_UPDATE + local_update,
            "length": len(batch[0].program),
            "cases": len(batch),
            "readout_positions": len(batch) * len(batch[0].program),
            "loss": loss,
        })
    train_phase = _phase_record(counter, train_before, train_failure_index, counter["failures"], time.monotonic() - train_started)
    identity_final = _arm_identity(parent_identity, arm=arm, local_update=SCIENCE_UPDATES, fixture=identity_fixture)
    final_payload = _science_checkpoint_payload(
        model, optimizer, arm=arm, parent_identity=parent_identity, identity=identity_final, manifest=manifest,
    )
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = arm_dir / "u42000.pt"
    _write_torch(checkpoint_path, final_payload, refuse=True)
    final_mode = "final_soft" if arm == "soft_register_reset" else "final_continuous"
    eval_before = _counter_mark(counter)
    eval_failure_index = len(counter["failures"])
    eval_started = time.monotonic()
    final_rows = _evaluate_science(
        model, optimizer, mode=arm, specs=specs, states=SCIENCE_EVAL_STATES,
        device=device, counter=counter, sink=sink,
    )
    evaluation_phases[final_mode] = _phase_record(
        counter, eval_before, eval_failure_index, counter["failures"], time.monotonic() - eval_started
    )
    _write_json(out / "evaluations" / f"{final_mode}.json", {"schema": "pc_learned_scratchpad_evaluation_v1", "phase": final_mode, "rows": final_rows}, refuse=True)
    record = {
        "schema": "pc_learned_scratchpad_science_arm_v1",
        "arm": arm,
        "parent_identity": parent_identity,
        "identity_u0": identity_u0,
        "identity_final": identity_final,
        "training": train_phase,
        "phases": evaluation_phases,
        "losses": losses,
        "checkpoint": {"path": _relative(checkpoint_path, root), "sha256": _sha256(checkpoint_path),
                       "payload_digest": final_payload["payload_digest"], "model_digest": final_payload["model_digest"],
                       "optimizer_digest": final_payload["optimizer_digest"]},
        "evaluation_phase": final_mode,
        "status": "complete",
    }
    _write_json(arm_dir / "report.json", record, refuse=True)
    return record, initial_rows, final_rows


def run_science(*, out: Path = SCIENCE_OUTPUT, manifest_path: Path = SCIENCE_MANIFEST, root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    manifest_path = _resolve(manifest_path, root)
    _refuse_nonempty(out)
    counter = _new_counter()
    sink = _counter_sink(out / "accounting.json")
    sink(counter)
    report: dict[str, Any] = {"schema": SCIENCE_RUN_SCHEMA, "status": "running", "manifest": {"path": _relative(manifest_path, root)}}
    _write_json(out / "status.json", report, refuse=True)
    try:
        device, settings = _configure_runtime()
        manifest, batches, specs, baseline_saved = _validate_science_manifest(manifest_path, root=root, settings=settings)
        report["manifest"]["sha256"] = _sha256(manifest_path)
        report["runtime"] = settings
        report["source"] = manifest["source"]
        report["phases"] = {}
        report["evaluations"] = {}
        report["models"] = {}
        common_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        with migration.windows_compatibility_adapter(), _count_deserializations(counter, sink):
            continuous, initial_rows, final_control = _run_science_arm(
                arm="continuous_control", batches=batches, specs=specs, manifest=manifest, device=device,
                common_cuda_rng=common_cuda_rng, counter=counter, sink=sink, out=out,
                initial_soft=True, root=root,
            )
            report["models"]["continuous_control"] = continuous
            report["phases"]["continuous_control"] = continuous["phases"]
            report["evaluations"]["initial_soft"] = {
                "path": _relative(out / "evaluations" / "initial_soft.json", root),
                "sha256": _sha256(out / "evaluations" / "initial_soft.json"), "rows": len(initial_rows or []),
            }
            report["evaluations"]["final_continuous"] = {
                "path": _relative(out / "evaluations" / "final_continuous.json", root),
                "sha256": _sha256(out / "evaluations" / "final_continuous.json"), "rows": len(final_control),
            }
            soft, _unused_initial, final_soft = _run_science_arm(
                arm="soft_register_reset", batches=batches, specs=specs, manifest=manifest, device=device,
                common_cuda_rng=common_cuda_rng, counter=counter, sink=sink, out=out,
                initial_soft=False, root=root,
            )
            report["models"]["soft_register_reset"] = soft
            report["phases"]["soft_register_reset"] = soft["phases"]
            report["evaluations"]["final_soft"] = {
                "path": _relative(out / "evaluations" / "final_soft.json", root),
                "sha256": _sha256(out / "evaluations" / "final_soft.json"), "rows": len(final_soft),
            }
        baseline_rows = _normalise_baseline_rows(baseline_saved)
        report["paired"] = [
            _paired_metrics(initial_rows or [], baseline_rows, label="initial_soft_vs_savedparentcontinuous"),
            _paired_metrics(final_soft, initial_rows or [], label="final_soft_vs_initial_soft"),
            _paired_metrics(final_control, baseline_rows, label="final_control_vs_savedparentcontinuous"),
            _paired_metrics(final_soft, final_control, label="final_soft_vs_final_control"),
        ]
        report["accounting"] = dict(counter)
        report["budget"] = dict(pure.SCIENCE_ACCOUNTING)
        report["status"] = "complete"
        _write_json(out / "report.json", report, refuse=True)
        _write_json(out / "status.json", {"schema": SCIENCE_RUN_SCHEMA, "status": "complete", "report": _relative(out / "report.json", root)}, refuse=False)
        return report
    except BaseException as exc:
        _failure(counter, "science", exc)
        sink(counter)
        report["status"] = "failed"
        report["accounting"] = dict(counter)
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(out / "report.json", report, refuse=True)
        _write_json(out / "status.json", {"schema": SCIENCE_RUN_SCHEMA, "status": "failed", "report": _relative(out / "report.json", root)}, refuse=False)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="run the fixed disposable six-update serialization QA")
    parser.add_argument("--prepare-science", action="store_true", help="freeze the zero-load learned scratchpad science manifest")
    parser.add_argument("--science", action="store_true", help="run the registered learned scratchpad science extension")
    parser.add_argument("--out", type=Path, default=QA_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    args = parser.parse_args(argv)
    selected = sum(bool(value) for value in (args.qa, args.prepare_science, args.science))
    if selected != 1:
        parser.error("select exactly one of --qa, --prepare-science, or --science")
    if args.qa:
        run_qa(out=args.out)
    elif args.prepare_science:
        prepare_science(out=args.manifest)
    else:
        run_science(out=args.out, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
