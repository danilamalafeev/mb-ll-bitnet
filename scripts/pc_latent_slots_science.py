"""Latent-slot science orchestration with a separate immutable prepare phase.

The module deliberately keeps the accepted latent numerical helper and the
accepted QA loader/optimizer path as dependencies.  ``--prepare-science``
only hashes and serializes data/evidence.  ``--science`` is the separately
authorized model run and has no resume or retry mode.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from scripts import continuation_e33 as e33
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import pc_inference_migration as migration
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_runtime as qa
from scripts import pc_learned_scratchpad as accepted
from scripts import pc_learned_scratchpad_runtime as accepted_runtime


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "pc_latent_slots_v1"
SCIENCE_MANIFEST = RUN_ROOT / "science_manifest.json"
SCIENCE_MANIFEST_DIGEST = RUN_ROOT / "science_manifest.sha256"
SCIENCE_STREAM = RUN_ROOT / "science_stream.json"
SCIENCE_SCOPE = RUN_ROOT / "science_eval_scope.json"
SCIENCE_OUTPUT = RUN_ROOT / "science"
QA_ACCEPT = RUN_ROOT / "qa_accept.json"
QA_OUTPUT = RUN_ROOT / "qa"
QA_ACCEPTED_SOURCE = RUN_ROOT / "qa_accepted_source"
BASELINE = ROOT / "runs" / "pc_learned_scratchpad_v1" / "science" / "evaluations" / "initial_soft.json"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LATENT_SLOTS_PROTOCOL.md"
SCIENCE_INTERFACE = RUN_ROOT / "SCIENCE_INTERFACE.md"
SCIENCE_SOURCE_INVENTORY = RUN_ROOT / "science_source_inventory_v1.json"
PARENT_LABEL = qa.PARENT_LABEL
PARENT_ARM = qa.PARENT_ARM
PARENT_BRANCH = qa.PARENT_BRANCH
PARENT_UPDATE = qa.PARENT_UPDATE
PARENT_CHECKPOINT_SHA256 = qa.PARENT_CHECKPOINT_SHA256
E36_PREFLIGHT = qa.E36_PREFLIGHT
E36_RUN = qa.E36_RUN
E36_MANIFEST_SHA256 = qa.E36_MANIFEST_SHA256
SCIENCE_SCHEMA = "pc_latent_slots_science_manifest_v1"
RUN_SCHEMA = "pc_latent_slots_science_run_v1"
STREAM_SCHEMA = "pc_latent_slots_v1_stream_freeze"
SCOPE_SCHEMA = "pc_latent_slots_v1_evaluation_scope"
CHECKPOINT_SCHEMA = "pc_latent_slots_science_checkpoint_v1"
CHECKPOINT_INDICES = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
TRAINING_UPDATES = 2000
TRAINING_BATCH_SIZE = 64
ACCEPTED_B_STREAM_BATCHES = 8000
EVALUATION_PROGRAMS = 69
EVALUATION_STATES = 256
EVALUATION_CALLS = 69
SCIENCE_CALLS = 2138
SCIENCE_CASES = 163328
SCIENCE_POSITIONS = 1111296
SCIENCE_NATIVE_STEPS = 8890368
SCIENCE_UPDATES = 2000
QA_ACCEPT_SHA256 = "590d1a5e67940f96d58889dafdb8d4bfa29502e9915c806b746199022b05bfba"
AUDIT_QA_SHA256 = "8547032e9d369e4a26dbcc013e39a6c218e0631b2de12173ac209715b3eb323b"
SCIENCE_INTERFACE_SHA256 = "1e6676b4541a6aba20c8388f92ea2568276eeb5b8d44920ce8889d889bbb7251"
LATENT_PURE_SHA256 = "ca1572b54f2ee1d88d310389584fc0af423f04b9a1dcee23abd175bb16af4752"
LATENT_QA_RUNTIME_SHA256 = "36f0dd5b2d69189a6abd8533bbe4d559bec23253de3ebaae08854a9f028b8b50"


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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


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


def _digest(value: Any) -> str:
    return old.digest_object(value)


def _path_from_record(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _resolve(path, root)


def _qa_binding(root: Path = ROOT) -> dict[str, Any]:
    accept_path = _resolve(QA_ACCEPT, root)
    if not accept_path.is_file() or _sha256(accept_path) != QA_ACCEPT_SHA256:
        raise ValueError("latent QA acceptance bytes changed")
    accepted_record = _read_object(accept_path)
    if accepted_record.get("status") != "QA_ACCEPT":
        raise ValueError("latent QA acceptance status changed")
    qa_files_record = accepted_record.get("qa_files")
    source_record = accepted_record.get("sources")
    if not isinstance(qa_files_record, Mapping) or not isinstance(source_record, Mapping):
        raise ValueError("latent QA acceptance inventory is malformed")
    qa_files: dict[str, dict[str, str]] = {}
    for relative, expected in qa_files_record.items():
        path = _resolve(QA_OUTPUT / Path(str(relative)), root)
        if not path.is_file() or _sha256(path) != str(expected):
            raise ValueError(f"latent QA artifact changed: {relative}")
        qa_files[str(Path(str(relative)))] = {"path": _relative(path, root), "sha256": str(expected)}
    sources: dict[str, dict[str, str]] = {}
    snapshots: dict[str, dict[str, str]] = {}
    for current_text, raw in source_record.items():
        if not isinstance(raw, Mapping) or "sha256" not in raw or "snapshot" not in raw:
            raise ValueError("latent QA source record is malformed")
        current = _path_from_record(str(current_text), root)
        snapshot = _path_from_record(str(raw["snapshot"]), root)
        expected = str(raw["sha256"])
        if not current.is_file() or _sha256(current) != expected:
            raise ValueError(f"latent QA reviewed source changed: {current_text}")
        if not snapshot.is_file() or _sha256(snapshot) != expected:
            raise ValueError(f"latent QA source snapshot changed: {raw['snapshot']}")
        sources[str(current_text)] = {"path": _relative(current, root), "sha256": expected}
        snapshots[str(raw["snapshot"])] = {"path": _relative(snapshot, root), "sha256": expected}
    snapshot_dir = _resolve(QA_ACCEPTED_SOURCE, root)
    actual_snapshot_paths = {
        str(path.resolve()) for path in snapshot_dir.rglob("*") if path.is_file()
    }
    expected_snapshot_paths = {str(_path_from_record(path, root).resolve()) for path in snapshots}
    if actual_snapshot_paths != expected_snapshot_paths:
        raise ValueError("latent QA source snapshot inventory changed")
    return {
        "path": _relative(accept_path, root),
        "sha256": QA_ACCEPT_SHA256,
        "qa_files": qa_files,
        "sources": sources,
        "snapshots": snapshots,
    }


def _source_inventory(root: Path = ROOT) -> dict[str, Any]:
    qa_binding = _qa_binding(root)
    inventory_path = _resolve(SCIENCE_SOURCE_INVENTORY, root)
    if not inventory_path.is_file():
        raise ValueError("science static source inventory is missing")
    raw_inventory = _read_object(inventory_path)
    if not raw_inventory or any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw_inventory.items()):
        raise ValueError("science static source inventory is malformed")
    files: dict[str, dict[str, str]] = {}
    for relative, expected in raw_inventory.items():
        path = _resolve(Path(relative.replace("\\", "/")), root)
        if not path.is_file():
            raise ValueError(f"science source is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"science source hash changed: {relative}")
        files[relative] = {"path": _relative(path, root), "sha256": actual}
    expected_paths = {str(_path_from_record(item["path"], root).resolve()) for item in files.values()}
    qa_paths: set[str] = set()
    root_path = Path(root).resolve()
    for item in qa_binding["sources"].values():
        candidate = _path_from_record(item["path"], root).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError:
            continue
        if candidate.suffix.lower() != ".py":
            continue
        qa_paths.add(str(candidate))
    if not qa_paths.issubset(expected_paths):
        raise ValueError("science static inventory omits accepted QA source")
    latent_path = _resolve(Path("scripts/pc_latent_slots.py"), root)
    runtime_path = _resolve(Path("scripts/pc_latent_slots_runtime.py"), root)
    if files.get("scripts\\pc_latent_slots.py", {}).get("sha256") != LATENT_PURE_SHA256 or _sha256(latent_path) != LATENT_PURE_SHA256:
        raise ValueError("accepted latent pure helper changed")
    if files.get("scripts\\pc_latent_slots_runtime.py", {}).get("sha256") != LATENT_QA_RUNTIME_SHA256 or _sha256(runtime_path) != LATENT_QA_RUNTIME_SHA256:
        raise ValueError("accepted latent QA runtime changed")
    inventory = {
        "schema": "pc_latent_slots_science_source_inventory_v1",
        "inventory": {"path": _relative(inventory_path, root), "sha256": _sha256(inventory_path)},
        "files": files,
        "qa_binding": qa_binding,
    }
    inventory["digest"] = _digest(inventory)
    return inventory


def _runtime_settings() -> dict[str, Any]:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "device": "cuda",
        "dtype": "torch.float32",
        "autocast": False,
        "deterministic_algorithms": True,
        "cpu_threads": 4,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "native_steps": latent.NATIVE_STEPS,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _configure_runtime() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("latent science requires CUDA")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    settings = _runtime_settings()
    if not settings["deterministic_algorithms"]:
        raise RuntimeError("deterministic algorithms were not enabled")
    return device, settings


def _expected_stream_lengths() -> tuple[int, ...]:
    return tuple(latent.FIXED_BATCH_LENGTHS)


def _accepted_b_stream_lengths() -> tuple[int, ...]:
    return (1, 2, 3, 4, 5, 6) * 1333 + (1, 4)


def _training_budget(lengths: Sequence[int], *, batch_size: int = TRAINING_BATCH_SIZE) -> dict[str, int]:
    values = tuple(int(value) for value in lengths)
    if values != _expected_stream_lengths():
        raise ValueError("latent training cycle or tail changed")
    return {
        "calls": len(values),
        "cases": len(values) * batch_size,
        "readout_positions": sum(values) * batch_size,
        "native_steps": sum(values) * batch_size * latent.NATIVE_STEPS,
        "optimizer_updates": len(values),
    }


def _example_record(example: dsl.RegisterExample) -> dict[str, Any]:
    return {
        "x": int(example.x),
        "y": int(example.y),
        "program": list(example.program),
        "target_trace": [list(pair) for pair in example.targets],
    }


def _validate_stream_batches(batches: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[list[list[dsl.RegisterExample]], dict[str, int]]:
    if len(batches) != TRAINING_UPDATES:
        raise ValueError("latent training batch count changed")
    examples: list[list[dsl.RegisterExample]] = []
    lengths: list[int] = []
    for raw_batch in batches:
        if len(raw_batch) != TRAINING_BATCH_SIZE:
            raise ValueError("latent training batch size changed")
        batch: list[dsl.RegisterExample] = []
        for raw in raw_batch:
            if not isinstance(raw, Mapping):
                raise ValueError("latent stream example is malformed")
            program = tuple(str(op) for op in raw["program"])
            example = dsl.RegisterExample(int(raw["x"]), int(raw["y"]), program)
            if raw.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("latent stream target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("latent stream batch is not homogeneous")
        lengths.append(len(batch[0].program))
        examples.append(batch)
    budget = _training_budget(lengths)
    return examples, budget


def _validate_accepted_b_stream(b_stream: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    if len(b_stream) != ACCEPTED_B_STREAM_BATCHES:
        raise ValueError("accepted B stream batch count changed")
    lengths: list[int] = []
    for batch in b_stream:
        if len(batch) != TRAINING_BATCH_SIZE:
            raise ValueError("accepted B stream batch size changed")
        length = len(tuple(getattr(batch[0], "program", ())))
        if any(len(tuple(getattr(example, "program", ()))) != length for example in batch):
            raise ValueError("accepted B stream batch is not homogeneous")
        lengths.append(length)
    expected = _accepted_b_stream_lengths()
    if tuple(lengths) != expected:
        raise ValueError("accepted B stream length schedule changed")
    return tuple(lengths)


def _select_b_training_prefix(b_stream: Sequence[Any]) -> list[Any]:
    if len(b_stream) != ACCEPTED_B_STREAM_BATCHES:
        raise ValueError("accepted B stream must be fully validated before prefix selection")
    prefix = list(b_stream[:TRAINING_UPDATES])
    if len(prefix) != TRAINING_UPDATES:
        raise ValueError("accepted B stream training prefix is incomplete")
    return prefix


def _freeze_stream(b_stream: Sequence[Sequence[dsl.RegisterExample]] | None = None) -> dict[str, Any]:
    if b_stream is None:
        _a_stream, b_stream = wave.build_streams()
    _validate_accepted_b_stream(b_stream)
    selected = _select_b_training_prefix(b_stream)
    batches = [[_example_record(example) for example in batch] for batch in selected]
    _validate_stream_batches(batches)
    examples = [[dsl.RegisterExample(int(raw["x"]), int(raw["y"]), tuple(raw["program"])) for raw in batch] for batch in batches]
    lengths = [len(batch[0].program) for batch in examples]
    budget = _training_budget(lengths)
    return {
        "schema": STREAM_SCHEMA,
        "batch_size": TRAINING_BATCH_SIZE,
        "batch_count": TRAINING_UPDATES,
        "batch_lengths": lengths,
        "length_cycle": list(_expected_stream_lengths()[:6]),
        "tail": list(_expected_stream_lengths()[-2:]),
        "sum_lengths": sum(lengths),
        "budget": budget,
        "stream_digest": old.batch_digest(examples),
        "target_digest": old.target_digest(examples),
        "batches": batches,
    }


def _validate_baseline_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EVALUATION_PROGRAMS:
        raise ValueError("saved initial-soft program count changed")
    state_order = [list(state) for state in dsl.STATE_ORDER]
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in seen:
            raise ValueError("saved initial-soft program IDs are not unique")
        seen.add(identifier)
        program = tuple(str(op) for op in row.get("program", ()))
        length = int(row.get("length", -1))
        if len(program) != length or str(row.get("suite")) not in {"padding", "compositions"}:
            raise ValueError(f"saved initial-soft program identity changed: {identifier}")
        raw_predictions = row.get("predictions")
        if not isinstance(raw_predictions, list) or len(raw_predictions) != EVALUATION_STATES:
            raise ValueError(f"saved initial-soft state count changed: {identifier}")
        predictions: list[dict[str, Any]] = []
        for raw_prediction, state in zip(raw_predictions, dsl.STATE_ORDER):
            if not isinstance(raw_prediction, Mapping) or raw_prediction.get("state") != list(state):
                raise ValueError(f"saved initial-soft state order changed: {identifier}")
            example = dsl.RegisterExample(state[0], state[1], program)
            target = [list(pair) for pair in example.targets]
            if raw_prediction.get("target_trace") != target or raw_prediction.get("predicted_trace") != target:
                raise ValueError(f"saved initial-soft trace is not exact DSL: {identifier}")
            if raw_prediction.get("full_trace_correct") is not True or raw_prediction.get("joint_final_correct") is not True or raw_prediction.get("first_error") is not None:
                raise ValueError(f"saved initial-soft perfect-trace claim changed: {identifier}")
            predictions.append({
                "state": list(state),
                "stratum": dsl.STATE_STRATUM[state] if hasattr(dsl, "STATE_STRATUM") else next(name for name, values in dsl.state_split().items() if state in values),
                "target_trace": target,
                "predicted_trace": target,
                **qa.accepted_runtime._trace_metrics(target, target),
                "first_subsequent_recovery": None,
            })
        normalized.append({"id": identifier, "suite": str(row["suite"]), "length": length, "program": list(program), "predictions": predictions})
    if len({row["id"] for row in normalized}) != EVALUATION_PROGRAMS:
        raise ValueError("saved initial-soft IDs are not unique")
    if Counter(row["suite"] for row in normalized) != Counter({"padding": 45, "compositions": 24}):
        raise ValueError("saved initial-soft suite inventory changed")
    if Counter(row["length"] for row in normalized) != Counter({4: 9, 12: 15, 16: 15, 24: 15, 32: 15}):
        raise ValueError("saved initial-soft length inventory changed")
    if state_order != [list(state) for state in dsl.STATE_ORDER]:
        raise ValueError("DSL state order changed")
    return normalized


def _baseline_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    programs = []
    for row in rows:
        programs.append({
            "id": str(row["id"]),
            "suite": str(row["suite"]),
            "length": int(row["length"]),
            "program": list(row["program"]),
            "states": [{"state": list(prediction["state"]), "stratum": prediction["stratum"], "target_trace": prediction["target_trace"]} for prediction in row["predictions"]],
        })
    scope = {
        "schema": SCOPE_SCHEMA,
        "program_count": len(programs),
        "state_count": EVALUATION_STATES,
        "states": [list(state) for state in dsl.STATE_ORDER],
        "programs": programs,
    }
    scope["digest"] = _digest(scope)
    return scope


def _validate_scope(scope: Mapping[str, Any]) -> None:
    if scope.get("schema") != SCOPE_SCHEMA or scope.get("program_count") != EVALUATION_PROGRAMS or scope.get("state_count") != EVALUATION_STATES:
        raise ValueError("science evaluation scope identity changed")
    programs = scope.get("programs")
    if not isinstance(programs, list) or len(programs) != EVALUATION_PROGRAMS:
        raise ValueError("science evaluation program scope changed")
    if scope.get("states") != [list(state) for state in dsl.STATE_ORDER]:
        raise ValueError("science evaluation state scope changed")
    for item in programs:
        if not isinstance(item, Mapping) or len(item.get("states", [])) != EVALUATION_STATES:
            raise ValueError("science evaluation state rows changed")
        program = tuple(str(op) for op in item["program"])
        for state_row, state in zip(item["states"], dsl.STATE_ORDER):
            example = dsl.RegisterExample(state[0], state[1], program)
            if state_row.get("state") != list(state) or state_row.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("science evaluation target join changed")
    expected_digest = _digest({key: scope[key] for key in ("schema", "program_count", "state_count", "states", "programs")})
    if scope.get("digest") != expected_digest:
        raise ValueError("science evaluation scope digest changed")


def _baseline_and_scope(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = _resolve(BASELINE, root)
    if not baseline_path.is_file():
        raise ValueError("saved initial-soft baseline is missing")
    raw = _read_object(baseline_path)
    if raw.get("phase") != "initial_soft" or not isinstance(raw.get("rows"), list):
        raise ValueError("saved initial-soft baseline schema changed")
    rows = _validate_baseline_rows(raw["rows"])
    return {"path": _relative(baseline_path, root), "sha256": _sha256(baseline_path), "rows": len(rows), "perfect": True}, _baseline_scope(rows)


def _source_and_evidence(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _source_inventory(root)
    baseline, scope = _baseline_and_scope(root)
    stream = _freeze_stream()
    return source, baseline, {"stream": stream, "scope": scope}


def _validate_parent_checkpoint_bytes(root: Path = ROOT) -> dict[str, str]:
    checkpoint = qa._parent_checkpoint(root)
    if not checkpoint.is_file() or _sha256(checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("accepted parent checkpoint bytes changed")
    return {"path": _relative(checkpoint, root), "sha256": _sha256(checkpoint)}


def prepare_science(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    stream_path = _resolve(SCIENCE_STREAM, root)
    scope_path = _resolve(SCIENCE_SCOPE, root)
    manifest_digest_path = _resolve(SCIENCE_MANIFEST_DIGEST, root)
    output = _resolve(SCIENCE_OUTPUT, root)
    for path in (manifest_path, manifest_digest_path, stream_path, scope_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen science evidence: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty science output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = _validate_parent_checkpoint_bytes(root)
    source, baseline, evidence = _source_and_evidence(root)
    stream = evidence["stream"]
    scope = evidence["scope"]
    _atomic_json(stream_path, stream, refuse=True)
    _atomic_json(scope_path, scope, refuse=True)
    e36_manifest_path = _resolve(E36_PREFLIGHT / "manifest.json", root)
    if not e36_manifest_path.is_file() or _sha256(e36_manifest_path) != E36_MANIFEST_SHA256:
        raise ValueError("accepted E36 manifest hash changed")
    inherited = _read_object(e36_manifest_path)
    manifest: dict[str, Any] = {
        "schema": SCIENCE_SCHEMA,
        "status": "prepared",
        "immutable": True,
        "manifest_digest_sidecar": _relative(manifest_digest_path, root),
        "protocol": {"path": _relative(PROTOCOL, root), "sha256": _sha256(_resolve(PROTOCOL, root))},
        "science_interface": {"path": _relative(SCIENCE_INTERFACE, root), "sha256": _sha256(_resolve(SCIENCE_INTERFACE, root)), "declared_sha256": SCIENCE_INTERFACE_SHA256},
        "source_inventory": source,
        "baseline": {**baseline, "scope_digest": scope["digest"]},
        "stream": {"path": _relative(stream_path, root), "sha256": _sha256(stream_path), "schema": stream["schema"], "stream_digest": stream["stream_digest"], "target_digest": stream["target_digest"], "batch_count": stream["batch_count"], "batch_size": stream["batch_size"], "budget": stream["budget"]},
        "evaluation_scope": {"path": _relative(scope_path, root), "sha256": _sha256(scope_path), "digest": scope["digest"], "program_count": EVALUATION_PROGRAMS, "state_count": EVALUATION_STATES},
        "parent": {
            "label": PARENT_LABEL,
            "arm": PARENT_ARM,
            "branch": PARENT_BRANCH,
            "update": PARENT_UPDATE,
            "checkpoint": parent_checkpoint["path"],
            "checkpoint_sha256": parent_checkpoint["sha256"],
            "manifest": _relative(e36_manifest_path, root),
            "manifest_sha256": E36_MANIFEST_SHA256,
            "manifest_digest": old.canonical_hash(inherited),
        },
        "adapter": {
            "architecture_id": latent.ARCHITECTURE_ID,
            "parameter_count": latent.ADAPTER_PARAMETER_COUNT,
            "initialization": {"seed": latent.LOCAL_INITIALIZATION_SEED, "generator_device": "cpu", "order": list(latent.INITIALIZER_ORDER), "std": latent.LOCAL_INITIALIZATION_STD, "bias": 0.0},
            "optimizer_group": {"association": list(latent.ADAPTER_PARAMETER_NAMES), "moments": "absent_before_first_update", "hyperparameters": "copied_from_original_e36_group"},
        },
        "runtime": _runtime_settings(),
        "training": {"updates": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "batch_lengths": list(latent.FIXED_BATCH_LENGTHS), "sum_lengths": sum(latent.FIXED_BATCH_LENGTHS), "checkpoint_indices": list(CHECKPOINT_INDICES)},
        "evaluation": {"calls_per_phase": EVALUATION_CALLS, "cases": EVALUATION_STATES * EVALUATION_PROGRAMS, "positions": 331776, "native_steps": 2654208, "sequence": ["checkpoint_local0", "initial_latent", "training_2000", "final_latent"]},
        "budget": {"calls": SCIENCE_CALLS, "cases": SCIENCE_CASES, "readout_positions": SCIENCE_POSITIONS, "native_steps": SCIENCE_NATIVE_STEPS, "optimizer_updates": SCIENCE_UPDATES},
    }
    _atomic_json(manifest_path, manifest, refuse=True)
    _atomic_json(manifest_digest_path, {"schema": "pc_latent_slots_science_manifest_digest_v1", "manifest": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, refuse=True)
    return manifest


def _validate_source_against_manifest(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = manifest.get("source_inventory")
    if not isinstance(expected, Mapping):
        raise ValueError("science source inventory is missing")
    current = _source_inventory(root)
    if current != expected:
        raise ValueError("science source/evidence inventory changed")
    if _sha256(_resolve(SCIENCE_INTERFACE, root)) != SCIENCE_INTERFACE_SHA256:
        raise ValueError("science interface bytes changed")
    return current


def _load_frozen_data(manifest: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], list[list[dsl.RegisterExample]]]:
    baseline, scope = _baseline_and_scope(root)
    if manifest.get("baseline", {}).get("sha256") != baseline["sha256"]:
        raise ValueError("saved initial-soft baseline changed")
    scope_path = _resolve(Path(manifest["evaluation_scope"]["path"]), root)
    if not scope_path.is_file() or _sha256(scope_path) != manifest["evaluation_scope"]["sha256"]:
        raise ValueError("science evaluation scope bytes changed")
    frozen_scope = _read_object(scope_path)
    _validate_scope(frozen_scope)
    if frozen_scope.get("digest") != manifest["evaluation_scope"]["digest"] or frozen_scope.get("digest") != scope["digest"]:
        raise ValueError("science evaluation scope lineage changed")
    stream_path = _resolve(Path(manifest["stream"]["path"]), root)
    if not stream_path.is_file() or _sha256(stream_path) != manifest["stream"]["sha256"]:
        raise ValueError("science stream bytes changed")
    frozen_stream = _read_object(stream_path)
    batches, budget = _validate_stream_batches(frozen_stream.get("batches", []))
    if frozen_stream.get("stream_digest") != manifest["stream"]["stream_digest"] or frozen_stream.get("target_digest") != manifest["stream"]["target_digest"] or frozen_stream.get("budget") != budget:
        raise ValueError("science stream lineage or budget changed")
    return frozen_scope, {"baseline": baseline, "scope": scope, "stream": frozen_stream}, batches


def _new_counter() -> dict[str, Any]:
    counter = qa._new_counter()
    counter["phase_counts"] = {
        "load": {"attempted_endpoint_loads": 0, "completed_endpoint_loads": 0, "attempted_underlying_deserializations": 0, "completed_underlying_deserializations": 0, "failures": 0},
        "evaluation_initial": {"attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "failures": 0},
        "training": {"attempted_updates": 0, "completed_updates": 0, "attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "attempted_backwards": 0, "completed_backwards": 0, "attempted_optimizer_steps": 0, "completed_optimizer_steps": 0, "failures": 0},
        "evaluation_final": {"attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "failures": 0},
        "checkpoint_save": {"attempted": 0, "completed": 0, "failures": 0},
    }
    return counter


def _phase_add(counter: dict[str, Any], phase: str, key: str, amount: int = 1) -> None:
    counter.setdefault("phase_counts", {}).setdefault(phase, {})[key] = counter["phase_counts"][phase].get(key, 0) + amount


def _phase_failure(counter: dict[str, Any], phase: str, kind: str, exc: BaseException, sink: Callable[[dict[str, Any]], None] | None) -> None:
    _phase_add(counter, phase, "failures")
    qa._failure(counter, kind, exc, phase=phase)
    qa._flush(counter, sink)


def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None):
    @contextmanager
    def context() -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            timings = counter.setdefault("timings_seconds", {})
            timings[name] = float(timings.get(name, 0.0)) + time.perf_counter() - started
            qa._flush(counter, sink)
    return context()


@contextmanager
def _count_deserializations(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    original = torch.load
    def counted(*args: Any, **kwargs: Any) -> Any:
        counter["attempted_underlying_deserializations"] += 1
        _phase_add(counter, "load", "attempted_underlying_deserializations")
        qa._flush(counter, sink)
        try:
            value = original(*args, **kwargs)
        except BaseException as exc:
            _phase_failure(counter, "load", "underlying_deserialize", exc, sink)
            raise
        counter["completed_underlying_deserializations"] += 1
        _phase_add(counter, "load", "completed_underlying_deserializations")
        qa._flush(counter, sink)
        return value
    torch.load = counted  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch or len(batch) != TRAINING_BATCH_SIZE and len(batch) != EVALUATION_STATES:
        raise ValueError("latent batch size is outside the registered science scope")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("latent batch must be homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _run_update(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], *, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None, forward: Callable[[torch.nn.Module, latent.LatentSlotAdapter, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None) -> float:
    length = len(batch[0].program)
    cases = len(batch)
    update_number = counter["attempted_updates"] + 1
    phase = f"training:update{update_number}"
    counter["attempted_updates"] += 1
    counter["attempted_forwards"] += 1
    counter["attempted_cases"] += cases
    counter["attempted_readout_positions"] += cases * length
    counter["attempted_native_steps"] += cases * length * latent.NATIVE_STEPS
    for key, amount in (("attempted_updates", 1), ("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, "training", key, amount)
    qa._flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with _timed(counter, "training_forward", sink), torch.autocast(device_type=device.type, enabled=False):
            if forward is None:
                logits_x, logits_y = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops)
            else:
                logits_x, logits_y = forward(model, adapter, x_bits, y_bits, ops)
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
            raise ValueError("latent science logits are malformed or nonfinite")
    except BaseException as exc:
        _phase_failure(counter, "training", "forward", exc, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
    for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
        _phase_add(counter, "training", key, amount)
    qa._flush(counter, sink)
    try:
        with _timed(counter, "training_loss", sink):
            loss = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("latent science loss is nonfinite")
    except BaseException as exc:
        _phase_failure(counter, "training", "loss", exc, sink)
        raise
    counter["attempted_backwards"] += 1
    _phase_add(counter, "training", "attempted_backwards")
    qa._flush(counter, sink)
    try:
        with _timed(counter, "training_backward", sink):
            loss.backward()
    except BaseException as exc:
        _phase_failure(counter, "training", "backward", exc, sink)
        raise
    counter["completed_backwards"] += 1
    _phase_add(counter, "training", "completed_backwards")
    qa._flush(counter, sink)
    try:
        with _timed(counter, "training_gradient_clip", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
    except BaseException as exc:
        _phase_failure(counter, "training", "gradient_clip", exc, sink)
        raise
    counter["attempted_optimizer_steps"] += 1
    _phase_add(counter, "training", "attempted_optimizer_steps")
    qa._flush(counter, sink)
    try:
        with _timed(counter, "training_optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        _phase_failure(counter, "training", "optimizer_step", exc, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    _phase_add(counter, "training", "completed_optimizer_steps")
    _phase_add(counter, "training", "completed_updates")
    qa._flush(counter, sink)
    return float(loss.detach().item())


def _cuda_rng() -> list[torch.Tensor]:
    return [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []


def _full_state_identity(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = _cuda_rng()
    return {
        "model_digest": _digest(model.state_dict()),
        "adapter_digest": _digest(adapter.state_dict()),
        "optimizer_digest": _digest(optimizer.state_dict()),
        "cpu_rng_digest": _digest(cpu_rng),
        "cuda_rng_digest": _digest(cuda_rng),
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
    }


def _capture_state(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "model_state_dict": qa._cpu_copy(model.state_dict()),
        "adapter_state_dict": qa._cpu_copy(adapter.state_dict()),
        "optimizer_state_dict": qa._cpu_copy(optimizer.state_dict()),
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": _cuda_rng(),
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
    }


def _restore_state(state: Mapping[str, Any], model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    model.load_state_dict(state["model_state_dict"], strict=True)
    adapter.load_state_dict(state["adapter_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    qa._optimizer_to_device(optimizer, device)
    torch.set_rng_state(state["cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    model.train(bool(state["training_mode"]))
    adapter.train(bool(state["adapter_training_mode"]))


@contextmanager
def _preserve_evaluation_state(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, *, device: torch.device) -> Iterator[None]:
    saved = _capture_state(model, adapter, optimizer)
    model.eval()
    adapter.eval()
    try:
        yield
    finally:
        _restore_state(saved, model, adapter, optimizer, device)
        restored = _full_state_identity(model, adapter, optimizer)
        expected_model = _digest(saved["model_state_dict"])
        expected_adapter = _digest(saved["adapter_state_dict"])
        expected_optimizer = _digest(saved["optimizer_state_dict"])
        expected_cpu = _digest(saved["cpu_rng_state"])
        expected_cuda = _digest(saved["cuda_rng_state"])
        if (restored["model_digest"], restored["adapter_digest"], restored["optimizer_digest"], restored["cpu_rng_digest"], restored["cuda_rng_digest"]) != (expected_model, expected_adapter, expected_optimizer, expected_cpu, expected_cuda):
            raise ValueError("evaluation changed preserved model/adapter/optimizer/RNG state")


def _eval_batch(spec: Mapping[str, Any], device: torch.device) -> tuple[list[dsl.RegisterExample], torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [tuple(int(v) for v in item["state"]) for item in spec["states"]]
    program = tuple(str(op) for op in spec["program"])
    batch = [dsl.RegisterExample(state[0], state[1], program) for state in states]
    x_bits, y_bits, ops, _tx, _ty = _batch_tensors(batch, device)
    return batch, x_bits, y_bits, ops


def _evaluation_rows(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, scope: Mapping[str, Any], *, phase: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> list[dict[str, Any]]:
    if phase not in {"evaluation_initial", "evaluation_final"}:
        raise ValueError("unknown evaluation phase")
    rows: list[dict[str, Any]] = []
    with _preserve_evaluation_state(model, adapter, optimizer, device=device):
        for spec in scope["programs"]:
            batch, x_bits, y_bits, ops = _eval_batch(spec, device)
            cases = len(batch)
            length = len(batch[0].program)
            counter["attempted_forwards"] += 1
            counter["attempted_cases"] += cases
            counter["attempted_readout_positions"] += cases * length
            counter["attempted_native_steps"] += cases * length * latent.NATIVE_STEPS
            for key, amount in (("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", cases * length), ("attempted_native_steps", cases * length * latent.NATIVE_STEPS)):
                _phase_add(counter, phase, key, amount)
            qa._flush(counter, sink)
            try:
                with _timed(counter, f"{phase}_forward", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    (logits_x, logits_y), diagnostics = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
                expected = (cases, length, latent.SLOT_WIDTH)
                if (tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all()):
                    raise ValueError("latent evaluation logits are malformed or nonfinite")
                decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
                decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
                predicted = [[[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)] for index in range(cases)]
                diagnostic_rows = []
                for position, (slot_input, slot_write) in enumerate(zip(diagnostics["slot_inputs"], diagnostics["slot_writes"])):
                    input_norm = torch.linalg.vector_norm(slot_input, dim=-1)
                    write_norm = torch.linalg.vector_norm(slot_write, dim=-1)
                    diagnostic_rows.append({
                        "position": position + 1,
                        "input_l2_mean": float(input_norm.mean().detach().cpu().item()),
                        "input_l2_max": float(input_norm.max().detach().cpu().item()),
                        "write_l2_mean": float(write_norm.mean().detach().cpu().item()),
                        "write_l2_max": float(write_norm.max().detach().cpu().item()),
                        "finite": bool(torch.isfinite(slot_input).all() and torch.isfinite(slot_write).all()),
                    })
            except BaseException as exc:
                _phase_failure(counter, phase, "evaluation_forward", exc, sink)
                raise
            counter["completed_forwards"] += 1
            counter["completed_cases"] += cases
            counter["completed_readout_positions"] += cases * length
            counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
            for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", cases * length), ("completed_native_steps", cases * length * latent.NATIVE_STEPS)):
                _phase_add(counter, phase, key, amount)
            qa._flush(counter, sink)
            predictions = []
            for state, example, trace in zip(dsl.STATE_ORDER, batch, predicted):
                target = [list(pair) for pair in example.targets]
                metrics = accepted_runtime._trace_metrics(target, trace)
                first_recovery = None
                if metrics["first_error"] is not None:
                    for position in range(int(metrics["first_error"]), len(target)):
                        if tuple(target[position]) == tuple(trace[position]):
                            first_recovery = position + 1
                            break
                predictions.append({"state": list(state), "stratum": next(name for name, values in dsl.state_split().items() if state in values), "target_trace": target, "predicted_trace": trace, **metrics, "first_subsequent_recovery": first_recovery})
            rows.append({"id": str(spec["id"]), "suite": str(spec["suite"]), "length": length, "program": list(spec["program"]), "predictions": predictions, "writing_diagnostics": {"positions": diagnostic_rows, "native_steps": latent.NATIVE_STEPS, "semantic_slot_accuracy": None}})
    return rows


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def empty() -> dict[str, Any]:
        return {"cases": 0, "final_correct": 0, "full_trace_correct": 0, "first_error": {}, "first_subsequent_recovery": {}, "recovered_final": 0}

    def add(group: dict[str, Any], prediction: Mapping[str, Any]) -> None:
        group["cases"] += 1
        group["final_correct"] += int(bool(prediction["joint_final_correct"]))
        group["full_trace_correct"] += int(bool(prediction["full_trace_correct"]))
        first = "none" if prediction["first_error"] is None else str(prediction["first_error"])
        group["first_error"][first] = group["first_error"].get(first, 0) + 1
        recovery_value = prediction.get("first_subsequent_recovery")
        recovery = "none" if recovery_value is None else str(recovery_value)
        group["first_subsequent_recovery"][recovery] = group["first_subsequent_recovery"].get(recovery, 0) + 1
        group["recovered_final"] += int(bool(prediction.get("recovered_final", False)))

    groups: dict[str, dict[str, Any]] = {}
    per_program: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row["id"])
        program_groups = per_program.setdefault(identifier, {"all": empty(), "heldout64": empty()})
        for prediction in row["predictions"]:
            suite = str(row["suite"])
            length = int(row["length"])
            stratum = str(prediction["stratum"])
            joint_key = f"suite:{suite}|length:{length}|stratum:{stratum}"
            heldout = stratum in {"validation", "test"}
            keys = ["all", f"suite:{suite}", f"length:{length}", f"stratum:{stratum}", joint_key]
            if heldout:
                keys.extend(["heldout64", f"heldout64|{joint_key}"])
            for key in keys:
                group = groups.setdefault(key, empty())
                add(group, prediction)
            add(program_groups["all"], prediction)
            if heldout:
                add(program_groups["heldout64"], prediction)
    joint = {key: value for key, value in groups.items() if "|length:" in key and not key.startswith("heldout64|")}
    heldout_joint = {key.removeprefix("heldout64|"): value for key, value in groups.items() if key.startswith("heldout64|")}
    return {
        "groups": groups,
        "joint_suite_length_stratum": joint,
        "heldout64_within_joint": heldout_joint,
        "per_program": per_program,
        "denominators": {key: value["cases"] for key, value in groups.items()},
    }


def _checkpoint_metadata(local_update: int, *, manifest_digest: str, source_binding_digest: str, parent_identity: Mapping[str, Any], adapter_identity: Mapping[str, Any], next_batch_index: int) -> dict[str, Any]:
    if local_update not in CHECKPOINT_INDICES or next_batch_index != local_update:
        raise ValueError("checkpoint boundary or next stream index changed")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "committed": True,
        "complete": True,
        "manifest_digest": manifest_digest,
        "source_binding_digest": source_binding_digest,
        "parent_identity": dict(parent_identity),
        "architecture_id": latent.ARCHITECTURE_ID,
        "adapter_identity": dict(adapter_identity),
        "local_update": local_update,
        "absolute_update": PARENT_UPDATE + local_update,
        "next_batch_index": next_batch_index,
    }


def _checkpoint_payload(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, *, local_update: int, manifest_digest: str, source_binding_digest: str, parent_identity: Mapping[str, Any], adapter_identity: Mapping[str, Any], adapter_initialization: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _checkpoint_metadata(local_update, manifest_digest=manifest_digest, source_binding_digest=source_binding_digest, parent_identity=parent_identity, adapter_identity=adapter_identity, next_batch_index=local_update)
    model_state = qa._cpu_copy(model.state_dict())
    adapter_state = qa._cpu_copy(adapter.state_dict())
    optimizer_state = qa._cpu_copy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = _cuda_rng()
    payload: dict[str, Any] = {
        **metadata,
        "adapter_initialization": dict(adapter_initialization),
        "model_state_dict": model_state,
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_metadata": qa._optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()),
        "adapter_parameter_names": [name for name, _ in adapter.named_parameters()],
        "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
        "model_digest": _digest(model_state),
        "adapter_digest": _digest(adapter_state),
        "optimizer_digest": _digest(optimizer_state),
        "cpu_rng_digest": _digest(cpu_rng),
        "cuda_rng_digest": _digest(cuda_rng),
    }
    payload["checkpoint_digest"] = _digest(payload)
    return payload


def _write_checkpoint(out: Path, payload: Mapping[str, Any], *, local_update: int, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    if int(payload["local_update"]) != local_update:
        raise ValueError("checkpoint payload boundary changed")
    path = out / "checkpoints" / f"local{local_update:04d}.pt"
    _phase_add(counter, "checkpoint_save", "attempted")
    qa._flush(counter, sink)
    try:
        with _timed(counter, "checkpoint_save", sink):
            _atomic_torch(path, payload, refuse=True)
    except BaseException as exc:
        _phase_failure(counter, "checkpoint_save", "checkpoint_save", exc, sink)
        raise
    _phase_add(counter, "checkpoint_save", "completed")
    qa._flush(counter, sink)
    return {"local_update": local_update, "absolute_update": PARENT_UPDATE + local_update, "next_batch_index": local_update, "path": _relative(path), "sha256": _sha256(path), "checkpoint_digest": payload["checkpoint_digest"], "committed": True, "complete": True}


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing non-empty science output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_manifest_for_science(root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST) -> dict[str, Any]:
    manifest_path = _resolve(manifest_path, root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare-science manifest is missing: {manifest_path}")
    manifest = _read_object(manifest_path)
    if manifest.get("schema") != SCIENCE_SCHEMA or manifest.get("status") != "prepared" or manifest.get("immutable") is not True:
        raise ValueError("science manifest is not an immutable prepared manifest")
    digest_sidecar = _resolve(Path(str(manifest.get("manifest_digest_sidecar", SCIENCE_MANIFEST_DIGEST))), root)
    if not digest_sidecar.is_file():
        raise ValueError("science manifest digest sidecar is missing")
    digest_record = _read_object(digest_sidecar)
    if digest_record.get("schema") != "pc_latent_slots_science_manifest_digest_v1" or digest_record.get("manifest") != _relative(manifest_path, root) or digest_record.get("sha256") != _sha256(manifest_path):
        raise ValueError("science manifest bytes changed")
    _validate_source_against_manifest(manifest, root)
    frozen_scope, evidence, _batches = _load_frozen_data(manifest, root)
    baseline = evidence["baseline"]
    if manifest.get("evaluation_scope", {}).get("digest") != frozen_scope.get("digest") or manifest.get("baseline", {}).get("scope_digest") != frozen_scope.get("digest"):
        raise ValueError("science baseline/evaluation scope join changed")
    parent_binding = _validate_parent_checkpoint_bytes(root)
    if manifest.get("parent", {}).get("checkpoint_sha256") != parent_binding["sha256"] or manifest.get("parent", {}).get("checkpoint") != parent_binding["path"]:
        raise ValueError("science parent checkpoint binding changed")
    parent = _resolve(E36_RUN / PARENT_LABEL / PARENT_BRANCH / "u40000.pt", root)
    e36_manifest = _resolve(E36_PREFLIGHT / "manifest.json", root)
    if not parent.is_file() or _sha256(parent) != PARENT_CHECKPOINT_SHA256 or not e36_manifest.is_file() or _sha256(e36_manifest) != E36_MANIFEST_SHA256:
        raise ValueError("science parent or E36 manifest bytes changed")
    if manifest.get("runtime", {}).get("native_steps") != latent.NATIVE_STEPS or manifest.get("training", {}).get("checkpoint_indices") != list(CHECKPOINT_INDICES):
        raise ValueError("science fixed runtime/checkpoint scope changed")
    return manifest


def _expected_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 1, "completed_endpoint_loads": 1,
        "attempted_underlying_deserializations": 2, "completed_underlying_deserializations": 2,
        "attempted_updates": TRAINING_UPDATES, "completed_updates": TRAINING_UPDATES,
        "attempted_forwards": SCIENCE_CALLS, "completed_forwards": SCIENCE_CALLS,
        "attempted_cases": SCIENCE_CASES, "completed_cases": SCIENCE_CASES,
        "attempted_readout_positions": SCIENCE_POSITIONS, "completed_readout_positions": SCIENCE_POSITIONS,
        "attempted_native_steps": SCIENCE_NATIVE_STEPS, "completed_native_steps": SCIENCE_NATIVE_STEPS,
        "attempted_backwards": TRAINING_UPDATES, "completed_backwards": TRAINING_UPDATES,
        "attempted_optimizer_steps": TRAINING_UPDATES, "completed_optimizer_steps": TRAINING_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"science accounting mismatch: {key}={counter.get(key)} expected {value}")


def run_science(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = SCIENCE_OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest = _load_manifest_for_science(root, manifest_path)
    out = _resolve(out, root)
    _refuse_nonempty(out)
    device, settings = _configure_runtime()
    if settings != manifest.get("runtime"):
        raise ValueError("science runtime settings drifted from immutable manifest")
    scope, evidence, batches = _load_frozen_data(manifest, root)
    counter = _new_counter()
    sink = lambda value: _atomic_json(out / "accounting.json", value)
    sink(counter)
    report: dict[str, Any] = {"schema": RUN_SCHEMA, "status": "running", "manifest": {"path": _relative(_resolve(manifest_path, root), root), "sha256": _sha256(_resolve(manifest_path, root))}, "runtime": settings, "source_inventory_digest": manifest["source_inventory"]["digest"], "models": {}, "checkpoints": [], "evaluations": {}, "metrics": {}, "accounting": dict(counter)}
    ledger: list[dict[str, Any]] = []
    _atomic_json(out / "update_ledger.json", ledger, refuse=True)
    try:
        with migration.windows_compatibility_adapter():
            inherited = followup._load_inherited(root)
            if old.canonical_hash(inherited) != manifest["parent"]["manifest_digest"]:
                raise ValueError("inherited E36 manifest digest changed")
            common_cuda_rng = _cuda_rng()
            with _count_deserializations(counter, sink):
                _phase_add(counter, "load", "attempted_endpoint_loads")
                qa._flush(counter, sink)
                try:
                    with _timed(counter, "load", sink):
                        model, optimizer, parent = qa._load_parent(manifest=inherited, counter=counter, sink=sink, root=root)
                except BaseException as exc:
                    _phase_failure(counter, "load", "endpoint_load", exc, sink)
                    raise
                _phase_add(counter, "load", "completed_endpoint_loads")
                model.to(device)
                qa._optimizer_to_device(optimizer, device)
                torch.set_rng_state(parent["rng_state"])
                torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
                model.train(True)
                adapter, initialization, adapter_initial_identity, parent_names = qa._append_adapter_group(model, optimizer, device=device)
                adapter.train(True)
                parent_identity = qa._parent_identity(parent, qa._parent_checkpoint(root), manifest["parent"]["manifest_digest"], common_cuda_rng)
                source_digest = str(manifest["source_inventory"]["digest"])
                local0_payload = _checkpoint_payload(model, adapter, optimizer, local_update=0, manifest_digest=_sha256(_resolve(manifest_path, root)), source_binding_digest=source_digest, parent_identity=parent_identity, adapter_identity=adapter_initial_identity, adapter_initialization=initialization)
                report["checkpoints"].append(_write_checkpoint(out, local0_payload, local_update=0, counter=counter, sink=sink))
                with _timed(counter, "evaluation_initial", sink):
                    initial_rows = _evaluation_rows(model, adapter, optimizer, scope, phase="evaluation_initial", device=device, counter=counter, sink=sink)
                initial_path = out / "evaluations" / "initial_latent.json"
                _atomic_json(initial_path, {"schema": "pc_latent_slots_evaluation_v1", "phase": "initial_latent", "rows": initial_rows}, refuse=True)
                report["evaluations"]["initial_latent"] = {"path": _relative(initial_path, root), "sha256": _sha256(initial_path), "rows": len(initial_rows)}
                with _timed(counter, "training", sink):
                    for local_index, batch in enumerate(batches, start=1):
                        loss = _run_update(model, adapter, optimizer, batch, device=device, counter=counter, sink=sink)
                        ledger.append({"local_update": local_index, "absolute_update": PARENT_UPDATE + local_index, "next_batch_index": local_index, "length": len(batch[0].program), "cases": len(batch), "loss": loss})
                        _atomic_json(out / "update_ledger.json", ledger)
                        if local_index in CHECKPOINT_INDICES[1:]:
                            payload = _checkpoint_payload(model, adapter, optimizer, local_update=local_index, manifest_digest=_sha256(_resolve(manifest_path, root)), source_binding_digest=source_digest, parent_identity=parent_identity, adapter_identity=adapter_initial_identity, adapter_initialization=initialization)
                            report["checkpoints"].append(_write_checkpoint(out, payload, local_update=local_index, counter=counter, sink=sink))
                with _timed(counter, "evaluation_final", sink):
                    final_rows = _evaluation_rows(model, adapter, optimizer, scope, phase="evaluation_final", device=device, counter=counter, sink=sink)
                final_path = out / "evaluations" / "final_latent.json"
                _atomic_json(final_path, {"schema": "pc_latent_slots_evaluation_v1", "phase": "final_latent", "rows": final_rows}, refuse=True)
                report["evaluations"]["final_latent"] = {"path": _relative(final_path, root), "sha256": _sha256(final_path), "rows": len(final_rows)}
                baseline_rows = _validate_baseline_rows(_read_object(_resolve(BASELINE, root))["rows"])
                report["metrics"] = {
                    "final_latent_vs_initial_latent": accepted_runtime._paired_metrics(final_rows, initial_rows, label="final_latent_vs_initial_latent"),
                    "initial_latent_vs_saved_initial_soft": accepted_runtime._paired_metrics(initial_rows, baseline_rows, label="initial_latent_vs_saved_initial_soft"),
                    "final_latent_vs_saved_initial_soft": accepted_runtime._paired_metrics(final_rows, baseline_rows, label="final_latent_vs_saved_initial_soft"),
                }
                report["aggregates"] = {"initial_latent": _aggregate_rows(initial_rows), "final_latent": _aggregate_rows(final_rows), "saved_initial_soft": _aggregate_rows(baseline_rows)}
        _expected_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "budget": {"calls": SCIENCE_CALLS, "cases": SCIENCE_CASES, "readout_positions": SCIENCE_POSITIONS, "native_steps": SCIENCE_NATIVE_STEPS, "optimizer_updates": SCIENCE_UPDATES}, "initial_cuda_rng_digest": _digest(common_cuda_rng), "parent_identity": parent_identity, "adapter_initialization": initialization, "adapter_identity": adapter_initial_identity, "parameter_names": {"parent": parent_names, "adapter": list(latent.ADAPTER_PARAMETER_NAMES)}, "evaluation_state_preservation": {"initial_latent": True, "final_latent": True}, "limitations": ["single float seed0 B child", "saved initial-soft is a descriptive perfect reference and was not retrained", "program/state pool was opened before this pilot"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        qa._failure(counter, "science", exc)
        qa._flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-science", action="store_true")
    mode.add_argument("--science", action="store_true")
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=SCIENCE_OUTPUT)
    args = parser.parse_args(argv)
    if args.prepare_science:
        prepare_science(manifest_path=args.manifest)
    else:
        run_science(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
