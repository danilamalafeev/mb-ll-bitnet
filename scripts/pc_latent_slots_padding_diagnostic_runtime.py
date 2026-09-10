"""Bounded observational padding-drift diagnostic.

The entry point consumes the already accepted latent science manifest and its
saved local-2000 endpoint.  It performs one fresh, read-only 69-program
evaluation; only the frozen 18 padding programs retain per-state, per-position
slot and writer summaries.  It does not train, resume, or run a control.
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
from scripts import followup_e37_e38 as followup
from scripts import pc_inference_migration as migration
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_runtime as qa
from scripts import pc_latent_slots_science as science
from scripts import pc_latent_slots_padding_diagnostic as diagnostic
from scripts import pc_learned_scratchpad_runtime as accepted_runtime


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "pc_latent_slots_v1"
SCIENCE_MANIFEST = RUN_ROOT / "science_manifest.json"
SCIENCE_MANIFEST_DIGEST = RUN_ROOT / "science_manifest.sha256"
SCIENCE_REPORT = RUN_ROOT / "science" / "report.json"
SCIENCE_SCOPE = RUN_ROOT / "science_eval_scope.json"
SCIENCE_FINAL = RUN_ROOT / "science" / "evaluations" / "final_latent.json"
SCIENCE_ENDPOINT = RUN_ROOT / "science" / "checkpoints" / "local2000.pt"
SCIENCE_SOURCE_INVENTORY = RUN_ROOT / "science_source_inventory_v1.json"
QA_ACCEPT = RUN_ROOT / "qa_accept.json"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LATENT_SLOTS_PADDING_DIAGNOSTIC_PROTOCOL.md"
HELPER = ROOT / "scripts" / "pc_latent_slots_padding_diagnostic.py"
TESTS = ROOT / "tests" / "test_pc_latent_slots_padding_diagnostic.py"
RUNTIME_TESTS = ROOT / "tests" / "test_pc_latent_slots_padding_diagnostic_runtime.py"
RUNTIME = Path(__file__).resolve()
OUTPUT = RUN_ROOT / "padding_diagnostic_v1"

SCHEMA = "pc_latent_slots_padding_diagnostic_v1"
ACCOUNTING_SCHEMA = "pc_latent_slots_padding_diagnostic_accounting_v1"
INPUT_SCHEMA = "pc_latent_slots_padding_diagnostic_input_freeze_v1"
PROGRAM_SCHEMA = "pc_latent_slots_padding_diagnostic_program_v1"
EVALUATION_PROGRAMS = 69
EVALUATION_STATES = 256
EVALUATION_CALLS = 69
EVALUATION_CASES = 17_664
EVALUATION_POSITIONS = 331_776
EVALUATION_NATIVE_STEPS = 2_654_208
FOCUS_CALLS = 18
FOCUS_CASES = 4_608
FOCUS_POSITIONS = 129_024
FOCUS_NATIVE_STEPS = 1_032_192
ENDPOINT_LOCAL_UPDATE = 2_000
ENDPOINT_ABSOLUTE_UPDATE = 42_000
ENDPOINT_SHA256 = "bb7d4ea2739d97ccd9b43663657460c5f775bc8514543b71e8a14e88e03417d5"
SCIENCE_MANIFEST_SHA256 = "ef64fdd4ed8af03d7631b9809187ab1902f864325925f883277102dff45fbb0a"
SCIENCE_MANIFEST_DIGEST_SHA256 = "ee9475040f18ddb75a1d621a5a09b5534df7ac2b975b46399ea9e190411ac963"
SCIENCE_REPORT_SHA256 = "64f449882b10707ca3a81e1c170588bf61182385f63f5407ef71e6648164fbc0"
SCIENCE_FINAL_SHA256 = "b7444110f76dc7871280c134a752be9cbdea63fb2580dd3cf967e2247776ef55"
SCIENCE_SCOPE_SHA256 = "5cb130e2e0027eb9faf773283d5676cc067560e4cf2e7294f57b0d80afa1863e"
REGISTER_E15_SHA256 = "bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac"
DIAGNOSTIC_HELPER_SHA256 = "5cc238be977612db6db99a56426162d76d73c02964cb6ec527baff993118c6eb"
DIAGNOSTIC_TEST_SHA256 = "7820acdc7db07e65ca91744a160efa0b0503b18f3d6f6c7370d1bf0218154592"
E15_STRATUM_ALGORITHM = diagnostic.E15_STATE_SPLIT_ALGORITHM
FOCUS_IDS = tuple(identifier for identifier, _program, _roles in diagnostic.FOCUS_PROGRAM_MAP)


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


def _read_json(path: Path) -> dict[str, Any]:
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


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty diagnostic output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _new_counter() -> dict[str, Any]:
    return {
        "schema": ACCOUNTING_SCHEMA,
        "attempted_endpoint_loads": 0,
        "completed_endpoint_loads": 0,
        "attempted_parent_loads": 0,
        "completed_parent_loads": 0,
        "attempted_endpoint_restores": 0,
        "completed_endpoint_restores": 0,
        "attempted_underlying_deserializations": 0,
        "completed_underlying_deserializations": 0,
        "attempted_forwards": 0,
        "completed_forwards": 0,
        "attempted_cases": 0,
        "completed_cases": 0,
        "attempted_readout_positions": 0,
        "completed_readout_positions": 0,
        "attempted_native_steps": 0,
        "completed_native_steps": 0,
        "attempted_updates": 0,
        "completed_updates": 0,
        "attempted_backwards": 0,
        "completed_backwards": 0,
        "attempted_optimizer_steps": 0,
        "completed_optimizer_steps": 0,
        "failures": [],
        "timings_seconds": {},
        "phase_counts": {
            "load": {"attempted_endpoint_loads": 0, "completed_endpoint_loads": 0, "attempted_parent_loads": 0, "completed_parent_loads": 0, "attempted_endpoint_restores": 0, "completed_endpoint_restores": 0, "attempted_underlying_deserializations": 0, "completed_underlying_deserializations": 0, "failures": 0},
            "evaluation": {"attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "failures": 0},
        },
    }


def _sink(path: Path) -> Callable[[dict[str, Any]], None]:
    return lambda counter: _atomic_json(path, counter)


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> None:
    sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    counter["failures"].append({"kind": kind, "phase": phase, "type": type(exc).__name__, "message": str(exc)})
    counter["phase_counts"][phase]["failures"] += 1


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        counter["timings_seconds"][name] = float(counter["timings_seconds"].get(name, 0.0)) + time.perf_counter() - started
        _flush(counter, sink)


@contextmanager
def _count_deserializations(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    original = torch.load

    def counted(*args: Any, **kwargs: Any) -> Any:
        counter["attempted_underlying_deserializations"] += 1
        counter["phase_counts"]["load"]["attempted_underlying_deserializations"] += 1
        _flush(counter, sink)
        try:
            value = original(*args, **kwargs)
        except BaseException as exc:
            _failure(counter, "underlying_deserialize", exc, phase="load")
            _flush(counter, sink)
            raise
        counter["completed_underlying_deserializations"] += 1
        counter["phase_counts"]["load"]["completed_underlying_deserializations"] += 1
        _flush(counter, sink)
        return value

    torch.load = counted  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _begin_load(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> None:
    """Persist every load attempt before entering the accepted loader."""

    for key in ("attempted_endpoint_loads", "attempted_parent_loads", "attempted_endpoint_restores"):
        counter[key] += 1
        counter["phase_counts"]["load"][key] += 1
    _flush(counter, sink)


def _mark_parent_loaded(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> None:
    counter["completed_parent_loads"] += 1
    counter["phase_counts"]["load"]["completed_parent_loads"] += 1
    _flush(counter, sink)


def _mark_endpoint_loaded(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> None:
    counter["completed_endpoint_restores"] += 1
    counter["completed_endpoint_loads"] += 1
    counter["phase_counts"]["load"]["completed_endpoint_restores"] += 1
    counter["phase_counts"]["load"]["completed_endpoint_loads"] += 1
    _flush(counter, sink)


def _canonical_manifest_hash(value: Mapping[str, Any]) -> str:
    """Use the canonical JSON digest used by accepted science manifests."""

    return science.old.canonical_hash(value)


@contextmanager
def _accepted_load_context(*, manifest: Mapping[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    """Keep inherited manifest and parent loading inside the Windows adapter."""

    with migration.windows_compatibility_adapter():
        inherited = followup._load_inherited(root)
        if _canonical_manifest_hash(inherited) != manifest["parent"]["manifest_digest"]:
            raise ValueError("inherited E36 manifest canonical digest changed")
        yield inherited


def _file_binding(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _validate_strata(scope: Mapping[str, Any]) -> None:
    if scope.get("states") != [list(state) for state in dsl.STATE_ORDER]:
        raise ValueError("diagnostic state order changed")
    programs = scope.get("programs")
    if not isinstance(programs, list) or len(programs) != EVALUATION_PROGRAMS or len({str(spec.get("id", "")) for spec in programs if isinstance(spec, Mapping)}) != EVALUATION_PROGRAMS:
        raise ValueError("diagnostic program identity/count changed")
    for spec in programs:
        states = spec.get("states")
        if not isinstance(states, list) or len(states) != EVALUATION_STATES:
            raise ValueError("diagnostic scope state count changed")
        for state_row, state in zip(states, dsl.STATE_ORDER):
            raw_state = tuple(int(value) for value in state_row.get("state", ()))
            if raw_state != state or state_row.get("stratum") != diagnostic._state_stratum(state):
                raise ValueError("diagnostic E15 state/stratum join changed")


def _focus_scope(scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(spec.get("id")): spec for spec in scope.get("programs", [])}
    if len(by_id) != EVALUATION_PROGRAMS:
        raise ValueError("diagnostic scope IDs changed")
    rows: list[dict[str, Any]] = []
    for identifier, program, roles in diagnostic.FOCUS_PROGRAM_MAP:
        spec = by_id.get(identifier)
        if spec is None or tuple(spec.get("program", ())) != program:
            raise ValueError(f"diagnostic focus program changed: {identifier}")
        rows.append({
            "id": identifier,
            "suite": "padding",
            "length": len(program),
            "program": list(program),
            "position_map": [{"position": index, "opcode": opcode, "role": role} for index, (opcode, role) in enumerate(zip(program, roles), start=1)],
            "states": [{"state": list(state_row["state"]), "stratum": state_row["stratum"]} for state_row in spec["states"]],
        })
    diagnostic.validate_focus_scope(rows)
    return rows


def _validate_saved_final(scope: Mapping[str, Any], final: Mapping[str, Any]) -> None:
    rows = final.get("rows")
    specs = scope.get("programs")
    if final.get("phase") != "final_latent" or not isinstance(rows, list) or not isinstance(specs, list) or len(rows) != EVALUATION_PROGRAMS:
        raise ValueError("saved final evaluation scope changed")
    for spec, row in zip(specs, rows):
        if not isinstance(row, Mapping) or row.get("id") != spec.get("id") or row.get("suite") != spec.get("suite") or row.get("length") != spec.get("length") or row.get("program") != spec.get("program"):
            raise ValueError("saved final evaluation program join changed")
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != EVALUATION_STATES:
            raise ValueError("saved final evaluation state count changed")
        for expected, prediction in zip(spec["states"], predictions):
            if not isinstance(prediction, Mapping) or prediction.get("state") != expected.get("state") or prediction.get("stratum") != expected.get("stratum") or prediction.get("target_trace") != expected.get("target_trace"):
                raise ValueError("saved final evaluation target/state join changed")


def _validate_evidence(root: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = _resolve(manifest_path, root)
    manifest = science._load_manifest_for_science(root, manifest_path)
    if _sha256(manifest_path) != SCIENCE_MANIFEST_SHA256:
        raise ValueError("science manifest bytes changed")
    manifest_digest_path = _resolve(SCIENCE_MANIFEST_DIGEST, root)
    if _sha256(manifest_digest_path) != SCIENCE_MANIFEST_DIGEST_SHA256:
        raise ValueError("science manifest digest bytes changed")
    report_path = _resolve(SCIENCE_REPORT, root)
    final_path = _resolve(SCIENCE_FINAL, root)
    endpoint_path = _resolve(SCIENCE_ENDPOINT, root)
    scope_path = _resolve(SCIENCE_SCOPE, root)
    for path, expected in ((report_path, SCIENCE_REPORT_SHA256), (final_path, SCIENCE_FINAL_SHA256), (endpoint_path, ENDPOINT_SHA256), (scope_path, SCIENCE_SCOPE_SHA256)):
        if _sha256(path) != expected:
            raise ValueError(f"saved evidence bytes changed: {path}")
    report = _read_json(report_path)
    if report.get("status") != "complete" or report.get("manifest", {}).get("sha256") != SCIENCE_MANIFEST_SHA256:
        raise ValueError("saved science report lineage changed")
    checkpoints = report.get("checkpoints")
    endpoint_record = next((item for item in checkpoints if item.get("local_update") == ENDPOINT_LOCAL_UPDATE), None) if isinstance(checkpoints, list) else None
    if not isinstance(endpoint_record, Mapping) or endpoint_record.get("path") != _relative(endpoint_path, root) or endpoint_record.get("sha256") != ENDPOINT_SHA256 or endpoint_record.get("absolute_update") != ENDPOINT_ABSOLUTE_UPDATE or endpoint_record.get("next_batch_index") != ENDPOINT_LOCAL_UPDATE or endpoint_record.get("committed") is not True or endpoint_record.get("complete") is not True:
        raise ValueError("saved local-2000 endpoint metadata changed")
    final_record = report.get("evaluations", {}).get("final_latent", {})
    if final_record.get("path") != _relative(final_path, root) or final_record.get("sha256") != SCIENCE_FINAL_SHA256 or final_record.get("rows") != EVALUATION_PROGRAMS:
        raise ValueError("saved final evaluation metadata changed")
    scope = _read_json(scope_path)
    science._validate_scope(scope)
    _validate_strata(scope)
    _focus_scope(scope)
    final = _read_json(final_path)
    _validate_saved_final(scope, final)
    return manifest, report, scope, final, {"endpoint": _file_binding(endpoint_path, root), "report": _file_binding(report_path, root), "final": _file_binding(final_path, root), "scope": _file_binding(scope_path, root), "manifest": _file_binding(manifest_path, root), "manifest_digest": _file_binding(manifest_digest_path, root)}


def _source_binding(root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    register = _resolve(Path("looped_bitnet/register_e15.py"), root)
    files = {
        "runtime": _file_binding(RUNTIME, root),
        "diagnostic_helper": _file_binding(_resolve(HELPER, root), root),
        "diagnostic_tests": _file_binding(_resolve(TESTS, root), root),
        "runtime_tests": _file_binding(_resolve(RUNTIME_TESTS, root), root),
        "register_e15": _file_binding(register, root),
        "protocol": _file_binding(_resolve(PROTOCOL, root), root),
        "science_source_inventory": _file_binding(_resolve(SCIENCE_SOURCE_INVENTORY, root), root),
        **{name: dict(value) for name, value in evidence.items()},
    }
    if files["register_e15"]["sha256"] != REGISTER_E15_SHA256 or files["diagnostic_helper"]["sha256"] != DIAGNOSTIC_HELPER_SHA256 or files["diagnostic_tests"]["sha256"] != DIAGNOSTIC_TEST_SHA256:
        raise ValueError("diagnostic pure source bytes changed")
    accepted_inventory = manifest.get("source_inventory")
    if not isinstance(accepted_inventory, Mapping):
        raise ValueError("accepted science source inventory is missing")
    result: dict[str, Any] = {
        "schema": "pc_latent_slots_padding_diagnostic_source_binding_v1",
        "accepted_science_source_inventory": accepted_inventory,
        "files": files,
        "runtime_settings": dict(settings),
        "scope_digest": manifest.get("evaluation_scope", {}).get("digest"),
        "endpoint_local_update": ENDPOINT_LOCAL_UPDATE,
        "e15_state_split": {"algorithm": E15_STRATUM_ALGORITHM, "source": diagnostic.E15_STATE_SPLIT_SOURCE, "source_sha256": diagnostic.E15_STATE_SPLIT_SOURCE_SHA256},
    }
    result["digest"] = science._digest(result)
    return result


def _account_attempt(counter: dict[str, Any], length: int, sink: Callable[[dict[str, Any]], None]) -> None:
    cases = EVALUATION_STATES
    positions = cases * length
    native = positions * latent.NATIVE_STEPS
    for key, amount in (("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", positions), ("attempted_native_steps", native)):
        counter[key] += amount
        counter["phase_counts"]["evaluation"][key] += amount
    _flush(counter, sink)


def _account_complete(counter: dict[str, Any], length: int, sink: Callable[[dict[str, Any]], None]) -> None:
    cases = EVALUATION_STATES
    positions = cases * length
    native = positions * latent.NATIVE_STEPS
    for key, amount in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", positions), ("completed_native_steps", native)):
        counter[key] += amount
        counter["phase_counts"]["evaluation"][key] += amount
    _flush(counter, sink)


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 1,
        "completed_endpoint_loads": 1,
        "attempted_parent_loads": 1,
        "completed_parent_loads": 1,
        "attempted_endpoint_restores": 1,
        "completed_endpoint_restores": 1,
        "attempted_underlying_deserializations": 3,
        "completed_underlying_deserializations": 3,
        "attempted_forwards": EVALUATION_CALLS,
        "completed_forwards": EVALUATION_CALLS,
        "attempted_cases": EVALUATION_CASES,
        "completed_cases": EVALUATION_CASES,
        "attempted_readout_positions": EVALUATION_POSITIONS,
        "completed_readout_positions": EVALUATION_POSITIONS,
        "attempted_native_steps": EVALUATION_NATIVE_STEPS,
        "completed_native_steps": EVALUATION_NATIVE_STEPS,
        "attempted_updates": 0,
        "completed_updates": 0,
        "attempted_backwards": 0,
        "completed_backwards": 0,
        "attempted_optimizer_steps": 0,
        "completed_optimizer_steps": 0,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"diagnostic accounting mismatch: {key}={counter.get(key)} expected {value}")


def _validate_endpoint_payload(payload: Mapping[str, Any], *, manifest: Mapping[str, Any], report: Mapping[str, Any], manifest_digest: str, source_digest: str) -> None:
    required = ("schema", "committed", "complete", "manifest_digest", "source_binding_digest", "parent_identity", "architecture_id", "local_update", "absolute_update", "next_batch_index", "adapter_identity", "adapter_initialization", "model_state_dict", "adapter_state_dict", "optimizer_state_dict", "optimizer_metadata", "cpu_rng_state", "cuda_rng_state", "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names", "optimizer_group_param_names", "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest", "checkpoint_digest")
    if any(key not in payload for key in required):
        raise ValueError("saved local-2000 endpoint schema changed")
    expected = {"schema": "pc_latent_slots_science_checkpoint_v1", "committed": True, "complete": True, "manifest_digest": manifest_digest, "source_binding_digest": source_digest, "architecture_id": latent.ARCHITECTURE_ID, "local_update": ENDPOINT_LOCAL_UPDATE, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE, "next_batch_index": ENDPOINT_LOCAL_UPDATE}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"saved endpoint metadata changed: {key}")
    if payload.get("parent_identity") != report.get("parent_identity"):
        raise ValueError("saved endpoint parent lineage changed")
    digest_payload = dict(payload)
    checkpoint_digest = digest_payload.pop("checkpoint_digest")
    if science._digest(digest_payload) != checkpoint_digest:
        raise ValueError("saved endpoint checkpoint digest changed")


def _load_endpoint(*, manifest: Mapping[str, Any], report: Mapping[str, Any], endpoint: Path, manifest_path: Path, source_binding_digest: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], root: Path) -> tuple[torch.nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer, dict[str, Any]]:
    payload: Mapping[str, Any] | None = None
    _begin_load(counter, sink)
    try:
        with _accepted_load_context(manifest=manifest, root=root) as inherited, _count_deserializations(counter, sink), _timed(counter, "endpoint_load", sink):
            parent_counter = _new_counter()
            model, optimizer, _parent = qa._load_parent(manifest=inherited, counter=parent_counter, sink=None, root=root)
            _mark_parent_loaded(counter, sink)
            payload = torch.load(endpoint, map_location="cpu")
            _validate_endpoint_payload(payload, manifest=manifest, report=report, manifest_digest=_sha256(manifest_path), source_digest=source_binding_digest)
            model.to(device)
            qa._optimizer_to_device(optimizer, device)
            adapter, initialization, initial_identity, parent_names = qa._append_adapter_group(model, optimizer, device=device)
            if payload.get("adapter_identity") != initial_identity or payload.get("adapter_initialization") != initialization:
                raise ValueError("saved endpoint adapter initialization changed")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            qa._optimizer_to_device(optimizer, device)
            torch.set_rng_state(payload["cpu_rng_state"])
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
            model.train(bool(payload["training_mode"]))
            adapter.train(bool(payload["adapter_training_mode"]))
            identity = qa._state_identity(model, adapter, optimizer)
            for key in ("model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest"):
                if identity[key] != payload[key]:
                    raise ValueError(f"saved endpoint state digest changed: {key}")
            if identity["model_parameter_names"] != payload["model_parameter_names"] or identity["adapter_parameter_names"] != payload["adapter_parameter_names"] or identity["optimizer_group_param_names"] != payload["optimizer_group_param_names"]:
                raise ValueError("saved endpoint parameter association changed")
            if qa._optimizer_metadata(optimizer) != payload["optimizer_metadata"]:
                raise ValueError("saved endpoint optimizer metadata changed")
            latent.validate_optimizer_adapter_association(optimizer, adapter)
            if parent_names != [name for name, _ in model.named_parameters()]:
                raise ValueError("saved endpoint parent parameter names changed")
            _mark_endpoint_loaded(counter, sink)
        return model, adapter, optimizer, dict(payload)
    except BaseException as exc:
        # Completed parent work is retained when endpoint restore fails; the
        # endpoint completion remains zero because _mark_endpoint_loaded was
        # never reached.
        _failure(counter, "endpoint_load", exc, phase="load")
        _flush(counter, sink)
        raise


def _capture_forward(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, x_bits: torch.Tensor, y_bits: torch.Tensor, ops: torch.Tensor, *, detailed: bool) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any] | None, list[dict[str, torch.Tensor]]]:
    captured: list[dict[str, torch.Tensor]] = []
    if not detailed:
        with torch.inference_mode(), torch.autocast(device_type=x_bits.device.type, enabled=False):
            logits = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops)
        return logits, None, captured
    original_step = model.step

    def observing_step(cache: Mapping[str, Any], op: torch.Tensor) -> Any:
        result = original_step(cache, op)
        logits, returned_cache = result
        hidden = returned_cache.get("h") if isinstance(returned_cache, Mapping) else None
        if not isinstance(hidden, torch.Tensor) or not isinstance(logits, tuple) or len(logits) != 2:
            raise ValueError("diagnostic observation received malformed native step")
        captured.append({"hidden": hidden.detach().cpu().clone(), "logits_x": logits[0].detach().cpu().clone(), "logits_y": logits[1].detach().cpu().clone()})
        return result

    setattr(model, "step", observing_step)
    try:
        with torch.inference_mode(), torch.autocast(device_type=x_bits.device.type, enabled=False):
            logits, diagnostics = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
        return logits, diagnostics, captured
    finally:
        setattr(model, "step", original_step)


def _trace_prediction(target: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> dict[str, Any]:
    metrics = accepted_runtime._trace_metrics(target, predicted)
    first_recovery = None
    if metrics["first_error"] is not None:
        for position in range(int(metrics["first_error"]), len(target)):
            if tuple(target[position]) == tuple(predicted[position]):
                first_recovery = position + 1
                break
    metrics["first_subsequent_recovery"] = first_recovery
    return metrics


def _detailed_state_record(*, identifier: str, program: Sequence[str], state: tuple[int, int], stratum: str, target: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]], diagnostics: Mapping[str, Any], captured: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    trace = diagnostic.trace_diagnostics(program, target, predicted, padding_extension_positions=tuple(index for index, (_opcode, role) in enumerate(zip(program, diagnostic._focus_entry(identifier)[2]), start=1) if role == "padding_extension"), focus_id=identifier)
    steps: list[dict[str, Any]] = []
    slot_inputs = diagnostics.get("slot_inputs")
    slot_writes = diagnostics.get("slot_writes")
    if not isinstance(slot_inputs, list) or not isinstance(slot_writes, list) or len(slot_inputs) != len(program) or len(captured) != len(program):
        raise ValueError("diagnostic native capture length changed")
    state_index = diagnostic.STATE_ORDER.index(state)
    for position, (slot_input, slot_write, capture) in enumerate(zip(slot_inputs, slot_writes, captured), start=1):
        z_tensor = slot_input.detach().cpu()[state_index]
        writer_tensor = slot_write.detach().cpu()[state_index]
        hidden = capture["hidden"][state_index]
        logits = torch.cat((capture["logits_x"][state_index], capture["logits_y"][state_index]), dim=-1)
        steps.append({
            "position": position,
            "opcode": str(program[position - 1]),
            "role": trace["positions"][position - 1]["role"],
            "z": z_tensor.tolist(),
            "writer": writer_tensor.tolist(),
            "hidden_norm": float(torch.linalg.vector_norm(hidden).item()),
            "output_norm": float(torch.linalg.vector_norm(logits).item()),
            "hidden_finite": bool(torch.isfinite(hidden).all().item()),
            "output_finite": bool(torch.isfinite(logits).all().item()),
        })
    metrics = diagnostic.summarize_step_metrics(steps, focus_id=identifier)
    first_error = trace["first_error"]
    first_role = trace["positions"][first_error - 1]["role"] if first_error is not None else None
    drift = diagnostic.cumulative_drift_summary(metrics, first_error=first_error, first_error_role=first_role, error_positions=tuple(item["position"] for item in trace["positions"] if item["wrong"]))
    return {"id": identifier, "suite": "padding", "length": len(program), "program": list(program), "state": list(state), "stratum": stratum, "trace": trace, "steps": metrics, "drift": drift, "finite": bool(all(item["finite"] for item in metrics))}


def _evaluate_program(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, spec: Mapping[str, Any], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identifier = str(spec["id"])
    program = tuple(str(opcode) for opcode in spec["program"])
    batch, x_bits, y_bits, ops = science._eval_batch(spec, device)
    length = len(program)
    _account_attempt(counter, length, sink)
    detailed = identifier in FOCUS_IDS
    try:
        with _timed(counter, f"forward_{identifier}", sink):
            logits, raw_diagnostics, captured = _capture_forward(model, adapter, x_bits, y_bits, ops, detailed=detailed)
        logits_x, logits_y = logits
        expected = (EVALUATION_STATES, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not bool(torch.isfinite(logits_x).all() and torch.isfinite(logits_y).all()):
            raise ValueError("diagnostic logits are malformed or nonfinite")
        decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
        decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
        predictions: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for state_index, (state, example) in enumerate(zip(dsl.STATE_ORDER, batch)):
            target = [list(pair) for pair in example.targets]
            predicted = [[int(decoded_x[state_index][position]), int(decoded_y[state_index][position])] for position in range(length)]
            prediction = {"state": list(state), "stratum": diagnostic._state_stratum(state), "target_trace": target, "predicted_trace": predicted, **_trace_prediction(target, predicted)}
            predictions.append(prediction)
            if detailed:
                if raw_diagnostics is None:
                    raise ValueError("detailed diagnostic capture is missing")
                records.append(_detailed_state_record(identifier=identifier, program=program, state=state, stratum=prediction["stratum"], target=target, predicted=predicted, diagnostics=raw_diagnostics, captured=captured))
        _account_complete(counter, length, sink)
        return {"schema": PROGRAM_SCHEMA, "id": identifier, "suite": str(spec["suite"]), "length": length, "program": list(program), "predictions": predictions, "detailed": detailed, "diagnostic_records": records if detailed else None}, records
    except BaseException as exc:
        _failure(counter, "forward", exc, phase="evaluation")
        _flush(counter, sink)
        raise


def _load_runtime_settings(manifest: Mapping[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    device, settings = science._configure_runtime()
    if settings != manifest.get("runtime"):
        raise ValueError("diagnostic runtime settings drifted from accepted science manifest")
    return device, settings


def run_diagnostic(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = _sink(accounting_path)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "manifest": {"path": _relative(manifest_path, root), "sha256": None}, "budget": {"calls": EVALUATION_CALLS, "cases": EVALUATION_CASES, "readout_positions": EVALUATION_POSITIONS, "native_steps": EVALUATION_NATIVE_STEPS, "optimizer_updates": 0}, "focus_budget": {"calls": FOCUS_CALLS, "cases": FOCUS_CASES, "readout_positions": FOCUS_POSITIONS, "native_steps": FOCUS_NATIVE_STEPS}, "accounting": dict(counter)}
    model: torch.nn.Module | None = None
    adapter: latent.LatentSlotAdapter | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        manifest, science_report, scope, _final, evidence = _validate_evidence(root, manifest_path)
        device, settings = _load_runtime_settings(manifest)
        binding = _source_binding(root, manifest, evidence, settings)
        input_freeze = {"schema": INPUT_SCHEMA, "immutable": True, "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, "source_binding": binding, "scope": {"path": _relative(_resolve(SCIENCE_SCOPE, root), root), "sha256": evidence["scope"]["sha256"], "digest": scope["digest"], "programs": EVALUATION_PROGRAMS, "states": EVALUATION_STATES, "targets": "bound by saved scope bytes"}, "focus": {"ids": list(FOCUS_IDS), "scope": _focus_scope(scope)}, "runtime": settings, "semantics": {"latent_helper": DIAGNOSTIC_HELPER_SHA256, "native_steps": latent.NATIVE_STEPS, "writer_alignment": "writer[t]=z[t+1] checked after capture", "semantic_slot_accuracy": None}}
        input_freeze["digest"] = science._digest(input_freeze)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report["manifest"]["sha256"] = _sha256(manifest_path)
        report["source_binding_digest"] = binding["digest"]
        report["runtime"] = settings
        report["endpoint"] = evidence["endpoint"]
        report["focus_ids"] = list(FOCUS_IDS)
        with _timed(counter, "load", sink):
            model, adapter, optimizer, endpoint_payload = _load_endpoint(manifest=manifest, report=science_report, endpoint=_resolve(SCIENCE_ENDPOINT, root), manifest_path=manifest_path, source_binding_digest=str(manifest["source_inventory"]["digest"]), device=device, counter=counter, sink=sink, root=root)
        if counter["completed_endpoint_loads"] != 1:
            raise ValueError("endpoint load did not complete exactly once")
        summary_rows: list[dict[str, Any]] = []
        focus_records: list[dict[str, Any]] = []
        rows_dir = out / "program_rows"
        rows_dir.mkdir(parents=True, exist_ok=False)
        with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
            for spec in scope["programs"]:
                row, records = _evaluate_program(model=model, adapter=adapter, spec=spec, device=device, counter=counter, sink=sink)
                _atomic_json(rows_dir / f"{len(summary_rows):03d}_{row['id']}.json", row, refuse=True)
                summary_rows.append({key: row[key] for key in ("id", "suite", "length", "program", "predictions")})
                focus_records.extend(records)
        if len(summary_rows) != EVALUATION_PROGRAMS or len(focus_records) != FOCUS_CASES:
            raise ValueError("diagnostic output scope count changed")
        report["aggregates"] = science._aggregate_rows(summary_rows)
        report["focus_aggregates"] = diagnostic.aggregate_trace_records(focus_records)
        _validate_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "limitations": ["single saved latent local2000 endpoint", "observational measurements do not establish causality or semantic slot labels", "no writer-frozen control was run"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        _failure(counter, "diagnostic", exc, phase="evaluation" if model is not None else "load")
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise
    finally:
        # No persistent observation wrapper is used; the temporary step wrapper
        # is removed by _capture_forward even when a native call fails.
        model = None
        adapter = None
        optimizer = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_diagnostic(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
