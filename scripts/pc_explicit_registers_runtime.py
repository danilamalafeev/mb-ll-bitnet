"""Bounded runtime for the explicit-register primitive tables.

The default entry point does no work.  ``--qa`` runs the two-call gate and
``--science`` runs the registered six-endpoint table construction.  Both modes
use the accepted strict checkpoint loader and write a fresh, separate output.
No training, baseline replay, or program-search code is present here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

from looped_bitnet import register_e15 as dsl
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import pc_explicit_registers as contract
from scripts import pc_inference_migration as migration
from scripts import width_e32 as e32


ROOT = migration.ROOT
OUTPUT_ROOT = ROOT / "runs" / "pc_explicit_registers_v1"
QA_OUTPUT = OUTPUT_ROOT / "qa"
SCIENCE_OUTPUT = OUTPUT_ROOT / "science"
SAVED_PROGRAMS = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "exact_map_proof.json"
LINEAGE = migration.LINEAGE
QA_LABEL = "float128_seed0/B"
QA_STATES = ((0, 0), (15, 1))
QA_PROGRAM = ("ADD", "XOR")
SCIENCE_ENDPOINTS = tuple(item for item in migration.ENDPOINTS if item[-1] == "B")
NATIVE_STEPS_PER_POSITION = contract.NATIVE_STEPS_PER_POSITION
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_EXPLICIT_REGISTERS_PROTOCOL.md"
SCIENCE_MANIFEST = OUTPUT_ROOT / "science_manifest.json"
QA_ACCEPT = OUTPUT_ROOT / "qa_accept.json"
QA_ARTIFACTS = {
    "accounting": QA_OUTPUT / "accounting.json",
    "endpoint": QA_OUTPUT / "endpoint.json",
    "manifest": QA_OUTPUT / "manifest.json",
    "report": QA_OUTPUT / "report.json",
}
BASE_SCIENCE_MANIFEST = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "manifest.json"
BASELINE_ENDPOINT_ROOT = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1" / "endpoints"
STRATA = {state: name for name, values in dsl.state_split().items() for state in values}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite fresh output {path}")
    path.mkdir(parents=True, exist_ok=True)


def _source_inventory() -> dict[str, dict[str, str]]:
    files = {
        "runner": Path(__file__).resolve(),
        "contract": Path(contract.__file__).resolve(),
        "migration": Path(migration.__file__).resolve(),
        "package_init": Path((ROOT / "looped_bitnet" / "__init__.py")).resolve(),
        "dsl": Path(dsl.__file__).resolve(),
        "width_model": Path((ROOT / "looped_bitnet" / "width_e32.py")).resolve(),
        "register_model": Path((ROOT / "looped_bitnet" / "register_e15.py")).resolve(),
        "model_core": Path((ROOT / "looped_bitnet" / "model.py")).resolve(),
        "model_quantization": Path((ROOT / "looped_bitnet" / "quantization.py")).resolve(),
        "model_runtime": Path((ROOT / "looped_bitnet" / "runtime.py")).resolve(),
        "model_engine": Path((ROOT / "looped_bitnet" / "engine.py")).resolve(),
        "model_longer_native8": Path((ROOT / "looped_bitnet" / "longer_native8_e20.py")).resolve(),
        "model_replication": Path((ROOT / "looped_bitnet" / "replication_e22.py")).resolve(),
        "model_qat_match": Path((ROOT / "looped_bitnet" / "qat_match_e25.py")).resolve(),
        "model_float_qat": Path((ROOT / "looped_bitnet" / "float_qat_e16.py")).resolve(),
        "model_w4": Path((ROOT / "looped_bitnet" / "w4_e27.py")).resolve(),
        "model_weight_only": Path((ROOT / "looped_bitnet" / "weight_only_e26.py")).resolve(),
        "model_step_budget": Path((ROOT / "looped_bitnet" / "step_budget_e18.py")).resolve(),
        "model_data": Path((ROOT / "looped_bitnet" / "data.py")).resolve(),
        "model_config": Path((ROOT / "looped_bitnet" / "config.py")).resolve(),
        "model_bit_input": Path((ROOT / "looped_bitnet" / "bit_input_e17.py")).resolve(),
        "e32_loader": Path((ROOT / "scripts" / "width_e32.py")).resolve(),
        "e36_loader": Path(wave.__file__).resolve(),
        "e38_loader": Path(followup.__file__).resolve(),
    }
    return {
        name: {"path": migration.posix_relative(path), "sha256": migration.sha256_file(path)}
        for name, path in files.items()
    }


def _runtime_settings() -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return {
        "device": "cuda",
        "default_dtype": str(torch.get_default_dtype()),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast": False,
        "native_steps_per_position": NATIVE_STEPS_PER_POSITION,
        "eval": True,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, path: Any = None) -> None:
    item: dict[str, Any] = {"kind": kind, "type": type(exc).__name__, "message": str(exc)}
    if path is not None:
        item["path"] = str(path)
    counter["failures"].append(item)


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = migration.sha256_file(Path(path))
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return actual


def _flush_accounting(counter: dict[str, Any], sink: Any = None) -> None:
    if sink is not None:
        sink(counter)


@contextmanager
def _count_torch_loads(counter: dict[str, Any], sink: Any = None) -> Iterator[None]:
    original = torch.load

    def counted(*args: Any, **kwargs: Any) -> Any:
        source = args[0] if args else kwargs.get("f")
        counter["attempted_loads"] += 1
        _flush_accounting(counter, sink)
        try:
            value = original(*args, **kwargs)
        except BaseException as exc:
            _failure(counter, "torch.load", exc, path=source)
            _flush_accounting(counter, sink)
            raise
        counter["completed_loads"] += 1
        _flush_accounting(counter, sink)
        return value

    torch.load = counted  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _new_counter() -> dict[str, Any]:
    return {
        "attempted_loads": 0,
        "completed_loads": 0,
        "attempted_endpoint_loads": 0,
        "completed_endpoint_loads": 0,
        "attempted_forwards": 0,
        "completed_forwards": 0,
        "attempted_cases": 0,
        "completed_cases": 0,
        "attempted_readout_positions": 0,
        "completed_readout_positions": 0,
        "attempted_native_steps": 0,
        "completed_native_steps": 0,
        "updates": 0,
        "failures": [],
    }


def _state(value: Sequence[int], *, label: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be a length-2 sequence")
    if any(type(item) is not int or item < 0 or item > 15 for item in value):
        raise ValueError(f"{label} must contain registers in 0..15")
    return int(value[0]), int(value[1])


def _one_op_forward(
    model: torch.nn.Module,
    states: Sequence[Sequence[int]],
    opcode: str,
    device: torch.device,
    counter: dict[str, Any],
    sink: Any = None,
) -> list[list[int]]:
    if opcode not in contract.OPS:
        raise ValueError(f"unknown opcode: {opcode!r}")
    normalized = [_state(state, label="forward state") for state in states]
    if not normalized:
        raise ValueError("forward state batch cannot be empty")
    cases = len(normalized)
    counter["attempted_forwards"] += 1
    counter["attempted_cases"] += cases
    counter["attempted_readout_positions"] += cases
    counter["attempted_native_steps"] += cases * NATIVE_STEPS_PER_POSITION
    _flush_accounting(counter, sink)
    try:
        x = torch.tensor([state[0] for state in normalized], dtype=torch.long, device=device)
        y = torch.tensor([state[1] for state in normalized], dtype=torch.long, device=device)
        op_ids = torch.full((cases, 1), dsl.OP_TO_ID[opcode], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits_x, logits_y = model(x, y, op_ids)
        expected_shape = (cases, 1, 16)
        if tuple(logits_x.shape) != expected_shape or tuple(logits_y.shape) != expected_shape:
            raise ValueError(f"{opcode} model output shape changed: {tuple(logits_x.shape)}, {tuple(logits_y.shape)}")
        decoded = torch.stack((logits_x[:, 0].argmax(-1), logits_y[:, 0].argmax(-1)), dim=-1)
        result = decoded.detach().cpu().tolist()
        result = [[int(pair[0]), int(pair[1])] for pair in result]
        for pair in result:
            _state(pair, label=f"{opcode} decoded output")
    except BaseException as exc:
        _failure(counter, "forward", exc, path=opcode)
        _flush_accounting(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases
    counter["completed_native_steps"] += cases * NATIVE_STEPS_PER_POSITION
    _flush_accounting(counter, sink)
    return result


def _load_counter_model(
    endpoint: tuple[str, str, str, int, str],
    lineage: Mapping[str, Any],
    counter: dict[str, Any],
    sink: Any = None,
) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    label, kind, arm, seed, branch = endpoint
    checkpoint, _reference = migration.endpoint_paths(kind, arm, seed, branch)
    expected_hash = migration.manifest_hash_for_checkpoint(label)
    if not checkpoint.is_file() or migration.sha256_file(checkpoint) != expected_hash:
        exc = ValueError(f"checkpoint identity mismatch: {label}")
        _failure(counter, "checkpoint", exc, path=checkpoint)
        _flush_accounting(counter, sink)
        raise exc
    counter["attempted_endpoint_loads"] += 1
    _flush_accounting(counter, sink)
    try:
        model, loaded_path, payload = migration.load_endpoint(kind, arm, seed, branch, lineage)
        if loaded_path != checkpoint or migration.sha256_file(checkpoint) != expected_hash:
            raise ValueError(f"checkpoint identity mismatch after load: {label}")
        if payload.get("model_digest") != migration.core.digest_state_dict(model):
            raise ValueError(f"model digest mismatch after load: {label}")
        model.eval()
        counter["completed_endpoint_loads"] += 1
        _flush_accounting(counter, sink)
        return model, checkpoint, payload
    except BaseException as exc:
        _failure(counter, "checkpoint_load", exc, path=checkpoint)
        _flush_accounting(counter, sink)
        raise


def _model_identity(model: torch.nn.Module) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "model_digest": migration.core.digest_state_dict(model),
        "training": bool(model.training),
        "cpu_rng_digest": migration.core.digest_object(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        identity["cuda_rng_digest"] = migration.core.digest_object(torch.cuda.get_rng_state_all())
    return identity


def _assert_model_preserved(before: Mapping[str, Any], after: Mapping[str, Any], label: str) -> None:
    if dict(before) != dict(after):
        raise ValueError(f"model or RNG identity changed during {label}")


def _program_binding(mode: str, endpoints: Sequence[tuple[str, str, str, int, str]], states: Sequence[Sequence[int]], programs: Sequence[Sequence[str]]) -> dict[str, Any]:
    endpoint_rows = []
    for label, kind, arm, seed, branch in endpoints:
        checkpoint, _ = migration.endpoint_paths(kind, arm, seed, branch)
        endpoint_rows.append({
            "label": label,
            "kind": kind,
            "arm": arm,
            "seed": seed,
            "branch": branch,
            "checkpoint": migration.posix_relative(checkpoint),
            "checkpoint_sha256": migration.manifest_hash_for_checkpoint(label),
        })
    return {
        "mode": mode,
        "endpoint_rows": endpoint_rows,
        "state_order": [list(_state(state, label="binding state")) for state in states],
        "primitive_ops": list(contract.OPS),
        "programs": [list(program) for program in programs],
    }


def _load_saved_programs() -> tuple[list[dict[str, Any]], str]:
    if not SAVED_PROGRAMS.is_file():
        raise FileNotFoundError(SAVED_PROGRAMS)
    digest = migration.sha256_file(SAVED_PROGRAMS)
    value = migration.read_json(SAVED_PROGRAMS)
    if value.get("schema") != "pc_inference_science_exact_maps_v1":
        raise ValueError("saved science program proof schema changed")
    if value.get("states") != [list(state) for state in contract.STATES]:
        raise ValueError("saved science state order changed")
    rows: list[dict[str, Any]] = []
    padding = value.get("padding", {}).get("rows")
    if not isinstance(padding, list) or len(padding) != 45:
        raise ValueError("saved padding program count changed")
    rows.extend({**row, "suite": "padding"} for row in padding)
    by_length = value.get("compositions", {}).get("by_length")
    if not isinstance(by_length, Mapping):
        raise ValueError("saved composition programs missing")
    for length in ("12", "16", "24", "32"):
        values = by_length.get(length)
        if not isinstance(values, Mapping) or not isinstance(values.get("rows"), list) or len(values["rows"]) != 6:
            raise ValueError(f"saved composition count changed at length {length}")
        rows.extend({**row, "suite": "compositions"} for row in values["rows"])
    if len(rows) != 69:
        raise ValueError("saved program count changed")
    identifiers: set[str] = set()
    for index, row in enumerate(rows):
        identifier = str(row.get("id", f"saved_program_{index}"))
        if identifier in identifiers:
            raise ValueError("saved program IDs are duplicated")
        identifiers.add(identifier)
        program = tuple(row.get("program", ()))
        if not program or any(op not in contract.OPS for op in program):
            raise ValueError(f"saved program is malformed: {identifier}")
        if int(row.get("length", -1)) != len(program):
            raise ValueError(f"saved program length changed: {identifier}")
        if "map" in row:
            expected_map = [list(contract.dsl_trace(program, state)[-1]) for state in contract.STATES]
            if row["map"] != expected_map:
                raise ValueError(f"saved DSL map changed: {identifier}")
    return rows, digest


def _baseline_path(label: str) -> Path:
    return BASELINE_ENDPOINT_ROOT / f"{label.replace('/', '_')}.json"


def _program_index(programs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {"padding": {}, "compositions": {}}
    for row in programs:
        suite = str(row["suite"])
        identifier = str(row["id"])
        if suite not in result or identifier in result[suite]:
            raise ValueError(f"saved program identity changed: {suite}/{identifier}")
        result[suite][identifier] = row
    return result


def _validate_baseline_endpoint(
    path: Path,
    endpoint: tuple[str, str, str, int, str],
    programs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    label, _kind, _arm, _seed, _branch = endpoint
    value = migration.read_json(path)
    if value.get("label") != label or value.get("branch") != "B":
        raise ValueError(f"baseline endpoint identity changed: {label}")
    expected_checkpoint = migration.manifest_hash_for_checkpoint(label)
    if value.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError(f"baseline checkpoint join changed: {label}")
    expected = _program_index(programs)
    result: dict[str, dict[str, dict[str, Any]]] = {"padding": {}, "compositions": {}}
    for suite in ("padding", "compositions"):
        rows = value.get(suite)
        if not isinstance(rows, list) or len(rows) != len(expected[suite]):
            raise ValueError(f"baseline {suite} count changed: {label}")
        for row in rows:
            identifier = str(row.get("id", ""))
            if identifier not in expected[suite] or identifier in result[suite]:
                raise ValueError(f"baseline {suite} ID join changed: {label}/{identifier}")
            expected_row = expected[suite][identifier]
            if row.get("program") != expected_row.get("program") or int(row.get("length", -1)) != int(expected_row["length"]):
                raise ValueError(f"baseline program string changed: {label}/{identifier}")
            predictions = row.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != len(contract.STATES):
                raise ValueError(f"baseline state count changed: {label}/{identifier}")
            by_state: dict[tuple[int, int], dict[str, Any]] = {}
            program = tuple(str(op) for op in expected_row["program"])
            for prediction in predictions:
                state = _state(prediction.get("state", ()), label="baseline state")
                if state in by_state or state not in STRATA:
                    raise ValueError(f"baseline state join changed: {label}/{identifier}/{state}")
                if prediction.get("stratum") != STRATA[state]:
                    raise ValueError(f"baseline stratum changed: {label}/{identifier}/{state}")
                target = contract.dsl_trace(program, state)
                if prediction.get("target_trace") != target:
                    raise ValueError(f"baseline DSL target changed: {label}/{identifier}/{state}")
                predicted = prediction.get("predicted_trace")
                contract.trace_diagnostic(predicted, target)
                by_state[state] = prediction
            if set(by_state) != set(contract.STATES):
                raise ValueError(f"baseline state order changed: {label}/{identifier}")
            result[suite][identifier] = {"row": row, "by_state": by_state}
    return result


def _baseline_bindings(
    programs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, Any]]]]:
    bindings: list[dict[str, str]] = []
    validated: dict[str, dict[str, dict[str, Any]]] = {}
    for endpoint in SCIENCE_ENDPOINTS:
        label = endpoint[0]
        path = _baseline_path(label)
        expected_checkpoint = migration.manifest_hash_for_checkpoint(label)
        if not path.is_file():
            raise ValueError(f"baseline endpoint missing: {label}")
        value = migration.read_json(path)
        if value.get("checkpoint_sha256") != expected_checkpoint:
            raise ValueError(f"baseline endpoint checkpoint hash mismatch: {label}")
        validated[label] = _validate_baseline_endpoint(path, endpoint, programs)
        bindings.append({
            "label": label,
            "path": migration.posix_relative(path),
            "sha256": migration.sha256_file(path),
            "checkpoint_sha256": expected_checkpoint,
        })
    return bindings, validated


def _summarize_composed_program(
    row: Mapping[str, Any],
    composed: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    program = tuple(str(op) for op in row["program"])
    by_state = baseline["by_state"]
    strata: dict[str, dict[str, Any]] = {}

    def fresh_stratum() -> dict[str, Any]:
        return {
            "denominator_states": 0,
            "denominator_positions": 0,
            "baseline_final_correct": 0,
            "composed_final_correct": 0,
            "baseline_full_trace_correct": 0,
            "composed_full_trace_correct": 0,
            "paired_final": {key: 0 for key in contract.PAIR_CATEGORIES},
            "paired_full_trace": {key: 0 for key in contract.PAIR_CATEGORIES},
            "baseline_first_error_position": {},
            "baseline_first_recovery_position": {},
            "composed_first_error_position": {},
            "composed_first_recovery_position": {},
            "carried_input_local": {
                "input_matches_true_local_correct": {"denominator": 0, "output_correct": 0},
                "input_matches_true_local_wrong": {"denominator": 0, "output_correct": 0},
                "input_differs_true_local_correct": {"denominator": 0, "output_correct": 0},
                "input_differs_true_local_wrong": {"denominator": 0, "output_correct": 0},
            },
        }

    def bump_position(mapping: dict[str, int], value: Any) -> None:
        key = "none" if value is None else str(value)
        mapping[key] = mapping.get(key, 0) + 1

    def pair_category(left: bool, right: bool) -> str:
        if left and right:
            return "both_correct"
        if left:
            return "baseline_correct_composed_wrong"
        if right:
            return "baseline_wrong_composed_correct"
        return "both_wrong"

    for item in composed:
        state = _state(item["state"], label="composed state")
        if state not in by_state:
            raise ValueError(f"composed state missing from baseline: {state}")
        predicted = item["predicted_trace"]
        target = contract.dsl_trace(program, state)
        composed_diag = contract.trace_diagnostic(predicted, target)
        baseline_predicted = by_state[state]["predicted_trace"]
        baseline_diag = contract.trace_diagnostic(baseline_predicted, target)
        stratum = STRATA[state]
        stats = strata.setdefault(stratum, fresh_stratum())
        stats["denominator_states"] += 1
        stats["baseline_final_correct"] += int(baseline_diag["final_correct"])
        stats["composed_final_correct"] += int(composed_diag["final_correct"])
        stats["baseline_full_trace_correct"] += int(baseline_diag["full_trace_correct"])
        stats["composed_full_trace_correct"] += int(composed_diag["full_trace_correct"])
        stats["paired_final"][pair_category(baseline_diag["final_correct"], composed_diag["final_correct"])] += 1
        stats["paired_full_trace"][pair_category(baseline_diag["full_trace_correct"], composed_diag["full_trace_correct"])] += 1
        bump_position(stats["baseline_first_error_position"], baseline_diag["first_error_position"])
        bump_position(stats["baseline_first_recovery_position"], baseline_diag["first_recovery_position"])
        bump_position(stats["composed_first_error_position"], composed_diag["first_error_position"])
        bump_position(stats["composed_first_recovery_position"], composed_diag["first_recovery_position"])
        stats["denominator_positions"] += len(program)
        for observation in _carried_local_observations(program, state, predicted, target):
            input_matches = observation["input_matches_true_prestate"]
            local_correct = observation["local_step_correct"]
            output_correct = observation["output_matches_true_poststate"]
            cell = (
                "input_matches_true_local_correct" if input_matches and local_correct else
                "input_matches_true_local_wrong" if input_matches else
                "input_differs_true_local_correct" if local_correct else
                "input_differs_true_local_wrong"
            )
            stats["carried_input_local"][cell]["denominator"] += 1
            stats["carried_input_local"][cell]["output_correct"] += int(output_correct)
    if sum(stats["denominator_states"] for stats in strata.values()) != len(contract.STATES):
        raise ValueError("composed state coverage changed")
    for item in composed:
        _state(item["state"], label="composed state")
    return {
        "id": str(row["id"]),
        "suite": str(row["suite"]),
        "program": list(program),
        "length": len(program),
        "states": len(composed),
        "strata": strata,
    }


def _endpoint_record(
    endpoint: tuple[str, str, str, int, str],
    model: torch.nn.Module,
    checkpoint: Path,
    payload: Mapping[str, Any],
    states: Sequence[Sequence[int]],
    programs: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    counter: dict[str, Any],
    sink: Any = None,
    table_sink: Any = None,
) -> dict[str, Any]:
    label, kind, arm, seed, branch = endpoint
    model = model.to(device)
    model.eval()
    before = _model_identity(model)
    tables: dict[str, list[dict[str, Any]]] = {}
    normalized_states = [_state(state, label="science state") for state in states]
    table_started = time.monotonic()
    for opcode in contract.OPS:
        decoded = _one_op_forward(model, normalized_states, opcode, device, counter, sink)
        tables[opcode] = [
            {"state": list(state), "output": output}
            for state, output in zip(normalized_states, decoded)
        ]
    table_seconds = time.monotonic() - table_started
    contract.validate_transition_tables(tables, normalized_states)
    if table_sink is not None:
        table_sink({
            "schema": "pc_explicit_registers_tables_v1",
            "status": "tables_complete",
            "label": label,
            "checkpoint": migration.posix_relative(checkpoint),
            "checkpoint_sha256": migration.sha256_file(checkpoint),
            "expected_checkpoint_sha256": migration.manifest_hash_for_checkpoint(label),
            "model_digest": payload.get("model_digest"),
            "tables": tables,
            "timing": {"table_construction_seconds": table_seconds},
        })
    derived = []
    lookup_started = time.monotonic()
    for program_row in programs:
        composed = contract.compose_transition_tables(tables, program_row["program"], normalized_states)
        identifier = str(program_row["id"])
        suite = str(program_row["suite"])
        if suite not in baseline or identifier not in baseline[suite]:
            raise ValueError(f"baseline join missing: {label}/{suite}/{identifier}")
        derived.append(_summarize_composed_program(program_row, composed, baseline[suite][identifier]))
    lookup_seconds = time.monotonic() - lookup_started
    after = _model_identity(model)
    _assert_model_preserved(before, after, label)
    return {
        "label": label,
        "kind": kind,
        "arm": arm,
        "seed": seed,
        "branch": branch,
        "checkpoint": migration.posix_relative(checkpoint),
        "checkpoint_sha256": migration.sha256_file(checkpoint),
        "expected_checkpoint_sha256": migration.manifest_hash_for_checkpoint(label),
        "model_digest": payload.get("model_digest"),
        "baseline_endpoint": migration.posix_relative(_baseline_path(label)),
        "baseline_endpoint_sha256": migration.sha256_file(_baseline_path(label)),
        "model_identity_before": before,
        "model_identity_after": after,
        "tables": tables,
        "programs": derived,
        "timing": {
            "table_construction_seconds": table_seconds,
            "lookup_composition_seconds": lookup_seconds,
            "measurement_note": "decoded CPU transfer is included in table construction; lookup timing is not neural execution time",
        },
    }


def _record_file(path: Path) -> dict[str, str]:
    return {"path": migration.posix_relative(path), "sha256": migration.sha256_file(path)}


def _accepted_manifests() -> dict[str, Path]:
    return {
        "e36_preflight_manifest": migration.E36_PREFLIGHT / "manifest.json",
        "e37_preflight_manifest": migration.E37_PREFLIGHT / "manifest.json",
        "e32_preflight_manifest": migration.E32_PREFLIGHT / "manifest.json",
        "runtime_lineage": LINEAGE,
        "base_science_manifest": BASE_SCIENCE_MANIFEST,
    }


def _qa_bindings() -> dict[str, Any]:
    if not QA_ACCEPT.is_file():
        raise ValueError("accepted QA result is missing")
    accepted = migration.read_json(QA_ACCEPT)
    if accepted.get("status") != "QA_ACCEPT" or accepted.get("model_calls_by_auditor") != 0:
        raise ValueError("accepted QA result is not valid")
    expected = dict(contract.QA_BUDGET_MAX)
    expected.update({"attempted_loads": 2, "completed_loads": 2})
    accounting = accepted.get("accounting")
    if not isinstance(accounting, Mapping) or any(accounting.get(key) != value for key, value in expected.items()):
        raise ValueError("accepted QA accounting changed")
    if accounting.get("failures") != [] or accepted.get("source_unchanged") is not True or accepted.get("checkpoint_unchanged") is not True or accepted.get("model_rng_mode_preserved") is not True:
        raise ValueError("accepted QA integrity gate changed")
    accepted_files = accepted.get("qa_files")
    if not isinstance(accepted_files, Mapping):
        raise ValueError("accepted QA artifact hash binding is missing")
    artifacts = {}
    for name, path in QA_ARTIFACTS.items():
        if not path.is_file():
            raise ValueError(f"accepted QA artifact is missing: {name}")
        record = _record_file(path)
        expected_hash = accepted_files.get(path.name)
        if not isinstance(expected_hash, str) or record["sha256"] != expected_hash:
            raise ValueError(f"accepted QA artifact hash mismatch: {path.name}")
        artifacts[name] = record
    source_review = OUTPUT_ROOT / "qa_source_review.json"
    if not source_review.is_file():
        raise ValueError("accepted QA source review is missing")
    return {
        "accept": _record_file(QA_ACCEPT),
        "artifacts": artifacts,
        "source_review": _record_file(source_review),
    }


def _validate_reviewed_model_sources(source_review: Path | None = None) -> None:
    source_review = source_review or (OUTPUT_ROOT / "qa_source_review.json")
    review = migration.read_json(source_review)
    expected = review.get("additional_model_source_inventory")
    if not isinstance(expected, Mapping):
        raise ValueError("accepted QA source inventory is malformed")
    for raw_path, expected_hash in expected.items():
        path = Path(str(raw_path))
        if not path.is_file() or migration.sha256_file(path) != expected_hash:
            raise ValueError(f"accepted model source hash mismatch: {raw_path}")


def _validate_retained_qa_sources() -> None:
    qa_manifest = QA_OUTPUT / "manifest.json"
    value = migration.read_json(qa_manifest)
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("accepted QA source binding is malformed")
    for name, record in source.items():
        if name == "runner":
            continue
        if not isinstance(record, Mapping):
            raise ValueError(f"accepted QA source record is malformed: {name}")
        path = _path_from_record(record)
        if not path.is_file() or migration.sha256_file(path) != record.get("sha256"):
            raise ValueError(f"accepted QA retained source hash mismatch: {name}")


def _validate_qa_bindings(value: Mapping[str, Any]) -> None:
    accept = value.get("accept")
    if not isinstance(accept, Mapping):
        raise ValueError("science manifest QA acceptance binding is malformed")
    _require_file_hash(_path_from_record(accept), str(accept.get("sha256")), "QA acceptance")
    current = _qa_bindings()
    if current != value:
        raise ValueError("accepted QA binding changed")
    _validate_reviewed_model_sources(_path_from_record(current["source_review"]))
    _validate_retained_qa_sources()


def _path_from_record(record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("file binding path is malformed")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _validate_file_records(records: Mapping[str, Any], label: str) -> None:
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} binding is malformed: {name}")
        _require_file_hash(_path_from_record(record), str(record.get("sha256")), f"{label} {name}")


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _science_manifest_payload(
    programs: Sequence[Mapping[str, Any]],
    programs_digest: str,
    baseline_bindings: Sequence[Mapping[str, str]],
    qa: Mapping[str, Any],
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _program_binding(
        "science", SCIENCE_ENDPOINTS, contract.STATES, [tuple(row["program"]) for row in programs]
    )
    binding_digest = contract.canonical_digest(binding)
    contract.validate_frozen_identity(binding, binding, expected_digest=binding_digest, label="science binding")
    accepted = _accepted_manifests()
    return {
        "schema": "pc_explicit_registers_science_manifest_v1",
        "status": "frozen",
        "source": dict(source),
        "runtime": {"identity": dict(runtime_identity), "settings": dict(settings)},
        "binding": {"value": binding, "sha256": binding_digest},
        "saved_programs": {"path": migration.posix_relative(SAVED_PROGRAMS), "sha256": programs_digest,
                           "count": len(programs), "ids_and_programs": [
                               {"id": str(row["id"]), "program": list(row["program"]), "length": int(row["length"]),
                                "suite": str(row["suite"])} for row in programs]},
        "baseline_endpoints": list(baseline_bindings),
        "qa": dict(qa),
        "protocol": _record_file(PROTOCOL),
        "accepted_manifests": {name: _record_file(path) for name, path in accepted.items()},
        "budget": contract.SCIENCE_BUDGET,
        "timing_definition": "table construction includes decoded CPU transfer; CPU lookup/composition is separate and neither is uncached sequential-neural speed",
    }


def prepare_science() -> dict[str, Any]:
    if SCIENCE_MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite frozen science manifest {SCIENCE_MANIFEST}")
    programs, programs_digest = _load_saved_programs()
    baseline_bindings, _ = _baseline_bindings(programs)
    qa = _qa_bindings()
    source = _source_inventory()
    settings = _runtime_settings()
    runtime_identity = _runtime_identity()
    manifest = _science_manifest_payload(
        programs, programs_digest, baseline_bindings, qa, source, settings, runtime_identity
    )
    _write_json(SCIENCE_MANIFEST, manifest)
    return {"status": "prepared", "manifest": migration.posix_relative(SCIENCE_MANIFEST),
            "manifest_sha256": migration.sha256_file(SCIENCE_MANIFEST), "forwards": 0, "loads": 0}


def _validate_science_manifest() -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    if not SCIENCE_MANIFEST.is_file():
        raise FileNotFoundError(f"frozen science manifest is missing: {SCIENCE_MANIFEST}")
    manifest_digest = migration.sha256_file(SCIENCE_MANIFEST)
    manifest = migration.read_json(SCIENCE_MANIFEST)
    if manifest.get("schema") != "pc_explicit_registers_science_manifest_v1" or manifest.get("status") != "frozen":
        raise ValueError("frozen science manifest identity changed")
    programs, programs_digest = _load_saved_programs()
    saved = manifest.get("saved_programs")
    if not isinstance(saved, Mapping) or saved.get("sha256") != programs_digest or saved.get("count") != len(programs):
        raise ValueError("saved science program binding changed")
    if saved.get("path") != migration.posix_relative(SAVED_PROGRAMS):
        raise ValueError("saved science program path changed")
    expected_programs = [
        {"id": str(row["id"]), "program": list(row["program"]), "length": int(row["length"]), "suite": str(row["suite"])}
        for row in programs
    ]
    if saved.get("ids_and_programs") != expected_programs:
        raise ValueError("saved science IDs or program strings changed")
    _require_file_hash(SAVED_PROGRAMS, programs_digest, "saved science programs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or dict(source) != _source_inventory():
        raise ValueError("science source hash binding changed")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("settings") != _runtime_settings() or runtime.get("identity") != _runtime_identity():
        raise ValueError("science runtime settings changed")
    _validate_qa_bindings(manifest.get("qa", {}))
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("science protocol binding is malformed")
    if protocol.get("path") != migration.posix_relative(PROTOCOL):
        raise ValueError("science protocol path changed")
    _require_file_hash(_path_from_record(protocol), str(protocol.get("sha256")), "science protocol")
    accepted = manifest.get("accepted_manifests")
    if not isinstance(accepted, Mapping):
        raise ValueError("accepted manifest bindings are malformed")
    _validate_file_records(accepted, "accepted manifest")
    baseline = manifest.get("baseline_endpoints")
    if not isinstance(baseline, list) or len(baseline) != len(SCIENCE_ENDPOINTS):
        raise ValueError("baseline endpoint binding count changed")
    expected_baseline = {item["label"]: item for item in baseline if isinstance(item, Mapping) and "label" in item}
    if set(expected_baseline) != {item[0] for item in SCIENCE_ENDPOINTS}:
        raise ValueError("baseline endpoint labels changed")
    for endpoint in SCIENCE_ENDPOINTS:
        label = endpoint[0]
        item = expected_baseline[label]
        if item.get("path") != migration.posix_relative(_baseline_path(label)):
            raise ValueError(f"baseline endpoint path changed: {label}")
        _require_file_hash(_path_from_record(item), str(item.get("sha256")), f"baseline endpoint {label}")
        if item.get("checkpoint_sha256") != migration.manifest_hash_for_checkpoint(label):
            raise ValueError(f"baseline checkpoint binding changed: {label}")
    binding_value = manifest.get("binding", {}).get("value") if isinstance(manifest.get("binding"), Mapping) else None
    binding_digest = manifest.get("binding", {}).get("sha256") if isinstance(manifest.get("binding"), Mapping) else None
    current_binding = _program_binding(
        "science", SCIENCE_ENDPOINTS, contract.STATES, [tuple(row["program"]) for row in programs]
    )
    if not isinstance(binding_value, Mapping) or not isinstance(binding_digest, str):
        raise ValueError("science binding is malformed")
    contract.validate_frozen_identity(current_binding, binding_value, expected_digest=binding_digest, label="science binding")
    for endpoint in SCIENCE_ENDPOINTS:
        item = next(row for row in binding_value["endpoint_rows"] if row["label"] == endpoint[0])
        _require_file_hash(ROOT / item["checkpoint"], item["checkpoint_sha256"], f"checkpoint {endpoint[0]}")
    baseline_data: dict[str, dict[str, dict[str, Any]]] = {}
    for endpoint in SCIENCE_ENDPOINTS:
        label = endpoint[0]
        baseline_data[label] = _validate_baseline_endpoint(_baseline_path(label), endpoint, programs)
    if manifest.get("budget") != contract.SCIENCE_BUDGET:
        raise ValueError("science budget binding changed")
    return manifest, manifest_digest, programs, baseline_data


def _carried_local_observations(
    program: Sequence[str],
    initial_state: Sequence[int],
    predicted_trace: Sequence[Sequence[int]],
    target_trace: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    initial = _state(initial_state, label="initial state")
    if len(program) != len(predicted_trace) or len(program) != len(target_trace):
        raise ValueError("carried/local trace lengths changed")
    observations = []
    for position, opcode in enumerate(program):
        predicted_prestate = initial if position == 0 else _state(predicted_trace[position - 1], label="predicted prestate")
        true_prestate = initial if position == 0 else _state(target_trace[position - 1], label="true prestate")
        predicted_output = _state(predicted_trace[position], label="predicted output")
        true_output = _state(target_trace[position], label="target output")
        observations.append({
            "position": position + 1,
            "input_matches_true_prestate": predicted_prestate == true_prestate,
            "local_step_correct": predicted_output == contract.dsl_step(opcode, predicted_prestate),
            "output_matches_true_poststate": predicted_output == true_output,
        })
    return observations


def run_science() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for explicit-register science")
    manifest, manifest_digest, programs, baseline_data = _validate_science_manifest()
    output = SCIENCE_OUTPUT
    _refuse_nonempty(output)
    started = time.monotonic()
    settings = manifest["runtime"]["settings"]
    counter = _new_counter()

    def sink(value: Mapping[str, Any]) -> None:
        _write_json(output / "accounting.json", value)
        _write_json(output / "progress.json", {
            "schema": "pc_explicit_registers_progress_v2",
            "status": "running",
            "manifest_sha256": manifest_digest,
            "accounting": value,
        })

    _write_json(output / "status.json", {
        "schema": "pc_explicit_registers_status_v1",
        "status": "running",
        "manifest_sha256": manifest_digest,
        "accounting": counter,
    })
    records = []
    lineage = migration.read_json(LINEAGE)
    try:
        with _count_torch_loads(counter, sink), migration.windows_compatibility_adapter():
            for endpoint_index, endpoint in enumerate(SCIENCE_ENDPOINTS, 1):
                load_started = time.monotonic()
                model, checkpoint, payload = _load_counter_model(endpoint, lineage, counter, sink)
                load_seconds = time.monotonic() - load_started
                label = endpoint[0]
                table_path = output / "endpoints" / f"{label.replace('/', '_')}.json"

                def table_sink(value: Mapping[str, Any], path: Path = table_path) -> None:
                    _write_json(path, value)

                record = _endpoint_record(
                    endpoint, model, checkpoint, payload, contract.STATES, programs, baseline_data[label],
                    torch.device("cuda"), counter, sink, table_sink,
                )
                record["timing"]["load_seconds"] = load_seconds
                records.append(record)
                _write_json(table_path, record)
                _write_json(output / "status.json", {
                    "schema": "pc_explicit_registers_status_v1",
                    "status": "running",
                    "manifest_sha256": manifest_digest,
                    "endpoint_index": endpoint_index,
                    "endpoint_count": len(SCIENCE_ENDPOINTS),
                    "accounting": counter,
                })
    except BaseException as exc:
        _failure(counter, "science", exc)
        sink(counter)
        _write_json(output / "status.json", {
            "schema": "pc_explicit_registers_status_v1",
            "status": "failed",
            "manifest_sha256": manifest_digest,
            "accounting": counter,
        })
        raise
    expected_forwards = len(SCIENCE_ENDPOINTS) * len(contract.OPS)
    expected_cases = expected_forwards * len(contract.STATES)
    if (counter["attempted_forwards"], counter["completed_forwards"], counter["attempted_cases"],
            counter["completed_cases"], counter["attempted_native_steps"], counter["completed_native_steps"]) != (
                expected_forwards, expected_forwards, expected_cases, expected_cases,
                contract.SCIENCE_NATIVE_STEPS, contract.SCIENCE_NATIVE_STEPS):
        raise ValueError("science forward accounting mismatch")
    timing = {
        "load_seconds": sum(record["timing"]["load_seconds"] for record in records),
        "table_construction_seconds": sum(record["timing"]["table_construction_seconds"] for record in records),
        "lookup_composition_seconds": sum(record["timing"]["lookup_composition_seconds"] for record in records),
        "note": "table construction includes decoded CPU transfer; lookup/composition is not uncached sequential-neural speed",
    }
    report = {
        "schema": "pc_explicit_registers_science_report_v1",
        "mode": "science",
        "status": "complete",
        "manifest_sha256": manifest_digest,
        "runtime": manifest["runtime"],
        "budget": contract.SCIENCE_BUDGET,
        "accounting": counter,
        "endpoint_count": len(records),
        "program_count": len(programs),
        "timing": timing,
        "endpoints": [{key: value for key, value in record.items() if key not in {"tables", "programs"}} for record in records],
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output / "accounting.json", counter)
    _write_json(output / "report.json", report)
    _write_json(output / "status.json", {
        "schema": "pc_explicit_registers_status_v1",
        "status": "complete",
        "manifest_sha256": manifest_digest,
        "accounting": counter,
    })
    return report


def run_qa() -> dict[str, Any]:
    output = QA_OUTPUT
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for explicit-register QA")
    _refuse_nonempty(output)
    started = time.monotonic()
    settings = _runtime_settings()
    endpoint = next(item for item in migration.ENDPOINTS if item[0] == QA_LABEL)
    binding = _program_binding("qa", (endpoint,), QA_STATES, (QA_PROGRAM,))
    binding_digest = contract.canonical_digest(binding)
    contract.validate_frozen_identity(binding, binding, expected_digest=binding_digest, label="QA binding")
    source = _source_inventory()
    counter = _new_counter()
    manifest = {
        "schema": "pc_explicit_registers_runtime_manifest_v1",
        "mode": "qa",
        "status": "running",
        "source": source,
        "runtime": settings,
        "binding": {"value": binding, "sha256": binding_digest},
        "budget": contract.QA_BUDGET_MAX,
    }
    _write_json(output / "manifest.json", manifest)
    lineage = migration.read_json(LINEAGE)
    try:
        with _count_torch_loads(counter), migration.windows_compatibility_adapter():
            model, checkpoint, payload = _load_counter_model(endpoint, lineage, counter)
            model = model.to(torch.device("cuda"))
            model.eval()
            before = _model_identity(model)
            first_inputs = [list(state) for state in QA_STATES]
            first_outputs = _one_op_forward(model, first_inputs, "ADD", torch.device("cuda"), counter)
            second_inputs = [list(pair) for pair in first_outputs]
            second_outputs = _one_op_forward(model, second_inputs, "XOR", torch.device("cuda"), counter)
            cases = []
            for initial, add_output, xor_output in zip(QA_STATES, first_outputs, second_outputs):
                target = contract.dsl_trace(QA_PROGRAM, initial)
                predicted = [add_output, xor_output]
                diagnostic = contract.trace_diagnostic(predicted, target)
                cases.append({
                    "initial_state": list(initial),
                    "first_call_input": list(initial),
                    "first_call_output": add_output,
                    "second_call_input_own_prediction": list(add_output),
                    "second_call_output": xor_output,
                    "target_trace": target,
                    "predicted_trace": predicted,
                    "diagnostic": diagnostic,
                })
            after = _model_identity(model)
            _assert_model_preserved(before, after, "QA")
    except BaseException:
        manifest["status"] = "failed"
        manifest["accounting"] = counter
        _write_json(output / "manifest.json", manifest)
        raise
    if counter["attempted_forwards"] > contract.QA_MAX_FORWARDS or counter["completed_forwards"] > contract.QA_MAX_FORWARDS:
        raise ValueError("QA forward budget exceeded")
    if counter["attempted_cases"] > contract.QA_CASES or counter["completed_cases"] > contract.QA_CASES:
        raise ValueError("QA case budget exceeded")
    if counter["attempted_native_steps"] > contract.QA_NATIVE_STEPS or counter["completed_native_steps"] > contract.QA_NATIVE_STEPS:
        raise ValueError("QA native-step budget exceeded")
    record = {
        "label": endpoint[0],
        "checkpoint": migration.posix_relative(checkpoint),
        "checkpoint_sha256": migration.sha256_file(checkpoint),
        "expected_checkpoint_sha256": migration.manifest_hash_for_checkpoint(endpoint[0]),
        "model_digest": payload.get("model_digest"),
        "model_identity_before": before,
        "model_identity_after": after,
        "program": list(QA_PROGRAM),
        "cases": cases,
    }
    report = {
        "schema": "pc_explicit_registers_runtime_report_v1",
        "mode": "qa",
        "status": "complete",
        "runtime": settings,
        "budget": contract.QA_BUDGET_MAX,
        "accounting": counter,
        "endpoint": record,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output / "endpoint.json", record)
    _write_json(output / "accounting.json", counter)
    _write_json(output / "report.json", report)
    manifest["status"] = "complete"
    manifest["accounting"] = counter
    _write_json(output / "manifest.json", manifest)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qa", action="store_true", help="run the bounded two-call QA")
    modes.add_argument("--prepare-science", action="store_true", help="freeze the immutable science manifest without model work")
    modes.add_argument("--science", action="store_true", help="run the registered six-endpoint science tables")
    args = parser.parse_args(argv)
    result = run_qa() if args.qa else prepare_science() if args.prepare_science else run_science()
    summary = {"status": result["status"]}
    if "mode" in result:
        summary["mode"] = result["mode"]
    if "manifest" in result:
        summary["manifest"] = result["manifest"]
        summary["manifest_sha256"] = result["manifest_sha256"]
    if "accounting" in result:
        summary["accounting"] = result["accounting"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
