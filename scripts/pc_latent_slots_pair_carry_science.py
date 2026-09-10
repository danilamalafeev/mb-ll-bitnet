"""Bounded two-arm science run for the registered latent pair-carry study.

The entry point performs exactly one strict local-2000 endpoint load followed
by 18 focus-program forwards for each of the sham and pair-carry arms.  It
does not train, prepare a new manifest, rerun QA, or execute any exploratory
forward.  All source, QA, baseline, scope, and endpoint bindings are frozen
before the model load.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_pair_carry as pair
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_latent_slots_pair_carry_v1" / "science"
SCIENCE_MANIFEST = diagnostic_runtime.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic_runtime.SCIENCE_ENDPOINT
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LATENT_SLOTS_PAIR_CARRY_PROTOCOL.md"
QA_ACCEPT = ROOT / "runs" / "pc_latent_slots_pair_carry_v1" / "qa_accept.json"
BASELINE_DIR = ROOT / "runs" / "pc_latent_slots_v1" / "padding_diagnostic_v1"
BASELINE_ROWS = BASELINE_DIR / "program_rows"
BASELINE_REPORT = BASELINE_DIR / "report.json"
BASELINE_INPUT_FREEZE = BASELINE_DIR / "input_freeze.json"
BASELINE_ACCOUNTING = BASELINE_DIR / "accounting.json"
PURE_HELPER = ROOT / "scripts" / "pc_latent_slots_pair_carry.py"
PURE_TESTS = ROOT / "tests" / "test_pc_latent_slots_pair_carry.py"
QA_RUNTIME = ROOT / "scripts" / "pc_latent_slots_pair_carry_runtime.py"
QA_RUNTIME_TESTS = ROOT / "tests" / "test_pc_latent_slots_pair_carry_runtime.py"
SCIENCE_RUNTIME = Path(__file__).resolve()
SCIENCE_TESTS = ROOT / "tests" / "test_pc_latent_slots_pair_carry_science.py"

SCHEMA = "pc_latent_slots_pair_carry_science_v1"
SOURCE_SCHEMA = "pc_latent_slots_pair_carry_science_source_binding_v1"
INPUT_SCHEMA = "pc_latent_slots_pair_carry_science_input_freeze_v1"
PROGRAM_SCHEMA = "pc_latent_slots_pair_carry_science_program_v1"
QA_ACCEPT_SHA256 = "4952893a0e164d3f84c1fa3f120ad5c70bf38ae7cc08846dd7a32bc7b7eb e9d3".replace(" ", "")
BASELINE_BINDING_DIGEST = "d5dab546ced033f46704a629e2c67c56e405737a2c60754c7db5b128b8f1d958"

FOCUS_PROGRAMS = 18
EVALUATION_STATES = 256
SCIENCE_ARMS = (pair.SHAM_ARM, pair.PAIR_CARRY_ARM)
SCIENCE_CALLS = 36
SCIENCE_CASES = 9_216
SCIENCE_POSITIONS = 258_048
SCIENCE_NATIVE_STEPS = 2_064_384
SCIENCE_UPDATES = 0
SCIENCE_DESERIALIZATIONS = 3


def _resolve(path: Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    return diagnostic_runtime._sha256(path)


def _atomic_json(path: Path, value: Any, *, refuse: bool = False) -> None:
    diagnostic_runtime._atomic_json(path, value, refuse=refuse)


def _refuse_fresh(path: Path) -> None:
    diagnostic_runtime._refuse_fresh(path)


def _new_counter() -> dict[str, Any]:
    return diagnostic_runtime._new_counter()


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    diagnostic_runtime._flush(counter, sink)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    diagnostic_runtime._failure(counter, kind, exc, phase=phase)


def _file_binding(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _path_from_saved(root: Path, value: str) -> Path:
    return _resolve(Path(value.replace("\\", "/")), root)


def _validate_qa_accept(root: Path = ROOT) -> dict[str, Any]:
    """Validate the immutable QA acceptance record and every listed artifact."""

    path = _resolve(QA_ACCEPT, root)
    if not path.is_file() or _sha256(path) != QA_ACCEPT_SHA256:
        raise ValueError("pair-carry QA acceptance bytes changed")
    accepted_record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(accepted_record, Mapping) or accepted_record.get("status") != "QA_ACCEPT":
        raise ValueError("pair-carry QA acceptance record is malformed")
    files: dict[str, dict[str, str]] = {"qa_accept": _file_binding(path, root)}
    for field in ("files", "source_files"):
        values = accepted_record.get(field)
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"pair-carry QA acceptance {field} are missing")
        for raw_name, expected in values.items():
            if not isinstance(raw_name, str) or not isinstance(expected, str):
                raise ValueError("pair-carry QA acceptance file binding is malformed")
            artifact = _path_from_saved(root, raw_name)
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ValueError(f"pair-carry QA accepted artifact changed: {raw_name}")
            files[f"{field}:{raw_name}"] = {"path": _relative(artifact, root), "sha256": expected}
    return {"record": dict(accepted_record), "files": files, "sha256": QA_ACCEPT_SHA256}


def _focus_specs(scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact 18 focus rows and freeze their pair schedules."""

    rows = diagnostic_runtime._focus_scope(scope)
    if len(rows) != FOCUS_PROGRAMS:
        raise ValueError("pair-carry focus program count changed")
    expected_ids = tuple(identifier for identifier, _program, _roles in diagnostic_runtime.diagnostic.FOCUS_PROGRAM_MAP)
    if tuple(row.get("id") for row in rows) != expected_ids:
        raise ValueError("pair-carry focus program IDs/order changed")
    result: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row["id"])
        program = tuple(str(opcode) for opcode in row["program"])
        expected_pairs = pair.expected_pair_schedule(program)
        pairs = pair.validate_pair_schedule(program, expected_pairs)
        states = row.get("states")
        if not isinstance(states, list) or len(states) != EVALUATION_STATES:
            raise ValueError(f"pair-carry state count changed: {identifier}")
        normalized_states = [tuple(int(value) for value in state_row.get("state", ())) for state_row in states]
        if tuple(normalized_states) != tuple(dsl.STATE_ORDER):
            raise ValueError(f"pair-carry state order changed: {identifier}")
        strata = [str(state_row.get("stratum", "")) for state_row in states]
        expected_strata = [diagnostic_runtime.diagnostic._state_stratum(state) for state in dsl.STATE_ORDER]
        if strata != expected_strata:
            raise ValueError(f"pair-carry state strata changed: {identifier}")
        result.append({"id": identifier, "program": program, "pairs": pairs, "states": tuple(dsl.STATE_ORDER), "strata": tuple(strata), "suite": "padding"})
    return result


def _validate_program_ops(program: Sequence[str], ops: torch.Tensor) -> tuple[str, ...]:
    """Require every batch row to use the accepted DSL opcode IDs exactly."""

    expected_program = tuple(program)
    expected = tuple(dsl.OP_TO_ID[opcode] for opcode in expected_program)
    if ops.ndim != 2 or tuple(ops.shape) != (EVALUATION_STATES, len(expected_program)) or ops.dtype is not torch.long:
        raise ValueError("science program-to-ops tensor shape or dtype changed")
    if ops.detach().cpu().tolist() != [list(expected) for _ in range(EVALUATION_STATES)]:
        raise ValueError("science program-to-ops binding changed")
    return expected_program


def _build_inputs(spec: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    program = tuple(str(opcode) for opcode in spec["program"])
    examples = [dsl.RegisterExample(state[0], state[1], program) for state in dsl.STATE_ORDER]
    ids_x = torch.tensor([example.x for example in examples], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in examples], dtype=torch.long, device=device)
    bits = accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor(
        [[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in examples],
        dtype=torch.long,
        device=device,
    )
    _validate_program_ops(program, ops)
    return {"examples": examples, "x_bits": x_bits, "y_bits": y_bits, "ops": ops, "program": program}


def _validate_baseline(root: Path, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate saved accepted detailed focus traces and return their predictions."""

    row_paths = sorted(_resolve(BASELINE_ROWS, root).glob("*.json"), key=lambda path: path.name)
    if len(row_paths) != diagnostic_runtime.EVALUATION_PROGRAMS:
        raise ValueError("saved baseline program-row count changed")
    files: dict[str, dict[str, str]] = {}
    for path in (_resolve(BASELINE_REPORT, root), _resolve(BASELINE_INPUT_FREEZE, root), _resolve(BASELINE_ACCOUNTING, root)):
        files[_relative(path, root)] = _file_binding(path, root)
    expected_by_id = {str(spec["id"]): spec for spec in specs}
    baseline_predictions: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for path in row_paths:
        relative = _relative(path, root)
        files[relative] = _file_binding(path, root)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("saved baseline row is malformed")
        identifier = str(raw.get("id", ""))
        if identifier not in expected_by_id:
            continue
        if identifier in seen_ids:
            raise ValueError("saved baseline focus row is duplicated")
        spec = expected_by_id[identifier]
        if raw.get("suite") != "padding" or raw.get("program") != list(spec["program"]) or raw.get("length") != len(spec["program"]) or raw.get("detailed") is not True:
            raise ValueError(f"saved baseline focus identity changed: {identifier}")
        predictions = raw.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != EVALUATION_STATES:
            raise ValueError(f"saved baseline focus state count changed: {identifier}")
        normalized: list[dict[str, Any]] = []
        for state, expected_stratum, example, prediction in zip(dsl.STATE_ORDER, spec["strata"], [dsl.RegisterExample(state[0], state[1], tuple(spec["program"])) for state in dsl.STATE_ORDER], predictions):
            if not isinstance(prediction, Mapping) or tuple(int(value) for value in prediction.get("state", ())) != state or prediction.get("stratum") != expected_stratum:
                raise ValueError(f"saved baseline state/stratum join changed: {identifier}")
            target = [list(item) for item in example.targets]
            if prediction.get("target_trace") != target:
                raise ValueError(f"saved baseline target trace changed: {identifier}")
            predicted = prediction.get("predicted_trace")
            if not isinstance(predicted, list) or len(predicted) != len(target) or any(not isinstance(item, list) or len(item) != 2 or any(type(value) is not int or value < 0 or value >= 16 for value in item) for item in predicted):
                raise ValueError(f"saved baseline decoded trace changed: {identifier}")
            normalized.append({"state": list(state), "stratum": expected_stratum, "target_trace": target, "predicted_trace": predicted})
        baseline_predictions[identifier] = normalized
        seen_ids.add(identifier)
    if seen_ids != set(expected_by_id):
        raise ValueError("saved baseline focus coverage changed")
    binding_payload = {"files": files, "focus_ids": sorted(seen_ids)}
    binding_digest = science._digest(binding_payload)
    if BASELINE_BINDING_DIGEST == "__TO_BE_FILLED__" or binding_digest != BASELINE_BINDING_DIGEST:
        raise ValueError("saved baseline binding digest changed")
    return {"files": files, "digest": binding_digest, "predictions": baseline_predictions}


def _source_binding(*, root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], qa_accept: Mapping[str, Any], baseline: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "science_runtime": SCIENCE_RUNTIME,
        "science_tests": _resolve(SCIENCE_TESTS, root),
        "pair_carry_helper": _resolve(PURE_HELPER, root),
        "pair_carry_tests": _resolve(PURE_TESTS, root),
        "qa_runtime": _resolve(QA_RUNTIME, root),
        "qa_runtime_tests": _resolve(QA_RUNTIME_TESTS, root),
        "diagnostic_runtime": Path(diagnostic_runtime.__file__).resolve(),
        "protocol": _resolve(PROTOCOL, root),
        "register_e15": _resolve(Path("looped_bitnet/register_e15.py"), root),
    }
    files = {name: _file_binding(path, root) for name, path in paths.items()}
    files.update({f"qa:{name}": dict(value) for name, value in qa_accept["files"].items()})
    files.update({f"baseline:{name}": dict(value) for name, value in baseline["files"].items()})
    for name, value in evidence.items():
        if isinstance(value, Mapping) and "path" in value and "sha256" in value:
            files[f"evidence:{name}"] = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("accepted science source inventory is missing")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "accepted_science_source_inventory": inventory,
        "qa_accept_sha256": qa_accept["sha256"],
        "qa_accept_record": qa_accept["record"],
        "baseline_binding_digest": baseline["digest"],
        "evidence": dict(evidence),
        "runtime_settings": dict(settings),
        "endpoint_local_update": diagnostic_runtime.ENDPOINT_LOCAL_UPDATE,
        "endpoint_sha256": diagnostic_runtime.ENDPOINT_SHA256,
    }
    binding["digest"] = science._digest(binding)
    return binding


def _input_freeze(*, specs: Sequence[Mapping[str, Any]], source_binding: Mapping[str, Any], qa_accept: Mapping[str, Any], baseline: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any], manifest_path: Path, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "immutable": True,
        "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
        "programs": [{"id": spec["id"], "program": list(spec["program"]), "pairs": [list(item) for item in spec["pairs"]], "states": [list(state) for state in spec["states"]], "strata": list(spec["strata"])} for spec in specs],
        "arms": list(SCIENCE_ARMS),
        "budget": dict(pair.SCIENCE_ACCOUNTING),
        "source_binding_digest": source_binding["digest"],
        "qa_accept_sha256": qa_accept["sha256"],
        "baseline_binding_digest": baseline["digest"],
        "evidence": dict(evidence),
        "runtime": dict(settings),
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
        "attempted_endpoint_loads": 1, "completed_endpoint_loads": 1,
        "attempted_parent_loads": 1, "completed_parent_loads": 1,
        "attempted_endpoint_restores": 1, "completed_endpoint_restores": 1,
        "attempted_underlying_deserializations": SCIENCE_DESERIALIZATIONS,
        "completed_underlying_deserializations": SCIENCE_DESERIALIZATIONS,
        "attempted_forwards": SCIENCE_CALLS, "completed_forwards": SCIENCE_CALLS,
        "attempted_cases": SCIENCE_CASES, "completed_cases": SCIENCE_CASES,
        "attempted_readout_positions": SCIENCE_POSITIONS, "completed_readout_positions": SCIENCE_POSITIONS,
        "attempted_native_steps": SCIENCE_NATIVE_STEPS, "completed_native_steps": SCIENCE_NATIVE_STEPS,
        "attempted_updates": SCIENCE_UPDATES, "completed_updates": SCIENCE_UPDATES,
        "attempted_backwards": SCIENCE_UPDATES, "completed_backwards": SCIENCE_UPDATES,
        "attempted_optimizer_steps": SCIENCE_UPDATES, "completed_optimizer_steps": SCIENCE_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"pair-carry science accounting mismatch: {key}={counter.get(key)} expected {value}")


def _validate_logits(logits: Any, *, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, tuple) or len(logits) != 2 or not all(isinstance(item, torch.Tensor) for item in logits):
        raise ValueError("science forward returned malformed logits")
    logits_x, logits_y = logits
    expected = (EVALUATION_STATES, length, latent.SLOT_WIDTH)
    if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
        raise ValueError("science logits are malformed or nonfinite")
    return logits_x, logits_y


def _trace_prediction(target: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> dict[str, Any]:
    metrics = diagnostic_runtime.accepted_runtime._trace_metrics(target, predicted)
    recovery = None
    if metrics["first_error"] is not None:
        for position in range(int(metrics["first_error"]), len(target)):
            if tuple(target[position]) == tuple(predicted[position]):
                recovery = position + 1
                break
    metrics["first_subsequent_recovery"] = recovery
    return metrics


def _compact_displacements(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = diagnostics.get("pair_carry")
    if not isinstance(payload, Mapping):
        raise ValueError("pair-carry diagnostics are missing")
    rows = payload.get("pairs")
    if not isinstance(rows, list):
        raise ValueError("pair-carry displacement rows are missing")
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("l2_by_case"), list) or not row["l2_by_case"]:
            raise ValueError("pair-carry displacement row is malformed")
        values = torch.tensor(row["l2_by_case"], dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("pair-carry displacement is nonfinite")
        compact.append({"start": int(row["start"]), "end": int(row["end"]), "mean": float(values.mean().item()), "max": float(values.max().item())})
    return compact


def _run_program(*, arm: str, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, spec: Mapping[str, Any], baseline_predictions: Mapping[str, list[dict[str, Any]]], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    identifier = str(spec["id"])
    inputs = _build_inputs(spec, device)
    program = inputs["program"]
    length = len(program)
    _account_attempt(counter, length, sink)
    try:
        with diagnostic_runtime._timed(counter, f"forward_{arm}_{identifier}", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            _validate_program_ops(program, inputs["ops"])
            logits, diagnostics = pair.pair_carry_forward(
                model, adapter, inputs["x_bits"], inputs["y_bits"], inputs["ops"],
                program=program, pairs=spec["pairs"], arm=arm, return_diagnostics=True,
            )
        logits_x, logits_y = _validate_logits(logits, length=length)
        decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
        decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
        predictions: list[dict[str, Any]] = []
        for index, (state, stratum) in enumerate(zip(dsl.STATE_ORDER, spec["strata"])):
            example = inputs["examples"][index]
            target = [list(item) for item in example.targets]
            predicted = [[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)]
            predictions.append({"state": list(state), "stratum": stratum, "target_trace": target, "predicted_trace": predicted, **_trace_prediction(target, predicted)})
        if arm == pair.SHAM_ARM:
            accepted_predictions = baseline_predictions.get(identifier)
            if accepted_predictions is None or [row["predicted_trace"] for row in predictions] != [row["predicted_trace"] for row in accepted_predictions]:
                raise ValueError(f"sham decoded traces do not exactly match saved baseline: {identifier}")
        pair_payload = diagnostics.get("pair_carry") if isinstance(diagnostics, Mapping) else None
        if not isinstance(pair_payload, Mapping) or pair_payload.get("reader_calls") != length or pair_payload.get("writer_calls") != length:
            raise ValueError(f"pair-carry hook call count changed: {identifier}")
        if len(adapter.reader._forward_pre_hooks) != 0 or len(adapter.writer._forward_hooks) != 0:
            raise ValueError(f"pair-carry hooks remained installed: {identifier}")
        _account_complete(counter, length, sink)
        return {"schema": PROGRAM_SCHEMA, "id": identifier, "suite": "padding", "length": length, "program": list(program), "arm": arm, "predictions": predictions, "pair_displacement_l2": _compact_displacements(diagnostics)}
    except BaseException as exc:
        _failure(counter, "forward", exc, phase="evaluation")
        _flush(counter, sink)
        raise


def run_science(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = diagnostic_runtime._sink(accounting_path)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "manifest": {"path": _relative(manifest_path, root), "sha256": None}, "budget": dict(pair.SCIENCE_ACCOUNTING), "accounting": dict(counter)}
    model: torch.nn.Module | None = None
    adapter: latent.LatentSlotAdapter | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        manifest, science_report, scope, _final, evidence = diagnostic_runtime._validate_evidence(root, manifest_path)
        device, settings = diagnostic_runtime._load_runtime_settings(manifest)
        qa_accept = _validate_qa_accept(root)
        specs = _focus_specs(scope)
        baseline = _validate_baseline(root, specs)
        source_binding = _source_binding(root=root, manifest=manifest, evidence=evidence, qa_accept=qa_accept, baseline=baseline, settings=settings)
        input_freeze = _input_freeze(specs=specs, source_binding=source_binding, qa_accept=qa_accept, baseline=baseline, evidence=evidence, settings=settings, manifest_path=manifest_path, root=root)
        _atomic_json(out / "source_binding.json", source_binding, refuse=True)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report.update({"manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]}, "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": input_freeze["digest"]}, "runtime": settings, "endpoint": evidence["endpoint"], "qa_accept": {"path": _relative(QA_ACCEPT, root), "sha256": qa_accept["sha256"]}, "baseline": {"digest": baseline["digest"], "focus_programs": FOCUS_PROGRAMS}, "science_report": science_report.get("manifest")})
        model, adapter, optimizer, _endpoint_payload = diagnostic_runtime._load_endpoint(manifest=manifest, report=science_report, endpoint=_resolve(SCIENCE_ENDPOINT, root), manifest_path=manifest_path, source_binding_digest=str(manifest["source_inventory"]["digest"]), device=device, counter=counter, sink=sink, root=root)
        if counter["completed_endpoint_loads"] != 1 or counter["completed_underlying_deserializations"] != SCIENCE_DESERIALIZATIONS:
            raise ValueError("science endpoint load did not complete at the registered boundary")
        rows_dir = out / "program_rows"
        rows_dir.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
            for arm in SCIENCE_ARMS:
                for spec in specs:
                    row = _run_program(arm=arm, model=model, adapter=adapter, spec=spec, baseline_predictions=baseline["predictions"], device=device, counter=counter, sink=sink)
                    _atomic_json(rows_dir / f"{len(rows):03d}_{arm}_{row['id']}.json", row, refuse=True)
                    rows.append(row)
        if len(rows) != SCIENCE_CALLS:
            raise ValueError("science output row count changed")
        _validate_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "rows": {"count": len(rows), "directory": _relative(rows_dir, root)}, "aggregates": science._aggregate_rows(rows), "limitations": ["single saved latent local2000 endpoint", "external program-aware pair-carry intervention", "pair displacement is descriptive and does not identify a unique mechanism"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        phase = "evaluation" if model is not None else "load"
        _failure(counter, "science", exc, phase=phase)
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise
    finally:
        model = None
        adapter = None
        optimizer = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_science(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
