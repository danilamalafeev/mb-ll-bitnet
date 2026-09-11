"""Fixed two-arm gated-carry science run.

This module is intentionally a small orchestration layer around the accepted
latent loader and numerical forward.  ``--science`` is the only entry point;
it freezes all inputs before either endpoint load, trains each arm for the
registered 1000-update prefix, and writes compact decoded rows/checkpoints.
There is no resume, retry, probe, or checkpoint reload path.
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
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_gated_carry_runtime as qa_runtime
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_latent_slots_runtime as accepted_qa_runtime
from scripts import pc_learned_scratchpad as accepted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_gated_carry_v1" / "science"
SCIENCE_MANIFEST = diagnostic_runtime.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic_runtime.SCIENCE_ENDPOINT
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_GATED_CARRY_PROTOCOL.md"
QA_ACCEPT = ROOT / "runs" / "pc_gated_carry_v1" / "qa_accept.json"
SCIENCE_RUNTIME = Path(__file__).resolve()
SCIENCE_TESTS = ROOT / "tests" / "test_pc_gated_carry_science.py"
PURE_HELPER = ROOT / "scripts" / "pc_gated_carry.py"
PURE_TESTS = ROOT / "tests" / "test_pc_gated_carry.py"
QA_RUNTIME = ROOT / "scripts" / "pc_gated_carry_runtime.py"
QA_RUNTIME_TESTS = ROOT / "tests" / "test_pc_gated_carry_runtime.py"
BASELINE_RUNTIME = Path(diagnostic_runtime.__file__).resolve()

SCHEMA = "pc_gated_carry_science_v1"
SOURCE_SCHEMA = "pc_gated_carry_science_source_binding_v1"
INPUT_SCHEMA = "pc_gated_carry_science_input_freeze_v1"
CHECKPOINT_SCHEMA = "pc_gated_carry_science_checkpoint_v1"
PROGRAM_SCHEMA = "pc_gated_carry_science_program_v1"
ARM_A = "A"
ARM_B = "B"
ARMS = (ARM_A, ARM_B)
ARM_LABELS = {ARM_A: "unhooked", ARM_B: "gated"}
TRAINING_UPDATES = 1000
TRAINING_BATCH_SIZE = 64
CHECKPOINT_INDICES = (0, 250, 500, 750, 1000)
ENDPOINT_ABSOLUTE_UPDATE = 42000
OLD_PROGRAMS = 69
NEW_PROGRAMS = gated.CONTROL_PROGRAM_COUNT
OLD_CALLS = 3 * OLD_PROGRAMS
NEW_CALLS = 4 * NEW_PROGRAMS
EVALUATION_CALLS = OLD_CALLS + NEW_CALLS
SCIENCE_CALLS = 2531
SCIENCE_CASES = 263936
SCIENCE_POSITIONS = 3239936
SCIENCE_NATIVE_STEPS = 25919488
SCIENCE_UPDATES = 2000
SCIENCE_DESERIALIZATIONS = 6
CHECKPOINT_COUNT = len(CHECKPOINT_INDICES) * len(ARMS)
TRAINING_POSITIONS_PER_ARM = gated.TRAINING_POSITIONS_PER_ARM
TRAINING_NATIVE_STEPS_PER_ARM = gated.TRAINING_NATIVE_STEPS_PER_ARM
EVALUATION_CASES = 135936
EVALUATION_POSITIONS = 2792448
QA_ACCEPT_SHA256 = "e63bbda44694e66377d461be7170d99818c5b11cb5879ae10ad83ae7bd19f94b"
PURE_HELPER_SHA256 = "4a35cb641d087e98531f3df50a76b478395b06ebe75102592a2d7f0eaa9b677d"
PURE_TESTS_SHA256 = "7de5c0027721a5c78e7ae88aa7c4b90e605ef24404503b100413e8485d389962"
QA_RUNTIME_SHA256 = "a751d57fb52d65e02bb2732c27d3ab72928a5350dfd35c7a7bc93f114f120972"
QA_RUNTIME_TESTS_SHA256 = "7445a8e3df7a786b6af415ba8d322992ef5bddd0e3171e2cab88a864f34bd79d"


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


def _file_binding(path: Path, root: Path = ROOT) -> dict[str, str]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _path_from_saved(root: Path, value: str) -> Path:
    return _resolve(Path(value.replace("\\", "/")), root)


def _new_counter() -> dict[str, Any]:
    counter = qa_runtime._new_counter()
    counter["schema"] = "pc_gated_carry_science_accounting_v1"
    counter["active_phase"] = "load"
    counter["attempted_checkpoints"] = 0
    counter["completed_checkpoints"] = 0
    phase_template = {
        "attempted_forwards": 0, "completed_forwards": 0,
        "attempted_cases": 0, "completed_cases": 0,
        "attempted_readout_positions": 0, "completed_readout_positions": 0,
        "attempted_native_steps": 0, "completed_native_steps": 0,
        "failures": 0,
    }
    counter["phase_counts"] = {
        "load": dict(counter["phase_counts"]["load"]),
        "training": {**phase_template, "attempted_updates": 0, "completed_updates": 0,
                      "attempted_backwards": 0, "completed_backwards": 0,
                      "attempted_optimizer_steps": 0, "completed_optimizer_steps": 0},
        "evaluation_initial": dict(phase_template),
        "evaluation_final": dict(phase_template),
        "checkpoint_save": {"attempted": 0, "completed": 0, "failures": 0},
    }
    return counter


def _phase_add(counter: dict[str, Any], phase: str, key: str, amount: int = 1) -> None:
    counter[key] += amount
    counter["phase_counts"][phase][key] = counter["phase_counts"][phase].get(key, 0) + amount


def _phase_failure(counter: dict[str, Any], phase: str, kind: str, exc: BaseException, sink: Callable[[dict[str, Any]], None] | None) -> None:
    qa_runtime._failure(counter, kind, exc, phase=phase)
    qa_runtime._flush(counter, sink)


def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None):
    return qa_runtime._timed(counter, name, sink)


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty science output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_qa_accept(root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(QA_ACCEPT, root)
    if not path.is_file() or _sha256(path) != QA_ACCEPT_SHA256:
        raise ValueError("gated QA acceptance bytes changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or record.get("status") != "QA_ACCEPT":
        raise ValueError("gated QA acceptance record is malformed")
    if record.get("qa_origin_absolute_update") != ENDPOINT_ABSOLUTE_UPDATE:
        raise ValueError("gated QA origin lineage changed")
    files: dict[str, dict[str, str]] = {"qa_accept": _file_binding(path, root)}
    for field in ("files", "source_files"):
        values = record.get(field)
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"gated QA acceptance {field} are missing")
        for raw_name, expected in values.items():
            if not isinstance(raw_name, str) or not isinstance(expected, str):
                raise ValueError("gated QA acceptance binding is malformed")
            artifact = _path_from_saved(root, raw_name)
            if not artifact.is_file() or _sha256(artifact) != expected.lower():
                raise ValueError(f"gated QA artifact changed: {raw_name}")
            files[f"{field}:{raw_name}"] = {"path": _relative(artifact, root), "sha256": expected.lower()}
    return {"record": dict(record), "files": files, "sha256": QA_ACCEPT_SHA256}


def _old_specs(scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    programs = scope.get("programs")
    if not isinstance(programs, list) or len(programs) != OLD_PROGRAMS:
        raise ValueError("old evaluation scope program count changed")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in programs:
        if not isinstance(raw, Mapping):
            raise ValueError("old evaluation scope row is malformed")
        identifier = str(raw.get("id", ""))
        program = tuple(str(op) for op in raw.get("program", ()))
        if not identifier or identifier in seen or not program or any(op not in dsl.OPS for op in program):
            raise ValueError("old evaluation scope identity changed")
        states = raw.get("states")
        if not isinstance(states, list) or len(states) != len(dsl.STATE_ORDER):
            raise ValueError(f"old scope state count changed: {identifier}")
        strata: list[str] = []
        for expected, state_row in zip(dsl.STATE_ORDER, states):
            if not isinstance(state_row, Mapping) or tuple(state_row.get("state", ())) != expected:
                raise ValueError(f"old scope state order changed: {identifier}")
            stratum = str(state_row.get("stratum", ""))
            if stratum != diagnostic_runtime.diagnostic._state_stratum(expected):
                raise ValueError(f"old scope state stratum changed: {identifier}")
            example = dsl.RegisterExample(expected[0], expected[1], program)
            if state_row.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError(f"old scope target binding changed: {identifier}")
            strata.append(stratum)
        suite = str(raw.get("suite", ""))
        if suite not in {"padding", "compositions"} or int(raw.get("length", -1)) != len(program):
            raise ValueError(f"old scope suite/length changed: {identifier}")
        result.append({"id": identifier, "suite": suite, "length": len(program), "program": program,
                       "states": tuple(dsl.STATE_ORDER), "strata": tuple(strata)})
        seen.add(identifier)
    if len({row["id"] for row in result}) != OLD_PROGRAMS:
        raise ValueError("old scope IDs are not unique")
    return result


def _new_specs() -> list[dict[str, Any]]:
    raw_specs = gated.validate_control_program_specs()
    if len(raw_specs) != NEW_PROGRAMS:
        raise ValueError("new control program count changed")
    strata = tuple(diagnostic_runtime.diagnostic._state_stratum(state) for state in dsl.STATE_ORDER)
    result: list[dict[str, Any]] = []
    for raw in raw_specs:
        program = tuple(str(op) for op in raw["program"])
        if int(raw["length"]) != len(program) or len(program) not in gated.CONTROL_LENGTHS:
            raise ValueError("new control program length changed")
        result.append({"id": str(raw["id"]), "suite": "identity_controls", "length": len(program),
                       "program": program, "cell": str(raw["cell"]), "prefix": str(raw["prefix"]),
                       "repeat": int(raw["repeat"]), "suffix": str(raw["suffix"]),
                       "states": tuple(dsl.STATE_ORDER), "strata": strata})
    if len({row["id"] for row in result}) != NEW_PROGRAMS:
        raise ValueError("new control IDs are not unique")
    if {length: sum(row["length"] == length for row in result) for length in gated.CONTROL_LENGTHS} != {11: 27, 19: 27, 35: 27}:
        raise ValueError("new control length coverage changed")
    return result


def _validate_program_ops(program: Sequence[str], ops: torch.Tensor) -> tuple[str, ...]:
    expected_program = tuple(str(op) for op in program)
    if not expected_program or any(op not in dsl.OPS for op in expected_program):
        raise ValueError("program contains an unknown opcode")
    expected = [dsl.OP_TO_ID[op] for op in expected_program]
    if not isinstance(ops, torch.Tensor) or ops.ndim != 2 or ops.shape[1] != len(expected) or ops.dtype is not torch.long:
        raise ValueError("program-to-ops tensor shape or dtype changed")
    if ops.detach().cpu().tolist() != [expected for _ in range(ops.shape[0])]:
        raise ValueError("program-to-ops binding changed")
    return expected_program


def _validate_batch_program_ops(programs: Sequence[Sequence[str]], ops: torch.Tensor) -> tuple[tuple[str, ...], ...]:
    """Validate every row's opcode IDs while allowing different programs."""

    expected_programs = tuple(tuple(str(op) for op in program) for program in programs)
    if not expected_programs or any(not program or any(op not in dsl.OPS for op in program) for program in expected_programs):
        raise ValueError("training program contains an unknown opcode")
    if len({len(program) for program in expected_programs}) != 1:
        raise ValueError("training batch must have one sequence length")
    expected_rows = [[dsl.OP_TO_ID[op] for op in program] for program in expected_programs]
    if not isinstance(ops, torch.Tensor) or ops.ndim != 2 or tuple(ops.shape) != (len(expected_rows), len(expected_rows[0])) or ops.dtype is not torch.long or ops.detach().cpu().tolist() != expected_rows:
        raise ValueError("training program-to-ops binding changed")
    return expected_programs


def _build_inputs(spec: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    program = tuple(str(op) for op in spec["program"])
    examples = [dsl.RegisterExample(state[0], state[1], program) for state in dsl.STATE_ORDER]
    bits = accepted.signed_bit_matrix(device=device)
    ids_x = torch.tensor([example.x for example in examples], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in examples], dtype=torch.long, device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[op] for op in program] for _ in examples], dtype=torch.long, device=device)
    _validate_program_ops(program, ops)
    return {"examples": examples, "x_bits": x_bits, "y_bits": y_bits, "ops": ops, "program": program}


def _validate_old_baseline(final: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = final.get("rows")
    if final.get("phase") != "final_latent" or not isinstance(rows, list) or len(rows) != OLD_PROGRAMS:
        raise ValueError("saved final A baseline is not the accepted final evaluation")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, Mapping)}
    if set(by_id) != {str(spec["id"]) for spec in specs}:
        raise ValueError("saved final A baseline IDs changed")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        row = by_id[spec["id"]]
        if row.get("suite") != spec["suite"] or row.get("length") != spec["length"] or row.get("program") != list(spec["program"]):
            raise ValueError(f"saved final A baseline program join changed: {spec['id']}")
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(dsl.STATE_ORDER):
            raise ValueError(f"saved final A baseline state count changed: {spec['id']}")
        result: list[dict[str, Any]] = []
        for state, stratum, prediction in zip(dsl.STATE_ORDER, spec["strata"], predictions):
            example = dsl.RegisterExample(state[0], state[1], spec["program"])
            target = [list(pair) for pair in example.targets]
            if not isinstance(prediction, Mapping) or tuple(prediction.get("state", ())) != state or prediction.get("stratum") != stratum or prediction.get("target_trace") != target:
                raise ValueError(f"saved final A baseline state/target join changed: {spec['id']}")
            predicted = prediction.get("predicted_trace")
            if not isinstance(predicted, list) or len(predicted) != len(target) or any(not isinstance(item, list) or len(item) != 2 or any(type(value) is not int or not 0 <= value < latent.SLOT_WIDTH for value in item) for item in predicted):
                raise ValueError(f"saved final A baseline decoded trace changed: {spec['id']}")
            result.append({"state": list(state), "stratum": stratum, "target_trace": target, "predicted_trace": predicted,
                           **diagnostic_runtime.accepted_runtime._trace_metrics(target, predicted)})
        normalized[spec["id"]] = result
    return {"digest": _digest({"phase": final["phase"], "ids": sorted(normalized), "rows": len(rows)}), "predictions": normalized}


def _stream_prefix_digest(batches: Sequence[Sequence[dsl.RegisterExample]]) -> str:
    if len(batches) < TRAINING_UPDATES or any(len(batch) != TRAINING_BATCH_SIZE for batch in batches[:TRAINING_UPDATES]):
        raise ValueError("science training prefix count/size changed")
    records: list[list[dict[str, Any]]] = []
    lengths: list[int] = []
    for batch in batches[:TRAINING_UPDATES]:
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("science training batch is not homogeneous")
        lengths.append(len(batch[0].program))
        records.append([{"x": int(example.x), "y": int(example.y), "program": list(example.program), "target_trace": [list(pair) for pair in example.targets]} for example in batch])
    if tuple(lengths) != gated.TRAINING_BATCH_LENGTHS:
        raise ValueError("science training length prefix changed")
    return _digest({"batch_count": len(records), "batch_size": TRAINING_BATCH_SIZE, "lengths": lengths, "records": records})


def _source_binding(*, root: Path, manifest: Mapping[str, Any], evidence: Mapping[str, Any], qa_accept: Mapping[str, Any], baseline: Mapping[str, Any], settings: Mapping[str, Any], prefix_digest: str) -> dict[str, Any]:
    paths = {
        "science_runtime": SCIENCE_RUNTIME,
        "science_tests": _resolve(SCIENCE_TESTS, root),
        "gated_helper": _resolve(PURE_HELPER, root),
        "gated_tests": _resolve(PURE_TESTS, root),
        "qa_runtime": _resolve(QA_RUNTIME, root),
        "qa_runtime_tests": _resolve(QA_RUNTIME_TESTS, root),
        "diagnostic_runtime": BASELINE_RUNTIME,
        "accepted_science": _resolve(Path("scripts/pc_latent_slots_science.py"), root),
        "latent_helper": _resolve(Path("scripts/pc_latent_slots.py"), root),
        "accepted_latent_runtime": _resolve(Path("scripts/pc_latent_slots_runtime.py"), root),
        "accepted_scratchpad": _resolve(Path("scripts/pc_learned_scratchpad.py"), root),
        "accepted_scratchpad_runtime": _resolve(Path("scripts/pc_learned_scratchpad_runtime.py"), root),
        "register_e15": _resolve(Path("looped_bitnet/register_e15.py"), root),
        "protocol": _resolve(PROTOCOL, root),
    }
    files = {name: _file_binding(path, root) for name, path in paths.items()}
    if files["gated_helper"]["sha256"] != PURE_HELPER_SHA256 or files["gated_tests"]["sha256"] != PURE_TESTS_SHA256:
        raise ValueError("accepted gated pure source bytes changed")
    if files["qa_runtime"]["sha256"] != QA_RUNTIME_SHA256 or files["qa_runtime_tests"]["sha256"] != QA_RUNTIME_TESTS_SHA256:
        raise ValueError("accepted gated QA runtime bytes changed")
    files.update({f"qa:{name}": dict(value) for name, value in qa_accept["files"].items()})
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, Mapping) or not inventory.get("digest"):
        raise ValueError("accepted source inventory is missing")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA, "files": files,
        "accepted_science_source_inventory": dict(inventory),
        "accepted_evidence": dict(evidence), "accepted_qa_record": qa_accept["record"],
        "qa_accept_sha256": qa_accept["sha256"], "baseline_digest": baseline["digest"],
        "training_prefix_digest": prefix_digest, "runtime_settings": dict(settings),
        "endpoint_local_update": diagnostic_runtime.ENDPOINT_LOCAL_UPDATE,
        "endpoint_sha256": diagnostic_runtime.ENDPOINT_SHA256,
    }
    binding["digest"] = _digest(binding)
    return binding


def _input_freeze(*, old_specs: Sequence[Mapping[str, Any]], new_specs: Sequence[Mapping[str, Any]], source_binding: Mapping[str, Any], qa_accept: Mapping[str, Any], baseline: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any], manifest_path: Path, prefix_digest: str, root: Path) -> dict[str, Any]:
    def compact(spec: Mapping[str, Any]) -> dict[str, Any]:
        value = {"id": spec["id"], "suite": spec["suite"], "length": int(spec["length"]), "program": list(spec["program"]),
                 "states": [list(state) for state in spec["states"]], "strata": list(spec["strata"])}
        for key in ("cell", "prefix", "repeat", "suffix"):
            if key in spec:
                value[key] = spec[key]
        return value
    freeze: dict[str, Any] = {
        "schema": INPUT_SCHEMA, "immutable": True,
        "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
        "protocol": {"path": _relative(_resolve(PROTOCOL, root), root), "sha256": _sha256(_resolve(PROTOCOL, root))},
        "arms": list(ARMS), "architecture_ids": {ARM_A: latent.ARCHITECTURE_ID, ARM_B: gated.GATED_ARCHITECTURE_ID},
        "training": {"updates_per_arm": TRAINING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": list(gated.TRAINING_BATCH_LENGTHS), "sum_lengths": sum(gated.TRAINING_BATCH_LENGTHS), "prefix_digest": prefix_digest},
        "checkpoints": list(CHECKPOINT_INDICES), "endpoint": {"local_update": diagnostic_runtime.ENDPOINT_LOCAL_UPDATE, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE, "sha256": diagnostic_runtime.ENDPOINT_SHA256},
        "old_programs": [compact(spec) for spec in old_specs], "new_programs": [compact(spec) for spec in new_specs],
        "source_binding_digest": source_binding["digest"], "qa_accept_sha256": qa_accept["sha256"], "baseline_digest": baseline["digest"],
        "evidence": dict(evidence), "runtime": dict(settings),
    }
    freeze["digest"] = _digest(freeze)
    return freeze


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return science._batch_tensors(batch, device)


def _science_batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != TRAINING_BATCH_SIZE or not batch:
        raise ValueError("science training batch size changed")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("science training batch must have one sequence length")
    result = science._batch_tensors(batch, device)
    _validate_batch_program_ops([example.program for example in batch], result[2])
    return result


@contextmanager
def _qa_batch_tensors_for_science() -> Iterator[None]:
    """Temporarily adapt the accepted QA step to the science batch size."""

    original = qa_runtime._batch_tensors
    original_validator = qa_runtime._validate_program_ops
    qa_runtime._batch_tensors = _science_batch_tensors  # type: ignore[assignment]
    qa_runtime._validate_program_ops = _validate_batch_program_ops  # type: ignore[assignment]
    try:
        yield
    finally:
        qa_runtime._batch_tensors = original  # type: ignore[assignment]
        qa_runtime._validate_program_ops = original_validator  # type: ignore[assignment]


def _validate_logits(logits: Any, cases: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, tuple) or len(logits) != 2 or not all(isinstance(item, torch.Tensor) for item in logits):
        raise ValueError("science logits are malformed")
    x_logits, y_logits = logits
    expected = (cases, length, latent.SLOT_WIDTH)
    if tuple(x_logits.shape) != expected or tuple(y_logits.shape) != expected or x_logits.dtype is not torch.float32 or y_logits.dtype is not torch.float32 or not bool(torch.isfinite(x_logits).all() and torch.isfinite(y_logits).all()):
        raise ValueError("science logits are malformed or nonfinite")
    return x_logits, y_logits


def _run_update(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> float:
    if arm not in ARMS or len(batch) != TRAINING_BATCH_SIZE:
        raise ValueError("science training arm or batch size changed")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("science batch must have one sequence length")
    # Reuse the accepted gated QA step's loss/backward/clip/step ordering.  Its
    # batch tensor helper is replaced only for this call and restored even when
    # forward, backward, or optimizer.step fails.
    counter["active_phase"] = "training"
    with _qa_batch_tensors_for_science():
        return qa_runtime._run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batch, arm=arm, device=device, counter=counter, sink=sink)


def _gate_summary(diagnostics: Mapping[str, Any], length: int) -> list[dict[str, Any]] | None:
    raw = diagnostics.get("gated_carry")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("reader_calls") != length or raw.get("writer_calls") != length:
        raise ValueError("gated hook diagnostics changed")
    values = raw.get("gate_values")
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("gated values are missing")
    result: list[dict[str, Any]] = []
    for position, value in enumerate(values, start=1):
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != (latent.SLOT_COUNT, 1) or not bool(torch.isfinite(value).all()):
            raise ValueError("gated values are malformed or nonfinite")
        per_slot = []
        for slot in range(latent.SLOT_COUNT):
            slot_value = value[:, slot, 0]
            per_slot.append({"slot": slot, "mean": float(slot_value.mean().detach().cpu().item()), "min": float(slot_value.min().detach().cpu().item()), "max": float(slot_value.max().detach().cpu().item())})
        result.append({"position": position, "mean": float(value.mean().detach().cpu().item()), "min": float(value.min().detach().cpu().item()), "max": float(value.max().detach().cpu().item()), "per_slot": per_slot})
    return result


def _trace_metrics(target: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> dict[str, Any]:
    metrics = diagnostic_runtime.accepted_runtime._trace_metrics(target, predicted)
    recovery = None
    if metrics.get("first_error") is not None:
        for index in range(int(metrics["first_error"]), len(target)):
            if tuple(target[index]) == tuple(predicted[index]):
                recovery = index + 1
                break
    metrics["first_subsequent_recovery"] = recovery
    return metrics


def _run_program(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, spec: Mapping[str, Any], arm: str, phase: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> dict[str, Any]:
    if phase not in {"initial", "final"} or arm not in ARMS:
        raise ValueError("science evaluation phase or arm changed")
    counter_phase = f"evaluation_{phase}"
    counter["active_phase"] = counter_phase
    inputs = _build_inputs(spec, device)
    program = inputs["program"]
    length = len(program)
    for key, amount in (("attempted_forwards", 1), ("attempted_cases", len(inputs["examples"])), ("attempted_readout_positions", len(inputs["examples"]) * length), ("attempted_native_steps", len(inputs["examples"]) * length * latent.NATIVE_STEPS)):
        _phase_add(counter, counter_phase, key, amount)
    qa_runtime._flush(counter, sink)
    try:
        with _timed(counter, f"evaluation_{phase}_{arm}_{spec['id']}", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            _validate_program_ops(program, inputs["ops"])
            if arm == ARM_A:
                logits, diagnostics = latent.latent_slots_forward(model, adapter, inputs["x_bits"], inputs["y_bits"], inputs["ops"], return_diagnostics=True)
            else:
                logits, diagnostics = gated.gated_latent_slots_forward(model, adapter, inputs["x_bits"], inputs["y_bits"], inputs["ops"], return_diagnostics=True)
            logits_x, logits_y = _validate_logits(logits, len(inputs["examples"]), length)
            gate_summary = _gate_summary(diagnostics, length)
            if not isinstance(diagnostics.get("slot_inputs"), list) or not isinstance(diagnostics.get("slot_writes"), list) or len(diagnostics["slot_inputs"]) != length or len(diagnostics["slot_writes"]) != length:
                raise ValueError("latent diagnostics are incomplete")
            if any(not bool(torch.isfinite(item).all()) for item in [*diagnostics["slot_inputs"], *diagnostics["slot_writes"]]):
                raise ValueError("latent diagnostics are nonfinite")
        _validate_program_ops(program, inputs["ops"])
        decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
        decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
        predictions: list[dict[str, Any]] = []
        for index, (state, stratum, example) in enumerate(zip(dsl.STATE_ORDER, spec["strata"], inputs["examples"])):
            target = [list(item) for item in example.targets]
            predicted = [[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)]
            predictions.append({"state": list(state), "stratum": stratum, "target_trace": target, "predicted_trace": predicted, **_trace_metrics(target, predicted)})
        for key, amount in (("completed_forwards", 1), ("completed_cases", len(inputs["examples"])), ("completed_readout_positions", len(inputs["examples"]) * length), ("completed_native_steps", len(inputs["examples"]) * length * latent.NATIVE_STEPS)):
            _phase_add(counter, counter_phase, key, amount)
        if arm == ARM_B and (len(adapter.reader._forward_pre_hooks) or len(adapter.writer._forward_hooks)):
            raise ValueError("gated forward left hooks installed")
        row: dict[str, Any] = {"schema": PROGRAM_SCHEMA, "id": str(spec["id"]), "suite": str(spec["suite"]), "length": length, "program": list(program), "arm": arm, "phase": phase, "predictions": predictions, "native_steps": latent.NATIVE_STEPS}
        for key in ("cell", "prefix", "repeat", "suffix"):
            if key in spec:
                row[key] = spec[key]
        if gate_summary is not None:
            row["gated_carry"] = {"positions": gate_summary, "reader_calls": length, "writer_calls": length}
        qa_runtime._flush(counter, sink)
        return row
    except BaseException as exc:
        _phase_failure(counter, counter_phase, "evaluation_forward", exc, sink)
        raise


def _reused_row(spec: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    predictions = deepcopy(baseline["predictions"][spec["id"]])
    return {"schema": PROGRAM_SCHEMA, "id": spec["id"], "suite": spec["suite"], "length": spec["length"], "program": list(spec["program"]), "arm": ARM_A, "phase": "initial_reused_saved", "provenance": {"source": "accepted_final_latent", "replayed": False}, "predictions": predictions, "native_steps": latent.NATIVE_STEPS}


def _checkpoint_payload(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, local_update: int, manifest_sha256: str, source_binding_digest: str, input_freeze_digest: str, parent_identity: Mapping[str, Any], endpoint_sha256: str, adapter_initialization: Mapping[str, Any]) -> dict[str, Any]:
    if local_update not in CHECKPOINT_INDICES:
        raise ValueError("science checkpoint boundary changed")
    model_state = _cpu_copy(model.state_dict())
    adapter_state = _cpu_copy(adapter.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    identity = qa_runtime._arm_identity(model, adapter, optimizer, arm)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA, "committed": True, "complete": True, "arm": arm, "architecture_id": identity["architecture_id"],
        "manifest_sha256": manifest_sha256, "endpoint_sha256": endpoint_sha256, "source_binding_digest": source_binding_digest, "input_freeze_digest": input_freeze_digest, "parent_identity": dict(parent_identity),
        "local_update": local_update, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE + local_update, "next_batch_index": local_update,
        "arm_identity": identity, "adapter_identity": qa_runtime._adapter_identity(adapter, arm), "adapter_initialization": dict(adapter_initialization),
        "model_state_dict": model_state, "adapter_state_dict": adapter_state, "optimizer_state_dict": optimizer_state, "optimizer_metadata": accepted_qa_runtime._optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng, "cuda_rng_state": cuda_rng, "training_mode": bool(model.training), "adapter_training_mode": bool(adapter.training),
        "model_parameter_names": list(model.state_dict()), "adapter_parameter_names": [name for name, _ in adapter.named_parameters()], "optimizer_group_param_names": [list(group.get("param_names", [])) for group in optimizer.param_groups],
        "model_digest": _digest(model_state), "adapter_digest": _digest(adapter_state), "optimizer_digest": _digest(optimizer_state), "cpu_rng_digest": _digest(cpu_rng), "cuda_rng_digest": _digest(cuda_rng),
    }
    payload["checkpoint_digest"] = _digest(payload)
    return payload


def _checkpoint_metadata(local_update: int, *, arm: str, source_binding_digest: str, input_freeze_digest: str, parent_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Pure boundary record used by tests and by the checkpoint writer."""

    if arm not in ARMS or local_update not in CHECKPOINT_INDICES:
        raise ValueError("science checkpoint arm or boundary changed")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "committed": True,
        "complete": True,
        "arm": arm,
        "local_update": int(local_update),
        "absolute_update": ENDPOINT_ABSOLUTE_UPDATE + int(local_update),
        "next_batch_index": int(local_update),
        "source_binding_digest": str(source_binding_digest),
        "input_freeze_digest": str(input_freeze_digest),
        "parent_identity": dict(parent_identity),
    }


def _write_checkpoint(*, arm_out: Path, payload: Mapping[str, Any], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None, root: Path) -> dict[str, Any]:
    local_update = int(payload["local_update"])
    path = arm_out / "checkpoints" / f"local{local_update:04d}.pt"
    counter["active_phase"] = "checkpoint_save"
    _phase_add(counter, "checkpoint_save", "attempted_checkpoints")
    qa_runtime._flush(counter, sink)
    try:
        with _timed(counter, f"checkpoint_save_{payload['arm']}_{local_update}", sink):
            _atomic_torch(path, payload, refuse=True)
    except BaseException as exc:
        _phase_failure(counter, "checkpoint_save", "checkpoint_save", exc, sink)
        raise
    _phase_add(counter, "checkpoint_save", "completed_checkpoints")
    qa_runtime._flush(counter, sink)
    return {"local_update": local_update, "absolute_update": int(payload["absolute_update"]), "next_batch_index": int(payload["next_batch_index"]), "path": _relative(path, root), "sha256": _sha256(path), "checkpoint_digest": payload["checkpoint_digest"], "committed": True, "complete": True}


def _persist_row(rows_dir: Path, row: Mapping[str, Any], row_index: list[int]) -> None:
    """Commit each program row immediately so a failed run retains evidence."""

    index = row_index[0]
    path = rows_dir / f"{index:04d}_{row['arm']}_{row['phase']}_{row['id']}.json"
    _atomic_json(path, dict(row), refuse=True)
    row_index[0] = index + 1


def _expected_program_row_count() -> int:
    """Rows include reused initial A old traces plus four endpoint phases."""

    return 4 * (OLD_PROGRAMS + NEW_PROGRAMS)


def _paired(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    def key(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> tuple[str, tuple[int, int]]:
        return str(row["id"]), tuple(int(value) for value in prediction["state"])
    left_map = {key(row, pred): pred for row in left for pred in row["predictions"]}
    right_map = {key(row, pred): pred for row in right for pred in row["predictions"]}
    if set(left_map) != set(right_map):
        raise ValueError(f"paired rows do not join: {label}")
    counts = {"repair": 0, "regression": 0, "both_correct": 0, "both_wrong": 0}
    for join in left_map:
        candidate, reference = left_map[join], right_map[join]
        candidate_ok = bool(candidate.get("full_trace_correct"))
        reference_ok = bool(reference.get("full_trace_correct"))
        if candidate_ok and not reference_ok:
            counts["repair"] += 1
        elif reference_ok and not candidate_ok:
            counts["regression"] += 1
        elif candidate_ok and reference_ok:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return {"label": label, "cases": len(left_map), **counts}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    def add(group: dict[str, int], prediction: Mapping[str, Any]) -> None:
        group["cases"] = group.get("cases", 0) + 1
        group["full_trace_correct"] = group.get("full_trace_correct", 0) + int(bool(prediction.get("full_trace_correct")))
        group["final_correct"] = group.get("final_correct", 0) + int(bool(prediction.get("joint_final_correct")))
    for row in rows:
        keys = ["all", f"suite:{row['suite']}", f"length:{row['length']}"]
        if "cell" in row:
            keys.append(f"cell:{row['cell']}")
        for prediction in row["predictions"]:
            keys_for_prediction = keys + [f"stratum:{prediction['stratum']}", f"suite:{row['suite']}|length:{row['length']}|stratum:{prediction['stratum']}"]
            for group_key in keys_for_prediction:
                add(groups.setdefault(group_key, {}), prediction)
    return {"groups": groups, "denominators": {key: value["cases"] for key, value in groups.items()}}


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 2, "completed_endpoint_loads": 2,
        "attempted_parent_loads": 2, "completed_parent_loads": 2,
        "attempted_endpoint_restores": 2, "completed_endpoint_restores": 2,
        "attempted_underlying_deserializations": SCIENCE_DESERIALIZATIONS, "completed_underlying_deserializations": SCIENCE_DESERIALIZATIONS,
        "attempted_forwards": SCIENCE_CALLS, "completed_forwards": SCIENCE_CALLS,
        "attempted_cases": SCIENCE_CASES, "completed_cases": SCIENCE_CASES,
        "attempted_readout_positions": SCIENCE_POSITIONS, "completed_readout_positions": SCIENCE_POSITIONS,
        "attempted_native_steps": SCIENCE_NATIVE_STEPS, "completed_native_steps": SCIENCE_NATIVE_STEPS,
        "attempted_updates": SCIENCE_UPDATES, "completed_updates": SCIENCE_UPDATES,
        "attempted_backwards": SCIENCE_UPDATES, "completed_backwards": SCIENCE_UPDATES,
        "attempted_optimizer_steps": SCIENCE_UPDATES, "completed_optimizer_steps": SCIENCE_UPDATES,
        "attempted_checkpoints": CHECKPOINT_COUNT, "completed_checkpoints": CHECKPOINT_COUNT,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"gated science accounting mismatch: {key}={counter.get(key)} expected {value}")
    if counter.get("failures"):
        raise ValueError("gated science recorded failures")
    if counter.get("phase_counts", {}).get("training", {}).get("completed_updates") != SCIENCE_UPDATES:
        raise ValueError("training update accounting mismatch")
    if counter.get("phase_counts", {}).get("evaluation_initial", {}).get("completed_forwards") != OLD_PROGRAMS + 2 * NEW_PROGRAMS:
        raise ValueError("initial evaluation accounting mismatch")
    if counter.get("phase_counts", {}).get("evaluation_final", {}).get("completed_forwards") != 2 * (OLD_PROGRAMS + NEW_PROGRAMS):
        raise ValueError("final evaluation accounting mismatch")


def _evaluate_phase(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, specs: Sequence[Mapping[str, Any]], arm: str, phase: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None, row_sink: Callable[[Mapping[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
        for spec in specs:
            row = _run_program(model=model, adapter=adapter, spec=spec, arm=arm, phase=phase, device=device, counter=counter, sink=sink)
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


def _run_arm(*, arm: str, old_specs: Sequence[Mapping[str, Any]], new_specs: Sequence[Mapping[str, Any]], batches: Sequence[Sequence[dsl.RegisterExample]], manifest: Mapping[str, Any], science_report: Mapping[str, Any], source_binding: Mapping[str, Any], input_freeze: Mapping[str, Any], device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None, out: Path, root: Path, rows_dir: Path, row_index: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm_out = out / ARM_LABELS[arm]
    arm_out.mkdir(parents=True, exist_ok=False)
    counter["active_phase"] = "load"
    model, adapter, optimizer, endpoint_payload = diagnostic_runtime._load_endpoint(manifest=manifest, report=science_report, endpoint=_resolve(SCIENCE_ENDPOINT, root), manifest_path=_resolve(SCIENCE_MANIFEST, root), source_binding_digest=str(manifest["source_inventory"]["digest"]), device=device, counter=counter, sink=sink, root=root)
    try:
        adapter_initialization = dict(endpoint_payload["adapter_initialization"])
        if arm == ARM_B:
            gated.attach_carry_gate(adapter)
            gated.append_carry_gate_optimizer_group(optimizer, adapter)
            gated.validate_optimizer_carry_gate_association(optimizer, adapter)
        identity = qa_runtime._arm_identity(model, adapter, optimizer, arm)
        qa_runtime._validate_arm_identity(identity, arm=arm)
        parent_identity = endpoint_payload.get("parent_identity")
        if not isinstance(parent_identity, Mapping):
            raise ValueError("endpoint parent identity is missing")
        model.train(True)
        adapter.train(True)
        ledger: list[dict[str, Any]] = []
        progress: list[dict[str, Any]] = []
        _atomic_json(arm_out / "training_progress.json", progress, refuse=True)
        for index in CHECKPOINT_INDICES:
            if index == 0:
                payload = _checkpoint_payload(model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=0, manifest_sha256=_sha256(_resolve(SCIENCE_MANIFEST, root)), source_binding_digest=str(source_binding["digest"]), input_freeze_digest=str(input_freeze["digest"]), parent_identity=parent_identity, endpoint_sha256=diagnostic_runtime.ENDPOINT_SHA256, adapter_initialization=adapter_initialization)
                ledger.append(_write_checkpoint(arm_out=arm_out, payload=payload, counter=counter, sink=sink, root=root))
                _atomic_json(arm_out / "checkpoint_ledger.json", ledger)
                break
        initial_specs = list(new_specs) if arm == ARM_A else [*old_specs, *new_specs]
        rows = _evaluate_phase(model=model, adapter=adapter, optimizer=optimizer, specs=initial_specs, arm=arm, phase="initial", device=device, counter=counter, sink=sink, row_sink=lambda row: _persist_row(rows_dir, row, row_index))
        for update_index, batch in enumerate(batches, start=1):
            loss = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batch, arm=arm, device=device, counter=counter, sink=sink)
            progress.append({"local_update": update_index, "absolute_update": ENDPOINT_ABSOLUTE_UPDATE + update_index, "next_batch_index": update_index, "length": len(batch[0].program), "loss": loss})
            _atomic_json(arm_out / "training_progress.json", progress)
            if update_index in CHECKPOINT_INDICES[1:]:
                payload = _checkpoint_payload(model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=update_index, manifest_sha256=_sha256(_resolve(SCIENCE_MANIFEST, root)), source_binding_digest=str(source_binding["digest"]), input_freeze_digest=str(input_freeze["digest"]), parent_identity=parent_identity, endpoint_sha256=diagnostic_runtime.ENDPOINT_SHA256, adapter_initialization=adapter_initialization)
                ledger.append(_write_checkpoint(arm_out=arm_out, payload=payload, counter=counter, sink=sink, root=root))
                _atomic_json(arm_out / "checkpoint_ledger.json", ledger)
        if len(ledger) != len(CHECKPOINT_INDICES):
            raise ValueError("science checkpoint ledger is incomplete")
        final_rows = _evaluate_phase(model=model, adapter=adapter, optimizer=optimizer, specs=[*old_specs, *new_specs], arm=arm, phase="final", device=device, counter=counter, sink=sink, row_sink=lambda row: _persist_row(rows_dir, row, row_index))
        _atomic_json(arm_out / "training.json", {"schema": "pc_gated_carry_science_training_v1", "arm": arm, "updates": TRAINING_UPDATES, "batch_count": len(batches), "checkpoints": ledger, "identity": identity}, refuse=True)
        return rows + final_rows, ledger
    finally:
        model = None
        adapter = None
        optimizer = None


def _rows_for(rows: Sequence[Mapping[str, Any]], *, arm: str, phase: str, suite: str | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("arm") == arm and row.get("phase") == phase and (suite is None or row.get("suite") == suite)]


def run_science(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = lambda value: _atomic_json(accounting_path, value)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "manifest": {"path": _relative(manifest_path, root), "sha256": None}, "budget": {"forwards": SCIENCE_CALLS, "cases": SCIENCE_CASES, "positions": SCIENCE_POSITIONS, "native_steps": SCIENCE_NATIVE_STEPS, "backwards": SCIENCE_UPDATES, "optimizer_updates": SCIENCE_UPDATES, "underlying_deserializations": SCIENCE_DESERIALIZATIONS}, "accounting": dict(counter)}
    model_loaded = False
    try:
        manifest, science_report, scope, final, evidence = diagnostic_runtime._validate_evidence(root, manifest_path)
        device, settings = diagnostic_runtime._load_runtime_settings(manifest)
        qa_accept = _validate_qa_accept(root)
        old_specs = _old_specs(scope)
        new_specs = _new_specs()
        baseline = _validate_old_baseline(final, old_specs)
        _scope, _evidence, all_batches = science._load_frozen_data(manifest, root)
        batches = all_batches[:TRAINING_UPDATES]
        prefix_digest = _stream_prefix_digest(batches)
        source_binding = _source_binding(root=root, manifest=manifest, evidence=evidence, qa_accept=qa_accept, baseline=baseline, settings=settings, prefix_digest=prefix_digest)
        input_freeze = _input_freeze(old_specs=old_specs, new_specs=new_specs, source_binding=source_binding, qa_accept=qa_accept, baseline=baseline, evidence=evidence, settings=settings, manifest_path=manifest_path, prefix_digest=prefix_digest, root=root)
        _atomic_json(out / "source_binding.json", source_binding, refuse=True)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report.update({"manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]}, "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": input_freeze["digest"]}, "runtime": settings, "endpoint": evidence["endpoint"], "qa_accept": {"path": _relative(QA_ACCEPT, root), "sha256": qa_accept["sha256"]}, "baseline": {"phase": "final_latent", "source": "accepted_saved_final_A", "replayed": False, "digest": baseline["digest"]}, "scope": {"old_programs": len(old_specs), "new_programs": len(new_specs), "new_lengths": {str(length): sum(spec["length"] == length for spec in new_specs) for length in gated.CONTROL_LENGTHS}}})
        rows_dir = out / "program_rows"
        rows_dir.mkdir(parents=True, exist_ok=False)
        all_rows: list[dict[str, Any]] = []
        row_index = [0]
        for spec in old_specs:
            row = _reused_row(spec, baseline)
            _persist_row(rows_dir, row, row_index)
            all_rows.append(row)
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        ledgers: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            result_rows, ledger = _run_arm(arm=arm, old_specs=old_specs, new_specs=new_specs, batches=batches, manifest=manifest, science_report=science_report, source_binding=source_binding, input_freeze=input_freeze, device=device, counter=counter, sink=sink, out=out, root=root, rows_dir=rows_dir, row_index=row_index)
            model_loaded = True
            arm_rows[arm] = result_rows
            ledgers[arm] = ledger
            all_rows.extend(result_rows)
        if len(all_rows) != _expected_program_row_count():
            raise ValueError(f"science row count changed: {len(all_rows)}")
        initial_b = _rows_for(arm_rows[ARM_B], arm=ARM_B, phase="initial")
        final_a = _rows_for(arm_rows[ARM_A], arm=ARM_A, phase="final")
        final_b = _rows_for(arm_rows[ARM_B], arm=ARM_B, phase="final")
        reused_a = [row for row in all_rows if row.get("phase") == "initial_reused_saved"]
        new_initial_a = _rows_for(arm_rows[ARM_A], arm=ARM_A, phase="initial")
        paired = {
            "new_initial_B_vs_A": _paired([row for row in initial_b if row["suite"] == "identity_controls"], new_initial_a, label="new_initial_B_vs_A"),
            "old_initial_B_vs_saved_initial_A": _paired([row for row in initial_b if row["suite"] != "identity_controls"], reused_a, label="old_initial_B_vs_saved_initial_A"),
            "old_padding_final_B_vs_A": _paired([row for row in final_b if row["suite"] == "padding"], [row for row in final_a if row["suite"] == "padding"], label="old_padding_final_B_vs_A"),
            "old_compositions_final_B_vs_A": _paired([row for row in final_b if row["suite"] == "compositions"], [row for row in final_a if row["suite"] == "compositions"], label="old_compositions_final_B_vs_A"),
            "new_final_B_vs_A": _paired([row for row in final_b if row["suite"] == "identity_controls"], [row for row in final_a if row["suite"] == "identity_controls"], label="new_final_B_vs_A"),
        }
        composition_b = [prediction for row in final_b if row["suite"] == "compositions" for prediction in row["predictions"]]
        report.update({"status": "complete", "accounting": dict(counter), "rows": {"count": len(all_rows), "directory": _relative(rows_dir, root), "program_rows": len(all_rows)}, "aggregates": {"final_A": _aggregate(final_a), "final_B": _aggregate(final_b), "initial_B": _aggregate(initial_b), "saved_initial_A": _aggregate(reused_a)}, "paired": paired, "checkpoints": ledgers, "positive_pilot": {"new_final_B_fewer_full_trace_errors": paired["new_final_B_vs_A"]["regression"] < paired["new_final_B_vs_A"]["repair"], "old_padding_final_B_fewer_full_trace_errors": paired["old_padding_final_B_vs_A"]["regression"] < paired["old_padding_final_B_vs_A"]["repair"], "old_compositions_zero_B_wrong_A_correct": paired["old_compositions_final_B_vs_A"]["regression"] == 0, "old_compositions_final_B_full_trace_correct": sum(bool(item.get("full_trace_correct")) for item in composition_b), "old_compositions_denominator": 6144, "old_compositions_retained": sum(bool(item.get("full_trace_correct")) for item in composition_b) == 6144}})
        _validate_accounting(counter)
        report["accounting"] = dict(counter)
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        phase = str(counter["active_phase"])
        _phase_failure(counter, phase, "science", exc, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report)
        raise


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
