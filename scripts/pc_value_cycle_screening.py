"""Short baseline-vs-cycle-loss screening for the projected-value contract.

The screening starts two independent arms from the same accepted E36 endpoint:
ordinary supervised latent-slot training and the same training with the
detached projected-value identity-cycle loss.  It uses a fixed 64-update
prefix and a small held-out evaluation scope.  This is a bounded screening
boundary, not a full pilot and not a scientific superiority claim.
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
from looped_bitnet import longer_native8_e20 as old
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_runtime as accepted_qa
from scripts import pc_latent_slots_science as science
from scripts import pc_value_cycle_contract as contract
from scripts import pc_value_cycle_runtime as cycle_qa


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "pc_projected_value_cycle_screening_v1"
OUTPUT = RUN_ROOT / "screening"
LAUNCH_GATE = RUN_ROOT / "screening_launch_gate_v1.json"
MANIFEST = ROOT / "runs" / "pc_latent_slots_v1" / "science_manifest.json"
SCREENING_PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_PROJECTED_VALUE_CYCLE_SCREENING_PROTOCOL.md"
SCREENING_TESTS = ROOT / "tests" / "test_pc_value_cycle_screening.py"
CYCLE_QA_ACCEPT = ROOT / "runs" / "pc_projected_value_cycle_v2" / "qa" / "qa_accept.json"
STREAM = ROOT / "runs" / "pc_latent_slots_v1" / "science_stream.json"
SCOPE = ROOT / "runs" / "pc_latent_slots_v1" / "science_eval_scope.json"

SCHEMA = "pc_projected_value_cycle_screening_v1"
ACCOUNTING_SCHEMA = "pc_projected_value_cycle_screening_accounting_v1"
SOURCE_SCHEMA = "pc_projected_value_cycle_screening_source_binding_v1"
INPUT_SCHEMA = "pc_projected_value_cycle_screening_input_freeze_v1"
CHECKPOINT_SCHEMA = "pc_projected_value_cycle_screening_checkpoint_v1"
ARM_ORDINARY = "ordinary"
ARM_CYCLE = "cycle_loss"
ARMS = (ARM_ORDINARY, ARM_CYCLE)
ARM_LABELS = {
    ARM_ORDINARY: "supervised_only",
    ARM_CYCLE: "supervised_plus_projected_v_cycle",
}

SCREENING_UPDATES = 64
TRAINING_BATCH_SIZE = 64
TRAINING_LENGTHS = tuple(int(value) for value in latent.FIXED_BATCH_LENGTHS[:SCREENING_UPDATES])
TRAINING_SUM_LENGTHS = sum(TRAINING_LENGTHS)
EVAL_STATE_STRATA = ("validation", "test")
EVAL_STATES = tuple(
    tuple(int(value) for value in state)
    for stratum in EVAL_STATE_STRATA
    for state in dsl.state_split()[stratum]
)
EVAL_STATE_COUNT = len(EVAL_STATES)
EVAL_PROGRAM_IDS = (
    "padding_ADDADD_ADD_k4",
    "padding_ADDADD_ADD_k10",
    "padding_ADDADD_XOR_k4",
    "padding_ADDADD_XOR_k10",
    "padding_XORSWAP_ADD_k4",
    "padding_XORSWAP_ADD_k10",
    "composition_L12_0",
    "composition_L16_0",
)
EVAL_PROGRAM_COUNT = len(EVAL_PROGRAM_IDS)
EVAL_FORWARDS_PER_PHASE_PER_ARM = EVAL_PROGRAM_COUNT
EVAL_CASES_PER_PHASE_PER_ARM = EVAL_PROGRAM_COUNT * EVAL_STATE_COUNT
EVAL_POSITIONS_PER_PHASE_PER_ARM = 8_704
EVAL_NATIVE_PER_PHASE_PER_ARM = EVAL_POSITIONS_PER_PHASE_PER_ARM * latent.NATIVE_STEPS
TRAINING_FORWARDS_TOTAL = len(ARMS) * SCREENING_UPDATES
TRAINING_CASES_TOTAL = TRAINING_FORWARDS_TOTAL * TRAINING_BATCH_SIZE
TRAINING_POSITIONS_TOTAL = len(ARMS) * TRAINING_SUM_LENGTHS * TRAINING_BATCH_SIZE
TRAINING_NATIVE_TOTAL = TRAINING_POSITIONS_TOTAL * latent.NATIVE_STEPS
EVAL_PHASES_TOTAL = len(ARMS) * 2
EVAL_FORWARDS_TOTAL = EVAL_PHASES_TOTAL * EVAL_FORWARDS_PER_PHASE_PER_ARM
EVAL_CASES_TOTAL = EVAL_PHASES_TOTAL * EVAL_CASES_PER_PHASE_PER_ARM
EVAL_POSITIONS_TOTAL = EVAL_PHASES_TOTAL * EVAL_POSITIONS_PER_PHASE_PER_ARM
EVAL_NATIVE_TOTAL = EVAL_PHASES_TOTAL * EVAL_NATIVE_PER_PHASE_PER_ARM
TOTAL_FORWARDS = TRAINING_FORWARDS_TOTAL + EVAL_FORWARDS_TOTAL
TOTAL_CASES = TRAINING_CASES_TOTAL + EVAL_CASES_TOTAL
TOTAL_POSITIONS = TRAINING_POSITIONS_TOTAL + EVAL_POSITIONS_TOTAL
TOTAL_NATIVE_STEPS = TRAINING_NATIVE_TOTAL + EVAL_NATIVE_TOTAL
TOTAL_BACKWARDS = len(ARMS) * SCREENING_UPDATES
TOTAL_OPTIMIZER_UPDATES = TOTAL_BACKWARDS
TOTAL_ENDPOINT_LOADS = len(ARMS)
TOTAL_UNDERLYING_DESERIALIZATIONS = len(ARMS) * 2
TOTAL_CHECKPOINTS = len(ARMS)
EXPECTED_TRAINING_CYCLE_WINDOWS = 1_225
EXPECTED_TRAINING_CYCLE_UPDATES = 53
EXPECTED_EVAL_CYCLE_WINDOWS = 12_544


def _resolve(path: Path | str, root: Path = ROOT) -> Path:
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


def _digest(value: Any) -> str:
    return old.digest_object(value)


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


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty screening output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _file_binding(path: Path, root: Path = ROOT) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _new_counter() -> dict[str, Any]:
    counter = accepted_qa._new_counter()
    counter.update(
        {
            "schema": ACCOUNTING_SCHEMA,
            "active_phase": "load",
            "cycle_windows": 0,
            "cycle_updates": 0,
            "cycle_windows_applied": 0,
            "cycle_updates_applied": 0,
            "attempted_checkpoint_saves": 0,
            "completed_checkpoint_saves": 0,
        }
    )
    return counter


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    if sink is not None:
        sink(counter)


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = counter.setdefault("timings_seconds", {})
        timings[name] = float(timings.get(name, 0.0)) + time.perf_counter() - started
        _flush(counter, sink)


def _load_launch_gate(root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(LAUNCH_GATE, root)
    gate = _read_object(path)
    if gate.get("schema") != "pc_projected_value_cycle_screening_launch_gate_v1" or gate.get("status") != "READY":
        raise ValueError("projected-value screening launch gate is not READY")
    expected_scope = {
        "updates_per_arm": SCREENING_UPDATES,
        "batch_size": TRAINING_BATCH_SIZE,
        "training_lengths": list(TRAINING_LENGTHS),
        "arms": list(ARMS),
        "eval_program_ids": list(EVAL_PROGRAM_IDS),
        "eval_state_strata": list(EVAL_STATE_STRATA),
        "eval_state_count": EVAL_STATE_COUNT,
    }
    if gate.get("scope") != expected_scope:
        raise ValueError("projected-value screening scope changed")
    expected_budget = {
        "endpoint_loads": TOTAL_ENDPOINT_LOADS,
        "underlying_deserializations": TOTAL_UNDERLYING_DESERIALIZATIONS,
        "training_forwards": TRAINING_FORWARDS_TOTAL,
        "evaluation_forwards": EVAL_FORWARDS_TOTAL,
        "forwards": TOTAL_FORWARDS,
        "cases": TOTAL_CASES,
        "positions": TOTAL_POSITIONS,
        "native_steps": TOTAL_NATIVE_STEPS,
        "backwards": TOTAL_BACKWARDS,
        "optimizer_updates": TOTAL_OPTIMIZER_UPDATES,
        "checkpoints": TOTAL_CHECKPOINTS,
        "training_cycle_windows": EXPECTED_TRAINING_CYCLE_WINDOWS,
        "training_cycle_updates": EXPECTED_TRAINING_CYCLE_UPDATES,
        "evaluation_cycle_windows": EXPECTED_EVAL_CYCLE_WINDOWS,
    }
    if gate.get("budget") != expected_budget:
        raise ValueError("projected-value screening budget changed")
    files = gate.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("projected-value screening source gate is malformed")
    for name, raw in files.items():
        if not isinstance(raw, Mapping) or "path" not in raw or "sha256" not in raw:
            raise ValueError(f"projected-value screening source binding is malformed: {name}")
        path_value = Path(str(raw["path"]).replace("\\", "/"))
        path = _resolve(path_value, root)
        expected = str(raw["sha256"]).lower()
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"projected-value screening source changed: {name}")
    output = gate.get("output")
    if str(output).replace("\\", "/") != _relative(OUTPUT, root).replace("\\", "/"):
        raise ValueError("projected-value screening output changed")
    return gate


def _source_binding(*, root: Path, manifest: Mapping[str, Any], prefix_digest: str, selection_digest: str, settings: Mapping[str, Any], cycle_qa_binding: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "screening_runtime": Path(__file__).resolve(),
        "screening_tests": SCREENING_TESTS,
        "screening_protocol": _resolve(SCREENING_PROTOCOL, root),
        "cycle_qa_runtime": root / "scripts" / "pc_value_cycle_runtime.py",
        "cycle_qa_tests": root / "tests" / "test_pc_value_cycle_runtime.py",
        "cycle_contract": root / "scripts" / "pc_value_cycle_contract.py",
        "cycle_contract_tests": root / "tests" / "test_pc_value_cycle_contract.py",
        "latent_science": root / "scripts" / "pc_latent_slots_science.py",
        "latent_helper": root / "scripts" / "pc_latent_slots.py",
        "latent_runtime": root / "scripts" / "pc_latent_slots_runtime.py",
        "accepted_scratchpad": root / "scripts" / "pc_learned_scratchpad.py",
        "accepted_scratchpad_runtime": root / "scripts" / "pc_learned_scratchpad_runtime.py",
        "register_e15": root / "looped_bitnet" / "register_e15.py",
        "manifest": _resolve(MANIFEST, root),
        "stream": _resolve(STREAM, root),
        "scope": _resolve(SCOPE, root),
        "cycle_qa_accept": _resolve(CYCLE_QA_ACCEPT, root),
        "parent_checkpoint": accepted_qa._parent_checkpoint(root),
    }
    files = {name: _file_binding(path, root) for name, path in paths.items()}
    accepted_record = _read_object(_resolve(CYCLE_QA_ACCEPT, root))
    if accepted_record.get("status") != "ACCEPT" or accepted_record.get("science_executed") is not False:
        raise ValueError("projected-value cycle QA acceptance is not the expected no-science record")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "accepted_cycle_qa_source_binding": dict(cycle_qa_binding),
        "cycle_qa_accept_status": accepted_record.get("status"),
        "manifest_schema": manifest.get("schema"),
        "manifest_digest": _digest(manifest),
        "training_prefix_digest": prefix_digest,
        "evaluation_selection_digest": selection_digest,
        "runtime_settings": dict(settings),
        "architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID,
        "target": "per_slot_projected_V",
        "target_detached": True,
        "weight": contract.VALUE_CYCLE_LOSS_WEIGHT,
    }
    binding["digest"] = _digest(binding)
    return binding


def _prefix_digest(batches: Sequence[Sequence[dsl.RegisterExample]]) -> tuple[str, int, int]:
    if len(batches) < SCREENING_UPDATES:
        raise ValueError("screening stream is shorter than the registered prefix")
    records: list[list[dict[str, Any]]] = []
    cycle_windows = 0
    cycle_updates = 0
    lengths: list[int] = []
    for index, batch in enumerate(batches[:SCREENING_UPDATES]):
        if len(batch) != TRAINING_BATCH_SIZE or not batch:
            raise ValueError(f"screening batch {index} size changed")
        length = len(batch[0].program)
        if length != TRAINING_LENGTHS[index] or any(len(example.program) != length for example in batch):
            raise ValueError(f"screening batch {index} length changed")
        rows = []
        batch_windows = 0
        for example in batch:
            rows.append({"x": int(example.x), "y": int(example.y), "program": list(example.program), "target_trace": [list(pair) for pair in example.targets]})
            ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program]], dtype=torch.long)
            batch_windows += len(contract.find_identity_cycle_windows(ops))
        cycle_windows += batch_windows
        cycle_updates += int(batch_windows > 0)
        lengths.append(length)
        records.append(rows)
    digest = _digest({"batch_count": SCREENING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": lengths, "records": records})
    return digest, cycle_windows, cycle_updates


def _selection(scope: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    programs = scope.get("programs")
    if not isinstance(programs, list):
        raise ValueError("evaluation scope programs are missing")
    by_id = {str(item.get("id")): item for item in programs if isinstance(item, Mapping)}
    if set(EVAL_PROGRAM_IDS) - set(by_id):
        raise ValueError("registered evaluation program is missing")
    specs = [dict(by_id[identifier]) for identifier in EVAL_PROGRAM_IDS]
    valid_states = {tuple(state) for state in EVAL_STATES}
    compact: list[dict[str, Any]] = []
    eval_windows = 0
    for spec in specs:
        program = tuple(str(op) for op in spec.get("program", ()))
        if not program or int(spec.get("length", -1)) != len(program):
            raise ValueError(f"evaluation program is malformed: {spec.get('id')}")
        windows = contract.find_identity_cycle_windows(torch.tensor([[dsl.OP_TO_ID[op] for op in program]], dtype=torch.long))
        eval_windows += len(windows) * EVAL_STATE_COUNT * 4
        compact.append({"id": str(spec["id"]), "suite": str(spec["suite"]), "length": len(program), "program": list(program), "cycle_windows_per_case": len(windows)})
    selection = {
        "schema": INPUT_SCHEMA,
        "program_ids": list(EVAL_PROGRAM_IDS),
        "programs": compact,
        "state_strata": list(EVAL_STATE_STRATA),
        "states": [list(state) for state in EVAL_STATES],
        "state_count": EVAL_STATE_COUNT,
        "phase_count": 4,
        "evaluation_cycle_windows": eval_windows,
    }
    return specs, selection, _digest(selection)


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) not in {TRAINING_BATCH_SIZE, EVAL_STATE_COUNT} or not batch:
        raise ValueError("screening batch size changed")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("screening batch is not homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = cycle_qa.accepted_qa.accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _run_update(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, batch: Sequence[dsl.RegisterExample], arm: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> tuple[float, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError("unknown screening arm")
    length = len(batch[0].program)
    cases = len(batch)
    counter["active_phase"] = "training"
    counter["attempted_updates"] += 1
    counter["attempted_forwards"] += 1
    counter["attempted_cases"] += cases
    counter["attempted_readout_positions"] += cases * length
    counter["attempted_native_steps"] += cases * length * latent.NATIVE_STEPS
    _flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with _timed(counter, "training_forward", sink), torch.autocast(device_type=device.type, enabled=False):
            (logits_x, logits_y), diagnostics = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or not bool(torch.isfinite(logits_x).all() and torch.isfinite(logits_y).all()):
            raise ValueError("screening logits are malformed or nonfinite")
    except BaseException as exc:
        accepted_qa._failure(counter, "training_forward", exc)
        _flush(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
    _flush(counter, sink)
    try:
        with _timed(counter, "training_loss", sink):
            supervised = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            if arm == ARM_CYCLE:
                cycle_term, cycle_meta = contract.projected_value_cycle_loss(model, adapter, diagnostics, ops)
            else:
                with torch.no_grad():
                    _unused, cycle_meta = contract.projected_value_cycle_loss(model, adapter, diagnostics, ops)
                cycle_term = supervised * 0.0
            loss = supervised + cycle_term
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("screening loss is nonfinite")
    except BaseException as exc:
        accepted_qa._failure(counter, "training_loss", exc)
        _flush(counter, sink)
        raise
    windows = int(cycle_meta["windows"])
    counter["cycle_windows"] += windows
    counter["cycle_updates"] += int(windows > 0)
    if arm == ARM_CYCLE:
        counter["cycle_windows_applied"] += windows
        counter["cycle_updates_applied"] += int(windows > 0)
    counter["attempted_backwards"] += 1
    _flush(counter, sink)
    try:
        with _timed(counter, "training_backward", sink):
            loss.backward()
        counter["completed_backwards"] += 1
        with _timed(counter, "training_gradient_clip", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
        counter["attempted_optimizer_steps"] += 1
        with _timed(counter, "training_optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        accepted_qa._failure(counter, "training_update", exc)
        _flush(counter, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    _flush(counter, sink)
    return float(loss.detach().item()), {
        "supervised_loss": float(supervised.detach().item()),
        "cycle": dict(cycle_meta),
        "cycle_applied": arm == ARM_CYCLE,
        "total_loss": float(loss.detach().item()),
    }


def _eval_inputs(spec: Mapping[str, Any], device: torch.device) -> tuple[list[dsl.RegisterExample], torch.Tensor, torch.Tensor, torch.Tensor]:
    program = tuple(str(op) for op in spec["program"])
    batch = [dsl.RegisterExample(state[0], state[1], program) for state in EVAL_STATES]
    x_bits, y_bits, ops, _tx, _ty = _batch_tensors(batch, device)
    return batch, x_bits, y_bits, ops


def _eval_phase(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, specs: Sequence[Mapping[str, Any]], arm: str, phase: str, device: torch.device, counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if phase not in {"initial", "final"}:
        raise ValueError("unknown screening evaluation phase")
    rows: list[dict[str, Any]] = []
    cycle_windows = 0
    cycle_raw_weighted = 0.0
    with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
        for spec in specs:
            batch, x_bits, y_bits, ops = _eval_inputs(spec, device)
            length = len(batch[0].program)
            cases = len(batch)
            counter["active_phase"] = f"evaluation_{phase}"
            counter["attempted_forwards"] += 1
            counter["attempted_cases"] += cases
            counter["attempted_readout_positions"] += cases * length
            counter["attempted_native_steps"] += cases * length * latent.NATIVE_STEPS
            _flush(counter, sink)
            try:
                with _timed(counter, f"evaluation_{phase}_forward", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    (logits_x, logits_y), diagnostics = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
                    expected = (cases, length, latent.SLOT_WIDTH)
                    if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected or logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not bool(torch.isfinite(logits_x).all() and torch.isfinite(logits_y).all()):
                        raise ValueError("screening evaluation logits are malformed or nonfinite")
                    _value, cycle_meta = contract.projected_value_cycle_loss(model, adapter, diagnostics, ops)
            except BaseException as exc:
                accepted_qa._failure(counter, f"evaluation_{phase}_forward", exc)
                _flush(counter, sink)
                raise
            counter["completed_forwards"] += 1
            counter["completed_cases"] += cases
            counter["completed_readout_positions"] += cases * length
            counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
            windows = int(cycle_meta["windows"])
            cycle_windows += windows
            cycle_raw_weighted += float(cycle_meta["raw_loss"]) * windows
            _flush(counter, sink)
            decoded_x = logits_x.argmax(-1).detach().cpu().tolist()
            decoded_y = logits_y.argmax(-1).detach().cpu().tolist()
            predictions: list[dict[str, Any]] = []
            for index, (state, example) in enumerate(zip(EVAL_STATES, batch)):
                target = [list(pair) for pair in example.targets]
                predicted = [[int(decoded_x[index][position]), int(decoded_y[index][position])] for position in range(length)]
                predictions.append({"state": list(state), "stratum": next(name for name, values in dsl.state_split().items() if state in values), "target_trace": target, "predicted_trace": predicted, **cycle_qa.accepted_qa.accepted_runtime._trace_metrics(target, predicted)})
            rows.append({"id": str(spec["id"]), "suite": str(spec["suite"]), "length": length, "program": list(spec["program"]), "phase": phase, "arm": arm, "predictions": predictions, "cycle": {"windows": windows, "raw_loss": float(cycle_meta["raw_loss"]), "weighted_loss": float(cycle_meta["weighted_loss"]), "by_cycle": dict(cycle_meta["by_cycle"]), "target_detached": True}})
    aggregate = science._aggregate_rows(rows)
    aggregate["cycle_windows"] = cycle_windows
    aggregate["cycle_raw_loss_weighted_mean"] = cycle_raw_weighted / cycle_windows if cycle_windows else 0.0
    return rows, aggregate


def _checkpoint_payload(*, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, local_update: int, parent_identity: Mapping[str, Any], source_binding_digest: str, input_digest: str, manifest_digest: str, adapter_identity: Mapping[str, Any], adapter_initialization: Mapping[str, Any]) -> dict[str, Any]:
    cpu_copy = accepted_qa._cpu_copy
    cpu_rng, cuda_rng = accepted_qa._rng_states()
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "committed": True,
        "complete": True,
        "arm": arm,
        "architecture_id": latent.ARCHITECTURE_ID,
        "local_update": local_update,
        "absolute_update": accepted_qa.PARENT_UPDATE + local_update,
        "next_batch_index": local_update,
        "manifest_digest": manifest_digest,
        "source_binding_digest": source_binding_digest,
        "input_digest": input_digest,
        "parent_identity": dict(parent_identity),
        "adapter_identity": dict(adapter_identity),
        "adapter_initialization": dict(adapter_initialization),
        "model_state_dict": cpu_copy(model.state_dict()),
        "adapter_state_dict": cpu_copy(adapter.state_dict()),
        "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
        "optimizer_metadata": accepted_qa._optimizer_metadata(optimizer),
        "cpu_rng_state": cpu_rng,
        "cuda_rng_state": cuda_rng,
        "training_mode": bool(model.training),
        "adapter_training_mode": bool(adapter.training),
    }
    payload["model_digest"] = _digest(payload["model_state_dict"])
    payload["adapter_digest"] = _digest(payload["adapter_state_dict"])
    payload["optimizer_digest"] = _digest(payload["optimizer_state_dict"])
    payload["cpu_rng_digest"] = _digest(cpu_rng)
    payload["cuda_rng_digest"] = _digest(cuda_rng)
    payload["checkpoint_digest"] = _digest({key: value for key, value in payload.items() if key != "checkpoint_digest"})
    return payload


def _save_checkpoint(*, out: Path, model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, arm: str, parent_identity: Mapping[str, Any], source_binding_digest: str, input_digest: str, manifest_digest: str, adapter_identity: Mapping[str, Any], adapter_initialization: Mapping[str, Any], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    counter["attempted_checkpoint_saves"] += 1
    _flush(counter, sink)
    payload = _checkpoint_payload(model=model, adapter=adapter, optimizer=optimizer, arm=arm, local_update=SCREENING_UPDATES, parent_identity=parent_identity, source_binding_digest=source_binding_digest, input_digest=input_digest, manifest_digest=manifest_digest, adapter_identity=adapter_identity, adapter_initialization=adapter_initialization)
    path = out / f"checkpoint_{arm}_u{SCREENING_UPDATES}.pt"
    try:
        _atomic_torch(path, payload, refuse=True)
    except BaseException as exc:
        accepted_qa._failure(counter, "checkpoint_save", exc)
        _flush(counter, sink)
        raise
    counter["completed_checkpoint_saves"] += 1
    _flush(counter, sink)
    return {"path": _relative(path), "sha256": _sha256(path), "checkpoint_digest": payload["checkpoint_digest"], "committed": True, "complete": True, "local_update": SCREENING_UPDATES}


def _load_arm(*, inherited: Mapping[str, Any], device: torch.device, common_cuda_rng: Sequence[torch.Tensor], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], root: Path) -> tuple[torch.nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    model, optimizer, parent = accepted_qa._load_parent(manifest=inherited, counter=counter, sink=sink, root=root)
    model.to(device)
    accepted_qa._optimizer_to_device(optimizer, device)
    torch.set_rng_state(parent["rng_state"])
    torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
    model.train(True)
    parent_identity = accepted_qa._parent_identity(parent, accepted_qa._parent_checkpoint(root), accepted_qa.old.canonical_hash(inherited), common_cuda_rng)
    adapter, initialization, adapter_identity, parent_names = accepted_qa._append_adapter_group(model, optimizer, device=device)
    adapter.train(True)
    latent.validate_optimizer_adapter_association(optimizer, adapter)
    initial_identity = accepted_qa._state_identity(model, adapter, optimizer)
    return model, adapter, optimizer, parent_identity, adapter_identity, initialization, parent_names


def _run_arm(*, arm: str, batches: Sequence[Sequence[dsl.RegisterExample]], specs: Sequence[Mapping[str, Any]], inherited: Mapping[str, Any], manifest: Mapping[str, Any], source_binding: Mapping[str, Any], selection: Mapping[str, Any], device: torch.device, common_cuda_rng: Sequence[torch.Tensor], counter: dict[str, Any], sink: Callable[[dict[str, Any]], None], out: Path, root: Path) -> dict[str, Any]:
    model, adapter, optimizer, parent_identity, adapter_identity, initialization, parent_names = _load_arm(inherited=inherited, device=device, common_cuda_rng=common_cuda_rng, counter=counter, sink=sink, root=root)
    initial_identity = accepted_qa._state_identity(model, adapter, optimizer)
    initial_rows, initial_aggregate = _eval_phase(model=model, adapter=adapter, optimizer=optimizer, specs=specs, arm=arm, phase="initial", device=device, counter=counter, sink=sink)
    losses: list[float] = []
    loss_metadata: list[dict[str, Any]] = []
    for batch in batches[:SCREENING_UPDATES]:
        loss, metadata = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batch, arm=arm, device=device, counter=counter, sink=sink)
        losses.append(loss)
        loss_metadata.append(metadata)
    input_digest = str(selection["digest"])
    checkpoint = _save_checkpoint(out=out, model=model, adapter=adapter, optimizer=optimizer, arm=arm, parent_identity=parent_identity, source_binding_digest=str(source_binding["digest"]), input_digest=input_digest, manifest_digest=_digest(manifest), adapter_identity=adapter_identity, adapter_initialization=initialization, counter=counter, sink=sink)
    final_rows, final_aggregate = _eval_phase(model=model, adapter=adapter, optimizer=optimizer, specs=specs, arm=arm, phase="final", device=device, counter=counter, sink=sink)
    final_identity = accepted_qa._state_identity(model, adapter, optimizer)
    record = {
        "schema": f"{SCHEMA}_arm_v1",
        "status": "complete",
        "arm": arm,
        "label": ARM_LABELS[arm],
        "initial_identity": initial_identity,
        "final_identity": final_identity,
        "training": {"updates": SCREENING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "sum_lengths": TRAINING_SUM_LENGTHS, "losses": losses, "first_loss": losses[0], "last_loss": losses[-1], "cycle_windows": sum(int(meta["cycle"]["windows"]) for meta in loss_metadata), "cycle_updates": sum(int(meta["cycle"]["windows"]) > 0 for meta in loss_metadata), "cycle_windows_applied": sum(int(meta["cycle"]["windows"]) for meta in loss_metadata) if arm == ARM_CYCLE else 0},
        "evaluation": {"state_strata": list(EVAL_STATE_STRATA), "state_count": EVAL_STATE_COUNT, "program_count": EVAL_PROGRAM_COUNT, "initial": initial_aggregate, "final": final_aggregate},
        "evaluation_rows": {"initial": initial_rows, "final": final_rows},
        "checkpoint": checkpoint,
        "parent_identity": parent_identity,
        "adapter_identity": adapter_identity,
        "adapter_initialization": initialization,
        "parameter_names": {"parent": parent_names, "adapter": list(latent.ADAPTER_PARAMETER_NAMES)},
        "contract": {"architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID, "target": "per_slot_projected_V", "target_detached": True, "weight": contract.VALUE_CYCLE_LOSS_WEIGHT, "applied": arm == ARM_CYCLE},
    }
    _atomic_json(out / f"{arm}_report.json", record, refuse=True)
    return record


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": TOTAL_ENDPOINT_LOADS,
        "completed_endpoint_loads": TOTAL_ENDPOINT_LOADS,
        "attempted_underlying_deserializations": TOTAL_UNDERLYING_DESERIALIZATIONS,
        "completed_underlying_deserializations": TOTAL_UNDERLYING_DESERIALIZATIONS,
        "attempted_updates": TOTAL_OPTIMIZER_UPDATES,
        "completed_updates": TOTAL_OPTIMIZER_UPDATES,
        "attempted_forwards": TOTAL_FORWARDS,
        "completed_forwards": TOTAL_FORWARDS,
        "attempted_cases": TOTAL_CASES,
        "completed_cases": TOTAL_CASES,
        "attempted_readout_positions": TOTAL_POSITIONS,
        "completed_readout_positions": TOTAL_POSITIONS,
        "attempted_native_steps": TOTAL_NATIVE_STEPS,
        "completed_native_steps": TOTAL_NATIVE_STEPS,
        "attempted_backwards": TOTAL_BACKWARDS,
        "completed_backwards": TOTAL_BACKWARDS,
        "attempted_optimizer_steps": TOTAL_OPTIMIZER_UPDATES,
        "completed_optimizer_steps": TOTAL_OPTIMIZER_UPDATES,
        "attempted_checkpoint_saves": TOTAL_CHECKPOINTS,
        "completed_checkpoint_saves": TOTAL_CHECKPOINTS,
        "cycle_windows": EXPECTED_TRAINING_CYCLE_WINDOWS * 2,
        "cycle_updates": EXPECTED_TRAINING_CYCLE_UPDATES * 2,
        "cycle_windows_applied": EXPECTED_TRAINING_CYCLE_WINDOWS,
        "cycle_updates_applied": EXPECTED_TRAINING_CYCLE_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"screening accounting mismatch: {key}={counter.get(key)} expected {value}")
    if counter.get("failures"):
        raise ValueError("screening recorded failures")


def run_screening(*, root: Path = ROOT, manifest_path: Path = MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    gate = _load_launch_gate(root)
    _refuse_nonempty(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = lambda current: _atomic_json(accounting_path, current)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "launch_gate": {"path": _relative(_resolve(LAUNCH_GATE, root), root), "sha256": _sha256(_resolve(LAUNCH_GATE, root))},
        "budget": {"endpoint_loads": TOTAL_ENDPOINT_LOADS, "underlying_deserializations": TOTAL_UNDERLYING_DESERIALIZATIONS, "training_forwards": TRAINING_FORWARDS_TOTAL, "evaluation_forwards": EVAL_FORWARDS_TOTAL, "forwards": TOTAL_FORWARDS, "cases": TOTAL_CASES, "positions": TOTAL_POSITIONS, "native_steps": TOTAL_NATIVE_STEPS, "backwards": TOTAL_BACKWARDS, "optimizer_updates": TOTAL_OPTIMIZER_UPDATES, "checkpoints": TOTAL_CHECKPOINTS, "training_cycle_windows": EXPECTED_TRAINING_CYCLE_WINDOWS, "training_cycle_updates": EXPECTED_TRAINING_CYCLE_UPDATES, "evaluation_cycle_windows": EXPECTED_EVAL_CYCLE_WINDOWS},
        "arms": {},
    }
    try:
        manifest = science._load_manifest_for_science(root, manifest_path)
        scope, evidence, batches = science._load_frozen_data(manifest, root)
        device, settings = cycle_qa.accepted_qa._configure_runtime()
        prefix_digest, prefix_cycle_windows, prefix_cycle_updates = _prefix_digest(batches)
        if (prefix_cycle_windows, prefix_cycle_updates) != (EXPECTED_TRAINING_CYCLE_WINDOWS, EXPECTED_TRAINING_CYCLE_UPDATES):
            raise ValueError("registered training cycle-window count changed")
        specs, selection, selection_digest = _selection(scope)
        selection["digest"] = selection_digest
        if selection["evaluation_cycle_windows"] != EXPECTED_EVAL_CYCLE_WINDOWS:
            raise ValueError("registered evaluation cycle-window count changed")
        if gate.get("training_prefix_digest") != prefix_digest or gate.get("evaluation_selection_digest") != selection_digest:
            raise ValueError("registered screening input digest changed")
        cycle_qa_binding = cycle_qa._source_binding(root)
        source_binding = _source_binding(root=root, manifest=manifest, prefix_digest=prefix_digest, selection_digest=selection_digest, settings=settings, cycle_qa_binding=cycle_qa_binding)
        input_freeze = {"schema": INPUT_SCHEMA, "immutable": True, "manifest": _file_binding(manifest_path, root), "stream": _file_binding(_resolve(STREAM, root), root), "scope": _file_binding(_resolve(SCOPE, root), root), "source_binding_digest": source_binding["digest"], "training": {"updates_per_arm": SCREENING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": list(TRAINING_LENGTHS), "sum_lengths": TRAINING_SUM_LENGTHS, "prefix_digest": prefix_digest}, "evaluation": selection, "arms": list(ARMS), "contract": {"architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID, "target": "per_slot_projected_V", "target_detached": True, "weight": contract.VALUE_CYCLE_LOSS_WEIGHT}}
        input_freeze["digest"] = _digest(input_freeze)
        _atomic_json(out / "source_binding.json", source_binding, refuse=True)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        _atomic_json(out / "stream_prefix.json", {"schema": "pc_projected_value_cycle_screening_stream_prefix_v1", "batch_count": SCREENING_UPDATES, "batch_size": TRAINING_BATCH_SIZE, "lengths": list(TRAINING_LENGTHS), "sum_lengths": TRAINING_SUM_LENGTHS, "digest": prefix_digest, "cycle_windows": prefix_cycle_windows, "cycle_updates": prefix_cycle_updates}, refuse=True)
        with accepted_qa.migration.windows_compatibility_adapter():
            inherited = accepted_qa.followup._load_inherited(root)
            common_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
            with accepted_qa._count_deserializations(counter, sink):
                for arm in ARMS:
                    report["arms"][arm] = _run_arm(arm=arm, batches=batches[:SCREENING_UPDATES], specs=specs, inherited=inherited, manifest=manifest, source_binding=source_binding, selection=selection, device=device, common_cuda_rng=common_cuda_rng, counter=counter, sink=sink, out=out, root=root)
        _validate_accounting(counter)
        report.update({"status": "complete", "manifest": _file_binding(manifest_path, root), "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]}, "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": input_freeze["digest"]}, "stream_prefix": {"path": _relative(out / "stream_prefix.json", root), "sha256": _sha256(out / "stream_prefix.json"), "digest": prefix_digest}, "evidence": evidence, "scope": selection, "accounting": dict(counter), "limitations": ["64 fixed stream updates per arm only", "one endpoint and one seed", "small validation+test program subset", "cycle consistency is diagnostic evidence, not proof of a usable latent representation", "no superiority claim or full pilot authorization"]})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        accepted_qa._failure(counter, "screening", exc)
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_screening(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
