"""Run the bounded writer-to-cache diagnostic at the accepted local-2000 endpoint.

This runtime performs inference only.  It compares one known-good and one
known-bad padding path, records ``phi -> z -> v -> normalized_v`` and all
native-step attention weights, then evaluates a four-way h/KV substitution.
It never trains, resumes, edits a checkpoint, or adds a model pathway.
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
from scripts import pc_latent_slots as latent
from scripts import pc_learned_scratchpad as scratchpad
from scripts import pc_latent_slots_padding_diagnostic_runtime as accepted
from scripts import pc_latent_slots_science as science
from scripts import pc_state_transition_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "pc_latent_slots_v1"
SCIENCE_MANIFEST = RUN_ROOT / "science_manifest.json"
SCIENCE_REPORT = RUN_ROOT / "science" / "report.json"
SCIENCE_ENDPOINT = RUN_ROOT / "science" / "checkpoints" / "local2000.pt"
OUTPUT = RUN_ROOT / "state_transition_diagnostic_v3"
RUNTIME = ROOT / "scripts" / "pc_state_transition_diagnostic_runtime.py"
HELPER = ROOT / "scripts" / "pc_state_transition_diagnostic.py"
TESTS = ROOT / "tests" / "test_pc_state_transition_diagnostic.py"
RUNTIME_TESTS = ROOT / "tests" / "test_pc_state_transition_diagnostic_runtime.py"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_STATE_TRANSITION_DIAGNOSTIC_PROTOCOL.md"
SOURCE_GATE = RUN_ROOT / "state_transition_diagnostic_v3_review" / "runtime_launch_gate_v3.json"
SCHEMA = "pc_state_transition_diagnostic_runtime_v1"
INPUT_SCHEMA = "pc_state_transition_diagnostic_input_freeze_v1"
ENDPOINT_LOCAL_UPDATE = 2_000
ENDPOINT_ABSOLUTE_UPDATE = 42_000
EXPECTED_ENDPOINT_SHA256 = "bb7d4ea2739d97ccd9b43663657460c5f775bc8514543b71e8a14e88e03417d5"
EXPECTED_MANIFEST_SHA256 = "ef64fdd4ed8af03d7631b9809187ab1902f864325925f883277102dff45fbb0a"
EXPECTED_REPORT_SHA256 = "64f449882b10707ca3a81e1c170588bf61182385f63f5407ef71e6648164fbc0"
EXPECTED_FINAL_SHA256 = "b7444110f76dc7871280c134a752be9cbdea63fb2580dd3cf967e2247776ef55"
EXPECTED_SCOPE_SHA256 = "5cb130e2e0027eb9faf773283d5676cc067560e4cf2e7294f57b0d80afa1863e"
EXPECTED_MANIFEST_DIGEST_SHA256 = "ee9475040f18ddb75a1d621a5a09b5534df7ac2b975b46399ea9e190411ac963"
EXPECTED_REGISTER_E15_SHA256 = "bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac"


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary collision: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (path.is_file() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty diagnostic output: {path}")


def _json_tensor(value: torch.Tensor) -> list[Any]:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
        raise ValueError("diagnostic tensor is malformed or nonfinite")
    return value.detach().cpu().tolist()


def _new_counter() -> dict[str, Any]:
    return {
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
        "attempted_backwards": 0,
        "completed_backwards": 0,
        "attempted_optimizer_steps": 0,
        "completed_optimizer_steps": 0,
        "failures": [],
        "phase_counts": {
            "load": {"attempted_endpoint_loads": 0, "completed_endpoint_loads": 0, "attempted_parent_loads": 0, "completed_parent_loads": 0, "attempted_endpoint_restores": 0, "completed_endpoint_restores": 0, "attempted_underlying_deserializations": 0, "completed_underlying_deserializations": 0, "failures": 0},
            "paths": {"attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "failures": 0},
            "substitution": {"attempted_forwards": 0, "completed_forwards": 0, "attempted_cases": 0, "completed_cases": 0, "attempted_readout_positions": 0, "completed_readout_positions": 0, "attempted_native_steps": 0, "completed_native_steps": 0, "failures": 0},
        },
        "timings_seconds": {},
    }


def _sink(path: Path) -> Callable[[dict[str, Any]], None]:
    def write(counter: dict[str, Any]) -> None:
        _atomic_json(path, counter)
    return write


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None]) -> None:
    sink(counter)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    counter["failures"].append({"kind": kind, "type": type(exc).__name__, "message": str(exc)})
    counter["phase_counts"][phase]["failures"] += 1


@contextmanager
def _timed(counter: dict[str, Any], name: str, sink: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    started = time.perf_counter()
    _flush(counter, sink)
    try:
        yield
    finally:
        counter["timings_seconds"][name] = time.perf_counter() - started
        _flush(counter, sink)


def _file_binding(path: Path, root: Path = ROOT) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic source is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _validate_source_gate(root: Path = ROOT) -> dict[str, Any]:
    gate_path = _resolve(SOURCE_GATE, root)
    gate = _read_json(gate_path)
    files = gate.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("diagnostic source gate files are missing")
    expected = {
        "helper": _resolve(HELPER, root),
        "runtime": _resolve(RUNTIME, root),
        "tests": _resolve(TESTS, root),
        "runtime_tests": _resolve(RUNTIME_TESTS, root),
        "register_e15": _resolve(Path("looped_bitnet/register_e15.py"), root),
        "protocol": _resolve(PROTOCOL, root),
    }
    for label, path in expected.items():
        record = files.get(label)
        if not isinstance(record, Mapping) or record.get("sha256") != _sha256(path):
            raise ValueError(f"diagnostic source gate changed: {label}")
    if gate.get("budget") != diagnostic.DIAGNOSTIC_BUDGET:
        raise ValueError("diagnostic source gate budget changed")
    return gate


def _pair_specs(scope: Mapping[str, Any], final: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    programs = scope.get("programs")
    final_rows = final.get("rows")
    if not isinstance(programs, list) or not isinstance(final_rows, list):
        raise ValueError("saved scope/final rows are malformed")
    by_id = {str(item.get("id")): item for item in programs if isinstance(item, Mapping)}
    final_by_id = {str(item.get("id")): item for item in final_rows if isinstance(item, Mapping)}
    if set((diagnostic.GOOD_ID, diagnostic.BAD_ID)) - set(by_id) or set((diagnostic.GOOD_ID, diagnostic.BAD_ID)) - set(final_by_id):
        raise ValueError("diagnostic pair is absent from frozen scope")
    good_spec = dict(by_id[diagnostic.GOOD_ID])
    bad_spec = dict(by_id[diagnostic.BAD_ID])
    good_final = final_by_id[diagnostic.GOOD_ID]
    bad_final = final_by_id[diagnostic.BAD_ID]
    if good_spec.get("program") != list(diagnostic.GOOD_PROGRAM) or bad_spec.get("program") != list(diagnostic.BAD_PROGRAM):
        raise ValueError("diagnostic pair program binding changed")
    selected: list[dict[str, Any]] = []
    for state in diagnostic.PAIR_STATES:
        state_list = list(state)
        good_saved = next((item for item in good_final.get("predictions", []) if item.get("state") == state_list), None)
        bad_saved = next((item for item in bad_final.get("predictions", []) if item.get("state") == state_list), None)
        if not isinstance(good_saved, Mapping) or not isinstance(bad_saved, Mapping):
            raise ValueError(f"diagnostic saved state missing: {state}")
        good_example = dsl.RegisterExample(state[0], state[1], tuple(diagnostic.GOOD_PROGRAM))
        bad_example = dsl.RegisterExample(state[0], state[1], tuple(diagnostic.BAD_PROGRAM))
        good_before = list(good_example.targets[-2])
        bad_before = list(bad_example.targets[-2])
        good_after = list(good_example.targets[-1])
        bad_after = list(bad_example.targets[-1])
        if good_before != bad_before or good_after != bad_after:
            raise ValueError(f"diagnostic pair semantic state mismatch: {state}")
        if not bool(good_saved.get("full_trace_correct")) or bool(bad_saved.get("full_trace_correct")):
            raise ValueError(f"diagnostic saved good/bad expectation changed: {state}")
        selected.append({
            "state": state_list,
            "final_semantic_state": good_before,
            "target": good_after,
            "good_saved_first_error": good_saved.get("first_error"),
            "bad_saved_first_error": bad_saved.get("first_error"),
        })
    good_meta = {"id": diagnostic.GOOD_ID, "program": list(diagnostic.GOOD_PROGRAM), "final_semantic_state": selected[0]["final_semantic_state"], "final_opcode": diagnostic.FINAL_OPCODE}
    bad_meta = {"id": diagnostic.BAD_ID, "program": list(diagnostic.BAD_PROGRAM), "final_semantic_state": selected[0]["final_semantic_state"], "final_opcode": diagnostic.FINAL_OPCODE}
    diagnostic.validate_pair_metadata(good_meta, bad_meta)
    return good_spec, bad_spec, {"cases": selected, "good_saved": good_final, "bad_saved": bad_final}


def _batch_tensors(states: Sequence[Sequence[int]], program: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dsl.RegisterExample]]:
    examples = [dsl.RegisterExample(int(state[0]), int(state[1]), tuple(program)) for state in states]
    ids_x = torch.tensor([example.x for example in examples], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in examples], dtype=torch.long, device=device)
    bits = scratchpad.signed_bit_matrix(device=device)
    x_bits = bits[ids_x]
    y_bits = bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in examples], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, examples


def _record_attention_hook(module: Any, args: tuple[Any, ...], sink: list[torch.Tensor]) -> None:
    if len(args) != 2 or not isinstance(args[0], torch.Tensor) or not isinstance(args[1], tuple) or len(args[1]) != 2:
        raise ValueError("attention reader hook received malformed arguments")
    state, cache = args
    keys, _values = cache
    if not hasattr(module, "q_norm") or not hasattr(module, "q_proj"):
        raise ValueError("attention reader lacks query projections")
    query = module.q_proj(module.q_norm(state)).view(-1, module.num_heads, 1, module.head_dim)
    scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)) / (module.head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1).squeeze(-2)
    if tuple(weights.shape[-1:]) != (diagnostic.SLOT_COUNT,) or not bool(torch.isfinite(weights).all().item()):
        raise ValueError("attention weights are malformed or nonfinite")
    sink.append(weights.detach().cpu().clone())


def _path_run(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, *, identifier: str, program: Sequence[str], states: Sequence[Sequence[int]], device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    x_bits, y_bits, op_ids, examples = _batch_tensors(states, program, device)
    slots = latent.initialize_slots(adapter, x_bits, y_bits)
    records: list[dict[str, Any]] = []
    final_internal: dict[str, torch.Tensor] = {}
    attention_capture: list[torch.Tensor] = []
    hook = model.reader.register_forward_pre_hook(lambda module, args: _record_attention_hook(module, args, attention_capture))
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for position, opcode_ids in enumerate(op_ids.T, start=1):
                z = slots.detach().clone()
                cache = latent.fresh_cache_from_slots(model, adapter, slots)
                v = adapter.reader(z)
                h = v[:, 0] + v[:, 1]
                normalized_v = model.memory_norm(v)
                keys, projected_values = cache["kv"]
                if not torch.allclose(h, cache["h"], atol=0.0, rtol=0.0):
                    raise ValueError("fresh cache h differs from captured slot sum")
                attention_capture.clear()
                (logits_x, logits_y), returned_cache = model.step(cache, opcode_ids)
                if len(attention_capture) != diagnostic.NATIVE_STEPS:
                    raise ValueError("native attention capture count changed")
                returned_h = returned_cache.get("h")
                if not isinstance(returned_h, torch.Tensor):
                    raise ValueError("native step did not return h")
                phi = model.output_norm(returned_h)
                writer_flat = adapter.writer(phi)
                next_slots = writer_flat.reshape(z.shape[0], diagnostic.SLOT_COUNT, diagnostic.SLOT_WIDTH)
                if not torch.isfinite(next_slots).all():
                    raise ValueError("writer output is nonfinite")
                rows = []
                for state_index, state in enumerate(states):
                    rows.append({
                        "state": list(state),
                        "z": _json_tensor(z[state_index]),
                        "h": _json_tensor(h[state_index]),
                        "v": _json_tensor(v[state_index]),
                        "normalized_v": _json_tensor(normalized_v[state_index]),
                        "projected_keys": _json_tensor(keys[state_index]),
                        "projected_values": _json_tensor(projected_values[state_index]),
                        "returned_h": _json_tensor(returned_h[state_index]),
                        "phi": _json_tensor(phi[state_index]),
                        "writer": _json_tensor(next_slots[state_index]),
                        "logits_x": _json_tensor(logits_x[state_index]),
                        "logits_y": _json_tensor(logits_y[state_index]),
                        "attention_weights": _json_tensor(torch.stack(attention_capture, dim=0)[:, state_index]),
                        "pre_substeps": int(cache["substeps"]),
                        "opcode": str(program[position - 1]),
                        "position": position,
                    })
                records.append({"position": position, "opcode": str(program[position - 1]), "pre_substeps": int(cache["substeps"]), "states": rows})
                slots = next_slots
                if position == len(program):
                    final_internal = {
                        "z": z.detach().clone(),
                        "h": h.detach().clone(),
                        "v": v.detach().clone(),
                        "normalized_v": normalized_v.detach().clone(),
                        "keys": keys.detach().clone(),
                        "values": projected_values.detach().clone(),
                        "phi": phi.detach().clone(),
                        "logits_x": logits_x.detach().clone(),
                        "logits_y": logits_y.detach().clone(),
                        "pre_substeps": torch.tensor(int(cache["substeps"]), dtype=torch.long),
                        "attention": torch.stack(attention_capture, dim=0).detach().clone(),
                    }
    finally:
        hook.remove()
    predictions: list[dict[str, Any]] = []
    final_row = records[-1]
    final_targets = [list(example.targets[-1]) for example in examples]
    for state_index, state in enumerate(states):
        row = final_row["states"][state_index]
        pred = [int(torch.tensor(row["logits_x"]).argmax().item()), int(torch.tensor(row["logits_y"]).argmax().item())]
        target = final_targets[state_index]
        predictions.append({"state": list(state), "target": target, "predicted": pred, "correct": pred == target})
    record = {"id": identifier, "program": list(program), "length": len(program), "states": list(states), "steps": records, "final_predictions": predictions}
    return record, final_internal


def _substitution_run(model: torch.nn.Module, good: Mapping[str, torch.Tensor], bad: Mapping[str, torch.Tensor], cases: Sequence[Mapping[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    good_h, bad_h = good["h"], bad["h"]
    good_keys, bad_keys = good["keys"], bad["keys"]
    good_values, bad_values = good["values"], bad["values"]
    good_step = int(good["pre_substeps"].item())
    bad_step = int(bad["pre_substeps"].item())
    if good_step % len(model.blocks) != bad_step % len(model.blocks):
        raise ValueError("good/bad final operation has different block phase")
    h_parts: list[torch.Tensor] = []
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    labels = diagnostic.SUBSTITUTION_LABELS
    for label in labels:
        h_parts.append(good_h if label.startswith("good_h") else bad_h)
        key_parts.append(good_keys if label.endswith("good_kv") else bad_keys)
        value_parts.append(good_values if label.endswith("good_kv") else bad_values)
    h_batch = torch.cat(h_parts, dim=0).to(device=device)
    key_batch = torch.cat(key_parts, dim=0).to(device=device)
    value_batch = torch.cat(value_parts, dim=0).to(device=device)
    op = torch.full((h_batch.shape[0],), dsl.OP_TO_ID[diagnostic.FINAL_OPCODE], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        (logits_x, logits_y), _returned = model.step({"h": h_batch, "kv": (key_batch, value_batch), "substeps": good_step}, op)
    outcomes: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        for state_index, case in enumerate(cases):
            index = label_index * len(cases) + state_index
            predicted = [int(logits_x[index].argmax().item()), int(logits_y[index].argmax().item())]
            target = list(case["target"])
            outcomes.append({
                "state": list(case["state"]),
                "label": label,
                "predicted": predicted,
                "target": target,
                "correct": predicted == target,
                "final_logits_finite": bool(torch.isfinite(logits_x[index]).all().item() and torch.isfinite(logits_y[index]).all().item()),
                "logits_x": _json_tensor(logits_x[index]),
                "logits_y": _json_tensor(logits_y[index]),
            })
    return diagnostic.validate_substitution_outcomes(outcomes)


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = diagnostic.DIAGNOSTIC_BUDGET
    phase_paths = counter["phase_counts"]["paths"]
    phase_sub = counter["phase_counts"]["substitution"]
    pairs = {
        "attempted_forwards": expected["path_forwards"],
        "completed_forwards": expected["path_forwards"],
        "attempted_cases": expected["path_cases"],
        "completed_cases": expected["path_cases"],
        "attempted_readout_positions": expected["path_positions"],
        "completed_readout_positions": expected["path_positions"],
        "attempted_native_steps": expected["path_native_steps"],
        "completed_native_steps": expected["path_native_steps"],
    }
    for key, value in pairs.items():
        if phase_paths.get(key) != value:
            raise ValueError(f"path accounting mismatch: {key}")
    substitutions = {
        "attempted_forwards": expected["substitution_forwards"],
        "completed_forwards": expected["substitution_forwards"],
        "attempted_cases": expected["substitution_cases"],
        "completed_cases": expected["substitution_cases"],
        "attempted_readout_positions": expected["substitution_positions"],
        "completed_readout_positions": expected["substitution_positions"],
        "attempted_native_steps": expected["substitution_native_steps"],
        "completed_native_steps": expected["substitution_native_steps"],
    }
    for key, value in substitutions.items():
        if phase_sub.get(key) != value:
            raise ValueError(f"substitution accounting mismatch: {key}")
    if counter.get("failures") or counter.get("attempted_backwards") or counter.get("completed_backwards") or counter.get("attempted_optimizer_steps") or counter.get("completed_optimizer_steps"):
        raise ValueError("diagnostic performed unexpected training work or recorded failures")


def _account_phase(counter: dict[str, Any], phase: str, *, cases: int, positions: int, native_steps: int, sink: Callable[[dict[str, Any]], None]) -> None:
    target = counter["phase_counts"][phase]
    for key, value in (("attempted_forwards", 1), ("attempted_cases", cases), ("attempted_readout_positions", positions), ("attempted_native_steps", native_steps)):
        counter[key] += value
        target[key] += value
    _flush(counter, sink)


def _complete_phase(counter: dict[str, Any], phase: str, *, cases: int, positions: int, native_steps: int, sink: Callable[[dict[str, Any]], None]) -> None:
    target = counter["phase_counts"][phase]
    for key, value in (("completed_forwards", 1), ("completed_cases", cases), ("completed_readout_positions", positions), ("completed_native_steps", native_steps)):
        counter[key] += value
        target[key] += value
    _flush(counter, sink)


@contextmanager
def _preserve_eval_state(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, optimizer: torch.optim.Optimizer, *, device: torch.device) -> Iterator[None]:
    with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
        model.eval()
        adapter.eval()
        yield


def run_diagnostic(*, root: Path = ROOT, out: Path = OUTPUT, manifest_path: Path = SCIENCE_MANIFEST) -> dict[str, Any]:
    root = Path(root)
    out = _resolve(out, root)
    manifest_path = _resolve(manifest_path, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = _sink(accounting_path)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "budget": diagnostic.DIAGNOSTIC_BUDGET, "accounting": counter}
    model: torch.nn.Module | None = None
    adapter: latent.LatentSlotAdapter | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        gate = _validate_source_gate(root)
        manifest, science_report, scope, final, evidence = accepted._validate_evidence(root, manifest_path)
        device, settings = accepted._load_runtime_settings(manifest)
        good_spec, bad_spec, pair = _pair_specs(scope, final)
        input_freeze = {
            "schema": INPUT_SCHEMA,
            "immutable": True,
            "source_gate": {"path": _relative(_resolve(SOURCE_GATE, root), root), "sha256": _sha256(_resolve(SOURCE_GATE, root))},
            "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
            "endpoint": evidence["endpoint"],
            "evidence": evidence,
            "pair": {"good": diagnostic.GOOD_ID, "bad": diagnostic.BAD_ID, "states": [list(state) for state in diagnostic.PAIR_STATES], "good_program": list(diagnostic.GOOD_PROGRAM), "bad_program": list(diagnostic.BAD_PROGRAM)},
            "runtime": settings,
            "semantics": {"native_steps": diagnostic.NATIVE_STEPS, "attention_weights": "pre-hook recomputation from actual q_norm/q_proj and cached keys", "writer_input": "output_norm(returned_h)", "substitution": "diagnostic cache h/KV replacement only; no architecture change"},
        }
        input_freeze["digest"] = science._digest(input_freeze)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report.update({"manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)}, "endpoint": evidence["endpoint"], "pair": {"good_id": diagnostic.GOOD_ID, "bad_id": diagnostic.BAD_ID, "states": pair["cases"]}, "source_gate": gate, "runtime": settings})
        with accepted._timed(counter, "endpoint_load", sink):
            model, adapter, optimizer, _payload = accepted._load_endpoint(manifest=manifest, report=science_report, endpoint=_resolve(SCIENCE_ENDPOINT, root), manifest_path=manifest_path, source_binding_digest=str(manifest["source_inventory"]["digest"]), device=device, counter=counter, sink=sink, root=root)
        if counter["completed_endpoint_loads"] != 1:
            raise ValueError("endpoint did not load exactly once")
        states = [item["state"] for item in pair["cases"]]
        path_records: dict[str, Any] = {}
        internals: dict[str, dict[str, torch.Tensor]] = {}
        with _preserve_eval_state(model, adapter, optimizer, device=device):
            for phase, identifier, spec in (("good", diagnostic.GOOD_ID, good_spec), ("bad", diagnostic.BAD_ID, bad_spec)):
                program = tuple(str(opcode) for opcode in spec["program"])
                positions = len(program) * len(states)
                native = positions * diagnostic.NATIVE_STEPS
                _account_phase(counter, "paths", cases=len(states), positions=positions, native_steps=native, sink=sink)
                try:
                    with _timed(counter, f"path_{phase}", sink):
                        record, internal = _path_run(model, adapter, identifier=identifier, program=program, states=states, device=device)
                    path_records[phase] = record
                    internals[phase] = internal
                    _complete_phase(counter, "paths", cases=len(states), positions=positions, native_steps=native, sink=sink)
                except BaseException as exc:
                    _failure(counter, f"path_{phase}", exc, phase="paths")
                    _flush(counter, sink)
                    raise
            substitutions_cases = len(states) * len(diagnostic.SUBSTITUTION_LABELS)
            _account_phase(counter, "substitution", cases=substitutions_cases, positions=substitutions_cases, native_steps=substitutions_cases * diagnostic.NATIVE_STEPS, sink=sink)
            try:
                with _timed(counter, "substitution", sink):
                    outcomes = _substitution_run(model, internals["good"], internals["bad"], pair["cases"], device)
                _complete_phase(counter, "substitution", cases=substitutions_cases, positions=substitutions_cases, native_steps=substitutions_cases * diagnostic.NATIVE_STEPS, sink=sink)
            except BaseException as exc:
                _failure(counter, "substitution", exc, phase="substitution")
                _flush(counter, sink)
                raise
            good_final = internals["good"]
            bad_final = internals["bad"]
            pair_summary = {
                "z": diagnostic.pair_distance(good_final["z"], bad_final["z"], label="z"),
                "h": diagnostic.pair_distance(good_final["h"], bad_final["h"], label="h"),
                "v": diagnostic.pair_distance(good_final["v"], bad_final["v"], label="v"),
                "normalized_v": diagnostic.pair_distance(good_final["normalized_v"], bad_final["normalized_v"], label="normalized_v"),
                "projected_values": diagnostic.pair_distance(good_final["values"], bad_final["values"], label="projected_values"),
                "phi": diagnostic.pair_distance(good_final["phi"], bad_final["phi"], label="phi"),
                "attention_good": diagnostic.attention_summary(good_final["attention"].float(), label="good_attention"),
                "attention_bad": diagnostic.attention_summary(bad_final["attention"].float(), label="bad_attention"),
                "head_writer_sensitivity": diagnostic.projection_summary(good_final["phi"], bad_final["phi"], x_head=model.x_head.weight, y_head=model.y_head.weight, writer_weight=adapter.writer.weight, writer_bias=adapter.writer.bias),
            }
        _validate_accounting(counter)
        _atomic_json(out / "good_path.json", path_records["good"], refuse=True)
        _atomic_json(out / "bad_path.json", path_records["bad"], refuse=True)
        _atomic_json(out / "substitutions.json", {"outcomes": outcomes}, refuse=True)
        report.update({"status": "complete", "accounting": counter, "paths": {"good": "good_path.json", "bad": "bad_path.json"}, "substitutions": "substitutions.json", "pair": {"good_id": diagnostic.GOOD_ID, "bad_id": diagnostic.BAD_ID, "states": pair["cases"]}, "pair_summary": pair_summary, "interpretation": {"causal_claim": False, "next_step": "Use the substitution outcomes to distinguish writer, memory preparation and joint h/KV dependence; do not treat any one result as proof of LayerNorm causality."}})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        _failure(counter, "diagnostic", exc, phase="load" if model is None else "paths")
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": counter, "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise
    finally:
        model = None
        adapter = None
        optimizer = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    args = parser.parse_args(argv)
    run_diagnostic(out=args.out, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
