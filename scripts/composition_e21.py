#!/usr/bin/env python3
"""E21 one-pass evaluation of six frozen novel compositions plus one control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import longer_native8_e20 as e20  # noqa: E402
from looped_bitnet.register_e15 import (  # noqa: E402
    OPS,
    STATE_ORDER,
    FROZEN_SOURCE_RELATIVE_PATHS,
    all_programs,
    evaluate_program,
    execute_program,
    forbidden,
    semantic_signature,
    sha256_file,
    state_split,
)
from scripts.longer_native8_e20 import load_manifest as load_e20_manifest  # noqa: E402
from scripts.step_budget_e18 import _atomic_json, _read_json  # noqa: E402


E21_SCHEMA = "e21_composition_v1"
DEFAULT_PREFLIGHT = Path("runs/e21_composition_preflight")
DEFAULT_RUN = Path("runs/e21_composition")
PROTECTED_SNAPSHOT = Path("results/E21_PROTECTED_HASHES.json")
SYMBOLIC_REFERENCE = Path("results/E21_SYMBOLIC_REFERENCE.json")
EXPECTED_SYMBOLIC_REFERENCE_SHA256 = "bc929003024ea3a482d518286f7f652efe3f93a85a1d68c47a04265ee45a9a14"
E20_CHECKPOINT = Path("runs/e20_longer_native8/u8000.pt")
E20_MANIFEST = Path("runs/e20_longer_native8/manifest.json")
E20_REPORT = Path("runs/e20_longer_native8/report.json")
E15_PREFLIGHT = Path("runs/register_e15_preflight/v7/manifest.json")
SEED = 0
NATIVE_STEPS = 8
PARAMETER_COUNT = 151232
PRIMARY_THRESHOLD = 244
PRIMARY_PROGRAMS: tuple[tuple[str, ...], ...] = (
    ("ADD", "XOR"),
    ("ADD", "ADD", "XOR"),
    ("ADD", "XOR", "ADD"),
    ("ADD", "XOR", "SWAP"),
    ("XOR", "ADD", "XOR"),
    ("SWAP", "ADD", "XOR"),
)
CONTROL_PROGRAM = ("ADD", "XOR", "XOR")
ALL_PROGRAMS = PRIMARY_PROGRAMS + (CONTROL_PROGRAM,)
EXPECTED_COST = {
    "training_updates": 0,
    "primary_program_state_cases": 1536,
    "primary_readout_positions": 4352,
    "primary_internal_state_substeps": 34816,
    "control_program_state_cases": 256,
    "control_readout_positions": 768,
    "control_internal_state_substeps": 6144,
    "program_state_cases": 1792,
    "readout_positions": 5120,
    "internal_state_substeps": 40960,
}
SOURCE_RELATIVE_PATHS = {
    "e21_runner": "scripts/composition_e21.py",
    "e21_tests": "tests/test_composition_e21.py",
    "e21_protocol": "results/E21_COMPOSITION_PROTOCOL.md",
    "e20_module": "looped_bitnet/longer_native8_e20.py",
    "e20_runner": "scripts/longer_native8_e20.py",
    "e20_tests": "tests/test_longer_native8_e20.py",
    "e20_protocol": "results/E20_LONGER_NATIVE8_PROTOCOL.md",
    "e18_module": "looped_bitnet/step_budget_e18.py",
    "e18_runner": "scripts/step_budget_e18.py",
    "e17_module": "looped_bitnet/bit_input_e17.py",
    "e16_module": "looped_bitnet/float_qat_e16.py",
    "e15_module": "looped_bitnet/register_e15.py",
    "model": "looped_bitnet/model.py",
    "quantization": "looped_bitnet/quantization.py",
    "runtime": "looped_bitnet/runtime.py",
    "config": "looped_bitnet/config.py",
    "data": "looped_bitnet/data.py",
    "engine": "looped_bitnet/engine.py",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def source_hashes(root: Path = e20.PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / relative) for name, relative in SOURCE_RELATIVE_PATHS.items()}


def verify_protected_hashes(root: Path = e20.PROJECT_ROOT, snapshot: Path = PROTECTED_SNAPSHOT) -> dict[str, str]:
    root = Path(root); snapshot = Path(snapshot)
    if not snapshot.is_absolute(): snapshot = root / snapshot
    hashes = _read_json(snapshot).get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 195:
        raise ValueError("E21 protected snapshot must contain exactly 195 files")
    for relative, expected in hashes.items():
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"E21 protected hash mismatch: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def _ordered_states() -> list[tuple[int, int]]:
    states = list(STATE_ORDER)
    if states != [(x, y) for x in range(16) for y in range(16)]:
        raise ValueError("E21 state order is not lexicographic 0..15 by x then y")
    return states


def _strata(root: Path = e20.PROJECT_ROOT) -> dict[str, list[tuple[int, int]]]:
    states = _ordered_states(); split = state_split()
    frozen = _read_json(Path(root) / E15_PREFLIGHT)["state_split"]
    if frozen != {name: [list(state) for state in values] for name, values in split.items()}:
        raise ValueError("E21 E15 frozen manifest state split changed")
    parts = {name: [tuple(state) for state in frozen[name]] for name in ("train", "validation", "test")}
    if set(parts) != {"train", "validation", "test"} or {name: len(values) for name, values in parts.items()} != {"train": 192, "validation": 32, "test": 32}:
        raise ValueError("E21 E15 state strata have wrong sizes")
    flattened = [state for name in ("train", "validation", "test") for state in parts[name]]
    if len(set(flattened)) != 256 or set(flattened) != set(states):
        raise ValueError("E21 E15 state strata are not disjoint and exhaustive")
    return parts


def _target_trace(program: Sequence[str], state: tuple[int, int]) -> list[list[int]]:
    current = state; trace = []
    for opcode in program:
        current = execute_program((opcode,), current)
        trace.append(list(current))
    return trace


def symbolic_audit() -> dict[str, Any]:
    states = _ordered_states(); strata = _strata()
    seen = [tuple(program) for program in all_programs() if not forbidden(program)]
    forbidden_programs = {tuple(program) for program in all_programs() if forbidden(program)}
    expected_forbidden = set(ALL_PROGRAMS)
    if forbidden_programs != expected_forbidden:
        raise ValueError("E21 forbidden set is not exactly six primary programs plus control")
    seen_signatures = {semantic_signature(program) for program in seen}
    primary_signatures = [semantic_signature(program) for program in PRIMARY_PROGRAMS]
    if len(set(primary_signatures)) != 6 or any(signature in seen_signatures for signature in primary_signatures):
        raise ValueError("E21 primary semantic novelty audit failed")
    if semantic_signature(CONTROL_PROGRAM) != semantic_signature(("ADD",)):
        raise ValueError("E21 control is not symbolically equivalent to ADD")
    programs = [{"name": f"primary_{index + 1}", "program": list(program), "role": "primary"}
                for index, program in enumerate(PRIMARY_PROGRAMS)]
    programs.append({"name": "control_add_xor_xor", "program": list(CONTROL_PROGRAM), "role": "control"})
    targets = {
        item["name"]: [{"state": list(state), "target_trace": _target_trace(item["program"], state)} for state in states]
        for item in programs
    }
    symbolic = {
        "programs": programs,
        "states": [list(state) for state in states],
        "strata": {name: [list(state) for state in values] for name, values in strata.items()},
        "targets": targets,
        "program_digest": _digest(programs),
        "state_digest": _digest([list(state) for state in states]),
        "strata_digest": _digest({name: [list(state) for state in values] for name, values in strata.items()}),
        "target_digest": _digest(targets),
        "semantic_signature_digest": _digest({"seen": [_digest(list(signature)) for signature in sorted(seen_signatures)],
                                               "primary": [_digest(list(signature)) for signature in primary_signatures],
                                               "control": _digest(list(semantic_signature(CONTROL_PROGRAM)))}),
        "forbidden_digest": _digest([list(program) for program in sorted(forbidden_programs)]),
    }
    reference = _read_json(e20.PROJECT_ROOT / SYMBOLIC_REFERENCE)
    if reference.get("primary_programs") != 6 or reference.get("seen_programs") != 32 or reference.get("primary_cases") != 1536:
        raise ValueError("E21 independent symbolic reference counts changed")
    if [tuple(item["program"]) for item in reference.get("primary", [])] != list(PRIMARY_PROGRAMS):
        raise ValueError("E21 independent symbolic primary order changed")
    if reference.get("control", {}).get("program") != list(CONTROL_PROGRAM) or reference.get("control", {}).get("equivalent_to") != ["ADD"]:
        raise ValueError("E21 independent symbolic control changed")
    if reference.get("strata") != {name: len(values) for name, values in strata.items()}:
        raise ValueError("E21 independent symbolic strata changed")
    if reference.get("evaluation_cost") != {"program_state_cases": 1792, "instruction_readouts": 5120, "native8_substeps": 40960}:
        raise ValueError("E21 independent symbolic cost changed")
    symbolic["reference_sha256"] = sha256_file(e20.PROJECT_ROOT / SYMBOLIC_REFERENCE)
    if symbolic["reference_sha256"] != EXPECTED_SYMBOLIC_REFERENCE_SHA256:
        raise ValueError("E21 independent symbolic reference hash changed")
    symbolic["reference"] = reference
    return symbolic


def _e15_manifest() -> dict[str, Any]:
    return _read_json(e20.PROJECT_ROOT / E15_PREFLIGHT)


def _load_frozen_e20() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    manifest = load_e20_manifest(e20.DEFAULT_E20_PREFLIGHT)
    checkpoint_path = e20.PROJECT_ROOT / E20_CHECKPOINT
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fresh, _, _ = e20.build_initial_model()
    raw_state = raw.get("state_dict")
    expected_state = fresh.state_dict()
    if not isinstance(raw_state, dict) or list(raw_state) != list(expected_state):
        raise ValueError("E21 E20 raw checkpoint state keys changed")
    if raw.get("params") != PARAMETER_COUNT or len(raw_state) != len(expected_state):
        raise ValueError("E21 E20 raw checkpoint parameter inventory changed")
    for name, expected in expected_state.items():
        actual = raw_state[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"E21 E20 raw checkpoint tensor inventory changed: {name}")
    model, _, payload = e20.load_checkpoint(checkpoint_path, manifest, expected_update=8000)
    if payload.get("schema") != e20.E20_SCHEMA or payload.get("arm") != "steps8" or payload.get("seed") != SEED:
        raise ValueError("E20 checkpoint identity mismatch")
    if payload.get("update") != 8000 or payload.get("params") != PARAMETER_COUNT or payload.get("native_steps") != NATIVE_STEPS:
        raise ValueError("E20 checkpoint native8/u8000 provenance mismatch")
    return manifest, model, payload


def _static_manifest(root: Path = e20.PROJECT_ROOT) -> dict[str, Any]:
    e20_manifest, model, payload = _load_frozen_e20()
    symbolic = symbolic_audit()
    e15 = _e15_manifest()
    e18_preflight = _read_json(root / e20.DEFAULT_E18_PREFLIGHT / "manifest.json")
    checkpoint_path = root / E20_CHECKPOINT; report_path = root / E20_REPORT; manifest_path = root / E20_MANIFEST
    return {
        "format_version": 1, "schema": E21_SCHEMA, "experiment": "composition_e21", "seed": SEED,
        "native_steps": NATIVE_STEPS, "parameter_count": PARAMETER_COUNT, "encoder_mode": "bits",
        "source_hashes": source_hashes(root), "e20_source_hashes": e20_manifest["source_hashes"],
        "protocol_hash": sha256_file(root / "results/E21_COMPOSITION_PROTOCOL.md"),
        "config": {"primary_programs": [list(p) for p in PRIMARY_PROGRAMS], "control_program": list(CONTROL_PROGRAM),
                   "primary_threshold": PRIMARY_THRESHOLD, "state_order": "lexicographic_x_then_y",
                   "evaluation_mode": "one_256_state_forward_per_program", "autocast": False},
        "config_hash": _digest({"primary_programs": [list(p) for p in PRIMARY_PROGRAMS], "control_program": list(CONTROL_PROGRAM),
                                 "primary_threshold": PRIMARY_THRESHOLD, "state_order": "lexicographic_x_then_y",
                                 "evaluation_mode": "one_256_state_forward_per_program", "autocast": False}),
        "e20_manifest_sha256": sha256_file(manifest_path), "e20_manifest_canonical_hash": e20.canonical_hash(e20_manifest),
        "e20_checkpoint_sha256": sha256_file(checkpoint_path), "e20_report_sha256": sha256_file(report_path),
        "e20_model_digest": payload["model_digest"], "e20_initial_digest": e20_manifest["initial_digest"],
        "e20_checkpoint_config_hash": payload["config_hash"], "e20_checkpoint_protocol_hash": payload["protocol_hash"],
        "e15_manifest_hash": e18_preflight["e15_manifest_hash"],
        "e15_manifest_file_sha256": sha256_file(root / E15_PREFLIGHT),
        "symbolic_reference_sha256": symbolic["reference_sha256"], "symbolic": {key: value for key, value in symbolic.items() if key != "reference"},
        "cost": dict(EXPECTED_COST),
        "scope": {"primary_programs": [list(p) for p in PRIMARY_PROGRAMS], "control_program": list(CONTROL_PROGRAM),
                  "states": 256, "strata": {name: len(values) for name, values in _strata().items()}},
        "e20_checkpoint_path": str(E20_CHECKPOINT), "e20_manifest_path": str(E20_MANIFEST),
        "e20_report_path": str(E20_REPORT), "e15_preflight_path": str(E15_PREFLIGHT),
        "model_inventory": {"class": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
                            "state_keys": list(model.state_dict())},
    }


def preflight(path: Path = DEFAULT_PREFLIGHT, *, root: Path = e20.PROJECT_ROOT) -> Path:
    path = Path(path); path = path if path.is_absolute() else Path(root) / path
    e20.refuse_nonempty(path); verify_protected_hashes(root)
    manifest = _static_manifest(root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    verify_protected_hashes(root)
    return path / "manifest.json"


def load_manifest(path: Path = DEFAULT_PREFLIGHT, *, root: Path = e20.PROJECT_ROOT) -> dict[str, Any]:
    path = Path(path); path = path if path.is_absolute() else Path(root) / path
    manifest = _read_json(path / "manifest.json")
    expected = _static_manifest(root)
    if manifest != expected:
        raise ValueError("E21 frozen manifest or imported reference changed")
    symbolic = manifest["symbolic"]
    if symbolic["reference_sha256"] != sha256_file(Path(root) / SYMBOLIC_REFERENCE):
        raise ValueError("E21 symbolic reference changed")
    return manifest


def _metric_for_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(predictions)
    if denominator == 0:
        raise ValueError("E21 metric slice is empty")
    length = len(predictions[0]["target_trace"])
    final_joint = sum(bool(item["joint_final_correct"]) for item in predictions)
    final_x = sum(item["predicted_trace"][-1][0] == item["target_trace"][-1][0] for item in predictions)
    final_y = sum(item["predicted_trace"][-1][1] == item["target_trace"][-1][1] for item in predictions)
    prefix_joint = [sum(bool(item["prefix_joint_correct"][index]) for item in predictions) for index in range(length)]
    full_trace = sum(all(bool(value) for value in item["prefix_joint_correct"]) for item in predictions)
    if any(full_trace > value for value in prefix_joint):
        raise ValueError("E21 full-trace count exceeds prefix count")
    if any(value > denominator for value in prefix_joint) or final_joint > denominator or final_x > denominator or final_y > denominator:
        raise ValueError("E21 metric count exceeds denominator")
    return {"denominator": denominator, "final_joint": int(final_joint), "final_x": int(final_x),
            "final_y": int(final_y), "prefix_joint": [int(value) for value in prefix_joint], "full_trace": int(full_trace)}


def score_rows(rows: Sequence[Mapping[str, Any]], strata: Mapping[str, Sequence[tuple[int, int]]]) -> dict[str, Any]:
    if len(rows) != 7:
        raise ValueError("E21 report requires six primary rows plus one control row")
    state_to_stratum = {tuple(state): name for name, states in strata.items() for state in states}
    result = {}
    for row in rows:
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 256:
            raise ValueError("E21 row does not contain exactly 256 predictions")
        if [tuple(item["state"]) for item in predictions] != list(STATE_ORDER):
            raise ValueError("E21 prediction states are not in frozen order")
        for item in predictions:
            if "stratum" in item and item.get("stratum") != state_to_stratum[tuple(item["state"])] :
                raise ValueError("E21 prediction stratum mismatch")
        slices = {"all": predictions}
        slices.update({name: [item for item in predictions if tuple(item["state"]) in set(states)] for name, states in strata.items()})
        metrics = {name: _metric_for_predictions(values) for name, values in slices.items()}
        for name in ("final_joint", "final_x", "final_y", "full_trace"):
            if metrics["all"][name] != sum(metrics[stratum][name] for stratum in strata):
                raise ValueError("E21 stratum totals do not equal all-state totals")
        for index, value in enumerate(metrics["all"]["prefix_joint"]):
            if value != sum(metrics[stratum]["prefix_joint"][index] for stratum in strata):
                raise ValueError("E21 prefix stratum totals do not equal all-state totals")
        result[tuple(row["program"])] = {"program": list(row["program"]), "role": row["role"], "metrics": metrics,
                                          "predictions": predictions}
    return result


def _attach_strata(row: Mapping[str, Any], strata: Mapping[str, Sequence[tuple[int, int]]]) -> dict[str, Any]:
    membership = {tuple(state): name for name, states in strata.items() for state in states}
    cloned = dict(row); predictions = []
    for item in row["predictions"]:
        value = dict(item); value["stratum"] = membership[tuple(value["state"])]
        predictions.append(value)
    cloned["predictions"] = predictions
    return cloned


def evaluate_model(model: Any, manifest: Mapping[str, Any], *, programs: Sequence[tuple[str, ...]] = ALL_PROGRAMS) -> dict[str, Any]:
    selected = list(programs)
    expected_count = 7
    if len(selected) != expected_count or len(set(selected)) != expected_count:
        raise ValueError("E21 evaluation program scope changed")
    strata = _strata(); states = _ordered_states(); calls = {"forward": 0}
    hook = model.register_forward_hook(lambda *args: calls.__setitem__("forward", calls["forward"] + 1))
    before = e20.digest_state_dict(model)
    was_training = model.training; model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            rows = []
            for program in selected:
                raw = evaluate_program(model, program, states, include_predictions=True)
                raw["role"] = "control" if tuple(program) == CONTROL_PROGRAM else "primary"
                expected_name = next((item["name"] for item in manifest["symbolic"]["programs"]
                                      if item["program"] == list(program)), None)
                if expected_name is not None:
                    expected_targets = manifest["symbolic"]["targets"][expected_name]
                    for prediction, expected in zip(raw["predictions"], expected_targets):
                        if prediction["state"] != expected["state"] or prediction["target_trace"] != expected["target_trace"]:
                            raise ValueError("E21 model row target trace differs from frozen symbolic trace")
                rows.append(_attach_strata(raw, strata))
    finally:
        hook.remove(); model.train(was_training)
    if calls["forward"] != len(selected):
        raise ValueError(f"E21 expected one forward per program, got {calls['forward']}")
    if e20.digest_state_dict(model) != before:
        raise ValueError("E21 evaluation changed model parameters")
    scored = score_rows(rows, strata)
    has_scientific_scope = all(program in scored for program in PRIMARY_PROGRAMS) and CONTROL_PROGRAM in scored
    if has_scientific_scope:
        primary = [scored[program] for program in PRIMARY_PROGRAMS]
        control = scored[CONTROL_PROGRAM]
        predicates = {" ".join(program): row["metrics"]["all"]["final_joint"] >= PRIMARY_THRESHOLD
                      for program, row in zip(PRIMARY_PROGRAMS, primary)}
        conjunction = all(predicates.values())
    else:
        predicates, conjunction, control = {}, None, None
    return {"rows": rows, "scored": scored, "primary_predicates": predicates,
            "primary_conjunction": conjunction, "control": control,
            "forward_passes": calls["forward"]}


def _report(result: Mapping[str, Any], *, qa: bool = False) -> dict[str, Any]:
    if qa:
        programs = [tuple(row["program"]) for row in result["rows"]]
        cost = {"training_updates": 0, "program_state_cases": len(programs) * 256,
                "readout_positions": sum(len(program) for program in programs) * 256,
                "internal_state_substeps": sum(len(program) for program in programs) * 256 * NATIVE_STEPS}
        return {"schema": E21_SCHEMA, "status": "qa_complete", "qa": True,
                "programs": [list(program) for program in programs], "rows": result["rows"],
                "scored": {" ".join(program): value for program, value in result["scored"].items()},
                "forward_passes": result["forward_passes"], "cost": cost}
    primary_rows = [result["scored"][program] for program in PRIMARY_PROGRAMS]
    return {"schema": E21_SCHEMA, "status": "complete", "qa": False,
            "primary_rows": primary_rows, "control_row": result["control"],
            "primary_predicates": result["primary_predicates"], "primary_conjunction": result["primary_conjunction"],
            "forward_passes": result["forward_passes"], "cost": dict(EXPECTED_COST), "rows": result["rows"]}


def qa_seen_report(model: Any, *, out: Path, manifest: Mapping[str, Any],
                   programs: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    """Actual report path smoke using only legal seen programs; never primary/control."""
    if any(tuple(program) in set(ALL_PROGRAMS) for program in programs):
        raise ValueError("E21 QA programs must exclude primary and control programs")
    if any(forbidden(program) for program in programs):
        raise ValueError("E21 QA programs must be legal seen programs")
    if len(programs) != 7:
        raise ValueError("E21 QA report smoke requires seven legal seen programs")
    result = evaluate_model(model, manifest, programs=programs)
    report = _report(result, qa=True)
    out = Path(out); e20.refuse_nonempty(out); out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "report.json", report, refuse=True)
    return report


def evaluate_cleared(*, out: Path = DEFAULT_RUN, preflight_dir: Path = DEFAULT_PREFLIGHT,
                     root: Path = e20.PROJECT_ROOT) -> dict[str, Any]:
    out = Path(out); out = out if out.is_absolute() else Path(root) / out
    e20.refuse_nonempty(out); verify_protected_hashes(root)
    manifest = load_manifest(preflight_dir, root=root)
    e20_manifest, model, payload = _load_frozen_e20()
    if sha256_file(Path(root) / E20_CHECKPOINT) != manifest["e20_checkpoint_sha256"] or payload["model_digest"] != manifest["e20_model_digest"]:
        raise ValueError("E21 E20 checkpoint changed before evaluation")
    result = evaluate_model(model, manifest)
    report = _report(result)
    report.update({"manifest": manifest, "e20_checkpoint_sha256": sha256_file(Path(root) / E20_CHECKPOINT),
                   "e20_model_digest": payload["model_digest"], "e20_schema": e20_manifest["schema"]})
    _atomic_json(out / "report.json", report, refuse=True)
    verify_protected_hashes(root)
    if sha256_file(Path(root) / E20_CHECKPOINT) != manifest["e20_checkpoint_sha256"] or e20.digest_state_dict(model) != manifest["e20_model_digest"]:
        raise ValueError("E21 E20 checkpoint/model changed after evaluation")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate-cleared", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    result = preflight(args.preflight_dir) if args.preflight else evaluate_cleared(out=args.out, preflight_dir=args.preflight_dir)
    print(json.dumps({"status": "preflight_ready" if args.preflight else result["status"],
                      "path": str(result) if args.preflight else str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
