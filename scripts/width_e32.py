#!/usr/bin/env python3
"""E32 fixed width128 runner, provenance, evaluator, and private QA."""

from __future__ import annotations

from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
import time
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import width_e32 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.register_e15 import RegisterExample, STATE_ORDER, evaluate_program
from looped_bitnet.runtime import seed_everything
from scripts import composition_e21 as e21
from scripts import continuation_e24 as e24
from scripts import length_transfer_e28 as e28
from scripts.longer_native8_e20 import _update, evaluate as evaluate_seen


ROOT = e.ROOT
STATES = [tuple(state) for state in STATE_ORDER]
SEEN_PROGRAMS = {tuple(program) for program in e._e25_manifest()["seen"]["programs"]["seen"]}
SELECTION = e._e28_selection()
LENGTH_PROGRAMS = tuple(tuple(item["program"]) for item in SELECTION["programs"])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision: {temporary}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision: {temporary}")
    torch.save(value, temporary)
    temporary.replace(path)


def _zero_cost() -> dict[str, int]:
    return dict.fromkeys(e.COST_KEYS, 0)


def _count(counter: dict[str, int], args: Sequence[Any]) -> None:
    x, _, ops = args
    cases, length = int(x.shape[0]), int(ops.shape[1])
    for key, value in zip(e.COST_KEYS, (1, cases, cases * length, cases * length * e.NATIVE_STEPS)):
        counter[key] += value


def _hooks(model: torch.nn.Module, record: dict[str, Any], phase: str):
    attempted = record.setdefault(phase + "_attempted", _zero_cost())
    completed = record.setdefault(phase, _zero_cost())
    pre = model.register_forward_pre_hook(lambda _model, args: _count(attempted, args))

    def post(_model, args, output):
        _count(completed, args)
        if not isinstance(output, tuple) or len(output) != 2 or any(not torch.isfinite(t).all() for t in output):
            raise FloatingPointError("nonfinite E32 model output")

    hook = model.register_forward_hook(post)
    return pre, hook


def _strata(manifest: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    result = {name: [tuple(state) for state in manifest["state_split"][name]] for name in ("train", "validation", "test")}
    if {name: len(states) for name, states in result.items()} != {"train": 192, "validation": 32, "test": 32}:
        raise ValueError("E32 strata changed")
    if len({state for states in result.values() for state in states}) != 256:
        raise ValueError("E32 strata overlap")
    return result


def _length_evaluate(model: torch.nn.Module, manifest: Mapping[str, Any]) -> dict[str, Any]:
    strata = _strata(manifest)
    membership = {state: name for name, states in strata.items() for state in states}
    rows = []
    for program in LENGTH_PROGRAMS:
        row = evaluate_program(model, program, STATES, include_predictions=True)
        for prediction in row["predictions"]:
            prediction["stratum"] = membership[tuple(prediction["state"])]
        rows.append(row)
    return e28.score(rows, SELECTION["programs"], strata)


def predicates(seen: Mapping[str, Any], composition: Mapping[str, Any], length: Mapping[str, Any]) -> dict[str, Any]:
    expected_seen = {tuple(program) for program in SEEN_PROGRAMS}
    validation = seen.get("validation")
    train = seen.get("train")
    if not isinstance(validation, list) or not isinstance(train, list):
        raise ValueError("E32 seen evaluation splits missing")
    if {tuple(row.get("program", ())) for row in validation} != expected_seen or len(validation) != 32:
        raise ValueError("E32 seen validation scope changed")
    if {tuple(row.get("program", ())) for row in train} != expected_seen or len(train) != 32:
        raise ValueError("E32 seen train scope changed")
    expected_primary = {tuple(program) for program in e21.PRIMARY_PROGRAMS}
    primary_rows = composition.get("primary_rows")
    if not isinstance(primary_rows, list) or {tuple(row.get("program", ())) for row in primary_rows} != expected_primary or len(primary_rows) != 6:
        raise ValueError("E32 E21 primary scope changed")
    length_rows = length.get("rows")
    if not isinstance(length_rows, list) or len(length_rows) != len(LENGTH_PROGRAMS) or {tuple(row.get("program", ())) for row in length_rows} != set(LENGTH_PROGRAMS):
        raise ValueError("E32 E28 length scope changed")
    l4_rows = [row for row in length_rows if row["length"] == 4]
    l5_rows = [row for row in length_rows if row["length"] == 5]
    if len(l4_rows) != 6 or len(l5_rows) != 6:
        raise ValueError("E32 E28 L4/L5 scope changed")
    seen_ok = all(int(row["final_joint"]) >= (32 if len(row["program"]) == 1 else 31) for row in validation)
    primary_ok = all(int(row["metrics"]["all"]["final_joint"]) >= 244 for row in primary_rows)
    l4_ok = all(row["metrics"]["all"]["final_joint"] >= 244 for row in l4_rows)
    l5_ok = all(row["metrics"]["all"]["final_joint"] >= 244 for row in l5_rows)
    return {
        "seen_prerequisite": seen_ok,
        "e21_primary_conjunction": primary_ok,
        "l4_conjunction": l4_ok,
        "l5_conjunction": l5_ok,
        "combined_conjunction": seen_ok and primary_ok and l4_ok and l5_ok,
        "e21_control_excluded": True,
    }


def evaluate(model: torch.nn.Module, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed seen+E21+E28 scope. Caller verifies state/RNG immutability."""
    was_training = model.training
    model.eval()
    try:
        seen = evaluate_seen(model, manifest)
        composition = e21._report(e21.evaluate_model(model, manifest["e21"]))
        length = _length_evaluate(model, manifest)
        return {"seen": seen, "composition": composition, "length": length,
                "predicates": predicates(seen, composition, length)}
    finally:
        model.train(was_training)


def _prediction_map(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    values = row.get("predictions")
    if not isinstance(values, list):
        raise ValueError("E32 predictions missing")
    result = {}
    for value in values:
        state = tuple(value["state"])
        if state in result:
            raise ValueError("E32 duplicate prediction state")
        result[state] = value
    return result


def _pair_rows(old_row: Mapping[str, Any], new_row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if old_row.get("program") != new_row.get("program"):
        raise ValueError(f"E32 paired {label} program mismatch")
    left, right = _prediction_map(old_row), _prediction_map(new_row)
    if set(left) != set(right):
        raise ValueError(f"E32 paired {label} state mismatch")
    result = {"new_wins": 0, "old_wins": 0, "both_correct": 0, "both_wrong": 0}
    full = {"new_wins": 0, "old_wins": 0, "both_correct": 0, "both_wrong": 0}
    for state in sorted(left):
        a, b = left[state], right[state]
        for field in ("target_trace", "stratum", "joint_final_correct", "prefix_joint_correct"):
            if field not in a or field not in b:
                raise ValueError(f"E32 paired {label} missing {field}")
        if a["target_trace"] != b["target_trace"] or a["stratum"] != b["stratum"]:
            raise ValueError(f"E32 paired {label} identity mismatch")
        if type(a["joint_final_correct"]) is not bool or type(b["joint_final_correct"]) is not bool:
            raise ValueError(f"E32 paired {label} final correctness type mismatch")
        if any(type(flag) is not bool for flag in a["prefix_joint_correct"] + b["prefix_joint_correct"]):
            raise ValueError(f"E32 paired {label} trace correctness type mismatch")
        ac = a["joint_final_correct"]
        bc = b["joint_final_correct"]
        result["both_correct" if ac and bc else "old_wins" if ac else "new_wins" if bc else "both_wrong"] += 1
        af, bf = all(a["prefix_joint_correct"]), all(b["prefix_joint_correct"])
        full["both_correct" if af and bf else "old_wins" if af else "new_wins" if bf else "both_wrong"] += 1
    old_metrics = old_row.get("metrics", {}).get("all", {})
    new_metrics = new_row.get("metrics", {}).get("all", {})
    old_final = int(old_metrics.get("final_joint", old_row.get("joint_final", old_row.get("final_joint", 0))))
    new_final = int(new_metrics.get("final_joint", new_row.get("joint_final", new_row.get("final_joint", 0))))
    if old_final != result["both_correct"] + result["old_wins"] or new_final != result["both_correct"] + result["new_wins"]:
        raise ValueError(f"E32 paired {label} marginals mismatch")
    old_full = int(old_metrics.get("full_trace", 0))
    new_full = int(new_metrics.get("full_trace", 0))
    if old_full != full["both_correct"] + full["old_wins"] or new_full != full["both_correct"] + full["new_wins"]:
        raise ValueError(f"E32 paired {label} full-trace marginals mismatch")
    return {"program": list(new_row["program"]), "denominator": len(left), **result,
            "old_final": old_final, "new_final": new_final, "delta_final": new_final - old_final,
            "full_trace": full, "old_full_trace": old_full, "new_full_trace": new_full,
            "delta_full_trace": new_full - old_full}


def _length_compare(old_length: Mapping[str, Any], new_length: Mapping[str, Any]) -> dict[str, Any]:
    old_rows = {tuple(row["program"]): row for row in old_length["rows"]}
    new_rows = {tuple(row["program"]): row for row in new_length["rows"]}
    if set(old_rows) != set(new_rows) or set(new_rows) != set(LENGTH_PROGRAMS):
        raise ValueError("E32 L4/L5 program scope mismatch")
    rows = []
    for program in LENGTH_PROGRAMS:
        old_row, new_row = old_rows[program], new_rows[program]
        rows.append(_pair_rows(
            {"program": list(program), "predictions": old_row["predictions"], "metrics": {"all": old_row["metrics"]["all"]}},
            {"program": list(program), "predictions": new_row["predictions"], "metrics": {"all": new_row["metrics"]["all"]}},
            label="length",
        ))
        rows[-1]["length"] = len(program)
        rows[-1]["old_full_trace"] = old_row["metrics"]["all"]["full_trace"]
        rows[-1]["new_full_trace"] = new_row["metrics"]["all"]["full_trace"]
    by_length = {}
    for length in (4, 5):
        scoped = [row for row in rows if row["length"] == length]
        by_length[str(length)] = {
            "old_final_errors": sum(256 - row["old_final"] for row in scoped),
            "new_final_errors": sum(256 - row["new_final"] for row in scoped),
            "old_full_trace_errors": sum(256 - row["old_full_trace"] for row in scoped),
            "new_full_trace_errors": sum(256 - row["new_full_trace"] for row in scoped),
        }
    return {"rows": rows, "by_length": by_length}


def _historical(label: str, seed: int, *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    if label == "E27_W4_64":
        run_label = "E27_W4"
        evaluation = _read_json(root / "runs/e27_w4/report.json")["seeds"][str(seed)]["evaluation"]
    elif label == "E24_float64":
        run_label = "E24_float"
        evaluation = _read_json(root / "runs/e24_continuation/report.json")["seeds"][str(seed)]["evaluation"]
    else:
        raise ValueError("unknown E32 historical comparator")
    length = _read_json(root / "runs/e28_length" / f"{run_label}_seed{seed}.json")["evaluation"]
    return evaluation, length


def compare_historical(current: Mapping[str, Any], label: str, seed: int, *, root: Path = ROOT) -> dict[str, Any]:
    historical, historical_length = _historical(label, seed, root=root)
    return {
        "seen": e24._seen_comparison(historical["seen"], current["seen"]),
        "composition": e24._composition_comparison(historical["composition"], current["composition"]),
        "length": _length_compare(historical_length, current["length"]),
    }


def _length_errors(length: Mapping[str, Any]) -> dict[str, int]:
    rows = {tuple(row["program"]): row for row in length["rows"]}
    if set(rows) != set(LENGTH_PROGRAMS):
        raise ValueError("E32 length error scope changed")
    return {str(size): sum(256 - int(row["metrics"]["all"]["final_joint"])
                             for row in rows.values() if int(row["length"]) == size)
            for size in (4, 5)}


def width_contrasts(report: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Build the frozen paired W4/float and width64/width128 descriptions."""
    models = report.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("E32 model records missing for contrasts")
    by_seed: dict[str, Any] = {}
    l5_benefit = {"w4": [], "float": []}
    for seed in e.SEEDS:
        float_record = models[f"float128_seed{seed}"]
        w4_record = models[f"w4128_seed{seed}"]
        current_float = float_record["evaluation"]["length"]
        current_w4 = w4_record["evaluation"]["length"]
        old_w4, old_w4_length = _historical("E27_W4_64", seed, root=root)
        old_float, old_float_length = _historical("E24_float64", seed, root=root)
        pair128 = e28.paired(current_w4, current_float)
        pair64 = e28.paired(old_w4_length, old_float_length)
        errors128 = {"w4": _length_errors(current_w4), "float": _length_errors(current_float)}
        errors64 = {"w4": _length_errors(old_w4_length), "float": _length_errors(old_float_length)}
        for arm in ("w4", "float"):
            l5_benefit[arm].append(errors128[arm]["5"] < errors64[arm]["5"])
        by_seed[str(seed)] = {
            "paired_128_w4_vs_float": pair128,
            "paired_64_w4_vs_float": pair64,
            "errors_128": errors128,
            "errors_64": errors64,
            "error_gap": {str(size): errors128["w4"][str(size)] - errors128["float"][str(size)] for size in (4, 5)},
            "error_gap_64": {str(size): errors64["w4"][str(size)] - errors64["float"][str(size)] for size in (4, 5)},
            "interaction": {str(size): (errors128["w4"][str(size)] - errors128["float"][str(size)]) -
                                      (errors64["w4"][str(size)] - errors64["float"][str(size)]) for size in (4, 5)},
            "l5_width_benefit": {arm: errors128[arm]["5"] < errors64[arm]["5"] for arm in ("w4", "float")},
        }
    return {"by_seed": by_seed,
            "l5_width_benefit_each_seed": {arm: all(values) for arm, values in l5_benefit.items()}}


def preflight(path: Path = e.PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = Path(path); path = path if path.is_absolute() else Path(root) / path
    old.refuse_nonempty(path)
    manifest = e.make_manifest(root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    for seed in e.SEEDS:
        for arm in e.ARMS:
            model = e.build_initial_model(e.WIDTH, arm, seed, root=root)
            _atomic_torch(path / f"{arm}128_seed{seed}.pt", {
                "schema": e.SCHEMA, "arm": arm, "width": e.WIDTH, "seed": seed,
                "state_dict": model.state_dict(), "model_digest": e.digest_state_dict(model),
                "rng_state": model.initial_rng_state, "rng_digest": e.digest_object(model.initial_rng_state),
                "parameter_count": e.PARAMETER_COUNT_128,
            }, refuse=True)
    e.verify_protected_hashes(root); e.verify_reference_hashes(root)
    return manifest


def load_manifest(path: Path = e.PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = Path(path); path = path if path.is_absolute() else Path(root) / path
    manifest = _read_json(path / "manifest.json")
    if manifest != e.make_manifest(root):
        raise ValueError("E32 frozen manifest changed")
    for seed in e.SEEDS:
        for arm in e.ARMS:
            saved = torch.load(path / f"{arm}128_seed{seed}.pt", map_location="cpu", weights_only=True)
            model = e.build_initial_model(e.WIDTH, arm, seed, root=root)
            expected_meta = {"schema": e.SCHEMA, "arm": arm, "width": e.WIDTH, "seed": seed,
                             "parameter_count": e.PARAMETER_COUNT_128}
            if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected_meta.items()):
                raise ValueError("E32 frozen initial metadata mismatch")
            for key in ("state_dict", "model_digest", "rng_state", "rng_digest"):
                if key not in saved:
                    raise ValueError(f"E32 frozen initial field missing: {key}")
            e.assert_state_equal(saved["state_dict"], model.state_dict())
            if (saved["model_digest"] != e.digest_state_dict(model) or
                    not torch.equal(saved["rng_state"], model.initial_rng_state) or
                    saved["rng_digest"] != e.digest_object(model.initial_rng_state)):
                raise ValueError("E32 frozen initial changed")
    return manifest


def _checkpoint_path(out: Path, label: str, update: int) -> Path:
    return Path(out) / label / f"u{update}.pt"


def execute(out: Path, manifest: Mapping[str, Any], batches: Sequence[Sequence[RegisterExample]], *, qa: bool = False,
            root: Path = ROOT, evaluator: Any = None, qa_snapshots: dict[str, Any] | None = None) -> dict[str, Any]:
    out = Path(out)
    old.refuse_nonempty(out)
    if not batches or any(tuple(item.program) not in SEEN_PROGRAMS for batch in batches for item in batch):
        raise ValueError("E32 training stream must use legal seen programs")
    if not qa and (len(batches) != e.UPDATES or old.batch_digest(batches) != manifest["full_stream_digest"] or
                   old.target_digest(batches) != manifest["full_target_digest"]):
        raise ValueError("E32 scientific stream mismatch")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", dict(manifest), refuse=True)
    report: dict[str, Any] = {"schema": e.SCHEMA, "qa": qa, "status": "running", "model_order": list(e.LABELS), "models": {}}
    try:
        for seed in e.SEEDS:
            for arm in e.ARMS:
                label = f"{arm}128_seed{seed}"
                record: dict[str, Any] = {
                    "label": label, "arm": arm, "width": e.WIDTH, "seed": seed,
                    "status": "running", "attempted_updates": 0, "completed_updates": 0,
                    "training_cost": _zero_cost(), "evaluation_cost": _zero_cost(), "progress": [],
                }
                report["models"][label] = record
                seed_everything(seed, deterministic=True, cpu_threads=4)
                model = e.build_initial_model(e.WIDTH, arm, seed, root=root)
                optimizer = old.make_optimizer(model)
                torch.set_rng_state(model.initial_rng_state)
                model.train(True)
                folder = out / label
                folder.mkdir(parents=True, exist_ok=False)
                started = time.monotonic()
                hooks = _hooks(model, record, "training_cost")
                try:
                    for update, batch in enumerate(batches, 1):
                        record["attempted_updates"] = update
                        loss = _update(model, optimizer, batch)
                        record["completed_updates"] = update
                        if update % 250 == 0 or update == len(batches):
                            record["progress"].append({"update": update, "loss": float(loss.detach()), "seconds": time.monotonic() - started})
                            _atomic_json(out / "progress.json", report)
                finally:
                    for hook in hooks:
                        hook.remove()
                if not qa and record["training_cost"] != e.TRAIN_COST_PER_MODEL:
                    raise ValueError("E32 training cost mismatch")
                record["training_seconds"] = time.monotonic() - started
                if qa_snapshots is not None:
                    qa_snapshots[label] = {
                        "state_dict": deepcopy(model.state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "rng_state": torch.get_rng_state().clone(),
                    }
                final_update = len(batches)
                checkpoint = e.checkpoint_payload(model, optimizer, manifest, arm=arm, width=e.WIDTH, seed=seed,
                                                   update=final_update, qa=qa,
                                                   training_cost=record["training_cost"])
                path = _checkpoint_path(out, label, final_update)
                _atomic_torch(path, checkpoint, refuse=True)
                restored, restored_optimizer, _ = e.load_checkpoint(path, manifest, arm=arm, width=e.WIDTH, seed=seed,
                                                                    update=final_update, qa=qa, root=root,
                                                                    training_cost=record["training_cost"])
                before = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
                evaluator_hooks = _hooks(restored, record, "evaluation_cost")
                try:
                    result = (evaluator or evaluate)(restored, manifest)
                finally:
                    for hook in evaluator_hooks:
                        hook.remove()
                after = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
                if before != after or not restored.training:
                    raise ValueError("E32 evaluation mutated state")
                record["evaluation"] = result
                if not qa:
                    if record["evaluation_cost"] != e.EVAL_COST_PER_MODEL:
                        raise ValueError("E32 evaluation cost mismatch")
                    record["predicates"] = result["predicates"]
                    record["historical_comparisons"] = {
                        "E27_W4_64": compare_historical(result, "E27_W4_64", seed, root=root),
                        "E24_float64": compare_historical(result, "E24_float64", seed, root=root),
                    }
                record["status"] = "complete"
                _atomic_json(folder / "report.json", record, refuse=True)
                _atomic_json(out / "progress.json", report)
        report["status"] = "complete"
        report["scientific_training_cost"] = {key: sum(record["training_cost"][key] for record in report["models"].values()) for key in e.COST_KEYS}
        report["scientific_evaluation_cost"] = {key: sum(record["evaluation_cost"][key] for record in report["models"].values()) for key in e.COST_KEYS}
        if not qa:
            report["combined_success_all_models"] = all(record["predicates"]["combined_conjunction"] for record in report["models"].values())
            report["width_contrasts"] = width_contrasts(report, root=root)
            report["restoration_gates"] = {
                "float128_all_seeds": all(report["models"][f"float128_seed{seed}"]["predicates"]["combined_conjunction"] for seed in e.SEEDS),
                "w4128_all_seeds": all(report["models"][f"w4128_seed{seed}"]["predicates"]["combined_conjunction"] for seed in e.SEEDS),
                "float128_l5_final_all_seeds": all(
                    all(row["metrics"]["all"]["final_joint"] >= 244 for row in report["models"][f"float128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 5)
                    for seed in e.SEEDS),
                "w4128_l5_final_all_seeds": all(
                    all(row["metrics"]["all"]["final_joint"] >= 244 for row in report["models"][f"w4128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 5)
                    for seed in e.SEEDS),
                "float128_l4_full_trace_all_seeds": all(
                    all(row["metrics"]["all"]["full_trace"] >= 244 for row in report["models"][f"float128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 4)
                    for seed in e.SEEDS),
                "w4128_l4_full_trace_all_seeds": all(
                    all(row["metrics"]["all"]["full_trace"] >= 244 for row in report["models"][f"w4128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 4)
                    for seed in e.SEEDS),
                "float128_l5_full_trace_all_seeds": all(
                    all(row["metrics"]["all"]["full_trace"] >= 244 for row in report["models"][f"float128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 5)
                    for seed in e.SEEDS),
                "w4128_l5_full_trace_all_seeds": all(
                    all(row["metrics"]["all"]["full_trace"] >= 244 for row in report["models"][f"w4128_seed{seed}"]["evaluation"]["length"]["rows"] if row["length"] == 5)
                    for seed in e.SEEDS),
            }
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"
        report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        for record in report["models"].values():
            if record.get("status") == "running":
                record["status"] = "technical_failure"
                record["error"] = report["technical_failure"]
        _atomic_json(out / "progress.json", report)
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def tiny_qa(out: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Six real cells, two legal ADD updates, two-state finite eval, reload-next-update check."""
    batch = [RegisterExample(0, 1, ("ADD",)), RegisterExample(2, 3, ("ADD",))]

    def qa_eval(model: torch.nn.Module, _manifest: Mapping[str, Any]) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                row = evaluate_program(model, ("ADD",), [(0, 1), (2, 3)], include_predictions=True)
            return {"qa": True, "program": ["ADD"], "cases": 2, "finite": True, "joint_final": row["joint_final"]}
        finally:
            model.train(was_training)

    # Build metadata once; this is not a scientific forward or update.
    manifest = e.make_manifest(root)
    snapshots: dict[str, Any] = {}
    report = execute(out, manifest, [batch, batch], qa=True, root=root, evaluator=qa_eval,
                     qa_snapshots=snapshots)
    manifest_on_disk = _read_json(Path(out) / "manifest.json")
    for label in e.LABELS:
        record = report["models"][label]
        path = _checkpoint_path(out, label, 2)
        restored, optimizer, payload = e.load_checkpoint(path, manifest_on_disk, arm=record["arm"], width=e.WIDTH,
                                                         seed=record["seed"], update=2, qa=True, root=root,
                                                         training_cost=record["training_cost"])
        snapshot = snapshots[label]
        uninterrupted = e.build_initial_model(e.WIDTH, record["arm"], record["seed"], root=root)
        uninterrupted.load_state_dict(snapshot["state_dict"], strict=True)
        uninterrupted_optimizer = old.make_optimizer(uninterrupted)
        uninterrupted_optimizer.load_state_dict(deepcopy(snapshot["optimizer_state_dict"]))
        torch.set_rng_state(snapshot["rng_state"])
        _update(uninterrupted, uninterrupted_optimizer, batch)
        left_rng = torch.get_rng_state().clone()
        torch.set_rng_state(payload["rng_state"])
        _update(restored, optimizer, batch)
        right_rng = torch.get_rng_state().clone()
        if (e.digest_state_dict(uninterrupted) != e.digest_state_dict(restored) or
                e.digest_object(uninterrupted_optimizer.state_dict()) != e.digest_object(optimizer.state_dict()) or
                not torch.equal(left_rng, right_rng)):
            raise ValueError(f"E32 QA reload next-update mismatch: {label}")
        record["reload_next_update_equal"] = True
    report["qa_reload_next_update_cost"] = {
        # Each cell is checked through two actual updates: one uninterrupted
        # branch and one reloaded branch.  Count both branches explicitly.
        "updates": 2 * len(e.LABELS), "examples": 2 * len(e.LABELS) * len(batch),
        "readouts": 2 * len(e.LABELS) * len(batch),
        "internal_state_substeps": 2 * len(e.LABELS) * len(batch) * e.NATIVE_STEPS,
    }
    _atomic_json(Path(out) / "report.json", report)
    return report


def run(out: Path = e.RUN, preflight_dir: Path = e.PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    manifest = load_manifest(preflight_dir, root=root)
    try:
        target = Path(out) if Path(out).is_absolute() else Path(root) / out
        return execute(target, manifest, e.stream(), root=root)
    finally:
        e.verify_protected_hashes(root); e.verify_reference_hashes(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-cleared", action="store_true")
    group.add_argument("--qa", action="store_true")
    parser.add_argument("--out", type=Path, default=e.RUN)
    parser.add_argument("--preflight-dir", type=Path, default=e.PREFLIGHT)
    args = parser.parse_args(argv)
    if args.preflight:
        value = preflight(args.preflight_dir)
        print(json.dumps({"status": "preflight_ready", "schema": value["schema"]}, sort_keys=True))
    elif args.qa:
        value = tiny_qa(args.out)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    else:
        value = run(args.out, args.preflight_dir)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
