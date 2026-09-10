#!/usr/bin/env python3
"""E24 fixed continuation runner; scientific budgets are not CLI-configurable."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import continuation_e24 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.register_e15 import forbidden
from scripts import composition_e21 as composition
from scripts import replication_e22 as replication
from scripts.longer_native8_e20 import _update as accepted_update
from scripts.longer_native8_e20 import evaluate as evaluate_seen


PRIMARY_BASELINE_ERRORS = {0: 14, 1: 73, 2: 50}
PRIMARY_PROGRAMS = tuple(tuple(p) for p in composition.PRIMARY_PROGRAMS)
CONTROL_PROGRAM = tuple(composition.CONTROL_PROGRAM)
ALL_RESERVED_PROGRAMS = set(composition.ALL_PROGRAMS)
SEEN_PROGRAMS = {tuple(p) for p in composition.all_programs() if not forbidden(p)} - ALL_RESERVED_PROGRAMS


def _resolve(root: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def protected(root: Path = e.ROOT) -> dict[str, str]:
    return e.verify_protected_hashes(root)


def expected_manifest(root: Path = e.ROOT) -> dict[str, Any]:
    protected(root)
    return e.make_manifest(root=root)


def preflight(path: Path = e.PREFLIGHT, *, root: Path = e.ROOT) -> Path:
    path = _resolve(root, Path(path))
    e.refuse_nonempty(path)
    manifest = expected_manifest(root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    protected(root)
    return path / "manifest.json"


def load_manifest(path: Path = e.PREFLIGHT, *, root: Path = e.ROOT) -> dict[str, Any]:
    path = _resolve(root, Path(path))
    manifest = _read_json(path / "manifest.json")
    expected = expected_manifest(root)
    if manifest != expected:
        raise ValueError("E24 frozen manifest or imported reference changed")
    return manifest


def evaluate(model: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run the accepted final E22 evaluator against the fixed E24 manifest."""
    return replication.evaluate(model, manifest)


def _source_metric(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or "all" not in metrics:
        raise ValueError("composition row has no all-state metrics")
    return metrics["all"]


def _prediction_map(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    predictions = row.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 256:
        raise ValueError("composition row prediction count changed")
    result = {}
    for prediction in predictions:
        state = tuple(prediction["state"])
        if state in result:
            raise ValueError("duplicate composition state")
        result[state] = prediction
    if len(result) != 256:
        raise ValueError("composition state identity changed")
    return result


def _paired_row(parent: Mapping[str, Any], current: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if parent.get("program") != current.get("program"):
        raise ValueError(f"paired {label} program identity changed")
    parent_cases = _prediction_map(parent)
    current_cases = _prediction_map(current)
    if set(parent_cases) != set(current_cases):
        raise ValueError(f"paired {label} state identity changed")
    wins = losses = both_correct = both_wrong = 0
    for state in sorted(parent_cases):
        left = parent_cases[state]
        right = current_cases[state]
        if left.get("stratum") != right.get("stratum"):
            raise ValueError(f"paired {label} stratum changed at {state}")
        if left.get("target_trace") != right.get("target_trace"):
            raise ValueError(f"paired {label} target trace changed at {state}")
        parent_correct = bool(left.get("joint_final_correct"))
        current_correct = bool(right.get("joint_final_correct"))
        if not parent_correct and current_correct:
            wins += 1
        elif parent_correct and not current_correct:
            losses += 1
        elif parent_correct:
            both_correct += 1
        else:
            both_wrong += 1
    if wins + losses + both_correct + both_wrong != len(parent_cases):
        raise ValueError(f"paired {label} outcomes do not sum to denominator")
    parent_metrics = _source_metric(parent)
    current_metrics = _source_metric(current)
    if parent_metrics["final_joint"] != both_correct + losses or current_metrics["final_joint"] != both_correct + wins:
        raise ValueError(f"paired {label} marginal final counts disagree with outcomes")
    prefix_delta = [b - a for a, b in zip(parent_metrics["prefix_joint"], current_metrics["prefix_joint"])]
    return {
        "program": list(current["program"]),
        "denominator": len(parent_cases),
        "wins_parent_wrong_new_correct": wins,
        "losses_parent_correct_new_wrong": losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "parent_metrics": dict(parent_metrics),
        "new_metrics": dict(current_metrics),
        "delta": {
            "final_joint": current_metrics["final_joint"] - parent_metrics["final_joint"],
            "final_x": current_metrics["final_x"] - parent_metrics["final_x"],
            "final_y": current_metrics["final_y"] - parent_metrics["final_y"],
            "full_trace": current_metrics["full_trace"] - parent_metrics["full_trace"],
            "prefix_joint": prefix_delta,
        },
    }


def _composition_comparison(parent: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    parent_rows = {tuple(row["program"]): row for row in parent["primary_rows"]}
    current_rows = {tuple(row["program"]): row for row in current["primary_rows"]}
    if set(parent_rows) != set(current_rows) or set(current_rows) != set(PRIMARY_PROGRAMS):
        raise ValueError("primary composition program scope changed")
    control_parent = parent.get("control_row")
    control_current = current.get("control_row")
    if not isinstance(control_parent, Mapping) or not isinstance(control_current, Mapping):
        raise ValueError("control composition row missing")
    return {
        "primary": [_paired_row(parent_rows[program], current_rows[program], label="primary") for program in PRIMARY_PROGRAMS],
        "control": _paired_row(control_parent, control_current, label="control"),
        "primary_final_errors_parent": sum(256 - _source_metric(parent_rows[p])["final_joint"] for p in PRIMARY_PROGRAMS),
        "primary_final_errors_new": sum(256 - _source_metric(current_rows[p])["final_joint"] for p in PRIMARY_PROGRAMS),
        "primary_conjunction_parent": parent.get("primary_conjunction"),
        "primary_conjunction_new": current.get("primary_conjunction"),
    }


def _seen_row_key(row: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    split = str(row.get("split"))
    return tuple(row["program"]), split


def _seen_comparison(parent: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        parent_rows = {_seen_row_key(row): row for row in parent[split]}
        current_rows = {_seen_row_key(row): row for row in current[split]}
        if set(parent_rows) != set(current_rows):
            raise ValueError(f"seen {split} program identity changed")
        rows = []
        for key in sorted(parent_rows):
            left = parent_rows[key]
            right = current_rows[key]
            left_cases = _prediction_map_seen(left)
            right_cases = _prediction_map_seen(right)
            if set(left_cases) != set(right_cases):
                raise ValueError(f"seen {split} state identity changed")
            if any(left_cases[state].get("target_trace") != right_cases[state].get("target_trace") for state in left_cases):
                raise ValueError(f"seen {split} target trace changed")
            parent_joint = int(left.get("joint_final", left.get("final_joint")))
            current_joint = int(right.get("joint_final", right.get("final_joint")))
            parent_full = int(left["full_trace"])
            current_full = int(right["full_trace"])
            parent_prefix = list(left.get("prefix_joint", left.get("prefix_joint_correct", [])))
            current_prefix = list(right.get("prefix_joint", right.get("prefix_joint_correct", [])))
            rows.append({
                "program": list(key[0]), "split": split, "denominator": len(left_cases),
                "parent_final_joint": parent_joint, "new_final_joint": current_joint,
                "delta_final_joint": current_joint - parent_joint,
                "parent_final_x": int(left.get("final_x", left.get("final_x_correct"))),
                "new_final_x": int(right.get("final_x", right.get("final_x_correct"))),
                "delta_final_x": int(right.get("final_x", right.get("final_x_correct"))) - int(left.get("final_x", left.get("final_x_correct"))),
                "parent_final_y": int(left.get("final_y", left.get("final_y_correct"))),
                "new_final_y": int(right.get("final_y", right.get("final_y_correct"))),
                "delta_final_y": int(right.get("final_y", right.get("final_y_correct"))) - int(left.get("final_y", left.get("final_y_correct"))),
                "parent_full_trace": parent_full, "new_full_trace": current_full,
                "delta_full_trace": current_full - parent_full,
                "parent_prefix_joint": parent_prefix, "new_prefix_joint": current_prefix,
                "delta_prefix_joint": [b - a for a, b in zip(parent_prefix, current_prefix)],
            })
        result[split] = rows
    return result


def _prediction_map_seen(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    predictions = row.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("seen row lacks predictions")
    result = {}
    for item in predictions:
        state = tuple(item["state"])
        if state in result:
            raise ValueError("duplicate seen state")
        result[state] = item
    return result


def parent_reports(seed: int, *, root: Path = e.ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(_resolve(root, e.PARENT_REPORTS[seed]))
    if seed == 0:
        seen = report["evaluations"]["8000"]
        composition_report = _read_json(_resolve(root, e.E21_REPORT))
    else:
        evaluation = report["evaluation"]
        seen = evaluation["seen"]
        composition_report = evaluation["composition"]
    return seen, composition_report


def compare_seed(seed: int, current: Mapping[str, Any], *, root: Path = e.ROOT) -> dict[str, Any]:
    parent_seen, parent_composition = parent_reports(seed, root=root)
    new_seen = current["seen"]
    new_composition = current["composition"]
    composition_delta = _composition_comparison(parent_composition, new_composition)
    return {
        "seen": _seen_comparison(parent_seen, new_seen),
        "composition": composition_delta,
        "primary_final_error_reduction": {
            "parent_error_count": PRIMARY_BASELINE_ERRORS[seed],
            "new_error_count": composition_delta["primary_final_errors_new"],
            "strictly_reduces": composition_delta["primary_final_errors_new"] < PRIMARY_BASELINE_ERRORS[seed],
        },
    }


def predicates(seen: Mapping[str, Any], composition_report: Mapping[str, Any]) -> dict[str, Any]:
    validation = seen["validation"]
    seen_prerequisite = all(
        int(row["final_joint"]) >= (32 if len(row["program"]) == 1 else 31)
        for row in validation
    )
    primary_rows = composition_report["primary_rows"]
    if len(primary_rows) != 6 or {tuple(row["program"]) for row in primary_rows} != set(PRIMARY_PROGRAMS):
        raise ValueError("primary predicate scope mismatch")
    primary = all(int(row["metrics"]["all"]["final_joint"]) >= 244 for row in primary_rows)
    if composition_report.get("primary_conjunction") is not primary:
        raise ValueError("primary predicate disagrees with counts")
    return {
        "seen_prerequisite": seen_prerequisite,
        "primary_conjunction": primary,
        "combined_conjunction": seen_prerequisite and primary if primary is not None else None,
    }


@dataclass(frozen=True)
class _TinySpec:
    """Private QA injection; never exposed through the scientific CLI."""

    parents: Mapping[int, Callable[[], tuple[Any, Any, Mapping[str, Any]]]]
    batches: list
    evaluate: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    update: Callable[[Any, Any, Any], Any] = accepted_update
    progress_interval: int = 1
    checkpoint_updates: tuple[int, ...] = ()
    parent_update: int = 0
    evaluation_cost: Mapping[str, int] | None = None


def _qa_manifest(manifest: Mapping[str, Any], qa: _TinySpec, batches: list) -> dict[str, Any]:
    result = deepcopy(dict(manifest))
    config = dict(result.get("config", e.CONFIG))
    config.update(qa_batch_size=len(batches[0]), qa_program_length=len(batches[0][0].program), parent_update=qa.parent_update, added_updates=len(batches), final_update=qa.parent_update + len(batches))
    result["config"] = config
    result["config_hash"] = old.canonical_hash(config)
    return result


def _parent_loader_for(seed: int, qa: _TinySpec | None, root: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    if qa is not None:
        factory = qa.parents.get(seed)
        if factory is None:
            raise ValueError(f"QA parent missing seed {seed}")
        value = factory()
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("QA parent factory must return (model, optimizer, payload)")
        return value
    return e.load_parent(seed, root=root)


def _count_forward(counter, model, args, output):
    x, y, op_ids = args
    cases = int(x.shape[0])
    length = int(op_ids.shape[1])
    counter["program_forwards"] += 1
    counter["program_state_cases"] += cases
    counter["readout_positions"] += cases * length
    counter["internal_state_substeps"] += cases * length * e.NATIVE_STEPS


def _record_cost(record: Mapping[str, Any]) -> dict[str, int]:
    return {
        "updates": int(record.get("added_updates", 0)),
        "examples": int(record.get("added_examples", 0)),
        "internal_state_substeps": int(record.get("added_internal_state_substeps", 0)),
    }


def _report_cost(report: Mapping[str, Any], qa: _TinySpec | None) -> dict[str, Any]:
    actual_evaluation = {key: 0 for key in e.EVAL_COST_TOTAL}
    for record in report.get("seeds", {}).values():
        for key, value in record.get("evaluation_cost", {}).items():
            if key in actual_evaluation:
                actual_evaluation[key] += int(value)
    return {
        "actual_added": {
            "updates": sum(_record_cost(record)["updates"] for record in report.get("seeds", {}).values()),
            "examples": sum(_record_cost(record)["examples"] for record in report.get("seeds", {}).values()),
            "internal_state_substeps": sum(_record_cost(record)["internal_state_substeps"] for record in report.get("seeds", {}).values()),
        },
        "actual_evaluation": actual_evaluation,
        "evaluation_budget": dict((qa.evaluation_cost if qa is not None and qa.evaluation_cost is not None else e.EVAL_COST_TOTAL)),
    }


def _train(out: Path, manifest: Mapping[str, Any], *, _qa: _TinySpec | None = None,
           root: Path = e.ROOT) -> dict[str, Any]:
    out = Path(out)
    e.refuse_nonempty(out)
    batches = old.full_stream() if _qa is None else list(_qa.batches)
    if not batches:
        raise ValueError("E24 continuation requires nonempty batches")
    if _qa is not None:
        if any(tuple(item.program) not in SEEN_PROGRAMS for batch in batches for item in batch):
            raise ValueError("QA must use legal seen-only programs outside primary/control")
        run_manifest = _qa_manifest(manifest, _qa, batches)
        checkpoint_updates = set(_qa.checkpoint_updates) | {len(batches)}
        progress_interval = _qa.progress_interval
        updater = _qa.update
    else:
        run_manifest = dict(manifest)
        checkpoint_updates = {len(batches)}
        progress_interval = 250
        updater = accepted_update
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", run_manifest, refuse=True)
    report: dict[str, Any] = {
        "schema": e.SCHEMA, "qa": _qa is not None, "status": "running", "seeds": {},
        "parent_update": e.PARENT_UPDATE if _qa is None else _qa.parent_update,
        "added_updates_per_seed": len(batches), "cumulative_final_update": (e.FINAL_UPDATE if _qa is None else (_qa.parent_update + len(batches))),
    }
    try:
        for seed in e.SEEDS:
            folder = out / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            model, optimizer, parent = _parent_loader_for(seed, _qa, root)
            parent_rng = parent.get("rng_state")
            if not isinstance(parent_rng, torch.Tensor):
                raise ValueError(f"parent seed{seed} lacks RNG state")
            torch.set_rng_state(parent_rng)
            model.train(True)
            parent_update = int(parent.get("update", report["parent_update"]))
            record: dict[str, Any] = {
                "status": "running", "parent_update": parent_update, "added_updates": 0,
                "cumulative_update": parent_update, "added_examples": 0,
                "added_internal_state_substeps": 0, "progress": [], "numerical_failures": [],
                "parent_model_digest": parent.get("model_digest"),
                "parent_optimizer_digest": parent.get("optimizer_digest"),
                "parent_rng_digest": parent.get("rng_digest"),
            }
            report["seeds"][str(seed)] = record
            started = time.monotonic()
            for added_index, batch in enumerate(batches, 1):
                record["attempted_updates"] = added_index
                forward_cost = record.setdefault("training_forward_cost", {key: 0 for key in e.EVAL_COST_TOTAL})
                hook = model.register_forward_hook(lambda model, args, output: _count_forward(forward_cost, model, args, output))
                try:
                    loss = updater(model, optimizer, batch)
                finally:
                    hook.remove()
                record["added_updates"] = added_index
                record["cumulative_update"] = parent_update + added_index
                record["added_examples"] += len(batch)
                record["added_internal_state_substeps"] += e.NATIVE_STEPS * sum(len(item.program) for item in batch)
                if added_index % progress_interval == 0 or added_index in checkpoint_updates:
                    record["progress"].append({
                        "added_update": added_index,
                        "cumulative_update": parent_update + added_index,
                        "loss": float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss),
                        "elapsed_seconds": time.monotonic() - started,
                    })
                    _atomic_json(out / "progress.json", report)
                if added_index in checkpoint_updates:
                    payload = e.checkpoint_payload(
                        model, optimizer, rng_state=torch.get_rng_state(), seed=seed,
                        update=parent_update + added_index, manifest=run_manifest, parent=parent,
                        qa=_qa is not None,
                    )
                    old.atomic_torch_save(folder / f"u{parent_update + added_index}.pt", payload)
            else:
                record["status"] = "trained"
                record["training_seconds"] = time.monotonic() - started
                final_path = folder / f"u{parent_update + len(batches)}.pt"
                if not final_path.exists():
                    payload = e.checkpoint_payload(
                        model, optimizer, rng_state=torch.get_rng_state(), seed=seed,
                        update=parent_update + len(batches), manifest=run_manifest, parent=parent,
                        qa=_qa is not None,
                    )
                    old.atomic_torch_save(final_path, payload)
                loader = (lambda selected_seed: _parent_loader_for(selected_seed, _qa, root))
                restored, restored_optimizer, _ = e.load_checkpoint(
                    final_path, run_manifest, seed=seed, expected_update=parent_update + len(batches),
                    qa=_qa is not None, parent_loader=loader,
                )
                before = old.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state()))
                evaluation_started = time.monotonic()
                record["evaluation_cost"] = {key: 0 for key in e.EVAL_COST_TOTAL}
                hook = restored.register_forward_hook(lambda model, args, output: _count_forward(record["evaluation_cost"], model, args, output))
                try:
                    evaluation = (_qa.evaluate if _qa is not None else evaluate)(restored, run_manifest)
                finally:
                    hook.remove()
                after = old.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state()))
                if not restored.training or before != after:
                    raise ValueError("E24 evaluation mutated model, optimizer, RNG, or training mode")
                record["evaluation_seconds"] = time.monotonic() - evaluation_started
                record["evaluation"] = dict(evaluation)
                expected_cost = (_qa.evaluation_cost if _qa is not None else e.EVAL_COST_PER_SEED)
                if expected_cost is not None and record["evaluation_cost"] != dict(expected_cost):
                    raise ValueError("E24 evaluation actual forward cost differs from fixed budget")
                record["status"] = "complete"
                record["predicates"] = (dict(evaluation["qa_predicates"]) if _qa is not None else predicates(evaluation["seen"], evaluation["composition"]))
                record["numerical_failure"] = record["predicates"]["combined_conjunction"] is False
            _atomic_json(out / "progress.json", report)
        if any(record.get("status") == "running" for record in report["seeds"].values()):
            raise RuntimeError("E24 internal seed state remained running")
        numerical_failure = any(record.get("numerical_failure") is True for record in report["seeds"].values())
        report["status"] = "complete_with_numerical_failures" if numerical_failure else "complete"
        if _qa is None:
            evaluations = {
                int(seed): record["evaluation"]
                for seed, record in report["seeds"].items()
                if record.get("status") == "complete"
            }
            report["predicates"] = {}
            for seed in e.SEEDS:
                record = report["seeds"].get(str(seed), {})
                if record.get("status") == "complete":
                    report["predicates"][str(seed)] = predicates(
                        evaluations[seed]["seen"], evaluations[seed]["composition"]
                    )
                else:
                    raise RuntimeError(f"unexpected final seed status for seed{seed}")
            report["primary_success_all_seeds"] = all(
                value["primary_conjunction"] is True for value in report["predicates"].values()
            )
            report["combined_success_all_seeds"] = all(
                value["combined_conjunction"] is True for value in report["predicates"].values()
            )
            report["paired_comparisons"] = {
                str(seed): compare_seed(seed, evaluations[seed], root=root)
                for seed in sorted(evaluations)
            }
            report["secondary_training_benefit"] = {
                "per_seed": {
                    str(seed): (
                        report["paired_comparisons"][str(seed)]["primary_final_error_reduction"]
                        if str(seed) in report["paired_comparisons"] else None
                    )
                    for seed in e.SEEDS
                },
                "all_seeds_strictly_reduce": (
                    all(
                        report["paired_comparisons"][str(seed)]["primary_final_error_reduction"]["strictly_reduces"]
                        for seed in e.SEEDS
                    )
                ),
            }
        report["cost"] = _report_cost(report, _qa)
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"
        report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        for seed in e.SEEDS:
            value = report["seeds"].get(str(seed))
            if value is None:
                report["seeds"][str(seed)] = {"status": "not_started", "reason": "suspended after technical failure"}
            elif value.get("status") in {"running", "trained"}:
                value["status"] = "technical_failure"
                value["error"] = report["technical_failure"]
        report["cost"] = _report_cost(report, _qa)
        _atomic_json(out / "progress.json", report)
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def run(out: Path = e.RUN, preflight_dir: Path = e.PREFLIGHT, *, root: Path = e.ROOT) -> dict[str, Any]:
    manifest = load_manifest(preflight_dir, root=root)
    try:
        return _train(_resolve(root, Path(out)), manifest, root=root)
    finally:
        protected(root)


def recovery(out: Path = e.RUN, preflight_dir: Path = e.PREFLIGHT, *, root: Path = e.ROOT) -> dict[str, Any]:
    manifest = load_manifest(preflight_dir, root=root)
    out = _resolve(root, Path(out))
    result: dict[str, Any] = {"status": "evaluation_only", "evaluations": {}, "missing": []}
    for seed in e.SEEDS:
        path = out / f"seed{seed}" / f"u{e.FINAL_UPDATE}.pt"
        if not path.exists():
            result["missing"].append(seed)
            continue
        model, optimizer, _ = e.load_checkpoint(path, manifest, seed=seed, expected_update=e.FINAL_UPDATE)
        before = old.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        result["evaluations"][str(seed)] = evaluate(model, manifest)
        if not model.training or before != old.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state())):
            raise ValueError("E24 recovery evaluation mutated state")
    result["all_finals_present"] = not result["missing"]
    protected(root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-cleared", action="store_true")
    group.add_argument("--eval-only", action="store_true")
    parser.add_argument("--out", type=Path, default=e.RUN)
    parser.add_argument("--preflight-dir", type=Path, default=e.PREFLIGHT)
    args = parser.parse_args(argv)
    if args.preflight:
        result: Any = preflight(args.preflight_dir)
        payload = {"status": "preflight_ready", "path": str(result)}
    elif args.train_cleared:
        result = run(args.out, args.preflight_dir)
        payload = {"status": result["status"], "path": str(args.out)}
    else:
        result = recovery(args.out, args.preflight_dir)
        payload = {"status": result["status"], "path": str(args.out), "missing": result["missing"]}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
