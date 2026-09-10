#!/usr/bin/env python3
"""Reproducible, report-only E23 error analysis.

This script reads only the frozen E21/E22 JSON reports.  It deliberately does
not import project code, load checkpoints, call a model, or run inference.
The output is write-once: an existing output is an error rather than something
to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "E23_ERROR_ANALYSIS.json"
HASH_MANIFEST = ROOT / "results" / "E23_ERROR_ANALYSIS_INPUT_HASHES.json"

INPUTS = {
    "seed0": ROOT / "runs/e21_composition/report.json",
    "seed1": ROOT / "runs/e22_replication/seed1/report.json",
    "seed2": ROOT / "runs/e22_replication/seed2/report.json",
}

PROGRAMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ADD XOR", ("ADD", "XOR")),
    ("ADD ADD XOR", ("ADD", "ADD", "XOR")),
    ("ADD XOR ADD", ("ADD", "XOR", "ADD")),
    ("ADD XOR SWAP", ("ADD", "XOR", "SWAP")),
    ("XOR ADD XOR", ("XOR", "ADD", "XOR")),
    ("SWAP ADD XOR", ("SWAP", "ADD", "XOR")),
)
EXPECTED_STRATA = {"train": 192, "validation": 32, "test": 32}
EXPECTED_COST = {
    "control_internal_state_substeps": 6144,
    "control_program_state_cases": 256,
    "control_readout_positions": 768,
    "internal_state_substeps": 40960,
    "primary_internal_state_substeps": 34816,
    "primary_program_state_cases": 1536,
    "primary_readout_positions": 4352,
    "program_state_cases": 1792,
    "readout_positions": 5120,
    "training_updates": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_op(state: tuple[int, int], op: str) -> tuple[int, int]:
    x, y = state
    if op == "ADD":
        return ((x + y) % 16, y)
    if op == "XOR":
        return (x ^ y, y)
    if op == "SWAP":
        return (y, x)
    raise AssertionError(f"unknown DSL operation: {op}")


def target_trace(initial: tuple[int, int], program: tuple[str, ...]) -> list[list[int]]:
    state = initial
    trace: list[list[int]] = []
    for op in program:
        state = apply_op(state, op)
        trace.append([state[0], state[1]])
    return trace


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def case_key(program_name: str, initial: tuple[int, int]) -> str:
    return f"{program_name}|{initial[0]},{initial[1]}"


def expected_metric(predictions: list[dict[str, Any]], stratum: str | None, trace_len: int) -> dict[str, Any]:
    selected = [p for p in predictions if stratum is None or p["stratum"] == stratum]
    denominator = len(selected)
    final_matches = [p["predicted_trace"][-1] == p["target_trace"][-1] for p in selected]
    final_x = [p["predicted_trace"][-1][0] == p["target_trace"][-1][0] for p in selected]
    final_y = [p["predicted_trace"][-1][1] == p["target_trace"][-1][1] for p in selected]
    full_trace = [p["predicted_trace"] == p["target_trace"] for p in selected]
    prefix_joint = [
        sum(p["predicted_trace"][i] == p["target_trace"][i] for p in selected)
        for i in range(trace_len)
    ]
    return {
        "denominator": denominator,
        "final_joint": sum(final_matches),
        "final_x": sum(final_x),
        "final_y": sum(final_y),
        "full_trace": sum(full_trace),
        "prefix_joint": prefix_joint,
    }


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def load_composition(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text())
    composition = report if "primary_rows" in report else report["evaluation"]["composition"]
    return report, composition


def validate_composition(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report, composition = load_composition(path)
    assert_equal(composition.get("status"), "complete", f"{label} status")
    assert_equal(composition.get("schema"), "e21_composition_v1", f"{label} schema")
    assert_equal(composition.get("forward_passes"), 7, f"{label} forward passes")
    assert_equal(composition.get("cost"), EXPECTED_COST, f"{label} cost accounting")

    rows = composition.get("primary_rows")
    assert isinstance(rows, list), f"{label}: primary_rows must be a list"
    assert_equal(len(rows), len(PROGRAMS), f"{label} primary row count")
    assert_equal(len(composition.get("rows", [])), 7, f"{label} all row count")
    assert_equal(composition.get("control_row", {}).get("role"), "control", f"{label} control role")

    for row, (program_name, program) in zip(rows, PROGRAMS):
        assert_equal(row.get("role"), "primary", f"{label} {program_name} role")
        assert_equal(tuple(row.get("program", [])), program, f"{label} {program_name} program")
        predictions = row.get("predictions")
        assert isinstance(predictions, list), f"{label} {program_name}: predictions must be a list"
        assert_equal(len(predictions), 256, f"{label} {program_name} case count")
        stratum_counts = Counter(p.get("stratum") for p in predictions)
        assert_equal(dict(stratum_counts), EXPECTED_STRATA, f"{label} {program_name} strata")

        for expected_x in range(16):
            for expected_y in range(16):
                index = expected_x * 16 + expected_y
                prediction = predictions[index]
                initial = (expected_x, expected_y)
                assert_equal(tuple(prediction.get("state", [])), initial, f"{label} {program_name} state {index}")
                direct = target_trace(initial, program)
                assert_equal(prediction.get("target_trace"), direct, f"{label} {program_name} target {index}")
                predicted = prediction.get("predicted_trace")
                assert isinstance(predicted, list), f"{label} {program_name} predicted trace {index} type"
                assert_equal(len(predicted), len(program), f"{label} {program_name} predicted trace {index} length")
                expected_prefix = [p == t for p, t in zip(predicted, direct)]
                assert_equal(prediction.get("prefix_joint_correct"), expected_prefix,
                             f"{label} {program_name} prefix {index}")
                assert_equal(prediction.get("joint_final_correct"), predicted[-1] == direct[-1],
                             f"{label} {program_name} final flag {index}")

        metrics = row.get("metrics")
        assert isinstance(metrics, dict), f"{label} {program_name}: metrics must be a dict"
        for stratum in (None, "train", "validation", "test"):
            metric_name = "all" if stratum is None else stratum
            expected = expected_metric(predictions, stratum, len(program))
            assert_equal(metrics.get(metric_name), expected, f"{label} {program_name} {metric_name} metrics")

    predicates = composition.get("primary_predicates")
    assert isinstance(predicates, dict), f"{label}: primary_predicates must be a dict"
    for row, (program_name, _) in zip(rows, PROGRAMS):
        expected = row["metrics"]["all"]["final_joint"] >= 244
        assert_equal(predicates.get(program_name), expected, f"{label} {program_name} predicate")
    return report, composition


def feature_values(input_state: tuple[int, int]) -> dict[str, bool]:
    x, y = input_state
    return {
        "gold_input_x_plus_y_ge_16": x + y >= 16,
        "gold_input_x_eq_y": x == y,
        "gold_input_x_zero": x == 0,
        "gold_input_y_zero": y == 0,
        "gold_input_any_zero": x == 0 or y == 0,
        "gold_input_both_zero": x == 0 and y == 0,
    }


FEATURE_NAMES = tuple(feature_values((0, 0)))


def first_error(predicted: list[list[int]], target: list[list[int]], program: tuple[str, ...]) -> dict[str, Any] | None:
    for index, (pred, gold) in enumerate(zip(predicted, target)):
        if pred != gold:
            return {
                "instruction_index": index + 1,
                "op": program[index],
                "predicted": pred,
                "target": gold,
            }
    return None


def error_record(prediction: dict[str, Any], program: tuple[str, ...]) -> dict[str, Any]:
    initial = [int(v) for v in prediction["state"]]
    predicted = prediction["predicted_trace"]
    target = prediction["target_trace"]
    return {
        "initial": initial,
        "stratum": prediction["stratum"],
        "predicted_trace": predicted,
        "target_trace": target,
        "first_error": first_error(predicted, target, program),
    }


def conditional_block(cases: list[tuple[dict[str, Any], tuple[str, ...], int]]) -> dict[str, Any]:
    """Summarize one instruction, where previous readouts are all correct."""
    if not cases:
        raise AssertionError("conditional block cannot have no cases")
    op = cases[0][1][cases[0][2]]
    for _, program, index in cases:
        assert_equal(program[index], op, "conditional operation identity")
    errors = sum(
        prediction["predicted_trace"][index] != prediction["target_trace"][index]
        for prediction, _, index in cases
    )
    features: dict[str, dict[str, Any]] = {}
    for name in FEATURE_NAMES:
        eligible = 0
        feature_errors = 0
        for prediction, _, index in cases:
            target = prediction["target_trace"]
            input_state = tuple(prediction["state"]) if index == 0 else tuple(target[index - 1])
            if feature_values(input_state)[name]:
                eligible += 1
                if prediction["predicted_trace"][index] != target[index]:
                    feature_errors += 1
        features[name] = {
            "denominator": eligible,
            "errors": feature_errors,
            "error_rate": rate(feature_errors, eligible),
        }
    return {
        "instruction_index": cases[0][2] + 1,
        "op": op,
        "denominator_all_previous_correct": len(cases),
        "errors": errors,
        "error_rate": rate(errors, len(cases)),
        "features": features,
    }


def analyze_row(row: dict[str, Any], program_name: str, program: tuple[str, ...]) -> dict[str, Any]:
    predictions = row["predictions"]
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in ("all", "train", "validation", "test"):
        selected = predictions if stratum == "all" else [p for p in predictions if p["stratum"] == stratum]
        final_errors = [not (p["predicted_trace"][-1] == p["target_trace"][-1]) for p in selected]
        x_errors = [p["predicted_trace"][-1][0] != p["target_trace"][-1][0] for p in selected]
        y_errors = [p["predicted_trace"][-1][1] != p["target_trace"][-1][1] for p in selected]
        both = [x and y for x, y in zip(x_errors, y_errors)]
        x_only = [x and not y for x, y in zip(x_errors, y_errors)]
        y_only = [not x and y for x, y in zip(x_errors, y_errors)]
        first_hist = Counter()
        recovery_cases: list[dict[str, Any]] = []
        error_cases: list[dict[str, Any]] = []
        for prediction in selected:
            mismatch_indices = [
                i for i, (pred, target) in enumerate(zip(prediction["predicted_trace"], prediction["target_trace"]))
                if pred != target
            ]
            if not mismatch_indices:
                first_hist["all_correct"] += 1
            else:
                first_hist[str(mismatch_indices[0] + 1)] += 1
            final_correct = prediction["predicted_trace"][-1] == prediction["target_trace"][-1]
            if final_correct and any(i < len(program) - 1 for i in mismatch_indices):
                record = error_record(prediction, program)
                recovery_cases.append(record)
            if not final_correct:
                error_cases.append(error_record(prediction, program))
        conditional_cases: list[tuple[dict[str, Any], tuple[str, ...], int]] = []
        for prediction in selected:
            for index in range(len(program)):
                previous_correct = all(
                    prediction["predicted_trace"][j] == prediction["target_trace"][j]
                    for j in range(index)
                )
                if previous_correct:
                    conditional_cases.append((prediction, program, index))
        per_instruction = [
            conditional_block(
                [case for case in conditional_cases if case[2] == index]
            )
            for index in range(len(program))
        ]
        by_stratum[stratum] = {
            "denominator": len(selected),
            "final_correct": len(selected) - sum(final_errors),
            "final_error": sum(final_errors),
            "final_error_rate": rate(sum(final_errors), len(selected)),
            "final_component_errors": {
                "x_only": sum(x_only),
                "y_only": sum(y_only),
                "both": sum(both),
                "joint_correct": len(selected) - sum(final_errors),
                "x_error": sum(x_errors),
                "y_error": sum(y_errors),
            },
            "first_divergence": {
                "histogram": {key: first_hist[key] for key in ["all_correct"] + [str(i) for i in range(1, len(program) + 1)]},
                "denominator": len(selected),
            },
            "recovery": {
                "count": len(recovery_cases),
                "rate_over_cases": rate(len(recovery_cases), len(selected)),
                "cases": recovery_cases,
            },
            "error_cases": error_cases,
            "conditional_by_instruction": per_instruction,
        }
    return {
        "program": list(program),
        "program_name": program_name,
        "source_metrics": row["metrics"],
        "by_stratum": by_stratum,
    }


def aggregate_conditional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate only matching instruction index and op; never mix positions."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row["by_stratum"]["all"]["conditional_by_instruction"]:
            grouped[(item["instruction_index"], item["op"])].append(item)
    result = []
    for (index, op), items in sorted(grouped.items()):
        denominator = sum(item["denominator_all_previous_correct"] for item in items)
        errors = sum(item["errors"] for item in items)
        features = {}
        for name in FEATURE_NAMES:
            feature_denominator = sum(item["features"][name]["denominator"] for item in items)
            feature_errors = sum(item["features"][name]["errors"] for item in items)
            features[name] = {
                "denominator": feature_denominator,
                "errors": feature_errors,
                "error_rate": rate(feature_errors, feature_denominator),
            }
        result.append({
            "instruction_index": index,
            "op": op,
            "denominator_all_previous_correct": denominator,
            "errors": errors,
            "error_rate": rate(errors, denominator),
            "features": features,
        })
    return result


def overlap_stats(left: set[str], right: set[str]) -> dict[str, Any]:
    intersection = sorted(left & right)
    union = left | right
    return {
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": rate(len(intersection), len(union)),
        "keys": intersection,
    }


def build_overlaps(error_sets: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    labels = list(error_sets)
    overall = {label: set(values["all"]) for label, values in error_sets.items()}
    by_program = {
        name: {label: set(values[name]) for label, values in error_sets.items()}
        for name, _ in PROGRAMS
    }
    pairwise: dict[str, Any] = {}
    for i, left_label in enumerate(labels):
        for right_label in labels[i + 1 :]:
            key = f"{left_label}_vs_{right_label}"
            pairwise[key] = overlap_stats(overall[left_label], overall[right_label])
    triple = set.intersection(*(overall[label] for label in labels))
    per_program_pairwise: dict[str, Any] = {}
    per_program_triple: dict[str, Any] = {}
    for name, _ in PROGRAMS:
        per_program_pairwise[name] = {}
        for i, left_label in enumerate(labels):
            for right_label in labels[i + 1 :]:
                per_program_pairwise[name][f"{left_label}_vs_{right_label}"] = overlap_stats(
                    by_program[name][left_label], by_program[name][right_label]
                )
        per_program_triple[name] = sorted(set.intersection(*(by_program[name][label] for label in labels)))
    return {
        "key_definition": "PROGRAM NAME|initial_x,initial_y; set contains primary final-error cases",
        "pairwise": pairwise,
        "triple": {"intersection_count": len(triple), "keys": sorted(triple)},
        "by_program": {
            "pairwise": per_program_pairwise,
            "triple": {
                name: {"intersection_count": len(keys), "initial_keys": keys}
                for name, keys in per_program_triple.items()
            },
        },
    }


def self_check() -> None:
    """Small DSL-only check; no source files or model code are touched."""
    assert_equal(target_trace((15, 1), ("ADD", "XOR", "SWAP")), [[0, 1], [1, 1], [1, 1]], "DSL self-check")
    assert_equal(target_trace((3, 5), ("SWAP", "ADD", "XOR")), [[5, 3], [8, 3], [11, 3]], "DSL self-check 2")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    self_check()
    frozen = json.loads(HASH_MANIFEST.read_text())
    expected_hashes = frozen.get("sha256", {})
    before_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()}
    for rel, actual in before_hashes.items():
        assert_equal(actual, expected_hashes.get(rel), f"frozen input hash {rel}")

    analyses: dict[str, dict[str, Any]] = {}
    error_sets: dict[str, dict[str, set[str]]] = {}
    reference_cases: dict[str, dict[tuple[str, tuple[int, int]], tuple[Any, Any, Any]]] = {}
    for label, path in INPUTS.items():
        report, composition = validate_composition(path, label)
        rows = composition["primary_rows"]
        analyses[label] = {
            "source_path": str(path.relative_to(ROOT)),
            "source_report_sha256": before_hashes[str(path.relative_to(ROOT))],
            "programs": [analyze_row(row, name, program) for row, (name, program) in zip(rows, PROGRAMS)],
        }
        by_program_sets: dict[str, set[str]] = {name: set() for name, _ in PROGRAMS}
        reference_cases[label] = {}
        for row, (name, program) in zip(rows, PROGRAMS):
            for prediction in row["predictions"]:
                initial = tuple(prediction["state"])
                identity = (name, initial)
                reference = (prediction["stratum"], prediction["target_trace"], prediction["state"])
                reference_cases[label][identity] = reference
                if prediction["predicted_trace"][-1] != prediction["target_trace"][-1]:
                    by_program_sets[name].add(case_key(name, initial))
        by_program_sets["all"] = set().union(*(by_program_sets[name] for name, _ in PROGRAMS))
        error_sets[label] = by_program_sets

    labels = list(INPUTS)
    for identity, reference in reference_cases[labels[0]].items():
        for label in labels[1:]:
            assert_equal(reference_cases[label].get(identity), reference, f"paired identity {identity} {label}")

    for label in labels:
        analyses[label]["conditional_by_instruction_and_op"] = aggregate_conditional(analyses[label]["programs"])
        analyses[label]["primary_totals"] = {
            "programs": len(PROGRAMS),
            "cases": sum(item["by_stratum"]["all"]["denominator"] for item in analyses[label]["programs"]),
            "final_errors": sum(item["by_stratum"]["all"]["final_error"] for item in analyses[label]["programs"]),
        }

    after_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()}
    assert_equal(after_hashes, before_hashes, "input hashes after analysis")

    output = {
        "schema": "e23_error_analysis_v1",
        "status": "complete",
        "scope": {
            "included": "six E21 primary program rows for seed0 and E22 seeds1/2",
            "primary_programs": [name for name, _ in PROGRAMS],
            "primary_cases_per_seed": 1536,
            "excluded": {
                "control": "excluded from all analyses",
                "seen": "E22 evaluation.seen subtree excluded from all analyses",
            },
            "descriptive_limit": "No predicted-state causal claims; conditional features use gold states only.",
        },
        "input_hashes": {
            "frozen_manifest": str(HASH_MANIFEST.relative_to(ROOT)),
            "expected": expected_hashes,
            "before": before_hashes,
            "after": after_hashes,
            "unchanged": before_hashes == after_hashes,
        },
        "validation": {
            "direct_tiny_dsl_targets": True,
            "source_report_metrics_recounted": True,
            "source_cost_totals_recounted": True,
            "paired_program_initial_state_identities_exact": True,
            "source_control_and_seen_not_analyzed": True,
        },
        "analyses_by_seed": analyses,
        "final_error_set_overlaps": build_overlaps(error_sets),
        "feature_definitions": {
            "gold_input_x_plus_y_ge_16": "gold ADD input sum is >=16; descriptive modulo-16 overflow indicator, not a generic carry claim",
            "gold_input_x_eq_y": "gold input x equals y",
            "gold_input_x_zero": "gold input x equals zero",
            "gold_input_y_zero": "gold input y equals zero",
            "gold_input_any_zero": "gold input x or y equals zero",
            "gold_input_both_zero": "gold input x and y both equal zero",
            "conditional_denominator": "cases whose all previous gold-vs-predicted readouts are correct; grouped by instruction index and op",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    fd, temporary_name = tempfile.mkstemp(prefix=".E23_ERROR_ANALYSIS.", suffix=".tmp", dir=str(OUTPUT.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
        os.replace(temporary_name, OUTPUT)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(OUTPUT), "schema": output["schema"], "status": output["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
