#!/usr/bin/env python3
"""Inference-only E15 train/validation diagnostic.

This file is deliberately separate from the registered E15 runner.  It loads
only the two frozen seed-0 selected checkpoints, reconstructs the training
stream, replays validation32 before any train inference, and writes one new
machine report after all checks pass.  There is no training path here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

# Allow ``python scripts/e15_train_validation_diagnostic.py`` from the project
# root without requiring an installation step.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from looped_bitnet.register_e15 import (  # noqa: E402
    FROZEN_SOURCE_RELATIVE_PATHS,
    GRURegisterModel,
    QATRegisterModel,
    batch_digest,
    evaluate_program,
    load_checkpoint,
    make_paired_batches,
    protocol_manifest,
    sha256_file,
    validate_manifest_schema,
)
from scripts.register_interpreter_e15 import RUN_CONFIG, _manifest_hash  # noqa: E402


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_PREFLIGHT = Path("runs/register_e15_preflight/v7")
DEFAULT_PROTECTED_HASHES = Path("results/E15_TRAIN_VALIDATION_PROTECTED_HASHES.json")
DEFAULT_REPORT = Path("runs/e15_train_validation_diagnostic/report.json")
DIAGNOSTIC_SCRIPT = Path("scripts/e15_train_validation_diagnostic.py")
DIAGNOSTIC_PROTOCOL = Path("results/E15_TRAIN_VALIDATION_DIAGNOSTIC_PROTOCOL.md")

EXPECTED_MANIFEST_HASH = "bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74"
EXPECTED_BATCH_DIGEST = "63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687"
EXPECTED_CHECKPOINTS = {
    "qat": {
        "path": Path("runs/register_interpreter_e15/qat_seed0/checkpoint_selected_u2000.pt"),
        "sha256": "227253c9a7a1fd3a44ba2f666c2026501e0f2302786b6872d48df50b845c7e13",
    },
    "gru": {
        "path": Path("runs/register_interpreter_e15/gru_seed0/checkpoint_selected_u2000.pt"),
        "sha256": "5107e292507da10a9ead1e9de641fd12191d77a4406e94e3c86f5051ed8228c8",
    },
}
EXPECTED_ARM_RECORDS = {
    "qat": Path("runs/register_interpreter_e15/qat_seed0/arm_complete.json"),
    "gru": Path("runs/register_interpreter_e15/gru_seed0/arm_complete.json"),
}
MODEL_FACTORIES = {"qat": QATRegisterModel, "gru": GRURegisterModel}
METRICS = ("final_joint", "final_x", "final_y", "full_trace")
PROTECTED_PATH_COUNT = 95


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _exact_programs(manifest: Mapping[str, Any]) -> list[tuple[str, ...]]:
    programs = manifest.get("programs")
    if not isinstance(programs, Mapping):
        raise ValueError("manifest programs are missing")
    seen = programs.get("seen")
    if not isinstance(seen, list) or len(seen) != 32:
        raise ValueError("diagnostic requires exactly frozen seen32 programs")
    result: list[tuple[str, ...]] = []
    for program in seen:
        if not isinstance(program, list) or not program or any(not isinstance(op, str) for op in program):
            raise ValueError("manifest contains an invalid seen program")
        result.append(tuple(program))
    expected = [tuple(program) for program in protocol_manifest()["programs"]["seen"]]
    if result != expected:
        raise ValueError("manifest seen programs do not match frozen seen32")
    if len(set(result)) != 32:
        raise ValueError("manifest seen programs contain duplicates")
    return result


def _exact_states(manifest: Mapping[str, Any], split: str) -> list[tuple[int, int]]:
    if split not in {"train", "validation"}:
        raise ValueError("diagnostic split must be train or validation")
    values = manifest.get("state_split")
    if not isinstance(values, Mapping):
        raise ValueError("manifest state_split is missing")
    actual = values.get(split)
    expected_split = protocol_manifest()["state_split"][split]
    if not isinstance(actual, list) or actual != expected_split:
        raise ValueError(f"manifest {split} states do not match frozen order")
    states: list[tuple[int, int]] = []
    for state in actual:
        if not isinstance(state, list) or len(state) != 2 or any(not isinstance(v, int) for v in state):
            raise ValueError(f"manifest {split} contains an invalid state")
        states.append((state[0], state[1]))
    if len(states) != (192 if split == "train" else 32) or len(set(states)) != len(states):
        raise ValueError(f"manifest {split} state set is invalid")
    return states


def validate_inference_scope(manifest: Mapping[str, Any], split: str, programs: Sequence[Sequence[str]] | None = None) -> None:
    """Reject every split/program scope outside the frozen seen32 train/val cells."""
    _exact_states(manifest, split)
    expected = _exact_programs(manifest)
    if programs is not None and [tuple(program) for program in programs] != expected:
        raise ValueError("diagnostic evaluator requires exactly frozen seen32 programs in frozen order")


def _verify_protected_hashes(root: Path, snapshot: Path) -> dict[str, str]:
    expected = _read_json(_resolve(root, snapshot))
    hashes = expected.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != PROTECTED_PATH_COUNT:
        raise ValueError(f"protected hash snapshot must contain exactly {PROTECTED_PATH_COUNT} paths")
    for relative, digest in hashes.items():
        path = _resolve(root, Path(relative))
        if not path.is_file():
            raise ValueError(f"protected artifact is missing: {relative}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"protected hash mismatch: {relative}")
    return {str(key): str(value) for key, value in hashes.items()}


def _load_manifest(root: Path, preflight: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = _resolve(root, preflight) / "manifest.json"
    manifest = _read_json(manifest_path)
    validate_manifest_schema(manifest)
    if _manifest_hash(manifest) != EXPECTED_MANIFEST_HASH:
        raise ValueError("V7 manifest canonical hash mismatch")
    if manifest.get("batch_digest_seed0") != EXPECTED_BATCH_DIGEST:
        raise ValueError("V7 manifest full batch digest mismatch")
    _exact_programs(manifest)
    _exact_states(manifest, "train")
    _exact_states(manifest, "validation")
    # The registered manifest includes the test split for provenance, but this
    # diagnostic deliberately never asks _exact_states for it.
    source_hashes = manifest.get("source_hashes")
    expected_names = set(FROZEN_SOURCE_RELATIVE_PATHS)
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_names:
        raise ValueError("V7 manifest source hash set mismatch")
    current_sources = {
        name: sha256_file(root / relative) for name, relative in FROZEN_SOURCE_RELATIVE_PATHS.items()
    }
    if source_hashes != current_sources:
        raise ValueError("frozen E15 source hashes changed")
    return manifest, {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": _manifest_hash(manifest),
    }


def _coverage_table_digest(rows: Sequence[Sequence[int]]) -> str:
    return _canonical_json_hash([list(row) for row in rows])


def reconstruct_training_coverage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct seed0's exact stream and count all train pair exposures."""
    programs = _exact_programs(manifest)
    train_states = _exact_states(manifest, "train")
    batches = make_paired_batches(0)
    if len(batches) != 2000 or any(len(batch) != 64 for batch in batches):
        raise ValueError("seed0 stream is not exactly 2000 batches of 64")
    digest = batch_digest(batches)
    if digest != EXPECTED_BATCH_DIGEST or digest != manifest.get("batch_digest_seed0"):
        raise ValueError("seed0 full batch digest mismatch")

    expected_programs = set(programs)
    expected_states = set(train_states)
    counters: dict[tuple[str, ...], Counter[tuple[int, int]]] = {
        program: Counter({state: 0 for state in train_states}) for program in programs
    }
    draws = 0
    for batch in batches:
        for example in batch:
            program = tuple(example.program)
            state = (int(example.x), int(example.y))
            if program not in expected_programs:
                raise ValueError("seed0 stream contains a program outside frozen seen32")
            if state not in expected_states:
                raise ValueError("seed0 stream contains a state outside frozen train192")
            counters[program][state] += 1
            draws += 1

    by_program: dict[str, Any] = {}
    unique_total = 0
    for program in programs:
        counts = counters[program]
        table = [[state[0], state[1], counts[state]] for state in train_states]
        positive = sum(count > 0 for count in counts.values())
        unique_total += positive
        by_program[" ".join(program)] = {
            "program": list(program),
            "draws": sum(counts.values()),
            "unique_train_pairs": positive,
            "zero_frequency_pairs": len(train_states) - positive,
            "min_frequency": min(counts.values()),
            "max_frequency": max(counts.values()),
            "pair_counts_sha256": _coverage_table_digest(table),
            "stream_exposed_pairs": positive,
        }
    if draws != 128000 or unique_total != 32 * 192:
        raise ValueError("seed0 stream coverage totals are inconsistent")
    return {
        "seed": 0,
        "updates": len(batches),
        "batch_size": len(batches[0]),
        "draws": draws,
        "expected_draws": 128000,
        "batch_digest": digest,
        "program_count": len(programs),
        "train_pair_count": len(train_states),
        "unique_count": unique_total,
        "expected_unique_count": 32 * 192,
        "programs": by_program,
    }


def _normalise_row(row: Mapping[str, Any], n: int, program: Sequence[str]) -> dict[str, Any]:
    aliases = {
        "final_joint": "joint_final",
        "final_x": "final_x_correct",
        "final_y": "final_y_correct",
    }
    values: dict[str, Any] = {}
    for metric in ("final_joint", "final_x", "final_y", "full_trace"):
        source = metric if metric in row else aliases[metric]
        value = row.get(source)
        if type(value) is not int or not 0 <= value <= n:
            raise ValueError(f"invalid integer metric {source} for program {list(program)}")
        values[metric] = value
    prefix = row.get("prefix_joint")
    if not isinstance(prefix, list) or len(prefix) != len(program):
        raise ValueError(f"invalid prefix_joint for program {list(program)}")
    if any(type(value) is not int or not 0 <= value <= n for value in prefix):
        raise ValueError(f"invalid prefix_joint counts for program {list(program)}")
    result: dict[str, Any] = {
        "program": list(program),
        "n": n,
        "counts": {metric: values[metric] for metric in METRICS},
        "prefix_joint": list(prefix),
    }
    result["counts"]["prefix_joint"] = list(prefix)
    result["rates"] = {metric: values[metric] / n for metric in METRICS}
    result["prefix_rates"] = [value / n for value in prefix]
    result["rates"]["prefix_joint"] = list(result["prefix_rates"])
    return result


def evaluate_split(model: torch.nn.Module, manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Evaluate only frozen seen32 on train192 or validation32 under inference mode."""
    programs = _exact_programs(manifest)
    validate_inference_scope(manifest, split, programs)
    states = _exact_states(manifest, split)
    model.eval()
    rows = []
    with torch.inference_mode():
        for program in programs:
            # evaluate_program also uses inference_mode; the outer guard makes
            # this invariant explicit for callers and test doubles.
            row = evaluate_program(model, program, states)
            rows.append(_normalise_row(row, len(states), program))
    return rows


def _gap(train_rate: float, validation_rate: float) -> float:
    return 100.0 * (train_rate - validation_rate)


def _macro_metric(rows: Sequence[Mapping[str, Any]], metric: str, split: str) -> dict[str, Any]:
    counts = [int(row[split]["counts"][metric]) for row in rows]
    rates = [float(row[split]["rates"][metric]) for row in rows]
    return {"per_program": counts, "mean": sum(counts) / len(counts), "rate": sum(rates) / len(rates)}


def aggregate_metrics(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build per-program and unweighted length/group macro metrics."""
    if len(train_rows) != len(validation_rows):
        raise ValueError("train and validation row counts differ")
    programs = [tuple(row.get("program", ())) for row in train_rows]
    if programs != [tuple(row.get("program", ())) for row in validation_rows]:
        raise ValueError("train and validation program order differs")
    combined: list[dict[str, Any]] = []
    for train, validation in zip(train_rows, validation_rows):
        if train["n"] != 192 or validation["n"] != 32:
            raise ValueError("diagnostic denominators must be 192 and 32")
        item: dict[str, Any] = {
            "program": list(train["program"]),
            "length": len(train["program"]),
            "train": train,
            "validation": validation,
            "gaps": {metric: _gap(train["rates"][metric], validation["rates"][metric]) for metric in METRICS},
            "prefix_gaps": [_gap(a, b) for a, b in zip(train["prefix_rates"], validation["prefix_rates"])],
        }
        combined.append(item)

    def macro(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
        if not rows:
            raise ValueError(f"empty macro group: {name}")
        out: dict[str, Any] = {"name": name, "program_count": len(rows), "lengths": sorted({row["length"] for row in rows})}
        out["counts"] = {metric: _macro_metric(rows, metric, "train") for metric in METRICS}
        out["rates"] = {metric: out["counts"][metric]["rate"] for metric in METRICS}
        out["validation_counts"] = {metric: _macro_metric(rows, metric, "validation") for metric in METRICS}
        out["validation_rates"] = {metric: out["validation_counts"][metric]["rate"] for metric in METRICS}
        out["gaps"] = {metric: _gap(out["rates"][metric], out["validation_rates"][metric]) for metric in METRICS}
        max_prefix = max(len(row["program"]) for row in rows)
        out["prefix_joint"] = []
        for index in range(max_prefix):
            eligible = [row for row in rows if len(row["program"]) > index]
            train_counts = [int(row["train"]["counts"]["prefix_joint"][index]) for row in eligible]
            validation_counts = [int(row["validation"]["counts"]["prefix_joint"][index]) for row in eligible]
            train_rates = [float(row["train"]["rates"]["prefix_joint"][index]) for row in eligible]
            validation_rates = [float(row["validation"]["rates"]["prefix_joint"][index]) for row in eligible]
            train_rate = sum(train_rates) / len(train_rates)
            validation_rate = sum(validation_rates) / len(validation_rates)
            out["prefix_joint"].append({
                "index": index + 1,
                "program_count": len(eligible),
                "counts": {"train": {"per_program": train_counts, "mean": sum(train_counts) / len(train_counts)},
                            "validation": {"per_program": validation_counts, "mean": sum(validation_counts) / len(validation_counts)}},
                "rates": {"train": train_rate, "validation": validation_rate},
                "gap": _gap(train_rate, validation_rate),
            })
        return out

    length_groups = {str(length): macro([row for row in combined if row["length"] == length], f"length{length}")
                     for length in (1, 2, 3)}
    composition = macro(combined[3:11], "seen_compositions_length2")  # replaced below with rate-balanced group
    length2 = length_groups["2"]
    length3 = length_groups["3"]

    # ``seen compositions`` is the mean of the length2 and length3 macros,
    # retaining equal weight for those two lengths.
    def mean_macros(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"name": "seen_compositions", "program_count": left["program_count"] + right["program_count"],
                               "lengths": [2, 3], "counts": {}, "validation_counts": {}, "rates": {}, "validation_rates": {}, "gaps": {}}
        for metric in METRICS:
            out["counts"][metric] = {"per_macro_mean": (left["counts"][metric]["mean"] + right["counts"][metric]["mean"]) / 2}
            out["validation_counts"][metric] = {"per_macro_mean": (left["validation_counts"][metric]["mean"] + right["validation_counts"][metric]["mean"]) / 2}
            out["rates"][metric] = (left["rates"][metric] + right["rates"][metric]) / 2
            out["validation_rates"][metric] = (left["validation_rates"][metric] + right["validation_rates"][metric]) / 2
            out["gaps"][metric] = _gap(out["rates"][metric], out["validation_rates"][metric])
        out["prefix_joint"] = []
        for index in range(3):
            available = [group["prefix_joint"][index] for group in (left, right)
                         if index < len(group["prefix_joint"])]
            train_rate = sum(row["rates"]["train"] for row in available) / len(available)
            validation_rate = sum(row["rates"]["validation"] for row in available) / len(available)
            out["prefix_joint"].append({"index": index + 1, "program_count": sum(row["program_count"] for row in available),
                                        "counts": {"train": {"per_macro_mean": sum(row["counts"]["train"]["mean"] for row in available) / len(available)},
                                                    "validation": {"per_macro_mean": sum(row["counts"]["validation"]["mean"] for row in available) / len(available)}},
                                        "rates": {"train": train_rate, "validation": validation_rate},
                                        "gap": _gap(train_rate, validation_rate)})
        return out

    composition = mean_macros(length2, length3)
    return {"programs": combined, "macros": {"length": length_groups, "primitives": length_groups["1"],
                                               "seen_compositions": composition}}


def _expected_replay_rows(record: Mapping[str, Any], programs: Sequence[Sequence[str]]) -> list[Mapping[str, Any]]:
    selected = record.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("arm_complete has no selected checkpoint")
    validation = selected.get("validation")
    rows = validation.get("rows") if isinstance(validation, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("selected checkpoint has no validation rows")
    expected = [tuple(program) for program in programs]
    if [tuple(row.get("program", ())) for row in rows] != expected or len(rows) != 32:
        raise ValueError("selected validation rows have missing, duplicate, or reordered programs")
    return rows


def _replay_validation(model: torch.nn.Module, manifest: Mapping[str, Any], record: Mapping[str, Any], programs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    rows = evaluate_split(model, manifest, "validation")
    expected = _expected_replay_rows(record, programs)
    for actual, listed, program in zip(rows, expected, programs):
        if listed.get("states") != 32:
            raise ValueError(f"validation replay mismatch for {list(program)}: states")
        listed_norm = _normalise_row(listed, 32, program)
        for metric in METRICS:
            if actual["counts"][metric] != listed_norm["counts"][metric]:
                raise ValueError(f"validation replay mismatch for {list(program)}: {metric}")
        if actual["prefix_joint"] != listed_norm["prefix_joint"]:
            raise ValueError(f"validation replay mismatch for {list(program)}: prefix_joint")
    return rows


def _load_model_and_checkpoint(root: Path, manifest: Mapping[str, Any], arm: str) -> tuple[torch.nn.Module, dict[str, Any], str]:
    spec = EXPECTED_CHECKPOINTS[arm]
    path = _resolve(root, spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{arm} selected checkpoint hash mismatch")
    arm_record = _read_json(_resolve(root, EXPECTED_ARM_RECORDS[arm]))
    selected = arm_record.get("selected")
    if arm_record.get("status") != "complete" or arm_record.get("arm") != arm or arm_record.get("seed") != 0:
        raise ValueError(f"{arm} arm_complete is not the frozen seed0 record")
    if not isinstance(selected, Mapping) or selected.get("path") != str(spec["path"]):
        raise ValueError(f"{arm} selected checkpoint path mismatch")
    if selected.get("update") != 2000 or selected.get("prefix_digest") != EXPECTED_BATCH_DIGEST:
        raise ValueError(f"{arm} selected checkpoint metadata mismatch")
    if arm_record.get("full_batch_digest") != EXPECTED_BATCH_DIGEST or arm_record.get("manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise ValueError(f"{arm} arm_complete provenance mismatch")
    model = MODEL_FACTORIES[arm]()
    payload = load_checkpoint(path, model, manifest_hash=EXPECTED_MANIFEST_HASH, expected_update=2000,
                              expected_prefix_digest=EXPECTED_BATCH_DIGEST, expected_seed=0, expected_tag="selected",
                              expected_config=RUN_CONFIG, expected_source_hashes=dict(manifest["source_hashes"]))
    if payload.get("full_batch_digest") != EXPECTED_BATCH_DIGEST or payload.get("objective") != "mean_over_instructions(CE(x)+CE(y))":
        raise ValueError(f"{arm} selected checkpoint full-stream provenance mismatch")
    model.eval()
    return model, arm_record, sha256_file(path)


def _refuse_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    if path.parent.exists() and any(path.parent.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty diagnostic directory: {path.parent}")


def _write_report(path: Path, value: Mapping[str, Any]) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_diagnostic(*, root: Path = PROJECT_ROOT, preflight: Path = DEFAULT_PREFLIGHT,
                   protected_hashes: Path = DEFAULT_PROTECTED_HASHES,
                   report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    """Run the one bounded diagnostic; every failure occurs before report write."""
    root = Path(root)
    report_path = _resolve(root, Path(report_path))
    _refuse_output(report_path)
    protected = _verify_protected_hashes(root, protected_hashes)
    manifest, manifest_provenance = _load_manifest(root, preflight)
    coverage = reconstruct_training_coverage(manifest)
    programs = _exact_programs(manifest)

    loaded: dict[str, tuple[torch.nn.Module, dict[str, Any], str]] = {}
    for arm in ("qat", "gru"):
        loaded[arm] = _load_model_and_checkpoint(root, manifest, arm)

    # All validation replays complete before the first train-split inference.
    validation_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ("qat", "gru"):
        validation_rows[arm] = _replay_validation(loaded[arm][0], manifest, loaded[arm][1], programs)

    models_report: dict[str, Any] = {}
    for arm in ("qat", "gru"):
        train_rows = evaluate_split(loaded[arm][0], manifest, "train")
        aggregates = aggregate_metrics(train_rows, validation_rows[arm])
        models_report[arm] = {
            "seed": 0,
            "checkpoint": {"path": str(EXPECTED_CHECKPOINTS[arm]["path"]), "sha256": loaded[arm][2], "update": 2000},
            **aggregates,
        }

    # The protected snapshot is checked again after every inference call and
    # before the exclusive report is created.
    protected_after = _verify_protected_hashes(root, protected_hashes)
    if protected_after != protected:
        raise ValueError("protected artifacts changed during diagnostic")

    report = {
        "format_version": 1,
        "experiment": "e15_train_validation_diagnostic",
        "status": "complete",
        "scope": {"models": ["qat", "gru"], "seed": 0, "programs": "frozen seen32",
                  "splits": {"train": 192, "validation": 32}, "reserved_test_inference": False,
                  "primary_or_secondary_inference": False},
        "provenance": {"protected_hash_snapshot": str(protected_hashes), "protected_hashes_verified": len(protected_after),
                        **manifest_provenance, "manifest_hash": EXPECTED_MANIFEST_HASH,
                        "batch_digest": EXPECTED_BATCH_DIGEST, "source_hashes": manifest["source_hashes"],
                        "evaluator_sha256": sha256_file(_resolve(root, DIAGNOSTIC_SCRIPT)),
                        "diagnostic_protocol_sha256": sha256_file(_resolve(root, DIAGNOSTIC_PROTOCOL))},
        "validation_replay": {"status": "passed", "models": ["qat", "gru"],
                              "before_train_inference": True},
        "coverage": coverage,
        "models": models_report,
        "cost": {"models": 2, "programs": 32, "train_states": 192, "validation_states": 32,
                 "full_program_state_runs": 2 * 32 * (192 + 32), "readout_positions": 36736,
                 "recurrent_substeps": 146944, "substeps_per_instruction": 4},
    }
    _write_report(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded E15 inference-only train/validation diagnostic")
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--protected-hashes", type=Path, default=DEFAULT_PROTECTED_HASHES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    run_diagnostic(preflight=args.preflight_dir, protected_hashes=args.protected_hashes, report_path=args.out)
    print(json.dumps({"status": "complete", "report": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
