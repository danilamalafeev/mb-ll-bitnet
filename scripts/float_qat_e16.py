#!/usr/bin/env python3
"""Guarded E16 float-versus-QAT pilot runner.

The default command only creates a fresh preflight.  Full training requires
the explicit ``--train-cleared`` flag and always runs exactly one seed-0 arm
pair at the registered 2,000-update budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from looped_bitnet.float_qat_e16 import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    CHECKPOINT_UPDATE,
    DEFAULT_E15_PREFLIGHT,
    DEFAULT_PROTECTED_SNAPSHOT,
    E16_SCHEMA,
    E16_SOURCE_RELATIVE_PATHS,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    PROGRESS_INTERVAL,
    RUN_CONFIG,
    UPDATES,
    aggregate_metrics,
    atomic_torch_save,
    batch_digest,
    bitlinear_inventory,
    build_paired_models,
    canonical_hash,
    checkpoint_payload,
    digest_state_dict,
    evaluate_split,
    fixed_stream,
    load_checkpoint,
    load_frozen_manifest,
    make_manifest,
    manifest_hash,
    paired_outcomes,
    reconstruct_training_coverage,
    refuse_nonempty,
    source_hashes,
    target_digest,
    verify_protected_hashes,
)
from looped_bitnet.register_e15 import QATRegisterModel, loss_for_batch, sha256_file
from looped_bitnet.runtime import environment, seed_everything


DEFAULT_PREFLIGHT = Path("runs/e16_float_qat_preflight")
DEFAULT_RUN = Path("runs/e16_float_qat")


def _resolve(root: Path, path: Path) -> Path:
    return path if Path(path).is_absolute() else Path(root) / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
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


def _load_e16_manifest(preflight: Path, root: Path) -> dict[str, Any]:
    manifest = _read_json(_resolve(root, preflight) / "manifest.json")
    if manifest.get("schema") != E16_SCHEMA:
        raise ValueError("E16 preflight schema mismatch")
    if manifest.get("e15_manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise ValueError("E16 preflight E15 manifest mismatch")
    if manifest.get("stream_digest") != EXPECTED_BATCH_DIGEST:
        raise ValueError("E16 preflight stream digest mismatch")
    if manifest.get("source_hashes") != source_hashes(root):
        raise ValueError("E16 source changed after preflight")
    if manifest.get("protocol_hash") != sha256_file(root / E16_SOURCE_RELATIVE_PATHS["protocol"]):
        raise ValueError("E16 protocol changed after preflight")
    if manifest.get("config") != RUN_CONFIG or manifest.get("optimizer") != OPTIMIZER_CONFIG:
        raise ValueError("E16 config or optimizer changed after preflight")
    return manifest


def _common_initial_state(preflight: Path, manifest: Mapping[str, Any], root: Path) -> dict[str, torch.Tensor]:
    path = _resolve(root, preflight) / "common_initial_state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != E16_SCHEMA or payload.get("digest") != manifest.get("initial_digest"):
        raise ValueError("E16 common initial-state provenance mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("E16 common initial state is malformed")
    model = QATRegisterModel()
    model.load_state_dict(state, strict=True)
    if digest_state_dict(model) != manifest.get("initial_digest"):
        raise ValueError("E16 common initial-state digest mismatch")
    return state


def _model_for_arm(arm: str, initial_state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    if arm == "qat":
        model: torch.nn.Module = QATRegisterModel()
    elif arm == "float":
        _, _, model, _, _ = build_paired_models()
    else:
        raise ValueError("unknown E16 arm")
    model.load_state_dict(initial_state, strict=True)
    if (arm == "qat" and len(bitlinear_inventory(model)) != 14) or (arm == "float" and bitlinear_inventory(model)):
        raise ValueError("E16 arm module inventory mismatch")
    return model


def write_preflight(*, preflight: Path = DEFAULT_PREFLIGHT, root: Path = PROJECT_ROOT,
                    protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> Path:
    """Freeze the reviewed E16 source/config/stream before any update."""
    preflight = _resolve(root, preflight)
    refuse_nonempty(preflight)
    verify_protected_hashes(root, protected_snapshot)
    e15 = load_frozen_manifest(_resolve(root, DEFAULT_E15_PREFLIGHT), root=root)
    batches = fixed_stream()
    _, qat, float_model, initial_digest, replaced = build_paired_models()
    if digest_state_dict(qat) != digest_state_dict(float_model) or digest_state_dict(qat) != initial_digest:
        raise ValueError("paired initial state is not exact")
    manifest = make_manifest(e15_manifest=e15, initial_digest=initial_digest, replaced=replaced, batches=batches, root=root)
    preflight.mkdir(parents=True, exist_ok=True)
    _atomic_json(preflight / "manifest.json", manifest, refuse=True)
    payload = {"schema": E16_SCHEMA, "seed": 0, "digest": initial_digest, "state_dict": qat.state_dict()}
    atomic_torch_save(preflight / "common_initial_state.pt", payload)
    prefix = [[{"x": e.x, "y": e.y, "program": list(e.program)} for e in batch] for batch in batches[:2]]
    _atomic_json(preflight / "batch_prefix_seed0.json", {"batches": prefix, "target_digest": target_digest(batches)}, refuse=True)
    verify_protected_hashes(root, protected_snapshot)
    return preflight / "manifest.json"


def train_arm(*, arm: str, out: Path, manifest: Mapping[str, Any], batches: list, initial_state: Mapping[str, torch.Tensor],
              root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run one fixed, non-resumable 2,000-update arm and save u2000 first."""
    if arm not in ARMS:
        raise ValueError("unknown E16 arm")
    refuse_nonempty(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(0, deterministic=True, cpu_threads=4)
    model = _model_for_arm(arm, initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=OPTIMIZER_CONFIG["lr"], weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        betas=tuple(OPTIMIZER_CONFIG["betas"]), eps=OPTIMIZER_CONFIG["eps"],
        amsgrad=OPTIMIZER_CONFIG["amsgrad"], foreach=OPTIMIZER_CONFIG["foreach"],
    )
    record: dict[str, Any] = {
        "schema": E16_SCHEMA, "status": "running", "arm": arm, "seed": 0,
        "params": sum(p.numel() for p in model.parameters()), "initial_digest": manifest["initial_digest"],
        "stream_digest": manifest["stream_digest"], "target_digest": manifest["target_digest"],
        "e15_manifest_hash": manifest["e15_manifest_hash"], "source_hashes": manifest["source_hashes"],
        "config": RUN_CONFIG, "optimizer": OPTIMIZER_CONFIG, "progress": [], "checkpoint": None,
        "cost": {"updates": UPDATES, "examples": UPDATES * BATCH_SIZE, "internal_state_updates": UPDATES * BATCH_SIZE * 8},
    }
    _atomic_json(out / "arm_progress.json", record, refuse=True)
    try:
        for update, batch in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", enabled=False):
                loss = loss_for_batch(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIMIZER_CONFIG["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if update % PROGRESS_INTERVAL == 0 or update == UPDATES:
                value = float(loss.detach().item())
                if not torch.isfinite(torch.tensor(value)):
                    raise FloatingPointError(f"nonfinite recorded loss at update {update}")
                record["progress"].append({"update": update, "loss": value})
                _atomic_json(out / "arm_progress.json", record)
        if len(batches) != UPDATES:
            raise ValueError("E16 training stream does not contain exactly 2000 updates")
        checkpoint = out / "final_u2000.pt"
        payload = checkpoint_payload(
            model, arm=arm, initial_digest=manifest["initial_digest"], stream_digest=manifest["stream_digest"],
            prefix_digest=batch_digest(batches), e15_manifest_hash=manifest["e15_manifest_hash"],
            source_map=manifest["source_hashes"], config=manifest["config"], optimizer=manifest["optimizer"], root=root,
        )
        # This is deliberately before arm completion/report/evaluation writes.
        atomic_torch_save(checkpoint, payload)
        record.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), update=UPDATES)
        _atomic_json(out / "arm_progress.json", record)
        _atomic_json(out / "arm_complete.json", record, refuse=True)
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(out / "arm_progress.json", record)
        raise


def _arm_evaluation(model: torch.nn.Module, manifest: Mapping[str, Any]) -> dict[str, Any]:
    train_rows = evaluate_split(model, manifest, "train")
    validation_rows = evaluate_split(model, manifest, "validation")
    return {"train": train_rows, "validation": validation_rows,
            "macros": aggregate_metrics(train_rows, validation_rows)}


def _display_rows(evaluations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    qat, float_model = evaluations["qat"], evaluations["float"]
    train_q = {tuple(row["program"]): row for row in qat["train"]}
    val_q = {tuple(row["program"]): row for row in qat["validation"]}
    train_f = {tuple(row["program"]): row for row in float_model["train"]}
    val_f = {tuple(row["program"]): row for row in float_model["validation"]}
    programs = [tuple(program) for program in protocol_programs(evaluations)]
    rows = []
    for program in programs:
        rows.append({"program": list(program), "train": {"qat": train_q[program], "float": train_f[program]},
                     "validation": {"qat": val_q[program], "float": val_f[program]}})
    return rows


def protocol_programs(evaluations: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    rows = evaluations["qat"]["train"]
    if len(rows) != 32:
        raise ValueError("E16 report requires all 32 programs")
    return [row["program"] for row in rows]


def _float_minus_qat_pp(evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for split in ("train", "validation"):
        left = evaluations["float"]["macros"][split]
        right = evaluations["qat"]["macros"][split]
        result[split] = {}
        for group in ("1", "2", "3", "primitives", "seen_compositions"):
            result[split][group] = {
                metric: 100.0 * (left[group]["rates"][metric] - right[group]["rates"][metric])
                for metric in ("final_joint", "final_x", "final_y", "full_trace")
            }
    return result


def run_experiment(*, out: Path = DEFAULT_RUN, preflight: Path = DEFAULT_PREFLIGHT,
                   root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    out = _resolve(root, out)
    refuse_nonempty(out)
    verify_protected_hashes(root, protected_snapshot)
    manifest = _load_e16_manifest(preflight, root)
    batches = fixed_stream()
    if target_digest(batches) != manifest.get("target_digest"):
        raise ValueError("E16 target serialization digest mismatch")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E16 stream does not expose all 6144 train combinations")
    initial_state = _common_initial_state(preflight, manifest, root)
    out.mkdir(parents=True, exist_ok=True)
    run_manifest = {"schema": E16_SCHEMA, "manifest_hash": canonical_hash(manifest), "manifest": manifest,
                    "coverage": coverage, "environment": environment(torch.device("cpu"))}
    _atomic_json(out / "manifest.json", run_manifest, refuse=True)
    started = time.time(); records = []
    for arm in ARMS:
        records.append(train_arm(arm=arm, out=out / f"{arm}_seed0", manifest=manifest,
                                 batches=batches, initial_state=initial_state, root=root))
    training_report = {"schema": E16_SCHEMA, "status": "training_complete", "manifest_hash": canonical_hash(manifest),
                       "source_hashes": manifest["source_hashes"], "coverage": coverage, "arms": records,
                       "cost": {"arms": 2, "updates": 4000, "examples": 256000, "internal_state_updates": 2048000},
                       "duration_seconds": time.time() - started}
    _atomic_json(out / "report.json", training_report, refuse=True)
    evaluations: dict[str, dict[str, Any]] = {}
    seed_everything(0, deterministic=True, cpu_threads=4)
    for record in records:
        arm = record["arm"]
        model, _ = load_checkpoint(record["checkpoint"], arm=arm, initial_digest=manifest["initial_digest"],
                                   stream_digest=manifest["stream_digest"], e15_manifest_hash=manifest["e15_manifest_hash"],
                                   source_map=manifest["source_hashes"], config=manifest["config"], root=root)
        evaluations[arm] = _arm_evaluation(model, manifest)
    verify_protected_hashes(root, protected_snapshot)
    final_report = dict(training_report)
    final_report.update(status="complete", evaluations=evaluations,
                        paired={split: paired_outcomes(evaluations["float"][split], evaluations["qat"][split])
                                for split in ("train", "validation")},
                        float_minus_qat_pp=_float_minus_qat_pp(evaluations),
                        rows=_display_rows(evaluations), duration_seconds=time.time() - started)
    _atomic_json(out / "report.json", final_report)
    return final_report


def recovery_evaluate(*, out: Path = DEFAULT_RUN, preflight: Path = DEFAULT_PREFLIGHT,
                      root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    """Load and evaluate only already complete u2000 checkpoints."""
    out = _resolve(root, out)
    verify_protected_hashes(root, protected_snapshot)
    manifest = _load_e16_manifest(preflight, root)
    batches = fixed_stream()
    if target_digest(batches) != manifest.get("target_digest"):
        raise ValueError("E16 recovery target serialization digest mismatch")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E16 recovery stream coverage is incomplete")
    evaluations = {}
    checkpoint_paths = {arm: out / f"{arm}_seed0" / "final_u2000.pt" for arm in ARMS}
    if any(not path.is_file() for path in checkpoint_paths.values()):
        raise ValueError("E16 recovery requires exactly two complete final checkpoints")
    # Validate both final checkpoints before performing any model inference.
    loaded_models = {}
    for arm, checkpoint in checkpoint_paths.items():
        model, _ = load_checkpoint(checkpoint, arm=arm, initial_digest=manifest["initial_digest"],
                                   stream_digest=manifest["stream_digest"], e15_manifest_hash=manifest["e15_manifest_hash"],
                                   source_map=manifest["source_hashes"], config=manifest["config"], root=root)
        model.eval()
        loaded_models[arm] = model
    seed_everything(0, deterministic=True, cpu_threads=4)
    for arm, model in loaded_models.items():
        evaluations[arm] = _arm_evaluation(model, manifest)
    if set(evaluations) != set(ARMS):
        raise ValueError("E16 recovery requires exactly qat and float arms")
    verify_protected_hashes(root, protected_snapshot)
    return {"schema": E16_SCHEMA, "status": "recovery_eval_only", "evaluations": evaluations,
            "paired": {split: paired_outcomes(evaluations["float"][split], evaluations["qat"][split])
                        for split in ("train", "validation")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train-cleared", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protected-snapshot", type=Path, default=DEFAULT_PROTECTED_SNAPSHOT)
    args = parser.parse_args(argv)
    if sum(bool(flag) for flag in (args.preflight, args.train_cleared, args.eval_only)) > 1:
        parser.error("--preflight, --train-cleared and --eval-only are mutually exclusive")
    if args.train_cleared:
        run_experiment(out=args.out, preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot)
        return 0
    if args.eval_only:
        print(json.dumps(recovery_evaluate(out=args.out, preflight=args.preflight_dir,
                                            protected_snapshot=args.protected_snapshot), sort_keys=True))
        return 0
    path = write_preflight(preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot)
    print(json.dumps({"preflight": str(path), "schema": E16_SCHEMA}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
