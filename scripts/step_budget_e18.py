#!/usr/bin/env python3
"""Guarded E18 native four-versus-eight substep pilot runner."""
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

from looped_bitnet.bit_input_e17 import load_manifest as load_e17_manifest  # noqa: E402
from looped_bitnet.step_budget_e18 import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    CHECKPOINT_UPDATE,
    DEFAULT_E17_PREFLIGHT,
    DEFAULT_E18_PREFLIGHT,
    DEFAULT_E18_RUN,
    DEFAULT_PROTECTED_SNAPSHOT,
    E18_SCHEMA,
    E18_SOURCE_RELATIVE_PATHS,
    ENCODER_SPEC,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    LEARNED_PARAMETER_COUNT,
    NATIVE_STEPS,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    PROGRESS_INTERVAL,
    RUN_CONFIG,
    UPDATES,
    aggregate_metrics,
    atomic_torch_save,
    build_paired_models,
    canonical_hash,
    checkpoint_payload,
    common_core_digest,
    digest_state_dict,
    e17_reference_hash,
    evaluate_split,
    fixed_stream,
    load_checkpoint,
    make_manifest,
    paired_outcomes,
    reconstruct_training_coverage,
    refuse_nonempty,
    source_hashes,
    target_digest,
)
from looped_bitnet.register_e15 import batch_digest, loss_for_batch, sha256_file
from looped_bitnet.runtime import environment, seed_everything


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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_protected_hashes(root: Path = PROJECT_ROOT, snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, str]:
    path = Path(snapshot) if Path(snapshot).is_absolute() else Path(root) / snapshot
    payload = _read_json(path); hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 138:
        raise ValueError("E18 protected snapshot must contain exactly 138 files")
    for relative, expected in hashes.items():
        target = Path(relative) if Path(relative).is_absolute() else Path(root) / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"protected hash mismatch: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def _load_manifest(preflight: Path, root: Path) -> dict[str, Any]:
    manifest = _read_json(_resolve(root, preflight) / "manifest.json")
    current = source_hashes(root)
    expected_e15 = load_e17_manifest(_resolve(root, Path("runs/register_e15_preflight/v7")), root=root)
    if manifest.get("format_version") != 1 or manifest.get("schema") != E18_SCHEMA or manifest.get("e15_manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise ValueError("E18 schema or E15 manifest mismatch")
    if manifest.get("source_hashes") != current or set(manifest.get("source_hashes", {})) != set(E18_SOURCE_RELATIVE_PATHS):
        raise ValueError("E18 source hashes changed after preflight")
    if manifest.get("protocol_hash") != current["e18_protocol"]:
        raise ValueError("E18 protocol changed after preflight")
    if manifest.get("config") != RUN_CONFIG or manifest.get("config_hash") != canonical_hash(RUN_CONFIG):
        raise ValueError("E18 config changed after preflight")
    if manifest.get("optimizer") != OPTIMIZER_CONFIG or manifest.get("encoder_spec") != ENCODER_SPEC:
        raise ValueError("E18 optimizer or encoder specification changed")
    if manifest.get("encoder_mode") != "bits" or manifest.get("parameter_count") != LEARNED_PARAMETER_COUNT:
        raise ValueError("E18 encoder mode or parameter count changed")
    if manifest.get("seed") != 0 or manifest.get("update") != CHECKPOINT_UPDATE:
        raise ValueError("E18 seed or fixed update changed")
    if manifest.get("stream_digest") != EXPECTED_BATCH_DIGEST or manifest.get("target_digest") != EXPECTED_TARGET_DIGEST:
        raise ValueError("E18 data digest changed")
    if manifest.get("e17_reference_sha256") != e17_reference_hash(root):
        raise ValueError("E17 initial reference changed")
    if manifest.get("programs") != expected_e15["programs"] or manifest.get("state_split") != expected_e15["state_split"]:
        raise ValueError("E18 program or state scope changed")
    if manifest.get("schedule") != {"updates": UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
                                     "counts": {"1": 666, "2": 668, "3": 666}}:
        raise ValueError("E18 schedule changed")
    expected_budgets = {"steps4": {"native_steps": 4, "internal_state_updates": 1024000},
                        "steps8": {"native_steps": 8, "internal_state_updates": 2048000},
                        "pair_internal_state_updates": 3072000}
    if manifest.get("budgets") != expected_budgets:
        raise ValueError("E18 budget accounting changed")
    return manifest


def _initial_states(preflight: Path, manifest: Mapping[str, Any], root: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    steps4, steps8, fresh_digest, fresh_common = build_paired_models(root=root)
    if manifest["initial_digest"] != fresh_digest or manifest["common_core_digest"] != fresh_common:
        raise ValueError("E18 manifest initial references do not match reconstructed E17 bits")
    states = []
    for arm, model in (("steps4", steps4), ("steps8", steps8)):
        payload = torch.load(_resolve(root, preflight) / f"{arm}_initial_state.pt", map_location="cpu", weights_only=True)
        if payload.get("schema") != E18_SCHEMA or payload.get("arm") != arm or payload.get("digest") != fresh_digest:
            raise ValueError(f"E18 {arm} initial-state metadata mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"E18 {arm} initial state is malformed")
        model.load_state_dict(state, strict=True)
        if digest_state_dict(model) != fresh_digest or common_core_digest(model) != fresh_common:
            raise ValueError(f"E18 {arm} initial state digest mismatch")
        states.append(state)
    return states[0], states[1]


def write_preflight(*, preflight: Path = DEFAULT_E18_PREFLIGHT, root: Path = PROJECT_ROOT,
                    protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> Path:
    preflight = _resolve(root, preflight); refuse_nonempty(preflight)
    verify_protected_hashes(root, protected_snapshot)
    e15_manifest = load_e17_manifest(_resolve(root, Path("runs/register_e15_preflight/v7")), root=root)
    batches = fixed_stream()
    if batch_digest(batches) != EXPECTED_BATCH_DIGEST or target_digest(batches) != EXPECTED_TARGET_DIGEST:
        raise ValueError("E18 fixed data digest mismatch")
    steps4, steps8, bits_digest, common_digest = build_paired_models(root=root)
    manifest = make_manifest(e15_manifest=e15_manifest, bits_digest=bits_digest, common_digest=common_digest,
                             batches=batches, root=root)
    preflight.mkdir(parents=True, exist_ok=True)
    _atomic_json(preflight / "manifest.json", manifest, refuse=True)
    for arm, model in (("steps4", steps4), ("steps8", steps8)):
        atomic_torch_save(preflight / f"{arm}_initial_state.pt",
                          {"schema": E18_SCHEMA, "arm": arm, "digest": bits_digest, "state_dict": model.state_dict()})
    _atomic_json(preflight / "initial_digests.json", {"schema": E18_SCHEMA, "initial_digest": bits_digest,
                                                       "common_core_digest": common_digest}, refuse=True)
    verify_protected_hashes(root, protected_snapshot)
    return preflight / "manifest.json"


def _model_for_arm(arm: str, state: Mapping[str, torch.Tensor], root: Path) -> torch.nn.Module:
    steps4, steps8, _, _ = build_paired_models(root=root)
    model = steps4 if arm == "steps4" else steps8 if arm == "steps8" else None
    if model is None:
        raise ValueError("unknown E18 arm")
    model.load_state_dict(state, strict=True)
    return model


def train_arm(*, arm: str, out: Path, manifest: Mapping[str, Any], state: Mapping[str, torch.Tensor],
              batches: list, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown E18 arm")
    refuse_nonempty(out); out.mkdir(parents=True, exist_ok=True)
    seed_everything(0, deterministic=True, cpu_threads=4)
    model = _model_for_arm(arm, state, root)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_CONFIG["lr"], weight_decay=OPTIMIZER_CONFIG["weight_decay"],
                                  betas=tuple(OPTIMIZER_CONFIG["betas"]), eps=OPTIMIZER_CONFIG["eps"],
                                  amsgrad=OPTIMIZER_CONFIG["amsgrad"], foreach=OPTIMIZER_CONFIG["foreach"])
    record: dict[str, Any] = {"schema": E18_SCHEMA, "status": "running", "arm": arm,
                              "native_steps": NATIVE_STEPS[arm], "encoder_mode": "bits", "seed": 0,
                              "params": sum(parameter.numel() for parameter in model.parameters()),
                              "initial_digest": manifest["initial_digest"], "common_core_digest": manifest["common_core_digest"],
                              "stream_digest": manifest["stream_digest"], "target_digest": manifest["target_digest"],
                              "source_hashes": manifest["source_hashes"], "config": manifest["config"],
                              "optimizer": manifest["optimizer"], "progress": [], "checkpoint": None,
                              "cost": {"updates": UPDATES, "examples": UPDATES * BATCH_SIZE,
                                       "internal_state_updates": UPDATES * BATCH_SIZE * NATIVE_STEPS[arm] * 2}}
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
                record["progress"].append({"update": update, "loss": float(loss.detach().item())})
                _atomic_json(out / "arm_progress.json", record)
        checkpoint = out / "final_u2000.pt"
        payload = checkpoint_payload(model, arm=arm, initial_digest=manifest["initial_digest"],
                                     common_digest=manifest["common_core_digest"], stream_digest=manifest["stream_digest"],
                                     target_stream_digest=manifest["target_digest"], e15_manifest_hash=manifest["e15_manifest_hash"],
                                     source_map=manifest["source_hashes"], config=manifest["config"], root=root)
        atomic_torch_save(checkpoint, payload)
        record.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), update=UPDATES)
        _atomic_json(out / "arm_progress.json", record); _atomic_json(out / "arm_complete.json", record, refuse=True)
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}"); _atomic_json(out / "arm_progress.json", record); raise


def _evaluate_arm(model: torch.nn.Module, manifest: Mapping[str, Any]) -> dict[str, Any]:
    model.eval(); train_rows = evaluate_split(model, manifest, "train"); validation_rows = evaluate_split(model, manifest, "validation")
    return {"train": train_rows, "validation": validation_rows, "macros": aggregate_metrics(train_rows, validation_rows)}


def _load_both_checkpoints(out: Path, manifest: Mapping[str, Any], root: Path) -> dict[str, torch.nn.Module]:
    paths = {arm: out / f"{arm}_seed0" / "final_u2000.pt" for arm in ARMS}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("E18 requires exactly two complete final checkpoints")
    models = {}
    for arm, path in paths.items():
        model, _ = load_checkpoint(path, arm=arm, initial_digest=manifest["initial_digest"], common_digest=manifest["common_core_digest"],
                                   stream_digest=manifest["stream_digest"], target_stream_digest=manifest["target_digest"],
                                   e15_manifest_hash=manifest["e15_manifest_hash"], source_map=manifest["source_hashes"],
                                   config=manifest["config"], root=root)
        model.eval(); models[arm] = model
    return models


def _display_rows(evaluations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    by = {arm: {split: {tuple(row["program"]): row for row in evaluations[arm][split]} for split in ("train", "validation")} for arm in ARMS}
    programs = [tuple(row["program"]) for row in evaluations["steps4"]["train"]]
    if len(programs) != 32:
        raise ValueError("E18 report requires all 32 programs")
    return [{"program": list(program), "train": {"steps4": by["steps4"]["train"][program], "steps8": by["steps8"]["train"][program]},
             "validation": {"steps4": by["steps4"]["validation"][program], "steps8": by["steps8"]["validation"][program]}}
            for program in programs]


def _steps8_minus_steps4_pp(evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for split in ("train", "validation"):
        eight = evaluations["steps8"]["macros"][split]; four = evaluations["steps4"]["macros"][split]
        result[split] = {group: {metric: 100.0 * (eight[group]["rates"][metric] - four[group]["rates"][metric])
                                 for metric in ("final_joint", "final_x", "final_y", "full_trace")}
                         for group in ("1", "2", "3", "primitives", "seen_compositions")}
    return result


def run_experiment(*, out: Path = DEFAULT_E18_RUN, preflight: Path = DEFAULT_E18_PREFLIGHT,
                   root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    out = _resolve(root, out); refuse_nonempty(out); verify_protected_hashes(root, protected_snapshot)
    manifest = _load_manifest(preflight, root); batches = fixed_stream()
    if target_digest(batches) != manifest["target_digest"]:
        raise ValueError("E18 target digest changed before training")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E18 stream does not expose all 6144 train combinations")
    state4, state8 = _initial_states(preflight, manifest, root)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", {"schema": E18_SCHEMA, "manifest_hash": canonical_hash(manifest), "manifest": manifest,
                                          "coverage": coverage, "environment": environment(torch.device("cpu"))}, refuse=True)
    started = time.time(); records = []
    for arm, state in (("steps4", state4), ("steps8", state8)):
        records.append(train_arm(arm=arm, out=out / f"{arm}_seed0", manifest=manifest, state=state, batches=batches, root=root))
    training_report = {"schema": E18_SCHEMA, "status": "training_complete", "manifest_hash": canonical_hash(manifest),
                       "source_hashes": manifest["source_hashes"], "coverage": coverage, "arms": records,
                       "cost": {"arms": 2, "updates": 4000, "examples": 256000, "internal_state_updates": 3072000},
                       "duration_seconds": time.time() - started}
    _atomic_json(out / "report.json", training_report, refuse=True)
    models = _load_both_checkpoints(out, manifest, root); seed_everything(0, deterministic=True, cpu_threads=4)
    evaluations = {arm: _evaluate_arm(model, manifest) for arm, model in models.items()}
    verify_protected_hashes(root, protected_snapshot)
    final = dict(training_report)
    final.update(status="complete", evaluations=evaluations,
                 evaluation_cost={"program_state_evaluations": 14336, "readout_positions": 36736,
                                  "internal_state_updates": 220416},
                 paired={split: paired_outcomes(evaluations["steps8"][split], evaluations["steps4"][split]) for split in ("train", "validation")},
                 steps8_minus_steps4_pp=_steps8_minus_steps4_pp(evaluations), rows=_display_rows(evaluations),
                 duration_seconds=time.time() - started)
    _atomic_json(out / "report.json", final)
    return final


def recovery_evaluate(*, out: Path = DEFAULT_E18_RUN, preflight: Path = DEFAULT_E18_PREFLIGHT,
                      root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    out = _resolve(root, out); verify_protected_hashes(root, protected_snapshot)
    manifest = _load_manifest(preflight, root); batches = fixed_stream()
    if target_digest(batches) != manifest["target_digest"]:
        raise ValueError("E18 recovery target digest changed")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E18 recovery stream coverage incomplete")
    _initial_states(preflight, manifest, root)
    models = _load_both_checkpoints(out, manifest, root); seed_everything(0, deterministic=True, cpu_threads=4)
    evaluations = {arm: _evaluate_arm(model, manifest) for arm, model in models.items()}; verify_protected_hashes(root, protected_snapshot)
    return {"schema": E18_SCHEMA, "status": "recovery_eval_only", "manifest_hash": canonical_hash(manifest),
            "source_hashes": manifest["source_hashes"], "coverage": coverage, "evaluations": evaluations,
            "cost": {"program_state_evaluations": 14336, "readout_positions": 36736, "internal_state_updates": 220416},
            "paired": {split: paired_outcomes(evaluations["steps8"][split], evaluations["steps4"][split]) for split in ("train", "validation")},
            "steps8_minus_steps4_pp": _steps8_minus_steps4_pp(evaluations), "rows": _display_rows(evaluations)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--train-cleared", action="store_true"); parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_E18_PREFLIGHT); parser.add_argument("--out", type=Path, default=DEFAULT_E18_RUN); parser.add_argument("--protected-snapshot", type=Path, default=DEFAULT_PROTECTED_SNAPSHOT)
    args = parser.parse_args(argv)
    if sum(bool(flag) for flag in (args.preflight, args.train_cleared, args.eval_only)) > 1: parser.error("modes are mutually exclusive")
    if args.train_cleared:
        run_experiment(out=args.out, preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot); return 0
    if args.eval_only:
        print(json.dumps(recovery_evaluate(out=args.out, preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot), sort_keys=True)); return 0
    path = write_preflight(preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot); print(json.dumps({"preflight": str(path), "schema": E18_SCHEMA}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
