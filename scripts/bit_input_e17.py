#!/usr/bin/env python3
"""Guarded E17 learned-embedding versus signed-bit-input pilot runner."""
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

from looped_bitnet.bit_input_e17 import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    CHECKPOINT_UPDATE,
    DEFAULT_E17_PREFLIGHT,
    DEFAULT_E17_RUN,
    DEFAULT_E15_PREFLIGHT,
    DEFAULT_PROTECTED_SNAPSHOT,
    E17_SCHEMA,
    E17_SOURCE_RELATIVE_PATHS,
    ENCODER_SPEC,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    LEARNED_PARAMETER_COUNT,
    BITS_PARAMETER_COUNT,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    RUN_CONFIG,
    UPDATES,
    aggregate_metrics,
    atomic_torch_save,
    build_paired_models,
    canonical_hash,
    checkpoint_payload,
    common_core_digest,
    digest_state_dict,
    evaluate_split,
    fixed_stream,
    load_checkpoint,
    load_manifest,
    make_manifest,
    paired_outcomes,
    reconstruct_training_coverage,
    refuse_nonempty,
    source_hashes,
    target_digest,
    verify_protected_hashes,
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


def _load_manifest(preflight: Path, root: Path) -> dict[str, Any]:
    manifest = _read_json(_resolve(root, preflight) / "manifest.json")
    expected_e15 = load_manifest(_resolve(root, DEFAULT_E15_PREFLIGHT), root=root)
    current_sources = source_hashes(root)
    if manifest.get("schema") != E17_SCHEMA:
        raise ValueError("E17 preflight schema mismatch")
    if manifest.get("e15_manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise ValueError("E17 E15 manifest hash mismatch")
    if manifest.get("stream_digest") != EXPECTED_BATCH_DIGEST:
        raise ValueError("E17 stream digest mismatch")
    if manifest.get("target_digest") != EXPECTED_TARGET_DIGEST:
        raise ValueError("E17 target digest mismatch")
    if manifest.get("source_hashes") != current_sources or set(manifest.get("source_hashes", {})) != set(E17_SOURCE_RELATIVE_PATHS):
        raise ValueError("E17 source changed after preflight")
    if manifest.get("dependency_hashes") != {name: current_sources[name] for name in current_sources
                                             if name not in {"e17_module", "e17_runner", "e17_tests", "e17_protocol"}}:
        raise ValueError("E17 dependency hash map is incomplete or changed")
    if manifest.get("protocol_hash") != current_sources["e17_protocol"]:
        raise ValueError("E17 protocol changed after preflight")
    if manifest.get("config") != RUN_CONFIG or manifest.get("config_hash") != canonical_hash(RUN_CONFIG) or manifest.get("optimizer") != OPTIMIZER_CONFIG:
        raise ValueError("E17 config or optimizer changed after preflight")
    if manifest.get("encoder_spec") != ENCODER_SPEC:
        raise ValueError("E17 encoder specification changed after preflight")
    if manifest.get("seed") != 0 or manifest.get("update") != CHECKPOINT_UPDATE:
        raise ValueError("E17 seed or fixed update changed")
    if manifest.get("programs") != expected_e15["programs"] or manifest.get("state_split") != expected_e15["state_split"]:
        raise ValueError("E17 program or state order changed")
    if manifest.get("schedule") != {"updates": UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
                                     "counts": {"1": 666, "2": 668, "3": 666}}:
        raise ValueError("E17 length schedule changed")
    return manifest


def _initial_states(preflight: Path, manifest: Mapping[str, Any], root: Path) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    # Local import keeps the annotation-only dependency out of runtime logic.
    from torch import Tensor
    result = []
    for arm, expected_digest in (("learned", manifest["learned_initial_digest"]), ("bits", manifest["bits_initial_digest"])):
        path = _resolve(root, preflight) / f"{arm}_initial_state.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != E17_SCHEMA or payload.get("arm") != arm or payload.get("digest") != expected_digest:
            raise ValueError(f"E17 {arm} initial-state provenance mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"E17 {arm} initial state is malformed")
        result.append(state)
    learned, bits, learned_digest, bits_digest, common_digest, _ = build_paired_models()
    if (manifest["learned_initial_digest"] != learned_digest or manifest["bits_initial_digest"] != bits_digest
            or manifest["common_core_digest"] != common_digest):
        raise ValueError("E17 manifest initial digests do not match reconstructed seed0 models")
    result_learned, result_bits = result
    learned.load_state_dict(result_learned, strict=True); bits.load_state_dict(result_bits, strict=True)
    if digest_state_dict(learned) != manifest["learned_initial_digest"] or digest_state_dict(bits) != manifest["bits_initial_digest"]:
        raise ValueError("E17 initial-state digest mismatch")
    if common_core_digest(learned) != manifest["common_core_digest"] or common_core_digest(bits) != manifest["common_core_digest"]:
        raise ValueError("E17 common initial-state digest mismatch")
    return result_learned, result_bits


def write_preflight(*, preflight: Path = DEFAULT_E17_PREFLIGHT, root: Path = PROJECT_ROOT,
                    protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> Path:
    preflight = _resolve(root, preflight)
    refuse_nonempty(preflight)
    verify_protected_hashes(root, protected_snapshot)
    e15_manifest = load_manifest(_resolve(root, DEFAULT_E15_PREFLIGHT), root=root)
    batches = fixed_stream()
    if batch_digest(batches) != EXPECTED_BATCH_DIGEST or target_digest(batches) != EXPECTED_TARGET_DIGEST:
        raise ValueError("E17 fixed data digest mismatch")
    learned, bits, learned_digest, bits_digest, common_digest, _ = build_paired_models()
    manifest = make_manifest(e15_manifest=e15_manifest, learned_initial_digest=learned_digest,
                             bits_initial_digest=bits_digest, common_digest=common_digest, batches=batches, root=root)
    preflight.mkdir(parents=True, exist_ok=True)
    _atomic_json(preflight / "manifest.json", manifest, refuse=True)
    atomic_torch_save(preflight / "learned_initial_state.pt",
                      {"schema": E17_SCHEMA, "arm": "learned", "digest": learned_digest, "state_dict": learned.state_dict()})
    atomic_torch_save(preflight / "bits_initial_state.pt",
                      {"schema": E17_SCHEMA, "arm": "bits", "digest": bits_digest, "state_dict": bits.state_dict()})
    _atomic_json(preflight / "initial_digests.json", {
        "schema": E17_SCHEMA, "learned": learned_digest, "bits": bits_digest, "common_core": common_digest,
        "projection_weight_sha256": sha256_tensor(learned, bits),
    }, refuse=True)
    verify_protected_hashes(root, protected_snapshot)
    return preflight / "manifest.json"


def sha256_tensor(learned: torch.nn.Module, bits: torch.nn.Module) -> dict[str, str]:
    import hashlib
    return {"x": hashlib.sha256(bits.x_embedding.projection.weight.detach().cpu().numpy().tobytes()).hexdigest(),
            "y": hashlib.sha256(bits.y_embedding.projection.weight.detach().cpu().numpy().tobytes()).hexdigest()}


def _model_for_arm(arm: str, state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    learned, bits, *_ = build_paired_models()
    model = learned if arm == "learned" else bits if arm == "bits" else None
    if model is None:
        raise ValueError("unknown E17 arm")
    model.load_state_dict(state, strict=True)
    return model


def train_arm(*, arm: str, out: Path, manifest: Mapping[str, Any], state: Mapping[str, torch.Tensor],
              batches: list, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown E17 arm")
    refuse_nonempty(out); out.mkdir(parents=True, exist_ok=True)
    seed_everything(0, deterministic=True, cpu_threads=4)
    model = _model_for_arm(arm, state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_CONFIG["lr"],
                                  weight_decay=OPTIMIZER_CONFIG["weight_decay"],
                                  betas=tuple(OPTIMIZER_CONFIG["betas"]), eps=OPTIMIZER_CONFIG["eps"],
                                  amsgrad=OPTIMIZER_CONFIG["amsgrad"], foreach=OPTIMIZER_CONFIG["foreach"])
    expected_params = LEARNED_PARAMETER_COUNT if arm == "learned" else BITS_PARAMETER_COUNT
    record: dict[str, Any] = {
        "schema": E17_SCHEMA, "status": "running", "arm": arm, "encoder_mode": arm, "seed": 0,
        "params": sum(p.numel() for p in model.parameters()), "initial_digest": manifest[f"{arm}_initial_digest"],
        "common_core_digest": manifest["common_core_digest"], "stream_digest": manifest["stream_digest"],
        "target_digest": manifest["target_digest"], "e15_manifest_hash": manifest["e15_manifest_hash"],
        "source_hashes": manifest["source_hashes"], "config": manifest["config"], "optimizer": manifest["optimizer"],
        "encoder_spec": manifest["encoder_spec"], "progress": [], "checkpoint": None,
        "cost": {"updates": UPDATES, "examples": UPDATES * BATCH_SIZE, "internal_state_updates": UPDATES * BATCH_SIZE * 8},
    }
    if record["params"] != expected_params:
        raise ValueError("E17 parameter count changed before training")
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
            if update % 250 == 0 or update == UPDATES:
                value = float(loss.detach().item())
                if not torch.isfinite(torch.tensor(value)):
                    raise FloatingPointError(f"nonfinite loss at update {update}")
                record["progress"].append({"update": update, "loss": value})
                _atomic_json(out / "arm_progress.json", record)
        checkpoint = out / "final_u2000.pt"
        payload = checkpoint_payload(model, arm=arm, initial_digest=manifest[f"{arm}_initial_digest"],
                                     common_digest=manifest["common_core_digest"], stream_digest=manifest["stream_digest"],
                                     target_stream_digest=manifest["target_digest"], e15_manifest_hash=manifest["e15_manifest_hash"],
                                     source_map=manifest["source_hashes"], config=manifest["config"],
                                     optimizer=manifest["optimizer"], root=root)
        atomic_torch_save(checkpoint, payload)
        record.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), update=UPDATES)
        _atomic_json(out / "arm_progress.json", record)
        _atomic_json(out / "arm_complete.json", record, refuse=True)
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(out / "arm_progress.json", record)
        raise


def _evaluate_arm(model: torch.nn.Module, manifest: Mapping[str, Any]) -> dict[str, Any]:
    model.eval()
    train_rows = evaluate_split(model, manifest, "train")
    validation_rows = evaluate_split(model, manifest, "validation")
    return {"train": train_rows, "validation": validation_rows, "macros": aggregate_metrics(train_rows, validation_rows)}


def _display_rows(evaluations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    learned = evaluations["learned"]; bits = evaluations["bits"]
    by = {arm: {split: {tuple(row["program"]): row for row in evaluations[arm][split]} for split in ("train", "validation")}
          for arm in ARMS}
    programs = [tuple(row["program"]) for row in learned["train"]]
    if len(programs) != 32:
        raise ValueError("E17 report requires all 32 programs")
    return [{"program": list(program), "train": {"learned": by["learned"]["train"][program], "bits": by["bits"]["train"][program]},
             "validation": {"learned": by["learned"]["validation"][program], "bits": by["bits"]["validation"][program]}}
            for program in programs]


def _bits_minus_learned_pp(evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for split in ("train", "validation"):
        bits = evaluations["bits"]["macros"][split]; learned = evaluations["learned"]["macros"][split]
        result[split] = {group: {metric: 100.0 * (bits[group]["rates"][metric] - learned[group]["rates"][metric])
                                 for metric in ("final_joint", "final_x", "final_y", "full_trace")}
                         for group in ("1", "2", "3", "primitives", "seen_compositions")}
    return result


def _load_both_checkpoints(out: Path, manifest: Mapping[str, Any], root: Path) -> dict[str, torch.nn.Module]:
    paths = {arm: out / f"{arm}_seed0" / "final_u2000.pt" for arm in ARMS}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("E17 requires exactly two complete final checkpoints")
    models = {}
    for arm, path in paths.items():
        model, _ = load_checkpoint(path, arm=arm, initial_digest=manifest[f"{arm}_initial_digest"],
                                   common_digest=manifest["common_core_digest"], stream_digest=manifest["stream_digest"],
                                   target_stream_digest=manifest["target_digest"], e15_manifest_hash=manifest["e15_manifest_hash"],
                                   source_map=manifest["source_hashes"], config=manifest["config"], root=root)
        model.eval(); models[arm] = model
    return models


def run_experiment(*, out: Path = DEFAULT_E17_RUN, preflight: Path = DEFAULT_E17_PREFLIGHT,
                   root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    out = _resolve(root, out); refuse_nonempty(out)
    verify_protected_hashes(root, protected_snapshot)
    manifest = _load_manifest(preflight, root)
    batches = fixed_stream()
    if target_digest(batches) != manifest["target_digest"]:
        raise ValueError("E17 target digest changed before training")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E17 stream does not expose all 6144 train combinations")
    learned_state, bits_state = _initial_states(preflight, manifest, root)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", {"schema": E17_SCHEMA, "manifest_hash": canonical_hash(manifest),
                                          "manifest": manifest, "coverage": coverage, "environment": environment(torch.device("cpu"))}, refuse=True)
    started = time.time(); records = []
    for arm, state in (("learned", learned_state), ("bits", bits_state)):
        records.append(train_arm(arm=arm, out=out / f"{arm}_seed0", manifest=manifest, state=state, batches=batches, root=root))
    training_report = {"schema": E17_SCHEMA, "status": "training_complete", "manifest_hash": canonical_hash(manifest),
                       "source_hashes": manifest["source_hashes"], "coverage": coverage, "arms": records,
                       "cost": {"arms": 2, "updates": 4000, "examples": 256000, "internal_state_updates": 2048000},
                       "duration_seconds": time.time() - started}
    _atomic_json(out / "report.json", training_report, refuse=True)
    models = _load_both_checkpoints(out, manifest, root)
    seed_everything(0, deterministic=True, cpu_threads=4)
    evaluations = {arm: _evaluate_arm(model, manifest) for arm, model in models.items()}
    verify_protected_hashes(root, protected_snapshot)
    final = dict(training_report)
    final.update(status="complete", evaluations=evaluations,
                 paired={split: paired_outcomes(evaluations["bits"][split], evaluations["learned"][split])
                         for split in ("train", "validation")},
                 bits_minus_learned_pp=_bits_minus_learned_pp(evaluations), rows=_display_rows(evaluations),
                 duration_seconds=time.time() - started)
    _atomic_json(out / "report.json", final)
    return final


def recovery_evaluate(*, out: Path = DEFAULT_E17_RUN, preflight: Path = DEFAULT_E17_PREFLIGHT,
                      root: Path = PROJECT_ROOT, protected_snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, Any]:
    out = _resolve(root, out); verify_protected_hashes(root, protected_snapshot)
    manifest = _load_manifest(preflight, root)
    batches = fixed_stream()
    if target_digest(batches) != manifest["target_digest"]:
        raise ValueError("E17 recovery target digest changed")
    coverage = reconstruct_training_coverage(batches, manifest)
    if not coverage["all_6144_exposed"]:
        raise ValueError("E17 recovery stream coverage is incomplete")
    models = _load_both_checkpoints(out, manifest, root)
    seed_everything(0, deterministic=True, cpu_threads=4)
    evaluations = {arm: _evaluate_arm(model, manifest) for arm, model in models.items()}
    verify_protected_hashes(root, protected_snapshot)
    return {"schema": E17_SCHEMA, "status": "recovery_eval_only", "evaluations": evaluations,
            "paired": {split: paired_outcomes(evaluations["bits"][split], evaluations["learned"][split])
                        for split in ("train", "validation")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train-cleared", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_E17_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_E17_RUN)
    parser.add_argument("--protected-snapshot", type=Path, default=DEFAULT_PROTECTED_SNAPSHOT)
    args = parser.parse_args(argv)
    if sum(bool(flag) for flag in (args.preflight, args.train_cleared, args.eval_only)) > 1:
        parser.error("--preflight, --train-cleared and --eval-only are mutually exclusive")
    if args.train_cleared:
        run_experiment(out=args.out, preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot); return 0
    if args.eval_only:
        print(json.dumps(recovery_evaluate(out=args.out, preflight=args.preflight_dir,
                                            protected_snapshot=args.protected_snapshot), sort_keys=True)); return 0
    path = write_preflight(preflight=args.preflight_dir, protected_snapshot=args.protected_snapshot)
    print(json.dumps({"preflight": str(path), "schema": E17_SCHEMA}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
