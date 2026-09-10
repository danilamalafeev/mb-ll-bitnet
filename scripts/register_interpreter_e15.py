#!/usr/bin/env python3
"""Guarded E15 runner: preflight is safe, training requires explicit clearance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

# Allow the documented ``python scripts/register_interpreter_e15.py`` command
# from the project root without requiring an installation step.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from looped_bitnet.register_e15 import (
    FROZEN_SOURCE_RELATIVE_PATHS, GRURegisterModel, OPS, PRIMARY_UPDATES,
    QATRegisterModel, batch_digest, checkpoint_digest_prefix, digest_state_dict,
    evaluate_program, gate_report, load_checkpoint, loss_for_batch,
    make_paired_batches, parameter_report, save_checkpoint, sha256_file,
    state_split, validate_manifest_schema, write_preflight, execute_program,
)
from looped_bitnet.runtime import environment, seed_everything

DEFAULT_PREFLIGHT = Path("runs/register_e15_preflight")
DEFAULT_RUN = Path("runs/register_interpreter_e15")
RUN_CONFIG = {"updates": 2000, "batch_size": 64, "cpu_threads": 4,
              "deterministic": True, "substeps_per_instruction": 4}
OPTIMIZER = {"type": "AdamW", "lr": .001, "weight_decay": .01, "foreach": False, "grad_clip": 1.0}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _seed(seed: int) -> None:
    seed_everything(seed, deterministic=True, cpu_threads=4)


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return _json_hash(manifest)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {name: sha256_file(root / relative) for name, relative in FROZEN_SOURCE_RELATIVE_PATHS.items()}


def _verify_frozen_sources(manifest: dict[str, Any]) -> None:
    validate_manifest_schema(manifest)
    if manifest["source_hashes"] != _source_hashes():
        raise ValueError("frozen source changed after preflight")


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")


def _final_targets(program: Sequence[str], states: Sequence[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    final = [execute_program(program, state) for state in states]
    return torch.tensor([x for x, _ in final]), torch.tensor([y for _, y in final])


def _validation(model: torch.nn.Module, manifest: dict[str, Any]) -> dict[str, Any]:
    """Registered selection: final joint macro, then FINAL dual-CE macro by length."""
    states = state_split()["validation"]
    rows = []
    for listed in manifest["programs"]["seen"]:
        program = tuple(listed)
        row = evaluate_program(model, program, states)
        x = torch.tensor([s[0] for s in states]); y = torch.tensor([s[1] for s in states])
        ops = torch.tensor([[OPS.index(op) for op in program] for _ in states])
        tx, ty = _final_targets(program, states)
        with torch.inference_mode():
            lx, ly = model(x, y, ops)
        row["final_dual_ce"] = float((F.cross_entropy(lx[:, -1], tx) + F.cross_entropy(ly[:, -1], ty)).item())
        rows.append(row)
    by_length = {}
    for length in (1, 2, 3):
        group = [row for row in rows if len(row["program"]) == length]
        by_length[str(length)] = {"program_count": len(group),
            "macro_final_joint": sum(row["joint_final"] / 32 for row in group) / len(group),
            "macro_final_dual_ce": sum(row["final_dual_ce"] for row in group) / len(group)}
    return {"rows": rows, "by_length": by_length,
            "macro_final_joint": sum(row["macro_final_joint"] for row in by_length.values()) / 3,
            "macro_final_dual_ce": sum(row["macro_final_dual_ce"] for row in by_length.values()) / 3}


def _better(validation: dict[str, Any], selected: dict[str, Any] | None) -> bool:
    if selected is None:
        return True
    old = selected["validation"]
    return (validation["macro_final_joint"], -validation["macro_final_dual_ce"], -validation["update"]) > (
        old["macro_final_joint"], -old["macro_final_dual_ce"], -selected["update"])


def _checkpoint_common(manifest: dict[str, Any], seed: int, batches: list, initial: str) -> dict[str, Any]:
    return {"seed": seed, "manifest_hash": _manifest_hash(manifest), "objective": manifest["objective"],
            "optimizer": OPTIMIZER, "config": RUN_CONFIG, "initial_model_digest": initial,
            "full_batch_digest": batch_digest(batches), "source_hashes": manifest["source_hashes"],
            "environment": environment(torch.device("cpu"))}


def train_toy(model: torch.nn.Module, out: Path, *, seed: int, manifest_hash: str, updates: int = 2) -> Path:
    """Test-only tiny optimizer path; the registered runner never calls this."""
    if updates not in (1, 2):
        raise ValueError("toy updates must be 1 or 2")
    _seed(seed); batches = make_paired_batches(seed)[:updates]; initial = digest_state_dict(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, foreach=False)
    for batch in batches:
        optimizer.zero_grad(set_to_none=True); loss_for_batch(model, batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
    path = out / "checkpoint_toy.pt"
    save_checkpoint(path, model, seed=seed, update=updates, manifest_hash=manifest_hash, tag="toy",
                    batch_prefix_digest=checkpoint_digest_prefix(batches, updates), optimizer=OPTIMIZER,
                    config={"toy": True}, initial_model_digest=initial, full_batch_digest=batch_digest(batches))
    return path


def train_arm(*, arm: str, seed: int, out: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """One non-resumable 2,000-update arm. Progress is durable before the next arm."""
    _refuse_nonempty(out); _seed(seed)
    model = QATRegisterModel() if arm == "qat" else GRURegisterModel()
    if model.parameter_count() != parameter_report()[f"{arm}_protocol"]:
        raise ValueError("parameter count does not match protocol")
    batches = make_paired_batches(seed)
    if len(batches) != PRIMARY_UPDATES or batch_digest(batches) != batch_digest(make_paired_batches(seed)):
        raise ValueError("batch stream is not a reproducible complete paired stream")
    initial = digest_state_dict(model); started = time.time(); out.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {"status": "running", "arm": arm, "seed": seed, "params": model.parameter_count(),
        "initial_model_digest": initial, "full_batch_digest": batch_digest(batches), "manifest_hash": _manifest_hash(manifest),
        "source_hashes": manifest["source_hashes"], "config": RUN_CONFIG, "optimizer": OPTIMIZER,
        "environment": environment(torch.device("cpu")), "validation": [], "selected": None, "latest": None,
        "primary_predicate_status": "not_evaluated_until_both_seed0_gates_pass",
        "cost": {"updates": 2000, "examples": 128000, "internal_state_updates": 1024000}}
    _atomic_json(out / "arm_progress.json", record)
    common = _checkpoint_common(manifest, seed, batches, initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, foreach=False)
    selected = None
    for update, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True); loss_for_batch(model, batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        if update % 250 == 0 or update == PRIMARY_UPDATES:
            validation = _validation(model, manifest); validation["update"] = update
            latest = out / f"checkpoint_latest_u{update}.pt"
            save_checkpoint(latest, model, update=update, tag="latest", validation=validation,
                            batch_prefix_digest=checkpoint_digest_prefix(batches, update), **common)
            record["latest"] = str(latest); record["validation"].append(validation)
            if _better(validation, selected):
                path = out / f"checkpoint_selected_u{update}.pt"
                save_checkpoint(path, model, update=update, tag="selected", validation=validation,
                                batch_prefix_digest=checkpoint_digest_prefix(batches, update), **common)
                selected = {"path": str(path), "update": update,
                            "prefix_digest": checkpoint_digest_prefix(batches, update), "validation": validation}
                record["selected"] = selected
            _atomic_json(out / "arm_progress.json", record)
    if selected is None:
        raise RuntimeError("no selected checkpoint")
    record.update(status="complete", duration_seconds=time.time() - started, selected=selected)
    _atomic_json(out / "arm_complete.json", record)
    return record


def _all_states() -> list[tuple[int, int]]:
    split = state_split(); return split["train"] + split["validation"] + split["test"]


def _subset_rows(model: torch.nn.Module, program: Sequence[str]) -> dict[str, Any]:
    result = {name: evaluate_program(model, program, states, include_predictions=True)
              for name, states in state_split().items()}
    result["all"] = evaluate_program(model, program, _all_states(), include_predictions=True)
    return result


def _clone_cache(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.clone()
    if isinstance(value, tuple): return tuple(_clone_cache(item) for item in value)
    if isinstance(value, dict): return {key: _clone_cache(item) for key, item in value.items()}
    return value


def _late_opcode_clone(model: torch.nn.Module) -> dict[str, Any]:
    states = _all_states(); x = torch.tensor([s[0] for s in states]); y = torch.tensor([s[1] for s in states])
    with torch.inference_mode():
        cache = model.init_cache(x, y); _, cache = model.step(cache, OPS.index("ADD"))
        alternatives = []
        for op in OPS:
            (lx, ly), _ = model.step(_clone_cache(cache), OPS.index(op))
            predicted = [[int(a), int(b)] for a, b in zip(lx.argmax(-1), ly.argmax(-1))]
            direct = [list(execute_program(("ADD", op), state)) for state in states]
            alternatives.append({"last_opcode": op, "joint_final": sum(a == b for a, b in zip(predicted, direct)),
                                 "predicted_final": predicted, "direct_final": direct})
    return {"prefix": ["ADD"], "alternatives": alternatives}


def _controls(model: torch.nn.Module, manifest: dict[str, Any]) -> dict[str, Any]:
    c = manifest["controls"]
    symbolic = {"first_only": _subset_rows(model, ["ADD"]), "last_only": _subset_rows(model, ["XOR"]),
        "sorted_opcode_OPS_ADD_XOR_SWAP": {"ADD_XOR_ADD": _subset_rows(model, ["ADD", "XOR", "ADD"]),
            "XOR_ADD_XOR": _subset_rows(model, ["XOR", "ADD", "XOR"]),
            "SWAP_ADD_XOR": _subset_rows(model, ["SWAP", "ADD", "XOR"])}}
    return {"identity_validation32": {"_".join(p): evaluate_program(model, p, state_split()["validation"], include_predictions=True) for p in c["identity"]},
        "equivalent": {"excluded": _subset_rows(model, c["equivalent_primary_excluded"]["program"]),
                       "allowed_equivalent": _subset_rows(model, c["equivalent_primary_excluded"]["equivalent_to"])},
        "symbolic": symbolic,
        "order_sensitive": {"ADD_XOR": _subset_rows(model, ["ADD", "XOR"]), "XOR_ADD": _subset_rows(model, ["XOR", "ADD"])},
        "late_opcode_clone": _late_opcode_clone(model)}


def _evaluate_checkpoint(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    model = QATRegisterModel() if record["arm"] == "qat" else GRURegisterModel(); selected = record["selected"]
    load_checkpoint(Path(selected["path"]), model, manifest_hash=_manifest_hash(manifest), expected_update=selected["update"],
        expected_prefix_digest=selected["prefix_digest"], expected_seed=record["seed"], expected_tag="selected",
        expected_config=RUN_CONFIG, expected_source_hashes=manifest["source_hashes"])
    primary = [{"program": program, "state_subsets": _subset_rows(model, program)} for program in manifest["programs"]["primary"]]
    predicate_rows = [{"program": row["program"], "joint_final_all": row["state_subsets"]["all"]["joint_final"],
        "required": 244, "passed": row["state_subsets"]["all"]["joint_final"] >= 244} for row in primary]
    secondary = {name: [_subset_rows(model, program) for program in programs] for name, programs in manifest["secondary"].items()}
    return {"arm": record["arm"], "seed": record["seed"], "checkpoint_path": selected["path"],
        "checkpoint_update": selected["update"], "checkpoint_prefix_digest": selected["prefix_digest"], "primary": primary,
        "primary_predicates": {"per_program": predicate_rows, "all_six_passed": all(r["passed"] for r in predicate_rows)},
        "secondary": secondary, "controls": _controls(model, manifest)}


def paired_per_example(qat: dict[str, Any], gru: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute wins/losses/ties from aligned states, preserving equal-aggregate disagreements."""
    qrows = {tuple(row["program"]): row for row in qat["primary"]}; grows = {tuple(row["program"]): row for row in gru["primary"]}; result = []
    for program in sorted(qrows):
        qpred = qrows[program]["state_subsets"]["all"]["predictions"]; gpred = grows[program]["state_subsets"]["all"]["predictions"]
        if [x["state"] for x in qpred] != [x["state"] for x in gpred]: raise ValueError("paired state order mismatch")
        outcomes = []
        for q, g in zip(qpred, gpred):
            qok, gok = q["joint_final_correct"], g["joint_final_correct"]
            outcome = "qat_win" if qok and not gok else "gru_win" if gok and not qok else "tie"
            outcomes.append({"state": q["state"], "qat_correct": qok, "gru_correct": gok, "outcome": outcome})
        result.append({"program": list(program), "wins": sum(x["outcome"] == "qat_win" for x in outcomes),
            "losses": sum(x["outcome"] == "gru_win" for x in outcomes), "ties": sum(x["outcome"] == "tie" for x in outcomes), "outcomes": outcomes})
    return result


def run_experiment(*, out: Path, preflight: Path) -> dict[str, Any]:
    _refuse_nonempty(out); manifest = json.loads((preflight / "manifest.json").read_text()); _verify_frozen_sources(manifest); out.mkdir(parents=True, exist_ok=True)
    started = time.time(); records = [train_arm(arm=arm, seed=0, out=out / f"{arm}_seed0", manifest=manifest) for arm in ("qat", "gru")]
    gate = []
    for record in records:
        model = QATRegisterModel() if record["arm"] == "qat" else GRURegisterModel(); s = record["selected"]
        load_checkpoint(Path(s["path"]), model, manifest_hash=_manifest_hash(manifest), expected_update=s["update"], expected_prefix_digest=s["prefix_digest"], expected_seed=0, expected_tag="selected", expected_config=RUN_CONFIG, expected_source_hashes=manifest["source_hashes"])
        gate.append({"arm": record["arm"], "selected_update": s["update"], "gate": gate_report(s["validation"]["rows"])})
    passed = all(item["gate"]["passed"] for item in gate)
    report: dict[str, Any] = {"format_version": 2, "status": "gate_pass" if passed else "gate_failure", "manifest_hash": _manifest_hash(manifest), "source_hashes": manifest["source_hashes"], "seed0": records, "gate": gate, "duration_seconds": time.time() - started,
        "cost": {"completed_arms": 2, "updates": 4000, "examples": 256000, "internal_state_updates": 2048000}}
    _atomic_json(out / "report.json", report)
    if not passed: return report
    for seed in (1, 2):
        for arm in ("qat", "gru"): records.append(train_arm(arm=arm, seed=seed, out=out / f"{arm}_seed{seed}", manifest=manifest))
    # Evaluation is recoverable but training is deliberately not resumable.  If
    # an evaluation/control error occurs below, this is the durable evidence
    # needed for --eval-only to load the already frozen selected checkpoints.
    report.update(status="training_complete", arms=records, duration_seconds=time.time() - started,
        cost={"completed_arms": 6, "updates": 12000, "examples": 768000, "internal_state_updates": 6144000})
    _atomic_json(out / "report.json", report)
    evaluations = [_evaluate_checkpoint(record, manifest) for record in records]
    paired = [{"seed": seed, "programs": paired_per_example(next(x for x in evaluations if x["arm"] == "qat" and x["seed"] == seed), next(x for x in evaluations if x["arm"] == "gru" and x["seed"] == seed))} for seed in (0, 1, 2)]
    report.update(status="complete", arms=records, evaluations=evaluations, paired_primary=paired, duration_seconds=time.time() - started,
        cost={"completed_arms": 6, "updates": 12000, "examples": 768000, "internal_state_updates": 6144000},
        primary_predicate_summary={f"{x['arm']}_seed{x['seed']}": x["primary_predicates"] for x in evaluations})
    _atomic_json(out / "report.json", report); return report


def recovery_evaluate(*, out: Path, preflight: Path) -> dict[str, Any]:
    """Read-only recovery. It cannot train, modify evidence, or bypass a gate failure."""
    manifest = json.loads((preflight / "manifest.json").read_text()); _verify_frozen_sources(manifest)
    report = json.loads((out / "report.json").read_text()); records = report.get("arms", [])
    gate = report.get("gate", [])
    if (report.get("status") not in {"training_complete", "complete"} or len(records) != 6
            or any(x.get("status") != "complete" for x in records) or len(gate) != 2
            or not all(row.get("gate", {}).get("passed") is True for row in gate)):
        raise ValueError("recovery requires a passed seed-0 gate and six completed frozen arms")
    return {"status": "recovery_eval_only", "evaluations": [_evaluate_checkpoint(record, manifest) for record in records]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--train-cleared", action="store_true"); parser.add_argument("--eval-only", action="store_true"); parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT); parser.add_argument("--out", type=Path, default=DEFAULT_RUN); args = parser.parse_args(argv)
    if args.train_cleared and args.eval_only: parser.error("--train-cleared and --eval-only are mutually exclusive")
    if not args.train_cleared and not args.eval_only:
        path = write_preflight(args.preflight_dir); print(json.dumps({"preflight": str(path), "params": parameter_report()}, sort_keys=True)); return 0
    if args.train_cleared: run_experiment(out=args.out, preflight=args.preflight_dir); return 0
    print(json.dumps(recovery_evaluate(out=args.out, preflight=args.preflight_dir), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
