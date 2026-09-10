"""E24 fixed u8000 -> u16000 continuation adapter.

The module is intentionally an adapter around the accepted E20/E22 loaders.
It contains no training loop and no scientific evaluation calls; those live in
``scripts/continuation_e24.py``.  Importing it only defines guarded helpers.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from . import longer_native8_e20 as e20
from . import replication_e22 as e22
from .float_qat_e16 import atomic_torch_save, canonical_hash, refuse_nonempty


ROOT = e20.PROJECT_ROOT
SCHEMA = "e24_continuation_v1"
SEEDS = (0, 1, 2)
PARENT_UPDATE = 8000
ADDED_UPDATES = 8000
FINAL_UPDATE = 16000
BASE_UPDATES = 2000
REPEATS = 4
CONTINUATION_REPEATS = 4
BATCH_SIZE = 64
NATIVE_STEPS = 8
PARAMETER_COUNT = 151232
PREFLIGHT = Path("runs/e24_continuation_preflight")
RUN = Path("runs/e24_continuation")
PROTECTED = Path("results/E24_PROTECTED_HASHES.json")
PARENT_REFERENCE = Path("results/E24_PARENT_REFERENCE.json")
PROTOCOL = Path("results/E24_CONTINUATION_PROTOCOL.md")

PARENT_CHECKPOINTS = {
    0: Path("runs/e20_longer_native8/u8000.pt"),
    1: Path("runs/e22_replication/seed1/u8000.pt"),
    2: Path("runs/e22_replication/seed2/u8000.pt"),
}
PARENT_MANIFESTS = {
    0: Path("runs/e20_longer_native8_preflight/manifest.json"),
    1: Path("runs/e22_replication_preflight/manifest.json"),
    2: Path("runs/e22_replication_preflight/manifest.json"),
}
PARENT_REPORTS = {
    0: Path("runs/e20_longer_native8/report.json"),
    1: Path("runs/e22_replication/seed1/report.json"),
    2: Path("runs/e22_replication/seed2/report.json"),
}
E21_REPORT = Path("runs/e21_composition/report.json")

SOURCE_RELATIVE_PATHS = {
    "e24_module": "looped_bitnet/continuation_e24.py",
    "e24_runner": "scripts/continuation_e24.py",
    "e24_tests": "tests/test_continuation_e24.py",
    "e24_protocol": str(PROTOCOL),
}

CONFIG: dict[str, Any] = {
    "seeds": list(SEEDS),
    "parent_update": PARENT_UPDATE,
    "added_updates": ADDED_UPDATES,
    "final_update": FINAL_UPDATE,
    "base_updates": BASE_UPDATES,
    "base_repeats": REPEATS,
    "continuation_repeats": CONTINUATION_REPEATS,
    "batch_size": BATCH_SIZE,
    "native_steps": NATIVE_STEPS,
    "parameter_count": PARAMETER_COUNT,
    "optimizer": dict(e20.OPTIMIZER_CONFIG),
    "cpu_threads": 4,
    "precision": "float32",
    "autocast": False,
    "clip_grad_norm": 1.0,
}

ADDED_TRAIN_COST = {
    "updates": ADDED_UPDATES * len(SEEDS),
    "examples": ADDED_UPDATES * BATCH_SIZE * len(SEEDS),
    "internal_state_substeps": ADDED_UPDATES * BATCH_SIZE * NATIVE_STEPS * 2 * len(SEEDS),
}
CUMULATIVE_TRAIN_COST_PER_SEED = {
    "updates": FINAL_UPDATE,
    "examples": FINAL_UPDATE * BATCH_SIZE,
    "internal_state_substeps": FINAL_UPDATE * BATCH_SIZE * NATIVE_STEPS * 2,
}
EVAL_COST_PER_SEED = {
    "program_state_cases": 26880 // 3,
    "readout_positions": 70464 // 3,
    "internal_state_substeps": 563712 // 3,
    "program_forwards": 213 // 3,
}
EVAL_COST_TOTAL = {
    "program_state_cases": 26880,
    "readout_positions": 70464,
    "internal_state_substeps": 563712,
    "program_forwards": 213,
}


def _resolve(root: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {name: sha256_file(_resolve(root, Path(path))) for name, path in SOURCE_RELATIVE_PATHS.items()}


def verify_protected_hashes(root: Path = ROOT) -> dict[str, str]:
    snapshot = _read_json(_resolve(root, PROTECTED))
    hashes = snapshot.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 231:
        raise ValueError("E24 protected snapshot must contain exactly 231 files")
    for relative, expected in hashes.items():
        path = _resolve(root, Path(relative))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"E24 protected hash mismatch: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def parent_reference(root: Path = ROOT) -> dict[str, Any]:
    value = _read_json(_resolve(root, PARENT_REFERENCE))
    if value.get("scope") != "root independent read-only parent inventory; zero model forwards/updates":
        raise ValueError("E24 parent reference scope changed")
    parents = value.get("parents")
    if set(parents or {}) != {str(seed) for seed in SEEDS}:
        raise ValueError("E24 parent reference seed set changed")
    return value


def verify_parent_files(root: Path = ROOT) -> dict[str, Any]:
    reference = parent_reference(root)
    for seed in SEEDS:
        item = reference["parents"][str(seed)]
        path = _resolve(root, PARENT_CHECKPOINTS[seed])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"E24 parent checkpoint bytes changed for seed {seed}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("schema", "seed", "update", "model_digest", "optimizer_digest", "rng_digest", "training_mode"):
            if raw.get(key) != item[key]:
                raise ValueError(f"E24 parent metadata mismatch for seed {seed}: {key}")
        if raw.get("training_mode") is not True or raw.get("update") != PARENT_UPDATE:
            raise ValueError(f"E24 parent is not a training-mode u8000 checkpoint for seed {seed}")
        state = raw.get("state_dict")
        if not isinstance(state, dict) or len(state) != 32:
            raise ValueError(f"E24 parent model tensor inventory changed for seed {seed}")
        if raw.get("params", raw.get("parameter_count")) != PARAMETER_COUNT:
            raise ValueError(f"E24 parent parameter count changed for seed {seed}")
        optimizer_state = raw.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict) or len(optimizer_state.get("state", {})) != 32:
            raise ValueError(f"E24 parent optimizer inventory changed for seed {seed}")
        steps = {int(value["step"]) for value in optimizer_state["state"].values() if "step" in value}
        if steps != {PARENT_UPDATE}:
            raise ValueError(f"E24 parent optimizer steps changed for seed {seed}")
    return reference


def stream_digests() -> dict[str, str]:
    batches = e20.full_stream()
    if len(batches) != ADDED_UPDATES or len(batches[0]) != BATCH_SIZE:
        raise ValueError("E24 continuation stream length or batch size changed")
    return {"full_stream_digest": e20.batch_digest(batches), "full_target_digest": e20.target_digest(batches)}


def make_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    """Build the fixed, metadata-only E24 manifest."""
    protected = verify_protected_hashes(root)
    reference = verify_parent_files(root)
    digests = stream_digests()
    e22_manifest = _read_json(_resolve(root, PARENT_MANIFESTS[1]))
    parent_entries = {}
    for seed in SEEDS:
        item = reference["parents"][str(seed)]
        parent_entries[str(seed)] = {
            "checkpoint": str(PARENT_CHECKPOINTS[seed]),
            "manifest": str(PARENT_MANIFESTS[seed]),
            "report": str(PARENT_REPORTS[seed]),
            **item,
            "checkpoint_sha256": item["sha256"],
            "manifest_sha256": sha256_file(_resolve(root, PARENT_MANIFESTS[seed])),
            "report_sha256": sha256_file(_resolve(root, PARENT_REPORTS[seed])),
        }
    report_paths = {"e21_composition": str(E21_REPORT), **{f"seed{seed}_parent": str(PARENT_REPORTS[seed]) for seed in SEEDS}}
    return {
        "format_version": 1,
        "schema": SCHEMA,
        "experiment": "continuation_e24",
        "config": dict(CONFIG),
        "config_hash": canonical_hash(CONFIG),
        "source_hashes": source_hashes(root),
        "protected_snapshot": str(PROTECTED),
        "protected_snapshot_sha256": sha256_file(_resolve(root, PROTECTED)),
        "protected_file_count": len(protected),
        "parent_reference": str(PARENT_REFERENCE),
        "parent_reference_sha256": sha256_file(_resolve(root, PARENT_REFERENCE)),
        "parents": parent_entries,
        "parent_reports": report_paths,
        "stream": {"base_updates": BASE_UPDATES, "base_repeats": REPEATS,
                    "continuation_repeats": CONTINUATION_REPEATS, **digests},
        "train_cost": {"added_all_seeds": dict(ADDED_TRAIN_COST),
                        "cumulative_per_seed": dict(CUMULATIVE_TRAIN_COST_PER_SEED)},
        "evaluation_cost": dict(EVAL_COST_TOTAL),
        "seen": e22_manifest["seen"],
        "symbolic": e22_manifest["symbolic"],
        "state_split": e22_manifest["seen"]["state_split"],
        "thresholds": {"primitive_validation": 32, "seen_composition_validation": 31,
                       "primary_each_program": 244},
    }


def _check_tensor_inventory(raw_state: Mapping[str, Tensor], model_state: Mapping[str, Tensor], label: str) -> None:
    if list(raw_state) != list(model_state):
        raise ValueError(f"{label} tensor keys mismatch")
    for name, expected in model_state.items():
        actual = raw_state[name]
        if not isinstance(actual, Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"{label} tensor dtype/shape mismatch: {name}")


def load_parent(seed: int, *, root: Path = ROOT) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    """Load one exact accepted parent and restore its RNG after all constructors."""
    if seed not in SEEDS:
        raise ValueError("unregistered E24 seed")
    root = Path(root)
    reference = parent_reference(root)["parents"][str(seed)]
    path = _resolve(root, PARENT_CHECKPOINTS[seed])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"parent checkpoint hash mismatch for seed {seed}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if seed == 0:
        manifest = _read_json(_resolve(root, PARENT_MANIFESTS[seed]))
        model, optimizer, payload = e20.load_checkpoint(path, manifest, expected_update=PARENT_UPDATE)
    else:
        manifest = _read_json(_resolve(root, PARENT_MANIFESTS[seed]))
        model, optimizer, payload = e22.load_checkpoint(path, manifest, seed=seed, update=PARENT_UPDATE)
    _check_tensor_inventory(raw["state_dict"], model.state_dict(), f"parent seed{seed}")
    if payload.get("schema") != reference["schema"] or payload.get("seed") != seed or payload.get("update") != PARENT_UPDATE:
        raise ValueError(f"parent provenance mismatch for seed {seed}")
    if e20.digest_state_dict(model) != reference["model_digest"] or e20.digest_object(optimizer.state_dict()) != reference["optimizer_digest"]:
        raise ValueError(f"parent model/optimizer digest mismatch for seed {seed}")
    if e20.digest_object(payload["rng_state"]) != reference["rng_digest"]:
        raise ValueError(f"parent RNG digest mismatch for seed {seed}")
    # Normalize the accepted E20/E22 payload to the provenance shape consumed
    # by the E24 checkpoint schema.  The old payloads intentionally predate
    # this explicit parent-file hash field.
    payload = dict(payload)
    payload["checkpoint_sha256"] = reference["sha256"]
    payload["parent_checkpoint_sha256"] = reference["sha256"]
    torch.set_rng_state(payload["rng_state"])
    model.train(True)
    if not model.training:
        raise ValueError(f"parent training mode mismatch for seed {seed}")
    return model, optimizer, payload


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, rng_state: Tensor,
                       seed: int, update: int, manifest: Mapping[str, Any], parent: Mapping[str, Any],
                       qa: bool = False) -> dict[str, Any]:
    if seed not in SEEDS or update <= 0:
        raise ValueError("E24 checkpoint seed/update")
    expected_final = int(manifest.get("config", {}).get("final_update", FINAL_UPDATE))
    if not qa and update != expected_final:
        raise ValueError("E24 scientific checkpoint must be final u16000")
    if getattr(model, "native_steps", NATIVE_STEPS) != NATIVE_STEPS or sum(p.numel() for p in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("E24 checkpoint model inventory")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer_state = deepcopy(optimizer.state_dict())
    rng = rng_state.detach().cpu().clone()
    parent_update = int(parent.get("update", manifest.get("config", {}).get("parent_update", PARENT_UPDATE)))
    added = update - parent_update
    if added < 0:
        raise ValueError("E24 added update counter is negative")
    source = dict(manifest.get("source_hashes", {}))
    batch_size = int(manifest["config"]["qa_batch_size"]) if qa else BATCH_SIZE
    program_length = int(manifest["config"]["qa_program_length"]) if qa else 2
    return {
        "format_version": 1,
        "schema": SCHEMA,
        "seed": seed,
        "parent_schema": parent.get("schema"),
        "parent_update": parent_update,
        "update": update,
        "added_updates": added,
        "cumulative_updates": update,
        "qa": bool(qa),
        "native_steps": NATIVE_STEPS,
        "encoder_mode": "bits",
        "parameter_count": PARAMETER_COUNT,
        "manifest_digest": canonical_hash(manifest),
        "parent_checkpoint_sha256": parent.get("checkpoint_sha256", parent.get("sha256")),
        "parent_model_digest": parent.get("model_digest"),
        "parent_optimizer_digest": parent.get("optimizer_digest"),
        "parent_rng_digest": parent.get("rng_digest"),
        "source_hashes": source,
        "config": dict(manifest.get("config", CONFIG)),
        "config_hash": canonical_hash(manifest.get("config", CONFIG)),
        "state_dict": state,
        "model_digest": e20.digest_state_dict(model),
        "optimizer_state_dict": optimizer_state,
        "optimizer_digest": e20.digest_object(optimizer_state),
        "rng_state": rng,
        "rng_digest": e20.digest_object(rng),
        "training_mode": bool(model.training),
        "added_cost": {"updates": added, "examples": added * batch_size,
                        "internal_state_substeps": added * batch_size * NATIVE_STEPS * program_length},
        "cumulative_cost": {"updates": update, "examples": update * batch_size,
                             "internal_state_substeps": update * batch_size * NATIVE_STEPS * program_length},
        "fixed_final": update == expected_final,
    }


def load_checkpoint(path: Path, manifest: Mapping[str, Any], *, seed: int,
                    expected_update: int = FINAL_UPDATE, qa: bool = False,
                    parent_loader: Any = None) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    """Strictly reload an E24 checkpoint, restoring its RNG after construction."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    loader = load_parent if parent_loader is None else parent_loader
    model, optimizer, parent = loader(seed)
    _check_tensor_inventory(payload.get("state_dict", {}), model.state_dict(), f"E24 seed{seed}")
    if payload.get("schema") != SCHEMA or payload.get("seed") != seed or payload.get("update") != expected_update:
        raise ValueError("E24 checkpoint schema/seed/update mismatch")
    if bool(payload.get("qa")) != bool(qa):
        raise ValueError("E24 checkpoint QA mismatch")
    if payload.get("parameter_count") != PARAMETER_COUNT or payload.get("native_steps") != NATIVE_STEPS:
        raise ValueError("E24 checkpoint inventory mismatch")
    if not qa and expected_update != int(manifest.get("config", {}).get("final_update", FINAL_UPDATE)):
        raise ValueError("E24 scientific checkpoint update does not match manifest")
    if payload.get("encoder_mode") != "bits" or payload.get("training_mode") is not True:
        raise ValueError("E24 checkpoint encoder or training mode mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _check_tensor_inventory(payload["state_dict"], model.state_dict(), f"E24 loaded seed{seed}")
    if len(optimizer.state) != len(list(model.parameters())) or any(int(value["step"]) != expected_update for value in optimizer.state.values()):
        raise ValueError("E24 optimizer step inventory mismatch")
    for group in optimizer.param_groups:
        reference = e20.make_optimizer(model).param_groups[0]
        for key in ("lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"):
            if group[key] != reference[key]:
                raise ValueError("E24 optimizer configuration mismatch")
    expected = checkpoint_payload(
        model, optimizer, rng_state=payload["rng_state"], seed=seed, update=expected_update,
        manifest=manifest, parent=parent, qa=qa,
    )
    if set(payload) != set(expected):
        raise ValueError("E24 checkpoint fields mismatch")
    for key, value in expected.items():
        if key in {"state_dict", "optimizer_state_dict", "rng_state"}:
            continue
        if payload.get(key) != value:
            raise ValueError(f"E24 checkpoint metadata mismatch: {key}")
    torch.set_rng_state(payload["rng_state"])
    model.train(bool(payload.get("training_mode")))
    return model, optimizer, payload


__all__ = [
    "ADDED_TRAIN_COST", "ADDED_UPDATES", "BATCH_SIZE", "CONFIG", "CONTINUATION_REPEATS",
    "CUMULATIVE_TRAIN_COST_PER_SEED", "EVAL_COST_PER_SEED", "EVAL_COST_TOTAL", "FINAL_UPDATE",
    "PARENT_CHECKPOINTS", "PARENT_MANIFESTS", "PARENT_REFERENCE", "PARENT_UPDATE", "PREFLIGHT",
    "PROTECTED", "ROOT", "RUN", "SCHEMA", "SEEDS", "SOURCE_RELATIVE_PATHS", "checkpoint_payload",
    "load_checkpoint", "load_parent", "make_manifest", "parent_reference", "sha256_file",
    "source_hashes", "stream_digests", "verify_parent_files", "verify_protected_hashes",
]
