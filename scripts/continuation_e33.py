#!/usr/bin/env python3
"""E33's fixed seed-0 continuation of the accepted E32 width-128 finals.

The E32 model and evaluator remain the source of truth.  This module owns the
continuation's absolute-update bookkeeping, provenance wrapper, and strict
loader for the new checkpoint schema.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import width_e32 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.register_e15 import RegisterExample, evaluate_program
from scripts import width_e32 as e32
from scripts import length_transfer_e28 as e28
from scripts.longer_native8_e20 import _update


ROOT = e.ROOT
SCHEMA = "e33_width128_seed0_continuation_v1"
QA_SCHEMA = "e33_width128_seed0_continuation_qa_v1"
SEED = 0
ARMS = ("float", "w4")
WIDTH = e.WIDTH
PARENT_UPDATE = e.UPDATES
ADDED_UPDATES = e.UPDATES
FINAL_UPDATE = PARENT_UPDATE + ADDED_UPDATES
BATCH_SIZE = e.BATCH_SIZE
PROGRESS_INTERVAL = 250
PARENT_RUN = Path("runs/e32_width")
PREFLIGHT = Path("runs/e33_continuation_preflight")
RUN = Path("runs/e33_continuation")
PROTOCOL = Path("results/E33_CONTINUATION_PROTOCOL.md")
E33_REFERENCE = Path("results/E33_REFERENCE.json")
E33_PROTECTED = Path("results/E33_PROTECTED_HASHES.json")

COST_KEYS = e.COST_KEYS
TRAIN_COST_PER_ARM = dict(e.TRAIN_COST_PER_MODEL)
EVAL_COST_PER_ARM = dict(e.EVAL_COST_PER_MODEL)
ADDED_COST_TOTAL = {key: value * len(ARMS) for key, value in TRAIN_COST_PER_ARM.items()}
EVAL_COST_TOTAL = {key: value * len(ARMS) for key, value in EVAL_COST_PER_ARM.items()}

CHECKPOINT_KEYS = frozenset({
    "schema", "qa", "arm", "width", "seed", "label", "parent_update", "added_updates",
    "cumulative_update", "update", "native_steps", "ffn_dim", "parameter_count", "training_mode",
    "parent", "parent_checkpoint", "parent_checkpoint_sha256", "parent_manifest_digest",
    "parent_source_hashes", "source_hashes", "continuation_source_hashes", "reference_hashes",
    "parent_stream_digest", "parent_target_digest", "added_stream_digest", "added_target_digest",
    "cumulative_stream_digest", "cumulative_target_digest", "manifest_digest", "config",
    "initial_digest", "initial_rng_digest", "parent_model_digest", "parent_optimizer_digest",
    "parent_rng_digest", "model_digest", "optimizer_state_dict", "optimizer_digest", "rng_state",
    "rng_digest", "state_dict", "parent_training_cost", "added_training_cost", "cumulative_training_cost",
})


def _resolve(path: Path, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


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


def _cost_zero() -> dict[str, int]:
    return dict.fromkeys(COST_KEYS, 0)


def _cost_add(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    if set(left) != set(COST_KEYS) or set(right) != set(COST_KEYS):
        raise ValueError("E33 cost keys changed")
    return {key: int(left[key]) + int(right[key]) for key in COST_KEYS}


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path))


def _continuation_source_hashes(root: Path) -> dict[str, str]:
    paths = [Path("scripts/continuation_e33.py"), Path("tests/test_continuation_e33.py"), PROTOCOL]
    return {str(path): e.sha256_file(_resolve(path, root)) for path in paths if _resolve(path, root).is_file()}


def _optional_reference_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in (E33_REFERENCE, E33_PROTECTED):
        if _resolve(path, root).is_file():
            result[str(path)] = e.sha256_file(_resolve(path, root))
    return result


def verify_e33_guards(*, root: Path = ROOT) -> None:
    """Verify E32 guards and, when frozen, the E33 reference/guard contents."""
    e.verify_protected_hashes(root)
    e.verify_reference_hashes(root)
    for guard_path, label in ((E33_REFERENCE, "reference"), (E33_PROTECTED, "protected")):
        path = _resolve(guard_path, root)
        if not path.is_file():
            continue
        guard = _read_json(path)
        hashes = guard.get("sha256")
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError(f"E33 {label} hash map is malformed")
        for relative, expected in hashes.items():
            artifact = _resolve(Path(relative), root)
            if not artifact.is_file() or e.sha256_file(artifact) != expected:
                raise ValueError(f"E33 {label} artifact changed: {relative}")


def _parent_label(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("E33 arm must be float or w4")
    return f"{arm}128_seed0"


def _parent_paths(arm: str, *, root: Path = ROOT) -> tuple[Path, Path]:
    label = _parent_label(arm)
    # E32's strict loader consumes its frozen preflight manifest; the parent
    # checkpoint itself lives under the canonical scientific run.
    return (_resolve(e.PREFLIGHT / "manifest.json", root),
            _resolve(PARENT_RUN / label / "u16000.pt", root))


def _validate_parent_metadata(payload: Mapping[str, Any], *, arm: str, path: Path) -> None:
    expected = {
        "schema": e.SCHEMA, "arm": arm, "width": WIDTH, "seed": SEED,
        "update": PARENT_UPDATE, "qa": False, "label": _parent_label(arm),
        "training_mode": True,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"E33 parent identity mismatch: {key} ({path})")
    for key in ("source_hashes", "stream_digest", "target_digest", "manifest_digest", "model_digest",
                "optimizer_digest", "rng_state", "rng_digest", "training_cost"):
        if key not in payload:
            raise ValueError(f"E33 parent provenance missing: {key}")
    if not isinstance(payload["rng_state"], torch.Tensor):
        raise ValueError("E33 parent RNG state missing")
    if set(payload["training_cost"]) != set(COST_KEYS):
        raise ValueError("E33 parent training cost keys changed")


def load_parent(arm: str, *, root: Path = ROOT, manifest: Mapping[str, Any] | None = None) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    """Load one canonical E32 parent through its strict loader."""
    manifest_path, checkpoint_path = _parent_paths(arm, root=root)
    if manifest is None:
        manifest = e32.load_manifest(manifest_path.parent, root=root)
    report = _read_json(checkpoint_path.parent / "report.json")
    if report.get("status") != "complete" or report.get("completed_updates") != PARENT_UPDATE:
        raise ValueError(f"E33 parent report is not a complete u16000 final: {checkpoint_path}")
    record_cost = report.get("training_cost")
    if record_cost != TRAIN_COST_PER_ARM:
        raise ValueError("E33 parent training cost mismatch")
    model, optimizer, payload = e.load_checkpoint(
        checkpoint_path, manifest, arm=arm, width=WIDTH, seed=SEED, update=PARENT_UPDATE,
        qa=False, root=root, training_cost=record_cost,
    )
    _validate_parent_metadata(payload, arm=arm, path=checkpoint_path)
    if payload["model_digest"] != e.digest_state_dict(model) or payload["optimizer_digest"] != e.digest_object(optimizer.state_dict()):
        raise ValueError("E33 strict parent digest mismatch")
    return model, optimizer, payload, manifest


def _validate_parent_lineage(parent: Mapping[str, Any], *, arm: str, root: Path, parent_path: Path) -> None:
    _validate_parent_metadata(parent, arm=arm, path=parent_path)
    if parent.get("stream_digest") != old.batch_digest(e.stream()) or parent.get("target_digest") != old.target_digest(e.stream()):
        raise ValueError("E33 parent stream/target digest mismatch")
    expected_path = _resolve(PARENT_RUN / _parent_label(arm) / "u16000.pt", root)
    if Path(parent_path).resolve() != expected_path.resolve():
        raise ValueError("E33 parent checkpoint path mismatch")


def make_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    """Create the immutable E33 continuation manifest from accepted E32 inputs."""
    root = Path(root)
    # Fail closed before any parent loading or prospective paid continuation.
    verify_e33_guards(root=root)
    parent_manifest = e32.load_manifest(_resolve(e.PREFLIGHT, root), root=root)
    batches = e.stream()
    parent_hashes: dict[str, str] = {}
    parent_paths: dict[str, str] = {}
    for arm in ARMS:
        _, checkpoint_path = _parent_paths(arm, root=root)
        model, optimizer, payload, _ = load_parent(arm, root=root, manifest=parent_manifest)
        del model, optimizer
        _validate_parent_lineage(payload, arm=arm, root=root, parent_path=checkpoint_path)
        parent_hashes[_parent_label(arm)] = e.sha256_file(checkpoint_path)
        parent_paths[_parent_label(arm)] = _relative(checkpoint_path, root)
    cumulative = list(e.stream()) + list(batches)
    return {
        "schema": SCHEMA,
        "config": {
            "width": WIDTH, "d_ff": e.FFN_DIM, "num_heads": e.NUM_HEADS,
            "native_steps": e.NATIVE_STEPS, "seed": SEED, "arms": list(ARMS),
            "parent_update": PARENT_UPDATE, "added_updates": ADDED_UPDATES,
            "final_update": FINAL_UPDATE, "batch_size": BATCH_SIZE,
            "progress_interval": PROGRESS_INTERVAL, "optimizer": deepcopy(e.OPTIMIZER_CONFIG),
        },
        "parent_run": str(PARENT_RUN), "parent_manifest_digest": old.canonical_hash(parent_manifest),
        "parent_checkpoints": parent_paths, "parent_checkpoint_sha256": parent_hashes,
        "parent_source_hashes": deepcopy(parent_manifest["source_hashes"]),
        "continuation_source_hashes": _continuation_source_hashes(root),
        "reference_hashes": _optional_reference_hashes(root),
        "parent_stream_digest": old.batch_digest(batches),
        "parent_target_digest": old.target_digest(batches),
        "added_stream_digest": old.batch_digest(batches),
        "added_target_digest": old.target_digest(batches),
        "cumulative_stream_digest": old.batch_digest(cumulative),
        "cumulative_target_digest": old.target_digest(cumulative),
        "added_training_cost_per_arm": deepcopy(TRAIN_COST_PER_ARM),
        "cumulative_training_cost_per_arm": _cost_add(TRAIN_COST_PER_ARM, TRAIN_COST_PER_ARM),
        "evaluation_cost_per_arm": deepcopy(EVAL_COST_PER_ARM),
        "evaluation_cost_total": deepcopy(EVAL_COST_TOTAL),
    }


def preflight(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    old.refuse_nonempty(path)
    manifest = make_manifest(root=root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    return manifest


def load_manifest(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    manifest = _read_json(path / "manifest.json")
    if manifest != make_manifest(root=root):
        raise ValueError("E33 frozen manifest changed")
    return manifest


def _checkpoint_parent_reference(parent: Mapping[str, Any], parent_path: Path, *, root: Path) -> dict[str, Any]:
    _validate_parent_lineage(parent, arm=str(parent["arm"]), root=root, parent_path=parent_path)
    return {
        "path": _relative(parent_path, root),
        "sha256": e.sha256_file(parent_path),
        "schema": parent["schema"], "label": parent["label"], "arm": parent["arm"],
        "width": parent["width"], "seed": parent["seed"], "update": parent["update"],
        "model_digest": parent["model_digest"], "optimizer_digest": parent["optimizer_digest"],
        "rng_digest": parent["rng_digest"], "manifest_digest": parent["manifest_digest"],
        "source_hashes": deepcopy(parent["source_hashes"]),
        "stream_digest": parent["stream_digest"], "target_digest": parent["target_digest"],
        "training_cost": deepcopy(parent["training_cost"]),
    }


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, manifest: Mapping[str, Any],
                       *, arm: str, seed: int = SEED, added_updates: int = ADDED_UPDATES,
                       parent: Mapping[str, Any], parent_path: Path, root: Path = ROOT,
                       training_cost: Mapping[str, int] | None = None, qa: bool = False,
                       added_batches: Sequence[Sequence[RegisterExample]] | None = None) -> dict[str, Any]:
    """Build an E33 checkpoint with absolute and incremental provenance."""
    if arm not in ARMS or type(seed) is not int or seed != SEED:
        raise ValueError("E33 checkpoint identity")
    if type(added_updates) is not int or added_updates <= 0 or (not qa and added_updates != ADDED_UPDATES):
        raise ValueError("E33 added update count")
    if not isinstance(parent, Mapping) or parent.get("update") != PARENT_UPDATE:
        raise ValueError("E33 parent update mismatch")
    if not model.training:
        raise ValueError("E33 checkpoint must preserve training mode")
    e.check_inventory(model, width=WIDTH, arm=arm)
    if training_cost is None:
        training_cost = {key: (value * added_updates // ADDED_UPDATES) for key, value in TRAIN_COST_PER_ARM.items()}
    if set(training_cost) != set(COST_KEYS) or any(type(training_cost[k]) is not int for k in COST_KEYS):
        raise ValueError("E33 added training cost malformed")
    if added_batches is None:
        added_batches = e.stream() if added_updates == ADDED_UPDATES else []
    if added_updates == ADDED_UPDATES:
        if old.batch_digest(added_batches) != manifest["added_stream_digest"] or old.target_digest(added_batches) != manifest["added_target_digest"]:
            raise ValueError("E33 added stream mismatch")
    parent_ref = _checkpoint_parent_reference(parent, parent_path, root=root)
    state = deepcopy(model.state_dict())
    opt = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    cumulative_cost = _cost_add(parent["training_cost"], training_cost)
    payload: dict[str, Any] = {
        "schema": QA_SCHEMA if qa else SCHEMA, "qa": bool(qa), "arm": arm, "width": WIDTH,
        "seed": seed, "label": _parent_label(arm), "parent_update": PARENT_UPDATE,
        "added_updates": added_updates, "cumulative_update": PARENT_UPDATE + added_updates,
        "update": PARENT_UPDATE + added_updates, "native_steps": e.NATIVE_STEPS, "ffn_dim": e.FFN_DIM,
        "parameter_count": e.PARAMETER_COUNT_128, "training_mode": True,
        "parent": parent_ref, "parent_checkpoint": parent_ref["path"],
        "parent_checkpoint_sha256": parent_ref["sha256"], "parent_manifest_digest": parent["manifest_digest"],
        "parent_source_hashes": deepcopy(parent["source_hashes"]), "source_hashes": deepcopy(parent["source_hashes"]),
        "continuation_source_hashes": deepcopy(manifest.get("continuation_source_hashes", {})),
        "reference_hashes": deepcopy(manifest.get("reference_hashes", {})),
        "parent_stream_digest": parent["stream_digest"], "parent_target_digest": parent["target_digest"],
        "added_stream_digest": manifest.get("added_stream_digest"), "added_target_digest": manifest.get("added_target_digest"),
        "cumulative_stream_digest": manifest.get("cumulative_stream_digest"),
        "cumulative_target_digest": manifest.get("cumulative_target_digest"),
        "manifest_digest": old.canonical_hash(manifest), "config": deepcopy(manifest["config"]),
        "initial_digest": parent["initial_digest"], "initial_rng_digest": parent["initial_rng_digest"],
        "parent_model_digest": parent["model_digest"], "parent_optimizer_digest": parent["optimizer_digest"],
        "parent_rng_digest": parent["rng_digest"], "model_digest": e.digest_state_dict(model),
        "optimizer_state_dict": opt, "optimizer_digest": e.digest_object(opt), "rng_state": rng,
        "rng_digest": e.digest_object(rng), "state_dict": state,
        "parent_training_cost": deepcopy(parent["training_cost"]), "added_training_cost": dict(training_cost),
        "cumulative_training_cost": cumulative_cost,
    }
    return payload


def _validate_source_hashes(payload: Mapping[str, Any], *, root: Path) -> None:
    for field in ("source_hashes", "continuation_source_hashes", "reference_hashes"):
        hashes = payload.get(field, {})
        if not isinstance(hashes, Mapping):
            raise ValueError(f"E33 {field} malformed")
        for relative, expected in hashes.items():
            path = _resolve(Path(relative), root)
            if not path.is_file() or e.sha256_file(path) != expected:
                raise ValueError(f"E33 source/reference hash mismatch: {relative}")


def load_checkpoint(path: Path, manifest: Mapping[str, Any], *, arm: str, root: Path = ROOT,
                    qa: bool = False, expected_update: int = FINAL_UPDATE):
    """Strictly restore an E33 scientific or QA checkpoint."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("E33 checkpoint payload must be a dict")
    if set(payload) != set(CHECKPOINT_KEYS):
        raise ValueError("E33 checkpoint fields mismatch")
    if type(expected_update) is not int or type(qa) is not bool:
        raise ValueError("E33 checkpoint load identity types")
    if not qa and expected_update != FINAL_UPDATE:
        raise ValueError("E33 scientific checkpoint update must be 32000")
    expected_schema = QA_SCHEMA if qa else SCHEMA
    expected = {"schema": expected_schema, "qa": qa, "arm": arm, "width": WIDTH, "seed": SEED,
                "parent_update": PARENT_UPDATE, "cumulative_update": expected_update,
                "update": expected_update, "label": _parent_label(arm), "training_mode": True}
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"E33 checkpoint identity mismatch: {key}")
    if type(payload.get("added_updates")) is not int or payload.get("added_updates") != expected_update - PARENT_UPDATE:
        raise ValueError("E33 added/cumulative update mismatch")
    parent_path = _resolve(Path(payload.get("parent_checkpoint", "")), root)
    expected_parent = _resolve(PARENT_RUN / _parent_label(arm) / "u16000.pt", root)
    if parent_path != expected_parent or not parent_path.is_file():
        raise ValueError("E33 parent checkpoint path mismatch")
    if e.sha256_file(parent_path) != payload.get("parent_checkpoint_sha256"):
        raise ValueError("E33 parent checkpoint hash mismatch")
    actual_parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if not isinstance(actual_parent, Mapping):
        raise ValueError("E33 parent checkpoint payload malformed")
    _validate_parent_metadata(actual_parent, arm=arm, path=parent_path)
    expected_parent_ref = _checkpoint_parent_reference(actual_parent, parent_path, root=root)
    if not _typed_equal(payload.get("parent"), expected_parent_ref):
        raise ValueError("E33 parent provenance mismatch")
    if payload.get("parent_source_hashes") != actual_parent.get("source_hashes") or payload.get("source_hashes") != actual_parent.get("source_hashes"):
        raise ValueError("E33 parent source provenance mismatch")
    for field in ("parent_stream_digest", "parent_target_digest", "parent_model_digest",
                  "parent_optimizer_digest", "parent_rng_digest", "parent_manifest_digest"):
        expected_field = {
            "parent_stream_digest": "stream_digest", "parent_target_digest": "target_digest",
            "parent_model_digest": "model_digest", "parent_optimizer_digest": "optimizer_digest",
            "parent_rng_digest": "rng_digest", "parent_manifest_digest": "manifest_digest",
        }[field]
        if payload.get(field) != actual_parent.get(expected_field):
            raise ValueError(f"E33 parent field mismatch: {field}")
    if payload.get("continuation_source_hashes") != manifest.get("continuation_source_hashes", {}):
        raise ValueError("E33 continuation source provenance mismatch")
    if payload.get("reference_hashes") != manifest.get("reference_hashes", {}):
        raise ValueError("E33 reference provenance mismatch")
    for field in ("added_stream_digest", "added_target_digest", "cumulative_stream_digest", "cumulative_target_digest"):
        if payload.get(field) != manifest.get(field):
            raise ValueError(f"E33 stream provenance mismatch: {field}")
    _validate_source_hashes(payload, root=root)
    model = e.build_initial_model(WIDTH, arm, SEED, root=root)
    optimizer = old.make_optimizer(model)
    e._validate_checkpoint_state(payload.get("state_dict"), model)
    e._validate_checkpoint_optimizer(payload.get("optimizer_state_dict"), model, optimizer, expected_update)
    rng = payload.get("rng_state")
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("E33 checkpoint RNG malformed")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    model.train(True)
    torch.set_rng_state(rng)
    if payload.get("model_digest") != e.digest_state_dict(model):
        raise ValueError("E33 model digest mismatch")
    if payload.get("optimizer_digest") != e.digest_object(optimizer.state_dict()):
        raise ValueError("E33 optimizer digest mismatch")
    if payload.get("rng_digest") != e.digest_object(rng):
        raise ValueError("E33 RNG digest mismatch")
    if not _typed_equal(payload.get("parent_training_cost"), actual_parent.get("training_cost")):
        raise ValueError("E33 parent training cost mismatch")
    if qa:
        added_cost = payload.get("added_training_cost")
        if (not isinstance(added_cost, Mapping) or set(added_cost) != set(COST_KEYS) or
                added_cost["program_forwards"] != expected_update - PARENT_UPDATE or
                any(type(added_cost[key]) is not int or added_cost[key] < 0 for key in COST_KEYS)):
            raise ValueError("E33 QA added training cost mismatch")
    else:
        expected_added_cost = {key: value * (expected_update - PARENT_UPDATE) // ADDED_UPDATES
                               for key, value in TRAIN_COST_PER_ARM.items()}
        if not _typed_equal(payload.get("added_training_cost"), expected_added_cost):
            raise ValueError("E33 added training cost mismatch")
    if not _typed_equal(payload.get("cumulative_training_cost"), _cost_add(payload["parent_training_cost"], payload["added_training_cost"])):
        raise ValueError("E33 cumulative training cost mismatch")
    if payload.get("manifest_digest") != old.canonical_hash(manifest):
        raise ValueError("E33 manifest digest mismatch")
    for key, value in (("config", manifest.get("config")), ("parameter_count", e.PARAMETER_COUNT_128),
                       ("native_steps", e.NATIVE_STEPS), ("ffn_dim", e.FFN_DIM)):
        if not _typed_equal(payload.get(key), value):
            raise ValueError(f"E33 checkpoint metadata mismatch: {key}")
    if payload.get("initial_digest") != actual_parent.get("initial_digest") or payload.get("initial_rng_digest") != actual_parent.get("initial_rng_digest"):
        raise ValueError("E33 initial provenance mismatch")
    return model, optimizer, payload


def _evaluation_with_cost(model: torch.nn.Module, manifest: Mapping[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    hooks = e32._hooks(model, record, "evaluation_cost")
    before = e.digest_object((model.state_dict(), old.make_optimizer(model).state_dict(), torch.get_rng_state(), model.training))
    try:
        result = e32.evaluate(model, manifest)
    finally:
        for hook in hooks:
            hook.remove()
    # Optimizer construction above has no state and is only a stable model/RNG
    # sentinel; the actual optimizer is checked by execute after this returns.
    after = e.digest_object((model.state_dict(), old.make_optimizer(model).state_dict(), torch.get_rng_state(), model.training))
    if before != after or not model.training:
        raise ValueError("E33 evaluation mutated model, RNG, or training mode")
    if record["evaluation_cost"] != EVAL_COST_PER_ARM:
        raise ValueError("E33 evaluation cost mismatch")
    return result


def _parent_comparison(parent_eval: Mapping[str, Any], current_eval: Mapping[str, Any]) -> dict[str, Any]:
    if set(parent_eval) != set(current_eval):
        raise ValueError("E33 parent/current evaluation fields changed")
    seen = {split: [] for split in ("train", "validation")}
    for split in seen:
        old_rows = {tuple(row["program"]): row for row in parent_eval["seen"][split]}
        new_rows = {tuple(row["program"]): row for row in current_eval["seen"][split]}
        if set(old_rows) != set(new_rows):
            raise ValueError("E33 seen program scope changed")
        for program in sorted(old_rows):
            left, right = old_rows[program], new_rows[program]
            def keyed(row: Mapping[str, Any]) -> dict[str, Any]:
                predictions = []
                for prediction in row["predictions"]:
                    value = dict(prediction)
                    value["stratum"] = split
                    predictions.append(value)
                return {"program": list(program), "predictions": predictions,
                        "metrics": {"all": {"final_joint": int(row["final_joint"]),
                                             "full_trace": int(row["full_trace"] )}}}
            seen[split].append(e32._pair_rows(keyed(left), keyed(right), label=f"seen-{split}"))
    primary = []
    old_primary = {tuple(row["program"]): row for row in parent_eval["composition"]["primary_rows"]}
    new_primary = {tuple(row["program"]): row for row in current_eval["composition"]["primary_rows"]}
    if set(old_primary) != set(new_primary):
        raise ValueError("E33 E21 program scope changed")
    for program in sorted(old_primary):
        a, b = old_primary[program], new_primary[program]
        primary.append(e32._pair_rows(
            {"program": list(program), "predictions": a["predictions"], "metrics": {"all": a["metrics"]["all"]}},
            {"program": list(program), "predictions": b["predictions"], "metrics": {"all": b["metrics"]["all"]}},
            label="E21",
        ))
    control_parent = parent_eval["composition"].get("control_row")
    control_current = current_eval["composition"].get("control_row")
    if not isinstance(control_parent, Mapping) or not isinstance(control_current, Mapping):
        raise ValueError("E33 E21 control row missing")
    control_program = tuple(control_parent["program"])
    control = e32._pair_rows(
        {"program": list(control_program), "predictions": control_parent["predictions"], "metrics": {"all": control_parent["metrics"]["all"]}},
        {"program": list(control_program), "predictions": control_current["predictions"], "metrics": {"all": control_current["metrics"]["all"]}},
        label="E21-control",
    )
    length = e32._length_compare(parent_eval["length"], current_eval["length"])
    old_length_rows = {tuple(row["program"]): row for row in parent_eval["length"]["rows"]}
    new_length_rows = {tuple(row["program"]): row for row in current_eval["length"]["rows"]}
    for paired_row in length["rows"]:
        program = tuple(paired_row["program"])
        old_row, new_row = old_length_rows[program], new_length_rows[program]
        by_stratum = {}
        for stratum in ("train", "validation", "test"):
            old_predictions = [item for item in old_row["predictions"] if item["stratum"] == stratum]
            new_predictions = [item for item in new_row["predictions"] if item["stratum"] == stratum]
            by_stratum[stratum] = e32._pair_rows(
                {"program": list(program), "predictions": old_predictions,
                 "metrics": {"all": old_row["metrics"][stratum]}},
                {"program": list(program), "predictions": new_predictions,
                 "metrics": {"all": new_row["metrics"][stratum]}},
                label=f"length-{stratum}",
            )
        paired_row["by_stratum"] = by_stratum
    return {"seen": seen, "composition": {"primary": primary, "control": control}, "length": length}


def assemble_report(report: Mapping[str, Any], parent_evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Join completed arm evaluations with frozen parents without model forwards."""
    result = deepcopy(dict(report))
    models = result.get("models")
    if not isinstance(models, Mapping) or set(models) != {_parent_label(arm) for arm in ARMS}:
        raise ValueError("E33 report model scope changed")
    result["status"] = "complete"
    result["added_training_cost_total"] = {
        key: sum(models[label]["added_training_cost"][key] for label in models) for key in COST_KEYS
    }
    result["evaluation_cost_total"] = {
        key: sum(models[label]["evaluation_cost"][key] for label in models) for key in COST_KEYS
    }
    result["parent_comparisons"] = {}
    for arm in ARMS:
        label = _parent_label(arm)
        if label not in parent_evaluations or "evaluation" not in models[label]:
            raise ValueError("E33 report evaluation missing")
        result["parent_comparisons"][label] = _parent_comparison(parent_evaluations[label], models[label]["evaluation"])
    result["training_benefit"] = {}
    for arm in ARMS:
        label = _parent_label(arm)
        by_length = result["parent_comparisons"][label]["length"]["by_length"]
        result["training_benefit"][label] = {
            "parent_l5_final_errors": by_length["5"]["old_final_errors"],
            "new_l5_final_errors": by_length["5"]["new_final_errors"],
            "strictly_reduces_l5_final_errors": by_length["5"]["new_final_errors"] < by_length["5"]["old_final_errors"],
        }
    result["paired_benefit"] = all(value["strictly_reduces_l5_final_errors"] for value in result["training_benefit"].values())
    result["restoration_gates"] = {
        label: {
            "seen_prerequisite": models[label]["evaluation"]["predicates"]["seen_prerequisite"],
            "e21_primary_conjunction": models[label]["evaluation"]["predicates"]["e21_primary_conjunction"],
            "l4_conjunction": models[label]["evaluation"]["predicates"]["l4_conjunction"],
            "l5_conjunction": models[label]["evaluation"]["predicates"]["l5_conjunction"],
            "combined_conjunction": models[label]["evaluation"]["predicates"]["combined_conjunction"],
            "l4_full_trace": all(row["metrics"]["all"]["full_trace"] >= 244 for row in models[label]["evaluation"]["length"]["rows"] if row["length"] == 4),
            "l5_full_trace": all(row["metrics"]["all"]["full_trace"] >= 244 for row in models[label]["evaluation"]["length"]["rows"] if row["length"] == 5),
        } for label in (_parent_label(arm) for arm in ARMS)
    }
    current_float = models[_parent_label("float")]["evaluation"]["length"]
    current_w4 = models[_parent_label("w4")]["evaluation"]["length"]
    result["paired_w4_vs_float"] = e28.paired(current_w4, current_float)
    result["signed_l5_error_gap_w4_minus_float"] = (
        sum(256 - int(row["metrics"]["all"]["final_joint"]) for row in current_w4["rows"] if row["length"] == 5)
        - sum(256 - int(row["metrics"]["all"]["final_joint"]) for row in current_float["rows"] if row["length"] == 5)
    )
    return result


def execute(out: Path, manifest: Mapping[str, Any], *, root: Path = ROOT,
            batches: Sequence[Sequence[RegisterExample]] | None = None,
            parent_loader: Any = None, evaluator: Any = None, qa: bool = False) -> dict[str, Any]:
    """Run the fixed two-arm continuation; ``qa`` is private and never scientific."""
    out = Path(out)
    old.refuse_nonempty(out)
    batches = list(e.stream() if batches is None else batches)
    if not batches or any(tuple(item.program) not in e32.SEEN_PROGRAMS for batch in batches for item in batch):
        raise ValueError("E33 stream must contain legal seen programs")
    if not qa and (len(batches) != ADDED_UPDATES or old.batch_digest(batches) != manifest["added_stream_digest"] or
                   old.target_digest(batches) != manifest["added_target_digest"]):
        raise ValueError("E33 scientific stream mismatch")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", dict(manifest), refuse=True)
    report: dict[str, Any] = {
        "schema": SCHEMA, "qa": bool(qa), "status": "running", "seed": SEED,
        "arms": list(ARMS), "parent_update": PARENT_UPDATE, "added_updates_per_arm": len(batches),
        "cumulative_final_update": PARENT_UPDATE + len(batches), "models": {},
    }
    try:
        for arm in ARMS:
            label = _parent_label(arm)
            folder = out / label
            folder.mkdir(parents=True, exist_ok=False)
            if parent_loader is None:
                model, optimizer, parent, parent_manifest = load_parent(arm, root=root)
            else:
                value = parent_loader(arm)
                if not isinstance(value, tuple) or len(value) != 4:
                    raise ValueError("E33 parent loader must return model, optimizer, payload, manifest")
                model, optimizer, parent, parent_manifest = value
            if not qa and old.canonical_hash(parent_manifest) != manifest.get("parent_manifest_digest"):
                raise ValueError("E33 parent manifest mismatch")
            parent_path = _resolve(PARENT_RUN / label / "u16000.pt", root)
            _validate_parent_lineage(parent, arm=arm, root=root, parent_path=parent_path)
            model.train(True)
            record: dict[str, Any] = {
                "label": label, "arm": arm, "width": WIDTH, "seed": SEED,
                "status": "running", "parent_update": PARENT_UPDATE, "added_updates": 0,
                "cumulative_update": PARENT_UPDATE, "attempted_updates": 0, "completed_updates": 0,
                "parent_model_digest": parent["model_digest"], "parent_optimizer_digest": parent["optimizer_digest"],
                "parent_rng_digest": parent["rng_digest"], "progress": [],
            }
            report["models"][label] = record
            started = time.monotonic()
            hooks = e32._hooks(model, record, "added_training_cost")
            try:
                for added_index, batch in enumerate(batches, 1):
                    record["attempted_updates"] = added_index
                    loss = _update(model, optimizer, batch)
                    record["completed_updates"] = added_index
                    record["added_updates"] = added_index
                    record["cumulative_update"] = PARENT_UPDATE + added_index
                    if added_index % PROGRESS_INTERVAL == 0 or added_index == len(batches):
                        record["progress"].append({"added_update": added_index,
                                                    "cumulative_update": PARENT_UPDATE + added_index,
                                                    "loss": float(loss.detach()),
                                                    "seconds": time.monotonic() - started})
                        _atomic_json(out / "progress.json", report)
            finally:
                for hook in hooks:
                    hook.remove()
            record["added_training_cost"] = record.get("added_training_cost", _cost_zero())
            record["added_training_cost_attempted"] = record.get("added_training_cost_attempted", _cost_zero())
            if not qa and record["added_training_cost"] != {key: value * len(batches) // ADDED_UPDATES for key, value in TRAIN_COST_PER_ARM.items()}:
                raise ValueError("E33 added training cost mismatch")
            record["training_seconds"] = time.monotonic() - started
            payload = checkpoint_payload(model, optimizer, manifest, arm=arm, parent=parent,
                                         parent_path=parent_path, root=root, added_updates=len(batches),
                                         training_cost=record["added_training_cost"], qa=qa, added_batches=batches)
            final_path = folder / f"u{PARENT_UPDATE + len(batches)}.pt"
            _atomic_torch(final_path, payload, refuse=True)
            restored, restored_optimizer, restored_payload = load_checkpoint(
                final_path, manifest, arm=arm, root=root, qa=qa, expected_update=PARENT_UPDATE + len(batches))
            before = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
            record["evaluation_cost"] = _cost_zero()
            eval_hooks = e32._hooks(restored, record, "evaluation_cost")
            try:
                # E32 owns the evaluation scope and expects its original manifest.
                result = (evaluator or e32.evaluate)(restored, parent_manifest)
            finally:
                for hook in eval_hooks:
                    hook.remove()
            after = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
            if before != after or not restored.training:
                raise ValueError("E33 evaluation mutated state")
            if record["evaluation_cost"] != EVAL_COST_PER_ARM:
                raise ValueError("E33 evaluation cost mismatch")
            record["evaluation"] = result
            record["checkpoint"] = _relative(final_path, root)
            record["checkpoint_sha256"] = e.sha256_file(final_path)
            record["cumulative_training_cost"] = restored_payload["cumulative_training_cost"]
            record["status"] = "complete"
            _atomic_json(folder / "report.json", record, refuse=True)
            _atomic_json(out / "progress.json", report)
        parent_evaluations = {
            _parent_label(arm): _read_json(_resolve(PARENT_RUN / _parent_label(arm) / "report.json", root))["evaluation"]
            for arm in ARMS
        }
        report = assemble_report(report, parent_evaluations)
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
    """Bounded real QA: 2 ADD updates plus two next-update branches per arm."""
    out = Path(out)
    old.refuse_nonempty(out)
    manifest = make_manifest(root=root)
    batch = [RegisterExample(0, 1, ("ADD",)), RegisterExample(2, 3, ("ADD",))]
    report: dict[str, Any] = {"schema": QA_SCHEMA, "status": "running", "qa": True, "arms": list(ARMS), "models": {}}
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", manifest, refuse=True)
    try:
        for arm in ARMS:
            model, optimizer, parent, _ = load_parent(arm, root=root)
            parent_path = _resolve(PARENT_RUN / _parent_label(arm) / "u16000.pt", root)
            model.train(True)
            record: dict[str, Any] = {"arm": arm, "seed": SEED, "parent_update": PARENT_UPDATE,
                                       "added_updates": 0, "actual_next_updates": 0,
                                       "attempted_next_updates": 0, "completed_next_updates": 0,
                                       "status": "running"}
            report["models"][_parent_label(arm)] = record
            hooks = e32._hooks(model, record, "added_training_cost")
            try:
                for index in range(1, 3):
                    _update(model, optimizer, batch)
                    record["added_updates"] = index
            finally:
                for hook in hooks:
                    hook.remove()
            payload = checkpoint_payload(model, optimizer, manifest, arm=arm, parent=parent, parent_path=parent_path,
                                         root=root, added_updates=2, training_cost=record["added_training_cost"],
                                         qa=True, added_batches=[batch, batch])
            path = out / _parent_label(arm) / "u16002.pt"
            path.parent.mkdir(parents=True, exist_ok=False)
            _atomic_torch(path, payload, refuse=True)
            restored, restored_optimizer, restored_payload = load_checkpoint(path, manifest, arm=arm, root=root, qa=True, expected_update=16002)
            before = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
            eval_hooks = e32._hooks(restored, record, "qa_evaluation_cost")
            try:
                with torch.inference_mode():
                    row = evaluate_program(restored, ("ADD",), [(0, 1), (2, 3)], include_predictions=True)
            finally:
                for hook in eval_hooks:
                    hook.remove()
            eval_cost = record["qa_evaluation_cost"]
            if eval_cost != {"program_forwards": 1, "program_state_cases": 2, "readout_positions": 2, "internal_state_substeps": 16}:
                raise ValueError("E33 QA evaluation cost mismatch")
            after = e.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
            if before != after:
                raise ValueError("E33 QA evaluation mutated state")
            snapshot = {"state_dict": deepcopy(model.state_dict()), "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "rng_state": torch.get_rng_state().clone()}
            uninterrupted = e.build_initial_model(WIDTH, arm, SEED, root=root)
            uninterrupted.load_state_dict(snapshot["state_dict"], strict=True)
            uninterrupted_optimizer = old.make_optimizer(uninterrupted)
            uninterrupted_optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            torch.set_rng_state(snapshot["rng_state"])
            record["attempted_next_updates"] += 1
            next_hooks = e32._hooks(uninterrupted, record, "uninterrupted_next_update_cost")
            try:
                _update(uninterrupted, uninterrupted_optimizer, batch)
                record["completed_next_updates"] += 1
            finally:
                for hook in next_hooks:
                    hook.remove()
            uninterrupted_rng = torch.get_rng_state().clone()
            reloaded, reloaded_optimizer, loaded_payload = load_checkpoint(path, manifest, arm=arm, root=root, qa=True, expected_update=16002)
            torch.set_rng_state(loaded_payload["rng_state"])
            record["attempted_next_updates"] += 1
            next_hooks = e32._hooks(reloaded, record, "reloaded_next_update_cost")
            try:
                _update(reloaded, reloaded_optimizer, batch)
                record["completed_next_updates"] += 1
            finally:
                for hook in next_hooks:
                    hook.remove()
            reloaded_rng = torch.get_rng_state().clone()
            if (e.digest_state_dict(uninterrupted) != e.digest_state_dict(reloaded) or
                    e.digest_object(uninterrupted_optimizer.state_dict()) != e.digest_object(reloaded_optimizer.state_dict()) or
                    not torch.equal(uninterrupted_rng, reloaded_rng)):
                raise ValueError(f"E33 QA reload next-update mismatch: {arm}")
            record.update({"status": "complete", "eval": {"joint_final": row["joint_final"], "cost": eval_cost},
                           "actual_next_updates": record["completed_next_updates"], "qa_checkpoint": _relative(path, root),
                           "reload_next_update_equal": True, "cumulative_update": 16002,
                           "added_training_cost": payload["added_training_cost"],
                           "cumulative_training_cost": payload["cumulative_training_cost"],
                           "uninterrupted_next_update_cost": record["uninterrupted_next_update_cost"],
                           "reloaded_next_update_cost": record["reloaded_next_update_cost"]})
        report["status"] = "complete"
        report["actual_cost"] = {
            "updates": sum(record["added_updates"] + record["actual_next_updates"] for record in report["models"].values()),
            "examples": sum(record["added_training_cost"]["program_state_cases"] +
                             record["uninterrupted_next_update_cost"]["program_state_cases"] +
                             record["reloaded_next_update_cost"]["program_state_cases"] for record in report["models"].values()),
            "readouts": sum(record["added_training_cost"]["readout_positions"] +
                             record["uninterrupted_next_update_cost"]["readout_positions"] +
                             record["reloaded_next_update_cost"]["readout_positions"] for record in report["models"].values()),
            "internal_state_substeps": sum(record["added_training_cost"]["internal_state_substeps"] +
                                            record["uninterrupted_next_update_cost"]["internal_state_substeps"] +
                                            record["reloaded_next_update_cost"]["internal_state_substeps"] for record in report["models"].values()),
            "evaluation_forwards": sum(record["qa_evaluation_cost"]["program_forwards"] for record in report["models"].values()),
            "evaluation_cases": sum(record["qa_evaluation_cost"]["program_state_cases"] for record in report["models"].values()),
            "evaluation_readouts": sum(record["qa_evaluation_cost"]["readout_positions"] for record in report["models"].values()),
            "evaluation_internal_state_substeps": sum(record["qa_evaluation_cost"]["internal_state_substeps"] for record in report["models"].values()),
        }
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"
        report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def run(out: Path = RUN, preflight_dir: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    manifest = load_manifest(preflight_dir, root=root)
    try:
        return execute(_resolve(out, root), manifest, root=root)
    finally:
        verify_e33_guards(root=root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train-cleared", action="store_true")
    group.add_argument("--qa", action="store_true")
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT)
    args = parser.parse_args(argv)
    if args.preflight:
        value = preflight(args.preflight_dir)
        print(json.dumps({"status": "preflight_ready", "schema": value["schema"]}, sort_keys=True))
    elif args.qa:
        value = tiny_qa(_resolve(args.out, ROOT))
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    else:
        value = run(args.out, args.preflight_dir)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
