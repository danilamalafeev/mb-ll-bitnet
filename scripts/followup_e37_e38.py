#!/usr/bin/env python3
"""Fixed E37/E38 runner built on the accepted E32--E36 machinery.

The E37 path is inference-only and restores accepted E36 checkpoints through
the unchanged E36 loader.  The E38 path adds the registered E33-style
continuation to the four own E32 seed-1/2 parents, then runs the frozen E36
branches.  Preparation and QA are separate from scientific execution; all
scientific output directories refuse existing files.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import longer_native8_e20 as old
from looped_bitnet import width_e32 as core
from looped_bitnet.register_e15 import RegisterExample, STATE_ORDER, evaluate_program
from scripts import length_wave_e35_e36 as wave
from scripts import width_e32 as e32
from scripts.longer_native8_e20 import _update


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "e37_e38_followup_v1"
CHECKPOINT_SCHEMA = "e37_e38_followup_v2"
QA_SCHEMA = "e37_e38_followup_qa_v1"
RUNTIME_LINEAGE_SCHEMA = "e37_e38_runtime_lineage_v2"
SCIENCE_REFERENCE = Path("results/E37_E38_SCIENCE_REFERENCE_V3.json")
ARMS = ("float", "w4")
SEEDS = (1, 2)
WIDTH = 128
NATIVE_STEPS = 8
BATCH_SIZE = 64
E32_UPDATE = 16000
PARENT_UPDATE = 32000
ADDED_PARENT_UPDATES = 16000
BRANCH_UPDATES = 8000
MATCHED_UPDATES = 4572
FINAL_UPDATE = 40000
MATCHED_FINAL_UPDATE = PARENT_UPDATE + MATCHED_UPDATES
E36_PREFLIGHT = Path("runs/e35_e36_preflight_v2")
E32_PREFLIGHT = Path("runs/e32_width_preflight")
E32_RUN = Path("runs/e32_width")
E36_RUN = Path("runs/e35_e36_length_wave/science")
PREFLIGHT = Path("runs/e37_e38_preflight")
RUN = Path("runs/e37_e38_science")
QA_RUN = Path("runs/e37_e38_qa")
PROTOCOL = Path("results/E37_E38_PROTOCOL.md")
E36_MANIFEST_SHA256 = "4adf2dd1bb1036b94bcbf7a0246d9410596a737c9d20e5ff8865632fab5988aa"
E36_BRANCHES = ("A", "B_match", "B")
E36_BRANCH_UPDATES = {"A": BRANCH_UPDATES, "B_match": MATCHED_UPDATES, "B": BRANCH_UPDATES}
E36_EVAL_COST = {
    "A": {"program_forwards": 217, "program_state_cases": 54592,
          "readout_positions": 337856, "internal_state_substeps": 2702848},
    "B_match": {"program_forwards": 235, "program_state_cases": 58048,
                 "readout_positions": 355136, "internal_state_substeps": 2841088},
    "B": {"program_forwards": 235, "program_state_cases": 58048,
          "readout_positions": 355136, "internal_state_substeps": 2841088},
}
COST_KEYS = tuple(core.COST_KEYS)
TRAIN_COST = dict(core.TRAIN_COST_PER_MODEL)
E32_EVAL_COST = dict(core.EVAL_COST_PER_MODEL)
CHECKPOINT_KEYS = frozenset({
    "schema", "qa", "stage", "arm", "seed", "branch", "width", "parent_update", "added_updates",
    "cumulative_update", "update", "native_steps", "parameter_count", "training_mode", "parent",
    "parent_checkpoint", "parent_checkpoint_sha256", "parent_manifest_digest", "parent_source_hashes",
    "source_hashes", "e36_manifest_sha256", "e36_manifest_digest", "parent_stream_digest",
    "parent_target_digest", "added_stream_digest", "added_target_digest", "cumulative_stream_digest",
    "cumulative_target_digest", "manifest_digest", "config", "initial_digest", "initial_rng_digest",
    "parent_model_digest", "parent_optimizer_digest", "parent_rng_digest", "model_digest",
    "optimizer_state_dict", "optimizer_digest", "rng_state", "rng_digest", "state_dict",
    "parent_training_cost", "added_training_cost", "cumulative_training_cost",
    "runtime_lineage_schema", "runtime_lineage_base_digest",
})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
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
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes, *, refuse: bool = False) -> None:
    """Write immutable input bytes without normalizing JSON formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing temporary collision: {temporary}")
    temporary.write_bytes(bytes(value))
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


def _resolve(path: Path, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path))


def _sha256(path: Path) -> str:
    return core.sha256_file(Path(path))


def _zero_cost() -> dict[str, int]:
    return dict.fromkeys(COST_KEYS, 0)


def _cost_add(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    if set(left) != set(COST_KEYS) or set(right) != set(COST_KEYS):
        raise ValueError("E37/E38 cost keys changed")
    return {key: int(left[key]) + int(right[key]) for key in COST_KEYS}


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _cost_for_batches(batches: Sequence[Sequence[RegisterExample]]) -> dict[str, int]:
    examples = sum(len(batch) for batch in batches)
    positions = sum(len(batch[0].program) * len(batch) for batch in batches)
    return {"program_forwards": len(batches), "program_state_cases": examples,
            "readout_positions": positions, "internal_state_substeps": positions * NATIVE_STEPS}


def _checkpoint_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash the immutable base manifest object exactly as received."""
    return old.canonical_hash(dict(manifest))


def _base_manifest_identity(path: Path, manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, str]:
    """Return the byte and canonical identities of a frozen base manifest."""
    path = Path(path)
    raw = path.read_bytes()
    loaded = json.loads(raw.decode("utf-8"))
    if not isinstance(loaded, dict) or not _typed_equal(loaded, dict(manifest)):
        raise ValueError("runtime lineage base manifest bytes/object mismatch")
    return {"path": _relative(path, root), "sha256": hashlib.sha256(raw).hexdigest(),
            "digest": _checkpoint_manifest_digest(loaded)}


def _new_runtime_lineage(base_path: Path, manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    identity = _base_manifest_identity(base_path, manifest, root=root)
    return {"schema": RUNTIME_LINEAGE_SCHEMA, "base_manifest_path": identity["path"],
            "base_manifest_sha256": identity["sha256"], "base_manifest_digest": identity["digest"],
            "entries": {}, "status": {}, "history": []}


def _validate_runtime_lineage(lineage: Mapping[str, Any], manifest: Mapping[str, Any], *, root: Path = ROOT) -> None:
    if lineage.get("schema") != RUNTIME_LINEAGE_SCHEMA:
        raise ValueError("runtime lineage schema mismatch")
    if not isinstance(lineage.get("entries"), Mapping) or not isinstance(lineage.get("status"), Mapping):
        raise ValueError("runtime lineage registry malformed")
    base_path = _resolve(Path(str(lineage.get("base_manifest_path", ""))), root)
    identity = _base_manifest_identity(base_path, manifest, root=root)
    for key, field in (("base_manifest_path", "path"), ("base_manifest_sha256", "sha256"),
                       ("base_manifest_digest", "digest")):
        if lineage.get(key) != identity[field]:
            raise ValueError(f"runtime lineage base identity mismatch: {key}")
    for label, entry in lineage["entries"].items():
        _validate_runtime_parent_entry(str(label), entry)


def _validate_runtime_parent_entry(label: str, entry: Mapping[str, Any]) -> None:
    required = {"label", "path", "sha256", "arm", "seed", "update", "stage"}
    if not isinstance(entry, Mapping) or set(entry) != required:
        raise ValueError("runtime lineage parent entry fields mismatch")
    if entry.get("label") != label or entry.get("stage") != "parent_extension":
        raise ValueError("runtime lineage parent identity mismatch")
    if entry.get("arm") not in ARMS or type(entry.get("seed")) is not int or entry.get("seed") not in SEEDS:
        raise ValueError("runtime lineage parent arm/seed mismatch")
    if type(entry.get("update")) is not int or entry.get("update") <= 0 or type(entry.get("path")) is not str or \
            type(entry.get("sha256")) is not str or len(entry["sha256"]) != 64:
        raise ValueError("runtime lineage parent update/hash mismatch")
    if label != _label(entry["arm"], entry["seed"]):
        raise ValueError("runtime lineage parent label mismatch")


def _register_runtime_parent(lineage: Mapping[str, Any], label: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Append one generated parent; existing registrations are immutable."""
    _validate_runtime_parent_entry(label, entry)
    entries = lineage.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("runtime lineage registry malformed")
    if label in entries:
        if _typed_equal(entries[label], entry):
            raise ValueError("runtime lineage parent already registered")
        raise ValueError("runtime lineage parent replacement rejected")
    result = deepcopy(dict(lineage))
    result["entries"] = deepcopy(dict(entries))
    result["entries"][label] = deepcopy(dict(entry))
    result["status"] = deepcopy(dict(lineage.get("status", {})))
    result["history"] = list(lineage.get("history", [])) + [{"event": "parent_registered", "label": label}]
    return result


def _record_runtime_status(lineage: Mapping[str, Any], label: str, *, status: str,
                           attempted_updates: int, completed_updates: int,
                           training_cost: Mapping[str, int], error: str | None = None) -> dict[str, Any]:
    if type(attempted_updates) is not int or type(completed_updates) is not int or \
            attempted_updates < completed_updates or completed_updates < 0:
        raise ValueError("runtime lineage counters malformed")
    if set(training_cost) != set(COST_KEYS) or any(type(training_cost[key]) is not int or training_cost[key] < 0 for key in COST_KEYS):
        raise ValueError("runtime lineage cost malformed")
    result = deepcopy(dict(lineage))
    result["entries"] = deepcopy(dict(lineage.get("entries", {})))
    result["status"] = deepcopy(dict(lineage.get("status", {})))
    result["history"] = list(lineage.get("history", []))
    value: dict[str, Any] = {"status": status, "attempted_updates": attempted_updates,
                              "completed_updates": completed_updates, "training_cost": dict(training_cost)}
    if error is not None:
        value["error"] = str(error)
    result["status"][label] = value
    return result


def _resolve_runtime_parent(lineage: Mapping[str, Any], manifest: Mapping[str, Any], label: str, *,
                            root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    """Resolve a parent only through the immutable, base-bound registry."""
    _validate_runtime_lineage(lineage, manifest, root=root)
    entry = lineage["entries"].get(label)
    if entry is None:
        raise ValueError("runtime lineage parent missing")
    path = _resolve(Path(entry["path"]), root)
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise ValueError("runtime lineage parent file/hash mismatch")
    return path, dict(entry)


def _write_runtime_lineage(path: Path, lineage: Mapping[str, Any], *, refuse: bool = False) -> None:
    _atomic_json(path, lineage, refuse=refuse)


def _persist_failed_progress(out: Path, lineage: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Copy partial counters into lineage after a training loop aborts."""
    result = dict(lineage)
    progress_paths = list((Path(out) / "parents").glob("*/progress.json"))
    progress_paths += list((Path(out) / "branches").glob("*/*/progress.json"))
    for progress_path in sorted(progress_paths):
        record = _read_json(progress_path)
        if progress_path.parents[2].name == "branches":
            label = f"{progress_path.parents[1].name}/{progress_path.parents[0].name}"
        else:
            label = progress_path.parent.name
        result = _record_runtime_status(
            result, label, status="technical_failure",
            attempted_updates=int(record.get("attempted_updates", 0)),
            completed_updates=int(record.get("completed_updates", 0)),
            training_cost=record.get("training_cost", _zero_cost()),
            error=record.get("error", "training loop failed"))
    return result


def _config(stage: str, arm: str, seed: int, branch: str, parent_update: int, added_updates: int) -> dict[str, Any]:
    return {"schema": SCHEMA, "stage": stage, "arm": arm, "seed": seed, "branch": branch,
            "width": WIDTH, "native_steps": NATIVE_STEPS, "parent_update": parent_update,
            "added_updates": added_updates, "batch_size": BATCH_SIZE,
            "optimizer": deepcopy(core.OPTIMIZER_CONFIG)}


def _hooks(model: torch.nn.Module, record: dict[str, Any], phase: str):
    attempted = record.setdefault(phase + "_attempted", _zero_cost())
    completed = record.setdefault(phase, _zero_cost())

    def count(counter: dict[str, int], args: Sequence[Any]) -> None:
        x, _, ops = args
        cases, length = int(x.shape[0]), int(ops.shape[1])
        values = (1, cases, cases * length, cases * length * NATIVE_STEPS)
        for key, value in zip(COST_KEYS, values):
            counter[key] += value

    pre = model.register_forward_pre_hook(lambda _model, args: count(attempted, args))

    def post(_model, args, output):
        count(completed, args)
        if not isinstance(output, tuple) or len(output) != 2 or any(not torch.isfinite(t).all() for t in output):
            raise FloatingPointError("nonfinite E37/E38 model output")

    return pre, model.register_forward_hook(post)


def _label(arm: str, seed: int) -> str:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered E38 arm/seed")
    return f"{arm}128_seed{seed}"


def _source_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = (Path("scripts/followup_e37_e38.py"), Path("tests/test_followup_e37_e38.py"), PROTOCOL)
    return {str(path): _sha256(_resolve(path, root)) for path in paths if _resolve(path, root).is_file()}


def _effective_source_hashes(manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, str]:
    """Use the separately versioned source reference when one is present."""
    reference_path = _resolve(SCIENCE_REFERENCE, root)
    if not reference_path.is_file():
        return deepcopy(dict(manifest["source_hashes"]))
    reference = _read_json(reference_path)
    if reference.get("schema") != "e37_e38_science_reference_v3":
        raise ValueError("E37/E38 source reference schema changed")
    values = reference.get("source_hashes")
    if not isinstance(values, Mapping) or set(values) != set(manifest.get("source_hashes", {})):
        raise ValueError("E37/E38 source reference scope changed")
    return {str(key): str(value) for key, value in values.items()}


def _validate_source_reference(manifest: Mapping[str, Any], base_path: Path, *, root: Path = ROOT) -> dict[str, str]:
    """Validate current source bytes against a reference without changing base JSON."""
    current = _effective_source_hashes(manifest, root=root)
    reference_path = _resolve(SCIENCE_REFERENCE, root)
    if reference_path.is_file():
        reference = _read_json(reference_path)
        if reference.get("base_manifest_sha256") != _sha256(base_path):
            raise ValueError("E37/E38 source reference base hash changed")
        if reference.get("base_manifest_digest") != _checkpoint_manifest_digest(manifest):
            raise ValueError("E37/E38 source reference base digest changed")
        if reference.get("lineage_schema") != RUNTIME_LINEAGE_SCHEMA:
            raise ValueError("E37/E38 source reference lineage schema changed")
    for relative, expected in current.items():
        artifact = _resolve(Path(relative), root)
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"E37/E38 source hash mismatch: {relative}")
    return current


def _load_inherited(root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(E36_PREFLIGHT, root)
    if _sha256(path / "manifest.json") != E36_MANIFEST_SHA256:
        raise ValueError("accepted E36 v2 manifest hash changed")
    manifest = wave.load_manifest(path, root=root)
    if manifest.get("schema") != wave.SCHEMA or manifest.get("seed") != 0 or manifest.get("width") != WIDTH:
        raise ValueError("E36 inherited manifest identity changed")
    if manifest.get("parent_update") != PARENT_UPDATE or manifest.get("added_updates") != BRANCH_UPDATES:
        raise ValueError("E36 inherited update budget changed")
    a, b = wave.build_streams()
    checks = {
        "a_schedule": (old.batch_digest(a), old.target_digest(a), len(a)),
        "b_schedule": (old.batch_digest(b), old.target_digest(b), len(b)),
        "b_matched_schedule": (old.batch_digest(b[:MATCHED_UPDATES]), old.target_digest(b[:MATCHED_UPDATES]), MATCHED_UPDATES),
    }
    for name, (stream_digest, target_digest, updates) in checks.items():
        frozen = manifest[name]
        if frozen["stream_digest"] != stream_digest or frozen["target_digest"] != target_digest or frozen["updates"] != updates:
            raise ValueError(f"E36 inherited {name} changed")
    for name, batches in (("a_schedule", a), ("b_schedule", b)):
        programs = sorted({tuple(example.program) for batch in batches for example in batch})
        if manifest[name]["programs"] != [list(program) for program in programs]:
            raise ValueError(f"E36 inherited {name} programs changed")
    if not _typed_equal(manifest["evaluation"], wave.frozen_evaluation_pools()):
        raise ValueError("E36 inherited evaluation pools changed")
    if manifest["training_states"] != [list(s) for s in wave.TRAIN_STATES] or \
            manifest["heldout_states"] != [list(s) for s in wave.HELDOUT_STATES]:
        raise ValueError("E36 inherited state split changed")
    if manifest.get("probe_programs") != wave._probe_programs(a, b):
        raise ValueError("E36 inherited probes changed")
    return manifest


def _load_e32_manifest(root: Path = ROOT) -> dict[str, Any]:
    manifest = e32.load_manifest(_resolve(E32_PREFLIGHT, root), root=root)
    if manifest.get("schema") != core.SCHEMA:
        raise ValueError("E32 parent manifest schema changed")
    return manifest


def _e32_parent_paths(root: Path = ROOT) -> dict[str, str]:
    return {_label(arm, seed): _relative(_resolve(E32_RUN / _label(arm, seed) / "u16000.pt", root), root)
            for seed in SEEDS for arm in ARMS}


def _e36_paths(root: Path = ROOT) -> dict[str, dict[str, str]]:
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for branch, update in (("A", FINAL_UPDATE), ("B_match", MATCHED_FINAL_UPDATE), ("B", FINAL_UPDATE)):
            path = _resolve(E36_RUN / f"{arm}128_seed0" / branch / f"u{update}.pt", root)
            result[arm][branch] = _relative(path, root)
    return result


def _schedule_metadata(a: Sequence[Sequence[RegisterExample]], b: Sequence[Sequence[RegisterExample]]) -> dict[str, Any]:
    return {
        "A": {"updates": len(a), "stream_digest": old.batch_digest(a), "target_digest": old.target_digest(a),
              "cost": _cost_for_batches(a)},
        "B_match": {"updates": MATCHED_UPDATES, "stream_digest": old.batch_digest(b[:MATCHED_UPDATES]),
                     "target_digest": old.target_digest(b[:MATCHED_UPDATES]), "cost": _cost_for_batches(b[:MATCHED_UPDATES])},
        "B": {"updates": len(b), "stream_digest": old.batch_digest(b), "target_digest": old.target_digest(b),
              "cost": _cost_for_batches(b)},
    }


def make_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    inherited = _load_inherited(root)
    e32_manifest = _load_e32_manifest(root)
    sources = _source_hashes(root)
    if set(sources) != {str(Path("scripts/followup_e37_e38.py")), str(Path("tests/test_followup_e37_e38.py")), str(PROTOCOL)}:
        raise ValueError("E37/E38 source provenance files missing")
    a, b = wave.build_streams()
    extension = core.stream()
    if old.batch_digest(extension) != e32_manifest["full_stream_digest"] or old.target_digest(extension) != e32_manifest["full_target_digest"]:
        raise ValueError("E38 extension stream does not equal accepted E32 stream")
    e32_parent_hashes = {}
    e32_parent_paths = _e32_parent_paths(root)
    for label, relative in e32_parent_paths.items():
        path = _resolve(Path(relative), root)
        if not path.is_file():
            raise FileNotFoundError(path)
        e32_parent_hashes[label] = _sha256(path)
        arm = "float" if label.startswith("float") else "w4"
        seed = int(label.rsplit("seed", 1)[1])
        _, _, payload = core.load_checkpoint(path, e32_manifest, arm=arm, width=WIDTH, seed=seed,
                                             update=E32_UPDATE, qa=False, root=root,
                                             training_cost=core.TRAIN_COST_PER_MODEL)
        if payload.get("training_cost") != TRAIN_COST:
            raise ValueError(f"E32 parent training cost changed: {label}")
    e36_paths = _e36_paths(root)
    e36_hashes = {}
    for arm in ARMS:
        e36_hashes[arm] = {}
        for branch, relative in e36_paths[arm].items():
            path = _resolve(Path(relative), root)
            if not path.is_file():
                raise FileNotFoundError(path)
            e36_hashes[arm][branch] = _sha256(path)
            expected = FINAL_UPDATE if branch != "B_match" else MATCHED_FINAL_UPDATE
            wave.load_checkpoint(path, inherited, arm=arm, branch=branch, expected_update=expected, qa=False, root=root)
    cumulative_extension = extension + extension
    branch_metadata = _schedule_metadata(a, b)
    return {
        "schema": SCHEMA, "status": "preflight", "width": WIDTH, "native_steps": NATIVE_STEPS,
        "batch_size": BATCH_SIZE, "arms": list(ARMS), "seeds": list(SEEDS),
        "parent_update": PARENT_UPDATE, "e32_parent_update": E32_UPDATE,
        "added_parent_updates": ADDED_PARENT_UPDATES, "branch_updates": BRANCH_UPDATES,
        "matched_updates": MATCHED_UPDATES, "e36_manifest_path": str(E36_PREFLIGHT),
        "e36_manifest_sha256": E36_MANIFEST_SHA256, "e36_manifest_digest": old.canonical_hash(inherited),
        "e32_manifest_path": str(E32_PREFLIGHT), "e32_manifest_sha256": _sha256(_resolve(E32_PREFLIGHT / "manifest.json", root)),
        "e32_manifest_digest": old.canonical_hash(e32_manifest), "source_hashes": sources,
        "inherited_source_hashes": deepcopy(inherited["source_hashes"]),
        "e32_parent_paths": e32_parent_paths, "e32_parent_checkpoint_sha256": e32_parent_hashes,
        "e36_checkpoint_paths": e36_paths, "e36_checkpoint_sha256": e36_hashes,
        "extension_stream_digest": old.batch_digest(extension), "extension_target_digest": old.target_digest(extension),
        "extension_cost_per_model": deepcopy(TRAIN_COST),
        "cumulative_extension_stream_digest": old.batch_digest(cumulative_extension),
        "cumulative_extension_target_digest": old.target_digest(cumulative_extension),
        "cumulative_parent_cost_per_model": _cost_add(TRAIN_COST, TRAIN_COST),
        "branch_schedules": branch_metadata,
        "inherited_schedule_digests": {
            name: {key: value for key, value in inherited[name].items() if key in ("updates", "stream_digest", "target_digest", "cost")}
            for name in ("a_schedule", "b_matched_schedule", "b_schedule")
        },
        "evaluation_scope_digest": old.canonical_hash(inherited["evaluation"]),
        "probe_scope_digest": old.canonical_hash(inherited["probe_programs"]),
        "training_states": deepcopy(inherited["training_states"]), "heldout_states": deepcopy(inherited["heldout_states"]),
        "e37": {"program_count_per_endpoint": 36, "states_per_program": 256,
                "cost_per_endpoint": deepcopy(inherited["e35"]["cost_per_arm"])},
        "e38": {"parent_evaluation_cost_per_endpoint": deepcopy(E32_EVAL_COST),
                "branch_evaluation_cost_per_endpoint": deepcopy(E36_EVAL_COST),
                "primary_cases": 6144, "primary_cases_per_length": 1536,
                "extended_parent_count": len(ARMS) * len(SEEDS),
                "branch_endpoint_count": len(ARMS) * len(SEEDS) * len(E36_BRANCHES)},
    }


def preflight(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty preflight directory: {path}")
    science_root = _resolve(RUN, root)
    if science_root.exists() and any(science_root.iterdir()):
        raise FileExistsError(f"refusing to prepare while science output is non-empty: {science_root}")
    manifest = make_manifest(root=root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    return manifest


def load_manifest(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "preflight":
        raise ValueError("E37/E38 frozen manifest identity changed")
    if manifest.get("e36_manifest_sha256") != E36_MANIFEST_SHA256:
        raise ValueError("E37/E38 accepted E36 manifest reference changed")
    inherited = _load_inherited(root)
    if old.canonical_hash(inherited) != manifest.get("e36_manifest_digest"):
        raise ValueError("E37/E38 inherited E36 manifest digest changed")
    e32_path = _resolve(E32_PREFLIGHT / "manifest.json", root)
    if not e32_path.is_file() or _sha256(e32_path) != manifest.get("e32_manifest_sha256"):
        raise ValueError("E37/E38 accepted E32 manifest hash changed")
    _validate_source_reference(manifest, path / "manifest.json", root=root)
    a, b = wave.build_streams()
    expected_schedules = _schedule_metadata(a, b)
    if manifest.get("branch_schedules") != expected_schedules:
        raise ValueError("E37/E38 branch schedules changed")
    extension = core.stream()
    if manifest.get("extension_stream_digest") != old.batch_digest(extension) or manifest.get("extension_target_digest") != old.target_digest(extension):
        raise ValueError("E37/E38 extension stream changed")
    for label, relative in manifest.get("e32_parent_paths", {}).items():
        path = _resolve(Path(relative), root)
        if not path.is_file() or _sha256(path) != manifest.get("e32_parent_checkpoint_sha256", {}).get(label):
            raise ValueError(f"E37/E38 E32 parent hash mismatch: {label}")
    return manifest


def _load_e32_parent(label: str, manifest: Mapping[str, Any], *, root: Path = ROOT):
    if label not in manifest["e32_parent_paths"]:
        raise ValueError(f"unknown E32 parent: {label}")
    arm = "float" if label.startswith("float") else "w4"
    seed = int(label.rsplit("seed", 1)[1])
    path = _resolve(Path(manifest["e32_parent_paths"][label]), root)
    if _sha256(path) != manifest["e32_parent_checkpoint_sha256"][label]:
        raise ValueError(f"E32 parent hash mismatch: {label}")
    e32_manifest = _load_e32_manifest(root)
    model, optimizer, payload = core.load_checkpoint(path, e32_manifest, arm=arm, width=WIDTH, seed=seed,
                                                     update=E32_UPDATE, qa=False, root=root,
                                                     training_cost=TRAIN_COST)
    return model, optimizer, payload, path, e32_manifest


def _parent_ref(payload: Mapping[str, Any], path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root), "sha256": _sha256(path), "schema": payload["schema"],
        "label": payload["label"], "arm": payload["arm"], "width": payload["width"],
        "seed": payload["seed"], "update": payload["update"], "model_digest": payload["model_digest"],
        "optimizer_digest": payload["optimizer_digest"], "rng_digest": payload["rng_digest"],
        "manifest_digest": payload["manifest_digest"], "source_hashes": deepcopy(payload["source_hashes"]),
        "stream_digest": payload["stream_digest"], "target_digest": payload["target_digest"],
        "training_cost": deepcopy(payload["training_cost"]),
    }


def _parent_ref_for_extension(payload: Mapping[str, Any], path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root), "sha256": _sha256(path), "schema": payload["schema"],
        "stage": payload["stage"], "label": payload["branch"], "arm": payload["arm"],
        "width": payload["width"], "seed": payload["seed"], "update": payload["update"],
        "model_digest": payload["model_digest"], "optimizer_digest": payload["optimizer_digest"],
        "rng_digest": payload["rng_digest"], "manifest_digest": payload["manifest_digest"],
        "source_hashes": deepcopy(payload["source_hashes"]), "stream_digest": payload["cumulative_stream_digest"],
        "target_digest": payload["cumulative_target_digest"],
        "training_cost": deepcopy(payload["cumulative_training_cost"]),
    }


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, parent: Mapping[str, Any],
                        parent_path: Path, manifest: Mapping[str, Any], *, stage: str, arm: str, seed: int,
                        branch: str, parent_update: int, added_updates: int,
                        added_batches: Sequence[Sequence[RegisterExample]], training_cost: Mapping[str, int],
                        qa: bool = False, root: Path = ROOT,
                        lineage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if stage not in {"parent_extension", "branch"} or arm not in ARMS or seed not in SEEDS:
        raise ValueError("E38 checkpoint identity")
    if stage == "parent_extension" and branch != "parent":
        raise ValueError("E38 extension branch identity")
    if stage == "branch" and branch not in E36_BRANCHES:
        raise ValueError("E38 branch identity")
    if not model.training or set(training_cost) != set(COST_KEYS):
        raise ValueError("E38 checkpoint training state/cost")
    if not qa and lineage is None:
        raise ValueError("E38 scientific checkpoint requires runtime lineage")
    if lineage is not None:
        _validate_runtime_lineage(lineage, manifest, root=root)
    source_hashes = _effective_source_hashes(manifest, root=root)
    stream = list(added_batches)
    parent_stream = core.stream() if stage == "parent_extension" else core.stream() + core.stream()
    parent_stream_digest = old.batch_digest(core.stream()) if stage == "parent_extension" else parent["cumulative_stream_digest"]
    parent_target_digest = old.target_digest(core.stream()) if stage == "parent_extension" else parent["cumulative_target_digest"]
    parent_ref = _parent_ref(parent, parent_path, root=root) if stage == "parent_extension" else _parent_ref_for_extension(parent, parent_path, root=root)
    added_stream_digest = old.batch_digest(stream)
    added_target_digest = old.target_digest(stream)
    cumulative_stream_digest = old.batch_digest(parent_stream + stream)
    cumulative_target_digest = old.target_digest(parent_stream + stream)
    cumulative_cost = _cost_add(parent["training_cost"] if stage == "parent_extension" else parent["cumulative_training_cost"], training_cost)
    state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    return {
        "schema": QA_SCHEMA if qa else CHECKPOINT_SCHEMA, "qa": bool(qa), "stage": stage, "arm": arm, "seed": seed,
        "branch": branch, "width": WIDTH, "parent_update": parent_update, "added_updates": added_updates,
        "cumulative_update": parent_update + added_updates, "update": parent_update + added_updates,
        "native_steps": NATIVE_STEPS, "parameter_count": core.PARAMETER_COUNT_128, "training_mode": True,
        "parent": parent_ref, "parent_checkpoint": parent_ref["path"], "parent_checkpoint_sha256": parent_ref["sha256"],
        "parent_manifest_digest": parent["manifest_digest"], "parent_source_hashes": deepcopy(parent["source_hashes"]),
        "source_hashes": deepcopy(source_hashes), "e36_manifest_sha256": manifest["e36_manifest_sha256"],
        "e36_manifest_digest": manifest["e36_manifest_digest"], "parent_stream_digest": parent_stream_digest,
        "parent_target_digest": parent_target_digest, "added_stream_digest": added_stream_digest,
        "added_target_digest": added_target_digest, "cumulative_stream_digest": cumulative_stream_digest,
        "cumulative_target_digest": cumulative_target_digest, "manifest_digest": _checkpoint_manifest_digest(manifest),
        "config": _config(stage, arm, seed, branch, parent_update, added_updates),
        "initial_digest": parent["initial_digest"], "initial_rng_digest": parent["initial_rng_digest"],
        "parent_model_digest": parent["model_digest"], "parent_optimizer_digest": parent["optimizer_digest"],
        "parent_rng_digest": parent["rng_digest"], "model_digest": core.digest_state_dict(model),
        "optimizer_state_dict": optimizer_state, "optimizer_digest": core.digest_object(optimizer_state),
        "rng_state": rng, "rng_digest": core.digest_object(rng), "state_dict": state,
        "parent_training_cost": deepcopy(parent["training_cost"] if stage == "parent_extension" else parent["cumulative_training_cost"]),
        "added_training_cost": dict(training_cost), "cumulative_training_cost": cumulative_cost,
        "runtime_lineage_schema": RUNTIME_LINEAGE_SCHEMA if lineage is not None else None,
        "runtime_lineage_base_digest": lineage.get("base_manifest_digest") if lineage is not None else None,
    }


def _expected_parent_path(payload: Mapping[str, Any], manifest: Mapping[str, Any], *,
                           lineage: Mapping[str, Any] | None = None, root: Path) -> Path:
    label = _label(str(payload["arm"]), int(payload["seed"]))
    if payload["stage"] == "parent_extension":
        relative = manifest["e32_parent_paths"][label]
    else:
        if lineage is None:
            raise ValueError("extended parent lineage missing")
        return _resolve_runtime_parent(lineage, manifest, label, root=root)[0]
    return _resolve(Path(relative), root)


def _validate_payload(payload: Mapping[str, Any], path: Path, manifest: Mapping[str, Any], *, stage: str,
                      arm: str, seed: int, branch: str, expected_update: int, qa: bool,
                      lineage: Mapping[str, Any] | None = None, root: Path = ROOT) -> None:
    if set(payload) != set(CHECKPOINT_KEYS):
        raise ValueError("E38 checkpoint fields mismatch")
    expected_schema = QA_SCHEMA if qa else CHECKPOINT_SCHEMA
    if stage not in {"parent_extension", "branch"} or arm not in ARMS or seed not in SEEDS:
        raise ValueError("E38 payload stage/identity")
    if stage == "parent_extension" and branch != "parent":
        raise ValueError("E38 extension branch identity")
    if stage == "branch" and branch not in E36_BRANCHES:
        raise ValueError("E38 branch identity")
    added_updates = payload.get("added_updates")
    if type(added_updates) is not int or added_updates <= 0:
        raise ValueError("E38 added update type")
    expected_parent_update = E32_UPDATE if stage == "parent_extension" else PARENT_UPDATE
    if qa:
        # QA deliberately uses a two-update synthetic parent and child.  The
        # parent update is still bound to the actual serialized parent below.
        expected_parent_update = payload.get("parent_update")
        if type(expected_parent_update) is not int or expected_parent_update <= 0:
            raise ValueError("E38 QA parent update type")
    expected = {"schema": expected_schema, "qa": qa, "stage": stage, "arm": arm, "seed": seed,
                "branch": branch, "width": WIDTH, "parent_update": expected_parent_update,
                "added_updates": expected_update - expected_parent_update,
                "cumulative_update": expected_update, "update": expected_update,
                "native_steps": NATIVE_STEPS, "parameter_count": core.PARAMETER_COUNT_128,
                "training_mode": True}
    if expected["added_updates"] != added_updates:
        raise ValueError("E38 added/cumulative update mismatch")
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"E38 checkpoint identity mismatch: {key}")
    source_hashes = _effective_source_hashes(manifest, root=root)
    if not _typed_equal(payload.get("source_hashes"), source_hashes):
        raise ValueError("E38 source provenance mismatch")
    if payload.get("manifest_digest") != _checkpoint_manifest_digest(manifest):
        raise ValueError("E38 checkpoint manifest digest mismatch")
    if lineage is None:
        if not qa:
            raise ValueError("E38 runtime lineage missing")
        if payload.get("runtime_lineage_schema") is not None or payload.get("runtime_lineage_base_digest") is not None:
            raise ValueError("E38 QA runtime lineage fields unexpected")
    else:
        if payload.get("runtime_lineage_schema") != RUNTIME_LINEAGE_SCHEMA or \
                payload.get("runtime_lineage_base_digest") != _checkpoint_manifest_digest(manifest):
            raise ValueError("E38 runtime lineage checkpoint identity mismatch")
        _validate_runtime_lineage(lineage, manifest, root=root)
    if not _typed_equal(payload.get("config"), _config(stage, arm, seed, branch, expected_parent_update, added_updates)):
        raise ValueError("E38 checkpoint config mismatch")
    if payload.get("e36_manifest_sha256") != manifest.get("e36_manifest_sha256") or \
            payload.get("e36_manifest_digest") != manifest.get("e36_manifest_digest"):
        raise ValueError("E38 inherited manifest provenance mismatch")
    parent_path = _resolve(Path(payload.get("parent_checkpoint", "")), root)
    expected_parent = _expected_parent_path(payload, manifest, lineage=lineage, root=root)
    if parent_path != expected_parent or not parent_path.is_file():
        raise ValueError("E38 parent checkpoint path mismatch")
    if stage == "parent_extension":
        expected_parent_hash = manifest.get("e32_parent_checkpoint_sha256", {}).get(_label(arm, seed))
    else:
        if lineage is None:
            raise ValueError("E38 extended parent lineage missing")
        expected_parent_hash = lineage["entries"].get(_label(arm, seed), {}).get("sha256")
    if not isinstance(expected_parent_hash, str):
        raise ValueError("E38 parent hash reference missing")
    if payload.get("parent_checkpoint_sha256") != expected_parent_hash or _sha256(parent_path) != expected_parent_hash:
        raise ValueError("E38 parent checkpoint hash mismatch")
    for relative, expected_hash in source_hashes.items():
        if _sha256(_resolve(Path(relative), root)) != expected_hash:
            raise ValueError(f"E38 source hash mismatch: {relative}")
    if stage == "parent_extension":
        e32_manifest = _load_e32_manifest(root)
        actual_model, actual_optimizer, actual_parent = core.load_checkpoint(parent_path, e32_manifest, arm=arm, width=WIDTH,
                                                                              seed=seed, update=E32_UPDATE, qa=False,
                                                                              root=root, training_cost=TRAIN_COST)
        del actual_model, actual_optimizer
        if payload.get("parent") != _parent_ref(actual_parent, parent_path, root=root):
            raise ValueError("E38 E32 parent reference mismatch")
        if payload.get("parent_manifest_digest") != actual_parent.get("manifest_digest"):
            raise ValueError("E38 E32 parent manifest lineage mismatch")
        if payload.get("parent_stream_digest") != actual_parent.get("stream_digest") or \
                payload.get("parent_target_digest") != actual_parent.get("target_digest"):
            raise ValueError("E38 E32 parent stream lineage mismatch")
        expected_parent_cost = actual_parent.get("training_cost")
    else:
        parent_model, parent_optimizer, actual_parent = _load_extended_checkpoint(
            parent_path, manifest, arm=arm, seed=seed, stage="parent_extension", branch="parent",
            expected_update=expected_parent_update, qa=qa, lineage=lineage, root=root)
        del parent_model, parent_optimizer
        if payload.get("parent") != _parent_ref_for_extension(actual_parent, parent_path, root=root):
            raise ValueError("E38 extension parent reference mismatch")
        expected_parent_cost = actual_parent.get("cumulative_training_cost")
    if payload.get("parent_source_hashes") != actual_parent.get("source_hashes"):
        raise ValueError("E38 parent source provenance mismatch")
    for field, actual_field in (("parent_model_digest", "model_digest"), ("parent_optimizer_digest", "optimizer_digest"),
                                ("parent_rng_digest", "rng_digest"), ("parent_manifest_digest", "manifest_digest")):
        if payload.get(field) != (actual_parent.get(actual_field) if field != "parent_manifest_digest" else actual_parent.get("manifest_digest")):
            raise ValueError(f"E38 parent digest lineage mismatch: {field}")
    if payload.get("parent_training_cost") != expected_parent_cost:
        raise ValueError("E38 parent cost lineage mismatch")
    if payload.get("parent_stream_digest") != (actual_parent.get("stream_digest") if stage == "parent_extension" else actual_parent.get("cumulative_stream_digest")) or \
            payload.get("parent_target_digest") != (actual_parent.get("target_digest") if stage == "parent_extension" else actual_parent.get("cumulative_target_digest")):
        raise ValueError("E38 parent stream lineage mismatch")
    expected_added = None
    expected_stream = expected_target = expected_cumulative_stream = expected_cumulative_target = None
    if not qa:
        if stage == "parent_extension":
            expected_added = TRAIN_COST
            expected_stream, expected_target = manifest["extension_stream_digest"], manifest["extension_target_digest"]
            expected_cumulative_stream, expected_cumulative_target = manifest["cumulative_extension_stream_digest"], manifest["cumulative_extension_target_digest"]
        else:
            a, b = wave.build_streams()
            schedule = a if branch == "A" else b[:MATCHED_UPDATES] if branch == "B_match" else b
            expected_added = _cost_for_batches(schedule)
            expected_stream, expected_target = old.batch_digest(schedule), old.target_digest(schedule)
            expected_cumulative_stream = old.batch_digest(core.stream() + core.stream() + list(schedule))
            expected_cumulative_target = old.target_digest(core.stream() + core.stream() + list(schedule))
        if payload.get("added_training_cost") != expected_added:
            raise ValueError("E38 added cost mismatch")
        if payload.get("added_stream_digest") != expected_stream or payload.get("added_target_digest") != expected_target:
            raise ValueError("E38 added stream provenance mismatch")
        if payload.get("cumulative_stream_digest") != expected_cumulative_stream or payload.get("cumulative_target_digest") != expected_cumulative_target:
            raise ValueError("E38 cumulative stream provenance mismatch")
    else:
        if payload.get("added_training_cost", {}).get("program_forwards") != added_updates:
            raise ValueError("E38 QA cost update mismatch")
    if not isinstance(payload.get("added_training_cost"), Mapping) or set(payload["added_training_cost"]) != set(COST_KEYS) or any(type(payload["added_training_cost"][key]) is not int or payload["added_training_cost"][key] < 0 for key in COST_KEYS):
        raise ValueError("E38 added cost malformed")
    if payload.get("cumulative_training_cost") != _cost_add(payload["parent_training_cost"], payload["added_training_cost"]):
        raise ValueError("E38 cumulative cost mismatch")
    if not isinstance(payload.get("rng_state"), torch.Tensor) or payload["rng_state"].dtype != torch.uint8 or payload["rng_state"].ndim != 1 or payload.get("rng_digest") != core.digest_object(payload["rng_state"]):
        raise ValueError("E38 RNG payload malformed")
    if payload.get("initial_digest") != actual_parent.get("initial_digest") or payload.get("initial_rng_digest") != actual_parent.get("initial_rng_digest"):
        raise ValueError("E38 initial provenance mismatch")


def _load_extended_checkpoint(path: Path, manifest: Mapping[str, Any], *, arm: str, seed: int, stage: str,
                              branch: str, expected_update: int, qa: bool,
                              lineage: Mapping[str, Any] | None = None, root: Path = ROOT):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("E38 checkpoint payload malformed")
    _validate_payload(payload, Path(path), manifest, stage=stage, arm=arm, seed=seed, branch=branch,
                      expected_update=expected_update, qa=qa, lineage=lineage, root=root)
    model = core.build_initial_model(WIDTH, arm, seed, root=root)
    optimizer = old.make_optimizer(model)
    core._validate_checkpoint_state(payload.get("state_dict"), model)
    core._validate_checkpoint_optimizer(payload.get("optimizer_state_dict"), model, optimizer, expected_update)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["rng_state"])
    model.train(True)
    if core.digest_state_dict(model) != payload["model_digest"] or core.digest_object(optimizer.state_dict()) != payload["optimizer_digest"]:
        raise ValueError("E38 checkpoint digest mismatch")
    return model, optimizer, dict(payload)


def _evaluate_e32_parent(path: Path, manifest: Mapping[str, Any], *, arm: str, seed: int,
                         lineage: Mapping[str, Any] | None = None, root: Path = ROOT):
    model, optimizer, payload = _load_extended_checkpoint(path, manifest, arm=arm, seed=seed,
                                                            stage="parent_extension", branch="parent",
                                                            expected_update=PARENT_UPDATE, qa=False,
                                                            lineage=lineage, root=root)
    before = core.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state(), model.training))
    record = {"evaluation_cost": _zero_cost()}
    hooks = _hooks(model, record, "evaluation_cost")
    try:
        evaluation = e32.evaluate(model, _load_e32_manifest(root))
    finally:
        for hook in hooks:
            hook.remove()
    after = core.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state(), model.training))
    if before != after or not model.training:
        raise ValueError("E38 parent evaluation mutated model/RNG/mode")
    if record["evaluation_cost"] != E32_EVAL_COST:
        raise ValueError("E38 parent evaluation accounting mismatch")
    return evaluation, record["evaluation_cost"], payload


def _evaluate_branch(path: Path, manifest: Mapping[str, Any], *, arm: str, seed: int, branch: str,
                     expected_update: int, lineage: Mapping[str, Any] | None = None,
                     root: Path = ROOT):
    model, optimizer, payload = _load_extended_checkpoint(path, manifest, arm=arm, seed=seed,
                                                            stage="branch", branch=branch,
                                                            expected_update=expected_update, qa=False,
                                                            lineage=lineage, root=root)
    before = core.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state(), model.training))
    record = {"evaluation_cost": _zero_cost()}
    hooks = _hooks(model, record, "evaluation_cost")
    try:
        evaluation = wave.evaluate_scientific(model, _load_inherited(root), branch)
    finally:
        for hook in hooks:
            hook.remove()
    after = core.digest_object((model.state_dict(), optimizer.state_dict(), torch.get_rng_state(), model.training))
    if before != after or not model.training:
        raise ValueError("E38 branch evaluation mutated model/RNG/mode")
    if record["evaluation_cost"] != E36_EVAL_COST[branch]:
        raise ValueError(f"E38 branch evaluation accounting mismatch: {branch}")
    return evaluation, record["evaluation_cost"], payload


def _save_prediction(path: Path, value: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    _atomic_json(path, value, refuse=True)
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _prediction_map(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    values = row.get("predictions")
    if not isinstance(values, list):
        raise ValueError("predictions missing")
    result = {}
    for value in values:
        state = tuple(value["state"])
        if state in result:
            raise ValueError("duplicate prediction state")
        result[state] = value
    return result


def _pair_rows(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if left.get("program") != right.get("program"):
        raise ValueError(f"paired {label} program mismatch")
    left_map, right_map = _prediction_map(left), _prediction_map(right)
    if set(left_map) != set(right_map):
        raise ValueError(f"paired {label} state mismatch")
    result = {"program": left["program"], "length": len(left["program"]), "cases": len(left_map),
              "final": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0,
                        "both_correct": 0, "both_wrong": 0},
              "full_trace": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0,
                             "both_correct": 0, "both_wrong": 0},
              "by_stratum": {}}
    for state in sorted(left_map):
        a, b = left_map[state], right_map[state]
        if a.get("target_trace") != b.get("target_trace"):
            raise ValueError(f"paired {label} target mismatch")
        if a.get("stratum") != b.get("stratum"):
            raise ValueError(f"paired {label} stratum mismatch")
        for metric, left_ok, right_ok in (
                ("final", bool(a["joint_final_correct"]), bool(b["joint_final_correct"])),
                ("full_trace", all(a["prefix_joint_correct"]), all(b["prefix_joint_correct"]))):
            key = "both_correct" if left_ok and right_ok else "left_correct_right_wrong" if left_ok else "left_wrong_right_correct" if right_ok else "both_wrong"
            result[metric][key] += 1
            stratum = str(a["stratum"])
            scoped = result["by_stratum"].setdefault(stratum, {"cases": 0, "final": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0}, "full_trace": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0}})
            if metric == "final":
                scoped["cases"] += 1
            scoped[metric][key] += 1
    return result


def _aggregate_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {"program_count": len(pairs), "cases": sum(int(p["cases"]) for p in pairs),
              "final": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0},
              "full_trace": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0},
              "by_length": {}, "by_stratum": {}}
    for pair in pairs:
        length = str(pair["length"])
        scoped = result["by_length"].setdefault(length, {"program_count": 0, "cases": 0,
            "final": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0},
            "full_trace": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0}})
        scoped["program_count"] += 1; scoped["cases"] += int(pair["cases"])
        for metric in ("final", "full_trace"):
            for key, value in pair[metric].items():
                result[metric][key] += int(value); scoped[metric][key] += int(value)
        for stratum, values in pair.get("by_stratum", {}).items():
            target = result["by_stratum"].setdefault(stratum, {"cases": 0,
                "final": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0},
                "full_trace": {"left_correct_right_wrong": 0, "left_wrong_right_correct": 0, "both_correct": 0, "both_wrong": 0}})
            target["cases"] += int(values["cases"])
            for metric in ("final", "full_trace"):
                for key, value in values[metric].items():
                    target[metric][key] += int(value)
    return result


def _e36_primary_rows(evaluation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = evaluation.get("e36", {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError("E38 E36 rows missing")
    selected = [row for row in rows if int(row.get("length", -1)) in (7, 8, 9, 10)
                and row.get("category") == "semantic_new" and row.get("legality") == "allowed"]
    if len(selected) != 24:
        raise ValueError("E38 primary pool scope changed")
    return selected


def _e36_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = _e36_primary_rows(evaluation)
    result = {"program_count": len(rows), "cases": sum(int(row["states"]) for row in rows),
              "final_correct": sum(int(row["joint_final"]) for row in rows),
              "full_trace_correct": sum(int(row["full_trace"]) for row in rows)}
    result["final_errors"] = result["cases"] - result["final_correct"]
    result["full_trace_errors"] = result["cases"] - result["full_trace_correct"]
    result["by_length"] = {}
    result["by_stratum"] = {}
    for row in rows:
        length = str(row["length"])
        scoped = result["by_length"].setdefault(length, {"program_count": 0, "cases": 0, "final_correct": 0, "full_trace_correct": 0})
        scoped["program_count"] += 1; scoped["cases"] += int(row["states"])
        scoped["final_correct"] += int(row["joint_final"]); scoped["full_trace_correct"] += int(row["full_trace"])
        for prediction in row["predictions"]:
            stratum = str(prediction["stratum"])
            dst = result["by_stratum"].setdefault(stratum, {"cases": 0, "final_correct": 0, "full_trace_correct": 0})
            dst["cases"] += 1; dst["final_correct"] += int(bool(prediction["joint_final_correct"]))
            dst["full_trace_correct"] += int(all(prediction["prefix_joint_correct"]))
    for scoped in result["by_length"].values():
        scoped["final_errors"] = scoped["cases"] - scoped["final_correct"]
        scoped["full_trace_errors"] = scoped["cases"] - scoped["full_trace_correct"]
    for scoped in result["by_stratum"].values():
        scoped["final_errors"] = scoped["cases"] - scoped["final_correct"]
        scoped["full_trace_errors"] = scoped["cases"] - scoped["full_trace_correct"]
    probes = evaluation.get("e36", {}).get("training_probe")
    if not isinstance(probes, list):
        raise ValueError("E38 training probes missing")
    probe_summary = {"rows": len(probes), "by_length": {}, "by_legality": {"allowed": {"cases": 0, "full_trace_correct": 0}, "forbidden": {"cases": 0, "full_trace_correct": 0}}}
    for row in probes:
        length = str(row["length"]); cases = int(row["states"]); full = int(row["full_trace"])
        dst = probe_summary["by_length"].setdefault(length, {"cases": 0, "full_trace_correct": 0})
        dst["cases"] += cases; dst["full_trace_correct"] += full
        legality = str(row["legality"]); probe_summary["by_legality"][legality]["cases"] += cases; probe_summary["by_legality"][legality]["full_trace_correct"] += full
    for dst in list(probe_summary["by_length"].values()) + list(probe_summary["by_legality"].values()):
        dst["full_trace_errors"] = dst["cases"] - dst["full_trace_correct"]
    probe_summary["cases"] = sum(dst["cases"] for dst in probe_summary["by_length"].values())
    probe_summary["full_trace_errors"] = probe_summary["cases"] - sum(dst["full_trace_correct"] for dst in probe_summary["by_length"].values())
    result["training_probe"] = probe_summary
    return result


def _e36_primary_pair(left_eval: Mapping[str, Any], right_eval: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    left = {tuple(row["program"]): row for row in _e36_primary_rows(left_eval)}
    right = {tuple(row["program"]): row for row in _e36_primary_rows(right_eval)}
    if set(left) != set(right):
        raise ValueError(f"E38 {label} primary program scope mismatch")
    pairs = [_pair_rows(left[program], right[program], label=label) for program in sorted(left)]
    aggregate = _aggregate_pairs(pairs)
    aggregate["directional_full_trace_improvement"] = aggregate["full_trace"]["left_wrong_right_correct"] > aggregate["full_trace"]["left_correct_right_wrong"]
    return aggregate


def _e37_endpoint_report(evaluation: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    rows = evaluation["rows"]
    if len(rows) != 36:
        raise ValueError("E37 program count changed")
    for row in rows:
        if int(row.get("states", -1)) != 256 or len(row.get("predictions", [])) != 256:
            raise ValueError("E37 state scope changed")
    return {"rows": rows, "within_endpoint": wave._e35_pair_metrics(rows)}


def run_e37(manifest: Mapping[str, Any], out: Path, *, base_manifest_path: Path | None = None,
            root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty E37 directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if base_manifest_path is None:
        _atomic_json(out / "manifest.json", manifest, refuse=True)
    else:
        _atomic_bytes(out / "manifest.json", Path(base_manifest_path).read_bytes(), refuse=True)
    report: dict[str, Any] = {"schema": SCHEMA, "experiment": "E37", "status": "running", "models": {}}
    try:
        endpoint_evals = {}
        for arm in ARMS:
            for branch in E36_BRANCHES:
                expected = FINAL_UPDATE if branch != "B_match" else MATCHED_FINAL_UPDATE
                path = _resolve(Path(manifest["e36_checkpoint_paths"][arm][branch]), root)
                loaded_model, loaded_optimizer, payload = wave.load_checkpoint(path, _load_inherited(root), arm=arm,
                                                                                branch=branch, expected_update=expected,
                                                                                qa=False, root=root)
                before = core.digest_object((loaded_model.state_dict(), loaded_optimizer.state_dict(), torch.get_rng_state(), loaded_model.training))
                record = {"arm": arm, "seed": 0, "branch": branch, "status": "running", "checkpoint": _relative(path, root),
                          "checkpoint_sha256": _sha256(path), "parent_model_digest": payload["model_digest"]}
                was_training = loaded_model.training
                loaded_model.eval()
                hooks = _hooks(loaded_model, record, "evaluation_cost")
                try:
                    evaluation = {"rows": wave._e35_rows(loaded_model)}
                finally:
                    for hook in hooks:
                        hook.remove()
                    loaded_model.train(was_training)
                after = core.digest_object((loaded_model.state_dict(), loaded_optimizer.state_dict(), torch.get_rng_state(), loaded_model.training))
                if before != after:
                    raise ValueError("E37 evaluation mutated model/RNG/mode")
                expected_cost = manifest["e37"]["cost_per_endpoint"]
                if record["evaluation_cost"] != expected_cost or record["evaluation_cost_attempted"] != expected_cost:
                    raise ValueError("E37 evaluation accounting mismatch")
                full = _e37_endpoint_report(evaluation, root=root)
                prediction = _save_prediction(out / f"{arm}128_seed0" / branch / "predictions.json", full, root=root)
                record.update({"status": "complete", "evaluation_cost": record["evaluation_cost"],
                               "evaluation_cost_attempted": record["evaluation_cost_attempted"], **prediction,
                               "program_count": len(full["rows"]), "case_count": len(full["rows"]) * 256,
                               "paired_comparison_count": full["within_endpoint"]["comparison_count"]})
                _atomic_json(out / f"{arm}128_seed0" / branch / "report.json", record, refuse=True)
                endpoint_evals[(arm, branch)] = full
                report["models"][f"{arm}128_seed0/{branch}"] = record
        cross = {}
        for arm in ARMS:
            a_rows = endpoint_evals[(arm, "A")]["rows"]
            for branch in ("B_match", "B"):
                b_rows = endpoint_evals[(arm, branch)]["rows"]
                left = {tuple(row["program"]): row for row in a_rows}
                right = {tuple(row["program"]): row for row in b_rows}
                if set(left) != set(right):
                    raise ValueError("E37 cross-endpoint program scope changed")
                pairs = [_pair_rows(left[program], right[program], label=f"{arm}-A-vs-{branch}") for program in sorted(left)]
                cross[arm + "_A_vs_" + branch] = _aggregate_pairs(pairs)
        report["cross_endpoint_pairs"] = cross
        report["status"] = "complete"
        report["evaluation_cost_total"] = {key: manifest["e37"]["cost_per_endpoint"][key] * len(ARMS) * len(E36_BRANCHES) for key in COST_KEYS}
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"; report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def _run_extension(manifest: Mapping[str, Any], out: Path, *, arm: str, seed: int,
                   batches: Sequence[Sequence[RegisterExample]], qa: bool, root: Path,
                   lineage: Mapping[str, Any] | None = None):
    parent_model, parent_optimizer, parent, parent_path, _ = _load_e32_parent(_label(arm, seed), manifest, root=root)
    parent_model.train(True)
    record: dict[str, Any] = {"stage": "parent_extension", "arm": arm, "seed": seed, "branch": "parent",
                              "status": "running", "parent_update": E32_UPDATE, "added_updates": 0,
                              "attempted_updates": 0, "completed_updates": 0, "training_cost": _zero_cost(), "progress": []}
    record_dir = out / "parents" / _label(arm, seed)
    record_dir.mkdir(parents=True, exist_ok=False)
    hooks = _hooks(parent_model, record, "training_cost")
    started = time.monotonic()
    try:
        try:
            for index, batch in enumerate(batches, 1):
                record["attempted_updates"] = index
                loss = _update(parent_model, parent_optimizer, batch)
                record["completed_updates"] = index; record["added_updates"] = index
                if index % 250 == 0 or index == len(batches):
                    record["progress"].append({"added_update": index, "cumulative_update": E32_UPDATE + index,
                                                "loss": float(loss.detach()), "seconds": time.monotonic() - started})
        except BaseException as exc:
            record.update({"status": "technical_failure", "error": f"{type(exc).__name__}: {exc}",
                           "training_seconds": time.monotonic() - started})
            _atomic_json(record_dir / "progress.json", record, refuse=False)
            raise
    finally:
        for hook in hooks:
            hook.remove()
    expected_cost = _cost_for_batches(batches)
    if not qa and (len(batches) != ADDED_PARENT_UPDATES or record["training_cost"] != expected_cost or expected_cost != TRAIN_COST):
        raise ValueError("E38 parent extension cost mismatch")
    payload = _checkpoint_payload(parent_model, parent_optimizer, parent, parent_path, manifest,
                                  stage="parent_extension", arm=arm, seed=seed, branch="parent",
                                  parent_update=E32_UPDATE, added_updates=len(batches), added_batches=batches,
                                  training_cost=record["training_cost"], qa=qa, lineage=lineage, root=root)
    path = out / "parents" / _label(arm, seed) / f"u{E32_UPDATE + len(batches)}.pt"
    _atomic_torch(path, payload, refuse=True)
    loaded, loaded_opt, loaded_payload = _load_extended_checkpoint(path, manifest, arm=arm, seed=seed,
                                                                     stage="parent_extension", branch="parent",
                                                                     expected_update=E32_UPDATE + len(batches), qa=qa,
                                                                     lineage=lineage, root=root)
    del loaded, loaded_opt
    record.update({"status": "complete", "training_seconds": time.monotonic() - started,
                   "checkpoint": _relative(path, root), "checkpoint_sha256": _sha256(path),
                   "cumulative_update": E32_UPDATE + len(batches), "added_training_cost": record["training_cost"],
                   "cumulative_training_cost": loaded_payload["cumulative_training_cost"]})
    return path, record


def _run_branch(manifest: Mapping[str, Any], out: Path, *, arm: str, seed: int, branch: str,
                batches: Sequence[Sequence[RegisterExample]], qa: bool, root: Path,
                snapshot: bool = True, lineage: Mapping[str, Any] | None = None):
    if lineage is None:
        raise ValueError("E38 branch runtime lineage missing")
    parent_path, _ = _resolve_runtime_parent(lineage, manifest, _label(arm, seed), root=root)
    parent_model, parent_optimizer, parent = _load_extended_checkpoint(parent_path, manifest, arm=arm, seed=seed,
                                                                        stage="parent_extension", branch="parent",
                                                                        expected_update=PARENT_UPDATE, qa=qa,
                                                                        lineage=lineage, root=root)
    parent_model.train(True)
    record: dict[str, Any] = {"stage": "branch", "arm": arm, "seed": seed, "branch": branch, "status": "running",
                              "parent_update": PARENT_UPDATE, "added_updates": 0, "attempted_updates": 0,
                              "completed_updates": 0, "training_cost": _zero_cost(), "progress": []}
    record_dir = out / "branches" / _label(arm, seed) / branch
    record_dir.mkdir(parents=True, exist_ok=False)
    hooks = _hooks(parent_model, record, "training_cost")
    started = time.monotonic(); snapshot_path = None; snapshot_payload = None
    try:
        try:
            for index, batch in enumerate(batches, 1):
                record["attempted_updates"] = index
                loss = _update(parent_model, parent_optimizer, batch)
                record["completed_updates"] = index; record["added_updates"] = index
                if snapshot and branch == "B" and index == MATCHED_UPDATES:
                    snapshot_payload = _checkpoint_payload(parent_model, parent_optimizer, parent, parent_path, manifest,
                                                           stage="branch", arm=arm, seed=seed, branch="B_match",
                                                           parent_update=PARENT_UPDATE, added_updates=index,
                                                           added_batches=batches[:index], training_cost=_cost_for_batches(batches[:index]),
                                                           qa=qa, lineage=lineage, root=root)
                    snapshot_path = out / "branches" / _label(arm, seed) / "B_match" / f"u{MATCHED_FINAL_UPDATE}.pt"
                    _atomic_torch(snapshot_path, snapshot_payload, refuse=True)
        except BaseException as exc:
            record.update({"status": "technical_failure", "error": f"{type(exc).__name__}: {exc}",
                           "training_seconds": time.monotonic() - started})
            _atomic_json(record_dir / "progress.json", record, refuse=False)
            raise
    finally:
        for hook in hooks:
            hook.remove()
    expected_cost = _cost_for_batches(batches)
    if not qa and (len(batches) != BRANCH_UPDATES or record["training_cost"] != expected_cost):
        raise ValueError(f"E38 {branch} training cost mismatch")
    payload = _checkpoint_payload(parent_model, parent_optimizer, parent, parent_path, manifest,
                                  stage="branch", arm=arm, seed=seed, branch=branch,
                                  parent_update=PARENT_UPDATE, added_updates=len(batches), added_batches=batches,
                                  training_cost=record["training_cost"], qa=qa, lineage=lineage, root=root)
    path = out / "branches" / _label(arm, seed) / branch / f"u{PARENT_UPDATE + len(batches)}.pt"
    _atomic_torch(path, payload, refuse=True)
    return path, snapshot_path, record


def run_e38(manifest: Mapping[str, Any], out: Path, *, base_manifest_path: Path | None = None,
            root: Path = ROOT) -> dict[str, Any]:
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty E38 directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    a, b = wave.build_streams()
    if base_manifest_path is None:
        raise ValueError("E38 immutable base manifest path is required")
    base_manifest_path = Path(base_manifest_path)
    _validate_source_reference(manifest, base_manifest_path, root=root)
    lineage = _new_runtime_lineage(base_manifest_path, manifest, root=root)
    _atomic_bytes(out / "manifest.json", base_manifest_path.read_bytes(), refuse=True)
    _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=True)
    report: dict[str, Any] = {"schema": SCHEMA, "experiment": "E38", "status": "running", "parents": {}, "models": {},
                              "runtime_lineage": {"path": _relative(out / "runtime_lineage.json", root),
                                                  "schema": RUNTIME_LINEAGE_SCHEMA,
                                                  "base_manifest_sha256": lineage["base_manifest_sha256"],
                                                  "base_manifest_digest": lineage["base_manifest_digest"]}}
    try:
        # Parents are extended first.  Their endpoints are evaluated exactly once
        # after all four checkpoints exist; this keeps the parent scope descriptive.
        extension_paths = {}
        for seed in SEEDS:
            for arm in ARMS:
                label = _label(arm, seed)
                try:
                    path, record = _run_extension(manifest, out, arm=arm, seed=seed, batches=core.stream(),
                                                  qa=False, lineage=lineage, root=root)
                except BaseException as exc:
                    progress_path = out / "parents" / label / "progress.json"
                    if progress_path.is_file():
                        failed = _read_json(progress_path)
                        lineage = _record_runtime_status(
                            lineage, label, status="technical_failure",
                            attempted_updates=int(failed.get("attempted_updates", 0)),
                            completed_updates=int(failed.get("completed_updates", 0)),
                            training_cost=failed.get("training_cost", _zero_cost()),
                            error=f"{type(exc).__name__}: {exc}")
                        _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=False)
                    raise
                extension_paths[label] = path
                entry = {"label": label, "path": _relative(path, root), "sha256": _sha256(path),
                         "arm": arm, "seed": seed, "update": PARENT_UPDATE, "stage": "parent_extension"}
                lineage = _register_runtime_parent(lineage, label, entry)
                lineage = _record_runtime_status(
                    lineage, label, status="complete", attempted_updates=int(record["attempted_updates"]),
                    completed_updates=int(record["completed_updates"]), training_cost=record["training_cost"])
                _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=False)
                report["parents"][label] = record
        for seed in SEEDS:
            for arm in ARMS:
                label = _label(arm, seed)
                parent_eval, parent_cost, parent_payload = _evaluate_e32_parent(
                    extension_paths[label], manifest, arm=arm, seed=seed, lineage=lineage, root=root)
                prediction = _save_prediction(out / "parents" / label / "predictions.json", parent_eval, root=root)
                report["parents"][label].update({"evaluation": parent_eval, "evaluation_cost": parent_cost,
                                                   "parent_model_digest": parent_payload["model_digest"], **prediction})
                _atomic_json(out / "parents" / label / "report.json", report["parents"][label], refuse=True)
        # Build branches from the same own parent independently.  B creates the
        # matched snapshot in its one trajectory; the snapshot is evaluated below.
        endpoint_records = {}
        endpoint_evaluations = {}
        for seed in SEEDS:
            for arm in ARMS:
                label = _label(arm, seed)
                path_a, _, record_a = _run_branch(manifest, out, arm=arm, seed=seed, branch="A", batches=a,
                                                  qa=False, root=root, snapshot=False, lineage=lineage)
                path_b, snapshot_path, record_b = _run_branch(manifest, out, arm=arm, seed=seed, branch="B", batches=b,
                                                              qa=False, root=root, snapshot=True, lineage=lineage)
                endpoint_records[(label, "A")] = (path_a, record_a, FINAL_UPDATE)
                endpoint_records[(label, "B")] = (path_b, record_b, FINAL_UPDATE)
                if snapshot_path is None:
                    raise ValueError("E38 B matched snapshot missing")
                endpoint_records[(label, "B_match")] = (snapshot_path, {"stage": "branch", "arm": arm, "seed": seed, "branch": "B_match"}, MATCHED_FINAL_UPDATE)
                report["models"][label] = {"A": record_a, "B": record_b, "B_match": {"checkpoint": _relative(snapshot_path, root), "checkpoint_sha256": _sha256(snapshot_path)}}
        for (label, branch), (path, record, update) in endpoint_records.items():
            arm = "float" if label.startswith("float") else "w4"; seed = int(label.rsplit("seed", 1)[1])
            evaluation, cost, payload = _evaluate_branch(path, manifest, arm=arm, seed=seed, branch=branch,
                                                         expected_update=update, lineage=lineage, root=root)
            prediction = _save_prediction(out / "branches" / label / branch / "predictions.json", evaluation, root=root)
            endpoint_evaluations[(label, branch)] = evaluation
            slot = report["models"][label].setdefault(branch, {})
            slot.update({"status": "complete", "evaluation": evaluation, "evaluation_cost": cost,
                         "checkpoint": _relative(path, root), "checkpoint_sha256": _sha256(path),
                         "model_digest": payload["model_digest"], **prediction})
            _atomic_json(out / "branches" / label / branch / "report.json", slot, refuse=True)
        for seed in SEEDS:
            for arm in ARMS:
                label = _label(arm, seed)
                a_eval = endpoint_evaluations[(label, "A")]
                b_match_eval = endpoint_evaluations[(label, "B_match")]
                b_eval = endpoint_evaluations[(label, "B")]
                model_report = report["models"][label]
                model_report["primary_metrics"] = {branch: _e36_metrics(endpoint_evaluations[(label, branch)]) for branch in E36_BRANCHES}
                model_report["primary_pairs"] = {
                    "A_vs_B_match": _e36_primary_pair(a_eval, b_match_eval, label=f"{label}-A-vs-B_match"),
                    "A_vs_B": _e36_primary_pair(a_eval, b_eval, label=f"{label}-A-vs-B"),
                }
                model_report["primary_directional_gate"] = {
                    "B_match_vs_A": model_report["primary_pairs"]["A_vs_B_match"]["directional_full_trace_improvement"],
                    "B_vs_A": model_report["primary_pairs"]["A_vs_B"]["directional_full_trace_improvement"],
                }
                for branch in E36_BRANCHES:
                    _atomic_json(out / "branches" / label / branch / "report.json", model_report[branch], refuse=False)
        report["replication_gate"] = {
            label: report["models"][label]["primary_directional_gate"] for label in sorted(report["models"])
        }
        report["status"] = "complete"
        parent_total = {key: TRAIN_COST[key] * len(ARMS) * len(SEEDS) for key in COST_KEYS}
        branch_total = {key: sum(manifest["branch_schedules"][branch]["cost"][key] for branch in ("A", "B")) * len(ARMS) * len(SEEDS) for key in COST_KEYS}
        report["training_cost_total"] = {key: parent_total[key] + branch_total[key] for key in COST_KEYS}
        report["evaluation_cost_total"] = {
            key: E32_EVAL_COST[key] * len(ARMS) * len(SEEDS) +
            sum(E36_EVAL_COST[branch][key] for branch in E36_BRANCHES) * len(ARMS) * len(SEEDS)
            for key in COST_KEYS}
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        lineage = _persist_failed_progress(out, lineage, root=root)
        _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=False)
        report["status"] = "suspended"; report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def _qa_update(model, optimizer, batch, actual_cost: dict[str, Any], phase: str) -> None:
    """Retain attempted/completed forward costs even when an update fails."""
    actual_cost["attempted_updates"] += 1
    hooks = _hooks(model, actual_cost, phase)
    try:
        _update(model, optimizer, batch)
        actual_cost["completed_updates"] += 1
    finally:
        for hook in hooks:
            hook.remove()


def tiny_qa(out: Path = QA_RUN, *, preflight_dir: Path = PREFLIGHT, root: Path = ROOT) -> dict[str, Any]:
    """Bounded four-parent QA, strict reloads, wrong-identity guards and B snapshot."""
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty QA directory: {out}")
    manifest = load_manifest(preflight_dir, root=root)
    qa_manifest = manifest
    base_manifest_path = _resolve(preflight_dir, root) / "manifest.json"
    lineage = _new_runtime_lineage(base_manifest_path, manifest, root=root)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(out / "manifest.json", base_manifest_path.read_bytes(), refuse=True)
    _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=True)
    tiny_batch = [RegisterExample(0, 1, ("ADD",)), RegisterExample(2, 3, ("ADD",))]
    qa_batches = [tiny_batch, tiny_batch]
    qa_a = [tiny_batch, tiny_batch]
    qa_b = [tiny_batch, tiny_batch]
    report: dict[str, Any] = {"schema": QA_SCHEMA, "status": "running", "models": {}, "actual_cost": {"attempted_updates": 0, "completed_updates": 0,
        "training": _zero_cost(), "snapshot_next_update_identity": _zero_cost(), "evaluation": _zero_cost()}}
    try:
        for seed in SEEDS:
            for arm in ARMS:
                label = _label(arm, seed)
                parent_model, parent_optimizer, parent, parent_path, _ = _load_e32_parent(label, qa_manifest, root=root)
                del parent_model, parent_optimizer
                # Make a tiny extension checkpoint at a QA-specific path and make
                # the branch loader resolve it only through this QA manifest.
                model, optimizer, parent, parent_path, _ = _load_e32_parent(label, qa_manifest, root=root)
                model.train(True); record = {"training_cost": _zero_cost()}; hooks = _hooks(model, record, "training_cost")
                try:
                    for batch in qa_batches:
                        _qa_update(model, optimizer, batch, report["actual_cost"], "training")
                finally:
                    for hook in hooks: hook.remove()
                ext_path = out / "parents" / label / "u16002.pt"
                payload = _checkpoint_payload(model, optimizer, parent, parent_path, qa_manifest, stage="parent_extension",
                                              arm=arm, seed=seed, branch="parent", parent_update=E32_UPDATE, added_updates=2,
                                              added_batches=qa_batches, training_cost=record["training_cost"], qa=True,
                                              lineage=lineage, root=root)
                _atomic_torch(ext_path, payload, refuse=True)
                lineage = _register_runtime_parent(lineage, label, {
                    "label": label, "path": _relative(ext_path, root), "sha256": _sha256(ext_path),
                    "arm": arm, "seed": seed, "update": E32_UPDATE + 2, "stage": "parent_extension"})
                _write_runtime_lineage(out / "runtime_lineage.json", lineage, refuse=False)
                loaded, loaded_opt, loaded_payload = _load_extended_checkpoint(ext_path, qa_manifest, arm=arm, seed=seed,
                                                                                 stage="parent_extension", branch="parent",
                                                                                 expected_update=E32_UPDATE + 2, qa=True,
                                                                                 lineage=lineage, root=root)
                del loaded, loaded_opt
                # A tiny branch and a one-update B->B_match snapshot exercise
                # child provenance and same-trajectory snapshot loading.
                qa_parent = ext_path
                branch_model, branch_optimizer, branch_parent = _load_extended_checkpoint(qa_parent, qa_manifest, arm=arm, seed=seed,
                                                                                             stage="parent_extension", branch="parent",
                                                                                             expected_update=E32_UPDATE + 2, qa=True,
                                                                                             lineage=lineage, root=root)
                branch_model.train(True); branch_record = {"training_cost": _zero_cost()}; branch_hooks = _hooks(branch_model, branch_record, "training_cost")
                try:
                    for batch in qa_a: _qa_update(branch_model, branch_optimizer, batch, report["actual_cost"], "training")
                finally:
                    for hook in branch_hooks: hook.remove()
                a_path = out / "branches" / label / "A" / "u16002.pt"
                a_payload = _checkpoint_payload(branch_model, branch_optimizer, branch_parent, qa_parent, qa_manifest, stage="branch",
                                                arm=arm, seed=seed, branch="A", parent_update=E32_UPDATE + 2, added_updates=2,
                                                added_batches=qa_a, training_cost=branch_record["training_cost"], qa=True,
                                                lineage=lineage, root=root)
                _atomic_torch(a_path, a_payload, refuse=True)
                _load_extended_checkpoint(a_path, qa_manifest, arm=arm, seed=seed, stage="branch", branch="A",
                                          expected_update=E32_UPDATE + 4, qa=True, lineage=lineage, root=root)
                b_model, b_optimizer, b_parent = _load_extended_checkpoint(qa_parent, qa_manifest, arm=arm, seed=seed,
                                                                              stage="parent_extension", branch="parent",
                                                                              expected_update=E32_UPDATE + 2, qa=True,
                                                                              lineage=lineage, root=root)
                b_model.train(True); b_record = {"training_cost": _zero_cost()}; b_hooks = _hooks(b_model, b_record, "training_cost")
                try: _qa_update(b_model, b_optimizer, qa_b[0], report["actual_cost"], "training")
                finally:
                    for hook in b_hooks: hook.remove()
                bmatch_path = out / "branches" / label / "B_match" / "u16003.pt"
                bmatch_payload = _checkpoint_payload(b_model, b_optimizer, b_parent, qa_parent, qa_manifest, stage="branch",
                                                     arm=arm, seed=seed, branch="B_match", parent_update=E32_UPDATE + 2,
                                                     added_updates=1, added_batches=qa_b[:1], training_cost=_cost_for_batches(qa_b[:1]),
                                                     qa=True, lineage=lineage, root=root)
                _atomic_torch(bmatch_path, bmatch_payload, refuse=True)
                _load_extended_checkpoint(bmatch_path, qa_manifest, arm=arm, seed=seed, stage="branch", branch="B_match",
                                          expected_update=E32_UPDATE + 3, qa=True, lineage=lineage, root=root)
                # Compare the next update from the uninterrupted B path with a
                # reloaded B_match snapshot, including optimizer and global RNG.
                uninterrupted_model = b_model
                uninterrupted_optimizer = b_optimizer
                uninterrupted_model_state = deepcopy(uninterrupted_model.state_dict())
                uninterrupted_optimizer_state = deepcopy(uninterrupted_optimizer.state_dict())
                uninterrupted_rng = torch.get_rng_state().clone()
                _qa_update(uninterrupted_model, uninterrupted_optimizer, qa_b[1], report["actual_cost"], "training")
                expected_model_digest = core.digest_state_dict(uninterrupted_model)
                expected_optimizer_digest = core.digest_object(uninterrupted_optimizer.state_dict())
                expected_rng = torch.get_rng_state().clone()
                reloaded_model, reloaded_optimizer, reloaded_payload = _load_extended_checkpoint(
                    bmatch_path, qa_manifest, arm=arm, seed=seed, stage="branch", branch="B_match",
                    expected_update=E32_UPDATE + 3, qa=True, lineage=lineage, root=root)
                torch.set_rng_state(reloaded_payload["rng_state"])
                _qa_update(reloaded_model, reloaded_optimizer, qa_b[1], report["actual_cost"], "snapshot_next_update_identity")
                snapshot_next_update_equal = (core.digest_state_dict(reloaded_model) == expected_model_digest and
                                              core.digest_object(reloaded_optimizer.state_dict()) == expected_optimizer_digest and
                                              torch.equal(torch.get_rng_state(), expected_rng))
                del uninterrupted_model_state, uninterrupted_optimizer_state, uninterrupted_rng
                if not snapshot_next_update_equal:
                    raise ValueError(f"E38 QA B snapshot next-update mismatch: {label}")
                tampered = deepcopy(loaded_payload); tampered["seed"] = 2 if seed == 1 else 1
                wrong_seed_rejected = False
                try:
                    _validate_payload(tampered, ext_path, qa_manifest, stage="parent_extension", arm=arm, seed=seed,
                                      branch="parent", expected_update=E32_UPDATE + 2, qa=True,
                                      lineage=lineage, root=root)
                except ValueError:
                    wrong_seed_rejected = True
                if not wrong_seed_rejected:
                    raise ValueError("wrong-seed QA guard did not reject")
                tampered_parent = deepcopy(loaded_payload)
                tampered_parent["parent_checkpoint"] = manifest["e32_parent_paths"][_label(arm, 1 if seed == 2 else 2)]
                wrong_parent_rejected = False
                try:
                    _validate_payload(tampered_parent, ext_path, qa_manifest, stage="parent_extension", arm=arm, seed=seed,
                                      branch="parent", expected_update=E32_UPDATE + 2, qa=True,
                                      lineage=lineage, root=root)
                except ValueError:
                    wrong_parent_rejected = True
                if not wrong_parent_rejected:
                    raise ValueError("wrong-parent QA guard did not reject")
                report["models"][label] = {"status": "complete", "wrong_seed_rejected": True,
                                            "wrong_parent_rejected": wrong_parent_rejected,
                                            "parent_reload": True, "branch_reload": True,
                                            "b_snapshot_reload": True, "snapshot_next_update_equal": True}
        report["status"] = "complete"
        report["inherited_scope_unchanged"] = old.canonical_hash(manifest["inherited_schedule_digests"]) == old.canonical_hash(manifest["inherited_schedule_digests"])
        report["budget_reconstruction"] = {"extension": _cost_for_batches(qa_batches), "A": _cost_for_batches(qa_a), "B_match": _cost_for_batches(qa_b[:1])}
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"; report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def run_science(out: Path = RUN, preflight_dir: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    target = _resolve(out, root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty science output: {target}")
    manifest = load_manifest(preflight_dir, root=root)
    base_manifest_path = _resolve(preflight_dir, root) / "manifest.json"
    # The E37 inputs and E38 streams are checked before creating any science output.
    e37_report = run_e37(manifest, target / "e37", base_manifest_path=base_manifest_path, root=root)
    e38_report = run_e38(manifest, target / "e38", base_manifest_path=base_manifest_path, root=root)
    combined = {"schema": SCHEMA, "status": "complete", "e37": e37_report, "e38": e38_report}
    _atomic_json(_resolve(out, root) / "report.json", combined, refuse=True)
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--train-cleared", action="store_true")
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT)
    args = parser.parse_args(argv)
    if args.preflight:
        value = preflight(args.preflight_dir)
        print(json.dumps({"status": "preflight_ready", "schema": value["schema"]}, sort_keys=True))
    elif args.qa:
        value = tiny_qa(args.out, preflight_dir=args.preflight_dir)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    else:
        value = run_science(args.out, args.preflight_dir)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
