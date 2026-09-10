#!/usr/bin/env python3
"""Deterministic data, provenance, and runner for the E35/E36 first wave.

The module has a deliberately small scientific surface.  It reuses the
accepted E33 checkpoint loader and the width-128 model factory, while keeping
the new stream, DSL pools, checkpoint schema, and prediction files separate
from every older experiment.  Importing it performs no model forward, load,
or update; ``--preflight`` is symbolic/data preparation only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from looped_bitnet import width_e32 as width
from looped_bitnet import longer_native8_e20 as old
from scripts import continuation_e33 as e33
from scripts import width_e32 as e32
from scripts.longer_native8_e20 import _update


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "e35_e36_length_wave_v1"
QA_SCHEMA = "e35_e36_length_wave_qa_v1"
ARMS = ("float", "w4")
BRANCHES = ("A", "B", "B_match")
SEED = 0
WIDTH = 128
PARENT_UPDATE = 32000
ADDED_UPDATES = 8000
MATCHED_UPDATES = 4572
BATCH_SIZE = 64
NATIVE_STEPS = 8
LONG_POOL_LIMIT = 32
POOL_LIMIT = 6
E35_PREFIXES = (("ADD", "ADD"), ("XOR", "SWAP"), ("SWAP", "XOR"))
E35_OPS = ("ADD", "XOR", "SWAP")
E35_PADDING = (0, 1, 2, 3)
TRAIN_STATES = tuple(tuple(s) for s in dsl.state_split()["train"])
HELDOUT_STATES = tuple(tuple(s) for s in dsl.state_split()["validation"] + dsl.state_split()["test"])
STATE_STRATUM = {tuple(s): name for name, values in dsl.state_split().items() for s in values}

PARENT_RUN = Path("runs/e33_continuation")
PARENT_PREFLIGHT = Path("runs/e33_continuation_preflight")
PARENT_MANIFEST_SHA256 = "563255aebf55cb39b963c0cfccdc512e19d726fb0bb00f82cd31a2894d0265bb"
PREFLIGHT = Path("runs/e35_e36_preflight")
RUN = Path("runs/e35_e36_length_wave")
PROTOCOL = Path("results/E35_E36_PROTOCOL.md")

COST_KEYS = ("program_forwards", "program_state_cases", "readout_positions", "internal_state_substeps")


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing temporary collision: {tmp}")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _atomic_torch(path: Path, value: Any, *, refuse: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing temporary collision: {tmp}")
    torch.save(value, tmp)
    tmp.replace(path)


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return width.sha256_file(Path(path))


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _zero_cost() -> dict[str, int]:
    return dict.fromkeys(COST_KEYS, 0)


def _cost_for_batches(batches: Sequence[Sequence[dsl.RegisterExample]]) -> dict[str, int]:
    examples = sum(len(batch) for batch in batches)
    positions = sum(len(batch[0].program) * len(batch) for batch in batches)
    return {"program_forwards": len(batches), "program_state_cases": examples,
            "readout_positions": positions, "internal_state_substeps": positions * NATIVE_STEPS}


def _program_json(program: Sequence[str]) -> list[str]:
    return list(program)


def _program_digest(program: Sequence[str], prefix: str) -> str:
    return hashlib.sha256((prefix + json.dumps(list(program), separators=(",", ":"))).encode()).hexdigest()


def _allowed(program: Sequence[str]) -> bool:
    return not dsl.forbidden(program)


@lru_cache(maxsize=100_000)
def _signature(program: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    return dsl.semantic_signature(program)


def _select_class_representatives(length: int, *, limit: int | None = None) -> list[tuple[str, ...]]:
    candidates = [p for p in itertools.product(dsl.OPS, repeat=length) if _allowed(p)]
    classes: dict[tuple[tuple[int, int], ...], list[tuple[str, ...]]] = {}
    for program in candidates:
        classes.setdefault(_signature(program), []).append(program)
    representatives = []
    for signature, members in sorted(classes.items(), key=lambda row: hashlib.sha256(
            (f"E36-class-v1:{length}:" + repr(row[0])).encode()).hexdigest()):
        representatives.append(min(members, key=lambda p: _program_digest(p, "E36-member-v1:")))
        if limit is not None and len(representatives) >= limit:
            break
    return representatives


def _long_program_pools() -> dict[int, list[tuple[str, ...]]]:
    return {length: _select_class_representatives(length, limit=LONG_POOL_LIMIT) for length in (4, 5, 6)}


def _source_short_batches() -> dict[int, list[list[dsl.RegisterExample]]]:
    source = width.stream()
    grouped = {length: [] for length in (1, 2, 3)}
    for batch in source:
        if len(batch) != BATCH_SIZE or len(batch[0].program) not in grouped:
            raise ValueError("E36 source short stream shape changed")
        if any(len(example.program) != len(batch[0].program) for example in batch):
            raise ValueError("E36 source short batch is not homogeneous")
        grouped[len(batch[0].program)].append(list(batch))
    if {length: len(values) for length, values in grouped.items()} != {1: 5328, 2: 5344, 3: 5328}:
        raise ValueError("E36 E33 source stream counts changed")
    return grouped


def _a_lengths() -> tuple[int, ...]:
    return (1, 2, 3) * 2666 + (3, 3)


def _b_lengths() -> tuple[int, ...]:
    return (1, 2, 3, 4, 5, 6) * 1333 + (1, 4)


def _build_a_short() -> list[list[dsl.RegisterExample]]:
    grouped = _source_short_batches()
    cursors = {length: 0 for length in grouped}
    result = []
    for length in _a_lengths():
        result.append(grouped[length][cursors[length]])
        cursors[length] += 1
    if cursors != {1: 2666, 2: 2666, 3: 2668}:
        raise ValueError("E36 A source short counts changed")
    return result


def _long_batch_stream(length: int, count: int, pool: Sequence[tuple[str, ...]]) -> list[list[dsl.RegisterExample]]:
    if length not in (4, 5, 6) or not pool:
        raise ValueError("invalid E36 long stream request")
    rng = random.Random(36000 + length)
    batches = []
    for _ in range(count):
        batch = [dsl.RegisterExample(*rng.choice(TRAIN_STATES), tuple(rng.choice(pool))) for _ in range(BATCH_SIZE)]
        batches.append(batch)
    return batches


@lru_cache(maxsize=1)
def build_streams() -> tuple[tuple[list[dsl.RegisterExample], ...], tuple[list[dsl.RegisterExample], ...]]:
    """Build the exact A/B streams without touching torch model state."""
    a = _build_a_short()
    long_pools = _long_program_pools()
    long_batches = {length: _long_batch_stream(length, 1334 if length == 4 else 1333, long_pools[length])
                    for length in (4, 5, 6)}
    b = []
    short_cursor = 0
    long_cursor = {length: 0 for length in (4, 5, 6)}
    for length in _b_lengths():
        if length <= 3:
            b.append(a[short_cursor])
            short_cursor += 1
        else:
            b.append(long_batches[length][long_cursor[length]])
            long_cursor[length] += 1
    if short_cursor != 4000 or long_cursor != {4: 1334, 5: 1333, 6: 1333}:
        raise ValueError("E36 B stream partition changed")
    if [len(batch[0].program) for batch in b[:MATCHED_UPDATES]] and sum(len(batch[0].program) for batch in b[:MATCHED_UPDATES]) != 16002:
        raise ValueError("E36 matched-instruction prefix changed")
    if sum(len(batch[0].program) for batch in a) != 16002 or sum(len(batch[0].program) for batch in b) != 27998:
        raise ValueError("E36 schedule instruction counts changed")
    for left, right in zip(a[:4000], [batch for batch in b if len(batch[0].program) <= 3]):
        if old.batch_digest([left]) != old.batch_digest([right]):
            raise ValueError("E36 common short stream mismatch")
    return tuple(a), tuple(b)


def _stream_programs(batches: Sequence[Sequence[dsl.RegisterExample]]) -> set[tuple[str, ...]]:
    return {tuple(example.program) for batch in batches for example in batch}


def _prefix_signatures(programs: Sequence[Sequence[str]]) -> set[tuple[tuple[int, int], ...]]:
    result: set[tuple[tuple[int, int], ...]] = set()
    for program in programs:
        for end in range(1, len(program) + 1):
            result.add(_signature(program[:end]))
    return result


def _class_pool(programs: Sequence[tuple[str, ...]], limit: int = POOL_LIMIT) -> list[tuple[str, ...]]:
    by_signature: dict[tuple[tuple[int, int], ...], list[tuple[str, ...]]] = {}
    for program in programs:
        by_signature.setdefault(_signature(program), []).append(program)
    ordered = sorted(by_signature.items(), key=lambda row: hashlib.sha256(
        ("E36-eval-v1:" + repr(row[0])).encode()).hexdigest())
    return [min(members, key=lambda p: _program_digest(p, "E36-eval-member-v1:")) for _, members in ordered[:limit]]


@lru_cache(maxsize=1)
def _enumerate_feasibility() -> dict[str, Any]:
    """Bounded exact semantic scan; returns unknown status if the cap fires."""
    started = time.monotonic()
    deadline = started + 60.0
    class_cap = 100_000
    all_seen: dict[tuple[tuple[int, int], ...], int] = {}
    allowed_seen: dict[tuple[tuple[int, int], ...], int] = {}
    per_length = {}
    complete_through = 0
    capped = False
    for length in range(1, 11):
        all_count = allowed_count = 0
        all_classes: set[tuple[tuple[int, int], ...]] = set()
        allowed_classes: set[tuple[tuple[int, int], ...]] = set()
        for program in itertools.product(dsl.OPS, repeat=length):
            if time.monotonic() > deadline or len(all_seen) >= class_cap:
                capped = True
                break
            signature = _signature(program)
            all_count += 1
            all_classes.add(signature)
            all_seen.setdefault(signature, length)
            if _allowed(program):
                allowed_count += 1
                allowed_classes.add(signature)
                allowed_seen.setdefault(signature, length)
        if capped:
            per_length[str(length)] = {"all_programs": all_count, "allowed_programs": allowed_count,
                                       "all_classes": len(all_classes), "allowed_classes": len(allowed_classes),
                                       "complete": False}
            break
        complete_through = length
        per_length[str(length)] = {"all_programs": all_count, "allowed_programs": allowed_count,
                                   "all_classes": len(all_classes), "allowed_classes": len(allowed_classes),
                                   "complete": True}
    return {"cap_seconds": 60, "cap_classes": class_cap, "capped": capped,
            "complete_through_length": complete_through,
            "unknown_minimality_from_length": complete_through + 1 if capped else None,
            "all_cumulative_classes": len(all_seen), "allowed_cumulative_classes": len(allowed_seen),
            "all_new_min_length": {str(n): sum(v == n for v in all_seen.values()) for n in range(1, complete_through + 1)},
            "allowed_new_min_length": {str(n): sum(v == n for v in allowed_seen.values()) for n in range(1, complete_through + 1)},
            "per_length": per_length}


def _evaluation_pools(a: Sequence[Sequence[dsl.RegisterExample]], b: Sequence[Sequence[dsl.RegisterExample]]) -> dict[str, Any]:
    ancestor_programs = _stream_programs(width.stream())
    a_programs, b_programs = _stream_programs(a), _stream_programs(b)
    all_trained = sorted(ancestor_programs | a_programs | b_programs)
    prefix_sigs = _prefix_signatures(all_trained)
    pools: dict[str, Any] = {}
    for length in range(1, 11):
        all_programs = list(itertools.product(dsl.OPS, repeat=length))
        categories: dict[str, list[tuple[str, ...]]] = {"train_syntax": [], "syntax_heldout_exposed": [], "semantic_new": []}
        for legal in (True, False):
            members = [p for p in all_programs if _allowed(p) == legal]
            for program in members:
                signature = _signature(program)
                if program in all_trained:
                    category = "train_syntax"
                elif signature in prefix_sigs:
                    category = "syntax_heldout_exposed"
                else:
                    category = "semantic_new"
                categories[category].append(program)
        pools[str(length)] = {
            category: {
                "allowed": [_program_json(p) for p in _class_pool([p for p in values if _allowed(p)])],
                "forbidden": [_program_json(p) for p in _class_pool([p for p in values if not _allowed(p)])],
            } for category, values in categories.items()
        }
    return {"programs": pools,
            "prefix_signature_count": len(prefix_sigs),
            "ancestor_program_count": len(ancestor_programs),
            "a_program_count": len(a_programs), "b_program_count": len(b_programs),
            "all_trained_program_count": len(all_trained)}


@lru_cache(maxsize=1)
def frozen_evaluation_pools() -> dict[str, Any]:
    a, b = build_streams()
    return _evaluation_pools(a, b)


def _probe_programs(a: Sequence[Sequence[dsl.RegisterExample]], b: Sequence[Sequence[dsl.RegisterExample]]) -> dict[str, dict[str, list[list[str]]]]:
    def pick(batches: Sequence[Sequence[dsl.RegisterExample]]) -> dict[str, list[list[str]]]:
        result = {}
        programs = _stream_programs(batches)
        for length in range(1, 11):
            values = sorted((p for p in programs if len(p) == length), key=lambda p: _program_digest(p, "E36-probe-v1:"))
            result[str(length)] = [_program_json(p) for p in values[:POOL_LIMIT]]
        return result
    return {"A8000": pick(a), "B4572_common": pick(b[:MATCHED_UPDATES])}


def make_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    # Read the already accepted E33 manifest directly.  Re-running E33's
    # ancestor guard walks here would deserialize historical models during a
    # symbolic preflight and would make the preparation cost misleading.
    parent_manifest_path = _resolve(PARENT_PREFLIGHT / "manifest.json", root)
    if _sha256(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("accepted E33 parent manifest hash changed")
    parent_manifest = _read_json(parent_manifest_path)
    if parent_manifest.get("schema") != e33.SCHEMA:
        raise ValueError("accepted E33 parent manifest schema changed")
    a, b = build_streams()
    a_cost = _cost_for_batches(a)
    b_cost = _cost_for_batches(b)
    b_match_cost = _cost_for_batches(b[:MATCHED_UPDATES])
    parent_hashes = {}
    parent_paths = {}
    for arm in ARMS:
        path = _resolve(PARENT_RUN / f"{arm}128_seed0" / "u32000.pt", root)
        parent_paths[arm] = _relative(path, root)
        parent_hashes[arm] = _sha256(path)
    source_paths = (Path("scripts/length_wave_e35_e36.py"), Path("tests/test_length_wave_e35_e36.py"), PROTOCOL)
    source_hashes = {str(p): _sha256(_resolve(p, root)) for p in source_paths if _resolve(p, root).is_file()}
    e35_programs = []
    for prefix in E35_PREFIXES:
        for opcode in E35_OPS:
            for padding in E35_PADDING:
                e35_programs.append(tuple(prefix) + ("SWAP", "SWAP") * padding + (opcode,))
    a_programs, b_programs = _stream_programs(a), _stream_programs(b)
    evaluation = frozen_evaluation_pools()
    return {
        "schema": SCHEMA, "status": "preflight", "seed": SEED, "width": WIDTH,
        "native_steps": NATIVE_STEPS, "batch_size": BATCH_SIZE, "arms": list(ARMS),
        "parent_update": PARENT_UPDATE, "added_updates": ADDED_UPDATES,
        "matched_updates": MATCHED_UPDATES, "parent_run": str(PARENT_RUN),
        "parent_manifest_digest": old.canonical_hash(parent_manifest),
        "parent_paths": parent_paths, "parent_checkpoint_sha256": parent_hashes,
        "parent_manifest_sha256": _sha256(_resolve(PARENT_PREFLIGHT / "manifest.json", root)),
        "source_hashes": source_hashes,
        "a_schedule": {"updates": len(a), "lengths": [len(batch[0].program) for batch in a],
                        "stream_digest": old.batch_digest(a), "target_digest": old.target_digest(a),
                        "cost": a_cost, "programs": [_program_json(p) for p in sorted(a_programs)]},
        "b_schedule": {"updates": len(b), "lengths": [len(batch[0].program) for batch in b],
                        "stream_digest": old.batch_digest(b), "target_digest": old.target_digest(b),
                        "cost": b_cost, "programs": [_program_json(p) for p in sorted(b_programs)]},
        "b_matched_schedule": {"updates": MATCHED_UPDATES, "stream_digest": old.batch_digest(b[:MATCHED_UPDATES]),
                               "target_digest": old.target_digest(b[:MATCHED_UPDATES]), "cost": b_match_cost},
        "long_program_pools": {str(n): [_program_json(p) for p in pool] for n, pool in _long_program_pools().items()},
        "e35": {"programs": [_program_json(p) for p in e35_programs], "program_count": len(e35_programs),
                "states": [list(s) for s in dsl.STATE_ORDER],
                "cost_per_arm": {"program_forwards": len(e35_programs), "program_state_cases": len(e35_programs) * 256,
                                  "readout_positions": sum(len(p) for p in e35_programs) * 256,
                                  "internal_state_substeps": sum(len(p) for p in e35_programs) * 256 * NATIVE_STEPS}},
        "feasibility": _enumerate_feasibility(),
        "evaluation": evaluation,
        "probe_programs": _probe_programs(a, b),
        "training_states": [list(s) for s in TRAIN_STATES],
        "heldout_states": [list(s) for s in HELDOUT_STATES],
        "state_split_digest": old.canonical_hash({"train": [list(s) for s in TRAIN_STATES], "heldout": [list(s) for s in HELDOUT_STATES]}),
        "ancestor_stream_digest": old.batch_digest(width.stream()),
        "ancestor_target_digest": old.target_digest(width.stream()),
        "training_program_prefix_semantics": {
            "excluded_from_semantic_new": True,
            "description": "full-domain function signatures of every nonempty prefix of E33 ancestor, A, and B trained programs",
        },
    }


def preflight(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty preflight directory: {path}")
    manifest = make_manifest(root=root)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "manifest.json", manifest, refuse=True)
    return manifest


def load_manifest(path: Path = PREFLIGHT, *, root: Path = ROOT) -> dict[str, Any]:
    path = _resolve(path, root)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("E35/E36 frozen manifest schema changed")
    for relative, expected in manifest.get("source_hashes", {}).items():
        artifact = _resolve(Path(relative), root)
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"E35/E36 source hash mismatch: {relative}")
    if manifest.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256:
        raise ValueError("E35/E36 accepted parent manifest reference changed")
    for arm, relative in manifest.get("parent_paths", {}).items():
        artifact = _resolve(Path(relative), root)
        if not artifact.is_file() or _sha256(artifact) != manifest["parent_checkpoint_sha256"].get(arm):
            raise ValueError(f"E35/E36 parent checkpoint hash mismatch: {arm}")
    return manifest


def _parent_label(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("unknown E36 arm")
    return f"{arm}128_seed0"


def load_parent(arm: str, *, root: Path = ROOT):
    parent_manifest_path = _resolve(PARENT_PREFLIGHT / "manifest.json", root)
    if _sha256(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("accepted E33 parent manifest hash changed")
    parent_manifest = _read_json(parent_manifest_path)
    path = _resolve(PARENT_RUN / f"{_parent_label(arm)}" / "u32000.pt", root)
    return e33.load_checkpoint(path, parent_manifest, arm=arm, root=root, qa=False, expected_update=PARENT_UPDATE)


def _hooks(model: torch.nn.Module, record: dict[str, Any], field: str):
    attempted = record.setdefault(field + "_attempted", _zero_cost())
    completed = record.setdefault(field, _zero_cost())

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
            raise FloatingPointError("nonfinite E35/E36 model output")

    return pre, model.register_forward_hook(post)


def _e35_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for prefix in E35_PREFIXES:
        for opcode in E35_OPS:
            for padding in E35_PADDING:
                program = tuple(prefix) + ("SWAP", "SWAP") * padding + (opcode,)
                row = dsl.evaluate_program(model, program, dsl.STATE_ORDER, include_predictions=True)
                for prediction in row["predictions"]:
                    prediction["stratum"] = STATE_STRATUM[tuple(prediction["state"])]
                row["prefix"] = list(prefix)
                row["opcode"] = opcode
                row["padding_pairs"] = padding
                row["length"] = len(program)
                rows.append(row)
    return rows


def _manifest_programs(manifest: Mapping[str, Any], branch: str) -> list[tuple[str, ...]]:
    values = manifest["evaluation"]["programs"]
    result = []
    for length in sorted(values, key=int):
        for category in ("train_syntax", "syntax_heldout_exposed", "semantic_new"):
            for legality in ("allowed", "forbidden"):
                result.extend(tuple(p) for p in values[length][category][legality])
    return result


def _evaluate_length(model: torch.nn.Module, manifest: Mapping[str, Any], branch: str) -> dict[str, Any]:
    rows = []
    for length in sorted(manifest["evaluation"]["programs"], key=int):
        for category in ("train_syntax", "syntax_heldout_exposed", "semantic_new"):
            for legality in ("allowed", "forbidden"):
                programs = manifest["evaluation"]["programs"][length][category][legality]
                for program_list in programs:
                    program = tuple(program_list)
                    row = dsl.evaluate_program(model, program, dsl.STATE_ORDER, include_predictions=True)
                    for prediction in row["predictions"]:
                        prediction["stratum"] = STATE_STRATUM[tuple(prediction["state"])]
                    row.update({"category": category, "legality": legality, "length": int(length), "branch": branch})
                    rows.append(row)
    probe_rows = []
    probes = manifest["probe_programs"]["A8000" if branch == "A" else "B4572_common"]
    for length in sorted(probes, key=int):
        for program_list in probes[length]:
            program = tuple(program_list)
            row = dsl.evaluate_program(model, program, TRAIN_STATES, include_predictions=True)
            for prediction in row["predictions"]:
                prediction["stratum"] = "train"
            row.update({"category": "training_probe", "legality": "allowed" if _allowed(program) else "forbidden",
                        "length": int(length), "branch": branch})
            probe_rows.append(row)
    return {"rows": rows, "training_probe": probe_rows}


def evaluate_scientific(model: torch.nn.Module, manifest: Mapping[str, Any], branch: str) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        return {"e36": _evaluate_length(model, manifest, branch)}
    finally:
        model.train(was_training)


def _e35_pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pair every padded trace with its k=0 trace using saved predictions."""
    by_key = {(tuple(row["prefix"]), row["opcode"], int(row["padding_pairs"])): row for row in rows}
    result = []
    for prefix in E35_PREFIXES:
        for opcode in E35_OPS:
            base = by_key[(prefix, opcode, 0)]
            for padding in E35_PADDING[1:]:
                current = by_key[(prefix, opcode, padding)]
                left = {tuple(p["state"]): p for p in base["predictions"]}
                right = {tuple(p["state"]): p for p in current["predictions"]}
                if set(left) != set(right) or len(left) != 256:
                    raise ValueError("E35 paired state scope changed")
                final = {"short_correct_long_wrong": 0, "short_wrong_long_correct": 0,
                         "both_correct": 0, "both_wrong": 0}
                full = {key: 0 for key in final}
                conditional = {"denominator": 0, "short_correct_long_wrong": 0,
                               "short_wrong_long_correct": 0, "both_correct": 0, "both_wrong": 0}
                for state in sorted(left):
                    a, b = left[state], right[state]
                    if a["target_trace"][-2:] != b["target_trace"][-2:]:
                        raise ValueError("E35 pre-O/final target mismatch within identity family")
                    ac, bc = bool(a["joint_final_correct"]), bool(b["joint_final_correct"])
                    key = "both_correct" if ac and bc else "short_correct_long_wrong" if ac else "short_wrong_long_correct" if bc else "both_wrong"
                    final[key] += 1
                    af, bf = all(a["prefix_joint_correct"]), all(b["prefix_joint_correct"])
                    key = "both_correct" if af and bf else "short_correct_long_wrong" if af else "short_wrong_long_correct" if bf else "both_wrong"
                    full[key] += 1
                    # The conditional O result has a denominator only when the
                    # immediately preceding readouts match in both branches.
                    if len(a["prefix_joint_correct"]) >= 2 and a["prefix_joint_correct"][-2] and b["prefix_joint_correct"][-2]:
                        conditional["denominator"] += 1
                        key = "both_correct" if ac and bc else "short_correct_long_wrong" if ac else "short_wrong_long_correct" if bc else "both_wrong"
                        conditional[key] += 1
                result.append({"prefix": list(prefix), "opcode": opcode, "padding_pairs": padding,
                               "length_short": 3, "length_long": len(current["program"]),
                               "denominator": 256, "final": final, "full_trace": full,
                               "conditional_on_pre_o_correct": conditional})
    return {"comparisons": result, "comparison_count": len(result)}


def execute_e35(out: Path, manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Run the fixed E35 inference-only diagnostic after CODE CLEAR."""
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty E35 directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", manifest, refuse=True)
    report: dict[str, Any] = {"schema": SCHEMA, "experiment": "E35", "status": "running", "arms": list(ARMS), "models": {}}
    try:
        for arm in ARMS:
            model, optimizer, parent = load_parent(arm, root=root)
            del optimizer
            if parent.get("update") != PARENT_UPDATE or parent.get("arm") != arm:
                raise ValueError("E35 parent identity mismatch")
            record: dict[str, Any] = {"arm": arm, "parent_update": PARENT_UPDATE,
                                      "parent_model_digest": parent["model_digest"],
                                      "parent_checkpoint": _relative(_resolve(PARENT_RUN / f"{_parent_label(arm)}" / "u32000.pt", root), root),
                                      "status": "running"}
            before = width.digest_object((model.state_dict(), torch.get_rng_state(), model.training))
            was_training = model.training
            model.eval()
            hooks = _hooks(model, record, "evaluation_cost")
            try:
                evaluation = {"rows": _e35_rows(model)}
            finally:
                for hook in hooks:
                    hook.remove()
                model.train(was_training)
            after = width.digest_object((model.state_dict(), torch.get_rng_state(), model.training))
            if before != after:
                raise ValueError("E35 evaluation mutated model/RNG/mode")
            expected_cost = manifest["e35"]["cost_per_arm"]
            if record["evaluation_cost"] != expected_cost or record["evaluation_cost_attempted"] != expected_cost:
                raise ValueError("E35 evaluation accounting mismatch")
            evaluation["paired"] = _e35_pair_metrics(evaluation["rows"])
            prediction_path = out / f"{arm}128_seed0" / "predictions.json"
            _atomic_json(prediction_path, evaluation, refuse=True)
            record.update({"status": "complete", "evaluation_cost": record["evaluation_cost"],
                           "evaluation_cost_attempted": record["evaluation_cost_attempted"],
                           "prediction_path": _relative(prediction_path, root),
                           "prediction_sha256": _sha256(prediction_path),
                           "program_count": len(evaluation["rows"]), "case_count": len(evaluation["rows"]) * 256,
                           "paired_comparison_count": evaluation["paired"]["comparison_count"]})
            _atomic_json(out / f"{arm}128_seed0" / "report.json", record, refuse=True)
            report["models"][f"{arm}128_seed0"] = record
        report["status"] = "complete"
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"
        report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def _parent_ref(parent: Mapping[str, Any], arm: str, *, root: Path) -> dict[str, Any]:
    path = _resolve(PARENT_RUN / f"{_parent_label(arm)}" / "u32000.pt", root)
    return {"path": _relative(path, root), "sha256": _sha256(path), "schema": parent["schema"],
            "label": parent["label"], "arm": parent["arm"], "width": parent["width"],
            "seed": parent["seed"], "update": parent["update"], "model_digest": parent["model_digest"],
            "optimizer_digest": parent["optimizer_digest"], "rng_digest": parent["rng_digest"],
            "manifest_digest": parent["manifest_digest"], "source_hashes": deepcopy(parent["source_hashes"]),
            "stream_digest": parent["cumulative_stream_digest"], "target_digest": parent["cumulative_target_digest"],
            "training_cost": deepcopy(parent["cumulative_training_cost"])}


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, parent: Mapping[str, Any],
                       manifest: Mapping[str, Any], *, arm: str, branch: str, added_updates: int,
                       training_cost: Mapping[str, int], qa: bool = False) -> dict[str, Any]:
    if arm not in ARMS or branch not in BRANCHES or not model.training:
        raise ValueError("E36 checkpoint identity")
    if added_updates <= 0:
        raise ValueError("E36 update count")
    if set(training_cost) != set(COST_KEYS):
        raise ValueError("E36 training cost keys")
    state = deepcopy(model.state_dict())
    opt = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    parent_ref = _parent_ref(parent, arm, root=ROOT)
    return {
        "schema": QA_SCHEMA if qa else SCHEMA, "qa": bool(qa), "arm": arm, "branch": branch,
        "width": WIDTH, "seed": SEED, "parent_update": PARENT_UPDATE, "added_updates": added_updates,
        "cumulative_update": PARENT_UPDATE + added_updates, "update": PARENT_UPDATE + added_updates,
        "native_steps": NATIVE_STEPS, "parameter_count": width.PARAMETER_COUNT_128, "training_mode": True,
        "parent": parent_ref, "parent_checkpoint": parent_ref["path"], "parent_checkpoint_sha256": parent_ref["sha256"],
        "parent_manifest_digest": parent["manifest_digest"], "manifest_digest": old.canonical_hash(manifest),
        "parent_model_digest": parent["model_digest"], "parent_optimizer_digest": parent["optimizer_digest"],
        "parent_rng_digest": parent["rng_digest"], "model_digest": width.digest_state_dict(model),
        "optimizer_state_dict": opt, "optimizer_digest": width.digest_object(opt), "rng_state": rng,
        "rng_digest": width.digest_object(rng), "state_dict": state,
        "parent_training_cost": deepcopy(parent["cumulative_training_cost"]), "added_training_cost": dict(training_cost),
        "cumulative_training_cost": {key: int(parent["cumulative_training_cost"][key]) + int(training_cost[key]) for key in COST_KEYS},
        "source_hashes": deepcopy(manifest["source_hashes"]), "parent_source_hashes": deepcopy(parent["source_hashes"]),
    }


def _validate_checkpoint_payload(payload: Mapping[str, Any], manifest: Mapping[str, Any], *, arm: str, branch: str,
                                 expected_update: int, qa: bool, root: Path) -> None:
    expected_schema = QA_SCHEMA if qa else SCHEMA
    for key, value in {"schema": expected_schema, "qa": qa, "arm": arm, "branch": branch, "width": WIDTH,
                       "seed": SEED, "parent_update": PARENT_UPDATE, "added_updates": expected_update - PARENT_UPDATE,
                       "cumulative_update": expected_update, "update": expected_update, "training_mode": True}.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"E36 checkpoint identity mismatch: {key}")
    if payload.get("manifest_digest") != old.canonical_hash(manifest):
        raise ValueError("E36 manifest digest mismatch")
    if payload.get("native_steps") != NATIVE_STEPS or payload.get("parameter_count") != width.PARAMETER_COUNT_128:
        raise ValueError("E36 checkpoint model inventory mismatch")
    if payload.get("source_hashes") != manifest.get("source_hashes"):
        raise ValueError("E36 checkpoint source provenance mismatch")
    parent_path = _resolve(Path(payload.get("parent_checkpoint", "")), root)
    expected_parent = _resolve(Path(manifest["parent_paths"][arm]), root)
    if parent_path != expected_parent or not parent_path.is_file():
        raise ValueError("E36 parent checkpoint path mismatch")
    if _sha256(parent_path) != manifest["parent_checkpoint_sha256"][arm] or payload.get("parent_checkpoint_sha256") != manifest["parent_checkpoint_sha256"][arm]:
        raise ValueError("E36 parent checkpoint hash mismatch")
    actual_parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if not isinstance(actual_parent, Mapping) or actual_parent.get("schema") != e33.SCHEMA:
        raise ValueError("E36 parent payload malformed")
    if payload.get("parent_manifest_digest") != actual_parent.get("manifest_digest"):
        raise ValueError("E36 parent manifest lineage mismatch")
    if payload.get("parent_model_digest") != actual_parent.get("model_digest") or payload.get("parent_optimizer_digest") != actual_parent.get("optimizer_digest") or payload.get("parent_rng_digest") != actual_parent.get("rng_digest"):
        raise ValueError("E36 parent digest lineage mismatch")
    if payload.get("parent_training_cost") != actual_parent.get("cumulative_training_cost"):
        raise ValueError("E36 parent cumulative cost mismatch")
    rng = payload.get("rng_state")
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or payload.get("rng_digest") != width.digest_object(rng):
        raise ValueError("E36 checkpoint RNG digest mismatch")
    expected_parent_ref = _parent_ref(actual_parent, arm, root=root)
    if payload.get("parent") != expected_parent_ref:
        raise ValueError("E36 parent reference mismatch")
    if not qa:
        a, b = build_streams()
        schedule = a if branch == "A" else b[:MATCHED_UPDATES] if branch == "B_match" else b
        expected_cost = _cost_for_batches(schedule[: expected_update - PARENT_UPDATE])
        if payload.get("added_training_cost") != expected_cost:
            raise ValueError("E36 added training cost mismatch")
        expected_cumulative = {key: int(payload["parent_training_cost"][key]) + int(expected_cost[key]) for key in COST_KEYS}
        if payload.get("cumulative_training_cost") != expected_cumulative:
            raise ValueError("E36 cumulative training cost mismatch")
    for path, expected in payload.get("source_hashes", {}).items():
        if not _resolve(Path(path), root).is_file() or _sha256(_resolve(Path(path), root)) != expected:
            raise ValueError(f"E36 source hash mismatch: {path}")


def load_checkpoint(path: Path, manifest: Mapping[str, Any], *, arm: str, branch: str,
                    expected_update: int, qa: bool = False, root: Path = ROOT):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("E36 checkpoint payload malformed")
    _validate_checkpoint_payload(payload, manifest, arm=arm, branch=branch, expected_update=expected_update, qa=qa, root=root)
    model = width.build_initial_model(WIDTH, arm, SEED, root=root)
    optimizer = old.make_optimizer(model)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["rng_state"])
    model.train(True)
    if width.digest_state_dict(model) != payload["model_digest"] or width.digest_object(optimizer.state_dict()) != payload["optimizer_digest"]:
        raise ValueError("E36 checkpoint digest mismatch")
    return model, optimizer, dict(payload)


def _evaluate_checkpoint(path: Path, manifest: Mapping[str, Any], *, arm: str, branch: str,
                         expected_update: int, root: Path, qa: bool) -> tuple[dict[str, Any], dict[str, int]]:
    restored, restored_optimizer, _ = load_checkpoint(path, manifest, arm=arm, branch=branch,
                                                      expected_update=expected_update, qa=qa, root=root)
    before = width.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
    record: dict[str, Any] = {"evaluation_cost": _zero_cost()}
    hooks = _hooks(restored, record, "evaluation_cost")
    try:
        evaluation = evaluate_scientific(restored, manifest, branch)
    finally:
        for hook in hooks:
            hook.remove()
    after = width.digest_object((restored.state_dict(), restored_optimizer.state_dict(), torch.get_rng_state(), restored.training))
    if before != after or not restored.training:
        raise ValueError("E36 evaluation mutated model/RNG/mode")
    return evaluation, record["evaluation_cost"]


def _run_branch(out: Path, manifest: Mapping[str, Any], *, arm: str, branch: str, batches: Sequence[Sequence[dsl.RegisterExample]],
                added_updates: int, qa: bool = False, root: Path = ROOT,
                snapshot_updates: Sequence[int] = ()) -> dict[str, Any]:
    model, optimizer, parent = load_parent(arm, root=root)
    model.train(True)
    record: dict[str, Any] = {"arm": arm, "branch": branch, "status": "running", "parent_update": PARENT_UPDATE,
                              "added_updates": 0, "attempted_updates": 0, "completed_updates": 0,
                              "training_cost": _zero_cost(), "progress": []}
    hooks = _hooks(model, record, "training_cost")
    started = time.monotonic()
    snapshots: dict[str, dict[str, Any]] = {}
    try:
        for index, batch in enumerate(batches, 1):
            record["attempted_updates"] = index
            _update(model, optimizer, batch)
            record["completed_updates"] = index
            record["added_updates"] = index
            if index in snapshot_updates:
                snapshot_branch = "B_match" if index == MATCHED_UPDATES else f"snapshot_{index}"
                snapshot_cost = _cost_for_batches(batches[:index])
                snapshot_payload = checkpoint_payload(model, optimizer, parent, manifest, arm=arm,
                                                      branch=snapshot_branch, added_updates=index,
                                                      training_cost=snapshot_cost, qa=qa)
                snapshot_path = out / f"{arm}128_seed0" / snapshot_branch / f"u{PARENT_UPDATE + index}.pt"
                _atomic_torch(snapshot_path, snapshot_payload, refuse=True)
                snapshots[snapshot_branch] = {"path": _relative(snapshot_path, root),
                                              "sha256": _sha256(snapshot_path), "update": PARENT_UPDATE + index,
                                              "added_training_cost": snapshot_cost}
    finally:
        for hook in hooks:
            hook.remove()
    if record["completed_updates"] != added_updates:
        raise ValueError("E36 update count mismatch")
    expected_cost = _cost_for_batches(batches)
    if record["training_cost"] != expected_cost:
        raise ValueError("E36 training accounting mismatch")
    payload = checkpoint_payload(model, optimizer, parent, manifest, arm=arm, branch=branch,
                                added_updates=added_updates, training_cost=record["training_cost"], qa=qa)
    path = out / f"{arm}128_seed0" / branch / f"u{PARENT_UPDATE + added_updates}.pt"
    _atomic_torch(path, payload, refuse=True)
    evaluation, evaluation_cost = _evaluate_checkpoint(path, manifest, arm=arm, branch=branch,
                                                       expected_update=PARENT_UPDATE + added_updates,
                                                       qa=qa, root=root)
    snapshot_evaluations = {}
    for snapshot_branch, snapshot in snapshots.items():
        snapshot_path = _resolve(Path(snapshot["path"]), root)
        snapshot_eval, snapshot_cost = _evaluate_checkpoint(snapshot_path, manifest, arm=arm,
                                                            branch=snapshot_branch, expected_update=snapshot["update"],
                                                            qa=qa, root=root)
        snapshot_evaluations[snapshot_branch] = {"evaluation": snapshot_eval, "evaluation_cost": snapshot_cost}
    record.update({"status": "complete", "training_seconds": time.monotonic() - started,
                   "checkpoint": _relative(path, root), "checkpoint_sha256": _sha256(path),
                   "evaluation": evaluation, "evaluation_cost": evaluation_cost,
                   "snapshot_checkpoints": snapshots, "snapshot_evaluations": snapshot_evaluations,
                   "cumulative_update": PARENT_UPDATE + added_updates,
                   "cumulative_training_cost": payload["cumulative_training_cost"]})
    _atomic_json(out / f"{arm}128_seed0" / branch / "report.json", record, refuse=True)
    return record


def execute(out: Path, manifest: Mapping[str, Any], *, root: Path = ROOT, qa: bool = False) -> dict[str, Any]:
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {out}")
    a, b = build_streams()
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", manifest, refuse=True)
    report = {"schema": SCHEMA, "status": "running", "seed": SEED, "arms": list(ARMS), "branches": ["A", "B_match", "B"], "models": {}}
    try:
        # Sequential ordering is deliberate: it keeps memory bounded and makes
        # the command's accounting easy to audit.  A/B and precision are still
        # paired by the immutable manifest and parent RNG, not by selection.
        for arm in ARMS:
            report["models"][f"{arm}128_seed0"] = {}
            report["models"][f"{arm}128_seed0"]["A"] = _run_branch(out, manifest, arm=arm, branch="A", batches=a,
                                                                        added_updates=ADDED_UPDATES, qa=qa, root=root)
            report["models"][f"{arm}128_seed0"]["B"] = _run_branch(out, manifest, arm=arm, branch="B", batches=b,
                                                                        added_updates=ADDED_UPDATES, qa=qa, root=root,
                                                                        snapshot_updates=(MATCHED_UPDATES,))
        report["status"] = "complete"
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        report["status"] = "suspended"
        report["technical_failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(out / "report.json", report, refuse=True)
        raise


def tiny_qa(out: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """One bounded resume/counter check; this path is never scientific evidence."""
    out = _resolve(out, root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty QA directory: {out}")
    manifest = make_manifest(root=root)
    a, b = build_streams()
    qa_batches = [a[0], a[1]]
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "manifest.json", manifest, refuse=True)
    result = {"schema": QA_SCHEMA, "status": "running", "models": {}}
    for arm in ARMS:
        model, optimizer, parent = load_parent(arm, root=root)
        model.train(True)
        record = {"arm": arm, "branch": "A", "training_cost": _zero_cost(), "attempted_updates": 0, "completed_updates": 0}
        hooks = _hooks(model, record, "training_cost")
        try:
            for index, batch in enumerate(qa_batches, 1):
                record["attempted_updates"] = index
                _update(model, optimizer, batch)
                record["completed_updates"] = index
        finally:
            for hook in hooks:
                hook.remove()
        if record["training_cost"] != _cost_for_batches(qa_batches):
            raise ValueError("E36 QA training accounting mismatch")
        payload = checkpoint_payload(model, optimizer, parent, manifest, arm=arm, branch="A",
                                    added_updates=2, training_cost=record["training_cost"], qa=True)
        path = out / f"{arm}128_seed0" / "A" / "u32002.pt"
        _atomic_torch(path, payload, refuse=True)
        restored, restored_optimizer, loaded = load_checkpoint(path, manifest, arm=arm, branch="A", expected_update=32002, qa=True, root=root)
        state = {"model": deepcopy(restored.state_dict()), "optimizer": deepcopy(restored_optimizer.state_dict()),
                 "rng": torch.get_rng_state().clone()}
        # Evaluate only one fixed primitive on two train states, then verify
        # that a reloaded branch takes the same next update as uninterrupted.
        eval_record = {"evaluation_cost": _zero_cost()}
        eval_hooks = _hooks(restored, eval_record, "evaluation_cost")
        try:
            with torch.inference_mode():
                row = dsl.evaluate_program(restored, ("ADD",), TRAIN_STATES[:2], include_predictions=True)
        finally:
            for hook in eval_hooks:
                hook.remove()
        if eval_record["evaluation_cost"] != {"program_forwards": 1, "program_state_cases": 2,
                                                "readout_positions": 2, "internal_state_substeps": 16}:
            raise ValueError("E36 QA evaluation accounting mismatch")
        uninterrupted = width.build_initial_model(WIDTH, arm, SEED, root=root)
        uninterrupted.load_state_dict(state["model"], strict=True)
        uninterrupted_optimizer = old.make_optimizer(uninterrupted)
        uninterrupted_optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        _update(uninterrupted, uninterrupted_optimizer, qa_batches[0])
        uninterrupted_rng = torch.get_rng_state().clone()
        torch.set_rng_state(loaded["rng_state"])
        _update(restored, restored_optimizer, qa_batches[0])
        if width.digest_state_dict(uninterrupted) != width.digest_state_dict(restored) or \
                width.digest_object(uninterrupted_optimizer.state_dict()) != width.digest_object(restored_optimizer.state_dict()) or \
                not torch.equal(uninterrupted_rng, torch.get_rng_state()):
            raise ValueError(f"E36 QA reload next-update mismatch: {arm}")
        # Exercise the registered B4572 snapshot lineage in a tiny two-update
        # branch: save B_match after update one, continue B, and compare the
        # reloaded snapshot's next update with the uninterrupted final state.
        bqa = [b[0], b[1]]
        branch_model, branch_optimizer, branch_parent = load_parent(arm, root=root)
        branch_model.train(True)
        branch_record = {"arm": arm, "branch": "B", "training_cost": _zero_cost()}
        branch_hooks = _hooks(branch_model, branch_record, "training_cost")
        try:
            _update(branch_model, branch_optimizer, bqa[0])
            snapshot_payload = checkpoint_payload(branch_model, branch_optimizer, branch_parent, manifest, arm=arm,
                                                  branch="B_match", added_updates=1,
                                                  training_cost=_cost_for_batches(bqa[:1]), qa=True)
            snapshot_path = out / f"{arm}128_seed0" / "B_match" / "u32001.pt"
            _atomic_torch(snapshot_path, snapshot_payload, refuse=True)
            _update(branch_model, branch_optimizer, bqa[1])
        finally:
            for hook in branch_hooks:
                hook.remove()
        uninterrupted_branch_model_digest = width.digest_state_dict(branch_model)
        uninterrupted_branch_optimizer_digest = width.digest_object(branch_optimizer.state_dict())
        branch_rng = torch.get_rng_state().clone()
        snap_model, snap_optimizer, snap_loaded = load_checkpoint(snapshot_path, manifest, arm=arm, branch="B_match",
                                                                  expected_update=32001, qa=True, root=root)
        _update(snap_model, snap_optimizer, bqa[1])
        snapshot_equal = (width.digest_state_dict(snap_model) == uninterrupted_branch_model_digest and
                          width.digest_object(snap_optimizer.state_dict()) == uninterrupted_branch_optimizer_digest and
                          torch.equal(torch.get_rng_state(), branch_rng))
        if not snapshot_equal:
            raise ValueError(f"E36 QA B snapshot next-update mismatch: {arm}")
        final_payload = checkpoint_payload(branch_model, branch_optimizer, branch_parent, manifest, arm=arm,
                                           branch="B", added_updates=2,
                                           training_cost=_cost_for_batches(bqa), qa=True)
        final_path = out / f"{arm}128_seed0" / "B" / "u32002.pt"
        _atomic_torch(final_path, final_payload, refuse=True)
        load_checkpoint(final_path, manifest, arm=arm, branch="B", expected_update=32002, qa=True, root=root)
        result["models"][arm] = {"status": "complete", "training_cost": record["training_cost"],
                                 "evaluation_cost": eval_record["evaluation_cost"], "reload_next_update_equal": True,
                                 "b_snapshot_reload_next_update_equal": snapshot_equal,
                                 "joint_final": row["joint_final"]}
    result["status"] = "complete"
    _atomic_json(out / "report.json", result, refuse=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--e35", action="store_true")
    group.add_argument("--train-cleared", action="store_true")
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT)
    args = parser.parse_args(argv)
    if args.preflight:
        value = preflight(args.preflight_dir)
        print(json.dumps({"status": "preflight_ready", "schema": value["schema"]}, sort_keys=True))
    elif args.qa:
        value = tiny_qa(args.out)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    elif args.e35:
        value = execute_e35(args.out, load_manifest(args.preflight_dir), root=ROOT)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    else:
        value = execute(args.out, load_manifest(args.preflight_dir), qa=False)
        print(json.dumps({"status": value["status"], "path": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
