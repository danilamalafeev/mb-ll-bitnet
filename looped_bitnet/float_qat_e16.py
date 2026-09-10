"""E16 same-architecture float versus QAT helpers.

This module is intentionally small and has no training side effects at import
time.  The E16 runner wraps the frozen E15 register model and data helpers; it
does not modify E15 code or reuse its checkpoint loader.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .quantization import BitLinear
from .register_e15 import (
    OPS,
    QAT_PARAMETER_COUNT,
    QATRegisterModel,
    RegisterExample,
    batch_digest,
    evaluate_program as _e15_evaluate_program,
    loss_for_batch,
    make_paired_batches,
    protocol_manifest,
    sha256_file,
    state_split,
    validate_manifest_schema,
)
from .runtime import environment, seed_everything


PROJECT_ROOT = Path(__file__).parents[1]
E16_SCHEMA = "e16_float_qat_v1"
ARMS = ("qat", "float")
SEED = 0
UPDATES = 2000
BATCH_SIZE = 64
CPU_THREADS = 4
CHECKPOINT_UPDATE = 2000
PROGRESS_INTERVAL = 250
EXPECTED_PARAMETER_COUNT = QAT_PARAMETER_COUNT
EXPECTED_MANIFEST_HASH = "bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74"
EXPECTED_BATCH_DIGEST = "63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687"
EXPECTED_TARGET_DIGEST = "fe43d8c00b9e31dd07e04e54b913effc390081c715932d6e2945ee2cca79cdfc"
DEFAULT_E15_PREFLIGHT = Path("runs/register_e15_preflight/v7")
DEFAULT_PROTECTED_SNAPSHOT = Path("results/E16_PROTECTED_HASHES.json")
E16_SOURCE_RELATIVE_PATHS = {
    "module": "looped_bitnet/float_qat_e16.py",
    "runner": "scripts/float_qat_e16.py",
    "tests": "tests/test_float_qat_e16.py",
    "protocol": "results/E16_FLOAT_QAT_PROTOCOL.md",
}
RUN_CONFIG: dict[str, Any] = {
    "schema": E16_SCHEMA,
    "seed": SEED,
    "updates": UPDATES,
    "batch_size": BATCH_SIZE,
    "cpu_threads": CPU_THREADS,
    "deterministic": True,
    "device": "cpu",
    "dtype": "float32",
    "substeps_per_instruction": 4,
    "progress_interval": PROGRESS_INTERVAL,
}
OPTIMIZER_CONFIG: dict[str, Any] = {
    "type": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "amsgrad": False,
    "foreach": False,
    "grad_clip": 1.0,
    "zero_grad_set_to_none": True,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def json_hash(value: Any) -> str:
    """Alias used by callers that distinguish JSON hashes from file hashes."""
    return canonical_hash(value)


def digest_state_dict(model: nn.Module) -> str:
    """Stable digest of state names, dtypes, shapes, and CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(cpu.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / relative) for name, relative in E16_SOURCE_RELATIVE_PATHS.items()}


def protocol_hash(root: Path = PROJECT_ROOT) -> str:
    return sha256_file(Path(root) / E16_SOURCE_RELATIVE_PATHS["protocol"])


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return canonical_hash(manifest)


def load_frozen_manifest(path: Path = DEFAULT_E15_PREFLIGHT, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen V7 manifest: {path}") from exc
    validate_manifest_schema(manifest)
    if manifest_hash(manifest) != EXPECTED_MANIFEST_HASH:
        raise ValueError("V7 manifest canonical hash mismatch")
    if manifest.get("batch_digest_seed0") != EXPECTED_BATCH_DIGEST:
        raise ValueError("V7 manifest full batch digest mismatch")
    expected = protocol_manifest()
    if manifest.get("programs") != expected.get("programs"):
        raise ValueError("V7 manifest program sets changed")
    if manifest.get("state_split") != expected.get("state_split"):
        raise ValueError("V7 manifest state order changed")
    if not isinstance(manifest.get("source_hashes"), dict):
        raise ValueError("V7 manifest source hashes are missing")
    # The V7 source map is checked explicitly without relying on list ordering.
    from .register_e15 import FROZEN_SOURCE_RELATIVE_PATHS
    current = {name: sha256_file(Path(root) / relative) for name, relative in FROZEN_SOURCE_RELATIVE_PATHS.items()}
    if manifest["source_hashes"] != current:
        raise ValueError("frozen E15 source hashes changed")
    return manifest


def bitlinear_inventory(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, BitLinear)]


def _replace_children(module: nn.Module, prefix: str = "") -> list[str]:
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(child, BitLinear):
            replacement = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
            module._modules[name] = replacement
            replaced.append(qualified)
        else:
            replaced.extend(_replace_children(child, qualified))
    return replaced


def replace_all_bitlinear(model: nn.Module) -> tuple[nn.Module, list[str]]:
    """Deep-copy a QAT model and replace every BitLinear recursively."""
    result = deepcopy(model)
    replaced = _replace_children(result)
    if len(replaced) != 14 or bitlinear_inventory(result):
        raise ValueError(f"E16 requires all 14 BitLinear modules replaced; got {replaced}")
    return result, replaced


def build_paired_models(seed: int = SEED) -> tuple[QATRegisterModel, QATRegisterModel, nn.Module, str, list[str]]:
    """Construct QAT and float arms from one seed-controlled master state."""
    if seed != SEED:
        raise ValueError("E16 is registered for seed0 only")
    seed_everything(seed, deterministic=True, cpu_threads=CPU_THREADS)
    base = QATRegisterModel()
    qat = deepcopy(base)
    float_model, replaced = replace_all_bitlinear(base)
    assert_paired_models(qat, float_model, expected_replaced=replaced)
    initial = digest_state_dict(qat)
    return base, qat, float_model, initial, replaced


def assert_paired_models(qat: nn.Module, float_model: nn.Module, *, expected_replaced: Sequence[str] | None = None) -> None:
    if not isinstance(qat, QATRegisterModel) or bitlinear_inventory(qat) != [
        name for name, _ in QATRegisterModel().named_modules() if isinstance(_, BitLinear)
    ]:
        raise ValueError("qat arm does not retain the complete BitLinear inventory")
    if bitlinear_inventory(float_model):
        raise ValueError("float arm still contains BitLinear modules")
    if expected_replaced is not None and list(expected_replaced) != [
        name for name, _ in QATRegisterModel().named_modules() if isinstance(_, BitLinear)
    ]:
        raise ValueError("replacement inventory is incomplete or unexpected")
    qat_state, float_state = qat.state_dict(), float_model.state_dict()
    if list(qat_state) != list(float_state) or set(qat_state) != set(float_state):
        raise ValueError("paired state keys differ")
    qat_params = list(qat.named_parameters()); float_params = list(float_model.named_parameters())
    if [name for name, _ in qat_params] != [name for name, _ in float_params]:
        raise ValueError("paired parameter ordering differs")
    if sum(p.numel() for p in qat.parameters()) != EXPECTED_PARAMETER_COUNT or sum(p.numel() for p in float_model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("paired parameter count mismatch")
    for (name_q, q), (name_f, f) in zip(qat_state.items(), float_state.items()):
        if name_q != name_f or q.shape != f.shape or q.dtype != f.dtype or not torch.equal(q, f):
            raise ValueError(f"paired initial state mismatch: {name_q}")
        if q.data_ptr() == f.data_ptr():
            raise ValueError(f"paired state storage is shared: {name_q}")


def target_digest(batches: Sequence[Sequence[RegisterExample]]) -> str:
    rows = []
    for batch in batches:
        rows.append([[list(target) for target in example.targets] for example in batch])
    return canonical_hash(rows)


def fixed_stream(seed: int = SEED) -> list[list[RegisterExample]]:
    if seed != SEED:
        raise ValueError("E16 is registered for seed0 only")
    batches = make_paired_batches(seed)
    if len(batches) != UPDATES or any(len(batch) != BATCH_SIZE for batch in batches):
        raise ValueError("E16 batch schedule is not 2000 x 64")
    if batch_digest(batches) != EXPECTED_BATCH_DIGEST:
        raise ValueError("E16 full stream digest mismatch")
    if target_digest(batches) != EXPECTED_TARGET_DIGEST:
        raise ValueError("E16 target serialization digest mismatch")
    lengths = [len(batch[0].program) for batch in batches]
    if {str(length): lengths.count(length) for length in (1, 2, 3)} != {"1": 666, "2": 668, "3": 666}:
        raise ValueError("E16 length schedule mismatch")
    return batches


def _exact_programs(manifest: Mapping[str, Any]) -> list[tuple[str, ...]]:
    expected = [tuple(program) for program in protocol_manifest()["programs"]["seen"]]
    actual = [tuple(program) for program in manifest.get("programs", {}).get("seen", [])]
    if actual != expected or len(actual) != 32 or len(set(actual)) != 32:
        raise ValueError("E16 requires exactly frozen seen32 programs in order")
    return actual


def _exact_states(manifest: Mapping[str, Any], split: str) -> list[tuple[int, int]]:
    if split not in {"train", "validation"}:
        raise ValueError("E16 evaluation split must be train or validation")
    expected = protocol_manifest()["state_split"][split]
    actual = manifest.get("state_split", {}).get(split)
    if actual != expected:
        raise ValueError(f"E16 {split} state order changed")
    states = [tuple(state) for state in actual]
    if len(states) != (192 if split == "train" else 32) or len(set(states)) != len(states):
        raise ValueError(f"E16 {split} state set is invalid")
    return states


def evaluate_split(model: nn.Module, manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    programs = _exact_programs(manifest)
    states = _exact_states(manifest, split)
    model.eval()
    device_type = next(model.parameters()).device.type
    rows = []
    for program in programs:
        with torch.autocast(device_type=device_type, enabled=False):
            raw = _e15_evaluate_program(model, program, states, include_predictions=True)
        row = {
            "program": list(program),
            "split": split,
            "n": int(len(states)),
            "states": int(len(states)),
            "final_joint": int(raw["joint_final"]),
            "joint_final": int(raw["joint_final"]),
            "final_x": int(raw["final_x_correct"]),
            "final_y": int(raw["final_y_correct"]),
            "final_x_correct": int(raw["final_x_correct"]),
            "final_y_correct": int(raw["final_y_correct"]),
            "prefix_joint": [int(v) for v in raw["prefix_joint"]],
            "prefix_joint_correct": [int(v) for v in raw["prefix_joint"]],
            "full_trace": int(raw["full_trace"]),
            "predictions": raw["predictions"],
        }
        rows.append(row)
    if [tuple(row["program"]) for row in rows] != programs:
        raise ValueError("E16 evaluator returned programs in the wrong order")
    return rows


def reconstruct_training_coverage(batches: Sequence[Sequence[RegisterExample]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    programs = _exact_programs(manifest)
    states = _exact_states(manifest, "train")
    allowed = {(program, state) for program in programs for state in states}
    counts: Counter[tuple[tuple[str, ...], tuple[int, int]]] = Counter()
    draws = 0
    for batch in batches:
        for example in batch:
            key = (tuple(example.program), (int(example.x), int(example.y)))
            if key not in allowed:
                raise ValueError("training stream contains a program/state outside E16 scope")
            counts[key] += 1
            draws += 1
    per_program = {}
    frequency_rows = []
    for program in programs:
        values = [counts[(program, state)] for state in states]
        frequency_rows.append(values)
        per_program[" ".join(program)] = {
            "draws": int(sum(values)), "unique_states": int(sum(value > 0 for value in values)),
            "min_count": int(min(values)), "max_count": int(max(values)),
            "frequency_digest": canonical_hash(values),
        }
    observed = len(counts)
    return {
        "draws": int(draws), "combinations_exposed": int(observed), "combinations_total": len(allowed),
        "all_6144_exposed": observed == len(allowed), "min_count": int(min(counts.values() or [0])),
        "max_count": int(max(counts.values() or [0])), "frequency_digest": canonical_hash(frequency_rows),
        "per_program": per_program,
    }


def paired_outcomes(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]], *, left: str = "float", right: str = "qat") -> list[dict[str, Any]]:
    if len(left_rows) != len(right_rows):
        raise ValueError("paired rows have different lengths")
    result = []
    for left_row, right_row in zip(left_rows, right_rows):
        if left_row["program"] != right_row["program"] or left_row["n"] != right_row["n"]:
            raise ValueError("paired rows are not aligned")
        lp = left_row.get("predictions", []); rp = right_row.get("predictions", [])
        if [item["state"] for item in lp] != [item["state"] for item in rp]:
            raise ValueError("paired prediction state order differs")
        both = float_only = right_only = neither = 0
        outcomes = []
        for lpred, rpred in zip(lp, rp):
            lok, rok = bool(lpred["joint_final_correct"]), bool(rpred["joint_final_correct"])
            if lok and rok: both += 1
            elif lok: float_only += 1
            elif rok: right_only += 1
            else: neither += 1
            outcomes.append({"state": lpred["state"], "float_correct": lok, "qat_correct": rok,
                             "outcome": "both_correct" if lok and rok else "float_only" if lok else "qat_only" if rok else "neither"})
        n = int(left_row["n"])
        if both + float_only + right_only + neither != n:
            raise ValueError("paired outcomes do not sum to denominator")
        result.append({"program": list(left_row["program"]), "n": n, "both_correct": both,
                       "float_only": float_only, "qat_only": right_only, "neither": neither,
                       "float_correct": both + float_only, "qat_correct": both + right_only,
                       "outcomes": outcomes,
                       "left": left, "right": right})
    return result


def aggregate_metrics(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(train_rows) != 32 or len(validation_rows) != 32:
        raise ValueError("E16 macro aggregation requires all 32 programs in both splits")
    def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = {}
        for length in (1, 2, 3):
            group = [row for row in rows if len(row["program"]) == length]
            if not group:
                raise ValueError("missing program length in E16 rows")
            rates = {name: sum(row[name] / row["n"] for row in group) / len(group)
                     for name in ("final_joint", "final_x", "final_y", "full_trace")}
            counts = {name: sum(row[name] for row in group) / len(group)
                      for name in ("final_joint", "final_x", "final_y", "full_trace")}
            prefixes = []
            for index in range(length):
                eligible = [row for row in group
                            if len(row.get("prefix_joint", [row["final_joint"]] * len(row["program"]))) > index]
                prefix_counts = [int(row.get("prefix_joint", [row["final_joint"]] * len(row["program"]))[index])
                                 for row in eligible]
                prefix_rates = [count / row["n"] for row, count in zip(eligible, prefix_counts)]
                prefixes.append({"index": index + 1, "program_count": len(eligible),
                                 "counts": {"mean": sum(prefix_counts) / len(prefix_counts)},
                                 "rate": sum(prefix_rates) / len(prefix_rates)})
            result[str(length)] = {"program_count": len(group), "rates": rates, "counts": counts,
                                   "prefix_joint": prefixes}
        for name in ("final_joint", "final_x", "final_y", "full_trace"):
            comp = (result["2"]["rates"][name] + result["3"]["rates"][name]) / 2
            result.setdefault("seen_compositions", {"rates": {}, "counts": {}})["rates"][name] = comp
            result["seen_compositions"]["counts"][name] = (result["2"]["counts"][name] + result["3"]["counts"][name]) / 2
        result["seen_compositions"]["prefix_joint"] = []
        for index in range(3):
            available = [result[str(length)]["prefix_joint"][index] for length in (2, 3) if index < length]
            result["seen_compositions"]["prefix_joint"].append({
                "index": index + 1, "program_count": sum(item["program_count"] for item in available),
                "counts": {"mean": sum(item["counts"]["mean"] for item in available) / len(available)},
                "rate": sum(item["rate"] for item in available) / len(available),
            })
        result["primitives"] = result["1"]
        return result
    train = metrics(train_rows); validation = metrics(validation_rows)
    gaps = {group: {name: 100.0 * (train[group]["rates"][name] - validation[group]["rates"][name])
                    for name in ("final_joint", "final_x", "final_y", "full_trace")}
            for group in ("1", "2", "3", "primitives", "seen_compositions")}
    return {"train": train, "validation": validation, "train_minus_validation_pp": gaps}


def _atomic_json(path: Path, value: Mapping[str, Any], *, refuse: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if refuse and path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary checkpoint: {temporary}")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def checkpoint_payload(model: nn.Module, *, arm: str, initial_digest: str, stream_digest: str,
                       prefix_digest: str, e15_manifest_hash: str, source_map: Mapping[str, str],
                       config: Mapping[str, Any] = RUN_CONFIG, optimizer: Mapping[str, Any] = OPTIMIZER_CONFIG,
                       update: int = CHECKPOINT_UPDATE, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown E16 arm")
    if update != CHECKPOINT_UPDATE:
        raise ValueError("E16 final checkpoint must be update2000")
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("E16 checkpoint parameter count mismatch")
    inventory = bitlinear_inventory(model)
    if (arm == "qat" and len(inventory) != 14) or (arm == "float" and inventory):
        raise ValueError("E16 checkpoint module inventory does not match arm")
    cfg = dict(config); opt = dict(optimizer)
    payload = {
        "format_version": 1, "schema": E16_SCHEMA, "arm": arm, "seed": SEED,
        "update": update, "params": sum(p.numel() for p in model.parameters()),
        "state_dict": model.state_dict(), "model_digest": digest_state_dict(model),
        "common_initial_digest": initial_digest, "full_stream_digest": stream_digest,
        "prefix_stream_digest": prefix_digest, "e15_manifest_hash": e15_manifest_hash,
        "source_hashes": dict(source_map), "protocol_hash": protocol_hash(root),
        "config": cfg, "config_hash": canonical_hash(cfg), "optimizer": opt,
        "module_inventory": inventory, "environment": environment(torch.device("cpu")),
        "tag": "final", "fixed_final": True,
    }
    return payload


def load_checkpoint(path: Path, *, arm: str, initial_digest: str, stream_digest: str,
                    e15_manifest_hash: str, source_map: Mapping[str, str],
                    config: Mapping[str, Any] = RUN_CONFIG, root: Path = PROJECT_ROOT) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError("unknown E16 arm")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "schema", "arm", "seed", "update", "state_dict", "model_digest", "common_initial_digest",
                "full_stream_digest", "prefix_stream_digest", "e15_manifest_hash", "source_hashes",
                "protocol_hash", "config", "config_hash", "optimizer", "module_inventory", "tag", "fixed_final"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint metadata missing: {sorted(missing)}")
    checks = ((payload["format_version"], 1, "format version"), (payload["schema"], E16_SCHEMA, "schema"), (payload["arm"], arm, "arm"),
              (payload["seed"], SEED, "seed"), (payload["update"], CHECKPOINT_UPDATE, "update"),
              (payload["common_initial_digest"], initial_digest, "initial digest"),
              (payload["full_stream_digest"], stream_digest, "stream digest"),
              (payload["prefix_stream_digest"], stream_digest, "prefix stream digest"),
              (payload["e15_manifest_hash"], e15_manifest_hash, "manifest hash"),
              (payload["source_hashes"], dict(source_map), "source hashes"),
              (payload["protocol_hash"], protocol_hash(root), "protocol hash"),
              (payload["config"], dict(config), "config"),
              (payload["config_hash"], canonical_hash(config), "config hash"),
              (payload["optimizer"], dict(OPTIMIZER_CONFIG), "optimizer"),
              (payload["tag"], "final", "tag"), (payload["fixed_final"], True, "fixed_final"))
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"checkpoint {label} mismatch")
    model: nn.Module = QATRegisterModel() if arm == "qat" else replace_all_bitlinear(QATRegisterModel())[0]
    if payload.get("params") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("checkpoint parameter count mismatch")
    expected_inventory = bitlinear_inventory(model)
    if payload["module_inventory"] != expected_inventory:
        raise ValueError("checkpoint module inventory mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    if digest_state_dict(model) != payload["model_digest"]:
        raise ValueError("checkpoint state digest mismatch")
    return model, dict(payload)


def verify_protected_hashes(root: Path = PROJECT_ROOT, snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, str]:
    snapshot = Path(snapshot)
    if not snapshot.is_absolute(): snapshot = Path(root) / snapshot
    try:
        payload = json.loads(snapshot.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read protected snapshot: {snapshot}") from exc
    hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 101:
        raise ValueError("E16 protected snapshot must contain exactly 101 files")
    for relative, expected in hashes.items():
        path = Path(relative)
        if not path.is_absolute(): path = Path(root) / path
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"protected hash mismatch: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def refuse_nonempty(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty E16 output: {path}")


def make_manifest(*, e15_manifest: Mapping[str, Any], initial_digest: str, replaced: Sequence[str],
                  batches: Sequence[Sequence[RegisterExample]], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if manifest_hash(e15_manifest) != EXPECTED_MANIFEST_HASH:
        raise ValueError("cannot build E16 manifest from non-V7 E15 source")
    return {
        "format_version": 1, "schema": E16_SCHEMA, "experiment": "float_qat_e16",
        "e15_manifest_hash": manifest_hash(e15_manifest), "e15_manifest_path": str(DEFAULT_E15_PREFLIGHT / "manifest.json"),
        "source_hashes": source_hashes(root), "protocol_hash": protocol_hash(root),
        "config": dict(RUN_CONFIG), "config_hash": canonical_hash(RUN_CONFIG),
        "optimizer": dict(OPTIMIZER_CONFIG), "initial_digest": initial_digest,
        "replaced_bitlinear": list(replaced), "qat_bitlinear": list(replaced), "float_bitlinear": [],
        "parameter_count": EXPECTED_PARAMETER_COUNT, "seed": SEED, "update": CHECKPOINT_UPDATE,
        "stream_digest": batch_digest(batches), "target_digest": target_digest(batches),
        "schedule": {"updates": UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
                     "counts": {"1": 666, "2": 668, "3": 666}},
        "programs": e15_manifest["programs"], "state_split": e15_manifest["state_split"],
    }


__all__ = [
    "ARMS", "BATCH_SIZE", "CHECKPOINT_UPDATE", "DEFAULT_E15_PREFLIGHT", "DEFAULT_PROTECTED_SNAPSHOT",
    "E16_SCHEMA", "E16_SOURCE_RELATIVE_PATHS", "EXPECTED_BATCH_DIGEST", "EXPECTED_MANIFEST_HASH", "EXPECTED_TARGET_DIGEST",
    "OPTIMIZER_CONFIG", "PROJECT_ROOT", "PROGRESS_INTERVAL", "RUN_CONFIG", "UPDATES", "aggregate_metrics",
    "assert_paired_models", "atomic_torch_save", "bitlinear_inventory", "build_paired_models", "canonical_hash",
    "batch_digest", "checkpoint_payload", "digest_state_dict", "evaluate_split", "fixed_stream", "load_checkpoint", "load_frozen_manifest",
    "make_manifest", "manifest_hash", "paired_outcomes", "refuse_nonempty", "replace_all_bitlinear",
    "reconstruct_training_coverage", "source_hashes", "target_digest", "verify_protected_hashes",
]
