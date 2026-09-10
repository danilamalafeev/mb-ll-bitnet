"""E17 learned value embeddings versus explicit signed four-bit input.

The E17 model is built from the frozen E16 float model.  Only the x/y input
value tables differ; all recurrent, reader, FFN, normalization, and output
parameters remain the same.  Importing this module performs no training or
inference.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .float_qat_e16 import (
    BATCH_SIZE,
    CHECKPOINT_UPDATE,
    DEFAULT_E15_PREFLIGHT,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    PROGRESS_INTERVAL,
    RUN_CONFIG as E16_RUN_CONFIG,
    UPDATES,
    aggregate_metrics,
    atomic_torch_save,
    bitlinear_inventory,
    build_paired_models as build_e16_models,
    canonical_hash,
    digest_state_dict,
    evaluate_split,
    fixed_stream as fixed_e16_stream,
    load_frozen_manifest,
    manifest_hash,
    reconstruct_training_coverage,
    refuse_nonempty,
    source_hashes as e16_source_hashes,
    target_digest,
)
from .register_e15 import RegisterExample, loss_for_batch, sha256_file
from .runtime import environment


E17_SCHEMA = "e17_bit_input_v1"
ARMS = ("learned", "bits")
SEED = 0
LEARNED_PARAMETER_COUNT = 152768
BITS_PARAMETER_COUNT = 151232
INPUT_TABLE_PARAMETER_COUNT = 2 * 16 * 64
INPUT_PROJECTION_PARAMETER_COUNT = 2 * 4 * 64
DEFAULT_E17_PREFLIGHT = Path("runs/e17_bit_input_preflight")
DEFAULT_E17_RUN = Path("runs/e17_bit_input")
DEFAULT_PROTECTED_SNAPSHOT = Path("results/E17_PROTECTED_HASHES.json")
E17_SOURCE_RELATIVE_PATHS = {
    "e17_module": "looped_bitnet/bit_input_e17.py",
    "e17_runner": "scripts/bit_input_e17.py",
    "e17_tests": "tests/test_bit_input_e17.py",
    "e17_protocol": "results/E17_BIT_INPUT_PROTOCOL.md",
    "e16_module": "looped_bitnet/float_qat_e16.py",
    "e16_runner": "scripts/float_qat_e16.py",
    "e16_tests": "tests/test_float_qat_e16.py",
    "e16_protocol": "results/E16_FLOAT_QAT_PROTOCOL.md",
    "e15_module": "looped_bitnet/register_e15.py",
    "e15_runner": "scripts/register_interpreter_e15.py",
    "e15_tests": "tests/test_register_e15.py",
    "e15_protocol": "results/E15_REGISTER_IMPLEMENTATION_PROTOCOL.md",
    "e15_semantic_audit": "results/E15_REGISTER_DSL_SEMANTIC_AUDIT.json",
    "e15_semantic_audit_script": "scripts/register_dsl_e15_audit.py",
    "model": "looped_bitnet/model.py",
    "quantization": "looped_bitnet/quantization.py",
    "runtime": "looped_bitnet/runtime.py",
    "data": "looped_bitnet/data.py",
    "config": "looped_bitnet/config.py",
    "engine": "looped_bitnet/engine.py",
}
E16_CONFIG = dict(E16_RUN_CONFIG)
RUN_CONFIG: dict[str, Any] = {
    **E16_CONFIG,
    "schema": E17_SCHEMA,
    "encoder_mode": "learned_or_bits",
    "input_bit_order": "lsb_first",
    "input_centering": "2*b-1",
    "input_projection_shape": [64, 4],
    "input_projection_bias": False,
    "input_projection_std": 0.01,
    "input_projection_generator": "local_cpu_generator_seed0_x_then_y",
}
ENCODER_SPEC: dict[str, Any] = {
    "bit_order": "least_significant_bit_first",
    "formula": "signed=2*((value>>i)&1)-1",
    "values": list(range(16)),
    "signed_dtype": "float32",
    "projection": {"in_features": 4, "out_features": 64, "bias": False,
                    "init": "normal", "std": 0.01, "generator": "local_cpu", "seed": 0,
                    "draw_order": ["x", "y"]},
}
SOURCE_DEPENDENCY_RELATIVE_PATHS = {
    name: E17_SOURCE_RELATIVE_PATHS[name]
    for name in E17_SOURCE_RELATIVE_PATHS
    if name not in {"e17_module", "e17_runner", "e17_tests", "e17_protocol"}
}


def protocol_hash(root: Path = PROJECT_ROOT) -> str:
    return sha256_file(Path(root) / E17_SOURCE_RELATIVE_PATHS["e17_protocol"])


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / relative) for name, relative in E17_SOURCE_RELATIVE_PATHS.items()}


def digest_named_state(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(cpu.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def signed_bits(value: Tensor) -> Tensor:
    """Encode integer values 0..15 as signed LSB-first float32 bits."""
    if value.dtype != torch.long:
        raise ValueError("register values must be torch.long")
    if torch.any((value < 0) | (value > 15)):
        raise ValueError("register values must be in 0..15")
    shifts = torch.arange(4, dtype=torch.long, device=value.device)
    return ((((value.unsqueeze(-1) >> shifts) & 1) * 2) - 1).to(torch.float32)


class SignedBitsEncoder(nn.Module):
    """One independent four-bit signed encoder with no trainable codebook."""

    def __init__(self, weight: Tensor):
        super().__init__()
        if tuple(weight.shape) != (64, 4) or weight.dtype != torch.float32:
            raise ValueError("signed-bit projection weight must have shape (64,4), float32")
        # Linear's constructor normally draws from the global RNG.  Isolate
        # that irrelevant draw so E16 initialization and stream RNG remain
        # unchanged, then install the explicitly local-generator weight.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(4, 64, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.projection.weight.copy_(weight)

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.projection(signed_bits(value))


def projection_weights() -> tuple[Tensor, Tensor]:
    """Draw x then y projection weights from the required local CPU generator."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    x = torch.randn((64, 4), generator=generator, dtype=torch.float32) * 0.01
    y = torch.randn((64, 4), generator=generator, dtype=torch.float32) * 0.01
    return x, y


def replace_input_embeddings(learned: nn.Module) -> tuple[nn.Module, tuple[Tensor, Tensor]]:
    """Clone E16 float and replace only x/y learned tables by bit encoders."""
    if bitlinear_inventory(learned):
        raise ValueError("E17 learned arm must be the E16 float model")
    bits = deepcopy(learned)
    x_weight, y_weight = projection_weights()
    bits.x_embedding = SignedBitsEncoder(x_weight)
    bits.y_embedding = SignedBitsEncoder(y_weight)
    return bits, (x_weight, y_weight)


def _common_state(model: nn.Module) -> dict[str, Tensor]:
    excluded = ("x_embedding", "y_embedding")
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith(excluded)}


def common_core_digest(model: nn.Module) -> str:
    return digest_named_state(_common_state(model))


def assert_paired_models(learned: nn.Module, bits: nn.Module) -> None:
    if bitlinear_inventory(learned) or bitlinear_inventory(bits):
        raise ValueError("E17 models must contain zero BitLinear modules")
    learned_count = sum(p.numel() for p in learned.parameters())
    bits_count = sum(p.numel() for p in bits.parameters())
    if learned_count != LEARNED_PARAMETER_COUNT or bits_count != BITS_PARAMETER_COUNT:
        raise ValueError(f"E17 parameter counts mismatch: learned={learned_count}, bits={bits_count}")
    left, right = _common_state(learned), _common_state(bits)
    if list(left) != list(right) or set(left) != set(right):
        raise ValueError("E17 common state keys differ")
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype or not torch.equal(left[name], right[name]):
            raise ValueError(f"E17 common state mismatch: {name}")
        if left[name].data_ptr() == right[name].data_ptr():
            raise ValueError(f"E17 common state storage is shared: {name}")
    if not isinstance(bits.x_embedding, SignedBitsEncoder) or not isinstance(bits.y_embedding, SignedBitsEncoder):
        raise ValueError("bits arm does not contain the two signed encoders")
    for module in (bits.x_embedding, bits.y_embedding):
        if module.projection.bias is not None or tuple(module.projection.weight.shape) != (64, 4):
            raise ValueError("invalid E17 projection")


def build_paired_models(seed: int = SEED) -> tuple[nn.Module, nn.Module, str, str, str, tuple[Tensor, Tensor]]:
    if seed != SEED:
        raise ValueError("E17 is registered for seed0 only")
    _, _, learned, _, _ = build_e16_models(seed)
    bits, weights = replace_input_embeddings(learned)
    assert_paired_models(learned, bits)
    return learned, bits, digest_state_dict(learned), digest_state_dict(bits), common_core_digest(learned), weights


def fixed_stream(seed: int = SEED) -> list[list[RegisterExample]]:
    return fixed_e16_stream(seed)


def paired_outcomes(bits_rows: Sequence[Mapping[str, Any]], learned_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Relabel E16-style per-state predictions with E17 semantic arm names."""
    if len(bits_rows) != len(learned_rows):
        raise ValueError("paired E17 rows have different lengths")
    result = []
    for bits_row, learned_row in zip(bits_rows, learned_rows):
        if bits_row["program"] != learned_row["program"] or bits_row["n"] != learned_row["n"]:
            raise ValueError("paired E17 rows are not aligned")
        bp = bits_row.get("predictions", []); lp = learned_row.get("predictions", [])
        if [item["state"] for item in bp] != [item["state"] for item in lp]:
            raise ValueError("paired E17 state order differs")
        both = bits_only = learned_only = neither = 0; outcomes = []
        for bit_pred, learned_pred in zip(bp, lp):
            bit_ok = bool(bit_pred["joint_final_correct"]); learned_ok = bool(learned_pred["joint_final_correct"])
            if bit_ok and learned_ok: both += 1; label = "both_correct"
            elif bit_ok: bits_only += 1; label = "bits_only"
            elif learned_ok: learned_only += 1; label = "learned_only"
            else: neither += 1; label = "neither"
            outcomes.append({"state": bit_pred["state"], "bits_correct": bit_ok,
                             "learned_correct": learned_ok, "outcome": label})
        n = int(bits_row["n"])
        if both + bits_only + learned_only + neither != n:
            raise ValueError("paired E17 outcomes do not sum to denominator")
        result.append({"program": list(bits_row["program"]), "n": n, "both_correct": both,
                       "bits_only": bits_only, "learned_only": learned_only, "neither": neither,
                       "bits_correct": both + bits_only, "learned_correct": both + learned_only,
                       "outcomes": outcomes, "bits_arm": "bits", "learned_arm": "learned"})
    return result


def load_manifest(path: Path = DEFAULT_E15_PREFLIGHT, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_frozen_manifest(path, root=root)


def make_manifest(*, e15_manifest: Mapping[str, Any], learned_initial_digest: str,
                  bits_initial_digest: str, common_digest: str,
                  batches: Sequence[Sequence[RegisterExample]], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if manifest_hash(e15_manifest) != EXPECTED_MANIFEST_HASH:
        raise ValueError("E17 requires the frozen V7 E15 manifest")
    return {
        "format_version": 1, "schema": E17_SCHEMA, "experiment": "bit_input_e17",
        "e15_manifest_hash": EXPECTED_MANIFEST_HASH,
        "e15_manifest_path": str(DEFAULT_E15_PREFLIGHT / "manifest.json"),
        "source_hashes": source_hashes(root), "dependency_hashes": {name: source_hashes(root)[name] for name in SOURCE_DEPENDENCY_RELATIVE_PATHS},
        "protocol_hash": protocol_hash(root), "config": dict(RUN_CONFIG), "config_hash": canonical_hash(RUN_CONFIG),
        "optimizer": dict(OPTIMIZER_CONFIG), "seed": SEED, "update": CHECKPOINT_UPDATE,
        "encoder_spec": dict(ENCODER_SPEC), "learned_initial_digest": learned_initial_digest,
        "bits_initial_digest": bits_initial_digest, "common_core_digest": common_digest,
        "learned_parameter_count": LEARNED_PARAMETER_COUNT, "bits_parameter_count": BITS_PARAMETER_COUNT,
        "stream_digest": EXPECTED_BATCH_DIGEST, "target_digest": target_digest(batches),
        "schedule": {"updates": UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
                     "counts": {"1": 666, "2": 668, "3": 666}},
        "programs": e15_manifest["programs"], "state_split": e15_manifest["state_split"],
    }


def checkpoint_payload(model: nn.Module, *, arm: str, initial_digest: str, common_digest: str,
                       stream_digest: str, target_stream_digest: str, e15_manifest_hash: str,
                       source_map: Mapping[str, str], config: Mapping[str, Any] = RUN_CONFIG,
                       optimizer: Mapping[str, Any] = OPTIMIZER_CONFIG, update: int = CHECKPOINT_UPDATE,
                       root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown E17 arm")
    expected = LEARNED_PARAMETER_COUNT if arm == "learned" else BITS_PARAMETER_COUNT
    params = sum(p.numel() for p in model.parameters())
    if params != expected or update != CHECKPOINT_UPDATE:
        raise ValueError("E17 checkpoint count or fixed update mismatch")
    if not isinstance(model.x_embedding, nn.Embedding) and arm == "learned":
        raise ValueError("learned checkpoint has wrong encoder mode")
    if arm == "bits" and (not isinstance(model.x_embedding, SignedBitsEncoder) or not isinstance(model.y_embedding, SignedBitsEncoder)):
        raise ValueError("bits checkpoint has wrong encoder mode")
    return {
        "format_version": 1, "schema": E17_SCHEMA, "arm": arm, "encoder_mode": arm,
        "seed": SEED, "update": update, "params": params, "state_dict": model.state_dict(),
        "model_digest": digest_state_dict(model), "initial_digest": initial_digest,
        "common_core_digest": common_digest, "full_stream_digest": stream_digest,
        "prefix_stream_digest": stream_digest, "target_digest": target_stream_digest,
        "e15_manifest_hash": e15_manifest_hash, "source_hashes": dict(source_map),
        "protocol_hash": protocol_hash(root), "config": dict(config), "config_hash": canonical_hash(config),
        "optimizer": dict(optimizer), "encoder_spec": dict(ENCODER_SPEC),
        "environment": environment(torch.device("cpu")),
        "learned_parameter_count": LEARNED_PARAMETER_COUNT, "bits_parameter_count": BITS_PARAMETER_COUNT,
        "module_inventory": {"x": type(model.x_embedding).__name__, "y": type(model.y_embedding).__name__,
                             "bitlinear": bitlinear_inventory(model)},
        "tag": "final", "fixed_final": True,
    }


def load_checkpoint(path: Path, *, arm: str, initial_digest: str, common_digest: str,
                    stream_digest: str, target_stream_digest: str, e15_manifest_hash: str,
                    source_map: Mapping[str, str], config: Mapping[str, Any] = RUN_CONFIG,
                    root: Path = PROJECT_ROOT) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError("unknown E17 arm")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "schema", "arm", "encoder_mode", "seed", "update", "params", "state_dict",
                "model_digest", "initial_digest", "common_core_digest", "full_stream_digest", "prefix_stream_digest",
                "target_digest", "e15_manifest_hash", "source_hashes", "protocol_hash", "config", "config_hash",
                "optimizer", "encoder_spec", "learned_parameter_count", "bits_parameter_count", "module_inventory",
                "environment", "tag", "fixed_final"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint metadata missing: {sorted(missing)}")
    checks = (
        (payload["format_version"], 1, "format version"), (payload["schema"], E17_SCHEMA, "schema"),
        (payload["arm"], arm, "arm"), (payload["encoder_mode"], arm, "encoder mode"),
        (payload["seed"], SEED, "seed"), (payload["update"], CHECKPOINT_UPDATE, "update"),
        (payload["initial_digest"], initial_digest, "initial digest"), (payload["common_core_digest"], common_digest, "common digest"),
        (payload["full_stream_digest"], stream_digest, "stream digest"), (payload["prefix_stream_digest"], stream_digest, "prefix stream digest"),
        (payload["target_digest"], target_stream_digest, "target digest"), (payload["e15_manifest_hash"], e15_manifest_hash, "manifest hash"),
        (payload["source_hashes"], dict(source_map), "source hashes"), (payload["protocol_hash"], protocol_hash(root), "protocol hash"),
        (payload["config"], dict(config), "config"), (payload["config_hash"], canonical_hash(config), "config hash"),
        (payload["optimizer"], dict(OPTIMIZER_CONFIG), "optimizer"), (payload["encoder_spec"], dict(ENCODER_SPEC), "encoder spec"),
        (payload["tag"], "final", "tag"), (payload["fixed_final"], True, "fixed final"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"checkpoint {label} mismatch")
    learned, bits, _, _, built_common, _ = build_paired_models()
    if built_common != common_digest:
        raise ValueError("checkpoint common digest cannot be reconstructed")
    model = learned if arm == "learned" else bits
    if digest_state_dict(model) != initial_digest:
        raise ValueError("checkpoint initial digest cannot be reconstructed")
    if common_core_digest(model) != common_digest:
        raise ValueError("checkpoint common initial digest cannot be reconstructed")
    expected_params = LEARNED_PARAMETER_COUNT if arm == "learned" else BITS_PARAMETER_COUNT
    if payload["params"] != expected_params or payload["learned_parameter_count"] != LEARNED_PARAMETER_COUNT or payload["bits_parameter_count"] != BITS_PARAMETER_COUNT:
        raise ValueError("checkpoint parameter metadata mismatch")
    expected_inventory = {"x": type(model.x_embedding).__name__, "y": type(model.y_embedding).__name__,
                          "bitlinear": bitlinear_inventory(model)}
    if payload["module_inventory"] != expected_inventory:
        raise ValueError("checkpoint encoder inventory mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    if digest_state_dict(model) != payload["model_digest"]:
        raise ValueError("checkpoint model digest mismatch")
    return model, dict(payload)


def verify_protected_hashes(root: Path = PROJECT_ROOT, snapshot: Path = DEFAULT_PROTECTED_SNAPSHOT) -> dict[str, str]:
    snapshot = Path(snapshot)
    if not snapshot.is_absolute(): snapshot = Path(root) / snapshot
    try:
        payload = json.loads(snapshot.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read E17 protected snapshot: {snapshot}") from exc
    hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 118:
        raise ValueError("E17 protected snapshot must contain exactly 118 files")
    for relative, expected in hashes.items():
        path = Path(relative) if Path(relative).is_absolute() else Path(root) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"protected hash mismatch: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


__all__ = [
    "ARMS", "BITS_PARAMETER_COUNT", "CHECKPOINT_UPDATE", "DEFAULT_E17_PREFLIGHT", "DEFAULT_E17_RUN",
    "DEFAULT_PROTECTED_SNAPSHOT", "ENCODER_SPEC", "E17_SCHEMA", "E17_SOURCE_RELATIVE_PATHS",
    "EXPECTED_BATCH_DIGEST", "EXPECTED_MANIFEST_HASH", "EXPECTED_TARGET_DIGEST", "LEARNED_PARAMETER_COUNT",
    "OPTIMIZER_CONFIG", "PROJECT_ROOT", "RUN_CONFIG", "SignedBitsEncoder", "aggregate_metrics", "assert_paired_models",
    "build_paired_models", "checkpoint_payload", "common_core_digest", "digest_named_state", "digest_state_dict",
    "evaluate_split", "fixed_stream", "load_checkpoint", "load_frozen_manifest", "load_manifest", "make_manifest",
    "paired_outcomes", "projection_weights", "reconstruct_training_coverage", "replace_input_embeddings", "signed_bits",
    "source_hashes", "target_digest", "verify_protected_hashes", "atomic_torch_save", "refuse_nonempty",
]
