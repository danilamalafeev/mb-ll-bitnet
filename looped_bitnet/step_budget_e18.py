"""E18 native four versus eight recurrent substeps on the E17 bits core."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .bit_input_e17 import (
    ARMS as E17_ARMS,
    DEFAULT_E17_PREFLIGHT,
    ENCODER_SPEC,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    RUN_CONFIG as E17_RUN_CONFIG,
    SignedBitsEncoder,
    build_paired_models as build_e17_models,
    common_core_digest,
    digest_state_dict,
    evaluate_split,
    fixed_stream as fixed_e17_stream,
    reconstruct_training_coverage,
    source_hashes as e17_source_hashes,
    target_digest,
    verify_protected_hashes as verify_e17_protected_hashes,
)
from .float_qat_e16 import (
    BATCH_SIZE,
    CHECKPOINT_UPDATE,
    PROGRESS_INTERVAL,
    aggregate_metrics,
    atomic_torch_save,
    canonical_hash,
    refuse_nonempty,
)
from .register_e15 import RegisterExample, QATRegisterModel, batch_digest, loss_for_batch, sha256_file
from .runtime import environment, seed_everything


E18_SCHEMA = "e18_step_budget_v1"
ARMS = ("steps4", "steps8")
NATIVE_STEPS = {"steps4": 4, "steps8": 8}
SEED = 0
UPDATES = 2000
LEARNED_PARAMETER_COUNT = 151232
DEFAULT_E18_PREFLIGHT = Path("runs/e18_step_budget_preflight")
DEFAULT_E18_RUN = Path("runs/e18_step_budget")
DEFAULT_PROTECTED_SNAPSHOT = Path("results/E18_PROTECTED_HASHES.json")
E17_BITS_INITIAL_STATE = DEFAULT_E17_PREFLIGHT / "bits_initial_state.pt"

E18_SOURCE_RELATIVE_PATHS = {
    "e18_module": "looped_bitnet/step_budget_e18.py",
    "e18_runner": "scripts/step_budget_e18.py",
    "e18_tests": "tests/test_step_budget_e18.py",
    "e18_protocol": "results/E18_STEP_BUDGET_PROTOCOL.md",
    "e17_module": "looped_bitnet/bit_input_e17.py",
    "e17_runner": "scripts/bit_input_e17.py",
    "e17_tests": "tests/test_bit_input_e17.py",
    "e17_protocol": "results/E17_BIT_INPUT_PROTOCOL.md",
    "e16_module": "looped_bitnet/float_qat_e16.py",
    "e15_module": "looped_bitnet/register_e15.py",
    "model": "looped_bitnet/model.py",
    "quantization": "looped_bitnet/quantization.py",
    "runtime": "looped_bitnet/runtime.py",
}
RUN_CONFIG: dict[str, Any] = {
    **dict(E17_RUN_CONFIG),
    "schema": E18_SCHEMA,
    "native_budgets": [4, 8],
    "encoder_mode": "bits",
    "substeps_per_instruction": "native_arm_budget",
    "updates": UPDATES,
    "batch_size": BATCH_SIZE,
}


def protocol_hash(root: Path = PROJECT_ROOT) -> str:
    return sha256_file(Path(root) / E18_SOURCE_RELATIVE_PATHS["e18_protocol"])


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / relative) for name, relative in E18_SOURCE_RELATIVE_PATHS.items()}


def e17_reference_hash(root: Path = PROJECT_ROOT) -> str:
    return sha256_file(Path(root) / E17_BITS_INITIAL_STATE)


class NativeStepRegisterModel(QATRegisterModel):
    """E17 bits model with only the native recurrent substep budget changed."""

    native_steps: int

    @classmethod
    def from_e17_bits(cls, base: nn.Module, native_steps: int) -> "NativeStepRegisterModel":
        if native_steps not in (4, 8):
            raise ValueError("E18 native budget must be 4 or 8")
        if not isinstance(base.x_embedding, SignedBitsEncoder) or not isinstance(base.y_embedding, SignedBitsEncoder):
            raise ValueError("E18 base must be the E17 bits model")
        # QATRegisterModel and this subclass have identical nn.Module layout;
        # changing the Python class preserves every state key and module object.
        base.__class__ = cls
        base.native_steps = native_steps
        return base  # type: ignore[return-value]

    def __init__(self, native_steps: int = 4):
        # Public construction is useful for tiny controls.  The factory below
        # supplies the E17 bits state and reference digest for actual runs.
        learned, bits, *_ = build_e17_models()
        base = self.from_e17_bits(bits, native_steps)
        self.__dict__.update(base.__dict__)
        self.native_steps = native_steps

    def step(self, cache: dict[str, Any], opcode: int | Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, Any]]:
        if isinstance(opcode, int):
            opcode = torch.full((cache["h"].shape[0],), opcode, dtype=torch.long, device=cache["h"].device)
        if opcode.ndim != 1 or opcode.shape[0] != cache["h"].shape[0] or torch.any((opcode < 0) | (opcode >= 3)):
            raise ValueError("opcode must be a batch-sized value in 0..2")
        h = cache["h"]
        with torch.autocast(device_type=h.device.type, enabled=False):
            for inner in range(self.native_steps):
                h = h + self.opcode_embedding(opcode) / (64 ** 0.5)
                h = h + self.reader(h, cache["kv"])
                h = self.blocks[(cache["substeps"] + inner) % 4](h)
            cache = dict(cache, h=h, substeps=cache["substeps"] + self.native_steps)
            logits = self.output(self.output_norm(h))
        return logits, cache


def _load_e17_reference(root: Path = PROJECT_ROOT) -> dict[str, Tensor]:
    path = Path(root) / E17_BITS_INITIAL_STATE
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("arm") != "bits" or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("E17 bits initial reference is malformed")
    return payload["state_dict"]


def build_paired_models(seed: int = SEED, *, root: Path = PROJECT_ROOT) -> tuple[NativeStepRegisterModel, NativeStepRegisterModel, str, str]:
    if seed != SEED:
        raise ValueError("E18 is registered for seed0 only")
    _, e17_bits, _, bits_digest, common_digest, _ = build_e17_models(seed)
    reference = _load_e17_reference(root)
    e17_bits.load_state_dict(reference, strict=True)
    if digest_state_dict(e17_bits) != bits_digest:
        raise ValueError("E17 saved bits initial reference digest mismatch")
    steps4 = NativeStepRegisterModel.from_e17_bits(deepcopy(e17_bits), 4)
    steps8 = NativeStepRegisterModel.from_e17_bits(deepcopy(e17_bits), 8)
    assert_paired_models(steps4, steps8, expected_common_digest=common_digest)
    return steps4, steps8, bits_digest, common_digest


def assert_paired_models(steps4: nn.Module, steps8: nn.Module, *, expected_common_digest: str | None = None) -> None:
    if not isinstance(steps4, NativeStepRegisterModel) or not isinstance(steps8, NativeStepRegisterModel):
        raise ValueError("E18 models must use NativeStepRegisterModel")
    if steps4.native_steps != 4 or steps8.native_steps != 8:
        raise ValueError("E18 native budgets are wrong")
    for model in (steps4, steps8):
        if sum(parameter.numel() for parameter in model.parameters()) != LEARNED_PARAMETER_COUNT:
            raise ValueError("E18 parameter count mismatch")
        if any(module.__class__.__name__ == "BitLinear" for module in model.modules()):
            raise ValueError("E18 contains BitLinear")
        if not isinstance(model.x_embedding, SignedBitsEncoder) or not isinstance(model.y_embedding, SignedBitsEncoder):
            raise ValueError("E18 encoder inventory mismatch")
    left, right = steps4.state_dict(), steps8.state_dict()
    if list(left) != list(right):
        raise ValueError("E18 state keys differ")
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype or not torch.equal(left[name], right[name]):
            raise ValueError(f"E18 initial tensor mismatch: {name}")
        if left[name].data_ptr() == right[name].data_ptr():
            raise ValueError(f"E18 initial storage is shared: {name}")
    common = common_core_digest(steps4)
    if common != common_core_digest(steps8) or (expected_common_digest is not None and common != expected_common_digest):
        raise ValueError("E18 common initial digest mismatch")


def fixed_stream(seed: int = SEED) -> list[list[RegisterExample]]:
    return fixed_e17_stream(seed)


def paired_outcomes(steps8_rows: Sequence[Mapping[str, Any]], steps4_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(steps8_rows) != len(steps4_rows):
        raise ValueError("E18 paired rows have different lengths")
    result = []
    for eight, four in zip(steps8_rows, steps4_rows):
        if eight["program"] != four["program"] or eight["n"] != four["n"]:
            raise ValueError("E18 paired rows are not aligned")
        ep, fp = eight.get("predictions", []), four.get("predictions", [])
        if [item["state"] for item in ep] != [item["state"] for item in fp]:
            raise ValueError("E18 state order differs")
        both = eight_only = four_only = neither = 0; outcomes = []
        for e, f in zip(ep, fp):
            eok, fok = bool(e["joint_final_correct"]), bool(f["joint_final_correct"])
            if eok and fok: both += 1; label = "both_correct"
            elif eok: eight_only += 1; label = "steps8_only"
            elif fok: four_only += 1; label = "steps4_only"
            else: neither += 1; label = "neither"
            outcomes.append({"state": e["state"], "steps8_correct": eok, "steps4_correct": fok, "outcome": label})
        n = int(eight["n"])
        if both + eight_only + four_only + neither != n:
            raise ValueError("E18 paired counts do not sum to denominator")
        result.append({"program": list(eight["program"]), "n": n, "both_correct": both,
                       "steps8_only": eight_only, "steps4_only": four_only, "neither": neither,
                       "steps8_correct": both + eight_only, "steps4_correct": both + four_only,
                       "outcomes": outcomes, "steps8_arm": "steps8", "steps4_arm": "steps4"})
    return result


def make_manifest(*, e15_manifest: Mapping[str, Any], bits_digest: str, common_digest: str,
                  batches: Sequence[Sequence[RegisterExample]], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    from .bit_input_e17 import EXPECTED_MANIFEST_HASH
    if canonical_hash(e15_manifest) != EXPECTED_MANIFEST_HASH:
        raise ValueError("E18 requires frozen V7 E15 manifest")
    return {
        "format_version": 1, "schema": E18_SCHEMA, "experiment": "step_budget_e18",
        "e15_manifest_hash": EXPECTED_MANIFEST_HASH, "source_hashes": source_hashes(root),
        "protocol_hash": protocol_hash(root), "config": dict(RUN_CONFIG), "config_hash": canonical_hash(RUN_CONFIG),
        "optimizer": dict(OPTIMIZER_CONFIG), "seed": SEED, "update": CHECKPOINT_UPDATE,
        "encoder_spec": dict(ENCODER_SPEC), "encoder_mode": "bits", "initial_digest": bits_digest,
        "common_core_digest": common_digest, "parameter_count": LEARNED_PARAMETER_COUNT,
        "e17_reference_sha256": e17_reference_hash(root),
        "stream_digest": EXPECTED_BATCH_DIGEST, "target_digest": target_digest(batches),
        "schedule": {"updates": UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
                     "counts": {"1": 666, "2": 668, "3": 666}},
        "budgets": {"steps4": {"native_steps": 4, "internal_state_updates": 1024000},
                    "steps8": {"native_steps": 8, "internal_state_updates": 2048000},
                    "pair_internal_state_updates": 3072000},
        "programs": e15_manifest["programs"], "state_split": e15_manifest["state_split"],
    }


def checkpoint_payload(model: NativeStepRegisterModel, *, arm: str, initial_digest: str, common_digest: str,
                       stream_digest: str, target_stream_digest: str, e15_manifest_hash: str,
                       source_map: Mapping[str, str], config: Mapping[str, Any] = RUN_CONFIG,
                       optimizer: Mapping[str, Any] = OPTIMIZER_CONFIG, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if arm not in ARMS or model.native_steps != NATIVE_STEPS[arm]:
        raise ValueError("E18 arm or native budget mismatch")
    if sum(parameter.numel() for parameter in model.parameters()) != LEARNED_PARAMETER_COUNT:
        raise ValueError("E18 checkpoint parameter count mismatch")
    return {
        "format_version": 1, "schema": E18_SCHEMA, "arm": arm, "native_steps": model.native_steps,
        "encoder_mode": "bits", "seed": SEED, "update": CHECKPOINT_UPDATE,
        "params": LEARNED_PARAMETER_COUNT, "state_dict": model.state_dict(), "model_digest": digest_state_dict(model),
        "initial_digest": initial_digest, "common_core_digest": common_digest,
        "full_stream_digest": stream_digest, "prefix_stream_digest": stream_digest,
        "target_digest": target_stream_digest, "e15_manifest_hash": e15_manifest_hash,
        "e17_reference_sha256": e17_reference_hash(root),
        "source_hashes": dict(source_map), "protocol_hash": protocol_hash(root),
        "config": dict(config), "config_hash": canonical_hash(config), "optimizer": dict(optimizer),
        "encoder_spec": dict(ENCODER_SPEC), "parameter_count": LEARNED_PARAMETER_COUNT,
        "internal_state_updates_per_arm": UPDATES * BATCH_SIZE * model.native_steps * 2,
        "environment": environment(torch.device("cpu")), "module_inventory": {
            "x": type(model.x_embedding).__name__, "y": type(model.y_embedding).__name__, "bitlinear": [],
            "native_steps": model.native_steps},
        "tag": "final", "fixed_final": True,
    }


def load_checkpoint(path: Path, *, arm: str, initial_digest: str, common_digest: str,
                    stream_digest: str, target_stream_digest: str, e15_manifest_hash: str,
                    source_map: Mapping[str, str], config: Mapping[str, Any] = RUN_CONFIG,
                    root: Path = PROJECT_ROOT) -> tuple[NativeStepRegisterModel, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError("unknown E18 arm")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "schema", "arm", "native_steps", "encoder_mode", "seed", "update", "params", "state_dict",
                "model_digest", "initial_digest", "common_core_digest", "full_stream_digest", "prefix_stream_digest",
                "target_digest", "e15_manifest_hash", "source_hashes", "protocol_hash", "config", "config_hash",
                "optimizer", "encoder_spec", "parameter_count", "internal_state_updates_per_arm", "environment",
                "e17_reference_sha256", "module_inventory", "tag", "fixed_final"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint metadata missing: {sorted(missing)}")
    checks = ((payload["format_version"], 1, "format version"), (payload["schema"], E18_SCHEMA, "schema"),
              (payload["arm"], arm, "arm"), (payload["native_steps"], NATIVE_STEPS[arm], "native budget"),
              (payload["encoder_mode"], "bits", "encoder mode"), (payload["seed"], SEED, "seed"),
              (payload["update"], CHECKPOINT_UPDATE, "update"), (payload["params"], LEARNED_PARAMETER_COUNT, "params"),
              (payload["initial_digest"], initial_digest, "initial digest"), (payload["common_core_digest"], common_digest, "common digest"),
              (payload["full_stream_digest"], stream_digest, "stream digest"), (payload["prefix_stream_digest"], stream_digest, "prefix stream digest"),
              (payload["target_digest"], target_stream_digest, "target digest"), (payload["e15_manifest_hash"], e15_manifest_hash, "manifest hash"),
              (payload["e17_reference_sha256"], e17_reference_hash(root), "E17 reference"),
              (payload["source_hashes"], dict(source_map), "source hashes"), (payload["protocol_hash"], protocol_hash(root), "protocol hash"),
              (payload["config"], dict(config), "config"), (payload["config_hash"], canonical_hash(config), "config hash"),
              (payload["optimizer"], dict(OPTIMIZER_CONFIG), "optimizer"), (payload["encoder_spec"], dict(ENCODER_SPEC), "encoder spec"),
              (payload["parameter_count"], LEARNED_PARAMETER_COUNT, "parameter count"),
              (payload["internal_state_updates_per_arm"], UPDATES * BATCH_SIZE * NATIVE_STEPS[arm] * 2, "budget cost"),
              (payload["tag"], "final", "tag"), (payload["fixed_final"], True, "fixed final"))
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"checkpoint {label} mismatch")
    steps4, steps8, fresh_digest, fresh_common = build_paired_models(root=root)
    fresh = steps4 if arm == "steps4" else steps8
    if fresh_digest != initial_digest or fresh_common != common_digest or digest_state_dict(fresh) != initial_digest:
        raise ValueError("checkpoint initial reference cannot be reconstructed")
    expected_inventory = {"x": type(fresh.x_embedding).__name__, "y": type(fresh.y_embedding).__name__,
                          "bitlinear": [], "native_steps": NATIVE_STEPS[arm]}
    if payload["module_inventory"] != expected_inventory:
        raise ValueError("checkpoint module inventory mismatch")
    fresh.load_state_dict(payload["state_dict"], strict=True)
    if digest_state_dict(fresh) != payload["model_digest"]:
        raise ValueError("checkpoint model digest mismatch")
    return fresh, dict(payload)


__all__ = [
    "ARMS", "BATCH_SIZE", "CHECKPOINT_UPDATE", "DEFAULT_E18_PREFLIGHT", "DEFAULT_E18_RUN", "DEFAULT_PROTECTED_SNAPSHOT",
    "E18_SCHEMA", "E18_SOURCE_RELATIVE_PATHS", "ENCODER_SPEC", "EXPECTED_BATCH_DIGEST", "EXPECTED_MANIFEST_HASH",
    "EXPECTED_TARGET_DIGEST", "LEARNED_PARAMETER_COUNT", "NATIVE_STEPS", "OPTIMIZER_CONFIG", "RUN_CONFIG",
    "NativeStepRegisterModel", "aggregate_metrics", "assert_paired_models", "build_paired_models", "checkpoint_payload",
    "common_core_digest", "digest_state_dict", "evaluate_split", "fixed_stream", "load_checkpoint", "make_manifest",
    "paired_outcomes", "protocol_hash", "reconstruct_training_coverage", "source_hashes", "target_digest", "e17_reference_hash",
]
