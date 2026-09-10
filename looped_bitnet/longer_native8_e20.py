"""E20 fixed 8,000-update continuation probe for the repaired E18 native8 arm."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .step_budget_e18 import (
    BATCH_SIZE,
    ENCODER_SPEC,
    E18_SCHEMA,
    LEARNED_PARAMETER_COUNT,
    NativeStepRegisterModel,
    OPTIMIZER_CONFIG,
    PROJECT_ROOT,
    RUN_CONFIG as E18_RUN_CONFIG,
    build_paired_models as build_e18_models,
    common_core_digest,
    digest_state_dict,
    fixed_stream as fixed_e18_stream,
    target_digest,
)
from .float_qat_e16 import (
    aggregate_metrics,
    atomic_torch_save,
    canonical_hash,
    refuse_nonempty,
)
from .register_e15 import RegisterExample, batch_digest, sha256_file
from .runtime import environment, seed_everything


E20_SCHEMA = "e20_longer_native8_v1"
ARM = "steps8"
SEED = 0
BASE_UPDATES = 2000
TOTAL_UPDATES = 8000
MILESTONES = (2000, 4000, 8000)
REPEATS = 4
NATIVE_STEPS = 8
LEARNED_PARAMETER_COUNT = 151232
DEFAULT_E18_PREFLIGHT = Path("runs/e18_step_budget_repaired_preflight")
DEFAULT_E18_RUN = Path("runs/e18_step_budget_repaired")
DEFAULT_E18_CHECKPOINT = DEFAULT_E18_RUN / "steps8_seed0" / "final_u2000.pt"
DEFAULT_E18_REPORT = DEFAULT_E18_RUN / "report.json"
DEFAULT_E20_PREFLIGHT = Path("runs/e20_longer_native8_preflight")
DEFAULT_E20_RUN = Path("runs/e20_longer_native8")
DEFAULT_PROTECTED_SNAPSHOT = Path("results/E20_PROTECTED_HASHES.json")
E18_INITIAL_STATE = DEFAULT_E18_PREFLIGHT / "steps8_initial_state.pt"

E18_BASE_STREAM_DIGEST = "63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687"
E18_BASE_TARGET_DIGEST = "fe43d8c00b9e31dd07e04e54b913effc390081c715932d6e2945ee2cca79cdfc"
EXPECTED_FULL_STREAM_DIGEST = "c5cca9ee862452edd5b314598a2f8ed1f989c041ab33d3600e4b813be270a581"
EXPECTED_FULL_TARGET_DIGEST = "967481fb003075432aedd75111908f52a9981a85818bbf3a56e8ba5720754ba5"
EXPECTED_E18_MANIFEST_HASH = "941840023f66c1ab1b1bec8d9bb81d08a5170b6e9cc05b6d1294fe172400cd37"
EXPECTED_E18_CHECKPOINT_SHA256 = "18d4bb093566fb7395c8e6d5cd988dbb60bacf436c04a63ea4d6af81d5b9c8ef"
EXPECTED_E18_REPORT_SHA256 = "cdb95bc637fa316b96b5cf0893413c9b1d617e1fed3733c16179b5d0f11519e0"
EXPECTED_E18_MODEL_DIGEST = "6e9e88fcbbff75148942077232aba30b26e6a76e6d78d1e0d04681302e3e8a5b"

TRAIN_COST = {"updates": TOTAL_UPDATES, "examples": 512000, "internal_state_updates": 8192000}
EVAL_COST_PER_MILESTONE = {"program_state_evaluations": 7168, "readout_positions": 18368,
                           "internal_state_updates": 146944}
EVAL_COST_TOTAL = {"program_state_evaluations": 21504, "readout_positions": 55104,
                   "internal_state_updates": 440832}

E20_SOURCE_RELATIVE_PATHS = {
    "e20_module": "looped_bitnet/longer_native8_e20.py",
    "e20_runner": "scripts/longer_native8_e20.py",
    "e20_tests": "tests/test_longer_native8_e20.py",
    "e20_protocol": "results/E20_LONGER_NATIVE8_PROTOCOL.md",
    "e18_module": "looped_bitnet/step_budget_e18.py",
    "e18_runner": "scripts/step_budget_e18.py",
    "e18_tests": "tests/test_step_budget_e18.py",
    "e18_protocol": "results/E18_STEP_BUDGET_PROTOCOL.md",
    "e17_module": "looped_bitnet/bit_input_e17.py",
    "e16_module": "looped_bitnet/float_qat_e16.py",
    "e15_module": "looped_bitnet/register_e15.py",
    "model": "looped_bitnet/model.py",
    "quantization": "looped_bitnet/quantization.py",
    "runtime": "looped_bitnet/runtime.py",
    "config": "looped_bitnet/config.py",
    "data": "looped_bitnet/data.py",
    "engine": "looped_bitnet/engine.py",
}

RUN_CONFIG: dict[str, Any] = {
    **dict(E18_RUN_CONFIG),
    "schema": E20_SCHEMA,
    "arm": ARM,
    "encoder_mode": "bits",
    "native_steps": NATIVE_STEPS,
    "base_updates": BASE_UPDATES,
    "updates": TOTAL_UPDATES,
    "stream_repeats": REPEATS,
    "milestones": list(MILESTONES),
    "batch_size": BATCH_SIZE,
}


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    return {name: sha256_file(root / relative) for name, relative in E20_SOURCE_RELATIVE_PATHS.items()}


def protocol_hash(root: Path = PROJECT_ROOT) -> str:
    return sha256_file(Path(root) / E20_SOURCE_RELATIVE_PATHS["e20_protocol"])


def digest_object(value: Any) -> str:
    """Stable digest for optimizer/RNG payloads, including tensor metadata."""
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0"); digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode()); digest.update(b"\0")
            digest.update(tensor.numpy().tobytes()); digest.update(b"\0")
        elif isinstance(item, Mapping):
            digest.update(b"map\0")
            for key in sorted(item, key=lambda x: str(x)):
                add(str(key)); add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"seq\0")
            for child in item: add(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()); digest.update(b"\0")
        else:
            raise TypeError(f"unsupported digest value: {type(item)!r}")

    add(value)
    return digest.hexdigest()


def reference_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(root)
    checkpoint = root / DEFAULT_E18_CHECKPOINT
    report = root / DEFAULT_E18_REPORT
    values = {"checkpoint": sha256_file(checkpoint), "report": sha256_file(report)}
    if values != {"checkpoint": EXPECTED_E18_CHECKPOINT_SHA256, "report": EXPECTED_E18_REPORT_SHA256}:
        raise ValueError("E18 reference checkpoint/report changed")
    return values


def base_stream(seed: int = SEED) -> list[list[RegisterExample]]:
    if seed != SEED:
        raise ValueError("E20 is registered for seed0 only")
    batches = fixed_e18_stream(seed)
    if batch_digest(batches) != E18_BASE_STREAM_DIGEST or target_digest(batches) != E18_BASE_TARGET_DIGEST:
        raise ValueError("E18 base stream changed")
    return batches


def full_stream(seed: int = SEED) -> list[list[RegisterExample]]:
    batches = base_stream(seed)
    full = [list(batch) for _ in range(REPEATS) for batch in batches]
    if batch_digest(full) != EXPECTED_FULL_STREAM_DIGEST or target_digest(full) != EXPECTED_FULL_TARGET_DIGEST:
        raise ValueError("E20 repeated stream digest changed")
    return full


def phase_boundaries() -> dict[str, Any]:
    batches = full_stream()
    counts = {str(length): sum(len(batch[0].program) == length for batch in batches) for length in (1, 2, 3)}
    if counts != {"1": 2664, "2": 2672, "3": 2664}:
        raise ValueError("E20 repeated schedule counts changed")
    return {"updates": TOTAL_UPDATES, "batch_size": BATCH_SIZE, "lengths": [1, 2, 3],
            "counts": counts, "phase_updates": [0, 2000, 4000, 6000, 8000]}


def build_initial_model(*, root: Path = PROJECT_ROOT) -> tuple[NativeStepRegisterModel, str, str]:
    """Reconstruct E18's exact native8 initial state for a fresh E20 optimizer."""
    with torch.random.fork_rng(devices=[]):
        seed_everything(SEED, deterministic=True, cpu_threads=4)
        _, model, initial_digest, common_digest = build_e18_models(SEED, root=root)
    payload = torch.load(Path(root) / E18_INITIAL_STATE, map_location="cpu", weights_only=True)
    if payload.get("arm") != ARM or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("E18 native8 initial reference is malformed")
    model.load_state_dict(payload["state_dict"], strict=True)
    if digest_state_dict(model) != initial_digest or payload.get("digest") != initial_digest:
        raise ValueError("E20 initial digest does not reproduce E18 native8 bits")
    if common_core_digest(model) != common_digest:
        raise ValueError("E20 common initial digest mismatch")
    if sum(parameter.numel() for parameter in model.parameters()) != LEARNED_PARAMETER_COUNT:
        raise ValueError("E20 parameter count mismatch")
    return model, initial_digest, common_digest


def initial_inventory(model: NativeStepRegisterModel) -> dict[str, Any]:
    return {"x": type(model.x_embedding).__name__, "y": type(model.y_embedding).__name__,
            "bitlinear": [], "native_steps": NATIVE_STEPS, "parameter_count": LEARNED_PARAMETER_COUNT,
            "state_keys": list(model.state_dict())}


def make_manifest(*, e18_manifest: Mapping[str, Any], e18_report: Mapping[str, Any],
                  initial_digest: str, common_digest: str, coverage: Mapping[str, Any],
                  root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if canonical_hash(e18_manifest) != EXPECTED_E18_MANIFEST_HASH:
        raise ValueError("E20 requires the frozen repaired E18 manifest")
    refs = reference_hashes(root)
    batches = full_stream()
    return {
        "format_version": 1, "schema": E20_SCHEMA, "experiment": "longer_native8_e20",
        "arm": ARM, "seed": SEED, "native_steps": NATIVE_STEPS,
        "source_hashes": source_hashes(root), "protocol_hash": protocol_hash(root),
        "config": dict(RUN_CONFIG), "config_hash": canonical_hash(RUN_CONFIG),
        "optimizer": dict(OPTIMIZER_CONFIG), "encoder_spec": dict(ENCODER_SPEC),
        "encoder_mode": "bits", "parameter_count": LEARNED_PARAMETER_COUNT,
        "initial_digest": initial_digest, "common_core_digest": common_digest,
        "e18_manifest_hash": EXPECTED_E18_MANIFEST_HASH,
        "e18_reference_checkpoint_sha256": refs["checkpoint"],
        "e18_reference_report_sha256": refs["report"],
        "e18_reference_model_digest": EXPECTED_E18_MODEL_DIGEST,
        "base_stream_digest": E18_BASE_STREAM_DIGEST, "base_target_digest": E18_BASE_TARGET_DIGEST,
        "full_stream_digest": batch_digest(batches), "full_target_digest": target_digest(batches),
        "schedule": phase_boundaries(), "milestones": list(MILESTONES),
        "train_cost": dict(TRAIN_COST), "evaluation_cost_per_milestone": dict(EVAL_COST_PER_MILESTONE),
        "evaluation_cost_total": dict(EVAL_COST_TOTAL), "coverage": dict(coverage),
        "programs": e18_manifest["programs"], "state_split": e18_manifest["state_split"],
        "e18_source_hashes": e18_manifest["source_hashes"],
        "environment": environment(torch.device("cpu")),
        "reference_status": e18_report.get("status"),
    }


def checkpoint_payload(model: NativeStepRegisterModel, optimizer: torch.optim.Optimizer,
                       *, rng_state: Tensor, update: int, manifest: Mapping[str, Any],
                       qa: bool = False) -> dict[str, Any]:
    if update <= 0 or update > TOTAL_UPDATES or (not qa and update not in MILESTONES):
        raise ValueError("E20 checkpoint update is outside the registered milestones")
    if model.native_steps != NATIVE_STEPS or sum(parameter.numel() for parameter in model.parameters()) != LEARNED_PARAMETER_COUNT:
        raise ValueError("E20 checkpoint model inventory mismatch")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer_state = deepcopy(optimizer.state_dict())
    rng = rng_state.detach().cpu().clone()
    return {
        "format_version": 1, "schema": E20_SCHEMA, "arm": ARM, "native_steps": NATIVE_STEPS,
        "encoder_mode": "bits", "seed": SEED, "update": update, "milestone": f"u{update}",
        "qa": bool(qa), "params": LEARNED_PARAMETER_COUNT, "state_dict": state,
        "model_digest": digest_state_dict(model), "optimizer_state_dict": optimizer_state,
        "optimizer_digest": digest_object(optimizer_state), "rng_state": rng,
        "rng_digest": digest_object(rng), "initial_digest": manifest["initial_digest"],
        "common_core_digest": manifest["common_core_digest"],
        "base_stream_digest": manifest["base_stream_digest"], "full_stream_digest": manifest["full_stream_digest"],
        "prefix_stream_digest": manifest["base_stream_digest"], "base_target_digest": manifest["base_target_digest"],
        "full_target_digest": manifest["full_target_digest"], "e18_manifest_hash": manifest["e18_manifest_hash"],
        "e18_reference_checkpoint_sha256": manifest["e18_reference_checkpoint_sha256"],
        "e18_reference_report_sha256": manifest["e18_reference_report_sha256"],
        "e18_reference_model_digest": manifest["e18_reference_model_digest"],
        "source_hashes": dict(manifest["source_hashes"]), "protocol_hash": manifest["protocol_hash"],
        "config": dict(manifest["config"]), "config_hash": manifest["config_hash"],
        "optimizer": dict(manifest["optimizer"]), "encoder_spec": dict(manifest["encoder_spec"]),
        "parameter_count": LEARNED_PARAMETER_COUNT,
        "cumulative_cost": {"updates": update, "examples": update * BATCH_SIZE,
                             "internal_state_updates": update * BATCH_SIZE * NATIVE_STEPS * 2},
        "environment": environment(torch.device("cpu")), "module_inventory": initial_inventory(model),
        "training_mode": bool(model.training), "tag": "milestone", "fixed_final": update == TOTAL_UPDATES,
    }


__all__ = [
    "ARM", "BASE_UPDATES", "BATCH_SIZE", "DEFAULT_E18_CHECKPOINT", "DEFAULT_E18_PREFLIGHT", "DEFAULT_E18_REPORT",
    "DEFAULT_E20_PREFLIGHT", "DEFAULT_E20_RUN", "DEFAULT_PROTECTED_SNAPSHOT", "E18_BASE_STREAM_DIGEST",
    "E18_BASE_TARGET_DIGEST", "E18_INITIAL_STATE", "E20_SCHEMA", "E20_SOURCE_RELATIVE_PATHS", "ENCODER_SPEC",
    "EVAL_COST_PER_MILESTONE", "EVAL_COST_TOTAL", "EXPECTED_E18_CHECKPOINT_SHA256", "EXPECTED_E18_MANIFEST_HASH",
    "EXPECTED_E18_MODEL_DIGEST", "EXPECTED_E18_REPORT_SHA256", "EXPECTED_FULL_STREAM_DIGEST", "EXPECTED_FULL_TARGET_DIGEST",
    "LEARNED_PARAMETER_COUNT", "MILESTONES", "NATIVE_STEPS", "OPTIMIZER_CONFIG", "PROJECT_ROOT", "REPEATS",
    "RUN_CONFIG", "SEED", "TOTAL_UPDATES", "TRAIN_COST", "base_stream", "batch_digest", "build_initial_model",
    "canonical_hash", "checkpoint_payload", "common_core_digest", "digest_object", "digest_state_dict", "full_stream",
    "initial_inventory", "make_manifest", "phase_boundaries", "protocol_hash", "reference_hashes", "source_hashes",
    "target_digest",
]


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), **{k: OPTIMIZER_CONFIG[k] for k in
        ('lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'foreach')})


def load_checkpoint(path, manifest, *, expected_update, qa=False):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model, _, _ = build_initial_model()
    optimizer = make_optimizer(model)
    if payload.get('update') != expected_update or payload.get('qa') != qa:
        raise ValueError('checkpoint milestone/QA mismatch')
    if not qa and expected_update == 2000 and payload.get('model_digest') != EXPECTED_E18_MODEL_DIGEST:
        raise ValueError('checkpoint prefix reference mismatch')
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    expected = checkpoint_payload(model, optimizer, rng_state=payload['rng_state'],
                                  update=expected_update, manifest=manifest, qa=qa)
    for key in expected:
        if key in ('state_dict', 'optimizer_state_dict', 'rng_state', 'environment'):
            continue
        if payload.get(key) != expected[key]:
            raise ValueError(f'checkpoint {key} mismatch')
    # Optimizer payload must agree with the independently frozen hyperparameters.
    for group in optimizer.param_groups:
        for key in ('lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'foreach'):
            if group[key] != make_optimizer(model).param_groups[0][key]:
                raise ValueError('checkpoint optimizer configuration mismatch')
    if any(int(state['step']) != expected_update for state in optimizer.state.values()):
        raise ValueError('checkpoint optimizer step mismatch')
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError('checkpoint optimizer states incomplete')
    return model, optimizer, payload
