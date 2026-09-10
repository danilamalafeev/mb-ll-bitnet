"""E32 width-package model, provenance, and fixed evaluation metadata.

This module owns only the new width-aware model and checkpoint mechanics.  The
accepted E20/E24/E27 helpers are imported as read-only references; no existing
scientific runner or model is modified.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import longer_native8_e20 as old
from . import w4_e27 as historical_w4
from .model import AttentionReader, FFNBlock, ModelConfig
from .quantization import BitLinear
from .register_e15 import OP_TO_ID, RegisterExample, _RegisterBase
from .runtime import seed_everything


ROOT = old.PROJECT_ROOT
SCHEMA = "e32_width128_v1"
SEEDS = (0, 1, 2)
WIDTH = 128
HISTORICAL_WIDTH = 64
FFN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 4
NATIVE_STEPS = 8
UPDATES = 16000
BATCH_SIZE = 64
REPEATS = 8
OPCODE_SCALE = 1.0 / 8.0  # Deliberately preserved from historical h64.
PARAMETER_COUNT_64 = 151232
PARAMETER_COUNT_128 = 335232
PREFLIGHT = Path("runs/e32_width_preflight")
RUN = Path("runs/e32_width")
PROTECTED = Path("results/E32_PROTECTED_HASHES.json")
REFERENCE = Path("results/E32_REFERENCE.json")
PROTOCOL = Path("results/E32_WIDTH_PROTOCOL.md")

QAT_NAMES = historical_w4.QAT_NAMES
ARMS = ("float", "w4")
# Frozen execution order: each nominal seed is a pair of float/W4 arms.
LABELS = tuple(f"{arm}128_seed{seed}" for seed in SEEDS for arm in ARMS)

COST_KEYS = ("program_forwards", "program_state_cases", "readout_positions", "internal_state_substeps")
TRAIN_COST_PER_MODEL = {
    "program_forwards": UPDATES,
    "program_state_cases": UPDATES * BATCH_SIZE,
    "readout_positions": UPDATES * BATCH_SIZE * 2,  # fixed stream average is two instructions
    "internal_state_substeps": UPDATES * BATCH_SIZE * 2 * NATIVE_STEPS,
}
TRAIN_COST_TOTAL = {key: value * len(ARMS) * len(SEEDS) for key, value in TRAIN_COST_PER_MODEL.items()}
SEEN_E21_COST_PER_MODEL = {
    "program_forwards": 71,
    "program_state_cases": 8960,
    "readout_positions": 23488,
    "internal_state_substeps": 187904,
}
LENGTH_COST_PER_MODEL = {
    "program_forwards": 12,
    "program_state_cases": 3072,
    "readout_positions": 13824,
    "internal_state_substeps": 110592,
}
EVAL_COST_PER_MODEL = {key: SEEN_E21_COST_PER_MODEL[key] + LENGTH_COST_PER_MODEL[key] for key in COST_KEYS}
EVAL_COST_TOTAL = {key: value * len(ARMS) * len(SEEDS) for key, value in EVAL_COST_PER_MODEL.items()}

HISTORICAL_REFERENCE_PATHS = {
    "e25_initial_reference": Path("results/E25_INITIAL_REFERENCE.json"),
    "e24_manifest": Path("runs/e24_continuation/manifest.json"),
    "e24_report": Path("runs/e24_continuation/report.json"),
    "e27_manifest": Path("runs/e27_w4/manifest.json"),
    "e27_report": Path("runs/e27_w4/report.json"),
    "e28_manifest": Path("runs/e28_length/manifest.json"),
    "e28_report": Path("runs/e28_length/report.json"),
    "e28_selection": Path("results/E28_SELECTION.json"),
}
INITIAL_REFERENCE_PATHS = {
    0: Path("runs/e18_step_budget_repaired_preflight/steps8_initial_state.pt"),
    1: Path("runs/e22_replication_preflight/initial_seed1.pt"),
    2: Path("runs/e22_replication_preflight/initial_seed2.pt"),
}

OPTIMIZER_CONFIG = dict(old.OPTIMIZER_CONFIG)
CONFIG: dict[str, Any] = {
    "width": WIDTH,
    "historical_width": HISTORICAL_WIDTH,
    "d_model": WIDTH,
    "d_ff": FFN_DIM,
    "num_blocks": NUM_BLOCKS,
    "num_heads": NUM_HEADS,
    "native_steps": NATIVE_STEPS,
    "seeds": list(SEEDS),
    "arms": list(ARMS),
    "labels": list(LABELS),
    "updates": UPDATES,
    "batch_size": BATCH_SIZE,
    "repeats": REPEATS,
    "data_seed": 0,
    "precision": "float32",
    "autocast": False,
    "cpu_threads": 4,
    "opcode_scale": OPCODE_SCALE,
    "quantization": "W4_absmax7_or_FP32_matching_linear",
    "optimizer": OPTIMIZER_CONFIG,
    "parameter_count": PARAMETER_COUNT_128,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(value: Any) -> str:
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
            for child in item:
                add(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()); digest.update(b"\0")
        else:
            raise TypeError(f"unsupported digest value: {type(item)!r}")

    add(value)
    return digest.hexdigest()


def digest_state_dict(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode()); digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _resolve(path: Path, root: Path = ROOT) -> Path:
    return path if path.is_absolute() else Path(root) / path


def verify_reference_hashes(root: Path = ROOT) -> dict[str, str]:
    """Guard the root-frozen historical references before any run."""
    root = Path(root)
    reference = _read_json(root / REFERENCE)
    hashes = reference.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("E32 reference hash map is malformed")
    for relative, expected in hashes.items():
        path = _resolve(Path(relative), root)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"E32 historical reference changed: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def verify_protected_hashes(root: Path = ROOT) -> dict[str, str]:
    root = Path(root)
    snapshot = _read_json(root / PROTECTED)
    hashes = snapshot.get("sha256")
    if not isinstance(hashes, dict) or len(hashes) != 355:
        raise ValueError("E32 protected snapshot must contain 355 files")
    for relative, expected in hashes.items():
        path = _resolve(Path(relative), root)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"E32 protected artifact changed: {relative}")
    return {str(k): str(v) for k, v in hashes.items()}


def _historical_rng(seed: int, root: Path = ROOT) -> Tensor:
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    if seed == 0:
        return torch.Generator(device="cpu").manual_seed(0).get_state().clone()
    payload = torch.load(_resolve(INITIAL_REFERENCE_PATHS[seed], root), map_location="cpu", weights_only=True)
    rng = payload.get("rng_state")
    if not isinstance(rng, Tensor):
        raise ValueError(f"historical seed{seed} initial RNG missing")
    return rng.detach().cpu().clone()


def _signed_bits(value: Tensor) -> Tensor:
    if value.dtype != torch.long or torch.any((value < 0) | (value > 15)):
        raise ValueError("signed bit input expects integer values in 0..15")
    shifts = torch.arange(4, dtype=torch.long, device=value.device)
    return ((((value.unsqueeze(-1) >> shifts) & 1) * 2) - 1).to(torch.float32)


class WidthSignedBitsEncoder(nn.Module):
    """Private width-generalized replacement for the historical 64-wide encoder."""

    def __init__(self, width: int, weight: Tensor):
        super().__init__()
        if tuple(weight.shape) != (width, 4) or weight.dtype != torch.float32:
            raise ValueError("signed-bit projection has wrong shape or dtype")
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(4, width, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.projection.weight.copy_(weight)

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.projection(_signed_bits(value))


class WidthRegisterModel(_RegisterBase):
    """Native8 register interpreter with generalized hidden width and FFN256."""

    def __init__(self, width: int = WIDTH, d_ff: int = FFN_DIM):
        super().__init__()
        if type(width) is not int or type(d_ff) is not int or width not in (HISTORICAL_WIDTH, WIDTH) or d_ff != FFN_DIM:
            raise ValueError("E32 supports only width64/width128 with FFN256")
        self.hidden_size = width
        self.width = width
        self.d_ff = d_ff
        self.native_steps = NATIVE_STEPS
        cfg = ModelConfig(d_model=width, d_ff=d_ff, num_blocks=NUM_BLOCKS,
                          num_heads=NUM_HEADS, num_objects=16, max_tokens=51,
                          steps=4, quantized=False)
        # Construction order intentionally mirrors QATRegisterModel.  This is
        # needed for the width64 factory identity control.
        self.x_embedding = nn.Embedding(16, width)
        self.y_embedding = nn.Embedding(16, width)
        self.opcode_embedding = nn.Embedding(3, width)
        self.role_keys = nn.Parameter(torch.empty(2, width))
        self.memory_norm = nn.LayerNorm(width)
        self.reader = AttentionReader(cfg)
        self.blocks = nn.ModuleList(FFNBlock(cfg) for _ in range(NUM_BLOCKS))
        self.output_norm = nn.LayerNorm(width)
        self.x_head = nn.Linear(width, 16, bias=False)
        self.y_head = nn.Linear(width, 16, bias=False)
        self._initialize()

    def _initialize(self) -> None:
        for emb in (self.x_embedding, self.y_embedding, self.opcode_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.role_keys, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def init_cache(self, x: Tensor, y: Tensor) -> dict[str, Any]:
        self._validate(x, y)
        xv, yv = self.x_embedding(x), self.y_embedding(y)
        key_memory = self.memory_norm(self.role_keys).unsqueeze(0).expand(x.shape[0], -1, -1)
        value_memory = self.memory_norm(torch.stack((xv, yv), dim=1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = self.reader.build_kv(key_memory, value_memory)
        return {"h": xv + yv, "x_initial": xv, "y_initial": yv, "kv": kv, "substeps": 0}

    def step(self, cache: dict[str, Any], opcode: int | Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, Any]]:
        if isinstance(opcode, int):
            opcode = torch.full((cache["h"].shape[0],), opcode, dtype=torch.long, device=cache["h"].device)
        if opcode.ndim != 1 or opcode.shape[0] != cache["h"].shape[0] or torch.any((opcode < 0) | (opcode >= 3)):
            raise ValueError("opcode must be a batch-sized value in 0..2")
        h = cache["h"]
        with torch.autocast(device_type=h.device.type, enabled=False):
            for inner in range(self.native_steps):
                # Keep E20/E27's 1/sqrt(64) coefficient as registered.
                h = h + self.opcode_embedding(opcode) * OPCODE_SCALE
                h = h + self.reader(h, cache["kv"])
                h = self.blocks[(cache["substeps"] + inner) % NUM_BLOCKS](h)
        cache = dict(cache, h=h, substeps=cache["substeps"] + self.native_steps)
        logits = self.output(self.output_norm(h))
        return logits, cache

    def output(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.x_head(h), self.y_head(h)

    def forward(self, x: Tensor, y: Tensor, op_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(x, y, op_ids)
        cache = self.init_cache(x, y)
        xs, ys = [], []
        for index in range(op_ids.shape[1]):
            (xl, yl), cache = self.step(cache, op_ids[:, index])
            xs.append(xl); ys.append(yl)
        return torch.stack(xs, 1), torch.stack(ys, 1)


def _linear_names() -> tuple[str, ...]:
    return tuple(QAT_NAMES)


def _as_w4(model: WidthRegisterModel) -> None:
    for name in _linear_names():
        module = model.get_submodule(name)
        if type(module) is not nn.Linear:
            raise ValueError(f"expected plain linear before W4 conversion: {name}")
        module.__class__ = historical_w4.W4BitLinear


def _as_float(model: WidthRegisterModel) -> None:
    for name in _linear_names():
        module = model.get_submodule(name)
        if type(module) is not historical_w4.W4BitLinear:
            raise ValueError(f"expected W4 linear before float conversion: {name}")
        module.__class__ = nn.Linear


def _base_model(width: int, seed: int, *, root: Path = ROOT) -> WidthRegisterModel:
    if type(width) is not int or type(seed) is not int or width not in (HISTORICAL_WIDTH, WIDTH) or seed not in SEEDS:
        raise ValueError("unregistered E32 width/seed")
    with torch.random.fork_rng(devices=[]):
        seed_everything(seed, deterministic=True, cpu_threads=4)
        model = WidthRegisterModel(width, FFN_DIM)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x_weight = torch.randn((width, 4), generator=generator, dtype=torch.float32) * 0.01
        y_weight = torch.randn((width, 4), generator=generator, dtype=torch.float32) * 0.01
        model.x_embedding = WidthSignedBitsEncoder(width, x_weight)
        model.y_embedding = WidthSignedBitsEncoder(width, y_weight)
        _as_w4(model)
        model.initial_rng_state = _historical_rng(seed, root)
    return model


def build_initial_model(width: int, arm: str, seed: int, *, root: Path = ROOT) -> WidthRegisterModel:
    """Build a fresh arm without model forwards or optimizer updates."""
    if type(width) is not int or type(seed) is not int or arm not in ARMS:
        raise ValueError("arm must be float or w4")
    model = _base_model(width, seed, root=root)
    if arm == "float":
        _as_float(model)
    check_inventory(model, width=width, arm=arm)
    return model


def check_inventory(model: nn.Module, *, width: int = WIDTH, arm: str | None = None) -> None:
    if type(width) is not int or width not in (HISTORICAL_WIDTH, WIDTH):
        raise ValueError("E32 unsupported width")
    if arm not in (None, "float", "w4"):
        raise ValueError("unknown E32 arm")
    if getattr(model, "width", None) != width or getattr(model, "native_steps", None) != NATIVE_STEPS:
        raise ValueError("E32 model width/native-step inventory")
    if getattr(model, "d_ff", None) != FFN_DIM or getattr(model, "native_steps", None) != NATIVE_STEPS:
        raise ValueError("E32 FFN/native-step inventory")
    reader = getattr(model, "reader", None)
    if getattr(reader, "num_heads", None) != NUM_HEADS or getattr(reader, "head_dim", None) != width // NUM_HEADS:
        raise ValueError("E32 attention inventory")
    if len(getattr(model, "blocks", ())) != NUM_BLOCKS:
        raise ValueError("E32 block inventory")
    if any(getattr(block, "up", None).in_features != width or
           getattr(block, "up", None).out_features != FFN_DIM or
           getattr(block, "down", None).in_features != FFN_DIM or
           getattr(block, "down", None).out_features != width
           for block in model.blocks):
        raise ValueError("E32 FFN shape inventory")
    if getattr(model, "x_head", None).in_features != width or getattr(model, "y_head", None).in_features != width:
        raise ValueError("E32 output head inventory")
    expected = PARAMETER_COUNT_128 if width == WIDTH else PARAMETER_COUNT_64
    if sum(parameter.numel() for parameter in model.parameters()) != expected:
        raise ValueError("E32 parameter count mismatch")
    if not isinstance(model.x_embedding, WidthSignedBitsEncoder) or not isinstance(model.y_embedding, WidthSignedBitsEncoder):
        raise ValueError("E32 signed-bit encoder inventory")
    if model.x_embedding.projection.weight.shape != (width, 4) or model.y_embedding.projection.weight.shape != (width, 4):
        raise ValueError("E32 input projection shape")
    locations = tuple(name for name, module in model.named_modules() if type(module) is historical_w4.W4BitLinear)
    if arm == "w4" and locations != QAT_NAMES:
        raise ValueError("E32 W4 projection inventory")
    if arm == "float" and locations:
        raise ValueError("E32 float arm contains W4 projections")
    if any(parameter.dtype != torch.float32 or not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("E32 nonfinite/non-FP32 master")


def inventory(model: nn.Module, *, width: int, arm: str) -> dict[str, Any]:
    check_inventory(model, width=width, arm=arm)
    quantized = sum(module.weight.numel() for module in model.modules() if type(module) is historical_w4.W4BitLinear)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"width": width, "arm": arm, "parameter_count": total,
            "quantized_matrix_parameters": quantized,
            "unquantized_fp32_parameters": total - quantized,
            "w4_projection_locations": [name for name, module in model.named_modules()
                                         if type(module) is historical_w4.W4BitLinear]}


def assert_state_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor], *, separate: bool = False) -> None:
    if list(left) != list(right):
        raise ValueError("E32 state key/order mismatch")
    for key, tensor in left.items():
        other = right[key]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise ValueError(f"E32 state mismatch: {key}")
        if separate and tensor.data_ptr() == other.data_ptr():
            raise ValueError(f"E32 paired state storage is shared: {key}")


def assert_paired_models(float_model: nn.Module, w4_model: nn.Module, *, width: int = WIDTH) -> None:
    check_inventory(float_model, width=width, arm="float")
    check_inventory(w4_model, width=width, arm="w4")
    assert_state_equal(float_model.state_dict(), w4_model.state_dict(), separate=True)
    assert_state_equal(dict(float_model.named_parameters()), dict(w4_model.named_parameters()), separate=True)
    if not torch.equal(float_model.initial_rng_state, w4_model.initial_rng_state):
        raise ValueError("E32 paired training-start RNG differs")


def stream() -> list[list[RegisterExample]]:
    base = old.base_stream()
    batches = [list(batch) for _ in range(REPEATS) for batch in base]
    if len(batches) != UPDATES or old.batch_digest(batches) != _e25_manifest()["full_stream_digest"]:
        raise ValueError("E32 fixed stream digest mismatch")
    if old.target_digest(batches) != _e25_manifest()["full_target_digest"]:
        raise ValueError("E32 fixed target digest mismatch")
    for start in range(0, UPDATES, 2000):
        if old.batch_digest(batches[start:start + 2000]) != old.E18_BASE_STREAM_DIGEST:
            raise ValueError("E32 repeated base stream mismatch")
    return batches


def _e25_manifest() -> dict[str, Any]:
    return _read_json(ROOT / "runs/e25_qat_match_preflight/manifest.json")


def _e21_manifest() -> dict[str, Any]:
    return _read_json(ROOT / "runs/e21_composition_preflight/manifest.json")


def _e28_selection() -> dict[str, Any]:
    return _read_json(ROOT / "results/E28_SELECTION.json")


def make_manifest(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    protected = verify_protected_hashes(root)
    references = verify_reference_hashes(root)
    batches = stream()
    initials: dict[str, str] = {}
    rngs: dict[str, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            float_model = build_initial_model(WIDTH, "float", seed, root=root)
            w4_model = build_initial_model(WIDTH, "w4", seed, root=root)
            assert_paired_models(float_model, w4_model)
            key = f"{arm}128_seed{seed}"
            model = float_model if arm == "float" else w4_model
            initials[key] = digest_state_dict(model)
            rngs[key] = digest_object(model.initial_rng_state)
    source_paths = [
        Path("looped_bitnet/width_e32.py"), Path("scripts/width_e32.py"),
        Path("tests/test_width_e32.py"), PROTOCOL,
    ]
    if any(not (root / path).is_file() for path in source_paths):
        raise ValueError("E32 source provenance file missing")
    source_hashes = {str(path): sha256_file(root / path) for path in source_paths}
    references_used = {str(path): sha256_file(root / path) for path in HISTORICAL_REFERENCE_PATHS.values()}
    inventories = {
        "h64_float": inventory(build_initial_model(HISTORICAL_WIDTH, "float", 0, root=root), width=HISTORICAL_WIDTH, arm="float"),
        "h64_w4": inventory(build_initial_model(HISTORICAL_WIDTH, "w4", 0, root=root), width=HISTORICAL_WIDTH, arm="w4"),
        "h128_float": inventory(build_initial_model(WIDTH, "float", 0, root=root), width=WIDTH, arm="float"),
        "h128_w4": inventory(build_initial_model(WIDTH, "w4", 0, root=root), width=WIDTH, arm="w4"),
    }
    return {
        "schema": SCHEMA,
        "config": deepcopy(CONFIG),
        "initial_digests": initials,
        "initial_rng_digests": rngs,
        "source_hashes": source_hashes,
        "protected_snapshot_hash": sha256_file(root / PROTECTED),
        "protected_count": len(protected),
        "historical_reference_hashes": references_used,
        "e32_reference_hash": sha256_file(root / REFERENCE),
        "model_inventories": inventories,
        "full_stream_digest": old.batch_digest(batches),
        "full_target_digest": old.target_digest(batches),
        "base_stream_digest": old.E18_BASE_STREAM_DIGEST,
        "base_target_digest": old.E18_BASE_TARGET_DIGEST,
        "train_cost_per_model": dict(TRAIN_COST_PER_MODEL),
        "train_cost_total": dict(TRAIN_COST_TOTAL),
        "evaluation_cost_per_model": dict(EVAL_COST_PER_MODEL),
        "evaluation_cost_total": dict(EVAL_COST_TOTAL),
        "checkpoint_cost": dict(TRAIN_COST_PER_MODEL),
        "seen": _e25_manifest()["seen"],
        "programs": _e25_manifest()["seen"]["programs"],
        "symbolic": _e25_manifest()["symbolic"],
        "state_split": _e25_manifest()["state_split"],
        "e21": _e21_manifest(),
        "e28_selection": _e28_selection(),
        "historical_comparators": {
            "E24_float64": {"run": "runs/e24_continuation", "eval": "runs/e28_length"},
            "E27_W4_64": {"run": "runs/e27_w4", "eval": "runs/e28_length"},
        },
    }


def checkpoint_payload(model: WidthRegisterModel, optimizer: torch.optim.Optimizer, manifest: Mapping[str, Any],
                       *, arm: str, width: int, seed: int, update: int, qa: bool = False,
                       training_cost: Mapping[str, int] | None = None) -> dict[str, Any]:
    if arm not in ARMS or type(width) is not int or width not in (HISTORICAL_WIDTH, WIDTH) or type(seed) is not int or seed not in SEEDS:
        raise ValueError("E32 checkpoint identity")
    if type(update) is not int or update <= 0 or (not qa and update != UPDATES):
        raise ValueError("E32 checkpoint update")
    if type(qa) is not bool or not isinstance(manifest, Mapping):
        raise ValueError("E32 checkpoint metadata types")
    check_inventory(model, width=width, arm=arm)
    if not model.training:
        raise ValueError("E32 checkpoint must preserve training mode")
    label = f"{arm}{width}_seed{seed}"
    initials = manifest.get("initial_digests")
    rng_digests = manifest.get("initial_rng_digests")
    if not isinstance(initials, Mapping) or not isinstance(rng_digests, Mapping):
        raise ValueError("E32 manifest initial provenance missing")
    if label not in initials or label not in rng_digests:
        raise ValueError(f"E32 manifest initial provenance missing: {label}")
    if not isinstance(manifest.get("source_hashes"), Mapping):
        raise ValueError("E32 manifest source provenance missing")
    if training_cost is None:
        training_cost = manifest["checkpoint_cost"]
    if not isinstance(training_cost, Mapping) or set(training_cost) != set(COST_KEYS) or any(type(training_cost[key]) is not int for key in COST_KEYS):
        raise ValueError("E32 checkpoint training cost malformed")
    state = deepcopy(model.state_dict())
    opt = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    return {
        "schema": SCHEMA, "arm": arm, "width": width, "seed": seed, "update": update,
        "qa": bool(qa), "native_steps": NATIVE_STEPS, "ffn_dim": FFN_DIM,
        "parameter_count": PARAMETER_COUNT_128 if width == WIDTH else PARAMETER_COUNT_64,
        "config": deepcopy(manifest["config"]), "manifest_digest": old.canonical_hash(manifest),
        "label": label, "initial_digest": initials[label],
        "initial_rng_digest": rng_digests[label],
        "source_hashes": deepcopy(manifest["source_hashes"]),
        "protected_snapshot_hash": manifest["protected_snapshot_hash"],
        "e32_reference_hash": manifest["e32_reference_hash"],
        "historical_reference_hashes": deepcopy(manifest["historical_reference_hashes"]),
        "stream_digest": manifest["full_stream_digest"], "target_digest": manifest["full_target_digest"],
        "state_dict": state, "model_digest": digest_state_dict(model),
        "optimizer_state_dict": opt, "optimizer_digest": digest_object(opt),
        "rng_state": rng, "rng_digest": digest_object(rng), "training_mode": bool(model.training),
        "training_cost": dict(training_cost),
    }


def _validate_checkpoint_state(raw: Any, model: nn.Module) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("E32 checkpoint state missing")
    expected = model.state_dict()
    if list(raw) != list(expected):
        raise ValueError("E32 checkpoint state key/order mismatch")
    for name, tensor in raw.items():
        if not isinstance(tensor, Tensor) or tensor.dtype != expected[name].dtype or tensor.shape != expected[name].shape:
            raise ValueError(f"E32 checkpoint state shape/dtype mismatch: {name}")
        if tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise ValueError(f"E32 checkpoint state is not finite CPU tensor: {name}")


def _validate_checkpoint_optimizer(raw: Any, model: nn.Module, optimizer: torch.optim.Optimizer, update: int) -> None:
    if not isinstance(raw, Mapping) or set(raw) != {"state", "param_groups"}:
        raise ValueError("E32 checkpoint optimizer payload malformed")
    reference = optimizer.state_dict()
    if raw["param_groups"] != reference["param_groups"]:
        raise ValueError("E32 optimizer parameter groups/configuration mismatch")
    states = raw["state"]
    if not isinstance(states, Mapping):
        raise ValueError("E32 optimizer state mapping missing")
    parameter_by_id = {}
    for group, parameters in zip(raw["param_groups"], optimizer.param_groups):
        if len(group["params"]) != len(parameters["params"]):
            raise ValueError("E32 optimizer parameter inventory mismatch")
        parameter_by_id.update(zip(group["params"], parameters["params"]))
    if set(states) != set(parameter_by_id) or len(states) != len(parameter_by_id):
        raise ValueError("E32 optimizer state inventory mismatch")
    for identifier, state in states.items():
        if not isinstance(identifier, int) or identifier not in parameter_by_id or not isinstance(state, Mapping):
            raise ValueError("E32 optimizer state identifier malformed")
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("E32 optimizer state fields mismatch")
        step = state["step"]
        if not isinstance(step, Tensor) or step.dtype != torch.float32 or step.shape != torch.Size([]):
            raise ValueError("E32 optimizer step tensor mismatch")
        if not torch.equal(step, torch.tensor(float(update), dtype=torch.float32)):
            raise ValueError("E32 optimizer step mismatch")
        parameter = parameter_by_id[identifier]
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if not isinstance(value, Tensor) or value.dtype != parameter.dtype or value.shape != parameter.shape:
                raise ValueError(f"E32 optimizer {name} shape/dtype mismatch")
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError(f"E32 optimizer {name} is not finite CPU tensor")


def load_checkpoint(path: Path, manifest: Mapping[str, Any], *, arm: str, width: int, seed: int,
                    update: int = UPDATES, qa: bool = False, root: Path = ROOT,
                    training_cost: Mapping[str, int] | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("E32 checkpoint payload must be a dict")
    if type(update) is not int or type(qa) is not bool:
        raise ValueError("E32 checkpoint load identity types")
    expected_identity = {
        "schema": SCHEMA, "arm": arm, "width": width, "seed": seed, "update": update,
        "qa": qa, "native_steps": NATIVE_STEPS, "ffn_dim": FFN_DIM,
        "parameter_count": PARAMETER_COUNT_128 if width == WIDTH else PARAMETER_COUNT_64,
        "label": f"{arm}{width}_seed{seed}", "training_mode": True,
    }
    for key, value in expected_identity.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise ValueError(f"E32 checkpoint identity mismatch: {key}")
    with torch.random.fork_rng(devices=[]):
        model = build_initial_model(width, arm, seed, root=root)
        raw = payload.get("state_dict")
        _validate_checkpoint_state(raw, model)
        rng = payload.get("rng_state")
        if not isinstance(rng, Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("E32 checkpoint RNG state malformed")
        optimizer_raw = payload.get("optimizer_state_dict")
        optimizer = old.make_optimizer(model)
        _validate_checkpoint_optimizer(optimizer_raw, model, optimizer, update)
        model.load_state_dict(raw, strict=True)
        optimizer.load_state_dict(optimizer_raw)
        assert_state_equal(raw, model.state_dict())
        model.train(True)
        torch.set_rng_state(rng)
        expected = checkpoint_payload(model, optimizer, manifest, arm=arm, width=width, seed=seed, update=update, qa=qa,
                                      training_cost=training_cost)
    if set(payload) != set(expected):
        raise ValueError("E32 checkpoint fields mismatch")
    for key, value in expected.items():
        if key in {"state_dict", "optimizer_state_dict", "rng_state"}:
            continue
        if payload.get(key) != value:
            raise ValueError(f"E32 checkpoint metadata mismatch: {key}")
    if digest_object(optimizer.state_dict()) != digest_object(payload["optimizer_state_dict"]):
        raise ValueError("E32 optimizer state changed during load")
    torch.set_rng_state(payload["rng_state"])
    return model, optimizer, payload


__all__ = [
    "ARMS", "BATCH_SIZE", "CONFIG", "COST_KEYS", "EVAL_COST_PER_MODEL", "EVAL_COST_TOTAL",
    "FFN_DIM", "HISTORICAL_WIDTH", "LABELS", "NATIVE_STEPS", "PARAMETER_COUNT_128", "PARAMETER_COUNT_64",
    "PREFLIGHT", "PROTECTED", "REFERENCE", "ROOT", "RUN", "SCHEMA", "SEEDS", "TRAIN_COST_PER_MODEL",
    "TRAIN_COST_TOTAL", "UPDATES", "WIDTH", "WidthRegisterModel", "WidthSignedBitsEncoder",
    "assert_paired_models", "assert_state_equal", "build_initial_model", "checkpoint_payload", "check_inventory", "inventory",
    "digest_object", "digest_state_dict", "load_checkpoint", "make_manifest", "sha256_file", "stream",
    "verify_protected_hashes", "verify_reference_hashes",
]
