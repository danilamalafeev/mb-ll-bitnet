"""Fixed E27 symmetric W4 masters, provenance, and strict checkpoints.

E27 changes only the weight fake quantizer used by the fourteen registered
projections.  The model, FP32 masters, activation path, optimizer, stream,
and initial states are inherited from the accepted E26 mechanics.
"""

from copy import deepcopy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from . import longer_native8_e20 as old
from . import qat_match_e25 as qat_reference
from .quantization import BitLinear


ROOT = old.PROJECT_ROOT
SCHEMA = "e27_w4_v1"
SEEDS = (0, 1, 2)
UPDATES = 16000
PREFLIGHT = Path("runs/e27_w4_preflight")
RUN = Path("runs/e27_w4")
PROTECTED = Path("results/E27_PROTECTED_HASHES.json")
PROTOCOL = Path("results/E27_W4_PROTOCOL.md")
COMPARATOR_REFERENCE = Path("results/E27_COMPARATOR_REFERENCE.json")
E26_PROTECTED = Path("results/E26_PROTECTED_HASHES.json")
E26_RUN = Path("runs/e26_weight_only")
FLOAT_RUN = Path("runs/e24_continuation")
QAT_RUN = qat_reference.RUN
QAT_NAMES = (
    "reader.q_proj", "reader.k_proj", "reader.v_proj", "reader.out_proj",
    *tuple(f"blocks.{i}.{p}" for i in range(4) for p in ("up", "down")),
    "x_head", "y_head",
)
CONFIG = {
    **qat_reference.CONFIG,
    "seeds": list(SEEDS),
    "updates": UPDATES,
    "repeats": 8,
    "quantization": "symmetric_W4_weights_absmax7_STE_identity_FP32_activations",
    "parameter_count": 151232,
    "qat_names": list(QAT_NAMES),
}
COST_KEYS = ("program_forwards", "program_state_cases", "readout_positions", "internal_state_substeps")
EVAL_COST = dict(zip(COST_KEYS, (71, 8960, 23488, 187904)))


def symmetric_w4_weight(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric signed W4 fake quantization with identity STE.

    The scale is one absmax scale for the full projection matrix.  The signed
    integer range is [-7, 7], so -8 remains unused.  Detaching the complete
    correction makes the forward quantized while preserving d(output)/d(weight)
    as the identity straight-through estimator.
    """

    weight = weight.float()
    scale = (weight.abs().amax() / 7.0).clamp_min(eps)
    quantized = (weight / scale).round().clamp(-7, 7) * scale
    return weight + (quantized - weight).detach()


# Short descriptive aliases make the registered operator easy to inspect.
w4_weight = symmetric_w4_weight
weight_fake_quant = symmetric_w4_weight


class W4BitLinear(BitLinear):
    """A projection with FP32 masters, symmetric W4 weights, and FP32 input."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            bias = self.bias.float() if self.bias is not None else None
            return F.linear(value.float(), symmetric_w4_weight(self.weight), bias)


def check_inventory(model):
    qat_reference.check_inventory(model)
    locations = tuple(n for n, module in model.named_modules() if type(module) is W4BitLinear)
    if locations != QAT_NAMES:
        raise ValueError("W4 projection inventory")


def assert_state_equal(left, right, separate=False):
    if list(left) != list(right):
        raise ValueError("state key/order mismatch")
    for key, tensor in left.items():
        other = right[key]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise ValueError(f"initial tensor mismatch: {key}")
        if separate and tensor.data_ptr() == other.data_ptr():
            raise ValueError(f"shared storage: {key}")


def build_initial_model(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered seed")
    model = qat_reference.build_initial_model(seed)
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    # Reclassifying existing modules changes only forward dispatch and consumes
    # no RNG or parameter storage.
    for name in QAT_NAMES:
        model.get_submodule(name).__class__ = W4BitLinear
    check_inventory(model)
    assert_state_equal(before, model.state_dict(), separate=True)
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("replacement consumed RNG")
    return model


def verify_protected_hashes(root=ROOT):
    root = Path(root)
    parent = json.loads((root / E26_PROTECTED).read_text())
    parent_hashes = parent.get("sha256")
    if not isinstance(parent_hashes, dict) or not parent_hashes:
        raise ValueError("E26 protected snapshot malformed")
    for path, digest in parent_hashes.items():
        if old.sha256_file(root / path) != digest:
            raise ValueError(f"E26 protected artifact changed: {path}")

    snapshot = json.loads((root / PROTECTED).read_text())
    hashes = snapshot.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("E27 protected snapshot malformed")
    for path, digest in hashes.items():
        if old.sha256_file(root / path) != digest:
            raise ValueError(f"E27 protected artifact changed: {path}")
    if hashes.get(str(E26_PROTECTED)) != old.sha256_file(root / E26_PROTECTED):
        raise ValueError("E26 protected snapshot missing from E27 guard")
    if hashes.get(str(COMPARATOR_REFERENCE)) != old.sha256_file(root / COMPARATOR_REFERENCE):
        raise ValueError("E27 comparator reference missing from guard")
    return hashes


# The public E26 spelling is retained as a compatibility-shaped helper for
# reviewer code that expects the accepted guard API.
protected = verify_protected_hashes


def stream():
    base = old.base_stream()
    batches = [list(batch) for _ in range(8) for batch in base]
    for start in range(0, UPDATES, 2000):
        part = batches[start:start + 2000]
        if old.batch_digest(part) != old.E18_BASE_STREAM_DIGEST or old.target_digest(part) != old.E18_BASE_TARGET_DIGEST:
            raise ValueError("repeat stream mismatch")
    return batches


def make_manifest(root=ROOT):
    root = Path(root)
    hashes = verify_protected_hashes(root)
    initials, rngs = {}, {}
    for seed in SEEDS:
        model = build_initial_model(seed)
        path = old.E18_INITIAL_STATE if seed == 0 else Path(f"runs/e22_replication_preflight/initial_seed{seed}.pt")
        saved = torch.load(root / path, map_location="cpu", weights_only=True)
        assert_state_equal(model.state_dict(), saved["state_dict"])
        if seed and not torch.equal(model.initial_rng_state, saved["rng_state"]):
            raise ValueError("frozen initial RNG mismatch")
        initials[str(seed)] = old.digest_state_dict(model)
        rngs[str(seed)] = old.digest_object(model.initial_rng_state)

    independent = json.loads((root / "results/E25_INITIAL_REFERENCE.json").read_text())
    for item in independent["seeds"]:
        key = str(item["seed"])
        if initials[key] != item["initial_digest"] or rngs[key] != item["float_rng_digest"] or item["bitlinear_locations"] != list(QAT_NAMES):
            raise ValueError("independent initializer reference mismatch")
    if {item["seed"] for item in independent["seeds"]} != set(SEEDS):
        raise ValueError("independent seed scope")

    comparator_hashes = json.loads((root / COMPARATOR_REFERENCE).read_text()).get("sha256")
    if not isinstance(comparator_hashes, dict):
        raise ValueError("comparator reference malformed")
    for path, digest in comparator_hashes.items():
        if old.sha256_file(root / path) != digest:
            raise ValueError(f"independent comparator changed: {path}")

    batches = stream()
    previous = json.loads((root / E26_RUN / "manifest.json").read_text())
    sources = dict(old.E20_SOURCE_RELATIVE_PATHS)
    sources.update({
        str(p): str(p) for p in (
            PROTOCOL, COMPARATOR_REFERENCE, E26_PROTECTED,
            Path("looped_bitnet/w4_e27.py"), Path("scripts/w4_e27.py"), Path("tests/test_w4_e27.py"),
            Path("looped_bitnet/weight_only_e26.py"), Path("scripts/weight_only_e26.py"),
            Path("tests/test_weight_only_e26.py"), Path("looped_bitnet/qat_match_e25.py"),
            Path("looped_bitnet/replication_e22.py"), Path("scripts/replication_e22.py"),
            Path("scripts/composition_e21.py"), Path("scripts/continuation_e24.py"),
        )
    })
    references = [
        COMPARATOR_REFERENCE, Path("results/E25_INITIAL_REFERENCE.json"),
        E26_RUN / "manifest.json", E26_RUN / "report.json",
        *(E26_RUN / f"seed{s}/u16000.pt" for s in SEEDS),
        FLOAT_RUN / "manifest.json", FLOAT_RUN / "report.json",
        *(FLOAT_RUN / f"seed{s}/u16000.pt" for s in SEEDS),
    ]
    return {
        "schema": SCHEMA,
        "config": deepcopy(CONFIG),
        "initial_digests": initials,
        "initial_rng_digests": rngs,
        "source_hashes": {k: old.sha256_file(root / p) for k, p in sources.items()},
        "protected_snapshot_hash": old.sha256_file(root / PROTECTED),
        "protected_count": len(hashes),
        "comparator_references": {str(p): old.sha256_file(root / p) for p in references},
        "full_stream_digest": old.batch_digest(batches),
        "full_target_digest": old.target_digest(batches),
        "base_stream_digest": old.E18_BASE_STREAM_DIGEST,
        "base_target_digest": old.E18_BASE_TARGET_DIGEST,
        "checkpoint_costs": {
            str(UPDATES): {
                "program_forwards": UPDATES,
                "program_state_cases": UPDATES * 64,
                "readout_positions": sum(len(x.program) for b in batches for x in b),
                "internal_state_substeps": 8 * sum(len(x.program) for b in batches for x in b),
            }
        },
        "seen": previous["seen"],
        "symbolic": previous["symbolic"],
        "state_split": previous["state_split"],
    }


def checkpoint_payload(model, optimizer, manifest, seed, update, *, qa=False):
    if (
        type(seed) is not int or seed not in SEEDS or type(update) is not int
        or type(qa) is not bool or update <= 0 or update > UPDATES
        or (not qa and update != UPDATES)
    ):
        raise ValueError("checkpoint seed/update")
    check_inventory(model)
    if not model.training:
        raise ValueError("checkpoint must preserve training mode")
    state = deepcopy(model.state_dict())
    opt = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    return {
        "schema": SCHEMA, "seed": seed, "update": update, "qa": qa,
        "config": deepcopy(CONFIG), "completed_updates": update,
        "training_cost": manifest["checkpoint_costs"][str(update)],
        "manifest_digest": old.canonical_hash(manifest),
        "initial_digest": manifest["initial_digests"][str(seed)],
        "initial_rng_digest": manifest["initial_rng_digests"][str(seed)],
        "qat_names": list(QAT_NAMES), "state_dict": state,
        "model_digest": old.digest_state_dict(model),
        "optimizer_state_dict": opt, "optimizer_digest": old.digest_object(opt),
        "rng_state": rng, "rng_digest": old.digest_object(rng),
        "training_mode": model.training,
    }


def load_checkpoint(path, manifest, *, seed, update=UPDATES, qa=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload type")
    if (
        type(payload.get("training_mode")) is not bool
        or type(payload.get("qa")) is not bool
        or any(type(payload.get(k)) is not int for k in ("seed", "update", "completed_updates"))
    ):
        raise ValueError("checkpoint metadata types")
    with torch.random.fork_rng(devices=[]):
        model = build_initial_model(seed)
        initial = model.state_dict()
        raw = payload["state_dict"]
        if list(raw) != list(initial) or any(raw[k].dtype != v.dtype or raw[k].shape != v.shape for k, v in initial.items()):
            raise ValueError("checkpoint tensor inventory")
        model.load_state_dict(raw, strict=True)
        optimizer = old.make_optimizer(model)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        model.train(payload["training_mode"])
        torch.set_rng_state(payload["rng_state"])
        expected = checkpoint_payload(model, optimizer, manifest, seed, update, qa=qa)
    if set(payload) != set(expected):
        raise ValueError("checkpoint fields")
    for key in expected:
        if key not in ("state_dict", "optimizer_state_dict", "rng_state") and payload[key] != expected[key]:
            raise ValueError(f"checkpoint {key} mismatch")
    reference = old.make_optimizer(model).state_dict()["param_groups"]
    if optimizer.state_dict()["param_groups"] != reference:
        raise ValueError("optimizer config/membership")
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("optimizer inventory")
    for param, value in optimizer.state.items():
        if set(value) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(value["step"], torch.Tensor) or value["step"].numel() != 1 or value["step"].item() != update:
            raise ValueError("optimizer step/fields")
        for key in ("exp_avg", "exp_avg_sq"):
            if value[key].shape != param.shape or value[key].dtype != param.dtype or not torch.isfinite(value[key]).all():
                raise ValueError("optimizer tensor inventory")
    torch.set_rng_state(payload["rng_state"])
    return model, optimizer, payload
