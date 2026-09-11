"""Inference-only latent state-transfer audit for the gated-carry checkpoints.

The audit compares semantically identity paths of different lengths and then a
common probe instruction. It performs no backward pass or optimizer update and
writes only aggregate metrics; no hidden tensors are serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_gated_carry as gated
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = diagnostic_runtime.SCIENCE_MANIFEST
ENDPOINT = diagnostic_runtime.SCIENCE_ENDPOINT
SCIENCE_REPORT = ROOT / "runs" / "pc_gated_carry_v1" / "science" / "report.json"
SCIENCE_DIR = ROOT / "runs" / "pc_gated_carry_v1" / "science"
OUTPUT = ROOT / "runs" / "pc_gated_carry_v1" / "state_transfer_audit"
AUDIT_SCHEMA = "pc_state_transfer_audit_v1"
ARMS = ("A", "B")
CHECKPOINTS = {"A": SCIENCE_DIR / "unhooked" / "checkpoints" / "local1000.pt", "B": SCIENCE_DIR / "gated" / "checkpoints" / "local1000.pt"}

# Every pair has the same semantic effect on the DSL state. The final ADD is a
# common probe, so its logits test whether the latent states agree downstream.
IDENTITY_PAIRS = (
    {"id": "swap_pair_vs_two_pairs", "left": ("SWAP", "SWAP"), "right": ("SWAP", "SWAP", "SWAP", "SWAP")},
    {"id": "xor_pair_vs_two_pairs", "left": ("XOR", "XOR"), "right": ("XOR", "XOR", "XOR", "XOR")},
    {"id": "mixed_identity_vs_two_cycles", "left": ("SWAP", "XOR", "XOR", "SWAP"), "right": ("SWAP", "XOR", "XOR", "SWAP", "SWAP", "XOR", "XOR", "SWAP")},
    {"id": "swap_vs_xor_identity", "left": ("SWAP", "SWAP"), "right": ("XOR", "XOR")},
)
PROBE = "ADD"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary collision: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _refuse_fresh(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing non-empty audit output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _states() -> list[tuple[int, int]]:
    return [tuple(int(v) for v in state) for state in dsl.STATE_ORDER]


def _inputs(program: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = _states()
    bits = accepted.signed_bit_matrix(device=device)
    ids_x = torch.tensor([state[0] for state in states], dtype=torch.long, device=device)
    ids_y = torch.tensor([state[1] for state in states], dtype=torch.long, device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[op] for op in program] for _ in states], dtype=torch.long, device=device)
    return x_bits, y_bits, ops


def _load_arm(arm: str, manifest: Mapping[str, Any], accepted_report: Mapping[str, Any], device: torch.device) -> tuple[torch.nn.Module, latent.LatentSlotAdapter, torch.optim.Optimizer, dict[str, Any]]:
    counter = diagnostic_runtime._new_counter()
    model, adapter, optimizer, _endpoint_payload = diagnostic_runtime._load_endpoint(
        manifest=manifest,
        report=accepted_report,
        endpoint=Path(diagnostic_runtime._resolve(ENDPOINT, ROOT)),
        manifest_path=Path(diagnostic_runtime._resolve(MANIFEST, ROOT)),
        source_binding_digest=str(manifest["source_inventory"]["digest"]),
        device=device,
        counter=counter,
        # The accepted diagnostic loader flushes its accounting callback
        # unconditionally; keep this audit read-only without writing a second
        # accounting artifact.
        sink=lambda _counter: None,
        root=ROOT,
    )
    checkpoint_path = CHECKPOINTS[arm]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("arm") != arm or checkpoint.get("local_update") != 1000 or checkpoint.get("absolute_update") != 43000 or checkpoint.get("next_batch_index") != 1000:
        raise ValueError(f"{arm} checkpoint lineage changed")
    if arm == "B":
        gated.attach_carry_gate(adapter)
        gated.append_carry_gate_optimizer_group(optimizer, adapter)
        gated.validate_optimizer_carry_gate_association(optimizer, adapter)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    model.train(False)
    adapter.train(False)
    identity = {
        "arm": arm,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_digest": science._digest(model.state_dict()),
        "adapter_digest": science._digest(adapter.state_dict()),
        "architecture_id": latent.ARCHITECTURE_ID if arm == "A" else gated.GATED_ARCHITECTURE_ID,
    }
    return model, adapter, optimizer, identity


def _run_path(model: torch.nn.Module, adapter: latent.LatentSlotAdapter, arm: str, program: Sequence[str], device: torch.device) -> dict[str, Any]:
    full_program = tuple(program) + (PROBE,)
    x_bits, y_bits, ops = _inputs(full_program, device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        if arm == "A":
            logits, diagnostics = latent.latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
        else:
            logits, diagnostics = gated.gated_latent_slots_forward(model, adapter, x_bits, y_bits, ops, return_diagnostics=True)
    if not isinstance(diagnostics, Mapping) or len(diagnostics.get("slot_inputs", [])) != len(full_program) or len(diagnostics.get("slot_writes", [])) != len(full_program):
        raise ValueError("state-transfer diagnostics changed")
    before = diagnostics["slot_inputs"][-1].detach().float()
    initial = diagnostics["slot_inputs"][0].detach().float()
    probe = logits[0][:, -1, :].detach().float()
    return {"before_probe": before, "initial": initial, "probe_logits": probe}


def _tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left - right).reshape(left.shape[0], -1)
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    l2 = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.maximum(torch.linalg.vector_norm(left_flat, dim=1), torch.linalg.vector_norm(right_flat, dim=1)).clamp_min(1e-12)
    relative = l2 / denominator
    return {
        "mean_l2": float(l2.mean().item()),
        "max_l2": float(l2.max().item()),
        "mean_relative_l2": float(relative.mean().item()),
        "max_relative_l2": float(relative.max().item()),
    }


def _probe_metrics(left: torch.Tensor, right: torch.Tensor, states: Sequence[tuple[int, int]]) -> dict[str, Any]:
    difference = (left - right).abs()
    x_pred, y_pred = left.argmax(-1), right.argmax(-1)
    target_x = torch.tensor([(x + y) % 16 for x, y in states], dtype=torch.long, device=left.device)
    target_y = torch.tensor([y for _x, y in states], dtype=torch.long, device=left.device)
    left_ok = bool(((x_pred == target_x) & (y_pred == target_y)).all())
    rx_pred, ry_pred = right.argmax(-1), right.argmax(-1)
    right_ok = bool(((rx_pred == target_x) & (ry_pred == target_y)).all())
    return {
        "mean_abs_logit_difference": float(difference.mean().item()),
        "max_abs_logit_difference": float(difference.max().item()),
        "argmax_agreement": float((left.argmax(-1) == right.argmax(-1)).float().mean().item()),
        "left_probe_all_correct": left_ok,
        "right_probe_all_correct": right_ok,
    }


def audit(out: Path) -> dict[str, Any]:
    manifest_path = Path(diagnostic_runtime._resolve(MANIFEST, ROOT))
    manifest, accepted_report, _scope, _final, _evidence = diagnostic_runtime._validate_evidence(ROOT, manifest_path)
    device, settings = diagnostic_runtime._load_runtime_settings(manifest)
    science_report = json.loads(SCIENCE_REPORT.read_text(encoding="utf-8"))
    if science_report.get("status") != "complete" or science_report.get("accounting", {}).get("failures"):
        raise ValueError("gated science report is not complete")
    states = _states()
    source_files = {"audit_runtime": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())}, "science_report": {"path": str(SCIENCE_REPORT.relative_to(ROOT)), "sha256": _sha256(SCIENCE_REPORT)}, "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)}}
    contexts = {}
    for arm in ARMS:
        model, adapter, optimizer, identity = _load_arm(arm, manifest, accepted_report, device)
        before_model = identity["model_digest"]
        before_adapter = identity["adapter_digest"]
        pairs = []
        for spec in IDENTITY_PAIRS:
            left = _run_path(model, adapter, arm, spec["left"], device)
            right = _run_path(model, adapter, arm, spec["right"], device)
            pairs.append({
                "id": spec["id"],
                "left": list(spec["left"]),
                "right": list(spec["right"]),
                "left_length": len(spec["left"]),
                "right_length": len(spec["right"]),
                "state_pair": _tensor_metrics(left["before_probe"], right["before_probe"]),
                "left_identity_drift": _tensor_metrics(left["before_probe"], left["initial"]),
                "right_identity_drift": _tensor_metrics(right["before_probe"], right["initial"]),
                "probe": _probe_metrics(left["probe_logits"], right["probe_logits"], states),
            })
        after_model = science._digest(model.state_dict())
        after_adapter = science._digest(adapter.state_dict())
        if after_model != before_model or after_adapter != before_adapter:
            raise ValueError(f"inference mutated {arm} model or adapter")
        contexts[arm] = {"identity": identity, "forwards": len(IDENTITY_PAIRS) * 2, "state_pairs": pairs, "model_digest_preserved": True, "adapter_digest_preserved": True}
        del model, adapter, optimizer
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "AUDIT_COMPLETE",
        "scope": {"arms": list(ARMS), "identity_pairs": len(IDENTITY_PAIRS), "probe": PROBE, "states": len(states), "forwards": len(ARMS) * len(IDENTITY_PAIRS) * 2, "backwards": 0, "optimizer_steps": 0, "checkpoint_loads": len(ARMS)},
        "runtime": settings,
        "source_files": source_files,
        "checkpoint_hashes": {arm: contexts[arm]["identity"]["checkpoint_sha256"] for arm in ARMS},
        "arms": contexts,
        "limitations": ["One fixed endpoint and one trained seed", "Identity-cycle invariance is a diagnostic, not proof of a usable latent representation", "This audit does not replay training or serialize hidden tensors"],
    }
    _atomic_json(out / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = Path(args.out)
    _refuse_fresh(out)
    result = audit(out)
    print(json.dumps({"status": result["status"], "forwards": result["scope"]["forwards"], "backwards": result["scope"]["backwards"], "optimizer_steps": result["scope"]["optimizer_steps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
