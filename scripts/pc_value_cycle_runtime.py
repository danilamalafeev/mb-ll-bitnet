"""Bounded QA for the projected-value identity-cycle contract.

This entry point reuses the accepted latent-slot endpoint loader and snapshot
format.  It adds one auxiliary projected-V cycle loss to a tiny fixed QA
fixture; it does not modify the writer, load a second model, run science, or
perform an evaluation sweep.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_runtime as accepted_qa
from scripts import pc_value_cycle_contract as contract


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_projected_value_cycle_v1" / "qa"
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_PROJECTED_VALUE_CYCLE_PROTOCOL.md"
PURE_HELPER = ROOT / "scripts" / "pc_value_cycle_contract.py"
PURE_TESTS = ROOT / "tests" / "test_pc_value_cycle_contract.py"
RUNTIME = Path(__file__).resolve()

SCHEMA = "pc_projected_value_cycle_qa_v1"
ACCOUNTING_SCHEMA = "pc_projected_value_cycle_qa_accounting_v1"
SNAPSHOT_SCHEMA = accepted_qa.SNAPSHOT_SCHEMA
QA_BATCHES = 2
QA_CASES_PER_CALL = 2
QA_CALLS = 3
QA_UPDATES = 3
QA_CASES = 6
QA_POSITIONS = 12
QA_NATIVE_STEPS = 96
QA_DESERIALIZATIONS = 3
QA_SNAPSHOT_LOADS = 1
QA_CYCLE_WINDOWS = 6
PARENT_UPDATE = accepted_qa.PARENT_UPDATE
PARENT_LABEL = accepted_qa.PARENT_LABEL
PARENT_ARM = accepted_qa.PARENT_ARM
PARENT_BRANCH = accepted_qa.PARENT_BRANCH
PURE_HELPER_SHA256 = "87eb3472df9878442143d32bcb314a36e15dddd946a774512e41f6499dc3f569"
PURE_TESTS_SHA256 = "dcc91a7b5f3f9bb716534433e86d37cb0e2d209f7367b3142e855259fa5fac90"
PROTOCOL_SHA256 = "6b14ae09f241329cfbe7449cc046c681749ce2007ca9dbfbab9291e5ffd1fadf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _source_binding(root: Path = ROOT) -> dict[str, Any]:
    """Freeze accepted loader inputs plus the new contract source bytes."""

    root = Path(root)
    accepted = accepted_qa._source_binding(root)
    paths = {
        "runtime": RUNTIME,
        "pure_helper": PURE_HELPER,
        "pure_tests": PURE_TESTS,
        "protocol": PROTOCOL,
    }
    files = dict(accepted.get("files", {}))
    for name, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        expected = {
            "pure_helper": PURE_HELPER_SHA256,
            "pure_tests": PURE_TESTS_SHA256,
            "protocol": PROTOCOL_SHA256,
        }.get(name)
        if expected and not expected.startswith("__") and digest != expected:
            raise ValueError(f"projected-value source bytes changed: {name}")
        files[name] = {"path": _relative(path, root), "sha256": digest}
    binding: dict[str, Any] = {
        "schema": "pc_projected_value_cycle_qa_source_binding_v1",
        "accepted_latent_source_binding": accepted,
        "files": files,
        "architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID,
        "loss_weight": contract.VALUE_CYCLE_LOSS_WEIGHT,
        "identity_cycles": [
            {"id": name, "program": list(program)}
            for name, program in contract.IDENTITY_CYCLES
        ],
    }
    binding["digest"] = accepted_qa.old.digest_object(binding)
    return binding


def _example_record(example: dsl.RegisterExample) -> dict[str, Any]:
    return {
        "x": int(example.x),
        "y": int(example.y),
        "program": list(example.program),
        "target_trace": [list(pair) for pair in example.targets],
    }


def _fixture(*, source_binding: Mapping[str, Any], settings: Mapping[str, Any]) -> tuple[list[list[dsl.RegisterExample]], dict[str, Any]]:
    """Return two fixed identity-cycle batches that exercise the contract."""

    batches = [
        [dsl.RegisterExample(1, 3, ("SWAP", "SWAP")), dsl.RegisterExample(2, 5, ("SWAP", "SWAP"))],
        [dsl.RegisterExample(1, 3, ("XOR", "XOR")), dsl.RegisterExample(2, 5, ("XOR", "XOR"))],
    ]
    raw = [[_example_record(example) for example in batch] for batch in batches]
    if [len(batch[0].program) for batch in batches] != [2, 2]:
        raise ValueError("projected-value QA fixture length changed")
    fixture: dict[str, Any] = {
        "schema": "pc_projected_value_cycle_qa_fixture_v1",
        "batches": raw,
        "batch_size": QA_CASES_PER_CALL,
        "batch_count": QA_BATCHES,
        "programs": [[list(example.program) for example in batch] for batch in batches],
        "target_digest": accepted_qa.old.target_digest(batches),
        "stream_digest": accepted_qa.old.batch_digest(batches),
        "source_binding_digest": source_binding["digest"],
        "runtime": dict(settings),
        "contract": {
            "architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID,
            "weight": contract.VALUE_CYCLE_LOSS_WEIGHT,
            "projected_state": "per_slot_projected_V",
            "target_detached": True,
        },
        "budget": {
            "forwards": QA_CALLS,
            "updates": QA_UPDATES,
            "cases": QA_CASES,
            "positions": QA_POSITIONS,
            "native_steps": QA_NATIVE_STEPS,
            "backwards": QA_UPDATES,
            "optimizer_updates": QA_UPDATES,
            "underlying_deserializations": QA_DESERIALIZATIONS,
        },
    }
    fixture["fixture_digest"] = accepted_qa.old.digest_object(fixture)
    return batches, fixture


def _fixture_batches(fixture: Mapping[str, Any]) -> list[list[dsl.RegisterExample]]:
    raw = fixture.get("batches")
    if not isinstance(raw, list) or len(raw) != QA_BATCHES:
        raise ValueError("projected-value QA fixture batch count changed")
    batches: list[list[dsl.RegisterExample]] = []
    for raw_batch in raw:
        if not isinstance(raw_batch, list) or len(raw_batch) != QA_CASES_PER_CALL:
            raise ValueError("projected-value QA fixture case count changed")
        batch: list[dsl.RegisterExample] = []
        for item in raw_batch:
            if not isinstance(item, Mapping):
                raise ValueError("projected-value QA fixture example is malformed")
            example = dsl.RegisterExample(int(item["x"]), int(item["y"]), tuple(str(op) for op in item["program"]))
            if item.get("target_trace") != [list(pair) for pair in example.targets]:
                raise ValueError("projected-value QA target trace changed")
            batch.append(example)
        if len({len(example.program) for example in batch}) != 1:
            raise ValueError("projected-value QA batch is not homogeneous")
        batches.append(batch)
    if accepted_qa.old.batch_digest(batches) != fixture.get("stream_digest") or accepted_qa.old.target_digest(batches) != fixture.get("target_digest"):
        raise ValueError("projected-value QA fixture digest changed")
    if accepted_qa.old.digest_object({key: fixture[key] for key in fixture if key != "fixture_digest"}) != fixture.get("fixture_digest"):
        raise ValueError("projected-value QA fixture identity changed")
    return batches


def _new_counter() -> dict[str, Any]:
    counter = accepted_qa._new_counter()
    counter["schema"] = ACCOUNTING_SCHEMA
    counter["cycle_windows"] = 0
    counter["cycle_updates"] = 0
    return counter


def _batch_tensors(batch: Sequence[dsl.RegisterExample], device: torch.device) -> tuple[torch.Tensor, ...]:
    if len(batch) != QA_CASES_PER_CALL:
        raise ValueError("projected-value QA batch must contain exactly two examples")
    length = len(batch[0].program)
    if any(len(example.program) != length for example in batch):
        raise ValueError("projected-value QA batch must be homogeneous")
    ids_x = torch.tensor([example.x for example in batch], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in batch], dtype=torch.long, device=device)
    bits = accepted_qa.accepted.signed_bit_matrix(device=device)
    x_bits, y_bits = bits[ids_x], bits[ids_y]
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in batch], dtype=torch.long, device=device)
    targets_x = torch.tensor([[target[0] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    targets_y = torch.tensor([[target[1] for target in example.targets] for example in batch], dtype=torch.long, device=device)
    return x_bits, y_bits, ops, targets_x, targets_y


def _run_update(
    *,
    model: torch.nn.Module,
    adapter: latent.LatentSlotAdapter,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[dsl.RegisterExample],
    device: torch.device,
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None] | None,
) -> tuple[float, dict[str, Any]]:
    length = len(batch[0].program)
    cases = len(batch)
    for key, amount in (
        ("attempted_updates", 1),
        ("attempted_forwards", 1),
        ("attempted_cases", cases),
        ("attempted_readout_positions", cases * length),
        ("attempted_native_steps", cases * length * latent.NATIVE_STEPS),
    ):
        counter[key] += amount
    accepted_qa._flush(counter, sink)
    optimizer.zero_grad(set_to_none=True)
    x_bits, y_bits, ops, targets_x, targets_y = _batch_tensors(batch, device)
    try:
        with accepted_qa._timed(counter, "forward", sink), torch.autocast(device_type=device.type, enabled=False):
            (logits_x, logits_y), diagnostics = latent.latent_slots_forward(
                model, adapter, x_bits, y_bits, ops, return_diagnostics=True
            )
        expected = (cases, length, latent.SLOT_WIDTH)
        if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected:
            raise ValueError("projected-value QA logits shape changed")
        if not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
            raise FloatingPointError("projected-value QA logits are nonfinite")
    except BaseException as exc:
        accepted_qa._failure(counter, "forward", exc)
        accepted_qa._flush(counter, sink)
        raise
    counter["completed_forwards"] += 1
    counter["completed_cases"] += cases
    counter["completed_readout_positions"] += cases * length
    counter["completed_native_steps"] += cases * length * latent.NATIVE_STEPS
    accepted_qa._flush(counter, sink)
    try:
        with accepted_qa._timed(counter, "loss", sink):
            supervised = latent.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            cycle_term, cycle_meta = contract.projected_value_cycle_loss(
                model, adapter, diagnostics, ops
            )
            loss = supervised + cycle_term
            if not torch.isfinite(loss):
                raise FloatingPointError("projected-value QA loss is nonfinite")
    except BaseException as exc:
        accepted_qa._failure(counter, "loss", exc)
        accepted_qa._flush(counter, sink)
        raise
    counter["cycle_windows"] += int(cycle_meta["windows"])
    counter["cycle_updates"] += 1 if cycle_meta["windows"] else 0
    counter["attempted_backwards"] += 1
    accepted_qa._flush(counter, sink)
    try:
        with accepted_qa._timed(counter, "backward", sink):
            loss.backward()
        counter["completed_backwards"] += 1
        with accepted_qa._timed(counter, "gradient_clip", sink):
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
        counter["attempted_optimizer_steps"] += 1
        with accepted_qa._timed(counter, "optimizer_step", sink):
            optimizer.step()
    except BaseException as exc:
        accepted_qa._failure(counter, "update", exc)
        accepted_qa._flush(counter, sink)
        raise
    counter["completed_optimizer_steps"] += 1
    counter["completed_updates"] += 1
    accepted_qa._flush(counter, sink)
    return float(loss.detach().item()), {
        "supervised_loss": float(supervised.detach().item()),
        "cycle": cycle_meta,
        "total_loss": float(loss.detach().item()),
    }


def _snapshot_payload(
    model: torch.nn.Module,
    adapter: latent.LatentSlotAdapter,
    optimizer: torch.optim.Optimizer,
    *,
    parent_identity: Mapping[str, Any],
    source_binding_digest: str,
    fixture_digest: str,
    manifest_digest: str,
    adapter_identity: Mapping[str, Any],
    adapter_initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return accepted_qa._snapshot_payload(
        model,
        adapter,
        optimizer,
        parent_identity=parent_identity,
        source_binding_digest=source_binding_digest,
        fixture_digest=fixture_digest,
        manifest_digest=manifest_digest,
        adapter_identity=adapter_identity,
        adapter_initialization=adapter_initialization,
        next_batch_index=1,
        local_update=1,
    )


def _run_arm(
    *,
    batches: Sequence[Sequence[dsl.RegisterExample]],
    manifest: Mapping[str, Any],
    fixture: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    device: torch.device,
    common_cuda_rng: Sequence[torch.Tensor],
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
    out: Path,
    root: Path,
) -> dict[str, Any]:
    arm_out = out / "projected_value_cycle"
    arm_out.mkdir(parents=True, exist_ok=False)
    with accepted_qa._timed(counter, "endpoint_load", sink):
        model, optimizer, parent = accepted_qa._load_parent(
            manifest=manifest, counter=counter, sink=sink, root=root
        )
    model.to(device)
    accepted_qa._optimizer_to_device(optimizer, device)
    torch.set_rng_state(parent["rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.clone() for state in common_cuda_rng])
    model.train(True)
    manifest_digest = accepted_qa.old.canonical_hash(manifest)
    parent_id = accepted_qa._parent_identity(
        parent,
        accepted_qa._parent_checkpoint(root),
        manifest_digest,
        common_cuda_rng,
    )
    adapter, initialization, adapter_identity, parent_names = accepted_qa._append_adapter_group(model, optimizer, device=device)
    adapter.train(True)
    latent.validate_optimizer_adapter_association(optimizer, adapter)
    initial_state = accepted_qa._state_identity(model, adapter, optimizer)
    loss_l1, meta_l1 = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[0], device=device, counter=counter, sink=sink)
    snapshot = _snapshot_payload(
        model,
        adapter,
        optimizer,
        parent_identity=parent_id,
        source_binding_digest=str(source_binding["digest"]),
        fixture_digest=str(fixture["fixture_digest"]),
        manifest_digest=manifest_digest,
        adapter_identity=adapter_identity,
        adapter_initialization=initialization,
    )
    snapshot_path = arm_out / "snapshot_cycle_l1.pt"
    accepted_qa._atomic_torch(snapshot_path, snapshot, refuse=True)
    loss_uninterrupted, meta_uninterrupted = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], device=device, counter=counter, sink=sink)
    uninterrupted = accepted_qa._state_identity(model, adapter, optimizer)
    counter["attempted_snapshot_loads"] += 1
    accepted_qa._flush(counter, sink)
    try:
        loaded = accepted_qa._load_snapshot(snapshot_path)
        counter["completed_snapshot_loads"] += 1
        accepted_qa._flush(counter, sink)
        snapshot_digest = accepted_qa._restore_snapshot(
            loaded,
            model,
            adapter,
            optimizer,
            parent_identity=parent_id,
            source_binding_digest=str(source_binding["digest"]),
            fixture_digest=str(fixture["fixture_digest"]),
            manifest_digest=manifest_digest,
            expected_adapter_identity=adapter_identity,
            device=device,
        )
    except BaseException as exc:
        accepted_qa._failure(counter, "snapshot_restore", exc)
        accepted_qa._flush(counter, sink)
        raise
    loss_reloaded, meta_reloaded = _run_update(model=model, adapter=adapter, optimizer=optimizer, batch=batches[1], device=device, counter=counter, sink=sink)
    reloaded = accepted_qa._state_identity(model, adapter, optimizer)
    equality = {
        key: uninterrupted[key] == reloaded[key]
        for key in (
            "model_digest",
            "adapter_digest",
            "optimizer_digest",
            "cpu_rng_digest",
            "cuda_rng_digest",
            "training_mode",
            "adapter_training_mode",
            "model_parameter_names",
            "adapter_parameter_names",
            "optimizer_group_count",
            "optimizer_group_param_names",
        )
    }
    equality["all_exact"] = all(equality.values())
    if not equality["all_exact"]:
        raise ValueError("projected-value QA reload next-update mismatch")
    record = {
        "schema": "pc_projected_value_cycle_qa_record_v1",
        "status": "complete",
        "parent_identity": parent_id,
        "adapter_identity": adapter_identity,
        "parameter_names": {"parent": parent_names, "adapter": list(latent.ADAPTER_PARAMETER_NAMES)},
        "contract": {
            "architecture_id": contract.VALUE_CYCLE_ARCHITECTURE_ID,
            "weight": contract.VALUE_CYCLE_LOSS_WEIGHT,
            "projected_state": "per_slot_projected_V",
            "target_detached": True,
        },
        "snapshot": {
            "path": _relative(snapshot_path, root),
            "sha256": _sha256(snapshot_path),
            "snapshot_digest": snapshot_digest,
            "committed": True,
            "complete": True,
        },
        "losses": [loss_l1, loss_uninterrupted, loss_reloaded],
        "loss_metadata": [meta_l1, meta_uninterrupted, meta_reloaded],
        "equality": equality,
        "calls": QA_CALLS,
        "updates": QA_UPDATES,
        "cases": QA_CASES,
        "readout_positions": QA_POSITIONS,
        "native_steps": QA_NATIVE_STEPS,
    }
    accepted_qa._atomic_json(arm_out / "report.json", record, refuse=True)
    return record


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 1,
        "completed_endpoint_loads": 1,
        "attempted_snapshot_loads": QA_SNAPSHOT_LOADS,
        "completed_snapshot_loads": QA_SNAPSHOT_LOADS,
        "attempted_forwards": QA_CALLS,
        "completed_forwards": QA_CALLS,
        "attempted_cases": QA_CASES,
        "completed_cases": QA_CASES,
        "attempted_readout_positions": QA_POSITIONS,
        "completed_readout_positions": QA_POSITIONS,
        "attempted_native_steps": QA_NATIVE_STEPS,
        "completed_native_steps": QA_NATIVE_STEPS,
        "attempted_backwards": QA_UPDATES,
        "completed_backwards": QA_UPDATES,
        "attempted_optimizer_steps": QA_UPDATES,
        "completed_optimizer_steps": QA_UPDATES,
        "completed_updates": QA_UPDATES,
        "attempted_underlying_deserializations": QA_DESERIALIZATIONS,
        "completed_underlying_deserializations": QA_DESERIALIZATIONS,
        "cycle_windows": QA_CYCLE_WINDOWS,
        "cycle_updates": QA_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"projected-value QA accounting mismatch: {key}={counter.get(key)} expected {value}")
    if counter.get("failures"):
        raise ValueError("projected-value QA recorded failures")


def run_qa(*, root: Path = ROOT, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    out = Path(out) if Path(out).is_absolute() else root / Path(out)
    accepted_qa._refuse_nonempty(out)
    source_binding = _source_binding(root)
    device, settings = accepted_qa._configure_runtime()
    batches, fixture = _fixture(source_binding=source_binding, settings=settings)
    accepted_qa._atomic_json(out / "source_binding.json", source_binding, refuse=True)
    accepted_qa._atomic_json(out / "fixture.json", fixture, refuse=True)
    counter = _new_counter()
    sink = accepted_qa._counter_sink(out / "accounting.json")
    sink(counter)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]},
        "fixture": {"path": _relative(out / "fixture.json", root), "sha256": _sha256(out / "fixture.json"), "digest": fixture["fixture_digest"]},
        "runtime": settings,
        "budget": {"forwards": QA_CALLS, "updates": QA_UPDATES, "cases": QA_CASES, "positions": QA_POSITIONS, "native_steps": QA_NATIVE_STEPS, "backwards": QA_UPDATES, "optimizer_updates": QA_UPDATES, "underlying_deserializations": QA_DESERIALIZATIONS},
    }
    try:
        # E36 checkpoint metadata was written with POSIX relative paths.
        # Keep the accepted loader unchanged and scope its existing Windows
        # compatibility adapter around the manifest/load boundary only.
        with accepted_qa.migration.windows_compatibility_adapter():
            manifest = accepted_qa.followup._load_inherited(root)
            common_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
            with accepted_qa._count_deserializations(counter, sink):
                record = _run_arm(
                    batches=batches,
                    manifest=manifest,
                    fixture=fixture,
                    source_binding=source_binding,
                    device=device,
                    common_cuda_rng=common_cuda_rng,
                    counter=counter,
                    sink=sink,
                    out=out,
                    root=root,
                )
        _validate_accounting(counter)
        report.update({"status": "complete", "arm": record, "accounting": dict(counter), "manifest_digest": accepted_qa.old.canonical_hash(manifest), "limitations": ["one endpoint and three tiny identity-cycle updates", "QA wiring evidence only; no efficacy or scientific claim"]})
        accepted_qa._atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        accepted_qa._failure(counter, "qa", exc)
        accepted_qa._flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        accepted_qa._atomic_json(out / "report.json", report, refuse=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_qa(out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
