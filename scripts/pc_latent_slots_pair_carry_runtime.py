"""Bounded QA runner for the registered latent pair-carry intervention.

``--qa`` performs exactly three read-only program forwards on the first two
states and the fixed ``ADD ADD SWAP SWAP ADD`` fixture: unhooked, sham, and
pair-carry.  It reuses the accepted diagnostic evidence gate, endpoint loader,
and evaluation-state preservation context.  Science preparation and science
execution are intentionally absent from this entry point.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_pair_carry as pair
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime
from scripts import pc_latent_slots_science as science
from scripts import pc_learned_scratchpad as accepted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pc_latent_slots_pair_carry_v1" / "qa"
SCIENCE_MANIFEST = diagnostic_runtime.SCIENCE_MANIFEST
SCIENCE_ENDPOINT = diagnostic_runtime.SCIENCE_ENDPOINT
PROTOCOL = ROOT.parent / "pc_all_docs_v1" / "project" / "results" / "PC_LATENT_SLOTS_PAIR_CARRY_PROTOCOL.md"
PURE_HELPER = ROOT / "scripts" / "pc_latent_slots_pair_carry.py"
PURE_TESTS = ROOT / "tests" / "test_pc_latent_slots_pair_carry.py"
DIAGNOSTIC_RUNTIME = Path(diagnostic_runtime.__file__).resolve()
RUNTIME = Path(__file__).resolve()

SCHEMA = "pc_latent_slots_pair_carry_qa_v1"
INPUT_SCHEMA = "pc_latent_slots_pair_carry_qa_input_freeze_v1"
SOURCE_SCHEMA = "pc_latent_slots_pair_carry_qa_source_binding_v1"
QA_PROGRAM = ("ADD", "ADD", "SWAP", "SWAP", "ADD")
QA_STATES = ((0, 0), (0, 1))
QA_PAIRS = ((3, 4),)
QA_ARMS = ("unhooked", pair.SHAM_ARM, pair.PAIR_CARRY_ARM)
QA_CALLS = 3
QA_CASES = 6
QA_POSITIONS = 30
QA_NATIVE_STEPS = 240
QA_UPDATES = 0
QA_DESERIALIZATIONS = 3


def _resolve(path: Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    return diagnostic_runtime._sha256(path)


def _atomic_json(path: Path, value: Any, *, refuse: bool = False) -> None:
    diagnostic_runtime._atomic_json(path, value, refuse=refuse)


def _refuse_fresh(path: Path) -> None:
    diagnostic_runtime._refuse_fresh(path)


def _new_counter() -> dict[str, Any]:
    # The diagnostic counter has the load/deserialization fields expected by
    # its validated endpoint loader and the phase ledger needed for partial
    # failure accounting.
    return diagnostic_runtime._new_counter()


def _flush(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None) -> None:
    diagnostic_runtime._flush(counter, sink)


def _failure(counter: dict[str, Any], kind: str, exc: BaseException, *, phase: str) -> None:
    diagnostic_runtime._failure(counter, kind, exc, phase=phase)


def _file_binding(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path)}


def _source_binding(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash reviewed inputs and new QA sources before the endpoint load."""

    paths = {
        "runtime": RUNTIME,
        "pair_carry_helper": _resolve(PURE_HELPER, root),
        "pair_carry_tests": _resolve(PURE_TESTS, root),
        "diagnostic_runtime": DIAGNOSTIC_RUNTIME,
        "protocol": _resolve(PROTOCOL, root),
        "register_e15": _resolve(Path("looped_bitnet/register_e15.py"), root),
    }
    files = {name: _file_binding(path, root) for name, path in paths.items()}
    for name, value in evidence.items():
        if isinstance(value, Mapping) and "path" in value and "sha256" in value:
            files[f"evidence_{name}"] = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("accepted science source inventory is missing")
    binding: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "files": files,
        "accepted_science_source_inventory": inventory,
        "evidence": dict(evidence),
        "runtime_settings": dict(settings),
        "endpoint_local_update": diagnostic_runtime.ENDPOINT_LOCAL_UPDATE,
        "endpoint_sha256": diagnostic_runtime.ENDPOINT_SHA256,
    }
    binding["digest"] = science._digest(binding)
    return binding


def _validate_program_ops(program: Sequence[str], ops: torch.Tensor) -> tuple[str, ...]:
    """Bind every supplied op id to the accepted DSL opcode map."""

    expected_program = tuple(program)
    if expected_program != QA_PROGRAM:
        raise ValueError("QA program changed")
    if ops.ndim != 2 or tuple(ops.shape) != (len(QA_STATES), len(expected_program)) or ops.dtype is not torch.long:
        raise ValueError("QA ops tensor shape or dtype changed")
    expected = tuple(dsl.OP_TO_ID[opcode] for opcode in expected_program)
    expected_rows = [expected for _ in QA_STATES]
    if ops.detach().cpu().tolist() != [list(row) for row in expected_rows]:
        raise ValueError("QA program-to-ops binding changed")
    return expected_program


def _qa_fixture(device: torch.device) -> dict[str, Any]:
    """Build and validate the small fixed input without model work."""

    program = tuple(QA_PROGRAM)
    schedule = pair.validate_pair_schedule(program, QA_PAIRS)
    examples = [dsl.RegisterExample(state[0], state[1], program) for state in QA_STATES]
    bits = accepted.signed_bit_matrix(device=device)
    ids_x = torch.tensor([example.x for example in examples], dtype=torch.long, device=device)
    ids_y = torch.tensor([example.y for example in examples], dtype=torch.long, device=device)
    x_bits = bits[ids_x]
    y_bits = bits[ids_y]
    ops = torch.tensor(
        [[dsl.OP_TO_ID[opcode] for opcode in example.program] for example in examples],
        dtype=torch.long,
        device=device,
    )
    _validate_program_ops(program, ops)
    return {
        "program": program,
        "pairs": schedule,
        "states": QA_STATES,
        "examples": examples,
        "x_bits": x_bits,
        "y_bits": y_bits,
        "ops": ops,
        "targets": [[list(target) for target in example.targets] for example in examples],
    }


def _input_freeze(*, fixture: Mapping[str, Any], source_binding: Mapping[str, Any], evidence: Mapping[str, Any], settings: Mapping[str, Any], root: Path) -> dict[str, Any]:
    ops = fixture["ops"]
    if not isinstance(ops, torch.Tensor):
        raise ValueError("QA fixture ops are missing")
    result: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "immutable": True,
        "program": list(fixture["program"]),
        "pairs": [list(item) for item in fixture["pairs"]],
        "states": [list(state) for state in fixture["states"]],
        "strata": [diagnostic_runtime.diagnostic._state_stratum(tuple(state)) for state in fixture["states"]],
        "ops": ops.detach().cpu().tolist(),
        "targets": fixture["targets"],
        "source_binding_digest": source_binding["digest"],
        "evidence": dict(evidence),
        "runtime": dict(settings),
        "budget": dict(pair.QA_ACCOUNTING),
        "source_root": _relative(root, root),
    }
    result["digest"] = science._digest(result)
    return result


def _account_attempt(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None = None) -> None:
    for key, amount in (
        ("attempted_forwards", 1),
        ("attempted_cases", len(QA_STATES)),
        ("attempted_readout_positions", len(QA_STATES) * len(QA_PROGRAM)),
        ("attempted_native_steps", len(QA_STATES) * len(QA_PROGRAM) * latent.NATIVE_STEPS),
    ):
        counter[key] += amount
        counter["phase_counts"]["evaluation"][key] += amount
    _flush(counter, sink)


def _account_complete(counter: dict[str, Any], sink: Callable[[dict[str, Any]], None] | None = None) -> None:
    for key, amount in (
        ("completed_forwards", 1),
        ("completed_cases", len(QA_STATES)),
        ("completed_readout_positions", len(QA_STATES) * len(QA_PROGRAM)),
        ("completed_native_steps", len(QA_STATES) * len(QA_PROGRAM) * latent.NATIVE_STEPS),
    ):
        counter[key] += amount
        counter["phase_counts"]["evaluation"][key] += amount
    _flush(counter, sink)


def _validate_accounting(counter: Mapping[str, Any]) -> None:
    expected = {
        "attempted_endpoint_loads": 1,
        "completed_endpoint_loads": 1,
        "attempted_parent_loads": 1,
        "completed_parent_loads": 1,
        "attempted_endpoint_restores": 1,
        "completed_endpoint_restores": 1,
        "attempted_underlying_deserializations": QA_DESERIALIZATIONS,
        "completed_underlying_deserializations": QA_DESERIALIZATIONS,
        "attempted_forwards": QA_CALLS,
        "completed_forwards": QA_CALLS,
        "attempted_cases": QA_CASES,
        "completed_cases": QA_CASES,
        "attempted_readout_positions": QA_POSITIONS,
        "completed_readout_positions": QA_POSITIONS,
        "attempted_native_steps": QA_NATIVE_STEPS,
        "completed_native_steps": QA_NATIVE_STEPS,
        "attempted_updates": QA_UPDATES,
        "completed_updates": QA_UPDATES,
        "attempted_backwards": QA_UPDATES,
        "completed_backwards": QA_UPDATES,
        "attempted_optimizer_steps": QA_UPDATES,
        "completed_optimizer_steps": QA_UPDATES,
    }
    for key, value in expected.items():
        if counter.get(key) != value:
            raise ValueError(f"pair-carry QA accounting mismatch: {key}={counter.get(key)} expected {value}")


def _validate_logits(logits: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(logits, tuple) or len(logits) != 2 or not all(isinstance(item, torch.Tensor) for item in logits):
        raise ValueError("QA forward returned malformed logits")
    logits_x, logits_y = logits
    expected = (len(QA_STATES), len(QA_PROGRAM), latent.SLOT_WIDTH)
    if tuple(logits_x.shape) != expected or tuple(logits_y.shape) != expected:
        raise ValueError("QA logits shape changed")
    if logits_x.dtype is not torch.float32 or logits_y.dtype is not torch.float32 or not torch.isfinite(logits_x).all() or not torch.isfinite(logits_y).all():
        raise ValueError("QA logits are malformed or nonfinite")
    return logits_x, logits_y


def _run_forward(
    *,
    arm: str,
    model: torch.nn.Module,
    adapter: latent.LatentSlotAdapter,
    fixture: Mapping[str, Any],
    device: torch.device,
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
) -> tuple[tuple[Tensor, Tensor], dict[str, Any]]:
    _account_attempt(counter, sink)
    try:
        with diagnostic_runtime._timed(counter, f"forward_{arm}", sink), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            _validate_program_ops(fixture["program"], fixture["ops"])
            if arm == "unhooked":
                logits, diagnostics = latent.latent_slots_forward(
                    model, adapter, fixture["x_bits"], fixture["y_bits"], fixture["ops"], return_diagnostics=True
                )
            else:
                logits, diagnostics = pair.pair_carry_forward(
                    model,
                    adapter,
                    fixture["x_bits"],
                    fixture["y_bits"],
                    fixture["ops"],
                    program=fixture["program"],
                    pairs=fixture["pairs"],
                    arm=arm,
                    return_diagnostics=True,
                )
        checked = _validate_logits(logits)
        if not isinstance(diagnostics, Mapping):
            raise ValueError("QA diagnostics are malformed")
    except BaseException as exc:
        _failure(counter, "forward", exc, phase="evaluation")
        _flush(counter, sink)
        raise
    _account_complete(counter, sink)
    return checked, dict(diagnostics)


def _run_interventions(
    *,
    model: torch.nn.Module,
    adapter: latent.LatentSlotAdapter,
    optimizer: torch.optim.Optimizer,
    fixture: Mapping[str, Any],
    device: torch.device,
    counter: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    before = science._full_state_identity(model, adapter, optimizer)
    outputs: dict[str, tuple[tuple[Tensor, Tensor], dict[str, Any]]] = {}
    with science._preserve_evaluation_state(model, adapter, optimizer, device=device):
        for arm in QA_ARMS:
            outputs[arm] = _run_forward(
                arm=arm, model=model, adapter=adapter, fixture=fixture,
                device=device, counter=counter, sink=sink,
            )
        unhooked, _unhooked_diagnostics = outputs["unhooked"]
        sham, sham_diagnostics = outputs[pair.SHAM_ARM]
        carried, carried_diagnostics = outputs[pair.PAIR_CARRY_ARM]
        if not torch.equal(unhooked[0], sham[0]) or not torch.equal(unhooked[1], sham[1]):
            raise ValueError("sham logits do not exactly reproduce unhooked logits")
        if not torch.equal(carried[0][..., :4, :], sham[0][..., :4, :]) or not torch.equal(carried[1][..., :4, :], sham[1][..., :4, :]):
            raise ValueError("pair-carry changed logits through pair end")
        pair_diagnostics = carried_diagnostics.get("pair_carry")
        if not isinstance(pair_diagnostics, Mapping) or pair_diagnostics.get("arm") != pair.PAIR_CARRY_ARM:
            raise ValueError("pair-carry diagnostics are missing")
        slot_inputs = carried_diagnostics.get("slot_inputs")
        slot_writes = carried_diagnostics.get("slot_writes")
        if not isinstance(slot_inputs, list) or len(slot_inputs) != len(QA_PROGRAM) or not isinstance(slot_writes, list) or len(slot_writes) != len(QA_PROGRAM):
            raise ValueError("pair-carry slot diagnostics changed")
        if not torch.equal(slot_inputs[4], slot_inputs[2]):
            raise ValueError("pair-carry restored input was not consumed by the future instruction")
        if not all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in slot_writes):
            raise ValueError("pair-carry slot writes are nonfinite")
        pair_rows = pair_diagnostics.get("pairs")
        if not isinstance(pair_rows, list) or len(pair_rows) != 1:
            raise ValueError("pair-carry pair diagnostics changed")
        if any(
            not isinstance(row, Mapping)
            or row.get("start") != 3
            or row.get("end") != 4
            or not isinstance(row.get("l2_by_case"), list)
            or len(row["l2_by_case"]) != len(QA_STATES)
            or not bool(torch.isfinite(torch.tensor(row["l2_by_case"], dtype=torch.float32)).all())
            for row in pair_rows
        ):
            raise ValueError("pair-carry raw writer displacement is nonfinite")
        no_targets = "target" not in inspect.signature(pair.pair_carry_forward).parameters
        if not no_targets:
            raise ValueError("pair-carry forward unexpectedly accepts targets")
        if len(adapter.reader._forward_pre_hooks) != 0 or len(adapter.writer._forward_hooks) != 0:
            raise ValueError("pair-carry hooks remained installed after forward")
    after = science._full_state_identity(model, adapter, optimizer)
    state_keys = (
        "model_digest", "adapter_digest", "optimizer_digest", "cpu_rng_digest", "cuda_rng_digest",
        "training_mode", "adapter_training_mode", "model_parameter_names", "adapter_parameter_names",
        "optimizer_group_count", "optimizer_group_param_names",
    )
    preservation = {key: before[key] == after[key] for key in state_keys}
    preservation["all_exact"] = all(preservation.values())
    if not preservation["all_exact"]:
        raise ValueError("pair-carry QA changed preserved model/adapter/optimizer/RNG/mode state")
    return {
        "checks": {
            "unhooked_equals_sham": True,
            "pair_carry_logits_equal_through_position4": True,
            "restored_future_input": True,
            "finite_logits_and_slot_writes": True,
            "finite_raw_writer_displacement": True,
            "hooks_removed": True,
            "no_targets_in_forward_signature": no_targets,
        },
        "state_preservation": preservation,
        "arms": list(QA_ARMS),
        "sham_pair_diagnostics": {
            "reader_calls": sham_diagnostics.get("pair_carry", {}).get("reader_calls"),
            "writer_calls": sham_diagnostics.get("pair_carry", {}).get("writer_calls"),
        },
        "pair_carry_diagnostics": {
            "pairs": pair_diagnostics.get("pairs"),
            "reader_calls": pair_diagnostics.get("reader_calls"),
            "writer_calls": pair_diagnostics.get("writer_calls"),
        },
    }


def run_qa(*, root: Path = ROOT, manifest_path: Path = SCIENCE_MANIFEST, out: Path = OUTPUT) -> dict[str, Any]:
    root = Path(root)
    manifest_path = _resolve(manifest_path, root)
    out = _resolve(out, root)
    _refuse_fresh(out)
    counter = _new_counter()
    accounting_path = out / "accounting.json"
    _atomic_json(accounting_path, counter, refuse=True)
    sink = diagnostic_runtime._sink(accounting_path)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "manifest": {"path": _relative(manifest_path, root), "sha256": None},
        "budget": dict(pair.QA_ACCOUNTING),
        "accounting": dict(counter),
    }
    model: torch.nn.Module | None = None
    adapter: latent.LatentSlotAdapter | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        manifest, science_report, _scope, _final, evidence = diagnostic_runtime._validate_evidence(root, manifest_path)
        device, settings = diagnostic_runtime._load_runtime_settings(manifest)
        source_binding = _source_binding(root=root, manifest=manifest, evidence=evidence, settings=settings)
        fixture = _qa_fixture(device)
        input_freeze = _input_freeze(fixture=fixture, source_binding=source_binding, evidence=evidence, settings=settings, root=root)
        _atomic_json(out / "source_binding.json", source_binding, refuse=True)
        _atomic_json(out / "input_freeze.json", input_freeze, refuse=True)
        report.update({
            "manifest": {"path": _relative(manifest_path, root), "sha256": _sha256(manifest_path)},
            "source_binding": {"path": _relative(out / "source_binding.json", root), "sha256": _sha256(out / "source_binding.json"), "digest": source_binding["digest"]},
            "input_freeze": {"path": _relative(out / "input_freeze.json", root), "sha256": _sha256(out / "input_freeze.json"), "digest": input_freeze["digest"]},
            "runtime": settings,
            "endpoint": evidence["endpoint"],
            "science_report": science_report.get("manifest"),
        })
        model, adapter, optimizer, _endpoint_payload = diagnostic_runtime._load_endpoint(
            manifest=manifest,
            report=science_report,
            endpoint=_resolve(SCIENCE_ENDPOINT, root),
            manifest_path=manifest_path,
            source_binding_digest=str(manifest["source_inventory"]["digest"]),
            device=device,
            counter=counter,
            sink=sink,
            root=root,
        )
        if counter["completed_endpoint_loads"] != 1 or counter["completed_underlying_deserializations"] != QA_DESERIALIZATIONS:
            raise ValueError("endpoint load did not complete at the registered QA boundary")
        checks = _run_interventions(
            model=model, adapter=adapter, optimizer=optimizer, fixture=fixture,
            device=device, counter=counter, sink=sink,
        )
        _validate_accounting(counter)
        report.update({"status": "complete", "accounting": dict(counter), "checks": checks})
        _atomic_json(out / "report.json", report, refuse=True)
        return report
    except BaseException as exc:
        phase = "evaluation" if model is not None else "load"
        _failure(counter, "qa", exc, phase=phase)
        _flush(counter, sink)
        report.update({"status": "failed", "accounting": dict(counter), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        _atomic_json(out / "report.json", report, refuse=True)
        raise
    finally:
        model = None
        adapter = None
        optimizer = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=SCIENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run_qa(manifest_path=args.manifest, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
