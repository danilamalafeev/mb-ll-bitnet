from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots_pair_carry as pair
from scripts import pc_latent_slots_pair_carry_science as science
from scripts import pc_latent_slots_padding_diagnostic_runtime as diagnostic_runtime


def _fake_scope() -> dict[str, object]:
    programs: list[dict[str, object]] = []
    for identifier, program, _roles in diagnostic_runtime.diagnostic.FOCUS_PROGRAM_MAP:
        states = [{"state": list(state), "stratum": diagnostic_runtime.diagnostic._state_stratum(state)} for state in dsl.STATE_ORDER]
        programs.append({"id": identifier, "suite": "padding", "length": len(program), "program": list(program), "states": states})
    for index in range(51):
        program = ("ADD",)
        states = [{"state": list(state), "stratum": diagnostic_runtime.diagnostic._state_stratum(state)} for state in dsl.STATE_ORDER]
        programs.append({"id": f"extra_{index:02d}", "suite": "compositions", "length": 1, "program": list(program), "states": states})
    return {"states": [list(state) for state in dsl.STATE_ORDER], "programs": programs}


def test_exact_science_budget_and_focus_schedule_fixture() -> None:
    assert (science.FOCUS_PROGRAMS, science.SCIENCE_ARMS) == (18, ("sham", "pair_carry"))
    assert (science.SCIENCE_CALLS, science.SCIENCE_CASES, science.SCIENCE_POSITIONS, science.SCIENCE_NATIVE_STEPS, science.SCIENCE_UPDATES) == (36, 9216, 258048, 2064384, 0)
    specs = science._focus_specs(_fake_scope())
    assert len(specs) == 18
    assert all(len(spec["states"]) == 256 for spec in specs)
    assert all(spec["pairs"] == pair.expected_pair_schedule(spec["program"]) for spec in specs)
    assert specs[0]["pairs"] == ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22))
    assert specs[1]["pairs"][-1] == (29, 30)


def test_science_ops_binding_and_accounting_are_strict() -> None:
    spec = science._focus_specs(_fake_scope())[0]
    fixture = science._build_inputs(spec, torch.device("cpu"))
    assert science._validate_program_ops(fixture["program"], fixture["ops"]) == fixture["program"]
    broken = fixture["ops"].clone()
    broken[0, 3] = dsl.OP_TO_ID["ADD"]
    with pytest.raises(ValueError, match="program-to-ops"):
        science._validate_program_ops(fixture["program"], broken)
    counter = science._new_counter()
    science._account_attempt(counter, len(fixture["program"]), lambda _value: None)
    science._account_complete(counter, len(fixture["program"]), lambda _value: None)
    assert counter["attempted_cases"] == counter["completed_cases"] == 256
    assert counter["attempted_readout_positions"] == counter["completed_readout_positions"] == 256 * 24
    assert counter["attempted_native_steps"] == counter["completed_native_steps"] == 256 * 24 * 8
    with pytest.raises(ValueError, match="accounting mismatch"):
        science._validate_accounting(counter)


def test_qa_acceptance_gate_rejects_mutated_artifact(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("original", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    accept = tmp_path / "qa_accept.json"
    accept.write_text(json.dumps({"status": "QA_ACCEPT", "files": {"artifact.json": digest}, "source_files": {"artifact.json": digest}}), encoding="utf-8")
    accept_digest = hashlib.sha256(accept.read_bytes()).hexdigest()
    monkeypatch.setattr(science, "QA_ACCEPT", accept)
    monkeypatch.setattr(science, "QA_ACCEPT_SHA256", accept_digest)
    validated = science._validate_qa_accept(tmp_path)
    assert validated["sha256"] == accept_digest
    artifact.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        science._validate_qa_accept(tmp_path)


def test_source_and_input_freezes_are_digest_bound() -> None:
    specs = science._focus_specs(_fake_scope())[:1]
    qa_accept = {"sha256": "a" * 64, "record": {"status": "QA_ACCEPT"}, "files": {}}
    baseline = {"digest": "b" * 64, "files": {}}
    evidence = {"endpoint": {"path": "endpoint.pt", "sha256": "c" * 64}}
    manifest = {"source_inventory": {"digest": "accepted"}}
    binding = science._source_binding(root=science.ROOT, manifest=manifest, evidence=evidence, qa_accept=qa_accept, baseline=baseline, settings={"device": "cpu"})
    frozen = science._input_freeze(specs=specs, source_binding=binding, qa_accept=qa_accept, baseline=baseline, evidence=evidence, settings={"device": "cpu"}, manifest_path=science.SCIENCE_MANIFEST, root=science.ROOT)
    assert len(binding["digest"]) == len(frozen["digest"]) == 64
    assert frozen["programs"][0]["pairs"][0] == [3, 4]
    changed = deepcopy(frozen)
    changed["programs"][0]["pairs"][0] = [4, 5]
    assert science.science._digest(changed) != frozen["digest"]


def test_refuse_overwrite_and_output_shape_gate(tmp_path) -> None:
    destination = tmp_path / "science"
    science._refuse_fresh(destination)
    (destination / "row.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        science._refuse_fresh(destination)
    good = (torch.zeros(256, 24, 16), torch.zeros(256, 24, 16))
    assert science._validate_logits(good, length=24)[0].shape == (256, 24, 16)
    with pytest.raises(ValueError, match="nonfinite"):
        broken = good[0].clone()
        broken[0, 0, 0] = float("inf")
        science._validate_logits((broken, good[1]), length=24)
