from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import pc_latent_slots_pair_carry as pair
from scripts import pc_latent_slots_pair_carry_runtime as runtime


def test_qa_fixture_is_exact_and_program_to_ops_binding_is_strict_on_cpu() -> None:
    fixture = runtime._qa_fixture(torch.device("cpu"))
    assert fixture["program"] == ("ADD", "ADD", "SWAP", "SWAP", "ADD")
    assert fixture["pairs"] == ((3, 4),)
    assert fixture["states"] == ((0, 0), (0, 1))
    assert fixture["ops"].tolist() == [[0, 0, 2, 2, 0], [0, 0, 2, 2, 0]]
    with pytest.raises(ValueError, match="program-to-ops"):
        broken = fixture["ops"].clone()
        broken[0, 2] = 1
        runtime._validate_program_ops(fixture["program"], broken)
    with pytest.raises(ValueError, match="QA program"):
        runtime._validate_program_ops(("ADD", "SWAP", "SWAP", "ADD", "ADD"), fixture["ops"])


def test_registered_qa_budget_and_accounting_require_exact_completion() -> None:
    assert (runtime.QA_CALLS, runtime.QA_CASES, runtime.QA_POSITIONS, runtime.QA_NATIVE_STEPS, runtime.QA_UPDATES) == (3, 6, 30, 240, 0)
    assert runtime.QA_DESERIALIZATIONS == 3
    assert pair.QA_ACCOUNTING == {
        "forwards": 3, "cases": 6, "positions": 30,
        "native_steps": 240, "backwards": 0, "optimizer_updates": 0,
    }
    counter = runtime._new_counter()
    writes: list[dict[str, object]] = []
    runtime._account_attempt(counter, lambda value: writes.append(deepcopy(value)))
    assert counter["attempted_forwards"] == 1
    assert counter["attempted_cases"] == 2
    assert counter["attempted_readout_positions"] == 10
    assert counter["attempted_native_steps"] == 80
    assert counter["completed_forwards"] == 0
    runtime._account_complete(counter, lambda value: writes.append(deepcopy(value)))
    assert counter["completed_forwards"] == 1
    assert counter["completed_cases"] == 2
    assert counter["completed_readout_positions"] == 10
    assert counter["completed_native_steps"] == 80
    assert counter["attempted_updates"] == counter["completed_updates"] == 0
    assert writes[-1]["completed_forwards"] == 1
    with pytest.raises(ValueError, match="accounting mismatch"):
        runtime._validate_accounting(counter)
    runtime._failure(counter, "mock_forward", RuntimeError("fixture failure"), phase="evaluation")
    assert counter["failures"][-1] == {
        "kind": "mock_forward", "phase": "evaluation",
        "type": "RuntimeError", "message": "fixture failure",
    }


def test_provenance_binding_hashes_new_sources_and_input_freeze_before_load() -> None:
    settings = {"device": "cpu", "native_steps": 8}
    manifest = {"source_inventory": {"digest": "accepted-inventory"}}
    evidence = {"endpoint": {"path": "runs/endpoint.pt", "sha256": "e" * 64}}
    binding = runtime._source_binding(
        root=runtime.ROOT, manifest=manifest, evidence=evidence, settings=settings
    )
    assert binding["schema"] == runtime.SOURCE_SCHEMA
    assert binding["files"]["pair_carry_helper"]["sha256"] == runtime._sha256(runtime.PURE_HELPER)
    assert binding["files"]["pair_carry_tests"]["sha256"] == runtime._sha256(runtime.PURE_TESTS)
    assert binding["files"]["protocol"]["path"].endswith("PC_LATENT_SLOTS_PAIR_CARRY_PROTOCOL.md")
    assert len(binding["digest"]) == 64
    fixture = runtime._qa_fixture(torch.device("cpu"))
    frozen = runtime._input_freeze(
        fixture=fixture,
        source_binding=binding,
        evidence=evidence,
        settings=settings,
        root=runtime.ROOT,
    )
    assert frozen["immutable"] is True
    assert frozen["ops"] == [[0, 0, 2, 2, 0], [0, 0, 2, 2, 0]]
    assert frozen["targets"] == [
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[1, 1], [2, 1], [1, 2], [2, 1], [3, 1]],
    ]
    assert len(frozen["digest"]) == 64


def test_output_refuses_reuse_and_logits_gate_rejects_nonfinite_or_wrong_shape(tmp_path) -> None:
    destination = tmp_path / "qa"
    runtime._refuse_fresh(destination)
    (destination / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        runtime._refuse_fresh(destination)
    good_x = torch.zeros(2, 5, 16, dtype=torch.float32)
    good_y = torch.zeros_like(good_x)
    assert runtime._validate_logits((good_x, good_y))[0].shape == (2, 5, 16)
    with pytest.raises(ValueError, match="nonfinite"):
        bad = good_x.clone()
        bad[0, 0, 0] = float("nan")
        runtime._validate_logits((bad, good_y))
    with pytest.raises(ValueError, match="shape"):
        runtime._validate_logits((torch.zeros(1, 5, 16), good_y))
