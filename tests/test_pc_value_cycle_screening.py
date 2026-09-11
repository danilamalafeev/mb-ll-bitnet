from __future__ import annotations

import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_value_cycle_contract as contract
from scripts import pc_value_cycle_screening as screening


def test_registered_scope_is_small_and_held_out() -> None:
    assert screening.SCREENING_UPDATES == 64
    assert screening.TRAINING_BATCH_SIZE == 64
    assert screening.TRAINING_SUM_LENGTHS == 220
    assert screening.EVAL_STATE_COUNT == 64
    assert set(screening.EVAL_STATE_STRATA) == {"validation", "test"}
    assert screening.EVAL_PROGRAM_COUNT == 8
    assert screening.EVAL_POSITIONS_PER_PHASE_PER_ARM == 8_704
    assert screening.TOTAL_FORWARDS == 160
    assert screening.TOTAL_CASES == 10_240
    assert screening.TOTAL_POSITIONS == 62_976
    assert screening.TOTAL_NATIVE_STEPS == 503_808


def test_selection_is_predeclared_and_contains_cycle_windows() -> None:
    scope = {
        "programs": [
            {
                "id": identifier,
                "suite": "padding" if identifier.startswith("padding_") else "compositions",
                "length": len(program),
                "program": list(program),
            }
            for identifier, program in (
                ("padding_ADDADD_ADD_k4", ("ADD", "ADD", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "ADD")),
                ("padding_ADDADD_ADD_k10", ("ADD", "ADD", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "ADD")),
                ("padding_ADDADD_XOR_k4", ("ADD", "ADD", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "XOR")),
                ("padding_ADDADD_XOR_k10", ("ADD", "ADD", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "XOR")),
                ("padding_XORSWAP_ADD_k4", ("XOR", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "ADD")),
                ("padding_XORSWAP_ADD_k10", ("XOR", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "SWAP", "ADD")),
                ("composition_L12_0", ("SWAP", "ADD", "SWAP", "XOR", "XOR", "XOR", "SWAP", "SWAP", "SWAP", "ADD", "ADD", "ADD")),
                ("composition_L16_0", ("ADD", "ADD", "SWAP", "SWAP", "SWAP", "ADD", "SWAP", "ADD", "ADD", "ADD", "SWAP", "XOR", "XOR", "ADD", "SWAP", "XOR")),
            )
        ]
    }
    specs, selection, digest = screening._selection(scope)
    assert [str(spec["id"]) for spec in specs] == list(screening.EVAL_PROGRAM_IDS)
    assert selection["state_count"] == screening.EVAL_STATE_COUNT
    assert selection["evaluation_cycle_windows"] > 0
    assert selection["digest"] if "digest" in selection else digest


def test_prefix_digest_is_deterministic_and_counts_identity_windows() -> None:
    batches = []
    for index, length in enumerate(screening.TRAINING_LENGTHS):
        program = ("SWAP", "SWAP") if index == 7 else ("ADD",) * length
        batches.append([dsl.RegisterExample((row + index) % 16, (row * 3 + index) % 16, program) for row in range(screening.TRAINING_BATCH_SIZE)])
    first = screening._prefix_digest(batches)
    second = screening._prefix_digest(batches)
    assert first == second
    assert first[0]
    assert first[1] == screening.TRAINING_BATCH_SIZE
    assert first[2] == 1


def test_cycle_contract_ids_match_screening_expectation() -> None:
    assert tuple(item[0] for item in contract.IDENTITY_CYCLES) == ("SWAP2", "XOR2", "SWAP_XOR2_SWAP")
    assert screening.EXPECTED_TRAINING_CYCLE_WINDOWS == 1_225
    assert screening.EXPECTED_TRAINING_CYCLE_UPDATES == 53
    assert screening.EXPECTED_EVAL_CYCLE_WINDOWS == 12_544
