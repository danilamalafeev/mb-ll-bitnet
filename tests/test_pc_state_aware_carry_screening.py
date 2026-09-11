from __future__ import annotations

import pytest
import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_state_aware_carry_screening as screening


def _batch(length: int, size: int = 64):
    program = ("ADD",) * length
    return [dsl.RegisterExample(index % 16, (index * 3) % 16, program) for index in range(size)]


def test_screening_budget_constants_are_registered():
    assert screening.TRAINING_UPDATES == 250
    assert screening.TRAINING_BATCH_SIZE == 64
    assert len(screening.TRAINING_BATCH_LENGTHS) == 250
    assert sum(screening.TRAINING_BATCH_LENGTHS) == 871
    assert screening.CHECKPOINT_INDICES == (0, 125, 250)
    assert screening.EVAL_FORWARDS_PER_ARM == 150
    assert screening.AUDIT_FORWARDS_PER_ARM == 8
    assert screening.EVAL_POSITIONS_PER_ARM == 781_056
    assert screening.AUDIT_POSITIONS_PER_ARM == 9_216
    assert screening.EVAL_NATIVE_PER_ARM == 6_248_448
    assert screening.AUDIT_NATIVE_PER_ARM == 73_728


def test_prefix_digest_is_stable_and_binds_size_and_lengths(monkeypatch):
    monkeypatch.setattr(screening, "TRAINING_UPDATES", 2)
    monkeypatch.setattr(screening, "TRAINING_BATCH_LENGTHS", (1, 2))
    batches = [_batch(1), _batch(2)]
    first = screening._prefix_digest(batches)
    assert first == screening._prefix_digest(batches)
    with pytest.raises(ValueError, match="homogeneous"):
        screening._prefix_digest([_batch(1, 63), _batch(2)])
    with pytest.raises(ValueError, match="length changed"):
        screening._prefix_digest([_batch(2), _batch(2)])


def test_compact_spec_drops_noncontract_objects():
    spec = {"id": "p", "suite": "padding", "length": 1, "program": ("ADD",), "states": ((0, 0),), "strata": ("s",), "cell": "x", "repeat": 2, "opaque": object()}
    compact = screening._compact_spec(spec)
    assert compact == {"id": "p", "suite": "padding", "length": 1, "program": ["ADD"], "states": [[0, 0]], "strata": ["s"], "cell": "x", "repeat": 2}


def test_validate_logits_and_diagnostic_summary():
    logits = (torch.zeros(2, 3, 16), torch.zeros(2, 3, 16))
    assert screening._validate_logits(logits, 2, 3)[0].shape == (2, 3, 16)
    diagnostics = {"slot_inputs": [torch.zeros(2, 2, 16) for _ in range(3)], "slot_writes": [torch.ones(2, 2, 16) for _ in range(3)]}
    summary = screening._diagnostic_summary(diagnostics, 3)
    assert len(summary["positions"]) == 3
    with pytest.raises(ValueError, match="incomplete"):
        screening._diagnostic_summary(diagnostics, 2)


def test_accounting_rejects_partial_counter():
    with pytest.raises(ValueError, match="accounting mismatch"):
        screening._validate_accounting(screening._new_counter())
