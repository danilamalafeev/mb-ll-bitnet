"""Focused E16 controls; no registered 2,000-update training is run here."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from looped_bitnet.float_qat_e16 import (
    DEFAULT_E15_PREFLIGHT,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    OPTIMIZER_CONFIG,
    RUN_CONFIG,
    aggregate_metrics,
    batch_digest,
    bitlinear_inventory,
    build_paired_models,
    canonical_hash,
    checkpoint_payload,
    digest_state_dict,
    fixed_stream,
    load_checkpoint,
    load_frozen_manifest,
    paired_outcomes,
    reconstruct_training_coverage,
    replace_all_bitlinear,
    source_hashes,
    target_digest,
)
from looped_bitnet.register_e15 import loss_for_batch
from looped_bitnet.quantization import BitLinear
from scripts import float_qat_e16 as runner


def _manifest():
    return load_frozen_manifest(DEFAULT_E15_PREFLIGHT)


def test_replaces_complete_inventory_and_preserves_exact_independent_state():
    base, qat, float_model, digest, replaced = build_paired_models()
    assert len(replaced) == 14
    assert len(bitlinear_inventory(qat)) == 14
    assert bitlinear_inventory(float_model) == []
    assert sum(p.numel() for p in qat.parameters()) == 152768
    assert sum(p.numel() for p in float_model.parameters()) == 152768
    assert list(qat.state_dict()) == list(float_model.state_dict())
    for left, right in zip(qat.state_dict().values(), float_model.state_dict().values()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
    assert digest == "418d819fc563320547b46de6c432f917c9514a483a15da1b5cc85b033bbdf45c"


def test_original_qat_and_new_qat_match_logits_loss_gradients_and_one_update():
    base, qat, _, _, _ = build_paired_models()
    batch = fixed_stream()[666]  # legal length-2 seen program with two prefix targets
    with torch.no_grad():
        old_x, old_y = base(
            torch.tensor([e.x for e in batch]), torch.tensor([e.y for e in batch]),
            torch.tensor([[{"ADD": 0, "XOR": 1, "SWAP": 2}[op] for op in e.program] for e in batch]),
        )
        new_x, new_y = qat(
            torch.tensor([e.x for e in batch]), torch.tensor([e.y for e in batch]),
            torch.tensor([[{"ADD": 0, "XOR": 1, "SWAP": 2}[op] for op in e.program] for e in batch]),
        )
    assert torch.equal(old_x, new_x); assert torch.equal(old_y, new_y)
    old_opt = torch.optim.AdamW(base.parameters(), **{k: v for k, v in OPTIMIZER_CONFIG.items() if k in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    new_opt = torch.optim.AdamW(qat.parameters(), **{k: v for k, v in OPTIMIZER_CONFIG.items() if k in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    old_opt.zero_grad(set_to_none=True); new_opt.zero_grad(set_to_none=True)
    old_loss = loss_for_batch(base, batch); new_loss = loss_for_batch(qat, batch)
    assert torch.equal(old_loss, new_loss)
    old_loss.backward(); new_loss.backward()
    for left, right in zip(base.parameters(), qat.parameters()):
        assert left.grad is not None and right.grad is not None
        assert torch.equal(left.grad, right.grad)
    torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(qat.parameters(), 1.0, error_if_nonfinite=True)
    old_opt.step(); new_opt.step()
    for left, right in zip(base.parameters(), qat.parameters()):
        assert torch.equal(left, right)


def test_float_modules_are_ordinary_float32_linears():
    _, _, float_model, _, _ = build_paired_models()
    assert all(type(module) is torch.nn.Linear for module in float_model.modules()
               if isinstance(module, torch.nn.Linear) and not isinstance(module, BitLinear))
    assert bitlinear_inventory(float_model) == []
    assert all(parameter.dtype == torch.float32 for parameter in float_model.parameters())


def test_fixed_stream_target_digest_schedule_and_coverage():
    manifest = _manifest(); batches = fixed_stream()
    assert len(batches) == 2000 and all(len(batch) == 64 for batch in batches)
    assert batch_digest(batches) == EXPECTED_BATCH_DIGEST
    assert target_digest(batches) == EXPECTED_TARGET_DIGEST
    coverage = reconstruct_training_coverage(batches, manifest)
    assert coverage["draws"] == 128000
    assert coverage["combinations_exposed"] == coverage["combinations_total"] == 6144
    assert coverage["all_6144_exposed"]


def test_checkpoint_round_trip_both_arms_and_strict_provenance(tmp_path):
    manifest = _manifest(); _, qat, float_model, initial, _ = build_paired_models()
    batches = fixed_stream(); sources = source_hashes()
    for arm, model in (("qat", qat), ("float", float_model)):
        path = tmp_path / f"{arm}.pt"
        payload = checkpoint_payload(model, arm=arm, initial_digest=initial,
                                     stream_digest=EXPECTED_BATCH_DIGEST, prefix_digest=EXPECTED_BATCH_DIGEST,
                                     e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=sources)
        runner.atomic_torch_save(path, payload)
        restored, loaded = load_checkpoint(path, arm=arm, initial_digest=initial,
                                           stream_digest=EXPECTED_BATCH_DIGEST, e15_manifest_hash=EXPECTED_MANIFEST_HASH,
                                           source_map=sources)
        assert loaded["arm"] == arm and digest_state_dict(restored) == loaded["model_digest"]
        assert bitlinear_inventory(restored) == (bitlinear_inventory(model))
        for left, right in zip(model.parameters(), restored.parameters()):
            assert torch.equal(left, right)
        batch = fixed_stream()[0]
        x = torch.tensor([item.x for item in batch]); y = torch.tensor([item.y for item in batch])
        op_ids = torch.tensor([[{"ADD": 0, "XOR": 1, "SWAP": 2}[op] for op in item.program] for item in batch])
        with torch.inference_mode():
            original_logits = model(x, y, op_ids)
            restored_logits = restored(x, y, op_ids)
        assert torch.equal(original_logits[0], restored_logits[0])
        assert torch.equal(original_logits[1], restored_logits[1])
        with pytest.raises(ValueError):
            load_checkpoint(path, arm="float" if arm == "qat" else "qat", initial_digest=initial,
                            stream_digest=EXPECTED_BATCH_DIGEST, e15_manifest_hash=EXPECTED_MANIFEST_HASH,
                            source_map=sources)
        tampered = dict(payload); tampered["prefix_stream_digest"] = "wrong"
        bad = tmp_path / f"{arm}-bad.pt"; runner.atomic_torch_save(bad, tampered)
        with pytest.raises(ValueError, match="prefix stream"):
            load_checkpoint(bad, arm=arm, initial_digest=initial, stream_digest=EXPECTED_BATCH_DIGEST,
                            e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=sources)
        for field, value, label in (("update", 1999, "update"), ("e15_manifest_hash", "wrong", "manifest"),
                                    ("optimizer", {**OPTIMIZER_CONFIG, "lr": 0.002}, "optimizer")):
            altered = dict(payload); altered[field] = value
            altered_path = tmp_path / f"{arm}-{field}.pt"; runner.atomic_torch_save(altered_path, altered)
            with pytest.raises(ValueError, match=label):
                load_checkpoint(altered_path, arm=arm, initial_digest=initial,
                                stream_digest=EXPECTED_BATCH_DIGEST, e15_manifest_hash=EXPECTED_MANIFEST_HASH,
                                source_map=sources)


def test_paired_outcomes_keep_asymmetric_counts_and_identity():
    states = [[i, 0] for i in range(4)]
    def rows(correct):
        return [{"program": ["ADD"], "n": 4,
                 "predictions": [{"state": state, "joint_final_correct": i in correct} for i, state in enumerate(states)]}]
    paired = paired_outcomes(rows({0, 1}), rows({1, 2}))[0]
    assert paired["both_correct"] == 1 and paired["float_only"] == 1
    assert paired["qat_only"] == 1 and paired["neither"] == 1
    assert paired["float_correct"] - paired["qat_correct"] == paired["float_only"] - paired["qat_only"]
    assert sum(paired[key] for key in ("both_correct", "float_only", "qat_only", "neither")) == 4


def test_macro_counts_are_unweighted_and_scope_is_manifest_bound():
    def row(program, n, count):
        return {"program": program, "n": n, "final_joint": count, "final_x": count,
                "final_y": count, "full_trace": count}
    programs = [list(p) for p in _manifest()["programs"]["seen"]]
    train = [row(p, 192, 192 if len(p) == 1 else 96 if len(p) == 2 else 0) for p in programs]
    validation = [row(p, 32, 32 if len(p) < 3 else 0) for p in programs]
    report = aggregate_metrics(train, validation)
    assert report["train"]["2"]["rates"]["final_joint"] == pytest.approx(.5)
    assert report["train"]["seen_compositions"]["rates"]["final_joint"] == pytest.approx(.25)
    with pytest.raises(ValueError):
        aggregate_metrics(train[:-1], validation)


def test_preflight_refuses_nonempty_output_and_never_trains(tmp_path, monkeypatch):
    out = tmp_path / "preflight"; out.mkdir(); (out / "keep").write_text("x")
    with pytest.raises(FileExistsError):
        runner.write_preflight(preflight=out)
