"""Focused E17 controls; no registered 2,000-update training is run."""
from __future__ import annotations

import hashlib

import pytest
import torch

from looped_bitnet.bit_input_e17 import (
    BITS_PARAMETER_COUNT,
    ENCODER_SPEC,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    LEARNED_PARAMETER_COUNT,
    OPTIMIZER_CONFIG,
    SignedBitsEncoder,
    build_paired_models,
    common_core_digest,
    digest_state_dict,
    fixed_stream,
    paired_outcomes,
    projection_weights,
    signed_bits,
    source_hashes,
    target_digest,
)
from looped_bitnet.float_qat_e16 import build_paired_models as build_e16_models
from looped_bitnet.register_e15 import OP_TO_ID, batch_digest, loss_for_batch
from scripts import bit_input_e17 as runner


def _inputs(batch):
    x = torch.tensor([example.x for example in batch], dtype=torch.long)
    y = torch.tensor([example.y for example in batch], dtype=torch.long)
    ops = torch.tensor([[OP_TO_ID[op] for op in example.program] for example in batch], dtype=torch.long)
    return x, y, ops


def test_all16_signed_lsb_encoding_range_dtype_and_projection_reference():
    values = torch.arange(16, dtype=torch.long)
    expected = torch.tensor([[-1 if ((value >> bit) & 1) == 0 else 1 for bit in range(4)] for value in range(16)], dtype=torch.float32)
    assert torch.equal(signed_bits(values), expected)
    with pytest.raises(ValueError): signed_bits(torch.tensor([16], dtype=torch.long))
    with pytest.raises(ValueError): signed_bits(torch.tensor([-1], dtype=torch.long))
    with pytest.raises(ValueError): signed_bits(torch.tensor([1], dtype=torch.int32))
    x, y = projection_weights()
    assert hashlib.sha256(x.numpy().tobytes()).hexdigest() == "ef34bde94b39eb81e08456b03f27f839a76a16b3de32a028b2d4db4d95f659f1"
    assert hashlib.sha256(y.numpy().tobytes()).hexdigest() == "785cbc8641cc6ec906dab6ceb7a511b7b0592b7df85cdb94c80d8d5a78d41180"
    assert ENCODER_SPEC["bit_order"] == "least_significant_bit_first"


def test_learned_arm_matches_original_e16_float_bitwise_through_one_update():
    _, _, e16_learned, _, _ = build_e16_models()
    learned, _, _, _, _, _ = build_paired_models()
    batch = fixed_stream()[666]
    inputs = _inputs(batch)
    with torch.inference_mode():
        left = e16_learned(*inputs); right = learned(*inputs)
    assert torch.equal(left[0], right[0]); assert torch.equal(left[1], right[1])
    old_optimizer = torch.optim.AdamW(e16_learned.parameters(), **{key: value for key, value in OPTIMIZER_CONFIG.items()
                                                                    if key in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    new_optimizer = torch.optim.AdamW(learned.parameters(), **{key: value for key, value in OPTIMIZER_CONFIG.items()
                                                               if key in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    old_optimizer.zero_grad(set_to_none=True); new_optimizer.zero_grad(set_to_none=True)
    left_loss = loss_for_batch(e16_learned, batch); right_loss = loss_for_batch(learned, batch)
    assert torch.equal(left_loss, right_loss)
    left_loss.backward(); right_loss.backward()
    for left_parameter, right_parameter in zip(e16_learned.parameters(), learned.parameters()):
        assert left_parameter.grad is not None and right_parameter.grad is not None
        assert torch.equal(left_parameter.grad, right_parameter.grad)
    torch.nn.utils.clip_grad_norm_(e16_learned.parameters(), 1.0, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(learned.parameters(), 1.0, error_if_nonfinite=True)
    old_optimizer.step(); new_optimizer.step()
    for left_parameter, right_parameter in zip(e16_learned.parameters(), learned.parameters()):
        assert torch.equal(left_parameter, right_parameter)


def test_bits_shared_core_storage_counts_and_projection_gradients():
    learned, bits, learned_digest, bits_digest, common_digest, _ = build_paired_models()
    assert learned_digest == digest_state_dict(learned)
    assert bits_digest == digest_state_dict(bits)
    assert common_digest == common_core_digest(learned) == common_core_digest(bits)
    assert sum(parameter.numel() for parameter in learned.parameters()) == LEARNED_PARAMETER_COUNT
    assert sum(parameter.numel() for parameter in bits.parameters()) == BITS_PARAMETER_COUNT
    assert not any(module.__class__.__name__ == "BitLinear" for module in bits.modules())
    for name, value in learned.state_dict().items():
        if name.startswith(("x_embedding", "y_embedding")):
            continue
        other_name = name
        if other_name in bits.state_dict():
            assert torch.equal(value, bits.state_dict()[other_name])
            assert value.data_ptr() != bits.state_dict()[other_name].data_ptr()
    batch = fixed_stream()[666]; bits.zero_grad(set_to_none=True)
    loss = loss_for_batch(bits, batch); loss.backward()
    for projection in (bits.x_embedding.projection, bits.y_embedding.projection):
        assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    assert bits.role_keys.grad is not None and bits.role_keys.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(bits.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999),
                                  eps=1e-8, amsgrad=False, foreach=False)
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {id(parameter) for parameter in bits.parameters()}
    assert not any("x_embedding.weight" == name or "y_embedding.weight" == name for name, _ in bits.named_parameters())
    x = torch.tensor([0, 1, 15], dtype=torch.long); y = torch.tensor([2, 3, 4], dtype=torch.long)
    cache = bits.init_cache(x, y)
    with torch.inference_mode():
        assert torch.equal(cache["x_initial"], bits.x_embedding(x))
        assert torch.equal(cache["y_initial"], bits.y_embedding(y))
        assert torch.equal(cache["h"], cache["x_initial"] + cache["y_initial"])
        assert torch.equal(bits.x_embedding(x), signed_bits(x) @ bits.x_embedding.projection.weight.t())
        assert torch.equal(bits.y_embedding(y), signed_bits(y) @ bits.y_embedding.projection.weight.t())


def test_stream_and_target_digest_are_exact():
    batches = fixed_stream()
    assert batch_digest(batches) == EXPECTED_BATCH_DIGEST
    assert target_digest(batches) == EXPECTED_TARGET_DIGEST
    assert len(batches) == 2000 and all(len(batch) == 64 for batch in batches)


def test_checkpoint_round_trip_both_modes_and_tamper_guards(tmp_path):
    learned, bits, learned_digest, bits_digest, common_digest, _ = build_paired_models()
    source_map = source_hashes()
    for arm, model, initial_digest in (("learned", learned, learned_digest), ("bits", bits, bits_digest)):
        tiny_batch = fixed_stream()[666]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999),
                                      eps=1e-8, amsgrad=False, foreach=False)
        optimizer.zero_grad(set_to_none=True); loss_for_batch(model, tiny_batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        path = tmp_path / f"{arm}.pt"
        payload = runner.checkpoint_payload(model, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                            stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                            e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_map)
        runner.atomic_torch_save(path, payload)
        restored, loaded = runner.load_checkpoint(path, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                                  stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                                  e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_map)
        assert loaded["encoder_mode"] == arm
        batch = fixed_stream()[0]; inputs = _inputs(batch)
        with torch.inference_mode():
            original = model(*inputs); reloaded = restored(*inputs)
        assert torch.equal(original[0], reloaded[0]); assert torch.equal(original[1], reloaded[1])
        for field, value, label in (("arm", "bits" if arm == "learned" else "learned", "arm"),
                                    ("encoder_mode", "wrong", "encoder mode"), ("update", 1999, "update"),
                                    ("e15_manifest_hash", "wrong", "manifest hash"),
                                    ("source_hashes", {**source_map, "e17_module": "wrong"}, "source hashes"),
                                    ("initial_digest", "wrong", "initial digest"),
                                    ("common_core_digest", "wrong", "common digest"),
                                    ("config", {"schema": "wrong"}, "config"),
                                    ("protocol_hash", "wrong", "protocol hash")):
            altered = dict(payload); altered[field] = value
            altered_path = tmp_path / f"{arm}-{field}.pt"; runner.atomic_torch_save(altered_path, altered)
            with pytest.raises(ValueError, match=label):
                runner.load_checkpoint(altered_path, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                       stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                       e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_map)


def test_semantic_paired_outcomes_use_bits_and_learned_names():
    states = [[0, 0], [1, 0], [2, 0], [3, 0]]
    def rows(correct):
        return [{"program": ["ADD"], "n": 4,
                 "predictions": [{"state": state, "joint_final_correct": i in correct} for i, state in enumerate(states)]}]
    pair = paired_outcomes(rows({0, 1, 3}), rows({1, 2}))[0]
    assert pair["both_correct"] == 1 and pair["bits_only"] == 2
    assert pair["learned_only"] == 1 and pair["neither"] == 0
    assert pair["bits_correct"] - pair["learned_correct"] == pair["bits_only"] - pair["learned_only"]
    assert pair["bits_arm"] == "bits" and pair["learned_arm"] == "learned"


def test_preflight_output_refuses_nonempty_root(tmp_path):
    output = tmp_path / "preflight"; output.mkdir(); (output / "keep").write_text("keep")
    with pytest.raises(FileExistsError): runner.write_preflight(preflight=output)


def test_recovery_needs_both_final_checkpoints_and_no_report_or_training(tmp_path, monkeypatch):
    preflight = tmp_path / "preflight"
    runner.write_preflight(preflight=preflight)
    learned, bits, learned_digest, bits_digest, common_digest, _ = build_paired_models()
    output = tmp_path / "run"
    for arm, model, initial_digest in (("learned", learned, learned_digest), ("bits", bits, bits_digest)):
        arm_dir = output / f"{arm}_seed0"; arm_dir.mkdir(parents=True)
        payload = runner.checkpoint_payload(model, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                            stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                            e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_hashes())
        runner.atomic_torch_save(arm_dir / "final_u2000.pt", payload)
    monkeypatch.setattr(runner, "train_arm", lambda **kwargs: pytest.fail("recovery entered training"))
    monkeypatch.setattr(runner, "_evaluate_arm", lambda model, manifest: {"train": [], "validation": []})
    monkeypatch.setattr(runner, "paired_outcomes", lambda left, right: [])
    recovered = runner.recovery_evaluate(out=output, preflight=preflight)
    assert recovered["status"] == "recovery_eval_only"
    (output / "bits_seed0" / "final_u2000.pt").unlink()
    with pytest.raises(ValueError, match="two complete final checkpoints"):
        runner.recovery_evaluate(out=output, preflight=preflight)
