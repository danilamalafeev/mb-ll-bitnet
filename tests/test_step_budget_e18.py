"""Focused E18 controls; no registered 2,000-update training is run."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from looped_bitnet.bit_input_e17 import build_paired_models as build_e17_models
from looped_bitnet.step_budget_e18 import (
    ENCODER_SPEC,
    EXPECTED_BATCH_DIGEST,
    EXPECTED_MANIFEST_HASH,
    EXPECTED_TARGET_DIGEST,
    NATIVE_STEPS,
    OPTIMIZER_CONFIG,
    NativeStepRegisterModel,
    build_paired_models,
    common_core_digest,
    digest_state_dict,
    fixed_stream,
    paired_outcomes,
    source_hashes,
    target_digest,
)
from looped_bitnet.register_e15 import OP_TO_ID, batch_digest, loss_for_batch
from scripts import step_budget_e18 as runner


def _inputs(batch):
    x = torch.tensor([item.x for item in batch], dtype=torch.long)
    y = torch.tensor([item.y for item in batch], dtype=torch.long)
    ops = torch.tensor([[OP_TO_ID[op] for op in item.program] for item in batch], dtype=torch.long)
    return x, y, ops


def test_initial_states_equal_e17_bits_and_storage_is_independent():
    four, eight, initial_digest, common_digest = build_paired_models()
    _, e17_bits, _, e17_digest, e17_common, _ = build_e17_models()
    assert initial_digest == e17_digest == digest_state_dict(e17_bits)
    assert common_digest == e17_common == common_core_digest(e17_bits)
    assert NATIVE_STEPS == {"steps4": 4, "steps8": 8}
    assert sum(parameter.numel() for parameter in four.parameters()) == 151232
    assert sum(parameter.numel() for parameter in eight.parameters()) == 151232
    assert list(four.state_dict()) == list(eight.state_dict()) == list(e17_bits.state_dict())
    for name, value in four.state_dict().items():
        assert torch.equal(value, eight.state_dict()[name])
        assert torch.equal(value, e17_bits.state_dict()[name])
        assert value.data_ptr() != eight.state_dict()[name].data_ptr()
    assert isinstance(four, NativeStepRegisterModel) and isinstance(eight, NativeStepRegisterModel)
    assert four.native_steps == 4 and eight.native_steps == 8
    assert not any(module.__class__.__name__ == "BitLinear" for module in four.modules())


def test_native4_matches_e17_bits_logits_loss_gradients_and_one_update():
    four, _, _, _ = build_paired_models()
    _, e17_bits, *_ = build_e17_models()
    batch = fixed_stream()[667]  # schedule index 667 is length 2
    assert len(batch[0].program) == 2
    inputs = _inputs(batch)
    with torch.inference_mode():
        four_logits = four(*inputs); e17_logits = e17_bits(*inputs)
    assert torch.equal(four_logits[0], e17_logits[0]); assert torch.equal(four_logits[1], e17_logits[1])
    old_optimizer = torch.optim.AdamW(e17_bits.parameters(), **{key: value for key, value in OPTIMIZER_CONFIG.items()
                                                                 if key in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    new_optimizer = torch.optim.AdamW(four.parameters(), **{key: value for key, value in OPTIMIZER_CONFIG.items()
                                                            if key in {"lr", "weight_decay", "betas", "eps", "amsgrad", "foreach"}})
    old_optimizer.zero_grad(set_to_none=True); new_optimizer.zero_grad(set_to_none=True)
    old_loss = loss_for_batch(e17_bits, batch); new_loss = loss_for_batch(four, batch)
    assert torch.equal(old_loss, new_loss)
    old_loss.backward(); new_loss.backward()
    for old, new in zip(e17_bits.parameters(), four.parameters()):
        assert old.grad is not None and new.grad is not None and torch.equal(old.grad, new.grad)
    torch.nn.utils.clip_grad_norm_(e17_bits.parameters(), 1.0, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(four.parameters(), 1.0, error_if_nonfinite=True)
    old_optimizer.step(); new_optimizer.step()
    for old, new in zip(e17_bits.parameters(), four.parameters()):
        assert torch.equal(old, new)


def test_native8_explicit_unroll_hooks_counter_and_gradients():
    _, eight, _, _ = build_paired_models()
    batch = fixed_stream()[667]
    x, y, ops = _inputs(batch)
    calls = {"blocks": [], "opcode": 0, "reader": 0, "output_norm": 0}
    hooks = []
    for block_index, block in enumerate(eight.blocks):
        hooks.append(block.register_forward_hook(lambda *args, index=block_index: calls["blocks"].append(index)))
    hooks.append(eight.opcode_embedding.register_forward_hook(lambda *args: calls.__setitem__("opcode", calls["opcode"] + 1)))
    hooks.append(eight.reader.register_forward_hook(lambda *args: calls.__setitem__("reader", calls["reader"] + 1)))
    hooks.append(eight.output_norm.register_forward_hook(lambda *args: calls.__setitem__("output_norm", calls["output_norm"] + 1)))
    with torch.inference_mode():
        logits = eight(x, y, ops)
    assert calls == {"blocks": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "opcode": 16, "reader": 16, "output_norm": 2}
    calls.update({"blocks": [], "opcode": 0, "reader": 0, "output_norm": 0})
    cache2 = eight.init_cache(x, y); expected_x, expected_y = [], []
    for instruction in range(ops.shape[1]):
        expected_h = cache2["h"]
        for inner in range(8):
            expected_h = expected_h + eight.opcode_embedding(ops[:, instruction]) / (64 ** 0.5)
            expected_h = expected_h + eight.reader(expected_h, cache2["kv"])
            expected_h = eight.blocks[(cache2["substeps"] + inner) % 4](expected_h)
        cache2 = dict(cache2, h=expected_h, substeps=cache2["substeps"] + 8)
        one_x, one_y = eight.output(eight.output_norm(expected_h)); expected_x.append(one_x); expected_y.append(one_y)
    expected_logits = (torch.stack(expected_x, 1), torch.stack(expected_y, 1))
    for hook in hooks: hook.remove()
    assert torch.equal(logits[0], expected_logits[0]); assert torch.equal(logits[1], expected_logits[1])
    replay_cache = eight.init_cache(x, y)
    with torch.inference_mode():
        for instruction in range(ops.shape[1]):
            _, replay_cache = eight.step(replay_cache, ops[:, instruction])
    assert torch.equal(replay_cache["h"], cache2["h"]) and replay_cache["substeps"] == 16
    assert calls == {"blocks": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "opcode": 16, "reader": 16, "output_norm": 2}
    x_targets = torch.tensor([[item.targets[index][0] for index in range(len(item.program))] for item in batch])
    y_targets = torch.tensor([[item.targets[index][1] for index in range(len(item.program))] for item in batch])
    eight.zero_grad(set_to_none=True); direct_loss = loss_for_batch(eight, batch); direct_loss.backward()
    direct_grads = [parameter.grad.detach().clone() if parameter.grad is not None else None for parameter in eight.parameters()]
    eight.zero_grad(set_to_none=True)
    explicit_loss = F.cross_entropy(expected_logits[0].transpose(1, 2), x_targets) + F.cross_entropy(expected_logits[1].transpose(1, 2), y_targets)
    explicit_loss.backward()
    for direct, explicit in zip(direct_grads, eight.parameters()):
        if direct is None:
            assert explicit.grad is None
        else:
            assert explicit.grad is not None and torch.equal(direct, explicit.grad)
    assert eight.x_embedding.projection.weight.grad is not None and eight.x_embedding.projection.weight.grad.abs().sum() > 0
    assert eight.y_embedding.projection.weight.grad is not None and eight.y_embedding.projection.weight.grad.abs().sum() > 0
    assert eight.role_keys.grad is not None and eight.role_keys.grad.abs().sum() > 0


def test_stream_target_and_encoder_metadata_are_frozen():
    batches = fixed_stream()
    assert batch_digest(batches) == EXPECTED_BATCH_DIGEST
    assert target_digest(batches) == EXPECTED_TARGET_DIGEST
    assert len(batches) == 2000 and all(len(batch) == 64 for batch in batches)
    assert ENCODER_SPEC["bit_order"] == "least_significant_bit_first"


def test_changed_weight_checkpoint_round_trip_and_budget_guards(tmp_path):
    four, eight, initial_digest, common_digest = build_paired_models()
    for arm, model in (("steps4", four), ("steps8", eight)):
        batch = fixed_stream()[667]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        optimizer.zero_grad(set_to_none=True); loss_for_batch(model, batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        path = tmp_path / f"{arm}.pt"
        payload = runner.checkpoint_payload(model, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                            stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                            e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_hashes())
        runner.atomic_torch_save(path, payload)
        restored, loaded = runner.load_checkpoint(path, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                                  stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                                  e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_hashes())
        assert loaded["native_steps"] == (4 if arm == "steps4" else 8)
        inputs = _inputs(fixed_stream()[0])
        with torch.inference_mode():
            before = model(*inputs); after = restored(*inputs)
        assert torch.equal(before[0], after[0]); assert torch.equal(before[1], after[1])
        altered = dict(payload); altered["native_steps"] = 8 if arm == "steps4" else 4
        bad = tmp_path / f"{arm}-budget.pt"; runner.atomic_torch_save(bad, altered)
        with pytest.raises(ValueError, match="native budget"):
            runner.load_checkpoint(bad, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                   stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                   e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_hashes())


def test_semantic_paired_outcomes_use_native_budget_names_and_delta_identity():
    states = [[0, 0], [1, 0], [2, 0], [3, 0]]
    def rows(correct):
        return [{"program": ["ADD"], "n": 4,
                 "predictions": [{"state": state, "joint_final_correct": i in correct} for i, state in enumerate(states)]}]
    pair = paired_outcomes(rows({0, 1, 3}), rows({1, 2}))[0]
    assert pair["both_correct"] == 1 and pair["steps8_only"] == 2
    assert pair["steps4_only"] == 1 and pair["neither"] == 0
    assert pair["steps8_correct"] - pair["steps4_correct"] == pair["steps8_only"] - pair["steps4_only"]
    assert pair["steps8_arm"] == "steps8" and pair["steps4_arm"] == "steps4"


def test_preflight_output_refuses_nonempty_root(tmp_path):
    output = tmp_path / "preflight"; output.mkdir(); (output / "keep").write_text("keep")
    with pytest.raises(FileExistsError): runner.write_preflight(preflight=output)


def test_recovery_needs_both_final_checkpoints_without_report_or_training(tmp_path, monkeypatch):
    preflight = tmp_path / "preflight"
    runner.write_preflight(preflight=preflight)
    four, eight, initial_digest, common_digest = build_paired_models()
    output = tmp_path / "run"
    for arm, model in (("steps4", four), ("steps8", eight)):
        arm_dir = output / f"{arm}_seed0"; arm_dir.mkdir(parents=True)
        payload = runner.checkpoint_payload(model, arm=arm, initial_digest=initial_digest, common_digest=common_digest,
                                            stream_digest=EXPECTED_BATCH_DIGEST, target_stream_digest=EXPECTED_TARGET_DIGEST,
                                            e15_manifest_hash=EXPECTED_MANIFEST_HASH, source_map=source_hashes())
        runner.atomic_torch_save(arm_dir / "final_u2000.pt", payload)
    monkeypatch.setattr(runner, "train_arm", lambda **kwargs: pytest.fail("recovery entered training"))
    def stub_evaluate(model, manifest):
        rows = [{"program": ["ADD"], "n": 1, "predictions": []} for _ in range(32)]
        metrics = {group: {"rates": {metric: 0.0 for metric in ("final_joint", "final_x", "final_y", "full_trace")}}
                   for group in ("1", "2", "3", "primitives", "seen_compositions")}
        return {"train": rows, "validation": rows, "macros": {"train": metrics, "validation": metrics}}
    monkeypatch.setattr(runner, "_evaluate_arm", stub_evaluate)
    monkeypatch.setattr(runner, "paired_outcomes", lambda left, right: [])
    assert runner.recovery_evaluate(out=output, preflight=preflight)["status"] == "recovery_eval_only"
    (output / "steps8_seed0" / "final_u2000.pt").unlink()
    with pytest.raises(ValueError, match="two complete final checkpoints"):
        runner.recovery_evaluate(out=output, preflight=preflight)
