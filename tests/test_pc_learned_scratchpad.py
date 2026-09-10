from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.pc_explicit_registers import paired_metrics
from scripts.pc_learned_scratchpad import (
    ARCHITECTURE_ID,
    ARMS,
    BATCH_SIZE,
    CHILD_UPDATES,
    EVALUATION_ACCOUNTING,
    EVALUATION_CALLS,
    EVALUATION_NATIVE_STEPS,
    EVALUATION_READOUT_POSITIONS,
    FIXED_BATCH_LENGTHS,
    NATIVE_STEPS,
    PARENT_UPDATE,
    SCIENCE_ACCOUNTING,
    TINY_QA_CASES,
    TINY_QA_CALLS,
    TINY_QA_NATIVE_STEPS,
    TINY_QA_READOUT_POSITIONS,
    TOTAL_ACCOUNTING,
    TRAINING_CASES,
    TRAINING_NATIVE_STEPS,
    TRAINING_READOUT_POSITIONS,
    cache_from_probabilities,
    child_identity,
    device_cross_entropy,
    evaluation_accounting,
    signed_bit_matrix,
    soft_register_forward,
    tiny_qa_accounting,
    training_accounting,
    validate_child_identity,
)


class _Reader(nn.Module):
    def build_kv(self, key_memory: torch.Tensor, value_memory: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"key": key_memory, "value": value_memory}


class _NonlinearNormalize(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value / (value.square().sum(dim=-1, keepdim=True).sqrt() + 1e-6)


class _Encoder(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.projection = nn.Linear(4, weight.shape[0], bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(weight)


class _ProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        x_weight = torch.zeros(4, 4)
        x_weight[0, 0] = 1.0
        x_weight[1, 1] = 2.0
        x_weight[2, 2] = 3.0
        x_weight[3, 3] = 4.0
        y_weight = torch.flip(x_weight, dims=(0,))
        self.x_embedding = _Encoder(x_weight)
        self.y_embedding = _Encoder(y_weight)
        self.role_keys = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        self.memory_norm = nn.Identity()
        self.reader = _Reader()


class _PropagationModel(_ProjectionModel):
    def step(self, cache: dict[str, torch.Tensor], opcode: torch.Tensor):
        del opcode
        batch = cache["h"].shape[0]
        logits_x = cache["h"].new_full((batch, 16), -5.0)
        logits_y = cache["h"].new_full((batch, 16), -5.0)
        logits_x[:, 2] = 5.0
        logits_y[:, 3] = 5.0
        # Returning a poisoned cache proves the caller does not reuse it.
        cache["h"].fill_(999.0)
        cache["kv"]["value"].fill_(999.0)
        return (logits_x, logits_y), cache


class _OriginalInputPoisonModel(_PropagationModel):
    def __init__(self, *, poison_on_first_step: bool) -> None:
        super().__init__()
        self.poison_on_first_step = poison_on_first_step
        self.original_x: torch.Tensor | None = None
        self.original_y: torch.Tensor | None = None
        self.step_count = 0

    def step(self, cache: dict[str, torch.Tensor], opcode: torch.Tensor):
        if self.poison_on_first_step and self.step_count == 0:
            assert self.original_x is not None and self.original_y is not None
            with torch.no_grad():
                self.original_x.fill_(15)
                self.original_y.fill_(15)
        self.step_count += 1
        return super().step(cache, opcode)


class _GradientModel(_ProjectionModel):
    def __init__(self) -> None:
        super().__init__()
        self.seen_logits: list[torch.Tensor] = []
        with torch.no_grad():
            self.x_embedding.projection.weight.zero_()
            self.x_embedding.projection.weight[0, 0] = 1.0
            self.y_embedding.projection.weight.zero_()

    def step(self, cache: dict[str, torch.Tensor], opcode: torch.Tensor):
        del opcode
        base = cache["h"][:, 0]
        zeros = base.new_zeros((base.shape[0], 14))
        logits_x = torch.cat((base[:, None], (-base)[:, None], zeros), dim=1)
        logits_y = cache["h"].new_zeros((base.shape[0], 16))
        self.seen_logits.append(logits_x)
        return (logits_x, logits_y), {"ignored": True}


def test_signed_matrix_is_lsb_first_and_has_signed_rows() -> None:
    matrix = signed_bit_matrix()
    assert matrix.shape == (16, 4)
    assert matrix.dtype is torch.float32
    assert matrix[0].tolist() == [-1.0, -1.0, -1.0, -1.0]
    assert matrix[1].tolist() == [1.0, -1.0, -1.0, -1.0]
    assert matrix[15].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_probability_matrix_and_one_hot_cache_match_independent_oracles() -> None:
    model = _ProjectionModel()
    matrix = signed_bit_matrix()
    px = torch.zeros(2, 16)
    py = torch.zeros(2, 16)
    px[0, 0] = 1.0
    px[1, 15] = 1.0
    py[0, 3] = 1.0
    py[1, 12] = 1.0
    cache = cache_from_probabilities(model, px, py)
    x_rows = matrix @ model.x_embedding.projection.weight.T
    y_rows = matrix @ model.y_embedding.projection.weight.T
    expected_x = torch.stack((x_rows[0], x_rows[15]))
    expected_y = torch.stack((y_rows[3], y_rows[12]))
    assert torch.allclose(cache["x_initial"], expected_x)
    assert torch.allclose(cache["y_initial"], expected_y)
    assert torch.allclose(cache["h"], expected_x + expected_y)
    assert torch.allclose(cache["kv"]["value"], torch.stack((expected_x, expected_y), dim=1))

    p = torch.zeros(1, 16)
    p[0, 0] = 0.5
    p[0, 3] = 0.5
    projected_rows = matrix @ model.x_embedding.projection.weight.T
    assert torch.allclose((p @ matrix) @ model.x_embedding.projection.weight.T, p @ projected_rows)
    q = torch.zeros(1, 16)
    q[0, 1] = 0.5
    q[0, 2] = 0.5
    assert torch.allclose(p @ matrix, q @ matrix)
    assert torch.allclose(
        cache_from_probabilities(model, p, p)["x_initial"],
        cache_from_probabilities(model, q, p)["x_initial"],
    )


def test_nonlinear_normalization_is_after_projected_expectation_for_role_and_value() -> None:
    model = _ProjectionModel()
    model.memory_norm = _NonlinearNormalize()
    matrix = signed_bit_matrix()
    px = torch.zeros(1, 16)
    px[0, 0] = 0.25
    px[0, 15] = 0.75
    py = torch.nn.functional.one_hot(torch.tensor([3]), num_classes=16).float()
    cache = cache_from_probabilities(model, px, py)
    expected_x = model.x_embedding.projection(px @ matrix)
    expected_y = model.y_embedding.projection(py @ matrix)
    expected_key = model.memory_norm(model.role_keys).unsqueeze(0).expand(1, -1, -1).clone()
    expected_value = model.memory_norm(torch.stack((expected_x, expected_y), dim=1))
    assert torch.allclose(cache["kv"]["key"], expected_key)
    assert torch.allclose(cache["kv"]["value"], expected_value)
    assert cache["substeps"] == 0

    discrete_rows = matrix @ model.x_embedding.projection.weight.T
    mixture_of_normalized_rows = px @ model.memory_norm(discrete_rows)
    assert not torch.allclose(cache["kv"]["value"][:, 0, :], mixture_of_normalized_rows)


def test_fresh_cache_ignores_poisoned_old_cache_and_reassigned_inputs() -> None:
    model = _ProjectionModel()
    px = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=16).float()
    py = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=16).float()
    clean = cache_from_probabilities(model, px, py)
    clean_h = clean["h"].clone()
    clean_x = clean["x_initial"].clone()
    clean_key = clean["kv"]["key"].clone()
    clean_value = clean["kv"]["value"].clone()
    original_role_keys = model.role_keys.detach().clone()
    with torch.no_grad():
        px.zero_()
        py.zero_()
    assert torch.allclose(clean["h"], clean_h)
    assert torch.allclose(clean["x_initial"], clean_x)
    fresh = cache_from_probabilities(model, px, py)
    with torch.no_grad():
        clean["h"].fill_(1234.0)
        clean["kv"]["key"].fill_(1234.0)
        clean["kv"]["value"].fill_(1234.0)
    fresh_again = cache_from_probabilities(model, px, py)
    assert torch.allclose(fresh_again["h"], fresh["h"])
    assert torch.allclose(fresh_again["kv"]["key"], clean_key)
    assert torch.allclose(fresh_again["kv"]["value"], fresh["kv"]["value"])
    assert torch.allclose(model.role_keys, original_role_keys)


def test_original_register_tensors_can_be_poisoned_inside_first_step_without_bypass() -> None:
    poisoned_model = _OriginalInputPoisonModel(poison_on_first_step=True)
    poisoned_x = torch.tensor([0], dtype=torch.long)
    poisoned_y = torch.tensor([1], dtype=torch.long)
    poisoned_model.original_x = poisoned_x
    poisoned_model.original_y = poisoned_y
    (poisoned_logits_x, poisoned_logits_y), poisoned_diagnostics = soft_register_forward(
        poisoned_model, poisoned_x, poisoned_y, [[0, 1]], return_diagnostics=True
    )

    clean_model = _OriginalInputPoisonModel(poison_on_first_step=False)
    clean_x = torch.tensor([0], dtype=torch.long)
    clean_y = torch.tensor([1], dtype=torch.long)
    clean_model.original_x = clean_x
    clean_model.original_y = clean_y
    (clean_logits_x, clean_logits_y), clean_diagnostics = soft_register_forward(
        clean_model, clean_x, clean_y, [[0, 1]], return_diagnostics=True
    )

    assert poisoned_x.tolist() == [15] and poisoned_y.tolist() == [15]
    assert torch.allclose(
        poisoned_diagnostics["cache_inputs"][1]["px"],
        poisoned_diagnostics["writes"][0]["px"],
    )
    assert torch.allclose(poisoned_logits_x, clean_logits_x)
    assert torch.allclose(poisoned_logits_y, clean_logits_y)
    assert torch.allclose(
        poisoned_diagnostics["cache_inputs"][1]["px"],
        clean_diagnostics["cache_inputs"][1]["px"],
    )


def test_soft_forward_propagates_own_writes_and_discards_returned_cache() -> None:
    model = _PropagationModel()
    x = torch.tensor([0], dtype=torch.long)
    y = torch.tensor([1], dtype=torch.long)
    (logits_x, logits_y), diagnostics = soft_register_forward(
        model, x, y, [[0, 1]], return_diagnostics=True
    )
    assert logits_x.shape == (1, 2, 16)
    assert logits_y.shape == (1, 2, 16)
    assert diagnostics["detached_writes"] is False
    assert torch.allclose(diagnostics["cache_inputs"][1]["px"], diagnostics["writes"][0]["px"])
    assert torch.argmax(diagnostics["cache_inputs"][1]["px"], dim=-1).item() == 2
    assert diagnostics["cache_inputs"][1]["h"].abs().max().item() < 100.0
    model.native_steps = 4
    with pytest.raises(ValueError, match="native8"):
        soft_register_forward(model, x, y, [[0]])


def _detached_forward(model: _GradientModel, x: torch.Tensor, y: torch.Tensor, ops: list[list[int]]):
    px = F.one_hot(x, num_classes=16).float()
    py = F.one_hot(y, num_classes=16).float()
    first = None
    final = None
    for position in range(2):
        cache = cache_from_probabilities(model, px, py)
        (logits_x, _logits_y), _ = model.step(cache, torch.tensor([ops[0][position]], dtype=torch.long))
        if first is None:
            first = logits_x
        final = logits_x
        px = F.softmax(logits_x.detach(), dim=-1)
        py = py.detach()
    assert first is not None and final is not None
    return first, final


def test_final_loss_backpropagates_to_first_write_and_detached_control_does_not() -> None:
    model = _GradientModel()
    x = torch.tensor([0], dtype=torch.long)
    y = torch.tensor([1], dtype=torch.long)
    (logits_x, _), _ = soft_register_forward(model, x, y, [[0, 1]], return_diagnostics=True)
    first = model.seen_logits[0]
    final = logits_x[:, 1, :]
    loss = F.cross_entropy(final.unsqueeze(1).transpose(1, 2), torch.tensor([[0]], dtype=torch.long))
    gradient = torch.autograd.grad(loss, first, allow_unused=True)[0]
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0

    detached_first, detached_final = _detached_forward(model, x, y, [[0, 1]])
    detached_loss = F.cross_entropy(
        detached_final.unsqueeze(1).transpose(1, 2), torch.tensor([[0]], dtype=torch.long)
    )
    detached_gradient = torch.autograd.grad(detached_loss, detached_first, allow_unused=True)[0]
    assert detached_gradient is None or detached_gradient.abs().sum().item() == 0.0


def test_device_cross_entropy_is_the_accepted_float32_bridge() -> None:
    logits_x = torch.randn(2, 3, 16, dtype=torch.float32)
    logits_y = torch.randn(2, 3, 16, dtype=torch.float32)
    targets_x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    targets_y = torch.tensor([[5, 4, 3], [2, 1, 0]], dtype=torch.long)
    expected = F.cross_entropy(logits_x.transpose(1, 2), targets_x) + F.cross_entropy(
        logits_y.transpose(1, 2), targets_y
    )
    assert torch.equal(device_cross_entropy(logits_x, logits_y, targets_x, targets_y), expected)
    with pytest.raises(ValueError, match="targets must be torch.long"):
        device_cross_entropy(logits_x, logits_y, targets_x.float(), targets_y)


def _parent_identity() -> dict[str, str]:
    return {
        "checkpoint_sha256": "a" * 64,
        "model_digest": "b" * 64,
        "optimizer_digest": "c" * 64,
        "cpu_rng_digest": "d" * 64,
        "cuda_rng_digest": "e" * 64,
    }


def test_child_identity_rejects_real_parent_arm_update_config_and_digest_mutations() -> None:
    parent = _parent_identity()
    kwargs = {
        "arm": ARMS[0],
        "local_update": CHILD_UPDATES,
        "stream_digest": "1" * 64,
        "target_digest": "2" * 64,
        "source_digest": "3" * 64,
        "config_digest": "4" * 64,
    }
    identity = child_identity(parent, **kwargs)
    assert identity["parent_update_absolute"] == PARENT_UPDATE
    assert identity["absolute_update"] == PARENT_UPDATE + CHILD_UPDATES
    assert validate_child_identity(identity, identity) == validate_child_identity(identity, identity)
    mutations = [
        {**identity, "parent": {**identity["parent"], "model_digest": "f" * 64}},
        {**identity, "arm": ARMS[1]},
        {**identity, "local_update": CHILD_UPDATES - 1},
        {**identity, "config_digest": "5" * 64},
        {**identity, "source_digest": "6" * 64},
    ]
    for changed in mutations:
        with pytest.raises(ValueError, match="mismatch"):
            validate_child_identity(changed, identity)


def test_accounting_matches_protocol_and_supports_exact_partial_prefixes() -> None:
    assert len(FIXED_BATCH_LENGTHS) == CHILD_UPDATES
    assert TRAINING_CASES == 128000
    assert TRAINING_READOUT_POSITIONS == 447744
    assert TRAINING_NATIVE_STEPS == 3581952
    qa_done = tiny_qa_accounting(TINY_QA_CALLS, TINY_QA_CALLS)
    assert qa_done["completed_cases"] == TINY_QA_CASES == 12
    assert qa_done["completed_readout_positions"] == TINY_QA_READOUT_POSITIONS == 20
    assert qa_done["completed_native_steps"] == TINY_QA_NATIVE_STEPS == 160
    partial_qa = tiny_qa_accounting(2, 1)
    assert partial_qa == {
        "attempted_calls": 2,
        "completed_calls": 1,
        "attempted_cases": 4,
        "completed_cases": 2,
        "attempted_readout_positions": 6,
        "completed_readout_positions": 2,
        "attempted_native_steps": 48,
        "completed_native_steps": 16,
        "attempted_optimizer_updates": 2,
        "completed_optimizer_updates": 1,
    }
    partial_training = training_accounting(2, 1)
    assert partial_training["attempted_cases"] == 128
    assert partial_training["completed_readout_positions"] == 64
    assert partial_training["attempted_readout_positions"] == 192
    assert partial_training["attempted_native_steps"] == 1536
    assert partial_training["completed_native_steps"] == 512
    assert FIXED_BATCH_LENGTHS[:12] == (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
    assert FIXED_BATCH_LENGTHS[-8:] == (1, 2, 3, 4, 5, 6, 1, 2)
    assert evaluation_accounting(1, 1)["completed_cases"] == 256
    assert evaluation_accounting(1, 1)["completed_readout_positions"] == 1024
    assert evaluation_accounting(1, 1)["completed_native_steps"] == 8192
    assert EVALUATION_ACCOUNTING["completed_calls"] == EVALUATION_CALLS == 69
    assert EVALUATION_READOUT_POSITIONS == 331776
    assert EVALUATION_NATIVE_STEPS == 2654208
    assert SCIENCE_ACCOUNTING == {
        "calls": 4207,
        "cases": 308992,
        "readout_positions": 1890816,
        "native_steps": 15126528,
        "optimizer_updates": 4000,
    }
    assert TOTAL_ACCOUNTING == {
        "calls": 4213,
        "cases": 309004,
        "readout_positions": 1890836,
        "native_steps": 15126688,
        "optimizer_updates": 4006,
    }
    assert NATIVE_STEPS == 8 and BATCH_SIZE == 64 and ARCHITECTURE_ID.endswith("_v1")


def test_final_and_full_trace_pairs_keep_recovery_categories_separate() -> None:
    program = ["ADD", "SWAP"]
    states = ((0, 0), (1, 1))
    baseline = [
        {"state": [0, 0], "target_trace": [[1, 0], [0, 1]], "predicted_trace": [[9, 9], [0, 1]]},
        {"state": [1, 1], "target_trace": [[2, 1], [1, 2]], "predicted_trace": [[2, 1], [1, 2]]},
    ]
    composed = [
        {"state": [0, 0], "predicted_trace": [[9, 9], [0, 1]]},
        {"state": [1, 1], "predicted_trace": [[9, 9], [1, 2]]},
    ]
    metrics = paired_metrics(baseline, composed, program, states)
    assert metrics["final"] == {
        "both_correct": 2,
        "baseline_correct_composed_wrong": 0,
        "baseline_wrong_composed_correct": 0,
        "both_wrong": 0,
    }
    assert metrics["full_trace"] == {
        "both_correct": 0,
        "baseline_correct_composed_wrong": 1,
        "baseline_wrong_composed_correct": 0,
        "both_wrong": 1,
    }
    assert metrics["final_ties"] == 2
    assert metrics["full_trace_regression"] == 1
