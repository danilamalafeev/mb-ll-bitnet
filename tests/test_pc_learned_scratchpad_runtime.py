from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_learned_scratchpad as pure
from scripts import pc_learned_scratchpad_runtime as runtime


def _identity() -> dict[str, object]:
    return {
        "schema": "pc_learned_scratchpad_child_identity_v1",
        "arm": "continuous_control",
        "local_update": 1,
        "absolute_update": 40001,
    }


def _batch(length: int) -> list[dsl.RegisterExample]:
    program = ("ADD",) if length == 1 else ("ADD", "SWAP")
    return [
        dsl.RegisterExample(0, 1, program),
        dsl.RegisterExample(2, 3, program),
    ]


def test_fixture_binds_first_b_l1_l2_examples_and_round_trips() -> None:
    fixture = runtime._fixture({"pure": {"sha256": "a" * 64}}, {"device": "cpu"})
    batches = runtime._fixture_batches(fixture)
    assert [len(batch[0].program) for batch in batches] == [1, 2]
    assert all(len(batch) == 2 for batch in batches)
    changed = copy.deepcopy(fixture)
    changed["batches"][0][0]["x"] = 15
    with pytest.raises(ValueError, match="target trace|stream or target digest"):
        runtime._fixture_batches(changed)


def test_parent_identity_accepts_actual_e36_payload_without_invented_label() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    payload = {
        "arm": "float",
        "branch": "B",
        "width": 128,
        "seed": 0,
        "update": 40000,
        "model_digest": runtime.width.digest_state_dict(model),
        "optimizer_digest": runtime.width.digest_object(optimizer.state_dict()),
    }
    runtime._validate_parent_identity(payload, model, optimizer)
    assert "label" not in payload
    wrong_branch = dict(payload, branch="A")
    with pytest.raises(ValueError, match="branch"):
        runtime._validate_parent_identity(wrong_branch, model, optimizer)


def test_windows_adapter_encloses_strict_loader_and_restores_on_success_and_failure() -> None:
    migration = runtime.migration
    original_wave = migration.wave._relative
    original_followup = migration.followup._relative
    original_e32_loader = migration.e32.load_manifest
    seen: list[str] = []

    def strict_loader() -> None:
        assert migration.wave._relative is migration.posix_relative
        assert migration.followup._relative is migration.posix_relative
        assert migration.e32.load_manifest is not original_e32_loader
        seen.append("loaded")

    with migration.windows_compatibility_adapter():
        strict_loader()
    assert migration.wave._relative is original_wave
    assert migration.followup._relative is original_followup
    assert migration.e32.load_manifest is original_e32_loader

    with pytest.raises(RuntimeError, match="stub failure"):
        with migration.windows_compatibility_adapter():
            strict_loader()
            raise RuntimeError("stub failure")
    assert migration.wave._relative is original_wave
    assert migration.followup._relative is original_followup
    assert migration.e32.load_manifest is original_e32_loader
    assert seen == ["loaded", "loaded"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the deterministic kernel fixture")
def test_flattened_device_cross_entropy_matches_manual_loss_and_gradient() -> None:
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[0], enabled=True):
            torch.manual_seed(41)
            device = torch.device("cuda")
            # Permutations keep [B,T,16] and [B,T] non-contiguous.
            logits_x = torch.randn(16, 3, 2, device=device).permute(2, 1, 0).requires_grad_()
            logits_y = torch.randn(16, 3, 2, device=device).permute(2, 1, 0).requires_grad_()
            targets_x = torch.randint(0, 16, (3, 2), device=device).transpose(0, 1)
            targets_y = torch.randint(0, 16, (3, 2), device=device).transpose(0, 1)
            assert not logits_x.is_contiguous() and not targets_x.is_contiguous()

            def manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                flat_logits = logits.reshape(-1, 16)
                flat_targets = targets.reshape(-1)
                return (-F.log_softmax(flat_logits, dim=-1)
                        .gather(1, flat_targets[:, None]).squeeze(1).mean())

            actual = pure.device_cross_entropy(logits_x, logits_y, targets_x, targets_y)
            expected = manual(logits_x, targets_x) + manual(logits_y, targets_y)
            assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
            actual.backward()
            actual_x_grad = logits_x.grad.detach().clone()
            actual_y_grad = logits_y.grad.detach().clone()

            ref_x = logits_x.detach().clone().requires_grad_()
            ref_y = logits_y.detach().clone().requires_grad_()
            (manual(ref_x, targets_x) + manual(ref_y, targets_y)).backward()
            assert torch.allclose(actual_x_grad, ref_x.grad, rtol=1e-6, atol=1e-7)
            assert torch.allclose(actual_y_grad, ref_y.grad, rtol=1e-6, atol=1e-7)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


def test_science_batch_gate_accepts_exact_cyclic_2000_update_budget() -> None:
    class FakeBatch:
        def __init__(self, length: int) -> None:
            self.example = SimpleNamespace(program=("ADD",) * length)

        def __len__(self) -> int:
            return pure.BATCH_SIZE

        def __getitem__(self, index: int) -> SimpleNamespace:
            if index < 0 or index >= pure.BATCH_SIZE:
                raise IndexError(index)
            return self.example

        def __iter__(self):
            return iter([self.example] * pure.BATCH_SIZE)

    batches = [FakeBatch(length) for length in pure.FIXED_BATCH_LENGTHS]
    assert runtime._validate_science_batch_shape(batches) == pure.FIXED_BATCH_LENGTHS
    assert sum(pure.FIXED_BATCH_LENGTHS) == 6996
    with pytest.raises(ValueError, match="update count"):
        runtime._validate_science_batch_shape(batches[:-1])


def test_science_manifest_digest_gate_rejects_before_any_loader(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "science_manifest.json"
    payload = {"schema": runtime.SCIENCE_SCHEMA}
    payload["manifest_digest"] = runtime._canonical_digest(payload)
    runtime._write_json(path, payload, refuse=True)
    called: list[str] = []
    monkeypatch.setattr(runtime, "_load_parent", lambda **_: called.append("load"))
    with pytest.raises(ValueError, match="source binding|accepted QA|stream|baseline|parent"):
        runtime._validate_science_manifest(path, root=runtime.ROOT, settings={})
    assert called == []


def test_science_evaluation_preserves_training_state_and_rng() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train(True)
    before = runtime._state_identity(model, optimizer)
    with runtime._preserve_evaluation_state(model, optimizer):
        assert model.training is False
        torch.rand(3, device="cuda")
    assert runtime._state_identity(model, optimizer) == before


def test_science_paired_metrics_keep_final_full_and_recovery_distinct() -> None:
    target = [[0, 0], [1, 1]]

    def row(identifier: str, predictions: list[list[list[int]]]) -> dict[str, object]:
        values = []
        for state, predicted in zip(((0, 0), (1, 1)), predictions):
            values.append({
                "state": list(state),
                "stratum": "test",
                "target_trace": target,
                "predicted_trace": predicted,
                **runtime._trace_metrics(target, predicted),
            })
        return {"id": identifier, "suite": "padding", "length": 2, "program": ["ADD", "ADD"], "predictions": values}

    initial = [row("p", [[[0, 1], [1, 1]], [[0, 0], [1, 1]]])]
    baseline = [row("p", [[[0, 1], [1, 1]], [[0, 0], [0, 0]]])]
    paired = runtime._paired_metrics(initial, baseline, label="initial_vs_baseline")
    assert paired["aggregate"]["final"]["both_correct"] == 1
    assert paired["aggregate"]["full_trace"]["both_correct"] == 0
    assert paired["aggregate"]["final"]["left_correct_right_wrong"] == 1
    assert paired["roles"] == {"left": "candidate", "right": "reference"}
    assert paired["improvements"] == {"final": 1, "full_trace": 1}
    assert paired["regressions"] == {"final": 0, "full_trace": 0}
    assert paired["programs"][0]["left_recovered_final"] == 1

    reference_good = [row("p", [[[0, 0], [1, 1]], [[0, 0], [1, 1]]])]
    introduced_error = [row("p", [[[0, 0], [0, 0]], [[0, 0], [1, 1]]])]
    converse = runtime._paired_metrics(introduced_error, reference_good, label="introduced_error")
    assert converse["improvements"] == {"final": 0, "full_trace": 0}
    assert converse["regressions"] == {"final": 1, "full_trace": 1}


def test_expected_bit_deviation_uses_true_dsl_target_bits() -> None:
    matrix = pure.signed_bit_matrix()
    probabilities_x = F.one_hot(torch.tensor([[0]]), num_classes=pure.REGISTER_WIDTH).float()
    probabilities_y = F.one_hot(torch.tensor([[0]]), num_classes=pure.REGISTER_WIDTH).float()
    target_x = torch.tensor([[1]], dtype=torch.long)
    target_y = torch.tensor([[2]], dtype=torch.long)
    own_x = matrix[probabilities_x.argmax(-1)]
    own_y = matrix[probabilities_y.argmax(-1)]
    own_argmax_error = ((probabilities_x @ matrix - own_x).abs().mean((0, 2))
                        + (probabilities_y @ matrix - own_y).abs().mean((0, 2)))
    true_target_error = runtime._target_bit_deviation(
        probabilities_x, probabilities_y, target_x, target_y, matrix
    )
    assert torch.equal(own_argmax_error, torch.zeros_like(own_argmax_error))
    assert torch.all(true_target_error > 0)
    assert true_target_error.shape == (1,)


def test_snapshot_serialization_restores_model_optimizer_rng_and_mode(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cuda_rng = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(runtime.torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in cuda_rng])
    monkeypatch.setattr(runtime.torch.cuda, "set_rng_state_all", lambda states: None)
    model = nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    model.eval()
    payload = runtime._snapshot_payload(
        model, optimizer, identity=_identity(), fixture_digest="f" * 64
    )
    path = tmp_path / "snapshot.pt"
    runtime._write_torch(path, payload, refuse=True)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored = nn.Linear(2, 3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    digest = runtime._restore_snapshot(
        loaded, restored, restored_optimizer,
        expected_identity=_identity(), expected_fixture_digest="f" * 64,
        device=torch.device("cpu"),
    )
    assert digest == payload["snapshot_digest"]
    assert runtime._state_identity(restored, restored_optimizer) == runtime._state_identity(model, optimizer)
    assert restored.training is False
    assert len(restored_optimizer.state) == len(optimizer.state)
    with pytest.raises(FileExistsError, match="overwrite"):
        runtime._write_torch(path, payload, refuse=True)


def test_snapshot_identity_and_digest_rejection_happens_before_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "get_rng_state_all", lambda: [torch.tensor([1], dtype=torch.uint8)])
    monkeypatch.setattr(runtime.torch.cuda, "set_rng_state_all", lambda states: None)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    payload = runtime._snapshot_payload(
        model, optimizer, identity=_identity(), fixture_digest="f" * 64
    )
    wrong_identity = copy.deepcopy(payload)
    wrong_identity["identity"]["arm"] = "soft_register_reset"
    with pytest.raises(ValueError, match="mismatch"):
        runtime._restore_snapshot(
            wrong_identity, model, optimizer, expected_identity=_identity(),
            expected_fixture_digest="f" * 64, device=torch.device("cpu"),
        )
    wrong_digest = copy.deepcopy(payload)
    wrong_digest["model_digest"] = "0" * 64
    with pytest.raises(ValueError, match="model digest changed"):
        runtime._restore_snapshot(
            wrong_digest, model, optimizer, expected_identity=_identity(),
            expected_fixture_digest="f" * 64, device=torch.device("cpu"),
        )


def test_forward_failure_has_no_backward_or_optimizer_attempt_and_persists_positions() -> None:
    model = nn.Linear(2, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    counter = runtime._new_counter()

    def fail_forward(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runtime._run_update(
            model, optimizer, _batch(2), arm="continuous_control",
            device=torch.device("cpu"), counter=counter, forward=fail_forward,
        )
    assert counter["attempted_updates"] == 1
    assert counter["attempted_forwards"] == 1
    assert counter["completed_forwards"] == 0
    assert counter["attempted_backwards"] == 0
    assert counter["attempted_optimizer_steps"] == 0
    assert counter["attempted_cases"] == 2
    assert counter["attempted_readout_positions"] == 4
    assert counter["attempted_native_steps"] == 32
    assert counter["completed_updates"] == 0
    assert counter["failures"][0]["kind"] == "forward"


def test_successful_update_accounts_forward_backward_optimizer_separately() -> None:
    model = nn.Linear(2, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    counter = runtime._new_counter()

    def forward(model, batch, *, arm, device):
        values = model(torch.ones(len(batch), 2, device=device)).unsqueeze(1)
        return values, values

    loss = runtime._run_update(
        model, optimizer, _batch(1), arm="continuous_control",
        device=torch.device("cpu"), counter=counter, forward=forward,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert counter["attempted_updates"] == counter["completed_updates"] == 1
    assert counter["attempted_forwards"] == counter["completed_forwards"] == 1
    assert counter["attempted_backwards"] == counter["completed_backwards"] == 1
    assert counter["attempted_optimizer_steps"] == counter["completed_optimizer_steps"] == 1
    assert counter["attempted_cases"] == counter["completed_cases"] == 2
    assert counter["attempted_readout_positions"] == counter["completed_readout_positions"] == 2
    assert counter["attempted_native_steps"] == counter["completed_native_steps"] == 16


def test_underlying_deserialization_counter_distinguishes_failed_load(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "payload.pt"
    torch.save({"value": 3}, path)
    counter = runtime._new_counter()
    with runtime._count_deserializations(counter):
        assert torch.load(path, map_location="cpu", weights_only=True)["value"] == 3
        with pytest.raises(FileNotFoundError):
            torch.load(tmp_path / "missing.pt", map_location="cpu", weights_only=True)
    assert counter["attempted_underlying_deserializations"] == 2
    assert counter["completed_underlying_deserializations"] == 1
    assert counter["failures"][0]["kind"] == "underlying_deserialize"
