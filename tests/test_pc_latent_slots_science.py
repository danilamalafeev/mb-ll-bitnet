from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots as latent
from scripts import pc_latent_slots_science as science


class _TinyParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)


class _MockStreamExample:
    def __init__(self, index: int, length: int) -> None:
        self.index = index
        self.program = ("ADD",) * length


def _toy_training_batch() -> list[dsl.RegisterExample]:
    return [dsl.RegisterExample(index % 16, (index + 1) % 16, ("ADD",)) for index in range(science.TRAINING_BATCH_SIZE)]


def _toy_model_with_adapter() -> tuple[_TinyParent, torch.optim.Optimizer, latent.LatentSlotAdapter]:
    model = _TinyParent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    adapter, _init, _identity, _names = science.qa._append_adapter_group(model, optimizer, device=torch.device("cpu"))
    return model, optimizer, adapter


def _scope_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = [("padding", 4, 9), ("padding", 12, 9), ("padding", 16, 9), ("padding", 24, 9), ("padding", 32, 9), ("compositions", 12, 6), ("compositions", 16, 6), ("compositions", 24, 6), ("compositions", 32, 6)]
    for suite, length, count in specs:
        for index in range(count):
            program = ("ADD",) * length
            predictions = []
            for state in dsl.STATE_ORDER:
                example = dsl.RegisterExample(state[0], state[1], program)
                target = [list(pair) for pair in example.targets]
                predictions.append({
                    "state": list(state),
                    "stratum": next(name for name, values in dsl.state_split().items() if state in values),
                    "target_trace": target,
                    "predicted_trace": target,
                    "full_trace_correct": True,
                    "joint_final_correct": True,
                    "first_error": None,
                })
            rows.append({"id": f"{suite}_{length}_{index}", "suite": suite, "length": length, "program": list(program), "predictions": predictions})
    return rows


def test_exact_training_budget_cycle_and_tail() -> None:
    budget = science._training_budget(latent.FIXED_BATCH_LENGTHS)
    assert budget == {"calls": 2000, "cases": 128000, "readout_positions": 447744, "native_steps": 3581952, "optimizer_updates": 2000}
    with pytest.raises(ValueError, match="cycle or tail"):
        science._training_budget(latent.FIXED_BATCH_LENGTHS[:-1] + (3,))


def test_mocked_8000_b_stream_freezes_only_validated_training_prefix() -> None:
    lengths = science._accepted_b_stream_lengths()
    stream = [[_MockStreamExample(index, length)] * science.TRAINING_BATCH_SIZE for index, length in enumerate(lengths)]
    assert science._validate_accepted_b_stream(stream) == lengths
    selected = science._select_b_training_prefix(stream)
    assert len(selected) == science.TRAINING_UPDATES
    assert [len(batch[0].program) for batch in selected] == list(latent.FIXED_BATCH_LENGTHS)
    assert selected[0][0].index == 0 and selected[-1][0].index == science.TRAINING_UPDATES - 1
    assert stream[-1][0].index == science.ACCEPTED_B_STREAM_BATCHES - 1


def test_parent_checkpoint_binding_checks_actual_bytes_before_prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "u40000.pt"
    checkpoint.write_bytes(b"accepted parent bytes")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(science.qa, "_parent_checkpoint", lambda root: checkpoint)
    monkeypatch.setattr(science, "PARENT_CHECKPOINT_SHA256", expected)
    binding = science._validate_parent_checkpoint_bytes(tmp_path)
    assert binding["sha256"] == expected
    checkpoint.write_bytes(b"mutated parent bytes")
    with pytest.raises(ValueError, match="parent checkpoint bytes"):
        science._validate_parent_checkpoint_bytes(tmp_path)


def test_source_and_scope_tamper_reject_before_model(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"source_inventory": {"original": True}}
    monkeypatch.setattr(science, "_source_inventory", lambda root: {"changed": True})
    with pytest.raises(ValueError, match="source/evidence inventory changed"):
        science._validate_source_against_manifest(manifest, Path("."))

    scope = science._baseline_scope(_scope_rows())
    changed = deepcopy(scope)
    changed["programs"][0]["states"][0]["target_trace"][0][0] ^= 1  # type: ignore[index]
    with pytest.raises(ValueError, match="target join changed|digest changed"):
        science._validate_scope(changed)


def test_static_inventory_hashes_transitive_local_sources_and_science_tests() -> None:
    inventory = science._source_inventory()
    assert len(inventory["files"]) == 35
    assert inventory["files"]["looped_bitnet\\float_qat_e16.py"]["sha256"] == "52319d0ffcee041cbf3ac156fb39a3959460dab5fb679e6190eba416f92de7f3"
    assert inventory["files"]["scripts\\pc_latent_slots_science.py"]["sha256"] == science._sha256(Path(science.__file__))
    assert inventory["files"]["tests\\test_pc_latent_slots_science.py"]["sha256"] == science._sha256(Path(__file__))


def test_baseline_fixture_requires_exact_dsl_traces_and_perfect_saved_reference() -> None:
    rows = _scope_rows()
    normalized = science._validate_baseline_rows(rows)
    assert len(normalized) == 69
    broken = deepcopy(rows)
    broken[0]["predictions"][0]["predicted_trace"][0][0] ^= 1  # type: ignore[index]
    with pytest.raises(ValueError, match="not exact DSL"):
        science._validate_baseline_rows(broken)


def test_evaluation_context_restores_full_parent_adapter_optimizer_and_rng_state() -> None:
    torch.manual_seed(123)
    model, optimizer, adapter = _toy_model_with_adapter()
    before = science._full_state_identity(model, adapter, optimizer)
    with science._preserve_evaluation_state(model, adapter, optimizer, device=torch.device("cpu")):
        with torch.no_grad():
            model.linear.weight.add_(7.0)
            adapter.writer.bias.add_(4.0)
        torch.manual_seed(909)
    assert science._full_state_identity(model, adapter, optimizer) == before
    assert model.training is True and adapter.training is True


def test_checkpoint_indices_complete_metadata_and_next_index_are_fixed() -> None:
    parent = {"checkpoint_sha256": "a" * 64, "update": 40000}
    adapter = {"architecture_id": latent.ARCHITECTURE_ID, "parameter_names": list(latent.ADAPTER_PARAMETER_NAMES)}
    record = science._checkpoint_metadata(1000, manifest_digest="b" * 64, source_binding_digest="c" * 64, parent_identity=parent, adapter_identity=adapter, next_batch_index=1000)
    assert record["committed"] is True and record["complete"] is True
    assert record["local_update"] == 1000 and record["absolute_update"] == 41000 and record["next_batch_index"] == 1000
    with pytest.raises(ValueError, match="boundary"):
        science._checkpoint_metadata(1001, manifest_digest="b" * 64, source_binding_digest="c" * 64, parent_identity=parent, adapter_identity=adapter, next_batch_index=1001)
    with pytest.raises(ValueError, match="boundary"):
        science._checkpoint_metadata(1000, manifest_digest="b" * 64, source_binding_digest="c" * 64, parent_identity=parent, adapter_identity=adapter, next_batch_index=999)


def test_failed_training_forward_persists_attempted_phase_without_retry() -> None:
    model, optimizer, adapter = _toy_model_with_adapter()
    counter = science._new_counter()
    def fail(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("fixture forward failure")
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        science._run_update(model, adapter, optimizer, _toy_training_batch(), device=torch.device("cpu"), counter=counter, sink=lambda value: None, forward=fail)
    assert counter["attempted_updates"] == 1
    assert counter["completed_updates"] == 0
    assert counter["attempted_forwards"] == 1 and counter["completed_forwards"] == 0
    assert counter["attempted_backwards"] == 0 and counter["attempted_optimizer_steps"] == 0
    assert counter["phase_counts"]["training"]["failures"] == 1
    assert len(counter["failures"]) == 1


def test_hand_paired_join_keeps_candidate_reference_roles_and_strata() -> None:
    target = [[0, 0], [1, 1]]
    def rows(finals: list[bool]) -> list[dict[str, object]]:
        predictions = []
        for state, final in zip(((0, 0), (0, 1)), finals):
            predicted = target if final else [[0, 0], [2, 2]]
            predictions.append({"state": list(state), "stratum": "train", "target_trace": target, "predicted_trace": predicted, **science.accepted_runtime._trace_metrics(target, predicted)})
        return [{"id": "hand", "suite": "padding", "length": 2, "program": ["ADD", "ADD"], "predictions": predictions}]
    paired = science.accepted_runtime._paired_metrics(rows([True, False]), rows([False, True]), label="hand")
    assert paired["roles"] == {"left": "candidate", "right": "reference"}
    assert paired["improvements"] == {"final": 1, "full_trace": 1}
    assert paired["regressions"] == {"final": 1, "full_trace": 1}
    aggregate = science._aggregate_rows(rows([True, False]))
    assert aggregate["groups"]["all"]["cases"] == 2
    assert aggregate["groups"]["stratum:train"]["final_correct"] == 1


def test_joint_length_stratum_and_program_aggregates_keep_recovery() -> None:
    def prediction(stratum: str, final: bool, full: bool, first: int | None, recovery: int | None) -> dict[str, object]:
        return {
            "state": [0, 0], "stratum": stratum, "joint_final_correct": final,
            "full_trace_correct": full, "first_error": first,
            "first_subsequent_recovery": recovery, "recovered_final": bool(first is not None and final),
        }

    rows = [
        {"id": "padding_2", "suite": "padding", "length": 2, "program": ["ADD", "ADD"], "predictions": [prediction("train", True, False, 1, 2), prediction("validation", False, False, 1, None)]},
        {"id": "composition_3", "suite": "compositions", "length": 3, "program": ["XOR", "SWAP", "ADD"], "predictions": [prediction("test", True, True, None, None)]},
    ]
    aggregate = science._aggregate_rows(rows)
    joint = aggregate["joint_suite_length_stratum"]
    assert joint["suite:padding|length:2|stratum:train"]["cases"] == 1
    assert joint["suite:padding|length:2|stratum:train"]["final_correct"] == 1
    assert joint["suite:padding|length:2|stratum:train"]["full_trace_correct"] == 0
    assert joint["suite:padding|length:2|stratum:train"]["first_subsequent_recovery"] == {"2": 1}
    assert aggregate["heldout64_within_joint"]["suite:padding|length:2|stratum:validation"]["cases"] == 1
    assert aggregate["heldout64_within_joint"]["suite:compositions|length:3|stratum:test"]["final_correct"] == 1
    assert aggregate["per_program"]["padding_2"]["all"]["cases"] == 2
    assert aggregate["per_program"]["padding_2"]["heldout64"]["cases"] == 1
