"""Bounded E15 implementation/preflight checks; no scientific training."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from looped_bitnet.register_e15 import (
    FROZEN_SOURCE_RELATIVE_PATHS, GRURegisterModel, QATRegisterModel, OPS,
    all_programs, batch_digest, checkpoint_digest_prefix, evaluate_program,
    execute_program, forbidden, gate_report, load_checkpoint, make_paired_batches,
    parameter_report, protocol_manifest, save_checkpoint, select_secondary,
    state_split, validate_manifest_schema, write_preflight,
)


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "register_interpreter_e15", Path(__file__).parents[1] / "scripts/register_interpreter_e15.py")
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
assert _RUNNER_SPEC.loader is not None
_RUNNER_SPEC.loader.exec_module(runner)


def test_semantics_frozen_sets_schedule_and_sources():
    assert execute_program(("ADD", "XOR"), (3, 5)) == ((3 + 5) % 16 ^ 5, 5)
    assert execute_program(("SWAP",), (3, 5)) == (5, 3)
    assert len(all_programs()) == 39
    assert sum(not forbidden(p) for p in all_programs()) == 32
    manifest = protocol_manifest()
    assert len(manifest["programs"]["primary"]) == 6
    assert manifest["schedule"]["counts"] == {"1": 666, "2": 668, "3": 666}
    assert set(manifest["required_source_names"]) == set(FROZEN_SOURCE_RELATIVE_PATHS)
    manifest["source_hashes"] = {name: "frozen" for name in FROZEN_SOURCE_RELATIVE_PATHS}
    validate_manifest_schema(manifest)


def test_split_secondary_schedule_and_parameter_counts():
    split = state_split()
    assert {key: len(value) for key, value in split.items()} == {"train": 192, "validation": 32, "test": 32}
    assert len({state for values in split.values() for state in values}) == 256
    assert len(select_secondary(4, "forbidden")) == 26
    assert len(select_secondary(4, "allowed")) == 32
    assert len(select_secondary(6, "forbidden")) == 32
    assert len(select_secondary(6, "allowed")) == 32
    batches = make_paired_batches(0)
    assert len(batches) == 2000 and all(len(batch) == 64 for batch in batches)
    assert batch_digest(batches[:3]) == batch_digest(batches[:3])
    assert parameter_report() == {"qat_actual": 152768, "qat_protocol": 152768,
                                  "gru_actual": 152720, "gru_protocol": 152720}


@pytest.mark.parametrize("factory", [QATRegisterModel, GRURegisterModel])
def test_gradients_prefix_causality_and_four_step_replay(factory):
    model = factory(); x = torch.tensor([1, 3]); y = torch.tensor([5, 7])
    prefix = torch.tensor([[0, 1], [0, 1]]); suffix = torch.tensor([[2, 0], [1, 2]])
    with torch.enable_grad():
        lx, ly = model(x, y, prefix); (lx.sum() + ly.sum()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
    with torch.inference_mode():
        left = model(x, y, torch.cat((prefix, suffix), dim=1)); right = model(x, y, torch.cat((prefix, torch.flip(suffix, (1,))), dim=1))
        cache = model.init_cache(x, y); replay_x, replay_y = [], []
        for index in range(prefix.shape[1]):
            (one_x, one_y), cache = model.step(cache, prefix[:, index]); replay_x.append(one_x); replay_y.append(one_y)
    torch.testing.assert_close(left[0][:, :2], right[0][:, :2]); torch.testing.assert_close(left[1][:, :2], right[1][:, :2])
    torch.testing.assert_close(torch.stack(replay_x, 1), lx); torch.testing.assert_close(torch.stack(replay_y, 1), ly)


def test_save_load_selected_checkpoint_guards(tmp_path):
    model = GRURegisterModel(); batches = make_paired_batches(0)[:2]; path = tmp_path / "checkpoint.pt"
    digest = checkpoint_digest_prefix(batches, 1)
    save_checkpoint(path, model, seed=0, update=1, manifest_hash="manifest", batch_prefix_digest=digest, tag="selected",
                    config=runner.RUN_CONFIG, source_hashes={"s": "h"})
    restored = GRURegisterModel()
    payload = load_checkpoint(path, restored, manifest_hash="manifest", expected_update=1, expected_prefix_digest=digest,
                              expected_seed=0, expected_tag="selected", expected_config=runner.RUN_CONFIG,
                              expected_source_hashes={"s": "h"})
    assert payload["tag"] == "selected"
    assert evaluate_program(restored, ("ADD",), state_split()["test"][:4])["joint_final"] >= 0
    with pytest.raises(ValueError, match="manifest"): load_checkpoint(path, GRURegisterModel(), manifest_hash="wrong")


def test_gate_and_true_paired_outcomes_with_equal_aggregate_accuracy():
    rows = [{"program": list(program), "states": 32, "joint_final": 32 if len(program) == 1 else 31}
            for program in all_programs() if not forbidden(program)]
    assert gate_report(rows)["passed"]
    assert not gate_report(rows[:-1])["passed"]
    rows[-1]["joint_final"] = 30; assert not gate_report(rows)["passed"]
    rows[-1]["joint_final"] = 31; rows[-1]["states"] = 33; assert not gate_report(rows)["passed"]
    states = [[i // 16, i % 16] for i in range(256)]
    qpred = [{"state": state, "joint_final_correct": index < 128} for index, state in enumerate(states)]
    gpred = [{"state": state, "joint_final_correct": 64 <= index < 192} for index, state in enumerate(states)]
    qat = {"primary": [{"program": ["ADD", "XOR"], "state_subsets": {"all": {"predictions": qpred}}}]}
    gru = {"primary": [{"program": ["ADD", "XOR"], "state_subsets": {"all": {"predictions": gpred}}}]}
    paired = runner.paired_per_example(qat, gru)[0]
    assert (paired["wins"], paired["losses"], paired["ties"]) == (64, 64, 128)
    assert paired["wins"] + paired["losses"] + paired["ties"] == 256


def test_validation_final_ce_macro_by_length_and_earliest_tie(monkeypatch):
    manifest = protocol_manifest()
    monkeypatch.setattr(runner, "evaluate_program", lambda model, program, states: {
        "program": list(program), "joint_final": 32 if len(program) == 1 else 0})

    class FakeModel:
        def __call__(self, x, y, ops):
            length = ops.shape[1]
            logits = torch.zeros((len(x), length, 16)); logits[:, -1, 0] = float(length)
            return logits, logits

    result = runner._validation(FakeModel(), manifest)
    assert result["by_length"]["1"]["program_count"] == 3
    assert result["by_length"]["2"]["program_count"] == 8
    assert result["by_length"]["3"]["program_count"] == 21
    # A macro of per-length CEs, unlike an all-prefix or count-weighted average.
    assert result["macro_final_dual_ce"] != pytest.approx(sum(row["final_dual_ce"] for row in result["rows"]) / 32)
    earlier = {"update": 250, "validation": {"macro_final_joint": .5, "macro_final_dual_ce": 1.0}}
    assert not runner._better({"update": 500, "macro_final_joint": .5, "macro_final_dual_ce": 1.0}, earlier)


def test_main_preflight_and_read_only_recovery_surface(tmp_path, monkeypatch, capsys):
    preflight = tmp_path / "preflight"; assert runner.main(["--preflight-dir", str(preflight)]) == 0
    manifest = json.loads((preflight / "manifest.json").read_text())
    assert manifest["source_hashes"] == runner._source_hashes()
    out = tmp_path / "run"; out.mkdir()
    arms = [{"status": "complete", "arm": arm, "seed": seed, "selected": {"path": "unused"}}
            for seed in (0, 1, 2) for arm in ("qat", "gru")]
    (out / "report.json").write_text(json.dumps({"status": "complete", "arms": arms,
        "gate": [{"gate": {"passed": True}}, {"gate": {"passed": True}}]}))
    monkeypatch.setattr(runner, "_evaluate_checkpoint", lambda record, frozen: {"arm": record["arm"], "seed": record["seed"]})
    assert runner.main(["--eval-only", "--preflight-dir", str(preflight), "--out", str(out)]) == 0
    assert "recovery_eval_only" in capsys.readouterr().out


def test_evaluation_failure_keeps_training_complete_recoverable_evidence(tmp_path, monkeypatch):
    preflight = tmp_path / "preflight"; assert runner.main(["--preflight-dir", str(preflight)]) == 0
    out = tmp_path / "run"
    seen_rows = [{"program": list(program), "states": 32, "joint_final": 32 if len(program) == 1 else 31}
                 for program in all_programs() if not forbidden(program)]
    calls = []

    def fake_train_arm(*, arm, seed, out, manifest):
        calls.append((arm, seed))
        return {"status": "complete", "arm": arm, "seed": seed,
                "selected": {"path": "unused", "update": 250, "prefix_digest": "digest",
                             "validation": {"rows": seen_rows}}}

    monkeypatch.setattr(runner, "train_arm", fake_train_arm)
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_evaluate_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("inference fault")))
    with pytest.raises(RuntimeError, match="inference fault"):
        runner.run_experiment(out=out, preflight=preflight)
    frozen = json.loads((out / "report.json").read_text())
    assert frozen["status"] == "training_complete"
    assert len(frozen["arms"]) == 6 and all(row["status"] == "complete" for row in frozen["arms"])
    assert calls == [("qat", 0), ("gru", 0), ("qat", 1), ("gru", 1), ("qat", 2), ("gru", 2)]
    monkeypatch.setattr(runner, "train_arm", lambda **kwargs: pytest.fail("recovery trained"))
    monkeypatch.setattr(runner, "_evaluate_checkpoint", lambda record, manifest: {"arm": record["arm"], "seed": record["seed"]})
    recovered = runner.recovery_evaluate(out=out, preflight=preflight)
    assert recovered["status"] == "recovery_eval_only"
    assert len(recovered["evaluations"]) == 6
