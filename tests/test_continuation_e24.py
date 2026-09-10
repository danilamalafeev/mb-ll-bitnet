from copy import deepcopy
import json

import pytest
import torch

from looped_bitnet import continuation_e24 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet import replication_e22 as e22
from looped_bitnet.register_e15 import RegisterExample
from scripts import continuation_e24 as r


def _qa_manifest():
    return {
        "schema": e.SCHEMA,
        "config": dict(e.CONFIG),
        "source_hashes": {"qa": "synthetic"},
    }


def _parent_factory(seed):
    def factory():
        model = e22.build_initial_model(seed)
        optimizer = old.make_optimizer(model)
        model.train(True)
        rng = torch.get_rng_state().clone()
        parent = {
            "schema": "qa_parent_v1",
            "seed": seed,
            "update": 0,
            "checkpoint_sha256": f"qa-parent-{seed}",
            "model_digest": old.digest_state_dict(model),
            "optimizer_digest": old.digest_object(optimizer.state_dict()),
            "rng_state": rng,
            "rng_digest": old.digest_object(rng),
        }
        return model, optimizer, parent
    return factory


def _tiny_spec(*, update=r.accepted_update, evaluate=None, checkpoint_updates=(2, 4)):
    batches = [
        [RegisterExample(0, 1, ("ADD",))],
        [RegisterExample(2, 3, ("XOR",))],
        [RegisterExample(4, 5, ("ADD",))],
        [RegisterExample(6, 7, ("XOR",))],
    ]
    if evaluate is None:
        def evaluate(model, manifest):
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    output = model(torch.tensor([0]), torch.tensor([1]), torch.tensor([[0]]))
                assert all(torch.isfinite(value).all() for value in output)
            finally:
                model.train(was_training)
            return {"qa_evaluation": True, "qa_predicates": {"combined_conjunction": True}}
    return r._TinySpec(
        parents={seed: _parent_factory(seed) for seed in e.SEEDS},
        batches=batches,
        evaluate=evaluate,
        update=update,
        progress_interval=1,
        checkpoint_updates=checkpoint_updates,
        parent_update=0,
        evaluation_cost={"program_state_cases": 1, "readout_positions": 1, "internal_state_substeps": 8, "program_forwards": 1},
    )


def test_fixed_manifest_and_parent_reference_are_metadata_only():
    manifest = r.expected_manifest()
    assert manifest["schema"] == e.SCHEMA
    assert manifest["config"]["added_updates"] == 8000
    assert manifest["config"]["final_update"] == 16000
    assert manifest["protected_file_count"] == 231
    assert set(manifest["parents"]) == {"0", "1", "2"}
    assert manifest["evaluation_cost"] == e.EVAL_COST_TOTAL


def test_three_seed_qa_resume_reload_next_update_and_strict_rejection(tmp_path):
    manifest = _qa_manifest()
    out = tmp_path / "run"
    report = r._train(out, manifest, _qa=_tiny_spec())
    assert report["status"] == "complete"
    assert report["cost"]["actual_added"]["updates"] == 12
    assert report["cost"]["actual_added"]["examples"] == 12
    assert report["cost"]["actual_added"]["internal_state_substeps"] == 96
    for seed in e.SEEDS:
        record = report["seeds"][str(seed)]
        assert record["status"] == "complete"
        assert record["added_updates"] == 4
        assert (out / f"seed{seed}" / "u2.pt").exists()
        assert (out / f"seed{seed}" / "u4.pt").exists()

    parent_loader = lambda seed: _parent_factory(seed)()
    qa_manifest = json.loads((out / "manifest.json").read_text())
    restored, optimizer, payload = e.load_checkpoint(
        out / "seed0" / "u2.pt", qa_manifest,
        seed=0, expected_update=2, qa=True, parent_loader=parent_loader,
    )
    batch_tail = _tiny_spec().batches[2:]
    for batch in batch_tail:
        r.accepted_update(restored, optimizer, batch)
    final_payload = torch.load(out / "seed0" / "u4.pt", weights_only=True)
    assert old.digest_state_dict(restored) == final_payload["model_digest"]
    assert old.digest_object(optimizer.state_dict()) == final_payload["optimizer_digest"]
    assert old.digest_object(torch.get_rng_state()) == final_payload["rng_digest"]
    assert payload["added_updates"] == 2 and final_payload["added_updates"] == 4

    bad_manifest = qa_manifest
    for key, value in (("schema", "bad"), ("seed", 2), ("update", 3),
                       ("parent_checkpoint_sha256", "bad"), ("config_hash", "bad"),
                       ("parent_update", 8), ("cumulative_updates", 8),
                       ("source_hashes", {}), ("training_mode", False)):
        corrupt = deepcopy(final_payload)
        corrupt[key] = value
        corrupt_path = tmp_path / f"bad-{key}.pt"
        torch.save(corrupt, corrupt_path)
        with pytest.raises(ValueError):
            e.load_checkpoint(corrupt_path, bad_manifest, seed=0, expected_update=4, qa=True, parent_loader=parent_loader)

    with pytest.raises(FileExistsError):
        r._train(out, manifest, _qa=_tiny_spec())


def test_numerical_failure_continues_remaining_seeds(tmp_path):
    calls = {"n": 0}

    actual_evaluator = _tiny_spec().evaluate
    def failed_predicate(model, manifest):
        value = actual_evaluator(model, manifest)
        calls["n"] += 1
        value["qa_predicates"]["combined_conjunction"] = calls["n"] != 1
        return value

    report = r._train(tmp_path / "numerical", _qa_manifest(), _qa=_tiny_spec(evaluate=failed_predicate))
    assert report["status"] == "complete_with_numerical_failures"
    assert all(row["status"] == "complete" for row in report["seeds"].values())
    assert report["seeds"]["0"]["numerical_failure"] is True
    assert report["cost"]["actual_added"]["updates"] == 12
    assert calls["n"] == 3


def test_technical_failure_records_and_suspends_remaining_seeds(tmp_path):
    calls = {"n": 0}
    def fail_technically(model, optimizer, batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return r.accepted_update(model, optimizer, batch)
        raise ValueError("synthetic runner failure")

    out = tmp_path / "technical"
    with pytest.raises(ValueError, match="synthetic runner failure"):
        r._train(out, _qa_manifest(), _qa=_tiny_spec(update=fail_technically))
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "suspended"
    assert report["seeds"]["0"]["status"] == "technical_failure"
    assert report["seeds"]["1"]["status"] == "not_started"
    assert report["seeds"]["2"]["status"] == "not_started"
    assert report["cost"]["actual_added"]["updates"] == 1
    assert report["seeds"]["0"]["attempted_updates"] == 2
    assert report["seeds"]["0"]["training_forward_cost"]["program_forwards"] == 1


def test_threshold_control_and_paired_metadata():
    seen = {"validation": [{"program": ["ADD"], "final_joint": 32}]}
    rows = [{"program": list(p), "metrics": {"all": {"final_joint": 244}}} for p in r.PRIMARY_PROGRAMS]
    report = {"primary_rows": rows, "primary_conjunction": True, "control_row": {"metrics": {"all": {"final_joint": 0}}}}
    assert r.predicates(seen, report)["combined_conjunction"] is True
    rows[0]["metrics"]["all"]["final_joint"] = 243
    report["primary_conjunction"] = False
    assert r.predicates(seen, report)["primary_conjunction"] is False
    def row(correct):
        return {"program": ["ADD"], "predictions": [
            {"state": [i//16, i%16], "stratum": "train", "target_trace": [[0,0]], "joint_final_correct": i < correct}
            for i in range(256)], "metrics": {"all": {"final_joint": correct, "final_x": correct, "final_y": correct, "full_trace": correct, "prefix_joint": [correct]}}}
    parent, current = row(100), row(110)
    paired = r._paired_row(parent, current, label="qa")
    assert paired["wins_parent_wrong_new_correct"] == 10
    assert paired["losses_parent_correct_new_wrong"] == 0
    reversed_pair = r._paired_row(current, parent, label="qa")
    assert reversed_pair["losses_parent_correct_new_wrong"] == 10
    for field, value in (("state", [99,99]), ("stratum", "test"), ("target_trace", [[1,1]])):
        corrupt = deepcopy(current)
        corrupt["predictions"][0][field] = value
        with pytest.raises(ValueError): r._paired_row(parent, corrupt, label="qa")
    current["metrics"]["all"]["final_joint"] = 111
    with pytest.raises(ValueError): r._paired_row(parent, current, label="qa")


def test_partial_evaluation_failure_records_actual_cost(tmp_path):
    evaluator = _tiny_spec().evaluate
    def fail_after_forward(model, manifest):
        evaluator(model, manifest)
        raise RuntimeError("after one legal evaluation forward")
    out = tmp_path / "partial-eval"
    with pytest.raises(RuntimeError, match="after one legal"):
        r._train(out, _qa_manifest(), _qa=_tiny_spec(evaluate=fail_after_forward))
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "suspended"
    assert report["cost"]["actual_added"]["updates"] == 4
    assert report["cost"]["actual_evaluation"]["program_forwards"] == 1
    assert report["cost"]["actual_evaluation"]["internal_state_substeps"] == 8
    assert report["seeds"]["1"]["status"] == "not_started"
