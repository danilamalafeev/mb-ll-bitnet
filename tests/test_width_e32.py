"""Bounded E32 model-factory QA; no scientific training or evaluation scopes."""

import json
from copy import deepcopy

import pytest
import torch

from looped_bitnet import longer_native8_e20 as old
from looped_bitnet import w4_e27 as historical_w4
from looped_bitnet import width_e32 as e
from scripts import width_e32 as runner


def test_width128_inventory_and_shapes_without_forward():
    float_model = e.build_initial_model(128, "float", 0)
    w4_model = e.build_initial_model(128, "w4", 0)
    e.assert_paired_models(float_model, w4_model)
    assert e.PARAMETER_COUNT_128 == 335232
    assert sum(parameter.numel() for parameter in float_model.parameters()) == 335232
    assert sum(parameter.numel() for parameter in w4_model.parameters()) == 335232
    assert float_model.reader.head_dim == 32
    assert float_model.d_ff == 256
    assert float_model.native_steps == 8
    assert e.OPCODE_SCALE == 1.0 / 8.0
    assert tuple(name for name, module in w4_model.named_modules() if type(module) is historical_w4.W4BitLinear) == e.QAT_NAMES
    assert not any(type(module) is historical_w4.W4BitLinear for module in float_model.modules())
    assert tuple(float_model.x_embedding.projection.weight.shape) == (128, 4)
    assert tuple(float_model.y_embedding.projection.weight.shape) == (128, 4)
    assert tuple(float_model.reader.q_proj.weight.shape) == (128, 128)
    assert tuple(float_model.blocks[0].up.weight.shape) == (256, 128)
    assert tuple(float_model.blocks[0].down.weight.shape) == (128, 256)
    assert tuple(float_model.x_head.weight.shape) == (16, 128)


@pytest.mark.parametrize("seed", e.SEEDS)
def test_width64_factory_matches_accepted_historical_masters_and_rng(seed):
    fresh_w4 = e.build_initial_model(64, "w4", seed)
    fresh_float = e.build_initial_model(64, "float", seed)
    historical = historical_w4.build_initial_model(seed)
    assert e.digest_state_dict(fresh_w4) == old.digest_state_dict(historical)
    assert e.digest_state_dict(fresh_float) == old.digest_state_dict(historical)
    assert e.digest_object(fresh_w4.initial_rng_state) == old.digest_object(historical.initial_rng_state)
    assert e.digest_object(fresh_float.initial_rng_state) == old.digest_object(historical.initial_rng_state)
    assert sum(parameter.numel() for parameter in fresh_w4.parameters()) == e.PARAMETER_COUNT_64


@pytest.mark.parametrize("seed", e.SEEDS)
def test_paired_width128_masters_and_training_rng_are_exactly_shared(seed):
    float_model = e.build_initial_model(128, "float", seed)
    w4_model = e.build_initial_model(128, "w4", seed)
    e.assert_paired_models(float_model, w4_model)
    assert list(float_model.state_dict()) == list(w4_model.state_dict())
    left_parameters = list(float_model.named_parameters())
    right_parameters = list(w4_model.named_parameters())
    assert [name for name, _ in left_parameters] == [name for name, _ in right_parameters]
    assert all(torch.equal(left, right) for (_, left), (_, right) in zip(left_parameters, right_parameters))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(float_model.state_dict().values(), w4_model.state_dict().values()))
    assert torch.equal(float_model.initial_rng_state, w4_model.initial_rng_state)


def test_width_specific_inventory_rejects_wrong_arm_or_width():
    model = e.build_initial_model(128, "float", 0)
    with pytest.raises(ValueError, match="projection inventory"):
        e.check_inventory(model, width=128, arm="w4")
    with pytest.raises(ValueError, match="width/native-step"):
        e.check_inventory(model, width=64, arm="float")
    with pytest.raises(ValueError, match="unknown E32 arm"):
        e.check_inventory(model, width=128, arm="bad")


def test_reference_rng_sources_match_frozen_initial_reference():
    reference = json.loads((e.ROOT / "results/E25_INITIAL_REFERENCE.json").read_text())
    expected = {str(row["seed"]): row["float_rng_digest"] for row in reference["seeds"]}
    for seed in e.SEEDS:
        model = e.build_initial_model(128, "float", seed)
        assert e.digest_object(model.initial_rng_state) == expected[str(seed)]


def test_no_scientific_outputs_created_by_factory():
    # Factory construction is deliberately side-effect free for run artifacts.
    assert not (e.ROOT / "runs/e32_width").exists()


def test_manifest_freezes_seed_major_order_and_actual_inventory():
    manifest = e.make_manifest()
    assert manifest["config"]["labels"] == [
        "float128_seed0", "w4128_seed0", "float128_seed1",
        "w4128_seed1", "float128_seed2", "w4128_seed2",
    ]
    assert manifest["model_inventories"]["h128_w4"]["quantized_matrix_parameters"] == 331776
    assert manifest["model_inventories"]["h128_w4"]["unquantized_fp32_parameters"] == 3456
    assert manifest["evaluation_cost_total"] == {
        "program_forwards": 498, "program_state_cases": 72192,
        "readout_positions": 223872, "internal_state_substeps": 1790976,
    }


def test_runner_tiny_qa_and_reload_identity(tmp_path):
    report = runner.tiny_qa(tmp_path / "e32_width_qa")
    assert report["status"] == "complete"
    assert report["model_order"] == list(e.LABELS)
    assert report["qa_reload_next_update_cost"] == {
        "updates": 12, "examples": 24, "readouts": 24, "internal_state_substeps": 192,
    }
    assert all(record["reload_next_update_equal"] for record in report["models"].values())
    assert all(record["training_cost"]["program_forwards"] == 2 for record in report["models"].values())
    checkpoint = tmp_path / "e32_width_qa" / "float128_seed0" / "u2.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tampered = deepcopy(payload)
    tampered["arm"] = "w4"
    bad = tmp_path / "tampered.pt"
    torch.save(tampered, bad)
    manifest = json.loads((tmp_path / "e32_width_qa" / "manifest.json").read_text())
    with pytest.raises(ValueError, match="identity mismatch"):
        e.load_checkpoint(bad, manifest, arm="float", width=128, seed=0, update=2, qa=True,
                          training_cost=report["models"]["float128_seed0"]["training_cost"])


def test_synthetic_report_scope_and_paired_metrics_fail_closed():
    seen_programs = sorted(runner.SEEN_PROGRAMS)
    seen = {
        "train": [{"program": list(program), "final_joint": 32} for program in seen_programs],
        "validation": [{"program": list(program), "final_joint": 32 if len(program) == 1 else 31}
                        for program in seen_programs],
    }
    composition = {"primary_rows": [{"program": list(program), "metrics": {"all": {"final_joint": 244}}}
                                     for program in runner.e21.PRIMARY_PROGRAMS]}
    length = {"rows": [{"program": list(program), "length": len(program),
                         "metrics": {"all": {"final_joint": 244, "full_trace": 244}}}
                        for program in runner.LENGTH_PROGRAMS]}
    assert runner.predicates(seen, composition, length)["combined_conjunction"]
    missing = dict(seen)
    missing["validation"] = missing["validation"][:-1]
    with pytest.raises(ValueError, match="seen validation scope"):
        runner.predicates(missing, composition, length)

    states = [(x, y) for x in range(16) for y in range(16)]
    predictions = [{"state": list(state), "target_trace": [[0, 0]], "stratum": "train",
                    "joint_final_correct": True, "prefix_joint_correct": [True]}
                   for state in states]
    old_row = {"program": ["ADD"], "predictions": predictions,
               "metrics": {"all": {"final_joint": 256, "full_trace": 256}}}
    paired = runner._pair_rows(old_row, old_row, label="synthetic")
    assert paired["full_trace"]["both_correct"] == 256
    malformed = {**old_row, "predictions": [{key: value for key, value in predictions[0].items()
                                               if key != "stratum"}] + predictions[1:]}
    with pytest.raises(ValueError, match="missing stratum"):
        runner._pair_rows(old_row, malformed, label="synthetic")


def test_saved_fixture_assembler_and_gate_boundaries_without_forwards():
    # Historical E28 predictions are a saved fixture here; this exercises the
    # final report joins and width contrasts without evaluating a model.
    models = {}
    for seed in e.SEEDS:
        old_w4, old_w4_length = runner._historical("E27_W4_64", seed)
        old_float, old_float_length = runner._historical("E24_float64", seed)
        models[f"w4128_seed{seed}"] = {"evaluation": {"length": old_w4_length}}
        models[f"float128_seed{seed}"] = {"evaluation": {"length": old_float_length}}
    contrasts = runner.width_contrasts({"models": models})
    assert set(contrasts["by_seed"]) == {"0", "1", "2"}
    assert all(len(value["paired_128_w4_vs_float"]) == 12 for value in contrasts["by_seed"].values())

    seen_programs = sorted(runner.SEEN_PROGRAMS)
    seen = {
        "train": [{"program": list(program), "final_joint": 32} for program in seen_programs],
        "validation": [{"program": list(program), "final_joint": 32 if len(program) == 1 else 31}
                        for program in seen_programs],
    }
    composition = {"primary_rows": [{"program": list(program), "metrics": {"all": {"final_joint": 244}}}
                                     for program in runner.e21.PRIMARY_PROGRAMS],
                   "control_row": {"program": ["ADD", "XOR", "XOR"], "metrics": {"all": {"final_joint": 0}}}}
    length = {"rows": [{"program": list(program), "length": len(program),
                         "metrics": {"all": {"final_joint": 244, "full_trace": 244}}}
                        for program in runner.LENGTH_PROGRAMS]}
    assert runner.predicates(seen, composition, length)["combined_conjunction"]
    below = deepcopy(composition)
    below["primary_rows"][0]["metrics"]["all"]["final_joint"] = 243
    assert not runner.predicates(seen, below, length)["combined_conjunction"]
    duplicate = deepcopy(length)
    duplicate["rows"][-1]["program"] = duplicate["rows"][0]["program"]
    with pytest.raises(ValueError, match="length scope"):
        runner.predicates(seen, composition, duplicate)


def test_runner_technical_failure_preserves_actual_counters(tmp_path):
    manifest = e.make_manifest()
    batch = [runner.RegisterExample(0, 1, ("ADD",)), runner.RegisterExample(2, 3, ("ADD",))]

    def fail_after_reload(_model, _manifest):
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        runner.execute(tmp_path / "failure", manifest, [batch, batch], qa=True, evaluator=fail_after_reload)
    report = json.loads((tmp_path / "failure" / "report.json").read_text())
    record = report["models"]["float128_seed0"]
    assert report["status"] == "suspended"
    assert record["status"] == "technical_failure"
    assert record["attempted_updates"] == record["completed_updates"] == 2
    assert record["training_cost"] == {
        "program_forwards": 2, "program_state_cases": 4,
        "readout_positions": 4, "internal_state_substeps": 32,
    }
