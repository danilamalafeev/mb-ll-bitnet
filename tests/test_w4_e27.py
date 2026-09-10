"""Bounded E27 QA: tiny three-seed runner plus independent W4 operator checks."""

from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F

from looped_bitnet import w4_e27 as e
from looped_bitnet import longer_native8_e20 as old
from scripts import w4_e27 as r


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("e27") / "qa"
    report = r.tiny_qa(out)
    return out, report


def test_w4_numeric_oracle_zero_small_and_clamp():
    weights = torch.tensor([[0.0, 0.25, -0.5, 1.0], [2.5, -3.0, 0.01, -0.02]], dtype=torch.float32)
    scale = (weights.abs().amax() / 7.0).clamp_min(1e-8)
    expected = weights + (((weights / scale).round().clamp(-7, 7) * scale) - weights).detach()
    actual = e.symmetric_w4_weight(weights)
    assert torch.equal(actual, expected)
    assert float(actual.abs().max()) <= float(scale * 7)

    zero = torch.zeros(3, 4)
    assert torch.equal(e.w4_weight(zero), zero)
    small = torch.tensor([[1e-12, -2e-12], [3e-12, 0.0]], dtype=torch.float32)
    small_scale = (small.abs().amax() / 7.0).clamp_min(1e-8)
    small_expected = (small / small_scale).round().clamp(-7, 7) * small_scale
    assert torch.equal(e.w4_weight(small), small_expected)


def test_w4_identity_ste_and_fp32_input_path():
    layer = e.W4BitLinear(3, 2, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.2, -0.7, 0.09], [0.4, 0.1, -0.3]]))
    x = torch.tensor([[0.123456, -0.34, 1.0]], requires_grad=True)
    actual = layer(x)
    expected = F.linear(x, e.w4_weight(layer.weight), layer.bias)
    assert torch.equal(actual, expected)
    from looped_bitnet.quantization import int8_activation
    assert not torch.equal(actual, F.linear(int8_activation(x), e.w4_weight(layer.weight), layer.bias))
    actual.sum().backward()
    assert torch.equal(layer.weight.grad, x.detach().expand(2, -1))
    assert torch.equal(x.grad, e.w4_weight(layer.weight).sum(0, keepdim=True))
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_initial_pairing_and_fourteen_locations():
    independent = json.loads((e.ROOT / "results/E25_INITIAL_REFERENCE.json").read_text())
    assert len(e.QAT_NAMES) == 14
    for item in independent["seeds"]:
        model = e.build_initial_model(item["seed"])
        assert old.digest_state_dict(model) == item["initial_digest"]
        assert old.digest_object(model.initial_rng_state) == item["float_rng_digest"]
        assert list(e.QAT_NAMES) == item["bitlinear_locations"]
        assert tuple(n for n, module in model.named_modules() if type(module) is e.W4BitLinear) == e.QAT_NAMES
        assert sum(parameter.numel() for parameter in model.parameters()) == 151232


def test_real_three_seed_qa_and_overwrite_refusal(tiny):
    out, report = tiny
    assert report["status"] == "complete_with_numerical_failures"
    assert list(report["seeds"]) == ["0", "1", "2"]
    assert all(row["reload_next_update_equal"] for row in report["seeds"].values())
    assert report["cost"]["completed_updates"] == 3
    assert report["cost"]["training_cost"] == dict(zip(e.COST_KEYS, (3, 6, 6, 48)))
    assert report["cost"]["evaluation_cost"] == dict(zip(e.COST_KEYS, (3, 6, 6, 48)))
    assert report["cost"]["reload_identity_cost"] == dict(zip(e.COST_KEYS, (6, 12, 12, 96)))
    with pytest.raises(FileExistsError):
        r.tiny_qa(out)


def test_strict_reload_tamper_and_types(tiny, tmp_path):
    out, _ = tiny
    manifest = json.loads((out / "manifest.json").read_text())
    original = torch.load(out / "seed0/u1.pt", weights_only=True)
    for key, value in [
        ("seed", 1), ("update", 2), ("qa", False), ("qat_names", []),
        ("initial_digest", "bad"), ("manifest_digest", "bad"),
        ("training_mode", False), ("completed_updates", 2), ("training_cost", {}),
        ("seed", False), ("update", 1.0), ("qa", 1), ("training_mode", 1),
        ("schema", "e26_weight_only_v1"),
    ]:
        changed = deepcopy(original)
        changed[key] = value
        path = tmp_path / f"{key}_{str(value).replace('/', '_')}.pt"
        torch.save(changed, path)
        with pytest.raises((ValueError, KeyError)):
            e.load_checkpoint(path, manifest, seed=0, update=1, qa=True)
    model, optimizer, payload = e.load_checkpoint(out / "seed0/u1.pt", manifest, seed=0, update=1, qa=True)
    assert model.training and optimizer is not None
    assert torch.equal(torch.get_rng_state(), payload["rng_state"])


def test_actual_inventory_tamper_and_nonfinite_guard():
    model = e.build_initial_model(0)
    model.reader.q_proj = torch.nn.Linear(64, 64)
    with pytest.raises(ValueError, match="projection inventory"):
        e.check_inventory(model)
    model = e.build_initial_model(0)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        e.check_inventory(model)


def test_paired_report_identities_threshold_and_control():
    baseline = json.loads((e.ROOT / e.E26_RUN / "report.json").read_text())["seeds"]["0"]["evaluation"]
    same = r.compare(baseline, baseline)
    for row in same["composition"]["primary"]:
        assert row["wins_parent_wrong_new_correct"] == row["losses_parent_correct_new_wrong"] == 0
    changed = deepcopy(baseline)
    changed["composition"]["primary_rows"][0]["predictions"][0]["stratum"] = "bad"
    with pytest.raises(ValueError, match="stratum"):
        r.compare(changed, baseline)
    changed = deepcopy(baseline)
    for row in changed["composition"]["primary_rows"]:
        row["metrics"]["all"]["final_joint"] = 244
    changed["composition"]["primary_conjunction"] = True
    assert r.accepted.predicates(changed["seen"], changed["composition"])["primary_conjunction"]
    changed["composition"]["control_row"]["metrics"]["all"]["final_joint"] = 0
    assert r.accepted.predicates(changed["seen"], changed["composition"])["primary_conjunction"]
    changed["composition"]["primary_rows"][0]["metrics"]["all"]["final_joint"] = 243
    changed["composition"]["primary_conjunction"] = False
    assert not r.accepted.predicates(changed["seen"], changed["composition"])["primary_conjunction"]


def test_paired_state_marginals_and_direction():
    baseline = json.loads((e.ROOT / e.E26_RUN / "report.json").read_text())["seeds"]["0"]["evaluation"]
    row = baseline["composition"]["primary_rows"][0]
    changed = deepcopy(row)
    changed["predictions"][0]["state"] = changed["predictions"][1]["state"]
    with pytest.raises(ValueError, match="duplicate"):
        r.accepted._paired_row(row, changed, label="test")
    changed = deepcopy(row)
    changed["metrics"]["all"]["final_joint"] -= 1
    with pytest.raises(ValueError, match="marginal"):
        r.accepted._paired_row(row, changed, label="test")
    changed = deepcopy(row)
    correct = next(item for item in changed["predictions"] if item["joint_final_correct"])
    correct["joint_final_correct"] = False
    changed["metrics"]["all"]["final_joint"] -= 1
    result = r.accepted._paired_row(row, changed, label="test")
    assert result["losses_parent_correct_new_wrong"] == 1
    assert result["wins_parent_wrong_new_correct"] == 0
    reverse = r.accepted._paired_row(changed, row, label="test")
    assert reverse["wins_parent_wrong_new_correct"] == 1
    assert reverse["losses_parent_correct_new_wrong"] == 0


def test_both_comparators_and_labels():
    baseline = json.loads((e.ROOT / e.E26_RUN / "report.json").read_text())["seeds"]["0"]["evaluation"]
    paired = r.compare_both(baseline, 0)
    assert set(paired) == {"E26_Wternary_A32", "E24_float"}
    for label, path in [("E26_Wternary_A32", e.E26_RUN), ("E24_float", e.FLOAT_RUN)]:
        comparator = json.loads((e.ROOT / path / "report.json").read_text())["seeds"]["0"]["evaluation"]
        assert paired[label] == r.compare(baseline, comparator)


def test_protected_stream_and_manifest():
    manifest = e.make_manifest()
    assert manifest["config"]["quantization"] == "symmetric_W4_weights_absmax7_STE_identity_FP32_activations"
    reference = json.loads((e.ROOT / e.E26_RUN / "manifest.json").read_text())
    for key in ("initial_digests", "initial_rng_digests", "full_stream_digest", "full_target_digest", "checkpoint_costs", "seen", "symbolic", "state_split"):
        assert manifest[key] == reference[key]
    assert str(e.COMPARATOR_REFERENCE) in manifest["comparator_references"]
    assert str(e.E26_RUN / "report.json") in manifest["comparator_references"]
    assert str(e.FLOAT_RUN / "report.json") in manifest["comparator_references"]
    assert manifest["protected_count"] == len(e.verify_protected_hashes())


def test_runtime_failure_suspends_and_preserves_partial_artifact(tmp_path):
    batch = old.base_stream()[0][:2]
    model = e.build_initial_model(0)
    manifest = {
        "qa": True,
        "initial_digests": {"0": old.digest_state_dict(model)},
        "initial_rng_digests": {"0": old.digest_object(model.initial_rng_state)},
        "checkpoint_costs": {"1": dict(zip(e.COST_KEYS, (1, 2, 2, 16)))},
    }

    def failure(model, manifest):
        raise RuntimeError("intentional QA evaluation failure before forward")

    out = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="intentional"):
        r.execute(out, manifest, [batch], failure, qa=True)
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "suspended"
    assert report["seeds"]["0"]["status"] == "technical_failure"
    assert report["seeds"]["1"]["status"] == report["seeds"]["2"]["status"] == "not_started"
    assert report["cost"]["completed_updates"] == 1
    assert report["cost"]["training_cost"]["program_forwards"] == 1
    assert report["cost"]["evaluation_cost"]["program_forwards"] == 0
    assert (out / "seed0/u1.pt").exists()
