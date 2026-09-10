"""Narrow E33 continuation checks; no scientific training or evaluation scope."""

from copy import deepcopy
import json

import pytest
import torch

from looped_bitnet import width_e32 as e
from scripts import continuation_e33 as r
from scripts import length_transfer_e28 as e28


def test_fixed_seed0_scope_and_stream_lineage():
    assert r.SEED == 0
    assert r.ARMS == ("float", "w4")
    assert r.PARENT_UPDATE == 16000
    assert r.ADDED_UPDATES == 16000
    assert r.FINAL_UPDATE == 32000
    batches = e.stream()
    assert len(batches) == r.ADDED_UPDATES
    assert r.old.batch_digest(batches) == r.old.batch_digest(e.stream())
    assert r.old.target_digest(batches) == r.old.target_digest(e.stream())
    assert r.ADDED_COST_TOTAL == {
        "program_forwards": 32000, "program_state_cases": 2048000,
        "readout_positions": 4096000, "internal_state_substeps": 32768000,
    }
    assert r.EVAL_COST_TOTAL == {
        "program_forwards": 166, "program_state_cases": 24064,
        "readout_positions": 74624, "internal_state_substeps": 596992,
    }


def test_parent_loader_uses_canonical_e32_final_and_restores_rng():
    model, optimizer, payload, manifest = r.load_parent("float")
    assert payload["schema"] == e.SCHEMA
    assert payload["update"] == r.PARENT_UPDATE
    assert model.training is True
    assert r.e.digest_state_dict(model) == payload["model_digest"]
    assert r.e.digest_object(optimizer.state_dict()) == payload["optimizer_digest"]
    assert r.e.digest_object(torch.get_rng_state()) == payload["rng_digest"]
    assert manifest["schema"] == e.SCHEMA


def test_parent_comparison_and_current_pair_fixture_are_scope_checked():
    evaluations = {}
    models = {}
    for arm in r.ARMS:
        path = r.ROOT / r.PARENT_RUN / r._parent_label(arm) / "report.json"
        evaluations[arm] = json.loads(path.read_text())["evaluation"]
        models[r._parent_label(arm)] = {
            "evaluation": evaluations[arm], "added_training_cost": dict(r.TRAIN_COST_PER_ARM),
            "evaluation_cost": dict(r.EVAL_COST_PER_ARM),
        }
        comparison = r._parent_comparison(evaluations[arm], evaluations[arm])
        assert all(all(row["delta_final"] == 0 for row in rows) for rows in comparison["seen"].values())
        assert all(row["delta_final"] == 0 for row in comparison["length"]["rows"])
        assert set(comparison["length"]["rows"][0]["by_stratum"]) == {"train", "validation", "test"}
        assert comparison["composition"]["control"]["delta_final"] == 0
    pair = e28.paired(evaluations["w4"]["length"], evaluations["float"]["length"])
    assert len(pair) == 12
    assembled = r.assemble_report({"status": "running", "models": models},
                                  {r._parent_label(arm): evaluations[arm] for arm in r.ARMS})
    assert assembled["status"] == "complete"
    assert set(assembled["training_benefit"]) == {"float128_seed0", "w4128_seed0"}
    assert assembled["paired_benefit"] is False
    assert len(assembled["paired_w4_vs_float"]) == 12


def test_qa_checkpoint_strict_reload_and_tamper_rejection(tmp_path):
    report = r.tiny_qa(tmp_path / "e33_qa")
    assert report["status"] == "complete"
    assert report["actual_cost"] == {
        "updates": 8, "examples": 16, "readouts": 16, "internal_state_substeps": 128,
        "evaluation_forwards": 2, "evaluation_cases": 4, "evaluation_readouts": 4,
        "evaluation_internal_state_substeps": 32,
    }
    assert all(record["reload_next_update_equal"] for record in report["models"].values())
    manifest = json.loads((tmp_path / "e33_qa" / "manifest.json").read_text())
    path = tmp_path / "e33_qa" / "float128_seed0" / "u16002.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in (("arm", "w4"), ("parent_checkpoint_sha256", "bad"),
                       ("added_updates", 3), ("native_steps", 8.0), ("unexpected", 1)):
        tampered = deepcopy(payload)
        tampered[key] = value
        bad = tmp_path / f"bad-{key}.pt"
        torch.save(tampered, bad)
        with pytest.raises(ValueError, match="E33"):
            r.load_checkpoint(bad, manifest, arm="float", root=r.ROOT, qa=True, expected_update=16002)
    restored, optimizer, loaded = r.load_checkpoint(path, manifest, arm="float", root=r.ROOT, qa=True, expected_update=16002)
    assert restored.training is True
    assert loaded["cumulative_update"] == 16002
    assert r.e.digest_object(optimizer.state_dict()) == loaded["optimizer_digest"]
