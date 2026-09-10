import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from looped_bitnet.config import Config
from looped_bitnet.data import collate, make_eval_sets
from looped_bitnet.model import ReasoningModel
from scripts.length_mixture_e14 import (
    HOPS, OBJECTIVE, UPDATES, VALIDATION_THRESHOLD, _native_eval, make_test_manifest,
    make_validation_manifest, mixture_digests, mixture_loss, mixture_schedule,
    validate_checkpoint_payload, validation_gate, validation_score, _records_from,
)
from scripts.state_scale_train_e13 import e13_forward


def cfg():
    return Config.load("configs/cycle16_two_hop_steps8_structured_reader.json")


def test_registered_schedule_and_compute_are_exact():
    schedule = mixture_schedule()
    assert len(schedule) == UPDATES
    assert {h: schedule.count(h) for h in HOPS} == {1: 666, 2: 668, 3: 666}
    assert sum(schedule) == 4000
    assert sum(schedule) * 64 == 256000


def test_mixture_digests_keep_base_h2_stream_separate_and_deterministic():
    a, b = mixture_digests(cfg(), 0), mixture_digests(cfg(), 0)
    assert a == b
    assert a["counts"] == {"1": 666, "2": 668, "3": 666}
    assert a["base_h2_stream_digest"] != a["mixture_input_target_digest"]


def test_pairing_manifests_reuse_transitions_start_and_order():
    m = make_validation_manifest(cfg(), count=8)
    assert m["stream_seed"] == 20260912 and m["count"] == 8
    for h in HOPS:
        rows = [e.canonical_identity[:2] for e in __import__("scripts.length_mixture_e14", fromlist=["make_examples"]).make_examples(m["records"], h)]
        assert len(set(rows)) == 8
    assert make_test_manifest(cfg(), count=8)["stream_seed"] == 20260913


def test_mixture_loss_has_weight_two_and_correct_intermediate_targets():
    c = cfg(); examples = make_eval_sets(c.data)["validation"][:2]
    model = ReasoningModel(c.model)
    b = collate(examples)
    _, ro = model.forward_with_readouts(b["input_ids"], (4, 8), steps=8)
    loss = mixture_loss(ro, examples, 2)
    expected = sum(torch.nn.functional.cross_entropy(ro[4*k], torch.tensor([e.target if k == 2 else __import__("looped_bitnet.data", fromlist=["solve"]).solve(e.transitions, e.start, k) for e in examples])) for k in (1, 2))
    torch.testing.assert_close(loss, expected)


def test_native_eval_has_prefix_readouts_and_validation_gate():
    c = cfg(); examples = make_eval_sets(c.data)["validation"][:4]
    out = _native_eval(ReasoningModel(c.model), {1: examples, 2: examples, 3: examples}, c)
    assert set(out) == {"1", "2", "3"}
    assert set(out["3"]["readouts"]) == {"1", "2", "3"}
    assert validation_score(out)[0] >= 0 and validation_gate(out) is False
    assert VALIDATION_THRESHOLD == 0.95


def test_h2_loss_logits_and_all_parameter_gradients_match_e13_baseline():
    c = cfg(); examples = make_eval_sets(c.data)["validation"][:3]; b = collate(examples)
    a, z = ReasoningModel(c.model), ReasoningModel(c.model); z.load_state_dict(a.state_dict())
    _, ro = a.forward_with_readouts(b["input_ids"], (4, 8), steps=8)
    final, old_ro, _ = e13_forward(z, b["input_ids"], steps=8, arm="baseline", readout_steps=(4, 8))
    torch.testing.assert_close(ro[4], old_ro[4], rtol=0, atol=0)
    torch.testing.assert_close(ro[8], final, rtol=0, atol=0)
    new_loss = mixture_loss(ro, examples, 2)
    old_loss = F.cross_entropy(final, b["target"]) + F.cross_entropy(
        old_ro[4], torch.tensor([__import__("looped_bitnet.data", fromlist=["solve"]).solve(e.transitions, e.start, 1) for e in examples]))
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    new_loss.backward(); old_loss.backward()
    for p, q in zip(a.parameters(), z.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_prefix_readouts_are_exact_for_real_h1_h2_h3_inputs():
    c = cfg(); model = ReasoningModel(c.model)
    rows = make_eval_sets(c.data)["validation"][:2]
    for h in HOPS:
        b = collate([type(rows[0])(rows[0].transitions, rows[0].start, h, rows[0].order)])
        _, readouts = model.forward_with_readouts(b["input_ids"], tuple(range(4, 4*h + 1, 4)), steps=4*h)
        for step, logits in readouts.items():
            torch.testing.assert_close(model(b["input_ids"], steps=step), logits, rtol=0, atol=0)


def test_synthetic_gate_requires_all_six_cells_and_macro_score_is_length_balanced():
    native = {str(h): {"final": {"accuracy": 1.0, "loss": 1.0}, "readouts": {str(k): {"accuracy": 1.0} for k in range(1, h + 1)}} for h in HOPS}
    assert validation_gate(native)
    native["3"]["readouts"]["2"]["accuracy"] = VALIDATION_THRESHOLD - 1e-6
    assert not validation_gate(native)
    native["3"]["readouts"]["2"]["accuracy"] = 1.0
    native["1"]["final"]["accuracy"] = 0.0
    assert validation_score(native)[0] == pytest.approx(2 / 3)


def test_missing_or_malformed_novelty_source_fails_closed(tmp_path):
    with pytest.raises(ValueError): _records_from(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"records": [{"start": 0}]}))
    with pytest.raises(ValueError): _records_from(bad)


def test_checkpoint_validation_rejects_bad_objective(tmp_path):
    c = cfg(); model = ReasoningModel(c.model)
    payload = {"format_version": 1, "diagnostic_checkpoint": True, "resume_supported": False,
               "objective": OBJECTIVE, "seed": 0, "config": c.to_dict(),
               "dynamics": {"input_hops": [1, 2, 3], "recurrent_budget": "4*h", "normalization": "none", "teacher_forcing": False}}
    validate_checkpoint_payload(payload, path=tmp_path / "x.pt", seed=0, config=c)
    payload["objective"] = "wrong"
    with pytest.raises(ValueError): validate_checkpoint_payload(payload, path=tmp_path / "x.pt", seed=0, config=c)


def test_no_training_or_frozen_result_overwrite_in_source():
    source = Path("scripts/length_mixture_e14.py").read_text()
    assert "--protocol-cleared" in source
    assert 'Path("results/STATE_SCALE_TRAIN_E13_CORRECTED.md").write_text' not in source
