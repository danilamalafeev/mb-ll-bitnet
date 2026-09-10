"""Small, zero-training controls for the fixed E37/E38 adapter."""

from copy import deepcopy

import pytest

from looped_bitnet.register_e15 import RegisterExample
from scripts import followup_e37_e38 as f


def _row(program=("ADD",), state=(0, 1), *, final=True, full=True):
    return {"program": list(program), "states": 1, "predictions": [{
        "state": list(state), "stratum": "train", "target_trace": [[1, 1]],
        "predicted_trace": [[1, 1]], "joint_final_correct": final,
        "prefix_joint_correct": [full],
    }]}


def test_checkpoint_digest_is_the_immutable_base_object():
    manifest = {"schema": "x", "fixed": {"a": 1}}
    left = f._checkpoint_manifest_digest(manifest)
    changed = deepcopy(manifest)
    changed["runtime_lineage"] = {"float128_seed1": "runs/example.pt"}
    assert left != f._checkpoint_manifest_digest(changed)
    changed["fixed"]["a"] = 2
    assert left != f._checkpoint_manifest_digest(changed)


def _runtime_fixture(tmp_path):
    manifest = {"schema": "base", "fixed": {"scope": "v2"}}
    base_path = tmp_path / "manifest.json"
    base_path.write_text('{"fixed":{"scope":"v2"},"schema":"base"}\n', encoding="utf-8")
    lineage = f._new_runtime_lineage(base_path, manifest, root=tmp_path)
    return manifest, base_path, lineage


def _runtime_entry(tmp_path, arm, seed, content):
    label = f._label(arm, seed)
    path = tmp_path / f"{label}.pt"
    path.write_bytes(content)
    return label, path, {"label": label, "path": path.name, "sha256": f._sha256(path),
                         "arm": arm, "seed": seed, "update": f.PARENT_UPDATE, "stage": "parent_extension"}


def test_runtime_lineage_registers_four_parents_without_mutating_base(tmp_path):
    manifest, base_path, lineage = _runtime_fixture(tmp_path)
    before = base_path.read_bytes()
    for arm in f.ARMS:
        for seed in f.SEEDS:
            label, _, entry = _runtime_entry(tmp_path, arm, seed, f"{arm}-{seed}".encode())
            lineage = f._register_runtime_parent(lineage, label, entry)
    f._validate_runtime_lineage(lineage, manifest, root=tmp_path)
    assert base_path.read_bytes() == before
    assert f._checkpoint_manifest_digest(manifest) == lineage["base_manifest_digest"]
    assert set(lineage["entries"]) == {f._label(arm, seed) for arm in f.ARMS for seed in f.SEEDS}


def test_runtime_lineage_rejects_missing_wrong_replaced_and_wrong_base(tmp_path):
    manifest, _, lineage = _runtime_fixture(tmp_path)
    label, path, entry = _runtime_entry(tmp_path, "float", 1, b"parent-1")
    with pytest.raises(ValueError, match="missing"):
        f._resolve_runtime_parent(lineage, manifest, label, root=tmp_path)
    lineage = f._register_runtime_parent(lineage, label, entry)
    assert f._resolve_runtime_parent(lineage, manifest, label, root=tmp_path)[0] == path
    path.write_bytes(b"replaced-bytes")
    with pytest.raises(ValueError, match="file/hash"):
        f._resolve_runtime_parent(lineage, manifest, label, root=tmp_path)
    with pytest.raises(ValueError, match="already registered"):
        f._register_runtime_parent(lineage, label, entry)
    wrong_seed = deepcopy(entry)
    wrong_seed["seed"] = 2
    with pytest.raises(ValueError, match="label mismatch"):
        f._register_runtime_parent(lineage, label, wrong_seed)
    wrong_label = deepcopy(entry)
    wrong_label["label"] = "w4128_seed1"
    with pytest.raises(ValueError, match="identity mismatch"):
        f._register_runtime_parent(lineage, label, wrong_label)
    changed_entry = deepcopy(entry)
    changed_entry["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="replacement rejected"):
        f._register_runtime_parent(lineage, label, changed_entry)
    wrong_base = deepcopy(lineage)
    wrong_base["base_manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="base identity"):
        f._resolve_runtime_parent(wrong_base, manifest, label, root=tmp_path)


def test_runtime_lineage_sibling_digest_and_failure_counters_are_stable(tmp_path):
    manifest, _, lineage = _runtime_fixture(tmp_path)
    first_label, _, first = _runtime_entry(tmp_path, "float", 1, b"first")
    second_label, _, second = _runtime_entry(tmp_path, "w4", 2, b"second")
    lineage = f._register_runtime_parent(lineage, first_label, first)
    digest = lineage["base_manifest_digest"]
    lineage = f._record_runtime_status(lineage, first_label, status="complete", attempted_updates=3,
                                        completed_updates=2, training_cost={key: 1 for key in f.COST_KEYS})
    lineage = f._register_runtime_parent(lineage, second_label, second)
    failed = f._record_runtime_status(lineage, second_label, status="technical_failure", attempted_updates=5,
                                      completed_updates=4, training_cost={key: 2 for key in f.COST_KEYS}, error="boom")
    assert failed["base_manifest_digest"] == digest
    assert failed["entries"][first_label] == first
    assert failed["entries"][second_label] == second
    assert failed["status"][first_label]["completed_updates"] == 2
    assert failed["status"][second_label]["attempted_updates"] == 5


def test_failed_progress_keeps_parent_and_branch_counters_separate(tmp_path):
    manifest, _, lineage = _runtime_fixture(tmp_path)
    out = tmp_path / "out"
    parent_progress = out / "parents" / "float128_seed1" / "progress.json"
    branch_progress = out / "branches" / "float128_seed1" / "B" / "progress.json"
    parent_progress.parent.mkdir(parents=True)
    branch_progress.parent.mkdir(parents=True)
    parent_progress.write_text('{"attempted_updates": 4, "completed_updates": 4, "training_cost": {"program_forwards": 4, "program_state_cases": 4, "readout_positions": 4, "internal_state_substeps": 32}}', encoding="utf-8")
    branch_progress.write_text('{"attempted_updates": 3, "completed_updates": 2, "training_cost": {"program_forwards": 3, "program_state_cases": 3, "readout_positions": 3, "internal_state_substeps": 24}, "error": "branch boom"}', encoding="utf-8")
    saved = f._persist_failed_progress(out, lineage, root=tmp_path)
    assert saved["status"]["float128_seed1"]["completed_updates"] == 4
    assert saved["status"]["float128_seed1/B"]["attempted_updates"] == 3
    assert saved["status"]["float128_seed1/B"]["completed_updates"] == 2
    assert saved["status"]["float128_seed1/B"]["error"] == "branch boom"


def test_exact_inherited_schedules_are_deterministic_without_model_calls():
    a, b = f.wave.build_streams()
    metadata = f._schedule_metadata(a, b)
    assert metadata["A"]["updates"] == 8000
    assert metadata["B_match"]["updates"] == 4572
    assert metadata["B"]["updates"] == 8000
    assert metadata["A"]["cost"]["readout_positions"] == 1024128
    assert metadata["B_match"]["cost"]["readout_positions"] == 1024128
    assert metadata["B"]["cost"]["readout_positions"] == 1791872


def test_e37_pair_requires_exact_target_and_state_identity():
    left = _row()
    right = _row(final=False, full=False)
    pair = f._pair_rows(left, right, label="fixture")
    assert pair["cases"] == 1
    assert pair["final"]["left_correct_right_wrong"] == 1
    assert pair["full_trace"]["left_correct_right_wrong"] == 1
    bad = _row(); bad["predictions"][0]["target_trace"] = [[2, 2]]
    with pytest.raises(ValueError, match="target mismatch"):
        f._pair_rows(left, bad, label="fixture")


def test_cost_for_tiny_add_batch_counts_native_steps():
    batch = [RegisterExample(0, 1, ("ADD",)), RegisterExample(2, 3, ("ADD",))]
    assert f._cost_for_batches([batch, batch]) == {
        "program_forwards": 2, "program_state_cases": 4,
        "readout_positions": 4, "internal_state_substeps": 32,
    }


def test_e36_primary_scope_is_exactly_24_allowed_semantic_new_rows():
    rows = []
    for length in range(7, 11):
        for index in range(6):
            rows.append({"length": length, "category": "semantic_new", "legality": "allowed"})
    evaluation = {"e36": {"rows": rows}}
    assert len(f._e36_primary_rows(evaluation)) == 24
    rows[-1]["legality"] = "forbidden"
    with pytest.raises(ValueError, match="primary pool scope"):
        f._e36_primary_rows(evaluation)


@pytest.mark.parametrize('a_correct,b_correct,benefit', [
    (False, True, True), (True, False, False), (True, True, False),
    (False, False, False),
])
@pytest.mark.parametrize('budget', ['B_match', 'B'])
def test_e36_primary_benefit_means_fewer_right_errors(a_correct, b_correct, benefit, budget):
    def evaluation(correct):
        rows = []
        for length in range(7, 11):
            for index in range(6):
                program = tuple('MUL' if (index >> bit) & 1 else 'ADD' for bit in range(length))
                row = _row(program=program, full=correct, final=correct)
                row.update(length=length, category='semantic_new', legality='allowed')
                rows.append(row)
        return {'e36': {'rows': rows}}
    pair = f._e36_primary_pair(evaluation(a_correct), evaluation(b_correct), label='A-vs-' + budget)
    assert pair['cases'] == 24
    assert pair['directional_full_trace_improvement'] is benefit


def test_qa_incremental_costs_survive_failed_update_without_double_count(monkeypatch):
    batch = [RegisterExample(0, 1, ('ADD',)), RegisterExample(2, 3, ('ADD',))]
    actual = {'attempted_updates': 0, 'completed_updates': 0,
              'training': f._zero_cost(), 'snapshot_next_update_identity': f._zero_cost()}
    active = []
    class Hook:
        def remove(self):
            active.remove(self)
    def hooks(model, record, phase):
        record.setdefault(phase + '_attempted', f._zero_cost())
        active.extend([Hook(), Hook()])
        return tuple(active)
    def update(model, optimizer, batch):
        phase = 'training' if actual['attempted_updates'] <= 6 else 'snapshot_next_update_identity'
        for key, value in f._cost_for_batches([batch]).items():
            actual[phase + '_attempted'][key] += value
            actual[phase][key] += value
        if actual['attempted_updates'] == 8:
            raise RuntimeError('optimizer failed after forward')
    monkeypatch.setattr(f, '_hooks', hooks)
    monkeypatch.setattr(f, '_update', update)
    for _ in range(6):
        f._qa_update(None, None, batch, actual, 'training')
    f._qa_update(None, None, batch, actual, 'snapshot_next_update_identity')
    assert actual['completed_updates'] == 7
    assert actual['training']['program_forwards'] == 6
    assert actual['snapshot_next_update_identity']['program_forwards'] == 1
    assert sum(actual[p]['internal_state_substeps'] for p in ('training', 'snapshot_next_update_identity')) == 112
    with pytest.raises(RuntimeError, match='optimizer failed'):
        f._qa_update(None, None, batch, actual, 'snapshot_next_update_identity')
    assert actual['attempted_updates'] == 8
    assert actual['completed_updates'] == 7
    assert actual['snapshot_next_update_identity']['program_forwards'] == 2
    assert not active
