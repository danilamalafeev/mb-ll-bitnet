"""Bounded E25 QA: 3 train + 3 eval + 6 reload-identity forwards total."""
from copy import deepcopy
import json
import pytest
import torch
from looped_bitnet import qat_match_e25 as e
from looped_bitnet import longer_native8_e20 as old
from scripts import qat_match_e25 as r


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp('e25') / 'qa'
    report = r.tiny_qa(out)
    return out, report


def test_initial_pairing():
    independent = json.loads((e.ROOT / 'results/E25_INITIAL_REFERENCE.json').read_text())
    for item in independent['seeds']:
        model = e.build_initial_model(item['seed'])
        assert old.digest_state_dict(model) == item['initial_digest']
        assert old.digest_object(model.initial_rng_state) == item['float_rng_digest']
        assert list(e.QAT_NAMES) == item['bitlinear_locations']


def test_real_three_seed_qa(tiny):
    out, report = tiny
    assert report['status'] == 'complete_with_numerical_failures'
    assert list(report['seeds']) == ['0', '1', '2']
    assert all(row['reload_next_update_equal'] for row in report['seeds'].values())
    assert report['cost']['completed_updates'] == 3
    assert report['cost']['training_cost'] == dict(zip(e.COST_KEYS, (3, 6, 6, 48)))
    assert report['cost']['evaluation_cost'] == dict(zip(e.COST_KEYS, (3, 6, 6, 48)))
    assert report['cost']['reload_identity_cost'] == dict(zip(e.COST_KEYS, (6, 12, 12, 96)))
    with pytest.raises(FileExistsError):
        r.tiny_qa(out)


def test_strict_reload_tamper(tiny, tmp_path):
    out, _ = tiny
    manifest = json.loads((out / 'manifest.json').read_text())
    original = torch.load(out / 'seed0/u1.pt', weights_only=True)
    for key, value in [('seed', 1), ('update', 2), ('qa', False), ('qat_names', []),
                       ('initial_digest', 'bad'), ('manifest_digest', 'bad'), ('training_mode', False),
                       ('completed_updates', 2), ('training_cost', {})]:
        changed = deepcopy(original); changed[key] = value
        path = tmp_path / f'{key}.pt'; torch.save(changed, path)
        with pytest.raises((ValueError, KeyError)):
            e.load_checkpoint(path, manifest, seed=0, update=1, qa=True)
    model, optimizer, payload = e.load_checkpoint(out / 'seed0/u1.pt', manifest, seed=0, update=1, qa=True)
    assert torch.equal(torch.get_rng_state(), payload['rng_state'])


def test_paired_report_identities_and_threshold():
    baseline = json.loads((e.ROOT / e.FLOAT_RUN / 'report.json').read_text())['seeds']['0']['evaluation']
    same = r.compare(baseline, baseline)
    for row in same['composition']['primary']:
        assert row['wins_parent_wrong_new_correct'] == row['losses_parent_correct_new_wrong'] == 0
    changed = deepcopy(baseline)
    changed['composition']['primary_rows'][0]['predictions'][0]['stratum'] = 'bad'
    with pytest.raises(ValueError, match='stratum'):
        r.compare(changed, baseline)
    changed = deepcopy(baseline)
    changed['composition']['primary_rows'][0]['predictions'][0]['target_trace'] = []
    with pytest.raises(ValueError, match='target'):
        r.compare(changed, baseline)
    changed = deepcopy(baseline)
    for row in changed['composition']['primary_rows']:
        row['metrics']['all']['final_joint'] = 244
    changed['composition']['primary_conjunction'] = True
    assert r.accepted.predicates(changed['seen'], changed['composition'])['primary_conjunction']
    changed['composition']['control_row']['metrics']['all']['final_joint'] = 0
    assert r.accepted.predicates(changed['seen'], changed['composition'])['primary_conjunction']
    changed['composition']['primary_rows'][0]['metrics']['all']['final_joint'] = 243
    changed['composition']['primary_conjunction'] = False
    assert not r.accepted.predicates(changed['seen'], changed['composition'])['primary_conjunction']


def test_actual_inventory_tamper():
    model = e.build_initial_model(0)
    model.reader.q_proj = torch.nn.Linear(64, 64)
    with pytest.raises(ValueError, match='projection inventory'):
        e.check_inventory(model)


def test_paired_state_marginal_and_direction():
    baseline = json.loads((e.ROOT / e.FLOAT_RUN / 'report.json').read_text())['seeds']['0']['evaluation']
    row = baseline['composition']['primary_rows'][0]
    changed = deepcopy(row)
    changed['predictions'][0]['state'] = changed['predictions'][1]['state']
    with pytest.raises(ValueError, match='duplicate'):
        r.accepted._paired_row(row, changed, label='test')
    changed = deepcopy(row)
    changed['metrics']['all']['final_joint'] -= 1
    with pytest.raises(ValueError, match='marginal'):
        r.accepted._paired_row(row, changed, label='test')
    changed = deepcopy(row)
    correct = next(v for v in changed['predictions'] if v['joint_final_correct'])
    correct['joint_final_correct'] = False
    changed['metrics']['all']['final_joint'] -= 1
    result = r.accepted._paired_row(row, changed, label='test')
    assert result['losses_parent_correct_new_wrong'] == 1
    assert result['wins_parent_wrong_new_correct'] == 0
    reverse = r.accepted._paired_row(changed, row, label='test')
    assert reverse['wins_parent_wrong_new_correct'] == 1
    assert reverse['losses_parent_correct_new_wrong'] == 0


def test_runtime_failure_suspends(tmp_path):
    batch = old.base_stream()[0][:2]
    model = e.build_initial_model(0)
    manifest = {'qa': True, 'initial_digests': {'0': old.digest_state_dict(model)},
                'initial_rng_digests': {'0': old.digest_object(model.initial_rng_state)},
                'checkpoint_costs': {'1': dict(zip(e.COST_KEYS, (1, 2, 2, 16)))}}
    def failure(model, manifest):
        raise RuntimeError('intentional QA evaluation failure before forward')
    out = tmp_path / 'failure'
    with pytest.raises(RuntimeError, match='intentional'):
        r.execute(out, manifest, [batch], failure, qa=True)
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'suspended'
    assert report['seeds']['0']['status'] == 'technical_failure'
    assert report['seeds']['1']['status'] == report['seeds']['2']['status'] == 'not_started'
    assert report['cost']['completed_updates'] == 1
    assert report['cost']['training_cost']['program_forwards'] == 1
    assert report['cost']['evaluation_cost']['program_forwards'] == 0
    assert (out / 'seed0/u1.pt').exists()
