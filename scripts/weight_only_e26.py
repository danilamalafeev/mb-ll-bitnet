#!/usr/bin/env python3
"""E26 fixed scientific runner and separately named legal-seen QA runner."""
from pathlib import Path
import argparse
import json
import sys
import time
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_bitnet import weight_only_e26 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.runtime import seed_everything
from scripts import continuation_e24 as accepted
from scripts.longer_native8_e20 import _update


def preflight(path=e.PREFLIGHT, *, root=e.ROOT):
    path = root / path
    old.refuse_nonempty(path)
    manifest = e.make_manifest(root)
    path.mkdir(parents=True, exist_ok=True)
    accepted._atomic_json(path / 'manifest.json', manifest, refuse=True)
    for seed in e.SEEDS:
        model = e.build_initial_model(seed)
        old.atomic_torch_save(path / f'initial_seed{seed}.pt',
            {'state_dict': model.state_dict(), 'rng_state': model.initial_rng_state})
    return manifest


def load_manifest(path=e.PREFLIGHT, *, root=e.ROOT):
    manifest = accepted._read_json(root / path / 'manifest.json')
    if manifest != e.make_manifest(root):
        raise ValueError('E26 frozen manifest changed')
    for seed in e.SEEDS:
        saved = torch.load(root / path / f'initial_seed{seed}.pt', weights_only=True)
        model = e.build_initial_model(seed)
        e.assert_state_equal(saved['state_dict'], model.state_dict())
        if not torch.equal(saved['rng_state'], model.initial_rng_state):
            raise ValueError('initial RNG changed')
    return manifest


def compare(current, baseline):
    result = {'composition': accepted._composition_comparison(baseline['composition'], current['composition']),
              'seen': accepted._seen_comparison(baseline['seen'], current['seen'])}
    for split, rows in result['seen'].items():
        left = {tuple(r['program']): r for r in baseline['seen'][split]}
        right = {tuple(r['program']): r for r in current['seen'][split]}
        for row in rows:
            key = tuple(row['program'])
            a, b = accepted._prediction_map_seen(left[key]), accepted._prediction_map_seen(right[key])
            outcomes = {'wins': 0, 'losses': 0, 'both_correct': 0, 'both_wrong': 0}
            for state in a:
                ac = bool(a[state]['joint_final_correct']); bc = bool(b[state]['joint_final_correct'])
                outcomes['both_correct' if ac and bc else 'losses' if ac else 'wins' if bc else 'both_wrong'] += 1
            if outcomes['both_correct'] + outcomes['losses'] != row['parent_final_joint'] or outcomes['both_correct'] + outcomes['wins'] != row['new_final_joint']:
                raise ValueError('seen paired marginals mismatch')
            row.update(outcomes)
            row['delta_final_joint_rate'] = row['delta_final_joint'] / row['denominator']
    for row in [*result['composition']['primary'], result['composition']['control']]:
        row['delta_final_joint_rate'] = row['delta']['final_joint'] / row['denominator']
    return result


def compare_both(current, seed, *, root=e.ROOT):
    return {label: compare(current, accepted._read_json(root / path / 'report.json')['seeds'][str(seed)]['evaluation'])
            for label, path in (('E25_QAT', e.QAT_RUN), ('E24_float', e.FLOAT_RUN))}


def zero_cost():
    return dict.fromkeys(e.COST_KEYS, 0)


def count_forward(counter, model, args):
    x, y, ops = args
    n, length = int(x.shape[0]), int(ops.shape[1])
    for key, value in zip(e.COST_KEYS, (1, n, n * length, n * length * 8)):
        counter[key] += value


def hooks(model, record, phase):
    attempted = record.setdefault(phase + '_attempted', zero_cost())
    pre = model.register_forward_pre_hook(lambda m, a: count_forward(attempted, m, a))
    def completed(m, args, output):
        count_forward(record[phase], m, args)
        if not isinstance(output, tuple) or not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in output):
            raise FloatingPointError('nonfinite model output')
    post = model.register_forward_hook(completed)
    return pre, post


def costs(report):
    records = list(report['seeds'].values())
    return {'attempted_updates': sum(r.get('attempted_updates', 0) for r in records),
            'completed_updates': sum(r.get('completed_updates', 0) for r in records),
            'completed_reload_identity_updates': sum(r.get('completed_reload_identity_updates', 0) for r in records),
            **{phase: {k: sum(r.get(phase, {}).get(k, 0) for r in records) for k in e.COST_KEYS}
               for phase in ('training_cost', 'evaluation_cost', 'reload_identity_cost',
                             'training_cost_attempted', 'evaluation_cost_attempted', 'reload_identity_cost_attempted')}}


def execute(out, manifest, batches, evaluator, *, qa, root=e.ROOT):
    """Shared mechanics; scientific run supplies only registered inputs."""
    out = Path(out)
    old.refuse_nonempty(out)
    if not batches or any(tuple(x.program) not in accepted.SEEN_PROGRAMS for b in batches for x in b):
        raise ValueError('training must use legal seen programs')
    if not qa and (len(batches) != e.UPDATES or old.batch_digest(batches) != manifest['full_stream_digest'] or old.target_digest(batches) != manifest['full_target_digest']):
        raise ValueError('scientific stream mismatch')
    out.mkdir(parents=True, exist_ok=True)
    accepted._atomic_json(out / 'manifest.json', manifest, refuse=True)
    report = {'schema': e.SCHEMA, 'qa': qa, 'status': 'running', 'seeds': {},
              'cost_note': 'Completed-forward costs exclude partial failed forwards; attempted workload is reported separately.'}
    try:
        for seed in e.SEEDS:
            record = {'status': 'running', 'completed_updates': 0, 'attempted_updates': 0,
                      'training_cost': zero_cost(), 'evaluation_cost': zero_cost(), 'reload_identity_cost': zero_cost(), 'progress': []}
            report['seeds'][str(seed)] = record
            seed_everything(seed, deterministic=True, cpu_threads=4)
            model = e.build_initial_model(seed); optimizer = old.make_optimizer(model)
            torch.set_rng_state(model.initial_rng_state); model.train()
            if old.digest_state_dict(model) != manifest['initial_digests'][str(seed)] or old.digest_object(torch.get_rng_state()) != manifest['initial_rng_digests'][str(seed)]:
                raise ValueError('initial provenance mismatch')
            folder = out / f'seed{seed}'; folder.mkdir()
            started = time.monotonic()
            active_hooks = hooks(model, record, 'training_cost')
            try:
                for update, batch in enumerate(batches, 1):
                    record['attempted_updates'] = update
                    loss = _update(model, optimizer, batch)
                    record['completed_updates'] = update
                    if update % 250 == 0 or update == len(batches):
                        record['progress'].append({'update': update, 'loss': float(loss.detach()), 'seconds': time.monotonic() - started})
                        accepted._atomic_json(out / 'progress.json', report)
            finally:
                [h.remove() for h in active_hooks]
            if record['training_cost'] != manifest['checkpoint_costs'][str(len(batches))]:
                raise ValueError('actual completed training cost mismatch')
            payload = e.checkpoint_payload(model, optimizer, manifest, seed, len(batches), qa=qa)
            path = folder / f'u{len(batches)}.pt'
            old.atomic_torch_save(path, payload)
            restored, restored_opt, restored_payload = e.load_checkpoint(path, manifest, seed=seed, update=len(batches), qa=qa)
            before = old.digest_object((restored.state_dict(), restored_opt.state_dict(), torch.get_rng_state()))
            active_hooks = hooks(restored, record, 'evaluation_cost')
            try:
                evaluation = evaluator(restored, manifest)
            finally:
                [h.remove() for h in active_hooks]
            if not restored.training or before != old.digest_object((restored.state_dict(), restored_opt.state_dict(), torch.get_rng_state())):
                raise ValueError('evaluation mutated state')
            record['evaluation'] = evaluation
            if not qa and record['evaluation_cost'] != e.EVAL_COST:
                raise ValueError('evaluation budget mismatch')
            record['predicates'] = ({'combined_conjunction': False, 'primary_conjunction': False, 'seen_prerequisite': False}
                if qa else accepted.predicates(evaluation['seen'], evaluation['composition']))
            if qa:
                # Both genuine next updates start at identical saved optimizer/RNG.
                next_rngs = []
                for candidate, opt in ((model, optimizer), (restored, restored_opt)):
                    torch.set_rng_state(restored_payload['rng_state'])
                    active_hooks = hooks(candidate, record, 'reload_identity_cost')
                    try:
                        _update(candidate, opt, batches[0])
                        record['completed_reload_identity_updates'] = record.get('completed_reload_identity_updates', 0) + 1
                    finally:
                        [h.remove() for h in active_hooks]
                    next_rngs.append(torch.get_rng_state().clone())
                if not torch.equal(*next_rngs):
                    raise ValueError('reload next RNG mismatch')
                e.assert_state_equal(model.state_dict(), restored.state_dict())
                if old.digest_object(optimizer.state_dict()) != old.digest_object(restored_opt.state_dict()):
                    raise ValueError('reload next optimizer mismatch')
                record['reload_next_update_equal'] = True
            else:
                record['paired_comparison'] = compare_both(evaluation, seed, root=root)
            record['status'] = 'complete'
            accepted._atomic_json(folder / 'report.json', record, refuse=True)
            accepted._atomic_json(out / 'progress.json', report)
        report['primary_success_all_seeds'] = all(r['predicates']['primary_conjunction'] for r in report['seeds'].values())
        report['combined_success_all_seeds'] = all(r['predicates']['combined_conjunction'] for r in report['seeds'].values())
        report['status'] = 'complete' if report['combined_success_all_seeds'] else 'complete_with_numerical_failures'
    except BaseException as exc:
        report['status'] = 'suspended'; report['error'] = f'{type(exc).__name__}: {exc}'
        for seed in e.SEEDS:
            report['seeds'].setdefault(str(seed), {'status': 'not_started'})
            if report['seeds'][str(seed)]['status'] == 'running':
                report['seeds'][str(seed)]['status'] = 'technical_failure'
        raise
    finally:
        report['cost'] = costs(report)
        accepted._atomic_json(out / 'report.json', report, refuse=True)
    return report


def run(out=e.RUN, preflight_dir=e.PREFLIGHT, *, root=e.ROOT):
    manifest = load_manifest(preflight_dir, root=root)
    try:
        return execute(root / out, manifest, e.stream(), accepted.evaluate, qa=False, root=root)
    finally:
        e.protected(root)


def tiny_qa(out):
    """Real three-seed QAT QA, one legal seen update and forward per seed."""
    batch = old.base_stream()[0][:2]
    models = {str(s): e.build_initial_model(s) for s in e.SEEDS}
    manifest = {'schema': e.SCHEMA, 'qa': True, 'initial_digests': {s: old.digest_state_dict(m) for s,m in models.items()},
                'initial_rng_digests': {s: old.digest_object(m.initial_rng_state) for s,m in models.items()},
                'checkpoint_costs': {'1': dict(zip(e.COST_KEYS, (1, len(batch), len(batch) * len(batch[0].program), len(batch) * len(batch[0].program) * 8)))}}
    def evaluate(model, manifest):
        from looped_bitnet.register_e15 import OP_TO_ID
        training = model.training
        try:
            model.eval()
            with torch.no_grad():
                x = torch.tensor([v.x for v in batch], dtype=torch.long)
                y = torch.tensor([v.y for v in batch], dtype=torch.long)
                ops = torch.tensor([[OP_TO_ID[o] for o in v.program] for v in batch], dtype=torch.long)
                output = model(x, y, ops)
                if not all(torch.isfinite(t).all() for t in output):
                    raise FloatingPointError('QA evaluation nonfinite')
            return {'legal_seen_cases': len(batch)}
        finally:
            model.train(training)
    return execute(out, manifest, [batch], evaluate, qa=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--train-cleared', action='store_true')
    args = parser.parse_args()
    result = preflight() if args.preflight else run()
    print(json.dumps({'status': result.get('status', 'preflight_ready')}))

if __name__ == '__main__':
    main()
