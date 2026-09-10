#!/usr/bin/env python3
"""Two fixed E22 runs; QA controls are private, never CLI budgets."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import argparse
import json
import sys
import time
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_bitnet import replication_e22 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.runtime import environment, seed_everything
from looped_bitnet.register_e15 import forbidden
from scripts import composition_e21 as c
from scripts.longer_native8_e20 import _update, evaluate as evaluate_seen
from scripts.step_budget_e18 import _atomic_json, _read_json


def protected():
    hashes = _read_json(e.ROOT/e.PROTECTED)['sha256']
    if len(hashes) != 206: raise ValueError('protected count')
    for path, digest in hashes.items():
        if old.sha256_file(e.ROOT/path) != digest: raise ValueError(f'protected {path}')


def expected_manifest():
    protected()
    sources = dict(c.SOURCE_RELATIVE_PATHS)
    sources.update(e22_module='looped_bitnet/replication_e22.py', e22_runner='scripts/replication_e22.py',
                   e22_tests='tests/test_replication_e22.py', e22_protocol='results/E22_REPLICATION_PROTOCOL.md')
    model0 = e.build_initial_model(0)
    reference = torch.load(e.ROOT/old.E18_INITIAL_STATE, weights_only=True)['state_dict']
    if list(reference) != list(model0.state_dict()) or any(reference[k].dtype != model0.state_dict()[k].dtype or reference[k].shape != model0.state_dict()[k].shape or not torch.equal(reference[k], model0.state_dict()[k]) for k in reference):
        raise ValueError('seed0 initial bitwise mismatch')
    initials = {str(s): old.digest_state_dict(e.build_initial_model(s)) for s in e.SEEDS}
    if len(set(initials.values()) | {old.digest_state_dict(model0)}) != 3: raise ValueError('initial seeds not distinct')
    rngs = {str(s): old.digest_object(e.build_initial_model(s).initial_rng_state) for s in e.SEEDS}
    independent = _read_json(e.ROOT/'results/E22_INITIALIZATION_REFERENCE.json')
    for item in independent['seeds']:
        model = e.build_initial_model(item['seed'])
        if old.digest_state_dict(model) != item['initial_digest'] or hashlib.sha256(model.initial_rng_state.numpy().tobytes()).hexdigest() != item['post_constructor_rng_sha256']:
            raise ValueError('independent initialization reference')
    batches = old.full_stream()
    seen = _read_json(e.ROOT/old.DEFAULT_E20_PREFLIGHT/'manifest.json')
    symbolic = c.symbolic_audit()
    accepted = _read_json(e.ROOT/c.DEFAULT_PREFLIGHT/'manifest.json')
    if old.canonical_hash({k:v for k,v in symbolic.items() if k != 'reference'}) != old.canonical_hash(accepted['symbolic']): raise ValueError('E21 symbolic identity')
    return {'schema': e.SCHEMA, 'config': e.CONFIG, 'source_hashes': {k: old.sha256_file(e.ROOT/v) for k,v in sources.items()},
            'protected_snapshot_hash': old.sha256_file(e.ROOT/e.PROTECTED),
            'reference_hashes': {str(p): old.sha256_file(e.ROOT/p) for p in
                (old.E18_INITIAL_STATE, old.DEFAULT_E20_PREFLIGHT/'manifest.json', c.DEFAULT_PREFLIGHT/'manifest.json', c.E15_PREFLIGHT)},
            'initial_rng_digests': rngs, 'initialization_reference_hash': old.sha256_file(e.ROOT/'results/E22_INITIALIZATION_REFERENCE.json'),
            'initial_digests': initials, 'seed0_initial_digest': old.digest_state_dict(model0),
            'base_stream_digest': old.E18_BASE_STREAM_DIGEST, 'base_target_digest': old.E18_BASE_TARGET_DIGEST,
            'full_stream_digest': old.batch_digest(batches), 'full_target_digest': old.target_digest(batches),
            'schedule': old.phase_boundaries(), 'seen': {'programs': seen['programs'], 'state_split': seen['state_split']},
            'symbolic': symbolic, 'train_cost': e.TRAIN_COST, 'evaluation_cost': e.EVAL_COST}


def preflight(path=e.PREFLIGHT):
    path=e.ROOT/path; old.refuse_nonempty(path); manifest=expected_manifest(); path.mkdir(parents=True,exist_ok=True)
    _atomic_json(path/'manifest.json',manifest,refuse=True)
    for seed in e.SEEDS:
        model=e.build_initial_model(seed)
        old.atomic_torch_save(path/f'initial_seed{seed}.pt', {'state_dict': model.state_dict(), 'digest': old.digest_state_dict(model), 'rng_state': model.initial_rng_state})
    protected(); return manifest


def load_manifest(path=e.PREFLIGHT):
    path=e.ROOT/path; manifest=_read_json(path/'manifest.json')
    if old.canonical_hash(manifest) != old.canonical_hash(expected_manifest()): raise ValueError('frozen manifest changed')
    for seed in e.SEEDS:
        saved=torch.load(path/f'initial_seed{seed}.pt',weights_only=True); model=e.build_initial_model(seed)
        if list(saved['state_dict']) != list(model.state_dict()) or any(saved['state_dict'][k].dtype != v.dtype or saved['state_dict'][k].shape != v.shape or not torch.equal(saved['state_dict'][k],v) for k,v in model.state_dict().items()) or saved['digest'] != manifest['initial_digests'][str(seed)]:
            raise ValueError('frozen initial changed')
        if not torch.equal(saved['rng_state'], model.initial_rng_state) or old.digest_object(saved['rng_state']) != manifest['initial_rng_digests'][str(seed)]:
            raise ValueError('frozen initial RNG changed')
    return manifest


def predicates(seen, composition):
    rows=seen['validation']
    prerequisite=all(row['final_joint'] >= (32 if len(row['program']) == 1 else 31) for row in rows)
    primary=composition.get('primary_conjunction')
    return {'seen_prerequisite': prerequisite, 'primary_conjunction': primary,
            'combined_conjunction': prerequisite and primary if primary is not None else None}


def evaluate(model, manifest):
    seen=evaluate_seen(model,manifest['seen'])
    composition=c._report(c.evaluate_model(model,manifest))
    return {'seen': seen, 'composition': composition, **predicates(seen,composition)}


@dataclass(frozen=True)
class _TinySpec:
    batches: list
    evaluate: object
    progress_interval: int = 1


def _train(out, manifest, *, _qa=None):
    out=Path(out); old.refuse_nonempty(out); out.mkdir(parents=True,exist_ok=True)
    batches=old.full_stream() if _qa is None else _qa.batches
    if _qa is not None and (not batches or any(forbidden(x.program) for b in batches for x in b)):
        raise ValueError('QA must use nonempty legal seen data')
    evaluator=evaluate if _qa is None else _qa.evaluate
    _atomic_json(out/'manifest.json',manifest,refuse=True)
    report={'schema': e.SCHEMA, 'qa': _qa is not None, 'status': 'running', 'seeds': {},
            'environment': environment(torch.device('cpu'))}
    try:
        for seed in e.SEEDS:
            seed_everything(seed, deterministic=True,cpu_threads=4)
            model=e.build_initial_model(seed); optimizer=old.make_optimizer(model); model.train()
            torch.set_rng_state(model.initial_rng_state)
            if old.digest_state_dict(model) != manifest['initial_digests'][str(seed)]: raise ValueError('initial digest')
            if {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in model.parameters()}: raise ValueError('optimizer membership')
            folder=out/f'seed{seed}'; folder.mkdir()
            record={'status': 'running', 'completed_updates': 0, 'examples': 0, 'internal_state_substeps': 0, 'progress': [],
                    'initial_rng_digest': old.digest_object(torch.get_rng_state())}
            report['seeds'][str(seed)]=record; started=time.monotonic()
            for update,batch in enumerate(batches,1):
                loss=_update(model,optimizer,batch)
                record.update(completed_updates=update, examples=record['examples']+len(batch),
                              internal_state_substeps=record['internal_state_substeps']+8*sum(len(x.program) for x in batch))
                if update % (250 if _qa is None else _qa.progress_interval) == 0 or update == len(batches):
                    record['progress'].append({'update': update, 'loss': float(loss.detach()), 'elapsed_seconds': time.monotonic()-started})
                    _atomic_json(out/'progress.json',report)
            record['training_seconds']=time.monotonic()-started
            payload=e.checkpoint_payload(model,optimizer,manifest,seed,len(batches),qa=_qa is not None)
            old.atomic_torch_save(folder/f'u{len(batches)}.pt',payload)
            before=old.digest_object((model.state_dict(),optimizer.state_dict(),torch.get_rng_state())); started=time.monotonic()
            record['evaluation']=evaluator(model,manifest)
            if not model.training or before != old.digest_object((model.state_dict(),optimizer.state_dict(),torch.get_rng_state())):
                raise ValueError('evaluation mutated training state')
            record.update(status='complete', evaluation_seconds=time.monotonic()-started)
            _atomic_json(folder/'report.json',record,refuse=True)
            _atomic_json(out/'progress.json',report)
        report.update(status='complete', primary_replication=all(r['evaluation']['primary_conjunction'] is True for r in report['seeds'].values()),
                      combined_replication=all(r['evaluation']['combined_conjunction'] is True for r in report['seeds'].values()))
        if _qa is None: report.update(train_cost=e.TRAIN_COST,evaluation_cost=e.EVAL_COST)
        _atomic_json(out/'report.json',report,refuse=True); return report
    except Exception as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}')
        for pending in e.SEEDS:
            if str(pending) not in report['seeds']:
                report['seeds'][str(pending)]={'status': 'not_started', 'reason': 'suspended after technical failure'}
            elif report['seeds'][str(pending)]['status'] == 'running':
                report['seeds'][str(pending)].update(status='incomplete',error=report['error'])
        _atomic_json(out/'progress.json',report); raise


def run(out=e.RUN,pre=e.PREFLIGHT):
    manifest=load_manifest(pre)
    try: return _train(e.ROOT/out,manifest)
    finally: protected()


def recovery(out=e.RUN,pre=e.PREFLIGHT):
    manifest=load_manifest(pre); result={'status': 'evaluation_only', 'evaluations': {}, 'missing': []}
    for seed in e.SEEDS:
        path=e.ROOT/out/f'seed{seed}'/'u8000.pt'
        if not path.exists(): result['missing'].append(seed); continue
        model,_,_=e.load_checkpoint(path,manifest,seed=seed)
        result['evaluations'][str(seed)]=evaluate(model,manifest)
    result['all_finals_present']=not result['missing']; protected(); return result


def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true');g.add_argument('--train-cleared',action='store_true');g.add_argument('--eval-only',action='store_true')
    p.add_argument('--out',type=Path,default=e.RUN);p.add_argument('--preflight-dir',type=Path,default=e.PREFLIGHT);a=p.parse_args()
    result=preflight(a.preflight_dir) if a.preflight else run(a.out,a.preflight_dir) if a.train_cleared else recovery(a.out,a.preflight_dir)
    print(json.dumps(result,sort_keys=True))
if __name__ == '__main__': main()
