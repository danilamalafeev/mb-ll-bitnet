#!/usr/bin/env python3
"""E20 fixed longer native8 run; no scientific budget CLI overrides."""
from pathlib import Path
import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_bitnet import longer_native8_e20 as e
from looped_bitnet.register_e15 import loss_for_batch, sha256_file
from looped_bitnet.step_budget_e18 import evaluate_split, reconstruct_training_coverage
from looped_bitnet.runtime import seed_everything
from scripts.step_budget_e18 import _atomic_json, _read_json


def protected():
    hashes = _read_json(e.PROJECT_ROOT/e.DEFAULT_PROTECTED_SNAPSHOT)['sha256']
    if len(hashes) != 176: raise ValueError('protected list size')
    for path, digest in hashes.items():
        if sha256_file(e.PROJECT_ROOT/path) != digest: raise ValueError(f'protected file {path}')


def expected_manifest():
    protected()
    model, initial, common = e.build_initial_model()
    old = _read_json(e.PROJECT_ROOT/e.DEFAULT_E18_PREFLIGHT/'manifest.json')
    report = _read_json(e.PROJECT_ROOT/e.DEFAULT_E18_REPORT)
    coverage = reconstruct_training_coverage(e.base_stream(), old)
    if not coverage['all_6144_exposed']: raise ValueError('coverage')
    return e.make_manifest(e18_manifest=old, e18_report=report, initial_digest=initial,
                           common_digest=common, coverage=coverage)


def preflight(path=e.DEFAULT_E20_PREFLIGHT):
    path = e.PROJECT_ROOT/path; e.refuse_nonempty(path)
    manifest = expected_manifest(); model, _, _ = e.build_initial_model()
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path/'manifest.json', manifest, refuse=True)
    e.atomic_torch_save(path/'initial.pt', {'state_dict': model.state_dict(), 'digest': e.digest_state_dict(model)})
    protected(); return manifest


def load_manifest(path):
    path = e.PROJECT_ROOT/path
    manifest = _read_json(path/'manifest.json'); expected = expected_manifest()
    # Environment is descriptive; frozen scientific fields are exact.
    if e.canonical_hash({k:v for k,v in manifest.items() if k!='environment'}) != e.canonical_hash({k:v for k,v in expected.items() if k!='environment'}):
        raise ValueError('frozen manifest mismatch')
    initial = torch.load(path/'initial.pt', weights_only=True)
    model, _, _ = e.build_initial_model(); model.load_state_dict(initial['state_dict'], strict=True)
    if e.digest_state_dict(model) != manifest['initial_digest'] or initial['digest'] != manifest['initial_digest']:
        raise ValueError('saved initial mismatch')
    return manifest


def evaluate(model, manifest):
    was_training = model.training; model.eval()
    try:
        rows = {s:evaluate_split(model, manifest, s) for s in ('train','validation')}
        rows['macros'] = e.aggregate_metrics(rows['train'], rows['validation'])
        return rows
    finally: model.train(was_training)


@dataclass(frozen=True)
class _TinySpec:
    """Private QA injection; never exposed through CLI or scientific run()."""
    batches: list
    milestones: tuple
    prefix: int
    reference: dict
    evaluate: Callable


def _update(model,optimizer,batch):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cpu',enabled=False): loss=loss_for_batch(model,batch)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
    return loss


def _train(out, manifest, *, _qa=None):
    out=Path(out); e.refuse_nonempty(out);out.mkdir(parents=True,exist_ok=True)
    seed_everything(0, deterministic=True, cpu_threads=4)
    model, _, _ = e.build_initial_model(); optimizer=e.make_optimizer(model);model.train()
    batches=e.full_stream() if _qa is None else _qa.batches
    milestones=e.MILESTONES if _qa is None else _qa.milestones
    prefix=2000 if _qa is None else _qa.prefix
    reference=torch.load(e.PROJECT_ROOT/e.DEFAULT_E18_CHECKPOINT,weights_only=True)['state_dict'] if _qa is None else _qa.reference
    evaluator=evaluate if _qa is None else _qa.evaluate
    record={'schema':e.E20_SCHEMA,'qa':_qa is not None,'status':'running','completed_updates':0,'progress':[], 'evaluations':{}}
    _atomic_json(out/'manifest.json',manifest,refuse=True)
    try:
        for update,batch in enumerate(batches,1):
            loss=_update(model,optimizer,batch)
            record['completed_updates']=update
            if update%250==0 or update in milestones:
                record['progress'].append({'update':update,'loss':float(loss.detach())})
                _atomic_json(out/'progress.json',record)
            if update in milestones:
                payload=e.checkpoint_payload(model,optimizer,rng_state=torch.get_rng_state(),update=update,manifest=manifest,qa=_qa is not None)
                e.atomic_torch_save(out/f'u{update}.pt',payload)
            if update==prefix:
                state=model.state_dict()
                if list(state)!=list(reference) or any(state[k].dtype!=reference[k].dtype or state[k].shape!=reference[k].shape or not torch.equal(state[k],reference[k]) for k in state):
                    raise ValueError('E18 prefix bitwise mismatch; stopped before next update')
                record['prefix_bitwise_equal']=True
            if update in milestones:
                before=e.digest_object((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
                record['evaluations'][str(update)]=evaluator(model,manifest)
                if not model.training or before!=e.digest_object((model.state_dict(),optimizer.state_dict(),torch.get_rng_state())):
                    raise ValueError('milestone inference mutated training state')
                _atomic_json(out/'progress.json',record)
        record.update(status='complete',fixed_final_update=len(batches),train_cost=e.TRAIN_COST if _qa is None else {'qa_updates':len(batches)},
                      evaluation_cost=e.EVAL_COST_TOTAL if _qa is None else {'qa_milestones':list(milestones)})
        if _qa is None:record.update(comparisons=comparisons(record['evaluations']))
        _atomic_json(out/'report.json',record,refuse=True)
        return record
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');_atomic_json(out/'progress.json',record);raise


def comparisons(evaluations):
    ref=_read_json(e.PROJECT_ROOT/e.DEFAULT_E18_REPORT)['evaluations']['steps4']['macros']
    base=evaluations['2000']['macros']; result={}
    for u,obs in evaluations.items():
        result[u]={}
        for split in ('train','validation'):
            result[u][split]={g:{'minus_2000_pp':100*(obs['macros'][split][g]['rates']['final_joint']-base[split][g]['rates']['final_joint']),
                'minus_steps4_u2000_pp':100*(obs['macros'][split][g]['rates']['final_joint']-ref[split][g]['rates']['final_joint'])}
                for g in ('1','2','3','primitives','seen_compositions')}
        result[u]['length3_catch_up']={s:obs['macros'][s]['3']['rates']['final_joint']>=ref[s]['3']['rates']['final_joint'] for s in ('train','validation')}
    return result


def run(out=e.DEFAULT_E20_RUN, pre=e.DEFAULT_E20_PREFLIGHT):
    manifest=load_manifest(pre); result=_train(e.PROJECT_ROOT/out,manifest);protected();return result


def recovery(out=e.DEFAULT_E20_RUN, pre=e.DEFAULT_E20_PREFLIGHT):
    manifest=load_manifest(pre);out=e.PROJECT_ROOT/out; result={'status':'evaluation_only','evaluations':{},'missing':[]}
    for update in e.MILESTONES:
        path=out/f'u{update}.pt'
        if not path.exists():result['missing'].append(update);continue
        model,_,_=e.load_checkpoint(path,manifest,expected_update=update)
        result['evaluations'][str(update)]=evaluate(model,manifest)
    result['final_present']=8000 not in result['missing'];protected();return result


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true');g.add_argument('--train-cleared',action='store_true');g.add_argument('--eval-only',action='store_true')
    p.add_argument('--out',type=Path,default=e.DEFAULT_E20_RUN);p.add_argument('--preflight-dir',type=Path,default=e.DEFAULT_E20_PREFLIGHT)
    a=p.parse_args()
    if a.preflight:result=preflight(a.preflight_dir)
    elif a.train_cleared:result=run(a.out,a.preflight_dir)
    else:result=recovery(a.out,a.preflight_dir)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
