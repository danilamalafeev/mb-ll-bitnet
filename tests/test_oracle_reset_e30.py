import copy
import pytest
import torch
from scripts import oracle_reset_e30 as o
from scripts import length_transfer_e28 as e

def fresh():
    return [dict(program=[op],predictions=[dict(state=s,target_trace=e.trace([op],s),predicted_trace=e.trace([op],s),prefix_joint_correct=[True],joint_final_correct=True) for s in e.STATES]) for op in o.OPS]

def test_keyed_lift_true_not_decoded():
    rows=o.saved()['E27_W4',0]; f=fresh()
    for r in f:r['predictions'].reverse()
    result=o.lift(rows,o.primitive_table(f));assert len(result['cases'])==1536
    assert result['paired']['all']['all']['fresh_correct']==1536
    assert result['paired']['prefix4_correct']['all']['saved_errors']==54
    for c in result['cases']:
        assert c['true_prefix_state']==e.trace(c['program'][:4],c['initial_state'])[-1]
    altered=copy.deepcopy(rows);p=altered[0]['predictions'][0];p['predicted_trace'][3]=[15,15];p['prefix_joint_correct'][3]=False
    c=o.lift(altered,o.primitive_table(f))['cases'][0]
    assert c['true_prefix_state']==e.trace(c['program'][:4],c['initial_state'])[-1] and not c['prefix4_all_correct']

@pytest.mark.parametrize('kind',['missing','duplicate','target','bool','scope','flag'])
def test_fresh_reject(kind):
    f=fresh();p=f[0]['predictions'][0]
    if kind=='missing':f[0]['predictions'].pop()
    elif kind=='duplicate':f[0]['predictions'].append(p)
    elif kind=='target':p['target_trace']=[[3,4]]
    elif kind=='bool':p['state']=[True,0]
    elif kind=='scope':f[0]['program']=['ADD']
    else:p['joint_final_correct']=1
    with pytest.raises(ValueError):o.primitive_table(f)

def test_pairs_and_strata_zero():
    cases=[dict(long_final_correct=a,fresh_correct=b,prefix4_all_correct=prefix,initial_stratum=s) for a,b,prefix,s in [(False,True,True,'train'),(True,False,True,'test'),(False,False,False,'validation'),(True,True,True,'train')]]
    r=o.summaries(cases)
    assert r['all']['all']==dict(denominator=4,recovered=1,introduced=1,both_correct=1,both_wrong=1,long_correct=2,fresh_correct=2,saved_errors=2,rescued_fraction=.5)
    assert r['prefix4_correct']['all']['saved_errors']==1
    assert r['prefix4_wrong']['train']['rescued_fraction'] is None

class Fake(torch.nn.Module):
    def __init__(self,kind):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.child=torch.nn.Dropout();self.child.eval();self.kind=kind
    def forward(self,x,y,ops):
        if self.kind=='exception':raise RuntimeError('injected')
        if self.kind=='mutate':self.p.data.add_(1)
        v=torch.zeros(len(x),ops.shape[1],16)
        if self.kind=='nonfinite':v[0,0,0]=float('nan')
        return v,v

@pytest.mark.parametrize('kind',['exception','nonfinite','mutate'])
def test_exception_accounting_and_immutability(kind):
    model=Fake(kind);opt=torch.optim.AdamW(model.parameters());rec={p:dict.fromkeys(e.KEYS,0) for p in ('attempted','completed')}
    with pytest.raises((RuntimeError,ValueError,FloatingPointError)):o.evaluate(model,opt,[['ADD']],[(0,0)],rec)
    assert model.training and not model.child.training
    assert rec['attempted']['program_forwards']==1
    assert rec['completed']['program_forwards']==(kind!='exception')


def test_overwrite_and_guard(tmp_path):
    with pytest.raises(FileExistsError):o.execute(tmp_path,qa=True)
    assert o.verify()


def test_failure_artifacts(tmp_path,monkeypatch):
    def loader(*args):
        m=Fake('exception');return m,torch.optim.AdamW(m.parameters()),{}
    monkeypatch.setattr(e,'load',loader)
    r=o.execute(tmp_path/'failure',qa=True)
    assert r['status']=='suspended' and r['protected_unchanged']
    assert e.read(tmp_path/'failure/report.json')==r
    assert r['cost']['attempted']['program_forwards']==1

@pytest.mark.parametrize('kind',['seed','stratum','target','program'])
def test_saved_scope_reject(monkeypatch,kind):
    original=e.read; report=original(e.ROOT/'runs/e28_length/report.json')
    if kind=='seed':report['models']['E27_W4'].pop('2')
    else:
        row=report['models']['E27_W4']['0']['evaluation']['rows'][0]
        if kind=='stratum':row['predictions'][0]['stratum']='test' if row['predictions'][0]['stratum']!='test' else 'train'
        elif kind=='target':row['predictions'][0]['target_trace'][0][0]^=1
        else:row['program']=['ADD']
    monkeypatch.setattr(e,'read',lambda path:report if str(path).endswith('runs/e28_length/report.json') else original(path))
    with pytest.raises(ValueError):o.saved()


def test_real_qa_cost():
    r=e.read(e.ROOT/'runs/e30_qa_seen_attempt1/report.json')
    assert r['status']=='complete' and r['qa'] and r['training_updates']==0
    assert r['cost']['attempted']==r['cost']['completed']==dict(zip(e.KEYS,(6,12,12,96)))
    assert all(rec['unchanged'] and rec['all_modes_restored'] and rec['external_rng_unchanged'] for seeds in r['models'].values() for rec in seeds.values())
