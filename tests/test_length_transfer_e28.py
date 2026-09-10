import copy
import json
from pathlib import Path
import pytest
import torch
from scripts import length_transfer_e28 as e

@pytest.fixture(scope='module')
def selected(): return e.selection()['programs']

def rows(selected):
    membership={tuple(s):k for k,ss in e._strata().items() for s in ss}
    return [dict(program=p['program'],predictions=[dict(state=c['state'],target_trace=c['target_trace'],predicted_trace=copy.deepcopy(c['target_trace']),prefix_joint_correct=[True]*p['length'],joint_final_correct=True,stratum=membership[tuple(c['state'])]) for c in p['cases']]) for p in selected]

def test_selection_and_boundary(selected):
    assert len(selected)==12
    assert e.trace(['ADD','XOR','SWAP'],[15,1])==[[0,1],[1,1],[1,1]]
    assert e.trace(['XOR','SWAP'],[15,8])==[[7,8],[8,7]]

def test_selection_reject(tmp_path):
    (tmp_path/'results').mkdir(); s=e.read(e.ROOT/'results/E28_SELECTION.json');s['programs'][0]['cases'][0]['target_trace'][0][0]=1
    (tmp_path/'results/E28_SELECTION.json').write_text(json.dumps(s))
    with pytest.raises(ValueError): e.selection(tmp_path)

def test_threshold_divergence_recovery_pair(selected):
    a=rows(selected)
    for n in range(13):
        v=a[0]['predictions'][n];v['predicted_trace'][-1][0]^=1;v['prefix_joint_correct'][-1]=False;v['joint_final_correct']=False
    scored=e.score(a,selected,e._strata());assert not scored['primary_by_length']['4']
    v=a[0]['predictions'][12];v['predicted_trace'][-1]=v['target_trace'][-1][:];v['prefix_joint_correct'][-1]=True;v['joint_final_correct']=True
    v=a[0]['predictions'][20];v['predicted_trace'][0][0]^=1;v['prefix_joint_correct'][0]=False
    scored=e.score(a,selected,e._strata());m=scored['rows'][0]['metrics']['all']
    assert scored['primary_by_length']['4'] and not scored['secondary_by_length']['4']
    assert m['final_joint']==244 and m['full_trace']==243 and m['recovery']==1 and m['first_divergence']['1']==1
    baseline=e.score(rows(selected),selected,e._strata());paired=e.paired(scored,baseline)[0]['metrics']['all']
    assert paired['final_joint']['float_wins']==12 and paired['full_trace']['float_wins']==13
    baseline['rows'][0]['predictions'][0]['state']=[1,1]
    with pytest.raises(ValueError):e.paired(scored,baseline)

@pytest.mark.parametrize('kind',['scope','state','stratum','prediction','target'])
def test_malformed(selected,kind):
    a=rows(selected)
    if kind=='scope':a.pop()
    elif kind=='state':a[0]['predictions'].reverse()
    elif kind=='stratum':a[0]['predictions'][0]['stratum']='bad'
    elif kind=='prediction':a[0]['predictions'][0]['predicted_trace'][0][0]=True
    else:a[0]['predictions'][0]['target_trace'][0][0]=17
    with pytest.raises(ValueError):e.score(a,selected,e._strata())

class Fake(torch.nn.Module):
    def __init__(self,kind): super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.kind=kind
    def forward(self,x,y,ops):
        if self.kind=='exception':raise RuntimeError('injected')
        v=torch.zeros(len(x),ops.shape[1],16);v[0,0,0]=float('nan');return v,v

@pytest.mark.parametrize('kind',['exception','nonfinite'])
def test_failures(kind):
    model=Fake(kind);opt=torch.optim.AdamW(model.parameters());record={k:dict.fromkeys(e.KEYS,0) for k in ('attempted','completed')}
    with pytest.raises((RuntimeError,FloatingPointError)):e.evaluate(model,opt,[['ADD']],[(0,0)],record)
    assert record['attempted']['program_forwards']==1
    assert record['completed']['program_forwards']==(kind=='nonfinite')
    assert model.training

def test_overwrite_and_guard(tmp_path):
    with pytest.raises(FileExistsError): e.execute(tmp_path)
    assert e.verify()
    with pytest.raises(ValueError): e.load('unknown',0)

@pytest.mark.parametrize('kind',['exception','nonfinite'])
def test_suspended_artifact(tmp_path,monkeypatch,kind):
    def fake_load(*args):
        model=Fake(kind);return model,torch.optim.AdamW(model.parameters()),{}
    monkeypatch.setattr(e,'load',fake_load)
    out=tmp_path/'failure'
    report=e.execute(out,qa=True)
    assert report['status']=='suspended' and report['training_updates']==0
    assert report['cost']['attempted']['program_forwards']==1
    assert e.read(out/'report.json')==report
    with pytest.raises(FileExistsError):e.execute(out,qa=True)

def test_real_qa_evidence():
    report=e.read(e.ROOT/'runs/e28_qa_seen_attempt1/report.json')
    assert report['status']=='complete' and report['qa'] and report['training_updates']==0
    assert report['cost']['attempted']==report['cost']['completed']==dict(zip(e.KEYS,(6,12,12,96)))
    assert all(r['unchanged'] for seeds in report['models'].values() for r in seeds.values())
