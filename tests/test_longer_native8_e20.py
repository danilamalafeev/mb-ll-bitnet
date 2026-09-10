from copy import deepcopy
import pytest
import torch
from looped_bitnet import longer_native8_e20 as e
from scripts import longer_native8_e20 as r


def noop(model,manifest):return {}


@pytest.fixture(scope='module')
def manifest(): return r.expected_manifest()


def test_frozen_stream_and_initial(manifest):
    batches=e.full_stream();base=e.base_stream()
    assert len(batches)==8000
    for phase in range(4):assert e.batch_digest(batches[phase*2000:(phase+1)*2000])==e.batch_digest(base)
    assert sum(len(b)*len(b[0].program)*8 for b in batches)==8192000
    m,d,c=e.build_initial_model();assert d==manifest['initial_digest']
    assert len(list(m.parameters()))==len({id(p) for p in e.make_optimizer(m).param_groups[0]['params']})


def test_actual_runner_milestones_continuity_reload_and_prefix_failure(tmp_path,manifest):
    batch=e.base_stream()[667][:2];assert len(batch[0].program)==2
    batches=[batch]*4
    # Baseline saves at2/4 but does not evaluate; candidate evaluates actual legal
    # tiny L2 rows through the inherited model using the same runner path.
    base=r._TinySpec(batches,(2,4),99,{},noop)
    r._train(tmp_path/'baseline',manifest,_qa=base)
    prefix=torch.load(tmp_path/'baseline/u2.pt',weights_only=True)['state_dict']
    def tiny_evaluate(model,manifest):
        was=model.training;model.eval()
        try:
            from looped_bitnet.register_e15 import evaluate_program
            return evaluate_program(model,['ADD','ADD'],[(1,2),(3,4)],include_predictions=True)
        finally:model.train(was)
    record=r._train(tmp_path/'observed',manifest,_qa=r._TinySpec(batches,(2,4),2,prefix,tiny_evaluate))
    assert record['status']=='complete' and record['prefix_bitwise_equal']
    assert [p['update'] for p in record['progress']]==[2,4]
    for u in (2,4):
        a=torch.load(tmp_path/f'baseline/u{u}.pt',weights_only=True);b=torch.load(tmp_path/f'observed/u{u}.pt',weights_only=True)
        for key in ('model_digest','optimizer_digest','rng_digest'):assert a[key]==b[key]
    restored,opt,payload=e.load_checkpoint(tmp_path/'observed/u2.pt',manifest,expected_update=2,qa=True)
    # Execute the next updates using the same runner update helper used below.
    for _ in range(2):r._update(restored,opt,batch)
    final=torch.load(tmp_path/'observed/u4.pt',weights_only=True)
    assert e.digest_state_dict(restored)==final['model_digest']
    assert e.digest_object(opt.state_dict())==final['optimizer_digest']
    bad=deepcopy(prefix);bad['role_keys'][0,0]+=1
    with pytest.raises(ValueError,match='prefix bitwise mismatch'):
        r._train(tmp_path/'mismatch',manifest,_qa=r._TinySpec(batches,(2,4),2,bad,noop))
    assert r._read_json(tmp_path/'mismatch/progress.json')['completed_updates']==2
    assert not (tmp_path/'mismatch/u4.pt').exists()
    for key,value in [('native_steps',4),('full_stream_digest','bad'),('initial_digest','bad'),('optimizer_digest','bad'),('source_hashes',{}),('rng_digest','bad')]:
        bad=deepcopy(payload);bad[key]=value;torch.save(bad,tmp_path/'bad.pt')
        with pytest.raises(ValueError):e.load_checkpoint(tmp_path/'bad.pt',manifest,expected_update=2,qa=True)
    with pytest.raises(ValueError):e.load_checkpoint(tmp_path/'observed/u2.pt',manifest,expected_update=2)


def test_preflight_and_recovery_missing(tmp_path):
    r.preflight(tmp_path/'pre')
    with pytest.raises(FileExistsError):r.preflight(tmp_path/'pre')
    assert r.load_manifest(tmp_path/'pre')['native_steps']==8
    rec=r.recovery(tmp_path/'absent',tmp_path/'pre')
    assert rec['missing']==[2000,4000,8000] and not rec['final_present']
