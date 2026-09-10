import json,tempfile
from pathlib import Path
import torch
from looped_bitnet import bit_input_e17 as e
from looped_bitnet.float_qat_e16 import build_paired_models as oldbuild
from looped_bitnet.register_e15 import loss_for_batch,OP_TO_ID
from scripts import bit_input_e17 as r
l,b,ld,bd,cd,_=e.build_paired_models(); old=oldbuild()[2]
vs=torch.arange(16,dtype=torch.long)
ref=torch.tensor([[2*((v>>i)&1)-1 for i in range(4)] for v in range(16)],dtype=torch.float32)
for enc in (b.x_embedding,b.y_embedding):
 assert torch.equal(enc(vs),ref@enc.projection.weight.t())
batch=e.fixed_stream()[1][:4]; assert all(len(z.program)==2 for z in batch)
inputs=(torch.tensor([z.x for z in batch]),torch.tensor([z.y for z in batch]),torch.tensor([[OP_TO_ID[o] for o in z.program] for z in batch]))
for x,y in zip(old.state_dict().values(),l.state_dict().values()):assert torch.equal(x,y)
for model in (old,l,b):
 model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False)
 opt.zero_grad(set_to_none=True); loss=loss_for_batch(model,batch); loss.backward()
 if model is old: oldloss=loss.detach().clone(); oldgrads=[p.grad.clone() for p in model.parameters()]
 if model is l:
  assert torch.equal(loss,oldloss)
  for p,g in zip(model.parameters(),oldgrads): assert torch.equal(p.grad,g)
 if model is b:
  for enc in (b.x_embedding,b.y_embedding):assert torch.isfinite(enc.projection.weight.grad).all() and enc.projection.weight.grad.abs().sum()>0
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
for x,y in zip(old.parameters(),l.parameters()):assert torch.equal(x,y)
with tempfile.TemporaryDirectory(prefix='e17review-') as td:
 td=Path(td)
 for arm,model,digest in [('learned',l,ld),('bits',b,bd)]:
  kw=dict(arm=arm,initial_digest=digest,common_digest=cd,stream_digest=e.EXPECTED_BATCH_DIGEST,target_stream_digest=e.EXPECTED_TARGET_DIGEST,e15_manifest_hash=e.EXPECTED_MANIFEST_HASH,source_map=e.source_hashes())
  e.atomic_torch_save(td/(arm+'.pt'),e.checkpoint_payload(model,**kw)); restored,_=e.load_checkpoint(td/(arm+'.pt'),**kw)
  model.eval();restored.eval()
  with torch.inference_mode():
   for x,y in zip(model(*inputs),restored(*inputs)):assert torch.equal(x,y)
 pre=td/'pre';r.write_preflight(preflight=pre); mf=r._load_manifest(pre,r.PROJECT_ROOT)
 r._initial_states(pre,mf,r.PROJECT_ROOT)
 for field,value in [('schema','bad'),('config_hash','bad'),('dependency_hashes',{}),('state_split',{}),('seed',1),('update',3),('protocol_hash','bad'),('schedule',{})]:
  bad=dict(mf);bad[field]=value;(pre/'manifest.json').write_text(json.dumps(bad))
  try:r._load_manifest(pre,r.PROJECT_ROOT)
  except ValueError:pass
  else:raise AssertionError(field)
 (pre/'manifest.json').write_text(json.dumps(mf))
 payload=torch.load(pre/'learned_initial_state.pt',weights_only=True); fresh=e.build_paired_models()[0]
 with torch.no_grad():fresh.x_embedding.weight.add_(.125)
 altered_digest=e.digest_state_dict(fresh);payload['state_dict']=fresh.state_dict();payload['digest']=altered_digest;torch.save(payload,pre/'learned_initial_state.pt')
 bad=dict(mf);bad['learned_initial_digest']=altered_digest
 try:r._initial_states(pre,bad,r.PROJECT_ROOT)
 except ValueError as exc:assert 'reconstructed seed0' in str(exc)
 else:raise AssertionError('substituted initial accepted')
print(json.dumps({'status':'pass','all16_projection_checks':2,'length2_batch':4,'learned_full_update_parity':True,'trained_eval_roundtrips':2,'negative_manifest_checks':8,'self_consistent_substituted_initial_rejected':True,'protected':len(e.verify_protected_hashes()),'source_hashes':e.source_hashes()},indent=2))
