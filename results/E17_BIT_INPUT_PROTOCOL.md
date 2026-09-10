# E17 — learned value embeddings versus explicit four-bit input

Status: prospective registration, 2026-09-07; no E17 training yet.
Owner: root; design/review Astra/low, sole implementation executor Luna/xhigh.

## Question and fixed intervention

Does replacing the two independent learned value tables by signed four-bit
features with learned linear projections change train fit and validation
accuracy on the SAME E16 float architecture and fixed budget?

Run exactly one fresh paired seed0 pilot, then independent review and stop.
Both arms train from scratch. The completed E16 float checkpoint is NOT reused.
This deliberately retains the exact E16 learned-arm initialization and avoids
conditional baseline reuse or a changed, bit-structured learned-table baseline.

- `learned`: E16 float construction, including the unchanged independent
  `nn.Embedding(16,64)` x/y tables initialized N(0,0.02²).
- `bits`: deep-copy that same initialized float model, replacing ONLY x/y input
  embedding modules with separate four-bit encoders. For integer v in 0..15,
  b_i(v) = (v >> i) & 1, i=0,1,2,3 (least-significant bit first).
  c_i(v)=2*b_i(v)-1, represented exactly in float32. Thus 0 encodes
  [-1,-1,-1,-1], 1 [1,-1,-1,-1], and 15 [1,1,1,1].
  Each register returns W_r*c(v), using `Linear(4,64,bias=False)` with its
  own learned W_r. No extra scaling, constant feature, bias, normalization,
  nonlinear adapter, learnable lookup table or unused padding parameters.

Initialize projection weights with independent normal entries, std=0.01.
Use a separate local CPU torch.Generator seeded 0, drawing x weight (64,4)
then y weight (64,4); explicit initialization must not perturb the initialized
common model. Module-constructor RNG draws must not change the common state
or the fixed data stream. Signed code norm²=4, so each projected coordinate
has marginal initialization variance 4*0.01²=0.02², matching E16 tables.
Do not normalize realized samples or tune scale after inspecting outcomes.

All non-input state tensors must be `torch.equal` across arms and to the
seed0 E16 float construction before optimizer creation; storage is independent.
Common state includes opcode embeddings, role keys, norms, reader, four FFN
blocks and both output heads. Both arms have zero BitLinear modules, float32
operations and autocast disabled. Keep the 64-dimensional state, four substeps
per instruction, memory and readout semantics, output 16-class heads, opcode
encoding, loss and data unchanged. Integer inputs still enter init_cache only;
intermediate predictions are not re-encoded or fed back as bits.

Parameter counts: learned 152768; bits 151232. The input maps change from
2*16*64=2048 to 2*4*64=512 parameters, a reduction of 1536. Count actual
trainable parameters and reject deviations. No dummy parameters to match counts.

This intervention changes input geometry, parameterization and its training
constraint together. Initial functions are NOT matched: matching functions by
initializing the learned tables from bits would change the E16 baseline.
Matched marginal scale is not matched covariance; signed projections impose
correlated value vectors and lower-dimensional structure. A result identifies
a local effect of this representation/optimization package, not a unique
cause, pure bit information advantage, or a capacity-independent mechanism.

## Frozen data, budget and evaluation

Use E16 `load_frozen_manifest`, `fixed_stream(0)`, `target_digest`, existing
E15 `loss_for_batch`, and checked E16 evaluation/aggregation/coverage helpers.
Do not call old training, selection, preflight writers or downstream gates.
Frozen E15 manifest canonical hash:
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.
Full stream digest:
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`.
Verify the existing E16 target digest and length schedule as well.

Arm order learned then bits. Each arm: exactly 2000 updates, batch64, 128000
examples; same ordered batches/targets. CPU4 deterministic float32; independent
fresh AdamW, lr0.001, weight_decay0.01, E16 betas/eps, foreach=False;
zero_grad(set_to_none=True), unchanged sum of x/y prefix cross-entropies,
clip_grad_norm_1.0(error_if_nonfinite=True), optimizer.step. No optimizer state
sharing. Total 4000 updates, 256000 examples, 2048000 internal training substeps.
Progress every250 updates is operational only. Stop on nonfinite loss/gradient
or invariant failure. Fixed final u2000; no intermediate validation, checkpoint
selection, early stopping, sweep, extra seed, extension or automatic restart.

Evaluate final checkpoints in eval/inference mode on exactly all32 seen
programs x train192 and validation32 in frozen order. No reserved test32,
new/forbidden compositions, lengths4/6 or additional evaluation sets. Pair total:
14336 program/state evaluations, 36736 readout positions, 146944 substeps.
Reviewer replay and tiny code controls are separately counted verification work.
Reconstruct coverage from the batches and verify all6144 training pairs exposed.

Report all32 program rows for both arms/splits: integer final joint, x, y,
each prefix joint and full-trace correctness with denominators192 or32.
Retain per-state predictions and true paired outcomes: both, bits_only,
learned_only, neither. Their sums and marginal totals must agree exactly.
E16 paired_outcomes uses hard-coded float/qat fields internally: implement a
small explicit E17 relabeling adapter or E17-specific paired-count helper;
never publish mislabeled arms. Report bits-minus-learned percentage points,
length1/2/3 unweighted program macros, primitives=length1 and compositions=
equal mean of length2 and length3 macros, plus per-arm train-validation gaps.
Use unrounded counts; preserve ADD/XOR/SWAP rows and all poor composition rows.
No significance claims from dependent program/state cases; validation has
informed prior experiments and is not an untouched independent test.

## Minimal implementation and pre-training clearance

Only new E17 module/runner/tests/artifacts; import tested E16/E15 pure helpers.
No old source edits and no copied runner framework or generic resume system.
Root has frozen the exact 118-file protection list in
`results/E17_PROTECTED_HASHES.json`, covering the selected prior protections
and E16 source/protocol/results/runs. Verify this exact list before/after;
it is not an assertion that every historical workspace file was snapshotted.
Use distinct fresh roots `runs/e17_bit_input_preflight/` and
`runs/e17_bit_input/`, refusing nonempty roots and checkpoint overwrite.
Freeze reviewed E17 source, protocol/config and imported dependency hashes
before update1. The following focused controls require independent clearance:

1. Exact signed coding for ALL16 values, bit order, dtype, range rejection;
   each register projection equals explicit signed-bit linear computation.
   Confirm both encoders enter h and memory through the unchanged init_cache.
2. Learned arm versus original E16 float construction: exact initial state,
   logits, loss, gradients and one tiny identical optimizer update on a legal
   seen multi-prefix batch. Bits shared-core state is bitwise identical before
   training; parameter counts, disjoint storage and zero BitLinear inventory.
3. Finite nonzero gradients reach both projection matrices and shared core
   on a legal tiny batch; integer features have no trainable codebook. Check
   optimizer contains exactly each arm's actual parameters with no stale tables.
4. Exact paired stream/targets, allowed sets, update2000-only selection and
   protected/output guards. Save/load BOTH modes with exact eval logits.
   Reject wrong encoder mode, arm, schema, source/config/protocol hash, update,
   initial digest or manifest; strict state loading alone is insufficient.
5. Asymmetric paired toy predictions verify bits_correct-learned_correct equals
   bits_only-learned_only and all counts have correct semantic arm names.

## Provenance, failure recovery and stop

Save both complete initial states/digests and the common non-input digest.
Manifest/checkpoints record encoder specification (bit order/centering/scale,
projection init generator/seed/order/std), actual counts, E17 schema, arm/mode,
seed, fixed update, model/initial/common digests, data/target/full-prefix stream
digests, source/protocol/config hashes, optimizer and environment. Validate
against frozen expected metadata; build the correct mode before strict load.
Save each u2000 checkpoint atomically BEFORE inference/report writing.

A partial training failure preserves artifacts and stops; no retry, missing-arm
completion or continuation is authorized by this protocol. Evaluation/report
recovery may load the two valid final checkpoints and run final inference only,
without needing an existing report. It must never enter training or overwrite
checkpoints. If only one arm is complete, report the partial failure and stop.

Expected results: E17 manifest, initial states, two final checkpoints/progress,
machine and human reports, and independent review reproducing all counts,
paired outcomes, macros, exclusions, costs, provenance and protected hashes.
Distinguish exposed train fit from novel-state validation behavior. Neither a
gain nor a null result identifies the only bottleneck or validates general
recurrent reasoning. Stop after this paired pilot and independent result review,
including a concrete blocker; no subsequent experiment starts automatically.
