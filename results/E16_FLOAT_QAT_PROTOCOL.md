# E16 — same-architecture float versus QAT pilot

Status: registered design, 2026-09-07; no E16 training or measurements yet.
Owner: root. Design/review: Astra/low; implementation: Luna/xhigh.

## Question and boundary

Does disabling the existing weight **and** activation fake quantization improve
fit to already exposed register programs under the same architecture,
initial master weights, data, objective and fixed training budget?

E15 confounded quantization with QAT-versus-GRU architecture. Its frozen QAT
checkpoint has length3 train joint macro 32.14%, despite near-perfect primitive
train counts, and poor ADD/XOR validation. E16 isolates the joint intervention
of replacing all BitLinear operations with float Linear operations during both
training and inference. It does not isolate weight quantization from activation
quantization, training from inference effects, or a mechanism of any difference.
Equal parameter counts do not imply equal effective representational capacity.
This diagnostic does not establish the broader recurrent-reasoning hypothesis.

Run exactly **one paired seed0 pilot** and stop after independent result review,
regardless of sign or size of the result. No automatic replication or success
threshold. Any additional seed requires a separate prospective registration;
this avoids expanding compute based on a favorable single-seed finding.

## Fixed arms

- `qat`: unchanged `looped_bitnet.register_e15.QATRegisterModel`.
- `float`: deep-copy that same initialized model and recursively replace **all**
  instances of `BitLinear` with ordinary `torch.nn.Linear`, preserving each
  module's weight, bias presence/value, dimensions, dtype and device. Preserve
  all other parameters, buffers, module positions and model methods.

Initialize the QAT base exactly with the frozen E15 seed0 initialization. Clone
before constructing either optimizer. Assert `torch.equal` for every named
initial state tensor, identical state keys and shapes, independent storage,
identical parameter ordering, and exactly **152768 parameters in each arm**.
Save the common initial state plus its digest. Enumerate replaced qualified
module names in the manifest; float must contain zero BitLinear modules and
QAT must retain the complete original inventory: 14 modules (4 reader, 8 FFN,
2 output heads). Include reader projections,
FFN projections and both output heads. Do not toggle only a config flag or
replace only the recurrent core. Float operations use float32 with autocast off.

Keep learned x/y/opcode embeddings, role keys, norms, memory/readout semantics,
64-dimensional state, four FFN blocks and four recurrent substeps per
instruction unchanged. No GRU arm, explicit 4-bit input, step change,
curriculum, auxiliary objective, quantization schedule or hyperparameter sweep.

## Frozen data and training

Use the existing V7 E15 manifest as a read-only source for seen32 programs and
train192 / validation32 state order. Expected canonical-JSON manifest hash (not file-byte SHA256):
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.
Use the exact existing `make_paired_batches(0)` and `loss_for_batch` pure helpers.
Both arms consume identical batches and prefix targets in identical order:
2000 updates, batch64, 128000 examples per arm; full stream digest
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`.
Check the recomputed digest, schedule and target serialization before training.

Use CPU, four threads, deterministic algorithms, float32, and frozen E15 seed
setup. Fresh independent AdamW optimizers: lr0.001, weight_decay0.01,
foreach=False, default E15 betas/eps; zero_grad(set_to_none=True), unchanged
sum of x/y prefix cross-entropies, clip_grad_norm_1.0 with
error_if_nonfinite=True, then optimizer.step. Record actual optimizer config.
Arm order is QAT then float. No shared optimizer state or changing data from
loss observations. Total budget: 4000 updates / 256000 examples /
2048000 internal state updates across both arms.

Checkpoint is **fixed update2000**, never validation-selected. No intermediate
validation, early stopping, best/latest selection or post-hoc checkpoint choice.
Persist progress and finite training loss at a modest fixed interval (250
updates is sufficient); this is operational telemetry, not selection evidence.
Stop on nonfinite loss/gradient or an invariant failure; preserve incomplete
artifacts and report the failure without silent restart or altered settings.

## Minimal implementation and clearance

Create only new E16 wrapper/runner, targeted tests and E16 artifacts. Import
existing pure model/data/loss/evaluation helpers without calling E15 training,
preflight writers, selection, gates or downstream evaluation. Avoid copying
an experiment framework or adding generic resumption infrastructure.

Before training, independent review must clear these focused controls:

1. Original QAT versus new QAT construction from the same seed: bitwise initial
   state, forward logits, loss, gradients and weights after one identical tiny
   optimizer update. Use a legal seen-program batch with more than one prefix.
2. QAT versus float: exact initial master state, parameter count/order and
   distinct storage; complete BitLinear replacement inventory. Float forward
   equals its ordinary-linear computation, and QAT fake quantization remains
   active. Equal outputs between float and QAT are not required.
3. Stream/target digest pairing, fixed final-checkpoint use, exact allowed
   program/state sets and refusal of protected/output overwrite.
4. Save/load round trip for **each arm**, retaining its module types and exact
   logits. A deliberately wrong arm, source hash, update or manifest must fail.
5. A small paired-outcome example with asymmetric errors checks the identity
   `float_correct - qat_correct = float_only - qat_only`.

Tiny controls do not authorize full extra pilot training. Freeze reviewed new
source and protocol hashes in the run manifest before update1.

## Checkpoints, provenance and recovery

Use a new nonempty-refusing root `runs/e16_float_qat/`; never overwrite a prior
attempt. Use the frozen 101-file snapshot `results/E16_PROTECTED_HASHES.json`,
created by root before E16 work and including the earlier E15 protections.
No old source edits are allowed. Verify all snapshot paths and hashes before
and after the run; do not replace or regenerate the snapshot to accept changes.

New checkpoint format must identify E16 schema, arm, seed0, fixed update2000,
model state digest, common initialization digest, full/prefix stream digest,
frozen E15 manifest hash, new protocol/config/source hashes, optimizer config
and environment. New loader validates metadata and digest, builds the correct
arm with a strict state load and refuses mismatched provenance. Do not reuse an
E15 loader by falsely tagging float as QAT or an E16 checkpoint as E15 selected.

Save each final checkpoint atomically **before** final inference/report writing.
Recovery may only load and evaluate already complete valid u2000 checkpoints;
it must never train, finish a missing arm, select a checkpoint or overwrite old
checkpoints. If training stops mid-arm, preserve the partial attempt and stop;
training restart needs explicit subsequent authorization. Report-writing or
inference failure does not justify retraining. A report is complete only when
both final checkpoints exist and all required rows verify.

## Fixed final evaluation

In eval/inference mode, evaluate each final arm on exactly seen32 programs
crossed with train192 and validation32 in frozen order. No reserved test32,
forbidden primary compositions, secondary lengths4/6 or other cases, even if
primitive scores look good. This is 14336 full program/state evaluations,
36736 readout positions and 146944 recurrent substeps; reviewer replay is
additional verification compute. Validate the exact set before model inference.

Reconstruct stream exposure counts from the fixed batches (no inference):
retain per-program/state frequencies or a digest plus per-program coverage,
min/max counts and draws. Verify 6144/6144 combinations were exposed before
calling the train set stream-exposed.

For every arm/program/split retain integer denominators and final joint, final
x, final y, each prefix joint, and full-trace correctness. Retain final per-case
joint correctness in frozen state order for paired summaries. Use true paired
counts: both correct, float only, QAT only, neither; these sum to192 or32 and
reproduce both marginal totals. Differences of aggregate accuracies alone are
not paired win/loss counts. Do not use significance tests treating dependent
states/programs as independent experimental replications.

Display **all32 rows** with train and validation counts for both arms. Give
length1/2/3 macros (unweighted program means), primitives=length1, and seen
compositions=equal mean of length2 and length3 macros. Report float-minus-QAT
percentage-point differences, and train-minus-validation gaps separately for
each arm. Rates derive from unrounded counts. Prefix summaries use only
programs having that prefix. Do not suppress primitive ADD/XOR/SWAP rows or
poor composition rows behind a macro.

## Interpretation and final stop

Train seen-composition fit and primitive validation generalization are separate
outcomes. A float gain on train compositions supports a local effect of the
joint fake-quantization intervention under this budget; it does not demonstrate
transfer to unseen initial states. Primitive train/validation gaps must be
reported individually. A null or adverse difference does not prove
quantization harmless: this is one seed, one budget and a coupled intervention.
Do not call validation an independent test; these states informed prior E15
research, although E16 does not use them for checkpoint selection.

Expected artifacts: frozen new manifest/protected snapshot, common initial
state, two final checkpoints and progress records, one machine report,
`results/E16_FLOAT_QAT.md`, and an independent result review. Reviewer reloads
both checkpoints, recomputes all counts and paired summaries, checks actual
budget/exclusions/provenance and protected hashes. Stop after ACCEPT or a
concrete blocker. The next possible 4-bit-input and step-budget experiments
remain unregistered and unimplemented.
