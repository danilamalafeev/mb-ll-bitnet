# E19 — balanced exposure on the native4 float bit-input baseline

Status: prospective design, 2026-09-07; no E19 implementation or training.
Owner root; design/review Astra/low, eventual sole executor Luna/xhigh.
This document authorizes no additional experiment after the bounded pilot.

## Question and fixed comparator

Does making exposure to seen program/state pairs maximally uniform within each
program length improve fit or existing-state validation under the SAME native4
float bit-input architecture, seed0 initialization, optimizer and2000 updates?

Train exactly ONE fresh `balanced` arm. The preregistered `random` comparator
is the existing frozen E17 bits checkpoint:
`runs/e17_bit_input/bits_seed0/final_u2000.pt`, with its original manifest and
report. E18 native4 independently reproduced all32 final tensors bitwise,
so repeating this deterministic baseline adds cost without an independent seed.
This is explicit artifact reuse, not a fresh randomized pair or seed replication.
Do not choose between E17/E18 baselines using outcomes. E17 is authoritative;
E18 native4 may serve only as an exact reproduction check.

Before update1, strict-load and hash the E17 checkpoint, manifest, protocol,
source provenance and initial-state artifact. Reconstruct its fixed seed0
initial model and original stream/target digest, and confirm parameter count,
optimizer, loss,2000 updates and native4 architecture agree with the new arm.
Assert complete new initial state `torch.equal` to E17 bits saved initial state.
If any prerequisite fails, HOLD before training; do not silently switch to a
new comparator or train another baseline under this protocol.

## Frozen architecture and optimization

Use unchanged E17 bits/E18 native4 behavior:151232 parameters, zero BitLinear,
64-dimensional float32 state, signed LSB-first four-bit x/y inputs with separate
bias-free4→64 learned projections; same opcode embedding, reader, memory,
four shared FFN blocks, four substeps/instruction, output16-class heads,
prefix x/y cross-entropy reduction and readout boundaries. No reset, detach,
extra loss, rescale, curriculum, new normalization or parameter change.

Fresh balanced model and AdamW from the exact E17 seed0 initialization;
CPU4 deterministic float32, autocast off, lr0.001, weight_decay0.01,
unchanged betas/eps/amsgrad and foreach=False. zero_grad(set_to_none=True),
existing loss_for_batch, clip_grad_norm_1.0(error_if_nonfinite=True), step.
Exactly2000 updates of64 examples,128000 examples,1024000 recurrent substeps.
Progress each250 updates; finite checks each update. Fixed u2000, no intermediate
validation, selection, early stopping, seed sweep, extra updates or automatic
restart. E17 baseline's historical compute is disclosed separately, not charged
as newly executed E19 training.

## Deterministic balanced stream

Use only the frozen V7 E15 seen32 programs and ordered train192 states.
Expected canonical manifest hash:
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.
Preserve EXACTLY the original ordered length schedule `(1,2,3)*666+(2,2)`;
length update counts1:666,2:668,3:666. Do not rebalance across lengths, replace
programs, change targets or use validation states in training.

For each length L independently:

1. Filter seen programs in their frozen manifest order to length L. Form the
   canonical list of all `(program, initial_state)` pairs in program-major,
   then frozen-train-state order. Sizes are576,1536,4032 for L=1,2,3.
2. Create a local Python `random.Random(19000+L)` generator. This is a fixed
   stream-design seed, not another model seed. Record Python version. It must
   never consume or reset model/global RNG. No observed outcome selects seed.
3. Repeatedly copy the COMPLETE canonical list and shuffle that copy in place
   with this same local generator, advancing its state across cycles. Append
   each permutation until at least64 times the length's update count is
   available. Truncate the concatenation to the exact required draw count.
   Never shuffle an already shuffled previous cycle or reseed between cycles.
4. Walk the unchanged global length schedule; at each update consume the next
   contiguous64 pairs from that length's stream and build RegisterExamples with
   the existing DSL target constructor. Do not reshuffle within each batch.

Required lengthwise frequency counts:

| Length | Draws | Pair universe | Counts over pairs |
|---|---:|---:|---|
| 1 | 42624 | 576 | all576 appear74 times |
| 2 | 42752 | 1536 |1280 appear28 times;256 appear27 times |
| 3 | 42624 | 4032 |2304 appear11 times;1728 appear10 times |

Thus every allowed pair appears and max−min≤1 within each length. The subset
getting the extra occurrence is the deterministic prefix of the next shuffled
cycle; no validation-dependent choice. A batch crossing a cycle boundary may
contain a repeated pair; do not add an undocumented deduplication rule. This
procedure changes frequencies, ordering and batch correlations together.
Its effect cannot be attributed uniquely to reduced count variance.

Register actual ordered-batch digest, target digest, per-pair count/frequency
digest, schedule digest and source/config/protocol hashes in a NEW preflight
before update1. They are to be derived from the rule above, not searched for
or selected from multiple candidates. Keep the original random stream digest
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`
and its frozen target/coverage evidence alongside the new stream. The two
streams intentionally need not have equal digests or ordered targets.

## Clearance, protection and failure rules

Use only new E19 source/tests/artifacts and reused pure model/data/evaluation
helpers; no old source, run or protocol edits. Root creates an exact protected
snapshot of the selected prior artifacts and baseline dependencies BEFORE
implementation, records its explicit list/hash, and verifies before/after.
Use fresh nonempty-refusing roots `runs/e19_balanced_exposure_preflight/` and
`runs/e19_balanced_exposure/`; never overwrite checkpoints or a prior attempt.
A concurrent separately registered E20 experiment must have disjoint output
roots and no shared mutable model/optimizer/RNG state or source edits.

Independent pre-training review must verify:

- A separately implemented count reconstruction checks all6144 allowed pairs,
  the exact frequency histogram above, global length schedule, batch sizes,
  stream/targets and exclusion of all validation/reserved pairs. Repeated
  construction has exact digests and leaves global RNG unchanged.
- Exact fresh initialization and legal length2 multi-prefix logits/loss/all
  gradients/one clipped AdamW update match unchanged E17 bits on the SAME tiny
  batch; actual encoder/core parameter membership and count remain unchanged.
- The ACTUAL train_arm path executes in a temporary tiny control, including
  progress/checkpoint/completion writing. Synthetic QA checkpoints are never
  scientific evidence. This control must exercise the runner path that earlier
  manual update tests failed to cover in E18.
- A trained tiny balanced checkpoint round-trips with exact eval logits;
  loader rejects wrong arm/native budget/encoder/schema/update, initial,
  stream/target, source/protocol/config or baseline provenance. Paired toy
  cases have unequal marginals and correct balanced_only/random_only labels.
- Recovery is inference/report-only with the valid new final checkpoint plus
  strict-validated frozen baseline; it never calls training or needs an
  existing report. Missing new final or invalid baseline fails closed.

Freeze reviewed source/protocol plus baseline byte hashes before update1.
Stop and preserve partial failure; no continuation/retraining/replacement arm
is authorized. Save new final u2000 atomically BEFORE inference/report writing.

## Fixed evaluation, reporting and interpretation

Evaluate balanced u2000 and the frozen random comparator on exactly all32 seen
programs x train192/validation32 in frozen order, native4 eval/inference mode.
No reserved test32, new/forbidden programs, lengths4/6 or post-hoc budgets.
Pair evaluation cost14336 program/state cases,36736 boundary readouts and146944
substeps; reviewer replay is separate verification compute. Retain all128 rows,
per-case predicted/target traces, integer final x/y/joint, prefix and full-trace
counts with denominators192/32. Report all32 programs, both splits/arms; count
both correct, balanced_only, random_only, neither and verify margins exactly.

Primary descriptive outcome: balanced-minus-random exposed train composition
final-joint macro, equally averaging length2 and length3 program macros.
Report validation composition macro and ADD/XOR/SWAP separately, plus all
length1/2/3 macros, prefix/full-trace outcomes and each arm's train-validation
gaps. No threshold, significance claim, checkpoint selection or expansion rule.
All comparisons derive from unrounded counts. A favorable mean must not hide
poor primitive or composition rows. Original primitive gate remains unchanged.

A gain supports a local effect of this exposure/ordering package at one budget;
a null/adverse result does not establish that sampling balance is irrelevant.
Existing validation has informed research and is not untouched test evidence.
The reused comparator is not an additional seed. Do not infer a unique
optimization mechanism or general recurrent reasoning. Produce one machine
report, human report and independent recount/provenance review, then stop.
