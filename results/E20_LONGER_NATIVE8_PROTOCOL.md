# E20 — fixed longer training of the native8 model

Prospective registration,2026-09-07; design Astra/low, root coordination and
shared-document ownership. Implementation: one bounded executor under current
AGENTS policy; independent code/result reviewer Astra/low. This protocol owns
only E20. E19 remains a separate proposed balanced-exposure experiment; its
stream or results must not change E20 settings.

## Question and fixed decision

Does the poor E18 native8 fit at2000 updates improve with more optimization on
the same finite stream, and does native8 at8000 updates reach the already
observed native4/u2000 performance? E18 native8 length3 train/validation joint
macros were41.344%/29.315%, versus native4's92.113%/80.655%. These are reference
observations, not proof of undertraining or an optimization mechanism.

Choose ONE native8 seed0 run to fixed8000 updates, with mandatory observations
at2000,4000,8000. Doubling to4000 alone would provide a weaker negative test;
8000 gives a fourfold update-budget probe while4000 documents the trajectory.
There is no4000-dependent decision to continue or stop:8000 is registered now.
No additional long native4 arm is necessary for this conditional catch-up
question. Consequently comparisons to native4/u2000 are unequal-budget
comparisons and cannot establish a fair architecture ranking. A matched long
native4 comparison would require separate authorization and registration.

Primary descriptive endpoint: length3 train final-joint macro at8000 versus
its reproduced2000 value. Report the exact difference and remaining gap to
native4/u2000. Co-report length3 validation and length2/composition macros,
all primitives and every row. At8000 record separately whether train and
validation length3 macros reach their exact unrounded native4/u2000 references;
this is a descriptive catch-up predicate, not a new gate or significance test.
Positive train improvement supports a local sensitivity to more optimization;
validation improvement additionally indicates better behavior on these reused
validation states. Failure to catch up by8000 only limits this fixed schedule;
it does not disprove every undertraining explanation. No threshold, plateau
judgment or metric permits another extension.

## Fresh reconstruction, not checkpoint resume

Existing E18 final checkpoints contain optimizer configuration but no optimizer
state. Loading their weights into fresh AdamW would change the training path.
Therefore reconstruct a FRESH native8 model and fresh optimizer from seed0;
do not describe this as resuming the E18 checkpoint.

Keep all151232 parameters, bit encoders, initial tensors, eight-substep core,
four reused FFNs, opcode/cache/readout semantics, CPU4 deterministic float32,
autocast disabled, unchanged x/y CE after each instruction, AdamW configuration,
clipping and batch64 exactly as accepted repaired E18. No inner4 readout/loss,
new layer, rescaling, LR schedule, weight decay change or reset of optimizer
moments at any milestone. Evaluate with model.eval/inference_mode, then restore
training mode without changing model/optimizer/RNG state.

For updates1..8000, batch[u-1] = E18 fixed_stream(0)[(u-1)%2000]. Thus the first
2000 batches/targets match E18 exactly; later phases repeat the same entire
ordered2000-batch stream three times. This preserves its existing exposure
imbalance and adds repeated exposure, not new random training examples. It is
neither E19 balancing nor a curriculum. Validate the frozen E15 manifest,
base stream and target hashes inherited from E18, plus a new full8000-update
stream/target digest and phase boundaries. The length schedule is repeated
four times: length1=2664,length2=2672,length3=2664 updates. All6144 legal train
program-state pairs remain exposed; no new pairs/programs/splits enter.

After update2000, BEFORE update2001, save the milestone atomically and compare
EVERY model state tensor via torch.equal to
`runs/e18_step_budget_repaired/steps8_seed0/final_u2000.pt` after validating its
provenance/hash. Exact keys/dtypes/shapes/digest and bitwise values are required.
Any mismatch stops E20 and preserves artifacts: do not continue, relax equality
or reinitialize moments. This prefix is reproduction, not an independent seed.

## Fixed observations and accounting

At2000,4000,8000 evaluate exactly seen32 × train192/validation32, native8 only,
with the same frozen state/program order. Save model and optimizer state before
inference at each milestone. No intermediate validation, selection, best model,
early stopping by results, reserved test32, new compositions, lengths4/6,
post-hoc native4 inference, seed sweep or extension. Nonfinite/invariant failure
stops and preserves the run. Register8000 as the fixed final regardless of which
observation has the highest accuracy.

Training cost, INCLUDING reproduction of the first2000 updates:
8000 updates,512000 batch examples,8192000 internal state substeps. Prefix2000
costs2048000 substeps; work beyond the E18-matching prefix is6000 updates,
384000 examples,6144000 substeps. Repeated exposure is counted as work even
though the base batches repeat. Milestone evaluation total:21504 program-state
evaluations,55104 readout positions,440832 substeps; per milestone7168/18368/
146944. Reusing already stored native4/u2000 predictions for reference adds
no new scientific inference. Tiny controls, failure/retry work and independent
reviewer replay must be separately disclosed, not hidden in the nominal budget.

Retain every prediction/target trace and all32 program rows for both splits at
each milestone: final joint/x/y, each instruction-prefix joint, full-trace
counts, denominators192/32. Compute unrounded length1/2/3 equal-program macros,
primitives and equally weighted length2/length3 compositions, train-validation
gaps, milestone-minus2000 deltas and gaps to E18 native4/u2000. Paired same-state
outcomes may compare milestones/reference with accurate labels and checked
margins; do not treat dependent cases as independent replication. Preserve
poor rows and primitive errors. Existing prerequisite criteria stay unchanged.

## Minimal implementation and mandatory clearance

Only NEW E20 module/runner/tests/artifacts and fresh roots
`runs/e20_longer_native8_preflight/`, `runs/e20_longer_native8/`. Do not edit E18
or earlier sources/runs/protocols. Import accepted E18/E17 pure helpers and use
a narrow E20 milestone loop; avoid a copied runner framework or generic resume
system. Root froze the exact176-file list in `results/E20_PROTECTED_HASHES.json`,
including E18 artifacts and source snapshots. Validate that list before/after; freeze reviewed E20 sources,
imported dependencies, this protocol/config and reference checkpoint/report
hashes before update1. Reject nonempty output roots and checkpoint overwrite.

Independent clearance requires:

1. Exact E18 native8 initialization/parameter inventory and repeated-stream
   identity/phase order, targets, length counts, exposure and cost calculations.
   Continuous AdamW moments and global update numbering cross2000/4000 without
   reset; actual optimizer parameter membership is exact.
2. An ACTUAL E20 runner/training-path tiny smoke, not only manually coded
   optimizer steps. It must cross at least one configurable QA stream boundary,
   save multiple tiny milestones, take the final-report path, restore train
   mode after inference and exercise operational progress logging. QA budgets
   are isolated test injection, never scientific CLI overrides. The E18 missing
   PROGRESS_INTERVAL failure motivates this mandatory executable-path check.
3. A tiny uninterrupted versus milestone-evaluated path comparison gives exact
   model/optimizer states, confirming inference/reporting does not alter the
   optimization trajectory. Prefix-reference mismatch fails BEFORE any next
   optimizer update; test this rejection with an intentionally wrong reference.
4. Model AND AdamW state round trips after actual tiny updates reproduce the
   next update exactly, including moments/step counters. Reject wrong native
   budget, update/milestone, encoder/schema, initial/stream/reference hashes,
   source/protocol/config/optimizer metadata and model/optimizer digest tampering.
5. Evaluation/report recovery reads existing valid milestones without training
   or checkpoint overwrite; missing/incomplete milestones remain explicit.
   Recovery must not manufacture optimizer steps or report an absent8000 final.

Checkpoints include complete model/optimizer states and digests, RNG state,
seed/global update, native budget, actual counts, source/protocol/config hashes,
initial and E18 reference digests, base/full stream/target digests, environment
and cumulative training work. Save enough optimizer provenance for a separately
authorized future continuation; storing it does not authorize continuation now.

## Stop

After this one fixed8000 run and independent result review, stop. No metric
permits additional updates, seeds or a changed stream. Preserve failures and
report the exact consumed work; no implicit scientific retry. Any technical
repair/restart requires a documented prospective exception with frozen failed
artifacts, following the E18 lesson. E19 may proceed independently under its own
registered conditions, but neither experiment may adapt to the other's results.
