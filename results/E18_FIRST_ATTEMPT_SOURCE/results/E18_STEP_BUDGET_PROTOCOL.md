# E18 — native four versus eight substeps on the float bit-input core

Status: prospective registration, 2026-09-07; no E18 training evidence.
Owner: root; design and independent review Astra/low, implementation Luna/xhigh.

## Question and intervention

Does training and evaluating with eight rather than four recurrent substeps per
instruction improve exposed train fit and seen-program validation at the same
2000 optimizer updates? Run one fresh paired seed0 pilot, independently review
it, then stop. This tests a native training/inference compute package: it does
not isolate inference-only compute or hold internal training work constant.

Arms are `steps4` and `steps8`, trained from scratch in that order. Both complete
initial state dicts must be bitwise equal to the seed0 E17 bits construction
and its saved initial reference, with independent parameter storage. Do not
reuse a trained E17 checkpoint. Each arm has exactly151232 trainable parameters,
zero BitLinear modules, float32 operations, autocast disabled, the same signed
LSB-first four-bit x/y inputs and separate bias-free4→64 projections, and the
same64-dimensional recurrent state, reader, four FFN blocks, norms and heads.
No extra block, head, router, rescale, normalization or learnable parameter.

For budget B∈{4,8}, each instruction executes the existing core recurrence B
times: inject the same instruction opcode with the existing1/sqrt(64) factor,
apply the existing reader residual, then block[(cache.substeps+inner)%4]. The
8 arm makes two passes through the SAME four blocks. Preserve arithmetic order.
Carry h and the existing KV cache across instruction boundaries without reset,
detach, prediction feedback or input re-encoding. Update the integer substeps
counter by B per instruction; init_cache starts at zero. Read out through the
existing output norm and x/y heads only at the native instruction boundary.
In particular there is no head call, CE target or auxiliary loss after substep4
inside an8-substep instruction. The unchanged loss supervises x/y after EACH
instruction, using the existing E15 loss reduction across prefix positions.

## Frozen data and budget

Reuse E17/E16 pure data, target, evaluation, coverage and aggregation helpers;
never call an old training/preflight writer or downstream gate. Frozen E15
manifest canonical hash:
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.
Full paired stream digest:
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`.
Validate the existing target digest and full fixed_stream(0), including length
counts1:666,2:668,3:666. Both arms consume identical ordered batches and targets.

Each arm:2000 updates, batch64,128000 examples, CPU4 deterministic float32;
fresh independent AdamW using E17 optimizer configuration (lr0.001,
weight_decay0.01, unchanged betas/eps, foreach=False),
zero_grad(set_to_none=True), existing x/y CE, clip_grad_norm_1.0 with
error_if_nonfinite=True, optimizer.step. Progress each250 updates is operational.
Stop on invariant failure or nonfinite loss/gradient. No intermediate validation,
selection, early stopping, extra seed, sweep, extension or automatic restart.
Use the fixed final u2000 only.

Internal training work: steps4=1024000 and steps8=2048000 substeps;
pair=3072000 substeps,4000 updates,256000 examples. Extra compute is disclosed,
not compensated by fewer8-arm updates or altered batch size.

Evaluate each final model ONLY at its native trained budget, on all32 seen
programs × train192 and validation32 in frozen order, eval/inference mode.
No reserved test32, new/forbidden programs, lengths4/6, extra evaluation sets,
or post-hoc8-step evaluation of the4-step-trained model. Pair totals:
14336 program-state evaluations,36736 readout positions,220416 internal substeps
(73472 for4;146944 for8). Tiny controls and reviewer replay are separately
accounted verification work. Reconstruct exposure and require all6144 train
program-state pairs to have appeared in the fixed stream.

## Required pre-training controls and implementation boundary

Use only new E18 module/runner/tests/artifacts. Reuse existing pure helpers and
a narrow local step wrapper/subclass shared by both budgets; initialize KV once
per program through unchanged init_cache. Do not implement8 by calling core.step
twice, since that computes an unwanted intermediate readout. Do not edit old
sources or copy a runner
framework. New roots `runs/e18_step_budget_preflight/` and
`runs/e18_step_budget/` must refuse nonempty roots and checkpoint overwrites.
Root froze the exact138-file historical protection list in
`results/E18_PROTECTED_HASHES.json`; validate its complete frozen list before
and after work. Freeze reviewed E18 source, imported dependency, protocol and
config hashes before update1. Independent code review must clear:

1. Both complete initial states equal each other and the E17 bits reference via
   torch.equal; correct counts, storage independence, encoder inventory and
   optimizer membership with no duplicates/stale parameters.
2. Native4 versus unchanged E17 bits: exact logits, loss, EVERY parameter
   gradient, and resulting state after one actual identical clipped AdamW
   update on a legal length2 multi-prefix batch. Assert length2 explicitly;
   fixed_stream[666] is length1 and cannot satisfy this control.
3. Native8 versus an explicit eight-substep unroll on a multi-instruction batch:
   exact outputs/cache and gradients; instrument FFN calls/order, opcode
   injection every substep, unchanged KV, carried h, integer counter progression,
   and exactly one boundary readout per instruction. Verify finite nonzero
   gradients reach both bit projections and the shared core. No inner4 CE.
4. Fixed paired stream/targets, exposure, allowed evaluation sets and costs;
   update2000-only selection and output/protection guards. Asymmetric toy
   outcomes check the semantic steps8_only/steps4_only pairing and delta signs.
5. Save/load BOTH budgets after an actual tiny training update with exact eval
   logits. Reject wrong native budget even though tensor shapes match; reject
   wrong arm/schema/encoder/update, initial/manifest/data/source/protocol/config
   metadata or tampered model state. Verify frozen manifest and saved initial
   states against reconstructed expectations before training and recovery.
6. Recovery from two valid final checkpoints without an existing report runs
   inference/reporting only, never training or checkpoint overwrite. One final
   checkpoint alone is an incomplete pair and must stop.

## Reporting, provenance and stop

Save complete initial states/digests, shared-core digest, manifest and both
atomic final_u2000 checkpoints BEFORE inference/report generation. Record E18
schema, arm, native substep budget, encoder specification, counts, seed/update,
model/initial/common digests, frozen manifest/stream/target hashes, source and
protocol/config hashes, optimizer, environment and actual compute accounting.
Loaders construct the correct budget and validate metadata against frozen
expectations before strict state loading; shape compatibility is insufficient.

Report all32 rows for both splits/arms: final joint/x/y, each instruction-prefix
joint and full-trace integer counts with denominators192/32. Retain per-state
predictions/targets and paired both_correct, steps8_only, steps4_only, neither;
verify sums and margins. Report steps8-minus-steps4 percentage-point deltas,
length1/2/3 equal-program macros, primitives=length1, compositions=equal mean of
length2 and length3 macros, and each arm's train-minus-validation gaps. Compute
from unrounded rates. Preserve ADD/XOR/SWAP and poor composition rows. Compare
the freshly trained4 arm to E17 bits for reproduction as a verification check,
not an independent replication or substitute for the fresh arm.

No success threshold or extension rule is introduced. Neither improvement nor
null result establishes a unique bottleneck, general arithmetic algorithm or
robustness across seeds; validation has informed previous experiments and is
not untouched test evidence. Distinguish exposed fit from novel-state validation.
The original primitive prerequisite is unchanged and does not open reserved
sets automatically. Preserve failures; no retry, continuation or missing-arm
completion is authorized. Independent result review must reproduce counts,
paired metrics, provenance, costs and protected hashes. Stop after this pair
and review, including a concrete blocker; no next experiment starts implicitly.
