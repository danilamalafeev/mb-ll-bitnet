# E18 final pre-training code review — CLEAR

Independent reviewer: Astra/low, 2026-09-07. Clearance covers precisely the
registered fresh seed0 native4/native8 pair, followed by independent result
review and stop. No scientific training was performed by this reviewer.
Initial HOLD remains preserved in `E18_STEP_BUDGET_CODE_REVIEW.md`.

## Frozen reviewed inputs

- Module `looped_bitnet/step_budget_e18.py`:
  `15555c14e88af5a13d731eeb497473abded4b74eab276262621a07cc92f947dc`
- Runner `scripts/step_budget_e18.py`:
  `31d1143c5194bae6a79f982610bf2baba3fa088a56dcd781aea0f550e6304065`
- Tests `tests/test_step_budget_e18.py`:
  `3eb57919f10938214cbf0fda3b2cdb5456ec3fdd834f8c5dc6e3e7208a80632c`
- Protocol `results/E18_STEP_BUDGET_PROTOCOL.md`:
  `2b7244e2d7534bb3262b651256fb2deb93c54cfd7d6113f6cad2dbbf541801b5`

These hashes were independently recomputed after controls and match the stable
executor handoff. All138 protected file hashes independently passed.

## Findings and evidence

All four HOLD items are resolved. Checkpoint creation/loading now agree on
1024000/2048000 training substeps; main reporting records220416 inference
substeps and14336 program-state evaluations/36736 readouts. Recovery validates
saved initial states and returns rows, deltas, coverage, manifest/source
provenance and evaluation costs without requiring a report. Its CLI emits the
recovered JSON to stdout; checkpoint files remain untouched. Manifest format,
encoder mode and parameter count are checked explicitly.

Reviewer reran only the focused E18 suite: **8 passed in11.73s**. It covers
bitwise initial E17 identity and independent storage; actual asserted length2
native4 logits/loss/all-gradient/one-clipped-AdamW-update parity; native8
explicit two-instruction unroll with block order0,1,2,3 repeated four times,
16 opcode/reader calls, two boundary norms, exact logits/final h/counter and
all-gradient equality; fixed data/targets; both budgets' changed-weight
checkpoint round trips and wrong-budget rejection; asymmetric paired labels;
nonempty output rejection; and recovery without report/training plus rejection
of an incomplete pair. Recovery's unit test stubs inference, so it demonstrates
control flow rather than scientific count correctness; actual result review
must recompute scientific counts after the authorized run.

Additional independent tiny controls confirmed unchanged KV object identity
through two instructions, integer counter8→16, exactly two x-head and two
y-head calls, and unique model parameter objects. Manifest format/mode/count
tampering was rejected; corrupting a saved8-arm initial tensor caused recovery
to fail before loading/evaluating finals. Both checkpoint arms rejected altered
schema, initial digest, E15 manifest hash, source hashes, protocol hash, config
hash and update. Both recorded training cost values were asserted exactly.
These controls used temporary artifacts only. An initial auxiliary command
failed on an unimported PROJECT_ROOT name; the corrected command passed.

Static inspection confirms no extra trained parameters or intermediate4-step
head/loss in the8 arm; cache is initialized once per program and inherited
forward carries h/KV across instructions. Both modes retain E17 signed-bit
inputs and shared modules; the local subclass changes only native loop budget.
Training uses unchanged loss/AdamW/clipping, paired frozen2000 batches and
fixed final checkpoints. Evaluation is restricted to the frozen seen32
train192/validation32 scope. Reused pure helpers avoid old source writes.

## Limits and next gate

CLEAR is code/protocol readiness, not a scientific outcome. Canonical preflight
must freeze these sources before update1; changed reviewed files invalidate this
clearance. Run exactly one registered pair, preserve any failure, and obtain
independent result review including native4 E17 reproduction, all report counts,
paired outcomes, costs/provenance and protected hashes. No extra seed, reserved
test, new program, post-hoc budget change or automatic continuation is cleared.
