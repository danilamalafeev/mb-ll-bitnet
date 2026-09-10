# E21 — first frozen-model novel-composition evaluation

Prospective registration,2026-09-07. Owner: root; design and independent review
Astra/low; one narrowly scoped implementation executor under current AGENTS
policy. No E21 model predictions or training have been performed by this design
step. Only this protocol is written by the designer.

## Authorized question and interpretation

Evaluate the accepted frozen E20 native8/u8000 model on the previously excluded
short compositions. E20 has primitive validation32/32 each and every seen
composition≥31/32. This numerically meets the single-model prerequisite, but
does not retroactively pass E15's paired QAT/GRU gate or its three-seed claim.
The user's separate E21 authorization permits the exact new-program/reserved
state evaluation below. E15/E20 historical protocols and results stay intact.

The tested system is the float core with signed-bit inputs,151232 parameters,
seed0, native8 substeps,8000 training updates. It was selected through an
adaptive research sequence using the reused validation states. This is its
first registered evaluation on these novel compositions, not a fresh
architecture-selection test or evidence across seeds. Do not compare it as if
it were the original E15 QAT/GRU pair.

## Frozen programs, states and numerical criterion

Reuse the exact E15v7 primary order:

1. ADD → XOR
2. ADD → ADD → XOR
3. ADD → XOR → ADD
4. ADD → XOR → SWAP
5. XOR → ADD → XOR
6. SWAP → ADD → XOR

Evaluate each on all256 initial (x,y) pairs. Canonical order is lexicographic
x=0..15 then y=0..15; freeze this exact list before model predictions. Derive
train192/validation32/reserved-test32 strata solely from the frozen E15v7
state_split. Verify pairwise disjointness, uniqueness and exact union of256;
never regenerate or resample the split. These strata indicate prior exposure
of initial states on OTHER seen programs; every E21 primary program was absent
from training. They are not three independent replications.

Register the inherited numeric criterion as a NEW single-model E21 predicate:
EACH of the six primary programs must have final joint correctness≥244/256
(95.3125%). No averaging can rescue a failing row. The E15 original criterion
required this per model in each of three seeds; E21 tests only its one-model,
one-seed numerical analogue. Record the six booleans and their conjunction.
No separate threshold is introduced for strata, prefix or full-trace metrics.
A failure stops just as a pass does; it does not authorize tuning/retraining.

ADD → XOR → XOR is the sole separate equivalent-function control. Its final
mapping equals ADD for all256 states, although its intermediate trace differs.
It is NOT a seventh novel transformation and NEVER enters the primary predicate
or primary aggregate. Evaluate it once on the same256 states. Do not separately
run ADD for this control: verify equivalence symbolically and score the control
against its own full instruction trace. No secondary length4/6 programs,
additional seen-program inference, other checkpoints, seeds or model variants.

## Symbolic preflight and immutability

Before ANY E21 model prediction, freeze a machine manifest containing this
protocol/config hash, reviewed evaluation source/imported dependency hashes,
exact E20u8000 checkpoint and E20 preflight/report hashes, model/initial/native
budget provenance, exact E15v7 manifest hash, ordered seven program lists,
ordered256 states and strata membership, full per-prefix DSL target traces,
and program/state/target/semantic-signature digests. Root froze the exact195-file
`results/E21_PROTECTED_HASHES.json` list covering E20 and earlier evidence.
The independent symbolic reference is `results/E21_SYMBOLIC_REFERENCE.json`. Reject changed
references, malformed manifests, unexpected program/state counts or nonempty
output roots; no historical source/runs/protocol writes.

Use cheap integer DSL enumeration now to verify ADD modulo16, XOR and SWAP
semantics, all targets, and final truth-table signatures on256 states. Require
six distinct primary signatures, each absent from the set of all32 seen-program
signatures; require the control signature equals ADD and is excluded from the
novel set. Verify the frozen forbidden set is exactly primary six plus control.
Fail closed on every mismatch; syntactic novelty alone is insufficient. This
symbolic audit is not a model evaluation and may precede code clearance.

Load only `runs/e20_longer_native8/u8000.pt`, validate its E20 schema/native8/
seed0/update8000, state keys/dtypes/shapes/count, model digest and frozen source/
config/protocol/data/reference metadata against accepted E20 provenance. Reuse
its verified loading logic when economical, but no optimizer.step, training or
resume path may be reachable from E21. No initialization or restored optimizer
state is treated as an additional experiment. Freeze and recheck the checkpoint
file hash and model-state digest before/after evaluation.

## One-pass measurement and reporting

Use CPU4 deterministic float32, autocast disabled, model.eval and
inference_mode. Eight recurrent substeps per instruction reuse the same four
FFNs twice, retain h/KV between instructions, and read out only at instruction
boundaries. No intermediate4-step head/loss, teacher forcing, oracle feedback,
weight changes, or output-dependent extra computation.

Execute one forward pass per unique program with one256-state batch: exactly
seven forward passes. Store each prediction and integer target trace once.
Compute all256 and192/32/32 stratum metrics from this stored result; do not
rerun by split, prefix, final register or equivalent control. Prefix sharing
between programs does not authorize a new cache intervention.

For each program and stratum/all256 report integer final joint, final x,
final y, each instruction-prefix joint, and full-trace correctness counts with
explicit denominators256/192/32/32. Preserve per-state predicted/target traces
and membership labels. Assert split totals equal all-state totals for every
metric and full-trace≤every prefix joint≤denominator. Print all six primary
rows, the six244/256 predicates and conjunction, plus the separate control row.
Any primary macro is descriptive only and excludes the control. Report the
control's prefix/full-trace behavior even if final behavior is correct.

Exact scientific cost:0 training updates. Primary6×256=1536 program-state cases,
(2+5×3)×256=4352 readout positions,34816 internal substeps. Control adds256
cases,768 readouts and6144 substeps. Total1792 cases,5120 readouts,40960 native
substeps. Stratum slicing and DSL enumeration add no model forwards. Tiny QA
and independent reviewer replay are separately reported verification work.

## Minimal implementation, clearance and stop

Use one new narrow evaluation module/runner and targeted tests, fresh E21
preflight/run artifacts. No framework, training loop, generic recovery system
or optimizer continuation is needed. Preflight creates targets/manifests only;
model predictions begin only after independent code clearance.

Required controls: symbolic novelty/control equivalence and frozen order; strict
checkpoint identity; exact native8 readout budget; one-forward-per-program
instrumentation; asymmetric toy predictions validating every count/stratum and
control exclusion from the primary conjunction;244 pass/243 fail boundaries;
actual tiny evaluation-to-report path using legal SEEN toy programs, with no
E21 primary/control model predictions before clearance. Fail if training is
entered, parameters change, state rows duplicate or counts/denominators drift.
Do not substitute a manual metric test for the actual runner/report path.

After one authorized evaluation, an independent reviewer reloads the frozen
model and reproduces all new-program traces/counts, symbolic novelty, numerical
predicate, provenance, scope/cost and protected hashes. Report the measured
result with the one-seed/adaptive-selection limitation and stop, whether the
predicate passes or fails. No automatic training, new sets, next checkpoint,
extra seed, secondary evaluation or reinterpretation of E15 is authorized.
