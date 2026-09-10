# E22 — fixed initialization-seed replication

Prospective registration, 2026-09-07. User authorizes exactly two fresh runs,
seeds 1 and 2, under the accepted E20 settings. Owner/shared-document writer:
root. Design and independent review: Astra/low; one bounded implementation
executor under current AGENTS policy. This design step writes only this file
and performs no training or new model evaluation.

## Question and scope

Does the accepted float-core, signed-bit-input native8 system at 8000 updates
retain its E21 six-program criterion under two new complete initializations,
conditional on the exact same training stream and evaluation sets?

E20 seed0 reached seen validation full-trace 1023/1024; E21 seed0 six primary
counts were 256,253,254,255,253,251 of256. These are prior observations, not
E22 results. The architecture was adaptively selected and E21 is already open.
This is conditional initialization robustness on reused finite data, not a
fresh holdout, full data-seed replication, original E15 paired QAT/GRU result,
or a general reasoning/length-generalization claim. E19 remains deferred.

## Frozen seed routing and initialization

Run seed1 and seed2 sequentially on CPU. Each begins from fresh parameters and
fresh AdamW, never from a trained checkpoint. Initialization seed s controls
both the master-model initialization and the independent local CPU generator
for the two signed-bit projection matrices. Draw x then y, each shape (64,4),
float32, torch.randn times0.01, from that local generator seeded s. Thus all
randomly initialized retained parameters vary under the same initialization
law; fixing projections to0 would test only a subset and is not this design.
Projection draws must not consume the master global RNG. Preserve the E17
SignedBitsEncoder constructor's RNG isolation.

E16/E17/E18 factories intentionally reject nonzero seeds; do not modify or
bypass their guards or replace historical constants. A narrow NEW E22 factory
may compose accepted pure constructors/conversion helpers: seed_everything(s),
QATRegisterModel master construction, complete float conversion, replacement
of x/y embeddings by the above signed-bit encoders, and native8 wrapping.
Retain exactly151232 parameters, zero BitLinear modules, all state keys and
shapes, four reused FFNs, cache/readout and instruction semantics.

Before training, construct s=0 with this new factory and require torch.equal
for EVERY tensor against accepted E20/E18 initial state, with matching keys,
dtypes/shapes/digest. This is initialization-only verification, no seed0
retraining. Freeze seed1/2 initial states and digests before update1; verify
repeatability and distinctness from each other and seed0. Record any harmless
constructor RNG consumption explicitly; freeze the post-initialization RNG
state. No stochastic training feature may silently make differing incidental
constructor draws an additional uncontrolled variable.

Data seed is ALWAYS0. Reuse exact accepted E20 base2000 ordered batches and
integer targets four times: batch[u-1]=base[(u-1)%2000]. Both new seeds have
identical full8000 stream/target hashes, equal to E20, and identical frozen
E15v7 program/state split. Do not pass s to fixed_stream. Length1/2/3 update
counts remain2664/2672/2664; all6144 legal training pairs retain their original
exposure imbalance. No new sampling, reshuffling, balancing or curriculum.

## Training and observations

Keep accepted E20 architecture, loss after each instruction, batch64,
AdamW configuration, clipping, CPU4 deterministic float32, autocast disabled
and native8 fixed. Run each model continuously for exactly8000 updates with
no optimizer reset or learning-rate schedule. No seed0/u2000 tensor-match
criterion applies to these independent initializations.

No intermediate scientific evaluation, best-checkpoint selection, tuning or
accuracy-dependent stopping. Operational progress may show update, elapsed
time and training loss; it never changes the registered budget. Save final
u8000 model/AdamW/RNG state before scientific evaluation. Both registered seeds
run regardless of the first seed's metrics. Nonfinite values or invariant
failures stop the affected run, preserve its artifacts and consumed work, and
mark it incomplete, never substitute a seed or silently restart it. Continue
the other seed only if the failure is demonstrably isolated; a shared code or
provenance defect suspends remaining work until a documented prospective
technical repair is independently cleared. No implicit scientific retries.

At each completed u8000, evaluate exactly seen32 programs × train192 and
validation32 states, plus the SAME E21 six primary programs and equivalent
control on all256 canonical states. Use the accepted E20 seen evaluation and
E21 one-pass-per-program semantics, frozen targets/orders, and192/32/32 strata.
No other checkpoint, seen-program reserved-state evaluation, new program,
length4/6 test or alternative inference budget. Store all prediction/target
traces once and derive metrics from them; do not rerun for strata or metrics.
Evaluation must preserve model, optimizer and RNG state.

## Prospective predicates and complete reporting

For EACH new seed, report separately:

1. Seen validation prerequisite: every primitive32/32 final joint and every
   seen composition at least31/32 final joint, using the inherited exact rule.
2. Primary replication predicate: EACH of the six E21 primary programs has
   final joint correctness at least244/256. Preserve six booleans and their
   conjunction; averaging cannot rescue a failing program.
3. Combined prerequisite-and-primary conjunction.

The E22 primary replication outcome is the conjunction of both seeds' primary
predicates. Report the two prerequisites and combined conjunction alongside it;
do not describe primary success with failed prerequisites as full gated success.
A missing final is INCOMPLETE, not an observed numerical failure or a pass.

Evaluate and retain all six primary programs and the separate ADD→XOR→XOR
control even when the seen prerequisite fails. This is prospectively authorized
fixed diagnostic reporting, avoiding selective omission; it does not reopen
E15's historical gate. Control never enters the primary conjunction/aggregate.
Always preserve all failed rows and both seeds. Report final joint/x/y, each
prefix joint, full trace, integer denominators, per-program and state-stratum
counts. Compare seed0 descriptively as historical context only; no significance
claim from three selected initializations and no independent-case inference
from correlated program/state observations.

## Budget, provenance and clearance

Nominal total training:16000 updates,1024000 batch examples,16384000 internal
substeps. Per final, seen evaluation7168 cases/18368 readouts/146944 substeps;
E21 evaluation1792 cases/5120 readouts/40960 substeps. Two-final total scientific
evaluation17920 cases/46976 readouts/375808 substeps. Report actual work and
wall time separately for training, scientific inference, tiny QA, failures,
recovery and independent reviewer replay.

Use only new E22 module/runner/tests and fresh E22 preflight/run roots; reject
nonempty roots and checkpoint overwrite. No generic framework or historical
source changes. Root froze the exact206-file `results/E22_PROTECTED_HASHES.json` list
covering accepted E21/E20 and earlier evidence. Check it before/after. Freeze before update1:
this protocol/config, reviewed sources and imported dependencies, protected
reference hashes, E15/E20/E21 manifests, exact evaluation sets/targets and
semantic signatures, seed routing, initial states, full/base stream/target
hashes, environment, optimizer settings and actual parameter membership.

Final checkpoints bind seed/update/native budget and complete model/AdamW/RNG
states/digests to that preflight, source/protocol/config and data provenance.
Strict validation rejects wrong seed, update, initial state, stream, sources,
architecture or digest. Keep operational partial artifacts but never report an
absent final as completed. Recovery may evaluate an existing valid final with
explicit accounting; it cannot train, overwrite or manufacture a final.

Independent pre-training clearance requires initialization seed0 bitwise
identity without retraining; reproducible seed1/2 full initialization and local
projection routing; identical data and actual optimizer membership; AND an
ACTUAL tiny production-runner execution with isolated QA injections, crossing
a stream-repeat boundary, taking final save/evaluation/report/progress paths.
Legal seen toy programs only in QA; no E22 model-result preview. Verify final
model/optimizer roundtrip and next-update equality, evaluation state/RNG
preservation, strict metadata/digest rejection, incomplete-run reporting,
244-pass/243-fail and prerequisite-fail-with-primary-still-reported cases,
control exclusion and both-seed execution after first-seed numerical failure.
Manual optimizer loops alone cannot satisfy actual-runner QA. Production CLI
must not expose tunable scientific budgets through QA controls.

After code clearance, run both fixed seeds and perform independent result
review of provenance, all stored traces/counts/predicates, budget, protected
hashes and final-checkpoint evaluation replay, accounting replay separately.
Stop after both outcomes and review, regardless of pass/fail. No extension,
new seed, retuning, new set or claim beyond this conditional replication.
