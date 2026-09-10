# E18 independent result review — ACCEPT

Reviewer: Astra/low, 2026-09-07. Accept the completed repaired paired seed0
pilot under the prospective technical restart amendment. This is acceptance of
valid evidence, not a positive result for native8. No training was performed
by this review, and the registered stop is reached.

## Verified result

| Outcome | Native4 | Native8 | Native8−4 |
|---|---:|---:|---:|
| Length3 exposed-train final-joint macro | 92.1131% | 41.3442% | -50.7688pp |
| Length3 validation final-joint macro | 80.6548% | 29.3155% | -51.3393pp |
| ADD validation | 29/32 | 26/32 | -3 states |
| XOR validation | 31/32 | 32/32 | +1 state |
| SWAP validation | 32/32 | 32/32 | 0 states |

Both arms fit all three primitives192/192 on train. Native8 substantially
worsens already exposed composition fit as well as validation. The original
perfect primitive prerequisite remains unmet for both arms because ADD is
imperfect; no downstream gate or reserved evaluation set is opened.

## Independent verification

Strict-loaded both repaired final checkpoints with native budgets4/8 and their
frozen source, configuration, protocol, encoder, optimizer, initial/common,
data, target and final-state provenance. Reconstructed initial models and
validated both persisted initial artifacts against the unchanged E17 bits
reference. Both arms have151232 parameters; native8 adds no parameters.

Recomputed the complete fixed stream/target digests and coverage structure;
all6144 train program/state pairs were exposed. Re-evaluated both models only
at their native budgets on seen32 x train192/validation32. ALL128 metric rows
matched exactly, including predictions, prefix correctness, final x/y/joint,
full trace, macros, train-validation gaps, paired outcomes, deltas and display
rows. Independently interpreted ADD-mod16/XOR/SWAP from initial states and
recounted every predicted trace's integer metrics. Independent paired counts
and length-balanced macros agreed. Root's human report all32 program table
also matches the machine report exactly.

All32 native4 final state tensors are bitwise identical to the existing E17
bits final checkpoint. This is exact baseline reproduction, not independent
seed replication. Both progress and completion records match the report, have
exactly eight finite observations at250..2000, and agree with checkpoint byte
hashes. All138 historical protected hashes remain unchanged.

Successful paired training cost is4000 optimizer updates,256000 examples and
3072000 recurrent substeps (1024000 native4;2048000 native8). Native8 doubles
internal work at the same update count. Final inference is14336 program/state
cases,36736 boundary readouts and220416 substeps. Recorded repaired-run duration
including its final evaluation is42.8897s. This review's replay is additional
verification inference, not scientific training.

## Technical failure preservation and provenance

The first attempt failed after one native4 optimizer update from the first
legal length1 batch:64 examples and256 substeps. There was no final checkpoint,
validation result or native8 training. Verified every original accounting hash:
failed-run and original-preflight bytes in place, original source/protocol
bytes in `results/E18_FIRST_ATTEMPT_SOURCE/`. No checkpoint exists under the
failed run. The live runner equals the preserved original with ONLY the
registered PROGRESS_INTERVAL import added. Its SHA256 is
`6c43e4e7266057232fc2f5d4e687578b4983e2b257625c8d52f0ade561be6b27`, matching
repaired preflight and checkpoint provenance. Original module/tests/protocol
are unchanged.

The separate prospective amendment authorizes this mechanical restart and
fresh initialization of BOTH arms; it does not silently overwrite the earlier
no-retry clause. Total scientific work INCLUDING failure is4001 updates,
256064 examples and3072256 substeps. The disclosed repair smoke adds6 tiny QA
updates/6 examples/64 substeps, separate from scientific work; other tiny
controls are also QA. The previous CLEAR missed the actual runner path and
must not be represented as sufficient validation of the original executable.
No scientific outcome from the failure selected this repair or altered the
intervention. This is a technical restart, not an additional replication.

Canonical evidence: `runs/e18_step_budget_repaired/report.json` and its two
final checkpoints; preflight `runs/e18_step_budget_repaired_preflight/`.
Independent read-only review scripts passed from `/tmp/e18_result_review.py`
and `/tmp/e18_counts_review.py`.

## Interpretation and stop

Under this one seed, fixed update budget and unchanged optimizer, native8's
training/inference compute package is worse. This does not isolate training
optimization from recurrent state dynamics, prove an insufficient-compute
hypothesis universally false, or show additional recurrent computation is
universally harmful. Native8 was trained natively; this is not an inference-only
comparison. Validation has informed previous experiments and is not untouched
test evidence. Equal parameters do not make the two trajectories equivalent.

A bounded next hypothesis, requiring separate prospective registration, is to
retain the native4 float bit-input baseline and test whether balanced exposure
to seen program/state pairs improves optimization at a fixed update budget.
E18 does not establish that sampling imbalance causes the errors; this is a
candidate following the completed quantization, representation and compute
controls. No curriculum, balancing implementation or further run is authorized
by this review. Stop after this accepted pair and review.
