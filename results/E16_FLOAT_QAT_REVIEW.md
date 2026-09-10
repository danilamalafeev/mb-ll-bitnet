# E16 independent result review

Status: **ACCEPT** as the registered one-seed, fixed-budget paired pilot.
Reviewer: Astra/low, 2026-09-07. Review performed inference only; no training,
new seed, reserved test or additional program family was evaluated.

## Verified evidence

Independently loaded both final u2000 checkpoints through the strict E16 loader,
checked checkpoint SHA256s, reviewed source/protocol hashes, V7 canonical
manifest hash, optimizer/config, final/full stream digests and common initial
state. Reconstructed seed0 initial masters match every saved tensor exactly.
QAT has14 BitLinear modules; float has0; each has152768 parameters.

Reconstructed frozen batches and target digest and matched the complete
coverage report, including frequency digests and6144/6144 exposed combinations.
Checked two completed arms,2000 updates each, eight finite loss records each at
updates250..2000,256000 total examples and2048000 internal state updates.
No validation checkpoint selection occurred.

Reloaded both models in eval mode with deterministic CPU4 execution and
recomputed **all128 model/program/split rows**, including every integer final
joint/x/y/prefix/full-trace count and every per-case prediction. Every row
matched the report exactly. Recomputed all macros, train-validation gaps,
float-QAT deltas, paired outcomes and the full display rows exactly; paired
marginals reproduce each arm's final-joint count and their difference equals
float-only minus QAT-only. Scope remained exactly seen32 x train192/validation32.
All101 protected files still match their frozen hashes.

Additional read-only baseline check: every E16 QAT final state tensor is
**bitwise identical** to the protected E15 seed0 selected-u2000 QAT checkpoint.
This supports exact baseline reproduction without another old-experiment run.

## Accepted interpretation

| Outcome | QAT | Float |
|---|---:|---:|
| ADD train |183/192|191/192|
| XOR train |186/192|192/192|
| SWAP train |192/192|192/192|
| ADD validation |4/32|9/32|
| XOR validation |15/32|24/32|
| SWAP validation |32/32|32/32|
| Length3 train joint macro |32.14%|63.57%|

Removing both weight and activation fake quantization improves train
seen-composition fit in this seed and fixed budget: length3 gains31.42 percentage
points; the equal length2/length3 composition macro gains24.60 points.
Primitive validation also improves, but remains distinct from train fit:
ADD9/32 and XOR24/32 still fall below their near-perfect train counts. Float
has substantial residual length3 train errors as well.

The paired intervention supports a local effect of the joint fake-quantization
package under the registered conditions. It does not identify weight versus
activation effects, training versus inference effects, quantization as the sole
cause, general seed robustness, or unseen-composition/length generalization.
Validation states informed prior research and are not an independent test.

## Stop and next recommendation

The registered pilot and independent review are complete; stop here. Following
the user's accepted order, prospectively register one next control comparing
learned register-value embeddings with explicit4-bit register inputs on the
float architecture, while holding the recurrent core, four substeps and training
budget fixed. Treat representation as the next hypothesis; do not simultaneously
change steps or reopen reserved evaluation. That control has not been designed
in full, implemented or trained by this review.
