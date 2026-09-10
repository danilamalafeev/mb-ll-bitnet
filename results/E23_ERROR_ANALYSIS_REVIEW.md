# E23 independent review — ACCEPT

Reviewer Astra/low,2026-09-07. The reproducible script, machine artifact and
human interpretation agree with independent saved-trace recounts. No model was
loaded, no forward pass or optimizer update occurred, and no new case was used.

## Verification

Used the repository skeleton tool before targeted source reading. The script
imports only standard-library modules and reads the three frozen JSON inputs;
there is no checkpoint/model/project-code loading path. Recomputed input byte
hashes match `E23_ERROR_ANALYSIS_INPUT_HASHES.json` before and after review.

Independently recounted all4608 primary program/state cases using a direct
ADD-mod16/XOR/SWAP interpreter, including per-boundary flags, earliest divergent
instruction, final recovery, x/y/both errors and denominators. Source boundary
correctness was not treated as cumulative prefix correctness. Verified all
program/seed/stratum metrics and every retained error/recovery case, including
its traces, state identity and first-error instruction/op.

Independently reconstructed every conditional feature numerator/denominator
from GOLD input state, restricted to all previous readouts correct, for all
programs/instructions/strata. Verified aggregation only within matching
instruction index and operation. Checked all equality/zero and sum-threshold
features and their rates, including null rates for empty feature groups.
For ADD, x+y>=16 is modulo16 overflow, not every inter-bit carry. In non-ADD
cells the same numerical feature is only a gold-input sum threshold and should
not be interpreted as an arithmetic overflow mechanism. The human report's
ADD-specific interpretation respects this distinction.

Verified exact program+initial-state identities for overall and per-program
pairwise intersections/unions/Jaccard rates and triple overlap. Control and
seen-program subtrees do not enter these primary analyses.

Executed the complete script with output redirected to a fresh temporary file:
the reproduced JSON equals the delivered JSON exactly. Calling the normal
entry point with its existing output raises FileExistsError and leaves its
bytes unchanged. The temporary reproduction was removed; all input and final
artifact hashes remained unchanged. Independent review scripts:
`/tmp/e23_review_recount.py` and `/tmp/e23_final_review.py`.

## Confirmed headline counts

| Seed | First error1/2/3 | Any trace error | Final error | Recovered by final |
|---|---|---:|---:|---:|
|0|0/0/14|14|14|0|
|1|4/6/67|77|73|4|
|2|0/9/42|51|50|1|

Final-error set intersections: seed0/1=1, seed0/2=4, seed1/2=5; triple=0.
The corresponding unions86/60/118 and one-sided counts match the human table.

For ADD→XOR→ADD after both previous readouts are correct, gold-overflow versus
no-overflow first-error counts are seed0:0/120 versus2/136; seed1:6/119 versus
6/135; seed2:1/118 versus11/135. Different risk-set denominators by seed are
preserved. All prose/table values in `E23_ERROR_ANALYSIS.md` agree.

## Interpretation and stop

Late observable divergence does not prove earlier latent state correctness;
small overlaps do not establish error independence, randomness or an evaluated
ensemble. The overflow analysis does not support an all-errors-from-overflow
explanation and does not exclude other carry-related failures. These are
post-hoc descriptive findings from already opened finite evaluation data.

The human report describes a possible all-three-checkpoint continuation as
unregistered and unlaunched, with no selected seed or changed threshold. That
suggestion is not authorized execution by this review. No blocking issue
remains in the requested analysis. E23 scientific cost is0 updates and0 model
forwards, including this review. Stop after the accepted analysis.
