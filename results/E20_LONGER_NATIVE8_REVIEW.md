# E20 independent result review — ACCEPT

Reviewer: Astra/low,2026-09-07. Accept the single fixed8000-update native8 run.
All mandatory milestones and the strict2000 reproduction gate verified. No
additional training or extended evaluation was performed by this result review.
The preregistered stop is reached; no further work follows from these metrics.

## Verified trajectory

| Update | Length3 train joint macro | Length3 validation joint macro | Primitive validation ADD/XOR/SWAP |
|---|---:|---:|---|
| 2000 | 41.3442% | 29.3155% |26/32,32/32,32/32 |
| 4000 | 97.5198% | 89.8810% |31/32,32/32,32/32 |
| 8000 | 100.0000% | 99.8512% |32/32,32/32,32/32 |

At8000, every one of the32 train-program rows is192/192 final-joint correct.
On validation, ADD→ADD→SWAP is31/32; every other program is32/32. Length2
validation macro is100%. Thus every seen composition is individually at least
31/32 and all primitives are32/32, as a local numerical prerequisite check.
The historical gate belonged to the separately registered paired QAT/GRU
experiment: these numbers do NOT automatically pass that original protocol or
authorize reserved states, new compositions or deeper programs for E20.

The primary registered8000-minus2000 length3 train improvement is+58.6558pp;
validation improves+70.5357pp. Both exact unrounded native4/u2000 catch-up
predicates are true at4000 and8000, and false at2000. Final checkpoint remains
fixed8000 irrespective of the intermediate result.

## Independent verification

- Strict-loaded u2000/u4000/u8000 in native8 mode with full model and AdamW
  states, RNG payload/digest, source/protocol/config/reference/stream/initial
  provenance, counts and cumulative costs. Each optimizer has all32 parameter
  states; every step counter equals its milestone. Model inventory is151232
  parameters and the registered native8 budget at all three milestones.
- Reconstructed and checked saved initial state, common digest, exact source
  map and E18 reference identity through the frozen preflight loader.
- Recomputed the entire8000-batch and target digests, repeated phase identity
  and total8192000 training substeps. This repeats the SAME2000-batch stream
  four times; no new random training cases or E19 balancing were introduced.
- Compared all32 u2000 model state tensors bitwise, with exact keys/dtypes/
  shapes, to repaired E18 native8. This confirms the mandatory prefix gate
  and baseline reproduction, not independent seed replication.
- Re-evaluated all three milestones on exactly seen32 x train192/validation32:
  ALL192 rows matched predictions, target traces, prefix/final x/y/joint and
  full-trace counts, macros and gaps exactly. No reserved/new cases entered.
- Independently interpreted ADD-mod16/XOR/SWAP and recounted each predicted
  trace's integer metrics and length macros, without relying solely on the
  report aggregation helper. Independently checked length3 deltas and catch-up
  booleans; full report comparisons equal recomputation against frozen E18
  native4/u2000 references.
- Report status is complete, completed/fixed-final update8000, prefix gate
  true, scientific mode (not QA). All32 progress observations occur exactly
  at250-update intervals and contain finite losses.
- All176 historical protected hashes remain unchanged. Source hashes remain
  the exact code-review-cleared/preflight values; no repair or scientific
  retry was needed during E20.

Independent read-only review script `/tmp/e20_result_review.py` passed.
Canonical evidence: `runs/e20_longer_native8/report.json`, its fixed
`u2000.pt`, `u4000.pt`, `u8000.pt`, and fresh preflight. Reviewer replay adds
only verification inference; it does not mutate checkpoints or train models.

## Cost and interpretation

Scientific training:8000 updates,512000 examples,8192000 recurrent substeps,
INCLUDING the first2000-update E18 reproduction. Beyond that prefix:6000
updates,384000 examples,6144000 substeps. Registered milestone inference:
21504 program/state cases,55104 boundary readouts,440832 recurrent substeps.
The independent full replay repeats that inference work as verification.

Root reports115.82s real CLI wall time; this review verifies the artifact work
counts, not an independently measured training timer. Prior code-clearance QA
is separate: executor24 tiny updates/48 examples/768 substeps; independent
review14 updates/26 examples/416 substeps. Result review adds no optimizer
updates. These QA costs are not independent experimental replications.

The poor native8/u2000 result was sensitive to the registered longer training
schedule: fit and existing-validation behavior improve greatly by8000 without
architectural, optimizer or data-distribution changes. This supports a local
sensitivity to more optimization/repeated exposure. It does not isolate a
unique optimization mechanism, show arbitrary algorithmic generalization or
prove robustness across seeds. Validation has informed previous research and
is not untouched independent test evidence.

Native8/u8000 versus native4/u2000 uses four times the updates and eight times
the internal training substeps. Catch-up is an explicitly unequal-budget
comparison, not a fair ranking against a native4 model trained equally long.
E19 remains governed by its separate prospective protocol and must not adapt
to this result without a new explicit registration. No additional milestone,
seed, training continuation or reserved evaluation is authorized here.
