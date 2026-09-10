# E15 runner — root review blockers

2026-09-07. No scientific training has run. This supersedes any inference of
training clearance from the early code-review preflight checks; those checks
covered a narrower scope and are retained unchanged.

Observed before Terra repair:

1. Paired wins/losses were computed from differences of aggregate correct
   counts. Example: QAT correctness [true,false] and GRU [false,true] have
   one win and one loss, not zero wins/losses. Ties must be an example count.
2. Validation tie CE averaged all prefixes and then programs without equal
   length weighting. Selected protocol requires final dual-head CE averaged
   within each length, then over lengths 1,2,3.
3. Runner did not yet produce the complete registered ID/equivalent/primary
   predicates and symbolic/order/late-op controls, or durable per-arm reports
   permitting evaluation-only recovery after a failure without retraining.
4. Passing a saved prefix digest back into a loader checks internal agreement,
   not the expected input stream. Reconstruct expected prefix at selected
   update from the frozen stream definition and enforce mandatory source set.

One executor `e15_runner_repair` Terra/high is assigned this bounded repair
after repeated significant Luna/high implementation mistakes. Model design,
task, budget, thresholds and old artifacts stay fixed. Required next evidence:
CLI-level temp smoke for save/load/gate paths, adversarial paired-count and
length-macro tests, stable final source hashes and separate read-only review.
