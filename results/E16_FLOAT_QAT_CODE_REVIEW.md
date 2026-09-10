# E16 independent code review

Reviewer: Astra/low, 2026-09-07. Status: **HOLD — bounded repairs in progress**.
No E16 pilot training authorized by this review yet.

Reviewed the new module, runner and focused tests against the frozen E16
protocol. Initial architecture/data inspection supports the intended comparison:
all 14 BitLinear sites are replaced in a deep copy; master values, ordering and
152768 parameter count are paired; frozen E15 loss and stream are imported;
four substeps and fixed u2000 are unchanged. Composition macros equally weight
length2 and length3. Paired outcomes use per-case correctness rather than
subtracting aggregate counts. Evaluation retains every prefix count per program;
additional prefix macros are not a clearance requirement.

Repair findings sent to root:

- Default preflight is nested inside the run directory, conflicting with the
  required refusal of nonempty run output. Use a separate preflight directory.
- Inference-only recovery must work from the two final checkpoints even if an
  arm completion or report write failed. Validate exactly both checkpoints
  before any inference; a prior report is not a prerequisite.
- Explicitly set eval mode and deterministic CPU4 execution during evaluation,
  including a fresh recovery process.
- Exact parity controls must use torch.equal for logits, loss, gradients and
  post-update state. Check checkpoint logits and wrong source/update/manifest
  rejection, alongside already repaired prefix/optimizer/format validation.
- Persist a canonical exposure-frequency digest and compare the frozen target
  digest with the reconstructed stream before training or recovery evaluation.

These findings do not require architecture changes or broader experiments.
Final clearance awaits stable source hashes and focused test evidence.
