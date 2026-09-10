# E14 implementation repair — root verification

Date: 2026-09-07. This file records the pre-run implementation repair; the
subsequent repaired run and its test result are documented in
`results/LENGTH_MIXTURE_E14.md`.

The Luna code-review task could not be started because the account model quota
was exhausted. Root repaired the implementation only to address the existing
pre-code-review blockers; this file is not an independent reviewer report.

Changes in `scripts/length_mixture_e14.py`:

- replaced permissive historical novelty loading with strict per-record checks
  for source shape, transition range, start, order, and all required E07–E13
  sources, including the referenced E10 manifest;
- recorded hashes for all novelty sources and the protected E14 preflight in
  the pre-training protocol, and wrote an immutable run `preflight.json`;
- strengthened selected-checkpoint reload against arm digests, optimizer,
  parameter count, dynamics, and report/checkpoint agreement;
- restored the model's train/eval mode around native evaluation;
- removed the incorrect `data.test_max_hops == 8` assumption while retaining
  the actual h8 encoded-length versus model-token-capacity guard.

Pre-run validation completed:

- E14 targeted tests: **11 passed**;
- full repository suite: **68 passed**;
- `py_compile`: passed;
- CLI semantic smoke without `--protocol-cleared`: passed; it generated fresh
  validation/test fingerprints and stopped at the training gate;
- mandatory novelty scopes loaded with counts
  `3200, 1280, 1024, 512, 512, 512, 512, 512, 512`.

The guarded training run remains HOLD until a separate Luna/high code review
can inspect this final implementation. Existing runs and protected artifacts
were not overwritten.
