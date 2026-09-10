# E32 preliminary concurrent runner review

2026-09-08, independent e32_review Astra/low. **BLOCKED for CODE CLEAR**, source still under active implementation. Read-only inspection only; no tests rerun, no forwards or updates. Findings sent directly to executor and root; this list is a snapshot, not a review of later repaired hashes.

- `_length_evaluate` passes full E28 selection dictionary into `e28.score`, which expects the ordered list of program objects. This would fail after executing length forwards.
- Required all-seed per-arm final/fulltrace restoration, combined gates, strict width-benefit flags, W4/float128 paired comparison and signed64/128 interaction summary not yet assembled.
- Length comparison has final paired counts but only fulltrace totals, without fulltrace paired/stratum counts. Case joins must require targets, strata, correctness fields rather than optional checks/defaultFalse.
- Tiny next-update QA builds both branches from already reloaded state and may alias optimizer tensors because optimizer.load_state_dict does not guarantee disjoint storage. Preserve genuine live presave branch for uninterrupted control and deep-copy tensors. Count additional controls separately.
- QA checkpoint cost presently imports scientific16000 budget despite two tiny updates; checkpoint cost must reflect actual QA exposure. `training_seconds` includes evaluation/comparison time.
- Strict checkpoint validation remains incomplete: typed metadata, exact raw tensor shape/dtype/order before load, full optimizer groups/membership, moment shape/dtype, exact scalar tensor step, no initial/provenance defaults.
- Preflight initial payload requires exact schema/arm/width/seed/count/RNG digest validation in addition to tensors. Required source hashes must not silently skip missing files. Manifest should retain verified h64 control and actual128 inventory counts.
- Predicate helpers need exact expected scope/duplicate validation to reject vacuous all([]) acceptance.

Positive source checks: seed-major execution now loops seed thenarm; fixed E20 update is reused with clipping and finite loss/gradient rejection; forward hooks separately measure attempted/completed cases/readouts/substeps; all fixed scopes would run regardless of low accuracy; technical exception preserves partial report and stops; old models are read only; seen/E21 comparison helpers are reused. Canonical run final cost checks enforce83forwards permodel and expected training budget. Complete evaluation replay remains required after training.
