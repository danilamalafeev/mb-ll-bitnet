# E26 independent final review — ACCEPT, restoration FAIL

2026-09-08. All three fixed scientific finals are technically valid. Registered all-three-seed restoration fails: combined predicates are false/true/false. No blocking finding or additional training is required.

| Seed | E26 primary correct counts, each /256 | E25 → E26 errors /1536 | Float errors | Seen / primary / combined | Paired wins/losses versus E25 | Versus float |
|---|---|---|---|---|---|---|
| 0 | 255,238,245,253,246,244 | 64 → 55 | 14 | false / false / false | 60 / 51 | 14 / 55 |
| 1 | 256,254,256,256,256,256 | 48 → 2 | 2 | true / true / true | 48 / 2 | 1 / 1 |
| 2 | 253,244,248,250,249,245 | 49 → 47 | 0 | false / true / false | 46 / 44 | 0 / 47 |

Controls are 250/255/245 and remain excluded. Seed0 seen failures: XOR→XOR→ADD30/32 and SWAP→ADD→ADD30/32. Seed2: ADD→ADD→ADD30, ADD→ADD→SWAP29, ADD→SWAP→SWAP30, XOR→XOR→XOR30, XOR→SWAP→XOR30, SWAP→XOR→XOR30, each /32. Seed1 has no seen failure. All primitives pass32/32; the registered composition prerequisite remains31/32 and each primary threshold244/256.

## Independent evidence

- Reconstructed current manifest and compared canonical/preflight manifests exactly; code-cleared source/protocol hashes match. Initial masters and actual historical RNG, ordered data and targets, repeated stream, config and comparator references match. All261 protected files and both frozen comparator sets remain unchanged.
- Strictly loaded all3 final checkpoints: exact14 weight-only layers,151232 FP32 parameters/native8, optimizer configuration/membership/moments and step16000, model/optimizer/RNG digests, training mode, initial and manifest provenance. CPU4 deterministic mode verified.
- Replayed all213 final evaluation forwards once. Complete evaluation dictionaries match saved results exactly, including every prediction, target trace, correctness flag, state/stratum, metric and conjunction. Model/AdamW/RNG and training mode remain unchanged; all outputs finite.
- Independently reconstructed ADD modulo16/XOR/SWAP target traces from raw saved predictions; recomputed final joint/x/y, per-instruction correctness, full-trace counts and every composition train/validation/test stratum. Independently paired states/programs/targets against BOTH frozen comparators, reconstructed all four outcomes, metric/rate deltas and seen/primary/control counts. Neither comparator was rerun or retrained.
- All3 per-seed reports equal aggregate entries, completed=attempted16000 each, progress spacing250 through16000; completed and attempted forward costs match registered budgets. Scientific reload-training cost is zero, all-seed predicates recompute correctly, no skipped seeds or retry artifacts are needed.
- Reviewed root human report `results/E26_WEIGHT_ONLY.md`: numerical tables and interpretation agree with independent results; its pending-review wording can now be replaced by ACCEPT.

Durable audit files: `runs/e26_weight_only_review_final/check_counts.py`, `counts-e25_qat_match.json`, `counts-e24_continuation.json`, `replay.py`, `replay.json`. Count script reuses independently written E25 reviewer DSL/count routines, adapted to each comparator; it does not call scientific metric/paired helpers. Replay uses the accepted evaluator and compares its entire output to saved evidence.

## Accounting and limits

This final review adds **zero training updates** and exactly **213 evaluation forwards /26880 cases /70464 readouts /563712 recurrent substeps**, completed in8.17s. No failed model replay or repeat. Direct JSON/DSL/provenance/hash checks add zero forwards. Earlier code-review QA is separately recorded in `results/E26_WEIGHT_ONLY_REVIEW_QA.json` and must not be added to scientific costs.

Removing A8 alone did not restore the registered task level across all3 seeds. One seed passes and matches float's error count but has different predictions; other seeds retain failures and worse seen performance than E25. This is a mixed seed effect under the fixed ternary training recipe, not equivalence to float, a forward-only rounding diagnosis, proof of ternary impossibility, or a complete weights-by-activations interaction. Training and inference changed together; floatweights/A8 is absent. Opened finite E21, fixed data seed0, three initializations and adaptive preceding work limit inference; no packed inference, general reasoning or length-transfer claim.

The registered fixed-run and final-review stop condition is satisfied. No variants or extra budget are authorized by this review.

## Canonical SHA256

- Report: `28448f44d69ba46fa7045a8ba97022f33e991c824e1b6bffbcb299dceee414eb`
- Manifest: `1178370699c769ee1cce01ad5bcb2b5bd5fad4aa42c85641234ca499e716e940`
- seed0/u16000.pt: `3922bf5482908cda5590e4ff995f240bf2f2aa0d40531a6dc7aecece1d71c505`
- seed1/u16000.pt: `69da97c01fee9ce5b8e367b1b82b45875b7ada312c4bcac4592f694d49e6aa6d`
- seed2/u16000.pt: `fc24e80ba04b1d3621f88afa557f099669815758d8fecf37a751f187edf7dc95`
