# E27 independent code review — CLEAR

2026-09-08. Focused independent design/source review and one targeted suite invocation before canonical preflight or scientific training. No remaining blocking finding.

- Fourteen existing projections retain exact FP32 masters and initial RNG, with no construction during reclassification. W4 now correctly inherits BitLinear; the earlier inventory blocker is repaired. Only forward weight quantization changes to per-matrix absmax/7, epsilon floor 1e-8, signed [-7,7] rounding and identity STE. Activations, bias and persistent state remain FP32. This is a range/resolution quantizer-package test, not isolated bit count or LSQ.
- Literal independent numeric oracle verifies scale-one levels, positive/negative ties-to-even, full-matrix scaling across rows, epsilon floor and identity STE. The targeted suite also verifies activation identity, finite gradients, all14 layers, exact initialization/RNG, repeated data/target stream, strict checkpoint rejection and genuine next-update optimizer/RNG equality.
- Fixed runner preserves three seeds, 16000 updates, native8, 151232 parameters, original AdamW/update mechanics, final-only evaluator and unchanged seen/primary/all-three predicates. Numerical predicate failure continues later seeds; technical exceptions preserve suspended costs and do not retry.
- Paired comparison uses frozen E26 Wternary/A32 and E24 float on identical states/targets/strata. Threshold244/256 and separate excluded control remain unchanged. Manifest provenance freezes sources, protocol, initialization/RNG, stream and comparator artifacts.
- Independently verified all281 protected entries, including all261 historical entries, and all14 root comparator-reference hashes. Canonical preflight/run directories remain absent.
- One independent targeted invocation: **11 passed in14.02s**, fresh `runs/e27_w4_review_qa_tests_v1`. No unintended failures or retries. The one intentional exception occurs before evaluation forward after one update and is preserved.

Reviewer QA totals:4 training updates,6 reload-identity updates,3 evaluation forwards;13 recurrent-model forwards,26cases/readouts and208internal substeps. Projection-only calculations are separate. Exact accounting: `results/E27_W4_REVIEW_QA.json`.

CLEAR permits the fixed canonical preflight/scientific run under root coordination. Final result ACCEPT still requires the registered independent213-forward replay and prediction-derived DSL/strata/paired/predicate/provenance verification.

## Reviewed SHA256

- `looped_bitnet/w4_e27.py`: `7aebe3b6bd5a5cc800e2287ccf2016df2db4ff4bc17498540f4114b04d1f6992`
- `scripts/w4_e27.py`: `b5584b18e6dbf1734a2e9beaec09677e44f9d6466ce51f327b8923988fc4a219`
- `tests/test_w4_e27.py`: `2f3d91544ca2c186f6ddb97325436aabd19360c16e0c8e5941751f68e8ef8987`
- `results/E27_W4_PROTOCOL.md`: `e1f6e55a38332e03422126bd5f8978ae88483b5af5e7d718502456f38a189798`
- `results/E27_PROTECTED_HASHES.json`: `effaf64c9f6aa567a3327c0c8524b0bc40b44a19f2fcd0522fc0cc311e83fbce`
- `results/E27_COMPARATOR_REFERENCE.json`: `d799ff1e13883ada738e0cc3415006e1f615ada4ac6cf57d677c82fca0f011c4`
