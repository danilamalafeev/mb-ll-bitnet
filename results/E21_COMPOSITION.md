# E21 — first novel-composition evaluation

Status: ACCEPT — independent Astra/low result review (`results/E21_COMPOSITION_REVIEW.md`). Frozen E20/u8000, seed0, native8, float core with signed-bit inputs,151232 parameters. Zero training or checkpoint selection during E21.

**All six registered primary programs pass the per-program criterion≥244/256.** Total1522/1536 correct final answers and full traces (99.0885%). This is measured transfer to six previously excluded, semantically distinct program mappings in this finite register task. It is one seed after adaptive architecture/training choices on reused validation; it is not a claim of general reasoning or an original QAT/GRU paired result. The tested core is float, not the earlier quantized core.

## Primary counts

| Program | All /256 | Train-state /192 | Validation-state /32 | Reserved-state /32 | Full trace /256 | Prefix joint counts /256 | ≥244 |
|---|---:|---:|---:|---:|---:|---|---|
| ADD → XOR | 256 | 192 | 32 | 32 | 256 | 256, 256 | True |
| ADD → ADD → XOR | 253 | 192 | 31 | 30 | 253 | 256, 256, 253 | True |
| ADD → XOR → ADD | 254 | 192 | 31 | 31 | 254 | 256, 256, 254 | True |
| ADD → XOR → SWAP | 255 | 191 | 32 | 32 | 255 | 256, 256, 255 | True |
| XOR → ADD → XOR | 253 | 191 | 32 | 30 | 253 | 256, 256, 253 | True |
| SWAP → ADD → XOR | 251 | 187 | 32 | 32 | 251 | 256, 256, 251 | True |

All initial states were enumerated. Strata refer to prior exposure of starting states on OTHER seen programs; none of these six programs was trained. Primary totals by stratum:1145/1152 train-state,190/192 validation-state,187/192 reserved-state. The reserved-state result is97.3958%; no separate stratum threshold was selected.

Every first and second intermediate answer is correct. The14 primary failures occur on the third instruction. Full-trace totals equal final-answer totals here, so the high final score is not hiding earlier wrong readouts. Correct intermediate readouts do not themselves prove the mechanism or causal use of a symbolic register representation.

## Separate equivalent-function control

ADD → XOR → XOR: **252/256** final and full trace; prefix counts[256, 256, 252]. Strata: train-state190/192, validation-state32/32, reserved-state30/32.

Its final truth table equals ADD on all256 states, despite the novel instruction sequence and different intermediate trace. It is not a seventh novel mapping and is excluded from the primary numerator, denominator and conjunction. No additional ADD model inference was performed.

## Register-wise counts

| Program | Final x /256 | Final y /256 |
|---|---:|---:|
| ADD → XOR | 256 | 256 |
| ADD → ADD → XOR | 253 | 256 |
| ADD → XOR → ADD | 254 | 256 |
| ADD → XOR → SWAP | 256 | 255 |
| XOR → ADD → XOR | 253 | 256 |
| SWAP → ADD → XOR | 251 | 256 |
| ADD → XOR → XOR | 252 | 256 |

## Errors retained for audit

| Program | Initial state | Predicted final | Correct final | Stratum |
|---|---|---|---|---|
| ADD → ADD → XOR | [4, 11] | [5, 11] | [1, 11] | test |
| ADD → ADD → XOR | [12, 4] | [8, 4] | [0, 4] | validation |
| ADD → ADD → XOR | [15, 1] | [1, 1] | [0, 1] | test |
| ADD → XOR → ADD | [6, 2] | [8, 2] | [12, 2] | test |
| ADD → XOR → ADD | [14, 10] | [0, 10] | [12, 10] | validation |
| ADD → XOR → SWAP | [1, 5] | [5, 1] | [5, 3] | train |
| XOR → ADD → XOR | [4, 3] | [13, 3] | [9, 3] | train |
| XOR → ADD → XOR | [4, 11] | [5, 11] | [1, 11] | test |
| XOR → ADD → XOR | [10, 6] | [8, 6] | [4, 6] | test |
| SWAP → ADD → XOR | [3, 7] | [13, 3] | [9, 3] | train |
| SWAP → ADD → XOR | [3, 15] | [5, 3] | [1, 3] | train |
| SWAP → ADD → XOR | [9, 3] | [1, 9] | [5, 9] | train |
| SWAP → ADD → XOR | [11, 7] | [13, 11] | [9, 11] | train |
| SWAP → ADD → XOR | [11, 15] | [5, 11] | [1, 11] | train |
| ADD → XOR → XOR | [9, 9] | [0, 9] | [2, 9] | test |
| ADD → XOR → XOR | [9, 13] | [4, 13] | [6, 13] | train |
| ADD → XOR → XOR | [12, 6] | [10, 6] | [2, 6] | train |
| ADD → XOR → XOR | [15, 1] | [1, 1] | [0, 1] | test |

## Scope and provenance

- Prospective protocol: `results/E21_COMPOSITION_PROTOCOL.md`; independent symbolic enumeration: `results/E21_SYMBOLIC_REFERENCE.json`. Six primary signatures are distinct and absent from the21 unique mappings of32 seen programs. The frozen forbidden set is exactly these six plus the equivalent control.
- Source clearance: `results/E21_COMPOSITION_CODE_REVIEW.md` preserves the pending HOLD and appends hash-specific FINAL CLEAR. Eight focused tests passed; additional synthetic checks covered the actual production predicate,244/243 boundary, asymmetric strata and complete exclusion of a zero-score control. Actual QA used legal seen toy programs; no real E21 model outputs preceded clearance.
- Fresh manifest: `runs/e21_composition_preflight/manifest.json`; measured evidence: `runs/e21_composition/report.json`. Includes ordered states/targets/signatures, prediction traces and stratum labels, exact checkpoint/protocol/source/dependency provenance.
- Frozen checkpoint: `runs/e20_longer_native8/u8000.pt`, SHA256 `4507dd19d95b8a90f6ee1b487abc6493a26ba19700b784eaa2949dbfaa75c85d`. Strict E20 loading and raw tensor key/dtype/shape checks passed. File and model-state hashes unchanged before/after inference.
- Scientific work: exactly7 model forwards,1792 program-state cases,5120 instruction readouts,40960 internal state substeps,0 training updates. Primary34816 substeps; equivalent control6144. Strata/metric slicing added no inference. CLI wall time17.28s includes validation/loading and output writing.
- Synthetic QA is separate verification, not scientific model inference. Its nominal eight-step cost must not be mistaken for executed recurrent operations; code review documents synthetic checks and a tiny untrained native8 ADD control. Independent result-review replay is additional verification and separately counted.

The new single-model, single-seed numerical predicate passes. This neither retroactively passes E15’s original three-seed paired QAT/GRU protocol nor establishes behavior on longer programs or a quantized version. E21 first composition results are now observed and must not be treated as untouched holdouts in later tuning.

Independent result review strict-loaded and replayed all seven frozen-program batches, exactly reproducing all1792 traces and report fields. Independent integer DSL/count/prefix/stratum reconstruction, six predicates, control exclusion and all195 protected hashes passed. Verification replay used another40960 substeps and0 training updates (14.997s including loading/audit). Root rechecked protected hashes after review. Registered stop reached; additional seeds need their own unchanged replication protocol. E19 remains deferred design only.
