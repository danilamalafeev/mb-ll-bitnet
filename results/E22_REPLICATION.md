# E22 — initialization-seed replication

Status: complete; independent result review **ACCEPT** (`results/E22_REPLICATION_REVIEW.md`). **The registered primary replication predicate is false.** Both fresh runs completed8000 updates; neither new seed passed all six primary program thresholds. This is a valid numerical negative result, not a technical failure.

Fixed architecture: float core with signed-bit inputs, native8,151232 parameters. Seeds1/2 vary both core and local x/y projection initialization; data seed0 and the exact E20 ordered2000 batches repeated four times are unchanged. Seed0 construction with the new factory was bitwise verified against the accepted original initial state, without retraining seed0.

## Six-program primary criterion

Each entry is correct final answers out of256. Each of the six rows must be≥244 for a seed to pass; both new seeds must pass for E22 primary replication. Historical seed0 is reused from E21, not new computation or a new replicate.

| Program | Seed0 historical | Seed1 | Seed2 |
|---|---:|---:|---:|
| ADD → XOR | 256 | 255 | 253 |
| ADD → ADD → XOR | 253 | **233** | 248 |
| ADD → XOR → ADD | 254 | **243** | **241** |
| ADD → XOR → SWAP | 255 | 248 | 250 |
| XOR → ADD → XOR | 253 | 244 | 248 |
| SWAP → ADD → XOR | 251 | **240** | 246 |

| New seed | Primary total /1536 | Programs meeting≥244 | Seen prerequisite | Primary conjunction | Combined conjunction |
|---|---:|---:|---|---|---|
| 1 | 1463 (95.2474%) | 3/6 | False | False | False |
| 2 | 1486 (96.7448%) | 5/6 | True | False | False |

Aggregate accuracy does not rescue a failing row. Seed1 fails ADD→ADD→XOR, ADD→XOR→ADD and SWAP→ADD→XOR. Seed2 fails ADD→XOR→ADD at241/256, while passing its seen prerequisite. Both final evaluations were executed regardless of seed1 failure.

## Seen-program prerequisite and fit

| Seed | ADD val /32 | XOR val /32 | SWAP val /32 | Length3 train macro | Length3 val macro |
|---|---:|---:|---:|---:|---:|
| 1 | 31 | 32 | 32 | 98.9335% | 97.3214% |
| 2 | 32 | 32 | 32 | 99.8264% | 98.8095% |

All prerequisite failures are retained below; primitives require32/32, other seen programs≥31/32. Seed2 has no prerequisite failure.

| Seed | Failed seen program | Joint /32 | Required |
|---|---|---:|---:|
| 1 | ADD | 31 | 32 |
| 1 | ADD → SWAP | 30 | 31 |
| 1 | ADD → ADD → ADD | 29 | 31 |
| 1 | ADD → ADD → SWAP | 30 | 31 |
| 1 | ADD → SWAP → ADD | 29 | 31 |
| 1 | ADD → SWAP → XOR | 30 | 31 |
| 1 | SWAP → ADD → ADD | 30 | 31 |

## Primary strata, traces and register-wise metrics

| Seed | Program | Train-state /192 | Val-state /32 | Reserved-state /32 | Full trace /256 | Prefix joint /256 | Final x /256 | Final y /256 |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | ADD → XOR | 192 | 32 | 31 | 254 | 255, 255 | 255 | 256 |
| 1 | ADD → ADD → XOR | 177 | 29 | 27 | 232 | 255, 254, 233 | 233 | 256 |
| 1 | ADD → XOR → ADD | 186 | 30 | 27 | 242 | 255, 255, 243 | 243 | 256 |
| 1 | ADD → XOR → SWAP | 187 | 31 | 30 | 247 | 255, 255, 248 | 256 | 248 |
| 1 | XOR → ADD → XOR | 183 | 29 | 32 | 244 | 256, 256, 244 | 244 | 256 |
| 1 | SWAP → ADD → XOR | 182 | 29 | 29 | 240 | 256, 255, 240 | 240 | 255 |
| 2 | ADD → XOR | 192 | 30 | 31 | 253 | 256, 253 | 253 | 256 |
| 2 | ADD → ADD → XOR | 187 | 29 | 32 | 248 | 256, 256, 248 | 248 | 256 |
| 2 | ADD → XOR → ADD | 184 | 30 | 27 | 241 | 256, 253, 241 | 241 | 256 |
| 2 | ADD → XOR → SWAP | 189 | 31 | 30 | 249 | 256, 253, 250 | 256 | 250 |
| 2 | XOR → ADD → XOR | 187 | 31 | 30 | 248 | 256, 256, 248 | 248 | 256 |
| 2 | SWAP → ADD → XOR | 184 | 31 | 31 | 246 | 256, 256, 246 | 246 | 256 |

Primary stratum totals: seed1 train-state1107/1152, validation-state180/192, reserved-state176/192; seed2:1123/1152,182/192,181/192. Strata are prior exposure of initial states on other programs, not independent replications. Some incorrect intermediate answers recover later: seed1 primary full traces1459/1536 versus1463 final answers; seed2 full traces1485 versus1486 final answers.

## Equivalent control, excluded from primary

| Seed | ADD→XOR→XOR final /256 | Full trace /256 | Prefix joint /256 |
|---|---:|---:|---|
| 1 | 246 | 245 | 255, 255, 246 |
| 2 | 248 | 248 | 256, 253, 248 |

## All seen-program final counts

| Program | Seed1 train /192 | Seed1 val /32 | Seed2 train /192 | Seed2 val /32 |
|---|---:|---:|---:|---:|
| ADD | 192 | 31 | 192 | 32 |
| XOR | 192 | 32 | 192 | 32 |
| SWAP | 192 | 32 | 192 | 32 |
| ADD → ADD | 192 | 32 | 192 | 32 |
| ADD → SWAP | 192 | 30 | 192 | 32 |
| XOR → ADD | 192 | 32 | 192 | 32 |
| XOR → XOR | 192 | 32 | 192 | 32 |
| XOR → SWAP | 192 | 32 | 192 | 32 |
| SWAP → ADD | 191 | 32 | 192 | 32 |
| SWAP → XOR | 192 | 32 | 192 | 32 |
| SWAP → SWAP | 192 | 32 | 192 | 32 |
| ADD → ADD → ADD | 185 | 29 | 190 | 32 |
| ADD → ADD → SWAP | 185 | 30 | 192 | 32 |
| ADD → SWAP → ADD | 191 | 29 | 190 | 31 |
| ADD → SWAP → XOR | 189 | 30 | 192 | 32 |
| ADD → SWAP → SWAP | 191 | 31 | 192 | 32 |
| XOR → ADD → ADD | 191 | 31 | 191 | 32 |
| XOR → ADD → SWAP | 187 | 32 | 192 | 31 |
| XOR → XOR → ADD | 192 | 31 | 191 | 31 |
| XOR → XOR → XOR | 192 | 32 | 192 | 32 |
| XOR → XOR → SWAP | 190 | 32 | 192 | 31 |
| XOR → SWAP → ADD | 190 | 32 | 192 | 32 |
| XOR → SWAP → XOR | 190 | 31 | 192 | 32 |
| XOR → SWAP → SWAP | 192 | 32 | 192 | 31 |
| SWAP → ADD → ADD | 188 | 30 | 191 | 32 |
| SWAP → ADD → SWAP | 190 | 31 | 192 | 32 |
| SWAP → XOR → ADD | 190 | 31 | 192 | 31 |
| SWAP → XOR → XOR | 190 | 32 | 192 | 31 |
| SWAP → XOR → SWAP | 192 | 32 | 192 | 31 |
| SWAP → SWAP → ADD | 191 | 32 | 192 | 32 |
| SWAP → SWAP → XOR | 191 | 32 | 192 | 32 |
| SWAP → SWAP → SWAP | 192 | 32 | 192 | 32 |

## Interpretation and limits

The strong E21 seed0 result is not uniformly reproduced under these two new complete initializations at the fixed8000-update budget. Transfer remains high, but the preregistered stringent per-program criterion fails. Seed1 retains seen-program fit/prerequisite errors; seed2 passes the seen prerequisite yet fails a novel composition. These observations do not isolate insufficient optimization from representation/generalization instability or establish that a larger budget would fix the failures.

This is conditional initialization robustness on the same finite data, not data-seed replication or a fresh holdout. E21 programs were already observed before this protocol; no settings or criteria were adapted to E22 outcomes. No QAT/GRU paired claim, general reasoning claim, longer-program test or equal-compute architecture ranking is made.

## Provenance and accounting

- Protocol `results/E22_REPLICATION_PROTOCOL.md`; independent initialization reference `results/E22_INITIALIZATION_REFERENCE.json`; code clearance `results/E22_REPLICATION_CODE_REVIEW.md`.
- Fresh preflight `runs/e22_replication_preflight/`; completed run `runs/e22_replication/` with both seed1/seed2 u8000 checkpoints, per-seed reports, progress and aggregate report. Checkpoints retain model/AdamW/RNG states and preflight/seed/config/data/source digests.
- Scientific training:16000 updates,1024000 batch examples,16384000 native substeps. Scientific final evaluation:17920 program-state cases,46976 instruction readouts,375808 native substeps. No failed scientific attempt, retry, replacement seed or extension.
- CLI wall time204.52s. Measured per-seed training and evaluation seconds are listed below; process startup/preflight verification/checkpointing/reporting account for the remaining wall time.

| Seed | Training seconds | Scientific evaluation seconds |
|---|---:|---:|
| 1 | 91.3907 | 1.9406 |
| 2 | 97.3751 | 2.2385 |

- Implementation QA:63 one-example updates,504 recurrent training substeps across three executable suites;42 legal-seen forwards,10752 cases,135168 evaluation substeps. Independent code QA:21 updates/21 examples/168 training substeps,14 legal-seen forwards/3584 cases/5632 readouts/45056 evaluation substeps. Separate from scientific counts; tests included deliberate incomplete paths and numerical failures.
- Four targeted E22 tests passed; independent review also ran the inherited E21 predicate/control test (5 total) and actual temporary preflight/load/reference/tamper checks. No old source changed or broad suite repeated.
- Root verified all206 protected historical file hashes unchanged after both runs. Independent result review ACCEPT: all142 forwards/17920 cases/375808 substeps replayed exactly, with independent DSL/count/stratum/predicate/provenance checks and all206 protected hashes unchanged. This verification adds zero training.

Registered stop reached after both completed seeds and independent review. No active training or evaluation remains. Future extra training or changed settings require a new prospective protocol; this E22 result must remain preserved as measured.
