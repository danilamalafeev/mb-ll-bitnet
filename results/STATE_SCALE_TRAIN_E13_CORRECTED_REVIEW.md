# E13 corrected: independent final review

## Verdict

The corrected run is internally consistent and complete for the registered E13 diagnostic. I accept its saved artifacts as evidence for the preregistered outcome, with the outcome recorded as negative: the primary predicate `N1-B0` fails because seed 2 has delta `-181`; the stronger `N1 >= 487/512` criterion also fails. This does not support a robust three-seed improvement in fixed two-hop training with centered-L2 normalization.

This review is read-only apart from this file. I did not retrain, rerun a suite, alter code, alter either E13 run, or alter the frozen invalid artifacts.

## Independent prediction and paired-count audit

I reconstructed each target as `f^k(start)` from the 512 records in `runs/state_scale_train_e13_corrected/manifest.json` and compared it with every saved prediction at readouts `4k`, for `k=1..8`. All 12 cell-by-seed count vectors (96 counts) match the report exactly:

| seed | cell | k1 | k2 | k3 | k4 | k5 | k6 | k7 | k8 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | B0 | 512 | 512 | 496 | 357 | 277 | 208 | 173 | 121 |
| 0 | B1 | 512 | 512 | 512 | 509 | 492 | 468 | 443 | 422 |
| 0 | N0 | 512 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 0 | N1 | 512 | 512 | 512 | 463 | 365 | 307 | 265 | 236 |
| 1 | B0 | 512 | 512 | 262 | 23 | 11 | 7 | 20 | 15 |
| 1 | B1 | 512 | 512 | 510 | 485 | 452 | 433 | 400 | 372 |
| 1 | N0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | N1 | 512 | 512 | 512 | 496 | 466 | 435 | 413 | 396 |
| 2 | B0 | 512 | 512 | 512 | 508 | 493 | 475 | 458 | 443 |
| 2 | B1 | 512 | 512 | 512 | 512 | 511 | 509 | 507 | 504 |
| 2 | N0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | N1 | 512 | 512 | 491 | 437 | 390 | 337 | 297 | 262 |

The saved paired k8 comparisons also independently recompute exactly:

| seed | comparison | delta | wins | losses | ties |
|---:|:---:|---:|---:|---:|---:|
| 0 | N1-B0 | +115 | 179 | 64 | 269 |
| 0 | N1-B1 | -186 | 36 | 222 | 254 |
| 0 | B1-B0 | +301 | 309 | 8 | 195 |
| 0 | N1-N0 | +236 | 236 | 0 | 276 |
| 1 | N1-B0 | +381 | 382 | 1 | 129 |
| 1 | N1-B1 | +24 | 108 | 84 | 320 |
| 1 | B1-B0 | +357 | 364 | 7 | 141 |
| 1 | N1-N0 | +396 | 396 | 0 | 116 |
| 2 | N1-B0 | -181 | 39 | 220 | 253 |
| 2 | N1-B1 | -242 | 5 | 247 | 260 |
| 2 | B1-B0 | +61 | 63 | 2 | 447 |
| 2 | N1-N0 | +262 | 262 | 0 | 250 |

The primary preregistered predicate requires deltas `>=26, >=26, >=0` for seeds 0, 1, and 2. It therefore passes for seeds 0 and 1 and fails for seed 2. The strong predicate requires N1 counts of at least 487 at every k3..k8 for all seeds. It fails for seed 0 at k4..k8 (`463,365,307,265,236`), seed 1 at k5..k8 (`466,435,413,396`; k3 and k4 are 512 and 496), and seed 2 at k4..k8 (`437,390,337,297,262`; k3 is 491).

As a bounded spot replay, I loaded the seed 0 centered-L2 `checkpoint_best.pt` with `weights_only=True` and replayed its 512 manifest examples through the current `e13_forward` at 32 steps. Saved predictions matched at every readout 4, 8, 12, 16, 20, 24, 28, and 32.

## Protocol, manifest, and novelty

The manifest has stream seed `20260911`, 512 records, fixed input hops 2, and fingerprint `6a687884419e8ee559c117144c760c86b39e146ffd7ddc83bb86b8a233531f95`. I independently recomputed this fingerprint, confirmed 512 unique transition tables, confirmed every record is in the test split, and confirmed the report and cross-evaluation fingerprints match it.

The required novelty scopes are present with sizes `3200, 1280, 1024, 512, 512, 512, 512, 512` for E07 persisted, E07 reconstructed, E08, E09, E10, E11, E12, and invalid E13 respectively. Independent set intersections with the new manifest are zero for all eight scopes. The invalid E13 manifest is included as a novelty source, as required.

The report records one test evaluation. Source chronology places cross-evaluation after training and the seed-0 gate; validation and checkpoint selection use only native validation. The test manifest is shared across seeds, so counts are paired on the same 512 examples.

## Training, gates, selection, and budgets

All six arms completed 2,000 updates with batch 64: 12,000 corrected updates, 768,000 examples, and 6,144,000 fixed eight-step example-steps. Every arm reports 152,512 parameters, 128,000 examples, and 1,024,000 example-steps. Seed 0 passed the extension gate with final and intermediate validation accuracy 1.0 for both arms; seeds 1 and 2 were then executed regardless of their own gate and also passed with both accuracies 1.0. Each validation history has eight entries, and the selected update is 2,000 for every arm.

The selected checkpoint rule is final validation accuracy, then final validation loss. I recomputed this ordering from every saved validation history and found the reported selected update in all six cases. The registered objective and fixed optimizer metadata agree within each paired seed. Initial-state, data-stream, input/target-sequence, and suite-fingerprint digests match between baseline and centered-L2 arms for seeds 0, 1, and 2. The implementation's input/target digest is the stream hash over collated input IDs and final targets; the intermediate target is deterministically derived from the same examples.

I independently hashed all 12 corrected best/latest checkpoint files and matched each report entry. Loading every best checkpoint with the current code passed the strict diagnostic checks for format, objective, seed, configuration, arm/dynamics, normalization schedule, radius, epsilon, and `resume_supported=false`; payload metadata also matches the report's digests, optimizer, selected validation, and update. The reviewed code preserves readout-before-reset ordering, uses the normalized arm for native validation, and evaluates all four B0/B1/N0/N1 cells.

The earlier invalid pilot is excluded from conclusions and remains unchanged. Its frozen audit records two seed-0 checkpoints at 2,000 updates each, so its separate audit cost is 4,000 updates, 256,000 examples, and 2,048,000 example-steps under the same batch and fixed-step accounting. This cost is not mixed into the corrected 12,000-update budget.

## Provenance and limitations

The direct SHA-256 audit of the six files named by `results/STATE_SCALE_TRAIN_E13_INVALID_AUDIT.json` matches all six authoritative audit hashes and byte counts. The initial `STATE_SCALE_TRAIN_E13_EXECUTION.json` contains a manually transcribed incorrect hash for the frozen invalid report. The intermediate correction file contains an incorrect report audit value and `matches_audit: false`; it must not be treated as the authoritative result. `STATE_SCALE_TRAIN_E13_EXECUTION_VERIFICATION.json` also retains a manual transcription error; the authoritative append-only record is `STATE_SCALE_TRAIN_E13_EXECUTION_VERIFICATION_FINAL.json`, which agrees with the audit. The direct recomputation above is the basis for accepting the provenance claim. The frozen invalid run and its report were not rewritten.

The executor reports the post-run full suite as 57 passed and the preflight targeted suite as 12 passed; this review did not duplicate the full suite. The result remains a diagnostic fixed-hop continuation experiment trained on two hops with an eight-step objective. It does not establish general depth transfer, untouched out-of-distribution behavior, a learned routing policy, or a language-model capability. No retry, tuning, coefficient sweep, or additional training is warranted within E13 after the preregistered predicates and finite budget have been resolved.

Reviewed artifacts: `runs/state_scale_train_e13_corrected/manifest.json`, `runs/state_scale_train_e13_corrected/report.json`, all corrected checkpoints, `scripts/state_scale_train_e13.py`, `results/STATE_SCALE_TRAIN_E13_INVALID_AUDIT.json`, and the append-only execution verification records.
