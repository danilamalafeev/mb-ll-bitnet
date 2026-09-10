# E13 corrected — state normalization training

Completed one corrected CPU run on 2026-09-07 with:

```text
PYTHONPATH=. .venv/bin/python scripts/state_scale_train_e13.py --protocol-cleared
```

The run used the preregistered stream (`stream_seed=20260911`, 512 test tables, manifest fingerprint `6a687884419e8ee559c117144c760c86b39e146ffd7ddc83bb86b8a233531f95`) and produced `runs/state_scale_train_e13_corrected/`. The machine report is [report.json](../runs/state_scale_train_e13_corrected/report.json); this document does not rewrite it.

## Training and gate

All six arms completed: baseline and centered-L2 for seeds 0, 1, and 2. Each arm ran 2,000 updates × batch 64 (128,000 examples; 1,024,000 example-steps). Total corrected budget was 12,000 updates, 768,000 examples, and 6,144,000 example-steps. Runtime was 193.43 s on CPU. The earlier invalid pilot cost is separate and remains frozen.

Seed 0 passed the extension gate: both arms had final validation accuracy 1.0 and intermediate validation accuracy 1.0. Seeds 1 and 2 were then run under the fixed protocol; both also passed the same gate. Checkpoints were selected by native final validation accuracy, then final validation loss.

For every seed, paired baseline/centered-L2 arms have equal initial-state, input/target-sequence, data-stream, and suite-fingerprint digests. Checkpoint metadata records the registered objective, dynamics, `resume_supported=false`, and the expected configuration.

## Four-cell cross-evaluation

Counts are correct predictions out of 512 at readout hop `k`; `B` means baseline-trained and `N` centered-L2-trained, while suffix `0/1` disables/enables normalization during evaluation.

| seed | cell | k3 | k4 | k5 | k6 | k7 | k8 |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | B0 | 496 | 357 | 277 | 208 | 173 | 121 |
| 0 | B1 | 512 | 509 | 492 | 468 | 443 | 422 |
| 0 | N0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 0 | N1 | 512 | 463 | 365 | 307 | 265 | 236 |
| 1 | B0 | 262 | 23 | 11 | 7 | 20 | 15 |
| 1 | B1 | 510 | 485 | 452 | 433 | 400 | 372 |
| 1 | N0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | N1 | 512 | 496 | 466 | 435 | 413 | 396 |
| 2 | B0 | 512 | 508 | 493 | 475 | 458 | 443 |
| 2 | B1 | 512 | 512 | 511 | 509 | 507 | 504 |
| 2 | N0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | N1 | 491 | 437 | 390 | 337 | 297 | 262 |

The evaluator performed exactly one test evaluation after training and gate resolution. Test results did not control training extension, checkpoint selection, or interpretation choices.

## Registered predicates

Primary predicate: `N1−B0` at `k8`, with minimum count delta `+26` for seeds 0 and 1 and nonnegative delta for seed 2.

| seed | N1 k8 | B0 k8 | delta | predicate |
|---:|---:|---:|---:|:---:|
| 0 | 236 | 121 | +115 | PASS |
| 1 | 396 | 15 | +381 | PASS |
| 2 | 262 | 443 | −181 | FAIL |

Primary predicate: **FAIL** (seed 2 fails its preregistered nonnegative threshold).

Strong predicate: N1 must be at least 487/512 at every `k3..k8` for all three seeds. It fails: seed 0 has 463/365/307/265/236 at k4..k8, seed 1 has 496/466/435/413/396, and seed 2 has 491 at k3 then 437/390/337/297/262. The strong predicate is therefore **FAIL**.

Additional preregistered paired k8 comparisons are retained in `report.json`: N1−B1 deltas are −186, +24, −242; B1−B0 deltas are +301, +357, +61; N1−N0 deltas are +236, +396, +262 for seeds 0, 1, 2 respectively.

## Verification and limitations

`PYTHONPATH=. .venv/bin/pytest -q tests/test_state_scale_train_e13.py` passed 12 tests before launch. After completion, `PYTHONPATH=. .venv/bin/pytest -q` passed **57 tests**. Frozen invalid artifacts were not modified. Their machine-computed provenance and audit comparison are recorded in `STATE_SCALE_TRAIN_E13_EXECUTION_VERIFICATION_FINAL.json`; the original pre-run provenance is `STATE_SCALE_TRAIN_E13_EXECUTION.json`.

This is a diagnostic training result for the fixed two-hop, eight-step objective. Both preregistered predicates fail; the result does not establish robust multi-seed continuation or a general solution to depth transfer. No retry, tuning, coefficient sweep, cloud run, or additional training was performed.

Independent [final review](STATE_SCALE_TRAIN_E13_CORRECTED_REVIEW.md) accepted the corrected run: all 96 counts and 12 paired comparisons were independently reproduced, checkpoint selection/gates/budgets/digests were checked, and a seed0/N1 replay matched every saved readout. The reviewer confirmed the negative outcome.

The initial execution provenance and intermediate correction records contain manually transcribed hash errors and must not be treated as authoritative audit results. The final verification file, the original invalid-run audit, and independent direct SHA-256 recomputation confirm the frozen files are unchanged. The separate invalid pilot cost was 4,000 updates / 256,000 examples / 2,048,000 example-steps; the corrected run cost above excludes it.
