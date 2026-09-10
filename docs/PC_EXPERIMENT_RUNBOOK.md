# AI2 experiment runbook

This is an inference-only continuation. The Mac checkpoints, manifests, and historical predictions are immutable inputs. All new files go under a fresh `runs/pc_length_pc_v1/` directory on AI2.

## Order

1. Verify `pc_transfer_v3` and `pc_predictions_v1` hashes and preserved relative paths. Do not use `pc_transfer_v1`.
2. Create a Windows Python environment with a CUDA-compatible PyTorch build. Record Python, PyTorch, CUDA, driver, GPU, dtype, and thread settings. Do not copy the Mac `.venv`.
3. Apply and test only the narrow Windows compatibility adapter needed for `resource` import and device selection. Preserve all provenance checks and source hashes.
4. Run pure hand-oracle tests for the fixed DSL: all 256 states, exact-function padding, candidate novelty, and budget/accounting predicates.
5. Run tiny QA on one endpoint and fixed states. Stop on any digest, path, state, or accounting mismatch.
6. Run the CPU migration gate: one pre-registered L7 program on all 256 states for each of the 12 A/B endpoints. Compare decoded traces and targets exactly with the attached historical predictions. A mismatch blocks science.
7. Run the same migration gate on CUDA. Record CPU/CUDA decoded equality and numeric logit differences separately; do not claim bitwise CPU/GPU equivalence.
8. After both gates pass, freeze the protocol/source/checkpoint manifest and run one detached CUDA science process for the registered L12/L16/L24/L32 padding and new-composition suites. No training, retries, sweeps, or extra seeds.
9. Write raw predictions/traces, accounting, runtime lineage, and one independent review to the new PC-only output directory. Never overwrite Mac results.

## Stop conditions

Stop before science if any archive or manifest hash fails, an expected relative path is missing, the Windows adapter changes a frozen source unexpectedly, CPU migration differs, CUDA decoded traces differ, candidate search cannot produce the registered six compositions at a length, or the forward/native-step budget changes. Report the exact failing path and digest.

## Scope

The result answers only whether these finite h128 checkpoints execute the registered finite DSL suites at lengths 12, 16, 24, and 32 under this protocol. It does not establish a general recurrent mechanism or arbitrary-chain generalization.
