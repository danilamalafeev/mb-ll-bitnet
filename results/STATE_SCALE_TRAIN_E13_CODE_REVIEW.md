# E13 pre-training code review

## Verdict

The repaired E13 implementation is cleared for the registered corrected run. The previously found ordering, native-validation, and output-overwrite defects are fixed. No training or new experiment was run during this review.

## Verified controls

- `e13_forward` performs the normal recurrent update and four-block selection, captures requested readouts, then applies centered-L2 normalization. At training step 8, `final_logits` is captured before the step-8 reset; step 4 readout is likewise captured before the reset that affects steps 5–8. The same ordering is used by native validation and the 32-step cross-evaluation.
- `centered_l2_normalize` preserves the row mean and centered direction, uses radius 1 and `eps=1e-8`, and leaves the denominator live in the autograd graph. The focused tests cover mean/radius, finite zero-centered behavior, gradient propagation, and double-precision `gradcheck`.
- `evaluate_native` passes the requested arm into `e13_forward`, computes both final and intermediate validation metrics from that arm, restores the model's prior training mode, and selects checkpoints by final validation accuracy followed by final validation loss. The normalized arm therefore does not silently use baseline dynamics for validation or selection.
- `protocol_check` fixes the registered d64/dff256, four-head/four-block, structured-QAT, eight-step, fixed-two-hop, 2,000-update, batch-64 configuration. Training requires the explicit `--protocol-cleared` switch. The loop reaches the fixed budget before emitting a completed arm report; incremental arm reports are written throughout validation checkpoints.
- Each arm is reseeded independently with the same seed before model construction and stream creation. `state_digest` records the initial model state. The stream digest covers the actual collated `input_ids` and final targets for every training batch; the intermediate target is deterministically derived from the same encoded transitions/start. `assert_paired_reports` enforces equal initial digest, input/target digest, suite fingerprint, optimizer metadata, and parameter count for each paired seed.
- Checkpoints are diagnostic-only and tagged with objective, arm/dynamics, normalization schedule, radius, epsilon, seed, config, selection rule, digests, and `resume_supported=false`. Cross-evaluation verifies the checkpoint file hash and rejects mismatched objective, seed, config, dynamics, normalization, radius, epsilon, or ordinary-resume metadata before loading.
- The manifest is written before training and records the B0/B1/N0/N1 cells, radius, schedule, validation gate, budget, and both preregistered predicates. The primary predicate is N1−B0 k8 delta `>=26` for seeds 0/1 and `>=0` for seed 2. The strong predicate is N1 `>=487/512` at k3–k8 for all three seeds and is marked unavailable for a seed0-only stop.
- `novelty_scopes` requires all historical scopes, E12, and the frozen invalid E13 manifest; missing or empty sources fail closed. A read-only smoke through the actual 512-record manifest generator produced 512 unique cycle16 test tables and zero overlap with all eight scopes: historical E07 persisted 3,200, reconstructed E07 1,280, E08 1,024, E09/E10/E11/E12 512 each, and invalid E13 512. The generated fingerprint was `6a687884419e8ee559c117144c760c86b39e146ffd7ddc83bb86b8a233531f95`.
- Corrected output uses `runs/state_scale_train_e13_corrected/` and `results/STATE_SCALE_TRAIN_E13_CORRECTED.md`. Both the output directory and corrected markdown path are guarded against nonempty/existing results, so the frozen invalid `runs/state_scale_train_e13/` and `results/STATE_SCALE_TRAIN_E13.md` are not overwritten.
- `evaluate_cross_cells` evaluates all four preregistered cells, stores all readout predictions and k1–k8 counts, and computes the declared paired comparisons N1−B0, N1−B1, B1−B0, and N1−N0. Test evaluation occurs only after the seed0 extension decision and never controls gating or checkpoint choice.

## Validation

`PYTHONPATH=. .venv/bin/pytest -q tests/test_state_scale_train_e13.py` returned **12 passed** after the final repairs. `python -m compileall -q scripts/state_scale_train_e13.py` also passed. The repository full-suite result was **55 passed** from the coordinator; no training was started and no corrected run artifacts exist yet.

The invalid original seed0 pair remains excluded from conclusions and is used only as a novelty source. The corrected run must still independently review its produced validation histories, checkpoint hashes/tags, paired digests, gate decision, four-cell predictions, counts, and actual budgets before interpreting any result. The strong all-three-seed criterion must not be reported if the seed0 gate stops the run early.
