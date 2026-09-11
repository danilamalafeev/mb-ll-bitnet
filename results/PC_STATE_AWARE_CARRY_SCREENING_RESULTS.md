# State-aware carry screening result — 2026-09-11

The one registered bounded screening completed successfully and passed the
independent saved-artifact audit. Ordinary arm **A** and state-aware arm **C**
started from the same accepted local-2000 endpoint and used the first 250
frozen stream batches per arm. The run recorded exactly 816 forwards, 500
updates/backward/optimizer steps, 112,896 cases, 1,692,032 readout positions,
13,536,256 native steps, six checkpoints and six underlying deserializations;
there were no runtime failures.

## Efficacy screen

Arm A remained perfect on the registered final evaluation:

| suite | A full traces | C full traces |
|---|---:|---:|
| padding | 11,520 / 11,520 | 6,310 / 11,520 |
| compositions | 6,144 / 6,144 | 2,805 / 6,144 |
| identity controls | 20,736 / 20,736 | 11,970 / 20,736 |
| all 150 programs | 38,400 / 38,400 | 21,085 / 38,400 |

The paired comparison found zero C repairs and 8,549 regressions on the old
69 programs, plus zero repairs and 8,766 regressions on the 81 identity
controls. C's final joint accuracy was 21,620 / 38,400 (56.30%); A was
38,400 / 38,400 (100%). C's training loss also became unstable (last loss
0.11894; maximum 2.60821 at update 216), while A stayed below 0.00009.

## State-transfer audit

The existing four identity-path pairs (eight forwards per arm, with a common
`ADD` probe) were replayed at each trained endpoint. Mean relative L2 distance
between the two semantically equivalent paths was:

| pair | A | C |
|---|---:|---:|
| SWAP pair vs two SWAP pairs | 0.0217 | 0.1348 |
| XOR pair vs two XOR pairs | 0.0156 | 0.1128 |
| mixed identity vs two cycles | 0.0015 | 0.0969 |
| SWAP pair vs XOR pair | 0.2066 | 0.3387 |

A had 1.0 probe argmax agreement for every pair. C fell to 0.8281, 0.9102,
0.8750 and 0.9492 respectively. Model and adapter digests were preserved by
the inference-only audit for both arms.

## Decision

This is a **negative structural screening**. The state-aware correction did not
reduce state inconsistency and caused broad accuracy regressions under the
registered 250-update schedule. The result does not justify a full 2,000-update
pilot, cycle-specific loss, automatic tuning, or a retry. Preserve the run and
return to the main Astra research decision point for any new hypothesis.

Saved artifacts:

- screening report: `runs/pc_state_aware_carry_v1/screening/report.json`
- exact accounting: `runs/pc_state_aware_carry_v1/screening/accounting.json`
- independent audit: `runs/pc_state_aware_carry_v1/screening/saved_audit.json`
- source/input freeze: `runs/pc_state_aware_carry_v1/screening/source_binding.json`, `input_freeze.json`

The audit was read-only after the run: it performed no model forwards or
optimizer updates.
