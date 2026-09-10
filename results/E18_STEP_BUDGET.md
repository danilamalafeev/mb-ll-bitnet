# E18 — native 4 versus 8 internal steps

Status: ACCEPT — independent Astra/low result review (`results/E18_STEP_BUDGET_REVIEW.md`). One fresh paired seed0 pilot on the E17 float core with signed 4-bit inputs. Both arms have151232 parameters and identical initial tensors, training stream, losses and2000 AdamW updates. Native8 reuses the same four blocks twice per instruction; it has double internal work and no readout or auxiliary loss at inner step4.

In this pilot,8 steps substantially worsened composition accuracy on both train and validation. This does not isolate the cause: longer training trajectories can change optimization and state dynamics. It is not evidence that extra recurrent computation is universally harmful.

## Final joint accuracy

| Program length | Train4 | Train8 | Validation4 | Validation8 | Validation delta8−4 |
|---|---:|---:|---:|---:|---:|
| 1 | 100.000% | 100.000% | 95.833% | 93.750% | -2.083pp |
| 2 | 99.154% | 82.878% | 91.016% | 67.969% | -23.047pp |
| 3 | 92.113% | 41.344% | 80.655% | 29.315% | -51.339pp |

Rates are macro means across3,8 and21 previously seen programs. Train has192 initial states per program; validation has32. Validation has been used in previous experiments and is not a fresh held-out test.

## Primitive counts and paired comparison

| Program | Train4/192 | Train8/192 | Validation4/32 | Validation8/32 | Both right | Only8 right | Only4 right | Neither |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADD | 192 | 192 | 29 | 26 | 25 | 1 | 4 | 2 |
| XOR | 192 | 192 | 31 | 32 | 31 | 1 | 0 | 0 |
| SWAP | 192 | 192 | 32 | 32 | 32 | 0 | 0 | 0 |

The original perfect primitive prerequisite remains unmet: ADD29/32 with4 steps and26/32 with8. No reserved test states, new compositions or longer programs were evaluated. No extra seed or budget was selected after observing this result.

## All32 seen programs

| Program | Train4/192 | Train8/192 | Validation4/32 | Validation8/32 |
|---|---:|---:|---:|---:|
| ADD | 192 | 192 | 29 | 26 |
| XOR | 192 | 192 | 31 | 32 |
| SWAP | 192 | 192 | 32 | 32 |
| ADD → ADD | 188 | 160 | 23 | 15 |
| ADD → SWAP | 191 | 180 | 26 | 22 |
| XOR → ADD | 188 | 136 | 30 | 24 |
| XOR → XOR | 192 | 153 | 30 | 19 |
| XOR → SWAP | 192 | 179 | 31 | 28 |
| SWAP → ADD | 192 | 147 | 30 | 19 |
| SWAP → XOR | 188 | 164 | 31 | 22 |
| SWAP → SWAP | 192 | 154 | 32 | 25 |
| ADD → ADD → ADD | 157 | 45 | 16 | 5 |
| ADD → ADD → SWAP | 180 | 95 | 22 | 6 |
| ADD → SWAP → ADD | 181 | 96 | 22 | 9 |
| ADD → SWAP → XOR | 172 | 52 | 21 | 5 |
| ADD → SWAP → SWAP | 189 | 87 | 26 | 12 |
| XOR → ADD → ADD | 168 | 71 | 22 | 8 |
| XOR → ADD → SWAP | 183 | 78 | 29 | 12 |
| XOR → XOR → ADD | 177 | 79 | 24 | 9 |
| XOR → XOR → XOR | 180 | 87 | 24 | 9 |
| XOR → XOR → SWAP | 182 | 87 | 30 | 17 |
| XOR → SWAP → ADD | 186 | 97 | 31 | 12 |
| XOR → SWAP → XOR | 186 | 82 | 31 | 10 |
| XOR → SWAP → SWAP | 188 | 103 | 30 | 15 |
| SWAP → ADD → ADD | 159 | 65 | 21 | 6 |
| SWAP → ADD → SWAP | 183 | 54 | 29 | 4 |
| SWAP → XOR → ADD | 158 | 70 | 25 | 11 |
| SWAP → XOR → XOR | 175 | 74 | 27 | 9 |
| SWAP → XOR → SWAP | 180 | 83 | 27 | 7 |
| SWAP → SWAP → ADD | 174 | 104 | 27 | 11 |
| SWAP → SWAP → XOR | 170 | 79 | 28 | 8 |
| SWAP → SWAP → SWAP | 186 | 79 | 30 | 12 |

## Provenance, cost and failure accounting

- Canonical completed evidence: `runs/e18_step_budget_repaired/report.json`, both `steps4_seed0/final_u2000.pt` and `steps8_seed0/final_u2000.pt`; fresh preflight `runs/e18_step_budget_repaired_preflight/`.
- Protocol: `results/E18_STEP_BUDGET_PROTOCOL.md`; prospective technical restart exception: `results/E18_TECHNICAL_RESTART_AMENDMENT.md`.
- Successful pair:4000 updates,256000 examples,3072000 internal state updates (1024000 for4;2048000 for8). Final evaluation:14336 program-state cases,36736 instruction readouts,220416 internal state updates.
- Completed runner duration including final evaluation: 42.8897s on local CPU.
- First attempt failed after one optimizer update due to missing PROGRESS_INTERVAL import, before any checkpoint or scientific evaluation. It is preserved at `runs/e18_step_budget/`; original preflight and source copies remain. Accounting: `results/E18_FIRST_ATTEMPT_ACCOUNTING.json`; postmortem: `results/E18_STEP_BUDGET_TECHNICAL_FAILURE.md`.
- Total scientific training including failure:4001 updates,256064 examples,3072256 internal state updates. The first update is wasted work, not replication. Repair smoke tests additionally used6 tiny QA updates and64 internal state updates; other unit/review controls are separate QA, not scientific training.
- Original8 focused tests and106 full tests passed but failed to exercise the actual runner control-flow defect. Astra repaired only the import, then directly tested train_arm for both budgets. Do not present the original CLEAR as sufficient validation of the unpatched runner.

Independent review reproduced all128 metric rows, predictions and DSL targets, paired counts and macros, both checkpoint reloads and provenance. All32 native4 final tensors exactly reproduce E17 bits. All138 protected files and first-attempt artifact hashes are preserved. Root rechecked the138 protected hashes after review. No further experiment is started by this report.
