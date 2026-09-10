# E15 — register interpreter paired pilot

Status: **GATE FAILURE; registered pilot stopped**. Independent result review: **ACCEPT — valid negative pilot**.

## Question and design

Can a recurrent QAT core compose explicit ADD/XOR/SWAP instructions on two registers modulo16, relative to a similarly sized float GRU? QAT has152768 parameters; GRU has152720. Both receive the current opcode, fixed initial-register context, four recurrent updates per instruction, and intermediate x/y supervision. No router or E14 checkpoint reuse.

Training used seen programs of lengths1/2/3 and192 initial register pairs. Validation used32 different pairs. The six semantically novel held-out programs were reserved for a gated primary test. This run did not reach that test.

## Measured result

| Model, seed0 | Selected update | Validation macro final joint | ADD | XOR | SWAP | Seen compositions meeting31/32 |
|---|---:|---:|---:|---:|---:|---:|
| QAT | 2000 | 0.347904 | 4/32 | 15/32 | 32/32 | 0/29 |
| GRU | 2000 | 0.471602 | 5/32 | 15/32 | 32/32 | 2/29 |

The preregistered gate requires every primitive32/32 and every seen composition>=31/32 in BOTH arms. Both failed. Seeds1/2, primary/secondary prediction evaluation and paired test win/loss counts were therefore not run. Their absence is required by the stopping rule, not a zero-accuracy test result.

Both models correctly predict SWAP on these32 held-out initial pairs. ADD/XOR do not meet the primitive gate. This establishes failure of this fixed configuration and training budget to satisfy the validation prerequisite; it does not distinguish inadequate optimization from weak arithmetic generalization. Training-set accuracy was not measured in this run. The single-seed macro difference does not establish a general GRU advantage, nor does the gate failure refute recurrent reasoning architectures.

## Cost and checks

Exactly2 completed arms,4000 optimizer updates,256000 examples,2048000 internal per-example state updates; runner wall time 52.677s. No restarts, additional seeds, sweeps or paid cloud. CPU4 threads; equal parameter/data/update budgets do not imply equal FLOPs.

Independent pre-run review: `results/E15_REGISTER_CODE_REVIEW_V7.md` CLEAR. Focused9 tests and root full suite77 tests passed. Frozen v7 canonical manifest hash: `bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`. Root rechecked all52 protected pre-existing files after execution: all hashes unchanged.

## Artifacts

- Protocol: `results/E15_REGISTER_IMPLEMENTATION_PROTOCOL.md`.
- Frozen preflight: `runs/register_e15_preflight/v7/manifest.json`.
- Machine report: `runs/register_interpreter_e15/report.json`.
- Each arm: `runs/register_interpreter_e15/{qat,gru}_seed0/arm_complete.json`, selected/latest checkpoints at validation updates250..2000.
- Independent result review: `results/E15_REGISTER_INTERPRETER_REVIEW.md` (ACCEPT).

Next research decision should isolate primitive learning/generalization before another composition experiment. No follow-up training is registered or started.
