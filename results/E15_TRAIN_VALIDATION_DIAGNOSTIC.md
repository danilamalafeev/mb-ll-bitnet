# E15 — frozen train/validation diagnostic

Status: completed inference-only run; independent Sol/high result review **ACCEPT**.

## Finding

Both models reproduce the primitive arithmetic much better on stream-exposed initial register pairs than on the validation pairs. GRU has192/192 train joint accuracy for each primitive; QAT has183/192 ADD,186/192 XOR and192/192 SWAP. Their old validation counts replayed exactly. This supports a descriptive train/validation generalization gap for ADD/XOR in these fixed checkpoints, rather than a claim that neither model learned any training examples.

QAT also has substantial errors on stream-exposed multi-instruction programs: its length3 train joint macro is about32.14%, versus91.89% for GRU. Thus the QAT result is not explained solely by a gap to unseen initial states. Low train performance alone cannot identify optimization, representation, objective, capacity or quantization as the cause.

## Every seen program — final joint correctness

Train denominators192, validation denominators32. All6144 program/train-state combinations were actually exposed in the reconstructed optimizer stream.

| Program | QAT train | QAT validation | GRU train | GRU validation |
|---|---:|---:|---:|---:|
| ADD | 183/192 | 4/32 | 192/192 | 5/32 |
| XOR | 186/192 | 15/32 | 192/192 | 15/32 |
| SWAP | 192/192 | 32/32 | 192/192 | 32/32 |
| ADD → ADD | 127/192 | 3/32 | 192/192 | 10/32 |
| ADD → SWAP | 154/192 | 1/32 | 192/192 | 3/32 |
| XOR → ADD | 113/192 | 11/32 | 192/192 | 19/32 |
| XOR → XOR | 144/192 | 18/32 | 192/192 | 32/32 |
| XOR → SWAP | 162/192 | 7/32 | 192/192 | 12/32 |
| SWAP → ADD | 106/192 | 8/32 | 192/192 | 11/32 |
| SWAP → XOR | 126/192 | 15/32 | 192/192 | 15/32 |
| SWAP → SWAP | 141/192 | 24/32 | 192/192 | 32/32 |
| ADD → ADD → ADD | 42/192 | 4/32 | 115/192 | 5/32 |
| ADD → ADD → SWAP | 51/192 | 4/32 | 185/192 | 7/32 |
| ADD → SWAP → ADD | 54/192 | 0/32 | 175/192 | 3/32 |
| ADD → SWAP → XOR | 40/192 | 0/32 | 170/192 | 4/32 |
| ADD → SWAP → SWAP | 81/192 | 4/32 | 187/192 | 5/32 |
| XOR → ADD → ADD | 50/192 | 3/32 | 131/192 | 7/32 |
| XOR → ADD → SWAP | 53/192 | 6/32 | 185/192 | 16/32 |
| XOR → XOR → ADD | 87/192 | 7/32 | 174/192 | 12/32 |
| XOR → XOR → XOR | 124/192 | 10/32 | 186/192 | 10/32 |
| XOR → XOR → SWAP | 112/192 | 15/32 | 187/192 | 30/32 |
| XOR → SWAP → ADD | 51/192 | 2/32 | 188/192 | 4/32 |
| XOR → SWAP → XOR | 60/192 | 2/32 | 186/192 | 8/32 |
| XOR → SWAP → SWAP | 97/192 | 12/32 | 192/192 | 6/32 |
| SWAP → ADD → ADD | 26/192 | 1/32 | 150/192 | 7/32 |
| SWAP → ADD → SWAP | 28/192 | 4/32 | 189/192 | 7/32 |
| SWAP → XOR → ADD | 27/192 | 3/32 | 170/192 | 14/32 |
| SWAP → XOR → XOR | 28/192 | 4/32 | 178/192 | 22/32 |
| SWAP → XOR → SWAP | 38/192 | 5/32 | 188/192 | 11/32 |
| SWAP → SWAP → ADD | 68/192 | 5/32 | 186/192 | 10/32 |
| SWAP → SWAP → XOR | 79/192 | 10/32 | 191/192 | 19/32 |
| SWAP → SWAP → SWAP | 100/192 | 15/32 | 192/192 | 28/32 |

## Length and group macros

Arithmetic mean of program rates within each length. Seen-compositions summary equally weights length2 and length3. Gaps are train minus validation in percentage points.

| Model | Group | Train joint % | Validation joint % | Gap pp | Train full trace % | Validation full trace % |
|---|---|---:|---:|---:|---:|---:|
| GRU | length1 | 100.00 | 54.17 | 45.83 | 100.00 | 54.17 |
| GRU | length2 | 100.00 | 52.34 | 47.66 | 100.00 | 37.11 |
| GRU | length3 | 91.89 | 34.97 | 56.92 | 91.89 | 21.58 |
| GRU | primitives | 100.00 | 54.17 | 45.83 | 100.00 | 54.17 |
| GRU | seen_compositions | 95.94 | 43.66 | 52.29 | 95.94 | 29.34 |
| QAT | length1 | 97.40 | 53.12 | 44.27 | 97.40 | 53.12 |
| QAT | length2 | 69.86 | 33.98 | 35.87 | 68.82 | 25.00 |
| QAT | length3 | 32.14 | 17.26 | 14.88 | 25.99 | 7.29 |
| QAT | primitives | 97.40 | 53.12 | 44.27 | 97.40 | 53.12 |
| QAT | seen_compositions | 51.00 | 25.62 | 25.38 | 47.40 | 16.15 |

Machine report preserves each register separately, all prefix joint counts, full-trace counts, rates, descriptive gaps, and exposure summaries for every program.

## Scope, verification and limits

- Exactly the two selected seed0/u2000 checkpoints. CPU4 threads, deterministic algorithms, eval/inference mode; zero optimizer updates.
- Only seen32 programs x train192/validation32. Reserved test32 initial pairs, forbidden primary programs and secondary lengths4/6 were not evaluated.
- Both complete validation replays succeeded before train-split inference.95 protected files were verified before and after inference.
- Reconstructed stream:128000 draws,6144/6144 unique program-state pairs. Each primitive training case appeared48–102 times; length2 cases12–47, length3 cases1–24. Exposure does not imply successful learning.
- One run:14336 full program-state evaluations,36736 readout positions,146944 recurrent substeps; independent reviewer replay is additional verification compute.
-14 focused+legacy tests passed; independent pre-run code review CLEAR. New evaluator/protocol hashes and frozen source/checkpoint provenance are in machine report.
- These are descriptive results for one seed and a fixed2000-update budget. Validation32 also selected the checkpoints, so it is not an independent test estimate. No new success threshold or original E15 gate revision. Differences do not isolate quantization or prove a generally superior architecture.

## Artifacts

- Protocol: `results/E15_TRAIN_VALIDATION_DIAGNOSTIC_PROTOCOL.md`.
- Evaluator: `scripts/e15_train_validation_diagnostic.py`.
- Machine report: `runs/e15_train_validation_diagnostic/report.json`.
- Code clearance: `results/E15_TRAIN_VALIDATION_CODE_REVIEW.md`.
- Protected snapshot: `results/E15_TRAIN_VALIDATION_PROTECTED_HASHES.json`.
- Independent result review: `results/E15_TRAIN_VALIDATION_DIAGNOSTIC_REVIEW.md` (ACCEPT).

No follow-up training has been registered or started. The next design decision should address primitive state generalization and distinguish it from QAT errors on already exposed compositions.

Independent review reloaded both selected models and matched all128 model/program/split rows, integer metrics, macros and gaps. All95 protected files remained unchanged.

Sol recommends a separate preregistered primitive-only GRU control comparing learned value embeddings with explicit4-bit register inputs, holding core/data budget fixed and selecting a predetermined update. This is a candidate only; neither its protocol nor training has been started.
