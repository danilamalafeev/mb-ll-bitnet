# E16 — same-architecture float versus QAT

Status: paired seed0 pilot completed; independent Astra/low result review **ACCEPT**.

## Finding

Removing both weight and activation fake quantization improved fit and validation accuracy in this fixed architecture and budget. Length3 stream-exposed train final-joint macro rose from32.14% QAT to63.57% float (+31.42 percentage points). Validation primitives improved from ADD4/32 to9/32 and XOR15/32 to24/32; SWAP remained32/32. Float still has major errors on exposed compositions and on novel initial-state pairs. This is a local effect of the joint intervention, not evidence that quantization is the only limitation.

Both arms used bitwise-identical initial masters,152768 parameters, the same128000-example stream, loss and four substeps/instruction, and fixed update2000. No validation checkpoint selection. All14 BitLinear projections were replaced in float, including output heads. No GRU, representation or depth change.

Root independently compared all32 E16 QAT final state tensors with E15 selectedu2000: bitwise identical. This confirms baseline reproduction, not an independent seed replication.

## Every seen program: final joint correctness

| Program | QAT train /192 | Float train /192 | QAT validation /32 | Float validation /32 |
|---|---:|---:|---:|---:|
| ADD | 183 | 191 | 4 | 9 |
| XOR | 186 | 192 | 15 | 24 |
| SWAP | 192 | 192 | 32 | 32 |
| ADD → ADD | 127 | 177 | 3 | 9 |
| ADD → SWAP | 154 | 189 | 1 | 6 |
| XOR → ADD | 113 | 160 | 11 | 15 |
| XOR → XOR | 144 | 152 | 18 | 14 |
| XOR → SWAP | 162 | 191 | 7 | 17 |
| SWAP → ADD | 106 | 140 | 8 | 14 |
| SWAP → XOR | 126 | 153 | 15 | 25 |
| SWAP → SWAP | 141 | 184 | 24 | 29 |
| ADD → ADD → ADD | 42 | 98 | 4 | 6 |
| ADD → ADD → SWAP | 51 | 142 | 4 | 2 |
| ADD → SWAP → ADD | 54 | 130 | 0 | 0 |
| ADD → SWAP → XOR | 40 | 89 | 0 | 2 |
| ADD → SWAP → SWAP | 81 | 158 | 4 | 4 |
| XOR → ADD → ADD | 50 | 92 | 3 | 4 |
| XOR → ADD → SWAP | 53 | 127 | 6 | 12 |
| XOR → XOR → ADD | 87 | 114 | 7 | 12 |
| XOR → XOR → XOR | 124 | 135 | 10 | 16 |
| XOR → XOR → SWAP | 112 | 162 | 15 | 20 |
| XOR → SWAP → ADD | 51 | 128 | 2 | 6 |
| XOR → SWAP → XOR | 60 | 116 | 2 | 7 |
| XOR → SWAP → SWAP | 97 | 152 | 12 | 18 |
| SWAP → ADD → ADD | 26 | 73 | 1 | 7 |
| SWAP → ADD → SWAP | 28 | 105 | 4 | 12 |
| SWAP → XOR → ADD | 27 | 96 | 3 | 16 |
| SWAP → XOR → XOR | 28 | 126 | 4 | 18 |
| SWAP → XOR → SWAP | 38 | 108 | 5 | 15 |
| SWAP → SWAP → ADD | 68 | 127 | 5 | 15 |
| SWAP → SWAP → XOR | 79 | 124 | 10 | 18 |
| SWAP → SWAP → SWAP | 100 | 161 | 15 | 23 |

## Length-balanced macros

Rates average programs within each length. Compositions equally weight length2 and length3; primitives equal length1. Deltas use unrounded rates.

| Group | QAT train % | Float train % | Float−QAT pp | QAT validation % | Float validation % | Float−QAT pp |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 97.40 | 99.83 | +2.43 | 53.12 | 67.71 | +14.58 |
| 2 | 69.86 | 87.63 | +17.77 | 33.98 | 50.39 | +16.41 |
| 3 | 32.14 | 63.57 | +31.42 | 17.26 | 34.67 | +17.41 |
| primitives | 97.40 | 99.83 | +2.43 | 53.12 | 67.71 | +14.58 |
| seen_compositions | 51.00 | 75.60 | +24.60 | 25.62 | 42.53 | +16.91 |

Train-minus-validation gaps remain substantial. For ADD: QAT82.81pp, float71.35pp; for XOR: QAT50.00pp, float25.00pp. These are descriptive gaps, not unbiased independent-test estimates. Full per-register, prefix, full-trace metrics and gaps plus true paired case outcomes are preserved in machine report.

## Execution, verification and limits

Exactly2 arms,4000 optimizer updates,256000 examples,2048000 internal training state updates. CPU4, deterministic; runner wall time49.519s. No restart or additional seed. Tiny code-clearance controls and independent evaluation replay are separate verification compute.

Final inference:14336 program/state evaluations on seen32 x train192/validation32 only;36736 readout positions and146944 recurrent substeps. All6144 train program/state pairs were stream-exposed. Reservedtest32, forbidden primary programs and secondary lengths4/6 were excluded.

Pre-run independent review CLEAR after bounded repairs;8 E16 tests passed and full suite90 passed. Source/protocol hashes frozen before training. Each u2000 checkpoint was saved before inference.101 protected files verified before/after. Initial, stream, target, source, config and checkpoint provenance are recorded. Both arms have all8 planned progress observations.

One seed and one fixed budget. Both weight and activation fake quantization changed together, during training and inference; their individual contributions are not identified. Effective representational capacity differs despite equal parameter counts. Validation states informed prior research, although E16 did not use them for checkpoint selection. No significance claim, primary success claim or claim about general recurrent reasoning. The remaining float errors do not identify a particular alternative cause.

## Artifacts

- Registered protocol: `results/E16_FLOAT_QAT_PROTOCOL.md`.
- Protected snapshot: `results/E16_PROTECTED_HASHES.json`.
- Final code review: `results/E16_FLOAT_QAT_CODE_REVIEW_FINAL.md` (CLEAR); initial HOLD preserved separately.
- Frozen preflight/common initial state: `runs/e16_float_qat_preflight/`.
- Run manifest/report: `runs/e16_float_qat/{manifest,report}.json`.
- Fixed checkpoints: `runs/e16_float_qat/{qat,float}_seed0/final_u2000.pt`.
- Independent result review: `results/E16_FLOAT_QAT_REVIEW.md` (ACCEPT).

The preregistered stop is after this paired pilot and review. No representation, step-budget experiment or extra seed has been launched.

Independent review reloaded both models, reproduced all128 model/program/split rows including predictions, prefixes, paired outcomes and all macros/deltas; checked stream/target/frequency digests, both8-point progress histories,101 protected files, and exact E15 QAT reproduction.

Next candidate per the accepted order: separately register learned-value embeddings versus explicit4-bit input on the float core with unchanged four substeps and budget. This controls representation while avoiding another simultaneous quantization change. It has not been implemented or run.
