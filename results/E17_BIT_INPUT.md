# E17 — learned value tables versus four-bit input

Status: fixed paired seed0 pilot completed; independent Astra/low result review **ACCEPT**.

## Finding

Signed four-bit inputs improved both stream-exposed fit and validation generalization in this float core. Primitive validation ADD rose9/32→29/32, XOR24/32→31/32, SWAP remained32/32. Length3 joint macro rose63.57%→92.11% on train and34.67%→80.65% on validation. The original primitive prerequisite32/32 for EVERY operation is still unmet; this diagnostic does not reopen reserved tests or new compositions.

This is one representation-package intervention: separate learned16→64 tables become signed LSB-first bits through separate bias-free4→64 learned projections. Core tensors start bitwise equal, four substeps/data/loss/optimizer/budget are fixed. Input functions and covariance differ; marginal initialization variance alone is matched. Learned has152768 parameters, bits151232 (1536 fewer); no padding parameters. The result is not an isolated capacity-independent effect or proof of a general arithmetic algorithm.

Root compared all32 learned-arm final state tensors to E16 float u2000: bitwise identical. This confirms baseline reproduction, not an independent replication.

## Every seen program — final joint counts

| Program | Learned train /192 | Bits train /192 | Learned validation /32 | Bits validation /32 |
|---|---:|---:|---:|---:|
| ADD | 191 | 192 | 9 | 29 |
| XOR | 192 | 192 | 24 | 31 |
| SWAP | 192 | 192 | 32 | 32 |
| ADD → ADD | 177 | 188 | 9 | 23 |
| ADD → SWAP | 189 | 191 | 6 | 26 |
| XOR → ADD | 160 | 188 | 15 | 30 |
| XOR → XOR | 152 | 192 | 14 | 30 |
| XOR → SWAP | 191 | 192 | 17 | 31 |
| SWAP → ADD | 140 | 192 | 14 | 30 |
| SWAP → XOR | 153 | 188 | 25 | 31 |
| SWAP → SWAP | 184 | 192 | 29 | 32 |
| ADD → ADD → ADD | 98 | 157 | 6 | 16 |
| ADD → ADD → SWAP | 142 | 180 | 2 | 22 |
| ADD → SWAP → ADD | 130 | 181 | 0 | 22 |
| ADD → SWAP → XOR | 89 | 172 | 2 | 21 |
| ADD → SWAP → SWAP | 158 | 189 | 4 | 26 |
| XOR → ADD → ADD | 92 | 168 | 4 | 22 |
| XOR → ADD → SWAP | 127 | 183 | 12 | 29 |
| XOR → XOR → ADD | 114 | 177 | 12 | 24 |
| XOR → XOR → XOR | 135 | 180 | 16 | 24 |
| XOR → XOR → SWAP | 162 | 182 | 20 | 30 |
| XOR → SWAP → ADD | 128 | 186 | 6 | 31 |
| XOR → SWAP → XOR | 116 | 186 | 7 | 31 |
| XOR → SWAP → SWAP | 152 | 188 | 18 | 30 |
| SWAP → ADD → ADD | 73 | 159 | 7 | 21 |
| SWAP → ADD → SWAP | 105 | 183 | 12 | 29 |
| SWAP → XOR → ADD | 96 | 158 | 16 | 25 |
| SWAP → XOR → XOR | 126 | 175 | 18 | 27 |
| SWAP → XOR → SWAP | 108 | 180 | 15 | 27 |
| SWAP → SWAP → ADD | 127 | 174 | 15 | 27 |
| SWAP → SWAP → XOR | 124 | 170 | 18 | 28 |
| SWAP → SWAP → SWAP | 161 | 186 | 23 | 30 |

## Length-balanced macros

Within each length, programs have equal weight; compositions equally weight length2 and length3. Deltas derive from unrounded rates.

| Group | Learned train % | Bits train % | Delta pp | Learned validation % | Bits validation % | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 99.83 | 100.00 | +0.17 | 67.71 | 95.83 | +28.12 |
| 2 | 87.63 | 99.15 | +11.52 | 50.39 | 91.02 | +40.62 |
| 3 | 63.57 | 92.11 | +28.55 | 34.67 | 80.65 | +45.98 |
| primitives | 99.83 | 100.00 | +0.17 | 67.71 | 95.83 | +28.12 |
| seen_compositions | 75.60 | 95.63 | +20.04 | 42.53 | 85.84 | +43.30 |

## Primitive validation paired cases

| Operation | Both correct | Bits only | Learned only | Neither |
|---|---:|---:|---:|---:|
| ADD | 8 | 21 | 1 | 2 |
| XOR | 23 | 8 | 1 | 0 |
| SWAP | 32 | 0 | 0 | 0 |

Machine report retains every predicted/target trace, x/y correctness, prefix joint and full-trace counts, per-arm train-minus-validation gaps, and semantically named per-case paired outcomes for all32 programs and both splits.

## Budget, checks and limits

Exactly two fresh seed0 arms,2000 updates each:4000 updates,256000 examples,2048000 internal training substeps; runner wall time27.044s on CPU4 deterministic float32. No restart, checkpoint selection, extra seed or sweep. Tiny QA controls and reviewer replay are separate verification work.

Fixed evaluation only seen32 x train192/validation32:14336 program-state evaluations,36736 readout positions,146944 recurrent substeps. All6144 train program-state cases were stream-exposed. Reservedtest32, forbidden/new compositions and lengths4/6 were excluded.

Eight targeted tests and full suite98 passed. Independent code review added actual all16 x/y projection checks, legal length2 one-update parity/gradients, trained checkpoint round trips, and tamper controls; these additional controls address the explicit coverage limitations of existing pytest cases. The temporary review script was archived as `results/E17_CODE_REVIEW_CONTROLS.py`. Source/protocol hashes frozen before training; both initial states saved;118 prior files verified before/after.

Validation states informed prior experiments; they are not an untouched independent test. One seed and fixed budget do not establish robustness or identify a sole cause. Train fit is still incomplete for long compositions, and ADD/XOR validation has errors. No original gate criterion was relaxed and no new task was declared solved.

## Artifacts

- Protocol: `results/E17_BIT_INPUT_PROTOCOL.md`.
- Encoding reference: `results/E17_BIT_ENCODING_REFERENCE.json`.
- Protected snapshot: `results/E17_PROTECTED_HASHES.json`.
- Code review: `results/E17_BIT_INPUT_CODE_REVIEW_FINAL.md` (CLEAR); initial HOLD retained.
- Frozen manifest/initial states: `runs/e17_bit_input_preflight/`.
- Run: `runs/e17_bit_input/{manifest,report}.json` and `{learned,bits}_seed0/final_u2000.pt`.
- Result review: `results/E17_BIT_INPUT_REVIEW.md` (ACCEPT).

Registered stop: this pair plus independent review. Next candidate per the agreed sequence is a separately registered4-versus8-substep control on the float bit-input core. It has not been implemented or run.

Independent review reloaded both arms and reproduced all128 rows, predictions, prefix/full-trace metrics, macros/gaps/deltas and paired outcomes, including a separate integer DSL/count recounter. Initial encoder references, data/coverage/provenance, both8-point progress histories and all118 protected hashes passed. Learned final tensors independently confirmed bitwise E16 float reproduction.
