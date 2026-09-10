# E20 — longer native8 training

Status: ACCEPT — independent Astra/low result review (`results/E20_LONGER_NATIVE8_REVIEW.md`). One fresh seed0 native8 run to the prospectively fixed8000 updates. The original2000-batch E18 stream repeats four times; initial state, model, loss and AdamW configuration are unchanged. One optimizer persists throughout. No outcome selected the8000 budget.

More training substantially improves this model: length3 train reaches100% and validation99.851% at8000, compared with41.344% and29.315% at2000. The mandatory u2000 weights exactly reproduce E18 before any later update. This supports a local undertraining explanation for the E18/u2000 result; it does not isolate the optimization mechanism or establish a fair architecture ranking.

## Learning trajectory

| Updates | Train L1 | Train L2 | Train L3 | Validation L1 | Validation L2 | Validation L3 |
|---|---:|---:|---:|---:|---:|---:|
| 2000 | 100.000% | 82.878% | 41.344% | 93.750% | 67.969% | 29.315% |
| 4000 | 100.000% | 99.870% | 97.520% | 98.958% | 95.703% | 89.881% |
| 8000 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 99.851% |

Final-joint macro means across3,8,21 previously seen programs. Each program has192 train and32 validation initial states. Validation states were used in prior experiments and are not a fresh reserved test.

## Primitive validation

| Updates | ADD /32 | XOR /32 | SWAP /32 |
|---|---:|---:|---:|
| 2000 | 26 | 32 | 32 |
| 4000 | 31 | 32 | 32 |
| 8000 | 32 | 32 | 32 |

At8000 all6144 train program-state cases are correct, including their full traces. Validation has1023/1024 correct full traces/final answers: all length1/2 cases and671/672 length3 cases. The one error is ADD→ADD→SWAP from(7,7): correct trace(14,7)→(5,7)→(7,5); predicted final(7,3) after correct first two steps.

The numerical single-model primitive32/32 and each seen-composition≥31/32 prerequisite is met. This does not retroactively pass E15’s original paired QAT/GRU gate or authorize reserved evaluations. No reserved test states, unseen compositions or longer programs were opened.

## Frozen native4 reference and interpretation

Native4/u2000 from E18 had length3 train92.113% and validation80.655%. Native8 exceeds both already at4000 and reaches100%/99.851% at8000. The8000 catch-up predicates are true for both train and validation using unrounded metrics. Native8/u8000 uses8 times the internal training work of native4/u2000 (8192000 versus1024000 substeps); training updates differ4 times, and native inference also costs twice per instruction. There is no long native4 control, so this is a conditional catch-up comparison, not evidence that native8 is the best allocation of equal compute.

## All32 program counts

| Program | Train2000 /192 | Train4000 /192 | Train8000 /192 | Val2000 /32 | Val4000 /32 | Val8000 /32 |
|---|---:|---:|---:|---:|---:|---:|
| ADD | 192 | 192 | 192 | 26 | 31 | 32 |
| XOR | 192 | 192 | 192 | 32 | 32 | 32 |
| SWAP | 192 | 192 | 192 | 32 | 32 | 32 |
| ADD → ADD | 160 | 192 | 192 | 15 | 30 | 32 |
| ADD → SWAP | 180 | 192 | 192 | 22 | 28 | 32 |
| XOR → ADD | 136 | 192 | 192 | 24 | 31 | 32 |
| XOR → XOR | 153 | 192 | 192 | 19 | 31 | 32 |
| XOR → SWAP | 179 | 192 | 192 | 28 | 32 | 32 |
| SWAP → ADD | 147 | 191 | 192 | 19 | 30 | 32 |
| SWAP → XOR | 164 | 191 | 192 | 22 | 31 | 32 |
| SWAP → SWAP | 154 | 192 | 192 | 25 | 32 | 32 |
| ADD → ADD → ADD | 45 | 183 | 192 | 5 | 27 | 32 |
| ADD → ADD → SWAP | 95 | 187 | 192 | 6 | 27 | 31 |
| ADD → SWAP → ADD | 96 | 187 | 192 | 9 | 25 | 32 |
| ADD → SWAP → XOR | 52 | 187 | 192 | 5 | 27 | 32 |
| ADD → SWAP → SWAP | 87 | 190 | 192 | 12 | 28 | 32 |
| XOR → ADD → ADD | 71 | 188 | 192 | 8 | 26 | 32 |
| XOR → ADD → SWAP | 78 | 189 | 192 | 12 | 29 | 32 |
| XOR → XOR → ADD | 79 | 189 | 192 | 9 | 29 | 32 |
| XOR → XOR → XOR | 87 | 190 | 192 | 9 | 28 | 32 |
| XOR → XOR → SWAP | 87 | 192 | 192 | 17 | 31 | 32 |
| XOR → SWAP → ADD | 97 | 191 | 192 | 12 | 32 | 32 |
| XOR → SWAP → XOR | 82 | 191 | 192 | 10 | 32 | 32 |
| XOR → SWAP → SWAP | 103 | 192 | 192 | 15 | 32 | 32 |
| SWAP → ADD → ADD | 65 | 174 | 192 | 6 | 26 | 32 |
| SWAP → ADD → SWAP | 54 | 183 | 192 | 4 | 28 | 32 |
| SWAP → XOR → ADD | 70 | 179 | 192 | 11 | 25 | 32 |
| SWAP → XOR → XOR | 74 | 185 | 192 | 9 | 29 | 32 |
| SWAP → XOR → SWAP | 83 | 187 | 192 | 7 | 31 | 32 |
| SWAP → SWAP → ADD | 104 | 191 | 192 | 11 | 30 | 32 |
| SWAP → SWAP → XOR | 79 | 190 | 192 | 8 | 31 | 32 |
| SWAP → SWAP → SWAP | 79 | 187 | 192 | 12 | 31 | 32 |

## Artifacts, validation and accounting

- Protocol `results/E20_LONGER_NATIVE8_PROTOCOL.md`; code clearance `results/E20_LONGER_NATIVE8_CODE_REVIEW.md`.
- Run `runs/e20_longer_native8/`: manifest, progress, report, u2000.pt/u4000.pt/u8000.pt. Fresh preflight `runs/e20_longer_native8_preflight/`. Milestones save complete model, AdamW and RNG state with hashes; the old E18 checkpoint could not resume AdamW, hence fresh prefix reconstruction.
- Exactly8000 scientific updates,512000 examples and8192000 recurrent training substeps, including the reproduced2000 prefix. No scientific failure or restart. Work beyond the prefix:6000 updates,384000 examples,6144000 substeps.
- Three scientific evaluation passes:21504 program-state cases,55104 readout positions,440832 internal substeps. Independent reviewer replay is additional verification, not hidden in these counts.
- Measured CLI wall time115.82seconds (/usr/bin/time, includes process startup, validation, training, checkpointing and evaluation).
- Executor focused QA:24 tiny updates,48 examples,768 recurrent substeps across two invocations; one initial preflight test failed during implementation and was corrected. Independent code review QA:14 tiny updates,26 examples,416 substeps. Combined known E20 pretraining QA:38 updates,74 examples,1184 substeps; separate from scientific work.
- Three focused tests passed on stable sources; they exercise the actual runner, optimizer/RNG continuity, next-update round trip, prefix mismatch stopping and recovery. Independent supplemental controls exercise scientific comparisons/reporting and existing-milestone recovery. No old full suite repeated for this isolated change.
- Root verified all176 protected historical file hashes unchanged after completion. E18’s separate failed/repaired attempt accounting remains in its own report.

Independent result review strict-loaded all three milestones and reproduced all192 rows with direct DSL/count checks. Unrounded macros/deltas/catch-up, all32 AdamW states at each milestone, RNG digests, exact E18/u2000 tensor identity and all176 protected hashes passed. Root rechecked protected files after review. Registered stop reached. E19 balanced-exposure protocol was designed independently; it has not been implemented or trained, and its settings have not been adapted to E20 outcomes.
