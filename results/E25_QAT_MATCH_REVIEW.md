# E25 independent result review — ACCEPT valid negative

Independent Astra/low review, 2026-09-07. The fixed three-seed QAT experiment is accepted as technically valid. **The registered task-level retention criterion fails:** none of the three QAT seeds satisfies the combined seen-plus-primary conjunction, whereas the matched accepted E24 float seeds all satisfy it. This is a negative result for this fixed quantization package and training recipe, not a technical failure or a universal claim about QAT.

## Verified results

| Seed | QAT six primary correct counts, each /256 | Float→QAT primary errors /1536 | Seen | Primary | Combined | Primary QAT wins/losses |
|---|---|---|---|---|---|---|
| 0 | 248,235,249,251,246,243 | 14→64 | true | false | false | 14 / 64 |
| 1 | 252,244,251,247,245,249 | 2→48 | false | true | false | 2 / 48 |
| 2 | 252,249,246,245,247,248 | 0→49 | false | true | false | 0 / 49 |

All-seed primary=false; all-seed combined=false. Wins mean QAT correct/float wrong, losses mean QAT wrong/float correct on identical program/state cases. Seed0 primary failures are ADD→ADD→XOR at 235/256 and SWAP→ADD→XOR at 243/256. Seeds1/2 primary counts remain fully reported as diagnostics despite their failed seen prerequisites.

Seed1 has seven failing seen validation compositions: ADD→SWAP29, ADD→ADD→ADD30, ADD→SWAP→ADD30, XOR→ADD→ADD29, XOR→XOR→ADD30, XOR→XOR→SWAP30, SWAP→ADD→SWAP29, each out of 32 against the fixed 31 threshold. Seed2 has four: SWAP→ADD→ADD29, SWAP→XOR→ADD29, SWAP→SWAP→ADD30, SWAP→SWAP→XOR29. Seed0 has no seen prerequisite failure.

Separate equivalent control counts are float→QAT 246→250,256→251,256→249 for seeds0/1/2; paired wins/losses 10/6,0/5,0/7. Control remains excluded from primary success.

## Independent verification

The initial phase inspected only saved seed0/1 artifacts while seed2 trained, with zero model work. After completion, seed2 received the same direct-count checks and all three final models were replayed once.

- Canonical run/preflight manifests match. Strict checkpoint loading verifies every QAT model, all fourteen exact BitLinear locations, FP32 inventory, model/optimizer/RNG digests, AdamW configuration/moments/step 16000, training mode, manifest-bound initial provenance and training counters. Checkpoint initial RNG agrees with saved preflight RNG and the independently corrected actual float training RNG reference: bare manual_seed(0) for seed0, E22 post-constructor RNG for seeds1/2. Final RNG is restored after construction.
- Replayed all seen train/validation programs plus all six primary programs and the separate control for every seed. Entire evaluation dictionaries exactly equal saved results, including every predicted and target instruction trace, correctness flag, state/stratum, metric and predicate. All output tensors were finite. Evaluation preserved model/AdamW/RNG digests and training mode. CPU4 deterministic execution verified.
- Independently implemented the ADD modulo 16, XOR and SWAP transitions to reconstruct every target trace. Recomputed final joint/x/y, instruction readout accuracy and complete-trace accuracy from saved predictions, including all composition train/validation/test strata. Reconstructed each seen prerequisite and six-row>=244/256 conjunction.
- Independently joined QAT and frozen float results by program/state identity, verified target and stratum identity, reconstructed all four paired outcomes, marginals, integer metric deltas and final-joint rate deltas for seen/primary/control results. Float models were never retrained or reevaluated.
- Verified 16000 attempted/completed updates per seed and 64 progress points at 250-update spacing. Completed training/evaluation forward costs equal their attempted counters and registered budgets. Final per-seed reports equal their entries in the aggregate report. No scientific retry or incomplete seed is present.
- All 249 protected file hashes and all frozen float/initial reference hashes match before and after final replay. Source identity remains the code-cleared identity in `results/E25_QAT_MATCH_CODE_REVIEW.md`.

## Exact artifact identities

| Artifact | SHA256 |
|---|---|
| runs/e25_qat_match/report.json | fee8f6f89c02ca0a71cd1ab8af9069420c901b3bb5320f10119c505bdc45da8f |
| runs/e25_qat_match_preflight/manifest.json | 139c949dfb2c0ff2459c048a3a8a45f632f2262d2dc597ca65dd672bfd5970b5 |
| runs/e25_qat_match/seed0/u16000.pt | 9cb638bca4427dc0b1b14f4a8fc8fc00087d2e97ed215c87c8904e1e9606c7c0 |
| runs/e25_qat_match/seed1/u16000.pt | ffd823706709715d5f13e8057766ac599b42a674c615f5580f239a5bf0133572 |
| runs/e25_qat_match/seed2/u16000.pt | 22283c930a30fdc8513edc5c70b526405ba7e155fed0b82f27cc8b08f54d1859 |

Machine-readable intermediate evidence: `/private/tmp/e25-independent-counts-01.json`, `/private/tmp/e25-independent-counts-2.json` and `/private/tmp/e25-independent-full-replay.json`; essential findings and hashes are recorded durably here.

## Accounting and limits

Original scientific work: 48000 updates, 3072000 training examples, 6144000 training readouts and 49152000 recurrent training substeps; 213 evaluation forwards, 26880 cases, 70464 readouts and 563712 evaluation internal substeps. Historical float training is not counted again. Scientific reload-identity training is zero.

This result review adds **zero training updates** and exactly **213 evaluation forwards, 26880 cases, 70464 readouts, 563712 internal substeps**. No failed replay or repeat occurred. Direct JSON/DSL/hash/checkpoint inspections add zero model forwards. Earlier executor and code-review QA costs remain separate.

The result compares the existing ternary-weight plus INT8-activation fake-quantization package against matched float masters/data/budget, with three initialization seeds on an opened finite DSL benchmark. It does not isolate weight versus activation quantization, establish a noninferiority margin, measure packed low-bit execution, prove impossibility under other optimization, or establish general reasoning/length transfer.

No blocking findings. The registered experiment and independent-review stop condition are complete. No QAT variant, extra training budget, replacement seed or new experiment follows automatically from this acceptance.
