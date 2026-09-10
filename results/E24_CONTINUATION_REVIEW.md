# E24 independent result review — ACCEPT

Independent Astra/low review, 2026-09-07. The registered fixed u8000→u16000 continuation is accepted. All three seeds satisfy the six-program primary conjunction and seen prerequisite, hence combined success. The separate strict error-reduction criterion is false because seed0 retains 14 primary errors.

## Independent checks performed

- Strictly loaded all three final checkpoints against canonical frozen preflight. Full model/AdamW/RNG, optimizer step 16000, parent provenance, source/config/manifest metadata and counters pass the accepted loader. Current RNG equals stored checkpoint RNG. Parent file SHA and all 231 protected hashes remain unchanged.
- Replayed each final exactly once across all seen train/validation programs and all six primary programs plus equivalent control. Every complete evaluation dictionary equals the original report, including every intermediate predicted/target trace, correctness flag, metric, state identity, stratum and inherited predicate. Model/optimizer/RNG digest and training mode remain unchanged by evaluation.
- Independently implemented ADD modulo 16, XOR and SWAP target transitions to recompute every seen and composition target trace. Recomputed final joint/x/y, each instruction readout count and full-trace count from predictions, including train/validation/test composition strata. These match stored metrics.
- Independently joined each parent/current row on identical program/state identity, checked targets and strata, and reconstructed all four paired outcomes, parent/current marginals and metric deltas. Seen metric deltas match separately. Parent reports were read, never reevaluated.
- Reconstructed seen prerequisites, each six-row >=244/256 conjunction, all-seed primary/combined success and the separate strict error reduction predicate. Control is excluded. Verified exactly 32 progress points per seed at 250-update spacing, 8000 attempted/completed updates each, actual training forward counters and final/cumulative checkpoint costs.

## Verified results

| Seed | Six primary final-joint counts, each /256 | Errors u8000→u16000 | Paired wins/losses | Seen / primary / combined |
|---|---|---|---|---|
| 0 | 254,252,255,255,254,252 | 14→14 | 7 / 7 | true / true / true |
| 1 | 256,255,256,256,255,256 | 73→2 | 73 / 2 | true / true / true |
| 2 | 256,256,256,256,256,256 | 50→0 | 50 / 0 | true / true / true |

All 6144 seen train cases and 1024 seen validation cases per seed have correct complete traces. Separate equivalent control: seed0 252→246 (1 win, 7 losses); seed1 246→256 (10 wins, 0 losses); seed2 248→256 (8 wins, 0 losses). Thus the result does not support uniform improvement from more training.

## Artifact identities

| Artifact | SHA256 |
|---|---|
| runs/e24_continuation/report.json | c260959ebb7ff5eb2055c66fa5e1e9353d9f88f8534593f24a30e26e3b99d202 |
| runs/e24_continuation_preflight/manifest.json | d1f9c34d1678ad099a36e757f5c73c10750d9dd807b1a3601f693da826e7095d |
| runs/e24_continuation/seed0/u16000.pt | bafe42f563737283053b32d6b3950f655fb2f05d91793a6539f7b737c81a4cc3 |
| runs/e24_continuation/seed1/u16000.pt | 6191151d76302b490e9471fa11acc919141011b429898c3fe3e0dba509f8b867 |
| runs/e24_continuation/seed2/u16000.pt | 621b51012b01f1a0b0166e8c74d957e3ed783c8d9112d793f145c909e1333f59 |

Reviewed source identities remain those in `results/E24_CONTINUATION_CODE_REVIEW.md`. Independent machine-readable replay summary was retained at `/private/tmp/e24-independent-result-review.json`; this review records its durable essential findings and artifact hashes.

## Work accounting and interpretation

Original scientific work verified: 24000 additional updates, 1536000 examples, 24576000 training internal substeps; 213 evaluation forwards, 26880 cases, 70464 readouts, 563712 evaluation internal substeps. Each seed checkpoint separately records 8000 added versus 16000 cumulative updates, avoiding double counting historical training.

This independent result review added **zero training updates**, exactly **213 evaluation forwards, 26880 cases, 70464 readouts, 563712 internal substeps**. No failed replay or retry occurred. Metadata/direct-count checks added zero model work. Code-review QA and executor repair costs remain separate in `results/E24_ATTEMPT_ACCOUNTING.json`.

The result is an adaptive additional-training diagnosis on already opened E21 data with three initializations and one fixed training stream. It is not a new holdout, an independent replication, a QAT result or evidence of unrestricted reasoning/length generalization. E22 remains the valid negative result at its original budget. The human report's counts and bounded conclusions agree with this review; its stratum columns should explicitly identify final-joint counts, distinct from full-trace counts.

No blocking findings. Registered E24 stop condition is satisfied; no further continuation, seed replacement, tuning or experiment is authorized by this acceptance.
