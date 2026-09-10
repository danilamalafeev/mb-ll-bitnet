# E27 independent final review — ACCEPT, restoration PASS

2026-09-08. All three fixed scientific finals are valid and pass the unchanged combined gate. No blocking finding or additional training is required.

| Seed | E27 primary correct counts, each /256 | E26 → E27 errors /1536 | Float errors | Seen / primary / combined | Paired wins/losses vs E26 | Vs float |
|---|---|---|---|---|---|---|
| 0 | 256,255,256,256,256,254 | 55 → 3 | 14 | true / true / true | 55 / 3 | 12 / 1 |
| 1 | 256,256,256,256,256,256 | 2 → 0 | 2 | true / true / true | 2 / 0 | 2 / 0 |
| 2 | 256,256,256,256,256,255 | 47 → 1 | 0 | true / true / true | 47 / 1 | 0 / 1 |

All controls are256/256 and remain excluded. Seen train final accuracy is6144/6144 for every seed; validation is1024/1024,1024/1024,1023/1024. All primitives pass32/32 and every seen composition reaches31/32; all primary programs exceed244/256.

## Independent verification

- Reconstructed manifest equals canonical and preflight manifests exactly. All code-cleared source/protocol hashes, exact initialization and actual RNG, stream/target order, costs, config, evaluator scope and comparator references match. All281 protected entries and14 independently frozen comparator-reference hashes remain unchanged.
- Strictly loaded all3 u16000 checkpoints:14 W4 layers,151232 FP32 masters/native8, model/optimizer/RNG digests, mode, provenance, AdamW membership/config and finite moment tensors with step16000. CPU4 deterministic mode checked.
- Replayed all213 evaluation forwards once. Entire output dictionaries match saved evidence exactly, including predictions, targets, traces, flags, states, strata, metrics and conjunctions. Model/optimizer/RNG and training mode remain unchanged; outputs finite.
- Independently recomputed ADD modulo16, XOR and SWAP target traces from raw predictions, final joint/x/y, instruction correctness and full-trace counts. Reconstructed each composition stratum and all paired outcomes/rate deltas against both E26 and E24 using state/program/target identity; scientific metric/paired helpers are not used by this count audit.
- Every per-seed report equals its aggregate entry. Completed/attempted updates16000 per seed and costs match; progress records occur every250 updates through16000. Scientific reload-update cost is zero. Seen/primary/combined predicates and all-three conjunctions recompute exactly, with the control excluded.

Evidence is preserved in `runs/e27_w4_review_final/`: `replay.py`, `replay.json`, `check_counts.py`, both `counts-*.json`, and `QA.json`. Final review adds **zero training updates**, exactly **213 forwards /26880 cases /70464 readouts /563712 recurrent substeps**, one successful replay in7.8869s. No failed replay, rerun or variant. Earlier code-review QA remains separate.

Human report `results/E27_W4.md` and attempt accounting were skimmed: tables, full-trace assertions and bounded interpretation agree with the independently checked evidence. Pending-review wording may now be replaced with ACCEPT. Preparation QA has a disclosed audit limitation: reused first pytest basetemp erased the earliest failed temporary artifacts. Final passing implementation QA, independent QA and canonical scientific artifacts are preserved; this does not invalidate the independently verified scientific finals, but preparatory attempt retention is incomplete.

## Interpretation and stop condition

The W4 absmax/A32 package restores the registered task level across all3 seeds under this fixed recipe. It changes both weight range policy and resolution versus ternary absmean, so this is not a pure bit-count effect. Training and inference both change; rounding and optimization are not separately identified. The result does not show universal superiority to float (seed2 has one error versus zero), packed-inference speed/memory gains, unlimited depth, language reasoning or arbitrary-task transfer. Fixed data seed0, opened E21, three initializations and adaptive prior experiments limit inference.

The fixed scientific run and independent final-review stop condition are satisfied. No new variant or budget is authorized by this review.

## Canonical SHA256

- Report: `85807e2400389cec2df24d189915a40a96354db811fd62717ebdfdde436c723b`
- Manifest: `de481458592a0b0f5cba08f50263153c2189bdaca4c7832d53739b2e2675570d`
- seed0/u16000.pt: `cd90d6ffa614f2dd98c21e6d61ec1498cd514f9e084fb5a3f93aba60c9cd3815`
- seed1/u16000.pt: `dddfa992c97019902083bb54ef48e86c43154d5b33417eed964e5fdd0a35e681`
- seed2/u16000.pt: `2d772c0e9f8b78de02776a64422edf236f679c04d4c34b0f7ed99189ec9c49fe`
