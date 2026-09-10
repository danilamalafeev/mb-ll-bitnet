# E14: обучение смеси длин — результат

Дата: 2026-09-07. Канонический исправленный run: `runs/length_mixture_e14/`.
Machine-readable report: `runs/length_mixture_e14/report.json`.
Artifact relocation and attempt provenance: `runs/length_mixture_e14/attempt_provenance.json`.

Seed-0 validation gate пройден: все 6 native readouts на 256/256. Поэтому по
зарегистрированному правилу выполнены все три seeds, по 2000 updates × 64.
Schedule точный: h1/h2/h3 = 666/668/666, `sum_h=4000`.

## Test h8@32: mixture against reused E13 baseline

| Seed | Mixture | Baseline | Delta | Primary |
|---|---:|---:|---:|---|
| 0 | 512/512 | 144/512 | +368 | pass |
| 1 | 507/512 | 10/512 | +497 | pass |
| 2 | 512/512 | 446/512 | +66 | pass |

All primary thresholds pass (`+26/+26/+0`). The strong criterion also passes:
mixture final accuracy is at least 487/512 for every h4–h8 cell in every seed.
Mixture final counts for h4..h8 are seed0 `512/512/512/512/512`, seed1
`512/512/512/512/507`, and seed2 `512/512/512/512/512`.

The paired h8 wins/losses against baseline are seed0 `368/0`, seed1 `497/0`,
and seed2 `66/0`; remaining examples are ties. Validation selected update
2000 for all three seeds. The shared validation and test manifests are fresh,
table-disjoint from all required historical scopes; fingerprints are recorded
in `report.json` and `preflight.json`.

Independent post-run checks recomputed all 24 final count cells from saved
predictions, reloaded all three selected checkpoints with provenance guards,
and confirmed checkpoint hashes, preflight hash, and novelty scope counts.

The first attempt is preserved byte-for-byte under
`runs/length_mixture_e14_invalid_first_attempt/`; its hashes are recorded in
`results/LENGTH_MIXTURE_E14_INVALID_FIRST_ATTEMPT_HASHES.json`. All arms in
that attempt trained, but a missing checkpoint `parameters` field correctly
stopped evaluation before test logits. It is not included in the scientific
result above. The canonical directory now contains the repaired run.

This is strong evidence for the registered mixture-training hypothesis on this
structured cycle16 task. It remains a three-seed diagnostic result, not proof
of general LLM or arbitrary-depth reasoning. Independent follow-up artifact review is complete: **ACCEPT**, see
`results/LENGTH_MIXTURE_E14_REVIEW.md`.

## Full attempt accounting

Both attempts trained all three seeds: **12000 total updates, 768000 examples,
6144000 example-steps**. Half belongs to the first metadata-failed attempt.
Selected model tensors are bitwise identical between attempts in all seeds;
the retraining was redundant and is not an independent replication. All 13
first-attempt files match the preserved hash snapshot. Machine evidence:
`results/LENGTH_MIXTURE_E14_ATTEMPT_ACCOUNTING.json`. The coordinator's recount
is a separate calculation, not an independent reviewer; scientific acceptance
is recorded by the external follow-up review.
