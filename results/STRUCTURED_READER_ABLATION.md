# Structured reader ablation — 6 сентября 2026

## Результат

Structured reader не достиг заранее выбранного критерия практического успеха: не менее 95% validation и ID во всех трёх seeds. На fixed two-hop задача осталась близкой к chance на новых таблицах после общего бюджета 2000 updates × batch 64. Это task-specific inductive-bias experiment; результат не опровергает рекуррентные LLM и не доказывает общую reasoning способность.

| Seed | Best update | Validation h=2 | ID h=2 | Secondary h=3 | Train seconds |
|---|---:|---:|---:|---:|---:|
| 0 | 1750 | 21/256 (8,20%) | 35/512 (6,84%) | 32/512 (6,25%) | 30,06 |
| 1 | 2000 | 24/256 (9,38%) | 34/512 (6,64%) | 38/512 (7,42%) | 31,08 |
| 2 | 250 | 23/256 (8,98%) | 43/512 (8,40%) | 36/512 (7,03%) | 30,79 |

Uniform expectation is 6,25%. The validation-selected ID results therefore show no reliable two-hop generalization. Relative to the prior mixed-memory baseline (ID 39/512, 27/512, 26/512), these values are a small noisy movement around chance, not a practical improvement. Best selection used validation only; ID and secondary sets were evaluated after selection.

The bounded one-hop sanity completed first with the same structured branch, seed 0, 2000 updates, batch 64, and eight compute steps: validation 256/256 and in-distribution 512/512. Longer two-hop accuracy was 0/512. This is a partial signal that the branch preserves basic one-hop reading while the fixed2 failure remains specific to composing two transitions.

## Method and comparability

Only `ModelConfig.structured_reader: bool = false` and the reader branch were added. The default remains the legacy mixed-memory path and is bitwise identical for the same seed: `memory_norm(source_embedding + destination_embedding)`, then the existing shared `k_proj` and `v_proj`. The structured path keeps the existing source and destination embeddings, applies the existing shared `memory_norm` separately to each, then calls the same `k_proj(source_rows)` and `v_proj(destination_rows)` on aligned rows. This separate application of the shared LayerNorm is intentional and is the only representation-normalization change; it is a comparability confound relative to the mixed baseline, with no new normalization parameters.

Queries, output projection, FFN blocks, recurrent state, eight compute steps, `persistent_query=false`, QAT, optimizer, data stream, and training budget are unchanged. No target, trajectory, router, gate, curriculum, or additional embedding reaches the reader. Module creation order is unchanged, so seeded initialization is unchanged. Both configurations have 152512 parameters and 31 state-dict keys. Legacy checkpoints without the new config field load as `structured_reader=false`; resuming across modes is rejected by the existing config compatibility check.

The fixed2 suites use the same fingerprint in all new runs:
`0c7c940047a0e04a34b2cbc7d51a18eba877772d689ed5f439f891ea67ad8b02`.
Each seed used 128000 examples and 1024000 example×compute-step visits. Training was run sequentially on CPU in new directories:
`runs/structured_reader_ablation/one_hop_seed0/` and
`runs/structured_reader_ablation/fixed2_seed{0,1,2}/`.

## Targeted verification

The tests cover source-only keys, destination-only values, row locality, aligned edge permutation invariance, one call each to cached K/V projections, repeated FFN visits and finite gradients in both reader modes, default bitwise equivalence, parameter/state-dict counts, config roundtrip and legacy missing-field parsing, structured checkpoint loading, and rejection of cross-mode resume. `.venv/bin/pytest -q` reports **24 passed**. The production default checkpoint `runs/fixed_two_hop/seed0/checkpoint.pt` also loads with the new code as `structured_reader=false` and 152512 parameters.

## Artifacts and limitations

Every new run contains `config.json`, `eval_sets.json`, `metrics.jsonl`, `train_summary.json`, `checkpoint.pt`, `checkpoint_best.pt`, and `evaluation_best.json`; the one-hop run additionally records its sanity evaluation. CPU timings are factual run measurements, not a speed benchmark. No paid cloud, CUDA kernel, mixed/long automatic extension, or curriculum run was used.

The single next test supported by this evidence is a structured-reader tiny32 fixed-two-hop memorization control, matched to the existing tiny32 protocol. It would distinguish failure to fit fixed examples from failure to generalize, without adding another architectural change.
