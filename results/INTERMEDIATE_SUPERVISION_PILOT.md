# Intermediate-node supervision pilot — 6 сентября 2026

## Результат

Auxiliary supervision на state после шага 4 достигла 100% validation accuracy для промежуточного узла `f(start)` и 100% для финального узла `f(f(start))` во всех трёх seed. Поэтому заранее выбранное правило для seed0 было выполнено, и bounded extension seeds1/2 завершена. Это диагностический результат: он показывает, что дополнительный сигнал облегчает обучение на этой задаче, но сам по себе не устанавливает причину прежнего failure и не доказывает общий reasoning.

| Seed | Arm | Best update | Validation final | Validation intermediate | ID final | ID intermediate | Longer final | Longer intermediate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | final-only baseline | 1750 | 21/256 (8,20%) | 14/256 (5,47%) | 35/512 (6,84%) | 37/512 (7,23%) | 32/512 (6,25%) | 40/512 (7,81%) |
| 0 | auxiliary weight 1 | 2000 | 256/256 (100%) | 256/256 (100%) | 512/512 (100%) | 512/512 (100%) | 0/512 (0%) | 512/512 (100%) |
| 1 | final-only baseline | 2000 | 24/256 (9,38%) | 18/256 (7,03%) | 34/512 (6,64%) | 27/512 (5,27%) | 38/512 (7,42%) | 34/512 (6,64%) |
| 1 | auxiliary weight 1 | 2000 | 256/256 (100%) | 256/256 (100%) | 512/512 (100%) | 512/512 (100%) | 0/512 (0%) | 512/512 (100%) |
| 2 | final-only baseline | 250 | 23/256 (8,98%) | 11/256 (4,30%) | 43/512 (8,40%) | 30/512 (5,86%) | 36/512 (7,03%) | 25/512 (4,88%) |
| 2 | auxiliary weight 1 | 2000 | 256/256 (100%) | 256/256 (100%) | 512/512 (100%) | 512/512 (100%) | 0/512 (0%) | 512/512 (100%) |

The final-only rows reuse the matched structured-reader baseline checkpoints. Their intermediate metrics are a post-hoc diagnostic using the same shared output head at step 4; the old training objective had no intermediate loss. Best checkpoints were selected from final validation accuracy, then final validation loss. ID and longer suites were not used for selection.

## Protocol

The model stayed at `d_model=64`, `d_ff=256`, 4 blocks, 4 heads, structured reader, `persistent_query=false`, QAT, and 8 compute steps. Each aux run used 2000 updates × batch 64, the existing cycle two-hop stream (`train_min_hops=train_max_hops=2`), the existing validation/ID/longer suites, and its matched seed 0/1/2 baseline. There were no architecture parameters, teacher forcing, target tokens, predicted-class feedback, or new embeddings. The step-4 readout is a schedule imposed by this diagnostic; it is not a claim that the current recurrent steps already correspond to graph hops.

The loss was `L_final(step8, f(f(start))) + L_intermediate(step4, f(start))`. Intermediate labels were computed from each table only after collation for the loss and held-out diagnostic. `input_ids` and the runtime evaluator remained unchanged. The optional model method returns final logits plus selected shared-head readouts while the default forward output remains unchanged.

The three aux runs had 152512 parameters and matching seed-specific initial-state digests with a freshly reconstructed model (`4b203163…`, `6f0bfeab…`, `445cdbdd…`). All checkpoints are explicitly tagged with objective `aux_intermediate_weight1`, readout step 4, final step 8, and `resume_supported=false` to prevent an ordinary final-only resume from silently changing objectives.

The initial-state comparison is an inferred deterministic reconstruction check; the matched baseline checkpoints do not contain a saved initial-state digest. Because the auxiliary loss is summed with the final loss at weight 1, this pilot changes both the training signal and the aggregate gradient scale; it is therefore an intervention on the objective, rather than a pure information-isolation test.

## Train trajectory and interpretation

At update 250, aux final/intermediate validation accuracies were respectively 36,72%/89,45% (seed0), 34,77%/96,48% (seed1), and 48,05%/95,70% (seed2). At update 500 and every later checkpoint through 2000, all three seeds were 100% on both validation readouts; final checkpoint selection then favored update 2000 by lower final validation loss.

The intermediate head generalizes perfectly on the familiar two-hop validation and ID tables, while the final head remains 0% on the longer three-hop suite. This separates learning the first transition from composing the second under the fixed step schedule. The result supports continuing diagnosis of the step/state alignment, but does not identify whether the earlier failure arose from quantization, representation, optimization, or another interaction. No further sweep was run.

## Artifacts and verification

Per-seed reports and tagged checkpoints:

- `runs/intermediate_supervision_pilot/seed{0,1,2}/report.json`
- `runs/intermediate_supervision_pilot/seed{0,1,2}/aux/report.json`
- `runs/intermediate_supervision_pilot/seed{0,1,2}/aux/checkpoint_best.pt`

The implementation is in `scripts/intermediate_supervision_pilot.py`. Targeted model tests cover default-output equivalence, readout schedule validation, gradient flow through both the step-4 and step-8 losses, and unchanged parameter count. `.venv/bin/pytest -q` reports **28 passed** for the pilot-era test suite.

Review status: completed with no blockers. The pilot's claims remain bounded to this diagnostic; the later-memory causal check is a separate inference-only experiment.
