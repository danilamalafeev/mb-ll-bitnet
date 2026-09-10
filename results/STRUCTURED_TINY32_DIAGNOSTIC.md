# Structured-reader tiny32 diagnostic — 6 сентября 2026

## Результат

Structured reader прошёл заранее заданный критерий **fit fixed set: 32/32 к 2000 updates** в обоих режимах. Первый наблюдаемый 32/32 был уже на checkpoint 250 для QAT и float32 и сохранялся до update 2000. Это проверяет запоминаемость именно 32 предъявленных examples; heldout accuracy при этом осталась около uniform chance и не показывает обобщение.

На validation-selected weights QAT получил 12/256 validation и 26/512 ID, float32 — 15/256 validation и 31/512 ID. Uniform expectation для 16 объектов — 6,25%: результаты находятся около chance. Best checkpoint выбирался только по validation accuracy, затем по validation loss; ID и secondary наборы не участвовали в выборе.

## Проверенный protocol и evidence

Сначала были прочитаны фактические артефакты прежнего mixed tiny32. Оба `runs/fixed_two_hop/tiny32/{qat,float}/config.json` содержат `batch_size=32`, `updates=2000`, `eval_every=250`, `eval_batch_size=64`, seed 0, `d_model=64`, `d_ff=256`, 4 blocks, 4 heads и 8 compute steps. Batch 64 относится к streaming fixed-two-hop baseline, поэтому новый mixed контроль не пересчитывался.

Новый запуск использовал тот же старый config плюс `--structured-reader`; reader mode — единственное изменение. `runs/fixed_two_hop/tiny32_structured/fixed_train_manifest.json` совпал с прежним файлом побайтно (`cmp`), включая порядок и targets. Fixed fingerprint: `518c0afdc96e3e42c43357c0e001a26af95e43c8e16fac7ba6c9e572acd093d6`. Validation/ID/secondary suite fingerprint: `0c7c940047a0e04a34b2cbc7d51a18eba877772d689ed5f439f891ea67ad8b02`.

Таблица и обучение используют 32 fixed two-hop examples, validation 256, ID 512 и secondary h=3 512 examples. Optimizer — прежний AdamW с learning rate 0,001, weight decay 0,01 и grad clip 1,0. Каждый режим запущен последовательно на CPU в отдельном подкаталоге; параметры и число state-dict keys сохранены на уровне прежнего reader (152512 parameters). Существующее структурированное ветвление строит keys из source rows и values из aligned destination rows; новых архитектурных механизмов не добавлялось.

## Train memorization trajectory

Train accuracy измерялась на тех же точных 32 examples на каждом eval checkpoint. Значения `loss` — cross-entropy на fixed set; это отдельная memorization диагностика, не heldout metric.

| Update | QAT train | QAT loss | Float train | Float loss |
|---:|---:|---:|---:|---:|
| 250 | 32/32 | 0,011360 | 32/32 | 0,007750 |
| 500 | 32/32 | 0,003611 | 32/32 | 0,002344 |
| 750 | 32/32 | 0,001757 | 32/32 | 0,001156 |
| 1000 | 32/32 | 0,001078 | 32/32 | 0,000694 |
| 1250 | 32/32 | 0,000713 | 32/32 | 0,000463 |
| 1500 | 32/32 | 0,000522 | 32/32 | 0,000331 |
| 1750 | 32/32 | 0,000393 | 32/32 | 0,000247 |
| 2000 | 32/32 | 0,000306 | 32/32 | 0,000191 |

Final weights are saved separately as `weights_final.pt`; the final fixed-set measurements above are from update 2000. Validation-selected weights are saved as `weights_best_validation.pt`; both modes selected update 250.

## Heldout results

| Mode | Best update | Validation (256) | ID (512) | Secondary h=3 (512) | Train seconds |
|---|---:|---:|---:|---:|---:|
| Structured QAT | 250 | 12/256 (4,69%) | 26/512 (5,08%) | 35/512 (6,84%) | 20,74 |
| Structured float32 | 250 | 15/256 (5,86%) | 31/512 (6,05%) | 32/512 (6,25%) | 15,06 |

The heldout values are evaluated after loading each validation-selected checkpoint. The fixed train set is not part of the heldout counts, and no test or ID value was used for checkpoint selection. The structured QAT/float results are close to the previous mixed tiny32 control (QAT 31/512 ID, float 30/512 ID), while both readers fit the fixed set; this bounded result does not identify a generalization benefit for the structured representation.

## Code and artifacts

The standalone script now accepts `--structured-reader`, whose default is false; a config with `model.structured_reader=true` also works. With no flag and the legacy config, the script retains the previous mixed-reader behavior. The report additionally stores `train_memorization_trajectory` while retaining the legacy `history` fields.

Artifacts:

- `runs/fixed_two_hop/tiny32_structured/qat/report.json`
- `runs/fixed_two_hop/tiny32_structured/float/report.json`
- `runs/fixed_two_hop/tiny32_structured/fixed_train_manifest.json`

Reproduction:

```bash
.venv/bin/python scripts/tiny_two_hop_overfit.py \
  --config configs/cycle16_two_hop_steps8.json \
  --structured-reader --out runs/fixed_two_hop/tiny32_structured
```

## Limitations and one next experiment

This is one fixed seed and one fixed 32-example set. Reaching 32/32 only establishes that this model and optimizer can memorize these inputs; it does not establish a two-hop algorithm or generalization. The heldout suite is the existing validation/ID/secondary suite, and CPU timings are run measurements rather than a speed benchmark. The comparison also inherits the reader normalization choice documented in `STRUCTURED_READER_ABLATION.md`.

The one next experiment supported by this control is **one additional structured-reader seed on the same tiny32 manifest**, keeping the protocol and checkpoints identical, to test whether the observed 32/32 fit and near-chance heldout behavior are seed-stable.
