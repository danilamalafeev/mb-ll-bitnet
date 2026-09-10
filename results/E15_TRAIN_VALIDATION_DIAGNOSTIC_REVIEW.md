# E15 train/validation diagnostic — independent result review

Статус: **ACCEPT — valid bounded descriptive diagnostic**. Дата: 2026-09-07.

Проверены machine report
`runs/e15_train_validation_diagnostic/report.json`, оба frozen selected seed0
checkpoint, V7 manifest, arm-complete validation evidence, protected snapshot
и текущие frozen sources. Reserved test32, primary forbidden programs и
secondary lengths не использовались в reviewer inference.

## Независимый пересчёт

Reviewer отдельно загрузил только QAT и GRU
`checkpoint_selected_u2000.pt` штатным guarded loader. Для exact seen32 на
train192 и validation32 logits пересчитаны в `eval()` / inference mode.
Targets и целочисленные final-x, final-y, final-joint, каждый prefix-joint и
full-trace counts вычислены отдельной символической реализацией ADD/XOR/SWAP,
без вызова evaluator `evaluate_program` или его aggregation helpers.

Совпали все **128 model × program × split rows**, все counts, rates и
train-minus-validation gaps. Независимые невзвешенные macros для length1/2/3,
а также равновзвешенный по length2/length3 `seen_compositions` совпали с
report. Validation32 rows также целиком совпали с исходными selected validation
rows обоих `arm_complete.json`.

Основные final-joint результаты:

| Model | Group | Train | Validation | Gap, pp |
|---|---|---:|---:|---:|
| QAT | length1 | 97.40% | 53.12% | +44.27 |
| QAT | length2 | 69.86% | 33.98% | +35.87 |
| QAT | length3 | 32.14% | 17.26% | +14.88 |
| QAT | seen compositions | 51.00% | 25.62% | +25.38 |
| GRU | length1 | 100.00% | 54.17% | +45.83 |
| GRU | length2 | 100.00% | 52.34% | +47.66 |
| GRU | length3 | 91.89% | 34.97% | +56.92 |
| GRU | seen compositions | 95.94% | 43.66% | +52.29 |

Primitive joint counts совпали точно: QAT train ADD183/XOR186/SWAP192 из192
против validation4/15/32 из32; GRU train192/192/192 против5/15/32.
Full-trace и prefix breakdown также пересчитаны и совпали; в частности QAT
length3 full trace25.99% train /7.29% validation, GRU91.89%/21.58%.

## Provenance, coverage и границы

Независимо восстановлен seed0 stream из зарегистрированного schedule и RNG,
без diagnostic coverage helper: digest
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`,
128000 draws и6144/6144 exact `(program, initial pair)` combinations. Все
комбинации stream-exposed; частоты на одну пару были48–102 для primitives,
12–47 для length2 и1–24 для length3. Все per-program frequency summaries и
их table digests совпали с report.

Все95 protected paths существуют и совпадают с snapshot; checkpoint,
evaluator, protocol, manifest canonical/file hashes в report совпадают с
текущими файлами и code clearance. В output-каталоге ровно один machine
artifact. Report фиксирует ноль optimizer updates и точную разрешённую
стоимость:14336 program-state runs,36736 readouts,146944 recurrent substeps.
Reviewer inference повторил этот объём только для независимой проверки и не
является новой научной репликацией или обучением.

- machine report SHA256:
  `5cf396ff88d530cb3d29d9b3b63d85f57b498117ff15e05ce39aca7ca7d0a33f`;
- evaluator SHA256:
  `e3ab437d9a7fc461dcb8342467368bce3621ffb932e3bc2de3175e3ec4752e45`;
- protected snapshot SHA256:
  `3ae89d54f0fe567f8ba8e26785117082ce2141bf9d4b4cd80954072dd9995bd7`.

## Интерпретация и следующий шаг

У GRU почти полный fit stream-exposed seen programs сочетается с большим
падением на disjoint validation pairs; это совместимо с сильным разрывом
переноса по initial-state pairs в этой конфигурации. У QAT к такому разрыву
добавляется низкое воспроизведение уже exposed длинных compositions. Поэтому
исходный E15 gate failure нельзя свести к одной общей причине.

Это один seed и фиксированный budget. Более того, checkpoint выбирался по той
же validation32, поэтому validation и train−validation gaps описательны и не
являются независимой оценкой generalization. Результаты не различают
optimization, capacity, representation, objective или quantization и не
устанавливают общее превосходство GRU.

Рекомендуемый следующий кандидат — один preregistered primitive-only control
на GRU, где при неизменных core, data budget и train192/validation32 сравнить
текущие learned value embeddings с явным 4-bit представлением регистров;
checkpoint заранее фиксировать по update, не выбирать по validation32. Это
проверит конкретную гипотезу о representation-driven state-pair gap до возврата
к composition transfer. Это рекомендация, а не зарегистрированный запуск;
обучение не начиналось.
