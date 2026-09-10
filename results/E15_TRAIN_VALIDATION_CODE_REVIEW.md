# E15 train/validation diagnostic — independent code review

Статус: **CLEAR для одного зарегистрированного inference-only запуска**.
Дата: 2026-09-07. Real checkpoint inference в ходе этого review не запускался.

Проверены `scripts/e15_train_validation_diagnostic.py` и
`tests/test_e15_train_validation_diagnostic.py` против
`results/E15_TRAIN_VALIDATION_DIAGNOSTIC_PROTOCOL.md`.

## Вывод

Evaluator допускает только оба frozen seed0 selected update2000 checkpoint,
точный seen32 в frozen порядке и только train192/validation32. Reserved test,
primary forbidden programs, secondary lengths, другие checkpoints, обучение,
backward и optimizer path отсутствуют. Hash checkpoint, manifest, batch stream,
config и frozen source guards проверяются до evaluation.

Обе полные validation32 replay выполняются до первого train-split inference.
Каждая из32 строк обязана точно совпасть с selected validation по denominator,
final joint/x/y, всем prefix joint counts и full trace; любое расхождение
останавливает запуск до train inference и до создания report.

Coverage восстанавливается из точного `make_paired_batches(0)`, проверяет
2000×64, полный digest, только seen32/train192 и сохраняет per-program draws,
unique/zero/min/max и digest таблицы частот. Независимый metadata-only recount
в review получил 128000 draws и 6144/6144 program/state combinations.

Per-program counts/rates/gaps реализованы для final joint/x/y, prefix и full
trace. Length1/2/3 macros — невзвешенные средние per-program rates;
`seen_compositions` — равновзвешенное среднее length2 и length3. Необязательный
общий seen32 macro, который ранее был program-weighted, удалён до clearance.
Новых thresholds, CE/significance и QAT-vs-GRU paired claims нет.

Protected snapshot проверяется до evaluation и после всех inference calls,
перед записью; требуется ровно95 путей, и обе проверки должны вернуть один
mapping. Report пишет hashes evaluator и diagnostic protocol. Output
exclusive: существующий report или непустой каталог отвергаются.

## Проверка и frozen review hashes

Независимо выполнено:

`python -m pytest -q tests/test_e15_train_validation_diagnostic.py tests/test_register_e15.py`
— **14 passed**. Тесты покрывают rejection reserved split/wrong programs,
length-balanced macro, replay failure before train inference, exclusive output
и повтор protected check before report. Дополнительно metadata-only проверены
95 protected hashes, exact32/192/32 scope, V7 canonical manifest hash и
6144/6144 coverage. Научный report ещё не существует.

- evaluator SHA256:
  `e3ab437d9a7fc461dcb8342467368bce3621ffb932e3bc2de3175e3ec4752e45`;
- focused tests SHA256:
  `b66f019068b811e8ffcbcdce71cdd49e12d9931f9801602be55926f231c98cba`;
- diagnostic protocol SHA256:
  `4dc12c629d887a26f978d7443f8fc23ab13f8fd2f3d2e0fbb78b5f6b9c87250c`;
- protected snapshot SHA256:
  `3ae89d54f0fe567f8ba8e26785117082ce2141bf9d4b4cd80954072dd9995bd7`.

Clearance действует только для этих hashes и одного запуска в пустой
`runs/e15_train_validation_diagnostic/`. После запуска нужен независимый
recount machine result. Интерпретация обязана отдельно сказать, что выбранные
checkpoint были выбраны по этой же validation32: train−validation comparison
описателен и не является независимой оценкой generalization. Низкий train
result сам по себе не различает optimization, capacity, representation и
objective.
