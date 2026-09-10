# Fixed two-hop diagnostic — 6 сентября 2026

## Результат

**Модель пока не выучила даже фиксированные два перехода на новых таблицах.** После общего бюджета 2000 updates для каждого seed, на checkpoint, выбранном только по validation (seed 0: update 1000, seed 1: 1750, seed 2: 750), ID accuracy осталась около случайного угадывания: 7,62%, 5,27%, 5,08%; среднее 5,99% против uniform expectation 6,25%. Это ограниченный отрицательный результат данного training regime, не доказательство невозможности архитектуры. Указанные checkpoints не являются автоматически финальными checkpoints на update 2000.

При этом и QAT, и float варианты запоминают 32 фиксированных two-hop примера: 32/32 уже на первой проверке после 250 updates. На новых таблицах они остаются около chance. Следовательно, обучение и подгонка ответов работают; это не свидетельство освоения алгоритма двух переходов. Причина отсутствия обобщения ещё не установлена.

## Условия и заранее выбранное ветвление

- Controlled directed cycle из 16 объектов, shuffled edge listing; таблицы train/validation/test разделены по canonical table hash.
- Train, validation и ID требуют **ровно 2 семантических перехода**. Вычислительных шагов модели всегда **8**. Secondary longer требует 3 перехода и не участвует в выборе checkpoint.
- Baseline `persistent_query=false`, d_model 64, d_ff 256, 4 FFN, 4 heads; 152512 параметров; тернарный QAT в FP32 simulation. Архитектура и training/resume API не менялись.
- По 2000 updates × batch 64 = 128000 примеров и 1024000 пример×вычислительный шаг на seed. Learning rate 0,001, AdamW, остальные настройки соответствуют предыдущему one-hop режиму.
- Best checkpoint выбирается только по 256 validation examples: accuracy, затем loss. ID содержит 512 новых test examples.
- До просмотра результатов зафиксирован рабочий критерий learnability: не менее 95% validation **и** ID во всех трёх seeds. При прохождении — curriculum; иначе — bounded tiny32 overfit. Это критерий этапа, не статистическая теорема. Критерий не пройден, curriculum не запускался.

## Fixed two-hop: streaming training

| Seed | Best update | Validation h=2 | ID h=2 | Secondary h=3 | Train seconds |
|---|---:|---:|---:|---:|---:|
| 0 | 1000 | 21/256 | 39/512 | 29/512 | 35,47 |
| 1 | 1750 | 23/256 | 27/512 | 38/512 | 35,71 |
| 2 | 750 | 17/256 | 26/512 | 27/512 | 37,45 |

Все три запуска имеют одинаковый evaluation suite fingerprint:
`0c7c940047a0e04a34b2cbc7d51a18eba877772d689ed5f439f891ea67ad8b02`.

Фактические данные: `runs/fixed_two_hop/seed{0,1,2}/evaluation_best.json`, `train_summary.json`, `metrics.jsonl`, `eval_sets.json` и checkpoints. Train time включает генерацию batch, transfer, forward/backward и optimizer; исключает evaluation/checkpoint writes. Запуски seed0 и seed1 частично пересеклись по CPU; seed2 частично пересёкся с evaluation предыдущих запусков. Поэтому время здесь описывает фактические запуски и не годится для сравнительного speed benchmark. Seeds и результаты точности не выбирались по времени.

## Tiny32: проверка способности запоминать

Один фиксированный набор из 32 train examples generated seed100000, ровно h=2, повторяется целым batch на каждом update. QAT и float начинают заново с seed0, имеют одинаковую архитектуру/число параметров и один набор; обучение продолжается 2000 updates без ранней остановки. Это 64000 предъявлений 32 уникальных примеров, а не 64000 новых таблиц. Batch tiny32 равен 32 для обоих режимов и отличается от streaming baseline (64), поэтому tiny32 не даёт строгой изоляции влияния quantization относительно baseline; сравнение скорости между режимами также не содержательно.

Фиксированный набор, в том числе targets, записан в `runs/fixed_two_hop/tiny32/fixed_train_manifest.json`. Fingerprint: `518c0afdc96e3e42c43357c0e001a26af95e43c8e16fac7ba6c9e572acd093d6`. Heldout suite совпадает с streaming диагностикой. Targets используются только как loss labels; модель получает только encoded input.

| Режим | Первый наблюдаемый 32/32 train | Final train | Best validation update | Validation | ID h=2 | Secondary h=3 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| QAT | 250 | 32/32 | 250 | 9/256 | 31/512 | 25/512 | 18,91 |
| Float | 250 | 32/32 | 250 | 11/256 | 30/512 | 29/512 | 12,23 |

Проверки проводились каждые 250 updates: точный момент достижения 32/32 внутри первых 250 не измерялся. Heldout metrics в таблице относятся к validation-selected weights; у этих weights train accuracy тоже 32/32. Final train loss после 2000 updates: QAT 0,000216, float 0,000126. Train time tiny32 включает forward/backward/optimizer и исключает подготовку фиксированного batch/evaluation; это другое определение, чем в streaming CLI.

Артефакты: `runs/fixed_two_hop/tiny32/{qat,float}/report.json`, `config.json`, `weights_best_validation.pt`, `weights_final.pt`. Эти weights-only файлы предназначены для данного diagnostic script, не являются resume checkpoints основного CLI. Test не используется для выбора weights. Диагностика норм evaluation выполняется отдельным untimed проходом существующего evaluator.

## Что это меняет

Смешение длин 1–4 не является единственным возможным объяснением предыдущего провала: ровно два перехода тоже не выучились. Повторное conditioning исходным запросом не включалось. Успех tiny32 исключает абсолютную неспособность модели подогнать эти примеры, но не разделяет недостаток inductive bias, оптимизацию, объём обучения и способ хранения промежуточного объекта. Float tiny32 не заменяет полноценное float обучение на streaming two-hop данных.

Следующий один эксперимент: **контроль структурированного reader на той же fixed-two-hop задаче** — keys строить только из source embedding, values только из destination embedding, сохранив текущие projections, восемь шагов, FFN, QAT, данные и budget. Сравнить seeds 0/1/2 с текущим baseline; не добавлять одновременно gates, router или curriculum. Это проверяет, мешает ли смешивание source/destination в каждой memory row выучить композицию чтений. Такой reader содержит task-specific inductive bias: даже успех не будет доказательством универсальной reasoning архитектуры. Это предложение, в данной работе reader не изменён.

## Воспроизведение и проверки

```bash
# Для каждого SEED = 0, 1, 2, последовательно и в новом RUN:
python -m looped_bitnet train --config configs/cycle16_two_hop_steps8.json \
  --device cpu --seed SEED --out RUN
python -m looped_bitnet evaluate --checkpoint RUN/checkpoint_best.pt \
  --device cpu --steps 8 --splits validation in_distribution longer \
  --out RUN/evaluation_best.json

python scripts/tiny_two_hop_overfit.py --out NEW_TINY32_RUN
python -m pytest -q
```

Оба сценария реально выполнены на CPU в `.venv` текущего проекта. 18 tests passed. Проверка нового standalone script — полный QAT/float прогон и чтение полученных manifests/metrics; production-код не менялся. CUDA/MPS и cloud не использовались. Независимое review завершено: blockers не обнаружены; ограничения включают различие tiny32 batch (32) со streaming baseline (64) и отсутствие строгой изоляции quantization в этой tiny32 comparison.
