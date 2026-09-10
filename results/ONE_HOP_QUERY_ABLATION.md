# Controlled-cycle one-hop и persistent-query абляция — 6 сентября 2026

## Вывод

Reader выучил один переход на новых cycle-таблицах с 16 объектами. Отдельно обученные QAT-модели с 1 и 8 вычислительными шагами получили по **256/256 validation и 512/512 ID test на каждом из seeds 0/1/2**. Восемь витков не испортили итоговую one-hop точность при 2000 updates, но обучались дольше: заметный рост validation начинался позже, а средний CPU train time был 30,42 с против 11,68 с. При одном шаге блоки B/C/D не участвовали и не обучались, поэтому это сравнение двух training regimes, а не post-hoc смена глубины одного checkpoint.

Persistent-query при 8 шагах также получил 256/256 validation и 512/512 ID на всех трёх seeds. На насыщенном one-hop результате улучшать accuracy было некуда. В multi-hop пилоте ни baseline, ни persistent-query не показали устойчивого обучения: средняя ID accuracy составила 7,49% и 7,42% при uniform expectation 6,25%. Эти результаты не дают свидетельств пользы persistent-query.

Адаптер добавляет 8192 параметра: 160704 против 152512. Дополнительную ёмкость не контролировали, а одинаковые seeds задавали одинаковые data streams, но не одинаковую инициализацию общих весов: создание адаптера сдвигает RNG до создания последующих слоёв. Поэтому даже при наблюдаемом разрыве нельзя было бы приписать его только постоянному conditioning. В этом пилоте содержательного разрыва нет.

## Реализованный режим

При `model.persistent_query=true` модель сохраняет исходный `h0 = E_start + hops * E_step / sqrt(d_model)`. На каждом витке bias-free QAT-адаптер `Linear([current_h, h0], d_model)` формирует вход attention query. Residual state, общая K/V-память и A→B→C→D FFN остаются прежними. Это не teacher forcing: режим не получает target, правильный текущий объект, промежуточную траекторию, remaining semantic hops или вторую копию таблицы. По умолчанию флаг выключен; тогда адаптера и его ключей в state dict нет.

## One-hop: отдельно обученные budgets

Общие условия: controlled cycle из 16 объектов, shuffled edge listing, train/validation/ID требуют ровно один переход; secondary longer требует 2 перехода. `d_model=64`, `d_ff=256`, 4 heads, 4 FFN, тернарный QAT в FP32 simulation, batch 64, 2000 updates = 128000 примеров. Best выбирался по validation accuracy, затем loss. Все варианты имеют один suite fingerprint `5bf9041e71857a02e4bdad3d1069dcac0afdb66b5b1be6c7200d312a4683d0da`.

| Режим | Seed | Best update | Validation | ID one-hop | Secondary h=2 | Train seconds | Пример × шаг |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline, 1 step | 0 | 2000 | 256/256 | 512/512 | 0/512 | 11,25 | 128000 |
| baseline, 1 step | 1 | 2000 | 256/256 | 512/512 | 0/512 | 11,42 | 128000 |
| baseline, 1 step | 2 | 2000 | 256/256 | 512/512 | 0/512 | 12,38 | 128000 |
| baseline, 8 steps | 0 | 2000 | 256/256 | 512/512 | 0/512 | 30,09 | 1024000 |
| baseline, 8 steps | 1 | 2000 | 256/256 | 512/512 | 0/512 | 30,61 | 1024000 |
| baseline, 8 steps | 2 | 2000 | 256/256 | 512/512 | 0/512 | 30,57 | 1024000 |
| persistent-query, 8 steps | 0 | 2000 | 256/256 | 512/512 | 0/512 | 33,66 | 1024000 |
| persistent-query, 8 steps | 1 | 2000 | 256/256 | 512/512 | 0/512 | 33,00 | 1024000 |
| persistent-query, 8 steps | 2 | 2000 | 256/256 | 512/512 | 0/512 | 36,95 | 1024000 |

Межseed standard deviation accuracy равен 0 для всех one-hop вариантов. Средний train time ± sample SD: baseline 1 step 11,68 ± 0,61 с; baseline 8 steps 30,42 ± 0,29 с; persistent-query 8 steps 34,54 ± 2,12 с. Равное число примеров не означает равную вычислительную цену; поле «пример × шаг» отличается ровно в восемь раз. Нулевой результат на h=2 показывает отсутствие length/task generalization после обучения только на constant hops=1; он не проверяет, использует ли модель length encoding в смешанном режиме.

## Multi-hop ограниченный пилот

После подтверждения one-hop чтения обучены baseline и persistent-query с train hops 1–4, validation/ID 1–4 и longer 5–8. Остальные условия и seeds те же. Suite fingerprint у всех шести запусков одинаков: `aa1ec2dace77dfb2935e345aea9baefe69dd47359cb907e3deb61109000f3a31`.

| Режим | Seed | Best update | Validation | ID 1–4 | Longer 5–8 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 2000 | 29/256 | 48/512 | 27/512 | 37,43 |
| baseline | 1 | 1000 | 22/256 | 34/512 | 24/512 | 33,22 |
| baseline | 2 | 750 | 25/256 | 33/512 | 33/512 | 32,07 |
| persistent-query | 0 | 1500 | 22/256 | 30/512 | 39/512 | 41,16 |
| persistent-query | 1 | 2000 | 32/256 | 47/512 | 39/512 | 36,25 |
| persistent-query | 2 | 500 | 26/256 | 37/512 | 33/512 | 38,78 |

Средние accuracy ± sample SD: validation baseline 9,90 ± 1,37%, query 10,42 ± 1,97%; ID baseline 7,49 ± 1,64%, query 7,42 ± 1,67%; longer baseline 5,47 ± 0,90%, query 7,23 ± 0,68%. Test не использовался для выбора checkpoint. При такой близкой к chance ID точности и трёх seeds разница longer не является доказательством обобщения или пользы механизма.

## Нормы состояния и обновлений

Диагностика выполняется отдельным untimed evaluation-проходом по тому же split и не влияет на logits, gradients или измеренный inference time. Ниже ID test seed 0; значения — средняя L2-норма по примерам. Полные значения каждого шага сохранены в `evaluation_best.json`.

| Режим | Шаг | `h` до | Reader update | FFN update | `h` после |
|---|---:|---:|---:|---:|---:|
| one-hop baseline 1-step | 1 | 0,167 | 0,445 | 1,012 | 1,212 |
| one-hop baseline 8-step | 1 | 0,225 | 0,407 | 3,571 | 3,694 |
| one-hop baseline 8-step | 8 | 7,129 | 0,080 | 1,132 | 7,406 |
| one-hop query 8-step | 1 | 0,203 | 0,370 | 3,382 | 3,563 |
| one-hop query 8-step | 8 | 6,726 | 0,246 | 1,290 | 7,070 |
| multi-hop baseline | 1 | 0,504 | 0,216 | 10,018 | 10,310 |
| multi-hop baseline | 8 | 16,881 | 0,173 | 0,246 | 16,994 |
| multi-hop query | 1 | 0,453 | 0,023 | 8,488 | 8,673 |
| multi-hop query | 8 | 14,073 | 0,025 | 0,691 | 14,470 |

Рост нормы состояния и различия норм reader/FFN не доказывают семантические шаги или правильное следование цепочке. Это только численная диагностика динамики.

## Команды и проверка

Speed probe: 25 updates заняли 0,15 с train time для 1 step и 0,38 с для 8 steps на CPU. После него выполнены:

```bash
python -m looped_bitnet train --config configs/cycle16_one_hop_steps1.json --device cpu --seed SEED --out RUN
python -m looped_bitnet train --config configs/cycle16_one_hop_steps8.json --device cpu --seed SEED --out RUN
python -m looped_bitnet train --config configs/cycle16_one_hop_steps8_persistent_query.json --device cpu --seed SEED --out RUN
python -m looped_bitnet train --config configs/multi_hop_pilot.json --device cpu --seed SEED --out RUN
python -m looped_bitnet train --config configs/multi_hop_pilot_persistent_query.json --device cpu --seed SEED --out RUN
python -m looped_bitnet evaluate --checkpoint RUN/checkpoint_best.pt --device cpu --steps 8 \
  --splits validation in_distribution longer --out RUN/evaluation_best.json
```

Для one-step evaluation использовалось `--steps 1`. `SEED` перебирал 0, 1, 2; каждый `RUN` был отдельным новым каталогом под `runs/one_hop_query_ablation/`. CUDA отсутствует (`torch.cuda.is_available() == false`), MPS доступен, но все сравнения проведены на CPU. Paid cloud не использовался.

`python -m pytest -q`: 18 passed. Тесты проверяют default без дополнительных state-dict ключей, сериализацию флага, запрет resume между baseline/query config, доступ неизменного исходного query на всех витках, зависимость query от start и hops, градиент через адаптер/start embedding, отсутствие target на входе модели и совпадение обычных logits с diagnostic forward. Существующий checkpoint-тест подтверждает точный save/load/resume для default режима.

Следующий один эксперимент: повторить multi-hop сравнение с активным адаптером одной формы в обоих вариантах — `[current_h, current_h] → query` против `[current_h, h0] → query` — и идентичной инициализацией всех общих параметров и адаптеров, сохранив текущие данные и budget. Это контролирует доступ к `h0` при одинаковой параметризации и убирает различие начальных весов, но не гарантирует одинаковую эффективную ёмкость двух входных представлений. Gated residual и post-norm в этот эксперимент добавлять не следует.
