# Controlled-cycle multi-hop pilot — 6 сентября 2026

## Вывод

В этом ограниченном pilot модель **не показала обучения нескольким переходам**. QAT checkpoint, выбранный только по validation, получил 37/512 = 7,23% на новых циклах со знакомыми длинами 1–4 при uniform expectation 6,25%; по отдельным hops результат 3,91%, 9,38%, 7,81%, 7,81%. На 5–8 переходах получилось 25/512 = 4,88%. Это один seed и небольшая модель, поэтому результат означает «свидетельств обучения нет», а не невозможность обучения архитектуры.

Увеличение inference budget одного QAT checkpoint с 1 до 4/8/16 шагов не помогло: accuracy на 1–4 hops была 8,20% / 7,62% / 7,23% / 6,64%; на 5–8 hops — 5,66% / 5,08% / 4,88% / 5,27%. Отдельно обученные budgets в этом pilot не запускались, поэтому о пользе большего training budget выводов нет.

Float-контроль той же архитектуры также не выучил задачу: 35/512 = 6,84% на 1–4 и 40/512 = 7,81% на 5–8. Значит, провал QAT нельзя уверенно объяснить одной только квантизацией. Градиенты оставались конечными: QAT grad norm 0,697–1,302, float 0,413–1,092; nonfinite loss не наблюдался.

## Данные и конфигурация

Оба запуска: `configs/multi_hop_pilot.json`, seed 42, 16 объектов, один случайный направленный цикл через все объекты, train hops 1–4, validation 256 примеров 1–4, два test по 512 примеров: 1–4 и 5–8. Канонические таблицы разделены hash-split 80/10/10; порядок рёбер и старт независимы. Внутри train таблицы не повторяются. Target и промежуточная траектория во вход не входят; длина использует прежний `STEP`.

Модель: d_model 64, d_ff 256, четыре FFN A→B→C→D, четыре головы, восемь вычислительных шагов, 152 512 параметров. Hops не управляют числом вычислительных шагов. Batch 64, AdamW, learning rate 0,001, weight decay 0,01, grad clip 1,0, 2000 updates = 128 000 примеров = 1 024 000 «пример × шаг». Checkpoint выбирался по validation accuracy, при равенстве — по меньшему validation loss.

## Команды и измерения

Перед длинным запуском выполнен speed probe:

```bash
python -m looped_bitnet train --config configs/multi_hop_pilot.json \
  --device cpu --out runs/multi_hop_cycle_benchmark_cpu --updates 25
```

25 updates заняли 0,42 с train time, то есть около 16,8 мс/update в этом коротком измерении.

```bash
python -m looped_bitnet train --config configs/multi_hop_pilot.json \
  --device cpu --out runs/multi_hop_cycle_qat_cpu
python -m looped_bitnet evaluate \
  --checkpoint runs/multi_hop_cycle_qat_cpu/checkpoint_best.pt \
  --device cpu --steps 1 4 8 16 \
  --out runs/multi_hop_cycle_qat_cpu/evaluation_best.json

python -m looped_bitnet train --config configs/multi_hop_pilot.json \
  --device cpu --float --out runs/multi_hop_cycle_float_cpu
python -m looped_bitnet evaluate \
  --checkpoint runs/multi_hop_cycle_float_cpu/checkpoint_best.pt \
  --device cpu --steps 1 4 8 16 \
  --out runs/multi_hop_cycle_float_cpu/evaluation_best.json
```

| Режим | Best update | Validation acc/loss | Train seconds | Wall seconds | Process peak RSS |
|---|---:|---:|---:|---:|---:|
| QAT | 1750 | 7,81% / 2,7672 | 28,71 | 29,09 | 298,17 MiB |
| Float | 1250 | 7,81% / 2,7751 | 20,22 | 20,47 | 295,38 MiB |

Process peak RSS — максимум всего Python-процесса, не чистая память модели. CUDA отсутствовала; VRAM, CUDA-fit и CUDA speedup не измерялись. QAT здесь является FP32-симуляцией и оказался медленнее float на CPU в этих единичных запусках; это не аппаратный benchmark.

## Контроли

Для cycle и hops < 16 `return_start` имеет 0%. Контроль «ровно один переход» получает 25% на смеси 1–4 и 0% на 5–8. У всех объектов destination-frequency одинакова; tie-break выбирает объект 0, то есть этот baseline становится константным.

На QAT test 1–4 модель и destination-frequency оба получили 7,23%, но совпали в **0/512 предсказаний**. На QAT test 5–8 оба получили 4,88%, но совпали лишь в 17/512 = 3,32%. Совпадение aggregate accuracy здесь явно не означает одинаковую стратегию.

## Ограничения и следующий эксперимент

Это один seed, CPU и маленькая модель; test не использовался для выбора checkpoint. Separate-training sweep 1/4/8/16 и основной d256/ff1024 конфиг только подготовлены. Результат не подтверждает length generalization и не измеряет RTX 3070 Ti.

Следующий один эксперимент: обучить тот же controlled-cycle d64/ff256/8-step QAT конфиг только на `hops=1` с seeds 0/1/2. Если он стабильно выучит один переход при 16 объектах, следующий bottleneck — смешанное multi-hop обучение; если нет, сначала нужно диагностировать reader на controlled-cycle данных, не расширяя архитектуру.
