# E25 — QAT против float при 16000 обновлениях

Статус: завершён; независимое ревью **ACCEPT как валидный отрицательный результат**. Заданный уровень задачи не сохранён. Технических сбоев и научных перезапусков не было.

## Основной результат

QAT обучен с нуля для seeds0/1/2, по16000 обновлений. Float comparator — сохранённые E24 финалы, без повторного обучения. Начальные FP32 masters побитно равны, данные/порядок/optimizer/число шагов одинаковы. Отличие —14 существующих BitLinear вместоLinear: тернарное fake-квантование весов И per-vector INT8 fake-квантование активаций. Входные signed-bit проекции остаются Linear; native8,151232 параметра.

| Seed | Float ошибки /1536 | QAT ошибки /1536 | QAT точность | Primary прошло | Seen prerequisite | Combined |
|---|---:|---:|---:|---:|---|---|
| 0 | 14 | 64 | 95.8333% | 4/6 | True | False |
| 1 | 2 | 48 | 96.8750% | 6/6 | False | False |
| 2 | 0 | 49 | 96.8099% | 6/6 | False | False |

Float прошёл все три predicates у всех seeds. QAT не прошёл combined ни у одного; all-seeds primary=false, retention (all-seeds combined)=false. Неизменный порог: каждая primary-программа≥244/256; каждый примитив validation32/32, каждая seen-композиция≥31/32. Все композиционные результаты QAT сообщены независимо от prerequisite, как предусмотрено протоколом.

| Программа | Float0 → QAT0 | Float1 → QAT1 | Float2 → QAT2 |
|---|---:|---:|---:|
| ADD → XOR | 254→248 | 256→252 | 256→252 |
| ADD → ADD → XOR | 252→235 | 255→244 | 256→249 |
| ADD → XOR → ADD | 255→249 | 256→251 | 256→246 |
| ADD → XOR → SWAP | 255→251 | 256→247 | 256→245 |
| XOR → ADD → XOR | 254→246 | 255→245 | 256→247 |
| SWAP → ADD → XOR | 252→243 | 256→249 | 256→248 |

Каждое число из256. Среднее не заменяет критерий каждой программы.

## Парные исходы

Win: QAT прав/float ошибался. Loss: QAT ошибся/float был прав. Сопоставлены идентичные программа, состояние, target и stratum.

| Seed | Wins | Losses | Оба ошиблись | QAT−float правильных |
|---|---:|---:|---:|---:|
| 0 | 14 | 64 | 0 | -50 |
| 1 | 2 | 48 | 0 | -46 |
| 2 | 0 | 49 | 0 | -49 |

Отсутствие общих ошибок в этих парах не доказывает независимость ошибок и не является проверкой ансамбля.

## Знакомые программы

Float: каждый seed6144/6144 train и1024/1024 validation по финалу и всей трассе. QAT:

| Seed | Train final /6144 | Train full trace /6144 | Validation final /1024 | Validation full trace /1024 | ADD/XOR/SWAP /32 |
|---|---:|---:|---:|---:|---|
| 0 | 6105 | 6103 | 1018 | 1018 | 32/32/32 |
| 1 | 6109 | 6089 | 1002 | 994 | 32/32/32 |
| 2 | 6105 | 6103 | 1007 | 1007 | 32/32/32 |

QAT оставляет35–39 финальных ошибок среди6144 train program/state cases. Разрыв не ограничен новыми композициями; причина этим не установлена. Возможна роль оптимизации при данном QAT-рецепте.

Провалы seen prerequisite:

- Seed0: нет.
- Seed1: ADD → SWAP 29/32; ADD → ADD → ADD 30/32; ADD → SWAP → ADD 30/32; XOR → ADD → ADD 29/32; XOR → XOR → ADD 30/32; XOR → XOR → SWAP 30/32; SWAP → ADD → SWAP 29/32.
- Seed2: SWAP → ADD → ADD 29/32; SWAP → XOR → ADD 29/32; SWAP → SWAP → ADD 30/32; SWAP → SWAP → XOR 29/32.

## Эквивалентный контроль

ADD→XOR→XOR эквивалентен ADD и исключён из primary.

| Seed | Float → QAT /256 | Wins | Losses |
|---|---:|---:|---:|
| 0 | 246→250 | 10 | 6 |
| 1 | 256→251 | 0 | 5 |
| 2 | 256→249 | 0 | 7 |

Контроль seed0 улучшился246→250, несмотря на ухудшение primary: эффект не одинаков для каждого примера или программы.

## Промежуточные ответы и страты

Boundary — правильность отдельного readout после инструкции, не накопительная точность. Full trace требует правильности всех readouts. Последние три столбца — финальные joint counts.

| Seed | Программа | Boundary /256 | Full trace /256 | Final train /192 | Final validation /32 | Final test /32 |
|---|---|---|---:|---:|---:|---:|
| 0 | ADD → XOR | 256/248 | 248 | 184 | 32 | 32 |
| 0 | ADD → ADD → XOR | 256/256/235 | 235 | 179 | 29 | 27 |
| 0 | ADD → XOR → ADD | 256/248/249 | 244 | 186 | 32 | 31 |
| 0 | ADD → XOR → SWAP | 256/248/251 | 246 | 187 | 32 | 32 |
| 0 | XOR → ADD → XOR | 256/256/246 | 246 | 187 | 29 | 30 |
| 0 | SWAP → ADD → XOR | 256/255/243 | 242 | 181 | 32 | 30 |
| 1 | ADD → XOR | 256/252 | 252 | 189 | 32 | 31 |
| 1 | ADD → ADD → XOR | 256/253/244 | 241 | 183 | 31 | 30 |
| 1 | ADD → XOR → ADD | 256/252/251 | 247 | 188 | 31 | 32 |
| 1 | ADD → XOR → SWAP | 256/252/247 | 245 | 186 | 31 | 30 |
| 1 | XOR → ADD → XOR | 256/254/245 | 243 | 184 | 30 | 31 |
| 1 | SWAP → ADD → XOR | 256/256/249 | 249 | 188 | 29 | 32 |
| 2 | ADD → XOR | 256/252 | 252 | 189 | 32 | 31 |
| 2 | ADD → ADD → XOR | 256/256/249 | 249 | 190 | 30 | 29 |
| 2 | ADD → XOR → ADD | 256/252/246 | 243 | 186 | 29 | 31 |
| 2 | ADD → XOR → SWAP | 256/252/245 | 244 | 182 | 31 | 32 |
| 2 | XOR → ADD → XOR | 256/256/247 | 247 | 186 | 31 | 30 |
| 2 | SWAP → ADD → XOR | 256/255/248 | 247 | 187 | 29 | 32 |

## Вывод и границы

При этом фиксированном matched-рецепте существующий BitLinear-пакет не сохранил заданный уровень float. QAT остаётся примерно96% на новых композициях, но уступает соответствующим float-моделям на50/46/49 правильных ответов из1536. Это не доказательство невозможности качественного QAT и не разделение weight- и activation-quantization. LR/расписание/бюджет специально под QAT не оптимизировались.

Данные фиксированы (data seed0), архитектура/бюджет выбраны адаптивно, E21-набор уже открыт. Только три инициализации и конечный DSL; нет нового holdout, переноса большей длины, GRU-сравнения, routing или проверки общего reasoning. Старые результаты E22/E24 сохранены.

FP32 masters и floating-point операции остаются: fake-QAT не использует packed low-bit kernels, экономия памяти/ускорение не установлены. Бюджет сопоставлен по updates и внутренним шагам, не FLOPs. Wall time не является speed benchmark; скорость третьего запуска заметно менялась.

Следующий кандидат — отдельно разделить влияние квантования весов и активаций, меняя один компонент. Дизайн пока не зарегистрирован, следующий код/обучение/проверка длины не запускались.

## Артефакты, проверки и стоимость

Протокол `results/E25_QAT_MATCH_PROTOCOL.md`; protected249 `results/E25_PROTECTED_HASHES.json`; independent initial audit `results/E25_INITIAL_REFERENCE.json`; code CLEAR `results/E25_QAT_MATCH_CODE_REVIEW.md`; result ACCEPT `results/E25_QAT_MATCH_REVIEW.md`. Исполнитель и отдельный reviewer Astra/low; root — shared writer.

До обучения проверены все32 tensors, независимость хранилищ,14 точных BitLinear locations и исходные saved initials. RNG соответствует фактическому float-обучению: seed0 bare manual_seed0 из E20; seeds1/2 saved E22 initial RNG. Отличие от incidental E22 seed0 constructor RNG выявлено и исправлено до preflight/обучения; исходный audit сохранён в `results/E25_INITIAL_REFERENCE_CONSTRUCTOR_AUDIT.json`.

Fresh `runs/e25_qat_match_preflight/` и `runs/e25_qat_match/`: три final checkpoints, полный report.json, progress. Checkpoints сохраняют QAT FP32 masters/AdamW/RNG, initial/source/config/data/float-reference provenance и counters.

Научно48000 updates/3072000 training examples/49152000 training substeps; оценка213 forwards/26880 cases/70464 readouts/563712 substeps. Attempted=completed; технических сбоев/научных перезапусков0. CLI1310.89s. Последние training timestamps seeds0/1/2:299.9596s/321.7671s/660.6113s, без последующего checkpointing/evaluation/reporting.

Executor QA:10updates/20examples/160train substeps +3eval forwards/6cases/48substeps. Independent code QA: столько же; финальные guards дополнительно проверены без forward. Подробности `results/E25_ATTEMPT_ACCOUNTING.json`; научных previews/float-retraining не было.

Независимо strict-loaded все QAT finals (14BitLinear/AdamWstep16000/RNG/provenance/counters), повторены все213 forwards и получено точное совпадение полных evaluation dictionaries. DSL, intermediate/final/stratum metrics, paired marginals и predicates пересчитаны отдельно. Review добавляет563712 inference substeps,0 обучения,0 retries. Все249 защищённых файлов и float comparator artifacts неизменны. Условие остановки выполнено; активных запусков нет.
