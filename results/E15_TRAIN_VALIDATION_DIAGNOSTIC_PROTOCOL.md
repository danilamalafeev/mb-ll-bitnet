# E15 — протокол train/validation diagnostic для frozen checkpoints

Статус: **зарегистрирован только inference-only diagnostic; запуск не входит в
этот документ**. Дата: 2026-09-07. Основание — принятый отрицательный pilot
`results/E15_REGISTER_INTERPRETER.md` и его независимое ревью. Цель — описать,
насколько два уже выбранных seed0 checkpoint воспроизводят 32 знакомые
программы на 192 парах из train split по сравнению с теми же программами на
32 validation-парах. Это не новый gate и не пересмотр исходного E15 gate.

## Неизменяемые входы и границы

Использовать ровно два существующих checkpoint:

- QAT `runs/register_interpreter_e15/qat_seed0/checkpoint_selected_u2000.pt`,
  SHA256 `227253c9a7a1fd3a44ba2f666c2026501e0f2302786b6872d48df50b845c7e13`;
- GRU `runs/register_interpreter_e15/gru_seed0/checkpoint_selected_u2000.pt`,
  SHA256 `5107e292507da10a9ead1e9de641fd12191d77a4406e94e3c86f5051ed8228c8`.

Оба должны загрузиться штатным `load_checkpoint` как `selected`, seed0,
update2000, с frozen V7 manifest hash
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`,
полным batch/prefix digest
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687`
и исходными config/source hashes. До и после запуска все 95 путей из
`results/E15_TRAIN_VALIDATION_PROTECTED_HASHES.json` должны существовать и
совпадать по SHA256.

Программы — ровно `manifest["programs"]["seen"]`: 32 программы длины1–3,
в том числе 3 primitives и 29 знакомых compositions. Состояния — ровно
`manifest["state_split"]["train"]` (192) и
`manifest["state_split"]["validation"]` (32), в frozen порядке. Разрешено
ровно `2 models × 32 programs × (192+32 states) = 14336` полных прогонов
program/state; это 36736 readout-позиций и 146944 внутренних recurrent
substeps при четырёх substeps на инструкцию.

Запрещены inference на reserved test32, шести primary forbidden-программах,
secondary длинах4/6 и любых других программах или состояниях. Не загружать
latest/промежуточные checkpoints. Не обучать, не делать backward, optimizer
step, подбор checkpoint, seed, порога или гиперпараметра. Модели перевести в
`eval()` и вычислять только под `torch.inference_mode()` на CPU; checkpoint и
старые артефакты не перезаписывать.

## Fail-closed preflight и validation replay

До train-split inference исполнитель должен:

1. проверить protected hashes, V7 manifest, оба checkpoint и точные множества
   32 программ / 192 train / 32 validation;
2. запретить evaluator-у принимать имя split кроме `train` или `validation`,
   а список программ — кроме точного frozen seen32;
3. сначала заново посчитать на validation32 все существующие поля
   `joint_final`, `final_x_correct`, `final_y_correct`, `prefix_joint` и
   `full_trace` для каждой модели и каждой программы;
4. потребовать точного совпадения этих целочисленных полей со всеми 32 rows в
   соответствующем `arm_complete.json` / selected validation. Любое
   отсутствие, дубликат, изменение порядка или несовпадение останавливает
   diagnostic до train192 inference и не создаёт научный result report.

Этот replay должен, среди прочего, восстановить зарегистрированные primitives:
QAT ADD4/XOR15/SWAP32 и GRU ADD5/XOR15/SWAP32 из32. Он служит проверкой
идентичности весов и evaluation semantics, а не новым измерением gate.

## Actual training-stream coverage

Термин `train split` сам по себе не означает, что каждая комбинация была
показана оптимизатору. Поэтому без model inference заново построить ровно
`make_paired_batches(0)`, проверить 2000×64=128000 examples и frozen digest,
затем посчитать частоты точных троек `(полная program, initial x, initial y)`.

Для каждой из32 программ сохранить: число draws; число уникальных train-пар из
192; число пар с нулевой частотой; минимум/максимум частоты по всем192 парам;
SHA256 канонической таблицы `[(x,y,count)]` во frozen train порядке. Также
сохранить общий unique count из `32×192=6144`. Реконструкция при регистрации
дала 6144/6144 фактически встреченных комбинаций (128000 draws; минимальная
частота любой комбинации не меньше1); реализация обязана пересчитать это, а не
копировать число. Только комбинацию с положительной частотой можно называть
`stream-exposed`. Accuracy ниже всегда называется `train-split inference`, а
не автоматически `training accuracy`.

## Метрики и агрегация

Для каждой модели, программы и каждого из двух splits сохранить целые counts с
denominator `n=192` или `n=32`:

- `final_joint`: оба конечных регистра верны;
- `final_x` и `final_y`: каждый конечный регистр отдельно;
- `prefix_joint[j]`: оба регистра верны после инструкции `j`, независимо от
  ошибок на других prefix;
- `full_trace`: все prefix readouts данной программы верны одновременно.

Рядом хранить rates `count/n`. Для каждой программы вычислить только
описательный gap в percentage points: `train rate - validation rate` для тех
же полей. Counts остаются первичными; округление rates не должно участвовать в
вычислениях.

Макроагрегация проводится отдельно для длины1 (3 программы), длины2
(8 программ) и длины3 (21 программа): невзвешенное среднее per-program rates
для `final_joint`, `final_x`, `final_y`, каждого доступного prefix index и
`full_trace`. Дополнительно дать две компактные строки: `primitives` = length1
и `seen compositions` = среднее двух length-macro для length2 и length3, чтобы
21 программа длины3 не подавляла length2. Общий seen32 macro, если показан,
равен среднему трёх length-macro. Для каждой макростроки gap считается как
train macro минус validation macro. Не добавлять CE, confidence intervals,
significance tests, paired QAT-vs-GRU win/loss или новые pass/fail thresholds.

Человекочитаемая таблица должна показать для QAT и GRU: primitives поимённо,
29 compositions поимённо и компактные length/group macros. Не скрывать
низкие строки усреднением.

## Допустимая интерпретация

Это описательная локализация принятого gate failure для двух фиксированных
seed0 моделей после фиксированных 2000 updates.

- Высокий train192 при низком validation32 совместим с разрывом переноса на
  disjoint initial register pairs для данной программы и модели.
- Низкий train192 показывает, что checkpoint плохо воспроизводит даже
  train-split program/state cases, которые coverage-анализ подтвердил как
  встречавшиеся в stream. Сам по себе этот результат не различает проблемы
  оптимизации, представления, ёмкости, objective или их взаимодействие.
- Различия primitives, compositions, register x/y, prefix и full trace можно
  описывать как место наблюдаемой ошибки. Они не доказывают механизм ошибки.
- Один seed и разные QAT/GRU transitions не поддерживают общий causal claim о
  квантовании или превосходстве архитектуры.

Diagnostic не изменяет исходный E15 gate, не открывает primary/test, не
позволяет retrofitted success claim и не регистрирует следующее обучение.

## Артефакты и стоп

Исполнитель создаёт только новый evaluator, его узкие tests и один новый
machine artifact `runs/e15_train_validation_diagnostic/report.json`; каталог
должен быть пуст/отсутствовать до запуска и не допускает overwrite. Machine
report включает hashes/provenance, validation-replay status, coverage и все
per-program/macro counts/rates/gaps; per-example predictions не нужны.
Координатор после проверки может создать один краткий human report
`results/E15_TRAIN_VALIDATION_DIAGNOSTIC.md`; независимый reviewer — один
`results/E15_TRAIN_VALIDATION_DIAGNOSTIC_REVIEW.md` с checkpoint reload,
пересчётом всех aggregate counts и повторной проверкой exclusions/protected
hashes.

Остановиться после одного успешного inference-only запуска и независимого
review либо при первом fail-closed нарушении. Никаких повторов для улучшения
результата, новых seeds, sweeps, retraining, cloud или автоматизации.
