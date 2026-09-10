# E15 — выбранный протокол реализации

2026-09-07. Owner root, executor e15_impl Luna/high. Пользователь поручил
обсудить выбор с субагентом и приступить. Design review:
`results/E15_REGISTER_DESIGN_REVIEW.md`. Этот документ закрывает варианты
предыдущего register draft; по расхождениям действуют точные решения ниже.
Разрешена реализация и проверка; обучение — после отдельного code review.

## Задача и доступная информация

ADD/XOR/SWAP над x,y в 0..15, семантика из независимого полного audit JSON.
Один текущий opcode на инструкцию, четыре внутренних обновления состояния.
Ни будущая программа, ни число будущих команд, ни labels в forward не входят.
Начальные регистры доступны обеим моделям на всех внутренних шагах;
изменяемые регистры должны сохраняться только в собственном скрытом состоянии.
Readout двух регистров после каждых четырёх шагов, выход не подаётся назад.

## Архитектура

QAT с нуля: d64, dff256, 4 cyclic FFNBlock, AttentionReader4heads,
два отдельных value embeddings16×64, opcode embedding3×64, role keys2×64,
memory LayerNorm64, output LayerNorm64, две bias-free BitLinear64→16.
Начальное h — сумма ролевых x/y embeddings. KV строится однократно:
ключи — нормированные role keys, values — нормированные начальные x/y
embeddings. На каждом substep: h += opcode_embedding/sqrt(64), затем
h += reader(h,KV), затем соответствующий остаточный FFN. Порядок FFN
0,1,2,3 на каждую инструкцию, блоки повторно используются. 152768 параметров.
Embeddings normal(std=.02), linear weights normal(std=.02), biases0;
LayerNorm defaults. Нет cycle-reset, router, дополнительного query.

GRU float: отдельные x/y16×64 и opcode3×64 embeddings,
bias-free initial projection128→132 над concat(x_embedding,y_embedding),
GRUCell192→132 (native PyTorch initialization), output LayerNorm132,
две bias-free Linear132→16. На каждом substep input — concat начальных
x/y embeddings и текущего opcode embedding; h переносится через всю
программу. Embeddings normal(std=.02), initial/output linear normal(.02),
LayerNorm defaults. 152720 параметров. H132 фиксирован по подсчёту:
3H²+744H+2240, разница48 параметров. По accuracy размер не подбирать.
Перед первым запуском фактический count подтвердить программно.

Это сравнение двух систем с одинаковой доступной информацией и близкой
ёмкостью. Reader против конкатенации, QAT против FP32 и разные recurrent
переходы остаются различиями; causal claim о квантовании не следует.

## Данные

Все39 программ длины1–3; train32 без ADD→XOR. Из7 исключённых одна
ADD,XOR,XOR эквивалентна ADD: только отдельный контроль. Primary —6
различных новых преобразований, все256 состояний для каждого.
Hash state split: сортировка пар x,y по SHA256 ASCII
`E15-state-v1:x:y`, train192 / validation32 / обычный test32.
Все пары общие для моделей/seeds. Marginals раскрыть; это disjoint пары,
а не незнакомые отдельные значения регистров. Primary результаты отдельно
для подмножеств192/32/32, основной denominator256.

Secondary — длины4 и6 × наличие/отсутствие запрещённой пары.
По32 программы на группу (или все, если меньше32). Сначала группировать
по полной семантике на256 состояниях. Упорядочить классы SHA256 от
`E15-secondary-v1:` + длина + `:` + group + `:` + JSON signature;
members по hash opcode sequence; выбор round-robin классов до32.
Точная сериализация и порядок сохраняются в manifest до обучения.
Отдельно отметить эквивалентность разрешённым коротким программам;
secondary не участвует в primary/gate/выборе checkpoint.

## Оптимизация и gate

Seed0/1/2, CPU4threads deterministic; AdamW lr=.001 wd=.01 foreach=False,
grad clip1, batch64, 2000updates. Однородные длины1/2/3, 666/668/666
updates, последние2updates длины2. Внутри длины равномерно выбирать
разрешённую программу и train state; local RNG seed+100000, независимый
от model RNG. Точный порядок batches общий для QAT/GRU; сохранять digest.
Initial weights между разными архитектурами не объявлять одинаковыми.
Loss=(1/L)sum_j[CE(x_j)+CE(y_j)]. Четыре обновления на инструкцию у обоих.
Max6arms=12000updates/768000examples/6144000 state updates, не equal-FLOPs.

Validation каждые250updates. Выбор: macro final joint accuracy,
сначала среднее по программам каждой длины, затем по3длинам; tie —
меньшая dual CE с тем же macro averaging, затем более ранний update.
На выбранном checkpoint seed0 КАЖДОЙ модели: primitive32/32 для каждой
операции, каждое знакомое составное program≥31/32. Если хотя бы одна
модель не проходит, закончить обе seed0 arms и сохранить gate failure;
основной test не запускать. Не менять настройки и не пробовать другие seeds.
Если обе проходят — обе модели seeds1/2 до бюджета независимо от gates.

## Метрики и проверки

Primary каждой модели отдельно: каждая из6программ в каждом из3seeds
имеет joint final≥244/256. Все prefix joint counts и full-trace counts,
пары QAT/GRU wins/losses/ties. Не объявлять отсутствие QAT superiority
провалом: численный критерий превосходства не задан.
Identity/first-only/last-only/sorted-opcode exact symbolic controls,
таблицы отдельных инструкций с последовательным исполнением. Для
перестановок с одинаковым multiset оценить отдельно случаи разных ответов;
ADD,XOR против XOR,ADD —224/256состояний. Causal late opcode replacement:
для общего prefix использовать клонированные h/cache и сравнить
альтернативные последние команды с direct execution; без oracle feedback.

## Артефакты и стоп реализации

Новый task module, runner и tests. Материализовать frozen manifests
в `runs/register_e15_preflight/`; scientific run только новый
`runs/register_interpreter_e15/`, непустой output запрещён.
Сохранить protocol/config/source/manifest hashes до первого update.
Selected/latest checkpoint tags, params, optimizer, seed, progress,
validation и digest stream prefix на фактическом selected update.
Complete per-arm и итоговый report, также честный failure report.
Metadata/evaluation failure не повод повторять обучение: восстановление
из сохранённых весов отдельным режимом, сохраняя исходные evidence.

До обучения: tests semantics, absence future/label leakage, four-step prefix
replay, gradients, save-load-eval на temp smoke, earlier best-update
digest/guards, positive/negative gate, exact split/schedule/parameter count.
Отдельный reviewer проверяет готовый код и выдаёт clearance. Никаких
научных обновлений весов или test inference до этого gate.
