# E15 — code review реализации

Статус: **не cleared для обучения**. Проверен текущий модуль, runner и
preflight; нейросетевое обучение не запускалось.

## Что подтверждено

`QATRegisterModel` действительно использует `AttentionReader`, четыре
`FFNBlock`, `d=64`, `d_ff=256`, четыре посещения блоков на инструкцию,
фиксированные KV из двух начальных ролевых значений и текущий opcode. В KV не
попадают targets, промежуточные истинные регистры или будущие команды.
`GRURegisterModel` получает те же начальные `x/y` embeddings и текущий opcode
на каждом из четырёх обновлений; скрытое состояние переносится между
инструкциями. Readout не возвращается в следующий шаг.

Инициализация соответствует протоколу: QAT embeddings, role keys и все
`nn.Linear` (включая `BitLinear`) получают `normal(std=.02)`, bias зануляется;
GRU embeddings, initial projection и обе собственные головы получают
`normal(.02)`, GRUCell сохраняет native PyTorch initialization. Фактические
counts совпадают: QAT `152768`, GRU `152720`.

Semantic secondary selection группирует длины 4 и 6 по полной таблице
преобразования на 256 состояниях, хеширует классы и членов, затем выбирает
round-robin. Получены группы `length4_allowed=32`,
`length4_forbidden=26`, `length6_allowed=32`, `length6_forbidden=32`.
Основной split и primary manifest также совпадают с протоколом.

Исправлены два ранее найденных дефекта: расписание теперь имеет ровно
`666/668/666` обновлений длин 1/2/3 и заканчивается двумя length-2; primitive
gate больше не является пустым фильтром и требует полный набор 32 seen
программ. Preflight успешно создаёт `manifest.json` и вложенный prefix batch,
а `pytest -q tests/test_register_e15.py` даёт `6 passed`. В manifest записаны
hashes модуля, протокола, design draft/review, semantic audit и audit script.

## Блокеры clearance

1. Runner не фиксирует заявленный режим CPU4threads и deterministic execution.
В текущем исходнике есть только `torch.manual_seed(seed)`; отсутствуют
`torch.set_num_threads(4)` и явная настройка deterministic algorithms. Это
нужно сделать до создания модели/optimizer и сохранить в metadata.

2. Digest раннего checkpoint сохраняется, но не проверяется при загрузке
selected checkpoint. `run_experiment` передаёт в `load_checkpoint` только
`manifest_hash` и `expected_update`; `batch_prefix_digest` не сверяется с
`checkpoint_digest_prefix(make_paired_batches(seed), update)`. Для gate и
последующей оценки требуется такой exact guard, иначе выбранный файл может
быть от другого batch prefix при совпадающих model/update metadata.

3. Runner принимает frozen manifest, но перед первым update не проверяет, что
сохранённые source hashes соответствуют текущим module/runner/protocol/audit
файлам. Один общий `manifest_hash` фиксирует содержимое manifest, но не
защищает от запуска изменённого runner с прежним manifest. Preflight hashes
нужно сверять fail-closed и включать подтверждённые hashes в итоговый report.

4. `gate_report` проверяет полноту множества программ, но не обнаруживает
дубликаты строк: duplicate row перезаписывается в `by_program`, после чего
полный set всё ещё может выглядеть корректным. Добавить `len(by_program) ==
len(rows)` или явный duplicate failure. Gate должен быть fail-closed на
неполном и неоднозначном отчёте.

Текущие тесты покрывают семантику, counts, параметры/градиенты, prefix
causality, checkpoint load и положительный/отрицательный gate, но не проверяют
эти четыре условия, runner end-to-end, source-hash verification или
secondary manifest digest. До их исправления результатом может быть только
implementation preflight, не scientific clearance.

## Проверенные hashes

`looped_bitnet/register_e15.py` —
`2557948b535ced2944c54bd8deaa955806e3c406e147ecef99635ba0bcd05ace`.

`scripts/register_interpreter_e15.py` —
`437ae47df8e07441678763cc68c6e1661570e06a36d731bd666f2058b9ee7754`.

`tests/test_register_e15.py` —
`dcf93006cb547d0ced37431f511e5d6af2983afb3b16b631e9cf6261c850c8a2`.
