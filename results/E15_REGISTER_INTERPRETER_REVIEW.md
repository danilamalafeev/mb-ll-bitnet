# E15 — independent interpreter pilot review

Статус: **ACCEPT — valid negative pilot / gate failure**.

Проверены `runs/register_interpreter_e15/report.json`, оба seed-0
`arm_complete.json`, selected checkpoints, все validation checkpoints и frozen
V7 manifest. Обучение действительно остановилось после seed 0; primary/test
inference, secondary inference и seeds 1/2 отсутствуют.

Report фиксирует ровно два completed arms, 4000 updates, 256000 examples и
2048000 internal state updates. В каталоге есть только `qat_seed0`,
`gru_seed0` и report; `evaluations` и `paired_primary` в gate-failure report
отсутствуют. Frozen manifest hash совпадает с V7:
`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.

Для каждого arm независимо перезагружены selected checkpoint и все восемь
`checkpoint_latest_u{250,500,750,1000,1250,1500,1750,2000}.pt` с проверками
manifest hash, seed, tag, update, config, source hashes и exact prefix digest.
Оба selected checkpoints выбраны на update 2000. Независимая реконструкция
schedule/RNG/state split дала полный batch digest
`63833e7e8f8501ceb31476b46f074c6d33d454521e764aa1be829bd6ecbe1687` для seed
0; он совпал между QAT и GRU, с `full_batch_digest` и с обоими selected
prefix digests. Пересчитанный validation selection по восьми точкам также
выбрал update 2000 для обоих arms.

Дополнительная bounded verification перезагрузила оба selected weights и
повторно вычислила все 32 seen-program validation rows; их joint counts
совпали с gate report. Все 8 latest validation checkpoints каждого arm также
перезагружены с exact prefix guards. Независимое сравнение всех 52 файлов из
`results/E15_PREEXISTING_HASHES.json` дало 0 missing и 0 mismatches; старые
E14/protected artifacts не изменились.

Gate rows полные: 3 primitives и 29 seen compositions, по 32 validation
states каждая. QAT selected validation дал `ADD=4/32`, `XOR=15/32`,
`SWAP=32/32`; GRU дал `ADD=5/32`, `XOR=15/32`, `SWAP=32/32`. Поэтому обе
модели нарушили primitive gate `32/32`; composition gate также провален
(минимум QAT `0/32`, GRU `3/32`). Report корректно имеет `status=gate_failure`,
а не скрывает отрицательный результат и не запускает следующий этап.

Никаких оснований считать этот результат провалом primary composition transfer
нет: primary programs не оценивались. Он показывает только, что обе seed-0
модели не достигли зарегистрированного learnability gate на seen-program
validation subset при бюджете 2000 updates. QAT и GRU нельзя сравнивать по
primary wins/losses, поскольку paired primary evaluation не запускалась.

## Exact artifact hashes

`runs/register_interpreter_e15/report.json` —
`5ddce0e376782edd8de85899d20833502830c21f6332955a2f1f99827d8e5aef`.

`runs/register_interpreter_e15/qat_seed0/arm_complete.json` —
`a2fcc742ac9217a887b1c24ab1a421c1bcb38bdc5c286fa509f42f78c78110a6`.

`runs/register_interpreter_e15/gru_seed0/arm_complete.json` —
`bb5450a03b251e925e7c669f2c3fa24ac34af469248402cba46303bd5fdd676b`.

`runs/register_interpreter_e15/qat_seed0/checkpoint_selected_u2000.pt` —
`227253c9a7a1fd3a44ba2f666c2026501e0f2302786b6872d48df50b845c7e13`.

`runs/register_interpreter_e15/gru_seed0/checkpoint_selected_u2000.pt` —
`5107e292507da10a9ead1e9de641fd12191d77a4406e94e3c86f5051ed8228c8`.

`runs/register_e15_preflight/v7/manifest.json` —
`72e240e2f70d32592ff8bc2d5ffeb851580cc62504af66c4850b3a2ea590f733`.
