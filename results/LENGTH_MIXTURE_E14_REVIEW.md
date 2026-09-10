# E14 length-mixture — follow-up review после relocation

Дата: 2026-09-07. Проверены только canonical artifacts:

- `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/runs/length_mixture_e14/`;
- `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/results/LENGTH_MIXTURE_E14.md`;
- `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/runs/length_mixture_e14/attempt_provenance.json`;
- `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/scripts/length_mixture_e14.py`.

`runs/length_mixture_e14_invalid_first_attempt/` не читался и не использован
как научный результат. Обучение и код этим ревью не изменялись.

## Решение

**ACCEPT.** После relocation canonical E14 artifact set завершён и
аудируем. Предыдущий blocker отсутствует: payload содержит `parameters`, а
selected checkpoint loader проходит для всех трёх seeds.

## Artifact relocation и provenance

`runs/length_mixture_e14/attempt_provenance.json` фиксирует:

- canonical run: `runs/length_mixture_e14`;
- source before relocation: `runs/length_mixture_e14_repaired`;
- `invalid_first_attempt_bytes_preserved=true`;
- `scientific_training_runs=1`, `test_evaluations=1`;
- relocation без редактирования checkpoint bytes, только смена directory paths.

Canonical `report.json` существует, содержит arms `0/1/2`, test evaluations
`mixture/baseline`, gate и predicates. `results/LENGTH_MIXTURE_E14.md`
ссылается теперь на canonical report и не использует invalid-first-attempt как
научный результат.

## Manifest novelty и hashes

В canonical `preflight.json` сохранены все обязательные scope counts:

`E07 persisted=3200`, `E07 reconstructed=1280`, `E08=1024`,
`E09=512`, `E10=512`, `E11=512`, `E12=512`, `E13 invalid=512`,
`E13 corrected=512`.

Для canonical validation/test manifests независимо проверены:

- validation: **256/256 unique**, fingerprint
  `520eaddb5cf25dfa11649c29c6bc439109ca3dff52e4a134e16e9ee6057504e1`;
- test: **512/512 unique**, fingerprint
  `2c3746f75aeca1815c56c5f5bce4f2bd4ad8cbf23c8a5e1406f2242999b9df8f`;
- обе manifests имеют корректный split membership;
- validation/test overlap: **0**;
- записанные `novelty_overlap_counts` для каждого из 9 scopes: **0**.

В рамках заданного follow-up scope historical source files не перечитывались;
проверены canonical manifest fingerprints, recorded novelty counts и hashes.

`protocol.json`, `preflight.json` и top-level report согласованы. Hashes
canonical validation/test manifests и checkpoint best/latest совпадают между
protocol, top-level report и per-arm reports. Текущий script hash exact-match:

`22976b322b92fc651de1685244002d3d50887e0a2be0392faea2b4f2eb9749f5`.

## Schedule, budget и checkpoint payloads

Независимый пересчёт `mixture_schedule()`:

- 2000 updates;
- h1/h2/h3 = **666/668/666**;
- `sum_h=4000`;
- exact schedule digest:
  `4fa565baeece54597ca88223a470ab9d41e061feb0e8041cc0fc86c82cd5f894`;
- 128000 examples;
- 256000 batch-weighted hop units;
- `4 × 4000 × 64 = 1024000` recurrent block-steps.

У всех arms status `complete`, updates `2000`, selected update `2000`,
parameters **152512**. Для каждого seed independently проверены
`checkpoint_best.pt` и `checkpoint.pt`: SHA совпадает с report, payload
содержит `parameters=152512`, objective
`length_mixture_h1_h2_h3_aux_uniform_weight2`, dynamics
`input_hops=[1,2,3]`, `recurrent_budget="4*h"`, `normalization="none"`,
`teacher_forcing=false`, правильные config/seed и update.

`load_mixture_checkpoint(..., report)` завершился **OK для 3/3 selected
checkpoints**. Per-arm report fields и top-level `report.json` exact-match.
Checkpoint best SHA-256:

| Seed | `checkpoint_best.pt` | `checkpoint.pt` |
|---:|---|---|
| 0 | `ade930359235960415cef22ce21ccb9020e3457058dd45bcb95f355ed33d3817` | `b2e48e7d812759b96c08698863e9d7a8e414511a1f883fe6516092a7f8aeea0e` |
| 1 | `a653c9dc53cd9a2fa19646570c821d2e6beafe7c26db90305e32916b640b7cdc` | `b6789f4db4522995933da4551280de50dc69dad0c7dde24577697ebcd96679b8` |
| 2 | `e93a958a086590096472d5d46d12d194ced184857881ee7a3e8891346a2e790c` | `281a4f8110d5b3d6d5e9e3379da960e92e537d4a77e39bcee674f0f050b1bd4d` |

Per-arm `sum_h=256000` — это batch-weighted значение; protocol `sum_h=4000`
и derived recurrent budget `1024000` согласованы. В finalized per-arm reports
поле `steps_seen` не заполнено (`null`), но точное значение однозначно
пересчитывается из frozen schedule; это metadata limitation, не blocker.

## Paired initial/data streams

Canonical reports и saved mixture digests exact-match independent reconstruction
и E13 base-h2 controls:

| Seed | Initial-state digest | Base-h2 digest | Mixture digest |
|---:|---|---|---|
| 0 | `4b203163fb2354bf8cc0ed681789454668c55afe8731253497c9d213d72d775c` | `dc163fc3f154dab76ae679af2210d97a972857f0fe7cf832e117efa6904a6979` | `ac3f66b1d23967dbe1c42a013445e2edf87c52dc512b607caa698547f74952a2` |
| 1 | `6f0bfeabd9a43a58cf2f3bdecca547a729f2fcdba272f9228c57d97239e9a06c` | `ae9826b3b5c2286d9468f7b616f56814bc0fd7f8acaebdc3a7cc7eec93de911e` | `0f5a5ba177ed8f5360898aa083c7098f5749b481d35ab977ea2e42af927e0477` |
| 2 | `445cdbdd75f676474ab7b5bd66fccf41dd718a7fede7df2815f8d9a4132c1f73` | `3cf817879b586d1f1d938c372267224a88cc2d159378a5052eba4a92ba7e0d2c` | `350ff52d5f66280efcac0eecaaf1329b5679d4d11594548384f1c1e87fd519d7` |

Mixture digests сохраняют отдельный actual input/target stream и не ошибочно
объявляют его равным base-h2 control.

## Validation selection и gate

У каждого seed сохранены validation updates
`250,500,750,1000,1250,1500,1750,2000`. Independent selection replay дал
selected update **2000** у всех seeds и macro final accuracy **1.0**.

Six-cell seed-0 gate: h1@4, h2@4/8, h3@4/8/12 — **256/256 в каждой клетке**.
Gate **PASS**, decision **extend**; arms seed1/2 действительно выполнены.

## Test predictions, counts и paired metrics

Для каждого seed и каждого mixture/baseline h1..h8 сохранены prediction
arrays для всех readouts `4k`, `k<=h`, по 512 элементов. Targets, построенные
из canonical test manifest, независимо пересчитали все readout counts и
совпали с report.

Final counts по h1..h8:

| Seed | Mixture | Baseline |
|---:|---|---|
| 0 | `512,512,512,512,512,512,512,512` | `512,512,484,367,286,211,179,144` |
| 1 | `512,512,512,512,512,512,512,507` | `512,512,252,20,7,12,15,10` |
| 2 | `512,512,512,512,512,512,512,512` | `512,512,512,505,497,487,473,446` |

Paired wins по h1..h8; losses во всех клетках равны zero:

| Seed | Wins h1..h8 | Losses h1..h8 |
|---:|---|---|
| 0 | `0,0,28,145,226,301,333,368` | `0,0,0,0,0,0,0,0` |
| 1 | `0,0,260,492,505,500,497,497` | `0,0,0,0,0,0,0,0` |
| 2 | `0,0,0,7,15,25,39,66` | `0,0,0,0,0,0,0,0` |

Каждая paired row имеет count **512**, а ties равны оставшимся examples.
Пересчёт из saved predictions exact-match `paired_test_metrics`.

## Predicates

Primary h8 delta:

| Seed | Mixture | Baseline | Delta | Threshold | Result |
|---:|---:|---:|---:|---:|---|
| 0 | 512 | 144 | **+368** | +26 | pass |
| 1 | 507 | 10 | **+497** | +26 | pass |
| 2 | 512 | 446 | **+66** | +0 | pass |

Independent predicate reconstruction exact-match report: all three primary
criteria **pass**; strong criterion (mixture >=487/512 for every h4..h8 in all
three seeds) **pass**.

## Ограничения

Это один three-seed CPU diagnostic на structured cycle16 task. Результат не
доказывает general LLM reasoning или arbitrary-depth generalization. Follow-up
проверял canonical relocation и сохранённые evidence; invalid-first-attempt не
использовался для научных выводов.

## Итог

Canonical E14 report, payload metadata, loader 3/3, manifests, recorded
novelty controls, script hash, validation gate, all h1..h8 predictions/counts,
paired metrics и predicates согласованы. Решение: **ACCEPT**.
