# E35/E36 length wave

Status: **E35 ACCEPT; E36 ACCEPT.** Independent raw-data auditing and exact replay of all six E36 checkpoints are complete. See [independent review](E35_E36_REVIEW.md).

The frozen v2 manifest is `runs/e35_e36_preflight_v2/manifest.json` (SHA256
`4adf2dd1bb1036b94bcbf7a0246d9410596a737c9d20e5ff8865632fab5988aa`). The
reviewed runner is `scripts/length_wave_e35_e36.py` (SHA256
`383658faa81b82a068f17a223450453eb8c1d2e591a3add2c3d7139d9af35c0f`).

E35 evaluated 36 fixed programs at all 256 inputs per precision: three
prefixes, three final operations, and four SWAP padding levels (lengths 3, 5,
7, 9). Final-correct counts (denominator 2304 per length) were:

| precision | L3 | L5 | L7 | L9 |
|---|---:|---:|---:|---:|
| float | 2304 | 2270 | 1588 | 660 |
| W4 | 2302 | 2249 | 1012 | 258 |

The independent E35 audit replayed all 18,432 saved cases and 884,736 native
steps exactly. Extra padding changes execution history and compute together;
this is evidence about this fixed package, not a causal result about capacity,
h64, or a universal length threshold.

E36 used the registered A8000 and B8000 continuations, with one continuous B
run and a saved B@4572 equal-training-cost snapshot. The primary heldout pool
is 24 program strings (six each at L7–L10), 23 distinct full-domain semantic
functions because one cross-length alias remains, and 6,144 cases per
checkpoint. All 23 are disjoint from the 94 semantic functions exposed by
ancestor and continuation prefixes. The A stream repeats L1–L3; B is 4,000
short and 4,000 long batches. Training-syntax labels are union exposure and
are not branch-specific mastery claims.

Primary full-trace errors (out of 1,536 cases per length) were:

| precision / checkpoint | L7 | L8 | L9 | L10 | total |
|---|---:|---:|---:|---:|---:|
| float / A | 435 | 1044 | 1099 | 1428 | 4006 |
| float / B@4572 | 0 | 0 | 4 | 10 | 14 |
| float / B8000 | 0 | 0 | 4 | 8 | 12 |
| W4 / A | 698 | 1226 | 1354 | 1507 | 4785 |
| W4 / B@4572 | 0 | 0 | 5 | 25 | 30 |
| W4 / B8000 | 0 | 0 | 1 | 13 | 14 |

Both precisions improved in the equal-update and the equal-instruction comparisons. The six sampled
training-probe cells had zero errors: A covered 2,880 cases; B/B@4572 covered
6,336 cases, including 3,456 L4–L6 cases. These are sampled attainment probes,
not full training-set mastery.

Scientific artifacts are under `runs/e35_e36_length_wave/science/`, with raw
reports and checkpoints for `float128_seed0` and `w4128_seed0`. E36 training
cost per precision was A: 8,000 forwards, 512,000 cases, 1,024,128 readouts,
8,193,024 internal steps; B: 8,000, 512,000, 1,791,872, 14,334,976.
Wall times were float 747.48 s and W4 900.88 s. B was not retrained for the
snapshot. No scientific retries occurred.

Bounded QA used 14 actual updates total (7 per arm), 896 examples, 1,280
readout positions, 10,240 internal steps, two evaluation forwards, and ten
loader calls; exact model, optimizer, and RNG continuation identity passed.
The loader reset Torch to four threads per process despite the intended two;
the two concurrent workers therefore used eight total threads. This is logged
as executed resource configuration, with budgets and data unchanged.

The earlier refused/nonempty preflight and inherited-loader preparation do not
provide an exact preparation load count; no scientific forwards or updates are
attributed to them. The separate historical six-program L5 regression was
omitted from this bounded wave. Results remain limited to one h128 seed-0
continuation and the bundled length/composition intervention; they do not
establish a causal mechanism or architectural capacity limit.

Independent E36 replay: 1,374 forwards, 341,376 cases, 2,096,256 readout positions, 16,770,048 internal steps, zero updates. Float and W4 replay commands took 170.90 s and 172.80 s. Every saved prediction and probe row matched exactly. The fixed first wave is complete.
