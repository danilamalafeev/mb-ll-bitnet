# E32 width128 execution report

2026-09-08. The single canonical scientific execution completed with six
models in the frozen order `float128_seed0`, `w4128_seed0`, `float128_seed1`,
`w4128_seed1`, `float128_seed2`, `w4128_seed2`. Every cell completed 16,000
updates with attempted and completed counters equal. The saved report is
complete. Independent final review is **PASS**: all six saved finals replayed
exactly across the fixed 498-forward scope, and the raw prediction,
comparison, join, provenance, checkpoint, and cost audits passed.

The execution used 96,000 training forwards, 6,144,000 training cases,
12,288,000 instruction readouts, and 98,304,000 native substeps. The fixed
evaluation used 498 forwards, 72,192 cases, 223,872 readouts, and 1,790,976
native substeps. Runner-measured training time summed to 2,153.838425416
seconds; command-level wall, user, and system time were not captured and are
recorded as null in `E32_ATTEMPT_ACCOUNTING.json`.

All six models passed the seen prerequisite. All six passed the six E21 primary
conjunction, with the equivalent control excluded from that gate. W4 passed
the L4 final-readout gate for all three seeds. Float passed L4 for seeds 0 and
2; seed1 failed because L4_5 reached 242/256. Every model failed the strict
all-six L5 final-readout gate, so the combined restoration predicate is false
for both precision arms. Full-trace gates are reported separately: W4 passed
L4 full trace for every seed, while neither arm passed the all-seed L5
full-trace gate.

## Per-model gates and costs

| model | seen | E21 primary min | L4 final values | L5 final values | combined | training seconds |
|---|---:|---:|---|---|---:|---:|
| float128 seed0 | pass | 256 | 256,256,256,256,249,256 | 219,253,232,203,239,244 | fail | 245.758830 |
| W4-128 seed0 | pass | 253 | 253,256,251,254,255,256 | 212,235,227,228,228,247 | fail | 393.233051 |
| float128 seed1 | pass | 255 | 254,256,255,253,242,256 | 215,247,227,208,240,229 | fail | 295.992561 |
| W4-128 seed1 | pass | 255 | 256,256,253,255,245,256 | 213,246,236,222,233,233 | fail | 417.975173 |
| float128 seed2 | pass | 253 | 255,256,254,255,252,256 | 222,247,239,224,250,245 | fail | 325.125547 |
| W4-128 seed2 | pass | 255 | 254,256,255,254,255,256 | 209,229,227,234,230,242 | fail | 475.753264 |

The L4 and L5 final lists are ordered L4_1 through L4_6 and L5_1 through
L5_6. The corresponding full-trace lists are:

| model | L4 full trace | L5 full trace |
|---|---|---|
| float128 seed0 | 256,256,256,256,249,256 | 215,253,232,203,239,243 |
| W4-128 seed0 | 253,256,251,254,255,256 | 211,235,227,228,223,247 |
| float128 seed1 | 254,256,255,253,242,256 | 215,246,227,208,240,229 |
| W4-128 seed1 | 256,256,252,255,245,256 | 210,245,236,222,231,233 |
| float128 seed2 | 255,256,254,255,252,256 | 219,245,238,224,249,245 |
| W4-128 seed2 | 254,256,255,254,255,256 | 209,229,227,234,224,242 |

The six E21 primary final-count arrays, in frozen primary-program order and
explicit canonical execution-label order, were:

- `float128_seed0`: `[256,256,256,256,256,256]`
- `w4128_seed0`: `[256,253,256,255,256,255]`
- `float128_seed1`: `[256,255,255,255,256,255]`
- `w4128_seed1`: `[255,255,255,256,256,255]`
- `float128_seed2`: `[255,253,255,256,255,254]`
- `w4128_seed2`: `[256,256,256,256,256,255]`

The equivalent control was not used in the primary gate.

## Width contrasts

The strict descriptive width-benefit predicate is false for W4 and float in
each seed. Each row gives L4 and L5 final-error gaps
`errors_W4(128)-errors_float(128)`, the imported h64 gap, and the registered
interaction `gap128-gap64`:

| seed | width128 gap L4/L5 | width64 gap L4/L5 | interaction L4/L5 | strict L5 benefit W4/float |
|---:|---|---|---|---|
| 0 | +4 / +13 | -5 / -70 | +9 / +83 | false / false |
| 1 | -5 / -17 | 0 / +15 | -5 / -32 | false / false |
| 2 | -2 / +56 | +6 / -2 | -8 / +58 | false / false |

The total L5 final errors by precision and seed were:

| arm | seed | h64 errors | h128 errors | h128 minus h64 |
|---|---:|---:|---:|---:|
| float | 0 | 138 | 146 | +8 |
| float | 1 | 6 | 170 | +164 |
| float | 2 | 64 | 109 | +45 |
| W4 | 0 | 68 | 159 | +91 |
| W4 | 1 | 21 | 153 | +132 |
| W4 | 2 | 62 | 165 | +103 |

This fixed width expansion worsened the L5 final-error total for every seed in
both arms. The comparison uses 335,232 versus 151,232 parameters with FFN256
fixed and opcode coefficient 1/8; it does not establish that hidden-state
capacity is irrelevant or support a causal bottleneck claim.

The report contains per-program paired W4/float outcomes for final and full
trace, along with the imported E27/E24 comparisons, state strata, first
divergence, and recovery histograms. These are descriptive registered
contrasts and do not establish a causal quantization or memory-bottleneck
claim.

## Artifact and provenance paths

- Canonical preflight: `runs/e32_width_preflight/`
- Scientific run: `runs/e32_width/`
- Top-level report: `runs/e32_width/report.json`
- Progress and counters: `runs/e32_width/progress.json`
- Per-model checkpoints and reports: `runs/e32_width/{float128_seed*,w4128_seed*}/`
- Execution accounting: `results/E32_ATTEMPT_ACCOUNTING.json`
- Independent final review: `results/E32_WIDTH_REVIEW.md`
- Independent replay artifact: `runs/e32_review_final/review.json`
- Independent raw audit: `results/E32_review_raw_counts.json`
- Independent historical comparison audit: `results/E32_review_comparison_counts.json`

No automatic retry, extra training, intermediate scientific evaluation, or
post-hoc selection was performed. This report is assembled from the saved
run artifacts. Final acceptance is recorded after the independent review
artifacts passed; no training replay or extra scientific budget was used.
