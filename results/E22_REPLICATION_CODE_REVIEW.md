# E22 independent code review — CLEAR

Reviewer Astra/low,2026-09-07. Clears only the registered fresh initialization
seeds1 and2, each native8/u8000 on the unchanged data-seed0 stream, followed by
fixed seen and E21 diagnostic evaluation and independent result review.
No scientific E22 training or primary-program preview occurred in this review.

## Stable reviewed hashes

- module: `250d44a475fb7991dd67c46fbdc6acdf6131ccc0eb5706ebb963bb83d8ac14f4`
- runner: `afc76ffd0302b159865d11b0cf9d3c41407f65449513f1b371e26e680b318908`
- tests: `14afd14db8bc13d7dfad5bdd51b996be81ff29e12e8f0797884214e6a40c83dc`

Independently checked these file-byte hashes. Root must freeze them with the
protocol, dependencies, references, initial states and data in fresh preflight.
Source changes require renewed review; no historical files were edited here.

## Independent controls performed

**5 focused tests passed in3.12s**: all four new E22 tests plus the inherited
E21 asymmetric243/244 and control-exclusion test. No full suite was repeated.

The actual production `_train` loop ran tiny legal batches for BOTH seeds,
repeating a two-batch stream twice. It exercised updates, progress, final
model/AdamW/RNG saving, legal seven-program evaluation/report generation and
aggregate reporting. A numerically failed first primary predicate did not omit
its report or prevent seed2 execution. Prerequisite and primary booleans remain
separate; prerequisite failure does not suppress primary/control diagnostics.
The actual evaluation helper ran only seven legal seen programs in QA.

Both tiny final model/AdamW states matched uninterrupted updates, and the next
update after strict reload remained exact. Wrong seed, initialization, native
budget, model/optimizer/RNG or manifest digest was rejected. A synthetic
post-save evaluation exception preserved its checkpoint, marked the affected
seed incomplete and the other not_started, and retained the operational record.
The inherited boundary test confirms244 passes,243 fails, one failing primary
prevents conjunction, and a perfect control cannot rescue that failure.

An additional independent temporary preflight/load check ran successfully from
`/tmp/e22_preflight_review.py`. It verified every seed0/1/2 complete initial
model digest, x/y projection byte hash and inside-constructor RNG byte hash
against root's E22 initialization reference; actual saved initial artifacts
and manifest reload agree. Wrong data seed and changed initial tensor dtype
were rejected. All206 protected files verified unchanged. This extra check
performed zero optimizer updates or model predictions.

## Source-path conclusions and repaired findings

Initialization seed controls both the master model and independent local x-then-y
projection generator. The fixed data seed remains0. New seed0 construction is
bitwise identical to accepted E20/E18 initial state; seeds1/2 repeat exactly and
are distinct. Counts/native budget and actual optimizer membership are fixed.
No historical nonzero-seed guard was changed or bypassed.

The factory captures torch RNG INSIDE its isolated constructor, attaches that
state without adding model parameters/buffers, and restores external RNG on
return. Preflight freezes the captured state; the runner explicitly installs
it before training. It is therefore unambiguous which state drives training.
This matches the independent post-constructor references and preserves the
registered initialization law. Accepted training has no stochastic layer; the
policy is nevertheless frozen for provenance. Saved initial state checks use
exact dtype/shape as well as torch.equal. JSON tuple/list representations are
compared canonically against the frozen E21 symbolic content; the independent
symbolic reference remains hash-bound.

Each seed gets one fresh continuous AdamW instance and all8000 fixed updates.
No intermediate scientific evaluation or metric-dependent selection exists.
Every completed final is saved before evaluation. Seen evaluation and E21's
one-pass-per-program scorer preserve mode and state; the runner checks model,
optimizer and torch RNG digests around evaluation. Primary six, control and
all strata are retained regardless of numerical prerequisite failure.

Final loader binds seed/native budget/update and complete model/optimizer/RNG
states to the preflight digest, which binds full sources, protocol/config,
initial states, stream/targets and evaluation references. It checks exact
state inventory, metadata/digests, optimizer settings and all step counters.
Recovery calls this same loader and evaluation only; source inspection confirms
no training or overwrite path and explicit missing-final reporting. A technical
failure conservatively suspends unstarted work; no isolated-failure continuation
or retry is inferred automatically. A numerical predicate failure is not a
technical exception and both seeds still run.

## Scope and QA accounting

Independent executable controls consumed21 tiny updates/21 examples/168
training substeps, including uninterrupted/reload comparisons and the deliberate
failure fixture. Legal seven-program QA added14 forwards/3584 cases/5632
readouts/45056 substeps. The inherited count-boundary test and independent
preflight audit added no training or neural inference. Executor separately
reported63 QA updates/63 examples/504 training substeps and42 legal forwards/
10752 cases/16896 readouts/135168 inference substeps; do not hide these costs
inside the nominal scientific budget or call them replications.

No blocking finding remains at these hashes. This is conditional robustness
across two complete initializations with reused data and opened E21 programs,
not a new holdout, data-seed replication or general reasoning result. Report
both primary predicates, prerequisites and combined predicates separately;
missing finals remain incomplete. Root alone may run the two registered seeds.
Stop after both outcomes and independent review, irrespective of pass/fail.
