# E15: fixed-core operation composition — preregistration proposal

**Status:** DRAFT / NOT CLEARED. Candidate protocol only; no implementation or
training has been run.

## Question and hypothesis

Can the existing recurrent core compose several non-commuting operations when
the operation at each step is supplied explicitly, and transfer to ordered
combinations that were absent from training? The narrow hypothesis is that a
fixed four-block cyclic core can learn the semantics of individual operations
and their composition. This is a scaffolded composition test; it is not yet a
test of learned routing.

## Task and encoding

The register is `x in {0,...,15}`. Four fixed bijections on `Z_16` are used:

```
A(x) = x + 1       (mod 16)
B(x) = 3*x + 1     (mod 16)
C(x) = 5*x + 2     (mod 16)
D(x) = 7*x + 3     (mod 16)
```

All multipliers are odd, so each operation is a permutation. They do not
commute: applying A then B to zero gives 4, whereas applying B then A gives 2.
A program has length `L in {1,2,3}` and is an ordered string over A--D. It is
padded with `N`, the identity operation, to three command slots, followed by a
fixed fourth `N` sentinel slot. The concrete input table has 16 rows:
`(source=j, destination=opcode_j)` for `j=0,1,2,3`, with each row repeated
four times. Opcode IDs A--D,N occupy five ordinary object IDs, and the
`START` object is the initial register value. The input then contains exactly
12 `STEP` tokens and `END`, so its length is 63 tokens and fits the current
64-token guard. This repeated table is a task-specific input adapter that is
syntactically accepted by the existing structured reader; it is not a claim
that the unchanged data generator already implements this grammar. The
recurrent computation receives the opcode for the current slot explicitly.

Each program slot is a four-step macro. The task-specific input adapter adds a
learned 5x64 opcode embedding to the recurrent state at each of its four
steps. The structured reader, d=64, FFN=256, four heads, four cyclic FFNs,
QAT settings, and fixed `select_block(t)=t mod 4` schedule remain as the
baseline. Thus every operation visits the four fixed blocks once; `N` macros
provide padding to a fixed 12-step budget. There is no router, block-specific
query, halting, or target-dependent input. If this exact adapter is used, the
parameter count is 152,832; this count is provisional until an implementation
preflight verifies the state-dict diff. The final output remains a 16-way
state classification.

For prefix `y_j = o_j(...o_1(x_0))`, with N acting as identity, train with

`CE(readout_12, y_L) + (1/L) * sum_{j=1..L} CE(readout_{4j}, y_j)`.

The prefix labels are used only in the loss and evaluation; no predicted or
correct intermediate state is fed back. The final readout is always after the
same 12-step budget.

## Split and held-out combinations

The finite program universe contains 4 + 16 + 64 = 84 strings. The test
programs are exactly the two length-2 strings `AB` and `BA`, plus every
length-3 string containing `AB` or `BA` as an adjacent ordered pair (16
programs total). These are the only held-out program identities; all four
operations and every other ordered pair occur in training. The remaining 68
programs use initial states 0--11 for training and 12--15 for validation.
The 16 held-out programs are evaluated on all 16 initial states (256 test
examples). Splits and ordering are deterministic and hashed before training.

This modest split avoids claiming a fully disjoint operator-pair universe from
a tiny program set. It tests one concrete novelty: AB/BA is absent as an
adjacent training pair, while each operation and other pairs are present.

## Controls and measurements

Report final and prefix accuracy separately for seen programs with unseen
initial states, held-out length-2 programs, and held-out length-3 programs.
For each initial state, pair AB with BA and report joint correctness and the
rate at which predictions are identical. Also report deterministic order-blind
controls that sort the opcode multiset, use only the first opcode, or use only
the last opcode. These checks distinguish composition and order sensitivity
from state priors and command shortcuts, and must include the N-padding
identity check.

## Budget, gate, and stopping rule

Use seeds 0, 1, and 2; AdamW, learning rate 0.001, weight decay 0.01,
gradient clip 1.0, batch size 64, deterministic CPU execution, and 2,000
updates per seed. Every update uses the fixed 12-step budget. Select only from updates 250, 500, ..., 2,000 by validation
macro final accuracy, breaking ties toward the earliest update. The maximum
training budget is 384,000 examples and 4,608,000 recurrent block-steps
(1,536,000 per seed).
There are no retries, sweeps, cloud runs, or test-based decisions.

Seed-0 gate: seen-program validation final accuracy must be at least 95% and
the mean actual-prefix accuracy at least 90%. If it fails, stop and report a
gate failure. If it passes, run seeds 1 and 2 at the fixed budget and evaluate
the test once.

The preregistered success criterion is, for every seed, at least 75% final
accuracy on all 256 held-out examples, at least 75% separately on the length-2
(32 examples) and length-3 (224 examples) subsets, and at least 12 of 16 AB/BA
initial-state pairs jointly correct. Failure of any criterion is a negative or
inconclusive pilot, not a reason to add a router in the same experiment.

## Interpretation

Passing would show that this fixed core can execute known non-commuting
operations and transfer to a small held-out ordered-pair family under an
explicit opcode/time scaffold. It would motivate a separately registered
router comparison with matched data and executed steps. It would not show
self-organized operations, routing benefit, arbitrary-depth transfer, or
language reasoning. Since A--D are a fixed finite affine family, the model
could memorize finite composites; this is not proof of a reusable symbolic
interpreter. A later router must be a new arm.

## Preflight required before clearance

Before training, code review must verify the 16-row table parsing, absence of
position/target leakage, exactly four opcode injections per slot, readouts at
4/8/12, and the declared state-dict diff. It must freeze hashes for the
84-program universe and all splits. Until then this is a design proposal, not
a registered experiment.
