# Deep research: diagnostics for operation composition

**Date:** 2026-09-07  
**Status:** research note; no training result is claimed here  
**Scope:** candidate CPU-sized tasks for a recurrent QAT model with four reused blocks and fixed KV memory

This note records the research discussion on the next diagnostic after short-to-long
transition transfer. The existing E15 proposal is the concrete preregistration
candidate; this document keeps the broader task comparison and its caveats.

## Starting point

The current project result is reported as at least 99% on eight transitions after
training on one to three transitions with intermediate supervision in three seeds.
That establishes a length-transfer result. It does not by itself establish transfer
to an ordered combination of different operations. Those are separate tests:

* **Productivity:** execute longer sequences than those used for training.
* **Systematicity:** execute a familiar operation in a combination or order absent
  from training.

This distinction follows the evaluation framework of Hupkes et al. It is a claim
about the definition of the tests, not a prediction about the current model.

## Candidate 1: tiny register interpreter (recommended)

### Published basis

*Learning to Execute* trains recurrent models to predict the result of short Python
programs and publishes a program/data generator. The original repository uses an
old Torch7/CUDA stack, so the relevant part for this project is the execution-task
semantics; a small CPU reimplementation is preferable to attempting to reproduce
that stack.

Sources: [paper](https://arxiv.org/abs/1410.4615),
[author repository](https://github.com/wojciechz/learning_to_execute).

### Proposed diagnostic

Use two registers `x,y ∈ {0,…,15}` and three explicit instructions:

```text
ADD   x <- (x + y) mod 16
XOR   x <- x xor y
SWAP  (x,y) <- (y,x)
```

The input contains the initial registers and an ordered instruction string. The
target is both final registers; intermediate register states are supervised after
each instruction. The current opcode is supplied at the relevant recurrent step,
so the first test measures operation-conditioned execution rather than learned
program-counter routing.

### Split and checks

Train on lengths one to three while excluding the adjacent pair `ADD → XOR`.
The primary test uses lengths two and three containing that pair. Lengths four and
six are a separate productivity test. Every individual operation and all other
ordered pairs remain represented in training.

The state space has only 256 initial states. Before sampling, enumerate each
program's full mapping over those states. Remove or separately mark programs that
are semantically equivalent to a training program; otherwise a nominally unseen
program could be an accidental duplicate. Check also for palindrome-like or
constant-output cases and for any operation/position correlation.

### Baselines and criterion

Use the exact interpreter as a data oracle, a sequential lookup-table baseline, and
a small GRU with the same inputs, loss, and update budget. A successful diagnostic
would require at least 95% exact correctness of both registers for every primary
held-out program in each of three seeds, plus high intermediate-state accuracy.
Longer programs should be reported separately and must not rescue a failure on the
held-out pair.

## Candidate 2: PCFG SET string transformations

### Published basis

Hupkes et al. provide a benchmark and public data/code for systematicity,
productivity, substitutivity, localism, and overgeneralisation. The grammar includes
operations such as copying, reversing, shifting, echoing, repeating, and appending.
Their data and evaluation scripts are available in the official repository.

Sources: [paper](https://arxiv.org/abs/1908.08351),
[official data and scripts](https://github.com/i-machine-think/am-i-compositional).

### Proposed CPU-sized subset

Keep a small alphabet and a bounded output string. Give the operation sequence
explicitly and choose three transformations, for example reverse, cyclic shift, and
repeat-last. Exclude one ordered pair from training while keeping each primitive and
the reverse order present.

Guard against shortcuts from palindromes, repeated symbols, output length, and
formatting variants. Balance input lengths and report exact-string accuracy by
operation pair. Minimal baselines are input copying, an order-blind multiset model,
and a small GRU executor.

This task is feasible on CPU, but it adds sequence storage and exact decoding to the
question. A failure would be harder to attribute specifically to operation
composition than in the register task.

## Candidate 3: execution-only subset of the DeepCoder DSL

### Published basis

DeepCoder studies program synthesis from input/output examples. Its public utility
repository contains DSL definitions, input/output generation, and enumerative
search; it does not provide the learned model itself. For this project, the useful
piece is the typed operation semantics, used as an explicit-program executor.

Sources: [paper](https://arxiv.org/abs/1611.01989),
[Microsoft utility repository](https://github.com/microsoft/DeepCoder-Utils),
[generator source](https://github.com/microsoft/DeepCoder-Utils/blob/master/generate_io_samples.py).

### Proposed subset

Use short lists and a few operations such as `MAP(+1)`, `FILTER(>0)`, `REVERSE`,
and `TAKE(2)`. Hold out one ordered pair while retaining the individual operations
and the reverse order in training. Evaluate exact list equality and intermediate
lists.

The official generator can derive admissible input ranges from a program. That is a
potential program-identity leak for an execution diagnostic. Use a common sampler,
fixed overflow/empty-list semantics, and deliberately include boundary cases.
Baselines are an exact DSL executor, a trivial empty/copy baseline, and a matched
small GRU.

This is still CPU-sized for lists of at most eight elements and programs of at most
six operations, but variable-length state and empty outputs make diagnosis less
clean than the register task.

## What requires operation choice?

There are three separable levels:

1. The opcode is supplied at each step: the model must apply the selected
   transformation to its current state.
2. The program is stored in memory and the model must read the next instruction:
   this additionally tests addressing and program-counter advancement.
3. The next instruction depends on computed state: this tests conditional control
   flow.

The first level does not require a learned router. A shared recurrent transition can
be conditioned on an opcode. It is still useful because it tests whether the model
can preserve state and compose non-commuting transformations. A learned router is a
separate hypothesis and should be compared in a new, matched arm.

Without a router, the following are already testable:

* exact transfer to an unseen ordered pair;
* sensitivity to operation order;
* correctness of every intermediate state;
* causal replacement of one operation after a shared prefix;
* interaction between pair generalisation and longer execution.

Passing these tests does not prove that individual reused blocks became distinct
operation experts. That requires block-level interventions or a controlled router
comparison.

## Minimal experiment to run first

Use the register interpreter with `ADD/XOR/SWAP` and explicit opcode input. Keep the
four-block recurrent schedule and QAT settings fixed. Do not feed predicted
intermediate states back into the model; intermediate targets are loss/evaluation
only.

Before training, freeze the complete program universe and split. Enumerate semantic
signatures over all 256 initial states and mark equivalent programs. Keep train,
validation, ordinary test, and held-out-program test disjoint at the
`(program, initial-state)` level. Select checkpoints only by validation.

Run a seed-0 gate first. It must solve primitive operations and familiar
compositions; otherwise stop without tuning on the held-out test. If it passes, run
seeds 1 and 2 at the same budget. Report final and intermediate exact accuracy,
per-program results, order-swapped pairs, and deterministic order-blind controls.

Interpretation is deliberately narrow:

* **Pass:** the fixed core executes known operations and transfers to the registered
  unseen ordered-pair family under an explicit opcode/time scaffold.
* **Fail:** no claim of router necessity; report whether the failure is primitive
  execution, familiar composition, or only the held-out pair.
* **Either outcome:** does not establish arbitrary-depth generalisation, autonomous
  program reading, or language reasoning.

The concrete affine `A/B/C/D` variant already drafted for E15 is a compatible
implementation choice. This note recommends the register family because its small
state space makes semantic-equivalence audits, exact intermediate checks, and
shortcut controls unusually cheap.

## Sources and status of claims

Published evidence is limited to the task definitions, evaluation distinctions, and
open implementations linked above. The register operations, split, thresholds,
baselines, and CPU budget in this note are project proposals. No result in this file
should be read as a completed experiment or as evidence that the current QAT model
has already passed an unseen-operation-combination test.

Related concrete protocol: [E15 operation-composition proposal](../results/E15_OPERATION_COMPOSITION_PROPOSAL.md).
