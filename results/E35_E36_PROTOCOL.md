# E35/E36 bounded first wave — protocol

Registered 2026-09-08 before scientific model execution. User launch authorization supersedes the proposal's earlier no-run status. Scientific owner/reviewer: length_wave_review (Astra/low); implementation/execution: length_wave_impl (Luna/xhigh); coordinator alone owns shared documents. Status: E35 and E36 have independent freeze/code gates; E35 can run after its own manifest and CODE CLEAR without waiting for E36 pools or implementation. E36 pool feasibility and CODE CLEAR pending. The frozen machine manifest will supply exact program lists, parent paths and hashes before science.

## Scope and hypotheses

E35 tests whether function-preserving longer histories change execution errors on existing accepted E33 seed0 h128 float and W4/A32 checkpoints at absolute update32000. E36 tests the package of longer training traces/compositions against continued short training. Four continuations only; no architecture changes, new seeds, branch C, second identity control or automatic extension. This finite DSL diagnostic does not establish general reasoning or an architectural limit.

## E35

Fix prefixes P=(ADD,ADD), (XOR,SWAP), (SWAP,XOR), each final opcode O in ADD/XOR/SWAP, and k=0,1,2,3 insertions of (SWAP,SWAP) between P and O. These give lengths3,5,7,9 and 36 programs per model, every one on all256 initial states. Verify complete DSL maps and state immediately before O agree across each family. Initial input/KV is unchanged; history and compute differ. Do not select cases using model answers.

Save every decoded two-register trace. Primary descriptive paired counts compare each padded version with k0: short-correct/long-wrong, short-wrong/long-correct, both-correct, both-wrong, for final operation and full trace. Report exact denominators. Supplemental O accuracy is conditional on both branches' immediately preceding readouts matching true state; report this subset's denominator and retain excluded cases in primary counts. SWAP identity controls cannot isolate h versus cache/reader interactions or generic history composition.

## E36 continuation and budget

Each precision's A/B restores its own accepted E33 absolute32000 parent including identical optimizer and Torch RNG state. Keep architecture, native8 internal steps/instruction, loss on both registers at every position, batch64, learning rate, clipping and optimizer unchanged. Use only the existing192 training initial states; never train on the other64. Training programs exclude adjacent ADD→XOR.

A length schedule is (1,2,3) repeated2666 times followed by (3,3): 8000updates and16002 batch-instructions. B is (1,2,3,4,5,6) repeated1333 times followed by (1,4): 8000updates,27998 batch-instructions, exactly4000 short and4000 long batches. B's jth short batch equals A's jth batch in programs and initial states. A reuses actual ancestral E33 batch contents in deterministic length order; B uses its first4000 batches for the common short stream. Thus A is a short-repeat control, not fresh independently sampled short data. Long batches use a fresh deterministic pool capped at32 programs per length, semantic representatives before aliases. Streams are frozen before model execution, shared across precision. Length-count rounding is explicit, no order tuning.

Save A8000, B4572 and B8000 (added updates). B4572 has16002 batch-instructions, exactly matching A8000: each has1,024,128 example-instructions and8,193,024 native internal steps. A/B8000 have512000 added examples each; B8000 has1,791,872 example-instructions and14,334,976 internal steps. Both-register target count is twice example-instructions. Equal-update and equal-instruction comparisons answer different questions.

Freeze small deterministic training program pools and their input schedule, retaining exact full lists and generation seeds in the manifest. Count unique syntax, program/input pairs, full-domain function signatures, opcode/pair exposure and states reached internally. Short syntax is finite: report actual diversity differences as intervention confounds; do not claim matched unique program/input coverage. Do not claim that trace length alone was isolated.

## Feasibility and evaluation freeze

Pure DSL feasibility precedes any scientific forward or training. Semantic exploration is capped at60seconds or100000 discovered function classes, whichever first; all selected functions are compared as exact maps over all256states. An incomplete search means unknown minimum length, never nonexistence of new functions. Candidate sampling can establish exact semantic disjointness without proving minimality.

Use deterministic capped evaluation pools at every length1–10, at most6programs per semantic/exposure category per length. Preserve actual counts, empty categories and opcode balance. Separate seen training syntax, heldout syntax with exposed semantic function, and heldout semantic functions. For the last category exclude the UNION of full-domain functions of ALL actually trained prefixes from the E33 ancestral stream and both A/B continuation streams. Keep allowed and ADD→XOR-containing programs separate. A common heldout pool is required for A/B comparisons. All selected programs use all256states, reported by seen initial192 versus heldout initial64. These are initial-state holdouts, not guaranteed unseen intermediate states. Historical six L5 programs are an explicitly open regression set, not blind evidence.

Freeze a training attainment probe of at most6 deterministically selected own actually trained programs per length on the192 training initial states, separately from heldout evaluations. Use the same branch probe across precisions and guarantee B probe programs have occurred before checkpoint4572. This measures sampled training attainment, not guaranteed mastery of every training example. Per length and stratum report final and full-trace correct/errors, first divergence position, next-step error conditioned on correct prefix with denominator, and per-program counts. Preserve decoded predictions and DSL targets, not only aggregates.

## Registered interpretation

Primary directional benefit is strictly fewer B full-trace errors than A on the common allowed semantic-heldout L7–10 pool, evaluated separately per precision at BOTH B8000 versus A8000 and B4572 versus A8000. Report every count and denominator; if a length has no available pool, omit it explicitly from the pooled denominator. If the whole primary pool is empty, this endpoint is unavailable. A zero-error baseline has no room for strict improvement. No25percent or1percentage-point gate is imposed.

Training attainment ≥99percent full-trace on the fixed own training probe is reported separately by length and pooled; B L4–6 attainment is the relevant adequacy check. Poor attainment makes an absent transfer benefit inconclusive about sufficient long training. Heldout L4–6 measures transfer within the trained length range, not training mastery. Short-length changes and historical regressions are descriptive. A benefit in both precisions warrants later confirmation, not a one-seed statistical comparison or automatic new run. Mixed/negative results end this wave at the same ceiling.

## Integrity, accounting, and stop

Freeze source/protocol/manifest hashes, exact parent checkpoint hashes and pool/schedule digests before science. Record parent/new/cumulative updates, examples, positions and internal steps; count QA, loads, evaluations and retries separately. Preserve old artifacts. One bounded executor QA precedes independent code review; scientific execution requires written CODE CLEAR. Final independent review reconstructs aggregates and targets from saved predictions, verifies parent lineage and costs, checks input hashes unchanged, and performs one strict saved-checkpoint evaluation replay with exact prediction agreement (no training). Stop after fixed outputs and explicit ACCEPT/BLOCKED artifact review. Repair only concrete correctness failures; no result-driven tuning or budget extension.
