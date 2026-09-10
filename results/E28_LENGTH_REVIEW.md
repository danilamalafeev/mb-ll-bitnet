# E28 independent final review — ACCEPT

Astra/low reviewer, 2026-09-08. The fixed experiment is valid; numerical transfer success across both lengths/all seeds is false for both model families. No blocking finding.

Independent saved-checkpoint replay completed all72 forwards, and the entire report exactly equals the canonical report, including every raw trace, metrics, paired counts, unchanged flags, predicates and costs. Canonical, preflight and replay manifests match. Six permodel/seed artifacts equal the report entries. Verified295 protected files, two source hashes and15 root reference hashes after replay; model/optimizer/RNG/mode unchanged in every evaluation. Strict accepted loaders and native8 carried-state path were separately inspected at the code gate.

A separate standard-library audit directly recomputed DSL traces with modulo16 ADD, XOR and SWAP; all18432 state/program cases, instruction correctness flags, final x/y/joint, fulltrace, perinstruction correctness, first observed divergence, recovery, initialstate strata, paired final/fulltrace outcomes and perlength/allseed conjunctions match. This audit does not call the production scorer. Symbolic selection was independently reproduced before inference, including exact novelty against all shorter functions and all3072 target traces.

All six selected length4 programs pass244/256 for every seed in both families. Length5 passes all six programs only for seed1 in each family; seeds0/2 fail. Therefore both model-wide primary conjunctions are false. Secondary fulltrace decisions have the same pattern, although a few cases recover by the final readout.

| Model | Seed | Length4 final errors /1536 | Length5 final errors /1536 | Length5 fulltrace errors /1536 |
|---|---:|---:|---:|---:|
| W4/A32 | 0 | 9 | 68 | 68 |
| W4/A32 | 1 | 0 | 21 | 22 |
| W4/A32 | 2 | 7 | 62 | 64 |
| Float | 0 | 14 | 138 | 141 |
| Float | 1 | 0 | 6 | 7 |
| Float | 2 | 1 | 64 | 64 |

These are the twelve selected functions at lengths4/5, not arbitrary-length generalization or an isolated causal effect of length: program length and composition content jointly change. Correct readouts do not establish hidden-state correctness. Initialstate strata are historical familiarity categories, not different program holdouts. Prior seen prerequisites are historical accepted E24/E27 results and were not re-evaluated. Neither a general W4 superiority claim nor packed hardware benefit follows.

Scientific run and independent replay each use72 forwards /18432 cases /82944 instruction readouts /663552 internal steps, zero updates. Independent replay wall time9.314711833s, exit0, no retry or scientific failure. Review tiny seen QA previously used6 forwards /12 cases /12 readouts /96 steps, separately recorded at the code gate. No other checkpoint probes or training.

Evidence: `runs/e28_length_review_final/replay_run/`, independent raw recomputation `runs/e28_length_review_final/recompute.py`, `runs/e28_length_review_final/independent_audit.json`, prospective `runs/e28_length_review_design/symbolic_audit.json`, and `results/E28_LENGTH_CODE_REVIEW.md`. Stop condition is met.
