# PC explicit registers v1 — prospective contract

2026-09-09. User authorized explicit registers. Root/sole shared-doc writer: 01a085b8-7f43-7d31-a17a-3835fe3e88da. Executor pc_repair Luna/xhigh; independent contract/code/raw reviewer pc_review Astra/low. Old PC replay remains cancelled and is not part of this experiment.

## Question and fixed scope

Does execution through the model's own explicit two-register predictions improve the saved long-program results of current h128 B checkpoints? Six endpoints: float/W4 × seeds0/1/2, B absolute u40000, seed0 E36 and seeds1/2 E38. Preserve checkpoint/provenance bytes and existing Windows strict loading path. No A/B_match endpoints, training, weight edits, tuning, new program search or baseline model rerun.

Reuse all45 padding and24 composition programs in saved PC science, all256 initial states, unchanged IDs and DSL. Baseline is the independently audited saved continuous B predictions. The opened pool makes this an adaptive diagnostic, not a fresh heldout confirmation. Endpoint hash mismatch blocks use of the baseline.

## Explicit transition interface

For each checkpoint, make three fresh-context one-op evaluations: ADD, XOR and SWAP, each on the frozen order of all256 register pairs, CUDA float32/eval/native8. Store the model's own argmax pair T_op[x,y]; do not replace it with DSL truth. Compose T on its previous predicted pair for every instruction, including after errors. Targets are used only for metrics.

This exactly defines a finite discrete transition system from measured model outputs. It models decode/re-encode with fresh h and initial-register KV per operation. Do not claim unconditional bitwise equivalence to every possible online batch shape: fixed256-state table measurements and two-state QA use different shapes. No benchmark claim about the runtime of uncached sequential neural execution. Report table construction and lookup composition time separately. Reset changes both h and KV; benefit does not isolate hidden-state drift as the sole cause.

## Budget and gates

Pure independent fixtures first: manual ADD modulo16/XOR/SWAP examples; deliberately wrong transition must propagate its own output; recovered/introduced/tie/both-wrong metrics; first divergence and recovery; exact total/partial counters; identity/digest mutation rejection. Executor first exposes this small interface, reviewer clears before full orchestration.

Tiny QA after QA CLEAR: first float seed0 B checkpoint, initial states (0,0) and (15,1), ADD then XOR with OWN intermediate predictions, fresh context each operation. Maximum2 forwards,4 one-op cases,32 native steps,0updates. No numerical baseline replay. Save mode/state/checkpoint preservation and actual attempted/completed loads/calls. QA has a distinct output; no automatic retries.

After saved QA acceptance and exact-hash CODE CLEAR, freeze one source/input/settings manifest and perform one detached science run. Science six endpoint loads (count underlying deserializations separately),18 forwards,4608 one-op cases,36864 native steps,0updates. Derived chain evaluation:6×69×256=105984 program-state cases and1990656 instruction readouts, produced by lookup, not model forwards. Model updates are prohibited. Record failed attempts separately; unknown historical costs are not zero. Do not rebuild whole historical experiments. Frozen base metadata and runtime results remain separate.

Fresh paths in code project: scripts/pc_explicit_registers.py, tests/test_pc_explicit_registers.py, runs/pc_explicit_registers_v1/{qa,science}; supervisor directories outside these outputs, versioned and new. Root launches using existing reviewed scripts/run_and_wake.py and one completion message to the root task. No periodic polling or model replay. Preparation failures block dependent science; no blind retry.

## Measurements and decision

For each endpoint/program/length/suite and inherited initial-state stratum, retain final and full-trace counts, paired corrected/introduced/both-correct/both-wrong versus saved baseline, first error and recoveries. Record each primitive map's errors over256 states. Separately show correct numeric state carried versus local primitive correctness; do not confuse correction of earlier arithmetic mistakes with fresh-state execution accuracy. Preserve all cases, including failed prefixes. Pooling is descriptive and cannot replace seed/precision results.

Primary descriptive direction: per-endpoint full-trace error change over each complete suite; explicitly report improvements, regressions and ties, plus each length. No invented universal-success threshold. If maps are exact on all states, induction establishes exact composition within this finite table-defined DSL, not arbitrary language reasoning or continuous latent generalization.

Final verification is one independent audit of saved tables, independently recomposed traces, DSL targets, baseline joins, paired/stratum metrics, hashes and accounting. No additional model forwards/replay. Stop after fixed result/report and root interpretation. Any future training or architecture work requires a new scope.
