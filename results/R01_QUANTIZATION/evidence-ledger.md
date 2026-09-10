# R01 evidence ledger / bounded search audit
Owner root; 2026-09-07. Canonical synthesis: report-source.md. No training or model inference.

## Plan and scope
Discovery -> applicability gaps -> targeted primary-source verification -> synthesis -> separate review -> PDF QA. update_plan unavailable; repository state/log used. One independent source lane (r01_qat_sources, Astra/low, fork none) on QAT/ternary optimization; root on recurrent PTQ. Stop after shortlist, caveats, and one discriminating next-test recommendation are supported.

## Search families used
- quantization recurrent weight sharing looped transformers quantization error recurrent neural networks mixed precision
- SmoothQuant AWQ GPTQ QAT low bit weights activation quantization learned step size LSQ paper
- LoopQ Rui Fang github quantization
- quantization recurrent neural networks ProxQuant learned step size weight only 4 bit recurrent
QAT source lane bounded to BitNet, LSQ, TTQ, ProxQuant. Social/search snippets excluded from evidence. No systematic completeness claim.

## Claim mapping and gaps
| Claim | Primary evidence / location | Confidence | Unresolved transfer |
|---|---|---|---|
| Current negative joint-QAT result | L1, final counts/predicates and independent replay review | High | Fixed data seed/open E21 |
| Current quantizer and FP32 state | L2, local implementation / protocol | High | No packed-kernel benefit |
| Learned quantizer step size, float warm-start | S1 §2–3 + Appendix B | High | CNN -> tiny shared recurrence; W4/A32 adaptation |
| Learned asymmetric ternary codebook | S2 §4–5 | High | Weight-only CNN, changes backward too |
| Proximal optimization has recurrent evidence | S3 §4.2 | High | Warm-start binary/multibit LSTM, not our ternary model |
| Native BitNet recipe and scale caveat | S4 §1–2; S5 §2–3 | High | Billions -> 151k, exact arithmetic |
| Recursive PTQ trajectory mechanisms | S6 experiments/limitations/Appendix B.1 | High for reported setup | Preprint; PTQ not QAT; transformed weights can depend on loop |
| PTQ families differ | S7/S8/S9 abstracts and papers | High | Off-the-shelf LLM tooling not verified compatible |
| Prefer W4 learned scale if quality first | Inference from S1 + L1, not reported outcome | Moderate | Requires prospective experiment |
| First disable A8 in ternary package | Local controlled-design inference | Moderate | Does not identify full 2x2 interaction |

No missing source gap requires another broad search. Remaining gap is experimental, not resolvable by literature. Official LoopQ code compatibility not verified; no claim of availability or deployment readiness. Sources S1–S9 URLs and bibliography are in canonical report. No verbatim quotations used.
