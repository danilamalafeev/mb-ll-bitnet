# R01 independent synthesis review — ACCEPT

Reviewer: independent Astra/low, 2026-09-07. One focused source/applicability pass over `report-source.md` and `evidence-ledger.md`, with targeted primary-source checks and existing local E25 evidence. No model forwards, training, model-code changes or shared-document edits. PDF layout is outside this review.

## Material claims verified

- **LSQ:** the paper's §2.3 initializes quantized networks from trained float networks and keeps first/last layers at8 bits; §2 defines four signed levels at2 bits, not ternary. Its ImageNet CNN evidence supports a learned-step-size mechanism, not demonstrated success for fresh training of this tiny recurrent model. The synthesis identifies W4/A32 and signed-activation choices as adaptations and accounts separately for warm-start cost. [Primary paper](https://arxiv.org/pdf/1902.08153).
- **TTQ:** learns positive and negative ternary scales and assignments; includes CNN training from scratch as well as fine-tuning. The report correctly treats it as a different weight-quantization codebook/backward package, rather than evidence that changing one scale alone fixes E25. [Primary paper](https://arxiv.org/pdf/1612.01064).
- **ProxQuant:** §4.2 contains binary/multibit LSTM experiments initialized from a pretrained full-precision LSTM. The updated synthesis now includes this warm-start limitation; it makes no claim that those recurrent results establish success here. [Primary paper](https://arxiv.org/pdf/1810.00861).
- **BitNet:** the2024 report explicitly describes matching its float baseline from3B scale; the2B4T report includes architecture and training choices beyond quantization. Neither supports direct extrapolation to151232 parameters and this arithmetic task. [b1.58](https://arxiv.org/html/2402.17764v1), [2B4T report](https://arxiv.org/html/2504.12285v1).
- **LoopQ:** the cited May8,2026 preprint is PTQ for recursive Transformers, with activation scaling, selected transforms, transition adapters and trajectory calibration. Appendix B.1 explicitly makes both transforms and quantized weights loop-dependent for selected groups. The report correctly distinguishes this larger intervention from shared-weight from-scratch ternary QAT, and avoids claiming monotonic error growth or a proven local remedy. [Primary preprint, especially §4–5 and Appendix B.1](https://arxiv.org/html/2605.16343v1).
- **PTQ distinctions:** the GPTQ approximate-second-order description, AWQ activation-informed weight-only description and SmoothQuant W8A8 description match their primary abstracts. These are separated from matched fresh QAT. [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438).
- **Local evidence:** E25 primary error counts, failed combined predicates and training-set errors agree with accepted local evidence. Persistent state/master storage and computation are already FP32; `int8_activation` dynamically derives per-vector scaling on every call. Retaining FP32 state is therefore not a new proposed fix. No packed-kernel or speed benefit is inferred.

## Corrections resolved

1. LoopQ's relevant selected transform/weight-sharing implementation locator was corrected to **Appendix B.1** in source and ledger.
2. ProxQuant's **pretrained float LSTM initialization** was added to the applicability caveat in source and ledger.

Both changes were verified in the updated files. No remaining material source mismatch or unsupported empirical claim was found.

## Recommendation and scope

W4 learned-scale QAT and TTQ are explicitly recommendations with uncertain local benefit, not established winners. The proposed activation-off ablation changes one component of E25 and can test its contribution within the current ternary recipe. Its limits concerning the full weight×activation interaction and missing float-weight/A8 arm are correctly stated. E26 is proposed only: no registered protocol, implementation, training or new result is claimed.

ACCEPT applies to this bounded synthesis and its clearly labeled inferences. It is not approval of an E26 training protocol or evidence that any candidate will improve performance. No further search or experiment is needed to complete this review.
