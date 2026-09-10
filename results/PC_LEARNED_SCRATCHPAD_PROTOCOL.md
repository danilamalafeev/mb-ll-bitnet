# PC learned scratchpad v1 — bounded prospective design

2026-09-09. Design only: no data generation, checkpoint load, model forward or training performed. Root owns shared coordination. Executor receives the small pure interface first; independent reviewer clears it before runtime expansion. The accepted explicit-register experiment remains immutable and the old replay remains cancelled.

## Question, parent and scope

Can additional training improve a continuous, self-written register bottleneck after temporary hidden state and KV are reset at every instruction? Use only float h128 seed0 B/u40000, accepted E36 checkpoint `runs/e35_e36_length_wave/science/float128_seed0/B/u40000.pt`, SHA256 `59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9`.

Two children start independently from this exact parent's model, optimizer and RNG: `continuous_control` and `soft_register_reset`. Both receive exactly2000 additional updates, batch64, the identical frozen first2000 batches of `wave.build_streams()[1]`, in original order. This deliberately repeats an already trained B-stream prefix; it is a matched diagnostic continuation, not newly sampled training. Preserve the existing optimizer type, hyperparameters, per-parameter states, loss reduction and gradient clipping. New absolute endpoint update is42000; local updates0–2000 do not imply the old8000-batch schedule has been extended or changed.

No new parameters, W4, extra seeds, extra budgets, hyperparameter/temperature sweep, learned routing, teacher forcing, hard argmax/straight-through training or baseline model rerun. This is a narrow learned continuous bottleneck, not an arbitrary scratchpad or a language-model result.

## Exact architecture

The parent uses `WidthSignedBitsEncoder`: each register encoder is a bias-free linear projection from four signed bits. Let fixed matrix S have16 rows, one per integer0–15 in ascending order, four least-significant-bit-first entries in {-1,+1}. Maintain p_x,p_y with shape(batch,16), initialized once to one-hot true initial registers. For every instruction:

1. Compute expected signed bits b_x=p_x S and b_y=p_y S. Apply the existing respective projection weights to get xv and yv. Do **not** average already normalized memories: projection precedes the existing nonlinear memory normalization.
2. Construct a brand-new cache exactly in the form of `WidthRegisterModel.init_cache`: h=xv+yv; x_initial=xv; y_initial=yv; key_memory=memory_norm(role_keys), broadcast over batch; value_memory=memory_norm(stack(xv,yv)); kv=reader.build_kv(key_memory,value_memory); substeps=0.
3. Call the unchanged `model.step(cache, opcode)` once, retaining its eight native steps and output normalization/heads. NUM_BLOCKS=4 and native_steps=8, so the reset does not change block-cycle phase.
4. Emit the returned x/y logits for the existing per-step loss/decoded evaluation. Write p_x=softmax(logits_x,dim=-1), p_y=softmax(logits_y,dim=-1), temperature exactly1. Pass only these own probability tensors to the next fresh cache. Discard the returned h/KV/cache; no reference to original registers after initial one-hot creation.

Do not detach the probabilities or cache-construction graph. Later losses must differentiate through earlier writes and memory reads. Numerical h/KV values do not bypass the scratchpad; autograd retaining the earlier computation graph is necessary and is not a recurrent-state bypass. No target argument belongs in the forward/write/cache interface. Targets enter only the unchanged summed x/y cross-entropies, each averaged over batch and instruction positions. Existing `_update` semantics remain zero_grad, finite loss, backward, clip_grad_norm_(1,error_if_nonfinite=True), optimizer.step.

**Scientific correction:** although stored as2×16 probabilities, the next instruction sees only eight expected signed-bit values through the current linear encoders. Distinct distributions with identical bit expectations are indistinguishable to the next computation. Temperature1 soft distributions may produce novel continuous memory inputs; they are not equivalent to marginalizing the nonlinear discrete interpreter. This is intentional and must be named in reporting. The existing heads are both supervised outputs and writes; there is no separate write head or new memory capacity.

## Evaluation and attribution

Run exactly three scientific evaluations, each the same saved PC45 padding+24 composition programs, all256 initial states, unchanged IDs/strings:

- Parent weights with soft-register-reset inference, before any child update:69 full-program forwards.
- Continuous-control child after2000 updates:69 forwards.
- Soft-register-reset child after2000 updates:69 forwards.

Reuse original saved continuous float seed0 B predictions as parent-control baseline; hash and join them, never regenerate them. The initial soft evaluation is indispensable: compare initial soft versus saved parent continuous to measure inference-path change, final soft versus initial soft for within-arm training change, final control versus saved parent continuous for ordinary continuation, and final soft versus final control for the matched endpoint comparison. A descriptive difference of changes can be reported, with no causal certainty from a single seed. Do not attribute all final improvement to scratchpad training. No intermediate evaluation or early stopping/selection on these opened PC programs.

At all evaluations decode logits only for metrics; the soft arm continues to feed its own probabilities, never argmax values. No256-state transition-table shortcut: the recurrent state is continuous and trained changes may depend on uncertainty. Run full programs through the differentiable architecture (inference_mode for evaluation).

Report per-program and per-suite/length/state-stratum final and full-trace counts, paired recoveries/introductions/both-correct/both-wrong for the four comparisons above, first-error and later-recovery histograms, and exact denominators. Retain train192, validation32, test32 and heldout64 aggregate. Save all decoded/target traces and compact per-position writing entropy/bit-expectation deviation from true bits; these are descriptive diagnostics, not new optimization targets. Pooling cannot replace this one-seed arm distinction. The pool was opened before design; call this adaptive research, not independent heldout confirmation.

## Exact compute budget

Freeze the first2000 B-batch lengths before work: (1,2,3,4,5,6)×333 then1,2. Their sum is6996. Each arm has128000 training examples and447744 instruction readouts, thus3581952 forward native steps. Two arms total4000 updates. One69-program evaluation has17664 cases,331776 readouts and2654208 forward native steps.

| Stage | Whole-program forward calls | Cases | Readout positions | Forward native steps | Optimizer updates |
|---|---:|---:|---:|---:|---:|
|Tiny QA, two paths|6|12|20|160|6|
|Continuous control training|2000|128000|447744|3581952|2000|
|Soft-reset training|2000|128000|447744|3581952|2000|
|Initial soft evaluation|69|17664|331776|2654208|0|
|Final control evaluation|69|17664|331776|2654208|0|
|Final soft evaluation|69|17664|331776|2654208|0|
|Total including QA|4213|309004|1890836|15126688|4006|

Science alone:4207 calls,308992 cases,1890816 readouts,15126528 forward native steps,4000updates. Backward/optimizer work is additional compute; forward-native-step counts are not total FLOPs or training runtime. A full-program forward means a batch call producing every instruction's logits, not each internal `step`; separately count actual step invocations to prevent ambiguity. Fixed batches need no chunking; any changed batch/forward budget requires a reviewed superseding contract before execution. Count failed attempted work separately, not zero.

Tiny QA is deliberately separate training on disposable children, never scientific descendants. For each architecture use the first two states/examples of accepted B batches0 (L1) and1 (L2): update batch0 once, save snapshot, update batch1 uninterrupted, reload snapshot and update the identical batch1 once. That is3 attempted/completed updates per path,2 cases per call,2+4+4=10 positions/path. Compare exact next-update model, optimizer and RNG states between uninterrupted and reloaded paths. Accepted inference QA is not rerun. Pure fixtures cover soft math and gradients before this QA; no additional real-model equivalence forwards.

Use CUDA float32, no autocast/precision tuning, deterministic algorithms and4CPU threads as in accepted PC work. Retain and freeze actual TF32/matmul settings, versions and device. Current `dsl.loss_for_batch` constructs CPU tensors; the **only** required training adapter moves the already specified input/target tensors to the model device while preserving the exact loss formula/reduction. Move restored optimizer-state tensors consistently to CUDA. Check this bridge and reload equality in the registered tiny QA, not an extra migration sweep. Preserve parent's CPU RNG, and freeze identical initial CUDA RNG state for both scientific children; no parent-derived CUDA RNG exists to invent.

## First executor deliverable: small contract and independent pure fixtures

Do not implement the full runner first. Expose only:

- `signed_bit_matrix()`; `cache_from_probabilities(model, px, py)`; `soft_register_forward(model,x,y,ops)` returning ordinary per-step logits with optional diagnostic writes. No global monkeypatch, original-model source edit or new parameters.
- A device-only equivalent of accepted batch CE and a child identity record: parent checkpoint/model/optimizer/RNG digests, architecture ID, local/absolute update, fixed stream+target digest, source/config digest. State dict parameter names/order must remain identical so the parent optimizer attaches to the correct parameters. Prefer a forwarding adapter with no separately registered copy of the model.
- Manual pure fixtures: signed-bit rows for0,1,15; one-hot projection/cache equality against independently constructed expected matrices; pS→projection equals p×projected-discrete-rows within a specified float32 numerical tolerance, not a false bitwise claim. Normalize only after averaging embeddings. Distribution pairs with the same bit expectations give the same next memory.
- No-bypass fixture: deliberately different discarded prior h/KV with identical p/op produce identical next-step inputs; poisoned originals after initial creation cannot affect later input; deliberately wrong writes propagate their own values. No target input and no use of targets to create writes.
- True cross-instruction gradient fixture: a hand-controlled two-op differentiable toy core, loss **only on the final instruction**, nonzero finite gradient to the retained first-write logits; detached-write negative control loses that path. Merely checking that head parameters have gradients is insufficient because final loss also uses the heads directly. Validate softmax temperature1 and that later-loss gradient is not accidentally restored via h/KV.
- Exact total/partial accounting, parent/arm/update/config and digest-mutation rejection; final/full recovered/regression/tie metrics with earlier-error/final-recovery distinction. No generic framework.

Stop this first deliverable at pure tests and source review. After PURE CLEAR, implement only the tiny-QA paths and serialization needed for exact reload equality. After saved QA ACCEPT and exact source CODE CLEAR, freeze one scientific input/source manifest and validate complete inputs once. A thin fresh checkpoint schema must preserve model, optimizer, CPU/CUDA RNG and identity metadata; frozen manifest never gets generated paths/status appended.

## Launch, artifacts and stop

Fresh versioned `runs/pc_learned_scratchpad_v1/{qa,science}`; new script/helpers/tests only. Root launches long QA and science through the reviewed detached supervisor with one completion wake per registered stage, no polling. Science validates one frozen parent/stream/protocol/source/baseline manifest, loads parent once per child via the strict E36 loader that returns the optimizer (the inference convenience loader discards it and is unsuitable), evaluates initial soft on its own loaded parent with state/RNG/mode preservation, trains each child once and evaluates only final endpoints. No repeat whole-history loading per update. Count requested parent loads and every underlying deserialize, snapshot load, backward and optimizer attempt/completion separately. Record source unchanged at execution, input hashes, CUDA settings, timers and partial failures; no retries on failure.

Save final child checkpoints, immutable lineage, loss/accounting trace, all207 evaluation rows, decoder targets/predictions and sufficient scratchpad diagnostics. Final independent verification is a saved-data audit only: DSL traces/joins/metrics, identity/accounting/digests and saved pure/QA evidence. It is not a replay of learned logits or training updates. Stop after this fixed2000-update paired comparison and report, regardless of improvement, tie or regression. Any additional training, parameters, memory width, harder task or seed is a new scope.
