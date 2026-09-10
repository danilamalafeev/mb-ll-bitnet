# PC latent slots v1 — prospective bounded pilot

2026-09-09. Design only; no code, checkpoint deserialization, model construction, forward or training in this review. This supersedes no old result: explicit registers remain accepted; interrupted soft-register paired training remains incomplete and is not resumed.

## Question and fixed scope

Can the existing interpreter learn to maintain task information through two unconstrained continuous slots when temporary hidden state and KV are reconstructed every instruction? This is learned storage with fixed opcode control, not learned routing. Use only float h128 seed0 B/u40000, checkpoint runs/e35_e36_length_wave/science/float128_seed0/B/u40000.pt, SHA256 59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9.

Choose ONE latent-slot child, 2000 updates, batch64; evaluate its initial and final states once each. Reuse the accepted saved initial-soft predictions (100% on all69 programs/all256 states) as a descriptive structured-memory reference. Do not retrain that ceiling-level reference, rerun continuous control, resume the failed soft arm, or add seeds/W4/sweeps. This is a feasibility/attainment pilot, not a matched training or parameter-efficiency comparison. A claim that latent slots train better than soft registers is outside scope. This smaller first pilot is justified by the already perfect saved reference and the new randomly initialized interface's distinct attainment question.

## Exact interface and initialization

Maintain z of shape [batch,2,16], real float32, with no probability simplex, bit encoding, numeric target, quantization or bounded activation imposed on its contents. Exactly three new modules, 6464 trainable parameters:

- Initializer I: Linear(8,32,bias=True), 288 parameters. Once per program, concatenate the true initial x and y four signed bits each, LSB first, and reshape I(input) to two slots. Numeric encoding is allowed only at this external input boundary. It does not specify later slot meanings.
- Shared reader R: Linear(16,128,bias=False), 2048 parameters; apply independently to each slot.
- Separate writer W: Linear(128,32,bias=True), 4128 parameters. After each instruction write reshape(W(model.output_norm(returned_cache['h'])),[batch,2,16]). This writer consumes hidden features directly, never numeric logits, softmax probabilities, decoded numbers, targets or oracle states.

Initialize each new weight independently with normal mean0/std0.02, biases zero, in fixed order I,R,W using one local CPU generator seed0. Copy initialized float32 tensors to the model device; preserve parent's global CPU/CUDA RNG. No initialization search, factorization of numeric heads, decoded-number initialization, learned slot supervision or hand-coded copy operation. Save initialized adapter bytes/digests before science. The slots are architecturally unrestricted vectors, although any learned solution may naturally encode registers.

For each opcode, compute v=R(z). Construct a NEW cache: h=v[:,0]+v[:,1]; x_initial=v[:,0], y_initial=v[:,1]; substeps=0; keys=model.memory_norm(model.role_keys), broadcast over batch; values=model.memory_norm(v); kv=model.reader.build_kv(keys,values) with autocast disabled. Invoke unchanged model.step once (native_steps8, four-block cycle). Emit its ordinary x/y logits. Compute W from its returned normalized hidden state and discard that cache. The next opcode reads only new z and fixed model parameters. Initializer runs once; original x/y are never read again. Reconstructed h is a function of slots, not a carried hidden-state path. Rebuilding KV must preserve gradients to z. No prior h/KV, residual cache, original-register shortcut, write detach or target argument belongs in this interface. Autograd graph retention across instructions is necessary and is not a state bypass.

All parent parameters and all new parameters are trainable. Restore the exact accepted parent optimizer with existing parameter order/state/hyperparameters unchanged. Append one new parameter group containing I,R,W in named order, copying the original group's optimizer hyperparameters; reject ambiguous multiple original groups instead of guessing. New parameters start with absent/zero optimizer moments as usual; do not fabricate 40000-step moments. Record this asymmetry. Parent x/y input encoders are unused by this latent path and receive no gradients; do not add dummy losses to activate them. Clip the combined parent+adapter parameter gradient norm at1 with error_if_nonfinite=True. Existing per-position x/y CE (each mean over batch×positions, then summed), with deterministic-compatible 2D flatten(-1,16), is the only training objective.

## Stream, cost and endpoint interpretation

Freeze the identical accepted first2000 B batches from wave.build_streams()[1]: lengths (1,2,3,4,5,6) repeated333 times, then1,2; sum6996. Preserve all programs, states, targets and order. This repeats a parent-trained prefix; it is diagnostic continuation, not newly sampled data. Final identity is parent absolute update42000/local2000, architecture latent_slots_2x16_v1. No selection or intermediate evaluation.

| Stage | Batch/program forwards | Cases | Instruction positions | Forward native steps | Updates |
|---|---:|---:|---:|---:|---:|
|Disposable QA|3|6|10|80|3|
|Latent training|2000|128000|447744|3581952|2000|
|Initial latent evaluation|69|17664|331776|2654208|0|
|Final latent evaluation|69|17664|331776|2654208|0|
|Science only|2138|163328|1111296|8890368|2000|
|QA plus science|2141|163334|1111306|8890448|2003|

Native steps count forward core work, not backward FLOPs, reader/writer overhead or wall time. Count actual step calls, backwards, optimizer calls, parent loads and all underlying deserializations independently. Historical failed/partial experiments stay separately accounted; they are not charged as zero or merged into this successful budget. No model replay in final review.

Both evaluations use the unchanged saved69 PC programs (45padding+24composition), all256 states, full own-slot trajectories. Initial evaluation is essential because random I/R/W can destroy parent operation accuracy. Distinguish initial architecture disruption, subsequent attainment on trained lengths1–6, and transfer at longer lengths through32. Improvement after2000 updates is not proof of a superior representation: parent+adapter both learn. The PC pool was already opened, so longer-length results are adaptive evidence, not fresh heldout confirmation or arbitrary-length guarantees.

Report per-program, suite/length and train192/validation32/test32/heldout64 metrics: final/full trace counts, first-error and first-subsequent-recovery position, and paired final/full tables for final latent versus initial latent, initial latent versus saved initial soft, final latent versus saved initial soft. Candidate-correct/reference-wrong is a repair; reverse is regression. Persist decoded and true DSL traces, exact program/state joins. Save per-position slot norm and finite-value diagnostics; do not call them bit error or semantic slot accuracy. Do not save all hidden states or expand into probes. Numeric readouts can be correct while the writer is poor; full own-state traces and cross-step-gradient fixtures are necessary.

## Small first deliverable and QA

New helper scripts/pc_latent_slots.py and a focused pure test file only. Implement I/R/W, initialize_slots, fresh_cache_from_slots and latent_slots_forward, plus identity/budget fixtures. No full runtime or model/checkpoint work in this first deliverable. Reuse accepted CE utility unchanged rather than fork its numerical formula. Required independent toy fixtures:

- Exact shapes, parameter count6464, named initialization order/repeatability and global RNG preservation. Manually computed nontrivial reader+LayerNorm oracle discriminates projection-before-normalization. Ordinary numeric logits are unchanged outputs of the supplied core, with writes taken from hidden features.
- Two runs with identical slots/opcode but poisoned discarded h/KV/original x/y produce identical next inputs. Poison original inputs during the first step to detect a hidden reread. Deliberately wrong slot writes must be consumed, not repaired by an oracle. Targets must be absent from forward/write signatures.
- Final-instruction-only toy loss has finite nonzero gradient to retained first-write slot tensor AND W parameters; detached-slot negative control eliminates the earlier-write path. Avoid a toy in which final loss also directly uses W. Verify eight native steps and substeps0 each instruction.
- Strict state/optimizer adapter-name association and mutation rejection, exact cyclic prefix/tail accounting, hand-computed paired metrics. Atomic checkpoint fixture must reject a partial temp file and select only a committed checkpoint with matching manifest/next batch index.

After PURE CLEAR, implement ONLY a disposable three-update QA: first two examples of B batch0/L1 update once; snapshot; B batch1/L2 update once uninterrupted; restore snapshot and repeat that same L2 update once. Total3calls/6cases/10positions/80steps/3updates. Exact comparison includes parent+I/R/W tensors, optimizer groups/moments, CPU/CUDA RNG, mode and parameter names/order. Save the snapshot and equality evidence. No extra real-model numerical probes. Finite loss/gradients required. QA child is discarded and never seeds science.

## Freeze, checkpoint recovery and stop

Fresh runs/pc_latent_slots_v1. Preserve accepted QA sources before runtime extension. Use strict accepted E36 optimizer-returning loader inside the historically accepted Windows compatibility adapter; no invented top-level label check. Before any model construction freeze protocol/source/helper/loader hashes, parent bytes, accepted QA bytes, saved structured-baseline raw bytes, full2000-batch stream,69 programs/state strata, all new-parameter initialization metadata and actual runtime settings in an immutable manifest. Runtime validates exact inventory and hashes; generated progress does not mutate the manifest.

CUDA float32, autocast off, deterministic algorithms enabled, four CPU threads, same accepted TF32/matmul and environment settings explicitly recorded. Preserve parent CPU RNG and a fixed recorded initial CUDA RNG. Evaluation must preserve model/adapter/optimizer/RNG/mode, including initial evaluation before training. Separate load/train/evaluation/save timers.

Commit a complete atomic checkpoint at local0 before initial evaluation and every250 completed updates through2000: nine checkpoints including0, eight after training milestones. Each includes parent+adapter states, optimizer/groups, CPU/CUDA RNG, mode, next batch index, local/absolute update and manifest lineage. Write a new temp file, flush and atomically rename on the same volume; keep previous committed files. No extra evaluation at checkpoints. Persist attempted/completed phase counters before and after forward/backward/update, plus loss/update ledger. A checkpoint is the recoverable boundary, not an optimistic accounting number; uncheckpointed work after it remains separately charged.

On interruption stop and preserve evidence; no automatic retry. A reviewed explicit continuation may resume only the latest complete matching checkpoint and original remaining stream. It must document any recomputed work after that checkpoint and consume a prospectively registered additional-cost budget; never silently repeat a whole arm or pretend the original ceiling still bounds actual work. Save cadence bounds possible lost progress to249 completed updates between checkpoints (250 if saving the next checkpoint itself fails). Never restore a checkpoint by reconstructing optimizer state from weights alone.

Root launches only after saved QA ACCEPT and exact CODE CLEAR, using one detached completion hook. Stop on nonfinite values, identity mismatch or exception; no rescue tuning. Stop after the fixed endpoint regardless of result. Final independent audit uses saved traces, hashes, fixtures, checkpoint metadata and accounting only; no model replay. Single-seed feasibility, unequal comparison training/capacity, opened evaluation pool and random interface disruption must remain explicit in the final conclusion.
