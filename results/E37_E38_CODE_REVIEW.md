# E37/E38 targeted code review

2026-09-08, followup_design Astra/low. **BLOCKED before model QA or science**, initial in-progress source inspection only. Executor is repairing the following concrete issues; this is not a gate for the eventual changed source.

- New branch loader hardcodes scientific parent32000/qa=False, rejecting bounded QA parent16002. Scientific and QA lineage must be strict separately.
- Scientific runtime parent-path/hash dictionaries are absent at start; first parent extension would finish before KeyError. Preserve immutable base manifest identity and separate appended generated-parent lineage.
- New strict loader must verify typed fixed updates/costs, config, initial identity, source/manifest/stream/target and RNG digests. A schedule-length mismatch must reject rather than skip expected-cost validation.
- Total training accounting must add four parent extensions plus four A and four B streams; equal updates do not imply equal processed positions.
- Persist actual attempted/completed costs for the currently failing loop; counters cannot disappear when an exception escapes.
- Bounded QA needs uninterrupted-versus-reloaded next-update model/optimizer/RNG exact identity, including snapshot continuation, using tiny primitive inputs and explicit total accounting.
- E37 must match accepted E35 eval-mode wrapper and restore previous mode.
- Before final reporting, assemble registered E38 primary paired comparisons and seed/precision conjunction from saved rows. No extra forwards required.

No scientific forwards/updates were executed by the reviewer. Accepted model-library paths were not reimplemented or broadly re-audited. Final CODE CLEAR requires repaired frozen source SHA256, passing targeted pure tests, prospectively capped QA command and saved actual QA accounting. Long QA must use the root detached completion workflow; no model wait loop.

## Repair review, before QA

Initial loader identity/runtime-map/eval-mode/training-total issues repaired in the inspected revision. Further blockers sent to executor: remove unused QA `branch_paths` mutation that changes checkpoint manifest digest after serialization; record actual QA7updates per model rather than double-counting the B second update; retain incremental failed-QA costs. Crucially, `_e36_primary_pair` had the directional inequality reversed: with left=A/right=B, improvement requires A-wrong/B-correct > A-correct/B-wrong. Require a pure benefit/regression/tie fixture before clearance. Planned bounded QA cap is28tiny ADD updates,56examples/positions,448native steps,0evaluation and0scientific updates. No model QA launched by reviewer.

## Bounded QA CLEAR — 2026-09-08

**QA CLEAR only; science remains BLOCKED pending successful saved QA and final freeze review.** Independently inspected narrowed Astra repair: correct B-benefit inequality; unused `branch_paths` mutations removed; `_qa_update` increments attempted/completed updates and actual hook costs across all training/replay calls, retaining partial costs on exceptions; CLI honors explicit preflight path. Repair agent reports14focused pure tests PASS9.37s, including both-budget benefit/regression/ties and failed-update accounting. Reviewer inspected fixtures and source, did not duplicate model QA or accepted library audits.

Reviewed source SHA256 `3cf3912622408b5682f7dd78988359a9f914d867168e973e69cffa5cd84acf5c`; tests `242dda93a38cdd3d5fd411a454c1917e0af282fa99f1cb8efc5d988a3f9cc3a6`; protocol `1725543ba5d9bf787bb8190d47fc2b8c6b4e9fdcadeafc292c20585f7975bbbb`.

Authorized fixed QA command after a fresh v2 preflight freezes these bytes:

```bash
python scripts/followup_e37_e38.py --qa --preflight-dir runs/e37_e38_preflight_v2 --out runs/e37_e38_qa_v1
```

Root must launch through the detached run-and-wake workflow and end the active turn; do not hold an agent waiting. Existing original preflight is audit-only and preserved. QA output must be new/nonempty-refused. Exact maximum if successful: four model identities ×(extension2+A2+B1+uninterruptedB1+snapshot-replay1)=28actual updates,56examples/positions,448native steps. Report's `training` bucket24updates/48positions/384steps plus `snapshot_next_update_identity`4updates/8positions/64steps; zero evaluation forwards and zero scientific updates. No automatic failed-QA repeat. Loads/deserializations and preparation costs are separate, not zero by implication.

Successful QA must have all four identities and snapshot-next-update/model/optimizer/RNG equality flags,28attempted/completed updates and the exact buckets above. Then independently verify manifest/source hashes and decide science CODE CLEAR in a subsequent gate.

Root selected fresh output `runs/e37_e38_qa_v2` for the same cleared command/cap; replace only `--out runs/e37_e38_qa_v1` with `--out runs/e37_e38_qa_v2`. This is prospective path registration, not authorization to run both paths or repeat QA.

## Saved QA ACCEPT; prospective science launch BLOCKED — 2026-09-08

**QA ACCEPT.** The one completed `runs/e37_e38_qa_v2` run has28attempted/28completed updates,56examples/readout positions,448native steps, zero evaluation and zero scientific updates. Every float/W4 seed1/2 identity reports strict parent/branch/snapshot reload, wrong-seed/parent rejection and next-update snapshot model/optimizer/RNG equality. Source SHA remains `3cf3912622408b5682f7dd78988359a9f914d867168e973e69cffa5cd84acf5c`.

Independent saved audit `results/E37_E38_QA_REVIEW.json` verifies all12checkpoint identities,32optimizer entries at each correct absolute update (parent16002,A16004,B_match16003), parent checkpoint hashes, checkpoint source provenance, and equality of QA manifest's immutable portion to v2. QA A filename `u16002.pt` is a QA-only naming inconsistency; payload and optimizer update16004 are correct. Reviewer used12checkpoint deserializations,0model constructions,0forwards,0updates; no repeated model QA or preflight. Frozen v2 manifest SHA `593cd1fb98ee180b13f968060a329d11936f61f514a45517b97d8227ffddb9b2`; QA report SHA `f5173db05a7839fd82a5c47a9de89de67049e9a0bebe15a9860fd1b18943d3fe`.

**Science launch BLOCKED on one prospective lifecycle change.** Current `run_e38` adds generated parent paths/hash dictionaries directly to a manifest copy and rewrites `e38/manifest.json`. Original preflight bytes are not overwritten, and the checkpoint base digest excludes those dictionaries; therefore this does not invalidate the completed QA. However the newly explicit immutable-manifest rule applies to future science. Root decision: grandfather completed QA/frozen v2 inputs only; require separate runtime lineage before science. Do not launch the current `--train-cleared` command. No new scientific failure or numerical finding exists.

### Narrow next repair contract (no implementation performed in this review)

1. Keep base manifest object and output `manifest.json` byte-content immutable throughout E38. Resolve generated-parent paths/hashes from a separate runtime lineage object/artifact explicitly bound to the immutable base manifest SHA/digest. Pass that lineage separately to new E38 save/load helpers; do not add it into the scientific manifest or accepted library schemas. Append parent entries once after successful save, reject replacing an existing entry, retain partial lineage on failure.
2. Keep all model operations, optimizer/RNG restore, training schedules, snapshot point, endpoint evaluators and budgets unchanged. This is serialization/provenance plumbing only. No architecture, numerical loader relaxation, stream generation change, new forwards or optimization work is authorized by this repair contract.
3. Add pure independent fixtures: base manifest bytes/digest unchanged before/after four synthetic parent registrations; correct child resolves only its own registered parent; wrong/missing/replaced parent/hash/base digest rejects; checkpoint base identity stable across sibling registration; failure preserves earlier lineage and actual counters. These fixtures must not call model factories or whole-manifest reconstruction.
4. Preserve source snapshot/hash and QA v2 evidence. Bind superseding source hashes and the lineage schema in a fresh versioned scientific reference derived from already verified frozen v2 inputs, explicitly referencing its SHA; do not regenerate pools or rerun expensive `make_manifest`. Independently verify only changed source/reference hashes and pure fixtures. Retain accepted QA as evidence for the unchanged numerical/save-reload behavior; new pure tests establish the separate-lineage interface. Do not automatically repeat model QA or preflight. If repair expands into numerical/save-reload semantics, return the concrete expansion to review before any computation.
5. Independent reviewer issues exact-hash science CODE CLEAR only after this narrow patch and fixtures. Then one detached launch of the original128000-update E37/E38 scope, one completion wake and final review/STOP.

### Delay finding and accounting limits

The supervisor interval15:10:54→15:34:25UTC is1411seconds for a command that FIRST waited for an already-running preflight, THEN ran QA. It is not an isolated QA timer. No per-stage profile was saved, so the measured time attributable to model QA, preflight wait, loading and hashing is unknown.

Static call-path inspection identifies substantial avoidable preparation work: new `_load_e32_manifest` calls accepted `scripts.width_e32.load_manifest`, which reconstructs the full E32 manifest/protected-reference checks and validates all six initializations. Its code invokes18initial model factories per call (12in `make_manifest`,6in the saved-initial checks). The successful QA control path invokes this loader an inferred32times via two initial parent loads and six recursive load routes per model identity, hence an inferred576initial factory calls before additional checkpoint factories. These are static path counts, NOT instrumented timing or complete deserialization accounting. Repeated full ancestral/cumulative stream hashing adds work. The cost buckets28updates/448steps omit these preparations by design; exact preparation/load duration and total deserialization count were not measured and must remain unknown. This diagnosis does not justify retroactive reruns or broad accepted-code edits. A later performance change requires its own bounded contract.

Workflow ordering was historically out of the newly corrected order: full runner and full preflight preceded tiny QA, and background used a preflight-file wait bridge. Record that fact; do not rewrite history or invalidate accepted numerical QA merely because workflow policy changed. The present review meets saved QA/hash audit and identifies the one remaining prospective gate.

## Science CODE CLEAR — runtime lineage v2 / source reference v3

**CODE CLEAR for the original fixed E37/E38 scientific scope.** User authorized the narrow prospective lifecycle repair. Independent contract gate preceded the patch; subsequent review found and closed the one branch-failure status collision. Final `_persist_failed_progress` keys branch failures as `<model>/<branch>`, preserving parent `<model>` counters. The added pure temporary-directory fixture and wrong-seed/label/replacement fixtures exercise these paths. Executor reports18pure tests PASS8.48s,0model runs; reviewer inspected the final changed lines/fixtures and verified current source/reference/base/accepted-QA hashes without repeating model QA or preflight.

- Runner SHA256 `fede3bb21c1a806c9e10068b1b82afdffb57cec00874f30dfa0bd666a4dfe0b5`.
- Tests SHA256 `3e9cee635f8d63f1c77aaa9b58f857e99dd412de5a49b40f23a5c5f167faaf4b`.
- `results/E37_E38_SCIENCE_REFERENCE_V3.json` SHA256 `d02e663f717953b224f56b2081dca5269909612bd89afcd8b7e945a7df4d8dd0`.
- Immutable existing v2 manifest SHA256 remains `593cd1fb98ee180b13f968060a329d11936f61f514a45517b97d8227ffddb9b2`, canonical digest `42bd72d9a4f2e7119282278c9e96eb40b888117eb1f4b59de63076b4dff5a956`.

The exact launch-compatible command is:

```bash
PYTHONPATH=. .venv/bin/python scripts/followup_e37_e38.py --train-cleared --preflight-dir runs/e37_e38_preflight_v2 --out runs/e37_e38_science_v3
```

Root must wrap it in one `run_and_wake.py` supervisor for the existing authorized task, verify detached start, then end the active turn. `runs/e37_e38_science_v3` did not exist at review. The fixed runner executes E37 followed by E38;128000new updates total, no duplicate B-match training, no additional science, no QA/preflight rerun. The separately validated source reference is loaded automatically from its fixed path above; it must remain unchanged through science/replay.

Both experiment output manifests are byte-preserving copies of frozen v2. E38 generated parents are append-once entries in separate `e38/runtime_lineage.json`, bound to base byte hash/canonical digest; child resolution verifies own registered file hash, strict arm/seed/update and checkpoint lineage. Runtime status/counters live separately from manifest data. Checkpoint schema is now `e37_e38_followup_v2`, runtime schema `e37_e38_runtime_lineage_v2`. Strict replay must pass this base manifest plus saved runtime lineage and retain source reference v3; do not attempt to use old E36/E38-v1 checkpoint schema for new E38 outputs. E37 still uses unchanged strict accepted E36 loader.

The repair changes provenance plumbing, not training/evaluation operations. Review compared the prepatch source snapshot and verified AST equality of11unchanged computation/metric helpers, then inspected changed numerical call sites for unchanged schedules, optimizer/RNG restore and snapshot/evaluation placement. Accepted QA v2 remains the model/optimizer/RNG continuity evidence; pure fixtures establish the new separate-lineage interface. Prepatch source `3cf39126...` and tests are preserved in `runs/e37_e38_manifest_repair_prepatch/source/`. No new model evidence is implied by this code gate.

The initial prospective reference candidate was refreshed before any science/code-clear to fix the review defect (earlier candidate runner `a8f1e723...`, reference `ccd813bd...`). No separate on-disk copy of that intermediate JSON was found in the registered repair directory; these identities and the prelaunch refresh are recorded here as audit history. It was never used for scientific execution, and accepted v2 inputs/QA were preserved. Final reference bytes above are frozen for launch; do not overwrite them afterward.

Final artifact validity/numerical replication remain unassessed until the one authorized run completes and the registered independent saved-artifact audit/replay returns ACCEPT/BLOCKED. Then STOP and discuss with user regardless of numerical outcome; no arbitrary-chain claim or automatic follow-up experiment.
