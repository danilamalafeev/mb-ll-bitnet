# E24 — fixed continuation from u8000 to u16000

Registered before implementation/scientific continuation, 2026-09-07. User authorized continuing all three seeds after E23. Root owns coordination/design/report; one Luna/xhigh executor and separate Astra/low reviewer. This is adaptive additional-training diagnosis on opened E21 data, not a fresh holdout or independent replication. E22 remains a valid negative result.

## Fixed conditions and budget

Resume seed0 runs/e20_longer_native8/u8000.pt and seeds1/2 runs/e22_replication/seed{1,2}/u8000.pt, using their accepted frozen manifests and loaders. Restore ALL model tensors, AdamW state including step8000, and stored torch RNG after any constructor calls. Preserve training mode and deterministic CPU4 float32 settings. Verify exact parent bytes/digests, dtype/shape, optimizer inventory/config, and parameter count151232. No fresh initialization, optimizer reset, changed learning rate, data, losses, architecture or clipping.

Each of seeds0,1,2 receives exactly8000 additional updates, absolute u8001..u16000. Same accepted E20 base2000 batches in order repeated4 additional times; continuation batch j is base[(8000+j-1)%2000]. Data seed0 remains fixed; total exposure eight repeats since initialization. Native8, batch64, same loss each instruction, same AdamW. Exactly24000 new updates/1536000 examples/24576000 internal training substeps overall. Do not count historical training again.

No intermediate scientific evaluation, checkpoint selection or early stopping. Save one final u16000 checkpoint per seed with complete model/optimizer/RNG, parent hash, manifest/source/config/data provenance and added/cumulative counters. Keep progress/loss every250 updates; no tuning from these. Both numerical failures and successes continue to remaining seeds. Technical failure suspends remaining work, records actual completed work, preserves all artifacts; no automatic retraining/retry or overwrite.

## Final observations and fixed predicates

At u16000 evaluate each seed once: same32 seen programs on train192/validation32, then E21 six primary programs plus separate equivalent control on all256 states, with complete intermediate instruction readouts and train/validation/test state strata. Always evaluate all three finals even if a seen prerequisite or primary predicate fails. Evaluation budget26880 program/state cases,70464 readouts,563712 internal substeps,213 program forwards. Restore mode and verify evaluation leaves model/optimizer/RNG unchanged.

Per seed report separately: seen prerequisite (each primitive32/32, each seen composition>=31/32); primary conjunction (EACH of six rows>=244/256); combined prerequisite AND primary. Overall primary success requires ALL THREE seeds' primary conjunctions, combined success all three combined. Equivalent ADD XOR XOR control is excluded. Historical thresholds unchanged.

Compare against frozen parent reports (seed0 E20 seen plus E21 composition, seeds1/2 E22 reports) without rerunning parents: per-program final correct delta, paired wins (parent wrong/new correct), losses (parent correct/new wrong), both correct/wrong on identical program-state identities, all primary and control separately; include intermediate/fulltrace metrics and seen deltas. Check source targets and strata identical. Secondary descriptive training benefit flag: EACH seed strictly reduces primary final error count relative to14/73/50. Report this separately from threshold success; neither substitutes for the other. No significance or independent-trial claims.

## Artifact integrity and checks

Only new scripts/continuation_e24.py, looped_bitnet/continuation_e24.py, tests/test_continuation_e24.py, results/E24_* and fresh runs/e24_continuation_preflight/ and runs/e24_continuation/. Do not edit old source, guards, runs or protocols. Freeze legacy files and parent/report/manifests SHA before work; freeze new source/protocol hashes in preflight after independent code CLEAR. Actual tiny three-seed runner must exercise restore/update/save/reload/evaluation/report, numerical failure continuation and technical failure recording using legal seen-only QA and private budgets. Check next-update identity of uninterrupted versus save/reload training for model/optimizer/RNG, wrong parent/seed/update/schema/config rejection and overwrite refusal. Keep QA work separately counted, no scientific predictions before code CLEAR. Root launches one fixed sequential scientific run after clearance, then separate independent result replay/count/provenance review.

Stop after all three fixed finals, paired report and independent result review; no u24000 extension, seed replacement, architecture change, E19 launch or cloud. Update shared handoff/log on completion. Automation remains PAUSED.
