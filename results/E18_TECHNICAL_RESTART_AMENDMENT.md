# E18 prospective technical restart amendment

Owner: root, 2026-09-07. Written before the repaired scientific run.

The original single-attempt stop is amended solely to finish the user-authorized comparison after an implementation failure. The first attempt made exactly one optimizer update on the first legal length1 batch (64 examples, 256 internal state updates), then raised NameError for the missing PROGRESS_INTERVAL import. No checkpoint, validation measurement, model selection or steps8 training exists from that attempt. This is a technical restart, not an independent replication or outcome-driven extension.

Preserve runs/e18_step_budget/ and runs/e18_step_budget_preflight/ unchanged. Exact original sources and protocol are retained in results/E18_FIRST_ATTEMPT_SOURCE/; hashes and actual cost are recorded in results/E18_FIRST_ATTEMPT_ACCOUNTING.json. Apply only the missing progress-constant import after Astra/low demonstrates the failure and successful actual train_arm smoke. Recompute source hashes in a fresh preflight.

New preflight: runs/e18_step_budget_repaired_preflight/.
New scientific run: runs/e18_step_budget_repaired/.

All scientific protocol choices remain unchanged: seed0; native4 and native8; same initial151232 parameters; fixed2000 updates per arm; identical data, optimizer, objectives and legal evaluation sets. No reserved tests or new compositions. No reuse of first-attempt state. Total planned scientific training including failure is4001 updates,256064 examples,3072256 internal state updates. Tiny control work is separate.

Release gate: repaired runner hash, actual training-path smoke, independent repair verdict, unchanged138 prior protected files and unchanged failed-run/preflight bytes. Stop after this one repaired pair and independent results review; any new failure requires explicit diagnosis rather than blind retries. Root is sole training process and shared-doc writer.
