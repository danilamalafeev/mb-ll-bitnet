# AGENTS.md

## Experiment workflow — user correction, 2026-09-08

This section supersedes older workflow ordering below. Scientific difficulty does not excuse an oversized first implementation, late review, weak test oracles, or repeated coordination. Optimize the complete verified result, including preparation and repairs.

1. **Contract and independent oracles first.** Before a full runner, main Astra defines and checks the smallest changed interfaces, immutable inputs, parent/child identity, exact budgets, output paths and stop condition. Reuse accepted training/evaluation functions. The first executor deliverable is the small changed contract and pure fixtures, not the complete experiment script. Luna verifies these against the independent oracles before orchestration expands; main Astra resolves scientific questions.
2. **Pure tests before model work.** Use hand-constructed expectations independent of the implementation: A/B benefit, regression and tie at each budget; wrong seed/parent/update/config rejection; digest stability and mutation rejection; snapshot identity metadata; exact total and partial-failure accounting. Test predicates themselves, not only report formatting or the same helper called twice. Missing checks block expansion.
3. **Tiny QA before full preflight.** Agree a minimal fixture and prospective attempted/completed update, example, step and load budget. With QA CLEAR, test one or two batches per necessary path and exact uninterrupted-versus-reloaded next-update equality for model, optimizer and RNG, including snapshot continuation. A small frozen QA fixture may identify existing parents; it must not require the expensive whole-experiment preflight. Keep QA and scientific outputs separate. No full training/evaluation sweep as a smoke test.
4. **CODE CLEAR, then canonical preflight.** Luna performs targeted implementation verification of the changed code, pure test evidence and saved tiny-QA results; records CODE CLEAR for exact source hashes. Main Astra checks scientific validity. A separate review is required only under Review Strategy; CODE CLEAR does not itself require another agent. Only then freeze the scientific manifest and validate complete inputs once. Check frozen hashes and scope against the cleared version before launch. A preparation failure blocks launch; investigate before any fresh attempt. Do not rerun accepted QA or preflight merely to generate another report.
5. **One detached scientific launch.** Run only the registered scope through one supervisor and completion wake. Long preflight, QA and replay also use this mechanism; never start a potentially long preparation synchronously and then poll it or add a file-wait bridge. On completion, perform one bounded Luna artifact verification/replay against independent expected results (a separate audit only under Review Strategy), write one result, update the short handoff, and stop at the registered boundary.

**Manifest lifecycle:** never mutate a frozen manifest to add generated parent paths, checkpoint hashes, runtime counters or status. Put append-only runtime lineage/accounting in separate artifacts tied to the immutable base digest. Change source or scientific scope through a fresh, explicitly superseding version; preserve old attempts and their costs. Coordination metadata is not scientific evidence. A process-policy edit alone does not invalidate a cleared code hash or justify repeating running QA. For work already in progress, record which new gates were met, which were historically out of order, and what remains; never retroactively claim compliance.

**Delegation discipline:** use one bounded executor when delegation is useful; add a second child only for genuinely independent work or a justified review. No standing reviewer role. Keep small staged deliverables with named files, acceptance fixtures and stop conditions. When a separate reviewer is justified, executor and reviewer exchange concrete defects directly; root receives milestone/blocker summaries instead of relaying every message. Do not repeatedly ask for unchanged status or resend context. A substantive defect gets a narrowed repair and targeted verification; further review follows Review Strategy, not an automatic loop. Do not expand scope while a gate is blocked. Read the current handoff and relevant changed symbols once; avoid reopening entire historical reports.

## Current model policy — user decision, 2026-09-10

This policy supersedes older model and automatic review/delegation recommendations in this file, handoffs and logs; preserve scientific contracts and evidence gates. It is a routing preference, not a verified pricing or capability claim. Optimize **research progress per weekly quota**, including verification and repairs.

- The user retains scope authority and final approval where required; there is no automatic model switch for an open root task.
- **Astra / low** (`gpt-6-astra`, `reasoning_effort="low"`) is the main research lead and orchestrator: retain project context, make architectural and research decisions, choose experiments and synthesize results. Solve simple reasoning locally. Do not spawn another Astra for routine design/audit already within the main agent's context.
- **Luna / xhigh** (`gpt-5.6-luna`, `reasoning_effort="xhigh"`) is the default for nontrivial implementation, debugging, bounded multi-file changes, targeted code review and implementing an accepted architectural decision.
- **Luna / medium or high** is the operational worker for fixed-scope experiment runs, repo exploration, code search, log aggregation, metric collection, simple edits and repetitive artifact/report preparation. Use medium for straightforward work and high when operational complexity warrants it; do not use xhigh when these suffice.
- Implementation verification belongs to Luna; scientific/architectural reasoning belongs to main Astra. Full independent review is conditional, as specified in Review Strategy.
- **Astra / medium+** is escalation only: contradictory experiment results, serious architectural uncertainty, deep comparison of plausible approaches, conceptual difficulty remaining after a reasonable bounded attempt, or a rare milestone-level scientific audit. Start at medium; increase only for a concrete unresolved need. Never use it for routine implementation/review, style, cleanup suggestions or optional refactors.
- Only root may spawn agents. Explicitly set model and effort; use `fork_turns="none"` and the existing handoff/context mechanism with only relevant context. Children must not spawn children. Default to at most 1–2 concurrent children; use none when local completion is cheaper and reliable.

## Long-running computations — user decision, 2026-09-08

Use the event-driven launcher documented in `docs/BACKGROUND_RUNS.md` for long training, evaluation and other standalone processes. Record the authorized scope, output paths, owning thread ID and next review step before launch. After confirming that the detached worker started, finish the active agent turn; do not keep the coordinator or executor in repeated wait/status loops merely while the process computes. The process completion hook queues one message to that exact existing thread on success or failure; resume artifact verification and reporting then. For parallel work, wrap the supervising command which waits for the whole authorized group, so it generates one completion wake rather than one model turn per worker.

Check the contract and independent pure test oracles early using the role split above. If a separate review is justified, return control when that bounded review is done; do not keep a reviewer active merely waiting for computed artifacts. Do not replace event completion with a periodically waking model or enable the old paused research heartbeat. Preserve normal scientific gates, output logs, exit status and checkpoint integrity. A completion notification is a trigger to inspect artifacts, not proof of scientific success and not authorization for another experiment. If notification delivery fails, retain its status/log and do not automatically retry or rerun the computation.

## Cross-chat coordination

Read `docs/PROJECT_VISION.md`, `docs/HANDOFF.md` and the latest entries in `docs/RESEARCH_LOG.md` before continuing project research. Keep the long-term architecture goal distinct from the pointer-chasing diagnostic. Register the task owner, hypothesis, write scope and stop condition in the log before starting a new experiment. Preserve existing runs. After completion, record measured results, artifact paths, verification and limitations, and update the handoff's current state. Use one coordinator as the writer of these shared coordination files when multiple chats work concurrently. Update the vision only when the project goal or architectural hypothesis deliberately changes. Do not repeat completed experiments or treat research suggestions as implemented features. Clearly distinguish planned work from executed work.

## Objective

Optimize for:

1. Correctness
2. Research progress per weekly quota, including verification and repairs
3. Minimal unnecessary repository exploration
4. Delegation of bounded tasks only when it adds value
5. Escalation to stronger models only when necessary

Use Astra/low as the research lead, Luna/xhigh for nontrivial implementation and Luna/medium-high for operations; reserve Astra/medium+ for justified escalation.

## Agent-experience tooling and cost control

Use the repository helpers in `scripts/agent_tooling.py` for coordination so
that handoffs do not depend on repeated manual instructions:

- run `skeleton` first when a delegated agent needs a quick map of a large
  file or directory; inspect full source only after locating relevant symbols;
- use `state` to record the active task, owner, status and latest checkpoint;
- use `log` at task start and completion for owner, scope, stop condition,
  results, artifacts, verification and limitations.

Keep this tooling outside the scientific path: it must not run inside training
or evaluation loops, change checkpoints/manifests, or be treated as evidence
for a model result. Prefer one narrowly scoped Luna worker; add an independent reviewer only under Review Strategy. Do not fan out duplicate investigations. Use the
cheapest model/effort within the current model policy, pass summaries and
file paths instead of full context, and escalate only for a concrete unresolved difficulty.

---

## Model Roles

### Coordinator

Default coordinator:

- Model: `gpt-6-astra`
- Reasoning effort: `low`

Use the coordinator primarily for:

- understanding the user request;
- breaking work into bounded tasks;
- deciding which files or areas need inspection;
- delegating independent work;
- integrating results;
- resolving conflicting subagent conclusions;
- final verification.

The coordinator owns project context and scientific decisions, and solves simple reasoning locally. Delegate substantial mechanical work when the benefit exceeds the context and coordination overhead; do not create an agent just to separate roles.

---

## Cost-Efficient Delegation

Apply the current model policy above. Narrow the task before delegation, reuse existing summaries/state/handoff documents, and select Luna/medium-high for operations or simple changes and Luna/xhigh for nontrivial implementation/debugging/review. Main Astra/low makes research decisions without a duplicate design agent.

---

## Strong-Model Escalation

### Escalation from Luna

After a failed attempt, narrow the defect with evidence and acceptance checks. Main Astra/low resolves ordinary reasoning blockers in its existing context. Escalate only the remaining conceptual question to Astra/medium when a reasonable bounded attempt leaves serious uncertainty; do not transfer routine repairs to expensive Astra. Historical E27/E30 escalation is not a standing route for later experiments.

### Astra

Astra/low owns normal research, planning and synthesis. Astra/medium+ is reserved for the explicit escalation cases in Current model policy. Pass only the disputed evidence, relevant context, failed approaches and decision required. A rare independent scientific audit starts at medium. Higher effort requires a specific unresolved difficulty and expected research value; return routine work to the normal routing afterward.

Do not spend Astra review passes on search, formatting, boilerplate, style suggestions or optional refactors.

---

## Default Execution Strategy

```text
User                  scope authority and final approval where required
Astra / low           project context, research decisions, specification, synthesis
    |
    +--> Luna / xhigh        nontrivial implementation/debugging
    +--> Luna / medium-high  bounded operations, experiments, simple edits
    |
    +--> targeted Luna verification --> result returns to main Astra / low

Conditional only: one full review for a meaningful diff/milestone, suspicious
result or high uncertainty; Astra / medium for a justified scientific audit.
```

Do not build an automatic main Astra -> design Astra -> Luna -> review Astra -> Luna fix -> review Astra chain. Simple work may be completed locally without any child agent.

---

## Fan-Out Limits

Default to at most **1–2 concurrent child agents**, and zero for tasks best completed locally. Only root may spawn; subagents must not spawn other subagents. Parallelize genuinely independent scopes, never roles alone or duplicate investigations. A second agent is optional, not a required reviewer slot.

---

## Delegation Rules

Every delegated task must be narrow.

A good delegation specifies:

- objective;
- relevant scope/files and pointers to existing context/handoff;
- whether modification is allowed;
- invariants and constraints;
- acceptance criteria and targeted verification;
- exactly what to return and expected output format;
- stopping condition.

Bad:

> Investigate the repo and fix auth.

Good:

> Inspect `src/auth/` and tests related to refresh-token rotation.
> Determine why expired refresh tokens sometimes remain valid.
> Do not modify files.
> Return:
> 1. root cause;
> 2. exact files/functions involved;
> 3. minimal proposed change;
> 4. tests that should fail before the fix.

---

## Context Discipline

Avoid loading unnecessary context.

Before reading large files:

1. search for the relevant symbol;
2. inspect nearby call sites;
3. read only the necessary ranges;
4. expand scope only when evidence requires it.

Do not recursively inspect unrelated directories.

Do not repeatedly reread files already summarized accurately by a subagent.

Before delegating, main Astra narrows the task using available context. Provide only the necessary existing summaries/state/handoff sections and file/symbol pointers; do not make workers reconstruct project history. Read large raw logs only when a specific unresolved question requires them; use summaries, filtered ranges and aggregated metrics first. Prefer targeted file reads and tests over full repo scans or full suites after small changes.

---

## Exploration Budget

Repository exploration should proceed in stages.

### Stage 1 — Locate

Use cheap search to identify:

- relevant files;
- entry points;
- important types/functions;
- tests;
- configuration.

### Stage 2 — Inspect

Read only likely-relevant code.

### Stage 3 — Reason

Form a hypothesis before expanding the search.

### Stage 4 — Expand only if needed

Expand repository scope only when the current hypothesis cannot explain the behavior.

Never begin with exhaustive repository reading.

---

## Escalation and Retry Policy

Start nontrivial implementation/debugging with Luna/xhigh and operations/simple edits with Luna/medium-high. After a failure, inspect the evidence, narrow the problem and retry only a concrete correction. Raise effort only if the task actually needs it; avoid automatic model ladders. Main Astra/low handles conceptual triage, using Astra/medium+ only under the escalation criteria above.

Preserve failed artifacts and account for retry costs. Targeted regression verification after a fix is required where relevant; it is not an automatic new full review. Follow Review Strategy for any second review.

---

## Parallelism Rules

Run agents in parallel only when tasks are independent.

Good parallel work:

```text
Agent A: implement the accepted API change in its assigned files
Agent B: aggregate existing experiment metrics in a disjoint scope
```

Bad parallel work:

```text
Agent A: understand entire bug
Agent B: understand entire bug
Agent C: understand entire bug
Agent D: understand entire bug
```

Duplicate full investigations waste context and credits.

---

## Implementation Policy

Before editing:

1. establish the likely root cause;
2. identify the smallest safe patch;
3. identify existing tests or expected behavior;
4. modify only necessary files.

Prefer minimal patches.

Avoid:

- unrelated cleanup;
- broad formatting changes;
- opportunistic refactoring;
- renaming unrelated symbols;
- touching generated files unless necessary.

---

## Testing Policy

Use the cheapest meaningful verification first.

Suggested order:

```text
targeted unit test
→ related test package
→ lint/typecheck for affected area
→ broader integration tests
→ full test suite
```

Do not run expensive full-repository verification after every small edit unless the repository requires it.

When a test fails, inspect the failure before blindly rerunning it.

---

## Review Strategy

- **Implementation verification — Luna:** run relevant tests, check acceptance criteria and invariants, and report evidence. Use xhigh for nontrivial targeted code review; medium/high suffices for mechanical checks. Small changes do not require a separate reviewer.
- **Scientific/architectural reasoning — main Astra/low:** interpret evidence and assess the accepted design in existing context; do not spawn routine design/audit Astra agents.
- **Full review — conditional:** use one bounded review after a meaningful diff or milestone, a suspicious result, or high uncertainty. Route code review to Luna and rare expensive independent scientific audits to Astra/medium under the escalation criteria.
- **One review by default.** A second review is allowed only if the first found a substantive correctness issue; restrict it to the repair and affected invariants. Do not automatically repeat review -> fix -> review after every small edit. Style, cleanup and optional refactors do not justify extra passes.

Check correctness, edge cases, regressions, scope, tests and relevant security implications. Supply the patch, intended behavior, invariant/acceptance checks and existing evidence; do not ask reviewers to rediscover the repository. Preserve independent scientific oracles, tiny QA, CODE CLEAR, immutable inputs and artifact integrity checks; these gates do not imply a standing separate Astra reviewer.

---

## Stop Conditions

Stop exploring when:

- the root cause is supported by concrete code evidence;
- the minimal patch is clear;
- relevant tests cover the behavior;
- no unresolved high-risk assumption remains.

Do not continue exploring merely to increase confidence from 95% to 99% on low-risk work.

---

## Token-Saving Rules

Always:

- summarize long findings instead of forwarding raw context;
- pass file paths and symbols instead of entire files where possible;
- avoid duplicate explanations across agents;
- avoid asking agents for long prose;
- request structured concise outputs;
- avoid maximum effort for simple tasks;
- avoid exhaustive full-repository scans.

Preferred subagent response format:

```text
Finding:
<1–3 sentences>

Evidence:
- file:line — explanation
- file:line — explanation

Recommendation:
<minimal change>

Risks:
<only meaningful risks>
```

---

## Default Decision Matrix

| Task | Model | Effort |
|---|---|---|
| Main research lead, orchestration, architecture, hypotheses, experiment selection, synthesis | Astra | low |
| Simple reasoning already within main context | Main Astra | low; no delegation |
| Nontrivial implementation/debugging, bounded multi-file edits, accepted-design implementation | Luna | xhigh |
| Targeted nontrivial code review/implementation verification | Luna | xhigh |
| Experiment execution, exploration/search, logs/metrics, simple edits, repetitive operations and mechanical verification | Luna | medium/high as needed |
| Contradictory evidence, serious architecture uncertainty, deep approach comparison, unresolved conceptual blocker | Astra | medium first; higher only if justified |
| Rare milestone-level independent scientific audit | Astra | medium |

---

## Coordinator Directive

The coordinator should continuously ask:

> Does a bounded delegation improve research progress per weekly quota after context, coordination and verification costs?

Delegate only if yes. Solve simple reasoning locally and reuse an existing suitable worker/context when useful; do not spawn agents merely because a cheaper model is available.

Also ask:

> Is additional reasoning likely to change the decision?

If no, stop escalating.

The objective is not maximum reasoning.

The objective is research progress per weekly quota, with enough reasoning and evidence for a correct, verified result.

---

## Recommended Default

- Main research lead/orchestrator: Astra/low; handles simple reasoning itself.
- Nontrivial implementation/debugging: one bounded Luna/xhigh worker when useful.
- Operations/simple changes: Luna/medium-high, matched to complexity.
- Verification: targeted Luna checks; main Astra synthesizes scientific results.
- Separate full review: conditional, one by default; second only for a substantive correctness issue found in the first.
- Astra/medium+: rare conceptual escalation or milestone scientific audit.
- At most 1–2 concurrent children by default, no nested spawning, explicit model/effort, minimal existing context, no full-history fork.

Measure economy by research progress per weekly quota, including verification, repairs and coordination. Preserve the existing handoff/state, experiment tracking, evidence gates and event-driven launch mechanism.
