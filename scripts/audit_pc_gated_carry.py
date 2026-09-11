"""Independent stdlib-only audit of saved gated-carry science rows.

This audit never imports torch or loads a checkpoint.  It reconstructs the DSL
targets, scope, strata, row joins, metrics and paired comparisons from JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


STATE_COUNT = 256
OLD_COUNT = 69
NEW_COUNT = 81
COMPOSITION_CASES = 24 * STATE_COUNT
SCIENCE_COMPUTED_CASES = 135936
SCIENCE_COMPUTED_POSITIONS = 2792448


def digest(value):
    """Match the project's canonical JSON digest without importing project code."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def states_and_strata():
    states = [(x, y) for x in range(16) for y in range(16)]
    ranked = sorted(states, key=lambda s: hashlib.sha256(f"E15-state-v1:{s[0]}:{s[1]}".encode()).hexdigest())
    return states, {state: ("train" if i < 192 else "validation" if i < 224 else "test") for i, state in enumerate(ranked)}


def truth(program, state):
    x, y = state
    trace = []
    for op in program:
        if op == "ADD":
            x = (x + y) % 16
        elif op == "XOR":
            x ^= y
        elif op == "SWAP":
            x, y = y, x
        else:
            raise ValueError(op)
        trace.append([x, y])
    return trace


def metrics(target, predicted):
    errors = [i + 1 for i, (a, b) in enumerate(zip(target, predicted)) if a != b]
    recovery = next((i + 1 for i in range((errors[0] if errors else len(target) + 1) - 1, len(target)) if target[i] == predicted[i]), None)
    return {
        "full_trace_correct": not errors,
        "joint_final_correct": bool(target and target[-1] == predicted[-1]),
        "first_error": errors[0] if errors else None,
        "first_subsequent_recovery": recovery,
        "errors": errors,
    }


def old_specs(scope):
    return {str(row["id"]): {"suite": row["suite"], "length": int(row["length"]), "program": tuple(row["program"])} for row in scope["programs"]}


def new_specs():
    prefixes = (("ADDADD", ("ADD", "ADD")), ("XORSWAP", ("XOR", "SWAP")), ("SWAPXOR", ("SWAP", "XOR")))
    cells = (("SSSS", ("SWAP",) * 4), ("XXXX", ("XOR",) * 4), ("SXXS", ("SWAP", "XOR", "XOR", "SWAP")))
    result = {}
    for prefix_name, prefix in prefixes:
        for cell_name, cell in cells:
            for repeat in (2, 4, 8):
                for suffix in ("ADD", "XOR", "SWAP"):
                    program = prefix + cell * repeat + (suffix,)
                    result[f"control_{prefix_name}_{cell_name}_r{repeat}_{suffix}"] = {"suite": "identity_controls", "length": len(program), "program": program, "cell": cell_name, "prefix": prefix_name, "repeat": repeat, "suffix": suffix}
    return result


def check_row(row, spec, arm, phase, states, strata):
    assert row["arm"] == arm and row["phase"] == phase
    assert row["suite"] == spec["suite"] and row["length"] == spec["length"] and tuple(row["program"]) == spec["program"]
    predictions = row["predictions"]
    assert len(predictions) == STATE_COUNT
    seen = set()
    for prediction in predictions:
        state = tuple(prediction["state"])
        assert state in strata and state not in seen
        seen.add(state)
        target = truth(spec["program"], state)
        assert prediction["stratum"] == strata[state] and prediction["target_trace"] == target
        predicted = prediction["predicted_trace"]
        assert len(predicted) == spec["length"] and all(len(pair) == 2 and all(isinstance(v, int) and 0 <= v < 16 for v in pair) for pair in predicted)
        expected = metrics(target, predicted)
        for key in ("full_trace_correct", "joint_final_correct", "first_error"):
            assert prediction[key] == expected[key], (row["id"], state, key, prediction[key], expected[key])
        if "first_subsequent_recovery" in prediction:
            assert prediction["first_subsequent_recovery"] == expected["first_subsequent_recovery"], (row["id"], state, "first_subsequent_recovery")
    assert seen == set(states)


def aggregate(rows, states, strata):
    result = defaultdict(Counter)
    for row in rows:
        keys = ["all", f"suite:{row['suite']}", f"length:{row['length']}"]
        if "cell" in row:
            keys.append(f"cell:{row['cell']}")
        for prediction in row["predictions"]:
            for key in keys + [f"stratum:{prediction['stratum']}", f"suite:{row['suite']}|length:{row['length']}|stratum:{prediction['stratum']}"]:
                g = result[key]
                g["cases"] += 1
                g["full_trace_correct"] += int(prediction["full_trace_correct"])
                g["final_correct"] += int(prediction["joint_final_correct"])
    return {key: dict(value) for key, value in result.items()}


def pair(candidate, reference):
    left = {(row["id"], tuple(p["state"])): p for row in candidate for p in row["predictions"]}
    right = {(row["id"], tuple(p["state"])): p for row in reference for p in row["predictions"]}
    assert set(left) == set(right)
    counts = Counter()
    for key in left:
        a, b = left[key]["full_trace_correct"], right[key]["full_trace_correct"]
        counts["repair" if a and not b else "regression" if b and not a else "both_correct" if a else "both_wrong"] += 1
    return {"cases": len(left), **{name: counts[name] for name in ("repair", "regression", "both_correct", "both_wrong")}}


def audit(out):
    out = Path(out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    freeze = json.loads((out / "input_freeze.json").read_text(encoding="utf-8"))
    scope = json.loads((Path(__file__).resolve().parents[1] / "runs/pc_latent_slots_v1/science_eval_scope.json").read_text(encoding="utf-8"))
    old = old_specs(scope)
    new = new_specs()
    assert len(old) == OLD_COUNT and len(new) == NEW_COUNT
    states, strata = states_and_strata()
    rows = {}
    for path in (out / "program_rows").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["arm"], row["phase"], row["id"])
        assert key not in rows
        rows[key] = row
    expected_keys = {("A", "initial_reused_saved", i) for i in old} | {("A", "initial", i) for i in new}
    expected_keys |= {("B", "initial", i) for i in {**old, **new}}
    expected_keys |= {(arm, "final", i) for arm in ("A", "B") for i in {**old, **new}}
    assert set(rows) == expected_keys
    for key, row in rows.items():
        _, phase, ident = key
        check_row(row, old.get(ident, new.get(ident)), key[0], phase, states, strata)
    for ident in old:
        reused = rows[("A", "initial_reused_saved", ident)]
        assert reused["provenance"] == {"source": "accepted_final_latent", "replayed": False}
    baseline = {str(row["id"]): row for row in json.loads((Path(__file__).resolve().parents[1] / "runs/pc_latent_slots_v1/science/evaluations/final_latent.json").read_text(encoding="utf-8"))["rows"]}
    for ident in old:
        saved_predictions = rows[("A", "initial_reused_saved", ident)]["predictions"]
        baseline_predictions = baseline[ident]["predictions"]
        assert len(saved_predictions) == len(baseline_predictions) == STATE_COUNT
        for saved, reference in zip(saved_predictions, baseline_predictions):
            assert saved["state"] == reference["state"] and saved["stratum"] == reference["stratum"]
            assert saved["target_trace"] == reference["target_trace"] and saved["predicted_trace"] == reference["predicted_trace"]
    initial_a = [rows[("A", "initial", i)] for i in new]
    initial_b = [rows[("B", "initial", i)] for i in {**old, **new}]
    final_a = [rows[("A", "final", i)] for i in {**old, **new}]
    final_b = [rows[("B", "final", i)] for i in {**old, **new}]
    paired = {
        "new_initial_B_vs_A": pair([rows[("B", "initial", i)] for i in new], initial_a),
        "old_initial_B_vs_saved_initial_A": pair([rows[("B", "initial", i)] for i in old], [rows[("A", "initial_reused_saved", i)] for i in old]),
        "old_padding_final_B_vs_A": pair([r for r in final_b if r["suite"] == "padding"], [r for r in final_a if r["suite"] == "padding"]),
        "old_compositions_final_B_vs_A": pair([r for r in final_b if r["suite"] == "compositions"], [r for r in final_a if r["suite"] == "compositions"]),
        "new_final_B_vs_A": pair([rows[("B", "final", i)] for i in new], [rows[("A", "final", i)] for i in new]),
    }
    computed = initial_a + initial_b + final_a + final_b
    computed_cases = sum(len(r["predictions"]) for r in computed)
    computed_positions = sum(len(r["program"]) * len(r["predictions"]) for r in computed)
    assert computed_cases == SCIENCE_COMPUTED_CASES and computed_positions == SCIENCE_COMPUTED_POSITIONS
    composition_b = [p for r in final_b if r["suite"] == "compositions" for p in r["predictions"]]
    result = {
        "status": "ACCEPT_TRACES",
        "model_calls": 0,
        "checkpoint_loads": 0,
        "rows": len(rows),
        "computed_cases": computed_cases,
        "computed_positions": computed_positions,
        "paired": paired,
        "aggregate_final_A": aggregate(final_a, states, strata),
        "aggregate_final_B": aggregate(final_b, states, strata),
        "aggregate_initial_B": aggregate(initial_b, states, strata),
        "primary": {
            "new_final_B_fewer_full_errors": paired["new_final_B_vs_A"]["regression"] < paired["new_final_B_vs_A"]["repair"],
            "old_padding_B_fewer_full_errors": paired["old_padding_final_B_vs_A"]["regression"] < paired["old_padding_final_B_vs_A"]["repair"],
            "old_compositions_zero_regression": paired["old_compositions_final_B_vs_A"]["regression"] == 0,
            "old_compositions_B_full_trace_correct": sum(bool(p["full_trace_correct"]) for p in composition_b),
            "old_compositions_denominator": COMPOSITION_CASES,
        },
        "runner_report_match": paired == {k: {x: v for x, v in report["paired"][k].items() if x not in {"label"}} for k in paired},
        "limitations": ["One fixed seed and endpoint; no model replay in this audit", "Compares a jointly trained architectural package", "Finite pointer-chasing controls do not establish broad reasoning generalization"],
    }
    assert result["runner_report_match"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out)
    with Path(args.report).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "rows", "computed_cases", "computed_positions", "primary", "runner_report_match")}, sort_keys=True))


if __name__ == "__main__":
    main()
