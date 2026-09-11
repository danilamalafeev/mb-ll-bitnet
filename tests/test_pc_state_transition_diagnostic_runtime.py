from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json

import pytest
import torch

from looped_bitnet import register_e15 as dsl
from scripts import pc_state_transition_diagnostic as diagnostic
from scripts import pc_state_transition_diagnostic_runtime as runtime


def _saved_row(identifier: str, program: tuple[str, ...], state: tuple[int, int], *, correct: bool) -> dict[str, object]:
    example = dsl.RegisterExample(state[0], state[1], program)
    target = [list(pair) for pair in example.targets]
    predicted = target if correct else [list(pair) for pair in target[:-1]] + [[0, 0]]
    return {
        "state": list(state),
        "stratum": "train",
        "target_trace": target,
        "predicted_trace": predicted,
        "full_trace_correct": correct,
        "first_error": None if correct else len(target),
    }


def test_pair_specs_binds_same_semantic_state_and_saved_expectations() -> None:
    good_spec = {"id": diagnostic.GOOD_ID, "program": list(diagnostic.GOOD_PROGRAM)}
    bad_spec = {"id": diagnostic.BAD_ID, "program": list(diagnostic.BAD_PROGRAM)}
    specs = [good_spec, bad_spec]
    good_rows = [_saved_row(diagnostic.GOOD_ID, diagnostic.GOOD_PROGRAM, state, correct=True) for state in diagnostic.PAIR_STATES]
    bad_rows = [_saved_row(diagnostic.BAD_ID, diagnostic.BAD_PROGRAM, state, correct=False) for state in diagnostic.PAIR_STATES]
    scope = {"programs": specs}
    final = {"rows": [{"id": diagnostic.GOOD_ID, "predictions": good_rows}, {"id": diagnostic.BAD_ID, "predictions": bad_rows}]}
    good, bad, pair = runtime._pair_specs(scope, final)
    assert good["id"] == diagnostic.GOOD_ID
    assert bad["id"] == diagnostic.BAD_ID
    assert [case["state"] for case in pair["cases"]] == [list(state) for state in diagnostic.PAIR_STATES]
    broken = deepcopy(final)
    broken["rows"][1]["predictions"][0]["full_trace_correct"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="expectation"):
        runtime._pair_specs(scope, broken)


def test_accounting_matches_two_paths_and_one_substitution() -> None:
    counter = runtime._new_counter()
    sink = lambda _counter: None
    runtime._account_phase(counter, "paths", cases=2, positions=len(diagnostic.GOOD_PROGRAM) * 2, native_steps=len(diagnostic.GOOD_PROGRAM) * 2 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._complete_phase(counter, "paths", cases=2, positions=len(diagnostic.GOOD_PROGRAM) * 2, native_steps=len(diagnostic.GOOD_PROGRAM) * 2 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._account_phase(counter, "paths", cases=2, positions=len(diagnostic.BAD_PROGRAM) * 2, native_steps=len(diagnostic.BAD_PROGRAM) * 2 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._complete_phase(counter, "paths", cases=2, positions=len(diagnostic.BAD_PROGRAM) * 2, native_steps=len(diagnostic.BAD_PROGRAM) * 2 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._account_phase(counter, "substitution", cases=8, positions=8, native_steps=8 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._complete_phase(counter, "substitution", cases=8, positions=8, native_steps=8 * diagnostic.NATIVE_STEPS, sink=sink)
    runtime._validate_accounting(counter)
    assert counter["attempted_forwards"] == counter["completed_forwards"] == 3
    assert counter["attempted_optimizer_steps"] == 0


def test_source_gate_checks_current_bytes_and_budget(tmp_path, monkeypatch) -> None:
    files = {}
    source_paths = {}
    for label in ("helper", "runtime", "tests", "runtime_tests", "protocol"):
        path = tmp_path / f"{label}.txt"
        path.write_text(label, encoding="utf-8")
        files[label] = {"sha256": runtime._sha256(path)}
        source_paths[label] = path
        monkeypatch.setattr(runtime, label.upper(), path)
    register_path = tmp_path / "looped_bitnet" / "register_e15.py"
    register_path.parent.mkdir()
    register_path.write_text("register", encoding="utf-8")
    files["register_e15"] = {"sha256": runtime._sha256(register_path)}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"files": files, "budget": diagnostic.DIAGNOSTIC_BUDGET}), encoding="utf-8")
    monkeypatch.setattr(runtime, "SOURCE_GATE", gate_path)
    assert runtime._validate_source_gate(tmp_path)["budget"] == diagnostic.DIAGNOSTIC_BUDGET
    source_paths["helper"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source gate changed"):
        runtime._validate_source_gate(tmp_path)


def test_attention_hook_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="malformed"):
        runtime._record_attention_hook(object(), (), [])


def test_json_tensor_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        runtime._json_tensor(torch.tensor([float("inf")]))


def test_preserve_eval_state_uses_science_context_manager(monkeypatch) -> None:
    events: list[str] = []

    @contextmanager
    def fake_preserve(*_args, **_kwargs):
        events.append("enter")
        yield
        events.append("exit")

    class Dummy:
        def eval(self) -> None:
            events.append("eval")

    monkeypatch.setattr(runtime.science, "_preserve_evaluation_state", fake_preserve)
    with runtime._preserve_eval_state(Dummy(), Dummy(), object(), device=torch.device("cpu")):
        events.append("body")
    assert events == ["enter", "eval", "eval", "body", "exit"]
