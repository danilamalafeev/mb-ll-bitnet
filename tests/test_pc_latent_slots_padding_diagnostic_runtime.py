from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import pytest

from looped_bitnet import register_e15 as dsl
from scripts import pc_latent_slots_padding_diagnostic as diagnostic
from scripts import pc_latent_slots_padding_diagnostic_runtime as runtime


def _scope() -> dict[str, object]:
    programs: list[dict[str, object]] = []
    for index in range(runtime.EVALUATION_PROGRAMS):
        if index < len(diagnostic.FOCUS_PROGRAM_MAP):
            identifier, program, _roles = diagnostic.FOCUS_PROGRAM_MAP[index]
            suite = "padding"
        else:
            identifier = f"saved_extra_{index:02d}"
            program = ("ADD",)
            suite = "compositions"
        states = [{"state": list(state), "stratum": diagnostic._state_stratum(state), "target_trace": []} for state in dsl.STATE_ORDER]
        programs.append({"id": identifier, "suite": suite, "length": len(program), "program": list(program), "states": states})
    return {"states": [list(state) for state in dsl.STATE_ORDER], "programs": programs}


def test_registered_budget_and_focus_scope_are_exact() -> None:
    assert (runtime.EVALUATION_CALLS, runtime.EVALUATION_CASES, runtime.EVALUATION_POSITIONS, runtime.EVALUATION_NATIVE_STEPS) == (69, 17664, 331776, 2654208)
    assert (runtime.FOCUS_CALLS, runtime.FOCUS_CASES, runtime.FOCUS_POSITIONS, runtime.FOCUS_NATIVE_STEPS) == (18, 4608, 129024, 1032192)
    scope = _scope()
    runtime._validate_strata(scope)
    focus = runtime._focus_scope(scope)
    assert len(focus) == 18
    assert tuple(row["id"] for row in focus) == tuple(runtime.FOCUS_IDS)


def test_e15_join_mutation_is_rejected_before_model_work() -> None:
    scope = _scope()
    assert scope["programs"][0]["states"][0]["state"] == [0, 0]  # type: ignore[index]
    assert scope["programs"][0]["states"][0]["stratum"] == "test"  # type: ignore[index]
    broken = deepcopy(scope)
    broken["programs"][0]["states"][0]["stratum"] = "train"  # type: ignore[index]
    with pytest.raises(ValueError, match="state/stratum"):
        runtime._validate_strata(broken)


def test_accounting_persists_attempt_before_completion_and_has_no_updates() -> None:
    counter = runtime._new_counter()
    writes: list[dict[str, object]] = []
    runtime._account_attempt(counter, 24, lambda value: writes.append(deepcopy(value)))
    assert counter["attempted_forwards"] == 1
    assert counter["attempted_cases"] == 256
    assert counter["attempted_readout_positions"] == 6144
    assert counter["attempted_native_steps"] == 49152
    assert counter["completed_forwards"] == 0
    assert writes[-1]["completed_forwards"] == 0
    runtime._account_complete(counter, 24, lambda value: writes.append(deepcopy(value)))
    assert counter["completed_forwards"] == 1
    assert counter["completed_cases"] == 256
    assert counter["completed_readout_positions"] == 6144
    assert counter["completed_native_steps"] == 49152
    assert counter["attempted_updates"] == counter["completed_updates"] == 0
    assert counter["attempted_backwards"] == counter["completed_backwards"] == 0
    assert writes[-1]["completed_forwards"] == 1


def test_atomic_json_is_same_volume_and_refuses_collision(tmp_path) -> None:
    destination = tmp_path / "accounting.json"
    runtime._atomic_json(destination, {"state": "first"}, refuse=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        runtime._atomic_json(destination, {"state": "second"}, refuse=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text("incomplete", encoding="utf-8")
    with pytest.raises(FileExistsError, match="temporary collision"):
        runtime._atomic_json(destination, {"state": "blocked"})


def test_accepted_loader_context_binds_adapter_and_canonical_manifest_digest(monkeypatch, tmp_path) -> None:
    inherited = {"b": [2, 1], "a": {"z": True}}
    canonical = runtime._canonical_manifest_hash(inherited)
    assert canonical != runtime.science._digest(inherited)
    active: list[bool] = []

    @contextmanager
    def fake_adapter():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def fake_inherited(_root):
        assert active == [True]
        return inherited

    monkeypatch.setattr(runtime.migration, "windows_compatibility_adapter", fake_adapter)
    monkeypatch.setattr(runtime.followup, "_load_inherited", fake_inherited)
    manifest = {"parent": {"manifest_digest": canonical}}
    with runtime._accepted_load_context(manifest=manifest, root=tmp_path) as loaded:
        assert active == [True]
        assert loaded == inherited
    assert active == []
    with pytest.raises(ValueError, match="canonical digest"):
        with runtime._accepted_load_context(manifest={"parent": {"manifest_digest": "0" * 64}}, root=tmp_path):
            pass


def test_load_failure_after_parent_success_preserves_partial_cost_and_success_parity() -> None:
    counter = runtime._new_counter()
    writes: list[dict[str, object]] = []
    sink = lambda value: writes.append(deepcopy(value))
    runtime._begin_load(counter, sink)
    runtime._mark_parent_loaded(counter, sink)
    assert counter["attempted_endpoint_loads"] == counter["attempted_parent_loads"] == counter["attempted_endpoint_restores"] == 1
    assert counter["completed_parent_loads"] == 1
    assert counter["completed_endpoint_loads"] == counter["completed_endpoint_restores"] == 0
    assert counter["phase_counts"]["load"]["completed_parent_loads"] == 1
    runtime._failure(counter, "mock_endpoint_restore", RuntimeError("restore failed"), phase="load")
    sink(counter)
    assert counter["completed_parent_loads"] == 1  # completed work is never decremented
    assert counter["completed_endpoint_loads"] == counter["completed_endpoint_restores"] == 0
    assert writes[-1]["completed_parent_loads"] == 1
    success = runtime._new_counter()
    runtime._begin_load(success, sink)
    runtime._mark_parent_loaded(success, sink)
    runtime._mark_endpoint_loaded(success, sink)
    assert success["completed_endpoint_loads"] == success["completed_endpoint_restores"] == 1
    assert success["phase_counts"]["load"]["completed_endpoint_loads"] == 1


def test_fresh_output_rejects_nonempty_or_file(tmp_path) -> None:
    destination = tmp_path / "diagnostic"
    runtime._refuse_fresh(destination)
    (destination / "committed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        runtime._refuse_fresh(destination)
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        runtime._refuse_fresh(file_path)
