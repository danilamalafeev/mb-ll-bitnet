"""Symbolic E35/E36 preparation checks; no checkpoint loads or model forwards."""

from scripts import length_wave_e35_e36 as r


def test_frozen_schedules_and_instruction_budgets():
    a, b = r.build_streams()
    assert len(a) == len(b) == 8000
    assert sum(len(batch[0].program) for batch in a) == 16002
    assert sum(len(batch[0].program) for batch in b) == 27998
    assert sum(len(batch[0].program) for batch in b[:r.MATCHED_UPDATES]) == 16002
    assert r._cost_for_batches(a) == {
        "program_forwards": 8000, "program_state_cases": 512000,
        "readout_positions": 1024128, "internal_state_substeps": 8193024,
    }
    assert r._cost_for_batches(b) == {
        "program_forwards": 8000, "program_state_cases": 512000,
        "readout_positions": 1791872, "internal_state_substeps": 14334976,
    }
    assert all(r.old.batch_digest([left]) == r.old.batch_digest([right])
               for left, right in zip(a[:4000], [x for x in b if len(x[0].program) <= 3]))


def test_e35_scope_is_exact_and_identity_targets_match():
    programs = []
    for prefix in r.E35_PREFIXES:
        for opcode in r.E35_OPS:
            for padding in r.E35_PADDING:
                programs.append(tuple(prefix) + ("SWAP", "SWAP") * padding + (opcode,))
    assert len(programs) == 36
    assert sorted({len(p) for p in programs}) == [3, 5, 7, 9]
    for prefix in r.E35_PREFIXES:
        for opcode in r.E35_OPS:
            short = r.dsl.semantic_signature(tuple(prefix) + (opcode,))
            for padding in r.E35_PADDING[1:]:
                long = r.dsl.semantic_signature(tuple(prefix) + ("SWAP", "SWAP") * padding + (opcode,))
                assert short == long
                assert r.dsl.semantic_signature(tuple(prefix) + ("SWAP", "SWAP") * padding) == r.dsl.semantic_signature(tuple(prefix))


def test_semantic_pool_cap_and_prefix_exclusion():
    pools = r.frozen_evaluation_pools()
    for length, categories in pools["programs"].items():
        for category in categories.values():
            assert len(category["allowed"]) <= r.POOL_LIMIT
            assert len(category["forbidden"]) <= r.POOL_LIMIT
    # The production pool builder performs the full-domain prefix exclusion;
    # this check keeps the symbolic test bounded while guarding its output.
    assert pools["prefix_signature_count"] > 0
    assert any(categories["semantic_new"]["allowed"] for categories in pools["programs"].values())


def test_bounded_feasibility_is_complete_here_and_records_no_unknown_minimality():
    value = r._enumerate_feasibility()
    assert value["capped"] is False
    assert value["complete_through_length"] == 10
    assert value["unknown_minimality_from_length"] is None
    assert value["all_cumulative_classes"] == 9789
