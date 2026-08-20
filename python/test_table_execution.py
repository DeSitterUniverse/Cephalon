from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from cephalon_core import storage
from cephalon_core.config import Settings
from cephalon_core.services.evidence_ledger import build_evidence_ledger
from cephalon_core.services.generation import _deterministic_unit_answer
from cephalon_core.services import evaluation, support
from cephalon_core.services.table_ingestion import persistence_rows
from cephalon_core.services.table_models import build_table
from cephalon_core.services.table_planning import (
    TableFilter,
    TablePlan,
    UnsafeTablePlan,
    plan_table_query,
)
from cephalon_core.services import table_retrieval
from cephalon_core.services.table_retrieval import (
    TableExecutionError,
    _sources_for_execution,
    document_unit_sources,
    execute_exact_year_route,
    execute_plan,
    execute_table_route,
    requested_unit_values,
)


def table_state(
    rows=None,
    *,
    doc_id="doc-1",
    name="study-results.csv",
    second_document=False,
    source_type="csv",
    number_formats=None,
):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    settings = Settings()
    storage.run_migrations(conn, settings)
    conn.execute(
        "INSERT INTO documents (id, path, display_name, content_hash, status, type) VALUES (?, ?, ?, 'hash', 'ready', 'file')",
        (doc_id, name, name),
    )
    rows = rows or [
        ["Name", "Group", "Mass", "Rate"],
        ["alpha", "A", "10 kg", "25%"],
        ["beta", "A", "20 kg", "50%"],
        ["gamma", "B", "30 kg", "75%"],
    ]
    table = build_table(rows, source_type=source_type, table_index=0, number_formats=number_formats)
    persisted = persistence_rows(doc_id, [table])
    _insert_rows(conn, persisted)
    if second_document:
        conn.execute(
            "INSERT INTO documents (id, path, display_name, content_hash, status, type) VALUES ('doc-2', 'other.csv', 'other.csv', 'hash2', 'ready', 'file')"
        )
        other = persistence_rows("doc-2", [build_table(rows, source_type="csv", table_index=0)])
        _insert_rows(conn, other)
    conn.commit()
    settings.table_execution = True
    return SimpleNamespace(sqlite=conn, settings=settings), persisted["tables"][0][0]


def _insert_rows(conn, rows):
    conn.executemany(
        "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows["tables"]
    )
    conn.executemany("INSERT INTO table_columns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows["columns"])
    conn.executemany("INSERT INTO table_rows VALUES (?, ?, ?, ?, ?, ?)", rows["rows"])
    conn.executemany("INSERT INTO table_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows["cells"])


def test_rule_router_resolves_named_document_unit_and_falls_back_on_cross_document_ambiguity():
    state, table_id = table_state()
    decision = plan_table_query(
        state.sqlite,
        'What value is expressed in kg? Use "Study results" as the target source.',
    )
    assert decision.status == "planned"
    assert decision.plan.table_ids == (table_id,)
    assert decision.plan.target_unit == "kg"

    ambiguous, _ = table_state(second_document=True)
    fallback = plan_table_query(ambiguous.sqlite, "What is the highest mass value?")
    assert fallback.status == "fallback"
    assert fallback.reason == "cross_document_ambiguity"


def test_direct_lookup_is_bounded_repeatable_and_source_compatible():
    state, table_id = table_state()
    prompt = 'What value is expressed in kg? Use "Study results" as the target source.'
    first_context, first_sources, first_trace = execute_table_route(state, prompt)
    second_context, second_sources, second_trace = execute_table_route(state, prompt)

    assert first_context == second_context
    assert [source.model_dump() for source in first_sources] == [source.model_dump() for source in second_sources]
    assert first_trace["status"] == second_trace["status"] == "executed"
    assert first_trace["model_calls"] == 0
    assert first_trace["result_count"] == 3
    assert first_sources[0].source_kind == "cell"
    assert first_sources[0].provenance["table_id"] == table_id
    assert first_sources[0].provenance["cell_refs"] == ["csv:r2c3", "csv:r3c3", "csv:r4c3"]

    ledger = build_evidence_ledger("query", prompt, [], first_sources)
    assert ledger["evidence"][0]["table_id"] == table_id
    assert ledger["evidence"][0]["cell_refs"] == first_sources[0].provenance["cell_refs"]


@pytest.mark.parametrize(
    ("name", "rows", "intent", "expected"),
    [
        (
            "noaa_global_temperature_timeseries.csv",
            [["Year", "Departure from Average"], ["1850", "-0.05"], ["2024", "1.26"]],
            "temperature",
            ["1850 | -0.05", "2024 | 1.26"],
        ),
        (
            "owid_global_co2.csv",
            [
                ["Entity", "Code", "Year", "Total (fossil fuels and land-use change)", "Land-use change", "Fossil fuels"],
                ["World", "OWID_WRL", "1850", "2910870000", "2714022400", "196847600"],
                ["World", "OWID_WRL", "2024", "43184087000", "4585508400", "38598580000"],
            ],
            "co2",
            ["1850 | 2910870000", "2024 | 43184087000"],
        ),
    ],
)
def test_exact_year_route_returns_requested_typed_rows_without_nearby_records(
    name, rows, intent, expected
):
    state, _ = table_state(rows=rows, name=name)
    prompt = (
        "According to the scientific records, compare the global CO2 totals and "
        "temperature departures in 1850 and 2024."
        if intent == "co2"
        else "According to the scientific temperature record, compare the global temperature departures in 1850 and 2024."
    )
    contexts, sources, trace = execute_exact_year_route(state, prompt)

    assert trace["status"] == "executed"
    assert trace["selected_intents"] == [intent]
    assert len(sources) == 1
    assert all(value in sources[0].evidence_text for value in expected)
    assert sources[0].source_kind == "cell"
    assert sources[0].cells
    assert contexts[0].startswith("[Source: S1 | ")


def test_exact_year_route_fails_closed_for_non_global_co2_rows():
    state, _ = table_state(
        rows=[
            ["Entity", "Code", "Year", "Total (fossil fuels and land-use change)"],
            ["Canada", "CAN", "1850", "100"],
            ["Canada", "CAN", "2024", "200"],
        ],
        name="owid_global_co2.csv",
    )

    _, sources, trace = execute_exact_year_route(
        state,
        "What were global CO2 emissions in 1850 and 2024?",
    )

    assert sources == []
    assert trace["status"] == "fallback"
    assert trace["fallback_reason"] == "exact_rows_not_found_or_ambiguous"


def test_filter_sort_count_and_aggregate_operations():
    state, table_id = table_state()
    filtered = execute_plan(state.sqlite, TablePlan(
        "filter", (table_id,), filters=(TableFilter(2, "gt", "15", "kg"),), limit=10
    ))
    assert [row["values"][0] for row in filtered.rows] == ["beta", "gamma"]

    sorted_rows = execute_plan(state.sqlite, TablePlan(
        "sort", (table_id,), value_column=2, sort_direction="desc", limit=2
    ))
    assert [row["values"][0] for row in sorted_rows.rows] == ["gamma", "beta"]

    counted = execute_plan(state.sqlite, TablePlan("count", (table_id,)))
    assert counted.rows[0]["value"] == "3"

    expected = {"min": "10", "max": "30", "sum": "60", "mean": "20"}
    for operation, value in expected.items():
        result = execute_plan(state.sqlite, TablePlan(operation, (table_id,), value_column=2))
        assert result.rows[0]["value"] == value
        assert result.rows[0]["unit"] == "kg"


def test_grouping_and_binary_arithmetic_are_exact():
    state, table_id = table_state()
    grouped = execute_plan(state.sqlite, TablePlan(
        "group", (table_id,), value_column=2, group_column=1, aggregate="mean"
    ))
    assert [(row["group"], row["value"]) for row in grouped.rows] == [("A", "15"), ("B", "30")]

    compare = execute_plan(state.sqlite, TablePlan(
        "compare", (table_id,), operand_cell_refs=("csv:r2c3", "csv:r3c3")
    ))
    difference = execute_plan(state.sqlite, TablePlan(
        "difference", (table_id,), operand_cell_refs=("csv:r2c3", "csv:r3c3")
    ))
    percentage = execute_plan(state.sqlite, TablePlan(
        "percentage", (table_id,), operand_cell_refs=("csv:r3c3", "csv:r2c3")
    ))
    assert compare.rows[0]["value"] == "less"
    assert difference.rows[0]["value"] == "-10"
    assert difference.rows[0]["unit"] == "kg"
    assert percentage.rows[0]["value"] == "100"
    assert percentage.rows[0]["unit"] == "%"


def test_executor_uses_xlsx_percent_points_not_underlying_scalars():
    state, table_id = table_state(
        [["Name", "Rate"], ["alpha", 0.125]],
        name="percentages.xlsx",
        source_type="xlsx",
        number_formats={(1, 1): "0.0%"},
    )

    result = execute_plan(state.sqlite, TablePlan("mean", (table_id,), value_column=1))

    assert result.rows[0]["value"] == "12.5"
    assert result.rows[0]["unit"] == "%"
    assert result.rows[0]["cell_refs"] == ["Sheet!B2"]


def test_mixed_units_zero_division_and_ambiguous_columns_fail_closed():
    mixed, table_id = table_state([
        ["Name", "Value"], ["a", "10 kg"], ["b", "20 m"],
    ])
    with pytest.raises(TableExecutionError, match="mixed_incompatible_units"):
        execute_plan(mixed.sqlite, TablePlan("sum", (table_id,), value_column=1))

    zero, zero_table = table_state([
        ["Name", "Value"], ["a", "10 kg"], ["b", "0 kg"],
    ])
    with pytest.raises(TableExecutionError, match="division_by_zero"):
        execute_plan(zero.sqlite, TablePlan(
            "percentage", (zero_table,), operand_cell_refs=("csv:r2c2", "csv:r3c2")
        ))

    distractor, _ = table_state([
        ["Name", "Mass", "Mass notes"], ["a", "10 kg", "estimated"],
    ])
    decision = plan_table_query(distractor.sqlite, 'What is the highest mass in "Study results"?')
    assert decision.status == "fallback"


def test_plan_schema_rejects_unknown_fields_identifiers_operators_and_limits():
    with pytest.raises(UnsafeTablePlan, match="unknown_plan_fields"):
        TablePlan.from_payload({"operation": "count", "table_ids": ["tbl-" + "a" * 24], "sql": "DROP TABLE tables"})
    with pytest.raises(UnsafeTablePlan, match="invalid_table_id"):
        TablePlan("count", ("tables; DROP TABLE documents",))
    with pytest.raises(UnsafeTablePlan, match="invalid_operator"):
        TableFilter(0, "LIKE; DELETE", "x")
    with pytest.raises(UnsafeTablePlan, match="result_limit"):
        TablePlan("count", ("tbl-" + "a" * 24,), limit=25)


def test_injection_text_is_bound_data_large_results_are_limited_and_timeout_falls_back(monkeypatch):
    rows = [["Name", "Value"], *[[f"row-{index}", str(index)] for index in range(40)]]
    state, table_id = table_state(rows)
    injection = execute_plan(state.sqlite, TablePlan(
        "lookup", (table_id,), search_text="x' OR 1=1; DROP TABLE tables; --"
    ))
    assert injection.rows == []
    assert state.sqlite.execute("SELECT COUNT(*) FROM tables").fetchone()[0] == 1

    bounded = execute_plan(state.sqlite, TablePlan("filter", (table_id,), limit=24))
    assert len(bounded.rows) == 24

    monkeypatch.setattr(table_retrieval, "EXECUTION_TIMEOUT_MS", -1)
    monkeypatch.setattr(table_retrieval, "SQL_PROGRESS_STEPS", 1)
    with pytest.raises(TimeoutError, match="table_execution_timeout"):
        execute_plan(state.sqlite, TablePlan("count", (table_id,)))


def test_execution_deadline_remains_active_after_sqlite_returns(monkeypatch):
    state, table_id = table_state()
    clock = {"now": 0.0}
    original_load_cells = table_retrieval._load_cells

    def load_cells_then_expire(conn, table_ids, deadline):
        rows = original_load_cells(conn, table_ids, deadline)
        clock["now"] = 1.0
        return rows

    monkeypatch.setattr(table_retrieval.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(table_retrieval, "_load_cells", load_cells_then_expire)

    with pytest.raises(TimeoutError, match="table_execution_timeout"):
        execute_plan(state.sqlite, TablePlan("count", (table_id,)))


def test_feature_flag_and_unrecognized_questions_fall_back_to_text_route():
    state, _ = table_state()
    state.settings.table_execution = False
    assert execute_table_route(state, "What is the highest mass?")[2]["fallback_reason"] == "feature_disabled"
    assert execute_exact_year_route(
        state, "What were global CO2 emissions in 1850 and 2024?"
    )[2]["fallback_reason"] == "feature_disabled"
    state.settings.table_execution = True
    assert execute_table_route(state, "Summarize the introduction.")[2]["fallback_reason"] == "not_a_table_question"


def test_requested_unit_candidates_preserve_central_values_and_case():
    prompt = "What value is expressed in km?"
    assert requested_unit_values(prompt, "Speeds were 197.5 ± 3.8 km and 14.0 +/- 3.2 km.") == [
        "197.5 ± 3.8 km", "14.0 +/- 3.2 km"
    ]
    assert requested_unit_values("What is expressed in M?", "33M parameters and 40m distance") == ["33M"]
    assert requested_unit_values("What is expressed in percent?", "The result was 40 percent (12.5%).") == [
        "40 percent", "12.5%"
    ]
    assert requested_unit_values("What is expressed in M?", "The limit was 0. 02 M.") == ["0.02 M"]


def test_named_document_unit_fallback_is_bounded_repeatable_and_provenance_preserving():
    state, _ = table_state(name="named-study.csv")
    state.sqlite.executemany(
        "INSERT INTO chunks (id, doc_id, chunk_index, text, page_number, provenance_json) VALUES (?, 'doc-1', ?, ?, ?, ?)",
        [
            ("chunk-a", 0, "A result of 4.2 kg was reported.", 3, '{"element_ids":["box-1"]}'),
            ("chunk-b", 1, "A second estimate was 7.0 kg.", 4, "{}"),
        ],
    )
    state.sqlite.commit()
    prompt = 'What result is expressed in kg? Use "Named Study" as the target source.'

    first_sources, first_trace = document_unit_sources(state, prompt)
    second_sources, second_trace = document_unit_sources(state, prompt)

    assert [source.model_dump() for source in first_sources] == [source.model_dump() for source in second_sources]
    assert first_trace["status"] == second_trace["status"] == "executed"
    assert first_trace["model_calls"] == 0
    assert first_trace["candidate_count"] == 2
    assert [source.chunk_id for source in first_sources] == ["chunk-a", "chunk-b"]
    assert first_sources[0].page_number == 3
    assert first_sources[0].provenance["element_ids"] == ["box-1"]
    assert first_sources[0].provenance["requested_unit_candidates"] == ["4.2 kg"]


def test_deterministic_unit_answer_lists_every_unique_candidate_with_valid_citations():
    meta = {"trace": {"table_execution": {"requested_unit_candidates": [
        {"source_id": "S3", "values": ["40 percent", "91.27%"]},
        {"source_id": "S4", "values": ["40 percent", "94.26%"]},
    ]}}}
    answer = _deterministic_unit_answer("What value is expressed in percent?", meta)

    assert "**40 percent** [[src:S3]]" in answer
    assert "**91.27%** [[src:S3]]" in answer
    assert "**94.26%** [[src:S4]]" in answer
    assert answer.count("40 percent") == 1
    assert "not guessed" in answer
    assert _deterministic_unit_answer("Summarize the paper.", meta) is None


@pytest.mark.parametrize(("plan", "claim", "method"), [
    ({"operation": "sum", "value_column": 2}, "The deterministic sum is 60 kg. [[src:S1]]", "cell_sum"),
    ({"operation": "mean", "value_column": 2}, "The deterministic mean is 20 kg. [[src:S1]]", "cell_mean"),
    ({"operation": "min", "value_column": 2}, "The deterministic minimum is 10 kg. [[src:S1]]", "cell_min"),
    ({"operation": "max", "value_column": 2}, "The deterministic maximum is 30 kg. [[src:S1]]", "cell_max"),
    ({"operation": "count"}, "The deterministic count is 3. [[src:S1]]", "cell_count"),
    ({"operation": "difference", "operand_cell_refs": ["csv:r2c3", "csv:r3c3"]}, "The deterministic difference is -10 kg. [[src:S1]]", "cell_difference"),
    ({"operation": "percentage", "operand_cell_refs": ["csv:r3c3", "csv:r2c3"]}, "The deterministic percentage is 100%. [[src:S1]]", "cell_percentage"),
])
def test_cell_verifier_recomputes_supported_arithmetic(plan, claim, method):
    state, table_id = table_state()
    plan = TablePlan.from_payload({**plan, "table_ids": [table_id]})
    execution = execute_plan(state.sqlite, plan)
    source = _sources_for_execution(state.sqlite, execution, plan)[0]

    result = support.validate_answer_claims(claim, [source])

    numeric = result["claims"][0]["numeric_verification"]
    assert numeric["status"] == "entailed"
    assert numeric["checks"][-1]["method"] == method
    assert source.table_id == table_id
    assert source.cells
    assert source.header_refs


def test_xlsx_percentage_points_reach_cell_citations_and_verifier():
    state, table_id = table_state(
        [["Name", "Rate"], ["alpha", 0.125]],
        name="percentages.xlsx",
        source_type="xlsx",
        number_formats={(1, 1): "0.0%"},
    )
    plan = TablePlan("mean", (table_id,), value_column=1)
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]

    cited = {cell.cell_ref: cell for cell in source.cells}["Sheet!B2"]
    correct = support.validate_answer_claims("The mean is 12.5%. [[src:S1]]", [source])
    underlying_scalar = support.validate_answer_claims("The mean is 0.125%. [[src:S1]]", [source])

    assert cited.raw_value == "0.125"
    assert cited.normalized_value == "12.5"
    assert cited.unit == "%"
    assert correct["claims"][0]["numeric_verification"]["status"] == "entailed"
    assert underlying_scalar["claims"][0]["numeric_verification"]["status"] == "contradicted"


def test_cell_verifier_reports_contradiction_unit_mismatch_missing_cell_and_comparison():
    state, table_id = table_state()
    sum_plan = TablePlan("sum", (table_id,), value_column=2)
    sum_source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, sum_plan), sum_plan)[0]

    wrong = support.validate_answer_claims("The deterministic sum is 59 kg. [[src:S1]]", [sum_source])
    mismatch = support.validate_answer_claims("The deterministic sum is 60 m. [[src:S1]]", [sum_source])
    missing = support.validate_answer_claims(
        "The deterministic sum is 60 kg. [[src:S1]]",
        [sum_source.model_copy(update={"cells": []})],
    )
    ambiguous = support.validate_answer_claims(
        "The deterministic sum is 60 kg. [[src:S1]]",
        [sum_source.model_copy(update={"table_operation": None})],
    )

    assert wrong["claims"][0]["numeric_verification"]["status"] == "contradicted"
    assert mismatch["claims"][0]["numeric_verification"]["status"] == "unit_mismatch"
    assert missing["claims"][0]["numeric_verification"]["status"] == "missing_cell"
    assert ambiguous["claims"][0]["numeric_verification"]["status"] == "ambiguous_operation"

    compare_plan = TablePlan(
        "compare", (table_id,), operand_cell_refs=("csv:r2c3", "csv:r3c3")
    )
    compare_source = _sources_for_execution(
        state.sqlite, execute_plan(state.sqlite, compare_plan), compare_plan
    )[0]
    comparison = support.validate_answer_claims(
        "10 kg is less than 20 kg. [[src:S1]]", [compare_source]
    )
    assert comparison["claims"][0]["numeric_verification"]["status"] == "entailed"


def test_cell_verifier_preserves_case_sensitive_single_letter_units():
    state, table_id = table_state([
        ["Name", "Value"], ["a", "10 M"], ["b", "20 M"],
    ])
    plan = TablePlan("sum", (table_id,), value_column=1)
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]

    exact = support.validate_answer_claims("The sum is 30 M. [[src:S1]]", [source])
    mismatched = support.validate_answer_claims("The sum is 30 m. [[src:S1]]", [source])

    assert exact["claims"][0]["numeric_verification"]["status"] == "entailed"
    assert mismatched["claims"][0]["numeric_verification"]["status"] == "unit_mismatch"


@pytest.mark.parametrize(("plan", "claim"), [
    (TablePlan("lookup", ("tbl-aaaaaaaaaaaaaaaaaaaaaaaa",), target_unit="kg"), "One result is 20 kg. [[src:S1]]"),
    (TablePlan("filter", ("tbl-aaaaaaaaaaaaaaaaaaaaaaaa",), filters=(TableFilter(2, "gt", "15", "kg"),)),
     "One filtered result is 30 kg. [[src:S1]]"),
    (TablePlan("sort", ("tbl-aaaaaaaaaaaaaaaaaaaaaaaa",), value_column=2, sort_direction="desc", limit=2),
     "The leading result is 30 kg. [[src:S1]]"),
    (TablePlan("group", ("tbl-aaaaaaaaaaaaaaaaaaaaaaaa",), value_column=2, group_column=1, aggregate="mean"),
     "The first group mean is 15 kg. [[src:S1]]"),
])
def test_cell_verifier_checks_values_for_non_scalar_table_operations(plan, claim):
    state, table_id = table_state()
    plan = TablePlan.from_payload({**plan.trace_payload(), "table_ids": [table_id]})
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]

    result = support.validate_answer_claims(claim, [source])

    assert result["claims"][0]["numeric_verification"]["status"] == "entailed"


def test_cell_source_round_trips_through_saved_conversation():
    state, table_id = table_state()
    plan = TablePlan("mean", (table_id,), value_column=2)
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]
    conversation = storage.create_conversation(state.sqlite, "Cell citation")
    message = storage.append_message(state.sqlite, conversation["id"], "assistant", "20 kg [[src:S1]]")
    storage.save_message_sources(state.sqlite, message["id"], [source.model_dump()])
    support_payload = support.classify_answer_support("The mean is 20 kg. [[src:S1]]", [source])
    storage.save_answer_record(state.sqlite, {
        "id": message["id"],
        "message_id": message["id"],
        "answer_text": "The mean is 20 kg. [[src:S1]]",
        "support_status": support_payload["status"],
        "citations": support_payload["citations"],
    })

    restored = storage.get_conversation(state.sqlite, conversation["id"])["messages"][0]["sources"][0]

    assert restored["table_id"] == table_id
    assert restored["cell_refs"] == source.cell_refs
    assert restored["header_refs"] == source.header_refs
    assert restored["cells"] == [cell.model_dump(mode="json") for cell in source.cells]
    assert restored["table_operation"] == "mean"
    citation = state.sqlite.execute(
        "SELECT payload_json FROM answer_citations WHERE answer_id = ?", (message["id"],)
    ).fetchone()[0]
    assert f'"table_id":"{table_id}"' in citation
    assert '"cell_refs":["csv:r2c3","csv:r3c3","csv:r4c3"]' in citation


@pytest.mark.parametrize(("payload", "expected_refs"), [
    ({"operation": "lookup", "target_unit": "kg"},
     {"csv:r2c3", "csv:r3c3", "csv:r4c3"}),
    ({"operation": "filter", "filters": [{"column_index": 2, "operator": "gt", "value": "15", "unit": "kg"}]},
     {f"csv:r{row}c{column}" for row in (3, 4) for column in range(1, 5)}),
    ({"operation": "sort", "value_column": 2, "sort_direction": "desc", "limit": 2},
     {f"csv:r{row}c{column}" for row in (3, 4) for column in range(1, 5)}),
    ({"operation": "group", "value_column": 2, "group_column": 1, "aggregate": "mean"},
     {"csv:r2c3", "csv:r3c3", "csv:r4c3"}),
    ({"operation": "sum", "value_column": 2},
     {"csv:r2c3", "csv:r3c3", "csv:r4c3"}),
    ({"operation": "mean", "value_column": 2},
     {"csv:r2c3", "csv:r3c3", "csv:r4c3"}),
    ({"operation": "min", "value_column": 2}, {"csv:r2c3"}),
    ({"operation": "max", "value_column": 2}, {"csv:r4c3"}),
    ({"operation": "difference", "operand_cell_refs": ["csv:r2c3", "csv:r3c3"]},
     {"csv:r2c3", "csv:r3c3"}),
    ({"operation": "percentage", "operand_cell_refs": ["csv:r3c3", "csv:r2c3"]},
     {"csv:r2c3", "csv:r3c3"}),
    ({"operation": "compare", "operand_cell_refs": ["csv:r2c3", "csv:r3c3"]},
     {"csv:r2c3", "csv:r3c3"}),
])
def test_all_cell_bearing_operations_reach_exact_gold_citation_scores(payload, expected_refs):
    state, table_id = table_state()
    plan = TablePlan.from_payload({**payload, "table_ids": [table_id]})
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]

    metrics = evaluation.answer_metrics(
        item={"gold_evidence": [{"source_kind": "cell", "cell_refs": sorted(expected_refs)}]},
        answer="Deterministic result. [[src:S1]]",
        sources=[source.model_dump(mode="json")],
    )

    assert set(source.cell_refs) == expected_refs
    assert metrics["cell_citation_precision"] == 1.0
    assert metrics["cell_citation_recall"] == 1.0


def test_count_has_table_level_provenance_when_no_result_cell_exists():
    state, table_id = table_state()
    plan = TablePlan("count", (table_id,))
    source = _sources_for_execution(state.sqlite, execute_plan(state.sqlite, plan), plan)[0]

    assert source.source_kind == "table"
    assert source.cell_refs == []
    assert source.verification_cell_refs == ["csv:r2c1", "csv:r3c1", "csv:r4c1"]
