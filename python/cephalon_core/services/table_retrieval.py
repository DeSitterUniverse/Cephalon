"""Execute validated table plans through bounded, application-owned queries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import time
from typing import Any

from .. import storage
from ..schemas import SourceChunk
from .table_planning import (
    PlanDecision,
    TableFilter,
    TablePlan,
    UnsafeTablePlan,
    plan_table_query,
    resolve_named_document,
)


MAX_SCANNED_CELLS = 50_000
EXECUTION_TIMEOUT_MS = 250
SQL_PROGRESS_STEPS = 1_000
MAX_CONTEXT_CHARACTERS = 8_000
MAX_REQUESTED_UNIT_CANDIDATES = 24
MAX_DOCUMENT_SCAN_CHUNKS = 5_000
MAX_DOCUMENT_VALUE_SOURCES = 24
MAX_DOCUMENT_SOURCE_CHARACTERS = 1_200
MAX_EXACT_YEAR_LOOKUP_YEARS = 8
NUMERIC_TYPES = {"integer", "decimal", "percentage"}


class TableExecutionError(ValueError):
    """A safe plan could not produce an unambiguous bounded result."""


def requested_unit_values(prompt: str, text: str) -> list[str]:
    """Extract bounded explicit number-unit spans without choosing among them."""
    unit_match = re.search(r"\bexpressed\s+in\s+([A-Za-zµμ%]+|percent)\b", prompt)
    if not unit_match:
        return []
    requested = unit_match.group(1)
    requested = "%" if requested.casefold() == "percent" else requested
    number = r"[−+\-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
    uncertainty = rf"(?:\s*(?:±|\+/-)\s*{number})?"
    if requested == "%":
        suffix = r"(?:%|percent(?:age)?)"
        flags = re.IGNORECASE
    else:
        suffix = rf"{re.escape(requested)}(?![A-Za-zµμ])"
        flags = 0 if len(requested) == 1 else re.IGNORECASE
    # PDF text layers often insert spaces around a decimal point (``0. 02``).
    # Removing only digit-adjacent spacing is deterministic and does not join
    # unrelated tokens or alter source storage.
    scan_text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    values = []
    for match in re.finditer(rf"{number}{uncertainty}\s*{suffix}", scan_text, flags):
        value = " ".join(match.group(0).split())
        if value not in values:
            values.append(value)
        if len(values) >= MAX_REQUESTED_UNIT_CANDIDATES:
            break
    return values


def document_unit_sources(app_state, prompt: str) -> tuple[list[SourceChunk], dict[str, Any]]:
    """Boundedly expose unit-bearing text from one explicitly named document.

    This is a provenance-preserving text fallback, not a table answer selector:
    every candidate remains attached to its original chunk and the caller/model
    must retain ambiguity when several values are supported.
    """
    started = time.perf_counter()
    bounds = {
        "max_chunks": MAX_DOCUMENT_SCAN_CHUNKS,
        "max_sources": MAX_DOCUMENT_VALUE_SOURCES,
        "max_candidates": MAX_REQUESTED_UNIT_CANDIDATES,
        "max_source_characters": MAX_DOCUMENT_SOURCE_CHARACTERS,
        "timeout_ms": EXECUTION_TIMEOUT_MS,
    }
    if not getattr(app_state.settings, "table_execution", True):
        return [], _document_scan_trace("disabled", "feature_disabled", started, bounds)
    if not re.search(r"\bexpressed\s+in\s+([A-Za-zµμ%]+|percent)\b", prompt):
        return [], _document_scan_trace("skipped", "no_requested_unit", started, bounds)
    document = resolve_named_document(app_state.sqlite, prompt)
    if document is None:
        return [], _document_scan_trace("fallback", "named_document_not_resolved", started, bounds)

    rows = storage.fetchall(
        app_state.sqlite,
        """
        SELECT id, parent_id, text, block_type, section_heading, heading_path,
               page_number, page_end, block_index, bounding_box, provenance_json
        FROM chunks
        WHERE doc_id = ?
        ORDER BY chunk_index, id
        LIMIT ?
        """,
        (document["id"], MAX_DOCUMENT_SCAN_CHUNKS + 1),
    )
    if len(rows) > MAX_DOCUMENT_SCAN_CHUNKS:
        return [], _document_scan_trace(
            "fallback", "document_chunk_limit", started, bounds,
            document_id=document["id"], scanned_chunks=len(rows),
        )

    deadline = started + EXECUTION_TIMEOUT_MS / 1000
    sources: list[SourceChunk] = []
    seen_values: set[str] = set()
    candidate_count = 0
    for row in rows:
        if time.perf_counter() > deadline:
            return [], _document_scan_trace(
                "fallback", "document_scan_timeout", started, bounds,
                document_id=document["id"], scanned_chunks=len(rows),
            )
        values = [
            value for value in requested_unit_values(prompt, row["text"])
            if value not in seen_values
        ]
        if not values:
            continue
        remaining = MAX_REQUESTED_UNIT_CANDIDATES - candidate_count
        values = values[:remaining]
        if not values:
            break
        seen_values.update(values)
        candidate_count += len(values)
        provenance = _json_object(row["provenance_json"])
        provenance = {
            **provenance,
            "document_unit_scan": True,
            "requested_unit_candidates": values,
            "document_path": document["path"],
        }
        evidence = (
            "Deterministic requested-unit matches in this named-document text "
            f"(all remain candidates): {'; '.join(values)}.\n"
            f"{row['text'][:MAX_DOCUMENT_SOURCE_CHARACTERS]}"
        )
        bounding_box = _json_list(row["bounding_box"])
        heading_path = _json_list(row["heading_path"])
        rank = len(sources) + 1
        sources.append(SourceChunk(
            rank=rank,
            source_id=f"S{rank}",
            doc_id=document["id"],
            doc_name=document["display_name"] or str(document["path"]).rsplit("\\", 1)[-1],
            chunk_id=row["id"],
            parent_id=row["parent_id"],
            source_kind="text",
            score=1.0,
            final_score=1.0,
            snippet=row["text"][:500],
            evidence_text=evidence,
            block_type=row["block_type"],
            section_heading=row["section_heading"],
            heading_path=heading_path,
            page_number=row["page_number"],
            page_end=row["page_end"],
            block_index=row["block_index"],
            bounding_box=tuple(bounding_box) if len(bounding_box) == 4 else None,
            element_ids=provenance.get("element_ids", []),
            provenance=provenance,
            context_selection={"decision": "bounded named-document unit fallback", "objective": 1.0},
        ))
        if candidate_count >= MAX_REQUESTED_UNIT_CANDIDATES or len(sources) >= MAX_DOCUMENT_VALUE_SOURCES:
            break

    status = "executed" if sources else "fallback"
    reason = None if sources else "requested_unit_not_found"
    return sources, _document_scan_trace(
        status,
        reason,
        started,
        bounds,
        document_id=document["id"],
        scanned_chunks=len(rows),
        source_count=len(sources),
        candidate_count=candidate_count,
    )


def execute_exact_year_route(
    app_state, prompt: str
) -> tuple[list[str], list[SourceChunk], dict[str, Any]]:
    """Return exact year rows for an explicitly scientific, multi-record query.

    Hybrid retrieval is intentionally approximate.  That is useful for prose,
    but it is unsafe when a question asks for two exact years in a long CSV:
    the nearest chunk can belong to another country or another year.  This
    route only handles the two scientific record shapes currently in the
    corpus, reads the typed table cells, and fails closed when a row or unit
    of selection is ambiguous.
    """
    started = time.perf_counter()
    years = _requested_years(prompt)
    intents = _scientific_year_intents(prompt)
    bounds = {
        "max_years": MAX_EXACT_YEAR_LOOKUP_YEARS,
        "max_tables": 16,
        "max_rows_per_table": MAX_EXACT_YEAR_LOOKUP_YEARS,
        "timeout_ms": EXECUTION_TIMEOUT_MS,
    }
    if not getattr(app_state.settings, "table_execution", True):
        return [], [], _exact_year_trace("disabled", "feature_disabled", started, bounds)
    if len(years) < 2:
        return [], [], _exact_year_trace("skipped", "fewer_than_two_years", started, bounds, years=years)
    if not intents:
        return [], [], _exact_year_trace("skipped", "not_a_supported_scientific_record", started, bounds, years=years)

    tables = storage.fetchall(
        app_state.sqlite,
        """
        SELECT tables.*, documents.id AS doc_id, documents.display_name, documents.path
        FROM tables JOIN documents ON documents.id = tables.doc_id
        WHERE documents.status = 'ready' AND tables.source_type = 'csv'
        ORDER BY documents.path, tables.table_index, tables.id
        LIMIT ?
        """,
        (bounds["max_tables"] + 1,),
    )
    if len(tables) > bounds["max_tables"]:
        return [], [], _exact_year_trace("fallback", "table_limit", started, bounds, years=years)

    sources: list[SourceChunk] = []
    selected_tables: list[str] = []
    selected_intents: list[str] = []
    for table in tables:
        if len(sources) >= len(intents) or time.perf_counter() > started + EXECUTION_TIMEOUT_MS / 1000:
            break
        columns = storage.fetchall(
            app_state.sqlite,
            """
            SELECT column_index, raw_header, normalized_header, inferred_type, inferred_unit
            FROM table_columns WHERE table_id = ? ORDER BY column_index
            """,
            (table["id"],),
        )
        intent = _best_scientific_intent(table, columns, intents)
        if intent is None:
            continue
        rows = _exact_year_rows(app_state.sqlite, table["id"], columns, years, intent)
        if rows is None:
            continue
        source = _exact_year_source(
            app_state.sqlite, table, columns, rows, years, intent, len(sources) + 1
        )
        sources.append(source)
        selected_tables.append(table["id"])
        selected_intents.append(intent)

    if not sources:
        return [], [], _exact_year_trace(
            "fallback", "exact_rows_not_found_or_ambiguous", started, bounds,
            years=years, intents=intents,
        )
    contexts = [
        f"[Source: {source.source_id} | {source.doc_name} | Exact typed-table evidence]\n"
        f"{source.evidence_text}"
        for source in sources
    ]
    return contexts, sources, _exact_year_trace(
        "executed", None, started, bounds,
        years=years,
        intents=intents,
        selected_table_ids=selected_tables,
        selected_intents=selected_intents,
        source_count=len(sources),
    )


def _requested_years(prompt: str) -> list[int]:
    years: list[int] = []
    for match in re.finditer(r"\b(?:18|19|20)\d{2}\b", prompt):
        year = int(match.group(0))
        if year not in years:
            years.append(year)
        if len(years) >= MAX_EXACT_YEAR_LOOKUP_YEARS:
            break
    return years


def _scientific_year_intents(prompt: str) -> list[str]:
    lowered = prompt.casefold()
    intents = []
    asks_for_global_series = bool(re.search(r"\b(?:world|global|worldwide)\b", lowered))
    if asks_for_global_series and any(
        term in lowered for term in ("co2", "carbon dioxide", "carbon emissions", "co₂")
    ):
        intents.append("co2")
    if any(term in lowered for term in ("temperature", "temperature departure", "departures", "warming")):
        intents.append("temperature")
    return intents


def _best_scientific_intent(table, columns, intents: list[str]) -> str | None:
    identity = " ".join(
        str(table[key] or "").casefold() for key in ("display_name", "path", "caption")
    )
    headers = " ".join(
        f"{column['raw_header']} {column['normalized_header']}".casefold()
        for column in columns
    )
    scores = {}
    for intent in intents:
        score = 0
        if intent == "co2":
            score += 4 if "co2" in identity or "co2" in headers else 0
            score += 3 if "total_fossil_fuels_and_land_use_change" in headers else 0
            score += 2 if "emission" in identity or "emission" in headers else 0
        elif intent == "temperature":
            score += 4 if "noaa" in identity else 0
            score += 3 if "temperature" in identity or "temperature" in headers else 0
            score += 2 if "departure" in identity or "departure" in headers else 0
        if score:
            scores[intent] = score
    if not scores:
        return None
    return max(scores, key=lambda item: (scores[item], -intents.index(item)))


def _exact_year_rows(conn, table_id: str, columns, years: list[int], intent: str):
    year_column = _year_column(conn, table_id, columns, years)
    if year_column is None:
        return None
    year_values = tuple(str(year) for year in years)
    placeholders = ",".join("?" for _ in year_values)
    if intent == "co2":
        entity_column = _column_index(columns, {"entity"})
        code_column = _column_index(columns, {"code"})
        if entity_column is None or code_column is None:
            return None
        candidate_rows = storage.fetchall(
            conn,
            f"""
            SELECT years.row_index FROM table_cells AS years
            WHERE years.table_id = ? AND years.column_index = ?
              AND years.value_type = 'integer'
              AND years.normalized_value IN ({placeholders})
              AND (
                EXISTS (
                    SELECT 1 FROM table_cells AS entity
                    WHERE entity.table_id = years.table_id
                      AND entity.row_index = years.row_index
                      AND entity.column_index = ? AND entity.raw_value = 'World'
                )
                OR EXISTS (
                    SELECT 1 FROM table_cells AS code
                    WHERE code.table_id = years.table_id
                      AND code.row_index = years.row_index
                      AND code.column_index = ? AND code.raw_value = 'OWID_WRL'
                )
              )
            ORDER BY years.row_index LIMIT ?
            """,
            (table_id, year_column, *year_values, entity_column, code_column, len(years)),
        )
    else:
        candidate_rows = storage.fetchall(
            conn,
            f"""
            SELECT row_index FROM table_cells
            WHERE table_id = ? AND column_index = ? AND value_type = 'integer'
              AND normalized_value IN ({placeholders})
            ORDER BY row_index LIMIT ?
            """,
            (table_id, year_column, *year_values, MAX_EXACT_YEAR_LOOKUP_YEARS),
        )
    row_indexes = [row["row_index"] for row in candidate_rows]
    if len(row_indexes) != len(years) or len(set(row_indexes)) != len(years):
        return None
    cell_placeholders = ",".join("?" for _ in row_indexes)
    cells = storage.fetchall(
        conn,
        f"""
        SELECT * FROM table_cells WHERE table_id = ? AND row_index IN ({cell_placeholders})
        ORDER BY row_index, column_index
        """,
        (table_id, *row_indexes),
    )
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_row[cell["row_index"]].append(dict(cell))
    value_column = _value_column(columns, by_row, year_column, intent)
    if value_column is None:
        return None
    selected = []
    for year in years:
        row = next(
            (items for items in by_row.values()
             if any(cell["column_index"] == year_column and cell["raw_value"] == str(year) for cell in items)),
            None,
        )
        if row is None:
            return None
        value = next((cell for cell in row if cell["column_index"] == value_column), None)
        if value is None or value["value_type"] not in NUMERIC_TYPES or value["normalized_value"] is None:
            return None
        selected.append({"year": year, "cells": row, "value_column": value_column})
    return {
        "year_column": year_column,
        "value_column": value_column,
        "rows": selected,
    }


def _year_column(conn, table_id: str, columns, years: list[int]) -> int | None:
    declared = [
        column["column_index"] for column in columns
        if str(column["normalized_header"] or "").casefold() == "year"
        or str(column["raw_header"] or "").strip().casefold() == "year"
    ]
    candidates = declared or [column["column_index"] for column in columns]
    scored = []
    for column_index in candidates:
        values = storage.fetchall(
            conn,
            """
            SELECT COUNT(DISTINCT row_index) AS count FROM table_cells
            WHERE table_id = ? AND column_index = ? AND value_type = 'integer'
              AND normalized_value IN ({})
            """.format(",".join("?" for _ in years)),
            (table_id, column_index, *(str(year) for year in years)),
        )[0]["count"]
        scored.append((int(values), column_index))
    if not scored:
        return None
    score, column_index = max(scored, key=lambda item: (item[0], -item[1]))
    return column_index if score >= len(years) else None


def _value_column(columns, rows_by_index, year_column: int, intent: str) -> int | None:
    if intent == "co2":
        preferred = [
            column["column_index"] for column in columns
            if "total_fossil_fuels_and_land_use_change" in str(column["normalized_header"] or "").casefold()
            or str(column["raw_header"] or "").casefold().startswith("total ")
        ]
        if preferred:
            return preferred[0]
    numeric_counts = []
    for column in columns:
        index = column["column_index"]
        if index == year_column or index in {0, 1} and intent == "co2":
            continue
        count = sum(
            1 for row in rows_by_index.values()
            for cell in row
            if cell["column_index"] == index
            and cell["value_type"] in NUMERIC_TYPES
            and cell["normalized_value"] is not None
        )
        numeric_counts.append((count, -index, index))
    if not numeric_counts:
        return None
    count, _, index = max(numeric_counts)
    return index if count == len(rows_by_index) else None


def _column_index(columns, names: set[str]) -> int | None:
    for column in columns:
        if str(column["normalized_header"] or "").casefold() in names:
            return column["column_index"]
    return None


def _exact_year_source(conn, table, columns, result, years: list[int], intent: str, rank: int) -> SourceChunk:
    refs = sorted({cell["cell_ref"] for row in result["rows"] for cell in row["cells"]})
    header_by_index = {column["column_index"]: column for column in columns}
    value_header = _display_header(conn, table["id"], header_by_index[result["value_column"]])
    year_header = _display_header(conn, table["id"], header_by_index[result["year_column"]])
    lines = [
        "Deterministic exact-year lookup over typed CSV rows; no interpolation or nearest-chunk selection was used.",
        f"Columns: {year_header} | {value_header}",
    ]
    if intent == "co2":
        lines.append("Selection: Entity = World (Code = OWID_WRL). The CSV does not declare a unit for this total column; values below are unchanged from the file.")
    else:
        lines.append("The NOAA CSV declares units as Degrees Celsius and a 1901-2000 base period.")
    for row in result["rows"]:
        by_index = {cell["column_index"]: cell for cell in row["cells"]}
        lines.append(
            f"{row['year']} | {by_index[result['value_column']]['raw_value']} | "
            + " | ".join(cell["raw_value"] for cell in row["cells"])
        )
    evidence = "\n".join(lines)
    digest = hashlib.sha256(
        json.dumps({"table_id": table["id"], "years": years, "refs": refs}, sort_keys=True).encode()
    ).hexdigest()[:16]
    cited_cells = _cell_citation_payloads(conn, table["id"], refs)
    header_refs = sorted({cell["header_ref"] for cell in cited_cells if cell.get("header_ref")})
    return SourceChunk(
        rank=rank,
        source_id=f"S{rank}",
        doc_id=table["doc_id"],
        doc_name=table["display_name"] or str(table["path"]).rsplit("\\", 1)[-1],
        chunk_id=f"exact-year-{digest}",
        source_kind="cell",
        score=1.0,
        final_score=1.0,
        snippet="\n".join(lines[:4]),
        evidence_text=evidence,
        block_type="table",
        page_number=table["page_number"],
        page_end=table["page_end"],
        table_id=table["id"],
        table_title=table["caption"],
        sheet_name=table["sheet_name"],
        cell_refs=refs,
        header_refs=header_refs,
        cells=cited_cells,
        table_operation="lookup",
        table_result=[
            {"year": row["year"], "value": next(cell["raw_value"] for cell in row["cells"] if cell["column_index"] == result["value_column"])}
            for row in result["rows"]
        ],
        provenance={
            "table_id": table["id"],
            "cell_refs": refs,
            "operation": "exact_year_lookup",
            "years": years,
            "intent": intent,
        },
        context_selection={"decision": "deterministic exact-year lookup", "objective": 1.0},
    )


def _display_header(conn, table_id: str, column) -> str:
    declared = column["raw_header"] or column["normalized_header"]
    if declared and not str(declared).lstrip().startswith("#"):
        return str(declared)
    candidates = storage.fetchall(
        conn,
        """
        SELECT raw_value FROM table_cells
        WHERE table_id = ? AND column_index = ? AND row_index < 16
          AND value_type = 'text' AND raw_value <> ''
        ORDER BY row_index
        """,
        (table_id, column["column_index"]),
    )
    values = [str(candidate["raw_value"]) for candidate in candidates]
    for value in values:
        if value.casefold() == "year" or "departure from average" in value.casefold():
            return value
    for value in values:
        if re.search(r"\b(?:year|departure|average)\b", value, re.IGNORECASE):
            return value
    return str(declared or f"column_{column['column_index'] + 1}")


def _exact_year_trace(status, reason, started, bounds, **extra):
    return {
        "route": "typed_table_exact_year",
        "status": status,
        "fallback_reason": reason,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_calls": 0,
        "bounds": bounds,
        **extra,
    }


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _document_scan_trace(status: str, reason: str | None, started: float, bounds: dict[str, int], **extra):
    return {
        "route": "named_document_unit_scan",
        "status": status,
        "fallback_reason": reason,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_calls": 0,
        "bounds": bounds,
        **extra,
    }


@dataclass(frozen=True)
class TableExecution:
    rows: list[dict[str, Any]]
    text: str
    trace: dict[str, Any]


def execute_table_route(app_state, prompt: str) -> tuple[list[str], list[SourceChunk], dict[str, Any]]:
    """Plan and execute a table question, returning an explicit fallback trace."""
    started = time.perf_counter()
    if not getattr(app_state.settings, "table_execution", True):
        return [], [], _route_trace("disabled", "feature_disabled", started)
    decision = plan_table_query(app_state.sqlite, prompt)
    if decision.plan is None:
        return [], [], _route_trace(decision.status, decision.reason, started, decision=decision)
    try:
        execution = execute_plan(app_state.sqlite, decision.plan)
    except (TableExecutionError, UnsafeTablePlan, TimeoutError) as error:
        return [], [], _route_trace(
            "fallback", str(error), started, decision=decision, error_type=type(error).__name__
        )
    if not execution.rows:
        return [], [], _route_trace("fallback", "no_results", started, decision=decision)

    sources = _sources_for_execution(app_state.sqlite, execution, decision.plan)
    contexts = [
        f"[Source: {source.source_id} | {source.doc_name} | Structured table evidence]\n{source.evidence_text}"
        for source in sources
    ]
    trace = {
        **execution.trace,
        "status": "executed",
        "fallback_reason": None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "source_count": len(sources),
        "model_calls": 0,
        "candidate_table_ids": list(decision.candidate_table_ids),
    }
    return contexts, sources, trace


def execute_plan(conn, plan: TablePlan) -> TableExecution:
    """Execute only a previously validated plan; never accepts raw SQL."""
    started = time.perf_counter()
    deadline = started + EXECUTION_TIMEOUT_MS / 1000

    def progress() -> int:
        return 1 if time.perf_counter() > deadline else 0

    with storage.SQLITE_LOCK:
        conn.set_progress_handler(progress, SQL_PROGRESS_STEPS)
        try:
            tables = _load_tables(conn, plan.table_ids, deadline)
            if len(tables) != len(plan.table_ids):
                raise TableExecutionError("unknown_table")
            columns = _load_columns(conn, plan.table_ids, deadline)
            _validate_columns(plan, columns, deadline)
            cells = _load_cells(conn, plan.table_ids, deadline)
        except Exception as error:
            if "interrupted" in str(error).casefold():
                raise TimeoutError("table_execution_timeout") from error
            raise
        finally:
            conn.set_progress_handler(None, 0)
    _check_deadline(deadline)
    rows = _execute_cells(plan, tables, columns, cells, deadline)
    text = _format_rows(plan, tables, columns, rows, deadline)[:MAX_CONTEXT_CHARACTERS]
    result_cell_set: set[str] = set()
    for row in rows:
        _check_deadline(deadline)
        result_cell_set.update(row.get("cell_refs", []))
    result_cells = sorted(result_cell_set)
    _check_deadline(deadline)
    return TableExecution(rows, text, {
        "route": "typed_table",
        "validated_plan": plan.trace_payload(),
        "selected_table_ids": list(plan.table_ids),
        "selected_columns": sorted({
            item for item in (*plan.select_columns, plan.value_column, plan.group_column) if item is not None
        }),
        "operation": plan.operation,
        "result_cell_refs": result_cells,
        "result_count": len(rows),
        "scanned_cell_count": len(cells),
        "bounds": {
            "max_tables": 16,
            "max_scanned_cells": MAX_SCANNED_CELLS,
            "max_results": plan.limit,
            "max_context_characters": MAX_CONTEXT_CHARACTERS,
            "timeout_ms": EXECUTION_TIMEOUT_MS,
        },
        "execution_ms": round((time.perf_counter() - started) * 1000, 3),
    })


def _load_tables(conn, table_ids: tuple[str, ...], deadline: float) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _ in table_ids)
    rows = conn.execute(
        f"""
        SELECT tables.*, documents.display_name, documents.path
        FROM tables JOIN documents ON documents.id = tables.doc_id
        WHERE tables.id IN ({placeholders}) ORDER BY tables.id
        """,
        table_ids,
    ).fetchall()
    result = {}
    for row in rows:
        _check_deadline(deadline)
        result[row["id"]] = dict(row)
    return result


def _load_columns(conn, table_ids: tuple[str, ...], deadline: float) -> dict[str, dict[int, dict[str, Any]]]:
    placeholders = ",".join("?" for _ in table_ids)
    rows = conn.execute(
        f"SELECT * FROM table_columns WHERE table_id IN ({placeholders}) ORDER BY table_id, column_index",
        table_ids,
    ).fetchall()
    result: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        _check_deadline(deadline)
        result[row["table_id"]][row["column_index"]] = dict(row)
    return result


def _load_cells(conn, table_ids: tuple[str, ...], deadline: float) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in table_ids)
    rows = conn.execute(
        f"""
        SELECT * FROM table_cells WHERE table_id IN ({placeholders})
        ORDER BY table_id, row_index, column_index LIMIT ?
        """,
        (*table_ids, MAX_SCANNED_CELLS + 1),
    ).fetchall()
    if len(rows) > MAX_SCANNED_CELLS:
        raise TableExecutionError("cell_scan_limit")
    result = []
    for row in rows:
        _check_deadline(deadline)
        result.append(dict(row))
    return result


def _validate_columns(plan: TablePlan, columns: dict[str, dict[int, dict[str, Any]]], deadline: float) -> None:
    _check_deadline(deadline)
    requested = {
        *plan.select_columns,
        *(item.column_index for item in plan.filters),
        *(() if plan.value_column is None else (plan.value_column,)),
        *(() if plan.group_column is None else (plan.group_column,)),
    }
    if requested and any(not requested <= set(columns[table_id]) for table_id in plan.table_ids):
        raise TableExecutionError("unknown_column")


def _execute_cells(
    plan: TablePlan,
    tables: dict[str, dict[str, Any]],
    columns: dict[str, dict[int, dict[str, Any]]],
    cells: list[dict[str, Any]],
    deadline: float,
) -> list[dict[str, Any]]:
    _check_deadline(deadline)
    if plan.operation == "lookup":
        return _lookup(plan, cells, deadline)
    if len(plan.table_ids) != 1:
        raise TableExecutionError("ambiguous_multi_table_operation")
    table_id = plan.table_ids[0]
    by_row = _rows_for_table(cells, table_id, deadline)
    # Row zero is the declared header row and is never aggregated as data.
    data_rows = []
    for index, row in sorted(by_row.items()):
        _check_deadline(deadline)
        if index > 0 and _matches_filters(row, plan.filters, deadline):
            data_rows.append(row)
    if plan.operation == "filter":
        return [_row_result(table_id, row, deadline) for row in data_rows[:plan.limit]]
    if plan.operation == "count":
        verification_refs = []
        for row in data_rows:
            _check_deadline(deadline)
            if row:
                verification_refs.append(row[min(row)]["cell_ref"])
        return [{
            "table_id": table_id,
            "value": str(len(data_rows)),
            "unit": None,
            "cell_refs": [],
            "verification_cell_refs": verification_refs,
        }]
    if plan.operation in {"compare", "difference", "percentage"}:
        return _binary_result(plan, cells, deadline)
    values = _numeric_values(data_rows, plan.value_column, plan.target_unit, deadline)
    if plan.operation in {"sum", "mean", "min", "max"}:
        return _aggregate_result(plan, table_id, values, deadline)
    if plan.operation == "sort":
        reverse = plan.sort_direction == "desc"
        ordered = sorted(values, key=lambda item: (item[0], item[2]["row_index"]), reverse=reverse)
        _check_deadline(deadline)
        return [_row_result(table_id, row, deadline) for _, row, _ in ordered[:plan.limit]]
    if plan.operation == "group":
        return _group_result(plan, table_id, data_rows, deadline)
    raise TableExecutionError("unsupported_operation")


def _lookup(plan: TablePlan, cells: list[dict[str, Any]], deadline: float) -> list[dict[str, Any]]:
    matched = []
    for cell in cells:
        _check_deadline(deadline)
        if plan.operand_cell_refs and cell["cell_ref"] not in plan.operand_cell_refs:
            continue
        unit_values = _unit_values(cell, plan.target_unit) if plan.target_unit else []
        if plan.target_unit and not unit_values:
            continue
        if plan.search_text and plan.search_text.casefold() not in cell["raw_value"].casefold():
            continue
        if not (plan.operand_cell_refs or plan.target_unit or plan.search_text):
            continue
        values = unit_values or [(cell["normalized_value"] if cell["normalized_value"] is not None else cell["raw_value"], None)]
        for value, matched_text in values:
            matched.append({
                "table_id": cell["table_id"],
                "row_index": cell["row_index"],
                "column_index": cell["column_index"],
                "value": value,
                "raw_value": cell["raw_value"],
                "matched_text": matched_text,
                "unit": cell["unit"] or plan.target_unit,
                "cell_refs": [cell["cell_ref"]],
                "page_number": cell["page_number"],
                "sheet_name": cell["sheet_name"],
                "bounding_box": cell["bounding_box"],
            })
            if len(matched) >= plan.limit:
                return matched
    return matched


def _rows_for_table(cells: list[dict[str, Any]], table_id: str, deadline: float) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        _check_deadline(deadline)
        if cell["table_id"] == table_id:
            result[cell["row_index"]][cell["column_index"]] = cell
    return result


def _matches_filters(row: dict[int, dict[str, Any]], filters: tuple[TableFilter, ...], deadline: float) -> bool:
    for item in filters:
        _check_deadline(deadline)
        cell = row.get(item.column_index)
        if cell is None or (item.unit and not _has_unit(cell, item.unit)):
            return False
        left = cell["normalized_value"] if cell["normalized_value"] is not None else cell["raw_value"]
        if not _compare(left, item.value, item.operator):
            return False
    return True


def _compare(left: str, right: str, operator: str) -> bool:
    if operator == "contains":
        return right.casefold() in left.casefold()
    try:
        pair: tuple[Any, Any] = (Decimal(left), Decimal(right))
    except InvalidOperation:
        pair = (left.casefold(), right.casefold())
    return {
        "eq": pair[0] == pair[1],
        "ne": pair[0] != pair[1],
        "gt": pair[0] > pair[1],
        "gte": pair[0] >= pair[1],
        "lt": pair[0] < pair[1],
        "lte": pair[0] <= pair[1],
    }[operator]


def _numeric_values(rows, column: int | None, requested_unit: str | None, deadline: float):
    if column is None:
        raise TableExecutionError("value_column_required")
    result = []
    units = set()
    for row in rows:
        _check_deadline(deadline)
        cell = row.get(column)
        if not cell or cell["value_type"] not in NUMERIC_TYPES or cell["normalized_value"] is None:
            continue
        if requested_unit and not _has_unit(cell, requested_unit):
            continue
        units.add(cell["unit"])
        result.append((Decimal(cell["normalized_value"]), row, cell))
    if len(units) > 1:
        raise TableExecutionError("mixed_incompatible_units")
    if not result:
        raise TableExecutionError("no_numeric_values")
    return result


def _aggregate_result(plan: TablePlan, table_id: str, values, deadline: float) -> list[dict[str, Any]]:
    _check_deadline(deadline)
    numbers = [item[0] for item in values]
    _check_deadline(deadline)
    if plan.operation == "sum":
        value = sum(numbers, Decimal(0))
        cells = [item[2] for item in values]
    elif plan.operation == "mean":
        value = sum(numbers, Decimal(0)) / Decimal(len(numbers))
        cells = [item[2] for item in values]
    elif plan.operation == "min":
        value, _, cell = min(values, key=lambda item: item[0])
        cells = [cell]
    else:
        value, _, cell = max(values, key=lambda item: item[0])
        cells = [cell]
    refs = []
    for cell in cells:
        _check_deadline(deadline)
        refs.append(cell["cell_ref"])
    result = {
        "table_id": table_id,
        "value": _decimal_text(value),
        "unit": cells[0]["unit"],
        "cell_refs": refs,
    }
    if plan.operation in {"min", "max"}:
        verification_refs = []
        for item in values:
            _check_deadline(deadline)
            verification_refs.append(item[2]["cell_ref"])
        result["verification_cell_refs"] = verification_refs
    return [result]


def _binary_result(plan: TablePlan, cells: list[dict[str, Any]], deadline: float) -> list[dict[str, Any]]:
    operands = []
    for ref in plan.operand_cell_refs:
        for cell in cells:
            _check_deadline(deadline)
            if cell["cell_ref"] == ref:
                operands.append(cell)
    if len(operands) != 2 or len({cell["cell_ref"] for cell in operands}) != 2:
        raise TableExecutionError("missing_or_ambiguous_operand")
    if any(cell["value_type"] not in NUMERIC_TYPES or cell["normalized_value"] is None for cell in operands):
        raise TableExecutionError("non_numeric_operand")
    units = {cell["unit"] for cell in operands}
    if len(units) != 1:
        raise TableExecutionError("mixed_incompatible_units")
    left, right = (Decimal(cell["normalized_value"]) for cell in operands)
    if plan.operation == "compare":
        value = "equal" if left == right else "greater" if left > right else "less"
        unit = None
    elif plan.operation == "difference":
        value = _decimal_text(left - right)
        unit = operands[0]["unit"]
    else:
        if right == 0:
            raise TableExecutionError("division_by_zero")
        value = _decimal_text((left - right) / right * Decimal(100))
        unit = "%"
    return [{
        "table_id": operands[0]["table_id"],
        "value": value,
        "unit": unit,
        "cell_refs": [cell["cell_ref"] for cell in operands],
    }]


def _group_result(plan: TablePlan, table_id: str, rows, deadline: float) -> list[dict[str, Any]]:
    if plan.group_column is None:
        raise TableExecutionError("group_column_required")
    groups: dict[str, list[dict[int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        _check_deadline(deadline)
        cell = row.get(plan.group_column)
        if cell:
            groups[cell["raw_value"]].append(row)
    output = []
    aggregate = plan.aggregate or "count"
    for key in sorted(groups, key=str.casefold):
        _check_deadline(deadline)
        group_rows = groups[key]
        if aggregate == "count":
            value, unit, refs = str(len(group_rows)), None, [row[plan.group_column]["cell_ref"] for row in group_rows]
        else:
            nested = TablePlan(
                operation=aggregate,
                table_ids=plan.table_ids,
                value_column=plan.value_column,
                target_unit=plan.target_unit,
            )
            values = _numeric_values(group_rows, plan.value_column, plan.target_unit, deadline)
            item = _aggregate_result(nested, table_id, values, deadline)[0]
            value, unit, refs = item["value"], item["unit"], item["cell_refs"]
        output.append({"table_id": table_id, "group": key, "value": value, "unit": unit, "cell_refs": refs})
        if len(output) >= plan.limit:
            break
    return output


def _row_result(table_id: str, row: dict[int, dict[str, Any]], deadline: float) -> dict[str, Any]:
    _check_deadline(deadline)
    ordered = [row[index] for index in sorted(row)]
    return {
        "table_id": table_id,
        "row_index": ordered[0]["row_index"] if ordered else None,
        "values": [cell["raw_value"] for cell in ordered],
        "cell_refs": [cell["cell_ref"] for cell in ordered],
    }


def _has_unit(cell: dict[str, Any], target: str) -> bool:
    return bool(_unit_values(cell, target))


def _unit_values(cell: dict[str, Any], target: str | None) -> list[tuple[str, str | None]]:
    if not target:
        return []
    stored = cell.get("unit")
    if stored == target or (len(target) > 1 and stored and stored.casefold() == target.casefold()):
        normalized = cell.get("normalized_value")
        if normalized is not None:
            return [(normalized, None)]
    raw = cell.get("raw_value") or ""
    number = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    flags = 0 if len(target) == 1 else re.IGNORECASE
    uncertainty = rf"(?:\s*(?:±|\+/-)\s*{number})?"
    suffix = "%" if target == "%" else rf"{re.escape(target)}(?![A-Za-zµμ])"
    values = []
    for match in re.finditer(rf"({number}){uncertainty}\s*{suffix}", raw, flags):
        try:
            normalized = _decimal_text(Decimal(match.group(1).replace(",", "")))
        except InvalidOperation:
            continue
        values.append((normalized, match.group(0)))
    return values


def _format_rows(plan, tables, columns, rows, deadline: float) -> str:
    lines = [f"Deterministic table operation: {plan.operation}."]
    for row in rows:
        _check_deadline(deadline)
        table = tables[row["table_id"]]
        location = table.get("sheet_name") or (f"page {table['page_number']}" if table.get("page_number") else "table")
        payload = {key: value for key, value in row.items() if key not in {"table_id", "bounding_box"}}
        serialized = json.dumps(payload, ensure_ascii=False)
        _check_deadline(deadline)
        lines.append(f"{table.get('display_name') or table.get('path')} | {location} | {serialized}")
    return "\n".join(lines)


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise TimeoutError("table_execution_timeout")


def _sources_for_execution(conn, execution: TableExecution, plan: TablePlan) -> list[SourceChunk]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in execution.rows:
        grouped[row["table_id"]].append(row)
    sources = []
    for rank, table_id in enumerate(plan.table_ids, start=1):
        rows = grouped.get(table_id)
        if not rows:
            continue
        table = conn.execute(
            """
            SELECT tables.*, documents.display_name, documents.path
            FROM tables JOIN documents ON documents.id = tables.doc_id WHERE tables.id = ?
            """,
            (table_id,),
        ).fetchone()
        refs = sorted({ref for row in rows for ref in row.get("cell_refs", [])})
        verification_refs = sorted({
            ref for row in rows for ref in row.get("verification_cell_refs", [])
        })
        cited_cells = _cell_citation_payloads(conn, table_id, sorted(set(refs) | set(verification_refs)))
        header_refs = sorted({
            cell["header_ref"] for cell in cited_cells if cell.get("header_ref")
        })
        evidence = "\n".join(
            line for line in execution.text.splitlines()
            if line.startswith("Deterministic") or (table["display_name"] or table["path"]) in line
        )
        result_payload = [
            {key: value for key, value in row.items() if key not in {"bounding_box"}}
            for row in rows
        ]
        digest_input = json.dumps(
            {"table_id": table_id, "operation": plan.operation, "rows": result_payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
        sources.append(SourceChunk(
            rank=rank,
            source_id=f"S{rank}",
            doc_id=table["doc_id"],
            doc_name=table["display_name"] or str(table["path"]).rsplit("\\", 1)[-1],
            chunk_id=f"table-result-{digest}",
            source_kind="cell" if refs else "table",
            score=1.0,
            final_score=1.0,
            snippet=evidence[:1000],
            evidence_text=evidence,
            block_type="table",
            page_number=table["page_number"],
            page_end=table["page_end"],
            bounding_box=json.loads(table["bounding_box"]) if table["bounding_box"] else None,
            table_id=table_id,
            table_title=table["caption"],
            sheet_name=table["sheet_name"],
            table_bounding_box=json.loads(table["bounding_box"]) if table["bounding_box"] else None,
            cell_refs=refs,
            verification_cell_refs=verification_refs,
            header_refs=header_refs,
            cells=cited_cells,
            table_operation=plan.operation,
            table_result=result_payload,
            provenance={
                "table_id": table_id,
                "cell_refs": refs,
                "operation": plan.operation,
                "validated_plan": plan.trace_payload(),
            },
            context_selection={"decision": "deterministic table execution", "objective": 1.0},
        ))
    return sources


def _cell_citation_payloads(conn, table_id: str, cell_refs: list[str]) -> list[dict[str, Any]]:
    if not cell_refs:
        return []
    placeholders = ",".join("?" for _ in cell_refs)
    rows = conn.execute(
        f"""
        SELECT cells.*, columns.raw_header, columns.normalized_header,
               headers.cell_ref AS header_ref
        FROM table_cells AS cells
        LEFT JOIN table_columns AS columns
          ON columns.table_id = cells.table_id AND columns.column_index = cells.column_index
        LEFT JOIN table_cells AS headers
          ON headers.table_id = cells.table_id
         AND headers.row_index = 0
         AND headers.column_index = cells.column_index
        WHERE cells.table_id = ? AND cells.cell_ref IN ({placeholders})
        ORDER BY cells.row_index, cells.column_index
        """,
        (table_id, *cell_refs),
    ).fetchall()
    return [
        {
            "cell_ref": row["cell_ref"],
            "row_index": row["row_index"],
            "column_index": row["column_index"],
            "raw_value": row["raw_value"],
            "normalized_value": row["normalized_value"],
            "value_type": row["value_type"],
            "unit": row["unit"],
            "header_ref": row["header_ref"],
            "header": row["raw_header"] or row["normalized_header"],
            "sheet_name": row["sheet_name"],
            "page_number": row["page_number"],
            "bounding_box": json.loads(row["bounding_box"]) if row["bounding_box"] else None,
        }
        for row in rows
    ]


def _decimal_text(value: Decimal) -> str:
    return "0" if value.is_zero() else format(value.normalize(), "f")


def _route_trace(status, reason, started, *, decision: PlanDecision | None = None, error_type: str | None = None):
    return {
        "route": "typed_table",
        "status": status,
        "validated_plan": decision.plan.trace_payload() if decision and decision.plan else None,
        "candidate_table_ids": list(decision.candidate_table_ids) if decision else [],
        "fallback_reason": reason,
        "error_type": error_type,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_calls": 0,
    }
