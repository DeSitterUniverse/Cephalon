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
            tables = _load_tables(conn, plan.table_ids)
            if len(tables) != len(plan.table_ids):
                raise TableExecutionError("unknown_table")
            columns = _load_columns(conn, plan.table_ids)
            _validate_columns(plan, columns)
            cells = _load_cells(conn, plan.table_ids)
        except Exception as error:
            if "interrupted" in str(error).casefold():
                raise TimeoutError("table_execution_timeout") from error
            raise
        finally:
            conn.set_progress_handler(None, 0)
    rows = _execute_cells(plan, tables, columns, cells)
    text = _format_rows(plan, tables, columns, rows)[:MAX_CONTEXT_CHARACTERS]
    result_cells = sorted({ref for row in rows for ref in row.get("cell_refs", [])})
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


def _load_tables(conn, table_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _ in table_ids)
    rows = conn.execute(
        f"""
        SELECT tables.*, documents.display_name, documents.path
        FROM tables JOIN documents ON documents.id = tables.doc_id
        WHERE tables.id IN ({placeholders}) ORDER BY tables.id
        """,
        table_ids,
    ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def _load_columns(conn, table_ids: tuple[str, ...]) -> dict[str, dict[int, dict[str, Any]]]:
    placeholders = ",".join("?" for _ in table_ids)
    rows = conn.execute(
        f"SELECT * FROM table_columns WHERE table_id IN ({placeholders}) ORDER BY table_id, column_index",
        table_ids,
    ).fetchall()
    result: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        result[row["table_id"]][row["column_index"]] = dict(row)
    return result


def _load_cells(conn, table_ids: tuple[str, ...]) -> list[dict[str, Any]]:
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
    return [dict(row) for row in rows]


def _validate_columns(plan: TablePlan, columns: dict[str, dict[int, dict[str, Any]]]) -> None:
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
) -> list[dict[str, Any]]:
    if plan.operation == "lookup":
        return _lookup(plan, cells)
    if len(plan.table_ids) != 1:
        raise TableExecutionError("ambiguous_multi_table_operation")
    table_id = plan.table_ids[0]
    by_row = _rows_for_table(cells, table_id)
    # Row zero is the declared header row and is never aggregated as data.
    data_rows = [row for index, row in sorted(by_row.items()) if index > 0 and _matches_filters(row, plan.filters)]
    if plan.operation == "filter":
        return [_row_result(table_id, row) for row in data_rows[:plan.limit]]
    if plan.operation == "count":
        return [{
            "table_id": table_id,
            "value": str(len(data_rows)),
            "unit": None,
            "cell_refs": [],
            "verification_cell_refs": [
                row[min(row)]["cell_ref"] for row in data_rows if row
            ],
        }]
    if plan.operation in {"compare", "difference", "percentage"}:
        return _binary_result(plan, cells)
    values = _numeric_values(data_rows, plan.value_column, plan.target_unit)
    if plan.operation in {"sum", "mean", "min", "max"}:
        return _aggregate_result(plan, table_id, values)
    if plan.operation == "sort":
        reverse = plan.sort_direction == "desc"
        ordered = sorted(values, key=lambda item: (item[0], item[2]["row_index"]), reverse=reverse)
        return [_row_result(table_id, row) for _, row, _ in ordered[:plan.limit]]
    if plan.operation == "group":
        return _group_result(plan, table_id, data_rows)
    raise TableExecutionError("unsupported_operation")


def _lookup(plan: TablePlan, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched = []
    for cell in cells:
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


def _rows_for_table(cells: list[dict[str, Any]], table_id: str) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        if cell["table_id"] == table_id:
            result[cell["row_index"]][cell["column_index"]] = cell
    return result


def _matches_filters(row: dict[int, dict[str, Any]], filters: tuple[TableFilter, ...]) -> bool:
    for item in filters:
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


def _numeric_values(rows, column: int | None, requested_unit: str | None):
    if column is None:
        raise TableExecutionError("value_column_required")
    result = []
    units = set()
    for row in rows:
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


def _aggregate_result(plan: TablePlan, table_id: str, values) -> list[dict[str, Any]]:
    numbers = [item[0] for item in values]
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
    result = {
        "table_id": table_id,
        "value": _decimal_text(value),
        "unit": cells[0]["unit"],
        "cell_refs": [cell["cell_ref"] for cell in cells],
    }
    if plan.operation in {"min", "max"}:
        result["verification_cell_refs"] = [item[2]["cell_ref"] for item in values]
    return [result]


def _binary_result(plan: TablePlan, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operands = [cell for ref in plan.operand_cell_refs for cell in cells if cell["cell_ref"] == ref]
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


def _group_result(plan: TablePlan, table_id: str, rows) -> list[dict[str, Any]]:
    if plan.group_column is None:
        raise TableExecutionError("group_column_required")
    groups: dict[str, list[dict[int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        cell = row.get(plan.group_column)
        if cell:
            groups[cell["raw_value"]].append(row)
    output = []
    aggregate = plan.aggregate or "count"
    for key in sorted(groups, key=str.casefold):
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
            item = _aggregate_result(nested, table_id, _numeric_values(group_rows, plan.value_column, plan.target_unit))[0]
            value, unit, refs = item["value"], item["unit"], item["cell_refs"]
        output.append({"table_id": table_id, "group": key, "value": value, "unit": unit, "cell_refs": refs})
        if len(output) >= plan.limit:
            break
    return output


def _row_result(table_id: str, row: dict[int, dict[str, Any]]) -> dict[str, Any]:
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


def _format_rows(plan, tables, columns, rows) -> str:
    lines = [f"Deterministic table operation: {plan.operation}."]
    for row in rows:
        table = tables[row["table_id"]]
        location = table.get("sheet_name") or (f"page {table['page_number']}" if table.get("page_number") else "table")
        payload = {key: value for key, value in row.items() if key not in {"table_id", "bounding_box"}}
        lines.append(f"{table.get('display_name') or table.get('path')} | {location} | {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


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
