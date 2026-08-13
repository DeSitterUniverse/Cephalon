from __future__ import annotations

import json
from typing import Iterable

from .table_models import StructuredTable, stable_id


def persistence_rows(doc_id: str, tables: Iterable[StructuredTable]) -> dict[str, list[tuple]]:
    result: dict[str, list[tuple]] = {"tables": [], "columns": [], "rows": [], "cells": []}
    for table in tables:
        table_id = stable_id("tbl", doc_id, table.table_key)
        result["tables"].append((
            table_id,
            doc_id,
            table.source_type,
            table.sheet_name,
            table.sheet_index,
            table.page_number,
            table.page_end,
            table.table_index,
            table.caption,
            _json(table.bounding_box),
            table.row_count,
            table.column_count,
            table.parser_version,
            _json(table.provenance),
            _json(table.warnings),
        ))
        for column in table.columns:
            result["columns"].append((
                stable_id("col", table_id, column.column_index),
                table_id,
                column.column_index,
                column.raw_header,
                column.normalized_header,
                column.inferred_type,
                column.inferred_unit,
                column.header_cell_ref,
            ))
        for row in table.rows:
            result["rows"].append((
                stable_id("row", table_id, row.row_index),
                table_id,
                row.row_index,
                row.page_number,
                row.sheet_name,
                row.row_label,
            ))
        for cell in table.cells:
            result["cells"].append((
                stable_id("cell", table_id, cell.row_index, cell.column_index),
                table_id,
                cell.row_index,
                cell.column_index,
                cell.cell_ref,
                cell.raw_value,
                cell.normalized_value,
                cell.value_type,
                cell.unit,
                cell.page_number,
                cell.sheet_name,
                _json(cell.bounding_box),
                cell.formula,
                cell.effective_value,
                _json(cell.parse_warnings),
            ))
    return result


def load_document_tables(conn, doc_id: str) -> list[dict]:
    tables = conn.execute(
        "SELECT * FROM tables WHERE doc_id = ? ORDER BY table_index, id",
        (doc_id,),
    ).fetchall()
    payload = []
    for table in tables:
        item = dict(table)
        item["columns"] = [dict(row) for row in conn.execute(
            "SELECT * FROM table_columns WHERE table_id = ? ORDER BY column_index", (item["id"],)
        ).fetchall()]
        item["rows"] = [dict(row) for row in conn.execute(
            "SELECT * FROM table_rows WHERE table_id = ? ORDER BY row_index", (item["id"],)
        ).fetchall()]
        item["cells"] = [dict(row) for row in conn.execute(
            "SELECT * FROM table_cells WHERE table_id = ? ORDER BY row_index, column_index", (item["id"],)
        ).fetchall()]
        payload.append(item)
    return payload


def _json(value) -> str | None:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value not in (None, {}, []) else None
