from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any


TABLE_PARSER_VERSION = "cephalon-typed-tables-2026-08-v2"
MAX_TABLE_ROWS = 100_000
MAX_TABLE_COLUMNS = 256
MAX_TABLE_CELLS = 2_000_000
MAX_CELL_CHARACTERS = 32_768

_NUMBER = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_PATTERN = re.compile(rf"^({_NUMBER})$")
_PERCENT_PATTERN = re.compile(rf"^({_NUMBER})\s*%$")
_UNIT_PATTERN = re.compile(rf"^({_NUMBER})\s*([A-Za-zµμ°][A-Za-z0-9µμ°/^·⋅.-]*)$")


def stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":"))
    return f"{kind}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class TypedValue:
    raw_value: str
    normalized_value: str | None
    value_type: str
    unit: str | None = None
    parse_warnings: tuple[str, ...] = ()


@dataclass
class TableCell:
    row_index: int
    column_index: int
    cell_ref: str
    raw_value: str
    normalized_value: str | None
    value_type: str
    unit: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    formula: str | None = None
    effective_value: str | None = None
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class TableColumn:
    column_index: int
    raw_header: str
    normalized_header: str
    inferred_type: str
    inferred_unit: str | None = None
    header_cell_ref: str | None = None


@dataclass
class TableRow:
    row_index: int
    page_number: int | None = None
    sheet_name: str | None = None
    row_label: str | None = None


@dataclass
class StructuredTable:
    source_type: str
    table_index: int
    raw_rows: list[list[str]]
    cells: list[TableCell]
    columns: list[TableColumn]
    rows: list[TableRow]
    sheet_name: str | None = None
    sheet_index: int | None = None
    page_number: int | None = None
    page_end: int | None = None
    caption: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    parser_version: str = TABLE_PARSER_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    table_key: str = ""

    def __post_init__(self) -> None:
        if not self.table_key:
            location = self.sheet_name if self.sheet_name is not None else self.page_number
            self.table_key = stable_id("table-key", self.source_type, location, self.table_index, self.raw_rows)

    @property
    def row_count(self) -> int:
        return len(self.raw_rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.raw_rows), default=0)

    @property
    def text(self) -> str:
        return "\n".join("\t".join(row) for row in self.raw_rows)


def raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def type_value(value: Any, *, number_format: str | None = None) -> TypedValue:
    raw = raw_text(value)
    if len(raw) > MAX_CELL_CHARACTERS:
        return TypedValue(raw, None, "text", parse_warnings=("cell_length_limit_exceeded",))
    stripped = raw.strip()
    if value is None or not stripped:
        return TypedValue(raw, None, "missing")
    if isinstance(value, bool):
        return TypedValue(raw, "true" if value else "false", "boolean")
    if isinstance(value, datetime):
        return TypedValue(raw, value.isoformat(), "datetime")
    if isinstance(value, date):
        return TypedValue(raw, value.isoformat(), "date")
    if isinstance(value, (int, float, Decimal)):
        decimal_value = Decimal(str(value))
        if _is_percentage_number_format(number_format):
            # XLSX stores 12.5% as 0.125. Normalized percentages consistently
            # use percentage points so XLSX and textual/CSV values agree.
            return TypedValue(raw, _decimal_string(decimal_value * 100), "percentage", "%")
        return TypedValue(raw, _decimal_string(decimal_value), "integer" if isinstance(value, int) else "decimal")
    lowered = stripped.casefold()
    if lowered in {"true", "false"}:
        return TypedValue(raw, lowered, "boolean")
    percent = _PERCENT_PATTERN.fullmatch(stripped)
    if percent:
        normalized = _parse_decimal(percent.group(1))
        return TypedValue(raw, normalized, "percentage", "%") if normalized is not None else TypedValue(raw, None, "text")
    number = _NUMBER_PATTERN.fullmatch(stripped)
    if number:
        normalized = _parse_decimal(number.group(1))
        if normalized is not None:
            integer = re.fullmatch(r"[-+]?\d+(?:,\d{3})*", stripped) is not None
            return TypedValue(raw, normalized, "integer" if integer else "decimal")
    unit = _UNIT_PATTERN.fullmatch(stripped)
    if unit:
        normalized = _parse_decimal(unit.group(1))
        if normalized is not None:
            return TypedValue(raw, normalized, "decimal", unit.group(2))
    return TypedValue(raw, stripped, "text")


def build_table(
    rows: list[list[Any]],
    *,
    source_type: str,
    table_index: int,
    sheet_name: str | None = None,
    sheet_index: int | None = None,
    page_number: int | None = None,
    page_end: int | None = None,
    bounding_box: tuple[float, float, float, float] | None = None,
    cell_boxes: list[list[tuple[float, float, float, float] | None]] | None = None,
    formulas: dict[tuple[int, int], str] | None = None,
    effective_values: dict[tuple[int, int], Any] | None = None,
    number_formats: dict[tuple[int, int], str] | None = None,
    merged_ranges: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> StructuredTable:
    if len(rows) > MAX_TABLE_ROWS:
        rows = rows[:MAX_TABLE_ROWS]
        warnings = [*(warnings or []), "row_limit_reached"]
    width = min(max((len(row) for row in rows), default=0), MAX_TABLE_COLUMNS)
    normalized_rows = [[raw_text(value) for value in row[:width]] + [""] * max(0, width - len(row)) for row in rows]
    if len(normalized_rows) * width > MAX_TABLE_CELLS:
        normalized_rows = normalized_rows[: max(1, MAX_TABLE_CELLS // max(1, width))]
        rows = rows[: len(normalized_rows)]
        warnings = [*(warnings or []), "cell_limit_reached"]
    headers = normalized_rows[0] if normalized_rows else []
    typed_cells: list[TableCell] = []
    column_values: list[list[TypedValue]] = [[] for _ in range(width)]
    for row_index, row in enumerate(rows[: len(normalized_rows)]):
        for column_index in range(width):
            value = row[column_index] if column_index < len(row) else None
            formula = (formulas or {}).get((row_index, column_index))
            effective_value = (effective_values or {}).get((row_index, column_index)) if formula else None
            typed_source = effective_value if effective_value is not None else value
            typed = type_value(typed_source, number_format=(number_formats or {}).get((row_index, column_index)))
            cell_ref = _cell_ref(source_type, row_index, column_index, sheet_name, page_number)
            cell_warnings = list(typed.parse_warnings)
            if formula and effective_value is None:
                cell_warnings.append("formula_cached_value_unavailable")
            typed_cells.append(TableCell(
                row_index=row_index,
                column_index=column_index,
                cell_ref=cell_ref,
                raw_value=raw_text(value),
                normalized_value=typed.normalized_value,
                value_type=typed.value_type,
                unit=typed.unit,
                page_number=page_number,
                sheet_name=sheet_name,
                bounding_box=_box_at(cell_boxes, row_index, column_index),
                formula=formula,
                effective_value=raw_text(effective_value) if effective_value is not None else None,
                parse_warnings=cell_warnings,
            ))
            if row_index > 0:
                column_values[column_index].append(typed)
    columns = []
    for index, header in enumerate(headers):
        values = [value for value in column_values[index] if value.value_type != "missing"]
        types = {value.value_type for value in values}
        units = {value.unit for value in values if value.unit}
        inferred_type = next(iter(types)) if len(types) == 1 else "mixed" if types else "unknown"
        columns.append(TableColumn(
            column_index=index,
            raw_header=header,
            normalized_header=normalize_header(header, index),
            inferred_type=inferred_type,
            inferred_unit=next(iter(units)) if len(units) == 1 else None,
            header_cell_ref=_cell_ref(source_type, 0, index, sheet_name, page_number),
        ))
    table_rows = [
        TableRow(index, page_number=page_number, sheet_name=sheet_name, row_label=(row[0] if row else None))
        for index, row in enumerate(normalized_rows)
    ]
    table_provenance = dict(provenance or {})
    if merged_ranges:
        table_provenance["merged_ranges"] = list(merged_ranges)
    return StructuredTable(
        source_type=source_type,
        table_index=table_index,
        raw_rows=normalized_rows,
        cells=typed_cells,
        columns=columns,
        rows=table_rows,
        sheet_name=sheet_name,
        sheet_index=sheet_index,
        page_number=page_number,
        page_end=page_end or page_number,
        bounding_box=bounding_box,
        provenance=table_provenance,
        warnings=list(warnings or []),
    )


def normalize_header(value: str, index: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized or f"column_{index + 1}"


def _parse_decimal(value: str) -> str | None:
    try:
        return _decimal_string(Decimal(value.replace(",", "")))
    except InvalidOperation:
        return None


def _is_percentage_number_format(number_format: str | None) -> bool:
    """Return whether an XLSX format contains a non-literal percent token."""
    if not number_format:
        return False
    quoted = False
    escaped = False
    for character in number_format:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if character == "%" and not quoted:
            return True
    return False


def _decimal_string(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def _cell_ref(source_type: str, row_index: int, column_index: int, sheet: str | None, page: int | None) -> str:
    if source_type == "xlsx":
        column = ""
        value = column_index + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            column = chr(65 + remainder) + column
        return f"{sheet or 'Sheet'}!{column}{row_index + 1}"
    prefix = f"p{page}" if page is not None else source_type
    return f"{prefix}:r{row_index + 1}c{column_index + 1}"


def _box_at(boxes, row_index: int, column_index: int):
    if boxes and row_index < len(boxes) and column_index < len(boxes[row_index]):
        return boxes[row_index][column_index]
    return None
