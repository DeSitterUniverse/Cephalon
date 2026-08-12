"""Create bounded, typed plans for deterministic table execution.

The rule router may decline a request at any point. A declined request follows
normal hybrid retrieval; uncertainty is never converted into executable SQL or
guessed arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any

from .. import storage


ALLOWED_OPERATIONS = {
    "lookup", "filter", "sort", "group", "min", "max", "sum", "mean",
    "count", "compare", "difference", "percentage",
}
ALLOWED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}
MAX_PLAN_TABLES = 16
MAX_CANDIDATE_TABLES = 64
MAX_PLANNING_CELLS_PER_TABLE = 5_000
PLANNING_TIMEOUT_MS = 250
MAX_RESULT_ROWS = 24
MAX_COLUMN_INDEX = 255
TABLE_CUES = re.compile(
    r"\b(?:table|row|rows|column|columns|cell|cells|value|values|average|mean|"
    r"sum|total|count|minimum|maximum|highest|lowest|sort|group|difference|percent)\b",
    re.IGNORECASE,
)
QUOTED_TARGET = re.compile(r"[\"“]([^\"”]{4,240})[\"”]")
WORD = re.compile(r"[a-z0-9µμ]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "does", "for", "from", "in", "is",
    "of", "on", "or", "paper", "report", "source", "study", "target", "the",
    "to", "use", "what", "which", "with",
}
GENERIC_IDENTITY_TERMS = {
    "analysis", "augmented", "benchmark", "data", "evaluation", "generation",
    "language", "learning", "model", "models", "paper", "retrieval", "study",
}


class UnsafeTablePlan(ValueError):
    """Raised when a plan violates the deterministic execution contract."""


@dataclass(frozen=True)
class TableFilter:
    column_index: int
    operator: str
    value: str
    unit: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.column_index <= MAX_COLUMN_INDEX:
            raise UnsafeTablePlan("invalid_column_index")
        if self.operator not in ALLOWED_OPERATORS:
            raise UnsafeTablePlan("invalid_operator")


@dataclass(frozen=True)
class TablePlan:
    operation: str
    table_ids: tuple[str, ...]
    value_column: int | None = None
    select_columns: tuple[int, ...] = ()
    filters: tuple[TableFilter, ...] = ()
    group_column: int | None = None
    aggregate: str | None = None
    sort_direction: str = "asc"
    operand_cell_refs: tuple[str, ...] = ()
    target_unit: str | None = None
    search_text: str | None = None
    limit: int = 12

    def __post_init__(self) -> None:
        if self.operation not in ALLOWED_OPERATIONS:
            raise UnsafeTablePlan("invalid_operation")
        if not self.table_ids or len(self.table_ids) > MAX_PLAN_TABLES:
            raise UnsafeTablePlan("table_limit")
        if any(not re.fullmatch(r"tbl-[a-f0-9]{24}", item) for item in self.table_ids):
            raise UnsafeTablePlan("invalid_table_id")
        indices = [*self.select_columns, *(item.column_index for item in self.filters)]
        if self.value_column is not None:
            indices.append(self.value_column)
        if self.group_column is not None:
            indices.append(self.group_column)
        if any(not 0 <= item <= MAX_COLUMN_INDEX for item in indices):
            raise UnsafeTablePlan("invalid_column_index")
        if self.sort_direction not in {"asc", "desc"}:
            raise UnsafeTablePlan("invalid_sort_direction")
        if not 1 <= self.limit <= MAX_RESULT_ROWS:
            raise UnsafeTablePlan("result_limit")
        if self.operation in {"min", "max", "sum", "mean", "sort", "group"} and self.value_column is None:
            raise UnsafeTablePlan("value_column_required")
        if self.operation in {"compare", "difference", "percentage"} and len(self.operand_cell_refs) != 2:
            raise UnsafeTablePlan("two_operands_required")
        if self.aggregate is not None and self.aggregate not in {"count", "sum", "mean", "min", "max"}:
            raise UnsafeTablePlan("invalid_aggregate")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TablePlan":
        """Validate a constrained planner payload; unknown keys fail closed."""
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        if set(payload) - allowed:
            raise UnsafeTablePlan("unknown_plan_fields")
        values = dict(payload)
        values["table_ids"] = tuple(values.get("table_ids") or ())
        values["select_columns"] = tuple(values.get("select_columns") or ())
        values["operand_cell_refs"] = tuple(values.get("operand_cell_refs") or ())
        values["filters"] = tuple(
            item if isinstance(item, TableFilter) else TableFilter(**item)
            for item in values.get("filters") or ()
        )
        return cls(**values)

    def trace_payload(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "table_ids": list(self.table_ids),
            "value_column": self.value_column,
            "select_columns": list(self.select_columns),
            "filters": [item.__dict__ for item in self.filters],
            "group_column": self.group_column,
            "aggregate": self.aggregate,
            "sort_direction": self.sort_direction,
            "operand_cell_refs": list(self.operand_cell_refs),
            "target_unit": self.target_unit,
            "search_text": self.search_text,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class PlanDecision:
    status: str
    plan: TablePlan | None = None
    reason: str | None = None
    candidate_table_ids: tuple[str, ...] = field(default_factory=tuple)


def plan_table_query(conn, prompt: str) -> PlanDecision:
    """Return a conservative deterministic plan or an explicit fallback."""
    deadline = time.perf_counter() + PLANNING_TIMEOUT_MS / 1000
    if not TABLE_CUES.search(prompt):
        return PlanDecision("fallback", reason="not_a_table_question")
    operation = _operation_for_prompt(prompt)
    targets = QUOTED_TARGET.findall(prompt)
    documents = storage.fetchall(
        conn,
        """
        SELECT DISTINCT documents.id, documents.display_name, documents.path
        FROM documents JOIN tables ON tables.doc_id = documents.id
        WHERE documents.type = 'file'
        ORDER BY documents.id
        """,
    )
    if not documents:
        return PlanDecision("fallback", reason="no_typed_tables")
    selected_document = _resolve_document(documents, targets[-1] if targets else None)
    if selected_document is None:
        return PlanDecision(
            "fallback",
            reason="cross_document_ambiguity" if len(documents) > 1 else "document_not_found",
        )
    tables = storage.fetchall(
        conn,
        """
        SELECT tables.*, GROUP_CONCAT(table_columns.normalized_header, ' ') AS headers
        FROM tables LEFT JOIN table_columns ON table_columns.table_id = tables.id
        WHERE tables.doc_id = ? GROUP BY tables.id ORDER BY tables.page_number, tables.table_index, tables.id
        """,
        (selected_document["id"],),
    )
    if not tables:
        return PlanDecision("fallback", reason="document_has_no_tables")
    if len(tables) > MAX_CANDIDATE_TABLES:
        return PlanDecision("fallback", reason="candidate_table_limit")
    unit = _target_unit(prompt)
    prompt_terms = _terms(prompt) - (_terms(targets[-1]) if targets else set())
    scored: list[tuple[int, Any, int | None, int]] = []
    for table in tables:
        if time.perf_counter() > deadline:
            return PlanDecision("fallback", reason="planning_timeout")
        columns = storage.fetchall(
            conn,
            "SELECT column_index, normalized_header FROM table_columns WHERE table_id = ? ORDER BY column_index",
            (table["id"],),
        )
        best_column, header_score = _best_column(columns, prompt_terms)
        unit_score = _unit_match_count(conn, table["id"], unit) if unit else 0
        if unit_score < 0:
            return PlanDecision("fallback", reason="planning_cell_limit")
        score = header_score * 10 + min(unit_score, 9)
        scored.append((score, table, best_column, unit_score))
    scored.sort(key=lambda item: (-item[0], item[1]["page_number"] or 0, item[1]["table_index"], item[1]["id"]))
    matching = [item for item in scored if item[0] > 0]
    if operation == "lookup" and unit:
        table_ids = tuple(item[1]["id"] for item in matching if item[3] > 0)
        if not table_ids:
            return PlanDecision("fallback", reason="unit_not_found")
        if len(table_ids) > MAX_PLAN_TABLES:
            return PlanDecision("fallback", reason="table_limit", candidate_table_ids=table_ids)
        return PlanDecision(
            "planned",
            TablePlan(operation="lookup", table_ids=table_ids, target_unit=unit, limit=MAX_RESULT_ROWS),
            candidate_table_ids=table_ids,
        )
    if operation in {"compare", "difference", "percentage", "group"}:
        return PlanDecision("fallback", reason="operation_requires_explicit_operands")
    if operation == "count" and not matching:
        if len(tables) != 1:
            return PlanDecision("fallback", reason="ambiguous_table")
        table = tables[0]
        return PlanDecision(
            "planned", TablePlan(operation="count", table_ids=(table["id"],)),
            candidate_table_ids=(table["id"],),
        )
    if not matching:
        return PlanDecision("fallback", reason="column_not_found")
    if len(matching) > 1 and matching[0][0] == matching[1][0]:
        return PlanDecision(
            "fallback", reason="ambiguous_table", candidate_table_ids=tuple(item[1]["id"] for item in matching[:MAX_PLAN_TABLES])
        )
    _, table, column, _ = matching[0]
    if operation not in {"lookup", "count"} and column is None:
        return PlanDecision("fallback", reason="column_not_found")
    filters: tuple[TableFilter, ...] = ()
    if operation == "filter":
        parsed_filter = _filter_for_prompt(prompt, column, unit)
        if parsed_filter is None:
            return PlanDecision("fallback", reason="ambiguous_filter")
        filters = (parsed_filter,)
    plan = TablePlan(
        operation=operation,
        table_ids=(table["id"],),
        value_column=column,
        filters=filters,
        target_unit=unit,
        sort_direction="desc" if operation in {"max"} else "asc",
    )
    return PlanDecision("planned", plan, candidate_table_ids=(table["id"],))


def resolve_named_document(conn, prompt: str):
    """Resolve one explicitly quoted file document without fuzzy guessing."""
    targets = QUOTED_TARGET.findall(prompt)
    if not targets:
        return None
    documents = storage.fetchall(
        conn,
        """
        SELECT id, display_name, path
        FROM documents
        WHERE type = 'file'
        ORDER BY id
        """,
    )
    return _resolve_document(documents, targets[-1])


def _operation_for_prompt(prompt: str) -> str:
    lowered = prompt.casefold()
    patterns = (
        ("percentage", ("percent change", "percentage change", "percentage difference")),
        ("difference", ("difference between", "subtract")),
        ("mean", ("average", "mean")),
        ("sum", ("sum", "total of")),
        ("count", ("how many", "count")),
        ("min", ("minimum", "lowest", "smallest")),
        ("max", ("maximum", "highest", "largest", "heaviest", "most")),
        ("sort", ("sort", "order by")),
        ("group", ("group by", "grouped by")),
        ("compare", ("compare", "versus", " vs ")),
        ("filter", ("where ", "greater than", "less than", "equal to")),
    )
    for operation, cues in patterns:
        if any(cue in lowered for cue in cues):
            return operation
    return "lookup"


def _filter_for_prompt(prompt: str, column: int | None, unit: str | None) -> TableFilter | None:
    if column is None:
        return None
    lowered = prompt.casefold()
    operators = (
        ("gte", ("at least", "greater than or equal")),
        ("lte", ("at most", "less than or equal")),
        ("gt", ("greater than", "more than", "above")),
        ("lt", ("less than", "below")),
        ("eq", ("equal to", "equals")),
    )
    operator = next((name for name, cues in operators if any(cue in lowered for cue in cues)), None)
    match = re.search(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", prompt)
    if operator is None or match is None:
        return None
    return TableFilter(column, operator, match.group(0).replace(",", ""), unit)


def _resolve_document(documents: list[Any], target: str | None):
    if not target:
        return documents[0] if len(documents) == 1 else None
    target_terms = _terms(target)
    scored = []
    for document in documents:
        identity = f"{document['display_name'] or ''} {document['path'] or ''}"
        identity_terms = _terms(identity)
        overlap = target_terms & identity_terms
        score = sum(1 if term in GENERIC_IDENTITY_TERMS else 10 for term in overlap)
        scored.append((score, document))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    if not scored or scored[0][0] == 0:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _best_column(columns: list[Any], prompt_terms: set[str]) -> tuple[int | None, int]:
    scores = [(len(_terms(column["normalized_header"]) & prompt_terms), column["column_index"]) for column in columns]
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0:
        return None, 0
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return None, -1
    return scores[0][1], scores[0][0]


def _target_unit(prompt: str) -> str | None:
    match = re.search(r"\bexpressed\s+in\s+([A-Za-zµμ%]+|percent)\b", prompt)
    if not match:
        return "%" if re.search(r"\bpercent(?:age)?\b", prompt, re.IGNORECASE) else None
    value = match.group(1)
    return "%" if value.casefold() == "percent" else value


def _unit_match_count(conn, table_id: str, unit: str | None) -> int:
    if not unit:
        return 0
    rows = storage.fetchall(
        conn,
        "SELECT raw_value, unit FROM table_cells WHERE table_id = ? LIMIT ?",
        (table_id, MAX_PLANNING_CELLS_PER_TABLE + 1),
    )
    if len(rows) > MAX_PLANNING_CELLS_PER_TABLE:
        return -1
    return sum(_cell_has_unit(row["raw_value"], row["unit"], unit) for row in rows)


def _cell_has_unit(raw: str, stored: str | None, target: str) -> bool:
    if stored == target or (len(target) > 1 and stored and stored.casefold() == target.casefold()):
        return True
    number = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    uncertainty = rf"(?:\s*(?:±|\+/-)\s*{number})?"
    if target == "%":
        return re.search(rf"{number}{uncertainty}\s*%", raw) is not None
    flags = 0 if len(target) == 1 else re.IGNORECASE
    return re.search(rf"{number}{uncertainty}\s*{re.escape(target)}(?![A-Za-zµμ])", raw, flags) is not None


def _terms(value: str) -> set[str]:
    return {term for term in WORD.findall(value.casefold()) if len(term) > 1 and term not in STOPWORDS}
