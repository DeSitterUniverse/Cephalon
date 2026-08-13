"""Verify cited answer claims textually and numerically.

This deterministic verifier is the non-negotiable output boundary beneath the
optional Gemma semantic audit. It assigns one of five entailment states:
``entailed``, ``partially_entailed``, ``unsupported``, ``contradicted``, or
``citation_missing``. Existing support-status aliases are emitted alongside the
new states so stored conversations and older clients remain valid.

Numeric checks preserve values and units, accept direct evidence, and recompute
simple differences, totals, means, and relative percentages. Tolerance is the
larger of 1 percent relative error and 1e-6 absolute error. The verifier never
executes generated code or model-produced expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from ..schemas import SourceChunk


ENTAILMENT_STATES = {
    "entailed", "partially_entailed", "unsupported", "contradicted", "citation_missing",
}
MAX_VERIFIED_CLAIMS = 32
ENTAILED_TERM_COVERAGE = 0.55
PARTIAL_TERM_COVERAGE = 0.25
CONTRADICTION_TERM_SIMILARITY = 0.45
NUMERIC_RELATIVE_TOLERANCE = 0.01
NUMERIC_ABSOLUTE_TOLERANCE = 1e-6

SOURCE_TAG_PATTERN = re.compile(r"\[\[\s*src\s*:\s*([A-Za-z0-9_-]+)\s*\]\]", re.IGNORECASE)
CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?!\[\[\s*src\s*:)|\n+", re.IGNORECASE)
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)
HIDDEN_THINK_BLOCK_PATTERN = re.compile(
    r"<think\b[^>]*>.*?(?:</think\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)
NUMBER_RE = re.compile(r"(?<![\w.-])(-?\d+(?:\.\d+)?)\s*(%|percent|[a-zA-Zµμ°]+)?", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|without|failed to|did not|does not|was not|were not)\b", re.IGNORECASE)
CLAIM_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between", "could", "does",
    "from", "have", "into", "more", "most", "other", "should", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "using", "very", "were", "what", "when", "where",
    "which", "while", "with", "would", "your", "did", "not", "never", "without", "failed",
}
UNIT_ALIASES = {
    "percent": "%",
    "percentage": "%",
    "millisecond": "ms",
    "milliseconds": "ms",
    "msec": "ms",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "minute": "min",
    "minutes": "min",
    "joule": "j",
    "joules": "j",
    "watt": "w",
    "watts": "w",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
}


@dataclass(frozen=True)
class NumericValue:
    """A parsed scalar and normalized unit from claim or evidence text."""

    value: float
    unit: str
    raw: str


def strip_hidden_reasoning(text: str) -> str:
    """Return only visible answer prose, removing explicit ``<think>`` blocks.

    llama.cpp can expose model reasoning in a separate ``reasoning_content``
    field, but templates that do not parse thoughts may put the same material
    directly in ``message.content``.  An unclosed opening tag is hidden through
    end-of-output; losing an incomplete thought is safer than persisting it as
    an answer.
    """

    clean = str(text or "")
    clean = HIDDEN_THINK_BLOCK_PATTERN.sub("\n", clean)
    return THINK_CLOSE_PATTERN.sub("", clean).strip()


class VisibleAnswerFilter:
    """Incrementally discard explicit thought blocks across SSE deltas.

    Both tags can be split across network chunks.  The small pending buffer
    therefore retains only a possible tag prefix and never buffers a complete
    visible answer.
    """

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False

    def feed(self, text: str) -> str:
        self._buffer += str(text or "")
        emitted: list[str] = []
        while self._buffer:
            if self._in_thinking:
                close = THINK_CLOSE_PATTERN.search(self._buffer)
                if close is None:
                    keep = len(self._CLOSE_TAG) - 1
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[close.end():]
                self._in_thinking = False
                continue

            opening = THINK_OPEN_PATTERN.search(self._buffer)
            if opening is not None:
                emitted.append(self._buffer[:opening.start()])
                self._buffer = self._buffer[opening.end():]
                self._in_thinking = True
                continue

            # Hold a possible partial opening tag until the next delta. A
            # normal answer ending in "I think" does not begin with '<'.
            keep = len(self._OPEN_TAG) - 1
            if len(self._buffer) > keep:
                emitted.append(self._buffer[:-keep] if keep else self._buffer)
                self._buffer = self._buffer[-keep:] if keep else ""
            break
        return "".join(emitted)

    def finish(self) -> str:
        if self._in_thinking:
            self._buffer = ""
            return ""
        pending = self._buffer
        self._buffer = ""
        if pending.casefold() in self._OPEN_TAG[: len(pending)].casefold():
            return ""
        return THINK_CLOSE_PATTERN.sub("", pending)


def verify_answer_claims(answer_text: str, sources: list[SourceChunk]) -> dict[str, Any]:
    """Verify at most 32 answer statements against their cited sources."""

    visible_answer = strip_hidden_reasoning(answer_text)
    source_by_id = {
        source.source_id.upper(): source
        for source in sources
        if source.source_id
    }
    claims: list[dict[str, Any]] = []
    for index, statement in enumerate(_claim_statements(visible_answer)[:MAX_VERIFIED_CLAIMS], start=1):
        clean = statement.strip()
        if not clean or clean.startswith("<think>") or clean.startswith("</think>"):
            continue
        cited_ids = _cited_source_ids(clean)
        claim_text = SOURCE_TAG_PATTERN.sub("", clean).strip(" -*#\t")
        claim_terms = _terms(claim_text)
        if len(claim_terms) < 2:
            continue
        known_sources = [source_by_id[source_id] for source_id in cited_ids if source_id in source_by_id]
        unknown_ids = [source_id for source_id in cited_ids if source_id not in source_by_id]
        evidence_by_source = {
            source.source_id or "unknown": (
                source.evidence_text if source.evidence_text is not None else source.snippet or ""
            )
            for source in known_sources
        }
        coverage_by_source = {
            source_id: _term_coverage(claim_terms, _terms(evidence))
            for source_id, evidence in evidence_by_source.items()
        }
        best_coverage = max(coverage_by_source.values(), default=0.0)
        combined_evidence = "\n".join(evidence_by_source.values())
        numeric = _verify_numbers(claim_text, combined_evidence, known_sources)
        negation_conflict = _negation_conflict(claim_text, evidence_by_source)

        if not cited_ids:
            entailment = "citation_missing"
            reason = "The claim has no source tag."
        elif unknown_ids:
            entailment = "unsupported"
            reason = "One or more citation tags do not identify supplied evidence."
        elif numeric["status"] in {"contradicted", "unit_mismatch"}:
            entailment = "contradicted"
            reason = numeric["reason"]
        elif negation_conflict:
            entailment = "contradicted"
            reason = "Highly similar cited evidence has opposite negation polarity."
        elif numeric["status"] in {"unsupported", "missing_cell", "ambiguous_operation"}:
            entailment = "unsupported"
            reason = numeric["reason"]
        elif best_coverage >= ENTAILED_TERM_COVERAGE:
            entailment = "entailed"
            reason = "The cited evidence contains most material terms and passes deterministic checks."
        elif best_coverage >= PARTIAL_TERM_COVERAGE:
            entailment = "partially_entailed"
            reason = "The cited evidence provides partial textual support."
        else:
            entailment = "unsupported"
            reason = "The cited evidence does not contain enough material support."

        claims.append({
            "claim_id": f"C{index}",
            "text": claim_text,
            "source_ids": cited_ids,
            "status": _legacy_status(entailment),
            "entailment_status": entailment,
            "reason": reason,
            "coverage": round(best_coverage, 6),
            "coverage_by_source": {key: round(value, 6) for key, value in coverage_by_source.items()},
            "numeric_verification": numeric,
            "negation_conflict": negation_conflict,
        })

    summary = _summary(claims)
    summary["hidden_reasoning_removed"] = visible_answer != str(answer_text or "")
    return summary


def _summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        state: sum(claim["entailment_status"] == state for claim in claims)
        for state in sorted(ENTAILMENT_STATES)
    }
    return {
        "method": "deterministic_entailment_and_numeric_v2",
        "claim_count": len(claims),
        "entailed_claim_count": counts["entailed"],
        "partially_entailed_claim_count": counts["partially_entailed"],
        "contradicted_claim_count": counts["contradicted"],
        "citation_missing_claim_count": counts["citation_missing"],
        # Compatibility aliases used by existing support panels and stored
        # answer records. New clients should prefer entailment counts.
        "supported_claim_count": counts["entailed"],
        "weak_claim_count": counts["partially_entailed"],
        "unsupported_claim_count": counts["unsupported"] + counts["contradicted"],
        "uncited_claim_count": counts["citation_missing"],
        "claims": claims,
    }


def _verify_numbers(claim: str, evidence: str, sources: list[SourceChunk] | None = None) -> dict[str, Any]:
    claim_values = _numeric_values(claim)
    if not claim_values:
        return _verify_structured_direction(claim, sources or [])
    structured = _verify_structured_numbers(claim, claim_values, sources or [])
    if structured is not None:
        return structured
    evidence_values = _numeric_values(evidence)
    checks: list[dict[str, Any]] = []
    contradicted = False
    unsupported = False
    for claim_value in claim_values:
        direct = next((value for value in evidence_values if _compatible_units(claim_value.unit, value.unit) and _close(claim_value.value, value.value)), None)
        if direct:
            checks.append({"claim_value": claim_value.raw, "status": "entailed", "method": "direct", "evidence_value": direct.raw})
            continue
        derived = _derived_value(claim, claim_value, evidence_values)
        if derived is not None:
            matches = _close(claim_value.value, derived[0])
            contradicted = contradicted or not matches
            checks.append({
                "claim_value": claim_value.raw,
                "status": "entailed" if matches else "contradicted",
                "method": derived[1],
                "computed_value": round(derived[0], 9),
            })
            continue
        comparable = [value for value in evidence_values if _compatible_units(claim_value.unit, value.unit)]
        if comparable:
            contradicted = True
            checks.append({
                "claim_value": claim_value.raw,
                "status": "contradicted",
                "method": "recomputed_or_direct_mismatch",
                "evidence_values": [value.raw for value in comparable[:8]],
            })
        else:
            unsupported = True
            checks.append({"claim_value": claim_value.raw, "status": "unsupported", "method": "no_comparable_evidence"})
    if _direction_conflict(claim, evidence, evidence_values):
        contradicted = True
        checks.append({"status": "contradicted", "method": "direction_mismatch"})
    if contradicted:
        return {"status": "contradicted", "reason": "A numerical claim does not match cited values or deterministic recomputation.", "checks": checks}
    if unsupported:
        return {"status": "unsupported", "reason": "A numerical claim has no comparable cited value.", "checks": checks}
    return {"status": "entailed", "reason": "All numerical claims match cited values or deterministic recomputation.", "checks": checks}


def _verify_structured_numbers(
    claim: str,
    claim_values: list[NumericValue],
    sources: list[SourceChunk],
) -> dict[str, Any] | None:
    structured = [source for source in sources if source.table_id or source.cells or source.table_operation]
    if not structured:
        return None
    operations = {source.table_operation for source in structured if source.table_operation}
    if len(operations) != 1:
        return _structured_failure("ambiguous_operation", "Cited table sources do not identify one operation.")
    operation = next(iter(operations))
    supported = {
        "lookup", "filter", "sort", "group", "min", "max", "sum", "mean",
        "count", "compare", "difference", "percentage",
    }
    if operation not in supported:
        return _structured_failure("ambiguous_operation", "The cited table operation is unsupported or missing.")

    checks: list[dict[str, Any]] = []
    computed: list[tuple[Decimal, str, str]] = []
    for source in structured:
        cell_by_ref = {cell.cell_ref: cell for cell in source.cells}
        required_refs = set(source.cell_refs) | set(source.verification_cell_refs)
        missing = sorted(required_refs - set(cell_by_ref))
        if missing:
            return _structured_failure(
                "missing_cell",
                "One or more cited arithmetic cells are unavailable.",
                missing_cell_refs=missing,
            )
        try:
            values = _structured_operation_values(source, operation, cell_by_ref)
        except ValueError as error:
            status = str(error)
            reason = {
                "unit_mismatch": "Cited cells use incompatible units.",
                "missing_cell": "The operation lacks required cited cells.",
                "ambiguous_operation": "The operation cannot be recomputed unambiguously from cited cells.",
            }.get(status, "The cited operation is unsupported.")
            return _structured_failure(status if status in {"unit_mismatch", "missing_cell", "ambiguous_operation"} else "ambiguous_operation", reason)
        computed.extend(values)

    contradicted = False
    unit_mismatch = False
    for claim_value in claim_values:
        compatible = [item for item in computed if _compatible_units(claim_value.unit, item[1])]
        exact = next((item for item in compatible if _close(claim_value.value, float(item[0]))), None)
        if exact:
            checks.append({
                "claim_value": claim_value.raw,
                "status": "entailed",
                "method": exact[2],
                "computed_value": str(exact[0]),
                "unit": exact[1] or None,
            })
            continue
        same_number = next((item for item in computed if _close(claim_value.value, float(item[0]))), None)
        if same_number and not _compatible_units(claim_value.unit, same_number[1]):
            unit_mismatch = True
            checks.append({
                "claim_value": claim_value.raw,
                "status": "unit_mismatch",
                "method": same_number[2],
                "computed_value": str(same_number[0]),
                "computed_unit": same_number[1] or None,
            })
        else:
            contradicted = True
            checks.append({
                "claim_value": claim_value.raw,
                "status": "contradicted",
                "method": "cited_cell_recomputation",
                "computed_values": [f"{value} {unit}".strip() for value, unit, _ in computed[:24]],
            })
    if operation == "compare":
        stated = next((
            word for word in ("greater", "less", "equal")
            if re.search(rf"\b{word}\b", claim, re.IGNORECASE)
        ), None)
        declared = [str(row.get("value") or "").casefold() for source in structured for row in source.table_result]
        if stated and stated not in declared:
            contradicted = True
            checks.append({"status": "contradicted", "method": "cell_compare", "computed_values": declared})
    if unit_mismatch:
        return {
            "status": "unit_mismatch",
            "reason": "A numerical claim has the right scalar but an incompatible unit.",
            "operation": operation,
            "checks": checks,
        }
    if contradicted:
        return {
            "status": "contradicted",
            "reason": "A numerical claim does not match deterministic cited-cell recomputation.",
            "operation": operation,
            "checks": checks,
        }
    return {
        "status": "entailed",
        "reason": "All numerical claims match deterministic cited-cell recomputation.",
        "operation": operation,
        "checks": checks,
    }


def _structured_operation_values(
    source: SourceChunk,
    operation: str,
    cell_by_ref: dict[str, Any],
) -> list[tuple[Decimal, str, str]]:
    results = source.table_result or []
    if operation in {"lookup", "filter", "sort"}:
        return _cell_values((cell_by_ref[ref] for ref in source.cell_refs), f"cell_{operation}")
    if operation == "group":
        output = _declared_result_values(results, f"table_{operation}")
        if output:
            return output
        return _cell_values(cell_by_ref.values(), f"cell_{operation}")
    if operation == "count":
        refs = source.verification_cell_refs
        if not refs and not results:
            raise ValueError("missing_cell")
        row_count = len({cell_by_ref[ref].row_index for ref in refs}) if refs else int(results[0]["value"])
        return [(Decimal(row_count), "", "cell_count")]

    ordered_refs = []
    if results:
        ordered_refs = list(results[0].get("verification_cell_refs") or results[0].get("cell_refs") or [])
    if not ordered_refs:
        ordered_refs = list(source.verification_cell_refs or source.cell_refs)
    if not ordered_refs or any(ref not in cell_by_ref for ref in ordered_refs):
        raise ValueError("missing_cell")
    operand_cells = [cell_by_ref[ref] for ref in ordered_refs]
    values = _decimal_cells(operand_cells)
    units = {unit for _, unit in values}
    if len(units) != 1:
        raise ValueError("unit_mismatch")
    unit = next(iter(units))
    numbers = [value for value, _ in values]
    if operation == "sum":
        result = sum(numbers, Decimal(0))
    elif operation == "mean":
        result = sum(numbers, Decimal(0)) / Decimal(len(numbers))
    elif operation == "min":
        result = min(numbers)
    elif operation == "max":
        result = max(numbers)
    elif operation == "difference":
        if len(numbers) != 2:
            raise ValueError("ambiguous_operation")
        result = numbers[0] - numbers[1]
    elif operation == "percentage":
        if len(numbers) != 2 or numbers[1] == 0:
            raise ValueError("ambiguous_operation")
        result = (numbers[0] - numbers[1]) / numbers[1] * Decimal(100)
        unit = "%"
    elif operation == "compare":
        return [(value, unit, "cell_compare_operand") for value, unit in values]
    else:
        raise ValueError("ambiguous_operation")
    return [(result, unit, f"cell_{operation}")]


def _declared_result_values(results: list[dict[str, Any]], method: str) -> list[tuple[Decimal, str, str]]:
    output = []
    for row in results:
        if row.get("value") is not None:
            try:
                output.append((Decimal(str(row["value"])), _normalize_unit(str(row.get("unit") or "")), method))
            except InvalidOperation:
                pass
        for raw in row.get("values") or []:
            for value in _numeric_values(str(raw)):
                output.append((Decimal(str(value.value)), value.unit, method))
    return output


def _cell_values(cells, method: str) -> list[tuple[Decimal, str, str]]:
    return [(value, unit, method) for value, unit in _decimal_cells(cells)]


def _decimal_cells(cells) -> list[tuple[Decimal, str]]:
    output = []
    for cell in cells:
        if cell.normalized_value is None:
            continue
        try:
            output.append((Decimal(cell.normalized_value), _normalize_unit(cell.unit or "")))
        except InvalidOperation:
            continue
    if not output:
        raise ValueError("missing_cell")
    return output


def _verify_structured_direction(claim: str, sources: list[SourceChunk]) -> dict[str, Any]:
    structured = [source for source in sources if source.table_operation == "compare"]
    if not structured:
        return {"status": "not_applicable", "checks": []}
    results = [str(row.get("value") or "").casefold() for source in structured for row in source.table_result]
    stated = next((word for word in ("greater", "less", "equal") if re.search(rf"\b{word}\b", claim, re.IGNORECASE)), None)
    if not stated:
        return _structured_failure("ambiguous_operation", "The comparison claim has no supported direction.")
    if stated not in results:
        return {
            "status": "contradicted",
            "reason": "The claimed comparison direction contradicts cited-cell execution.",
            "operation": "compare",
            "checks": [{"status": "contradicted", "method": "cell_compare", "computed_values": results}],
        }
    return {
        "status": "entailed",
        "reason": "The comparison direction matches deterministic cited-cell execution.",
        "operation": "compare",
        "checks": [{"status": "entailed", "method": "cell_compare", "computed_value": stated}],
    }


def _structured_failure(status: str, reason: str, **extra) -> dict[str, Any]:
    return {"status": status, "reason": reason, "checks": [], **extra}


def _direction_conflict(claim: str, evidence_text: str, evidence_values: list[NumericValue]) -> bool:
    """Check increase/decrease language only when evidence states from/to order."""

    if "from" not in evidence_text.lower() or " to " not in evidence_text.lower():
        return False
    groups = _unit_groups(evidence_values)
    pair = next((values[:2] for values in groups.values() if len(values) >= 2), None)
    if not pair:
        return False
    delta = pair[1].value - pair[0].value
    lowered = claim.lower()
    expects_decrease = bool(re.search(r"\b(?:decreas\w*|reduc\w*|lower\w*|faster)\b", lowered))
    expects_increase = bool(re.search(r"\b(?:increas\w*|higher\w*|grew|slower)\b", lowered))
    if "improv" in lowered and re.search(r"\b(?:latency|time|error|loss)\b", lowered):
        expects_decrease = True
    return (expects_decrease and delta > 0) or (expects_increase and delta < 0)


def _derived_value(claim: str, target: NumericValue, evidence: list[NumericValue]) -> tuple[float, str] | None:
    lowered = claim.lower()
    if target.unit == "%":
        groups = _unit_groups([value for value in evidence if value.unit != "%"])
        for values in groups.values():
            if len(values) < 2:
                continue
            left, right = values[0].value, values[1].value
            if abs(left) > NUMERIC_ABSOLUTE_TOLERANCE:
                candidate = abs(left - right) / abs(left) * 100.0
                return candidate, "relative_percent_change"
            if abs(right) > NUMERIC_ABSOLUTE_TOLERANCE:
                candidate = abs(left - right) / abs(right) * 100.0
                return candidate, "relative_percent_change"
    comparable = [value.value for value in evidence if _compatible_units(target.unit, value.unit)]
    if len(comparable) >= 2 and re.search(r"\b(?:difference|differed|gap)\b", lowered):
        return abs(comparable[0] - comparable[1]), "difference"
    if comparable and re.search(r"\b(?:average|mean)\b", lowered):
        return sum(comparable) / len(comparable), "mean"
    if comparable and re.search(r"\b(?:sum|total)\b", lowered):
        return sum(comparable), "sum"
    return None


def _numeric_values(text: str) -> list[NumericValue]:
    values: list[NumericValue] = []
    for match in NUMBER_RE.finditer(text or ""):
        unit = _normalize_unit(match.group(2) or "")
        values.append(NumericValue(float(match.group(1)), unit, match.group(0).strip()))
    return values[:32]


def _normalize_unit(unit: str) -> str:
    raw = unit.strip().replace("μ", "µ")
    # One-letter scientific units are case-sensitive: M, m, K and k cannot be
    # folded without turning a correct scalar into false supporting evidence.
    if len(raw) == 1 and raw.isalpha():
        return raw
    lowered = raw.lower()
    return UNIT_ALIASES.get(lowered, lowered)


def _compatible_units(left: str, right: str) -> bool:
    return left == right and bool(left or right) or (not left and not right)


def _close(expected: float, actual: float) -> bool:
    tolerance = max(NUMERIC_ABSOLUTE_TOLERANCE, abs(expected) * NUMERIC_RELATIVE_TOLERANCE)
    return abs(expected - actual) <= tolerance


def _unit_groups(values: list[NumericValue]) -> dict[str, list[NumericValue]]:
    groups: dict[str, list[NumericValue]] = {}
    for value in values:
        if value.unit:
            groups.setdefault(value.unit, []).append(value)
    return groups


def _negation_conflict(claim: str, evidence_by_source: dict[str, str]) -> bool:
    claim_terms = _terms(claim)
    claim_negated = bool(NEGATION_RE.search(claim))
    for evidence in evidence_by_source.values():
        evidence_terms = _terms(evidence)
        union = claim_terms | evidence_terms
        similarity = len(claim_terms & evidence_terms) / len(union) if union else 0.0
        if similarity >= CONTRADICTION_TERM_SIMILARITY and claim_negated != bool(NEGATION_RE.search(evidence)):
            return True
    return False


def _claim_statements(answer_text: str) -> list[str]:
    answer_text = strip_hidden_reasoning(answer_text)
    tag = r"\[\[\s*src\s*:\s*[A-Za-z0-9_-]+\s*\]\]"
    normalized = re.sub(
        rf"([.!?])\s+((?:{tag}\s*)+)",
        lambda match: f"{match.group(1)} {match.group(2).strip()}\n",
        answer_text or "",
        flags=re.IGNORECASE,
    )
    return [statement for statement in CLAIM_SPLIT_PATTERN.split(normalized) if statement.strip()]


def _cited_source_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(1).upper() for match in SOURCE_TAG_PATTERN.finditer(text or "")))


def _terms(text: str) -> set[str]:
    return {
        _term_root(term) for term in re.findall(r"[\w.-]+", text.lower(), flags=re.UNICODE)
        if len(term) >= 3 and term not in CLAIM_STOPWORDS
    }


def _term_root(term: str) -> str:
    """Normalize ordinary inflection while retaining numbers and short terms."""

    return term[:6] if len(term) > 6 and term.replace(".", "", 1).isalpha() else term


def _term_coverage(claim_terms: set[str], evidence_terms: set[str]) -> float:
    return len(claim_terms & evidence_terms) / len(claim_terms) if claim_terms else 0.0


def _legacy_status(entailment: str) -> str:
    return {
        "entailed": "supported",
        "partially_entailed": "weak",
        "citation_missing": "uncited",
        "unsupported": "unsupported",
        "contradicted": "unsupported",
    }[entailment]
