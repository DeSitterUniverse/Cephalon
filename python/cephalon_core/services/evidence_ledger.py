"""Create a bounded request-scoped ledger from selected retrieval evidence.

The ledger is the control-plane contract shared by context selection, gap
retrieval, and answer verification.  This first implementation is deliberately
observability-only: it reads deterministic subqueries and final sources but
does not change ranking, context, prompts, or generation.

At most eight requirements and twenty sources are examined. Requirement/source
assignment is O(r*s), and potential-conflict comparison is hard-capped at 64
pairs. Evidence excerpts are capped at 500 characters so a normal trace stays
well below the 32 KiB operational target.
"""

from __future__ import annotations

import re
from typing import Any
from unicodedata import normalize

from ..schemas import SourceChunk


MAX_REQUIREMENTS = 8
MAX_LEDGER_SOURCES = 20
MAX_EVIDENCE_PER_REQUIREMENT = 4
MAX_CONFLICT_COMPARISONS = 64
MAX_EVIDENCE_CHARS = 500
SUFFICIENT_TERM_COVERAGE = 0.34
PARTIAL_TERM_COVERAGE = 0.12
MIN_QUALIFYING_WORDS = 18

# Named-paper requirements carry a small evidence-need vocabulary rather than
# relying on literal query words.  A paper may say it "introduces" a method or
# "demonstrates" a result without repeating the exact words "contribution" or
# "result".  These bounded patterns keep sufficiency deterministic while
# preventing an identity-matching heading or bibliography fragment from
# becoming evidence for an unrelated need.
EVIDENCE_NEED_PATTERNS = {
    "contribution": re.compile(
        r"\b(?:contribut\w*|introduc\w*|propos\w*|present\w*|develop\w*|"
        r"design\w*|pioneer\w*|novel\w*|framework|approach)\b",
        re.IGNORECASE,
    ),
    "result": re.compile(
        r"\b(?:result\w*|achiev\w*|improv\w*|outperform\w*|demonstrat\w*|"
        r"show\w*|find\w*|measur\w*|evaluat\w*|performance|accuracy|"
        r"recall|precision|report\w*|observ\w*|yield\w*|reduc\w*)\b",
        re.IGNORECASE,
    ),
    "method": re.compile(
        r"\b(?:method\w*|approach\w*|pipeline\w*|algorithm\w*|architectur\w*|"
        r"procedure\w*|technique\w*|process\w*|us(?:e|es|ed|ing)|utiliz\w*)\b",
        re.IGNORECASE,
    ),
    "limitation": re.compile(
        r"\b(?:limit\w*|challeng\w*|fail\w*|cannot|without|drawback\w*|"
        r"caveat\w*|weakness\w*|shortcoming\w*|however)\b",
        re.IGNORECASE,
    ),
}

LEDGER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when", "which", "who",
    "with", "versus", "vs", "compare", "explain", "describe", "paper", "study",
}
NUMBER_WITH_UNIT_RE = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(%|[a-zA-Zµμ°]+)?")
NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|without|failed to|did not|does not)\b", re.IGNORECASE)
QUOTED_TARGET_RE = re.compile(
    r"“([^”\n]{2,240})”|‘([^’\n]{2,240})’|\"([^\"\n]{2,240})\"|'([^'\n]{2,240})'",
    re.UNICODE,
)
EVIDENCE_MARKER_RE = re.compile(
    r"\b(?:introduc\w*|propos\w*|present\w*|develop\w*|contribut\w*|"
    r"achiev\w*|improv\w*|outperform\w*|demonstrat\w*|show\w*|find\w*|"
    r"result\w*|method\w*|approach\w*|experiment\w*|evaluat\w*|"
    r"measur\w*|limit\w*|challeng\w*|fail\w*|cannot|however)\b",
    re.IGNORECASE,
)
REFERENCE_ONLY_RE = re.compile(
    r"^\s*(?:references?|bibliography|doi\s*[:=]|https?://|www\.|arxiv\s*[:=])\b",
    re.IGNORECASE,
)
METADATA_RE = re.compile(
    r"\b(?:doi|email|university|institute|department|received|accepted|copyright|"
    r"vol(?:ume)?|issue|pages?|issn)\b",
    re.IGNORECASE,
)


def build_evidence_ledger(
    query_id: str,
    raw_query: str,
    subqueries: list[dict[str, str]],
    sources: list[SourceChunk],
    *,
    retrieval_round: int = 0,
) -> dict[str, Any]:
    """Build and return a compact JSON-compatible evidence ledger.

    A direct ``subquery_id`` match is treated as an intentional assignment;
    otherwise material-term coverage supplies a deterministic fallback.
    Conflict records are conservative diagnostics, not factual judgments: they
    require high lexical similarity plus either opposite negation or differing
    numeric values with the same explicit unit.
    """

    requirements = plan_requirements(raw_query, subqueries)
    evidence: list[dict[str, Any]] = []
    assignments: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in requirements}
    for index, source in enumerate(sources[:MAX_LEDGER_SOURCES], start=1):
        evidence_id = f"E{index}"
        # ``None`` means no compression pass populated the field; an explicit
        # empty string means the source was selected but no sentence survived
        # into the model-visible context.  Do not resurrect the original
        # snippet in the latter state.
        text = (source.evidence_text if source.evidence_text is not None else source.snippet or "").strip()
        assigned_ids: list[str] = []
        for requirement in requirements:
            coverage = _coverage(requirement, text)
            direct = requirement["subquery_id"] in _source_subquery_ids(source)
            document_match = (
                document_identity_matches(requirement, source)
                if requirement.get("named_document")
                else True
            )
            qualifying = is_qualifying_evidence(text)
            eligible = (
                document_match and (qualifying or coverage >= PARTIAL_TERM_COVERAGE)
                if requirement.get("named_document")
                else direct or coverage >= PARTIAL_TERM_COVERAGE
            )
            if eligible:
                # Keep the measured need coverage.  A substantive paragraph
                # from the right document is still only partial when it
                # describes a different aspect of that paper; the status gate
                # below is intentionally not allowed to promote it.
                score = max(coverage, 0.5 if direct and not requirement.get("named_document") else 0.0)
                assignments[requirement["id"]].append({
                    "evidence_id": evidence_id,
                    "score": round(score, 6),
                    "qualifying": qualifying,
                    "document_match": document_match,
                })
                assigned_ids.append(requirement["id"])
        source.evidence_ids = [evidence_id]
        source.requirement_ids = assigned_ids
        evidence.append({
            "id": evidence_id,
            "source_id": source.source_id,
            "chunk_id": source.chunk_id,
            "doc_id": source.doc_id,
            "source_kind": source.source_kind or "text",
            "document_name": source.doc_name,
            "requirement_ids": assigned_ids,
            "retrieval_round": source.retrieval_round,
            "span": text[:MAX_EVIDENCE_CHARS],
            "page_number": source.page_number,
            "parent_id": source.parent_id,
            "status": "supporting" if assigned_ids else "unassigned",
        })

    conflicts = _potential_conflicts(evidence)
    conflicting_requirements = {
        requirement_id
        for conflict in conflicts
        for requirement_id in conflict["requirement_ids"]
    }
    requirement_records: list[dict[str, Any]] = []
    for requirement in requirements:
        ranked = sorted(assignments[requirement["id"]], key=lambda item: item["score"], reverse=True)[:MAX_EVIDENCE_PER_REQUIREMENT]
        best = ranked[0]["score"] if ranked else 0.0
        qualifying = [item for item in ranked if item.get("qualifying")]
        if requirement["id"] in conflicting_requirements:
            status = "conflicting"
        elif requirement.get("named_document") and qualifying and max(item["score"] for item in qualifying) >= SUFFICIENT_TERM_COVERAGE:
            status = "sufficient"
        elif best >= SUFFICIENT_TERM_COVERAGE and not requirement.get("named_document"):
            status = "sufficient"
        elif ranked:
            status = "partial"
        else:
            status = "missing"
        requirement_records.append({
            **requirement,
            "status": status,
            "evidence_ids": [item["evidence_id"] for item in ranked],
            "qualifying_evidence_ids": [item["evidence_id"] for item in qualifying],
            "best_coverage": round(best, 6),
            "matched_document_ids": sorted({
                evidence_item["doc_id"]
                for evidence_item in evidence
                if evidence_item["id"] in {item["evidence_id"] for item in ranked}
                and evidence_item.get("doc_id")
            }),
        })

    counts = {
        status: sum(item["status"] == status for item in requirement_records)
        for status in ("sufficient", "partial", "missing", "conflicting")
    }
    return {
        "ledger_id": query_id,
        "query": raw_query,
        "state": "assessed",
        "retrieval_round": retrieval_round,
        "requirements": requirement_records,
        "evidence": evidence,
        "conflicts": conflicts,
        "summary": {**counts, "requirement_count": len(requirement_records), "evidence_count": len(evidence)},
        "limits": {
            "max_requirements": MAX_REQUIREMENTS,
            "max_sources": MAX_LEDGER_SOURCES,
            "max_evidence_per_requirement": MAX_EVIDENCE_PER_REQUIREMENT,
            "max_conflict_comparisons": MAX_CONFLICT_COMPARISONS,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
        },
    }


def plan_requirements(raw_query: str, subqueries: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return bounded requirements, giving each explicitly named paper its own slot.

    Quoted titles are treated as document identities, not merely as lexical
    terms.  This prevents one generic paragraph from satisfying several named
    studies and gives the gap controller an exact title to retrieve when a
    selected context omits one of them.
    """

    named_targets = extract_named_document_targets(raw_query)
    if named_targets:
        evidence_need = requested_evidence_needs(raw_query)
        return [
            {
                "id": f"R{index}",
                "subquery_id": "",
                "text": f"{title} {' '.join(evidence_need)}",
                "named_document": True,
                "requested_title": title,
                "requested_aliases": document_aliases(title),
                "evidence_need": evidence_need,
            }
            for index, title in enumerate(named_targets[:MAX_REQUIREMENTS], start=1)
        ]

    components = [item for item in subqueries if item.get("id") != "q0"]
    if not components:
        components = subqueries[:1] or [{"id": "q1", "text": raw_query}]
    return [
        {
            "id": f"R{index}",
            "subquery_id": str(item.get("id") or f"q{index}"),
            "text": str(item.get("text") or raw_query),
            "named_document": False,
        }
        for index, item in enumerate(components[:MAX_REQUIREMENTS], start=1)
    ]


def extract_named_document_targets(raw_query: str) -> list[str]:
    """Extract explicit quoted paper/study targets in first-use order."""

    if not re.search(
        r"\b(?:target|paper|study|studies|source|synthesi[sz]|contribut|"
        r"result|finding|method|approach|performance|limitation|compare)\w*\b",
        raw_query,
        re.IGNORECASE,
    ):
        return []
    targets: list[str] = []
    for match in QUOTED_TARGET_RE.finditer(raw_query or ""):
        value = next((group for group in match.groups() if group), "").strip(" \t\r\n.,;:!?()[]")
        if len(value) < 2 or value.casefold() in {"paper", "study", "source"}:
            continue
        if value.casefold() not in {item.casefold() for item in targets}:
            targets.append(value)
    # Some clients omit quotation marks around a target title. Only accept the
    # tightly delimited ``use ... as the target`` form so ordinary prose is
    # not accidentally promoted into a document requirement.
    if not targets:
        unquoted = re.search(
            r"\buse\s+(?:the\s+)?(.{2,240}?)\s+as\s+(?:the\s+)?target(?:\s+(?:paper|study|source|document))?\b",
            raw_query or "",
            re.IGNORECASE,
        )
        if unquoted:
            value = unquoted.group(1).strip(" \t\r\n.,;:!?()[]")
            if value and value.casefold() not in {"paper", "study", "source", "document"}:
                targets.append(value)
    return targets


def requested_evidence_needs(raw_query: str) -> list[str]:
    """Map the request to a small, auditable evidence-need vocabulary."""

    lowered = raw_query.casefold()
    needs: list[str] = []
    if re.search(r"contribut|central finding|introduc|propos", lowered):
        needs.append("contribution")
    if re.search(r"result|achiev|performance|finding|improv|outperform", lowered):
        needs.append("result")
    if re.search(r"method|approach|how|pipeline|technique", lowered):
        needs.append("method")
    if re.search(r"limit|challenge|failure|drawback|caveat", lowered):
        needs.append("limitation")
    return needs or ["contribution", "result", "method", "limitation"]


def _source_subquery_ids(source: SourceChunk) -> set[str]:
    return {value.strip() for value in (source.subquery_id or "").split(",") if value.strip()}


def document_aliases(title: str) -> list[str]:
    """Return conservative aliases for a requested document title."""

    clean = " ".join(str(title or "").split()).strip()
    aliases = [clean]
    prefix = clean.split(":", 1)[0].strip(" -")
    if prefix and prefix.casefold() != clean.casefold():
        aliases.append(prefix)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def document_identity_matches(requirement: dict[str, Any], candidate: SourceChunk | dict[str, Any]) -> bool:
    """Match a named requirement only against document identity metadata.

    Chunk text is deliberately excluded. A citation or bibliography fragment
    can mention another paper, but it cannot turn the containing document into
    that requested paper.
    """

    if not requirement.get("named_document"):
        return True
    aliases = requirement.get("requested_aliases") or document_aliases(str(requirement.get("requested_title") or ""))
    identities = _candidate_identity_strings(candidate)
    normalized_identities = [_normalized_identifier(value) for value in identities]
    identity_tokens = set().union(*(_identifier_tokens(value) for value in identities)) if identities else set()
    for alias in aliases:
        normalized_alias = _normalized_identifier(alias)
        if normalized_alias and any(normalized_alias in value for value in normalized_identities):
            return True
        alias_tokens = _identifier_tokens(alias)
        shared = alias_tokens & identity_tokens
        if len(alias_tokens) == 1:
            token = next(iter(alias_tokens), "")
            if len(token) >= 4 and token in identity_tokens:
                return True
        elif len(shared) >= 2:
            # Two distinctive title tokens are enough for truncated corpus
            # filenames such as ``astronomy-02-oq-208-...``.
            return True
    return False


def _candidate_identity_strings(candidate: SourceChunk | dict[str, Any]) -> list[str]:
    if isinstance(candidate, dict):
        values: list[str] = [
            str(candidate.get("doc_name") or ""),
            str(candidate.get("document_name") or ""),
            str(candidate.get("document_path") or ""),
            str(candidate.get("doc_id") or ""),
        ]
        provenance = candidate.get("provenance")
    else:
        values = [str(candidate.doc_name or ""), str(candidate.doc_id or "")]
        provenance = candidate.provenance
    if isinstance(provenance, dict):
        for key in ("path", "filename", "display_name", "document_name", "document_path"):
            value = provenance.get(key)
            if isinstance(value, str):
                values.append(value)
    return [value for value in values if value]


def _normalized_identifier(value: str) -> str:
    clean = normalize("NFKC", str(value or "")).casefold()
    clean = re.sub(r"\.(?:pdf|txt|md|csv|json)$", "", clean)
    return " ".join(re.findall(r"[\wµμ]+", clean, flags=re.UNICODE))


def _identifier_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[\wµμ]+", normalize("NFKC", str(value or "")).casefold(), flags=re.UNICODE)
        if len(token) >= 2 and token not in LEDGER_STOPWORDS and token not in {"pdf", "txt", "md"}
    }


def is_qualifying_evidence(text: str) -> bool:
    """Reject short metadata/reference fragments as named-paper evidence.

    This is intentionally conservative. It is a sufficiency gate, not a
    document-ingestion validator: ordinary requirements retain their previous
    lexical behavior, while named-paper coverage needs a substantive paragraph
    with a contribution, result, method, or limitation signal.
    """

    clean = " ".join(str(text or "").split())
    words = re.findall(r"[\wµμ%.-]+", clean, flags=re.UNICODE)
    if len(words) < MIN_QUALIFYING_WORDS:
        return False
    if REFERENCE_ONLY_RE.search(clean):
        return False
    if not re.search(r"[A-Za-z]{3}", clean):
        return False
    # Long reference lists can contain an accidental verb such as
    # "introduces".  Multiple citation-shaped tokens without a sentence-like
    # subject are still bibliography, not substantive paper evidence.
    reference_signals = len(re.findall(r"\b(?:19|20)\d{2}\b|\bet\s+al\.?\b|\bdoi\b|\bpp?\.\b", clean, re.IGNORECASE))
    subject_signal = re.search(r"\b(?:we|our|this|the\s+(?:paper|study|work|authors?))\b", clean, re.IGNORECASE)
    if reference_signals >= 3 and not subject_signal:
        return False
    if METADATA_RE.search(clean) and not EVIDENCE_MARKER_RE.search(clean):
        return False
    return bool(EVIDENCE_MARKER_RE.search(clean))


def evidence_need_coverage(requirement: dict[str, Any], evidence: str) -> float:
    """Return the fraction of named evidence needs supported by a span.

    Document identity is checked by the caller.  This function only answers
    whether the span contains the requested contribution/result/method/
    limitation signal, with bounded synonym patterns.  A qualifying paragraph
    that describes a different aspect of the paper therefore remains partial
    and can still trigger the single bounded gap query.
    """

    needs = [str(value).casefold() for value in requirement.get("evidence_need", [])]
    if not needs:
        return 0.0
    matched = sum(
        1
        for need in dict.fromkeys(needs)
        if (pattern := EVIDENCE_NEED_PATTERNS.get(need)) is not None and pattern.search(evidence or "")
    )
    return matched / len(dict.fromkeys(needs))


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wµμ]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3 and token not in LEDGER_STOPWORDS
    }


def _coverage(requirement: dict[str, Any] | str, evidence: str) -> float:
    if isinstance(requirement, dict) and requirement.get("named_document"):
        return evidence_need_coverage(requirement, evidence)
    else:
        required = _terms(str(requirement))
    if not required:
        return 0.0
    return len(required & _terms(evidence)) / len(required)


def _potential_conflicts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    comparisons = 0
    for left_index, left in enumerate(evidence):
        for right in evidence[left_index + 1:]:
            if comparisons >= MAX_CONFLICT_COMPARISONS:
                return conflicts
            comparisons += 1
            shared_requirements = sorted(set(left["requirement_ids"]) & set(right["requirement_ids"]))
            if not shared_requirements or left["doc_id"] == right["doc_id"]:
                continue
            left_terms, right_terms = _terms(left["span"]), _terms(right["span"])
            union = left_terms | right_terms
            similarity = len(left_terms & right_terms) / len(union) if union else 0.0
            if similarity < 0.45:
                continue
            reason = _conflict_reason(left["span"], right["span"])
            if reason:
                conflicts.append({
                    "id": f"C{len(conflicts) + 1}",
                    "evidence_ids": [left["id"], right["id"]],
                    "requirement_ids": shared_requirements,
                    "status": "potential",
                    "reason": reason,
                })
    return conflicts


def _conflict_reason(left: str, right: str) -> str | None:
    if bool(NEGATION_RE.search(left)) != bool(NEGATION_RE.search(right)):
        return "highly similar evidence has opposite negation polarity"
    left_values = {(unit.lower(), float(value)) for value, unit in NUMBER_WITH_UNIT_RE.findall(left) if unit}
    right_values = {(unit.lower(), float(value)) for value, unit in NUMBER_WITH_UNIT_RE.findall(right) if unit}
    for unit, left_value in left_values:
        same_unit = [value for candidate_unit, value in right_values if candidate_unit == unit]
        if same_unit and all(abs(left_value - value) > max(1e-9, abs(left_value) * 0.001) for value in same_unit):
            return f"highly similar evidence reports different values with unit {unit}"
    return None
