from collections.abc import Iterator
from contextlib import nullcontext
import json
import re
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..schemas import Message, RagSettings
from . import models
from . import support
from .claim_verification import VisibleAnswerFilter, strip_hidden_reasoning
from .prompt_budget import budget_prompt


ResponseEffort = Literal["quick", "balanced", "thorough"]
THINKING_TOKEN_ALLOCATION = 2048
FINAL_OUTPUT_TOKEN_LIMITS = {
    "quick": 2048,
    "balanced": 4096,
    "thorough": 4096,
}
MAX_SEMANTIC_AUDIT_CLAIMS = 32
MAX_CONSTRAINED_AUDIT_CLAIMS = 12
MAX_REPAIR_AUDIT_CLAIM_CHARS = 180
MAX_REPAIR_AUDIT_REASON_CHARS = 220
HARD_FAILURE_STATES = {"contradicted", "citation_missing"}
CLAIM_AUDIT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cephalon_claim_audit",
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "maxItems": MAX_CONSTRAINED_AUDIT_CLAIMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "entailed", "partially_entailed", "unsupported",
                                    "contradicted", "citation_missing",
                                ],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["claim", "source_ids", "status", "reason"],
                        "additionalProperties": False,
                    },
                },
                "overall": {"type": "string", "enum": ["entailed", "mixed", "unsupported"]},
            },
            "required": ["claims", "overall"],
            "additionalProperties": False,
        },
    },
}


def _settings_for_response_effort(settings: RagSettings, effort: ResponseEffort) -> RagSettings:
    """Reserve separate server completion capacity for reasoning and the visible answer.

    llama.cpp's OpenAI-compatible ``max_tokens`` covers both reasoning and final
    answer tokens. The server's ``--reasoning-budget`` enforces the reasoning
    portion; the additional capacity here keeps the final-answer limits intact.
    """
    final_output_tokens = FINAL_OUTPUT_TOKEN_LIMITS[effort]
    return settings.model_copy(update={"max_tokens": final_output_tokens + THINKING_TOKEN_ALLOCATION})


def _chat_messages(system_instruction: str, history: list[Message], prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_instruction.strip()}]
    for message in history[-8:]:
        role = "assistant" if message.role == "assistant" else "user"
        messages.append({"role": role, "content": message.content.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def _server_completion(
    app_state,
    messages: list[dict[str, str]],
    settings: RagSettings,
    *,
    stream: bool,
    response_format: dict[str, Any] | None = None,
) -> Any:
    server = models.server_settings(app_state)
    request_payload: dict[str, Any] = {
        "model": getattr(app_state, "active_model_name", None) or server.model_name,
        "messages": messages,
        "stream": stream,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if response_format is not None:
        # llama.cpp converts this OpenAI-compatible response format into a
        # bounded grammar. The schema is used only for the semantic audit; the
        # user-facing answer remains ordinary prose and stays streamable.
        request_payload["response_format"] = response_format
    payload = json.dumps(request_payload).encode("utf-8")
    request = Request(
        f"{server.server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        method="POST",
    )
    try:
        return urlopen(request, timeout=300)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"llama.cpp server returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {server.server_url}: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {server.server_url}: {exc}") from exc


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    return strip_hidden_reasoning(str((choice.get("message") or {}).get("content") or choice.get("text") or ""))


def _stream_server_completion(app_state, messages: list[dict[str, str]], settings: RagSettings) -> Iterator[str]:
    with _server_completion(app_state, messages, settings, stream=True) as response:
        visible_filter = VisibleAnswerFilter()
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                # Flush a visible suffix that was held only because it could
                # have been the beginning of ``<think>``. Returning here
                # would silently drop the final few answer characters.
                break
            try:
                event = json.loads(payload)
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # Reasoning is internal telemetry, not answer content. Do
                    # not emit it, store it, or let it enter claim verification.
                    content = delta.get("content") or choices[0].get("text") or ""
                    if content:
                        visible = visible_filter.feed(str(content))
                        if visible:
                            yield visible
            except json.JSONDecodeError:
                continue
        tail = visible_filter.finish()
        if tail:
            yield tail


def build_system_instruction(
    app_state,
    prompt: str,
    context: str,
    query_meta: dict[str, Any] | None = None,
    *,
    extra_instruction: str = "",
) -> str:
    confidence = query_meta or {}
    architecture_context = ""
    if any(term in prompt.lower() for term in ("architecture", "how do you work", "tech stack", "cephalon internals", "your codebase")):
        architecture_context = (
            "--- SYSTEM ARCHITECTURE (INTERNAL KNOWLEDGE) ---\n"
            f"{app_state.architecture_context}\n"
        )
    no_answer_instruction = (
        "If retrieved sources are weak, be transparent about uncertainty, but still answer naturally when the user is asking "
        "for general reasoning, conversation, brainstorming, coding help, or synthesis that does not require document evidence. "
        "Use retrieved citations only for claims that rely on local documents. "
    )
    return (
        "You are Cephalon, a local assistant with persistent chat memory and optional document retrieval. "
        "Answer in a capable, direct voice that fits the current conversation and the configured external server model's natural style. "
        "Use chat history as normal conversation context. Treat retrieved files as supporting evidence, not as the only thing you can discuss. "
        "Reason through the request before answering, but do not expose hidden chain-of-thought. "
        "Do not repeat the user's prompt as part of the answer. "
        "When an answer depends on retrieved document evidence, cite the relevant source tags exactly as provided, for example [[src:S1]]. "
        "Before saying that the local documents do not contain an answer, inspect every supplied source for directly stated facts; "
        "if a source states the requested fact, answer from it rather than describing other sources as incomplete. "
        "For casual conversation, general knowledge, creative work, or coding guidance, citations are optional and should not be forced. "
        "Do not invent source tags. Do not expose internal parsing instructions. "
        "For multi-part questions, answer each subquestion separately and keep citations attached to the relevant subquestion. "
        f"{no_answer_instruction}"
        f"Current retrieval confidence: {confidence.get('confidence', 'unknown')} / uncertainty: {confidence.get('uncertainty', 'unknown')} / no_answer: {confidence.get('no_answer', False)}.\n\n"
        f"{extra_instruction.strip()}\n\n"
        f"{architecture_context}"
        f"--- START RECALLED MEMORIES & FILES ---\n{context}\n--- END RECALLED MEMORIES & FILES ---\n\n"
    )


def _draft_answer(
    app_state,
    prompt: str,
    context: str,
    history: list[Message],
    settings: RagSettings,
    query_meta: dict[str, Any] | None,
) -> str:
    bounded_history, bounded_context = budget_prompt(
        history,
        context,
        context_window=_active_context_window(app_state, settings.context_tokens),
        output_tokens=settings.max_tokens,
    )
    system_instruction = build_system_instruction(
        app_state,
        prompt,
        bounded_context,
        query_meta,
        extra_instruction=(
            "Produce a compact candidate answer for a second pass to improve. "
            "List the useful claims, citations, and any visible gaps. "
            "Do not output hidden chain-of-thought or discuss this instruction."
        ),
    )
    runtime = getattr(app_state, "model_runtime", None)
    guard = runtime.exclusive() if runtime is not None else nullcontext()
    with guard:
        draft_settings = settings.model_copy(update={"temperature": min(settings.temperature, 0.5)})
        with _server_completion(app_state, _chat_messages(system_instruction, bounded_history, prompt), draft_settings, stream=False) as response:
            result = json.loads(response.read().decode("utf-8"))
    return _response_content(result)


def validate_draft_claims(
    app_state,
    draft: str,
    context: str,
    settings: RagSettings,
) -> dict[str, Any]:
    clean_draft = strip_hidden_reasoning(draft)
    _, bounded_context = budget_prompt(
        [],
        context,
        context_window=_active_context_window(app_state, settings.context_tokens),
        output_tokens=min(settings.max_tokens, 2048),
    )
    deterministic = support.validate_answer_claims(
        clean_draft,
        support.sources_from_context(bounded_context),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Audit the candidate answer claim by claim against the supplied evidence. "
                f"Audit at most {MAX_CONSTRAINED_AUDIT_CLAIMS} material claims and keep each claim/reason short. "
                "Return JSON only with this shape: "
                '{"claims":[{"claim":"...","source_ids":["S1"],'
                '"status":"entailed|partially_entailed|unsupported|contradicted|citation_missing","reason":"..."}],'
                '"overall":"entailed|mixed|unsupported"}. '
                "A citation tag alone is not proof; the cited evidence must support the claim. "
                "Mark opposite statements as contradicted. Treat arithmetic as unsupported unless the cited values imply it. "
                "Do not add new facts or expose chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": (
                f"--- EVIDENCE ---\n{bounded_context}\n--- END EVIDENCE ---\n\n"
                f"--- CANDIDATE ANSWER ---\n{clean_draft}\n--- END CANDIDATE ANSWER ---"
            ),
        },
    ]
    validation_settings = settings.model_copy(update={
        "temperature": 0.0,
        # The previous 2048 cap truncated otherwise valid constrained JSON on
        # long synthesis drafts. This is still a single bounded audit call;
        # deterministic verification remains responsible for every claim.
        "max_tokens": min(max(settings.max_tokens, 3072), 4096),
    })
    runtime = getattr(app_state, "model_runtime", None)
    guard = runtime.exclusive() if runtime is not None else nullcontext()
    try:
        with guard:
            with _server_completion(
                app_state,
                messages,
                validation_settings,
                stream=False,
                response_format=CLAIM_AUDIT_RESPONSE_FORMAT,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        content = _response_content(response_payload)
        fenced = content.strip()
        if fenced.startswith("```"):
            fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", fenced, flags=re.IGNORECASE)
        parsed = json.loads(fenced)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
            raise ValueError("Validator response did not contain a claims list.")
        merged = _merge_semantic_and_deterministic_audits(deterministic, parsed)
        merged.update({
            "validator_constraint": "json_schema",
            "validator_json_parse_success": True,
            "validator_fallback": False,
        })
        return merged
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        return {
            "status": "deterministic_fallback",
            "overall": _deterministic_overall(deterministic),
            "claims": deterministic["claims"],
            "method": deterministic["method"],
            "deterministic": deterministic,
            "validator_constraint": "json_schema",
            "validator_json_parse_success": False,
            "validator_fallback": True,
            "validator_error": str(exc)[:500],
            "validator_fallback_reason": str(exc)[:500],
        }


def _merge_semantic_and_deterministic_audits(
    deterministic: dict[str, Any],
    semantic: dict[str, Any],
) -> dict[str, Any]:
    """Merge semantic judgments while preserving deterministic hard failures.

    Arithmetic mismatches, negation contradictions, and missing citations may
    never be upgraded by the model. A semantic entailment may upgrade lexical
    partial/unsupported text when no deterministic hard failure exists.
    """

    semantic_claims = semantic.get("claims", [])[:MAX_SEMANTIC_AUDIT_CLAIMS]
    merged: list[dict[str, Any]] = []
    for index, deterministic_claim in enumerate(deterministic.get("claims", [])):
        model_claim = semantic_claims[index] if index < len(semantic_claims) and isinstance(semantic_claims[index], dict) else {}
        model_status = _semantic_status(model_claim.get("status"))
        deterministic_status = deterministic_claim.get("entailment_status", "unsupported")
        numeric_status = (deterministic_claim.get("numeric_verification") or {}).get("status")
        unknown_citation = "do not identify supplied evidence" in str(deterministic_claim.get("reason", ""))
        if deterministic_status in HARD_FAILURE_STATES or numeric_status in {"contradicted", "unsupported"} or unknown_citation:
            final_status = deterministic_status
        elif model_status == "contradicted":
            final_status = "contradicted"
        elif model_status == "entailed":
            final_status = "entailed"
        elif model_status == "partially_entailed" and deterministic_status == "unsupported":
            final_status = "partially_entailed"
        else:
            final_status = deterministic_status
        merged.append({
            **deterministic_claim,
            "entailment_status": final_status,
            "semantic_status": model_status,
            "semantic_reason": str(model_claim.get("reason") or "")[:500],
        })
    overall = (
        "unsupported"
        if any(item["entailment_status"] in {"unsupported", "contradicted", "citation_missing"} for item in merged)
        else "mixed"
        if any(item["entailment_status"] == "partially_entailed" for item in merged)
        else "entailed"
    )
    return {
        "status": "completed",
        "overall": overall,
        "claims": merged,
        "method": "semantic_audit_with_deterministic_entailment_v2",
        "deterministic": deterministic,
    }


def _semantic_status(value: Any) -> str:
    normalized = str(value or "unsupported").strip().lower()
    aliases = {
        "supported": "entailed",
        "weak": "partially_entailed",
        "uncited": "citation_missing",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {
        "entailed", "partially_entailed", "unsupported", "contradicted", "citation_missing",
    } else "unsupported"


def _deterministic_overall(audit: dict[str, Any]) -> str:
    if audit.get("unsupported_claim_count") or audit.get("uncited_claim_count"):
        return "unsupported"
    if audit.get("weak_claim_count"):
        return "mixed"
    return "entailed"


def _compact_audit_for_repair(audit: dict[str, Any]) -> str:
    """Serialize only bounded audit directives for the single repair call.

    Deterministic verification retains full evidence for diagnostics, but that
    evidence can contain several long source excerpts. Embedding the complete
    audit in the repair prompt bypasses the ordinary context budget and can
    make an external llama.cpp server reject an otherwise valid request. The
    repair already receives the bounded source context separately, so it needs
    only short claim/status/source directives; hard-failure precedence remains
    enforced after the repaired answer is generated.
    """

    compact_claims: list[dict[str, Any]] = []
    for claim in audit.get("claims", [])[:MAX_CONSTRAINED_AUDIT_CLAIMS]:
        if not isinstance(claim, dict):
            continue
        numeric = claim.get("numeric_verification") or {}
        compact_claims.append({
            "claim": str(claim.get("text") or claim.get("claim") or "")[:MAX_REPAIR_AUDIT_CLAIM_CHARS],
            "source_ids": [str(source_id) for source_id in (claim.get("source_ids") or [])[:8]],
            "status": str(claim.get("entailment_status") or claim.get("status") or "unsupported"),
            "numeric_status": str(numeric.get("status") or "not_applicable"),
            "reason": str(claim.get("reason") or claim.get("semantic_reason") or "")[:MAX_REPAIR_AUDIT_REASON_CHARS],
        })
    return json.dumps(
        {"overall": str(audit.get("overall") or "unsupported"), "claims": compact_claims},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _audit_needs_repair(draft: str, audit: dict[str, Any]) -> bool:
    """Return whether Thorough mode may spend its single repair completion."""

    return not draft.strip() or not audit.get("claims") or audit.get("overall") != "entailed"


def stream_response_events(
    app_state,
    prompt: str,
    context: str,
    history: list[Message],
    settings: RagSettings,
    query_meta: dict[str, Any] | None = None,
    *,
    response_effort: ResponseEffort = "balanced",
) -> Iterator[tuple[str, str]]:
    effort = response_effort if response_effort in {"quick", "balanced", "thorough"} else "balanced"
    extra_instruction = ""
    generation_settings = _settings_for_response_effort(settings, effort)
    completion_calls = 0
    if effort == "quick":
        yield "phase", "answering"
    elif effort == "thorough":
        yield "phase", "drafting"
        draft = strip_hidden_reasoning(_draft_answer(app_state, prompt, context, history, generation_settings, query_meta))
        completion_calls += 1
        yield "phase", "validating"
        validation = validate_draft_claims(app_state, draft, context, generation_settings)
        completion_calls += 1
        repair_needed = not settings.verified_answer_repair or _audit_needs_repair(draft, validation)
        if query_meta is not None:
            query_meta["claim_verification"] = validation
            query_meta["repair_attempted"] = repair_needed
        if not repair_needed:
            if query_meta is not None:
                query_meta["completion_call_count"] = completion_calls
            yield "phase", "verified"
            yield "token", draft
            return
        yield "phase", "refining"
        extra_instruction = (
            "Repair the candidate answer once. Remove or qualify unsupported claims, correct every contradicted or numerically inaccurate claim, "
            "add missing citations only when supplied evidence supports them, preserve valid source tags, and return only the final answer. "
            "Do not mention the candidate draft or audit.\n\n"
            f"--- CANDIDATE ANSWER ---\n{strip_hidden_reasoning(draft)}\n--- END CANDIDATE ANSWER ---\n\n"
            f"--- CLAIM AUDIT ---\n{_compact_audit_for_repair(validation)}\n--- END CLAIM AUDIT ---"
        )
    else:
        yield "phase", "answering"

    bounded_history, bounded_context = budget_prompt(
        history,
        context,
        context_window=_active_context_window(app_state, generation_settings.context_tokens),
        output_tokens=generation_settings.max_tokens,
    )
    system_instruction = build_system_instruction(
        app_state,
        prompt,
        bounded_context,
        query_meta,
        extra_instruction=extra_instruction,
    )
    runtime = getattr(app_state, "model_runtime", None)
    guard = runtime.exclusive() if runtime is not None else nullcontext()
    with guard:
        completion_calls += 1
        for content in _stream_server_completion(app_state, _chat_messages(system_instruction, bounded_history, prompt), generation_settings):
            yield "token", content
    if query_meta is not None:
        query_meta["completion_call_count"] = completion_calls


def _active_context_window(app_state, fallback: int) -> int:
    """Return a usable prompt window for configured external llama.cpp servers.

    External servers do not have to report their loaded context size. In that
    valid state Cephalon keeps ``active_context_tokens`` as ``None``; the
    request's validated RAG setting is therefore the authoritative fallback.
    Keeping this decision at the generation boundary prevents every response
    mode from passing ``None`` into integer prompt-budget arithmetic.
    """

    active = getattr(app_state, "active_context_tokens", None)
    return int(active) if active is not None else int(fallback)


def stream_response(
    app_state,
    prompt: str,
    context: str,
    history: list[Message],
    settings: RagSettings,
    query_meta: dict[str, Any] | None = None,
    *,
    response_effort: ResponseEffort = "balanced",
) -> Iterator[str]:
    for event_type, value in stream_response_events(
        app_state,
        prompt,
        context,
        history,
        settings,
        query_meta,
        response_effort=response_effort,
    ):
        if event_type == "token":
            yield value


def stream_llama(app_state, prompt: str, context: str, history: list[Message], settings: RagSettings, query_meta: dict | None = None):
    yield from stream_response(app_state, prompt, context, history, settings, query_meta)
