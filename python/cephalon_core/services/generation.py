from collections.abc import Iterator
from contextlib import nullcontext
import json
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..schemas import Message, RagSettings
from . import models
from .prompt_budget import budget_prompt


ResponseEffort = Literal["quick", "balanced", "thorough"]
THINKING_TOKEN_ALLOCATION = 2048
FINAL_OUTPUT_TOKEN_LIMITS = {
    "quick": 2048,
    "balanced": 4096,
    "thorough": 4096,
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


def _server_completion(app_state, messages: list[dict[str, str]], settings: RagSettings, *, stream: bool) -> Any:
    server = models.server_settings(app_state)
    payload = json.dumps({
        "model": getattr(app_state, "active_model_name", None) or server.model_name,
        "messages": messages,
        "stream": stream,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }).encode("utf-8")
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
    return str((choice.get("message") or {}).get("content") or choice.get("text") or "").strip()


def _stream_server_completion(app_state, messages: list[dict[str, str]], settings: RagSettings) -> Iterator[str]:
    with _server_completion(app_state, messages, settings, stream=True) as response:
        in_thinking = False
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                event = json.loads(payload)
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content = delta.get("content") or choices[0].get("text") or ""
                    if reasoning:
                        if not in_thinking:
                            yield "<think>"
                            in_thinking = True
                        yield str(reasoning)
                    if content:
                        if in_thinking:
                            yield "</think>"
                            in_thinking = False
                        yield str(content)
            except json.JSONDecodeError:
                continue
        if in_thinking:
            yield "</think>"


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
        context_window=getattr(app_state, "active_context_tokens", settings.context_tokens),
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
    if effort == "quick":
        yield "phase", "answering"
    elif effort == "thorough":
        yield "phase", "drafting"
        draft = _draft_answer(app_state, prompt, context, history, generation_settings, query_meta)
        yield "phase", "refining"
        extra_instruction = (
            "Improve the candidate answer below. Correct unsupported claims, fill visible gaps, "
            "preserve valid source tags, and return only the final answer. Do not mention the candidate draft.\n\n"
            f"--- CANDIDATE ANSWER ---\n{draft}\n--- END CANDIDATE ANSWER ---"
        )
    else:
        yield "phase", "answering"

    bounded_history, bounded_context = budget_prompt(
        history,
        context,
        context_window=getattr(app_state, "active_context_tokens", generation_settings.context_tokens),
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
        for content in _stream_server_completion(app_state, _chat_messages(system_instruction, bounded_history, prompt), generation_settings):
            yield "token", content


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
