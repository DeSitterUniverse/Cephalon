from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, Literal

from ..schemas import Message, RagSettings
from .prompt_budget import budget_prompt


ResponseEffort = Literal["quick", "balanced", "thorough"]


def _format_prompt(system_instruction: str, history: list[Message], prompt: str) -> str:
    lines = [f"<|im_start|>system\n{system_instruction.strip()}<|im_end|>"]
    for message in history[-8:]:
        role = "assistant" if message.role == "assistant" else "user"
        lines.append(f"<|im_start|>{role}\n{message.content.strip()}<|im_end|>")
    lines.append(f"<|im_start|>user\n{prompt.strip()}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


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
        "Answer in a capable, direct voice that fits the current conversation and the selected local model's natural style. "
        "Use chat history as normal conversation context. Treat retrieved files as supporting evidence, not as the only thing you can discuss. "
        "Reason through the request before answering, but do not expose hidden chain-of-thought. "
        "Do not repeat the user's prompt as part of the answer. "
        "When an answer depends on retrieved document evidence, cite the relevant source tags exactly as provided, for example [[src:S1]]. "
        "For casual conversation, general knowledge, creative work, or coding guidance, citations are optional and should not be forced. "
        "Do not invent source tags. Do not expose internal parsing instructions. "
        "For multi-part questions, answer each subquestion separately and keep citations attached to the relevant subquestion. "
        f"{no_answer_instruction}"
        f"Current retrieval confidence: {confidence.get('confidence', 'unknown')} / uncertainty: {confidence.get('uncertainty', 'unknown')} / no_answer: {confidence.get('no_answer', False)}.\n\n"
        f"{extra_instruction.strip()}\n\n"
        f"{architecture_context}"
        f"--- START RECALLED MEMORIES & FILES ---\n{context}\n--- END RECALLED MEMORIES & FILES ---\n\n"
    )


def _completion_text(result: dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("text") or "").strip()


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
        output_tokens=max(128, min(settings.max_tokens, 512)),
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
        result = app_state.llm(
            _format_prompt(system_instruction, bounded_history, prompt),
            stream=False,
            temperature=min(settings.temperature, 0.5),
            max_tokens=max(128, min(settings.max_tokens, 512)),
            stop=["<|im_end|>"],
            echo=False,
        )
    return _completion_text(result)


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
    generation_settings = settings
    if effort == "quick":
        generation_settings = settings.model_copy(update={"max_tokens": min(settings.max_tokens, 384)})
        yield "phase", "answering"
    elif effort == "thorough":
        yield "phase", "drafting"
        draft = _draft_answer(app_state, prompt, context, history, settings, query_meta)
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
        stream = app_state.llm(
            _format_prompt(system_instruction, bounded_history, prompt),
            stream=True,
            temperature=generation_settings.temperature,
            max_tokens=generation_settings.max_tokens,
            stop=["<|im_end|>"],
            echo=False,
        )
        for chunk in stream:
            content = chunk["choices"][0].get("text", "")
            if content:
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
