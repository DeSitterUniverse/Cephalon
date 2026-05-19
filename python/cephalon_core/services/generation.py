from ..schemas import Message, RagSettings


def _format_prompt(system_instruction: str, history: list[Message], prompt: str) -> str:
    lines = [f"<|im_start|>system\n{system_instruction.strip()}<|im_end|>"]
    for message in history[-8:]:
        role = "assistant" if message.role == "assistant" else "user"
        lines.append(f"<|im_start|>{role}\n{message.content.strip()}<|im_end|>")
    lines.append(f"<|im_start|>user\n{prompt.strip()}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


def stream_llama(app_state, prompt: str, context: str, history: list[Message], settings: RagSettings, query_meta: dict | None = None):
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
    system_instruction = (
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
        f"{architecture_context}"
        f"--- START RECALLED MEMORIES & FILES ---\n{context}\n--- END RECALLED MEMORIES & FILES ---\n\n"
    )
    stream = app_state.llm(
        _format_prompt(system_instruction, history, prompt),
        stream=True,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stop=["<|im_end|>"],
        echo=False,
    )
    for chunk in stream:
        content = chunk["choices"][0].get("text", "")
        if content:
            yield content
