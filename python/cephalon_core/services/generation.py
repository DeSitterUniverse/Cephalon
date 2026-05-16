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
        "If retrieved sources are weak or insufficient, say that the library does not contain enough evidence, "
        "then list the closest cited matches and their scores instead of inventing an answer. "
    )
    system_instruction = (
        "You are Cephalon, an advanced, locally-hosted AI intelligence platform with persistent memory. "
        "You prioritize user privacy, remaining 100% offline. "
        "Reason through the evidence before answering, but only show a concise evidence plan and final answer to the user. "
        "Tone: Analytical, helpful, and highly competent. Avoid AI mannerisms like 'As an AI...'. "
        "Below are fragments of your past conversations and files added to your local memory library. "
        "Synthesize this context carefully to answer the user's prompt. "
        "Do not repeat the user's prompt as part of the answer. "
        "When using retrieved evidence, cite source tags exactly as provided, for example [[src:S1]]. "
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
