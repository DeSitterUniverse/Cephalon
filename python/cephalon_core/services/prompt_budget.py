from ..schemas import Message


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _truncate_to_tokens(text: str, tokens: int) -> str:
    if estimate_tokens(text) <= tokens:
        return text
    return text[: max(0, tokens * 4)].rstrip()


def budget_prompt(
    history: list[Message],
    context: str,
    *,
    context_window: int,
    output_tokens: int,
) -> tuple[list[Message], str]:
    usable = max(512, int(context_window) - max(128, int(output_tokens)) - 512)
    history_budget = max(256, int(usable * 0.35))
    context_budget = max(256, usable - history_budget)

    first_user = next((message for message in history if message.role == "user"), None)
    selected_reversed: list[Message] = []
    spent = 0
    for message in reversed(history):
        if first_user is message:
            continue
        cost = estimate_tokens(message.content) + 8
        if selected_reversed and spent + cost > history_budget:
            break
        if cost > history_budget - spent:
            content = _truncate_to_tokens(message.content, max(32, history_budget - spent - 8))
            selected_reversed.append(message.model_copy(update={"content": content}))
            spent = history_budget
            break
        selected_reversed.append(message)
        spent += cost

    selected = list(reversed(selected_reversed))
    if first_user is not None and first_user not in selected:
        first_budget = max(32, history_budget - spent - 8)
        selected.insert(0, first_user.model_copy(update={"content": _truncate_to_tokens(first_user.content, first_budget)}))

    return selected, _truncate_to_tokens(context, context_budget)
