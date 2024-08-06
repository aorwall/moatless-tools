
import instructor


def instructor_mode_by_model(model: str) -> instructor.Mode | None:
    if "gpt" in model:
        return instructor.Mode.TOOLS

    if "claude" in model:
        return instructor.Mode.TOOLS

    if model.startswith("claude"):
        return instructor.Mode.ANTHROPIC_TOOLS

    if model.startswith("openrouter/anthropic/claude"):
        return instructor.Mode.TOOLS

    return instructor.Mode.JSON
