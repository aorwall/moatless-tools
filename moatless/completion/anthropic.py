import json
import logging
from typing import Optional, Union, List

import anthropic
import tenacity
from anthropic import Anthropic, AnthropicBedrock, NOT_GIVEN
from anthropic.types import ToolUseBlock, TextBlock
from anthropic.types.beta import (
    BetaToolUseBlock,
    BetaTextBlock,
)
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from pydantic import Field, ValidationError

from moatless.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat, CompletionResponse
from moatless.completion.model import Completion, StructuredOutput, Usage
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


class AnthtropicCompletionModel(CompletionModel):
    response_format: Optional[LLMResponseFormat] = Field(
        LLMResponseFormat.TOOLS, description="The response format expected from the LLM"
    )

    @property
    def supports_anthropic_computer_use(self):
        return "claude-3-5-sonnet-20241022" in self.model

    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]] | type[StructuredOutput],
    ) -> CompletionResponse:
        # Convert Message objects to dictionaries if needed
        messages = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages
        ]

        total_usage = Usage()
        retry_count = 0

        tools = []
        tool_choice = {"type": "any"}

        actions = []
        if not response_model:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN
        else:
            if isinstance(response_model, list):
                actions = response_model
            elif response_model:
                actions = [response_model]

            for action in actions:
                if hasattr(action, "name") and action.name == "str_replace_editor":
                    tools.append(
                        {"name": "str_replace_editor", "type": "text_editor_20241022"}
                    )
                else:
                    schema = action.anthropic_schema()

                    # Remove scratch pad field, use regular text block for thoughts
                    if "thoughts" in schema["input_schema"]["properties"]:
                        del schema["input_schema"]["properties"]["thoughts"]

                    tools.append(schema)

        system_message = {"text": system_prompt, "type": "text"}

        anthropic_messages = anthropic_messages_pt(
            model=self.model,
            messages=messages,
            llm_provider="anthropic",
        )
        if "anthropic" in self.model:
            anthropic_client = AnthropicBedrock()
            betas = ["computer-use-2024-10-22"]  # , "prompt-caching-2024-07-31"]
            extra_headers = {}  # "X-Amzn-Bedrock-explicitPromptCaching": "enabled"}
        else:
            anthropic_client = Anthropic()
            extra_headers = {}
            betas = ["computer-use-2024-10-22", "prompt-caching-2024-07-31"]
            _inject_prompt_caching(anthropic_messages)
            system_message["cache_control"] = {"type": "ephemeral"}

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(anthropic.BadRequestError),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            nonlocal retry_count, total_usage

            completion_response = None
            try:
                completion_response = anthropic_client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[system_message],
                    tools=tools,
                    messages=anthropic_messages,
                    betas=betas,
                    extra_headers=extra_headers,
                )

                total_usage += Usage.from_completion_response(
                    completion_response, self.model
                )

                def get_response_format(name: str):
                    if len(actions) == 1:
                        return actions[0]
                    else:
                        for check_action in actions:
                            if check_action.name == block.name:
                                return check_action
                    return None

                text = None
                structured_outputs = []
                for block in completion_response.content:
                    if isinstance(block, ToolUseBlock) or isinstance(
                        block, BetaToolUseBlock
                    ):
                        action = None

                        tool_call_id = block.id

                        if len(actions) == 1:
                            action = actions[0]
                        else:
                            for check_action in actions:
                                if check_action.name == block.name:
                                    action = check_action
                                    break

                        if not action:
                            raise ValueError(f"Unknown action {block.name}")

                        action_args = action.model_validate(block.input)
                        structured_outputs.append(action_args)

                    elif isinstance(block, TextBlock) or isinstance(
                        block, BetaTextBlock
                    ):
                        text = block.text

                    else:
                        logger.warning(f"Unexpected block {block}]")

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                    usage=total_usage,
                    retries=retry_count,
                )

                # Log summary of the response
                action_names = [
                    output.__class__.__name__ for output in structured_outputs
                ]
                has_text = bool(text and text.strip())
                if action_names:
                    logger.info(
                        f"Completion response summary - Actions: {action_names}, Has text: {has_text}"
                    )
                else:
                    logger.info(
                        f"Completion response summary - Text only: {text[:200]}..."
                    )

                return CompletionResponse(
                    structured_outputs=structured_outputs,
                    text_response=text,
                    completion=completion,
                )

            except ValidationError as e:
                logger.warning(
                    f"Validation failed with error {e}. Response: {json.dumps(completion_response.model_dump() if completion_response else None, indent=2)}"
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            block.model_dump() for block in completion_response.content
                        ],
                    }
                )
                messages.append(
                    self._create_user_message(
                        tool_call_id, f"The response was invalid. Fix the errors: {e}"
                    )
                )
                retry_count += 1
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                ) from e
            except Exception as e:
                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                    messages=messages,
                    last_completion=completion_response,
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()


def _inject_prompt_caching(
    messages: list[
        Union[
            "AnthropicMessagesUserMessageParam", "AnthopicMessagesAssistantMessageParam"
        ]
    ],
):
    from anthropic.types.beta import BetaCacheControlEphemeralParam

    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        # message["role"] == "user" and
        if isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break
