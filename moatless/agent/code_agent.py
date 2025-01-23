import json
import logging
from typing import List

from moatless.actions import (
    FindClass,
    FindFunction,
    FindCodeSnippet,
    SemanticSearch,
    ViewCode,
)
from moatless.actions.action import Action
from moatless.actions.append_string import AppendString
from moatless.actions.claude_text_editor import ClaudeEditTool
from moatless.actions.create_file import CreateFile
from moatless.actions.finish import Finish
from moatless.actions.list_files import ListFiles
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.actions.string_replace import StringReplace
from moatless.actions.verified_finish import VerifiedFinish
from moatless.agent.agent import ActionAgent
from moatless.agent.code_prompts import (
    AGENT_ROLE,
    generate_react_guidelines,
    REACT_CORE_OPERATION_RULES,
    ADDITIONAL_NOTES,
    generate_workflow_prompt,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
    generate_guideline_prompt,
)
from moatless.completion.base import (
    LLMResponseFormat,
    BaseCompletionModel,
)
from moatless.index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.schema import MessageHistoryType

logger = logging.getLogger(__name__)


class CodingAgent(ActionAgent):
    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: BaseCompletionModel,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        edit_completion_model: BaseCompletionModel | None = None,
        message_history_type: MessageHistoryType | None = None,
        thoughts_in_action: bool = False,
        disable_thoughts: bool = False,
        few_shot_examples: bool = False,
        **kwargs,
    ):
        # Clone the completion model to ensure we have our own instance
        completion_model = completion_model.clone()

        if message_history_type is None:
            if completion_model.response_format == LLMResponseFormat.TOOLS:
                message_history_type = MessageHistoryType.MESSAGES
            else:
                message_history_type = MessageHistoryType.REACT

        action_completion_format = completion_model.response_format
        if action_completion_format != LLMResponseFormat.TOOLS:
            logger.info("Default to JSON as Response format for action completion model")
            action_completion_format = LLMResponseFormat.JSON

        # Create action completion model by cloning the input model with JSON response format
        action_completion_model = completion_model.clone(response_format=action_completion_format)
        action_completion_model.message_cache = False

        supports_anthropic_computer_use = completion_model.model.startswith("claude-3-5-sonnet")

        if supports_anthropic_computer_use:
            actions = create_claude_coding_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
            )

            action_type = "Claude actions with computer use capability"
        else:
            actions = create_edit_code_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
            )
            action_type = "standard edit code actions"

        # Generate workflow prompt based on available actions
        workflow_prompt = generate_workflow_prompt(actions, runtime is not None)

        # Compose system prompt based on model type and format
        system_prompt = AGENT_ROLE
        if completion_model.response_format == LLMResponseFormat.REACT:
            system_prompt += "\n"

            if disable_thoughts:
                system_prompt += REACT_CORE_OPERATION_RULES_NO_THOUGHTS
            else:
                system_prompt += REACT_CORE_OPERATION_RULES

        elif completion_model.response_format == LLMResponseFormat.TOOLS:
            system_prompt += generate_react_guidelines(thoughts_in_action)
        else:
            raise ValueError(f"Unsupported response format: {completion_model.response_format}")

        # Add workflow and guidelines
        system_prompt += (
            "\n"
            + workflow_prompt
            + "\n"
            + generate_guideline_prompt(runtime is not None, thoughts_in_action)
            + "\n"
            + ADDITIONAL_NOTES
        )

        message_generator = MessageHistoryGenerator.create(
            message_history_type=message_history_type,
            include_file_context=True,
            thoughts_in_action=thoughts_in_action,
        )

        config = {
            "completion_model": completion_model.__class__.__name__,
            "code_index_enabled": code_index is not None,
            "runtime_enabled": runtime is not None,
            "edit_completion_model": edit_completion_model.__class__.__name__ if edit_completion_model else None,
            "action_type": action_type,
            "actions": [a.__class__.__name__ for a in actions],
            "message_history_type": message_history_type.value,
            "thoughts_in_action": thoughts_in_action,
            "file_context_enabled": True,
        }

        logger.info(f"Created CodingAgent with configuration: {json.dumps(config, indent=2)}")

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            use_few_shots=few_shot_examples,
            thoughts_in_action=thoughts_in_action,
            **kwargs,
        )


def create_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
) -> List[Action]:
    """Create the common base actions used across all action creators."""
    return [
        SemanticSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        FindClass(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        FindFunction(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        FindCodeSnippet(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
    ]


def create_edit_code_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    """Create a list of simple code modification actions."""
    actions = create_base_actions(repository, code_index, completion_model.clone())

    edit_actions = [
        StringReplace(repository=repository, code_index=code_index),
        # InsertLine(repository=repository,  code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        AppendString(repository=repository, code_index=code_index),
    ]

    if runtime:
        edit_actions.append(RunTests(repository=repository, code_index=code_index, runtime=runtime))

    actions.extend(edit_actions)
    actions.extend([VerifiedFinish(), Reject()])
    return actions


def create_claude_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    actions = create_base_actions(repository, code_index, completion_model.clone())
    actions.append(
        ClaudeEditTool(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
    )
    actions.append(ListFiles())
    if runtime:
        actions.append(RunTests(repository=repository, code_index=code_index, runtime=runtime))
    actions.extend([Finish(), Reject()])
    return actions


def create_all_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
) -> List[Action]:
    actions = create_base_actions(repository, code_index, completion_model.clone())
    actions.extend(create_edit_code_actions(repository, code_index, completion_model.clone()))
    actions.append(
        ClaudeEditTool(code_index=code_index, repository=repository, completion_model=completion_model.clone())
    )
    return actions
