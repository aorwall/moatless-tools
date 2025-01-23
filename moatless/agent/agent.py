import importlib
import json
import logging
import traceback
from typing import List, Type, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
)
from moatless.agent.settings import AgentSettings
from moatless.completion import BaseCompletionModel, LLMResponseFormat
from moatless.completion.model import Completion
from moatless.exceptions import CompletionError, RejectError, RuntimeError, CompletionRejectError
from moatless.index.code_index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class ActionAgent(BaseModel):
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    use_few_shots: bool = Field(False, description="Whether to use few-shot examples for generating completions")
    thoughts_in_action: bool = Field(True, description="")
    actions: List[Action] = Field(default_factory=list)
    message_generator: MessageHistoryGenerator = Field(
        description="Generator for message history",
    )

    _completion: BaseCompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        completion: BaseCompletionModel,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        message_generator: MessageHistoryGenerator | None = None,
        **data,
    ):
        actions = actions or []
        message_generator = message_generator or MessageHistoryGenerator()
        super().__init__(
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            **data,
        )
        self._completion = completion
        self.set_actions(actions)

    @classmethod
    def from_agent_settings(cls, agent_settings: AgentSettings, actions: List[Action] | None = None):
        if agent_settings.actions:
            actions = [action for action in actions if action.__class__.__name__ in agent_settings.actions]

        return cls(
            completion_model=agent_settings.completion_model,
            system_prompt=agent_settings.system_prompt,
            actions=actions,
        )

    def set_actions(self, actions: List[Action]):
        self.actions = actions
        self._action_map = {action.args_schema: action for action in actions}
        action_args = [action.args_schema for action in self.actions]
        system_prompt = self.generate_system_prompt()
        self._completion.initialize(action_args, system_prompt)

    @model_validator(mode="after")
    def verify_actions(self) -> "ActionAgent":
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValueError(f"Invalid action type: {type(action)}. Expected Action subclass.")
            if not hasattr(action, "args_schema"):
                raise ValueError(f"Action {action.__class__.__name__} is missing args_schema attribute")
        return self

    def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""

        if node.action:
            logger.info(f"Node{node.node_id}: Resetting node")
            node.reset()

        node.possible_actions = [action.name for action in self.actions]

        try:
            messages = self.message_generator.generate_messages(node)
            logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")

            completion_response = self._completion.create_completion(
                messages=messages,
            )
            node.completions["build_action"] = completion_response.completion

            node.assistant_message = completion_response.text_response
            if not completion_response.structured_outputs:
                raise RejectError("No action found")

            if completion_response.structured_outputs:
                node.action_steps = [ActionStep(action=action) for action in completion_response.structured_outputs]

        except CompletionError as e:
            node.terminal = True
            node.error = traceback.format_exc()
            if e.last_completion:
                logger.error(f"Node{node.node_id}: Build action failed with completion error: {e}")

                node.completions["build_action"] = Completion.from_llm_completion(
                    input_messages=e.messages if hasattr(e, "messages") else [],
                    completion_response=e.last_completion,
                    model=self.completion.model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )
            else:
                logger.exception(f"Node{node.node_id}: Build action failed with error")

            if isinstance(e, CompletionRejectError):
                return
            else:
                raise e
        except Exception as e:
            node.terminal = True
            node.error = traceback.format_exc()
            logger.error(f"Node{node.node_id}: Build action failed with error: {e}. Type {type(e)}")

            raise e

        if node.action is None:
            return

        duplicate_node = node.find_duplicate()
        if duplicate_node:
            node.is_duplicate = True
            logger.info(f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution.")
            return

        action_names = [action_step.action.name for action_step in node.action_steps]
        logger.info(f"Node{node.node_id}: Execute actions: {action_names}")
        for action_step in node.action_steps:
            self._execute(node, action_step)

    def _execute(self, node: Node, action_step: ActionStep):
        action = self._action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self._action_map.keys()}"
            )
            raise RuntimeError(f"Action {type(node.action)} not found in action map.")

        try:
            action_step.observation = action.execute(action_step.action, file_context=node.file_context, workspace=node.workspace)
            if not action_step.observation:
                logger.warning(f"Node{node.node_id}: Action {action_step.action.name} returned no observation")
            else:
                node.terminal = action_step.observation.terminal
                if action_step.observation.execution_completion:
                    action_step.completion = action_step.observation.execution_completion

            logger.info(
                f"Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. "
                f"Output: {action_step.observation.message if node.observation else None}"
            )

        except CompletionRejectError as e:
            logger.warning(f"Node{node.node_id}: Action rejected: {e.message}")
            if e.last_completion:
                action_step.completion = Completion.from_llm_completion(
                    input_messages=e.messages,
                    completion_response=e.last_completion,
                    model=self.completion.model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )
            node.error = traceback.format_exc()
            node.terminal = True
            raise e

    def generate_system_prompt(self) -> str:
        """Generate a system prompt for the agent."""

        system_prompt = self.system_prompt
        if self.use_few_shots:
            system_prompt += "\n\n" + self.generate_few_shots()

        return system_prompt

    def generate_few_shots(self) -> str:
        few_shot_examples = []
        for action in self.actions:
            examples = action.get_few_shot_examples()
            if examples:
                few_shot_examples.extend(examples)

        prompt = ""
        if few_shot_examples:
            prompt += "\n\n# Examples\nHere are some examples of how to use the available actions:\n\n"
            for i, example in enumerate(few_shot_examples):
                if self.completion.response_format == LLMResponseFormat.REACT:
                    prompt += f"\n**Example {i + 1}**"
                    action_data = example.action.model_dump()
                    thoughts = action_data.pop("thoughts", "")

                    # Special handling for StringReplace and CreateFile action
                    if example.action.__class__.__name__ in [
                        "StringReplaceArgs",
                        "CreateFileArgs",
                        "AppendStringArgs",
                        "InsertLinesArgs",
                        "FindCodeSnippetArgs",
                    ]:
                        prompt += f"\nTask: {example.user_input}"
                        if self.thoughts_in_action:
                            prompt += f"\nThought: {thoughts}\n"
                        prompt += f"Action: {str(example.action.name)}\n"

                        if example.action.__class__.__name__ == "StringReplaceArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<old_str>\n{action_data['old_str']}\n</old_str>\n"
                            prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                        elif example.action.__class__.__name__ == "AppendStringArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                        elif example.action.__class__.__name__ == "CreateFileArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<file_text>\n{action_data['file_text']}\n</file_text>\n"
                        elif example.action.__class__.__name__ == "InsertLinesArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<insert_line>{action_data['insert_line']}</insert_line>\n"
                            prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                        elif example.action.__class__.__name__ == "FindCodeSnippetArgs":
                            if "file_pattern" in action_data:
                                prompt += f"<file_pattern>{action_data['file_pattern']}</file_pattern>\n"
                            prompt += f"<code_snippet>{action_data['code_snippet']}</code_snippet>\n"
                    else:
                        # Original JSON format for other actions
                        prompt += (
                            f"\nTask: {example.user_input}"
                            f"\nThought: {thoughts}\n"
                            f"Action: {str(example.action.name)}\n"
                            f"{json.dumps(action_data)}\n\n"
                        )

                elif self.completion.response_format == LLMResponseFormat.JSON:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name,
                    }
                    prompt += (
                        f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"
                    )

                elif self.completion.response_format == LLMResponseFormat.TOOLS:
                    tools_json = {"tool": example.action.name}
                    if self.thoughts_in_action:
                        tools_json.update(example.action.model_dump())
                    else:
                        tools_json.update(example.action.model_dump(exclude={"thoughts"}))

                    prompt += f"Task: {example.user_input}\n"
                    if not self.thoughts_in_action:
                        prompt += f"<thoughts>{example.action.thoughts}</thoughts>\n"
                    prompt += json.dumps(tools_json)
                    prompt += "\n\n"

        return prompt

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        dump["actions"] = []
        dump["agent_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))
        return dump

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion", None)
            agent_class_path = obj.pop("agent_class", None)

            message_generator_data = obj.get("message_generator", {})
            if message_generator_data:
                obj["message_generator"] = MessageHistoryGenerator.model_validate(message_generator_data)

            if completion_data:
                obj["completion"] = BaseCompletionModel.model_validate(completion_data)
            else:
                obj["completion"] = None

            if repository:
                obj["actions"] = [
                    Action.model_validate(
                        action_data,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                    for action_data in obj.get("actions", [])
                ]
            else:
                logger.info(f"No repository provided, skip initiating actions")
                obj["actions"] = []

            if agent_class_path:
                module_name, class_name = agent_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                instance = agent_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: Any | None = None,
    ) -> "ActionAgent":
        """Create an ActionAgent from a dictionary, properly handling dependencies."""
        data = data.copy()

        # Handle completion model
        if "completion" in data and isinstance(data["completion"], dict):
            data["completion"] = BaseCompletionModel.model_validate(data["completion"])

        # Handle actions with dependencies
        if repository and "actions" in data and isinstance(data["actions"], list):
            data["actions"] = [
                Action.model_validate(
                    action_data,
                    repository=repository,
                    runtime=runtime,
                    code_index=code_index,
                )
                for action_data in data["actions"]
            ]

        # Handle message generator
        if "message_generator" in data and isinstance(data["message_generator"], dict):
            data["message_generator"] = MessageHistoryGenerator.model_validate(data["message_generator"])

        # Handle agent class if specified
        if "agent_class" in data:
            module_name, class_name = data["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            return agent_class(**data)

        return cls.model_validate(data)

    @property
    def completion(self) -> BaseCompletionModel:
        return self._completion

    @completion.setter
    def completion(self, value: BaseCompletionModel):
        self._completion = value
