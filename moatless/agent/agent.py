import importlib
import json
import logging
import traceback
from typing import List, Type, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from moatless.actions.action import Action
from moatless.actions.action import CompletionModelMixin
from moatless.actions.schema import (
    ActionArguments,
)
from moatless.agent.events import (
    AgentEvent,
    AgentStarted,
    AgentActionCreated,
    AgentActionExecuted,
)
from moatless.agent.settings import AgentSettings
from moatless.artifacts.artifact import ArtifactHandler
from moatless.completion import BaseCompletionModel, LLMResponseFormat
from moatless.completion.model import Completion
from moatless.events import event_bus
from moatless.exceptions import (
    CompletionError,
    RejectError,
    RuntimeError,
    CompletionRejectError,
)
from moatless.index.code_index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class ActionAgent(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: List[Action] = Field(default_factory=list)

    _completion_model: BaseCompletionModel | None = PrivateAttr()
    _message_generator: MessageHistoryGenerator | None = PrivateAttr(default_factory=MessageHistoryGenerator)
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)
    _workspace: Workspace | None = PrivateAttr()

    def __init__(
        self,
        agent_id: str,
        actions: List[Action],
        system_prompt: str,
        completion_model: BaseCompletionModel | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: Any | None = None,
        workspace: Workspace | None = None,
        artifact_handlers: List[ArtifactHandler] | None = None,
        **data,
    ):
        super().__init__(
            system_prompt=system_prompt,
            actions=actions,
            agent_id=agent_id,
            **data,
        )
        
        self.completion_model = completion_model

        if workspace:
            self.workspace = workspace
        elif repository and runtime and code_index and artifact_handlers:
            self.workspace = Workspace(
                repository=repository,
                runtime=runtime,
                code_index=code_index,
                artifact_handlers=artifact_handlers,
            )
        else:
            self.workspace = None

    @classmethod
    def from_agent_settings(cls, agent_settings: AgentSettings, actions: List[Action] | None = None):
        if agent_settings.actions:
            actions = [action for action in actions if action.__class__.__name__ in agent_settings.actions]

        return cls(
            completion_model=agent_settings.completion_model,
            system_prompt=agent_settings.system_prompt,
            actions=actions,
        )

    @property
    def completion_model(self):
        return self._completion_model

    @property
    def action_map(self):
        if not self._action_map:
            self._action_map = {action.args_schema: action for action in self.actions}
        return self._action_map

    @completion_model.setter
    def completion_model(self, completion_model: BaseCompletionModel | None):
        """Set completion model on agent and all actions that support it"""
        if completion_model is None:
            self._completion_model = None
            for action in self.actions:
                if isinstance(action, CompletionModelMixin):
                    action.completion_model = None
        else:
            self._completion_model = completion_model.clone()
            self._message_generator = MessageHistoryGenerator.create(completion_model.message_history_type)
            if self._workspace:
                self._message_generator.workspace = self._workspace

            action_args = [action.args_schema for action in self.actions]
            if not action_args:
                raise RuntimeError("No actions found")
        
            system_prompt = self.generate_system_prompt()
            self._completion_model.initialize(action_args, system_prompt)

            for action in self.actions:
                if hasattr(action, "completion_model"):
                    action.completion_model = self._completion_model

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: Workspace):
        self._workspace = workspace
        if self._message_generator:
            self._message_generator.workspace = workspace
        for action in self.actions:
            action.workspace = workspace

    async def _emit_event(self, event: AgentEvent):
        """Emit a pure agent event"""
        await event_bus.publish(event)

    async def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""
        if not self._completion_model:
            raise RuntimeError("Completion model not set")

        if node.is_executed():
            raise RuntimeError("Node already executed")

        node.possible_actions = [action.name for action in self.actions]

        try:
            await self._emit_event(AgentStarted(agent_id=self.agent_id, node_id=node.node_id))
            if self._message_generator:
                messages = self._message_generator.generate_messages(node)
                logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")

            completion_response = await self._completion_model.create_completion(
                messages=messages,
            )
            node.completions["build_action"] = completion_response.completion

            node.assistant_message = completion_response.text_response
            if not completion_response.structured_outputs:
                raise RejectError("No action found")

            if completion_response.structured_outputs:
                node.action_steps = [ActionStep(action=action) for action in completion_response.structured_outputs]
                # Emit action created events
                for step in node.action_steps:
                    await self._emit_event(
                        AgentActionCreated(
                            agent_id=self.agent_id,
                            node_id=node.node_id,
                            action_name=step.action.name,
                        )
                    )

        except CompletionError as e:
            node.terminal = True
            node.error = traceback.format_exc()
            if e.last_completion:
                logger.error(f"Node{node.node_id}: Build action failed with completion error: {e}")

                node.completions["build_action"] = Completion.from_llm_completion(
                    input_messages=e.messages if hasattr(e, "messages") else [],
                    completion_response=e.last_completion,
                    model=self.completion_model.model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )
            else:
                logger.exception(f"Node{node.node_id}: Build action failed with error ")

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
            await self._execute(node, action_step)

    async def _execute(self, node: Node, action_step: ActionStep):
        action = self.action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self.action_map.keys()}"
            )
            raise RuntimeError(
                f"Action {type(node.action)} not found in action map with actions: {self.action_map.keys()}"
            )

        try:
            action_step.observation = await action.execute(action_step.action, file_context=node.file_context)

            await self._emit_event(
                AgentActionExecuted(
                    agent_id=self.agent_id,
                    node_id=node.node_id,
                    action_name=action_step.action.name,
                )
            )

            if not action_step.observation:
                logger.warning(f"Node{node.node_id}: Action {action_step.action.name} returned no observation")
            else:
                node.terminal = action_step.observation.terminal
                if action_step.observation.execution_completion:
                    action_step.completion = action_step.observation.execution_completion

            logger.info(
                f"Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. ")
            
            logger.debug(
                f"Observation: {action_step.observation.message if node.observation else None}"
            )

        except CompletionRejectError as e:
            logger.warning(f"Node{node.node_id}: Action rejected: {e.message}")
            if e.last_completion:
                action_step.completion = Completion.from_llm_completion(
                    input_messages=e.messages,
                    completion_response=e.last_completion,
                    model=self.completion_model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )
            node.error = traceback.format_exc()
            node.terminal = True
            raise e
        
        except Exception as e:
            logger.exception(f"Node{node.node_id}: Execution of action {action_step.action.name} failed.")
            node.error = traceback.format_exc()
            node.terminal = True
            raise e

    def generate_system_prompt(self) -> str:
        """Generate a system prompt for the agent."""

        system_prompt = self.system_prompt
        if self._completion_model and self._completion_model.use_few_shots:
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
                if self.completion_model.response_format == LLMResponseFormat.REACT:
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
                        prompt += f"\nTask: {example.user_input}\n"
                        if not self.disable_thoughts:
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

                elif self.completion_model.response_format == LLMResponseFormat.JSON:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name,
                    }
                    prompt += (
                        f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"
                    )

                elif self.completion_model.response_format == LLMResponseFormat.TOOLS:
                    tools_json = {"tool": example.action.name}
                    if self.disable_thoughts:
                        tools_json.update(example.action.model_dump(exclude={"thoughts"}))
                    else:
                        tools_json.update(example.action.model_dump())

                    prompt += f"Task: {example.user_input}\n"
                    if not self.disable_thoughts:
                        prompt += f"<thoughts>{example.action.thoughts}</thoughts>\n"
                    prompt += json.dumps(tools_json)
                    prompt += "\n\n"

        return prompt


    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["actions"] = []
        dump["agent_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()
            agent_class_path = obj.pop("agent_class", None)

            logger.info(f"Validating agent with actions: {obj.get('actions', [])}")

            obj["actions"] = [Action.model_validate(action_data) for action_data in obj.get("actions", [])]
            if agent_class_path:
                module_name, class_name = agent_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                instance = agent_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    @model_validator(mode="after")
    def verify_actions(self) -> "ActionAgent":
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValueError(f"Invalid action type: {type(action)}. Expected Action subclass.")
            if not hasattr(action, "args_schema"):
                raise ValueError(f"Action {action.__class__.__name__} is missing args_schema attribute")
        return self

    def execute_action(self, action: Action) -> Dict:
        """Execute an action and return the observation."""
        observation = super().execute_action(action)

        self._emit_event(
            AgentActionExecuted(
                agent_id=self.agent_id,
                node_id=self.current_node.node_id,
                action_name=action.name,
                observation=observation,
            )
        )

        return observation
