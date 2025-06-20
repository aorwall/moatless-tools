from datetime import datetime
import logging
import traceback
from collections.abc import Callable
from typing import Any, Awaitable, Optional, cast

from opentelemetry import trace
from pydantic import Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
)
from moatless.agent.events import (
    ActionCreatedEvent,
    ActionExecutedEvent,
    AgentErrorEvent,
    AgentEvent,
    RunAgentEvent,
)
from moatless.completion import BaseCompletionModel
from moatless.completion.schema import ResponseSchema
from moatless.component import MoatlessComponent
from moatless.context_data import current_action_step
from moatless.events import BaseEvent
from moatless.exceptions import (
    CompletionError,
    CompletionRejectError,
    RejectError,
    RuntimeError,
)
from moatless.message_history.base import BaseMemory
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import ActionStep, Node, Thoughts
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.agent")


class ActionAgent(MoatlessComponent):
    agent_id: Optional[str] = Field(None, description="Agent ID")
    model_id: Optional[str] = Field(None, description="Model ID")
    description: Optional[str] = Field(None, description="Description of the agent")

    completion_model: Optional[BaseCompletionModel] = Field(
        None, description="Completion model to be used for generating completions"
    )

    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: list[Action] = Field(default_factory=list)
    memory: BaseMemory = Field(
        default_factory=MessageHistoryGenerator,
        description="Message history generator to be used for generating completions",
    )
    shadow_mode: bool = Field(True, description="Set to true to not persist changes done by the agent.")

    _action_map: dict[type[ActionArguments], Action] = PrivateAttr(default_factory=dict)
    _workspace: Workspace | None = PrivateAttr(default=None)
    _on_event: Optional[Callable[[BaseEvent], Awaitable[None]]] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        action_args: list[type[ActionArguments]] = []
        for action in self.actions:
            self._action_map[action.args_schema] = action

            if not action.hidden:
                action_args.append(action.args_schema)

        if not action_args:
            raise RuntimeError("No actions found")

        if self.completion_model:
            # Cast action_args to the expected type since ActionArguments inherits from ResponseSchema
            response_schemas = cast(list[type[ResponseSchema]], action_args)
            self.completion_model.initialize(response_schemas, self.system_prompt)

    async def run_simple(self, user_message: str) -> Observation | None:
        node = Node.create(user_message)
        child_node = node.create_child()
        await self.run(child_node)
        return child_node.observation

    @tracer.start_as_current_span("ActionAgent.run")
    async def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""
        if not self.completion_model:
            raise RuntimeError("Completion model not set")

        if node.is_executed():
            raise RuntimeError("Node already executed")

        try:
            if not node.action_steps:
                await self._generate_actions(node)

                duplicate_node = node.find_node_with_same_action_steps()
                if duplicate_node:
                    node.is_duplicate = True
                    logger.info(f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping")
                    return

            else:
                logger.info(f"Node{node.node_id}: Action steps already generated. Skipping action generation.")

            if not node.action_steps:
                logger.warning(f"Node{node.node_id}: No action steps generated. Skipping execution.")
                return

            action_names = [action_step.action.name for action_step in node.action_steps]
            logger.info(f"Node{node.node_id}: Execute actions: {action_names}")
            for i, action_step in enumerate(node.action_steps):
                current_action_step.set(i)  # Used in logging to identify the action step
                await self._execute(node, action_step)
                current_action_step.set(None)
                if node.terminal:
                    break

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Failed to run.")
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            raise e

    async def _generate_actions(self, node: Node):
        node.possible_actions = [action.name for action in self.actions]

        # TODO: This is a hack to get the file context to work with the existing repository, remove when we got rid of the legacy file_context..
        if node.file_context and not node.file_context._repo and self._workspace:
            node.file_context._repo = self._workspace.repository

        try:
            await self._emit_event(RunAgentEvent(agent_id=self.agent_id, node_id=node.node_id))
            messages = await self.memory.generate_messages(node, self._workspace)
            logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")

            completion_response = await self.completion_model.create_completion(messages=messages)
            if completion_response.completion_invocation:
                node.completions["build_action"] = completion_response.completion_invocation

            if completion_response.thought:
                node.thoughts = Thoughts(text=completion_response.thought)

            node.assistant_message = completion_response.text_response
            if not completion_response.structured_outputs:
                raise RejectError("No action found")

            node.thinking_blocks = completion_response.thinking_blocks

            if completion_response.structured_outputs:
                node.action_steps = [
                    ActionStep(action=action)
                    for action in completion_response.structured_outputs
                    if isinstance(action, ActionArguments)
                ]
                for step in node.action_steps:
                    await self._emit_event(
                        ActionCreatedEvent(
                            agent_id=self.agent_id,
                            node_id=node.node_id,
                            action_name=step.action.name,
                        )
                    )

        except CompletionError as e:

            if e.completion_invocation:
                logger.error(f"Node{node.node_id}: Build action failed with completion error: {e}")
                node.completions["build_action"] = e.completion_invocation
            else:
                logger.exception(f"Node{node.node_id}: Build action failed with error ")

            if isinstance(e, CompletionRejectError):
                node.error = f"Completion validation error: {str(e)}\n\n{traceback.format_exc()}"
                await self._emit_event(
                    AgentErrorEvent(
                        agent_id=self.agent_id,
                        node_id=node.node_id,
                        error=f"Completion validation error: {e.message}",
                    )
                )
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                await self._emit_event(
                    AgentErrorEvent(
                        agent_id=self.agent_id,
                        node_id=node.node_id,
                        error=f"{e.__class__.__name__}: {str(e)}",
                    )
                )
                raise e
        except Exception as e:
            node.terminal = True
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"Node{node.node_id}: Build action failed with error: {e}. Type {type(e)}")
            await self._emit_event(
                AgentErrorEvent(
                    agent_id=self.agent_id,
                    node_id=node.node_id,
                    error=f"{e.__class__.__name__}: {str(e)}",
                )
            )

            raise e

    @tracer.start_as_current_span("ActionAgent._execute_action_step")
    async def _execute_action_step(self, node: Node, action_step: ActionStep) -> Observation:
        action = self.action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {type(action_step.action)} not found in action map. "
                f"Available actions: {list(self.action_map.keys())}"
            )
            raise RuntimeError(
                f"Action {type(action_step.action)} not found in action map with actions: {list(self.action_map.keys())}"
            )

        observation = await action.execute(action_step.action, file_context=node.file_context)
        if action.is_terminal:
            node.terminal = True
        return observation

    @tracer.start_as_current_span("ActionAgent._execute")
    async def _execute(self, node: Node, action_step: ActionStep):
        try:
            action_step.start_time = datetime.now()
            previous_context = node.file_context.clone() if node.file_context else None
            action_step.observation = await self._execute_action_step(node, action_step)

            if previous_context and node.file_context:
                action_step.observation.artifact_changes = node.file_context.get_artifact_changes(previous_context)

            await self._emit_event(
                ActionExecutedEvent(
                    agent_id=self.agent_id,
                    node_id=node.node_id,
                    action_name=action_step.action.name,
                )
            )

            if not action_step.observation:
                logger.warning(f"Node{node.node_id}: Action {action_step.action.name} returned no observation")
            else:
                if action_step.observation.terminal:
                    node.terminal = True
                if action_step.observation.execution_completion:
                    action_step.completion = action_step.observation.execution_completion

            action_step.end_time = datetime.now()

            logger.info(
                f"Node{node.node_id}: Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. "
            )

            logger.debug(
                f"Node{node.node_id}: Observation: {action_step.observation.message if node.observation else None}"
            )

        except CompletionError as e:
            if e.completion_invocation:
                action_step.completion = e.completion_invocation

            node.terminal = True

            if isinstance(e, CompletionRejectError):
                node.error = f"Completion validation error: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
                # action_step.observation.error = f"Completion validation error: {e.message}"
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                # action_step.observation.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
                raise e

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Execution of action {action_step.action.name} failed.")
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
            # action_step.observation.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            node.terminal = True
            raise e

    @classmethod
    def get_component_type(cls) -> str:
        return "agent"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.agent"

    @classmethod
    def _get_base_class(cls) -> type:
        return ActionAgent

    @property
    def action_map(self):
        return self._action_map

    @property
    def workspace(self) -> Workspace:
        if not self._workspace:
            raise RuntimeError("Workspace not set")
        return self._workspace

    # TODO: Replace this with initialize method
    @workspace.setter
    def workspace(self, workspace: Workspace):
        self._workspace = workspace
        for action in self.actions:
            action.workspace = workspace

    async def initialize(self, workspace: Workspace):
        if not self.completion_model:
            raise RuntimeError("Completion model not set")

        self._workspace = workspace
        
        try:
            self.workspace.repository.shadow_mode = self.shadow_mode
        except ValueError:
            # Repository not set, skip setting shadow_mode
            pass
            
        for action in self.actions:
            await action.initialize(workspace)

    async def _emit_event(self, event: AgentEvent):
        """Emit a pure agent event"""
        if self._on_event:
            await self._on_event(event)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        dump = super().model_dump(**kwargs)

        dump["actions"] = []
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))

        dump["memory"] = self.memory.model_dump(**kwargs)

        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump(**kwargs)

        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()

            obj["actions"] = [Action.from_dict(action_data) for action_data in obj.get("actions", [])]

            if "memory" in obj:
                obj["memory"] = BaseMemory.from_dict(obj["memory"])
            else:
                obj["memory"] = MessageHistoryGenerator()

            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj["completion_model"])

            instance = super().model_validate(obj)

            for action in instance.actions:
                if not isinstance(action, Action):
                    raise ValueError(f"Invalid action type: {type(action)}. Expected Action subclass.")
                if not hasattr(action, "args_schema"):
                    raise ValueError(f"Action {action.__class__.__name__} has no args_schema attribute")
                instance._action_map[action.args_schema] = action

            return instance

        return obj
