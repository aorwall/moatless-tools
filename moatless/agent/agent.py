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
    Observation,
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
from moatless.component import MoatlessComponent
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

from moatless.completion.manager import create_completion_model

logger = logging.getLogger(__name__)


class ActionAgent(MoatlessComponent):
    agent_id: str = Field(..., description="Agent ID")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: List[Action] = Field(default_factory=list)

    _completion_model: BaseCompletionModel | None = PrivateAttr(default=None)
    _message_generator: MessageHistoryGenerator | None = PrivateAttr(default=None)
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)
    _workspace: Workspace | None = PrivateAttr(default=None)

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
            self._workspace = None
        
        self._message_generator = MessageHistoryGenerator()
        self.completion_model = completion_model

    @classmethod
    def from_agent_settings(cls, agent_settings: AgentSettings, actions: List[Action] | None = None):
        if agent_settings.actions:
            actions = [action for action in actions if action.__class__.__name__ in agent_settings.actions]

        return cls(
            completion_model=agent_settings.completion_model,
            system_prompt=agent_settings.system_prompt,
            actions=actions,
        )
    
    @classmethod
    def get_component_type(cls) -> str:
        return "agent"
    
    @classmethod
    def _get_package(cls) -> str:
        return "moatless.agent"

    @classmethod
    def _get_base_class(cls) -> Type:
        return ActionAgent
    
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
        
            self._completion_model.initialize(action_args, self.system_prompt)

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
            messages = await self._message_generator.generate_messages(node)
            logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")

            completion_response = await self._completion_model.create_completion(
                messages=messages
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
                node.error = f"Completion validation error: {e.message}"
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                raise e
        except Exception as e:
            node.terminal = True
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
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

    async def _execute_action_step(self, node: Node, action_step: ActionStep) -> Observation:
        action = self.action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self.action_map.keys()}"
            )
            raise RuntimeError(
                f"Action {type(node.action)} not found in action map with actions: {self.action_map.keys()}"
            )

        return await action.execute(action_step.action, file_context=node.file_context)

    async def _execute(self, node: Node, action_step: ActionStep):
        try:
            action_step.observation = await self._execute_action_step(node, action_step)

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

        except CompletionError as e:
            if e.last_completion:
                action_step.completion = Completion.from_llm_completion(
                    input_messages=e.messages,
                    completion_response=e.last_completion,
                    model=self.completion_model.model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )

            node.terminal = True

            if isinstance(e, CompletionRejectError):
                node.error = f"Completion validation error: {e.message}"
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                raise e
        
        except Exception as e:
            logger.exception(f"Node{node.node_id}: Execution of action {action_step.action.name} failed.")
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            node.terminal = True
            raise e


    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)

        dump["actions"] = []
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))

        if self.completion_model:
            dump["model_id"] = self.completion_model.model_id
    
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()

            obj["actions"] = [Action.model_validate(action_data) for action_data in obj.get("actions", [])]
            
            if "model_id" in obj:
                obj["completion_model"] = create_completion_model(obj["model_id"])

            return super().model_validate(obj)

        return obj

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
