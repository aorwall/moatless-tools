import sys
import json
import logging
from typing import Optional, Any, List, Dict

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from moatless.state import AgenticState
from moatless.types import ActionRequest

from moatless.state import AgenticState, Finished, Rejected, Pending, NoopState
from moatless.edit.clarify import ClarifyCodeChange, LineNumberClarification
from moatless.edit.edit import EditCode, CodeChange
from moatless.edit.plan import PlanToCode, TakeAction
from moatless.find.identify import IdentifyCode, Identify
from moatless.find.search import SearchCode, Search

from moatless.codeblocks import CodeBlockType
from moatless.file_context import FileContext, RankedFileSpan
from moatless.state import AgenticState


STATE_CLASS_MAP = {
    'ClarifyCodeChange': ClarifyCodeChange,
    'EditCode': EditCode,
    'PlanToCode': PlanToCode,
    'IdentifyCode': IdentifyCode,
    'SearchCode': SearchCode,
    "NoopState": NoopState,
    "Finished": Finished,
    "Rejected": Rejected,
    "Pending": Pending,
}

def force_set_attributes(obj: BaseModel, data: Dict[str, Any]):
    for key, value in data.items():
        if hasattr(obj, key):
            if key == 'ranked_spans' and isinstance(value, list):
                # Convert dict to RankedFileSpan if necessary
                setattr(obj, key, [
                    RankedFileSpan(**span) if isinstance(span, dict) else span
                    for span in value
                ])
            else:
                setattr(obj, key, value)
        elif key.startswith('_'):
            # Handle private attributes
            object.__setattr__(obj, key, value)
            
def trajectory_from_dict(data: Dict[str, Any], loop):
    trajectory = Trajectory(
        name=data['name'],
        initial_message=data.get('initial_message'),
        persist_path=None  # We don't have this information in the dict
    )
    trajectory.save_info(data.get('info', {}))
    print(f"traj data: {data}")
    transitions = []
    
    for transition_data in data.get('transitions', []):
        state_data = transition_data.get('state', {})
        state_class_name = transition_data.get('name')
        state_class = STATE_CLASS_MAP.get(state_class_name)
        
        if state_class and issubclass(state_class, AgenticState):
            # Remove some attrs
            attrs_to_remove = ['include_message_history']
            for attr in attrs_to_remove:
                state_data.pop(attr, None)
            print(f"state_data: {state_data}")
            
            # Check if the state_class is a Pydantic BaseModel
            if issubclass(state_class, BaseModel):
                state = state_class.model_validate(state_data)
            else:
                # Initialize the state with only the fields defined in the class
                state = state_class(**state_data)
                # Now forcefully set all attributes, including those not defined in the class
                force_set_attributes(state, state_data)
            
            # Set loop
            state._set_loop(loop)
            print(f"state_attrs: {state.__dict__}")
        else:
            raise NotImplementedError(f"State class {state_class_name} not found")
        
        transition = TrajectoryTransition(state=state).model_validate(transition_data)
        
        for action_data in transition_data.get('actions', []):
            # Reconstruct the ActionRequest object
            action_request_data = action_data['action']
            print(f"action_request_data: {action_request_data}")
            action_request = ActionRequest.model_validate(action_request_data)
            action = TrajectoryAction.model_validate(action_request_data)
            print(f"action: {action}")
            transition.actions.append(action)
        
        transitions.append(transition)
    
    trajectory.set_transitions(transitions)
    return trajectory


STATE_CLASS_MAP = {
    'ClarifyCodeChange': ClarifyCodeChange,
    'EditCode': EditCode,
    'PlanToCode': PlanToCode,
    'IdentifyCode': IdentifyCode,
    'SearchCode': SearchCode,
    "NoopState": NoopState,
    "Finished": Finished,
    "Rejected": Rejected,
    "Pending": Pending,
}


import importlib


def get_concrete_state_class(state_name: str):
    return STATE_CLASS_MAP.get(state_name, AgenticState)

def get_class(class_name: str):
    # Check if the class is in the global namespace
    if class_name in globals():
        return globals()[class_name]
    
    # If not found in globals, check in sys.modules
    for module in sys.modules.values():
        if hasattr(module, class_name):
            return getattr(module, class_name)
    
    raise ValueError(f"Class {class_name} not found in the global namespace or imported modules")


def reconstruct_state(transition: Dict[str, Any], loop=None):
    state_class_name = transition.get('name', 'AgenticState')
    state_data = transition['state']
    
    attrs_to_remove = ['include_message_history']
    for attr in attrs_to_remove:
        state_data.pop(attr, None)

    state_class = get_concrete_state_class(state_class_name)
    if state_class is AgenticState:
        print(f"Warning: Unable to find concrete class for '{state_class_name}'. Using default AgenticState.")

    try:
        state = state_class.model_validate(state_data)
    except TypeError as e:
        print(f"TypeError encountered: {e}")
        state = state_class(**state_data)

    return state


def trajectory_from_dict(data: Dict[str, Any], loop=None):
    # Step 1: Extract basic information
    name = data['name']
    initial_message = data.get('initial_message')
    
    # Step 2: Create a new Trajectory instance
    trajectory = Trajectory(name, initial_message)
    
    # Step 3: Reconstruct transitions
    transitions = []
    for transition_data in data['transitions']:
        state = reconstruct_state(transition_data)
        
        if loop is not None:
            state._set_loop(loop)  # Set the loop for the state
        
        # Step 3b: Create a new TrajectoryTransition
        transition = TrajectoryTransition(state=state)
        
        # Step 3c: Reconstruct actions
        for action_data in transition_data['actions']:
            # Get the action class dynamically
            action_class_name = action_data['action'].get('__class__', 'ActionRequest')
            action_class = get_class(action_class_name)
            
            # Reconstruct the action
            action = action_class.model_validate(action_data['action'])
            
            # Create the TrajectoryAction
            trajectory_action = TrajectoryAction(
                action=action,
                retry_message=action_data.get('retry_message'),
                output=action_data.get('output'),
                completion_cost=action_data.get('completion_cost')
            )
            transition.actions.append(trajectory_action)
        
        transitions.append(transition)
    
    # Step 4: Set the reconstructed transitions
    trajectory.set_transitions(transitions)
    
    # Step 5: Set the info dictionary
    trajectory.save_info(data.get('info', {}))
    
    return trajectory


logger = logging.getLogger(__name__)


class TrajectoryAction(BaseModel):
    action: ActionRequest
    retry_message: Optional[str] = None
    output: Optional[dict[str, Any]] = None
    completion_cost: Optional[float] = None

    def model_dump(self, **kwargs):
        dict = super().model_dump(**kwargs)
        action_dict = self.action.model_dump(**kwargs)
        action_dict['__class__'] = self.action.__class__.__name__
        dict["action"] = action_dict
        return dict


class TrajectoryTransition(BaseModel):
    state: Optional[AgenticState] = None
    actions: List[TrajectoryAction] = []

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        return {
            "name": self.name,
            "state": self.state.model_dump(**kwargs) if self.state else None,
            "actions": [action.model_dump(**kwargs) for action in self.actions],
            "file_context": str(self.state.loop._workspace.file_context.create_prompt())
        }


class Trajectory:

    def __init__(
        self,
        name: str,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message

        self._transitions: list[TrajectoryTransition] = []
        self._current_transition: Optional[TrajectoryTransition] = None

        self._info: dict[str, Any] = {}

    @property
    def current_step(self):
        return self._current_transition

    @property
    def initial_message(self):
        return self._initial_message

    def get_transitions(self, name: str):
        return [
            transition for transition in self._transitions if transition.name == name
        ]
    
    def set_transitions(self, transitions: list[TrajectoryTransition]):
        self._transitions = transitions

    def transition_count(self, state: AgenticState):
        return len(self.get_transitions(state.name))

    def create_action_dict(
        self,
        action: ActionRequest,
        output: Optional[dict[str, Any]] = None,
        retry_message: Optional[str] = None,
        completion_cost: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        if self._current_transition:
            action_dict = TrajectoryAction(
                action=action,
                output=output,
                retry_message=retry_message,
                completion_cost=completion_cost,
            ).model_dump(exclude_none=True)

            logger.info(
                f"Creating action dict for {action.__class__.__name__} to {self._current_transition.name} ({len(self._current_transition.actions) + 1} actions)"
            )

            return action_dict
        else:
            logger.warning(
                f"No current trajectory step to create action dict for {action.model_dump_json()}."
            )
            return None

    def save_action(
        self,
        action: ActionRequest,
        output: Optional[dict[str, Any]] = None,
        retry_message: Optional[str] = None,
        completion_cost: Optional[float] = None,
    ):
        if self._current_transition:
            self._current_transition.actions.append(
                TrajectoryAction(
                    action=action,
                    output=output,
                    retry_message=retry_message,
                    completion_cost=completion_cost,
                )
            )
            logger.info(
                f"Saving action {action.__class__.__name__} to {self._current_transition.name} ({len(self._current_transition.actions)} actions)"
            )

            self._maybe_persist()
        else:
            logger.warning(
                f"No current trajectory step to save action {action.model_zdump_json()}."
            )

    def new_transition(self, state: AgenticState):
        if self._current_transition:
            self._transitions.append(self._current_transition)

        self._current_transition = TrajectoryTransition(state=state)
        self._maybe_persist()

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def to_dict(self, **kwargs):
        transition_dicts = [
            transition.model_dump(**kwargs) for transition in self._transitions
        ]
        if self._current_transition:
            transition_dicts.append(self._current_transition.model_dump(**kwargs))
        
        return {
        "name": self._name,
        "initial_message": self._initial_message,
        "transitions": transition_dicts,
        "info": self._info,
        }

    def total_cost(self):
        total_cost = 0
        for step in self._transitions:
            for action in step.actions:
                if action.completion_cost:
                    total_cost += action.completion_cost

        return total_cost

    def _maybe_persist(self):
        if self._persist_path:
            self.persist(self._persist_path)

    def persist(self, file_path: str):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(exclude_none=True),
                    indent=2,
                    default=to_jsonable_python,
                )
            )
