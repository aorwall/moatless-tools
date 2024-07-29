import logging
import random
import string
from typing import Optional, Type, Any, List, Tuple, Callable

import instructor
import litellm
from litellm import token_counter, completion_cost, ModelResponse
from pydantic import BaseModel, Field

from moatless import Workspace
from moatless.state import (
    AgenticState,
    NoopState,
    Finished,
    Rejected,
    Pending,
)
from moatless.types import Response, Message, AssistantMessage, UserMessage
from moatless.trajectory import Trajectory, TrajectoryTransition, trajectory_from_dict, reconstruct_state
from moatless.types import (
    ActionRequest,
    Content,
)

# from .loop import AgenticLoop as DefaultLoop
from .benchmark.swebench.utils import create_workspace, load_instance
from moatless.benchmark.utils import get_file_spans_from_patch

from moatless.repository import FileRepository
from moatless.file_context import FileContext
from .utils_search.deepcopy import safe_deepcopy, custom_deepcopy
from copy import deepcopy, copy
import inspect
from typing import Any

import traceback
import subprocess

import json
import os


# logger = logging.getLogger("Loop")
logger = logging.getLogger(__name__)

# log all info
# logging.basicConfig(level=logging.INFO)


def save_json_dict(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def get_actions(trajectory: dict):
    actions = []
    for transition in trajectory["transitions"]:
        state_name = transition["name"]
        print(f"state_name: {state_name}, actions: {transition['actions']}")
        if state_name in ['Finished', 'Rejected']:
            transition['actions'] = []
        for action in transition["actions"]:
            actions.append(action["action"])
    return actions

def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string


class Transition(BaseModel):
    trigger: str
    source: Type[AgenticState]
    dest: Type[AgenticState]
    required_fields: set[str] = Field(default_factory=set)

class Transitions:

    def __init__(
        self,
        initial_state: Type[AgenticState],
        transitions: List[Transition],
        global_params: Optional[dict[str, Any]] = None,
        state_params: Optional[dict[Type[AgenticState], dict[str, Any]]] = None,
    ):
        self._initial_state = initial_state
        self._global_params = global_params or {}
        self._state_params = state_params or {}
        self._source_trigger_index: dict[tuple[Type[AgenticState], str], list] = {}

        for transition in transitions:
            if (
                transition.source,
                transition.trigger,
            ) not in self._source_trigger_index:
                self._source_trigger_index[(transition.source, transition.trigger)] = []
            self._source_trigger_index[(transition.source, transition.trigger)].append(
                transition
            )

    def find_transition_by_source_and_trigger(
        self, source: Type[AgenticState], trigger: str
    ) -> List[Transition]:
        return self._source_trigger_index.get((source, trigger), [])

    def initial_state(self, **data) -> AgenticState:
        return self._initial_state(**self._global_params, **data)

    def next_state(
        self, source: AgenticState, trigger: str, data: dict[str, Any]
    ) -> Optional[AgenticState]:
        transitions = self.find_transition_by_source_and_trigger(
            source.__class__, trigger
        )
        for transition in transitions:
            if transition.required_fields.issubset(data.keys()):
                params = {}
                params.update(self._global_params)
                params.update(self._state_params.get(transition.dest, {}))
                return transition.dest(**params, **data)
        return None

def reinitialize_agent(transitions, workspace,
                       trajectory_path,
                       metadata, trajectory, state):
    """
    Reinitialize the agent.
    """
    loop = AgenticLoop(transitions=transitions, 
                       workspace=workspace, 
                       metadata=metadata, 
                       trajectory_path=trajectory_path, 
                       max_cost=0.5)
    loop.transition_to(state)
    return loop

class AgenticLoop:

    def __init__(
        self,
        transitions: Transitions,
        workspace: Workspace,
        mocked_actions: Optional[List[dict]] = None,
        verify_state_func: Optional[Callable] = None,
        max_cost: float = 0.25,
        max_transitions: int = 25,
        max_message_tokens: int = 16000,
        max_retries: int = 2,
        max_rejections: int = 2,
        metadata: Optional[dict[str, Any]] = None,
        trajectory_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Loop instance.

        Args:

        """

        self._workspace = workspace
        self._trajectory_path = trajectory_path

        self._mocked_actions = mocked_actions
        self._verify_state_func = verify_state_func

        self._max_cost = max_cost
        self._max_message_tokens = max_message_tokens
        self._max_transitions = max_transitions
        self._max_retries = max_retries
        self._max_rejections = max_rejections

        self._transition_count = 0
        self._rejections = 0

        self._transitions = transitions
        
        self._initial_message = ""
        self._state: AgenticState = Pending()

        self._metadata = metadata

        # MCTS
        self.mcts: Optional[MCTS] = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def run(self, message: Optional[str] = None, input_data: Optional[dict[str, Any]] = None) -> Response:
        """
        Run the loop and handle exceptions and cost checking.
        """

        if self.is_running():
            raise Exception("Loop is already running.")

        self._trajectory = Trajectory(
            "AgenticLoop", initial_message=message, persist_path=self._trajectory_path
        )

        self.transition_to(self._transitions.initial_state(**input_data or {}))

        while self.is_running():
            try:
                self._run()
            except Exception as e:
                logger.warning(f"Failed to run loop. Error: {e}")
                raise

            if self.retries() > self._max_retries:
                logger.warning(f"Max retries reached ({self._max_retries}). Exiting.")
                self.trajectory.save_info({"error": "Max retries reached."})
                return Response(
                    status="rejected",
                    message="The loop was aborted because the number of retries exceeded the limit.",
                )

            total_cost = self._trajectory.total_cost()
            if total_cost > self._max_cost:
                logger.warning(f"Max cost reached ({total_cost} > {self._max_cost}). Exiting.")
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError(
                    "The loop was aborted because the cost exceeded the limit.",
                )

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message)
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message)

        raise RuntimeError(f"Loop exited with unknown state {self.state}.")
        
    def get_git_diff(self):
        self._workspace.save()
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Repository directory: {self._workspace.repo_dir}")
        
        # Check git status
        status_output = subprocess.run(
            ["git", "status"], capture_output=True, text=True, cwd=self._workspace.repo_dir
        )
        logging.info(f"Git status:\n{status_output.stdout}")

        # Run git diff
        output = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, cwd=self._workspace.repo_dir
        )
        logging.info(f"Git diff output:\n{output.stdout}")
        
        return output
        # except Exception as e:
        #     logger.error(f"Error getting git diff: {e}")
        #     return ""
    
    def _rerun_optimal_trajectory(self, best_trajectory: List[Tuple[ActionRequest, AgenticState]]):
        actions = get_actions(best_trajectory)
        
        trajectory = trajectory_from_dict(best_trajectory)
        
        workspace = Workspace.from_dirs(
            repo_dir=self._workspace.repo_dir,
            index_dir=self._workspace.index_dir
        
        )
        
        loop = AgenticLoop(
            transitions=self._transitions,
            workspace=workspace,
            mocked_actions=actions,
        )
        loop.run(message=self._initial_message)
        
        # Return the final state and trajectory
        return loop.state, loop._trajectory
        

    def run_search(self, message: Optional[str] = None, input_data: Optional[dict[str, Any]] = None) -> Response:
        if self.is_running():
            raise Exception("Loop is already running.")

        self._trajectory = Trajectory(
            "AgenticLoop", initial_message=message, persist_path=self._trajectory_path
        )

        initial_state = self._transitions.initial_state(**input_data or {})
        self.transition_to(initial_state)

        # Initialize MCTS here
        self.mcts = MCTS(initial_state, self, max_actions=self.max_actions)

        best_trajectory, final_step, total_reward = self._run_mcts()

        # Re-initialize the loop and run the optimal trajectory
        # self._rerun_optimal_trajectory(best_trajectory)
        
        # Evaluate the final state
        final_state = self.state
        
        return best_trajectory, final_step.git_diff

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def _set_state_loop(self, state: AgenticState):
        state._set_loop(self)

    def retries(self) -> int:
        retries = 0
        for action in reversed(self.trajectory.current_step.actions):
            if action.retry_message:
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self, state: AgenticState) -> List[Message]:
        messages: list[Message] = []

        if self.trajectory.current_step.name != state.name:
            return messages

        current_step = self.trajectory.current_step
        for action in current_step.actions:
            if action.retry_message:
                if isinstance(action.action, Content):
                    messages.append(
                        AssistantMessage(
                            content=action.action.content,
                        )
                    )
                else:
                    messages.append(AssistantMessage(action=action.action))

                messages.append(
                    UserMessage(
                        content=action.retry_message,
                    )
                )

        return messages

    def transition_to(self, new_state: AgenticState):
        logger.info(f"Transitioning from {self.state} to {new_state}")

        self._transition_count += 1
        print(f"transition count: {self._transition_count}")
        if self._transition_count > self._max_transitions:
            new_state = Rejected(message="Max transitions exceeded.")

        # if self.trajectory.transition_count(new_state) > new_state.max_iterations:
            # new_state = Rejected(message=f"Max transitions exceeded for state {new_state.name}.")


        self._state = new_state # set the new state
        self._set_state_loop(self.state) # set the loop in the new state
        self.trajectory.new_transition(new_state)   # record transision in the trajectory

    def transition_to_state(self, new_state: AgenticState) -> AgenticState:
        """
        Transition to a new state without altering transitions or performing checks.
        This function is primarily for use in MCTS simulations.
        """
        logger.info(f"Transitioning state from {self.state} to {new_state}")

        # Create a copy of the new state to avoid modifying the original
        # new_state_copy = safe_deepcopy(new_state)
        new_state_copy = new_state

        # Set the loop for the new state
        self._set_state_loop(new_state_copy)

        # Update the current state
        self._state = new_state_copy

        return new_state_copy

    @property
    def state(self):
        return self._state

    @property
    def workspace(self):
        return self._workspace

    @property
    def trajectory(self):
        return self._trajectory

    def _to_completion_messages(self, state: AgenticState) -> list[dict]:
        messages = [{"role": "system", "content": state.system_prompt()}]

        tool_call_id = None
        state_messages = state.messages()
        for message in state_messages:
            if message.role == "user":
                if tool_call_id:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                if message.action:
                    tool_call_id = generate_call_id()
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": message.action.action_name,
                                        "arguments": message.action.model_dump_json(
                                            exclude_none=True
                                        ),
                                    },
                                }
                            ],
                        }
                    )
                else:
                    tool_call_id = None
                    messages.append({"role": "assistant", "content": message.content})

        return messages

    def _run(self):
        if not self.is_running():
            logger.info("Loop is not running.")
            return

        action, completion_response, messages = self._next_action(self.state)

        cost = None
        if completion_response:
            try:
                cost = completion_cost(completion_response=completion_response)
            except Exception as e:
                logger.info(f"Error calculating completion cost: {e}")

        logger.info(f"{self.state}: Received new action {action.action_name}.")
        response = self.state.handle_action(action)

        self._trajectory.save_action(
            action=action,
            output=response.output,
            retry_message=response.retry_message,
            completion_cost=cost,
        )

        if not response.trigger:
            logger.info(
                f"{self.state}: No transition found. Staying in the same state."
            )
            return

        if response.trigger == "retry":
            logger.info(f"{self.state}: Retry requested. {response.retry_message}")
            return

        try:
            next_state = self._transitions.next_state(
                source=self.state,
                trigger=response.trigger,
                data=response.output,
            )
        except Exception as e:
            logger.error(f"Failed to initiate next state with trigger {response.trigger} and output {response.output}")
            raise

        if not next_state:
            raise ValueError(
                f"No transition found for {self.state} with trigger {response.trigger}"
            )

        if response.trigger == "rejected" and next_state.__class__ != Rejected:
            self._rejections += 1
            next_state = Rejected(message=f"Got {self._rejections} rejections, aborting.")
        else:
            self._rejections = 0

        logger.info(f"{self.state}: Transitioning to {next_state.name}")
        self.transition_to(next_state)
        print(f"STATE: {self.state}, ACTION: {action}")

    def _run_mcts(self):
        if not self.is_running():
            logger.info("Loop is not running.")
            return

        best_trajectory, final_step, total_reward = self.mcts.run_search(num_iterations=self._max_transitions)  # Adjust number of iterations as needed

        # print(f"Best Trajectory: {best_trajectory}")
        # print(f"Final Step: {final_step}")
        # print(f"Total Reward: {total_reward}")

        return best_trajectory, final_step, total_reward

    def _next_action(self, state: AgenticState) -> Tuple[ActionRequest, Optional[ModelResponse]]:
        messages = self._to_completion_messages(state)
        logger.info(f"{state} Create completion with {len(messages)} messages")

        if self._verify_state_func:
            self._verify_state_func(state)

        if self._mocked_actions is not None:
            if len(self._mocked_actions) == 0:
                raise Exception("No more mocked responses available.")

            action = self._mocked_actions.pop(0)
            print(f"Mocked action: {action}")
            if state.action_type():
                try:
                    logger.info(
                        f"{state} Return mocked response with type {state.action_type().__name__} ({len(self._mocked_actions)} left)."
                    )
                    return state.action_type().model_validate(action), None, None
                except Exception as e:
                    logger.error(f"Failed to parse {action} to {state.action_type().__name__} in state {state.name}")
                    raise
            elif "content" in action:
                logger.info(f"{state} Return mocked response ({len(self._mocked_actions)} left).")
                return Content(content=action["content"]), None, None
            else:
                raise ValueError(f"Mocked action {action} does not have 'content' field.")

        metadata = {}
        if self._metadata:
            metadata.update(self._metadata)
        metadata["generation_name"] = str(state)

        tokens = token_counter(messages=messages[-1:])
        if tokens > self._max_message_tokens:
            raise ValueError(f"Too many tokens in the new message: {tokens}")
        
        print(f"messages: {[message['role'] for message in messages]}")
        print(f"action_type: {state.action_type()}")
        if state.action_type() is None:
            completion_response = litellm.completion(
                model=state.model,
                max_tokens=state.max_tokens,
                temperature=state.temperature,
                stop=state.stop_words(),
                metadata=metadata,
                messages=messages,
            )
            return Content(content=completion_response.choices[0].message.content), completion_response, messages
        else:
            if "mixtral" in state.model:
                mode = instructor.Mode.MISTRAL_TOOLS
            else:
                mode = instructor.Mode.TOOLS

            client = instructor.from_litellm(litellm.completion, mode=mode)
            try:
                action, completion_response = client.chat.completions.create_with_completion(
                    model=state.model,
                    max_tokens=state.max_tokens,
                    temperature=state.temperature,
                    stop=state.stop_words(),
                    response_model=state.action_type(),
                    metadata=metadata,
                    messages=messages,
                )
                return action, completion_response, messages

            except Exception as e:
                print(f"""Messages: {messages},
                        Metadata: {metadata},
                        Model: {state.model},
                        Max Tokens: {state.max_tokens},
                        response_model: {state.action_type()},
                        state: {state},
                        action_type: {state.action_type()},
                        actio_type: {state.action_type()}""")
                raise e
            
    def is_terminal(self) -> bool:
        return isinstance(self, (Finished, Rejected))

    def _handle_retry(self, retry_message: str):
        # Implement retry logic here
        # This might involve updating the MCTS tree or resetting the search
        pass


import graphviz
from typing import Tuple, Optional, Dict, Any
from collections import OrderedDict
import random
import logging
from .utils_search.visualize_tree import MCTSVisualizer
from .search.reward import LLM_Value_Function
import math

# logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, id, state, parent=None, 
                 last_action=None, 
                 last_completion_messages=None,
                 last_completion_response=None,
                 next_completion_messages=None,
                 loop=None, step=0, **kwargs):
        self.id = id
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.raw_value = 0.0
        self.last_action = last_action
        self.last_completion_messages = last_completion_messages
        self.last_completion_response = last_completion_response
        self.next_completion_messages = next_completion_messages
        self.loop = loop
        self.step = step
        self.trajectory = []

        for key, value in kwargs.items():
            setattr(self, key, value)


class MCTS:

    def __init__(self, root_state, loop, c_param: float=1.41, max_actions: int=2):
        self.node_count = 1  # Class variable to keep track of total nodes across all searches
        self.loop = loop
        self.taskname = self.loop._trajectory_path.replace('trajs', 'flow_chart').replace('.json', '')
        self.visualizer = MCTSVisualizer(name=self.taskname)
        self.root = MCTSNode(id=self.node_count, 
                             state=TrajectoryTransition(state=root_state).model_dump(), 
                             loop=loop)
        self.nodes = OrderedDict({self.root.id: self.root})
        self.c_param = c_param
        self.max_actions = max_actions
        self.max_depth = 20
        self.first = True
        self.context_history = {}
        value_fun_taskname = self.loop._trajectory_path.replace('trajs', 'rews')
        self.tree_filename = self.loop._trajectory_path.replace('trajs', 'tree')
        self.value_function = LLM_Value_Function(filename=value_fun_taskname)

    def save_search_tree(self):
        os.makedirs(os.path.dirname(self.tree_filename), exist_ok=True)
        
        traj_vars_ommit = ["name", "initial_message"]
        
        def save_trajectory(trajectory):
            if isinstance(trajectory, dict):
                return {k: v for k,v in trajectory.items() if k not in traj_vars_ommit}
            else:
                return trajectory
        
        def build_tree_dict(node):
            node_dict = {
                'node_info': {
                    'id': node.id,
                    'visits': node.visits,
                    'value': node.value,
                },
                'trajectory': save_trajectory(node.trajectory),
                'children': {}
            }
            for child in node.children:
                node_dict['children'][child.id] = build_tree_dict(child)
            return node_dict

        tree_dict = build_tree_dict(self.root)
        full_dict = {
            "Reported Issue": self.loop._trajectory._initial_message,
            "Search Tree": tree_dict,
        }
        
        # Save the tree dictionary to a JSON file
        with open(self.tree_filename, 'w') as f:
            json.dump(full_dict, f, indent=2)
        
        print(f"Hierarchical search tree saved to {self.tree_filename}")

        return full_dict

    def copy_state(self, state, loop):
        try:
            if hasattr(state, "_loop"):
                print(f"loop: {state._loop}")
                if state._loop is not None:
                    state._set_loop(None)
            
            copy_state = safe_deepcopy(state)
            copy_loop = safe_deepcopy(loop)
            
            state._set_loop(loop)  # Restore original state
            copy_state._set_loop(copy_loop)
            copy_loop.transition_to_state(copy_state)
        except Exception as e:
            print(f"Error in copy_state: {e}")
            traceback.print_exc()
            raise
    
        return copy_state, copy_loop

    def get_state(self, node):
        trajectory = trajectory_from_dict(deepcopy(node.trajectory), self.loop)
        transitions = trajectory._transitions
        if transitions:
            state = reconstruct_state(node.state, self.loop)
        else:
            print("No transitions found in the trajectory")
            return None

        # Deserialize and set file context
        deserialized_file_context = FileContext.from_json(node._file_context)
        
        # Save any changes in the current file context before replacing it
        self.loop._workspace._file_context.save()
        
        # Replace the file context
        self.loop._workspace._file_context = deserialized_file_context
        
        # # Debug comparison (consider wrapping in a debug flag condition)
        # if self.debug:
        #     differences = compare_file_contexts(self.loop._workspace._file_context, deserialized_file_context)
        #     if differences:
        #         print(f"File context differences: {differences}")
        #         view_file_differences_compact(self.loop._workspace._file_context, deserialized_file_context)

        # Set up the state and trajectory
        state._set_loop(self.loop)
        self.loop._trajectory.set_transitions(transitions)
        self.loop.transition_to_state(state)

        # Ensure any changes in the new file context are saved to disk
        self.loop._workspace._file_context.save()

        return state

    def filter_nodes(self, nodes: List[MCTSNode]) -> List[MCTSNode]:
        nodes_to_explore = []
        if len(nodes.keys()) > 6:
            if random.random() < 0.5:
                self.max_actions = 1
            else:
                self.max_actions = 2
        else:
            self.max_actions = 2
        for node_id, node in nodes.items():
            if isinstance(node.state, (Finished, Rejected)):
                continue
            elif len(node.children) >= self.max_actions:
                print(f"Node{node.id} has {len(node.children)} children, skipping")
                continue
            else:
                nodes_to_explore.append(node_id)
        return nodes_to_explore

    def ucb_score(self, parent: MCTSNode, child: MCTSNode, exploration_weight: float = 1.41) -> float:
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.value / child.visits
        exploration = exploration_weight * math.sqrt(math.log(parent.visits) / child.visits)
        
        return exploitation + exploration

    def get_best_explore_from_ucb(self, parent: MCTSNode, nodes: List[MCTSNode]) -> MCTSNode:
        return max(nodes, key=lambda n: self.ucb_score(parent, n))
        
    def filter_mature_nodes(self) -> List[MCTSNode]:
        filtered_nodes = []
        for node_id, node in self.nodes.items():
            if node.state['name'] in ['Finished', 'Rejected']:
                print(f"Node{node.id} is terminal, skipping")
                continue
            elif len(node.children) < self.max_actions:
                filtered_nodes.append(node)
            # else:
            #     # Calculate average reward for the node and its best child
            #     node_avg_reward = node.value / node.visits if node.visits > 0 else 0
            #     best_child_avg_reward = max((child.value / child.visits if child.visits > 0 else 0) for child in node.children)
                
            #     # If the node's average reward is less than its best child's, it might still be worth exploring
            #     if node_avg_reward < best_child_avg_reward:
            #         filtered_nodes.append(node)
        
        return filtered_nodes
    
    def select(self) -> MCTSNode:
        # Get all unexpanded or promising nodes
        available_nodes = self.filter_mature_nodes()
        
        if not available_nodes:
            print("No available nodes to expand.")
            return None

        # Select the best node according to UCB
        best_node = max(available_nodes, key=lambda n: self.ucb_score(n.parent, n) if n.parent else float('inf'))
        
        print(f"Selected node: {best_node.id}")
        return best_node

    def update_visits_for_state(self, target_state: AgenticState, visit_increment: int = 1):
        for node in self.nodes.values():
            if node.state == target_state:
                node.visits += visit_increment
                print(f"Updated Node{node.id} visits to {node.visits}")

    def run_search(self, num_iterations: int) -> List[Tuple[ActionRequest, AgenticState]]:
        for _ in range(num_iterations):
            node = self.select()
            
            if node is None or isinstance(node.state, (Finished, Rejected)):
                break  # No more nodes to explore or we've reached a terminal state
            
            if len(node.children) < self.max_actions:
                child = self.expand(node)
                if child is not None:
                    node = child
            
            result = self.simulate(node)
            
            self.backpropagate(node, result)

            self.visualizer.update_graph(self.root)
            
            # save json dict
            self.search = self.save_search_tree()
            
        # eval tree
        self.value_function.eval_tree(self.tree_filename)
            
        # self.visualizer.update_graph(self.root)  # Final update after all iterations
        return self.get_best_trajectory()

    def get_best_trajectory(self) -> Tuple[List[Tuple[ActionRequest, AgenticState]], int]:
        finished_nodes = [node for node in self.nodes.values() if node.state['name'] == "Finished"]
        
        if not finished_nodes:
            print("No finished nodes found. Returning the best path based on UCB scores.")
            return self.get_best_path_ucb()
        
        best_path = None
        best_total_reward = float('-inf')
        
        for finished_node in finished_nodes:
            path = []
            node = finished_node
            total_reward = 0
            
            while node is not None:
                if node.last_action:
                    path.append((node.last_action, node.state))
                total_reward += node.value / node.visits if node.visits > 0 else 0
                node = node.parent
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_path = list(reversed(path))
        
        if best_path:
            print(f"Best finished path found with total reward: {best_total_reward}")
            return trajectory_from_dict(finished_node.trajectory), finished_node, best_total_reward
        else:
            print("No valid finished path found. This should not happen if there are finished nodes.")
            return None, None, 0

    def get_best_path_ucb(self) -> Tuple[List[Tuple[ActionRequest, AgenticState]], int]:
        trajectory = []
        node = self.root
        total_reward = 0
        
        while node.children:
            best_child = max(node.children, key=lambda c: self.ucb_score(node, c, exploration_weight=0))
            if best_child.last_action:
                trajectory.append((best_child.last_action, best_child.state))
            total_reward += best_child.value / best_child.visits if best_child.visits > 0 else 0
            node = best_child
        
        return trajectory, node, total_reward

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        # if node.state._loop is None:
        #     node.state._set_loop(self.loop)
        
        if node.id <= 1:
            self.loop._trajectory = Trajectory(
            "AgenticLoop", 
            initial_message=self.loop._trajectory._initial_message, 
            persist_path=self.loop._trajectory_path
        )   
            self.loop.transition_to(self.loop._transitions.initial_state())
            expand_state, expand_loop = self.loop._transitions.initial_state(), self.loop
            expand_state._set_loop(expand_loop)
        else:
            # Create a copy of the original state
            # expand_state, expand_loop = self.copy_state(node.state, node.loop)
            expand_state, expand_loop = self.get_state(node), self.loop

        print(f"STATE: {expand_state}")
        action, completion_response, messages = expand_loop._next_action(expand_state)
        response = expand_state.handle_action(action)
        
        # print(f"RESPONSE: {response}, ACTION: {action}, MESSAGES: {messages}")
        
        expand_loop._trajectory.save_action(
            action=action,
            output=response.output,
            retry_message=response.retry_message,
        ) 
        
        # print(f"Node{node.id} - Action generated: {action}")
        # print(f"Node{node.id} - Response: {response}")
        # workspace = expand_loop._workspace.serialize()

        if not response.trigger or response.trigger == 'retry':
            print(f"Node{node.id} - No valid trigger, {response.trigger} requested, response: {response}")
            next_state, next_loop = self.get_state(node), self.loop
            # next_completion_messages = expand_loop._to_completion_messages(next_state)
            # raise ValueError(f"No valid trigger for {expand_state} with action {action}")
        else:
            try:
                next_state = expand_loop._transitions.next_state(
                    source=expand_state,
                    trigger=response.trigger,
                    data=response.output,
                )
                # next_completion_messages = expand_loop._to_completion_messages(next_state)
            except Exception as e:
                logger.error(f"Failed to initiate next state with trigger {response.trigger} and output {response.output}")
                raise
        
        if not next_state:
            raise ValueError(
                f"No transition found for {self.state} with trigger {response.trigger}"
            )
        
        if response.trigger == "rejected" and next_state.__class__ != Rejected:
            expand_loop._rejections += 1
            next_state = Rejected(message=f"Got {self._rejections} rejections, aborting.")
        else:
            expand_loop._rejections = 0


        # print(f"ACTIONS: {expand_loop._trajectory._transitions}")  
        expand_loop.transition_to(next_state)
        _file_context = expand_loop._workspace._file_context.dict()
        # workspace = expand_loop._workspace.serialize()
        print(f"STATE: {expand_state}, NEXT STATE: {next_state}")
        # next_completion_messages = expand_loop._to_completion_messages(next_state)
        # child_node.next_completion_messages = next_completion_messages
        saved_trajectory = expand_loop._trajectory.to_dict()
        last_transition = saved_trajectory['transitions'].pop()
        step_count = node.step + 1
        self.node_count += 1  # Increment the class variable
        
        git_diff = expand_loop.get_git_diff()
        # print(f"git diff: {git_diff.stdout}")
        
        child_node = MCTSNode(
            id=self.node_count,
            state=last_transition,
            parent=node,
            last_action=action.model_dump(),
            last_completion_messages=messages,
            last_completion_response=completion_response,
            # next_completion_messages=next_completion_messages,
            loop=expand_loop,
            step=step_count,
            trajectory=saved_trajectory,
            git_diff=git_diff,
            _file_context=_file_context,
            # workspace=workspace
        )
        
        self.nodes[node.id].children.append(child_node)
        self.nodes[child_node.id] = child_node
        # self.context_history[child_node.id] = expand_state.file_context
        # save_json_dict(self.context_history, "context_history.json")
        print(f"file context: {expand_state.file_context}")
            
        if next_state is None:
            print(f"Node{node.id} - No valid next state")
            raise ValueError(f"No valid next state for {expand_state} with trigger {response.trigger}")
        
        print(f"""Node{node.id} expanded to Node{child_node.id}_{id(child_node)}
                  State: {next_state}""")
        
        self.visualizer.add_node_to_graph(child_node)
        return child_node
    
    def update_root(self, new_state: AgenticState):
        # Try to find a child node that matches the new state
        for child in self.root.children:
            if child.state == new_state:
                self.root = child
                self.root.parent = None  # The new root has no parent
                logger.info(f"MCTS root updated to existing node: {self.root.id}")
                return

        # If no matching child is found, create a new root
        self.node_count += 1
        self.root = MCTSNode(id=self.node_count, state=new_state)
        logger.info(f"MCTS root reset to new node: {self.root.id}")

    def simulate(self, node: MCTSNode) -> float:
        # reward = self.calculate_reward(node.state, node.last_action, node.state)
        # reward = 1

        # LLM reward
        reward = self.value_function.get_reward(
            problem_statement=deepcopy(self.loop._trajectory.initial_message),
            state_message=deepcopy(node.last_completion_messages), 
            state_response=deepcopy(node.last_completion_response),
            # next_state_message=deepcopy(node.next_completion_messages),
            step_count=node.step,
            node_id=node.id
        )
        return reward

    def calculate_reward(self, current_state: AgenticState, action: ActionRequest, next_state: AgenticState) -> float:
        # Implement a more comprehensive reward calculation here
        # This should take into account the desirability of the transition
        # For example:
        if isinstance(next_state, Finished):
            return 1.0  # High reward for reaching a finished state
        elif isinstance(next_state, Rejected):
            return -1.0  # Negative reward for reaching a rejected state
        else:
            # Calculate a reward based on how "good" the transition is
            # This could involve comparing some properties of current_state and next_state
            # For now, we'll return a small positive reward for non-terminal transitions
            return 0.1
    
    def backpropagate(self, node: MCTSNode, result: float):
        path = []
        visited_node_ids = set()  # To keep track of node IDs we've already updated
        node.raw_value = result

        while node is not None:
            # Update the current node
            node.visits += 1
            node.value += result
            path.append(f"Node{node.id}_({node.visits}, {node.value:.2f})")

            # Update all nodes with the same state
            if node.id not in visited_node_ids:
                for other_node_id, other_node in self.nodes.items():
                    if other_node_id != node.id:
                        other_node.visits += 1
                        other_node.value += result
                        # Explicitly update the node in self.nodes
                        self.nodes[other_node_id] = other_node
                        # print(f"Updated Node{other_node.id} with same state as Node{node.id}")
                visited_node_ids.add(node.id)

            # Explicitly update the current node in self.nodes
            self.nodes[node.id] = node

            node = node.parent

        print(f"Backpropagation path: {' -> '.join(reversed(path))}")
        self.visualizer.update_graph(self.root)

    def best_child(self, node: MCTSNode) -> MCTSNode:
        best = max(node.children, key=lambda n: n.value / n.visits + self.c_param * (2 * node.visits / n.visits) ** 0.5)
        print(f"Best child of Node{node.id}: Node{best.id}")
        return best

    def progressive_widening_probability(self, node: MCTSNode) -> float:
        # prob = self.max_actions / (node.visits + self.max_actions)
        # print(f"Node{node.id} - Progressive widening probability: {prob:.2f}")
        prob = 0.5
        return prob


def print_attr(_class):
    for attr in dir(_class):
        if not callable(getattr(_class, attr)) and \
            not attr.startswith("__"):
            print(f"{attr}: {getattr(_class, attr)}")
            
            
            
def compare_objects(obj1: Any, obj2: Any, path: str = "") -> list[str]:
    differences = []

    if type(obj1) != type(obj2):
        return [f"{path}: Type mismatch - {type(obj1)} vs {type(obj2)}"]

    if isinstance(obj1, (int, float, str, bool, type(None))):
        if obj1 != obj2:
            differences.append(f"{path}: Value mismatch - {obj1} vs {obj2}")

    elif isinstance(obj1, (list, tuple, set)):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: Length mismatch - {len(obj1)} vs {len(obj2)}")
        else:
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                differences.extend(compare_objects(item1, item2, f"{path}[{i}]"))

    elif isinstance(obj1, dict):
        keys1, keys2 = set(obj1.keys()), set(obj2.keys())
        if keys1 != keys2:
            differences.append(f"{path}: Key mismatch - {keys1.symmetric_difference(keys2)}")
        for key in keys1 & keys2:
            differences.extend(compare_objects(obj1[key], obj2[key], f"{path}.{key}"))

    else:  # Custom object
        attrs = [attr for attr in dir(obj1) if not attr.startswith('__') and not callable(getattr(obj1, attr))]
        for attr in attrs:
            value1, value2 = getattr(obj1, attr), getattr(obj2, attr)
            if inspect.ismethod(value1) or inspect.isfunction(value1):
                continue
            differences.extend(compare_objects(value1, value2, f"{path}.{attr}"))

    return differences

def compare_file_contexts(fc1: 'FileContext', fc2: 'FileContext') -> list[str]:
    return compare_objects(fc1, fc2, "FileContext")


def view_file_differences_compact(fc1, fc2):
    files1 = set(fc1._repo._files)
    files2 = set(fc2._repo._files)

    for file in sorted(files1 | files2):
        if file in files1 and file not in files2:
            print(f"- {file}")
        elif file in files2 and file not in files1:
            print(f"+ {file}")