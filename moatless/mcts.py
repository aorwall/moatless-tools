import random
from typing import Optional, List, Tuple
from pydantic import BaseModel

# New classes for MCTS
class MCTSNode(BaseModel):
    state: AgenticState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = []
    visits: int = 0
    value: float = 0.0
    untried_actions: List[ActionRequest] = []

class MCTS:
    def __init__(self, root_state: AgenticState, transitions: Transitions, c_param: float = 1.41):
        self.root = MCTSNode(state=root_state)
        self.transitions = transitions
        self.c_param = c_param

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.untried_actions == [] and node.children != []:
            node = max(node.children, key=lambda n: n.value / n.visits + self.c_param * (2 * node.visits / n.visits) ** 0.5)
        return node

    def expand(self, node: MCTSNode, action: ActionRequest) -> MCTSNode:
        new_state = self.transitions.next_state(node.state, action.action_name, {})  # Simplified transition
        child_node = MCTSNode(state=new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def simulate(self, node: MCTSNode) -> float:
        # Placeholder for simulation - replace with actual simulation logic
        return random.random()

    def backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def search(self, num_simulations: int) -> ActionRequest:
        for _ in range(num_simulations):
            node = self.select(self.root)
            if node.untried_actions:
                action = node.untried_actions.pop()
                child = self.expand(node, action)
                result = self.simulate(child)
                self.backpropagate(child, result)
            else:
                result = self.simulate(node)
                self.backpropagate(node, result)
        
        best_child = max(self.root.children, key=lambda n: n.visits)
        return best_child.state.last_action
