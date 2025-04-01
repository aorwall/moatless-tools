import json
from datetime import datetime

import pytest
from moatless.actions.action import Action
from moatless.actions.finish import FinishArgs
from moatless.actions.schema import Observation, ActionArguments
from moatless.file_context import FileContext
from moatless.node import ActionStep, Node
from moatless.repository.repository import InMemRepository


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class TestAction(Action):
    def execute(self, file_context):
        return "test"

class TestActionArguments(ActionArguments):
    pass

def test_node_model_dump():
    # Create a root node
    root = Node(node_id=0, max_expansions=3)

    # Create some child nodes
    child1 = Node(node_id=1, max_expansions=2)
    child2 = Node(node_id=2, max_expansions=2)
    grandchild1 = Node(node_id=3, max_expansions=1)

    # Build the tree structure
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild1)

    # Add some data to the nodes
    root.visits = 10
    root.value = 100.0
    root.message = "Root node"

    child1.visits = 5
    child1.value = 50.0
    child1.message = "Child 1"

    child2.visits = 3
    child2.value = 30.0
    child2.message = "Child 2"

    grandchild1.visits = 2
    grandchild1.value = 20.0
    grandchild1.message = "Grandchild 1"

    # Add a FinishArgs action to one of the nodes
    finish_args = FinishArgs(
        thoughts="Task is complete because all requirements are met.",
        finish_reason="All tests pass and code is optimized."
    )
    child1.action_steps = [ActionStep(action=finish_args)]

    # Get the model dump
    dumped_data = root.model_dump()
    print(json.dumps(dumped_data, indent=2, cls=DateTimeEncoder))

    # Verify the structure and content of the dumped data
    assert dumped_data["node_id"] == 0
    assert dumped_data["visits"] == 10
    assert dumped_data["value"] == 100.0
    assert dumped_data["user_message"] == "Root node"
    assert len(dumped_data["children"]) == 2
    assert "parent" not in dumped_data  # Ensure parent is not included

    child1_data = dumped_data["children"][0]
    assert child1_data["node_id"] == 1
    assert child1_data["visits"] == 5
    assert child1_data["value"] == 50.0
    assert child1_data["user_message"] == "Child 1"
    assert child1_data["action_steps"][0]["action"]["thoughts"] == "Task is complete because all requirements are met."
    assert child1_data["action_steps"][0]["action"]["finish_reason"] == "All tests pass and code is optimized."
    assert len(child1_data["children"]) == 1
    assert "parent" not in child1_data  # Ensure parent is not included

    child2_data = dumped_data["children"][1]
    assert child2_data["node_id"] == 2
    assert child2_data["visits"] == 3
    assert child2_data["value"] == 30.0
    assert child2_data["user_message"] == "Child 2"
    assert len(child2_data["children"]) == 0

    grandchild1_data = child1_data["children"][0]
    assert grandchild1_data["node_id"] == 3
    assert grandchild1_data["visits"] == 2
    assert grandchild1_data["value"] == 20.0
    assert len(grandchild1_data["children"]) == 0

    # Test loading the dumped data
    loaded_root = Node.reconstruct(dumped_data)
    loaded_child1 = loaded_root.children[0]
    
    assert isinstance(loaded_child1.action_steps[0].action, FinishArgs)
    assert isinstance(loaded_child1.action, FinishArgs)
    assert loaded_child1.action.name == "Finish"
    assert loaded_child1.action.thoughts == "Task is complete because all requirements are met."
    assert loaded_child1.action.finish_reason == "All tests pass and code is optimized."

    # Test with custom exclude
    custom_dump = root.model_dump(exclude={"value", "message"})
    assert "value" not in custom_dump
    assert "parent" not in custom_dump  # 'parent' should still be excluded

    # Test with exclude_none=True 
    none_dump = root.model_dump(exclude_none=True)
    assert "reward" not in none_dump  # 'reward' should be excluded as it's None
