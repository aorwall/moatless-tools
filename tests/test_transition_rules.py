import json

import pytest
from pydantic import BaseModel
from moatless.transition_rules import TransitionRules, TransitionRule
from moatless.state import AgenticState, Finished, Rejected, Pending
from moatless.types import ActionResponse


class MockStateA(AgenticState):
    value: int = 0

    def _execute_action(self, action: str, **kwargs):
        if action == "to_b":
            return ActionResponse(output={"message": "Moving to B"}, trigger="to_b")
        return ActionResponse(output={"message": "Staying in A"}, trigger=None)


class MockStateB(AgenticState):
    default_name: str = ""

    def _execute_action(self, action: str, **kwargs):
        if action == "finish":
            return ActionResponse(output={"message": "Finishing"}, trigger="finish")
        elif action == "reject":
            return ActionResponse(output={"message": "Rejecting"}, trigger="reject")
        return ActionResponse(output={"message": "Staying in B"}, trigger=None)

    @property
    def name(self):
        return self.default_name or super().name


def test_transition_rules_serialization_deserialization():
    rules = TransitionRules(
        initial_state=MockStateA,
        transition_rules=[
            TransitionRule(
                source=MockStateA,
                dest=MockStateB,
                trigger="to_b",
                required_fields={"foo"},
            ),
            TransitionRule(source=MockStateB, dest=Finished, trigger="finish"),
            TransitionRule(source=MockStateB, dest=Rejected, trigger="reject"),
        ],
        global_params={"model": "gpt-4o"},
        state_params={
            MockStateB: {"model": "claude-3.5-sonnet"},
        },
    )

    # Serialize to dict instead of JSON
    dict_data = rules.model_dump(exclude_none=True, exclude_unset=True)

    # Verify that all values are set in dict_data
    assert dict_data["initial_state"] == "MockStateA"
    assert len(dict_data["transition_rules"]) == 3
    assert dict_data["global_params"] == {"model": "gpt-4o"}
    assert dict_data["state_params"] == {"MockStateB": {"model": "claude-3.5-sonnet"}}

    # Check transition rules
    transition_rules = dict_data["transition_rules"]
    assert transition_rules[0] == {
        "source": "MockStateA",
        "dest": "MockStateB",
        "trigger": "to_b",
        "required_fields": ["foo"],
    }
    assert transition_rules[1] == {
        "source": "MockStateB",
        "dest": "Finished",
        "trigger": "finish",
    }
    assert transition_rules[2] == {
        "source": "MockStateB",
        "dest": "Rejected",
        "trigger": "reject",
    }

    # Deserialize from dict
    deserialized_rules = TransitionRules.model_validate(dict_data)

    # Check if the deserialized object matches the original
    assert deserialized_rules.initial_state == rules.initial_state
    assert len(deserialized_rules.transition_rules) == len(rules.transition_rules)
    assert deserialized_rules.global_params == rules.global_params
    assert deserialized_rules.state_params == rules.state_params

    # Check if the internal _source_trigger_index is rebuilt correctly
    assert deserialized_rules._source_trigger_index == rules._source_trigger_index

    json_data = json.dumps(
        rules.model_dump(exclude_none=True, exclude_unset=True), indent=2
    )
    assert (
        json_data
        == """{
  "initial_state": "MockStateA",
  "transition_rules": [
    {
      "trigger": "to_b",
      "source": "MockStateA",
      "dest": "MockStateB",
      "required_fields": [
        "foo"
      ]
    },
    {
      "trigger": "finish",
      "source": "MockStateB",
      "dest": "Finished"
    },
    {
      "trigger": "reject",
      "source": "MockStateB",
      "dest": "Rejected"
    }
  ],
  "global_params": {
    "model": "gpt-4o"
  },
  "state_params": {
    "MockStateB": {
      "model": "claude-3.5-sonnet"
    }
  }
}"""
    )


def test_find_transition_rule():
    rules = TransitionRules(
        initial_state=MockStateA,
        transition_rules=[
            TransitionRule(source=MockStateA, dest=MockStateB, trigger="to_b"),
            TransitionRule(source=MockStateB, dest=Finished, trigger="finish"),
        ],
        global_params={"model": "gpt-4o"},
        state_params={
            MockStateB: {"model": "claude-3.5-sonnet"},
        },
    )

    # Test finding an existing rule
    found_rules = rules.find_transition_rule_by_source_and_trigger(MockStateA, "to_b")
    assert len(found_rules) == 1
    assert found_rules[0].source == MockStateA
    assert found_rules[0].dest == MockStateB
    assert found_rules[0].trigger == "to_b"

    # Test finding a non-existent rule
    not_found_rules = rules.find_transition_rule_by_source_and_trigger(
        MockStateA, "non_existent_trigger"
    )
    assert len(not_found_rules) == 0


def test_next_state():
    rules = TransitionRules(
        initial_state=MockStateA,
        transition_rules=[
            TransitionRule(
                source=MockStateA,
                dest=MockStateB,
                trigger="to_b",
                required_fields={"value"},
            ),
            TransitionRule(source=MockStateB, dest=Finished, trigger="finish"),
            TransitionRule(source=MockStateB, dest=Rejected, trigger="reject"),
        ],
        global_params={"model": "gpt-4o"},
        state_params={
            MockStateB: {"model": "claude-3.5-sonnet"},
        },
    )

    # Test successful transition
    source_state = MockStateA(value=5)
    action_response = source_state._execute_action("to_b")
    next_state = rules.next_state(source_state, action_response.trigger, {"value": 5})
    assert isinstance(next_state, MockStateB)
    assert next_state.name == "MockStateB"
    assert next_state.model == "claude-3.5-sonnet"

    # Test transition with missing required fields
    action_response = source_state._execute_action("to_b")
    next_state = rules.next_state(source_state, action_response.trigger, {})
    assert next_state is None

    # Test transition to Finished state
    source_state = MockStateB(default_name="TestB")
    action_response = source_state._execute_action("finish")
    next_state = rules.next_state(source_state, action_response.trigger, {})
    assert isinstance(next_state, Finished)

    # Test transition to Rejected state
    action_response = source_state._execute_action("reject")
    next_state = rules.next_state(
        source_state, action_response.trigger, {"message": "Custom rejection message"}
    )
    assert isinstance(next_state, Rejected)
    assert next_state.message == "Custom rejection message"


def test_initial_state_creation():
    rules = TransitionRules(
        initial_state=MockStateA,
        transition_rules=[],
        global_params={"model": "gpt-4o"},
        state_params={
            MockStateB: {"model": "claude-3.5-sonnet"},
        },
    )

    initial_state = rules.create_initial_state()
    print(initial_state)
    assert isinstance(initial_state, MockStateA)
    assert initial_state.model == "gpt-4o"

    # Test overriding with custom data
    custom_initial_state = rules.create_initial_state(value=20, model="custom-model")
    assert isinstance(custom_initial_state, MockStateA)
    assert custom_initial_state.model == "custom-model"
    assert custom_initial_state.value == 20


if __name__ == "__main__":
    pytest.main([__file__])
