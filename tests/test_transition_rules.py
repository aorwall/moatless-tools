import json

import pytest
from moatless.transition_rules import TransitionRules, TransitionRule
from moatless.state import Finished, Rejected, Pending, State, ActionRequest, StateOutcome


class MockStateA(State):
    value: int = 0

    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        return StateOutcome(output={"value": self.value})


class MockStateB(State):
    default_name: str = ""

    @property
    def name(self):
        return self.default_name or super().name
    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        return StateOutcome()

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

    data = rules.model_dump(exclude_none=True, exclude_unset=True)

    assert (
        data
        == {
  "global_params": {
    "model": "gpt-4o"
  },
  "state_params": {
    "MockStateB": {
      "model": "claude-3.5-sonnet"
    }
  },
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
  ]
})


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


def test_next_transition_rule():
    rules = TransitionRules(
        transition_rules=[
            TransitionRule(
                source=Pending,
                dest=MockStateA,
                trigger="init",
            ),
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
    source_state = MockStateA(id=1, value=5)
    next_transition_rule = rules.get_next_rule(source_state, "to_b", {"value": 5})
    assert isinstance(next_transition_rule, TransitionRule)
    assert next_transition_rule.source == MockStateA
    assert next_transition_rule.dest == MockStateB
    assert next_transition_rule.trigger == "to_b"
    assert next_transition_rule.required_fields == {"value"}

    # Test transition with missing required fields
    next_transition_rule = rules.get_next_rule(source_state, "to_b", {})
    assert next_transition_rule is None

    # Test transition to Finished state
    source_state = MockStateB(id=2, default_name="TestB")
    next_transition_rule = rules.get_next_rule(source_state, "finish", {})
    assert next_transition_rule is not None
    assert next_transition_rule.source == MockStateB
    assert next_transition_rule.dest == Finished
    assert next_transition_rule.trigger == "finish"

    # Test transition to Rejected state
    next_transition_rule = rules.get_next_rule(
        source_state, "reject", {}
    )   
    assert next_transition_rule is not None
    assert next_transition_rule.source == MockStateB
    assert next_transition_rule.dest == Rejected
    assert next_transition_rule.trigger == "reject"


if __name__ == "__main__":
    pytest.main([__file__])
