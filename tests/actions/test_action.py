from typing import Union, Type

import pytest
from moatless.actions.action import Action, ActionArguments
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.semantic_search import SemanticSearch, SemanticSearchArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.view_code import ViewCode, ViewCodeArgs
from pydantic import Field


def test_action_schema():
    schema = ViewCodeArgs.model_json_schema()
    assert "description" in schema
    assert "title" in schema


def test_action_name():
    class TestAction(Action):
        args_schema: Type[ActionArguments] = FinishArgs

    action = TestAction()
    assert action.name == "TestAction"


def test_action_args_name():
    print(FinishArgs.name)

    assert FinishArgs.name == "Finish"


def test_get_action_by_args_class():
    assert Action.get_action_by_args_class(FinishArgs) == Finish
    assert Action.get_action_by_args_class(StringReplaceArgs) == StringReplace
    assert Action.get_action_by_args_class(ViewCodeArgs) == ViewCode
    assert Action.get_action_by_args_class(SemanticSearchArgs) == SemanticSearch


def test_get_action_by_name():
    """Test that get_action_by_name correctly finds actions by their class name."""
    # Test with some real actions from our imports
    assert Action.get_action_by_name("Finish") == Finish
    assert Action.get_action_by_name("StringReplace") == StringReplace
    assert Action.get_action_by_name("ViewCode") == ViewCode
    assert Action.get_action_by_name("FindFunction") == FindFunction

    # Test with an invalid action name
    with pytest.raises(ValueError) as exc_info:
        Action.get_action_by_name("NonExistentAction")
    assert "Unknown action: NonExistentAction" in str(exc_info.value)
    # Verify that the error message contains some real action names
    assert "Finish" in str(exc_info.value)
    assert "StringReplace" in str(exc_info.value)
