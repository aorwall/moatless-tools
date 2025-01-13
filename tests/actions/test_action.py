from typing import Union, Type

import pytest
from instructor import OpenAISchema
from pydantic import Field

from moatless.actions.action import Action, ActionArguments
from moatless.actions.code_change import RequestCodeChange, RequestCodeChangeArgs
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.semantic_search import SemanticSearch, SemanticSearchArgs
from moatless.actions.view_code import ViewCode, ViewCodeArgs


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


def test_take_action():
    actions = [SemanticSearchArgs, RequestCodeChangeArgs, FinishArgs]

    class TakeAction(OpenAISchema):
        action: Union[tuple(actions)] = Field(...)

        class Config:
            smart_union = True

    action_type = TakeAction
    schema = action_type.model_json_schema()
    assert "properties" in schema
    assert "action" in schema["properties"]


def test_get_action_by_args_class():
    assert Action.get_action_by_args_class(FinishArgs) == Finish
    assert Action.get_action_by_args_class(RequestCodeChangeArgs) == RequestCodeChange
    assert Action.get_action_by_args_class(ViewCodeArgs) == ViewCode
    assert Action.get_action_by_args_class(SemanticSearchArgs) == SemanticSearch

def test_get_action_by_name():
    assert Action.get_action_by_name("Finish") == Finish
    assert Action.get_action_by_name("RequestCodeChange") == RequestCodeChange
    assert Action.get_action_by_name("ViewCode") == ViewCode
    assert Action.get_action_by_name("SemanticSearch") == SemanticSearch
    assert Action.get_action_by_name("FindFunction") == FindFunction

    with pytest.raises(ValueError):
        Action.get_action_by_name("NonExistentAction")
