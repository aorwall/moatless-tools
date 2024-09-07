import pytest
from unittest.mock import Mock, patch

from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.benchmark.utils import get_moatless_instance
from moatless.edit.plan import (
    PlanToCode,
    TakeAction,
    RequestCodeChange,
    PlanRequest,
    Finish,
    ChangeType,
)


@pytest.fixture
def scikit_learn_workspace():
    instance = get_moatless_instance("scikit-learn__scikit-learn-25570", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "sklearn/compose/tests/test_column_transformer.py",
        ["test_column_transformer_sparse_stacking"],
    )
    return workspace


@pytest.fixture
def django_workspace():
    instance = get_moatless_instance("django__django-14016", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "django/db/models/query_utils.py", ["_combine"]
    )
    return workspace


@pytest.fixture
def sympy_workspace():
    instance = load_instance("sympy__sympy-16988")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "sympy/sets/sets.py", ["imports", "imageset"]
    )
    return workspace


def test_request_for_adding_one_function(scikit_learn_workspace):
    plan_to_code = PlanToCode(
        id=0, _workspace=scikit_learn_workspace, initial_message="Test initial message"
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.addition,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=475,
        end_line=500,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sklearn/compose/tests/test_column_transformer.py",
        "change_type": "addition",
        "span_ids": ["test_column_transformer_sparse_stacking"],
        "start_line": 475,
        "end_line": 475,
    }


def test_request_for_delete_one_import(scikit_learn_workspace):
    """
    Include more than one line when deleting a single row to avoid deleting the wrong line
    """
    plan_to_code = PlanToCode(
        id=0, _workspace=scikit_learn_workspace, initial_message="Test initial message"
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.deletion,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=12,
        end_line=12,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sklearn/compose/tests/test_column_transformer.py",
        "change_type": "modification",
        "span_ids": ["imports"],
        "start_line": 7,
        "end_line": 16,
    }


def test_request_for_delete_one_function(scikit_learn_workspace):
    plan_to_code = PlanToCode(
        id=0, _workspace=scikit_learn_workspace, initial_message="Test initial message"
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.deletion,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=452,
        end_line=474,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sklearn/compose/tests/test_column_transformer.py",
        "change_type": "deletion",
        "span_ids": ["test_column_transformer_sparse_stacking"],
        "start_line": 452,
        "end_line": 474,
    }


def test_request_for_adding_function_on_outcommented_line(scikit_learn_workspace):
    plan_to_code = PlanToCode(
        id=0, _workspace=scikit_learn_workspace, initial_message="Test initial message"
    )
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.addition,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=475,
        end_line=475,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sklearn/compose/tests/test_column_transformer.py",
        "change_type": "addition",
        "span_ids": ["test_column_transformer_sparse_stacking"],
        "start_line": 475,
        "end_line": 475,
    }


def test_request_for_modification_one_small_function(django_workspace):
    plan_to_code = PlanToCode(
        id=0, _workspace=django_workspace, initial_message="Test initial message"
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.modification,
        file_path="django/db/models/query_utils.py",
        start_line=43,
        end_line=58,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "django/db/models/query_utils.py",
        "change_type": "modification",
        "span_ids": ["Q._combine"],
        "start_line": 43,
        "end_line": 58,
    }

    plan_to_code = PlanToCode(
        id=0,
        _workspace=django_workspace,
        initial_message="Test initial message",
        min_tokens_in_edit_prompt=200,
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "django/db/models/query_utils.py",
        "change_type": "modification",
        "span_ids": ["Q.__init__", "Q._combine", "Q.__or__"],
        "start_line": 40,
        "end_line": 61,
    }


def test_unexpected_end_line(sympy_workspace):
    plan_to_code = PlanToCode(
        id=0, _workspace=sympy_workspace, initial_message="Test initial message"
    )

    print(plan_to_code.file_context.create_prompt(show_line_numbers=True))
    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.modification,
        file_path="sympy/sets/sets.py",
        start_line=1774,
        end_line=1894,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sympy/sets/sets.py",
        "change_type": "modification",
        "span_ids": ["imageset"],
        "start_line": 1774,
        "end_line": 1896,
    }

    plan_to_code = PlanToCode(
        id=0,
        _workspace=sympy_workspace,
        max_tokens_in_edit_prompt=500,
        initial_message="Test initial message",
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "retry"

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.modification,
        file_path="sympy/sets/sets.py",
        start_line=1864,
        end_line=1894,
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "sympy/sets/sets.py",
        "change_type": "modification",
        "span_ids": ["imageset"],
        "start_line": 1859,
        "end_line": 1896,
    }


def test_include_full_blocks_in_prompt():
    instance = get_moatless_instance("django__django-13768", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "django/dispatch/dispatcher.py", ["Signal", "Signal.__init__"]
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.addition,
        file_path="django/dispatch/dispatcher.py",
        start_line=22,
        end_line=23,
    )

    plan_to_code = PlanToCode(
        id=0,
        _workspace=workspace,
        max_tokens_in_edit_prompt=500,
        initial_message="Test initial message",
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "file_path": "django/dispatch/dispatcher.py",
        "change_type": "modification",
        "span_ids": ["Signal"],
        "start_line": 21,
        "end_line": 29,
    }


def test_verify_get_start_and_end_lines():
    instance = get_moatless_instance("pytest-dev__pytest-11143", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "src/_pytest/assertion/rewrite.py",
        ["AssertionRewriter", "AssertionRewriter.is_rewrite_disabled"],
    )

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="",
        change_type=ChangeType.modification,
        file_path="src/_pytest/assertion/rewrite.py",
        start_line=745,
        end_line=746,
    )

    plan_to_code = PlanToCode(
        id=0,
        _workspace=workspace,
        max_tokens_in_edit_prompt=500,
        initial_message="Test initial message",
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        "instructions": "Fix",
        "pseudo_code": "",
        "end_line": 753,
        "change_type": "modification",
        "file_path": "src/_pytest/assertion/rewrite.py",
        "span_ids": [
            "AssertionRewriter.is_rewrite_disabled",
            "AssertionRewriter.variable",
        ],
        "start_line": 744,
    }


def test_action_dump():
    take_action = PlanRequest(
        action=Finish(scratch_pad="", reason="Task completed successfully")
    )
    dump = take_action.model_dump()
    assert dump == {
        "action": {"scratch_pad": "", "reason": "Task completed successfully"},
        "action_name": "Finish",
    }

    action = PlanRequest.model_validate(dump)
    assert isinstance(action.action, Finish)
    assert action == take_action
