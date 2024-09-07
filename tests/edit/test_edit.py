import pytest
from unittest.mock import Mock, patch

from moatless import AgenticLoop
from moatless.benchmark.swebench import load_instance, create_workspace
from moatless.benchmark.utils import get_moatless_instance
from moatless.edit.edit import EditCode
from moatless.repository.file import UpdateResult
from moatless.schema import ChangeType
from moatless.state import StateOutcome, Content
from moatless.settings import Settings
from moatless.trajectory import Trajectory
from moatless.workspace import Workspace
from moatless.file_context import FileContext
from moatless.repository import CodeFile, GitRepository


class TestEditCode:
    @pytest.fixture
    def edit_code(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)

        return EditCode(
            id=1,
            instructions="Update function",
            file_path="test.py",
            span_id="span1",
            verify=False,
            start_line=1,
            end_line=5,
            _workspace=mock_workspace,
            model="gpt-3.5-turbo",
        )

    def test_required_fields(self, edit_code: EditCode):
        assert edit_code.required_fields() == {
            "instructions",
            "file_path",
            "start_line",
            "end_line",
        }

    @patch("moatless.edit.edit.EditCode.file_context")
    def test_execute_action_reject(self, mock_file_context, edit_code: EditCode):
        content = Content(content="<reject>Cannot complete the task</reject>")

        response = edit_code._execute_action(content)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "reject"
        assert response.output["message"] == "Cannot complete the task"

    # FIXME
    @pytest.mark.skip
    @patch("moatless.edit.edit.EditCode.file_context")
    def test_execute_action_edit_code(self, mock_file_context, edit_code: EditCode):
        mock_file = Mock(spec=CodeFile)
        mock_context_file = Mock()
        mock_context_file.file = mock_file

        mock_file_context.get_file.return_value = mock_context_file

        content = Content(content="<replace>updated code</replace>")

        response = edit_code._execute_action(content)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "finish"
        assert "Applied the change to test.py." in response.output["message"]
        assert response.output["diff"] == "diff"

        mock_context_file.update_content_by_line_numbers.assert_called_once()

    # FIXME
    @pytest.mark.skip
    @patch("moatless.edit.edit.EditCode.file_context")
    def test_execute_action_retry(self, mock_file_context, edit_code: EditCode):
        mock_file = Mock(spec=CodeFile)
        mock_file.update_content_by_line_numbers.return_value = Mock(
            diff=None, updated=False
        )
        mock_context_file = Mock()
        mock_context_file.file = mock_file
        mock_file_context.get_file.return_value = mock_context_file

        content = Content(content="<replace>unchanged code</replace>")

        response = edit_code._execute_action(content)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "retry"
        assert (
            "The code in the replace tag is the same as in the search"
            in response.retry_message
        )

    def test_system_prompt(self, edit_code: EditCode):
        system_prompt = edit_code.system_prompt()

        assert (
            "You are autonomous AI assisistant with superior programming skills."
            in system_prompt
        )

    def test_action_type(self, edit_code: EditCode):
        assert edit_code.action_type() is None

    def test_stop_words(self, edit_code: EditCode):
        assert edit_code.stop_words() == ["</replace>"]


def test_expect_failed_edit():
    def create_state():
        instance = load_instance("pytest-dev__pytest-7373")
        workspace = create_workspace(instance)
        workspace.file_context.add_spans_to_context(
            "src/_pytest/mark/evaluate.py", ["imports", "cached_eval", "MarkEvaluator"]
        )

        edit_code = EditCode(
            id=0,
            _workspace=workspace,
            verify=False,
            instructions="Test",
            file_path="src/_pytest/mark/evaluate.py",
            span_id="cached_eval",
            start_line=21,
            end_line=31,
        )
        edit_code.init()
        return edit_code

    edit_code = create_state()
    outcome = edit_code.execute(Content(content="<replace>\n"))
    assert outcome.trigger == "finish"
    assert "-def cached_eval" in outcome.output["diff"]

    edit_code = create_state()
    outcome = edit_code.execute(
        Content(
            content="To remove the `cached_eval` function as requested, I'll simply return an empty <replace> tag:\n\n<replace>\n"
        )
    )
    assert outcome.trigger == "finish"
    assert "-def cached_eval" in outcome.output["diff"]


def test_incomplete_replace():
    instance = load_instance("sympy__sympy-16988")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "sympy/sets/sets.py", ["imports", "imageset"]
    )

    edit_code = EditCode(
        id=0,
        _workspace=workspace,
        instructions="Test",
        change_type=ChangeType.modification,
        file_path="sympy/sets/sets.py",
        start_line=1774,
        end_line=1896,
    )

    replace_content = """"<replace>\n    if not set_list:\n        return S.EmptySet\n\n    if len(set_list) == 1:\n        set = set_list[0]\n        try:\n            # TypeError if arg count != set dimensions\n            r = set_function(f, set)\n            if r is None:\n                raise TypeError\n            if not r:\n                return r\n        except TypeError:\n            r = ImageSet(f, set)\n        if isinstance(r, ImageSet):\n            f, set = r.args\n\n        if f.variables[0] == f.expr:\n            return set\n\n        if isinstance(set, ImageSet):\n            if len(set.lamda.variables) == 1 and len(f.variables) == 1:\n                x = set.lamda.variables[0]\n                y = f.variables[0]\n                return imageset(\n                    Lambda(x, f.expr.subs(y, set.lamda.expr)),\n                    set.base_set)\n\n        if r is not None:\n            return r\n\n    return ImageSet(f, *set_list)\n"""

    outcome = edit_code.execute(Content(content=replace_content))
    assert outcome.trigger == "retry"


def test_invalid_replacement():
    instance = get_moatless_instance("django__django-13768", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context(
        "django/dispatch/dispatcher.py", ["Signal", "Signal.__init__"]
    )

    edit_code = EditCode(
        id=0,
        _workspace=workspace,
        instructions="Test",
        change_type=ChangeType.modification,
        file_path="django/dispatch/dispatcher.py",
        start_line=21,
        end_line=28,
    )

    replace_content = """"<replace>
class Signal:
    \"""
    Base class for all signals

    Internal attributes:

        receivers
            { receiverkey (id) : weakref(receiver) }
    \"""
    logger = logging.getLogger('django.dispatch')
</replace>"""

    outcome = edit_code.execute(Content(content=replace_content))
    assert outcome.trigger == "retry"
