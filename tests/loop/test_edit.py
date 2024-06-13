from moatless.loop.base import Loop, Rejected, Finished
from moatless.loop.clarify import ClarifyCodeChange
from moatless.loop.writecode import PlanToCode
from moatless.loop.old import RequestForChangeRequest
from moatless.loop.writecode import WriteCode
from moatless.types import RejectRequest, FinishRequest
from .utils import create_workspace


def create_coder(mocker, instance_id: str, file_path: str, span_ids: set[str], instructions: str = ""):
    coding = PlanToCode(
        instructions=instructions
    )
    mock_loop = mocker.create_autospec(Loop)
    mock_loop.workspace = create_workspace(instance_id)
    mock_loop.workspace.file_context.add_spans_to_context(
        file_path=file_path,
        span_ids=span_ids
    )
    coding._set_loop(mock_loop)
    return coding, mock_loop


def test_rejection(mocker):
    coding = PlanToCode(
        instructions="Write a function that takes a list of integers and returns the sum of the list."
    )
    mock_loop = mocker.create_autospec(Loop)
    coding._set_loop(mock_loop)

    reason = "I don't know how to do this."
    coding.handle_action(RejectRequest(reason=reason))
    mock_loop.transition_to.assert_called_once_with(Rejected(reason=reason))


def test_finish(mocker):
    coding = PlanToCode(
        instructions="Write a function that takes a list of integers and returns the sum of the list."
    )
    mock_loop = mocker.create_autospec(Loop)
    coding._set_loop(mock_loop)

    reason = "I'm done."
    coding.handle_action(FinishRequest(reason=reason))
    mock_loop.transition_to.assert_called_once_with(Finished(reason=reason))


def test_message_history(mocker):
    instance_id = "astropy__astropy-14995"
    file_path = "astropy/nddata/mixins/ndarithmetic.py"
    span_id = "NDArithmeticMixin._arithmetic_mask"

    coding, mock_loop = create_coder(
        mocker,
        instructions="Update this",
        instance_id=instance_id,
        file_path=file_path,
        span_ids={span_id}
    )

    messages = coding.message_history()
    assert len(messages) == 1
    assert "Update this" in messages[0]["content"]
    assert "def _arithmetic_mask(self, operation, operand, handle_mask, axis=None, **kwds)" in messages[0]["content"]


def test_request_for_change(mocker):
    instance_id = "astropy__astropy-14995"
    file_path = "astropy/nddata/mixins/ndarithmetic.py"
    span_id = "NDArithmeticMixin._arithmetic_mask"

    coding, mock_loop = create_coder(
        mocker,
        instance_id=instance_id,
        file_path=file_path,
        span_ids={span_id}
    )

    request = RequestForChangeRequest(
        file_path=file_path,
        span_id=span_id,
        description="I want to update this"
    )

    response = coding.handle_action(request)
    assert response is None
    mock_loop.transition_to.assert_called_once_with(WriteCode(description=request.description, file_path=request.file_path, span_id=request.span_id))


def test_clarify_changes(mocker):
    instance_id = "django__django-11001"
    file_path = "django/db/models/sql/compiler.py"
    span_id = "SQLCompiler.get_order_by"

    coding, mock_loop = create_coder(
        mocker,
        instance_id=instance_id,
        file_path=file_path,
        span_ids={span_id}
    )

    request = RequestForChangeRequest(
        file_path=file_path,
        span_id=span_id,
        description="I want to update this"
    )

    response = coding.handle_action(request)
    assert response is None
    mock_loop.transition_to.assert_called_once_with(ClarifyCodeChange(description=request.description, file_path=file_path, span_id=span_id))
