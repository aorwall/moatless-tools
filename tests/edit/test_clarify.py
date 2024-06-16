from moatless.edit.clarify import ClarifyCodeChange, LineNumberClarification
from moatless.loop import AgenticLoop
from moatless.types import ActionResponse
from utils import create_workspace


def create_clarify(
    mocker, instance_id: str, file_path: str, span_id: str, instructions: str = ""
):
    clarify = ClarifyCodeChange(
        instructions=instructions, file_path=file_path, span_id=span_id
    )
    mock_loop = mocker.create_autospec(AgenticLoop)
    mock_loop.workspace = create_workspace(instance_id)
    mock_loop.workspace.file_context.add_span_to_context(
        file_path=file_path, span_id=span_id
    )
    clarify._set_loop(mock_loop)
    return clarify, mock_loop


def test_line_span_in_end_of_class(mocker):
    instance_id = "scikit-learn__scikit-learn-13439"
    file_path = "sklearn/pipeline.py"
    span_id = "Pipeline"

    coding, mock_loop = create_clarify(
        mocker, instance_id=instance_id, file_path=file_path, span_id=span_id
    )

    request = LineNumberClarification(start_line=562, end_line=563, thoughts="")

    response = coding.handle_action(request)
    assert response == ActionResponse(
        trigger="edit_code",
        output={
            "instructions": "",
            "file_path": "sklearn/pipeline.py",
            "span_id": "Pipeline",
            "start_line": 559,
            "end_line": 562,
        },
    )


def test_impl_span(mocker):
    instance_id = "django__django-10914"
    file_path = "django/conf/global_settings.py"
    span_id = "impl:105"
    start_line = 307
    end_line = 307

    coding, mock_loop = create_clarify(
        mocker, instance_id=instance_id, file_path=file_path, span_id=span_id
    )

    request = LineNumberClarification(
        start_line=start_line, end_line=end_line, thoughts=""
    )

    response = coding.handle_action(request)
    assert response == ActionResponse(
        trigger="edit_code",
        output={
            "instructions": "",
            "file_path": "django/conf/global_settings.py",
            "span_id": "impl:105",
            "start_line": 303,
            "end_line": 311,
        },
    )


def test_line_span_in_class(mocker):
    instance_id = "psf__requests-863"
    file_path = "requests/models.py"
    span_id = "Request"
    start_line = 151
    end_line = 153

    coding, mock_loop = create_clarify(
        mocker, instance_id=instance_id, file_path=file_path, span_id=span_id
    )

    request = LineNumberClarification(
        start_line=start_line, end_line=end_line, thoughts=""
    )

    response = coding.handle_action(request)
    assert response == ActionResponse(
        trigger="edit_code",
        output={
            "instructions": "",
            "file_path": "requests/models.py",
            "span_id": "Request",
            "start_line": 147,
            "end_line": 157,
        },
    )
