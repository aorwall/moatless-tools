from moatless.edit.clarify import ClarifyCodeChange, LineNumberClarification
from moatless.edit.edit import EditCode
from moatless.loop import AgenticLoop
from utils import create_workspace


def create_clarify(
    mocker,
    instance_id: str,
    file_path: str,
    span_id: str,
    start_line: int,
    end_line: int,
    instructions: str = "",
):
    clarify = EditCode(
        instructions=instructions,
        file_path=file_path,
        span_id=span_id,
        start_line=start_line,
        end_line=end_line,
    )
    mock_loop = mocker.create_autospec(AgenticLoop)
    mock_loop.workspace = create_workspace(instance_id)
    mock_loop.workspace.file_context.add_span_to_context(
        file_path=file_path, span_id=span_id
    )
    clarify._set_loop(mock_loop)
    return clarify, mock_loop


def test_search_block(mocker):
    instance_id = "scikit-learn__scikit-learn-10297"
    file_path = "sklearn/linear_model/ridge.py"
    span_id = "RidgeClassifierCV.fit"
    start_line = 1342
    end_line = 1377

    coding, mock_loop = create_clarify(
        mocker,
        instance_id=instance_id,
        file_path=file_path,
        span_id=span_id,
        start_line=start_line,
        end_line=end_line,
    )

    # Assert that the first line is correct in the search block
    found_search = False
    for line in coding.messages()[0].content.split("\n"):
        if found_search:
            assert line == "    def fit(self, X, y, sample_weight=None):"
            break
        if "<search>" in line:
            found_search = True
