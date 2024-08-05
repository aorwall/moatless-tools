from moatless.file_context import RankedFileSpan
from moatless.find import IdentifyCode


def test_model_dump():
    identify = IdentifyCode(
        ranked_spans=[
            RankedFileSpan(
                file_path="file1.py",
                span_id="span1",
                rank=1,
            ),
            RankedFileSpan(file_path="file2.py", span_id="span2", rank=2, tokens=50),
        ]
    )

    assert identify.model_dump() == {
        "include_message_history": False,
        "model": None,
        "temperature": 0.0,
        "max_tokens": 1000,
        "max_iterations": None,
        "ranked_spans": [
            {"file_path": "file1.py", "span_id": "span1", "rank": 1, "tokens": 0},
            {"file_path": "file2.py", "span_id": "span2", "rank": 2, "tokens": 50},
        ],
        "expand_context": True,
        "max_prompt_file_tokens": 4000,
        "name": "IdentifyCode",
    }
