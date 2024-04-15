import json
import logging
import shutil
import tempfile
from typing import Optional, List

from litellm import mock_completion
from pydantic import BaseModel

from moatless.codeblocks.parser.python import PythonParser
from moatless.coder import Coder
from moatless.coder.write_code import write_code
from moatless.types import Span, ContextFile, CodingTask


logging.getLogger("LiteLLM").setLevel(logging.INFO)


class UpdateCodeCase(BaseModel):
    content: str
    span: Optional[Span] = None
    action: str = "update"

    def __hash__(self):
        return hash(self.content) + hash(self.span) + hash(self.action)


def _test_instance(instance_id: str):
    with open(f"data/python/regressions/{instance_id}/expected.py", "r") as f:
        expected_updated_code = f.read()

    with open(f"data/python/regressions/{instance_id}/updates.json", "r") as f:
        dicts = json.load(f)
        updated_block_cases = [UpdateCodeCase(**d) for d in dicts]

    temp_dir = tempfile.TemporaryDirectory()
    shutil.copy2(
        f"data/python/regressions/{instance_id}/original.py",
        f"{temp_dir.name}/original.py",
    )

    _run_regression_test(updated_block_cases, expected_updated_code, temp_dir.name)


def _run_regression_test(
    updated_block_cases: List[UpdateCodeCase], expected_updated_code: str, dir: str
):
    spans = []
    tasks = []
    mocked_code_responses = []
    for update_code_case in updated_block_cases:
        tasks.append(
            {
                "file_path": "original.py",
                "instructions": "",
                "span_id": update_code_case.span.span_id,
                "action": update_code_case.action,
            }
        )

        spans.append(update_code_case.span)
        mocked_code_responses.append(f"```python\n{update_code_case.content}\n```")

    coder = Coder(
        repo_path=dir,
        requirement="",
        files=[ContextFile(file_path="original.py", spans=spans)],
    )

    mocked_plan = json.dumps({"tasks": tasks}, indent=2)

    coder.run(mock_responses=[mocked_plan] + mocked_code_responses)

    with open(f"{dir}/original.py", "r") as f:
        updated_code = f.read()

    assert updated_code.strip() == expected_updated_code.strip()


def test_instance():
    _test_instance("django__django-11905")
