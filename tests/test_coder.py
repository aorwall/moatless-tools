import os

from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.base import Coder


def test_update_part_of_code_block():
    with open("data/python/marshmallow-code__marshmallow-1343/schema.py") as f:
        original_code = f.read()

    with open("data/python/marshmallow-code__marshmallow-1343/schema_update.py") as f:
        update = f.read()

    coder = Coder(repo_path=None)

    parser = PythonParser()
    original_codeblock = parser.parse(original_code)

    block_path = ["BaseSchema", "_invoke_field_validators"]

    result = coder._write_code(original_codeblock, block_path, update)

    print(result)


def test_regression():
    dir = "data/python/regressions"
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for subdir in subdirs:
        print(f"Running test for {subdir}")

        with open(f"{subdir}/original.py") as f:
            original_code = f.read()
        with open(f"{subdir}/update.py") as f:
            update = f.read()
        with open(f"{subdir}/diff.txt") as f:
            expected_diff = f.read()
        with open(f"{subdir}/expected.py") as f:
            expected = f.read()
        with open(f"{subdir}/block_path.txt") as f:
            block_path = f.read().split(".")

        coder = Coder(repo_path=None)

        parser = PythonParser()
        original_codeblock = parser.parse(original_code)

        result = coder._write_code(original_codeblock, block_path, update)

        assert result.content == expected
        diff = do_diff("file", original_code, result.content)

        print(diff == expected_diff)
