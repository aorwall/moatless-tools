import os

from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.base import Coder
from moatless.coder.code_utils import do_diff
from moatless.coder.write_code import write_code
from moatless.types import Span
from tests.test_utils import assert_diff


def test_update_span():
    with open("data/python/sympy__sympy-18698/polytools.py") as f:
        original_code = f.read()

    with open("data/python/sympy__sympy-18698/update_span.py") as f:
        updated_code = f.read()

    with open("data/python/sympy__sympy-18698/expected_span_diff.txt") as f:
        expected_span_diff = f.read()

    parser = PythonParser()
    original_codeblock = parser.parse(original_code)
    updated_codeblock = parser.parse(updated_code)

    result = write_code(
        original_codeblock, updated_codeblock, original_codeblock, Span(1, 57), "update"
    )

    assert_diff(original_code, result, expected_span_diff)


def test_update_function():
    with open("data/python/sympy__sympy-18698/polytools.py") as f:
        original_code = f.read()

    with open("data/python/sympy__sympy-18698/update_function.py") as f:
        updated_code = f.read()

    with open("data/python/sympy__sympy-18698/expected_function_diff.txt") as f:
        expected_span_diff = f.read()

    parser = PythonParser()
    original_codeblock = parser.parse(original_code)
    updated_codeblock = parser.parse(updated_code)

    expected_codeblock = original_codeblock.find_by_path(["_symbolic_factor_list"])

    result = write_code(
        original_codeblock, updated_codeblock, expected_codeblock, action="update"
    )

    assert_diff(original_code, result, expected_span_diff)
