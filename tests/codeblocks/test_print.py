from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print import print_by_line_numbers, print_by_block_path


def test_print_by_line_numbers():
    file_path = "/tmp/repos/pytest-dev_pytest/src/_pytest/faulthandler.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    line_numbers = [
        (1, 36),
        (54, 205)
    ]

    print(print_by_line_numbers(codeblock, line_numbers))


def test_print_by_block_path():
    file_path = "../data/python/replace_function/original.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    printed = print_by_block_path(codeblock, ["MatrixShaping", "_eval_col_insert"])

    print("Printed by block path:")
    print(printed)
