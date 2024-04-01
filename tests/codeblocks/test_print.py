from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print import print_by_line_numbers

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
