from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import PathTree
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_by_line_numbers, print_by_block_path, print_by_block_paths, \
    BlockMarker, print_block


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



def test_print_by_block_paths_and_marker():
    file_path = "/tmp/repos/marshmallow-code_marshmallow/src/marshmallow/fields.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    block_paths = [["List","__init__"], ["List", "_bind_to_schema"]]

    list_block = codeblock.find_by_path(["List"])
    print(list_block.sum_tokens())

    print(list_block.to_tree(include_references=True))

    printed = print_block(codeblock,
                          path_tree=PathTree.from_block_paths(block_paths),
                          show_types=[CodeBlockType.CLASS, CodeBlockType.FUNCTION],
                          block_marker=BlockMarker.COMMENT)

    print("Printed by block path:")
    print(printed)
