from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import PathTree
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import (
    print_by_line_numbers,
    print_by_block_path,
    print_by_block_paths,
    SpanMarker,
    print_block,
)
from moatless.types import Span


def test_print_by_line_numbers():
    file_path = "data/python/print_block/schema.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    line_numbers = [Span(1, 23), Span(151, 152)]

    print(print_by_line_numbers(codeblock, line_numbers))


def test_print_with_references():
    file_path = "data/python/marshmallow-code__marshmallow-1343/schema.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    line_numbers = [Span(1, 23), Span(151, 152)]

    print(
        print_by_block_paths(
            codeblock,
            block_paths=[["BaseSchema", "_invoke_field_validators"]],
            include_references=True,
        )
    )


def test_print_with_comments():
    file_path = "data/python/print_block/schema.py"

    parser = PythonParser()
    with open(file_path) as f:
        content = f.read()
    codeblock = parser.parse(content)

    printed = print_block(codeblock)

    print(printed)

    assert printed == content


def test_print_by_block_path():
    file_path = "data/python/replace_function/original.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    printed = print_by_block_path(codeblock, ["MatrixShaping", "_eval_col_insert"])

    print("Printed by block path:")
    print(printed)


def test_print_by_block_path_2():
    file_path = "data/python/print_block/schema.py"
    expected_file_path = "data/python/print_block/schema_expected_by_block_path.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    with open(expected_file_path) as f:
        expected = f.read()

    printed = print_by_block_path(codeblock, ["BaseSchema", "_invoke_field_validators"])

    print("Printed by block path:")
    print(printed)

    assert printed == expected


def test_print_by_block_paths_and_marker():
    file_path = "/tmp/repos/marshmallow-code_marshmallow/src/marshmallow/fields.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    block_paths = [["List", "__init__"], ["List", "_bind_to_schema"]]

    list_block = codeblock.find_by_path(["List"])
    print(list_block.sum_tokens())

    print(list_block.to_tree(include_references=True))

    printed = print_block(
        codeblock,
        path_tree=PathTree.from_block_paths(block_paths),
        block_marker=SpanMarker.COMMENT,
    )

    print("Printed by block path:")
    print(printed)


def test_print_by_line_numbers_astropy_wcs():
    file_path = "data/python/astropy__astropy-7746/wcs.py"

    parser = PythonParser()
    with open(file_path) as f:
        codeblock = parser.parse(f.read())

    line_numbers = [
        Span(start_line=1418, end_line=1556),
        Span(start_line=1717, end_line=1814),
    ]

    print(
        print_by_line_numbers(
            codeblock, line_numbers, block_marker=SpanMarker.TAG, show_line_numbers=True
        )
    )
