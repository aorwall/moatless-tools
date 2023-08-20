from codeblocks.codeblocks import CodeBlockType, CodeBlock

code = CodeBlock(content="",
                 type=CodeBlockType.MODULE,
                 children=[CodeBlock(
                     content="public class TreeSitterExample ",
                     type=CodeBlockType.CLASS,
                     pre_code="",
                     children=[
                         CodeBlock(content="{",
                                   type=CodeBlockType.BLOCK_DELIMITER,
                                   pre_code=""),
                         CodeBlock(
                             content="int myVariable = 10;",
                             type=CodeBlockType.CODE,
                             pre_code="\n\n    "),
                         CodeBlock(
                             content="public void myMethod(int parameter) ",
                             type=CodeBlockType.FUNCTION,
                             pre_code="\n\n    ",
                             children=[
                                 CodeBlock(
                                     content="{",
                                     type=CodeBlockType.BLOCK_DELIMITER,
                                     pre_code=""),
                                 CodeBlock(
                                     content="myVariable = parameter;",
                                     type=CodeBlockType.CODE,
                                     pre_code="\n        "),
                                 CodeBlock(
                                     content="}",
                                     pre_code="\n    ",
                                     type=CodeBlockType.BLOCK_DELIMITER)]),
                         CodeBlock(content="}",
                                   type=CodeBlockType.BLOCK_DELIMITER,
                                   pre_code="\n")
                     ])])


def test_to_string():
    assert code.to_string() == """public class TreeSitterExample {

    int myVariable = 10;

    public void myMethod(int parameter) {
        myVariable = parameter;
    }
}"""


def test_find_nested_matching_block():
    find_block = CodeBlock(content="",
                           type=CodeBlockType.MODULE,
                           children=[CodeBlock(
                               content="public void myMethod(int parameter) ",
                               type=CodeBlockType.FUNCTION,
                               pre_code="\n\n    ",
                               children=[
                                   CodeBlock(
                                       content="{",
                                       type=CodeBlockType.BLOCK_DELIMITER,
                                       pre_code=""),
                                   CodeBlock(
                                       content="myVariable = parameter;",
                                       type=CodeBlockType.CODE,
                                       pre_code="\n        "),
                                   CodeBlock(
                                       content="}",
                                       pre_code="\n    ",
                                       type=CodeBlockType.BLOCK_DELIMITER)])])

    assert code.find_nested_matching_block(find_block) == code.children[0].children[2]
