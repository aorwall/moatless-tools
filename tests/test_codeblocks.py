from code_blocks.codeblocks import CodeBlockType, CodeBlock


def test_to_string():
    code = CodeBlock(
        content="public class TreeSitterExample ",
        type=CodeBlockType.CLASS,
        pre_code="",
        children=[
            CodeBlock(content="{",
                      type=CodeBlockType.BLOCK_DELIMITER,
                      pre_code=""),
            CodeBlock(
                content="int myVariable = 10;",
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
        ])

    assert str(code) == """public class TreeSitterExample {

    int myVariable = 10;

    public void myMethod(int parameter) {
        myVariable = parameter;
    }
}"""