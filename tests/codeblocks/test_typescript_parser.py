from ghostcoder.codeblocks import CodeBlockType
from ghostcoder.codeblocks.parser.typescript import TypeScriptParser


def test_typescript_treesitter_types():
    with open("typescript/treesitter_types.ts", "r") as f:
        content = f.read()
    with open("typescript/treesitter_types_expected.txt", "r") as f:
        expected_tree = f.read()

    parser = TypeScriptParser()
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == expected_tree


def test_react_tsx():
    with open("typescript/react_component.tsx", "r") as f:
        content = f.read()
    with open("typescript/react_component_expected.txt", "r") as f:
        expected_tree = f.read()

    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == expected_tree
    assert codeblock.to_string() == content

def test_lexical_declaration():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myVariable: number;")
    assert codeblock.to_tree() == """ 0 module ``
  1 code `let myVariable: number`
   2 block_delimiter `;`
"""

    codeblock_assigned = parser.parse("let myVariable: number = 5;")
    assert codeblock_assigned.to_tree() == """ 0 module ``
  1 code `let myVariable: number`
   2 code `=`
   2 code `5`
   2 block_delimiter `;`
"""

def test_lexical_declaration_class():
    parser = TypeScriptParser()
    codeblock = parser.parse("const Foo = () => {}")
    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == """ 0 module ``
  1 class `const Foo = () =>`
   2 block_delimiter `{`
   2 block_delimiter `}`
"""

def test_lexical_declaration_function():
    parser = TypeScriptParser()
    codeblock = parser.parse("const getFoo = useCallback(() => {})")
    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == """ 0 module ``
  1 function `const getFoo = useCallback(() =>`
   2 block_delimiter `{`
   2 block_delimiter `}`
   2 block_delimiter `)`
"""

def test_css():
    content = """
const SeparatorContainer = styled.div`
  hr,
  span {
    flex: 1;
    color: ${({ theme }) => theme.colors.muted.dark};
  }

  hr {
    flex: 2;
  }
`;
"""

    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_export_declarations():
    content = """
export class Foo extends Bar {
  constructor() {
    super();
  }
}

export interface FooInterface {
  foo: 'push' | 'replace';
}

export const fooFunction = () =>
  console.log('foo');"""

    parser = TypeScriptParser("typescript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert len(codeblock.children) == 3
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[1].type == CodeBlockType.CLASS
    assert codeblock.children[2].type == CodeBlockType.FUNCTION


def test_forwardRef_class():
    content = """const FooComponent = forwardRef<HTMLDivElement, Props>(
  (
    { bar }
  ) => {
    
});
"""
    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)

    assert len(codeblock.children) == 2
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[0].content == "const FooComponent = forwardRef<HTMLDivElement, Props>(\n  (\n    { bar }\n  ) =>"
    assert codeblock.to_string() == content
