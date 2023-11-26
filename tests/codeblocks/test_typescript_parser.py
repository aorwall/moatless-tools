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

    print(codeblock.to_tree(include_tree_sitter_type=True, only_identifiers=False))

    assert codeblock.to_tree() == expected_tree
    assert codeblock.to_string() == content

def test_import():
    parser = TypeScriptParser()
    codeblock = parser.parse("""
const MyComponent: React.FC<MyComponentProps> = ({ initialCount = 0 }) => {
  const [count, setCount] = useState(initialCount);

}

""")
    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_assignment():
    parser = TypeScriptParser()
    codeblock = parser.parse("let a = 1 + 2;")
    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_new_date():
    parser = TypeScriptParser()
    codeblock = parser.parse("let a = new Date();")
    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_type_annotations():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myVariable: Array<number>;")
    print(codeblock.to_tree(include_tree_sitter_type=True))
    assert codeblock.to_tree() == """ 0 module ``
  1 code `myVariable`
   2 block_delimiter `;`
"""


def test_nullable_var():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myNullableVar: number | null = null;")
    print(codeblock.to_tree(include_tree_sitter_type=False))
    assert codeblock.to_tree() == """ 0 module ``
  1 code `myNullableVar`
   2 code `=`
   2 code `null`
   2 block_delimiter `;`
"""

def test_lexical_declaration():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myVariable: number;")
    assert codeblock.to_tree(include_tree_sitter_type=False) == """ 0 module ``
  1 code `let`
   2 code `myVariable`
   2 code `: number`
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

    assert codeblock.to_tree(only_identifiers=False) == """ 0 module ``
  1 class `const Foo = () =>`
   2 block_delimiter `{`
   2 block_delimiter `}`
"""

def test_lexical_declaration_function():
    parser = TypeScriptParser()
    codeblock = parser.parse("const getFoo = useCallback(() => {})")
    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == """ 0 module ``
  1 function `getFoo`
   2 block_delimiter `{`
   2 block_delimiter `}`
   2 block_delimiter `)`
"""

def test_export_react_component():
    parser = TypeScriptParser()
    codeblock = parser.parse("export const Foo: React.FC<Props> = ({\n  bar \n}) => {}")
    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert len(codeblock.children) == 1
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[0].content == "export const Foo: React.FC<Props> = ({\n  bar \n}) => "
    assert codeblock.children[0].identifier == "Foo"

def test_set_const():
    parser = TypeScriptParser("typescript")

    content = """const SEARCH_CARE_UNITS_THRESHOLD = 5"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))


def test_react_use_functions():
    parser = TypeScriptParser("typescript")

    content = """const handleSearchInput = useMemo(() => {
      const handler: React.ChangeEventHandler<HTMLInputElement> = ({ target: { value } }) => {
        setSearchValue(value);
      };

      return handler;
    }, []);"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))

    content = """const position = useRef<GeolocationPosition>();"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))

    content = """const[sortBy, setSortBy] = useState <SortBy> ();"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))




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

def test_declarations():
    content = """
class Foo extends Bar {
  constructor() {
    super();
  }
}

interface FooInterface {}

const fooFunction = () =>
  console.log('foo');"""

    parser = TypeScriptParser("typescript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert len(codeblock.children) == 3
    assert codeblock.children[0].identifier == "Foo"
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[1].identifier == "FooInterface"
    assert codeblock.children[1].type == CodeBlockType.CLASS
    assert codeblock.children[2].identifier == "fooFunction"
    assert codeblock.children[2].type == CodeBlockType.FUNCTION

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
    assert codeblock.children[0].identifier == "Foo"
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[1].identifier == "FooInterface"
    assert codeblock.children[1].type == CodeBlockType.CLASS
    assert codeblock.children[2].identifier == "fooFunction"
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
    assert codeblock.children[0].identifier == "FooComponent"
    assert codeblock.children[0].content == "const FooComponent = forwardRef<HTMLDivElement, Props>(\n  (\n    { bar }\n  ) =>"
    assert codeblock.to_string() == content

def test_console_log():
    content = """console.log('foo');"""
    parser = TypeScriptParser("typescript")
    codeblock = parser.parse(content)

    assert codeblock.to_tree() == """ 0 module ``
  1 code `console.log`
   2 block_delimiter `(`
   2 code `'foo'`
   2 block_delimiter `)`
   2 block_delimiter `;`
"""


def test_jest_test():
    content = """
describe('Foo', () => {
  let FooMock: SpyInstance;
  
  beforeAll(() => {
    FooMock = vi.spyOn(FooTable, 'FooTable');
  });

  beforeEach(vi.clearAllMocks);

  afterAll(vi.resetAllMocks);

  it('should do something', () => {
    renderWithProviders(<Foo {...MOCK_PROPS} />);
    expect(FooMock).toHaveBeenCalled();
  });
});
"""
    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)
    print(codeblock.to_tree(only_identifiers=False))