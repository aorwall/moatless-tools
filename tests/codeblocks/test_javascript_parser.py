from ghostcoder.codeblocks import CodeBlockType
from ghostcoder.codeblocks.parser.javascript import JavaScriptParser

parser = JavaScriptParser("javascript", debug=False)


def test_javascript_treesitter_types():
    with open("javascript/treesitter_types.js", "r") as f:
        content = f.read()
    with open("javascript/treesitter_types.txt", "r") as f:
        expected_tree = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_tree() == expected_tree
    assert codeblock.to_string() == content


def test_javascript_async_function():
    with open("javascript/async_function.js", "r") as f:
        content = f.read()
    with open("javascript/async_function.txt", "r") as f:
        expected_tree = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=False))

    assert codeblock.to_tree() == expected_tree
    assert codeblock.to_string() == content


def test_javascript_object_literal():
    content = """const obj = {
  key: 'value',
  method() {
    return 'This is a method';
  }
};
"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_if_statement():
    content = """if (number > 5) {
  console.log('Number is greater than 5');
} else if (number === 5) {
  console.log('Number is 5 ');
} else {
  console.log('Number is smaller than 5');
}
"""
    parser = JavaScriptParser("javascript", debug=True)

    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))


def test_switch_statement():
    content = """switch (number) {
  case 10:
    console.log('Number is 10');
    break;
  default:
    console.log('Number is not 10');
    break;
}"""

    parser = JavaScriptParser("javascript", debug=True)

    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_old_school_var():
    content = """var foo = 1;"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))


def test_test_function():
    content = """describe('Accessibility', () => {
  it('should not have traceable title', async () => {
    await renderFoo();
    await waitFor(() => {
      expect(
        screen.getByTestId('foo-container')
      ).toHaveAttribute('tabIndex', '-1');
    });
  });
});"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content

def test_function_field_definition():
    content = """class FooForm extends Component {
    isContactFormValid = () =>
        isValidPhone(this.state.mobileNumber) && this.validEmail(this.state.email);
}"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))


def test_lexical_declaration_function():
    content = """const isDisabled = () =>
    this.props.disabled;
"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.children[0].type == CodeBlockType.FUNCTION

def test_field_definition_function():
    content = """class Foo extends Component {
  foo = () => {
    return (
      "foo"
    );
  };
};
"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[0].children[1].type == CodeBlockType.FUNCTION

def test_map_and_jsx():
    content = """const baz = foo?.map(
      ({ name, bar }) => (
        <Foo key={bar}>
          {name}
          <FooButton
            onClick={() =>
              this.setState({
                foo: "",
              })
            }
          >
          </FooButton>
        </Foo>
      )
    );"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

def test_css():
    content = """const Styled = {
  Subtitle: styled.p`
    color: ${({ theme }) => theme.colors.dark};
    font-size: 0.9em;
  `,
};"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content

def test_constructor():
    content = """class Foo extends Component {

  constructor(props) {
    super(props);
    const {
      foo = ''
    } = props;

    this.state = {
      bar: null
    };
  }
}
"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content


def test_assignment():
    content = """this.state = {
      bar: null
    };"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content

def test_solo_constructor():
    content = """constructor(props) {
  this.state = {
    foo: foo || '',
    // ... other state properties
  };
}"""

    parser = JavaScriptParser("javascript", apply_gpt_tweaks=True)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content
    assert codeblock.type == CodeBlockType.MODULE
    assert codeblock.children[0].type == CodeBlockType.CONSTRUCTOR

def test_solo_function():
    content = """useEffect(() => {

  const foo = async () => {
  };

  foo();
}, []);"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.to_string() == content
    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 1
    assert codeblock.children[0].type == CodeBlockType.FUNCTION

def test_function_indent():
    content ="""  isValid = () => {
    return false;
  };

  isInvalid = () => {
    return true;
  };"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 2
    assert codeblock.children[0].type == CodeBlockType.FUNCTION
    assert codeblock.children[1].type == CodeBlockType.FUNCTION
    assert codeblock.to_string() == content


def test_commented_out():
    content = """this.state = {
        foo: foo || '',
        // ... other state properties
    };"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.type == CodeBlockType.MODULE


def test_array_call():
    content = """
array.forEach((element) => {
  console.log(element);
});
"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.children[0].type == CodeBlockType.BLOCK
    assert codeblock.to_string() == content


def test_root_functions_indent():
    content = """
  componentDidMount() {
    this.setState({
      foo: true,
    });
  }

  checkFoo = () => {
    return false;
  };"""

    parser = JavaScriptParser("javascript", apply_gpt_tweaks=True, debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 2
    assert codeblock.children[0].type == CodeBlockType.FUNCTION
    assert codeblock.children[1].type == CodeBlockType.FUNCTION


def test_const_id():
    content = """const foo = await bar({
});"""
    parser = JavaScriptParser("javascript", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 1
    assert codeblock.children[0].type == CodeBlockType.BLOCK


def test_incorrect_outcommented_code():
    content = """function foo() {
    ...
}"""
    parser = JavaScriptParser("javascript", apply_gpt_tweaks=False)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))
    assert len(codeblock.find_errors()) == 1

    parser = JavaScriptParser("javascript", apply_gpt_tweaks=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_tree_sitter_type=True))
    assert len(codeblock.find_errors()) == 0

    assert codeblock.children[0].children[1].type == CodeBlockType.COMMENTED_OUT_CODE