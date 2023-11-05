from ghostcoder.codeblocks import CodeBlockType
from ghostcoder.codeblocks.parser.javascript import JavaScriptParser

parser = JavaScriptParser("javascript")


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
});
    """
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


def test_function():
    content = """const isSaveButtonDisabled = () =>
    this.props.disabled;
"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_tree_sitter_type=True))

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