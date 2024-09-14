from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.parser.javascript import JavaScriptParser

parser = JavaScriptParser("javascript", debug=True)


def test_javascript_treesitter_types():
    with open("javascript/treesitter_types.js", "r") as f:
        content = f.read()
    with open("javascript/treesitter_types.txt", "r") as f:
        expected_tree = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True, use_colors=True))

    assert codeblock.to_tree(use_colors=False) == expected_tree
    assert codeblock.to_string() == content


def test_javascript_async_function():
    with open("javascript/async_function.js", "r") as f:
        content = f.read()
    with open("javascript/async_function.txt", "r") as f:
        expected_tree = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=False))

    assert codeblock.to_tree(use_colors=False) == expected_tree
    assert codeblock.to_string() == content


def test_class():
    content = """class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  introduce() {
    age = this.age;
    return `Hi, I'm ${this.name} and I'm ${age} years old.`;
  }
}"""

    codeblock = parser.parse(content)
    print(codeblock.to_tree(include_references=True, include_parameters=True))

    assert codeblock.to_string() == content
    assert codeblock.to_tree(use_colors=False, include_references=True, include_parameters=True) == """ 0 module ``
  1 class `Person`
   2 constructor `constructor(name, age)`
    3 assignment `this.name` references: (this.name) [local:dependency] -> name
    3 assignment `this.age` references: (this.age) [local:dependency] -> age
   2 function `introduce`
    3 assignment `age` references: (age) [class:dependency] -> age
    3 statement `return`
     4 code ``Hi, I'm ${this.name} and I'm ${age} years old.`` references: () [class:dependency] -> name, () [local:dependency] -> age
"""


def test_javascript_object_literal():
    content = """const obj = {
  key: 'value',
  method() {
    return 'This is a method';
  }
};
"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))

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
    print(codeblock.to_tree(debug=True))


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
    print(codeblock.to_tree(debug=True))

def test_old_school_var():
    content = """var foo = 1;"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))


def test_jest_test_suite():
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

    print(codeblock.to_tree(debug=True, include_references=True))

    assert codeblock.to_string() == content

def test_call_expressions():
    content = """expect()
expect(screen)
expect(screen, foo(), this.bar)
expect(screen.getByTestId('foo-container'))
foo().bar()
expect(screen).toHaveAttribute('tabIndex', '-1');"""

    parser = JavaScriptParser("javascript", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True))

    assert codeblock.to_string() == content

def test_function_field_definition():
    content = """class FooForm extends Component {
    isContactFormValid = (foo) =>
        isValidPhone(this.state.mobileNumber) && this.validEmail(this.state.email);
}"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(use_colors=False, debug=True, include_references=True, include_parameters=True))


def test_lexical_declaration_function():
    content = """const isDisabled = () =>
    this.props.disabled;
"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

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

    #content = "const baz = foo?.map()"
    parser = JavaScriptParser("javascript", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

def test_map():
    content = "const baz = foo?.map()"
    parser = JavaScriptParser("javascript", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

def test_css():
    content = """const Styled = {
  Subtitle: styled.p`
    color: ${({ theme }) => theme.colors.dark};
    font-size: 0.9em;
  `,
};"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

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

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

    assert codeblock.to_string() == content


def test_assignment():
    content = """this.state = {
      bar: null
    };"""

    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

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

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True))

    assert codeblock.type == CodeBlockType.MODULE


def test_array_call():
    content = """
array.forEach((element) => {
  console.log(element);
});
"""
    parser = JavaScriptParser("javascript")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

    assert codeblock.children[0].type == CodeBlockType.CALL
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

    print(codeblock.to_tree(debug=True))

    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 2
    assert codeblock.children[0].type == CodeBlockType.FUNCTION
    assert codeblock.children[1].type == CodeBlockType.FUNCTION


def test_imports():
    content = """import { myFunction as func } from './myModule';
import myDefault, { myFunction } from './myModule';
import * as myModule from './myModule';
"""
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True))

    assert codeblock.to_string() == content


def test_const_id():
    content = """const foo = await bar({
});"""
    parser = JavaScriptParser("javascript", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

    assert codeblock.type == CodeBlockType.MODULE
    assert len(codeblock.children) == 1
    assert codeblock.children[0].type == CodeBlockType.ASSIGNMENT


def test_incorrect_outcommented_code():
    content = """function foo() {
    ...
}"""
    parser = JavaScriptParser("javascript", apply_gpt_tweaks=False)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))
    assert len(codeblock.find_errors()) == 1

    parser = JavaScriptParser("javascript", apply_gpt_tweaks=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))
    assert len(codeblock.find_errors()) == 0

    assert codeblock.children[0].children[1].type == CodeBlockType.COMMENTED_OUT_CODE

def test_return_empty_root_tag():
    content = """
  return (
    <>
      <Button></Button>
    </>
  );"""

    parser = JavaScriptParser("javascript", apply_gpt_tweaks=False)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))


def test_jest_test():
    content = """
  test(
    'context change should prevent bailout of memoized component (useMemo -> ' +
      'no intermediate fiber)',
    async () => {
      const root = ReactNoop.createRoot();
      expect(root).toMatchRenderedOutput('1');
    },
  );
  
  test('multiple contexts are propagated across retries', async () => {
    const root = ReactNoop.createRoot();
    expect(root).toMatchRenderedOutput('1');
  });
"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))

    test_cases = [x for x in codeblock.children if x.type == CodeBlockType.TEST_CASE]
    assert len(test_cases) == 2


def test_real_example_ReactContextPropagation_test():
    with open("javascript/ReactContextPropagation-test.js", "r") as f:
        content = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))

    print(codeblock.to_string_with_blocks(['', 'ReactLazyContextPropagation.context is propagated across retries (legacy)'], include_references=False, show_commented_out_code_comment=False))

    blocks = codeblock.get_blocks(has_identifier=True,
                                  include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS,
                                                 CodeBlockType.TEST_CASE, CodeBlockType.TEST_SUITE])

    #for block in blocks:
    #    block_content = block.to_context_string(show_commented_out_code_comment=False)
    #    tokens = count_tokens(block_content)
    #    print("path: " + str(block.path_string()) + "  , tokens:" + str(tokens))

    print(len(blocks))
    #assert codeblock.to_string() == content



def test_react():
    with open("/home/albert/repos/stuffs/react/packages/react-dom/src/__tests__/ReactDOMForm-test.js") as f:
        content = f.read()

    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))
    print(codeblock.to_string_with_blocks([''], include_references=False, show_commented_out_code_comment=False))
