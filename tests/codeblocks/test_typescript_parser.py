from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.parser.typescript import TypeScriptParser


def test_typescript_treesitter_types():
    with open("typescript/treesitter_types.ts", "r") as f:
        content = f.read()
    with open("typescript/treesitter_types_expected.txt", "r") as f:
        expected_tree = f.read()

    parser = TypeScriptParser()
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True))

    assert codeblock.to_tree(use_colors=False) == expected_tree


def test_react_tsx():
    with open("typescript/react_component.tsx", "r") as f:
        content = f.read()
    with open("typescript/react_component_expected.txt", "r") as f:
        expected_tree = f.read()

    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, only_identifiers=False))

    assert codeblock.to_tree(use_colors=False) == expected_tree
    assert codeblock.to_string() == content


def test_precomments():
    content = """
/**
 * This hook gives access the [router object](https://nextjs.org/docs/pages/api-reference/functions/use-router#router-object)
 * inside the [Pages Router](https://nextjs.org/docs/pages/building-your-application).
 *
 * Read more: [Next.js Docs: `useRouter`](https://nextjs.org/docs/pages/api-reference/functions/use-router)
 */
export function useRouter(): NextRouter {
  const router = React.useContext(RouterContext)
  if (!router) {
    throw new Error(
      'NextRouter was not mounted. https://nextjs.org/docs/messages/next-router-not-mounted'
    )
  }

  return router
}
"""
    parser = TypeScriptParser("tsx", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True))

    assert codeblock.to_string() == content


def test_imports():
    content = """import { myFunction as func } from './myModule';
import myDefault, { myFunction } from './myModule';
import * as myModule from './myModule';
import myModule = require('myModule');
import * as React from 'react';
import * as myModule from './myModule';
"""
    parser = TypeScriptParser("tsx", debug=True)
    codeblock = parser.parse(content)

    print(codeblock.to_tree(debug=True, include_references=True))

    assert codeblock.to_string() == content

def test_component():
    parser = TypeScriptParser()
    codeblock = parser.parse("""
const MyComponent: React.FC<MyComponentProps> = ({ initialCount = 0 }) => {
  const [count, setCount] = useState(initialCount);
}

""")
    print(codeblock.to_tree(debug=True))

def test_assignment():
    parser = TypeScriptParser()
    codeblock = parser.parse("let a = 1 + 2;")
    print(codeblock.to_tree(debug=True))

def test_new_date():
    parser = TypeScriptParser()
    codeblock = parser.parse("let a = new Date();")
    print(codeblock.to_tree(debug=True))

def test_type_annotations():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myVariable: Array<number>;")
    print(codeblock.to_tree(debug=True))
    assert codeblock.to_tree(use_colors=False) == """ 0 module ``
  1 code `myVariable`
   2 block_delimiter `;`
"""


def test_nullable_var():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myNullableVar: number | null = null;")
    print(codeblock.to_tree(debug=False))
    assert codeblock.to_tree(use_colors=False) == """ 0 module ``
  1 code `myNullableVar`
   2 code `=`
   2 code `null`
   2 block_delimiter `;`
"""

def test_lexical_declaration():
    parser = TypeScriptParser()
    codeblock = parser.parse("let myVariable: number;")
    assert codeblock.to_tree(use_colors=False, debug=False) == """ 0 module ``
  1 code `let`
   2 code `myVariable`
   2 code `: number`
   2 block_delimiter `;`
"""

    codeblock_assigned = parser.parse("let myVariable: number = 5;")

    assert codeblock_assigned.to_tree(use_colors=False) == """ 0 module ``
  1 code `let myVariable: number`
   2 code `=`
   2 code `5`
   2 block_delimiter `;`
"""

def test_lexical_declaration_class():
    parser = TypeScriptParser()
    codeblock = parser.parse("const Foo = () => {}")
    print(codeblock.to_tree(debug=True))

    assert codeblock.to_tree(use_colors=False, only_identifiers=False) == """ 0 module ``
  1 class `const Foo = () =>`
   2 block_delimiter `{`
   2 block_delimiter `}`
"""

def test_lexical_declaration_function():
    parser = TypeScriptParser()
    codeblock = parser.parse("const getFoo = useCallback(() => {})")
    print(codeblock.to_tree(debug=True))

    assert codeblock.to_tree(use_colors=False) == """ 0 module ``
  1 function `getFoo`
   2 block_delimiter `{`
   2 block_delimiter `}`
   2 block_delimiter `)`
"""

def test_export_react_component():
    parser = TypeScriptParser()
    codeblock = parser.parse("export const Foo: React.FC<Props> = ({\n  bar \n}) => {}")
    print(codeblock.to_tree(debug=True))

    assert len(codeblock.children) == 1
    assert codeblock.children[0].type == CodeBlockType.CLASS
    assert codeblock.children[0].content == "export const Foo: React.FC<Props> = ({\n  bar \n}) => "
    assert codeblock.children[0].identifier == "Foo"

def test_set_const():
    parser = TypeScriptParser("typescript")

    content = """const SEARCH_CARE_UNITS_THRESHOLD = 5"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))


def test_react_use_functions():
    parser = TypeScriptParser("typescript")

    content = """const handleSearchInput = useMemo(() => {
      const handler: React.ChangeEventHandler<HTMLInputElement> = ({ target: { value } }) => {
        setSearchValue(value);
      };

      return handler;
    }, []);"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))

    content = """const position = useRef<GeolocationPosition>();"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))

    content = """const[sortBy, setSortBy] = useState <SortBy> ();"""
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True))




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

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True))

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

    print(codeblock.to_tree(debug=True))

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

    assert codeblock.to_tree(use_colors=False) == """ 0 module ``
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
  
  test('should do something else', () => {
    renderWithProviders(<Foo {...MOCK_PROPS} />);
    expect(FooMock).toHaveBeenCalled();
  });
  
});
"""
    parser = TypeScriptParser("tsx")
    codeblock = parser.parse(content)
    print(codeblock.to_tree(only_identifiers=False, debug=True))

    assert codeblock.children[0].type == CodeBlockType.TEST_SUITE
    assert codeblock.children[0].identifier == "Foo"

    test_cases = [x for x in codeblock.children[0].children if x.type == CodeBlockType.TEST_CASE]
    assert len(test_cases) == 2

def test_return_tsx_elements():
    content = """
export const DeleteUser = ({ id }: DeleteUserProps) => {
  return (
    <ConfirmationDialog
      icon="danger"
      triggerButton={<Button variant="danger">Delete</Button>}
      confirmButton={
        <Button
          isLoading={deleteUserMutation.isLoading}
          type="button"
          onClick={() => deleteUserMutation.mutate({ userId: id })}
        >
          Delete User
        </Button>
      }
    />
  );
};
"""
    parser = TypeScriptParser("tsx", debug=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(only_identifiers=True, include_parameters=True, debug=True, include_references=True))


def test_react_component():
    content = """
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      type = 'button',
      className = '',
      variant = 'primary',
      size = 'md',
      isLoading = false,
      startIcon,
      endIcon,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        type={type}
        className={clsx(
          'flex justify-center items-center border border-gray-300 disabled:opacity-70 disabled:cursor-not-allowed rounded-md shadow-sm font-medium focus:outline-none hover:opacity-80',
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {isLoading && <Spinner size="sm" className="text-current" />}
        {!isLoading && startIcon}
        <span className="mx-2">{props.children}</span> {!isLoading && endIcon}
      </button>
    );
  }
);
"""
    parser = TypeScriptParser("tsx", debug=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(only_identifiers=True, include_parameters=True, debug=True, include_references=True))



def test_const_class():
    content = """
export const MDPreview = ({ value = '' }: MDPreviewProps) => {
  return (
    <div/>
  );
};
"""

    parser = TypeScriptParser("tsx", debug=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(only_identifiers=True, include_parameters=True, debug=True, include_references=True))

def test_export_and_return_call():
    content = """
export const createComment = ({ data }: CreateCommentDTO): Promise<Comment> => {
  return axios.post('/comments', data);
};"""

    parser = TypeScriptParser(debug=True)
    codeblock = parser.parse(content)
    print(codeblock.to_tree(debug=True, include_references=True, include_parameters=True))

def test_return_call():
    content = """
  return useMutation({
    onMutate: async (newComment) => {
      const previousComments = queryClient.getQueryData<Comment[]>(['comments', discussionId]);
      return { previousComments };
    },
    ...config,
    mutationFn: createComment,
  });"""

    parser = TypeScriptParser(debug=True)
    codeblock = parser.parse(content)
    print(
        codeblock.to_tree(debug=True, include_references=True, include_parameters=True))
