from moatless.benchmark.swebench import setup_swebench_repo, load_instance
from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import (
    Relationship,
    RelationshipType,
    ReferenceScope,
    SpanType,
)
from moatless.codeblocks.parser.python import PythonParser


def _verify_parsing(content, assertion, apply_gpt_tweaks=True, debug=True):
    parser = PythonParser(apply_gpt_tweaks=apply_gpt_tweaks, debug=debug)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_references=True, show_spans=True, show_tokens=True))

    assert codeblock.to_string() == content

    assertion(codeblock)


def test_function():
    content = """def foo():
    # ... existing code
    print('hello world')"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "foo"
        assert len(codeblock.children[0].children) == 2

    _verify_parsing(content, assertion)


def test_outcommented_function():
    content = """def foo():
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "foo"
        assert codeblock.children[0].type == CodeBlockType.COMMENTED_OUT_CODE

    _verify_parsing(content, assertion)


def test_function_followed_by_comment():
    content = """def foo():
    print('hello world')

# comment
"""

    def assertion(codeblock):
        assert len(codeblock.children) == 3

    _verify_parsing(content, assertion)


def test_outcommented_function_with_decorator():
    content = """@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "pytest_configure"
        assert codeblock.children[0].type == CodeBlockType.COMMENTED_OUT_CODE

    _verify_parsing(content, assertion)


def test_outcommented_functions():
    content = """@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # ... existing code

def pytest_unconfigure(config):
    # ... existing code

def create_new_paste(contents):
    import re

def pytest_terminal_summary(terminalreporter):
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 4
        assert [child.identifier for child in codeblock.children] == [
            "pytest_configure",
            "pytest_unconfigure",
            "create_new_paste",
            "pytest_terminal_summary",
        ]
        assert [child.type for child in codeblock.children] == [
            CodeBlockType.COMMENTED_OUT_CODE,
            CodeBlockType.COMMENTED_OUT_CODE,
            CodeBlockType.FUNCTION,
            CodeBlockType.COMMENTED_OUT_CODE,
        ]

    _verify_parsing(content, assertion)


def test_function_in_function():
    content = """def foo():
    def bar():
        print('hello world')
        return 42

    print("hello")
    return bar()"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1

    _verify_parsing(content, assertion)


def test_class_with_comment():
    content = """class BaseSchema(base.SchemaABC):
    # ... other code

    def _invoke_field_validators(self, unmarshal, data, many):
        foo
    bar = 1"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1

    _verify_parsing(content, assertion)


def test_raise_string_line_break():
    content = """def foo():
    raise ValueError(
                     \"""
FITS WCS distortion paper lookup tables and SIP distortions only work
in 2 dimensions.  However, WCSLIB has detected {0} dimensions in the
core WCS keywords.  To use core WCS in conjunction with FITS WCS
distortion paper lookup tables or SIP distortion, you must select or
reduce these to 2 dimensions using the naxis kwarg.
\""".format(wcsprm.naxis))
"""

    def assertion(codeblock):
        print(codeblock.to_tree())
        print(codeblock.to_string())

    _verify_parsing(content, assertion)


def test_referenced_blocks():
    content = """class Event:
    def __init__(self, name):
        self.name = name

class EventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event: Event):
        self.events.append(event)
        self.show_last_event()

    def show_last_event(self):
        if self.events:
            print(f"Last event: {self.events[-1]}")
"""

    def assertion(codeblock):
        # TODO: Verify!
        pass

    _verify_parsing(content, assertion, debug=False)


def test_decoratated_function():
    content = """class Foo:
    @classmethod
    def bar(cls):
        return 42
"""

    def assertion(codeblock):
        assert content == codeblock.to_string()
        func = codeblock.find_by_path(["Foo", "bar"])
        assert func is not None
        assert len(func.children) == 1
        assert func.children[0].type == CodeBlockType.STATEMENT

    _verify_parsing(content, assertion, debug=False)


def test_decoratated_function_with_comment():
    content = """class Foo:
    @classmethod
    def bar(cls):
        # ... other code
        return 42
"""

    def assertion(codeblock):
        func = codeblock.find_by_path(["Foo", "bar"])
        assert func is not None
        assert len(func.children) == 2
        assert func.children[0].type == CodeBlockType.COMMENTED_OUT_CODE

    _verify_parsing(content, assertion, apply_gpt_tweaks=True, debug=False)


def test_decorated_function():
    content = """@property
def identity(self):
    foo = 1
    return super().identity + (
        self.through
    )"""

    def assertion(codeblock):
        pass

    _verify_parsing(content, assertion, debug=True)


def test_parse_function_with_class_relationship():
    content = """class Foo:

    def _reset(self):
        self.bar = None
"""

    def assertion(codeblock):
        pass

    _verify_parsing(content, assertion, debug=True, apply_gpt_tweaks=False)


def test_parse_function_with_relationship():
    content = """import time

def print_reset():
    print('reset1')

class Base:

    def baz():
        pass

class Foo(Base):

    def __init__(self, bar):
        self.bar = bar

    def reset(self):
        self._reset()
        print_reset()

    def _reset(self):
        time.sleep(0.1)
        self.bar = None
"""

    def assertion(codeblock):
        foo_class = codeblock.find_by_path(["Foo"])
        relationships = foo_class.get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.IS_A,
            path=["Base"],
        )

        init_func = codeblock.find_by_path(["Foo", "__init__"])
        relationships = init_func.get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.CLASS,
            identifier="self.bar",
            type=RelationshipType.USES,  # TODO: Change?
            path=["Foo", "bar"],
        )

        reset_func = codeblock.find_by_path(["Foo", "reset"])
        relationships = reset_func.get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.CLASS,
            identifier="",
            type=RelationshipType.CALLS,  # TODO: Change?
            path=["Foo", "_reset"],
        )

        assert relationships[1] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.CALLS,  # TODO: Change?
            path=["print_reset"],
        )

    _verify_parsing(content, assertion, debug=False, apply_gpt_tweaks=False)


def test_parse_class_relationships():
    content = """class ForeignObjectRel(FieldCacheMixin):

    def __init__(self, field):
        self.field = field

    @property
    def identity(self):
        return (
            self.field
        )

class ManyToManyRel(ForeignObjectRel):

    def __init__(self, field, through):
        super().__init__(
            field
        )
        self.through = through

    @property
    def identity(self):
        return super().identity + (
            self.through
        )
"""

    def assertion(codeblock):
        m2m_class = codeblock.find_by_path(["ManyToManyRel"])
        relationships = m2m_class.get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.IS_A,
            path=["ForeignObjectRel"],
        )

        init_func = codeblock.find_by_path(["ManyToManyRel", "__init__"])
        relationships = init_func.get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.LOCAL,
            type=RelationshipType.USES,
            path=["ForeignObjectRel", "__init__"],
        )

        id_func = codeblock.find_by_path(["ManyToManyRel", "identity"])
        relationships = id_func.get_all_relationships()
        # TODO: Support and assert relationships to super and self!

    _verify_parsing(content, assertion, debug=True, apply_gpt_tweaks=False)


def test_parse_class_with_method_mixin():
    content = """class SuperClass:
    pass

class SubClass(SuperClass):
    pass

class SubClassWithMultipleClasses(SuperClass, AnotherClass):
    pass

class SubClassWithMethod(foo.bar(SuperClass, AnotherClass)):
    pass
"""

    def assertion(codeblock):
        assert codeblock.find_by_path(["SuperClass"]).type == CodeBlockType.CLASS
        assert codeblock.find_by_path(["SubClass"]).type == CodeBlockType.CLASS
        assert (
            codeblock.find_by_path(["SubClassWithMultipleClasses"]).type
            == CodeBlockType.CLASS
        )
        assert (
            codeblock.find_by_path(["SubClassWithMethod"]).type == CodeBlockType.CLASS
        )

        relationships = codeblock.find_by_path(["SubClass"]).get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.IS_A,
            path=["SuperClass"],
        )
        relationships = codeblock.find_by_path(
            ["SubClassWithMultipleClasses"]
        ).get_all_relationships()
        assert relationships[0] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.IS_A,
            path=["SuperClass"],
        )
        assert relationships[1] == Relationship(
            scope=ReferenceScope.LOCAL,
            identifier="",
            type=RelationshipType.IS_A,
            path=["AnotherClass"],
        )
        relationships = codeblock.find_by_path(
            ["SubClassWithMethod"]
        ).get_all_relationships()
        # TODO: Support method calls

    _verify_parsing(content, assertion, debug=True, apply_gpt_tweaks=False)


def test_init_spans():
    content = """\"""
"Rel objects" for related fields.
\"""

from django.core import exceptions

class ForeignObjectRel(FieldCacheMixin):
    \"""
    Used by ForeignObject to store information about the relation.
    \"""

    # Field flags
    auto_created = True

    def __init__(self, field, to):
        self.field = field

    # Some of the following cached_properties can't be initialized in
    @cached_property
    def hidden(self):
        return self.is_hidden()
"""

    def assertion(codeblock):
        pass  # TODO

    _verify_parsing(content, assertion, debug=False, apply_gpt_tweaks=False)


def test_spans():
    content = """def foo():
    row_1 = 1
    row_2 = 2
    row_3 = 3

bar = 42
"""

    parser = PythonParser(max_tokens_in_span=12)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_references=True, show_spans=True, show_tokens=True))

    assert codeblock.to_string() == content


def test_assignment_with_line_break():
    content = """BAR = \\
42

_FOO = \\
    r\"""
foo
\"""
"""

    def assertion(codeblock):
        print(codeblock.to_prompt(show_span_id=True))

    _verify_parsing(content, assertion, debug=True)


def test_init_span():
    content = """class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField()

    class Meta:
        model = User

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def foo(self):
        pass
"""

    def assertion(codeblock):
        assert (
            codeblock.find_by_path(["UserChangeForm"]).belongs_to_span.span_type
            == SpanType.INITATION
        )
        assert (
            codeblock.find_by_path(["UserChangeForm", "Meta"]).belongs_to_span.span_type
            == SpanType.INITATION
        )
        assert codeblock.find_by_path(
            ["UserChangeForm", "Meta"]
        ).belongs_to_span.parent_block_path == ["UserChangeForm", "Meta"]
        assert (
            codeblock.find_by_path(
                ["UserChangeForm", "__init__"]
            ).belongs_to_span.span_type
            == SpanType.INITATION
        )
        assert codeblock.find_by_path(
            ["UserChangeForm", "__init__"]
        ).belongs_to_span.parent_block_path == ["UserChangeForm", "__init__"]
        assert (
            codeblock.find_by_path(["UserChangeForm", "foo"]).belongs_to_span.span_type
            == SpanType.IMPLEMENTATION
        )
        assert codeblock.find_by_path(
            ["UserChangeForm", "foo"]
        ).belongs_to_span.parent_block_path == ["UserChangeForm", "foo"]

        print(codeblock.to_prompt(show_span_id=True))

    _verify_parsing(content, assertion, debug=False)


def test_class_with_methods_spans():
    content = """class Q(tree.Node):

    def __and__(self, other):
        return self._combine(other, self.AND)

    def __rand__(self, other):
        return self.__and__(other)"""

    def assertion(codeblock):
        class_span = codeblock.find_span_by_id("Q")
        assert class_span is not None
        assert class_span.parent_block_path == ["Q"]
        assert codeblock.find_span_by_id("Q.__and__") is not None
        assert codeblock.find_span_by_id("Q.__rand__") is not None

    _verify_parsing(content, assertion, debug=False)


def test_spans():
    content = """def uniq(seq, result=None):
    def check():
        # check that size of seq did not change during iteration;
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    try:
        seen = set()
    except TypeError:
        yield s
"""
    parser = PythonParser(max_tokens_in_span=2000, debug=True)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_references=True, show_spans=True, show_tokens=True))

    assert codeblock.to_string() == content


def test_with_line_numbers():
    content = """foo = 42

link = add_domain(
    current_site.domain,
    self._get_dynamic_attr('item_link', item),
    request.is_secure())

def foo():
    line_1 = 9
    line_2 = 10
    line_3 = 11
"""

    def assertion(codeblock):
        print("Tree:\n", codeblock.to_tree())

        prompt = codeblock.to_prompt()
        print("Prompt:\n", prompt)

        prompt_with_line_numbers = codeblock.to_prompt(show_line_numbers=True)
        print("Prompt with line numbers:\n", prompt_with_line_numbers)

        prompt_with_line_10_11 = codeblock.to_prompt(
            start_line=10,
            end_line=11,
            show_outcommented_code=True,
            outcomment_code_comment="... other code",
        ).strip()
        print("Prompt line 10-11:\n", prompt_with_line_10_11)

        assert (
            prompt_with_line_10_11
            == """# ... other code

def foo():
    # ... other code
    line_2 = 10
    line_3 = 11"""
        )

    _verify_parsing(content, assertion, debug=False)


def test_if_else_clause():
    content = """if foo == 42:
        bar = 42
    elif foo == 43:
        bar = 43
    else:
        bar = 44
"""

    def assertion(codeblock):
        assert codeblock.children[0].type == CodeBlockType.COMPOUND

    _verify_parsing(content, assertion, debug=True)


def test_commented_out_if_else_clause():
    content = """if foo == 42:
    pass
else:
    bar = 44"""

    def assertion(codeblock):
        assert codeblock.children[0].type == CodeBlockType.COMPOUND

    _verify_parsing(content, assertion, apply_gpt_tweaks=True, debug=True)


def test_query_match():
    content = """if foo == 42:
    return True
"""

    parser = PythonParser()
    content_in_bytes = bytes(content, "utf8")
    tree = parser.tree_parser.parse(content_in_bytes)

    for label, node_type, query in parser.queries:
        if label == "python.scm:17":
            print(label, node_type, query)
            break

    captured = query.captures(tree.walk().node.children[0])
    print(captured)


def test_find_nested_blocks_by_line_numbers():
    content = """def foo():
    if True:
        line_1 = 9
    else:
        line_3 = 11
        while True:
            line_2 = 10
    return False"""

    def assertion(codeblock):
        prompt_with_line_numbers = codeblock.to_prompt(show_line_numbers=True)
        print(prompt_with_line_numbers)

        assert (
                prompt_with_line_numbers
                ==
    """     1	def foo():
     2	    if True:
     3	        line_1 = 9
     4	    else:
     5	        line_3 = 11
     6	        while True:
     7	            line_2 = 10
     8	    return False""")

def test_next_and_previous():
    content = """class Foo:

    def __init__(self):
        self.bar = 42
        self.baz = 43

    def foo(self):
        return self.bar

    def bar(self):
        self.baz = 44
        return self.baz
"""

    def assertion(codeblock):
        expected_block_order = [
            "",
            "Foo",
            "Foo.__init__",
            "Foo.__init__.self.bar",
            "Foo.__init__.self.bar.42",
            "Foo.__init__.self.baz",
            "Foo.__init__.self.baz.43",
            "Foo.foo",
            "Foo.foo.return",
            "Foo.foo.return.self_bar",
            "Foo.bar",
            "Foo.bar.self.baz",
            "Foo.bar.self.baz.44",
            "Foo.bar.return",
            "Foo.bar.return.self_baz",
        ]

        next_block_order = []
        first_block = codeblock
        while first_block:
            next_block_order.append(first_block.path_string())
            first_block = first_block.next

        assert next_block_order == expected_block_order

        last_block = codeblock.last()

        previous_block_order = []
        while last_block:
            previous_block_order.append(last_block.path_string())
            last_block = last_block.previous

        assert previous_block_order == list(reversed(expected_block_order))

    _verify_parsing(content, assertion, debug=False)


def test_new_spans():
    content = """class PsBackendHelper:
    def __init__(self):
        self._cached = {}

ps_backend_helper = PsBackendHelper()

papersize = {'letter': (8.5, 11)}
"""

    def assertion(codeblock):
        print(codeblock.to_tree(show_spans=True, include_types=True))

        assert len(codeblock.span_ids) == 3

    _verify_parsing(content, assertion, debug=False)


def test_invalid_content():
    content = """evalcache_key = StoreKey[Dict[str, Any]]()

    # ...
    def _istrue(self) -> bool:
        if hasattr(self, "result"):
            result = getattr(self, "result")  # type: bool
            return result
        self._marks = self._get_marks()


        return False
    # ...
"""

    def assertion(codeblock):
        print(codeblock.to_tree(show_spans=True))

        placeholders = codeblock.find_blocks_with_type(CodeBlockType.COMMENTED_OUT_CODE)

        assert len(placeholders) == 2

    _verify_parsing(content, assertion, debug=False)


def test_ignored_spans():
    instance = load_instance("psf__requests-2674")
    repo_dir = setup_swebench_repo(instance)
    file_path = f"{repo_dir}/requests/adapters.py"
    with open(file_path, "r") as file:
        content = file.read()

    # def assertion(codeblock):
    #    print(codeblock.to_tree(show_spans=True))

    # _verify_parsing(content, assertion, debug=False)

    file_path = f"{repo_dir}/requests/sessions.py"
    with open(file_path, "r") as file:
        content = file.read()

    def assertion(codeblock):
        print(codeblock.to_tree(show_spans=True))
        assert "imports" in codeblock.span_ids

    _verify_parsing(content, assertion, debug=False)


def test_invalid_extra_comment():
    content = """import threading

class Signal:
    \"""
    Base class for all signals

    Internal attributes:

        receivers
            { receiverkey (id) : weakref(receiver) }
    \"""
    logger = logging.getLogger('django.dispatch')
    \"""
    def __init__(self, providing_args=None, use_caching=False):
        \"""
        Create a new signal.
        \"""
        self.receivers = []
"""

    def assertion(codeblock):
        print(codeblock.to_prompt(include_block_types=[CodeBlockType.ERROR]))

    _verify_parsing(content, assertion, debug=False)
