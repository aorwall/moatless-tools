import pytest

from moatless.actions.string_replace import StringReplace, StringReplaceArgs, find_potential_matches, \
    find_exact_matches, find_match_when_ignoring_indentation
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("test.py", """def hello():
    message = "Hello World"
    print(message)
""")
    return repo

@pytest.fixture
def file_context(repository):
    context = FileContext(repo=repository)
    # Add file to context to ensure it's available
    context.add_file("test.py", show_all_spans=True)
    return context


def test_string_replace_basic(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='    message = "Hello World"',
        new_str='    message = "Hello Universe"',
        scratch_pad="Updating greeting message"
    )
    
    observation = action.execute(args, file_context)
    print(observation.message)
    
    content = file_context.get_file("test.py").content
    assert 'message = "Hello Universe"' in content
    assert "def hello():" in content  # Verify the rest of the file is intact
    assert "print(message)" in content
    assert "diff" in observation.properties

def test_string_replace_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='not_existing_string',
        new_str='new_string',
        scratch_pad="Trying to replace non-existent string"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "string_not_found"
    assert observation.expect_correction

def test_string_replace_multiple_occurrences(repository, file_context):
    # Create file with multiple occurrences
    repository.save_file("test2.py", """def hello():
    message = "test"
    print(message)
    message = "test"
""")
    file_context.add_file("test2.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test2.py",
        old_str='    message = "test"',  # Include proper indentation in search string
        new_str='    message = "updated"',
        scratch_pad="Updating test messages"
    )
    
    observation = action.execute(args, file_context)
    
    # Verify the error message contains the content of each match
    assert 'Lines 2-2:\n```\n    message = "test"\n```' in observation.message
    assert 'Lines 4-4:\n```\n    message = "test"\n```' in observation.message
    assert "Try including more surrounding lines to create a unique match" in observation.message
    assert observation.properties["flags"] == ["multiple_occurrences"]
    assert observation.expect_correction

def test_string_replace_file_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="nonexistent.py",
        old_str='old_string',
        new_str='new_string',
        scratch_pad="Trying to modify non-existent file"
    )
    
    observation = action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "file_not_found"

def test_string_replace_same_string(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='    message = "Hello World"',
        new_str='    message = "Hello World"',
        scratch_pad="Trying to replace with same string"
    )
    
    observation = action.execute(args, file_context)
    print(observation.message)
    
    assert observation.properties["fail_reason"] == "no_changes"

def test_string_replace_with_indentation(repository, file_context):
    # Create file with indented content - note the proper indentation
    repository.save_file("test3.py", """class Test:
    def method(self):
        value = "old"
        return value
""")
    file_context.add_file("test3.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test3.py",
        old_str='        value = "old"',  # Include proper indentation in search string
        new_str='        value = "new"',
        scratch_pad="Updating indented value"
    )
    
    observation = action.execute(args, file_context)
    
    content = file_context.get_file("test3.py").content
    assert '        value = "new"' in content
    assert "class Test:" in content  # Verify the rest of the file is intact
    assert "def method(self):" in content
    assert "diff" in observation.properties

def test_string_replace_with_newlines(repository, file_context):
    # Create file with multiline content - note the proper indentation
    repository.save_file("test4.py", """def old_function():
    print("line1")
    print("line2")
""")
    file_context.add_file("test4.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test4.py",
        old_str='''def old_function():
    print("line1")
    print("line2")''',
        new_str='''def new_function():
    print("new_line1")
    print("new_line2")''',
        scratch_pad="Replacing entire function"
    )
    
    observation = action.execute(args, file_context)
    
    content = file_context.get_file("test4.py").content
    assert 'def new_function():' in content
    assert '    print("new_line1")' in content  # Check indented content
    assert '    print("new_line2")' in content
    assert "diff" in observation.properties



def test_find_potential_matches_indentation_differs():
    content = """def hello():
        message = "Hello World"
        print(message)"""

    # Search with different indentation
    old_str = 'message = "Hello World"'

    matches = find_match_when_ignoring_indentation(old_str, content)
    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'indentation_differs'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 2
    assert matches[0]['content'] == '        message = "Hello World"'
    
    # Verify auto-correction information
    assert matches[0]['can_auto_correct'] == True
    assert matches[0]['uniform_indent_diff'] == 8
    assert len(matches[0]['differences']) == 1
    assert matches[0]['differences'][0] == 'Line 1: expected 0 spaces, found 8 spaces'


def test_find_potential_matches_line_breaks_differ():
    content = """def process():
    result = (
        data
        .filter()
        .transform()
    )"""

    # Search with single line version
    old_str = 'result = (data.filter().transform())'

    matches = find_potential_matches(old_str, content)
    print(matches)

    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 6
    assert '    result = (\n        data\n        .filter()\n        .transform()\n    )' in matches[0]['content']


def test_find_potential_matches_multiple_matches():
    content = """def process():
    data = transform().filter()

    result = (
        transform()
        .filter()
    )
    
    data = transform().filter()

    final = transform().filter()"""

    old_str = 'data = transform().filter()'
    matches = find_match_when_ignoring_indentation(old_str, content)
    assert len(matches) == 2
    assert all(m['diff_reason'] == 'indentation_differs' for m in matches)

    old_str = 'transform().filter()'
    matches = find_potential_matches(old_str, content)
    assert len(matches) == 1
    assert all(m['diff_reason'] == 'line_breaks_differ' for m in matches)


def test_find_potential_matches_no_match():
    content = """def hello():
    message = "Hello World"
    print(message)"""

    old_str = 'not_existing_content'

    matches = find_potential_matches(old_str, content)

    assert len(matches) == 0


def test_find_match_when_ignoring_indentation():
    content = """def example():
    if condition:
        with open('file.txt') as f:
            process(f)"""

    # Search with different indentation levels
    old_str = """if condition:
with open('file.txt') as f:
    process(f)"""

    matches = find_match_when_ignoring_indentation(old_str, content)

    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'indentation_differs'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert '    if condition:\n        with open(\'file.txt\') as f:\n            process(f)' in matches[0]['content']


def test_find_exact_matches_basic():
    # Matches test_string_replace_basic setup
    content = """def hello():
    message = "Hello World"
    print(message)
"""
    
    matches = find_exact_matches('    message = "Hello World"', content)
    assert len(matches) == 1
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 2
    assert matches[0]['content'] == '    message = "Hello World"'
    assert matches[0]['diff_reason'] == 'exact_match'

def test_find_exact_matches_multiline():
    content = """def process():
    if True:
        print("a")
        print("b")
    return None"""
    
    old_str = '''    if True:
        print("a")
        print("b")'''
    
    matches = find_exact_matches(old_str, content)
    assert len(matches) == 1
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert matches[0]['content'] == old_str
    assert matches[0]['diff_reason'] == 'exact_match'

def test_find_exact_matches_multiple_occurrences():
    content = """def test():
    print("hello")
    print("world")
    print("hello")"""
    
    matches = find_exact_matches('    print("hello")', content)
    assert len(matches) == 2
    assert matches[0]['start_line'] == 2
    assert matches[1]['start_line'] == 4
    assert matches[0]['end_line'] == 2
    assert matches[1]['end_line'] == 4
    assert all(m['diff_reason'] == 'exact_match' for m in matches)
    assert all(m['content'] == '    print("hello")' for m in matches)

    # Test no matches
    matches = find_exact_matches('not_found', content)
    assert len(matches) == 0

def test_find_potential_matches_latex_case():
    # Real case from sympy/printing/latex.py
    content = """def test():
        return (r"\\left\\["
              + r", ".join(self._print(el) for el in printset)
              + r"\\right\\]")"""
    
    old_str = """return (r"\\left\\[" + r", ".join(self._print(el) for el in printset) + r"\\right\\]")"""
    
    matches = find_potential_matches(old_str, content)
    
    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert matches[0]['content'] == '''        return (r"\\left\\["
              + r", ".join(self._print(el) for el in printset)
              + r"\\right\\]")'''


def test_find_potential_matches_multiline_string_concatenation():
    content = """    default_error_messages = {
        'invalid': _("'%(value)s' value has an invalid format. It must be in "
                     "[DD] [HH:[MM:]]ss[.uuuuuu] format.")
    }"""

    # Test with single-line version
    old_str = "'invalid': _(\"'%(value)s' value has an invalid format. It must be in [DD] [HH:[MM:]]ss[.uuuuuu] format.\")"

    matches = find_potential_matches(old_str, content)
    print(matches)


    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'

    print(matches[0]['content'])
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 3

    assert matches[0]['content'] == """        'invalid': _("'%(value)s' value has an invalid format. It must be in "
                     "[DD] [HH:[MM:]]ss[.uuuuuu] format.")"""

def test_find_potential_matches_string_concatenation_with_params():
    content = r"""                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ForeignKey("%s", symmetrical=False, through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            )"""
    
    old_str = r'''hint=(
    'If you want to create a recursive relationship, '
    'use ForeignKey("%s", symmetrical=False, through="%s").'
) % (
    RECURSIVE_RELATIONSHIP_CONSTANT,
    relationship_model_name,
)'''
    
    matches = find_potential_matches(old_str, content)
    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'

    # Check specific formatting differences
    assert matches[0]['content'].startswith('                            hint=('), "Content should preserve original indentation"

    # TODO: Verify the differences list contains the expected differences
    differences = matches[0]['differences']
    print(differences)
    #assert len(differences) == 2
    #assert any('Line break count differs' in diff for diff in differences)
    #assert any('Additional characters found:' in diff for diff in differences)


def test_find_potiential_match():
    content = """
        output_config = _get_output_config("transform", self)
        for name, trans, columns in transformers:
            if replace_strings:
                # replace 'passthrough' with identity transformer and
                # skip in case of 'drop'
                if trans == "passthrough":
                    trans = FunctionTransformer(
                        accept_sparse=True,
                        check_inverse=False,
                        feature_names_out="one-to-one",
                    ).set_output(transform=output_config["dense"])
                elif trans == "drop":
                    continue
                elif _is_empty_column_selection(columns):
                    continue

            if column_as_strings:
                # Convert all columns to using their string labels
                columns_is_scalar = np.isscalar(columns)

                indices = self._transformer_to_input_indices[name]
                columns = self.feature_names_in_[indices]

                if columns_is_scalar:
                    # selection is done with one dimension
                    columns = columns[0]

            yield (name, trans, columns, get_weight(name))
"""

    old_str = """if column_as_strings:
    # Convert all columns to using their string labels
    columns_is_scalar = np.isscalar(columns)

    indices = self._transformer_to_input_indices[name]
    columns = self.feature_names_in_[indices]

    if columns_is_scalar:
        # selection is done with one dimension
        columns = columns[0]

yield (name, trans, columns, get_weight(name))"""

    matches = find_potential_matches(old_str, content)
    assert len(matches) == 1

    assert matches[0]['start_line'] == 18
    assert matches[0]['end_line'] == 29

    matches = find_match_when_ignoring_indentation(old_str, content)
    assert len(matches) == 1

    assert matches[0]['start_line'] == 18
    assert matches[0]['end_line'] == 29

def test_string_replace_args_strips_line_numbers():
    # Test with line numbers at the start of each line
    args = StringReplaceArgs(
        path="test.py",
        old_str="""    515	        # If only one mask is present we need not bother about any type checks
    516	        if (
    517	            self.mask is None and operand is not None and operand.mask is None
    518	        ) or handle_mask is None:""",
        new_str="        # New comment\n        if condition:",
        scratch_pad="Testing line number stripping"
    )
    
    # Verify line numbers were stripped from old_str
    expected = """        # If only one mask is present we need not bother about any type checks
        if (
            self.mask is None and operand is not None and operand.mask is None
        ) or handle_mask is None:"""
    
    # Print both strings for debugging
    print("Actual:\n" + repr(args.old_str))
    print("Expected:\n" + repr(expected))
    
    assert args.old_str == expected
    
    # Test with different line number formats
    args = StringReplaceArgs(
        path="test.py",
        old_str="""515        # Comment
516        code_line
517        another_line""",
        new_str="new_content",
        scratch_pad="Testing different line number format"
    )
    
    assert args.old_str == """        # Comment
        code_line
        another_line"""

def test_string_replace_with_negative_indentation(repository, file_context):
    # Create file with deeply indented content
    repository.save_file("test_indent.py", """def outer():
        def inner():
            value = "test"
            return value""")
    file_context.add_file("test_indent.py", show_all_spans=True)
    
    action = StringReplace(repository=repository, auto_correct_indentation=True)
    args = StringReplaceArgs(
        path="test_indent.py",
        old_str="""            value = "test"
            return value""",  # Indented with 12 spaces
        new_str="""    value = "updated"
    return value""",  # Indented with 4 spaces
        scratch_pad="Reducing indentation level"
    )
    
    observation = action.execute(args, file_context)
    
    content = file_context.get_file("test_indent.py").content
    assert "def outer():" in content
    assert "def inner():" in content
    assert '            value = "updated"' in content  # Should maintain original indentation
    assert '            return value' in content
    assert "diff" in observation.properties
    assert "auto_corrected_indentation" in observation.properties.get("flags", [])

def test_string_replace_with_empty_new_str(repository, file_context):
    # Create file with content to remove
    repository.save_file("test_remove.py", """def example():
    # Old comment to remove
    value = 42
    return value""")
    file_context.add_file("test_remove.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test_remove.py",
        old_str='    # Old comment to remove\n',  # Include newline to maintain formatting
        new_str='',  # Empty string to remove the line
        scratch_pad="Removing comment line"
    )
    
    observation = action.execute(args, file_context)
    
    content = file_context.get_file("test_remove.py").content
    assert content == """def example():
    value = 42
    return value"""
    assert "diff" in observation.properties