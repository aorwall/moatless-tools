import json

from ghostcoder.codeblocks import CodeBlockType
from ghostcoder.codeblocks.parser.python import PythonParser

parser = PythonParser(debug=True)

def test_python_all_treesitter_types():
    with open("python/treesitter_types.py", "r") as f:
        content = f.read()
    with open("python/treesitter_types_expected.txt", "r") as f:
        expected_tree = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_tree() == expected_tree
    assert code_blocks.to_string() == content

def test_python_calculator():
    with open("python/calculator.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    print(code_blocks.to_string())

    assert code_blocks.to_string() == content

def test_python_with_comment():
    with open("python/calculator_insert1.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_with_comment_2():
    with open("python/calculator_insert2.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content


def test_python_class():
    content = """class Calculator:

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content


def test_python_class_with_out_commented_code():
    content = """class Calculator:

    # ... (existing methods)

    def multiply(self, a, b):
        return a * b
"""
    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_function_with_only_comment():
    content = """def foo():
# ... rest of the code
i = 0"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content


def test_python_function_followed_by_unintended_comment():
    content = """def foo():\n# ... rest of the code"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_example():
    with open("python/example.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content


def test_expression_list():
    code_blocks = parser.parse("1, 2, 3")
    assert code_blocks.to_tree() == """ 0 module ``
  1 code `1`
  1 code `,`
  1 code `2`
  1 code `,`
  1 code `3`
"""


def test_outer_inner_def():
    codeblocks = parser.parse("""def outer():
    x = 10
    def inner():
        nonlocal x
        x = 20
    inner()
    print(x)
""")
    print(codeblocks.to_tree(include_tree_sitter_type=True))


def test_python_has_error_blocks():
    with open("python/dsl_dsl_find_update.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))
    print(code_blocks.find_errors()[0].to_string())

    assert len(code_blocks.find_errors()) == 1
    assert code_blocks.find_errors()[0].to_string() == """for item in data:
    # ...
    elif item[0] == EDGE:
        if len(item) != 3 or not isinstance(item[1], str) or not isinstance(item[2], str) or not isinstance(item[3], dict):
            raise ValueError("Edge is malformed")
        self.edges.append(Edge(item[1], item[2], item[3]))"""



def test_python_has_two_error_blocks():
    with open("python/proverb_replace.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert len(code_blocks.find_errors()) == 2
    assert code_blocks.find_errors()[0].to_string() == """
    for i in reversed(range(len(len(input_list)):
        result.append(f"For {qualifier} of a {input_list[i]}")"""
    assert code_blocks.find_errors()[1].to_string() == """
    return "."""

    print(code_blocks.find_errors()[0].copy_with_trimmed_parents().root().to_string())

def test_python_if_and_indentation_parsing_2():
    with open("python/if_clause.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content


def test_python_two_classes():
    with open("python/binary_search.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content


def test_python_comments():
    with open("python/word_search_update.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content



def test_python_sublist():
    with open("python/sublist.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content

def test_python_indentation_empty_lines():
    with open("python/circular_buffer.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content

def test_python_indentation_empty_lines():
    with open("python/battleship.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_comment_between_functions():
    with open("python/outcommented_functions/update.txt", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

# FIXME
def test_python_if_after_comment():

    content = """
if placement.direction == "horizontal":
    for i in range(ship_length):
        if (start_row, start_column + i) in game.board:
            return True
else:
    return False
"""

    code_blocks = parser.parse(content)
    print(code_blocks.to_tree(include_tree_sitter_type=True))

    content = """
if placement.direction == "horizontal":
    for i in range(ship_length):
        if (start_row, start_column + i) in game.board:
            foo = True
else:
    return False
    """
    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_outcommented_method():
    content = """class Battleship(AbstractBattleship):

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        # ... same as before ...

    def get_game(self, game_id: str) -> Game:
        return self.games.get(game_id)
"""
    parser = PythonParser(debug=True, apply_gpt_tweaks=True)

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=False))

    assert code_blocks.to_string() == content

    assert code_blocks.to_tree() == """ 0 module ``
  1 class `Battleship`
   2 commented_out_code `def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:`
    3 commented_out_code `# ... same as before ...`
   2 function `get_game`
    3 code `return self.games.get(game_id)`
  1 space ``
"""

def test_pyton_outcommented_code():
    content = """
class WordSearch:
    # ... existing methods remain unchanged ...

    def search(self, word):
        result = []
        return result or None

    # ... existing methods remain unchanged ..."""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content
    assert code_blocks.to_tree() == """ 0 module ``
  1 class `WordSearch`
   2 commented_out_code `# ... existing methods remain unchanged ...`
   2 function `search`
    3 code `result = []`
    3 code `return result or None`
   2 commented_out_code `# ... existing methods remain unchanged ...`
"""

def test_python_method_starting_with_comment():
    content = """def get_game(game_id: str):
    # comment
    return "foo"
"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content
    assert code_blocks.to_tree() == """ 0 module ``
  1 function `get_game`
   2 comment `# comment`
   2 code `return "foo"`
  1 space ``
"""

def test_python_pass_method():
    content = """def sublist(list_one, list_two):
    pass
"""
    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content
    assert code_blocks.to_tree() == """ 0 module ``
  1 function `sublist`
   2 code `pass`
  1 space ``
"""


def test_python_weird_indentation():
    content = """
    def get_game(self, game_id: str) -> Game:
        return self.games.get(game_id)

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        return 1
"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content


def test_python_right_level():
    content = """
class Battleship(AbstractBattleship):
    # foo
    def create_ship_placement(self, game_id, placement):
        a = 2

    def create_turn(self, game_id, turn):
        game = self.get_game(game_id)
"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

def test_python_comments():
    content = """def get_game(self, game_id: str) -> Game:
    # ... code
    # Return ship type when a hit is made"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    content = """def get_game(self, game_id: str) -> Game:  # Add game_id argument
    # ... code"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

def test_python_comments_2():

    content = """class Battleship(AbstractBattleship):
    # ... code

    def get_bar():
        # ... code
        return "bar"

    def set_foo():
        foo = 1

    def set_bar():
        bar = 5"""
    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

def test_python_comment_on_line():
    content = """def get_foo():  # Add game_id argument
    return "foo"
"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

def test_python_function_after_init():
    content = """class Battleship(AbstractBattleship):
    def __init__(self):
        self.game_id = None

    def create_game(self):
        self.game_id = "1"
"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))


def test_python_print():
    content = """print("Hello, World!")"""

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))