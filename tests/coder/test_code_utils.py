from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_utils import create_instruction_code_block


def test_create_instruction_code_block():
    content = """
class ManyToManyRel(ForeignObjectRel):

    # ...

    @property
    def identity(self):
        return super().identity + (
            self.through,
            self.through_fields,
            self.db_constraint,
        )
"""

    parser = PythonParser()

    codeblock = parser.parse(content)

    expected_block = codeblock.find_by_path(["ManyToManyRel", "identity"])

    instruction_code = create_instruction_code_block(expected_block)

    print(instruction_code)

    expected_code = """class ManyToManyRel(ForeignObjectRel):

    @property
    def identity(self):
        # Write the implementation here..."""

    assert instruction_code == expected_code