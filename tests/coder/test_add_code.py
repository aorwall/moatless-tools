from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.types import UpdateCodeTask
from moatless.coder.update_code import UpdateCodeAction, find_block_and_span


def test_find_block_and_span_with_if_clause():
    content = """
    
    """

    parser = PythonParser(apply_gpt_tweaks=True)
    original_module = parser.parse(content)

    print(original_module.to_tree())

    task = UpdateCodeTask(
        instructions="",
        start_line=105,
        end_line=105,
        span_id="Float.__new__",
        file_path="numbers.py",
    )

    print(original_module.to_prompt(show_line_numbers=True))

    code_block, start_index, end_index = find_block_and_span(
        original_module, "Float.__new__", 105, 105
    )

    print(code_block.to_tree())

    print(code_block.full_path(), start_index, end_index)

    update_code_action = UpdateCodeAction()

    instruction_code = update_code_action._task_instructions(task, original_module)
    print(instruction_code)

