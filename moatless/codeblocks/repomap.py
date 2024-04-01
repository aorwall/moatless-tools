import os

from llama_index.core import get_tokenizer

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser


def generate_repomap(repo_dir: str):
    parser = PythonParser()

    repo_map = ""
    file_count = 0

    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
            if not content:
                continue
            codeblock = parser.parse(content)
            stripped_content = _to_context_string(codeblock)
            if stripped_content.strip():
                file_count += 1
                repo_map += f"{file_path}\n{stripped_content}\n\n"

    tokens = len(get_tokenizer()(repo_map))
    print(f"Files: {file_count}")
    print(f"Token count: {tokens}")

    return repomap



def _to_context_string(codeblock: CodeBlock) -> str:
    contents = ""

    if codeblock.pre_lines:
        contents += "\n" * (codeblock.pre_lines - 1)
        for line in codeblock.content_lines:
            if line:
                contents += "\n" + codeblock.indentation + line
            else:
                contents += "\n"
    else:
        contents += codeblock.pre_code + codeblock.content

    has_outcommented_code = False
    for i, child in enumerate(codeblock.children):
        if child.type in [CodeBlockType.CLASS, CodeBlockType.FUNCTION, CodeBlockType.CONSTRUCTOR]:
            if has_outcommented_code:
                contents += child.create_commented_out_block("... code").to_string()
                has_outcommented_code = False
            contents += _to_context_string(codeblock=child)
        else:
            has_outcommented_code = True

    if has_outcommented_code:
        contents += child.create_commented_out_block("... code").to_string()

    return contents

if __name__ == "__main__":
    repomap = generate_repomap("/tmp/repos/django_django/django")
    print(repomap)