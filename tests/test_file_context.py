from moatless.benchmark.swebench import setup_swebench_repo
from moatless.benchmark.utils import get_moatless_instance
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.module import Module
from moatless.file_context import FileContext, ContextFile
from moatless.repository import FileRepository
from moatless.repository.repository import InMemRepository
from moatless.schema import FileWithSpans
import pytest
from unittest.mock import Mock, patch
import textwrap
import os
import tempfile
import subprocess
from git import Repo


def test_file_context_to_dict():
    repo_dir = setup_swebench_repo(instance_id="psf__requests-863")
    file_context = FileContext.from_dir(repo_dir, max_tokens=5000)
    assert file_context.model_dump() == {"files": [], "max_tokens": 5000}

    file_context.add_span_to_context(
        file_path="requests/models.py", span_id="Request.register_hook"
    )

    dump = file_context.model_dump()
    assert dump == {
        "files": [
            {
                "file_path": "requests/models.py",
                "patch": None,
                "show_all_spans": False,
                "spans": [
                    {"pinned": True, "span_id": "imports"},
                    {"pinned": False, "span_id": "Request.register_hook"},
                    {"pinned": False, "span_id": "Request"},
                    {"pinned": False, "span_id": "Request.__init__"},
                ],
            }
        ],
        "max_tokens": 5000,
    }

    prompt = file_context.create_prompt(show_outcommented_code=False)
    assert "import os" in prompt
    assert "class Request(object):" in prompt
    assert "def __init__(self" in prompt
    assert "def register_hook(self, event, hook):" in prompt

    file_context = FileContext.from_dict(dump, repo_dir=repo_dir)
    assert file_context.model_dump() == dump
    assert file_context.create_prompt(show_outcommented_code=False) == prompt


def test_to_prompt_string_outcommented_code_block_with_line_numbers():
    module = Module(type=CodeBlockType.MODULE, start_line=1, content="")
    codeblock1 = CodeBlock(
        type=CodeBlockType.COMMENTED_OUT_CODE,
        start_line=1,
        end_line=1,
        content="# ...",
        pre_lines=0,
    )
    codeblock2 = CodeBlock(
        type=CodeBlockType.COMMENTED_OUT_CODE,
        start_line=3,
        end_line=6,
        content="# ...",
        pre_lines=2,
    )
    codeblock3 = CodeBlock(
        type=CodeBlockType.COMMENT,
        start_line=7,
        end_line=8,
        content="# Regular comment\n# with linebreak",
        pre_lines=2,
    )
    codeblock4 = CodeBlock(
        type=CodeBlockType.COMMENTED_OUT_CODE,
        start_line=9,
        end_line=9,
        content="# ...",
        pre_lines=1,
    )
    module.append_children([codeblock1, codeblock2, codeblock3, codeblock4])

    print(module.to_prompt(show_line_numbers=True))

    assert (
        module.to_prompt(show_line_numbers=True)
        == """      # ...
2     
      # ...
6     
7     # Regular comment
8     # with linebreak
      # ..."""
    )


def test_add_line_span_to_context():
    repo_dir = setup_swebench_repo(instance_id="django__django-13768")
    file_context = FileContext.from_dir(repo_dir, max_tokens=5000)
    file_context.add_line_span_to_context("tests/dispatch/tests.py", 27, 27)

    assert "class Callable:" in file_context.create_prompt()


def test_to_prompt():
    instance_id = "django__django-13768"
    instance = get_moatless_instance(instance_id)
    repo_dir = setup_swebench_repo(instance)

    data = {
        "files": [
            {
                "spans": [
                    {"span_id": "imports", "pinned": True},
                    {"span_id": "Signal.send_robust", "pinned": False},
                    {"span_id": "Signal", "pinned": True},
                    {"span_id": "Signal.__init__", "pinned": False},
                    {"span_id": "Signal.has_listeners", "pinned": False},
                    {"span_id": "Signal._clear_dead_receivers", "pinned": False},
                    {"span_id": "_make_id", "pinned": True},
                ],
                "show_all_spans": False,
                "file_path": "django/dispatch/dispatcher.py",
            }
        ]
    }

    file_context = FileContext.from_dict(data, repo_dir=repo_dir)
    prompt = file_context.create_prompt(
        show_span_ids=False,
        show_line_numbers=True,
        exclude_comments=False,
        show_outcommented_code=True,
        outcomment_code_comment="... rest of the code",
    )

    print(prompt)

    assert prompt.startswith("django/dispatch/dispatcher.py")
    assert "1     import threading" in prompt
    assert "9     def _make_id(target):" in prompt
    assert "224   \n          # ... rest of the code" in prompt


def test_generate_patch():
    original_content = "Line 1\nLine 2\nLine 3\n"
    modified_content = "Line 1 modified\nLine 2\nLine 3\n"

    repo = InMemRepository({"test_file.txt": original_content})
    context_file = ContextFile(file_path="test_file.txt", repo=repo)

    patch = context_file.generate_patch(original_content, modified_content)

    expected_patch = """--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified
 Line 2
 Line 3
"""

    assert patch == expected_patch, "Generated patch does not match expected patch."


def test_generate_one_line_patch():
    original_content = "Content of file 1"
    modified_content = "Modified content of file 1"

    repo = InMemRepository({"file1.txt": original_content})
    context_file = ContextFile(file_path="file1.txt", repo=repo)

    patch = context_file.generate_patch(original_content, modified_content)
    expected_patch = textwrap.dedent("""
        --- a/file1.txt
        +++ b/file1.txt
        @@ -1 +1 @@
        -Content of file 1
        +Modified content of file 1
    """).strip()

    assert patch.strip() == expected_patch


def test_apply_patch_to_content():
    original_content = "Line 1\nLine 2\nLine 3\n"
    patch = """diff --git a/test_file.txt b/test_file.txt\n'
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified
 Line 2
 Line 3
"""

    repo = InMemRepository({"test_file.txt": original_content})
    context_file = ContextFile(file_path="test_file.txt", repo=repo)

    # Apply the generated patch to the original content
    result_content = context_file.apply_patch_to_content(original_content, patch)

    expected_content = "Line 1 modified\nLine 2\nLine 3\n"

    assert (
        result_content == expected_content
    ), "Content after applying patch does not match expected content."


def test_contextfile_flow_verification():
    """
    Tests the flow of generating and applying patches in ContextFile and FileXContext.
    """
    # Setup original and modified contents
    base_content = "Line 1\nLine 2\nLine 3\n"
    new_content1 = "Line 1 modified\nLine 2\nLine 3\n"
    new_content2 = "Line 1 modified\nLine 2 modified\nLine 3\n"
    new_content3 = "Line 1 modified\nLine 2 modified\nLine 3 modified\n"

    # Initialize the mock repository with the base content
    repo = InMemRepository({"test_file.txt": base_content})

    # Initialize the first ContextFile with no initial_patch
    context_file1 = ContextFile(
        content=base_content, file_path="test_file.txt", initial_patch=None, repo=repo
    )

    # First change
    context_file1.apply_changes(new_content1)
    expected_patch1 = context_file1.generate_patch(base_content, new_content1)
    assert (
        context_file1.patch == expected_patch1
    ), "First change patch does not match expected patch."
    assert (
        context_file1.content == new_content1
    ), "Content after first change does not match expected content."

    # Second change
    context_file2 = ContextFile(
        repo=repo, file_path="test_file.txt", initial_patch=context_file1.patch
    )
    context_file2.apply_changes(new_content2)
    expected_patch2 = context_file2.generate_patch(new_content1, new_content2)
    assert (
        context_file2.patch == expected_patch2
    ), "Second change patch does not match expected patch."
    assert (
        context_file2.content == new_content2
    ), "Content after second change does not match expected content."

    # Third change
    context_file3 = ContextFile(
        repo=repo,
        file_path="test_file.txt",
        initial_patch=context_file2.generate_full_patch(),
    )
    context_file3.apply_changes(new_content3)
    expected_patch3 = context_file3.generate_patch(new_content2, new_content3)
    assert (
        context_file3.patch == expected_patch3
    ), "Third change patch does not match expected patch."
    assert (
        context_file3.content == new_content3
    ), "Content after third change does not match expected content."


def test_context_file_model_dump():
    # Setup
    repo = InMemRepository({"test_file.txt": "Original content"})

    # Test without patch
    context_file = ContextFile(file_path="test_file.txt", spans=[], repo=repo)

    dump = context_file.model_dump()
    assert dump == {
        "file_path": "test_file.txt",
        "spans": [],
        "show_all_spans": False,
        "patch": None,
    }

    # Test with patch
    context_file.apply_changes("Modified content")
    dump_with_patch = context_file.model_dump()

    assert dump_with_patch["file_path"] == "test_file.txt"
    assert dump_with_patch["spans"] == []
    assert dump_with_patch["show_all_spans"] == False
    assert dump_with_patch["patch"] is not None
    assert isinstance(dump_with_patch["patch"], str)
    assert "Modified content" in dump_with_patch["patch"]


def test_file_context_model_dump():
    # Setup
    repo = InMemRepository(
        {"file1.txt": "Content of file 1", "file2.txt": "Content of file 2"}
    )
    file_context = FileContext(repo=repo)

    # Add files to the context
    file_context.add_file("file1.txt")
    file_context.add_file("file2.txt")

    # Modify one of the files
    context_file = file_context.get_context_file("file1.txt")
    context_file.apply_changes("Modified content of file 1")

    # Test model dump
    dump = file_context.model_dump()

    assert "max_tokens" in dump
    assert "files" in dump
    assert len(dump["files"]) == 2

    file1_dump = next(f for f in dump["files"] if f["file_path"] == "file1.txt")
    file2_dump = next(f for f in dump["files"] if f["file_path"] == "file2.txt")

    assert file1_dump["patch"] is not None
    assert "Modified content of file 1" in file1_dump["patch"]
    assert "patch" in file2_dump
    assert file2_dump["patch"] is None


def test_generate_full_patch():
    # Create a mock repository
    mock_repo = Mock()
    mock_repo.get_file_content.side_effect = lambda path: {
        "file1.py": "original content 1\n",
        "file2.py": "original content 2\n",
    }.get(path)

    # Create a FileContext instance
    file_context = FileContext(repo=mock_repo)
    file1 = file_context.add_file("file1.py")
    file1.apply_changes("modified content 1\n")
    assert file1.patch is not None
    file2 = file_context.add_file("file2.py")
    file2.apply_changes("modified content 2\n")
    assert file2.patch is not None

    # Generate the full patch
    full_patch = file_context.generate_git_patch()

    # Define the expected patch
    expected_patch = textwrap.dedent("""\
        --- a/file1.py
        +++ b/file1.py
        @@ -1 +1 @@
        -original content 1
        +modified content 1

        --- a/file2.py
        +++ b/file2.py
        @@ -1 +1 @@
        -original content 2
        +modified content 2
    """).strip()

    # Assert that the generated patch matches the expected patch
    assert full_patch.strip() == expected_patch

    print(f"Full patch:\n{full_patch}")

    # Verify that the patch can be applied to a real Git repository
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a new Git repository
        repo = Repo.init(temp_dir)

        # Create the original files
        for file_name, content in [
            ("file1.py", "original content 1\n"),
            ("file2.py", "original content 2\n"),
        ]:
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "w") as f:
                f.write(content)
            repo.index.add([file_name])

        # Commit the original files
        repo.index.commit("Initial commit")

        # Apply the patch
        patch_path = os.path.join(temp_dir, "changes.patch")
        with open(patch_path, "w") as f:
            f.write(full_patch)

        print("Original file contents:")
        for file_name in ["file1.py", "file2.py"]:
            with open(os.path.join(temp_dir, file_name), "r") as f:
                print(f"{file_name}:\n{f.read()}\n")

        try:
            # Use Git's apply command to apply the patch with verbose output
            result = subprocess.run(
                ["git", "apply", "--verbose", patch_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"Git apply output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error output:\n{e.stderr}")
            print(f"Standard output:\n{e.stdout}")
            with open(patch_path, "r") as f:
                print(f"Patch content:\n{f.read()}")

            print("File contents after failed patch attempt:")
            for file_name in ["file1.py", "file2.py"]:
                with open(os.path.join(temp_dir, file_name), "r") as f:
                    print(f"{file_name}:\n{f.read()}\n")

            pytest.fail(f"Failed to apply the patch to the Git repository: {e}")

        print("File contents after applying patch:")
        for file_name, expected_content in [
            ("file1.py", "modified content 1\n"),
            ("file2.py", "modified content 2\n"),
        ]:
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "r") as f:
                content = f.read()
                print(f"{file_name}:\n{content}\n")
                assert (
                    content == expected_content
                ), f"Content of {file_name} does not match expected content after applying patch"

        print("Patch successfully applied and verified in a real Git repository")
