from moatless.benchmark.swebench import setup_swebench_repo, create_workspace
from moatless.benchmark.utils import get_moatless_instance
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.module import Module
from moatless.file_context import FileContext
from moatless.schema import FileWithSpans


def test_file_context_to_dict():
    repo_dir = setup_swebench_repo(instance_id="psf__requests-863")
    file_context = FileContext.from_dir(repo_dir, max_tokens=5000)
    assert file_context.model_dump() == {"files": [], "max_tokens": 5000}

    file_context.add_span_to_context(
        file_path="requests/models.py", span_id="Request.register_hook", tokens=500
    )

    dump = file_context.model_dump()
    assert dump == {
        "files": [
            {
                "file_path": "requests/models.py",
                "show_all_spans": False,
                "spans": [
                    {"pinned": True, "span_id": "imports"},
                    {
                        "pinned": False,
                        "span_id": "Request.register_hook",
                        "tokens": 500,
                    },
                    {"pinned": False, "span_id": "Request"},
                    {"pinned": False, "span_id": "Request.__init__"},
                ],
            }
        ],
        "max_tokens": 5000,
    }

    prompt = file_context.create_prompt(show_outcommented_code=True)
    assert "class Request(object):" in prompt
    assert "def register_hook(self, event, hook):" in prompt

    file_context = FileContext.from_dict(repo_dir, dump)
    assert file_context.model_dump() == dump
    assert file_context.create_prompt(show_outcommented_code=True) == prompt


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
                    {"span_id": "receiver", "pinned": False},
                    {"span_id": "Signal.has_listeners", "pinned": False},
                    {"span_id": "Signal._clear_dead_receivers", "pinned": False},
                    {"span_id": "_make_id", "pinned": True},
                ],
                "show_all_spans": False,
                "file_path": "django/dispatch/dispatcher.py",
            }
        ]
    }

    print("prompt")
    file_context = FileContext.from_dict(repo_dir, data)
    prompt = file_context.create_prompt(
        show_span_ids=False,
        show_line_numbers=True,
        exclude_comments=False,
        show_outcommented_code=True,
        outcomment_code_comment="... rest of the code",
    )

    assert prompt.startswith("django/dispatch/dispatcher.py")
    assert "1     import threading" in prompt
    assert "9     def _make_id(target):" in prompt
