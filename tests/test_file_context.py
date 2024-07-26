from moatless.benchmark.swebench import setup_swebench_repo
from moatless.file_context import FileContext
from moatless.types import FileWithSpans


def test_file_context_to_dict():
    repo_dir = setup_swebench_repo(instance_id="psf__requests-863")
    file_context = FileContext.from_dir(repo_dir, max_tokens=5000)
    assert file_context.dict() == {'files': [], 'max_tokens': 5000}

    file_context.add_span_to_context(
        file_path="requests/models.py",
        span_id="Request.register_hook",
        tokens=500
    )

    assert file_context.dict() == {
        'files': [
            {
                'file_path': 'requests/models.py',
                'show_all_spans': False,
                'spans': [
                    {
                        'span_id': 'Request.register_hook',
                        'tokens': 500
                    }
                ]
            }
        ],
        'max_tokens': 5000
    }





