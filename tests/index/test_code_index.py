import pytest

from moatless.benchmark.swebench import (
    setup_swebench_repo,
    create_index,
)
from moatless.benchmark.utils import get_moatless_instance, get_moatless_instances
from moatless.index import IndexSettings, CodeIndex
from moatless.index.code_index import is_test
from moatless.index.settings import CommentStrategy
from moatless.repository import FileRepository


@pytest.mark.parametrize(
    "file_path, expected",
    [
        # Test files in test directories
        ("tests/test_file.py", True),
        ("test/file_test.py", True),
        ("src/tests/unit/test_module.py", True),
        # Test files with test_ prefix
        ("test_requests.py", True),
        ("api/test_views.py", True),
        # Test files with _test suffix
        ("utils_test.py", True),
        ("models/user_test.py", True),
        # Non-test files
        ("main.py", False),
        ("utils.py", False),
        ("api/views.py", False),
        # Edge cases
        ("testing_utils.py", False),  # Contains "test" but not a test file
        ("contest_results.py", False),  # Contains "test" but not a test file
        ("test_data/sample.py", True),  # In a test directory
        ("tests/data/sample_data.py", True),  # In a tests directory
        # Subdirectories with "test" in the name
        ("test_suite/helpers.py", True),
        ("integration_tests/conftest.py", True),
        # Files with "test" in the middle of the name
        ("my_test_utils.py", False),  # Not a standard test file naming convention
        ("tests/my_test_utils.py", True),  # In a tests directory, so considered a test
        # Additional cases
        ("tests/functional/test_api.py", True),
        ("src/package/tests/integration/test_database.py", True),
        ("docs/test_documentation.md", False),  # Not a .py file
        ("test.py", True),  # Simple test file in root
        ("subpackage/test/__init__.py", True),  # __init__.py in test directory
    ],
)
def test_is_test(file_path, expected):
    assert is_test(file_path) == expected


def test_all_instance_test_fils():
    instances = get_moatless_instances()
    missed_test_files = []

    for instance_id, instance in instances.items():
        for test_file in instance["test_file_spans"].keys():
            if not is_test(test_file):
                missed_test_files.append(f"{instance_id}\t{test_file}")

        for expected_file in instance["expected_spans"].keys():
            assert not is_test(
                expected_file
            ), f"{expected_file} for {instance_id} is a test file"

    for missed_test_file in missed_test_files:
        print(missed_test_file)

    assert (
        len(missed_test_files) < 25
    ), f"Expected less than 25 missed test files. Got {len(missed_test_files)} missed matches."


def test_find_function():
    instance_id = "django__django-11039"
    instance = get_moatless_instance(instance_id)
    print(f"Spans: ")
    print(instance["expected_spans"])

    code_index = create_index(instance)
    files = code_index.semantic_search(
        query="non-atomic migration", file_pattern="*.py"
    )

    print(files)


def test_find_test_files():
    instance_id = "sympy__sympy-11400"
    instance = get_moatless_instance(instance_id)
    code_index = create_index(instance)

    files = code_index.find_test_files(
        "sympy/printing/ccode.py", max_results=3
    )
    assert len(files) == 3
    assert files == {
        "sympy/utilities/tests/test_codegen.py",
        "sympy/printing/tests/test_fcode.py",
        "sympy/printing/tests/test_ccode.py",
    }


def test_find_test_files_with_filename_match_but_low_semantic_rank():
    instance_id = "sympy__sympy-12236"
    instance = get_moatless_instance(instance_id)
    code_index = create_index(instance)

    files = code_index.find_test_files(
        "sympy/polys/domains/polynomialring.py", max_results=3
    )
    assert len(files) == 3
    assert "sympy/polys/domains/tests/test_polynomialring.py" in files


def test_find_test_files_by_span():
    instance_id = "django__django-13315"
    instance = get_moatless_instance(instance_id)
    code_index = create_index(instance)

    files = code_index.find_test_files(
        "django/db/models/fields/related.py",
        span_id="ForeignKey.formfield",
        max_results=3,
    )
    assert len(files) == 3
    assert "tests/model_forms/tests.py" in files


def test_find_test_function():
    instance_id = "django__django-13768"
    instance = get_moatless_instance(instance_id)
    code_index = create_index(instance)

    files = code_index.find_test_files(
        "django/dispatch/dispatcher.py", max_results=3, max_spans=1
    )
    for file in files:
        assert "test" in file.span_ids[0].lower()


def test_find_test_function():
    instance_id = "django__django-12983"
    instance = get_moatless_instance(instance_id)
    code_index = create_index(instance)

    query = """set_ticks"""
    files = code_index.find_test_files(
        "django/utils/text.py", query=query, max_results=3, max_spans=1
    )
    for file in files:
        print(file)
        assert "test" in file.span_ids[0].lower()



def test_ingestion():
    index_settings = IndexSettings(
        embed_model="voyage-code-2",
        dimensions=1536,
        language="python",
        min_chunk_size=200,
        chunk_size=750,
        hard_token_limit=3000,
        max_chunks=200,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    instance_id = "django__django-12419"
    instance = get_moatless_instance(instance_id, split="verified")
    repo_dir = setup_swebench_repo(instance)
    print(repo_dir)
    repo = FileRepository(repo_dir)
    code_index = CodeIndex(settings=index_settings, file_repo=repo)

    vectors, indexed_tokens = code_index.run_ingestion(
        num_workers=1, input_files=["django/conf/global_settings.py"]
    )

    results = code_index._vector_search("SECURE_REFERRER_POLICY setting")

    for result in results:
        print(result)


def test_wildcard_file_patterh():
    instance_id = "django__django-12039"
    file_pattern = "**/*.py"
    query = "Index class implementation and CREATE INDEX SQL generation"

    instance = get_moatless_instance(instance_id, split="verified")
    code_index = create_index(instance)

    results = code_index.search(
        query, file_pattern=file_pattern, max_results=250, max_tokens=8000
    )

    for result in results.hits:
        print(result)
