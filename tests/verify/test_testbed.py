import json
from unittest.mock import Mock

import pytest

from moatless.benchmark.swebench import setup_swebench_repo
from moatless.benchmark.utils import get_moatless_instance
from moatless.repository import FileRepository
from moatless.runtime.runtime import TestStatus
from moatless.runtime.testbed import TestbedEnvironment
from testbeds.schema import TestRunResponse
from testbeds.sdk import TestbedSDK
from testbeds.swebench.test_spec import TestSpec
from testbeds.swebench.utils import load_swebench_instance

try:
    import testbeds
    TESTBEDS_AVAILABLE = True
except ImportError:
    TESTBEDS_AVAILABLE = False


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_syntax_error():
    # TODO: We have to load swebench instance as 'version' is missing in moatless instance
    instance = load_swebench_instance("django__django-13710")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/syntax_error.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("django__django-15213")
    repo_dir = setup_swebench_repo(moatless_instance)

    file_repo = FileRepository(repo_path=repo_dir)

    testbed = TestbedEnvironment(testbed=None, repository=file_repo)
    issues = testbed._map_test_results_to_issues(result)
    assert len(issues) == 1
    issue = issues[0]

    assert issue.file_path == "django/contrib/admin/options.py"
    assert issue.span_id == "imports"

    assert len(issue.relevant_files) == 10

    assert [
        (relevant_file.file_path, relevant_file.span_id)
        for relevant_file in issue.relevant_files[:4]
    ] == [
        ("django/contrib/admin/options.py", "imports"),
        ("django/contrib/admin/filters.py", "imports"),
        ("django/contrib/admin/__init__.py", "imports"),
        ("django/apps/config.py", "AppConfig.create"),
    ]


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_syntax_error_2():
    # TODO: We have to load swebench instance as 'version' is missing in moatless instance
    instance = load_swebench_instance("pytest-dev__pytest-5692")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/syntax_error_2.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("pytest-dev__pytest-5692")
    repo_dir = setup_swebench_repo(moatless_instance)

    file_repo = FileRepository(repo_path=repo_dir)

    testbed = TestbedEnvironment(testbed=None, repository=file_repo)
    issues = testbed._map_test_results_to_issues(result)
    assert len(issues) == 1
    issue = issues[0]

    assert issue.file_path == "src/_pytest/junitxml.py"
    assert issue.span_id == "Junit"


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_django_errors():
    instance = load_swebench_instance("django__django-11019")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/django_output_1.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("django__django-11019")
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(testbed=None, repository=file_repo)

    results = testbed._map_test_results_to_issues(result)

    assert len(results) == 77

    errors = [issue for issue in results if issue.status == TestStatus.ERROR]
    assert 55 == len(errors)

    errors = [
        issue for issue in results if issue.status == TestStatus.ERROR and issue.message
    ]
    assert 3 == len(errors)

    for error in errors:
        assert (
            error.relevant_files[0].file_path == "django/forms/widgets.py"
        ), f"Expected first relevant file to be 'widgets.py', but got '{error.relevant_files[0].file_path}'"

    assert [(error.file_path, error.span_id) for error in errors] == [
        ("tests/admin_inlines/tests.py", "TestInline.test_callable_lookup"),
        ("tests/admin_inlines/tests.py", "TestInline.test_can_delete"),
        (
            "tests/forms_tests/tests/test_media.py",
            "FormsMediaTestCase.test_combine_media",
        ),
    ]

    failures = [
        issue
        for issue in results
        if issue.status == TestStatus.FAILED and issue.message
    ]
    assert 4 == len(failures)
    assert [(failure.file_path, failure.span_id) for failure in failures] == [
        (
            "tests/forms_tests/tests/test_media.py",
            "FormsMediaTestCase.test_media_inheritance_from_property",
        ),
        (
            "tests/forms_tests/tests/test_media.py",
            "FormsMediaTestCase.test_media_property",
        ),
        ("tests/forms_tests/tests/test_media.py", "FormsMediaTestCase.test_merge"),
        (
            "tests/forms_tests/tests/test_media.py",
            "FormsMediaTestCase.test_merge_warning",
        ),
    ]

    for failure in failures:
        if failure.relevant_files:
            assert (
                failure.relevant_files[0].file_path == failure.file_path
            ), f"Expected first relevant file to be '{failure.file_path}', but got '{error.relevant_files[0].file_path}'"
            assert (
                failure.relevant_files[0].span_id == failure.span_id
            ), f"Expected first relevant file to be '{failure.span_id}', but got '{error.relevant_files[0].span_id}'"


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_pylint_failures():
    instance = load_swebench_instance("pylint-dev__pylint-7993")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/pylint_output_1.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("pylint-dev__pylint-7993")
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(testbed=None, repository=file_repo)

    results = testbed._map_test_results_to_issues(result)
    assert len(results) == 72

    issues = [
        issue
        for issue in results
        if issue.status in [TestStatus.FAILED, TestStatus.ERROR] and issue.message
    ]
    assert len(issues) == 1
    assert issues[0].file_path == "tests/reporters/unittest_reporting.py"


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_xarray_errors():
    instance = load_swebench_instance("pydata__xarray-3364")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/xarray_output_1.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("pydata__xarray-3364")
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(testbed=None, repository=file_repo)

    results = testbed._map_test_results_to_issues(result)
    assert len(results) == 112

    issues = [
        issue
        for issue in results
        if issue.status in [TestStatus.FAILED, TestStatus.ERROR] and issue.message
    ]
    for issue in issues:
        print("\n\n==========================")
        print(issue.message)
    assert len(issues) == 2
    assert issues[0].file_path == "xarray/tests/test_combine.py"
    assert issues[1].file_path == "xarray/tests/test_combine.py"


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_xarray_errors_2():
    instance = load_swebench_instance("pydata__xarray-3364")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/xarray_output_2.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("pydata__xarray-3364")
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(testbed=None, repository=file_repo)

    results = testbed._map_test_results_to_issues(result)
    assert len(results) == 111

    issues = [
        issue
        for issue in results
        if issue.status in [TestStatus.FAILED, TestStatus.ERROR] and issue.message
    ]

    for issue in issues:
        print("\n\n==========================")
        print(issue.message)

    assert len(issues) == 2

    assert issues[0].file_path == "xarray/tests/test_combine.py"
    assert issues[1].file_path == "xarray/tests/test_concat.py"


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_psf_requests_errors():
    instance = load_swebench_instance("psf__requests-2317")
    test_spec = TestSpec.from_instance(instance)

    with open("tests/verify/data/requests_output_1.txt") as f:
        result = test_spec.parse_logs(f.read())

    moatless_instance = get_moatless_instance("psf__requests-2317")
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(testbed=None, repository=file_repo)

    results = testbed._map_test_results_to_issues(result)
    assert len(results) == 1

    issues = [
        issue
        for issue in results
        if issue.status in [TestStatus.FAILED, TestStatus.ERROR] and issue.message
    ]

    for issue in issues:
        print("\n\n==========================")
        print(issue.message)
    assert len(issues) == 1


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_redundant_tests():
    with open("tests/verify/data/seaborn_output_1.txt") as f:
        log = f.read()

    result = parse_log_pytest(log)
    assert len(result) == 85

    instance = get_moatless_instance("mwaskom__seaborn-3190")
    repo_dir = setup_swebench_repo(instance)

    file_repo = FileRepository(repo_path=repo_dir)

    testbed = TestbedEnvironment(testbed=None, repository=file_repo)
    results = testbed._map_test_results_to_issues(result)
    assert len(results) == 85

    for issue in results:
        if issue.message:
            print(f"\n\n============ {issue.file_path} {issue.span_id} ========")
            print(issue.message)

    issues = [
        issue
        for issue in results
        if issue.status in [TestStatus.FAILED, TestStatus.ERROR] and issue.message
    ]
    assert len(issues) == 1


@pytest.mark.skipif(not TESTBEDS_AVAILABLE, reason="testbeds package not available")
def test_map_test_results_from_json():
    # Load test results from JSON file
    with open("tests/verify/data/test_results.json") as f:
        test_result_dict = json.load(f)
        response = TestRunResponse.model_validate(test_result_dict)

    test_results = response.test_results

    # Set up testbed environment
    moatless_instance = get_moatless_instance("django__django-16139") 
    repo_dir = setup_swebench_repo(moatless_instance)
    file_repo = FileRepository(repo_path=repo_dir)
    testbed = TestbedEnvironment(repository=file_repo, testbed_sdk=Mock(spec=TestbedSDK))

    # Map test results to issues
    issues = testbed._map_test_results_to_issues(test_results)

    # Verify results
    assert len(issues) == 328
