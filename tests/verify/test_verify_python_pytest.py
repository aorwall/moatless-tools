from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as temp_dir:
        #open(os.path.join(temp_dir, "food_chain_test.py"), "w").close()

        yield temp_dir

@pytest.fixture
def verifier(temp_dir):
    return PythonPytestTestTool(current_dir=Path(temp_dir))


def test_no_tests(temp_dir, verifier):
    output = """=============================================== test session starts ===============================================
platform linux -- Python 3.11.5, pytest-7.4.2, pluggy-1.3.0
rootdir: /home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/7afce770-b94b-4ae6-bae2-1f0db81c7bc7
plugins: pyfakefs-5.2.4, anyio-4.0.0
collected 0 items
Clearing the cache

============================================== no tests ran in 0.00s ==============================================
"""

    failed_tests_count, passed_tests_count = verifier.parse_test_results(output)
    assert failed_tests_count == 0
    assert passed_tests_count == 0

    assert "no tests ran" in output