from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ghostcoder.schema import VerificationFailureItem, VerificationResult
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

    failed_tests_count, passed_tests_count = verifier.get_number_of_tests(output)
    assert failed_tests_count == 0
    assert passed_tests_count == 0

    assert "no tests ran" in output


def test_python_code_test(temp_dir, verifier):
    output = """=============================================== test session starts ===============================================
platform linux -- Python 3.11.5, pytest-7.4.2, pluggy-1.3.0
rootdir: /home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/temp
plugins: pyfakefs-5.2.4, anyio-4.0.0
collected 1 item                                                                                                  

test_tic_tac_toe.py F                                                                                       [100%]Clearing the cache


==================================================== FAILURES =====================================================
____________________________________________________ test_game ____________________________________________________

    def test_game():
        process = subprocess.Popen(
            ['python', 'tic_tac_toe.py'],
            stdout=subprocess.PIPE,
            text=True
        )
    
        output, _ = process.communicate('\n'.join(["0,0", "1,0", "0,1", "1,1", "0,2"]))
    
>       assert "Player 1 won!" in output
E       AssertionError: assert 'Player 1 won!' in ' | | \n-----\n | | \n-----\n | | \n-----\nPlayer 1, enter your move (row,column): '

test_tic_tac_toe.py:12: AssertionError
---------------------------------------------- Captured stderr call -----------------------------------------------
Traceback (most recent call last):
  File "/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/temp/tic_tac_toe.py", line 69, in <module>
    main()
  File "/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/temp/tic_tac_toe.py", line 56, in main
    row, col = prompt_move(player)
               ^^^^^^^^^^^^^^^^^^^
  File "/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/temp/tic_tac_toe.py", line 40, in prompt_move
    move = input(f"Player {player}, enter your move (row,column): ")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
EOFError: EOF when reading a line
============================================= short test summary info =============================================
FAILED test_tic_tac_toe.py::test_game - AssertionError: assert 'Player 1 won!' in ' | | \n-----\n | | \n-----\n | | \n-----\nPlayer 1, enter your move...
================================================ 1 failed in 0.06s ================================================
"""

    failed_tests_count, passed_tests_count = verifier.get_number_of_tests(output)
    assert failed_tests_count == 1
    assert passed_tests_count == 0

    failed_tests = verifier.find_failed_tests(output)
    expected = VerificationFailureItem(
        type='verification_failure',
        test_code='    def test_game():\n        process = subprocess.Popen(\n            [\'python\', \'tic_tac_toe.py\'],\n            stdout=subprocess.PIPE,\n            text=True\n        )\n        output, _ = process.communicate(\'\n\'.join(["0,0", "1,0", "0,1", "1,1", "0,2"]))\n>       assert "Player 1 won!" in output',
        output="AssertionError: assert 'Player 1 won!' in ' | |\n-----\n| |\n-----\n| |\n-----\nPlayer 1, enter your move (row,column): '\n\n",
        test_method='test_game',
        test_class=None,
        test_file='test_tic_tac_toe.py'
    )

    assert failed_tests == [expected]

def test_python_count(temp_dir, verifier):
    output = """====================================== 9 failed, 3 passed, 7 errors in 0.14s ======================================"""

    failed_tests_count, passed_tests_count, error_count = verifier.get_number_of_tests(output)
    assert failed_tests_count == 9
    assert passed_tests_count == 3
    assert error_count == 7

def test_pytest_without_parsing(temp_dir, verifier):
    with open(f"resources/pytest_fail_output_1.txt", 'r') as f:
        output = f.read()

    verifier = PythonPytestTestTool(
        current_dir=Path("/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace/2a877b44-b4ab-4a12-a910-96191d571b9d/9-gpt-3.5-turbo-16k-1.0"),
        parse_test_results=False
    )

    failed_tests_count, passed_tests_count, error_count = verifier.get_number_of_tests(output)
    assert failed_tests_count == 8
    assert passed_tests_count == 11

    failed_tests = verifier.find_failed_tests(output)
    for failure in failed_tests:
        print(failure.output)
    assert len(failed_tests) == 1


def test_with_syntax_error(temp_dir, verifier):
    output = """____________________________________________ test_game_over_condition _____________________________________________

battleship_game = <battleship.Battleship object at 0x7fa4b7e30250>, initialized_game_id = '9528'

    def test_game_over_condition(battleship_game, initialized_game_id):
        for row in range(1, 11):
            for column in list("ABCDEFGHIJ"):
                turn = Turn(target={"row": row, "column": column})
>               battleship_game.create_turn(initialized_game_id, turn)

test_positive.py:144:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
battleship.py:84: in create_turn
    game.is_game_over = True
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   ValueError: "Game" object has no field "is_game_over"

pydantic/main.py:357: ValueError"""


def test_pytest_fail_output_1(temp_dir, verifier):
    with open(f"resources/pytest_fail_output_1.txt", 'r') as f:
        output = f.read()

    failed_tests_count, passed_tests_count, error_count = verifier.get_number_of_tests(output)
    assert failed_tests_count == 8
    assert passed_tests_count == 11

    failed_tests = verifier.find_failed_tests(output)
    for failure in failed_tests:
        print(failure.to_prompt())
    assert len(failed_tests) == 8
