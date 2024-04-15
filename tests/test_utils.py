from moatless.coder.code_utils import do_diff


def assert_diff(original_code, result, expected_diff):
    diff = do_diff("", original_code, result.content)
    print(f"Diff: \n{diff}")
    diff = "\n".join([line.strip() for line in diff.split("\n")])
    expected_span_diff = "\n".join([line.strip() for line in expected_diff.split("\n")])
    assert diff == expected_span_diff
