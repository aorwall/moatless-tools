from moatless.coder.search_replace import _get_pre_start_line_index, _get_post_end_line_index, remove_duplicate_lines


def test_get_pre_start_line_index():
    # Test Case 1: Find the nearest non-empty line within 3 lines before start_line
    assert _get_pre_start_line_index(6, ["Line 0", "Line 1", "", "Line 3", "", "Line 5"]) == 3, "Test 1 Failed: Should find 'Line 3' at index 3"

    # Test Case 2: All lines before and including start_line are empty
    try:
        _get_pre_start_line_index(3, ["", "", ""])
        assert False, "Test 2 Failed: Should raise ValueError when no non-empty lines are found"
    except ValueError as e:
        assert str(e) == "No non-empty line found within 3 lines above the start_line."

    # Test Case 3: Start line is the first line and it's non-empty
    assert _get_pre_start_line_index(1, ["Line 0", "Line 1", "Line 2"]) == 0, "Test 3 Failed: Start line is index 0"

    # Test Case 4: No non-empty lines within two lines before start_line, but start_line is non-empty
    assert _get_pre_start_line_index(3, ["", "", "Line 2"]) == 2, "Test 4 Failed: Should return start_line when no non-empty lines found within 2 lines above"

    # Test Case 5: When there are exactly three non-empty lines available within the range
    assert _get_pre_start_line_index(7, ["Line 0", "Line 1", "", "Line 3", "", "Line 5", "Line 6"]) == 3, "Test 5 Failed: Should return the last non-empty line within range"

    # Test Case 6: When the search should consider the start_line
    assert _get_pre_start_line_index(3, ["", "Line 1", ""]) == 1, "Test 6 Failed: Should return index 1"

    # Test Case 7: When all lines within the range are empty but the start_line is non-empty
    assert _get_pre_start_line_index(3, ["", "", "Line 2"]) == 2, "Test 7 Failed: Should return index 2"

    # Test Case 8: Large input with mixed lines
    lines = ["Line " + str(i) if i % 2 == 0 else "" for i in range(1000)]
    assert _get_pre_start_line_index(1000, lines) == 996, "Test 8 Failed: Should return last non-empty line within range"

    # Test Case 9: Start line beyond the range of content_lines
    try:
        _get_pre_start_line_index(5, ["Line 0", "Line 1"])
        assert False, "Test 9 Failed: Should raise IndexError when start_line is beyond content_lines range"
    except IndexError:
        pass


def test_get_post_end_line_index():
    # Test Case 1: Find the nearest non-empty line within 3 lines after end_line
    assert _get_post_end_line_index(3, ["Line 1", "Line 2", "Line 3", "", "", "Line 6"]) == 5, "Test 1 Failed: Should find 'Line 6' at index 5"

    # Test Case 2: All lines after and including end_line are empty
    try:
        _get_post_end_line_index(1, ["", "", ""])
        assert False, "Test 2 Failed: Should raise ValueError when no non-empty lines are found"
    except ValueError as e:
        assert str(e) == "No non-empty line found within 3 lines after the end_line."

    # Test Case 3: End line is the last line and it's non-empty
    assert _get_post_end_line_index(3, ["Line 1", "Line 2", "Line 3"]) == 2, "Test 3 Failed: End line is index 2"

    # Test Case 4: No non-empty lines within two lines after end_line, but end_line is non-empty
    assert _get_post_end_line_index(1, ["Line 1", "", ""]) == 0, "Test 4 Failed: Should return end_line when no non-empty lines found within 2 lines after"

    # Test Case 5: When there are exactly three non-empty lines available within the range
    assert _get_post_end_line_index(1, ["Line 1", "", "Line 3", "Line 4", "Line 5"]) == 3, "Test 5 Failed: Should return the last non-empty line within range"

    # Test Case 6: When the search should consider the end_line
    assert _get_post_end_line_index(1, ["Line 1", "", ""]) == 0, "Test 6 Failed: Should return index 0"

    # Test Case 7: When all lines within the range are empty but the end_line is non-empty
    assert _get_post_end_line_index(1, ["Line 1", "", ""]) == 0, "Test 7 Failed: Should return index 0"

    # Test Case 8: Large input with mixed lines
    lines = ["Line " + str(i) if i % 2 == 0 else "" for i in range(1000)]
    assert _get_post_end_line_index(996, lines) == 998, "Test 8 Failed: Should return last non-empty line within range"

    # Test Case 9: End line beyond the range of content_lines
    try:
        _get_post_end_line_index(5, ["Line 0", "Line 1"])
        assert False, "Test 9 Failed: Should raise IndexError when end_line is beyond content_lines range"
    except IndexError:
        pass


def test_remove_duplicate_lines():
    # Test Case 1: No duplicates
    original_lines = ["foo", "bar"]
    replacement_lines = ["new start", "new end"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == ["new start", "new end"], "Test Case 1 Failed: No duplicates should result in unchanged replacement_lines"

    # Test Case 2: Complete overlap
    original_lines = ["foo", "bar"]
    replacement_lines = ["foo", "bar"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == [], "Test Case 2 Failed: Complete overlap should result in empty replacement_lines"

    # Test Case 3: Partial overlap
    original_lines = ["foo", "bar"]
    replacement_lines = ["new start", "foo", "bar"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == ["new start"], "Test Case 3 Failed: Partial overlap should result in non-overlapping part of replacement_lines"

    # Test Case 4: Overlap with extra lines
    original_lines = ["foo", "bar"]
    replacement_lines = ["new start", "foo", "extra", "bar"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == ["new start", "foo", "extra", "bar"], "Test Case 4 Failed: Overlap with extra lines should keep the entire replacement_lines"

    # Test Case 5: No overlap at all
    original_lines = ["foo", "bar"]
    replacement_lines = ["new start", "new middle", "new end"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == ["new start", "new middle", "new end"], "Test Case 5 Failed: No overlap at all should keep the entire replacement_lines"

    # Test Case 6: Overlap when original_lines is short
    original_lines = ["world"]
    replacement_lines = ["world"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == [], "Test Case 6 Failed: Overlap with short original_lines should result in empty replacement_lines"

    # Test Case 7: Edge case when post_end_line_index is at the end
    original_lines = []
    replacement_lines = ["foo"]
    result = remove_duplicate_lines(replacement_lines, original_lines)
    assert result == ["foo"], "Test Case 7 Failed: When post_end_line_index is at the end, replacement_lines should be unchanged"
