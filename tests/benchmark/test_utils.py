import pytest
from moatless.benchmark.utils import get_missing_spans


def test_get_missing_spans_no_misses():
    expected_files_with_spans = {
        "django/core/cache/backends/filebased.py": ["FileBasedCache.has_key"]
    }
    actual_files_with_spans = {
        "django/core/cache/backends/filebased.py": [
            "FileBasedCache.has_key",
            "FileBasedCache._is_expired",
        ]
    }

    result = get_missing_spans(expected_files_with_spans, actual_files_with_spans)

    assert result == {}, "Expected no missing spans, but got some"


def test_get_missing_spans_with_misses():
    expected_files_with_spans = {
        "django/core/cache/backends/filebased.py": [
            "FileBasedCache.has_key",
            "FileBasedCache.set",
        ]
    }
    actual_files_with_spans = {
        "django/core/cache/backends/filebased.py": [
            "FileBasedCache.has_key",
            "FileBasedCache._is_expired",
        ]
    }

    result = get_missing_spans(expected_files_with_spans, actual_files_with_spans)

    expected_result = {
        "django/core/cache/backends/filebased.py": ["FileBasedCache.set"]
    }
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_get_missing_spans_missing_file():
    expected_files_with_spans = {
        "django/core/cache/backends/filebased.py": ["FileBasedCache.has_key"],
        "django/core/cache/backends/locmem.py": ["LocMemCache.get"],
    }
    actual_files_with_spans = {
        "django/core/cache/backends/filebased.py": [
            "FileBasedCache.has_key",
            "FileBasedCache._is_expired",
        ]
    }

    result = get_missing_spans(expected_files_with_spans, actual_files_with_spans)

    expected_result = {"django/core/cache/backends/locmem.py": ["LocMemCache.get"]}
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
