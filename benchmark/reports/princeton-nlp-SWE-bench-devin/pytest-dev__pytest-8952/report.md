# pytest-dev__pytest-8952

| **pytest-dev/pytest** | `6d6bc97231f2d9a68002f1d191828fd3476ca8b8` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 172 |
| **Any found context length** | 172 |
| **Avg pos** | 7.0 |
| **Min pos** | 1 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/pytester.py b/src/_pytest/pytester.py
--- a/src/_pytest/pytester.py
+++ b/src/_pytest/pytester.py
@@ -588,6 +588,7 @@ def assert_outcomes(
         errors: int = 0,
         xpassed: int = 0,
         xfailed: int = 0,
+        warnings: int = 0,
     ) -> None:
         """Assert that the specified outcomes appear with the respective
         numbers (0 means it didn't occur) in the text output from a test run."""
@@ -603,6 +604,7 @@ def assert_outcomes(
             errors=errors,
             xpassed=xpassed,
             xfailed=xfailed,
+            warnings=warnings,
         )
 
 
diff --git a/src/_pytest/pytester_assertions.py b/src/_pytest/pytester_assertions.py
--- a/src/_pytest/pytester_assertions.py
+++ b/src/_pytest/pytester_assertions.py
@@ -42,6 +42,7 @@ def assert_outcomes(
     errors: int = 0,
     xpassed: int = 0,
     xfailed: int = 0,
+    warnings: int = 0,
 ) -> None:
     """Assert that the specified outcomes appear with the respective
     numbers (0 means it didn't occur) in the text output from a test run."""
@@ -54,6 +55,7 @@ def assert_outcomes(
         "errors": outcomes.get("errors", 0),
         "xpassed": outcomes.get("xpassed", 0),
         "xfailed": outcomes.get("xfailed", 0),
+        "warnings": outcomes.get("warnings", 0),
     }
     expected = {
         "passed": passed,
@@ -62,5 +64,6 @@ def assert_outcomes(
         "errors": errors,
         "xpassed": xpassed,
         "xfailed": xfailed,
+        "warnings": warnings,
     }
     assert obtained == expected

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/pytester.py | 591 | 591 | 1 | 1 | 172
| src/_pytest/pytester.py | 606 | 606 | 1 | 1 | 172
| src/_pytest/pytester_assertions.py | 45 | 45 | 4 | 3 | 1007
| src/_pytest/pytester_assertions.py | 57 | 57 | 4 | 3 | 1007
| src/_pytest/pytester_assertions.py | 65 | 65 | 4 | 3 | 1007


## Problem Statement

```
Enhance `RunResult` warning assertion capabilities
while writing some other bits and pieces, I had a use case for checking the `warnings` omitted, `RunResult` has a `assert_outcomes()` that doesn't quite offer `warnings=` yet the information is already available in there, I suspect there is a good reason why we don't have `assert_outcomes(warnings=...)` so I propose some additional capabilities on `RunResult` to handle warnings in isolation.

With `assert_outcomes()` the full dict comparison may get a bit intrusive as far as warning capture is concerned.

something simple like:

\`\`\`python
result = pytester.runpytest(...)
result.assert_warnings(count=1)
\`\`\`

Thoughts?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/pytester.py** | 583 | 606| 172 | 172 | 15703 | 
| 2 | 2 src/_pytest/recwarn.py | 85 | 101| 104 | 276 | 17969 | 
| 3 | 2 src/_pytest/recwarn.py | 104 | 155| 488 | 764 | 17969 | 
| **-> 4 <-** | **3 src/_pytest/pytester_assertions.py** | 37 | 67| 243 | 1007 | 18467 | 
| 5 | **3 src/_pytest/pytester.py** | 547 | 557| 118 | 1125 | 18467 | 
| 6 | 3 src/_pytest/recwarn.py | 1 | 50| 284 | 1409 | 18467 | 
| 7 | 3 src/_pytest/recwarn.py | 261 | 297| 301 | 1710 | 18467 | 
| 8 | 3 src/_pytest/recwarn.py | 230 | 259| 228 | 1938 | 18467 | 
| 9 | 4 src/_pytest/warnings.py | 28 | 79| 387 | 2325 | 19412 | 
| 10 | 4 src/_pytest/recwarn.py | 53 | 82| 291 | 2616 | 19412 | 
| 11 | 4 src/_pytest/warnings.py | 82 | 140| 398 | 3014 | 19412 | 
| 12 | 4 src/_pytest/warnings.py | 1 | 25| 159 | 3173 | 19412 | 
| 13 | 4 src/_pytest/recwarn.py | 158 | 211| 454 | 3627 | 19412 | 
| 14 | **4 src/_pytest/pytester_assertions.py** | 1 | 34| 253 | 3880 | 19412 | 
| 15 | 5 src/_pytest/terminal.py | 266 | 307| 340 | 4220 | 30731 | 
| 16 | 6 src/_pytest/deprecated.py | 84 | 126| 343 | 4563 | 31767 | 
| 17 | 7 src/pytest/__init__.py | 1 | 72| 640 | 5203 | 32774 | 
| 18 | 7 src/_pytest/recwarn.py | 213 | 227| 131 | 5334 | 32774 | 
| 19 | 8 src/_pytest/warning_types.py | 52 | 68| 112 | 5446 | 33469 | 
| 20 | 8 src/_pytest/warning_types.py | 1 | 49| 231 | 5677 | 33469 | 
| 21 | 8 src/_pytest/terminal.py | 911 | 963| 435 | 6112 | 33469 | 
| 22 | **8 src/_pytest/pytester.py** | 425 | 441| 127 | 6239 | 33469 | 
| 23 | 9 src/_pytest/outcomes.py | 48 | 68| 178 | 6417 | 35243 | 
| 24 | 9 src/_pytest/warning_types.py | 71 | 115| 241 | 6658 | 35243 | 
| 25 | 10 src/_pytest/doctest.py | 1 | 63| 430 | 7088 | 40914 | 
| 26 | 10 src/_pytest/warning_types.py | 118 | 133| 111 | 7199 | 40914 | 
| 27 | **10 src/_pytest/pytester.py** | 401 | 423| 151 | 7350 | 40914 | 
| 28 | 11 src/_pytest/config/__init__.py | 1335 | 1376| 332 | 7682 | 53760 | 
| 29 | 11 src/_pytest/config/__init__.py | 1527 | 1553| 218 | 7900 | 53760 | 
| 30 | 12 src/_pytest/junitxml.py | 264 | 278| 144 | 8044 | 59503 | 
| 31 | 13 src/_pytest/assertion/__init__.py | 1 | 19| 120 | 8164 | 60952 | 
| 32 | 13 src/_pytest/deprecated.py | 1 | 82| 693 | 8857 | 60952 | 
| 33 | 14 src/_pytest/nodes.py | 1 | 48| 321 | 9178 | 66334 | 
| 34 | 14 src/_pytest/config/__init__.py | 1607 | 1651| 400 | 9578 | 66334 | 
| 35 | 15 src/_pytest/assertion/util.py | 1 | 32| 249 | 9827 | 70313 | 
| 36 | 15 src/pytest/__init__.py | 74 | 144| 366 | 10193 | 70313 | 
| 37 | 16 testing/python/collect.py | 144 | 185| 265 | 10458 | 80788 | 
| 38 | 16 src/_pytest/assertion/util.py | 138 | 175| 314 | 10772 | 80788 | 
| 39 | 17 src/_pytest/skipping.py | 262 | 297| 325 | 11097 | 83076 | 
| 40 | 17 src/_pytest/assertion/__init__.py | 154 | 182| 271 | 11368 | 83076 | 
| 41 | **17 src/_pytest/pytester.py** | 559 | 581| 202 | 11570 | 83076 | 
| 42 | 18 src/_pytest/hookspec.py | 777 | 809| 313 | 11883 | 90158 | 
| 43 | 18 src/_pytest/outcomes.py | 25 | 45| 192 | 12075 | 90158 | 
| 44 | **18 src/_pytest/pytester.py** | 1116 | 1158| 297 | 12372 | 90158 | 
| 45 | 19 doc/en/conf.py | 350 | 376| 236 | 12608 | 93323 | 
| 46 | **19 src/_pytest/pytester.py** | 1 | 86| 522 | 13130 | 93323 | 
| 47 | 19 src/_pytest/outcomes.py | 1 | 22| 141 | 13271 | 93323 | 
| 48 | 19 src/_pytest/doctest.py | 176 | 209| 234 | 13505 | 93323 | 
| 49 | 20 src/_pytest/runner.py | 265 | 302| 302 | 13807 | 97621 | 
| 50 | 20 src/_pytest/doctest.py | 288 | 298| 128 | 13935 | 97621 | 
| 51 | 20 src/_pytest/outcomes.py | 71 | 120| 348 | 14283 | 97621 | 
| 52 | 20 src/_pytest/runner.py | 202 | 228| 226 | 14509 | 97621 | 
| 53 | 20 src/_pytest/runner.py | 366 | 403| 395 | 14904 | 97621 | 
| 54 | 21 doc/en/example/xfail_demo.py | 1 | 39| 143 | 15047 | 97765 | 
| 55 | 21 src/_pytest/hookspec.py | 812 | 841| 259 | 15306 | 97765 | 
| 56 | 22 testing/python/raises.py | 1 | 52| 341 | 15647 | 99992 | 
| 57 | 23 src/_pytest/debugging.py | 150 | 235| 650 | 16297 | 102980 | 
| 58 | 24 src/_pytest/capture.py | 490 | 545| 453 | 16750 | 110385 | 
| 59 | 24 src/_pytest/hookspec.py | 616 | 632| 136 | 16886 | 110385 | 
| 60 | 25 src/_pytest/unraisableexception.py | 62 | 78| 149 | 17035 | 111117 | 
| 61 | **25 src/_pytest/pytester.py** | 511 | 545| 279 | 17314 | 111117 | 
| 62 | 26 src/_pytest/config/argparsing.py | 1 | 30| 169 | 17483 | 115658 | 
| 63 | 26 src/_pytest/doctest.py | 447 | 463| 148 | 17631 | 115658 | 
| 64 | 26 src/_pytest/unraisableexception.py | 81 | 94| 118 | 17749 | 115658 | 
| 65 | 27 src/_pytest/pathlib.py | 1 | 63| 398 | 18147 | 121209 | 
| 66 | 28 src/_pytest/assertion/rewrite.py | 1064 | 1100| 427 | 18574 | 131306 | 
| 67 | 28 src/_pytest/assertion/util.py | 371 | 385| 118 | 18692 | 131306 | 
| 68 | 29 src/_pytest/unittest.py | 361 | 407| 347 | 19039 | 134295 | 
| 69 | 30 src/_pytest/mark/__init__.py | 1 | 44| 252 | 19291 | 136333 | 
| 70 | 31 src/_pytest/config/compat.py | 1 | 17| 135 | 19426 | 136799 | 
| 71 | 31 src/_pytest/nodes.py | 260 | 295| 251 | 19677 | 136799 | 
| 72 | 31 src/_pytest/config/__init__.py | 1556 | 1586| 185 | 19862 | 136799 | 
| 73 | 31 src/_pytest/skipping.py | 195 | 227| 259 | 20121 | 136799 | 
| 74 | 31 src/_pytest/outcomes.py | 165 | 177| 116 | 20237 | 136799 | 
| 75 | 31 src/_pytest/runner.py | 1 | 44| 278 | 20515 | 136799 | 
| 76 | 31 src/_pytest/assertion/rewrite.py | 298 | 345| 364 | 20879 | 136799 | 
| 77 | 32 src/_pytest/stepwise.py | 89 | 120| 263 | 21142 | 137689 | 
| 78 | 32 src/_pytest/assertion/rewrite.py | 1 | 53| 331 | 21473 | 137689 | 
| 79 | 32 src/_pytest/runner.py | 159 | 182| 184 | 21657 | 137689 | 
| 80 | 32 testing/python/collect.py | 899 | 925| 173 | 21830 | 137689 | 
| 81 | 33 src/_pytest/assertion/truncate.py | 1 | 36| 244 | 22074 | 138473 | 
| 82 | 33 src/_pytest/assertion/rewrite.py | 467 | 499| 223 | 22297 | 138473 | 
| 83 | 34 src/pytest/collect.py | 1 | 40| 220 | 22517 | 138693 | 
| 84 | 35 src/_pytest/python.py | 1 | 80| 594 | 23111 | 152693 | 
| 85 | 35 src/_pytest/skipping.py | 230 | 242| 123 | 23234 | 152693 | 
| 86 | 35 testing/python/collect.py | 1147 | 1167| 176 | 23410 | 152693 | 
| 87 | **35 src/_pytest/pytester.py** | 167 | 189| 243 | 23653 | 152693 | 
| 88 | 36 testing/python/approx.py | 731 | 757| 219 | 23872 | 161754 | 
| 89 | 36 src/_pytest/assertion/__init__.py | 115 | 152| 357 | 24229 | 161754 | 
| 90 | 36 src/_pytest/assertion/__init__.py | 22 | 44| 162 | 24391 | 161754 | 
| 91 | 36 src/_pytest/runner.py | 118 | 139| 247 | 24638 | 161754 | 
| 92 | **36 src/_pytest/pytester.py** | 365 | 399| 220 | 24858 | 161754 | 
| 93 | 36 testing/python/collect.py | 1240 | 1266| 192 | 25050 | 161754 | 
| 94 | 36 src/_pytest/assertion/rewrite.py | 263 | 276| 134 | 25184 | 161754 | 
| 95 | 36 src/_pytest/unittest.py | 334 | 358| 217 | 25401 | 161754 | 
| 96 | 36 src/_pytest/hookspec.py | 635 | 667| 270 | 25671 | 161754 | 
| 97 | 37 src/_pytest/main.py | 53 | 172| 771 | 26442 | 168677 | 
| 98 | 37 src/_pytest/python.py | 217 | 249| 342 | 26784 | 168677 | 
| 99 | 37 src/_pytest/skipping.py | 245 | 259| 152 | 26936 | 168677 | 
| 100 | 38 src/_pytest/compat.py | 323 | 335| 114 | 27050 | 171711 | 
| 101 | 38 src/_pytest/assertion/rewrite.py | 849 | 885| 320 | 27370 | 171711 | 
| 102 | 38 src/_pytest/debugging.py | 367 | 389| 205 | 27575 | 171711 | 
| 103 | 38 src/_pytest/skipping.py | 27 | 43| 113 | 27688 | 171711 | 
| 104 | 38 src/_pytest/doctest.py | 312 | 378| 605 | 28293 | 171711 | 
| 105 | 38 src/_pytest/doctest.py | 139 | 173| 292 | 28585 | 171711 | 
| 106 | 38 src/_pytest/hookspec.py | 734 | 758| 250 | 28835 | 171711 | 
| 107 | 38 src/_pytest/terminal.py | 1004 | 1019| 147 | 28982 | 171711 | 
| 108 | 38 src/_pytest/assertion/rewrite.py | 411 | 432| 208 | 29190 | 171711 | 
| 109 | 38 src/_pytest/assertion/util.py | 429 | 473| 360 | 29550 | 171711 | 
| 110 | 39 src/_pytest/reports.py | 1 | 56| 380 | 29930 | 176064 | 
| 111 | 39 src/_pytest/skipping.py | 46 | 82| 383 | 30313 | 176064 | 
| 112 | 39 src/_pytest/unittest.py | 241 | 291| 364 | 30677 | 176064 | 
| 113 | 39 src/_pytest/unittest.py | 208 | 239| 273 | 30950 | 176064 | 
| 114 | 40 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 31113 | 177713 | 
| 115 | 40 src/_pytest/doctest.py | 669 | 707| 253 | 31366 | 177713 | 
| 116 | 40 src/_pytest/doctest.py | 581 | 666| 792 | 32158 | 177713 | 
| 117 | 41 src/_pytest/_code/code.py | 1 | 54| 348 | 32506 | 187653 | 
| 118 | 41 src/_pytest/capture.py | 1 | 34| 203 | 32709 | 187653 | 
| 119 | 42 src/_pytest/faulthandler.py | 1 | 32| 221 | 32930 | 188391 | 
| 120 | 42 src/_pytest/assertion/util.py | 388 | 426| 382 | 33312 | 188391 | 
| 121 | 42 testing/python/raises.py | 54 | 80| 185 | 33497 | 188391 | 
| 122 | 42 src/_pytest/compat.py | 1 | 73| 437 | 33934 | 188391 | 
| 123 | 42 testing/python/collect.py | 209 | 262| 342 | 34276 | 188391 | 
| 124 | 43 testing/python/integration.py | 44 | 78| 253 | 34529 | 191519 | 
| 125 | 43 src/_pytest/python.py | 164 | 175| 141 | 34670 | 191519 | 
| 126 | 43 src/_pytest/terminal.py | 830 | 846| 145 | 34815 | 191519 | 


## Patch

```diff
diff --git a/src/_pytest/pytester.py b/src/_pytest/pytester.py
--- a/src/_pytest/pytester.py
+++ b/src/_pytest/pytester.py
@@ -588,6 +588,7 @@ def assert_outcomes(
         errors: int = 0,
         xpassed: int = 0,
         xfailed: int = 0,
+        warnings: int = 0,
     ) -> None:
         """Assert that the specified outcomes appear with the respective
         numbers (0 means it didn't occur) in the text output from a test run."""
@@ -603,6 +604,7 @@ def assert_outcomes(
             errors=errors,
             xpassed=xpassed,
             xfailed=xfailed,
+            warnings=warnings,
         )
 
 
diff --git a/src/_pytest/pytester_assertions.py b/src/_pytest/pytester_assertions.py
--- a/src/_pytest/pytester_assertions.py
+++ b/src/_pytest/pytester_assertions.py
@@ -42,6 +42,7 @@ def assert_outcomes(
     errors: int = 0,
     xpassed: int = 0,
     xfailed: int = 0,
+    warnings: int = 0,
 ) -> None:
     """Assert that the specified outcomes appear with the respective
     numbers (0 means it didn't occur) in the text output from a test run."""
@@ -54,6 +55,7 @@ def assert_outcomes(
         "errors": outcomes.get("errors", 0),
         "xpassed": outcomes.get("xpassed", 0),
         "xfailed": outcomes.get("xfailed", 0),
+        "warnings": outcomes.get("warnings", 0),
     }
     expected = {
         "passed": passed,
@@ -62,5 +64,6 @@ def assert_outcomes(
         "errors": errors,
         "xpassed": xpassed,
         "xfailed": xfailed,
+        "warnings": warnings,
     }
     assert obtained == expected

```

## Test Patch

```diff
diff --git a/testing/test_nose.py b/testing/test_nose.py
--- a/testing/test_nose.py
+++ b/testing/test_nose.py
@@ -335,7 +335,7 @@ def test_failing():
         """
     )
     result = pytester.runpytest(p)
-    result.assert_outcomes(skipped=1)
+    result.assert_outcomes(skipped=1, warnings=1)
 
 
 def test_SkipTest_in_test(pytester: Pytester) -> None:
diff --git a/testing/test_pytester.py b/testing/test_pytester.py
--- a/testing/test_pytester.py
+++ b/testing/test_pytester.py
@@ -847,3 +847,17 @@ def test_testdir_makefile_ext_empty_string_makes_file(testdir) -> None:
     """For backwards compat #8192"""
     p1 = testdir.makefile("", "")
     assert "test_testdir_makefile" in str(p1)
+
+
+@pytest.mark.filterwarnings("default")
+def test_pytester_assert_outcomes_warnings(pytester: Pytester) -> None:
+    pytester.makepyfile(
+        """
+        import warnings
+
+        def test_with_warning():
+            warnings.warn(UserWarning("some custom warning"))
+        """
+    )
+    result = pytester.runpytest()
+    result.assert_outcomes(passed=1, warnings=1)

```


## Code snippets

### 1 - src/_pytest/pytester.py:

Start line: 583, End line: 606

```python
class RunResult:

    def assert_outcomes(
        self,
        passed: int = 0,
        skipped: int = 0,
        failed: int = 0,
        errors: int = 0,
        xpassed: int = 0,
        xfailed: int = 0,
    ) -> None:
        """Assert that the specified outcomes appear with the respective
        numbers (0 means it didn't occur) in the text output from a test run."""
        __tracebackhide__ = True
        from _pytest.pytester_assertions import assert_outcomes

        outcomes = self.parseoutcomes()
        assert_outcomes(
            outcomes,
            passed=passed,
            skipped=skipped,
            failed=failed,
            errors=errors,
            xpassed=xpassed,
            xfailed=xfailed,
        )
```
### 2 - src/_pytest/recwarn.py:

Start line: 85, End line: 101

```python
@overload
def warns(
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]] = ...,
    *,
    match: Optional[Union[str, Pattern[str]]] = ...,
) -> "WarningsChecker":
    ...


@overload
def warns(
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]],
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    ...
```
### 3 - src/_pytest/recwarn.py:

Start line: 104, End line: 155

```python
def warns(
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]] = Warning,
    *args: Any,
    match: Optional[Union[str, Pattern[str]]] = None,
    **kwargs: Any,
) -> Union["WarningsChecker", Any]:
    r"""Assert that code raises a particular class of warning.

    Specifically, the parameter ``expected_warning`` can be a warning class or
    sequence of warning classes, and the inside the ``with`` block must issue a warning of that class or
    classes.

    This helper produces a list of :class:`warnings.WarningMessage` objects,
    one for each warning raised.

    This function can be used as a context manager, or any of the other ways
    :func:`pytest.raises` can be used::

        >>> import pytest
        >>> with pytest.warns(RuntimeWarning):
        ...    warnings.warn("my warning", RuntimeWarning)

    In the context manager form you may use the keyword argument ``match`` to assert
    that the warning matches a text or regex::

        >>> with pytest.warns(UserWarning, match='must be 0 or None'):
        ...     warnings.warn("value must be 0 or None", UserWarning)

        >>> with pytest.warns(UserWarning, match=r'must be \d+$'):
        ...     warnings.warn("value must be 42", UserWarning)

        >>> with pytest.warns(UserWarning, match=r'must be \d+$'):
        ...     warnings.warn("this is not here", UserWarning)
        Traceback (most recent call last):
          ...
        Failed: DID NOT WARN. No warnings of type ...UserWarning... was emitted...

    """
    __tracebackhide__ = True
    if not args:
        if kwargs:
            msg = "Unexpected keyword arguments passed to pytest.warns: "
            msg += ", ".join(sorted(kwargs))
            msg += "\nUse context-manager form instead?"
            raise TypeError(msg)
        return WarningsChecker(expected_warning, match_expr=match, _ispytest=True)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(f"{func!r} object (type: {type(func)}) must be callable")
        with WarningsChecker(expected_warning, _ispytest=True):
            return func(*args[1:], **kwargs)
```
### 4 - src/_pytest/pytester_assertions.py:

Start line: 37, End line: 67

```python
def assert_outcomes(
    outcomes: Dict[str, int],
    passed: int = 0,
    skipped: int = 0,
    failed: int = 0,
    errors: int = 0,
    xpassed: int = 0,
    xfailed: int = 0,
) -> None:
    """Assert that the specified outcomes appear with the respective
    numbers (0 means it didn't occur) in the text output from a test run."""
    __tracebackhide__ = True

    obtained = {
        "passed": outcomes.get("passed", 0),
        "skipped": outcomes.get("skipped", 0),
        "failed": outcomes.get("failed", 0),
        "errors": outcomes.get("errors", 0),
        "xpassed": outcomes.get("xpassed", 0),
        "xfailed": outcomes.get("xfailed", 0),
    }
    expected = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "xpassed": xpassed,
        "xfailed": xfailed,
    }
    assert obtained == expected
```
### 5 - src/_pytest/pytester.py:

Start line: 547, End line: 557

```python
class RunResult:

    def parseoutcomes(self) -> Dict[str, int]:
        """Return a dictionary of outcome noun -> count from parsing the terminal
        output that the test process produced.

        The returned nouns will always be in plural form::

            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====

        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.
        """
        return self.parse_summary_nouns(self.outlines)
```
### 6 - src/_pytest/recwarn.py:

Start line: 1, End line: 50

```python
"""Record warnings during test function execution."""
import re
import warnings
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from _pytest.compat import final
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import WARNS_NONE_ARG
from _pytest.fixtures import fixture
from _pytest.outcomes import fail


T = TypeVar("T")


@fixture
def recwarn() -> Generator["WarningsRecorder", None, None]:
    """Return a :class:`WarningsRecorder` instance that records all warnings emitted by test functions.

    See https://docs.python.org/library/how-to/capture-warnings.html for information
    on warning categories.
    """
    wrec = WarningsRecorder(_ispytest=True)
    with wrec:
        warnings.simplefilter("default")
        yield wrec


@overload
def deprecated_call(
    *, match: Optional[Union[str, Pattern[str]]] = ...
) -> "WarningsRecorder":
    ...


@overload
def deprecated_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    ...
```
### 7 - src/_pytest/recwarn.py:

Start line: 261, End line: 297

```python
@final
class WarningsChecker(WarningsRecorder):

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)

        __tracebackhide__ = True

        # only check if we're not currently handling an exception
        if exc_type is None and exc_val is None and exc_tb is None:
            if self.expected_warning is not None:
                if not any(issubclass(r.category, self.expected_warning) for r in self):
                    __tracebackhide__ = True
                    fail(
                        "DID NOT WARN. No warnings of type {} was emitted. "
                        "The list of emitted warnings is: {}.".format(
                            self.expected_warning, [each.message for each in self]
                        )
                    )
                elif self.match_expr is not None:
                    for r in self:
                        if issubclass(r.category, self.expected_warning):
                            if re.compile(self.match_expr).search(str(r.message)):
                                break
                    else:
                        fail(
                            "DID NOT WARN. No warnings of type {} matching"
                            " ('{}') was emitted. The list of emitted warnings"
                            " is: {}.".format(
                                self.expected_warning,
                                self.match_expr,
                                [each.message for each in self],
                            )
                        )
```
### 8 - src/_pytest/recwarn.py:

Start line: 230, End line: 259

```python
@final
class WarningsChecker(WarningsRecorder):
    def __init__(
        self,
        expected_warning: Optional[
            Union[Type[Warning], Tuple[Type[Warning], ...]]
        ] = Warning,
        match_expr: Optional[Union[str, Pattern[str]]] = None,
        *,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        super().__init__(_ispytest=True)

        msg = "exceptions must be derived from Warning, not %s"
        if expected_warning is None:
            warnings.warn(WARNS_NONE_ARG, stacklevel=4)
            expected_warning_tup = None
        elif isinstance(expected_warning, tuple):
            for exc in expected_warning:
                if not issubclass(exc, Warning):
                    raise TypeError(msg % type(exc))
            expected_warning_tup = expected_warning
        elif issubclass(expected_warning, Warning):
            expected_warning_tup = (expected_warning,)
        else:
            raise TypeError(msg % type(expected_warning))

        self.expected_warning = expected_warning_tup
        self.match_expr = match_expr
```
### 9 - src/_pytest/warnings.py:

Start line: 28, End line: 79

```python
@contextmanager
def catch_warnings_for_item(
    config: Config,
    ihook,
    when: "Literal['config', 'collect', 'runtest']",
    item: Optional[Item],
) -> Generator[None, None, None]:
    """Context manager that catches warnings generated in the contained execution block.

    ``item`` can be None if we are not in the context of an item execution.

    Each warning captured triggers the ``pytest_warning_recorded`` hook.
    """
    config_filters = config.getini("filterwarnings")
    cmdline_filters = config.known_args_namespace.pythonwarnings or []
    with warnings.catch_warnings(record=True) as log:
        # mypy can't infer that record=True means log is not None; help it.
        assert log is not None

        if not sys.warnoptions:
            # If user is not explicitly configuring warning filters, show deprecation warnings by default (#2908).
            warnings.filterwarnings("always", category=DeprecationWarning)
            warnings.filterwarnings("always", category=PendingDeprecationWarning)

        apply_warning_filters(config_filters, cmdline_filters)

        # apply filters from "filterwarnings" marks
        nodeid = "" if item is None else item.nodeid
        if item is not None:
            for mark in item.iter_markers(name="filterwarnings"):
                for arg in mark.args:
                    warnings.filterwarnings(*parse_warning_filter(arg, escape=False))

        yield

        for warning_message in log:
            ihook.pytest_warning_captured.call_historic(
                kwargs=dict(
                    warning_message=warning_message,
                    when=when,
                    item=item,
                    location=None,
                )
            )
            ihook.pytest_warning_recorded.call_historic(
                kwargs=dict(
                    warning_message=warning_message,
                    nodeid=nodeid,
                    when=when,
                    location=None,
                )
            )
```
### 10 - src/_pytest/recwarn.py:

Start line: 53, End line: 82

```python
def deprecated_call(
    func: Optional[Callable[..., Any]] = None, *args: Any, **kwargs: Any
) -> Union["WarningsRecorder", Any]:
    """Assert that code produces a ``DeprecationWarning`` or ``PendingDeprecationWarning``.

    This function can be used as a context manager::

        >>> import warnings
        >>> def api_call_v2():
        ...     warnings.warn('use v3 of this api', DeprecationWarning)
        ...     return 200

        >>> import pytest
        >>> with pytest.deprecated_call():
        ...    assert api_call_v2() == 200

    It can also be used by passing a function and ``*args`` and ``**kwargs``,
    in which case it will ensure calling ``func(*args, **kwargs)`` produces one of
    the warnings types above. The return value is the return value of the function.

    In the context manager form you may use the keyword argument ``match`` to assert
    that the warning matches a text or regex.

    The context manager produces a list of :class:`warnings.WarningMessage` objects,
    one for each warning raised.
    """
    __tracebackhide__ = True
    if func is not None:
        args = (func,) + args
    return warns((DeprecationWarning, PendingDeprecationWarning), *args, **kwargs)
```
### 14 - src/_pytest/pytester_assertions.py:

Start line: 1, End line: 34

```python
"""Helper plugin for pytester; should not be loaded on its own."""
# This plugin contains assertions used by pytester. pytester cannot
# contain them itself, since it is imported by the `pytest` module,
# hence cannot be subject to assertion rewriting, which requires a
# module to not be already imported.
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

from _pytest.reports import CollectReport
from _pytest.reports import TestReport


def assertoutcome(
    outcomes: Tuple[
        Sequence[TestReport],
        Sequence[Union[CollectReport, TestReport]],
        Sequence[Union[CollectReport, TestReport]],
    ],
    passed: int = 0,
    skipped: int = 0,
    failed: int = 0,
) -> None:
    __tracebackhide__ = True

    realpassed, realskipped, realfailed = outcomes
    obtained = {
        "passed": len(realpassed),
        "skipped": len(realskipped),
        "failed": len(realfailed),
    }
    expected = {"passed": passed, "skipped": skipped, "failed": failed}
    assert obtained == expected, outcomes
```
### 22 - src/_pytest/pytester.py:

Start line: 425, End line: 441

```python
class HookRecorder:

    def countoutcomes(self) -> List[int]:
        return [len(x) for x in self.listoutcomes()]

    def assertoutcome(self, passed: int = 0, skipped: int = 0, failed: int = 0) -> None:
        __tracebackhide__ = True
        from _pytest.pytester_assertions import assertoutcome

        outcomes = self.listoutcomes()
        assertoutcome(
            outcomes,
            passed=passed,
            skipped=skipped,
            failed=failed,
        )

    def clear(self) -> None:
        self.calls[:] = []
```
### 27 - src/_pytest/pytester.py:

Start line: 401, End line: 423

```python
class HookRecorder:

    def listoutcomes(
        self,
    ) -> Tuple[
        Sequence[TestReport],
        Sequence[Union[CollectReport, TestReport]],
        Sequence[Union[CollectReport, TestReport]],
    ]:
        passed = []
        skipped = []
        failed = []
        for rep in self.getreports(
            ("pytest_collectreport", "pytest_runtest_logreport")
        ):
            if rep.passed:
                if rep.when == "call":
                    assert isinstance(rep, TestReport)
                    passed.append(rep)
            elif rep.skipped:
                skipped.append(rep)
            else:
                assert rep.failed, f"Unexpected outcome: {rep!r}"
                failed.append(rep)
        return passed, skipped, failed
```
### 41 - src/_pytest/pytester.py:

Start line: 559, End line: 581

```python
class RunResult:

    @classmethod
    def parse_summary_nouns(cls, lines) -> Dict[str, int]:
        """Extract the nouns from a pytest terminal summary line.

        It always returns the plural noun for consistency::

            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====

        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.
        """
        for line in reversed(lines):
            if rex_session_duration.search(line):
                outcomes = rex_outcome.findall(line)
                ret = {noun: int(count) for (count, noun) in outcomes}
                break
        else:
            raise ValueError("Pytest terminal summary report not found")

        to_plural = {
            "warning": "warnings",
            "error": "errors",
        }
        return {to_plural.get(k, k): v for k, v in ret.items()}
```
### 44 - src/_pytest/pytester.py:

Start line: 1116, End line: 1158

```python
@final
class Pytester:

    def runpytest_inprocess(
        self, *args: Union[str, "os.PathLike[str]"], **kwargs: Any
    ) -> RunResult:
        """Return result of running pytest in-process, providing a similar
        interface to what self.runpytest() provides."""
        syspathinsert = kwargs.pop("syspathinsert", False)

        if syspathinsert:
            self.syspathinsert()
        now = timing.time()
        capture = _get_multicapture("sys")
        capture.start_capturing()
        try:
            try:
                reprec = self.inline_run(*args, **kwargs)
            except SystemExit as e:
                ret = e.args[0]
                try:
                    ret = ExitCode(e.args[0])
                except ValueError:
                    pass

                class reprec:  # type: ignore
                    ret = ret

            except Exception:
                traceback.print_exc()

                class reprec:  # type: ignore
                    ret = ExitCode(3)

        finally:
            out, err = capture.readouterr()
            capture.stop_capturing()
            sys.stdout.write(out)
            sys.stderr.write(err)

        assert reprec.ret is not None
        res = RunResult(
            reprec.ret, out.splitlines(), err.splitlines(), timing.time() - now
        )
        res.reprec = reprec  # type: ignore
        return res
```
### 46 - src/_pytest/pytester.py:

Start line: 1, End line: 86

```python
"""(Disabled by default) support for testing pytest and pytest plugins.

PYTEST_DONT_REWRITE
"""
import collections.abc
import contextlib
import gc
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from fnmatch import fnmatch
from io import StringIO
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary

import attr
from iniconfig import IniConfig
from iniconfig import SectionWrapper

from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import final
from _pytest.compat import LEGACY_PATH
from _pytest.compat import legacy_path
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import copytree
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning


if TYPE_CHECKING:
    from typing_extensions import Final
    from typing_extensions import Literal

    import pexpect


pytest_plugins = ["pytester_assertions"]


IGNORE_PAM = [  # filenames added when obtaining details about the current user
    "/var/lib/sss/mc/passwd"
]
```
### 61 - src/_pytest/pytester.py:

Start line: 511, End line: 545

```python
class RunResult:
    """The result of running a command."""

    def __init__(
        self,
        ret: Union[int, ExitCode],
        outlines: List[str],
        errlines: List[str],
        duration: float,
    ) -> None:
        try:
            self.ret: Union[int, ExitCode] = ExitCode(ret)
            """The return value."""
        except ValueError:
            self.ret = ret
        self.outlines = outlines
        """List of lines captured from stdout."""
        self.errlines = errlines
        """List of lines captured from stderr."""
        self.stdout = LineMatcher(outlines)
        """:class:`LineMatcher` of stdout.

        Use e.g. :func:`str(stdout) <LineMatcher.__str__()>` to reconstruct stdout, or the commonly used
        :func:`stdout.fnmatch_lines() <LineMatcher.fnmatch_lines()>` method.
        """
        self.stderr = LineMatcher(errlines)
        """:class:`LineMatcher` of stderr."""
        self.duration = duration
        """Duration in seconds."""

    def __repr__(self) -> str:
        return (
            "<RunResult ret=%s len(stdout.lines)=%d len(stderr.lines)=%d duration=%.2fs>"
            % (self.ret, len(self.stdout.lines), len(self.stderr.lines), self.duration)
        )
```
### 87 - src/_pytest/pytester.py:

Start line: 167, End line: 189

```python
class LsofFdLeakChecker:

    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, None, None]:
        lines1 = self.get_open_files()
        yield
        if hasattr(sys, "pypy_version_info"):
            gc.collect()
        lines2 = self.get_open_files()

        new_fds = {t[0] for t in lines2} - {t[0] for t in lines1}
        leaked_files = [t for t in lines2 if t[0] in new_fds]
        if leaked_files:
            error = [
                "***** %s FD leakage detected" % len(leaked_files),
                *(str(f) for f in leaked_files),
                "*** Before:",
                *(str(f) for f in lines1),
                "*** After:",
                *(str(f) for f in lines2),
                "***** %s FD leakage detected" % len(leaked_files),
                "*** function %s:%s: %s " % item.location,
                "See issue #2366",
            ]
            item.warn(PytestWarning("\n".join(error)))
```
### 92 - src/_pytest/pytester.py:

Start line: 365, End line: 399

```python
class HookRecorder:

    @overload
    def getfailures(
        self,
        names: "Literal['pytest_collectreport']",
    ) -> Sequence[CollectReport]:
        ...

    @overload
    def getfailures(
        self,
        names: "Literal['pytest_runtest_logreport']",
    ) -> Sequence[TestReport]:
        ...

    @overload
    def getfailures(
        self,
        names: Union[str, Iterable[str]] = (
            "pytest_collectreport",
            "pytest_runtest_logreport",
        ),
    ) -> Sequence[Union[CollectReport, TestReport]]:
        ...

    def getfailures(
        self,
        names: Union[str, Iterable[str]] = (
            "pytest_collectreport",
            "pytest_runtest_logreport",
        ),
    ) -> Sequence[Union[CollectReport, TestReport]]:
        return [rep for rep in self.getreports(names) if rep.failed]

    def getfailedcollections(self) -> Sequence[CollectReport]:
        return self.getfailures("pytest_collectreport")
```
