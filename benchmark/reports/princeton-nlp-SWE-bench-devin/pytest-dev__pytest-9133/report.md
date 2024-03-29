# pytest-dev__pytest-9133

| **pytest-dev/pytest** | `7720154ca023da23581d87244a31acf5b14979f2` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 185 |
| **Any found context length** | 185 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/pytester.py b/src/_pytest/pytester.py
--- a/src/_pytest/pytester.py
+++ b/src/_pytest/pytester.py
@@ -589,6 +589,7 @@ def assert_outcomes(
         xpassed: int = 0,
         xfailed: int = 0,
         warnings: int = 0,
+        deselected: int = 0,
     ) -> None:
         """Assert that the specified outcomes appear with the respective
         numbers (0 means it didn't occur) in the text output from a test run."""
@@ -605,6 +606,7 @@ def assert_outcomes(
             xpassed=xpassed,
             xfailed=xfailed,
             warnings=warnings,
+            deselected=deselected,
         )
 
 
diff --git a/src/_pytest/pytester_assertions.py b/src/_pytest/pytester_assertions.py
--- a/src/_pytest/pytester_assertions.py
+++ b/src/_pytest/pytester_assertions.py
@@ -43,6 +43,7 @@ def assert_outcomes(
     xpassed: int = 0,
     xfailed: int = 0,
     warnings: int = 0,
+    deselected: int = 0,
 ) -> None:
     """Assert that the specified outcomes appear with the respective
     numbers (0 means it didn't occur) in the text output from a test run."""
@@ -56,6 +57,7 @@ def assert_outcomes(
         "xpassed": outcomes.get("xpassed", 0),
         "xfailed": outcomes.get("xfailed", 0),
         "warnings": outcomes.get("warnings", 0),
+        "deselected": outcomes.get("deselected", 0),
     }
     expected = {
         "passed": passed,
@@ -65,5 +67,6 @@ def assert_outcomes(
         "xpassed": xpassed,
         "xfailed": xfailed,
         "warnings": warnings,
+        "deselected": deselected,
     }
     assert obtained == expected

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/pytester.py | 592 | 592 | 1 | 1 | 185
| src/_pytest/pytester.py | 608 | 608 | 1 | 1 | 185
| src/_pytest/pytester_assertions.py | 46 | 46 | 2 | 2 | 454
| src/_pytest/pytester_assertions.py | 59 | 59 | 2 | 2 | 454
| src/_pytest/pytester_assertions.py | 68 | 68 | 2 | 2 | 454


## Problem Statement

```
Add a `deselected` parameter to `assert_outcomes()`
<!--
Thanks for suggesting a feature!

Quick check-list while suggesting features:
-->

#### What's the problem this feature will solve?
<!-- What are you trying to do, that you are unable to achieve with pytest as it currently stands? -->
I'd like to be able to use `pytester.RunResult.assert_outcomes()` to check deselected count.

#### Describe the solution you'd like
<!-- A clear and concise description of what you want to happen. -->
Add a `deselected` parameter to `pytester.RunResult.assert_outcomes()`

<!-- Provide examples of real-world use cases that this would enable and how it solves the problem described above. -->
Plugins that use `pytest_collection_modifyitems` to change the `items` and add change the deselected items need to be tested. Using `assert_outcomes()` to check the deselected count would be helpful.

#### Alternative Solutions
<!-- Have you tried to workaround the problem using a pytest plugin or other tools? Or a different approach to solving this issue? Please elaborate here. -->
Use `parseoutcomes()` instead of `assert_outcomes()`. `parseoutcomes()` returns a dictionary that includes `deselected`, if there are any.
However, if we have a series of tests, some that care about deselected, and some that don't, then we may have some tests using `assert_outcomes()` and some using `parseoutcomes()`, which is slightly annoying.

#### Additional context
<!-- Add any other context, links, etc. about the feature here. -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/pytester.py** | 583 | 608| 185 | 185 | 15716 | 
| **-> 2 <-** | **2 src/_pytest/pytester_assertions.py** | 37 | 70| 269 | 454 | 16240 | 
| 3 | 3 src/_pytest/mark/__init__.py | 238 | 265| 213 | 667 | 18281 | 
| 4 | 3 src/_pytest/mark/__init__.py | 187 | 217| 247 | 914 | 18281 | 
| 5 | **3 src/_pytest/pytester_assertions.py** | 1 | 34| 253 | 1167 | 18281 | 
| 6 | 4 src/_pytest/outcomes.py | 48 | 68| 175 | 1342 | 20049 | 
| 7 | 5 src/_pytest/doctest.py | 1 | 63| 430 | 1772 | 25697 | 
| 8 | **5 src/_pytest/pytester.py** | 425 | 441| 127 | 1899 | 25697 | 
| 9 | **5 src/_pytest/pytester.py** | 547 | 557| 118 | 2017 | 25697 | 
| 10 | 5 src/_pytest/doctest.py | 286 | 296| 128 | 2145 | 25697 | 
| 11 | 5 src/_pytest/doctest.py | 445 | 461| 148 | 2293 | 25697 | 
| 12 | 6 src/_pytest/main.py | 406 | 421| 112 | 2405 | 32620 | 
| 13 | 6 src/_pytest/main.py | 173 | 229| 400 | 2805 | 32620 | 
| 14 | **6 src/_pytest/pytester.py** | 401 | 423| 151 | 2956 | 32620 | 
| 15 | 6 src/_pytest/mark/__init__.py | 74 | 113| 357 | 3313 | 32620 | 
| 16 | **6 src/_pytest/pytester.py** | 1 | 86| 522 | 3835 | 32620 | 
| 17 | 7 testing/python/fixtures.py | 3847 | 4477| 4315 | 8150 | 61489 | 
| 18 | 7 src/_pytest/doctest.py | 66 | 113| 326 | 8476 | 61489 | 
| 19 | 8 src/_pytest/assertion/__init__.py | 22 | 44| 162 | 8638 | 62938 | 
| 20 | 8 src/_pytest/assertion/__init__.py | 154 | 182| 271 | 8909 | 62938 | 
| 21 | 8 src/_pytest/doctest.py | 578 | 663| 786 | 9695 | 62938 | 
| 22 | 9 src/_pytest/assertion/util.py | 371 | 385| 118 | 9813 | 66917 | 
| 23 | 9 src/_pytest/assertion/__init__.py | 1 | 19| 120 | 9933 | 66917 | 
| 24 | 9 src/_pytest/main.py | 53 | 172| 771 | 10704 | 66917 | 
| 25 | 9 src/_pytest/doctest.py | 310 | 376| 605 | 11309 | 66917 | 
| 26 | 10 src/_pytest/unittest.py | 360 | 406| 347 | 11656 | 69903 | 
| 27 | 11 src/_pytest/skipping.py | 262 | 297| 325 | 11981 | 72194 | 
| 28 | 12 testing/python/collect.py | 1200 | 1237| 221 | 12202 | 82669 | 
| 29 | 13 src/_pytest/python.py | 84 | 127| 306 | 12508 | 96637 | 
| 30 | 14 src/pytest/__init__.py | 1 | 74| 659 | 13167 | 97674 | 
| 31 | 15 src/_pytest/cacheprovider.py | 336 | 396| 541 | 13708 | 102342 | 
| 32 | 16 testing/conftest.py | 1 | 64| 384 | 14092 | 103779 | 
| 33 | 16 src/_pytest/unittest.py | 44 | 57| 127 | 14219 | 103779 | 
| 34 | 16 src/_pytest/doctest.py | 531 | 559| 248 | 14467 | 103779 | 
| 35 | 16 src/_pytest/doctest.py | 298 | 309| 123 | 14590 | 103779 | 
| 36 | 16 src/_pytest/skipping.py | 27 | 43| 113 | 14703 | 103779 | 
| 37 | 16 testing/python/collect.py | 292 | 344| 402 | 15105 | 103779 | 
| 38 | 17 testing/example_scripts/issue_519.py | 1 | 33| 362 | 15467 | 104253 | 
| 39 | 17 src/_pytest/assertion/util.py | 1 | 32| 249 | 15716 | 104253 | 
| 40 | 17 testing/python/fixtures.py | 995 | 1964| 6120 | 21836 | 104253 | 
| 41 | 18 src/_pytest/debugging.py | 150 | 235| 650 | 22486 | 107241 | 
| 42 | 18 src/_pytest/python.py | 218 | 250| 342 | 22828 | 107241 | 
| 43 | 18 testing/python/collect.py | 1170 | 1197| 188 | 23016 | 107241 | 
| 44 | 18 testing/python/collect.py | 817 | 849| 222 | 23238 | 107241 | 
| 45 | 18 src/_pytest/assertion/util.py | 388 | 426| 382 | 23620 | 107241 | 
| 46 | 18 src/_pytest/outcomes.py | 25 | 45| 189 | 23809 | 107241 | 
| 47 | 18 testing/python/collect.py | 1325 | 1342| 125 | 23934 | 107241 | 
| 48 | 18 src/_pytest/skipping.py | 230 | 242| 124 | 24058 | 107241 | 
| 49 | 18 testing/python/collect.py | 851 | 864| 128 | 24186 | 107241 | 
| 50 | 18 testing/python/collect.py | 722 | 749| 261 | 24447 | 107241 | 
| 51 | 18 src/_pytest/mark/__init__.py | 1 | 44| 255 | 24702 | 107241 | 
| 52 | 18 testing/python/fixtures.py | 41 | 993| 6183 | 30885 | 107241 | 
| 53 | 18 testing/python/collect.py | 209 | 262| 342 | 31227 | 107241 | 
| 54 | 18 src/_pytest/python.py | 406 | 442| 308 | 31535 | 107241 | 
| 55 | 19 src/_pytest/assertion/truncate.py | 1 | 36| 244 | 31779 | 108025 | 
| 56 | 19 src/_pytest/assertion/util.py | 138 | 175| 314 | 32093 | 108025 | 
| 57 | 19 testing/python/fixtures.py | 2840 | 3845| 6143 | 38236 | 108025 | 
| 58 | 19 src/_pytest/skipping.py | 245 | 259| 152 | 38388 | 108025 | 
| 59 | 19 src/_pytest/skipping.py | 46 | 82| 383 | 38771 | 108025 | 
| 60 | 20 src/_pytest/runner.py | 46 | 67| 137 | 38908 | 112323 | 
| 61 | 20 testing/python/collect.py | 608 | 636| 214 | 39122 | 112323 | 
| 62 | 21 src/_pytest/faulthandler.py | 1 | 32| 225 | 39347 | 113065 | 
| 63 | **21 src/_pytest/pytester.py** | 89 | 124| 237 | 39584 | 113065 | 
| 64 | 21 src/pytest/__init__.py | 76 | 148| 377 | 39961 | 113065 | 
| 65 | 21 src/_pytest/runner.py | 159 | 182| 184 | 40145 | 113065 | 
| 66 | 22 src/_pytest/terminal.py | 114 | 226| 781 | 40926 | 124384 | 
| 67 | 22 src/_pytest/doctest.py | 414 | 442| 212 | 41138 | 124384 | 
| 68 | 22 src/_pytest/doctest.py | 176 | 207| 226 | 41364 | 124384 | 
| 69 | 22 src/_pytest/outcomes.py | 71 | 120| 348 | 41712 | 124384 | 
| 70 | 23 testing/python/approx.py | 737 | 763| 219 | 41931 | 133488 | 
| 71 | 24 src/_pytest/setuponly.py | 1 | 28| 172 | 42103 | 134226 | 
| 72 | 24 src/_pytest/doctest.py | 139 | 173| 292 | 42395 | 134226 | 
| 73 | **24 src/_pytest/pytester.py** | 1663 | 1736| 701 | 43096 | 134226 | 
| 74 | 24 src/_pytest/skipping.py | 195 | 227| 259 | 43355 | 134226 | 
| 75 | 24 src/_pytest/doctest.py | 707 | 729| 204 | 43559 | 134226 | 
| 76 | 25 src/_pytest/setupplan.py | 1 | 41| 269 | 43828 | 134496 | 
| 77 | 25 testing/python/collect.py | 1345 | 1362| 136 | 43964 | 134496 | 
| 78 | 25 testing/python/collect.py | 1365 | 1375| 122 | 44086 | 134496 | 
| 79 | 25 testing/python/collect.py | 899 | 925| 173 | 44259 | 134496 | 
| 80 | 26 testing/python/integration.py | 1 | 42| 296 | 44555 | 137624 | 
| 81 | 26 src/_pytest/faulthandler.py | 67 | 98| 219 | 44774 | 137624 | 
| 82 | 26 src/_pytest/assertion/util.py | 287 | 315| 265 | 45039 | 137624 | 
| 83 | 27 testing/python/metafunc.py | 1626 | 1643| 158 | 45197 | 152282 | 
| 84 | **27 src/_pytest/pytester.py** | 1118 | 1160| 297 | 45494 | 152282 | 
| 85 | 27 src/_pytest/skipping.py | 160 | 192| 225 | 45719 | 152282 | 
| 86 | 27 src/_pytest/runner.py | 366 | 403| 395 | 46114 | 152282 | 
| 87 | 28 src/_pytest/capture.py | 772 | 824| 395 | 46509 | 159687 | 
| 88 | 28 src/_pytest/python.py | 1439 | 1464| 258 | 46767 | 159687 | 
| 89 | 29 src/_pytest/junitxml.py | 379 | 423| 335 | 47102 | 165433 | 
| 90 | 29 src/_pytest/unittest.py | 240 | 290| 364 | 47466 | 165433 | 
| 91 | 29 testing/python/fixtures.py | 1966 | 2838| 5920 | 53386 | 165433 | 
| 92 | 29 src/_pytest/terminal.py | 1319 | 1347| 309 | 53695 | 165433 | 
| 93 | 29 src/_pytest/capture.py | 490 | 545| 453 | 54148 | 165433 | 
| 94 | 29 testing/python/collect.py | 928 | 960| 300 | 54448 | 165433 | 
| 95 | 29 testing/python/collect.py | 963 | 1002| 364 | 54812 | 165433 | 
| 96 | 30 src/_pytest/helpconfig.py | 46 | 98| 373 | 55185 | 167323 | 
| 97 | 31 src/_pytest/stepwise.py | 1 | 36| 218 | 55403 | 168238 | 
| 98 | 31 testing/python/collect.py | 751 | 771| 139 | 55542 | 168238 | 
| 99 | 31 src/_pytest/cacheprovider.py | 445 | 500| 416 | 55958 | 168238 | 
| 100 | 31 testing/python/collect.py | 653 | 681| 202 | 56160 | 168238 | 
| 101 | 31 src/_pytest/doctest.py | 666 | 704| 253 | 56413 | 168238 | 
| 102 | 31 src/_pytest/outcomes.py | 1 | 22| 141 | 56554 | 168238 | 
| 103 | 31 testing/python/metafunc.py | 1 | 29| 151 | 56705 | 168238 | 
| 104 | 31 testing/python/collect.py | 1 | 42| 293 | 56998 | 168238 | 
| 105 | 31 testing/python/collect.py | 346 | 359| 114 | 57112 | 168238 | 
| 106 | 31 testing/python/integration.py | 44 | 78| 253 | 57365 | 168238 | 
| 107 | 31 testing/python/integration.py | 295 | 332| 225 | 57590 | 168238 | 
| 108 | 31 src/_pytest/terminal.py | 1232 | 1262| 311 | 57901 | 168238 | 
| 109 | 32 testing/python/raises.py | 1 | 52| 341 | 58242 | 170465 | 
| 110 | 32 src/_pytest/debugging.py | 43 | 64| 163 | 58405 | 170465 | 
| 111 | 32 testing/python/collect.py | 516 | 527| 110 | 58515 | 170465 | 
| 112 | 32 testing/python/integration.py | 335 | 374| 251 | 58766 | 170465 | 
| 113 | 32 testing/python/collect.py | 529 | 556| 206 | 58972 | 170465 | 
| 114 | 32 src/_pytest/assertion/util.py | 429 | 473| 360 | 59332 | 170465 | 
| 115 | 33 src/_pytest/assertion/rewrite.py | 467 | 499| 223 | 59555 | 180565 | 
| 116 | 33 src/_pytest/runner.py | 118 | 139| 247 | 59802 | 180565 | 
| 117 | 34 src/_pytest/fixtures.py | 311 | 354| 429 | 60231 | 194706 | 
| 118 | 34 testing/python/integration.py | 191 | 217| 225 | 60456 | 194706 | 
| 119 | 34 testing/python/collect.py | 791 | 815| 188 | 60644 | 194706 | 
| 120 | 34 src/_pytest/cacheprovider.py | 291 | 334| 455 | 61099 | 194706 | 
| 121 | 34 src/_pytest/main.py | 377 | 403| 251 | 61350 | 194706 | 
| 122 | 34 testing/python/collect.py | 574 | 591| 188 | 61538 | 194706 | 
| 123 | 34 src/_pytest/python.py | 1405 | 1437| 266 | 61804 | 194706 | 
| 124 | 34 src/_pytest/mark/__init__.py | 268 | 283| 143 | 61947 | 194706 | 
| 125 | 34 src/_pytest/doctest.py | 116 | 136| 160 | 62107 | 194706 | 
| 126 | 34 src/_pytest/unittest.py | 1 | 41| 254 | 62361 | 194706 | 
| 127 | 34 testing/python/collect.py | 265 | 290| 193 | 62554 | 194706 | 
| 128 | 34 src/_pytest/stepwise.py | 92 | 123| 263 | 62817 | 194706 | 


### Hint

```
Sounds reasonable. 👍 
Hi! I would like to work on this proposal. I went ahead and modified `pytester.RunResult.assert_outcomes()` to also compare the `deselected` count to that returned by `parseoutcomes()`. I also modified `pytester_assertions.assert_outcomes()` called by `pytester.RunResult.assert_outcomes()`.

While all tests pass after the above changes, I think none of the tests presently would be using `assert_outcomes()` with deselected as a parameter since it's a new feature, so should I also write a test for the same?  Can you suggest how I should go about doing the same? I am a bit confused as there are many `test_` files.

I am new here, so any help/feedback will be highly appreciated. Thanks!
Looking at the [pull request 8952](https://github.com/pytest-dev/pytest/pull/8952/files) would be a good place to start.
That PR involved adding `warnings` to `assert_outcomes()`, so it will also show you where all you need to make sure to have changes.
It also includes a test. 
The test for `deselected` will have to be different.
Using `-k` or `-m` will be effective in creating deselected tests.
One option is to have two tests,  `test_one` and `test_two`, for example, and call `pytester.runpytest("-k", "two")`.
That should produce one passed and one deselected.
One exception to the necessary changes. I don't think `test_nose.py` mods would be necessary. Of course, I'd double check with @nicoddemus.
didn't mean to close
```

## Patch

```diff
diff --git a/src/_pytest/pytester.py b/src/_pytest/pytester.py
--- a/src/_pytest/pytester.py
+++ b/src/_pytest/pytester.py
@@ -589,6 +589,7 @@ def assert_outcomes(
         xpassed: int = 0,
         xfailed: int = 0,
         warnings: int = 0,
+        deselected: int = 0,
     ) -> None:
         """Assert that the specified outcomes appear with the respective
         numbers (0 means it didn't occur) in the text output from a test run."""
@@ -605,6 +606,7 @@ def assert_outcomes(
             xpassed=xpassed,
             xfailed=xfailed,
             warnings=warnings,
+            deselected=deselected,
         )
 
 
diff --git a/src/_pytest/pytester_assertions.py b/src/_pytest/pytester_assertions.py
--- a/src/_pytest/pytester_assertions.py
+++ b/src/_pytest/pytester_assertions.py
@@ -43,6 +43,7 @@ def assert_outcomes(
     xpassed: int = 0,
     xfailed: int = 0,
     warnings: int = 0,
+    deselected: int = 0,
 ) -> None:
     """Assert that the specified outcomes appear with the respective
     numbers (0 means it didn't occur) in the text output from a test run."""
@@ -56,6 +57,7 @@ def assert_outcomes(
         "xpassed": outcomes.get("xpassed", 0),
         "xfailed": outcomes.get("xfailed", 0),
         "warnings": outcomes.get("warnings", 0),
+        "deselected": outcomes.get("deselected", 0),
     }
     expected = {
         "passed": passed,
@@ -65,5 +67,6 @@ def assert_outcomes(
         "xpassed": xpassed,
         "xfailed": xfailed,
         "warnings": warnings,
+        "deselected": deselected,
     }
     assert obtained == expected

```

## Test Patch

```diff
diff --git a/testing/test_pytester.py b/testing/test_pytester.py
--- a/testing/test_pytester.py
+++ b/testing/test_pytester.py
@@ -861,3 +861,17 @@ def test_with_warning():
     )
     result = pytester.runpytest()
     result.assert_outcomes(passed=1, warnings=1)
+
+
+def test_pytester_outcomes_deselected(pytester: Pytester) -> None:
+    pytester.makepyfile(
+        """
+        def test_one():
+            pass
+
+        def test_two():
+            pass
+        """
+    )
+    result = pytester.runpytest("-k", "test_one")
+    result.assert_outcomes(passed=1, deselected=1)

```


## Code snippets

### 1 - src/_pytest/pytester.py:

Start line: 583, End line: 608

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
        warnings: int = 0,
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
            warnings=warnings,
        )
```
### 2 - src/_pytest/pytester_assertions.py:

Start line: 37, End line: 70

```python
def assert_outcomes(
    outcomes: Dict[str, int],
    passed: int = 0,
    skipped: int = 0,
    failed: int = 0,
    errors: int = 0,
    xpassed: int = 0,
    xfailed: int = 0,
    warnings: int = 0,
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
        "warnings": outcomes.get("warnings", 0),
    }
    expected = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "xpassed": xpassed,
        "xfailed": xfailed,
        "warnings": warnings,
    }
    assert obtained == expected
```
### 3 - src/_pytest/mark/__init__.py:

Start line: 238, End line: 265

```python
def deselect_by_mark(items: "List[Item]", config: Config) -> None:
    matchexpr = config.option.markexpr
    if not matchexpr:
        return

    expr = _parse_expression(matchexpr, "Wrong expression passed to '-m'")
    remaining: List[Item] = []
    deselected: List[Item] = []
    for item in items:
        if expr.evaluate(MarkMatcher.from_item(item)):
            remaining.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def _parse_expression(expr: str, exc_message: str) -> Expression:
    try:
        return Expression.compile(expr)
    except ParseError as e:
        raise UsageError(f"{exc_message}: {expr}: {e}") from None


def pytest_collection_modifyitems(items: "List[Item]", config: Config) -> None:
    deselect_by_keyword(items, config)
    deselect_by_mark(items, config)
```
### 4 - src/_pytest/mark/__init__.py:

Start line: 187, End line: 217

```python
def deselect_by_keyword(items: "List[Item]", config: Config) -> None:
    keywordexpr = config.option.keyword.lstrip()
    if not keywordexpr:
        return

    if keywordexpr.startswith("-"):
        # To be removed in pytest 8.0.0.
        warnings.warn(MINUS_K_DASH, stacklevel=2)
        keywordexpr = "not " + keywordexpr[1:]
    selectuntil = False
    if keywordexpr[-1:] == ":":
        # To be removed in pytest 8.0.0.
        warnings.warn(MINUS_K_COLON, stacklevel=2)
        selectuntil = True
        keywordexpr = keywordexpr[:-1]

    expr = _parse_expression(keywordexpr, "Wrong expression passed to '-k'")

    remaining = []
    deselected = []
    for colitem in items:
        if keywordexpr and not expr.evaluate(KeywordMatcher.from_item(colitem)):
            deselected.append(colitem)
        else:
            if selectuntil:
                keywordexpr = None
            remaining.append(colitem)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining
```
### 5 - src/_pytest/pytester_assertions.py:

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
### 6 - src/_pytest/outcomes.py:

Start line: 48, End line: 68

```python
TEST_OUTCOME = (OutcomeException, Exception)


class Skipped(OutcomeException):
    # XXX hackish: on 3k we fake to live in the builtins
    # in order to have Skipped exception printing shorter/nicer
    __module__ = "builtins"

    def __init__(
        self,
        msg: Optional[str] = None,
        pytrace: bool = True,
        allow_module_level: bool = False,
        *,
        _use_item_location: bool = False,
    ) -> None:
        super().__init__(msg=msg, pytrace=pytrace)
        self.allow_module_level = allow_module_level
        # If true, the skip location is reported as the item's location,
        # instead of the place that raises the exception/calls skip().
        self._use_item_location = _use_item_location
```
### 7 - src/_pytest/doctest.py:

Start line: 1, End line: 63

```python
"""Discover and run doctests in modules and test files."""
import bdb
import inspect
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import legacy_path
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning

if TYPE_CHECKING:
    import doctest

DOCTEST_REPORT_CHOICE_NONE = "none"
DOCTEST_REPORT_CHOICE_CDIFF = "cdiff"
DOCTEST_REPORT_CHOICE_NDIFF = "ndiff"
DOCTEST_REPORT_CHOICE_UDIFF = "udiff"
DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE = "only_first_failure"

DOCTEST_REPORT_CHOICES = (
    DOCTEST_REPORT_CHOICE_NONE,
    DOCTEST_REPORT_CHOICE_CDIFF,
    DOCTEST_REPORT_CHOICE_NDIFF,
    DOCTEST_REPORT_CHOICE_UDIFF,
    DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE,
)

# Lazy definition of runner class
RUNNER_CLASS = None
# Lazy definition of output checker class
CHECKER_CLASS: Optional[Type["doctest.OutputChecker"]] = None
```
### 8 - src/_pytest/pytester.py:

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
### 9 - src/_pytest/pytester.py:

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
### 10 - src/_pytest/doctest.py:

Start line: 286, End line: 296

```python
class DoctestItem(pytest.Item):

    def runtest(self) -> None:
        assert self.dtest is not None
        assert self.runner is not None
        _check_all_skipped(self.dtest)
        self._disable_output_capturing_for_darwin()
        failures: List["doctest.DocTestFailure"] = []
        # Type ignored because we change the type of `out` from what
        # doctest expects.
        self.runner.run(self.dtest, out=failures)  # type: ignore[arg-type]
        if failures:
            raise MultipleDoctestFailures(failures)
```
### 14 - src/_pytest/pytester.py:

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
### 16 - src/_pytest/pytester.py:

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
### 63 - src/_pytest/pytester.py:

Start line: 89, End line: 124

```python
def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--lsof",
        action="store_true",
        dest="lsof",
        default=False,
        help="run FD checks if lsof is available",
    )

    parser.addoption(
        "--runpytest",
        default="inprocess",
        dest="runpytest",
        choices=("inprocess", "subprocess"),
        help=(
            "run pytest sub runs in tests using an 'inprocess' "
            "or 'subprocess' (python -m main) method"
        ),
    )

    parser.addini(
        "pytester_example_dir", help="directory to take the pytester example files from"
    )


def pytest_configure(config: Config) -> None:
    if config.getvalue("lsof"):
        checker = LsofFdLeakChecker()
        if checker.matching_platform():
            config.pluginmanager.register(checker)

    config.addinivalue_line(
        "markers",
        "pytester_example_path(*path_segments): join the given path "
        "segments to `pytester_example_dir` for this test.",
    )
```
### 73 - src/_pytest/pytester.py:

Start line: 1663, End line: 1736

```python
@final
@attr.s(repr=False, str=False, init=False)
class Testdir:

    def runpytest(self, *args, **kwargs) -> RunResult:
        """See :meth:`Pytester.runpytest`."""
        return self._pytester.runpytest(*args, **kwargs)

    def parseconfig(self, *args) -> Config:
        """See :meth:`Pytester.parseconfig`."""
        return self._pytester.parseconfig(*args)

    def parseconfigure(self, *args) -> Config:
        """See :meth:`Pytester.parseconfigure`."""
        return self._pytester.parseconfigure(*args)

    def getitem(self, source, funcname="test_func"):
        """See :meth:`Pytester.getitem`."""
        return self._pytester.getitem(source, funcname)

    def getitems(self, source):
        """See :meth:`Pytester.getitems`."""
        return self._pytester.getitems(source)

    def getmodulecol(self, source, configargs=(), withinit=False):
        """See :meth:`Pytester.getmodulecol`."""
        return self._pytester.getmodulecol(
            source, configargs=configargs, withinit=withinit
        )

    def collect_by_name(
        self, modcol: Collector, name: str
    ) -> Optional[Union[Item, Collector]]:
        """See :meth:`Pytester.collect_by_name`."""
        return self._pytester.collect_by_name(modcol, name)

    def popen(
        self,
        cmdargs,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=CLOSE_STDIN,
        **kw,
    ):
        """See :meth:`Pytester.popen`."""
        return self._pytester.popen(cmdargs, stdout, stderr, stdin, **kw)

    def run(self, *cmdargs, timeout=None, stdin=CLOSE_STDIN) -> RunResult:
        """See :meth:`Pytester.run`."""
        return self._pytester.run(*cmdargs, timeout=timeout, stdin=stdin)

    def runpython(self, script) -> RunResult:
        """See :meth:`Pytester.runpython`."""
        return self._pytester.runpython(script)

    def runpython_c(self, command):
        """See :meth:`Pytester.runpython_c`."""
        return self._pytester.runpython_c(command)

    def runpytest_subprocess(self, *args, timeout=None) -> RunResult:
        """See :meth:`Pytester.runpytest_subprocess`."""
        return self._pytester.runpytest_subprocess(*args, timeout=timeout)

    def spawn_pytest(
        self, string: str, expect_timeout: float = 10.0
    ) -> "pexpect.spawn":
        """See :meth:`Pytester.spawn_pytest`."""
        return self._pytester.spawn_pytest(string, expect_timeout=expect_timeout)

    def spawn(self, cmd: str, expect_timeout: float = 10.0) -> "pexpect.spawn":
        """See :meth:`Pytester.spawn`."""
        return self._pytester.spawn(cmd, expect_timeout=expect_timeout)

    def __repr__(self) -> str:
        return f"<Testdir {self.tmpdir!r}>"

    def __str__(self) -> str:
        return str(self.tmpdir)
```
### 84 - src/_pytest/pytester.py:

Start line: 1118, End line: 1160

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
