# pytest-dev__pytest-8906

| **pytest-dev/pytest** | `69356d20cfee9a81972dcbf93d8caf9eabe113e8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 47480 |
| **Any found context length** | 47480 |
| **Avg pos** | 72.0 |
| **Min pos** | 72 |
| **Max pos** | 72 |
| **Top file pos** | 11 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/python.py b/src/_pytest/python.py
--- a/src/_pytest/python.py
+++ b/src/_pytest/python.py
@@ -608,10 +608,10 @@ def _importtestmodule(self):
             if e.allow_module_level:
                 raise
             raise self.CollectError(
-                "Using pytest.skip outside of a test is not allowed. "
-                "To decorate a test function, use the @pytest.mark.skip "
-                "or @pytest.mark.skipif decorators instead, and to skip a "
-                "module use `pytestmark = pytest.mark.{skip,skipif}."
+                "Using pytest.skip outside of a test will skip the entire module. "
+                "If that's your intention, pass `allow_module_level=True`. "
+                "If you want to skip a specific test or an entire class, "
+                "use the @pytest.mark.skip or @pytest.mark.skipif decorators."
             ) from e
         self.config.pluginmanager.consider_module(mod)
         return mod

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/python.py | 611 | 614 | 72 | 11 | 47480


## Problem Statement

```
Improve handling of skip for module level
This is potentially about updating docs, updating error messages or introducing a new API.

Consider the following scenario:

`pos_only.py` is using Python 3,8 syntax:
\`\`\`python
def foo(a, /, b):
    return a + b
\`\`\`

It should not be tested under Python 3.6 and 3.7.
This is a proper way to skip the test in Python older than 3.8:
\`\`\`python
from pytest import raises, skip
import sys
if sys.version_info < (3, 8):
    skip(msg="Requires Python >= 3.8", allow_module_level=True)

# import must be after the module level skip:
from pos_only import *

def test_foo():
    assert foo(10, 20) == 30
    assert foo(10, b=20) == 30
    with raises(TypeError):
        assert foo(a=10, b=20)
\`\`\`

My actual test involves parameterize and a 3.8 only class, so skipping the test itself is not sufficient because the 3.8 class was used in the parameterization.

A naive user will try to initially skip the module like:

\`\`\`python
if sys.version_info < (3, 8):
    skip(msg="Requires Python >= 3.8")
\`\`\`
This issues this error:

>Using pytest.skip outside of a test is not allowed. To decorate a test function, use the @pytest.mark.skip or @pytest.mark.skipif decorators instead, and to skip a module use `pytestmark = pytest.mark.{skip,skipif}.

The proposed solution `pytestmark = pytest.mark.{skip,skipif}`, does not work  in my case: pytest continues to process the file and fail when it hits the 3.8 syntax (when running with an older version of Python).

The correct solution, to use skip as a function is actively discouraged by the error message.

This area feels a bit unpolished.
A few ideas to improve:

1. Explain skip with  `allow_module_level` in the error message. this seems in conflict with the spirit of the message.
2. Create an alternative API to skip a module to make things easier: `skip_module("reason")`, which can call `_skip(msg=msg, allow_module_level=True)`.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/_pytest/outcomes.py | 123 | 144| 240 | 240 | 1774 | 
| 2 | 2 src/_pytest/skipping.py | 46 | 82| 383 | 623 | 4062 | 
| 3 | 2 src/_pytest/skipping.py | 1 | 24| 143 | 766 | 4062 | 
| 4 | 3 testing/python/collect.py | 608 | 636| 214 | 980 | 14537 | 
| 5 | 4 src/_pytest/doctest.py | 441 | 457| 148 | 1128 | 20174 | 
| 6 | 4 src/_pytest/outcomes.py | 48 | 68| 178 | 1306 | 20174 | 
| 7 | 4 src/_pytest/skipping.py | 160 | 192| 225 | 1531 | 20174 | 
| 8 | 4 testing/python/collect.py | 593 | 606| 111 | 1642 | 20174 | 
| 9 | 4 src/_pytest/skipping.py | 27 | 43| 113 | 1755 | 20174 | 
| 10 | 5 src/_pytest/terminal.py | 1319 | 1347| 309 | 2064 | 31493 | 
| 11 | 5 src/_pytest/skipping.py | 230 | 242| 123 | 2187 | 31493 | 
| 12 | 5 src/_pytest/skipping.py | 245 | 259| 152 | 2339 | 31493 | 
| 13 | 5 src/_pytest/skipping.py | 85 | 157| 560 | 2899 | 31493 | 
| 14 | 5 src/_pytest/doctest.py | 490 | 526| 299 | 3198 | 31493 | 
| 15 | 5 testing/python/collect.py | 1325 | 1342| 125 | 3323 | 31493 | 
| 16 | 5 src/_pytest/skipping.py | 262 | 297| 325 | 3648 | 31493 | 
| 17 | 5 src/_pytest/outcomes.py | 180 | 233| 428 | 4076 | 31493 | 
| 18 | 5 testing/python/collect.py | 71 | 89| 208 | 4284 | 31493 | 
| 19 | 6 src/pytest/__init__.py | 1 | 72| 640 | 4924 | 32500 | 
| 20 | 7 testing/python/metafunc.py | 115 | 137| 190 | 5114 | 47142 | 
| 21 | 8 src/_pytest/deprecated.py | 1 | 82| 693 | 5807 | 48178 | 
| 22 | 9 bench/skip.py | 1 | 10| 0 | 5807 | 48213 | 
| 23 | 9 testing/python/metafunc.py | 1407 | 1431| 214 | 6021 | 48213 | 
| 24 | 10 doc/en/conf.py | 1 | 110| 794 | 6815 | 51378 | 
| 25 | 10 testing/python/collect.py | 963 | 1002| 364 | 7179 | 51378 | 
| 26 | 10 testing/python/collect.py | 1 | 42| 293 | 7472 | 51378 | 
| 27 | **11 src/_pytest/python.py** | 164 | 175| 141 | 7613 | 65376 | 
| 28 | 12 testing/python/fixtures.py | 995 | 1976| 6180 | 13793 | 94138 | 
| 29 | 13 src/_pytest/compat.py | 338 | 422| 572 | 14365 | 97172 | 
| 30 | 13 src/_pytest/deprecated.py | 84 | 126| 343 | 14708 | 97172 | 
| 31 | 13 testing/python/fixtures.py | 41 | 993| 6183 | 20891 | 97172 | 
| 32 | 13 testing/python/fixtures.py | 2999 | 3974| 6078 | 26969 | 97172 | 
| 33 | 13 testing/python/fixtures.py | 2031 | 2942| 6163 | 33132 | 97172 | 
| 34 | 13 src/_pytest/terminal.py | 1395 | 1414| 153 | 33285 | 97172 | 
| 35 | 14 doc/en/example/xfail_demo.py | 1 | 39| 143 | 33428 | 97316 | 
| 36 | 14 testing/python/collect.py | 91 | 126| 253 | 33681 | 97316 | 
| 37 | 15 src/_pytest/mark/__init__.py | 74 | 113| 357 | 34038 | 99345 | 
| 38 | 15 src/_pytest/skipping.py | 195 | 227| 259 | 34297 | 99345 | 
| 39 | 15 testing/python/fixtures.py | 3976 | 4462| 3250 | 37547 | 99345 | 
| 40 | 16 src/_pytest/pytester.py | 1 | 86| 522 | 38069 | 115048 | 
| 41 | 16 testing/python/collect.py | 44 | 69| 215 | 38284 | 115048 | 
| 42 | 17 src/_pytest/main.py | 53 | 172| 771 | 39055 | 121971 | 
| 43 | 18 src/_pytest/fixtures.py | 804 | 824| 173 | 39228 | 136294 | 
| 44 | 19 src/_pytest/pathlib.py | 1 | 63| 398 | 39626 | 141845 | 
| 45 | 19 src/_pytest/compat.py | 1 | 73| 437 | 40063 | 141845 | 
| 46 | 19 src/_pytest/fixtures.py | 479 | 566| 770 | 40833 | 141845 | 
| 47 | 20 src/_pytest/cacheprovider.py | 270 | 288| 171 | 41004 | 146513 | 
| 48 | 20 src/_pytest/doctest.py | 528 | 556| 248 | 41252 | 146513 | 
| 49 | 20 testing/python/metafunc.py | 1721 | 1741| 203 | 41455 | 146513 | 
| 50 | 20 src/_pytest/outcomes.py | 71 | 120| 348 | 41803 | 146513 | 
| 51 | 20 testing/python/metafunc.py | 1640 | 1655| 135 | 41938 | 146513 | 
| 52 | 20 src/pytest/__init__.py | 74 | 144| 366 | 42304 | 146513 | 
| 53 | 21 src/_pytest/unittest.py | 361 | 407| 347 | 42651 | 149502 | 
| 54 | 21 src/_pytest/main.py | 377 | 403| 251 | 42902 | 149502 | 
| 55 | 21 testing/python/metafunc.py | 1689 | 1703| 133 | 43035 | 149502 | 
| 56 | 21 src/_pytest/mark/__init__.py | 271 | 286| 143 | 43178 | 149502 | 
| 57 | 21 testing/python/metafunc.py | 1303 | 1329| 212 | 43390 | 149502 | 
| 58 | 21 testing/python/metafunc.py | 1673 | 1687| 130 | 43520 | 149502 | 
| 59 | 21 src/_pytest/outcomes.py | 165 | 177| 116 | 43636 | 149502 | 
| 60 | 21 testing/python/metafunc.py | 1331 | 1363| 211 | 43847 | 149502 | 
| 61 | 21 testing/python/metafunc.py | 85 | 113| 261 | 44108 | 149502 | 
| 62 | 21 testing/python/metafunc.py | 139 | 186| 471 | 44579 | 149502 | 
| 63 | 21 testing/python/metafunc.py | 1705 | 1719| 138 | 44717 | 149502 | 
| 64 | 21 src/_pytest/mark/__init__.py | 1 | 44| 252 | 44969 | 149502 | 
| 65 | 21 testing/python/collect.py | 899 | 925| 173 | 45142 | 149502 | 
| 66 | 21 testing/python/metafunc.py | 1868 | 1903| 296 | 45438 | 149502 | 
| 67 | 21 testing/python/metafunc.py | 242 | 286| 339 | 45777 | 149502 | 
| 68 | 21 src/_pytest/pathlib.py | 424 | 451| 225 | 46002 | 149502 | 
| 69 | 21 src/_pytest/main.py | 173 | 229| 400 | 46402 | 149502 | 
| 70 | **21 src/_pytest/python.py** | 1659 | 1701| 405 | 46807 | 149502 | 
| 71 | 22 src/_pytest/config/__init__.py | 1527 | 1553| 218 | 47025 | 162348 | 
| **-> 72 <-** | **22 src/_pytest/python.py** | 572 | 617| 455 | 47480 | 162348 | 
| 73 | 23 testing/python/integration.py | 1 | 42| 296 | 47776 | 165476 | 
| 74 | 23 src/_pytest/unittest.py | 241 | 291| 364 | 48140 | 165476 | 
| 75 | 23 testing/python/collect.py | 791 | 815| 188 | 48328 | 165476 | 
| 76 | 23 src/_pytest/doctest.py | 1 | 63| 430 | 48758 | 165476 | 
| 77 | 23 testing/python/metafunc.py | 1389 | 1405| 110 | 48868 | 165476 | 
| 78 | 24 src/_pytest/python_api.py | 779 | 891| 1008 | 49876 | 173907 | 
| 79 | 24 testing/python/metafunc.py | 721 | 751| 239 | 50115 | 173907 | 
| 80 | **24 src/_pytest/python.py** | 1 | 80| 594 | 50709 | 173907 | 
| 81 | 24 testing/python/collect.py | 653 | 681| 202 | 50911 | 173907 | 
| 82 | 24 testing/python/metafunc.py | 1786 | 1809| 230 | 51141 | 173907 | 
| 83 | 24 src/_pytest/config/__init__.py | 776 | 841| 308 | 51449 | 173907 | 
| 84 | 24 testing/python/metafunc.py | 188 | 222| 298 | 51747 | 173907 | 
| 85 | 25 testing/python/raises.py | 1 | 52| 341 | 52088 | 176134 | 
| 86 | 26 src/_pytest/faulthandler.py | 1 | 32| 221 | 52309 | 176872 | 
| 87 | 26 testing/python/metafunc.py | 1 | 28| 144 | 52453 | 176872 | 
| 88 | 26 testing/python/metafunc.py | 1264 | 1281| 180 | 52633 | 176872 | 
| 89 | 26 src/_pytest/config/__init__.py | 416 | 431| 163 | 52796 | 176872 | 
| 90 | 26 testing/python/metafunc.py | 70 | 83| 181 | 52977 | 176872 | 
| 91 | 26 testing/python/collect.py | 866 | 897| 279 | 53256 | 176872 | 
| 92 | 26 testing/python/collect.py | 638 | 651| 112 | 53368 | 176872 | 
| 93 | 27 src/_pytest/nodes.py | 1 | 48| 321 | 53689 | 182254 | 
| 94 | 27 src/_pytest/outcomes.py | 147 | 162| 127 | 53816 | 182254 | 
| 95 | 27 src/_pytest/doctest.py | 66 | 113| 326 | 54142 | 182254 | 
| 96 | 27 testing/python/metafunc.py | 224 | 240| 197 | 54339 | 182254 | 
| 97 | 27 testing/python/raises.py | 184 | 213| 260 | 54599 | 182254 | 
| 98 | 27 testing/python/metafunc.py | 1621 | 1638| 158 | 54757 | 182254 | 
| 99 | 28 src/_pytest/_code/code.py | 1 | 54| 348 | 55105 | 192194 | 


### Hint

```
SyntaxErrors are thrown before execution, so how would the skip call stop the interpreter from parsing the 'incorrect' syntax?
unless we hook the interpreter that is.
A solution could be to ignore syntax errors based on some parameter
if needed we can extend this to have some functionality to evaluate conditions in which syntax errors should be ignored
please note what i suggest will not fix other compatibility issues, just syntax errors

> SyntaxErrors are thrown before execution, so how would the skip call stop the interpreter from parsing the 'incorrect' syntax?

The Python 3.8 code is included by an import. the idea is that the import should not happen if we are skipping the module.
\`\`\`python
if sys.version_info < (3, 8):
    skip(msg="Requires Python >= 3.8", allow_module_level=True)

# import must be after the module level skip:
from pos_only import *
\`\`\`
Hi @omry,

Thanks for raising this.

Definitely we should improve that message. 

> Explain skip with allow_module_level in the error message. this seems in conflict with the spirit of the message.

I'm 👍 on this. 2 is also good, but because `allow_module_level` already exists and is part of the public API, I don't think introducing a new API will really help, better to improve the docs of what we already have.

Perhaps improve the message to something like this:

\`\`\`
Using pytest.skip outside of a test will skip the entire module, if that's your intention pass `allow_module_level=True`. 
If you want to skip a specific test or entire class, use the @pytest.mark.skip or @pytest.mark.skipif decorators.
\`\`\`

I think we can drop the `pytestmark` remark from there, it is not skip-specific and passing `allow_module_level` already accomplishes the same.

Thanks @nicoddemus.

> Using pytest.skip outside of a test will skip the entire module, if that's your intention pass `allow_module_level=True`. 
If you want to skip a specific test or entire class, use the @pytest.mark.skip or @pytest.mark.skipif decorators.

This sounds clearer.
Can you give a bit of context of why the message is there in the first place?
It sounds like we should be able to automatically detect if this is skipping a test or skipping the entire module (based on the fact that we can issue the warning).

Maybe this is addressing some past confusion, or we want to push people toward `pytest.mark.skip[if]`, but if we can detect it automatically - we can also deprecate allow_module_level and make `skip()` do the right thing based on the context it's used in.
> Maybe this is addressing some past confusion

That's exactly it, people would use `@pytest.skip` instead of `@pytest.mark.skip` and skip the whole module:

https://github.com/pytest-dev/pytest/issues/2338#issuecomment-290324255

For that reason we don't really want to automatically detect things, but want users to explicitly pass that flag which proves they are not doing it by accident.

Original issue: https://github.com/pytest-dev/pytest/issues/607
Having looked at the links, I think the alternative API to skip a module is more appealing.
Here is a proposed end state:

1. pytest.skip_module is introduced, can be used to skip a module.
2. pytest.skip() is only legal inside of a test. If called outside of a test, an error message is issues.
Example:

> pytest.skip should only be used inside tests. To skip a module use pytest.skip_module. To completely skip a test function or a test class, use the @pytest.mark.skip or @pytest.mark.skipif decorators.

Getting to this end state would include deprecating allow_module_level first, directing people using pytest.skip(allow_module_level=True) to use pytest.skip_module().

I am also fine with just changing the message as you initially proposed but I feel this proposal will result in an healthier state.

-0.5 from my side - I think this is too minor to warrant another deprecation and change.
I agree it would be healthier, but -1 from me for the same reasons as @The-Compiler: we already had a deprecation/change period in order to introduce `allow_module_level`, having yet another one is frustrating/confusing to users, in comparison to the small gains.
Hi, I see that this is still open. If available, I'd like to take this up.
```

## Patch

```diff
diff --git a/src/_pytest/python.py b/src/_pytest/python.py
--- a/src/_pytest/python.py
+++ b/src/_pytest/python.py
@@ -608,10 +608,10 @@ def _importtestmodule(self):
             if e.allow_module_level:
                 raise
             raise self.CollectError(
-                "Using pytest.skip outside of a test is not allowed. "
-                "To decorate a test function, use the @pytest.mark.skip "
-                "or @pytest.mark.skipif decorators instead, and to skip a "
-                "module use `pytestmark = pytest.mark.{skip,skipif}."
+                "Using pytest.skip outside of a test will skip the entire module. "
+                "If that's your intention, pass `allow_module_level=True`. "
+                "If you want to skip a specific test or an entire class, "
+                "use the @pytest.mark.skip or @pytest.mark.skipif decorators."
             ) from e
         self.config.pluginmanager.consider_module(mod)
         return mod

```

## Test Patch

```diff
diff --git a/testing/test_skipping.py b/testing/test_skipping.py
--- a/testing/test_skipping.py
+++ b/testing/test_skipping.py
@@ -1341,7 +1341,7 @@ def test_func():
     )
     result = pytester.runpytest()
     result.stdout.fnmatch_lines(
-        ["*Using pytest.skip outside of a test is not allowed*"]
+        ["*Using pytest.skip outside of a test will skip the entire module*"]
     )
 
 

```


## Code snippets

### 1 - src/_pytest/outcomes.py:

Start line: 123, End line: 144

```python
@_with_exception(Skipped)
def skip(msg: str = "", *, allow_module_level: bool = False) -> "NoReturn":
    """Skip an executing test with the given message.

    This function should be called only during testing (setup, call or teardown) or
    during collection by using the ``allow_module_level`` flag.  This function can
    be called in doctests as well.

    :param bool allow_module_level:
        Allows this function to be called at module level, skipping the rest
        of the module. Defaults to False.

    .. note::
        It is better to use the :ref:`pytest.mark.skipif ref` marker when
        possible to declare a test to be skipped under certain conditions
        like mismatching platforms or dependencies.
        Similarly, use the ``# doctest: +SKIP`` directive (see `doctest.SKIP
        <https://docs.python.org/3/library/how-to/doctest.html#doctest.SKIP>`_)
        to skip a doctest statically.
    """
    __tracebackhide__ = True
    raise Skipped(msg=msg, allow_module_level=allow_module_level)
```
### 2 - src/_pytest/skipping.py:

Start line: 46, End line: 82

```python
def pytest_configure(config: Config) -> None:
    if config.option.runxfail:
        # yay a hack
        import pytest

        old = pytest.xfail
        config._cleanup.append(lambda: setattr(pytest, "xfail", old))

        def nop(*args, **kwargs):
            pass

        nop.Exception = xfail.Exception  # type: ignore[attr-defined]
        setattr(pytest, "xfail", nop)

    config.addinivalue_line(
        "markers",
        "skip(reason=None): skip the given test function with an optional reason. "
        'Example: skip(reason="no way of currently testing this") skips the '
        "test.",
    )
    config.addinivalue_line(
        "markers",
        "skipif(condition, ..., *, reason=...): "
        "skip the given test function if any of the conditions evaluate to True. "
        "Example: skipif(sys.platform == 'win32') skips the test if we are on the win32 platform. "
        "See https://docs.pytest.org/en/stable/reference/reference.html#pytest-mark-skipif",
    )
    config.addinivalue_line(
        "markers",
        "xfail(condition, ..., *, reason=..., run=True, raises=None, strict=xfail_strict): "
        "mark the test function as an expected failure if any of the conditions "
        "evaluate to True. Optionally specify a reason for better reporting "
        "and run=False if you don't even want to execute the test function. "
        "If only specific exception(s) are expected, you can list them in "
        "raises, and if the test fails in other ways, it will be reported as "
        "a true failure. See https://docs.pytest.org/en/stable/reference/reference.html#pytest-mark-xfail",
    )
```
### 3 - src/_pytest/skipping.py:

Start line: 1, End line: 24

```python
"""Support for skip/xfail functions and markers."""
import os
import platform
import sys
import traceback
from collections.abc import Mapping
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Type

import attr

from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.mark.structures import Mark
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.reports import BaseReport
from _pytest.runner import CallInfo
from _pytest.store import StoreKey
```
### 4 - testing/python/collect.py:

Start line: 608, End line: 636

```python
class TestFunction:

    def test_parametrize_skip(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.skip('')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skipif_no_skip(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_skipif_no_skip(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 1 failed, 2 passed in *"])
```
### 5 - src/_pytest/doctest.py:

Start line: 441, End line: 457

```python
def _check_all_skipped(test: "doctest.DocTest") -> None:
    """Raise pytest.skip() if all examples in the given DocTest have the SKIP
    option set."""
    import doctest

    all_skipped = all(x.options.get(doctest.SKIP, False) for x in test.examples)
    if all_skipped:
        pytest.skip("all tests skipped by +SKIP option")


def _is_mocked(obj: object) -> bool:
    """Return if an object is possibly a mock object by checking the
    existence of a highly improbable attribute."""
    return (
        safe_getattr(obj, "pytest_mock_example_attribute_that_shouldnt_exist", None)
        is not None
    )
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
        OutcomeException.__init__(self, msg=msg, pytrace=pytrace)
        self.allow_module_level = allow_module_level
        # If true, the skip location is reported as the item's location,
        # instead of the place that raises the exception/calls skip().
        self._use_item_location = _use_item_location
```
### 7 - src/_pytest/skipping.py:

Start line: 160, End line: 192

```python
@attr.s(slots=True, frozen=True)
class Skip:
    """The result of evaluate_skip_marks()."""

    reason = attr.ib(type=str, default="unconditional skip")


def evaluate_skip_marks(item: Item) -> Optional[Skip]:
    """Evaluate skip and skipif marks on item, returning Skip if triggered."""
    for mark in item.iter_markers(name="skipif"):
        if "condition" not in mark.kwargs:
            conditions = mark.args
        else:
            conditions = (mark.kwargs["condition"],)

        # Unconditional.
        if not conditions:
            reason = mark.kwargs.get("reason", "")
            return Skip(reason)

        # If any of the conditions are true.
        for condition in conditions:
            result, reason = evaluate_condition(item, mark, condition)
            if result:
                return Skip(reason)

    for mark in item.iter_markers(name="skip"):
        try:
            return Skip(*mark.args, **mark.kwargs)
        except TypeError as e:
            raise TypeError(str(e) + " - maybe you meant pytest.mark.skipif?") from None

    return None
```
### 8 - testing/python/collect.py:

Start line: 593, End line: 606

```python
class TestFunction:

    def test_parametrize_skipif(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip_if(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])
```
### 9 - src/_pytest/skipping.py:

Start line: 27, End line: 43

```python
def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption(
        "--runxfail",
        action="store_true",
        dest="runxfail",
        default=False,
        help="report the results of xfail tests as if they were not marked",
    )

    parser.addini(
        "xfail_strict",
        "default for the strict parameter of xfail "
        "markers when not given explicitly (default: False)",
        default=False,
        type="bool",
    )
```
### 10 - src/_pytest/terminal.py:

Start line: 1319, End line: 1347

```python
def _folded_skips(
    startpath: Path,
    skipped: Sequence[CollectReport],
) -> List[Tuple[int, str, Optional[int], str]]:
    d: Dict[Tuple[str, Optional[int], str], List[CollectReport]] = {}
    for event in skipped:
        assert event.longrepr is not None
        assert isinstance(event.longrepr, tuple), (event, event.longrepr)
        assert len(event.longrepr) == 3, (event, event.longrepr)
        fspath, lineno, reason = event.longrepr
        # For consistency, report all fspaths in relative form.
        fspath = bestrelpath(startpath, Path(fspath))
        keywords = getattr(event, "keywords", {})
        # Folding reports with global pytestmark variable.
        # This is a workaround, because for now we cannot identify the scope of a skip marker
        # TODO: Revisit after marks scope would be fixed.
        if (
            event.when == "setup"
            and "skip" in keywords
            and "pytestmark" not in keywords
        ):
            key: Tuple[str, Optional[int], str] = (fspath, None, reason)
        else:
            key = (fspath, lineno, reason)
        d.setdefault(key, []).append(event)
    values: List[Tuple[int, str, Optional[int], str]] = []
    for key, events in d.items():
        values.append((len(events), *key))
    return values
```
### 27 - src/_pytest/python.py:

Start line: 164, End line: 175

```python
def async_warn_and_skip(nodeid: str) -> None:
    msg = "async def functions are not natively supported and have been skipped.\n"
    msg += (
        "You need to install a suitable plugin for your async framework, for example:\n"
    )
    msg += "  - anyio\n"
    msg += "  - pytest-asyncio\n"
    msg += "  - pytest-tornasync\n"
    msg += "  - pytest-trio\n"
    msg += "  - pytest-twisted"
    warnings.warn(PytestUnhandledCoroutineWarning(msg.format(nodeid)))
    skip(msg="async def function and no async plugin installed (see warnings)")
```
### 70 - src/_pytest/python.py:

Start line: 1659, End line: 1701

```python
class Function(PyobjMixin, nodes.Item):

    def _prunetraceback(self, excinfo: ExceptionInfo[BaseException]) -> None:
        if hasattr(self, "_obj") and not self.config.getoption("fulltrace", False):
            code = _pytest._code.Code.from_function(get_real_func(self.obj))
            path, firstlineno = code.path, code.firstlineno
            traceback = excinfo.traceback
            ntraceback = traceback.cut(path=path, firstlineno=firstlineno)
            if ntraceback == traceback:
                ntraceback = ntraceback.cut(path=path)
                if ntraceback == traceback:
                    ntraceback = ntraceback.filter(filter_traceback)
                    if not ntraceback:
                        ntraceback = traceback

            excinfo.traceback = ntraceback.filter()
            # issue364: mark all but first and last frames to
            # only show a single-line message for each frame.
            if self.config.getoption("tbstyle", "auto") == "auto":
                if len(excinfo.traceback) > 2:
                    for entry in excinfo.traceback[1:-1]:
                        entry.set_repr_style("short")

    # TODO: Type ignored -- breaks Liskov Substitution.
    def repr_failure(  # type: ignore[override]
        self,
        excinfo: ExceptionInfo[BaseException],
    ) -> Union[str, TerminalRepr]:
        style = self.config.getoption("tbstyle", "auto")
        if style == "auto":
            style = "long"
        return self._repr_failure_py(excinfo, style=style)


class FunctionDefinition(Function):
    """
    This class is a step gap solution until we evolve to have actual function definition nodes
    and manage to get rid of ``metafunc``.
    """

    def runtest(self) -> None:
        raise RuntimeError("function definitions are not supposed to be run as tests")

    setup = runtest
```
### 72 - src/_pytest/python.py:

Start line: 572, End line: 617

```python
class Module(nodes.File, PyCollector):

    def _importtestmodule(self):
        # We assume we are only called once per module.
        importmode = self.config.getoption("--import-mode")
        try:
            mod = import_path(self.path, mode=importmode, root=self.config.rootpath)
        except SyntaxError as e:
            raise self.CollectError(
                ExceptionInfo.from_current().getrepr(style="short")
            ) from e
        except ImportPathMismatchError as e:
            raise self.CollectError(
                "import file mismatch:\n"
                "imported module %r has this __file__ attribute:\n"
                "  %s\n"
                "which is not the same as the test file we want to collect:\n"
                "  %s\n"
                "HINT: remove __pycache__ / .pyc files and/or use a "
                "unique basename for your test file modules" % e.args
            ) from e
        except ImportError as e:
            exc_info = ExceptionInfo.from_current()
            if self.config.getoption("verbose") < 2:
                exc_info.traceback = exc_info.traceback.filter(filter_traceback)
            exc_repr = (
                exc_info.getrepr(style="short")
                if exc_info.traceback
                else exc_info.exconly()
            )
            formatted_tb = str(exc_repr)
            raise self.CollectError(
                "ImportError while importing test module '{path}'.\n"
                "Hint: make sure your test modules/packages have valid Python names.\n"
                "Traceback:\n"
                "{traceback}".format(path=self.path, traceback=formatted_tb)
            ) from e
        except skip.Exception as e:
            if e.allow_module_level:
                raise
            raise self.CollectError(
                "Using pytest.skip outside of a test is not allowed. "
                "To decorate a test function, use the @pytest.mark.skip "
                "or @pytest.mark.skipif decorators instead, and to skip a "
                "module use `pytestmark = pytest.mark.{skip,skipif}."
            ) from e
        self.config.pluginmanager.consider_module(mod)
        return mod
```
### 80 - src/_pytest/python.py:

Start line: 1, End line: 80

```python
"""Python test discovery, setup and run of test functions."""
import enum
import fnmatch
import inspect
import itertools
import os
import sys
import types
import warnings
from collections import Counter
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import saferepr
from _pytest.compat import ascii_escaped
from _pytest.compat import final
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_async_function
from _pytest.compat import is_generator
from _pytest.compat import LEGACY_PATH
from _pytest.compat import legacy_path
from _pytest.compat import NOTSET
from _pytest.compat import REGEX_TYPE
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.compat import STRING_TYPES
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import FSCOLLECTOR_GETHOOKPROXY_ISINITPATH
from _pytest.fixtures import FuncFixtureInfo
from _pytest.main import Session
from _pytest.mark import MARK_GEN
from _pytest.mark import ParameterSet
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import parts
from _pytest.pathlib import visit
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning

if TYPE_CHECKING:
    from typing_extensions import Literal
    from _pytest.fixtures import _Scope
```
