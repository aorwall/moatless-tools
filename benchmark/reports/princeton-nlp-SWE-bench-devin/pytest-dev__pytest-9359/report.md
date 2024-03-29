# pytest-dev__pytest-9359

| **pytest-dev/pytest** | `e2ee3144ed6e241dea8d96215fcdca18b3892551` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 34 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/_code/source.py b/src/_pytest/_code/source.py
--- a/src/_pytest/_code/source.py
+++ b/src/_pytest/_code/source.py
@@ -149,6 +149,11 @@ def get_statement_startend2(lineno: int, node: ast.AST) -> Tuple[int, Optional[i
     values: List[int] = []
     for x in ast.walk(node):
         if isinstance(x, (ast.stmt, ast.ExceptHandler)):
+            # Before Python 3.8, the lineno of a decorated class or function pointed at the decorator.
+            # Since Python 3.8, the lineno points to the class/def, so need to include the decorators.
+            if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
+                for d in x.decorator_list:
+                    values.append(d.lineno - 1)
             values.append(x.lineno - 1)
             for name in ("finalbody", "orelse"):
                 val: Optional[List[ast.stmt]] = getattr(x, name, None)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/_code/source.py | 152 | 152 | - | 34 | -


## Problem Statement

```
Error message prints extra code line when using assert in python3.9
<!--
Thanks for submitting an issue!

Quick check-list while reporting bugs:
-->

- [x] a detailed description of the bug or problem you are having
- [x] output of `pip list` from the virtual environment you are using
- [x] pytest and operating system versions
- [ ] minimal example if possible
### Description
I have a test like this:
\`\`\`
from pytest import fixture


def t(foo):
    return foo


@fixture
def foo():
    return 1


def test_right_statement(foo):
    assert foo == (3 + 2) * (6 + 9)

    @t
    def inner():
        return 2

    assert 2 == inner


@t
def outer():
    return 2
\`\`\`
The test "test_right_statement" fails at the first assertion,but print extra code (the "t" decorator) in error details, like this:

\`\`\`
 ============================= test session starts =============================
platform win32 -- Python 3.9.6, pytest-6.2.5, py-1.10.0, pluggy-0.13.1 -- 
cachedir: .pytest_cache
rootdir: 
plugins: allure-pytest-2.9.45
collecting ... collected 1 item

test_statement.py::test_right_statement FAILED                           [100%]

================================== FAILURES ===================================
____________________________ test_right_statement _____________________________

foo = 1

    def test_right_statement(foo):
>       assert foo == (3 + 2) * (6 + 9)
    
        @t
E       assert 1 == 75
E         +1
E         -75

test_statement.py:14: AssertionError
=========================== short test summary info ===========================
FAILED test_statement.py::test_right_statement - assert 1 == 75
============================== 1 failed in 0.12s ==============================
\`\`\`
And the same thing **did not** happen when using python3.7.10：
\`\`\`
============================= test session starts =============================
platform win32 -- Python 3.7.10, pytest-6.2.5, py-1.11.0, pluggy-1.0.0 -- 
cachedir: .pytest_cache
rootdir: 
collecting ... collected 1 item

test_statement.py::test_right_statement FAILED                           [100%]

================================== FAILURES ===================================
____________________________ test_right_statement _____________________________

foo = 1

    def test_right_statement(foo):
>       assert foo == (3 + 2) * (6 + 9)
E       assert 1 == 75
E         +1
E         -75

test_statement.py:14: AssertionError
=========================== short test summary info ===========================
FAILED test_statement.py::test_right_statement - assert 1 == 75
============================== 1 failed in 0.03s ==============================
\`\`\`
Is there some problems when calculate the statement lineno?

### pip list 
\`\`\`
$ pip list
Package            Version
------------------ -------
atomicwrites       1.4.0
attrs              21.2.0
colorama           0.4.4
importlib-metadata 4.8.2
iniconfig          1.1.1
packaging          21.3
pip                21.3.1
pluggy             1.0.0
py                 1.11.0
pyparsing          3.0.6
pytest             6.2.5
setuptools         59.4.0
toml               0.10.2
typing_extensions  4.0.0
zipp               3.6.0

\`\`\`
### pytest and operating system versions
pytest 6.2.5
Windows 10 
Seems to happen in python 3.9,not 3.7


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/pytest/__init__.py | 1 | 79| 714 | 714 | 1184 | 
| 2 | 2 doc/en/example/xfail_demo.py | 1 | 39| 143 | 857 | 1328 | 
| 3 | 3 testing/example_scripts/issue_519.py | 1 | 33| 362 | 1219 | 1802 | 
| 4 | 4 src/_pytest/_code/code.py | 1 | 56| 357 | 1576 | 11728 | 
| 5 | 5 testing/python/fixtures.py | 2838 | 3843| 6155 | 7731 | 40578 | 
| 6 | 6 testing/python/collect.py | 293 | 345| 400 | 8131 | 51294 | 
| 7 | 6 testing/python/fixtures.py | 994 | 1962| 6100 | 14231 | 51294 | 
| 8 | 6 testing/python/fixtures.py | 3845 | 4475| 4315 | 18546 | 51294 | 
| 9 | 7 doc/en/conf.py | 1 | 115| 838 | 19384 | 54898 | 
| 10 | 8 doc/en/example/assertion/failure_demo.py | 163 | 202| 270 | 19654 | 56547 | 
| 11 | 8 testing/python/collect.py | 210 | 263| 342 | 19996 | 56547 | 
| 12 | 8 testing/python/fixtures.py | 41 | 992| 6172 | 26168 | 56547 | 
| 13 | 8 doc/en/example/assertion/failure_demo.py | 205 | 252| 228 | 26396 | 56547 | 
| 14 | 9 src/_pytest/python.py | 1516 | 1573| 437 | 26833 | 71043 | 
| 15 | 9 testing/python/collect.py | 1 | 43| 300 | 27133 | 71043 | 
| 16 | 9 src/pytest/__init__.py | 82 | 168| 469 | 27602 | 71043 | 
| 17 | 10 src/_pytest/doctest.py | 139 | 173| 292 | 27894 | 76760 | 
| 18 | 11 src/_pytest/setuponly.py | 60 | 98| 323 | 28217 | 77498 | 
| 19 | 11 testing/python/fixtures.py | 1964 | 2836| 5920 | 34137 | 77498 | 
| 20 | 11 doc/en/example/assertion/failure_demo.py | 42 | 120| 680 | 34817 | 77498 | 
| 21 | 11 testing/python/collect.py | 1278 | 1304| 192 | 35009 | 77498 | 
| 22 | 11 doc/en/example/assertion/failure_demo.py | 123 | 160| 143 | 35152 | 77498 | 
| 23 | 11 src/_pytest/python.py | 1 | 87| 624 | 35776 | 77498 | 
| 24 | 12 testing/python/metafunc.py | 1412 | 1436| 214 | 35990 | 92156 | 
| 25 | 12 testing/python/collect.py | 994 | 1033| 364 | 36354 | 92156 | 
| 26 | 13 src/_pytest/pytester.py | 1 | 83| 502 | 36856 | 105913 | 
| 27 | 14 src/_pytest/python_api.py | 774 | 884| 1012 | 37868 | 114320 | 
| 28 | 14 testing/python/collect.py | 347 | 360| 114 | 37982 | 114320 | 
| 29 | 14 src/_pytest/doctest.py | 584 | 669| 786 | 38768 | 114320 | 
| 30 | 14 testing/python/metafunc.py | 1336 | 1368| 211 | 38979 | 114320 | 
| 31 | 15 src/_pytest/fixtures.py | 1 | 114| 760 | 39739 | 128388 | 
| 32 | 15 testing/python/metafunc.py | 1394 | 1410| 110 | 39849 | 128388 | 
| 33 | 15 testing/python/collect.py | 707 | 720| 124 | 39973 | 128388 | 
| 34 | 15 src/_pytest/python.py | 1444 | 1486| 335 | 40308 | 128388 | 
| 35 | 15 src/_pytest/doctest.py | 1 | 63| 425 | 40733 | 128388 | 
| 36 | 15 testing/python/collect.py | 1238 | 1275| 221 | 40954 | 128388 | 
| 37 | 15 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 41117 | 128388 | 
| 38 | 16 src/_pytest/assertion/__init__.py | 1 | 19| 120 | 41237 | 129837 | 
| 39 | 16 testing/python/collect.py | 1035 | 1054| 184 | 41421 | 129837 | 
| 40 | 16 src/_pytest/python.py | 1488 | 1513| 258 | 41679 | 129837 | 
| 41 | 17 testing/python/raises.py | 184 | 213| 260 | 41939 | 132064 | 
| 42 | 18 src/_pytest/faulthandler.py | 1 | 32| 225 | 42164 | 132806 | 
| 43 | 18 testing/python/collect.py | 959 | 991| 300 | 42464 | 132806 | 
| 44 | 19 src/_pytest/nodes.py | 1 | 48| 328 | 42792 | 138233 | 
| 45 | 20 testing/python/integration.py | 336 | 375| 251 | 43043 | 141557 | 
| 46 | 20 testing/python/integration.py | 1 | 43| 302 | 43345 | 141557 | 
| 47 | 20 src/_pytest/doctest.py | 310 | 376| 620 | 43965 | 141557 | 
| 48 | 20 testing/python/collect.py | 654 | 682| 202 | 44167 | 141557 | 
| 49 | 20 testing/python/raises.py | 1 | 52| 341 | 44508 | 141557 | 
| 50 | 21 testing/python/approx.py | 737 | 763| 219 | 44727 | 150661 | 
| 51 | 21 testing/python/collect.py | 1459 | 1494| 270 | 44997 | 150661 | 
| 52 | 21 src/_pytest/pytester.py | 494 | 515| 175 | 45172 | 150661 | 
| 53 | 21 testing/python/metafunc.py | 1 | 29| 151 | 45323 | 150661 | 
| 54 | 22 src/_pytest/unittest.py | 240 | 290| 364 | 45687 | 153647 | 
| 55 | 23 src/_pytest/skipping.py | 46 | 82| 382 | 46069 | 155930 | 
| 56 | 23 testing/python/fixtures.py | 1 | 38| 209 | 46278 | 155930 | 
| 57 | 23 testing/python/raises.py | 215 | 232| 205 | 46483 | 155930 | 
| 58 | 23 doc/en/example/assertion/failure_demo.py | 255 | 282| 161 | 46644 | 155930 | 
| 59 | 24 src/_pytest/runner.py | 202 | 228| 226 | 46870 | 160228 | 
| 60 | 24 src/_pytest/runner.py | 159 | 182| 184 | 47054 | 160228 | 
| 61 | 24 testing/python/collect.py | 639 | 652| 112 | 47166 | 160228 | 
| 62 | 24 testing/python/collect.py | 1208 | 1235| 188 | 47354 | 160228 | 
| 63 | 25 src/_pytest/config/__init__.py | 1502 | 1528| 218 | 47572 | 173230 | 
| 64 | 25 src/_pytest/assertion/__init__.py | 154 | 182| 271 | 47843 | 173230 | 
| 65 | 25 src/_pytest/python_api.py | 1 | 40| 215 | 48058 | 173230 | 
| 66 | 25 testing/python/collect.py | 1403 | 1413| 122 | 48180 | 173230 | 
| 67 | 25 testing/python/collect.py | 1337 | 1360| 207 | 48387 | 173230 | 
| 68 | 25 testing/python/approx.py | 227 | 275| 468 | 48855 | 173230 | 
| 69 | 26 src/_pytest/assertion/rewrite.py | 1 | 54| 339 | 49194 | 183318 | 
| 70 | 26 src/_pytest/python.py | 136 | 168| 330 | 49524 | 183318 | 
| 71 | 26 src/_pytest/doctest.py | 494 | 535| 349 | 49873 | 183318 | 
| 72 | 26 testing/python/collect.py | 1156 | 1176| 191 | 50064 | 183318 | 
| 73 | 26 src/_pytest/fixtures.py | 901 | 925| 231 | 50295 | 183318 | 
| 74 | 26 testing/python/collect.py | 72 | 90| 208 | 50503 | 183318 | 
| 75 | 27 extra/get_issues.py | 55 | 86| 231 | 50734 | 183872 | 
| 76 | 27 src/_pytest/fixtures.py | 1223 | 1238| 114 | 50848 | 183872 | 
| 77 | 27 src/_pytest/pytester.py | 479 | 491| 138 | 50986 | 183872 | 
| 78 | 28 src/_pytest/outcomes.py | 74 | 109| 261 | 51247 | 186180 | 
| 79 | 28 testing/python/collect.py | 1416 | 1456| 356 | 51603 | 186180 | 
| 80 | 28 testing/python/collect.py | 1178 | 1205| 215 | 51818 | 186180 | 
| 81 | 28 testing/python/metafunc.py | 1125 | 1147| 179 | 51997 | 186180 | 
| 82 | 28 src/_pytest/pytester.py | 651 | 737| 717 | 52714 | 186180 | 
| 83 | 28 src/_pytest/unittest.py | 207 | 238| 273 | 52987 | 186180 | 
| 84 | 28 doc/en/conf.py | 116 | 221| 855 | 53842 | 186180 | 
| 85 | 28 testing/python/integration.py | 45 | 79| 252 | 54094 | 186180 | 
| 86 | 29 src/_pytest/debugging.py | 367 | 389| 205 | 54299 | 189167 | 
| 87 | 29 src/_pytest/fixtures.py | 489 | 578| 796 | 55095 | 189167 | 
| 88 | 29 testing/python/metafunc.py | 1238 | 1267| 211 | 55306 | 189167 | 
| 89 | 30 src/_pytest/_io/saferepr.py | 1 | 35| 262 | 55568 | 190270 | 
| 90 | 30 src/_pytest/_code/code.py | 680 | 726| 354 | 55922 | 190270 | 
| 91 | 30 testing/python/metafunc.py | 972 | 991| 128 | 56050 | 190270 | 
| 92 | 30 doc/en/conf.py | 424 | 472| 411 | 56461 | 190270 | 
| 93 | 30 testing/python/collect.py | 609 | 637| 214 | 56675 | 190270 | 
| 94 | 30 testing/python/raises.py | 130 | 157| 219 | 56894 | 190270 | 
| 95 | 30 src/_pytest/pytester.py | 591 | 618| 204 | 57098 | 190270 | 
| 96 | 30 src/_pytest/_code/code.py | 960 | 974| 124 | 57222 | 190270 | 
| 97 | 30 extra/get_issues.py | 33 | 52| 152 | 57374 | 190270 | 
| 98 | 30 testing/python/approx.py | 57 | 92| 303 | 57677 | 190270 | 
| 99 | 31 src/_pytest/reports.py | 1 | 56| 380 | 58057 | 194682 | 
| 100 | 31 src/_pytest/debugging.py | 150 | 235| 650 | 58707 | 194682 | 
| 101 | 31 testing/python/metafunc.py | 1049 | 1075| 208 | 58915 | 194682 | 
| 102 | 31 src/_pytest/_code/code.py | 1026 | 1050| 202 | 59117 | 194682 | 
| 103 | 31 testing/python/approx.py | 154 | 201| 534 | 59651 | 194682 | 
| 104 | 32 src/_pytest/assertion/util.py | 429 | 473| 360 | 60011 | 198714 | 
| 105 | 32 src/_pytest/pytester.py | 164 | 186| 243 | 60254 | 198714 | 
| 106 | 32 testing/python/approx.py | 95 | 152| 572 | 60826 | 198714 | 
| 107 | 32 testing/python/metafunc.py | 994 | 1023| 227 | 61053 | 198714 | 
| 108 | 32 testing/python/approx.py | 1 | 54| 269 | 61322 | 198714 | 
| 109 | 32 src/_pytest/unittest.py | 1 | 41| 254 | 61576 | 198714 | 
| 110 | 32 src/_pytest/faulthandler.py | 48 | 64| 185 | 61761 | 198714 | 
| 111 | 32 src/_pytest/python.py | 337 | 353| 142 | 61903 | 198714 | 
| 112 | 32 testing/python/collect.py | 575 | 592| 184 | 62087 | 198714 | 
| 113 | 32 src/_pytest/python_api.py | 885 | 919| 370 | 62457 | 198714 | 
| 114 | 33 doc/en/example/assertion/global_testmodule_config/conftest.py | 1 | 15| 0 | 62457 | 198793 | 
| 115 | 33 src/_pytest/assertion/rewrite.py | 288 | 304| 229 | 62686 | 198793 | 
| 116 | 33 src/_pytest/fixtures.py | 1241 | 1256| 116 | 62802 | 198793 | 
| 117 | 33 src/_pytest/assertion/rewrite.py | 475 | 507| 223 | 63025 | 198793 | 


## Patch

```diff
diff --git a/src/_pytest/_code/source.py b/src/_pytest/_code/source.py
--- a/src/_pytest/_code/source.py
+++ b/src/_pytest/_code/source.py
@@ -149,6 +149,11 @@ def get_statement_startend2(lineno: int, node: ast.AST) -> Tuple[int, Optional[i
     values: List[int] = []
     for x in ast.walk(node):
         if isinstance(x, (ast.stmt, ast.ExceptHandler)):
+            # Before Python 3.8, the lineno of a decorated class or function pointed at the decorator.
+            # Since Python 3.8, the lineno points to the class/def, so need to include the decorators.
+            if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
+                for d in x.decorator_list:
+                    values.append(d.lineno - 1)
             values.append(x.lineno - 1)
             for name in ("finalbody", "orelse"):
                 val: Optional[List[ast.stmt]] = getattr(x, name, None)

```

## Test Patch

```diff
diff --git a/testing/code/test_source.py b/testing/code/test_source.py
--- a/testing/code/test_source.py
+++ b/testing/code/test_source.py
@@ -618,6 +618,19 @@ def something():
     assert str(source) == "def func(): raise ValueError(42)"
 
 
+def test_decorator() -> None:
+    s = """\
+def foo(f):
+    pass
+
+@foo
+def bar():
+    pass
+    """
+    source = getstatement(3, s)
+    assert "@foo" in str(source)
+
+
 def XXX_test_expression_multiline() -> None:
     source = """\
 something

```


## Code snippets

### 1 - src/pytest/__init__.py:

Start line: 1, End line: 79

```python
# PYTHON_ARGCOMPLETE_OK
"""pytest: unit and functional testing with Python."""
from . import collect
from _pytest import __version__
from _pytest import version_tuple
from _pytest._code import ExceptionInfo
from _pytest.assertion import register_assert_rewrite
from _pytest.cacheprovider import Cache
from _pytest.capture import CaptureFixture
from _pytest.config import cmdline
from _pytest.config import Config
from _pytest.config import console_main
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config import UsageError
from _pytest.config.argparsing import OptionGroup
from _pytest.config.argparsing import Parser
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.logging import LogCaptureFixture
from _pytest.main import Session
from _pytest.mark import Mark
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import MarkDecorator
from _pytest.mark import MarkGenerator
from _pytest.mark import param
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.pytester import HookRecorder
from _pytest.pytester import LineMatcher
from _pytest.pytester import Pytester
from _pytest.pytester import RecordedHookCall
from _pytest.pytester import RunResult
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Metafunc
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import WarningsRecorder
from _pytest.recwarn import warns
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.runner import CallInfo
from _pytest.stash import Stash
from _pytest.stash import StashKey
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestRemovedIn7Warning
from _pytest.warning_types import PytestRemovedIn8Warning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnhandledThreadExceptionWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestUnraisableExceptionWarning
from _pytest.warning_types import PytestWarning

set_trace = __pytestPDB.set_trace
```
### 2 - doc/en/example/xfail_demo.py:

Start line: 1, End line: 39

```python
import pytest

xfail = pytest.mark.xfail


@xfail
def test_hello():
    assert 0


@xfail(run=False)
def test_hello2():
    assert 0


@xfail("hasattr(os, 'sep')")
def test_hello3():
    assert 0


@xfail(reason="bug 110")
def test_hello4():
    assert 0


@xfail('pytest.__version__[0] != "17"')
def test_hello5():
    assert 0


def test_hello6():
    pytest.xfail("reason")


@xfail(raises=IndexError)
def test_hello7():
    x = []
    x[1] = 1
```
### 3 - testing/example_scripts/issue_519.py:

Start line: 1, End line: 33

```python
import pprint
from typing import List
from typing import Tuple

import pytest


def pytest_generate_tests(metafunc):
    if "arg1" in metafunc.fixturenames:
        metafunc.parametrize("arg1", ["arg1v1", "arg1v2"], scope="module")

    if "arg2" in metafunc.fixturenames:
        metafunc.parametrize("arg2", ["arg2v1", "arg2v2"], scope="function")


@pytest.fixture(scope="session")
def checked_order():
    order: List[Tuple[str, str, str]] = []

    yield order
    pprint.pprint(order)
    assert order == [
        ("issue_519.py", "fix1", "arg1v1"),
        ("test_one[arg1v1-arg2v1]", "fix2", "arg2v1"),
        ("test_two[arg1v1-arg2v1]", "fix2", "arg2v1"),
        ("test_one[arg1v1-arg2v2]", "fix2", "arg2v2"),
        ("test_two[arg1v1-arg2v2]", "fix2", "arg2v2"),
        ("issue_519.py", "fix1", "arg1v2"),
        ("test_one[arg1v2-arg2v1]", "fix2", "arg2v1"),
        ("test_two[arg1v2-arg2v1]", "fix2", "arg2v1"),
        ("test_one[arg1v2-arg2v2]", "fix2", "arg2v2"),
        ("test_two[arg1v2-arg2v2]", "fix2", "arg2v2"),
    ]
```
### 4 - src/_pytest/_code/code.py:

Start line: 1, End line: 56

```python
import ast
import inspect
import re
import sys
import traceback
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
from pathlib import Path
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from weakref import ref

import attr
import pluggy

import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import final
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath

if TYPE_CHECKING:
    from typing_extensions import Literal
    from typing_extensions import SupportsIndex
    from weakref import ReferenceType

    _TracebackStyle = Literal["long", "short", "line", "no", "native", "value", "auto"]
```
### 5 - testing/python/fixtures.py:

Start line: 2838, End line: 3843

```python
class TestFixtureMarker:

    def test_parametrized_fixture_teardown_order(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(params=[1,2], scope="class")
            def param1(request):
                return request.param

            values = []

            class TestClass(object):
                @classmethod
                @pytest.fixture(scope="class", autouse=True)
                def setup1(self, request, param1):
                    values.append(1)
                    request.addfinalizer(self.teardown1)
                @classmethod
                def teardown1(self):
                    assert values.pop() == 1
                @pytest.fixture(scope="class", autouse=True)
                def setup2(self, request, param1):
                    values.append(2)
                    request.addfinalizer(self.teardown2)
                @classmethod
                def teardown2(self):
                    assert values.pop() == 2
                def test(self):
                    pass

            def test_finish():
                assert not values
        """
        )
        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines(
            """
            *3 passed*
        """
        )
        result.stdout.no_fnmatch_line("*error*")

    def test_fixture_finalizer(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            import sys

            @pytest.fixture
            def browser(request):

                def finalize():
                    sys.stdout.write_text('Finalized')
                request.addfinalizer(finalize)
                return {}
        """
        )
        b = pytester.mkdir("subdir")
        b.joinpath("test_overridden_fixture_finalizer.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def browser(browser):
                    browser['visited'] = True
                    return browser

                def test_browser(browser):
                    assert browser['visited'] is True
                """
            )
        )
        reprec = pytester.runpytest("-s")
        for test in ["test_browser"]:
            reprec.stdout.fnmatch_lines(["*Finalized*"])

    def test_class_scope_with_normal_tests(self, pytester: Pytester) -> None:
        testpath = pytester.makepyfile(
            """
            import pytest

            class Box(object):
                value = 0

            @pytest.fixture(scope='class')
            def a(request):
                Box.value += 1
                return Box.value

            def test_a(a):
                assert a == 1

            class Test1(object):
                def test_b(self, a):
                    assert a == 2

            class Test2(object):
                def test_c(self, a):
                    assert a == 3"""
        )
        reprec = pytester.inline_run(testpath)
        for test in ["test_a", "test_b", "test_c"]:
            assert reprec.matchreport(test).passed

    def test_request_is_clean(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1, 2])
            def fix(request):
                request.addfinalizer(lambda: values.append(request.param))
            def test_fix(fix):
                pass
        """
        )
        reprec = pytester.inline_run("-s")
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2]

    def test_parametrize_separated_lifecycle(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            values = []
            @pytest.fixture(scope="module", params=[1, 2])
            def arg(request):
                x = request.param
                request.addfinalizer(lambda: values.append("fin%s" % x))
                return request.param
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
        """
        )
        reprec = pytester.inline_run("-vs")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        import pprint

        pprint.pprint(values)
        # assert len(values) == 6
        assert values[0] == values[1] == 1
        assert values[2] == "fin1"
        assert values[3] == values[4] == 2
        assert values[5] == "fin2"

    def test_parametrize_function_scoped_finalizers_called(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="function", params=[1, 2])
            def arg(request):
                x = request.param
                request.addfinalizer(lambda: values.append("fin%s" % x))
                return request.param

            values = []
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
            def test_3():
                assert len(values) == 8
                assert values == [1, "fin1", 2, "fin2", 1, "fin1", 2, "fin2"]
        """
        )
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=5)

    @pytest.mark.parametrize("scope", ["session", "function", "module"])
    def test_finalizer_order_on_parametrization(
        self, scope, pytester: Pytester
    ) -> None:
        """#246"""
        pytester.makepyfile(
            """
            import pytest
            values = []

            @pytest.fixture(scope=%(scope)r, params=["1"])
            def fix1(request):
                return request.param

            @pytest.fixture(scope=%(scope)r)
            def fix2(request, base):
                def cleanup_fix2():
                    assert not values, "base should not have been finalized"
                request.addfinalizer(cleanup_fix2)

            @pytest.fixture(scope=%(scope)r)
            def base(request, fix1):
                def cleanup_base():
                    values.append("fin_base")
                    print("finalizing base")
                request.addfinalizer(cleanup_base)

            def test_begin():
                pass
            def test_baz(base, fix2):
                pass
            def test_other():
                pass
        """
            % {"scope": scope}
        )
        reprec = pytester.inline_run("-lvs")
        reprec.assertoutcome(passed=3)

    def test_class_scope_parametrization_ordering(self, pytester: Pytester) -> None:
        """#396"""
        pytester.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=["John", "Doe"], scope="class")
            def human(request):
                request.addfinalizer(lambda: values.append("fin %s" % request.param))
                return request.param

            class TestGreetings(object):
                def test_hello(self, human):
                    values.append("test_hello")

            class TestMetrics(object):
                def test_name(self, human):
                    values.append("test_name")

                def test_population(self, human):
                    values.append("test_population")
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=6)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [
            "test_hello",
            "fin John",
            "test_hello",
            "fin Doe",
            "test_name",
            "test_population",
            "fin John",
            "test_name",
            "test_population",
            "fin Doe",
        ]

    def test_parametrize_setup_function(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module", params=[1, 2])
            def arg(request):
                return request.param

            @pytest.fixture(scope="module", autouse=True)
            def mysetup(request, arg):
                request.addfinalizer(lambda: values.append("fin%s" % arg))
                values.append("setup%s" % arg)

            values = []
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
            def test_3():
                import pprint
                pprint.pprint(values)
                if arg == 1:
                    assert values == ["setup1", 1, 1, ]
                elif arg == 2:
                    assert values == ["setup1", 1, 1, "fin1",
                                 "setup2", 2, 2, ]

        """
        )
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=6)

    def test_fixture_marked_function_not_collected_as_test(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
            def test_app():
                return 1

            def test_something(test_app):
                assert test_app == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_params_and_ids(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[object(), object()],
                            ids=['alpha', 'beta'])
            def fix(request):
                return request.param

            def test_foo(fix):
                assert 1
        """
        )
        res = pytester.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_params_and_ids_yieldfixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[object(), object()], ids=['alpha', 'beta'])
            def fix(request):
                 yield request.param

            def test_foo(fix):
                assert 1
        """
        )
        res = pytester.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_deterministic_fixture_collection(
        self, pytester: Pytester, monkeypatch
    ) -> None:
        """#920"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module",
                            params=["A",
                                    "B",
                                    "C"])
            def A(request):
                return request.param

            @pytest.fixture(scope="module",
                            params=["DDDDDDDDD", "EEEEEEEEEEEE", "FFFFFFFFFFF", "banansda"])
            def B(request, A):
                return request.param

            def test_foo(B):
                # Something funky is going on here.
                # Despite specified seeds, on what is collected,
                # sometimes we get unexpected passes. hashing B seems
                # to help?
                assert hash(B) or True
            """
        )
        monkeypatch.setenv("PYTHONHASHSEED", "1")
        out1 = pytester.runpytest_subprocess("-v")
        monkeypatch.setenv("PYTHONHASHSEED", "2")
        out2 = pytester.runpytest_subprocess("-v")
        output1 = [
            line
            for line in out1.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        output2 = [
            line
            for line in out2.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        assert len(output1) == 12
        assert output1 == output2


class TestRequestScopeAccess:
    pytestmark = pytest.mark.parametrize(
        ("scope", "ok", "error"),
        [
            ["session", "", "path class function module"],
            ["module", "module path", "cls function"],
            ["class", "module path cls", "function"],
            ["function", "module path cls function", ""],
        ],
    )

    def test_setup(self, pytester: Pytester, scope, ok, error) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope=%r, autouse=True)
            def myscoped(request):
                for x in %r:
                    assert hasattr(request, x)
                for x in %r:
                    pytest.raises(AttributeError, lambda:
                        getattr(request, x))
                assert request.session
                assert request.config
            def test_func():
                pass
        """
            % (scope, ok.split(), error.split())
        )
        reprec = pytester.inline_run("-l")
        reprec.assertoutcome(passed=1)

    def test_funcarg(self, pytester: Pytester, scope, ok, error) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope=%r)
            def arg(request):
                for x in %r:
                    assert hasattr(request, x)
                for x in %r:
                    pytest.raises(AttributeError, lambda:
                        getattr(request, x))
                assert request.session
                assert request.config
            def test_func(arg):
                pass
        """
            % (scope, ok.split(), error.split())
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)


class TestErrors:
    def test_subfactory_missing_funcarg(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def gen(qwe123):
                return 1
            def test_something(gen):
                pass
        """
        )
        result = pytester.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )

    def test_issue498_fixture_finalizer_failing(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
            def fix1(request):
                def f():
                    raise KeyError
                request.addfinalizer(f)
                return object()

            values = []
            def test_1(fix1):
                values.append(fix1)
            def test_2(fix1):
                values.append(fix1)
            def test_3():
                assert values[0] != values[1]
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            """
            *ERROR*teardown*test_1*
            *KeyError*
            *ERROR*teardown*test_2*
            *KeyError*
            *3 pass*2 errors*
        """
        )

    def test_setupfunc_missing_funcarg(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def gen(qwe123):
                return 1
            def test_something():
                pass
        """
        )
        result = pytester.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )


class TestShowFixtures:
    def test_funcarg_compat(self, pytester: Pytester) -> None:
        config = pytester.parseconfigure("--funcargs")
        assert config.option.showfixtures

    def test_show_fixtures(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- .../_pytest/tmpdir.py:*",
                "*for the test session*",
                "tmp_path -- .../_pytest/tmpdir.py:*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_verbose(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures", "-v")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- .../_pytest/tmpdir.py:*",
                "*for the test session*",
                "tmp_path -- .../_pytest/tmpdir.py:*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_testmodule(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def _arg0():
                """ hidden """
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            """
            *tmp_path -- *
            *fixtures defined from*
            *arg1 -- test_show_fixtures_testmodule.py:6*
            *hello world*
        """
        )
        result.stdout.no_fnmatch_line("*arg0*")

    @pytest.mark.parametrize("testmod", [True, False])
    def test_show_fixtures_conftest(self, pytester: Pytester, testmod) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        if testmod:
            pytester.makepyfile(
                """
                def test_hello():
                    pass
            """
            )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            *tmp_path*
            *fixtures defined from*conftest*
            *arg1*
            *hello world*
        """
        )

    def test_show_fixtures_trimmed_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def arg1():
                    """
                    line1
                    line2

                    """
                @pytest.fixture
                def arg2():
                    """
                    line1
                    line2

                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_trimmed_doc *
                arg2 -- test_show_fixtures_trimmed_doc.py:10
                    line1
                    line2
                arg1 -- test_show_fixtures_trimmed_doc.py:3
                    line1
                    line2
                """
            )
        )

    def test_show_fixtures_indented_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """
                    line1
                        indented line
                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc *
                fixture1 -- test_show_fixtures_indented_doc.py:3
                    line1
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_doc_first_line_unindented(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """line1
                    line2
                        indented line
                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc_first_line_unindented *
                fixture1 -- test_show_fixtures_indented_doc_first_line_unindented.py:3
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_in_class(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                class TestClass(object):
                    @pytest.fixture
                    def fixture1(self):
                        """line1
                        line2
                            indented line
                        """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_in_class *
                fixture1 -- test_show_fixtures_indented_in_class.py:4
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_different_files(self, pytester: Pytester) -> None:
        """`--fixtures` only shows fixtures from first file (#833)."""
        pytester.makepyfile(
            test_a='''
            import pytest

            @pytest.fixture
            def fix_a():
                """Fixture A"""
                pass

            def test_a(fix_a):
                pass
        '''
        )
        pytester.makepyfile(
            test_b='''
            import pytest

            @pytest.fixture
            def fix_b():
                """Fixture B"""
                pass

            def test_b(fix_b):
                pass
        '''
        )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from test_a *
            fix_a -- test_a.py:4
                Fixture A

            * fixtures defined from test_b *
            fix_b -- test_b.py:4
                Fixture B
        """
        )

    def test_show_fixtures_with_same_name(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hello World in conftest.py"""
                return "Hello World"
        '''
        )
        pytester.makepyfile(
            """
            def test_foo(arg1):
                assert arg1 == "Hello World"
        """
        )
        pytester.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hi from test module"""
                return "Hi"
            def test_bar(arg1):
                assert arg1 == "Hi"
        '''
        )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from conftest *
            arg1 -- conftest.py:3
                Hello World in conftest.py

            * fixtures defined from test_show_fixtures_with_same_name *
            arg1 -- test_show_fixtures_with_same_name.py:3
                Hi from test module
        """
        )

    def test_fixture_disallow_twice(self):
        """Test that applying @pytest.fixture twice generates an error (#2334)."""
        with pytest.raises(ValueError):

            @pytest.fixture
            @pytest.fixture
            def foo():
                raise NotImplementedError()


class TestContextManagerFixtureFuncs:
    def test_simple(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
            def arg1():
                print("setup")
                yield 1
                print("teardown")
            def test_1(arg1):
                print("test1", arg1)
            def test_2(arg1):
                print("test2", arg1)
                assert 0
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *setup*
            *test1 1*
            *teardown*
            *setup*
            *test2 1*
            *teardown*
        """
        )

    def test_scoped(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                print("setup")
                yield 1
                print("teardown")
            def test_1(arg1):
                print("test1", arg1)
            def test_2(arg1):
                print("test2", arg1)
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *setup*
            *test1 1*
            *test2 1*
            *teardown*
        """
        )

    def test_setup_exception(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                pytest.fail("setup")
                yield 1
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*setup*
            *1 error*
        """
        )

    def test_teardown_exception(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                yield 1
                pytest.fail("teardown")
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*teardown*
            *1 passed*1 error*
        """
        )

    def test_yields_more_than_one(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                yield 1
                yield 2
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *fixture function*
            *test_yields*:2*
        """
        )

    def test_custom_name(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(name='meow')
            def arg1():
                return 'mew'
            def test_1(meow):
                print(meow)
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(["*mew*"])


class TestParameterizedSubRequest:
    def test_call_from_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_call_from_fixture="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            @pytest.fixture
            def get_named_fixture(request):
                return request.getfixturevalue('fix_with_param')

            def test_foo(request, get_named_fixture):
                pass
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_fixture.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_fixture.py:4",
                "Requested here:",
                "test_call_from_fixture.py:9",
                "*1 error in*",
            ]
        )

    def test_call_from_test(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_call_from_test="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_test.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_test.py:4",
                "Requested here:",
                "test_call_from_test.py:8",
                "*1 failed*",
            ]
        )

    def test_external_fixture(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param
            """
        )

        pytester.makepyfile(
            test_external_fixture="""
            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_external_fixture.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "conftest.py:4",
                "Requested here:",
                "test_external_fixture.py:2",
                "*1 failed*",
            ]
        )
```
### 6 - testing/python/collect.py:

Start line: 293, End line: 345

```python
class TestFunction:

    @staticmethod
    def make_function(pytester: Pytester, **kwargs: Any) -> Any:
        from _pytest.fixtures import FixtureManager

        config = pytester.parseconfigure()
        session = Session.from_config(config)
        session._fixturemanager = FixtureManager(session)

        return pytest.Function.from_parent(parent=session, **kwargs)

    def test_function_equality(self, pytester: Pytester) -> None:
        def func1():
            pass

        def func2():
            pass

        f1 = self.make_function(pytester, name="name", callobj=func1)
        assert f1 == f1
        f2 = self.make_function(
            pytester, name="name", callobj=func2, originalname="foobar"
        )
        assert f1 != f2

    def test_repr_produces_actual_test_id(self, pytester: Pytester) -> None:
        f = self.make_function(
            pytester, name=r"test[\xe5]", callobj=self.test_repr_produces_actual_test_id
        )
        assert repr(f) == r"<Function test[\xe5]>"

    def test_issue197_parametrize_emptyset(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize('arg', [])
            def test_function(arg):
                pass
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(skipped=1)

    def test_single_tuple_unwraps_values(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize(('arg',), [(1,)])
            def test_function(arg):
                assert arg == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)
```
### 7 - testing/python/fixtures.py:

Start line: 994, End line: 1962

```python
class TestRequestBasic:

    def test_request_fixturenames_dynamic_fixture(self, pytester: Pytester) -> None:
        """Regression test for #3057"""
        pytester.copy_example("fixtures/test_getfixturevalue_dynamic.py")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_setupdecorator_and_xunit(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope='module', autouse=True)
            def setup_module():
                values.append("module")
            @pytest.fixture(autouse=True)
            def setup_function():
                values.append("function")

            def test_func():
                pass

            class TestClass(object):
                @pytest.fixture(scope="class", autouse=True)
                def setup_class(self):
                    values.append("class")
                @pytest.fixture(autouse=True)
                def setup_method(self):
                    values.append("method")
                def test_method(self):
                    pass
            def test_all():
                assert values == ["module", "function", "class",
                             "function", "method", "function"]
        """
        )
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=3)

    def test_fixtures_sub_subdir_normalize_sep(self, pytester: Pytester) -> None:
        # this tests that normalization of nodeids takes place
        b = pytester.path.joinpath("tests", "unit")
        b.mkdir(parents=True)
        b.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def arg1():
                    pass
                """
            )
        )
        p = b.joinpath("test_module.py")
        p.write_text("def test_func(arg1): pass")
        result = pytester.runpytest(p, "--fixtures")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            """
            *fixtures defined*conftest*
            *arg1*
        """
        )

    def test_show_fixtures_color_yes(self, pytester: Pytester) -> None:
        pytester.makepyfile("def test_this(): assert 1")
        result = pytester.runpytest("--color=yes", "--fixtures")
        assert "\x1b[32mtmp_path" in result.stdout.str()

    def test_newstyle_with_request(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg(request):
                pass
            def test_1(arg):
                pass
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_setupcontext_no_param(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(params=[1,2])
            def arg(request):
                return request.param

            @pytest.fixture(autouse=True)
            def mysetup(request, arg):
                assert not hasattr(request, "param")
            def test_1(arg):
                assert arg in (1,2)
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)


class TestRequestSessionScoped:
    @pytest.fixture(scope="session")
    def session_request(self, request):
        return request

    @pytest.mark.parametrize("name", ["path", "module"])
    def test_session_scoped_unavailable_attributes(self, session_request, name):
        with pytest.raises(
            AttributeError,
            match=f"{name} not available in session-scoped context",
        ):
            getattr(session_request, name)


class TestRequestMarking:
    def test_applymarker(self, pytester: Pytester) -> None:
        item1, item2 = pytester.getitems(
            """
            import pytest

            @pytest.fixture
            def something(request):
                pass
            class TestClass(object):
                def test_func1(self, something):
                    pass
                def test_func2(self, something):
                    pass
        """
        )
        req1 = fixtures.FixtureRequest(item1, _ispytest=True)
        assert "xfail" not in item1.keywords
        req1.applymarker(pytest.mark.xfail)
        assert "xfail" in item1.keywords
        assert "skipif" not in item1.keywords
        req1.applymarker(pytest.mark.skipif)
        assert "skipif" in item1.keywords
        with pytest.raises(ValueError):
            req1.applymarker(42)  # type: ignore[arg-type]

    def test_accesskeywords(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords
            @pytest.mark.XYZ
            def test_function(keywords):
                assert keywords["XYZ"]
                assert "abc" not in keywords
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_accessmarker_dynamic(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords

            @pytest.fixture(scope="class", autouse=True)
            def marking(request):
                request.applymarker(pytest.mark.XYZ("hello"))
        """
        )
        pytester.makepyfile(
            """
            import pytest
            def test_fun1(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
            def test_fun2(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)


class TestFixtureUsages:
    def test_noargfixturedec(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
            def arg1():
                return 1

            def test_func(arg1):
                assert arg1 == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_receives_funcargs(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg1():
                return 1

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
            def test_all(arg1, arg2):
                assert arg1 == 1
                assert arg2 == 2
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_receives_funcargs_scope_mismatch(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*involved factories*",
                "test_receives_funcargs_scope_mismatch.py:6:  def arg2(arg1)",
                "test_receives_funcargs_scope_mismatch.py:2:  def arg1()",
                "*1 error*",
            ]
        )

    def test_receives_funcargs_scope_mismatch_issue660(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            ["*ScopeMismatch*involved factories*", "* def arg2*", "*1 error*"]
        )

    def test_invalid_scope(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="functions")
            def badscope():
                pass

            def test_nothing(badscope):
                pass
        """
        )
        result = pytester.runpytest_inprocess()
        result.stdout.fnmatch_lines(
            "*Fixture 'badscope' from test_invalid_scope.py got an unexpected scope value 'functions'"
        )

    @pytest.mark.parametrize("scope", ["function", "session"])
    def test_parameters_without_eq_semantics(self, scope, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            class NoEq1:  # fails on `a == b` statement
                def __eq__(self, _):
                    raise RuntimeError

            class NoEq2:  # fails on `if a == b:` statement
                def __eq__(self, _):
                    class NoBool:
                        def __bool__(self):
                            raise RuntimeError
                    return NoBool()

            import pytest
            @pytest.fixture(params=[NoEq1(), NoEq2()], scope={scope!r})
            def no_eq(request):
                return request.param

            def test1(no_eq):
                pass

            def test2(no_eq):
                pass
        """.format(
                scope=scope
            )
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*4 passed*"])

    def test_funcarg_parametrized_and_used_twice(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1,2])
            def arg1(request):
                values.append(1)
                return request.param

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == arg1 + 1
                assert len(values) == arg1
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*2 passed*"])

    def test_factory_uses_unknown_funcarg_as_dependency_error(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture()
            def fail(missing):
                return

            @pytest.fixture()
            def call_fail(fail):
                return

            def test_missing(call_fail):
                pass
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            """
            *pytest.fixture()*
            *def call_fail(fail)*
            *pytest.fixture()*
            *def fail*
            *fixture*'missing'*not found*
        """
        )

    def test_factory_setup_as_classes_fails(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            class arg1(object):
                def __init__(self, request):
                    self.x = 1
            arg1 = pytest.fixture()(arg1)

        """
        )
        reprec = pytester.inline_run()
        values = reprec.getfailedcollections()
        assert len(values) == 1

    def test_usefixtures_marker(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            values = []

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"
                values.append(1)

            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                    assert len(values) == 1
                def test_two(self):
                    assert self.hello == "world"
                    assert len(values) == 1
            pytest.mark.usefixtures("myfix")(TestClass)
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_ini(self, pytester: Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            usefixtures = myfix
        """
        )
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"

        """
        )
        pytester.makepyfile(
            """
            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                def test_two(self):
                    assert self.hello == "world"
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_seen_in_showmarkers(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(
            """
            *usefixtures(fixturename1*mark tests*fixtures*
        """
        )

    def test_request_instance_issue203(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            class TestClass(object):
                @pytest.fixture
                def setup1(self, request):
                    assert self == request.instance
                    self.arg1 = 1
                def test_hello(self, setup1):
                    assert self.arg1 == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_fixture_parametrized_with_iterator(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            values = []
            def f():
                yield 1
                yield 2
            dec = pytest.fixture(scope="module", params=f())

            @dec
            def arg(request):
                return request.param
            @dec
            def arg2(request):
                return request.param

            def test_1(arg):
                values.append(arg)
            def test_2(arg2):
                values.append(arg2*10)
        """
        )
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2, 10, 20]

    def test_setup_functions_as_fixtures(self, pytester: Pytester) -> None:
        """Ensure setup_* methods obey fixture scope rules (#517, #3094)."""
        pytester.makepyfile(
            """
            import pytest

            DB_INITIALIZED = None

            @pytest.fixture(scope="session", autouse=True)
            def db():
                global DB_INITIALIZED
                DB_INITIALIZED = True
                yield
                DB_INITIALIZED = False

            def setup_module():
                assert DB_INITIALIZED

            def teardown_module():
                assert DB_INITIALIZED

            class TestClass(object):

                def setup_method(self, method):
                    assert DB_INITIALIZED

                def teardown_method(self, method):
                    assert DB_INITIALIZED

                def test_printer_1(self):
                    pass

                def test_printer_2(self):
                    pass
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed in *"])


class TestFixtureManagerParseFactories:
    @pytest.fixture
    def pytester(self, pytester: Pytester) -> Pytester:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                return "conftest"

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return pytester

    def test_parsefactories_evil_objects_issue214(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            class A(object):
                def __call__(self):
                    pass
                def __getattr__(self, name):
                    raise RuntimeError()
            a = A()
            def test_hello():
                pass
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1, failed=0)

    def test_parsefactories_conftest(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test_hello(item, fm):
                for name in ("fm", "hello", "item"):
                    faclist = fm.getfixturedefs(name, item.nodeid)
                    assert len(faclist) == 1
                    fac = faclist[0]
                    assert fac.func.__name__ == name
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_conftest_and_module_and_class(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """\
            import pytest

            @pytest.fixture
            def hello(request):
                return "module"
            class TestClass(object):
                @pytest.fixture
                def hello(self, request):
                    return "class"
                def test_hello(self, item, fm):
                    faclist = fm.getfixturedefs("hello", item.nodeid)
                    print(faclist)
                    assert len(faclist) == 3

                    assert faclist[0].func(item._request) == "conftest"
                    assert faclist[1].func(item._request) == "module"
                    assert faclist[2].func(item._request) == "class"
            """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_relative_node_ids(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        # example mostly taken from:
        # https://mail.python.org/pipermail/pytest-dev/2014-September/002617.html
        runner = pytester.mkdir("runner")
        package = pytester.mkdir("package")
        package.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            @pytest.fixture
            def one():
                return 1
            """
            )
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 1
                """
            )
        )
        sub = package.joinpath("sub")
        sub.mkdir()
        sub.joinpath("__init__.py").touch()
        sub.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def one():
                    return 2
                """
            )
        )
        sub.joinpath("test_y.py").write_text(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 2
                """
            )
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)
        with monkeypatch.context() as mp:
            mp.chdir(runner)
            reprec = pytester.inline_run("..")
            reprec.assertoutcome(passed=2)

    def test_package_xunit_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            __init__="""\
            values = []
        """
        )
        package = pytester.mkdir("package")
        package.joinpath("__init__.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package"]
                """
            )
        )
        package = pytester.mkdir("package2")
        package.joinpath("__init__.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package2")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package2"]
                """
            )
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_package_fixture_complex(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            __init__="""\
            values = []
        """
        )
        pytester.syspathinsert(pytester.path.name)
        package = pytester.mkdir("package")
        package.joinpath("__init__.py").write_text("")
        package.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                from .. import values
                @pytest.fixture(scope="package")
                def one():
                    values.append("package")
                    yield values
                    values.pop()
                @pytest.fixture(scope="package", autouse=True)
                def two():
                    values.append("package-auto")
                    yield values
                    values.pop()
                """
            )
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_package_autouse():
                    assert values == ["package-auto"]
                def test_package(one):
                    assert values == ["package-auto", "package"]
                """
            )
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_collect_custom_items(self, pytester: Pytester) -> None:
        pytester.copy_example("fixtures/custom_item")
        result = pytester.runpytest("foo")
        result.stdout.fnmatch_lines(["*passed*"])


class TestAutouseDiscovery:
    @pytest.fixture
    def pytester(self, pytester: Pytester) -> Pytester:
        pytester.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def perfunction(request, tmp_path):
                pass

            @pytest.fixture()
            def arg1(tmp_path):
                pass
            @pytest.fixture(autouse=True)
            def perfunction2(arg1):
                pass

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return pytester

    def test_parsefactories_conftest(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            from _pytest.pytester import get_public_names
            def test_check_setup(item, fm):
                autousenames = list(fm._getautousenames(item.nodeid))
                assert len(get_public_names(autousenames)) == 2
                assert "perfunction2" in autousenames
                assert "perfunction" in autousenames
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_two_classes_separated_autouse(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            class TestA(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup1(self):
                    self.values.append(1)
                def test_setup1(self):
                    assert self.values == [1]
            class TestB(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup2(self):
                    self.values.append(1)
                def test_setup2(self):
                    assert self.values == [1]
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_setup_at_classlevel(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            class TestClass(object):
                @pytest.fixture(autouse=True)
                def permethod(self, request):
                    request.instance.funcname = request.function.__name__
                def test_method1(self):
                    assert self.funcname == "test_method1"
                def test_method2(self):
                    assert self.funcname == "test_method2"
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=2)

    @pytest.mark.xfail(reason="'enabled' feature not implemented")
    def test_setup_enabled_functionnode(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def enabled(parentnode, markers):
                return "needsdb" in markers

            @pytest.fixture(params=[1,2])
            def db(request):
                return request.param

            @pytest.fixture(enabled=enabled, autouse=True)
            def createdb(db):
                pass

            def test_func1(request):
                assert "db" not in request.fixturenames

            @pytest.mark.needsdb
            def test_func2(request):
                assert "db" in request.fixturenames
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_callables_nocode(self, pytester: Pytester) -> None:
        """An imported mock.call would break setup/factory discovery due to
        it being callable and __code__ not being a code object."""
        pytester.makepyfile(
            """
           class _call(tuple):
               def __call__(self, *k, **kw):
                   pass
               def __getattr__(self, k):
                   return self

           call = _call()
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(failed=0, passed=0)

    def test_autouse_in_conftests(self, pytester: Pytester) -> None:
        a = pytester.mkdir("a")
        b = pytester.mkdir("a1")
        conftest = pytester.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def hello():
                xxx
        """
        )
        conftest.rename(a.joinpath(conftest.name))
        a.joinpath("test_something.py").write_text("def test_func(): pass")
        b.joinpath("test_otherthing.py").write_text("def test_func(): pass")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            """
            *1 passed*1 error*
        """
        )

    def test_autouse_in_module_and_two_classes(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(autouse=True)
            def append1():
                values.append("module")
            def test_x():
                assert values == ["module"]

            class TestA(object):
                @pytest.fixture(autouse=True)
                def append2(self):
                    values.append("A")
                def test_hello(self):
                    assert values == ["module", "module", "A"], values
            class TestA2(object):
                def test_world(self):
                    assert values == ["module", "module", "A", "module"], values
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=3)


class TestAutouseManagement:
    def test_autouse_conftest_mid_directory(self, pytester: Pytester) -> None:
        pkgdir = pytester.mkpydir("xyz123")
        pkgdir.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture(autouse=True)
                def app():
                    import sys
                    sys._myapp = "hello"
                """
            )
        )
        sub = pkgdir.joinpath("tests")
        sub.mkdir()
        t = sub.joinpath("test_app.py")
        t.touch()
        t.write_text(
            textwrap.dedent(
                """\
                import sys
                def test_app():
                    assert sys._myapp == "hello"
                """
            )
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=1)
```
### 8 - testing/python/fixtures.py:

Start line: 3845, End line: 4475

```python
class TestParameterizedSubRequest:

    def test_non_relative_path(self, pytester: Pytester) -> None:
        tests_dir = pytester.mkdir("tests")
        fixdir = pytester.mkdir("fixtures")
        fixfile = fixdir.joinpath("fix.py")
        fixfile.write_text(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[0, 1, 2])
                def fix_with_param(request):
                    return request.param
                """
            )
        )

        testfile = tests_dir.joinpath("test_foos.py")
        testfile.write_text(
            textwrap.dedent(
                """\
                from fix import fix_with_param

                def test_foo(request):
                    request.getfixturevalue('fix_with_param')
                """
            )
        )

        os.chdir(tests_dir)
        pytester.syspathinsert(fixdir)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                f"{fixfile}:4",
                "Requested here:",
                "test_foos.py:4",
                "*1 failed*",
            ]
        )

        # With non-overlapping rootdir, passing tests_dir.
        rootdir = pytester.mkdir("rootdir")
        os.chdir(rootdir)
        result = pytester.runpytest("--rootdir", rootdir, tests_dir)
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                f"{fixfile}:4",
                "Requested here:",
                f"{testfile}:4",
                "*1 failed*",
            ]
        )


def test_pytest_fixture_setup_and_post_finalizer_hook(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        def pytest_fixture_setup(fixturedef, request):
            print('ROOT setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        def pytest_fixture_post_finalizer(fixturedef, request):
            print('ROOT finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
    """
    )
    pytester.makepyfile(
        **{
            "tests/conftest.py": """
            def pytest_fixture_setup(fixturedef, request):
                print('TESTS setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
            def pytest_fixture_post_finalizer(fixturedef, request):
                print('TESTS finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        """,
            "tests/test_hooks.py": """
            import pytest

            @pytest.fixture()
            def my_fixture():
                return 'some'

            def test_func(my_fixture):
                print('TEST test_func')
                assert my_fixture == 'some'
        """,
        }
    )
    result = pytester.runpytest("-s")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*TESTS setup hook called for my_fixture from test_func*",
            "*ROOT setup hook called for my_fixture from test_func*",
            "*TEST test_func*",
            "*TESTS finalizer hook called for my_fixture from test_func*",
            "*ROOT finalizer hook called for my_fixture from test_func*",
        ]
    )


class TestScopeOrdering:
    """Class of tests that ensure fixtures are ordered based on their scopes (#2405)"""

    @pytest.mark.parametrize("variant", ["mark", "autouse"])
    def test_func_closure_module_auto(
        self, pytester: Pytester, variant, monkeypatch
    ) -> None:
        """Semantically identical to the example posted in #2405 when ``use_mark=True``"""
        monkeypatch.setenv("FIXTURE_ACTIVATION_VARIANT", variant)
        pytester.makepyfile(
            """
            import warnings
            import os
            import pytest
            VAR = 'FIXTURE_ACTIVATION_VARIANT'
            VALID_VARS = ('autouse', 'mark')

            VARIANT = os.environ.get(VAR)
            if VARIANT is None or VARIANT not in VALID_VARS:
                warnings.warn("{!r} is not  in {}, assuming autouse".format(VARIANT, VALID_VARS) )
                variant = 'mark'

            @pytest.fixture(scope='module', autouse=VARIANT == 'autouse')
            def m1(): pass

            if VARIANT=='mark':
                pytestmark = pytest.mark.usefixtures('m1')

            @pytest.fixture(scope='function', autouse=True)
            def f1(): pass

            def test_func(m1):
                pass
        """
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_with_native_fixtures(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Sanity check that verifies the order returned by the closures and the actual fixture execution order:
        The execution order may differ because of fixture inter-dependencies.
        """
        monkeypatch.setattr(pytest, "FIXTURE_ORDER", [], raising=False)
        pytester.makepyfile(
            """
            import pytest

            FIXTURE_ORDER = pytest.FIXTURE_ORDER

            @pytest.fixture(scope="session")
            def s1():
                FIXTURE_ORDER.append('s1')

            @pytest.fixture(scope="package")
            def p1():
                FIXTURE_ORDER.append('p1')

            @pytest.fixture(scope="module")
            def m1():
                FIXTURE_ORDER.append('m1')

            @pytest.fixture(scope='session')
            def my_tmp_path_factory():
                FIXTURE_ORDER.append('my_tmp_path_factory')

            @pytest.fixture
            def my_tmp_path(my_tmp_path_factory):
                FIXTURE_ORDER.append('my_tmp_path')

            @pytest.fixture
            def f1(my_tmp_path):
                FIXTURE_ORDER.append('f1')

            @pytest.fixture
            def f2():
                FIXTURE_ORDER.append('f2')

            def test_foo(f1, p1, m1, f2, s1): pass
        """
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        # order of fixtures based on their scope and position in the parameter list
        assert (
            request.fixturenames
            == "s1 my_tmp_path_factory p1 m1 f1 f2 my_tmp_path".split()
        )
        pytester.runpytest()
        # actual fixture execution differs: dependent fixtures must be created first ("my_tmp_path")
        FIXTURE_ORDER = pytest.FIXTURE_ORDER  # type: ignore[attr-defined]
        assert FIXTURE_ORDER == "s1 my_tmp_path_factory p1 m1 my_tmp_path f1 f2".split()

    def test_func_closure_module(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            def test_func(f1, m1):
                pass
        """
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_scopes_reordered(self, pytester: Pytester) -> None:
        """Test ensures that fixtures are ordered by scope regardless of the order of the parameters, although
        fixtures of same scope keep the declared order
        """
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class')
                def c1(cls): pass

                def test_func(self, f2, f1, c1, m1, s1):
                    pass
        """
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        assert request.fixturenames == "s1 m1 c1 f2 f1".split()

    def test_func_closure_same_scope_closer_root_first(
        self, pytester: Pytester
    ) -> None:
        """Auto-use fixtures of same scope are ordered by closer-to-root first"""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m_conf(): pass
        """
        )
        pytester.makepyfile(
            **{
                "sub/conftest.py": """
                import pytest

                @pytest.fixture(scope='package', autouse=True)
                def p_sub(): pass

                @pytest.fixture(scope='module', autouse=True)
                def m_sub(): pass
            """,
                "sub/__init__.py": "",
                "sub/test_func.py": """
                import pytest

                @pytest.fixture(scope='module', autouse=True)
                def m_test(): pass

                @pytest.fixture(scope='function')
                def f1(): pass

                def test_func(m_test, f1):
                    pass
        """,
            }
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        assert request.fixturenames == "p_sub m_conf m_sub m_test f1".split()

    def test_func_closure_all_scopes_complex(self, pytester: Pytester) -> None:
        """Complex test involving all scopes and mixing autouse with normal fixtures"""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='package', autouse=True)
            def p1(): pass
        """
        )
        pytester.makepyfile(**{"__init__.py": ""})
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m1(): pass

            @pytest.fixture(scope='module')
            def m2(s1): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class', autouse=True)
                def c1(self):
                    pass

                def test_func(self, f2, f1, m2):
                    pass
        """
        )
        items, _ = pytester.inline_genitems()
        request = FixtureRequest(items[0], _ispytest=True)
        assert request.fixturenames == "s1 p1 m1 m2 c1 f2 f1".split()

    def test_multiple_packages(self, pytester: Pytester) -> None:
        """Complex test involving multiple package fixtures. Make sure teardowns
        are executed in order.
        .
        └── root
            ├── __init__.py
            ├── sub1
            │   ├── __init__.py
            │   ├── conftest.py
            │   └── test_1.py
            └── sub2
                ├── __init__.py
                ├── conftest.py
                └── test_2.py
        """
        root = pytester.mkdir("root")
        root.joinpath("__init__.py").write_text("values = []")
        sub1 = root.joinpath("sub1")
        sub1.mkdir()
        sub1.joinpath("__init__.py").touch()
        sub1.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub1")
                yield values
                assert values.pop() == "pre-sub1"
        """
            )
        )
        sub1.joinpath("test_1.py").write_text(
            textwrap.dedent(
                """\
            from .. import values
            def test_1(fix):
                assert values == ["pre-sub1"]
        """
            )
        )
        sub2 = root.joinpath("sub2")
        sub2.mkdir()
        sub2.joinpath("__init__.py").touch()
        sub2.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub2")
                yield values
                assert values.pop() == "pre-sub2"
        """
            )
        )
        sub2.joinpath("test_2.py").write_text(
            textwrap.dedent(
                """\
            from .. import values
            def test_2(fix):
                assert values == ["pre-sub2"]
        """
            )
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_class_fixture_self_instance(self, pytester: Pytester) -> None:
        """Check that plugin classes which implement fixtures receive the plugin instance
        as self (see #2270).
        """
        pytester.makeconftest(
            """
            import pytest

            def pytest_configure(config):
                config.pluginmanager.register(MyPlugin())

            class MyPlugin():
                def __init__(self):
                    self.arg = 1

                @pytest.fixture(scope='function')
                def myfix(self):
                    assert isinstance(self, MyPlugin)
                    return self.arg
        """
        )

        pytester.makepyfile(
            """
            class TestClass(object):
                def test_1(self, myfix):
                    assert myfix == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)


def test_call_fixture_function_error():
    """Check if an error is raised if a fixture function is called directly (#4545)"""

    @pytest.fixture
    def fix():
        raise NotImplementedError()

    with pytest.raises(pytest.fail.Exception):
        assert fix() == 1


def test_fixture_param_shadowing(pytester: Pytester) -> None:
    """Parametrized arguments would be shadowed if a fixture with the same name also exists (#5036)"""
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture(params=['a', 'b'])
        def argroot(request):
            return request.param

        @pytest.fixture
        def arg(argroot):
            return argroot

        # This should only be parametrized directly
        @pytest.mark.parametrize("arg", [1])
        def test_direct(arg):
            assert arg == 1

        # This should be parametrized based on the fixtures
        def test_normal_fixture(arg):
            assert isinstance(arg, str)

        # Indirect should still work:

        @pytest.fixture
        def arg2(request):
            return 2*request.param

        @pytest.mark.parametrize("arg2", [1], indirect=True)
        def test_indirect(arg2):
            assert arg2 == 2
    """
    )
    # Only one test should have run
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(["*::test_direct[[]1[]]*"])
    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]a[]]*"])
    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]b[]]*"])
    result.stdout.fnmatch_lines(["*::test_indirect[[]1[]]*"])


def test_fixture_named_request(pytester: Pytester) -> None:
    pytester.copy_example("fixtures/test_fixture_named_request.py")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "*'request' is a reserved word for fixtures, use another name:",
            "  *test_fixture_named_request.py:5",
        ]
    )


def test_indirect_fixture_does_not_break_scope(pytester: Pytester) -> None:
    """Ensure that fixture scope is respected when using indirect fixtures (#570)"""
    pytester.makepyfile(
        """
        import pytest
        instantiated  = []

        @pytest.fixture(scope="session")
        def fixture_1(request):
            instantiated.append(("fixture_1", request.param))


        @pytest.fixture(scope="session")
        def fixture_2(request):
            instantiated.append(("fixture_2", request.param))


        scenarios = [
            ("A", "a1"),
            ("A", "a2"),
            ("B", "b1"),
            ("B", "b2"),
            ("C", "c1"),
            ("C", "c2"),
        ]

        @pytest.mark.parametrize(
            "fixture_1,fixture_2", scenarios, indirect=["fixture_1", "fixture_2"]
        )
        def test_create_fixtures(fixture_1, fixture_2):
            pass


        def test_check_fixture_instantiations():
            assert instantiated == [
                ('fixture_1', 'A'),
                ('fixture_2', 'a1'),
                ('fixture_2', 'a2'),
                ('fixture_1', 'B'),
                ('fixture_2', 'b1'),
                ('fixture_2', 'b2'),
                ('fixture_1', 'C'),
                ('fixture_2', 'c1'),
                ('fixture_2', 'c2'),
            ]
    """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=7)


def test_fixture_parametrization_nparray(pytester: Pytester) -> None:
    pytest.importorskip("numpy")

    pytester.makepyfile(
        """
        from numpy import linspace
        from pytest import fixture

        @fixture(params=linspace(1, 10, 10))
        def value(request):
            return request.param

        def test_bug(value):
            assert value == value
    """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=10)


def test_fixture_arg_ordering(pytester: Pytester) -> None:
    """
    This test describes how fixtures in the same scope but without explicit dependencies
    between them are created. While users should make dependencies explicit, often
    they rely on this order, so this test exists to catch regressions in this regard.
    See #6540 and #6492.
    """
    p1 = pytester.makepyfile(
        """
        import pytest

        suffixes = []

        @pytest.fixture
        def fix_1(): suffixes.append("fix_1")
        @pytest.fixture
        def fix_2(): suffixes.append("fix_2")
        @pytest.fixture
        def fix_3(): suffixes.append("fix_3")
        @pytest.fixture
        def fix_4(): suffixes.append("fix_4")
        @pytest.fixture
        def fix_5(): suffixes.append("fix_5")

        @pytest.fixture
        def fix_combined(fix_1, fix_2, fix_3, fix_4, fix_5): pass

        def test_suffix(fix_combined):
            assert suffixes == ["fix_1", "fix_2", "fix_3", "fix_4", "fix_5"]
        """
    )
    result = pytester.runpytest("-vv", str(p1))
    assert result.ret == 0


def test_yield_fixture_with_no_value(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.fixture(name='custom')
        def empty_yield():
            if False:
                yield

        def test_fixt(custom):
            pass
        """
    )
    expected = "E               ValueError: custom did not yield a value"
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([expected])
    assert result.ret == ExitCode.TESTS_FAILED
```
### 9 - doc/en/conf.py:

Start line: 1, End line: 115

```python
#
# pytest documentation build configuration file, created by
# sphinx-quickstart on Fri Oct  8 17:54:28 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
# The short X.Y version.
import ast
import os
import shutil
import sys
from textwrap import dedent
from typing import List
from typing import TYPE_CHECKING

from _pytest import __version__ as version

if TYPE_CHECKING:
    import sphinx.application


release = ".".join(version.split(".")[:2])

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

autodoc_member_order = "bysource"
autodoc_typehints = "description"
todo_include_todos = 1

latex_engine = "lualatex"

latex_elements = {
    "preamble": dedent(
        r"""
        \directlua{
            luaotfload.add_fallback("fallbacks", {
                "Noto Serif CJK SC:style=Regular;",
                "Symbola:Style=Regular;"
            })
        }

        \setmainfont{FreeSerif}[RawFeature={fallback=fallbacks}]
        """
    )
}

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "pallets_sphinx_themes",
    "pygments_pytest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_removed_in",
    "sphinxcontrib_trio",
]

# Building PDF docs on readthedocs requires inkscape for svg to pdf
# conversion. The relevant plugin is not useful for normal HTML builds, but
# it still raises warnings and fails CI if inkscape is not available. So
# only use the plugin if inkscape is actually available.
if shutil.which("inkscape"):
    extensions.append("sphinxcontrib.inkscapeconverter")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "pytest"
copyright = "2015–2021, holger krekel and pytest-dev team"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
```
### 10 - doc/en/example/assertion/failure_demo.py:

Start line: 163, End line: 202

```python
class TestRaises:
    def test_raises(self):
        s = "qwe"
        raises(TypeError, int, s)

    def test_raises_doesnt(self):
        raises(OSError, int, "3")

    def test_raise(self):
        raise ValueError("demo error")

    def test_tupleerror(self):
        a, b = [1]  # NOQA

    def test_reinterpret_fails_with_print_for_the_fun_of_it(self):
        items = [1, 2, 3]
        print(f"items is {items!r}")
        a, b = items.pop()

    def test_some_error(self):
        if namenotexi:  # NOQA
            pass

    def func1(self):
        assert 41 == 42


# thanks to Matthew Scott for this test
def test_dynamic_compile_shows_nicely():
    import importlib.util
    import sys

    src = "def foo():\n assert 1 == 0\n"
    name = "abc-123"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    code = compile(src, name, "exec")
    exec(code, module.__dict__)
    sys.modules[name] = module
    module.foo()
```
### 118 - src/_pytest/_code/source.py:

Start line: 168, End line: 213

```python
def getstatementrange_ast(
    lineno: int,
    source: Source,
    assertion: bool = False,
    astnode: Optional[ast.AST] = None,
) -> Tuple[ast.AST, int, int]:
    if astnode is None:
        content = str(source)
        # See #4260:
        # Don't produce duplicate warnings when compiling source to find AST.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            astnode = ast.parse(content, "source", "exec")

    start, end = get_statement_startend2(lineno, astnode)
    # We need to correct the end:
    # - ast-parsing strips comments
    # - there might be empty lines
    # - we might have lesser indented code blocks at the end
    if end is None:
        end = len(source.lines)

    if end > start + 1:
        # Make sure we don't span differently indented code blocks
        # by using the BlockFinder helper used which inspect.getsource() uses itself.
        block_finder = inspect.BlockFinder()
        # If we start with an indented line, put blockfinder to "started" mode.
        block_finder.started = source.lines[start][0].isspace()
        it = ((x + "\n") for x in source.lines[start:end])
        try:
            for tok in tokenize.generate_tokens(lambda: next(it)):
                block_finder.tokeneater(*tok)
        except (inspect.EndOfBlock, IndentationError):
            end = block_finder.last + start
        except Exception:
            pass

    # The end might still point to a comment or empty line, correct it.
    while end:
        line = source.lines[end - 1].lstrip()
        if line.startswith("#") or not line:
            end -= 1
        else:
            break
    return astnode, start, end
```
