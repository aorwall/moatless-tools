# pytest-dev__pytest-7535

| **pytest-dev/pytest** | `7ec6401ffabf79d52938ece5b8ff566a8b9c260e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/_code/code.py b/src/_pytest/_code/code.py
--- a/src/_pytest/_code/code.py
+++ b/src/_pytest/_code/code.py
@@ -262,7 +262,15 @@ def __str__(self) -> str:
             raise
         except BaseException:
             line = "???"
-        return "  File %r:%d in %s\n  %s\n" % (self.path, self.lineno + 1, name, line)
+        # This output does not quite match Python's repr for traceback entries,
+        # but changing it to do so would break certain plugins.  See
+        # https://github.com/pytest-dev/pytest/pull/7535/ for details.
+        return "  File %r:%d in %s\n  %s\n" % (
+            str(self.path),
+            self.lineno + 1,
+            name,
+            line,
+        )
 
     @property
     def name(self) -> str:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/_code/code.py | 265 | 265 | - | 1 | -


## Problem Statement

```
pytest 6: Traceback in pytest.raises contains repr of py.path.local
The [werkzeug](https://github.com/pallets/werkzeug) tests fail with pytest 6:

\`\`\`python
    def test_import_string_provides_traceback(tmpdir, monkeypatch):
        monkeypatch.syspath_prepend(str(tmpdir))
        # Couple of packages
        dir_a = tmpdir.mkdir("a")
        dir_b = tmpdir.mkdir("b")
        # Totally packages, I promise
        dir_a.join("__init__.py").write("")
        dir_b.join("__init__.py").write("")
        # 'aa.a' that depends on 'bb.b', which in turn has a broken import
        dir_a.join("aa.py").write("from b import bb")
        dir_b.join("bb.py").write("from os import a_typo")
    
        # Do we get all the useful information in the traceback?
        with pytest.raises(ImportError) as baz_exc:
            utils.import_string("a.aa")
        traceback = "".join(str(line) for line in baz_exc.traceback)
>       assert "bb.py':1" in traceback  # a bit different than typical python tb
E       assert "bb.py':1" in "  File local('/home/florian/tmp/werkzeugtest/werkzeug/tests/test_utils.py'):205 in test_import_string_provides_traceb...l('/tmp/pytest-of-florian/pytest-29/test_import_string_provides_tr0/b/bb.py'):1 in <module>\n  from os import a_typo\n"
\`\`\`

This is because of 2ee90887b77212e2e8f427ed6db9feab85f06b49 (#7274, "code: remove last usage of py.error") - it removed the `str(...)`, but the format string uses `%r`, so now we get the repr of the `py.path.local` object instead of the repr of a string.

I believe this should continue to use `"%r" % str(self.path)` so the output is the same in all cases.

cc @bluetech @hroncok 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 src/_pytest/_code/code.py** | 960 | 984| 218 | 218 | 9759 | 
| 2 | 2 testing/python/collect.py | 965 | 984| 178 | 396 | 19525 | 
| 3 | 3 src/pytest/__init__.py | 1 | 100| 671 | 1067 | 20196 | 
| 4 | **3 src/_pytest/_code/code.py** | 890 | 904| 130 | 1197 | 20196 | 
| 5 | 4 testing/python/raises.py | 1 | 51| 332 | 1529 | 22273 | 
| 6 | **4 src/_pytest/_code/code.py** | 987 | 1000| 135 | 1664 | 22273 | 
| 7 | 4 testing/python/collect.py | 924 | 963| 352 | 2016 | 22273 | 
| 8 | **4 src/_pytest/_code/code.py** | 1087 | 1110| 215 | 2231 | 22273 | 
| 9 | **4 src/_pytest/_code/code.py** | 1170 | 1180| 138 | 2369 | 22273 | 
| 10 | 5 src/_pytest/doctest.py | 134 | 167| 279 | 2648 | 27934 | 
| 11 | **5 src/_pytest/_code/code.py** | 327 | 341| 138 | 2786 | 27934 | 
| 12 | 5 testing/python/collect.py | 1035 | 1061| 216 | 3002 | 27934 | 
| 13 | **5 src/_pytest/_code/code.py** | 926 | 957| 288 | 3290 | 27934 | 
| 14 | 5 testing/python/raises.py | 190 | 219| 261 | 3551 | 27934 | 
| 15 | **5 src/_pytest/_code/code.py** | 1059 | 1084| 236 | 3787 | 27934 | 
| 16 | 5 testing/python/raises.py | 221 | 238| 205 | 3992 | 27934 | 
| 17 | 5 testing/python/collect.py | 1013 | 1033| 191 | 4183 | 27934 | 
| 18 | 5 testing/python/collect.py | 280 | 347| 474 | 4657 | 27934 | 
| 19 | 6 src/_pytest/debugging.py | 357 | 379| 205 | 4862 | 30836 | 
| 20 | 6 src/_pytest/doctest.py | 305 | 374| 615 | 5477 | 30836 | 
| 21 | 6 testing/python/raises.py | 129 | 156| 223 | 5700 | 30836 | 
| 22 | 6 testing/python/collect.py | 81 | 114| 241 | 5941 | 30836 | 
| 23 | **6 src/_pytest/_code/code.py** | 768 | 799| 264 | 6205 | 30836 | 
| 24 | 7 src/_pytest/nodes.py | 1 | 49| 317 | 6522 | 36020 | 
| 25 | 8 testing/python/metafunc.py | 1375 | 1391| 110 | 6632 | 50586 | 
| 26 | **8 src/_pytest/_code/code.py** | 1 | 50| 316 | 6948 | 50586 | 
| 27 | 9 src/_pytest/_io/saferepr.py | 1 | 35| 263 | 7211 | 51517 | 
| 28 | 9 testing/python/metafunc.py | 1393 | 1417| 210 | 7421 | 51517 | 
| 29 | 10 src/_pytest/compat.py | 1 | 87| 476 | 7897 | 54644 | 
| 30 | 11 src/_pytest/python_api.py | 682 | 713| 312 | 8209 | 61319 | 
| 31 | 12 doc/en/example/assertion/failure_demo.py | 163 | 202| 270 | 8479 | 62968 | 
| 32 | 13 src/_pytest/config/__init__.py | 96 | 120| 206 | 8685 | 74218 | 
| 33 | 13 testing/python/collect.py | 1 | 35| 241 | 8926 | 74218 | 
| 34 | 14 src/_pytest/fixtures.py | 865 | 896| 248 | 9174 | 88617 | 
| 35 | **14 src/_pytest/_code/code.py** | 1113 | 1133| 173 | 9347 | 88617 | 
| 36 | 15 testing/python/fixtures.py | 1 | 82| 490 | 9837 | 115915 | 
| 37 | 16 src/_pytest/__init__.py | 1 | 9| 0 | 9837 | 115971 | 
| 38 | 17 src/_pytest/reports.py | 1 | 54| 369 | 10206 | 120128 | 
| 39 | 18 testing/python/integration.py | 1 | 40| 272 | 10478 | 123145 | 
| 40 | 19 doc/en/example/xfail_demo.py | 1 | 39| 143 | 10621 | 123289 | 
| 41 | 19 testing/python/metafunc.py | 112 | 134| 190 | 10811 | 123289 | 
| 42 | 19 src/_pytest/python_api.py | 716 | 734| 165 | 10976 | 123289 | 
| 43 | 19 testing/python/raises.py | 53 | 79| 179 | 11155 | 123289 | 
| 44 | **19 src/_pytest/_code/code.py** | 708 | 733| 252 | 11407 | 123289 | 
| 45 | 20 src/_pytest/unittest.py | 166 | 197| 265 | 11672 | 126105 | 
| 46 | 20 testing/python/metafunc.py | 1317 | 1349| 211 | 11883 | 126105 | 
| 47 | 20 testing/python/collect.py | 1252 | 1276| 196 | 12079 | 126105 | 
| 48 | 20 testing/python/collect.py | 116 | 129| 110 | 12189 | 126105 | 
| 49 | 21 src/_pytest/python.py | 325 | 340| 171 | 12360 | 139581 | 
| 50 | 21 testing/python/collect.py | 986 | 1011| 170 | 12530 | 139581 | 
| 51 | 21 testing/python/collect.py | 37 | 59| 187 | 12717 | 139581 | 
| 52 | 22 src/_pytest/config/exceptions.py | 1 | 10| 0 | 12717 | 139626 | 
| 53 | 22 testing/python/raises.py | 158 | 188| 238 | 12955 | 139626 | 
| 54 | **22 src/_pytest/_code/code.py** | 735 | 766| 348 | 13303 | 139626 | 
| 55 | 22 src/_pytest/reports.py | 382 | 411| 225 | 13528 | 139626 | 
| 56 | 22 src/_pytest/compat.py | 203 | 241| 264 | 13792 | 139626 | 
| 57 | 22 src/_pytest/python_api.py | 1 | 42| 229 | 14021 | 139626 | 
| 58 | 23 src/_pytest/pytester.py | 1 | 56| 332 | 14353 | 152188 | 
| 59 | 24 src/_pytest/pathlib.py | 447 | 559| 919 | 15272 | 156379 | 
| 60 | 24 src/_pytest/python_api.py | 568 | 681| 1010 | 16282 | 156379 | 
| 61 | 25 src/_pytest/_io/__init__.py | 1 | 9| 0 | 16282 | 156413 | 
| 62 | 26 src/_pytest/pastebin.py | 95 | 112| 174 | 16456 | 157326 | 
| 63 | 26 testing/python/collect.py | 1102 | 1120| 154 | 16610 | 157326 | 
| 64 | 26 testing/python/collect.py | 575 | 663| 591 | 17201 | 157326 | 
| 65 | 26 src/_pytest/doctest.py | 496 | 531| 297 | 17498 | 157326 | 
| 66 | 26 testing/python/integration.py | 331 | 369| 237 | 17735 | 157326 | 
| 67 | 27 src/pytest/__main__.py | 1 | 8| 0 | 17735 | 157352 | 
| 68 | 27 testing/python/collect.py | 197 | 250| 318 | 18053 | 157352 | 
| 69 | 27 testing/python/collect.py | 1193 | 1219| 179 | 18232 | 157352 | 
| 70 | 27 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 18395 | 157352 | 
| 71 | 27 testing/python/collect.py | 1153 | 1190| 209 | 18604 | 157352 | 
| 72 | 27 testing/python/integration.py | 42 | 74| 250 | 18854 | 157352 | 
| 73 | 28 src/_pytest/monkeypatch.py | 52 | 78| 188 | 19042 | 160215 | 
| 74 | 28 src/_pytest/python.py | 1322 | 1354| 259 | 19301 | 160215 | 
| 75 | 28 testing/python/metafunc.py | 573 | 599| 238 | 19539 | 160215 | 
| 76 | 29 src/_pytest/mark/structures.py | 1 | 42| 219 | 19758 | 164465 | 
| 77 | 29 src/_pytest/python.py | 1575 | 1616| 386 | 20144 | 164465 | 
| 78 | 29 src/_pytest/config/__init__.py | 954 | 970| 170 | 20314 | 164465 | 
| 79 | **29 src/_pytest/_code/code.py** | 907 | 923| 180 | 20494 | 164465 | 
| 80 | 29 src/_pytest/python.py | 1384 | 1443| 452 | 20946 | 164465 | 
| 81 | 30 src/_pytest/terminal.py | 524 | 573| 416 | 21362 | 175129 | 
| 82 | 30 src/_pytest/debugging.py | 275 | 297| 197 | 21559 | 175129 | 
| 83 | 30 testing/python/collect.py | 61 | 79| 187 | 21746 | 175129 | 
| 84 | 30 testing/python/metafunc.py | 709 | 739| 239 | 21985 | 175129 | 
| 85 | 30 src/_pytest/pathlib.py | 1 | 56| 288 | 22273 | 175129 | 
| 86 | **30 src/_pytest/_code/code.py** | 842 | 887| 428 | 22701 | 175129 | 
| 87 | 30 testing/python/collect.py | 831 | 853| 200 | 22901 | 175129 | 
| 88 | 30 src/_pytest/python.py | 1 | 75| 531 | 23432 | 175129 | 
| 89 | 30 testing/python/fixtures.py | 1067 | 2061| 6159 | 29591 | 175129 | 
| 90 | 31 src/_pytest/runner.py | 188 | 214| 230 | 29821 | 178783 | 
| 91 | 32 src/_pytest/main.py | 671 | 690| 185 | 30006 | 185014 | 
| 92 | 32 testing/python/fixtures.py | 2509 | 3536| 6205 | 36211 | 185014 | 
| 93 | 32 testing/python/metafunc.py | 381 | 416| 299 | 36510 | 185014 | 
| 94 | 32 src/_pytest/terminal.py | 262 | 275| 117 | 36627 | 185014 | 
| 95 | 32 src/_pytest/monkeypatch.py | 81 | 110| 203 | 36830 | 185014 | 
| 96 | 33 src/_pytest/outcomes.py | 51 | 66| 124 | 36954 | 186741 | 
| 97 | 33 testing/python/integration.py | 95 | 141| 268 | 37222 | 186741 | 
| 98 | 34 src/_pytest/assertion/rewrite.py | 450 | 465| 111 | 37333 | 196434 | 
| 99 | 35 doc/en/conftest.py | 1 | 2| 0 | 37333 | 196441 | 
| 100 | 35 src/_pytest/assertion/rewrite.py | 1 | 53| 327 | 37660 | 196441 | 
| 101 | 35 src/_pytest/outcomes.py | 69 | 119| 358 | 38018 | 196441 | 
| 102 | 35 testing/python/collect.py | 665 | 686| 121 | 38139 | 196441 | 


## Patch

```diff
diff --git a/src/_pytest/_code/code.py b/src/_pytest/_code/code.py
--- a/src/_pytest/_code/code.py
+++ b/src/_pytest/_code/code.py
@@ -262,7 +262,15 @@ def __str__(self) -> str:
             raise
         except BaseException:
             line = "???"
-        return "  File %r:%d in %s\n  %s\n" % (self.path, self.lineno + 1, name, line)
+        # This output does not quite match Python's repr for traceback entries,
+        # but changing it to do so would break certain plugins.  See
+        # https://github.com/pytest-dev/pytest/pull/7535/ for details.
+        return "  File %r:%d in %s\n  %s\n" % (
+            str(self.path),
+            self.lineno + 1,
+            name,
+            line,
+        )
 
     @property
     def name(self) -> str:

```

## Test Patch

```diff
diff --git a/testing/code/test_code.py b/testing/code/test_code.py
--- a/testing/code/test_code.py
+++ b/testing/code/test_code.py
@@ -1,3 +1,4 @@
+import re
 import sys
 from types import FrameType
 from unittest import mock
@@ -170,6 +171,15 @@ def test_getsource(self) -> None:
         assert len(source) == 6
         assert "assert False" in source[5]
 
+    def test_tb_entry_str(self):
+        try:
+            assert False
+        except AssertionError:
+            exci = ExceptionInfo.from_current()
+        pattern = r"  File '.*test_code.py':\d+ in test_tb_entry_str\n  assert False"
+        entry = str(exci.traceback[0])
+        assert re.match(pattern, entry)
+
 
 class TestReprFuncArgs:
     def test_not_raise_exception_with_mixed_encoding(self, tw_mock) -> None:

```


## Code snippets

### 1 - src/_pytest/_code/code.py:

Start line: 960, End line: 984

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprTraceback(TerminalRepr):
    reprentries = attr.ib(type=Sequence[Union["ReprEntry", "ReprEntryNative"]])
    extraline = attr.ib(type=Optional[str])
    style = attr.ib(type="_TracebackStyle")

    entrysep = "_ "

    def toterminal(self, tw: TerminalWriter) -> None:
        # the entries might have different styles
        for i, entry in enumerate(self.reprentries):
            if entry.style == "long":
                tw.line("")
            entry.toterminal(tw)
            if i < len(self.reprentries) - 1:
                next_entry = self.reprentries[i + 1]
                if (
                    entry.style == "long"
                    or entry.style == "short"
                    and next_entry.style == "long"
                ):
                    tw.sep(self.entrysep)

        if self.extraline:
            tw.line(self.extraline)
```
### 2 - testing/python/collect.py:

Start line: 965, End line: 984

```python
class TestTracebackCutting:

    def test_traceback_error_during_import(self, testdir):
        testdir.makepyfile(
            """
            x = 1
            x = 2
            x = 17
            asd
        """
        )
        result = testdir.runpytest()
        assert result.ret != 0
        out = result.stdout.str()
        assert "x = 1" not in out
        assert "x = 2" not in out
        result.stdout.fnmatch_lines([" *asd*", "E*NameError*"])
        result = testdir.runpytest("--fulltrace")
        out = result.stdout.str()
        assert "x = 1" in out
        assert "x = 2" in out
        result.stdout.fnmatch_lines([">*asd*", "E*NameError*"])
```
### 3 - src/pytest/__init__.py:

Start line: 1, End line: 100

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
from . import collect
from _pytest import __version__
from _pytest.assertion import register_assert_rewrite
from _pytest.config import cmdline
from _pytest.config import console_main
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import UsageError
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import fillfixtures as _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.main import Session
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import param
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Instance
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import warns
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestWarning

set_trace = __pytestPDB.set_trace

__all__ = [
    "__version__",
    "_fillfuncargs",
    "approx",
    "Class",
    "cmdline",
    "collect",
    "Collector",
    "console_main",
    "deprecated_call",
    "exit",
    "ExitCode",
    "fail",
    "File",
    "fixture",
    "FixtureLookupError",
    "freeze_includes",
    "Function",
    "hookimpl",
    "hookspec",
    "importorskip",
    "Instance",
    "Item",
    "main",
    "mark",
    "Module",
    "Package",
    "param",
    "PytestAssertRewriteWarning",
    "PytestCacheWarning",
    "PytestCollectionWarning",
    "PytestConfigWarning",
    "PytestDeprecationWarning",
    "PytestExperimentalApiWarning",
    "PytestUnhandledCoroutineWarning",
    "PytestUnknownMarkWarning",
    "PytestWarning",
    "raises",
    "register_assert_rewrite",
    "Session",
    "set_trace",
    "skip",
    "UsageError",
    "warns",
    "xfail",
    "yield_fixture",
]
```
### 4 - src/_pytest/_code/code.py:

Start line: 890, End line: 904

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class TerminalRepr:
    def __str__(self) -> str:
        # FYI this is called from pytest-xdist's serialization of exception
        # information.
        io = StringIO()
        tw = TerminalWriter(file=io)
        self.toterminal(tw)
        return io.getvalue().strip()

    def __repr__(self) -> str:
        return "<{} instance at {:0x}>".format(self.__class__, id(self))

    def toterminal(self, tw: TerminalWriter) -> None:
        raise NotImplementedError()
```
### 5 - testing/python/raises.py:

Start line: 1, End line: 51

```python
import re
import sys

import pytest
from _pytest.outcomes import Failed


class TestRaises:
    def test_check_callable(self) -> None:
        with pytest.raises(TypeError, match=r".* must be callable"):
            pytest.raises(RuntimeError, "int('qwe')")  # type: ignore[call-overload]

    def test_raises(self):
        excinfo = pytest.raises(ValueError, int, "qwe")
        assert "invalid literal" in str(excinfo.value)

    def test_raises_function(self):
        excinfo = pytest.raises(ValueError, int, "hello")
        assert "invalid literal" in str(excinfo.value)

    def test_raises_callable_no_exception(self) -> None:
        class A:
            def __call__(self):
                pass

        try:
            pytest.raises(ValueError, A())
        except pytest.fail.Exception:
            pass

    def test_raises_falsey_type_error(self) -> None:
        with pytest.raises(TypeError):
            with pytest.raises(AssertionError, match=0):  # type: ignore[call-overload]
                raise AssertionError("ohai")

    def test_raises_repr_inflight(self):
        """Ensure repr() on an exception info inside a pytest.raises with block works (#4386)"""

        class E(Exception):
            pass

        with pytest.raises(E) as excinfo:
            # this test prints the inflight uninitialized object
            # using repr and str as well as pprint to demonstrate
            # it works
            print(str(excinfo))
            print(repr(excinfo))
            import pprint

            pprint.pprint(excinfo)
            raise E()
```
### 6 - src/_pytest/_code/code.py:

Start line: 987, End line: 1000

```python
class ReprTracebackNative(ReprTraceback):
    def __init__(self, tblines: Sequence[str]) -> None:
        self.style = "native"
        self.reprentries = [ReprEntryNative(tblines)]
        self.extraline = None


@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprEntryNative(TerminalRepr):
    lines = attr.ib(type=Sequence[str])
    style = "native"  # type: _TracebackStyle

    def toterminal(self, tw: TerminalWriter) -> None:
        tw.write("".join(self.lines))
```
### 7 - testing/python/collect.py:

Start line: 924, End line: 963

```python
def test_modulecol_roundtrip(testdir):
    modcol = testdir.getmodulecol("pass", withinit=False)
    trail = modcol.nodeid
    newcol = modcol.session.perform_collect([trail], genitems=0)[0]
    assert modcol.name == newcol.name


class TestTracebackCutting:
    def test_skip_simple(self):
        with pytest.raises(pytest.skip.Exception) as excinfo:
            pytest.skip("xxx")
        assert excinfo.traceback[-1].frame.code.name == "skip"
        assert excinfo.traceback[-1].ishidden()
        assert excinfo.traceback[-2].frame.code.name == "test_skip_simple"
        assert not excinfo.traceback[-2].ishidden()

    def test_traceback_argsetup(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                raise ValueError("xyz")
        """
        )
        p = testdir.makepyfile("def test(hello): pass")
        result = testdir.runpytest(p)
        assert result.ret != 0
        out = result.stdout.str()
        assert "xyz" in out
        assert "conftest.py:5: ValueError" in out
        numentries = out.count("_ _ _")  # separator for traceback entries
        assert numentries == 0

        result = testdir.runpytest("--fulltrace", p)
        out = result.stdout.str()
        assert "conftest.py:5: ValueError" in out
        numentries = out.count("_ _ _ _")  # separator for traceback entries
        assert numentries > 3
```
### 8 - src/_pytest/_code/code.py:

Start line: 1087, End line: 1110

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprFileLocation(TerminalRepr):
    path = attr.ib(type=str, converter=str)
    lineno = attr.ib(type=int)
    message = attr.ib(type=str)

    def toterminal(self, tw: TerminalWriter) -> None:
        # filename and lineno output for each entry,
        # using an output format that most editors understand
        msg = self.message
        i = msg.find("\n")
        if i != -1:
            msg = msg[:i]
        tw.write(self.path, bold=True, red=True)
        tw.line(":{}: {}".format(self.lineno, msg))


@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprLocals(TerminalRepr):
    lines = attr.ib(type=Sequence[str])

    def toterminal(self, tw: TerminalWriter, indent="") -> None:
        for line in self.lines:
            tw.line(indent + line)
```
### 9 - src/_pytest/_code/code.py:

Start line: 1170, End line: 1180

```python
# relative paths that we use to filter traceback entries from appearing to the user;
# see filter_traceback
# note: if we need to add more paths than what we have now we should probably use a list
# for better maintenance

_PLUGGY_DIR = py.path.local(pluggy.__file__.rstrip("oc"))
# pluggy is either a package or a single module depending on the version
if _PLUGGY_DIR.basename == "__init__.py":
    _PLUGGY_DIR = _PLUGGY_DIR.dirpath()
_PYTEST_DIR = py.path.local(_pytest.__file__).dirpath()
_PY_DIR = py.path.local(py.__file__).dirpath()
```
### 10 - src/_pytest/doctest.py:

Start line: 134, End line: 167

```python
def _is_setup_py(path: py.path.local) -> bool:
    if path.basename != "setup.py":
        return False
    contents = path.read_binary()
    return b"setuptools" in contents or b"distutils" in contents


def _is_doctest(config: Config, path: py.path.local, parent) -> bool:
    if path.ext in (".txt", ".rst") and parent.session.isinitpath(path):
        return True
    globs = config.getoption("doctestglob") or ["test*.txt"]
    for glob in globs:
        if path.check(fnmatch=glob):
            return True
    return False


class ReprFailDoctest(TerminalRepr):
    def __init__(
        self, reprlocation_lines: Sequence[Tuple[ReprFileLocation, Sequence[str]]]
    ) -> None:
        self.reprlocation_lines = reprlocation_lines

    def toterminal(self, tw: TerminalWriter) -> None:
        for reprlocation, lines in self.reprlocation_lines:
            for line in lines:
                tw.line(line)
            reprlocation.toterminal(tw)


class MultipleDoctestFailures(Exception):
    def __init__(self, failures: "Sequence[doctest.DocTestFailure]") -> None:
        super().__init__()
        self.failures = failures
```
### 11 - src/_pytest/_code/code.py:

Start line: 327, End line: 341

```python
class Traceback(List[TracebackEntry]):

    @overload
    def __getitem__(self, key: int) -> TracebackEntry:
        raise NotImplementedError()

    @overload  # noqa: F811
    def __getitem__(self, key: slice) -> "Traceback":  # noqa: F811
        raise NotImplementedError()

    def __getitem__(  # noqa: F811
        self, key: Union[int, slice]
    ) -> Union[TracebackEntry, "Traceback"]:
        if isinstance(key, slice):
            return self.__class__(super().__getitem__(key))
        else:
            return super().__getitem__(key)
```
### 13 - src/_pytest/_code/code.py:

Start line: 926, End line: 957

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ExceptionChainRepr(ExceptionRepr):
    chain = attr.ib(
        type=Sequence[
            Tuple["ReprTraceback", Optional["ReprFileLocation"], Optional[str]]
        ]
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        # reprcrash and reprtraceback of the outermost (the newest) exception
        # in the chain
        self.reprtraceback = self.chain[-1][0]
        self.reprcrash = self.chain[-1][1]

    def toterminal(self, tw: TerminalWriter) -> None:
        for element in self.chain:
            element[0].toterminal(tw)
            if element[2] is not None:
                tw.line("")
                tw.line(element[2], yellow=True)
        super().toterminal(tw)


@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprExceptionInfo(ExceptionRepr):
    reprtraceback = attr.ib(type="ReprTraceback")
    reprcrash = attr.ib(type="ReprFileLocation")

    def toterminal(self, tw: TerminalWriter) -> None:
        self.reprtraceback.toterminal(tw)
        super().toterminal(tw)
```
### 15 - src/_pytest/_code/code.py:

Start line: 1059, End line: 1084

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprEntry(TerminalRepr):

    def toterminal(self, tw: TerminalWriter) -> None:
        if self.style == "short":
            assert self.reprfileloc is not None
            self.reprfileloc.toterminal(tw)
            self._write_entry_lines(tw)
            if self.reprlocals:
                self.reprlocals.toterminal(tw, indent=" " * 8)
            return

        if self.reprfuncargs:
            self.reprfuncargs.toterminal(tw)

        self._write_entry_lines(tw)

        if self.reprlocals:
            tw.line("")
            self.reprlocals.toterminal(tw)
        if self.reprfileloc:
            if self.lines:
                tw.line("")
            self.reprfileloc.toterminal(tw)

    def __str__(self) -> str:
        return "{}\n{}\n{}".format(
            "\n".join(self.lines), self.reprlocals, self.reprfileloc
        )
```
### 23 - src/_pytest/_code/code.py:

Start line: 768, End line: 799

```python
@attr.s
class FormattedExcinfo:

    def _makepath(self, path):
        if not self.abspath:
            try:
                np = py.path.local().bestrelpath(path)
            except OSError:
                return path
            if len(np) < len(str(path)):
                path = np
        return path

    def repr_traceback(self, excinfo: ExceptionInfo) -> "ReprTraceback":
        traceback = excinfo.traceback
        if self.tbfilter:
            traceback = traceback.filter()

        if isinstance(excinfo.value, RecursionError):
            traceback, extraline = self._truncate_recursive_traceback(traceback)
        else:
            extraline = None

        last = traceback[-1]
        entries = []
        if self.style == "value":
            reprentry = self.repr_traceback_entry(last, excinfo)
            entries.append(reprentry)
            return ReprTraceback(entries, None, style=self.style)

        for index, entry in enumerate(traceback):
            einfo = (last == entry) and excinfo or None
            reprentry = self.repr_traceback_entry(entry, einfo)
            entries.append(reprentry)
        return ReprTraceback(entries, extraline, style=self.style)
```
### 26 - src/_pytest/_code/code.py:

Start line: 1, End line: 50

```python
import inspect
import re
import sys
import traceback
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from weakref import ref

import attr
import pluggy
import py

import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import ATTRS_EQ_FIELD
from _pytest.compat import get_real_func
from _pytest.compat import overload
from _pytest.compat import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type
    from typing_extensions import Literal
    from weakref import ReferenceType

    _TracebackStyle = Literal["long", "short", "line", "no", "native", "value", "auto"]
```
### 35 - src/_pytest/_code/code.py:

Start line: 1113, End line: 1133

```python
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ReprFuncArgs(TerminalRepr):
    args = attr.ib(type=Sequence[Tuple[str, object]])

    def toterminal(self, tw: TerminalWriter) -> None:
        if self.args:
            linesofar = ""
            for name, value in self.args:
                ns = "{} = {}".format(name, value)
                if len(ns) + len(linesofar) + 2 > tw.fullwidth:
                    if linesofar:
                        tw.line(linesofar)
                    linesofar = ns
                else:
                    if linesofar:
                        linesofar += ", " + ns
                    else:
                        linesofar = ns
            if linesofar:
                tw.line(linesofar)
            tw.line("")
```
### 44 - src/_pytest/_code/code.py:

Start line: 708, End line: 733

```python
@attr.s
class FormattedExcinfo:

    def repr_locals(self, locals: Mapping[str, object]) -> Optional["ReprLocals"]:
        if self.showlocals:
            lines = []
            keys = [loc for loc in locals if loc[0] != "@"]
            keys.sort()
            for name in keys:
                value = locals[name]
                if name == "__builtins__":
                    lines.append("__builtins__ = <builtins>")
                else:
                    # This formatting could all be handled by the
                    # _repr() function, which is only reprlib.Repr in
                    # disguise, so is very configurable.
                    if self.truncate_locals:
                        str_repr = saferepr(value)
                    else:
                        str_repr = safeformat(value)
                    # if len(str_repr) < 70 or not isinstance(value,
                    #                            (list, tuple, dict)):
                    lines.append("{:<10} = {}".format(name, str_repr))
                    # else:
                    #    self._line("%-10s =\\" % (name,))
                    #    # XXX
                    #    pprint.pprint(value, stream=self.excinfowriter)
            return ReprLocals(lines)
        return None
```
### 54 - src/_pytest/_code/code.py:

Start line: 735, End line: 766

```python
@attr.s
class FormattedExcinfo:

    def repr_traceback_entry(
        self, entry: TracebackEntry, excinfo: Optional[ExceptionInfo] = None
    ) -> "ReprEntry":
        lines = []  # type: List[str]
        style = entry._repr_style if entry._repr_style is not None else self.style
        if style in ("short", "long"):
            source = self._getentrysource(entry)
            if source is None:
                source = Source("???")
                line_index = 0
            else:
                line_index = entry.lineno - entry.getfirstlinesource()
            short = style == "short"
            reprargs = self.repr_args(entry) if not short else None
            s = self.get_source(source, line_index, excinfo, short=short)
            lines.extend(s)
            if short:
                message = "in %s" % (entry.name)
            else:
                message = excinfo and excinfo.typename or ""
            path = self._makepath(entry.path)
            reprfileloc = ReprFileLocation(path, entry.lineno + 1, message)
            localsrepr = self.repr_locals(entry.locals)
            return ReprEntry(lines, reprargs, localsrepr, reprfileloc, style)
        elif style == "value":
            if excinfo:
                lines.extend(str(excinfo.value).split("\n"))
            return ReprEntry(lines, None, None, None, style)
        else:
            if excinfo:
                lines.extend(self.get_exconly(excinfo, indent=4))
            return ReprEntry(lines, None, None, None, style)
```
### 79 - src/_pytest/_code/code.py:

Start line: 907, End line: 923

```python
# This class is abstract -- only subclasses are instantiated.
@attr.s(**{ATTRS_EQ_FIELD: False})  # type: ignore
class ExceptionRepr(TerminalRepr):
    # Provided by in subclasses.
    reprcrash = None  # type: Optional[ReprFileLocation]
    reprtraceback = None  # type: ReprTraceback

    def __attrs_post_init__(self) -> None:
        self.sections = []  # type: List[Tuple[str, str, str]]

    def addsection(self, name: str, content: str, sep: str = "-") -> None:
        self.sections.append((name, content, sep))

    def toterminal(self, tw: TerminalWriter) -> None:
        for name, content, sep in self.sections:
            tw.sep(sep, name)
            tw.line(content)
```
### 86 - src/_pytest/_code/code.py:

Start line: 842, End line: 887

```python
@attr.s
class FormattedExcinfo:

    def repr_excinfo(self, excinfo: ExceptionInfo) -> "ExceptionChainRepr":
        repr_chain = (
            []
        )  # type: List[Tuple[ReprTraceback, Optional[ReprFileLocation], Optional[str]]]
        e = excinfo.value
        excinfo_ = excinfo  # type: Optional[ExceptionInfo]
        descr = None
        seen = set()  # type: Set[int]
        while e is not None and id(e) not in seen:
            seen.add(id(e))
            if excinfo_:
                reprtraceback = self.repr_traceback(excinfo_)
                reprcrash = (
                    excinfo_._getreprcrash() if self.style != "value" else None
                )  # type: Optional[ReprFileLocation]
            else:
                # fallback to native repr if the exception doesn't have a traceback:
                # ExceptionInfo objects require a full traceback to work
                reprtraceback = ReprTracebackNative(
                    traceback.format_exception(type(e), e, None)
                )
                reprcrash = None

            repr_chain += [(reprtraceback, reprcrash, descr)]
            if e.__cause__ is not None and self.chain:
                e = e.__cause__
                excinfo_ = (
                    ExceptionInfo((type(e), e, e.__traceback__))
                    if e.__traceback__
                    else None
                )
                descr = "The above exception was the direct cause of the following exception:"
            elif (
                e.__context__ is not None and not e.__suppress_context__ and self.chain
            ):
                e = e.__context__
                excinfo_ = (
                    ExceptionInfo((type(e), e, e.__traceback__))
                    if e.__traceback__
                    else None
                )
                descr = "During handling of the above exception, another exception occurred:"
            else:
                e = None
        repr_chain.reverse()
        return ExceptionChainRepr(repr_chain)
```
