# pytest-dev__pytest-9646

| **pytest-dev/pytest** | `6aaa017b1e81f6eccc48ee4f6b52d25c49747554` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 5 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/nodes.py b/src/_pytest/nodes.py
--- a/src/_pytest/nodes.py
+++ b/src/_pytest/nodes.py
@@ -656,20 +656,6 @@ class Item(Node):
 
     nextitem = None
 
-    def __init_subclass__(cls) -> None:
-        problems = ", ".join(
-            base.__name__ for base in cls.__bases__ if issubclass(base, Collector)
-        )
-        if problems:
-            warnings.warn(
-                f"{cls.__name__} is an Item subclass and should not be a collector, "
-                f"however its bases {problems} are collectors.\n"
-                "Please split the Collectors and the Item into separate node types.\n"
-                "Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html\n"
-                "example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/",
-                PytestWarning,
-            )
-
     def __init__(
         self,
         name,
@@ -697,6 +683,37 @@ def __init__(
         #: for this test.
         self.user_properties: List[Tuple[str, object]] = []
 
+        self._check_item_and_collector_diamond_inheritance()
+
+    def _check_item_and_collector_diamond_inheritance(self) -> None:
+        """
+        Check if the current type inherits from both File and Collector
+        at the same time, emitting a warning accordingly (#8447).
+        """
+        cls = type(self)
+
+        # We inject an attribute in the type to avoid issuing this warning
+        # for the same class more than once, which is not helpful.
+        # It is a hack, but was deemed acceptable in order to avoid
+        # flooding the user in the common case.
+        attr_name = "_pytest_diamond_inheritance_warning_shown"
+        if getattr(cls, attr_name, False):
+            return
+        setattr(cls, attr_name, True)
+
+        problems = ", ".join(
+            base.__name__ for base in cls.__bases__ if issubclass(base, Collector)
+        )
+        if problems:
+            warnings.warn(
+                f"{cls.__name__} is an Item subclass and should not be a collector, "
+                f"however its bases {problems} are collectors.\n"
+                "Please split the Collectors and the Item into separate node types.\n"
+                "Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html\n"
+                "example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/",
+                PytestWarning,
+            )
+
     def runtest(self) -> None:
         """Run the test case for this item.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/nodes.py | 659 | 672 | - | 5 | -
| src/_pytest/nodes.py | 700 | 700 | - | 5 | -


## Problem Statement

```
Pytest 7 not ignoring warnings as instructed on `pytest.ini`
<!--
Thanks for submitting an issue!

Quick check-list while reporting bugs:
-->

- [x] a detailed description of the bug or problem you are having
- [x] output of `pip list` from the virtual environment you are using
- [x] pytest and operating system versions
- [x] minimal example if possible

## Problem

Hello, with the latest version of Pytest a series of new issues has started to pop-up to warn users (and plugin maintainers) about future changes on how Pytest will work. This is a great work done by Pytest maintainers, and pushes the community towards a better future.

However, after informing plugin maintainers about the upcoming changes, I would like to silence these warnings so I can focus in fixing the errors detected on the tests (otherwise the output is too big and difficult to scroll through to find useful information).

To solve this, I naturally tried to modify my `pytest.ini` file with a `ignore` filter for these warnings, however Pytest is still printing some of them even after I added the warning filters.

### Example for reproduction

The following script can be used to reproduce the problem:

\`\`\`bash
git clone https://github.com/pypa/sampleproject.git
cd sampleproject
sed -i '/^\s*pytest$/a \    pytest-flake8' tox.ini
sed -i '/^\s*pytest$/a \    pytest-black' tox.ini
sed -i 's/py.test tests/py.test tests --black --flake8/' tox.ini
cat << _STR > pytest.ini
[pytest]
filterwarnings=
    # Fail on warnings
    error

    # tholo/pytest-flake8#83
    # shopkeep/pytest-black#55
    # dbader/pytest-mypy#131
    ignore:<class '.*'> is not using a cooperative constructor:pytest.PytestDeprecationWarning
    ignore:The \(fspath. py.path.local\) argument to .* is deprecated.:pytest.PytestDeprecationWarning
    ignore:.* is an Item subclass and should not be a collector.*:pytest.PytestWarning
_STR
tox -e py38
\`\`\`

Relevant output:

\`\`\`
(...)
py38 run-test: commands[3] | py.test tests --black --flake8
/tmp/sampleproject/.tox/py38/lib/python3.8/site-packages/_pytest/nodes.py:664: PytestWarning: BlackItem is an Item subclass and should not be a collector, however its bases File are collectors.
Please split the Collectors and the Item into separate node types.
Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html
example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/
  warnings.warn(
/tmp/sampleproject/.tox/py38/lib/python3.8/site-packages/_pytest/nodes.py:664: PytestWarning: Flake8Item is an Item subclass and should not be a collector, however its bases File are collectors.
Please split the Collectors and the Item into separate node types.
Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html
example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/
  warnings.warn(
(...)
\`\`\`

(The warnings seem to be shown only once per worker per plugin, but considering you have 32 workers with `pytest-xdist` and 4 plugins that haven't adapted to the changes yet, the output can be very verbose and difficult to navigate)

### Expected behaviour

Pytest should not print the warnings:
- `Flake8Item is an Item subclass and should not be a collector...`
- `BlackItem is an Item subclass and should not be a collector...`

### Environment information

\`\`\`bash
$ .tox/py38/bin/python -V
Python 3.8.10


$ .tox/py38/bin/pip list
Package           Version
----------------- -------
attrs             21.4.0
black             22.1.0
build             0.7.0
check-manifest    0.47
click             8.0.3
flake8            4.0.1
iniconfig         1.1.1
mccabe            0.6.1
mypy-extensions   0.4.3
packaging         21.3
pathspec          0.9.0
pep517            0.12.0
peppercorn        0.6
pip               21.3.1
platformdirs      2.4.1
pluggy            1.0.0
py                1.11.0
pycodestyle       2.8.0
pyflakes          2.4.0
pyparsing         3.0.7
pytest            7.0.0
pytest-black      0.3.12
pytest-flake8     1.0.7
sampleproject     2.0.0
setuptools        60.5.0
toml              0.10.2
tomli             2.0.0
typing_extensions 4.0.1
wheel             0.37.1


$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.3 LTS
Release:        20.04
Codename:       focal


$ tox --version
3.24.4 imported from ~/.local/lib/python3.8/site-packages/tox/__init__.py
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/_pytest/deprecated.py | 1 | 80| 732 | 732 | 1131 | 
| 2 | 2 src/pytest/__init__.py | 1 | 78| 710 | 1442 | 2303 | 
| 3 | 3 src/_pytest/warnings.py | 1 | 25| 159 | 1601 | 3207 | 
| 4 | 3 src/_pytest/deprecated.py | 82 | 129| 399 | 2000 | 3207 | 
| 5 | 4 src/_pytest/pytester.py | 1 | 83| 502 | 2502 | 16983 | 
| 6 | **5 src/_pytest/nodes.py** | 1 | 48| 328 | 2830 | 22666 | 
| 7 | 6 testing/python/collect.py | 1278 | 1304| 192 | 3022 | 33383 | 
| 8 | 7 src/_pytest/warning_types.py | 77 | 121| 241 | 3263 | 34112 | 
| 9 | 8 src/_pytest/recwarn.py | 85 | 101| 104 | 3367 | 36378 | 
| 10 | 8 src/_pytest/warnings.py | 28 | 71| 346 | 3713 | 36378 | 
| 11 | 8 src/pytest/__init__.py | 81 | 166| 461 | 4174 | 36378 | 
| 12 | 8 src/_pytest/recwarn.py | 261 | 297| 301 | 4475 | 36378 | 
| 13 | 9 testing/python/fixtures.py | 990 | 1958| 6100 | 10575 | 65192 | 
| 14 | 10 src/_pytest/python.py | 1 | 87| 624 | 11199 | 80320 | 
| 15 | 10 src/_pytest/warning_types.py | 1 | 55| 268 | 11467 | 80320 | 
| 16 | 10 src/_pytest/recwarn.py | 230 | 259| 228 | 11695 | 80320 | 
| 17 | 11 src/_pytest/junitxml.py | 264 | 278| 144 | 11839 | 86036 | 
| 18 | 11 src/_pytest/python.py | 171 | 182| 141 | 11980 | 86036 | 
| 19 | 12 src/_pytest/config/__init__.py | 1665 | 1698| 261 | 12241 | 99121 | 
| 20 | 12 src/_pytest/config/__init__.py | 1513 | 1539| 218 | 12459 | 99121 | 
| 21 | 13 scripts/update-plugin-list.py | 1 | 48| 285 | 12744 | 100086 | 
| 22 | 14 src/_pytest/main.py | 51 | 170| 771 | 13515 | 106962 | 
| 23 | 15 src/_pytest/skipping.py | 46 | 82| 382 | 13897 | 109245 | 
| 24 | 15 testing/python/collect.py | 145 | 186| 265 | 14162 | 109245 | 
| 25 | 16 src/_pytest/doctest.py | 1 | 63| 425 | 14587 | 114962 | 
| 26 | 17 src/_pytest/cacheprovider.py | 327 | 387| 541 | 15128 | 119558 | 
| 27 | 17 src/_pytest/warning_types.py | 58 | 74| 112 | 15240 | 119558 | 
| 28 | 17 scripts/update-plugin-list.py | 51 | 100| 421 | 15661 | 119558 | 
| 29 | 17 src/_pytest/config/__init__.py | 1593 | 1662| 497 | 16158 | 119558 | 
| 30 | 17 src/_pytest/recwarn.py | 1 | 50| 284 | 16442 | 119558 | 
| 31 | 18 src/_pytest/_code/code.py | 1 | 57| 360 | 16802 | 129546 | 
| 32 | 18 testing/python/collect.py | 930 | 956| 173 | 16975 | 129546 | 
| 33 | 18 testing/python/collect.py | 1178 | 1205| 215 | 17190 | 129546 | 
| 34 | 18 src/_pytest/python.py | 226 | 263| 400 | 17590 | 129546 | 
| 35 | 19 doc/en/example/xfail_demo.py | 1 | 39| 143 | 17733 | 129690 | 
| 36 | 19 src/_pytest/recwarn.py | 104 | 155| 488 | 18221 | 129690 | 
| 37 | 19 src/_pytest/config/__init__.py | 630 | 649| 212 | 18433 | 129690 | 
| 38 | 20 src/_pytest/terminal.py | 898 | 950| 435 | 18868 | 140911 | 
| 39 | 20 testing/python/collect.py | 1 | 43| 300 | 19168 | 140911 | 
| 40 | 20 testing/python/collect.py | 1238 | 1275| 221 | 19389 | 140911 | 
| 41 | 20 testing/python/fixtures.py | 41 | 988| 6136 | 25525 | 140911 | 
| 42 | 20 src/_pytest/doctest.py | 310 | 376| 620 | 26145 | 140911 | 
| 43 | 21 doc/en/conf.py | 396 | 416| 196 | 26341 | 144505 | 
| 44 | 21 src/_pytest/warnings.py | 74 | 132| 398 | 26739 | 144505 | 
| 45 | 21 testing/python/fixtures.py | 3841 | 4471| 4315 | 31054 | 144505 | 
| 46 | 21 testing/python/collect.py | 1363 | 1380| 125 | 31179 | 144505 | 
| 47 | 21 src/_pytest/main.py | 375 | 401| 253 | 31432 | 144505 | 
| 48 | 22 src/_pytest/faulthandler.py | 1 | 32| 225 | 31657 | 145247 | 
| 49 | 23 src/_pytest/config/argparsing.py | 1 | 30| 169 | 31826 | 149780 | 
| 50 | 23 src/_pytest/python.py | 358 | 374| 142 | 31968 | 149780 | 
| 51 | 23 src/_pytest/doctest.py | 286 | 296| 128 | 32096 | 149780 | 
| 52 | 23 src/_pytest/config/__init__.py | 1325 | 1358| 288 | 32384 | 149780 | 
| 53 | 24 src/_pytest/logging.py | 707 | 745| 361 | 32745 | 156239 | 
| 54 | 25 src/_pytest/runner.py | 159 | 182| 184 | 32929 | 160537 | 
| 55 | 25 testing/python/collect.py | 1208 | 1235| 188 | 33117 | 160537 | 
| 56 | 25 src/_pytest/recwarn.py | 213 | 227| 131 | 33248 | 160537 | 
| 57 | 26 src/_pytest/legacypath.py | 107 | 179| 765 | 34013 | 164545 | 
| 58 | 27 src/_pytest/pathlib.py | 1 | 63| 398 | 34411 | 170131 | 
| 59 | 27 src/_pytest/doctest.py | 494 | 535| 349 | 34760 | 170131 | 
| 60 | 27 doc/en/conf.py | 116 | 221| 855 | 35615 | 170131 | 
| 61 | 27 src/_pytest/pytester.py | 164 | 186| 243 | 35858 | 170131 | 
| 62 | 27 src/_pytest/pytester.py | 655 | 741| 717 | 36575 | 170131 | 
| 63 | 27 src/_pytest/skipping.py | 245 | 259| 152 | 36727 | 170131 | 
| 64 | 27 src/_pytest/config/__init__.py | 1254 | 1288| 252 | 36979 | 170131 | 
| 65 | 27 src/_pytest/config/__init__.py | 1290 | 1323| 315 | 37294 | 170131 | 
| 66 | 27 src/_pytest/skipping.py | 262 | 297| 325 | 37619 | 170131 | 
| 67 | 27 src/_pytest/config/__init__.py | 105 | 129| 204 | 37823 | 170131 | 
| 68 | 27 src/_pytest/_code/code.py | 1236 | 1245| 118 | 37941 | 170131 | 
| 69 | 27 src/_pytest/config/__init__.py | 1 | 79| 475 | 38416 | 170131 | 
| 70 | 27 testing/python/collect.py | 210 | 263| 342 | 38758 | 170131 | 
| 71 | 28 src/_pytest/reports.py | 1 | 57| 385 | 39143 | 174565 | 
| 72 | 28 src/_pytest/recwarn.py | 158 | 211| 454 | 39597 | 174565 | 
| 73 | 29 src/_pytest/helpconfig.py | 158 | 228| 576 | 40173 | 176435 | 
| 74 | 29 src/_pytest/skipping.py | 1 | 24| 145 | 40318 | 176435 | 
| 75 | 29 doc/en/conf.py | 385 | 393| 132 | 40450 | 176435 | 
| 76 | 29 src/_pytest/recwarn.py | 53 | 82| 291 | 40741 | 176435 | 
| 77 | **29 src/_pytest/nodes.py** | 274 | 309| 251 | 40992 | 176435 | 
| 78 | 29 testing/python/fixtures.py | 2834 | 3839| 6155 | 47147 | 176435 | 
| 79 | 30 src/_pytest/outcomes.py | 51 | 71| 175 | 47322 | 178743 | 
| 80 | 30 src/_pytest/terminal.py | 112 | 224| 781 | 48103 | 178743 | 
| 81 | 30 src/_pytest/terminal.py | 443 | 507| 554 | 48657 | 178743 | 
| 82 | 30 testing/python/collect.py | 882 | 895| 128 | 48785 | 178743 | 
| 83 | 30 src/_pytest/faulthandler.py | 67 | 98| 219 | 49004 | 178743 | 
| 84 | 30 src/_pytest/doctest.py | 298 | 309| 123 | 49127 | 178743 | 
| 85 | 31 src/_pytest/unittest.py | 369 | 415| 347 | 49474 | 181837 | 
| 86 | 31 src/_pytest/terminal.py | 264 | 306| 336 | 49810 | 181837 | 
| 87 | 32 testing/example_scripts/issue_519.py | 1 | 33| 362 | 50172 | 182311 | 
| 88 | 33 testing/python/metafunc.py | 1458 | 1482| 214 | 50386 | 197208 | 
| 89 | 33 src/_pytest/main.py | 171 | 227| 400 | 50786 | 197208 | 
| 90 | 33 testing/python/metafunc.py | 1 | 28| 142 | 50928 | 197208 | 
| 91 | 33 testing/python/collect.py | 959 | 991| 300 | 51228 | 197208 | 
| 92 | 33 src/_pytest/cacheprovider.py | 261 | 279| 171 | 51399 | 197208 | 
| 93 | 33 src/_pytest/config/__init__.py | 680 | 704| 254 | 51653 | 197208 | 
| 94 | 33 testing/python/collect.py | 1056 | 1082| 178 | 51831 | 197208 | 
| 95 | 33 testing/python/collect.py | 1035 | 1054| 184 | 52015 | 197208 | 
| 96 | 33 src/_pytest/doctest.py | 139 | 173| 292 | 52307 | 197208 | 
| 97 | 33 testing/python/collect.py | 1139 | 1154| 139 | 52446 | 197208 | 
| 98 | 33 src/_pytest/cacheprovider.py | 86 | 120| 277 | 52723 | 197208 | 
| 99 | 34 src/_pytest/stepwise.py | 92 | 123| 263 | 52986 | 198123 | 
| 100 | 34 testing/python/collect.py | 994 | 1033| 364 | 53350 | 198123 | 
| 101 | 34 src/_pytest/python.py | 1585 | 1642| 437 | 53787 | 198123 | 
| 102 | 34 src/_pytest/doctest.py | 584 | 669| 786 | 54573 | 198123 | 
| 103 | 35 scripts/towncrier-draft-to-file.py | 1 | 16| 105 | 54678 | 198228 | 
| 104 | 35 testing/python/collect.py | 92 | 127| 253 | 54931 | 198228 | 
| 105 | 35 src/_pytest/config/__init__.py | 450 | 471| 167 | 55098 | 198228 | 
| 106 | 35 testing/python/metafunc.py | 1440 | 1456| 110 | 55208 | 198228 | 
| 107 | 35 testing/python/collect.py | 1383 | 1400| 136 | 55344 | 198228 | 
| 108 | 35 src/_pytest/terminal.py | 1 | 72| 386 | 55730 | 198228 | 
| 109 | 35 src/_pytest/stepwise.py | 55 | 90| 308 | 56038 | 198228 | 
| 110 | 35 src/_pytest/python.py | 1557 | 1582| 258 | 56296 | 198228 | 
| 111 | 36 extra/get_issues.py | 55 | 86| 231 | 56527 | 198782 | 
| 112 | 37 src/_pytest/config/compat.py | 1 | 18| 141 | 56668 | 199298 | 
| 113 | 37 src/_pytest/python.py | 1529 | 1555| 238 | 56906 | 199298 | 
| 114 | 37 testing/python/collect.py | 848 | 880| 222 | 57128 | 199298 | 


## Patch

```diff
diff --git a/src/_pytest/nodes.py b/src/_pytest/nodes.py
--- a/src/_pytest/nodes.py
+++ b/src/_pytest/nodes.py
@@ -656,20 +656,6 @@ class Item(Node):
 
     nextitem = None
 
-    def __init_subclass__(cls) -> None:
-        problems = ", ".join(
-            base.__name__ for base in cls.__bases__ if issubclass(base, Collector)
-        )
-        if problems:
-            warnings.warn(
-                f"{cls.__name__} is an Item subclass and should not be a collector, "
-                f"however its bases {problems} are collectors.\n"
-                "Please split the Collectors and the Item into separate node types.\n"
-                "Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html\n"
-                "example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/",
-                PytestWarning,
-            )
-
     def __init__(
         self,
         name,
@@ -697,6 +683,37 @@ def __init__(
         #: for this test.
         self.user_properties: List[Tuple[str, object]] = []
 
+        self._check_item_and_collector_diamond_inheritance()
+
+    def _check_item_and_collector_diamond_inheritance(self) -> None:
+        """
+        Check if the current type inherits from both File and Collector
+        at the same time, emitting a warning accordingly (#8447).
+        """
+        cls = type(self)
+
+        # We inject an attribute in the type to avoid issuing this warning
+        # for the same class more than once, which is not helpful.
+        # It is a hack, but was deemed acceptable in order to avoid
+        # flooding the user in the common case.
+        attr_name = "_pytest_diamond_inheritance_warning_shown"
+        if getattr(cls, attr_name, False):
+            return
+        setattr(cls, attr_name, True)
+
+        problems = ", ".join(
+            base.__name__ for base in cls.__bases__ if issubclass(base, Collector)
+        )
+        if problems:
+            warnings.warn(
+                f"{cls.__name__} is an Item subclass and should not be a collector, "
+                f"however its bases {problems} are collectors.\n"
+                "Please split the Collectors and the Item into separate node types.\n"
+                "Pytest Doc example: https://docs.pytest.org/en/latest/example/nonpython.html\n"
+                "example pull request on a plugin: https://github.com/asmeurer/pytest-flakes/pull/40/",
+                PytestWarning,
+            )
+
     def runtest(self) -> None:
         """Run the test case for this item.
 

```

## Test Patch

```diff
diff --git a/testing/test_nodes.py b/testing/test_nodes.py
--- a/testing/test_nodes.py
+++ b/testing/test_nodes.py
@@ -1,3 +1,5 @@
+import re
+import warnings
 from pathlib import Path
 from typing import cast
 from typing import List
@@ -58,30 +60,31 @@ def test_subclassing_both_item_and_collector_deprecated(
     request, tmp_path: Path
 ) -> None:
     """
-    Verifies we warn on diamond inheritance
-    as well as correctly managing legacy inheritance ctors with missing args
-    as found in plugins
+    Verifies we warn on diamond inheritance as well as correctly managing legacy
+    inheritance constructors with missing args as found in plugins.
     """
 
-    with pytest.warns(
-        PytestWarning,
-        match=(
-            "(?m)SoWrong is an Item subclass and should not be a collector, however its bases File are collectors.\n"
-            "Please split the Collectors and the Item into separate node types.\n.*"
-        ),
-    ):
+    # We do not expect any warnings messages to issued during class definition.
+    with warnings.catch_warnings():
+        warnings.simplefilter("error")
 
         class SoWrong(nodes.Item, nodes.File):
             def __init__(self, fspath, parent):
                 """Legacy ctor with legacy call # don't wana see"""
                 super().__init__(fspath, parent)
 
-    with pytest.warns(
-        PytestWarning, match=".*SoWrong.* not using a cooperative constructor.*"
-    ):
+    with pytest.warns(PytestWarning) as rec:
         SoWrong.from_parent(
             request.session, fspath=legacy_path(tmp_path / "broken.txt")
         )
+    messages = [str(x.message) for x in rec]
+    assert any(
+        re.search(".*SoWrong.* not using a cooperative constructor.*", x)
+        for x in messages
+    )
+    assert any(
+        re.search("(?m)SoWrong .* should not be a collector", x) for x in messages
+    )
 
 
 @pytest.mark.parametrize(

```


## Code snippets

### 1 - src/_pytest/deprecated.py:

Start line: 1, End line: 80

```python
"""Deprecation messages and bits of code used elsewhere in the codebase that
is planned to be removed in the next pytest release.

Keeping it in a central location makes it easy to track what is deprecated and should
be removed when the time comes.

All constants defined in this module should be either instances of
:class:`PytestWarning`, or :class:`UnformattedWarning`
in case of warnings which need to format their messages.
"""
from warnings import warn

from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestRemovedIn8Warning
from _pytest.warning_types import UnformattedWarning

# set of plugins which have been integrated into the core; we use this list to ignore
# them during registration to avoid conflicts
DEPRECATED_EXTERNAL_PLUGINS = {
    "pytest_catchlog",
    "pytest_capturelog",
    "pytest_faulthandler",
}


# This can be* removed pytest 8, but it's harmless and common, so no rush to remove.
# * If you're in the future: "could have been".
YIELD_FIXTURE = PytestDeprecationWarning(
    "@pytest.yield_fixture is deprecated.\n"
    "Use @pytest.fixture instead; they are the same."
)

WARNING_CMDLINE_PREPARSE_HOOK = PytestRemovedIn8Warning(
    "The pytest_cmdline_preparse hook is deprecated and will be removed in a future release. \n"
    "Please use pytest_load_initial_conftests hook instead."
)

FSCOLLECTOR_GETHOOKPROXY_ISINITPATH = PytestRemovedIn8Warning(
    "The gethookproxy() and isinitpath() methods of FSCollector and Package are deprecated; "
    "use self.session.gethookproxy() and self.session.isinitpath() instead. "
)

STRICT_OPTION = PytestRemovedIn8Warning(
    "The --strict option is deprecated, use --strict-markers instead."
)

# This deprecation is never really meant to be removed.
PRIVATE = PytestDeprecationWarning("A private pytest class or function was used.")

UNITTEST_SKIP_DURING_COLLECTION = PytestRemovedIn8Warning(
    "Raising unittest.SkipTest to skip tests during collection is deprecated. "
    "Use pytest.skip() instead."
)

ARGUMENT_PERCENT_DEFAULT = PytestRemovedIn8Warning(
    'pytest now uses argparse. "%default" should be changed to "%(default)s"',
)

ARGUMENT_TYPE_STR_CHOICE = UnformattedWarning(
    PytestRemovedIn8Warning,
    "`type` argument to addoption() is the string {typ!r}."
    " For choices this is optional and can be omitted, "
    " but when supplied should be a type (for example `str` or `int`)."
    " (options: {names})",
)

ARGUMENT_TYPE_STR = UnformattedWarning(
    PytestRemovedIn8Warning,
    "`type` argument to addoption() is the string {typ!r}, "
    " but when supplied should be a type (for example `str` or `int`)."
    " (options: {names})",
)


HOOK_LEGACY_PATH_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "The ({pylib_path_arg}: py.path.local) argument is deprecated, please use ({pathlib_path_arg}: pathlib.Path)\n"
    "see https://docs.pytest.org/en/latest/deprecations.html"
    "#py-path-local-arguments-for-hooks-replaced-with-pathlib-path",
)
```
### 2 - src/pytest/__init__.py:

Start line: 1, End line: 78

```python
# PYTHON_ARGCOMPLETE_OK
"""pytest: unit and functional testing with Python."""
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
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.legacypath import TempdirFactory
from _pytest.legacypath import Testdir
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
from _pytest.warning_types import PytestRemovedIn8Warning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnhandledThreadExceptionWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestUnraisableExceptionWarning
from _pytest.warning_types import PytestWarning

set_trace = __pytestPDB.set_trace
```
### 3 - src/_pytest/warnings.py:

Start line: 1, End line: 25

```python
import sys
import warnings
from contextlib import contextmanager
from typing import Generator
from typing import Optional
from typing import TYPE_CHECKING

import pytest
from _pytest.config import apply_warning_filters
from _pytest.config import Config
from _pytest.config import parse_warning_filter
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.terminal import TerminalReporter

if TYPE_CHECKING:
    from typing_extensions import Literal


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "filterwarnings(warning): add a warning filter to the given test. "
        "see https://docs.pytest.org/en/stable/how-to/capture-warnings.html#pytest-mark-filterwarnings ",
    )
```
### 4 - src/_pytest/deprecated.py:

Start line: 82, End line: 129

```python
NODE_CTOR_FSPATH_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "The (fspath: py.path.local) argument to {node_type_name} is deprecated. "
    "Please use the (path: pathlib.Path) argument instead.\n"
    "See https://docs.pytest.org/en/latest/deprecations.html"
    "#fspath-argument-for-node-constructors-replaced-with-pathlib-path",
)

WARNS_NONE_ARG = PytestRemovedIn8Warning(
    "Passing None has been deprecated.\n"
    "See https://docs.pytest.org/en/latest/how-to/capture-warnings.html"
    "#additional-use-cases-of-warnings-in-tests"
    " for alternatives in common use cases."
)

KEYWORD_MSG_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "pytest.{func}(msg=...) is now deprecated, use pytest.{func}(reason=...) instead",
)

INSTANCE_COLLECTOR = PytestRemovedIn8Warning(
    "The pytest.Instance collector type is deprecated and is no longer used. "
    "See https://docs.pytest.org/en/latest/deprecations.html#the-pytest-instance-collector",
)

# You want to make some `__init__` or function "private".
#
#   def my_private_function(some, args):
#       ...
#
# Do this:
#
#   def my_private_function(some, args, *, _ispytest: bool = False):
#       check_ispytest(_ispytest)
#       ...
#
# Change all internal/allowed calls to
#
#   my_private_function(some, args, _ispytest=True)
#
# All other calls will get the default _ispytest=False and trigger
# the warning (possibly error in the future).


def check_ispytest(ispytest: bool) -> None:
    if not ispytest:
        warn(PRIVATE, stacklevel=3)
```
### 5 - src/_pytest/pytester.py:

Start line: 1, End line: 83

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

from iniconfig import IniConfig
from iniconfig import SectionWrapper

from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import final
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
### 6 - src/_pytest/nodes.py:

Start line: 1, End line: 48

```python
import os
import warnings
from inspect import signature
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import _pytest._code
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest.compat import cached_property
from _pytest.compat import LEGACY_PATH
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.deprecated import FSCOLLECTOR_GETHOOKPROXY_ISINITPATH
from _pytest.deprecated import NODE_CTOR_FSPATH_ARG
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import NodeKeywords
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.stash import Stash
from _pytest.warning_types import PytestWarning

if TYPE_CHECKING:
    # Imported here due to circular import.
    from _pytest.main import Session
    from _pytest._code.code import _TracebackStyle


SEP = "/"

tracebackcutdir = Path(_pytest.__file__).parent
```
### 7 - testing/python/collect.py:

Start line: 1278, End line: 1304

```python
@pytest.mark.filterwarnings("default::pytest.PytestCollectionWarning")
def test_dont_collect_non_function_callable(pytester: Pytester) -> None:
    """Test for issue https://github.com/pytest-dev/pytest/issues/331

    In this case an INTERNALERROR occurred trying to report the failure of
    a test like this one because pytest failed to get the source lines.
    """
    pytester.makepyfile(
        """
        class Oh(object):
            def __call__(self):
                pass

        test_a = Oh()

        def test_real():
            pass
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "*collected 1 item*",
            "*test_dont_collect_non_function_callable.py:2: *cannot collect 'test_a' because it is not a function*",
            "*1 passed, 1 warning in *",
        ]
    )
```
### 8 - src/_pytest/warning_types.py:

Start line: 77, End line: 121

```python
@final
class PytestUnhandledCoroutineWarning(PytestWarning):
    """Warning emitted for an unhandled coroutine.

    A coroutine was encountered when collecting test functions, but was not
    handled by any async-aware plugin.
    Coroutine test functions are not natively supported.
    """

    __module__ = "pytest"


@final
class PytestUnknownMarkWarning(PytestWarning):
    """Warning emitted on use of unknown markers.

    See :ref:`mark` for details.
    """

    __module__ = "pytest"


@final
class PytestUnraisableExceptionWarning(PytestWarning):
    """An unraisable exception was reported.

    Unraisable exceptions are exceptions raised in :meth:`__del__ <object.__del__>`
    implementations and similar situations when the exception cannot be raised
    as normal.
    """

    __module__ = "pytest"


@final
class PytestUnhandledThreadExceptionWarning(PytestWarning):
    """An unhandled exception occurred in a :class:`~threading.Thread`.

    Such exceptions don't propagate normally.
    """

    __module__ = "pytest"


_W = TypeVar("_W", bound=PytestWarning)
```
### 9 - src/_pytest/recwarn.py:

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
### 10 - src/_pytest/warnings.py:

Start line: 28, End line: 71

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
            ihook.pytest_warning_recorded.call_historic(
                kwargs=dict(
                    warning_message=warning_message,
                    nodeid=nodeid,
                    when=when,
                    location=None,
                )
            )
```
### 77 - src/_pytest/nodes.py:

Start line: 274, End line: 309

```python
class Node(metaclass=NodeMeta):

    def warn(self, warning: Warning) -> None:
        """Issue a warning for this Node.

        Warnings will be displayed after the test session, unless explicitly suppressed.

        :param Warning warning:
            The warning instance to issue.

        :raises ValueError: If ``warning`` instance is not a subclass of Warning.

        Example usage:

        .. code-block:: python

            node.warn(PytestWarning("some message"))
            node.warn(UserWarning("some message"))

        .. versionchanged:: 6.2
            Any subclass of :class:`Warning` is now accepted, rather than only
            :class:`PytestWarning <pytest.PytestWarning>` subclasses.
        """
        # enforce type checks here to avoid getting a generic type error later otherwise.
        if not isinstance(warning, Warning):
            raise ValueError(
                "warning must be an instance of Warning or subclass, got {!r}".format(
                    warning
                )
            )
        path, lineno = get_fslocation_from_item(self)
        assert lineno is not None
        warnings.warn_explicit(
            warning,
            category=None,
            filename=str(path),
            lineno=lineno + 1,
        )
```
