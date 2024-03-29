# pytest-dev__pytest-5840

| **pytest-dev/pytest** | `73c5b7f4b11a81e971f7d1bb18072e06a87060f4` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 26555 |
| **Any found context length** | 798 |
| **Avg pos** | 145.5 |
| **Min pos** | 4 |
| **Max pos** | 138 |
| **Top file pos** | 2 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/config/__init__.py b/src/_pytest/config/__init__.py
--- a/src/_pytest/config/__init__.py
+++ b/src/_pytest/config/__init__.py
@@ -30,7 +30,6 @@
 from _pytest.compat import importlib_metadata
 from _pytest.outcomes import fail
 from _pytest.outcomes import Skipped
-from _pytest.pathlib import unique_path
 from _pytest.warning_types import PytestConfigWarning
 
 hookimpl = HookimplMarker("pytest")
@@ -367,7 +366,7 @@ def _set_initial_conftests(self, namespace):
         """
         current = py.path.local()
         self._confcutdir = (
-            unique_path(current.join(namespace.confcutdir, abs=True))
+            current.join(namespace.confcutdir, abs=True)
             if namespace.confcutdir
             else None
         )
@@ -406,13 +405,11 @@ def _getconftestmodules(self, path):
         else:
             directory = path
 
-        directory = unique_path(directory)
-
         # XXX these days we may rather want to use config.rootdir
         # and allow users to opt into looking into the rootdir parent
         # directories instead of requiring to specify confcutdir
         clist = []
-        for parent in directory.parts():
+        for parent in directory.realpath().parts():
             if self._confcutdir and self._confcutdir.relto(parent):
                 continue
             conftestpath = parent.join("conftest.py")
@@ -432,12 +429,14 @@ def _rget_with_confmod(self, name, path):
         raise KeyError(name)
 
     def _importconftest(self, conftestpath):
-        # Use realpath to avoid loading the same conftest twice
+        # Use a resolved Path object as key to avoid loading the same conftest twice
         # with build systems that create build directories containing
         # symlinks to actual files.
-        conftestpath = unique_path(conftestpath)
+        # Using Path().resolve() is better than py.path.realpath because
+        # it resolves to the correct path/drive in case-insensitive file systems (#5792)
+        key = Path(str(conftestpath)).resolve()
         try:
-            return self._conftestpath2mod[conftestpath]
+            return self._conftestpath2mod[key]
         except KeyError:
             pkgpath = conftestpath.pypkgpath()
             if pkgpath is None:
@@ -454,7 +453,7 @@ def _importconftest(self, conftestpath):
                 raise ConftestImportFailure(conftestpath, sys.exc_info())
 
             self._conftest_plugins.add(mod)
-            self._conftestpath2mod[conftestpath] = mod
+            self._conftestpath2mod[key] = mod
             dirpath = conftestpath.dirpath()
             if dirpath in self._dirpath2confmods:
                 for path, mods in self._dirpath2confmods.items():
diff --git a/src/_pytest/pathlib.py b/src/_pytest/pathlib.py
--- a/src/_pytest/pathlib.py
+++ b/src/_pytest/pathlib.py
@@ -11,7 +11,6 @@
 from os.path import expanduser
 from os.path import expandvars
 from os.path import isabs
-from os.path import normcase
 from os.path import sep
 from posixpath import sep as posix_sep
 
@@ -335,12 +334,3 @@ def fnmatch_ex(pattern, path):
 def parts(s):
     parts = s.split(sep)
     return {sep.join(parts[: i + 1]) or sep for i in range(len(parts))}
-
-
-def unique_path(path):
-    """Returns a unique path in case-insensitive (but case-preserving) file
-    systems such as Windows.
-
-    This is needed only for ``py.path.local``; ``pathlib.Path`` handles this
-    natively with ``resolve()``."""
-    return type(path)(normcase(str(path.realpath())))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/config/__init__.py | 33 | 33 | 11 | 2 | 2594
| src/_pytest/config/__init__.py | 370 | 370 | 99 | 2 | 26555
| src/_pytest/config/__init__.py | 409 | 415 | 35 | 2 | 6127
| src/_pytest/config/__init__.py | 435 | 440 | 4 | 2 | 798
| src/_pytest/config/__init__.py | 457 | 457 | 4 | 2 | 798
| src/_pytest/pathlib.py | 14 | 14 | 138 | 50 | 40778
| src/_pytest/pathlib.py | 338 | 346 | - | 50 | -


## Problem Statement

```
5.1.2 ImportError while loading conftest (windows import folder casing issues)
5.1.1 works fine. after upgrade to 5.1.2, the path was converted to lower case
\`\`\`
Installing collected packages: pytest
  Found existing installation: pytest 5.1.1
    Uninstalling pytest-5.1.1:
      Successfully uninstalled pytest-5.1.1
Successfully installed pytest-5.1.2
PS C:\Azure\KMS\ComponentTest\Python> pytest --collect-only .\PIsys -m smoke
ImportError while loading conftest 'c:\azure\kms\componenttest\python\pisys\conftest.py'.
ModuleNotFoundError: No module named 'python'
PS C:\Azure\KMS\ComponentTest\Python>
\`\`\`



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 testing/python/collect.py | 713 | 728| 143 | 143 | 9277 | 
| 2 | 1 testing/python/collect.py | 730 | 753| 170 | 313 | 9277 | 
| 3 | **2 src/_pytest/config/__init__.py** | 216 | 226| 154 | 467 | 17857 | 
| **-> 4 <-** | **2 src/_pytest/config/__init__.py** | 434 | 466| 331 | 798 | 17857 | 
| 5 | 3 src/pytest.py | 1 | 107| 708 | 1506 | 18565 | 
| 6 | 3 testing/python/collect.py | 1 | 33| 225 | 1731 | 18565 | 
| 7 | 3 testing/python/collect.py | 788 | 810| 195 | 1926 | 18565 | 
| 8 | 4 testing/example_scripts/config/collect_pytest_prefix/conftest.py | 1 | 3| 0 | 1926 | 18573 | 
| 9 | 4 testing/python/collect.py | 59 | 77| 189 | 2115 | 18573 | 
| 10 | 4 testing/python/collect.py | 755 | 786| 204 | 2319 | 18573 | 
| **-> 11 <-** | **4 src/_pytest/config/__init__.py** | 1 | 44| 275 | 2594 | 18573 | 
| 12 | 4 testing/python/collect.py | 35 | 57| 187 | 2781 | 18573 | 
| 13 | **4 src/_pytest/config/__init__.py** | 995 | 1007| 120 | 2901 | 18573 | 
| 14 | 4 testing/python/collect.py | 846 | 878| 304 | 3205 | 18573 | 
| 15 | 4 testing/python/collect.py | 920 | 939| 178 | 3383 | 18573 | 
| 16 | 5 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_conftest/pkg/conftest.py | 1 | 7| 0 | 3383 | 18591 | 
| 17 | 6 src/_pytest/capture.py | 38 | 59| 193 | 3576 | 24628 | 
| 18 | 7 doc/en/example/py2py3/conftest.py | 1 | 17| 0 | 3576 | 24714 | 
| 19 | 8 setup.py | 1 | 16| 147 | 3723 | 24976 | 
| 20 | 9 doc/en/example/costlysetup/conftest.py | 1 | 21| 0 | 3723 | 25055 | 
| 21 | 10 testing/example_scripts/conftest_usageerror/conftest.py | 1 | 9| 0 | 3723 | 25086 | 
| 22 | 11 src/_pytest/doctest.py | 413 | 435| 208 | 3931 | 29586 | 
| 23 | **11 src/_pytest/config/__init__.py** | 854 | 901| 421 | 4352 | 29586 | 
| 24 | 12 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_module/conftest.py | 1 | 7| 0 | 4352 | 29600 | 
| 25 | 13 src/_pytest/python.py | 497 | 545| 468 | 4820 | 41001 | 
| 26 | 14 doc/en/example/assertion/global_testmodule_config/conftest.py | 1 | 15| 0 | 4820 | 41082 | 
| 27 | 14 testing/python/collect.py | 196 | 249| 320 | 5140 | 41082 | 
| 28 | 15 doc/en/conftest.py | 1 | 2| 0 | 5140 | 41089 | 
| 29 | 16 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 8| 0 | 5140 | 41115 | 
| 30 | 16 testing/python/collect.py | 1076 | 1103| 183 | 5323 | 41115 | 
| 31 | 17 doc/en/example/conftest.py | 1 | 2| 0 | 5323 | 41122 | 
| 32 | **17 src/_pytest/config/__init__.py** | 468 | 529| 471 | 5794 | 41122 | 
| 33 | 18 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 5794 | 41175 | 
| 34 | 19 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_conftest/conftest.py | 1 | 7| 0 | 5794 | 41189 | 
| **-> 35 <-** | **19 src/_pytest/config/__init__.py** | 391 | 432| 333 | 6127 | 41189 | 
| 36 | 20 src/_pytest/main.py | 42 | 150| 757 | 6884 | 46476 | 
| 37 | 21 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub2/conftest.py | 1 | 7| 0 | 6884 | 46501 | 
| 38 | 21 src/_pytest/main.py | 309 | 360| 303 | 7187 | 46501 | 
| 39 | 21 src/_pytest/main.py | 614 | 628| 132 | 7319 | 46501 | 
| 40 | 21 testing/python/collect.py | 1266 | 1303| 320 | 7639 | 46501 | 
| 41 | 22 src/_pytest/pytester.py | 1 | 29| 170 | 7809 | 56828 | 
| 42 | 22 testing/python/collect.py | 1306 | 1339| 239 | 8048 | 56828 | 
| 43 | 22 testing/python/collect.py | 1106 | 1143| 209 | 8257 | 56828 | 
| 44 | 23 testing/example_scripts/fixtures/custom_item/conftest.py | 1 | 11| 0 | 8257 | 56866 | 
| 45 | 24 src/_pytest/faulthandler.py | 39 | 51| 121 | 8378 | 57441 | 
| 46 | 24 src/_pytest/python.py | 1 | 53| 373 | 8751 | 57441 | 
| 47 | 24 src/_pytest/python.py | 132 | 149| 212 | 8963 | 57441 | 
| 48 | **24 src/_pytest/config/__init__.py** | 342 | 359| 154 | 9117 | 57441 | 
| 49 | 25 doc/en/example/assertion/failure_demo.py | 191 | 203| 114 | 9231 | 59100 | 
| 50 | 26 src/_pytest/compat.py | 1 | 46| 225 | 9456 | 61568 | 
| 51 | 26 testing/python/collect.py | 1205 | 1230| 186 | 9642 | 61568 | 
| 52 | 26 src/_pytest/python.py | 1204 | 1230| 208 | 9850 | 61568 | 
| 53 | 27 testing/python/setup_only.py | 203 | 226| 129 | 9979 | 63101 | 
| 54 | 27 testing/python/collect.py | 881 | 918| 321 | 10300 | 63101 | 
| 55 | **27 src/_pytest/config/__init__.py** | 903 | 943| 310 | 10610 | 63101 | 
| 56 | 27 setup.py | 19 | 40| 115 | 10725 | 63101 | 
| 57 | **27 src/_pytest/config/__init__.py** | 599 | 623| 141 | 10866 | 63101 | 
| 58 | 27 testing/python/setup_only.py | 62 | 93| 190 | 11056 | 63101 | 
| 59 | 28 src/_pytest/skipping.py | 28 | 65| 363 | 11419 | 64579 | 
| 60 | 29 testing/python/integration.py | 325 | 362| 224 | 11643 | 67504 | 
| 61 | 30 src/_pytest/junitxml.py | 446 | 465| 143 | 11786 | 72559 | 
| 62 | 30 src/_pytest/faulthandler.py | 1 | 36| 249 | 12035 | 72559 | 
| 63 | 31 src/_pytest/debugging.py | 47 | 69| 182 | 12217 | 74969 | 
| 64 | 31 testing/python/integration.py | 37 | 68| 226 | 12443 | 74969 | 
| 65 | 31 src/_pytest/doctest.py | 87 | 128| 311 | 12754 | 74969 | 
| 66 | 31 src/_pytest/python.py | 548 | 573| 238 | 12992 | 74969 | 
| 67 | 32 testing/python/fixtures.py | 2193 | 3119| 6102 | 19094 | 100640 | 
| 68 | 32 src/_pytest/python.py | 111 | 129| 176 | 19270 | 100640 | 
| 69 | 32 testing/python/fixtures.py | 1 | 82| 492 | 19762 | 100640 | 
| 70 | 33 src/_pytest/__init__.py | 1 | 9| 0 | 19762 | 100696 | 
| 71 | 33 src/_pytest/compat.py | 147 | 185| 244 | 20006 | 100696 | 
| 72 | 34 doc/en/example/costlysetup/sub_b/__init__.py | 1 | 2| 0 | 20006 | 100697 | 
| 73 | 34 testing/python/collect.py | 559 | 658| 662 | 20668 | 100697 | 
| 74 | 34 testing/python/fixtures.py | 2074 | 2191| 714 | 21382 | 100697 | 
| 75 | 34 testing/python/collect.py | 279 | 344| 471 | 21853 | 100697 | 
| 76 | 34 src/_pytest/python.py | 575 | 600| 263 | 22116 | 100697 | 
| 77 | 34 src/_pytest/python.py | 630 | 658| 246 | 22362 | 100697 | 
| 78 | 35 src/_pytest/_code/__init__.py | 1 | 11| 0 | 22362 | 100798 | 
| 79 | 35 testing/python/collect.py | 812 | 843| 264 | 22626 | 100798 | 
| 80 | 35 src/_pytest/python.py | 56 | 108| 354 | 22980 | 100798 | 
| 81 | 35 testing/python/collect.py | 131 | 172| 251 | 23231 | 100798 | 
| 82 | 35 testing/python/collect.py | 941 | 966| 170 | 23401 | 100798 | 
| 83 | 36 doc/en/example/nonpython/conftest.py | 1 | 47| 314 | 23715 | 101112 | 
| 84 | 36 testing/python/setup_only.py | 123 | 153| 183 | 23898 | 101112 | 
| 85 | 36 src/_pytest/python.py | 602 | 628| 215 | 24113 | 101112 | 
| 86 | 37 extra/setup-py.test/setup.py | 1 | 12| 0 | 24113 | 101190 | 
| 87 | 37 testing/python/setup_only.py | 229 | 248| 122 | 24235 | 101190 | 
| 88 | 38 doc/en/example/costlysetup/sub_a/__init__.py | 1 | 2| 0 | 24235 | 101191 | 
| 89 | 38 src/_pytest/python.py | 1262 | 1331| 487 | 24722 | 101191 | 
| 90 | 39 src/_pytest/setuponly.py | 49 | 85| 265 | 24987 | 101755 | 
| 91 | 40 testing/python/metafunc.py | 1295 | 1333| 243 | 25230 | 114856 | 
| 92 | 40 testing/python/collect.py | 395 | 422| 193 | 25423 | 114856 | 
| 93 | 40 testing/python/collect.py | 79 | 113| 255 | 25678 | 114856 | 
| 94 | 40 src/_pytest/doctest.py | 392 | 411| 159 | 25837 | 114856 | 
| 95 | 41 testing/python/raises.py | 187 | 202| 135 | 25972 | 116578 | 
| 96 | 42 testing/example_scripts/collect/package_infinite_recursion/conftest.py | 1 | 3| 0 | 25972 | 116588 | 
| 97 | 42 testing/python/collect.py | 252 | 277| 181 | 26153 | 116588 | 
| 98 | 42 src/_pytest/junitxml.py | 430 | 443| 119 | 26272 | 116588 | 
| **-> 99 <-** | **42 src/_pytest/config/__init__.py** | 360 | 389| 283 | 26555 | 116588 | 
| 100 | 42 src/_pytest/pytester.py | 418 | 445| 160 | 26715 | 116588 | 
| 101 | 42 src/_pytest/compat.py | 268 | 357| 560 | 27275 | 116588 | 
| 102 | 42 src/_pytest/python.py | 152 | 173| 241 | 27516 | 116588 | 
| 103 | **42 src/_pytest/config/__init__.py** | 742 | 754| 139 | 27655 | 116588 | 
| 104 | 42 testing/python/integration.py | 1 | 35| 239 | 27894 | 116588 | 
| 105 | 42 testing/python/setup_only.py | 33 | 59| 167 | 28061 | 116588 | 
| 106 | **42 src/_pytest/config/__init__.py** | 229 | 265| 305 | 28366 | 116588 | 
| 107 | **42 src/_pytest/config/__init__.py** | 964 | 993| 231 | 28597 | 116588 | 
| 108 | 42 testing/python/collect.py | 1017 | 1032| 133 | 28730 | 116588 | 
| 109 | 42 testing/python/collect.py | 424 | 441| 123 | 28853 | 116588 | 
| 110 | 42 src/_pytest/main.py | 363 | 418| 473 | 29326 | 116588 | 
| 111 | 42 src/_pytest/capture.py | 729 | 767| 321 | 29647 | 116588 | 
| 112 | **42 src/_pytest/config/__init__.py** | 718 | 740| 172 | 29819 | 116588 | 
| 113 | 43 src/_pytest/hookspec.py | 70 | 85| 104 | 29923 | 121067 | 
| 114 | 43 testing/python/raises.py | 52 | 78| 179 | 30102 | 121067 | 
| 115 | 43 testing/python/integration.py | 364 | 381| 134 | 30236 | 121067 | 
| 116 | 43 src/_pytest/capture.py | 770 | 826| 466 | 30702 | 121067 | 
| 117 | **43 src/_pytest/config/__init__.py** | 785 | 800| 175 | 30877 | 121067 | 
| 118 | 43 src/_pytest/main.py | 281 | 306| 226 | 31103 | 121067 | 
| 119 | 43 src/_pytest/doctest.py | 313 | 339| 198 | 31301 | 121067 | 
| 120 | 43 src/_pytest/doctest.py | 1 | 34| 251 | 31552 | 121067 | 
| 121 | 44 src/_pytest/fixtures.py | 1246 | 1278| 246 | 31798 | 132146 | 
| 122 | 44 testing/python/setup_only.py | 185 | 200| 119 | 31917 | 132146 | 
| 123 | 44 testing/python/setup_only.py | 156 | 182| 167 | 32084 | 132146 | 
| 124 | 44 testing/python/collect.py | 1146 | 1172| 182 | 32266 | 132146 | 
| 125 | 44 testing/python/setup_only.py | 96 | 120| 150 | 32416 | 132146 | 
| 126 | 45 src/_pytest/mark/__init__.py | 77 | 95| 141 | 32557 | 133317 | 
| 127 | 46 src/_pytest/terminal.py | 140 | 167| 216 | 32773 | 141549 | 
| 128 | 47 src/_pytest/helpconfig.py | 209 | 245| 252 | 33025 | 143238 | 
| 129 | 47 src/_pytest/pytester.py | 57 | 82| 201 | 33226 | 143238 | 
| 130 | 47 testing/python/metafunc.py | 900 | 929| 221 | 33447 | 143238 | 
| 131 | **47 src/_pytest/config/__init__.py** | 1057 | 1081| 149 | 33596 | 143238 | 
| 132 | 48 testing/example_scripts/issue_519.py | 1 | 31| 350 | 33946 | 143704 | 
| 133 | 48 testing/python/fixtures.py | 3121 | 4097| 5881 | 39827 | 143704 | 
| 134 | 48 testing/python/integration.py | 158 | 179| 160 | 39987 | 143704 | 
| 135 | 48 testing/python/collect.py | 1055 | 1073| 154 | 40141 | 143704 | 
| 136 | 49 testing/conftest.py | 1 | 57| 333 | 40474 | 144195 | 
| 137 | 49 testing/python/collect.py | 115 | 128| 110 | 40584 | 144195 | 
| **-> 138 <-** | **50 src/_pytest/pathlib.py** | 1 | 39| 194 | 40778 | 146594 | 
| 139 | 50 testing/python/fixtures.py | 1067 | 2072| 6201 | 46979 | 146594 | 
| 140 | **50 src/_pytest/config/__init__.py** | 802 | 818| 152 | 47131 | 146594 | 
| 141 | **50 src/_pytest/config/__init__.py** | 185 | 213| 222 | 47353 | 146594 | 
| 142 | 51 src/_pytest/pastebin.py | 41 | 54| 145 | 47498 | 147364 | 
| 143 | 52 doc/en/example/xfail_demo.py | 1 | 39| 143 | 47641 | 147508 | 
| 144 | 52 testing/python/metafunc.py | 1237 | 1269| 199 | 47840 | 147508 | 
| 145 | 53 src/_pytest/_code/code.py | 1 | 26| 133 | 47973 | 155600 | 
| 146 | 54 src/_pytest/unittest.py | 242 | 283| 286 | 48259 | 157597 | 
| 147 | 54 src/_pytest/python.py | 291 | 305| 141 | 48400 | 157597 | 
| 148 | 54 src/_pytest/python.py | 339 | 354| 146 | 48546 | 157597 | 
| 149 | 54 testing/python/metafunc.py | 1 | 32| 173 | 48719 | 157597 | 
| 150 | 54 testing/python/integration.py | 439 | 466| 160 | 48879 | 157597 | 
| 151 | 55 testing/python/setup_plan.py | 1 | 20| 119 | 48998 | 157716 | 
| 152 | 55 src/_pytest/unittest.py | 106 | 155| 349 | 49347 | 157716 | 
| 153 | 56 src/_pytest/python_api.py | 720 | 737| 149 | 49496 | 164286 | 
| 154 | 57 src/_pytest/setupplan.py | 1 | 28| 163 | 49659 | 164450 | 
| 155 | 58 src/_pytest/resultlog.py | 38 | 60| 191 | 49850 | 165171 | 
| 156 | 58 src/_pytest/fixtures.py | 682 | 733| 470 | 50320 | 165171 | 
| 157 | 59 src/_pytest/config/findpaths.py | 105 | 152| 433 | 50753 | 166282 | 
| 158 | 59 testing/python/metafunc.py | 513 | 552| 329 | 51082 | 166282 | 
| 159 | 59 testing/python/collect.py | 1034 | 1053| 167 | 51249 | 166282 | 
| 160 | 59 src/_pytest/capture.py | 241 | 251| 108 | 51357 | 166282 | 
| 161 | 59 testing/python/integration.py | 137 | 156| 117 | 51474 | 166282 | 
| 162 | 59 testing/python/collect.py | 174 | 194| 127 | 51601 | 166282 | 
| 163 | 60 src/_pytest/config/exceptions.py | 1 | 10| 0 | 51601 | 166327 | 
| 164 | 61 doc/en/example/pythoncollection.py | 1 | 15| 0 | 51601 | 166374 | 
| 165 | 61 testing/python/collect.py | 540 | 557| 182 | 51783 | 166374 | 
| 166 | 61 src/_pytest/debugging.py | 274 | 283| 132 | 51915 | 166374 | 
| 167 | 61 testing/python/raises.py | 204 | 236| 253 | 52168 | 166374 | 
| 168 | 61 testing/python/metafunc.py | 1463 | 1495| 225 | 52393 | 166374 | 
| 169 | **61 src/_pytest/config/__init__.py** | 1009 | 1022| 140 | 52533 | 166374 | 
| 170 | 61 src/_pytest/setuponly.py | 1 | 46| 297 | 52830 | 166374 | 
| 171 | 62 doc/en/example/multipython.py | 25 | 46| 153 | 52983 | 166812 | 
| 172 | 63 testing/example_scripts/tmpdir/tmpdir_fixture.py | 1 | 8| 0 | 52983 | 166853 | 
| 173 | 63 src/_pytest/main.py | 227 | 261| 245 | 53228 | 166853 | 
| 174 | 63 testing/python/metafunc.py | 1144 | 1173| 199 | 53427 | 166853 | 
| 175 | 63 testing/python/integration.py | 266 | 282| 121 | 53548 | 166853 | 
| 176 | 63 src/_pytest/unittest.py | 1 | 26| 176 | 53724 | 166853 | 
| 177 | 64 src/_pytest/runner.py | 113 | 126| 129 | 53853 | 169581 | 
| 178 | 64 testing/python/collect.py | 495 | 522| 205 | 54058 | 169581 | 
| 179 | 64 src/_pytest/helpconfig.py | 117 | 136| 125 | 54183 | 169581 | 
| 180 | 64 src/_pytest/python.py | 176 | 196| 171 | 54354 | 169581 | 
| 181 | 65 src/_pytest/logging.py | 462 | 482| 206 | 54560 | 174552 | 
| 182 | 65 src/_pytest/python.py | 385 | 426| 367 | 54927 | 174552 | 
| 183 | 65 testing/python/collect.py | 1253 | 1263| 110 | 55037 | 174552 | 
| 184 | 65 testing/python/collect.py | 661 | 688| 227 | 55264 | 174552 | 
| 185 | 65 src/_pytest/_code/code.py | 1056 | 1066| 138 | 55402 | 174552 | 
| 186 | 65 src/_pytest/main.py | 151 | 177| 175 | 55577 | 174552 | 
| 187 | 65 src/_pytest/fixtures.py | 1 | 105| 672 | 56249 | 174552 | 
| 188 | 65 src/_pytest/pytester.py | 310 | 346| 210 | 56459 | 174552 | 
| 189 | 65 src/_pytest/python.py | 1232 | 1259| 217 | 56676 | 174552 | 
| 190 | 65 testing/python/metafunc.py | 955 | 981| 196 | 56872 | 174552 | 
| 191 | 65 testing/python/metafunc.py | 488 | 511| 149 | 57021 | 174552 | 
| 192 | 65 testing/python/metafunc.py | 1055 | 1076| 157 | 57178 | 174552 | 
| 193 | 65 testing/python/integration.py | 242 | 264| 183 | 57361 | 174552 | 
| 194 | 65 src/_pytest/unittest.py | 29 | 77| 397 | 57758 | 174552 | 
| 195 | 65 testing/python/raises.py | 128 | 155| 194 | 57952 | 174552 | 
| 196 | 66 src/_pytest/stepwise.py | 1 | 23| 131 | 58083 | 175266 | 
| 197 | 66 testing/python/collect.py | 690 | 710| 133 | 58216 | 175266 | 
| 198 | 66 testing/python/metafunc.py | 1386 | 1401| 112 | 58328 | 175266 | 
| 199 | **66 src/_pytest/config/__init__.py** | 267 | 294| 257 | 58585 | 175266 | 
| 200 | 66 src/_pytest/pytester.py | 448 | 540| 773 | 59358 | 175266 | 
| 201 | 66 doc/en/example/multipython.py | 1 | 22| 115 | 59473 | 175266 | 
| 202 | 67 src/_pytest/outcomes.py | 39 | 54| 124 | 59597 | 176780 | 
| 203 | 67 src/_pytest/runner.py | 151 | 193| 338 | 59935 | 176780 | 
| 204 | 67 src/_pytest/debugging.py | 122 | 158| 277 | 60212 | 176780 | 
| 205 | 67 src/_pytest/hookspec.py | 123 | 147| 206 | 60418 | 176780 | 
| 206 | 67 testing/python/setup_only.py | 251 | 270| 125 | 60543 | 176780 | 
| 207 | 67 src/_pytest/unittest.py | 157 | 188| 205 | 60748 | 176780 | 
| 208 | 67 testing/python/fixtures.py | 85 | 1065| 6303 | 67051 | 176780 | 
| 209 | 68 doc/en/conf.py | 1 | 116| 789 | 67840 | 179239 | 
| 210 | **68 src/_pytest/config/__init__.py** | 626 | 716| 648 | 68488 | 179239 | 
| 211 | 69 src/_pytest/cacheprovider.py | 38 | 63| 188 | 68676 | 182611 | 
| 212 | **69 src/_pytest/config/__init__.py** | 296 | 311| 158 | 68834 | 182611 | 
| 213 | 69 testing/python/metafunc.py | 1031 | 1053| 173 | 69007 | 182611 | 
| 214 | 69 src/_pytest/main.py | 420 | 445| 241 | 69248 | 182611 | 
| 215 | 70 src/_pytest/nodes.py | 345 | 372| 244 | 69492 | 185707 | 
| 216 | 70 src/_pytest/python_api.py | 687 | 717| 318 | 69810 | 185707 | 
| 217 | 70 doc/en/example/assertion/failure_demo.py | 164 | 188| 159 | 69969 | 185707 | 
| 218 | 70 src/_pytest/helpconfig.py | 86 | 114| 219 | 70188 | 185707 | 
| 219 | 71 src/_pytest/monkeypatch.py | 39 | 65| 180 | 70368 | 188122 | 
| 220 | 71 testing/python/metafunc.py | 983 | 998| 138 | 70506 | 188122 | 
| 221 | 71 testing/python/metafunc.py | 1078 | 1100| 156 | 70662 | 188122 | 
| 222 | 72 src/_pytest/deprecated.py | 1 | 32| 253 | 70915 | 188375 | 
| 223 | 72 testing/python/raises.py | 80 | 103| 159 | 71074 | 188375 | 
| 224 | 72 testing/python/raises.py | 105 | 126| 145 | 71219 | 188375 | 
| 225 | 72 src/_pytest/main.py | 597 | 612| 140 | 71359 | 188375 | 
| 226 | 72 src/_pytest/python.py | 429 | 466| 284 | 71643 | 188375 | 
| 227 | 72 testing/python/metafunc.py | 1193 | 1209| 139 | 71782 | 188375 | 
| 228 | 72 src/_pytest/python.py | 199 | 235| 355 | 72137 | 188375 | 
| 229 | 72 testing/python/raises.py | 1 | 50| 300 | 72437 | 188375 | 
| 230 | 72 src/_pytest/doctest.py | 37 | 84| 321 | 72758 | 188375 | 
| 231 | 72 src/_pytest/monkeypatch.py | 1 | 36| 245 | 73003 | 188375 | 
| 232 | 72 src/_pytest/python_api.py | 1 | 41| 229 | 73232 | 188375 | 
| 233 | 72 src/_pytest/monkeypatch.py | 68 | 97| 176 | 73408 | 188375 | 
| 234 | 72 src/_pytest/fixtures.py | 475 | 502| 208 | 73616 | 188375 | 
| 235 | **72 src/_pytest/config/__init__.py** | 756 | 783| 254 | 73870 | 188375 | 
| 236 | 72 src/_pytest/unittest.py | 223 | 239| 138 | 74008 | 188375 | 
| 237 | 72 testing/python/metafunc.py | 291 | 312| 229 | 74237 | 188375 | 
| 238 | 72 src/_pytest/logging.py | 563 | 593| 233 | 74470 | 188375 | 
| 239 | 72 src/_pytest/junitxml.py | 280 | 292| 132 | 74602 | 188375 | 
| 240 | 72 src/_pytest/debugging.py | 252 | 271| 156 | 74758 | 188375 | 
| 241 | 72 testing/python/metafunc.py | 878 | 897| 129 | 74887 | 188375 | 
| 242 | 72 doc/en/example/assertion/failure_demo.py | 206 | 253| 228 | 75115 | 188375 | 
| 243 | 72 src/_pytest/main.py | 575 | 595| 181 | 75296 | 188375 | 
| 244 | 72 src/_pytest/helpconfig.py | 39 | 83| 297 | 75593 | 188375 | 
| 245 | 72 src/_pytest/doctest.py | 509 | 534| 237 | 75830 | 188375 | 


### Hint

```
Can you show the import line that it is trying to import exactly? The cause might be https://github.com/pytest-dev/pytest/pull/5792.

cc @Oberon00
Seems very likely, unfortunately. If instead of using `os.normcase`, we could find a way to get the path with correct casing (`Path.resolve`?) that would probably be a safe fix. But I probably won't have time to fix that myself in the near future 😟
A unit test that imports a conftest from a module with upppercase characters in the package name sounds like a good addition too.
This bit me too.

* In `conftest.py` I `import muepy.imageProcessing.wafer.sawStreets as sawStreets`.
* This results in `ModuleNotFoundError: No module named 'muepy.imageprocessing'`.  Note the different case of the `P` in `imageProcessing`.
* The module actually lives in 
`C:\Users\angelo.peronio\AppData\Local\Continuum\miniconda3\envs\packaging\conda-bld\muepy_1567627432048\_test_env\Lib\site-packages\muepy\imageProcessing\wafer\sawStreets`.
* This happens after upgrading form pytest 5.1.1 to 5.1.2 on Windows 10.

Let me know whether I can help further.

### pytest output
\`\`\`
(%PREFIX%) %SRC_DIR%>pytest --pyargs muepy
============================= test session starts =============================
platform win32 -- Python 3.6.7, pytest-5.1.2, py-1.8.0, pluggy-0.12.0
rootdir: %SRC_DIR%
collected 0 items / 1 errors

=================================== ERRORS ====================================
________________________ ERROR collecting test session ________________________
..\_test_env\lib\site-packages\_pytest\config\__init__.py:440: in _importconftest
    return self._conftestpath2mod[conftestpath]
E   KeyError: local('c:\\users\\angelo.peronio\\appdata\\local\\continuum\\miniconda3\\envs\\packaging\\conda-bld\\muepy_1567627432048\\_test_env\\lib\\site-packages\\muepy\\imageprocessing\\wafer\\sawstreets\\tests\\conftest.py')

During handling of the above exception, another exception occurred:
..\_test_env\lib\site-packages\_pytest\config\__init__.py:446: in _importconftest
    mod = conftestpath.pyimport()
..\_test_env\lib\site-packages\py\_path\local.py:701: in pyimport
    __import__(modname)
E   ModuleNotFoundError: No module named 'muepy.imageprocessing'

During handling of the above exception, another exception occurred:
..\_test_env\lib\site-packages\py\_path\common.py:377: in visit
    for x in Visitor(fil, rec, ignore, bf, sort).gen(self):
..\_test_env\lib\site-packages\py\_path\common.py:429: in gen
    for p in self.gen(subdir):
..\_test_env\lib\site-packages\py\_path\common.py:429: in gen
    for p in self.gen(subdir):
..\_test_env\lib\site-packages\py\_path\common.py:429: in gen
    for p in self.gen(subdir):
..\_test_env\lib\site-packages\py\_path\common.py:418: in gen
    dirs = self.optsort([p for p in entries
..\_test_env\lib\site-packages\py\_path\common.py:419: in <listcomp>
    if p.check(dir=1) and (rec is None or rec(p))])
..\_test_env\lib\site-packages\_pytest\main.py:606: in _recurse
    ihook = self.gethookproxy(dirpath)
..\_test_env\lib\site-packages\_pytest\main.py:424: in gethookproxy
    my_conftestmodules = pm._getconftestmodules(fspath)
..\_test_env\lib\site-packages\_pytest\config\__init__.py:420: in _getconftestmodules
    mod = self._importconftest(conftestpath)
..\_test_env\lib\site-packages\_pytest\config\__init__.py:454: in _importconftest
    raise ConftestImportFailure(conftestpath, sys.exc_info())
E   _pytest.config.ConftestImportFailure: (local('c:\\users\\angelo.peronio\\appdata\\local\\continuum\\miniconda3\\envs\\packaging\\conda-bld\\muepy_1567627432048\\_test_env\\lib\\site-packages\\muepy\\imageprocessing\\wafer\\sawstreets\\tests\\conftest.py'), (<class 'ModuleNotFoundError'>, ModuleNotFoundError("No module named 'muepy.imageprocessing'",), <traceback object at 0x0000018F0D6C9A48>))
!!!!!!!!!!!!!!!!!!! Interrupted: 1 errors during collection !!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.32s ===============================
\`\`\`
```

## Patch

```diff
diff --git a/src/_pytest/config/__init__.py b/src/_pytest/config/__init__.py
--- a/src/_pytest/config/__init__.py
+++ b/src/_pytest/config/__init__.py
@@ -30,7 +30,6 @@
 from _pytest.compat import importlib_metadata
 from _pytest.outcomes import fail
 from _pytest.outcomes import Skipped
-from _pytest.pathlib import unique_path
 from _pytest.warning_types import PytestConfigWarning
 
 hookimpl = HookimplMarker("pytest")
@@ -367,7 +366,7 @@ def _set_initial_conftests(self, namespace):
         """
         current = py.path.local()
         self._confcutdir = (
-            unique_path(current.join(namespace.confcutdir, abs=True))
+            current.join(namespace.confcutdir, abs=True)
             if namespace.confcutdir
             else None
         )
@@ -406,13 +405,11 @@ def _getconftestmodules(self, path):
         else:
             directory = path
 
-        directory = unique_path(directory)
-
         # XXX these days we may rather want to use config.rootdir
         # and allow users to opt into looking into the rootdir parent
         # directories instead of requiring to specify confcutdir
         clist = []
-        for parent in directory.parts():
+        for parent in directory.realpath().parts():
             if self._confcutdir and self._confcutdir.relto(parent):
                 continue
             conftestpath = parent.join("conftest.py")
@@ -432,12 +429,14 @@ def _rget_with_confmod(self, name, path):
         raise KeyError(name)
 
     def _importconftest(self, conftestpath):
-        # Use realpath to avoid loading the same conftest twice
+        # Use a resolved Path object as key to avoid loading the same conftest twice
         # with build systems that create build directories containing
         # symlinks to actual files.
-        conftestpath = unique_path(conftestpath)
+        # Using Path().resolve() is better than py.path.realpath because
+        # it resolves to the correct path/drive in case-insensitive file systems (#5792)
+        key = Path(str(conftestpath)).resolve()
         try:
-            return self._conftestpath2mod[conftestpath]
+            return self._conftestpath2mod[key]
         except KeyError:
             pkgpath = conftestpath.pypkgpath()
             if pkgpath is None:
@@ -454,7 +453,7 @@ def _importconftest(self, conftestpath):
                 raise ConftestImportFailure(conftestpath, sys.exc_info())
 
             self._conftest_plugins.add(mod)
-            self._conftestpath2mod[conftestpath] = mod
+            self._conftestpath2mod[key] = mod
             dirpath = conftestpath.dirpath()
             if dirpath in self._dirpath2confmods:
                 for path, mods in self._dirpath2confmods.items():
diff --git a/src/_pytest/pathlib.py b/src/_pytest/pathlib.py
--- a/src/_pytest/pathlib.py
+++ b/src/_pytest/pathlib.py
@@ -11,7 +11,6 @@
 from os.path import expanduser
 from os.path import expandvars
 from os.path import isabs
-from os.path import normcase
 from os.path import sep
 from posixpath import sep as posix_sep
 
@@ -335,12 +334,3 @@ def fnmatch_ex(pattern, path):
 def parts(s):
     parts = s.split(sep)
     return {sep.join(parts[: i + 1]) or sep for i in range(len(parts))}
-
-
-def unique_path(path):
-    """Returns a unique path in case-insensitive (but case-preserving) file
-    systems such as Windows.
-
-    This is needed only for ``py.path.local``; ``pathlib.Path`` handles this
-    natively with ``resolve()``."""
-    return type(path)(normcase(str(path.realpath())))

```

## Test Patch

```diff
diff --git a/testing/test_conftest.py b/testing/test_conftest.py
--- a/testing/test_conftest.py
+++ b/testing/test_conftest.py
@@ -1,12 +1,12 @@
-import os.path
+import os
 import textwrap
+from pathlib import Path
 
 import py
 
 import pytest
 from _pytest.config import PytestPluginManager
 from _pytest.main import ExitCode
-from _pytest.pathlib import unique_path
 
 
 def ConftestWithSetinitial(path):
@@ -143,11 +143,11 @@ def test_conftestcutdir(testdir):
     # but we can still import a conftest directly
     conftest._importconftest(conf)
     values = conftest._getconftestmodules(conf.dirpath())
-    assert values[0].__file__.startswith(str(unique_path(conf)))
+    assert values[0].__file__.startswith(str(conf))
     # and all sub paths get updated properly
     values = conftest._getconftestmodules(p)
     assert len(values) == 1
-    assert values[0].__file__.startswith(str(unique_path(conf)))
+    assert values[0].__file__.startswith(str(conf))
 
 
 def test_conftestcutdir_inplace_considered(testdir):
@@ -156,7 +156,7 @@ def test_conftestcutdir_inplace_considered(testdir):
     conftest_setinitial(conftest, [conf.dirpath()], confcutdir=conf.dirpath())
     values = conftest._getconftestmodules(conf.dirpath())
     assert len(values) == 1
-    assert values[0].__file__.startswith(str(unique_path(conf)))
+    assert values[0].__file__.startswith(str(conf))
 
 
 @pytest.mark.parametrize("name", "test tests whatever .dotdir".split())
@@ -165,11 +165,12 @@ def test_setinitial_conftest_subdirs(testdir, name):
     subconftest = sub.ensure("conftest.py")
     conftest = PytestPluginManager()
     conftest_setinitial(conftest, [sub.dirpath()], confcutdir=testdir.tmpdir)
+    key = Path(str(subconftest)).resolve()
     if name not in ("whatever", ".dotdir"):
-        assert unique_path(subconftest) in conftest._conftestpath2mod
+        assert key in conftest._conftestpath2mod
         assert len(conftest._conftestpath2mod) == 1
     else:
-        assert subconftest not in conftest._conftestpath2mod
+        assert key not in conftest._conftestpath2mod
         assert len(conftest._conftestpath2mod) == 0
 
 
@@ -282,7 +283,7 @@ def fixture():
     reason="only relevant for case insensitive file systems",
 )
 def test_conftest_badcase(testdir):
-    """Check conftest.py loading when directory casing is wrong."""
+    """Check conftest.py loading when directory casing is wrong (#5792)."""
     testdir.tmpdir.mkdir("JenkinsRoot").mkdir("test")
     source = {"setup.py": "", "test/__init__.py": "", "test/conftest.py": ""}
     testdir.makepyfile(**{"JenkinsRoot/%s" % k: v for k, v in source.items()})
@@ -292,6 +293,16 @@ def test_conftest_badcase(testdir):
     assert result.ret == ExitCode.NO_TESTS_COLLECTED
 
 
+def test_conftest_uppercase(testdir):
+    """Check conftest.py whose qualified name contains uppercase characters (#5819)"""
+    source = {"__init__.py": "", "Foo/conftest.py": "", "Foo/__init__.py": ""}
+    testdir.makepyfile(**source)
+
+    testdir.tmpdir.chdir()
+    result = testdir.runpytest()
+    assert result.ret == ExitCode.NO_TESTS_COLLECTED
+
+
 def test_no_conftest(testdir):
     testdir.makeconftest("assert 0")
     result = testdir.runpytest("--noconftest")

```


## Code snippets

### 1 - testing/python/collect.py:

Start line: 713, End line: 728

```python
class TestConftestCustomization:
    def test_pytest_pycollect_module(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            class MyModule(pytest.Module):
                pass
            def pytest_pycollect_makemodule(path, parent):
                if path.basename == "test_xyz.py":
                    return MyModule(path, parent)
        """
        )
        testdir.makepyfile("def test_some(): pass")
        testdir.makepyfile(test_xyz="def test_func(): pass")
        result = testdir.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*<Module*test_pytest*", "*<MyModule*xyz*"])
```
### 2 - testing/python/collect.py:

Start line: 730, End line: 753

```python
class TestConftestCustomization:

    def test_customized_pymakemodule_issue205_subdir(self, testdir):
        b = testdir.mkdir("a").mkdir("b")
        b.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.hookimpl(hookwrapper=True)
                def pytest_pycollect_makemodule():
                    outcome = yield
                    mod = outcome.get_result()
                    mod.obj.hello = "world"
                """
            )
        )
        b.join("test_module.py").write(
            textwrap.dedent(
                """\
                def test_hello():
                    assert hello == "world"
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)
```
### 3 - src/_pytest/config/__init__.py:

Start line: 216, End line: 226

```python
def _fail_on_non_top_pytest_plugins(conftestpath, confcutdir):
    msg = (
        "Defining 'pytest_plugins' in a non-top-level conftest is no longer supported:\n"
        "It affects the entire test suite instead of just below the conftest as expected.\n"
        "  {}\n"
        "Please move it to a top level conftest file at the rootdir:\n"
        "  {}\n"
        "For more information, visit:\n"
        "  https://docs.pytest.org/en/latest/deprecations.html#pytest-plugins-in-non-top-level-conftest-files"
    )
    fail(msg.format(conftestpath, confcutdir), pytrace=False)
```
### 4 - src/_pytest/config/__init__.py:

Start line: 434, End line: 466

```python
class PytestPluginManager(PluginManager):

    def _importconftest(self, conftestpath):
        # Use realpath to avoid loading the same conftest twice
        # with build systems that create build directories containing
        # symlinks to actual files.
        conftestpath = unique_path(conftestpath)
        try:
            return self._conftestpath2mod[conftestpath]
        except KeyError:
            pkgpath = conftestpath.pypkgpath()
            if pkgpath is None:
                _ensure_removed_sysmodule(conftestpath.purebasename)
            try:
                mod = conftestpath.pyimport()
                if (
                    hasattr(mod, "pytest_plugins")
                    and self._configured
                    and not self._using_pyargs
                ):
                    _fail_on_non_top_pytest_plugins(conftestpath, self._confcutdir)
            except Exception:
                raise ConftestImportFailure(conftestpath, sys.exc_info())

            self._conftest_plugins.add(mod)
            self._conftestpath2mod[conftestpath] = mod
            dirpath = conftestpath.dirpath()
            if dirpath in self._dirpath2confmods:
                for path, mods in self._dirpath2confmods.items():
                    if path and path.relto(dirpath) or path == dirpath:
                        assert mod not in mods
                        mods.append(mod)
            self.trace("loaded conftestmodule %r" % (mod))
            self.consider_conftest(mod)
            return mod
```
### 5 - src/pytest.py:

Start line: 1, End line: 107

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
from _pytest import __version__
from _pytest.assertion import register_assert_rewrite
from _pytest.config import cmdline
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import UsageError
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import fillfixtures as _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.main import ExitCode
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
    "Collector",
    "deprecated_call",
    "exit",
    "ExitCode",
    "fail",
    "File",
    "fixture",
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

if __name__ == "__main__":
    # if run as a script or by 'python -m pytest'
    # we trigger the below "else" condition by the following import
    import pytest

    raise SystemExit(pytest.main())
else:

    from _pytest.compat import _setup_collect_fakemodule

    _setup_collect_fakemodule()
```
### 6 - testing/python/collect.py:

Start line: 1, End line: 33

```python
import os
import sys
import textwrap

import _pytest._code
import pytest
from _pytest.main import ExitCode
from _pytest.nodes import Collector


class TestModule:
    def test_failing_import(self, testdir):
        modcol = testdir.getmodulecol("import alksdjalskdjalkjals")
        pytest.raises(Collector.CollectError, modcol.collect)

    def test_import_duplicate(self, testdir):
        a = testdir.mkdir("a")
        b = testdir.mkdir("b")
        p = a.ensure("test_whatever.py")
        p.pyimport()
        del sys.modules["test_whatever"]
        b.ensure("test_whatever.py")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*import*mismatch*",
                "*imported*test_whatever*",
                "*%s*" % a.join("test_whatever.py"),
                "*not the same*",
                "*%s*" % b.join("test_whatever.py"),
                "*HINT*",
            ]
        )
```
### 7 - testing/python/collect.py:

Start line: 788, End line: 810

```python
class TestConftestCustomization:

    def test_pytest_pycollect_makeitem(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            class MyFunction(pytest.Function):
                pass
            def pytest_pycollect_makeitem(collector, name, obj):
                if name == "some":
                    return MyFunction(name, collector)
        """
        )
        testdir.makepyfile("def some(): pass")
        result = testdir.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*MyFunction*some*"])

    def test_makeitem_non_underscore(self, testdir, monkeypatch):
        modcol = testdir.getmodulecol("def _hello(): pass")
        values = []
        monkeypatch.setattr(
            pytest.Module, "_makeitem", lambda self, name, obj: values.append(name)
        )
        values = modcol.collect()
        assert "_hello" not in values
```
### 8 - testing/example_scripts/config/collect_pytest_prefix/conftest.py:

Start line: 1, End line: 3

```python

```
### 9 - testing/python/collect.py:

Start line: 59, End line: 77

```python
class TestModule:

    def test_syntax_error_in_module(self, testdir):
        modcol = testdir.getmodulecol("this is a syntax error")
        pytest.raises(modcol.CollectError, modcol.collect)
        pytest.raises(modcol.CollectError, modcol.collect)

    def test_module_considers_pluginmanager_at_import(self, testdir):
        modcol = testdir.getmodulecol("pytest_plugins='xasdlkj',")
        pytest.raises(ImportError, lambda: modcol.obj)

    def test_invalid_test_module_name(self, testdir):
        a = testdir.mkdir("a")
        a.ensure("test_one.part1.py")
        result = testdir.runpytest("-rw")
        result.stdout.fnmatch_lines(
            [
                "ImportError while importing test module*test_one.part1*",
                "Hint: make sure your test modules/packages have valid Python names.",
            ]
        )
```
### 10 - testing/python/collect.py:

Start line: 755, End line: 786

```python
class TestConftestCustomization:

    def test_customized_pymakeitem(self, testdir):
        b = testdir.mkdir("a").mkdir("b")
        b.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.hookimpl(hookwrapper=True)
                def pytest_pycollect_makeitem():
                    outcome = yield
                    if outcome.excinfo is None:
                        result = outcome.get_result()
                        if result:
                            for func in result:
                                func._some123 = "world"
                """
            )
        )
        b.join("test_module.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture()
                def obj(request):
                    return request.node._some123
                def test_hello(obj):
                    assert obj == "world"
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)
```
### 11 - src/_pytest/config/__init__.py:

Start line: 1, End line: 44

```python
""" command line options, ini-file and conftest.py processing. """
import argparse
import copy
import inspect
import os
import shlex
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path

import attr
import py
from packaging.version import Version
from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager

import _pytest._code
import _pytest.assertion
import _pytest.deprecated
import _pytest.hookspec  # the extension point definitions
from .exceptions import PrintHelp
from .exceptions import UsageError
from .findpaths import determine_setup
from .findpaths import exists
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest.compat import importlib_metadata
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import unique_path
from _pytest.warning_types import PytestConfigWarning

hookimpl = HookimplMarker("pytest")
hookspec = HookspecMarker("pytest")


class ConftestImportFailure(Exception):
    def __init__(self, path, excinfo):
        Exception.__init__(self, path, excinfo)
        self.path = path
        self.excinfo = excinfo
```
### 13 - src/_pytest/config/__init__.py:

Start line: 995, End line: 1007

```python
class Config:

    def _getconftest_pathlist(self, name, path):
        try:
            mod, relroots = self.pluginmanager._rget_with_confmod(name, path)
        except KeyError:
            return None
        modpath = py.path.local(mod.__file__).dirpath()
        values = []
        for relroot in relroots:
            if not isinstance(relroot, py.path.local):
                relroot = relroot.replace("/", py.path.local.sep)
                relroot = modpath.join(relroot, abs=True)
            values.append(relroot)
        return values
```
### 23 - src/_pytest/config/__init__.py:

Start line: 854, End line: 901

```python
class Config:

    def _preparse(self, args, addopts=True):
        if addopts:
            env_addopts = os.environ.get("PYTEST_ADDOPTS", "")
            if len(env_addopts):
                args[:] = (
                    self._validate_args(shlex.split(env_addopts), "via PYTEST_ADDOPTS")
                    + args
                )
        self._initini(args)
        if addopts:
            args[:] = (
                self._validate_args(self.getini("addopts"), "via addopts config") + args
            )

        self._checkversion()
        self._consider_importhook(args)
        self.pluginmanager.consider_preparse(args)
        if not os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
            # Don't autoload from setuptools entry point. Only explicitly specified
            # plugins are going to be loaded.
            self.pluginmanager.load_setuptools_entrypoints("pytest11")
        self.pluginmanager.consider_env()
        self.known_args_namespace = ns = self._parser.parse_known_args(
            args, namespace=copy.copy(self.option)
        )
        if self.known_args_namespace.confcutdir is None and self.inifile:
            confcutdir = py.path.local(self.inifile).dirname
            self.known_args_namespace.confcutdir = confcutdir
        try:
            self.hook.pytest_load_initial_conftests(
                early_config=self, args=args, parser=self._parser
            )
        except ConftestImportFailure:
            e = sys.exc_info()[1]
            if ns.help or ns.version:
                # we don't want to prevent --help/--version to work
                # so just let is pass and print a warning at the end
                from _pytest.warnings import _issue_warning_captured

                _issue_warning_captured(
                    PytestConfigWarning(
                        "could not load initial conftests: {}".format(e.path)
                    ),
                    self.hook,
                    stacklevel=2,
                )
            else:
                raise
```
### 32 - src/_pytest/config/__init__.py:

Start line: 468, End line: 529

```python
class PytestPluginManager(PluginManager):

    #
    # API for bootstrapping plugin loading
    #
    #

    def consider_preparse(self, args):
        i = 0
        n = len(args)
        while i < n:
            opt = args[i]
            i += 1
            if isinstance(opt, str):
                if opt == "-p":
                    try:
                        parg = args[i]
                    except IndexError:
                        return
                    i += 1
                elif opt.startswith("-p"):
                    parg = opt[2:]
                else:
                    continue
                self.consider_pluginarg(parg)

    def consider_pluginarg(self, arg):
        if arg.startswith("no:"):
            name = arg[3:]
            if name in essential_plugins:
                raise UsageError("plugin %s cannot be disabled" % name)

            # PR #4304 : remove stepwise if cacheprovider is blocked
            if name == "cacheprovider":
                self.set_blocked("stepwise")
                self.set_blocked("pytest_stepwise")

            self.set_blocked(name)
            if not name.startswith("pytest_"):
                self.set_blocked("pytest_" + name)
        else:
            name = arg
            # Unblock the plugin.  None indicates that it has been blocked.
            # There is no interface with pluggy for this.
            if self._name2plugin.get(name, -1) is None:
                del self._name2plugin[name]
            if not name.startswith("pytest_"):
                if self._name2plugin.get("pytest_" + name, -1) is None:
                    del self._name2plugin["pytest_" + name]
            self.import_plugin(arg, consider_entry_points=True)

    def consider_conftest(self, conftestmodule):
        self.register(conftestmodule, name=conftestmodule.__file__)

    def consider_env(self):
        self._import_plugin_specs(os.environ.get("PYTEST_PLUGINS"))

    def consider_module(self, mod):
        self._import_plugin_specs(getattr(mod, "pytest_plugins", []))

    def _import_plugin_specs(self, spec):
        plugins = _get_plugin_specs_as_list(spec)
        for import_spec in plugins:
            self.import_plugin(import_spec)
```
### 35 - src/_pytest/config/__init__.py:

Start line: 391, End line: 432

```python
class PytestPluginManager(PluginManager):

    def _try_load_conftest(self, anchor):
        self._getconftestmodules(anchor)
        # let's also consider test* subdirs
        if anchor.check(dir=1):
            for x in anchor.listdir("test*"):
                if x.check(dir=1):
                    self._getconftestmodules(x)

    @lru_cache(maxsize=128)
    def _getconftestmodules(self, path):
        if self._noconftest:
            return []

        if path.isfile():
            directory = path.dirpath()
        else:
            directory = path

        directory = unique_path(directory)

        # XXX these days we may rather want to use config.rootdir
        # and allow users to opt into looking into the rootdir parent
        # directories instead of requiring to specify confcutdir
        clist = []
        for parent in directory.parts():
            if self._confcutdir and self._confcutdir.relto(parent):
                continue
            conftestpath = parent.join("conftest.py")
            if conftestpath.isfile():
                mod = self._importconftest(conftestpath)
                clist.append(mod)
        self._dirpath2confmods[directory] = clist
        return clist

    def _rget_with_confmod(self, name, path):
        modules = self._getconftestmodules(path)
        for mod in reversed(modules):
            try:
                return mod, getattr(mod, name)
            except AttributeError:
                continue
        raise KeyError(name)
```
### 48 - src/_pytest/config/__init__.py:

Start line: 342, End line: 359

```python
class PytestPluginManager(PluginManager):

    def pytest_configure(self, config):
        # XXX now that the pluginmanager exposes hookimpl(tryfirst...)
        # we should remove tryfirst/trylast as markers
        config.addinivalue_line(
            "markers",
            "tryfirst: mark a hook implementation function such that the "
            "plugin machinery will try to call it first/as early as possible.",
        )
        config.addinivalue_line(
            "markers",
            "trylast: mark a hook implementation function such that the "
            "plugin machinery will try to call it last/as late as possible.",
        )
        self._configured = True

    #
    # internal API for local conftest plugin handling
    #
```
### 55 - src/_pytest/config/__init__.py:

Start line: 903, End line: 943

```python
class Config:

    def _checkversion(self):
        import pytest

        minver = self.inicfg.get("minversion", None)
        if minver:
            if Version(minver) > Version(pytest.__version__):
                raise pytest.UsageError(
                    "%s:%d: requires pytest-%s, actual pytest-%s'"
                    % (
                        self.inicfg.config.path,
                        self.inicfg.lineof("minversion"),
                        minver,
                        pytest.__version__,
                    )
                )

    def parse(self, args, addopts=True):
        # parse given cmdline arguments into this config object.
        assert not hasattr(
            self, "args"
        ), "can only parse cmdline args at most once per Config object"
        assert self.invocation_params.args == args
        self.hook.pytest_addhooks.call_historic(
            kwargs=dict(pluginmanager=self.pluginmanager)
        )
        self._preparse(args, addopts=addopts)
        # XXX deprecated hook:
        self.hook.pytest_cmdline_preparse(config=self, args=args)
        self._parser.after_preparse = True
        try:
            args = self._parser.parse_setoption(
                args, self.option, namespace=self.option
            )
            if not args:
                if self.invocation_dir == self.rootdir:
                    args = self.getini("testpaths")
                if not args:
                    args = [str(self.invocation_dir)]
            self.args = args
        except PrintHelp:
            pass
```
### 57 - src/_pytest/config/__init__.py:

Start line: 599, End line: 623

```python
def _ensure_removed_sysmodule(modname):
    try:
        del sys.modules[modname]
    except KeyError:
        pass


class Notset:
    def __repr__(self):
        return "<NOTSET>"


notset = Notset()


def _iter_rewritable_modules(package_files):
    for fn in package_files:
        is_simple_module = "/" not in fn and fn.endswith(".py")
        is_package = fn.count("/") == 1 and fn.endswith("__init__.py")
        if is_simple_module:
            module_name, _ = os.path.splitext(fn)
            yield module_name
        elif is_package:
            package_name = os.path.dirname(fn)
            yield package_name
```
### 99 - src/_pytest/config/__init__.py:

Start line: 360, End line: 389

```python
class PytestPluginManager(PluginManager):
    def _set_initial_conftests(self, namespace):
        """ load initial conftest files given a preparsed "namespace".
            As conftest files may add their own command line options
            which have arguments ('--my-opt somepath') we might get some
            false positives.  All builtin and 3rd party plugins will have
            been loaded, however, so common options will not confuse our logic
            here.
        """
        current = py.path.local()
        self._confcutdir = (
            unique_path(current.join(namespace.confcutdir, abs=True))
            if namespace.confcutdir
            else None
        )
        self._noconftest = namespace.noconftest
        self._using_pyargs = namespace.pyargs
        testpaths = namespace.file_or_dir
        foundanchor = False
        for path in testpaths:
            path = str(path)
            # remove node-id syntax
            i = path.find("::")
            if i != -1:
                path = path[:i]
            anchor = current.join(path, abs=1)
            if exists(anchor):  # we found some file object
                self._try_load_conftest(anchor)
                foundanchor = True
        if not foundanchor:
            self._try_load_conftest(current)
```
### 103 - src/_pytest/config/__init__.py:

Start line: 742, End line: 754

```python
class Config:

    def notify_exception(self, excinfo, option=None):
        if option and getattr(option, "fulltrace", False):
            style = "long"
        else:
            style = "native"
        excrepr = excinfo.getrepr(
            funcargs=True, showlocals=getattr(option, "showlocals", False), style=style
        )
        res = self.hook.pytest_internalerror(excrepr=excrepr, excinfo=excinfo)
        if not any(res):
            for line in str(excrepr).split("\n"):
                sys.stderr.write("INTERNALERROR> %s\n" % line)
                sys.stderr.flush()
```
### 106 - src/_pytest/config/__init__.py:

Start line: 229, End line: 265

```python
class PytestPluginManager(PluginManager):
    """
    Overwrites :py:class:`pluggy.PluginManager <pluggy.PluginManager>` to add pytest-specific
    functionality:

    * loading plugins from the command line, ``PYTEST_PLUGINS`` env variable and
      ``pytest_plugins`` global variables found in plugins being loaded;
    * ``conftest.py`` loading during start-up;
    """

    def __init__(self):
        super().__init__("pytest")
        self._conftest_plugins = set()

        # state related to local conftest plugins
        self._dirpath2confmods = {}
        self._conftestpath2mod = {}
        self._confcutdir = None
        self._noconftest = False
        self._duplicatepaths = set()

        self.add_hookspecs(_pytest.hookspec)
        self.register(self)
        if os.environ.get("PYTEST_DEBUG"):
            err = sys.stderr
            encoding = getattr(err, "encoding", "utf8")
            try:
                err = py.io.dupfile(err, encoding=encoding)
            except Exception:
                pass
            self.trace.root.setwriter(err.write)
            self.enable_tracing()

        # Config._consider_importhook will set a real object if required.
        self.rewrite_hook = _pytest.assertion.DummyRewriteHook()
        # Used to know when we are importing conftests after the pytest_configure stage
        self._configured = False
```
### 107 - src/_pytest/config/__init__.py:

Start line: 964, End line: 993

```python
class Config:

    def _getini(self, name):
        try:
            description, type, default = self._parser._inidict[name]
        except KeyError:
            raise ValueError("unknown configuration value: {!r}".format(name))
        value = self._get_override_ini_value(name)
        if value is None:
            try:
                value = self.inicfg[name]
            except KeyError:
                if default is not None:
                    return default
                if type is None:
                    return ""
                return []
        if type == "pathlist":
            dp = py.path.local(self.inicfg.config.path).dirpath()
            values = []
            for relpath in shlex.split(value):
                values.append(dp.join(relpath, abs=True))
            return values
        elif type == "args":
            return shlex.split(value)
        elif type == "linelist":
            return [t for t in map(lambda x: x.strip(), value.split("\n")) if t]
        elif type == "bool":
            return bool(_strtobool(value.strip()))
        else:
            assert type is None
            return value
```
### 112 - src/_pytest/config/__init__.py:

Start line: 718, End line: 740

```python
class Config:

    def pytest_cmdline_parse(self, pluginmanager, args):
        try:
            self.parse(args)
        except UsageError:

            # Handle --version and --help here in a minimal fashion.
            # This gets done via helpconfig normally, but its
            # pytest_cmdline_main is not called in case of errors.
            if getattr(self.option, "version", False) or "--version" in args:
                from _pytest.helpconfig import showversion

                showversion(self)
            elif (
                getattr(self.option, "help", False) or "--help" in args or "-h" in args
            ):
                self._parser._getparser().print_help()
                sys.stdout.write(
                    "\nNOTE: displaying only minimal help due to UsageError.\n\n"
                )

            raise

        return self
```
### 117 - src/_pytest/config/__init__.py:

Start line: 785, End line: 800

```python
class Config:

    def _initini(self, args):
        ns, unknown_args = self._parser.parse_known_and_unknown_args(
            args, namespace=copy.copy(self.option)
        )
        r = determine_setup(
            ns.inifilename,
            ns.file_or_dir + unknown_args,
            rootdir_cmd_arg=ns.rootdir or None,
            config=self,
        )
        self.rootdir, self.inifile, self.inicfg = r
        self._parser.extra_info["rootdir"] = self.rootdir
        self._parser.extra_info["inifile"] = self.inifile
        self._parser.addini("addopts", "extra command line options", "args")
        self._parser.addini("minversion", "minimally required pytest version")
        self._override_ini = ns.override_ini or ()
```
### 131 - src/_pytest/config/__init__.py:

Start line: 1057, End line: 1081

```python
def _assertion_supported():
    try:
        assert False
    except AssertionError:
        return True
    else:
        return False


def _warn_about_missing_assertion(mode):
    if not _assertion_supported():
        if mode == "plain":
            sys.stderr.write(
                "WARNING: ASSERTIONS ARE NOT EXECUTED"
                " and FAILING TESTS WILL PASS.  Are you"
                " using python -O?"
            )
        else:
            sys.stderr.write(
                "WARNING: assertions not in test modules or"
                " plugins will be ignored"
                " because assert statements are not executed "
                "by the underlying Python interpreter "
                "(are you using python -O?)\n"
            )
```
### 138 - src/_pytest/pathlib.py:

Start line: 1, End line: 39

```python
import atexit
import fnmatch
import itertools
import operator
import os
import shutil
import sys
import uuid
import warnings
from functools import partial
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import normcase
from os.path import sep
from posixpath import sep as posix_sep

from _pytest.warning_types import PytestWarning

if sys.version_info[:2] >= (3, 6):
    from pathlib import Path, PurePath
else:
    from pathlib2 import Path, PurePath

__all__ = ["Path", "PurePath"]


LOCK_TIMEOUT = 60 * 60 * 3

get_lock_path = operator.methodcaller("joinpath", ".lock")


def ensure_reset_dir(path):
    """
    ensures the given path is an empty directory
    """
    if path.exists():
        rm_rf(path)
    path.mkdir()
```
### 140 - src/_pytest/config/__init__.py:

Start line: 802, End line: 818

```python
class Config:

    def _consider_importhook(self, args):
        """Install the PEP 302 import hook if using assertion rewriting.

        Needs to parse the --assert=<mode> option from the commandline
        and find all the installed plugins to mark them for rewriting
        by the importhook.
        """
        ns, unknown_args = self._parser.parse_known_and_unknown_args(args)
        mode = getattr(ns, "assertmode", "plain")
        if mode == "rewrite":
            try:
                hook = _pytest.assertion.install_importhook(self)
            except SystemError:
                mode = "plain"
            else:
                self._mark_plugins_for_rewrite(hook)
        _warn_about_missing_assertion(mode)
```
### 141 - src/_pytest/config/__init__.py:

Start line: 185, End line: 213

```python
def _prepareconfig(args=None, plugins=None):
    warning = None
    if args is None:
        args = sys.argv[1:]
    elif isinstance(args, py.path.local):
        args = [str(args)]
    elif not isinstance(args, (tuple, list)):
        msg = "`args` parameter expected to be a list or tuple of strings, got: {!r} (type: {})"
        raise TypeError(msg.format(args, type(args)))

    config = get_config(args, plugins)
    pluginmanager = config.pluginmanager
    try:
        if plugins:
            for plugin in plugins:
                if isinstance(plugin, str):
                    pluginmanager.consider_pluginarg(plugin)
                else:
                    pluginmanager.register(plugin)
        if warning:
            from _pytest.warnings import _issue_warning_captured

            _issue_warning_captured(warning, hook=config.hook, stacklevel=4)
        return pluginmanager.hook.pytest_cmdline_parse(
            pluginmanager=pluginmanager, args=args
        )
    except BaseException:
        config._ensure_unconfigure()
        raise
```
### 169 - src/_pytest/config/__init__.py:

Start line: 1009, End line: 1022

```python
class Config:

    def _get_override_ini_value(self, name):
        value = None
        # override_ini is a list of "ini=value" options
        # always use the last item if multiple values are set for same ini-name,
        # e.g. -o foo=bar1 -o foo=bar2 will set foo to bar2
        for ini_config in self._override_ini:
            try:
                key, user_ini_value = ini_config.split("=", 1)
            except ValueError:
                raise UsageError("-o/--override-ini expects option=value style.")
            else:
                if key == name:
                    value = user_ini_value
        return value
```
### 199 - src/_pytest/config/__init__.py:

Start line: 267, End line: 294

```python
class PytestPluginManager(PluginManager):

    def parse_hookimpl_opts(self, plugin, name):
        # pytest hooks are always prefixed with pytest_
        # so we avoid accessing possibly non-readable attributes
        # (see issue #1073)
        if not name.startswith("pytest_"):
            return
        # ignore names which can not be hooks
        if name == "pytest_plugins":
            return

        method = getattr(plugin, name)
        opts = super().parse_hookimpl_opts(plugin, name)

        # consider only actual functions for hooks (#3775)
        if not inspect.isroutine(method):
            return

        # collect unmarked hooks as long as they have the `pytest_' prefix
        if opts is None and name.startswith("pytest_"):
            opts = {}
        if opts is not None:
            # TODO: DeprecationWarning, people should use hookimpl
            # https://github.com/pytest-dev/pytest/issues/4562
            known_marks = {m.name for m in getattr(method, "pytestmark", [])}

            for name in ("tryfirst", "trylast", "optionalhook", "hookwrapper"):
                opts.setdefault(name, hasattr(method, name) or name in known_marks)
        return opts
```
### 210 - src/_pytest/config/__init__.py:

Start line: 626, End line: 716

```python
class Config:
    """
    Access to configuration values, pluginmanager and plugin hooks.

    :ivar PytestPluginManager pluginmanager: the plugin manager handles plugin registration and hook invocation.

    :ivar argparse.Namespace option: access to command line option as attributes.

    :ivar InvocationParams invocation_params:

        Object containing the parameters regarding the ``pytest.main``
        invocation.

        Contains the following read-only attributes:

        * ``args``: list of command-line arguments as passed to ``pytest.main()``.
        * ``plugins``: list of extra plugins, might be None.
        * ``dir``: directory where ``pytest.main()`` was invoked from.
    """

    @attr.s(frozen=True)
    class InvocationParams:
        """Holds parameters passed during ``pytest.main()``

        .. note::

            Currently the environment variable PYTEST_ADDOPTS is also handled by
            pytest implicitly, not being part of the invocation.

            Plugins accessing ``InvocationParams`` must be aware of that.
        """

        args = attr.ib()
        plugins = attr.ib()
        dir = attr.ib()

    def __init__(self, pluginmanager, *, invocation_params=None):
        from .argparsing import Parser, FILE_OR_DIR

        if invocation_params is None:
            invocation_params = self.InvocationParams(
                args=(), plugins=None, dir=Path().resolve()
            )

        self.option = argparse.Namespace()
        self.invocation_params = invocation_params

        _a = FILE_OR_DIR
        self._parser = Parser(
            usage="%(prog)s [options] [{}] [{}] [...]".format(_a, _a),
            processopt=self._processopt,
        )
        self.pluginmanager = pluginmanager
        self.trace = self.pluginmanager.trace.root.get("config")
        self.hook = self.pluginmanager.hook
        self._inicache = {}
        self._override_ini = ()
        self._opt2dest = {}
        self._cleanup = []
        self.pluginmanager.register(self, "pytestconfig")
        self._configured = False
        self.hook.pytest_addoption.call_historic(kwargs=dict(parser=self._parser))

    @property
    def invocation_dir(self):
        """Backward compatibility"""
        return py.path.local(str(self.invocation_params.dir))

    def add_cleanup(self, func):
        """ Add a function to be called when the config object gets out of
        use (usually coninciding with pytest_unconfigure)."""
        self._cleanup.append(func)

    def _do_configure(self):
        assert not self._configured
        self._configured = True
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            self.hook.pytest_configure.call_historic(kwargs=dict(config=self))

    def _ensure_unconfigure(self):
        if self._configured:
            self._configured = False
            self.hook.pytest_unconfigure(config=self)
            self.hook.pytest_configure._call_history = []
        while self._cleanup:
            fin = self._cleanup.pop()
            fin()

    def get_terminal_writer(self):
        return self.pluginmanager.get_plugin("terminalreporter")._tw
```
### 212 - src/_pytest/config/__init__.py:

Start line: 296, End line: 311

```python
class PytestPluginManager(PluginManager):

    def parse_hookspec_opts(self, module_or_class, name):
        opts = super().parse_hookspec_opts(module_or_class, name)
        if opts is None:
            method = getattr(module_or_class, name)

            if name.startswith("pytest_"):
                # todo: deprecate hookspec hacks
                # https://github.com/pytest-dev/pytest/issues/4562
                known_marks = {m.name for m in getattr(method, "pytestmark", [])}
                opts = {
                    "firstresult": hasattr(method, "firstresult")
                    or "firstresult" in known_marks,
                    "historic": hasattr(method, "historic")
                    or "historic" in known_marks,
                }
        return opts
```
### 235 - src/_pytest/config/__init__.py:

Start line: 756, End line: 783

```python
class Config:

    def cwd_relative_nodeid(self, nodeid):
        # nodeid's are relative to the rootpath, compute relative to cwd
        if self.invocation_dir != self.rootdir:
            fullpath = self.rootdir.join(nodeid)
            nodeid = self.invocation_dir.bestrelpath(fullpath)
        return nodeid

    @classmethod
    def fromdictargs(cls, option_dict, args):
        """ constructor useable for subprocesses. """
        config = get_config(args)
        config.option.__dict__.update(option_dict)
        config.parse(args, addopts=False)
        for x in config.option.plugins:
            config.pluginmanager.consider_pluginarg(x)
        return config

    def _processopt(self, opt):
        for name in opt._short_opts + opt._long_opts:
            self._opt2dest[name] = opt.dest

        if hasattr(opt, "default") and opt.dest:
            if not hasattr(self.option, opt.dest):
                setattr(self.option, opt.dest, opt.default)

    @hookimpl(trylast=True)
    def pytest_load_initial_conftests(self, early_config):
        self.pluginmanager._set_initial_conftests(early_config.known_args_namespace)
```
