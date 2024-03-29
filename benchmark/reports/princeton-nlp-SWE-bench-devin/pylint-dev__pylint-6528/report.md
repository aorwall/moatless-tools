# pylint-dev__pylint-6528

| **pylint-dev/pylint** | `273a8b25620467c1e5686aa8d2a1dbb8c02c78d0` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 29207 |
| **Avg pos** | 39.0 |
| **Min pos** | 78 |
| **Max pos** | 78 |
| **Top file pos** | 4 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -46,6 +46,20 @@ def _is_in_ignore_list_re(element: str, ignore_list_re: list[Pattern[str]]) -> b
     return any(file_pattern.match(element) for file_pattern in ignore_list_re)
 
 
+def _is_ignored_file(
+    element: str,
+    ignore_list: list[str],
+    ignore_list_re: list[Pattern[str]],
+    ignore_list_paths_re: list[Pattern[str]],
+) -> bool:
+    basename = os.path.basename(element)
+    return (
+        basename in ignore_list
+        or _is_in_ignore_list_re(basename, ignore_list_re)
+        or _is_in_ignore_list_re(element, ignore_list_paths_re)
+    )
+
+
 def expand_modules(
     files_or_modules: Sequence[str],
     ignore_list: list[str],
@@ -61,10 +75,8 @@ def expand_modules(
 
     for something in files_or_modules:
         basename = os.path.basename(something)
-        if (
-            basename in ignore_list
-            or _is_in_ignore_list_re(os.path.basename(something), ignore_list_re)
-            or _is_in_ignore_list_re(something, ignore_list_paths_re)
+        if _is_ignored_file(
+            something, ignore_list, ignore_list_re, ignore_list_paths_re
         ):
             continue
         module_path = get_python_path(something)
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -31,7 +31,7 @@
 )
 from pylint.lint.base_options import _make_linter_options
 from pylint.lint.caching import load_results, save_results
-from pylint.lint.expand_modules import expand_modules
+from pylint.lint.expand_modules import _is_ignored_file, expand_modules
 from pylint.lint.message_state_handler import _MessageStateHandler
 from pylint.lint.parallel import check_parallel
 from pylint.lint.report_functions import (
@@ -564,8 +564,7 @@ def initialize(self) -> None:
             if not msg.may_be_emitted():
                 self._msgs_state[msg.msgid] = False
 
-    @staticmethod
-    def _discover_files(files_or_modules: Sequence[str]) -> Iterator[str]:
+    def _discover_files(self, files_or_modules: Sequence[str]) -> Iterator[str]:
         """Discover python modules and packages in sub-directory.
 
         Returns iterator of paths to discovered modules and packages.
@@ -579,6 +578,16 @@ def _discover_files(files_or_modules: Sequence[str]) -> Iterator[str]:
                     if any(root.startswith(s) for s in skip_subtrees):
                         # Skip subtree of already discovered package.
                         continue
+
+                    if _is_ignored_file(
+                        root,
+                        self.config.ignore,
+                        self.config.ignore_patterns,
+                        self.config.ignore_paths,
+                    ):
+                        skip_subtrees.append(root)
+                        continue
+
                     if "__init__.py" in files:
                         skip_subtrees.append(root)
                         yield root

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/lint/expand_modules.py | 49 | 49 | - | 43 | -
| pylint/lint/expand_modules.py | 64 | 67 | - | 43 | -
| pylint/lint/pylinter.py | 34 | 34 | 78 | 4 | 29207
| pylint/lint/pylinter.py | 567 | 568 | - | 4 | -
| pylint/lint/pylinter.py | 582 | 582 | - | 4 | -


## Problem Statement

```
Pylint does not respect ignores in `--recursive=y` mode
### Bug description

Pylint does not respect the `--ignore`, `--ignore-paths`, or `--ignore-patterns` setting when running in recursive mode. This contradicts the documentation and seriously compromises the usefulness of recursive mode.

### Configuration

_No response_

### Command used

\`\`\`shell
### .a/foo.py
# import re

### bar.py
# import re

pylint --recursive=y .
pylint --recursive=y --ignore=.a .
pylint --recursive=y --ignore-paths=.a .
pylint --recursive=y --ignore-patterns="^\.a" .
\`\`\`


### Pylint output

All of these commands give the same output:

\`\`\`
************* Module bar
bar.py:1:0: C0104: Disallowed name "bar" (disallowed-name)
bar.py:1:0: C0114: Missing module docstring (missing-module-docstring)
bar.py:1:0: W0611: Unused import re (unused-import)
************* Module foo
.a/foo.py:1:0: C0104: Disallowed name "foo" (disallowed-name)
.a/foo.py:1:0: C0114: Missing module docstring (missing-module-docstring)
.a/foo.py:1:0: W0611: Unused import re (unused-import)
\`\`\`


### Expected behavior

`foo.py` should be ignored by all of the above commands, because it is in an ignored directory (even the first command with no ignore setting should skip it, since the default value of `ignore-patterns` is `"^\."`.

For reference, the docs for the various ignore settings from `pylint --help`:

\`\`\`
    --ignore=<file>[,<file>...]
                        Files or directories to be skipped. They should be
                        base names, not paths. [current: CVS]
    --ignore-patterns=<pattern>[,<pattern>...]
                        Files or directories matching the regex patterns are
                        skipped. The regex matches against base names, not
                        paths. The default value ignores emacs file locks
                        [current: ^\.#]
    --ignore-paths=<pattern>[,<pattern>...]
                        Add files or directories matching the regex patterns
                        to the ignore-list. The regex matches against paths
                        and can be in Posix or Windows format. [current: none]
\`\`\`

### Pylint version

\`\`\`shell
pylint 2.13.7
python 3.9.12
\`\`\`


### OS / Environment

_No response_

### Additional dependencies

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/constants.py | 92 | 199| 1388 | 1388 | 2654 | 
| 2 | 2 pylint/pyreverse/main.py | 35 | 203| 892 | 2280 | 4123 | 
| 3 | 3 pylint/__init__.py | 5 | 57| 335 | 2615 | 4852 | 
| 4 | 3 pylint/pyreverse/main.py | 7 | 33| 160 | 2775 | 4852 | 
| 5 | **4 pylint/lint/pylinter.py** | 97 | 215| 982 | 3757 | 14344 | 
| 6 | 5 pylint/config/__init__.py | 5 | 37| 273 | 4030 | 14917 | 
| 7 | 6 pylint/lint/__init__.py | 17 | 47| 183 | 4213 | 15230 | 
| 8 | 7 pylint/config/option.py | 5 | 55| 305 | 4518 | 16974 | 
| 9 | 8 pylint/checkers/stdlib.py | 92 | 243| 1148 | 5666 | 23123 | 
| 10 | 9 pylint/epylint.py | 126 | 143| 114 | 5780 | 24824 | 
| 11 | 10 doc/conf.py | 143 | 246| 596 | 6376 | 26744 | 
| 12 | 11 pylint/lint/run.py | 5 | 27| 148 | 6524 | 28258 | 
| 13 | 12 pylint/checkers/design_analysis.py | 7 | 97| 739 | 7263 | 33129 | 
| 14 | 12 pylint/checkers/stdlib.py | 332 | 433| 1075 | 8338 | 33129 | 
| 15 | 12 pylint/constants.py | 200 | 213| 122 | 8460 | 33129 | 
| 16 | 12 pylint/checkers/design_analysis.py | 98 | 176| 510 | 8970 | 33129 | 
| 17 | 12 pylint/checkers/stdlib.py | 7 | 41| 300 | 9270 | 33129 | 
| 18 | 12 pylint/checkers/stdlib.py | 43 | 89| 556 | 9826 | 33129 | 
| 19 | 13 pylint/testutils/constants.py | 5 | 30| 280 | 10106 | 33474 | 
| 20 | 14 pylint/config/utils.py | 7 | 28| 118 | 10224 | 35426 | 
| 21 | 14 pylint/constants.py | 5 | 90| 608 | 10832 | 35426 | 
| 22 | 14 pylint/epylint.py | 65 | 123| 521 | 11353 | 35426 | 
| 23 | 15 pylint/pyreverse/__init__.py | 1 | 8| 0 | 11353 | 35504 | 
| 24 | 16 pylint/config/arguments_manager.py | 7 | 56| 280 | 11633 | 41490 | 
| 25 | 17 pylint/checkers/spelling.py | 7 | 109| 575 | 12208 | 44955 | 
| 26 | 17 pylint/epylint.py | 201 | 216| 120 | 12328 | 44955 | 
| 27 | 18 pylint/config/option_parser.py | 5 | 54| 371 | 12699 | 45392 | 
| 28 | 18 pylint/pyreverse/main.py | 230 | 250| 147 | 12846 | 45392 | 
| 29 | 19 pylint/checkers/imports.py | 288 | 403| 707 | 13553 | 53235 | 
| 30 | 20 pylint/checkers/base/basic_error_checker.py | 99 | 207| 952 | 14505 | 57751 | 
| 31 | 21 pylint/checkers/format.py | 597 | 617| 197 | 14702 | 63720 | 
| 32 | 22 pylint/checkers/typecheck.py | 800 | 952| 1076 | 15778 | 80905 | 
| 33 | 23 pylint/pyreverse/utils.py | 7 | 108| 541 | 16319 | 82938 | 
| 34 | 24 doc/data/messages/r/reimported/bad.py | 1 | 3| 0 | 16319 | 82950 | 
| 35 | 25 pylint/pyreverse/dot_printer.py | 7 | 33| 236 | 16555 | 84267 | 
| 36 | 26 pylint/testutils/pyreverse.py | 5 | 66| 467 | 17022 | 85084 | 
| 37 | 27 doc/data/messages/c/confusing-consecutive-elif/bad.py | 1 | 7| 0 | 17022 | 85142 | 
| 38 | 28 pylint/checkers/base/docstring_checker.py | 7 | 43| 203 | 17225 | 86745 | 
| 39 | 28 pylint/checkers/stdlib.py | 246 | 299| 260 | 17485 | 86745 | 
| 40 | 29 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 17485 | 86768 | 
| 41 | 29 pylint/pyreverse/main.py | 206 | 228| 201 | 17686 | 86768 | 
| 42 | **29 pylint/lint/pylinter.py** | 481 | 518| 308 | 17994 | 86768 | 
| 43 | 30 doc/data/messages/u/ungrouped-imports/bad.py | 1 | 6| 0 | 17994 | 86798 | 
| 44 | 31 pylint/interfaces.py | 7 | 49| 291 | 18285 | 87701 | 
| 45 | 32 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 18285 | 87720 | 
| 46 | 33 pylint/checkers/classes/class_checker.py | 7 | 65| 418 | 18703 | 104153 | 
| 47 | 33 doc/conf.py | 16 | 141| 963 | 19666 | 104153 | 
| 48 | 33 pylint/epylint.py | 146 | 198| 491 | 20157 | 104153 | 
| 49 | 34 pylint/checkers/__init__.py | 44 | 64| 117 | 20274 | 105151 | 
| 50 | 35 pylint/testutils/functional/__init__.py | 5 | 24| 129 | 20403 | 105345 | 
| 51 | 35 pylint/config/option.py | 149 | 182| 313 | 20716 | 105345 | 
| 52 | 36 pylint/checkers/exceptions.py | 7 | 29| 123 | 20839 | 109850 | 
| 53 | 37 pylint/checkers/similar.py | 20 | 84| 411 | 21250 | 117474 | 
| 54 | 37 pylint/config/utils.py | 149 | 213| 547 | 21797 | 117474 | 
| 55 | 38 doc/data/messages/w/wildcard-import/bad.py | 1 | 2| 0 | 21797 | 117485 | 
| 56 | 38 pylint/checkers/format.py | 14 | 114| 718 | 22515 | 117485 | 
| 57 | 39 pylint/extensions/bad_builtin.py | 7 | 22| 107 | 22622 | 117996 | 
| 58 | 40 doc/data/messages/r/reimported/good.py | 1 | 2| 0 | 22622 | 117999 | 
| 59 | 41 pylint/checkers/refactoring/refactoring_checker.py | 5 | 63| 392 | 23014 | 135342 | 
| 60 | 42 pylint/checkers/utils.py | 7 | 70| 442 | 23456 | 148705 | 
| 61 | **43 pylint/lint/expand_modules.py** | 5 | 23| 133 | 23589 | 149883 | 
| 62 | 44 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 23589 | 149897 | 
| 63 | 45 doc/data/messages/g/global-at-module-level/bad.py | 1 | 3| 0 | 23589 | 149912 | 
| 64 | 46 doc/data/messages/u/ungrouped-imports/good.py | 1 | 6| 0 | 23589 | 149932 | 
| 65 | 46 pylint/checkers/stdlib.py | 435 | 481| 510 | 24099 | 149932 | 
| 66 | 47 pylint/checkers/base_checker.py | 5 | 32| 166 | 24265 | 152204 | 
| 67 | 48 doc/data/messages/a/arguments-out-of-order/bad.py | 1 | 14| 0 | 24265 | 152288 | 
| 68 | 49 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 75| 495 | 24760 | 154074 | 
| 69 | 49 pylint/checkers/utils.py | 71 | 187| 710 | 25470 | 154074 | 
| 70 | 50 pylint/checkers/base/name_checker/checker.py | 7 | 45| 257 | 25727 | 159129 | 
| 71 | 51 doc/data/messages/a/arguments-renamed/bad.py | 1 | 14| 0 | 25727 | 159231 | 
| 72 | 52 doc/data/messages/r/redefined-loop-name/good.py | 1 | 3| 0 | 25727 | 159244 | 
| 73 | 52 pylint/checkers/typecheck.py | 7 | 92| 472 | 26199 | 159244 | 
| 74 | 53 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 5| 0 | 26199 | 159255 | 
| 75 | 54 pylint/pyreverse/vcg_printer.py | 14 | 115| 706 | 26905 | 161450 | 
| 76 | 55 pylint/checkers/base/basic_checker.py | 104 | 251| 1362 | 28267 | 168648 | 
| 77 | 56 pylint/checkers/strings.py | 7 | 59| 342 | 28609 | 176884 | 
| **-> 78 <-** | **56 pylint/lint/pylinter.py** | 5 | 96| 598 | 29207 | 176884 | 
| 79 | 57 pylint/utils/__init__.py | 9 | 56| 268 | 29475 | 177237 | 
| 80 | 58 pylint/checkers/refactoring/recommendation_checker.py | 5 | 81| 680 | 30155 | 180667 | 
| 81 | 58 pylint/checkers/imports.py | 763 | 787| 269 | 30424 | 180667 | 
| 82 | 59 doc/data/messages/u/unreachable/bad.py | 1 | 4| 0 | 30424 | 180690 | 
| 83 | 60 pylint/extensions/private_import.py | 7 | 39| 215 | 30639 | 182858 | 
| 84 | 61 pylint/extensions/typing.py | 78 | 161| 762 | 31401 | 186747 | 
| 85 | 62 doc/data/messages/a/arguments-out-of-order/good.py | 1 | 12| 0 | 31401 | 186819 | 
| 86 | 63 doc/data/messages/c/consider-using-with/bad.py | 1 | 4| 0 | 31401 | 186849 | 
| 87 | 64 pylint/testutils/__init__.py | 7 | 36| 228 | 31629 | 187151 | 
| 88 | 65 doc/data/messages/b/broad-except/good.py | 1 | 5| 0 | 31629 | 187168 | 
| 89 | 66 doc/data/messages/u/unreachable/good.py | 1 | 4| 0 | 31629 | 187185 | 
| 90 | 67 doc/data/messages/c/confusing-consecutive-elif/good.py | 1 | 23| 131 | 31760 | 187316 | 
| 91 | 68 doc/data/messages/r/redefined-loop-name/bad.py | 1 | 3| 0 | 31760 | 187335 | 
| 92 | 68 pylint/checkers/typecheck.py | 95 | 137| 364 | 32124 | 187335 | 
| 93 | 69 pylint/checkers/unsupported_version.py | 9 | 62| 420 | 32544 | 188007 | 
| 94 | 70 pylint/lint/base_options.py | 7 | 392| 188 | 32732 | 191941 | 
| 95 | 71 doc/data/messages/u/undefined-all-variable/good.py | 1 | 5| 0 | 32732 | 191960 | 
| 96 | 72 pylint/checkers/base/__init__.py | 5 | 47| 281 | 33013 | 192306 | 
| 97 | 73 pylint/extensions/redefined_loop_name.py | 7 | 33| 165 | 33178 | 192979 | 
| 98 | 74 pylint/config/config_initialization.py | 5 | 106| 805 | 33983 | 193850 | 
| 99 | 75 pylint/checkers/base/name_checker/naming_style.py | 5 | 24| 147 | 34130 | 195401 | 
| 100 | 76 doc/data/messages/g/global-at-module-level/good.py | 1 | 2| 0 | 34130 | 195406 | 
| 101 | 77 doc/data/messages/m/missing-yield-doc/bad.py | 1 | 10| 0 | 34130 | 195470 | 
| 102 | 78 doc/data/messages/u/undefined-all-variable/bad.py | 1 | 5| 0 | 34130 | 195496 | 
| 103 | 78 pylint/checkers/refactoring/refactoring_checker.py | 449 | 502| 382 | 34512 | 195496 | 
| 104 | 78 pylint/config/option.py | 184 | 202| 179 | 34691 | 195496 | 
| 105 | 79 doc/data/messages/b/broad-except/bad.py | 1 | 5| 0 | 34691 | 195519 | 
| 106 | 79 pylint/checkers/imports.py | 119 | 145| 207 | 34898 | 195519 | 
| 107 | 80 doc/exts/pylint_features.py | 1 | 42| 241 | 35139 | 195839 | 
| 108 | 81 pylint/checkers/misc.py | 7 | 41| 213 | 35352 | 197143 | 
| 109 | 82 examples/custom_raw.py | 1 | 46| 272 | 35624 | 197415 | 
| 110 | 83 pylint/lint/utils.py | 66 | 85| 144 | 35768 | 198159 | 
| 111 | 84 doc/data/messages/a/arguments-renamed/good.py | 1 | 14| 0 | 35768 | 198255 | 
| 112 | 85 doc/data/messages/c/consider-using-sys-exit/bad.py | 1 | 5| 0 | 35768 | 198297 | 


### Hint

```
I suppose that ignored paths needs to be filtered here:
https://github.com/PyCQA/pylint/blob/0220a39f6d4dddd1bf8f2f6d83e11db58a093fbe/pylint/lint/pylinter.py#L676
```

## Patch

```diff
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -46,6 +46,20 @@ def _is_in_ignore_list_re(element: str, ignore_list_re: list[Pattern[str]]) -> b
     return any(file_pattern.match(element) for file_pattern in ignore_list_re)
 
 
+def _is_ignored_file(
+    element: str,
+    ignore_list: list[str],
+    ignore_list_re: list[Pattern[str]],
+    ignore_list_paths_re: list[Pattern[str]],
+) -> bool:
+    basename = os.path.basename(element)
+    return (
+        basename in ignore_list
+        or _is_in_ignore_list_re(basename, ignore_list_re)
+        or _is_in_ignore_list_re(element, ignore_list_paths_re)
+    )
+
+
 def expand_modules(
     files_or_modules: Sequence[str],
     ignore_list: list[str],
@@ -61,10 +75,8 @@ def expand_modules(
 
     for something in files_or_modules:
         basename = os.path.basename(something)
-        if (
-            basename in ignore_list
-            or _is_in_ignore_list_re(os.path.basename(something), ignore_list_re)
-            or _is_in_ignore_list_re(something, ignore_list_paths_re)
+        if _is_ignored_file(
+            something, ignore_list, ignore_list_re, ignore_list_paths_re
         ):
             continue
         module_path = get_python_path(something)
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -31,7 +31,7 @@
 )
 from pylint.lint.base_options import _make_linter_options
 from pylint.lint.caching import load_results, save_results
-from pylint.lint.expand_modules import expand_modules
+from pylint.lint.expand_modules import _is_ignored_file, expand_modules
 from pylint.lint.message_state_handler import _MessageStateHandler
 from pylint.lint.parallel import check_parallel
 from pylint.lint.report_functions import (
@@ -564,8 +564,7 @@ def initialize(self) -> None:
             if not msg.may_be_emitted():
                 self._msgs_state[msg.msgid] = False
 
-    @staticmethod
-    def _discover_files(files_or_modules: Sequence[str]) -> Iterator[str]:
+    def _discover_files(self, files_or_modules: Sequence[str]) -> Iterator[str]:
         """Discover python modules and packages in sub-directory.
 
         Returns iterator of paths to discovered modules and packages.
@@ -579,6 +578,16 @@ def _discover_files(files_or_modules: Sequence[str]) -> Iterator[str]:
                     if any(root.startswith(s) for s in skip_subtrees):
                         # Skip subtree of already discovered package.
                         continue
+
+                    if _is_ignored_file(
+                        root,
+                        self.config.ignore,
+                        self.config.ignore_patterns,
+                        self.config.ignore_paths,
+                    ):
+                        skip_subtrees.append(root)
+                        continue
+
                     if "__init__.py" in files:
                         skip_subtrees.append(root)
                         yield root

```

## Test Patch

```diff
diff --git a/tests/lint/unittest_lint.py b/tests/lint/unittest_lint.py
--- a/tests/lint/unittest_lint.py
+++ b/tests/lint/unittest_lint.py
@@ -864,6 +864,49 @@ def test_by_module_statement_value(initialized_linter: PyLinter) -> None:
         assert module_stats["statement"] == linter2.stats.statement
 
 
+@pytest.mark.parametrize(
+    "ignore_parameter,ignore_parameter_value",
+    [
+        ("--ignore", "failing.py"),
+        ("--ignore", "ignored_subdirectory"),
+        ("--ignore-patterns", "failing.*"),
+        ("--ignore-patterns", "ignored_*"),
+        ("--ignore-paths", ".*directory/ignored.*"),
+        ("--ignore-paths", ".*ignored.*/failing.*"),
+    ],
+)
+def test_recursive_ignore(ignore_parameter, ignore_parameter_value) -> None:
+    run = Run(
+        [
+            "--recursive",
+            "y",
+            ignore_parameter,
+            ignore_parameter_value,
+            join(REGRTEST_DATA_DIR, "directory"),
+        ],
+        exit=False,
+    )
+
+    linted_files = run.linter._iterate_file_descrs(
+        tuple(run.linter._discover_files([join(REGRTEST_DATA_DIR, "directory")]))
+    )
+    linted_file_paths = [file_item.filepath for file_item in linted_files]
+
+    ignored_file = os.path.abspath(
+        join(REGRTEST_DATA_DIR, "directory", "ignored_subdirectory", "failing.py")
+    )
+    assert ignored_file not in linted_file_paths
+
+    for regrtest_data_module in (
+        ("directory", "subdirectory", "subsubdirectory", "module.py"),
+        ("directory", "subdirectory", "module.py"),
+        ("directory", "package", "module.py"),
+        ("directory", "package", "subpackage", "module.py"),
+    ):
+        module = os.path.abspath(join(REGRTEST_DATA_DIR, *regrtest_data_module))
+    assert module in linted_file_paths
+
+
 def test_import_sibling_module_from_namespace(initialized_linter: PyLinter) -> None:
     """If the parent directory above `namespace` is on sys.path, ensure that
     modules under `namespace` can import each other without raising `import-error`."""
diff --git a/tests/regrtest_data/directory/ignored_subdirectory/failing.py b/tests/regrtest_data/directory/ignored_subdirectory/failing.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/directory/ignored_subdirectory/failing.py
@@ -0,0 +1 @@
+import re
diff --git a/tests/test_self.py b/tests/test_self.py
--- a/tests/test_self.py
+++ b/tests/test_self.py
@@ -1228,17 +1228,91 @@ def test_max_inferred_for_complicated_class_hierarchy() -> None:
         assert not ex.value.code % 2
 
     def test_regression_recursive(self):
+        """Tests if error is raised when linter is executed over directory not using --recursive=y"""
         self._test_output(
             [join(HERE, "regrtest_data", "directory", "subdirectory"), "--recursive=n"],
             expected_output="No such file or directory",
         )
 
     def test_recursive(self):
+        """Tests if running linter over directory using --recursive=y"""
         self._runtest(
             [join(HERE, "regrtest_data", "directory", "subdirectory"), "--recursive=y"],
             code=0,
         )
 
+    def test_ignore_recursive(self):
+        """Tests recursive run of linter ignoring directory using --ignore parameter.
+
+        Ignored directory contains files yielding lint errors. If directory is not ignored
+        test would fail due these errors.
+        """
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore=ignored_subdirectory",
+            ],
+            code=0,
+        )
+
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore=failing.py",
+            ],
+            code=0,
+        )
+
+    def test_ignore_pattern_recursive(self):
+        """Tests recursive run of linter ignoring directory using --ignore-parameter parameter.
+
+        Ignored directory contains files yielding lint errors. If directory is not ignored
+        test would fail due these errors.
+        """
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore-pattern=ignored_.*",
+            ],
+            code=0,
+        )
+
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore-pattern=failing.*",
+            ],
+            code=0,
+        )
+
+    def test_ignore_path_recursive(self):
+        """Tests recursive run of linter ignoring directory using --ignore-path parameter.
+
+        Ignored directory contains files yielding lint errors. If directory is not ignored
+        test would fail due these errors.
+        """
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore-path=.*ignored.*",
+            ],
+            code=0,
+        )
+
+        self._runtest(
+            [
+                join(HERE, "regrtest_data", "directory"),
+                "--recursive=y",
+                "--ignore-path=.*failing.*",
+            ],
+            code=0,
+        )
+
     def test_recursive_current_dir(self):
         with _test_sys_path():
             # pytest is including directory HERE/regrtest_data to sys.path which causes
@@ -1249,7 +1323,7 @@ def test_recursive_current_dir(self):
                 if not os.path.basename(path) == "regrtest_data"
             ]
             with _test_cwd():
-                os.chdir(join(HERE, "regrtest_data", "directory"))
+                os.chdir(join(HERE, "regrtest_data", "directory", "subdirectory"))
                 self._runtest(
                     [".", "--recursive=y"],
                     code=0,

```


## Code snippets

### 1 - pylint/constants.py:

Start line: 92, End line: 199

```python
DELETED_MESSAGES = [
    # Everything until the next comment is from the
    # PY3K+ checker, see https://github.com/PyCQA/pylint/pull/4942
    DeletedMessage("W1601", "apply-builtin"),
    DeletedMessage("E1601", "print-statement"),
    DeletedMessage("E1602", "parameter-unpacking"),
    DeletedMessage(
        "E1603", "unpacking-in-except", [("W0712", "old-unpacking-in-except")]
    ),
    DeletedMessage("E1604", "old-raise-syntax", [("W0121", "old-old-raise-syntax")]),
    DeletedMessage("E1605", "backtick", [("W0333", "old-backtick")]),
    DeletedMessage("E1609", "import-star-module-level"),
    DeletedMessage("W1601", "apply-builtin"),
    DeletedMessage("W1602", "basestring-builtin"),
    DeletedMessage("W1603", "buffer-builtin"),
    DeletedMessage("W1604", "cmp-builtin"),
    DeletedMessage("W1605", "coerce-builtin"),
    DeletedMessage("W1606", "execfile-builtin"),
    DeletedMessage("W1607", "file-builtin"),
    DeletedMessage("W1608", "long-builtin"),
    DeletedMessage("W1609", "raw_input-builtin"),
    DeletedMessage("W1610", "reduce-builtin"),
    DeletedMessage("W1611", "standarderror-builtin"),
    DeletedMessage("W1612", "unicode-builtin"),
    DeletedMessage("W1613", "xrange-builtin"),
    DeletedMessage("W1614", "coerce-method"),
    DeletedMessage("W1615", "delslice-method"),
    DeletedMessage("W1616", "getslice-method"),
    DeletedMessage("W1617", "setslice-method"),
    DeletedMessage("W1618", "no-absolute-import"),
    DeletedMessage("W1619", "old-division"),
    DeletedMessage("W1620", "dict-iter-method"),
    DeletedMessage("W1621", "dict-view-method"),
    DeletedMessage("W1622", "next-method-called"),
    DeletedMessage("W1623", "metaclass-assignment"),
    DeletedMessage(
        "W1624", "indexing-exception", [("W0713", "old-indexing-exception")]
    ),
    DeletedMessage("W1625", "raising-string", [("W0701", "old-raising-string")]),
    DeletedMessage("W1626", "reload-builtin"),
    DeletedMessage("W1627", "oct-method"),
    DeletedMessage("W1628", "hex-method"),
    DeletedMessage("W1629", "nonzero-method"),
    DeletedMessage("W1630", "cmp-method"),
    DeletedMessage("W1632", "input-builtin"),
    DeletedMessage("W1633", "round-builtin"),
    DeletedMessage("W1634", "intern-builtin"),
    DeletedMessage("W1635", "unichr-builtin"),
    DeletedMessage(
        "W1636", "map-builtin-not-iterating", [("W1631", "implicit-map-evaluation")]
    ),
    DeletedMessage("W1637", "zip-builtin-not-iterating"),
    DeletedMessage("W1638", "range-builtin-not-iterating"),
    DeletedMessage("W1639", "filter-builtin-not-iterating"),
    DeletedMessage("W1640", "using-cmp-argument"),
    DeletedMessage("W1642", "div-method"),
    DeletedMessage("W1643", "idiv-method"),
    DeletedMessage("W1644", "rdiv-method"),
    DeletedMessage("W1645", "exception-message-attribute"),
    DeletedMessage("W1646", "invalid-str-codec"),
    DeletedMessage("W1647", "sys-max-int"),
    DeletedMessage("W1648", "bad-python3-import"),
    DeletedMessage("W1649", "deprecated-string-function"),
    DeletedMessage("W1650", "deprecated-str-translate-call"),
    DeletedMessage("W1651", "deprecated-itertools-function"),
    DeletedMessage("W1652", "deprecated-types-field"),
    DeletedMessage("W1653", "next-method-defined"),
    DeletedMessage("W1654", "dict-items-not-iterating"),
    DeletedMessage("W1655", "dict-keys-not-iterating"),
    DeletedMessage("W1656", "dict-values-not-iterating"),
    DeletedMessage("W1657", "deprecated-operator-function"),
    DeletedMessage("W1658", "deprecated-urllib-function"),
    DeletedMessage("W1659", "xreadlines-attribute"),
    DeletedMessage("W1660", "deprecated-sys-function"),
    DeletedMessage("W1661", "exception-escape"),
    DeletedMessage("W1662", "comprehension-escape"),
    # https://github.com/PyCQA/pylint/pull/3578
    DeletedMessage("W0312", "mixed-indentation"),
    # https://github.com/PyCQA/pylint/pull/3577
    DeletedMessage(
        "C0326",
        "bad-whitespace",
        [
            ("C0323", "no-space-after-operator"),
            ("C0324", "no-space-after-comma"),
            ("C0322", "no-space-before-operator"),
        ],
    ),
    # https://github.com/PyCQA/pylint/pull/3571
    DeletedMessage("C0330", "bad-continuation"),
    # No PR
    DeletedMessage("R0921", "abstract-class-not-used"),
    # https://github.com/PyCQA/pylint/pull/3577
    DeletedMessage("C0326", "bad-whitespace"),
    # Pylint 1.4.3
    DeletedMessage("W0142", "star-args"),
    # https://github.com/PyCQA/pylint/issues/2409
    DeletedMessage("W0232", "no-init"),
    # https://github.com/PyCQA/pylint/pull/6421
    DeletedMessage("W0111", "assign-to-new-keyword"),
]


# ignore some messages when emitting useless-suppression:
# - cyclic-import: can show false positives due to incomplete context
# - deprecated-{module, argument, class, method, decorator}:
#   can cause false positives for multi-interpreter projects
#   when linting with an interpreter on a lower python version
```
### 2 - pylint/pyreverse/main.py:

Start line: 35, End line: 203

```python
OPTIONS: Options = (
    (
        "filter-mode",
        dict(
            short="f",
            default="PUB_ONLY",
            dest="mode",
            type="string",
            action="store",
            metavar="<mode>",
            help="""filter attributes and functions according to
    <mode>. Correct modes are :
                            'PUB_ONLY' filter all non public attributes
                                [DEFAULT], equivalent to PRIVATE+SPECIAL_A
                            'ALL' no filter
                            'SPECIAL' filter Python special functions
                                except constructor
                            'OTHER' filter protected and private
                                attributes""",
        ),
    ),
    (
        "class",
        dict(
            short="c",
            action="extend",
            metavar="<class>",
            type="csv",
            dest="classes",
            default=None,
            help="create a class diagram with all classes related to <class>;\
 this uses by default the options -ASmy",
        ),
    ),
    (
        "show-ancestors",
        dict(
            short="a",
            action="store",
            metavar="<ancestor>",
            type="int",
            default=None,
            help="show <ancestor> generations of ancestor classes not in <projects>",
        ),
    ),
    (
        "all-ancestors",
        dict(
            short="A",
            default=None,
            action="store_true",
            help="show all ancestors off all classes in <projects>",
        ),
    ),
    (
        "show-associated",
        dict(
            short="s",
            action="store",
            metavar="<association_level>",
            type="int",
            default=None,
            help="show <association_level> levels of associated classes not in <projects>",
        ),
    ),
    (
        "all-associated",
        dict(
            short="S",
            default=None,
            action="store_true",
            help="show recursively all associated off all associated classes",
        ),
    ),
    (
        "show-builtin",
        dict(
            short="b",
            action="store_true",
            default=False,
            help="include builtin objects in representation of classes",
        ),
    ),
    (
        "module-names",
        dict(
            short="m",
            default=None,
            type="yn",
            metavar="<y or n>",
            help="include module name in representation of classes",
        ),
    ),
    (
        "only-classnames",
        dict(
            short="k",
            action="store_true",
            default=False,
            help="don't show attributes and methods in the class boxes; this disables -f values",
        ),
    ),
    (
        "output",
        dict(
            short="o",
            dest="output_format",
            action="store",
            default="dot",
            metavar="<format>",
            type="string",
            help=(
                f"create a *.<format> output file if format is available. Available formats are: {', '.join(DIRECTLY_SUPPORTED_FORMATS)}. "
                f"Any other format will be tried to create by means of the 'dot' command line tool, which requires a graphviz installation."
            ),
        ),
    ),
    (
        "colorized",
        dict(
            dest="colorized",
            action="store_true",
            default=False,
            help="Use colored output. Classes/modules of the same package get the same color.",
        ),
    ),
    (
        "max-color-depth",
        dict(
            dest="max_color_depth",
            action="store",
            default=2,
            metavar="<depth>",
            type="int",
            help="Use separate colors up to package depth of <depth>",
        ),
    ),
    (
        "ignore",
        dict(
            type="csv",
            metavar="<file[,file...]>",
            dest="ignore_list",
            default=("CVS",),
            help="Files or directories to be skipped. They should be base names, not paths.",
        ),
    ),
    (
        "project",
        dict(
            default="",
            type="string",
            short="p",
            metavar="<project name>",
            help="set the project name.",
        ),
    ),
    (
        "output-directory",
        dict(
            default="",
            type="path",
            short="d",
            action="store",
            metavar="<output_directory>",
            help="set the output directory path.",
        ),
    ),
)
```
### 3 - pylint/__init__.py:

Start line: 5, End line: 57

```python
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import NoReturn

from pylint.__pkginfo__ import __version__

# pylint: disable=import-outside-toplevel


def run_pylint(argv: Sequence[str] | None = None) -> None:
    """Run pylint.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.lint import Run as PylintRun

    try:
        PylintRun(argv or sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)


def run_epylint(argv: Sequence[str] | None = None) -> NoReturn:
    """Run epylint.

    argv can be a list of strings normally supplied as arguments on the command line
    """
    from pylint.epylint import Run as EpylintRun

    EpylintRun(argv)


def run_pyreverse(argv: Sequence[str] | None = None) -> NoReturn:  # type: ignore[misc]
    """Run pyreverse.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.pyreverse.main import Run as PyreverseRun

    PyreverseRun(argv or sys.argv[1:])


def run_symilar(argv: Sequence[str] | None = None) -> NoReturn:
    """Run symilar.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.checkers.similar import Run as SimilarRun

    SimilarRun(argv or sys.argv[1:])
```
### 4 - pylint/pyreverse/main.py:

Start line: 7, End line: 33

```python
from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import NoReturn

from pylint.config.arguments_manager import _ArgumentsManager
from pylint.config.arguments_provider import _ArgumentsProvider
from pylint.lint.utils import fix_import_path
from pylint.pyreverse import writer
from pylint.pyreverse.diadefslib import DiadefsHandler
from pylint.pyreverse.inspector import Linker, project_from_files
from pylint.pyreverse.utils import (
    check_graphviz_availability,
    check_if_graphviz_supports_format,
    insert_default_options,
)
from pylint.typing import Options

DIRECTLY_SUPPORTED_FORMATS = (
    "dot",
    "vcg",
    "puml",
    "plantuml",
    "mmd",
    "html",
)
```
### 5 - pylint/lint/pylinter.py:

Start line: 97, End line: 215

```python
MSGS: dict[str, MessageDefinitionTuple] = {
    "F0001": (
        "%s",
        "fatal",
        "Used when an error occurred preventing the analysis of a \
              module (unable to find it for instance).",
        {"scope": WarningScope.LINE},
    ),
    "F0002": (
        "%s: %s",
        "astroid-error",
        "Used when an unexpected error occurred while building the "
        "Astroid  representation. This is usually accompanied by a "
        "traceback. Please report such errors !",
        {"scope": WarningScope.LINE},
    ),
    "F0010": (
        "error while code parsing: %s",
        "parse-error",
        "Used when an exception occurred while building the Astroid "
        "representation which could be handled by astroid.",
        {"scope": WarningScope.LINE},
    ),
    "F0011": (
        "error while parsing the configuration: %s",
        "config-parse-error",
        "Used when an exception occurred while parsing a pylint configuration file.",
        {"scope": WarningScope.LINE},
    ),
    "I0001": (
        "Unable to run raw checkers on built-in module %s",
        "raw-checker-failed",
        "Used to inform that a built-in module has not been checked "
        "using the raw checkers.",
        {"scope": WarningScope.LINE},
    ),
    "I0010": (
        "Unable to consider inline option %r",
        "bad-inline-option",
        "Used when an inline option is either badly formatted or can't "
        "be used inside modules.",
        {"scope": WarningScope.LINE},
    ),
    "I0011": (
        "Locally disabling %s (%s)",
        "locally-disabled",
        "Used when an inline option disables a message or a messages category.",
        {"scope": WarningScope.LINE},
    ),
    "I0013": (
        "Ignoring entire file",
        "file-ignored",
        "Used to inform that the file will not be checked",
        {"scope": WarningScope.LINE},
    ),
    "I0020": (
        "Suppressed %s (from line %d)",
        "suppressed-message",
        "A message was triggered on a line, but suppressed explicitly "
        "by a disable= comment in the file. This message is not "
        "generated for messages that are ignored due to configuration "
        "settings.",
        {"scope": WarningScope.LINE},
    ),
    "I0021": (
        "Useless suppression of %s",
        "useless-suppression",
        "Reported when a message is explicitly disabled for a line or "
        "a block of code, but never triggered.",
        {"scope": WarningScope.LINE},
    ),
    "I0022": (
        'Pragma "%s" is deprecated, use "%s" instead',
        "deprecated-pragma",
        "Some inline pylint options have been renamed or reworked, "
        "only the most recent form should be used. "
        "NOTE:skip-all is only available with pylint >= 0.26",
        {
            "old_names": [("I0014", "deprecated-disable-all")],
            "scope": WarningScope.LINE,
        },
    ),
    "E0001": (
        "%s",
        "syntax-error",
        "Used when a syntax error is raised for a module.",
        {"scope": WarningScope.LINE},
    ),
    "E0011": (
        "Unrecognized file option %r",
        "unrecognized-inline-option",
        "Used when an unknown inline option is encountered.",
        {"scope": WarningScope.LINE},
    ),
    "E0012": (
        "Bad option value for %s",
        "bad-option-value",
        "Used when a bad value for an inline option is encountered.",
        {"scope": WarningScope.LINE},
    ),
    "E0013": (
        "Plugin '%s' is impossible to load, is it installed ? ('%s')",
        "bad-plugin-value",
        "Used when a bad value is used in 'load-plugins'.",
        {"scope": WarningScope.LINE},
    ),
    "E0014": (
        "Out-of-place setting encountered in top level configuration-section '%s' : '%s'",
        "bad-configuration-section",
        "Used when we detect a setting in the top level of a toml configuration that shouldn't be there.",
        {"scope": WarningScope.LINE},
    ),
    "E0015": (
        "Unrecognized option found: %s",
        "unrecognized-option",
        "Used when we detect an option that we do not recognize.",
        {"scope": WarningScope.LINE},
    ),
}
```
### 6 - pylint/config/__init__.py:

Start line: 5, End line: 37

```python
from __future__ import annotations

__all__ = [
    "ConfigurationMixIn",  # Deprecated
    "find_default_config_files",
    "find_pylintrc",  # Deprecated
    "Option",  # Deprecated
    "OptionsManagerMixIn",  # Deprecated
    "OptionParser",  # Deprecated
    "OptionsProviderMixIn",  # Deprecated
    "UnsupportedAction",  # Deprecated
    "PYLINTRC",
    "USER_HOME",  # Compatibility with the old API
    "PYLINT_HOME",  # Compatibility with the old API
    "save_results",  # Compatibility with the old API # Deprecated
    "load_results",  # Compatibility with the old API # Deprecated
]

import warnings

from pylint.config.arguments_provider import UnsupportedAction
from pylint.config.configuration_mixin import ConfigurationMixIn
from pylint.config.environment_variable import PYLINTRC
from pylint.config.find_default_config_files import (
    find_default_config_files,
    find_pylintrc,
)
from pylint.config.option import Option
from pylint.config.option_manager_mixin import OptionsManagerMixIn
from pylint.config.option_parser import OptionParser
from pylint.config.options_provider_mixin import OptionsProviderMixIn
from pylint.constants import PYLINT_HOME, USER_HOME
from pylint.utils import LinterStats
```
### 7 - pylint/lint/__init__.py:

Start line: 17, End line: 47

```python
import sys

from pylint.config.exceptions import ArgumentPreprocessingError
from pylint.lint.caching import load_results, save_results
from pylint.lint.parallel import check_parallel
from pylint.lint.pylinter import PyLinter
from pylint.lint.report_functions import (
    report_messages_by_module_stats,
    report_messages_stats,
    report_total_messages_stats,
)
from pylint.lint.run import Run
from pylint.lint.utils import _patch_sys_path, fix_import_path

__all__ = [
    "check_parallel",
    "PyLinter",
    "report_messages_by_module_stats",
    "report_messages_stats",
    "report_total_messages_stats",
    "Run",
    "ArgumentPreprocessingError",
    "_patch_sys_path",
    "fix_import_path",
    "save_results",
    "load_results",
]

if __name__ == "__main__":
    Run(sys.argv[1:])
```
### 8 - pylint/config/option.py:

Start line: 5, End line: 55

```python
from __future__ import annotations

import copy
import optparse  # pylint: disable=deprecated-module
import pathlib
import re
import warnings
from re import Pattern

from pylint import utils


# pylint: disable=unused-argument
def _csv_validator(_, name, value):
    return utils._check_csv(value)


# pylint: disable=unused-argument
def _regexp_validator(_, name, value):
    if hasattr(value, "pattern"):
        return value
    return re.compile(value)


# pylint: disable=unused-argument
def _regexp_csv_validator(_, name, value):
    return [_regexp_validator(_, name, val) for val in _csv_validator(_, name, value)]


def _regexp_paths_csv_validator(
    _, name: str, value: str | list[Pattern[str]]
) -> list[Pattern[str]]:
    if isinstance(value, list):
        return value
    patterns = []
    for val in _csv_validator(_, name, value):
        patterns.append(
            re.compile(
                str(pathlib.PureWindowsPath(val)).replace("\\", "\\\\")
                + "|"
                + pathlib.PureWindowsPath(val).as_posix()
            )
        )
    return patterns


def _choice_validator(choices, name, value):
    if value not in choices:
        msg = "option %s: invalid value: %r, should be in %s"
        raise optparse.OptionValueError(msg % (name, value, choices))
    return value
```
### 9 - pylint/checkers/stdlib.py:

Start line: 92, End line: 243

```python
DEPRECATED_METHODS: dict = {
    0: {
        "cgi.parse_qs",
        "cgi.parse_qsl",
        "ctypes.c_buffer",
        "distutils.command.register.register.check_metadata",
        "distutils.command.sdist.sdist.check_metadata",
        "tkinter.Misc.tk_menuBar",
        "tkinter.Menu.tk_bindForTraversal",
    },
    2: {
        (2, 6, 0): {
            "commands.getstatus",
            "os.popen2",
            "os.popen3",
            "os.popen4",
            "macostools.touched",
        },
        (2, 7, 0): {
            "unittest.case.TestCase.assertEquals",
            "unittest.case.TestCase.assertNotEquals",
            "unittest.case.TestCase.assertAlmostEquals",
            "unittest.case.TestCase.assertNotAlmostEquals",
            "unittest.case.TestCase.assert_",
            "xml.etree.ElementTree.Element.getchildren",
            "xml.etree.ElementTree.Element.getiterator",
            "xml.etree.ElementTree.XMLParser.getiterator",
            "xml.etree.ElementTree.XMLParser.doctype",
        },
    },
    3: {
        (3, 0, 0): {
            "inspect.getargspec",
            "failUnlessEqual",
            "assertEquals",
            "failIfEqual",
            "assertNotEquals",
            "failUnlessAlmostEqual",
            "assertAlmostEquals",
            "failIfAlmostEqual",
            "assertNotAlmostEquals",
            "failUnless",
            "assert_",
            "failUnlessRaises",
            "failIf",
            "assertRaisesRegexp",
            "assertRegexpMatches",
            "assertNotRegexpMatches",
        },
        (3, 1, 0): {
            "base64.encodestring",
            "base64.decodestring",
            "ntpath.splitunc",
            "os.path.splitunc",
            "os.stat_float_times",
            "turtle.RawTurtle.settiltangle",
        },
        (3, 2, 0): {
            "cgi.escape",
            "configparser.RawConfigParser.readfp",
            "xml.etree.ElementTree.Element.getchildren",
            "xml.etree.ElementTree.Element.getiterator",
            "xml.etree.ElementTree.XMLParser.getiterator",
            "xml.etree.ElementTree.XMLParser.doctype",
        },
        (3, 3, 0): {
            "inspect.getmoduleinfo",
            "logging.warn",
            "logging.Logger.warn",
            "logging.LoggerAdapter.warn",
            "nntplib._NNTPBase.xpath",
            "platform.popen",
            "sqlite3.OptimizedUnicode",
            "time.clock",
        },
        (3, 4, 0): {
            "importlib.find_loader",
            "importlib.abc.Loader.load_module",
            "importlib.abc.Loader.module_repr",
            "importlib.abc.PathEntryFinder.find_loader",
            "importlib.abc.PathEntryFinder.find_module",
            "plistlib.readPlist",
            "plistlib.writePlist",
            "plistlib.readPlistFromBytes",
            "plistlib.writePlistToBytes",
        },
        (3, 4, 4): {"asyncio.tasks.async"},
        (3, 5, 0): {
            "fractions.gcd",
            "inspect.formatargspec",
            "inspect.getcallargs",
            "platform.linux_distribution",
            "platform.dist",
        },
        (3, 6, 0): {
            "importlib._bootstrap_external.FileLoader.load_module",
            "_ssl.RAND_pseudo_bytes",
        },
        (3, 7, 0): {
            "sys.set_coroutine_wrapper",
            "sys.get_coroutine_wrapper",
            "aifc.openfp",
            "threading.Thread.isAlive",
            "asyncio.Task.current_task",
            "asyncio.Task.all_task",
            "locale.format",
            "ssl.wrap_socket",
            "ssl.match_hostname",
            "sunau.openfp",
            "wave.openfp",
        },
        (3, 8, 0): {
            "gettext.lgettext",
            "gettext.ldgettext",
            "gettext.lngettext",
            "gettext.ldngettext",
            "gettext.bind_textdomain_codeset",
            "gettext.NullTranslations.output_charset",
            "gettext.NullTranslations.set_output_charset",
            "threading.Thread.isAlive",
        },
        (3, 9, 0): {
            "binascii.b2a_hqx",
            "binascii.a2b_hqx",
            "binascii.rlecode_hqx",
            "binascii.rledecode_hqx",
        },
        (3, 10, 0): {
            "_sqlite3.enable_shared_cache",
            "importlib.abc.Finder.find_module",
            "pathlib.Path.link_to",
            "zipimport.zipimporter.load_module",
            "zipimport.zipimporter.find_module",
            "zipimport.zipimporter.find_loader",
            "threading.currentThread",
            "threading.activeCount",
            "threading.Condition.notifyAll",
            "threading.Event.isSet",
            "threading.Thread.setName",
            "threading.Thread.getName",
            "threading.Thread.isDaemon",
            "threading.Thread.setDaemon",
            "cgi.log",
        },
        (3, 11, 0): {
            "locale.getdefaultlocale",
            "unittest.TestLoader.findTestCases",
            "unittest.TestLoader.loadTestsFromTestCase",
            "unittest.TestLoader.getTestCaseNames",
        },
    },
}
```
### 10 - pylint/epylint.py:

Start line: 126, End line: 143

```python
@overload
def py_run(
    command_options: str = ...,
    return_std: Literal[False] = ...,
    stdout: TextIO | int | None = ...,
    stderr: TextIO | int | None = ...,
) -> None:
    ...


@overload
def py_run(
    command_options: str,
    return_std: Literal[True],
    stdout: TextIO | int | None = ...,
    stderr: TextIO | int | None = ...,
) -> tuple[StringIO, StringIO]:
    ...
```
### 42 - pylint/lint/pylinter.py:

Start line: 481, End line: 518

```python
class PyLinter(
    _ArgumentsManager,
    _MessageStateHandler,
    reporters.ReportsHandlerMixIn,
    checkers.BaseChecker,
):

    def any_fail_on_issues(self) -> bool:
        return any(x in self.fail_on_symbols for x in self.stats.by_msg.keys())

    def disable_reporters(self) -> None:
        """Disable all reporters."""
        for _reporters in self._reports.values():
            for report_id, _, _ in _reporters:
                self.disable_report(report_id)

    def _parse_error_mode(self) -> None:
        """Parse the current state of the error mode.

        Error mode: enable only errors; no reports, no persistent.
        """
        if not self._error_mode:
            return

        self.disable_noerror_messages()
        self.disable("miscellaneous")
        self.set_option("reports", False)
        self.set_option("persistent", False)
        self.set_option("score", False)

    # code checking methods ###################################################

    def get_checkers(self) -> list[BaseChecker]:
        """Return all available checkers as an ordered list."""
        return sorted(c for _checkers in self._checkers.values() for c in _checkers)

    def get_checker_names(self) -> list[str]:
        """Get all the checker names that this linter knows about."""
        return sorted(
            {
                checker.name
                for checker in self.get_checkers()
                if checker.name != MAIN_CHECKER_NAME
            }
        )
```
### 61 - pylint/lint/expand_modules.py:

Start line: 5, End line: 23

```python
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from re import Pattern

from astroid import modutils

from pylint.typing import ErrorDescriptionDict, ModuleDescriptionDict


def _modpath_from_file(filename: str, is_namespace: bool, path: list[str]) -> list[str]:
    def _is_package_cb(inner_path: str, parts: list[str]) -> bool:
        return modutils.check_modpath_has_init(inner_path, parts) or is_namespace

    return modutils.modpath_from_file_with_callback(
        filename, path=path, is_package_cb=_is_package_cb
    )
```
### 78 - pylint/lint/pylinter.py:

Start line: 5, End line: 96

```python
from __future__ import annotations

import collections
import contextlib
import functools
import os
import sys
import tokenize
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from io import TextIOWrapper
from typing import Any

import astroid
from astroid import AstroidError, nodes

from pylint import checkers, exceptions, interfaces, reporters
from pylint.checkers.base_checker import BaseChecker
from pylint.config.arguments_manager import _ArgumentsManager
from pylint.constants import (
    MAIN_CHECKER_NAME,
    MSG_TYPES,
    MSG_TYPES_STATUS,
    WarningScope,
)
from pylint.lint.base_options import _make_linter_options
from pylint.lint.caching import load_results, save_results
from pylint.lint.expand_modules import expand_modules
from pylint.lint.message_state_handler import _MessageStateHandler
from pylint.lint.parallel import check_parallel
from pylint.lint.report_functions import (
    report_messages_by_module_stats,
    report_messages_stats,
    report_total_messages_stats,
)
from pylint.lint.utils import (
    fix_import_path,
    get_fatal_error_message,
    prepare_crash_report,
)
from pylint.message import Message, MessageDefinition, MessageDefinitionStore
from pylint.reporters.base_reporter import BaseReporter
from pylint.reporters.text import TextReporter
from pylint.reporters.ureports import nodes as report_nodes
from pylint.typing import (
    FileItem,
    ManagedMessage,
    MessageDefinitionTuple,
    MessageLocationTuple,
    ModuleDescriptionDict,
    Options,
)
from pylint.utils import ASTWalker, FileState, LinterStats, utils

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


MANAGER = astroid.MANAGER


class GetAstProtocol(Protocol):
    def __call__(
        self, filepath: str, modname: str, data: str | None = None
    ) -> nodes.Module:
        ...


def _read_stdin() -> str:
    # See https://github.com/python/typeshed/pull/5623 for rationale behind assertion
    assert isinstance(sys.stdin, TextIOWrapper)
    sys.stdin = TextIOWrapper(sys.stdin.detach(), encoding="utf-8")
    return sys.stdin.read()


def _load_reporter_by_class(reporter_class: str) -> type[BaseReporter]:
    qname = reporter_class
    module_part = astroid.modutils.get_module_part(qname)
    module = astroid.modutils.load_module_from_name(module_part)
    class_name = qname.split(".")[-1]
    klass = getattr(module, class_name)
    assert issubclass(klass, BaseReporter), f"{klass} is not a BaseReporter"
    return klass


# Python Linter class #########################################################

# pylint: disable-next=consider-using-namedtuple-or-dataclass
```
