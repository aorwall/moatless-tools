# pylint-dev__pylint-4516

| **pylint-dev/pylint** | `0b5a44359d8255c136af27c0ef5f5b196a526430` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 30897 |
| **Avg pos** | 73.5 |
| **Min pos** | 62 |
| **Max pos** | 85 |
| **Top file pos** | 2 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -1,5 +1,6 @@
 import os
 import sys
+from typing import List, Pattern, Tuple
 
 from astroid import modutils
 
@@ -28,32 +29,33 @@ def get_python_path(filepath: str) -> str:
             return os.getcwd()
 
 
-def _basename_in_ignore_list_re(base_name, ignore_list_re):
-    """Determines if the basename is matched in a regex ignorelist
-
-    :param str base_name: The basename of the file
-    :param list ignore_list_re: A collection of regex patterns to match against.
-        Successful matches are ignored.
-
-    :returns: `True` if the basename is ignored, `False` otherwise.
-    :rtype: bool
-    """
+def _is_in_ignore_list_re(element: str, ignore_list_re: List[Pattern]) -> bool:
+    """determines if the element is matched in a regex ignore-list"""
     for file_pattern in ignore_list_re:
-        if file_pattern.match(base_name):
+        if file_pattern.match(element):
             return True
     return False
 
 
-def expand_modules(files_or_modules, ignore_list, ignore_list_re):
-    """Take a list of files/modules/packages and return the list of tuple
-    (file, module name) which have to be actually checked."""
+def expand_modules(
+    files_or_modules: List[str],
+    ignore_list: List[str],
+    ignore_list_re: List[Pattern],
+    ignore_list_paths_re: List[Pattern],
+) -> Tuple[List[dict], List[dict]]:
+    """take a list of files/modules/packages and return the list of tuple
+    (file, module name) which have to be actually checked
+    """
     result = []
     errors = []
     path = sys.path.copy()
+
     for something in files_or_modules:
         basename = os.path.basename(something)
-        if basename in ignore_list or _basename_in_ignore_list_re(
-            basename, ignore_list_re
+        if (
+            basename in ignore_list
+            or _is_in_ignore_list_re(os.path.basename(something), ignore_list_re)
+            or _is_in_ignore_list_re(something, ignore_list_paths_re)
         ):
             continue
         module_path = get_python_path(something)
@@ -117,10 +119,11 @@ def expand_modules(files_or_modules, ignore_list, ignore_list_re):
             ):
                 if filepath == subfilepath:
                     continue
-                if _basename_in_ignore_list_re(
+                if _is_in_ignore_list_re(
                     os.path.basename(subfilepath), ignore_list_re
-                ):
+                ) or _is_in_ignore_list_re(subfilepath, ignore_list_paths_re):
                     continue
+
                 modpath = _modpath_from_file(
                     subfilepath, is_namespace, path=additional_search_path
                 )
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -186,6 +186,17 @@ def make_options():
                     " skipped. The regex matches against base names, not paths.",
                 },
             ),
+            (
+                "ignore-paths",
+                {
+                    "type": "regexp_csv",
+                    "metavar": "<pattern>[,<pattern>...]",
+                    "dest": "ignore_list_paths_re",
+                    "default": (),
+                    "help": "Add files or directories matching the regex patterns to the"
+                    " ignore-list. The regex matches against paths.",
+                },
+            ),
             (
                 "persistent",
                 {
@@ -1046,7 +1057,10 @@ def _iterate_file_descrs(self, files_or_modules):
     def _expand_files(self, modules):
         """get modules and errors from a list of modules and handle errors"""
         result, errors = expand_modules(
-            modules, self.config.black_list, self.config.black_list_re
+            modules,
+            self.config.black_list,
+            self.config.black_list_re,
+            self.config.ignore_list_paths_re,
         )
         for error in errors:
             message = modname = error["mod"]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/lint/expand_modules.py | 3 | 3 | 85 | 20 | 39205
| pylint/lint/expand_modules.py | 31 | 56 | - | 20 | -
| pylint/lint/expand_modules.py | 120 | 122 | - | 20 | -
| pylint/lint/pylinter.py | 189 | 189 | 62 | 2 | 30897
| pylint/lint/pylinter.py | 1049 | 1049 | - | 2 | -


## Problem Statement

```
Ignore clause not ignoring directories
This is a different issue to [issues/908](https://github.com/PyCQA/pylint/issues/908).

### Steps to reproduce
1. Create a directory `test` and within that a directory `stuff`.
2. Create files `test/a.py` and `test/stuff/b.py`. Put syntax errors in both.
3. From `test`, run `pylint *.py **/*.py --ignore stuff`.

### Current behavior
Pylint does not ignore `stuff/b.py`, producing the message
\`\`\`************* Module a
a.py:1:0: E0001: invalid syntax (<unknown>, line 1) (syntax-error)
************* Module b
stuff/b.py:1:0: E0001: invalid syntax (<unknown>, line 1) (syntax-error)
\`\`\`

### Expected behavior
Pylint ignores the file `stuff/b.py`.

### pylint --version output
\`\`\`pylint 2.2.2
astroid 2.1.0
Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
[GCC 7.3.0]\`\`\`

ignore-patterns does not skip non-top-level directories.
<!--
  Hi there! Thank you for discovering and submitting an issue.

  Before you submit this, make sure that the issue doesn't already exist
  or if it is not closed.

  Is your issue fixed on the preview release?: pip install pylint astroid --pre -U

-->

### Steps to reproduce
1.  create a a/b/c.py (where c.py will generate a pylint message, so that we get output) (along with the appropriate \_\_init\_\_.py files)
2.  Run pylint: pylint --ignore-patterns=b
3.  Run pylint: pylint --ignore-patterns=a

### Current behavior
c.py is skipped for ignore-patterns=a, but not for ignore-patterns=b

### Expected behavior
c.py should be skipped for both

### pylint --version output
pylint 2.1.1
astroid 2.1.0-dev
Python 3.6.3 (v3.6.3:2c5fed8, Oct  3 2017, 17:26:49) [MSC v.1900 32 bit (Intel)]


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/checkers/design_analysis.py | 102 | 180| 511 | 511 | 4460 | 
| 2 | **2 pylint/lint/pylinter.py** | 56 | 131| 640 | 1151 | 14235 | 
| 3 | 3 pylint/checkers/stdlib.py | 37 | 101| 714 | 1865 | 19561 | 
| 4 | 4 pylint/pyreverse/main.py | 22 | 173| 799 | 2664 | 21105 | 
| 5 | 5 pylint/__init__.py | 12 | 46| 161 | 2825 | 21805 | 
| 6 | 6 pylint/extensions/bad_builtin.py | 11 | 49| 261 | 3086 | 22425 | 
| 7 | 7 pylint/checkers/imports.py | 133 | 151| 137 | 3223 | 30976 | 
| 8 | 8 pylint/checkers/classes.py | 45 | 89| 268 | 3491 | 49234 | 
| 9 | 9 pylint/constants.py | 4 | 51| 350 | 3841 | 49625 | 
| 10 | 9 pylint/checkers/design_analysis.py | 25 | 101| 641 | 4482 | 49625 | 
| 11 | 10 pylint/checkers/variables.py | 51 | 153| 639 | 5121 | 66672 | 
| 12 | 11 pylint/checkers/python3.py | 42 | 110| 375 | 5496 | 78101 | 
| 13 | 12 pylint/checkers/base.py | 63 | 94| 217 | 5713 | 100082 | 
| 14 | 13 pylint/extensions/typing.py | 72 | 174| 785 | 6498 | 102746 | 
| 15 | **13 pylint/lint/pylinter.py** | 750 | 770| 182 | 6680 | 102746 | 
| 16 | 13 pylint/checkers/base.py | 939 | 1081| 1329 | 8009 | 102746 | 
| 17 | 14 pylint/checkers/typecheck.py | 126 | 171| 370 | 8379 | 118794 | 
| 18 | 15 pylint/lint/__init__.py | 73 | 105| 170 | 8549 | 120258 | 
| 19 | 16 pylint/extensions/overlapping_exceptions.py | 6 | 87| 576 | 9125 | 120881 | 
| 20 | **16 pylint/lint/pylinter.py** | 687 | 704| 186 | 9311 | 120881 | 
| 21 | 16 pylint/checkers/imports.py | 221 | 316| 769 | 10080 | 120881 | 
| 22 | 16 pylint/checkers/imports.py | 319 | 443| 794 | 10874 | 120881 | 
| 23 | 17 pylint/extensions/confusing_elif.py | 7 | 57| 452 | 11326 | 121416 | 
| 24 | **17 pylint/lint/pylinter.py** | 4 | 54| 331 | 11657 | 121416 | 
| 25 | 17 pylint/checkers/stdlib.py | 302 | 392| 905 | 12562 | 121416 | 
| 26 | 17 pylint/checkers/typecheck.py | 55 | 123| 394 | 12956 | 121416 | 
| 27 | 17 pylint/checkers/stdlib.py | 104 | 224| 897 | 13853 | 121416 | 
| 28 | 17 pylint/checkers/base.py | 459 | 566| 937 | 14790 | 121416 | 
| 29 | 18 pylint/lint/run.py | 4 | 478| 339 | 15129 | 124874 | 
| 30 | 19 pylint/checkers/spelling.py | 29 | 128| 542 | 15671 | 128605 | 
| 31 | **20 pylint/lint/expand_modules.py** | 31 | 44| 115 | 15786 | 129631 | 
| 32 | 21 pylint/checkers/newstyle.py | 24 | 136| 782 | 16568 | 130866 | 
| 33 | 21 pylint/checkers/python3.py | 1165 | 1248| 701 | 17269 | 130866 | 
| 34 | **21 pylint/lint/pylinter.py** | 726 | 748| 284 | 17553 | 130866 | 
| 35 | 21 pylint/checkers/typecheck.py | 726 | 866| 1021 | 18574 | 130866 | 
| 36 | 22 pylint/checkers/refactoring/not_checker.py | 4 | 44| 297 | 18871 | 131505 | 
| 37 | 23 pylint/checkers/async.py | 11 | 52| 334 | 19205 | 132312 | 
| 38 | 23 pylint/checkers/base.py | 1172 | 1223| 459 | 19664 | 132312 | 
| 39 | 24 pylint/checkers/format.py | 53 | 118| 345 | 20009 | 139163 | 
| 40 | 24 pylint/checkers/base.py | 1652 | 1692| 245 | 20254 | 139163 | 
| 41 | 25 pylint/utils/__init__.py | 45 | 90| 236 | 20490 | 140242 | 
| 42 | 26 pylint/extensions/broad_try_clause.py | 12 | 77| 412 | 20902 | 140821 | 
| 43 | 26 pylint/checkers/python3.py | 1250 | 1273| 190 | 21092 | 140821 | 
| 44 | 27 pylint/testutils/__init__.py | 31 | 58| 214 | 21306 | 141619 | 
| 45 | 27 pylint/checkers/base.py | 159 | 210| 485 | 21791 | 141619 | 
| 46 | 27 pylint/extensions/typing.py | 1 | 69| 834 | 22625 | 141619 | 
| 47 | 28 pylint/checkers/exceptions.py | 489 | 507| 195 | 22820 | 146708 | 
| 48 | 28 pylint/checkers/python3.py | 608 | 668| 316 | 23136 | 146708 | 
| 49 | 28 pylint/checkers/base.py | 1560 | 1574| 178 | 23314 | 146708 | 
| 50 | 28 pylint/checkers/exceptions.py | 34 | 80| 299 | 23613 | 146708 | 
| 51 | 28 pylint/checkers/python3.py | 896 | 964| 564 | 24177 | 146708 | 
| 52 | 28 pylint/checkers/classes.py | 740 | 830| 575 | 24752 | 146708 | 
| 53 | 28 pylint/lint/run.py | 56 | 296| 1587 | 26339 | 146708 | 
| 54 | 29 doc/conf.py | 107 | 243| 889 | 27228 | 148554 | 
| 55 | 29 pylint/checkers/python3.py | 1055 | 1073| 206 | 27434 | 148554 | 
| 56 | 29 pylint/checkers/exceptions.py | 314 | 333| 171 | 27605 | 148554 | 
| 57 | 29 pylint/checkers/exceptions.py | 258 | 282| 148 | 27753 | 148554 | 
| 58 | 29 pylint/checkers/exceptions.py | 82 | 185| 992 | 28745 | 148554 | 
| 59 | 30 pylint/config/__init__.py | 34 | 79| 330 | 29075 | 149855 | 
| 60 | 31 pylint/checkers/refactoring/refactoring_checker.py | 4 | 55| 360 | 29435 | 164617 | 
| 61 | 31 pylint/checkers/python3.py | 670 | 894| 1208 | 30643 | 164617 | 
| **-> 62 <-** | **31 pylint/lint/pylinter.py** | 134 | 450| 254 | 30897 | 164617 | 
| 63 | 31 pylint/checkers/python3.py | 993 | 1024| 252 | 31149 | 164617 | 
| 64 | 31 pylint/checkers/imports.py | 805 | 830| 270 | 31419 | 164617 | 
| 65 | 31 pylint/checkers/exceptions.py | 383 | 430| 383 | 31802 | 164617 | 
| 66 | 32 pylint/checkers/utils.py | 51 | 110| 363 | 32165 | 176726 | 
| 67 | 33 pylint/checkers/strings.py | 36 | 83| 323 | 32488 | 185022 | 
| 68 | 33 pylint/checkers/stdlib.py | 227 | 269| 200 | 32688 | 185022 | 
| 69 | **33 pylint/lint/pylinter.py** | 538 | 568| 247 | 32935 | 185022 | 
| 70 | 33 pylint/checkers/python3.py | 1112 | 1136| 219 | 33154 | 185022 | 
| 71 | 33 pylint/checkers/base.py | 717 | 754| 325 | 33479 | 185022 | 
| 72 | 34 pylint/testutils/output_line.py | 4 | 22| 148 | 33627 | 185689 | 
| 73 | 34 pylint/checkers/classes.py | 2200 | 2223| 162 | 33789 | 185689 | 
| 74 | 34 pylint/checkers/imports.py | 574 | 599| 268 | 34057 | 185689 | 
| 75 | 34 pylint/checkers/utils.py | 111 | 225| 702 | 34759 | 185689 | 
| 76 | 34 pylint/extensions/typing.py | 285 | 316| 237 | 34996 | 185689 | 
| 77 | 35 pylint/checkers/refactoring/recommendation_checker.py | 3 | 58| 465 | 35461 | 187979 | 
| 78 | 36 pylint/checkers/logging.py | 25 | 107| 659 | 36120 | 191381 | 
| 79 | **36 pylint/lint/pylinter.py** | 962 | 985| 245 | 36365 | 191381 | 
| 80 | 36 pylint/checkers/utils.py | 1048 | 1120| 523 | 36888 | 191381 | 
| 81 | 36 pylint/checkers/base.py | 1760 | 1848| 644 | 37532 | 191381 | 
| 82 | 36 pylint/checkers/imports.py | 460 | 483| 178 | 37710 | 191381 | 
| 83 | 36 pylint/lint/run.py | 297 | 405| 957 | 38667 | 191381 | 
| 84 | 37 pylint/config/option_parser.py | 4 | 48| 324 | 38991 | 191747 | 
| **-> 85 <-** | **37 pylint/lint/expand_modules.py** | 1 | 28| 214 | 39205 | 191747 | 
| 86 | 37 pylint/checkers/stdlib.py | 394 | 434| 431 | 39636 | 191747 | 
| 87 | 37 pylint/checkers/classes.py | 2225 | 2271| 456 | 40092 | 191747 | 
| 88 | 37 pylint/checkers/exceptions.py | 352 | 381| 370 | 40462 | 191747 | 
| 89 | 38 pylint/testutils/reporter_for_tests.py | 4 | 81| 458 | 40920 | 192246 | 
| 90 | 39 pylint/extensions/redefined_variable_type.py | 12 | 58| 311 | 41231 | 193237 | 
| 91 | 39 pylint/checkers/classes.py | 270 | 292| 169 | 41400 | 193237 | 
| 92 | 40 pylint/extensions/check_elif.py | 12 | 79| 484 | 41884 | 193918 | 
| 93 | 40 pylint/checkers/stdlib.py | 478 | 521| 319 | 42203 | 193918 | 
| 94 | 41 pylint/testutils/constants.py | 4 | 29| 272 | 42475 | 194231 | 
| 95 | 42 pylint/lint/utils.py | 4 | 46| 285 | 42760 | 194731 | 
| 96 | 43 pylint/message/message_handler_mix_in.py | 4 | 21| 112 | 42872 | 197739 | 
| 97 | 43 pylint/checkers/exceptions.py | 284 | 312| 191 | 43063 | 197739 | 
| 98 | 43 doc/conf.py | 1 | 106| 753 | 43816 | 197739 | 
| 99 | 44 pylint/testutils/global_test_linter.py | 5 | 21| 111 | 43927 | 197891 | 
| 100 | 44 pylint/checkers/variables.py | 2015 | 2081| 551 | 44478 | 197891 | 
| 101 | 44 pylint/checkers/base.py | 1617 | 1649| 277 | 44755 | 197891 | 
| 102 | 44 pylint/extensions/typing.py | 193 | 205| 120 | 44875 | 197891 | 
| 103 | 44 pylint/message/message_handler_mix_in.py | 56 | 66| 132 | 45007 | 197891 | 
| 104 | 44 pylint/message/message_handler_mix_in.py | 325 | 360| 261 | 45268 | 197891 | 
| 105 | **44 pylint/lint/pylinter.py** | 1138 | 1153| 145 | 45413 | 197891 | 
| 106 | 44 pylint/checkers/classes.py | 869 | 900| 254 | 45667 | 197891 | 
| 107 | 44 pylint/checkers/spelling.py | 131 | 140| 109 | 45776 | 197891 | 


### Hint

```
The problem seems to be in `utils.expand_modules()` in which we find the code that actually performs the ignoring:

\`\`\`python
if os.path.basename(something) in black_list:
    continue
if _basename_in_blacklist_re(os.path.basename(something), black_list_re):
    continue
\`\`\`
Here `something` will be of the form `"stuff/b.py"` and `os.path.basename(something)` will be of the form `b.py`. In other words, _before_ we do the ignoring, we specifically remove from the filepath all of the information about what directory it's in, so that it's impossible to have any way of ignoring a directory. Is this the intended behavior?
Hi @geajack Thanks for reporting an issue. That behaviour it's probably not intended, I think we definitely need to fix this to allow ignoring directories as well.
@geajack  @PCManticore Is there a work around to force pylint to ignore directories?  I've tried `ignore`, `ignored-modules`, and `ignore-patterns` and not getting to a working solution.  Background is I want to pylint scan source repositories (during our TravisCI PR builds), but want to exclude any python source found in certain directories: specifically directories brought in using git-submodules (as those submodules are already covered by their own TravisCI builds).  Would like to set something in the project's `.pylintrc` that would configure pylint to ignore those directories...
Has there been any progress on this issue? It's still apparent in `pylint 2.3.1`.
@bgehman Right now ignoring directories is not supported, as per this issue suggests. We should add support for ignoring directories to `--ignore` and `--ignore-patterns`, while `--ignored-modules` does something else entirely (ignores modules from being type checked, not completely analysed).

@Michionlion There was no progress on this issue, as you can see there are 400 issues opened, so depending on my time, it's entirely possible that an issue could stay open for months or years. Feel free to tackle a PR if you need this fixed sooner.
Relates to #2541
I also meet this problem.
Can we check path directly? I think it's more convenient for usage.
workaround... add this to your .pylintrc:

\`\`\`
init-hook=
    sys.path.append(os.getcwd());
    from pylint_ignore import PylintIgnorePaths;
    PylintIgnorePaths('my/thirdparty/subdir', 'my/other/badcode')
\`\`\`

then create `pylint_ignore.py`:

\`\`\`
from pylint.utils import utils


class PylintIgnorePaths:
    def __init__(self, *paths):
        self.paths = paths
        self.original_expand_modules = utils.expand_modules
        utils.expand_modules = self.patched_expand

    def patched_expand(self, *args, **kwargs):
        result, errors = self.original_expand_modules(*args, **kwargs)

        def keep_item(item):
            if any(1 for path in self.paths if item['path'].startswith(path)):
                return False

            return True

        result = list(filter(keep_item, result))

        return result, errors
When will we get a fix for this issue?
This is still broken, one and a half year later... The documentation still claims that these parameters can ignore directories.

```

## Patch

```diff
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -1,5 +1,6 @@
 import os
 import sys
+from typing import List, Pattern, Tuple
 
 from astroid import modutils
 
@@ -28,32 +29,33 @@ def get_python_path(filepath: str) -> str:
             return os.getcwd()
 
 
-def _basename_in_ignore_list_re(base_name, ignore_list_re):
-    """Determines if the basename is matched in a regex ignorelist
-
-    :param str base_name: The basename of the file
-    :param list ignore_list_re: A collection of regex patterns to match against.
-        Successful matches are ignored.
-
-    :returns: `True` if the basename is ignored, `False` otherwise.
-    :rtype: bool
-    """
+def _is_in_ignore_list_re(element: str, ignore_list_re: List[Pattern]) -> bool:
+    """determines if the element is matched in a regex ignore-list"""
     for file_pattern in ignore_list_re:
-        if file_pattern.match(base_name):
+        if file_pattern.match(element):
             return True
     return False
 
 
-def expand_modules(files_or_modules, ignore_list, ignore_list_re):
-    """Take a list of files/modules/packages and return the list of tuple
-    (file, module name) which have to be actually checked."""
+def expand_modules(
+    files_or_modules: List[str],
+    ignore_list: List[str],
+    ignore_list_re: List[Pattern],
+    ignore_list_paths_re: List[Pattern],
+) -> Tuple[List[dict], List[dict]]:
+    """take a list of files/modules/packages and return the list of tuple
+    (file, module name) which have to be actually checked
+    """
     result = []
     errors = []
     path = sys.path.copy()
+
     for something in files_or_modules:
         basename = os.path.basename(something)
-        if basename in ignore_list or _basename_in_ignore_list_re(
-            basename, ignore_list_re
+        if (
+            basename in ignore_list
+            or _is_in_ignore_list_re(os.path.basename(something), ignore_list_re)
+            or _is_in_ignore_list_re(something, ignore_list_paths_re)
         ):
             continue
         module_path = get_python_path(something)
@@ -117,10 +119,11 @@ def expand_modules(files_or_modules, ignore_list, ignore_list_re):
             ):
                 if filepath == subfilepath:
                     continue
-                if _basename_in_ignore_list_re(
+                if _is_in_ignore_list_re(
                     os.path.basename(subfilepath), ignore_list_re
-                ):
+                ) or _is_in_ignore_list_re(subfilepath, ignore_list_paths_re):
                     continue
+
                 modpath = _modpath_from_file(
                     subfilepath, is_namespace, path=additional_search_path
                 )
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -186,6 +186,17 @@ def make_options():
                     " skipped. The regex matches against base names, not paths.",
                 },
             ),
+            (
+                "ignore-paths",
+                {
+                    "type": "regexp_csv",
+                    "metavar": "<pattern>[,<pattern>...]",
+                    "dest": "ignore_list_paths_re",
+                    "default": (),
+                    "help": "Add files or directories matching the regex patterns to the"
+                    " ignore-list. The regex matches against paths.",
+                },
+            ),
             (
                 "persistent",
                 {
@@ -1046,7 +1057,10 @@ def _iterate_file_descrs(self, files_or_modules):
     def _expand_files(self, modules):
         """get modules and errors from a list of modules and handle errors"""
         result, errors = expand_modules(
-            modules, self.config.black_list, self.config.black_list_re
+            modules,
+            self.config.black_list,
+            self.config.black_list_re,
+            self.config.ignore_list_paths_re,
         )
         for error in errors:
             message = modname = error["mod"]

```

## Test Patch

```diff
diff --git a/tests/lint/unittest_expand_modules.py b/tests/lint/unittest_expand_modules.py
--- a/tests/lint/unittest_expand_modules.py
+++ b/tests/lint/unittest_expand_modules.py
@@ -7,19 +7,29 @@
 
 import pytest
 
-from pylint.lint.expand_modules import _basename_in_ignore_list_re, expand_modules
+from pylint.lint.expand_modules import _is_in_ignore_list_re, expand_modules
 
 
-def test__basename_in_ignore_list_re_match():
-    patterns = [re.compile(".*enchilada.*"), re.compile("unittest_.*")]
-    assert _basename_in_ignore_list_re("unittest_utils.py", patterns)
-    assert _basename_in_ignore_list_re("cheese_enchiladas.xml", patterns)
+def test__is_in_ignore_list_re_match():
+    patterns = [
+        re.compile(".*enchilada.*"),
+        re.compile("unittest_.*"),
+        re.compile(".*tests/.*"),
+    ]
+    assert _is_in_ignore_list_re("unittest_utils.py", patterns)
+    assert _is_in_ignore_list_re("cheese_enchiladas.xml", patterns)
+    assert _is_in_ignore_list_re("src/tests/whatever.xml", patterns)
 
 
-def test__basename_in_ignore_list_re_nomatch():
-    patterns = [re.compile(".*enchilada.*"), re.compile("unittest_.*")]
-    assert not _basename_in_ignore_list_re("test_utils.py", patterns)
-    assert not _basename_in_ignore_list_re("enchilad.py", patterns)
+def test__is_in_ignore_list_re_nomatch():
+    patterns = [
+        re.compile(".*enchilada.*"),
+        re.compile("unittest_.*"),
+        re.compile(".*tests/.*"),
+    ]
+    assert not _is_in_ignore_list_re("test_utils.py", patterns)
+    assert not _is_in_ignore_list_re("enchilad.py", patterns)
+    assert not _is_in_ignore_list_re("src/tests.py", patterns)
 
 
 TEST_DIRECTORY = Path(__file__).parent.parent
@@ -70,8 +80,10 @@ def test__basename_in_ignore_list_re_nomatch():
     ],
 )
 def test_expand_modules(files_or_modules, expected):
-    ignore_list, ignore_list_re = [], []
-    modules, errors = expand_modules(files_or_modules, ignore_list, ignore_list_re)
+    ignore_list, ignore_list_re, ignore_list_paths_re = [], [], []
+    modules, errors = expand_modules(
+        files_or_modules, ignore_list, ignore_list_re, ignore_list_paths_re
+    )
     modules.sort(key=lambda d: d["name"])
     assert modules == expected
     assert not errors

```


## Code snippets

### 1 - pylint/checkers/design_analysis.py:

Start line: 102, End line: 180

```python
STDLIB_CLASSES_IGNORE_ANCESTOR = frozenset(
    (
        "builtins.object",
        "builtins.tuple",
        "builtins.dict",
        "builtins.list",
        "builtins.set",
        "bulitins.frozenset",
        "collections.ChainMap",
        "collections.Counter",
        "collections.OrderedDict",
        "collections.UserDict",
        "collections.UserList",
        "collections.UserString",
        "collections.defaultdict",
        "collections.deque",
        "collections.namedtuple",
        "_collections_abc.Awaitable",
        "_collections_abc.Coroutine",
        "_collections_abc.AsyncIterable",
        "_collections_abc.AsyncIterator",
        "_collections_abc.AsyncGenerator",
        "_collections_abc.Hashable",
        "_collections_abc.Iterable",
        "_collections_abc.Iterator",
        "_collections_abc.Generator",
        "_collections_abc.Reversible",
        "_collections_abc.Sized",
        "_collections_abc.Container",
        "_collections_abc.Collection",
        "_collections_abc.Set",
        "_collections_abc.MutableSet",
        "_collections_abc.Mapping",
        "_collections_abc.MutableMapping",
        "_collections_abc.MappingView",
        "_collections_abc.KeysView",
        "_collections_abc.ItemsView",
        "_collections_abc.ValuesView",
        "_collections_abc.Sequence",
        "_collections_abc.MutableSequence",
        "_collections_abc.ByteString",
        "typing.Tuple",
        "typing.List",
        "typing.Dict",
        "typing.Set",
        "typing.FrozenSet",
        "typing.Deque",
        "typing.DefaultDict",
        "typing.OrderedDict",
        "typing.Counter",
        "typing.ChainMap",
        "typing.Awaitable",
        "typing.Coroutine",
        "typing.AsyncIterable",
        "typing.AsyncIterator",
        "typing.AsyncGenerator",
        "typing.Iterable",
        "typing.Iterator",
        "typing.Generator",
        "typing.Reversible",
        "typing.Container",
        "typing.Collection",
        "typing.AbstractSet",
        "typing.MutableSet",
        "typing.Mapping",
        "typing.MutableMapping",
        "typing.Sequence",
        "typing.MutableSequence",
        "typing.ByteString",
        "typing.MappingView",
        "typing.KeysView",
        "typing.ItemsView",
        "typing.ValuesView",
        "typing.ContextManager",
        "typing.AsyncContextManger",
        "typing.Hashable",
        "typing.Sized",
    )
)
```
### 2 - pylint/lint/pylinter.py:

Start line: 56, End line: 131

```python
MSGS = {
    "F0001": (
        "%s",
        "fatal",
        "Used when an error occurred preventing the analysis of a \
              module (unable to find it for instance).",
    ),
    "F0002": (
        "%s: %s",
        "astroid-error",
        "Used when an unexpected error occurred while building the "
        "Astroid  representation. This is usually accompanied by a "
        "traceback. Please report such errors !",
    ),
    "F0010": (
        "error while code parsing: %s",
        "parse-error",
        "Used when an exception occurred while building the Astroid "
        "representation which could be handled by astroid.",
    ),
    "I0001": (
        "Unable to run raw checkers on built-in module %s",
        "raw-checker-failed",
        "Used to inform that a built-in module has not been checked "
        "using the raw checkers.",
    ),
    "I0010": (
        "Unable to consider inline option %r",
        "bad-inline-option",
        "Used when an inline option is either badly formatted or can't "
        "be used inside modules.",
    ),
    "I0011": (
        "Locally disabling %s (%s)",
        "locally-disabled",
        "Used when an inline option disables a message or a messages category.",
    ),
    "I0013": (
        "Ignoring entire file",
        "file-ignored",
        "Used to inform that the file will not be checked",
    ),
    "I0020": (
        "Suppressed %s (from line %d)",
        "suppressed-message",
        "A message was triggered on a line, but suppressed explicitly "
        "by a disable= comment in the file. This message is not "
        "generated for messages that are ignored due to configuration "
        "settings.",
    ),
    "I0021": (
        "Useless suppression of %s",
        "useless-suppression",
        "Reported when a message is explicitly disabled for a line or "
        "a block of code, but never triggered.",
    ),
    "I0022": (
        'Pragma "%s" is deprecated, use "%s" instead',
        "deprecated-pragma",
        "Some inline pylint options have been renamed or reworked, "
        "only the most recent form should be used. "
        "NOTE:skip-all is only available with pylint >= 0.26",
        {"old_names": [("I0014", "deprecated-disable-all")]},
    ),
    "E0001": ("%s", "syntax-error", "Used when a syntax error is raised for a module."),
    "E0011": (
        "Unrecognized file option %r",
        "unrecognized-inline-option",
        "Used when an unknown inline option is encountered.",
    ),
    "E0012": (
        "Bad option value %r",
        "bad-option-value",
        "Used when a bad value for an inline option is encountered.",
    ),
}
```
### 3 - pylint/checkers/stdlib.py:

Start line: 37, End line: 101

```python
import sys

import astroid

from pylint.checkers import BaseChecker, DeprecatedMixin, utils
from pylint.interfaces import IAstroidChecker

OPEN_FILES = {"open", "file"}
UNITTEST_CASE = "unittest.case"
THREADING_THREAD = "threading.Thread"
COPY_COPY = "copy.copy"
OS_ENVIRON = "os._Environ"
ENV_GETTERS = {"os.getenv"}
SUBPROCESS_POPEN = "subprocess.Popen"
SUBPROCESS_RUN = "subprocess.run"
OPEN_MODULE = "_io"


DEPRECATED_MODULES = {
    (0, 0, 0): {"tkinter.tix", "fpectl"},
    (3, 2, 0): {"optparse"},
    (3, 4, 0): {"imp"},
    (3, 5, 0): {"formatter"},
    (3, 6, 0): {"asynchat", "asyncore"},
    (3, 7, 0): {"macpath"},
    (3, 9, 0): {"lib2to3", "parser", "symbol", "binhex"},
}

DEPRECATED_ARGUMENTS = {
    (0, 0, 0): {
        "int": ((None, "x"),),
        "bool": ((None, "x"),),
        "float": ((None, "x"),),
    },
    (3, 8, 0): {
        "asyncio.tasks.sleep": ((None, "loop"),),
        "asyncio.tasks.gather": ((None, "loop"),),
        "asyncio.tasks.shield": ((None, "loop"),),
        "asyncio.tasks.wait_for": ((None, "loop"),),
        "asyncio.tasks.wait": ((None, "loop"),),
        "asyncio.tasks.as_completed": ((None, "loop"),),
        "asyncio.subprocess.create_subprocess_exec": ((None, "loop"),),
        "asyncio.subprocess.create_subprocess_shell": ((4, "loop"),),
        "gettext.translation": ((5, "codeset"),),
        "gettext.install": ((2, "codeset"),),
        "functools.partialmethod": ((None, "func"),),
        "weakref.finalize": ((None, "func"), (None, "obj")),
        "profile.Profile.runcall": ((None, "func"),),
        "cProfile.Profile.runcall": ((None, "func"),),
        "bdb.Bdb.runcall": ((None, "func"),),
        "trace.Trace.runfunc": ((None, "func"),),
        "curses.wrapper": ((None, "func"),),
        "unittest.case.TestCase.addCleanup": ((None, "function"),),
        "concurrent.futures.thread.ThreadPoolExecutor.submit": ((None, "fn"),),
        "concurrent.futures.process.ProcessPoolExecutor.submit": ((None, "fn"),),
        "contextlib._BaseExitStack.callback": ((None, "callback"),),
        "contextlib.AsyncExitStack.push_async_callback": ((None, "callback"),),
        "multiprocessing.managers.Server.create": ((None, "c"), (None, "typeid")),
        "multiprocessing.managers.SharedMemoryServer.create": (
            (None, "c"),
            (None, "typeid"),
        ),
    },
    (3, 9, 0): {"random.Random.shuffle": ((1, "random"),)},
}
```
### 4 - pylint/pyreverse/main.py:

Start line: 22, End line: 173

```python
import os
import subprocess
import sys

from pylint.config import ConfigurationMixIn
from pylint.pyreverse import writer
from pylint.pyreverse.diadefslib import DiadefsHandler
from pylint.pyreverse.inspector import Linker, project_from_files
from pylint.pyreverse.utils import insert_default_options

OPTIONS = (
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
            action="append",
            metavar="<class>",
            dest="classes",
            default=[],
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
            help="show <ancestor> generations of ancestor classes not in <projects>",
        ),
    ),
    (
        "all-ancestors",
        dict(
            short="A",
            default=None,
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
            help="show <association_level> levels of associated classes not in <projects>",
        ),
    ),
    (
        "all-associated",
        dict(
            short="S",
            default=None,
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
            metavar="[yn]",
            help="include module name in representation of classes",
        ),
    ),
    (
        "only-classnames",
        dict(
            short="k",
            action="store_true",
            default=False,
            help="don't show attributes and methods in the class boxes; \
this disables -f values",
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
            help="create a *.<format> output file if format available.",
        ),
    ),
    (
        "ignore",
        {
            "type": "csv",
            "metavar": "<file[,file...]>",
            "dest": "ignore_list",
            "default": ("CVS",),
            "help": "Files or directories to be skipped. They "
            "should be base names, not paths.",
        },
    ),
    (
        "project",
        {
            "default": "",
            "type": "string",
            "short": "p",
            "metavar": "<project name>",
            "help": "set the project name.",
        },
    ),
    (
        "output-directory",
        {
            "default": "",
            "type": "string",
            "short": "d",
            "action": "store",
            "metavar": "<output_directory>",
            "help": "set the output directory path.",
        },
    ),
)
```
### 5 - pylint/__init__.py:

Start line: 12, End line: 46

```python
import os
import sys

from pylint.__pkginfo__ import __version__

# pylint: disable=import-outside-toplevel


def run_pylint():
    from pylint.lint import Run as PylintRun

    try:
        PylintRun(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)


def run_epylint():
    from pylint.epylint import Run as EpylintRun

    EpylintRun()


def run_pyreverse():
    """run pyreverse"""
    from pylint.pyreverse.main import Run as PyreverseRun

    PyreverseRun(sys.argv[1:])


def run_symilar():
    """run symilar"""
    from pylint.checkers.similar import Run as SimilarRun

    SimilarRun(sys.argv[1:])
```
### 6 - pylint/extensions/bad_builtin.py:

Start line: 11, End line: 49

```python
import astroid

from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages
from pylint.interfaces import IAstroidChecker

BAD_FUNCTIONS = ["map", "filter"]
# Some hints regarding the use of bad builtins.
BUILTIN_HINTS = {"map": "Using a list comprehension can be clearer."}
BUILTIN_HINTS["filter"] = BUILTIN_HINTS["map"]


class BadBuiltinChecker(BaseChecker):

    __implements__ = (IAstroidChecker,)
    name = "deprecated_builtins"
    msgs = {
        "W0141": (
            "Used builtin function %s",
            "bad-builtin",
            "Used when a disallowed builtin function is used (see the "
            "bad-function option). Usual disallowed functions are the ones "
            "like map, or filter , where Python offers now some cleaner "
            "alternative like list comprehension.",
        )
    }

    options = (
        (
            "bad-functions",
            {
                "default": BAD_FUNCTIONS,
                "type": "csv",
                "metavar": "<builtin function names>",
                "help": "List of builtins function names that should not be "
                "used, separated by a comma",
            },
        ),
    )
```
### 7 - pylint/checkers/imports.py:

Start line: 133, End line: 151

```python
def _ignore_import_failure(node, modname, ignored_modules):
    for submodule in _qualified_names(modname):
        if submodule in ignored_modules:
            return True

    # ignore import failure if guarded by `sys.version_info` test
    if isinstance(node.parent, astroid.If) and isinstance(
        node.parent.test, astroid.Compare
    ):
        value = node.parent.test.left
        if isinstance(value, astroid.Subscript):
            value = value.value
        if (
            isinstance(value, astroid.Attribute)
            and value.as_string() == "sys.version_info"
        ):
            return True

    return node_ignores_exception(node, ImportError)
```
### 8 - pylint/checkers/classes.py:

Start line: 45, End line: 89

```python
import collections
from itertools import chain, zip_longest
from typing import List, Pattern

import astroid

from pylint.checkers import BaseChecker
from pylint.checkers.utils import (
    PYMETHODS,
    SPECIAL_METHODS_PARAMS,
    check_messages,
    class_is_abstract,
    decorated_with,
    decorated_with_property,
    has_known_bases,
    is_attr_private,
    is_attr_protected,
    is_builtin_object,
    is_comprehension,
    is_iterable,
    is_overload_stub,
    is_property_setter,
    is_property_setter_or_deleter,
    is_protocol_class,
    node_frame_class,
    overrides_a_method,
    safe_infer,
    unimplemented_abstract_methods,
)
from pylint.interfaces import IAstroidChecker
from pylint.utils import get_global_option

NEXT_METHOD = "__next__"
INVALID_BASE_CLASSES = {"bool", "range", "slice", "memoryview"}
BUILTIN_DECORATORS = {"builtins.property", "builtins.classmethod"}

# Dealing with useless override detection, with regard
# to parameters vs arguments

_CallSignature = collections.namedtuple(
    "_CallSignature", "args kws starred_args starred_kws"
)
_ParameterSignature = collections.namedtuple(
    "_ParameterSignature", "args kwonlyargs varargs kwargs"
)
```
### 9 - pylint/constants.py:

Start line: 4, End line: 51

```python
import sys

import astroid

from pylint.__pkginfo__ import __version__

PY38_PLUS = sys.version_info[:2] >= (3, 8)
PY39_PLUS = sys.version_info[:2] >= (3, 9)
PY310_PLUS = sys.version_info[:2] >= (3, 10)


PY_EXTS = (".py", ".pyc", ".pyo", ".pyw", ".so", ".dll")

MSG_STATE_CONFIDENCE = 2
_MSG_ORDER = "EWRCIF"
MSG_STATE_SCOPE_CONFIG = 0
MSG_STATE_SCOPE_MODULE = 1

# The line/node distinction does not apply to fatal errors and reports.
_SCOPE_EXEMPT = "FR"

MSG_TYPES = {
    "I": "info",
    "C": "convention",
    "R": "refactor",
    "W": "warning",
    "E": "error",
    "F": "fatal",
}
MSG_TYPES_LONG = {v: k for k, v in MSG_TYPES.items()}

MSG_TYPES_STATUS = {"I": 0, "C": 16, "R": 8, "W": 4, "E": 2, "F": 1}

# You probably don't want to change the MAIN_CHECKER_NAME
# This would affect rcfile generation and retro-compatibility
# on all project using [MASTER] in their rcfile.
MAIN_CHECKER_NAME = "master"


class WarningScope:
    LINE = "line-based-msg"
    NODE = "node-based-msg"


full_version = f"""pylint {__version__}
astroid {astroid.__version__}
Python {sys.version}"""
```
### 10 - pylint/checkers/design_analysis.py:

Start line: 25, End line: 101

```python
import re
from collections import defaultdict

import astroid
from astroid import nodes

from pylint import utils
from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages
from pylint.interfaces import IAstroidChecker

MSGS = {
    "R0901": (
        "Too many ancestors (%s/%s)",
        "too-many-ancestors",
        "Used when class has too many parent classes, try to reduce "
        "this to get a simpler (and so easier to use) class.",
    ),
    "R0902": (
        "Too many instance attributes (%s/%s)",
        "too-many-instance-attributes",
        "Used when class has too many instance attributes, try to reduce "
        "this to get a simpler (and so easier to use) class.",
    ),
    "R0903": (
        "Too few public methods (%s/%s)",
        "too-few-public-methods",
        "Used when class has too few public methods, so be sure it's "
        "really worth it.",
    ),
    "R0904": (
        "Too many public methods (%s/%s)",
        "too-many-public-methods",
        "Used when class has too many public methods, try to reduce "
        "this to get a simpler (and so easier to use) class.",
    ),
    "R0911": (
        "Too many return statements (%s/%s)",
        "too-many-return-statements",
        "Used when a function or method has too many return statement, "
        "making it hard to follow.",
    ),
    "R0912": (
        "Too many branches (%s/%s)",
        "too-many-branches",
        "Used when a function or method has too many branches, "
        "making it hard to follow.",
    ),
    "R0913": (
        "Too many arguments (%s/%s)",
        "too-many-arguments",
        "Used when a function or method takes too many arguments.",
    ),
    "R0914": (
        "Too many local variables (%s/%s)",
        "too-many-locals",
        "Used when a function or method has too many local variables.",
    ),
    "R0915": (
        "Too many statements (%s/%s)",
        "too-many-statements",
        "Used when a function or method has too many statements. You "
        "should then split it in smaller functions / methods.",
    ),
    "R0916": (
        "Too many boolean expressions in if statement (%s/%s)",
        "too-many-boolean-expressions",
        "Used when an if statement contains too many boolean expressions.",
    ),
}
SPECIAL_OBJ = re.compile("^_{2}[a-z]+_{2}$")
DATACLASSES_DECORATORS = frozenset({"dataclass", "attrs"})
DATACLASS_IMPORT = "dataclasses"
TYPING_NAMEDTUPLE = "typing.NamedTuple"
TYPING_TYPEDDICT = "typing.TypedDict"

# Set of stdlib classes to ignore when calculating number of ancestors
```
### 15 - pylint/lint/pylinter.py:

Start line: 750, End line: 770

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def list_messages_enabled(self):
        enabled = [
            f"  {message.symbol} ({message.msgid})"
            for message in self.msgs_store.messages
            if self.is_message_enabled(message.msgid)
        ]
        disabled = [
            f"  {message.symbol} ({message.msgid})"
            for message in self.msgs_store.messages
            if not self.is_message_enabled(message.msgid)
        ]
        print("Enabled messages:")
        for msg in sorted(enabled):
            print(msg)
        print("\nDisabled messages:")
        for msg in sorted(disabled):
            print(msg)
        print("")

    # block level option handling #############################################
    # see func_block_disable_msg.py test case for expected behaviour
```
### 20 - pylint/lint/pylinter.py:

Start line: 687, End line: 704

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def any_fail_on_issues(self):
        return any(x in self.fail_on_symbols for x in self.stats["by_msg"])

    def disable_noerror_messages(self):
        for msgcat, msgids in self.msgs_store._msgs_by_category.items():
            # enable only messages with 'error' severity and above ('fatal')
            if msgcat in ["E", "F"]:
                for msgid in msgids:
                    self.enable(msgid)
            else:
                for msgid in msgids:
                    self.disable(msgid)

    def disable_reporters(self):
        """disable all reporters"""
        for _reporters in self._reports.values():
            for report_id, _, _ in _reporters:
                self.disable_report(report_id)
```
### 24 - pylint/lint/pylinter.py:

Start line: 4, End line: 54

```python
import collections
import contextlib
import functools
import operator
import os
import sys
import tokenize
import traceback
import warnings
from io import TextIOWrapper

import astroid

from pylint import checkers, config, exceptions, interfaces, reporters
from pylint.constants import MAIN_CHECKER_NAME, MSG_TYPES
from pylint.lint.expand_modules import expand_modules
from pylint.lint.parallel import check_parallel
from pylint.lint.report_functions import (
    report_messages_by_module_stats,
    report_messages_stats,
    report_total_messages_stats,
)
from pylint.lint.utils import fix_import_path
from pylint.message import MessageDefinitionStore, MessagesHandlerMixIn
from pylint.reporters.ureports import nodes as report_nodes
from pylint.utils import ASTWalker, FileState, utils
from pylint.utils.pragma_parser import (
    OPTION_PO,
    InvalidPragmaError,
    UnRecognizedOptionError,
    parse_pragma,
)

MANAGER = astroid.MANAGER


def _read_stdin():
    # https://mail.python.org/pipermail/python-list/2012-November/634424.html
    sys.stdin = TextIOWrapper(sys.stdin.detach(), encoding="utf-8")
    return sys.stdin.read()


def _load_reporter_by_class(reporter_class: str) -> type:
    qname = reporter_class
    module_part = astroid.modutils.get_module_part(qname)
    module = astroid.modutils.load_module_from_name(module_part)
    class_name = qname.split(".")[-1]
    return getattr(module, class_name)


# Python Linter class #########################################################
```
### 31 - pylint/lint/expand_modules.py:

Start line: 31, End line: 44

```python
def _basename_in_ignore_list_re(base_name, ignore_list_re):
    """Determines if the basename is matched in a regex ignorelist

    :param str base_name: The basename of the file
    :param list ignore_list_re: A collection of regex patterns to match against.
        Successful matches are ignored.

    :returns: `True` if the basename is ignored, `False` otherwise.
    :rtype: bool
    """
    for file_pattern in ignore_list_re:
        if file_pattern.match(base_name):
            return True
    return False
```
### 34 - pylint/lint/pylinter.py:

Start line: 726, End line: 748

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def python3_porting_mode(self):
        """Disable all other checkers and enable Python 3 warnings."""
        self.disable("all")
        # re-enable some errors, or 'print', 'raise', 'async', 'await' will mistakenly lint fine
        self.enable("fatal")  # F0001
        self.enable("astroid-error")  # F0002
        self.enable("parse-error")  # F0010
        self.enable("syntax-error")  # E0001
        self.enable("python3")
        if self._error_mode:
            # The error mode was activated, using the -E flag.
            # So we'll need to enable only the errors from the
            # Python 3 porting checker.
            for msg_id in self._checker_messages("python3"):
                if msg_id.startswith("E"):
                    self.enable(msg_id)
                else:
                    self.disable(msg_id)
        config_parser = self.cfgfile_parser
        if config_parser.has_option("MESSAGES CONTROL", "disable"):
            value = config_parser.get("MESSAGES CONTROL", "disable")
            self.global_set_option("disable", value)
        self._python3_porting_mode = True
```
### 62 - pylint/lint/pylinter.py:

Start line: 134, End line: 450

```python
# pylint: disable=too-many-instance-attributes,too-many-public-methods
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):
    """lint Python modules using external checkers.

    This is the main checker controlling the other ones and the reports
    generation. It is itself both a raw checker and an astroid checker in order
    to:
    * handle message activation / deactivation at the module level
    * handle some basic but necessary stats'data (number of classes, methods...)

    IDE plugin developers: you may have to call
    `astroid.builder.MANAGER.astroid_cache.clear()` across runs if you want
    to ensure the latest code version is actually checked.

    This class needs to support pickling for parallel linting to work. The exception
    is reporter member; see check_parallel function for more details.
    """

    __implements__ = (interfaces.ITokenChecker,)

    name = MAIN_CHECKER_NAME
    priority = 0
    level = 0
    msgs = MSGS

    @staticmethod
    def make_options():
        return
 # ... other code
```
### 69 - pylint/lint/pylinter.py:

Start line: 538, End line: 568

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def _load_reporters(self) -> None:
        sub_reporters = []
        output_files = []
        with contextlib.ExitStack() as stack:
            for reporter_name in self._reporter_names.split(","):
                reporter_name, *reporter_output = reporter_name.split(":", 1)

                reporter = self._load_reporter_by_name(reporter_name)
                sub_reporters.append(reporter)

                if reporter_output:
                    (reporter_output,) = reporter_output

                    # pylint: disable=consider-using-with
                    output_file = stack.enter_context(open(reporter_output, "w"))

                    reporter.set_output(output_file)
                    output_files.append(output_file)

            # Extend the lifetime of all opened output files
            close_output_files = stack.pop_all().close

        if len(sub_reporters) > 1 or output_files:
            self.set_reporter(
                reporters.MultiReporter(
                    sub_reporters,
                    close_output_files,
                )
            )
        else:
            self.set_reporter(sub_reporters[0])
```
### 79 - pylint/lint/pylinter.py:

Start line: 962, End line: 985

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def check_single_file(self, name, filepath, modname):
        """Check single file

        The arguments are the same that are documented in _check_files

        The initialize() method should be called before calling this method
        """
        with self._astroid_module_checker() as check_astroid_module:
            self._check_file(
                self.get_ast, check_astroid_module, name, filepath, modname
            )

    def _check_files(self, get_ast, file_descrs):
        """Check all files from file_descrs

        The file_descrs should be iterable of tuple (name, filepath, modname)
        where
        - name: full name of the module
        - filepath: path of the file
        - modname: module name
        """
        with self._astroid_module_checker() as check_astroid_module:
            for name, filepath, modname in file_descrs:
                self._check_file(get_ast, check_astroid_module, name, filepath, modname)
```
### 85 - pylint/lint/expand_modules.py:

Start line: 1, End line: 28

```python
import os
import sys

from astroid import modutils


def _modpath_from_file(filename, is_namespace, path=None):
    def _is_package_cb(path, parts):
        return modutils.check_modpath_has_init(path, parts) or is_namespace

    return modutils.modpath_from_file_with_callback(
        filename, path=path, is_package_cb=_is_package_cb
    )


def get_python_path(filepath: str) -> str:
    """TODO This get the python path with the (bad) assumption that there is always
    an __init__.py. This is not true since python 3.3 and is causing problem."""
    dirname = os.path.realpath(os.path.expanduser(filepath))
    if not os.path.isdir(dirname):
        dirname = os.path.dirname(dirname)
    while True:
        if not os.path.exists(os.path.join(dirname, "__init__.py")):
            return dirname
        old_dirname = dirname
        dirname = os.path.dirname(dirname)
        if old_dirname == dirname:
            return os.getcwd()
```
### 105 - pylint/lint/pylinter.py:

Start line: 1138, End line: 1153

```python
class PyLinter(
    config.OptionsManagerMixIn,
    MessagesHandlerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def check_astroid_module(self, ast_node, walker, rawcheckers, tokencheckers):
        """Check a module from its astroid representation.

        For return value see _check_astroid_module
        """
        before_check_statements = walker.nbstatements

        retval = self._check_astroid_module(
            ast_node, walker, rawcheckers, tokencheckers
        )

        self.stats["by_module"][self.current_name]["statement"] = (
            walker.nbstatements - before_check_statements
        )

        return retval
```
