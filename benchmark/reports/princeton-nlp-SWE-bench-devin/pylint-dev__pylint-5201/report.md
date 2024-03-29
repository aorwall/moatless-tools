# pylint-dev__pylint-5201

| **pylint-dev/pylint** | `772b3dcc0b0770a843653783e5c93b4256e5ec6f` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | - |
| **Any found context length** | 3136 |
| **Avg pos** | 29.0 |
| **Min pos** | 8 |
| **Max pos** | 28 |
| **Top file pos** | 2 |
| **Missing snippets** | 12 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/config/option.py b/pylint/config/option.py
--- a/pylint/config/option.py
+++ b/pylint/config/option.py
@@ -3,7 +3,9 @@
 
 import copy
 import optparse  # pylint: disable=deprecated-module
+import pathlib
 import re
+from typing import List, Pattern
 
 from pylint import utils
 
@@ -25,6 +27,19 @@ def _regexp_csv_validator(_, name, value):
     return [_regexp_validator(_, name, val) for val in _csv_validator(_, name, value)]
 
 
+def _regexp_paths_csv_validator(_, name: str, value: str) -> List[Pattern[str]]:
+    patterns = []
+    for val in _csv_validator(_, name, value):
+        patterns.append(
+            re.compile(
+                str(pathlib.PureWindowsPath(val)).replace("\\", "\\\\")
+                + "|"
+                + pathlib.PureWindowsPath(val).as_posix()
+            )
+        )
+    return patterns
+
+
 def _choice_validator(choices, name, value):
     if value not in choices:
         msg = "option %s: invalid value: %r, should be in %s"
@@ -80,6 +95,7 @@ def _py_version_validator(_, name, value):
     "float": float,
     "regexp": lambda pattern: re.compile(pattern or ""),
     "regexp_csv": _regexp_csv_validator,
+    "regexp_paths_csv": _regexp_paths_csv_validator,
     "csv": _csv_validator,
     "yn": _yn_validator,
     "choice": lambda opt, name, value: _choice_validator(opt["choices"], name, value),
@@ -122,6 +138,7 @@ class Option(optparse.Option):
     TYPES = optparse.Option.TYPES + (
         "regexp",
         "regexp_csv",
+        "regexp_paths_csv",
         "csv",
         "yn",
         "multiple_choice",
@@ -132,6 +149,7 @@ class Option(optparse.Option):
     TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
     TYPE_CHECKER["regexp"] = _regexp_validator
     TYPE_CHECKER["regexp_csv"] = _regexp_csv_validator
+    TYPE_CHECKER["regexp_paths_csv"] = _regexp_paths_csv_validator
     TYPE_CHECKER["csv"] = _csv_validator
     TYPE_CHECKER["yn"] = _yn_validator
     TYPE_CHECKER["multiple_choice"] = _multiple_choices_validating_option
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -43,7 +43,7 @@ def expand_modules(
     files_or_modules: List[str],
     ignore_list: List[str],
     ignore_list_re: List[Pattern],
-    ignore_list_paths_re: List[Pattern],
+    ignore_list_paths_re: List[Pattern[str]],
 ) -> Tuple[List[ModuleDescriptionDict], List[ErrorDescriptionDict]]:
     """take a list of files/modules/packages and return the list of tuple
     (file, module name) which have to be actually checked
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -46,7 +46,14 @@
     MessageLocationTuple,
     ModuleDescriptionDict,
 )
-from pylint.utils import ASTWalker, FileState, LinterStats, ModuleStats, utils
+from pylint.utils import (
+    ASTWalker,
+    FileState,
+    LinterStats,
+    ModuleStats,
+    get_global_option,
+    utils,
+)
 from pylint.utils.pragma_parser import (
     OPTION_PO,
     InvalidPragmaError,
@@ -220,12 +227,12 @@ def make_options():
             (
                 "ignore-paths",
                 {
-                    "type": "regexp_csv",
+                    "type": "regexp_paths_csv",
                     "metavar": "<pattern>[,<pattern>...]",
-                    "dest": "ignore_list_paths_re",
-                    "default": (),
-                    "help": "Add files or directories matching the regex patterns to the"
-                    " ignore-list. The regex matches against paths.",
+                    "default": [],
+                    "help": "Add files or directories matching the regex patterns to the "
+                    "ignore-list. The regex matches against paths and can be in "
+                    "Posix or Windows format.",
                 },
             ),
             (
@@ -1101,7 +1108,7 @@ def _expand_files(self, modules) -> List[ModuleDescriptionDict]:
             modules,
             self.config.black_list,
             self.config.black_list_re,
-            self.config.ignore_list_paths_re,
+            self._ignore_paths,
         )
         for error in errors:
             message = modname = error["mod"]
@@ -1259,6 +1266,7 @@ def open(self):
                 self.config.extension_pkg_whitelist
             )
         self.stats.reset_message_count()
+        self._ignore_paths = get_global_option(self, "ignore-paths")
 
     def generate_reports(self):
         """close the whole package /module, it's time to make reports !
diff --git a/pylint/utils/utils.py b/pylint/utils/utils.py
--- a/pylint/utils/utils.py
+++ b/pylint/utils/utils.py
@@ -56,16 +56,24 @@
 GLOBAL_OPTION_PATTERN = Literal[
     "no-docstring-rgx", "dummy-variables-rgx", "ignored-argument-names"
 ]
+GLOBAL_OPTION_PATTERN_LIST = Literal["ignore-paths"]
 GLOBAL_OPTION_TUPLE_INT = Literal["py-version"]
 GLOBAL_OPTION_NAMES = Union[
     GLOBAL_OPTION_BOOL,
     GLOBAL_OPTION_INT,
     GLOBAL_OPTION_LIST,
     GLOBAL_OPTION_PATTERN,
+    GLOBAL_OPTION_PATTERN_LIST,
     GLOBAL_OPTION_TUPLE_INT,
 ]
 T_GlobalOptionReturnTypes = TypeVar(
-    "T_GlobalOptionReturnTypes", bool, int, List[str], Pattern[str], Tuple[int, ...]
+    "T_GlobalOptionReturnTypes",
+    bool,
+    int,
+    List[str],
+    Pattern[str],
+    List[Pattern[str]],
+    Tuple[int, ...],
 )
 
 
@@ -220,6 +228,15 @@ def get_global_option(
     ...
 
 
+@overload
+def get_global_option(
+    checker: "BaseChecker",
+    option: GLOBAL_OPTION_PATTERN_LIST,
+    default: Optional[List[Pattern[str]]] = None,
+) -> List[Pattern[str]]:
+    ...
+
+
 @overload
 def get_global_option(
     checker: "BaseChecker",

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/config/option.py | 6 | 6 | 28 | 15 | 13204
| pylint/config/option.py | 28 | 28 | 28 | 15 | 13204
| pylint/config/option.py | 83 | 83 | 28 | 15 | 13204
| pylint/config/option.py | 125 | 125 | - | 15 | -
| pylint/config/option.py | 135 | 135 | - | 15 | -
| pylint/lint/expand_modules.py | 46 | 46 | - | 2 | -
| pylint/lint/pylinter.py | 49 | 49 | 24 | 9 | 11216
| pylint/lint/pylinter.py | 223 | 228 | - | 9 | -
| pylint/lint/pylinter.py | 1104 | 1104 | - | 9 | -
| pylint/lint/pylinter.py | 1262 | 1262 | - | 9 | -
| pylint/utils/utils.py | 59 | 59 | - | 7 | -
| pylint/utils/utils.py | 223 | 223 | 8 | 7 | 3136


## Problem Statement

```
ignore-paths: normalize path to PosixPath
### Current problem

In a project of mine, there is an entire directory, "dummy", that I want to exclude running pylint in.  I've added the directory name to the "ignore" option and it works great when used from the command line.

\`\`\`toml
# Files or directories to be skipped. They should be base names, not paths.
ignore = [
  'dummy',
]
\`\`\`

However, when using vscode, the full path is provided.  It calls pylint like this:

\`\`\`
~\Documents\<snip>\.venv\Scripts\python.exe -m pylint --msg-template='{line},{column},{category},{symbol}:{msg} --reports=n --output-format=text ~\Documents\<snip>\dummy\file.py
\`\`\`

In this case, the ignore rule doesn't work and vscode still reports errors.  So I decided to switch to the "ignore-paths" option.  The following works:

\`\`\`toml
# Add files or directories matching the regex patterns to the ignore-list. The
# regex matches against paths.
ignore-paths = [
  '.*/dummy/.*$',
  '.*\\dummy\\.*$',
]
\`\`\`

However, I need to duplciate each path, onces for Linux (/ as path separator) and the second for Windows (\ as path separator).  Would it be possible to normalize the paths (could use pathlib PosixPath) so that just the linux one would work on both systems?  Note also that vscode passes the full path, so starting the regex with a ^, like '^dummy/.*$', won't work.

### Desired solution

I'd like to be able to define the path only once in the "ignore-paths" settings.  Even better would be to respect the "ignore" setting even for a path provided with the full path (just as if it was run from the command line).

\`\`\`toml
# Add files or directories matching the regex patterns to the ignore-list. The
# regex matches against paths.
ignore-paths = [
  '.*/dummy/.*$',
]
\`\`\`

### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/pyreverse/main.py | 25 | 194| 884 | 884 | 1531 | 
| 2 | **2 pylint/lint/expand_modules.py** | 1 | 39| 297 | 1181 | 2617 | 
| 3 | 3 pylint/config/__init__.py | 37 | 126| 738 | 1919 | 4398 | 
| 4 | 4 pylint/lint/utils.py | 106 | 134| 172 | 2091 | 5312 | 
| 5 | 5 pylint/__init__.py | 12 | 46| 161 | 2252 | 6017 | 
| 6 | 5 pylint/__init__.py | 49 | 79| 328 | 2580 | 6017 | 
| 7 | 6 pylint/constants.py | 3 | 51| 351 | 2931 | 6409 | 
| **-> 8 <-** | **7 pylint/utils/utils.py** | 191 | 229| 205 | 3136 | 9139 | 
| 9 | 8 pylint/testutils/constants.py | 4 | 31| 297 | 3433 | 9477 | 
| 10 | **9 pylint/lint/pylinter.py** | 81 | 161| 690 | 4123 | 22333 | 
| 11 | 10 pylint/lint/run.py | 4 | 52| 339 | 4462 | 25746 | 
| 12 | 11 pylint/checkers/imports.py | 203 | 298| 769 | 5231 | 34471 | 
| 13 | 11 pylint/checkers/imports.py | 445 | 468| 178 | 5409 | 34471 | 
| 14 | 12 pylint/checkers/stdlib.py | 122 | 261| 1034 | 6443 | 40870 | 
| 15 | 13 pylint/checkers/variables.py | 57 | 161| 653 | 7096 | 58989 | 
| 16 | 13 pylint/checkers/stdlib.py | 40 | 72| 309 | 7405 | 58989 | 
| 17 | 14 doc/conf.py | 107 | 243| 889 | 8294 | 60835 | 
| 18 | **15 pylint/config/option.py** | 164 | 180| 176 | 8470 | 62250 | 
| 19 | 16 pylint/checkers/design_analysis.py | 107 | 185| 510 | 8980 | 67280 | 
| 20 | 17 pylint/checkers/spelling.py | 31 | 132| 548 | 9528 | 71080 | 
| 21 | 18 pylint/lint/__init__.py | 74 | 106| 170 | 9698 | 72571 | 
| 22 | 18 pylint/checkers/design_analysis.py | 29 | 106| 670 | 10368 | 72571 | 
| 23 | 18 pylint/checkers/variables.py | 263 | 308| 382 | 10750 | 72571 | 
| **-> 24 <-** | **18 pylint/lint/pylinter.py** | 4 | 79| 466 | 11216 | 72571 | 
| 25 | 19 pylint/checkers/format.py | 54 | 119| 345 | 11561 | 79640 | 
| 26 | 20 pylint/config/find_default_config_files.py | 33 | 70| 354 | 11915 | 80182 | 
| 27 | 20 pylint/checkers/stdlib.py | 74 | 119| 538 | 12453 | 80182 | 
| **-> 28 <-** | **20 pylint/config/option.py** | 4 | 105| 751 | 13204 | 80182 | 
| 29 | 20 doc/conf.py | 1 | 106| 753 | 13957 | 80182 | 
| 30 | 21 pylint/checkers/base.py | 1657 | 1697| 245 | 14202 | 102530 | 
| 31 | 21 pylint/config/find_default_config_files.py | 4 | 30| 147 | 14349 | 102530 | 
| 32 | 22 pylint/checkers/refactoring/refactoring_checker.py | 4 | 55| 358 | 14707 | 118873 | 
| 33 | 22 pylint/checkers/stdlib.py | 339 | 447| 1086 | 15793 | 118873 | 
| 34 | 23 pylint/config/option_parser.py | 4 | 48| 319 | 16112 | 119234 | 
| 35 | 24 examples/custom_raw.py | 1 | 41| 236 | 16348 | 119470 | 
| 36 | 24 pylint/checkers/base.py | 1700 | 1729| 214 | 16562 | 119470 | 
| 37 | 24 pylint/lint/run.py | 55 | 287| 1540 | 18102 | 119470 | 
| 38 | 24 pylint/checkers/base.py | 171 | 221| 466 | 18568 | 119470 | 
| 39 | 24 pylint/lint/run.py | 396 | 464| 553 | 19121 | 119470 | 
| 40 | 24 pylint/checkers/imports.py | 301 | 425| 794 | 19915 | 119470 | 
| 41 | 25 pylint/pyreverse/utils.py | 22 | 121| 531 | 20446 | 121659 | 
| 42 | 26 pylint/utils/__init__.py | 46 | 93| 268 | 20714 | 122797 | 
| 43 | 26 pylint/pyreverse/main.py | 197 | 231| 219 | 20933 | 122797 | 
| 44 | 27 pylint/checkers/classes.py | 51 | 106| 404 | 21337 | 141939 | 
| 45 | 28 pylint/checkers/utils.py | 119 | 233| 697 | 22034 | 154615 | 
| 46 | 28 pylint/lint/utils.py | 63 | 103| 318 | 22352 | 154615 | 
| 47 | 28 pylint/checkers/base.py | 67 | 92| 146 | 22498 | 154615 | 
| 48 | 29 pylint/pyreverse/dot_printer.py | 13 | 38| 246 | 22744 | 156037 | 
| 49 | 30 pylint/epylint.py | 1 | 74| 123 | 22867 | 157998 | 
| 50 | 30 pylint/lint/utils.py | 4 | 60| 377 | 23244 | 157998 | 
| 51 | 30 pylint/checkers/stdlib.py | 449 | 499| 545 | 23789 | 157998 | 
| 52 | 31 pylint/checkers/similar.py | 44 | 103| 376 | 24165 | 166088 | 
| 53 | 32 pylint/lint/parallel.py | 4 | 46| 282 | 24447 | 167451 | 
| 54 | 32 pylint/checkers/refactoring/refactoring_checker.py | 612 | 631| 153 | 24600 | 167451 | 
| 55 | 32 pylint/checkers/similar.py | 863 | 879| 163 | 24763 | 167451 | 
| 56 | 32 pylint/checkers/base.py | 95 | 106| 127 | 24890 | 167451 | 
| 57 | 33 pylint/checkers/exceptions.py | 84 | 187| 1007 | 25897 | 172616 | 
| 58 | 34 pylint/pyreverse/__init__.py | 1 | 9| 0 | 25897 | 172671 | 
| 59 | 35 pylint/extensions/typing.py | 1 | 68| 832 | 26729 | 175235 | 
| 60 | 35 pylint/checkers/imports.py | 123 | 149| 207 | 26936 | 175235 | 
| 61 | 35 pylint/checkers/variables.py | 381 | 510| 1123 | 28059 | 175235 | 
| 62 | 36 pylint/config/option_manager_mixin.py | 5 | 44| 269 | 28328 | 178220 | 
| 63 | 36 pylint/checkers/format.py | 689 | 713| 198 | 28526 | 178220 | 
| 64 | **36 pylint/lint/pylinter.py** | 724 | 753| 296 | 28822 | 178220 | 
| 65 | 37 pylint/pyreverse/vcg_printer.py | 23 | 121| 698 | 29520 | 180634 | 
| 66 | 37 pylint/checkers/variables.py | 2103 | 2183| 675 | 30195 | 180634 | 
| 67 | 37 pylint/checkers/stdlib.py | 264 | 306| 200 | 30395 | 180634 | 
| 68 | 38 pylint/checkers/refactoring/recommendation_checker.py | 3 | 79| 697 | 31092 | 184020 | 
| 69 | 39 doc/exts/pylint_extensions.py | 1 | 22| 115 | 31207 | 184962 | 
| 70 | 39 pylint/checkers/similar.py | 898 | 945| 324 | 31531 | 184962 | 
| 71 | 40 pylint/testutils/functional_test_file.py | 4 | 74| 420 | 31951 | 185423 | 
| 72 | 41 pylint/testutils/__init__.py | 32 | 59| 216 | 32167 | 186250 | 
| 73 | **41 pylint/lint/pylinter.py** | 1496 | 1519| 220 | 32387 | 186250 | 
| 74 | 42 pylint/graph.py | 23 | 84| 369 | 32756 | 188006 | 
| 75 | 43 pylint/config/man_help_formatter.py | 4 | 37| 237 | 32993 | 188851 | 
| 76 | 44 pylint/utils/file_state.py | 4 | 30| 111 | 33104 | 190281 | 
| 77 | 44 pylint/checkers/utils.py | 57 | 118| 374 | 33478 | 190281 | 
| 78 | 45 pylint/testutils/global_test_linter.py | 5 | 21| 111 | 33589 | 190433 | 
| 79 | 45 pylint/checkers/refactoring/refactoring_checker.py | 1166 | 1207| 365 | 33954 | 190433 | 
| 80 | 46 pylint/checkers/strings.py | 216 | 237| 129 | 34083 | 199156 | 
| 81 | 46 pylint/checkers/imports.py | 805 | 828| 265 | 34348 | 199156 | 
| 82 | **46 pylint/lint/pylinter.py** | 755 | 783| 257 | 34605 | 199156 | 


### Hint

```
Thank you for opening the issue, this seems like a sensible thing to do.
```

## Patch

```diff
diff --git a/pylint/config/option.py b/pylint/config/option.py
--- a/pylint/config/option.py
+++ b/pylint/config/option.py
@@ -3,7 +3,9 @@
 
 import copy
 import optparse  # pylint: disable=deprecated-module
+import pathlib
 import re
+from typing import List, Pattern
 
 from pylint import utils
 
@@ -25,6 +27,19 @@ def _regexp_csv_validator(_, name, value):
     return [_regexp_validator(_, name, val) for val in _csv_validator(_, name, value)]
 
 
+def _regexp_paths_csv_validator(_, name: str, value: str) -> List[Pattern[str]]:
+    patterns = []
+    for val in _csv_validator(_, name, value):
+        patterns.append(
+            re.compile(
+                str(pathlib.PureWindowsPath(val)).replace("\\", "\\\\")
+                + "|"
+                + pathlib.PureWindowsPath(val).as_posix()
+            )
+        )
+    return patterns
+
+
 def _choice_validator(choices, name, value):
     if value not in choices:
         msg = "option %s: invalid value: %r, should be in %s"
@@ -80,6 +95,7 @@ def _py_version_validator(_, name, value):
     "float": float,
     "regexp": lambda pattern: re.compile(pattern or ""),
     "regexp_csv": _regexp_csv_validator,
+    "regexp_paths_csv": _regexp_paths_csv_validator,
     "csv": _csv_validator,
     "yn": _yn_validator,
     "choice": lambda opt, name, value: _choice_validator(opt["choices"], name, value),
@@ -122,6 +138,7 @@ class Option(optparse.Option):
     TYPES = optparse.Option.TYPES + (
         "regexp",
         "regexp_csv",
+        "regexp_paths_csv",
         "csv",
         "yn",
         "multiple_choice",
@@ -132,6 +149,7 @@ class Option(optparse.Option):
     TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
     TYPE_CHECKER["regexp"] = _regexp_validator
     TYPE_CHECKER["regexp_csv"] = _regexp_csv_validator
+    TYPE_CHECKER["regexp_paths_csv"] = _regexp_paths_csv_validator
     TYPE_CHECKER["csv"] = _csv_validator
     TYPE_CHECKER["yn"] = _yn_validator
     TYPE_CHECKER["multiple_choice"] = _multiple_choices_validating_option
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -43,7 +43,7 @@ def expand_modules(
     files_or_modules: List[str],
     ignore_list: List[str],
     ignore_list_re: List[Pattern],
-    ignore_list_paths_re: List[Pattern],
+    ignore_list_paths_re: List[Pattern[str]],
 ) -> Tuple[List[ModuleDescriptionDict], List[ErrorDescriptionDict]]:
     """take a list of files/modules/packages and return the list of tuple
     (file, module name) which have to be actually checked
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -46,7 +46,14 @@
     MessageLocationTuple,
     ModuleDescriptionDict,
 )
-from pylint.utils import ASTWalker, FileState, LinterStats, ModuleStats, utils
+from pylint.utils import (
+    ASTWalker,
+    FileState,
+    LinterStats,
+    ModuleStats,
+    get_global_option,
+    utils,
+)
 from pylint.utils.pragma_parser import (
     OPTION_PO,
     InvalidPragmaError,
@@ -220,12 +227,12 @@ def make_options():
             (
                 "ignore-paths",
                 {
-                    "type": "regexp_csv",
+                    "type": "regexp_paths_csv",
                     "metavar": "<pattern>[,<pattern>...]",
-                    "dest": "ignore_list_paths_re",
-                    "default": (),
-                    "help": "Add files or directories matching the regex patterns to the"
-                    " ignore-list. The regex matches against paths.",
+                    "default": [],
+                    "help": "Add files or directories matching the regex patterns to the "
+                    "ignore-list. The regex matches against paths and can be in "
+                    "Posix or Windows format.",
                 },
             ),
             (
@@ -1101,7 +1108,7 @@ def _expand_files(self, modules) -> List[ModuleDescriptionDict]:
             modules,
             self.config.black_list,
             self.config.black_list_re,
-            self.config.ignore_list_paths_re,
+            self._ignore_paths,
         )
         for error in errors:
             message = modname = error["mod"]
@@ -1259,6 +1266,7 @@ def open(self):
                 self.config.extension_pkg_whitelist
             )
         self.stats.reset_message_count()
+        self._ignore_paths = get_global_option(self, "ignore-paths")
 
     def generate_reports(self):
         """close the whole package /module, it's time to make reports !
diff --git a/pylint/utils/utils.py b/pylint/utils/utils.py
--- a/pylint/utils/utils.py
+++ b/pylint/utils/utils.py
@@ -56,16 +56,24 @@
 GLOBAL_OPTION_PATTERN = Literal[
     "no-docstring-rgx", "dummy-variables-rgx", "ignored-argument-names"
 ]
+GLOBAL_OPTION_PATTERN_LIST = Literal["ignore-paths"]
 GLOBAL_OPTION_TUPLE_INT = Literal["py-version"]
 GLOBAL_OPTION_NAMES = Union[
     GLOBAL_OPTION_BOOL,
     GLOBAL_OPTION_INT,
     GLOBAL_OPTION_LIST,
     GLOBAL_OPTION_PATTERN,
+    GLOBAL_OPTION_PATTERN_LIST,
     GLOBAL_OPTION_TUPLE_INT,
 ]
 T_GlobalOptionReturnTypes = TypeVar(
-    "T_GlobalOptionReturnTypes", bool, int, List[str], Pattern[str], Tuple[int, ...]
+    "T_GlobalOptionReturnTypes",
+    bool,
+    int,
+    List[str],
+    Pattern[str],
+    List[Pattern[str]],
+    Tuple[int, ...],
 )
 
 
@@ -220,6 +228,15 @@ def get_global_option(
     ...
 
 
+@overload
+def get_global_option(
+    checker: "BaseChecker",
+    option: GLOBAL_OPTION_PATTERN_LIST,
+    default: Optional[List[Pattern[str]]] = None,
+) -> List[Pattern[str]]:
+    ...
+
+
 @overload
 def get_global_option(
     checker: "BaseChecker",

```

## Test Patch

```diff
diff --git a/tests/lint/unittest_expand_modules.py b/tests/lint/unittest_expand_modules.py
--- a/tests/lint/unittest_expand_modules.py
+++ b/tests/lint/unittest_expand_modules.py
@@ -4,10 +4,14 @@
 
 import re
 from pathlib import Path
+from typing import Dict, Tuple, Type
 
 import pytest
 
+from pylint.checkers import BaseChecker
 from pylint.lint.expand_modules import _is_in_ignore_list_re, expand_modules
+from pylint.testutils import CheckerTestCase, set_config
+from pylint.utils.utils import get_global_option
 
 
 def test__is_in_ignore_list_re_match() -> None:
@@ -21,17 +25,6 @@ def test__is_in_ignore_list_re_match() -> None:
     assert _is_in_ignore_list_re("src/tests/whatever.xml", patterns)
 
 
-def test__is_in_ignore_list_re_nomatch() -> None:
-    patterns = [
-        re.compile(".*enchilada.*"),
-        re.compile("unittest_.*"),
-        re.compile(".*tests/.*"),
-    ]
-    assert not _is_in_ignore_list_re("test_utils.py", patterns)
-    assert not _is_in_ignore_list_re("enchilad.py", patterns)
-    assert not _is_in_ignore_list_re("src/tests.py", patterns)
-
-
 TEST_DIRECTORY = Path(__file__).parent.parent
 INIT_PATH = str(TEST_DIRECTORY / "lint/__init__.py")
 EXPAND_MODULES = str(TEST_DIRECTORY / "lint/unittest_expand_modules.py")
@@ -84,27 +77,70 @@ def test__is_in_ignore_list_re_nomatch() -> None:
 }
 
 
-@pytest.mark.parametrize(
-    "files_or_modules,expected",
-    [
-        ([__file__], [this_file]),
-        (
-            [Path(__file__).parent],
-            [
-                init_of_package,
-                test_pylinter,
-                test_utils,
-                this_file_from_init,
-                unittest_lint,
-            ],
-        ),
-    ],
-)
-def test_expand_modules(files_or_modules, expected):
-    ignore_list, ignore_list_re, ignore_list_paths_re = [], [], []
-    modules, errors = expand_modules(
-        files_or_modules, ignore_list, ignore_list_re, ignore_list_paths_re
+class TestExpandModules(CheckerTestCase):
+    """Test the expand_modules function while allowing options to be set"""
+
+    class Checker(BaseChecker):
+        """This dummy checker is needed to allow options to be set"""
+
+        name = "checker"
+        msgs: Dict[str, Tuple[str, ...]] = {}
+        options = (("An option", {"An option": "dict"}),)
+
+    CHECKER_CLASS: Type = Checker
+
+    @pytest.mark.parametrize(
+        "files_or_modules,expected",
+        [
+            ([__file__], [this_file]),
+            (
+                [str(Path(__file__).parent)],
+                [
+                    init_of_package,
+                    test_pylinter,
+                    test_utils,
+                    this_file_from_init,
+                    unittest_lint,
+                ],
+            ),
+        ],
+    )
+    @set_config(ignore_paths="")
+    def test_expand_modules(self, files_or_modules, expected):
+        """Test expand_modules with the default value of ignore-paths"""
+        ignore_list, ignore_list_re = [], []
+        modules, errors = expand_modules(
+            files_or_modules,
+            ignore_list,
+            ignore_list_re,
+            get_global_option(self, "ignore-paths"),
+        )
+        modules.sort(key=lambda d: d["name"])
+        assert modules == expected
+        assert not errors
+
+    @pytest.mark.parametrize(
+        "files_or_modules,expected",
+        [
+            ([__file__], []),
+            (
+                [str(Path(__file__).parent)],
+                [
+                    init_of_package,
+                ],
+            ),
+        ],
     )
-    modules.sort(key=lambda d: d["name"])
-    assert modules == expected
-    assert not errors
+    @set_config(ignore_paths=".*/lint/.*")
+    def test_expand_modules_with_ignore(self, files_or_modules, expected):
+        """Test expand_modules with a non-default value of ignore-paths"""
+        ignore_list, ignore_list_re = [], []
+        modules, errors = expand_modules(
+            files_or_modules,
+            ignore_list,
+            ignore_list_re,
+            get_global_option(self.checker, "ignore-paths"),
+        )
+        modules.sort(key=lambda d: d["name"])
+        assert modules == expected
+        assert not errors
diff --git a/tests/unittest_config.py b/tests/unittest_config.py
--- a/tests/unittest_config.py
+++ b/tests/unittest_config.py
@@ -16,10 +16,14 @@
 
 import re
 import sre_constants
+from typing import Dict, Tuple, Type
 
 import pytest
 
 from pylint import config
+from pylint.checkers import BaseChecker
+from pylint.testutils import CheckerTestCase, set_config
+from pylint.utils.utils import get_global_option
 
 RE_PATTERN_TYPE = getattr(re, "Pattern", getattr(re, "_pattern_type", None))
 
@@ -65,3 +69,33 @@ def test__regexp_csv_validator_invalid() -> None:
     pattern_strings = ["test_.*", "foo\\.bar", "^baz)$"]
     with pytest.raises(sre_constants.error):
         config.option._regexp_csv_validator(None, None, ",".join(pattern_strings))
+
+
+class TestPyLinterOptionSetters(CheckerTestCase):
+    """Class to check the set_config decorator and get_global_option util
+    for options declared in PyLinter."""
+
+    class Checker(BaseChecker):
+        name = "checker"
+        msgs: Dict[str, Tuple[str, ...]] = {}
+        options = (("An option", {"An option": "dict"}),)
+
+    CHECKER_CLASS: Type = Checker
+
+    @set_config(ignore_paths=".*/tests/.*,.*\\ignore\\.*")
+    def test_ignore_paths_with_value(self) -> None:
+        """Test ignore-paths option with value"""
+        options = get_global_option(self.checker, "ignore-paths")
+
+        assert any(i.match("dir/tests/file.py") for i in options)
+        assert any(i.match("dir\\tests\\file.py") for i in options)
+        assert any(i.match("dir/ignore/file.py") for i in options)
+        assert any(i.match("dir\\ignore\\file.py") for i in options)
+
+    def test_ignore_paths_with_no_value(self) -> None:
+        """Test ignore-paths option with no value.
+        Compare against actual list to see if validator works."""
+        options = get_global_option(self.checker, "ignore-paths")
+
+        # pylint: disable-next=use-implicit-booleaness-not-comparison
+        assert options == []

```


## Code snippets

### 1 - pylint/pyreverse/main.py:

Start line: 25, End line: 194

```python
import sys
from typing import Iterable

from pylint.config import ConfigurationMixIn
from pylint.lint.utils import fix_import_path
from pylint.pyreverse import writer
from pylint.pyreverse.diadefslib import DiadefsHandler
from pylint.pyreverse.inspector import Linker, project_from_files
from pylint.pyreverse.utils import check_graphviz_availability, insert_default_options

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
            help="create a *.<format> output file if format available.",
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
            type="string",
            short="d",
            action="store",
            metavar="<output_directory>",
            help="set the output directory path.",
        ),
    ),
)
```
### 2 - pylint/lint/expand_modules.py:

Start line: 1, End line: 39

```python
import os
import sys
from typing import List, Pattern, Tuple

from astroid import modutils

from pylint.typing import ErrorDescriptionDict, ModuleDescriptionDict


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


def _is_in_ignore_list_re(element: str, ignore_list_re: List[Pattern]) -> bool:
    """determines if the element is matched in a regex ignore-list"""
    for file_pattern in ignore_list_re:
        if file_pattern.match(element):
            return True
    return False
```
### 3 - pylint/config/__init__.py:

Start line: 37, End line: 126

```python
import os
import pathlib
import pickle
import sys
from datetime import datetime

import platformdirs

from pylint.config.configuration_mixin import ConfigurationMixIn
from pylint.config.find_default_config_files import find_default_config_files
from pylint.config.man_help_formatter import _ManHelpFormatter
from pylint.config.option import Option
from pylint.config.option_manager_mixin import OptionsManagerMixIn
from pylint.config.option_parser import OptionParser
from pylint.config.options_provider_mixin import OptionsProviderMixIn, UnsupportedAction
from pylint.utils import LinterStats

__all__ = [
    "ConfigurationMixIn",
    "find_default_config_files",
    "_ManHelpFormatter",
    "Option",
    "OptionsManagerMixIn",
    "OptionParser",
    "OptionsProviderMixIn",
    "UnsupportedAction",
]

USER_HOME = os.path.expanduser("~")
if "PYLINTHOME" in os.environ:
    PYLINT_HOME = os.environ["PYLINTHOME"]
    if USER_HOME == "~":
        USER_HOME = os.path.dirname(PYLINT_HOME)
elif USER_HOME == "~":
    PYLINT_HOME = ".pylint.d"
else:
    PYLINT_HOME = platformdirs.user_cache_dir("pylint")
    # The spam prevention is due to pylint being used in parallel by
    # pre-commit, and the message being spammy in this context
    # Also if you work with old version of pylint that recreate the
    # old pylint home, you can get the old message for a long time.
    prefix_spam_prevention = "pylint_warned_about_old_cache_already"
    spam_prevention_file = os.path.join(
        PYLINT_HOME,
        datetime.now().strftime(prefix_spam_prevention + "_%Y-%m-%d.temp"),
    )
    old_home = os.path.join(USER_HOME, ".pylint.d")
    if os.path.exists(old_home) and not os.path.exists(spam_prevention_file):
        print(
            f"PYLINTHOME is now '{PYLINT_HOME}' but obsolescent '{old_home}' is found; "
            "you can safely remove the latter",
            file=sys.stderr,
        )
        # Remove old spam prevention file
        if os.path.exists(PYLINT_HOME):
            for filename in os.listdir(PYLINT_HOME):
                if prefix_spam_prevention in filename:
                    try:
                        os.remove(os.path.join(PYLINT_HOME, filename))
                    except OSError:
                        pass

        # Create spam prevention file for today
        try:
            pathlib.Path(PYLINT_HOME).mkdir(parents=True, exist_ok=True)
            with open(spam_prevention_file, "w", encoding="utf8") as f:
                f.write("")
        except Exception:  # pylint: disable=broad-except
            # Can't write in PYLINT_HOME ?
            print(
                "Can't write the file that was supposed to "
                f"prevent pylint.d deprecation spam in {PYLINT_HOME}."
            )


def _get_pdata_path(base_name, recurs):
    base_name = base_name.replace(os.sep, "_")
    return os.path.join(PYLINT_HOME, f"{base_name}{recurs}.stats")


def load_results(base):
    data_file = _get_pdata_path(base, 1)
    try:
        with open(data_file, "rb") as stream:
            data = pickle.load(stream)
            if not isinstance(data, LinterStats):
                raise TypeError
            return data
    except Exception:  # pylint: disable=broad-except
        return None
```
### 4 - pylint/lint/utils.py:

Start line: 106, End line: 134

```python
def _patch_sys_path(args):
    original = list(sys.path)
    changes = []
    seen = set()
    for arg in args:
        path = get_python_path(arg)
        if path not in seen:
            changes.append(path)
            seen.add(path)

    sys.path[:] = changes + sys.path
    return original


@contextlib.contextmanager
def fix_import_path(args):
    """Prepare sys.path for running the linter checks.

    Within this context, each of the given arguments is importable.
    Paths are added to sys.path in corresponding order to the arguments.
    We avoid adding duplicate directories to sys.path.
    `sys.path` is reset to its original value upon exiting this context.
    """
    original = _patch_sys_path(args)
    try:
        yield
    finally:
        sys.path[:] = original
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
### 6 - pylint/__init__.py:

Start line: 49, End line: 79

```python
def modify_sys_path() -> None:
    """Modify sys path for execution as Python module.

    Strip out the current working directory from sys.path.
    Having the working directory in `sys.path` means that `pylint` might
    inadvertently import user code from modules having the same name as
    stdlib or pylint's own modules.
    CPython issue: https://bugs.python.org/issue33053

    - Remove the first entry. This will always be either "" or the working directory
    - Remove the working directory from the second and third entries
      if PYTHONPATH includes a ":" at the beginning or the end.
      https://github.com/PyCQA/pylint/issues/3636
      Don't remove it if PYTHONPATH contains the cwd or '.' as the entry will
      only be added once.
    - Don't remove the working directory from the rest. It will be included
      if pylint is installed in an editable configuration (as the last item).
      https://github.com/PyCQA/pylint/issues/4161
    """
    sys.path.pop(0)
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    cwd = os.getcwd()
    if env_pythonpath.startswith(":") and env_pythonpath not in (f":{cwd}", ":."):
        sys.path.pop(0)
    elif env_pythonpath.endswith(":") and env_pythonpath not in (f"{cwd}:", ".:"):
        sys.path.pop(1)


version = __version__
__all__ = ["__version__", "version", "modify_sys_path"]
```
### 7 - pylint/constants.py:

Start line: 3, End line: 51

```python
import platform
import sys

import astroid

from pylint.__pkginfo__ import __version__

PY38_PLUS = sys.version_info[:2] >= (3, 8)
PY39_PLUS = sys.version_info[:2] >= (3, 9)

IS_PYPY = platform.python_implementation() == "PyPy"

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
### 8 - pylint/utils/utils.py:

Start line: 191, End line: 229

```python
@overload
def get_global_option(
    checker: "BaseChecker", option: GLOBAL_OPTION_BOOL, default: Optional[bool] = None
) -> bool:
    ...


@overload
def get_global_option(
    checker: "BaseChecker", option: GLOBAL_OPTION_INT, default: Optional[int] = None
) -> int:
    ...


@overload
def get_global_option(
    checker: "BaseChecker",
    option: GLOBAL_OPTION_LIST,
    default: Optional[List[str]] = None,
) -> List[str]:
    ...


@overload
def get_global_option(
    checker: "BaseChecker",
    option: GLOBAL_OPTION_PATTERN,
    default: Optional[Pattern[str]] = None,
) -> Pattern[str]:
    ...


@overload
def get_global_option(
    checker: "BaseChecker",
    option: GLOBAL_OPTION_TUPLE_INT,
    default: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    ...
```
### 9 - pylint/testutils/constants.py:

Start line: 4, End line: 31

```python
import operator
import re
import sys
from os.path import abspath, dirname
from pathlib import Path

SYS_VERS_STR = (
    "%d%d%d" % sys.version_info[:3]  # pylint: disable=consider-using-f-string
)
TITLE_UNDERLINES = ["", "=", "-", "."]
PREFIX = abspath(dirname(__file__))
UPDATE_OPTION = "--update-functional-output"
UPDATE_FILE = Path("pylint-functional-test-update")
# Common sub-expressions.
_MESSAGE = {"msg": r"[a-z][a-z\-]+"}
# Matches a #,
#  - followed by a comparison operator and a Python version (optional),
#  - followed by a line number with a +/- (optional),
#  - followed by a list of bracketed message symbols.
# Used to extract expected messages from testdata files.
_EXPECTED_RE = re.compile(
    r"\s*#\s*(?:(?P<line>[+-]?[0-9]+):)?"  # pylint: disable=consider-using-f-string
    r"(?:(?P<op>[><=]+) *(?P<version>[0-9.]+):)?"
    r"\s*\[(?P<msgs>%(msg)s(?:,\s*%(msg)s)*)]" % _MESSAGE
)

_OPERATORS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
```
### 10 - pylint/lint/pylinter.py:

Start line: 81, End line: 161

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
    "E0013": (
        "Plugin '%s' is impossible to load, is it installed ? ('%s')",
        "bad-plugin-value",
        "Used when a bad value is used in 'load-plugins'.",
    ),
}
```
### 18 - pylint/config/option.py:

Start line: 164, End line: 180

```python
class Option(optparse.Option):

    # pylint: disable=unsupported-assignment-operation
    optparse.Option.CHECK_METHODS[2] = _check_choice  # type: ignore

    def process(self, opt, value, values, parser):
        # First, convert the value(s) to the right type.  Howl if any
        # value(s) are bogus.
        value = self.convert_value(opt, value)
        if self.type == "named":
            existent = getattr(values, self.dest)
            if existent:
                existent.update(value)
                value = existent
        # And then take whatever action is expected of us.
        # This is a separate method to make life easier for
        # subclasses to add new actions.
        return self.take_action(self.action, self.dest, opt, value, values, parser)
```
### 24 - pylint/lint/pylinter.py:

Start line: 4, End line: 79

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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import astroid
from astroid import AstroidError, nodes

from pylint import checkers, config, exceptions, interfaces, reporters
from pylint.constants import (
    MAIN_CHECKER_NAME,
    MSG_STATE_CONFIDENCE,
    MSG_STATE_SCOPE_CONFIG,
    MSG_STATE_SCOPE_MODULE,
    MSG_TYPES,
    MSG_TYPES_LONG,
    MSG_TYPES_STATUS,
)
from pylint.lint.expand_modules import expand_modules
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
from pylint.reporters.ureports import nodes as report_nodes
from pylint.typing import (
    FileItem,
    ManagedMessage,
    MessageLocationTuple,
    ModuleDescriptionDict,
)
from pylint.utils import ASTWalker, FileState, LinterStats, ModuleStats, utils
from pylint.utils.pragma_parser import (
    OPTION_PO,
    InvalidPragmaError,
    UnRecognizedOptionError,
    parse_pragma,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

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
### 28 - pylint/config/option.py:

Start line: 4, End line: 105

```python
import copy
import optparse  # pylint: disable=deprecated-module
import re

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


def _choice_validator(choices, name, value):
    if value not in choices:
        msg = "option %s: invalid value: %r, should be in %s"
        raise optparse.OptionValueError(msg % (name, value, choices))
    return value


def _yn_validator(opt, _, value):
    if isinstance(value, int):
        return bool(value)
    if value in ("y", "yes", "true"):
        return True
    if value in ("n", "no", "false"):
        return False
    msg = "option %s: invalid yn value %r, should be in (y, yes, true, n, no, false)"
    raise optparse.OptionValueError(msg % (opt, value))


def _multiple_choice_validator(choices, name, value):
    values = utils._check_csv(value)
    for csv_value in values:
        if csv_value not in choices:
            msg = "option %s: invalid value: %r, should be in %s"
            raise optparse.OptionValueError(msg % (name, csv_value, choices))
    return values


def _non_empty_string_validator(opt, _, value):
    if not value:
        msg = "indent string can't be empty."
        raise optparse.OptionValueError(msg)
    return utils._unquote(value)


def _multiple_choices_validating_option(opt, name, value):
    return _multiple_choice_validator(opt.choices, name, value)


def _py_version_validator(_, name, value):
    if not isinstance(value, tuple):
        try:
            value = tuple(int(val) for val in value.split("."))
        except (ValueError, AttributeError):
            raise optparse.OptionValueError(
                f"Invalid format for {name}, should be version string. E.g., '3.8'"
            ) from None
    return value


VALIDATORS = {
    "string": utils._unquote,
    "int": int,
    "float": float,
    "regexp": lambda pattern: re.compile(pattern or ""),
    "regexp_csv": _regexp_csv_validator,
    "csv": _csv_validator,
    "yn": _yn_validator,
    "choice": lambda opt, name, value: _choice_validator(opt["choices"], name, value),
    "multiple_choice": lambda opt, name, value: _multiple_choice_validator(
        opt["choices"], name, value
    ),
    "non_empty_string": _non_empty_string_validator,
    "py_version": _py_version_validator,
}


def _call_validator(opttype, optdict, option, value):
    if opttype not in VALIDATORS:
        raise Exception(f'Unsupported type "{opttype}"')
    try:
        return VALIDATORS[opttype](optdict, option, value)
    except TypeError:
        try:
            return VALIDATORS[opttype](value)
        except Exception as e:
            raise optparse.OptionValueError(
                f"{option} value ({value!r}) should be of type {opttype}"
            ) from e
```
### 64 - pylint/lint/pylinter.py:

Start line: 724, End line: 753

```python
class PyLinter(
    config.OptionsManagerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def enable_fail_on_messages(self):
        """enable 'fail on' msgs

        Convert values in config.fail_on (which might be msg category, msg id,
        or symbol) to specific msgs, then enable and flag them for later.
        """
        fail_on_vals = self.config.fail_on
        if not fail_on_vals:
            return

        fail_on_cats = set()
        fail_on_msgs = set()
        for val in fail_on_vals:
            # If value is a cateogry, add category, else add message
            if val in MSG_TYPES:
                fail_on_cats.add(val)
            else:
                fail_on_msgs.add(val)

        # For every message in every checker, if cat or msg flagged, enable check
        for all_checkers in self._checkers.values():
            for checker in all_checkers:
                for msg in checker.messages:
                    if msg.msgid in fail_on_msgs or msg.symbol in fail_on_msgs:
                        # message id/symbol matched, enable and flag it
                        self.enable(msg.msgid)
                        self.fail_on_symbols.append(msg.symbol)
                    elif msg.msgid[0] in fail_on_cats:
                        # message starts with a cateogry value, flag (but do not enable) it
                        self.fail_on_symbols.append(msg.symbol)
```
### 73 - pylint/lint/pylinter.py:

Start line: 1496, End line: 1519

```python
class PyLinter(
    config.OptionsManagerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def add_ignored_message(
        self,
        msgid: str,
        line: int,
        node: Optional[nodes.NodeNG] = None,
        confidence: Optional[interfaces.Confidence] = interfaces.UNDEFINED,
    ) -> None:
        """Prepares a message to be added to the ignored message storage

        Some checks return early in special cases and never reach add_message(),
        even though they would normally issue a message.
        This creates false positives for useless-suppression.
        This function avoids this by adding those message to the ignored msgs attribute
        """
        message_definitions = self.msgs_store.get_message_definitions(msgid)
        for message_definition in message_definitions:
            message_definition.check_message_definition(line, node)
            self.file_state.handle_ignored_message(
                self._get_message_state_scope(
                    message_definition.msgid, line, confidence
                ),
                message_definition.msgid,
                line,
            )
```
### 82 - pylint/lint/pylinter.py:

Start line: 755, End line: 783

```python
class PyLinter(
    config.OptionsManagerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def any_fail_on_issues(self):
        return self.stats and any(
            x in self.fail_on_symbols for x in self.stats.by_msg.keys()
        )

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

    def error_mode(self):
        """error mode: enable only errors; no reports, no persistent"""
        self._error_mode = True
        self.disable_noerror_messages()
        self.disable("miscellaneous")
        self.set_option("reports", False)
        self.set_option("persistent", False)
        self.set_option("score", False)
```
