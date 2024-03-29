# pylint-dev__pylint-8124

| **pylint-dev/pylint** | `eb950615d77a6b979af6e0d9954fdb4197f4a722` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 7552 |
| **Avg pos** | 59.0 |
| **Min pos** | 13 |
| **Max pos** | 46 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/checkers/imports.py b/pylint/checkers/imports.py
--- a/pylint/checkers/imports.py
+++ b/pylint/checkers/imports.py
@@ -439,6 +439,15 @@ class ImportsChecker(DeprecatedMixin, BaseChecker):
                 "help": "Allow wildcard imports from modules that define __all__.",
             },
         ),
+        (
+            "allow-reexport-from-package",
+            {
+                "default": False,
+                "type": "yn",
+                "metavar": "<y or n>",
+                "help": "Allow explicit reexports by alias from a package __init__.",
+            },
+        ),
     )
 
     def __init__(self, linter: PyLinter) -> None:
@@ -461,6 +470,7 @@ def open(self) -> None:
         self.linter.stats = self.linter.stats
         self.import_graph = defaultdict(set)
         self._module_pkg = {}  # mapping of modules to the pkg they belong in
+        self._current_module_package = False
         self._excluded_edges: defaultdict[str, set[str]] = defaultdict(set)
         self._ignored_modules: Sequence[str] = self.linter.config.ignored_modules
         # Build a mapping {'module': 'preferred-module'}
@@ -470,6 +480,7 @@ def open(self) -> None:
             if ":" in module
         )
         self._allow_any_import_level = set(self.linter.config.allow_any_import_level)
+        self._allow_reexport_package = self.linter.config.allow_reexport_from_package
 
     def _import_graph_without_ignored_edges(self) -> defaultdict[str, set[str]]:
         filtered_graph = copy.deepcopy(self.import_graph)
@@ -495,6 +506,10 @@ def deprecated_modules(self) -> set[str]:
                 all_deprecated_modules = all_deprecated_modules.union(mod_set)
         return all_deprecated_modules
 
+    def visit_module(self, node: nodes.Module) -> None:
+        """Store if current module is a package, i.e. an __init__ file."""
+        self._current_module_package = node.package
+
     def visit_import(self, node: nodes.Import) -> None:
         """Triggered when an import statement is seen."""
         self._check_reimport(node)
@@ -917,8 +932,11 @@ def _check_import_as_rename(self, node: ImportNode) -> None:
             if import_name != aliased_name:
                 continue
 
-            if len(splitted_packages) == 1:
-                self.add_message("useless-import-alias", node=node)
+            if len(splitted_packages) == 1 and (
+                self._allow_reexport_package is False
+                or self._current_module_package is False
+            ):
+                self.add_message("useless-import-alias", node=node, confidence=HIGH)
             elif len(splitted_packages) == 2:
                 self.add_message(
                     "consider-using-from-import",

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/checkers/imports.py | 442 | 442 | 13 | 1 | 7552
| pylint/checkers/imports.py | 464 | 464 | - | 1 | -
| pylint/checkers/imports.py | 473 | 473 | - | 1 | -
| pylint/checkers/imports.py | 498 | 498 | - | 1 | -
| pylint/checkers/imports.py | 920 | 921 | 46 | 1 | 18695


## Problem Statement

```
false positive 'useless-import-alias' error for mypy-compatible explicit re-exports
### Bug description

Suppose a package has the following layout:
\`\`\`console
package/
  _submodule1.py  # defines Api1
  _submodule2.py  # defines Api2
  __init__.py     # imports and re-exports Api1 and Api2
\`\`\`
Since the submodules here implement public APIs, `__init__.py` imports and re-exports them, expecting users to import them from the public, top-level package, e.g. `from package import Api1`.

Since the implementations of `Api1` and `Api2` are complex, they are split into `_submodule1.py` and `_submodule2.py` for better maintainability and separation of concerns.

So `__init__.py` looks like this:
\`\`\`python
from ._submodule1 import Api1 as Api1
from ._submodule2 import APi2 as Api2
\`\`\`

The reason for the `as` aliases here is to be explicit that these imports are for the purpose of re-export (without having to resort to defining `__all__`, which is error-prone). Without the `as` aliases, popular linters such as `mypy` will raise an "implicit re-export" error ([docs](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport) -- part of `mypy --strict`).

However, pylint does not currently understand this usage, and raises "useless-import-alias" errors.

Example real-world code triggering pylint false positive errors: https://github.com/jab/bidict/blob/caf703e959ed4471bc391a7794411864c1d6ab9d/bidict/__init__.py#L61-L78

### Configuration

_No response_

### Command used

\`\`\`shell
pylint
\`\`\`


### Pylint output

\`\`\`shell
************* Module bidict
bidict/__init__.py:61:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:61:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:62:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:62:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:62:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:63:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:63:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:64:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:65:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:66:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:66:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:67:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:68:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:69:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:69:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:69:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:70:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:70:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:70:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:70:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:70:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:71:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:71:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:72:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:72:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:72:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:73:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
bidict/__init__.py:74:0: C0414: Import alias does not rename original package (useless-import-alias)
\`\`\`


### Expected behavior

No "useless-import-alias" errors should be flagged.

### Pylint version

\`\`\`shell
pylint 2.13.2
astroid 2.11.2
Python 3.10.2 (main, Feb  2 2022, 07:36:01) [Clang 12.0.0 (clang-1200.0.32.29)]
\`\`\`


### OS / Environment

_No response_

### Additional dependencies

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 pylint/checkers/imports.py** | 7 | 78| 549 | 549 | 8680 | 
| 2 | **1 pylint/checkers/imports.py** | 234 | 324| 764 | 1313 | 8680 | 
| 3 | 2 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 1313 | 8693 | 
| 4 | 3 doc/data/messages/u/useless-import-alias/good.py | 1 | 2| 0 | 1313 | 8698 | 
| 5 | 4 pylint/lint/pylinter.py | 104 | 235| 1089 | 2402 | 19652 | 
| 6 | 5 pylint/checkers/stdlib.py | 99 | 257| 1200 | 3602 | 26660 | 
| 7 | 5 pylint/checkers/stdlib.py | 7 | 46| 351 | 3953 | 26660 | 
| 8 | 6 pylint/constants.py | 5 | 99| 732 | 4685 | 29275 | 
| 9 | 7 pylint/extensions/private_import.py | 7 | 38| 215 | 4900 | 31475 | 
| 10 | 8 pylint/checkers/design_analysis.py | 7 | 91| 703 | 5603 | 36328 | 
| 11 | 8 pylint/checkers/stdlib.py | 48 | 96| 587 | 6190 | 36328 | 
| 12 | 8 pylint/lint/pylinter.py | 5 | 103| 643 | 6833 | 36328 | 
| **-> 13 <-** | **8 pylint/checkers/imports.py** | 327 | 442| 719 | 7552 | 36328 | 
| 14 | 9 pylint/pyreverse/main.py | 7 | 34| 165 | 7717 | 37936 | 
| 15 | 10 doc/data/messages/m/multiple-imports/bad.py | 1 | 2| 0 | 7717 | 37947 | 
| 16 | 11 pylint/extensions/bad_builtin.py | 7 | 49| 272 | 7989 | 38456 | 
| 17 | 12 pylint/message/_deleted_message_ids.py | 5 | 126| 1415 | 9404 | 40364 | 
| 18 | 13 doc/data/messages/c/consider-using-alias/good.py | 1 | 2| 0 | 9404 | 40376 | 
| 19 | 14 pylint/checkers/strings.py | 7 | 66| 381 | 9785 | 48833 | 
| 20 | 15 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 75| 499 | 10284 | 50792 | 
| 21 | 16 pylint/config/__init__.py | 5 | 39| 291 | 10575 | 51395 | 
| 22 | **16 pylint/checkers/imports.py** | 746 | 824| 697 | 11272 | 51395 | 
| 23 | 17 pylint/extensions/typing.py | 32 | 85| 765 | 12037 | 55879 | 
| 24 | 18 pylint/config/arguments_manager.py | 7 | 58| 302 | 12339 | 62366 | 
| 25 | 19 pylint/checkers/typecheck.py | 7 | 106| 579 | 12918 | 80645 | 
| 26 | 20 pylint/checkers/classes/class_checker.py | 7 | 76| 496 | 13414 | 98076 | 
| 27 | 21 pylint/checkers/format.py | 14 | 119| 746 | 14160 | 104179 | 
| 28 | 22 doc/data/messages/m/multiple-imports/good.py | 1 | 3| 0 | 14160 | 104185 | 
| 29 | 23 pylint/checkers/base/basic_error_checker.py | 7 | 25| 145 | 14305 | 108951 | 
| 30 | 24 doc/data/messages/u/unused-import/bad.py | 1 | 5| 0 | 14305 | 108972 | 
| 31 | 24 pylint/checkers/stdlib.py | 260 | 316| 288 | 14593 | 108972 | 
| 32 | 25 pylint/checkers/refactoring/recommendation_checker.py | 5 | 81| 698 | 15291 | 112546 | 
| 33 | 26 pylint/checkers/logging.py | 7 | 94| 727 | 16018 | 115778 | 
| 34 | 27 doc/data/messages/d/deprecated-typing-alias/good.py | 1 | 2| 0 | 16018 | 115790 | 
| 35 | 28 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 16018 | 115809 | 
| 36 | 28 pylint/checkers/base/basic_error_checker.py | 100 | 207| 941 | 16959 | 115809 | 
| 37 | **28 pylint/checkers/imports.py** | 826 | 856| 303 | 17262 | 115809 | 
| 38 | 29 doc/data/messages/c/consider-using-from-import/bad.py | 1 | 2| 0 | 17262 | 115822 | 
| 39 | 30 pylint/pyreverse/inspector.py | 10 | 39| 174 | 17436 | 118974 | 
| 40 | **30 pylint/checkers/imports.py** | 929 | 956| 237 | 17673 | 118974 | 
| 41 | 31 pylint/checkers/exceptions.py | 7 | 34| 175 | 17848 | 124353 | 
| 42 | 32 pylint/lint/expand_modules.py | 5 | 23| 143 | 17991 | 125643 | 
| 43 | 32 pylint/extensions/typing.py | 5 | 29| 132 | 18123 | 125643 | 
| 44 | 33 pylint/config/utils.py | 7 | 28| 118 | 18241 | 127873 | 
| 45 | **33 pylint/checkers/imports.py** | 651 | 672| 218 | 18459 | 127873 | 
| **-> 46 <-** | **33 pylint/checkers/imports.py** | 899 | 927| 236 | 18695 | 127873 | 
| 47 | 33 pylint/checkers/stdlib.py | 349 | 468| 1273 | 19968 | 127873 | 
| 48 | 33 pylint/checkers/design_analysis.py | 92 | 170| 510 | 20478 | 127873 | 
| 49 | 34 pylint/checkers/base/basic_checker.py | 107 | 267| 1493 | 21971 | 136199 | 
| 50 | 35 pylint/lint/__init__.py | 17 | 47| 183 | 22154 | 136512 | 
| 51 | 36 pylint/checkers/utils.py | 7 | 74| 494 | 22648 | 153313 | 
| 52 | 37 pylint/interfaces.py | 7 | 53| 291 | 22939 | 154228 | 
| 53 | 38 doc/data/messages/u/unused-import/good.py | 1 | 4| 0 | 22939 | 154239 | 
| 54 | 38 pylint/constants.py | 169 | 267| 1131 | 24070 | 154239 | 
| 55 | 39 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 5| 0 | 24070 | 154250 | 
| 56 | 40 doc/data/messages/c/consider-using-from-import/good.py | 1 | 2| 0 | 24070 | 154255 | 
| 57 | 41 doc/data/messages/u/ungrouped-imports/bad.py | 1 | 6| 0 | 24070 | 154285 | 
| 58 | 42 pylint/checkers/base_checker.py | 5 | 32| 168 | 24238 | 156692 | 
| 59 | 43 doc/data/messages/c/cyclic-import/good.py | 1 | 2| 0 | 24238 | 156704 | 
| 60 | 44 doc/data/messages/w/wrong-import-order/bad.py | 1 | 5| 0 | 24238 | 156730 | 
| 61 | 45 pylint/checkers/variables.py | 393 | 536| 1305 | 25543 | 182447 | 
| 62 | 46 pylint/checkers/similar.py | 30 | 94| 415 | 25958 | 190207 | 
| 63 | 47 doc/data/messages/r/redundant-keyword-arg/bad.py | 1 | 6| 0 | 25958 | 190237 | 
| 64 | 48 doc/data/messages/n/non-ascii-module-import/bad.py | 1 | 4| 0 | 25958 | 190264 | 
| 65 | **48 pylint/checkers/imports.py** | 1026 | 1060| 277 | 26235 | 190264 | 
| 66 | 49 pylint/checkers/base/docstring_checker.py | 7 | 47| 231 | 26466 | 191925 | 
| 67 | **49 pylint/checkers/imports.py** | 551 | 577| 287 | 26753 | 191925 | 
| 68 | 50 pylint/checkers/base/name_checker/checker.py | 7 | 59| 330 | 27083 | 197297 | 
| 69 | 50 pylint/extensions/typing.py | 88 | 161| 759 | 27842 | 197297 | 
| 70 | **50 pylint/checkers/imports.py** | 444 | 456| 165 | 28007 | 197297 | 
| 71 | 51 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 28007 | 197320 | 
| 72 | 52 doc/data/messages/w/wildcard-import/bad.py | 1 | 2| 0 | 28007 | 197331 | 
| 73 | 53 pylint/utils/file_state.py | 5 | 31| 125 | 28132 | 199730 | 


### Hint

```
> The reason for the as aliases here is to be explicit that these imports are for the purpose of re-export (without having to resort to defining __all__, which is error-prone).

I think ``__all__``is the way to be explicit about the API of a module.That way you have the API documented in one place at the top of the module without having to check what exactly is imported with  ``import x as x``.  I never heard about  ``import x as x`` and never did the implementer of the check, but I saw the mypy documentation you linked, let's see how widely used this is.
I don't think there is a way for pylint to detect if the reexport is intended or not. Maybe we could ignore `__init__.py` files 🤔 However, that might be unexpected to the general user.

Probably the easiest solution in your case would be to add a module level `pylint: disable=useless-import-alias` (before any imports).
Yeah, other linters like mypy specifically support `as` rather than just `__all__` for this, since so many people have been burned by using `__all__`.


`__all__` requires maintaining exports as a list of strings, which are all-too-easy to typo (and often tools and IDEs can’t detect when this happens), and also separately (and often far away) from where they’re imported / defined, which is also fragile and error-prone.
>  tools and IDEs can’t detect when this happens

Yeah I remember when I used liclipse (eclipse + pydev) this was a pain. This is an issue with the IDE though, Pycharm Community Edition handle this correctly.
Sure, some IDEs can help with typos in `__all__`, but that's only one part of the problem with `__all__`. More problematic is that it forces you to maintain exports separately from where they're imported / defined, which makes it too easy for `__all__` to drift out of sync as changes are made to the intended exports.
As @cdce8p said, the solution is to disable. I think this is going to stay that way because I don't see how pylint can guess the intent of the implementer. We could make this check optional but I think if you're making library API using this  you're more able to disable the check than a beginner making a genuine mistake is to activate it. We're going to document this in the ``useless-import-alias`` documentation.
Ok, thanks for that. As usage of mypy and mypy-style explicit re-imports continues to grow, it would be interesting to know how many pylint users end up having to disable `useless-import-alias`, and whether that amount ever crosses some threshold for being better as an opt-in rather than an opt-out. Not sure how much usage data you collect though for such decisions (e.g. by looking at usage from open source codebases).
> crosses some threshold for being better as an opt-in rather than an opt-out

An alternative solution would be to not raise this message in ``__init__.py``.

>  Not sure how much usage data you collect though for such decisions (e.g. by looking at usage from open source codebases).

To be frank, it's mostly opened issues and the thumbs-up / comments those issues gather. I'm not looking specifically at open sources projects for each messages it's really time consuming. A well researched comments on this issue with stats and sources, a proposition that is easily implementable with a better result than what we have currently, or this issue gathering 50+ thumbs up and a lot of attention would definitely make us reconsider.
Got it, good to know.
Just noticed https://github.com/microsoft/pyright/releases/tag/1.1.278

> Changed the `reportUnusedImport` check to not report an error for "from y import x as x" since x is considered to be re-exported in this case. Previously, this case was exempted only for type stubs.

One more tool (pyright) flipping in this direction, fwiw.
We'll need an option to exclude ``__init__`` for the check if this become widespread. Reopening in order to not duplicate info.
```

## Patch

```diff
diff --git a/pylint/checkers/imports.py b/pylint/checkers/imports.py
--- a/pylint/checkers/imports.py
+++ b/pylint/checkers/imports.py
@@ -439,6 +439,15 @@ class ImportsChecker(DeprecatedMixin, BaseChecker):
                 "help": "Allow wildcard imports from modules that define __all__.",
             },
         ),
+        (
+            "allow-reexport-from-package",
+            {
+                "default": False,
+                "type": "yn",
+                "metavar": "<y or n>",
+                "help": "Allow explicit reexports by alias from a package __init__.",
+            },
+        ),
     )
 
     def __init__(self, linter: PyLinter) -> None:
@@ -461,6 +470,7 @@ def open(self) -> None:
         self.linter.stats = self.linter.stats
         self.import_graph = defaultdict(set)
         self._module_pkg = {}  # mapping of modules to the pkg they belong in
+        self._current_module_package = False
         self._excluded_edges: defaultdict[str, set[str]] = defaultdict(set)
         self._ignored_modules: Sequence[str] = self.linter.config.ignored_modules
         # Build a mapping {'module': 'preferred-module'}
@@ -470,6 +480,7 @@ def open(self) -> None:
             if ":" in module
         )
         self._allow_any_import_level = set(self.linter.config.allow_any_import_level)
+        self._allow_reexport_package = self.linter.config.allow_reexport_from_package
 
     def _import_graph_without_ignored_edges(self) -> defaultdict[str, set[str]]:
         filtered_graph = copy.deepcopy(self.import_graph)
@@ -495,6 +506,10 @@ def deprecated_modules(self) -> set[str]:
                 all_deprecated_modules = all_deprecated_modules.union(mod_set)
         return all_deprecated_modules
 
+    def visit_module(self, node: nodes.Module) -> None:
+        """Store if current module is a package, i.e. an __init__ file."""
+        self._current_module_package = node.package
+
     def visit_import(self, node: nodes.Import) -> None:
         """Triggered when an import statement is seen."""
         self._check_reimport(node)
@@ -917,8 +932,11 @@ def _check_import_as_rename(self, node: ImportNode) -> None:
             if import_name != aliased_name:
                 continue
 
-            if len(splitted_packages) == 1:
-                self.add_message("useless-import-alias", node=node)
+            if len(splitted_packages) == 1 and (
+                self._allow_reexport_package is False
+                or self._current_module_package is False
+            ):
+                self.add_message("useless-import-alias", node=node, confidence=HIGH)
             elif len(splitted_packages) == 2:
                 self.add_message(
                     "consider-using-from-import",

```

## Test Patch

```diff
diff --git a/tests/checkers/unittest_imports.py b/tests/checkers/unittest_imports.py
--- a/tests/checkers/unittest_imports.py
+++ b/tests/checkers/unittest_imports.py
@@ -137,3 +137,46 @@ def test_preferred_module(capsys: CaptureFixture[str]) -> None:
         assert "Prefer importing 'sys' instead of 'os'" in output
         # assert there were no errors
         assert len(errors) == 0
+
+    @staticmethod
+    def test_allow_reexport_package(capsys: CaptureFixture[str]) -> None:
+        """Test --allow-reexport-from-package option."""
+
+        # Option disabled - useless-import-alias should always be emitted
+        Run(
+            [
+                f"{os.path.join(REGR_DATA, 'allow_reexport')}",
+                "--allow-reexport-from-package=no",
+                "-sn",
+            ],
+            exit=False,
+        )
+        output, errors = capsys.readouterr()
+        assert len(output.split("\n")) == 5
+        assert (
+            "__init__.py:1:0: C0414: Import alias does not rename original package (useless-import-alias)"
+            in output
+        )
+        assert (
+            "file.py:2:0: C0414: Import alias does not rename original package (useless-import-alias)"
+            in output
+        )
+        assert len(errors) == 0
+
+        # Option enabled - useless-import-alias should only be emitted for 'file.py'
+        Run(
+            [
+                f"{os.path.join(REGR_DATA, 'allow_reexport')}",
+                "--allow-reexport-from-package=yes",
+                "-sn",
+            ],
+            exit=False,
+        )
+        output, errors = capsys.readouterr()
+        assert len(output.split("\n")) == 3
+        assert "__init__.py" not in output
+        assert (
+            "file.py:2:0: C0414: Import alias does not rename original package (useless-import-alias)"
+            in output
+        )
+        assert len(errors) == 0
diff --git a/tests/functional/i/import_aliasing.txt b/tests/functional/i/import_aliasing.txt
--- a/tests/functional/i/import_aliasing.txt
+++ b/tests/functional/i/import_aliasing.txt
@@ -1,10 +1,10 @@
-useless-import-alias:6:0:6:50::Import alias does not rename original package:UNDEFINED
+useless-import-alias:6:0:6:50::Import alias does not rename original package:HIGH
 consider-using-from-import:8:0:8:22::Use 'from os import path' instead:UNDEFINED
 consider-using-from-import:10:0:10:31::Use 'from foo.bar import foobar' instead:UNDEFINED
-useless-import-alias:14:0:14:24::Import alias does not rename original package:UNDEFINED
-useless-import-alias:17:0:17:28::Import alias does not rename original package:UNDEFINED
-useless-import-alias:18:0:18:38::Import alias does not rename original package:UNDEFINED
-useless-import-alias:20:0:20:38::Import alias does not rename original package:UNDEFINED
-useless-import-alias:21:0:21:38::Import alias does not rename original package:UNDEFINED
-useless-import-alias:23:0:23:36::Import alias does not rename original package:UNDEFINED
+useless-import-alias:14:0:14:24::Import alias does not rename original package:HIGH
+useless-import-alias:17:0:17:28::Import alias does not rename original package:HIGH
+useless-import-alias:18:0:18:38::Import alias does not rename original package:HIGH
+useless-import-alias:20:0:20:38::Import alias does not rename original package:HIGH
+useless-import-alias:21:0:21:38::Import alias does not rename original package:HIGH
+useless-import-alias:23:0:23:36::Import alias does not rename original package:HIGH
 relative-beyond-top-level:26:0:26:27::Attempted relative import beyond top-level package:UNDEFINED
diff --git a/tests/regrtest_data/allow_reexport/__init__.py b/tests/regrtest_data/allow_reexport/__init__.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/allow_reexport/__init__.py
@@ -0,0 +1 @@
+import os as os
diff --git a/tests/regrtest_data/allow_reexport/file.py b/tests/regrtest_data/allow_reexport/file.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/allow_reexport/file.py
@@ -0,0 +1,2 @@
+# pylint: disable=unused-import
+import os as os

```


## Code snippets

### 1 - pylint/checkers/imports.py:

Start line: 7, End line: 78

```python
from __future__ import annotations

import collections
import copy
import os
import sys
from collections import defaultdict
from collections.abc import ItemsView, Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Union

import astroid
from astroid import nodes
from astroid.nodes._base_nodes import ImportNode

from pylint.checkers import BaseChecker, DeprecatedMixin
from pylint.checkers.utils import (
    get_import_name,
    is_from_fallback_block,
    is_node_in_guarded_import_block,
    is_typing_guard,
    node_ignores_exception,
)
from pylint.exceptions import EmptyReportError
from pylint.graph import DotBackend, get_cycles
from pylint.interfaces import HIGH
from pylint.reporters.ureports.nodes import Paragraph, Section, VerbatimText
from pylint.typing import MessageDefinitionTuple
from pylint.utils import IsortDriver
from pylint.utils.linterstats import LinterStats

if TYPE_CHECKING:
    from pylint.lint import PyLinter

# The dictionary with Any should actually be a _ImportTree again
# but mypy doesn't support recursive types yet
_ImportTree = Dict[str, Union[List[Dict[str, Any]], List[str]]]

DEPRECATED_MODULES = {
    (0, 0, 0): {"tkinter.tix", "fpectl"},
    (3, 2, 0): {"optparse"},
    (3, 3, 0): {"xml.etree.cElementTree"},
    (3, 4, 0): {"imp"},
    (3, 5, 0): {"formatter"},
    (3, 6, 0): {"asynchat", "asyncore", "smtpd"},
    (3, 7, 0): {"macpath"},
    (3, 9, 0): {"lib2to3", "parser", "symbol", "binhex"},
    (3, 10, 0): {"distutils", "typing.io", "typing.re"},
    (3, 11, 0): {
        "aifc",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "imghdr",
        "msilib",
        "mailcap",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "sndhdr",
        "spwd",
        "sunau",
        "sre_compile",
        "sre_constants",
        "sre_parse",
        "telnetlib",
        "uu",
        "xdrlib",
    },
}
```
### 2 - pylint/checkers/imports.py:

Start line: 234, End line: 324

```python
MSGS: dict[str, MessageDefinitionTuple] = {
    "E0401": (
        "Unable to import %s",
        "import-error",
        "Used when pylint has been unable to import a module.",
        {"old_names": [("F0401", "old-import-error")]},
    ),
    "E0402": (
        "Attempted relative import beyond top-level package",
        "relative-beyond-top-level",
        "Used when a relative import tries to access too many levels "
        "in the current package.",
    ),
    "R0401": (
        "Cyclic import (%s)",
        "cyclic-import",
        "Used when a cyclic import between two or more modules is detected.",
    ),
    "R0402": (
        "Use 'from %s import %s' instead",
        "consider-using-from-import",
        "Emitted when a submodule of a package is imported and "
        "aliased with the same name, "
        "e.g., instead of ``import concurrent.futures as futures`` use "
        "``from concurrent import futures``.",
    ),
    "W0401": (
        "Wildcard import %s",
        "wildcard-import",
        "Used when `from module import *` is detected.",
    ),
    "W0404": (
        "Reimport %r (imported line %s)",
        "reimported",
        "Used when a module is imported more than once.",
    ),
    "W0406": (
        "Module import itself",
        "import-self",
        "Used when a module is importing itself.",
    ),
    "W0407": (
        "Prefer importing %r instead of %r",
        "preferred-module",
        "Used when a module imported has a preferred replacement module.",
    ),
    "W0410": (
        "__future__ import is not the first non docstring statement",
        "misplaced-future",
        "Python 2.5 and greater require __future__ import to be the "
        "first non docstring statement in the module.",
    ),
    "C0410": (
        "Multiple imports on one line (%s)",
        "multiple-imports",
        "Used when import statement importing multiple modules is detected.",
    ),
    "C0411": (
        "%s should be placed before %s",
        "wrong-import-order",
        "Used when PEP8 import order is not respected (standard imports "
        "first, then third-party libraries, then local imports).",
    ),
    "C0412": (
        "Imports from package %s are not grouped",
        "ungrouped-imports",
        "Used when imports are not grouped by packages.",
    ),
    "C0413": (
        'Import "%s" should be placed at the top of the module',
        "wrong-import-position",
        "Used when code and imports are mixed.",
    ),
    "C0414": (
        "Import alias does not rename original package",
        "useless-import-alias",
        "Used when an import alias is same as original package, "
        "e.g., using import numpy as numpy instead of import numpy as np.",
    ),
    "C0415": (
        "Import outside toplevel (%s)",
        "import-outside-toplevel",
        "Used when an import statement is used anywhere other than the module "
        "toplevel. Move this import to the top of the file.",
    ),
    "W0416": (
        "Shadowed %r (imported line %s)",
        "shadowed-import",
        "Used when a module is aliased with a name that shadows another import.",
    ),
}
```
### 3 - doc/data/messages/u/useless-import-alias/bad.py:

Start line: 1, End line: 2

```python

```
### 4 - doc/data/messages/u/useless-import-alias/good.py:

Start line: 1, End line: 2

```python

```
### 5 - pylint/lint/pylinter.py:

Start line: 104, End line: 235

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
    "W0012": (
        "Unknown option value for '%s', expected a valid pylint message and got '%s'",
        "unknown-option-value",
        "Used when an unknown value is encountered for an option.",
        {
            "scope": WarningScope.LINE,
            "old_names": [("E0012", "bad-option-value")],
        },
    ),
    "R0022": (
        "Useless option value for '%s', %s",
        "useless-option-value",
        "Used when a value for an option that is now deleted from pylint"
        " is encountered.",
        {
            "scope": WarningScope.LINE,
            "old_names": [("E0012", "bad-option-value")],
        },
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
### 6 - pylint/checkers/stdlib.py:

Start line: 99, End line: 257

```python
DEPRECATED_METHODS: dict[int, DeprecationDict] = {
    0: {
        (0, 0, 0): {
            "cgi.parse_qs",
            "cgi.parse_qsl",
            "ctypes.c_buffer",
            "distutils.command.register.register.check_metadata",
            "distutils.command.sdist.sdist.check_metadata",
            "tkinter.Misc.tk_menuBar",
            "tkinter.Menu.tk_bindForTraversal",
        }
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
            "locale.resetlocale",
            "re.template",
            "unittest.findTestCases",
            "unittest.makeSuite",
            "unittest.getTestCaseNames",
            "unittest.TestLoader.loadTestsFromModule",
            "unittest.TestLoader.loadTestsFromTestCase",
            "unittest.TestLoader.getTestCaseNames",
        },
    },
}
```
### 7 - pylint/checkers/stdlib.py:

Start line: 7, End line: 46

```python
from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, Set, Tuple

import astroid
from astroid import nodes
from astroid.typing import InferenceResult

from pylint import interfaces
from pylint.checkers import BaseChecker, DeprecatedMixin, utils
from pylint.interfaces import INFERENCE
from pylint.typing import MessageDefinitionTuple

if TYPE_CHECKING:
    from pylint.lint import PyLinter

DeprecationDict = Dict[Tuple[int, int, int], Set[str]]

OPEN_FILES_MODE = ("open", "file")
OPEN_FILES_FUNCS = OPEN_FILES_MODE + ("read_text", "write_text")
UNITTEST_CASE = "unittest.case"
THREADING_THREAD = "threading.Thread"
COPY_COPY = "copy.copy"
OS_ENVIRON = "os._Environ"
ENV_GETTERS = ("os.getenv",)
SUBPROCESS_POPEN = "subprocess.Popen"
SUBPROCESS_RUN = "subprocess.run"
OPEN_MODULE = {"_io", "pathlib"}
DEBUG_BREAKPOINTS = ("builtins.breakpoint", "sys.breakpointhook", "pdb.set_trace")
LRU_CACHE = {
    "functools.lru_cache",  # Inferred for @lru_cache
    "functools._lru_cache_wrapper.wrapper",  # Inferred for @lru_cache() on >= Python 3.8
    "functools.lru_cache.decorating_function",  # Inferred for @lru_cache() on <= Python 3.7
}
NON_INSTANCE_METHODS = {"builtins.staticmethod", "builtins.classmethod"}


# For modules, see ImportsChecker
```
### 8 - pylint/constants.py:

Start line: 5, End line: 99

```python
from __future__ import annotations

import os
import pathlib
import platform
import sys
from datetime import datetime

import astroid
import platformdirs

from pylint.__pkginfo__ import __version__
from pylint.typing import MessageTypesFullName

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

MSG_TYPES: dict[str, MessageTypesFullName] = {
    "I": "info",
    "C": "convention",
    "R": "refactor",
    "W": "warning",
    "E": "error",
    "F": "fatal",
}
MSG_TYPES_LONG: dict[str, str] = {v: k for k, v in MSG_TYPES.items()}

MSG_TYPES_STATUS = {"I": 0, "C": 16, "R": 8, "W": 4, "E": 2, "F": 1}

# You probably don't want to change the MAIN_CHECKER_NAME
# This would affect rcfile generation and retro-compatibility
# on all project using [MAIN] in their rcfile.
MAIN_CHECKER_NAME = "main"

USER_HOME = os.path.expanduser("~")
# TODO: 3.0: Remove in 3.0 with all the surrounding code
OLD_DEFAULT_PYLINT_HOME = ".pylint.d"
DEFAULT_PYLINT_HOME = platformdirs.user_cache_dir("pylint")

DEFAULT_IGNORE_LIST = ("CVS",)


class WarningScope:
    LINE = "line-based-msg"
    NODE = "node-based-msg"


full_version = f"""pylint {__version__}
astroid {astroid.__version__}
Python {sys.version}"""

HUMAN_READABLE_TYPES = {
    "file": "file",
    "module": "module",
    "const": "constant",
    "class": "class",
    "function": "function",
    "method": "method",
    "attr": "attribute",
    "argument": "argument",
    "variable": "variable",
    "class_attribute": "class attribute",
    "class_const": "class constant",
    "inlinevar": "inline iteration",
    "typevar": "type variable",
}

# ignore some messages when emitting useless-suppression:
# - cyclic-import: can show false positives due to incomplete context
# - deprecated-{module, argument, class, method, decorator}:
#   can cause false positives for multi-interpreter projects
#   when linting with an interpreter on a lower python version
INCOMPATIBLE_WITH_USELESS_SUPPRESSION = frozenset(
    [
        "R0401",  # cyclic-import
        "W0402",  # deprecated-module
        "W1505",  # deprecated-method
        "W1511",  # deprecated-argument
        "W1512",  # deprecated-class
        "W1513",  # deprecated-decorator
        "R0801",  # duplicate-code
    ]
)
```
### 9 - pylint/extensions/private_import.py:

Start line: 7, End line: 38

```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from astroid import nodes

from pylint.checkers import BaseChecker, utils
from pylint.interfaces import HIGH

if TYPE_CHECKING:
    from pylint.lint.pylinter import PyLinter


class PrivateImportChecker(BaseChecker):
    name = "import-private-name"
    msgs = {
        "C2701": (
            "Imported private %s (%s)",
            "import-private-name",
            "Used when a private module or object prefixed with _ is imported. "
            "PEP8 guidance on Naming Conventions states that public attributes with "
            "leading underscores should be considered private.",
        ),
    }

    def __init__(self, linter: PyLinter) -> None:
        BaseChecker.__init__(self, linter)

        # A mapping of private names used as a type annotation to whether it is an acceptable import
        self.all_used_type_annotations: dict[str, bool] = {}
        self.populated_annotations = False
```
### 10 - pylint/checkers/design_analysis.py:

Start line: 7, End line: 91

```python
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import astroid
from astroid import nodes

from pylint.checkers import BaseChecker
from pylint.checkers.utils import is_enum, only_required_for_messages
from pylint.typing import MessageDefinitionTuple

if TYPE_CHECKING:
    from pylint.lint import PyLinter

MSGS: dict[
    str, MessageDefinitionTuple
] = {  # pylint: disable=consider-using-namedtuple-or-dataclass
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
### 13 - pylint/checkers/imports.py:

Start line: 327, End line: 442

```python
DEFAULT_STANDARD_LIBRARY = ()
DEFAULT_KNOWN_THIRD_PARTY = ("enchant",)
DEFAULT_PREFERRED_MODULES = ()


class ImportsChecker(DeprecatedMixin, BaseChecker):
    """BaseChecker for import statements.

    Checks for
    * external modules dependencies
    * relative / wildcard imports
    * cyclic imports
    * uses of deprecated modules
    * uses of modules instead of preferred modules
    """

    name = "imports"
    msgs = {**DeprecatedMixin.DEPRECATED_MODULE_MESSAGE, **MSGS}
    default_deprecated_modules = ()

    options = (
        (
            "deprecated-modules",
            {
                "default": default_deprecated_modules,
                "type": "csv",
                "metavar": "<modules>",
                "help": "Deprecated modules which should not be used,"
                " separated by a comma.",
            },
        ),
        (
            "preferred-modules",
            {
                "default": DEFAULT_PREFERRED_MODULES,
                "type": "csv",
                "metavar": "<module:preferred-module>",
                "help": "Couples of modules and preferred modules,"
                " separated by a comma.",
            },
        ),
        (
            "import-graph",
            {
                "default": "",
                "type": "path",
                "metavar": "<file.gv>",
                "help": "Output a graph (.gv or any supported image format) of"
                " all (i.e. internal and external) dependencies to the given file"
                " (report RP0402 must not be disabled).",
            },
        ),
        (
            "ext-import-graph",
            {
                "default": "",
                "type": "path",
                "metavar": "<file.gv>",
                "help": "Output a graph (.gv or any supported image format)"
                " of external dependencies to the given file"
                " (report RP0402 must not be disabled).",
            },
        ),
        (
            "int-import-graph",
            {
                "default": "",
                "type": "path",
                "metavar": "<file.gv>",
                "help": "Output a graph (.gv or any supported image format)"
                " of internal dependencies to the given file"
                " (report RP0402 must not be disabled).",
            },
        ),
        (
            "known-standard-library",
            {
                "default": DEFAULT_STANDARD_LIBRARY,
                "type": "csv",
                "metavar": "<modules>",
                "help": "Force import order to recognize a module as part of "
                "the standard compatibility libraries.",
            },
        ),
        (
            "known-third-party",
            {
                "default": DEFAULT_KNOWN_THIRD_PARTY,
                "type": "csv",
                "metavar": "<modules>",
                "help": "Force import order to recognize a module as part of "
                "a third party library.",
            },
        ),
        (
            "allow-any-import-level",
            {
                "default": (),
                "type": "csv",
                "metavar": "<modules>",
                "help": (
                    "List of modules that can be imported at any level, not just "
                    "the top level one."
                ),
            },
        ),
        (
            "allow-wildcard-with-all",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Allow wildcard imports from modules that define __all__.",
            },
        ),
    )
```
### 22 - pylint/checkers/imports.py:

Start line: 746, End line: 824

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def _check_imports_order(
        self, _module_node: nodes.Module
    ) -> tuple[
        list[tuple[ImportNode, str]],
        list[tuple[ImportNode, str]],
        list[tuple[ImportNode, str]],
    ]:
        # ... other code
        for node, modname in self._imports_stack:
            if modname.startswith("."):
                package = "." + modname.split(".")[1]
            else:
                package = modname.split(".")[0]
            nested = not isinstance(node.parent, nodes.Module)
            ignore_for_import_order = not self.linter.is_message_enabled(
                "wrong-import-order", node.fromlineno
            )
            import_category = isort_driver.place_module(package)
            node_and_package_import = (node, package)
            if import_category in {"FUTURE", "STDLIB"}:
                std_imports.append(node_and_package_import)
                wrong_import = (
                    third_party_not_ignored
                    or first_party_not_ignored
                    or local_not_ignored
                )
                if self._is_fallback_import(node, wrong_import):
                    continue
                if wrong_import and not nested:
                    self.add_message(
                        "wrong-import-order",
                        node=node,
                        args=(
                            f'standard import "{node.as_string()}"',
                            f'"{wrong_import[0][0].as_string()}"',
                        ),
                    )
            elif import_category == "THIRDPARTY":
                third_party_imports.append(node_and_package_import)
                external_imports.append(node_and_package_import)
                if not nested:
                    if not ignore_for_import_order:
                        third_party_not_ignored.append(node_and_package_import)
                    else:
                        self.linter.add_ignored_message(
                            "wrong-import-order", node.fromlineno, node
                        )
                wrong_import = first_party_not_ignored or local_not_ignored
                if wrong_import and not nested:
                    self.add_message(
                        "wrong-import-order",
                        node=node,
                        args=(
                            f'third party import "{node.as_string()}"',
                            f'"{wrong_import[0][0].as_string()}"',
                        ),
                    )
            elif import_category == "FIRSTPARTY":
                first_party_imports.append(node_and_package_import)
                external_imports.append(node_and_package_import)
                if not nested:
                    if not ignore_for_import_order:
                        first_party_not_ignored.append(node_and_package_import)
                    else:
                        self.linter.add_ignored_message(
                            "wrong-import-order", node.fromlineno, node
                        )
                wrong_import = local_not_ignored
                if wrong_import and not nested:
                    self.add_message(
                        "wrong-import-order",
                        node=node,
                        args=(
                            f'first party import "{node.as_string()}"',
                            f'"{wrong_import[0][0].as_string()}"',
                        ),
                    )
            elif import_category == "LOCALFOLDER":
                local_imports.append((node, package))
                if not nested:
                    if not ignore_for_import_order:
                        local_not_ignored.append((node, package))
                    else:
                        self.linter.add_ignored_message(
                            "wrong-import-order", node.fromlineno, node
                        )
        return std_imports, external_imports, local_imports
```
### 37 - pylint/checkers/imports.py:

Start line: 826, End line: 856

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def _get_imported_module(
        self, importnode: ImportNode, modname: str | None
    ) -> nodes.Module | None:
        try:
            return importnode.do_import_module(modname)
        except astroid.TooManyLevelsError:
            if _ignore_import_failure(importnode, modname, self._ignored_modules):
                return None
            self.add_message("relative-beyond-top-level", node=importnode)
        except astroid.AstroidSyntaxError as exc:
            message = f"Cannot import {modname!r} due to '{exc.error}'"
            self.add_message(
                "syntax-error", line=importnode.lineno, args=message, confidence=HIGH
            )

        except astroid.AstroidBuildingError:
            if not self.linter.is_message_enabled("import-error"):
                return None
            if _ignore_import_failure(importnode, modname, self._ignored_modules):
                return None
            if (
                not self.linter.config.analyse_fallback_blocks
                and is_from_fallback_block(importnode)
            ):
                return None

            dotted_modname = get_import_name(importnode, modname)
            self.add_message("import-error", args=repr(dotted_modname), node=importnode)
        except Exception as e:  # pragma: no cover
            raise astroid.AstroidError from e
        return None
```
### 40 - pylint/checkers/imports.py:

Start line: 929, End line: 956

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def _check_reimport(
        self,
        node: ImportNode,
        basename: str | None = None,
        level: int | None = None,
    ) -> None:
        """Check if a module with the same name is already imported or aliased."""
        if not self.linter.is_message_enabled(
            "reimported"
        ) and not self.linter.is_message_enabled("shadowed-import"):
            return

        frame = node.frame(future=True)
        root = node.root()
        contexts = [(frame, level)]
        if root is not frame:
            contexts.append((root, None))

        for known_context, known_level in contexts:
            for name, alias in node.names:
                first, msg = _get_first_import(
                    node, known_context, name, basename, known_level, alias
                )
                if first is not None and msg is not None:
                    name = name if msg == "reimported" else alias
                    self.add_message(
                        msg, node=node, args=(name, first.fromlineno), confidence=HIGH
                    )
```
### 45 - pylint/checkers/imports.py:

Start line: 651, End line: 672

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    visit_classdef = visit_for = visit_while = visit_functiondef

    def _check_misplaced_future(self, node: nodes.ImportFrom) -> None:
        basename = node.modname
        if basename == "__future__":
            # check if this is the first non-docstring statement in the module
            prev = node.previous_sibling()
            if prev:
                # consecutive future statements are possible
                if not (
                    isinstance(prev, nodes.ImportFrom) and prev.modname == "__future__"
                ):
                    self.add_message("misplaced-future", node=node)
            return

    def _check_same_line_imports(self, node: nodes.ImportFrom) -> None:
        # Detect duplicate imports on the same line.
        names = (name for name, _ in node.names)
        counter = collections.Counter(names)
        for name, count in counter.items():
            if count > 1:
                self.add_message("reimported", node=node, args=(name, node.fromlineno))
```
### 46 - pylint/checkers/imports.py:

Start line: 899, End line: 927

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def _check_preferred_module(self, node: ImportNode, mod_path: str) -> None:
        """Check if the module has a preferred replacement."""
        if mod_path in self.preferred_modules:
            self.add_message(
                "preferred-module",
                node=node,
                args=(self.preferred_modules[mod_path], mod_path),
            )

    def _check_import_as_rename(self, node: ImportNode) -> None:
        names = node.names
        for name in names:
            if not all(name):
                return

            splitted_packages = name[0].rsplit(".", maxsplit=1)
            import_name = splitted_packages[-1]
            aliased_name = name[1]
            if import_name != aliased_name:
                continue

            if len(splitted_packages) == 1:
                self.add_message("useless-import-alias", node=node)
            elif len(splitted_packages) == 2:
                self.add_message(
                    "consider-using-from-import",
                    node=node,
                    args=(splitted_packages[0], import_name),
                )
```
### 65 - pylint/checkers/imports.py:

Start line: 1026, End line: 1060

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def _wildcard_import_is_allowed(self, imported_module: nodes.Module | None) -> bool:
        return (
            self.linter.config.allow_wildcard_with_all
            and imported_module is not None
            and "__all__" in imported_module.locals
        )

    def _check_toplevel(self, node: ImportNode) -> None:
        """Check whether the import is made outside the module toplevel."""
        # If the scope of the import is a module, then obviously it is
        # not outside the module toplevel.
        if isinstance(node.scope(), nodes.Module):
            return

        module_names = [
            f"{node.modname}.{name[0]}"
            if isinstance(node, nodes.ImportFrom)
            else name[0]
            for name in node.names
        ]

        # Get the full names of all the imports that are only allowed at the module level
        scoped_imports = [
            name for name in module_names if name not in self._allow_any_import_level
        ]

        if scoped_imports:
            self.add_message(
                "import-outside-toplevel", args=", ".join(scoped_imports), node=node
            )


def register(linter: PyLinter) -> None:
    linter.register_checker(ImportsChecker(linter))
```
### 67 - pylint/checkers/imports.py:

Start line: 551, End line: 577

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def leave_module(self, node: nodes.Module) -> None:
        # Check imports are grouped by category (standard, 3rd party, local)
        std_imports, ext_imports, loc_imports = self._check_imports_order(node)

        # Check that imports are grouped by package within a given category
        met_import: set[str] = set()  # set for 'import x' style
        met_from: set[str] = set()  # set for 'from x import y' style
        current_package = None
        for import_node, import_name in std_imports + ext_imports + loc_imports:
            met = met_from if isinstance(import_node, nodes.ImportFrom) else met_import
            package, _, _ = import_name.partition(".")
            if (
                current_package
                and current_package != package
                and package in met
                and is_node_in_guarded_import_block(import_node) is False
            ):
                self.add_message("ungrouped-imports", node=import_node, args=package)
            current_package = package
            if not self.linter.is_message_enabled(
                "ungrouped-imports", import_node.fromlineno
            ):
                continue
            met.add(package)

        self._imports_stack = []
        self._first_non_import_node = None
```
### 70 - pylint/checkers/imports.py:

Start line: 444, End line: 456

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    def __init__(self, linter: PyLinter) -> None:
        BaseChecker.__init__(self, linter)
        self.import_graph: defaultdict[str, set[str]] = defaultdict(set)
        self._imports_stack: list[tuple[ImportNode, str]] = []
        self._first_non_import_node = None
        self._module_pkg: dict[
            Any, Any
        ] = {}  # mapping of modules to the pkg they belong in
        self._allow_any_import_level: set[Any] = set()
        self.reports = (
            ("RP0401", "External dependencies", self._report_external_dependencies),
            ("RP0402", "Modules dependencies graph", self._report_dependencies_graph),
        )
```
