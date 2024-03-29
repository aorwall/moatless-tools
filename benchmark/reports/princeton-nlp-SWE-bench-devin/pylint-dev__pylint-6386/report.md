# pylint-dev__pylint-6386

| **pylint-dev/pylint** | `754b487f4d892e3d4872b6fc7468a71db4e31c13` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | 9638 |
| **Any found context length** | 9638 |
| **Avg pos** | 138.0 |
| **Min pos** | 38 |
| **Max pos** | 175 |
| **Top file pos** | 4 |
| **Missing snippets** | 8 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -457,6 +457,7 @@ def __init__(
         kwargs: dict[str, Any],
         hide_help: bool,
         section: str | None,
+        metavar: str,
     ) -> None:
         super().__init__(
             flags=flags, arg_help=arg_help, hide_help=hide_help, section=section
@@ -467,3 +468,10 @@ def __init__(
 
         self.kwargs = kwargs
         """Any additional arguments passed to the action."""
+
+        self.metavar = metavar
+        """The metavar of the argument.
+
+        See:
+        https://docs.python.org/3/library/argparse.html#metavar
+        """
diff --git a/pylint/config/arguments_manager.py b/pylint/config/arguments_manager.py
--- a/pylint/config/arguments_manager.py
+++ b/pylint/config/arguments_manager.py
@@ -218,6 +218,7 @@ def _add_parser_option(
                 **argument.kwargs,
                 action=argument.action,
                 help=argument.help,
+                metavar=argument.metavar,
             )
         elif isinstance(argument, _ExtendArgument):
             section_group.add_argument(
diff --git a/pylint/config/utils.py b/pylint/config/utils.py
--- a/pylint/config/utils.py
+++ b/pylint/config/utils.py
@@ -71,6 +71,7 @@ def _convert_option_to_argument(
             kwargs=optdict.get("kwargs", {}),
             hide_help=optdict.get("hide", False),
             section=optdict.get("group", None),
+            metavar=optdict.get("metavar", None),
         )
     try:
         default = optdict["default"]
@@ -207,6 +208,7 @@ def _enable_all_extensions(run: Run, value: str | None) -> None:
     "--output": (True, _set_output),
     "--load-plugins": (True, _add_plugins),
     "--verbose": (False, _set_verbose_mode),
+    "-v": (False, _set_verbose_mode),
     "--enable-all-extensions": (False, _enable_all_extensions),
 }
 
@@ -218,7 +220,7 @@ def _preprocess_options(run: Run, args: Sequence[str]) -> list[str]:
     i = 0
     while i < len(args):
         argument = args[i]
-        if not argument.startswith("--"):
+        if not argument.startswith("-"):
             processed_args.append(argument)
             i += 1
             continue
diff --git a/pylint/lint/base_options.py b/pylint/lint/base_options.py
--- a/pylint/lint/base_options.py
+++ b/pylint/lint/base_options.py
@@ -544,6 +544,7 @@ def _make_run_options(self: Run) -> Options:
                 "help": "In verbose mode, extra non-checker-related info "
                 "will be displayed.",
                 "hide_from_config_file": True,
+                "metavar": "",
             },
         ),
         (
@@ -554,6 +555,7 @@ def _make_run_options(self: Run) -> Options:
                 "help": "Load and enable all available extensions. "
                 "Use --list-extensions to see a list all available extensions.",
                 "hide_from_config_file": True,
+                "metavar": "",
             },
         ),
         (

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/config/argument.py | 460 | 460 | - | 14 | -
| pylint/config/argument.py | 470 | 470 | - | 14 | -
| pylint/config/arguments_manager.py | 221 | 221 | 38 | 4 | 9638
| pylint/config/utils.py | 74 | 74 | 162 | 11 | 37623
| pylint/config/utils.py | 210 | 210 | 175 | 11 | 39765
| pylint/config/utils.py | 221 | 221 | 91 | 11 | 20205
| pylint/lint/base_options.py | 547 | 547 | 43 | 32 | 11009
| pylint/lint/base_options.py | 557 | 557 | 43 | 32 | 11009


## Problem Statement

```
Argument expected for short verbose option
### Bug description

The short option of the `verbose` option expects an argument.
Also, the help message for the `verbose` option suggests a value `VERBOSE` should be provided.

The long option works ok & doesn't expect an argument:
`pylint mytest.py --verbose`


### Command used

\`\`\`shell
pylint mytest.py -v
\`\`\`


### Pylint output

\`\`\`shell
usage: pylint [options]
pylint: error: argument --verbose/-v: expected one argument
\`\`\`

### Expected behavior

Similar behaviour to the long option.

### Pylint version

\`\`\`shell
pylint 2.14.0-dev0
astroid 2.11.2
Python 3.10.0b2 (v3.10.0b2:317314165a, May 31 2021, 10:02:22) [Clang 12.0.5 (clang-1205.0.22.9)]
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/config/option_parser.py | 5 | 54| 378 | 378 | 444 | 
| 2 | 2 pylint/config/option.py | 184 | 202| 179 | 557 | 2195 | 
| 3 | 3 doc/data/messages/b/bad-option-value/bad.py | 1 | 3| 0 | 557 | 2216 | 
| 4 | 3 pylint/config/option.py | 149 | 182| 320 | 877 | 2216 | 
| 5 | **4 pylint/config/arguments_manager.py** | 7 | 56| 280 | 1157 | 8239 | 
| 6 | 5 doc/data/messages/b/bad-option-value/good.py | 1 | 3| 0 | 1157 | 8254 | 
| 7 | 5 pylint/config/option.py | 204 | 219| 177 | 1334 | 8254 | 
| 8 | 6 pylint/__init__.py | 5 | 56| 306 | 1640 | 8954 | 
| 9 | 6 pylint/config/option.py | 5 | 55| 305 | 1945 | 8954 | 
| 10 | 7 doc/data/messages/t/too-few-format-args/bad.py | 1 | 2| 0 | 1945 | 8982 | 
| 11 | 8 pylint/lint/pylinter.py | 92 | 187| 828 | 2773 | 20728 | 
| 12 | 9 doc/data/messages/m/missing-format-argument-key/bad.py | 1 | 2| 0 | 2773 | 20753 | 
| 13 | 9 pylint/config/option.py | 58 | 68| 114 | 2887 | 20753 | 
| 14 | 10 pylint/pyreverse/main.py | 7 | 32| 154 | 3041 | 22181 | 
| 15 | **11 pylint/config/utils.py** | 123 | 145| 263 | 3304 | 24103 | 
| 16 | 12 pylint/config/help_formatter.py | 32 | 66| 331 | 3635 | 24684 | 
| 17 | 13 pylint/config/arguments_provider.py | 7 | 28| 152 | 3787 | 26363 | 
| 18 | **13 pylint/config/arguments_manager.py** | 619 | 645| 259 | 4046 | 26363 | 
| 19 | **14 pylint/config/argument.py** | 116 | 129| 130 | 4176 | 29343 | 
| 20 | 15 pylint/utils/utils.py | 215 | 262| 246 | 4422 | 32440 | 
| 21 | 16 doc/data/messages/t/too-few-format-args/good.py | 1 | 2| 0 | 4422 | 32461 | 
| 22 | 17 pylint/config/callback_actions.py | 307 | 323| 136 | 4558 | 35087 | 
| 23 | 18 doc/data/messages/t/too-many-arguments/bad.py | 1 | 17| 0 | 4558 | 35158 | 
| 24 | 19 pylint/epylint.py | 170 | 185| 116 | 4674 | 36639 | 
| 25 | 20 doc/data/messages/m/missing-format-argument-key/good.py | 1 | 2| 0 | 4674 | 36660 | 
| 26 | 20 pylint/pyreverse/main.py | 34 | 202| 892 | 5566 | 36660 | 
| 27 | 21 doc/data/messages/t/too-many-arguments/good.py | 1 | 19| 0 | 5566 | 36736 | 
| 28 | 22 pylint/lint/__init__.py | 17 | 44| 160 | 5726 | 37026 | 
| 29 | 23 pylint/lint/run.py | 80 | 191| 837 | 6563 | 38540 | 
| 30 | 24 doc/data/messages/t/too-many-format-args/bad.py | 1 | 2| 0 | 6563 | 38573 | 
| 31 | 24 pylint/epylint.py | 120 | 167| 452 | 7015 | 38573 | 
| 32 | 25 doc/data/messages/s/super-with-arguments/bad.py | 1 | 8| 0 | 7015 | 38608 | 
| 33 | 25 pylint/config/arguments_provider.py | 180 | 195| 137 | 7152 | 38608 | 
| 34 | **25 pylint/config/arguments_manager.py** | 377 | 413| 386 | 7538 | 38608 | 
| 35 | 26 pylint/constants.py | 91 | 196| 1355 | 8893 | 40749 | 
| 36 | 27 doc/data/messages/t/too-many-format-args/good.py | 1 | 2| 0 | 8893 | 40770 | 
| 37 | **27 pylint/config/arguments_manager.py** | 357 | 375| 181 | 9074 | 40770 | 
| **-> 38 <-** | **27 pylint/config/arguments_manager.py** | 158 | 234| 564 | 9638 | 40770 | 
| 39 | 28 doc/data/messages/a/arguments-out-of-order/bad.py | 1 | 14| 0 | 9638 | 40854 | 
| 40 | 29 doc/data/messages/d/duplicate-argument-name/bad.py | 1 | 3| 0 | 9638 | 40876 | 
| 41 | 30 pylint/testutils/decorator.py | 5 | 44| 241 | 9879 | 41183 | 
| 42 | 31 doc/data/messages/u/use-dict-literal/bad.py | 1 | 2| 0 | 9879 | 41197 | 
| **-> 43 <-** | **32 pylint/lint/base_options.py** | 395 | 570| 1130 | 11009 | 45119 | 
| 44 | 33 pylint/testutils/constants.py | 5 | 30| 280 | 11289 | 45464 | 
| 45 | 34 pylint/__pkginfo__.py | 5 | 39| 214 | 11503 | 45743 | 
| 46 | 34 pylint/config/option.py | 71 | 146| 601 | 12104 | 45743 | 
| 47 | 34 pylint/utils/utils.py | 336 | 354| 214 | 12318 | 45743 | 
| 48 | 35 pylint/testutils/__init__.py | 7 | 36| 228 | 12546 | 46045 | 
| 49 | 36 doc/data/messages/b/bad-staticmethod-argument/bad.py | 1 | 5| 0 | 12546 | 46069 | 
| 50 | 37 doc/data/messages/s/super-with-arguments/good.py | 1 | 8| 0 | 12546 | 46092 | 
| 51 | 38 doc/exts/pylint_options.py | 66 | 131| 521 | 13067 | 47457 | 
| 52 | 38 pylint/pyreverse/main.py | 228 | 248| 141 | 13208 | 47457 | 
| 53 | 39 doc/data/messages/a/arguments-out-of-order/good.py | 1 | 12| 0 | 13208 | 47529 | 
| 54 | 39 pylint/lint/pylinter.py | 5 | 90| 528 | 13736 | 47529 | 
| 55 | **39 pylint/config/arguments_manager.py** | 647 | 655| 118 | 13854 | 47529 | 
| 56 | 40 doc/data/messages/a/arguments-renamed/bad.py | 1 | 14| 0 | 13854 | 47631 | 
| 57 | 41 doc/data/messages/u/use-maxsplit-arg/bad.py | 1 | 3| 0 | 13854 | 47655 | 
| 58 | 41 pylint/lint/pylinter.py | 315 | 337| 202 | 14056 | 47655 | 
| 59 | 42 doc/data/messages/d/duplicate-argument-name/good.py | 1 | 3| 0 | 14056 | 47669 | 
| 60 | 43 doc/data/messages/u/use-dict-literal/good.py | 1 | 2| 0 | 14056 | 47673 | 
| 61 | 44 doc/data/messages/u/use-list-literal/bad.py | 1 | 2| 0 | 14056 | 47686 | 
| 62 | 45 doc/data/messages/u/undefined-variable/bad.py | 1 | 2| 0 | 14056 | 47698 | 
| 63 | 46 doc/data/messages/l/literal-comparison/bad.py | 1 | 3| 0 | 14056 | 47719 | 
| 64 | 47 doc/data/messages/b/bad-staticmethod-argument/good.py | 1 | 5| 0 | 14056 | 47736 | 
| 65 | 48 script/bump_changelog.py | 28 | 44| 162 | 14218 | 49289 | 
| 66 | **48 pylint/lint/base_options.py** | 7 | 392| 188 | 14406 | 49289 | 
| 67 | 49 pylint/checkers/stdlib.py | 44 | 90| 556 | 14962 | 55504 | 
| 68 | 50 pylint/pyreverse/utils.py | 7 | 108| 541 | 15503 | 57537 | 
| 69 | 51 pylint/extensions/docparams.py | 135 | 193| 382 | 15885 | 62506 | 
| 70 | 51 pylint/config/arguments_provider.py | 115 | 133| 220 | 16105 | 62506 | 
| 71 | **51 pylint/config/arguments_manager.py** | 320 | 355| 312 | 16417 | 62506 | 
| 72 | 52 doc/data/messages/u/undefined-all-variable/bad.py | 1 | 5| 0 | 16417 | 62532 | 
| 73 | 53 doc/data/messages/u/undefined-variable/good.py | 1 | 3| 0 | 16417 | 62543 | 
| 74 | 53 pylint/extensions/docparams.py | 408 | 448| 282 | 16699 | 62543 | 
| 75 | 54 doc/data/messages/a/assert-on-tuple/bad.py | 1 | 2| 0 | 16699 | 62557 | 
| 76 | 55 doc/data/messages/t/typevar-name-mismatch/bad.py | 1 | 4| 0 | 16699 | 62579 | 
| 77 | 56 doc/data/messages/c/consider-using-with/bad.py | 1 | 4| 0 | 16699 | 62609 | 
| 78 | 57 pylint/config/option_manager_mixin.py | 52 | 91| 312 | 17011 | 65384 | 
| 79 | 57 pylint/lint/run.py | 5 | 27| 148 | 17159 | 65384 | 
| 80 | 58 pylint/testutils/functional/__init__.py | 5 | 24| 129 | 17288 | 65578 | 
| 81 | 59 doc/conf.py | 143 | 244| 572 | 17860 | 67475 | 
| 82 | 59 pylint/lint/pylinter.py | 500 | 521| 216 | 18076 | 67475 | 
| 83 | 60 pylint/checkers/base/comparison_checker.py | 7 | 21| 146 | 18222 | 70008 | 
| 84 | 61 doc/data/messages/b/bad-classmethod-argument/bad.py | 1 | 6| 0 | 18222 | 70035 | 
| 85 | 62 pylint/checkers/similar.py | 861 | 873| 105 | 18327 | 77598 | 
| 86 | **62 pylint/config/arguments_manager.py** | 264 | 278| 157 | 18484 | 77598 | 
| 87 | 63 doc/data/messages/a/arguments-renamed/good.py | 1 | 14| 0 | 18484 | 77694 | 
| 88 | 64 pylint/config/exceptions.py | 5 | 24| 116 | 18600 | 77875 | 
| 89 | 65 pylint/config/config_initialization.py | 5 | 107| 818 | 19418 | 78759 | 
| 90 | **65 pylint/config/arguments_manager.py** | 59 | 123| 545 | 19963 | 78759 | 
| **-> 91 <-** | **65 pylint/config/utils.py** | 214 | 250| 242 | 20205 | 78759 | 
| 92 | 66 pylint/utils/__init__.py | 9 | 56| 268 | 20473 | 79112 | 
| 93 | 66 pylint/config/option_manager_mixin.py | 5 | 49| 287 | 20760 | 79112 | 
| 94 | 67 doc/data/messages/a/assert-on-string-literal/bad.py | 1 | 3| 0 | 20760 | 79132 | 
| 95 | 68 doc/data/messages/u/unnecessary-dunder-call/bad.py | 1 | 7| 0 | 20760 | 79195 | 
| 96 | 69 doc/data/messages/b/broad-except/bad.py | 1 | 5| 0 | 20760 | 79218 | 
| 97 | 70 doc/data/messages/c/consider-using-with/good.py | 1 | 3| 0 | 20760 | 79241 | 
| 98 | 70 pylint/checkers/stdlib.py | 315 | 433| 1219 | 21979 | 79241 | 
| 99 | 71 doc/data/messages/u/undefined-all-variable/good.py | 1 | 5| 0 | 21979 | 79260 | 
| 100 | 72 pylint/checkers/logging.py | 7 | 96| 709 | 22688 | 82253 | 
| 101 | 73 doc/data/messages/b/broad-except/good.py | 1 | 5| 0 | 22688 | 82270 | 
| 102 | 73 pylint/pyreverse/utils.py | 283 | 309| 231 | 22919 | 82270 | 
| 103 | 74 doc/data/messages/u/use-list-literal/good.py | 1 | 2| 0 | 22919 | 82274 | 
| 104 | 75 doc/data/messages/c/confusing-consecutive-elif/good.py | 1 | 23| 131 | 23050 | 82405 | 
| 105 | **75 pylint/config/arguments_manager.py** | 484 | 523| 373 | 23423 | 82405 | 
| 106 | 76 doc/data/messages/u/use-maxsplit-arg/good.py | 1 | 3| 0 | 23423 | 82425 | 
| 107 | 77 pylint/testutils/output_line.py | 5 | 33| 186 | 23609 | 83861 | 
| 108 | 78 pylint/extensions/broad_try_clause.py | 7 | 75| 446 | 24055 | 84388 | 
| 109 | 79 pylint/config/options_provider_mixin.py | 5 | 53| 353 | 24408 | 85363 | 
| 110 | **79 pylint/config/utils.py** | 7 | 28| 118 | 24526 | 85363 | 
| 111 | 80 pylint/checkers/unsupported_version.py | 9 | 64| 434 | 24960 | 86049 | 
| 112 | **80 pylint/config/arguments_manager.py** | 559 | 573| 127 | 25087 | 86049 | 
| 113 | **80 pylint/config/arguments_manager.py** | 728 | 754| 229 | 25316 | 86049 | 
| 114 | 80 pylint/checkers/stdlib.py | 93 | 237| 1089 | 26405 | 86049 | 
| 115 | 81 doc/data/messages/t/typevar-name-mismatch/good.py | 1 | 4| 0 | 26405 | 86062 | 
| 116 | 82 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 77| 510 | 26915 | 87857 | 
| 117 | 83 pylint/checkers/base/basic_checker.py | 393 | 417| 160 | 27075 | 95036 | 
| 118 | 84 doc/data/messages/s/super-without-brackets/bad.py | 1 | 12| 0 | 27075 | 95092 | 
| 119 | 85 doc/data/messages/m/missing-function-docstring/bad.py | 1 | 6| 0 | 27075 | 95113 | 
| 120 | 86 doc/data/messages/t/typevar-double-variance/bad.py | 1 | 4| 0 | 27075 | 95144 | 
| 121 | **86 pylint/config/argument.py** | 10 | 113| 694 | 27769 | 95144 | 
| 122 | 87 doc/data/messages/u/unidiomatic-typecheck/bad.py | 1 | 4| 0 | 27769 | 95180 | 
| 123 | 88 doc/data/messages/m/missing-module-docstring/bad.py | 1 | 6| 0 | 27769 | 95200 | 
| 124 | 89 doc/data/messages/u/unnecessary-dunder-call/good.py | 1 | 7| 0 | 27769 | 95233 | 
| 125 | 89 pylint/extensions/docparams.py | 369 | 406| 256 | 28025 | 95233 | 
| 126 | 90 doc/data/messages/b/bad-classmethod-argument/good.py | 1 | 6| 0 | 28025 | 95251 | 
| 127 | 90 pylint/extensions/docparams.py | 49 | 133| 701 | 28726 | 95251 | 
| 128 | 90 pylint/checkers/stdlib.py | 7 | 42| 307 | 29033 | 95251 | 
| 129 | 90 pylint/epylint.py | 59 | 117| 502 | 29535 | 95251 | 
| 130 | 91 doc/data/messages/u/useless-return/bad.py | 1 | 7| 0 | 29535 | 95275 | 
| 131 | 92 pylint/testutils/pyreverse.py | 5 | 57| 418 | 29953 | 95759 | 
| 132 | 93 doc/data/messages/u/unnecessary-ellipsis/bad.py | 1 | 4| 0 | 29953 | 95779 | 
| 133 | 93 pylint/checkers/base/basic_checker.py | 106 | 255| 1373 | 31326 | 95779 | 
| 134 | 94 pylint/extensions/bad_builtin.py | 26 | 52| 177 | 31503 | 96306 | 
| 135 | 95 pylint/checkers/format.py | 652 | 672| 192 | 31695 | 102532 | 
| 136 | 96 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 31695 | 102551 | 
| 137 | 97 pylint/testutils/functional_test_file.py | 5 | 24| 93 | 31788 | 102709 | 
| 138 | 97 pylint/checkers/unsupported_version.py | 66 | 87| 171 | 31959 | 102709 | 
| 139 | 98 doc/exts/pylint_features.py | 1 | 42| 241 | 32200 | 103029 | 
| 140 | 98 pylint/checkers/format.py | 14 | 85| 360 | 32560 | 103029 | 
| 141 | 99 doc/data/messages/l/literal-comparison/good.py | 1 | 3| 0 | 32560 | 103044 | 
| 142 | 99 pylint/utils/utils.py | 265 | 282| 171 | 32731 | 103044 | 
| 143 | 100 doc/data/messages/a/assert-on-string-literal/good.py | 1 | 3| 0 | 32731 | 103062 | 
| 144 | 101 doc/data/messages/r/redundant-unittest-assert/bad.py | 1 | 7| 0 | 32731 | 103095 | 
| 145 | 102 pylint/checkers/base_checker.py | 5 | 23| 129 | 32860 | 104877 | 
| 146 | 103 doc/data/messages/b/bad-builtin/bad.py | 1 | 3| 0 | 32860 | 104908 | 
| 147 | 104 pylint/checkers/utils.py | 67 | 183| 710 | 33570 | 117775 | 
| 148 | 105 doc/data/messages/u/unbalanced-tuple-unpacking/bad.py | 1 | 3| 0 | 33570 | 117809 | 
| 149 | 106 pylint/checkers/typecheck.py | 725 | 879| 1088 | 34658 | 134399 | 
| 150 | 107 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 34658 | 134412 | 
| 151 | 108 doc/data/messages/m/missing-class-docstring/bad.py | 1 | 6| 0 | 34658 | 134452 | 
| 152 | 108 pylint/config/arguments_provider.py | 100 | 113| 145 | 34803 | 134452 | 
| 153 | 109 doc/data/messages/u/useless-return/good.py | 1 | 6| 0 | 34803 | 134465 | 
| 154 | 110 doc/data/messages/s/super-without-brackets/good.py | 1 | 12| 0 | 34803 | 134513 | 
| 155 | 111 doc/data/messages/s/self-assigning-variable/bad.py | 1 | 3| 0 | 34803 | 134531 | 
| 156 | 112 doc/data/messages/c/confusing-consecutive-elif/bad.py | 1 | 7| 0 | 34803 | 134589 | 
| 157 | 113 pylint/checkers/base/basic_error_checker.py | 99 | 207| 952 | 35755 | 139073 | 
| 158 | 113 pylint/constants.py | 5 | 89| 609 | 36364 | 139073 | 
| 159 | 114 doc/data/messages/a/assert-on-tuple/good.py | 1 | 4| 0 | 36364 | 139088 | 
| 160 | 115 doc/data/messages/c/comparison-with-callable/bad.py | 1 | 7| 0 | 36364 | 139161 | 
| 161 | 115 pylint/extensions/docparams.py | 555 | 620| 496 | 36860 | 139161 | 
| **-> 162 <-** | **115 pylint/config/utils.py** | 31 | 122| 763 | 37623 | 139161 | 
| 163 | 116 doc/data/messages/b/bad-super-call/bad.py | 1 | 8| 0 | 37623 | 139196 | 
| 164 | 117 pylint/config/__init__.py | 5 | 109| 876 | 38499 | 140137 | 
| 165 | 118 doc/data/messages/b/bad-indentation/bad.py | 1 | 3| 0 | 38499 | 140152 | 
| 166 | 119 doc/data/messages/u/useless-import-alias/good.py | 1 | 2| 0 | 38499 | 140157 | 
| 167 | 119 pylint/config/arguments_provider.py | 135 | 151| 152 | 38651 | 140157 | 
| 168 | 119 pylint/checkers/typecheck.py | 206 | 407| 180 | 38831 | 140157 | 
| 169 | 120 doc/data/messages/u/unreachable/bad.py | 1 | 4| 0 | 38831 | 140180 | 
| 170 | 121 doc/data/messages/u/unspecified-encoding/bad.py | 1 | 4| 0 | 38831 | 140208 | 
| 171 | **121 pylint/config/arguments_manager.py** | 236 | 262| 226 | 39057 | 140208 | 
| 172 | 121 pylint/testutils/output_line.py | 101 | 119| 173 | 39230 | 140208 | 
| 173 | 122 pylint/config/environment_variable.py | 1 | 12| 0 | 39230 | 140314 | 
| 174 | 123 doc/data/messages/u/unidiomatic-typecheck/good.py | 1 | 4| 0 | 39230 | 140340 | 
| **-> 175 <-** | **123 pylint/config/utils.py** | 148 | 211| 535 | 39765 | 140340 | 
| 176 | 124 doc/data/messages/a/anomalous-backslash-in-string/bad.py | 1 | 2| 0 | 39765 | 140356 | 
| 177 | 124 pylint/checkers/stdlib.py | 724 | 755| 243 | 40008 | 140356 | 
| 178 | 125 doc/data/messages/b/bad-super-call/good.py | 1 | 8| 0 | 40008 | 140379 | 
| 179 | 126 doc/data/messages/a/abstract-method/bad.py | 1 | 21| 0 | 40008 | 140443 | 
| 180 | 127 doc/data/messages/u/unnecessary-ellipsis/good.py | 1 | 3| 0 | 40008 | 140453 | 
| 181 | 127 pylint/extensions/bad_builtin.py | 7 | 23| 114 | 40122 | 140453 | 
| 182 | 127 pylint/checkers/utils.py | 7 | 66| 402 | 40524 | 140453 | 
| 183 | 128 doc/data/messages/a/anomalous-backslash-in-string/good.py | 1 | 3| 0 | 40524 | 140464 | 
| 184 | 129 pylint/checkers/variables.py | 372 | 511| 1232 | 41756 | 162930 | 
| 185 | 130 doc/data/messages/b/bad-builtin/good.py | 1 | 3| 0 | 41756 | 162951 | 
| 186 | 131 doc/data/messages/m/missing-module-docstring/good.py | 1 | 7| 0 | 41756 | 162972 | 
| 187 | 132 pylint/__main__.py | 1 | 11| 0 | 41756 | 163059 | 
| 188 | 133 script/fix_documentation.py | 66 | 108| 304 | 42060 | 163826 | 
| 189 | **133 pylint/config/arguments_manager.py** | 142 | 156| 133 | 42193 | 163826 | 
| 190 | 134 doc/data/messages/m/missing-function-docstring/good.py | 1 | 7| 0 | 42193 | 163846 | 
| 191 | 135 doc/data/messages/t/typevar-double-variance/good.py | 1 | 5| 0 | 42193 | 163881 | 
| 192 | 136 doc/data/messages/b/bad-str-strip-call/bad.py | 1 | 5| 0 | 42193 | 163927 | 
| 193 | 137 pylint/testutils/global_test_linter.py | 5 | 21| 111 | 42304 | 164103 | 
| 194 | 138 doc/data/messages/u/unreachable/good.py | 1 | 4| 0 | 42304 | 164120 | 
| 195 | 138 pylint/checkers/stdlib.py | 435 | 481| 510 | 42814 | 164120 | 
| 196 | 139 pylint/checkers/strings.py | 60 | 187| 1211 | 44025 | 172286 | 
| 197 | 140 doc/data/messages/b/bare-except/bad.py | 1 | 5| 0 | 44025 | 172307 | 
| 198 | 140 pylint/config/callback_actions.py | 122 | 162| 226 | 44251 | 172307 | 
| 199 | 141 pylint/extensions/typing.py | 78 | 163| 774 | 45025 | 176218 | 
| 200 | 141 pylint/utils/utils.py | 139 | 161| 219 | 45244 | 176218 | 
| 201 | 141 pylint/checkers/logging.py | 119 | 148| 193 | 45437 | 176218 | 
| 202 | 141 pylint/lint/pylinter.py | 224 | 313| 811 | 46248 | 176218 | 
| 203 | 142 doc/data/messages/d/duplicate-value/bad.py | 1 | 2| 0 | 46248 | 176240 | 
| 204 | **142 pylint/config/arguments_manager.py** | 280 | 318| 350 | 46598 | 176240 | 
| 205 | 143 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 46598 | 176263 | 
| 206 | 144 doc/data/messages/y/yield-inside-async-function/bad.py | 1 | 3| 0 | 46598 | 176289 | 


## Patch

```diff
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -457,6 +457,7 @@ def __init__(
         kwargs: dict[str, Any],
         hide_help: bool,
         section: str | None,
+        metavar: str,
     ) -> None:
         super().__init__(
             flags=flags, arg_help=arg_help, hide_help=hide_help, section=section
@@ -467,3 +468,10 @@ def __init__(
 
         self.kwargs = kwargs
         """Any additional arguments passed to the action."""
+
+        self.metavar = metavar
+        """The metavar of the argument.
+
+        See:
+        https://docs.python.org/3/library/argparse.html#metavar
+        """
diff --git a/pylint/config/arguments_manager.py b/pylint/config/arguments_manager.py
--- a/pylint/config/arguments_manager.py
+++ b/pylint/config/arguments_manager.py
@@ -218,6 +218,7 @@ def _add_parser_option(
                 **argument.kwargs,
                 action=argument.action,
                 help=argument.help,
+                metavar=argument.metavar,
             )
         elif isinstance(argument, _ExtendArgument):
             section_group.add_argument(
diff --git a/pylint/config/utils.py b/pylint/config/utils.py
--- a/pylint/config/utils.py
+++ b/pylint/config/utils.py
@@ -71,6 +71,7 @@ def _convert_option_to_argument(
             kwargs=optdict.get("kwargs", {}),
             hide_help=optdict.get("hide", False),
             section=optdict.get("group", None),
+            metavar=optdict.get("metavar", None),
         )
     try:
         default = optdict["default"]
@@ -207,6 +208,7 @@ def _enable_all_extensions(run: Run, value: str | None) -> None:
     "--output": (True, _set_output),
     "--load-plugins": (True, _add_plugins),
     "--verbose": (False, _set_verbose_mode),
+    "-v": (False, _set_verbose_mode),
     "--enable-all-extensions": (False, _enable_all_extensions),
 }
 
@@ -218,7 +220,7 @@ def _preprocess_options(run: Run, args: Sequence[str]) -> list[str]:
     i = 0
     while i < len(args):
         argument = args[i]
-        if not argument.startswith("--"):
+        if not argument.startswith("-"):
             processed_args.append(argument)
             i += 1
             continue
diff --git a/pylint/lint/base_options.py b/pylint/lint/base_options.py
--- a/pylint/lint/base_options.py
+++ b/pylint/lint/base_options.py
@@ -544,6 +544,7 @@ def _make_run_options(self: Run) -> Options:
                 "help": "In verbose mode, extra non-checker-related info "
                 "will be displayed.",
                 "hide_from_config_file": True,
+                "metavar": "",
             },
         ),
         (
@@ -554,6 +555,7 @@ def _make_run_options(self: Run) -> Options:
                 "help": "Load and enable all available extensions. "
                 "Use --list-extensions to see a list all available extensions.",
                 "hide_from_config_file": True,
+                "metavar": "",
             },
         ),
         (

```

## Test Patch

```diff
diff --git a/tests/config/test_config.py b/tests/config/test_config.py
--- a/tests/config/test_config.py
+++ b/tests/config/test_config.py
@@ -100,3 +100,10 @@ def test_unknown_py_version(capsys: CaptureFixture) -> None:
         Run([str(EMPTY_MODULE), "--py-version=the-newest"], exit=False)
     output = capsys.readouterr()
     assert "the-newest has an invalid format, should be a version string." in output.err
+
+
+def test_short_verbose(capsys: CaptureFixture) -> None:
+    """Check that we correctly handle the -v flag."""
+    Run([str(EMPTY_MODULE), "-v"], exit=False)
+    output = capsys.readouterr()
+    assert "Using config file" in output.err

```


## Code snippets

### 1 - pylint/config/option_parser.py:

Start line: 5, End line: 54

```python
import optparse  # pylint: disable=deprecated-module
import warnings

from pylint.config.option import Option


def _level_options(group, outputlevel):
    return [
        option
        for option in group.option_list
        if (getattr(option, "level", 0) or 0) <= outputlevel
        and option.help is not optparse.SUPPRESS_HELP
    ]


class OptionParser(optparse.OptionParser):
    def __init__(self, option_class, *args, **kwargs):
        # TODO: 3.0: Remove deprecated class # pylint: disable=fixme
        warnings.warn(
            "OptionParser has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        super().__init__(option_class=Option, *args, **kwargs)

    def format_option_help(self, formatter=None):
        if formatter is None:
            formatter = self.formatter
        outputlevel = getattr(formatter, "output_level", 0)
        formatter.store_option_strings(self)
        result = [formatter.format_heading("Options")]
        formatter.indent()
        if self.option_list:
            result.append(optparse.OptionContainer.format_option_help(self, formatter))
            result.append("\n")
        for group in self.option_groups:
            if group.level <= outputlevel and (
                group.description or _level_options(group, outputlevel)
            ):
                result.append(group.format_help(formatter))
                result.append("\n")
        formatter.dedent()
        # Drop the last "\n", or the header if no options or option groups:
        return "".join(result[:-1])

    def _match_long_opt(self, opt):  # pragma: no cover # Unused
        """Disable abbreviations."""
        if opt not in self._long_opt:
            raise optparse.BadOptionError(opt)
        return opt
```
### 2 - pylint/config/option.py:

Start line: 184, End line: 202

```python
class Option(optparse.Option):

    def _check_choice(self):
        if self.type in {"choice", "multiple_choice", "confidence"}:
            if self.choices is None:
                raise optparse.OptionError(
                    "must supply a list of choices for type 'choice'", self
                )
            if not isinstance(self.choices, (tuple, list)):
                raise optparse.OptionError(
                    # pylint: disable-next=consider-using-f-string
                    "choices must be a list of strings ('%s' supplied)"
                    % str(type(self.choices)).split("'")[1],
                    self,
                )
        elif self.choices is not None:
            raise optparse.OptionError(
                f"must not supply choices for type {self.type!r}", self
            )

    optparse.Option.CHECK_METHODS[2] = _check_choice  # type: ignore[index]
```
### 3 - doc/data/messages/b/bad-option-value/bad.py:

Start line: 1, End line: 3

```python

```
### 4 - pylint/config/option.py:

Start line: 149, End line: 182

```python
# pylint: disable=no-member
class Option(optparse.Option):
    TYPES = optparse.Option.TYPES + (
        "regexp",
        "regexp_csv",
        "regexp_paths_csv",
        "csv",
        "yn",
        "confidence",
        "multiple_choice",
        "non_empty_string",
        "py_version",
    )
    ATTRS = optparse.Option.ATTRS + ["hide", "level"]
    TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
    TYPE_CHECKER["regexp"] = _regexp_validator
    TYPE_CHECKER["regexp_csv"] = _regexp_csv_validator
    TYPE_CHECKER["regexp_paths_csv"] = _regexp_paths_csv_validator
    TYPE_CHECKER["csv"] = _csv_validator
    TYPE_CHECKER["yn"] = _yn_validator
    TYPE_CHECKER["confidence"] = _multiple_choices_validating_option
    TYPE_CHECKER["multiple_choice"] = _multiple_choices_validating_option
    TYPE_CHECKER["non_empty_string"] = _non_empty_string_validator
    TYPE_CHECKER["py_version"] = _py_version_validator

    def __init__(self, *opts, **attrs):
        # TODO: 3.0: Remove deprecated class # pylint: disable=fixme
        warnings.warn(
            "Option has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        super().__init__(*opts, **attrs)
        if hasattr(self, "hide") and self.hide:
            self.help = optparse.SUPPRESS_HELP
```
### 5 - pylint/config/arguments_manager.py:

Start line: 7, End line: 56

```python
from __future__ import annotations

import argparse
import configparser
import copy
import optparse  # pylint: disable=deprecated-module
import os
import re
import sys
import textwrap
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, Union

import tomlkit

from pylint import utils
from pylint.config.argument import (
    _Argument,
    _CallableArgument,
    _ExtendArgument,
    _StoreArgument,
    _StoreNewNamesArgument,
    _StoreOldNamesArgument,
    _StoreTrueArgument,
)
from pylint.config.exceptions import (
    UnrecognizedArgumentAction,
    _UnrecognizedOptionError,
)
from pylint.config.help_formatter import _HelpFormatter
from pylint.config.option import Option
from pylint.config.option_parser import OptionParser
from pylint.config.options_provider_mixin import OptionsProviderMixIn
from pylint.config.utils import _convert_option_to_argument, _parse_rich_type_value
from pylint.constants import MAIN_CHECKER_NAME
from pylint.typing import OptionDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


if TYPE_CHECKING:
    from pylint.config.arguments_provider import _ArgumentsProvider

ConfigProvider = Union["_ArgumentsProvider", OptionsProviderMixIn]
```
### 6 - doc/data/messages/b/bad-option-value/good.py:

Start line: 1, End line: 3

```python

```
### 7 - pylint/config/option.py:

Start line: 204, End line: 219

```python
class Option(optparse.Option):

    def process(self, opt, value, values, parser):  # pragma: no cover # Argparse
        if self.callback and self.callback.__module__ == "pylint.lint.run":
            return 1
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
### 8 - pylint/__init__.py:

Start line: 5, End line: 56

```python
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from pylint.__pkginfo__ import __version__

# pylint: disable=import-outside-toplevel


def run_pylint(argv: Sequence[str] | None = None):
    """Run pylint.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.lint import Run as PylintRun

    try:
        PylintRun(argv or sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)


def run_epylint(argv: Sequence[str] | None = None):
    """Run epylint.

    argv can be a list of strings normally supplied as arguments on the command line
    """
    from pylint.epylint import Run as EpylintRun

    EpylintRun(argv)


def run_pyreverse(argv: Sequence[str] | None = None):
    """Run pyreverse.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.pyreverse.main import Run as PyreverseRun

    PyreverseRun(argv or sys.argv[1:])


def run_symilar(argv: Sequence[str] | None = None):
    """Run symilar.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.checkers.similar import Run as SimilarRun

    SimilarRun(argv or sys.argv[1:])
```
### 9 - pylint/config/option.py:

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
### 10 - doc/data/messages/t/too-few-format-args/bad.py:

Start line: 1, End line: 2

```python

```
### 15 - pylint/config/utils.py:

Start line: 123, End line: 145

```python
def _convert_option_to_argument(
    opt: str, optdict: dict[str, Any]
) -> (
    _StoreArgument
    | _StoreTrueArgument
    | _CallableArgument
    | _StoreOldNamesArgument
    | _StoreNewNamesArgument
    | _ExtendArgument
):
    # ... other code
    if "dest" in optdict:
        return _StoreOldNamesArgument(
            flags=flags,
            default=default,
            arg_type=optdict["type"],
            choices=optdict.get("choices", None),
            arg_help=optdict.get("help", ""),
            metavar=optdict.get("metavar", ""),
            hide_help=optdict.get("hide", False),
            kwargs={"old_names": [optdict["dest"]]},
            section=optdict.get("group", None),
        )
    return _StoreArgument(
        flags=flags,
        action=action,
        default=default,
        arg_type=optdict["type"],
        choices=optdict.get("choices", None),
        arg_help=optdict.get("help", ""),
        metavar=optdict.get("metavar", ""),
        hide_help=optdict.get("hide", False),
        section=optdict.get("group", None),
    )
```
### 18 - pylint/config/arguments_manager.py:

Start line: 619, End line: 645

```python
class _ArgumentsManager:

    def help(self, level: int | None = None) -> str:
        """Return the usage string based on the available options."""
        if level is not None:
            warnings.warn(
                "Supplying a 'level' argument to help() has been deprecated."
                "You can call help() without any arguments.",
                DeprecationWarning,
            )
        return self._arg_parser.format_help()

    def cb_set_provider_option(self, option, opt, value, parser):  # pragma: no cover
        """DEPRECATED: Optik callback for option setting."""
        # TODO: 3.0: Remove deprecated method. # pylint: disable=fixme
        warnings.warn(
            "cb_set_provider_option has been deprecated. It will be removed in pylint 3.0.",
            DeprecationWarning,
        )
        if opt.startswith("--"):
            # remove -- on long option
            opt = opt[2:]
        else:
            # short option, get its long equivalent
            opt = self._short_options[opt[1:]]
        # trick since we can't set action='store_true' on options
        if value is None:
            value = 1
        self.set_option(opt, value)
```
### 19 - pylint/config/argument.py:

Start line: 116, End line: 129

```python
_TYPE_TRANSFORMERS: dict[str, Callable[[str], _ArgumentTypes]] = {
    "choice": str,
    "csv": _csv_transformer,
    "float": float,
    "int": int,
    "confidence": _confidence_transformer,
    "non_empty_string": _non_empty_string_transformer,
    "py_version": _py_version_transformer,
    "regexp": re.compile,
    "regexp_csv": _regexp_csv_transfomer,
    "regexp_paths_csv": _regexp_paths_csv_transfomer,
    "string": pylint_utils._unquote,
    "yn": _yn_transformer,
}
```
### 34 - pylint/config/arguments_manager.py:

Start line: 377, End line: 413

```python
class _ArgumentsManager:

    def optik_option(
        self, provider: ConfigProvider, opt: str, optdict: OptionDict
    ) -> tuple[list[str], OptionDict]:  # pragma: no cover
        """DEPRECATED: Get our personal option definition and return a suitable form for
        use with optik/optparse
        """
        warnings.warn(
            "optik_option has been deprecated. Parsing of option dictionaries should be done "
            "automatically by initializing an ArgumentsProvider.",
            DeprecationWarning,
        )
        optdict = copy.copy(optdict)
        if "action" in optdict:
            self._nocallback_options[provider] = opt
        else:
            optdict["action"] = "callback"
            optdict["callback"] = self.cb_set_provider_option
        # default is handled here and *must not* be given to optik if you
        # want the whole machinery to work
        if "default" in optdict:
            if (
                "help" in optdict
                and optdict.get("default") is not None
                and optdict["action"] not in ("store_true", "store_false")
            ):
                optdict["help"] += " [current: %default]"  # type: ignore[operator]
            del optdict["default"]
        args = ["--" + str(opt)]
        if "short" in optdict:
            self._short_options[optdict["short"]] = opt  # type: ignore[index]
            args.append("-" + optdict["short"])  # type: ignore[operator]
            del optdict["short"]
        # cleanup option definition dict before giving it to optik
        for key in list(optdict.keys()):
            if key not in self._optik_option_attrs:
                optdict.pop(key)
        return args, optdict
```
### 37 - pylint/config/arguments_manager.py:

Start line: 357, End line: 375

```python
class _ArgumentsManager:

    def add_optik_option(
        self,
        provider: ConfigProvider,
        optikcontainer: optparse.OptionParser | optparse.OptionGroup,
        opt: str,
        optdict: OptionDict,
    ) -> None:  # pragma: no cover
        """DEPRECATED."""
        warnings.warn(
            "add_optik_option has been deprecated. Options should be automatically "
            "added by initializing an ArgumentsProvider.",
            DeprecationWarning,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            args, optdict = self.optik_option(provider, opt, optdict)
        option = optikcontainer.add_option(*args, **optdict)
        self._all_options[opt] = provider
        self._maxlevel = max(self._maxlevel, option.level or 0)
```
### 38 - pylint/config/arguments_manager.py:

Start line: 158, End line: 234

```python
class _ArgumentsManager:

    @staticmethod
    def _add_parser_option(
        section_group: argparse._ArgumentGroup, argument: _Argument
    ) -> None:
        """Add an argument."""
        if isinstance(argument, _StoreArgument):
            section_group.add_argument(
                *argument.flags,
                action=argument.action,
                default=argument.default,
                type=argument.type,  # type: ignore[arg-type] # incorrect typing in typeshed
                help=argument.help,
                metavar=argument.metavar,
                choices=argument.choices,
            )
        elif isinstance(argument, _StoreOldNamesArgument):
            section_group.add_argument(
                *argument.flags,
                **argument.kwargs,
                action=argument.action,
                default=argument.default,
                type=argument.type,  # type: ignore[arg-type] # incorrect typing in typeshed
                help=argument.help,
                metavar=argument.metavar,
                choices=argument.choices,
            )
            # We add the old name as hidden option to make it's default value gets loaded when
            # argparse initializes all options from the checker
            assert argument.kwargs["old_names"]
            for old_name in argument.kwargs["old_names"]:
                section_group.add_argument(
                    f"--{old_name}",
                    action="store",
                    default=argument.default,
                    type=argument.type,  # type: ignore[arg-type] # incorrect typing in typeshed
                    help=argparse.SUPPRESS,
                    metavar=argument.metavar,
                    choices=argument.choices,
                )
        elif isinstance(argument, _StoreNewNamesArgument):
            section_group.add_argument(
                *argument.flags,
                **argument.kwargs,
                action=argument.action,
                default=argument.default,
                type=argument.type,  # type: ignore[arg-type] # incorrect typing in typeshed
                help=argument.help,
                metavar=argument.metavar,
                choices=argument.choices,
            )
        elif isinstance(argument, _StoreTrueArgument):
            section_group.add_argument(
                *argument.flags,
                action=argument.action,
                default=argument.default,
                help=argument.help,
            )
        elif isinstance(argument, _CallableArgument):
            section_group.add_argument(
                *argument.flags,
                **argument.kwargs,
                action=argument.action,
                help=argument.help,
            )
        elif isinstance(argument, _ExtendArgument):
            section_group.add_argument(
                *argument.flags,
                action=argument.action,
                default=argument.default,
                type=argument.type,  # type: ignore[arg-type] # incorrect typing in typeshed
                help=argument.help,
                metavar=argument.metavar,
                choices=argument.choices,
                dest=argument.dest,
            )
        else:
            raise UnrecognizedArgumentAction
```
### 43 - pylint/lint/base_options.py:

Start line: 395, End line: 570

```python
def _make_run_options(self: Run) -> Options:
    """Return the options used in a Run class."""
    return (
        (
            "rcfile",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "group": "Commands",
                "help": "Specify a configuration file to load.",
                "hide_from_config_file": True,
            },
        ),
        (
            "output",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "group": "Commands",
                "help": "Specify an output file.",
                "hide_from_config_file": True,
            },
        ),
        (
            "init-hook",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "help": "Python code to execute, usually for sys.path "
                "manipulation such as pygtk.require().",
            },
        ),
        (
            "help-msg",
            {
                "action": _MessageHelpAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a help message for the given message id and "
                "exit. The value may be a comma separated list of message ids.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-msgs",
            {
                "action": _ListMessagesAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a list of all pylint's messages divided by whether "
                "they are emittable with the given interpreter.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-msgs-enabled",
            {
                "action": _ListMessagesEnabledAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a list of what messages are enabled, "
                "disabled and non-emittable with the given configuration.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-groups",
            {
                "action": _ListCheckGroupsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "List pylint's message groups.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-conf-levels",
            {
                "action": _ListConfidenceLevelsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate pylint's confidence levels.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-extensions",
            {
                "action": _ListExtensionsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "List available extensions.",
                "hide_from_config_file": True,
            },
        ),
        (
            "full-documentation",
            {
                "action": _FullDocumentationAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate pylint's full documentation.",
                "hide_from_config_file": True,
            },
        ),
        (
            "generate-rcfile",
            {
                "action": _GenerateRCFileAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate a sample configuration file according to "
                "the current configuration. You can put other options "
                "before this one to get them in the generated "
                "configuration.",
                "hide_from_config_file": True,
            },
        ),
        (
            "generate-toml-config",
            {
                "action": _GenerateConfigFileAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate a sample configuration file according to "
                "the current configuration. You can put other options "
                "before this one to get them in the generated "
                "configuration. The config is in the .toml format.",
                "hide_from_config_file": True,
            },
        ),
        (
            "errors-only",
            {
                "action": _ErrorsOnlyModeAction,
                "kwargs": {"Run": self},
                "short": "E",
                "help": "In error mode, checkers without error messages are "
                "disabled and for others, only the ERROR messages are "
                "displayed, and no reports are done by default.",
                "hide_from_config_file": True,
            },
        ),
        (
            "verbose",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "short": "v",
                "help": "In verbose mode, extra non-checker-related info "
                "will be displayed.",
                "hide_from_config_file": True,
            },
        ),
        (
            "enable-all-extensions",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "help": "Load and enable all available extensions. "
                "Use --list-extensions to see a list all available extensions.",
                "hide_from_config_file": True,
            },
        ),
        (
            "long-help",
            {
                "action": _LongHelpAction,
                "kwargs": {"Run": self},
                "help": "Show more verbose help.",
                "group": "Commands",
                "hide_from_config_file": True,
            },
        ),
    )
```
### 55 - pylint/config/arguments_manager.py:

Start line: 647, End line: 655

```python
class _ArgumentsManager:

    def global_set_option(self, opt: str, value: Any) -> None:  # pragma: no cover
        """DEPRECATED: Set option on the correct option provider."""
        # TODO: 3.0: Remove deprecated method. # pylint: disable=fixme
        warnings.warn(
            "global_set_option has been deprecated. You can use _arguments_manager.set_option "
            "or linter.set_option to set options on the global configuration object.",
            DeprecationWarning,
        )
        self.set_option(opt, value)
```
### 66 - pylint/lint/base_options.py:

Start line: 7, End line: 392

```python
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from pylint import interfaces
from pylint.config.callback_actions import (
    _DisableAction,
    _DoNothingAction,
    _EnableAction,
    _ErrorsOnlyModeAction,
    _FullDocumentationAction,
    _GenerateConfigFileAction,
    _GenerateRCFileAction,
    _ListCheckGroupsAction,
    _ListConfidenceLevelsAction,
    _ListExtensionsAction,
    _ListMessagesAction,
    _ListMessagesEnabledAction,
    _LongHelpAction,
    _MessageHelpAction,
    _OutputFormatAction,
)
from pylint.typing import Options

if TYPE_CHECKING:
    from pylint.lint import PyLinter, Run


def _make_linter_options(linter: PyLinter) -> Options:
    """Return the options used in a PyLinter class."""
    return
 # ... other code
```
### 71 - pylint/config/arguments_manager.py:

Start line: 320, End line: 355

```python
class _ArgumentsManager:

    def add_option_group(
        self,
        group_name: str,
        _: str | None,
        options: list[tuple[str, OptionDict]],
        provider: ConfigProvider,
    ) -> None:  # pragma: no cover
        """DEPRECATED."""
        warnings.warn(
            "add_option_group has been deprecated. Option groups should be "
            "registered by initializing ArgumentsProvider. "
            "This automatically registers the group on the ArgumentsManager.",
            DeprecationWarning,
        )
        # add option group to the command line parser
        if group_name in self._mygroups:
            group = self._mygroups[group_name]
        else:
            group = optparse.OptionGroup(
                self.cmdline_parser, title=group_name.capitalize()
            )
            self.cmdline_parser.add_option_group(group)
            self._mygroups[group_name] = group
            # add section to the config file
            if (
                group_name != "DEFAULT"
                and group_name not in self.cfgfile_parser._sections  # type: ignore[attr-defined]
            ):
                self.cfgfile_parser.add_section(group_name)
        # add provider's specific options
        for opt, optdict in options:
            if not isinstance(optdict.get("action", "store"), str):
                optdict["action"] = "callback"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.add_optik_option(provider, group, opt, optdict)
```
### 86 - pylint/config/arguments_manager.py:

Start line: 264, End line: 278

```python
class _ArgumentsManager:

    def reset_parsers(self, usage: str = "") -> None:  # pragma: no cover
        """DEPRECATED."""
        warnings.warn(
            "reset_parsers has been deprecated. Parsers should be instantiated "
            "once during initialization and do not need to be reset.",
            DeprecationWarning,
        )
        # configuration file parser
        self.cfgfile_parser = configparser.ConfigParser(
            inline_comment_prefixes=("#", ";")
        )
        # command line parser
        self.cmdline_parser = OptionParser(Option, usage=usage)
        self.cmdline_parser.options_manager = self  # type: ignore[attr-defined]
        self._optik_option_attrs = set(self.cmdline_parser.option_class.ATTRS)
```
### 90 - pylint/config/arguments_manager.py:

Start line: 59, End line: 123

```python
# pylint: disable-next=too-many-instance-attributes
class _ArgumentsManager:
    """Arguments manager class used to handle command-line arguments and options."""

    def __init__(
        self, prog: str, usage: str | None = None, description: str | None = None
    ) -> None:
        self._config = argparse.Namespace()
        """Namespace for all options."""

        self._arg_parser = argparse.ArgumentParser(
            prog=prog,
            usage=usage or "%(prog)s [options]",
            description=description,
            formatter_class=_HelpFormatter,
        )
        """The command line argument parser."""

        self._argument_groups_dict: dict[str, argparse._ArgumentGroup] = {}
        """Dictionary of all the argument groups."""

        self._option_dicts: dict[str, OptionDict] = {}
        """All option dictionaries that have been registered."""

        # pylint: disable=fixme
        # TODO: 3.0: Remove deprecated attributes introduced to keep API
        # parity with optparse. Until '_maxlevel'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.reset_parsers(usage or "")
        # list of registered options providers
        self._options_providers: list[ConfigProvider] = []
        # dictionary associating option name to checker
        self._all_options: OrderedDict[str, ConfigProvider] = OrderedDict()
        self._short_options: dict[str, str] = {}
        self._nocallback_options: dict[ConfigProvider, str] = {}
        self._mygroups: dict[str, optparse.OptionGroup] = {}
        # verbosity
        self._maxlevel: int = 0

    @property
    def config(self) -> argparse.Namespace:
        """Namespace for all options."""
        return self._config

    @config.setter
    def config(self, value: argparse.Namespace) -> None:
        self._config = value

    @property
    def options_providers(self) -> list[ConfigProvider]:
        # TODO: 3.0: Remove deprecated attribute. # pylint: disable=fixme
        warnings.warn(
            "options_providers has been deprecated. It will be removed in pylint 3.0.",
            DeprecationWarning,
        )
        return self._options_providers

    @options_providers.setter
    def options_providers(self, value: list[ConfigProvider]) -> None:
        warnings.warn(
            "Setting options_providers has been deprecated. It will be removed in pylint 3.0.",
            DeprecationWarning,
        )
        self._options_providers = value
```
### 91 - pylint/config/utils.py:

Start line: 214, End line: 250

```python
def _preprocess_options(run: Run, args: Sequence[str]) -> list[str]:
    """Preprocess options before full config parsing has started."""
    processed_args: list[str] = []

    i = 0
    while i < len(args):
        argument = args[i]
        if not argument.startswith("--"):
            processed_args.append(argument)
            i += 1
            continue

        try:
            option, value = argument.split("=", 1)
        except ValueError:
            option, value = argument, None

        if option not in PREPROCESSABLE_OPTIONS:
            processed_args.append(argument)
            i += 1
            continue

        takearg, cb = PREPROCESSABLE_OPTIONS[option]

        if takearg and value is None:
            i += 1
            if i >= len(args) or args[i].startswith("-"):
                raise ArgumentPreprocessingError(f"Option {option} expects a value")
            value = args[i]
        elif not takearg and value is not None:
            raise ArgumentPreprocessingError(f"Option {option} doesn't expects a value")

        cb(run, value)
        i += 1

    return processed_args
```
### 105 - pylint/config/arguments_manager.py:

Start line: 484, End line: 523

```python
class _ArgumentsManager:

    def read_config_file(
        self, config_file: Path | None = None, verbose: bool = False
    ) -> None:  # pragma: no cover
        """DEPRECATED: Read the configuration file but do not load it (i.e. dispatching
        values to each option's provider)

        :raises OSError: Whem the specified config file doesn't exist
        """
        warnings.warn(
            "read_config_file has been deprecated. It will be removed in pylint 3.0.",
            DeprecationWarning,
        )
        if not config_file:
            if verbose:
                print(
                    "No config file found, using default configuration", file=sys.stderr
                )
            return
        config_file = Path(os.path.expandvars(config_file)).expanduser()
        if not config_file.exists():
            raise OSError(f"The config file {str(config_file)} doesn't exist!")
        parser = self.cfgfile_parser
        if config_file.suffix == ".toml":
            try:
                self._parse_toml(config_file, parser)
            except tomllib.TOMLDecodeError:
                pass
        else:
            # Use this encoding in order to strip the BOM marker, if any.
            with open(config_file, encoding="utf_8_sig") as fp:
                parser.read_file(fp)
            # normalize each section's title
            for sect, values in list(parser._sections.items()):  # type: ignore[attr-defined]
                if sect.startswith("pylint."):
                    sect = sect[len("pylint.") :]
                if not sect.isupper() and values:
                    parser._sections[sect.upper()] = values  # type: ignore[attr-defined]

        if verbose:
            print(f"Using config file '{config_file}'", file=sys.stderr)
```
### 110 - pylint/config/utils.py:

Start line: 7, End line: 28

```python
from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pylint import extensions, utils
from pylint.config.argument import (
    _CallableArgument,
    _ExtendArgument,
    _StoreArgument,
    _StoreNewNamesArgument,
    _StoreOldNamesArgument,
    _StoreTrueArgument,
)
from pylint.config.callback_actions import _CallbackAction
from pylint.config.exceptions import ArgumentPreprocessingError

if TYPE_CHECKING:
    from pylint.lint.run import Run
```
### 112 - pylint/config/arguments_manager.py:

Start line: 559, End line: 573

```python
class _ArgumentsManager:

    def load_config_file(self) -> None:  # pragma: no cover
        """DEPRECATED: Dispatch values previously read from a configuration file to each
        option's provider
        """
        warnings.warn(
            "load_config_file has been deprecated. It will be removed in pylint 3.0.",
            DeprecationWarning,
        )
        parser = self.cfgfile_parser
        for section in parser.sections():
            for option, value in parser.items(section):
                try:
                    self.global_set_option(option, value)
                except (KeyError, optparse.OptionError):
                    continue
```
### 113 - pylint/config/arguments_manager.py:

Start line: 728, End line: 754

```python
class _ArgumentsManager:

    def set_option(
        self,
        optname: str,
        value: Any,
        action: str | None = "default_value",
        optdict: None | str | OptionDict = "default_value",
    ) -> None:
        """Set an option on the namespace object."""
        # TODO: 3.0: Remove deprecated arguments. # pylint: disable=fixme
        if action != "default_value":
            warnings.warn(
                "The 'action' argument has been deprecated. You can use set_option "
                "without the 'action' or 'optdict' arguments.",
                DeprecationWarning,
            )
        if optdict != "default_value":
            warnings.warn(
                "The 'optdict' argument has been deprecated. You can use set_option "
                "without the 'action' or 'optdict' arguments.",
                DeprecationWarning,
            )

        self.config = self._arg_parser.parse_known_args(
            [f"--{optname.replace('_', '-')}", _parse_rich_type_value(value)],
            self.config,
        )[0]
```
### 121 - pylint/config/argument.py:

Start line: 10, End line: 113

```python
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Callable
from typing import Any, Pattern, Sequence, Tuple, Union

from pylint import interfaces
from pylint import utils as pylint_utils
from pylint.config.callback_actions import _CallbackAction, _ExtendAction
from pylint.config.deprecation_actions import _NewNamesAction, _OldNamesAction
from pylint.constants import PY38_PLUS

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


_ArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    Pattern[str],
    Sequence[str],
    Sequence[Pattern[str]],
    Tuple[int, ...],
]
"""List of possible argument types."""


def _confidence_transformer(value: str) -> Sequence[str]:
    """Transforms a comma separated string of confidence values."""
    values = pylint_utils._check_csv(value)
    for confidence in values:
        if confidence not in interfaces.CONFIDENCE_LEVEL_NAMES:
            raise argparse.ArgumentTypeError(
                f"{value} should be in {*interfaces.CONFIDENCE_LEVEL_NAMES,}"
            )
    return values


def _csv_transformer(value: str) -> Sequence[str]:
    """Transforms a comma separated string."""
    return pylint_utils._check_csv(value)


YES_VALUES = {"y", "yes", "true"}
NO_VALUES = {"n", "no", "false"}


def _yn_transformer(value: str) -> bool:
    """Transforms a yes/no or stringified bool into a bool."""
    value = value.lower()
    if value in YES_VALUES:
        return True
    if value in NO_VALUES:
        return False
    raise argparse.ArgumentTypeError(
        None, f"Invalid yn value '{value}', should be in {*YES_VALUES, *NO_VALUES}"
    )


def _non_empty_string_transformer(value: str) -> str:
    """Check that a string is not empty and remove quotes."""
    if not value:
        raise argparse.ArgumentTypeError("Option cannot be an empty string.")
    return pylint_utils._unquote(value)


def _py_version_transformer(value: str) -> tuple[int, ...]:
    """Transforms a version string into a version tuple."""
    try:
        version = tuple(int(val) for val in value.replace(",", ".").split("."))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value} has an invalid format, should be a version string. E.g., '3.8'"
        ) from None
    return version


def _regexp_csv_transfomer(value: str) -> Sequence[Pattern[str]]:
    """Transforms a comma separated list of regular expressions."""
    patterns: list[Pattern[str]] = []
    for pattern in _csv_transformer(value):
        patterns.append(re.compile(pattern))
    return patterns


def _regexp_paths_csv_transfomer(value: str) -> Sequence[Pattern[str]]:
    """Transforms a comma separated list of regular expressions paths."""
    patterns: list[Pattern[str]] = []
    for pattern in _csv_transformer(value):
        patterns.append(
            re.compile(
                str(pathlib.PureWindowsPath(pattern)).replace("\\", "\\\\")
                + "|"
                + pathlib.PureWindowsPath(pattern).as_posix()
            )
        )
    return patterns
```
### 162 - pylint/config/utils.py:

Start line: 31, End line: 122

```python
def _convert_option_to_argument(
    opt: str, optdict: dict[str, Any]
) -> (
    _StoreArgument
    | _StoreTrueArgument
    | _CallableArgument
    | _StoreOldNamesArgument
    | _StoreNewNamesArgument
    | _ExtendArgument
):
    """Convert an optdict to an Argument class instance."""
    if "level" in optdict and "hide" not in optdict:
        warnings.warn(
            "The 'level' key in optdicts has been deprecated. "
            "Use 'hide' with a boolean to hide an option from the help message.",
            DeprecationWarning,
        )

    # Get the long and short flags
    flags = [f"--{opt}"]
    if "short" in optdict:
        flags += [f"-{optdict['short']}"]

    # Get the action type
    action = optdict.get("action", "store")

    if action == "store_true":
        return _StoreTrueArgument(
            flags=flags,
            action=action,
            default=optdict.get("default", True),
            arg_help=optdict.get("help", ""),
            hide_help=optdict.get("hide", False),
            section=optdict.get("group", None),
        )
    if not isinstance(action, str) and issubclass(action, _CallbackAction):
        return _CallableArgument(
            flags=flags,
            action=action,
            arg_help=optdict.get("help", ""),
            kwargs=optdict.get("kwargs", {}),
            hide_help=optdict.get("hide", False),
            section=optdict.get("group", None),
        )
    try:
        default = optdict["default"]
    except KeyError:
        warnings.warn(
            "An option dictionary should have a 'default' key to specify "
            "the option's default value. This key will be required in pylint "
            "3.0. It is not required for 'store_true' and callable actions.",
            DeprecationWarning,
        )
        default = None
    if action == "extend":
        return _ExtendArgument(
            flags=flags,
            action=action,
            default=default,
            arg_type=optdict["type"],
            choices=optdict.get("choices", None),
            arg_help=optdict.get("help", ""),
            metavar=optdict.get("metavar", ""),
            hide_help=optdict.get("hide", False),
            section=optdict.get("group", None),
            dest=optdict.get("dest", None),
        )
    if "kwargs" in optdict:
        if "old_names" in optdict["kwargs"]:
            return _StoreOldNamesArgument(
                flags=flags,
                default=default,
                arg_type=optdict["type"],
                choices=optdict.get("choices", None),
                arg_help=optdict.get("help", ""),
                metavar=optdict.get("metavar", ""),
                hide_help=optdict.get("hide", False),
                kwargs=optdict.get("kwargs", {}),
                section=optdict.get("group", None),
            )
        if "new_names" in optdict["kwargs"]:
            return _StoreNewNamesArgument(
                flags=flags,
                default=default,
                arg_type=optdict["type"],
                choices=optdict.get("choices", None),
                arg_help=optdict.get("help", ""),
                metavar=optdict.get("metavar", ""),
                hide_help=optdict.get("hide", False),
                kwargs=optdict.get("kwargs", {}),
                section=optdict.get("group", None),
            )
    # ... other code
```
### 171 - pylint/config/arguments_manager.py:

Start line: 236, End line: 262

```python
class _ArgumentsManager:

    def _load_default_argument_values(self) -> None:
        """Loads the default values of all registered options."""
        self.config = self._arg_parser.parse_args([], self.config)

    def _parse_configuration_file(self, arguments: list[str]) -> None:
        """Parse the arguments found in a configuration file into the namespace."""
        self.config, parsed_args = self._arg_parser.parse_known_args(
            arguments, self.config
        )
        unrecognized_options: list[str] = []
        for opt in parsed_args:
            if opt.startswith("--"):
                unrecognized_options.append(opt[2:])
        if unrecognized_options:
            raise _UnrecognizedOptionError(options=unrecognized_options)

    def _parse_command_line_configuration(
        self, arguments: Sequence[str] | None = None
    ) -> list[str]:
        """Parse the arguments found on the command line into the namespace."""
        arguments = sys.argv[1:] if arguments is None else arguments

        self.config, parsed_args = self._arg_parser.parse_known_args(
            arguments, self.config
        )

        return parsed_args
```
### 175 - pylint/config/utils.py:

Start line: 148, End line: 211

```python
def _parse_rich_type_value(value: Any) -> str:
    """Parse rich (toml) types into strings."""
    if isinstance(value, (list, tuple)):
        return ",".join(_parse_rich_type_value(i) for i in value)
    if isinstance(value, re.Pattern):
        return value.pattern
    if isinstance(value, dict):
        return ",".join(f"{k}:{v}" for k, v in value.items())
    return str(value)


# pylint: disable-next=unused-argument
def _init_hook(run: Run, value: str | None) -> None:
    """Execute arbitrary code from the init_hook.

    This can be used to set the 'sys.path' for example.
    """
    assert value is not None
    exec(value)  # pylint: disable=exec-used


def _set_rcfile(run: Run, value: str | None) -> None:
    """Set the rcfile."""
    assert value is not None
    run._rcfile = value


def _set_output(run: Run, value: str | None) -> None:
    """Set the output."""
    assert value is not None
    run._output = value


def _add_plugins(run: Run, value: str | None) -> None:
    """Add plugins to the list of loadable plugins."""
    assert value is not None
    run._plugins.extend(utils._splitstrip(value))


def _set_verbose_mode(run: Run, value: str | None) -> None:
    assert value is None
    run.verbose = True


def _enable_all_extensions(run: Run, value: str | None) -> None:
    """Enable all extensions."""
    assert value is None
    for filename in Path(extensions.__file__).parent.iterdir():
        if filename.suffix == ".py" and not filename.stem.startswith("_"):
            extension_name = f"pylint.extensions.{filename.stem}"
            if extension_name not in run._plugins:
                run._plugins.append(extension_name)


PREPROCESSABLE_OPTIONS: dict[
    str, tuple[bool, Callable[[Run, str | None], None]]
] = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    "--init-hook": (True, _init_hook),
    "--rcfile": (True, _set_rcfile),
    "--output": (True, _set_output),
    "--load-plugins": (True, _add_plugins),
    "--verbose": (False, _set_verbose_mode),
    "--enable-all-extensions": (False, _enable_all_extensions),
}
```
### 189 - pylint/config/arguments_manager.py:

Start line: 142, End line: 156

```python
class _ArgumentsManager:

    def _add_arguments_to_parser(
        self, section: str, section_desc: str | None, argument: _Argument
    ) -> None:
        """Add an argument to the correct argument section/group."""
        try:
            section_group = self._argument_groups_dict[section]
        except KeyError:
            if section_desc:
                section_group = self._arg_parser.add_argument_group(
                    section, section_desc
                )
            else:
                section_group = self._arg_parser.add_argument_group(title=section)
            self._argument_groups_dict[section] = section_group
        self._add_parser_option(section_group, argument)
```
### 204 - pylint/config/arguments_manager.py:

Start line: 280, End line: 318

```python
class _ArgumentsManager:

    def register_options_provider(
        self, provider: ConfigProvider, own_group: bool = True
    ) -> None:  # pragma: no cover
        """DEPRECATED: Register an options provider."""
        warnings.warn(
            "register_options_provider has been deprecated. Options providers and "
            "arguments providers should be registered by initializing ArgumentsProvider. "
            "This automatically registers the provider on the ArgumentsManager.",
            DeprecationWarning,
        )
        self.options_providers.append(provider)
        non_group_spec_options = [
            option for option in provider.options if "group" not in option[1]
        ]
        groups = getattr(provider, "option_groups", ())
        if own_group and non_group_spec_options:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.add_option_group(
                    provider.name.upper(),
                    provider.__doc__,
                    non_group_spec_options,
                    provider,
                )
        else:
            for opt, optdict in non_group_spec_options:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    self.add_optik_option(provider, self.cmdline_parser, opt, optdict)
        for gname, gdoc in groups:
            gname = gname.upper()
            goptions = [
                option
                for option in provider.options
                if option[1].get("group", "").upper() == gname  # type: ignore[union-attr]
            ]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.add_option_group(gname, gdoc, goptions, provider)
```
