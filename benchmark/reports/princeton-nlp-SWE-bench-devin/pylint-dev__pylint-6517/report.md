# pylint-dev__pylint-6517

| **pylint-dev/pylint** | `58c4f370c7395d9d4e202ba83623768abcc3ac24` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 59 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -44,6 +44,8 @@
 
 def _confidence_transformer(value: str) -> Sequence[str]:
     """Transforms a comma separated string of confidence values."""
+    if not value:
+        return interfaces.CONFIDENCE_LEVEL_NAMES
     values = pylint_utils._check_csv(value)
     for confidence in values:
         if confidence not in interfaces.CONFIDENCE_LEVEL_NAMES:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/config/argument.py | 47 | 47 | - | 59 | -


## Problem Statement

```
Pylint runs unexpectedly pass if `confidence=` in pylintrc
### Bug description

Runs unexpectedly pass in 2.14 if a pylintrc file has `confidence=`.

(Default pylintrc files have `confidence=`. `pylint`'s own config was fixed in #6140 to comment it out, but this might bite existing projects.)

\`\`\`python
import time
\`\`\`

### Configuration

\`\`\`ini
[MESSAGES CONTROL]
confidence=
\`\`\`


### Command used

\`\`\`shell
python3 -m pylint a.py --enable=all
\`\`\`


### Pylint output

\`\`\`shell
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
\`\`\`


### Expected behavior
\`\`\`
************* Module a

a.py:2:0: C0305: Trailing newlines (trailing-newlines)
a.py:1:0: C0114: Missing module docstring (missing-module-docstring)
a.py:1:0: W0611: Unused import time (unused-import)

--------------------------------------------------------------------
Your code has been rated at 0.00/10 (previous run: 10.00/10, -10.00)
\`\`\`
### Pylint version

\`\`\`shell
pylint 2.14.0-dev0
astroid 2.12.0-dev0
Python 3.10.2 (v3.10.2:a58ebcc701, Jan 13 2022, 14:50:16) [Clang 13.0.0 (clang-1300.0.29.30)]
\`\`\`


### OS / Environment

_No response_

### Additional dependencies

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/__init__.py | 5 | 57| 335 | 335 | 729 | 
| 2 | 2 pylint/lint/run.py | 5 | 27| 148 | 483 | 2243 | 
| 3 | 3 pylint/epylint.py | 126 | 143| 114 | 597 | 3944 | 
| 4 | 4 pylint/lint/__init__.py | 17 | 44| 160 | 757 | 4234 | 
| 5 | 5 pylint/config/__init__.py | 5 | 37| 273 | 1030 | 4807 | 
| 6 | 6 pylint/constants.py | 92 | 199| 1388 | 2418 | 7461 | 
| 7 | 7 pylint/interfaces.py | 7 | 49| 291 | 2709 | 8364 | 
| 8 | 8 doc/data/messages/c/confusing-consecutive-elif/bad.py | 1 | 7| 0 | 2709 | 8422 | 
| 9 | 9 pylint/lint/pylinter.py | 103 | 198| 836 | 3545 | 20695 | 
| 10 | 10 pylint/testutils/_run.py | 28 | 46| 161 | 3706 | 21087 | 
| 11 | 10 pylint/epylint.py | 146 | 198| 491 | 4197 | 21087 | 
| 12 | 10 pylint/lint/run.py | 80 | 191| 837 | 5034 | 21087 | 
| 13 | 10 pylint/epylint.py | 201 | 216| 120 | 5154 | 21087 | 
| 14 | 11 pylint/lint/utils.py | 5 | 65| 419 | 5573 | 21939 | 
| 15 | 12 pylint/pyreverse/main.py | 230 | 250| 141 | 5714 | 23402 | 
| 16 | 12 pylint/constants.py | 5 | 90| 608 | 6322 | 23402 | 
| 17 | 12 pylint/pyreverse/main.py | 7 | 33| 160 | 6482 | 23402 | 
| 18 | 12 pylint/constants.py | 200 | 213| 122 | 6604 | 23402 | 
| 19 | 12 pylint/lint/pylinter.py | 5 | 101| 616 | 7220 | 23402 | 
| 20 | 13 pylint/config/callback_actions.py | 254 | 268| 133 | 7353 | 26021 | 
| 21 | 14 pylint/config/utils.py | 7 | 28| 118 | 7471 | 27973 | 
| 22 | 15 doc/data/messages/c/consider-using-sys-exit/bad.py | 1 | 5| 0 | 7471 | 28015 | 
| 23 | 16 pylint/checkers/stdlib.py | 7 | 41| 300 | 7771 | 34164 | 
| 24 | 17 doc/conf.py | 143 | 246| 596 | 8367 | 36084 | 
| 25 | 18 pylint/extensions/confusing_elif.py | 5 | 52| 411 | 8778 | 36560 | 
| 26 | 19 pylint/checkers/base_checker.py | 5 | 32| 166 | 8944 | 38832 | 
| 27 | 19 pylint/checkers/stdlib.py | 92 | 243| 1148 | 10092 | 38832 | 
| 28 | 20 pylint/config/find_default_config_files.py | 5 | 39| 241 | 10333 | 39584 | 
| 29 | 21 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 10333 | 39607 | 
| 30 | 21 pylint/epylint.py | 65 | 123| 521 | 10854 | 39607 | 
| 31 | 21 pylint/config/__init__.py | 53 | 64| 119 | 10973 | 39607 | 
| 32 | 21 pylint/testutils/_run.py | 10 | 25| 141 | 11114 | 39607 | 
| 33 | 22 doc/data/messages/a/arguments-out-of-order/bad.py | 1 | 14| 0 | 11114 | 39691 | 
| 34 | 23 pylint/testutils/constants.py | 5 | 30| 280 | 11394 | 40036 | 
| 35 | 24 doc/data/messages/c/consider-using-with/bad.py | 1 | 4| 0 | 11394 | 40066 | 
| 36 | 25 pylint/checkers/__init__.py | 44 | 64| 117 | 11511 | 41064 | 
| 37 | 26 pylint/checkers/design_analysis.py | 7 | 97| 739 | 12250 | 45935 | 
| 38 | 26 pylint/checkers/stdlib.py | 332 | 433| 1075 | 13325 | 45935 | 
| 39 | 27 doc/data/messages/p/potential-index-error/bad.py | 1 | 2| 0 | 13325 | 45954 | 
| 40 | 28 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 75| 495 | 13820 | 47740 | 
| 41 | 29 pylint/testutils/__init__.py | 7 | 36| 228 | 14048 | 48042 | 
| 42 | 29 pylint/checkers/stdlib.py | 43 | 89| 556 | 14604 | 48042 | 
| 43 | 30 doc/data/messages/c/consider-using-sys-exit/good.py | 1 | 7| 0 | 14604 | 48078 | 
| 44 | 30 pylint/lint/pylinter.py | 201 | 231| 278 | 14882 | 48078 | 
| 45 | 31 pylint/extensions/broad_try_clause.py | 7 | 73| 433 | 15315 | 48592 | 
| 46 | 32 pylint/config/arguments_manager.py | 7 | 56| 280 | 15595 | 54578 | 
| 47 | 32 pylint/lint/pylinter.py | 323 | 345| 202 | 15797 | 54578 | 
| 48 | 33 pylint/__main__.py | 1 | 11| 0 | 15797 | 54665 | 
| 49 | 34 doc/data/messages/a/assert-on-tuple/bad.py | 1 | 2| 0 | 15797 | 54679 | 
| 50 | 35 pylint/config/option.py | 184 | 202| 179 | 15976 | 56423 | 
| 51 | 35 pylint/config/__init__.py | 40 | 50| 115 | 16091 | 56423 | 
| 52 | 35 pylint/checkers/design_analysis.py | 98 | 176| 510 | 16601 | 56423 | 
| 53 | 36 doc/data/messages/r/redundant-unittest-assert/bad.py | 1 | 7| 0 | 16601 | 56456 | 
| 54 | 37 pylint/lint/parallel.py | 5 | 31| 153 | 16754 | 57908 | 
| 55 | 38 doc/data/messages/u/undefined-variable/bad.py | 1 | 2| 0 | 16754 | 57920 | 
| 56 | 39 doc/data/messages/u/undefined-all-variable/bad.py | 1 | 5| 0 | 16754 | 57946 | 
| 57 | 40 doc/data/messages/c/confusing-consecutive-elif/good.py | 1 | 23| 131 | 16885 | 58077 | 
| 58 | 41 pylint/checkers/refactoring/refactoring_checker.py | 5 | 63| 392 | 17277 | 75420 | 
| 59 | 42 doc/data/messages/p/potential-index-error/good.py | 1 | 2| 0 | 17277 | 75432 | 
| 60 | 42 pylint/lint/pylinter.py | 506 | 527| 216 | 17493 | 75432 | 
| 61 | 43 doc/data/messages/g/global-statement/bad.py | 1 | 12| 0 | 17493 | 75465 | 
| 62 | 44 pylint/checkers/typecheck.py | 800 | 952| 1076 | 18569 | 92650 | 
| 63 | 45 doc/data/messages/u/unreachable/bad.py | 1 | 4| 0 | 18569 | 92673 | 
| 64 | 46 pylint/extensions/bad_builtin.py | 7 | 22| 107 | 18676 | 93184 | 
| 65 | 47 pylint/checkers/similar.py | 20 | 84| 411 | 19087 | 100808 | 
| 66 | 47 pylint/config/utils.py | 149 | 213| 547 | 19634 | 100808 | 
| 67 | 48 doc/data/messages/a/arguments-out-of-order/good.py | 1 | 12| 0 | 19634 | 100880 | 
| 68 | 49 doc/data/messages/b/bad-super-call/bad.py | 1 | 8| 0 | 19634 | 100915 | 
| 69 | 50 doc/data/messages/g/global-at-module-level/bad.py | 1 | 3| 0 | 19634 | 100930 | 
| 70 | 51 doc/data/messages/a/assignment-from-no-return/bad.py | 1 | 6| 0 | 19634 | 100959 | 
| 71 | 52 doc/data/messages/r/reimported/bad.py | 1 | 3| 0 | 19634 | 100971 | 
| 72 | 52 pylint/pyreverse/main.py | 35 | 203| 892 | 20526 | 100971 | 
| 73 | 52 pylint/config/find_default_config_files.py | 79 | 92| 131 | 20657 | 100971 | 
| 74 | 53 pylint/checkers/unsupported_version.py | 9 | 62| 420 | 21077 | 101643 | 
| 75 | 54 pylint/testutils/output_line.py | 36 | 71| 256 | 21333 | 103081 | 
| 76 | 55 pylint/checkers/base/__init__.py | 5 | 47| 281 | 21614 | 103427 | 
| 77 | 56 doc/data/messages/r/return-in-init/bad.py | 1 | 5| 0 | 21614 | 103454 | 
| 78 | 57 pylint/checkers/format.py | 14 | 114| 718 | 22332 | 109423 | 
| 79 | 58 pylint/checkers/refactoring/__init__.py | 7 | 34| 180 | 22512 | 109679 | 
| 80 | 58 pylint/testutils/output_line.py | 5 | 33| 186 | 22698 | 109679 | 
| 81 | **59 pylint/config/argument.py** | 122 | 136| 139 | 22837 | 112747 | 
| 82 | 60 pylint/utils/file_state.py | 5 | 30| 122 | 22959 | 114227 | 
| 83 | 60 pylint/checkers/stdlib.py | 435 | 481| 510 | 23469 | 114227 | 
| 84 | 61 doc/data/messages/m/missing-raises-doc/bad.py | 1 | 9| 0 | 23469 | 114304 | 
| 85 | 62 doc/data/messages/a/assignment-from-none/bad.py | 1 | 6| 0 | 23469 | 114322 | 
| 86 | 63 doc/data/messages/a/assert-on-string-literal/bad.py | 1 | 4| 0 | 23469 | 114355 | 
| 87 | 64 doc/data/messages/c/consider-using-with/good.py | 1 | 3| 0 | 23469 | 114378 | 
| 88 | 65 doc/data/messages/m/missing-return-doc/bad.py | 1 | 7| 0 | 23469 | 114428 | 
| 89 | 66 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 23469 | 114442 | 
| 90 | 67 doc/data/messages/u/unnecessary-lambda-assignment/bad.py | 1 | 2| 0 | 23469 | 114467 | 
| 91 | 68 doc/data/messages/u/unnecessary-direct-lambda-call/bad.py | 1 | 2| 0 | 23469 | 114496 | 
| 92 | 69 doc/data/messages/u/undefined-variable/good.py | 1 | 3| 0 | 23469 | 114507 | 
| 93 | 70 pylint/lint/caching.py | 5 | 46| 291 | 23760 | 115023 | 
| 94 | 71 pylint/checkers/newstyle.py | 7 | 32| 144 | 23904 | 115913 | 
| 95 | 71 pylint/config/option.py | 5 | 55| 305 | 24209 | 115913 | 
| 96 | 72 doc/data/messages/t/too-many-arguments/bad.py | 1 | 17| 0 | 24209 | 115984 | 
| 97 | 73 doc/data/messages/u/unnecessary-dunder-call/bad.py | 1 | 7| 0 | 24209 | 116047 | 
| 98 | 74 pylint/checkers/logging.py | 7 | 99| 727 | 24936 | 119047 | 
| 99 | 75 doc/data/messages/c/catching-non-exception/bad.py | 1 | 9| 0 | 24936 | 119079 | 
| 100 | 76 pylint/testutils/global_test_linter.py | 5 | 21| 116 | 25052 | 119260 | 
| 101 | 77 pylint/checkers/exceptions.py | 7 | 29| 123 | 25175 | 123765 | 
| 102 | 78 doc/data/messages/u/undefined-all-variable/good.py | 1 | 5| 0 | 25175 | 123784 | 
| 103 | 78 pylint/lint/pylinter.py | 473 | 504| 278 | 25453 | 123784 | 
| 104 | 79 doc/data/messages/m/missing-yield-doc/bad.py | 1 | 10| 0 | 25453 | 123848 | 
| 105 | 80 doc/data/messages/s/self-assigning-variable/bad.py | 1 | 3| 0 | 25453 | 123866 | 
| 106 | 81 pylint/utils/__init__.py | 9 | 56| 268 | 25721 | 124219 | 
| 107 | 82 pylint/lint/base_options.py | 395 | 572| 1142 | 26863 | 128153 | 
| 108 | 83 pylint/checkers/variables.py | 7 | 132| 738 | 27601 | 150460 | 
| 109 | 84 doc/data/messages/u/unreachable/good.py | 1 | 4| 0 | 27601 | 150477 | 
| 110 | 85 doc/data/messages/s/super-with-arguments/bad.py | 1 | 8| 0 | 27601 | 150512 | 
| 111 | 86 pylint/checkers/refactoring/recommendation_checker.py | 5 | 81| 680 | 28281 | 153942 | 
| 112 | 87 pylint/reporters/reports_handler_mix_in.py | 5 | 20| 114 | 28395 | 154632 | 
| 113 | 88 doc/data/messages/u/unidiomatic-typecheck/bad.py | 1 | 4| 0 | 28395 | 154668 | 
| 114 | 89 doc/data/messages/m/missing-param-doc/bad.py | 1 | 6| 0 | 28395 | 154708 | 
| 115 | 90 doc/data/messages/u/useless-return/bad.py | 1 | 7| 0 | 28395 | 154732 | 
| 116 | 91 doc/data/messages/a/assignment-from-no-return/good.py | 1 | 6| 0 | 28395 | 154753 | 
| 117 | 92 doc/exts/pylint_features.py | 1 | 42| 241 | 28636 | 155073 | 
| 118 | 93 doc/data/messages/b/bad-builtin/bad.py | 1 | 3| 0 | 28636 | 155104 | 
| 119 | 94 doc/data/messages/u/unnecessary-ellipsis/bad.py | 1 | 4| 0 | 28636 | 155124 | 
| 120 | 95 pylint/checkers/base/pass_checker.py | 5 | 29| 162 | 28798 | 155351 | 
| 121 | 95 pylint/lint/pylinter.py | 990 | 1014| 190 | 28988 | 155351 | 
| 122 | 96 doc/data/messages/b/bad-super-call/good.py | 1 | 8| 0 | 28988 | 155374 | 
| 123 | 97 pylint/checkers/base/docstring_checker.py | 7 | 43| 203 | 29191 | 156977 | 
| 124 | 97 pylint/testutils/output_line.py | 74 | 99| 187 | 29378 | 156977 | 
| 125 | 98 pylint/checkers/strings.py | 7 | 59| 342 | 29720 | 165213 | 
| 126 | 99 doc/data/messages/n/no-else-continue/bad.py | 1 | 7| 0 | 29720 | 165256 | 
| 127 | 99 pylint/lint/utils.py | 68 | 104| 245 | 29965 | 165256 | 
| 128 | 99 pylint/checkers/stdlib.py | 246 | 299| 260 | 30225 | 165256 | 
| 129 | 100 pylint/pyreverse/__init__.py | 1 | 8| 0 | 30225 | 165334 | 
| 130 | 101 doc/data/messages/a/arguments-renamed/bad.py | 1 | 14| 0 | 30225 | 165436 | 
| 131 | 102 pylint/checkers/base/basic_error_checker.py | 99 | 207| 952 | 31177 | 169952 | 
| 132 | 103 pylint/checkers/ellipsis_checker.py | 7 | 59| 365 | 31542 | 170390 | 
| 133 | 104 doc/data/messages/n/no-else-break/bad.py | 1 | 7| 0 | 31542 | 170431 | 
| 134 | 105 pylint/checkers/base/basic_checker.py | 104 | 251| 1362 | 32904 | 177629 | 
| 135 | 106 doc/data/messages/a/await-outside-async/bad.py | 1 | 6| 0 | 32904 | 177651 | 
| 136 | 107 pylint/__pkginfo__.py | 10 | 44| 224 | 33128 | 177960 | 
| 137 | 108 doc/data/messages/m/missing-module-docstring/bad.py | 1 | 6| 0 | 33128 | 177980 | 
| 138 | 109 doc/data/messages/m/misplaced-future/bad.py | 1 | 4| 0 | 33128 | 177998 | 
| 139 | 110 doc/data/messages/u/use-dict-literal/bad.py | 1 | 2| 0 | 33128 | 178012 | 
| 140 | 111 doc/data/messages/c/comparison-with-itself/bad.py | 1 | 4| 0 | 33128 | 178041 | 
| 141 | 112 doc/data/messages/m/missing-format-argument-key/bad.py | 1 | 2| 0 | 33128 | 178066 | 
| 142 | 113 doc/data/messages/m/missing-yield-type-doc/bad.py | 1 | 13| 0 | 33128 | 178139 | 
| 143 | 114 doc/data/messages/a/assert-on-tuple/good.py | 1 | 4| 0 | 33128 | 178154 | 
| 144 | 115 doc/data/messages/c/chained-comparison/bad.py | 1 | 6| 0 | 33128 | 178189 | 
| 145 | 116 doc/data/messages/n/no-else-raise/bad.py | 1 | 6| 0 | 33128 | 178247 | 
| 146 | 116 pylint/lint/pylinter.py | 759 | 779| 212 | 33340 | 178247 | 
| 147 | 117 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 33340 | 178266 | 
| 148 | 118 doc/data/messages/s/super-without-brackets/bad.py | 1 | 12| 0 | 33340 | 178322 | 
| 149 | 119 doc/data/messages/u/ungrouped-imports/bad.py | 1 | 6| 0 | 33340 | 178352 | 
| 150 | 120 pylint/extensions/private_import.py | 7 | 39| 215 | 33555 | 180520 | 
| 151 | 121 pylint/utils/utils.py | 5 | 136| 809 | 34364 | 183615 | 
| 152 | 122 pylint/config/help_formatter.py | 32 | 66| 331 | 34695 | 184196 | 
| 153 | 123 doc/data/messages/t/too-many-arguments/good.py | 1 | 19| 0 | 34695 | 184272 | 
| 154 | 124 doc/data/messages/b/bad-exception-context/bad.py | 1 | 8| 0 | 34695 | 184333 | 
| 155 | 124 pylint/config/option.py | 149 | 182| 313 | 35008 | 184333 | 
| 156 | 125 doc/data/messages/r/redundant-unittest-assert/good.py | 1 | 8| 0 | 35008 | 184363 | 
| 157 | 126 doc/data/messages/a/attribute-defined-outside-init/bad.py | 1 | 4| 0 | 35008 | 184386 | 
| 158 | 127 doc/data/messages/u/use-maxsplit-arg/bad.py | 1 | 3| 0 | 35008 | 184410 | 
| 159 | 128 doc/data/messages/l/literal-comparison/bad.py | 1 | 3| 0 | 35008 | 184431 | 
| 160 | 129 doc/data/messages/r/redefined-loop-name/bad.py | 1 | 3| 0 | 35008 | 184450 | 
| 161 | 130 doc/data/messages/m/missing-raises-doc/good.py | 1 | 10| 0 | 35008 | 184533 | 
| 162 | 131 pylint/utils/linterstats.py | 143 | 160| 133 | 35141 | 187365 | 
| 163 | 132 script/__init__.py | 1 | 4| 0 | 35141 | 187430 | 
| 164 | 132 pylint/testutils/output_line.py | 101 | 119| 175 | 35316 | 187430 | 
| 165 | 132 pylint/lint/pylinter.py | 781 | 814| 329 | 35645 | 187430 | 
| 166 | 133 doc/data/messages/b/broad-except/bad.py | 1 | 5| 0 | 35645 | 187453 | 
| 167 | 134 doc/data/messages/u/unbalanced-tuple-unpacking/bad.py | 1 | 3| 0 | 35645 | 187487 | 
| 168 | 135 doc/data/messages/c/catching-non-exception/good.py | 1 | 9| 0 | 35645 | 187511 | 
| 169 | 135 pylint/checkers/typecheck.py | 7 | 92| 472 | 36117 | 187511 | 
| 170 | 136 doc/data/messages/u/unnecessary-list-index-lookup/bad.py | 1 | 5| 0 | 36117 | 187547 | 
| 171 | 136 pylint/checkers/variables.py | 2817 | 2844| 239 | 36356 | 187547 | 
| 172 | 137 doc/data/messages/b/bad-exception-context/good.py | 1 | 8| 0 | 36356 | 187603 | 
| 173 | 137 pylint/checkers/base_checker.py | 149 | 162| 148 | 36504 | 187603 | 
| 174 | 138 doc/data/messages/b/bad-str-strip-call/bad.py | 1 | 5| 0 | 36504 | 187649 | 
| 175 | 139 doc/data/messages/m/missing-type-doc/bad.py | 1 | 7| 0 | 36504 | 187697 | 
| 176 | 139 pylint/lint/run.py | 60 | 77| 147 | 36651 | 187697 | 
| 177 | 140 pylint/checkers/threading_checker.py | 5 | 43| 222 | 36873 | 188133 | 
| 178 | 141 doc/data/messages/u/unnecessary-direct-lambda-call/good.py | 1 | 2| 0 | 36873 | 188146 | 
| 179 | 141 pylint/lint/caching.py | 49 | 64| 159 | 37032 | 188146 | 
| 180 | 142 pylint/checkers/refactoring/not_checker.py | 5 | 41| 278 | 37310 | 188800 | 
| 181 | 142 pylint/lint/parallel.py | 55 | 97| 318 | 37628 | 188800 | 
| 182 | 142 pylint/lint/base_options.py | 7 | 392| 188 | 37816 | 188800 | 
| 183 | 143 pylint/extensions/redefined_loop_name.py | 7 | 33| 165 | 37981 | 189473 | 
| 184 | 144 doc/data/messages/r/reimported/good.py | 1 | 2| 0 | 37981 | 189476 | 
| 185 | 144 pylint/checkers/base/basic_error_checker.py | 7 | 24| 135 | 38116 | 189476 | 
| 186 | 145 doc/data/messages/m/missing-return-type-doc/bad.py | 1 | 8| 0 | 38116 | 189538 | 
| 187 | 146 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 38116 | 189551 | 
| 188 | 147 doc/data/messages/a/assignment-from-none/good.py | 1 | 6| 0 | 38116 | 189569 | 
| 189 | 148 pylint/pyreverse/utils.py | 7 | 108| 541 | 38657 | 191602 | 
| 190 | 149 doc/data/messages/u/unnecessary-dunder-call/good.py | 1 | 7| 0 | 38657 | 191635 | 


### Hint

```
The documentation of the option says "Leave empty to show all."
\`\`\`diff
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
index 8eb6417dc..bbaa7d0d8 100644
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -44,6 +44,8 @@ _ArgumentTypes = Union[
 
 def _confidence_transformer(value: str) -> Sequence[str]:
     """Transforms a comma separated string of confidence values."""
+    if not value:
+        return interfaces.CONFIDENCE_LEVEL_NAMES
     values = pylint_utils._check_csv(value)
     for confidence in values:
         if confidence not in interfaces.CONFIDENCE_LEVEL_NAMES:
\`\`\`

This will fix it.

I do wonder though: is this worth breaking? Probably not, but it is counter-intuitive and against all other config options behaviour to let an empty option mean something different. `confidence` should contain all confidence levels you want to show, so if it is empty you want none. Seems like a bad design choice when we added this, perhaps we can still fix it?...
Thanks for the speedy reply! I don't think we should bother to change how the option works. It's clearly documented, so that's something!
Hm okay. Would you be willing to prepare a PR with the patch? I had intended not to spend too much time on `pylint` this evening 😄 
```

## Patch

```diff
diff --git a/pylint/config/argument.py b/pylint/config/argument.py
--- a/pylint/config/argument.py
+++ b/pylint/config/argument.py
@@ -44,6 +44,8 @@
 
 def _confidence_transformer(value: str) -> Sequence[str]:
     """Transforms a comma separated string of confidence values."""
+    if not value:
+        return interfaces.CONFIDENCE_LEVEL_NAMES
     values = pylint_utils._check_csv(value)
     for confidence in values:
         if confidence not in interfaces.CONFIDENCE_LEVEL_NAMES:

```

## Test Patch

```diff
diff --git a/tests/config/test_config.py b/tests/config/test_config.py
--- a/tests/config/test_config.py
+++ b/tests/config/test_config.py
@@ -10,6 +10,7 @@
 import pytest
 from pytest import CaptureFixture
 
+from pylint.interfaces import CONFIDENCE_LEVEL_NAMES
 from pylint.lint import Run as LintRun
 from pylint.testutils._run import _Run as Run
 from pylint.testutils.configuration_test import run_using_a_configuration_file
@@ -88,6 +89,12 @@ def test_unknown_confidence(capsys: CaptureFixture) -> None:
     assert "argument --confidence: UNKNOWN_CONFIG should be in" in output.err
 
 
+def test_empty_confidence() -> None:
+    """An empty confidence value indicates all errors should be emitted."""
+    r = Run([str(EMPTY_MODULE), "--confidence="], exit=False)
+    assert r.linter.config.confidence == CONFIDENCE_LEVEL_NAMES
+
+
 def test_unknown_yes_no(capsys: CaptureFixture) -> None:
     """Check that we correctly error on an unknown yes/no value."""
     with pytest.raises(SystemExit):

```


## Code snippets

### 1 - pylint/__init__.py:

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
### 2 - pylint/lint/run.py:

Start line: 5, End line: 27

```python
from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pylint import config
from pylint.config.config_initialization import _config_initialization
from pylint.config.exceptions import ArgumentPreprocessingError
from pylint.config.utils import _preprocess_options
from pylint.constants import full_version
from pylint.lint.base_options import _make_run_options
from pylint.lint.pylinter import PyLinter
from pylint.reporters.base_reporter import BaseReporter

try:
    import multiprocessing
    from multiprocessing import synchronize  # noqa pylint: disable=unused-import
except ImportError:
    multiprocessing = None  # type: ignore[assignment]
```
### 3 - pylint/epylint.py:

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
### 4 - pylint/lint/__init__.py:

Start line: 17, End line: 44

```python
import sys

from pylint.config.exceptions import ArgumentPreprocessingError
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
]

if __name__ == "__main__":
    Run(sys.argv[1:])
```
### 5 - pylint/config/__init__.py:

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
### 6 - pylint/constants.py:

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
### 7 - pylint/interfaces.py:

Start line: 7, End line: 49

```python
from __future__ import annotations

import warnings
from collections import namedtuple
from tokenize import TokenInfo
from typing import TYPE_CHECKING

from astroid import nodes

if TYPE_CHECKING:
    from pylint.checkers import BaseChecker
    from pylint.message import Message
    from pylint.reporters.ureports.nodes import Section

__all__ = (
    "IRawChecker",
    "IAstroidChecker",
    "ITokenChecker",
    "IReporter",
    "IChecker",
    "HIGH",
    "CONTROL_FLOW",
    "INFERENCE",
    "INFERENCE_FAILURE",
    "UNDEFINED",
    "CONFIDENCE_LEVELS",
    "CONFIDENCE_LEVEL_NAMES",
)

Confidence = namedtuple("Confidence", ["name", "description"])
# Warning Certainties
HIGH = Confidence("HIGH", "Warning that is not based on inference result.")
CONTROL_FLOW = Confidence(
    "CONTROL_FLOW", "Warning based on assumptions about control flow."
)
INFERENCE = Confidence("INFERENCE", "Warning based on inference result.")
INFERENCE_FAILURE = Confidence(
    "INFERENCE_FAILURE", "Warning based on inference with failures."
)
UNDEFINED = Confidence("UNDEFINED", "Warning without any associated confidence level.")

CONFIDENCE_LEVELS = [HIGH, CONTROL_FLOW, INFERENCE, INFERENCE_FAILURE, UNDEFINED]
CONFIDENCE_LEVEL_NAMES = [i.name for i in CONFIDENCE_LEVELS]
```
### 8 - doc/data/messages/c/confusing-consecutive-elif/bad.py:

Start line: 1, End line: 7

```python

```
### 9 - pylint/lint/pylinter.py:

Start line: 103, End line: 198

```python
MSGS: dict[str, MessageDefinitionTuple] = {
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
    "F0011": (
        "error while parsing the configuration: %s",
        "config-parse-error",
        "Used when an exception occurred while parsing a pylint configuration file.",
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
        "Bad option value for %s",
        "bad-option-value",
        "Used when a bad value for an inline option is encountered.",
    ),
    "E0013": (
        "Plugin '%s' is impossible to load, is it installed ? ('%s')",
        "bad-plugin-value",
        "Used when a bad value is used in 'load-plugins'.",
    ),
    "E0014": (
        "Out-of-place setting encountered in top level configuration-section '%s' : '%s'",
        "bad-configuration-section",
        "Used when we detect a setting in the top level of a toml configuration that shouldn't be there.",
    ),
    "E0015": (
        "Unrecognized option found: %s",
        "unrecognized-option",
        "Used when we detect an option that we do not recognize.",
    ),
}
```
### 10 - pylint/testutils/_run.py:

Start line: 28, End line: 46

```python
class _Run(LintRun):

    """Like Run, but we're using an explicitly set empty pylintrc.

    We don't want to use the project's pylintrc during tests, because
    it means that a change in our config could break tests.
    But we want to see if the changes to the default break tests.
    """

    def __init__(
        self,
        args: Sequence[str],
        reporter: BaseReporter | None = None,
        exit: bool = True,  # pylint: disable=redefined-builtin
        do_exit: Any = UNUSED_PARAM_SENTINEL,
    ) -> None:
        args = _add_rcfile_default_pylintrc(list(args))
        super().__init__(args, reporter, exit, do_exit)
```
### 81 - pylint/config/argument.py:

Start line: 122, End line: 136

```python
_TYPE_TRANSFORMERS: dict[str, Callable[[str], _ArgumentTypes]] = {
    "choice": str,
    "csv": _csv_transformer,
    "float": float,
    "int": int,
    "confidence": _confidence_transformer,
    "non_empty_string": _non_empty_string_transformer,
    "path": _path_transformer,
    "py_version": _py_version_transformer,
    "regexp": re.compile,
    "regexp_csv": _regexp_csv_transfomer,
    "regexp_paths_csv": _regexp_paths_csv_transfomer,
    "string": pylint_utils._unquote,
    "yn": _yn_transformer,
}
```
