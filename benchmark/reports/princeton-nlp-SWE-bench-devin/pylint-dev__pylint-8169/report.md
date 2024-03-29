# pylint-dev__pylint-8169

| **pylint-dev/pylint** | `4689b195d8539ef04fd0c30423037a5f4932a20f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 28270 |
| **Any found context length** | 28270 |
| **Avg pos** | 76.0 |
| **Min pos** | 76 |
| **Max pos** | 76 |
| **Top file pos** | 18 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/checkers/variables.py b/pylint/checkers/variables.py
--- a/pylint/checkers/variables.py
+++ b/pylint/checkers/variables.py
@@ -2933,7 +2933,7 @@ def _check_module_attrs(
                 break
             try:
                 module = next(module.getattr(name)[0].infer())
-                if module is astroid.Uninferable:
+                if not isinstance(module, nodes.Module):
                     return None
             except astroid.NotFoundError:
                 if module.name in self._ignored_modules:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/checkers/variables.py | 2936 | 2936 | 76 | 18 | 28270


## Problem Statement

```
False positive `no-name-in-module` when importing from ``from ccxt.base.errors`` even when using the ``ignored-modules`` option
### Bug description

Simply importing exceptions from the [`ccxt`](https://github.com/ccxt/ccxt) library is giving this error. Here's an example of how we import them:

\`\`\`python
from ccxt.base.errors import (
    AuthenticationError,
    ExchangeError,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    RequestTimeout,
)
\`\`\`

Pycharm can find the exception classes just fine. I know they exist. It could have something to do with how the library is using `__all__`, but I don't know too much about how that works to draw that conclusion.

Also, note that we're using version 1.95.1 of `ccxt`. We use it in some critical paths, so we can't update it to the latest version quite yet.

The configuration written below is what I've tried, but it seems based on googling that that doesn't stop all errors from being ignored regarding those modules. So I'm still getting the issue.

### Configuration

\`\`\`ini
# List of module names for which member attributes should not be checked
# (useful for modules/projects where namespaces are manipulated during runtime
# and thus existing member attributes cannot be deduced by static analysis). It
# supports qualified module names, as well as Unix pattern matching.
ignored-modules=ccxt,ccxt.base,ccxt.base.errors
\`\`\`


### Command used

\`\`\`shell
pylint test_ccxt_base_errors.py
\`\`\`


### Pylint output

\`\`\`shell
************* Module test_ccxt_base_errors
test_ccxt_base_errors.py:1:0: E0611: No name 'errors' in module 'list' (no-name-in-module)
\`\`\`


### Expected behavior

No error to be reported

### Pylint version

\`\`\`shell
pylint 2.14.5
astroid 2.11.7
Python 3.9.16 (main, Dec  7 2022, 10:16:11)
[Clang 14.0.0 (clang-1400.0.29.202)]
\`\`\`


### OS / Environment

Intel based 2019 Mac Book Pro. Mac OS 13.1 (Ventura). Fish shell.

### Additional dependencies

ccxt==1.95.1

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/checkers/imports.py | 327 | 451| 776 | 776 | 8845 | 
| 2 | 1 pylint/checkers/imports.py | 7 | 78| 549 | 1325 | 8845 | 
| 3 | 1 pylint/checkers/imports.py | 842 | 872| 303 | 1628 | 8845 | 
| 4 | 1 pylint/checkers/imports.py | 234 | 324| 764 | 2392 | 8845 | 
| 5 | 1 pylint/checkers/imports.py | 1045 | 1079| 277 | 2669 | 8845 | 
| 6 | 1 pylint/checkers/imports.py | 762 | 840| 697 | 3366 | 8845 | 
| 7 | 2 pylint/checkers/stdlib.py | 99 | 257| 1200 | 4566 | 15866 | 
| 8 | 2 pylint/checkers/imports.py | 566 | 592| 287 | 4853 | 15866 | 
| 9 | 2 pylint/checkers/imports.py | 634 | 664| 238 | 5091 | 15866 | 
| 10 | 3 pylint/checkers/base/basic_error_checker.py | 100 | 207| 941 | 6032 | 20632 | 
| 11 | 4 pylint/extensions/private_import.py | 7 | 38| 215 | 6247 | 22832 | 
| 12 | 4 pylint/checkers/imports.py | 666 | 687| 218 | 6465 | 22832 | 
| 13 | 5 pylint/checkers/non_ascii_names.py | 155 | 184| 313 | 6778 | 24529 | 
| 14 | 6 pylint/constants.py | 5 | 99| 732 | 7510 | 27144 | 
| 15 | 7 pylint/lint/pylinter.py | 104 | 235| 1089 | 8599 | 38098 | 
| 16 | 8 doc/data/messages/n/non-ascii-module-import/bad.py | 1 | 4| 0 | 8599 | 38125 | 
| 17 | 9 pylint/checkers/typecheck.py | 812 | 964| 1076 | 9675 | 56425 | 
| 18 | 9 pylint/checkers/imports.py | 594 | 632| 312 | 9987 | 56425 | 
| 19 | 10 doc/data/messages/n/non-ascii-module-import/good.py | 1 | 4| 0 | 9987 | 56444 | 
| 20 | 10 pylint/checkers/stdlib.py | 7 | 46| 351 | 10338 | 56444 | 
| 21 | 10 pylint/extensions/private_import.py | 57 | 96| 333 | 10671 | 56444 | 
| 22 | 11 pylint/checkers/utils.py | 2126 | 2157| 209 | 10880 | 73294 | 
| 23 | 11 pylint/checkers/imports.py | 453 | 465| 165 | 11045 | 73294 | 
| 24 | 11 pylint/checkers/imports.py | 948 | 975| 237 | 11282 | 73294 | 
| 25 | 12 pylint/checkers/base/name_checker/checker.py | 325 | 355| 295 | 11577 | 78666 | 
| 26 | 13 pylint/checkers/exceptions.py | 7 | 34| 175 | 11752 | 84045 | 
| 27 | 13 pylint/checkers/non_ascii_names.py | 30 | 72| 400 | 12152 | 84045 | 
| 28 | 14 doc/data/messages/m/multiple-imports/bad.py | 1 | 2| 0 | 12152 | 84056 | 
| 29 | 15 doc/data/messages/u/ungrouped-imports/bad.py | 1 | 6| 0 | 12152 | 84086 | 
| 30 | 15 pylint/checkers/imports.py | 915 | 946| 263 | 12415 | 84086 | 
| 31 | 16 doc/data/messages/c/consider-using-from-import/bad.py | 1 | 2| 0 | 12415 | 84099 | 
| 32 | 17 doc/data/messages/c/cyclic-import/good.py | 1 | 2| 0 | 12415 | 84111 | 
| 33 | 17 pylint/checkers/base/basic_error_checker.py | 7 | 25| 145 | 12560 | 84111 | 
| 34 | 17 pylint/checkers/stdlib.py | 349 | 468| 1273 | 13833 | 84111 | 
| 35 | **18 pylint/checkers/variables.py** | 3029 | 3109| 684 | 14517 | 109933 | 
| 36 | 19 pylint/checkers/design_analysis.py | 92 | 170| 510 | 15027 | 114786 | 
| 37 | **19 pylint/checkers/variables.py** | 1900 | 1923| 228 | 15255 | 114786 | 
| 38 | **19 pylint/checkers/variables.py** | 1878 | 1898| 200 | 15455 | 114786 | 
| 39 | 19 pylint/checkers/imports.py | 513 | 536| 206 | 15661 | 114786 | 
| 40 | 19 pylint/checkers/base/name_checker/checker.py | 152 | 192| 381 | 16042 | 114786 | 
| 41 | 20 doc/data/messages/m/multiple-imports/good.py | 1 | 3| 0 | 16042 | 114792 | 
| 42 | 21 doc/data/messages/w/wildcard-import/bad.py | 1 | 2| 0 | 16042 | 114803 | 
| 43 | 21 pylint/checkers/exceptions.py | 435 | 486| 402 | 16444 | 114803 | 
| 44 | 21 pylint/checkers/imports.py | 538 | 564| 259 | 16703 | 114803 | 
| 45 | 22 pylint/checkers/base/basic_checker.py | 107 | 268| 1505 | 18208 | 123141 | 
| 46 | 22 pylint/checkers/imports.py | 148 | 180| 279 | 18487 | 123141 | 
| 47 | 23 pylint/interfaces.py | 7 | 53| 291 | 18778 | 124056 | 
| 48 | 23 pylint/checkers/imports.py | 485 | 511| 290 | 19068 | 124056 | 
| 49 | 24 pylint/checkers/classes/class_checker.py | 7 | 76| 496 | 19564 | 141524 | 
| 50 | 25 pylint/message/_deleted_message_ids.py | 5 | 126| 1415 | 20979 | 143432 | 
| 51 | 26 pylint/checkers/logging.py | 180 | 194| 150 | 21129 | 146664 | 
| 52 | 26 pylint/checkers/imports.py | 874 | 913| 375 | 21504 | 146664 | 
| 53 | 26 pylint/extensions/private_import.py | 40 | 55| 159 | 21663 | 146664 | 
| 54 | 26 pylint/checkers/base/name_checker/checker.py | 78 | 97| 168 | 21831 | 146664 | 
| 55 | 27 doc/data/messages/n/no-name-in-module/bad.py | 1 | 2| 0 | 21831 | 146676 | 
| 56 | 27 pylint/checkers/exceptions.py | 389 | 433| 466 | 22297 | 146676 | 
| 57 | 27 pylint/checkers/design_analysis.py | 7 | 91| 703 | 23000 | 146676 | 
| 58 | 28 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 23000 | 146695 | 
| 59 | 29 doc/data/messages/u/ungrouped-imports/good.py | 1 | 6| 0 | 23000 | 146715 | 
| 60 | 29 pylint/checkers/non_ascii_names.py | 130 | 153| 287 | 23287 | 146715 | 
| 61 | 30 pylint/checkers/strings.py | 7 | 66| 381 | 23668 | 155193 | 
| 62 | 30 pylint/checkers/imports.py | 93 | 145| 377 | 24045 | 155193 | 
| 63 | 30 pylint/lint/pylinter.py | 5 | 103| 643 | 24688 | 155193 | 
| 64 | 30 pylint/checkers/imports.py | 741 | 761| 267 | 24955 | 155193 | 
| 65 | 30 pylint/checkers/non_ascii_names.py | 13 | 27| 133 | 25088 | 155193 | 
| 66 | 30 pylint/checkers/non_ascii_names.py | 74 | 97| 234 | 25322 | 155193 | 
| 67 | 30 pylint/checkers/imports.py | 467 | 483| 203 | 25525 | 155193 | 
| 68 | 30 pylint/extensions/private_import.py | 113 | 131| 145 | 25670 | 155193 | 
| 69 | 30 pylint/checkers/exceptions.py | 290 | 321| 232 | 25902 | 155193 | 
| 70 | 31 doc/data/messages/c/consider-using-from-import/good.py | 1 | 2| 0 | 25902 | 155198 | 
| 71 | 31 pylint/checkers/exceptions.py | 63 | 182| 1254 | 27156 | 155198 | 
| 72 | 31 pylint/checkers/non_ascii_names.py | 99 | 128| 246 | 27402 | 155198 | 
| 73 | 31 pylint/extensions/private_import.py | 245 | 262| 149 | 27551 | 155198 | 
| 74 | 32 pylint/checkers/base_checker.py | 5 | 32| 168 | 27719 | 157605 | 
| 75 | 32 pylint/extensions/private_import.py | 187 | 212| 286 | 28005 | 157605 | 
| **-> 76 <-** | **32 pylint/checkers/variables.py** | 2931 | 2966| 265 | 28270 | 157605 | 
| 77 | 33 doc/data/messages/n/no-name-in-module/good.py | 1 | 2| 0 | 28270 | 157610 | 
| 78 | 33 pylint/checkers/imports.py | 708 | 740| 317 | 28587 | 157610 | 
| 79 | **33 pylint/checkers/variables.py** | 1472 | 1542| 467 | 29054 | 157610 | 
| 80 | 33 pylint/checkers/stdlib.py | 470 | 522| 573 | 29627 | 157610 | 
| 81 | 34 doc/data/messages/w/wrong-import-order/bad.py | 1 | 5| 0 | 29627 | 157636 | 
| 82 | 34 pylint/checkers/stdlib.py | 260 | 316| 288 | 29915 | 157636 | 
| 83 | 34 pylint/checkers/base/name_checker/checker.py | 396 | 475| 686 | 30601 | 157636 | 
| 84 | 35 pylint/config/__init__.py | 5 | 39| 291 | 30892 | 158239 | 
| 85 | 35 pylint/checkers/utils.py | 7 | 74| 494 | 31386 | 158239 | 
| 86 | 36 doc/data/messages/g/global-at-module-level/bad.py | 1 | 3| 0 | 31386 | 158254 | 
| 87 | 37 pylint/checkers/base/__init__.py | 7 | 49| 281 | 31667 | 158621 | 
| 88 | 38 doc/data/messages/w/wrong-import-order/good.py | 1 | 7| 0 | 31667 | 158635 | 
| 89 | 39 doc/data/messages/s/shadowed-import/bad.py | 1 | 4| 0 | 31667 | 158653 | 
| 90 | **39 pylint/checkers/variables.py** | 269 | 320| 456 | 32123 | 158653 | 
| 91 | 39 pylint/checkers/classes/class_checker.py | 1078 | 1153| 644 | 32767 | 158653 | 
| 92 | 40 pylint/extensions/bad_builtin.py | 7 | 49| 272 | 33039 | 159162 | 
| 93 | 40 pylint/checkers/base/basic_error_checker.py | 301 | 327| 180 | 33219 | 159162 | 
| 94 | 40 pylint/lint/pylinter.py | 238 | 269| 284 | 33503 | 159162 | 
| 95 | 40 pylint/checkers/stdlib.py | 48 | 96| 587 | 34090 | 159162 | 
| 96 | 40 pylint/checkers/imports.py | 1019 | 1043| 224 | 34314 | 159162 | 
| 97 | 41 doc/data/messages/w/wildcard-import/good.py | 1 | 5| 0 | 34314 | 159187 | 
| 98 | 42 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 34314 | 159200 | 
| 99 | 42 pylint/lint/pylinter.py | 1054 | 1078| 195 | 34509 | 159200 | 
| 100 | 43 doc/data/messages/r/reimported/bad.py | 1 | 3| 0 | 34509 | 159212 | 
| 101 | 43 pylint/checkers/base/name_checker/checker.py | 357 | 394| 333 | 34842 | 159212 | 
| 102 | 43 pylint/checkers/base/basic_error_checker.py | 397 | 418| 209 | 35051 | 159212 | 
| 103 | 44 pylint/checkers/base/name_checker/__init__.py | 5 | 26| 116 | 35167 | 159393 | 
| 104 | 45 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 5| 0 | 35167 | 159404 | 
| 105 | 45 pylint/checkers/stdlib.py | 800 | 838| 303 | 35470 | 159404 | 
| 106 | 46 doc/data/messages/u/unused-import/bad.py | 1 | 5| 0 | 35470 | 159425 | 
| 107 | 47 doc/data/messages/i/import-error/good.py | 1 | 2| 0 | 35470 | 159430 | 
| 108 | 48 doc/data/messages/i/import-error/bad.py | 1 | 2| 0 | 35470 | 159441 | 
| 109 | 49 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 35470 | 159464 | 
| 110 | 50 pylint/checkers/format.py | 14 | 119| 746 | 36216 | 165572 | 
| 111 | 50 pylint/checkers/exceptions.py | 323 | 349| 202 | 36418 | 165572 | 
| 112 | 51 pylint/lint/__init__.py | 17 | 47| 183 | 36601 | 165885 | 
| 113 | 52 pylint/extensions/typing.py | 88 | 161| 759 | 37360 | 170369 | 
| 114 | 52 pylint/checkers/exceptions.py | 351 | 370| 177 | 37537 | 170369 | 
| 115 | 53 doc/data/messages/u/useless-import-alias/good.py | 1 | 2| 0 | 37537 | 170374 | 
| 116 | 54 doc/data/messages/p/preferred-module/bad.py | 1 | 2| 0 | 37537 | 170382 | 
| 117 | 54 pylint/checkers/logging.py | 7 | 94| 727 | 38264 | 170382 | 
| 118 | 55 pylint/checkers/unsupported_version.py | 9 | 62| 420 | 38684 | 171054 | 
| 119 | 56 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 75| 499 | 39183 | 173013 | 
| 120 | 57 pylint/checkers/spelling.py | 7 | 116| 692 | 39875 | 176629 | 
| 121 | 57 pylint/checkers/base/name_checker/checker.py | 7 | 59| 330 | 40205 | 176629 | 
| 122 | 57 pylint/checkers/exceptions.py | 488 | 543| 478 | 40683 | 176629 | 
| 123 | 58 pylint/checkers/misc.py | 7 | 40| 192 | 40875 | 177727 | 
| 124 | 58 pylint/checkers/classes/class_checker.py | 1558 | 1585| 248 | 41123 | 177727 | 
| 125 | 58 pylint/checkers/base/basic_error_checker.py | 209 | 224| 164 | 41287 | 177727 | 
| 126 | 59 doc/data/messages/w/wrong-import-position/bad.py | 1 | 8| 0 | 41287 | 177759 | 
| 127 | 59 pylint/checkers/base/basic_error_checker.py | 329 | 355| 181 | 41468 | 177759 | 
| 128 | 59 pylint/checkers/base/basic_checker.py | 7 | 63| 348 | 41816 | 177759 | 
| 129 | 60 doc/data/messages/w/wrong-import-position/good.py | 1 | 6| 0 | 41816 | 177785 | 
| 130 | 60 pylint/checkers/base/basic_error_checker.py | 259 | 299| 376 | 42192 | 177785 | 
| 131 | 60 pylint/checkers/base/basic_checker.py | 303 | 383| 664 | 42856 | 177785 | 
| 132 | 60 pylint/checkers/typecheck.py | 789 | 809| 180 | 43036 | 177785 | 
| 133 | 60 pylint/checkers/base/name_checker/checker.py | 194 | 279| 657 | 43693 | 177785 | 
| 134 | 61 doc/data/messages/c/catching-non-exception/bad.py | 1 | 9| 0 | 43693 | 177817 | 
| 135 | 61 pylint/extensions/private_import.py | 214 | 243| 244 | 43937 | 177817 | 
| 136 | 61 pylint/checkers/exceptions.py | 562 | 660| 680 | 44617 | 177817 | 
| 137 | 62 pylint/checkers/newstyle.py | 7 | 32| 144 | 44761 | 178758 | 
| 138 | 63 pylint/checkers/threading_checker.py | 5 | 43| 222 | 44983 | 179194 | 
| 139 | 63 pylint/checkers/base/name_checker/checker.py | 137 | 149| 113 | 45096 | 179194 | 
| 140 | **63 pylint/checkers/variables.py** | 393 | 537| 1326 | 46422 | 179194 | 
| 141 | 64 doc/data/messages/g/global-at-module-level/good.py | 1 | 2| 0 | 46422 | 179199 | 
| 142 | 65 doc/data/messages/i/import-self/bad.py | 1 | 6| 0 | 46422 | 179217 | 
| 143 | 65 pylint/checkers/imports.py | 1008 | 1017| 125 | 46547 | 179217 | 
| 144 | 66 pylint/extensions/overlapping_exceptions.py | 7 | 91| 593 | 47140 | 179881 | 
| 145 | 66 pylint/extensions/private_import.py | 98 | 111| 160 | 47300 | 179881 | 
| 146 | 66 pylint/lint/pylinter.py | 887 | 902| 163 | 47463 | 179881 | 
| 147 | 67 doc/data/messages/s/shadowed-import/good.py | 1 | 4| 0 | 47463 | 179895 | 
| 148 | 68 doc/data/messages/u/unused-import/good.py | 1 | 4| 0 | 47463 | 179906 | 
| 149 | 68 pylint/checkers/typecheck.py | 7 | 106| 579 | 48042 | 179906 | 
| 150 | 69 pylint/pyreverse/inspector.py | 10 | 39| 174 | 48216 | 183058 | 
| 151 | 70 examples/deprecation_checker.py | 1 | 48| 462 | 48678 | 183994 | 
| 152 | 70 pylint/checkers/imports.py | 689 | 706| 168 | 48846 | 183994 | 
| 153 | 71 pylint/checkers/__init__.py | 43 | 63| 117 | 48963 | 185000 | 
| 154 | 72 doc/data/messages/r/reimported/good.py | 1 | 2| 0 | 48963 | 185003 | 
| 155 | 73 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 48963 | 185017 | 
| 156 | 74 doc/data/messages/c/c-extension-no-member/good.py | 1 | 2| 0 | 48963 | 185029 | 
| 157 | 74 pylint/checkers/stdlib.py | 697 | 759| 454 | 49417 | 185029 | 
| 158 | 75 doc/data/messages/c/catching-non-exception/good.py | 1 | 9| 0 | 49417 | 185053 | 
| 159 | 75 pylint/lint/pylinter.py | 753 | 774| 234 | 49651 | 185053 | 
| 160 | 75 pylint/checkers/classes/class_checker.py | 877 | 910| 252 | 49903 | 185053 | 
| 161 | 76 pylint/lint/report_functions.py | 45 | 86| 366 | 50269 | 185753 | 
| 162 | **76 pylint/checkers/variables.py** | 323 | 342| 177 | 50446 | 185753 | 
| 163 | **76 pylint/checkers/variables.py** | 2968 | 3028| 504 | 50950 | 185753 | 
| 164 | **76 pylint/checkers/variables.py** | 2562 | 2654| 723 | 51673 | 185753 | 
| 165 | 77 pylint/lint/expand_modules.py | 5 | 23| 143 | 51816 | 187058 | 
| 166 | 77 pylint/checkers/misc.py | 53 | 94| 219 | 52035 | 187058 | 
| 167 | 77 pylint/checkers/typecheck.py | 1821 | 1889| 549 | 52584 | 187058 | 
| 168 | 77 pylint/extensions/typing.py | 381 | 422| 309 | 52893 | 187058 | 
| 169 | 77 pylint/checkers/stdlib.py | 761 | 798| 283 | 53176 | 187058 | 


### Hint

```
Could you upgrade to at least 2.15.10 (better yet 2.16.0b1) and confirm the issue still exists, please ?
Tried with

\`\`\`
pylint 2.15.10
astroid 2.13.4
Python 3.9.16 (main, Dec  7 2022, 10:16:11)
[Clang 14.0.0 (clang-1400.0.29.202)]
\`\`\`
and also with pylint 2.16.0b1 and I still get the same issue.
Thank you ! I can reproduce, and ``ignored-modules`` does work with a simpler example like ``random.foo``
@Pierre-Sassoulas is the fix here:
1. figure out why the ccxt library causes a `no-name-in-module` msg
2. figure out why using `ignored-modules` is still raising `no-name-in-module`
?
Yes, I think 2/ is the one to prioritize as it's going to be useful for everyone and not just ccxt users. But if we manage find the root cause of 1/ it's going to be generic too.
There is a non-ccxt root cause. This issue can be reproduced with the following dir structure:

\`\`\`
pkg_mod_imports/__init__.py
pkg_mod_imports/base/__init__.py
pkg_mod_imports/base/errors.py
\`\`\`
pkg_mod_imports/__init__.py should have :
\`\`\`
base = [
    'Exchange',
    'Precise',
    'exchanges',
    'decimal_to_precision',
]
\`\`\`

and  pkg_mod_imports/base/errors.py

\`\`\`
class SomeError(Exception):
    pass
\`\`\`

in a test.py module add
\`\`\`
from pkg_mod_imports.base.errors import SomeError
\`\`\`
And then running `pylint test.py` the result is
\`\`\`
test.py:1:0: E0611: No name 'errors' in module 'list' (no-name-in-module)
\`\`\`

It's coming from the fact that `errors` is both a list inside the init file and the name of a module. variable.py does `module = next(module.getattr(name)[0].infer())` . `getattr` fetches the `errors` list, not the module!
```

## Patch

```diff
diff --git a/pylint/checkers/variables.py b/pylint/checkers/variables.py
--- a/pylint/checkers/variables.py
+++ b/pylint/checkers/variables.py
@@ -2933,7 +2933,7 @@ def _check_module_attrs(
                 break
             try:
                 module = next(module.getattr(name)[0].infer())
-                if module is astroid.Uninferable:
+                if not isinstance(module, nodes.Module):
                     return None
             except astroid.NotFoundError:
                 if module.name in self._ignored_modules:

```

## Test Patch

```diff
diff --git a/tests/regrtest_data/pkg_mod_imports/__init__.py b/tests/regrtest_data/pkg_mod_imports/__init__.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/pkg_mod_imports/__init__.py
@@ -0,0 +1,6 @@
+base = [
+    'Exchange',
+    'Precise',
+    'exchanges',
+    'decimal_to_precision',
+]
diff --git a/tests/regrtest_data/pkg_mod_imports/base/__init__.py b/tests/regrtest_data/pkg_mod_imports/base/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/pkg_mod_imports/base/errors.py b/tests/regrtest_data/pkg_mod_imports/base/errors.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/pkg_mod_imports/base/errors.py
@@ -0,0 +1,2 @@
+class SomeError(Exception):
+    pass
diff --git a/tests/regrtest_data/test_no_name_in_module.py b/tests/regrtest_data/test_no_name_in_module.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/test_no_name_in_module.py
@@ -0,0 +1 @@
+from pkg_mod_imports.base.errors import SomeError
diff --git a/tests/test_self.py b/tests/test_self.py
--- a/tests/test_self.py
+++ b/tests/test_self.py
@@ -1293,6 +1293,15 @@ def test_output_no_header(self) -> None:
             args, expected_output=expected, unexpected_output=not_expected
         )
 
+    def test_no_name_in_module(self) -> None:
+        """Test that a package with both a variable name `base` and a module `base`
+        does not emit a no-name-in-module msg."""
+        module = join(HERE, "regrtest_data", "test_no_name_in_module.py")
+        unexpected = "No name 'errors' in module 'list' (no-name-in-module)"
+        self._test_output(
+            [module, "-E"], expected_output="", unexpected_output=unexpected
+        )
+
 
 class TestCallbackOptions:
     """Test for all callback options we support."""

```


## Code snippets

### 1 - pylint/checkers/imports.py:

Start line: 327, End line: 451

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
        (
            "allow-reexport-from-package",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Allow explicit reexports by alias from a package __init__.",
            },
        ),
    )
```
### 2 - pylint/checkers/imports.py:

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
### 3 - pylint/checkers/imports.py:

Start line: 842, End line: 872

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
### 4 - pylint/checkers/imports.py:

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
### 5 - pylint/checkers/imports.py:

Start line: 1045, End line: 1079

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
### 6 - pylint/checkers/imports.py:

Start line: 762, End line: 840

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
### 7 - pylint/checkers/stdlib.py:

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
### 8 - pylint/checkers/imports.py:

Start line: 566, End line: 592

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
### 9 - pylint/checkers/imports.py:

Start line: 634, End line: 664

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):

    visit_tryfinally = (
        visit_tryexcept
    ) = (
        visit_assignattr
    ) = (
        visit_assign
    ) = (
        visit_ifexp
    ) = visit_comprehension = visit_expr = visit_if = compute_first_non_import_node

    def visit_functiondef(
        self, node: nodes.FunctionDef | nodes.While | nodes.For | nodes.ClassDef
    ) -> None:
        # If it is the first non import instruction of the module, record it.
        if self._first_non_import_node:
            return

        # Check if the node belongs to an `If` or a `Try` block. If they
        # contain imports, skip recording this node.
        if not isinstance(node.parent.scope(), nodes.Module):
            return

        root = node
        while not isinstance(root.parent, nodes.Module):
            root = root.parent

        if isinstance(root, (nodes.If, nodes.TryFinally, nodes.TryExcept)):
            if any(root.nodes_of_class((nodes.Import, nodes.ImportFrom))):
                return

        self._first_non_import_node = node
```
### 10 - pylint/checkers/base/basic_error_checker.py:

Start line: 100, End line: 207

```python
class BasicErrorChecker(_BasicChecker):
    msgs = {
        "E0100": (
            "__init__ method is a generator",
            "init-is-generator",
            "Used when the special class method __init__ is turned into a "
            "generator by a yield in its body.",
        ),
        "E0101": (
            "Explicit return in __init__",
            "return-in-init",
            "Used when the special class method __init__ has an explicit "
            "return value.",
        ),
        "E0102": (
            "%s already defined line %s",
            "function-redefined",
            "Used when a function / class / method is redefined.",
        ),
        "E0103": (
            "%r not properly in loop",
            "not-in-loop",
            "Used when break or continue keywords are used outside a loop.",
        ),
        "E0104": (
            "Return outside function",
            "return-outside-function",
            'Used when a "return" statement is found outside a function or method.',
        ),
        "E0105": (
            "Yield outside function",
            "yield-outside-function",
            'Used when a "yield" statement is found outside a function or method.',
        ),
        "E0106": (
            "Return with argument inside generator",
            "return-arg-in-generator",
            'Used when a "return" statement with an argument is found '
            "outside in a generator function or method (e.g. with some "
            '"yield" statements).',
            {"maxversion": (3, 3)},
        ),
        "E0107": (
            "Use of the non-existent %s operator",
            "nonexistent-operator",
            "Used when you attempt to use the C-style pre-increment or "
            "pre-decrement operator -- and ++, which doesn't exist in Python.",
        ),
        "E0108": (
            "Duplicate argument name %s in function definition",
            "duplicate-argument-name",
            "Duplicate argument names in function definitions are syntax errors.",
        ),
        "E0110": (
            "Abstract class %r with abstract methods instantiated",
            "abstract-class-instantiated",
            "Used when an abstract class with `abc.ABCMeta` as metaclass "
            "has abstract methods and is instantiated.",
        ),
        "W0120": (
            "Else clause on loop without a break statement, remove the else and"
            " de-indent all the code inside it",
            "useless-else-on-loop",
            "Loops should only have an else clause if they can exit early "
            "with a break statement, otherwise the statements under else "
            "should be on the same scope as the loop itself.",
        ),
        "E0112": (
            "More than one starred expression in assignment",
            "too-many-star-expressions",
            "Emitted when there are more than one starred "
            "expressions (`*x`) in an assignment. This is a SyntaxError.",
        ),
        "E0113": (
            "Starred assignment target must be in a list or tuple",
            "invalid-star-assignment-target",
            "Emitted when a star expression is used as a starred assignment target.",
        ),
        "E0114": (
            "Can use starred expression only in assignment target",
            "star-needs-assignment-target",
            "Emitted when a star expression is not used in an assignment target.",
        ),
        "E0115": (
            "Name %r is nonlocal and global",
            "nonlocal-and-global",
            "Emitted when a name is both nonlocal and global.",
        ),
        "E0116": (
            "'continue' not supported inside 'finally' clause",
            "continue-in-finally",
            "Emitted when the `continue` keyword is found "
            "inside a finally clause, which is a SyntaxError.",
        ),
        "E0117": (
            "nonlocal name %s found without binding",
            "nonlocal-without-binding",
            "Emitted when a nonlocal variable does not have an attached "
            "name somewhere in the parent scopes",
        ),
        "E0118": (
            "Name %r is used prior to global declaration",
            "used-prior-global-declaration",
            "Emitted when a name is used prior a global declaration, "
            "which results in an error since Python 3.6.",
            {"minversion": (3, 6)},
        ),
    }
```
### 35 - pylint/checkers/variables.py:

Start line: 3029, End line: 3109

```python
class VariablesChecker(BaseChecker):
    def _check_imports(self, not_consumed: dict[str, list[nodes.NodeNG]]) -> None:
        local_names = _fix_dot_imports(not_consumed)
        checked = set()
        unused_wildcard_imports: defaultdict[
            tuple[str, nodes.ImportFrom], list[str]
        ] = collections.defaultdict(list)
        for name, stmt in local_names:
            for imports in stmt.names:
                real_name = imported_name = imports[0]
                if imported_name == "*":
                    real_name = name
                as_name = imports[1]
                if real_name in checked:
                    continue
                if name not in (real_name, as_name):
                    continue
                checked.add(real_name)

                is_type_annotation_import = (
                    imported_name in self._type_annotation_names
                    or as_name in self._type_annotation_names
                )
                if isinstance(stmt, nodes.Import) or (
                    isinstance(stmt, nodes.ImportFrom) and not stmt.modname
                ):
                    if isinstance(stmt, nodes.ImportFrom) and SPECIAL_OBJ.search(
                        imported_name
                    ):
                        # Filter special objects (__doc__, __all__) etc.,
                        # because they can be imported for exporting.
                        continue

                    if is_type_annotation_import:
                        # Most likely a typing import if it wasn't used so far.
                        continue

                    if as_name == "_":
                        continue
                    if as_name is None:
                        msg = f"import {imported_name}"
                    else:
                        msg = f"{imported_name} imported as {as_name}"
                    if not in_type_checking_block(stmt):
                        self.add_message("unused-import", args=msg, node=stmt)
                elif isinstance(stmt, nodes.ImportFrom) and stmt.modname != FUTURE:
                    if SPECIAL_OBJ.search(imported_name):
                        # Filter special objects (__doc__, __all__) etc.,
                        # because they can be imported for exporting.
                        continue

                    if _is_from_future_import(stmt, name):
                        # Check if the name is in fact loaded from a
                        # __future__ import in another module.
                        continue

                    if is_type_annotation_import:
                        # Most likely a typing import if it wasn't used so far.
                        continue

                    if imported_name == "*":
                        unused_wildcard_imports[(stmt.modname, stmt)].append(name)
                    else:
                        if as_name is None:
                            msg = f"{imported_name} imported from {stmt.modname}"
                        else:
                            msg = f"{imported_name} imported from {stmt.modname} as {as_name}"
                        if not in_type_checking_block(stmt):
                            self.add_message("unused-import", args=msg, node=stmt)

        # Construct string for unused-wildcard-import message
        for module, unused_list in unused_wildcard_imports.items():
            if len(unused_list) == 1:
                arg_string = unused_list[0]
            else:
                arg_string = (
                    f"{', '.join(i for i in unused_list[:-1])} and {unused_list[-1]}"
                )
            self.add_message(
                "unused-wildcard-import", args=(arg_string, module[0]), node=module[1]
            )
        del self._to_consume
```
### 37 - pylint/checkers/variables.py:

Start line: 1900, End line: 1923

```python
class VariablesChecker(BaseChecker):

    @utils.only_required_for_messages("no-name-in-module")
    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """Check modules attribute accesses."""
        if not self._analyse_fallback_blocks and utils.is_from_fallback_block(node):
            # No need to verify this, since ImportError is already
            # handled by the client code.
            return
        if utils.is_node_in_guarded_import_block(node) is True:
            # Don't verify import if part of guarded import block
            # I.e. `sys.version_info` or `typing.TYPE_CHECKING`
            return

        name_parts = node.modname.split(".")
        try:
            module = node.do_import_module(name_parts[0])
        except astroid.AstroidBuildingException:
            return
        module = self._check_module_attrs(node, module, name_parts[1:])
        if not module:
            return
        for name, _ in node.names:
            if name == "*":
                continue
            self._check_module_attrs(node, module, name.split("."))
```
### 38 - pylint/checkers/variables.py:

Start line: 1878, End line: 1898

```python
class VariablesChecker(BaseChecker):

    @utils.only_required_for_messages("no-name-in-module")
    def visit_import(self, node: nodes.Import) -> None:
        """Check modules attribute accesses."""
        if not self._analyse_fallback_blocks and utils.is_from_fallback_block(node):
            # No need to verify this, since ImportError is already
            # handled by the client code.
            return
        if utils.is_node_in_guarded_import_block(node) is True:
            # Don't verify import if part of guarded import block
            # I.e. `sys.version_info` or `typing.TYPE_CHECKING`
            return

        for name, _ in node.names:
            parts = name.split(".")
            try:
                module = next(_infer_name_module(node, parts[0]))
            except astroid.ResolveError:
                continue
            if not isinstance(module, nodes.Module):
                continue
            self._check_module_attrs(node, module, parts[1:])
```
### 76 - pylint/checkers/variables.py:

Start line: 2931, End line: 2966

```python
class VariablesChecker(BaseChecker):

    def _check_module_attrs(
        self,
        node: _base_nodes.ImportNode,
        module: nodes.Module,
        module_names: list[str],
    ) -> nodes.Module | None:
        """Check that module_names (list of string) are accessible through the
        given module, if the latest access name corresponds to a module, return it.
        """
        while module_names:
            name = module_names.pop(0)
            if name == "__dict__":
                module = None
                break
            try:
                module = next(module.getattr(name)[0].infer())
                if module is astroid.Uninferable:
                    return None
            except astroid.NotFoundError:
                if module.name in self._ignored_modules:
                    return None
                self.add_message(
                    "no-name-in-module", args=(name, module.name), node=node
                )
                return None
            except astroid.InferenceError:
                return None
        if module_names:
            modname = module.name if module else "__dict__"
            self.add_message(
                "no-name-in-module", node=node, args=(".".join(module_names), modname)
            )
            return None
        if isinstance(module, nodes.Module):
            return module
        return None
```
### 79 - pylint/checkers/variables.py:

Start line: 1472, End line: 1542

```python
class VariablesChecker(BaseChecker):

    @utils.only_required_for_messages(
        "global-variable-undefined",
        "global-variable-not-assigned",
        "global-statement",
        "global-at-module-level",
        "redefined-builtin",
    )
    def visit_global(self, node: nodes.Global) -> None:
        """Check names imported exists in the global scope."""
        frame = node.frame(future=True)
        if isinstance(frame, nodes.Module):
            self.add_message("global-at-module-level", node=node, confidence=HIGH)
            return

        module = frame.root()
        default_message = True
        locals_ = node.scope().locals
        for name in node.names:
            try:
                assign_nodes = module.getattr(name)
            except astroid.NotFoundError:
                # unassigned global, skip
                assign_nodes = []

            not_defined_locally_by_import = not any(
                isinstance(local, (nodes.Import, nodes.ImportFrom))
                for local in locals_.get(name, ())
            )
            if (
                not utils.is_reassigned_after_current(node, name)
                and not utils.is_deleted_after_current(node, name)
                and not_defined_locally_by_import
            ):
                self.add_message(
                    "global-variable-not-assigned",
                    args=name,
                    node=node,
                    confidence=HIGH,
                )
                default_message = False
                continue

            for anode in assign_nodes:
                if (
                    isinstance(anode, nodes.AssignName)
                    and anode.name in module.special_attributes
                ):
                    self.add_message("redefined-builtin", args=name, node=node)
                    break
                if anode.frame(future=True) is module:
                    # module level assignment
                    break
                if (
                    isinstance(anode, (nodes.ClassDef, nodes.FunctionDef))
                    and anode.parent is module
                ):
                    # module level function assignment
                    break
            else:
                if not_defined_locally_by_import:
                    # global undefined at the module scope
                    self.add_message(
                        "global-variable-undefined",
                        args=name,
                        node=node,
                        confidence=HIGH,
                    )
                    default_message = False

        if default_message:
            self.add_message("global-statement", node=node, confidence=HIGH)
```
### 90 - pylint/checkers/variables.py:

Start line: 269, End line: 320

```python
def _infer_name_module(
    node: nodes.Import, name: str
) -> Generator[InferenceResult, None, None]:
    context = astroid.context.InferenceContext()
    context.lookupname = name
    return node.infer(context, asname=False)  # type: ignore[no-any-return]


def _fix_dot_imports(
    not_consumed: dict[str, list[nodes.NodeNG]]
) -> list[tuple[str, _base_nodes.ImportNode]]:
    """Try to fix imports with multiple dots, by returning a dictionary
    with the import names expanded.

    The function unflattens root imports,
    like 'xml' (when we have both 'xml.etree' and 'xml.sax'), to 'xml.etree'
    and 'xml.sax' respectively.
    """
    names: dict[str, _base_nodes.ImportNode] = {}
    for name, stmts in not_consumed.items():
        if any(
            isinstance(stmt, nodes.AssignName)
            and isinstance(stmt.assign_type(), nodes.AugAssign)
            for stmt in stmts
        ):
            continue
        for stmt in stmts:
            if not isinstance(stmt, (nodes.ImportFrom, nodes.Import)):
                continue
            for imports in stmt.names:
                second_name = None
                import_module_name = imports[0]
                if import_module_name == "*":
                    # In case of wildcard imports,
                    # pick the name from inside the imported module.
                    second_name = name
                else:
                    name_matches_dotted_import = False
                    if (
                        import_module_name.startswith(name)
                        and import_module_name.find(".") > -1
                    ):
                        name_matches_dotted_import = True

                    if name_matches_dotted_import or name in imports:
                        # Most likely something like 'xml.etree',
                        # which will appear in the .locals as 'xml'.
                        # Only pick the name if it wasn't consumed.
                        second_name = import_module_name
                if second_name and second_name not in names:
                    names[second_name] = stmt
    return sorted(names.items(), key=lambda a: a[1].fromlineno)  # type: ignore[no-any-return]
```
### 140 - pylint/checkers/variables.py:

Start line: 393, End line: 537

```python
MSGS: dict[str, MessageDefinitionTuple] = {
    "E0601": (
        "Using variable %r before assignment",
        "used-before-assignment",
        "Emitted when a local variable is accessed before its assignment took place. "
        "Assignments in try blocks are assumed not to have occurred when evaluating "
        "associated except/finally blocks. Assignments in except blocks are assumed "
        "not to have occurred when evaluating statements outside the block, except "
        "when the associated try block contains a return statement.",
    ),
    "E0602": (
        "Undefined variable %r",
        "undefined-variable",
        "Used when an undefined variable is accessed.",
    ),
    "E0603": (
        "Undefined variable name %r in __all__",
        "undefined-all-variable",
        "Used when an undefined variable name is referenced in __all__.",
    ),
    "E0604": (
        "Invalid object %r in __all__, must contain only strings",
        "invalid-all-object",
        "Used when an invalid (non-string) object occurs in __all__.",
    ),
    "E0605": (
        "Invalid format for __all__, must be tuple or list",
        "invalid-all-format",
        "Used when __all__ has an invalid format.",
    ),
    "E0611": (
        "No name %r in module %r",
        "no-name-in-module",
        "Used when a name cannot be found in a module.",
    ),
    "W0601": (
        "Global variable %r undefined at the module level",
        "global-variable-undefined",
        'Used when a variable is defined through the "global" statement '
        "but the variable is not defined in the module scope.",
    ),
    "W0602": (
        "Using global for %r but no assignment is done",
        "global-variable-not-assigned",
        "When a variable defined in the global scope is modified in an inner scope, "
        "the 'global' keyword is required in the inner scope only if there is an "
        "assignment operation done in the inner scope.",
    ),
    "W0603": (
        "Using the global statement",  # W0121
        "global-statement",
        'Used when you use the "global" statement to update a global '
        "variable. Pylint just try to discourage this "
        "usage. That doesn't mean you cannot use it !",
    ),
    "W0604": (
        "Using the global statement at the module level",  # W0103
        "global-at-module-level",
        'Used when you use the "global" statement at the module level '
        "since it has no effect",
    ),
    "W0611": (
        "Unused %s",
        "unused-import",
        "Used when an imported module or variable is not used.",
    ),
    "W0612": (
        "Unused variable %r",
        "unused-variable",
        "Used when a variable is defined but not used.",
    ),
    "W0613": (
        "Unused argument %r",
        "unused-argument",
        "Used when a function or method argument is not used.",
    ),
    "W0614": (
        "Unused import(s) %s from wildcard import of %s",
        "unused-wildcard-import",
        "Used when an imported module or variable is not used from a "
        "`'from X import *'` style import.",
    ),
    "W0621": (
        "Redefining name %r from outer scope (line %s)",
        "redefined-outer-name",
        "Used when a variable's name hides a name defined in an outer scope or except handler.",
    ),
    "W0622": (
        "Redefining built-in %r",
        "redefined-builtin",
        "Used when a variable or function override a built-in.",
    ),
    "W0631": (
        "Using possibly undefined loop variable %r",
        "undefined-loop-variable",
        "Used when a loop variable (i.e. defined by a for loop or "
        "a list comprehension or a generator expression) is used outside "
        "the loop.",
    ),
    "W0632": (
        "Possible unbalanced tuple unpacking with sequence %s: left side has %d "
        "label%s, right side has %d value%s",
        "unbalanced-tuple-unpacking",
        "Used when there is an unbalanced tuple unpacking in assignment",
        {"old_names": [("E0632", "old-unbalanced-tuple-unpacking")]},
    ),
    "E0633": (
        "Attempting to unpack a non-sequence%s",
        "unpacking-non-sequence",
        "Used when something which is not a sequence is used in an unpack assignment",
        {"old_names": [("W0633", "old-unpacking-non-sequence")]},
    ),
    "W0640": (
        "Cell variable %s defined in loop",
        "cell-var-from-loop",
        "A variable used in a closure is defined in a loop. "
        "This will result in all closures using the same value for "
        "the closed-over variable.",
    ),
    "W0641": (
        "Possibly unused variable %r",
        "possibly-unused-variable",
        "Used when a variable is defined but might not be used. "
        "The possibility comes from the fact that locals() might be used, "
        "which could consume or not the said variable",
    ),
    "W0642": (
        "Invalid assignment to %s in method",
        "self-cls-assignment",
        "Invalid assignment to self or cls in instance or class method "
        "respectively.",
    ),
    "E0643": (
        "Invalid index for iterable length",
        "potential-index-error",
        "Emitted when an index used on an iterable goes beyond the length of that "
        "iterable.",
    ),
    "W0644": (
        "Possible unbalanced dict unpacking with %s: "
        "left side has %d label%s, right side has %d value%s",
        "unbalanced-dict-unpacking",
        "Used when there is an unbalanced dict unpacking in assignment or for loop",
    ),
}
```
### 162 - pylint/checkers/variables.py:

Start line: 323, End line: 342

```python
def _find_frame_imports(name: str, frame: nodes.LocalsDictNodeNG) -> bool:
    """Detect imports in the frame, with the required *name*.

    Such imports can be considered assignments if they are not globals.
    Returns True if an import for the given name was found.
    """
    if name in _flattened_scope_names(frame.nodes_of_class(nodes.Global)):
        return False

    imports = frame.nodes_of_class((nodes.Import, nodes.ImportFrom))
    for import_node in imports:
        for import_name, import_alias in import_node.names:
            # If the import uses an alias, check only that.
            # Otherwise, check only the import name.
            if import_alias:
                if import_alias == name:
                    return True
            elif import_name and import_name == name:
                return True
    return False
```
### 163 - pylint/checkers/variables.py:

Start line: 2968, End line: 3028

```python
class VariablesChecker(BaseChecker):

    def _check_all(
        self, node: nodes.Module, not_consumed: dict[str, list[nodes.NodeNG]]
    ) -> None:
        assigned = next(node.igetattr("__all__"))
        if assigned is astroid.Uninferable:
            return
        if not assigned.pytype() in {"builtins.list", "builtins.tuple"}:
            line, col = assigned.tolineno, assigned.col_offset
            self.add_message("invalid-all-format", line=line, col_offset=col, node=node)
            return
        for elt in getattr(assigned, "elts", ()):
            try:
                elt_name = next(elt.infer())
            except astroid.InferenceError:
                continue
            if elt_name is astroid.Uninferable:
                continue
            if not elt_name.parent:
                continue

            if not isinstance(elt_name, nodes.Const) or not isinstance(
                elt_name.value, str
            ):
                self.add_message("invalid-all-object", args=elt.as_string(), node=elt)
                continue

            elt_name = elt_name.value
            # If elt is in not_consumed, remove it from not_consumed
            if elt_name in not_consumed:
                del not_consumed[elt_name]
                continue

            if elt_name not in node.locals:
                if not node.package:
                    self.add_message(
                        "undefined-all-variable", args=(elt_name,), node=elt
                    )
                else:
                    basename = os.path.splitext(node.file)[0]
                    if os.path.basename(basename) == "__init__":
                        name = node.name + "." + elt_name
                        try:
                            astroid.modutils.file_from_modpath(name.split("."))
                        except ImportError:
                            self.add_message(
                                "undefined-all-variable", args=(elt_name,), node=elt
                            )
                        except SyntaxError:
                            # don't yield a syntax-error warning,
                            # because it will be later yielded
                            # when the file will be checked
                            pass

    def _check_globals(self, not_consumed: dict[str, nodes.NodeNG]) -> None:
        if self._allow_global_unused_variables:
            return
        for name, node_lst in not_consumed.items():
            for node in node_lst:
                self.add_message("unused-variable", args=(name,), node=node)

    # pylint: disable = too-many-branches
```
### 164 - pylint/checkers/variables.py:

Start line: 2562, End line: 2654

```python
class VariablesChecker(BaseChecker):

    # pylint: disable = too-many-branches
    def _check_is_unused(
        self,
        name: str,
        node: nodes.FunctionDef,
        stmt: nodes.NodeNG,
        global_names: set[str],
        nonlocal_names: Iterable[str],
        comprehension_target_names: Iterable[str],
    ) -> None:
        # Ignore some special names specified by user configuration.
        if self._is_name_ignored(stmt, name):
            return
        # Ignore names that were added dynamically to the Function scope
        if (
            isinstance(node, nodes.FunctionDef)
            and name == "__class__"
            and len(node.locals["__class__"]) == 1
            and isinstance(node.locals["__class__"][0], nodes.ClassDef)
        ):
            return

        # Ignore names imported by the global statement.
        if isinstance(stmt, (nodes.Global, nodes.Import, nodes.ImportFrom)):
            # Detect imports, assigned to global statements.
            if global_names and _import_name_is_global(stmt, global_names):
                return

        # Ignore names in comprehension targets
        if name in comprehension_target_names:
            return

        # Ignore names in string literal type annotation.
        if name in self._type_annotation_names:
            return

        argnames = node.argnames()
        # Care about functions with unknown argument (builtins)
        if name in argnames:
            self._check_unused_arguments(name, node, stmt, argnames, nonlocal_names)
        else:
            if stmt.parent and isinstance(
                stmt.parent, (nodes.Assign, nodes.AnnAssign, nodes.Tuple, nodes.For)
            ):
                if name in nonlocal_names:
                    return

            qname = asname = None
            if isinstance(stmt, (nodes.Import, nodes.ImportFrom)):
                # Need the complete name, which we don't have in .locals.
                if len(stmt.names) > 1:
                    import_names = next(
                        (names for names in stmt.names if name in names), None
                    )
                else:
                    import_names = stmt.names[0]
                if import_names:
                    qname, asname = import_names
                    name = asname or qname

            if _has_locals_call_after_node(stmt, node.scope()):
                message_name = "possibly-unused-variable"
            else:
                if isinstance(stmt, nodes.Import):
                    if asname is not None:
                        msg = f"{qname} imported as {asname}"
                    else:
                        msg = f"import {name}"
                    self.add_message("unused-import", args=msg, node=stmt)
                    return
                if isinstance(stmt, nodes.ImportFrom):
                    if asname is not None:
                        msg = f"{qname} imported from {stmt.modname} as {asname}"
                    else:
                        msg = f"{name} imported from {stmt.modname}"
                    self.add_message("unused-import", args=msg, node=stmt)
                    return
                message_name = "unused-variable"

            if isinstance(stmt, nodes.FunctionDef) and stmt.decorators:
                return

            # Don't check function stubs created only for type information
            if utils.is_overload_stub(node):
                return

            # Special case for exception variable
            if isinstance(stmt.parent, nodes.ExceptHandler) and any(
                n.name == name for n in stmt.parent.nodes_of_class(nodes.Name)
            ):
                return

            self.add_message(message_name, args=name, node=stmt)
```
