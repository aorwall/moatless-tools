# pylint-dev__pylint-6059

| **pylint-dev/pylint** | `789a3818fec81754cf95bef2a0b591678142c227` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 6 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/checkers/base_checker.py b/pylint/checkers/base_checker.py
--- a/pylint/checkers/base_checker.py
+++ b/pylint/checkers/base_checker.py
@@ -61,7 +61,15 @@ def __init__(
 
     def __gt__(self, other):
         """Permit to sort a list of Checker by name."""
-        return f"{self.name}{self.msgs}" > (f"{other.name}{other.msgs}")
+        return f"{self.name}{self.msgs}" > f"{other.name}{other.msgs}"
+
+    def __eq__(self, other):
+        """Permit to assert Checkers are equal."""
+        return f"{self.name}{self.msgs}" == f"{other.name}{other.msgs}"
+
+    def __hash__(self):
+        """Make Checker hashable."""
+        return hash(f"{self.name}{self.msgs}")
 
     def __repr__(self):
         status = "Checker" if self.enabled else "Disabled checker"

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/checkers/base_checker.py | 64 | 64 | - | 6 | -


## Problem Statement

```
Is `BaseChecker.__gt__` required
### Bug description

As noted by @DanielNoord [here](https://github.com/PyCQA/pylint/pull/5938#discussion_r837867526), [`BaseCheck.__gt__`](https://github.com/PyCQA/pylint/blob/742e60dc07077cdd3338dffc3bb809cd4c27085f/pylint/checkers/base_checker.py#L62-L64) is not currently covered. If this required then we should add a unit test, otherwise we can remove this method.

### Configuration

\`\`\`ini
N/A
\`\`\`


### Command used

\`\`\`shell
N/A
\`\`\`


### Pylint output

\`\`\`shell
N/A
\`\`\`


### Expected behavior

N/A

### Pylint version

\`\`\`shell
N/A
\`\`\`


### OS / Environment

_No response_

### Additional dependencies

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/checkers/base/basic_checker.py | 105 | 249| 1332 | 1332 | 7057 | 
| 2 | 1 pylint/checkers/base/basic_checker.py | 278 | 339| 477 | 1809 | 7057 | 
| 3 | 2 pylint/checkers/base/comparison_checker.py | 24 | 74| 464 | 2273 | 9584 | 
| 4 | 3 pylint/checkers/imports.py | 259 | 385| 801 | 3074 | 17184 | 
| 5 | 3 pylint/checkers/base/comparison_checker.py | 215 | 236| 235 | 3309 | 17184 | 
| 6 | 3 pylint/checkers/base/comparison_checker.py | 238 | 267| 257 | 3566 | 17184 | 
| 7 | 3 pylint/checkers/base/comparison_checker.py | 175 | 186| 143 | 3709 | 17184 | 
| 8 | 4 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 78| 516 | 4225 | 18991 | 
| 9 | 4 pylint/checkers/base/basic_checker.py | 567 | 586| 165 | 4390 | 18991 | 
| 10 | 5 pylint/checkers/base/basic_error_checker.py | 297 | 324| 173 | 4563 | 23511 | 
| 11 | 5 pylint/checkers/base/basic_error_checker.py | 464 | 478| 164 | 4727 | 23511 | 
| 12 | **6 pylint/checkers/base_checker.py** | 5 | 23| 127 | 4854 | 25239 | 
| 13 | 7 pylint/checkers/misc.py | 54 | 107| 321 | 5175 | 26530 | 
| 14 | 7 pylint/checkers/base/basic_error_checker.py | 394 | 412| 141 | 5316 | 26530 | 
| 15 | 7 pylint/checkers/base/basic_checker.py | 608 | 623| 157 | 5473 | 26530 | 
| 16 | 8 pylint/checkers/base/pass_checker.py | 5 | 29| 160 | 5633 | 26755 | 
| 17 | 9 pylint/checkers/base/docstring_checker.py | 46 | 114| 534 | 6167 | 28346 | 
| 18 | 10 pylint/checkers/classes/class_checker.py | 6 | 65| 433 | 6600 | 44797 | 
| 19 | 10 pylint/checkers/base/docstring_checker.py | 7 | 43| 203 | 6803 | 44797 | 
| 20 | 11 pylint/checkers/base/__init__.py | 6 | 46| 275 | 7078 | 45137 | 
| 21 | 12 pylint/extensions/comparison_placement.py | 57 | 72| 155 | 7233 | 45711 | 
| 22 | 12 pylint/checkers/base/comparison_checker.py | 134 | 173| 319 | 7552 | 45711 | 
| 23 | 13 pylint/testutils/checker_test_case.py | 5 | 33| 207 | 7759 | 46576 | 
| 24 | 14 pylint/checkers/variables.py | 971 | 1076| 676 | 8435 | 69094 | 
| 25 | 15 pylint/checkers/base/name_checker/checker.py | 326 | 357| 265 | 8700 | 73899 | 
| 26 | 16 pylint/checkers/base/name_checker/__init__.py | 5 | 26| 116 | 8816 | 74080 | 
| 27 | 16 pylint/checkers/base/docstring_checker.py | 116 | 151| 306 | 9122 | 74080 | 
| 28 | 16 pylint/checkers/base/basic_error_checker.py | 251 | 295| 406 | 9528 | 74080 | 
| 29 | 16 pylint/checkers/base/name_checker/checker.py | 125 | 159| 317 | 9845 | 74080 | 
| 30 | 16 pylint/checkers/base/basic_checker.py | 648 | 662| 139 | 9984 | 74080 | 
| 31 | 16 pylint/checkers/base/basic_error_checker.py | 207 | 218| 120 | 10104 | 74080 | 
| 32 | 17 pylint/checkers/stdlib.py | 441 | 491| 539 | 10643 | 80464 | 
| 33 | 18 pylint/checkers/modified_iterating_checker.py | 129 | 157| 253 | 10896 | 81766 | 
| 34 | 18 pylint/checkers/base/basic_checker.py | 798 | 831| 286 | 11182 | 81766 | 
| 35 | 18 pylint/checkers/base/basic_error_checker.py | 354 | 381| 284 | 11466 | 81766 | 
| 36 | 18 pylint/checkers/classes/class_checker.py | 321 | 341| 132 | 11598 | 81766 | 
| 37 | 19 pylint/checkers/typecheck.py | 887 | 917| 350 | 11948 | 98191 | 
| 38 | 19 pylint/extensions/comparison_placement.py | 10 | 55| 340 | 12288 | 98191 | 
| 39 | 19 pylint/checkers/classes/class_checker.py | 1404 | 1430| 227 | 12515 | 98191 | 
| 40 | 19 pylint/checkers/typecheck.py | 1155 | 1172| 135 | 12650 | 98191 | 
| 41 | 19 pylint/checkers/classes/class_checker.py | 809 | 819| 117 | 12767 | 98191 | 
| 42 | 19 pylint/checkers/base/basic_checker.py | 387 | 411| 160 | 12927 | 98191 | 
| 43 | 19 pylint/checkers/base/basic_error_checker.py | 383 | 392| 120 | 13047 | 98191 | 
| 44 | 19 pylint/checkers/base/name_checker/checker.py | 250 | 264| 147 | 13194 | 98191 | 
| 45 | 19 pylint/checkers/base/name_checker/checker.py | 161 | 248| 618 | 13812 | 98191 | 
| 46 | 19 pylint/checkers/base/basic_checker.py | 341 | 385| 435 | 14247 | 98191 | 
| 47 | 19 pylint/checkers/base/basic_checker.py | 7 | 61| 344 | 14591 | 98191 | 
| 48 | 19 pylint/checkers/base/basic_error_checker.py | 326 | 352| 164 | 14755 | 98191 | 
| 49 | 19 pylint/checkers/base/comparison_checker.py | 7 | 21| 146 | 14901 | 98191 | 
| 50 | 19 pylint/checkers/stdlib.py | 323 | 439| 1195 | 16096 | 98191 | 
| 51 | 20 pylint/checkers/refactoring/not_checker.py | 44 | 84| 317 | 16413 | 98857 | 
| 52 | 20 pylint/checkers/base/basic_error_checker.py | 427 | 462| 270 | 16683 | 98857 | 
| 53 | 20 pylint/checkers/base/name_checker/checker.py | 292 | 316| 243 | 16926 | 98857 | 
| 54 | 20 pylint/checkers/refactoring/implicit_booleaness_checker.py | 80 | 119| 395 | 17321 | 98857 | 
| 55 | 21 pylint/checkers/classes/special_methods_checker.py | 315 | 361| 454 | 17775 | 101896 | 
| 56 | 21 pylint/checkers/stdlib.py | 7 | 50| 461 | 18236 | 101896 | 
| 57 | 22 pylint/extensions/eq_without_hash.py | 12 | 42| 243 | 18479 | 102264 | 
| 58 | 22 pylint/checkers/base/comparison_checker.py | 269 | 296| 264 | 18743 | 102264 | 
| 59 | 22 pylint/checkers/classes/class_checker.py | 1901 | 1935| 289 | 19032 | 102264 | 
| 60 | 22 pylint/checkers/classes/special_methods_checker.py | 46 | 135| 763 | 19795 | 102264 | 
| 61 | 23 pylint/checkers/design_analysis.py | 94 | 172| 510 | 20305 | 107058 | 
| 62 | 24 pylint/extensions/bad_builtin.py | 23 | 49| 177 | 20482 | 107577 | 
| 63 | 24 pylint/checkers/base/basic_error_checker.py | 7 | 22| 128 | 20610 | 107577 | 
| 64 | 24 pylint/checkers/base/basic_error_checker.py | 97 | 205| 952 | 21562 | 107577 | 
| 65 | 25 pylint/checkers/ellipsis_checker.py | 6 | 30| 167 | 21729 | 108062 | 
| 66 | 25 pylint/checkers/classes/class_checker.py | 1284 | 1337| 416 | 22145 | 108062 | 
| 67 | 25 pylint/checkers/classes/special_methods_checker.py | 137 | 152| 167 | 22312 | 108062 | 
| 68 | 25 pylint/checkers/classes/class_checker.py | 821 | 842| 162 | 22474 | 108062 | 
| 69 | 25 pylint/checkers/classes/class_checker.py | 1009 | 1078| 612 | 23086 | 108062 | 
| 70 | 25 pylint/checkers/base/name_checker/checker.py | 318 | 324| 113 | 23199 | 108062 | 
| 71 | 25 pylint/checkers/base/docstring_checker.py | 153 | 210| 488 | 23687 | 108062 | 
| 72 | 26 pylint/checkers/dunder_methods.py | 5 | 28| 191 | 23878 | 109606 | 
| 73 | 26 pylint/extensions/bad_builtin.py | 51 | 66| 170 | 24048 | 109606 | 
| 74 | 27 pylint/extensions/comparetozero.py | 47 | 78| 310 | 24358 | 110202 | 
| 75 | 27 pylint/checkers/refactoring/not_checker.py | 5 | 42| 292 | 24650 | 110202 | 
| 76 | 27 pylint/checkers/stdlib.py | 602 | 625| 192 | 24842 | 110202 | 
| 77 | 27 pylint/checkers/base/comparison_checker.py | 76 | 132| 459 | 25301 | 110202 | 
| 78 | 27 pylint/checkers/classes/class_checker.py | 1339 | 1369| 263 | 25564 | 110202 | 
| 79 | 27 pylint/checkers/typecheck.py | 735 | 885| 1094 | 26658 | 110202 | 
| 80 | 27 pylint/checkers/stdlib.py | 546 | 567| 218 | 26876 | 110202 | 
| 81 | 28 pylint/checkers/newstyle.py | 6 | 25| 122 | 26998 | 111087 | 
| 82 | 28 pylint/checkers/typecheck.py | 1818 | 1867| 439 | 27437 | 111087 | 
| 83 | 28 pylint/checkers/dunder_methods.py | 30 | 123| 1095 | 28532 | 111087 | 
| 84 | 29 pylint/checkers/refactoring/refactoring_checker.py | 449 | 501| 382 | 28914 | 128337 | 
| 85 | 29 pylint/checkers/modified_iterating_checker.py | 90 | 114| 206 | 29120 | 128337 | 
| 86 | 30 pylint/checkers/non_ascii_names.py | 173 | 202| 293 | 29413 | 130149 | 
| 87 | 30 pylint/checkers/classes/class_checker.py | 683 | 780| 629 | 30042 | 130149 | 
| 88 | 30 pylint/checkers/refactoring/implicit_booleaness_checker.py | 121 | 144| 232 | 30274 | 130149 | 
| 89 | 30 pylint/checkers/typecheck.py | 1699 | 1762| 524 | 30798 | 130149 | 
| 90 | 31 pylint/extensions/emptystring.py | 42 | 73| 313 | 31111 | 130723 | 
| 91 | 32 pylint/extensions/code_style.py | 5 | 21| 113 | 31224 | 133297 | 
| 92 | 32 pylint/checkers/classes/special_methods_checker.py | 363 | 392| 211 | 31435 | 133297 | 
| 93 | 33 pylint/constants.py | 90 | 187| 1251 | 32686 | 135344 | 
| 94 | 33 pylint/checkers/base/basic_checker.py | 413 | 471| 534 | 33220 | 135344 | 
| 95 | 34 pylint/extensions/empty_comment.py | 41 | 68| 182 | 33402 | 135831 | 
| 96 | 34 pylint/checkers/base/name_checker/checker.py | 266 | 290| 240 | 33642 | 135831 | 
| 97 | 34 pylint/checkers/base/basic_checker.py | 757 | 796| 315 | 33957 | 135831 | 
| 98 | 34 pylint/checkers/base/basic_checker.py | 741 | 755| 182 | 34139 | 135831 | 
| 99 | 34 pylint/checkers/modified_iterating_checker.py | 72 | 88| 155 | 34294 | 135831 | 
| 100 | 34 pylint/checkers/base/basic_checker.py | 684 | 739| 430 | 34724 | 135831 | 
| 101 | 35 pylint/checkers/exceptions.py | 227 | 251| 159 | 34883 | 140321 | 
| 102 | 35 pylint/checkers/base/basic_error_checker.py | 414 | 425| 124 | 35007 | 140321 | 
| 103 | 35 pylint/checkers/refactoring/refactoring_checker.py | 1368 | 1388| 168 | 35175 | 140321 | 
| 104 | 35 pylint/extensions/bad_builtin.py | 6 | 20| 107 | 35282 | 140321 | 
| 105 | 35 pylint/checkers/stdlib.py | 695 | 732| 273 | 35555 | 140321 | 
| 106 | 35 pylint/checkers/imports.py | 387 | 399| 167 | 35722 | 140321 | 
| 107 | 35 pylint/checkers/variables.py | 2631 | 2687| 468 | 36190 | 140321 | 
| 108 | 36 pylint/checkers/refactoring/recommendation_checker.py | 5 | 81| 697 | 36887 | 143764 | 
| 109 | 36 pylint/checkers/typecheck.py | 1606 | 1643| 338 | 37225 | 143764 | 
| 110 | 36 pylint/checkers/classes/special_methods_checker.py | 285 | 313| 220 | 37445 | 143764 | 
| 111 | 36 pylint/checkers/modified_iterating_checker.py | 116 | 127| 120 | 37565 | 143764 | 
| 112 | 36 pylint/checkers/design_analysis.py | 273 | 410| 821 | 38386 | 143764 | 
| 113 | 36 pylint/checkers/non_ascii_names.py | 113 | 146| 286 | 38672 | 143764 | 
| 114 | 36 pylint/checkers/base/basic_error_checker.py | 498 | 573| 585 | 39257 | 143764 | 
| 115 | 36 pylint/checkers/refactoring/refactoring_checker.py | 984 | 1011| 266 | 39523 | 143764 | 
| 116 | 36 pylint/checkers/refactoring/refactoring_checker.py | 693 | 720| 283 | 39806 | 143764 | 
| 117 | 36 pylint/checkers/stdlib.py | 101 | 245| 1089 | 40895 | 143764 | 
| 118 | 36 pylint/checkers/non_ascii_names.py | 13 | 40| 239 | 41134 | 143764 | 
| 119 | 36 pylint/checkers/typecheck.py | 1869 | 1927| 464 | 41598 | 143764 | 
| 120 | 36 pylint/checkers/base/basic_checker.py | 588 | 606| 204 | 41802 | 143764 | 
| 121 | 36 pylint/checkers/classes/special_methods_checker.py | 235 | 283| 329 | 42131 | 143764 | 
| 122 | 37 pylint/checkers/utils.py | 1237 | 1263| 183 | 42314 | 156444 | 
| 123 | 38 pylint/checkers/refactoring/__init__.py | 8 | 33| 174 | 42488 | 156695 | 
| 124 | 38 pylint/checkers/base/name_checker/checker.py | 517 | 546| 240 | 42728 | 156695 | 
| 125 | 38 pylint/checkers/stdlib.py | 52 | 98| 556 | 43284 | 156695 | 
| 126 | 38 pylint/checkers/typecheck.py | 919 | 957| 344 | 43628 | 156695 | 
| 127 | 39 pylint/utils/utils.py | 203 | 250| 252 | 43880 | 159595 | 
| 128 | 39 pylint/checkers/classes/class_checker.py | 1478 | 1498| 192 | 44072 | 159595 | 
| 129 | 39 pylint/checkers/typecheck.py | 7 | 100| 485 | 44557 | 159595 | 
| 130 | 39 pylint/checkers/design_analysis.py | 544 | 578| 254 | 44811 | 159595 | 
| 131 | 39 pylint/checkers/refactoring/refactoring_checker.py | 768 | 785| 169 | 44980 | 159595 | 
| 132 | 39 pylint/checkers/base/basic_error_checker.py | 480 | 496| 147 | 45127 | 159595 | 
| 133 | 39 pylint/checkers/variables.py | 2689 | 2769| 673 | 45800 | 159595 | 
| 134 | 39 pylint/checkers/typecheck.py | 1389 | 1490| 827 | 46627 | 159595 | 
| 135 | 39 pylint/checkers/dunder_methods.py | 124 | 152| 208 | 46835 | 159595 | 
| 136 | 39 pylint/checkers/base/name_checker/checker.py | 6 | 34| 218 | 47053 | 159595 | 
| 137 | 39 pylint/checkers/classes/class_checker.py | 409 | 660| 166 | 47219 | 159595 | 
| 138 | 39 pylint/checkers/classes/class_checker.py | 879 | 933| 410 | 47629 | 159595 | 
| 139 | 39 pylint/extensions/code_style.py | 24 | 110| 752 | 48381 | 159595 | 
| 140 | 39 pylint/checkers/design_analysis.py | 7 | 93| 724 | 49105 | 159595 | 
| 141 | 39 pylint/checkers/classes/class_checker.py | 863 | 877| 118 | 49223 | 159595 | 
| 142 | 40 pylint/extensions/typing.py | 212 | 227| 183 | 49406 | 163551 | 
| 143 | 40 pylint/checkers/refactoring/refactoring_checker.py | 917 | 927| 118 | 49524 | 163551 | 
| 144 | 40 pylint/checkers/refactoring/refactoring_checker.py | 621 | 640| 153 | 49677 | 163551 | 
| 145 | 41 pylint/extensions/consider_ternary_expression.py | 7 | 58| 326 | 50003 | 163958 | 
| 146 | 41 pylint/checkers/base/basic_error_checker.py | 233 | 249| 172 | 50175 | 163958 | 
| 147 | 42 pylint/extensions/check_elif.py | 5 | 46| 297 | 50472 | 164458 | 
| 148 | 43 pylint/checkers/threading_checker.py | 5 | 43| 229 | 50701 | 164900 | 
| 149 | 43 pylint/checkers/classes/class_checker.py | 1937 | 2018| 667 | 51368 | 164900 | 
| 150 | 43 pylint/extensions/comparetozero.py | 7 | 45| 224 | 51592 | 164900 | 
| 151 | 43 pylint/checkers/base/basic_error_checker.py | 220 | 231| 136 | 51728 | 164900 | 
| 152 | 43 pylint/checkers/typecheck.py | 713 | 732| 157 | 51885 | 164900 | 
| 153 | 43 pylint/checkers/exceptions.py | 452 | 464| 196 | 52081 | 164900 | 
| 154 | 43 pylint/checkers/refactoring/refactoring_checker.py | 1276 | 1289| 158 | 52239 | 164900 | 
| 155 | 43 pylint/extensions/typing.py | 77 | 163| 782 | 53021 | 164900 | 
| 156 | 43 pylint/checkers/typecheck.py | 1956 | 1968| 133 | 53154 | 164900 | 
| 157 | 44 doc/data/messages/r/redundant-unittest-assert/good.py | 1 | 7| 0 | 53154 | 164930 | 
| 158 | 44 pylint/checkers/typecheck.py | 2020 | 2085| 606 | 53760 | 164930 | 
| 159 | 44 pylint/extensions/empty_comment.py | 5 | 24| 127 | 53887 | 164930 | 
| 160 | 44 pylint/checkers/classes/class_checker.py | 958 | 1007| 394 | 54281 | 164930 | 
| 161 | 44 pylint/checkers/base/name_checker/checker.py | 475 | 515| 383 | 54664 | 164930 | 
| 162 | 45 pylint/checkers/format.py | 196 | 307| 655 | 55319 | 171128 | 
| 163 | 45 pylint/extensions/typing.py | 229 | 256| 216 | 55535 | 171128 | 
| 164 | 45 pylint/checkers/refactoring/refactoring_checker.py | 606 | 619| 128 | 55663 | 171128 | 
| 165 | 45 pylint/checkers/exceptions.py | 6 | 25| 113 | 55776 | 171128 | 
| 166 | 45 pylint/checkers/refactoring/refactoring_checker.py | 1013 | 1031| 177 | 55953 | 171128 | 
| 167 | 45 pylint/extensions/check_elif.py | 48 | 62| 147 | 56100 | 171128 | 
| 168 | 46 pylint/extensions/while_used.py | 6 | 37| 182 | 56282 | 171383 | 
| 169 | 46 pylint/checkers/classes/special_methods_checker.py | 154 | 185| 238 | 56520 | 171383 | 
| 170 | 47 pylint/checkers/strings.py | 280 | 379| 946 | 57466 | 179451 | 
| 171 | 48 pylint/lint/pylinter.py | 191 | 541| 304 | 57770 | 193584 | 


### Hint

```
I think this was used originally to be able to assert that  a list of checker is equal to another one in tests. If it's not covered it means we do not do that anymore.
It's used in Sphinx and maybe downstream libraries see #6047 .
Shall we add a no coverage param then?
It's pretty easy to add a unit test for this so will make a quick PR
```

## Patch

```diff
diff --git a/pylint/checkers/base_checker.py b/pylint/checkers/base_checker.py
--- a/pylint/checkers/base_checker.py
+++ b/pylint/checkers/base_checker.py
@@ -61,7 +61,15 @@ def __init__(
 
     def __gt__(self, other):
         """Permit to sort a list of Checker by name."""
-        return f"{self.name}{self.msgs}" > (f"{other.name}{other.msgs}")
+        return f"{self.name}{self.msgs}" > f"{other.name}{other.msgs}"
+
+    def __eq__(self, other):
+        """Permit to assert Checkers are equal."""
+        return f"{self.name}{self.msgs}" == f"{other.name}{other.msgs}"
+
+    def __hash__(self):
+        """Make Checker hashable."""
+        return hash(f"{self.name}{self.msgs}")
 
     def __repr__(self):
         status = "Checker" if self.enabled else "Disabled checker"

```

## Test Patch

```diff
diff --git a/tests/checkers/unittest_base_checker.py b/tests/checkers/unittest_base_checker.py
--- a/tests/checkers/unittest_base_checker.py
+++ b/tests/checkers/unittest_base_checker.py
@@ -33,6 +33,17 @@ class LessBasicChecker(OtherBasicChecker):
     )
 
 
+class DifferentBasicChecker(BaseChecker):
+    name = "different"
+    msgs = {
+        "W0002": (
+            "Blah blah example.",
+            "blah-blah-example",
+            "I only exist to be different to OtherBasicChecker :(",
+        )
+    }
+
+
 def test_base_checker_doc() -> None:
     basic = OtherBasicChecker()
     expected_beginning = """\
@@ -65,3 +76,13 @@ def test_base_checker_doc() -> None:
 
     assert str(less_basic) == expected_beginning + expected_middle + expected_end
     assert repr(less_basic) == repr(basic)
+
+
+def test_base_checker_ordering() -> None:
+    """Test ordering of checkers based on their __gt__ method."""
+    fake_checker_1 = OtherBasicChecker()
+    fake_checker_2 = LessBasicChecker()
+    fake_checker_3 = DifferentBasicChecker()
+    assert fake_checker_1 < fake_checker_3
+    assert fake_checker_2 < fake_checker_3
+    assert fake_checker_1 == fake_checker_2

```


## Code snippets

### 1 - pylint/checkers/base/basic_checker.py:

Start line: 105, End line: 249

```python
class BasicChecker(_BasicChecker):
    """Basic checker.

    Checks for :
    * doc strings
    * number of arguments, local variables, branches, returns and statements in
    functions, methods
    * required module attributes
    * dangerous default values as arguments
    * redefinition of function / method / class
    * uses of the global statement
    """

    __implements__ = interfaces.IAstroidChecker

    name = "basic"
    msgs = {
        "W0101": (
            "Unreachable code",
            "unreachable",
            'Used when there is some code behind a "return" or "raise" '
            "statement, which will never be accessed.",
        ),
        "W0102": (
            "Dangerous default value %s as argument",
            "dangerous-default-value",
            "Used when a mutable value as list or dictionary is detected in "
            "a default value for an argument.",
        ),
        "W0104": (
            "Statement seems to have no effect",
            "pointless-statement",
            "Used when a statement doesn't have (or at least seems to) any effect.",
        ),
        "W0105": (
            "String statement has no effect",
            "pointless-string-statement",
            "Used when a string is used as a statement (which of course "
            "has no effect). This is a particular case of W0104 with its "
            "own message so you can easily disable it if you're using "
            "those strings as documentation, instead of comments.",
        ),
        "W0106": (
            'Expression "%s" is assigned to nothing',
            "expression-not-assigned",
            "Used when an expression that is not a function call is assigned "
            "to nothing. Probably something else was intended.",
        ),
        "W0108": (
            "Lambda may not be necessary",
            "unnecessary-lambda",
            "Used when the body of a lambda expression is a function call "
            "on the same argument list as the lambda itself; such lambda "
            "expressions are in all but a few cases replaceable with the "
            "function being called in the body of the lambda.",
        ),
        "W0109": (
            "Duplicate key %r in dictionary",
            "duplicate-key",
            "Used when a dictionary expression binds the same key multiple times.",
        ),
        "W0122": (
            "Use of exec",
            "exec-used",
            'Used when you use the "exec" statement (function for Python '
            "3), to discourage its usage. That doesn't "
            "mean you cannot use it !",
        ),
        "W0123": (
            "Use of eval",
            "eval-used",
            'Used when you use the "eval" function, to discourage its '
            "usage. Consider using `ast.literal_eval` for safely evaluating "
            "strings containing Python expressions "
            "from untrusted sources.",
        ),
        "W0150": (
            "%s statement in finally block may swallow exception",
            "lost-exception",
            "Used when a break or a return statement is found inside the "
            "finally clause of a try...finally block: the exceptions raised "
            "in the try clause will be silently swallowed instead of being "
            "re-raised.",
        ),
        "W0199": (
            "Assert called on a 2-item-tuple. Did you mean 'assert x,y'?",
            "assert-on-tuple",
            "A call of assert on a tuple will always evaluate to true if "
            "the tuple is not empty, and will always evaluate to false if "
            "it is.",
        ),
        "W0124": (
            'Following "as" with another context manager looks like a tuple.',
            "confusing-with-statement",
            "Emitted when a `with` statement component returns multiple values "
            "and uses name binding with `as` only for a part of those values, "
            "as in with ctx() as a, b. This can be misleading, since it's not "
            "clear if the context manager returns a tuple or if the node without "
            "a name binding is another context manager.",
        ),
        "W0125": (
            "Using a conditional statement with a constant value",
            "using-constant-test",
            "Emitted when a conditional statement (If or ternary if) "
            "uses a constant value for its test. This might not be what "
            "the user intended to do.",
        ),
        "W0126": (
            "Using a conditional statement with potentially wrong function or method call due to missing parentheses",
            "missing-parentheses-for-call-in-test",
            "Emitted when a conditional statement (If or ternary if) "
            "seems to wrongly call a function due to missing parentheses",
        ),
        "W0127": (
            "Assigning the same variable %r to itself",
            "self-assigning-variable",
            "Emitted when we detect that a variable is assigned to itself",
        ),
        "W0128": (
            "Redeclared variable %r in assignment",
            "redeclared-assigned-name",
            "Emitted when we detect that a variable was redeclared in the same assignment.",
        ),
        "E0111": (
            "The first reversed() argument is not a sequence",
            "bad-reversed-sequence",
            "Used when the first argument to reversed() builtin "
            "isn't a sequence (does not implement __reversed__, "
            "nor __getitem__ and __len__",
        ),
        "E0119": (
            "format function is not called on str",
            "misplaced-format-function",
            "Emitted when format function is not called on str object. "
            'e.g doing print("value: {}").format(123) instead of '
            'print("value: {}".format(123)). This might not be what the user '
            "intended to do.",
        ),
        "W0129": (
            "Assert statement has a string literal as its first argument. The assert will %s fail.",
            "assert-on-string-literal",
            "Used when an assert statement has a string literal as its first argument, which will "
            "cause the assert to always pass.",
        ),
    }
```
### 2 - pylint/checkers/base/basic_checker.py:

Start line: 278, End line: 339

```python
class BasicChecker(_BasicChecker):

    def _check_using_constant_test(self, node, test):
        const_nodes = (
            nodes.Module,
            nodes.GeneratorExp,
            nodes.Lambda,
            nodes.FunctionDef,
            nodes.ClassDef,
            astroid.bases.Generator,
            astroid.UnboundMethod,
            astroid.BoundMethod,
            nodes.Module,
        )
        structs = (nodes.Dict, nodes.Tuple, nodes.Set, nodes.List)

        # These nodes are excepted, since they are not constant
        # values, requiring a computation to happen.
        except_nodes = (
            nodes.Call,
            nodes.BinOp,
            nodes.BoolOp,
            nodes.UnaryOp,
            nodes.Subscript,
        )
        inferred = None
        emit = isinstance(test, (nodes.Const,) + structs + const_nodes)
        if not isinstance(test, except_nodes):
            inferred = utils.safe_infer(test)

        if emit:
            self.add_message("using-constant-test", node=node)
        elif isinstance(inferred, const_nodes):
            # If the constant node is a FunctionDef or Lambda then
            # it may be an illicit function call due to missing parentheses
            call_inferred = None
            try:
                if isinstance(inferred, nodes.FunctionDef):
                    call_inferred = inferred.infer_call_result()
                elif isinstance(inferred, nodes.Lambda):
                    call_inferred = inferred.infer_call_result(node)
            except astroid.InferenceError:
                call_inferred = None
            if call_inferred:
                try:
                    for inf_call in call_inferred:
                        if inf_call != astroid.Uninferable:
                            self.add_message(
                                "missing-parentheses-for-call-in-test", node=node
                            )
                            break
                except astroid.InferenceError:
                    pass
            self.add_message("using-constant-test", node=node)

    def visit_module(self, _: nodes.Module) -> None:
        """Check module name, docstring and required arguments."""
        self.linter.stats.node_count["module"] += 1

    def visit_classdef(self, _: nodes.ClassDef) -> None:
        """Check module name, docstring and redefinition
        increment branch counter
        """
        self.linter.stats.node_count["klass"] += 1
```
### 3 - pylint/checkers/base/comparison_checker.py:

Start line: 24, End line: 74

```python
class ComparisonChecker(_BasicChecker):
    """Checks for comparisons.

    - singleton comparison: 'expr == True', 'expr == False' and 'expr == None'
    - yoda condition: 'const "comp" right' where comp can be '==', '!=', '<',
      '<=', '>' or '>=', and right can be a variable, an attribute, a method or
      a function
    """

    msgs = {
        "C0121": (
            "Comparison %s should be %s",
            "singleton-comparison",
            "Used when an expression is compared to singleton "
            "values like True, False or None.",
        ),
        "C0123": (
            "Use isinstance() rather than type() for a typecheck.",
            "unidiomatic-typecheck",
            "The idiomatic way to perform an explicit typecheck in "
            "Python is to use isinstance(x, Y) rather than "
            "type(x) == Y, type(x) is Y. Though there are unusual "
            "situations where these give different results.",
            {"old_names": [("W0154", "old-unidiomatic-typecheck")]},
        ),
        "R0123": (
            "Comparison to literal",
            "literal-comparison",
            "Used when comparing an object to a literal, which is usually "
            "what you do not want to do, since you can compare to a different "
            "literal than what was expected altogether.",
        ),
        "R0124": (
            "Redundant comparison - %s",
            "comparison-with-itself",
            "Used when something is compared against itself.",
        ),
        "W0143": (
            "Comparing against a callable, did you omit the parenthesis?",
            "comparison-with-callable",
            "This message is emitted when pylint detects that a comparison with a "
            "callable was made, which might suggest that some parenthesis were omitted, "
            "resulting in potential unwanted behaviour.",
        ),
        "W0177": (
            "Comparison %s should be %s",
            "nan-comparison",
            "Used when an expression is compared to NaN"
            "values like numpy.NaN and float('nan')",
        ),
    }
```
### 4 - pylint/checkers/imports.py:

Start line: 259, End line: 385

```python
class ImportsChecker(DeprecatedMixin, BaseChecker):
    """BaseChecker for import statements.

    Checks for
    * external modules dependencies
    * relative / wildcard imports
    * cyclic imports
    * uses of deprecated modules
    * uses of modules instead of preferred modules
    """

    __implements__ = IAstroidChecker

    name = "imports"
    msgs = MSGS
    priority = -2
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
                "type": "string",
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
                "type": "string",
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
                "type": "string",
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
            "analyse-fallback-blocks",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Analyse import fallback blocks. This can be used to "
                "support both Python 2 and 3 compatible code, which "
                "means that the block might have code that exists "
                "only in one or another interpreter, leading to false "
                "positives when analysed.",
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
### 5 - pylint/checkers/base/comparison_checker.py:

Start line: 215, End line: 236

```python
class ComparisonChecker(_BasicChecker):

    def _check_callable_comparison(self, node):
        operator = node.ops[0][0]
        if operator not in COMPARISON_OPERATORS:
            return

        bare_callables = (nodes.FunctionDef, astroid.BoundMethod)
        left_operand, right_operand = node.left, node.ops[0][1]
        # this message should be emitted only when there is comparison of bare callable
        # with non bare callable.
        number_of_bare_callables = 0
        for operand in left_operand, right_operand:
            inferred = utils.safe_infer(operand)
            # Ignore callables that raise, as well as typing constants
            # implemented as functions (that raise via their decorator)
            if (
                isinstance(inferred, bare_callables)
                and "typing._SpecialForm" not in inferred.decoratornames()
                and not any(isinstance(x, nodes.Raise) for x in inferred.body)
            ):
                number_of_bare_callables += 1
        if number_of_bare_callables == 1:
            self.add_message("comparison-with-callable", node=node)
```
### 6 - pylint/checkers/base/comparison_checker.py:

Start line: 238, End line: 267

```python
class ComparisonChecker(_BasicChecker):

    @utils.check_messages(
        "singleton-comparison",
        "unidiomatic-typecheck",
        "literal-comparison",
        "comparison-with-itself",
        "comparison-with-callable",
    )
    def visit_compare(self, node: nodes.Compare) -> None:
        self._check_callable_comparison(node)
        self._check_logical_tautology(node)
        self._check_unidiomatic_typecheck(node)
        # NOTE: this checker only works with binary comparisons like 'x == 42'
        # but not 'x == y == 42'
        if len(node.ops) != 1:
            return

        left = node.left
        operator, right = node.ops[0]

        if operator in {"==", "!="}:
            self._check_singleton_comparison(
                left, right, node, checking_for_absence=operator == "!="
            )

        if operator in {"==", "!=", "is", "is not"}:
            self._check_nan_comparison(
                left, right, node, checking_for_absence=operator in {"!=", "is not"}
            )
        if operator in {"is", "is not"}:
            self._check_literal_comparison(right, node)
```
### 7 - pylint/checkers/base/comparison_checker.py:

Start line: 175, End line: 186

```python
class ComparisonChecker(_BasicChecker):

    def _check_literal_comparison(self, literal, node: nodes.Compare):
        """Check if we compare to a literal, which is usually what we do not want to do."""
        is_other_literal = isinstance(literal, (nodes.List, nodes.Dict, nodes.Set))
        is_const = False
        if isinstance(literal, nodes.Const):
            if isinstance(literal.value, bool) or literal.value is None:
                # Not interested in these values.
                return
            is_const = isinstance(literal.value, (bytes, str, int, float))

        if is_const or is_other_literal:
            self.add_message("literal-comparison", node=node)
```
### 8 - pylint/checkers/refactoring/implicit_booleaness_checker.py:

Start line: 5, End line: 78

```python
from typing import List, Union

import astroid
from astroid import bases, nodes

from pylint import checkers, interfaces
from pylint.checkers import utils


class ImplicitBooleanessChecker(checkers.BaseChecker):
    """Checks for incorrect usage of comparisons or len() inside conditions.

    Incorrect usage of len()
    Pep8 states:
    For sequences, (strings, lists, tuples), use the fact that empty sequences are false.

        Yes: if not seq:
             if seq:

        No: if len(seq):
            if not len(seq):

    Problems detected:
    * if len(sequence):
    * if not len(sequence):
    * elif len(sequence):
    * elif not len(sequence):
    * while len(sequence):
    * while not len(sequence):
    * assert len(sequence):
    * assert not len(sequence):
    * bool(len(sequence))

    Incorrect usage of empty literal sequences; (), [], {},

    For empty sequences, (dicts, lists, tuples), use the fact that empty sequences are false.

        Yes: if variable:
             if not variable

        No: if variable == empty_literal:
            if variable != empty_literal:

    Problems detected:
    * comparison such as variable == empty_literal:
    * comparison such as variable != empty_literal:
    """

    __implements__ = (interfaces.IAstroidChecker,)

    # configuration section name
    name = "refactoring"
    msgs = {
        "C1802": (
            "Do not use `len(SEQUENCE)` without comparison to determine if a sequence is empty",
            "use-implicit-booleaness-not-len",
            "Used when Pylint detects that len(sequence) is being used "
            "without explicit comparison inside a condition to determine if a sequence is empty. "
            "Instead of coercing the length to a boolean, either "
            "rely on the fact that empty sequences are false or "
            "compare the length against a scalar.",
            {"old_names": [("C1801", "len-as-condition")]},
        ),
        "C1803": (
            "'%s' can be simplified to '%s' as an empty sequence is falsey",
            "use-implicit-booleaness-not-comparison",
            "Used when Pylint detects that collection literal comparison is being "
            "used to check for emptiness; Use implicit booleaness instead"
            "of a collection classes; empty collections are considered as false",
        ),
    }

    priority = -2
    options = ()
```
### 9 - pylint/checkers/base/basic_checker.py:

Start line: 567, End line: 586

```python
class BasicChecker(_BasicChecker):

    def _check_misplaced_format_function(self, call_node):
        if not isinstance(call_node.func, nodes.Attribute):
            return
        if call_node.func.attrname != "format":
            return

        expr = utils.safe_infer(call_node.func.expr)
        if expr is astroid.Uninferable:
            return
        if not expr:
            # we are doubtful on inferred type of node, so here just check if format
            # was called on print()
            call_expr = call_node.func.expr
            if not isinstance(call_expr, nodes.Call):
                return
            if (
                isinstance(call_expr.func, nodes.Name)
                and call_expr.func.name == "print"
            ):
                self.add_message("misplaced-format-function", node=call_node)
```
### 10 - pylint/checkers/base/basic_error_checker.py:

Start line: 297, End line: 324

```python
class BasicErrorChecker(_BasicChecker):

    visit_asyncfunctiondef = visit_functiondef

    def _check_name_used_prior_global(self, node):

        scope_globals = {
            name: child
            for child in node.nodes_of_class(nodes.Global)
            for name in child.names
            if child.scope() is node
        }

        if not scope_globals:
            return

        for node_name in node.nodes_of_class(nodes.Name):
            if node_name.scope() is not node:
                continue

            name = node_name.name
            corresponding_global = scope_globals.get(name)
            if not corresponding_global:
                continue

            global_lineno = corresponding_global.fromlineno
            if global_lineno and global_lineno > node_name.fromlineno:
                self.add_message(
                    "used-prior-global-declaration", node=node_name, args=(name,)
                )
```
### 12 - pylint/checkers/base_checker.py:

Start line: 5, End line: 23

```python
import functools
import sys
from inspect import cleandoc
from typing import Any, Optional

from astroid import nodes

from pylint.config import OptionsProviderMixIn
from pylint.config.exceptions import MissingArgumentManager
from pylint.constants import _MSG_ORDER, WarningScope
from pylint.exceptions import InvalidMessageError
from pylint.interfaces import Confidence, IRawChecker, ITokenChecker, implements
from pylint.message.message_definition import MessageDefinition
from pylint.utils import get_rst_section, get_rst_title

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
```
