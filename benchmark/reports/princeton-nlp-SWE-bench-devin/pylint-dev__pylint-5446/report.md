# pylint-dev__pylint-5446

| **pylint-dev/pylint** | `a1df7685a4e6a05b519ea011f16a2f0d49d08032` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/checkers/similar.py b/pylint/checkers/similar.py
--- a/pylint/checkers/similar.py
+++ b/pylint/checkers/similar.py
@@ -381,10 +381,19 @@ def append_stream(
         else:
             readlines = stream.readlines  # type: ignore[assignment] # hint parameter is incorrectly typed as non-optional
         try:
+            active_lines: List[str] = []
+            if hasattr(self, "linter"):
+                # Remove those lines that should be ignored because of disables
+                for index, line in enumerate(readlines()):
+                    if self.linter._is_one_message_enabled("R0801", index + 1):  # type: ignore[attr-defined]
+                        active_lines.append(line)
+            else:
+                active_lines = readlines()
+
             self.linesets.append(
                 LineSet(
                     streamid,
-                    readlines(),
+                    active_lines,
                     self.ignore_comments,
                     self.ignore_docstrings,
                     self.ignore_imports,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/checkers/similar.py | 384 | 384 | - | 5 | -


## Problem Statement

```
The duplicate-code (R0801) can't be disabled
Originally reported by: **Anonymous**

---

It's seems like it's not possible to disable the duplicate code check on portions of a file. Looking at the source, I can see why as it's not a trivial thing to do (if you want to maintain the same scope semantics as other #pylint:enable/disable comments. This would be nice to have though (or I guess I could just cleanup my duplicate code).

---
- Bitbucket: https://bitbucket.org/logilab/pylint/issue/214


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/lint/pylinter.py | 1791 | 1822| 246 | 246 | 14023 | 
| 2 | 2 pylint/constants.py | 87 | 179| 1201 | 1447 | 15890 | 
| 3 | 3 pylint/extensions/empty_comment.py | 37 | 64| 182 | 1629 | 16312 | 
| 4 | 4 pylint/checkers/stdlib.py | 348 | 464| 1195 | 2824 | 23296 | 
| 5 | 4 pylint/lint/pylinter.py | 807 | 835| 258 | 3082 | 23296 | 
| 6 | **5 pylint/checkers/similar.py** | 729 | 802| 481 | 3563 | 31487 | 
| 7 | 6 pylint/checkers/design_analysis.py | 114 | 192| 510 | 4073 | 36793 | 
| 8 | 7 pylint/extensions/bad_builtin.py | 31 | 57| 177 | 4250 | 37484 | 
| 9 | 7 pylint/lint/pylinter.py | 97 | 187| 788 | 5038 | 37484 | 
| 10 | 7 pylint/lint/pylinter.py | 837 | 858| 212 | 5250 | 37484 | 
| 11 | 7 pylint/checkers/stdlib.py | 83 | 128| 538 | 5788 | 37484 | 
| 12 | **7 pylint/checkers/similar.py** | 707 | 726| 162 | 5950 | 37484 | 
| 13 | **7 pylint/checkers/similar.py** | 345 | 371| 202 | 6152 | 37484 | 
| 14 | 8 pylint/checkers/refactoring/implicit_booleaness_checker.py | 3 | 76| 516 | 6668 | 39268 | 
| 15 | 9 pylint/checkers/modified_iterating_checker.py | 127 | 155| 253 | 6921 | 40537 | 
| 16 | 10 pylint/checkers/base.py | 464 | 572| 952 | 7873 | 62710 | 
| 17 | 11 pylint/checkers/classes/class_checker.py | 692 | 787| 631 | 8504 | 79931 | 
| 18 | 11 pylint/extensions/bad_builtin.py | 59 | 74| 170 | 8674 | 79931 | 
| 19 | 12 pylint/checkers/typecheck.py | 783 | 933| 1094 | 9768 | 97438 | 
| 20 | 13 pylint/checkers/misc.py | 77 | 128| 314 | 10082 | 99194 | 
| 21 | 14 pylint/checkers/format.py | 694 | 714| 192 | 10274 | 106346 | 
| 22 | 15 pylint/checkers/ellipsis_checker.py | 28 | 53| 230 | 10504 | 106744 | 
| 23 | 15 pylint/checkers/misc.py | 155 | 204| 346 | 10850 | 106744 | 
| 24 | 15 pylint/checkers/misc.py | 30 | 64| 223 | 11073 | 106744 | 
| 25 | 15 pylint/checkers/base.py | 941 | 1083| 1328 | 12401 | 106744 | 
| 26 | 15 pylint/checkers/stdlib.py | 466 | 514| 518 | 12919 | 106744 | 
| 27 | 16 pylint/extensions/comparison_placement.py | 56 | 71| 155 | 13074 | 107294 | 
| 28 | 16 pylint/checkers/stdlib.py | 131 | 270| 1034 | 14108 | 107294 | 
| 29 | 16 pylint/constants.py | 3 | 85| 625 | 14733 | 107294 | 
| 30 | 17 pylint/checkers/refactoring/refactoring_checker.py | 606 | 625| 153 | 14886 | 123886 | 
| 31 | 17 pylint/checkers/classes/class_checker.py | 1173 | 1278| 760 | 15646 | 123886 | 
| 32 | 17 pylint/checkers/refactoring/refactoring_checker.py | 900 | 910| 118 | 15764 | 123886 | 
| 33 | 17 pylint/checkers/stdlib.py | 273 | 315| 200 | 15964 | 123886 | 
| 34 | 17 pylint/extensions/empty_comment.py | 1 | 20| 127 | 16091 | 123886 | 
| 35 | 17 pylint/checkers/classes/class_checker.py | 881 | 929| 360 | 16451 | 123886 | 
| 36 | 17 pylint/checkers/refactoring/refactoring_checker.py | 4 | 53| 359 | 16810 | 123886 | 
| 37 | 17 pylint/checkers/refactoring/refactoring_checker.py | 878 | 898| 203 | 17013 | 123886 | 
| 38 | 17 pylint/checkers/refactoring/refactoring_checker.py | 434 | 486| 386 | 17399 | 123886 | 
| 39 | 17 pylint/checkers/classes/class_checker.py | 816 | 826| 117 | 17516 | 123886 | 
| 40 | 18 pylint/checkers/variables.py | 413 | 546| 1186 | 18702 | 147155 | 
| 41 | 18 pylint/checkers/base.py | 1478 | 1492| 139 | 18841 | 147155 | 
| 42 | 18 pylint/lint/pylinter.py | 1427 | 1445| 220 | 19061 | 147155 | 
| 43 | 18 pylint/checkers/modified_iterating_checker.py | 70 | 86| 155 | 19216 | 147155 | 
| 44 | 18 pylint/checkers/design_analysis.py | 562 | 596| 254 | 19470 | 147155 | 
| 45 | 19 pylint/checkers/spelling.py | 187 | 202| 151 | 19621 | 151030 | 
| 46 | 19 pylint/checkers/stdlib.py | 569 | 590| 218 | 19839 | 151030 | 
| 47 | 20 pylint/checkers/unicode.py | 297 | 380| 703 | 20542 | 155340 | 
| 48 | 20 pylint/checkers/refactoring/refactoring_checker.py | 591 | 604| 128 | 20670 | 155340 | 
| 49 | 21 pylint/extensions/overlapping_exceptions.py | 6 | 90| 608 | 21278 | 155995 | 
| 50 | 21 pylint/checkers/base.py | 574 | 585| 120 | 21398 | 155995 | 
| 51 | 22 pylint/extensions/broad_try_clause.py | 13 | 80| 446 | 21844 | 156664 | 
| 52 | 23 pylint/utils/file_state.py | 150 | 179| 247 | 22091 | 158120 | 
| 53 | 23 pylint/checkers/refactoring/refactoring_checker.py | 1351 | 1371| 168 | 22259 | 158120 | 
| 54 | 24 pylint/extensions/comparetozero.py | 54 | 85| 310 | 22569 | 158886 | 
| 55 | **24 pylint/checkers/similar.py** | 865 | 882| 164 | 22733 | 158886 | 
| 56 | 24 pylint/checkers/format.py | 545 | 565| 212 | 22945 | 158886 | 
| 57 | 24 pylint/checkers/refactoring/refactoring_checker.py | 659 | 674| 176 | 23121 | 158886 | 
| 58 | 25 pylint/checkers/utils.py | 1390 | 1406| 139 | 23260 | 172573 | 
| 59 | 25 pylint/checkers/spelling.py | 205 | 290| 629 | 23889 | 172573 | 
| 60 | 25 pylint/checkers/modified_iterating_checker.py | 114 | 125| 120 | 24009 | 172573 | 
| 61 | 25 pylint/checkers/design_analysis.py | 293 | 428| 813 | 24822 | 172573 | 
| 62 | 25 pylint/checkers/design_analysis.py | 33 | 113| 689 | 25511 | 172573 | 
| 63 | 26 pylint/extensions/docparams.py | 69 | 153| 701 | 26212 | 177981 | 
| 64 | 26 pylint/checkers/base.py | 2245 | 2262| 138 | 26350 | 177981 | 
| 65 | 26 pylint/checkers/base.py | 1455 | 1476| 177 | 26527 | 177981 | 
| 66 | 26 pylint/checkers/refactoring/refactoring_checker.py | 1180 | 1221| 365 | 26892 | 177981 | 
| 67 | 26 pylint/checkers/base.py | 2287 | 2337| 464 | 27356 | 177981 | 
| 68 | 27 pylint/lint/parallel.py | 71 | 106| 288 | 27644 | 179459 | 
| 69 | 27 pylint/checkers/refactoring/refactoring_checker.py | 1854 | 1956| 868 | 28512 | 179459 | 
| 70 | 27 pylint/checkers/refactoring/refactoring_checker.py | 751 | 768| 169 | 28681 | 179459 | 
| 71 | 27 pylint/checkers/unicode.py | 459 | 474| 137 | 28818 | 179459 | 
| 72 | 27 pylint/checkers/refactoring/refactoring_checker.py | 1259 | 1272| 158 | 28976 | 179459 | 
| 73 | 27 pylint/checkers/stdlib.py | 720 | 757| 273 | 29249 | 179459 | 
| 74 | 27 pylint/checkers/typecheck.py | 558 | 579| 214 | 29463 | 179459 | 
| 75 | 27 pylint/checkers/typecheck.py | 2043 | 2122| 733 | 30196 | 179459 | 
| 76 | 28 pylint/testutils/functional/lint_module_output_update.py | 38 | 55| 170 | 30366 | 179903 | 
| 77 | 28 pylint/checkers/base.py | 2339 | 2395| 459 | 30825 | 179903 | 
| 78 | 28 pylint/checkers/stdlib.py | 40 | 81| 429 | 31254 | 179903 | 
| 79 | 28 pylint/checkers/refactoring/implicit_booleaness_checker.py | 78 | 117| 395 | 31649 | 179903 | 
| 80 | 28 pylint/checkers/refactoring/refactoring_checker.py | 1016 | 1032| 162 | 31811 | 179903 | 
| 81 | **28 pylint/checkers/similar.py** | 44 | 108| 398 | 32209 | 179903 | 
| 82 | 28 pylint/checkers/refactoring/refactoring_checker.py | 627 | 642| 160 | 32369 | 179903 | 
| 83 | 28 pylint/checkers/refactoring/refactoring_checker.py | 996 | 1014| 177 | 32546 | 179903 | 
| 84 | 29 pylint/extensions/code_style.py | 275 | 310| 283 | 32829 | 182412 | 
| 85 | 29 pylint/checkers/refactoring/refactoring_checker.py | 933 | 965| 302 | 33131 | 182412 | 
| 86 | 29 pylint/checkers/unicode.py | 13 | 51| 412 | 33543 | 182412 | 
| 87 | 29 pylint/checkers/stdlib.py | 625 | 650| 195 | 33738 | 182412 | 
| 88 | 29 pylint/checkers/modified_iterating_checker.py | 88 | 112| 206 | 33944 | 182412 | 
| 89 | 29 pylint/checkers/base.py | 2500 | 2529| 257 | 34201 | 182412 | 
| 90 | 29 pylint/checkers/base.py | 618 | 662| 406 | 34607 | 182412 | 
| 91 | 30 pylint/checkers/refactoring/not_checker.py | 5 | 42| 292 | 34899 | 183054 | 
| 92 | 30 pylint/checkers/refactoring/not_checker.py | 44 | 84| 317 | 35216 | 183054 | 
| 93 | 31 pylint/message/message_id_store.py | 86 | 105| 235 | 35451 | 184360 | 
| 94 | 32 pylint/pyreverse/__init__.py | 1 | 7| 0 | 35451 | 184414 | 
| 95 | 32 pylint/checkers/variables.py | 794 | 864| 548 | 35999 | 184414 | 
| 96 | 32 pylint/checkers/refactoring/refactoring_checker.py | 967 | 994| 266 | 36265 | 184414 | 
| 97 | 32 pylint/checkers/ellipsis_checker.py | 1 | 26| 175 | 36440 | 184414 | 
| 98 | 32 pylint/checkers/base.py | 721 | 758| 380 | 36820 | 184414 | 
| 99 | 32 pylint/checkers/typecheck.py | 1193 | 1210| 135 | 36955 | 184414 | 
| 100 | 32 pylint/checkers/refactoring/refactoring_checker.py | 1556 | 1635| 589 | 37544 | 184414 | 
| 101 | 33 pylint/extensions/redefined_variable_type.py | 65 | 119| 452 | 37996 | 185471 | 
| 102 | 33 pylint/extensions/bad_builtin.py | 14 | 28| 107 | 38103 | 185471 | 
| 103 | 33 pylint/extensions/empty_comment.py | 23 | 34| 111 | 38214 | 185471 | 
| 104 | 33 pylint/lint/pylinter.py | 1760 | 1789| 272 | 38486 | 185471 | 
| 105 | 33 pylint/checkers/refactoring/refactoring_checker.py | 676 | 703| 283 | 38769 | 185471 | 
| 106 | 33 pylint/checkers/base.py | 72 | 100| 167 | 38936 | 185471 | 
| 107 | 34 pylint/extensions/_check_docs_utils.py | 557 | 652| 512 | 39448 | 191250 | 
| 108 | 34 pylint/checkers/base.py | 864 | 938| 585 | 40033 | 191250 | 
| 109 | 34 pylint/checkers/refactoring/refactoring_checker.py | 1398 | 1427| 259 | 40292 | 191250 | 
| 110 | 34 pylint/checkers/typecheck.py | 935 | 965| 358 | 40650 | 191250 | 
| 111 | 34 pylint/lint/pylinter.py | 1083 | 1099| 189 | 40839 | 191250 | 


### Hint

```
_Original comment by_ **Radek Holý (BitBucket: [PyDeq](http://bitbucket.org/PyDeq), GitHub: @PyDeq?)**:

---

Pylint marks even import blocks as duplicates. In my case, it is:

\`\`\`
#!python
import contextlib
import io
import itertools
import os
import subprocess
import tempfile

\`\`\`

I doubt it is possible to clean up/refactor the code to prevent this, thus it would be nice if it would ignore imports or if it would be possible to use the disable comment.

_Original comment by_ **Buck Evan (BitBucket: [bukzor](http://bitbucket.org/bukzor), GitHub: @bukzor?)**:

---

I've run into this today.

My project has several setup.py files, which of course look quite similar, but can't really share code.
I tried to `# pylint:disable=duplicate-code` just the line in question (`setup()`), but it did nothing.

I'll have to turn the checker off entirely I think.

any news on this issue ?

No one is currently working on this. This would be nice to have fixed, but unfortunately I didn't have time to look into it. A pull request would be appreciated though and would definitely move it forward.

@gaspaio 
FYI I have created a fix in the following PR https://github.com/PyCQA/pylint/pull/1055
Could you check if It work for you?

Is this related to duplicate-except? it cannot be disabled either. 

sorry, Seems it is able to be disabled.
@PCManticore as you mentioned, there is probably no good way to fix this issue for this version; What about fixing this in a "[bad way](https://github.com/PyCQA/pylint/pull/1055)" for this version, and later on in the planned version 3 you can [utilize a better engineered two-phase design](https://github.com/PyCQA/pylint/pull/1014#issuecomment-233185403) ?

While sailing, sometimes the only way to fix your ship is ugly patching, and it's better than not fixing.
A functioning tool is absolutely better than a "perfectly engineered" tool.
> A functioning tool is absolutely better than a "perfectly engineered" tool.

It's better because it exists, unlike perfect engineering :D
Is this issue resolved? I'm using pylint 2.1.1 and cannot seem to disable the warning with comments in the files which are printed in the warning.
Hi @levsa 
I have created a custom patch that you can use locally from:
 - https://github.com/PyCQA/pylint/pull/1055#issuecomment-384382126

We are using it 2.5 years ago and it is working fine.

The main issue is that the `state_lines` are not propagated for the methods used.
This patch just propagates them correctly.

You can use:
\`\`\`bash
wget -q https://raw.githubusercontent.com/Vauxoo/pylint-conf/master/conf/pylint_pr1055.patch -O pylint_pr1055.patch
patch -f -p0 $(python -c "from __future__ import print_function; from pylint.checkers import similar; print(similar.__file__.rstrip('c'))") -i pylint_pr1055.patch
\`\`\`
@moylop260 Your fix doesn't work. Even if I have `# pylint: disable=duplicate-code` in each file with duplicate code, I still get errors.
You have duplicated comments too.
Check answer https://github.com/PyCQA/pylint/pull/1055#issuecomment-470572805
Still have this issue on pylint = "==2.3.1"
The issue still there for latest python 2 pylint version 1.9.4

`# pylint: disable=duplicate-code` does not work, while # pylint: disable=all` can disable the dup check.
Also have this issue with `pylint==2.3.1`

Even if i add the the pragma in every file with duplicate code like @filips123, i still get the files with the pragma reported ....

Any progress on this? I don't want to `# pylint: disable=all` for all affected files ...
I am using pylint 2.4.0 for python 3 and I can see that `disable=duplicate-code` is working.
I'm using pylint 2.4.4 and `# pylint: disable=duplicate-code` is not working for me.
Also on `2.4.4`
And on 2.4.2
another confirmation of this issue on 2.4.4
Not working for me `# pylint: disable=duplicate-code` at 2.4.4 version python 3.8
Also **not** woking at 2.4.4 with python 3.7.4 and `# pylint: disable=R0801` or `# pylint: disable=duplicate-code`
Also doesn't work. pylint 2.4.4, astroid 2.3.3, Python 3.8.2
I found a solution to partially solve this.

https://github.com/PyCQA/pylint/pull/1055#issuecomment-477253153

Use pylintrc. Try changing the `min-similarity-lines` in the similarities section of your pylintrc config file.

\`\`\`INI
[SIMILARITIES]

# Minimum lines number of a similarity.
min-similarity-lines=4

# Ignore comments when computing similarities.
ignore-comments=yes

# Ignore docstrings when computing similarities.
ignore-docstrings=yes

# Ignore imports when computing similarities.
ignore-imports=no
\`\`\`


+1
Happened to me on a overloaded method declaration, which is obviously identical as its parent class, and in my case its sister class, so pylint reports "Similar lines in 3 files" with the `def` and its 4 arguments (spanning on 6 lines, I have one argument per line due to type anotations).

It think this could be mitigated by whitelisting import statements and method declaration.
Just been bitten by this issue today.

> pylint 2.6.0
astroid 2.4.2
Python 3.9.1 (default, Jan 20 2021, 00:00:00) 

Increasing `min-similarity-lines` is not actually a solution, as it will turn off duplicate code evaluation.
In my particular case I had two separate classes with similar inputs in the `def __init__`

As a workaround and what worked for me was to include the following in our pyproject.toml (which is the SIMILARITIES section if you use pylinr.rc):
\`\`\`
[tool.pylint.SIMULARITIES]
ignore-comments = "no"
\`\`\`

Doing this then allows pylint to not disregard the comments on lines. From there we added the `# pylint: disable=duplicate-code` to the end of one of the lines which now made this "unique" (although you could in practice use any comment text)

This is a better temporary workaround than the `min-similarity-lines` option as that:
* misses other instance where you'd want that check
* also doesnt return with a valid exit code even though rating is 10.00/10
it is possible now with file pylint.rc with disable key and value duplicate-code, this issue can be closed and solve in my opnion #214
> it is possible now with file pylint.rc with disable key and value duplicate-code, this issue can be closed and solve in my opnion #214

The main issue is not being able to use `# pylint: disable=duplicate-code` to exclude particular blocks of code. Disabling `duplicate-code` all together is not an acceptable solution. 
Since the issue was opened we added multiple option to ignore import, ignore docstrings and ignore signatures, so there is less and less reasons to want to disable it. But the problem still exists.
So any idea how to exclude particular blocks of code from the duplicated code checks without disabling the whole feature?
> So any idea how to exclude particular blocks of code from the duplicated code checks without disabling the whole feature?

Right now there isn't, this require a change in pylint.
any news about this? It looks like many people have been bitten by this and it's marked as high prio
thanks
```

## Patch

```diff
diff --git a/pylint/checkers/similar.py b/pylint/checkers/similar.py
--- a/pylint/checkers/similar.py
+++ b/pylint/checkers/similar.py
@@ -381,10 +381,19 @@ def append_stream(
         else:
             readlines = stream.readlines  # type: ignore[assignment] # hint parameter is incorrectly typed as non-optional
         try:
+            active_lines: List[str] = []
+            if hasattr(self, "linter"):
+                # Remove those lines that should be ignored because of disables
+                for index, line in enumerate(readlines()):
+                    if self.linter._is_one_message_enabled("R0801", index + 1):  # type: ignore[attr-defined]
+                        active_lines.append(line)
+            else:
+                active_lines = readlines()
+
             self.linesets.append(
                 LineSet(
                     streamid,
-                    readlines(),
+                    active_lines,
                     self.ignore_comments,
                     self.ignore_docstrings,
                     self.ignore_imports,

```

## Test Patch

```diff
diff --git a/tests/regrtest_data/duplicate_data_raw_strings/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_all/__init__.py
similarity index 100%
rename from tests/regrtest_data/duplicate_data_raw_strings/__init__.py
rename to tests/regrtest_data/duplicate_code/raw_strings_all/__init__.py
diff --git a/tests/regrtest_data/duplicate_data_raw_strings/first.py b/tests/regrtest_data/duplicate_code/raw_strings_all/first.py
similarity index 100%
rename from tests/regrtest_data/duplicate_data_raw_strings/first.py
rename to tests/regrtest_data/duplicate_code/raw_strings_all/first.py
diff --git a/tests/regrtest_data/duplicate_data_raw_strings/second.py b/tests/regrtest_data/duplicate_code/raw_strings_all/second.py
similarity index 100%
rename from tests/regrtest_data/duplicate_data_raw_strings/second.py
rename to tests/regrtest_data/duplicate_code/raw_strings_all/second.py
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_all/third.py b/tests/regrtest_data/duplicate_code/raw_strings_all/third.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_all/third.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/first.py
@@ -0,0 +1,12 @@
+# pylint: disable=duplicate-code
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file/third.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/third.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file/third.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/first.py
@@ -0,0 +1,12 @@
+# pylint: disable=duplicate-code
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/second.py
@@ -0,0 +1,12 @@
+# pylint: disable=duplicate-code
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/third.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/third.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_file_double/third.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/first.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1  # pylint: disable=duplicate-code
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_begin/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/first.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1  # pylint: disable=duplicate-code
+    yyyy = 2  # pylint: disable=duplicate-code
+    zzzz = 3  # pylint: disable=duplicate-code
+    wwww = 4  # pylint: disable=duplicate-code
+    vvvv = xxxx + yyyy + zzzz + wwww  # pylint: disable=duplicate-code
+    return vvvv  # pylint: disable=duplicate-code
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_disable_all/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/first.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv  # pylint: disable=duplicate-code
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_end/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/first.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3  # pylint: disable=duplicate-code
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_line_middle/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/first.py
@@ -0,0 +1,12 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    # pylint: disable=duplicate-code
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/second.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/third.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/third.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope/third.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/first.py
@@ -0,0 +1,12 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    # pylint: disable=duplicate-code
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/second.py
@@ -0,0 +1,12 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    # pylint: disable=duplicate-code
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/third.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/third.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_double/third.py
@@ -0,0 +1,11 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/__init__.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/__init__.py
new file mode 100644
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/first.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/first.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/first.py
@@ -0,0 +1,21 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    # pylint: disable=duplicate-code
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
+
+
+def look_busy_two():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/second.py b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/second.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/duplicate_code/raw_strings_disable_scope_second_function/second.py
@@ -0,0 +1,20 @@
+r"""A raw docstring.
+"""
+
+
+def look_busy():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
+
+
+def look_busy_two():
+    xxxx = 1
+    yyyy = 2
+    zzzz = 3
+    wwww = 4
+    vvvv = xxxx + yyyy + zzzz + wwww
+    return vvvv
diff --git a/tests/test_self.py b/tests/test_self.py
--- a/tests/test_self.py
+++ b/tests/test_self.py
@@ -1121,14 +1121,6 @@ def test_jobs_score(self) -> None:
         expected = "Your code has been rated at 7.50/10"
         self._test_output([path, "--jobs=2", "-ry"], expected_output=expected)
 
-    def test_duplicate_code_raw_strings(self) -> None:
-        path = join(HERE, "regrtest_data", "duplicate_data_raw_strings")
-        expected_output = "Similar lines in 2 files"
-        self._test_output(
-            [path, "--disable=all", "--enable=duplicate-code"],
-            expected_output=expected_output,
-        )
-
     def test_regression_parallel_mode_without_filepath(self) -> None:
         # Test that parallel mode properly passes filepath
         # https://github.com/PyCQA/pylint/issues/3564
diff --git a/tests/test_similar.py b/tests/test_similar.py
new file mode 100644
--- /dev/null
+++ b/tests/test_similar.py
@@ -0,0 +1,141 @@
+# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
+# For details: https://github.com/PyCQA/pylint/blob/main/LICENSE
+
+
+import contextlib
+import os
+import re
+import sys
+import warnings
+from io import StringIO
+from os.path import abspath, dirname, join
+from typing import Iterator, List, TextIO
+
+import pytest
+
+from pylint.lint import Run
+
+HERE = abspath(dirname(__file__))
+DATA = join(HERE, "regrtest_data", "duplicate_code")
+CLEAN_PATH = re.escape(dirname(dirname(__file__)) + os.path.sep)
+
+
+@contextlib.contextmanager
+def _patch_streams(out: TextIO) -> Iterator:
+    sys.stderr = sys.stdout = out
+    try:
+        yield
+    finally:
+        sys.stderr = sys.__stderr__
+        sys.stdout = sys.__stdout__
+
+
+class TestSimilarCodeChecker:
+    def _runtest(self, args: List[str], code: int) -> None:
+        """Runs the tests and sees if output code is as expected."""
+        out = StringIO()
+        pylint_code = self._run_pylint(args, out=out)
+        output = out.getvalue()
+        msg = f"expected output status {code}, got {pylint_code}"
+        if output is not None:
+            msg = f"{msg}. Below pylint output: \n{output}"
+        assert pylint_code == code, msg
+
+    @staticmethod
+    def _run_pylint(args: List[str], out: TextIO) -> int:
+        """Runs pylint with a patched output."""
+        args = args + ["--persistent=no"]
+        with _patch_streams(out):
+            with pytest.raises(SystemExit) as cm:
+                with warnings.catch_warnings():
+                    warnings.simplefilter("ignore")
+                    Run(args)
+            return cm.value.code
+
+    @staticmethod
+    def _clean_paths(output: str) -> str:
+        """Normalize path to the tests directory."""
+        output = re.sub(CLEAN_PATH, "", output, flags=re.MULTILINE)
+        return output.replace("\\", "/")
+
+    def _test_output(self, args: List[str], expected_output: str) -> None:
+        """Tests if the output of a pylint run is as expected."""
+        out = StringIO()
+        self._run_pylint(args, out=out)
+        actual_output = self._clean_paths(out.getvalue())
+        expected_output = self._clean_paths(expected_output)
+        assert expected_output.strip() in actual_output.strip()
+
+    def test_duplicate_code_raw_strings_all(self) -> None:
+        """Test similar lines in 3 similar files."""
+        path = join(DATA, "raw_strings_all")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )
+
+    def test_duplicate_code_raw_strings_disable_file(self) -> None:
+        """Tests disabling duplicate-code at the file level in a single file."""
+        path = join(DATA, "raw_strings_disable_file")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )
+
+    def test_duplicate_code_raw_strings_disable_file_double(self) -> None:
+        """Tests disabling duplicate-code at the file level in two files."""
+        path = join(DATA, "raw_strings_disable_file_double")
+        self._runtest([path, "--disable=all", "--enable=duplicate-code"], code=0)
+
+    def test_duplicate_code_raw_strings_disable_line_two(self) -> None:
+        """Tests disabling duplicate-code at a line at the begin of a piece of similar code."""
+        path = join(DATA, "raw_strings_disable_line_begin")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )
+
+    def test_duplicate_code_raw_strings_disable_line_disable_all(self) -> None:
+        """Tests disabling duplicate-code with all similar lines disabled per line."""
+        path = join(DATA, "raw_strings_disable_line_disable_all")
+        self._runtest([path, "--disable=all", "--enable=duplicate-code"], code=0)
+
+    def test_duplicate_code_raw_strings_disable_line_midle(self) -> None:
+        """Tests disabling duplicate-code at a line in the middle of a piece of similar code."""
+        path = join(DATA, "raw_strings_disable_line_middle")
+        self._runtest([path, "--disable=all", "--enable=duplicate-code"], code=0)
+
+    def test_duplicate_code_raw_strings_disable_line_end(self) -> None:
+        """Tests disabling duplicate-code at a line at the end of a piece of similar code."""
+        path = join(DATA, "raw_strings_disable_line_end")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )
+
+    def test_duplicate_code_raw_strings_disable_scope(self) -> None:
+        """Tests disabling duplicate-code at an inner scope level."""
+        path = join(DATA, "raw_strings_disable_scope")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )
+
+    def test_duplicate_code_raw_strings_disable_scope_double(self) -> None:
+        """Tests disabling duplicate-code at an inner scope level in two files."""
+        path = join(DATA, "raw_strings_disable_scope_double")
+        self._runtest([path, "--disable=all", "--enable=duplicate-code"], code=0)
+
+    def test_duplicate_code_raw_strings_disable_scope_function(self) -> None:
+        """Tests disabling duplicate-code at an inner scope level with another scope with similarity."""
+        path = join(DATA, "raw_strings_disable_scope_second_function")
+        expected_output = "Similar lines in 2 files"
+        self._test_output(
+            [path, "--disable=all", "--enable=duplicate-code"],
+            expected_output=expected_output,
+        )

```


## Code snippets

### 1 - pylint/lint/pylinter.py:

Start line: 1791, End line: 1822

```python
class PyLinter(
    config.OptionsManagerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def disable_next(
        self,
        msgid: str,
        scope: str = "package",
        line: Optional[int] = None,
        ignore_unknown: bool = False,
    ) -> None:
        """Disable a message for the next line."""
        if not line:
            raise exceptions.NoLineSuppliedError
        self._set_msg_status(
            msgid,
            enable=False,
            scope=scope,
            line=line + 1,
            ignore_unknown=ignore_unknown,
        )
        self._register_by_id_managed_msg(msgid, line + 1)

    def enable(
        self,
        msgid: str,
        scope: str = "package",
        line: Optional[int] = None,
        ignore_unknown: bool = False,
    ) -> None:
        """Enable a message for a scope."""
        self._set_msg_status(
            msgid, enable=True, scope=scope, line=line, ignore_unknown=ignore_unknown
        )
        self._register_by_id_managed_msg(msgid, line, is_disabled=False)
```
### 2 - pylint/constants.py:

Start line: 87, End line: 179

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
    DeletedMessage("W1641", "eq-without-hash"),
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
]
```
### 3 - pylint/extensions/empty_comment.py:

Start line: 37, End line: 64

```python
class CommentChecker(BaseChecker):
    __implements__ = IRawChecker

    name = "refactoring"
    msgs = {
        "R2044": (
            "Line with empty comment",
            "empty-comment",
            (
                "Used when a # symbol appears on a line not followed by an actual comment"
            ),
        )
    }
    options = ()
    priority = -1  # low priority

    def process_module(self, node: nodes.Module) -> None:
        with node.stream() as stream:
            for (line_num, line) in enumerate(stream):
                line = line.rstrip()
                if line.endswith(b"#"):
                    if not is_line_commented(line[:-1]):
                        self.add_message("empty-comment", line=line_num + 1)


def register(linter: "PyLinter") -> None:
    linter.register_checker(CommentChecker(linter))
```
### 4 - pylint/checkers/stdlib.py:

Start line: 348, End line: 464

```python
class StdlibChecker(DeprecatedMixin, BaseChecker):
    __implements__ = (IAstroidChecker,)
    name = "stdlib"

    msgs = {
        "W1501": (
            '"%s" is not a valid mode for open.',
            "bad-open-mode",
            "Python supports: r, w, a[, x] modes with b, +, "
            "and U (only with r) options. "
            "See https://docs.python.org/2/library/functions.html#open",
        ),
        "W1502": (
            "Using datetime.time in a boolean context.",
            "boolean-datetime",
            "Using datetime.time in a boolean context can hide "
            "subtle bugs when the time they represent matches "
            "midnight UTC. This behaviour was fixed in Python 3.5. "
            "See https://bugs.python.org/issue13936 for reference.",
            {"maxversion": (3, 5)},
        ),
        "W1503": (
            "Redundant use of %s with constant value %r",
            "redundant-unittest-assert",
            "The first argument of assertTrue and assertFalse is "
            "a condition. If a constant is passed as parameter, that "
            "condition will be always true. In this case a warning "
            "should be emitted.",
        ),
        "W1505": (
            "Using deprecated method %s()",
            "deprecated-method",
            "The method is marked as deprecated and will be removed in "
            "a future version of Python. Consider looking for an "
            "alternative in the documentation.",
        ),
        "W1506": (
            "threading.Thread needs the target function",
            "bad-thread-instantiation",
            "The warning is emitted when a threading.Thread class "
            "is instantiated without the target function being passed. "
            "By default, the first parameter is the group param, not the target param.",
        ),
        "W1507": (
            "Using copy.copy(os.environ). Use os.environ.copy() instead. ",
            "shallow-copy-environ",
            "os.environ is not a dict object but proxy object, so "
            "shallow copy has still effects on original object. "
            "See https://bugs.python.org/issue15373 for reference.",
        ),
        "E1507": (
            "%s does not support %s type argument",
            "invalid-envvar-value",
            "Env manipulation functions support only string type arguments. "
            "See https://docs.python.org/3/library/os.html#os.getenv.",
        ),
        "W1508": (
            "%s default type is %s. Expected str or None.",
            "invalid-envvar-default",
            "Env manipulation functions return None or str values. "
            "Supplying anything different as a default may cause bugs. "
            "See https://docs.python.org/3/library/os.html#os.getenv.",
        ),
        "W1509": (
            "Using preexec_fn keyword which may be unsafe in the presence "
            "of threads",
            "subprocess-popen-preexec-fn",
            "The preexec_fn parameter is not safe to use in the presence "
            "of threads in your application. The child process could "
            "deadlock before exec is called. If you must use it, keep it "
            "trivial! Minimize the number of libraries you call into."
            "https://docs.python.org/3/library/subprocess.html#popen-constructor",
        ),
        "W1510": (
            "Using subprocess.run without explicitly set `check` is not recommended.",
            "subprocess-run-check",
            "The check parameter should always be used with explicitly set "
            "`check` keyword to make clear what the error-handling behavior is."
            "https://docs.python.org/3/library/subprocess.html#subprocess.run",
        ),
        "W1511": (
            "Using deprecated argument %s of method %s()",
            "deprecated-argument",
            "The argument is marked as deprecated and will be removed in the future.",
        ),
        "W1512": (
            "Using deprecated class %s of module %s",
            "deprecated-class",
            "The class is marked as deprecated and will be removed in the future.",
        ),
        "W1513": (
            "Using deprecated decorator %s()",
            "deprecated-decorator",
            "The decorator is marked as deprecated and will be removed in the future.",
        ),
        "W1514": (
            "Using open without explicitly specifying an encoding",
            "unspecified-encoding",
            "It is better to specify an encoding when opening documents. "
            "Using the system default implicitly can create problems on other operating systems. "
            "See https://www.python.org/dev/peps/pep-0597/",
        ),
        "W1515": (
            "Leaving functions creating breakpoints in production code is not recommended",
            "forgotten-debug-statement",
            "Calls to breakpoint(), sys.breakpointhook() and pdb.set_trace() should be removed "
            "from code that is not actively being debugged.",
        ),
        "W1516": (
            "'lru_cache' without 'maxsize' will keep all method args alive indefinitely, including 'self'",
            "lru-cache-decorating-method",
            "By decorating a method with lru_cache the 'self' argument will be linked to "
            "the lru_cache function and therefore never garbage collected. Unless your instance "
            "will never need to be garbage collected (singleton) it is recommended to refactor "
            "code to avoid this pattern or add a maxsize to the cache.",
        ),
    }
```
### 5 - pylint/lint/pylinter.py:

Start line: 807, End line: 835

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
            if msgcat in {"E", "F"}:
                for msgid in msgids:
                    self.enable(msgid)
            else:
                for msgid in msgids:
                    self.disable(msgid)

    def disable_reporters(self):
        """Disable all reporters."""
        for _reporters in self._reports.values():
            for report_id, _, _ in _reporters:
                self.disable_report(report_id)

    def error_mode(self):
        """Error mode: enable only errors; no reports, no persistent."""
        self._error_mode = True
        self.disable_noerror_messages()
        self.disable("miscellaneous")
        self.set_option("reports", False)
        self.set_option("persistent", False)
        self.set_option("score", False)
```
### 6 - pylint/checkers/similar.py:

Start line: 729, End line: 802

```python
# wrapper to get a pylint checker from the similar class
class SimilarChecker(BaseChecker, Similar, MapReduceMixin):
    """Checks for similarities and duplicated code. This computation may be
    memory / CPU intensive, so you should disable it if you experiment some
    problems.
    """

    __implements__ = (IRawChecker,)
    # configuration section name
    name = "similarities"
    # messages
    msgs = MSGS
    # configuration options
    # for available dict keys/values see the optik parser 'add_option' method
    options = (
        (
            "min-similarity-lines",
            {
                "default": DEFAULT_MIN_SIMILARITY_LINE,
                "type": "int",
                "metavar": "<int>",
                "help": "Minimum lines number of a similarity.",
            },
        ),
        (
            "ignore-comments",
            {
                "default": True,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Comments are removed from the similarity computation",
            },
        ),
        (
            "ignore-docstrings",
            {
                "default": True,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Docstrings are removed from the similarity computation",
            },
        ),
        (
            "ignore-imports",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Imports are removed from the similarity computation",
            },
        ),
        (
            "ignore-signatures",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Signatures are removed from the similarity computation",
            },
        ),
    )
    # reports
    reports = (("RP0801", "Duplication", report_similarities),)

    def __init__(self, linter=None) -> None:
        BaseChecker.__init__(self, linter)
        Similar.__init__(
            self,
            min_lines=self.config.min_similarity_lines,
            ignore_comments=self.config.ignore_comments,
            ignore_docstrings=self.config.ignore_docstrings,
            ignore_imports=self.config.ignore_imports,
            ignore_signatures=self.config.ignore_signatures,
        )
```
### 7 - pylint/checkers/design_analysis.py:

Start line: 114, End line: 192

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
        "typing.AsyncContextManager",
        "typing.Hashable",
        "typing.Sized",
    )
)
```
### 8 - pylint/extensions/bad_builtin.py:

Start line: 31, End line: 57

```python
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
### 9 - pylint/lint/pylinter.py:

Start line: 97, End line: 187

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
        "Bad option value %r",
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
}
```
### 10 - pylint/lint/pylinter.py:

Start line: 837, End line: 858

```python
class PyLinter(
    config.OptionsManagerMixIn,
    reporters.ReportsHandlerMixIn,
    checkers.BaseTokenChecker,
):

    def list_messages_enabled(self):
        emittable, non_emittable = self.msgs_store.find_emittable_messages()
        enabled = []
        disabled = []
        for message in emittable:
            if self.is_message_enabled(message.msgid):
                enabled.append(f"  {message.symbol} ({message.msgid})")
            else:
                disabled.append(f"  {message.symbol} ({message.msgid})")
        print("Enabled messages:")
        for msg in enabled:
            print(msg)
        print("\nDisabled messages:")
        for msg in disabled:
            print(msg)
        print("\nNon-emittable messages with current interpreter:")
        for msg in non_emittable:
            print(f"  {msg.symbol} ({msg.msgid})")
        print("")

    # block level option handling #############################################
    # see func_block_disable_msg.py test case for expected behaviour
```
### 12 - pylint/checkers/similar.py:

Start line: 707, End line: 726

```python
MSGS = {
    "R0801": (
        "Similar lines in %s files\n%s",
        "duplicate-code",
        "Indicates that a set of similar lines has been detected "
        "among multiple file. This usually means that the code should "
        "be refactored to avoid this duplication.",
    )
}


def report_similarities(
    sect,
    stats: LinterStats,
    old_stats: Optional[LinterStats],
) -> None:
    """Make a layout with some stats about duplication."""
    lines = ["", "now", "previous", "difference"]
    lines += table_lines_from_stats(stats, old_stats, "duplicated_lines")
    sect.append(Table(children=lines, cols=4, rheaders=1, cheaders=1))
```
### 13 - pylint/checkers/similar.py:

Start line: 345, End line: 371

```python
class Commonality(NamedTuple):
    cmn_lines_nb: int
    fst_lset: "LineSet"
    fst_file_start: LineNumber
    fst_file_end: LineNumber
    snd_lset: "LineSet"
    snd_file_start: LineNumber
    snd_file_end: LineNumber


class Similar:
    """Finds copy-pasted lines of code in a project."""

    def __init__(
        self,
        min_lines: int = DEFAULT_MIN_SIMILARITY_LINE,
        ignore_comments: bool = False,
        ignore_docstrings: bool = False,
        ignore_imports: bool = False,
        ignore_signatures: bool = False,
    ) -> None:
        self.min_lines = min_lines
        self.ignore_comments = ignore_comments
        self.ignore_docstrings = ignore_docstrings
        self.ignore_imports = ignore_imports
        self.ignore_signatures = ignore_signatures
        self.linesets: List["LineSet"] = []
```
### 55 - pylint/checkers/similar.py:

Start line: 865, End line: 882

```python
class SimilarChecker(BaseChecker, Similar, MapReduceMixin):

    def get_map_data(self):
        """Passthru override."""
        return Similar.get_map_data(self)

    def reduce_map_data(self, linter, data):
        """Reduces and recombines data into a format that we can report on.

        The partner function of get_map_data()
        """
        recombined = SimilarChecker(linter)
        recombined.min_lines = self.min_lines
        recombined.ignore_comments = self.ignore_comments
        recombined.ignore_docstrings = self.ignore_docstrings
        recombined.ignore_imports = self.ignore_imports
        recombined.ignore_signatures = self.ignore_signatures
        recombined.open()
        Similar.combine_mapreduce_data(recombined, linesets_collection=data)
        recombined.close()
```
### 81 - pylint/checkers/similar.py:

Start line: 44, End line: 108

```python
import copy
import functools
import itertools
import operator
import re
import sys
import warnings
from collections import defaultdict
from getopt import getopt
from io import BufferedIOBase, BufferedReader, BytesIO
from itertools import chain, groupby
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

import astroid
from astroid import nodes

from pylint.checkers import BaseChecker, MapReduceMixin, table_lines_from_stats
from pylint.interfaces import IRawChecker
from pylint.reporters.ureports.nodes import Table
from pylint.utils import LinterStats, decoding_stream

if TYPE_CHECKING:
    from pylint.lint import PyLinter

DEFAULT_MIN_SIMILARITY_LINE = 4

REGEX_FOR_LINES_WITH_CONTENT = re.compile(r".*\w+")

# Index defines a location in a LineSet stripped lines collection
Index = NewType("Index", int)

# LineNumber defines a location in a LinesSet real lines collection (the whole file lines)
LineNumber = NewType("LineNumber", int)


# LineSpecifs holds characteristics of a line in a file
class LineSpecifs(NamedTuple):
    line_number: LineNumber
    text: str


# Links LinesChunk object to the starting indices (in lineset's stripped lines)
# of the different chunk of lines that are used to compute the hash
HashToIndex_T = Dict["LinesChunk", List[Index]]

# Links index in the lineset's stripped lines to the real lines in the file
IndexToLines_T = Dict[Index, "SuccessiveLinesLimits"]

# The types the streams read by pylint can take. Originating from astroid.nodes.Module.stream() and open()
STREAM_TYPES = Union[TextIO, BufferedReader, BytesIO]
```
