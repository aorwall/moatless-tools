# pylint-dev__pylint-4175

| **pylint-dev/pylint** | `ae6cbd1062c0a8e68d32a5cdc67c993da26d0f4a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/pylint/lint/parallel.py b/pylint/lint/parallel.py
--- a/pylint/lint/parallel.py
+++ b/pylint/lint/parallel.py
@@ -160,7 +160,7 @@ def check_parallel(linter, jobs, files, arguments=None):
         pool.join()
 
     _merge_mapreduce_data(linter, all_mapreduce_data)
-    linter.stats = _merge_stats(all_stats)
+    linter.stats = _merge_stats([linter.stats] + all_stats)
 
     # Insert stats data to local checkers.
     for checker in linter.get_checkers():

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/lint/parallel.py | 163 | 163 | - | - | -


## Problem Statement

```
Pylint 2.7.0 seems to ignore the min-similarity-lines setting
<!--
  Hi there! Thank you for discovering and submitting an issue.

  Before you submit this, make sure that the issue doesn't already exist
  or if it is not closed.

  Is your issue fixed on the preview release?: pip install pylint astroid --pre -U

-->

### Steps to reproduce
1. Have two Python source files that share 8 common lines 
2. Have min-similarity-lines=40 in the pylint config
3. Run pylint 2.7.0 on the source files

### Current behavior

Before pylint 2.7.0, the min-similarity-lines setting was honored and caused shorter similar lines to be accepted.

Starting with pylint 2.7.0, the min-similarity-lines setting seems to be ignored and the common lines are always reported as an issue R0801, even when the min-similarity-lines setting is significantly larger than the number of common lines.

### Expected behavior

The min-similarity-lines setting should be respected again as it was before pylint 2.7.0.

### pylint --version output

pylint 2.7.0
astroid 2.5
Python 3.9.1 (default, Feb  1 2021, 20:41:56) 
[Clang 12.0.0 (clang-1200.0.32.29)]

Pylint 2.7.0 seems to ignore the min-similarity-lines setting
<!--
  Hi there! Thank you for discovering and submitting an issue.

  Before you submit this, make sure that the issue doesn't already exist
  or if it is not closed.

  Is your issue fixed on the preview release?: pip install pylint astroid --pre -U

-->

### Steps to reproduce
1. Have two Python source files that share 8 common lines 
2. Have min-similarity-lines=40 in the pylint config
3. Run pylint 2.7.0 on the source files

### Current behavior

Before pylint 2.7.0, the min-similarity-lines setting was honored and caused shorter similar lines to be accepted.

Starting with pylint 2.7.0, the min-similarity-lines setting seems to be ignored and the common lines are always reported as an issue R0801, even when the min-similarity-lines setting is significantly larger than the number of common lines.

### Expected behavior

The min-similarity-lines setting should be respected again as it was before pylint 2.7.0.

### pylint --version output

pylint 2.7.0
astroid 2.5
Python 3.9.1 (default, Feb  1 2021, 20:41:56) 
[Clang 12.0.0 (clang-1200.0.32.29)]


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/constants.py | 3 | 54| 381 | 381 | 422 | 
| 2 | 2 pylint/lint/pylinter.py | 56 | 136| 690 | 1071 | 10394 | 
| 3 | 3 pylint/checkers/design_analysis.py | 26 | 103| 668 | 1739 | 15186 | 
| 4 | 4 pylint/checkers/similar.py | 43 | 95| 316 | 2055 | 23071 | 
| 5 | 5 pylint/checkers/stdlib.py | 40 | 115| 786 | 2841 | 29138 | 
| 6 | 6 pylint/checkers/format.py | 53 | 118| 345 | 3186 | 35977 | 
| 7 | 7 pylint/__init__.py | 12 | 46| 161 | 3347 | 36677 | 
| 8 | 8 pylint/checkers/python3.py | 43 | 111| 375 | 3722 | 48123 | 
| 9 | 8 pylint/lint/pylinter.py | 4 | 54| 331 | 4053 | 48123 | 
| 10 | 9 pylint/extensions/bad_builtin.py | 12 | 50| 261 | 4314 | 48770 | 
| 11 | 10 pylint/checkers/refactoring/recommendation_checker.py | 3 | 58| 465 | 4779 | 51234 | 
| 12 | 10 pylint/checkers/design_analysis.py | 104 | 182| 511 | 5290 | 51234 | 
| 13 | 11 pylint/reporters/text.py | 25 | 59| 210 | 5500 | 53356 | 
| 14 | 12 pylint/lint/__init__.py | 74 | 106| 170 | 5670 | 54847 | 
| 15 | 13 pylint/checkers/refactoring/len_checker.py | 3 | 51| 305 | 5975 | 55801 | 
| 16 | 14 pylint/checkers/refactoring/not_checker.py | 5 | 42| 296 | 6271 | 56439 | 
| 17 | 15 pylint/extensions/broad_try_clause.py | 13 | 78| 412 | 6683 | 57045 | 
| 18 | 16 pylint/checkers/newstyle.py | 24 | 136| 782 | 7465 | 58280 | 
| 19 | 17 pylint/pyreverse/main.py | 23 | 174| 799 | 8264 | 59859 | 
| 20 | 18 pylint/checkers/classes.py | 48 | 92| 270 | 8534 | 78609 | 
| 21 | 19 pylint/checkers/refactoring/refactoring_checker.py | 4 | 54| 354 | 8888 | 94678 | 
| 22 | 20 pylint/checkers/logging.py | 26 | 108| 674 | 9562 | 98122 | 
| 23 | 21 pylint/checkers/strings.py | 37 | 84| 327 | 9889 | 106453 | 
| 24 | 21 pylint/checkers/similar.py | 341 | 367| 202 | 10091 | 106453 | 
| 25 | 22 pylint/utils/__init__.py | 46 | 91| 236 | 10327 | 107559 | 
| 26 | 22 pylint/checkers/format.py | 661 | 695| 296 | 10623 | 107559 | 
| 27 | 22 pylint/checkers/similar.py | 557 | 636| 636 | 11259 | 107559 | 
| 28 | 23 pylint/checkers/base.py | 940 | 1082| 1329 | 12588 | 129603 | 
| 29 | 24 pylint/extensions/confusing_elif.py | 8 | 58| 450 | 13038 | 130163 | 
| 30 | 24 pylint/checkers/format.py | 724 | 772| 459 | 13497 | 130163 | 
| 31 | 24 pylint/checkers/similar.py | 691 | 708| 157 | 13654 | 130163 | 
| 32 | 25 pylint/extensions/typing.py | 72 | 166| 762 | 14416 | 132889 | 
| 33 | 26 pylint/lint/run.py | 4 | 477| 339 | 14755 | 136386 | 
| 34 | 26 pylint/checkers/similar.py | 711 | 785| 498 | 15253 | 136386 | 
| 35 | 26 pylint/checkers/stdlib.py | 118 | 257| 1032 | 16285 | 136386 | 
| 36 | 26 pylint/checkers/python3.py | 1166 | 1249| 701 | 16986 | 136386 | 
| 37 | 27 pylint/checkers/imports.py | 199 | 294| 769 | 17755 | 144920 | 
| 38 | 27 pylint/checkers/stdlib.py | 335 | 437| 1021 | 18776 | 144920 | 
| 39 | 27 pylint/lint/run.py | 297 | 404| 996 | 19772 | 144920 | 
| 40 | 28 pylint/config/__init__.py | 35 | 90| 401 | 20173 | 146319 | 
| 41 | 28 pylint/checkers/base.py | 65 | 96| 221 | 20394 | 146319 | 
| 42 | 28 pylint/checkers/refactoring/refactoring_checker.py | 610 | 625| 148 | 20542 | 146319 | 
| 43 | 28 pylint/lint/pylinter.py | 753 | 775| 284 | 20826 | 146319 | 
| 44 | 29 pylint/extensions/overlapping_exceptions.py | 6 | 87| 576 | 21402 | 146942 | 
| 45 | 29 pylint/checkers/refactoring/refactoring_checker.py | 752 | 829| 575 | 21977 | 146942 | 
| 46 | 29 pylint/lint/run.py | 56 | 296| 1587 | 23564 | 146942 | 
| 47 | 29 pylint/checkers/refactoring/refactoring_checker.py | 1219 | 1232| 159 | 23723 | 146942 | 
| 48 | 30 pylint/extensions/code_style.py | 1 | 60| 485 | 24208 | 148200 | 
| 49 | 31 pylint/checkers/async.py | 12 | 53| 334 | 24542 | 149034 | 
| 50 | 31 pylint/checkers/python3.py | 1113 | 1137| 219 | 24761 | 149034 | 
| 51 | 31 pylint/checkers/base.py | 1661 | 1701| 245 | 25006 | 149034 | 
| 52 | 31 pylint/checkers/similar.py | 309 | 338| 345 | 25351 | 149034 | 
| 53 | 32 pylint/checkers/typecheck.py | 732 | 872| 1021 | 26372 | 165267 | 
| 54 | 33 pylint/epylint.py | 136 | 205| 569 | 26941 | 167197 | 
| 55 | 33 pylint/checkers/refactoring/refactoring_checker.py | 1140 | 1181| 364 | 27305 | 167197 | 
| 56 | 34 pylint/checkers/variables.py | 54 | 156| 639 | 27944 | 184421 | 
| 57 | 34 pylint/lint/pylinter.py | 712 | 731| 196 | 28140 | 184421 | 
| 58 | 34 pylint/checkers/format.py | 523 | 543| 212 | 28352 | 184421 | 
| 59 | 34 pylint/checkers/refactoring/refactoring_checker.py | 417 | 469| 393 | 28745 | 184421 | 
| 60 | 34 pylint/checkers/base.py | 460 | 567| 937 | 29682 | 184421 | 
| 61 | 35 pylint/checkers/utils.py | 54 | 115| 366 | 30048 | 196706 | 
| 62 | 35 pylint/checkers/stdlib.py | 439 | 483| 478 | 30526 | 196706 | 
| 63 | 35 pylint/checkers/python3.py | 1139 | 1164| 203 | 30729 | 196706 | 
| 64 | 36 pylint/testutils/output_line.py | 4 | 23| 156 | 30885 | 197384 | 
| 65 | 37 pylint/extensions/check_elif.py | 13 | 80| 484 | 31369 | 198092 | 
| 66 | 37 pylint/checkers/typecheck.py | 58 | 124| 385 | 31754 | 198092 | 
| 67 | 38 pylint/checkers/raw_metrics.py | 17 | 44| 257 | 32011 | 199211 | 


## Missing Patch Files

 * 1: pylint/lint/parallel.py

### Hint

```
We are seeing the same problem. All of our automated builds failed this morning.

\`\`\` bash
$ pylint --version
pylint 2.7.0
astroid 2.5
Python 3.7.7 (tags/v3.7.7:d7c567b08f, Mar 10 2020, 10:41:24) [MSC v.1900 64 bit (AMD64)]
\`\`\`

**Workaround**

Reverting back to pylint 2.6.2.


My projects pass if I disable `--jobs=N` in my CI/CD (for projects with `min_similarity_lines` greater than the default).

This suggests that the `min-similarity-lines`/`min_similarity_lines` option isn't being passed to sub-workers by the `check_parallel` code-path.

As an aside, when I last did a profile of multi-job vs single-job runs, there was no wall-clock gain, but much higher CPU use (aka cost), so I would advice *not* using `--jobs=>2`.
We have circumvented the problem by excluding R0801 in the "disable" setting of the pylintrc file. We needed to specify the issue by number - specifying it by name did not work.

We run with jobs=4, BTW.
I'm not sure if this is related, but when running pylint 2.7.1 with --jobs=0, pylint reports duplicate lines, but in the summary and final score, duplicate lines are not reported. For example:
\`\`\`
pylint --jobs=0 --reports=y test_duplicate.py test_duplicate_2.py 
************* Module test_duplicate
test_duplicate.py:1:0: R0801: Similar lines in 2 files
==test_duplicate:1
==test_duplicate_2:1
for i in range(10):
    print(i)
    print(i+1)
    print(i+2)
    print(i+3) (duplicate-code)


Report
======
10 statements analysed.

Statistics by type
------------------

+---------+-------+-----------+-----------+------------+---------+
|type     |number |old number |difference |%documented |%badname |
+=========+=======+===========+===========+============+=========+
|module   |2      |NC         |NC         |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+
|class    |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|method   |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|function |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+



Raw metrics
-----------

+----------+-------+------+---------+-----------+
|type      |number |%     |previous |difference |
+==========+=======+======+=========+===========+
|code      |14     |87.50 |NC       |NC         |
+----------+-------+------+---------+-----------+
|docstring |2      |12.50 |NC       |NC         |
+----------+-------+------+---------+-----------+
|comment   |0      |0.00  |NC       |NC         |
+----------+-------+------+---------+-----------+
|empty     |0      |0.00  |NC       |NC         |
+----------+-------+------+---------+-----------+



Duplication
-----------

+-------------------------+------+---------+-----------+
|                         |now   |previous |difference |
+=========================+======+=========+===========+
|nb duplicated lines      |0     |NC       |NC         |
+-------------------------+------+---------+-----------+
|percent duplicated lines |0.000 |NC       |NC         |
+-------------------------+------+---------+-----------+



Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|refactor   |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|warning    |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|error      |0      |NC       |NC         |
+-----------+-------+---------+-----------+




--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
\`\`\`
It looks like it's due to:
\`\`\`
@classmethod
def reduce_map_data(cls, linter, data):
    """Reduces and recombines data into a format that we can report on

    The partner function of get_map_data()"""
    recombined = SimilarChecker(linter)
    recombined.open()
    Similar.combine_mapreduce_data(recombined, linesets_collection=data)
    recombined.close()
\`\`\`

the `SimilarChecker` instance created gets default values, not values from config. I double checked by trying to fix it:

\`\`\`--- a/pylint/checkers/similar.py
+++ b/pylint/checkers/similar.py
@@ -428,6 +428,8 @@ class SimilarChecker(BaseChecker, Similar, MapReduceMixin):
 
         The partner function of get_map_data()"""
         recombined = SimilarChecker(linter)
+        checker = [c for c in linter.get_checkers() if c.name == cls.name][0]
+        recombined.min_lines = checker.min_lines
         recombined.open()
         Similar.combine_mapreduce_data(recombined, linesets_collection=data)
         recombined.close()
\`\`\`

by simply copying the `min_lines` attribute from the "root" checker in the `recombined` checker. I bet this is not the proper way to do it, but I'm not familiar with pylint codebase.


We are seeing the same problem. All of our automated builds failed this morning.

\`\`\` bash
$ pylint --version
pylint 2.7.0
astroid 2.5
Python 3.7.7 (tags/v3.7.7:d7c567b08f, Mar 10 2020, 10:41:24) [MSC v.1900 64 bit (AMD64)]
\`\`\`

**Workaround**

Reverting back to pylint 2.6.2.


My projects pass if I disable `--jobs=N` in my CI/CD (for projects with `min_similarity_lines` greater than the default).

This suggests that the `min-similarity-lines`/`min_similarity_lines` option isn't being passed to sub-workers by the `check_parallel` code-path.

As an aside, when I last did a profile of multi-job vs single-job runs, there was no wall-clock gain, but much higher CPU use (aka cost), so I would advice *not* using `--jobs=>2`.
We have circumvented the problem by excluding R0801 in the "disable" setting of the pylintrc file. We needed to specify the issue by number - specifying it by name did not work.

We run with jobs=4, BTW.
I'm not sure if this is related, but when running pylint 2.7.1 with --jobs=0, pylint reports duplicate lines, but in the summary and final score, duplicate lines are not reported. For example:
\`\`\`
pylint --jobs=0 --reports=y test_duplicate.py test_duplicate_2.py 
************* Module test_duplicate
test_duplicate.py:1:0: R0801: Similar lines in 2 files
==test_duplicate:1
==test_duplicate_2:1
for i in range(10):
    print(i)
    print(i+1)
    print(i+2)
    print(i+3) (duplicate-code)


Report
======
10 statements analysed.

Statistics by type
------------------

+---------+-------+-----------+-----------+------------+---------+
|type     |number |old number |difference |%documented |%badname |
+=========+=======+===========+===========+============+=========+
|module   |2      |NC         |NC         |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+
|class    |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|method   |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|function |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+



Raw metrics
-----------

+----------+-------+------+---------+-----------+
|type      |number |%     |previous |difference |
+==========+=======+======+=========+===========+
|code      |14     |87.50 |NC       |NC         |
+----------+-------+------+---------+-----------+
|docstring |2      |12.50 |NC       |NC         |
+----------+-------+------+---------+-----------+
|comment   |0      |0.00  |NC       |NC         |
+----------+-------+------+---------+-----------+
|empty     |0      |0.00  |NC       |NC         |
+----------+-------+------+---------+-----------+



Duplication
-----------

+-------------------------+------+---------+-----------+
|                         |now   |previous |difference |
+=========================+======+=========+===========+
|nb duplicated lines      |0     |NC       |NC         |
+-------------------------+------+---------+-----------+
|percent duplicated lines |0.000 |NC       |NC         |
+-------------------------+------+---------+-----------+



Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|refactor   |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|warning    |0      |NC       |NC         |
+-----------+-------+---------+-----------+
|error      |0      |NC       |NC         |
+-----------+-------+---------+-----------+




--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
\`\`\`
It looks like it's due to:
\`\`\`
@classmethod
def reduce_map_data(cls, linter, data):
    """Reduces and recombines data into a format that we can report on

    The partner function of get_map_data()"""
    recombined = SimilarChecker(linter)
    recombined.open()
    Similar.combine_mapreduce_data(recombined, linesets_collection=data)
    recombined.close()
\`\`\`

the `SimilarChecker` instance created gets default values, not values from config. I double checked by trying to fix it:

\`\`\`--- a/pylint/checkers/similar.py
+++ b/pylint/checkers/similar.py
@@ -428,6 +428,8 @@ class SimilarChecker(BaseChecker, Similar, MapReduceMixin):
 
         The partner function of get_map_data()"""
         recombined = SimilarChecker(linter)
+        checker = [c for c in linter.get_checkers() if c.name == cls.name][0]
+        recombined.min_lines = checker.min_lines
         recombined.open()
         Similar.combine_mapreduce_data(recombined, linesets_collection=data)
         recombined.close()
\`\`\`

by simply copying the `min_lines` attribute from the "root" checker in the `recombined` checker. I bet this is not the proper way to do it, but I'm not familiar with pylint codebase.


```

## Patch

```diff
diff --git a/pylint/lint/parallel.py b/pylint/lint/parallel.py
--- a/pylint/lint/parallel.py
+++ b/pylint/lint/parallel.py
@@ -160,7 +160,7 @@ def check_parallel(linter, jobs, files, arguments=None):
         pool.join()
 
     _merge_mapreduce_data(linter, all_mapreduce_data)
-    linter.stats = _merge_stats(all_stats)
+    linter.stats = _merge_stats([linter.stats] + all_stats)
 
     # Insert stats data to local checkers.
     for checker in linter.get_checkers():

```

## Test Patch

```diff
diff --git a/tests/test_check_parallel.py b/tests/test_check_parallel.py
--- a/tests/test_check_parallel.py
+++ b/tests/test_check_parallel.py
@@ -67,6 +67,68 @@ def process_module(self, _astroid):
         self.data.append(record)
 
 
+class ParallelTestChecker(BaseChecker):
+    """A checker that does need to consolidate data.
+
+    To simulate the need to consolidate data, this checker only
+    reports a message for pairs of files.
+
+    On non-parallel builds: it works on all the files in a single run.
+
+    On parallel builds: lint.parallel calls ``open`` once per file.
+
+    So if files are treated by separate processes, no messages will be
+    raised from the individual process, all messages will be raised
+    from reduce_map_data.
+    """
+
+    __implements__ = (pylint.interfaces.IRawChecker,)
+
+    name = "parallel-checker"
+    test_data = "parallel"
+    msgs = {
+        "R9999": (
+            "Test %s",
+            "parallel-test-check",
+            "Some helpful text.",
+        )
+    }
+
+    def __init__(self, linter, *args, **kwargs):
+        super().__init__(linter, *args, **kwargs)
+        self.data = []
+        self.linter = linter
+        self.stats = None
+
+    def open(self):
+        """init the checkers: reset statistics information"""
+        self.stats = self.linter.add_stats()
+        self.data = []
+
+    def close(self):
+        for _ in self.data[1::2]:  # Work on pairs of files, see class docstring.
+            self.add_message("R9999", args=("From process_module, two files seen.",))
+
+    def get_map_data(self):
+        return self.data
+
+    def reduce_map_data(self, linter, data):
+        recombined = type(self)(linter)
+        recombined.open()
+        aggregated = []
+        for d in data:
+            aggregated.extend(d)
+        for _ in aggregated[1::2]:  # Work on pairs of files, see class docstring.
+            self.add_message("R9999", args=("From reduce_map_data",))
+        recombined.close()
+
+    def process_module(self, _astroid):
+        """Called once per stream/file/astroid object"""
+        # record the number of invocations with the data object
+        record = self.test_data + str(len(self.data))
+        self.data.append(record)
+
+
 class ExtraSequentialTestChecker(SequentialTestChecker):
     """A checker that does not need to consolidate data across run invocations"""
 
@@ -74,6 +136,13 @@ class ExtraSequentialTestChecker(SequentialTestChecker):
     test_data = "extra-sequential"
 
 
+class ExtraParallelTestChecker(ParallelTestChecker):
+    """A checker that does need to consolidate data across run invocations"""
+
+    name = "extra-parallel-checker"
+    test_data = "extra-parallel"
+
+
 class ThirdSequentialTestChecker(SequentialTestChecker):
     """A checker that does not need to consolidate data across run invocations"""
 
@@ -81,6 +150,13 @@ class ThirdSequentialTestChecker(SequentialTestChecker):
     test_data = "third-sequential"
 
 
+class ThirdParallelTestChecker(ParallelTestChecker):
+    """A checker that does need to consolidate data across run invocations"""
+
+    name = "third-parallel-checker"
+    test_data = "third-parallel"
+
+
 class TestCheckParallelFramework:
     """Tests the check_parallel() function's framework"""
 
@@ -402,3 +478,69 @@ def test_compare_workers_to_single_proc(self, num_files, num_jobs, num_checkers)
         assert (
             stats_check_parallel == expected_stats
         ), "The lint is returning unexpected results, has something changed?"
+
+    @pytest.mark.parametrize(
+        "num_files,num_jobs,num_checkers",
+        [
+            (2, 2, 1),
+            (2, 2, 2),
+            (2, 2, 3),
+            (3, 2, 1),
+            (3, 2, 2),
+            (3, 2, 3),
+            (3, 1, 1),
+            (3, 1, 2),
+            (3, 1, 3),
+            (3, 5, 1),
+            (3, 5, 2),
+            (3, 5, 3),
+            (10, 2, 1),
+            (10, 2, 2),
+            (10, 2, 3),
+            (2, 10, 1),
+            (2, 10, 2),
+            (2, 10, 3),
+        ],
+    )
+    def test_map_reduce(self, num_files, num_jobs, num_checkers):
+        """Compares the 3 key parameters for check_parallel() produces the same results
+
+        The intent here is to validate the reduce step: no stats should be lost.
+
+        Checks regression of https://github.com/PyCQA/pylint/issues/4118
+        """
+
+        # define the stats we expect to get back from the runs, these should only vary
+        # with the number of files.
+        file_infos = _gen_file_datas(num_files)
+
+        # Loop for single-proc and mult-proc so we can ensure the same linter-config
+        for do_single_proc in range(2):
+            linter = PyLinter(reporter=Reporter())
+
+            # Assign between 1 and 3 checkers to the linter, they should not change the
+            # results of the lint
+            linter.register_checker(ParallelTestChecker(linter))
+            if num_checkers > 1:
+                linter.register_checker(ExtraParallelTestChecker(linter))
+            if num_checkers > 2:
+                linter.register_checker(ThirdParallelTestChecker(linter))
+
+            if do_single_proc:
+                # establish the baseline
+                assert (
+                    linter.config.jobs == 1
+                ), "jobs>1 are ignored when calling _check_files"
+                linter._check_files(linter.get_ast, file_infos)
+                stats_single_proc = linter.stats
+            else:
+                check_parallel(
+                    linter,
+                    jobs=num_jobs,
+                    files=file_infos,
+                    arguments=None,
+                )
+                stats_check_parallel = linter.stats
+        assert (
+            stats_single_proc["by_msg"] == stats_check_parallel["by_msg"]
+        ), "Single-proc and check_parallel() should return the same thing"

```


## Code snippets

### 1 - pylint/constants.py:

Start line: 3, End line: 54

```python
import builtins
import platform
import sys

import astroid

from pylint.__pkginfo__ import __version__

BUILTINS = builtins.__name__
PY38_PLUS = sys.version_info[:2] >= (3, 8)
PY39_PLUS = sys.version_info[:2] >= (3, 9)
PY310_PLUS = sys.version_info[:2] >= (3, 10)

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
### 2 - pylint/lint/pylinter.py:

Start line: 56, End line: 136

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
### 3 - pylint/checkers/design_analysis.py:

Start line: 26, End line: 103

```python
import re
from collections import defaultdict
from typing import FrozenSet, List, Set, cast

import astroid
from astroid import nodes

from pylint import utils
from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages
from pylint.interfaces import IAstroidChecker

MSGS = {  # pylint: disable=consider-using-namedtuple-or-dataclass
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
### 4 - pylint/checkers/similar.py:

Start line: 43, End line: 95

```python
import copy
import functools
import itertools
import operator
import re
import sys
from collections import defaultdict
from getopt import getopt
from io import TextIOWrapper
from itertools import chain, groupby
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Set,
    Tuple,
)

import astroid

from pylint.checkers import BaseChecker, MapReduceMixin, table_lines_from_stats
from pylint.interfaces import IRawChecker
from pylint.reporters.ureports.nodes import Table
from pylint.utils import decoding_stream

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
# of the different chunk of lines that are used to compute the hash
HashToIndex_T = Dict["LinesChunk", List[Index]]

# Links index in the lineset's stripped lines to the real lines in the file
IndexToLines_T = Dict[Index, "SuccessiveLinesLimits"]
```
### 5 - pylint/checkers/stdlib.py:

Start line: 40, End line: 115

```python
import sys
from collections.abc import Iterable

import astroid

from pylint.checkers import BaseChecker, DeprecatedMixin, utils
from pylint.interfaces import IAstroidChecker

OPEN_FILES_MODE = ("open", "file")
OPEN_FILES_ENCODING = ("open",)
UNITTEST_CASE = "unittest.case"
THREADING_THREAD = "threading.Thread"
COPY_COPY = "copy.copy"
OS_ENVIRON = "os._Environ"
ENV_GETTERS = ("os.getenv",)
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

DEPRECATED_DECORATORS = {
    (3, 8, 0): {"asyncio.coroutine"},
    (3, 3, 0): {
        "abc.abstractclassmethod",
        "abc.abstractstaticmethod",
        "abc.abstractproperty",
    },
}
```
### 6 - pylint/checkers/format.py:

Start line: 53, End line: 118

```python
import tokenize
from functools import reduce  # pylint: disable=redefined-builtin
from typing import List

from astroid import nodes

from pylint.checkers import BaseTokenChecker
from pylint.checkers.utils import (
    check_messages,
    is_overload_stub,
    is_protocol_class,
    node_frame_class,
)
from pylint.constants import WarningScope
from pylint.interfaces import IAstroidChecker, IRawChecker, ITokenChecker
from pylint.utils.pragma_parser import OPTION_PO, PragmaParserError, parse_pragma

_ASYNC_TOKEN = "async"
_KEYWORD_TOKENS = [
    "assert",
    "del",
    "elif",
    "except",
    "for",
    "if",
    "in",
    "not",
    "raise",
    "return",
    "while",
    "yield",
    "with",
]

_SPACED_OPERATORS = [
    "==",
    "<",
    ">",
    "!=",
    "<>",
    "<=",
    ">=",
    "+=",
    "-=",
    "*=",
    "**=",
    "/=",
    "//=",
    "&=",
    "|=",
    "^=",
    "%=",
    ">>=",
    "<<=",
]
_OPENING_BRACKETS = ["(", "[", "{"]
_CLOSING_BRACKETS = [")", "]", "}"]
_TAB_LENGTH = 8

_EOL = frozenset([tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT])
_JUNK_TOKENS = (tokenize.COMMENT, tokenize.NL)

# Whitespace checking policy constants
_MUST = 0
_MUST_NOT = 1
_IGNORE = 2
```
### 7 - pylint/__init__.py:

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
### 8 - pylint/checkers/python3.py:

Start line: 43, End line: 111

```python
import itertools
import re
import tokenize
from collections import namedtuple

import astroid

from pylint import checkers, interfaces
from pylint.checkers import utils
from pylint.checkers.utils import find_try_except_wrapper_node, node_ignores_exception
from pylint.constants import WarningScope
from pylint.interfaces import INFERENCE, INFERENCE_FAILURE

_ZERO = re.compile("^0+$")


def _is_old_octal(literal):
    if _ZERO.match(literal):
        return False
    if re.match(r"0\d+", literal):
        try:
            int(literal, 8)
        except ValueError:
            return False
        return True
    return None


def _inferred_value_is_dict(value):
    if isinstance(value, astroid.Dict):
        return True
    return isinstance(value, astroid.Instance) and "dict" in value.basenames


def _infer_if_relevant_attr(node, relevant_attrs):
    return node.expr.infer() if node.attrname in relevant_attrs else []


def _is_builtin(node):
    return getattr(node, "name", None) in ("__builtin__", "builtins")


_ACCEPTS_ITERATOR = {
    "iter",
    "list",
    "tuple",
    "sorted",
    "set",
    "sum",
    "any",
    "all",
    "enumerate",
    "dict",
    "filter",
    "reversed",
    "max",
    "min",
    "frozenset",
    "OrderedDict",
    "zip",
    "map",
}
ATTRIBUTES_ACCEPTS_ITERATOR = {"join", "from_iterable"}
_BUILTIN_METHOD_ACCEPTS_ITERATOR = {
    "builtins.list.extend",
    "builtins.dict.update",
    "builtins.set.update",
}
DICT_METHODS = {"items", "keys", "values"}
```
### 9 - pylint/lint/pylinter.py:

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
### 10 - pylint/extensions/bad_builtin.py:

Start line: 12, End line: 50

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
