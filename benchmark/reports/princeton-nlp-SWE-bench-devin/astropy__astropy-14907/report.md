# astropy__astropy-14907

| **astropy/astropy** | `7f0df518e6bd5542b64bd7073052d099ea09dcb4` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 3 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/astropy/table/index.py b/astropy/table/index.py
--- a/astropy/table/index.py
+++ b/astropy/table/index.py
@@ -94,7 +94,7 @@ def __init__(self, columns, engine=None, unique=False):
             raise ValueError("Cannot create index without at least one column")
         elif len(columns) == 1:
             col = columns[0]
-            row_index = Column(col.argsort())
+            row_index = Column(col.argsort(kind="stable"))
             data = Table([col[row_index]])
         else:
             num_rows = len(columns[0])
@@ -117,7 +117,7 @@ def __init__(self, columns, engine=None, unique=False):
             try:
                 lines = table[np.lexsort(sort_columns)]
             except TypeError:  # arbitrary mixins might not work with lexsort
-                lines = table[table.argsort()]
+                lines = table[table.argsort(kind="stable")]
             data = lines[lines.colnames[:-1]]
             row_index = lines[lines.colnames[-1]]
 
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -1441,13 +1441,28 @@ def argmax(self, axis=None, out=None):
 
         return dt.argmax(axis, out)
 
-    def argsort(self, axis=-1):
+    def argsort(self, axis=-1, kind="stable"):
         """Returns the indices that would sort the time array.
 
-        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure
-        that the full precision given by the two doubles ``jd1`` and ``jd2``
-        is used, and that corresponding attributes are copied.  Internally,
-        it uses :func:`~numpy.lexsort`, and hence no sort method can be chosen.
+        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure that
+        the full precision given by the two doubles ``jd1`` and ``jd2`` is used, and
+        that corresponding attributes are copied.  Internally, it uses
+        :func:`~numpy.lexsort`, and hence no sort method can be chosen.
+
+        Parameters
+        ----------
+        axis : int, optional
+            Axis along which to sort. Default is -1, which means sort along the last
+            axis.
+        kind : 'stable', optional
+            Sorting is done with :func:`~numpy.lexsort` so this argument is ignored, but
+            kept for compatibility with :func:`~numpy.argsort`. The sorting is stable,
+            meaning that the order of equal elements is preserved.
+
+        Returns
+        -------
+        indices : ndarray
+            An array of indices that sort the time array.
         """
         # For procedure, see comment on argmin.
         jd1, jd2 = self.jd1, self.jd2

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/table/index.py | 97 | 97 | - | - | -
| astropy/table/index.py | 120 | 120 | - | - | -
| astropy/time/core.py | 1444 | 1450 | - | - | -


## Problem Statement

```
TST: test_table_group_by[True] and test_group_by_masked[True] failed with numpy 1.25rc1
I see this in the predeps job that pulls in numpy 1.25rc1. Example log: https://github.com/astropy/astropy/actions/runs/5117103756/jobs/9199883166

Hard to discern between the other 100+ failures from https://github.com/astropy/astropy/issues/14881 and I do not understand why we didn't catch this earlier in devdeps. @mhvk , does this look familiar to you?

https://github.com/astropy/astropy/blob/88790514bdf248e43c2fb15ee18cfd3390846145/astropy/table/tests/test_groups.py#L35

\`\`\`
__________________________ test_table_group_by[True] ___________________________

T1 = <QTable length=8>
  a    b      c      d      q   
                            m   
int64 str1 float64 int64 float64
-...   0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0

    def test_table_group_by(T1):
        """
        Test basic table group_by functionality for possible key types and for
        masked/unmasked tables.
        """
        for masked in (False, True):
            t1 = QTable(T1, masked=masked)
            # Group by a single column key specified by name
            tg = t1.group_by("a")
            assert np.all(tg.groups.indices == np.array([0, 1, 4, 8]))
            assert str(tg.groups) == "<TableGroups indices=[0 1 4 8]>"
            assert str(tg["a"].groups) == "<ColumnGroups indices=[0 1 4 8]>"
    
            # Sorted by 'a' and in original order for rest
>           assert tg.pformat() == [
                " a   b   c   d   q ",
                "                 m ",
                "--- --- --- --- ---",
                "  0   a 0.0   4 4.0",
                "  1   b 3.0   5 5.0",
                "  1   a 2.0   6 6.0",
                "  1   a 1.0   7 7.0",
                "  2   c 7.0   0 0.0",
                "  2   b 5.0   1 1.0",
                "  2   b 6.0   2 2.0",
                "  2   a 4.0   3 3.0",
            ]
E           AssertionError: assert [' a   b   c ...  5 5.0', ...] == [' a   b   c ...  6 6.0', ...]
E             At index 4 diff: '  1   a 1.0   7 7.0' != '  1   b 3.0   5 5.0'
E             Full diff:
E               [
E                ' a   b   c   d   q ',
E                '                 m ',
E                '--- --- --- --- ---',
E                '  0   a 0.0   4 4.0',
E             +  '  1   a 1.0   7 7.0',
E                '  1   b 3.0   5 5.0',
E                '  1   a 2.0   6 6.0',
E             -  '  1   a 1.0   7 7.0',
E             ?     ^     ^     ^^^
E             +  '  2   a 4.0   3 3.0',
E             ?     ^     ^     ^^^
E             +  '  2   b 6.0   2 2.0',
E             +  '  2   b 5.0   1 1.0',
E                '  2   c 7.0   0 0.0',
E             -  '  2   b 5.0   1 1.0',
E             -  '  2   b 6.0   2 2.0',
E             -  '  2   a 4.0   3 3.0',
E               ]

astropy/table/tests/test_groups.py:49: AssertionError
\`\`\`

https://github.com/astropy/astropy/blob/88790514bdf248e43c2fb15ee18cfd3390846145/astropy/table/tests/test_groups.py#L326

\`\`\`
__________________________ test_group_by_masked[True] __________________________

T1 = <QTable length=8>
  a    b      c      d      q   
                            m   
int64 str1 float64 int64 float64
-...   0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0

    def test_group_by_masked(T1):
        t1m = QTable(T1, masked=True)
        t1m["c"].mask[4] = True
        t1m["d"].mask[5] = True
>       assert t1m.group_by("a").pformat() == [
            " a   b   c   d   q ",
            "                 m ",
            "--- --- --- --- ---",
            "  0   a  --   4 4.0",
            "  1   b 3.0  -- 5.0",
            "  1   a 2.0   6 6.0",
            "  1   a 1.0   7 7.0",
            "  2   c 7.0   0 0.0",
            "  2   b 5.0   1 1.0",
            "  2   b 6.0   2 2.0",
            "  2   a 4.0   3 3.0",
        ]
E       AssertionError: assert [' a   b   c ... -- 5.0', ...] == [' a   b   c ...  6 6.0', ...]
E         At index 4 diff: '  1   a 1.0   7 7.0' != '  1   b 3.0  -- 5.0'
E         Full diff:
E           [
E            ' a   b   c   d   q ',
E            '                 m ',
E            '--- --- --- --- ---',
E            '  0   a  --   4 4.0',
E         +  '  1   a 1.0   7 7.0',
E            '  1   b 3.0  -- 5.0',
E            '  1   a 2.0   6 6.0',
E         -  '  1   a 1.0   7 7.0',
E         ?     ^     ^     ^^^
E         +  '  2   a 4.0   3 3.0',
E         ?     ^     ^     ^^^
E         +  '  2   b 6.0   2 2.0',
E         +  '  2   b 5.0   1 1.0',
E            '  2   c 7.0   0 0.0',
E         -  '  2   b 5.0   1 1.0',
E         -  '  2   b 6.0   2 2.0',
E         -  '  2   a 4.0   3 3.0',
E           ]

astropy/table/tests/test_groups.py:330: AssertionError
\`\`\`
TST: test_table_group_by[True] and test_group_by_masked[True] failed with numpy 1.25rc1
I see this in the predeps job that pulls in numpy 1.25rc1. Example log: https://github.com/astropy/astropy/actions/runs/5117103756/jobs/9199883166

Hard to discern between the other 100+ failures from https://github.com/astropy/astropy/issues/14881 and I do not understand why we didn't catch this earlier in devdeps. @mhvk , does this look familiar to you?

https://github.com/astropy/astropy/blob/88790514bdf248e43c2fb15ee18cfd3390846145/astropy/table/tests/test_groups.py#L35

\`\`\`
__________________________ test_table_group_by[True] ___________________________

T1 = <QTable length=8>
  a    b      c      d      q   
                            m   
int64 str1 float64 int64 float64
-...   0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0

    def test_table_group_by(T1):
        """
        Test basic table group_by functionality for possible key types and for
        masked/unmasked tables.
        """
        for masked in (False, True):
            t1 = QTable(T1, masked=masked)
            # Group by a single column key specified by name
            tg = t1.group_by("a")
            assert np.all(tg.groups.indices == np.array([0, 1, 4, 8]))
            assert str(tg.groups) == "<TableGroups indices=[0 1 4 8]>"
            assert str(tg["a"].groups) == "<ColumnGroups indices=[0 1 4 8]>"
    
            # Sorted by 'a' and in original order for rest
>           assert tg.pformat() == [
                " a   b   c   d   q ",
                "                 m ",
                "--- --- --- --- ---",
                "  0   a 0.0   4 4.0",
                "  1   b 3.0   5 5.0",
                "  1   a 2.0   6 6.0",
                "  1   a 1.0   7 7.0",
                "  2   c 7.0   0 0.0",
                "  2   b 5.0   1 1.0",
                "  2   b 6.0   2 2.0",
                "  2   a 4.0   3 3.0",
            ]
E           AssertionError: assert [' a   b   c ...  5 5.0', ...] == [' a   b   c ...  6 6.0', ...]
E             At index 4 diff: '  1   a 1.0   7 7.0' != '  1   b 3.0   5 5.0'
E             Full diff:
E               [
E                ' a   b   c   d   q ',
E                '                 m ',
E                '--- --- --- --- ---',
E                '  0   a 0.0   4 4.0',
E             +  '  1   a 1.0   7 7.0',
E                '  1   b 3.0   5 5.0',
E                '  1   a 2.0   6 6.0',
E             -  '  1   a 1.0   7 7.0',
E             ?     ^     ^     ^^^
E             +  '  2   a 4.0   3 3.0',
E             ?     ^     ^     ^^^
E             +  '  2   b 6.0   2 2.0',
E             +  '  2   b 5.0   1 1.0',
E                '  2   c 7.0   0 0.0',
E             -  '  2   b 5.0   1 1.0',
E             -  '  2   b 6.0   2 2.0',
E             -  '  2   a 4.0   3 3.0',
E               ]

astropy/table/tests/test_groups.py:49: AssertionError
\`\`\`

https://github.com/astropy/astropy/blob/88790514bdf248e43c2fb15ee18cfd3390846145/astropy/table/tests/test_groups.py#L326

\`\`\`
__________________________ test_group_by_masked[True] __________________________

T1 = <QTable length=8>
  a    b      c      d      q   
                            m   
int64 str1 float64 int64 float64
-...   0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0

    def test_group_by_masked(T1):
        t1m = QTable(T1, masked=True)
        t1m["c"].mask[4] = True
        t1m["d"].mask[5] = True
>       assert t1m.group_by("a").pformat() == [
            " a   b   c   d   q ",
            "                 m ",
            "--- --- --- --- ---",
            "  0   a  --   4 4.0",
            "  1   b 3.0  -- 5.0",
            "  1   a 2.0   6 6.0",
            "  1   a 1.0   7 7.0",
            "  2   c 7.0   0 0.0",
            "  2   b 5.0   1 1.0",
            "  2   b 6.0   2 2.0",
            "  2   a 4.0   3 3.0",
        ]
E       AssertionError: assert [' a   b   c ... -- 5.0', ...] == [' a   b   c ...  6 6.0', ...]
E         At index 4 diff: '  1   a 1.0   7 7.0' != '  1   b 3.0  -- 5.0'
E         Full diff:
E           [
E            ' a   b   c   d   q ',
E            '                 m ',
E            '--- --- --- --- ---',
E            '  0   a  --   4 4.0',
E         +  '  1   a 1.0   7 7.0',
E            '  1   b 3.0  -- 5.0',
E            '  1   a 2.0   6 6.0',
E         -  '  1   a 1.0   7 7.0',
E         ?     ^     ^     ^^^
E         +  '  2   a 4.0   3 3.0',
E         ?     ^     ^     ^^^
E         +  '  2   b 6.0   2 2.0',
E         +  '  2   b 5.0   1 1.0',
E            '  2   c 7.0   0 0.0',
E         -  '  2   b 5.0   1 1.0',
E         -  '  2   b 6.0   2 2.0',
E         -  '  2   a 4.0   3 3.0',
E           ]

astropy/table/tests/test_groups.py:330: AssertionError
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/table/table.py | 2 | 90| 660 | 660 | 33427 | 
| 2 | 1 astropy/table/table.py | 4121 | 4172| 499 | 1159 | 33427 | 
| 3 | 1 astropy/table/table.py | 929 | 972| 307 | 1466 | 33427 | 
| 4 | 2 astropy/table/groups.py | 198 | 228| 279 | 1745 | 36497 | 
| 5 | 3 astropy/table/serialize.py | 2 | 74| 794 | 2539 | 40853 | 
| 6 | 4 astropy/utils/masked/core.py | 643 | 672| 302 | 2841 | 51559 | 
| 7 | 4 astropy/table/groups.py | 257 | 291| 298 | 3139 | 51559 | 
| 8 | 4 astropy/utils/masked/core.py | 969 | 1027| 645 | 3784 | 51559 | 
| 9 | 5 astropy/utils/masked/function_helpers.py | 84 | 161| 756 | 4540 | 60466 | 
| 10 | 6 astropy/table/table_helpers.py | 9 | 55| 421 | 4961 | 62036 | 
| 11 | 6 astropy/utils/masked/function_helpers.py | 164 | 231| 599 | 5560 | 62036 | 
| 12 | 7 astropy/table/operations.py | 1598 | 1635| 315 | 5875 | 76522 | 
| 13 | 8 astropy/io/votable/validator/result.py | 235 | 338| 806 | 6681 | 79048 | 
| 14 | 9 astropy/io/fits/hdu/groups.py | 326 | 365| 369 | 7050 | 84066 | 
| 15 | 10 astropy/io/ascii/core.py | 186 | 270| 404 | 7454 | 97897 | 
| 16 | 10 astropy/io/fits/hdu/groups.py | 519 | 534| 213 | 7667 | 97897 | 
| 17 | 11 astropy/io/votable/tree.py | 5 | 132| 590 | 8257 | 127314 | 
| 18 | 11 astropy/utils/masked/function_helpers.py | 654 | 669| 156 | 8413 | 127314 | 
| 19 | 11 astropy/io/fits/hdu/groups.py | 138 | 222| 739 | 9152 | 127314 | 
| 20 | 11 astropy/utils/masked/core.py | 865 | 911| 379 | 9531 | 127314 | 
| 21 | 12 astropy/table/__init__.py | 3 | 44| 218 | 9749 | 128123 | 
| 22 | 12 astropy/io/fits/hdu/groups.py | 404 | 467| 642 | 10391 | 128123 | 
| 23 | 13 astropy/units/format/ogip_parsetab.py | 39 | 83| 1026 | 11417 | 131288 | 
| 24 | 13 astropy/table/table.py | 1622 | 1677| 446 | 11863 | 131288 | 
| 25 | 13 astropy/utils/masked/function_helpers.py | 672 | 691| 139 | 12002 | 131288 | 
| 26 | 13 astropy/table/groups.py | 3 | 118| 915 | 12917 | 131288 | 
| 27 | 13 astropy/io/votable/tree.py | 2794 | 2909| 915 | 13832 | 131288 | 
| 28 | 14 astropy/io/fits/diff.py | 1490 | 1554| 553 | 14385 | 144377 | 
| 29 | 15 astropy/io/fits/column.py | 86 | 146| 757 | 15142 | 167065 | 
| 30 | 15 astropy/utils/masked/core.py | 1183 | 1220| 357 | 15499 | 167065 | 
| 31 | 15 astropy/io/fits/diff.py | 1344 | 1488| 1295 | 16794 | 167065 | 
| 32 | 15 astropy/table/table.py | 3688 | 3732| 388 | 17182 | 167065 | 
| 33 | 16 astropy/timeseries/core.py | 46 | 104| 465 | 17647 | 167801 | 
| 34 | 17 astropy/table/column.py | 1808 | 1835| 269 | 17916 | 182630 | 
| 35 | 18 astropy/units/quantity_helper/function_helpers.py | 190 | 276| 771 | 18687 | 192877 | 
| 36 | 18 astropy/units/format/ogip_parsetab.py | 22 | 22| 1114 | 19801 | 192877 | 
| 37 | 18 astropy/io/fits/column.py | 3 | 85| 770 | 20571 | 192877 | 
| 38 | 18 astropy/table/table.py | 3734 | 3814| 650 | 21221 | 192877 | 


## Missing Patch Files

 * 1: astropy/table/index.py
 * 2: astropy/time/core.py

### Hint

```
I cannot reproduce this locally. 🤯  The error log above looks like some lines moved about... but it does not make sense.

Also, to run this in an interactive session:

\`\`\`python
import numpy as np
from astropy import units as u
from astropy.table import QTable

T = QTable.read(
        [
            " a b c d",
            " 2 c 7.0 0",
            " 2 b 5.0 1",
            " 2 b 6.0 2",
            " 2 a 4.0 3",
            " 0 a 0.0 4",
            " 1 b 3.0 5",
            " 1 a 2.0 6",
            " 1 a 1.0 7",
        ],
        format="ascii",
)
T["q"] = np.arange(len(T)) * u.m
T.meta.update({"ta": 1})
T["c"].meta.update({"a": 1})
T["c"].description = "column c"
T.add_index("a")

t1 = QTable(T, masked=True)
tg = t1.group_by("a")
\`\`\`
\`\`\`python
>>> tg
<QTable length=8>
  a    b      c      d      q
                            m
int64 str1 float64 int64 float64
----- ---- ------- ----- -------
    0    a     0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0
    2    c     7.0     0     0.0
    2    b     5.0     1     1.0
    2    b     6.0     2     2.0
    2    a     4.0     3     3.0
\`\`\`
@pllim - I also cannot reproduce the problem locally on my Mac. What to do?
@taldcroft , does the order matter? I am guessing yes?

Looks like maybe somehow this test triggers some race condition but only in CI, or some global var is messing it up from a different test. But I don't know enough about internals to make a more educated guess.
What about this: could the sort order have changed? Is the test failure on an "AVX-512 enabled processor"?

https://numpy.org/devdocs/release/1.25.0-notes.html#faster-np-sort-on-avx-512-enabled-processors
Hmmm.... maybe?

* https://github.com/actions/runner-images/discussions/5734
@pllim - Grouping is supposed to maintain the original order within a group. That depends on the numpy sorting doing the same, which depends on the specific sort algorithm.
So that looks promising. Let me remind myself of the code in there...
So if we decide that https://github.com/numpy/numpy/pull/22315 is changing our result, is that a numpy bug?
It turns out this is likely in the table indexing code, which is never easy to understand. But it does look like this is the issue, because indexing appears to use the numpy default sorting, which is quicksort. But quicksort is not guaranteed to be stable, so maybe it was passing accidentally before.
I cannot reproduce this locally. 🤯  The error log above looks like some lines moved about... but it does not make sense.

Also, to run this in an interactive session:

\`\`\`python
import numpy as np
from astropy import units as u
from astropy.table import QTable

T = QTable.read(
        [
            " a b c d",
            " 2 c 7.0 0",
            " 2 b 5.0 1",
            " 2 b 6.0 2",
            " 2 a 4.0 3",
            " 0 a 0.0 4",
            " 1 b 3.0 5",
            " 1 a 2.0 6",
            " 1 a 1.0 7",
        ],
        format="ascii",
)
T["q"] = np.arange(len(T)) * u.m
T.meta.update({"ta": 1})
T["c"].meta.update({"a": 1})
T["c"].description = "column c"
T.add_index("a")

t1 = QTable(T, masked=True)
tg = t1.group_by("a")
\`\`\`
\`\`\`python
>>> tg
<QTable length=8>
  a    b      c      d      q
                            m
int64 str1 float64 int64 float64
----- ---- ------- ----- -------
    0    a     0.0     4     4.0
    1    b     3.0     5     5.0
    1    a     2.0     6     6.0
    1    a     1.0     7     7.0
    2    c     7.0     0     0.0
    2    b     5.0     1     1.0
    2    b     6.0     2     2.0
    2    a     4.0     3     3.0
\`\`\`
@pllim - I also cannot reproduce the problem locally on my Mac. What to do?
@taldcroft , does the order matter? I am guessing yes?

Looks like maybe somehow this test triggers some race condition but only in CI, or some global var is messing it up from a different test. But I don't know enough about internals to make a more educated guess.
What about this: could the sort order have changed? Is the test failure on an "AVX-512 enabled processor"?

https://numpy.org/devdocs/release/1.25.0-notes.html#faster-np-sort-on-avx-512-enabled-processors
Hmmm.... maybe?

* https://github.com/actions/runner-images/discussions/5734
@pllim - Grouping is supposed to maintain the original order within a group. That depends on the numpy sorting doing the same, which depends on the specific sort algorithm.
So that looks promising. Let me remind myself of the code in there...
So if we decide that https://github.com/numpy/numpy/pull/22315 is changing our result, is that a numpy bug?
It turns out this is likely in the table indexing code, which is never easy to understand. But it does look like this is the issue, because indexing appears to use the numpy default sorting, which is quicksort. But quicksort is not guaranteed to be stable, so maybe it was passing accidentally before.
```

## Patch

```diff
diff --git a/astropy/table/index.py b/astropy/table/index.py
--- a/astropy/table/index.py
+++ b/astropy/table/index.py
@@ -94,7 +94,7 @@ def __init__(self, columns, engine=None, unique=False):
             raise ValueError("Cannot create index without at least one column")
         elif len(columns) == 1:
             col = columns[0]
-            row_index = Column(col.argsort())
+            row_index = Column(col.argsort(kind="stable"))
             data = Table([col[row_index]])
         else:
             num_rows = len(columns[0])
@@ -117,7 +117,7 @@ def __init__(self, columns, engine=None, unique=False):
             try:
                 lines = table[np.lexsort(sort_columns)]
             except TypeError:  # arbitrary mixins might not work with lexsort
-                lines = table[table.argsort()]
+                lines = table[table.argsort(kind="stable")]
             data = lines[lines.colnames[:-1]]
             row_index = lines[lines.colnames[-1]]
 
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -1441,13 +1441,28 @@ def argmax(self, axis=None, out=None):
 
         return dt.argmax(axis, out)
 
-    def argsort(self, axis=-1):
+    def argsort(self, axis=-1, kind="stable"):
         """Returns the indices that would sort the time array.
 
-        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure
-        that the full precision given by the two doubles ``jd1`` and ``jd2``
-        is used, and that corresponding attributes are copied.  Internally,
-        it uses :func:`~numpy.lexsort`, and hence no sort method can be chosen.
+        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure that
+        the full precision given by the two doubles ``jd1`` and ``jd2`` is used, and
+        that corresponding attributes are copied.  Internally, it uses
+        :func:`~numpy.lexsort`, and hence no sort method can be chosen.
+
+        Parameters
+        ----------
+        axis : int, optional
+            Axis along which to sort. Default is -1, which means sort along the last
+            axis.
+        kind : 'stable', optional
+            Sorting is done with :func:`~numpy.lexsort` so this argument is ignored, but
+            kept for compatibility with :func:`~numpy.argsort`. The sorting is stable,
+            meaning that the order of equal elements is preserved.
+
+        Returns
+        -------
+        indices : ndarray
+            An array of indices that sort the time array.
         """
         # For procedure, see comment on argmin.
         jd1, jd2 = self.jd1, self.jd2

```

## Test Patch

```diff
diff --git a/astropy/table/tests/test_groups.py b/astropy/table/tests/test_groups.py
--- a/astropy/table/tests/test_groups.py
+++ b/astropy/table/tests/test_groups.py
@@ -690,3 +690,23 @@ def test_group_mixins_unsupported(col):
     tg = t.group_by("a")
     with pytest.warns(AstropyUserWarning, match="Cannot aggregate column 'mix'"):
         tg.groups.aggregate(np.sum)
+
+
+@pytest.mark.parametrize("add_index", [False, True])
+def test_group_stable_sort(add_index):
+    """Test that group_by preserves the order of the table.
+
+    This table has 5 groups with an average of 200 rows per group, so it is not
+    statistically possible that the groups will be in order by chance.
+
+    This tests explicitly the case where grouping is done via the index sort.
+    See: https://github.com/astropy/astropy/issues/14882
+    """
+    a = np.random.randint(0, 5, 1000)
+    b = np.arange(len(a))
+    t = Table([a, b], names=["a", "b"])
+    if add_index:
+        t.add_index("a")
+    tg = t.group_by("a")
+    for grp in tg.groups:
+        assert np.all(grp["b"] == np.sort(grp["b"]))

```


## Code snippets

### 1 - astropy/table/table.py:

Start line: 2, End line: 90

```python
import itertools
import sys
import types
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from numpy import ma

from astropy import log
from astropy.io.registry import UnifiedReadWriteMethod
from astropy.units import Quantity, QuantityInfo
from astropy.utils import ShapedLikeNDArray, isiterable
from astropy.utils.console import color_print
from astropy.utils.data_info import BaseColumnInfo, DataInfo, MixinInfo
from astropy.utils.decorators import format_doc
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
from astropy.utils.metadata import MetaAttribute, MetaData

from . import conf, groups
from .column import (
    BaseColumn,
    Column,
    FalseArray,
    MaskedColumn,
    _auto_names,
    _convert_sequence_data_to_array,
    col_copy,
)
from .connect import TableRead, TableWrite
from .index import (
    Index,
    SlicedIndex,
    TableILoc,
    TableIndices,
    TableLoc,
    TableLocIndices,
    _IndexModeContext,
    get_index,
)
from .info import TableInfo
from .mixins.registry import get_mixin_handler
from .ndarray_mixin import NdarrayMixin  # noqa: F401
from .pprint import TableFormatter
from .row import Row

_implementation_notes = """
This string has informal notes concerning Table implementation for developers.

Things to remember:

- Table has customizable attributes ColumnClass, Column, MaskedColumn.
  Table.Column is normally just column.Column (same w/ MaskedColumn)
  but in theory they can be different.  Table.ColumnClass is the default
  class used to create new non-mixin columns, and this is a function of
  the Table.masked attribute.  Column creation / manipulation in a Table
  needs to respect these.

- Column objects that get inserted into the Table.columns attribute must
  have the info.parent_table attribute set correctly.  Beware just dropping
  an object into the columns dict since an existing column may
  be part of another Table and have parent_table set to point at that
  table.  Dropping that column into `columns` of this Table will cause
  a problem for the old one so the column object needs to be copied (but
  not necessarily the data).

  Currently replace_column is always making a copy of both object and
  data if parent_table is set.  This could be improved but requires a
  generic way to copy a mixin object but not the data.

- Be aware of column objects that have indices set.

- `cls.ColumnClass` is a property that effectively uses the `masked` attribute
  to choose either `cls.Column` or `cls.MaskedColumn`.
"""

__doctest_skip__ = [
    "Table.read",
    "Table.write",
    "Table._read",
    "Table.convert_bytestring_to_unicode",
    "Table.convert_unicode_to_bytestring",
]

__doctest_requires__ = {"*pandas": ["pandas>=1.1"]}
```
### 2 - astropy/table/table.py:

Start line: 4121, End line: 4172

```python
class Table:

    @classmethod
    def from_pandas(cls, dataframe, index=False, units=None):
        # ... other code

        for name, column, data, mask, unit in zip(names, columns, datas, masks, units):
            if column.dtype.kind in ["u", "i"] and np.any(mask):
                # Special-case support for pandas nullable int
                np_dtype = str(column.dtype).lower()
                data = np.zeros(shape=column.shape, dtype=np_dtype)
                data[~mask] = column[~mask]
                out[name] = MaskedColumn(
                    data=data, name=name, mask=mask, unit=unit, copy=False
                )
                continue

            if data.dtype.kind == "O":
                # If all elements of an object array are string-like or np.nan
                # then coerce back to a native numpy str/unicode array.
                string_types = (str, bytes)
                nan = np.nan
                if all(isinstance(x, string_types) or x is nan for x in data):
                    # Force any missing (null) values to b''.  Numpy will
                    # upcast to str/unicode as needed.
                    data[mask] = b""

                    # When the numpy object array is represented as a list then
                    # numpy initializes to the correct string or unicode type.
                    data = np.array([x for x in data])

            # Numpy datetime64
            if data.dtype.kind == "M":
                from astropy.time import Time

                out[name] = Time(data, format="datetime64")
                if np.any(mask):
                    out[name][mask] = np.ma.masked
                out[name].format = "isot"

            # Numpy timedelta64
            elif data.dtype.kind == "m":
                from astropy.time import TimeDelta

                data_sec = data.astype("timedelta64[ns]").astype(np.float64) / 1e9
                out[name] = TimeDelta(data_sec, format="sec")
                if np.any(mask):
                    out[name][mask] = np.ma.masked

            else:
                if np.any(mask):
                    out[name] = MaskedColumn(data=data, name=name, mask=mask, unit=unit)
                else:
                    out[name] = Column(data=data, name=name, unit=unit)

        return cls(out)

    info = TableInfo()
```
### 3 - astropy/table/table.py:

Start line: 929, End line: 972

```python
class Table:

    def __getstate__(self):
        columns = OrderedDict(
            (key, col if isinstance(col, BaseColumn) else col_copy(col))
            for key, col in self.columns.items()
        )
        return (columns, self.meta)

    def __setstate__(self, state):
        columns, meta = state
        self.__init__(columns, meta=meta)

    @property
    def mask(self):
        # Dynamic view of available masks
        if self.masked or self.has_masked_columns or self.has_masked_values:
            mask_table = Table(
                [
                    getattr(col, "mask", FalseArray(col.shape))
                    for col in self.itercols()
                ],
                names=self.colnames,
                copy=False,
            )

            # Set hidden attribute to force inplace setitem so that code like
            # t.mask['a'] = [1, 0, 1] will correctly set the underlying mask.
            # See #5556 for discussion.
            mask_table._setitem_inplace = True
        else:
            mask_table = None

        return mask_table

    @mask.setter
    def mask(self, val):
        self.mask[:] = val

    @property
    def _mask(self):
        """This is needed so that comparison of a masked Table and a
        MaskedArray works.  The requirement comes from numpy.ma.core
        so don't remove this property.
        """
        return self.as_array().mask
```
### 4 - astropy/table/groups.py:

Start line: 198, End line: 228

```python
class BaseGroups:

    def __getitem__(self, item):
        parent = self.parent

        if isinstance(item, (int, np.integer)):
            i0, i1 = self.indices[item], self.indices[item + 1]
            out = parent[i0:i1]
            out.groups._keys = parent.groups.keys[item]
        else:
            indices0, indices1 = self.indices[:-1], self.indices[1:]
            try:
                i0s, i1s = indices0[item], indices1[item]
            except Exception as err:
                raise TypeError(
                    "Index item for groups attribute must be a slice, "
                    "numpy mask or int array"
                ) from err
            mask = np.zeros(len(parent), dtype=bool)
            # Is there a way to vectorize this in numpy?
            for i0, i1 in zip(i0s, i1s):
                mask[i0:i1] = True
            out = parent[mask]
            out.groups._keys = parent.groups.keys[item]
            out.groups._indices = np.concatenate([[0], np.cumsum(i1s - i0s)])

        return out

    def __repr__(self):
        return f"<{self.__class__.__name__} indices={self.indices}>"

    def __len__(self):
        return len(self.indices) - 1
```
### 5 - astropy/table/serialize.py:

Start line: 2, End line: 74

```python
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import numpy as np

from astropy.units.quantity import QuantityInfo
from astropy.utils.data_info import MixinInfo

from .column import Column, MaskedColumn
from .table import QTable, Table, has_info_class

# TODO: some of this might be better done programmatically, through
# code like
# __construct_mixin_classes += tuple(
#        f'astropy.coordinates.representation.{cls.__name__}'
#        for cls in (list(coorep.REPRESENTATION_CLASSES.values())
#                    + list(coorep.DIFFERENTIAL_CLASSES.values()))
#        if cls.__name__ in coorep.__all__)
# However, to avoid very hard to track import issues, the definition
# should then be done at the point where it is actually needed,
# using local imports.  See also
# https://github.com/astropy/astropy/pull/10210#discussion_r419087286
__construct_mixin_classes = (
    "astropy.time.core.Time",
    "astropy.time.core.TimeDelta",
    "astropy.units.quantity.Quantity",
    "astropy.units.function.logarithmic.Magnitude",
    "astropy.units.function.logarithmic.Decibel",
    "astropy.units.function.logarithmic.Dex",
    "astropy.coordinates.angles.Latitude",
    "astropy.coordinates.angles.Longitude",
    "astropy.coordinates.angles.Angle",
    "astropy.coordinates.distances.Distance",
    "astropy.coordinates.earth.EarthLocation",
    "astropy.coordinates.sky_coordinate.SkyCoord",
    "astropy.coordinates.polarization.StokesCoord",
    "astropy.table.ndarray_mixin.NdarrayMixin",
    "astropy.table.table_helpers.ArrayWrapper",
    "astropy.table.column.Column",
    "astropy.table.column.MaskedColumn",
    "astropy.utils.masked.core.MaskedNDArray",
    # Representations
    "astropy.coordinates.representation.cartesian.CartesianRepresentation",
    "astropy.coordinates.representation.spherical.UnitSphericalRepresentation",
    "astropy.coordinates.representation.spherical.RadialRepresentation",
    "astropy.coordinates.representation.spherical.SphericalRepresentation",
    "astropy.coordinates.representation.spherical.PhysicsSphericalRepresentation",
    "astropy.coordinates.representation.cylindrical.CylindricalRepresentation",
    "astropy.coordinates.representation.cartesian.CartesianDifferential",
    "astropy.coordinates.representation.spherical.UnitSphericalDifferential",
    "astropy.coordinates.representation.spherical.SphericalDifferential",
    "astropy.coordinates.representation.spherical.UnitSphericalCosLatDifferential",
    "astropy.coordinates.representation.spherical.SphericalCosLatDifferential",
    "astropy.coordinates.representation.spherical.RadialDifferential",
    "astropy.coordinates.representation.spherical.PhysicsSphericalDifferential",
    "astropy.coordinates.representation.cylindrical.CylindricalDifferential",
    # Deprecated paths
    "astropy.coordinates.representation.CartesianRepresentation",
    "astropy.coordinates.representation.UnitSphericalRepresentation",
    "astropy.coordinates.representation.RadialRepresentation",
    "astropy.coordinates.representation.SphericalRepresentation",
    "astropy.coordinates.representation.PhysicsSphericalRepresentation",
    "astropy.coordinates.representation.CylindricalRepresentation",
    "astropy.coordinates.representation.CartesianDifferential",
    "astropy.coordinates.representation.UnitSphericalDifferential",
    "astropy.coordinates.representation.SphericalDifferential",
    "astropy.coordinates.representation.UnitSphericalCosLatDifferential",
    "astropy.coordinates.representation.SphericalCosLatDifferential",
    "astropy.coordinates.representation.RadialDifferential",
    "astropy.coordinates.representation.PhysicsSphericalDifferential",
    "astropy.coordinates.representation.CylindricalDifferential",
)
```
### 6 - astropy/utils/masked/core.py:

Start line: 643, End line: 672

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    _eq_simple = _comparison_method("__eq__")
    _ne_simple = _comparison_method("__ne__")
    __lt__ = _comparison_method("__lt__")
    __le__ = _comparison_method("__le__")
    __gt__ = _comparison_method("__gt__")
    __ge__ = _comparison_method("__ge__")

    def __eq__(self, other):
        if not self.dtype.names:
            return self._eq_simple(other)

        # For structured arrays, we treat this as a reduction over the fields,
        # where masked fields are skipped and thus do not influence the result.
        other = np.asanyarray(other, dtype=self.dtype)
        result = np.stack(
            [self[field] == other[field] for field in self.dtype.names], axis=-1
        )
        return result.all(axis=-1)

    def __ne__(self, other):
        if not self.dtype.names:
            return self._ne_simple(other)

        # For structured arrays, we treat this as a reduction over the fields,
        # where masked fields are skipped and thus do not influence the result.
        other = np.asanyarray(other, dtype=self.dtype)
        result = np.stack(
            [self[field] != other[field] for field in self.dtype.names], axis=-1
        )
        return result.any(axis=-1)
```
### 7 - astropy/table/groups.py:

Start line: 257, End line: 291

```python
class ColumnGroups(BaseGroups):

    def aggregate(self, func):
        from .column import MaskedColumn

        i0s, i1s = self.indices[:-1], self.indices[1:]
        par_col = self.parent_column
        masked = isinstance(par_col, MaskedColumn)
        reduceat = hasattr(func, "reduceat")
        sum_case = func is np.sum
        mean_case = func is np.mean
        try:
            if not masked and (reduceat or sum_case or mean_case):
                if mean_case:
                    vals = np.add.reduceat(par_col, i0s) / np.diff(self.indices)
                else:
                    if sum_case:
                        func = np.add
                    vals = func.reduceat(par_col, i0s)
            else:
                vals = np.array([func(par_col[i0:i1]) for i0, i1 in zip(i0s, i1s)])
            out = par_col.__class__(vals)
        except Exception as err:
            raise TypeError(
                "Cannot aggregate column '{}' with type '{}': {}".format(
                    par_col.info.name, par_col.info.dtype, err
                )
            ) from err

        out_info = out.info
        for attr in ("name", "unit", "format", "description", "meta"):
            try:
                setattr(out_info, attr, getattr(par_col.info, attr))
            except AttributeError:
                pass

        return out
```
### 8 - astropy/utils/masked/core.py:

Start line: 969, End line: 1027

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        # Unfortunately, cannot override the call to diagonal inside trace, so
        # duplicate implementation in numpy/core/src/multiarray/calculation.c.
        diagonal = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        return diagonal.sum(-1, dtype=dtype, out=out)

    def min(self, axis=None, out=None, **kwargs):
        return super().min(
            axis=axis, out=out, **self._reduce_defaults(kwargs, np.nanmax)
        )

    def max(self, axis=None, out=None, **kwargs):
        return super().max(
            axis=axis, out=out, **self._reduce_defaults(kwargs, np.nanmin)
        )

    def nonzero(self):
        unmasked_nonzero = self.unmasked.nonzero()
        if self.ndim >= 1:
            not_masked = ~self.mask[unmasked_nonzero]
            return tuple(u[not_masked] for u in unmasked_nonzero)
        else:
            return unmasked_nonzero if not self.mask else np.nonzero(0)

    def compress(self, condition, axis=None, out=None):
        if out is not None:
            raise NotImplementedError("cannot yet give output")
        return self._apply("compress", condition, axis=axis)

    def repeat(self, repeats, axis=None):
        return self._apply("repeat", repeats, axis=axis)

    def choose(self, choices, out=None, mode="raise"):
        # Let __array_function__ take care since choices can be masked too.
        return np.choose(self, choices, out=out, mode=mode)

    if NUMPY_LT_1_22:

        def argmin(self, axis=None, out=None):
            # Todo: should this return a masked integer array, with masks
            # if all elements were masked?
            at_min = self == self.min(axis=axis, keepdims=True)
            return at_min.filled(False).argmax(axis=axis, out=out)

        def argmax(self, axis=None, out=None):
            at_max = self == self.max(axis=axis, keepdims=True)
            return at_max.filled(False).argmax(axis=axis, out=out)

    else:

        def argmin(self, axis=None, out=None, *, keepdims=False):
            # Todo: should this return a masked integer array, with masks
            # if all elements were masked?
            at_min = self == self.min(axis=axis, keepdims=True)
            return at_min.filled(False).argmax(axis=axis, out=out, keepdims=keepdims)

        def argmax(self, axis=None, out=None, *, keepdims=False):
            at_max = self == self.max(axis=axis, keepdims=True)
            return at_max.filled(False).argmax(axis=axis, out=out, keepdims=keepdims)
```
### 9 - astropy/utils/masked/function_helpers.py:

Start line: 84, End line: 161

```python
MASKED_SAFE_FUNCTIONS |= {
    # built-in from multiarray
    np.may_share_memory, np.can_cast, np.min_scalar_type, np.result_type,
    np.shares_memory,
    # np.core.arrayprint
    np.array_repr,
    # np.core.function_base
    np.linspace, np.logspace, np.geomspace,
    # np.core.numeric
    np.isclose, np.allclose, np.flatnonzero, np.argwhere,
    # np.core.shape_base
    np.atleast_1d, np.atleast_2d, np.atleast_3d, np.stack, np.hstack, np.vstack,
    # np.lib.function_base
    np.average, np.diff, np.extract, np.meshgrid, np.trapz, np.gradient,
    # np.lib.index_tricks
    np.diag_indices_from, np.triu_indices_from, np.tril_indices_from,
    np.fill_diagonal,
    # np.lib.shape_base
    np.column_stack, np.row_stack, np.dstack,
    np.array_split, np.split, np.hsplit, np.vsplit, np.dsplit,
    np.expand_dims, np.apply_along_axis, np.kron, np.tile,
    np.take_along_axis, np.put_along_axis,
    # np.lib.type_check (all but asfarray, nan_to_num)
    np.iscomplexobj, np.isrealobj, np.imag, np.isreal, np.real,
    np.real_if_close, np.common_type,
    # np.lib.ufunclike
    np.fix, np.isneginf, np.isposinf,
    # np.lib.function_base
    np.angle, np.i0,
}  # fmt: skip
IGNORED_FUNCTIONS = {
    # I/O - useless for Masked, since no way to store the mask.
    np.save, np.savez, np.savetxt, np.savez_compressed,
    # Polynomials
    np.poly, np.polyadd, np.polyder, np.polydiv, np.polyfit, np.polyint,
    np.polymul, np.polysub, np.polyval, np.roots, np.vander,
}  # fmt: skip
IGNORED_FUNCTIONS |= {
    np.pad, np.searchsorted, np.digitize,
    np.is_busday, np.busday_count, np.busday_offset,
    # numpy.lib.function_base
    np.cov, np.corrcoef, np.trim_zeros,
    # numpy.core.numeric
    np.correlate, np.convolve,
    # numpy.lib.histograms
    np.histogram, np.histogram2d, np.histogramdd, np.histogram_bin_edges,
    # TODO!!
    np.dot, np.vdot, np.inner, np.tensordot, np.cross,
    np.einsum, np.einsum_path,
}  # fmt: skip

# Really should do these...
IGNORED_FUNCTIONS |= {
    getattr(np, setopsname) for setopsname in np.lib.arraysetops.__all__
}


if NUMPY_LT_1_23:
    IGNORED_FUNCTIONS |= {
        # Deprecated, removed in numpy 1.23
        np.asscalar,
        np.alen,
    }

# Explicitly unsupported functions
UNSUPPORTED_FUNCTIONS |= {
    np.unravel_index,
    np.ravel_multi_index,
    np.ix_,
}

# No support for the functions also not supported by Quantity
# (io, polynomial, etc.).
UNSUPPORTED_FUNCTIONS |= IGNORED_FUNCTIONS


apply_to_both = FunctionAssigner(APPLY_TO_BOTH_FUNCTIONS)
dispatched_function = FunctionAssigner(DISPATCHED_FUNCTIONS)
```
### 10 - astropy/table/table_helpers.py:

Start line: 9, End line: 55

```python
import string
from itertools import cycle

import numpy as np

from astropy.utils.data_info import ParentDtypeInfo

from .table import Column, Table


class TimingTables:
    """
    Object which contains two tables and various other attributes that
    are useful for timing and other API tests.
    """

    def __init__(self, size=1000, masked=False):
        self.masked = masked

        # Initialize table
        self.table = Table(masked=self.masked)

        # Create column with mixed types
        np.random.seed(12345)
        self.table["i"] = np.arange(size)
        self.table["a"] = np.random.random(size)  # float
        self.table["b"] = np.random.random(size) > 0.5  # bool
        self.table["c"] = np.random.random((size, 10))  # 2d column
        self.table["d"] = np.random.choice(np.array(list(string.ascii_letters)), size)

        self.extra_row = {"a": 1.2, "b": True, "c": np.repeat(1, 10), "d": "Z"}
        self.extra_column = np.random.randint(0, 100, size)
        self.row_indices = np.where(self.table["a"] > 0.9)[0]
        self.table_grouped = self.table.group_by("d")

        # Another table for testing joining
        self.other_table = Table(masked=self.masked)
        self.other_table["i"] = np.arange(1, size, 3)
        self.other_table["f"] = np.random.random()
        self.other_table.sort("f")

        # Another table for testing hstack
        self.other_table_2 = Table(masked=self.masked)
        self.other_table_2["g"] = np.random.random(size)
        self.other_table_2["h"] = np.random.random((size, 10))

        self.bool_mask = self.table["a"] > 0.6
```
