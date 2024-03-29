# astropy__astropy-13838

| **astropy/astropy** | `a6c712375ed38d422812e013566a34f928677acd` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 1338 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/table/pprint.py b/astropy/table/pprint.py
--- a/astropy/table/pprint.py
+++ b/astropy/table/pprint.py
@@ -392,7 +392,8 @@ def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
         if multidims:
             multidim0 = tuple(0 for n in multidims)
             multidim1 = tuple(n - 1 for n in multidims)
-            trivial_multidims = np.prod(multidims) == 1
+            multidims_all_ones = np.prod(multidims) == 1
+            multidims_has_zero = 0 in multidims
 
         i_dashes = None
         i_centers = []  # Line indexes where content should be centered
@@ -475,8 +476,11 @@ def format_col_str(idx):
                 # Prevents columns like Column(data=[[(1,)],[(2,)]], name='a')
                 # with shape (n,1,...,1) from being printed as if there was
                 # more than one element in a row
-                if trivial_multidims:
+                if multidims_all_ones:
                     return format_func(col_format, col[(idx,) + multidim0])
+                elif multidims_has_zero:
+                    # Any zero dimension means there is no data to print
+                    return ""
                 else:
                     left = format_func(col_format, col[(idx,) + multidim0])
                     right = format_func(col_format, col[(idx,) + multidim1])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/table/pprint.py | 395 | 395 | - | 1 | -
| astropy/table/pprint.py | 478 | 478 | 3 | 1 | 1338


## Problem Statement

```
Printing tables doesn't work correctly with 0-length array cells
<!-- This comments are hidden when you submit the issue,
so you do not need to remove them! -->

<!-- Please be sure to check out our contributing guidelines,
https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md .
Please be sure to check out our code of conduct,
https://github.com/astropy/astropy/blob/main/CODE_OF_CONDUCT.md . -->

<!-- Please have a search on our GitHub repository to see if a similar
issue has already been posted.
If a similar issue is closed, have a quick look to see if you are satisfied
by the resolution.
If not please go ahead and open an issue! -->

<!-- Please check that the development version still produces the same bug.
You can install development version with
pip install git+https://github.com/astropy/astropy
command. -->

### Description

I have data in form of a list of dictionaries.
Each dictionary contains some items with an integer value and some of these items set the length for 1 or more array values.

I am creating a Table using the `rows` attribute and feeding to it the list of dictionaries.

As long as I create a table until the first event with data in the array fields the table gets printed correctly.
If I fill the table only with events with null array data (but the rest of the fields have something to show) I get an IndexError.

### Expected behavior
<!-- What did you expect to happen. -->

The table should print fine also when there are only "bad" events

### Actual behavior
<!-- What actually happened. -->
<!-- Was the output confusing or poorly described? -->

I get the following error Traceback

\`\`\`
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/IPython/core/formatters.py:707, in PlainTextFormatter.__call__(self, obj)
    700 stream = StringIO()
    701 printer = pretty.RepresentationPrinter(stream, self.verbose,
    702     self.max_width, self.newline,
    703     max_seq_length=self.max_seq_length,
    704     singleton_pprinters=self.singleton_printers,
    705     type_pprinters=self.type_printers,
    706     deferred_pprinters=self.deferred_printers)
--> 707 printer.pretty(obj)
    708 printer.flush()
    709 return stream.getvalue()

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/IPython/lib/pretty.py:410, in RepresentationPrinter.pretty(self, obj)
    407                         return meth(obj, self, cycle)
    408                 if cls is not object \
    409                         and callable(cls.__dict__.get('__repr__')):
--> 410                     return _repr_pprint(obj, self, cycle)
    412     return _default_pprint(obj, self, cycle)
    413 finally:

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/IPython/lib/pretty.py:778, in _repr_pprint(obj, p, cycle)
    776 """A pprint that just redirects to the normal repr function."""
    777 # Find newlines and replace them with p.break_()
--> 778 output = repr(obj)
    779 lines = output.splitlines()
    780 with p.group():

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/table.py:1534, in Table.__repr__(self)
   1533 def __repr__(self):
-> 1534     return self._base_repr_(html=False, max_width=None)

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/table.py:1516, in Table._base_repr_(self, html, descr_vals, max_width, tableid, show_dtype, max_lines, tableclass)
   1513 if tableid is None:
   1514     tableid = f'table{id(self)}'
-> 1516 data_lines, outs = self.formatter._pformat_table(
   1517     self, tableid=tableid, html=html, max_width=max_width,
   1518     show_name=True, show_unit=None, show_dtype=show_dtype,
   1519     max_lines=max_lines, tableclass=tableclass)
   1521 out = descr + '\n'.join(data_lines)
   1523 return out

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:589, in TableFormatter._pformat_table(self, table, max_lines, max_width, show_name, show_unit, show_dtype, html, tableid, tableclass, align)
    586 if col.info.name not in pprint_include_names:
    587     continue
--> 589 lines, outs = self._pformat_col(col, max_lines, show_name=show_name,
    590                                 show_unit=show_unit, show_dtype=show_dtype,
    591                                 align=align_)
    592 if outs['show_length']:
    593     lines = lines[:-1]

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:276, in TableFormatter._pformat_col(self, col, max_lines, show_name, show_unit, show_dtype, show_length, html, align)
    268 col_strs_iter = self._pformat_col_iter(col, max_lines, show_name=show_name,
    269                                        show_unit=show_unit,
    270                                        show_dtype=show_dtype,
    271                                        show_length=show_length,
    272                                        outs=outs)
    274 # Replace tab and newline with text representations so they display nicely.
    275 # Newline in particular is a problem in a multicolumn table.
--> 276 col_strs = [val.replace('\t', '\\t').replace('\n', '\\n') for val in col_strs_iter]
    277 if len(col_strs) > 0:
    278     col_width = max(len(x) for x in col_strs)

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:276, in <listcomp>(.0)
    268 col_strs_iter = self._pformat_col_iter(col, max_lines, show_name=show_name,
    269                                        show_unit=show_unit,
    270                                        show_dtype=show_dtype,
    271                                        show_length=show_length,
    272                                        outs=outs)
    274 # Replace tab and newline with text representations so they display nicely.
    275 # Newline in particular is a problem in a multicolumn table.
--> 276 col_strs = [val.replace('\t', '\\t').replace('\n', '\\n') for val in col_strs_iter]
    277 if len(col_strs) > 0:
    278     col_width = max(len(x) for x in col_strs)

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:493, in TableFormatter._pformat_col_iter(self, col, max_lines, show_name, show_unit, outs, show_dtype, show_length)
    491 else:
    492     try:
--> 493         yield format_col_str(idx)
    494     except ValueError:
    495         raise ValueError(
    496             'Unable to parse format string "{}" for entry "{}" '
    497             'in column "{}"'.format(col_format, col[idx],
    498                                     col.info.name))

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:481, in TableFormatter._pformat_col_iter.<locals>.format_col_str(idx)
    479     return format_func(col_format, col[(idx,) + multidim0])
    480 else:
--> 481     left = format_func(col_format, col[(idx,) + multidim0])
    482     right = format_func(col_format, col[(idx,) + multidim1])
    483     return f'{left} .. {right}'

File astropy/table/_column_mixins.pyx:74, in astropy.table._column_mixins._ColumnGetitemShim.__getitem__()

File astropy/table/_column_mixins.pyx:57, in astropy.table._column_mixins.base_getitem()

File astropy/table/_column_mixins.pyx:69, in astropy.table._column_mixins.column_getitem()

IndexError: index 0 is out of bounds for axis 1 with size 0
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/IPython/core/formatters.py:343, in BaseFormatter.__call__(self, obj)
    341     method = get_real_method(obj, self.print_method)
    342     if method is not None:
--> 343         return method()
    344     return None
    345 else:

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/table.py:1526, in Table._repr_html_(self)
   1525 def _repr_html_(self):
-> 1526     out = self._base_repr_(html=True, max_width=-1,
   1527                            tableclass=conf.default_notebook_table_class)
   1528     # Wrap <table> in <div>. This follows the pattern in pandas and allows
   1529     # table to be scrollable horizontally in VS Code notebook display.
   1530     out = f'<div>{out}</div>'

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/table.py:1516, in Table._base_repr_(self, html, descr_vals, max_width, tableid, show_dtype, max_lines, tableclass)
   1513 if tableid is None:
   1514     tableid = f'table{id(self)}'
-> 1516 data_lines, outs = self.formatter._pformat_table(
   1517     self, tableid=tableid, html=html, max_width=max_width,
   1518     show_name=True, show_unit=None, show_dtype=show_dtype,
   1519     max_lines=max_lines, tableclass=tableclass)
   1521 out = descr + '\n'.join(data_lines)
   1523 return out

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:589, in TableFormatter._pformat_table(self, table, max_lines, max_width, show_name, show_unit, show_dtype, html, tableid, tableclass, align)
    586 if col.info.name not in pprint_include_names:
    587     continue
--> 589 lines, outs = self._pformat_col(col, max_lines, show_name=show_name,
    590                                 show_unit=show_unit, show_dtype=show_dtype,
    591                                 align=align_)
    592 if outs['show_length']:
    593     lines = lines[:-1]

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:276, in TableFormatter._pformat_col(self, col, max_lines, show_name, show_unit, show_dtype, show_length, html, align)
    268 col_strs_iter = self._pformat_col_iter(col, max_lines, show_name=show_name,
    269                                        show_unit=show_unit,
    270                                        show_dtype=show_dtype,
    271                                        show_length=show_length,
    272                                        outs=outs)
    274 # Replace tab and newline with text representations so they display nicely.
    275 # Newline in particular is a problem in a multicolumn table.
--> 276 col_strs = [val.replace('\t', '\\t').replace('\n', '\\n') for val in col_strs_iter]
    277 if len(col_strs) > 0:
    278     col_width = max(len(x) for x in col_strs)

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:276, in <listcomp>(.0)
    268 col_strs_iter = self._pformat_col_iter(col, max_lines, show_name=show_name,
    269                                        show_unit=show_unit,
    270                                        show_dtype=show_dtype,
    271                                        show_length=show_length,
    272                                        outs=outs)
    274 # Replace tab and newline with text representations so they display nicely.
    275 # Newline in particular is a problem in a multicolumn table.
--> 276 col_strs = [val.replace('\t', '\\t').replace('\n', '\\n') for val in col_strs_iter]
    277 if len(col_strs) > 0:
    278     col_width = max(len(x) for x in col_strs)

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:493, in TableFormatter._pformat_col_iter(self, col, max_lines, show_name, show_unit, outs, show_dtype, show_length)
    491 else:
    492     try:
--> 493         yield format_col_str(idx)
    494     except ValueError:
    495         raise ValueError(
    496             'Unable to parse format string "{}" for entry "{}" '
    497             'in column "{}"'.format(col_format, col[idx],
    498                                     col.info.name))

File ~/Applications/mambaforge/envs/swgo/lib/python3.9/site-packages/astropy/table/pprint.py:481, in TableFormatter._pformat_col_iter.<locals>.format_col_str(idx)
    479     return format_func(col_format, col[(idx,) + multidim0])
    480 else:
--> 481     left = format_func(col_format, col[(idx,) + multidim0])
    482     right = format_func(col_format, col[(idx,) + multidim1])
    483     return f'{left} .. {right}'

File astropy/table/_column_mixins.pyx:74, in astropy.table._column_mixins._ColumnGetitemShim.__getitem__()

File astropy/table/_column_mixins.pyx:57, in astropy.table._column_mixins.base_getitem()

File astropy/table/_column_mixins.pyx:69, in astropy.table._column_mixins.column_getitem()

IndexError: index 0 is out of bounds for axis 1 with size 0

\`\`\`

### Steps to Reproduce
<!-- Ideally a code example could be provided so we can run it ourselves. -->
<!-- If you are pasting code, use triple backticks (\`\`\`) around
your code snippet. -->
<!-- If necessary, sanitize your screen output to be pasted so you do not
reveal secrets like tokens and passwords. -->

This is an example dataset: field "B" set the length of field "C", so the first 2 events have an empty array in "C"
\`\`\`
events = [{"A":0,"B":0, "C":np.array([], dtype=np.uint64)},
          {"A":1,"B":0, "C":np.array([], dtype=np.uint64)},
          {"A":2,"B":2, "C":np.array([0,1], dtype=np.uint64)}]
\`\`\`
Showing just the first event prints the column names as a column,
<img width="196" alt="image" src="https://user-images.githubusercontent.com/17836610/195900814-50554a2b-8479-418c-b643-1c70018f5c0d.png">

Printing the first 2 throws the Traceback above
`QTable(rows=events[:2])`

Plotting all 3 events works

<img width="177" alt="image" src="https://user-images.githubusercontent.com/17836610/195901501-ba13445c-880e-4797-8619-d564c5e82de3.png">



### System Details
<!-- Even if you do not think this is necessary, it is useful information for the maintainers.
Please run the following snippet and paste the output below:
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("Numpy", numpy.__version__)
import erfa; print("pyerfa", erfa.__version__)
import astropy; print("astropy", astropy.__version__)
import scipy; print("Scipy", scipy.__version__)
import matplotlib; print("Matplotlib", matplotlib.__version__)
-->
macOS-11.7-x86_64-i386-64bit
Python 3.9.13 | packaged by conda-forge | (main, May 27 2022, 17:00:52) 
[Clang 13.0.1 ]
Numpy 1.23.3
pyerfa 2.0.0.1
astropy 5.1
Scipy 1.9.1
Matplotlib 3.6.0

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/table/pprint.py** | 710 | 772| 507 | 507 | 6312 | 
| 2 | 2 astropy/table/table.py | 2 | 74| 638 | 1145 | 39152 | 
| **-> 3 <-** | **2 astropy/table/pprint.py** | 473 | 485| 193 | 1338 | 39152 | 
| 4 | 3 astropy/io/votable/tree.py | 2557 | 2671| 915 | 2253 | 68126 | 
| 5 | **3 astropy/table/pprint.py** | 487 | 503| 160 | 2413 | 68126 | 
| 6 | **3 astropy/table/pprint.py** | 452 | 471| 256 | 2669 | 68126 | 
| 7 | **3 astropy/table/pprint.py** | 280 | 345| 715 | 3384 | 68126 | 
| 8 | 4 astropy/io/fits/column.py | 76 | 126| 753 | 4137 | 90463 | 
| 9 | 5 examples/template/example-template.py | 24 | 101| 553 | 4690 | 91144 | 
| 10 | 6 astropy/constants/constant.py | 33 | 74| 379 | 5069 | 93055 | 
| 11 | 6 astropy/table/table.py | 3857 | 3905| 495 | 5564 | 93055 | 
| 12 | 7 astropy/io/ascii/qdp.py | 519 | 609| 1053 | 6617 | 98102 | 
| 13 | **7 astropy/table/pprint.py** | 604 | 653| 467 | 7084 | 98102 | 
| 14 | 8 examples/io/fits-tables.py | 27 | 64| 196 | 7280 | 98482 | 
| 15 | 9 astropy/table/jsviewer.py | 30 | 105| 623 | 7903 | 100092 | 
| 16 | 10 astropy/io/ascii/ui.py | 489 | 561| 771 | 8674 | 107958 | 
| 17 | 10 astropy/table/table.py | 76 | 173| 713 | 9387 | 107958 | 
| 18 | 10 astropy/table/table.py | 1501 | 1527| 232 | 9619 | 107958 | 
| 19 | 11 astropy/table/__init__.py | 3 | 15| 188 | 9807 | 108785 | 
| 20 | 12 astropy/table/table_helpers.py | 9 | 55| 421 | 10228 | 110340 | 
| 21 | 13 astropy/table/scripts/showtable.py | 45 | 89| 326 | 10554 | 111659 | 
| 22 | 13 astropy/io/votable/tree.py | 2403 | 2500| 782 | 11336 | 111659 | 
| 23 | 13 astropy/table/__init__.py | 48 | 81| 324 | 11660 | 111659 | 
| 24 | 14 astropy/io/fits/hdu/table.py | 752 | 782| 312 | 11972 | 124761 | 
| 25 | 14 astropy/io/fits/column.py | 3 | 75| 758 | 12730 | 124761 | 
| 26 | 14 astropy/io/ascii/ui.py | 12 | 37| 157 | 12887 | 124761 | 
| 27 | 14 astropy/io/votable/tree.py | 2727 | 2790| 494 | 13381 | 124761 | 
| 28 | 14 astropy/table/scripts/showtable.py | 92 | 161| 691 | 14072 | 124761 | 
| 29 | 14 astropy/io/votable/tree.py | 2502 | 2555| 493 | 14565 | 124761 | 
| 30 | 14 astropy/io/votable/tree.py | 2875 | 2922| 403 | 14968 | 124761 | 
| 31 | 15 astropy/table/row.py | 142 | 187| 306 | 15274 | 126146 | 
| 32 | 15 astropy/table/table.py | 1876 | 1904| 342 | 15616 | 126146 | 
| 33 | 15 astropy/io/votable/tree.py | 5 | 66| 526 | 16142 | 126146 | 
| 34 | 15 astropy/io/fits/hdu/table.py | 1000 | 1051| 676 | 16818 | 126146 | 
| 35 | 15 astropy/table/table.py | 2036 | 2049| 144 | 16962 | 126146 | 
| 36 | **15 astropy/table/pprint.py** | 655 | 709| 460 | 17422 | 126146 | 
| 37 | 15 astropy/io/fits/hdu/table.py | 974 | 998| 241 | 17663 | 126146 | 
| 38 | 16 astropy/io/fits/diff.py | 1399 | 1450| 534 | 18197 | 139053 | 
| 39 | 16 astropy/table/table.py | 3041 | 3090| 451 | 18648 | 139053 | 
| 40 | 16 astropy/table/table.py | 2307 | 2323| 178 | 18826 | 139053 | 
| 41 | 17 astropy/table/pandas.py | 10 | 32| 126 | 18952 | 139866 | 
| 42 | 17 astropy/io/fits/hdu/table.py | 1237 | 1277| 366 | 19318 | 139866 | 
| 43 | 18 astropy/io/ascii/__init__.py | 7 | 28| 279 | 19597 | 140181 | 
| 44 | 18 astropy/table/table.py | 2167 | 2220| 558 | 20155 | 140181 | 
| 45 | 19 astropy/io/votable/exceptions.py | 1309 | 1348| 436 | 20591 | 153505 | 
| 46 | 19 astropy/table/table.py | 3714 | 3768| 506 | 21097 | 153505 | 
| 47 | 19 astropy/table/table.py | 274 | 295| 200 | 21297 | 153505 | 
| 48 | 20 astropy/table/column.py | 3 | 53| 388 | 21685 | 168159 | 
| 49 | 21 astropy/io/ascii/cds.py | 194 | 298| 1129 | 22814 | 171238 | 
| 50 | 22 astropy/wcs/docstrings.py | 938 | 1043| 728 | 23542 | 192691 | 
| 51 | 23 astropy/io/ascii/latex.py | 197 | 303| 1029 | 24571 | 196677 | 


### Hint

```
The root cause of this is that astropy delegates to numpy to convert a list of values into a numpy array. Notice the differences in output `dtype` here:
\`\`\`
In [25]: np.array([[], []])
Out[25]: array([], shape=(2, 0), dtype=float64)

In [26]: np.array([[], [], [1, 2]])
Out[26]: array([list([]), list([]), list([1, 2])], dtype=object)
\`\`\`
In your example you are expecting an `object` array of Python `lists` in both cases, but making this happen is not entirely practical since we rely on numpy for fast and general conversion of inputs.

The fact that a `Column` with a shape of `(2,0)` fails to print is indeed a bug, but for your use case it is likely not the real problem. In your examples if you ask for the `.info` attribute you will see this reflected.

As a workaround, a reliable way to get a true object array is something like:
\`\`\`
t = Table()
col = [[], []]
t["c"] = np.empty(len(col), dtype=object)
t["c"][:] = [[], []]
print(t)
 c 
---
 []
 []
\`\`\`

```

## Patch

```diff
diff --git a/astropy/table/pprint.py b/astropy/table/pprint.py
--- a/astropy/table/pprint.py
+++ b/astropy/table/pprint.py
@@ -392,7 +392,8 @@ def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
         if multidims:
             multidim0 = tuple(0 for n in multidims)
             multidim1 = tuple(n - 1 for n in multidims)
-            trivial_multidims = np.prod(multidims) == 1
+            multidims_all_ones = np.prod(multidims) == 1
+            multidims_has_zero = 0 in multidims
 
         i_dashes = None
         i_centers = []  # Line indexes where content should be centered
@@ -475,8 +476,11 @@ def format_col_str(idx):
                 # Prevents columns like Column(data=[[(1,)],[(2,)]], name='a')
                 # with shape (n,1,...,1) from being printed as if there was
                 # more than one element in a row
-                if trivial_multidims:
+                if multidims_all_ones:
                     return format_func(col_format, col[(idx,) + multidim0])
+                elif multidims_has_zero:
+                    # Any zero dimension means there is no data to print
+                    return ""
                 else:
                     left = format_func(col_format, col[(idx,) + multidim0])
                     right = format_func(col_format, col[(idx,) + multidim1])

```

## Test Patch

```diff
diff --git a/astropy/table/tests/test_pprint.py b/astropy/table/tests/test_pprint.py
--- a/astropy/table/tests/test_pprint.py
+++ b/astropy/table/tests/test_pprint.py
@@ -972,3 +972,18 @@ def test_embedded_newline_tab():
         r'   a b \n c \t \n d',
         r'   x            y\n']
     assert t.pformat_all() == exp
+
+
+def test_multidims_with_zero_dim():
+    """Test of fix for #13836 when a zero-dim column is present"""
+    t = Table()
+    t["a"] = ["a", "b"]
+    t["b"] = np.ones(shape=(2, 0, 1), dtype=np.float64)
+    exp = [
+        " a        b      ",
+        "str1 float64[0,1]",
+        "---- ------------",
+        "   a             ",
+        "   b             ",
+    ]
+    assert t.pformat_all(show_dtype=True) == exp

```


## Code snippets

### 1 - astropy/table/pprint.py:

Start line: 710, End line: 772

```python
class TableFormatter:

    def _more_tabcol(self, tabcol, max_lines=None, max_width=None,
                     show_name=True, show_unit=None, show_dtype=False):
        # ... other code
        while True:
            i1 = i0 + delta_lines  # Last table/col row to show
            if showlines:  # Don't always show the table (e.g. after help)
                try:
                    os.system('cls' if os.name == 'nt' else 'clear')
                except Exception:
                    pass  # No worries if clear screen call fails
                lines = tabcol[i0:i1].pformat(**kwargs)
                colors = ('red' if i < n_header else 'default'
                          for i in range(len(lines)))
                for color, line in zip(colors, lines):
                    color_print(line, color)
            showlines = True
            print()
            print("-- f, <space>, b, r, p, n, <, >, q h (help) --", end=' ')
            # Get a valid key
            while True:
                try:
                    key = inkey().lower()
                except Exception:
                    print("\n")
                    log.error('Console does not support getting a character'
                              ' as required by more().  Use pprint() instead.')
                    return
                if key in allowed_keys:
                    break
            print(key)

            if key.lower() == 'q':
                break
            elif key == ' ' or key == 'f':
                i0 += delta_lines
            elif key == 'b':
                i0 = i0 - delta_lines
            elif key == 'r':
                pass
            elif key == '<':
                i0 = 0
            elif key == '>':
                i0 = len(tabcol)
            elif key == 'p':
                i0 -= 1
            elif key == 'n':
                i0 += 1
            elif key == 'h':
                showlines = False
                print("""
                s:
                > : forward one page
                one page
                sh same page
                row
                ous row
                 beginning
                 end
                browsing
                 this help""", end=' ')
            if i0 < 0:
                i0 = 0
            if i0 >= len(tabcol) - delta_lines:
                i0 = len(tabcol) - delta_lines
            print("\n")
```
### 2 - astropy/table/table.py:

Start line: 2, End line: 74

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
    BaseColumn, Column, FalseArray, MaskedColumn, _auto_names, _convert_sequence_data_to_array,
    col_copy)
from .connect import TableRead, TableWrite
from .index import (
    Index, SlicedIndex, TableILoc, TableIndices, TableLoc, TableLocIndices, _IndexModeContext,
    get_index)
from .info import TableInfo
from .mixins.registry import get_mixin_handler
from .ndarray_mixin import NdarrayMixin
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

__doctest_skip__ = ['Table.read', 'Table.write', 'Table._read',
                    'Table.convert_bytestring_to_unicode',
                    'Table.convert_unicode_to_bytestring',
                    ]

__doctest_requires__ = {'*pandas': ['pandas>=1.1']}
```
### 3 - astropy/table/pprint.py:

Start line: 473, End line: 485

```python
class TableFormatter:

    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
                          show_dtype=False, show_length=None):
        # ... other code

        def format_col_str(idx):
            if multidims:
                # Prevents columns like Column(data=[[(1,)],[(2,)]], name='a')
                # with shape (n,1,...,1) from being printed as if there was
                # more than one element in a row
                if trivial_multidims:
                    return format_func(col_format, col[(idx,) + multidim0])
                else:
                    left = format_func(col_format, col[(idx,) + multidim0])
                    right = format_func(col_format, col[(idx,) + multidim1])
                    return f'{left} .. {right}'
            else:
                return format_func(col_format, col[idx])
        # ... other code
```
### 4 - astropy/io/votable/tree.py:

Start line: 2557, End line: 2671

```python
class Table(Element, _IDProperty, _NameProperty, _UcdProperty,
            _DescriptionProperty):

    def _parse_tabledata(self, iterator, colnumbers, fields, config):
        # Since we don't know the number of rows up front, we'll
        # reallocate the record array to make room as we go.  This
        # prevents the need to scan through the XML twice.  The
        # allocation is by factors of 1.5.
        invalid = config.get('invalid', 'exception')

        # Need to have only one reference so that we can resize the
        # array
        array = self.array
        del self.array

        parsers = [field.converter.parse for field in fields]
        binparsers = [field.converter.binparse for field in fields]

        numrows = 0
        alloc_rows = len(array)
        colnumbers_bits = [i in colnumbers for i in range(len(fields))]
        row_default = [x.converter.default for x in fields]
        mask_default = [True] * len(fields)
        array_chunk = []
        mask_chunk = []
        chunk_size = config.get('chunk_size', DEFAULT_CHUNK_SIZE)
        for start, tag, data, pos in iterator:
            if tag == 'TR':
                # Now parse one row
                row = row_default[:]
                row_mask = mask_default[:]
                i = 0
                for start, tag, data, pos in iterator:
                    if start:
                        binary = (data.get('encoding', None) == 'base64')
                        warn_unknown_attrs(
                            tag, data.keys(), config, pos, ['encoding'])
                    else:
                        if tag == 'TD':
                            if i >= len(fields):
                                vo_raise(E20, len(fields), config, pos)

                            if colnumbers_bits[i]:
                                try:
                                    if binary:
                                        rawdata = base64.b64decode(
                                            data.encode('ascii'))
                                        buf = io.BytesIO(rawdata)
                                        buf.seek(0)
                                        try:
                                            value, mask_value = binparsers[i](
                                                buf.read)
                                        except Exception as e:
                                            vo_reraise(
                                                e, config, pos,
                                                "(in row {:d}, col '{}')".format(
                                                    len(array_chunk),
                                                    fields[i].ID))
                                    else:
                                        try:
                                            value, mask_value = parsers[i](
                                                data, config, pos)
                                        except Exception as e:
                                            vo_reraise(
                                                e, config, pos,
                                                "(in row {:d}, col '{}')".format(
                                                    len(array_chunk),
                                                    fields[i].ID))
                                except Exception as e:
                                    if invalid == 'exception':
                                        vo_reraise(e, config, pos)
                                else:
                                    row[i] = value
                                    row_mask[i] = mask_value
                        elif tag == 'TR':
                            break
                        else:
                            self._add_unknown_tag(
                                iterator, tag, data, config, pos)
                        i += 1

                if i < len(fields):
                    vo_raise(E21, (i, len(fields)), config, pos)

                array_chunk.append(tuple(row))
                mask_chunk.append(tuple(row_mask))

                if len(array_chunk) == chunk_size:
                    while numrows + chunk_size > alloc_rows:
                        alloc_rows = self._resize_strategy(alloc_rows)
                    if alloc_rows != len(array):
                        array = _resize(array, alloc_rows)
                    array[numrows:numrows + chunk_size] = array_chunk
                    array.mask[numrows:numrows + chunk_size] = mask_chunk
                    numrows += chunk_size
                    array_chunk = []
                    mask_chunk = []

            elif not start and tag == 'TABLEDATA':
                break

        # Now, resize the array to the exact number of rows we need and
        # put the last chunk values in there.
        alloc_rows = numrows + len(array_chunk)

        array = _resize(array, alloc_rows)
        array[numrows:] = array_chunk
        if alloc_rows != 0:
            array.mask[numrows:] = mask_chunk
        numrows += len(array_chunk)

        if (self.nrows is not None and
            self.nrows >= 0 and
            self.nrows != numrows):
            warn_or_raise(W18, W18, (self.nrows, numrows), config, pos)
        self._nrows = numrows

        return array
```
### 5 - astropy/table/pprint.py:

Start line: 487, End line: 503

```python
class TableFormatter:

    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
                          show_dtype=False, show_length=None):

        # Add formatted values if within bounds allowed by max_lines
        for idx in indices:
            if idx == i0:
                yield '...'
            else:
                try:
                    yield format_col_str(idx)
                except ValueError:
                    raise ValueError(
                        'Unable to parse format string "{}" for entry "{}" '
                        'in column "{}"'.format(col_format, col[idx],
                                                col.info.name))

        outs['show_length'] = show_length
        outs['n_header'] = n_header
        outs['i_centers'] = i_centers
        outs['i_dashes'] = i_dashes
```
### 6 - astropy/table/pprint.py:

Start line: 452, End line: 471

```python
class TableFormatter:

    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
                          show_dtype=False, show_length=None):
        # - get_auto_format_func() returns a wrapped version of auto_format_func
        #    with the column id and possible_string_format_functions as
        #    enclosed variables.
        col_format = col.info.format or getattr(col.info, 'default_format',
                                                None)
        pssf = (getattr(col.info, 'possible_string_format_functions', None)
                or _possible_string_format_functions)
        auto_format_func = get_auto_format_func(col, pssf)
        format_func = col.info._format_funcs.get(col_format, auto_format_func)

        if len(col) > max_lines:
            if show_length is None:
                show_length = True
            i0 = n_print2 - (1 if show_length else 0)
            i1 = n_rows - n_print2 - max_lines % 2
            indices = np.concatenate([np.arange(0, i0 + 1),
                                      np.arange(i1 + 1, len(col))])
        else:
            i0 = -1
            indices = np.arange(len(col))
        # ... other code
```
### 7 - astropy/table/pprint.py:

Start line: 280, End line: 345

```python
class TableFormatter:

    def _pformat_col(self, col, max_lines=None, show_name=True, show_unit=None,
                     show_dtype=False, show_length=None, html=False, align=None):
        # ... other code

        if html:
            from astropy.utils.xml.writer import xml_escape
            n_header = outs['n_header']
            for i, col_str in enumerate(col_strs):
                # _pformat_col output has a header line '----' which is not needed here
                if i == n_header - 1:
                    continue
                td = 'th' if i < n_header else 'td'
                val = f'<{td}>{xml_escape(col_str.strip())}</{td}>'
                row = ('<tr>' + val + '</tr>')
                if i < n_header:
                    row = ('<thead>' + row + '</thead>')
                col_strs[i] = row

            if n_header > 0:
                # Get rid of '---' header line
                col_strs.pop(n_header - 1)
            col_strs.insert(0, '<table>')
            col_strs.append('</table>')

        # Now bring all the column string values to the same fixed width
        else:
            col_width = max(len(x) for x in col_strs) if col_strs else 1

            # Center line header content and generate dashed headerline
            for i in outs['i_centers']:
                col_strs[i] = col_strs[i].center(col_width)
            if outs['i_dashes'] is not None:
                col_strs[outs['i_dashes']] = '-' * col_width

            # Format columns according to alignment.  `align` arg has precedent, otherwise
            # use `col.format` if it starts as a legal alignment string.  If neither applies
            # then right justify.
            re_fill_align = re.compile(r'(?P<fill>.?)(?P<align>[<^>=])')
            match = None
            if align:
                # If there is an align specified then it must match
                match = re_fill_align.match(align)
                if not match:
                    raise ValueError("column align must be one of '<', '^', '>', or '='")
            elif isinstance(col.info.format, str):
                # col.info.format need not match, in which case rjust gets used
                match = re_fill_align.match(col.info.format)

            if match:
                fill_char = match.group('fill')
                align_char = match.group('align')
                if align_char == '=':
                    if fill_char != '0':
                        raise ValueError("fill character must be '0' for '=' align")
                    fill_char = ''  # str.zfill gets used which does not take fill char arg
            else:
                fill_char = ''
                align_char = '>'

            justify_methods = {'<': 'ljust', '^': 'center', '>': 'rjust', '=': 'zfill'}
            justify_method = justify_methods[align_char]
            justify_args = (col_width, fill_char) if fill_char else (col_width,)

            for i, col_str in enumerate(col_strs):
                col_strs[i] = getattr(col_str, justify_method)(*justify_args)

        if outs['show_length']:
            col_strs.append(f'Length = {len(col)} rows')

        return col_strs, outs
```
### 8 - astropy/io/fits/column.py:

Start line: 76, End line: 126

```python
ASCII_DEFAULT_WIDTHS = {'A': (1, 0), 'I': (10, 0), 'J': (15, 0),
                        'E': (15, 7), 'F': (16, 7), 'D': (25, 17)}

# TDISPn for both ASCII and Binary tables
TDISP_RE_DICT = {}
TDISP_RE_DICT['F'] = re.compile(r'(?:(?P<formatc>[F])(?:(?P<width>[0-9]+)\.{1}'
                                r'(?P<precision>[0-9])+)+)|')
TDISP_RE_DICT['A'] = TDISP_RE_DICT['L'] = \
    re.compile(r'(?:(?P<formatc>[AL])(?P<width>[0-9]+)+)|')
TDISP_RE_DICT['I'] = TDISP_RE_DICT['B'] = \
    TDISP_RE_DICT['O'] = TDISP_RE_DICT['Z'] =  \
    re.compile(r'(?:(?P<formatc>[IBOZ])(?:(?P<width>[0-9]+)'
               r'(?:\.{0,1}(?P<precision>[0-9]+))?))|')
TDISP_RE_DICT['E'] = TDISP_RE_DICT['G'] = \
    TDISP_RE_DICT['D'] = \
    re.compile(r'(?:(?P<formatc>[EGD])(?:(?P<width>[0-9]+)\.'
               r'(?P<precision>[0-9]+))+)'
               r'(?:E{0,1}(?P<exponential>[0-9]+)?)|')
TDISP_RE_DICT['EN'] = TDISP_RE_DICT['ES'] = \
    re.compile(r'(?:(?P<formatc>E[NS])(?:(?P<width>[0-9]+)\.{1}'
               r'(?P<precision>[0-9])+)+)')

# mapping from TDISP format to python format
# A: Character
# L: Logical (Boolean)
# I: 16-bit Integer
#    Can't predefine zero padding and space padding before hand without
#    knowing the value being formatted, so grabbing precision and using that
#    to zero pad, ignoring width. Same with B, O, and Z
# B: Binary Integer
# O: Octal Integer
# Z: Hexadecimal Integer
# F: Float (64-bit; fixed decimal notation)
# EN: Float (engineering fortran format, exponential multiple of thee
# ES: Float (scientific, same as EN but non-zero leading digit
# E: Float, exponential notation
#    Can't get exponential restriction to work without knowing value
#    before hand, so just using width and precision, same with D, G, EN, and
#    ES formats
# D: Double-precision Floating Point with exponential
#    (E but for double precision)
# G: Double-precision Floating Point, may or may not show exponent
TDISP_FMT_DICT = {
    'I': '{{:{width}d}}',
    'B': '{{:{width}b}}',
    'O': '{{:{width}o}}',
    'Z': '{{:{width}x}}',
    'F': '{{:{width}.{precision}f}}',
    'G': '{{:{width}.{precision}g}}'
}
TDISP_FMT_DICT['A'] = TDISP_FMT_DICT['L'] = '{{:>{width}}}'
```
### 9 - examples/template/example-template.py:

Start line: 24, End line: 101

```python
import matplotlib.pyplot as plt
import numpy as np

from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)
# uncomment if including figures:
# import matplotlib.pyplot as plt
# from astropy.visualization import astropy_mpl_style
# plt.style.use(astropy_mpl_style)

##############################################################################
# This code block is executed, although it produces no output. Lines starting
# with a simple hash are code comment and get treated as part of the code
# block. To include this new comment string we started the new block with a
# long line of hashes.
#
# The sphinx-gallery parser will assume everything after this splitter and that
# continues to start with a **comment hash and space** (respecting code style)
# is text that has to be rendered in
# html format. Keep in mind to always keep your comments always together by
# comment hashes. That means to break a paragraph you still need to comment
# that line break.
#
# In this example the next block of code produces some plotable data. Code is
# executed, figure is saved and then code is presented next, followed by the
# inlined figure.

x = np.linspace(-np.pi, np.pi, 300)
xx, yy = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')

###########################################################################
# Again it is possible to continue the discussion with a new Python string. This
# time to introduce the next code block generates 2 separate figures.

plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('hot'))
plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('Spectral'), interpolation='none')

##########################################################################
# There's some subtle differences between rendered html rendered comment
# strings and code comment strings which I'll demonstrate below. (Some of this
# only makes sense if you look at the
# :download:`raw Python script <plot_notebook.py>`)
#
# Comments in comment blocks remain nested in the text.


def dummy():
    """Dummy function to make sure docstrings don't get rendered as text"""
    pass


# Code comments not preceded by the hash splitter are left in code blocks.

string = """
Triple-quoted string which tries to break parser but doesn't.
"""

############################################################################
# Output of the script is captured:

print('Some output from Python')

############################################################################
# Finally, I'll call ``show`` at the end just so someone running the Python
# code directly will see the plots; this is not necessary for creating the docs

plt.show()
```
### 10 - astropy/constants/constant.py:

Start line: 33, End line: 74

```python
class ConstantMeta(type):

    def __new__(mcls, name, bases, d):
        def wrap(meth):
            @functools.wraps(meth)
            def wrapper(self, *args, **kwargs):
                name_lower = self.name.lower()
                instances = self._registry[name_lower]
                if not self._checked_units:
                    for inst in instances.values():
                        try:
                            self.unit.to(inst.unit)
                        except UnitsError:
                            self._has_incompatible_units.add(name_lower)
                    self._checked_units = True

                if (not self.system and
                        name_lower in self._has_incompatible_units):
                    systems = sorted(x for x in instances if x)
                    raise TypeError(
                        'Constant {!r} does not have physically compatible '
                        'units across all systems of units and cannot be '
                        'combined with other values without specifying a '
                        'system (eg. {}.{})'.format(self.abbrev, self.abbrev,
                                                    systems[0]))

                return meth(self, *args, **kwargs)

            return wrapper

        # The wrapper applies to so many of the __ methods that it's easier to
        # just exclude the ones it doesn't apply to
        exclude = {'__new__', '__array_finalize__', '__array_wrap__',
                   '__dir__', '__getattr__', '__init__', '__str__',
                   '__repr__', '__hash__', '__iter__', '__getitem__',
                   '__len__', '__bool__', '__quantity_subclass__',
                   '__setstate__'}
        for attr, value in vars(Quantity).items():
            if (isinstance(value, types.FunctionType) and
                    attr.startswith('__') and attr.endswith('__') and
                    attr not in exclude):
                d[attr] = wrap(value)

        return super().__new__(mcls, name, bases, d)
```
### 13 - astropy/table/pprint.py:

Start line: 604, End line: 653

```python
class TableFormatter:

    def _pformat_table(self, table, max_lines=None, max_width=None,
                       show_name=True, show_unit=None, show_dtype=False,
                       html=False, tableid=None, tableclass=None, align=None):
        # ... other code

        def outwidth(cols):
            return sum(len(c[0]) for c in cols) + len(cols) - 1

        dots_col = ['...'] * n_rows
        middle = len(cols) // 2
        while outwidth(cols) > max_width:
            if len(cols) == 1:
                break
            if len(cols) == 2:
                cols[1] = dots_col
                break
            if cols[middle] is dots_col:
                cols.pop(middle)
                middle = len(cols) // 2
            cols[middle] = dots_col

        # Now "print" the (already-stringified) column values into a
        # row-oriented list.
        rows = []
        if html:
            from astropy.utils.xml.writer import xml_escape

            if tableid is None:
                tableid = f'table{id(table)}'

            if tableclass is not None:
                if isinstance(tableclass, list):
                    tableclass = ' '.join(tableclass)
                rows.append(f'<table id="{tableid}" class="{tableclass}">')
            else:
                rows.append(f'<table id="{tableid}">')

            for i in range(n_rows):
                # _pformat_col output has a header line '----' which is not needed here
                if i == n_header - 1:
                    continue
                td = 'th' if i < n_header else 'td'
                vals = (f'<{td}>{xml_escape(col[i].strip())}</{td}>'
                        for col in cols)
                row = ('<tr>' + ''.join(vals) + '</tr>')
                if i < n_header:
                    row = ('<thead>' + row + '</thead>')
                rows.append(row)
            rows.append('</table>')
        else:
            for i in range(n_rows):
                row = ' '.join(col[i] for col in cols)
                rows.append(row)

        return rows, outs
```
### 36 - astropy/table/pprint.py:

Start line: 655, End line: 709

```python
class TableFormatter:

    def _more_tabcol(self, tabcol, max_lines=None, max_width=None,
                     show_name=True, show_unit=None, show_dtype=False):
        """Interactive "more" of a table or column.

        Parameters
        ----------
        max_lines : int or None
            Maximum number of rows to output

        max_width : int or None
            Maximum character width of output

        show_name : bool
            Include a header row for column names. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        show_dtype : bool
            Include a header row for column dtypes. Default is False.
        """
        allowed_keys = 'f br<>qhpn'

        # Count the header lines
        n_header = 0
        if show_name:
            n_header += 1
        if show_unit:
            n_header += 1
        if show_dtype:
            n_header += 1
        if show_name or show_unit or show_dtype:
            n_header += 1

        # Set up kwargs for pformat call.  Only Table gets max_width.
        kwargs = dict(max_lines=-1, show_name=show_name, show_unit=show_unit,
                      show_dtype=show_dtype)
        if hasattr(tabcol, 'columns'):  # tabcol is a table
            kwargs['max_width'] = max_width

        # If max_lines is None (=> query screen size) then increase by 2.
        # This is because get_pprint_size leaves 6 extra lines so that in
        # ipython you normally see the last input line.
        max_lines1, max_width = self._get_pprint_size(max_lines, max_width)
        if max_lines is None:
            max_lines1 += 2
        delta_lines = max_lines1 - n_header

        # Set up a function to get a single character on any platform
        inkey = Getch()

        i0 = 0  # First table/column row to show
        showlines = True
        # ... other code
```
