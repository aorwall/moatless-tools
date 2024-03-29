# matplotlib__matplotlib-22929

| **matplotlib/matplotlib** | `89b21b517df0b2a9c378913bae8e1f184988b554` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 3026 |
| **Avg pos** | 4.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 4 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -1083,10 +1083,10 @@ def hlines(self, y, xmin, xmax, colors=None, linestyles='solid',
         lines._internal_update(kwargs)
 
         if len(y) > 0:
-            minx = min(xmin.min(), xmax.min())
-            maxx = max(xmin.max(), xmax.max())
-            miny = y.min()
-            maxy = y.max()
+            minx = min(np.nanmin(xmin), np.nanmin(xmax))
+            maxx = max(np.nanmax(xmin), np.nanmax(xmax))
+            miny = np.nanmin(y)
+            maxy = np.nanmax(y)
 
             corners = (minx, miny), (maxx, maxy)
 
@@ -1162,10 +1162,10 @@ def vlines(self, x, ymin, ymax, colors=None, linestyles='solid',
         lines._internal_update(kwargs)
 
         if len(x) > 0:
-            minx = x.min()
-            maxx = x.max()
-            miny = min(ymin.min(), ymax.min())
-            maxy = max(ymin.max(), ymax.max())
+            minx = np.nanmin(x)
+            maxx = np.nanmax(x)
+            miny = min(np.nanmin(ymin), np.nanmin(ymax))
+            maxy = max(np.nanmax(ymin), np.nanmax(ymax))
 
             corners = (minx, miny), (maxx, maxy)
             self.update_datalim(corners)
@@ -2674,7 +2674,7 @@ def sign(x):
                 extrema = max(x0, x1) if dat >= 0 else min(x0, x1)
                 length = abs(x0 - x1)
 
-            if err is None:
+            if err is None or np.size(err) == 0:
                 endpt = extrema
             elif orientation == "vertical":
                 endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
@@ -3504,7 +3504,9 @@ def apply_mask(arrays, mask): return [array[mask] for array in arrays]
                     f"'{dep_axis}err' (shape: {np.shape(err)}) must be a "
                     f"scalar or a 1D or (2, n) array-like whose shape matches "
                     f"'{dep_axis}' (shape: {np.shape(dep)})") from None
-            if np.any(err < -err):  # like err<0, but also works for timedelta.
+            res = np.zeros_like(err, dtype=bool)  # Default in case of nan
+            if np.any(np.less(err, -err, out=res, where=(err == err))):
+                # like err<0, but also works for timedelta and nan.
                 raise ValueError(
                     f"'{dep_axis}err' must not contain negative values")
             # This is like

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/axes/_axes.py | 1086 | 1089 | - | 4 | -
| lib/matplotlib/axes/_axes.py | 1165 | 1168 | - | 4 | -
| lib/matplotlib/axes/_axes.py | 2677 | 2677 | - | 4 | -
| lib/matplotlib/axes/_axes.py | 3507 | 3507 | 4 | 4 | 3026


## Problem Statement

```
[Bug]: bar_label fails with nan errorbar values
### Bug summary

`ax.bar_label` appears not to be robust to bars with missing (nan) values when also including error values. This issue is similar to [#20058](https://github.com/matplotlib/matplotlib/issues/20058/), but occurs in each of three cases:

Case 1.  When a dependent value is missing.
Case 2.  When an error value is missing.
Case 3.  When both a dependent value and an error value are missing.

The error seems to happen here, but I don't know the code well enough to pinpoint what should change to fix this:
https://github.com/matplotlib/matplotlib/blob/925b27ff3ab3d3bff621695fccfd49a7e095d329/lib/matplotlib/axes/_axes.py#L2677-L2682

### Code for reproduction

\`\`\`python
#%% Case 1: Missing dependent value
import matplotlib.pyplot as plt
import numpy as np
ax = plt.gca()
bars = ax.bar([0, 1, 2], [np.nan, 0.3, 0.4], yerr=[1, 0.1, 0.1])
ax.bar_label(bars)

#%% Case 2: Missing error value
import matplotlib.pyplot as plt
import numpy as np
ax = plt.gca()
bars = ax.bar([0, 1, 2], [0, 0.3, 0.4], yerr=[np.nan, 0.1, 0.1])
ax.bar_label(bars)

#%% Case 3: Missing dependent and error values
import matplotlib.pyplot as plt
import numpy as np
ax = plt.gca()
bars = ax.bar([0, 1, 2], [np.nan, 0.3, 0.4], yerr=[np.nan, 0.1, 0.1])
ax.bar_label(bars)
\`\`\`


### Actual outcome

runcell('Case 3: Missing dependent and error values', 'C:/Users/jam/Documents/GitHub/ci-greedy-agents-base/untitled2.py')
Traceback (most recent call last):

  File "C:\ProgramData\Miniconda3\lib\site-packages\spyder_kernels\py3compat.py", line 356, in compat_exec
    exec(code, globals, locals)

  File "c:\users\jam\documents\github\ci-greedy-agents-base\untitled2.py", line 27, in <module>
    ax.bar_label(bars)

  File "C:\ProgramData\Miniconda3\lib\site-packages\matplotlib\axes\_axes.py", line 2641, in bar_label
    endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()

IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

### Expected outcome

Maybe either raise an error telling me what I should do instead, or have the code resolve whatever the source is on the backend? Ideally, I think the following should happen:

Case 1. Raise an error that there is no value to apply the errorbar value to.
Cases 2 & 3. Ignore the missing value and move on to the next.

### Additional information

_No response_

### Operating system

Windows 10.1

### Matplotlib Version

3.5.1

### Matplotlib Backend

module://matplotlib_inline.backend_inline

### Python version

3.9.5

### Jupyter version

Spyder 5.3.0

### Installation

conda

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/lines_bars_and_markers/bar_label_demo.py | 1 | 107| 820 | 820 | 820 | 
| 2 | 2 lib/matplotlib/pyplot.py | 2505 | 2517| 229 | 1049 | 27405 | 
| 3 | 3 examples/statistics/errorbar_limits.py | 1 | 86| 841 | 1890 | 28246 | 
| **-> 4 <-** | **4 lib/matplotlib/axes/_axes.py** | 3483 | 3863| 1136 | 3026 | 101815 | 
| 5 | 5 examples/lines_bars_and_markers/broken_barh.py | 1 | 27| 247 | 3273 | 102062 | 
| 6 | 6 plot_types/stats/errorbar_plot.py | 1 | 28| 175 | 3448 | 102237 | 
| 7 | 7 examples/lines_bars_and_markers/errorbar_limits_simple.py | 1 | 64| 529 | 3977 | 102766 | 
| 8 | 7 lib/matplotlib/pyplot.py | 2396 | 2417| 231 | 4208 | 102766 | 
| 9 | 8 examples/statistics/errorbar_features.py | 1 | 57| 409 | 4617 | 103175 | 
| 10 | 9 examples/statistics/errorbar.py | 1 | 30| 167 | 4784 | 103342 | 
| 11 | 9 lib/matplotlib/pyplot.py | 2445 | 2456| 146 | 4930 | 103342 | 
| 12 | 10 examples/lines_bars_and_markers/filled_step.py | 179 | 237| 493 | 5423 | 104987 | 
| 13 | 11 examples/statistics/errorbars_and_boxes.py | 63 | 81| 117 | 5540 | 105633 | 
| 14 | **11 lib/matplotlib/axes/_axes.py** | 2894 | 3482| 5596 | 11136 | 105633 | 
| 15 | 12 examples/lines_bars_and_markers/errorbar_subsample.py | 1 | 41| 360 | 11496 | 105993 | 
| 16 | 13 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 11799 | 106296 | 
| 17 | 14 examples/lines_bars_and_markers/curve_error_band.py | 66 | 89| 209 | 12008 | 107103 | 
| 18 | 15 examples/mplot3d/errorbar3d.py | 1 | 30| 228 | 12236 | 107331 | 
| 19 | 15 examples/statistics/errorbars_and_boxes.py | 42 | 60| 189 | 12425 | 107331 | 
| 20 | 16 lib/mpl_toolkits/axes_grid1/axes_grid.py | 20 | 51| 229 | 12654 | 112220 | 
| 21 | 17 lib/matplotlib/colorbar.py | 336 | 491| 1302 | 13956 | 126899 | 
| 22 | 18 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 82| 738 | 14694 | 127768 | 
| 23 | 19 lib/mpl_toolkits/mplot3d/axes3d.py | 2776 | 2882| 1206 | 15900 | 157695 | 
| 24 | 19 lib/matplotlib/pyplot.py | 2083 | 2096| 138 | 16038 | 157695 | 
| 25 | 20 lib/matplotlib/legend_handler.py | 538 | 620| 756 | 16794 | 164297 | 
| 26 | 21 examples/lines_bars_and_markers/masked_demo.py | 1 | 52| 410 | 17204 | 164707 | 
| 27 | 22 examples/images_contours_and_fields/barb_demo.py | 1 | 66| 663 | 17867 | 165370 | 
| 28 | 23 examples/mplot3d/3d_bars.py | 1 | 35| 212 | 18079 | 165582 | 
| 29 | 24 examples/lines_bars_and_markers/barchart.py | 1 | 47| 321 | 18400 | 165903 | 
| 30 | 25 examples/lines_bars_and_markers/barh.py | 1 | 31| 179 | 18579 | 166082 | 
| 31 | 26 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 19104 | 166607 | 
| 32 | 27 examples/specialty_plots/leftventricle_bulleye.py | 130 | 193| 708 | 19812 | 168774 | 
| 33 | 28 examples/misc/anchored_artists.py | 71 | 94| 132 | 19944 | 169567 | 
| 34 | 29 examples/lines_bars_and_markers/bar_stacked.py | 1 | 33| 260 | 20204 | 169827 | 
| 35 | 29 lib/mpl_toolkits/mplot3d/axes3d.py | 3108 | 3127| 409 | 20613 | 169827 | 
| 36 | 30 tutorials/colors/colorbar_only.py | 1 | 86| 734 | 21347 | 170918 | 
| 37 | 31 examples/units/bar_demo2.py | 1 | 38| 359 | 21706 | 171277 | 
| 38 | 31 lib/mpl_toolkits/mplot3d/axes3d.py | 3049 | 3106| 862 | 22568 | 171277 | 
| 39 | 32 examples/ticks/ticks_too_many.py | 1 | 77| 742 | 23310 | 172019 | 
| 40 | 33 examples/axes_grid1/simple_anchored_artists.py | 61 | 85| 150 | 23460 | 172634 | 
| 41 | 34 examples/color/colorbar_basics.py | 1 | 59| 506 | 23966 | 173140 | 
| 42 | 34 tutorials/colors/colorbar_only.py | 88 | 131| 357 | 24323 | 173140 | 
| 43 | 35 plot_types/basic/bar.py | 1 | 26| 164 | 24487 | 173304 | 
| 44 | 36 tutorials/introductory/lifecycle.py | 98 | 188| 740 | 25227 | 175632 | 
| 45 | 36 lib/mpl_toolkits/mplot3d/axes3d.py | 2883 | 2941| 781 | 26008 | 175632 | 
| 46 | 37 examples/style_sheets/style_sheets_reference.py | 46 | 54| 135 | 26143 | 177067 | 
| 47 | 38 examples/images_contours_and_fields/barcode_demo.py | 1 | 49| 618 | 26761 | 177685 | 
| 48 | 38 lib/matplotlib/legend_handler.py | 516 | 536| 177 | 26938 | 177685 | 
| 49 | 39 examples/pie_and_polar_charts/bar_of_pie.py | 1 | 83| 791 | 27729 | 178476 | 
| 50 | 39 examples/lines_bars_and_markers/curve_error_band.py | 27 | 63| 443 | 28172 | 178476 | 
| 51 | 40 examples/mplot3d/bars3d.py | 1 | 43| 313 | 28485 | 178789 | 


### Hint

```
I have a solution that works when running in the shell, but not in the test as I get a runtime warning because of the nan-values. Will see if I can find a solution in the next few days.
```

## Patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -1083,10 +1083,10 @@ def hlines(self, y, xmin, xmax, colors=None, linestyles='solid',
         lines._internal_update(kwargs)
 
         if len(y) > 0:
-            minx = min(xmin.min(), xmax.min())
-            maxx = max(xmin.max(), xmax.max())
-            miny = y.min()
-            maxy = y.max()
+            minx = min(np.nanmin(xmin), np.nanmin(xmax))
+            maxx = max(np.nanmax(xmin), np.nanmax(xmax))
+            miny = np.nanmin(y)
+            maxy = np.nanmax(y)
 
             corners = (minx, miny), (maxx, maxy)
 
@@ -1162,10 +1162,10 @@ def vlines(self, x, ymin, ymax, colors=None, linestyles='solid',
         lines._internal_update(kwargs)
 
         if len(x) > 0:
-            minx = x.min()
-            maxx = x.max()
-            miny = min(ymin.min(), ymax.min())
-            maxy = max(ymin.max(), ymax.max())
+            minx = np.nanmin(x)
+            maxx = np.nanmax(x)
+            miny = min(np.nanmin(ymin), np.nanmin(ymax))
+            maxy = max(np.nanmax(ymin), np.nanmax(ymax))
 
             corners = (minx, miny), (maxx, maxy)
             self.update_datalim(corners)
@@ -2674,7 +2674,7 @@ def sign(x):
                 extrema = max(x0, x1) if dat >= 0 else min(x0, x1)
                 length = abs(x0 - x1)
 
-            if err is None:
+            if err is None or np.size(err) == 0:
                 endpt = extrema
             elif orientation == "vertical":
                 endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
@@ -3504,7 +3504,9 @@ def apply_mask(arrays, mask): return [array[mask] for array in arrays]
                     f"'{dep_axis}err' (shape: {np.shape(err)}) must be a "
                     f"scalar or a 1D or (2, n) array-like whose shape matches "
                     f"'{dep_axis}' (shape: {np.shape(dep)})") from None
-            if np.any(err < -err):  # like err<0, but also works for timedelta.
+            res = np.zeros_like(err, dtype=bool)  # Default in case of nan
+            if np.any(np.less(err, -err, out=res, where=(err == err))):
+                # like err<0, but also works for timedelta and nan.
                 raise ValueError(
                     f"'{dep_axis}err' must not contain negative values")
             # This is like

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -7549,6 +7549,26 @@ def test_bar_label_nan_ydata_inverted():
     assert labels[0].get_va() == 'bottom'
 
 
+def test_nan_barlabels():
+    fig, ax = plt.subplots()
+    bars = ax.bar([1, 2, 3], [np.nan, 1, 2], yerr=[0.2, 0.4, 0.6])
+    labels = ax.bar_label(bars)
+    assert [l.get_text() for l in labels] == ['', '1', '2']
+    assert np.allclose(ax.get_ylim(), (0.0, 3.0))
+
+    fig, ax = plt.subplots()
+    bars = ax.bar([1, 2, 3], [0, 1, 2], yerr=[0.2, np.nan, 0.6])
+    labels = ax.bar_label(bars)
+    assert [l.get_text() for l in labels] == ['0', '1', '2']
+    assert np.allclose(ax.get_ylim(), (-0.5, 3.0))
+
+    fig, ax = plt.subplots()
+    bars = ax.bar([1, 2, 3], [np.nan, 1, 2], yerr=[np.nan, np.nan, 0.6])
+    labels = ax.bar_label(bars)
+    assert [l.get_text() for l in labels] == ['', '1', '2']
+    assert np.allclose(ax.get_ylim(), (0.0, 3.0))
+
+
 def test_patch_bounds():  # PR 19078
     fig, ax = plt.subplots()
     ax.add_patch(mpatches.Wedge((0, -1), 1.05, 60, 120, 0.1))

```


## Code snippets

### 1 - examples/lines_bars_and_markers/bar_label_demo.py:

Start line: 1, End line: 107

```python
"""
==============
Bar Label Demo
==============

This example shows how to use the `~.Axes.bar_label` helper function
to create bar chart labels.

See also the :doc:`grouped bar
</gallery/lines_bars_and_markers/barchart>`,
:doc:`stacked bar
</gallery/lines_bars_and_markers/bar_stacked>` and
:doc:`horizontal bar chart
</gallery/lines_bars_and_markers/barh>` examples.
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Define the data

N = 5
menMeans = (20, 35, 30, 35, -27)
womenMeans = (25, 32, 34, 20, -25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

###############################################################################
# Stacked bar plot with error bars

fig, ax = plt.subplots()

p1 = ax.bar(ind, menMeans, width, yerr=menStd, label='Men')
p2 = ax.bar(ind, womenMeans, width,
            bottom=menMeans, yerr=womenStd, label='Women')

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind, labels=['G1', 'G2', 'G3', 'G4', 'G5'])
ax.legend()

# Label with label_type 'center' instead of the default 'edge'
ax.bar_label(p1, label_type='center')
ax.bar_label(p2, label_type='center')
ax.bar_label(p2)

plt.show()

###############################################################################
# Horizontal bar chart

# Fixing random state for reproducibility
np.random.seed(19680801)

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

fig, ax = plt.subplots()

hbars = ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

# Label with specially formatted floats
ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(right=15)  # adjust xlim to fit labels

plt.show()

###############################################################################
# Some of the more advanced things that one can do with bar labels

fig, ax = plt.subplots()

hbars = ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

# Label with given captions, custom padding and annotate options
ax.bar_label(hbars, labels=['±%.2f' % e for e in error],
             padding=8, color='b', fontsize=14)
ax.set_xlim(right=16)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
```
### 2 - lib/matplotlib/pyplot.py:

Start line: 2505, End line: 2517

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.errorbar)
def errorbar(
        x, y, yerr=None, xerr=None, fmt='', ecolor=None,
        elinewidth=None, capsize=None, barsabove=False, lolims=False,
        uplims=False, xlolims=False, xuplims=False, errorevery=1,
        capthick=None, *, data=None, **kwargs):
    return gca().errorbar(
        x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
        elinewidth=elinewidth, capsize=capsize, barsabove=barsabove,
        lolims=lolims, uplims=uplims, xlolims=xlolims,
        xuplims=xuplims, errorevery=errorevery, capthick=capthick,
        **({"data": data} if data is not None else {}), **kwargs)
```
### 3 - examples/statistics/errorbar_limits.py:

Start line: 1, End line: 86

```python
"""
==============================================
Including upper and lower limits in error bars
==============================================

In matplotlib, errors bars can have "limits". Applying limits to the
error bars essentially makes the error unidirectional. Because of that,
upper and lower limits can be applied in both the y- and x-directions
via the ``uplims``, ``lolims``, ``xuplims``, and ``xlolims`` parameters,
respectively. These parameters can be scalar or boolean arrays.

For example, if ``xlolims`` is ``True``, the x-error bars will only
extend from the data towards increasing values. If ``uplims`` is an
array filled with ``False`` except for the 4th and 7th values, all of the
y-error bars will be bidirectional, except the 4th and 7th bars, which
will extend from the data towards decreasing y-values.
"""

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y = np.exp(-x)
xerr = 0.1
yerr = 0.2

# lower & upper limits of the error
lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
ls = 'dotted'

fig, ax = plt.subplots(figsize=(7, 4))

# standard error bars
ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)

# including upper limits
ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims,
            linestyle=ls)

# including lower limits
ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims,
            linestyle=ls)

# including upper and lower limits
ax.errorbar(x, y + 1.5, xerr=xerr, yerr=yerr,
            lolims=lolims, uplims=uplims,
            marker='o', markersize=8,
            linestyle=ls)

# Plot a series with lower and upper limits in both x & y
# constant x-error with varying y-error
xerr = 0.2
yerr = np.full_like(x, 0.2)
yerr[[3, 6]] = 0.3

# mock up some limits by modifying previous data
xlolims = lolims
xuplims = uplims
lolims = np.zeros_like(x)
uplims = np.zeros_like(x)
lolims[[6]] = True  # only limited at this index
uplims[[3]] = True  # only limited at this index

# do the plotting
ax.errorbar(x, y + 2.1, xerr=xerr, yerr=yerr,
            xlolims=xlolims, xuplims=xuplims,
            uplims=uplims, lolims=lolims,
            marker='o', markersize=8,
            linestyle='none')

# tidy up the figure
ax.set_xlim((0, 5.5))
ax.set_title('Errorbar upper and lower limits')
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```
### 4 - lib/matplotlib/axes/_axes.py:

Start line: 3483, End line: 3863

```python
@_docstring.interpd
class Axes(_AxesBase):

    @_preprocess_data(replace_names=["x", "y", "xerr", "yerr"],
                      label_namer="y")
    @_docstring.dedent_interpd
    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='', ecolor=None, elinewidth=None, capsize=None,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, errorevery=1, capthick=None,
                 **kwargs):
        # ... other code
        for (dep_axis, dep, err, lolims, uplims, indep, lines_func,
             marker, lomarker, himarker) in [
                ("x", x, xerr, xlolims, xuplims, y, self.hlines,
                 "|", mlines.CARETRIGHTBASE, mlines.CARETLEFTBASE),
                ("y", y, yerr, lolims, uplims, x, self.vlines,
                 "_", mlines.CARETUPBASE, mlines.CARETDOWNBASE),
        ]:
            if err is None:
                continue
            lolims = np.broadcast_to(lolims, len(dep)).astype(bool)
            uplims = np.broadcast_to(uplims, len(dep)).astype(bool)
            try:
                np.broadcast_to(err, (2, len(dep)))
            except ValueError:
                raise ValueError(
                    f"'{dep_axis}err' (shape: {np.shape(err)}) must be a "
                    f"scalar or a 1D or (2, n) array-like whose shape matches "
                    f"'{dep_axis}' (shape: {np.shape(dep)})") from None
            if np.any(err < -err):  # like err<0, but also works for timedelta.
                raise ValueError(
                    f"'{dep_axis}err' must not contain negative values")
            # This is like
            #     elow, ehigh = np.broadcast_to(...)
            #     return dep - elow * ~lolims, dep + ehigh * ~uplims
            # except that broadcast_to would strip units.
            low, high = dep + np.row_stack([-(1 - lolims), 1 - uplims]) * err

            barcols.append(lines_func(
                *apply_mask([indep, low, high], everymask), **eb_lines_style))
            # Normal errorbars for points without upper/lower limits.
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                indep_masked, lo_masked, hi_masked = apply_mask(
                    [indep, low, high], nolims & everymask)
                for lh_masked in [lo_masked, hi_masked]:
                    # Since this has to work for x and y as dependent data, we
                    # first set both x and y to the independent variable and
                    # overwrite the respective dependent data in a second step.
                    line = mlines.Line2D(indep_masked, indep_masked,
                                         marker=marker, **eb_cap_style)
                    line.set(**{f"{dep_axis}data": lh_masked})
                    caplines.append(line)
            for idx, (lims, hl) in enumerate([(lolims, high), (uplims, low)]):
                if not lims.any():
                    continue
                hlmarker = (
                    himarker
                    if getattr(self, f"{dep_axis}axis").get_inverted() ^ idx
                    else lomarker)
                x_masked, y_masked, hl_masked = apply_mask(
                    [x, y, hl], lims & everymask)
                # As above, we set the dependent data in a second step.
                line = mlines.Line2D(x_masked, y_masked,
                                     marker=hlmarker, **eb_cap_style)
                line.set(**{f"{dep_axis}data": hl_masked})
                caplines.append(line)
                if capsize > 0:
                    caplines.append(mlines.Line2D(
                        x_masked, y_masked, marker=marker, **eb_cap_style))

        for l in caplines:
            self.add_line(l)

        self._request_autoscale_view()
        errorbar_container = ErrorbarContainer(
            (data_line, tuple(caplines), tuple(barcols)),
            has_xerr=(xerr is not None), has_yerr=(yerr is not None),
            label=label)
        self.containers.append(errorbar_container)

        return errorbar_container  # (l0, caplines, barcols)

    @_preprocess_data()
    def boxplot(self, x, notch=None, sym=None, vert=None, whis=None,
                positions=None, widths=None, patch_artist=None,
                bootstrap=None, usermedians=None, conf_intervals=None,
                meanline=None, showmeans=None, showcaps=None,
                showbox=None, showfliers=None, boxprops=None,
                labels=None, flierprops=None, medianprops=None,
                meanprops=None, capprops=None, whiskerprops=None,
                manage_ticks=True, autorange=False, zorder=None,
                capwidths=None):
        # ... other code
```
### 5 - examples/lines_bars_and_markers/broken_barh.py:

Start line: 1, End line: 27

```python
"""
===========
Broken Barh
===========

Make a "broken" horizontal bar plot, i.e., one with gaps
"""
import matplotlib.pyplot as plt

# Horizontal bar plot with gaps
fig, ax = plt.subplots()
ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
               facecolors=('tab:orange', 'tab:green', 'tab:red'))
ax.set_ylim(5, 35)
ax.set_xlim(0, 200)
ax.set_xlabel('seconds since start')
ax.set_yticks([15, 25], labels=['Bill', 'Jim'])     # Modify y-axis tick labels
ax.grid(True)                                       # Make grid lines visible
ax.annotate('race interrupted', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')

plt.show()
```
### 6 - plot_types/stats/errorbar_plot.py:

Start line: 1, End line: 28

```python
"""
==========================
errorbar(x, y, yerr, xerr)
==========================

See `~matplotlib.axes.Axes.errorbar`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data:
np.random.seed(1)
x = [2, 4, 6]
y = [3.6, 5, 4.2]
yerr = [0.9, 1.2, 0.5]

# plot:
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
```
### 7 - examples/lines_bars_and_markers/errorbar_limits_simple.py:

Start line: 1, End line: 64

```python
"""
========================
Errorbar limit selection
========================

Illustration of selectively drawing lower and/or upper limit symbols on
errorbars using the parameters ``uplims``, ``lolims`` of `~.pyplot.errorbar`.

Alternatively, you can use 2xN values to draw errorbars in only one direction.
"""

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)

plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')

plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
             label='uplims=True, lolims=True')

upperlimits = [True, False] * 5
lowerlimits = [False, True] * 5
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
             label='subsets of uplims and lolims')

plt.legend(loc='lower right')


##############################################################################
# Similarly ``xuplims`` and ``xlolims`` can be used on the horizontal ``xerr``
# errorbars.

fig = plt.figure()
x = np.arange(10) / 10
y = (x + 0.1)**2

plt.errorbar(x, y, xerr=0.1, xlolims=True, label='xlolims=True')
y = (x + 0.1)**3

plt.errorbar(x + 0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits,
             label='subsets of xuplims and xlolims')

y = (x + 0.1)**4
plt.errorbar(x + 1.2, y, xerr=0.1, xuplims=True, label='xuplims=True')

plt.legend()
plt.show()

##############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```
### 8 - lib/matplotlib/pyplot.py:

Start line: 2396, End line: 2417

```python
@_copy_docstring_and_deprecators(Axes.barbs)
def barbs(*args, data=None, **kwargs):
    return gca().barbs(
        *args, **({"data": data} if data is not None else {}),
        **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.barh)
def barh(y, width, height=0.8, left=None, *, align='center', **kwargs):
    return gca().barh(
        y, width, height=height, left=left, align=align, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.bar_label)
def bar_label(
        container, labels=None, *, fmt='%g', label_type='edge',
        padding=0, **kwargs):
    return gca().bar_label(
        container, labels=labels, fmt=fmt, label_type=label_type,
        padding=padding, **kwargs)
```
### 9 - examples/statistics/errorbar_features.py:

Start line: 1, End line: 57

```python
"""
=======================================
Different ways of specifying error bars
=======================================

Errors can be specified as a constant value (as shown in
:doc:`/gallery/statistics/errorbar`). However, this example demonstrates
how they vary by specifying arrays of error values.

If the raw ``x`` and ``y`` data have length N, there are two options:

Array of shape (N,):
    Error varies for each point, but the error values are
    symmetric (i.e. the lower and upper values are equal).

Array of shape (2, N):
    Error varies for each point, and the lower and upper limits
    (in that order) are different (asymmetric case)

In addition, this example demonstrates how to use log
scale with error bars.
"""

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

# example error bar values that vary with x-position
error = 0.1 + 0.2 * x

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')

# error bar values w/ different -/+ errors that
# also vary with the x-position
lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]

ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')
ax1.set_yscale('log')
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```
### 10 - examples/statistics/errorbar.py:

Start line: 1, End line: 30

```python
"""
=================
Errorbar function
=================

This exhibits the most basic use of the error bar method.
In this case, constant values are provided for the error
in both the x- and y-directions.
"""

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

fig, ax = plt.subplots()
ax.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```
### 14 - lib/matplotlib/axes/_axes.py:

Start line: 2894, End line: 3482

```python
@_docstring.interpd
class Axes(_AxesBase):

    @_preprocess_data()
    def stem(self, *args, linefmt=None, markerfmt=None, basefmt=None, bottom=0,
             label=None, use_line_collection=True, orientation='vertical'):
        if not 1 <= len(args) <= 5:
            raise TypeError('stem expected between 1 and 5 positional '
                            'arguments, got {}'.format(args))
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)

        if len(args) == 1:
            heads, = args
            locs = np.arange(len(heads))
            args = ()
        else:
            locs, heads, *args = args
        if args:
            _api.warn_deprecated(
                "3.5",
                message="Passing the linefmt parameter positionally is "
                        "deprecated since Matplotlib %(since)s; the "
                        "parameter will become keyword-only %(removal)s.")

        if orientation == 'vertical':
            locs, heads = self._process_unit_info([("x", locs), ("y", heads)])
        else:
            heads, locs = self._process_unit_info([("x", heads), ("y", locs)])

        # defaults for formats
        if linefmt is None:
            linefmt = args[0] if len(args) > 0 else "C0-"
        linestyle, linemarker, linecolor = _process_plot_format(linefmt)

        if markerfmt is None:
            markerfmt = args[1] if len(args) > 1 else "C0o"
        markerstyle, markermarker, markercolor = \
            _process_plot_format(markerfmt)

        if basefmt is None:
            basefmt = (args[2] if len(args) > 2 else
                       "C2-" if rcParams["_internal.classic_mode"] else "C3-")
        basestyle, basemarker, basecolor = _process_plot_format(basefmt)

        # New behaviour in 3.1 is to use a LineCollection for the stemlines
        if use_line_collection:
            if linestyle is None:
                linestyle = rcParams['lines.linestyle']
            xlines = self.vlines if orientation == "vertical" else self.hlines
            stemlines = xlines(
                locs, bottom, heads,
                colors=linecolor, linestyles=linestyle, label="_nolegend_")
        # Old behaviour is to plot each of the lines individually
        else:
            stemlines = []
            for loc, head in zip(locs, heads):
                if orientation == 'horizontal':
                    xs = [bottom, head]
                    ys = [loc, loc]
                else:
                    xs = [loc, loc]
                    ys = [bottom, head]
                l, = self.plot(xs, ys,
                               color=linecolor, linestyle=linestyle,
                               marker=linemarker, label="_nolegend_")
                stemlines.append(l)

        if orientation == 'horizontal':
            marker_x = heads
            marker_y = locs
            baseline_x = [bottom, bottom]
            baseline_y = [np.min(locs), np.max(locs)]
        else:
            marker_x = locs
            marker_y = heads
            baseline_x = [np.min(locs), np.max(locs)]
            baseline_y = [bottom, bottom]

        markerline, = self.plot(marker_x, marker_y,
                                color=markercolor, linestyle=markerstyle,
                                marker=markermarker, label="_nolegend_")

        baseline, = self.plot(baseline_x, baseline_y,
                              color=basecolor, linestyle=basestyle,
                              marker=basemarker, label="_nolegend_")

        stem_container = StemContainer((markerline, stemlines, baseline),
                                       label=label)
        self.add_container(stem_container)
        return stem_container

    @_preprocess_data(replace_names=["x", "explode", "labels", "colors"])
    def pie(self, x, explode=None, labels=None, colors=None,
            autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1,
            startangle=0, radius=1, counterclock=True,
            wedgeprops=None, textprops=None, center=(0, 0),
            frame=False, rotatelabels=False, *, normalize=True):
        """
        Plot a pie chart.

        Make a pie chart of array *x*.  The fractional area of each wedge is
        given by ``x/sum(x)``.

        The wedges are plotted counterclockwise, by default starting from the
        x-axis.

        Parameters
        ----------
        x : 1D array-like
            The wedge sizes.

        explode : array-like, default: None
            If not *None*, is a ``len(x)`` array which specifies the fraction
            of the radius with which to offset each wedge.

        labels : list, default: None
            A sequence of strings providing the labels for each wedge

        colors : array-like, default: None
            A sequence of colors through which the pie chart will cycle.  If
            *None*, will use the colors in the currently active cycle.

        autopct : None or str or callable, default: None
            If not *None*, is a string or function used to label the wedges
            with their numeric value.  The label will be placed inside the
            wedge.  If it is a format string, the label will be ``fmt % pct``.
            If it is a function, it will be called.

        pctdistance : float, default: 0.6
            The ratio between the center of each pie slice and the start of
            the text generated by *autopct*.  Ignored if *autopct* is *None*.

        shadow : bool, default: False
            Draw a shadow beneath the pie.

        normalize : bool, default: True
            When *True*, always make a full pie by normalizing x so that
            ``sum(x) == 1``. *False* makes a partial pie if ``sum(x) <= 1``
            and raises a `ValueError` for ``sum(x) > 1``.

        labeldistance : float or None, default: 1.1
            The radial distance at which the pie labels are drawn.
            If set to ``None``, label are not drawn, but are stored for use in
            ``legend()``

        startangle : float, default: 0 degrees
            The angle by which the start of the pie is rotated,
            counterclockwise from the x-axis.

        radius : float, default: 1
            The radius of the pie.

        counterclock : bool, default: True
            Specify fractions direction, clockwise or counterclockwise.

        wedgeprops : dict, default: None
            Dict of arguments passed to the wedge objects making the pie.
            For example, you can pass in ``wedgeprops = {'linewidth': 3}``
            to set the width of the wedge border lines equal to 3.
            For more details, look at the doc/arguments of the wedge object.
            By default ``clip_on=False``.

        textprops : dict, default: None
            Dict of arguments to pass to the text objects.

        center : (float, float), default: (0, 0)
            The coordinates of the center of the chart.

        frame : bool, default: False
            Plot Axes frame with the chart if true.

        rotatelabels : bool, default: False
            Rotate each label to the angle of the corresponding slice if true.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        Returns
        -------
        patches : list
            A sequence of `matplotlib.patches.Wedge` instances

        texts : list
            A list of the label `.Text` instances.

        autotexts : list
            A list of `.Text` instances for the numeric labels. This will only
            be returned if the parameter *autopct* is not *None*.

        Notes
        -----
        The pie chart will probably look best if the figure and Axes are
        square, or the Axes aspect is equal.
        This method sets the aspect ratio of the axis to "equal".
        The Axes aspect ratio can be controlled with `.Axes.set_aspect`.
        """
        self.set_aspect('equal')
        # The use of float32 is "historical", but can't be changed without
        # regenerating the test baselines.
        x = np.asarray(x, np.float32)
        if x.ndim > 1:
            raise ValueError("x must be 1D")

        if np.any(x < 0):
            raise ValueError("Wedge sizes 'x' must be non negative values")

        sx = x.sum()

        if normalize:
            x = x / sx
        elif sx > 1:
            raise ValueError('Cannot plot an unnormalized pie with sum(x) > 1')
        if labels is None:
            labels = [''] * len(x)
        if explode is None:
            explode = [0] * len(x)
        if len(x) != len(labels):
            raise ValueError("'label' must be of length 'x'")
        if len(x) != len(explode):
            raise ValueError("'explode' must be of length 'x'")
        if colors is None:
            get_next_color = self._get_patches_for_fill.get_next_color
        else:
            color_cycle = itertools.cycle(colors)

            def get_next_color():
                return next(color_cycle)

        _api.check_isinstance(Number, radius=radius, startangle=startangle)
        if radius <= 0:
            raise ValueError(f'radius must be a positive number, not {radius}')

        # Starting theta1 is the start fraction of the circle
        theta1 = startangle / 360

        if wedgeprops is None:
            wedgeprops = {}
        if textprops is None:
            textprops = {}

        texts = []
        slices = []
        autotexts = []

        for frac, label, expl in zip(x, labels, explode):
            x, y = center
            theta2 = (theta1 + frac) if counterclock else (theta1 - frac)
            thetam = 2 * np.pi * 0.5 * (theta1 + theta2)
            x += expl * math.cos(thetam)
            y += expl * math.sin(thetam)

            w = mpatches.Wedge((x, y), radius, 360. * min(theta1, theta2),
                               360. * max(theta1, theta2),
                               facecolor=get_next_color(),
                               clip_on=False,
                               label=label)
            w.set(**wedgeprops)
            slices.append(w)
            self.add_patch(w)

            if shadow:
                # Make sure to add a shadow after the call to add_patch so the
                # figure and transform props will be set.
                shad = mpatches.Shadow(w, -0.02, -0.02, label='_nolegend_')
                self.add_patch(shad)

            if labeldistance is not None:
                xt = x + labeldistance * radius * math.cos(thetam)
                yt = y + labeldistance * radius * math.sin(thetam)
                label_alignment_h = 'left' if xt > 0 else 'right'
                label_alignment_v = 'center'
                label_rotation = 'horizontal'
                if rotatelabels:
                    label_alignment_v = 'bottom' if yt > 0 else 'top'
                    label_rotation = (np.rad2deg(thetam)
                                      + (0 if xt > 0 else 180))
                t = self.text(xt, yt, label,
                              clip_on=False,
                              horizontalalignment=label_alignment_h,
                              verticalalignment=label_alignment_v,
                              rotation=label_rotation,
                              size=rcParams['xtick.labelsize'])
                t.set(**textprops)
                texts.append(t)

            if autopct is not None:
                xt = x + pctdistance * radius * math.cos(thetam)
                yt = y + pctdistance * radius * math.sin(thetam)
                if isinstance(autopct, str):
                    s = autopct % (100. * frac)
                elif callable(autopct):
                    s = autopct(100. * frac)
                else:
                    raise TypeError(
                        'autopct must be callable or a format string')
                t = self.text(xt, yt, s,
                              clip_on=False,
                              horizontalalignment='center',
                              verticalalignment='center')
                t.set(**textprops)
                autotexts.append(t)

            theta1 = theta2

        if frame:
            self._request_autoscale_view()
        else:
            self.set(frame_on=False, xticks=[], yticks=[],
                     xlim=(-1.25 + center[0], 1.25 + center[0]),
                     ylim=(-1.25 + center[1], 1.25 + center[1]))

        if autopct is None:
            return slices, texts
        else:
            return slices, texts, autotexts

    @_preprocess_data(replace_names=["x", "y", "xerr", "yerr"],
                      label_namer="y")
    @_docstring.dedent_interpd
    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='', ecolor=None, elinewidth=None, capsize=None,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, errorevery=1, capthick=None,
                 **kwargs):
        """
        Plot y versus x as lines and/or markers with attached errorbars.

        *x*, *y* define the data locations, *xerr*, *yerr* define the errorbar
        sizes. By default, this draws the data markers/lines as well the
        errorbars. Use fmt='none' to draw errorbars without any data markers.

        Parameters
        ----------
        x, y : float or array-like
            The data positions.

        xerr, yerr : float or array-like, shape(N,) or shape(2, N), optional
            The errorbar sizes:

            - scalar: Symmetric +/- values for all data points.
            - shape(N,): Symmetric +/-values for each data point.
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar.

            All values must be >= 0.

            See :doc:`/gallery/statistics/errorbar_features`
            for an example on the usage of ``xerr`` and ``yerr``.

        fmt : str, default: ''
            The format for the data points / data lines. See `.plot` for
            details.

            Use 'none' (case insensitive) to plot errorbars without any data
            markers.

        ecolor : color, default: None
            The color of the errorbar lines.  If None, use the color of the
            line connecting the markers.

        elinewidth : float, default: None
            The linewidth of the errorbar lines. If None, the linewidth of
            the current style is used.

        capsize : float, default: :rc:`errorbar.capsize`
            The length of the error bar caps in points.

        capthick : float, default: None
            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
            This setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*. This may change in future
            releases.

        barsabove : bool, default: False
            If True, will plot the errorbars above the plot
            symbols. Default is below.

        lolims, uplims, xlolims, xuplims : bool, default: False
            These arguments can be used to indicate that a value gives only
            upper/lower limits.  In that case a caret symbol is used to
            indicate this. *lims*-arguments may be scalars, or array-likes of
            the same length as *xerr* and *yerr*.  To use limits with inverted
            axes, `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before
            :meth:`errorbar`.  Note the tricky parameter names: setting e.g.
            *lolims* to True means that the y-value is a *lower* limit of the
            True value, so, only an *upward*-pointing arrow will be drawn!

        errorevery : int or (int, int), default: 1
            draws error bars on a subset of the data. *errorevery* =N draws
            error bars on the points (x[::N], y[::N]).
            *errorevery* =(start, N) draws error bars on the points
            (x[start::N], y[start::N]). e.g. errorevery=(6, 3)
            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
            Used to avoid overlapping error bars when two series share x-axis
            values.

        Returns
        -------
        `.ErrorbarContainer`
            The container contains:

            - plotline: `.Line2D` instance of x, y plot markers and/or line.
            - caplines: A tuple of `.Line2D` instances of the error bar caps.
            - barlinecols: A tuple of `.LineCollection` with the horizontal and
              vertical error ranges.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments are passed on to the `~.Axes.plot` call
            drawing the markers. For example, this code makes big red squares
            with thick green edges::

                x, y, yerr = rand(3, 10)
                errorbar(x, y, yerr, marker='s', mfc='red',
                         mec='green', ms=20, mew=4)

            where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
            property names, *markerfacecolor*, *markeredgecolor*, *markersize*
            and *markeredgewidth*.

            Valid kwargs for the marker properties are `.Line2D` properties:

            %(Line2D:kwdoc)s
        """
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        # Drop anything that comes in as None to use the default instead.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)

        # Casting to object arrays preserves units.
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=object)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=object)

        def _upcast_err(err):
            """
            Safely handle tuple of containers that carry units.

            This function covers the case where the input to the xerr/yerr is a
            length 2 tuple of equal length ndarray-subclasses that carry the
            unit information in the container.

            If we have a tuple of nested numpy array (subclasses), we defer
            coercing the units to be consistent to the underlying unit
            library (and implicitly the broadcasting).

            Otherwise, fallback to casting to an object array.
            """

            if (
                    # make sure it is not a scalar
                    np.iterable(err) and
                    # and it is not empty
                    len(err) > 0 and
                    # and the first element is an array sub-class use
                    # safe_first_element because getitem is index-first not
                    # location first on pandas objects so err[0] almost always
                    # fails.
                    isinstance(cbook.safe_first_element(err), np.ndarray)
            ):
                # Get the type of the first element
                atype = type(cbook.safe_first_element(err))
                # Promote the outer container to match the inner container
                if atype is np.ndarray:
                    # Converts using np.asarray, because data cannot
                    # be directly passed to init of np.ndarray
                    return np.asarray(err, dtype=object)
                # If atype is not np.ndarray, directly pass data to init.
                # This works for types such as unyts and astropy units
                return atype(err)
            # Otherwise wrap it in an object array
            return np.asarray(err, dtype=object)

        if xerr is not None and not isinstance(xerr, np.ndarray):
            xerr = _upcast_err(xerr)
        if yerr is not None and not isinstance(yerr, np.ndarray):
            yerr = _upcast_err(yerr)
        x, y = np.atleast_1d(x, y)  # Make sure all the args are iterable.
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same size")

        if isinstance(errorevery, Integral):
            errorevery = (0, errorevery)
        if isinstance(errorevery, tuple):
            if (len(errorevery) == 2 and
                    isinstance(errorevery[0], Integral) and
                    isinstance(errorevery[1], Integral)):
                errorevery = slice(errorevery[0], None, errorevery[1])
            else:
                raise ValueError(
                    f'errorevery={errorevery!r} is a not a tuple of two '
                    f'integers')
        elif isinstance(errorevery, slice):
            pass
        elif not isinstance(errorevery, str) and np.iterable(errorevery):
            # fancy indexing
            try:
                x[errorevery]
            except (ValueError, IndexError) as err:
                raise ValueError(
                    f"errorevery={errorevery!r} is iterable but not a valid "
                    f"NumPy fancy index to match 'xerr'/'yerr'") from err
        else:
            raise ValueError(
                f"errorevery={errorevery!r} is not a recognized value")
        everymask = np.zeros(len(x), bool)
        everymask[errorevery] = True

        label = kwargs.pop("label", None)
        kwargs['label'] = '_nolegend_'

        # Create the main line and determine overall kwargs for child artists.
        # We avoid calling self.plot() directly, or self._get_lines(), because
        # that would call self._process_unit_info again, and do other indirect
        # data processing.
        (data_line, base_style), = self._get_lines._plot_args(
            (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)

        # Do this after creating `data_line` to avoid modifying `base_style`.
        if barsabove:
            data_line.set_zorder(kwargs['zorder'] - .1)
        else:
            data_line.set_zorder(kwargs['zorder'] + .1)

        # Add line to plot, or throw it away and use it to determine kwargs.
        if fmt.lower() != 'none':
            self.add_line(data_line)
        else:
            data_line = None
            # Remove alpha=0 color that _get_lines._plot_args returns for
            # 'none' format, and replace it with user-specified color, if
            # supplied.
            base_style.pop('color')
            if 'color' in kwargs:
                base_style['color'] = kwargs.pop('color')

        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']

        # Eject any line-specific information from format string, as it's not
        # needed for bars or caps.
        for key in ['marker', 'markersize', 'markerfacecolor',
                    'markeredgewidth', 'markeredgecolor', 'markevery',
                    'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle',
                    'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle',
                    'dashes']:
            base_style.pop(key, None)

        # Make the style dict for the line collections (the bars).
        eb_lines_style = {**base_style, 'color': ecolor}

        if elinewidth is not None:
            eb_lines_style['linewidth'] = elinewidth
        elif 'linewidth' in kwargs:
            eb_lines_style['linewidth'] = kwargs['linewidth']

        for key in ('transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_lines_style[key] = kwargs[key]

        # Make the style dict for caps (the "hats").
        eb_cap_style = {**base_style, 'linestyle': 'none'}
        if capsize is None:
            capsize = rcParams["errorbar.capsize"]
        if capsize > 0:
            eb_cap_style['markersize'] = 2. * capsize
        if capthick is not None:
            eb_cap_style['markeredgewidth'] = capthick

        # For backwards-compat, allow explicit setting of
        # 'markeredgewidth' to over-ride capthick.
        for key in ('markeredgewidth', 'transform', 'alpha',
                    'zorder', 'rasterized'):
            if key in kwargs:
                eb_cap_style[key] = kwargs[key]
        eb_cap_style['color'] = ecolor

        barcols = []
        caplines = []

        # Vectorized fancy-indexer.
        def apply_mask(arrays, mask): return [array[mask] for array in arrays]

        # dep: dependent dataset, indep: independent dataset
  # ... other code
```
