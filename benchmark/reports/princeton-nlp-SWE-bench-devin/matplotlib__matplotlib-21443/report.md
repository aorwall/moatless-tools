# matplotlib__matplotlib-21443

| **matplotlib/matplotlib** | `d448de31b7deaec8310caaf8bba787e097bf9211` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 14 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -1059,8 +1059,12 @@ def axes(arg=None, **kwargs):
         plt.axes((left, bottom, width, height), facecolor='w')
     """
     fig = gcf()
+    pos = kwargs.pop('position', None)
     if arg is None:
-        return fig.add_subplot(**kwargs)
+        if pos is None:
+            return fig.add_subplot(**kwargs)
+        else:
+            return fig.add_axes(pos, **kwargs)
     else:
         return fig.add_axes(arg, **kwargs)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/pyplot.py | 1062 | 1062 | - | 14 | -


## Problem Statement

```
[Bug]: axes(position = [...]) behavior
### Bug summary

when setting axes position with `ax = plt.axes(position = [...])` the position data is not being incorporated.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt

fig = plt.figure()

pos1 = [0.1, 0.1, 0.3, 0.8]
pos2 = [0.5, 0.1, 0.4, 0.6]

ax1 = plt.axes(position = pos1)
ax1.plot([0,1], [0, 1], color = 'r', linewidth = 3)

ax2 = plt.axes(position = pos2)
ax2.plot([1, 0], [0, 1], color = 'b', linestyle = '--')
\`\`\`


### Actual outcome

The two axes completely overlap
![test1](https://user-images.githubusercontent.com/11670408/138557633-5a375766-ac87-4fd0-9305-7c0ca7c5121c.png)


### Expected outcome

Would expect two separate axes (these were created by adding
`ax1.set_axes(pos1)` and `ax2.set_axes(pos2)`, which should not be necessary)
![test2](https://user-images.githubusercontent.com/11670408/138557661-690221c9-8cb1-4496-8316-72c5bcbe9764.png)



### Operating system

Windows

### Matplotlib Version

3.4.2

### Matplotlib Backend

Qt5Agg

### Python version

3.8.8

### Jupyter version

_No response_

### Other libraries

_No response_

### Installation

conda

### Conda channel

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/subplots_axes_and_figures/shared_axis_demo.py | 1 | 58| 490 | 490 | 490 | 
| 2 | 2 lib/matplotlib/axes/_base.py | 242 | 312| 721 | 1211 | 40201 | 
| 3 | 3 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 1996 | 42168 | 
| 4 | 4 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 2521 | 42693 | 
| 5 | 5 examples/axisartist/demo_parasite_axes.py | 1 | 61| 518 | 3039 | 43211 | 
| 6 | 6 examples/axisartist/demo_parasite_axes2.py | 1 | 61| 564 | 3603 | 43775 | 
| 7 | 7 examples/axes_grid1/demo_fixed_size_axes.py | 1 | 48| 326 | 3929 | 44101 | 
| 8 | 8 examples/axes_grid1/inset_locator_demo.py | 77 | 145| 718 | 4647 | 45617 | 
| 9 | 9 examples/subplots_axes_and_figures/axes_box_aspect.py | 111 | 156| 344 | 4991 | 46742 | 
| 10 | 10 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 701 | 5692 | 47852 | 
| 11 | 11 lib/matplotlib/axis.py | 349 | 404| 600 | 6292 | 68067 | 
| 12 | 11 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 441 | 6733 | 68067 | 
| 13 | 12 examples/axisartist/demo_floating_axes.py | 145 | 164| 186 | 6919 | 69461 | 
| 14 | 12 lib/matplotlib/axis.py | 2149 | 2187| 374 | 7293 | 69461 | 
| 15 | 13 examples/pyplots/align_ylabels.py | 40 | 86| 346 | 7639 | 70095 | 
| 16 | **14 lib/matplotlib/pyplot.py** | 1587 | 1600| 108 | 7747 | 96489 | 
| 17 | 15 examples/axisartist/simple_axis_pad.py | 56 | 106| 390 | 8137 | 97224 | 
| 18 | 15 lib/matplotlib/axes/_base.py | 486 | 4737| 535 | 8672 | 97224 | 
| 19 | 16 examples/axes_grid1/demo_axes_grid2.py | 1 | 99| 783 | 9455 | 98007 | 
| 20 | 17 examples/ticks/auto_ticks.py | 1 | 49| 354 | 9809 | 98361 | 
| 21 | 18 examples/subplots_axes_and_figures/axes_zoom_effect.py | 81 | 109| 262 | 10071 | 99376 | 
| 22 | 19 examples/spines/multiple_yaxis_with_spines.py | 1 | 56| 517 | 10588 | 99893 | 
| 23 | 19 examples/subplots_axes_and_figures/axes_zoom_effect.py | 43 | 78| 304 | 10892 | 99893 | 
| 24 | 19 examples/subplots_axes_and_figures/axes_zoom_effect.py | 112 | 126| 104 | 10996 | 99893 | 
| 25 | 20 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 86| 750 | 11746 | 100779 | 
| 26 | **20 lib/matplotlib/pyplot.py** | 1603 | 1616| 107 | 11853 | 100779 | 
| 27 | 21 lib/matplotlib/axes/_axes.py | 940 | 1631| 3911 | 15764 | 173584 | 
| 28 | 22 tutorials/intermediate/constrainedlayout_guide.py | 591 | 676| 788 | 16552 | 179687 | 
| 29 | 23 examples/axisartist/simple_axis_direction03.py | 1 | 35| 189 | 16741 | 179876 | 
| 30 | 24 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 116| 759 | 17500 | 180758 | 
| 31 | 25 lib/mpl_toolkits/axes_grid/parasite_axes.py | 1 | 13| 116 | 17616 | 180874 | 


### Hint

```
Tried updating to 3.4.3 and got the same plotting result.

\`\`\`
The following NEW packages will be INSTALLED:

  charls             pkgs/main/win-64::charls-2.2.0-h6c2663c_0
  giflib             pkgs/main/win-64::giflib-5.2.1-h62dcd97_0
  imagecodecs        pkgs/main/win-64::imagecodecs-2021.6.8-py38he57d016_1
  lcms2              pkgs/main/win-64::lcms2-2.12-h83e58a3_0
  lerc               pkgs/main/win-64::lerc-2.2.1-hd77b12b_0
  libaec             pkgs/main/win-64::libaec-1.0.4-h33f27b4_1
  libdeflate         pkgs/main/win-64::libdeflate-1.8-h2bbff1b_5
  libwebp            pkgs/main/win-64::libwebp-1.2.0-h2bbff1b_0
  libzopfli          pkgs/main/win-64::libzopfli-1.0.3-ha925a31_0
  zfp                pkgs/main/win-64::zfp-0.5.5-hd77b12b_6

The following packages will be UPDATED:

  certifi                          2021.5.30-py38haa95532_0 --> 2021.10.8-py38haa95532_0
  cryptography                         3.4.7-py38h71e12ea_0 --> 3.4.8-py38h71e12ea_0
  dask                                2021.8.1-pyhd3eb1b0_0 --> 2021.9.1-pyhd3eb1b0_0
  dask-core                           2021.8.1-pyhd3eb1b0_0 --> 2021.9.1-pyhd3eb1b0_0
  decorator                              5.0.9-pyhd3eb1b0_0 --> 5.1.0-pyhd3eb1b0_0
  distributed                       2021.8.1-py38haa95532_0 --> 2021.9.1-py38haa95532_0
  ipykernel                            6.2.0-py38haa95532_1 --> 6.4.1-py38haa95532_1
  ipywidgets                             7.6.3-pyhd3eb1b0_1 --> 7.6.5-pyhd3eb1b0_1
  jupyter_core                         4.7.1-py38haa95532_0 --> 4.8.1-py38haa95532_0
  jupyterlab_server                      2.8.1-pyhd3eb1b0_0 --> 2.8.2-pyhd3eb1b0_0
  libblas                           3.9.0-1_h8933c1f_netlib --> 3.9.0-12_win64_mkl
  libcblas                          3.9.0-5_hd5c7e75_netlib --> 3.9.0-12_win64_mkl
  liblapack                         3.9.0-5_hd5c7e75_netlib --> 3.9.0-12_win64_mkl
  llvmlite                            0.36.0-py38h34b8924_4 --> 0.37.0-py38h23ce68f_1
  matplotlib                           3.4.2-py38haa95532_0 --> 3.4.3-py38haa95532_0
  matplotlib-base                      3.4.2-py38h49ac443_0 --> 3.4.3-py38h49ac443_0
  mkl                  pkgs/main::mkl-2021.3.0-haa95532_524 --> conda-forge::mkl-2021.4.0-h0e2418a_729
  mkl_fft                              1.3.0-py38h277e83a_2 --> 1.3.1-py38h277e83a_0
  networkx                               2.6.2-pyhd3eb1b0_0 --> 2.6.3-pyhd3eb1b0_0
  nltk                                   3.6.2-pyhd3eb1b0_0 --> 3.6.5-pyhd3eb1b0_0
  numba              pkgs/main::numba-0.53.1-py38hf11a4ad_0 --> conda-forge::numba-0.54.1-py38h5858985_0
  openpyxl                               3.0.7-pyhd3eb1b0_0 --> 3.0.9-pyhd3eb1b0_0
  pandas                               1.3.2-py38h6214cd6_0 --> 1.3.3-py38h6214cd6_0
  patsy                                        0.5.1-py38_0 --> 0.5.2-py38haa95532_0
  pillow                               8.3.1-py38h4fa10fc_0 --> 8.4.0-py38hd45dc43_0
  prompt-toolkit                        3.0.17-pyhca03da5_0 --> 3.0.20-pyhd3eb1b0_0
  prompt_toolkit                          3.0.17-hd3eb1b0_0 --> 3.0.20-hd3eb1b0_0
  pycurl                            7.43.0.6-py38h7a1dbc1_0 --> 7.44.1-py38hcd4344a_1
  pytz                                  2021.1-pyhd3eb1b0_0 --> 2021.3-pyhd3eb1b0_0
  qtconsole                              5.1.0-pyhd3eb1b0_0 --> 5.1.1-pyhd3eb1b0_0
  tbb                                     2020.3-h74a9793_0 --> 2021.4.0-h59b6b97_0
  tifffile           pkgs/main/win-64::tifffile-2020.10.1-~ --> pkgs/main/noarch::tifffile-2021.7.2-pyhd3eb1b0_2
  tk                                      8.6.10-he774522_0 --> 8.6.11-h2bbff1b_0
  traitlets                              5.0.5-pyhd3eb1b0_0 --> 5.1.0-pyhd3eb1b0_0
  urllib3                               1.26.6-pyhd3eb1b0_1 --> 1.26.7-pyhd3eb1b0_0
  wincertstore                                   0.2-py38_0 --> 0.2-py38haa95532_2
  zipp                                   3.5.0-pyhd3eb1b0_0 --> 3.6.0-pyhd3eb1b0_0

The following packages will be DOWNGRADED:

  fiona                         1.8.13.post1-py38hd760492_0 --> 1.8.13.post1-py38h758c064_0
  shapely                              1.7.1-py38h210f175_0 --> 1.7.1-py38h06580b3_0
\`\`\`
The [docstring for `plt.axes`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axes.html) reads:

\`\`\`
Add an axes to the current figure and make it the current axes.

Call signatures::

    plt.axes()
    plt.axes(rect, projection=None, polar=False, **kwargs)
    plt.axes(ax)

Parameters
----------
arg : None or 4-tuple
    The exact behavior of this function depends on the type:

    - *None*: A new full window axes is added using
      ``subplot(**kwargs)``.
    - 4-tuple of floats *rect* = ``[left, bottom, width, height]``.
      A new axes is added with dimensions *rect* in normalized
      (0, 1) units using `~.Figure.add_axes` on the current figure.
...
\`\`\`

The `mpl.axes.Axes` constructor accepts a `position` parameter and so it shows it up in list of additional keyword arguments, but it's overridden by the handling of  `arg=None` in this interface function.

All *you* need to do is change your code to `plt.axes(pos)`, etc.

`plt.axes()` should probably at least warn that it's ignoring `position=` in this case.
Thank you. Is this a change in behavior? Writing the code as I had it above in Google Colab gives the behavior I had expected.
It's definitely a change. Whether it was on purpose or not I'm not quite sure. 
The default version on Colab is older (3.2.2) and does indeed work differently, but the documentation for the parameters is the same.
The changed in 261f7062860d   https://github.com/matplotlib/matplotlib/pull/18564  While I agree that one need not pass `position=rect`, I guess we shouldn't have broken this, and we should definitely not document this as something that is possible.  
```

## Patch

```diff
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -1059,8 +1059,12 @@ def axes(arg=None, **kwargs):
         plt.axes((left, bottom, width, height), facecolor='w')
     """
     fig = gcf()
+    pos = kwargs.pop('position', None)
     if arg is None:
-        return fig.add_subplot(**kwargs)
+        if pos is None:
+            return fig.add_subplot(**kwargs)
+        else:
+            return fig.add_axes(pos, **kwargs)
     else:
         return fig.add_axes(arg, **kwargs)
 

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_pyplot.py b/lib/matplotlib/tests/test_pyplot.py
--- a/lib/matplotlib/tests/test_pyplot.py
+++ b/lib/matplotlib/tests/test_pyplot.py
@@ -1,4 +1,5 @@
 import difflib
+import numpy as np
 import subprocess
 import sys
 from pathlib import Path
@@ -320,3 +321,17 @@ def test_polar_second_call():
     ln2, = plt.polar(1.57, .5, 'bo')
     assert isinstance(ln2, mpl.lines.Line2D)
     assert ln1.axes is ln2.axes
+
+
+def test_fallback_position():
+    # check that position kwarg works if rect not supplied
+    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
+    axtest = plt.axes(position=[0.2, 0.2, 0.5, 0.5])
+    np.testing.assert_allclose(axtest.bbox.get_points(),
+                               axref.bbox.get_points())
+
+    # check that position kwarg ignored if rect is supplied
+    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
+    axtest = plt.axes([0.2, 0.2, 0.5, 0.5], position=[0.1, 0.1, 0.8, 0.8])
+    np.testing.assert_allclose(axtest.bbox.get_points(),
+                               axref.bbox.get_points())

```


## Code snippets

### 1 - examples/subplots_axes_and_figures/shared_axis_demo.py:

Start line: 1, End line: 58

```python
"""
===========
Shared Axis
===========

You can share the x or y axis limits for one axis with another by
passing an axes instance as a *sharex* or *sharey* keyword argument.

Changing the axis limits on one axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the axes will follow each other on their shared axes.  Ditto for
changes in the axis scaling (e.g., log vs. linear).  However, it is
possible to have differences in tick labeling, e.g., you can selectively
turn off the tick labels on one axes.

The example below shows how to customize the tick labels on the
various axes.  Shared axes share the tick locator, tick formatter,
view limits, and transformation (e.g., log, linear).  But the ticklabels
themselves do not share properties.  This is a feature and not a bug,
because you may want to make the tick labels smaller on the upper
axes, e.g., in the example below.

If you want to turn off the ticklabels for a given axes (e.g., on
subplot(211) or subplot(212), you cannot do the standard trick::

   setp(ax2, xticklabels=[])

because this changes the tick Formatter, which is shared among all
axes.  But you can alter the visibility of the labels, which is a
property::

  setp(ax2.get_xticklabels(), visible=False)

"""
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 5.0, 0.01)
s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = np.sin(4 * np.pi * t)

ax1 = plt.subplot(311)
plt.plot(t, s1)
plt.tick_params('x', labelsize=6)

# share x only
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(t, s2)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)

# share x and y
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot(t, s3)
plt.xlim(0.01, 5.0)
plt.show()
```
### 2 - lib/matplotlib/axes/_base.py:

Start line: 242, End line: 312

```python
class _process_plot_var_args:

    def __call__(self, *args, data=None, **kwargs):
        self.axes._process_unit_info(kwargs=kwargs)

        for pos_only in "xy":
            if pos_only in kwargs:
                raise TypeError("{} got an unexpected keyword argument {!r}"
                                .format(self.command, pos_only))

        if not args:
            return

        if data is None:  # Process dict views
            args = [cbook.sanitize_sequence(a) for a in args]
        else:  # Process the 'data' kwarg.
            replaced = [mpl._replacer(data, arg) for arg in args]
            if len(args) == 1:
                label_namer_idx = 0
            elif len(args) == 2:  # Can be x, y or y, c.
                # Figure out what the second argument is.
                # 1) If the second argument cannot be a format shorthand, the
                #    second argument is the label_namer.
                # 2) Otherwise (it could have been a format shorthand),
                #    a) if we did perform a substitution, emit a warning, and
                #       use it as label_namer.
                #    b) otherwise, it is indeed a format shorthand; use the
                #       first argument as label_namer.
                try:
                    _process_plot_format(args[1])
                except ValueError:  # case 1)
                    label_namer_idx = 1
                else:
                    if replaced[1] is not args[1]:  # case 2a)
                        _api.warn_external(
                            f"Second argument {args[1]!r} is ambiguous: could "
                            f"be a format string but is in 'data'; using as "
                            f"data.  If it was intended as data, set the "
                            f"format string to an empty string to suppress "
                            f"this warning.  If it was intended as a format "
                            f"string, explicitly pass the x-values as well.  "
                            f"Alternatively, rename the entry in 'data'.",
                            RuntimeWarning)
                        label_namer_idx = 1
                    else:  # case 2b)
                        label_namer_idx = 0
            elif len(args) == 3:
                label_namer_idx = 1
            else:
                raise ValueError(
                    "Using arbitrary long args with data is not supported due "
                    "to ambiguity of arguments; use multiple plotting calls "
                    "instead")
            if kwargs.get("label") is None:
                kwargs["label"] = mpl._label_from_arg(
                    replaced[label_namer_idx], args[label_namer_idx])
            args = replaced

        if len(args) >= 4 and not cbook.is_scalar_or_string(
                kwargs.get("label")):
            raise ValueError("plot() with multiple groups of data (i.e., "
                             "pairs of x and y) does not support multiple "
                             "labels")

        # Repeatedly grab (x, y) or (x, y, format) from the front of args and
        # massage them into arguments to plot() or fill().

        while args:
            this, args = args[:2], args[2:]
            if args and isinstance(args[0], str):
                this += args[0],
                args = args[1:]
            yield from self._plot_args(this, kwargs)
```
### 3 - examples/subplots_axes_and_figures/subplots_demo.py:

Start line: 87, End line: 170

```python
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

###############################################################################
# You can use tuple-unpacking also in 2D to assign all subplots to dedicated
# variables:

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x, -y, 'tab:green')
ax4.plot(x, -y**2, 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()

###############################################################################
# Sharing axes
# """"""""""""
#
# By default, each Axes is scaled individually. Thus, if the ranges are
# different the tick values of the subplots do not align.

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Axes values are scaled individually by default')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

###############################################################################
# You can use *sharex* or *sharey* to align the horizontal or vertical axis.

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

###############################################################################
# Setting *sharex* or *sharey* to ``True`` enables global sharing across the
# whole grid, i.e. also the y-axes of vertically stacked subplots have the
# same scale when using ``sharey=True``.

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

###############################################################################
# For subplots that are sharing axes one set of tick labels is enough. Tick
# labels of inner Axes are automatically removed by *sharex* and *sharey*.
# Still there remains an unused empty space between the subplots.
#
# To precisely control the positioning of the subplots, one can explicitly
# create a `.GridSpec` with `.Figure.add_gridspec`, and then call its
# `~.GridSpecBase.subplots` method.  For example, we can reduce the height
# between vertical subplots using ``add_gridspec(hspace=0)``.
#
# `.label_outer` is a handy method to remove labels and ticks from subplots
# that are not at the edge of the grid.

fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

###############################################################################
# Apart from ``True`` and ``False``, both *sharex* and *sharey* accept the
# values 'row' and 'col' to share the values only per row or column.

fig = plt.figure()
```
### 4 - examples/subplots_axes_and_figures/broken_axis.py:

Start line: 1, End line: 55

```python
"""
===========
Broken Axis
===========

Broken axis example, where the y-axis will have a portion cut out.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

pts = np.random.rand(30)*.2
# Now let's make two outlier points which are far away from everything.
pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax1) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# plot the same data on both axes
ax1.plot(pts)
ax2.plot(pts)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(.78, 1.)  # outliers only
ax2.set_ylim(0, .22)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the axes size or scale
# Finally, we need to disable clipping.

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


plt.show()
```
### 5 - examples/axisartist/demo_parasite_axes.py:

Start line: 1, End line: 61

```python
"""
==================
Parasite Axes demo
==================

Create a parasite axes. Such axes would share the x scale with a host axes,
but show a different scale in y direction.

This approach uses `mpl_toolkits.axes_grid1.parasite_axes.HostAxes` and
`mpl_toolkits.axes_grid1.parasite_axes.ParasiteAxes`.

An alternative approach using standard Matplotlib subplots is shown in the
:doc:`/gallery/spines/multiple_yaxis_with_spines` example.

An alternative approach using :mod:`mpl_toolkits.axes_grid1`
and :mod:`mpl_toolkits.axisartist` is found in the
:doc:`/gallery/axisartist/demo_parasite_axes2` example.
"""

from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt


fig = plt.figure()

host = fig.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)
par1 = ParasiteAxes(host, sharex=host)
par2 = ParasiteAxes(host, sharex=host)
host.parasites.append(par1)
host.parasites.append(par2)

host.axis["right"].set_visible(False)

par1.axis["right"].set_visible(True)
par1.axis["right"].major_ticklabels.set_visible(True)
par1.axis["right"].label.set_visible(True)

par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")
p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

host.set_xlim(0, 2)
host.set_ylim(0, 2)
par1.set_ylim(0, 4)
par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right2"].label.set_color(p3.get_color())

plt.show()
```
### 6 - examples/axisartist/demo_parasite_axes2.py:

Start line: 1, End line: 61

```python
"""
==================
Parasite axis demo
==================

This example demonstrates the use of parasite axis to plot multiple datasets
onto one single plot.

Notice how in this example, *par1* and *par2* are both obtained by calling
``twinx()``, which ties their x-limits with the host's x-axis. From there, each
of those two axis behave separately from each other: different datasets can be
plotted, and the y-limits are adjusted separately.

Note that this approach uses the `mpl_toolkits.axes_grid1.parasite_axes`'
`~mpl_toolkits.axes_grid1.parasite_axes.host_subplot` and
`mpl_toolkits.axisartist.axislines.Axes`. An alternative approach using the
`~mpl_toolkits.axes_grid1.parasite_axes`'s
`~.mpl_toolkits.axes_grid1.parasite_axes.HostAxes` and
`~.mpl_toolkits.axes_grid1.parasite_axes.ParasiteAxes` is the
:doc:`/gallery/axisartist/demo_parasite_axes` example.
An alternative approach using the usual Matplotlib subplots is shown in
the :doc:`/gallery/spines/multiple_yaxis_with_spines` example.
"""

from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt

host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")
p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

host.set_xlim(0, 2)
host.set_ylim(0, 2)
par1.set_ylim(0, 4)
par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.show()
```
### 7 - examples/axes_grid1/demo_fixed_size_axes.py:

Start line: 1, End line: 48

```python
"""
===============================
Axes with a fixed physical size
===============================
"""

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import Divider, Size

###############################################################################


fig = plt.figure(figsize=(6, 6))

# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(1.0), Size.Fixed(4.5)]
v = [Size.Fixed(0.7), Size.Fixed(5.)]

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# The width and height of the rectangle are ignored.

ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

ax.plot([1, 2, 3])

###############################################################################


fig = plt.figure(figsize=(6, 6))

# The first & third items are for padding and the second items are for the
# axes. Sizes are in inches.
h = [Size.Fixed(1.0), Size.Scaled(1.), Size.Fixed(.2)]
v = [Size.Fixed(0.7), Size.Scaled(1.), Size.Fixed(.5)]

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# The width and height of the rectangle are ignored.

ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

ax.plot([1, 2, 3])

plt.show()
```
### 8 - examples/axes_grid1/inset_locator_demo.py:

Start line: 77, End line: 145

```python
ax.set(xlim=(0, 10), ylim=(0, 10))


# Note how the two following insets are created at the same positions, one by
# use of the default parent axes' bbox and the other via a bbox in axes
# coordinates and the respective transform.
ax2 = fig.add_subplot(222)
axins2 = inset_axes(ax2, width="30%", height="50%")

ax3 = fig.add_subplot(224)
axins3 = inset_axes(ax3, width="100%", height="100%",
                    bbox_to_anchor=(.7, .5, .3, .5),
                    bbox_transform=ax3.transAxes)

# For visualization purposes we mark the bounding box by a rectangle
ax2.add_patch(plt.Rectangle((0, 0), 1, 1, ls="--", lw=2, ec="c", fc="none"))
ax3.add_patch(plt.Rectangle((.7, .5), .3, .5, ls="--", lw=2,
                            ec="c", fc="none"))

# Turn ticklabels off
for axi in [axins2, axins3, ax2, ax3]:
    axi.tick_params(labelleft=False, labelbottom=False)

plt.show()


###############################################################################
# In the above the axes transform together with 4-tuple bounding boxes has been
# used as it mostly is useful to specify an inset relative to the axes it is
# an inset to. However other use cases are equally possible. The following
# example examines some of those.
#

fig = plt.figure(figsize=[5.5, 2.8])
ax = fig.add_subplot(131)

# Create an inset outside the axes
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(1.05, .6, .5, .4),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
axins.tick_params(left=False, right=True, labelleft=False, labelright=True)

# Create an inset with a 2-tuple bounding box. Note that this creates a
# bbox without extent. This hence only makes sense when specifying
# width and height in absolute units (inches).
axins2 = inset_axes(ax, width=0.5, height=0.4,
                    bbox_to_anchor=(0.33, 0.25),
                    bbox_transform=ax.transAxes, loc=3, borderpad=0)


ax2 = fig.add_subplot(133)
ax2.set_xscale("log")
ax2.set(xlim=(1e-6, 1e6), ylim=(-2, 6))

# Create inset in data coordinates using ax.transData as transform
axins3 = inset_axes(ax2, width="100%", height="100%",
                    bbox_to_anchor=(1e-2, 2, 1e3, 3),
                    bbox_transform=ax2.transData, loc=2, borderpad=0)

# Create an inset horizontally centered in figure coordinates and vertically
# bound to line up with the axes.
from matplotlib.transforms import blended_transform_factory  # noqa
transform = blended_transform_factory(fig.transFigure, ax2.transAxes)
axins4 = inset_axes(ax2, width="16%", height="34%",
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=transform, loc=8, borderpad=0)

plt.show()
```
### 9 - examples/subplots_axes_and_figures/axes_box_aspect.py:

Start line: 111, End line: 156

```python
axs[1, 0].scatter(x, y)
axs[0, 0].hist(x)
axs[1, 1].hist(y, orientation="horizontal")

plt.show()

############################################################################
# Square joint/marginal plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When setting the box aspect, one may still set the data aspect as well.
# Here we create an axes with a box twice as long as tall and use an "equal"
# data aspect for its contents, i.e. the circle actually stays circular.

fig6, ax = plt.subplots()

ax.add_patch(plt.Circle((5, 3), 1))
ax.set_aspect("equal", adjustable="datalim")
ax.set_box_aspect(0.5)
ax.autoscale()

plt.show()

############################################################################
# Box aspect for many subplots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It is possible to pass the box aspect to an axes at initialization. The
# following creates a 2 by 3 subplot grid with all square axes.

fig7, axs = plt.subplots(2, 3, subplot_kw=dict(box_aspect=1),
                         sharex=True, sharey=True, constrained_layout=True)

for i, ax in enumerate(axs.flat):
    ax.scatter(i % 3, -((i // 3) - 0.5)*200, c=[plt.cm.hsv(i / 6)], s=300)
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.set_box_aspect`
```
### 10 - examples/lines_bars_and_markers/scatter_hist.py:

Start line: 57, End line: 123

```python
#############################################################################
#
# Defining the axes positions using a gridspec
# --------------------------------------------
#
# We define a gridspec with unequal width- and height-ratios to achieve desired
# layout.  Also see the :doc:`/tutorials/intermediate/gridspec` tutorial.

# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)


#############################################################################
#
# Defining the axes positions using inset_axes
# --------------------------------------------
#
# `~.Axes.inset_axes` can be used to position marginals *outside* the main
# axes.  The advantage of doing so is that the aspect ratio of the main axes
# can be fixed, and the marginals will always be drawn relative to the position
# of the axes.

# Create a Figure, which doesn't have to be square.
fig = plt.figure(constrained_layout=True)
# Create the main axes, leaving 25% of the figure space at the top and on the
# right to position marginals.
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
# The main axes' aspect can be fixed.
ax.set(aspect=1)
# Create marginal axes, which have 25% of the size of the main axes.  Note that
# the inset axes are positioned *outside* (on the right and the top) of the
# main axes, by specifying axes coordinates greater than 1.  Axes coordinates
# less than 0 would likewise specify positions on the left and the bottom of
# the main axes.
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)

plt.show()


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.add_subplot`
#    - `matplotlib.figure.Figure.add_gridspec`
#    - `matplotlib.axes.Axes.inset_axes`
#    - `matplotlib.axes.Axes.scatter`
#    - `matplotlib.axes.Axes.hist`
```
### 16 - lib/matplotlib/pyplot.py:

Start line: 1587, End line: 1600

```python
def twinx(ax=None):
    """
    Make and return a second axes that shares the *x*-axis.  The new axes will
    overlay *ax* (or the current axes if *ax* is *None*), and its ticks will be
    on the right.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
    if ax is None:
        ax = gca()
    ax1 = ax.twinx()
    return ax1
```
### 26 - lib/matplotlib/pyplot.py:

Start line: 1603, End line: 1616

```python
def twiny(ax=None):
    """
    Make and return a second axes that shares the *y*-axis.  The new axes will
    overlay *ax* (or the current axes if *ax* is *None*), and its ticks will be
    on the top.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
    if ax is None:
        ax = gca()
    ax1 = ax.twiny()
    return ax1
```
