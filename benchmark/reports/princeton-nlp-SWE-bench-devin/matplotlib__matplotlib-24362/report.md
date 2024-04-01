# matplotlib__matplotlib-24362

| **matplotlib/matplotlib** | `aca6e9d5e98811ca37c442217914b15e78127c89` |
| ---- | ---- |
| **Indexed vectors** | 3941 |
| **Indexed tokens** | 1584041 |
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
diff --git a/lib/matplotlib/gridspec.py b/lib/matplotlib/gridspec.py
--- a/lib/matplotlib/gridspec.py
+++ b/lib/matplotlib/gridspec.py
@@ -276,21 +276,12 @@ def subplots(self, *, sharex=False, sharey=False, squeeze=True,
             raise ValueError("GridSpec.subplots() only works for GridSpecs "
                              "created with a parent figure")
 
-        if isinstance(sharex, bool):
+        if not isinstance(sharex, str):
             sharex = "all" if sharex else "none"
-        if isinstance(sharey, bool):
+        if not isinstance(sharey, str):
             sharey = "all" if sharey else "none"
-        # This check was added because it is very easy to type
-        # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
-        # In most cases, no error will ever occur, but mysterious behavior
-        # will result because what was intended to be the subplot index is
-        # instead treated as a bool for sharex.  This check should go away
-        # once sharex becomes kwonly.
-        if isinstance(sharex, Integral):
-            _api.warn_external(
-                "sharex argument to subplots() was an integer.  Did you "
-                "intend to use subplot() (without 's')?")
-        _api.check_in_list(["all", "row", "col", "none"],
+
+        _api.check_in_list(["all", "row", "col", "none", False, True],
                            sharex=sharex, sharey=sharey)
         if subplot_kw is None:
             subplot_kw = {}

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/gridspec.py | 279 | 293 | - | - | -


## Problem Statement

```
[Bug]: sharex and sharey don't accept 0 and 1 as bool values
### Bug summary

When using `0` or `1` in place of `False` or `True` in `sharex` or `sharex` arguments of `pyplot.subplots` an error is raised.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2,sharey=1)
\`\`\`


### Actual outcome

We get the following error : 
\`\`\`
Traceback (most recent call last):
  File "/***/shareyArg.py", line 3, in <module>
    fig, ax = plt.subplots(ncols=2,sharey=1)
  File "/***/matplotlib/lib/matplotlib/pyplot.py", line 1448, in subplots
    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
  File "/***/matplotlib/lib/matplotlib/figure.py", line 889, in subplots
    axs = gs.subplots(sharex=sharex, sharey=sharey, squeeze=squeeze,
  File "/***/matplotlib/lib/matplotlib/gridspec.py", line 293, in subplots
    _api.check_in_list(["all", "row", "col", "none"],
  File "/***/matplotlib/lib/matplotlib/_api/__init__.py", line 131, in check_in_list
    raise ValueError(msg)
ValueError: 1 is not a valid value for sharey; supported values are 'all', 'row', 'col', 'none'
\`\`\`

Note that using `sharex` instead of `sharey` produces the same error (albeit with the following warning :
\`\`\`
UserWarning: sharex argument to subplots() was an integer.  Did you intend to use subplot() (without 's')?
\`\`\`
but this is expected and not part of the present issue)

### Expected outcome

I expected values 1 and 0 to be understood as bool.



### Additional information

Suggested fix : 

\`\`\`patch
diff --git a/lib/matplotlib/gridspec.py b/lib/matplotlib/gridspec.py
index 06dd3f19f6..32ee7c306e 100644
--- a/lib/matplotlib/gridspec.py
+++ b/lib/matplotlib/gridspec.py
@@ -276,9 +276,9 @@ class GridSpecBase:
             raise ValueError("GridSpec.subplots() only works for GridSpecs "
                              "created with a parent figure")
 
-        if isinstance(sharex, bool):
+        if isinstance(sharex, bool) or sharex == 1 or sharex == 0:
             sharex = "all" if sharex else "none"
-        if isinstance(sharey, bool):
+        if isinstance(sharey, bool) or sharey == 1 or sharey == 0:
             sharey = "all" if sharey else "none"
         # This check was added because it is very easy to type
         # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
\`\`\`

Maybe not accepting 1 or 0 was done on purpose, but I did not find it very clear from the error message as `True` and `False` are accepted but not listed. 

I am happy to chat about an other fix, if this one doesn't do the trick. I can also create a PR in case you think this fix is good enough !

### Operating system

Linux 5.10.0-19-amd64 #1 SMP Debian 5.10.149-2

### Matplotlib Version

3.7.0.dev600+g0b6d3703ff

### Matplotlib Backend

TkAgg

### Python version

3.10.0

### Jupyter version

Not applicable

### Installation

git checkout

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 442 | 442 | 1968 | 
| 2 | 1 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 1227 | 1968 | 
| 3 | 2 examples/subplots_axes_and_figures/shared_axis_demo.py | 1 | 58| 496 | 1723 | 2464 | 
| 4 | 3 lib/matplotlib/pyplot.py | 1234 | 1298| 676 | 2399 | 30544 | 
| 5 | 4 lib/matplotlib/figure.py | 876 | 891| 245 | 2644 | 59159 | 
| 6 | 4 lib/matplotlib/pyplot.py | 1 | 89| 669 | 3313 | 59159 | 
| 7 | 5 examples/subplots_axes_and_figures/share_axis_lims_views.py | 1 | 25| 194 | 3507 | 59353 | 
| 8 | 6 tutorials/intermediate/arranging_axes.py | 264 | 343| 760 | 4267 | 63302 | 
| 9 | 7 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 4969 | 64413 | 
| 10 | 8 examples/ticks/ticks_too_many.py | 1 | 77| 742 | 5711 | 65155 | 
| 11 | 9 lib/matplotlib/backends/backend_wx.py | 10 | 60| 343 | 6054 | 77197 | 
| 12 | 9 lib/matplotlib/figure.py | 1865 | 1882| 262 | 6316 | 77197 | 
| 13 | 10 lib/matplotlib/_cm.py | 106 | 156| 787 | 7103 | 105640 | 
| 14 | 11 lib/mpl_toolkits/axes_grid1/axes_grid.py | 1 | 17| 119 | 7222 | 110563 | 
| 15 | 12 lib/matplotlib/rcsetup.py | 530 | 577| 356 | 7578 | 122504 | 
| 16 | 13 lib/matplotlib/axes/_base.py | 550 | 1259| 6200 | 13778 | 161866 | 
| 17 | 14 tutorials/intermediate/constrainedlayout_guide.py | 442 | 530| 776 | 14554 | 168430 | 
| 18 | 15 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 878 | 15432 | 170581 | 
| 19 | 16 doc/conf.py | 111 | 158| 481 | 15913 | 176192 | 
| 20 | 17 examples/subplots_axes_and_figures/axes_box_aspect.py | 111 | 156| 344 | 16257 | 177317 | 
| 21 | 17 tutorials/intermediate/arranging_axes.py | 345 | 413| 671 | 16928 | 177317 | 
| 22 | 17 tutorials/intermediate/constrainedlayout_guide.py | 636 | 721| 777 | 17705 | 177317 | 
| 23 | 17 tutorials/intermediate/arranging_axes.py | 100 | 178| 775 | 18480 | 177317 | 
| 24 | 17 lib/matplotlib/pyplot.py | 2835 | 2854| 226 | 18706 | 177317 | 
| 25 | 17 doc/conf.py | 178 | 273| 760 | 19466 | 177317 | 
| 26 | 17 lib/matplotlib/_cm.py | 767 | 791| 613 | 20079 | 177317 | 
| 27 | 17 lib/matplotlib/pyplot.py | 341 | 359| 175 | 20254 | 177317 | 
| 28 | 17 lib/matplotlib/figure.py | 2035 | 2056| 240 | 20494 | 177317 | 
| 29 | 17 lib/matplotlib/rcsetup.py | 126 | 171| 320 | 20814 | 177317 | 
| 30 | 18 examples/shapes_and_collections/artist_reference.py | 90 | 130| 319 | 21133 | 178477 | 
| 31 | 18 lib/matplotlib/_cm.py | 819 | 841| 545 | 21678 | 178477 | 
| 32 | 19 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 22203 | 179002 | 
| 33 | 19 lib/matplotlib/_cm.py | 1080 | 1211| 458 | 22661 | 179002 | 
| 34 | 19 doc/conf.py | 274 | 331| 412 | 23073 | 179002 | 
| 35 | 20 examples/userdemo/demo_gridspec03.py | 1 | 52| 421 | 23494 | 179423 | 
| 36 | 20 lib/matplotlib/_cm.py | 942 | 959| 398 | 23892 | 179423 | 
| 37 | 21 lib/matplotlib/__init__.py | 107 | 156| 328 | 24220 | 191441 | 
| 38 | 21 lib/matplotlib/pyplot.py | 2520 | 2551| 369 | 24589 | 191441 | 
| 39 | 22 examples/subplots_axes_and_figures/ganged_plots.py | 1 | 41| 344 | 24933 | 191785 | 
| 40 | 23 examples/subplots_axes_and_figures/gridspec_multicolumn.py | 1 | 34| 259 | 25192 | 192044 | 
| 41 | 23 lib/matplotlib/axes/_base.py | 1 | 32| 212 | 25404 | 192044 | 
| 42 | 23 lib/matplotlib/_cm.py | 611 | 635| 657 | 26061 | 192044 | 
| 43 | 23 lib/matplotlib/axes/_base.py | 469 | 4640| 813 | 26874 | 192044 | 
| 44 | 23 lib/matplotlib/pyplot.py | 753 | 812| 624 | 27498 | 192044 | 
| 45 | 24 examples/axes_grid1/demo_fixed_size_axes.py | 1 | 48| 326 | 27824 | 192370 | 
| 46 | 24 examples/subplots_axes_and_figures/axes_box_aspect.py | 1 | 110| 781 | 28605 | 192370 | 
| 47 | 25 lib/matplotlib/backends/backend_gtk3.py | 135 | 184| 461 | 29066 | 197212 | 
| 48 | 25 lib/matplotlib/_cm.py | 637 | 661| 660 | 29726 | 197212 | 
| 49 | 25 lib/matplotlib/_cm.py | 663 | 687| 670 | 30396 | 197212 | 
| 50 | 25 lib/matplotlib/figure.py | 1998 | 2033| 381 | 30777 | 197212 | 
| 51 | 25 tutorials/intermediate/constrainedlayout_guide.py | 532 | 634| 1011 | 31788 | 197212 | 
| 52 | 25 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 32283 | 197212 | 


## Missing Patch Files

 * 1: lib/matplotlib/gridspec.py

## Patch

```diff
diff --git a/lib/matplotlib/gridspec.py b/lib/matplotlib/gridspec.py
--- a/lib/matplotlib/gridspec.py
+++ b/lib/matplotlib/gridspec.py
@@ -276,21 +276,12 @@ def subplots(self, *, sharex=False, sharey=False, squeeze=True,
             raise ValueError("GridSpec.subplots() only works for GridSpecs "
                              "created with a parent figure")
 
-        if isinstance(sharex, bool):
+        if not isinstance(sharex, str):
             sharex = "all" if sharex else "none"
-        if isinstance(sharey, bool):
+        if not isinstance(sharey, str):
             sharey = "all" if sharey else "none"
-        # This check was added because it is very easy to type
-        # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
-        # In most cases, no error will ever occur, but mysterious behavior
-        # will result because what was intended to be the subplot index is
-        # instead treated as a bool for sharex.  This check should go away
-        # once sharex becomes kwonly.
-        if isinstance(sharex, Integral):
-            _api.warn_external(
-                "sharex argument to subplots() was an integer.  Did you "
-                "intend to use subplot() (without 's')?")
-        _api.check_in_list(["all", "row", "col", "none"],
+
+        _api.check_in_list(["all", "row", "col", "none", False, True],
                            sharex=sharex, sharey=sharey)
         if subplot_kw is None:
             subplot_kw = {}

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_subplots.py b/lib/matplotlib/tests/test_subplots.py
--- a/lib/matplotlib/tests/test_subplots.py
+++ b/lib/matplotlib/tests/test_subplots.py
@@ -84,7 +84,7 @@ def test_shared():
     plt.close(f)
 
     # test all option combinations
-    ops = [False, True, 'all', 'none', 'row', 'col']
+    ops = [False, True, 'all', 'none', 'row', 'col', 0, 1]
     for xo in ops:
         for yo in ops:
             f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, sharex=xo, sharey=yo)

```


## Code snippets

### 1 - examples/subplots_axes_and_figures/subplots_demo.py:

Start line: 171, End line: 212

```python
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x + 1, -y, 'tab:green')
ax4.plot(x + 2, -y**2, 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()

###############################################################################
# If you want a more complex sharing structure, you can first create the
# grid of axes with no sharing, and then call `.axes.Axes.sharex` or
# `.axes.Axes.sharey` to add sharing info a posteriori.

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title("main")
axs[1, 0].plot(x, y**2)
axs[1, 0].set_title("shares x with main")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(x + 1, y + 1)
axs[0, 1].set_title("unrelated")
axs[1, 1].plot(x + 2, y + 2)
axs[1, 1].set_title("also unrelated")
fig.tight_layout()

###############################################################################
# Polar axes
# """"""""""
#
# The parameter *subplot_kw* of `.pyplot.subplots` controls the subplot
# properties (see also `.Figure.add_subplot`). In particular, this can be used
# to create a grid of polar Axes.

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax1.plot(x, y)
ax2.plot(x, y ** 2)

plt.show()
```
### 2 - examples/subplots_axes_and_figures/subplots_demo.py:

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
### 3 - examples/subplots_axes_and_figures/shared_axis_demo.py:

Start line: 1, End line: 58

```python
"""
===========
Shared Axis
===========

You can share the x or y axis limits for one axis with another by
passing an `~.axes.Axes` instance as a *sharex* or *sharey* keyword argument.

Changing the axis limits on one axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the Axes will follow each other on their shared axis.  Ditto for
changes in the axis scaling (e.g., log vs. linear).  However, it is
possible to have differences in tick labeling, e.g., you can selectively
turn off the tick labels on one Axes.

The example below shows how to customize the tick labels on the
various axes.  Shared axes share the tick locator, tick formatter,
view limits, and transformation (e.g., log, linear).  But the ticklabels
themselves do not share properties.  This is a feature and not a bug,
because you may want to make the tick labels smaller on the upper
axes, e.g., in the example below.

If you want to turn off the ticklabels for a given Axes (e.g., on
subplot(211) or subplot(212), you cannot do the standard trick::

   setp(ax2, xticklabels=[])

because this changes the tick Formatter, which is shared among all
Axes.  But you can alter the visibility of the labels, which is a
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
### 4 - lib/matplotlib/pyplot.py:

Start line: 1234, End line: 1298

```python
@_docstring.dedent_interpd
def subplot(*args, **kwargs):
    # Here we will only normalize `polar=True` vs `projection='polar'` and let
    # downstream code deal with the rest.
    unset = object()
    projection = kwargs.get('projection', unset)
    polar = kwargs.pop('polar', unset)
    if polar is not unset and polar:
        # if we got mixed messages from the user, raise
        if projection is not unset and projection != 'polar':
            raise ValueError(
                f"polar={polar}, yet projection={projection!r}. "
                "Only one of these arguments should be supplied."
            )
        kwargs['projection'] = projection = 'polar'

    # if subplot called without arguments, create subplot(1, 1, 1)
    if len(args) == 0:
        args = (1, 1, 1)

    # This check was added because it is very easy to type subplot(1, 2, False)
    # when subplots(1, 2, False) was intended (sharex=False, that is). In most
    # cases, no error will ever occur, but mysterious behavior can result
    # because what was intended to be the sharex argument is instead treated as
    # a subplot index for subplot()
    if len(args) >= 3 and isinstance(args[2], bool):
        _api.warn_external("The subplot index argument to subplot() appears "
                           "to be a boolean. Did you intend to use "
                           "subplots()?")
    # Check for nrows and ncols, which are not valid subplot args:
    if 'nrows' in kwargs or 'ncols' in kwargs:
        raise TypeError("subplot() got an unexpected keyword argument 'ncols' "
                        "and/or 'nrows'.  Did you intend to call subplots()?")

    fig = gcf()

    # First, search for an existing subplot with a matching spec.
    key = SubplotSpec._from_subplot_args(fig, args)

    for ax in fig.axes:
        # if we found an Axes at the position sort out if we can re-use it
        if ax.get_subplotspec() == key:
            # if the user passed no kwargs, re-use
            if kwargs == {}:
                break
            # if the axes class and kwargs are identical, reuse
            elif ax._projection_init == fig._process_projection_requirements(
                *args, **kwargs
            ):
                break
    else:
        # we have exhausted the known Axes and none match, make a new one!
        ax = fig.add_subplot(*args, **kwargs)

    fig.sca(ax)

    axes_to_delete = [other for other in fig.axes
                      if other != ax and ax.bbox.fully_overlaps(other.bbox)]
    if axes_to_delete:
        _api.warn_deprecated(
            "3.6", message="Auto-removal of overlapping axes is deprecated "
            "since %(since)s and will be removed %(removal)s; explicitly call "
            "ax.remove() as needed.")
    for ax_to_del in axes_to_delete:
        delaxes(ax_to_del)

    return ax
```
### 5 - lib/matplotlib/figure.py:

Start line: 876, End line: 891

```python
class FigureBase(Artist):

    def subplots(self, nrows=1, ncols=1, *, sharex=False, sharey=False,
                 squeeze=True, width_ratios=None, height_ratios=None,
                 subplot_kw=None, gridspec_kw=None):
        gridspec_kw = dict(gridspec_kw or {})
        if height_ratios is not None:
            if 'height_ratios' in gridspec_kw:
                raise ValueError("'height_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['height_ratios'] = height_ratios
        if width_ratios is not None:
            if 'width_ratios' in gridspec_kw:
                raise ValueError("'width_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['width_ratios'] = width_ratios

        gs = self.add_gridspec(nrows, ncols, figure=self, **gridspec_kw)
        axs = gs.subplots(sharex=sharex, sharey=sharey, squeeze=squeeze,
                          subplot_kw=subplot_kw)
        return axs
```
### 6 - lib/matplotlib/pyplot.py:

Start line: 1, End line: 89

```python
# Note: The first part of this file can be modified in place, but the latter
# part is autogenerated by the boilerplate.py script.

"""
`matplotlib.pyplot` is a state-based interface to matplotlib. It provides
an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
screen, and acts as the figure GUI manager.

pyplot is mainly intended for interactive plots and simple cases of
programmatic plot generation::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

The explicit object-oriented API is recommended for complex plots, though
pyplot is still usually used to create the figure and often the axes in the
figure. See `.pyplot.figure`, `.pyplot.subplots`, and
`.pyplot.subplot_mosaic` to create figures, and
:doc:`Axes API </api/axes_api>` for the plotting methods on an Axes::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)


See :ref:`api_interfaces` for an explanation of the tradeoffs between the
implicit and explicit interfaces.
"""

from contextlib import ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
from numbers import Number
import re
import sys
import threading
import time

from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import rcsetup, style
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import FigureCanvasBase, MouseButton
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcParams, rcParamsDefault, get_backend, rcParamsOrig
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names

from matplotlib import cm
from matplotlib.cm import _colormaps as colormaps, register_cmap
from matplotlib.colors import _color_sequences as color_sequences

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget

from .ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)

_log = logging.getLogger(__name__)
```
### 7 - examples/subplots_axes_and_figures/share_axis_lims_views.py:

Start line: 1, End line: 25

```python
"""
Sharing axis limits and views
=============================

It's common to make two or more plots which share an axis, e.g., two subplots
with time as a common axis.  When you pan and zoom around on one, you want the
other to move around with you.  To facilitate this, matplotlib Axes support a
``sharex`` and ``sharey`` attribute.  When you create a `~.pyplot.subplot` or
`~.pyplot.axes`, you can pass in a keyword indicating what axes you want to
share with.
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 10, 0.01)

ax1 = plt.subplot(211)
ax1.plot(t, np.sin(2*np.pi*t))

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(t, np.sin(4*np.pi*t))

plt.show()
```
### 8 - tutorials/intermediate/arranging_axes.py:

Start line: 264, End line: 343

```python
spec = fig.add_gridspec(ncols=2, nrows=2)

ax0 = fig.add_subplot(spec[0, 0])
annotate_axes(ax0, 'ax0')

ax1 = fig.add_subplot(spec[0, 1])
annotate_axes(ax1, 'ax1')

ax2 = fig.add_subplot(spec[1, 0])
annotate_axes(ax2, 'ax2')

ax3 = fig.add_subplot(spec[1, 1])
annotate_axes(ax3, 'ax3')

fig.suptitle('Manually added subplots using add_gridspec')

##############################################################################
# Axes spanning rows or grids in a grid
# -------------------------------------
#
# We can index the *spec* array using `NumPy slice syntax
# <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
# and the new Axes will span the slice.  This would be the same
# as ``fig, axd = plt.subplot_mosaic([['ax0', 'ax0'], ['ax1', 'ax2']], ...)``:

fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
spec = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(spec[0, :])
annotate_axes(ax0, 'ax0')

ax10 = fig.add_subplot(spec[1, 0])
annotate_axes(ax10, 'ax10')

ax11 = fig.add_subplot(spec[1, 1])
annotate_axes(ax11, 'ax11')

fig.suptitle('Manually added subplots, spanning a column')

###############################################################################
# Manual adjustments to a *GridSpec* layout
# -----------------------------------------
#
# When a  *GridSpec* is explicitly used, you can adjust the layout
# parameters of subplots that are created from the  *GridSpec*.  Note this
# option is not compatible with ``constrained_layout`` or
# `.Figure.tight_layout` which both ignore *left* and *right* and adjust
# subplot sizes to fill the figure.  Usually such manual placement
# requires iterations to make the Axes tick labels not overlap the Axes.
#
# These spacing parameters can also be passed to `~.pyplot.subplots` and
# `~.pyplot.subplot_mosaic` as the *gridspec_kw* argument.

fig = plt.figure(layout=None, facecolor='0.9')
gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.75,
                      hspace=0.1, wspace=0.05)
ax0 = fig.add_subplot(gs[:-1, :])
annotate_axes(ax0, 'ax0')
ax1 = fig.add_subplot(gs[-1, :-1])
annotate_axes(ax1, 'ax1')
ax2 = fig.add_subplot(gs[-1, -1])
annotate_axes(ax2, 'ax2')
fig.suptitle('Manual gridspec with right=0.75')

###############################################################################
# Nested layouts with SubplotSpec
# -------------------------------
#
# You can create nested layout similar to `~.Figure.subfigures` using
# `~.gridspec.SubplotSpec.subgridspec`.  Here the Axes spines *are*
# aligned.
#
# Note this is also available from the more verbose
# `.gridspec.GridSpecFromSubplotSpec`.

fig = plt.figure(layout="constrained")
gs0 = fig.add_gridspec(1, 2)

gs00 = gs0[0].subgridspec(2, 2)
gs01 = gs0[1].subgridspec(3, 1)
```
### 9 - examples/lines_bars_and_markers/scatter_hist.py:

Start line: 57, End line: 123

```python
#############################################################################
#
# Defining the axes positions using a gridspec
# --------------------------------------------
#
# We define a gridspec with unequal width- and height-ratios to achieve desired
# layout.  Also see the :doc:`/tutorials/intermediate/arranging_axes` tutorial.

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
### 10 - examples/ticks/ticks_too_many.py:

Start line: 1, End line: 77

```python
"""
=====================
Fixing too many ticks
=====================

One common cause for unexpected tick behavior is passing a list of strings
instead of numbers or datetime objects. This can easily happen without notice
when reading in a comma-delimited text file. Matplotlib treats lists of strings
as *categorical* variables
(:doc:`/gallery/lines_bars_and_markers/categorical_variables`), and by default
puts one tick per category, and plots them in the order in which they are
supplied.  If this is not desired, the solution is to convert the strings to
a numeric type as in the following examples.

"""

############################################################################
# Example 1: Strings can lead to an unexpected order of number ticks
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.5))
x = ['1', '5', '2', '3']
y = [1, 4, 2, 3]
ax[0].plot(x, y, 'd')
ax[0].tick_params(axis='x', color='r', labelcolor='r')
ax[0].set_xlabel('Categories')
ax[0].set_title('Ticks seem out of order / misplaced')

# convert to numbers:
x = np.asarray(x, dtype='float')
ax[1].plot(x, y, 'd')
ax[1].set_xlabel('Floats')
ax[1].set_title('Ticks as expected')

############################################################################
# Example 2: Strings can lead to very many ticks
# ----------------------------------------------
# If *x* has 100 elements, all strings, then we would have 100 (unreadable)
# ticks, and again the solution is to convert the strings to floats:

fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
x = [f'{xx}' for xx in np.arange(100)]
y = np.arange(100)
ax[0].plot(x, y)
ax[0].tick_params(axis='x', color='r', labelcolor='r')
ax[0].set_title('Too many ticks')
ax[0].set_xlabel('Categories')

ax[1].plot(np.asarray(x, float), y)
ax[1].set_title('x converted to numbers')
ax[1].set_xlabel('Floats')

############################################################################
# Example 3: Strings can lead to an unexpected order of datetime ticks
# --------------------------------------------------------------------
# A common case is when dates are read from a CSV file, they need to be
# converted from strings to datetime objects to get the proper date locators
# and formatters.

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.75))
x = ['2021-10-01', '2021-11-02', '2021-12-03', '2021-09-01']
y = [0, 2, 3, 1]
ax[0].plot(x, y, 'd')
ax[0].tick_params(axis='x', labelrotation=90, color='r', labelcolor='r')
ax[0].set_title('Dates out of order')

# convert to datetime64
x = np.asarray(x, dtype='datetime64[s]')
ax[1].plot(x, y, 'd')
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('x converted to datetimes')

plt.show()
```
