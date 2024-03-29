# matplotlib__matplotlib-23198

| **matplotlib/matplotlib** | `3407cbc42f0e70595813e2b1816d432591558921` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 7 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/lib/matplotlib/backends/qt_editor/figureoptions.py b/lib/matplotlib/backends/qt_editor/figureoptions.py
--- a/lib/matplotlib/backends/qt_editor/figureoptions.py
+++ b/lib/matplotlib/backends/qt_editor/figureoptions.py
@@ -230,12 +230,12 @@ def apply_callback(data):
         # re-generate legend, if checkbox is checked
         if generate_legend:
             draggable = None
-            ncol = 1
+            ncols = 1
             if axes.legend_ is not None:
                 old_legend = axes.get_legend()
                 draggable = old_legend._draggable is not None
-                ncol = old_legend._ncol
-            new_legend = axes.legend(ncol=ncol)
+                ncols = old_legend._ncols
+            new_legend = axes.legend(ncols=ncols)
             if new_legend:
                 new_legend.set_draggable(draggable)
 
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -162,9 +162,12 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
 
         loc='upper right', bbox_to_anchor=(0.5, 0.5)
 
-ncol : int, default: 1
+ncols : int, default: 1
     The number of columns that the legend has.
 
+    For backward compatibility, the spelling *ncol* is also supported
+    but it is discouraged. If both are given, *ncols* takes precedence.
+
 prop : None or `matplotlib.font_manager.FontProperties` or dict
     The font properties of the legend. If None (default), the current
     :data:`matplotlib.rcParams` will be used.
@@ -317,7 +320,7 @@ def __init__(
         borderaxespad=None,  # pad between the axes and legend border
         columnspacing=None,  # spacing between columns
 
-        ncol=1,     # number of columns
+        ncols=1,     # number of columns
         mode=None,  # horizontal distribution of columns: None or "expand"
 
         fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
@@ -333,6 +336,8 @@ def __init__(
         frameon=None,         # draw frame
         handler_map=None,
         title_fontproperties=None,  # properties for the legend title
+        *,
+        ncol=1  # synonym for ncols (backward compatibility)
     ):
         """
         Parameters
@@ -418,8 +423,8 @@ def val_or_rc(val, rc_name):
 
         handles = list(handles)
         if len(handles) < 2:
-            ncol = 1
-        self._ncol = ncol
+            ncols = 1
+        self._ncols = ncols if ncols != 1 else ncol
 
         if self.numpoints <= 0:
             raise ValueError("numpoints must be > 0; it was %d" % numpoints)
@@ -581,6 +586,10 @@ def _set_loc(self, loc):
         self.stale = True
         self._legend_box.set_offset(self._findoffset)
 
+    def set_ncols(self, ncols):
+        """Set the number of columns."""
+        self._ncols = ncols
+
     def _get_loc(self):
         return self._loc_real
 
@@ -767,12 +776,12 @@ def _init_legend_box(self, handles, labels, markerfirst=True):
                 handles_and_labels.append((handlebox, textbox))
 
         columnbox = []
-        # array_split splits n handles_and_labels into ncol columns, with the
-        # first n%ncol columns having an extra entry.  filter(len, ...) handles
-        # the case where n < ncol: the last ncol-n columns are empty and get
-        # filtered out.
-        for handles_and_labels_column \
-                in filter(len, np.array_split(handles_and_labels, self._ncol)):
+        # array_split splits n handles_and_labels into ncols columns, with the
+        # first n%ncols columns having an extra entry.  filter(len, ...)
+        # handles the case where n < ncols: the last ncols-n columns are empty
+        # and get filtered out.
+        for handles_and_labels_column in filter(
+                len, np.array_split(handles_and_labels, self._ncols)):
             # pack handlebox and labelbox into itembox
             itemboxes = [HPacker(pad=0,
                                  sep=self.handletextpad * fontsize,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/backends/qt_editor/figureoptions.py | 233 | 238 | - | - | -
| lib/matplotlib/legend.py | 165 | 165 | - | - | -
| lib/matplotlib/legend.py | 320 | 320 | - | - | -
| lib/matplotlib/legend.py | 336 | 336 | - | - | -
| lib/matplotlib/legend.py | 421 | 422 | - | - | -
| lib/matplotlib/legend.py | 584 | 584 | - | - | -
| lib/matplotlib/legend.py | 770 | 775 | - | - | -


## Problem Statement

```
Inconsistency in keyword-arguments ncol/ncols, nrow/nrows
I find it quite inconsistent that one sometimes has to specify `ncols` and sometimes `ncol`. For example:

\`\`\`python
plt.subplots(ncols=2)
\`\`\`

while

\`\`\`python
axis.legend(ncol=2)
\`\`\`

(Likewise for `nrows`/`nrow`)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 tutorials/intermediate/arranging_axes.py | 100 | 178| 775 | 775 | 3949 | 
| 2 | 1 tutorials/intermediate/arranging_axes.py | 264 | 343| 760 | 1535 | 3949 | 
| 3 | 2 lib/matplotlib/axes/_base.py | 469 | 4613| 813 | 2348 | 42670 | 
| 4 | 2 tutorials/intermediate/arranging_axes.py | 345 | 413| 671 | 3019 | 42670 | 
| 5 | 3 tutorials/intermediate/constrainedlayout_guide.py | 442 | 530| 776 | 3795 | 49247 | 
| 6 | 3 tutorials/intermediate/constrainedlayout_guide.py | 636 | 721| 789 | 4584 | 49247 | 
| 7 | 3 lib/matplotlib/axes/_base.py | 241 | 313| 751 | 5335 | 49247 | 
| 8 | 4 lib/matplotlib/pyplot.py | 1258 | 1322| 686 | 6021 | 76865 | 
| 9 | 5 tutorials/colors/colormaps.py | 203 | 260| 653 | 6674 | 81647 | 
| 10 | 6 examples/images_contours_and_fields/pcolormesh_grids.py | 82 | 129| 484 | 7158 | 82952 | 
| 11 | 6 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 7939 | 82952 | 
| 12 | 6 tutorials/intermediate/constrainedlayout_guide.py | 270 | 352| 804 | 8743 | 82952 | 
| 13 | 7 lib/mpl_toolkits/axes_grid1/axes_grid.py | 206 | 257| 336 | 9079 | 87841 | 
| 14 | 7 examples/images_contours_and_fields/pcolormesh_grids.py | 1 | 81| 821 | 9900 | 87841 | 
| 15 | 7 tutorials/intermediate/constrainedlayout_guide.py | 532 | 634| 1011 | 10911 | 87841 | 
| 16 | 7 lib/matplotlib/pyplot.py | 2217 | 2292| 763 | 11674 | 87841 | 
| 17 | 7 tutorials/intermediate/arranging_axes.py | 179 | 263| 908 | 12582 | 87841 | 
| 18 | 8 tutorials/introductory/quick_start.py | 475 | 561| 1039 | 13621 | 93782 | 
| 19 | 8 tutorials/intermediate/constrainedlayout_guide.py | 354 | 441| 772 | 14393 | 93782 | 
| 20 | 9 examples/images_contours_and_fields/irregulardatagrid.py | 79 | 95| 144 | 14537 | 94690 | 
| 21 | 10 examples/misc/keyword_plotting.py | 1 | 29| 216 | 14753 | 94906 | 
| 22 | 10 tutorials/intermediate/constrainedlayout_guide.py | 187 | 269| 814 | 15567 | 94906 | 
| 23 | 11 examples/images_contours_and_fields/colormap_normalizations_symlognorm.py | 1 | 85| 820 | 16387 | 95726 | 
| 24 | 12 tutorials/introductory/pyplot.py | 91 | 250| 1519 | 17906 | 100202 | 
| 25 | 13 examples/misc/pythonic_matplotlib.py | 1 | 81| 610 | 18516 | 100812 | 
| 26 | 14 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 19301 | 102779 | 
| 27 | 14 tutorials/introductory/quick_start.py | 291 | 388| 986 | 20287 | 102779 | 
| 28 | 14 examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 741 | 21028 | 102779 | 
| 29 | 14 lib/matplotlib/pyplot.py | 2425 | 2435| 182 | 21210 | 102779 | 
| 30 | 14 lib/matplotlib/pyplot.py | 1 | 88| 666 | 21876 | 102779 | 
| 31 | 15 tutorials/provisional/mosaic.py | 59 | 170| 789 | 22665 | 105068 | 
| 32 | 15 tutorials/provisional/mosaic.py | 171 | 296| 779 | 23444 | 105068 | 
| 33 | 16 examples/statistics/boxplot.py | 1 | 73| 747 | 24191 | 106129 | 
| 34 | 17 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 24858 | 107552 | 
| 35 | 17 examples/images_contours_and_fields/irregulardatagrid.py | 1 | 78| 764 | 25622 | 107552 | 
| 36 | 17 tutorials/colors/colormaps.py | 261 | 367| 1202 | 26824 | 107552 | 
| 37 | 18 examples/subplots_axes_and_figures/gridspec_multicolumn.py | 1 | 34| 259 | 27083 | 107811 | 
| 38 | 19 lib/matplotlib/gridspec.py | 251 | 262| 135 | 27218 | 114293 | 
| 39 | 19 tutorials/introductory/quick_start.py | 389 | 474| 931 | 28149 | 114293 | 
| 40 | 19 lib/matplotlib/pyplot.py | 782 | 834| 572 | 28721 | 114293 | 
| 41 | 19 lib/matplotlib/gridspec.py | 615 | 665| 442 | 29163 | 114293 | 
| 42 | 19 lib/matplotlib/pyplot.py | 2499 | 2530| 369 | 29532 | 114293 | 
| 43 | 20 examples/statistics/confidence_ellipse.py | 183 | 227| 338 | 29870 | 116143 | 
| 44 | 21 examples/subplots_axes_and_figures/secondary_axis.py | 1 | 101| 778 | 30648 | 117577 | 
| 45 | 22 examples/scales/asinh_demo.py | 1 | 86| 729 | 31377 | 118494 | 
| 46 | 22 tutorials/intermediate/constrainedlayout_guide.py | 1 | 112| 829 | 32206 | 118494 | 
| 47 | 23 examples/subplots_axes_and_figures/two_scales.py | 1 | 52| 407 | 32613 | 118901 | 
| 48 | 23 lib/matplotlib/gridspec.py | 101 | 113| 137 | 32750 | 118901 | 
| 49 | 24 tutorials/colors/colormapnorms.py | 92 | 168| 831 | 33581 | 122591 | 
| 50 | 25 examples/images_contours_and_fields/pcolormesh_levels.py | 1 | 83| 773 | 34354 | 123813 | 
| 51 | 25 lib/mpl_toolkits/axes_grid1/axes_grid.py | 411 | 573| 1606 | 35960 | 123813 | 
| 52 | 26 tutorials/intermediate/autoscale.py | 104 | 173| 729 | 36689 | 125386 | 
| 53 | 27 examples/scales/scales.py | 1 | 119| 771 | 37460 | 126157 | 
| 54 | 28 tutorials/text/text_intro.py | 263 | 330| 831 | 38291 | 130033 | 
| 55 | 28 lib/matplotlib/pyplot.py | 2988 | 3084| 792 | 39083 | 130033 | 
| 56 | 29 examples/color/colormap_reference.py | 1 | 45| 553 | 39636 | 130916 | 


## Missing Patch Files

 * 1: lib/matplotlib/backends/qt_editor/figureoptions.py
 * 2: lib/matplotlib/legend.py

## Patch

```diff
diff --git a/lib/matplotlib/backends/qt_editor/figureoptions.py b/lib/matplotlib/backends/qt_editor/figureoptions.py
--- a/lib/matplotlib/backends/qt_editor/figureoptions.py
+++ b/lib/matplotlib/backends/qt_editor/figureoptions.py
@@ -230,12 +230,12 @@ def apply_callback(data):
         # re-generate legend, if checkbox is checked
         if generate_legend:
             draggable = None
-            ncol = 1
+            ncols = 1
             if axes.legend_ is not None:
                 old_legend = axes.get_legend()
                 draggable = old_legend._draggable is not None
-                ncol = old_legend._ncol
-            new_legend = axes.legend(ncol=ncol)
+                ncols = old_legend._ncols
+            new_legend = axes.legend(ncols=ncols)
             if new_legend:
                 new_legend.set_draggable(draggable)
 
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -162,9 +162,12 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
 
         loc='upper right', bbox_to_anchor=(0.5, 0.5)
 
-ncol : int, default: 1
+ncols : int, default: 1
     The number of columns that the legend has.
 
+    For backward compatibility, the spelling *ncol* is also supported
+    but it is discouraged. If both are given, *ncols* takes precedence.
+
 prop : None or `matplotlib.font_manager.FontProperties` or dict
     The font properties of the legend. If None (default), the current
     :data:`matplotlib.rcParams` will be used.
@@ -317,7 +320,7 @@ def __init__(
         borderaxespad=None,  # pad between the axes and legend border
         columnspacing=None,  # spacing between columns
 
-        ncol=1,     # number of columns
+        ncols=1,     # number of columns
         mode=None,  # horizontal distribution of columns: None or "expand"
 
         fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
@@ -333,6 +336,8 @@ def __init__(
         frameon=None,         # draw frame
         handler_map=None,
         title_fontproperties=None,  # properties for the legend title
+        *,
+        ncol=1  # synonym for ncols (backward compatibility)
     ):
         """
         Parameters
@@ -418,8 +423,8 @@ def val_or_rc(val, rc_name):
 
         handles = list(handles)
         if len(handles) < 2:
-            ncol = 1
-        self._ncol = ncol
+            ncols = 1
+        self._ncols = ncols if ncols != 1 else ncol
 
         if self.numpoints <= 0:
             raise ValueError("numpoints must be > 0; it was %d" % numpoints)
@@ -581,6 +586,10 @@ def _set_loc(self, loc):
         self.stale = True
         self._legend_box.set_offset(self._findoffset)
 
+    def set_ncols(self, ncols):
+        """Set the number of columns."""
+        self._ncols = ncols
+
     def _get_loc(self):
         return self._loc_real
 
@@ -767,12 +776,12 @@ def _init_legend_box(self, handles, labels, markerfirst=True):
                 handles_and_labels.append((handlebox, textbox))
 
         columnbox = []
-        # array_split splits n handles_and_labels into ncol columns, with the
-        # first n%ncol columns having an extra entry.  filter(len, ...) handles
-        # the case where n < ncol: the last ncol-n columns are empty and get
-        # filtered out.
-        for handles_and_labels_column \
-                in filter(len, np.array_split(handles_and_labels, self._ncol)):
+        # array_split splits n handles_and_labels into ncols columns, with the
+        # first n%ncols columns having an extra entry.  filter(len, ...)
+        # handles the case where n < ncols: the last ncols-n columns are empty
+        # and get filtered out.
+        for handles_and_labels_column in filter(
+                len, np.array_split(handles_and_labels, self._ncols)):
             # pack handlebox and labelbox into itembox
             itemboxes = [HPacker(pad=0,
                                  sep=self.handletextpad * fontsize,

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -4013,7 +4013,7 @@ def test_hist_stacked_bar():
     fig, ax = plt.subplots()
     ax.hist(d, bins=10, histtype='barstacked', align='mid', color=colors,
             label=labels)
-    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1)
+    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncols=1)
 
 
 def test_hist_barstacked_bottom_unchanged():
diff --git a/lib/matplotlib/tests/test_legend.py b/lib/matplotlib/tests/test_legend.py
--- a/lib/matplotlib/tests/test_legend.py
+++ b/lib/matplotlib/tests/test_legend.py
@@ -5,7 +5,7 @@
 import numpy as np
 import pytest
 
-from matplotlib.testing.decorators import image_comparison
+from matplotlib.testing.decorators import check_figures_equal, image_comparison
 from matplotlib.testing._markers import needs_usetex
 import matplotlib.pyplot as plt
 import matplotlib as mpl
@@ -148,7 +148,7 @@ def test_fancy():
     plt.errorbar(np.arange(10), np.arange(10), xerr=0.5,
                  yerr=0.5, label='XX')
     plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
-               ncol=2, shadow=True, title="My legend", numpoints=1)
+               ncols=2, shadow=True, title="My legend", numpoints=1)
 
 
 @image_comparison(['framealpha'], remove_text=True,
@@ -190,7 +190,7 @@ def test_legend_expand():
         ax.plot(x, x - 50, 'o', label='y=-1')
         l2 = ax.legend(loc='right', mode=mode)
         ax.add_artist(l2)
-        ax.legend(loc='lower left', mode=mode, ncol=2)
+        ax.legend(loc='lower left', mode=mode, ncols=2)
 
 
 @image_comparison(['hatching'], remove_text=True, style='default')
@@ -926,3 +926,12 @@ def test_legend_markers_from_line2d():
 
     assert markers == new_markers == _markers
     assert labels == new_labels
+
+
+@check_figures_equal()
+def test_ncol_ncols(fig_test, fig_ref):
+    # Test that both ncol and ncols work
+    strings = ["a", "b", "c", "d", "e", "f"]
+    ncols = 3
+    fig_test.legend(strings, ncol=ncols)
+    fig_ref.legend(strings, ncols=ncols)
diff --git a/lib/matplotlib/tests/test_offsetbox.py b/lib/matplotlib/tests/test_offsetbox.py
--- a/lib/matplotlib/tests/test_offsetbox.py
+++ b/lib/matplotlib/tests/test_offsetbox.py
@@ -117,7 +117,7 @@ def test_expand_with_tight_layout():
     d2 = [2, 1]
     ax.plot(d1, label='series 1')
     ax.plot(d2, label='series 2')
-    ax.legend(ncol=2, mode='expand')
+    ax.legend(ncols=2, mode='expand')
 
     fig.tight_layout()  # where the crash used to happen
 

```


## Code snippets

### 1 - tutorials/intermediate/arranging_axes.py:

Start line: 100, End line: 178

```python
import numpy as np

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5),
                        layout="constrained")
# add an artist, in this case a nice label in the middle...
for row in range(2):
    for col in range(2):
        axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
                               transform=axs[row, col].transAxes,
                               ha='center', va='center', fontsize=18,
                               color='darkgrey')
fig.suptitle('plt.subplots()')

##############################################################################
# We will annotate a lot of Axes, so lets encapsulate the annotation, rather
# than having that large piece of annotation code every time we need it:


def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")


##############################################################################
# The same effect can be achieved with `~.pyplot.subplot_mosaic`,
# but the return type is a dictionary instead of an array, where the user
# can give the keys useful meanings.  Here we provide two lists, each list
# representing a row, and each element in the list a key representing the
# column.

fig, axd = plt.subplot_mosaic([['upper left', 'upper right'],
                               ['lower left', 'lower right']],
                              figsize=(5.5, 3.5), layout="constrained")
for k in axd:
    annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')

#############################################################################
#
# Grids of fixed-aspect ratio Axes
# --------------------------------
#
# Fixed-aspect ratio axes are common for images or maps.  However, they
# present a challenge to layout because two sets of constraints are being
# imposed on the size of the Axes - that they fit in the figure and that they
# have a set aspect ratio.  This leads to large gaps between Axes by default:
#

fig, axs = plt.subplots(2, 2, layout="constrained", figsize=(5.5, 3.5))
for ax in axs.flat:
    ax.set_aspect(1)
fig.suptitle('Fixed aspect Axes')

############################################################################
# One way to address this is to change the aspect of the figure to be close
# to the aspect ratio of the Axes, however that requires trial and error.
# Matplotlib also supplies ``layout="compressed"``, which will work with
# simple grids to reduce the gaps between Axes.  (The ``mpl_toolkits`` also
# provides `~.mpl_toolkits.axes_grid1.axes_grid.ImageGrid` to accomplish
# a similar effect, but with a non-standard Axes class).

fig, axs = plt.subplots(2, 2, layout="compressed", figsize=(5.5, 3.5))
for ax in axs.flat:
    ax.set_aspect(1)
fig.suptitle('Fixed aspect Axes: compressed')


############################################################################
# Axes spanning rows or columns in a grid
# ---------------------------------------
#
# Sometimes we want Axes to span rows or columns of the grid.
# There are actually multiple ways to accomplish this, but the most
# convenient is probably to use `~.pyplot.subplot_mosaic` by repeating one
# of the keys:

fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(5.5, 3.5), layout="constrained")
```
### 2 - tutorials/intermediate/arranging_axes.py:

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
### 3 - lib/matplotlib/axes/_base.py:

Start line: 469, End line: 4613

```python
class _process_plot_var_args:

    def _plot_args(self, tup, kwargs, *,
                   return_kwargs=False, ambiguous_fmt_datakey=False):
        # ... other code
        for prop_name, val in zip(('linestyle', 'marker', 'color'),
                                  (linestyle, marker, color)):
            if val is not None:
                # check for conflicts between fmt and kwargs
                if (fmt.lower() != 'none'
                        and prop_name in kwargs
                        and val != 'None'):
                    # Technically ``plot(x, y, 'o', ls='--')`` is a conflict
                    # because 'o' implicitly unsets the linestyle
                    # (linestyle='None').
                    # We'll gracefully not warn in this case because an
                    # explicit set via kwargs can be seen as intention to
                    # override an implicit unset.
                    # Note: We don't val.lower() != 'none' because val is not
                    # necessarily a string (can be a tuple for colors). This
                    # is safe, because *val* comes from _process_plot_format()
                    # which only returns 'None'.
                    _api.warn_external(
                        f"{prop_name} is redundantly defined by the "
                        f"'{prop_name}' keyword argument and the fmt string "
                        f'"{fmt}" (-> {prop_name}={val!r}). The keyword '
                        f"argument will take precedence.")
                kw[prop_name] = val

        if len(xy) == 2:
            x = _check_1d(xy[0])
            y = _check_1d(xy[1])
        else:
            x, y = index_of(xy[-1])

        if self.axes.xaxis is not None:
            self.axes.xaxis.update_units(x)
        if self.axes.yaxis is not None:
            self.axes.yaxis.update_units(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same first dimension, but "
                             f"have shapes {x.shape} and {y.shape}")
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError(f"x and y can be no greater than 2D, but have "
                             f"shapes {x.shape} and {y.shape}")
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.command == 'plot':
            make_artist = self._makeline
        else:
            kw['closed'] = kwargs.get('closed', True)
            make_artist = self._makefill

        ncx, ncy = x.shape[1], y.shape[1]
        if ncx > 1 and ncy > 1 and ncx != ncy:
            raise ValueError(f"x has {ncx} columns but y has {ncy} columns")
        if ncx == 0 or ncy == 0:
            return []

        label = kwargs.get('label')
        n_datasets = max(ncx, ncy)
        if n_datasets > 1 and not cbook.is_scalar_or_string(label):
            if len(label) != n_datasets:
                raise ValueError(f"label must be scalar or have the same "
                                 f"length as the input data, but found "
                                 f"{len(label)} for {n_datasets} datasets.")
            labels = label
        else:
            labels = [label] * n_datasets

        result = (make_artist(x[:, j % ncx], y[:, j % ncy], kw,
                              {**kwargs, 'label': label})
                  for j, label in enumerate(labels))

        if return_kwargs:
            return list(result)
        else:
            return [l[0] for l in result]


@_api.define_aliases({"facecolor": ["fc"]})
class _AxesBase(martist.Artist):
```
### 4 - tutorials/intermediate/arranging_axes.py:

Start line: 345, End line: 413

```python
for a in range(2):
    for b in range(2):
        ax = fig.add_subplot(gs00[a, b])
        annotate_axes(ax, f'axLeft[{a}, {b}]', fontsize=10)
        if a == 1 and b == 1:
            ax.set_xlabel('xlabel')
for a in range(3):
    ax = fig.add_subplot(gs01[a])
    annotate_axes(ax, f'axRight[{a}, {b}]')
    if a == 2:
        ax.set_ylabel('ylabel')

fig.suptitle('nested gridspecs')

###############################################################################
# Here's a more sophisticated example of nested *GridSpec*: We create an outer
# 4x4 grid with each cell containing an inner 3x3 grid of Axes. We outline
# the outer 4x4 grid by hiding appropriate spines in each of the inner 3x3
# grids.


def squiggle_xy(a, b, c, d, i=np.arange(0.0, 2*np.pi, 0.05)):
    return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)

fig = plt.figure(figsize=(8, 8), constrained_layout=False)
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)

for a in range(4):
    for b in range(4):
        # gridspec inside gridspec
        inner_grid = outer_grid[a, b].subgridspec(3, 3, wspace=0, hspace=0)
        axs = inner_grid.subplots()  # Create all subplots for the inner grid.
        for (c, d), ax in np.ndenumerate(axs):
            ax.plot(*squiggle_xy(a + 1, b + 1, c + 1, d + 1))
            ax.set(xticks=[], yticks=[])

# show only the outside spines
for ax in fig.get_axes():
    ss = ax.get_subplotspec()
    ax.spines.top.set_visible(ss.is_first_row())
    ax.spines.bottom.set_visible(ss.is_last_row())
    ax.spines.left.set_visible(ss.is_first_col())
    ax.spines.right.set_visible(ss.is_last_col())

plt.show()

#############################################################################
#
# More reading
# ============
#
#  - More details about :doc:`subplot mosaic </tutorials/provisional/mosaic>`.
#  - More details about :doc:`constrained layout
#    </tutorials/intermediate/constrainedlayout_guide>`, used to align
#    spacing in most of these examples.
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.figure.Figure.add_gridspec`
#    - `matplotlib.figure.Figure.add_subplot`
#    - `matplotlib.gridspec.GridSpec`
#    - `matplotlib.gridspec.SubplotSpec.subgridspec`
#    - `matplotlib.gridspec.GridSpecFromSubplotSpec`
```
### 5 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 442, End line: 530

```python
fig.colorbar(pcm, ax=axs_right)
fig.suptitle('Nested plots using subfigures')

###############################################################################
# Manually setting axes positions
# ================================
#
# There can be good reasons to manually set an Axes position.  A manual call
# to `~.axes.Axes.set_position` will set the axes so constrained_layout has
# no effect on it anymore. (Note that ``constrained_layout`` still leaves the
# space for the axes that is moved).

fig, axs = plt.subplots(1, 2, layout="constrained")
example_plot(axs[0], fontsize=12)
axs[1].set_position([0.2, 0.2, 0.4, 0.4])

###############################################################################
# .. _compressed_layout:
#
# Grids of fixed aspect-ratio Axes: "compressed" layout
# =====================================================
#
# ``constrained_layout`` operates on the grid of "original" positions for
# axes. However, when Axes have fixed aspect ratios, one side is usually made
# shorter, and leaves large gaps in the shortened direction. In the following,
# the Axes are square, but the figure quite wide so there is a horizontal gap:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout="constrained")
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='constrained'")

###############################################################################
# One obvious way of fixing this is to make the figure size more square,
# however, closing the gaps exactly requires trial and error.  For simple grids
# of Axes we can use ``layout="compressed"`` to do the job for us:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout='compressed')
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='compressed'")


###############################################################################
# Manually turning off ``constrained_layout``
# ===========================================
#
# ``constrained_layout`` usually adjusts the axes positions on each draw
# of the figure.  If you want to get the spacing provided by
# ``constrained_layout`` but not have it update, then do the initial
# draw and then call ``fig.set_layout_engine(None)``.
# This is potentially useful for animations where the tick labels may
# change length.
#
# Note that ``constrained_layout`` is turned off for ``ZOOM`` and ``PAN``
# GUI events for the backends that use the toolbar.  This prevents the
# axes from changing position during zooming and panning.
#
#
# Limitations
# ===========
#
# Incompatible functions
# ----------------------
#
# ``constrained_layout`` will work with `.pyplot.subplot`, but only if the
# number of rows and columns is the same for each call.
# The reason is that each call to `.pyplot.subplot` will create a new
# `.GridSpec` instance if the geometry is not the same, and
# ``constrained_layout``.  So the following works fine:

fig = plt.figure(layout="constrained")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
# third axes that spans both rows in second column:
ax3 = plt.subplot(2, 2, (2, 4))

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.suptitle('Homogenous nrows, ncols')

###############################################################################
# but the following leads to a poor layout:

fig = plt.figure(layout="constrained")
```
### 6 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 636, End line: 721

```python
fig, ax = plt.subplots(layout="constrained")
example_plot(ax, fontsize=24)
plot_children(fig)

#######################################################################
# Simple case: two Axes
# ---------------------
# When there are multiple axes they have their layouts bound in
# simple ways.  In this example the left axes has much larger decorations
# than the right, but they share a bottom margin, which is made large
# enough to accommodate the larger xlabel.   Same with the shared top
# margin.  The left and right margins are not shared, and hence are
# allowed to be different.

fig, ax = plt.subplots(1, 2, layout="constrained")
example_plot(ax[0], fontsize=32)
example_plot(ax[1], fontsize=8)
plot_children(fig, printit=False)

#######################################################################
# Two Axes and colorbar
# ---------------------
#
# A colorbar is simply another item that expands the margin of the parent
# layoutgrid cell:

fig, ax = plt.subplots(1, 2, layout="constrained")
im = ax[0].pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=ax[0], shrink=0.6)
im = ax[1].pcolormesh(arr, **pc_kwargs)
plot_children(fig)

#######################################################################
# Colorbar associated with a Gridspec
# -----------------------------------
#
# If a colorbar belongs to more than one cell of the grid, then
# it makes a larger margin for each:

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)
plot_children(fig, printit=False)

#######################################################################
# Uneven sized Axes
# -----------------
#
# There are two ways to make axes have an uneven size in a
# Gridspec layout, either by specifying them to cross Gridspecs rows
# or columns, or by specifying width and height ratios.
#
# The first method is used here.  Note that the middle ``top`` and
# ``bottom`` margins are not affected by the left-hand column.  This
# is a conscious decision of the algorithm, and leads to the case where
# the two right-hand axes have the same height, but it is not 1/2 the height
# of the left-hand axes.  This is consistent with how ``gridspec`` works
# without constrained layout.

fig = plt.figure(layout="constrained")
gs = gridspec.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[:, 0])
im = ax.pcolormesh(arr, **pc_kwargs)
ax = fig.add_subplot(gs[0, 1])
im = ax.pcolormesh(arr, **pc_kwargs)
ax = fig.add_subplot(gs[1, 1])
im = ax.pcolormesh(arr, **pc_kwargs)
plot_children(fig, printit=False)

#######################################################################
# One case that requires finessing is if margins do not have any artists
# constraining their width. In the case below, the right margin for column 0
# and the left margin for column 3 have no margin artists to set their width,
# so we take the maximum width of the margin widths that do have artists.
# This makes all the axes have the same size:

fig = plt.figure(layout="constrained")
gs = fig.add_gridspec(2, 4)
ax00 = fig.add_subplot(gs[0, 0:2])
ax01 = fig.add_subplot(gs[0, 2:])
ax10 = fig.add_subplot(gs[1, 1:3])
example_plot(ax10, fontsize=14)
plot_children(fig)
plt.show()
```
### 7 - lib/matplotlib/axes/_base.py:

Start line: 241, End line: 313

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
        ambiguous_fmt_datakey = data is not None and len(args) == 2

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
            yield from self._plot_args(
                this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
```
### 8 - lib/matplotlib/pyplot.py:

Start line: 1258, End line: 1322

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
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() == key:
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
### 9 - tutorials/colors/colormaps.py:

Start line: 203, End line: 260

```python
plot_color_gradients('Qualitative',
                     ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c'])

###############################################################################
# Miscellaneous
# -------------
#
# Some of the miscellaneous colormaps have particular uses for which
# they have been created. For example, gist_earth, ocean, and terrain
# all seem to be created for plotting topography (green/brown) and water
# depths (blue) together. We would expect to see a divergence in these
# colormaps, then, but multiple kinks may not be ideal, such as in
# gist_earth and terrain. CMRmap was created to convert well to
# grayscale, though it does appear to have some small kinks in
# :math:`L^*`.  cubehelix was created to vary smoothly in both lightness
# and hue, but appears to have a small hump in the green hue area. turbo
# was created to display depth and disparity data.
#
# The often-used jet colormap is included in this set of colormaps. We can see
# that the :math:`L^*` values vary widely throughout the colormap, making it a
# poor choice for representing data for viewers to see perceptually. See an
# extension on this idea at [mycarta-jet]_ and [turbo]_.


plot_color_gradients('Miscellaneous',
                     ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                      'turbo', 'nipy_spectral', 'gist_ncar'])

plt.show()

###############################################################################
# Lightness of Matplotlib colormaps
# =================================
#
# Here we examine the lightness values of the matplotlib colormaps.
# Note that some documentation on the colormaps is available
# ([list-colormaps]_).

mpl.rcParams.update({'font.size': 12})

# Number of colormap per subplot for particular cmap categories
_DSUBS = {'Perceptually Uniform Sequential': 5, 'Sequential': 6,
          'Sequential (2)': 6, 'Diverging': 6, 'Cyclic': 3,
          'Qualitative': 4, 'Miscellaneous': 6}

# Spacing between the colormaps of a subplot
_DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
       'Sequential (2)': 1.4, 'Diverging': 1.4, 'Cyclic': 1.4,
       'Qualitative': 1.4, 'Miscellaneous': 1.4}

# Indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
```
### 10 - examples/images_contours_and_fields/pcolormesh_grids.py:

Start line: 82, End line: 129

```python
ax.pcolormesh(x, y, Z, shading='nearest', vmin=Z.min(), vmax=Z.max())
_annotate(ax, x, y, "shading='nearest'")

###############################################################################
# Auto Shading
# ------------
#
# It's possible that the user would like the code to automatically choose which
# to use, in this case ``shading='auto'`` will decide whether to use 'flat' or
# 'nearest' shading based on the shapes of *X*, *Y* and *Z*.

fig, axs = plt.subplots(2, 1, constrained_layout=True)
ax = axs[0]
x = np.arange(ncols)
y = np.arange(nrows)
ax.pcolormesh(x, y, Z, shading='auto', vmin=Z.min(), vmax=Z.max())
_annotate(ax, x, y, "shading='auto'; X, Y, Z: same shape (nearest)")

ax = axs[1]
x = np.arange(ncols + 1)
y = np.arange(nrows + 1)
ax.pcolormesh(x, y, Z, shading='auto', vmin=Z.min(), vmax=Z.max())
_annotate(ax, x, y, "shading='auto'; X, Y one larger than Z (flat)")

###############################################################################
# Gouraud Shading
# ---------------
#
# `Gouraud shading <https://en.wikipedia.org/wiki/Gouraud_shading>`_ can also
# be specified, where the color in the quadrilaterals is linearly interpolated
# between the grid points.  The shapes of *X*, *Y*, *Z* must be the same.

fig, ax = plt.subplots(constrained_layout=True)
x = np.arange(ncols)
y = np.arange(nrows)
ax.pcolormesh(x, y, Z, shading='gouraud', vmin=Z.min(), vmax=Z.max())
_annotate(ax, x, y, "shading='gouraud'; X, Y same shape as Z")

plt.show()
#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
```
