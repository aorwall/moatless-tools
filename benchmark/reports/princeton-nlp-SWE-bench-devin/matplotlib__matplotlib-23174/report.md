# matplotlib__matplotlib-23174

| **matplotlib/matplotlib** | `d73ba9e00eddae34610bf9982876b5aa62114ad5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 15 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -2060,6 +2060,14 @@ def dpi(self):
     def dpi(self, value):
         self._parent.dpi = value
 
+    @property
+    def _cachedRenderer(self):
+        return self._parent._cachedRenderer
+
+    @_cachedRenderer.setter
+    def _cachedRenderer(self, renderer):
+        self._parent._cachedRenderer = renderer
+
     def _redo_transform_rel_fig(self, bbox=None):
         """
         Make the transSubfigure bbox relative to Figure transform.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 2063 | 2063 | - | 15 | -


## Problem Statement

```
[Bug]: Crash when adding clabels to subfigures
### Bug summary

Adding a clabel to a contour plot of a subfigure results in a traceback.

### Code for reproduction

\`\`\`python
# Taken from the Contour Demo example
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-(X**2) - Y**2)
Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
Z = (Z1 - Z2) * 2

fig = plt.figure()
figs = fig.subfigures(nrows=1, ncols=2)

for f in figs:
    ax = f.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title("Simplest default with labels")

plt.show()
\`\`\`


### Actual outcome

\`\`\`
    ax.clabel(CS, inline=True, fontsize=10)
  File "/usr/lib/python3.9/site-packages/matplotlib/axes/_axes.py", line 6335, in clabel
    return CS.clabel(levels, **kwargs)
  File "/usr/lib/python3.9/site-packages/matplotlib/contour.py", line 235, in clabel
    self.labels(inline, inline_spacing)
  File "/usr/lib/python3.9/site-packages/matplotlib/contour.py", line 582, in labels
    lw = self._get_nth_label_width(idx)
  File "/usr/lib/python3.9/site-packages/matplotlib/contour.py", line 285, in _get_nth_label_width
    .get_window_extent(mpl.tight_layout.get_renderer(fig)).width)
  File "/usr/lib/python3.9/site-packages/matplotlib/tight_layout.py", line 206, in get_renderer
    if fig._cachedRenderer:
AttributeError: 'SubFigure' object has no attribute '_cachedRenderer'
\`\`\`

### Expected outcome

The two subfigures appearing side by side, each showing the Contour Demo example

### Additional information

_No response_

### Operating system

Gentoo

### Matplotlib Version

3.5.2

### Matplotlib Backend

QtAgg

### Python version

3.9.13

### Jupyter version

_No response_

### Installation

Linux package manager

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/images_contours_and_fields/contour_label_demo.py | 1 | 87| 613 | 613 | 613 | 
| 2 | 2 lib/matplotlib/contour.py | 167 | 238| 712 | 1325 | 17294 | 
| 3 | 3 examples/images_contours_and_fields/contour_demo.py | 1 | 83| 749 | 2074 | 18384 | 
| 4 | 4 lib/matplotlib/axes/_axes.py | 6229 | 6818| 1191 | 3265 | 92095 | 
| 5 | 5 examples/images_contours_and_fields/contourf_demo.py | 96 | 129| 338 | 3603 | 93212 | 
| 6 | 6 examples/text_labels_and_annotations/label_subplots.py | 1 | 71| 603 | 4206 | 93815 | 
| 7 | 6 examples/images_contours_and_fields/contour_demo.py | 84 | 122| 341 | 4547 | 93815 | 
| 8 | 7 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 5303 | 95238 | 
| 9 | 7 examples/images_contours_and_fields/contourf_demo.py | 1 | 95| 779 | 6082 | 95238 | 
| 10 | 7 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 6749 | 95238 | 
| 11 | 7 lib/matplotlib/contour.py | 47 | 73| 309 | 7058 | 95238 | 
| 12 | 8 examples/subplots_axes_and_figures/figure_title.py | 1 | 65| 544 | 7602 | 95782 | 
| 13 | 9 lib/matplotlib/pyplot.py | 2466 | 2477| 146 | 7748 | 122596 | 
| 14 | 9 lib/matplotlib/contour.py | 418 | 438| 235 | 7983 | 122596 | 
| 15 | 10 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 8768 | 124563 | 
| 16 | 10 lib/matplotlib/contour.py | 440 | 457| 181 | 8949 | 124563 | 
| 17 | 10 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 441 | 9390 | 124563 | 
| 18 | 11 examples/pyplots/align_ylabels.py | 40 | 86| 346 | 9736 | 125197 | 
| 19 | 12 examples/images_contours_and_fields/contour_image.py | 1 | 76| 763 | 10499 | 126274 | 
| 20 | 13 examples/images_contours_and_fields/tricontour_demo.py | 139 | 161| 190 | 10689 | 128794 | 
| 21 | 14 examples/showcase/anatomy.py | 84 | 121| 453 | 11142 | 130034 | 
| 22 | 14 examples/images_contours_and_fields/tricontour_demo.py | 1 | 87| 672 | 11814 | 130034 | 
| 23 | **15 lib/matplotlib/figure.py** | 389 | 396| 129 | 11943 | 157025 | 
| 24 | 16 tutorials/intermediate/arranging_axes.py | 100 | 185| 881 | 12824 | 160677 | 
| 25 | 16 lib/matplotlib/contour.py | 459 | 468| 130 | 12954 | 160677 | 
| 26 | 16 examples/images_contours_and_fields/contour_image.py | 77 | 109| 314 | 13268 | 160677 | 
| 27 | 17 examples/text_labels_and_annotations/figlegend_demo.py | 1 | 31| 254 | 13522 | 160931 | 
| 28 | 18 examples/text_labels_and_annotations/line_with_text.py | 51 | 87| 260 | 13782 | 161523 | 
| 29 | 18 lib/matplotlib/contour.py | 76 | 165| 821 | 14603 | 161523 | 
| 30 | 18 lib/matplotlib/contour.py | 536 | 595| 475 | 15078 | 161523 | 
| 31 | 19 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 15381 | 161826 | 
| 32 | 20 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 115| 756 | 16137 | 162705 | 
| 33 | **20 lib/matplotlib/figure.py** | 2133 | 2155| 157 | 16294 | 162705 | 
| 34 | 20 examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 741 | 17035 | 162705 | 
| 35 | 21 examples/subplots_axes_and_figures/align_labels_demo.py | 1 | 38| 294 | 17329 | 162999 | 
| 36 | **21 lib/matplotlib/figure.py** | 398 | 406| 137 | 17466 | 162999 | 
| 37 | 22 examples/images_contours_and_fields/pcolormesh_grids.py | 82 | 129| 484 | 17950 | 164304 | 
| 38 | 22 lib/matplotlib/pyplot.py | 2493 | 2510| 196 | 18146 | 164304 | 
| 39 | 22 lib/matplotlib/pyplot.py | 2272 | 2347| 767 | 18913 | 164304 | 
| 40 | 22 tutorials/intermediate/arranging_axes.py | 187 | 260| 754 | 19667 | 164304 | 
| 41 | 23 examples/images_contours_and_fields/contourf_hatching.py | 1 | 56| 476 | 20143 | 164780 | 
| 42 | 23 lib/matplotlib/contour.py | 386 | 416| 385 | 20528 | 164780 | 
| 43 | **23 lib/matplotlib/figure.py** | 380 | 387| 132 | 20660 | 164780 | 
| 44 | 24 examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 430 | 21090 | 165210 | 
| 45 | 24 tutorials/intermediate/arranging_axes.py | 262 | 337| 755 | 21845 | 165210 | 
| 46 | 24 examples/showcase/anatomy.py | 1 | 55| 510 | 22355 | 165210 | 
| 47 | 25 tutorials/introductory/quick_start.py | 475 | 561| 1039 | 23394 | 171151 | 
| 48 | 26 examples/subplots_axes_and_figures/custom_figure_class.py | 35 | 52| 115 | 23509 | 171522 | 
| 49 | 27 examples/lines_bars_and_markers/bar_label_demo.py | 1 | 107| 820 | 24329 | 172342 | 
| 50 | 27 examples/subplots_axes_and_figures/demo_tight_layout.py | 116 | 135| 123 | 24452 | 172342 | 
| 51 | 28 examples/images_contours_and_fields/irregulardatagrid.py | 79 | 95| 144 | 24596 | 173250 | 
| 52 | 29 examples/images_contours_and_fields/contourf_log.py | 1 | 62| 492 | 25088 | 173742 | 
| 53 | 29 lib/matplotlib/pyplot.py | 1258 | 1322| 686 | 25774 | 173742 | 
| 54 | 29 lib/matplotlib/pyplot.py | 2869 | 2888| 226 | 26000 | 173742 | 
| 55 | 30 examples/shapes_and_collections/fancybox_demo.py | 80 | 127| 501 | 26501 | 175010 | 
| 56 | 31 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 27026 | 175535 | 
| 57 | 32 examples/text_labels_and_annotations/legend_demo.py | 76 | 116| 567 | 27593 | 177442 | 
| 58 | 33 examples/lines_bars_and_markers/filled_step.py | 179 | 237| 493 | 28086 | 179087 | 
| 59 | 33 lib/matplotlib/pyplot.py | 2950 | 2986| 401 | 28487 | 179087 | 
| 60 | 34 examples/subplots_axes_and_figures/subplot.py | 1 | 52| 343 | 28830 | 179430 | 
| 61 | 35 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 29532 | 180541 | 
| 62 | 35 examples/text_labels_and_annotations/legend_demo.py | 164 | 183| 198 | 29730 | 180541 | 
| 63 | 35 lib/matplotlib/pyplot.py | 782 | 834| 572 | 30302 | 180541 | 
| 64 | **35 lib/matplotlib/figure.py** | 1900 | 1935| 377 | 30679 | 180541 | 
| 65 | 36 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 878 | 31557 | 182692 | 
| 66 | 36 lib/matplotlib/contour.py | 1 | 44| 290 | 31847 | 182692 | 
| 67 | 37 examples/misc/contour_manual.py | 1 | 58| 621 | 32468 | 183313 | 
| 68 | 38 tutorials/intermediate/constrainedlayout_guide.py | 187 | 260| 763 | 33231 | 189333 | 
| 69 | **38 lib/matplotlib/figure.py** | 1 | 51| 335 | 33566 | 189333 | 
| 70 | 39 examples/mplot3d/contour3d_2.py | 1 | 19| 130 | 33696 | 189463 | 
| 71 | 39 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 34191 | 189463 | 
| 72 | 39 lib/matplotlib/pyplot.py | 1 | 88| 665 | 34856 | 189463 | 
| 73 | 39 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 35637 | 189463 | 
| 74 | 40 lib/matplotlib/axes/_subplots.py | 62 | 79| 205 | 35842 | 190610 | 
| 75 | 41 examples/images_contours_and_fields/tricontour_smooth_user.py | 27 | 89| 561 | 36403 | 191448 | 
| 76 | 41 lib/matplotlib/contour.py | 847 | 903| 644 | 37047 | 191448 | 
| 77 | 42 examples/images_contours_and_fields/pcolormesh_levels.py | 84 | 133| 448 | 37495 | 192670 | 
| 78 | 42 lib/matplotlib/axes/_subplots.py | 81 | 98| 212 | 37707 | 192670 | 
| 79 | 43 examples/images_contours_and_fields/quadmesh_demo.py | 1 | 51| 450 | 38157 | 193120 | 
| 80 | **43 lib/matplotlib/figure.py** | 1425 | 1477| 532 | 38689 | 193120 | 
| 81 | 43 examples/text_labels_and_annotations/legend_demo.py | 1 | 75| 738 | 39427 | 193120 | 
| 82 | 44 examples/images_contours_and_fields/image_demo.py | 123 | 185| 545 | 39972 | 194704 | 
| 83 | 45 examples/axes_grid1/demo_axes_grid2.py | 1 | 99| 783 | 40755 | 195487 | 
| 84 | 46 lib/matplotlib/backends/backend_gtk3agg.py | 56 | 87| 300 | 41055 | 196181 | 
| 85 | 46 examples/showcase/anatomy.py | 58 | 81| 277 | 41332 | 196181 | 
| 86 | 47 examples/subplots_axes_and_figures/axes_demo.py | 1 | 46| 423 | 41755 | 196604 | 
| 87 | 48 examples/specialty_plots/leftventricle_bulleye.py | 130 | 193| 708 | 42463 | 198771 | 


### Hint

```
Not sure if one should add `self._cachedRenderer = None` to `FigureBase` (and remove in `Figure`) or to `SubFigure` init-functions, but that should fix it.

I thought it was a recent regression, but it doesn't look like it, so maybe should be labelled 3.6.0 instead?
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -2060,6 +2060,14 @@ def dpi(self):
     def dpi(self, value):
         self._parent.dpi = value
 
+    @property
+    def _cachedRenderer(self):
+        return self._parent._cachedRenderer
+
+    @_cachedRenderer.setter
+    def _cachedRenderer(self, renderer):
+        self._parent._cachedRenderer = renderer
+
     def _redo_transform_rel_fig(self, bbox=None):
         """
         Make the transSubfigure bbox relative to Figure transform.

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_contour.py b/lib/matplotlib/tests/test_contour.py
--- a/lib/matplotlib/tests/test_contour.py
+++ b/lib/matplotlib/tests/test_contour.py
@@ -585,3 +585,23 @@ def test_all_algorithms():
         ax.contourf(x, y, z, algorithm=algorithm)
         ax.contour(x, y, z, algorithm=algorithm, colors='k')
         ax.set_title(algorithm)
+
+
+def test_subfigure_clabel():
+    # Smoke test for gh#23173
+    delta = 0.025
+    x = np.arange(-3.0, 3.0, delta)
+    y = np.arange(-2.0, 2.0, delta)
+    X, Y = np.meshgrid(x, y)
+    Z1 = np.exp(-(X**2) - Y**2)
+    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
+    Z = (Z1 - Z2) * 2
+
+    fig = plt.figure()
+    figs = fig.subfigures(nrows=1, ncols=2)
+
+    for f in figs:
+        ax = f.subplots()
+        CS = ax.contour(X, Y, Z)
+        ax.clabel(CS, inline=True, fontsize=10)
+        ax.set_title("Simplest default with labels")

```


## Code snippets

### 1 - examples/images_contours_and_fields/contour_label_demo.py:

Start line: 1, End line: 87

```python
"""
==================
Contour Label Demo
==================

Illustrate some of the more advanced things that one can do with
contour labels.

See also the :doc:`contour demo example
</gallery/images_contours_and_fields/contour_demo>`.
"""

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

###############################################################################
# Define our surface

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

###############################################################################
# Make contour labels with custom level formatters


# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a percent sign.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# Basic contour plot
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)

ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

###############################################################################
# Label contours with arbitrary strings using a dictionary

fig1, ax1 = plt.subplots()

# Basic contour plot
CS1 = ax1.contour(X, Y, Z)

fmt = {}
strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s

# Label every other level using strings
ax1.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=10)

###############################################################################
# Use a Formatter

fig2, ax2 = plt.subplots()

CS2 = ax2.contour(X, Y, 100**Z, locator=plt.LogLocator())
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
ax2.clabel(CS2, CS2.levels, fmt=fmt)
ax2.set_title("$100^Z$")

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.ticker.LogFormatterMathtext`
#    - `matplotlib.ticker.TickHelper.create_dummy_axis`
```
### 2 - lib/matplotlib/contour.py:

Start line: 167, End line: 238

```python
class ContourLabeler:

    def clabel(self, levels=None, *,
               fontsize=None, inline=True, inline_spacing=5, fmt=None,
               colors=None, use_clabeltext=False, manual=False,
               rightside_up=True, zorder=None):

        # clabel basically takes the input arguments and uses them to
        # add a list of "label specific" attributes to the ContourSet
        # object.  These attributes are all of the form label* and names
        # should be fairly self explanatory.
        #
        # Once these attributes are set, clabel passes control to the
        # labels method (case of automatic label placement) or
        # `BlockingContourLabeler` (case of manual label placement).

        if fmt is None:
            fmt = ticker.ScalarFormatter(useOffset=False)
            fmt.create_dummy_axis()
        self.labelFmt = fmt
        self._use_clabeltext = use_clabeltext
        # Detect if manual selection is desired and remove from argument list.
        self.labelManual = manual
        self.rightside_up = rightside_up
        if zorder is None:
            self._clabel_zorder = 2+self._contour_zorder
        else:
            self._clabel_zorder = zorder

        if levels is None:
            levels = self.levels
            indices = list(range(len(self.cvalues)))
        else:
            levlabs = list(levels)
            indices, levels = [], []
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                raise ValueError(f"Specified levels {levlabs} don't match "
                                 f"available levels {self.levels}")
        self.labelLevelList = levels
        self.labelIndiceList = indices

        self.labelFontProps = font_manager.FontProperties()
        self.labelFontProps.set_size(fontsize)
        font_size_pts = self.labelFontProps.get_size_in_points()
        self.labelFontSizeList = [font_size_pts] * len(levels)

        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap,
                                                   norm=mcolors.NoNorm())

        self.labelXYs = []

        if np.iterable(self.labelManual):
            for x, y in self.labelManual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif self.labelManual:
            print('Select label locations manually using first mouse button.')
            print('End manual selection with second mouse button.')
            if not inline:
                print('Remove last label by clicking third mouse button.')
            mpl._blocking_input.blocking_input_loop(
                self.axes.figure, ["button_press_event", "key_press_event"],
                timeout=-1, handler=functools.partial(
                    _contour_labeler_event_handler,
                    self, inline, inline_spacing))
        else:
            self.labels(inline, inline_spacing)

        self.labelTextsList = cbook.silent_list('text.Text', self.labelTexts)
        return self.labelTextsList
```
### 3 - examples/images_contours_and_fields/contour_demo.py:

Start line: 1, End line: 83

```python
"""
============
Contour Demo
============

Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also the :doc:`contour image example
</gallery/images_contours_and_fields/contour_image>`.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

###############################################################################
# Create a simple contour plot with labels using default colors.  The inline
# argument to clabel will control whether the labels are draw over the line
# segments of the contour, removing the lines beneath the label.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')

###############################################################################
# Contour labels can be placed manually by providing list of positions (in data
# coordinate).  See :doc:`/gallery/event_handling/ginput_manual_clabel_sgskip`
# for interactive placement.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
manual_locations = [
    (-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
ax.clabel(CS, inline=True, fontsize=10, manual=manual_locations)
ax.set_title('labels at selected locations')

###############################################################################
# You can force all the contours to be the same color.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')  # Negative contours default to dashed.
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours dashed')

###############################################################################
# You can set negative contours to be solid instead of dashed:

plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')  # Negative contours default to dashed.
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours solid')

###############################################################################
# And you can manually specify the colors of the contour

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6,
                linewidths=np.arange(.5, 4, .5),
                colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'),
                )
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Crazy lines')

###############################################################################
# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.gray, extent=(-3, 3, -2, 2))
levels = np.arange(-1.2, 1.6, 0.2)
```
### 4 - lib/matplotlib/axes/_axes.py:

Start line: 6229, End line: 6818

```python
@_docstring.interpd
class Axes(_AxesBase):

    @_preprocess_data()
    @_docstring.dedent_interpd
    def pcolorfast(self, *args, alpha=None, norm=None, cmap=None, vmin=None,
                   vmax=None, **kwargs):

        C = args[-1]
        nr, nc = np.shape(C)[:2]
        if len(args) == 1:
            style = "image"
            x = [0, nc]
            y = [0, nr]
        elif len(args) == 3:
            x, y = args[:2]
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 1 and y.ndim == 1:
                if x.size == 2 and y.size == 2:
                    style = "image"
                else:
                    dx = np.diff(x)
                    dy = np.diff(y)
                    if (np.ptp(dx) < 0.01 * abs(dx.mean()) and
                            np.ptp(dy) < 0.01 * abs(dy.mean())):
                        style = "image"
                    else:
                        style = "pcolorimage"
            elif x.ndim == 2 and y.ndim == 2:
                style = "quadmesh"
            else:
                raise TypeError("arguments do not match valid signatures")
        else:
            raise TypeError("need 1 argument or 3 arguments")

        if style == "quadmesh":
            # data point in each cell is value at lower left corner
            coords = np.stack([x, y], axis=-1)
            if np.ndim(C) == 2:
                qm_kwargs = {"array": np.ma.ravel(C)}
            elif np.ndim(C) == 3:
                qm_kwargs = {"color": np.ma.reshape(C, (-1, C.shape[-1]))}
            else:
                raise ValueError("C must be 2D or 3D")
            collection = mcoll.QuadMesh(
                coords, **qm_kwargs,
                alpha=alpha, cmap=cmap, norm=norm,
                antialiased=False, edgecolors="none")
            self.add_collection(collection, autolim=False)
            xl, xr, yb, yt = x.min(), x.max(), y.min(), y.max()
            ret = collection

        else:  # It's one of the two image styles.
            extent = xl, xr, yb, yt = x[0], x[-1], y[0], y[-1]
            if style == "image":
                im = mimage.AxesImage(
                    self, cmap, norm,
                    data=C, alpha=alpha, extent=extent,
                    interpolation='nearest', origin='lower',
                    **kwargs)
            elif style == "pcolorimage":
                im = mimage.PcolorImage(
                    self, x, y, C,
                    cmap=cmap, norm=norm, alpha=alpha, extent=extent,
                    **kwargs)
            self.add_image(im)
            ret = im

        if np.ndim(C) == 2:  # C.ndim == 3 is RGB(A) so doesn't need scaling.
            ret._scale_norm(norm, vmin, vmax)

        if ret.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            ret.set_clip_path(self.patch)

        ret.sticky_edges.x[:] = [xl, xr]
        ret.sticky_edges.y[:] = [yb, yt]
        self.update_datalim(np.array([[xl, yb], [xr, yt]]))
        self._request_autoscale_view(tight=True)
        return ret

    @_preprocess_data()
    @_docstring.dedent_interpd
    def contour(self, *args, **kwargs):
        """
        Plot contour lines.

        Call signature::

            contour([X, Y,] Z, [levels], **kwargs)
        %(contour_doc)s
        """
        kwargs['filled'] = False
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    @_preprocess_data()
    @_docstring.dedent_interpd
    def contourf(self, *args, **kwargs):
        """
        Plot filled contours.

        Call signature::

            contourf([X, Y,] Z, [levels], **kwargs)
        %(contour_doc)s
        """
        kwargs['filled'] = True
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    def clabel(self, CS, levels=None, **kwargs):
        """
        Label a contour plot.

        Adds labels to line contours in given `.ContourSet`.

        Parameters
        ----------
        CS : `.ContourSet` instance
            Line contours to label.

        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``CS.levels``. If not given, all levels are labeled.

        **kwargs
            All other parameters are documented in `~.ContourLabeler.clabel`.
        """
        return CS.clabel(levels, **kwargs)

    #### Data analysis

    @_preprocess_data(replace_names=["x", 'weights'], label_namer="x")
    def hist(self, x, bins=None, range=None, density=False, weights=None,
             cumulative=False, bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False,
             color=None, label=None, stacked=False, **kwargs):
        # ... other code
```
### 5 - examples/images_contours_and_fields/contourf_demo.py:

Start line: 96, End line: 129

```python
extends = ["neither", "both", "min", "max"]
cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")
# Note: contouring simply excludes masked or nan regions, so
# instead of using the "bad" colormap value for them, it draws
# nothing at all in them.  Therefore the following would have
# no effect:
# cmap.set_bad("red")

fig, axs = plt.subplots(2, 2, constrained_layout=True)

for ax, extend in zip(axs.flat, extends):
    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
    fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.set_title("extend = %s" % extend)
    ax.locator_params(nbins=4)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.set_bad`
#    - `matplotlib.colors.Colormap.set_under`
#    - `matplotlib.colors.Colormap.set_over`
```
### 6 - examples/text_labels_and_annotations/label_subplots.py:

Start line: 1, End line: 71

```python
"""
==================
Labelling subplots
==================

Labelling subplots is relatively straightforward, and varies,
so Matplotlib does not have a general method for doing this.

Simplest is putting the label inside the axes.  Note, here
we use `.pyplot.subplot_mosaic`, and use the subplot labels
as keys for the subplots, which is a nice convenience.  However,
the same method works with `.pyplot.subplots` or keys that are
different than what you want to label the subplot with.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

plt.show()

##############################################################################
# We may prefer the labels outside the axes, but still aligned
# with each other, in which case we use a slightly different transform:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')

plt.show()

##############################################################################
# If we want it aligned with the title, either incorporate in the title or
# use the *loc* keyword argument:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              constrained_layout=True)

for label, ax in axs.items():
    ax.set_title('Normal Title', fontstyle='italic')
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.subplot_mosaic` /
#      `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.text`
#    - `matplotlib.transforms.ScaledTranslation`
```
### 7 - examples/images_contours_and_fields/contour_demo.py:

Start line: 84, End line: 122

```python
CS = ax.contour(Z, levels, origin='lower', cmap='flag', extend='both',
                linewidths=2, extent=(-3, 3, -2, 2))

# Thicken the zero contour.
CS.collections[6].set_linewidth(4)

ax.clabel(CS, levels[1::2],  # label every second level
          inline=True, fmt='%1.1f', fontsize=14)

# make a colorbar for the contour lines
CB = fig.colorbar(CS, shrink=0.8)

ax.set_title('Lines with colorbar')

# We can still add a colorbar for the image, too.
CBI = fig.colorbar(im, orientation='horizontal', shrink=0.8)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

l, b, w, h = ax.get_position().bounds
ll, bb, ww, hh = CB.ax.get_position().bounds
CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.axes.Axes.get_position`
#    - `matplotlib.axes.Axes.set_position`
```
### 8 - examples/subplots_axes_and_figures/subfigures.py:

Start line: 1, End line: 83

```python
"""
=================
Figure subfigures
=================

Sometimes it is desirable to have a figure with two different layouts in it.
This can be achieved with
:doc:`nested gridspecs</gallery/subplots_axes_and_figures/gridspec_nested>`,
but having a virtual figure with its own artists is helpful, so
Matplotlib also has "subfigures", accessed by calling
`matplotlib.figure.Figure.add_subfigure` in a way that is analogous to
`matplotlib.figure.Figure.add_subplot`, or
`matplotlib.figure.Figure.subfigures` to make an array of subfigures.  Note
that subfigures can also have their own child subfigures.

.. note::
    ``subfigure`` is new in v3.4, and the API is still provisional.

"""
import matplotlib.pyplot as plt
import numpy as np


def example_plot(ax, fontsize=12, hide_labels=False):
    pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc

np.random.seed(19680808)
# gridspec inside gridspec
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07)

axsLeft = subfigs[0].subplots(1, 2, sharey=True)
subfigs[0].set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfigs[0].suptitle('Left plots', fontsize='x-large')
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

axsRight = subfigs[1].subplots(3, 1, sharex=True)
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')

subfigs[1].set_facecolor('0.85')
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
subfigs[1].suptitle('Right plots', fontsize='x-large')

fig.suptitle('Figure suptitle', fontsize='xx-large')

plt.show()

##############################################################################
# It is possible to mix subplots and subfigures using
# `matplotlib.figure.Figure.add_subfigure`.  This requires getting
# the gridspec that the subplots are laid out on.

fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 4))
gridspec = axs[0, 0].get_subplotspec().get_gridspec()

# clear the left column for the subfigure:
for a in axs[:, 0]:
    a.remove()

# plot data in remaining axes:
for a in axs[:, 1:].flat:
    a.plot(np.arange(10))

# make the subfigure in the empty gridspec slots:
subfig = fig.add_subfigure(gridspec[:, 0])

axsLeft = subfig.subplots(1, 2, sharey=True)
subfig.set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfig.suptitle('Left plots', fontsize='x-large')
```
### 9 - examples/images_contours_and_fields/contourf_demo.py:

Start line: 1, End line: 95

```python
"""
=============
Contourf Demo
=============

How to use the `.axes.Axes.contourf` method to create filled contour plots.
"""
import numpy as np
import matplotlib.pyplot as plt

origin = 'lower'

delta = 0.025

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

nr, nc = Z.shape

# put NaNs in one corner:
Z[-nr // 6:, -nc // 6:] = np.nan
# contourf will convert these to masked


Z = np.ma.array(Z)
# mask another corner:
Z[:nr // 6, :nc // 6] = np.ma.masked

# mask a circle in the middle:
interior = np.sqrt(X**2 + Y**2) < 0.5
Z[interior] = np.ma.masked

#############################################################################
# Automatic contour levels
# ------------------------
# We are using automatic selection of contour levels; this is usually not such
# a good idea, because they don't occur on nice boundaries, but we do it here
# for purposes of illustration.

fig1, ax2 = plt.subplots(constrained_layout=True)
CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin=origin)

# Note that in the following, we explicitly pass in a subset of the contour
# levels used for the filled contours.  Alternatively, we could pass in
# additional levels to provide extra resolution, or leave out the *levels*
# keyword argument to use all of the original levels.

CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r', origin=origin)

ax2.set_title('Nonsense (3 masked regions)')
ax2.set_xlabel('word length anomaly')
ax2.set_ylabel('sentence length anomaly')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS2)

#############################################################################
# Explicit contour levels
# -----------------------
# Now make a contour plot with the levels specified, and with the colormap
# generated automatically from a list of colors.

fig2, ax2 = plt.subplots(constrained_layout=True)
levels = [-1.5, -1, -0.5, 0, 0.5, 1]
CS3 = ax2.contourf(X, Y, Z, levels,
                   colors=('r', 'g', 'b'),
                   origin=origin,
                   extend='both')
# Our data range extends outside the range of levels; make
# data below the lowest contour level yellow, and above the
# highest level cyan:
CS3.cmap.set_under('yellow')
CS3.cmap.set_over('cyan')

CS4 = ax2.contour(X, Y, Z, levels,
                  colors=('k',),
                  linewidths=(3,),
                  origin=origin)
ax2.set_title('Listed colors (3 masked regions)')
ax2.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)

# Notice that the colorbar gets all the information it
# needs from the ContourSet object, CS3.
fig2.colorbar(CS3)

#############################################################################
# Extension settings
# ------------------
# Illustrate all 4 possible "extend" settings:
```
### 10 - examples/subplots_axes_and_figures/subfigures.py:

Start line: 84, End line: 149

```python
subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

fig.suptitle('Figure suptitle', fontsize='xx-large')
plt.show()

##############################################################################
# Subfigures can have different widths and heights.  This is exactly the
# same example as the first example, but *width_ratios* has been changed:

fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[2, 1])

axsLeft = subfigs[0].subplots(1, 2, sharey=True)
subfigs[0].set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfigs[0].suptitle('Left plots', fontsize='x-large')
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

axsRight = subfigs[1].subplots(3, 1, sharex=True)
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')

subfigs[1].set_facecolor('0.85')
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
subfigs[1].suptitle('Right plots', fontsize='x-large')

fig.suptitle('Figure suptitle', fontsize='xx-large')

plt.show()

##############################################################################
# Subfigures can be also be nested:

fig = plt.figure(constrained_layout=True, figsize=(10, 8))

fig.suptitle('fig')

subfigs = fig.subfigures(1, 2, wspace=0.07)

subfigs[0].set_facecolor('coral')
subfigs[0].suptitle('subfigs[0]')

subfigs[1].set_facecolor('coral')
subfigs[1].suptitle('subfigs[1]')

subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 1.4])
subfigsnest[0].suptitle('subfigsnest[0]')
subfigsnest[0].set_facecolor('r')
axsnest0 = subfigsnest[0].subplots(1, 2, sharey=True)
for nn, ax in enumerate(axsnest0):
    pc = example_plot(ax, hide_labels=True)
subfigsnest[0].colorbar(pc, ax=axsnest0)

subfigsnest[1].suptitle('subfigsnest[1]')
subfigsnest[1].set_facecolor('g')
axsnest1 = subfigsnest[1].subplots(3, 1, sharex=True)

axsRight = subfigs[1].subplots(2, 2)

plt.show()
```
### 23 - lib/matplotlib/figure.py:

Start line: 389, End line: 396

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.5, y0=0.01, name='supxlabel', ha='center',
                             va='bottom')
    @_docstring.copy(_suplabels)
    def supxlabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
                'ha': 'center', 'va': 'bottom', 'rotation': 0}
        return self._suplabels(t, info, **kwargs)
```
### 33 - lib/matplotlib/figure.py:

Start line: 2133, End line: 2155

```python
@_docstring.interpd
class SubFigure(FigureBase):

    get_axes = axes.fget

    def draw(self, renderer):
        # docstring inherited
        self._cachedRenderer = renderer

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = self._get_draw_artists(renderer)

        try:
            renderer.open_group('subfigure', gid=self.get_gid())
            self.patch.draw(renderer)
            mimage._draw_list_compositing_images(
                renderer, self, artists, self.figure.suppressComposite)
            for sfig in self.subfigs:
                sfig.draw(renderer)
            renderer.close_group('subfigure')

        finally:
            self.stale = False
```
### 36 - lib/matplotlib/figure.py:

Start line: 398, End line: 406

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.02, y0=0.5, name='supylabel', ha='left',
                             va='center')
    @_docstring.copy(_suplabels)
    def supylabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supylabel', 'x0': 0.02, 'y0': 0.5,
                'ha': 'left', 'va': 'center', 'rotation': 'vertical',
                'rotation_mode': 'anchor'}
        return self._suplabels(t, info, **kwargs)
```
### 43 - lib/matplotlib/figure.py:

Start line: 380, End line: 387

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.5, y0=0.98, name='suptitle', ha='center',
                             va='top')
    @_docstring.copy(_suplabels)
    def suptitle(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.98,
                'ha': 'center', 'va': 'top', 'rotation': 0}
        return self._suplabels(t, info, **kwargs)
```
### 64 - lib/matplotlib/figure.py:

Start line: 1900, End line: 1935

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       subplot_kw=None, gridspec_kw=None, empty_sentinel='.'):

        def _do_layout(gs, mosaic, unique_ids, nested):
            # ... other code
            for key in sorted(this_level):
                name, arg, method = this_level[key]
                # we are doing some hokey function dispatch here based
                # on the 'method' string stashed above to sort out if this
                # element is an Axes or a nested mosaic.
                if method == 'axes':
                    slc = arg
                    # add a single axes
                    if name in output:
                        raise ValueError(f"There are duplicate keys {name} "
                                         f"in the layout\n{mosaic!r}")
                    ax = self.add_subplot(
                        gs[slc], **{'label': str(name), **subplot_kw}
                    )
                    output[name] = ax
                elif method == 'nested':
                    nested_mosaic = arg
                    j, k = key
                    # recursively add the nested mosaic
                    rows, cols = nested_mosaic.shape
                    nested_output = _do_layout(
                        gs[j, k].subgridspec(rows, cols, **gridspec_kw),
                        nested_mosaic,
                        *_identify_keys_and_nested(nested_mosaic)
                    )
                    overlap = set(output) & set(nested_output)
                    if overlap:
                        raise ValueError(
                            f"There are duplicate keys {overlap} "
                            f"between the outer layout\n{mosaic!r}\n"
                            f"and the nested layout\n{nested_mosaic}"
                        )
                    output.update(nested_output)
                else:
                    raise RuntimeError("This should never happen")
            return output
        # ... other code
```
### 69 - lib/matplotlib/figure.py:

Start line: 1, End line: 51

```python
"""
`matplotlib.figure` implements the following classes:

`Figure`
    Top level `~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in `FigureBase`.

`SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent
    `SubFigure`) with `Figure.add_subfigure` or `Figure.subfigures` methods
    (provisional API v3.4).

`SubplotParams`
    Control the default spacing between subplots.
"""

from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, _docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes, SubplotBase, subplot_class_factory
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
                                      TightLayoutEngine, LayoutEngine)
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)

_log = logging.getLogger(__name__)


def _stale_figure_callback(self, val):
    if self.figure:
        self.figure.stale = val
```
### 80 - lib/matplotlib/figure.py:

Start line: 1425, End line: 1477

```python
class FigureBase(Artist):

    def subfigures(self, nrows=1, ncols=1, squeeze=True,
                   wspace=None, hspace=None,
                   width_ratios=None, height_ratios=None,
                   **kwargs):
        """
        Add a subfigure to this figure or subfigure.

        A subfigure has the same artist methods as a figure, and is logically
        the same as a figure, but cannot print itself.
        See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subfigure grid.

        squeeze : bool, default: True
            If True, extra dimensions are squeezed out from the returned
            array of subfigures.

        wspace, hspace : float, default: None
            The amount of width/height reserved for space between subfigures,
            expressed as a fraction of the average subfigure width/height.
            If not given, the values will be inferred from a figure or
            rcParams when necessary.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self,
                      wspace=wspace, hspace=hspace,
                      width_ratios=width_ratios,
                      height_ratios=height_ratios)

        sfarr = np.empty((nrows, ncols), dtype=object)
        for i in range(ncols):
            for j in range(nrows):
                sfarr[j, i] = self.add_subfigure(gs[j, i], **kwargs)

        if squeeze:
            # Discarding unneeded dimensions that equal 1.  If we only have one
            # subfigure, just return it instead of a 1-element array.
            return sfarr.item() if sfarr.size == 1 else sfarr.squeeze()
        else:
            # Returned axis array will be always 2-d, even if nrows=ncols=1.
            return sfarr
```
