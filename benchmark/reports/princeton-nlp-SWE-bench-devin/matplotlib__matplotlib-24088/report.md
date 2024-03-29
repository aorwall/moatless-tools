# matplotlib__matplotlib-24088

| **matplotlib/matplotlib** | `0517187b9c91061d2ec87e70442615cf4f47b6f3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 323 |
| **Any found context length** | 323 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1253,11 +1253,13 @@ def colorbar(
         # Store the value of gca so that we can set it back later on.
         if cax is None:
             if ax is None:
-                raise ValueError(
+                _api.warn_deprecated("3.6", message=(
                     'Unable to determine Axes to steal space for Colorbar. '
+                    'Using gca(), but will raise in the future. '
                     'Either provide the *cax* argument to use as the Axes for '
                     'the Colorbar, provide the *ax* argument to steal space '
-                    'from it, or add *mappable* to an Axes.')
+                    'from it, or add *mappable* to an Axes.'))
+                ax = self.gca()
             current_ax = self.gca()
             userax = False
             if (use_gridspec and isinstance(ax, SubplotBase)):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 1256 | 1260 | 1 | 1 | 323


## Problem Statement

```
[Bug]: ValueError: Unable to determine Axes to steal space for Colorbar.
### Bug summary

`matplotlib==3.6.0` started raising an error when trying to add a colorbar to `plt.hist()`:

ValueError: Unable to determine Axes to steal space for Colorbar. Either provide the *cax* argument to use as the Axes for the Colorbar, provide the *ax* argument to steal space from it, or add *mappable* to an Axes.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

xs = np.random.rand(100)

_, bin_edges, bars = plt.hist(xs)
color_map = getattr(plt.cm, "hot")
for x_val, rect in zip(bin_edges, bars.patches):
    rect.set_color(color_map(x_val))

cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=color_map),
    # cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),
)
\`\`\`

### Actual outcome

In `matplotlib==3.6.0`:

![mpl==3 6 0](https://user-images.githubusercontent.com/30958850/191547778-033472e7-e739-4beb-a1f4-eecdcb587e22.png)


### Expected outcome

In `matplotlib==3.5.1`:

![mpl==3 5 1](https://user-images.githubusercontent.com/30958850/191547733-cd4911a5-67c8-4070-a708-ce3399e8c0ba.png)

### Operating system

macOS 12.6

### Matplotlib Version

3.6.0

### Python version

3.10

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 lib/matplotlib/figure.py** | 1254 | 1281| 323 | 323 | 28715 | 
| 2 | 2 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 82| 738 | 1061 | 29584 | 
| 3 | 3 lib/matplotlib/pyplot.py | 2043 | 2056| 138 | 1199 | 57595 | 
| 4 | 4 lib/matplotlib/colorbar.py | 264 | 419| 1285 | 2484 | 71602 | 
| 5 | 4 lib/matplotlib/colorbar.py | 1 | 28| 213 | 2697 | 71602 | 
| 6 | 5 tutorials/colors/colorbar_only.py | 88 | 131| 357 | 3054 | 72693 | 
| 7 | 5 examples/subplots_axes_and_figures/colorbar_placement.py | 83 | 97| 131 | 3185 | 72693 | 
| 8 | 6 examples/color/colorbar_basics.py | 1 | 59| 506 | 3691 | 73199 | 
| 9 | 6 lib/matplotlib/colorbar.py | 690 | 712| 268 | 3959 | 73199 | 
| 10 | 6 lib/matplotlib/colorbar.py | 199 | 262| 513 | 4472 | 73199 | 
| 11 | 6 lib/matplotlib/colorbar.py | 608 | 622| 201 | 4673 | 73199 | 
| 12 | 6 lib/matplotlib/colorbar.py | 118 | 144| 254 | 4927 | 73199 | 
| 13 | **6 lib/matplotlib/figure.py** | 1176 | 1253| 748 | 5675 | 73199 | 
| 14 | 6 lib/matplotlib/colorbar.py | 1505 | 1566| 624 | 6299 | 73199 | 
| 15 | 6 lib/matplotlib/colorbar.py | 1356 | 1437| 799 | 7098 | 73199 | 
| 16 | 6 tutorials/colors/colorbar_only.py | 1 | 86| 734 | 7832 | 73199 | 
| 17 | 7 examples/axes_grid1/simple_colorbar.py | 1 | 22| 135 | 7967 | 73334 | 
| 18 | 8 examples/lines_bars_and_markers/filled_step.py | 179 | 237| 493 | 8460 | 74979 | 
| 19 | 9 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 8763 | 75282 | 
| 20 | 10 examples/axes_grid1/demo_colorbar_of_inset_axes.py | 1 | 34| 250 | 9013 | 75532 | 
| 21 | 10 lib/matplotlib/colorbar.py | 1438 | 1455| 169 | 9182 | 75532 | 
| 22 | 11 lib/mpl_toolkits/axes_grid1/axes_grid.py | 20 | 49| 211 | 9393 | 80509 | 
| 23 | 11 lib/matplotlib/pyplot.py | 2475 | 2487| 229 | 9622 | 80509 | 
| 24 | 11 lib/matplotlib/colorbar.py | 1013 | 1048| 263 | 9885 | 80509 | 
| 25 | 11 lib/matplotlib/colorbar.py | 1135 | 1150| 191 | 10076 | 80509 | 
| 26 | 11 lib/matplotlib/pyplot.py | 2415 | 2426| 146 | 10222 | 80509 | 
| 27 | 11 lib/matplotlib/colorbar.py | 584 | 606| 246 | 10468 | 80509 | 
| 28 | 11 lib/matplotlib/colorbar.py | 30 | 115| 961 | 11429 | 80509 | 
| 29 | 11 lib/matplotlib/colorbar.py | 1279 | 1309| 267 | 11696 | 80509 | 
| 30 | 11 lib/matplotlib/colorbar.py | 1311 | 1332| 248 | 11944 | 80509 | 
| 31 | 11 lib/matplotlib/colorbar.py | 1458 | 1503| 398 | 12342 | 80509 | 
| 32 | 12 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 13044 | 81620 | 
| 33 | 12 lib/matplotlib/colorbar.py | 624 | 689| 684 | 13728 | 81620 | 
| 34 | 13 tutorials/introductory/quick_start.py | 475 | 561| 1039 | 14767 | 87561 | 
| 35 | 14 examples/axes_grid1/demo_colorbar_with_inset_locator.py | 1 | 43| 383 | 15150 | 87944 | 
| 36 | 15 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 15645 | 90095 | 
| 37 | 16 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 16426 | 96659 | 
| 38 | 16 lib/matplotlib/colorbar.py | 474 | 508| 310 | 16736 | 96659 | 
| 39 | 16 lib/matplotlib/colorbar.py | 421 | 472| 384 | 17120 | 96659 | 
| 40 | 16 lib/matplotlib/colorbar.py | 147 | 196| 450 | 17570 | 96659 | 
| 41 | 17 examples/images_contours_and_fields/pcolor_demo.py | 1 | 97| 788 | 18358 | 97704 | 
| 42 | 18 examples/lines_bars_and_markers/broken_barh.py | 1 | 27| 247 | 18605 | 97951 | 
| 43 | 19 examples/specialty_plots/leftventricle_bulleye.py | 133 | 196| 708 | 19313 | 100128 | 
| 44 | 20 plot_types/stats/errorbar_plot.py | 1 | 28| 175 | 19488 | 100303 | 
| 45 | 20 lib/matplotlib/pyplot.py | 2556 | 2568| 183 | 19671 | 100303 | 
| 46 | 21 examples/statistics/errorbars_and_boxes.py | 42 | 60| 189 | 19860 | 100949 | 
| 47 | 21 lib/matplotlib/pyplot.py | 2363 | 2387| 252 | 20112 | 100949 | 
| 48 | 21 lib/matplotlib/colorbar.py | 714 | 792| 779 | 20891 | 100949 | 
| 49 | 22 lib/matplotlib/axes/_axes.py | 3493 | 3964| 2054 | 22945 | 174988 | 
| 50 | 23 examples/axes_grid1/demo_colorbar_with_axes_divider.py | 1 | 35| 350 | 23295 | 175338 | 
| 51 | 23 lib/matplotlib/colorbar.py | 1216 | 1252| 414 | 23709 | 175338 | 
| 52 | 23 lib/matplotlib/colorbar.py | 1103 | 1133| 306 | 24015 | 175338 | 
| 53 | 23 lib/matplotlib/colorbar.py | 794 | 850| 497 | 24512 | 175338 | 
| 54 | 23 lib/matplotlib/colorbar.py | 1050 | 1101| 584 | 25096 | 175338 | 
| 55 | 23 lib/matplotlib/axes/_axes.py | 1 | 39| 284 | 25380 | 175338 | 
| 56 | 23 examples/lines_bars_and_markers/filled_step.py | 1 | 77| 477 | 25857 | 175338 | 
| 57 | 23 examples/statistics/errorbars_and_boxes.py | 63 | 81| 117 | 25974 | 175338 | 
| 58 | 23 lib/matplotlib/colorbar.py | 893 | 928| 313 | 26287 | 175338 | 
| 59 | 23 lib/matplotlib/colorbar.py | 930 | 967| 301 | 26588 | 175338 | 
| 60 | 24 examples/images_contours_and_fields/colormap_interactive_adjustment.py | 1 | 34| 306 | 26894 | 175644 | 
| 61 | 25 examples/lines_bars_and_markers/errorbar_limits_simple.py | 1 | 64| 529 | 27423 | 176173 | 
| 62 | 26 lib/matplotlib/_constrained_layout.py | 560 | 577| 156 | 27579 | 183574 | 


### Hint

```
The error disappears in 3.6.0 by following the error message and passing `cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8])`.
If it is ambiguous what axes to use, pass in the axes directly: 

\`\`\`
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=color_map),
    ax=plt.gca()
)
\`\`\`

You _could_ make an axes, and use that, but you will lose some layout convenience. 
Good to know I can keep auto-layout for the color bar.

> If it is ambiguous what axes to use, pass in the axes directly:

What changed between 3.5.1 and 3.6.0? Why wasn't it ambiguous before?
This happened in https://github.com/matplotlib/matplotlib/pull/23740.   However, I think we were under the impression this was deprecated, but it sounds like perhaps that was not the case?
I had the problem when trying to use create a simple SHAP plot using the [shap package](https://shap.readthedocs.io/en/latest/index.html). 

\`\`\`python
import shap

# use SHAP (SHapley Additive exPlanations) to explain the output of the generated model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X, feature_names=X.columns, show=False, max_display=20, plot_size=(15, 10))
\`\`\`

`ValueError: Unable to determine Axes to steal space for Colorbar. Either provide the cax argument to use as the Axes for the Colorbar, provide the ax argument to steal space from it, or add mappable to an Axes.`

I had to downgrade to matplotlib 3.5.1 to fix the issue.
Please report to shap

I will put on the agenda for next weeks call what we should do about this.  It seems we screwed up the deprecation warning somehow, or a lot of downstream packages didn't see it.  


There was a deprecation warning here, but it was only triggered if the mappable had an axes that was different from the current axes:

https://github.com/matplotlib/matplotlib/blob/a86271c139a056a5c217ec5820143dca9e19f9b8/lib/matplotlib/figure.py#L1182-L1191

In the OP's case, I think the mappable has no axes.
I guess this should get fixed for the bug release, though maybe not the most critical thing in the world.  But we have probably broken a few libraries by doing this with no notice.
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1253,11 +1253,13 @@ def colorbar(
         # Store the value of gca so that we can set it back later on.
         if cax is None:
             if ax is None:
-                raise ValueError(
+                _api.warn_deprecated("3.6", message=(
                     'Unable to determine Axes to steal space for Colorbar. '
+                    'Using gca(), but will raise in the future. '
                     'Either provide the *cax* argument to use as the Axes for '
                     'the Colorbar, provide the *ax* argument to steal space '
-                    'from it, or add *mappable* to an Axes.')
+                    'from it, or add *mappable* to an Axes.'))
+                ax = self.gca()
             current_ax = self.gca()
             userax = False
             if (use_gridspec and isinstance(ax, SubplotBase)):

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_colorbar.py b/lib/matplotlib/tests/test_colorbar.py
--- a/lib/matplotlib/tests/test_colorbar.py
+++ b/lib/matplotlib/tests/test_colorbar.py
@@ -1,10 +1,12 @@
 import numpy as np
 import pytest
 
+from matplotlib import _api
 from matplotlib import cm
 import matplotlib.colors as mcolors
 import matplotlib as mpl
 
+
 from matplotlib import rc_context
 from matplotlib.testing.decorators import image_comparison
 import matplotlib.pyplot as plt
@@ -319,7 +321,8 @@ def test_parentless_mappable():
     pc = mpl.collections.PatchCollection([], cmap=plt.get_cmap('viridis'))
     pc.set_array([])
 
-    with pytest.raises(ValueError, match='Unable to determine Axes to steal'):
+    with pytest.warns(_api.MatplotlibDeprecationWarning,
+                      match='Unable to determine Axes to steal'):
         plt.colorbar(pc)
 
 

```


## Code snippets

### 1 - lib/matplotlib/figure.py:

Start line: 1254, End line: 1281

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def colorbar(
            self, mappable, cax=None, ax=None, use_gridspec=True, **kwargs):
        # ... other code
        if cax is None:
            if ax is None:
                raise ValueError(
                    'Unable to determine Axes to steal space for Colorbar. '
                    'Either provide the *cax* argument to use as the Axes for '
                    'the Colorbar, provide the *ax* argument to steal space '
                    'from it, or add *mappable* to an Axes.')
            current_ax = self.gca()
            userax = False
            if (use_gridspec and isinstance(ax, SubplotBase)):
                cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
            else:
                cax, kwargs = cbar.make_axes(ax, **kwargs)
            cax.grid(visible=False, which='both', axis='both')
        else:
            userax = True

        # need to remove kws that cannot be passed to Colorbar
        NON_COLORBAR_KEYS = ['fraction', 'pad', 'shrink', 'aspect', 'anchor',
                             'panchor']
        cb_kw = {k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS}

        cb = cbar.Colorbar(cax, mappable, **cb_kw)

        if not userax:
            self.sca(current_ax)
        self.stale = True
        return cb
```
### 2 - examples/subplots_axes_and_figures/colorbar_placement.py:

Start line: 1, End line: 82

```python
"""
=================
Placing Colorbars
=================

Colorbars indicate the quantitative extent of image data.  Placing in
a figure is non-trivial because room needs to be made for them.

The simplest case is just attaching a colorbar to each axes:
"""
import matplotlib.pyplot as plt
import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

fig, axs = plt.subplots(2, 2)
cmaps = ['RdBu_r', 'viridis']
for col in range(2):
    for row in range(2):
        ax = axs[row, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        fig.colorbar(pcm, ax=ax)

######################################################################
# The first column has the same type of data in both rows, so it may
# be desirable to combine the colorbar which we do by calling
# `.Figure.colorbar` with a list of axes instead of a single axes.

fig, axs = plt.subplots(2, 2)
cmaps = ['RdBu_r', 'viridis']
for col in range(2):
    for row in range(2):
        ax = axs[row, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
    fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)

######################################################################
# Relatively complicated colorbar layouts are possible using this
# paradigm.  Note that this example works far better with
# ``constrained_layout=True``

fig, axs = plt.subplots(3, 3, constrained_layout=True)
for ax in axs.flat:
    pcm = ax.pcolormesh(np.random.random((20, 20)))

fig.colorbar(pcm, ax=axs[0, :2], shrink=0.6, location='bottom')
fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
fig.colorbar(pcm, ax=axs[1:, :], location='right', shrink=0.6)
fig.colorbar(pcm, ax=[axs[2, 1]], location='left')

######################################################################
# Colorbars with fixed-aspect-ratio axes
# ======================================
#
# Placing colorbars for axes with a fixed aspect ratio pose a particular
# challenge as the parent axes changes size depending on the data view.

fig, axs = plt.subplots(2, 2,  constrained_layout=True)
cmaps = ['RdBu_r', 'viridis']
for col in range(2):
    for row in range(2):
        ax = axs[row, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        if col == 0:
            ax.set_aspect(2)
        else:
            ax.set_aspect(1/2)
        if row == 1:
            fig.colorbar(pcm, ax=ax, shrink=0.6)

######################################################################
# One way around this issue is to use an `.Axes.inset_axes` to locate the
# axes in axes coordinates.  Note that if you zoom in on the axes, and
# change the shape of the axes, the colorbar will also change position.

fig, axs = plt.subplots(2, 2, constrained_layout=True)
cmaps = ['RdBu_r', 'viridis']
```
### 3 - lib/matplotlib/pyplot.py:

Start line: 2043, End line: 2056

```python
## Plotting part 1: manually generated functions and wrappers ##


@_copy_docstring_and_deprecators(Figure.colorbar)
def colorbar(mappable=None, cax=None, ax=None, **kwargs):
    if mappable is None:
        mappable = gci()
        if mappable is None:
            raise RuntimeError('No mappable was found to use for colorbar '
                               'creation. First define a mappable such as '
                               'an image (with imshow) or a contour set ('
                               'with contourf).')
    ret = gcf().colorbar(mappable, cax=cax, ax=ax, **kwargs)
    return ret
```
### 4 - lib/matplotlib/colorbar.py:

Start line: 264, End line: 419

```python
@_docstring.interpd
class Colorbar:

    @_api.delete_parameter("3.6", "filled")
    def __init__(self, ax, mappable=None, *, cmap=None,
                 norm=None,
                 alpha=None,
                 values=None,
                 boundaries=None,
                 orientation='vertical',
                 ticklocation='auto',
                 extend=None,
                 spacing='uniform',  # uniform or proportional
                 ticks=None,
                 format=None,
                 drawedges=False,
                 filled=True,
                 extendfrac=None,
                 extendrect=False,
                 label='',
                 ):

        if mappable is None:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Ensure the given mappable's norm has appropriate vmin and vmax
        # set even if mappable.draw has not yet been called.
        if mappable.get_array() is not None:
            mappable.autoscale_None()

        self.mappable = mappable
        cmap = mappable.cmap
        norm = mappable.norm

        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            alpha = cs.get_alpha()
            boundaries = cs._levels
            values = cs.cvalues
            extend = cs.extend
            filled = cs.filled
            if ticks is None:
                ticks = ticker.FixedLocator(cs.levels, nbins=10)
        elif isinstance(mappable, martist.Artist):
            alpha = mappable.get_alpha()

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

        _api.check_in_list(
            ['vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        self.ax = ax
        self.ax._axes_locator = _ColorbarAxesLocator(self)

        if extend is None:
            if (not isinstance(mappable, contour.ContourSet)
                    and getattr(cmap, 'colorbar_extend', False) is not False):
                extend = cmap.colorbar_extend
            elif hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = None
        # Call set_alpha to handle array-like alphas properly
        self.set_alpha(alpha)
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self._filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self._extend_patches = []
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)
        # Only kept for backcompat; remove after deprecation of .patch elapses.
        self._patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        ax.add_artist(self._patch)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']],
            clip_on=False)
        self.ax.add_collection(self.dividers)

        self._locator = None
        self._minorlocator = None
        self._formatter = None
        self._minorformatter = None
        self.__scale = None  # linear, log10 for now.  Hopefully more?

        if ticklocation == 'auto':
            ticklocation = 'bottom' if orientation == 'horizontal' else 'right'
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self._locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self._locator = ticks

        if isinstance(format, str):
            # Check format between FormatStrFormatter and StrMethodFormatter
            try:
                self._formatter = ticker.FormatStrFormatter(format)
                _ = self._formatter(0)
            except TypeError:
                self._formatter = ticker.StrMethodFormatter(format)
        else:
            self._formatter = format  # Assume it is a Formatter or None
        self._draw_all()

        if isinstance(mappable, contour.ContourSet) and not mappable.filled:
            self.add_lines(mappable)

        # Link the Axes and Colorbar for interactive use
        self.ax._colorbar = self
        # Don't navigate on any of these types of mappables
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm)) or
                isinstance(self.mappable, contour.ContourSet)):
            self.ax.set_navigate(False)

        # These are the functions that set up interactivity on this colorbar
        self._interactive_funcs = ["_get_view", "_set_view",
                                   "_set_view_from_bbox", "drag_pan"]
        for x in self._interactive_funcs:
            setattr(self.ax, x, getattr(self, x))
        # Set the cla function to the cbar's method to override it
        self.ax.cla = self._cbar_cla
        # Callbacks for the extend calculations to handle inverting the axis
        self._extend_cid1 = self.ax.callbacks.connect(
            "xlim_changed", self._do_extends)
        self._extend_cid2 = self.ax.callbacks.connect(
            "ylim_changed", self._do_extends)
```
### 5 - lib/matplotlib/colorbar.py:

Start line: 1, End line: 28

```python
"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which internally use `.Colorbar` together with
   `.make_axes_gridspec` (for `.GridSpec`-positioned axes) or `.make_axes` (for
   non-`.GridSpec`-positioned axes).

   End-users most likely won't need to directly use this module's API.
"""

import logging

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.scale as mscale
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring

_log = logging.getLogger(__name__)
```
### 6 - tutorials/colors/colorbar_only.py:

Start line: 88, End line: 131

```python
cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        .with_extremes(over='0.25', under='0.75'))

bounds = [1, 2, 4, 7, 8]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax,
    extend='both',
    ticks=bounds,
    spacing='proportional',
    orientation='horizontal',
    label='Discrete intervals, some other units',
)

###############################################################################
# Colorbar with custom extension lengths
# --------------------------------------
#
# Here we illustrate the use of custom length colorbar extensions, on a
# colorbar with discrete intervals. To make the length of each extension the
# same as the length of the interior colors, use ``extendfrac='auto'``.

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = (mpl.colors.ListedColormap(['royalblue', 'cyan', 'yellow', 'orange'])
        .with_extremes(over='red', under='blue'))

bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax,
    extend='both',
    extendfrac='auto',
    ticks=bounds,
    spacing='uniform',
    orientation='horizontal',
    label='Custom extension lengths, some other units',
)

plt.show()
```
### 7 - examples/subplots_axes_and_figures/colorbar_placement.py:

Start line: 83, End line: 97

```python
for col in range(2):
    for row in range(2):
        ax = axs[row, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        if col == 0:
            ax.set_aspect(2)
        else:
            ax.set_aspect(1/2)
        if row == 1:
            cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
            fig.colorbar(pcm, ax=ax, cax=cax)

plt.show()
```
### 8 - examples/color/colorbar_basics.py:

Start line: 1, End line: 59

```python
"""
========
Colorbar
========

Use `~.Figure.colorbar` by specifying the mappable object (here
the `.AxesImage` returned by `~.axes.Axes.imshow`)
and the axes to attach the colorbar to.
"""

import numpy as np
import matplotlib.pyplot as plt

# setup some generic data
N = 37
x, y = np.mgrid[:N, :N]
Z = (np.cos(x*0.2) + np.sin(y*0.3))

# mask out the negative and positive values, respectively
Zpos = np.ma.masked_less(Z, 0)
Zneg = np.ma.masked_greater(Z, 0)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

# plot just the positive data and save the
# color "mappable" object returned by ax1.imshow
pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
fig.colorbar(pos, ax=ax1)

# repeat everything above for the negative data
# you can specify location, anchor and shrink the colorbar
neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
fig.colorbar(neg, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.7)

# Plot both positive and negative values between +/- 1.2
pos_neg_clipped = ax3.imshow(Z, cmap='RdBu', vmin=-1.2, vmax=1.2,
                             interpolation='none')
# Add minorticks on the colorbar to make it easy to read the
# values off the colorbar.
cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
cbar.minorticks_on()
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colorbar.Colorbar.minorticks_on`
#    - `matplotlib.colorbar.Colorbar.minorticks_off`
```
### 9 - lib/matplotlib/colorbar.py:

Start line: 690, End line: 712

```python
@_docstring.interpd
class Colorbar:

    def _do_extends(self, ax=None):
        # ... other code
        if self._extend_upper():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 1], [0.5, top], [1, 1]])
            else:
                # rectangle
                xy = np.array([[0, 1], [0, top], [1, top], [1, 1]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            val = 0 if self._long_axis().get_inverted() else -1
            color = self.cmap(self.norm(self._values[val]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, alpha=self.alpha,
                linewidth=0, antialiased=False,
                transform=self.ax.transAxes, hatch=hatches[-1], clip_on=False,
                # Place it right behind the standard patches, which is
                # needed if we updated the extends
                zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
            self.ax.add_patch(patch)
            self._extend_patches.append(patch)

        self._update_dividers()
```
### 10 - lib/matplotlib/colorbar.py:

Start line: 199, End line: 262

```python
@_docstring.interpd
class Colorbar:
    r"""
    Draw a colorbar in an existing axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.

    mappable : `.ScalarMappable`
        The mappable whose colormap and norm will be used.

        To show the under- and over- value colors, the mappable's norm should
        be specified as ::

            norm = colors.Normalize(clip=False)

        To show the colors versus index instead of on a 0-1 scale, use::

            norm=colors.NoNorm()

    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.  This parameter is ignored, unless *mappable* is
        None.

    norm : `~matplotlib.colors.Normalize`
        The normalization to use.  This parameter is ignored, unless *mappable*
        is None.

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    orientation : {'vertical', 'horizontal'}

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}

    drawedges : bool

    filled : bool
    %(_colormap_kw_doc)s
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize
```
### 13 - lib/matplotlib/figure.py:

Start line: 1176, End line: 1253

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def colorbar(
            self, mappable, cax=None, ax=None, use_gridspec=True, **kwargs):
        """
        Add a colorbar to a plot.

        Parameters
        ----------
        mappable
            The `matplotlib.cm.ScalarMappable` (i.e., `.AxesImage`,
            `.ContourSet`, etc.) described by this colorbar.  This argument is
            mandatory for the `.Figure.colorbar` method but optional for the
            `.pyplot.colorbar` function, which sets the default to the current
            image.

            Note that one can create a `.ScalarMappable` "on-the-fly" to
            generate colorbars not attached to a previously drawn artist, e.g.
            ::

                fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        cax : `~matplotlib.axes.Axes`, optional
            Axes into which the colorbar will be drawn.

        ax : `~.axes.Axes` or list or `numpy.ndarray` of Axes, optional
            One or more parent axes from which space for a new colorbar axes
            will be stolen, if *cax* is None.  This has no effect if *cax* is
            set.

        use_gridspec : bool, optional
            If *cax* is ``None``, a new *cax* is created as an instance of
            Axes.  If *ax* is an instance of Subplot and *use_gridspec* is
            ``True``, *cax* is created as an instance of Subplot using the
            :mod:`.gridspec` module.

        Returns
        -------
        colorbar : `~matplotlib.colorbar.Colorbar`

        Other Parameters
        ----------------
        %(_make_axes_kw_doc)s
        %(_colormap_kw_doc)s

        Notes
        -----
        If *mappable* is a `~.contour.ContourSet`, its *extend* kwarg is
        included automatically.

        The *shrink* kwarg provides a simple way to scale the colorbar with
        respect to the axes. Note that if *cax* is specified, it determines the
        size of the colorbar and *shrink* and *aspect* kwargs are ignored.

        For more precise control, you can manually specify the positions of the
        axes objects in which the mappable and the colorbar are drawn.  In this
        case, do not use any of the axes properties kwargs.

        It is known that some vector graphics viewers (svg and pdf) renders
        white gaps between segments of the colorbar.  This is due to bugs in
        the viewers, not Matplotlib.  As a workaround, the colorbar can be
        rendered with overlapping segments::

            cbar = colorbar()
            cbar.solids.set_edgecolor("face")
            draw()

        However this has negative consequences in other circumstances, e.g.
        with semi-transparent images (alpha < 1) and colorbar extensions;
        therefore, this workaround is not used by default (see issue #1188).
        """

        if ax is None:
            ax = getattr(mappable, "axes", None)

        if (self.get_layout_engine() is not None and
                not self.get_layout_engine().colorbar_gridspec):
            use_gridspec = False
        # Store the value of gca so that we can set it back later on.
        # ... other code
```
