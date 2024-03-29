# matplotlib__matplotlib-24971

| **matplotlib/matplotlib** | `a3011dfd1aaa2487cce8aa7369475533133ef777` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 41223 |
| **Avg pos** | 73.0 |
| **Min pos** | 73 |
| **Max pos** | 73 |
| **Top file pos** | 32 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/_tight_bbox.py b/lib/matplotlib/_tight_bbox.py
--- a/lib/matplotlib/_tight_bbox.py
+++ b/lib/matplotlib/_tight_bbox.py
@@ -17,8 +17,6 @@ def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
     """
     origBbox = fig.bbox
     origBboxInches = fig.bbox_inches
-    orig_layout = fig.get_layout_engine()
-    fig.set_layout_engine(None)
     _boxout = fig.transFigure._boxout
 
     old_aspect = []
@@ -46,7 +44,6 @@ def restore_bbox():
 
         fig.bbox = origBbox
         fig.bbox_inches = origBboxInches
-        fig.set_layout_engine(orig_layout)
         fig.transFigure._boxout = _boxout
         fig.transFigure.invalidate()
         fig.patch.set_bounds(0, 0, 1, 1)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/_tight_bbox.py | 20 | 21 | - | 32 | -
| lib/matplotlib/_tight_bbox.py | 49 | 49 | 73 | 32 | 41223


## Problem Statement

```
[Bug]: compressed layout setting can be forgotten on second save
### Bug summary

I'm not sure whether this is really a bug or I'm just using an inconsistent combination of options.  Under some specific circumstances (see below) compressed layout is not applied the second time a figure is saved.

### Code for reproduction

\`\`\`python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

arr = np.arange(100).reshape((10, 10))

matplotlib.rcParams['figure.constrained_layout.use'] = True

fig, ax_dict = plt.subplot_mosaic('AB;AC', figsize=(6, 9), width_ratios=[3, 2],
                                  layout='compressed')

for key in ["B", "C"]:
    ax_dict[key].imshow(arr)
    
fig.savefig("test1.png", bbox_inches="tight")
fig.savefig("test2.png", bbox_inches="tight")
\`\`\`


### Actual outcome

test1.png
![test1](https://user-images.githubusercontent.com/10599679/212073531-4841d847-29a5-45a4-aaa1-1d3b81277ddc.png)

test2.png
![test2](https://user-images.githubusercontent.com/10599679/212073574-f6286243-690d-4199-b6f4-4033e5d14635.png)


### Expected outcome

Both images should look like the first.

### Additional information

If I do not set the `rcParams`, all is well.  If I do not set `bbox_inches="tight"` in my calls to `savefig`, the images are identical (but I have too much white space top and bottom).  Maybe there is a better option than `bbox_inches="tight"` when using compressed layout?

For context, my real example is a script that makes several figures.  For most of them I want constrained layout, so I set that once in the `rcParams` for convenience.  Only one figure needs "compressed", and I am saving twice because I want both a png and a pdf.  Fixed it in my current example by just reverting the `rcParams` setting for the one figure.

### Operating system

RHEL7

### Matplotlib Version

3.6.2 and main

### Matplotlib Backend

QtAgg

### Python version

3.9 and 3.11

### Jupyter version

_No response_

### Installation

conda

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 tutorials/intermediate/constrainedlayout_guide.py | 442 | 530| 776 | 776 | 6566 | 
| 2 | 2 lib/matplotlib/_constrained_layout.py | 263 | 297| 327 | 1103 | 14133 | 
| 3 | 2 lib/matplotlib/_constrained_layout.py | 111 | 149| 471 | 1574 | 14133 | 
| 4 | 2 tutorials/intermediate/constrainedlayout_guide.py | 187 | 269| 813 | 2387 | 14133 | 
| 5 | 2 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 3168 | 14133 | 
| 6 | 2 tutorials/intermediate/constrainedlayout_guide.py | 532 | 634| 1013 | 4181 | 14133 | 
| 7 | 2 tutorials/intermediate/constrainedlayout_guide.py | 636 | 721| 777 | 4958 | 14133 | 
| 8 | 3 examples/scales/semilogx_demo.py | 1 | 24| 104 | 5062 | 14237 | 
| 9 | 3 tutorials/intermediate/constrainedlayout_guide.py | 270 | 352| 804 | 5866 | 14237 | 
| 10 | 4 tutorials/intermediate/tight_layout_guide.py | 1 | 104| 778 | 6644 | 16390 | 
| 11 | 5 tutorials/intermediate/arranging_axes.py | 101 | 179| 776 | 7420 | 20349 | 
| 12 | 5 tutorials/intermediate/constrainedlayout_guide.py | 354 | 441| 772 | 8192 | 20349 | 
| 13 | 6 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 115| 756 | 8948 | 21228 | 
| 14 | 7 examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 430 | 9378 | 21658 | 
| 15 | 7 tutorials/intermediate/constrainedlayout_guide.py | 1 | 112| 829 | 10207 | 21658 | 
| 16 | 7 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 880 | 11087 | 21658 | 
| 17 | 7 lib/matplotlib/_constrained_layout.py | 511 | 574| 719 | 11806 | 21658 | 
| 18 | 7 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 12301 | 21658 | 
| 19 | 7 lib/matplotlib/_constrained_layout.py | 1 | 59| 602 | 12903 | 21658 | 
| 20 | 8 lib/matplotlib/figure.py | 2776 | 2803| 266 | 13169 | 50861 | 
| 21 | 8 lib/matplotlib/_constrained_layout.py | 480 | 509| 297 | 13466 | 50861 | 
| 22 | 8 lib/matplotlib/figure.py | 2042 | 2081| 401 | 13867 | 50861 | 
| 23 | 8 lib/matplotlib/figure.py | 2742 | 2774| 295 | 14162 | 50861 | 
| 24 | 8 lib/matplotlib/figure.py | 2357 | 2526| 1513 | 15675 | 50861 | 
| 25 | 9 examples/subplots_axes_and_figures/mosaic.py | 56 | 167| 789 | 16464 | 53532 | 
| 26 | 9 examples/subplots_axes_and_figures/demo_tight_layout.py | 116 | 135| 123 | 16587 | 53532 | 
| 27 | 10 examples/shapes_and_collections/fancybox_demo.py | 80 | 127| 501 | 17088 | 54800 | 
| 28 | 10 tutorials/intermediate/arranging_axes.py | 265 | 344| 760 | 17848 | 54800 | 
| 29 | 11 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 18515 | 56223 | 
| 30 | 11 lib/matplotlib/_constrained_layout.py | 357 | 421| 780 | 19295 | 56223 | 
| 31 | 12 tutorials/introductory/quick_start.py | 291 | 388| 986 | 20281 | 62166 | 
| 32 | 12 lib/matplotlib/_constrained_layout.py | 752 | 766| 140 | 20421 | 62166 | 
| 33 | 12 lib/matplotlib/_constrained_layout.py | 422 | 438| 233 | 20654 | 62166 | 
| 34 | 13 lib/matplotlib/pyplot.py | 993 | 1010| 148 | 20802 | 90643 | 
| 35 | 14 doc/conf.py | 111 | 158| 481 | 21283 | 96359 | 
| 36 | 14 lib/matplotlib/pyplot.py | 1 | 90| 675 | 21958 | 96359 | 
| 37 | 14 lib/matplotlib/_constrained_layout.py | 243 | 260| 153 | 22111 | 96359 | 
| 38 | 15 examples/userdemo/connectionstyle_demo.py | 33 | 64| 523 | 22634 | 97117 | 
| 39 | 15 lib/matplotlib/pyplot.py | 776 | 842| 679 | 23313 | 97117 | 
| 40 | 16 lib/matplotlib/backends/backend_cairo.py | 447 | 501| 496 | 23809 | 101397 | 
| 41 | 17 lib/matplotlib/_cm.py | 713 | 739| 719 | 24528 | 129840 | 
| 42 | 17 tutorials/intermediate/arranging_axes.py | 346 | 415| 676 | 25204 | 129840 | 
| 43 | 17 lib/matplotlib/_cm.py | 919 | 940| 502 | 25706 | 129840 | 
| 44 | 18 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 26408 | 130951 | 
| 45 | 18 lib/matplotlib/pyplot.py | 2273 | 2348| 763 | 27171 | 130951 | 
| 46 | 19 lib/matplotlib/backends/backend_agg.py | 379 | 557| 1430 | 28601 | 135744 | 
| 47 | 19 examples/subplots_axes_and_figures/mosaic.py | 168 | 288| 779 | 29380 | 135744 | 
| 48 | 20 examples/images_contours_and_fields/image_antialiasing.py | 70 | 121| 538 | 29918 | 137042 | 
| 49 | 21 lib/matplotlib/image.py | 1623 | 1687| 745 | 30663 | 154191 | 
| 50 | 22 lib/matplotlib/backends/backend_gtk3.py | 134 | 194| 568 | 31231 | 159187 | 
| 51 | 22 examples/subplots_axes_and_figures/mosaic.py | 289 | 392| 654 | 31885 | 159187 | 
| 52 | 23 examples/misc/customize_rc.py | 1 | 60| 446 | 32331 | 159633 | 
| 53 | 23 lib/matplotlib/_cm.py | 741 | 765| 637 | 32968 | 159633 | 
| 54 | 24 lib/matplotlib/testing/__init__.py | 28 | 50| 183 | 33151 | 160748 | 
| 55 | 25 lib/matplotlib/testing/conftest.py | 36 | 99| 431 | 33582 | 161478 | 
| 56 | 26 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 442 | 34024 | 163446 | 
| 57 | 26 lib/matplotlib/figure.py | 2528 | 2548| 283 | 34307 | 163446 | 
| 58 | 26 lib/matplotlib/_cm.py | 942 | 959| 398 | 34705 | 163446 | 
| 59 | 27 lib/matplotlib/layout_engine.py | 237 | 254| 149 | 34854 | 165812 | 
| 60 | 28 examples/ticks/ticks_too_many.py | 1 | 77| 742 | 35596 | 166554 | 
| 61 | 29 examples/misc/rasterization_demo.py | 75 | 95| 166 | 35762 | 167482 | 
| 62 | 29 lib/matplotlib/pyplot.py | 2260 | 2270| 137 | 35899 | 167482 | 
| 63 | 29 lib/matplotlib/pyplot.py | 1261 | 1325| 676 | 36575 | 167482 | 
| 64 | 29 lib/matplotlib/figure.py | 1762 | 1789| 203 | 36778 | 167482 | 
| 65 | 30 lib/matplotlib/backends/backend_gtk3agg.py | 50 | 75| 246 | 37024 | 168078 | 
| 66 | 30 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 37809 | 168078 | 
| 67 | 31 lib/matplotlib/testing/compare.py | 319 | 339| 237 | 38046 | 172397 | 
| 68 | 31 lib/matplotlib/_cm.py | 767 | 791| 613 | 38659 | 172397 | 
| 69 | 31 lib/matplotlib/_cm.py | 663 | 687| 670 | 39329 | 172397 | 
| 70 | 31 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 40085 | 172397 | 
| 71 | 31 lib/matplotlib/_cm.py | 1293 | 1314| 809 | 40894 | 172397 | 
| 72 | 31 lib/matplotlib/figure.py | 2259 | 2281| 174 | 41068 | 172397 | 
| **-> 73 <-** | **32 lib/matplotlib/_tight_bbox.py** | 38 | 52| 155 | 41223 | 173101 | 
| 74 | 32 lib/matplotlib/_cm.py | 637 | 661| 660 | 41883 | 173101 | 
| 75 | 32 doc/conf.py | 178 | 273| 760 | 42643 | 173101 | 
| 76 | 33 lib/matplotlib/backends/_backend_tk.py | 1 | 53| 360 | 43003 | 182607 | 
| 77 | 34 examples/subplots_axes_and_figures/axes_margins.py | 1 | 88| 770 | 43773 | 183377 | 
| 78 | 35 lib/matplotlib/_tight_layout.py | 266 | 302| 362 | 44135 | 186306 | 
| 79 | 35 lib/matplotlib/testing/compare.py | 342 | 355| 183 | 44318 | 186306 | 
| 80 | 36 tutorials/introductory/lifecycle.py | 189 | 279| 816 | 45134 | 188653 | 


### Hint

```
Yeah we do some dancing around when we save with bbox inches - so this seems to get caught in that. I tried to track it down, but the figure-saving stack is full of context managers, and I can't see where the layout manager gets reset.  Hopefully someone more cognizant of that part of the codebase can explain.  

Thanks for looking @jklymak 🙂
I think it is set (temporarily) here;
https://github.com/matplotlib/matplotlib/blob/018c5efbbec68f27cfea66ca2620702dd976d1b9/lib/matplotlib/backend_bases.py#L2356-L2357
It is, but I don't understand what `_cm_set` does to reset the layout engine after this.  Somehow it is dropping the old layout engine and making a new one, and the new one doesn't know that the old one was a 'compressed' engine.  
It calls `get_{kwarg}` and after running calls `set({kwarg}={old value})`. So here it calls `oldvalue = figure.get_layout_engine()` and `figure.set(layout_engine=oldvalue)`. Is `figure.set_layout_engine(figure.get_layout_engine())` working correctly?
I am way out of my depth here but

\`\`\`python
import matplotlib.pyplot as plt

plt.rcParams['figure.constrained_layout.use'] = True
fig = plt.figure(layout="compressed")

print(fig.get_layout_engine()._compress)
fig.set_layout_engine(fig.get_layout_engine())
print(fig.get_layout_engine()._compress)

fig.savefig('foo.png', bbox_inches='tight')
print(fig.get_layout_engine()._compress)
\`\`\`

\`\`\`
True
True
False
\`\`\`

Without the `rcParams` line, `fig.get_layout_engine()` returns `None` after the `savefig`.
I _think_ the problem is the call to `adjust_bbox`
https://github.com/matplotlib/matplotlib/blob/018c5efbbec68f27cfea66ca2620702dd976d1b9/lib/matplotlib/backend_bases.py#L2349-L2350

which explicity calls
https://github.com/matplotlib/matplotlib/blob/a3011dfd1aaa2487cce8aa7369475533133ef777/lib/matplotlib/_tight_bbox.py#L21

which will use the default constrained layout engine if the `rcParams` is set
https://github.com/matplotlib/matplotlib/blob/a3011dfd1aaa2487cce8aa7369475533133ef777/lib/matplotlib/figure.py#L2599-L2610
```

## Patch

```diff
diff --git a/lib/matplotlib/_tight_bbox.py b/lib/matplotlib/_tight_bbox.py
--- a/lib/matplotlib/_tight_bbox.py
+++ b/lib/matplotlib/_tight_bbox.py
@@ -17,8 +17,6 @@ def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
     """
     origBbox = fig.bbox
     origBboxInches = fig.bbox_inches
-    orig_layout = fig.get_layout_engine()
-    fig.set_layout_engine(None)
     _boxout = fig.transFigure._boxout
 
     old_aspect = []
@@ -46,7 +44,6 @@ def restore_bbox():
 
         fig.bbox = origBbox
         fig.bbox_inches = origBboxInches
-        fig.set_layout_engine(orig_layout)
         fig.transFigure._boxout = _boxout
         fig.transFigure.invalidate()
         fig.patch.set_bounds(0, 0, 1, 1)

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_figure.py b/lib/matplotlib/tests/test_figure.py
--- a/lib/matplotlib/tests/test_figure.py
+++ b/lib/matplotlib/tests/test_figure.py
@@ -532,6 +532,13 @@ def test_savefig_pixel_ratio(backend):
     assert ratio1 == ratio2
 
 
+def test_savefig_preserve_layout_engine(tmp_path):
+    fig = plt.figure(layout='compressed')
+    fig.savefig(tmp_path / 'foo.png', bbox_inches='tight')
+
+    assert fig.get_layout_engine()._compress
+
+
 def test_figure_repr():
     fig = plt.figure(figsize=(10, 20), dpi=10)
     assert repr(fig) == "<Figure size 100x200 with 0 Axes>"

```


## Code snippets

### 1 - tutorials/intermediate/constrainedlayout_guide.py:

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
### 2 - lib/matplotlib/_constrained_layout.py:

Start line: 263, End line: 297

```python
def compress_fixed_aspect(layoutgrids, fig):
    gs = None
    for ax in fig.axes:
        if ax.get_subplotspec() is None:
            continue
        ax.apply_aspect()
        sub = ax.get_subplotspec()
        _gs = sub.get_gridspec()
        if gs is None:
            gs = _gs
            extraw = np.zeros(gs.ncols)
            extrah = np.zeros(gs.nrows)
        elif _gs != gs:
            raise ValueError('Cannot do compressed layout if axes are not'
                                'all from the same gridspec')
        orig = ax.get_position(original=True)
        actual = ax.get_position(original=False)
        dw = orig.width - actual.width
        if dw > 0:
            extraw[sub.colspan] = np.maximum(extraw[sub.colspan], dw)
        dh = orig.height - actual.height
        if dh > 0:
            extrah[sub.rowspan] = np.maximum(extrah[sub.rowspan], dh)

    if gs is None:
        raise ValueError('Cannot do compressed layout if no axes '
                         'are part of a gridspec.')
    w = np.sum(extraw) / 2
    layoutgrids[fig].edit_margin_min('left', w)
    layoutgrids[fig].edit_margin_min('right', w)

    h = np.sum(extrah) / 2
    layoutgrids[fig].edit_margin_min('top', h)
    layoutgrids[fig].edit_margin_min('bottom', h)
    return layoutgrids
```
### 3 - lib/matplotlib/_constrained_layout.py:

Start line: 111, End line: 149

```python
def do_constrained_layout(fig, h_pad, w_pad,
                          hspace=None, wspace=None, rect=(0, 0, 1, 1),
                          compress=False):
    # ... other code

    for _ in range(2):
        # do the algorithm twice.  This has to be done because decorations
        # change size after the first re-position (i.e. x/yticklabels get
        # larger/smaller).  This second reposition tends to be much milder,
        # so doing twice makes things work OK.

        # make margins for all the axes and subfigures in the
        # figure.  Add margins for colorbars...
        make_layout_margins(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
        make_margin_suptitles(layoutgrids, fig, renderer, h_pad=h_pad,
                              w_pad=w_pad)

        # if a layout is such that a columns (or rows) margin has no
        # constraints, we need to make all such instances in the grid
        # match in margin size.
        match_submerged_margins(layoutgrids, fig)

        # update all the variables in the layout.
        layoutgrids[fig].update_variables()

        warn_collapsed = ('constrained_layout not applied because '
                          'axes sizes collapsed to zero.  Try making '
                          'figure larger or axes decorations smaller.')
        if check_no_collapsed_axes(layoutgrids, fig):
            reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
            if compress:
                layoutgrids = compress_fixed_aspect(layoutgrids, fig)
                layoutgrids[fig].update_variables()
                if check_no_collapsed_axes(layoutgrids, fig):
                    reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                                    w_pad=w_pad, hspace=hspace, wspace=wspace)
                else:
                    _api.warn_external(warn_collapsed)
        else:
            _api.warn_external(warn_collapsed)
        reset_margins(layoutgrids, fig)
    return layoutgrids
```
### 4 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 187, End line: 269

```python
axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
leg = axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
leg.set_in_layout(False)
# trigger a draw so that constrained_layout is executed once
# before we turn it off when printing....
fig.canvas.draw()
# we want the legend included in the bbox_inches='tight' calcs.
leg.set_in_layout(True)
# we don't want the layout to change at this point.
fig.set_layout_engine(None)
try:
    fig.savefig('../../doc/_static/constrained_layout_1b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # this allows the script to keep going if run interactively and
    # the directory above doesn't exist
    pass

#############################################
# The saved file looks like:
#
# .. image:: /_static/constrained_layout_1b.png
#    :align: center
#
# A better way to get around this awkwardness is to simply
# use the legend method provided by `.Figure.legend`:
fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")
axs[0].plot(np.arange(10))
lines = axs[1].plot(np.arange(10), label='This is a plot')
labels = [l.get_label() for l in lines]
leg = fig.legend(lines, labels, loc='center left',
                 bbox_to_anchor=(0.8, 0.5), bbox_transform=axs[1].transAxes)
try:
    fig.savefig('../../doc/_static/constrained_layout_2b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # this allows the script to keep going if run interactively and
    # the directory above doesn't exist
    pass


#############################################
# The saved file looks like:
#
# .. image:: /_static/constrained_layout_2b.png
#    :align: center
#

###############################################################################
# Padding and Spacing
# ===================
#
# Padding between axes is controlled in the horizontal by *w_pad* and
# *wspace*, and vertical by *h_pad* and *hspace*.  These can be edited
# via `~.layout_engine.ConstrainedLayoutEngine.set`.  *w/h_pad* are
# the minimum space around the axes in units of inches:

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
                            wspace=0)

##########################################
# Spacing between subplots is further set by *wspace* and *hspace*. These
# are specified as a fraction of the size of the subplot group as a whole.
# If these values are smaller than *w_pad* or *h_pad*, then the fixed pads are
# used instead. Note in the below how the space at the edges doesn't change
# from the above, but the space between subplots does.

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

##########################################
# If there are more than two columns, the *wspace* is shared between them,
# so here the wspace is divided in two, with a *wspace* of 0.1 between each
# column:

fig, axs = plt.subplots(2, 3, layout="constrained")
```
### 5 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 113, End line: 185

```python
norm = mcolors.Normalize(vmin=0., vmax=100.)
# see note above: this makes all pcolormesh calls consistent:
pc_kwargs = {'rasterized': True, 'cmap': 'viridis', 'norm': norm}
fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=ax, shrink=0.6)

############################################################################
# If you specify a list of axes (or other iterable container) to the
# ``ax`` argument of ``colorbar``, constrained_layout will take space from
# the specified axes.

fig, axs = plt.subplots(2, 2, figsize=(4, 4), layout="constrained")
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)

############################################################################
# If you specify a list of axes from inside a grid of axes, the colorbar
# will steal space appropriately, and leave a gap, but all subplots will
# still be the same size.

fig, axs = plt.subplots(3, 3, figsize=(4, 4), layout="constrained")
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs[1:, ][:, 1], shrink=0.8)
fig.colorbar(im, ax=axs[:, -1], shrink=0.6)

####################################################
# Suptitle
# =========
#
# ``constrained_layout`` can also make room for `~.Figure.suptitle`.

fig, axs = plt.subplots(2, 2, figsize=(4, 4), layout="constrained")
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)
fig.suptitle('Big Suptitle')

####################################################
# Legends
# =======
#
# Legends can be placed outside of their parent axis.
# Constrained-layout is designed to handle this for :meth:`.Axes.legend`.
# However, constrained-layout does *not* handle legends being created via
# :meth:`.Figure.legend` (yet).

fig, ax = plt.subplots(layout="constrained")
ax.plot(np.arange(10), label='This is a plot')
ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

#############################################
# However, this will steal space from a subplot layout:

fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")
axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

#############################################
# In order for a legend or other artist to *not* steal space
# from the subplot layout, we can ``leg.set_in_layout(False)``.
# Of course this can mean the legend ends up
# cropped, but can be useful if the plot is subsequently called
# with ``fig.savefig('outname.png', bbox_inches='tight')``.  Note,
# however, that the legend's ``get_in_layout`` status will have to be
# toggled again to make the saved file work, and we must manually
# trigger a draw if we want constrained_layout to adjust the size
# of the axes before printing.

fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")
```
### 6 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 532, End line: 634

```python
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(1, 2, 2)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.suptitle('Mixed nrows, ncols')

###############################################################################
# Similarly,
# `~matplotlib.pyplot.subplot2grid` works with the same limitation
# that nrows and ncols cannot change for the layout to look good.

fig = plt.figure(layout="constrained")

ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
fig.suptitle('subplot2grid')

###############################################################################
# Other Caveats
# -------------
#
# * ``constrained_layout`` only considers ticklabels, axis labels, titles, and
#   legends.  Thus, other artists may be clipped and also may overlap.
#
# * It assumes that the extra space needed for ticklabels, axis labels,
#   and titles is independent of original location of axes. This is
#   often true, but there are rare cases where it is not.
#
# * There are small differences in how the backends handle rendering fonts,
#   so the results will not be pixel-identical.
#
# * An artist using axes coordinates that extend beyond the axes
#   boundary will result in unusual layouts when added to an
#   axes. This can be avoided by adding the artist directly to the
#   :class:`~matplotlib.figure.Figure` using
#   :meth:`~matplotlib.figure.Figure.add_artist`. See
#   :class:`~matplotlib.patches.ConnectionPatch` for an example.

###########################################################
# Debugging
# =========
#
# Constrained-layout can fail in somewhat unexpected ways.  Because it uses
# a constraint solver the solver can find solutions that are mathematically
# correct, but that aren't at all what the user wants.  The usual failure
# mode is for all sizes to collapse to their smallest allowable value. If
# this happens, it is for one of two reasons:
#
# 1. There was not enough room for the elements you were requesting to draw.
# 2. There is a bug - in which case open an issue at
#    https://github.com/matplotlib/matplotlib/issues.
#
# If there is a bug, please report with a self-contained example that does
# not require outside data or dependencies (other than numpy).

###########################################################
# Notes on the algorithm
# ======================
#
# The algorithm for the constraint is relatively straightforward, but
# has some complexity due to the complex ways we can lay out a figure.
#
# Layout in Matplotlib is carried out with gridspecs
# via the `.GridSpec` class. A gridspec is a logical division of the figure
# into rows and columns, with the relative width of the Axes in those
# rows and columns set by *width_ratios* and *height_ratios*.
#
# In constrained_layout, each gridspec gets a *layoutgrid* associated with
# it.  The *layoutgrid* has a series of ``left`` and ``right`` variables
# for each column, and ``bottom`` and ``top`` variables for each row, and
# further it has a margin for each of left, right, bottom and top.  In each
# row, the bottom/top margins are widened until all the decorators
# in that row are accommodated.  Similarly, for columns and the left/right
# margins.
#
#
# Simple case: one Axes
# ---------------------
#
# For a single Axes the layout is straight forward.  There is one parent
# layoutgrid for the figure consisting of one column and row, and
# a child layoutgrid for the gridspec that contains the axes, again
# consisting of one row and column. Space is made for the "decorations" on
# each side of the axes.  In the code, this is accomplished by the entries in
# ``do_constrained_layout()`` like::
#
#     gridspec._layoutgrid[0, 0].edit_margin_min('left',
#           -bbox.x0 + pos.x0 + w_pad)
#
# where ``bbox`` is the tight bounding box of the axes, and ``pos`` its
# position.  Note how the four margins encompass the axes decorations.

from matplotlib._layoutgrid import plot_children
```
### 7 - tutorials/intermediate/constrainedlayout_guide.py:

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
plot_children(fig)

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
plot_children(fig)

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
plot_children(fig)

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
### 8 - examples/scales/semilogx_demo.py:

Start line: 1, End line: 24

```python
"""
========
Log Axis
========

.. redirect-from:: /gallery/scales/log_test

This is an example of assigning a log-scale for the x-axis using
`~.axes.Axes.semilogx`.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

dt = 0.01
t = np.arange(dt, 20.0, dt)

ax.semilogx(t, np.exp(-t / 5.0))
ax.grid()

plt.show()
```
### 9 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 270, End line: 352

```python
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

##########################################
# GridSpecs also have optional *hspace* and *wspace* keyword arguments,
# that will be used instead of the pads set by ``constrained_layout``:

fig, axs = plt.subplots(2, 2, layout="constrained",
                        gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
# this has no effect because the space set in the gridspec trumps the
# space set in constrained_layout.
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0,
                            wspace=0.0)

##########################################
# Spacing with colorbars
# -----------------------
#
# Colorbars are placed a distance *pad* from their parent, where *pad*
# is a fraction of the width of the parent(s).  The spacing to the
# next subplot is then given by *w/hspace*.

fig, axs = plt.subplots(2, 2, layout="constrained")
pads = [0, 0.05, 0.1, 0.2]
for pad, ax in zip(pads, axs.flat):
    pc = ax.pcolormesh(arr, **pc_kwargs)
    fig.colorbar(pc, ax=ax, shrink=0.6, pad=pad)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'pad: {pad}')
fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,
                            wspace=0.2)

##########################################
# rcParams
# ========
#
# There are five :ref:`rcParams<customizing-with-dynamic-rc-settings>`
# that can be set, either in a script or in the :file:`matplotlibrc`
# file. They all have the prefix ``figure.constrained_layout``:
#
# - *use*: Whether to use constrained_layout. Default is False
# - *w_pad*, *h_pad*:    Padding around axes objects.
#   Float representing inches.  Default is 3./72. inches (3 pts)
# - *wspace*, *hspace*:  Space between subplot groups.
#   Float representing a fraction of the subplot widths being separated.
#   Default is 0.02.

plt.rcParams['figure.constrained_layout.use'] = True
fig, axs = plt.subplots(2, 2, figsize=(3, 3))
for ax in axs.flat:
    example_plot(ax)

#############################
# Use with GridSpec
# =================
#
# constrained_layout is meant to be used
# with :func:`~matplotlib.figure.Figure.subplots`,
# :func:`~matplotlib.figure.Figure.subplot_mosaic`, or
# :func:`~matplotlib.gridspec.GridSpec` with
# :func:`~matplotlib.figure.Figure.add_subplot`.
#
# Note that in what follows ``layout="constrained"``

plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(layout="constrained")

gs1 = gridspec.GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])

example_plot(ax1)
example_plot(ax2)

###############################################################################
# More complicated gridspec layouts are possible.  Note here we use the
# convenience functions `~.Figure.add_gridspec` and
# `~.SubplotSpec.subgridspec`.
```
### 10 - tutorials/intermediate/tight_layout_guide.py:

Start line: 1, End line: 104

```python
"""
==================
Tight Layout guide
==================

How to use tight-layout to fit plots within your figure cleanly.

*tight_layout* automatically adjusts subplot params so that the
subplot(s) fits in to the figure area. This is an experimental
feature and may not work for some cases. It only checks the extents
of ticklabels, axis labels, and titles.

An alternative to *tight_layout* is :doc:`constrained_layout
</tutorials/intermediate/constrainedlayout_guide>`.


Simple Example
==============

In matplotlib, the location of axes (including subplots) are specified in
normalized figure coordinates. It can happen that your axis labels or
titles (or sometimes even ticklabels) go outside the figure area, and are thus
clipped.

"""

# sphinx_gallery_thumbnail_number = 7

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['savefig.facecolor'] = "0.8"


def example_plot(ax, fontsize=12):
    ax.plot([1, 2])

    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

plt.close('all')
fig, ax = plt.subplots()
example_plot(ax, fontsize=24)

###############################################################################
# To prevent this, the location of axes needs to be adjusted. For
# subplots, this can be done manually by adjusting the subplot parameters
# using `.Figure.subplots_adjust`. `.Figure.tight_layout` does this
# automatically.

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()

###############################################################################
# Note that :func:`matplotlib.pyplot.tight_layout` will only adjust the
# subplot params when it is called.  In order to perform this adjustment each
# time the figure is redrawn, you can call ``fig.set_tight_layout(True)``, or,
# equivalently, set :rc:`figure.autolayout` to ``True``.
#
# When you have multiple subplots, often you see labels of different
# axes overlapping each other.

plt.close('all')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)

###############################################################################
# :func:`~matplotlib.pyplot.tight_layout` will also adjust spacing between
# subplots to minimize the overlaps.

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()

###############################################################################
# :func:`~matplotlib.pyplot.tight_layout` can take keyword arguments of
# *pad*, *w_pad* and *h_pad*. These control the extra padding around the
# figure border and between subplots. The pads are specified in fraction
# of fontsize.

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

###############################################################################
# :func:`~matplotlib.pyplot.tight_layout` will work even if the sizes of
# subplots are different as far as their grid specification is
# compatible. In the example below, *ax1* and *ax2* are subplots of a 2x2
# grid, while *ax3* is of a 1x2 grid.

plt.close('all')
```
### 73 - lib/matplotlib/_tight_bbox.py:

Start line: 38, End line: 52

```python
def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
    # ... other code

    def restore_bbox():
        for ax, loc, aspect in zip(fig.axes, locator_list, old_aspect):
            ax.set_axes_locator(loc)
            if aspect is sentinel:
                # delete our no-op function which un-hides the original method
                del ax.apply_aspect
            else:
                ax.apply_aspect = aspect

        fig.bbox = origBbox
        fig.bbox_inches = origBboxInches
        fig.set_layout_engine(orig_layout)
        fig.transFigure._boxout = _boxout
        fig.transFigure.invalidate()
        fig.patch.set_bounds(0, 0, 1, 1)
    # ... other code
```
