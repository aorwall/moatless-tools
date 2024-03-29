# matplotlib__matplotlib-21481

| **matplotlib/matplotlib** | `d448de31b7deaec8310caaf8bba787e097bf9211` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 19 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/_layoutgrid.py b/lib/matplotlib/_layoutgrid.py
--- a/lib/matplotlib/_layoutgrid.py
+++ b/lib/matplotlib/_layoutgrid.py
@@ -169,7 +169,8 @@ def hard_constraints(self):
                 self.solver.addConstraint(c | 'required')
 
     def add_child(self, child, i=0, j=0):
-        self.children[i, j] = child
+        # np.ix_ returns the cross product of i and j indices
+        self.children[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] = child
 
     def parent_constraints(self):
         # constraints that are due to the parent...

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/_layoutgrid.py | 172 | 172 | - | 19 | -


## Problem Statement

```
[Bug]: Subfigure breaks for some `Gridspec` slices when using `constrained_layout`
### Bug summary

When creating a figure with `constrained_layout=True` you cannot use arbitrary gridspecs to create subfigures as it throws an error at some point ( I think once the layout manager actually takes effect?). This happened immediately on the `add_subfigure` call in `3.4.3` and only on the `add_subplot` call on `main`

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 3)
subfig = fig.add_subfigure(gs[0:, 1:])
subfig.add_subplot()
\`\`\`


### Actual outcome

\`\`\`
Traceback (most recent call last):
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/backends/backend_qt.py", line 455, in _draw_idle
    self.draw()
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/backends/backend_agg.py", line 436, in draw
    self.figure.draw(self.renderer)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/artist.py", line 73, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/artist.py", line 50, in draw_wrapper
    return draw(artist, renderer)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/figure.py", line 2795, in draw
    self.execute_constrained_layout(renderer)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/figure.py", line 3153, in execute_constrained_layout
    return do_constrained_layout(fig, renderer, h_pad, w_pad,
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/_constrained_layout.py", line 95, in do_constrained_layout
    layoutgrids = make_layoutgrids(fig, None)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/_constrained_layout.py", line 167, in make_layoutgrids
    layoutgrids = make_layoutgrids(sfig, layoutgrids)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/_constrained_layout.py", line 158, in make_layoutgrids
    layoutgrids[fig] = mlayoutgrid.LayoutGrid(
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/_layoutgrid.py", line 59, in __init__
    parent.add_child(self, *parent_pos)
  File "/home/ian/Documents/oss/matplotlib/matplotlib/lib/matplotlib/_layoutgrid.py", line 172, in add_child
    self.children[i, j] = child
IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (2,) 
\`\`\`

### Expected outcome

No error. Should be the same as with `constrained_layout=False`

### Operating system

Ubuntu

### Matplotlib Version

3.5.0.dev2428+g8daad3364a

### Matplotlib Backend

QtAgg

### Python version

3.9.2

### Jupyter version

_No response_

### Other libraries

_No response_

### Installation

source

### Conda channel

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 756 | 1423 | 
| 2 | 2 tutorials/intermediate/gridspec.py | 1 | 78| 757 | 1513 | 4199 | 
| 3 | 3 examples/subplots_axes_and_figures/gridspec_multicolumn.py | 1 | 34| 259 | 1772 | 4458 | 
| 4 | 4 tutorials/intermediate/constrainedlayout_guide.py | 591 | 676| 788 | 2560 | 10561 | 
| 5 | 4 tutorials/intermediate/constrainedlayout_guide.py | 269 | 350| 769 | 3329 | 10561 | 
| 6 | 4 tutorials/intermediate/gridspec.py | 79 | 136| 719 | 4048 | 10561 | 
| 7 | 4 tutorials/intermediate/gridspec.py | 137 | 206| 756 | 4804 | 10561 | 
| 8 | 4 tutorials/intermediate/constrainedlayout_guide.py | 426 | 589| 1490 | 6294 | 10561 | 
| 9 | 4 tutorials/intermediate/constrainedlayout_guide.py | 352 | 393| 338 | 6632 | 10561 | 
| 10 | 5 lib/matplotlib/figure.py | 1868 | 1903| 377 | 7009 | 36806 | 
| 11 | 6 lib/matplotlib/_constrained_layout.py | 134 | 175| 373 | 7382 | 43655 | 
| 12 | 7 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 877 | 8259 | 45805 | 
| 13 | 7 lib/matplotlib/_constrained_layout.py | 436 | 499| 719 | 8978 | 45805 | 
| 14 | 8 examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 430 | 9408 | 46235 | 
| 15 | 8 lib/matplotlib/_constrained_layout.py | 178 | 219| 430 | 9838 | 46235 | 
| 16 | 8 tutorials/intermediate/constrainedlayout_guide.py | 122 | 194| 777 | 10615 | 46235 | 
| 17 | 8 tutorials/intermediate/gridspec.py | 208 | 266| 545 | 11160 | 46235 | 
| 18 | 9 examples/userdemo/demo_gridspec03.py | 1 | 51| 419 | 11579 | 46654 | 
| 19 | 9 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 12246 | 46654 | 
| 20 | 9 tutorials/intermediate/constrainedlayout_guide.py | 196 | 268| 763 | 13009 | 46654 | 
| 21 | 9 tutorials/intermediate/constrainedlayout_guide.py | 1 | 121| 916 | 13925 | 46654 | 
| 22 | 10 examples/subplots_axes_and_figures/gridspec_and_subplots.py | 1 | 28| 217 | 14142 | 46871 | 
| 23 | 11 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 116| 759 | 14901 | 47753 | 
| 24 | 12 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 441 | 15342 | 49720 | 
| 25 | 13 lib/matplotlib/pyplot.py | 1229 | 1293| 649 | 15991 | 76114 | 
| 26 | 14 lib/matplotlib/gridspec.py | 570 | 612| 387 | 16378 | 82635 | 
| 27 | 14 lib/matplotlib/_constrained_layout.py | 1 | 25| 200 | 16578 | 82635 | 
| 28 | 15 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 701 | 17279 | 83745 | 
| 29 | 15 lib/matplotlib/_constrained_layout.py | 27 | 131| 1005 | 18284 | 83745 | 
| 30 | 15 lib/matplotlib/figure.py | 1905 | 1926| 223 | 18507 | 83745 | 
| 31 | 15 tutorials/intermediate/constrainedlayout_guide.py | 396 | 423| 261 | 18768 | 83745 | 
| 32 | 16 examples/subplots_axes_and_figures/gridspec_nested.py | 1 | 46| 339 | 19107 | 84084 | 
| 33 | 16 lib/matplotlib/_constrained_layout.py | 405 | 434| 300 | 19407 | 84084 | 
| 34 | 16 lib/matplotlib/_constrained_layout.py | 366 | 402| 473 | 19880 | 84084 | 
| 35 | 16 lib/matplotlib/figure.py | 2092 | 2114| 156 | 20036 | 84084 | 
| 36 | 16 lib/matplotlib/gridspec.py | 326 | 390| 602 | 20638 | 84084 | 
| 37 | 16 lib/matplotlib/_constrained_layout.py | 301 | 363| 771 | 21409 | 84084 | 
| 38 | 17 lib/matplotlib/backends/backend_gtk3agg.py | 56 | 87| 289 | 21698 | 84764 | 
| 39 | 17 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 22483 | 84764 | 
| 40 | 18 examples/userdemo/demo_gridspec01.py | 1 | 30| 240 | 22723 | 85004 | 
| 41 | **19 lib/matplotlib/_layoutgrid.py** | 28 | 110| 897 | 23620 | 90389 | 
| 42 | 19 lib/matplotlib/gridspec.py | 204 | 223| 214 | 23834 | 90389 | 
| 43 | 19 lib/matplotlib/figure.py | 3131 | 3154| 185 | 24019 | 90389 | 
| 44 | 19 examples/subplots_axes_and_figures/demo_tight_layout.py | 117 | 136| 123 | 24142 | 90389 | 
| 45 | 19 lib/matplotlib/figure.py | 1396 | 1448| 532 | 24674 | 90389 | 
| 46 | 19 lib/matplotlib/figure.py | 1 | 48| 312 | 24986 | 90389 | 
| 47 | 19 lib/matplotlib/figure.py | 1352 | 1394| 272 | 25258 | 90389 | 
| 48 | 20 tutorials/provisional/mosaic.py | 170 | 295| 794 | 26052 | 92589 | 
| 49 | 21 lib/matplotlib/backends/backend_gtk3.py | 249 | 277| 281 | 26333 | 98530 | 
| 50 | 21 lib/matplotlib/_constrained_layout.py | 222 | 241| 160 | 26493 | 98530 | 
| 51 | 21 tutorials/provisional/mosaic.py | 59 | 169| 783 | 27276 | 98530 | 
| 52 | 22 lib/matplotlib/axes/_subplots.py | 1 | 86| 807 | 28083 | 100168 | 
| 53 | 22 lib/matplotlib/figure.py | 2053 | 2075| 173 | 28256 | 100168 | 
| 54 | 22 lib/matplotlib/gridspec.py | 262 | 323| 652 | 28908 | 100168 | 
| 55 | 22 lib/matplotlib/backends/backend_gtk3.py | 222 | 247| 253 | 29161 | 100168 | 
| 56 | 22 lib/matplotlib/gridspec.py | 249 | 260| 135 | 29296 | 100168 | 
| 57 | 22 lib/matplotlib/gridspec.py | 1 | 22| 137 | 29433 | 100168 | 
| 58 | 22 lib/matplotlib/figure.py | 749 | 786| 432 | 29865 | 100168 | 
| 59 | 22 lib/matplotlib/backends/backend_gtk3agg.py | 1 | 54| 416 | 30281 | 100168 | 
| 60 | 22 lib/matplotlib/_constrained_layout.py | 678 | 693| 150 | 30431 | 100168 | 
| 61 | 22 tutorials/provisional/mosaic.py | 1 | 37| 325 | 30756 | 100168 | 
| 62 | 23 lib/matplotlib/backends/backend_gtk4.py | 206 | 241| 336 | 31092 | 105451 | 
| 63 | 23 tutorials/intermediate/tight_layout_guide.py | 1 | 104| 778 | 31870 | 105451 | 
| 64 | **23 lib/matplotlib/_layoutgrid.py** | 512 | 563| 604 | 32474 | 105451 | 
| 65 | **23 lib/matplotlib/_layoutgrid.py** | 214 | 251| 489 | 32963 | 105451 | 
| 66 | 23 examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 741 | 33704 | 105451 | 
| 67 | 23 lib/matplotlib/backends/backend_gtk3.py | 157 | 203| 465 | 34169 | 105451 | 
| 68 | 23 lib/matplotlib/figure.py | 1814 | 1867| 570 | 34739 | 105451 | 
| 69 | 24 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 35264 | 105976 | 
| 70 | 24 lib/matplotlib/gridspec.py | 614 | 667| 459 | 35723 | 105976 | 
| 71 | 24 lib/matplotlib/figure.py | 1450 | 1482| 208 | 35931 | 105976 | 
| 72 | 24 lib/matplotlib/figure.py | 1670 | 1752| 719 | 36650 | 105976 | 
| 73 | 24 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 37145 | 105976 | 
| 74 | 25 examples/pyplots/auto_subplots_adjust.py | 66 | 85| 140 | 37285 | 106678 | 
| 75 | 26 lib/matplotlib/backends/backend_gtk4agg.py | 1 | 55| 404 | 37689 | 107083 | 
| 76 | 27 tutorials/toolkits/axes_grid.py | 1 | 332| 3425 | 41114 | 110508 | 
| 77 | 27 lib/matplotlib/figure.py | 1929 | 2023| 759 | 41873 | 110508 | 
| 78 | 28 lib/mpl_toolkits/axes_grid1/axes_grid.py | 411 | 573| 1606 | 43479 | 115397 | 
| 79 | 29 examples/subplots_axes_and_figures/ganged_plots.py | 1 | 41| 344 | 43823 | 115741 | 
| 80 | 29 lib/matplotlib/backends/backend_gtk4.py | 52 | 112| 472 | 44295 | 115741 | 
| 81 | **29 lib/matplotlib/_layoutgrid.py** | 1 | 25| 225 | 44520 | 115741 | 
| 82 | 29 lib/matplotlib/pyplot.py | 1437 | 1522| 702 | 45222 | 115741 | 
| 83 | 29 lib/matplotlib/pyplot.py | 1 | 85| 643 | 45865 | 115741 | 
| 84 | 30 examples/subplots_axes_and_figures/axes_box_aspect.py | 111 | 156| 344 | 46209 | 116866 | 
| 85 | 30 lib/matplotlib/backends/backend_gtk3.py | 83 | 140| 529 | 46738 | 116866 | 
| 86 | 30 lib/matplotlib/backends/backend_gtk4.py | 191 | 204| 138 | 46876 | 116866 | 
| 87 | **30 lib/matplotlib/_layoutgrid.py** | 174 | 212| 399 | 47275 | 116866 | 
| 88 | 31 lib/matplotlib/colorbar.py | 1510 | 1568| 627 | 47902 | 131115 | 
| 89 | 31 lib/mpl_toolkits/axes_grid1/axes_grid.py | 314 | 409| 939 | 48841 | 131115 | 
| 90 | 31 lib/matplotlib/pyplot.py | 294 | 319| 199 | 49040 | 131115 | 
| 91 | 31 lib/matplotlib/gridspec.py | 485 | 515| 277 | 49317 | 131115 | 
| 92 | 32 lib/mpl_toolkits/axisartist/axislines.py | 550 | 580| 198 | 49515 | 135555 | 
| 93 | 32 lib/matplotlib/figure.py | 2780 | 2814| 228 | 49743 | 135555 | 
| 94 | 32 lib/matplotlib/figure.py | 2148 | 2313| 1350 | 51093 | 135555 | 
| 95 | 33 doc/conf.py | 117 | 193| 763 | 51856 | 140414 | 
| 96 | 33 lib/matplotlib/gridspec.py | 715 | 756| 260 | 52116 | 140414 | 
| 97 | 34 examples/images_contours_and_fields/pcolormesh_grids.py | 1 | 81| 821 | 52937 | 141719 | 
| 98 | 34 lib/matplotlib/backends/backend_gtk3.py | 279 | 304| 171 | 53108 | 141719 | 
| 99 | 34 lib/matplotlib/backends/backend_gtk3.py | 1 | 40| 264 | 53372 | 141719 | 
| 100 | 34 lib/matplotlib/_constrained_layout.py | 282 | 299| 206 | 53578 | 141719 | 
| 101 | 35 examples/user_interfaces/embedding_in_gtk3_sgskip.py | 1 | 41| 246 | 53824 | 141965 | 
| 102 | 35 lib/matplotlib/figure.py | 788 | 899| 1026 | 54850 | 141965 | 
| 103 | 35 lib/matplotlib/figure.py | 627 | 644| 190 | 55040 | 141965 | 
| 104 | 36 lib/matplotlib/tight_layout.py | 322 | 351| 296 | 55336 | 145391 | 
| 105 | 37 tutorials/intermediate/artists.py | 123 | 338| 2331 | 57667 | 153095 | 
| 106 | 38 lib/mpl_toolkits/axisartist/__init__.py | 1 | 18| 181 | 57848 | 153276 | 
| 107 | 39 examples/axes_grid1/demo_fixed_size_axes.py | 1 | 48| 326 | 58174 | 153602 | 
| 108 | 39 examples/pyplots/auto_subplots_adjust.py | 1 | 46| 411 | 58585 | 153602 | 
| 109 | 40 examples/axes_grid1/demo_axes_grid2.py | 1 | 99| 783 | 59368 | 154385 | 
| 110 | 41 lib/mpl_toolkits/axes_grid/parasite_axes.py | 1 | 13| 116 | 59484 | 154501 | 
| 111 | **41 lib/matplotlib/_layoutgrid.py** | 375 | 397| 249 | 59733 | 154501 | 
| 112 | 41 lib/matplotlib/gridspec.py | 56 | 66| 141 | 59874 | 154501 | 
| 113 | 42 examples/mplot3d/subplot3d.py | 1 | 47| 367 | 60241 | 154868 | 
| 114 | 42 lib/matplotlib/figure.py | 218 | 248| 208 | 60449 | 154868 | 
| 115 | 42 lib/matplotlib/pyplot.py | 2249 | 2324| 767 | 61216 | 154868 | 
| 116 | 43 lib/matplotlib/backends/backend_wx.py | 1295 | 1313| 169 | 61385 | 167235 | 
| 117 | 44 examples/text_labels_and_annotations/label_subplots.py | 1 | 71| 603 | 61988 | 167838 | 
| 118 | **44 lib/matplotlib/_layoutgrid.py** | 112 | 125| 233 | 62221 | 167838 | 
| 119 | 44 lib/matplotlib/gridspec.py | 392 | 418| 247 | 62468 | 167838 | 
| 120 | 44 lib/matplotlib/pyplot.py | 1525 | 1584| 487 | 62955 | 167838 | 
| 121 | 45 examples/shapes_and_collections/artist_reference.py | 90 | 130| 319 | 63274 | 168997 | 
| 122 | 45 lib/matplotlib/backends/backend_gtk3.py | 721 | 757| 296 | 63570 | 168997 | 
| 123 | 45 lib/matplotlib/gridspec.py | 540 | 568| 240 | 63810 | 168997 | 
| 124 | 46 examples/axes_grid1/demo_axes_divider.py | 28 | 63| 306 | 64116 | 169842 | 
| 125 | 46 lib/mpl_toolkits/axes_grid1/axes_grid.py | 1 | 17| 119 | 64235 | 169842 | 
| 126 | 46 lib/matplotlib/figure.py | 181 | 216| 322 | 64557 | 169842 | 
| 127 | 46 lib/matplotlib/pyplot.py | 186 | 203| 141 | 64698 | 169842 | 
| 128 | 46 lib/matplotlib/backends/backend_gtk4.py | 243 | 268| 171 | 64869 | 169842 | 


### Hint

```
Not 100% sure what is going on, but pretty sure we never tested add_subfigure on arbitrary gridspec slices.  Maybe it can be made to work, but you may be stretching the limits of what is possible.  

For the layout above I'm not sure why you don't just have two columns.  But maybe that si just a simple example.  
```

## Patch

```diff
diff --git a/lib/matplotlib/_layoutgrid.py b/lib/matplotlib/_layoutgrid.py
--- a/lib/matplotlib/_layoutgrid.py
+++ b/lib/matplotlib/_layoutgrid.py
@@ -169,7 +169,8 @@ def hard_constraints(self):
                 self.solver.addConstraint(c | 'required')
 
     def add_child(self, child, i=0, j=0):
-        self.children[i, j] = child
+        # np.ix_ returns the cross product of i and j indices
+        self.children[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] = child
 
     def parent_constraints(self):
         # constraints that are due to the parent...

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_constrainedlayout.py b/lib/matplotlib/tests/test_constrainedlayout.py
--- a/lib/matplotlib/tests/test_constrainedlayout.py
+++ b/lib/matplotlib/tests/test_constrainedlayout.py
@@ -560,3 +560,10 @@ def test_suplabels():
     pos = ax.get_tightbbox(fig.canvas.get_renderer())
     assert pos.y0 > pos0.y0 + 10.0
     assert pos.x0 > pos0.x0 + 10.0
+
+
+def test_gridspec_addressing():
+    fig = plt.figure()
+    gs = fig.add_gridspec(3, 3)
+    sp = fig.add_subplot(gs[0:, 1:])
+    fig.draw_without_rendering()
diff --git a/lib/matplotlib/tests/test_figure.py b/lib/matplotlib/tests/test_figure.py
--- a/lib/matplotlib/tests/test_figure.py
+++ b/lib/matplotlib/tests/test_figure.py
@@ -1073,6 +1073,7 @@ def test_subfigure_spanning():
         fig.add_subfigure(gs[0, 0]),
         fig.add_subfigure(gs[0:2, 1]),
         fig.add_subfigure(gs[2, 1:3]),
+        fig.add_subfigure(gs[0:, 1:])
     ]
 
     w = 640
@@ -1086,6 +1087,12 @@ def test_subfigure_spanning():
     np.testing.assert_allclose(sub_figs[2].bbox.min, [w / 3, 0])
     np.testing.assert_allclose(sub_figs[2].bbox.max, [w, h / 3])
 
+    # check here that slicing actually works.  Last sub_fig
+    # with open slices failed, but only on draw...
+    for i in range(4):
+        sub_figs[i].add_subplot()
+    fig.draw_without_rendering()
+
 
 @mpl.style.context('mpl20')
 def test_subfigure_ticks():

```


## Code snippets

### 1 - examples/subplots_axes_and_figures/subfigures.py:

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
### 2 - tutorials/intermediate/gridspec.py:

Start line: 1, End line: 78

```python
"""
=============================================================
Customizing Figure Layouts Using GridSpec and Other Functions
=============================================================

How to create grid-shaped combinations of axes.

`~matplotlib.pyplot.subplots`
    The primary function used to create figures and axes.  It is similar to
    `.pyplot.subplot`, but creates and places all axes on the figure at once.
    See also `.Figure.subplots`.

`~matplotlib.gridspec.GridSpec`
    Specifies the geometry of the grid that a subplot will be
    placed. The number of rows and number of columns of the grid
    need to be set. Optionally, the subplot layout parameters
    (e.g., left, right, etc.) can be tuned.

`~matplotlib.gridspec.SubplotSpec`
    Specifies the location of the subplot in the given `.GridSpec`.

`~matplotlib.pyplot.subplot2grid`
    A helper function that is similar to `.pyplot.subplot`,
    but uses 0-based indexing and let subplot to occupy multiple cells.
    This function is not covered in this tutorial.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

############################################################################
# Basic Quickstart Guide
# ======================
#
# These first two examples show how to create a basic 2-by-2 grid using
# both :func:`~matplotlib.pyplot.subplots` and :mod:`~matplotlib.gridspec`.
#
# Using :func:`~matplotlib.pyplot.subplots` is quite simple.
# It returns a :class:`~matplotlib.figure.Figure` instance and an array of
# :class:`~matplotlib.axes.Axes` objects.

fig1, f1_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

############################################################################
# For a simple use case such as this, :mod:`~matplotlib.gridspec` is
# perhaps overly verbose.
# You have to create the figure and :class:`~matplotlib.gridspec.GridSpec`
# instance separately, then pass elements of gridspec instance to the
# :func:`~matplotlib.figure.Figure.add_subplot` method to create the axes
# objects.
# The elements of the gridspec are accessed in generally the same manner as
# numpy arrays.

fig2 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
f2_ax1 = fig2.add_subplot(spec2[0, 0])
f2_ax2 = fig2.add_subplot(spec2[0, 1])
f2_ax3 = fig2.add_subplot(spec2[1, 0])
f2_ax4 = fig2.add_subplot(spec2[1, 1])

#############################################################################
# The power of gridspec comes in being able to create subplots that span
# rows and columns.  Note the `NumPy slice syntax
# <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
# for selecting the part of the gridspec each subplot will occupy.
#
# Note that we have also used the convenience method `.Figure.add_gridspec`
# instead of `.gridspec.GridSpec`, potentially saving the user an import,
# and keeping the namespace cleaner.

fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 3)
f3_ax1 = fig3.add_subplot(gs[0, :])
f3_ax1.set_title('gs[0, :]')
f3_ax2 = fig3.add_subplot(gs[1, :-1])
f3_ax2.set_title('gs[1, :-1]')
f3_ax3 = fig3.add_subplot(gs[1:, -1])
f3_ax3.set_title('gs[1:, -1]')
```
### 3 - examples/subplots_axes_and_figures/gridspec_multicolumn.py:

Start line: 1, End line: 34

```python
"""
=======================================================
Using Gridspec to make multi-column/row subplot layouts
=======================================================

`.GridSpec` is a flexible way to layout
subplot grids.  Here is an example with a 3x3 grid, and
axes spanning all three columns, two columns, and two rows.

"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

fig = plt.figure(constrained_layout=True)

gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])
ax4 = fig.add_subplot(gs[-1, 0])
ax5 = fig.add_subplot(gs[-1, -2])

fig.suptitle("GridSpec")
format_axes(fig)

plt.show()
```
### 4 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 591, End line: 676

```python
fig, ax = plt.subplots(constrained_layout=True)
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

fig, ax = plt.subplots(1, 2, constrained_layout=True)
example_plot(ax[0], fontsize=32)
example_plot(ax[1], fontsize=8)
plot_children(fig, printit=False)

#######################################################################
# Two Axes and colorbar
# ---------------------
#
# A colorbar is simply another item that expands the margin of the parent
# layoutgrid cell:

fig, ax = plt.subplots(1, 2, constrained_layout=True)
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

fig, axs = plt.subplots(2, 2, constrained_layout=True)
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
# of the left-hand axes.  This is consietent with how ``gridspec`` works
# without constrained layout.

fig = plt.figure(constrained_layout=True)
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

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 4)
ax00 = fig.add_subplot(gs[0, 0:2])
ax01 = fig.add_subplot(gs[0, 2:])
ax10 = fig.add_subplot(gs[1, 1:3])
example_plot(ax10, fontsize=14)
plot_children(fig)
plt.show()
```
### 5 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 269, End line: 350

```python
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                                wspace=0.2)

##########################################
# GridSpecs also have optional *hspace* and *wspace* keyword arguments,
# that will be used instead of the pads set by ``constrained_layout``:

fig, axs = plt.subplots(2, 2, constrained_layout=True,
                        gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
# this has no effect because the space set in the gridspec trumps the
# space set in constrained_layout.
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0,
                                wspace=0.0)
plt.show()

##########################################
# Spacing with colorbars
# -----------------------
#
# Colorbars are placed a distance *pad* from their parent, where *pad*
# is a fraction of the width of the parent(s).  The spacing to the
# next subplot is then given by *w/hspace*.

fig, axs = plt.subplots(2, 2, constrained_layout=True)
pads = [0, 0.05, 0.1, 0.2]
for pad, ax in zip(pads, axs.flat):
    pc = ax.pcolormesh(arr, **pc_kwargs)
    fig.colorbar(pc, ax=ax, shrink=0.6, pad=pad)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'pad: {pad}')
fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,
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
# with :func:`~matplotlib.figure.Figure.subplots` or
# :func:`~matplotlib.gridspec.GridSpec` and
# :func:`~matplotlib.figure.Figure.add_subplot`.
#
# Note that in what follows ``constrained_layout=True``

fig = plt.figure()

gs1 = gridspec.GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])

example_plot(ax1)
example_plot(ax2)

###############################################################################
# More complicated gridspec layouts are possible.  Note here we use the
# convenience functions `~.Figure.add_gridspec` and
# `~.SubplotSpec.subgridspec`.

fig = plt.figure()
```
### 6 - tutorials/intermediate/gridspec.py:

Start line: 79, End line: 136

```python
f3_ax4 = fig3.add_subplot(gs[-1, 0])
f3_ax4.set_title('gs[-1, 0]')
f3_ax5 = fig3.add_subplot(gs[-1, -2])
f3_ax5.set_title('gs[-1, -2]')

#############################################################################
# :mod:`~matplotlib.gridspec` is also indispensable for creating subplots
# of different widths via a couple of methods.
#
# The method shown here is similar to the one above and initializes a
# uniform grid specification,
# and then uses numpy indexing and slices to allocate multiple
# "cells" for a given subplot.

fig4 = plt.figure(constrained_layout=True)
spec4 = fig4.add_gridspec(ncols=2, nrows=2)
anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center')

f4_ax1 = fig4.add_subplot(spec4[0, 0])
f4_ax1.annotate('GridSpec[0, 0]', **anno_opts)
fig4.add_subplot(spec4[0, 1]).annotate('GridSpec[0, 1:]', **anno_opts)
fig4.add_subplot(spec4[1, 0]).annotate('GridSpec[1:, 0]', **anno_opts)
fig4.add_subplot(spec4[1, 1]).annotate('GridSpec[1:, 1:]', **anno_opts)

############################################################################
# Another option is to use the ``width_ratios`` and ``height_ratios``
# parameters. These keyword arguments are lists of numbers.
# Note that absolute values are meaningless, only their relative ratios
# matter. That means that ``width_ratios=[2, 4, 8]`` is equivalent to
# ``width_ratios=[1, 2, 4]`` within equally wide figures.
# For the sake of demonstration, we'll blindly create the axes within
# ``for`` loops since we won't need them later.

fig5 = plt.figure(constrained_layout=True)
widths = [2, 3, 1.5]
heights = [1, 3, 2]
spec5 = fig5.add_gridspec(ncols=3, nrows=3, width_ratios=widths,
                          height_ratios=heights)
for row in range(3):
    for col in range(3):
        ax = fig5.add_subplot(spec5[row, col])
        label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
        ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')

############################################################################
# Learning to use ``width_ratios`` and ``height_ratios`` is particularly
# useful since the top-level function :func:`~matplotlib.pyplot.subplots`
# accepts them within the ``gridspec_kw`` parameter.
# For that matter, any parameter accepted by
# :class:`~matplotlib.gridspec.GridSpec` can be passed to
# :func:`~matplotlib.pyplot.subplots` via the ``gridspec_kw`` parameter.
# This example recreates the previous figure without directly using a
# gridspec instance.

gs_kw = dict(width_ratios=widths, height_ratios=heights)
fig6, f6_axes = plt.subplots(ncols=3, nrows=3, constrained_layout=True,
                             gridspec_kw=gs_kw)
```
### 7 - tutorials/intermediate/gridspec.py:

Start line: 137, End line: 206

```python
for r, row in enumerate(f6_axes):
    for c, ax in enumerate(row):
        label = 'Width: {}\nHeight: {}'.format(widths[c], heights[r])
        ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')

############################################################################
# The ``subplots`` and ``get_gridspec`` methods can be combined since it is
# sometimes more convenient to make most of the subplots using ``subplots``
# and then remove some and combine them.  Here we create a layout with
# the bottom two axes in the last column combined.

fig7, f7_axs = plt.subplots(ncols=3, nrows=3)
gs = f7_axs[1, 2].get_gridspec()
# remove the underlying axes
for ax in f7_axs[1:, -1]:
    ax.remove()
axbig = fig7.add_subplot(gs[1:, -1])
axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
               xycoords='axes fraction', va='center')

fig7.tight_layout()

###############################################################################
# Fine Adjustments to a Gridspec Layout
# =====================================
#
# When a GridSpec is explicitly used, you can adjust the layout
# parameters of subplots that are created from the GridSpec.  Note this
# option is not compatible with ``constrained_layout`` or
# `.Figure.tight_layout` which both adjust subplot sizes to fill the
# figure.

fig8 = plt.figure(constrained_layout=False)
gs1 = fig8.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)
f8_ax1 = fig8.add_subplot(gs1[:-1, :])
f8_ax2 = fig8.add_subplot(gs1[-1, :-1])
f8_ax3 = fig8.add_subplot(gs1[-1, -1])

###############################################################################
# This is similar to :func:`~matplotlib.pyplot.subplots_adjust`, but it only
# affects the subplots that are created from the given GridSpec.
#
# For example, compare the left and right sides of this figure:

fig9 = plt.figure(constrained_layout=False)
gs1 = fig9.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48,
                        wspace=0.05)
f9_ax1 = fig9.add_subplot(gs1[:-1, :])
f9_ax2 = fig9.add_subplot(gs1[-1, :-1])
f9_ax3 = fig9.add_subplot(gs1[-1, -1])

gs2 = fig9.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.98,
                        hspace=0.05)
f9_ax4 = fig9.add_subplot(gs2[:, :-1])
f9_ax5 = fig9.add_subplot(gs2[:-1, -1])
f9_ax6 = fig9.add_subplot(gs2[-1, -1])

###############################################################################
# GridSpec using SubplotSpec
# ==========================
#
# You can create GridSpec from the :class:`~matplotlib.gridspec.SubplotSpec`,
# in which case its layout parameters are set to that of the location of
# the given SubplotSpec.
#
# Note this is also available from the more verbose
# `.gridspec.GridSpecFromSubplotSpec`.

fig10 = plt.figure(constrained_layout=True)
gs0 = fig10.add_gridspec(1, 2)
```
### 8 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 426, End line: 589

```python
docomplicated()

###############################################################################
# Manually setting axes positions
# ================================
#
# There can be good reasons to manually set an axes position.  A manual call
# to `~.axes.Axes.set_position` will set the axes so constrained_layout has
# no effect on it anymore. (Note that ``constrained_layout`` still leaves the
# space for the axes that is moved).

fig, axs = plt.subplots(1, 2)
example_plot(axs[0], fontsize=12)
axs[1].set_position([0.2, 0.2, 0.4, 0.4])

###############################################################################
# Manually turning off ``constrained_layout``
# ===========================================
#
# ``constrained_layout`` usually adjusts the axes positions on each draw
# of the figure.  If you want to get the spacing provided by
# ``constrained_layout`` but not have it update, then do the initial
# draw and then call ``fig.set_constrained_layout(False)``.
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

fig = plt.figure()

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

fig = plt.figure()

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

fig = plt.figure()

ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
fig.suptitle('subplot2grid')
plt.show()

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
# has some complexity due to the complex ways we can layout a figure.
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
# in that row are accommodated.  Similarly for columns and the left/right
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
### 9 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 352, End line: 393

```python
gs0 = fig.add_gridspec(1, 2)

gs1 = gs0[0].subgridspec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])

example_plot(ax1)
example_plot(ax2)

gs2 = gs0[1].subgridspec(3, 1)

for ss in gs2:
    ax = fig.add_subplot(ss)
    example_plot(ax)
    ax.set_title("")
    ax.set_xlabel("")

ax.set_xlabel("x-label", fontsize=12)

############################################################################
# Note that in the above the left and right columns don't have the same
# vertical extent.  If we want the top and bottom of the two grids to line up
# then they need to be in the same gridspec.  We need to make this figure
# larger as well in order for the axes not to collapse to zero height:

fig = plt.figure(figsize=(4, 6))

gs0 = fig.add_gridspec(6, 2)

ax1 = fig.add_subplot(gs0[:3, 0])
ax2 = fig.add_subplot(gs0[3:, 0])

example_plot(ax1)
example_plot(ax2)

ax = fig.add_subplot(gs0[0:2, 1])
example_plot(ax, hide_labels=True)
ax = fig.add_subplot(gs0[2:4, 1])
example_plot(ax, hide_labels=True)
ax = fig.add_subplot(gs0[4:, 1])
example_plot(ax, hide_labels=True)
fig.suptitle('Overlapping Gridspecs')
```
### 10 - lib/matplotlib/figure.py:

Start line: 1868, End line: 1903

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
                # element is an axes or a nested mosaic.
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
### 41 - lib/matplotlib/_layoutgrid.py:

Start line: 28, End line: 110

```python
class LayoutGrid:
    """
    Analogous to a gridspec, and contained in another LayoutGrid.
    """

    def __init__(self, parent=None, parent_pos=(0, 0),
                 parent_inner=False, name='', ncols=1, nrows=1,
                 h_pad=None, w_pad=None, width_ratios=None,
                 height_ratios=None):
        Variable = kiwi.Variable
        self.parent = parent
        self.parent_pos = parent_pos
        self.parent_inner = parent_inner
        self.name = name + seq_id()
        if parent is not None:
            self.name = f'{parent.name}.{self.name}'
        self.nrows = nrows
        self.ncols = ncols
        self.height_ratios = np.atleast_1d(height_ratios)
        if height_ratios is None:
            self.height_ratios = np.ones(nrows)
        self.width_ratios = np.atleast_1d(width_ratios)
        if width_ratios is None:
            self.width_ratios = np.ones(ncols)

        sn = self.name + '_'
        if parent is None:
            self.parent = None
            self.solver = kiwi.Solver()
        else:
            self.parent = parent
            parent.add_child(self, *parent_pos)
            self.solver = self.parent.solver
        # keep track of artist associated w/ this layout.  Can be none
        self.artists = np.empty((nrows, ncols), dtype=object)
        self.children = np.empty((nrows, ncols), dtype=object)

        self.margins = {}
        self.margin_vals = {}
        # all the boxes in each column share the same left/right margins:
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            # track the value so we can change only if a margin is larger
            # than the current value
            self.margin_vals[todo] = np.zeros(ncols)

        sol = self.solver

        # These are redundant, but make life easier if
        # we define them all.  All that is really
        # needed is left/right, margin['left'], and margin['right']
        self.widths = [Variable(f'{sn}widths[{i}]') for i in range(ncols)]
        self.lefts = [Variable(f'{sn}lefts[{i}]') for i in range(ncols)]
        self.rights = [Variable(f'{sn}rights[{i}]') for i in range(ncols)]
        self.inner_widths = [Variable(f'{sn}inner_widths[{i}]')
                             for i in range(ncols)]
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(ncols)]
            for i in range(ncols):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = np.empty((nrows), dtype=object)
            self.margin_vals[todo] = np.zeros(nrows)

        self.heights = [Variable(f'{sn}heights[{i}]') for i in range(nrows)]
        self.inner_heights = [Variable(f'{sn}inner_heights[{i}]')
                              for i in range(nrows)]
        self.bottoms = [Variable(f'{sn}bottoms[{i}]') for i in range(nrows)]
        self.tops = [Variable(f'{sn}tops[{i}]') for i in range(nrows)]
        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(nrows)]
            for i in range(nrows):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        # set these margins to zero by default. They will be edited as
        # children are filled.
        self.reset_margins()
        self.add_constraints()

        self.h_pad = h_pad
        self.w_pad = w_pad
```
### 64 - lib/matplotlib/_layoutgrid.py:

Start line: 512, End line: 563

```python
def plot_children(fig, lg=None, level=0, printit=False):
    """Simple plotting to show where boxes are."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if lg is None:
        _layoutgrids = fig.execute_constrained_layout()
        lg = _layoutgrids[fig]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    col = colors[level]
    for i in range(lg.nrows):
        for j in range(lg.ncols):
            bb = lg.get_outer_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=1,
                                   edgecolor='0.7', facecolor='0.7',
                                   alpha=0.2, transform=fig.transFigure,
                                   zorder=-3))
            bbi = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=2,
                                   edgecolor=col, facecolor='none',
                                   transform=fig.transFigure, zorder=-2))

            bbi = lg.get_left_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.7, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_right_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.5, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_bottom_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.5, 0.7],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_top_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.2, 0.7],
                                   transform=fig.transFigure, zorder=-2))
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level+1)
```
### 65 - lib/matplotlib/_layoutgrid.py:

Start line: 214, End line: 251

```python
class LayoutGrid:

    def grid_constraints(self):
        # constrain the ratio of the inner part of the grids
        # to be the same (relative to width_ratios)

        # constrain widths:
        w = (self.rights[0] - self.margins['right'][0] -
             self.margins['rightcb'][0])
        w = (w - self.lefts[0] - self.margins['left'][0] -
             self.margins['leftcb'][0])
        w0 = w / self.width_ratios[0]
        # from left to right
        for i in range(1, self.ncols):
            w = (self.rights[i] - self.margins['right'][i] -
                 self.margins['rightcb'][i])
            w = (w - self.lefts[i] - self.margins['left'][i] -
                 self.margins['leftcb'][i])
            c = (w == w0 * self.width_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly next to each other.
            c = (self.rights[i - 1] == self.lefts[i])
            self.solver.addConstraint(c | 'strong')

        # constrain heights:
        h = self.tops[0] - self.margins['top'][0] - self.margins['topcb'][0]
        h = (h - self.bottoms[0] - self.margins['bottom'][0] -
             self.margins['bottomcb'][0])
        h0 = h / self.height_ratios[0]
        # from top to bottom:
        for i in range(1, self.nrows):
            h = (self.tops[i] - self.margins['top'][i] -
                 self.margins['topcb'][i])
            h = (h - self.bottoms[i] - self.margins['bottom'][i] -
                 self.margins['bottomcb'][i])
            c = (h == h0 * self.height_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly above each other.
            c = (self.bottoms[i - 1] == self.tops[i])
            self.solver.addConstraint(c | 'strong')
```
### 81 - lib/matplotlib/_layoutgrid.py:

Start line: 1, End line: 25

```python
"""
A layoutgrid is a nrows by ncols set of boxes, meant to be used by
`._constrained_layout`, each box is analogous to a subplotspec element of
a gridspec.

Each box is defined by left[ncols], right[ncols], bottom[nrows] and top[nrows],
and by two editable margins for each side.  The main margin gets its value
set by the size of ticklabels, titles, etc on each axes that is in the figure.
The outer margin is the padding around the axes, and space for any
colorbars.

The "inner" widths and heights of these boxes are then constrained to be the
same (relative the values of `width_ratios[ncols]` and `height_ratios[nrows]`).

The layoutgrid is then constrained to be contained within a parent layoutgrid,
its column(s) and row(s) specified when it is created.
"""

import itertools
import kiwisolver as kiwi
import logging
import numpy as np
from matplotlib.transforms import Bbox

_log = logging.getLogger(__name__)
```
### 87 - lib/matplotlib/_layoutgrid.py:

Start line: 174, End line: 212

```python
class LayoutGrid:

    def parent_constraints(self):
        # constraints that are due to the parent...
        # i.e. the first column's left is equal to the
        # parent's left, the last column right equal to the
        # parent's right...
        parent = self.parent
        if parent is None:
            hc = [self.lefts[0] == 0,
                  self.rights[-1] == 1,
                  # top and bottom reversed order...
                  self.tops[0] == 1,
                  self.bottoms[-1] == 0]
        else:
            rows, cols = self.parent_pos
            rows = np.atleast_1d(rows)
            cols = np.atleast_1d(cols)

            left = parent.lefts[cols[0]]
            right = parent.rights[cols[-1]]
            top = parent.tops[rows[0]]
            bottom = parent.bottoms[rows[-1]]
            if self.parent_inner:
                # the layout grid is contained inside the inner
                # grid of the parent.
                left += parent.margins['left'][cols[0]]
                left += parent.margins['leftcb'][cols[0]]
                right -= parent.margins['right'][cols[-1]]
                right -= parent.margins['rightcb'][cols[-1]]
                top -= parent.margins['top'][rows[0]]
                top -= parent.margins['topcb'][rows[0]]
                bottom += parent.margins['bottom'][rows[-1]]
                bottom += parent.margins['bottomcb'][rows[-1]]
            hc = [self.lefts[0] == left,
                  self.rights[-1] == right,
                  # from top to bottom
                  self.tops[0] == top,
                  self.bottoms[-1] == bottom]
        for c in hc:
            self.solver.addConstraint(c | 'required')
```
### 111 - lib/matplotlib/_layoutgrid.py:

Start line: 375, End line: 397

```python
class LayoutGrid:

    def get_inner_bbox(self, rows=0, cols=0):
        """
        Return the inner bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['left'][cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value() +
                self.margins['bottomcb'][rows[-1]].value()),
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            (self.tops[rows[0]].value() -
                self.margins['top'][rows[0]].value() -
                self.margins['topcb'][rows[0]].value())
        )
        return bbox
```
### 118 - lib/matplotlib/_layoutgrid.py:

Start line: 112, End line: 125

```python
class LayoutGrid:

    def __repr__(self):
        str = f'LayoutBox: {self.name:25s} {self.nrows}x{self.ncols},\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                str += f'{i}, {j}: '\
                       f'L({self.lefts[j].value():1.3f}, ' \
                       f'B{self.bottoms[i].value():1.3f}, ' \
                       f'W{self.widths[j].value():1.3f}, ' \
                       f'H{self.heights[i].value():1.3f}, ' \
                       f'innerW{self.inner_widths[j].value():1.3f}, ' \
                       f'innerH{self.inner_heights[i].value():1.3f}, ' \
                       f'ML{self.margins["left"][j].value():1.3f}, ' \
                       f'MR{self.margins["right"][j].value():1.3f}, \n'
        return str
```
