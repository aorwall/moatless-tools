# matplotlib__matplotlib-25960

| **matplotlib/matplotlib** | `1d0d255b79e84dfc9f2123c5eb85a842d342f72b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 10939 |
| **Any found context length** | 10939 |
| **Avg pos** | 38.0 |
| **Min pos** | 19 |
| **Max pos** | 19 |
| **Top file pos** | 4 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1564,8 +1564,9 @@ def subfigures(self, nrows=1, ncols=1, squeeze=True,
         wspace, hspace : float, default: None
             The amount of width/height reserved for space between subfigures,
             expressed as a fraction of the average subfigure width/height.
-            If not given, the values will be inferred from a figure or
-            rcParams when necessary.
+            If not given, the values will be inferred from rcParams if using
+            constrained layout (see `~.ConstrainedLayoutEngine`), or zero if
+            not using a layout engine.
 
         width_ratios : array-like of length *ncols*, optional
             Defines the relative widths of the columns. Each column gets a
@@ -1580,13 +1581,24 @@ def subfigures(self, nrows=1, ncols=1, squeeze=True,
         gs = GridSpec(nrows=nrows, ncols=ncols, figure=self,
                       wspace=wspace, hspace=hspace,
                       width_ratios=width_ratios,
-                      height_ratios=height_ratios)
+                      height_ratios=height_ratios,
+                      left=0, right=1, bottom=0, top=1)
 
         sfarr = np.empty((nrows, ncols), dtype=object)
         for i in range(ncols):
             for j in range(nrows):
                 sfarr[j, i] = self.add_subfigure(gs[j, i], **kwargs)
 
+        if self.get_layout_engine() is None and (wspace is not None or
+                                                 hspace is not None):
+            # Gridspec wspace and hspace is ignored on subfigure instantiation,
+            # and no space is left.  So need to account for it here if required.
+            bottoms, tops, lefts, rights = gs.get_grid_positions(self)
+            for sfrow, bottom, top in zip(sfarr, bottoms, tops):
+                for sf, left, right in zip(sfrow, lefts, rights):
+                    bbox = Bbox.from_extents(left, bottom, right, top)
+                    sf._redo_transform_rel_fig(bbox=bbox)
+
         if squeeze:
             # Discarding unneeded dimensions that equal 1.  If we only have one
             # subfigure, just return it instead of a 1-element array.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 1567 | 1568 | 19 | 4 | 10939
| lib/matplotlib/figure.py | 1583 | 1583 | 19 | 4 | 10939


## Problem Statement

```
[Bug]: wspace and hspace in subfigures not working
### Bug summary

`wspace` and `hspace` in `Figure.subfigures` do nothing.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt

figs = plt.figure().subfigures(2, 2, wspace=0, hspace=0)
for fig in figs.flat:
    fig.subplots().plot([1, 2])
plt.show()
\`\`\`


### Actual outcome

Same figure independently of the values of hspace and wspace.

### Expected outcome

https://github.com/matplotlib/matplotlib/blob/b3bd929cf07ea35479fded8f739126ccc39edd6d/lib/matplotlib/figure.py#L1550-L1554

### Additional information

_No response_

### Operating system

OS/X

### Matplotlib Version

3.7.1

### Matplotlib Backend

MacOSX

### Python version

Python 3.10.9

### Jupyter version

_No response_

### Installation

conda

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 galleries/examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 667 | 1426 | 
| 2 | 1 galleries/examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 759 | 1426 | 1426 | 
| 3 | 2 galleries/users_explain/axes/constrainedlayout_guide.py | 184 | 268| 863 | 2289 | 8136 | 
| 4 | 2 galleries/users_explain/axes/constrainedlayout_guide.py | 270 | 351| 845 | 3134 | 8136 | 
| 5 | 3 galleries/examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 444 | 3578 | 10117 | 
| 6 | **4 lib/matplotlib/figure.py** | 2289 | 2311| 162 | 3740 | 39796 | 
| 7 | 4 galleries/users_explain/axes/constrainedlayout_guide.py | 442 | 527| 795 | 4535 | 39796 | 
| 8 | 4 galleries/examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 791 | 5326 | 39796 | 
| 9 | **4 lib/matplotlib/figure.py** | 2331 | 2352| 149 | 5475 | 39796 | 
| 10 | 5 galleries/users_explain/axes/arranging_axes.py | 342 | 437| 927 | 6402 | 43922 | 
| 11 | 6 galleries/examples/subplots_axes_and_figures/ganged_plots.py | 1 | 41| 345 | 6747 | 44267 | 
| 12 | **6 lib/matplotlib/figure.py** | 917 | 932| 245 | 6992 | 44267 | 
| 13 | 7 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 69 | 88| 140 | 7132 | 44967 | 
| 14 | 8 galleries/examples/subplots_axes_and_figures/figure_size_units.py | 1 | 83| 629 | 7761 | 45596 | 
| 15 | 8 galleries/users_explain/axes/arranging_axes.py | 186 | 260| 795 | 8556 | 45596 | 
| 16 | 9 galleries/examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 116| 762 | 9318 | 46470 | 
| 17 | 9 galleries/users_explain/axes/arranging_axes.py | 101 | 185| 854 | 10172 | 46470 | 
| 18 | 10 galleries/examples/subplots_axes_and_figures/subplots_adjust.py | 1 | 32| 207 | 10379 | 46677 | 
| **-> 19 <-** | **10 lib/matplotlib/figure.py** | 1545 | 1600| 560 | 10939 | 46677 | 
| 20 | 11 galleries/examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 695 | 11634 | 47781 | 
| 21 | 12 galleries/examples/mplot3d/subplot3d.py | 1 | 46| 367 | 12001 | 48148 | 
| 22 | 13 galleries/users_explain/axes/tight_layout_guide.py | 1 | 108| 798 | 12799 | 50324 | 
| 23 | 13 galleries/users_explain/axes/tight_layout_guide.py | 109 | 225| 879 | 13678 | 50324 | 
| 24 | 14 lib/matplotlib/_constrained_layout.py | 111 | 149| 471 | 14149 | 57887 | 
| 25 | 15 galleries/examples/mplot3d/mixed_subplots.py | 1 | 47| 356 | 14505 | 58243 | 
| 26 | 16 galleries/tutorials/pyplot.py | 257 | 335| 909 | 15414 | 62731 | 
| 27 | 16 galleries/users_explain/axes/constrainedlayout_guide.py | 352 | 441| 761 | 16175 | 62731 | 
| 28 | 16 galleries/users_explain/axes/constrainedlayout_guide.py | 529 | 648| 1131 | 17306 | 62731 | 
| 29 | 17 galleries/users_explain/axes/mosaic.py | 169 | 289| 785 | 18091 | 65437 | 
| 30 | **17 lib/matplotlib/figure.py** | 1 | 69| 493 | 18584 | 65437 | 
| 31 | 18 lib/matplotlib/pyplot.py | 739 | 1567| 5724 | 24308 | 99518 | 
| 32 | 18 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 52 | 66| 151 | 24459 | 99518 | 
| 33 | 18 lib/matplotlib/_constrained_layout.py | 443 | 479| 473 | 24932 | 99518 | 
| 34 | **18 lib/matplotlib/figure.py** | 155 | 177| 205 | 25137 | 99518 | 
| 35 | **18 lib/matplotlib/figure.py** | 391 | 399| 154 | 25291 | 99518 | 
| 36 | 19 galleries/examples/text_labels_and_annotations/label_subplots.py | 1 | 72| 606 | 25897 | 100124 | 
| 37 | **19 lib/matplotlib/figure.py** | 2141 | 2239| 781 | 26678 | 100124 | 
| 38 | **19 lib/matplotlib/figure.py** | 2863 | 2904| 321 | 26999 | 100124 | 
| 39 | 20 galleries/examples/subplots_axes_and_figures/subplot.py | 1 | 52| 345 | 27344 | 100469 | 
| 40 | 20 galleries/examples/subplots_axes_and_figures/demo_tight_layout.py | 118 | 135| 112 | 27456 | 100469 | 
| 41 | 20 galleries/users_explain/axes/constrainedlayout_guide.py | 650 | 735| 780 | 28236 | 100469 | 
| 42 | 20 galleries/examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 746 | 28982 | 100469 | 
| 43 | 21 galleries/examples/lines_bars_and_markers/vline_hline_demo.py | 1 | 35| 295 | 29277 | 100764 | 
| 44 | **21 lib/matplotlib/figure.py** | 1935 | 1960| 316 | 29593 | 100764 | 
| 45 | 22 galleries/examples/userdemo/demo_gridspec03.py | 1 | 53| 422 | 30015 | 101186 | 
| 46 | 22 galleries/users_explain/axes/mosaic.py | 57 | 168| 799 | 30814 | 101186 | 
| 47 | 22 galleries/tutorials/pyplot.py | 96 | 255| 1523 | 32337 | 101186 | 
| 48 | **22 lib/matplotlib/figure.py** | 2074 | 2113| 401 | 32738 | 101186 | 
| 49 | 23 galleries/users_explain/quick_start.py | 489 | 575| 1016 | 33754 | 107193 | 
| 50 | **23 lib/matplotlib/figure.py** | 2115 | 2138| 266 | 34020 | 107193 | 
| 51 | 24 galleries/users_explain/text/text_intro.py | 180 | 268| 762 | 34782 | 111070 | 
| 52 | 25 galleries/users_explain/text/annotations.py | 641 | 714| 793 | 35575 | 118467 | 
| 53 | 25 galleries/users_explain/axes/arranging_axes.py | 261 | 341| 743 | 36318 | 118467 | 
| 54 | 26 galleries/examples/text_labels_and_annotations/align_ylabels.py | 41 | 87| 346 | 36664 | 119114 | 
| 55 | 26 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 1 | 49| 409 | 37073 | 119114 | 
| 56 | 27 lib/matplotlib/backends/backend_wx.py | 404 | 462| 682 | 37755 | 130810 | 
| 57 | 27 lib/matplotlib/backends/backend_wx.py | 877 | 907| 271 | 38026 | 130810 | 
| 58 | 28 galleries/examples/text_labels_and_annotations/figlegend_demo.py | 1 | 54| 511 | 38537 | 131321 | 
| 59 | **28 lib/matplotlib/figure.py** | 1602 | 1627| 148 | 38685 | 131321 | 
| 60 | 28 galleries/users_explain/axes/constrainedlayout_guide.py | 106 | 183| 750 | 39435 | 131321 | 
| 61 | 28 lib/matplotlib/backends/backend_wx.py | 672 | 685| 143 | 39578 | 131321 | 
| 62 | 29 galleries/examples/subplots_axes_and_figures/axes_box_aspect.py | 110 | 156| 366 | 39944 | 132448 | 
| 63 | 30 galleries/examples/subplots_axes_and_figures/axhspan_demo.py | 1 | 37| 369 | 40313 | 132817 | 
| 64 | 31 lib/matplotlib/_mathtext.py | 1156 | 1218| 577 | 40890 | 156499 | 
| 65 | 31 lib/matplotlib/_constrained_layout.py | 300 | 335| 391 | 41281 | 156499 | 
| 66 | 32 galleries/examples/subplots_axes_and_figures/multiple_figs_demo.py | 1 | 52| 317 | 41598 | 156816 | 
| 67 | **32 lib/matplotlib/figure.py** | 1283 | 1313| 413 | 42011 | 156816 | 
| 68 | 33 galleries/examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 42536 | 157341 | 
| 69 | 34 galleries/examples/showcase/anatomy.py | 85 | 122| 453 | 42989 | 158581 | 
| 70 | **34 lib/matplotlib/figure.py** | 758 | 796| 446 | 43435 | 158581 | 
| 71 | 35 galleries/examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 425 | 43860 | 159006 | 
| 72 | 36 galleries/users_explain/axes/colorbar_placement.py | 1 | 85| 764 | 44624 | 159901 | 
| 73 | 37 galleries/examples/lines_bars_and_markers/filled_step.py | 180 | 238| 493 | 45117 | 161527 | 
| 74 | 37 lib/matplotlib/_constrained_layout.py | 359 | 423| 780 | 45897 | 161527 | 
| 75 | 38 galleries/users_explain/artists/transforms_tutorial.py | 259 | 356| 1006 | 46903 | 167889 | 
| 76 | **38 lib/matplotlib/figure.py** | 2313 | 2329| 149 | 47052 | 167889 | 
| 77 | 39 galleries/examples/mplot3d/wire3d_zero_stride.py | 1 | 30| 232 | 47284 | 168121 | 
| 78 | 40 lib/matplotlib/_tight_layout.py | 96 | 157| 738 | 48022 | 171050 | 
| 79 | 41 galleries/examples/subplots_axes_and_figures/figure_title.py | 1 | 54| 456 | 48478 | 171506 | 
| 80 | **41 lib/matplotlib/figure.py** | 3345 | 3382| 363 | 48841 | 171506 | 
| 81 | 42 galleries/examples/axisartist/demo_floating_axes.py | 149 | 168| 186 | 49027 | 172904 | 
| 82 | 42 galleries/users_explain/axes/constrainedlayout_guide.py | 1 | 105| 783 | 49810 | 172904 | 
| 83 | 42 lib/matplotlib/_constrained_layout.py | 424 | 440| 233 | 50043 | 172904 | 
| 84 | 43 galleries/examples/pyplots/pyplot_two_subplots.py | 1 | 38| 220 | 50263 | 173124 | 
| 85 | 44 galleries/examples/subplots_axes_and_figures/custom_figure_class.py | 36 | 53| 115 | 50378 | 173495 | 
| 86 | 44 lib/matplotlib/_constrained_layout.py | 338 | 357| 214 | 50592 | 173495 | 
| 87 | **44 lib/matplotlib/figure.py** | 634 | 655| 230 | 50822 | 173495 | 
| 88 | 44 lib/matplotlib/_constrained_layout.py | 1 | 59| 591 | 51413 | 173495 | 
| 89 | 44 galleries/users_explain/axes/tight_layout_guide.py | 226 | 297| 499 | 51912 | 173495 | 
| 90 | 45 galleries/examples/shapes_and_collections/hatch_style_reference.py | 1 | 65| 558 | 52470 | 174053 | 
| 91 | 45 galleries/users_explain/quick_start.py | 305 | 402| 968 | 53438 | 174053 | 
| 92 | **45 lib/matplotlib/figure.py** | 1315 | 1355| 369 | 53807 | 174053 | 
| 93 | 46 lib/matplotlib/backends/backend_gtk3.py | 232 | 260| 282 | 54089 | 178786 | 
| 94 | 47 galleries/examples/subplots_axes_and_figures/two_scales.py | 1 | 52| 407 | 54496 | 179193 | 
| 95 | **47 lib/matplotlib/figure.py** | 2829 | 2861| 323 | 54819 | 179193 | 
| 96 | **47 lib/matplotlib/figure.py** | 180 | 212| 286 | 55105 | 179193 | 
| 97 | 48 galleries/examples/axisartist/demo_parasite_axes.py | 1 | 54| 499 | 55604 | 179692 | 
| 98 | 48 lib/matplotlib/backends/backend_wx.py | 863 | 874| 151 | 55755 | 179692 | 
| 99 | 49 galleries/examples/axisartist/demo_parasite_axes2.py | 1 | 57| 521 | 56276 | 180213 | 
| 100 | 50 galleries/examples/subplots_axes_and_figures/axes_margins.py | 1 | 89| 770 | 57046 | 180983 | 
| 101 | 50 lib/matplotlib/_tight_layout.py | 266 | 302| 362 | 57408 | 180983 | 
| 102 | 51 galleries/examples/axes_grid1/demo_axes_hbox_divider.py | 1 | 55| 394 | 57802 | 181377 | 
| 103 | 52 galleries/examples/statistics/errorbars_and_boxes.py | 64 | 82| 117 | 57919 | 182023 | 
| 104 | 52 lib/matplotlib/backends/backend_wx.py | 527 | 538| 128 | 58047 | 182023 | 
| 105 | 53 galleries/examples/event_handling/figure_axes_enter_leave.py | 1 | 53| 323 | 58370 | 182346 | 
| 106 | 54 lib/matplotlib/backends/backend_gtk4.py | 225 | 260| 337 | 58707 | 187066 | 
| 107 | **54 lib/matplotlib/figure.py** | 1796 | 1823| 203 | 58910 | 187066 | 


### Hint

```
Thanks for the report @maurosilber.  The problem is clearer if we set a facecolor for each subfigure:

\`\`\`python
import matplotlib.pyplot as plt

for space in [0, 0.2]:
    figs = plt.figure().subfigures(2, 2, hspace=space, wspace=space)
    for fig, color in zip(figs.flat, 'cmyw'):
        fig.set_facecolor(color)
        fig.subplots().plot([1, 2])
plt.show()
\`\`\`

With `main`, both figures look like
![test](https://user-images.githubusercontent.com/10599679/226331428-3969469e-fee7-4b1d-95da-8d46ab2b31ee.png)

Just to add something, it looks like it works when using 'constrained' layout.
The relevant code is https://github.com/matplotlib/matplotlib/blob/0b4e615d72eb9f131feb877a8e3bf270f399fe77/lib/matplotlib/figure.py#L2237
This didn't break - it never worked.  I imagine the position logic could be borrowed from Axes....  
I am a first time contributer, and would like to attempt to work on this if possible! Will get a PR in the coming days
I have been trying to understand the code associated with this and have run into some issues.
In the function _redo_transform_rel_fig (linked above), I feel that I would have to be able to access all of the subfigures within this figure in order to give the correct amount of space based on the average subfigure width/height. Would this be possible? 
I have been looking to this function as inspiration for the logic, but I am still trying to understand all the parameters as well:
https://github.com/matplotlib/matplotlib/blob/0b4e615d72eb9f131feb877a8e3bf270f399fe77/lib/matplotlib/gridspec.py#L145

There is a `fig.subfigs` attribute which is a list of the `SubFigures` in a `Figure`.
Apologies for the slow progress, had a busier week than expected.
Below is the code I am running to test.

\`\`\`
import matplotlib.pyplot as plt

figs = plt.figure().subfigures(2, 2, hspace=0.2, wspace=0.2)
for fig, color in zip(figs.flat, 'cmyw'):
    fig.set_facecolor(color)
    fig.subplots().plot([1, 2])
# plt.show()

figs = plt.figure(constrained_layout=True).subfigures(2, 2, hspace=0.2, wspace=0.2)
for fig, color in zip(figs.flat, 'cmyw'):
    fig.set_facecolor(color)
    fig.subplots().plot([1, 2])
plt.show()
\`\`\`

This creates two figures, one with constrained layout and one without. Below is my current output.
On the right is the constrained layout figure, and the left is the one without.
<img width="1278" alt="Screenshot 2023-04-04 at 6 20 33 PM" src="https://user-images.githubusercontent.com/90582921/229935570-8ec26074-421c-4b78-a746-ce711ff6bea9.png">

My code currently fits the facecolors in the background to the correct spots, however the actual graphs do not match. They seem to need resizing to the right and upwards in order to match the constrained layout. Would a non-constrained layout figure be expected to resize those graphs to fit the background? I would assume so but I wanted to check since I couldn't find the answer in the documentation I looked at.
> My code currently fits the facecolors in the background to the correct spots, however the actual graphs do not match. They seem to need resizing to the right and upwards in order to match the constrained layout. Would a non-constrained layout figure be expected to resize those graphs to fit the background? I would assume so but I wanted to check since I couldn't find the answer in the documentation I looked at.

I'm not quite sure what you are asking here?  Constrained layout adjusts the axes sizes to fit in the figure.  If you don't do constrained layout the axes labels can definitely spill out of the figure if you just use default axes positioning.  

I've been digging into this.  We have a test that adds a subplot and a subfigure using the same gridspec, and the subfigure is expected to ignore the wspace on the gridspec.

https://github.com/matplotlib/matplotlib/blob/ffd3b12969e4ab630e678617c68492bc238924fa/lib/matplotlib/tests/test_figure.py#L1425-L1439

 <img src="https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/baseline_images/test_figure/test_subfigure_scatter_size.png?raw=true"> 

The use-case envisioned in the test seems entirely reasonable to me, but I'm struggling to see how we can support that while also fixing this issue.
Why do you say the subfigure is expected to ignore the wspace?  I don't see that wspace is set in the test. 
Since no _wspace_ is passed, I assume the gridspec will have the default from rcParams, which is 0.2.
Sure, but I don't understand what wouldn't work in that example with `wspace` argument. Do you just mean that the default would be too large for this case? 
Yes, I think in the test example, if both subfigure and subplot were respecting the 0.2 wspace then the left-hand subplots would be narrower and we’d have more whitespace in the middle.  Currently in this example the total width of the two lefthand subplots looks about the same as the width of the righthand one, so overall the figure seems well-balanced.

Another test here explicitly doesn’t expect any whitespace between subfigures, though in this case there are no subplots so you could just pass `wspace=0, hspace=0` to the gridspec and retain this result.
https://github.com/matplotlib/matplotlib/blob/8293774ba930fb039d91c3b3d4dd68c49ff997ba/lib/matplotlib/tests/test_figure.py#L1368-L1388


Can we just make the default wspace for subfigures be zero?
`wspace` is a property of the gridspec.  Do you mean we should have a separate property for subfigures, e.g. `GridSpec.subfig_wspace`, with its own default?
`gridspec` is still public API, but barely, as there are usually more elegant ways to do the same things that folks used to use gridspecs for.  

In this case, as you point out, it is better if subplots and subfigures get different wspace values, even if they are the same grid spec level.  I'm suggesting that subfigures ignores the grid spec wspace (as it currently does) and if we want a wspace for a set of subfigures that be a kwarg of the subfigure call.  

However, I never use wspace nor hspace, and given that all these things work so much better with constrained layout, I'm not sure what the goal of manually tweaking the spacing is.  
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1564,8 +1564,9 @@ def subfigures(self, nrows=1, ncols=1, squeeze=True,
         wspace, hspace : float, default: None
             The amount of width/height reserved for space between subfigures,
             expressed as a fraction of the average subfigure width/height.
-            If not given, the values will be inferred from a figure or
-            rcParams when necessary.
+            If not given, the values will be inferred from rcParams if using
+            constrained layout (see `~.ConstrainedLayoutEngine`), or zero if
+            not using a layout engine.
 
         width_ratios : array-like of length *ncols*, optional
             Defines the relative widths of the columns. Each column gets a
@@ -1580,13 +1581,24 @@ def subfigures(self, nrows=1, ncols=1, squeeze=True,
         gs = GridSpec(nrows=nrows, ncols=ncols, figure=self,
                       wspace=wspace, hspace=hspace,
                       width_ratios=width_ratios,
-                      height_ratios=height_ratios)
+                      height_ratios=height_ratios,
+                      left=0, right=1, bottom=0, top=1)
 
         sfarr = np.empty((nrows, ncols), dtype=object)
         for i in range(ncols):
             for j in range(nrows):
                 sfarr[j, i] = self.add_subfigure(gs[j, i], **kwargs)
 
+        if self.get_layout_engine() is None and (wspace is not None or
+                                                 hspace is not None):
+            # Gridspec wspace and hspace is ignored on subfigure instantiation,
+            # and no space is left.  So need to account for it here if required.
+            bottoms, tops, lefts, rights = gs.get_grid_positions(self)
+            for sfrow, bottom, top in zip(sfarr, bottoms, tops):
+                for sf, left, right in zip(sfrow, lefts, rights):
+                    bbox = Bbox.from_extents(left, bottom, right, top)
+                    sf._redo_transform_rel_fig(bbox=bbox)
+
         if squeeze:
             # Discarding unneeded dimensions that equal 1.  If we only have one
             # subfigure, just return it instead of a 1-element array.

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_figure.py b/lib/matplotlib/tests/test_figure.py
--- a/lib/matplotlib/tests/test_figure.py
+++ b/lib/matplotlib/tests/test_figure.py
@@ -1449,6 +1449,31 @@ def test_subfigure_pdf():
     fig.savefig(buffer, format='pdf')
 
 
+def test_subfigures_wspace_hspace():
+    sub_figs = plt.figure().subfigures(2, 3, hspace=0.5, wspace=1/6.)
+
+    w = 640
+    h = 480
+
+    np.testing.assert_allclose(sub_figs[0, 0].bbox.min, [0., h * 0.6])
+    np.testing.assert_allclose(sub_figs[0, 0].bbox.max, [w * 0.3, h])
+
+    np.testing.assert_allclose(sub_figs[0, 1].bbox.min, [w * 0.35, h * 0.6])
+    np.testing.assert_allclose(sub_figs[0, 1].bbox.max, [w * 0.65, h])
+
+    np.testing.assert_allclose(sub_figs[0, 2].bbox.min, [w * 0.7, h * 0.6])
+    np.testing.assert_allclose(sub_figs[0, 2].bbox.max, [w, h])
+
+    np.testing.assert_allclose(sub_figs[1, 0].bbox.min, [0, 0])
+    np.testing.assert_allclose(sub_figs[1, 0].bbox.max, [w * 0.3, h * 0.4])
+
+    np.testing.assert_allclose(sub_figs[1, 1].bbox.min, [w * 0.35, 0])
+    np.testing.assert_allclose(sub_figs[1, 1].bbox.max, [w * 0.65, h * 0.4])
+
+    np.testing.assert_allclose(sub_figs[1, 2].bbox.min, [w * 0.7, 0])
+    np.testing.assert_allclose(sub_figs[1, 2].bbox.max, [w, h * 0.4])
+
+
 def test_add_subplot_kwargs():
     # fig.add_subplot() always creates new axes, even if axes kwargs differ.
     fig = plt.figure()

```


## Code snippets

### 1 - galleries/examples/subplots_axes_and_figures/subfigures.py:

Start line: 84, End line: 149

```python
subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

fig.suptitle('Figure suptitle', fontsize='xx-large')
plt.show()

# %%
# Subfigures can have different widths and heights.  This is exactly the
# same example as the first example, but *width_ratios* has been changed:

fig = plt.figure(layout='constrained', figsize=(10, 4))
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

# %%
# Subfigures can be also be nested:

fig = plt.figure(layout='constrained', figsize=(10, 8))

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
### 2 - galleries/examples/subplots_axes_and_figures/subfigures.py:

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
    The *subfigure* concept is new in v3.4, and the API is still provisional.

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
fig = plt.figure(layout='constrained', figsize=(10, 4))
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

# %%
# It is possible to mix subplots and subfigures using
# `matplotlib.figure.Figure.add_subfigure`.  This requires getting
# the gridspec that the subplots are laid out on.

fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(10, 4))
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
### 3 - galleries/users_explain/axes/constrainedlayout_guide.py:

Start line: 184, End line: 268

```python
axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

# %%
# In order for a legend or other artist to *not* steal space
# from the subplot layout, we can ``leg.set_in_layout(False)``.
# Of course this can mean the legend ends up
# cropped, but can be useful if the plot is subsequently called
# with ``fig.savefig('outname.png', bbox_inches='tight')``.  Note,
# however, that the legend's ``get_in_layout`` status will have to be
# toggled again to make the saved file work, and we must manually
# trigger a draw if we want *constrained layout* to adjust the size
# of the Axes before printing.

fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")

axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
leg = axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
leg.set_in_layout(False)
# trigger a draw so that constrained layout is executed once
# before we turn it off when printing....
fig.canvas.draw()
# we want the legend included in the bbox_inches='tight' calcs.
leg.set_in_layout(True)
# we don't want the layout to change at this point.
fig.set_layout_engine('none')
try:
    fig.savefig('../../../doc/_static/constrained_layout_1b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # this allows the script to keep going if run interactively and
    # the directory above doesn't exist
    pass

# %%
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
    fig.savefig('../../../doc/_static/constrained_layout_2b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # this allows the script to keep going if run interactively and
    # the directory above doesn't exist
    pass


# %%
# The saved file looks like:
#
# .. image:: /_static/constrained_layout_2b.png
#    :align: center
#

# %%
# Padding and spacing
# ===================
#
# Padding between Axes is controlled in the horizontal by *w_pad* and
# *wspace*, and vertical by *h_pad* and *hspace*.  These can be edited
# via `~.layout_engine.ConstrainedLayoutEngine.set`.  *w/h_pad* are
# the minimum space around the Axes in units of inches:

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
                            wspace=0)

# %%
# Spacing between subplots is further set by *wspace* and *hspace*. These
# are specified as a fraction of the size of the subplot group as a whole.
# If these values are smaller than *w_pad* or *h_pad*, then the fixed pads are
# used instead. Note in the below how the space at the edges doesn't change
# from the above, but the space between subplots does.
```
### 4 - galleries/users_explain/axes/constrainedlayout_guide.py:

Start line: 270, End line: 351

```python
fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

# %%
# If there are more than two columns, the *wspace* is shared between them,
# so here the wspace is divided in two, with a *wspace* of 0.1 between each
# column:

fig, axs = plt.subplots(2, 3, layout="constrained")
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

# %%
# GridSpecs also have optional *hspace* and *wspace* keyword arguments,
# that will be used instead of the pads set by *constrained layout*:

fig, axs = plt.subplots(2, 2, layout="constrained",
                        gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
# this has no effect because the space set in the gridspec trumps the
# space set in *constrained layout*.
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0,
                            wspace=0.0)

# %%
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

# %%
# rcParams
# ========
#
# There are five :ref:`rcParams<customizing-with-dynamic-rc-settings>`
# that can be set, either in a script or in the :file:`matplotlibrc`
# file. They all have the prefix ``figure.constrained_layout``:
#
# - *use*: Whether to use *constrained layout*. Default is False
# - *w_pad*, *h_pad*:    Padding around Axes objects.
#   Float representing inches.  Default is 3./72. inches (3 pts)
# - *wspace*, *hspace*:  Space between subplot groups.
#   Float representing a fraction of the subplot widths being separated.
#   Default is 0.02.

plt.rcParams['figure.constrained_layout.use'] = True
fig, axs = plt.subplots(2, 2, figsize=(3, 3))
for ax in axs.flat:
    example_plot(ax)

# %%
# Use with GridSpec
# =================
#
# *Constrained layout* is meant to be used
# with :func:`~matplotlib.figure.Figure.subplots`,
# :func:`~matplotlib.figure.Figure.subplot_mosaic`, or
# :func:`~matplotlib.gridspec.GridSpec` with
# :func:`~matplotlib.figure.Figure.add_subplot`.
#
# Note that in what follows ``layout="constrained"``

plt.rcParams['figure.constrained_layout.use'] = False
```
### 5 - galleries/examples/subplots_axes_and_figures/subplots_demo.py:

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

# %%
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

# %%
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
### 6 - lib/matplotlib/figure.py:

Start line: 2289, End line: 2311

```python
@_docstring.interpd
class SubFigure(FigureBase):

    def get_constrained_layout(self):
        """
        Return whether constrained layout is being used.

        See :ref:`constrainedlayout_guide`.
        """
        return self._parent.get_constrained_layout()

    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        return self._parent.get_constrained_layout_pads(relative=relative)
```
### 7 - galleries/users_explain/axes/constrainedlayout_guide.py:

Start line: 442, End line: 527

```python
sfigs = fig.subfigures(1, 2, width_ratios=[1, 2])

axs_left = sfigs[0].subplots(2, 1)
for ax in axs_left.flat:
    example_plot(ax)

axs_right = sfigs[1].subplots(2, 2)
for ax in axs_right.flat:
    pcm = ax.pcolormesh(arr, **pc_kwargs)
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_title('title')
fig.colorbar(pcm, ax=axs_right)
fig.suptitle('Nested plots using subfigures')

# %%
# Manually setting Axes positions
# ================================
#
# There can be good reasons to manually set an Axes position.  A manual call
# to `~.axes.Axes.set_position` will set the Axes so *constrained layout* has
# no effect on it anymore. (Note that *constrained layout* still leaves the
# space for the Axes that is moved).

fig, axs = plt.subplots(1, 2, layout="constrained")
example_plot(axs[0], fontsize=12)
axs[1].set_position([0.2, 0.2, 0.4, 0.4])

# %%
# .. _compressed_layout:
#
# Grids of fixed aspect-ratio Axes: "compressed" layout
# =====================================================
#
# *Constrained layout* operates on the grid of "original" positions for
# Axes. However, when Axes have fixed aspect ratios, one side is usually made
# shorter, and leaves large gaps in the shortened direction. In the following,
# the Axes are square, but the figure quite wide so there is a horizontal gap:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout="constrained")
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='constrained'")

# %%
# One obvious way of fixing this is to make the figure size more square,
# however, closing the gaps exactly requires trial and error.  For simple grids
# of Axes we can use ``layout="compressed"`` to do the job for us:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout='compressed')
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='compressed'")


# %%
# Manually turning off *constrained layout*
# ===========================================
#
# *Constrained layout* usually adjusts the Axes positions on each draw
# of the figure.  If you want to get the spacing provided by
# *constrained layout* but not have it update, then do the initial
# draw and then call ``fig.set_layout_engine('none')``.
# This is potentially useful for animations where the tick labels may
# change length.
#
# Note that *constrained layout* is turned off for ``ZOOM`` and ``PAN``
# GUI events for the backends that use the toolbar.  This prevents the
# Axes from changing position during zooming and panning.
#
#
# Limitations
# ===========
#
# Incompatible functions
# ----------------------
#
# *Constrained layout* will work with `.pyplot.subplot`, but only if the
# number of rows and columns is the same for each call.
# The reason is that each call to `.pyplot.subplot` will create a new
# `.GridSpec` instance if the geometry is not the same, and
# *constrained layout*.  So the following works fine:

fig = plt.figure(layout="constrained")
```
### 8 - galleries/examples/subplots_axes_and_figures/subplots_demo.py:

Start line: 87, End line: 170

```python
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
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

# %%
# Sharing axes
# """"""""""""
#
# By default, each Axes is scaled individually. Thus, if the ranges are
# different the tick values of the subplots do not align.

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Axes values are scaled individually by default')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

# %%
# You can use *sharex* or *sharey* to align the horizontal or vertical axis.

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

# %%
# Setting *sharex* or *sharey* to ``True`` enables global sharing across the
# whole grid, i.e. also the y-axes of vertically stacked subplots have the
# same scale when using ``sharey=True``.

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

# %%
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

# %%
# Apart from ``True`` and ``False``, both *sharex* and *sharey* accept the
# values 'row' and 'col' to share the values only per row or column.

fig = plt.figure()
```
### 9 - lib/matplotlib/figure.py:

Start line: 2331, End line: 2352

```python
@_docstring.interpd
class SubFigure(FigureBase):

    get_axes = axes.fget

    def draw(self, renderer):
        # docstring inherited

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
### 10 - galleries/users_explain/axes/arranging_axes.py:

Start line: 342, End line: 437

```python
gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.75,
                      hspace=0.1, wspace=0.05)
ax0 = fig.add_subplot(gs[:-1, :])
annotate_axes(ax0, 'ax0')
ax1 = fig.add_subplot(gs[-1, :-1])
annotate_axes(ax1, 'ax1')
ax2 = fig.add_subplot(gs[-1, -1])
annotate_axes(ax2, 'ax2')
fig.suptitle('Manual gridspec with right=0.75')

# %%
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

# %%
# Here's a more sophisticated example of nested *GridSpec*: We create an outer
# 4x4 grid with each cell containing an inner 3x3 grid of Axes. We outline
# the outer 4x4 grid by hiding appropriate spines in each of the inner 3x3
# grids.


def squiggle_xy(a, b, c, d, i=np.arange(0.0, 2*np.pi, 0.05)):
    return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)

fig = plt.figure(figsize=(8, 8), layout='constrained')
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

# %%
#
# More reading
# ============
#
#  - More details about :ref:`subplot mosaic <mosaic>`.
#  - More details about :ref:`constrained layout
#    <constrainedlayout_guide>`, used to align
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
### 12 - lib/matplotlib/figure.py:

Start line: 917, End line: 932

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
### 19 - lib/matplotlib/figure.py:

Start line: 1545, End line: 1600

```python
class FigureBase(Artist):

    def subfigures(self, nrows=1, ncols=1, squeeze=True,
                   wspace=None, hspace=None,
                   width_ratios=None, height_ratios=None,
                   **kwargs):
        """
        Add a set of subfigures to this figure or subfigure.

        A subfigure has the same artist methods as a figure, and is logically
        the same as a figure, but cannot print itself.
        See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

        .. note::
            The *subfigure* concept is new in v3.4, and the API is still provisional.

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
### 30 - lib/matplotlib/figure.py:

Start line: 1, End line: 69

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

Figures are typically created using pyplot methods `~.pyplot.figure`,
`~.pyplot.subplots`, and `~.pyplot.subplot_mosaic`.

.. plot::
    :include-source:

    fig, ax = plt.subplots(figsize=(2, 2), facecolor='lightskyblue',
                           layout='constrained')
    fig.suptitle('Figure')
    ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')

Some situations call for directly instantiating a `~.figure.Figure` class,
usually inside an application of some sort (see :ref:`user_interfaces` for a
list of examples) .  More information about Figures can be found at
:ref:`figure_explanation`.
"""

from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    DrawEvent, FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
    ConstrainedLayoutEngine, TightLayoutEngine, LayoutEngine,
    PlaceHolderLayoutEngine
)
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
### 34 - lib/matplotlib/figure.py:

Start line: 155, End line: 177

```python
class SubplotParams:

    def update(self, left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
        if ((left if left is not None else self.left)
                >= (right if right is not None else self.right)):
            raise ValueError('left cannot be >= right')
        if ((bottom if bottom is not None else self.bottom)
                >= (top if top is not None else self.top)):
            raise ValueError('bottom cannot be >= top')
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if bottom is not None:
            self.bottom = bottom
        if top is not None:
            self.top = top
        if wspace is not None:
            self.wspace = wspace
        if hspace is not None:
            self.hspace = hspace
```
### 35 - lib/matplotlib/figure.py:

Start line: 391, End line: 399

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.5, y0=0.98, name='suptitle', ha='center',
                             va='top', rc='title')
    @_docstring.copy(_suplabels)
    def suptitle(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.98,
                'ha': 'center', 'va': 'top', 'rotation': 0,
                'size': 'figure.titlesize', 'weight': 'figure.titleweight'}
        return self._suplabels(t, info, **kwargs)
```
### 37 - lib/matplotlib/figure.py:

Start line: 2141, End line: 2239

```python
@_docstring.interpd
class SubFigure(FigureBase):
    """
    Logical figure that can be placed inside a figure.

    Typically instantiated using `.Figure.add_subfigure` or
    `.SubFigure.add_subfigure`, or `.SubFigure.subfigures`.  A subfigure has
    the same methods as a figure except for those particularly tied to the size
    or dpi of the figure, and is confined to a prescribed region of the figure.
    For example the following puts two subfigures side-by-side::

        fig = plt.figure()
        sfigs = fig.subfigures(1, 2)
        axsL = sfigs[0].subplots(1, 2)
        axsR = sfigs[1].subplots(2, 1)

    See :doc:`/gallery/subplots_axes_and_figures/subfigures`

    .. note::
        The *subfigure* concept is new in v3.4, and the API is still provisional.
    """

    def __init__(self, parent, subplotspec, *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 **kwargs):
        """
        Parameters
        ----------
        parent : `.Figure` or `.SubFigure`
            Figure or subfigure that contains the SubFigure.  SubFigures
            can be nested.

        subplotspec : `.gridspec.SubplotSpec`
            Defines the region in a parent gridspec where the subfigure will
            be placed.

        facecolor : default: ``"none"``
            The figure patch face color; transparent by default.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.

        Other Parameters
        ----------------
        **kwargs : `.SubFigure` properties, optional

            %(SubFigure:kwdoc)s
        """
        super().__init__(**kwargs)
        if facecolor is None:
            facecolor = "none"
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        self._subplotspec = subplotspec
        self._parent = parent
        self.figure = parent.figure

        # subfigures use the parent axstack
        self._axstack = parent._axstack
        self.subplotpars = parent.subplotpars
        self.dpi_scale_trans = parent.dpi_scale_trans
        self._axobservers = parent._axobservers
        self.canvas = parent.canvas
        self.transFigure = parent.transFigure
        self.bbox_relative = None
        self._redo_transform_rel_fig()
        self.figbbox = self._parent.figbbox
        self.bbox = TransformedBbox(self.bbox_relative,
                                    self._parent.transSubfigure)
        self.transSubfigure = BboxTransformTo(self.bbox)

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # Don't let the figure patch influence bbox calculation.
            in_layout=False, transform=self.transSubfigure)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

    @property
    def dpi(self):
        return self._parent.dpi

    @dpi.setter
    def dpi(self, value):
        self._parent.dpi = value
```
### 38 - lib/matplotlib/figure.py:

Start line: 2863, End line: 2904

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.deprecated("3.6", alternative="fig.get_layout_engine().get()",
                     pending=True)
    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.
        All values are None if ``constrained_layout`` is not used.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        if not isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
            return None, None, None, None
        info = self.get_layout_engine().get()
        w_pad = info['w_pad']
        h_pad = info['h_pad']
        wspace = info['wspace']
        hspace = info['hspace']

        if relative and (w_pad is not None or h_pad is not None):
            renderer = self._get_renderer()
            dpi = renderer.dpi
            w_pad = w_pad * dpi / renderer.width
            h_pad = h_pad * dpi / renderer.height

        return w_pad, h_pad, wspace, hspace

    def set_canvas(self, canvas):
        """
        Set the canvas that contains the figure

        Parameters
        ----------
        canvas : FigureCanvas
        """
        self.canvas = canvas
```
### 44 - lib/matplotlib/figure.py:

Start line: 1935, End line: 1960

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.',
                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
        subplot_kw = subplot_kw or {}
        gridspec_kw = dict(gridspec_kw or {})
        per_subplot_kw = per_subplot_kw or {}

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

        # special-case string input
        if isinstance(mosaic, str):
            mosaic = self._normalize_grid_string(mosaic)
            per_subplot_kw = {
                tuple(k): v for k, v in per_subplot_kw.items()
            }

        per_subplot_kw = self._norm_per_subplot_kw(per_subplot_kw)

        # Only accept strict bools to allow a possible future API expansion.
        _api.check_isinstance(bool, sharex=sharex, sharey=sharey)
        # ... other code
```
### 48 - lib/matplotlib/figure.py:

Start line: 2074, End line: 2113

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.',
                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):

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
                        gs[slc], **{
                            'label': str(name),
                            **subplot_kw,
                            **per_subplot_kw.get(name, {})
                        }
                    )
                    output[name] = ax
                elif method == 'nested':
                    nested_mosaic = arg
                    j, k = key
                    # recursively add the nested mosaic
                    rows, cols = nested_mosaic.shape
                    nested_output = _do_layout(
                        gs[j, k].subgridspec(rows, cols),
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
### 50 - lib/matplotlib/figure.py:

Start line: 2115, End line: 2138

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.',
                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
        # ... other code

        mosaic = _make_array(mosaic)
        rows, cols = mosaic.shape
        gs = self.add_gridspec(rows, cols, **gridspec_kw)
        ret = _do_layout(gs, mosaic, *_identify_keys_and_nested(mosaic))
        ax0 = next(iter(ret.values()))
        for ax in ret.values():
            if sharex:
                ax.sharex(ax0)
                ax._label_outer_xaxis(check_patch=True)
            if sharey:
                ax.sharey(ax0)
                ax._label_outer_yaxis(check_patch=True)
        if extra := set(per_subplot_kw) - set(ret):
            raise ValueError(
                f"The keys {extra} are in *per_subplot_kw* "
                "but not in the mosaic."
            )
        return ret

    def _set_artist_props(self, a):
        if a != self:
            a.set_figure(self)
        a.stale_callback = _stale_figure_callback
        a.set_transform(self.transSubfigure)
```
### 59 - lib/matplotlib/figure.py:

Start line: 1602, End line: 1627

```python
class FigureBase(Artist):

    def add_subfigure(self, subplotspec, **kwargs):
        """
        Add a `.SubFigure` to the figure as part of a subplot arrangement.

        Parameters
        ----------
        subplotspec : `.gridspec.SubplotSpec`
            Defines the region in a parent gridspec where the subfigure will
            be placed.

        Returns
        -------
        `.SubFigure`

        Other Parameters
        ----------------
        **kwargs
            Are passed to the `.SubFigure` object.

        See Also
        --------
        .Figure.subfigures
        """
        sf = SubFigure(self, subplotspec, **kwargs)
        self.subfigs += [sf]
        return sf
```
### 67 - lib/matplotlib/figure.py:

Start line: 1283, End line: 1313

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
            fig = (  # Figure of first axes; logic copied from make_axes.
                [*ax.flat] if isinstance(ax, np.ndarray)
                else [*ax] if np.iterable(ax)
                else [ax])[0].figure
            current_ax = fig.gca()
            if (fig.get_layout_engine() is not None and
                    not fig.get_layout_engine().colorbar_gridspec):
                use_gridspec = False
            if (use_gridspec
                    and isinstance(ax, mpl.axes._base._AxesBase)
                    and ax.get_subplotspec()):
                cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
            else:
                cax, kwargs = cbar.make_axes(ax, **kwargs)
            # make_axes calls add_{axes,subplot} which changes gca; undo that.
            fig.sca(current_ax)
            cax.grid(visible=False, which='both', axis='both')

        NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
            'fraction', 'pad', 'shrink', 'aspect', 'anchor', 'panchor']
        cb = cbar.Colorbar(cax, mappable, **{
            k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS})
        cax.figure.stale = True
        return cb
```
### 70 - lib/matplotlib/figure.py:

Start line: 758, End line: 796

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def add_subplot(self, *args, **kwargs):
        if 'figure' in kwargs:
            # Axes itself allows for a 'figure' kwarg, but since we want to
            # bind the created Axes to self, it is not allowed here.
            raise _api.kwarg_error("add_subplot", "figure")

        if (len(args) == 1
                and isinstance(args[0], mpl.axes._base._AxesBase)
                and args[0].get_subplotspec()):
            ax = args[0]
            key = ax._projection_init
            if ax.get_figure() is not self:
                raise ValueError("The Axes must have been created in "
                                 "the present figure")
        else:
            if not args:
                args = (1, 1, 1)
            # Normalize correct ijk values to (i, j, k) here so that
            # add_subplot(211) == add_subplot(2, 1, 1).  Invalid values will
            # trigger errors later (via SubplotSpec._from_subplot_args).
            if (len(args) == 1 and isinstance(args[0], Integral)
                    and 100 <= args[0] <= 999):
                args = tuple(map(int, str(args[0])))
            projection_class, pkw = self._process_projection_requirements(**kwargs)
            ax = projection_class(self, *args, **pkw)
            key = (projection_class, pkw)
        return self._add_axes_internal(ax, key)

    def _add_axes_internal(self, ax, key):
        """Private helper for `add_axes` and `add_subplot`."""
        self._axstack.add(ax)
        if ax not in self._localaxes:
            self._localaxes.append(ax)
        self.sca(ax)
        ax._remove_method = self.delaxes
        # this is to support plt.subplot's re-selection logic
        ax._projection_init = key
        self.stale = True
        ax.stale_callback = _stale_figure_callback
        return ax
```
### 76 - lib/matplotlib/figure.py:

Start line: 2313, End line: 2329

```python
@_docstring.interpd
class SubFigure(FigureBase):

    def get_layout_engine(self):
        return self._parent.get_layout_engine()

    @property
    def axes(self):
        """
        List of Axes in the SubFigure.  You can access and modify the Axes
        in the SubFigure through this list.

        Modifying this list has no effect. Instead, use `~.SubFigure.add_axes`,
        `~.SubFigure.add_subplot` or `~.SubFigure.delaxes` to add or remove an
        Axes.

        Note: The `.SubFigure.axes` property and `~.SubFigure.get_axes` method
        are equivalent.
        """
        return self._localaxes[:]
```
### 80 - lib/matplotlib/figure.py:

Start line: 3345, End line: 3382

```python
@_docstring.interpd
class Figure(FigureBase):

    def savefig(self, fname, *, transparent=None, **kwargs):

        kwargs.setdefault('dpi', mpl.rcParams['savefig.dpi'])
        if transparent is None:
            transparent = mpl.rcParams['savefig.transparent']

        with ExitStack() as stack:
            if transparent:
                def _recursively_make_subfig_transparent(exit_stack, subfig):
                    exit_stack.enter_context(
                        subfig.patch._cm_set(
                            facecolor="none", edgecolor="none"))
                    for ax in subfig.axes:
                        exit_stack.enter_context(
                            ax.patch._cm_set(
                                facecolor="none", edgecolor="none"))
                    for sub_subfig in subfig.subfigs:
                        _recursively_make_subfig_transparent(
                            exit_stack, sub_subfig)

                def _recursively_make_axes_transparent(exit_stack, ax):
                    exit_stack.enter_context(
                        ax.patch._cm_set(facecolor="none", edgecolor="none"))
                    for child_ax in ax.child_axes:
                        exit_stack.enter_context(
                            child_ax.patch._cm_set(
                                facecolor="none", edgecolor="none"))
                    for child_childax in ax.child_axes:
                        _recursively_make_axes_transparent(
                            exit_stack, child_childax)

                kwargs.setdefault('facecolor', 'none')
                kwargs.setdefault('edgecolor', 'none')
                # set subfigure to appear transparent in printed image
                for subfig in self.subfigs:
                    _recursively_make_subfig_transparent(stack, subfig)
                # set axes to be transparent
                for ax in self.axes:
                    _recursively_make_axes_transparent(stack, ax)
            self.canvas.print_figure(fname, **kwargs)
```
### 87 - lib/matplotlib/figure.py:

Start line: 634, End line: 655

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def add_axes(self, *args, **kwargs):
        # ... other code

        if isinstance(args[0], Axes):
            a, *extra_args = args
            key = a._projection_init
            if a.get_figure() is not self:
                raise ValueError(
                    "The Axes must have been created in the present figure")
        else:
            rect, *extra_args = args
            if not np.isfinite(rect).all():
                raise ValueError(f'all entries in rect must be finite not {rect}')
            projection_class, pkw = self._process_projection_requirements(**kwargs)

            # create the new axes using the axes class given
            a = projection_class(self, rect, **pkw)
            key = (projection_class, pkw)

        if extra_args:
            _api.warn_deprecated(
                "3.8",
                name="Passing more than one positional argument to Figure.add_axes",
                addendum="Any additional positional arguments are currently ignored.")
        return self._add_axes_internal(a, key)
```
### 92 - lib/matplotlib/figure.py:

Start line: 1315, End line: 1355

```python
class FigureBase(Artist):

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None):
        """
        Adjust the subplot layout parameters.

        Unset parameters are left unmodified; initial values are given by
        :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float, optional
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float, optional
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float, optional
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float, optional
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float, optional
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float, optional
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
        if (self.get_layout_engine() is not None and
                not self.get_layout_engine().adjust_compatible):
            _api.warn_external(
                "This figure was using a layout engine that is "
                "incompatible with subplots_adjust and/or tight_layout; "
                "not calling subplots_adjust.")
            return
        self.subplotpars.update(left, bottom, right, top, wspace, hspace)
        for ax in self.axes:
            if ax.get_subplotspec() is not None:
                ax._set_position(ax.get_subplotspec().get_position(self))
        self.stale = True
```
### 95 - lib/matplotlib/figure.py:

Start line: 2829, End line: 2861

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.deprecated(
         "3.6", alternative="figure.get_layout_engine().set()",
         pending=True)
    def set_constrained_layout_pads(self, **kwargs):
        """
        Set padding for ``constrained_layout``.

        Tip: The parameters can be passed from a dictionary by using
        ``fig.set_constrained_layout(**pad_dict)``.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        w_pad : float, default: :rc:`figure.constrained_layout.w_pad`
            Width padding in inches.  This is the pad around Axes
            and is meant to make sure there is enough room for fonts to
            look good.  Defaults to 3 pts = 0.04167 inches

        h_pad : float, default: :rc:`figure.constrained_layout.h_pad`
            Height padding in inches. Defaults to 3 pts.

        wspace : float, default: :rc:`figure.constrained_layout.wspace`
            Width padding between subplots, expressed as a fraction of the
            subplot width.  The total padding ends up being w_pad + wspace.

        hspace : float, default: :rc:`figure.constrained_layout.hspace`
            Height padding between subplots, expressed as a fraction of the
            subplot width. The total padding ends up being h_pad + hspace.

        """
        if isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
            self.get_layout_engine().set(**kwargs)
```
### 96 - lib/matplotlib/figure.py:

Start line: 180, End line: 212

```python
class FigureBase(Artist):
    """
    Base class for `.Figure` and `.SubFigure` containing the methods that add
    artists to the figure or subfigure, create Axes, etc.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # remove the non-figure artist _axes property
        # as it makes no sense for a figure to be _in_ an Axes
        # this is used by the property methods in the artist base class
        # which are over-ridden in this class
        del self._axes

        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None

        # groupers to keep track of x and y labels we want to align.
        # see self.align_xlabels and self.align_ylabels and
        # axis._get_tick_boxes_siblings
        self._align_label_groups = {"x": cbook.Grouper(), "y": cbook.Grouper()}

        self._localaxes = []  # track all axes
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts = []
        self.images = []
        self.legends = []
        self.subfigs = []
        self.stale = True
        self.suppressComposite = None
        self.set(**kwargs)
```
### 107 - lib/matplotlib/figure.py:

Start line: 1796, End line: 1823

```python
class FigureBase(Artist):

    @staticmethod
    def _norm_per_subplot_kw(per_subplot_kw):
        expanded = {}
        for k, v in per_subplot_kw.items():
            if isinstance(k, tuple):
                for sub_key in k:
                    if sub_key in expanded:
                        raise ValueError(
                            f'The key {sub_key!r} appears multiple times.'
                            )
                    expanded[sub_key] = v
            else:
                if k in expanded:
                    raise ValueError(
                        f'The key {k!r} appears multiple times.'
                    )
                expanded[k] = v
        return expanded

    @staticmethod
    def _normalize_grid_string(layout):
        if '\n' not in layout:
            # single-line string
            return [list(ln) for ln in layout.split(';')]
        else:
            # multi-line string
            layout = inspect.cleandoc(layout)
            return [list(ln) for ln in layout.strip('\n').split('\n')]
```
