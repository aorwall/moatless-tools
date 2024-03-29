# matplotlib__matplotlib-23047

| **matplotlib/matplotlib** | `3699ff34d6e2d6d649ee0ced5dc3c74936449d67` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 47 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -6651,6 +6651,7 @@ def hist(self, x, bins=None, range=None, density=False, weights=None,
             m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
             tops.append(m)
         tops = np.array(tops, float)  # causes problems later if it's an int
+        bins = np.array(bins, float)  # causes problems if float16
         if stacked:
             tops = tops.cumsum(axis=0)
             # If a stacked density plot, normalize so the area of all the

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/axes/_axes.py | 6654 | 6654 | - | 47 | -


## Problem Statement

```
[Bug]: Gaps and overlapping areas between bins when using float16
### Bug summary

When creating a histogram out of float16 data, the bins are also calculated in float16. The lower precision can cause two errors: 
1) Gaps between certain bins. 
2) Two neighboring bins overlap each other (only visible when alpha < 1)


### Code for reproduction

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
values = np.clip(np.random.normal(0.5, 0.3, size=1000), 0, 1).astype(np.float16)
plt.hist(values, bins=100, alpha=0.5)
plt.show()
\`\`\`


### Actual outcome

![float16](https://user-images.githubusercontent.com/101181208/157218682-bc52d999-b6c7-4ed3-9681-fde46b04dd87.png)


### Expected outcome

![float32](https://user-images.githubusercontent.com/101181208/157218716-641edabf-fc4a-4826-b830-8ff2978d0b7a.png)


Created by `plt.hist(values.astype(np.float32), bins=100, alpha=0.5)
plt.show()`

### Additional information

**Possible solution**
Calculate the bins in float32:
- Determine minimal and maximal value in float16.  
- Convert min and max to float32. 
- Calculate the bin edges. 



**Theoretical possible, but unwanted solution**
Convert data into float32 before calculating the histogram. This behavior does not make a lot of sense, as float16 is mostly used because of memory limitations (arrays with billions of values can easily take several gigabytes).


### Operating system

Windows 10

### Matplotlib Version

3.4.3

### Matplotlib Backend

TkAgg

### Python version

3.7.1

### Jupyter version

_No response_

### Installation

pip
Refactor hist for less numerical errors
## PR Summary

Should help with #22622

Idea is to do computation on the edges rather than the widths and then do diff on the result. This may be numerically better (or not...). Or rather, it is probably numerically worse, but will give visually better results...

Probably the alternative approach of providing a flag to `bar`/`barh`, making sure that adjacent bars are actually exactly adjacent may be a better approach, but I wanted to see what comes out of this first...

## PR Checklist

<!-- Please mark any checkboxes that do not apply to this PR as [N/A]. -->
**Tests and Styling**
- [ ] Has pytest style unit tests (and `pytest` passes).
- [ ] Is [Flake 8](https://flake8.pycqa.org/en/latest/) compliant (install `flake8-docstrings` and run `flake8 --docstring-convention=all`).

**Documentation**
- [ ] New features are documented, with examples if plot related.
- [ ] New features have an entry in `doc/users/next_whats_new/` (follow instructions in README.rst there).
- [ ] API changes documented in `doc/api/next_api_changes/` (follow instructions in README.rst there).
- [ ] Documentation is sphinx and numpydoc compliant (the docs should [build](https://matplotlib.org/devel/documenting_mpl.html#building-the-docs) without error).

<!--
Thank you so much for your PR!  To help us review your contribution, please
consider the following points:

- A development guide is available at https://matplotlib.org/devdocs/devel/index.html.

- Help with git and github is available at
  https://matplotlib.org/devel/gitwash/development_workflow.html.

- Do not create the PR out of main, but out of a separate branch.

- The PR title should summarize the changes, for example "Raise ValueError on
  non-numeric input to set_xlim".  Avoid non-descriptive titles such as
  "Addresses issue #8576".

- The summary should provide at least 1-2 sentences describing the pull request
  in detail (Why is this change required?  What problem does it solve?) and
  link to any relevant issues.

- If you are contributing fixes to docstrings, please pay attention to
  http://matplotlib.org/devel/documenting_mpl.html#formatting.  In particular,
  note the difference between using single backquotes, double backquotes, and
  asterisks in the markup.

We understand that PRs can sometimes be overwhelming, especially as the
reviews start coming in.  Please let us know if the reviews are unclear or
the recommended next step seems overly demanding, if you would like help in
addressing a reviewer's comments, or if you have been waiting too long to hear
back on your PR.
-->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/lines_bars_and_markers/filled_step.py | 179 | 237| 493 | 493 | 1645 | 
| 2 | 2 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 1195 | 2756 | 
| 3 | 3 tutorials/introductory/quick_start.py | 291 | 388| 986 | 2181 | 8697 | 
| 4 | 4 examples/statistics/hist.py | 1 | 113| 840 | 3021 | 9537 | 
| 5 | 5 examples/statistics/histogram_features.py | 1 | 60| 416 | 3437 | 9953 | 
| 6 | 6 examples/statistics/time_series_histogram.py | 1 | 72| 770 | 4207 | 11087 | 
| 7 | 7 tutorials/introductory/pyplot.py | 329 | 422| 877 | 5084 | 15563 | 
| 8 | 8 examples/lines_bars_and_markers/psd_demo.py | 82 | 153| 751 | 5835 | 17258 | 
| 9 | 9 lib/matplotlib/pyplot.py | 2588 | 2600| 183 | 6018 | 43883 | 
| 10 | 10 examples/statistics/multiple_histograms_side_by_side.py | 1 | 72| 601 | 6619 | 44484 | 
| 11 | 10 examples/lines_bars_and_markers/filled_step.py | 1 | 77| 477 | 7096 | 44484 | 
| 12 | 10 examples/statistics/time_series_histogram.py | 73 | 107| 364 | 7460 | 44484 | 
| 13 | 10 lib/matplotlib/pyplot.py | 1 | 91| 679 | 8139 | 44484 | 
| 14 | 11 examples/user_interfaces/svg_histogram_sgskip.py | 37 | 160| 907 | 9046 | 45652 | 
| 15 | 12 examples/statistics/hexbin_demo.py | 1 | 44| 341 | 9387 | 45993 | 
| 16 | 12 tutorials/introductory/quick_start.py | 475 | 561| 1039 | 10426 | 45993 | 
| 17 | 13 examples/lines_bars_and_markers/fill_between_demo.py | 82 | 142| 602 | 11028 | 47349 | 
| 18 | 14 tutorials/advanced/transforms_tutorial.py | 254 | 351| 1003 | 12031 | 53675 | 
| 19 | 15 examples/lines_bars_and_markers/fill_between_alpha.py | 1 | 82| 754 | 12785 | 55011 | 
| 20 | 16 examples/statistics/boxplot_color.py | 1 | 63| 481 | 13266 | 55492 | 
| 21 | 17 tutorials/colors/colormaps.py | 203 | 260| 653 | 13919 | 60274 | 
| 22 | 18 examples/statistics/histogram_cumulative.py | 1 | 81| 691 | 14610 | 60965 | 
| 23 | 19 examples/specialty_plots/leftventricle_bulleye.py | 130 | 193| 708 | 15318 | 63132 | 
| 24 | 20 examples/images_contours_and_fields/image_annotated_heatmap.py | 218 | 312| 851 | 16169 | 65942 | 
| 25 | 21 examples/misc/histogram_path.py | 1 | 98| 730 | 16899 | 66672 | 
| 26 | 22 lib/matplotlib/_cm.py | 106 | 156| 787 | 17686 | 95115 | 
| 27 | 23 examples/ticks/ticks_too_many.py | 1 | 77| 742 | 18428 | 95857 | 
| 28 | 24 examples/statistics/histogram_histtypes.py | 1 | 60| 501 | 18929 | 96358 | 
| 29 | 25 examples/statistics/errorbar_limits.py | 1 | 86| 841 | 19770 | 97199 | 
| 30 | 26 examples/lines_bars_and_markers/stairs_demo.py | 1 | 92| 831 | 20601 | 98030 | 
| 31 | 27 examples/statistics/boxplot_demo.py | 1 | 88| 771 | 21372 | 100366 | 
| 32 | 28 tutorials/colors/colorbar_only.py | 88 | 131| 357 | 21729 | 101457 | 
| 33 | 29 examples/images_contours_and_fields/pcolor_demo.py | 1 | 97| 788 | 22517 | 102502 | 
| 34 | 29 tutorials/colors/colorbar_only.py | 1 | 86| 734 | 23251 | 102502 | 
| 35 | 30 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 23554 | 102805 | 
| 36 | 31 examples/mplot3d/hist3d.py | 1 | 34| 254 | 23808 | 103059 | 
| 37 | 32 examples/images_contours_and_fields/colormap_normalizations.py | 1 | 75| 752 | 24560 | 104566 | 
| 38 | 33 examples/lines_bars_and_markers/bar_label_demo.py | 1 | 107| 820 | 25380 | 105386 | 
| 39 | 34 tools/github_stats.py | 109 | 246| 1198 | 26578 | 107399 | 
| 40 | 35 examples/statistics/boxplot.py | 1 | 73| 747 | 27325 | 108460 | 
| 41 | 36 tutorials/toolkits/axisartist.py | 1 | 564| 4724 | 32049 | 113184 | 
| 42 | 37 examples/images_contours_and_fields/colormap_normalizations_symlognorm.py | 1 | 85| 820 | 32869 | 114004 | 
| 43 | 37 lib/matplotlib/_cm.py | 1080 | 1211| 458 | 33327 | 114004 | 
| 44 | 38 doc/conf.py | 176 | 274| 769 | 34096 | 118974 | 
| 45 | 39 tutorials/introductory/lifecycle.py | 98 | 188| 740 | 34836 | 121302 | 
| 46 | 39 examples/statistics/boxplot_demo.py | 89 | 165| 778 | 35614 | 121302 | 
| 47 | 40 examples/pyplots/boxplot_demo_pyplot.py | 1 | 90| 585 | 36199 | 121887 | 
| 48 | 40 lib/matplotlib/_cm.py | 637 | 661| 660 | 36859 | 121887 | 
| 49 | 40 lib/matplotlib/pyplot.py | 2535 | 2566| 369 | 37228 | 121887 | 
| 50 | 40 lib/matplotlib/_cm.py | 461 | 506| 1095 | 38323 | 121887 | 
| 51 | 40 tutorials/introductory/pyplot.py | 252 | 328| 910 | 39233 | 121887 | 
| 52 | 41 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 40014 | 127909 | 
| 53 | 42 examples/images_contours_and_fields/image_antialiasing.py | 1 | 69| 758 | 40772 | 129204 | 
| 54 | 42 doc/conf.py | 109 | 156| 482 | 41254 | 129204 | 
| 55 | 42 examples/lines_bars_and_markers/fill_between_alpha.py | 83 | 141| 582 | 41836 | 129204 | 
| 56 | 42 lib/matplotlib/pyplot.py | 2447 | 2458| 146 | 41982 | 129204 | 
| 57 | 43 examples/style_sheets/style_sheets_reference.py | 87 | 103| 203 | 42185 | 130639 | 
| 58 | 43 lib/matplotlib/pyplot.py | 2253 | 2328| 767 | 42952 | 130639 | 
| 59 | 44 lib/matplotlib/dates.py | 1 | 172| 1572 | 44524 | 147643 | 
| 60 | 45 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 82| 738 | 45262 | 148512 | 
| 61 | 45 examples/lines_bars_and_markers/scatter_hist.py | 1 | 36| 251 | 45513 | 148512 | 
| 62 | 45 examples/statistics/boxplot_demo.py | 166 | 255| 786 | 46299 | 148512 | 
| 63 | 45 tutorials/introductory/quick_start.py | 389 | 474| 931 | 47230 | 148512 | 
| 64 | 46 examples/color/colorbar_basics.py | 1 | 59| 506 | 47736 | 149018 | 


### Hint

```
To be checked: Can the same effect occur when using (numpy) int arrays?
Just a note that `np.hist(float16)` returns `float16` edges.

You may want to try using "stairs" here instead, which won't draw the bars all the way down to zero and help avoid those artifacts.
`plt.stairs(*np.histogram(values, bins=100), fill=True, alpha=0.5)`
I am not sure, but it seems like possibly a problem in NumPy.

\`\`\`
In[9]: cnt, bins = np.histogram(values, 100)

In [10]: bins
Out[10]: 
array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
       0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
       0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
       0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
       0.99, 1.  ], dtype=float16)

In [11]: np.diff(bins)
Out[11]: 
array([0.01    , 0.01    , 0.009995, 0.01001 , 0.00998 , 0.01001 ,
       0.01001 , 0.01001 , 0.01001 , 0.00995 , 0.01001 , 0.01001 ,
       0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.00989 , 0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.01001 , 0.009766, 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 , 0.01001 ,
       0.01001 , 0.01001 , 0.009766, 0.010254, 0.009766, 0.010254,
       0.009766, 0.010254, 0.009766, 0.010254, 0.009766, 0.010254,
       0.009766, 0.010254, 0.009766, 0.010254, 0.009766, 0.010254,
       0.009766, 0.010254, 0.009766, 0.010254, 0.009766, 0.010254,
       0.009766, 0.010254, 0.009766, 0.009766, 0.010254, 0.009766,
       0.010254, 0.009766, 0.010254, 0.009766, 0.010254, 0.009766,
       0.010254, 0.009766, 0.010254, 0.009766, 0.010254, 0.009766,
       0.010254, 0.009766, 0.010254, 0.009766, 0.010254, 0.009766,
       0.010254, 0.009766, 0.010254, 0.009766], dtype=float16)
\`\`\`

It looks like the diff is not really what is expected.
~I am actually a bit doubtful if the bins are really float16 here though.~ I guess they are, since it is float16, not bfloat16.
It is possible to trigger it with quite high probability using three bins, so that may be an easier case to debug (second and third bar overlap). Bin edges and diff seems to be the same independent of overlap or not.

\`\`\`
In [44]: bins
Out[44]: array([0.    , 0.3333, 0.6665, 1.    ], dtype=float16)

In [45]: np.diff(bins)
Out[45]: array([0.3333, 0.3333, 0.3335], dtype=float16)
\`\`\`
There is an overlap in the plot data (so it is not caused by the actual plotting, possibly rounding the wrong way):

\`\`\`
In [98]: bc.patches[1].get_corners()
Out[98]: 
array([[3.33251953e-01, 0.00000000e+00],
       [6.66992188e-01, 0.00000000e+00],
       [6.66992188e-01, 4.05000000e+02],
       [3.33251953e-01, 4.05000000e+02]])

In [99]: bc.patches[2].get_corners()
Out[99]: 
array([[  0.66601562,   0.        ],
       [  0.99951172,   0.        ],
       [  0.99951172, 314.        ],
       [  0.66601562, 314.        ]])
\`\`\` 
As the second bar ends at 6.66992188e-01 and the third bar starts at 0.66601562, this will happen.
A possibly easy way to solve this is to provide a keyword argument to `bar`/`barh` that makes sure that the bars are always adjacent, i.e., let `bar`/`barh` know that the next bar should have the same starting point as the previous bars end point. That keyword argument can then be called from from `hist` in case of an `rwidth` of 1.
This is probably the line causing the error:
https://github.com/matplotlib/matplotlib/blob/8b1881fd49b49bf85a7b91575f4653be41c26294/lib/matplotlib/axes/_axes.py#L2382
Something like `np.diff(np.cumsum(x) - width/2)` may work, but should then only be conditionally executed if the keyword argument is set.

(Then, I am not sure to what extent np.diff and np.cumsum are 100% numerically invariant, it is not trivial under floating-point arithmetic. But probably this will reduce the probability of errors anyway.)
> To be checked: Can the same effect occur when using (numpy) int arrays?

Yes and no. As the int array will become a float64 after multiplying with a float (dr in the code), it is quite unlikely to happen. However, it is not theoretically impossible to obtain the same effect with float64, although not very likely that it will actually be seen in a plot (the accumulated numerical error should correspond to something close to half(?) a pixel). But I am quite sure that one can trigger this by trying.
If you force the bins to be float64, then you won't have this problem:

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
values = np.clip(np.random.normal(0.5, 0.3, size=1000), 0, 1).astype(np.float16)
n, bins = np.histogram(values, bins=100)
n, bins, patches = plt.hist(values, bins=np.array(bins, dtype='float64'), alpha=0.5)

plt.show()
\`\`\`
so I think the reasonable fix here is simply for matplotlib to coerce the output from `np.histogram` to be floats - the output is turned to float64 when rendered anyways, and the extra memory for any visible number of bins is not going to matter.  

Is the numerical problem the diff?   Would it make sense to just convert the numpy bin edges to float64 before the diff?
> Is the numerical problem the diff? 

Hard to say. But the problem is that one does quite a bit of computations and at some stage there are rounding errors that leads to that there are overlaps or gaps between edges. So postponing diff will reduce the risk that this happens (on the other hand, one may get cancellations as a result, but I do not think that will happen more now since the only things we add here are about the same order of magnitude).

> Would it make sense to just convert the numpy bin edges to float64 before the diff?

Yes, or even float32, but as argued in the issue, one tend to use float16 for memory limited environments, so not clear if one can afford it.

Here, I am primarily trying to see the effect of it. As we do not deal with all involved computations here, some are also in `bar`/`barh`, the better approach may be to use a flag, "fill", or something that makes sure that all edges are adjacent if set (I'm quite sure a similar problem can arise if feeding `bar`-edges in `float16` as well.
It seems like we do not have any test images that are negatively affected by this at least... But it may indeed not be the best solution to the problem.


Ahh, but even if the data to hist is `float16`, the actual histogram array doesn't have to be that... And that is probably much smaller compared to the data. So probably a simpler fix is to change the data type of the histogram data before starting to process it...
I think you just want another type catch here (I guess I'm not sure the difference between `float` and `"float64"`), or at least that fixes the problem for me.

\`\`\`diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
index f1ec9406ea..88d90294a3 100644
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -6614,6 +6614,7 @@ such objects
             m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
             tops.append(m)
         tops = np.array(tops, float)  # causes problems later if it's an int
+        bins = np.array(bins, float)  # causes problems is float16!
         if stacked:
             tops = tops.cumsum(axis=0)
             # If a stacked density plot, normalize so the area of all the
\`\`\`
> I guess I'm not sure the difference between float and "float64"

Numpy accepts builtin python types and maps them to numpy types:

https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types
(scroll a bit to "Built-in Python types").

The mapping can be platform specific. E.g. `int` maps to `np.int64` on linux but `np.int32` on win.
`float` maps on x86 linux and win to `np.float64`. But I don't know if that's true on arm etc.
```

## Patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -6651,6 +6651,7 @@ def hist(self, x, bins=None, range=None, density=False, weights=None,
             m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
             tops.append(m)
         tops = np.array(tops, float)  # causes problems later if it's an int
+        bins = np.array(bins, float)  # causes problems if float16
         if stacked:
             tops = tops.cumsum(axis=0)
             # If a stacked density plot, normalize so the area of all the

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -1863,6 +1863,21 @@ def test_hist_bar_empty():
     ax.hist([], histtype='bar')
 
 
+def test_hist_float16():
+    np.random.seed(19680801)
+    values = np.clip(
+        np.random.normal(0.5, 0.3, size=1000), 0, 1).astype(np.float16)
+    h = plt.hist(values, bins=3, alpha=0.5)
+    bc = h[2]
+    # Check that there are no overlapping rectangles
+    for r in range(1, len(bc)):
+        rleft = bc[r-1].get_corners()
+        rright = bc[r].get_corners()
+        # right hand position of left rectangle <=
+        # left hand position of right rectangle
+        assert rleft[1][0] <= rright[0][0]
+
+
 @image_comparison(['hist_step_empty.png'], remove_text=True)
 def test_hist_step_empty():
     # From #3886: creating hist from empty dataset raises ValueError

```


## Code snippets

### 1 - examples/lines_bars_and_markers/filled_step.py:

Start line: 179, End line: 237

```python
# set up histogram function to fixed bins
edges = np.linspace(-3, 3, 20, endpoint=True)
hist_func = partial(np.histogram, bins=edges)

# set up style cycles
color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
label_cycle = cycler(label=['set {n}'.format(n=n) for n in range(4)])
hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

# Fixing random state for reproducibility
np.random.seed(19680801)

stack_data = np.random.randn(4, 12250)
dict_data = dict(zip((c['label'] for c in label_cycle), stack_data))

###############################################################################
# Work with plain arrays

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
arts = stack_hist(ax1, stack_data, color_cycle + label_cycle + hatch_cycle,
                  hist_func=hist_func)

arts = stack_hist(ax2, stack_data, color_cycle,
                  hist_func=hist_func,
                  plot_kwargs=dict(edgecolor='w', orientation='h'))
ax1.set_ylabel('counts')
ax1.set_xlabel('x')
ax2.set_xlabel('counts')
ax2.set_ylabel('x')

###############################################################################
# Work with labeled data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5),
                               tight_layout=True, sharey=True)

arts = stack_hist(ax1, dict_data, color_cycle + hatch_cycle,
                  hist_func=hist_func)

arts = stack_hist(ax2, dict_data, color_cycle + hatch_cycle,
                  hist_func=hist_func, labels=['set 0', 'set 3'])
ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax1.set_xlabel('counts')
ax1.set_ylabel('x')
ax2.set_ylabel('x')

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.fill_betweenx` / `matplotlib.pyplot.fill_betweenx`
#    - `matplotlib.axes.Axes.fill_between` / `matplotlib.pyplot.fill_between`
#    - `matplotlib.axis.Axis.set_major_locator`
```
### 2 - examples/lines_bars_and_markers/scatter_hist.py:

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
### 3 - tutorials/introductory/quick_start.py:

Start line: 291, End line: 388

```python
x = mu + sigma * np.random.randn(10000)
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# the histogram of the data
n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)

ax.set_xlabel('Length [cm]')
ax.set_ylabel('Probability')
ax.set_title('Aardvark lengths\n (not really)')
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
ax.axis([55, 175, 0, 0.03])
ax.grid(True);

###############################################################################
# All of the `~.Axes.text` functions return a `matplotlib.text.Text`
# instance.  Just as with lines above, you can customize the properties by
# passing keyword arguments into the text functions::
#
#   t = ax.set_xlabel('my data', fontsize=14, color='red')
#
# These properties are covered in more detail in
# :doc:`/tutorials/text/text_props`.
#
# Using mathematical expressions in text
# --------------------------------------
#
# Matplotlib accepts TeX equation expressions in any text expression.
# For example to write the expression :math:`\sigma_i=15` in the title,
# you can write a TeX expression surrounded by dollar signs::
#
#     ax.set_title(r'$\sigma_i=15$')
#
# where the ``r`` preceding the title string signifies that the string is a
# *raw* string and not to treat backslashes as python escapes.
# Matplotlib has a built-in TeX expression parser and
# layout engine, and ships its own math fonts – for details see
# :doc:`/tutorials/text/mathtext`.  You can also use LaTeX directly to format
# your text and incorporate the output directly into your display figures or
# saved postscript – see :doc:`/tutorials/text/usetex`.
#
# Annotations
# -----------
#
# We can also annotate points on a plot, often by connecting an arrow pointing
# to *xy*, to a piece of text at *xytext*:

fig, ax = plt.subplots(figsize=(5, 2.7))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_ylim(-2, 2);

###############################################################################
# In this basic example, both *xy* and *xytext* are in data coordinates.
# There are a variety of other coordinate systems one can choose -- see
# :ref:`annotations-tutorial` and :ref:`plotting-guide-annotation` for
# details.  More examples also can be found in
# :doc:`/gallery/text_labels_and_annotations/annotation_demo`.
#
# Legends
# -------
#
# Often we want to identify lines or markers with a `.Axes.legend`:

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(np.arange(len(data1)), data1, label='data1')
ax.plot(np.arange(len(data2)), data2, label='data2')
ax.plot(np.arange(len(data3)), data3, 'd', label='data3')
ax.legend();

##############################################################################
# Legends in Matplotlib are quite flexible in layout, placement, and what
# Artists they can represent. They are discussed in detail in
# :doc:`/tutorials/intermediate/legend_guide`.
#
# Axis scales and ticks
# =====================
#
# Each Axes has two (or three) `~.axis.Axis` objects representing the x- and
# y-axis. These control the *scale* of the Axis, the tick *locators* and the
# tick *formatters*. Additional Axes can be attached to display further Axis
# objects.
#
# Scales
# ------
#
# In addition to the linear scale, Matplotlib supplies non-linear scales,
# such as a log-scale.  Since log-scales are used so much there are also
# direct methods like `~.Axes.loglog`, `~.Axes.semilogx`, and
# `~.Axes.semilogy`.  There are a number of scales (see
# :doc:`/gallery/scales/scales` for other examples).  Here we set the scale
# manually:

fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), layout='constrained')
```
### 4 - examples/statistics/hist.py:

Start line: 1, End line: 113

```python
"""
==========
Histograms
==========

How to plot histograms with Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

###############################################################################
# Generate data and plot a simple histogram
# -----------------------------------------
#
# To generate a 1D histogram we only need a single vector of numbers. For a 2D
# histogram we'll need a second vector. We'll generate both below, and show
# the histogram for each vector.

N_points = 100000
n_bins = 20

# Generate two normal distributions
dist1 = rng.standard_normal(N_points)
dist2 = 0.4 * rng.standard_normal(N_points) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(dist1, bins=n_bins)
axs[1].hist(dist2, bins=n_bins)


###############################################################################
# Updating histogram colors
# -------------------------
#
# The histogram method returns (among other things) a ``patches`` object. This
# gives us access to the properties of the objects drawn. Using this, we can
# edit the histogram to our liking. Let's change the color of each bar
# based on its y value.

fig, axs = plt.subplots(1, 2, tight_layout=True)

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs[0].hist(dist1, bins=n_bins)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# We can also normalize our inputs by the total number of counts
axs[1].hist(dist1, bins=n_bins, density=True)

# Now we format the y-axis to display percentage
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))


###############################################################################
# Plot a 2D histogram
# -------------------
#
# To plot a 2D histogram, one only needs two vectors of the same length,
# corresponding to each axis of the histogram.

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(dist1, dist2)


###############################################################################
# Customizing your histogram
# --------------------------
#
# Customizing a 2D histogram is similar to the 1D case, you can control
# visual components such as the bin size or color normalization.

fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True,
                        tight_layout=True)

# We can increase the number of bins on each axis
axs[0].hist2d(dist1, dist2, bins=40)

# As well as define normalization of the colors
axs[1].hist2d(dist1, dist2, bins=40, norm=colors.LogNorm())

# We can also define custom numbers of bins for each axis
axs[2].hist2d(dist1, dist2, bins=(80, 10), norm=colors.LogNorm())

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.pyplot.hist2d`
#    - `matplotlib.ticker.PercentFormatter`
```
### 5 - examples/statistics/histogram_features.py:

Start line: 1, End line: 60

```python
"""
==============================================
Some features of the histogram (hist) function
==============================================

In addition to the basic histogram, this demo shows a few optional features:

* Setting the number of data bins.
* The *density* parameter, which normalizes bin heights so that the integral of
  the histogram is 1. The resulting histogram is an approximation of the
  probability density function.

Selecting different bin counts and sizes can significantly affect the shape
of a histogram. The Astropy docs have a great section_ on how to select these
parameters.

.. _section: http://docs.astropy.org/en/stable/visualization/histogram.html
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_xlabel`
#    - `matplotlib.axes.Axes.set_ylabel`
```
### 6 - examples/statistics/time_series_histogram.py:

Start line: 1, End line: 72

```python
"""
=====================
Time Series Histogram
=====================

This example demonstrates how to efficiently visualize large numbers of time
series in a way that could potentially reveal hidden substructure and patterns
that are not immediately obvious, and display them in a visually appealing way.

In this example, we generate multiple sinusoidal "signal" series that are
buried under a larger number of random walk "noise/background" series. For an
unbiased Gaussian random walk with standard deviation of σ, the RMS deviation
from the origin after n steps is σ*sqrt(n). So in order to keep the sinusoids
visible on the same scale as the random walks, we scale the amplitude by the
random walk RMS. In addition, we also introduce a small random offset ``phi``
to shift the sines left/right, and some additive random noise to shift
individual data points up/down to make the signal a bit more "realistic" (you
wouldn't expect a perfect sine wave to appear in your data).

The first plot shows the typical way of visualizing multiple time series by
overlaying them on top of each other with ``plt.plot`` and a small value of
``alpha``. The second and third plots show how to reinterpret the data as a 2d
histogram, with optional interpolation between data points, by using
``np.histogram2d`` and ``plt.pcolormesh``.
"""
from copy import copy
import time

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(nrows=3, figsize=(6, 8), constrained_layout=True)

# Make some data; a 1D random walk + small fraction of sine waves
num_series = 1000
num_points = 100
SNR = 0.10  # Signal to Noise Ratio
x = np.linspace(0, 4 * np.pi, num_points)
# Generate unbiased Gaussian random walks
Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)
# Generate sinusoidal signals
num_signal = int(round(SNR * num_series))
phi = (np.pi / 8) * np.random.randn(num_signal, 1)  # small random offset
Y[-num_signal:] = (
    np.sqrt(np.arange(num_points))[None, :]  # random walk RMS scaling factor
    * (np.sin(x[None, :] - phi)
       + 0.05 * np.random.randn(num_signal, num_points))  # small random noise
)


# Plot series using `plot` and a small value of `alpha`. With this view it is
# very difficult to observe the sinusoidal behavior because of how many
# overlapping series there are. It also takes a bit of time to run because so
# many individual artists need to be generated.
tic = time.time()
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
toc = time.time()
axes[0].set_title("Line plot with alpha")
print(f"{toc-tic:.3f} sec. elapsed")


# Now we will convert the multiple time series into a histogram. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# Linearly interpolate between the points in each time series
num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.empty((num_series, num_fine), dtype=float)
for i in range(num_series):
    y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
```
### 7 - tutorials/introductory/pyplot.py:

Start line: 329, End line: 422

```python
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

###############################################################################
# All of the `~.pyplot.text` functions return a `matplotlib.text.Text`
# instance.  Just as with lines above, you can customize the properties by
# passing keyword arguments into the text functions or using `~.pyplot.setp`::
#
#   t = plt.xlabel('my data', fontsize=14, color='red')
#
# These properties are covered in more detail in :doc:`/tutorials/text/text_props`.
#
#
# Using mathematical expressions in text
# --------------------------------------
#
# matplotlib accepts TeX equation expressions in any text expression.
# For example to write the expression :math:`\sigma_i=15` in the title,
# you can write a TeX expression surrounded by dollar signs::
#
#     plt.title(r'$\sigma_i=15$')
#
# The ``r`` preceding the title string is important -- it signifies
# that the string is a *raw* string and not to treat backslashes as
# python escapes.  matplotlib has a built-in TeX expression parser and
# layout engine, and ships its own math fonts -- for details see
# :doc:`/tutorials/text/mathtext`.  Thus you can use mathematical text across platforms
# without requiring a TeX installation.  For those who have LaTeX and
# dvipng installed, you can also use LaTeX to format your text and
# incorporate the output directly into your display figures or saved
# postscript -- see :doc:`/tutorials/text/usetex`.
#
#
# Annotating text
# ---------------
#
# The uses of the basic `~.pyplot.text` function above
# place text at an arbitrary position on the Axes.  A common use for
# text is to annotate some feature of the plot, and the
# `~.pyplot.annotate` method provides helper
# functionality to make annotations easy.  In an annotation, there are
# two points to consider: the location being annotated represented by
# the argument ``xy`` and the location of the text ``xytext``.  Both of
# these arguments are ``(x, y)`` tuples.

ax = plt.subplot()

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()

###############################################################################
# In this basic example, both the ``xy`` (arrow tip) and ``xytext``
# locations (text location) are in data coordinates.  There are a
# variety of other coordinate systems one can choose -- see
# :ref:`annotations-tutorial` and :ref:`plotting-guide-annotation` for
# details.  More examples can be found in
# :doc:`/gallery/text_labels_and_annotations/annotation_demo`.
#
#
# Logarithmic and other nonlinear axes
# ====================================
#
# :mod:`matplotlib.pyplot` supports not only linear axis scales, but also
# logarithmic and logit scales. This is commonly used if data spans many orders
# of magnitude. Changing the scale of an axis is easy:
#
#     plt.xscale('log')
#
# An example of four plots with the same data and different scales for the y axis
# is shown below.

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
```
### 8 - examples/lines_bars_and_markers/psd_demo.py:

Start line: 82, End line: 153

```python
ax3.set_ylabel('')
ax3.set_title('block size')

# Plot the PSD with different amounts of overlap between blocks
ax4 = fig.add_subplot(gs[1, 2], sharex=ax2, sharey=ax2)
ax4.psd(y, NFFT=len(t) // 2, pad_to=len(t), noverlap=0, Fs=fs)
ax4.psd(y, NFFT=len(t) // 2, pad_to=len(t),
        noverlap=int(0.05 * len(t) / 2.), Fs=fs)
ax4.psd(y, NFFT=len(t) // 2, pad_to=len(t),
        noverlap=int(0.2 * len(t) / 2.), Fs=fs)
ax4.set_ylabel('')
ax4.set_title('overlap')

plt.show()


###############################################################################
# This is a ported version of a MATLAB example from the signal
# processing toolbox that showed some difference at one time between
# Matplotlib's and MATLAB's scaling of the PSD.

fs = 1000
t = np.linspace(0, 0.3, 301)
A = np.array([2, 8]).reshape(-1, 1)
f = np.array([150, 140]).reshape(-1, 1)
xn = (A * np.sin(2 * np.pi * f * t)).sum(axis=0)
xn += 5 * np.random.randn(*t.shape)

fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

yticks = np.arange(-50, 30, 10)
yrange = (yticks[0], yticks[-1])
xticks = np.arange(0, 550, 100)

ax0.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
        scale_by_freq=True)
ax0.set_title('Periodogram')
ax0.set_yticks(yticks)
ax0.set_xticks(xticks)
ax0.grid(True)
ax0.set_ylim(yrange)

ax1.psd(xn, NFFT=150, Fs=fs, window=mlab.window_none, pad_to=512, noverlap=75,
        scale_by_freq=True)
ax1.set_title('Welch')
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_ylabel('')  # overwrite the y-label added by `psd`
ax1.grid(True)
ax1.set_ylim(yrange)

plt.show()

###############################################################################
# This is a ported version of a MATLAB example from the signal
# processing toolbox that showed some difference at one time between
# Matplotlib's and MATLAB's scaling of the PSD.
#
# It uses a complex signal so we can see that complex PSD's work properly.

prng = np.random.RandomState(19680801)  # to ensure reproducibility

fs = 1000
t = np.linspace(0, 0.3, 301)
A = np.array([2, 8]).reshape(-1, 1)
f = np.array([150, 140]).reshape(-1, 1)
xn = (A * np.exp(2j * np.pi * f * t)).sum(axis=0) + 5 * prng.randn(*t.shape)

fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

yticks = np.arange(-50, 30, 10)
yrange = (yticks[0], yticks[-1])
```
### 9 - lib/matplotlib/pyplot.py:

Start line: 2588, End line: 2600

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.hist)
def hist(
        x, bins=None, range=None, density=False, weights=None,
        cumulative=False, bottom=None, histtype='bar', align='mid',
        orientation='vertical', rwidth=None, log=False, color=None,
        label=None, stacked=False, *, data=None, **kwargs):
    return gca().hist(
        x, bins=bins, range=range, density=density, weights=weights,
        cumulative=cumulative, bottom=bottom, histtype=histtype,
        align=align, orientation=orientation, rwidth=rwidth, log=log,
        color=color, label=label, stacked=stacked,
        **({"data": data} if data is not None else {}), **kwargs)
```
### 10 - examples/statistics/multiple_histograms_side_by_side.py:

Start line: 1, End line: 72

```python
"""
==========================================
Producing multiple histograms side by side
==========================================

This example plots horizontal histograms of different samples along
a categorical x-axis. Additionally, the histograms are plotted to
be symmetrical about their x-position, thus making them very similar
to violin plots.

To make this highly specialized plot, we can't use the standard ``hist``
method. Instead we use ``barh`` to draw the horizontal bars directly. The
vertical positions and lengths of the bars are computed via the
``np.histogram`` function. The histograms for all the samples are
computed using the same range (min and max values) and number of bins,
so that the bins for each sample are in the same vertical positions.

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)
number_of_bins = 20

# An example of three data sets to compare
number_of_data_points = 387
labels = ["A", "B", "C"]
data_sets = [np.random.normal(0, 1, number_of_data_points),
             np.random.normal(6, 1, number_of_data_points),
             np.random.normal(-3, 1, number_of_data_points)]

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets
]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)

ax.set_xticks(x_locations, labels)

ax.set_ylabel("Data values")
ax.set_xlabel("Data sets")

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
```
### 65 - lib/matplotlib/axes/_axes.py:

Start line: 1, End line: 38

```python
import functools
import itertools
import logging
import math
from numbers import Integral, Number

import numpy as np
from numpy import ma

import matplotlib.category  # Register category unit converter as side-effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # Register date unit converter as side-effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data, rcParams
from matplotlib.axes._base import (
    _AxesBase, _TransformedBoundsLocator, _process_plot_format)
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer

_log = logging.getLogger(__name__)
```
