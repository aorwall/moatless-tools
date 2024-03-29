# matplotlib__matplotlib-26249

| **matplotlib/matplotlib** | `f017315dd5e56c367e43fc7458fd0ed5fd9482a2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 28685 |
| **Any found context length** | 28685 |
| **Avg pos** | 29.0 |
| **Min pos** | 29 |
| **Max pos** | 29 |
| **Top file pos** | 8 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/mpl_toolkits/mplot3d/axes3d.py b/lib/mpl_toolkits/mplot3d/axes3d.py
--- a/lib/mpl_toolkits/mplot3d/axes3d.py
+++ b/lib/mpl_toolkits/mplot3d/axes3d.py
@@ -2249,7 +2249,11 @@ def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,
             *[np.ravel(np.ma.filled(t, np.nan)) for t in [xs, ys, zs]])
         s = np.ma.ravel(s)  # This doesn't have to match x, y in size.
 
-        xs, ys, zs, s, c = cbook.delete_masked_points(xs, ys, zs, s, c)
+        xs, ys, zs, s, c, color = cbook.delete_masked_points(
+            xs, ys, zs, s, c, kwargs.get('color', None)
+            )
+        if kwargs.get('color', None):
+            kwargs['color'] = color
 
         # For xs and ys, 2D scatter() will do the copying.
         if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/mpl_toolkits/mplot3d/axes3d.py | 2252 | 2252 | 29 | 8 | 28685


## Problem Statement

```
[Bug]: ax.scatter (projection='3d') - incorrect handling of NaN 
### Bug summary

In axis 3D projection NaN values are not handled correctly, apparently the values are masked out (as it should be) but the mask is not applied to a color array that may not have NaN in the same position.

### Code for reproduction

\`\`\`python
import numpy as np
from matplotlib import pylab as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter([1,np.nan,3], [2,np.nan,4], [3, np.nan,5], color=[[.5,.5,.5,.5]]*3, s=11.5)
\`\`\`


### Actual outcome

\`\`\`python
ValueError                                Traceback (most recent call last)
Cell In[24], line 1
----> 1 ax.scatter([1,np.nan,3], [2,np.nan,4], [3, np.nan,5], color=[[.5,.5,.5,.5]]*3, s=11.5)

File ~/Python/lib/python3.11/site-packages/matplotlib/__init__.py:1442, in _preprocess_data.<locals>.inner(ax, data, *args, **kwargs)
   1439 @functools.wraps(func)
   1440 def inner(ax, *args, data=None, **kwargs):
   1441     if data is None:
-> 1442         return func(ax, *map(sanitize_sequence, args), **kwargs)
   1444     bound = new_sig.bind(ax, *args, **kwargs)
   1445     auto_label = (bound.arguments.get(label_namer)
   1446                   or bound.kwargs.get(label_namer))

File ~/Python/lib/python3.11/site-packages/mpl_toolkits/mplot3d/axes3d.py:2275, in Axes3D.scatter(self, xs, ys, zs, zdir, s, c, depthshade, *args, **kwargs)
   2272 if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.
   2273     zs = zs.copy()
-> 2275 patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
   2276 art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir,
   2277                                 depthshade=depthshade)
   2279 if self._zmargin < 0.05 and xs.size > 0:

File ~/Python/lib/python3.11/site-packages/matplotlib/__init__.py:1442, in _preprocess_data.<locals>.inner(ax, data, *args, **kwargs)
   1439 @functools.wraps(func)
   1440 def inner(ax, *args, data=None, **kwargs):
   1441     if data is None:
-> 1442         return func(ax, *map(sanitize_sequence, args), **kwargs)
   1444     bound = new_sig.bind(ax, *args, **kwargs)
   1445     auto_label = (bound.arguments.get(label_namer)
   1446                   or bound.kwargs.get(label_namer))

File ~/Python/lib/python3.11/site-packages/matplotlib/axes/_axes.py:4602, in Axes.scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, edgecolors, plotnonfinite, **kwargs)
   4599 if edgecolors is None:
   4600     orig_edgecolor = kwargs.get('edgecolor', None)
   4601 c, colors, edgecolors = \
-> 4602     self._parse_scatter_color_args(
   4603         c, edgecolors, kwargs, x.size,
   4604         get_next_color_func=self._get_patches_for_fill.get_next_color)
   4606 if plotnonfinite and colors is None:
   4607     c = np.ma.masked_invalid(c)

File ~/Python/lib/python3.11/site-packages/matplotlib/axes/_axes.py:4455, in Axes._parse_scatter_color_args(c, edgecolors, kwargs, xsize, get_next_color_func)
   4451     else:
   4452         if len(colors) not in (0, 1, xsize):
   4453             # NB: remember that a single color is also acceptable.
   4454             # Besides *colors* will be an empty array if c == 'none'.
-> 4455             raise invalid_shape_exception(len(colors), xsize)
   4456 else:
   4457     colors = None  # use cmap, norm after collection is created

ValueError: 'c' argument has 3 elements, which is inconsistent with 'x' and 'y' with size 2.

\`\`\`

### Expected outcome

A plot with the first and 3rd data point.

### Additional information

Unconditionally reproducible.  

I have not seen this before, but I may never have called it this way before.

### Operating system

Fedora 38

### Matplotlib Version

3.7.1

### Matplotlib Backend

TkAgg

### Python version

3.11.4

### Jupyter version

IPython 8.14.0

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 lib/matplotlib/axes/_axes.py | 4052 | 4686| 6250 | 6250 | 76546 | 
| 2 | 2 galleries/examples/images_contours_and_fields/pcolor_demo.py | 1 | 97| 793 | 7043 | 77596 | 
| 3 | 3 galleries/examples/mplot3d/scatter3d.py | 1 | 41| 278 | 7321 | 77874 | 
| 4 | 4 galleries/users_explain/quick_start.py | 489 | 575| 1016 | 8337 | 83881 | 
| 5 | 5 galleries/plot_types/3D/scatter3d_simple.py | 1 | 30| 156 | 8493 | 84037 | 
| 6 | 6 galleries/examples/lines_bars_and_markers/masked_demo.py | 1 | 51| 410 | 8903 | 84447 | 
| 7 | 6 lib/matplotlib/axes/_axes.py | 4687 | 5159| 978 | 9881 | 84447 | 
| 8 | 7 galleries/examples/mplot3d/2dcollections3d.py | 1 | 49| 413 | 10294 | 84860 | 
| 9 | 7 galleries/users_explain/quick_start.py | 227 | 304| 835 | 11129 | 84860 | 
| 10 | **8 lib/mpl_toolkits/mplot3d/axes3d.py** | 1593 | 1666| 790 | 11919 | 115846 | 
| 11 | 8 galleries/users_explain/quick_start.py | 305 | 402| 968 | 12887 | 115846 | 
| 12 | 9 galleries/examples/pie_and_polar_charts/polar_scatter.py | 1 | 70| 513 | 13400 | 116359 | 
| 13 | 10 galleries/examples/lines_bars_and_markers/filled_step.py | 180 | 238| 493 | 13893 | 117985 | 
| 14 | 11 galleries/examples/specialty_plots/leftventricle_bullseye.py | 94 | 155| 697 | 14590 | 119590 | 
| 15 | 11 galleries/users_explain/quick_start.py | 403 | 488| 925 | 15515 | 119590 | 
| 16 | 12 galleries/examples/shapes_and_collections/scatter.py | 1 | 32| 169 | 15684 | 119759 | 
| 17 | 13 galleries/examples/images_contours_and_fields/pcolormesh_grids.py | 1 | 80| 802 | 16486 | 121048 | 
| 18 | 13 lib/matplotlib/axes/_axes.py | 5295 | 5857| 5675 | 22161 | 121048 | 
| 19 | 14 lib/matplotlib/axes/_base.py | 241 | 312| 744 | 22905 | 159893 | 
| 20 | 14 galleries/examples/images_contours_and_fields/pcolormesh_grids.py | 81 | 128| 487 | 23392 | 159893 | 
| 21 | 15 galleries/examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 695 | 24087 | 160997 | 
| 22 | 15 lib/matplotlib/axes/_base.py | 468 | 4536| 813 | 24900 | 160997 | 
| 23 | 16 galleries/examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 25425 | 161522 | 
| 24 | 17 galleries/examples/subplots_axes_and_figures/secondary_axis.py | 104 | 195| 656 | 26081 | 162955 | 
| 25 | 18 galleries/examples/lines_bars_and_markers/scatter_masked.py | 1 | 33| 244 | 26325 | 163199 | 
| 26 | 18 galleries/examples/images_contours_and_fields/pcolor_demo.py | 98 | 125| 257 | 26582 | 163199 | 
| 27 | 19 galleries/examples/images_contours_and_fields/quadmesh_demo.py | 1 | 52| 450 | 27032 | 163649 | 
| 28 | **19 lib/mpl_toolkits/mplot3d/axes3d.py** | 3092 | 3149| 862 | 27894 | 163649 | 
| **-> 29 <-** | **19 lib/mpl_toolkits/mplot3d/axes3d.py** | 2195 | 2267| 791 | 28685 | 163649 | 
| 30 | 20 lib/matplotlib/_cm.py | 1080 | 1211| 458 | 29143 | 192092 | 
| 31 | 21 galleries/users_explain/axes/colorbar_placement.py | 1 | 85| 764 | 29907 | 192987 | 
| 32 | **21 lib/mpl_toolkits/mplot3d/axes3d.py** | 2955 | 3015| 806 | 30713 | 192987 | 
| 33 | 22 galleries/users_explain/axes/arranging_axes.py | 101 | 185| 854 | 31567 | 197113 | 
| 34 | **22 lib/mpl_toolkits/mplot3d/axes3d.py** | 1772 | 1815| 484 | 32051 | 197113 | 


### Hint

```
Thank you for your clear report and diagnosis @2sn.  I have reproduced this with our `main` development branch.
Change this: 
https://github.com/matplotlib/matplotlib/blob/f017315dd5e56c367e43fc7458fd0ed5fd9482a2/lib/mpl_toolkits/mplot3d/axes3d.py#L2252

to 

\`\`\`  
if kwargs.get('color', None):
    xs, ys, zs, s, c, kwargs['color'] = cbook.delete_masked_points(xs, ys, zs, s, c, kwargs['color'])
else:
    xs, ys, zs, s, c = cbook.delete_masked_points(xs, ys, zs, s, c)
\`\`\`

on first sight solve the problem. 
I am willing to take this issue if no one alredy did it.
@artemshekh go for it
```

## Patch

```diff
diff --git a/lib/mpl_toolkits/mplot3d/axes3d.py b/lib/mpl_toolkits/mplot3d/axes3d.py
--- a/lib/mpl_toolkits/mplot3d/axes3d.py
+++ b/lib/mpl_toolkits/mplot3d/axes3d.py
@@ -2249,7 +2249,11 @@ def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,
             *[np.ravel(np.ma.filled(t, np.nan)) for t in [xs, ys, zs]])
         s = np.ma.ravel(s)  # This doesn't have to match x, y in size.
 
-        xs, ys, zs, s, c = cbook.delete_masked_points(xs, ys, zs, s, c)
+        xs, ys, zs, s, c, color = cbook.delete_masked_points(
+            xs, ys, zs, s, c, kwargs.get('color', None)
+            )
+        if kwargs.get('color', None):
+            kwargs['color'] = color
 
         # For xs and ys, 2D scatter() will do the copying.
         if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.

```

## Test Patch

```diff
diff --git a/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py b/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py
--- a/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py
+++ b/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py
@@ -2226,3 +2226,29 @@ def test_mutating_input_arrays_y_and_z(fig_test, fig_ref):
     y = [0.0, 0.0, 0.0]
     z = [0.0, 0.0, 0.0]
     ax2.plot(x, y, z, 'o-')
+
+
+def test_scatter_masked_color():
+    """
+    Test color parameter usage with non-finite coordinate arrays.
+
+    GH#26236
+    """
+
+    x = [np.nan, 1, 2,  1]
+    y = [0, np.inf, 2,  1]
+    z = [0, 1, -np.inf, 1]
+    colors = [
+        [0.0, 0.0, 0.0, 1],
+        [0.0, 0.0, 0.0, 1],
+        [0.0, 0.0, 0.0, 1],
+        [0.0, 0.0, 0.0, 1]
+    ]
+
+    fig = plt.figure()
+    ax = fig.add_subplot(projection='3d')
+    path3d = ax.scatter(x, y, z, color=colors)
+
+    # Assert sizes' equality
+    assert len(path3d.get_offsets()) ==\
+           len(super(type(path3d), path3d).get_facecolors())

```


## Code snippets

### 1 - lib/matplotlib/axes/_axes.py:

Start line: 4052, End line: 4686

```python
@_docstring.interpd
class Axes(_AxesBase):

    def bxp(self, bxpstats, positions=None, widths=None, vert=True,
            patch_artist=False, shownotches=False, showmeans=False,
            showcaps=True, showbox=True, showfliers=True,
            boxprops=None, whiskerprops=None, flierprops=None,
            medianprops=None, capprops=None, meanprops=None,
            meanline=False, manage_ticks=True, zorder=None,
            capwidths=None):
        """
        Drawing function for box and whisker plots.

        Make a box and whisker plot for each column of *x* or each
        vector in sequence *x*.  The box extends from the lower to
        upper quartile values of the data, with a line at the median.
        The whiskers extend from the box to show the range of the
        data.  Flier points are those past the end of the whiskers.

        Parameters
        ----------
        bxpstats : list of dicts
          A list of dictionaries containing stats for each boxplot.
          Required keys are:

          - ``med``: Median (scalar).
          - ``q1``, ``q3``: First & third quartiles (scalars).
          - ``whislo``, ``whishi``: Lower & upper whisker positions (scalars).

          Optional keys are:

          - ``mean``: Mean (scalar).  Needed if ``showmeans=True``.
          - ``fliers``: Data beyond the whiskers (array-like).
            Needed if ``showfliers=True``.
          - ``cilo``, ``cihi``: Lower & upper confidence intervals
            about the median. Needed if ``shownotches=True``.
          - ``label``: Name of the dataset (str).  If available,
            this will be used a tick label for the boxplot

        positions : array-like, default: [1, 2, ..., n]
          The positions of the boxes. The ticks and limits
          are automatically set to match the positions.

        widths : float or array-like, default: None
          The widths of the boxes.  The default is
          ``clip(0.15*(distance between extreme positions), 0.15, 0.5)``.

        capwidths : float or array-like, default: None
          Either a scalar or a vector and sets the width of each cap.
          The default is ``0.5*(with of the box)``, see *widths*.

        vert : bool, default: True
          If `True` (default), makes the boxes vertical.
          If `False`, makes horizontal boxes.

        patch_artist : bool, default: False
          If `False` produces boxes with the `.Line2D` artist.
          If `True` produces boxes with the `~matplotlib.patches.Patch` artist.

        shownotches, showmeans, showcaps, showbox, showfliers : bool
          Whether to draw the CI notches, the mean value (both default to
          False), the caps, the box, and the fliers (all three default to
          True).

        boxprops, whiskerprops, capprops, flierprops, medianprops, meanprops :\
        ptional
          Artist properties for the boxes, whiskers, caps, fliers, medians, and
          means.

        meanline : bool, default: False
          If `True` (and *showmeans* is `True`), will try to render the mean
          as a line spanning the full width of the box according to
          *meanprops*. Not recommended if *shownotches* is also True.
          Otherwise, means will be shown as points.

        manage_ticks : bool, default: True
          If True, the tick locations and labels will be adjusted to match the
          boxplot positions.

        zorder : float, default: ``Line2D.zorder = 2``
          The zorder of the resulting boxplot.

        Returns
        -------
        dict
          A dictionary mapping each component of the boxplot to a list
          of the `.Line2D` instances created. That dictionary has the
          following keys (assuming vertical boxplots):

          - ``boxes``: main bodies of the boxplot showing the quartiles, and
            the median's confidence intervals if enabled.
          - ``medians``: horizontal lines at the median of each box.
          - ``whiskers``: vertical lines up to the last non-outlier data.
          - ``caps``: horizontal lines at the ends of the whiskers.
          - ``fliers``: points representing data beyond the whiskers (fliers).
          - ``means``: points or lines representing the means.

        Examples
        --------
        .. plot:: gallery/statistics/bxp.py
        """

        # lists of artists to be output
        whiskers = []
        caps = []
        boxes = []
        medians = []
        means = []
        fliers = []

        # empty list of xticklabels
        datalabels = []

        # Use default zorder if none specified
        if zorder is None:
            zorder = mlines.Line2D.zorder

        zdelta = 0.1

        def merge_kw_rc(subkey, explicit, zdelta=0, usemarker=True):
            d = {k.split('.')[-1]: v for k, v in mpl.rcParams.items()
                 if k.startswith(f'boxplot.{subkey}props')}
            d['zorder'] = zorder + zdelta
            if not usemarker:
                d['marker'] = ''
            d.update(cbook.normalize_kwargs(explicit, mlines.Line2D))
            return d

        box_kw = {
            'linestyle': mpl.rcParams['boxplot.boxprops.linestyle'],
            'linewidth': mpl.rcParams['boxplot.boxprops.linewidth'],
            'edgecolor': mpl.rcParams['boxplot.boxprops.color'],
            'facecolor': ('white' if mpl.rcParams['_internal.classic_mode']
                          else mpl.rcParams['patch.facecolor']),
            'zorder': zorder,
            **cbook.normalize_kwargs(boxprops, mpatches.PathPatch)
        } if patch_artist else merge_kw_rc('box', boxprops, usemarker=False)
        whisker_kw = merge_kw_rc('whisker', whiskerprops, usemarker=False)
        cap_kw = merge_kw_rc('cap', capprops, usemarker=False)
        flier_kw = merge_kw_rc('flier', flierprops)
        median_kw = merge_kw_rc('median', medianprops, zdelta, usemarker=False)
        mean_kw = merge_kw_rc('mean', meanprops, zdelta)
        removed_prop = 'marker' if meanline else 'linestyle'
        # Only remove the property if it's not set explicitly as a parameter.
        if meanprops is None or removed_prop not in meanprops:
            mean_kw[removed_prop] = ''

        # vertical or horizontal plot?
        maybe_swap = slice(None) if vert else slice(None, None, -1)

        def do_plot(xs, ys, **kwargs):
            return self.plot(*[xs, ys][maybe_swap], **kwargs)[0]

        def do_patch(xs, ys, **kwargs):
            path = mpath.Path._create_closed(
                np.column_stack([xs, ys][maybe_swap]))
            patch = mpatches.PathPatch(path, **kwargs)
            self.add_artist(patch)
            return patch

        # input validation
        N = len(bxpstats)
        datashape_message = ("List of boxplot statistics and `{0}` "
                             "values must have same the length")
        # check position
        if positions is None:
            positions = list(range(1, N + 1))
        elif len(positions) != N:
            raise ValueError(datashape_message.format("positions"))

        positions = np.array(positions)
        if len(positions) > 0 and not all(isinstance(p, Real) for p in positions):
            raise TypeError("positions should be an iterable of numbers")

        # width
        if widths is None:
            widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
        elif np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format("widths"))

        # capwidth
        if capwidths is None:
            capwidths = 0.5 * np.array(widths)
        elif np.isscalar(capwidths):
            capwidths = [capwidths] * N
        elif len(capwidths) != N:
            raise ValueError(datashape_message.format("capwidths"))

        for pos, width, stats, capwidth in zip(positions, widths, bxpstats,
                                               capwidths):
            # try to find a new label
            datalabels.append(stats.get('label', pos))

            # whisker coords
            whis_x = [pos, pos]
            whislo_y = [stats['q1'], stats['whislo']]
            whishi_y = [stats['q3'], stats['whishi']]
            # cap coords
            cap_left = pos - capwidth * 0.5
            cap_right = pos + capwidth * 0.5
            cap_x = [cap_left, cap_right]
            cap_lo = np.full(2, stats['whislo'])
            cap_hi = np.full(2, stats['whishi'])
            # box and median coords
            box_left = pos - width * 0.5
            box_right = pos + width * 0.5
            med_y = [stats['med'], stats['med']]
            # notched boxes
            if shownotches:
                notch_left = pos - width * 0.25
                notch_right = pos + width * 0.25
                box_x = [box_left, box_right, box_right, notch_right,
                         box_right, box_right, box_left, box_left, notch_left,
                         box_left, box_left]
                box_y = [stats['q1'], stats['q1'], stats['cilo'],
                         stats['med'], stats['cihi'], stats['q3'],
                         stats['q3'], stats['cihi'], stats['med'],
                         stats['cilo'], stats['q1']]
                med_x = [notch_left, notch_right]
            # plain boxes
            else:
                box_x = [box_left, box_right, box_right, box_left, box_left]
                box_y = [stats['q1'], stats['q1'], stats['q3'], stats['q3'],
                         stats['q1']]
                med_x = [box_left, box_right]

            # maybe draw the box
            if showbox:
                do_box = do_patch if patch_artist else do_plot
                boxes.append(do_box(box_x, box_y, **box_kw))
            # draw the whiskers
            whiskers.append(do_plot(whis_x, whislo_y, **whisker_kw))
            whiskers.append(do_plot(whis_x, whishi_y, **whisker_kw))
            # maybe draw the caps
            if showcaps:
                caps.append(do_plot(cap_x, cap_lo, **cap_kw))
                caps.append(do_plot(cap_x, cap_hi, **cap_kw))
            # draw the medians
            medians.append(do_plot(med_x, med_y, **median_kw))
            # maybe draw the means
            if showmeans:
                if meanline:
                    means.append(do_plot(
                        [box_left, box_right], [stats['mean'], stats['mean']],
                        **mean_kw
                    ))
                else:
                    means.append(do_plot([pos], [stats['mean']], **mean_kw))
            # maybe draw the fliers
            if showfliers:
                flier_x = np.full(len(stats['fliers']), pos, dtype=np.float64)
                flier_y = stats['fliers']
                fliers.append(do_plot(flier_x, flier_y, **flier_kw))

        if manage_ticks:
            axis_name = "x" if vert else "y"
            interval = getattr(self.dataLim, f"interval{axis_name}")
            axis = self._axis_map[axis_name]
            positions = axis.convert_units(positions)
            # The 0.5 additional padding ensures reasonable-looking boxes
            # even when drawing a single box.  We set the sticky edge to
            # prevent margins expansion, in order to match old behavior (back
            # when separate calls to boxplot() would completely reset the axis
            # limits regardless of what was drawn before).  The sticky edges
            # are attached to the median lines, as they are always present.
            interval[:] = (min(interval[0], min(positions) - .5),
                           max(interval[1], max(positions) + .5))
            for median, position in zip(medians, positions):
                getattr(median.sticky_edges, axis_name).extend(
                    [position - .5, position + .5])
            # Modified from Axis.set_ticks and Axis.set_ticklabels.
            locator = axis.get_major_locator()
            if not isinstance(axis.get_major_locator(),
                              mticker.FixedLocator):
                locator = mticker.FixedLocator([])
                axis.set_major_locator(locator)
            locator.locs = np.array([*locator.locs, *positions])
            formatter = axis.get_major_formatter()
            if not isinstance(axis.get_major_formatter(),
                              mticker.FixedFormatter):
                formatter = mticker.FixedFormatter([])
                axis.set_major_formatter(formatter)
            formatter.seq = [*formatter.seq, *datalabels]

            self._request_autoscale_view()

        return dict(whiskers=whiskers, caps=caps, boxes=boxes,
                    medians=medians, fliers=fliers, means=means)

    @staticmethod
    def _parse_scatter_color_args(c, edgecolors, kwargs, xsize,
                                  get_next_color_func):
        """
        Helper function to process color related arguments of `.Axes.scatter`.

        Argument precedence for facecolors:

        - c (if not None)
        - kwargs['facecolor']
        - kwargs['facecolors']
        - kwargs['color'] (==kwcolor)
        - 'b' if in classic mode else the result of ``get_next_color_func()``

        Argument precedence for edgecolors:

        - kwargs['edgecolor']
        - edgecolors (is an explicit kw argument in scatter())
        - kwargs['color'] (==kwcolor)
        - 'face' if not in classic mode else None

        Parameters
        ----------
        c : color or sequence or sequence of color or None
            See argument description of `.Axes.scatter`.
        edgecolors : color or sequence of color or {'face', 'none'} or None
            See argument description of `.Axes.scatter`.
        kwargs : dict
            Additional kwargs. If these keys exist, we pop and process them:
            'facecolors', 'facecolor', 'edgecolor', 'color'
            Note: The dict is modified by this function.
        xsize : int
            The size of the x and y arrays passed to `.Axes.scatter`.
        get_next_color_func : callable
            A callable that returns a color. This color is used as facecolor
            if no other color is provided.

            Note, that this is a function rather than a fixed color value to
            support conditional evaluation of the next color.  As of the
            current implementation obtaining the next color from the
            property cycle advances the cycle. This must only happen if we
            actually use the color, which will only be decided within this
            method.

        Returns
        -------
        c
            The input *c* if it was not *None*, else a color derived from the
            other inputs or defaults.
        colors : array(N, 4) or None
            The facecolors as RGBA values, or *None* if a colormap is used.
        edgecolors
            The edgecolor.

        """
        facecolors = kwargs.pop('facecolors', None)
        facecolors = kwargs.pop('facecolor', facecolors)
        edgecolors = kwargs.pop('edgecolor', edgecolors)

        kwcolor = kwargs.pop('color', None)

        if kwcolor is not None and c is not None:
            raise ValueError("Supply a 'c' argument or a 'color'"
                             " kwarg but not both; they differ but"
                             " their functionalities overlap.")

        if kwcolor is not None:
            try:
                mcolors.to_rgba_array(kwcolor)
            except ValueError as err:
                raise ValueError(
                    "'color' kwarg must be a color or sequence of color "
                    "specs.  For a sequence of values to be color-mapped, use "
                    "the 'c' argument instead.") from err
            if edgecolors is None:
                edgecolors = kwcolor
            if facecolors is None:
                facecolors = kwcolor

        if edgecolors is None and not mpl.rcParams['_internal.classic_mode']:
            edgecolors = mpl.rcParams['scatter.edgecolors']

        c_was_none = c is None
        if c is None:
            c = (facecolors if facecolors is not None
                 else "b" if mpl.rcParams['_internal.classic_mode']
                 else get_next_color_func())
        c_is_string_or_strings = (
            isinstance(c, str)
            or (np.iterable(c) and len(c) > 0
                and isinstance(cbook._safe_first_finite(c), str)))

        def invalid_shape_exception(csize, xsize):
            return ValueError(
                f"'c' argument has {csize} elements, which is inconsistent "
                f"with 'x' and 'y' with size {xsize}.")

        c_is_mapped = False  # Unless proven otherwise below.
        valid_shape = True  # Unless proven otherwise below.
        if not c_was_none and kwcolor is None and not c_is_string_or_strings:
            try:  # First, does 'c' look suitable for value-mapping?
                c = np.asanyarray(c, dtype=float)
            except ValueError:
                pass  # Failed to convert to float array; must be color specs.
            else:
                # handle the documented special case of a 2D array with 1
                # row which as RGB(A) to broadcast.
                if c.shape == (1, 4) or c.shape == (1, 3):
                    c_is_mapped = False
                    if c.size != xsize:
                        valid_shape = False
                # If c can be either mapped values or an RGB(A) color, prefer
                # the former if shapes match, the latter otherwise.
                elif c.size == xsize:
                    c = c.ravel()
                    c_is_mapped = True
                else:  # Wrong size; it must not be intended for mapping.
                    if c.shape in ((3,), (4,)):
                        _api.warn_external(
                            "*c* argument looks like a single numeric RGB or "
                            "RGBA sequence, which should be avoided as value-"
                            "mapping will have precedence in case its length "
                            "matches with *x* & *y*.  Please use the *color* "
                            "keyword-argument or provide a 2D array "
                            "with a single row if you intend to specify "
                            "the same RGB or RGBA value for all points.")
                    valid_shape = False
        if not c_is_mapped:
            try:  # Is 'c' acceptable as PathCollection facecolors?
                colors = mcolors.to_rgba_array(c)
            except (TypeError, ValueError) as err:
                if "RGBA values should be within 0-1 range" in str(err):
                    raise
                else:
                    if not valid_shape:
                        raise invalid_shape_exception(c.size, xsize) from err
                    # Both the mapping *and* the RGBA conversion failed: pretty
                    # severe failure => one may appreciate a verbose feedback.
                    raise ValueError(
                        f"'c' argument must be a color, a sequence of colors, "
                        f"or a sequence of numbers, not {c!r}") from err
            else:
                if len(colors) not in (0, 1, xsize):
                    # NB: remember that a single color is also acceptable.
                    # Besides *colors* will be an empty array if c == 'none'.
                    raise invalid_shape_exception(len(colors), xsize)
        else:
            colors = None  # use cmap, norm after collection is created
        return c, colors, edgecolors

    @_preprocess_data(replace_names=["x", "y", "s", "linewidths",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"],
                      label_namer="y")
    @_docstring.interpd
    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None, *,
                edgecolors=None, plotnonfinite=False, **kwargs):
        """
        A scatter plot of *y* vs. *x* with varying marker size and/or color.

        Parameters
        ----------
        x, y : float or array-like, shape (n, )
            The data positions.

        s : float or array-like, shape (n, ), optional
            The marker size in points**2 (typographic points are 1/72 in.).
            Default is ``rcParams['lines.markersize'] ** 2``.

            The linewidth and edgecolor can visually interact with the marker
            size, and can lead to artifacts if the marker size is smaller than
            the linewidth.

            If the linewidth is greater than 0 and the edgecolor is anything
            but *'none'*, then the effective size of the marker will be
            increased by half the linewidth because the stroke will be centered
            on the edge of the shape.

            To eliminate the marker edge either set *linewidth=0* or
            *edgecolor='none'*.

        c : array-like or list of colors or color, optional
            The marker colors. Possible values:

            - A scalar or sequence of n numbers to be mapped to colors using
              *cmap* and *norm*.
            - A 2D array in which the rows are RGB or RGBA.
            - A sequence of colors of length n.
            - A single color format string.

            Note that *c* should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values to be
            colormapped. If you want to specify the same RGB or RGBA value for
            all points, use a 2D array with a single row.  Otherwise,
            value-matching will have precedence in case of a size matching with
            *x* and *y*.

            If you wish to specify a single color for all points
            prefer the *color* keyword argument.

            Defaults to `None`. In that case the marker color is determined
            by the value of *color*, *facecolor* or *facecolors*. In case
            those are not specified or `None`, the marker color is determined
            by the next color of the ``Axes``' current "shape and fill" color
            cycle. This cycle defaults to :rc:`axes.prop_cycle`.

        marker : `~.markers.MarkerStyle`, default: :rc:`scatter.marker`
            The marker style. *marker* can be either an instance of the class
            or the text shorthand for a particular marker.
            See :mod:`matplotlib.markers` for more information about marker
            styles.

        %(cmap_doc)s

            This parameter is ignored if *c* is RGB(A).

        %(norm_doc)s

            This parameter is ignored if *c* is RGB(A).

        %(vmin_vmax_doc)s

            This parameter is ignored if *c* is RGB(A).

        alpha : float, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        linewidths : float or array-like, default: :rc:`lines.linewidth`
            The linewidth of the marker edges. Note: The default *edgecolors*
            is 'face'. You may want to change this as well.

        edgecolors : {'face', 'none', *None*} or color or sequence of color, \
         :rc:`scatter.edgecolors`
            The edge color of the marker. Possible values:

            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.

            For non-filled markers, *edgecolors* is ignored. Instead, the color
            is determined like with 'face', i.e. from *c*, *colors*, or
            *facecolors*.

        plotnonfinite : bool, default: False
            Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf``
            or ``nan``). If ``True`` the points are drawn with the *bad*
            colormap color (see `.Colormap.set_bad`).

        Returns
        -------
        `~matplotlib.collections.PathCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs : `~matplotlib.collections.Collection` properties

        See Also
        --------
        plot : To plot scatter plots when markers are identical in size and
            color.

        Notes
        -----
        * The `.plot` function will be faster for scatterplots where markers
          don't vary in size or color.

        * Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in which
          case all masks will be combined and only unmasked points will be
          plotted.

        * Fundamentally, scatter works with 1D arrays; *x*, *y*, *s*, and *c*
          may be input as N-D arrays, but within scatter they will be
          flattened. The exception is *c*, which will be flattened only if its
          size matches the size of *x* and *y*.

        """
        # add edgecolors and linewidths to kwargs so they
        # can be processed by normailze_kwargs
        if edgecolors is not None:
            kwargs.update({'edgecolors': edgecolors})
        if linewidths is not None:
            kwargs.update({'linewidths': linewidths})

        kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
        # re direct linewidth and edgecolor so it can be
        # further processed by the rest of the function
        linewidths = kwargs.pop('linewidth', None)
        edgecolors = kwargs.pop('edgecolor', None)
        # Process **kwargs to handle aliases, conflicts with explicit kwargs:
        x, y = self._process_unit_info([("x", x), ("y", y)], kwargs)
        # np.ma.ravel yields an ndarray, not a masked array,
        # unless its argument is a masked array.
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        if s is None:
            s = (20 if mpl.rcParams['_internal.classic_mode'] else
                 mpl.rcParams['lines.markersize'] ** 2.0)
        s = np.ma.ravel(s)
        if (len(s) not in (1, x.size) or
                (not np.issubdtype(s.dtype, np.floating) and
                 not np.issubdtype(s.dtype, np.integer))):
            raise ValueError(
                "s must be a scalar, "
                "or float array-like with the same size as x and y")

        # get the original edgecolor the user passed before we normalize
        orig_edgecolor = edgecolors
        if edgecolors is None:
            orig_edgecolor = kwargs.get('edgecolor', None)
        c, colors, edgecolors = \
            self._parse_scatter_color_args(
                c, edgecolors, kwargs, x.size,
                get_next_color_func=self._get_patches_for_fill.get_next_color)

        if plotnonfinite and colors is None:
            c = np.ma.masked_invalid(c)
            x, y, s, edgecolors, linewidths = \
                cbook._combine_masks(x, y, s, edgecolors, linewidths)
        else:
            x, y, s, c, colors, edgecolors, linewidths = \
                cbook._combine_masks(
                    x, y, s, c, colors, edgecolors, linewidths)
        # Unmask edgecolors if it was actually a single RGB or RGBA.
        if (x.size in (3, 4)
                and np.ma.is_masked(edgecolors)
                and not np.ma.is_masked(orig_edgecolor)):
            edgecolors = edgecolors.data

        scales = s   # Renamed for readability below.

        # load default marker from rcParams
        if marker is None:
            marker = mpl.rcParams['scatter.marker']

        if isinstance(marker, mmarkers.MarkerStyle):
            marker_obj = marker
        else:
            marker_obj = mmarkers.MarkerStyle(marker)

        path = marker_obj.get_path().transformed(
            marker_obj.get_transform())
        # ... other code
```
### 2 - galleries/examples/images_contours_and_fields/pcolor_demo.py:

Start line: 1, End line: 97

```python
"""
===========
Pcolor demo
===========

Generating images with `~.axes.Axes.pcolor`.

Pcolor allows you to generate 2D image-style plots. Below we will show how
to do so in Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

# Fixing random state for reproducibility
np.random.seed(19680801)

# %%
# A simple pcolor demo
# --------------------

Z = np.random.rand(6, 10)

fig, (ax0, ax1) = plt.subplots(2, 1)

c = ax0.pcolor(Z)
ax0.set_title('default: no edges')

c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')

fig.tight_layout()
plt.show()

# %%
# Comparing pcolor with similar functions
# ---------------------------------------
#
# Demonstrates similarities between `~.axes.Axes.pcolor`,
# `~.axes.Axes.pcolormesh`, `~.axes.Axes.imshow` and
# `~.axes.Axes.pcolorfast` for drawing quadrilateral grids.
# Note that we call ``imshow`` with ``aspect="auto"`` so that it doesn't force
# the data pixels to be square (the default is ``aspect="equal"``).

# make these smaller to increase the resolution
dx, dy = 0.15, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-3:3+dy:dy, -3:3+dx:dx]
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -abs(z).max(), abs(z).max()

fig, axs = plt.subplots(2, 2)

ax = axs[0, 0]
c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolor')
fig.colorbar(c, ax=ax)

ax = axs[0, 1]
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
fig.colorbar(c, ax=ax)

ax = axs[1, 0]
c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto')
ax.set_title('image (nearest, aspect="auto")')
fig.colorbar(c, ax=ax)

ax = axs[1, 1]
c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolorfast')
fig.colorbar(c, ax=ax)

fig.tight_layout()
plt.show()


# %%
# Pcolor with a log scale
# -----------------------
#
# The following shows pcolor plots with a log scale.

N = 100
X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))

# A low hump with a spike coming out.
# Needs to have z/colour axis on a log scale, so we see both hump and spike.
# A linear scale only shows the spike.
Z1 = np.exp(-X**2 - Y**2)
```
### 3 - galleries/examples/mplot3d/scatter3d.py:

Start line: 1, End line: 41

```python
"""
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```
### 4 - galleries/users_explain/quick_start.py:

Start line: 489, End line: 575

```python
l1, = ax1.plot(t, s)
ax2 = ax1.twinx()
l2, = ax2.plot(t, range(len(t)), 'C1')
ax2.legend([l1, l2], ['Sine (left)', 'Straight (right)'])

ax3.plot(t, s)
ax3.set_xlabel('Angle [rad]')
ax4 = ax3.secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))
ax4.set_xlabel('Angle [°]')

# %%
# Color mapped data
# =================
#
# Often we want to have a third dimension in a plot represented by a colors in
# a colormap. Matplotlib has a number of plot types that do this:

X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

fig, axs = plt.subplots(2, 2, layout='constrained')
pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')
fig.colorbar(pc, ax=axs[0, 0])
axs[0, 0].set_title('pcolormesh()')

co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
fig.colorbar(co, ax=axs[0, 1])
axs[0, 1].set_title('contourf()')

pc = axs[1, 0].imshow(Z**2 * 100, cmap='plasma',
                          norm=mpl.colors.LogNorm(vmin=0.01, vmax=100))
fig.colorbar(pc, ax=axs[1, 0], extend='both')
axs[1, 0].set_title('imshow() with LogNorm()')

pc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r')
fig.colorbar(pc, ax=axs[1, 1], extend='both')
axs[1, 1].set_title('scatter()')

# %%
# Colormaps
# ---------
#
# These are all examples of Artists that derive from  `~.ScalarMappable`
# objects.  They all can set a linear mapping between *vmin* and *vmax* into
# the colormap specified by *cmap*.  Matplotlib has many colormaps to choose
# from (:ref:`colormaps`) you can make your
# own (:ref:`colormap-manipulation`) or download as
# `third-party packages
# <https://matplotlib.org/mpl-third-party/#colormaps-and-styles>`_.
#
# Normalizations
# --------------
#
# Sometimes we want a non-linear mapping of the data to the colormap, as
# in the ``LogNorm`` example above.  We do this by supplying the
# ScalarMappable with the *norm* argument instead of *vmin* and *vmax*.
# More normalizations are shown at :ref:`colormapnorms`.
#
# Colorbars
# ---------
#
# Adding a `~.Figure.colorbar` gives a key to relate the color back to the
# underlying data. Colorbars are figure-level Artists, and are attached to
# a ScalarMappable (where they get their information about the norm and
# colormap) and usually steal space from a parent Axes.  Placement of
# colorbars can be complex: see
# :ref:`colorbar_placement` for
# details.  You can also change the appearance of colorbars with the
# *extend* keyword to add arrows to the ends, and *shrink* and *aspect* to
# control the size.  Finally, the colorbar will have default locators
# and formatters appropriate to the norm.  These can be changed as for
# other Axis objects.
#
#
# Working with multiple Figures and Axes
# ======================================
#
# You can open multiple Figures with multiple calls to
# ``fig = plt.figure()`` or ``fig2, ax = plt.subplots()``.  By keeping the
# object references you can add Artists to either Figure.
#
# Multiple Axes can be added a number of ways, but the most basic is
# ``plt.subplots()`` as used above.  One can achieve more complex layouts,
# with Axes objects spanning columns or rows, using `~.pyplot.subplot_mosaic`.

fig, axd = plt.subplot_mosaic([['upleft', 'right'],
                               ['lowleft', 'right']], layout='constrained')
```
### 5 - galleries/plot_types/3D/scatter3d_simple.py:

Start line: 1, End line: 30

```python
"""
==============
3D scatterplot
==============

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.scatter`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# Make data
np.random.seed(19680801)
n = 100
rng = np.random.default_rng()
xs = rng.uniform(23, 32, n)
ys = rng.uniform(0, 100, n)
zs = rng.uniform(-50, -25, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(xs, ys, zs)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()
```
### 6 - galleries/examples/lines_bars_and_markers/masked_demo.py:

Start line: 1, End line: 51

```python
"""
==============================
Plotting masked and NaN values
==============================

Sometimes you need to plot data with missing values.

One possibility is to simply remove undesired data points. The line plotted
through the remaining data will be continuous, and not indicate where the
missing data is located.

If it is useful to have gaps in the line where the data is missing, then the
undesired points can be indicated using a `masked array`_ or by setting their
values to NaN. No marker will be drawn where either x or y are masked and, if
plotting with a line, it will be broken there.

.. _masked array:
   https://numpy.org/doc/stable/reference/maskedarray.generic.html

The following example illustrates the three cases:

1) Removing points.
2) Masking points.
3) Setting to NaN.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi/2, np.pi/2, 31)
y = np.cos(x)**3

# 1) remove points where y > 0.7
x2 = x[y <= 0.7]
y2 = y[y <= 0.7]

# 2) mask points where y > 0.7
y3 = np.ma.masked_where(y > 0.7, y)

# 3) set to NaN where y > 0.7
y4 = y.copy()
y4[y3 > 0.7] = np.nan

plt.plot(x*0.1, y, 'o-', color='lightgrey', label='No mask')
plt.plot(x2*0.4, y2, 'o-', label='Points removed')
plt.plot(x*0.7, y3, 'o-', label='Masked values')
plt.plot(x*1.0, y4, 'o-', label='NaN values')
plt.legend()
plt.title('Masked and NaN data')
plt.show()
```
### 7 - lib/matplotlib/axes/_axes.py:

Start line: 4687, End line: 5159

```python
@_docstring.interpd
class Axes(_AxesBase):

    @_preprocess_data(replace_names=["x", "y", "s", "linewidths",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"],
                      label_namer="y")
    @_docstring.interpd
    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None, *,
                edgecolors=None, plotnonfinite=False, **kwargs):
        # ... other code
        if not marker_obj.is_filled():
            if orig_edgecolor is not None:
                _api.warn_external(
                    f"You passed a edgecolor/edgecolors ({orig_edgecolor!r}) "
                    f"for an unfilled marker ({marker!r}).  Matplotlib is "
                    "ignoring the edgecolor in favor of the facecolor.  This "
                    "behavior may change in the future."
                )
            # We need to handle markers that cannot be filled (like
            # '+' and 'x') differently than markers that can be
            # filled, but have their fillstyle set to 'none'.  This is
            # to get:
            #
            #  - respecting the fillestyle if set
            #  - maintaining back-compatibility for querying the facecolor of
            #    the un-fillable markers.
            #
            # While not an ideal situation, but is better than the
            # alternatives.
            if marker_obj.get_fillstyle() == 'none':
                # promote the facecolor to be the edgecolor
                edgecolors = colors
                # set the facecolor to 'none' (at the last chance) because
                # we cannot fill a path if the facecolor is non-null
                # (which is defendable at the renderer level).
                colors = 'none'
            else:
                # if we are not nulling the face color we can do this
                # simpler
                edgecolors = 'face'

            if linewidths is None:
                linewidths = mpl.rcParams['lines.linewidth']
            elif np.iterable(linewidths):
                linewidths = [
                    lw if lw is not None else mpl.rcParams['lines.linewidth']
                    for lw in linewidths]

        offsets = np.ma.column_stack([x, y])

        collection = mcoll.PathCollection(
            (path,), scales,
            facecolors=colors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            offsets=offsets,
            offset_transform=kwargs.pop('transform', self.transData),
            alpha=alpha,
        )
        collection.set_transform(mtransforms.IdentityTransform())
        if colors is None:
            collection.set_array(c)
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            collection._scale_norm(norm, vmin, vmax)
        else:
            extra_kwargs = {
                    'cmap': cmap, 'norm': norm, 'vmin': vmin, 'vmax': vmax
                    }
            extra_keys = [k for k, v in extra_kwargs.items() if v is not None]
            if any(extra_keys):
                keys_str = ", ".join(f"'{k}'" for k in extra_keys)
                _api.warn_external(
                    "No data for colormapping provided via 'c'. "
                    f"Parameters {keys_str} will be ignored")
        collection._internal_update(kwargs)

        # Classic mode only:
        # ensure there are margins to allow for the
        # finite size of the symbols.  In v2.x, margins
        # are present by default, so we disable this
        # scatter-specific override.
        if mpl.rcParams['_internal.classic_mode']:
            if self._xmargin < 0.05 and x.size > 0:
                self.set_xmargin(0.05)
            if self._ymargin < 0.05 and x.size > 0:
                self.set_ymargin(0.05)

        self.add_collection(collection)
        self._request_autoscale_view()

        return collection

    @_preprocess_data(replace_names=["x", "y", "C"], label_namer="y")
    @_docstring.dedent_interpd
    def hexbin(self, x, y, C=None, gridsize=100, bins=None,
               xscale='linear', yscale='linear', extent=None,
               cmap=None, norm=None, vmin=None, vmax=None,
               alpha=None, linewidths=None, edgecolors='face',
               reduce_C_function=np.mean, mincnt=None, marginals=False,
               **kwargs):
        # ... other code
```
### 8 - galleries/examples/mplot3d/2dcollections3d.py:

Start line: 1, End line: 49

```python
"""
=======================
Plot 2D data on 3D plot
=======================

Demonstrates using ax.plot's *zdir* keyword to plot 2D data on
selective axes of a 3D plot.
"""

import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

# Plot scatterplot data (20 2D points per colour) on the x and z axes.
colors = ('r', 'g', 'b', 'k')

# Fixing random state for reproducibility
np.random.seed(19680801)

x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)
# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x, y) points are plotted on the x and z axes.
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35, roll=0)

plt.show()
```
### 9 - galleries/users_explain/quick_start.py:

Start line: 227, End line: 304

```python
data1, data2, data3, data4 = np.random.randn(4, 100)  # make 4 random data sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})

# %%
# Note that if you want to install these as a python package, or any other
# customizations you could use one of the many templates on the web;
# Matplotlib has one at `mpl-cookiecutter
# <https://github.com/matplotlib/matplotlib-extension-cookiecutter>`_
#
#
# Styling Artists
# ===============
#
# Most plotting methods have styling options for the Artists, accessible either
# when a plotting method is called, or from a "setter" on the Artist.  In the
# plot below we manually set the *color*, *linewidth*, and *linestyle* of the
# Artists created by `~.Axes.plot`, and we set the linestyle of the second line
# after the fact with `~.Line2D.set_linestyle`.

fig, ax = plt.subplots(figsize=(5, 2.7))
x = np.arange(len(data1))
ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')
l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)
l.set_linestyle(':')

# %%
# Colors
# ------
#
# Matplotlib has a very flexible array of colors that are accepted for most
# Artists; see :ref:`allowable color definitions <colors_def>` for a
# list of specifications. Some Artists will take multiple colors.  i.e. for
# a `~.Axes.scatter` plot, the edge of the markers can be different colors
# from the interior:

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter(data1, data2, s=50, facecolor='C0', edgecolor='k')

# %%
# Linewidths, linestyles, and markersizes
# ---------------------------------------
#
# Line widths are typically in typographic points (1 pt = 1/72 inch) and
# available for Artists that have stroked lines.  Similarly, stroked lines
# can have a linestyle.  See the :doc:`linestyles example
# </gallery/lines_bars_and_markers/linestyles>`.
#
# Marker size depends on the method being used.  `~.Axes.plot` specifies
# markersize in points, and is generally the "diameter" or width of the
# marker.  `~.Axes.scatter` specifies markersize as approximately
# proportional to the visual area of the marker.  There is an array of
# markerstyles available as string codes (see :mod:`~.matplotlib.markers`), or
# users can define their own `~.MarkerStyle` (see
# :doc:`/gallery/lines_bars_and_markers/marker_reference`):

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(data1, 'o', label='data1')
ax.plot(data2, 'd', label='data2')
ax.plot(data3, 'v', label='data3')
ax.plot(data4, 's', label='data4')
ax.legend()

# %%
#
# Labelling plots
# ===============
#
# Axes labels and text
# --------------------
#
# `~.Axes.set_xlabel`, `~.Axes.set_ylabel`, and `~.Axes.set_title` are used to
# add text in the indicated locations (see :ref:`text_intro`
# for more discussion).  Text can also be directly added to plots using
# `~.Axes.text`:

mu, sigma = 115, 15
```
### 10 - lib/mpl_toolkits/mplot3d/axes3d.py:

Start line: 1593, End line: 1666

```python
@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):

    def plot_surface(self, X, Y, Z, *, norm=None, vmin=None,
                     vmax=None, lightsource=None, **kwargs):
        # ... other code
        ccount = kwargs.pop('ccount', 50)

        if mpl.rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            compute_strides = has_count
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            compute_strides = not has_stride

        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))

        fcolors = kwargs.pop('facecolors', None)

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        if shade is None:
            raise ValueError("shade cannot be None.")

        colset = []  # the sampled facecolor
        if (rows - 1) % rstride == 0 and \
           (cols - 1) % cstride == 0 and \
           fcolors is None:
            polys = np.stack(
                [cbook._array_patch_perimeters(a, rstride, cstride)
                 for a in (X, Y, Z)],
                axis=-1)
        else:
            # evenly spaced, and including both endpoints
            row_inds = list(range(0, rows-1, rstride)) + [rows-1]
            col_inds = list(range(0, cols-1, cstride)) + [cols-1]

            polys = []
            for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
                for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                    ps = [
                        # +1 ensures we share edges between polygons
                        cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
                        for a in (X, Y, Z)
                    ]
                    # ps = np.stack(ps, axis=-1)
                    ps = np.array(ps).T
                    polys.append(ps)

                    if fcolors is not None:
                        colset.append(fcolors[rs][cs])

        # In cases where there are NaNs in the data (possibly from masked
        # arrays), artifacts can be introduced. Here check whether NaNs exist
        # and remove the entries if so
        if not isinstance(polys, np.ndarray) or np.isnan(polys).any():
            new_polys = []
            new_colset = []

            # Depending on fcolors, colset is either an empty list or has as
            # many elements as polys. In the former case new_colset results in
            # a list with None entries, that is discarded later.
            for p, col in itertools.zip_longest(polys, colset):
                new_poly = np.array(p)[~np.isnan(p).any(axis=1)]
                if len(new_poly):
                    new_polys.append(new_poly)
                    new_colset.append(col)

            # Replace previous polys and, if fcolors is not None, colset
            polys = new_polys
            if fcolors is not None:
                colset = new_colset

        # note that the striding causes some polygons to have more coordinates
        # than others
        # ... other code
```
### 28 - lib/mpl_toolkits/mplot3d/axes3d.py:

Start line: 3092, End line: 3149

```python
@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):

    @_preprocess_data(replace_names=["x", "y", "z", "xerr", "yerr", "zerr"])
    def errorbar(self, x, y, z, zerr=None, yerr=None, xerr=None, fmt='',
                 barsabove=False, errorevery=1, ecolor=None, elinewidth=None,
                 capsize=None, capthick=None, xlolims=False, xuplims=False,
                 ylolims=False, yuplims=False, zlolims=False, zuplims=False,
                 **kwargs):
        # ... other code
        for zdir, data, err, lolims, uplims in zip(
                ['x', 'y', 'z'], [x, y, z], [xerr, yerr, zerr],
                [xlolims, ylolims, zlolims], [xuplims, yuplims, zuplims]):

            dir_vector = art3d.get_dir_vector(zdir)
            i_zdir = i_xyz[zdir]

            if err is None:
                continue

            if not np.iterable(err):
                err = [err] * len(data)

            err = np.atleast_1d(err)

            # arrays fine here, they are booleans and hence not units
            lolims = np.broadcast_to(lolims, len(data)).astype(bool)
            uplims = np.broadcast_to(uplims, len(data)).astype(bool)

            # a nested list structure that expands to (xl,xh),(yl,yh),(zl,zh),
            # where x/y/z and l/h correspond to dimensions and low/high
            # positions of errorbars in a dimension we're looping over
            coorderr = [
                _extract_errs(err * dir_vector[i], coord, lolims, uplims)
                for i, coord in enumerate([x, y, z])]
            (xl, xh), (yl, yh), (zl, zh) = coorderr

            # draws capmarkers - flat caps orthogonal to the error bars
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                lo_caps_xyz = _apply_mask([xl, yl, zl], nolims & everymask)
                hi_caps_xyz = _apply_mask([xh, yh, zh], nolims & everymask)

                # setting '_' for z-caps and '|' for x- and y-caps;
                # these markers will rotate as the viewing angle changes
                cap_lo = art3d.Line3D(*lo_caps_xyz, ls='',
                                      marker=capmarker[i_zdir],
                                      **eb_cap_style)
                cap_hi = art3d.Line3D(*hi_caps_xyz, ls='',
                                      marker=capmarker[i_zdir],
                                      **eb_cap_style)
                self.add_line(cap_lo)
                self.add_line(cap_hi)
                caplines.append(cap_lo)
                caplines.append(cap_hi)

            if lolims.any():
                xh0, yh0, zh0 = _apply_mask([xh, yh, zh], lolims & everymask)
                self.quiver(xh0, yh0, zh0, *dir_vector, **eb_quiver_style)
            if uplims.any():
                xl0, yl0, zl0 = _apply_mask([xl, yl, zl], uplims & everymask)
                self.quiver(xl0, yl0, zl0, *-dir_vector, **eb_quiver_style)

            errline = art3d.Line3DCollection(np.array(coorderr).T,
                                             **eb_lines_style)
            self.add_collection(errline)
            errlines.append(errline)
            coorderrs.append(coorderr)
        # ... other code
```
### 29 - lib/mpl_toolkits/mplot3d/axes3d.py:

Start line: 2195, End line: 2267

```python
@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):

    @_preprocess_data(replace_names=["xs", "ys", "zs", "s",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"])
    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,
                *args, **kwargs):
        """
        Create a scatter plot.

        Parameters
        ----------
        xs, ys : array-like
            The data positions.
        zs : float or array-like, default: 0
            The z-positions. Either an array of the same length as *xs* and
            *ys* or a single value to place all points in the same plane.
        zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, default: 'z'
            The axis direction for the *zs*. This is useful when plotting 2D
            data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting
            *zdir* to 'y' then plots the data to the x-z-plane.

            See also :doc:`/gallery/mplot3d/2dcollections3d`.

        s : float or array-like, default: 20
            The marker size in points**2. Either an array of the same length
            as *xs* and *ys* or a single value to make all markers the same
            size.
        c : color, sequence, or sequence of colors, optional
            The marker color. Possible values:

            - A single color format string.
            - A sequence of colors of length n.
            - A sequence of n numbers to be mapped to colors using *cmap* and
              *norm*.
            - A 2D array in which the rows are RGB or RGBA.

            For more details see the *c* argument of `~.axes.Axes.scatter`.
        depthshade : bool, default: True
            Whether to shade the scatter markers to give the appearance of
            depth. Each call to ``scatter()`` will perform its depthshading
            independently.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            All other keyword arguments are passed on to `~.axes.Axes.scatter`.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`
        """

        had_data = self.has_data()
        zs_orig = zs

        xs, ys, zs = np.broadcast_arrays(
            *[np.ravel(np.ma.filled(t, np.nan)) for t in [xs, ys, zs]])
        s = np.ma.ravel(s)  # This doesn't have to match x, y in size.

        xs, ys, zs, s, c = cbook.delete_masked_points(xs, ys, zs, s, c)

        # For xs and ys, 2D scatter() will do the copying.
        if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.
            zs = zs.copy()

        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir,
                                        depthshade=depthshade)

        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)

        self.auto_scale_xyz(xs, ys, zs, had_data)

        return patches
```
### 32 - lib/mpl_toolkits/mplot3d/axes3d.py:

Start line: 2955, End line: 3015

```python
@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):

    @_preprocess_data(replace_names=["x", "y", "z", "xerr", "yerr", "zerr"])
    def errorbar(self, x, y, z, zerr=None, yerr=None, xerr=None, fmt='',
                 barsabove=False, errorevery=1, ecolor=None, elinewidth=None,
                 capsize=None, capthick=None, xlolims=False, xuplims=False,
                 ylolims=False, yuplims=False, zlolims=False, zuplims=False,
                 **kwargs):
        had_data = self.has_data()

        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        # Drop anything that comes in as None to use the default instead.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)

        self._process_unit_info([("x", x), ("y", y), ("z", z)], kwargs,
                                convert=False)

        # make sure all the args are iterable; use lists not arrays to
        # preserve units
        x = x if np.iterable(x) else [x]
        y = y if np.iterable(y) else [y]
        z = z if np.iterable(z) else [z]

        if not len(x) == len(y) == len(z):
            raise ValueError("'x', 'y', and 'z' must have the same size")

        everymask = self._errorevery_to_mask(x, errorevery)

        label = kwargs.pop("label", None)
        kwargs['label'] = '_nolegend_'

        # Create the main line and determine overall kwargs for child artists.
        # We avoid calling self.plot() directly, or self._get_lines(), because
        # that would call self._process_unit_info again, and do other indirect
        # data processing.
        (data_line, base_style), = self._get_lines._plot_args(
            (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
        art3d.line_2d_to_3d(data_line, zs=z)

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
            # Remove alpha=0 color that _process_plot_format returns.
            base_style.pop('color')

        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']

        # Eject any line-specific information from format string, as it's not
        # needed for bars or caps.
        for key in ['marker', 'markersize', 'markerfacecolor',
                    'markeredgewidth', 'markeredgecolor', 'markevery',
                    'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle',
                    'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle']:
            base_style.pop(key, None)

        # Make the style dict for the line collections (the bars).
        eb_lines_style = {**base_style, 'color': ecolor}
        # ... other code
```
### 34 - lib/mpl_toolkits/mplot3d/axes3d.py:

Start line: 1772, End line: 1815

```python
@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):

    def plot_wireframe(self, X, Y, Z, **kwargs):
        # ... other code
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        if rstride:
            rii = list(range(0, rows, rstride))
            # Add the last index only if needed
            if rows > 0 and rii[-1] != (rows - 1):
                rii += [rows-1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            # Add the last index only if needed
            if cols > 0 and cii[-1] != (cols - 1):
                cii += [cols-1]
        else:
            cii = []

        if rstride == 0 and cstride == 0:
            raise ValueError("Either rstride or cstride must be non zero")

        # If the inputs were empty, then just
        # reset everything.
        if Z.size == 0:
            rii = []
            cii = []

        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = ([list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(xlines, ylines, zlines)]
                 + [list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(txlines, tylines, tzlines)])

        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return linec
```
