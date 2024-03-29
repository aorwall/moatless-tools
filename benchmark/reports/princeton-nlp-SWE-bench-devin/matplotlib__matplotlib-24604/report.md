# matplotlib__matplotlib-24604

| **matplotlib/matplotlib** | `3393a4f22350e5df7aa8d3c7904e26e81428d2cd` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 8284 |
| **Any found context length** | 760 |
| **Avg pos** | 57.666666666666664 |
| **Min pos** | 1 |
| **Max pos** | 20 |
| **Top file pos** | 1 |
| **Missing snippets** | 14 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1759,6 +1759,25 @@ def get_tightbbox(self, renderer=None, bbox_extra_artists=None):
 
         return _bbox
 
+    @staticmethod
+    def _norm_per_subplot_kw(per_subplot_kw):
+        expanded = {}
+        for k, v in per_subplot_kw.items():
+            if isinstance(k, tuple):
+                for sub_key in k:
+                    if sub_key in expanded:
+                        raise ValueError(
+                            f'The key {sub_key!r} appears multiple times.'
+                            )
+                    expanded[sub_key] = v
+            else:
+                if k in expanded:
+                    raise ValueError(
+                        f'The key {k!r} appears multiple times.'
+                    )
+                expanded[k] = v
+        return expanded
+
     @staticmethod
     def _normalize_grid_string(layout):
         if '\n' not in layout:
@@ -1771,7 +1790,8 @@ def _normalize_grid_string(layout):
 
     def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                        width_ratios=None, height_ratios=None,
-                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
+                       empty_sentinel='.',
+                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
         """
         Build a layout of Axes based on ASCII art or nested lists.
 
@@ -1821,6 +1841,9 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
             The string notation allows only single character Axes labels and
             does not support nesting but is very terse.
 
+            The Axes identifiers may be `str` or a non-iterable hashable
+            object (e.g. `tuple` s may not be used).
+
         sharex, sharey : bool, default: False
             If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
             among all subplots.  In that case, tick label visibility and axis
@@ -1843,7 +1866,21 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
 
         subplot_kw : dict, optional
             Dictionary with keywords passed to the `.Figure.add_subplot` call
-            used to create each subplot.
+            used to create each subplot.  These values may be overridden by
+            values in *per_subplot_kw*.
+
+        per_subplot_kw : dict, optional
+            A dictionary mapping the Axes identifiers or tuples of identifiers
+            to a dictionary of keyword arguments to be passed to the
+            `.Figure.add_subplot` call used to create each subplot.  The values
+            in these dictionaries have precedence over the values in
+            *subplot_kw*.
+
+            If *mosaic* is a string, and thus all keys are single characters,
+            it is possible to use a single string instead of a tuple as keys;
+            i.e. ``"AB"`` is equivalent to ``("A", "B")``.
+
+            .. versionadded:: 3.7
 
         gridspec_kw : dict, optional
             Dictionary with keywords passed to the `.GridSpec` constructor used
@@ -1868,6 +1905,8 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
         """
         subplot_kw = subplot_kw or {}
         gridspec_kw = dict(gridspec_kw or {})
+        per_subplot_kw = per_subplot_kw or {}
+
         if height_ratios is not None:
             if 'height_ratios' in gridspec_kw:
                 raise ValueError("'height_ratios' must not be defined both as "
@@ -1882,6 +1921,12 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
         # special-case string input
         if isinstance(mosaic, str):
             mosaic = self._normalize_grid_string(mosaic)
+            per_subplot_kw = {
+                tuple(k): v for k, v in per_subplot_kw.items()
+            }
+
+        per_subplot_kw = self._norm_per_subplot_kw(per_subplot_kw)
+
         # Only accept strict bools to allow a possible future API expansion.
         _api.check_isinstance(bool, sharex=sharex, sharey=sharey)
 
@@ -2011,7 +2056,11 @@ def _do_layout(gs, mosaic, unique_ids, nested):
                         raise ValueError(f"There are duplicate keys {name} "
                                          f"in the layout\n{mosaic!r}")
                     ax = self.add_subplot(
-                        gs[slc], **{'label': str(name), **subplot_kw}
+                        gs[slc], **{
+                            'label': str(name),
+                            **subplot_kw,
+                            **per_subplot_kw.get(name, {})
+                        }
                     )
                     output[name] = ax
                 elif method == 'nested':
@@ -2048,9 +2097,11 @@ def _do_layout(gs, mosaic, unique_ids, nested):
             if sharey:
                 ax.sharey(ax0)
                 ax._label_outer_yaxis(check_patch=True)
-        for k, ax in ret.items():
-            if isinstance(k, str):
-                ax.set_label(k)
+        if extra := set(per_subplot_kw) - set(ret):
+            raise ValueError(
+                f"The keys {extra} are in *per_subplot_kw* "
+                "but not in the mosaic."
+            )
         return ret
 
     def _set_artist_props(self, a):
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -1479,7 +1479,8 @@ def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,
 
 def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
                    width_ratios=None, height_ratios=None, empty_sentinel='.',
-                   subplot_kw=None, gridspec_kw=None, **fig_kw):
+                   subplot_kw=None, gridspec_kw=None,
+                   per_subplot_kw=None, **fig_kw):
     """
     Build a layout of Axes based on ASCII art or nested lists.
 
@@ -1550,7 +1551,21 @@ def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
 
     subplot_kw : dict, optional
         Dictionary with keywords passed to the `.Figure.add_subplot` call
-        used to create each subplot.
+        used to create each subplot.  These values may be overridden by
+        values in *per_subplot_kw*.
+
+    per_subplot_kw : dict, optional
+        A dictionary mapping the Axes identifiers or tuples of identifiers
+        to a dictionary of keyword arguments to be passed to the
+        `.Figure.add_subplot` call used to create each subplot.  The values
+        in these dictionaries have precedence over the values in
+        *subplot_kw*.
+
+        If *mosaic* is a string, and thus all keys are single characters,
+        it is possible to use a single string instead of a tuple as keys;
+        i.e. ``"AB"`` is equivalent to ``("A", "B")``.
+
+        .. versionadded:: 3.7
 
     gridspec_kw : dict, optional
         Dictionary with keywords passed to the `.GridSpec` constructor used
@@ -1576,7 +1591,8 @@ def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
         mosaic, sharex=sharex, sharey=sharey,
         height_ratios=height_ratios, width_ratios=width_ratios,
         subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
-        empty_sentinel=empty_sentinel
+        empty_sentinel=empty_sentinel,
+        per_subplot_kw=per_subplot_kw,
     )
     return fig, ax_dict
 
diff --git a/tutorials/provisional/mosaic.py b/tutorials/provisional/mosaic.py
--- a/tutorials/provisional/mosaic.py
+++ b/tutorials/provisional/mosaic.py
@@ -202,8 +202,8 @@ def identify_axes(ax_dict, fontsize=48):
 # empty sentinel with the string shorthand because it may be stripped
 # while processing the input.
 #
-# Controlling mosaic and subplot creation
-# =======================================
+# Controlling mosaic creation
+# ===========================
 #
 # This feature is built on top of `.gridspec` and you can pass the
 # keyword arguments through to the underlying `.gridspec.GridSpec`
@@ -278,8 +278,12 @@ def identify_axes(ax_dict, fontsize=48):
 
 
 ###############################################################################
+# Controlling subplot creation
+# ============================
+#
 # We can also pass through arguments used to create the subplots
-# (again, the same as `.Figure.subplots`).
+# (again, the same as `.Figure.subplots`) which will apply to all
+# of the Axes created.
 
 
 axd = plt.figure(constrained_layout=True).subplot_mosaic(
@@ -287,6 +291,58 @@ def identify_axes(ax_dict, fontsize=48):
 )
 identify_axes(axd)
 
+###############################################################################
+# Per-Axes subplot keyword arguments
+# ----------------------------------
+#
+# If you need to control the parameters passed to each subplot individually use
+# *per_subplot_kw* to pass a mapping between the Axes identifiers (or
+# tuples of Axes identifiers) to dictionaries of keywords to be passed.
+#
+# .. versionadded:: 3.7
+#
+
+
+fig, axd = plt.subplot_mosaic(
+    "AB;CD",
+    per_subplot_kw={
+        "A": {"projection": "polar"},
+        ("C", "D"): {"xscale": "log"}
+    },
+)
+identify_axes(axd)
+
+###############################################################################
+# If the layout is specified with the string short-hand, then we know the
+# Axes labels will be one character and can unambiguously interpret longer
+# strings in *per_subplot_kw* to specify a set of Axes to apply the
+# keywords to:
+
+
+fig, axd = plt.subplot_mosaic(
+    "AB;CD",
+    per_subplot_kw={
+        "AD": {"projection": "polar"},
+        "BC": {"facecolor": ".9"}
+    },
+)
+identify_axes(axd)
+
+###############################################################################
+# If *subplot_kw* and *per_subplot_kw* are used together, then they are
+# merged with *per_subplot_kw* taking priority:
+
+
+axd = plt.figure(constrained_layout=True).subplot_mosaic(
+    "AB;CD",
+    subplot_kw={"facecolor": "xkcd:tangerine"},
+    per_subplot_kw={
+        "B": {"facecolor": "xkcd:water blue"},
+        "D": {"projection": "polar", "facecolor": "w"},
+    }
+)
+identify_axes(axd)
+
 
 ###############################################################################
 # Nested list input

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 1762 | 1762 | - | 8 | -
| lib/matplotlib/figure.py | 1774 | 1774 | 20 | 8 | 9847
| lib/matplotlib/figure.py | 1824 | 1824 | 20 | 8 | 9847
| lib/matplotlib/figure.py | 1846 | 1846 | 20 | 8 | 9847
| lib/matplotlib/figure.py | 1871 | 1871 | 16 | 8 | 7977
| lib/matplotlib/figure.py | 1885 | 1885 | 16 | 8 | 7977
| lib/matplotlib/figure.py | 2014 | 2014 | 13 | 8 | 6678
| lib/matplotlib/figure.py | 2051 | 2053 | 19 | 8 | 8945
| lib/matplotlib/pyplot.py | 1482 | 1482 | - | 4 | -
| lib/matplotlib/pyplot.py | 1553 | 1553 | 15 | 4 | 7715
| lib/matplotlib/pyplot.py | 1579 | 1579 | 15 | 4 | 7715
| tutorials/provisional/mosaic.py | 205 | 206 | 1 | 1 | 760
| tutorials/provisional/mosaic.py | 281 | 281 | 1 | 1 | 760
| tutorials/provisional/mosaic.py | 290 | 290 | 17 | 1 | 8284


## Problem Statement

```
[ENH]: gridspec_mosaic
### Problem

Trying to combine subplot_mosaic with axes using various different projections (e.g. one rectilinear axes and one polar axes and one 3d axes) has been requested a few times (once in the original subplot_mosaic thread IIRC, and in #20392 too), and it's something I would recently have been happy to have, too.

Pushing projections directly into subplot_mosaic seems ripe for API bloat, but perhaps another solution would be to add `figure.gridspec_mosaic(...)` which takes the same arguments as subplot_mosaic, but returns a dict of *subplotspecs*, such that one can do something like
\`\`\`
specs = fig.gridspec_mosaic(...)
d = {
    "foo": fig.add_subplot(specs["foo"], projection=...),
    "bar": fig.add_subplot(specs["bar"], projection=...),
    ...
}
\`\`\`
As a side point, I do find the repetition of `fig` in each call to add_subplot a bit jarring (as the underlying gridspec is bound to the figure, so that information is actually redundant).  Back in #13280 I had proposed to add SubplotSpec.add_subplot() (adding to the figure to which the gridspec is bound), which would allow one to write, here,
\`\`\`
specs = fig.gridspec_mosaic(...)
d = {
    "foo": specs["foo"].add_subplot(projection=...),
    "bar": specs["bar"].add_subplot(projection=...),
    ...
}
\`\`\`
but that idea got shot down back then and even if we decide not to revisit it, even the first form would be nice to have.

Thoughts?

### Proposed solution

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 tutorials/provisional/mosaic.py** | 171 | 287| 760 | 760 | 2330 | 
| 2 | 2 tutorials/intermediate/arranging_axes.py | 345 | 413| 671 | 1431 | 6280 | 
| 3 | 2 tutorials/intermediate/arranging_axes.py | 264 | 343| 760 | 2191 | 6280 | 
| 4 | 3 examples/subplots_axes_and_figures/gridspec_and_subplots.py | 1 | 31| 242 | 2433 | 6522 | 
| 5 | **3 tutorials/provisional/mosaic.py** | 1 | 37| 327 | 2760 | 6522 | 
| 6 | **4 lib/matplotlib/pyplot.py** | 1272 | 1336| 676 | 3436 | 34974 | 
| 7 | 5 examples/subplots_axes_and_figures/gridspec_multicolumn.py | 1 | 34| 259 | 3695 | 35233 | 
| 8 | 6 lib/matplotlib/gridspec.py | 556 | 599| 391 | 4086 | 41537 | 
| 9 | 7 examples/subplots_axes_and_figures/gridspec_nested.py | 1 | 46| 337 | 4423 | 41874 | 
| 10 | 7 lib/matplotlib/gridspec.py | 1 | 24| 177 | 4600 | 41874 | 
| 11 | **7 tutorials/provisional/mosaic.py** | 59 | 170| 789 | 5389 | 41874 | 
| 12 | 7 tutorials/intermediate/arranging_axes.py | 179 | 263| 908 | 6297 | 41874 | 
| **-> 13 <-** | **8 lib/matplotlib/figure.py** | 2002 | 2037| 381 | 6678 | 70700 | 
| 14 | 8 lib/matplotlib/gridspec.py | 252 | 263| 135 | 6813 | 70700 | 
| **-> 15 <-** | **8 lib/matplotlib/pyplot.py** | 1493 | 1594| 902 | 7715 | 70700 | 
| **-> 16 <-** | **8 lib/matplotlib/figure.py** | 1869 | 1886| 262 | 7977 | 70700 | 
| **-> 17 <-** | **8 tutorials/provisional/mosaic.py** | 288 | 339| 307 | 8284 | 70700 | 
| 18 | 9 examples/userdemo/demo_gridspec03.py | 1 | 52| 421 | 8705 | 71121 | 
| **-> 19 <-** | **9 lib/matplotlib/figure.py** | 2039 | 2060| 240 | 8945 | 71121 | 
| **-> 20 <-** | **9 lib/matplotlib/figure.py** | 1772 | 1868| 902 | 9847 | 71121 | 
| 21 | **9 lib/matplotlib/figure.py** | 1948 | 2001| 581 | 10428 | 71121 | 
| 22 | 9 lib/matplotlib/gridspec.py | 413 | 441| 217 | 10645 | 71121 | 
| 23 | 9 lib/matplotlib/gridspec.py | 503 | 523| 218 | 10863 | 71121 | 
| 24 | **9 lib/matplotlib/figure.py** | 615 | 715| 958 | 11821 | 71121 | 
| 25 | 9 lib/matplotlib/gridspec.py | 385 | 411| 248 | 12069 | 71121 | 
| 26 | 9 lib/matplotlib/gridspec.py | 265 | 316| 500 | 12569 | 71121 | 
| 27 | 9 lib/matplotlib/gridspec.py | 207 | 226| 205 | 12774 | 71121 | 
| 28 | **9 lib/matplotlib/figure.py** | 1471 | 1512| 262 | 13036 | 71121 | 
| 29 | 10 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 13792 | 72544 | 
| 30 | 10 tutorials/intermediate/arranging_axes.py | 100 | 178| 776 | 14568 | 72544 | 
| 31 | 11 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 442 | 15010 | 74512 | 
| 32 | 12 lib/matplotlib/_constrained_layout.py | 197 | 240| 435 | 15445 | 81878 | 
| 33 | 12 lib/matplotlib/gridspec.py | 693 | 734| 260 | 15705 | 81878 | 
| 34 | 12 lib/matplotlib/gridspec.py | 228 | 250| 216 | 15921 | 81878 | 
| 35 | 12 lib/matplotlib/gridspec.py | 601 | 651| 442 | 16363 | 81878 | 
| 36 | 12 lib/matplotlib/gridspec.py | 58 | 68| 141 | 16504 | 81878 | 
| 37 | 13 examples/userdemo/demo_gridspec06.py | 1 | 39| 325 | 16829 | 82203 | 
| 38 | 13 lib/matplotlib/gridspec.py | 319 | 383| 598 | 17427 | 82203 | 
| 39 | 14 examples/userdemo/demo_gridspec01.py | 1 | 30| 240 | 17667 | 82443 | 
| 40 | **14 lib/matplotlib/figure.py** | 1647 | 1689| 350 | 18017 | 82443 | 
| 41 | 14 lib/matplotlib/gridspec.py | 471 | 501| 277 | 18294 | 82443 | 
| 42 | 15 tutorials/intermediate/constrainedlayout_guide.py | 354 | 441| 772 | 19066 | 89009 | 
| 43 | 15 tutorials/intermediate/constrainedlayout_guide.py | 270 | 352| 804 | 19870 | 89009 | 
| 44 | **15 lib/matplotlib/figure.py** | 716 | 756| 454 | 20324 | 89009 | 
| 45 | 15 lib/matplotlib/gridspec.py | 526 | 554| 250 | 20574 | 89009 | 
| 46 | 16 lib/matplotlib/projections/__init__.py | 1 | 58| 563 | 21137 | 89865 | 
| 47 | **16 lib/matplotlib/pyplot.py** | 1136 | 1271| 1196 | 22333 | 89865 | 
| 48 | **16 lib/matplotlib/figure.py** | 1888 | 1918| 320 | 22653 | 89865 | 
| 49 | 16 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 23438 | 89865 | 
| 50 | 16 tutorials/intermediate/arranging_axes.py | 1 | 99| 834 | 24272 | 89865 | 
| 51 | 17 lib/matplotlib/_tight_layout.py | 160 | 191| 242 | 24514 | 92794 | 
| 52 | 18 examples/lines_bars_and_markers/scatter_hist.py | 57 | 123| 702 | 25216 | 93905 | 
| 53 | 18 lib/matplotlib/gridspec.py | 70 | 99| 242 | 25458 | 93905 | 
| 54 | **18 lib/matplotlib/figure.py** | 1920 | 1946| 242 | 25700 | 93905 | 
| 55 | **18 lib/matplotlib/figure.py** | 758 | 876| 1131 | 26831 | 93905 | 
| 56 | 19 lib/matplotlib/axes/_base.py | 550 | 1261| 6189 | 33020 | 133595 | 
| 57 | 19 examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 741 | 33761 | 133595 | 
| 58 | 19 lib/matplotlib/gridspec.py | 27 | 56| 311 | 34072 | 133595 | 
| 59 | 20 ci/check_wheel_licenses.py | 1 | 37| 213 | 34285 | 133852 | 
| 60 | 20 tutorials/intermediate/constrainedlayout_guide.py | 442 | 530| 776 | 35061 | 133852 | 
| 61 | 20 tutorials/intermediate/constrainedlayout_guide.py | 636 | 721| 777 | 35838 | 133852 | 
| 62 | 21 examples/scales/semilogx_demo.py | 1 | 24| 104 | 35942 | 133956 | 
| 63 | **21 lib/matplotlib/figure.py** | 877 | 892| 245 | 36187 | 133956 | 
| 64 | 22 lib/matplotlib/projections/geo.py | 110 | 191| 807 | 36994 | 138122 | 
| 65 | 22 lib/matplotlib/_constrained_layout.py | 1 | 59| 602 | 37596 | 138122 | 
| 66 | 23 examples/misc/custom_projection.py | 100 | 170| 799 | 38395 | 141945 | 
| 67 | 24 lib/mpl_toolkits/axes_grid1/axes_grid.py | 318 | 417| 990 | 39385 | 146994 | 
| 68 | 24 lib/matplotlib/gridspec.py | 653 | 667| 166 | 39551 | 146994 | 
| 69 | 24 lib/matplotlib/gridspec.py | 669 | 691| 207 | 39758 | 146994 | 
| 70 | 25 tutorials/toolkits/mplot3d.py | 1 | 157| 1133 | 40891 | 148127 | 
| 71 | 26 examples/color/named_colors.py | 1 | 22| 118 | 41009 | 148947 | 
| 72 | 27 tutorials/toolkits/axes_grid.py | 1 | 332| 3454 | 44463 | 152401 | 
| 73 | 27 examples/misc/custom_projection.py | 172 | 258| 778 | 45241 | 152401 | 
| 74 | 28 lib/matplotlib/colorbar.py | 1534 | 1595| 624 | 45865 | 166702 | 
| 75 | 29 examples/images_contours_and_fields/pcolormesh_grids.py | 1 | 80| 799 | 46664 | 167985 | 
| 76 | **29 lib/matplotlib/figure.py** | 1568 | 1593| 148 | 46812 | 167985 | 
| 77 | 30 lib/matplotlib/_layoutgrid.py | 31 | 115| 925 | 47737 | 173421 | 


### Hint

```
I like this better than the current  create/remove/replace scheme, but just to be clear- using this method means folks would have to go in manually and create a subplot for each spec, right? So this feature is just providing the ability to layout and identify axes in the same way as subplot_mosaic? 
We could do this, but it seems relatively convoluted, and likely to end up in a dusty corner...  Isn't the fundamental problem that we can't change projections post-facto?  Can we not do that somehow?  `ax.change_projection()` would be best, but `axnew = fig.change_projection(ax, proj)` would perhaps be bearable.  

> and likely to end up in a dusty corner

I think that the loops for adding a new subplot would probably look almost identical to the ones for changing the projection, so I'm not sure that the discoverablity/documentation problem favors one over the other, especially if either solution is added to the mosaic tutorial and as a nice gallery example. 
The projection machinery can pick a different class to use for the returned `Axes` instance.  While you can change classes in Python (which we do in mplot3d), I am not a huge fan of doing more of that to I think changing the projection is off the table.

----

Unfortunately we already pass `subplot_kw` to all of them and do not want to try to de-conflict the namespaces.  Making something like:

\`\`\`python
from matplotlib import pyplot as plt

fig, axd = plt.subplot_mosaic(
    "AB;CC",
    needs_a_better_name={"A": {"projection": "polar"}, "B": {"projection": "3d"}},
)

\`\`\`

work is not that bad (I used a bad name to avoid getting lost in discussions of the parameter name just yet ;) ).

![so](https://user-images.githubusercontent.com/199813/204940851-cb3b7304-d18f-4a7b-9657-88eaee19ec58.png)


\`\`\`diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
index 6c18ba1a64..e75b973044 100644
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1771,7 +1771,8 @@ default: %(va)s

     def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                        width_ratios=None, height_ratios=None,
-                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
+                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None,
+                       needs_a_better_name=None):
         """
         Build a layout of Axes based on ASCII art or nested lists.

@@ -1868,6 +1869,7 @@ default: %(va)s
         """
         subplot_kw = subplot_kw or {}
         gridspec_kw = dict(gridspec_kw or {})
+        needs_a_better_name = needs_a_better_name or {}
         if height_ratios is not None:
             if 'height_ratios' in gridspec_kw:
                 raise ValueError("'height_ratios' must not be defined both as "
@@ -2011,7 +2013,11 @@ default: %(va)s
                         raise ValueError(f"There are duplicate keys {name} "
                                          f"in the layout\n{mosaic!r}")
                     ax = self.add_subplot(
-                        gs[slc], **{'label': str(name), **subplot_kw}
+                        gs[slc], **{
+                            'label': str(name),
+                            **subplot_kw,
+                            **needs_a_better_name.get(name, {})
+                        }
                     )
                     output[name] = ax
                 elif method == 'nested':
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
index 79c33a6bac..f384fe747b 100644
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -1474,7 +1474,8 @@ def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,

 def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
                    width_ratios=None, height_ratios=None, empty_sentinel='.',
-                   subplot_kw=None, gridspec_kw=None, **fig_kw):
+                   subplot_kw=None, gridspec_kw=None,
+                   needs_a_better_name=None, **fig_kw):
     """
     Build a layout of Axes based on ASCII art or nested lists.

@@ -1571,7 +1572,8 @@ def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
         mosaic, sharex=sharex, sharey=sharey,
         height_ratios=height_ratios, width_ratios=width_ratios,
         subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
-        empty_sentinel=empty_sentinel
+        empty_sentinel=empty_sentinel,
+        needs_a_better_name=needs_a_better_name,
     )
     return fig, ax_dict
\`\`\`
Just to chime in and say that I have also had the desire to have a dropdown selector of geographic projections to click through/change at will. (A nice example here: https://observablehq.com/@d3/projection-transitions) If someone calls `ax.change_projection()` I think we could get away with some documentation that the class type may change underneath them..., and do some fiddling on our end to update the transforms and `__class__` attributes properly (which is probably not trivial to get right).
Ah, I see that #20392 proposed to pass subplot_kws as an array of dicts (which is a bit of a mess once you have axes spanning multiple cells), and I can't find if a concrete proposal had been made in the original thread (#16603); but I agree that passing instead a dict of dicts (label -> subplot_kw) as proposed by @tacaswell seems not too bad.
My first follow-up feature request would be to do

\`\`\`python
from matplotlib import pyplot as plt

fig, axd = plt.subplot_mosaic(
    "AB;CC",
    needs_a_better_name={"AB": {"projection": "polar"}},
)
\`\`\`
to make both the top panels polar.  Sometimes you might want like 10 axes with 5 in one projection and 5 in another.
> \`\`\`
> needs_a_better_name={"AB": {"projection": "polar"}},
> \`\`\`

I think the generic solution is
\`\`\`
needs_a_better_name={
    ("name1", "name2"): {"projection": "polar"}},
    "name3": {"projection": "3d"},
}
\`\`\`
i.e. keys are strings or tuples of strings.

Maybe with the extension: If the mosaic spec is a string (i.e. we only have single-char names), any key with len>1 is interpreted as `tuple(key)`, i.e. "AB" is internally converted to ("A", "B").
> we already pass subplot_kw to all of them and do not want to try to de-conflict the namespaces

So which one would get priority if like `subplot_kw` and `needs_a_better_name` get passed the same key?
In Tom's diff, `needs_a_better_name` would get priority as it is later in the unpacking.

\`\`\`python
>>> {"a": 1, "a":2}
{'a': 2}
\`\`\`

This makes intuitive sense as it allows you to set some generic settings for most subplots but override with the more specific kwargs for individual subplot, which you simply would not do if that had no effect.
> This makes intuitive sense 

Yeah, it's just gonna have to be documented in neon lights and commented as this is a for free of python dict implementation. I'm worried though about two keywords that do almost the same thing but one is vectorized, since that feels like `color` vs `colors`
Implementing @rcomer 's idea for the short hand was easy, but I have more concerns about the general case as we will technically work with _any_ hashables as the keys so there is an ambiguity there.  Not insurmountable, but annoying.

Pulling out a (maybe private) `gridspec_mosaic` may still be useful so we can implement `subfigure_mosaic` as well....
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1759,6 +1759,25 @@ def get_tightbbox(self, renderer=None, bbox_extra_artists=None):
 
         return _bbox
 
+    @staticmethod
+    def _norm_per_subplot_kw(per_subplot_kw):
+        expanded = {}
+        for k, v in per_subplot_kw.items():
+            if isinstance(k, tuple):
+                for sub_key in k:
+                    if sub_key in expanded:
+                        raise ValueError(
+                            f'The key {sub_key!r} appears multiple times.'
+                            )
+                    expanded[sub_key] = v
+            else:
+                if k in expanded:
+                    raise ValueError(
+                        f'The key {k!r} appears multiple times.'
+                    )
+                expanded[k] = v
+        return expanded
+
     @staticmethod
     def _normalize_grid_string(layout):
         if '\n' not in layout:
@@ -1771,7 +1790,8 @@ def _normalize_grid_string(layout):
 
     def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                        width_ratios=None, height_ratios=None,
-                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
+                       empty_sentinel='.',
+                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
         """
         Build a layout of Axes based on ASCII art or nested lists.
 
@@ -1821,6 +1841,9 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
             The string notation allows only single character Axes labels and
             does not support nesting but is very terse.
 
+            The Axes identifiers may be `str` or a non-iterable hashable
+            object (e.g. `tuple` s may not be used).
+
         sharex, sharey : bool, default: False
             If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
             among all subplots.  In that case, tick label visibility and axis
@@ -1843,7 +1866,21 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
 
         subplot_kw : dict, optional
             Dictionary with keywords passed to the `.Figure.add_subplot` call
-            used to create each subplot.
+            used to create each subplot.  These values may be overridden by
+            values in *per_subplot_kw*.
+
+        per_subplot_kw : dict, optional
+            A dictionary mapping the Axes identifiers or tuples of identifiers
+            to a dictionary of keyword arguments to be passed to the
+            `.Figure.add_subplot` call used to create each subplot.  The values
+            in these dictionaries have precedence over the values in
+            *subplot_kw*.
+
+            If *mosaic* is a string, and thus all keys are single characters,
+            it is possible to use a single string instead of a tuple as keys;
+            i.e. ``"AB"`` is equivalent to ``("A", "B")``.
+
+            .. versionadded:: 3.7
 
         gridspec_kw : dict, optional
             Dictionary with keywords passed to the `.GridSpec` constructor used
@@ -1868,6 +1905,8 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
         """
         subplot_kw = subplot_kw or {}
         gridspec_kw = dict(gridspec_kw or {})
+        per_subplot_kw = per_subplot_kw or {}
+
         if height_ratios is not None:
             if 'height_ratios' in gridspec_kw:
                 raise ValueError("'height_ratios' must not be defined both as "
@@ -1882,6 +1921,12 @@ def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
         # special-case string input
         if isinstance(mosaic, str):
             mosaic = self._normalize_grid_string(mosaic)
+            per_subplot_kw = {
+                tuple(k): v for k, v in per_subplot_kw.items()
+            }
+
+        per_subplot_kw = self._norm_per_subplot_kw(per_subplot_kw)
+
         # Only accept strict bools to allow a possible future API expansion.
         _api.check_isinstance(bool, sharex=sharex, sharey=sharey)
 
@@ -2011,7 +2056,11 @@ def _do_layout(gs, mosaic, unique_ids, nested):
                         raise ValueError(f"There are duplicate keys {name} "
                                          f"in the layout\n{mosaic!r}")
                     ax = self.add_subplot(
-                        gs[slc], **{'label': str(name), **subplot_kw}
+                        gs[slc], **{
+                            'label': str(name),
+                            **subplot_kw,
+                            **per_subplot_kw.get(name, {})
+                        }
                     )
                     output[name] = ax
                 elif method == 'nested':
@@ -2048,9 +2097,11 @@ def _do_layout(gs, mosaic, unique_ids, nested):
             if sharey:
                 ax.sharey(ax0)
                 ax._label_outer_yaxis(check_patch=True)
-        for k, ax in ret.items():
-            if isinstance(k, str):
-                ax.set_label(k)
+        if extra := set(per_subplot_kw) - set(ret):
+            raise ValueError(
+                f"The keys {extra} are in *per_subplot_kw* "
+                "but not in the mosaic."
+            )
         return ret
 
     def _set_artist_props(self, a):
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -1479,7 +1479,8 @@ def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,
 
 def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
                    width_ratios=None, height_ratios=None, empty_sentinel='.',
-                   subplot_kw=None, gridspec_kw=None, **fig_kw):
+                   subplot_kw=None, gridspec_kw=None,
+                   per_subplot_kw=None, **fig_kw):
     """
     Build a layout of Axes based on ASCII art or nested lists.
 
@@ -1550,7 +1551,21 @@ def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
 
     subplot_kw : dict, optional
         Dictionary with keywords passed to the `.Figure.add_subplot` call
-        used to create each subplot.
+        used to create each subplot.  These values may be overridden by
+        values in *per_subplot_kw*.
+
+    per_subplot_kw : dict, optional
+        A dictionary mapping the Axes identifiers or tuples of identifiers
+        to a dictionary of keyword arguments to be passed to the
+        `.Figure.add_subplot` call used to create each subplot.  The values
+        in these dictionaries have precedence over the values in
+        *subplot_kw*.
+
+        If *mosaic* is a string, and thus all keys are single characters,
+        it is possible to use a single string instead of a tuple as keys;
+        i.e. ``"AB"`` is equivalent to ``("A", "B")``.
+
+        .. versionadded:: 3.7
 
     gridspec_kw : dict, optional
         Dictionary with keywords passed to the `.GridSpec` constructor used
@@ -1576,7 +1591,8 @@ def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
         mosaic, sharex=sharex, sharey=sharey,
         height_ratios=height_ratios, width_ratios=width_ratios,
         subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
-        empty_sentinel=empty_sentinel
+        empty_sentinel=empty_sentinel,
+        per_subplot_kw=per_subplot_kw,
     )
     return fig, ax_dict
 
diff --git a/tutorials/provisional/mosaic.py b/tutorials/provisional/mosaic.py
--- a/tutorials/provisional/mosaic.py
+++ b/tutorials/provisional/mosaic.py
@@ -202,8 +202,8 @@ def identify_axes(ax_dict, fontsize=48):
 # empty sentinel with the string shorthand because it may be stripped
 # while processing the input.
 #
-# Controlling mosaic and subplot creation
-# =======================================
+# Controlling mosaic creation
+# ===========================
 #
 # This feature is built on top of `.gridspec` and you can pass the
 # keyword arguments through to the underlying `.gridspec.GridSpec`
@@ -278,8 +278,12 @@ def identify_axes(ax_dict, fontsize=48):
 
 
 ###############################################################################
+# Controlling subplot creation
+# ============================
+#
 # We can also pass through arguments used to create the subplots
-# (again, the same as `.Figure.subplots`).
+# (again, the same as `.Figure.subplots`) which will apply to all
+# of the Axes created.
 
 
 axd = plt.figure(constrained_layout=True).subplot_mosaic(
@@ -287,6 +291,58 @@ def identify_axes(ax_dict, fontsize=48):
 )
 identify_axes(axd)
 
+###############################################################################
+# Per-Axes subplot keyword arguments
+# ----------------------------------
+#
+# If you need to control the parameters passed to each subplot individually use
+# *per_subplot_kw* to pass a mapping between the Axes identifiers (or
+# tuples of Axes identifiers) to dictionaries of keywords to be passed.
+#
+# .. versionadded:: 3.7
+#
+
+
+fig, axd = plt.subplot_mosaic(
+    "AB;CD",
+    per_subplot_kw={
+        "A": {"projection": "polar"},
+        ("C", "D"): {"xscale": "log"}
+    },
+)
+identify_axes(axd)
+
+###############################################################################
+# If the layout is specified with the string short-hand, then we know the
+# Axes labels will be one character and can unambiguously interpret longer
+# strings in *per_subplot_kw* to specify a set of Axes to apply the
+# keywords to:
+
+
+fig, axd = plt.subplot_mosaic(
+    "AB;CD",
+    per_subplot_kw={
+        "AD": {"projection": "polar"},
+        "BC": {"facecolor": ".9"}
+    },
+)
+identify_axes(axd)
+
+###############################################################################
+# If *subplot_kw* and *per_subplot_kw* are used together, then they are
+# merged with *per_subplot_kw* taking priority:
+
+
+axd = plt.figure(constrained_layout=True).subplot_mosaic(
+    "AB;CD",
+    subplot_kw={"facecolor": "xkcd:tangerine"},
+    per_subplot_kw={
+        "B": {"facecolor": "xkcd:water blue"},
+        "D": {"projection": "polar", "facecolor": "w"},
+    }
+)
+identify_axes(axd)
+
 
 ###############################################################################
 # Nested list input

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_figure.py b/lib/matplotlib/tests/test_figure.py
--- a/lib/matplotlib/tests/test_figure.py
+++ b/lib/matplotlib/tests/test_figure.py
@@ -848,7 +848,12 @@ def test_animated_with_canvas_change(fig_test, fig_ref):
 class TestSubplotMosaic:
     @check_figures_equal(extensions=["png"])
     @pytest.mark.parametrize(
-        "x", [[["A", "A", "B"], ["C", "D", "B"]], [[1, 1, 2], [3, 4, 2]]]
+        "x", [
+            [["A", "A", "B"], ["C", "D", "B"]],
+            [[1, 1, 2], [3, 4, 2]],
+            (("A", "A", "B"), ("C", "D", "B")),
+            ((1, 1, 2), (3, 4, 2))
+        ]
     )
     def test_basic(self, fig_test, fig_ref, x):
         grid_axes = fig_test.subplot_mosaic(x)
@@ -998,6 +1003,10 @@ def test_fail_list_of_str(self):
             plt.subplot_mosaic(['foo', 'bar'])
         with pytest.raises(ValueError, match='must be 2D'):
             plt.subplot_mosaic(['foo'])
+        with pytest.raises(ValueError, match='must be 2D'):
+            plt.subplot_mosaic([['foo', ('bar',)]])
+        with pytest.raises(ValueError, match='must be 2D'):
+            plt.subplot_mosaic([['a', 'b'], [('a', 'b'), 'c']])
 
     @check_figures_equal(extensions=["png"])
     @pytest.mark.parametrize("subplot_kw", [{}, {"projection": "polar"}, None])
@@ -1011,8 +1020,26 @@ def test_subplot_kw(self, fig_test, fig_ref, subplot_kw):
 
         axB = fig_ref.add_subplot(gs[0, 1], **subplot_kw)
 
+    @check_figures_equal(extensions=["png"])
+    @pytest.mark.parametrize("multi_value", ['BC', tuple('BC')])
+    def test_per_subplot_kw(self, fig_test, fig_ref, multi_value):
+        x = 'AB;CD'
+        grid_axes = fig_test.subplot_mosaic(
+            x,
+            subplot_kw={'facecolor': 'red'},
+            per_subplot_kw={
+                'D': {'facecolor': 'blue'},
+                multi_value: {'facecolor': 'green'},
+            }
+        )
+
+        gs = fig_ref.add_gridspec(2, 2)
+        for color, spec in zip(['red', 'green', 'green', 'blue'], gs):
+            fig_ref.add_subplot(spec, facecolor=color)
+
     def test_string_parser(self):
         normalize = Figure._normalize_grid_string
+
         assert normalize('ABC') == [['A', 'B', 'C']]
         assert normalize('AB;CC') == [['A', 'B'], ['C', 'C']]
         assert normalize('AB;CC;DE') == [['A', 'B'], ['C', 'C'], ['D', 'E']]
@@ -1029,6 +1056,25 @@ def test_string_parser(self):
                          DE
                          """) == [['A', 'B'], ['C', 'C'], ['D', 'E']]
 
+    def test_per_subplot_kw_expander(self):
+        normalize = Figure._norm_per_subplot_kw
+        assert normalize({"A": {}, "B": {}}) == {"A": {}, "B": {}}
+        assert normalize({("A", "B"): {}}) == {"A": {}, "B": {}}
+        with pytest.raises(
+                ValueError, match=f'The key {"B"!r} appears multiple times'
+        ):
+            normalize({("A", "B"): {}, "B": {}})
+        with pytest.raises(
+                ValueError, match=f'The key {"B"!r} appears multiple times'
+        ):
+            normalize({"B": {}, ("A", "B"): {}})
+
+    def test_extra_per_subplot_kw(self):
+        with pytest.raises(
+                ValueError, match=f'The keys {set("B")!r} are in'
+        ):
+            Figure().subplot_mosaic("A", per_subplot_kw={"B": {}})
+
     @check_figures_equal(extensions=["png"])
     @pytest.mark.parametrize("str_pattern",
                              ["AAA\nBBB", "\nAAA\nBBB\n", "ABC\nDEF"]

```


## Code snippets

### 1 - tutorials/provisional/mosaic.py:

Start line: 171, End line: 287

```python
identify_axes(axd)


###############################################################################
# If we prefer to use another character (rather than a period ``"."``)
# to mark the empty space, we can use *empty_sentinel* to specify the
# character to use.

axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    aX
    Xb
    """,
    empty_sentinel="X",
)
identify_axes(axd)


###############################################################################
#
# Internally there is no meaning attached to the letters we use, any
# Unicode code point is valid!

axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """αб
       ℝ☢"""
)
identify_axes(axd)

###############################################################################
# It is not recommended to use white space as either a label or an
# empty sentinel with the string shorthand because it may be stripped
# while processing the input.
#
# Controlling mosaic and subplot creation
# =======================================
#
# This feature is built on top of `.gridspec` and you can pass the
# keyword arguments through to the underlying `.gridspec.GridSpec`
# (the same as `.Figure.subplots`).
#
# In this case we want to use the input to specify the arrangement,
# but set the relative widths of the rows / columns.  For convenience,
# `.gridspec.GridSpec`'s *height_ratios* and *width_ratios* are exposed in the
# `.Figure.subplot_mosaic` calling sequence.


axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    .a.
    bAc
    .d.
    """,
    # set the height ratios between the rows
    height_ratios=[1, 3.5, 1],
    # set the width ratios between the columns
    width_ratios=[1, 3.5, 1],
)
identify_axes(axd)

###############################################################################
# Other `.gridspec.GridSpec` keywords can be passed via *gridspec_kw*.  For
# example, use the {*left*, *right*, *bottom*, *top*} keyword arguments to
# position the overall mosaic to put multiple versions of the same
# mosaic in a figure.

mosaic = """AA
            BC"""
fig = plt.figure()
axd = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "bottom": 0.25,
        "top": 0.95,
        "left": 0.1,
        "right": 0.5,
        "wspace": 0.5,
        "hspace": 0.5,
    },
)
identify_axes(axd)

axd = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "bottom": 0.05,
        "top": 0.75,
        "left": 0.6,
        "right": 0.95,
        "wspace": 0.5,
        "hspace": 0.5,
    },
)
identify_axes(axd)

###############################################################################
# Alternatively, you can use the sub-Figure functionality:

mosaic = """AA
            BC"""
fig = plt.figure(constrained_layout=True)
left, right = fig.subfigures(nrows=1, ncols=2)
axd = left.subplot_mosaic(mosaic)
identify_axes(axd)

axd = right.subplot_mosaic(mosaic)
identify_axes(axd)


###############################################################################
# We can also pass through arguments used to create the subplots
# (again, the same as `.Figure.subplots`).


axd = plt.figure(constrained_layout=True).subplot_mosaic(
    "AB", subplot_kw={"projection": "polar"}
)
```
### 2 - tutorials/intermediate/arranging_axes.py:

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
### 3 - tutorials/intermediate/arranging_axes.py:

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
### 4 - examples/subplots_axes_and_figures/gridspec_and_subplots.py:

Start line: 1, End line: 31

```python
"""
==================================================
Combining two subplots using subplots and GridSpec
==================================================

Sometimes we want to combine two subplots in an axes layout created with
`~.Figure.subplots`.  We can get the `~.gridspec.GridSpec` from the axes
and then remove the covered axes and fill the gap with a new bigger axes.
Here we create a layout with the bottom two axes in the last column combined.

To start with this layout (rather than removing the overlapping axes) use
`~.pyplot.subplot_mosaic`.

See also :doc:`/tutorials/intermediate/arranging_axes`.
"""

import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=3, nrows=3)
gs = axs[1, 2].get_gridspec()
# remove the underlying axes
for ax in axs[1:, -1]:
    ax.remove()
axbig = fig.add_subplot(gs[1:, -1])
axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
               xycoords='axes fraction', va='center')

fig.tight_layout()

plt.show()
```
### 5 - tutorials/provisional/mosaic.py:

Start line: 1, End line: 37

```python
"""
=======================================
Complex and semantic figure composition
=======================================

.. warning::

   This tutorial documents experimental / provisional API.
   We are releasing this in v3.3 to get user feedback.  We may
   make breaking changes in future versions with no warning.


Laying out Axes in a Figure in a non-uniform grid can be both tedious
and verbose.  For dense, even grids we have `.Figure.subplots` but for
more complex layouts, such as Axes that span multiple columns / rows
of the layout or leave some areas of the Figure blank, you can use
`.gridspec.GridSpec` (see :doc:`/tutorials/intermediate/arranging_axes`) or
manually place your axes.  `.Figure.subplot_mosaic` aims to provide an
interface to visually lay out your axes (as either ASCII art or nested
lists) to streamline this process.

This interface naturally supports naming your axes.
`.Figure.subplot_mosaic` returns a dictionary keyed on the
labels used to lay out the Figure.  By returning data structures with
names, it is easier to write plotting code that is independent of the
Figure layout.


This is inspired by a `proposed MEP
<https://github.com/matplotlib/matplotlib/pull/4384>`__ and the
`patchwork <https://github.com/thomasp85/patchwork>`__ library for R.
While we do not implement the operator overloading style, we do
provide a Pythonic API for specifying (nested) Axes layouts.

"""
import matplotlib.pyplot as plt
import numpy as np
```
### 6 - lib/matplotlib/pyplot.py:

Start line: 1272, End line: 1336

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
### 7 - examples/subplots_axes_and_figures/gridspec_multicolumn.py:

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
### 8 - lib/matplotlib/gridspec.py:

Start line: 556, End line: 599

```python
class SubplotSpec:

    @staticmethod
    def _from_subplot_args(figure, args):
        """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
        if len(args) == 1:
            arg, = args
            if isinstance(arg, SubplotSpec):
                return arg
            elif not isinstance(arg, Integral):
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}")
            try:
                rows, cols, num = map(int, str(arg))
            except ValueError:
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}") from None
        elif len(args) == 3:
            rows, cols, num = args
        else:
            raise _api.nargs_error("subplot", takes="1 or 3", given=len(args))

        gs = GridSpec._check_gridspec_exists(figure, rows, cols)
        if gs is None:
            gs = GridSpec(rows, cols, figure=figure)
        if isinstance(num, tuple) and len(num) == 2:
            if not all(isinstance(n, Integral) for n in num):
                raise ValueError(
                    f"Subplot specifier tuple must contain integers, not {num}"
                )
            i, j = num
        else:
            if not isinstance(num, Integral) or num < 1 or num > rows*cols:
                raise ValueError(
                    f"num must be an integer with 1 <= num <= {rows*cols}, "
                    f"not {num!r}"
                )
            i = j = num
        return gs[i-1:j]
```
### 9 - examples/subplots_axes_and_figures/gridspec_nested.py:

Start line: 1, End line: 46

```python
"""
================
Nested Gridspecs
================

GridSpecs can be nested, so that a subplot from a parent GridSpec can
set the position for a nested grid of subplots.

Note that the same functionality can be achieved more directly with
`~.FigureBase.subfigures`; see
:doc:`/gallery/subplots_axes_and_figures/subfigures`.

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
fig = plt.figure()

gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

ax1 = fig.add_subplot(gs00[:-1, :])
ax2 = fig.add_subplot(gs00[-1, :-1])
ax3 = fig.add_subplot(gs00[-1, -1])

# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs01 = gs0[1].subgridspec(3, 3)

ax4 = fig.add_subplot(gs01[:, :-1])
ax5 = fig.add_subplot(gs01[:-1, -1])
ax6 = fig.add_subplot(gs01[-1, -1])

plt.suptitle("GridSpec Inside GridSpec")
format_axes(fig)

plt.show()
```
### 10 - lib/matplotlib/gridspec.py:

Start line: 1, End line: 24

```python
r"""
:mod:`~matplotlib.gridspec` contains classes that help to layout multiple
`~.axes.Axes` in a grid-like pattern within a figure.

The `GridSpec` specifies the overall grid structure. Individual cells within
the grid are referenced by `SubplotSpec`\s.

Often, users need not access this module directly, and can use higher-level
methods like `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` and
`~.Figure.subfigures`. See the tutorial
:doc:`/tutorials/intermediate/arranging_axes` for a guide.
"""

import copy
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox

_log = logging.getLogger(__name__)
```
### 11 - tutorials/provisional/mosaic.py:

Start line: 59, End line: 170

```python
###############################################################################
# If we want a 2x2 grid we can use `.Figure.subplots` which returns a 2D array
# of `.axes.Axes` which we can index into to do our plotting.
np.random.seed(19680801)
hist_data = np.random.randn(1_500)


fig = plt.figure(constrained_layout=True)
ax_array = fig.subplots(2, 2, squeeze=False)

ax_array[0, 0].bar(["a", "b", "c"], [5, 7, 9])
ax_array[0, 1].plot([1, 2, 3])
ax_array[1, 0].hist(hist_data, bins="auto")
ax_array[1, 1].imshow([[1, 2], [2, 1]])

identify_axes(
    {(j, k): a for j, r in enumerate(ax_array) for k, a in enumerate(r)},
)

###############################################################################
# Using `.Figure.subplot_mosaic` we can produce the same mosaic but give the
# axes semantic names

fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(
    [
        ["bar", "plot"],
        ["hist", "image"],
    ],
)
ax_dict["bar"].bar(["a", "b", "c"], [5, 7, 9])
ax_dict["plot"].plot([1, 2, 3])
ax_dict["hist"].hist(hist_data)
ax_dict["image"].imshow([[1, 2], [2, 1]])
identify_axes(ax_dict)

###############################################################################
# A key difference between `.Figure.subplots` and
# `.Figure.subplot_mosaic` is the return value. While the former
# returns an array for index access, the latter returns a dictionary
# mapping the labels to the `.axes.Axes` instances created

print(ax_dict)


###############################################################################
# String short-hand
# =================
#
# By restricting our axes labels to single characters we can
# "draw" the Axes we want as "ASCII art".  The following


mosaic = """
    AB
    CD
    """

###############################################################################
# will give us 4 Axes laid out in a 2x2 grid and generates the same
# figure mosaic as above (but now labeled with ``{"A", "B", "C",
# "D"}`` rather than ``{"bar", "plot", "hist", "image"}``).

fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)

###############################################################################
# Alternatively, you can use the more compact string notation
mosaic = "AB;CD"

###############################################################################
# will give you the same composition, where the ``";"`` is used
# as the row separator instead of newline.

fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)

###############################################################################
# Axes spanning multiple rows/columns
# ===================================
#
# Something we can do with `.Figure.subplot_mosaic`, that we cannot
# do with `.Figure.subplots`, is to specify that an Axes should span
# several rows or columns.


###############################################################################
# If we want to re-arrange our four Axes to have ``"C"`` be a horizontal
# span on the bottom and ``"D"`` be a vertical span on the right we would do

axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    ABD
    CCD
    """
)
identify_axes(axd)

###############################################################################
# If we do not want to fill in all the spaces in the Figure with Axes,
# we can specify some spaces in the grid to be blank


axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    A.C
    BBB
    .D.
    """
)
```
### 13 - lib/matplotlib/figure.py:

Start line: 2002, End line: 2037

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):

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
### 15 - lib/matplotlib/pyplot.py:

Start line: 1493, End line: 1594

```python
def subplot_mosaic(mosaic, *, sharex=False, sharey=False,
                   width_ratios=None, height_ratios=None, empty_sentinel='.',
                   subplot_kw=None, gridspec_kw=None, **fig_kw):
    """
    Build a layout of Axes based on ASCII art or nested lists.

    This is a helper function to build complex GridSpec layouts visually.

    .. note::

       This API is provisional and may be revised in the future based on
       early user feedback.

    See :doc:`/tutorials/provisional/mosaic`
    for an example and full API documentation

    Parameters
    ----------
    mosaic : list of list of {hashable or nested} or str

        A visual layout of how you want your Axes to be arranged
        labeled as strings.  For example ::

           x = [['A panel', 'A panel', 'edge'],
                ['C panel', '.',       'edge']]

        produces 4 axes:

        - 'A panel' which is 1 row high and spans the first two columns
        - 'edge' which is 2 rows high and is on the right edge
        - 'C panel' which in 1 row and 1 column wide in the bottom left
        - a blank space 1 row and 1 column wide in the bottom center

        Any of the entries in the layout can be a list of lists
        of the same form to create nested layouts.

        If input is a str, then it must be of the form ::

          '''
          AAE
          C.E
          '''

        where each character is a column and each line is a row.
        This only allows only single character Axes labels and does
        not allow nesting but is very terse.

    sharex, sharey : bool, default: False
        If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
        among all subplots.  In that case, tick label visibility and axis units
        behave as for `subplots`.  If False, each subplot's x- or y-axis will
        be independent.

    width_ratios : array-like of length *ncols*, optional
        Defines the relative widths of the columns. Each column gets a
        relative width of ``width_ratios[i] / sum(width_ratios)``.
        If not given, all columns will have the same width.  Convenience
        for ``gridspec_kw={'width_ratios': [...]}``.

    height_ratios : array-like of length *nrows*, optional
        Defines the relative heights of the rows. Each row gets a
        relative height of ``height_ratios[i] / sum(height_ratios)``.
        If not given, all rows will have the same height. Convenience
        for ``gridspec_kw={'height_ratios': [...]}``.

    empty_sentinel : object, optional
        Entry in the layout to mean "leave this space empty".  Defaults
        to ``'.'``. Note, if *layout* is a string, it is processed via
        `inspect.cleandoc` to remove leading white space, which may
        interfere with using white-space as the empty sentinel.

    subplot_kw : dict, optional
        Dictionary with keywords passed to the `.Figure.add_subplot` call
        used to create each subplot.

    gridspec_kw : dict, optional
        Dictionary with keywords passed to the `.GridSpec` constructor used
        to create the grid the subplots are placed on.

    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.

    Returns
    -------
    fig : `.Figure`
       The new figure

    dict[label, Axes]
       A dictionary mapping the labels to the Axes objects.  The order of
       the axes is left-to-right and top-to-bottom of their position in the
       total layout.

    """
    fig = figure(**fig_kw)
    ax_dict = fig.subplot_mosaic(
        mosaic, sharex=sharex, sharey=sharey,
        height_ratios=height_ratios, width_ratios=width_ratios,
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
        empty_sentinel=empty_sentinel
    )
    return fig, ax_dict
```
### 16 - lib/matplotlib/figure.py:

Start line: 1869, End line: 1886

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
        subplot_kw = subplot_kw or {}
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

        # special-case string input
        if isinstance(mosaic, str):
            mosaic = self._normalize_grid_string(mosaic)
        # Only accept strict bools to allow a possible future API expansion.
        _api.check_isinstance(bool, sharex=sharex, sharey=sharey)
        # ... other code
```
### 17 - tutorials/provisional/mosaic.py:

Start line: 288, End line: 339

```python
identify_axes(axd)


###############################################################################
# Nested list input
# =================
#
# Everything we can do with the string shorthand we can also do when
# passing in a list (internally we convert the string shorthand to a nested
# list), for example using spans, blanks, and *gridspec_kw*:

axd = plt.figure(constrained_layout=True).subplot_mosaic(
    [
        ["main", "zoom"],
        ["main", "BLANK"],
    ],
    empty_sentinel="BLANK",
    width_ratios=[2, 1],
)
identify_axes(axd)


###############################################################################
# In addition, using the list input we can specify nested mosaics.  Any element
# of the inner list can be another set of nested lists:

inner = [
    ["inner A"],
    ["inner B"],
]

outer_nested_mosaic = [
    ["main", inner],
    ["bottom", "bottom"],
]
axd = plt.figure(constrained_layout=True).subplot_mosaic(
    outer_nested_mosaic, empty_sentinel=None
)
identify_axes(axd, fontsize=36)


###############################################################################
# We can also pass in a 2D NumPy array to do things like
mosaic = np.zeros((4, 4), dtype=int)
for j in range(4):
    mosaic[j, j] = j + 1
axd = plt.figure(constrained_layout=True).subplot_mosaic(
    mosaic,
    empty_sentinel=0,
)
identify_axes(axd)
```
### 19 - lib/matplotlib/figure.py:

Start line: 2039, End line: 2060

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
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
        for k, ax in ret.items():
            if isinstance(k, str):
                ax.set_label(k)
        return ret

    def _set_artist_props(self, a):
        if a != self:
            a.set_figure(self)
        a.stale_callback = _stale_figure_callback
        a.set_transform(self.transSubfigure)
```
### 20 - lib/matplotlib/figure.py:

Start line: 1772, End line: 1868

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
        """
        Build a layout of Axes based on ASCII art or nested lists.

        This is a helper function to build complex GridSpec layouts visually.

        .. note::

           This API is provisional and may be revised in the future based on
           early user feedback.

        See :doc:`/tutorials/provisional/mosaic`
        for an example and full API documentation

        Parameters
        ----------
        mosaic : list of list of {hashable or nested} or str

            A visual layout of how you want your Axes to be arranged
            labeled as strings.  For example ::

               x = [['A panel', 'A panel', 'edge'],
                    ['C panel', '.',       'edge']]

            produces 4 Axes:

            - 'A panel' which is 1 row high and spans the first two columns
            - 'edge' which is 2 rows high and is on the right edge
            - 'C panel' which in 1 row and 1 column wide in the bottom left
            - a blank space 1 row and 1 column wide in the bottom center

            Any of the entries in the layout can be a list of lists
            of the same form to create nested layouts.

            If input is a str, then it can either be a multi-line string of
            the form ::

              '''
              AAE
              C.E
              '''

            where each character is a column and each line is a row. Or it
            can be a single-line string where rows are separated by ``;``::

              'AB;CC'

            The string notation allows only single character Axes labels and
            does not support nesting but is very terse.

        sharex, sharey : bool, default: False
            If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
            among all subplots.  In that case, tick label visibility and axis
            units behave as for `subplots`.  If False, each subplot's x- or
            y-axis will be independent.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Equivalent
            to ``gridspec_kw={'height_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        subplot_kw : dict, optional
            Dictionary with keywords passed to the `.Figure.add_subplot` call
            used to create each subplot.

        gridspec_kw : dict, optional
            Dictionary with keywords passed to the `.GridSpec` constructor used
            to create the grid the subplots are placed on. In the case of
            nested layouts, this argument applies only to the outer layout.
            For more complex layouts, users should use `.Figure.subfigures`
            to create the nesting.

        empty_sentinel : object, optional
            Entry in the layout to mean "leave this space empty".  Defaults
            to ``'.'``. Note, if *layout* is a string, it is processed via
            `inspect.cleandoc` to remove leading white space, which may
            interfere with using white-space as the empty sentinel.

        Returns
        -------
        dict[label, Axes]
           A dictionary mapping the labels to the Axes objects.  The order of
           the axes is left-to-right and top-to-bottom of their position in the
           total layout.

        """
        # ... other code
```
### 21 - lib/matplotlib/figure.py:

Start line: 1948, End line: 2001

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
        # ... other code

        def _do_layout(gs, mosaic, unique_ids, nested):
            """
            Recursively do the mosaic.

            Parameters
            ----------
            gs : GridSpec
            mosaic : 2D object array
                The input converted to a 2D numpy array for this level.
            unique_ids : tuple
                The identified scalar labels at this level of nesting.
            nested : dict[tuple[int, int]], 2D object array
                The identified nested mosaics, if any.

            Returns
            -------
            dict[label, Axes]
                A flat dict of all of the Axes created.
            """
            output = dict()

            # we need to merge together the Axes at this level and the axes
            # in the (recursively) nested sub-mosaics so that we can add
            # them to the figure in the "natural" order if you were to
            # ravel in c-order all of the Axes that will be created
            #
            # This will stash the upper left index of each object (axes or
            # nested mosaic) at this level
            this_level = dict()

            # go through the unique keys,
            for name in unique_ids:
                # sort out where each axes starts/ends
                indx = np.argwhere(mosaic == name)
                start_row, start_col = np.min(indx, axis=0)
                end_row, end_col = np.max(indx, axis=0) + 1
                # and construct the slice object
                slc = (slice(start_row, end_row), slice(start_col, end_col))
                # some light error checking
                if (mosaic[slc] != name).any():
                    raise ValueError(
                        f"While trying to layout\n{mosaic!r}\n"
                        f"we found that the label {name!r} specifies a "
                        "non-rectangular or non-contiguous area.")
                # and stash this slice for later
                this_level[(start_row, start_col)] = (name, slc, 'axes')

            # do the same thing for the nested mosaics (simpler because these
            # can not be spans yet!)
            for (j, k), nested_mosaic in nested.items():
                this_level[(j, k)] = (None, nested_mosaic, 'nested')

            # now go through the things in this level and add them
            # in order left-to-right top-to-bottom
            # ... other code
        # ... other code
```
### 24 - lib/matplotlib/figure.py:

Start line: 615, End line: 715

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def add_subplot(self, *args, **kwargs):
        """
        Add an `~.axes.Axes` to the figure as part of a subplot arrangement.

        Call signatures::

           add_subplot(nrows, ncols, index, **kwargs)
           add_subplot(pos, **kwargs)
           add_subplot(ax)
           add_subplot()

        Parameters
        ----------
        *args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
            The position of the subplot described by one of

            - Three integers (*nrows*, *ncols*, *index*). The subplot will
              take the *index* position on a grid with *nrows* rows and
              *ncols* columns. *index* starts at 1 in the upper left corner
              and increases to the right.  *index* can also be a two-tuple
              specifying the (*first*, *last*) indices (1-based, and including
              *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))``
              makes a subplot that spans the upper 2/3 of the figure.
            - A 3-digit integer. The digits are interpreted as if given
              separately as three single-digit integers, i.e.
              ``fig.add_subplot(235)`` is the same as
              ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
              if there are no more than 9 subplots.
            - A `.SubplotSpec`.

            In rare circumstances, `.add_subplot` may be called with a single
            argument, a subplot Axes instance already created in the
            present figure but not in the figure's list of Axes.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
         'rectilinear', str}, optional
            The projection type of the subplot (`~.axes.Axes`). *str* is the
            name of a custom projection, see `~matplotlib.projections`. The
            default None results in a 'rectilinear' projection.

        polar : bool, default: False
            If True, equivalent to projection='polar'.

        axes_class : subclass type of `~.axes.Axes`, optional
            The `.axes.Axes` subclass that is instantiated.  This parameter
            is incompatible with *projection* and *polar*.  See
            :ref:`axisartist_users-guide-index` for examples.

        sharex, sharey : `~.axes.Axes`, optional
            Share the x or y `~matplotlib.axis` with sharex and/or sharey.
            The axis will have the same limits, ticks, and scale as the axis
            of the shared axes.

        label : str
            A label for the returned Axes.

        Returns
        -------
        `~.axes.Axes`

            The Axes of the subplot. The returned Axes can actually be an
            instance of a subclass, such as `.projections.polar.PolarAxes` for
            polar projections.

        Other Parameters
        ----------------
        **kwargs
            This method also takes the keyword arguments for the returned Axes
            base class; except for the *figure* argument. The keyword arguments
            for the rectilinear base class `~.axes.Axes` can be found in
            the following table but there might also be other keyword
            arguments if another projection is used.

            %(Axes:kwdoc)s

        See Also
        --------
        .Figure.add_axes
        .pyplot.subplot
        .pyplot.axes
        .Figure.subplots
        .pyplot.subplots

        Examples
        --------
        ::

            fig = plt.figure()

            fig.add_subplot(231)
            ax1 = fig.add_subplot(2, 3, 1)  # equivalent but more general

            fig.add_subplot(232, frameon=False)  # subplot with no frame
            fig.add_subplot(233, projection='polar')  # polar subplot
            fig.add_subplot(234, sharex=ax1)  # subplot sharing x-axis with ax1
            fig.add_subplot(235, facecolor="red")  # red subplot

            ax1.remove()  # delete ax1 from the figure
            fig.add_subplot(ax1)  # add ax1 back to the figure
        """
        # ... other code
```
### 28 - lib/matplotlib/figure.py:

Start line: 1471, End line: 1512

```python
class FigureBase(Artist):

    def add_gridspec(self, nrows=1, ncols=1, **kwargs):
        """
        Return a `.GridSpec` that has this figure as a parent.  This allows
        complex layout of Axes in the figure.

        Parameters
        ----------
        nrows : int, default: 1
            Number of rows in grid.

        ncols : int, default: 1
            Number of columns in grid.

        Returns
        -------
        `.GridSpec`

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments are passed to `.GridSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding a subplot that spans two rows::

            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            # spans two rows:
            ax3 = fig.add_subplot(gs[:, 1])

        """

        _ = kwargs.pop('figure', None)  # pop in case user has added this...
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self, **kwargs)
        return gs
```
### 40 - lib/matplotlib/figure.py:

Start line: 1647, End line: 1689

```python
class FigureBase(Artist):

    def _process_projection_requirements(
            self, *args, axes_class=None, polar=False, projection=None,
            **kwargs):
        """
        Handle the args/kwargs to add_axes/add_subplot/gca, returning::

            (axes_proj_class, proj_class_kwargs)

        which can be used for new Axes initialization/identification.
        """
        if axes_class is not None:
            if polar or projection is not None:
                raise ValueError(
                    "Cannot combine 'axes_class' and 'projection' or 'polar'")
            projection_class = axes_class
        else:

            if polar:
                if projection is not None and projection != 'polar':
                    raise ValueError(
                        f"polar={polar}, yet projection={projection!r}. "
                        "Only one of these arguments should be supplied."
                    )
                projection = 'polar'

            if isinstance(projection, str) or projection is None:
                projection_class = projections.get_projection_class(projection)
            elif hasattr(projection, '_as_mpl_axes'):
                projection_class, extra_kwargs = projection._as_mpl_axes()
                kwargs.update(**extra_kwargs)
            else:
                raise TypeError(
                    f"projection must be a string, None or implement a "
                    f"_as_mpl_axes method, not {projection!r}")
        return projection_class, kwargs

    def get_default_bbox_extra_artists(self):
        bbox_artists = [artist for artist in self.get_children()
                        if (artist.get_visible() and artist.get_in_layout())]
        for ax in self.axes:
            if ax.get_visible():
                bbox_artists.extend(ax.get_default_bbox_extra_artists())
        return bbox_artists
```
### 44 - lib/matplotlib/figure.py:

Start line: 716, End line: 756

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def add_subplot(self, *args, **kwargs):
        if 'figure' in kwargs:
            # Axes itself allows for a 'figure' kwarg, but since we want to
            # bind the created Axes to self, it is not allowed here.
            raise TypeError(
                "add_subplot() got an unexpected keyword argument 'figure'")

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
            projection_class, pkw = self._process_projection_requirements(
                *args, **kwargs)
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
### 47 - lib/matplotlib/pyplot.py:

Start line: 1136, End line: 1271

```python
## More ways of creating axes ##

@_docstring.dedent_interpd
def subplot(*args, **kwargs):
    """
    Add an Axes to the current figure or retrieve an existing Axes.

    This is a wrapper of `.Figure.add_subplot` which provides additional
    behavior when working with the implicit API (see the notes section).

    Call signatures::

       subplot(nrows, ncols, index, **kwargs)
       subplot(pos, **kwargs)
       subplot(**kwargs)
       subplot(ax)

    Parameters
    ----------
    *args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
        The position of the subplot described by one of

        - Three integers (*nrows*, *ncols*, *index*). The subplot will take the
          *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the
          right. *index* can also be a two-tuple specifying the (*first*,
          *last*) indices (1-based, and including *last*) of the subplot, e.g.,
          ``fig.add_subplot(3, 1, (1, 2))`` makes a subplot that spans the
          upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately
          as three single-digit integers, i.e. ``fig.add_subplot(235)`` is the
          same as ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
          if there are no more than 9 subplots.
        - A `.SubplotSpec`.

    projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
    ar', 'rectilinear', str}, optional
        The projection type of the subplot (`~.axes.Axes`). *str* is the name
        of a custom projection, see `~matplotlib.projections`. The default
        None results in a 'rectilinear' projection.

    polar : bool, default: False
        If True, equivalent to projection='polar'.

    sharex, sharey : `~.axes.Axes`, optional
        Share the x or y `~matplotlib.axis` with sharex and/or sharey. The
        axis will have the same limits, ticks, and scale as the axis of the
        shared axes.

    label : str
        A label for the returned axes.

    Returns
    -------
    `~.axes.Axes`

        The Axes of the subplot. The returned Axes can actually be an instance
        of a subclass, such as `.projections.polar.PolarAxes` for polar
        projections.

    Other Parameters
    ----------------
    **kwargs
        This method also takes the keyword arguments for the returned axes
        base class; except for the *figure* argument. The keyword arguments
        for the rectilinear base class `~.axes.Axes` can be found in
        the following table but there might also be other keyword
        arguments if another projection is used.

        %(Axes:kwdoc)s

    Notes
    -----
    Creating a new Axes will delete any preexisting Axes that
    overlaps with it beyond sharing a boundary::

        import matplotlib.pyplot as plt
        # plot a line, implicitly creating a subplot(111)
        plt.plot([1, 2, 3])
        # now create a subplot which represents the top plot of a grid
        # with 2 rows and 1 column. Since this subplot will overlap the
        # first, the plot (and its axes) previously created, will be removed
        plt.subplot(211)

    If you do not want this behavior, use the `.Figure.add_subplot` method
    or the `.pyplot.axes` function instead.

    If no *kwargs* are passed and there exists an Axes in the location
    specified by *args* then that Axes will be returned rather than a new
    Axes being created.

    If *kwargs* are passed and there exists an Axes in the location
    specified by *args*, the projection type is the same, and the
    *kwargs* match with the existing Axes, then the existing Axes is
    returned.  Otherwise a new Axes is created with the specified
    parameters.  We save a reference to the *kwargs* which we use
    for this comparison.  If any of the values in *kwargs* are
    mutable we will not detect the case where they are mutated.
    In these cases we suggest using `.Figure.add_subplot` and the
    explicit Axes API rather than the implicit pyplot API.

    See Also
    --------
    .Figure.add_subplot
    .pyplot.subplots
    .pyplot.axes
    .Figure.subplots

    Examples
    --------
    ::

        plt.subplot(221)

        # equivalent but more general
        ax1 = plt.subplot(2, 2, 1)

        # add a subplot with no frame
        ax2 = plt.subplot(222, frameon=False)

        # add a polar subplot
        plt.subplot(223, projection='polar')

        # add a red subplot that shares the x-axis with ax1
        plt.subplot(224, sharex=ax1, facecolor='red')

        # delete ax2 from the figure
        plt.delaxes(ax2)

        # add ax2 to the figure again
        plt.subplot(ax2)

        # make the first axes "current" again
        plt.subplot(221)

    """
    # ... other code
```
### 48 - lib/matplotlib/figure.py:

Start line: 1888, End line: 1918

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
        # ... other code

        def _make_array(inp):
            """
            Convert input into 2D array

            We need to have this internal function rather than
            ``np.asarray(..., dtype=object)`` so that a list of lists
            of lists does not get converted to an array of dimension >
            2

            Returns
            -------
            2D object array

            """
            r0, *rest = inp
            if isinstance(r0, str):
                raise ValueError('List mosaic specification must be 2D')
            for j, r in enumerate(rest, start=1):
                if isinstance(r, str):
                    raise ValueError('List mosaic specification must be 2D')
                if len(r0) != len(r):
                    raise ValueError(
                        "All of the rows must be the same length, however "
                        f"the first row ({r0!r}) has length {len(r0)} "
                        f"and row {j} ({r!r}) has length {len(r)}."
                    )
            out = np.zeros((len(inp), len(r0)), dtype=object)
            for j, r in enumerate(inp):
                for k, v in enumerate(r):
                    out[j, k] = v
            return out
        # ... other code
```
### 54 - lib/matplotlib/figure.py:

Start line: 1920, End line: 1946

```python
class FigureBase(Artist):

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.', subplot_kw=None, gridspec_kw=None):
        # ... other code

        def _identify_keys_and_nested(mosaic):
            """
            Given a 2D object array, identify unique IDs and nested mosaics

            Parameters
            ----------
            mosaic : 2D numpy object array

            Returns
            -------
            unique_ids : tuple
                The unique non-sub mosaic entries in this mosaic
            nested : dict[tuple[int, int]], 2D object array
            """
            # make sure we preserve the user supplied order
            unique_ids = cbook._OrderedSet()
            nested = {}
            for j, row in enumerate(mosaic):
                for k, v in enumerate(row):
                    if v == empty_sentinel:
                        continue
                    elif not cbook.is_scalar_or_string(v):
                        nested[(j, k)] = _make_array(v)
                    else:
                        unique_ids.add(v)

            return tuple(unique_ids), nested
        # ... other code
```
### 55 - lib/matplotlib/figure.py:

Start line: 758, End line: 876

```python
class FigureBase(Artist):

    def subplots(self, nrows=1, ncols=1, *, sharex=False, sharey=False,
                 squeeze=True, width_ratios=None, height_ratios=None,
                 subplot_kw=None, gridspec_kw=None):
        """
        Add a set of subplots to this figure.

        This utility wrapper makes it convenient to create common layouts of
        subplots in a single call.

        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subplot grid.

        sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
            Controls sharing of x-axis (*sharex*) or y-axis (*sharey*):

            - True or 'all': x- or y-axis will be shared among all subplots.
            - False or 'none': each subplot x- or y-axis will be independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

            When subplots have a shared x-axis along a column, only the x tick
            labels of the bottom subplot are created. Similarly, when subplots
            have a shared y-axis along a row, only the y tick labels of the
            first column subplot are created. To later turn other subplots'
            ticklabels on, use `~matplotlib.axes.Axes.tick_params`.

            When subplots have a shared axis that has units, calling
            `.Axis.set_units` will update each axis with the new units.

        squeeze : bool, default: True
            - If True, extra dimensions are squeezed out from the returned
              array of Axes:

              - if only one subplot is constructed (nrows=ncols=1), the
                resulting single Axes object is returned as a scalar.
              - for Nx1 or 1xM subplots, the returned object is a 1D numpy
                object array of Axes objects.
              - for NxM, subplots with N>1 and M>1 are returned as a 2D array.

            - If False, no squeezing at all is done: the returned Axes object
              is always a 2D array containing Axes instances, even if it ends
              up being 1x1.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Equivalent
            to ``gridspec_kw={'height_ratios': [...]}``.

        subplot_kw : dict, optional
            Dict with keywords passed to the `.Figure.add_subplot` call used to
            create each subplot.

        gridspec_kw : dict, optional
            Dict with keywords passed to the
            `~matplotlib.gridspec.GridSpec` constructor used to create
            the grid the subplots are placed on.

        Returns
        -------
        `~.axes.Axes` or array of Axes
            Either a single `~matplotlib.axes.Axes` object or an array of Axes
            objects if more than one subplot was created. The dimensions of the
            resulting array can be controlled with the *squeeze* keyword, see
            above.

        See Also
        --------
        .pyplot.subplots
        .Figure.add_subplot
        .pyplot.subplot

        Examples
        --------
        ::

            # First create some toy data:
            x = np.linspace(0, 2*np.pi, 400)
            y = np.sin(x**2)

            # Create a figure
            plt.figure()

            # Create a subplot
            ax = fig.subplots()
            ax.plot(x, y)
            ax.set_title('Simple plot')

            # Create two subplots and unpack the output array immediately
            ax1, ax2 = fig.subplots(1, 2, sharey=True)
            ax1.plot(x, y)
            ax1.set_title('Sharing Y axis')
            ax2.scatter(x, y)

            # Create four polar Axes and access them through the returned array
            axes = fig.subplots(2, 2, subplot_kw=dict(projection='polar'))
            axes[0, 0].plot(x, y)
            axes[1, 1].scatter(x, y)

            # Share an X-axis with each column of subplots
            fig.subplots(2, 2, sharex='col')

            # Share a Y-axis with each row of subplots
            fig.subplots(2, 2, sharey='row')

            # Share both X- and Y-axes with all subplots
            fig.subplots(2, 2, sharex='all', sharey='all')

            # Note that this is the same as
            fig.subplots(2, 2, sharex=True, sharey=True)
        """
        # ... other code
```
### 63 - lib/matplotlib/figure.py:

Start line: 877, End line: 892

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
### 76 - lib/matplotlib/figure.py:

Start line: 1568, End line: 1593

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
