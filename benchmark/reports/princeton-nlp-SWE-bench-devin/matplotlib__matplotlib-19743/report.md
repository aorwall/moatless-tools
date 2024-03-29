# matplotlib__matplotlib-19743

| **matplotlib/matplotlib** | `5793ebb2201bf778f08ac1d4cd0b8dd674c96053` |
| ---- | ---- |
| **No of patches** | 6 |
| **All found context length** | 10614 |
| **Any found context length** | 10614 |
| **Avg pos** | 61.166666666666664 |
| **Min pos** | 18 |
| **Max pos** | 70 |
| **Top file pos** | 3 |
| **Missing snippets** | 8 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/examples/text_labels_and_annotations/figlegend_demo.py b/examples/text_labels_and_annotations/figlegend_demo.py
--- a/examples/text_labels_and_annotations/figlegend_demo.py
+++ b/examples/text_labels_and_annotations/figlegend_demo.py
@@ -28,3 +28,26 @@
 
 plt.tight_layout()
 plt.show()
+
+##############################################################################
+# Sometimes we do not want the legend to overlap the axes.  If you use
+# constrained_layout you can specify "outside right upper", and
+# constrained_layout will make room for the legend.
+
+fig, axs = plt.subplots(1, 2, layout='constrained')
+
+x = np.arange(0.0, 2.0, 0.02)
+y1 = np.sin(2 * np.pi * x)
+y2 = np.exp(-x)
+l1, = axs[0].plot(x, y1)
+l2, = axs[0].plot(x, y2, marker='o')
+
+y3 = np.sin(4 * np.pi * x)
+y4 = np.exp(-2 * x)
+l3, = axs[1].plot(x, y3, color='tab:green')
+l4, = axs[1].plot(x, y4, color='tab:red', marker='^')
+
+fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')
+fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')
+
+plt.show()
diff --git a/lib/matplotlib/_constrained_layout.py b/lib/matplotlib/_constrained_layout.py
--- a/lib/matplotlib/_constrained_layout.py
+++ b/lib/matplotlib/_constrained_layout.py
@@ -418,6 +418,25 @@ def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
         # pass the new margins down to the layout grid for the solution...
         layoutgrids[gs].edit_outer_margin_mins(margin, ss)
 
+    # make margins for figure-level legends:
+    for leg in fig.legends:
+        inv_trans_fig = None
+        if leg._outside_loc and leg._bbox_to_anchor is None:
+            if inv_trans_fig is None:
+                inv_trans_fig = fig.transFigure.inverted().transform_bbox
+            bbox = inv_trans_fig(leg.get_tightbbox(renderer))
+            w = bbox.width + 2 * w_pad
+            h = bbox.height + 2 * h_pad
+            legendloc = leg._outside_loc
+            if legendloc == 'lower':
+                layoutgrids[fig].edit_margin_min('bottom', h)
+            elif legendloc == 'upper':
+                layoutgrids[fig].edit_margin_min('top', h)
+            if legendloc == 'right':
+                layoutgrids[fig].edit_margin_min('right', w)
+            elif legendloc == 'left':
+                layoutgrids[fig].edit_margin_min('left', w)
+
 
 def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
     # Figure out how large the suptitle is and make the
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -294,7 +294,7 @@ def legend(self, *args, **kwargs):
 
         Other Parameters
         ----------------
-        %(_legend_kw_doc)s
+        %(_legend_kw_axes)s
 
         See Also
         --------
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1085,7 +1085,8 @@ def legend(self, *args, **kwargs):
 
         Other Parameters
         ----------------
-        %(_legend_kw_doc)s
+        %(_legend_kw_figure)s
+
 
         See Also
         --------
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -94,51 +94,7 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
         self.legend.set_bbox_to_anchor(loc_in_bbox)
 
 
-_docstring.interpd.update(_legend_kw_doc="""
-loc : str or pair of floats, default: :rc:`legend.loc` ('best' for axes, \
-'upper right' for figures)
-    The location of the legend.
-
-    The strings
-    ``'upper left', 'upper right', 'lower left', 'lower right'``
-    place the legend at the corresponding corner of the axes/figure.
-
-    The strings
-    ``'upper center', 'lower center', 'center left', 'center right'``
-    place the legend at the center of the corresponding edge of the
-    axes/figure.
-
-    The string ``'center'`` places the legend at the center of the axes/figure.
-
-    The string ``'best'`` places the legend at the location, among the nine
-    locations defined so far, with the minimum overlap with other drawn
-    artists.  This option can be quite slow for plots with large amounts of
-    data; your plotting speed may benefit from providing a specific location.
-
-    The location can also be a 2-tuple giving the coordinates of the lower-left
-    corner of the legend in axes coordinates (in which case *bbox_to_anchor*
-    will be ignored).
-
-    For back-compatibility, ``'center right'`` (but no other location) can also
-    be spelled ``'right'``, and each "string" locations can also be given as a
-    numeric value:
-
-        ===============   =============
-        Location String   Location Code
-        ===============   =============
-        'best'            0
-        'upper right'     1
-        'upper left'      2
-        'lower left'      3
-        'lower right'     4
-        'right'           5
-        'center left'     6
-        'center right'    7
-        'lower center'    8
-        'upper center'    9
-        'center'          10
-        ===============   =============
-
+_legend_kw_doc_base = """
 bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
     Box that is used to position the legend in conjunction with *loc*.
     Defaults to `axes.bbox` (if called as a method to `.Axes.legend`) or
@@ -295,7 +251,79 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
 
 draggable : bool, default: False
     Whether the legend can be dragged with the mouse.
-""")
+"""
+
+_loc_doc_base = """
+loc : str or pair of floats, {0}
+    The location of the legend.
+
+    The strings
+    ``'upper left', 'upper right', 'lower left', 'lower right'``
+    place the legend at the corresponding corner of the axes/figure.
+
+    The strings
+    ``'upper center', 'lower center', 'center left', 'center right'``
+    place the legend at the center of the corresponding edge of the
+    axes/figure.
+
+    The string ``'center'`` places the legend at the center of the axes/figure.
+
+    The string ``'best'`` places the legend at the location, among the nine
+    locations defined so far, with the minimum overlap with other drawn
+    artists.  This option can be quite slow for plots with large amounts of
+    data; your plotting speed may benefit from providing a specific location.
+
+    The location can also be a 2-tuple giving the coordinates of the lower-left
+    corner of the legend in axes coordinates (in which case *bbox_to_anchor*
+    will be ignored).
+
+    For back-compatibility, ``'center right'`` (but no other location) can also
+    be spelled ``'right'``, and each "string" locations can also be given as a
+    numeric value:
+
+        ===============   =============
+        Location String   Location Code
+        ===============   =============
+        'best'            0
+        'upper right'     1
+        'upper left'      2
+        'lower left'      3
+        'lower right'     4
+        'right'           5
+        'center left'     6
+        'center right'    7
+        'lower center'    8
+        'upper center'    9
+        'center'          10
+        ===============   =============
+    {1}"""
+
+_legend_kw_axes_st = (_loc_doc_base.format("default: :rc:`legend.loc`", '') +
+                      _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_axes=_legend_kw_axes_st)
+
+_outside_doc = """
+    If a figure is using the constrained layout manager, the string codes
+    of the *loc* keyword argument can get better layout behaviour using the
+    prefix 'outside'. There is ambiguity at the corners, so 'outside
+    upper right' will make space for the legend above the rest of the
+    axes in the layout, and 'outside right upper' will make space on the
+    right side of the layout.  In addition to the values of *loc*
+    listed above, we have 'outside right upper', 'outside right lower',
+    'outside left upper', and 'outside left lower'.  See
+    :doc:`/tutorials/intermediate/legend_guide` for more details.
+"""
+
+_legend_kw_figure_st = (_loc_doc_base.format("default: 'upper right'",
+                                             _outside_doc) +
+                        _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_figure=_legend_kw_figure_st)
+
+_legend_kw_both_st = (
+    _loc_doc_base.format("default: 'best' for axes, 'upper right' for figures",
+                         _outside_doc) +
+    _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_doc=_legend_kw_both_st)
 
 
 class Legend(Artist):
@@ -482,13 +510,37 @@ def val_or_rc(val, rc_name):
             )
         self.parent = parent
 
+        loc0 = loc
         self._loc_used_default = loc is None
         if loc is None:
             loc = mpl.rcParams["legend.loc"]
             if not self.isaxes and loc in [0, 'best']:
                 loc = 'upper right'
+
+        # handle outside legends:
+        self._outside_loc = None
         if isinstance(loc, str):
+            if loc.split()[0] == 'outside':
+                # strip outside:
+                loc = loc.split('outside ')[1]
+                # strip "center" at the beginning
+                self._outside_loc = loc.replace('center ', '')
+                # strip first
+                self._outside_loc = self._outside_loc.split()[0]
+                locs = loc.split()
+                if len(locs) > 1 and locs[0] in ('right', 'left'):
+                    # locs doesn't accept "left upper", etc, so swap
+                    if locs[0] != 'center':
+                        locs = locs[::-1]
+                    loc = locs[0] + ' ' + locs[1]
+            # check that loc is in acceptable strings
             loc = _api.check_getitem(self.codes, loc=loc)
+
+        if self.isaxes and self._outside_loc:
+            raise ValueError(
+                f"'outside' option for loc='{loc0}' keyword argument only "
+                "works for figure legends")
+
         if not self.isaxes and loc == 0:
             raise ValueError(
                 "Automatic legend placement (loc='best') not implemented for "
diff --git a/tutorials/intermediate/legend_guide.py b/tutorials/intermediate/legend_guide.py
--- a/tutorials/intermediate/legend_guide.py
+++ b/tutorials/intermediate/legend_guide.py
@@ -135,7 +135,54 @@
 ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1),
                          loc='upper left', borderaxespad=0.)
 
-plt.show()
+##############################################################################
+# Figure legends
+# --------------
+#
+# Sometimes it makes more sense to place a legend relative to the (sub)figure
+# rather than individual Axes.  By using ``constrained_layout`` and
+# specifying "outside" at the beginning of the *loc* keyword argument,
+# the legend is drawn outside the Axes on the (sub)figure.
+
+fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
+
+axs['left'].plot([1, 2, 3], label="test1")
+axs['left'].plot([3, 2, 1], label="test2")
+
+axs['right'].plot([1, 2, 3], 'C2', label="test3")
+axs['right'].plot([3, 2, 1], 'C3', label="test4")
+# Place a legend to the right of this smaller subplot.
+fig.legend(loc='outside upper right')
+
+##############################################################################
+# This accepts a slightly different grammar than the normal *loc* keyword,
+# where "outside right upper" is different from "outside upper right".
+#
+ucl = ['upper', 'center', 'lower']
+lcr = ['left', 'center', 'right']
+fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
+
+ax.plot([1, 2], [1, 2], label='TEST')
+# Place a legend to the right of this smaller subplot.
+for loc in [
+        'outside upper left',
+        'outside upper center',
+        'outside upper right',
+        'outside lower left',
+        'outside lower center',
+        'outside lower right']:
+    fig.legend(loc=loc, title=loc)
+
+fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
+ax.plot([1, 2], [1, 2], label='test')
+
+for loc in [
+        'outside left upper',
+        'outside right upper',
+        'outside left lower',
+        'outside right lower']:
+    fig.legend(loc=loc, title=loc)
+
 
 ###############################################################################
 # Multiple legends on the same Axes

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| examples/text_labels_and_annotations/figlegend_demo.py | 31 | 31 | 44 | 14 | 23381
| lib/matplotlib/_constrained_layout.py | 421 | 421 | - | 5 | -
| lib/matplotlib/axes/_axes.py | 297 | 297 | - | - | -
| lib/matplotlib/figure.py | 1088 | 1088 | 18 | 3 | 10614
| lib/matplotlib/legend.py | 97 | 141 | 70 | 9 | 36385
| lib/matplotlib/legend.py | 298 | 298 | 70 | 9 | 36385
| lib/matplotlib/legend.py | 485 | 485 | 70 | 9 | 36385
| tutorials/intermediate/legend_guide.py | 138 | 138 | 35 | 10 | 20140


## Problem Statement

```
constrained_layout support for figure.legend
Just a feature request to have constrained_layout support `figure.legend`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 tutorials/intermediate/constrainedlayout_guide.py | 113 | 185| 781 | 781 | 6566 | 
| 2 | 1 tutorials/intermediate/constrainedlayout_guide.py | 1 | 112| 829 | 1610 | 6566 | 
| 3 | 1 tutorials/intermediate/constrainedlayout_guide.py | 532 | 634| 1013 | 2623 | 6566 | 
| 4 | 1 tutorials/intermediate/constrainedlayout_guide.py | 187 | 269| 813 | 3436 | 6566 | 
| 5 | 2 examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 430 | 3866 | 6996 | 
| 6 | **3 lib/matplotlib/figure.py** | 2781 | 2809| 264 | 4130 | 36213 | 
| 7 | 4 lib/matplotlib/layout_engine.py | 237 | 254| 149 | 4279 | 38579 | 
| 8 | 4 tutorials/intermediate/constrainedlayout_guide.py | 354 | 441| 772 | 5051 | 38579 | 
| 9 | **4 lib/matplotlib/figure.py** | 2264 | 2286| 174 | 5225 | 38579 | 
| 10 | 4 tutorials/intermediate/constrainedlayout_guide.py | 270 | 352| 804 | 6029 | 38579 | 
| 11 | 4 tutorials/intermediate/constrainedlayout_guide.py | 636 | 721| 777 | 6806 | 38579 | 
| 12 | 4 tutorials/intermediate/constrainedlayout_guide.py | 442 | 530| 776 | 7582 | 38579 | 
| 13 | **5 lib/matplotlib/_constrained_layout.py** | 111 | 149| 471 | 8053 | 45945 | 
| 14 | **5 lib/matplotlib/_constrained_layout.py** | 62 | 109| 386 | 8439 | 45945 | 
| 15 | **5 lib/matplotlib/figure.py** | 2747 | 2779| 286 | 8725 | 45945 | 
| 16 | **5 lib/matplotlib/_constrained_layout.py** | 1 | 59| 602 | 9327 | 45945 | 
| 17 | **5 lib/matplotlib/_constrained_layout.py** | 422 | 458| 473 | 9800 | 45945 | 
| **-> 18 <-** | **5 lib/matplotlib/figure.py** | 989 | 1098| 814 | 10614 | 45945 | 
| 19 | **5 lib/matplotlib/figure.py** | 1100 | 1122| 242 | 10856 | 45945 | 
| 20 | **5 lib/matplotlib/_constrained_layout.py** | 357 | 419| 766 | 11622 | 45945 | 
| 21 | **5 lib/matplotlib/figure.py** | 2811 | 2843| 329 | 11951 | 45945 | 
| 22 | 6 tutorials/intermediate/tight_layout_guide.py | 1 | 104| 778 | 12729 | 48098 | 
| 23 | 6 tutorials/intermediate/tight_layout_guide.py | 222 | 293| 495 | 13224 | 48098 | 
| 24 | 7 examples/text_labels_and_annotations/legend_demo.py | 76 | 116| 567 | 13791 | 49999 | 
| 25 | **7 lib/matplotlib/_constrained_layout.py** | 492 | 555| 719 | 14510 | 49999 | 
| 26 | **7 lib/matplotlib/_constrained_layout.py** | 263 | 297| 327 | 14837 | 49999 | 
| 27 | **7 lib/matplotlib/_constrained_layout.py** | 338 | 355| 206 | 15043 | 49999 | 
| 28 | **7 lib/matplotlib/_constrained_layout.py** | 461 | 490| 297 | 15340 | 49999 | 
| 29 | 8 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 115| 756 | 16096 | 50878 | 
| 30 | **8 lib/matplotlib/_constrained_layout.py** | 152 | 194| 391 | 16487 | 50878 | 
| 31 | 8 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 880 | 17367 | 50878 | 
| 32 | **9 lib/matplotlib/legend.py** | 486 | 612| 1598 | 18965 | 61979 | 
| 33 | 9 lib/matplotlib/layout_engine.py | 256 | 282| 279 | 19244 | 61979 | 
| 34 | **9 lib/matplotlib/_constrained_layout.py** | 750 | 763| 128 | 19372 | 61979 | 
| **-> 35 <-** | **10 tutorials/intermediate/legend_guide.py** | 123 | 195| 768 | 20140 | 64643 | 
| 36 | **10 lib/matplotlib/legend.py** | 846 | 861| 176 | 20316 | 64643 | 
| 37 | 11 lib/matplotlib/_layoutgrid.py | 180 | 219| 425 | 20741 | 70079 | 
| 38 | 11 lib/matplotlib/layout_engine.py | 1 | 24| 229 | 20970 | 70079 | 
| 39 | 11 lib/matplotlib/layout_engine.py | 188 | 235| 506 | 21476 | 70079 | 
| 40 | **11 lib/matplotlib/figure.py** | 1762 | 1789| 203 | 21679 | 70079 | 
| 41 | 12 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 22435 | 71502 | 
| 42 | 13 lib/matplotlib/_tight_layout.py | 266 | 302| 362 | 22797 | 74431 | 
| 43 | **13 lib/matplotlib/figure.py** | 2845 | 2886| 328 | 23125 | 74431 | 
| **-> 44 <-** | **14 examples/text_labels_and_annotations/figlegend_demo.py** | 1 | 31| 256 | 23381 | 74687 | 
| 45 | 15 examples/text_labels_and_annotations/custom_legends.py | 1 | 72| 557 | 23938 | 75244 | 
| 46 | 15 examples/text_labels_and_annotations/legend_demo.py | 1 | 75| 735 | 24673 | 75244 | 
| 47 | 16 examples/userdemo/simple_legend01.py | 1 | 27| 199 | 24872 | 75443 | 
| 48 | **16 lib/matplotlib/_constrained_layout.py** | 733 | 747| 140 | 25012 | 75443 | 
| 49 | **16 lib/matplotlib/figure.py** | 2574 | 2638| 562 | 25574 | 75443 | 
| 50 | **16 tutorials/intermediate/legend_guide.py** | 1 | 121| 890 | 26464 | 75443 | 
| 51 | **16 lib/matplotlib/legend.py** | 614 | 644| 229 | 26693 | 75443 | 
| 52 | **16 lib/matplotlib/figure.py** | 2555 | 2572| 159 | 26852 | 75443 | 
| 53 | **16 lib/matplotlib/legend.py** | 1007 | 1048| 337 | 27189 | 75443 | 
| 54 | **16 lib/matplotlib/figure.py** | 1 | 53| 340 | 27529 | 75443 | 
| 55 | **16 lib/matplotlib/figure.py** | 2533 | 2553| 283 | 27812 | 75443 | 
| 56 | **16 tutorials/intermediate/legend_guide.py** | 197 | 283| 812 | 28624 | 75443 | 
| 57 | 16 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 29291 | 75443 | 
| 58 | **16 lib/matplotlib/legend.py** | 898 | 938| 307 | 29598 | 75443 | 
| 59 | **16 lib/matplotlib/figure.py** | 2362 | 2531| 1513 | 31111 | 75443 | 
| **-> 60 <-** | **16 lib/matplotlib/legend.py** | 302 | 485| 1639 | 32750 | 75443 | 
| 61 | 16 lib/matplotlib/_layoutgrid.py | 221 | 258| 489 | 33239 | 75443 | 
| 62 | **16 lib/matplotlib/figure.py** | 2640 | 2652| 141 | 33380 | 75443 | 
| 63 | **16 lib/matplotlib/figure.py** | 388 | 397| 158 | 33538 | 75443 | 
| 64 | **16 lib/matplotlib/legend.py** | 663 | 692| 238 | 33776 | 75443 | 
| 65 | 16 examples/text_labels_and_annotations/legend_demo.py | 164 | 182| 195 | 33971 | 75443 | 
| 66 | 17 examples/lines_bars_and_markers/scatter_with_legend.py | 1 | 96| 827 | 34798 | 76373 | 
| 67 | 18 lib/matplotlib/backends/qt_editor/figureoptions.py | 113 | 175| 618 | 35416 | 78549 | 
| 68 | **18 lib/matplotlib/_constrained_layout.py** | 197 | 240| 435 | 35851 | 78549 | 
| 69 | **18 lib/matplotlib/figure.py** | 378 | 386| 150 | 36001 | 78549 | 
| **-> 70 <-** | **18 lib/matplotlib/legend.py** | 53 | 1157| 384 | 36385 | 78549 | 
| 71 | 19 lib/matplotlib/legend_handler.py | 569 | 651| 762 | 37147 | 85420 | 
| 72 | **19 lib/matplotlib/_constrained_layout.py** | 606 | 645| 391 | 37538 | 85420 | 
| 73 | **19 lib/matplotlib/figure.py** | 3464 | 3502| 369 | 37907 | 85420 | 
| 74 | 20 tutorials/intermediate/arranging_axes.py | 100 | 178| 776 | 38683 | 89370 | 
| 75 | 20 examples/text_labels_and_annotations/legend_demo.py | 118 | 162| 403 | 39086 | 89370 | 
| 76 | **20 lib/matplotlib/_constrained_layout.py** | 243 | 260| 153 | 39239 | 89370 | 
| 77 | **20 lib/matplotlib/legend.py** | 1 | 50| 393 | 39632 | 89370 | 
| 78 | 21 lib/matplotlib/pyplot.py | 2696 | 2711| 169 | 39801 | 117987 | 
| 79 | 21 lib/matplotlib/legend_handler.py | 439 | 450| 135 | 39936 | 117987 | 
| 80 | **21 lib/matplotlib/figure.py** | 2047 | 2086| 401 | 40337 | 117987 | 
| 81 | 22 examples/text_labels_and_annotations/legend.py | 1 | 40| 233 | 40570 | 118220 | 
| 82 | 22 lib/matplotlib/backends/qt_editor/figureoptions.py | 86 | 111| 293 | 40863 | 118220 | 
| 83 | 23 examples/userdemo/simple_legend02.py | 1 | 24| 136 | 40999 | 118356 | 
| 84 | 24 lib/matplotlib/backends/_backend_gtk.py | 251 | 265| 182 | 41181 | 120880 | 
| 85 | 25 lib/matplotlib/artist.py | 1408 | 1417| 118 | 41299 | 134600 | 
| 86 | **25 lib/matplotlib/legend.py** | 1050 | 1067| 179 | 41478 | 134600 | 
| 87 | 25 examples/subplots_axes_and_figures/demo_tight_layout.py | 116 | 135| 123 | 41601 | 134600 | 
| 88 | 25 lib/matplotlib/legend_handler.py | 280 | 295| 144 | 41745 | 134600 | 
| 89 | **25 lib/matplotlib/figure.py** | 596 | 613| 190 | 41935 | 134600 | 
| 90 | 25 lib/matplotlib/_layoutgrid.py | 1 | 28| 238 | 42173 | 134600 | 
| 91 | 25 tutorials/intermediate/arranging_axes.py | 264 | 343| 760 | 42933 | 134600 | 
| 92 | **25 lib/matplotlib/figure.py** | 368 | 376| 154 | 43087 | 134600 | 
| 93 | 25 lib/matplotlib/_tight_layout.py | 194 | 264| 733 | 43820 | 134600 | 
| 94 | **25 lib/matplotlib/legend.py** | 763 | 844| 810 | 44630 | 134600 | 
| 95 | **25 lib/matplotlib/figure.py** | 877 | 892| 245 | 44875 | 134600 | 
| 96 | 25 lib/matplotlib/pyplot.py | 993 | 1010| 148 | 45023 | 134600 | 
| 97 | **25 lib/matplotlib/figure.py** | 1251 | 1282| 367 | 45390 | 134600 | 
| 98 | **25 lib/matplotlib/figure.py** | 1568 | 1593| 148 | 45538 | 134600 | 
| 99 | 25 lib/matplotlib/layout_engine.py | 158 | 185| 233 | 45771 | 134600 | 
| 100 | 25 lib/matplotlib/_layoutgrid.py | 151 | 178| 285 | 46056 | 134600 | 
| 101 | **25 lib/matplotlib/figure.py** | 1284 | 1324| 369 | 46425 | 134600 | 
| 102 | 26 examples/text_labels_and_annotations/label_subplots.py | 1 | 71| 603 | 47028 | 135203 | 
| 103 | 27 lib/matplotlib/backends/backend_gtk4.py | 33 | 87| 427 | 47455 | 139925 | 
| 104 | 27 lib/matplotlib/artist.py | 1080 | 1130| 301 | 47756 | 139925 | 
| 105 | **27 lib/matplotlib/_constrained_layout.py** | 558 | 575| 156 | 47912 | 139925 | 
| 106 | 28 examples/pie_and_polar_charts/polar_legend.py | 1 | 41| 337 | 48249 | 140262 | 
| 107 | 28 lib/matplotlib/_tight_layout.py | 96 | 157| 738 | 48987 | 140262 | 
| 108 | 29 lib/matplotlib/backends/_backend_tk.py | 514 | 541| 297 | 49284 | 149768 | 
| 109 | 29 lib/matplotlib/backends/_backend_tk.py | 456 | 491| 359 | 49643 | 149768 | 
| 110 | 29 tutorials/intermediate/arranging_axes.py | 345 | 413| 671 | 50314 | 149768 | 
| 111 | 29 lib/matplotlib/legend_handler.py | 408 | 418| 124 | 50438 | 149768 | 
| 112 | **29 lib/matplotlib/figure.py** | 716 | 756| 454 | 50892 | 149768 | 
| 113 | **29 lib/matplotlib/figure.py** | 288 | 366| 707 | 51599 | 149768 | 
| 114 | **29 lib/matplotlib/figure.py** | 2114 | 2210| 788 | 52387 | 149768 | 
| 115 | 30 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 82| 738 | 53125 | 150637 | 
| 116 | **30 tutorials/intermediate/legend_guide.py** | 286 | 305| 193 | 53318 | 150637 | 
| 117 | 30 lib/matplotlib/pyplot.py | 776 | 842| 679 | 53997 | 150637 | 
| 118 | 30 lib/matplotlib/legend_handler.py | 363 | 381| 205 | 54202 | 150637 | 
| 119 | **30 lib/matplotlib/figure.py** | 2236 | 2262| 312 | 54514 | 150637 | 
| 120 | 30 lib/matplotlib/backends/qt_editor/figureoptions.py | 177 | 264| 735 | 55249 | 150637 | 
| 121 | 31 examples/event_handling/legend_picking.py | 1 | 52| 401 | 55650 | 151038 | 
| 122 | 31 lib/matplotlib/backends/backend_gtk4.py | 206 | 219| 139 | 55789 | 151038 | 
| 123 | 32 lib/matplotlib/backends/backend_gtk3.py | 207 | 241| 351 | 56140 | 156034 | 
| 124 | **32 lib/matplotlib/figure.py** | 2306 | 2327| 149 | 56289 | 156034 | 
| 125 | **32 lib/matplotlib/legend.py** | 646 | 661| 189 | 56478 | 156034 | 
| 126 | 33 tutorials/provisional/mosaic.py | 59 | 170| 789 | 57267 | 158730 | 
| 127 | **33 lib/matplotlib/legend.py** | 1160 | 1205| 390 | 57657 | 158730 | 
| 128 | 34 tutorials/colors/colorbar_only.py | 88 | 131| 357 | 58014 | 159821 | 
| 129 | **34 lib/matplotlib/figure.py** | 2288 | 2304| 149 | 58163 | 159821 | 
| 130 | 34 lib/matplotlib/legend_handler.py | 766 | 797| 264 | 58427 | 159821 | 
| 131 | 34 lib/matplotlib/pyplot.py | 2289 | 2364| 763 | 59190 | 159821 | 
| 132 | 35 examples/subplots_axes_and_figures/custom_figure_class.py | 35 | 52| 115 | 59305 | 160192 | 
| 133 | 36 examples/misc/anchored_artists.py | 51 | 68| 211 | 59516 | 160985 | 
| 134 | 36 tutorials/colors/colorbar_only.py | 1 | 86| 734 | 60250 | 160985 | 
| 135 | 36 lib/matplotlib/backends/backend_gtk4.py | 174 | 187| 154 | 60404 | 160985 | 
| 136 | **36 lib/matplotlib/legend.py** | 863 | 896| 242 | 60646 | 160985 | 
| 137 | 36 lib/matplotlib/backends/backend_gtk3.py | 55 | 110| 524 | 61170 | 160985 | 
| 138 | 37 lib/matplotlib/collections.py | 1072 | 1144| 676 | 61846 | 178792 | 
| 139 | 37 lib/matplotlib/backends/backend_gtk3.py | 196 | 205| 121 | 61967 | 178792 | 
| 140 | **37 lib/matplotlib/_constrained_layout.py** | 300 | 335| 391 | 62358 | 178792 | 
| 141 | 38 lib/matplotlib/offsetbox.py | 1067 | 1086| 154 | 62512 | 191203 | 
| 142 | 38 examples/misc/anchored_artists.py | 1 | 48| 450 | 62962 | 191203 | 
| 143 | **38 lib/matplotlib/figure.py** | 3108 | 3138| 209 | 63171 | 191203 | 
| 144 | **38 lib/matplotlib/figure.py** | 3429 | 3462| 252 | 63423 | 191203 | 


## Missing Patch Files

 * 1: examples/text_labels_and_annotations/figlegend_demo.py
 * 2: lib/matplotlib/_constrained_layout.py
 * 3: lib/matplotlib/axes/_axes.py
 * 4: lib/matplotlib/figure.py
 * 5: lib/matplotlib/legend.py
 * 6: tutorials/intermediate/legend_guide.py

### Hint

```
What behaviour would you expect?  

If you want the legend to steal space on the figure from the axes, then call `axes.legend` with the correct handles and it will make room.  
Yes. Here's an example from seaborn. I would expect this to be the result of `figure.legend(handles, labels, loc='right')`

![image](https://user-images.githubusercontent.com/2448579/50259219-07214f00-03b8-11e9-9527-dca898d66c17.png)


How would constrained layout know which subplots to steal space from for the legend?   A figure legend doesn’t belong to any axes, so there is no natural way to do what you are asking.  

Again if you attach the legend to one of the rightmost axes, it will do what you want.  
> How would constrained layout know which subplots to steal space from for the legend? 

Do what colorbar does? colorbar does have an `ax` argument though... hmmm.

Here's an example. I'd like the `constrained_layout` version of this:

\`\`\` python
f, ax = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=False)
h = list()
for aa in ax.flat:
    h.append(aa.plot(np.random.randn(5), np.random.randn(5), '*')[0])
    h.append(aa.plot(np.random.randn(5), np.random.randn(5), '*')[0])

hleg = f.legend(handles=h[-2:], labels=['a', 'b'],
                loc='center right')
\`\`\`
![mpl-test1](https://user-images.githubusercontent.com/2448579/50300034-05976b80-0438-11e9-8808-074d7669650b.png)


Here's my attempt at a constrained_layout version using `ax.legend`. Is there a way to do this without the space between the two rows of subplots?

\`\`\` python
f, ax = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
h = list()
for aa in ax.flat:
    h.append(aa.plot(np.random.randn(5), np.random.randn(5), '*')[0])
    h.append(aa.plot(np.random.randn(5), np.random.randn(5), '*')[0])

hleg = ax[1, 1].legend(handles=h[-2:], labels=['a', 'b'],
                       loc='center right',
                       bbox_to_anchor=(1.2, 1.2))
\`\`\`
![mpl-test2](https://user-images.githubusercontent.com/2448579/50300078-2c55a200-0438-11e9-8761-19dd6d97c65a.png)

What is happening above is ax[1, 1] is saying it is bigger than the other axes, and makes space for itself in the layout.  To avoid this, don't anchor it above the top of the axes.  Yes, your legend will not be vertically centred. 

We could come up with an API to automatically insert anchored boxes and steal space from the other elements in a gridspec (the container of your four subplots).  We do that now for suptitle (sort of) and, as you say, colorbar.  So the question is how to pass that info down to `fig.legend`.  I suppose we could add an `axs` kwarg.  
Thought about this some more, and its quite a hard problem. 

`figure.legend` allows more than one legend. Conversely, `figure.suptitle` only has one suptitle, so layout is easy.  `figure.colorbar` has a well-understood set of conventions for where colorbars go, but legends have been more free-form, with placement being arbitrary, and anchored using `bbox_to_anchor`.  


Is it feasible to only add support for certain legend configurations in a first step? E.g. don‘t handle legends with `bbox_to_anchor` and/or certain `loc` values.
I was going to say the same. The `loc` values should be well constrained and a good first step. Throw a warning and disable constrained_layout if `bbox_to_anchor` is set?
Hmmmm.  Not a fan of that.  Seems mysterious and a documentation PITA 

I was leaning towards a new method, ideally  attached to gridspec. That’s easier from the point of view of being able to specify a new API.  

I think conceptually this is easy enough to implement in constrained_layout.  But I’m not sure about implementing without constrained_layout.  Legends are packable boxes and hence hard to predict their size a priori.  For instance they can be arbitrarily wide or high.  

I’d propose writing a separate method for now and then figure.legend could call that given the right conditions. 
> I’d propose writing a separate method for now and then figure.legend could call that given the right conditions.

Sounds like a reasonable approach. Though, I'm not quite involved in this stuff, and therefore I'm not a good judge. Withdrawing from the discussion for now.
I have a proof of concept that works great for this.  Up for ideas about how to specify locations and whether the arrangement with the rest of the gridspec is horizontal or vertical.   The typical location “northeast” or “southeast” is ambiguous because it doesn’t specify if the gridspec’s Axes give way vertically or horizontally.  We’d could either have another kwarg, or let the `location=“verticalnortheast”` to specify vertical arrangement instead of horizontal (which I think would be the default for all locations except for “north” and “south”)...

Suggestions welcome.   For now, I think I’ll go w/ implementing horizontal arrangement, and just not support “north” or “south” (I.e. “northeast” will arrange horizontally, with the legend in the upper right corner). 
Using an extra kwarg `arrange=vertical` or `arrange=horizontal` seems less clunky than `verticalnortheast`
```

## Patch

```diff
diff --git a/examples/text_labels_and_annotations/figlegend_demo.py b/examples/text_labels_and_annotations/figlegend_demo.py
--- a/examples/text_labels_and_annotations/figlegend_demo.py
+++ b/examples/text_labels_and_annotations/figlegend_demo.py
@@ -28,3 +28,26 @@
 
 plt.tight_layout()
 plt.show()
+
+##############################################################################
+# Sometimes we do not want the legend to overlap the axes.  If you use
+# constrained_layout you can specify "outside right upper", and
+# constrained_layout will make room for the legend.
+
+fig, axs = plt.subplots(1, 2, layout='constrained')
+
+x = np.arange(0.0, 2.0, 0.02)
+y1 = np.sin(2 * np.pi * x)
+y2 = np.exp(-x)
+l1, = axs[0].plot(x, y1)
+l2, = axs[0].plot(x, y2, marker='o')
+
+y3 = np.sin(4 * np.pi * x)
+y4 = np.exp(-2 * x)
+l3, = axs[1].plot(x, y3, color='tab:green')
+l4, = axs[1].plot(x, y4, color='tab:red', marker='^')
+
+fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')
+fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')
+
+plt.show()
diff --git a/lib/matplotlib/_constrained_layout.py b/lib/matplotlib/_constrained_layout.py
--- a/lib/matplotlib/_constrained_layout.py
+++ b/lib/matplotlib/_constrained_layout.py
@@ -418,6 +418,25 @@ def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
         # pass the new margins down to the layout grid for the solution...
         layoutgrids[gs].edit_outer_margin_mins(margin, ss)
 
+    # make margins for figure-level legends:
+    for leg in fig.legends:
+        inv_trans_fig = None
+        if leg._outside_loc and leg._bbox_to_anchor is None:
+            if inv_trans_fig is None:
+                inv_trans_fig = fig.transFigure.inverted().transform_bbox
+            bbox = inv_trans_fig(leg.get_tightbbox(renderer))
+            w = bbox.width + 2 * w_pad
+            h = bbox.height + 2 * h_pad
+            legendloc = leg._outside_loc
+            if legendloc == 'lower':
+                layoutgrids[fig].edit_margin_min('bottom', h)
+            elif legendloc == 'upper':
+                layoutgrids[fig].edit_margin_min('top', h)
+            if legendloc == 'right':
+                layoutgrids[fig].edit_margin_min('right', w)
+            elif legendloc == 'left':
+                layoutgrids[fig].edit_margin_min('left', w)
+
 
 def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
     # Figure out how large the suptitle is and make the
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -294,7 +294,7 @@ def legend(self, *args, **kwargs):
 
         Other Parameters
         ----------------
-        %(_legend_kw_doc)s
+        %(_legend_kw_axes)s
 
         See Also
         --------
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -1085,7 +1085,8 @@ def legend(self, *args, **kwargs):
 
         Other Parameters
         ----------------
-        %(_legend_kw_doc)s
+        %(_legend_kw_figure)s
+
 
         See Also
         --------
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -94,51 +94,7 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
         self.legend.set_bbox_to_anchor(loc_in_bbox)
 
 
-_docstring.interpd.update(_legend_kw_doc="""
-loc : str or pair of floats, default: :rc:`legend.loc` ('best' for axes, \
-'upper right' for figures)
-    The location of the legend.
-
-    The strings
-    ``'upper left', 'upper right', 'lower left', 'lower right'``
-    place the legend at the corresponding corner of the axes/figure.
-
-    The strings
-    ``'upper center', 'lower center', 'center left', 'center right'``
-    place the legend at the center of the corresponding edge of the
-    axes/figure.
-
-    The string ``'center'`` places the legend at the center of the axes/figure.
-
-    The string ``'best'`` places the legend at the location, among the nine
-    locations defined so far, with the minimum overlap with other drawn
-    artists.  This option can be quite slow for plots with large amounts of
-    data; your plotting speed may benefit from providing a specific location.
-
-    The location can also be a 2-tuple giving the coordinates of the lower-left
-    corner of the legend in axes coordinates (in which case *bbox_to_anchor*
-    will be ignored).
-
-    For back-compatibility, ``'center right'`` (but no other location) can also
-    be spelled ``'right'``, and each "string" locations can also be given as a
-    numeric value:
-
-        ===============   =============
-        Location String   Location Code
-        ===============   =============
-        'best'            0
-        'upper right'     1
-        'upper left'      2
-        'lower left'      3
-        'lower right'     4
-        'right'           5
-        'center left'     6
-        'center right'    7
-        'lower center'    8
-        'upper center'    9
-        'center'          10
-        ===============   =============
-
+_legend_kw_doc_base = """
 bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
     Box that is used to position the legend in conjunction with *loc*.
     Defaults to `axes.bbox` (if called as a method to `.Axes.legend`) or
@@ -295,7 +251,79 @@ def _update_bbox_to_anchor(self, loc_in_canvas):
 
 draggable : bool, default: False
     Whether the legend can be dragged with the mouse.
-""")
+"""
+
+_loc_doc_base = """
+loc : str or pair of floats, {0}
+    The location of the legend.
+
+    The strings
+    ``'upper left', 'upper right', 'lower left', 'lower right'``
+    place the legend at the corresponding corner of the axes/figure.
+
+    The strings
+    ``'upper center', 'lower center', 'center left', 'center right'``
+    place the legend at the center of the corresponding edge of the
+    axes/figure.
+
+    The string ``'center'`` places the legend at the center of the axes/figure.
+
+    The string ``'best'`` places the legend at the location, among the nine
+    locations defined so far, with the minimum overlap with other drawn
+    artists.  This option can be quite slow for plots with large amounts of
+    data; your plotting speed may benefit from providing a specific location.
+
+    The location can also be a 2-tuple giving the coordinates of the lower-left
+    corner of the legend in axes coordinates (in which case *bbox_to_anchor*
+    will be ignored).
+
+    For back-compatibility, ``'center right'`` (but no other location) can also
+    be spelled ``'right'``, and each "string" locations can also be given as a
+    numeric value:
+
+        ===============   =============
+        Location String   Location Code
+        ===============   =============
+        'best'            0
+        'upper right'     1
+        'upper left'      2
+        'lower left'      3
+        'lower right'     4
+        'right'           5
+        'center left'     6
+        'center right'    7
+        'lower center'    8
+        'upper center'    9
+        'center'          10
+        ===============   =============
+    {1}"""
+
+_legend_kw_axes_st = (_loc_doc_base.format("default: :rc:`legend.loc`", '') +
+                      _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_axes=_legend_kw_axes_st)
+
+_outside_doc = """
+    If a figure is using the constrained layout manager, the string codes
+    of the *loc* keyword argument can get better layout behaviour using the
+    prefix 'outside'. There is ambiguity at the corners, so 'outside
+    upper right' will make space for the legend above the rest of the
+    axes in the layout, and 'outside right upper' will make space on the
+    right side of the layout.  In addition to the values of *loc*
+    listed above, we have 'outside right upper', 'outside right lower',
+    'outside left upper', and 'outside left lower'.  See
+    :doc:`/tutorials/intermediate/legend_guide` for more details.
+"""
+
+_legend_kw_figure_st = (_loc_doc_base.format("default: 'upper right'",
+                                             _outside_doc) +
+                        _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_figure=_legend_kw_figure_st)
+
+_legend_kw_both_st = (
+    _loc_doc_base.format("default: 'best' for axes, 'upper right' for figures",
+                         _outside_doc) +
+    _legend_kw_doc_base)
+_docstring.interpd.update(_legend_kw_doc=_legend_kw_both_st)
 
 
 class Legend(Artist):
@@ -482,13 +510,37 @@ def val_or_rc(val, rc_name):
             )
         self.parent = parent
 
+        loc0 = loc
         self._loc_used_default = loc is None
         if loc is None:
             loc = mpl.rcParams["legend.loc"]
             if not self.isaxes and loc in [0, 'best']:
                 loc = 'upper right'
+
+        # handle outside legends:
+        self._outside_loc = None
         if isinstance(loc, str):
+            if loc.split()[0] == 'outside':
+                # strip outside:
+                loc = loc.split('outside ')[1]
+                # strip "center" at the beginning
+                self._outside_loc = loc.replace('center ', '')
+                # strip first
+                self._outside_loc = self._outside_loc.split()[0]
+                locs = loc.split()
+                if len(locs) > 1 and locs[0] in ('right', 'left'):
+                    # locs doesn't accept "left upper", etc, so swap
+                    if locs[0] != 'center':
+                        locs = locs[::-1]
+                    loc = locs[0] + ' ' + locs[1]
+            # check that loc is in acceptable strings
             loc = _api.check_getitem(self.codes, loc=loc)
+
+        if self.isaxes and self._outside_loc:
+            raise ValueError(
+                f"'outside' option for loc='{loc0}' keyword argument only "
+                "works for figure legends")
+
         if not self.isaxes and loc == 0:
             raise ValueError(
                 "Automatic legend placement (loc='best') not implemented for "
diff --git a/tutorials/intermediate/legend_guide.py b/tutorials/intermediate/legend_guide.py
--- a/tutorials/intermediate/legend_guide.py
+++ b/tutorials/intermediate/legend_guide.py
@@ -135,7 +135,54 @@
 ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1),
                          loc='upper left', borderaxespad=0.)
 
-plt.show()
+##############################################################################
+# Figure legends
+# --------------
+#
+# Sometimes it makes more sense to place a legend relative to the (sub)figure
+# rather than individual Axes.  By using ``constrained_layout`` and
+# specifying "outside" at the beginning of the *loc* keyword argument,
+# the legend is drawn outside the Axes on the (sub)figure.
+
+fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
+
+axs['left'].plot([1, 2, 3], label="test1")
+axs['left'].plot([3, 2, 1], label="test2")
+
+axs['right'].plot([1, 2, 3], 'C2', label="test3")
+axs['right'].plot([3, 2, 1], 'C3', label="test4")
+# Place a legend to the right of this smaller subplot.
+fig.legend(loc='outside upper right')
+
+##############################################################################
+# This accepts a slightly different grammar than the normal *loc* keyword,
+# where "outside right upper" is different from "outside upper right".
+#
+ucl = ['upper', 'center', 'lower']
+lcr = ['left', 'center', 'right']
+fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
+
+ax.plot([1, 2], [1, 2], label='TEST')
+# Place a legend to the right of this smaller subplot.
+for loc in [
+        'outside upper left',
+        'outside upper center',
+        'outside upper right',
+        'outside lower left',
+        'outside lower center',
+        'outside lower right']:
+    fig.legend(loc=loc, title=loc)
+
+fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
+ax.plot([1, 2], [1, 2], label='test')
+
+for loc in [
+        'outside left upper',
+        'outside right upper',
+        'outside left lower',
+        'outside right lower']:
+    fig.legend(loc=loc, title=loc)
+
 
 ###############################################################################
 # Multiple legends on the same Axes

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_legend.py b/lib/matplotlib/tests/test_legend.py
--- a/lib/matplotlib/tests/test_legend.py
+++ b/lib/matplotlib/tests/test_legend.py
@@ -4,6 +4,7 @@
 import warnings
 
 import numpy as np
+from numpy.testing import assert_allclose
 import pytest
 
 from matplotlib.testing.decorators import check_figures_equal, image_comparison
@@ -18,7 +19,6 @@
 import matplotlib.legend as mlegend
 from matplotlib import rc_context
 from matplotlib.font_manager import FontProperties
-from numpy.testing import assert_allclose
 
 
 def test_legend_ordereddict():
@@ -486,6 +486,47 @@ def test_warn_args_kwargs(self):
             "be discarded.")
 
 
+def test_figure_legend_outside():
+    todos = ['upper ' + pos for pos in ['left', 'center', 'right']]
+    todos += ['lower ' + pos for pos in ['left', 'center', 'right']]
+    todos += ['left ' + pos for pos in ['lower', 'center', 'upper']]
+    todos += ['right ' + pos for pos in ['lower', 'center', 'upper']]
+
+    upperext = [20.347556,  27.722556, 790.583, 545.499]
+    lowerext = [20.347556,  71.056556, 790.583, 588.833]
+    leftext = [151.681556, 27.722556, 790.583, 588.833]
+    rightext = [20.347556,  27.722556, 659.249, 588.833]
+    axbb = [upperext, upperext, upperext,
+            lowerext, lowerext, lowerext,
+            leftext, leftext, leftext,
+            rightext, rightext, rightext]
+
+    legbb = [[10., 555., 133., 590.],     # upper left
+             [338.5, 555., 461.5, 590.],  # upper center
+             [667, 555., 790.,  590.],    # upper right
+             [10., 10., 133.,  45.],      # lower left
+             [338.5, 10., 461.5,  45.],   # lower center
+             [667., 10., 790.,  45.],     # lower right
+             [10., 10., 133., 45.],       # left lower
+             [10., 282.5, 133., 317.5],   # left center
+             [10., 555., 133., 590.],     # left upper
+             [667, 10., 790., 45.],       # right lower
+             [667., 282.5, 790., 317.5],  # right center
+             [667., 555., 790., 590.]]    # right upper
+
+    for nn, todo in enumerate(todos):
+        print(todo)
+        fig, axs = plt.subplots(constrained_layout=True, dpi=100)
+        axs.plot(range(10), label='Boo1')
+        leg = fig.legend(loc='outside ' + todo)
+        fig.draw_without_rendering()
+
+        assert_allclose(axs.get_window_extent().extents,
+                        axbb[nn])
+        assert_allclose(leg.get_window_extent().extents,
+                        legbb[nn])
+
+
 @image_comparison(['legend_stackplot.png'])
 def test_legend_stackplot():
     """Test legend for PolyCollection using stackplot."""

```


## Code snippets

### 1 - tutorials/intermediate/constrainedlayout_guide.py:

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
### 2 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 1, End line: 112

```python
"""
================================
Constrained Layout Guide
================================

How to use constrained-layout to fit plots within your figure cleanly.

*constrained_layout* automatically adjusts subplots and decorations like
legends and colorbars so that they fit in the figure window while still
preserving, as best they can, the logical layout requested by the user.

*constrained_layout* is similar to
:doc:`tight_layout</tutorials/intermediate/tight_layout_guide>`,
but uses a constraint solver to determine the size of axes that allows
them to fit.

*constrained_layout* typically needs to be activated before any axes are
added to a figure. Two ways of doing so are

* using the respective argument to :func:`~.pyplot.subplots` or
  :func:`~.pyplot.figure`, e.g.::

      plt.subplots(layout="constrained")

* activate it via :ref:`rcParams<customizing-with-dynamic-rc-settings>`,
  like::

      plt.rcParams['figure.constrained_layout.use'] = True

Those are described in detail throughout the following sections.

Simple Example
==============

In Matplotlib, the location of axes (including subplots) are specified in
normalized figure coordinates. It can happen that your axis labels or
titles (or sometimes even ticklabels) go outside the figure area, and are thus
clipped.
"""

# sphinx_gallery_thumbnail_number = 18


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams['figure.figsize'] = 4.5, 4.
plt.rcParams['figure.max_open_warning'] = 50


def example_plot(ax, fontsize=12, hide_labels=False):
    ax.plot([1, 2])

    ax.locator_params(nbins=3)
    if hide_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots(layout=None)
example_plot(ax, fontsize=24)

###############################################################################
# To prevent this, the location of axes needs to be adjusted. For
# subplots, this can be done manually by adjusting the subplot parameters
# using `.Figure.subplots_adjust`. However, specifying your figure with the
# # ``layout="constrained"`` keyword argument will do the adjusting
# # automatically.

fig, ax = plt.subplots(layout="constrained")
example_plot(ax, fontsize=24)

###############################################################################
# When you have multiple subplots, often you see labels of different
# axes overlapping each other.

fig, axs = plt.subplots(2, 2, layout=None)
for ax in axs.flat:
    example_plot(ax)

###############################################################################
# Specifying ``layout="constrained"`` in the call to ``plt.subplots``
# causes the layout to be properly constrained.

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    example_plot(ax)

###############################################################################
# Colorbars
# =========
#
# If you create a colorbar with `.Figure.colorbar`,
# you need to make room for it.  ``constrained_layout`` does this
# automatically.  Note that if you specify ``use_gridspec=True`` it will be
# ignored because this option is made for improving the layout via
# ``tight_layout``.
#
# .. note::
#
#   For the `~.axes.Axes.pcolormesh` keyword arguments (``pc_kwargs``) we use a
#   dictionary. Below we will assign one colorbar to a number of axes each
#   containing a `~.cm.ScalarMappable`; specifying the norm and colormap
#   ensures the colorbar is accurate for all the axes.

arr = np.arange(100).reshape((10, 10))
```
### 3 - tutorials/intermediate/constrainedlayout_guide.py:

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
### 5 - examples/subplots_axes_and_figures/demo_constrained_layout.py:

Start line: 1, End line: 72

```python
"""
=====================================
Resizing axes with constrained layout
=====================================

Constrained layout attempts to resize subplots in
a figure so that there are no overlaps between axes objects and labels
on the axes.

See :doc:`/tutorials/intermediate/constrainedlayout_guide` for more details and
:doc:`/tutorials/intermediate/tight_layout_guide` for an alternative.

"""

import matplotlib.pyplot as plt


def example_plot(ax):
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=12)
    ax.set_ylabel('y-label', fontsize=12)
    ax.set_title('Title', fontsize=14)


###############################################################################
# If we don't use constrained_layout, then labels overlap the axes

fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=False)

for ax in axs.flat:
    example_plot(ax)

###############################################################################
# adding ``constrained_layout=True`` automatically adjusts.

fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

for ax in axs.flat:
    example_plot(ax)

###############################################################################
# Below is a more complicated example using nested gridspecs.

fig = plt.figure(constrained_layout=True)

import matplotlib.gridspec as gridspec

gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])
for n in range(3):
    ax = fig.add_subplot(gs1[n])
    example_plot(ax)


gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])
for n in range(2):
    ax = fig.add_subplot(gs2[n])
    example_plot(ax)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.gridspec.GridSpec`
#    - `matplotlib.gridspec.GridSpecFromSubplotSpec`
```
### 6 - lib/matplotlib/figure.py:

Start line: 2781, End line: 2809

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.deprecated("3.6", alternative="set_layout_engine('constrained')",
                     pending=True)
    def set_constrained_layout(self, constrained):
        """
        [*Discouraged*] Set whether ``constrained_layout`` is used upon
        drawing.

        If None, :rc:`figure.constrained_layout.use` value will be used.

        When providing a dict containing the keys ``w_pad``, ``h_pad``
        the default ``constrained_layout`` paddings will be
        overridden.  These pads are in inches and default to 3.0/72.0.
        ``w_pad`` is the width padding and ``h_pad`` is the height padding.

        .. admonition:: Discouraged

            This method is discouraged in favor of `~.set_layout_engine`.

        Parameters
        ----------
        constrained : bool or dict or None
        """
        if constrained is None:
            constrained = mpl.rcParams['figure.constrained_layout.use']
        _constrained = bool(constrained)
        _parameters = constrained if isinstance(constrained, dict) else {}
        if _constrained:
            self.set_layout_engine(ConstrainedLayoutEngine(**_parameters))
        self.stale = True
```
### 7 - lib/matplotlib/layout_engine.py:

Start line: 237, End line: 254

```python
class ConstrainedLayoutEngine(LayoutEngine):

    def execute(self, fig):
        """
        Perform constrained_layout and move and resize axes accordingly.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.
        """
        width, height = fig.get_size_inches()
        # pads are relative to the current state of the figure...
        w_pad = self._params['w_pad'] / width
        h_pad = self._params['h_pad'] / height

        return do_constrained_layout(fig, w_pad=w_pad, h_pad=h_pad,
                                     wspace=self._params['wspace'],
                                     hspace=self._params['hspace'],
                                     rect=self._params['rect'],
                                     compress=self._compress)
```
### 8 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 354, End line: 441

```python
fig = plt.figure(layout="constrained")

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

fig = plt.figure(figsize=(4, 6), layout="constrained")

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

############################################################################
# This example uses two gridspecs to have the colorbar only pertain to
# one set of pcolors.  Note how the left column is wider than the
# two right-hand columns because of this.  Of course, if you wanted the
# subplots to be the same size you only needed one gridspec.  Note that
# the same effect can be achieved using `~.Figure.subfigures`.

fig = plt.figure(layout="constrained")
gs0 = fig.add_gridspec(1, 2, figure=fig, width_ratios=[1, 2])
gs_left = gs0[0].subgridspec(2, 1)
gs_right = gs0[1].subgridspec(2, 2)

for gs in gs_left:
    ax = fig.add_subplot(gs)
    example_plot(ax)
axs = []
for gs in gs_right:
    ax = fig.add_subplot(gs)
    pcm = ax.pcolormesh(arr, **pc_kwargs)
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_title('title')
    axs += [ax]
fig.suptitle('Nested plots using subgridspec')
fig.colorbar(pcm, ax=axs)

###############################################################################
# Rather than using subgridspecs, Matplotlib now provides `~.Figure.subfigures`
# which also work with ``constrained_layout``:

fig = plt.figure(layout="constrained")
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
```
### 9 - lib/matplotlib/figure.py:

Start line: 2264, End line: 2286

```python
@_docstring.interpd
class SubFigure(FigureBase):

    def get_constrained_layout(self):
        """
        Return whether constrained layout is being used.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.
        """
        return self._parent.get_constrained_layout()

    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        return self._parent.get_constrained_layout_pads(relative=relative)
```
### 10 - tutorials/intermediate/constrainedlayout_guide.py:

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
### 13 - lib/matplotlib/_constrained_layout.py:

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
### 14 - lib/matplotlib/_constrained_layout.py:

Start line: 62, End line: 109

```python
######################################################
def do_constrained_layout(fig, h_pad, w_pad,
                          hspace=None, wspace=None, rect=(0, 0, 1, 1),
                          compress=False):
    """
    Do the constrained_layout.  Called at draw time in
     ``figure.constrained_layout()``

    Parameters
    ----------
    fig : Figure
        ``Figure`` instance to do the layout in.

    renderer : Renderer
        Renderer to use.

    h_pad, w_pad : float
      Padding around the axes elements in figure-normalized units.

    hspace, wspace : float
       Fraction of the figure to dedicate to space between the
       axes.  These are evenly spread between the gaps between the axes.
       A value of 0.2 for a three-column layout would have a space
       of 0.1 of the figure width between each column.
       If h/wspace < h/w_pad, then the pads are used instead.

    rect : tuple of 4 floats
        Rectangle in figure coordinates to perform constrained layout in
        [left, bottom, width, height], each from 0-1.

    compress : bool
        Whether to shift Axes so that white space in between them is
        removed. This is useful for simple grids of fixed-aspect Axes (e.g.
        a grid of images).

    Returns
    -------
    layoutgrid : private debugging structure
    """

    renderer = fig._get_renderer()
    # make layoutgrid tree...
    layoutgrids = make_layoutgrids(fig, None, rect=rect)
    if not layoutgrids['hasgrids']:
        _api.warn_external('There are no gridspecs with layoutgrids. '
                           'Possibly did not call parent GridSpec with the'
                           ' "figure" keyword')
        return
    # ... other code
```
### 15 - lib/matplotlib/figure.py:

Start line: 2747, End line: 2779

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.deprecated("3.6", alternative="set_layout_engine",
                     pending=True)
    def set_tight_layout(self, tight):
        """
        [*Discouraged*] Set whether and how `.tight_layout` is called when
        drawing.

        .. admonition:: Discouraged

            This method is discouraged in favor of `~.set_layout_engine`.

        Parameters
        ----------
        tight : bool or dict with keys "pad", "w_pad", "h_pad", "rect" or None
            If a bool, sets whether to call `.tight_layout` upon drawing.
            If ``None``, use :rc:`figure.autolayout` instead.
            If a dict, pass it as kwargs to `.tight_layout`, overriding the
            default paddings.
        """
        if tight is None:
            tight = mpl.rcParams['figure.autolayout']
        _tight_parameters = tight if isinstance(tight, dict) else {}
        if bool(tight):
            self.set_layout_engine(TightLayoutEngine(**_tight_parameters))
        self.stale = True

    def get_constrained_layout(self):
        """
        Return whether constrained layout is being used.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.
        """
        return isinstance(self.get_layout_engine(), ConstrainedLayoutEngine)
```
### 16 - lib/matplotlib/_constrained_layout.py:

Start line: 1, End line: 59

```python
"""
Adjust subplot layouts so that there are no overlapping axes or axes
decorations.  All axes decorations are dealt with (labels, ticks, titles,
ticklabels) and some dependent artists are also dealt with (colorbar,
suptitle).

Layout is done via `~matplotlib.gridspec`, with one constraint per gridspec,
so it is possible to have overlapping axes if the gridspecs overlap (i.e.
using `~matplotlib.gridspec.GridSpecFromSubplotSpec`).  Axes placed using
``figure.subplots()`` or ``figure.add_subplots()`` will participate in the
layout.  Axes manually placed via ``figure.add_axes()`` will not.

See Tutorial: :doc:`/tutorials/intermediate/constrainedlayout_guide`

General idea:
-------------

First, a figure has a gridspec that divides the figure into nrows and ncols,
with heights and widths set by ``height_ratios`` and ``width_ratios``,
often just set to 1 for an equal grid.

Subplotspecs that are derived from this gridspec can contain either a
``SubPanel``, a ``GridSpecFromSubplotSpec``, or an ``Axes``.  The ``SubPanel``
and ``GridSpecFromSubplotSpec`` are dealt with recursively and each contain an
analogous layout.

Each ``GridSpec`` has a ``_layoutgrid`` attached to it.  The ``_layoutgrid``
has the same logical layout as the ``GridSpec``.   Each row of the grid spec
has a top and bottom "margin" and each column has a left and right "margin".
The "inner" height of each row is constrained to be the same (or as modified
by ``height_ratio``), and the "inner" width of each column is
constrained to be the same (as modified by ``width_ratio``), where "inner"
is the width or height of each column/row minus the size of the margins.

Then the size of the margins for each row and column are determined as the
max width of the decorators on each axes that has decorators in that margin.
For instance, a normal axes would have a left margin that includes the
left ticklabels, and the ylabel if it exists.  The right margin may include a
colorbar, the bottom margin the xaxis decorations, and the top margin the
title.

With these constraints, the solver then finds appropriate bounds for the
columns and rows.  It's possible that the margins take up the whole figure,
in which case the algorithm is not applied and a warning is raised.

See the tutorial doc:`/tutorials/intermediate/constrainedlayout_guide`
for more discussion of the algorithm with examples.
"""

import logging

import numpy as np

from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid


_log = logging.getLogger(__name__)
```
### 17 - lib/matplotlib/_constrained_layout.py:

Start line: 422, End line: 458

```python
def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
    # Figure out how large the suptitle is and make the
    # top level figure margin larger.

    inv_trans_fig = fig.transFigure.inverted().transform_bbox
    # get the h_pad and w_pad as distances in the local subfigure coordinates:
    padbox = mtransforms.Bbox([[0, 0], [w_pad, h_pad]])
    padbox = (fig.transFigure -
                   fig.transSubfigure).transform_bbox(padbox)
    h_pad_local = padbox.height
    w_pad_local = padbox.width

    for sfig in fig.subfigs:
        make_margin_suptitles(layoutgrids, sfig, renderer,
                              w_pad=w_pad, h_pad=h_pad)

    if fig._suptitle is not None and fig._suptitle.get_in_layout():
        p = fig._suptitle.get_position()
        if getattr(fig._suptitle, '_autopos', False):
            fig._suptitle.set_position((p[0], 1 - h_pad_local))
            bbox = inv_trans_fig(fig._suptitle.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('top', bbox.height + 2 * h_pad)

    if fig._supxlabel is not None and fig._supxlabel.get_in_layout():
        p = fig._supxlabel.get_position()
        if getattr(fig._supxlabel, '_autopos', False):
            fig._supxlabel.set_position((p[0], h_pad_local))
            bbox = inv_trans_fig(fig._supxlabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('bottom',
                                             bbox.height + 2 * h_pad)

    if fig._supylabel is not None and fig._supylabel.get_in_layout():
        p = fig._supylabel.get_position()
        if getattr(fig._supylabel, '_autopos', False):
            fig._supylabel.set_position((w_pad_local, p[1]))
            bbox = inv_trans_fig(fig._supylabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('left', bbox.width + 2 * w_pad)
```
### 18 - lib/matplotlib/figure.py:

Start line: 989, End line: 1098

```python
class FigureBase(Artist):
    @_docstring.dedent_interpd
    def legend(self, *args, **kwargs):
        """
        Place a legend on the figure.

        Call signatures::

            legend()
            legend(handles, labels)
            legend(handles=handles)
            legend(labels)

        The call signatures correspond to the following different ways to use
        this method:

        **1. Automatic detection of elements to be shown in the legend**

        The elements to be added to the legend are automatically determined,
        when you do not pass in any extra arguments.

        In this case, the labels are taken from the artist. You can specify
        them either at artist creation or by calling the
        :meth:`~.Artist.set_label` method on the artist::

            ax.plot([1, 2, 3], label='Inline label')
            fig.legend()

        or::

            line, = ax.plot([1, 2, 3])
            line.set_label('Label via method')
            fig.legend()

        Specific lines can be excluded from the automatic legend element
        selection by defining a label starting with an underscore.
        This is default for all artists, so calling `.Figure.legend` without
        any arguments and without setting the labels manually will result in
        no legend being drawn.


        **2. Explicitly listing the artists and labels in the legend**

        For full control of which artists have a legend entry, it is possible
        to pass an iterable of legend artists followed by an iterable of
        legend labels respectively::

            fig.legend([line1, line2, line3], ['label1', 'label2', 'label3'])


        **3. Explicitly listing the artists in the legend**

        This is similar to 2, but the labels are taken from the artists'
        label properties. Example::

            line1, = ax1.plot([1, 2, 3], label='label1')
            line2, = ax2.plot([1, 2, 3], label='label2')
            fig.legend(handles=[line1, line2])


        **4. Labeling existing plot elements**

        .. admonition:: Discouraged

            This call signature is discouraged, because the relation between
            plot elements and labels is only implicit by their order and can
            easily be mixed up.

        To make a legend for all artists on all Axes, call this function with
        an iterable of strings, one for each legend item. For example::

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot([1, 3, 5], color='blue')
            ax2.plot([2, 4, 6], color='red')
            fig.legend(['the blues', 'the reds'])


        Parameters
        ----------
        handles : list of `.Artist`, optional
            A list of Artists (lines, patches) to be added to the legend.
            Use this together with *labels*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

            The length of handles and labels should be the same in this
            case. If they are not, they are truncated to the smaller length.

        labels : list of str, optional
            A list of labels to show next to the artists.
            Use this together with *handles*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

        Returns
        -------
        `~matplotlib.legend.Legend`

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        See Also
        --------
        .Axes.legend

        Notes
        -----
        Some artists are not supported by this function.  See
        :doc:`/tutorials/intermediate/legend_guide` for details.
        """
        # ... other code
```
### 19 - lib/matplotlib/figure.py:

Start line: 1100, End line: 1122

```python
class FigureBase(Artist):
    @_docstring.dedent_interpd
    def legend(self, *args, **kwargs):

        handles, labels, extra_args, kwargs = mlegend._parse_legend_args(
                self.axes,
                *args,
                **kwargs)
        # check for third arg
        if len(extra_args):
            # _api.warn_deprecated(
            #     "2.1",
            #     message="Figure.legend will accept no more than two "
            #     "positional arguments in the future.  Use "
            #     "'fig.legend(handles, labels, loc=location)' "
            #     "instead.")
            # kwargs['loc'] = extra_args[0]
            # extra_args = extra_args[1:]
            pass
        transform = kwargs.pop('bbox_transform', self.transSubfigure)
        # explicitly set the bbox transform if the user hasn't.
        l = mlegend.Legend(self, handles, labels, *extra_args,
                           bbox_transform=transform, **kwargs)
        self.legends.append(l)
        l._remove_method = self.legends.remove
        self.stale = True
        return l
```
### 20 - lib/matplotlib/_constrained_layout.py:

Start line: 357, End line: 419

```python
def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
                        hspace=0, wspace=0):
    # ... other code

    for ax in fig._localaxes:
        if not ax.get_subplotspec() or not ax.get_in_layout():
            continue

        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()

        if gs not in layoutgrids:
            return

        margin = get_margin_from_padding(ax, w_pad=w_pad, h_pad=h_pad,
                                         hspace=hspace, wspace=wspace)
        pos, bbox = get_pos_and_bbox(ax, renderer)
        # the margin is the distance between the bounding box of the axes
        # and its position (plus the padding from above)
        margin['left'] += pos.x0 - bbox.x0
        margin['right'] += bbox.x1 - pos.x1
        # remember that rows are ordered from top:
        margin['bottom'] += pos.y0 - bbox.y0
        margin['top'] += bbox.y1 - pos.y1

        # make margin for colorbars.  These margins go in the
        # padding margin, versus the margin for axes decorators.
        for cbax in ax._colorbars:
            # note pad is a fraction of the parent width...
            pad = colorbar_get_pad(layoutgrids, cbax)
            # colorbars can be child of more than one subplot spec:
            cbp_rspan, cbp_cspan = get_cb_parent_spans(cbax)
            loc = cbax._colorbar_info['location']
            cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)
            if loc == 'right':
                if cbp_cspan.stop == ss.colspan.stop:
                    # only increase if the colorbar is on the right edge
                    margin['rightcb'] += cbbbox.width + pad
            elif loc == 'left':
                if cbp_cspan.start == ss.colspan.start:
                    # only increase if the colorbar is on the left edge
                    margin['leftcb'] += cbbbox.width + pad
            elif loc == 'top':
                if cbp_rspan.start == ss.rowspan.start:
                    margin['topcb'] += cbbbox.height + pad
            else:
                if cbp_rspan.stop == ss.rowspan.stop:
                    margin['bottomcb'] += cbbbox.height + pad
            # If the colorbars are wider than the parent box in the
            # cross direction
            if loc in ['top', 'bottom']:
                if (cbp_cspan.start == ss.colspan.start and
                        cbbbox.x0 < bbox.x0):
                    margin['left'] += bbox.x0 - cbbbox.x0
                if (cbp_cspan.stop == ss.colspan.stop and
                        cbbbox.x1 > bbox.x1):
                    margin['right'] += cbbbox.x1 - bbox.x1
            # or taller:
            if loc in ['left', 'right']:
                if (cbp_rspan.stop == ss.rowspan.stop and
                        cbbbox.y0 < bbox.y0):
                    margin['bottom'] += bbox.y0 - cbbbox.y0
                if (cbp_rspan.start == ss.rowspan.start and
                        cbbbox.y1 > bbox.y1):
                    margin['top'] += cbbbox.y1 - bbox.y1
        # pass the new margins down to the layout grid for the solution...
        layoutgrids[gs].edit_outer_margin_mins(margin, ss)
```
### 21 - lib/matplotlib/figure.py:

Start line: 2811, End line: 2843

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

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

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
### 25 - lib/matplotlib/_constrained_layout.py:

Start line: 492, End line: 555

```python
def match_submerged_margins(layoutgrids, fig):
    # ... other code

    for ax1 in axs:
        ss1 = ax1.get_subplotspec()
        if ss1.get_gridspec() not in layoutgrids:
            axs.remove(ax1)
            continue
        lg1 = layoutgrids[ss1.get_gridspec()]

        # interior columns:
        if len(ss1.colspan) > 1:
            maxsubl = np.max(
                lg1.margin_vals['left'][ss1.colspan[1:]] +
                lg1.margin_vals['leftcb'][ss1.colspan[1:]]
            )
            maxsubr = np.max(
                lg1.margin_vals['right'][ss1.colspan[:-1]] +
                lg1.margin_vals['rightcb'][ss1.colspan[:-1]]
            )
            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None and len(ss2.colspan) > 1:
                    maxsubl2 = np.max(
                        lg2.margin_vals['left'][ss2.colspan[1:]] +
                        lg2.margin_vals['leftcb'][ss2.colspan[1:]])
                    if maxsubl2 > maxsubl:
                        maxsubl = maxsubl2
                    maxsubr2 = np.max(
                        lg2.margin_vals['right'][ss2.colspan[:-1]] +
                        lg2.margin_vals['rightcb'][ss2.colspan[:-1]])
                    if maxsubr2 > maxsubr:
                        maxsubr = maxsubr2
            for i in ss1.colspan[1:]:
                lg1.edit_margin_min('left', maxsubl, cell=i)
            for i in ss1.colspan[:-1]:
                lg1.edit_margin_min('right', maxsubr, cell=i)

        # interior rows:
        if len(ss1.rowspan) > 1:
            maxsubt = np.max(
                lg1.margin_vals['top'][ss1.rowspan[1:]] +
                lg1.margin_vals['topcb'][ss1.rowspan[1:]]
            )
            maxsubb = np.max(
                lg1.margin_vals['bottom'][ss1.rowspan[:-1]] +
                lg1.margin_vals['bottomcb'][ss1.rowspan[:-1]]
            )

            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None:
                    if len(ss2.rowspan) > 1:
                        maxsubt = np.max([np.max(
                            lg2.margin_vals['top'][ss2.rowspan[1:]] +
                            lg2.margin_vals['topcb'][ss2.rowspan[1:]]
                        ), maxsubt])
                        maxsubb = np.max([np.max(
                            lg2.margin_vals['bottom'][ss2.rowspan[:-1]] +
                            lg2.margin_vals['bottomcb'][ss2.rowspan[:-1]]
                        ), maxsubb])
            for i in ss1.rowspan[1:]:
                lg1.edit_margin_min('top', maxsubt, cell=i)
            for i in ss1.rowspan[:-1]:
                lg1.edit_margin_min('bottom', maxsubb, cell=i)
```
### 26 - lib/matplotlib/_constrained_layout.py:

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
### 27 - lib/matplotlib/_constrained_layout.py:

Start line: 338, End line: 355

```python
def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
                        hspace=0, wspace=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.

    Then make room for colorbars.
    """
    for sfig in fig.subfigs:  # recursively make child panel margins
        ss = sfig._subplotspec
        make_layout_margins(layoutgrids, sfig, renderer,
                            w_pad=w_pad, h_pad=h_pad,
                            hspace=hspace, wspace=wspace)

        margins = get_margin_from_padding(sfig, w_pad=0, h_pad=0,
                                          hspace=hspace, wspace=wspace)
        layoutgrids[sfig].parent.edit_outer_margin_mins(margins, ss)
    # ... other code
```
### 28 - lib/matplotlib/_constrained_layout.py:

Start line: 461, End line: 490

```python
def match_submerged_margins(layoutgrids, fig):
    """
    Make the margins that are submerged inside an Axes the same size.

    This allows axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB\nCCDD")

    Without this routine, the axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    """

    for sfig in fig.subfigs:
        match_submerged_margins(layoutgrids, sfig)

    axs = [a for a in fig.get_axes()
           if a.get_subplotspec() is not None and a.get_in_layout()]
    # ... other code
```
### 30 - lib/matplotlib/_constrained_layout.py:

Start line: 152, End line: 194

```python
def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to axes get a layoutgrid so axes
    can have margins.
    """

    if layoutgrids is None:
        layoutgrids = dict()
        layoutgrids['hasgrids'] = False
    if not hasattr(fig, '_parent'):
        # top figure;  pass rect as parent to allow user-specified
        # margins
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=rect, name='figlb')
    else:
        # subfigure
        gs = fig._subplotspec.get_gridspec()
        # it is possible the gridspec containing this subfigure hasn't
        # been added to the tree yet:
        layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
        # add the layoutgrid for the subfigure:
        parentlb = layoutgrids[gs]
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(
            parent=parentlb,
            name='panellb',
            parent_inner=True,
            nrows=1, ncols=1,
            parent_pos=(fig._subplotspec.rowspan,
                        fig._subplotspec.colspan))
    # recursively do all subfigures in this figure...
    for sfig in fig.subfigs:
        layoutgrids = make_layoutgrids(sfig, layoutgrids)

    # for each axes at the local level add its gridspec:
    for ax in fig._localaxes:
        gs = ax.get_gridspec()
        if gs is not None:
            layoutgrids = make_layoutgrids_gs(layoutgrids, gs)

    return layoutgrids
```
### 32 - lib/matplotlib/legend.py:

Start line: 486, End line: 612

```python
class Legend(Artist):

    @_api.make_keyword_only("3.6", "loc")
    @_docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        *,
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        # ... other code
        if loc is None:
            loc = mpl.rcParams["legend.loc"]
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)
        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.

        if facecolor is None:
            facecolor = mpl.rcParams["legend.facecolor"]
        if facecolor == 'inherit':
            facecolor = mpl.rcParams["axes.facecolor"]

        if edgecolor is None:
            edgecolor = mpl.rcParams["legend.edgecolor"]
        if edgecolor == 'inherit':
            edgecolor = mpl.rcParams["axes.edgecolor"]

        if fancybox is None:
            fancybox = mpl.rcParams["legend.fancybox"]

        self.legendPatch = FancyBboxPatch(
            xy=(0, 0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            # If shadow is used, default to alpha=1 (#8943).
            alpha=(framealpha if framealpha is not None
                   else 1 if shadow
                   else mpl.rcParams["legend.framealpha"]),
            # The width and height of the legendPatch will be set (in draw())
            # to the length that includes the padding. Thus we set pad=0 here.
            boxstyle=("round,pad=0,rounding_size=0.2" if fancybox
                      else "square,pad=0"),
            mutation_scale=self._fontsize,
            snap=True,
            visible=(frameon if frameon is not None
                     else mpl.rcParams["legend.frameon"])
        )
        self._set_artist_props(self.legendPatch)

        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        self._alignment = alignment

        # init with null renderer
        self._init_legend_box(handles, labels, markerfirst)

        tmp = self._loc_used_default
        self._set_loc(loc)
        self._loc_used_default = tmp  # ignore changes done by _set_loc

        # figure out title font properties:
        if title_fontsize is not None and title_fontproperties is not None:
            raise ValueError(
                "title_fontsize and title_fontproperties can't be specified "
                "at the same time. Only use one of them. ")
        title_prop_fp = FontProperties._from_any(title_fontproperties)
        if isinstance(title_fontproperties, dict):
            if "size" not in title_fontproperties:
                title_fontsize = mpl.rcParams["legend.title_fontsize"]
                title_prop_fp.set_size(title_fontsize)
        elif title_fontsize is not None:
            title_prop_fp.set_size(title_fontsize)
        elif not isinstance(title_fontproperties, FontProperties):
            title_fontsize = mpl.rcParams["legend.title_fontsize"]
            title_prop_fp.set_size(title_fontsize)

        self.set_title(title, prop=title_prop_fp)

        self._draggable = None
        self.set_draggable(state=draggable)

        # set the text color

        color_getters = {  # getter function depends on line or patch
            'linecolor':       ['get_color',           'get_facecolor'],
            'markerfacecolor': ['get_markerfacecolor', 'get_facecolor'],
            'mfc':             ['get_markerfacecolor', 'get_facecolor'],
            'markeredgecolor': ['get_markeredgecolor', 'get_edgecolor'],
            'mec':             ['get_markeredgecolor', 'get_edgecolor'],
        }
        if labelcolor is None:
            if mpl.rcParams['legend.labelcolor'] is not None:
                labelcolor = mpl.rcParams['legend.labelcolor']
            else:
                labelcolor = mpl.rcParams['text.color']
        if isinstance(labelcolor, str) and labelcolor in color_getters:
            getter_names = color_getters[labelcolor]
            for handle, text in zip(self.legend_handles, self.texts):
                try:
                    if handle.get_array() is not None:
                        continue
                except AttributeError:
                    pass
                for getter_name in getter_names:
                    try:
                        color = getattr(handle, getter_name)()
                        if isinstance(color, np.ndarray):
                            if (
                                    color.shape[0] == 1
                                    or np.isclose(color, color[0]).all()
                            ):
                                text.set_color(color[0])
                            else:
                                pass
                        else:
                            text.set_color(color)
                        break
                    except AttributeError:
                        pass
        elif isinstance(labelcolor, str) and labelcolor == 'none':
            for text in self.texts:
                text.set_color(labelcolor)
        elif np.iterable(labelcolor):
            for text, color in zip(self.texts,
                                   itertools.cycle(
                                       colors.to_rgba_array(labelcolor))):
                text.set_color(color)
        else:
            raise ValueError(f"Invalid labelcolor: {labelcolor!r}")
```
### 34 - lib/matplotlib/_constrained_layout.py:

Start line: 750, End line: 763

```python
def colorbar_get_pad(layoutgrids, cax):
    parents = cax._colorbar_info['parents']
    gs = parents[0].get_gridspec()

    cb_rspans, cb_cspans = get_cb_parent_spans(cax)
    bboxouter = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)

    if cax._colorbar_info['location'] in ['right', 'left']:
        size = bboxouter.width
    else:
        size = bboxouter.height

    return cax._colorbar_info['pad'] * size
```
### 35 - tutorials/intermediate/legend_guide.py:

Start line: 123, End line: 195

```python
fig, ax_dict = plt.subplot_mosaic([['top', 'top'], ['bottom', 'BLANK']],
                                  empty_sentinel="BLANK")
ax_dict['top'].plot([1, 2, 3], label="test1")
ax_dict['top'].plot([3, 2, 1], label="test2")
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
ax_dict['top'].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=2, mode="expand", borderaxespad=0.)

ax_dict['bottom'].plot([1, 2, 3], label="test1")
ax_dict['bottom'].plot([3, 2, 1], label="test2")
# Place a legend to the right of this smaller subplot.
ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)

plt.show()

###############################################################################
# Multiple legends on the same Axes
# =================================
#
# Sometimes it is more clear to split legend entries across multiple
# legends. Whilst the instinctive approach to doing this might be to call
# the :func:`legend` function multiple times, you will find that only one
# legend ever exists on the Axes. This has been done so that it is possible
# to call :func:`legend` repeatedly to update the legend to the latest
# handles on the Axes. To keep old legend instances, we must add them
# manually to the Axes:

fig, ax = plt.subplots()
line1, = ax.plot([1, 2, 3], label="Line 1", linestyle='--')
line2, = ax.plot([3, 2, 1], label="Line 2", linewidth=4)

# Create a legend for the first line.
first_legend = ax.legend(handles=[line1], loc='upper right')

# Add the legend manually to the Axes.
ax.add_artist(first_legend)

# Create another legend for the second line.
ax.legend(handles=[line2], loc='lower right')

plt.show()

###############################################################################
# Legend Handlers
# ===============
#
# In order to create legend entries, handles are given as an argument to an
# appropriate :class:`~matplotlib.legend_handler.HandlerBase` subclass.
# The choice of handler subclass is determined by the following rules:
#
# 1. Update :func:`~matplotlib.legend.Legend.get_legend_handler_map`
#    with the value in the ``handler_map`` keyword.
# 2. Check if the ``handle`` is in the newly created ``handler_map``.
# 3. Check if the type of ``handle`` is in the newly created ``handler_map``.
# 4. Check if any of the types in the ``handle``'s mro is in the newly
#    created ``handler_map``.
#
# For completeness, this logic is mostly implemented in
# :func:`~matplotlib.legend.Legend.get_legend_handler`.
#
# All of this flexibility means that we have the necessary hooks to implement
# custom handlers for our own type of legend key.
#
# The simplest example of using custom handlers is to instantiate one of the
# existing `.legend_handler.HandlerBase` subclasses. For the
# sake of simplicity, let's choose `.legend_handler.HandlerLine2D`
# which accepts a *numpoints* argument (numpoints is also a keyword
# on the :func:`legend` function for convenience). We can then pass the mapping
# of instance to Handler as a keyword to legend.

from matplotlib.legend_handler import HandlerLine2D
```
### 36 - lib/matplotlib/legend.py:

Start line: 846, End line: 861

```python
class Legend(Artist):

    def _init_legend_box(self, handles, labels, markerfirst=True):
        # ... other code

        mode = "expand" if self._mode == "expand" else "fixed"
        sep = self.columnspacing * fontsize
        self._legend_handle_box = HPacker(pad=0,
                                          sep=sep, align="baseline",
                                          mode=mode,
                                          children=columnbox)
        self._legend_title_box = TextArea("")
        self._legend_box = VPacker(pad=self.borderpad * fontsize,
                                   sep=self.labelspacing * fontsize,
                                   align=self._alignment,
                                   children=[self._legend_title_box,
                                             self._legend_handle_box])
        self._legend_box.set_figure(self.figure)
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legend_handles = handle_list
```
### 40 - lib/matplotlib/figure.py:

Start line: 1762, End line: 1789

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
### 43 - lib/matplotlib/figure.py:

Start line: 2845, End line: 2886

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

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        if not isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
            return None, None, None, None
        info = self.get_layout_engine().get_info()
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
### 44 - examples/text_labels_and_annotations/figlegend_demo.py:

Start line: 1, End line: 31

```python
"""
==================
Figure legend demo
==================

Instead of plotting a legend on each axis, a legend for all the artists on all
the sub-axes of a figure can be plotted instead.
"""

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='upper right')

plt.tight_layout()
plt.show()
```
### 48 - lib/matplotlib/_constrained_layout.py:

Start line: 733, End line: 747

```python
def reset_margins(layoutgrids, fig):
    """
    Reset the margins in the layoutboxes of fig.

    Margins are usually set as a minimum, so if the figure gets smaller
    the minimum needs to be zero in order for it to grow again.
    """
    for sfig in fig.subfigs:
        reset_margins(layoutgrids, sfig)
    for ax in fig.axes:
        if ax.get_in_layout():
            gs = ax.get_gridspec()
            if gs in layoutgrids:  # also implies gs is not None.
                layoutgrids[gs].reset_margins()
    layoutgrids[fig].reset_margins()
```
### 49 - lib/matplotlib/figure.py:

Start line: 2574, End line: 2638

```python
@_docstring.interpd
class Figure(FigureBase):

    def set_layout_engine(self, layout=None, **kwargs):
        """
        Set the layout engine for this figure.

        Parameters
        ----------
        layout: {'constrained', 'compressed', 'tight', 'none'} or \
        ngine` or None

            - 'constrained' will use `~.ConstrainedLayoutEngine`
            - 'compressed' will also use `~.ConstrainedLayoutEngine`, but with
              a correction that attempts to make a good layout for fixed-aspect
              ratio Axes.
            - 'tight' uses `~.TightLayoutEngine`
            - 'none' removes layout engine.

            If `None`, the behavior is controlled by :rc:`figure.autolayout`
            (which if `True` behaves as if 'tight' was passed) and
            :rc:`figure.constrained_layout.use` (which if `True` behaves as if
            'constrained' was passed).  If both are `True`,
            :rc:`figure.autolayout` takes priority.

            Users and libraries can define their own layout engines and pass
            the instance directly as well.

        kwargs: dict
            The keyword arguments are passed to the layout engine to set things
            like padding and margin sizes.  Only used if *layout* is a string.

        """
        if layout is None:
            if mpl.rcParams['figure.autolayout']:
                layout = 'tight'
            elif mpl.rcParams['figure.constrained_layout.use']:
                layout = 'constrained'
            else:
                self._layout_engine = None
                return
        if layout == 'tight':
            new_layout_engine = TightLayoutEngine(**kwargs)
        elif layout == 'constrained':
            new_layout_engine = ConstrainedLayoutEngine(**kwargs)
        elif layout == 'compressed':
            new_layout_engine = ConstrainedLayoutEngine(compress=True,
                                                        **kwargs)
        elif layout == 'none':
            if self._layout_engine is not None:
                new_layout_engine = PlaceHolderLayoutEngine(
                    self._layout_engine.adjust_compatible,
                    self._layout_engine.colorbar_gridspec
                )
            else:
                new_layout_engine = None
        elif isinstance(layout, LayoutEngine):
            new_layout_engine = layout
        else:
            raise ValueError(f"Invalid value for 'layout': {layout!r}")

        if self._check_layout_engines_compat(self._layout_engine,
                                             new_layout_engine):
            self._layout_engine = new_layout_engine
        else:
            raise RuntimeError('Colorbar layout of new layout engine not '
                               'compatible with old engine, and a colorbar '
                               'has been created.  Engine not changed.')
```
### 50 - tutorials/intermediate/legend_guide.py:

Start line: 1, End line: 121

```python
"""
============
Legend guide
============

Generating legends flexibly in Matplotlib.

.. currentmodule:: matplotlib.pyplot

This legend guide is an extension of the documentation available at
:func:`~matplotlib.pyplot.legend` - please ensure you are familiar with
contents of that documentation before proceeding with this guide.


This guide makes use of some common terms, which are documented here for
clarity:

.. glossary::

    legend entry
        A legend is made up of one or more legend entries. An entry is made up
        of exactly one key and one label.

    legend key
        The colored/patterned marker to the left of each legend label.

    legend label
        The text which describes the handle represented by the key.

    legend handle
        The original object which is used to generate an appropriate entry in
        the legend.


Controlling the legend entries
==============================

Calling :func:`legend` with no arguments automatically fetches the legend
handles and their associated labels. This functionality is equivalent to::

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

The :meth:`~matplotlib.axes.Axes.get_legend_handles_labels` function returns
a list of handles/artists which exist on the Axes which can be used to
generate entries for the resulting legend - it is worth noting however that
not all artists can be added to a legend, at which point a "proxy" will have
to be created (see :ref:`proxy_legend_handles` for further details).

.. note::
    Artists with an empty string as label or with a label starting with an
    underscore, "_", will be ignored.

For full control of what is being added to the legend, it is common to pass
the appropriate handles directly to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend(handles=[line_up, line_down])

In some cases, it is not possible to set the label of the handle, so it is
possible to pass through the list of labels to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend([line_up, line_down], ['Line Up', 'Line Down'])


.. _proxy_legend_handles:

Creating artists specifically for adding to the legend (aka. Proxy artists)
===========================================================================

Not all handles can be turned into legend entries automatically,
so it is often necessary to create an artist which *can*. Legend handles
don't have to exist on the Figure or Axes in order to be used.

Suppose we wanted to create a legend which has an entry for some data which
is represented by a red color:
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
red_patch = mpatches.Patch(color='red', label='The red data')
ax.legend(handles=[red_patch])

plt.show()

###############################################################################
# There are many supported legend handles. Instead of creating a patch of color
# we could have created a line with a marker:

import matplotlib.lines as mlines

fig, ax = plt.subplots()
blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')
ax.legend(handles=[blue_line])

plt.show()

###############################################################################
# Legend location
# ===============
#
# The location of the legend can be specified by the keyword argument
# *loc*. Please see the documentation at :func:`legend` for more details.
#
# The ``bbox_to_anchor`` keyword gives a great degree of control for manual
# legend placement. For example, if you want your axes legend located at the
# figure's top right-hand corner instead of the axes' corner, simply specify
# the corner's location and the coordinate system of that location::
#
#     ax.legend(bbox_to_anchor=(1, 1),
#               bbox_transform=fig.transFigure)
#
# More examples of custom legend placement:
```
### 51 - lib/matplotlib/legend.py:

Start line: 614, End line: 644

```python
class Legend(Artist):

    legendHandles = _api.deprecated('3.7', alternative="legend_handles")(
        property(lambda self: self.legend_handles))

    def _set_artist_props(self, a):
        """
        Set the boilerplate props for artists added to axes.
        """
        a.set_figure(self.figure)
        if self.isaxes:
            # a.set_axes(self.axes)
            a.axes = self.axes

        a.set_transform(self.get_transform())

    def _set_loc(self, loc):
        # find_offset function will be provided to _legend_box and
        # _legend_box will draw itself at the location of the return
        # value of the find_offset.
        self._loc_used_default = False
        self._loc_real = loc
        self.stale = True
        self._legend_box.set_offset(self._findoffset)

    def set_ncols(self, ncols):
        """Set the number of columns."""
        self._ncols = ncols

    def _get_loc(self):
        return self._loc_real

    _loc = property(_get_loc, _set_loc)
```
### 52 - lib/matplotlib/figure.py:

Start line: 2555, End line: 2572

```python
@_docstring.interpd
class Figure(FigureBase):

    def _check_layout_engines_compat(self, old, new):
        """
        Helper for set_layout engine

        If the figure has used the old engine and added a colorbar then the
        value of colorbar_gridspec must be the same on the new engine.
        """
        if old is None or new is None:
            return True
        if old.colorbar_gridspec == new.colorbar_gridspec:
            return True
        # colorbar layout different, so check if any colorbars are on the
        # figure...
        for ax in self.axes:
            if hasattr(ax, '_colorbar'):
                # colorbars list themselves as a colorbar.
                return False
        return True
```
### 53 - lib/matplotlib/legend.py:

Start line: 1007, End line: 1048

```python
class Legend(Artist):

    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the legend will be anchored to.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or tuple
            The bounding box can be specified in the following ways:

            - A `.BboxBase` instance
            - A tuple of ``(left, bottom, width, height)`` in the given
              transform (normalized axes coordinate if None)
            - A tuple of ``(left, bottom)`` where the width and height will be
              assumed to be zero.
            - *None*, to remove the bbox anchoring, and use the parent bbox.

        transform : `~matplotlib.transforms.Transform`, optional
            A transform to apply to the bounding box. If not specified, this
            will use a transform to the bounding box of the parent.
        """
        if bbox is None:
            self._bbox_to_anchor = None
            return
        elif isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f"Invalid bbox: {bbox}") from err

            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        if transform is None:
            transform = BboxTransformTo(self.parent.bbox)

        self._bbox_to_anchor = TransformedBbox(self._bbox_to_anchor,
                                               transform)
        self.stale = True
```
### 54 - lib/matplotlib/figure.py:

Start line: 1, End line: 53

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
### 55 - lib/matplotlib/figure.py:

Start line: 2533, End line: 2553

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.make_keyword_only("3.6", "facecolor")
    def __init__(self,
                 figsize=None,
                 dpi=None,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 subplotpars=None,  # rc figure.subplot.*
                 tight_layout=None,  # rc figure.autolayout
                 constrained_layout=None,  # rc figure.constrained_layout.use
                 *,
                 layout=None,
                 **kwargs
                 ):
        # ... other code

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # Don't let the figure patch influence bbox calculation.
            in_layout=False)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

        FigureCanvasBase(self)  # Set self.canvas.

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars

        self._axstack = _AxesStack()  # track all figure axes and current axes
        self.clear()

    def pick(self, mouseevent):
        if not self.canvas.widgetlock.locked():
            super().pick(mouseevent)
```
### 56 - tutorials/intermediate/legend_guide.py:

Start line: 197, End line: 283

```python
fig, ax = plt.subplots()
line1, = ax.plot([3, 2, 1], marker='o', label='Line 1')
line2, = ax.plot([1, 2, 3], marker='o', label='Line 2')

ax.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

###############################################################################
# As you can see, "Line 1" now has 4 marker points, where "Line 2" has 2 (the
# default). Try the above code, only change the map's key from ``line1`` to
# ``type(line1)``. Notice how now both `.Line2D` instances get 4 markers.
#
# Along with handlers for complex plot types such as errorbars, stem plots
# and histograms, the default ``handler_map`` has a special ``tuple`` handler
# (`.legend_handler.HandlerTuple`) which simply plots the handles on top of one
# another for each item in the given tuple. The following example demonstrates
# combining two legend keys on top of one another:

from numpy.random import randn

z = randn(10)

fig, ax = plt.subplots()
red_dot, = ax.plot(z, "ro", markersize=15)
# Put a white cross over some of the data.
white_cross, = ax.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

ax.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])

###############################################################################
# The `.legend_handler.HandlerTuple` class can also be used to
# assign several legend keys to the same entry:

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

fig, ax = plt.subplots()
p1, = ax.plot([1, 2.5, 3], 'r-d')
p2, = ax.plot([3, 2, 1], 'k-o')

l = ax.legend([(p1, p2)], ['Two keys'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)})

###############################################################################
# Implementing a custom legend handler
# ------------------------------------
#
# A custom handler can be implemented to turn any handle into a legend key
# (handles don't necessarily need to be matplotlib artists).  The handler must
# implement a ``legend_artist`` method which returns a single artist for the
# legend to use. The required signature for ``legend_artist`` is documented at
# `~.legend_handler.HandlerBase.legend_artist`.

import matplotlib.patches as mpatches


class AnyObject:
    pass


class AnyObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

fig, ax = plt.subplots()

ax.legend([AnyObject()], ['My first handler'],
          handler_map={AnyObject: AnyObjectHandler()})

###############################################################################
# Alternatively, had we wanted to globally accept ``AnyObject`` instances
# without needing to manually set the *handler_map* keyword all the time, we
# could have registered the new handler with::
#
#     from matplotlib.legend import Legend
#     Legend.update_default_handler_map({AnyObject: AnyObjectHandler()})
#
# Whilst the power here is clear, remember that there are already many handlers
# implemented and what you want to achieve may already be easily possible with
# existing classes. For example, to produce elliptical legend keys, rather than
# rectangular ones:

from matplotlib.legend_handler import HandlerPatch
```
### 58 - lib/matplotlib/legend.py:

Start line: 898, End line: 938

```python
class Legend(Artist):

    def get_children(self):
        # docstring inherited
        return [self._legend_box, self.get_frame()]

    def get_frame(self):
        """Return the `~.patches.Rectangle` used to frame the legend."""
        return self.legendPatch

    def get_lines(self):
        r"""Return the list of `~.lines.Line2D`\s in the legend."""
        return [h for h in self.legend_handles if isinstance(h, Line2D)]

    def get_patches(self):
        r"""Return the list of `~.patches.Patch`\s in the legend."""
        return silent_list('Patch',
                           [h for h in self.legend_handles
                            if isinstance(h, Patch)])

    def get_texts(self):
        r"""Return the list of `~.text.Text`\s in the legend."""
        return silent_list('Text', self.texts)

    def set_alignment(self, alignment):
        """
        Set the alignment of the legend title and the box of entries.

        The entries are aligned as a single block, so that markers always
        lined up.

        Parameters
        ----------
        alignment : {'center', 'left', 'right'}.

        """
        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        self._alignment = alignment
        self._legend_box.align = alignment

    def get_alignment(self):
        """Get the alignment value of the legend box"""
        return self._legend_box.align
```
### 59 - lib/matplotlib/figure.py:

Start line: 2362, End line: 2531

```python
@_docstring.interpd
class Figure(FigureBase):

    @_api.make_keyword_only("3.6", "facecolor")
    def __init__(self,
                 figsize=None,
                 dpi=None,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 subplotpars=None,  # rc figure.subplot.*
                 tight_layout=None,  # rc figure.autolayout
                 constrained_layout=None,  # rc figure.constrained_layout.use
                 *,
                 layout=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        figsize : 2-tuple of floats, default: :rc:`figure.figsize`
            Figure dimension ``(width, height)`` in inches.

        dpi : float, default: :rc:`figure.dpi`
            Dots per inch.

        facecolor : default: :rc:`figure.facecolor`
            The figure patch facecolor.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.

        subplotpars : `SubplotParams`
            Subplot parameters. If not given, the default subplot
            parameters :rc:`figure.subplot.*` are used.

        tight_layout : bool or dict, default: :rc:`figure.autolayout`
            Whether to use the tight layout mechanism. See `.set_tight_layout`.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='tight'`` instead for the common case of
                ``tight_layout=True`` and use `.set_tight_layout` otherwise.

        constrained_layout : bool, default: :rc:`figure.constrained_layout.use`
            This is equal to ``layout='constrained'``.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='constrained'`` instead.

        layout : {'constrained', 'compressed', 'tight', `.LayoutEngine`, None}
            The layout mechanism for positioning of plot elements to avoid
            overlapping Axes decorations (labels, ticks, etc). Note that
            layout managers can have significant performance penalties.
            Defaults to *None*.

            - 'constrained': The constrained layout solver adjusts axes sizes
               to avoid overlapping axes decorations.  Can handle complex plot
               layouts and colorbars, and is thus recommended.

              See :doc:`/tutorials/intermediate/constrainedlayout_guide`
              for examples.

            - 'compressed': uses the same algorithm as 'constrained', but
              removes extra space between fixed-aspect-ratio Axes.  Best for
              simple grids of axes.

            - 'tight': Use the tight layout mechanism. This is a relatively
              simple algorithm that adjusts the subplot parameters so that
              decorations do not overlap. See `.Figure.set_tight_layout` for
              further details.

            - A `.LayoutEngine` instance. Builtin layout classes are
              `.ConstrainedLayoutEngine` and `.TightLayoutEngine`, more easily
              accessible by 'constrained' and 'tight'.  Passing an instance
              allows third parties to provide their own layout engine.

            If not given, fall back to using the parameters *tight_layout* and
            *constrained_layout*, including their config defaults
            :rc:`figure.autolayout` and :rc:`figure.constrained_layout.use`.

        Other Parameters
        ----------------
        **kwargs : `.Figure` properties, optional

            %(Figure:kwdoc)s
        """
        super().__init__(**kwargs)
        self._layout_engine = None

        if layout is not None:
            if (tight_layout is not None):
                _api.warn_external(
                    "The Figure parameters 'layout' and 'tight_layout' cannot "
                    "be used together. Please use 'layout' only.")
            if (constrained_layout is not None):
                _api.warn_external(
                    "The Figure parameters 'layout' and 'constrained_layout' "
                    "cannot be used together. Please use 'layout' only.")
            self.set_layout_engine(layout=layout)
        elif tight_layout is not None:
            if constrained_layout is not None:
                _api.warn_external(
                    "The Figure parameters 'tight_layout' and "
                    "'constrained_layout' cannot be used together. Please use "
                    "'layout' parameter")
            self.set_layout_engine(layout='tight')
            if isinstance(tight_layout, dict):
                self.get_layout_engine().set(**tight_layout)
        elif constrained_layout is not None:
            if isinstance(constrained_layout, dict):
                self.set_layout_engine(layout='constrained')
                self.get_layout_engine().set(**constrained_layout)
            elif constrained_layout:
                self.set_layout_engine(layout='constrained')

        else:
            # everything is None, so use default:
            self.set_layout_engine(layout=layout)

        self._fig_callbacks = cbook.CallbackRegistry(signals=["dpi_changed"])
        # Callbacks traditionally associated with the canvas (and exposed with
        # a proxy property), but that actually need to be on the figure for
        # pickling.
        self._canvas_callbacks = cbook.CallbackRegistry(
            signals=FigureCanvasBase.events)
        connect = self._canvas_callbacks._connect_picklable
        self._mouse_key_ids = [
            connect('key_press_event', backend_bases._key_handler),
            connect('key_release_event', backend_bases._key_handler),
            connect('key_release_event', backend_bases._key_handler),
            connect('button_press_event', backend_bases._mouse_handler),
            connect('button_release_event', backend_bases._mouse_handler),
            connect('scroll_event', backend_bases._mouse_handler),
            connect('motion_notify_event', backend_bases._mouse_handler),
        ]
        self._button_pick_id = connect('button_press_event', self.pick)
        self._scroll_pick_id = connect('scroll_event', self.pick)

        if figsize is None:
            figsize = mpl.rcParams['figure.figsize']
        if dpi is None:
            dpi = mpl.rcParams['figure.dpi']
        if facecolor is None:
            facecolor = mpl.rcParams['figure.facecolor']
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        if not np.isfinite(figsize).all() or (np.array(figsize) < 0).any():
            raise ValueError('figure size must be positive finite not '
                             f'{figsize}')
        self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)

        self.dpi_scale_trans = Affine2D().scale(dpi)
        # do not use property as it will trigger
        self._dpi = dpi
        self.bbox = TransformedBbox(self.bbox_inches, self.dpi_scale_trans)
        self.figbbox = self.bbox
        self.transFigure = BboxTransformTo(self.bbox)
        self.transSubfigure = self.transFigure
        # ... other code
```
### 60 - lib/matplotlib/legend.py:

Start line: 302, End line: 485

```python
class Legend(Artist):
    """
    Place a legend on the axes at location loc.
    """

    # 'best' is only implemented for axes legends
    codes = {'best': 0, **AnchoredOffsetbox.codes}
    zorder = 5

    def __str__(self):
        return "Legend"

    @_api.make_keyword_only("3.6", "loc")
    @_docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        *,
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        """
        Parameters
        ----------
        parent : `~matplotlib.axes.Axes` or `.Figure`
            The artist that contains the legend.

        handles : list of `.Artist`
            A list of Artists (lines, patches) to be added to the legend.

        labels : list of str
            A list of labels to show next to the artists. The length of handles
            and labels should be the same. If they are not, they are truncated
            to the length of the shorter list.

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        Attributes
        ----------
        legend_handles
            List of `.Artist` objects added as legend entries.

            .. versionadded:: 3.7

        Notes
        -----
        Users can specify any arbitrary location for the legend using the
        *bbox_to_anchor* keyword argument. *bbox_to_anchor* can be a
        `.BboxBase` (or derived there from) or a tuple of 2 or 4 floats.
        See `set_bbox_to_anchor` for more detail.

        The legend location can be specified by setting *loc* with a tuple of
        2 floats, which is interpreted as the lower-left corner of the legend
        in the normalized axes coordinate.
        """
        # local import only to avoid circularity
        from matplotlib.axes import Axes
        from matplotlib.figure import FigureBase

        super().__init__()

        if prop is None:
            if fontsize is not None:
                self.prop = FontProperties(size=fontsize)
            else:
                self.prop = FontProperties(
                    size=mpl.rcParams["legend.fontsize"])
        else:
            self.prop = FontProperties._from_any(prop)
            if isinstance(prop, dict) and "size" not in prop:
                self.prop.set_size(mpl.rcParams["legend.fontsize"])

        self._fontsize = self.prop.get_size_in_points()

        self.texts = []
        self.legend_handles = []
        self._legend_title_box = None

        #: A dictionary with the extra handler mappings for this Legend
        #: instance.
        self._custom_handler_map = handler_map

        def val_or_rc(val, rc_name):
            return val if val is not None else mpl.rcParams[rc_name]

        self.numpoints = val_or_rc(numpoints, 'legend.numpoints')
        self.markerscale = val_or_rc(markerscale, 'legend.markerscale')
        self.scatterpoints = val_or_rc(scatterpoints, 'legend.scatterpoints')
        self.borderpad = val_or_rc(borderpad, 'legend.borderpad')
        self.labelspacing = val_or_rc(labelspacing, 'legend.labelspacing')
        self.handlelength = val_or_rc(handlelength, 'legend.handlelength')
        self.handleheight = val_or_rc(handleheight, 'legend.handleheight')
        self.handletextpad = val_or_rc(handletextpad, 'legend.handletextpad')
        self.borderaxespad = val_or_rc(borderaxespad, 'legend.borderaxespad')
        self.columnspacing = val_or_rc(columnspacing, 'legend.columnspacing')
        self.shadow = val_or_rc(shadow, 'legend.shadow')
        # trim handles and labels if illegal label...
        _lab, _hand = [], []
        for label, handle in zip(labels, handles):
            if isinstance(label, str) and label.startswith('_'):
                _api.warn_external(f"The label {label!r} of {handle!r} starts "
                                   "with '_'. It is thus excluded from the "
                                   "legend.")
            else:
                _lab.append(label)
                _hand.append(handle)
        labels, handles = _lab, _hand

        if reverse:
            labels.reverse()
            handles.reverse()

        if len(handles) < 2:
            ncols = 1
        self._ncols = ncols if ncols != 1 else ncol

        if self.numpoints <= 0:
            raise ValueError("numpoints must be > 0; it was %d" % numpoints)

        # introduce y-offset for handles of the scatter plot
        if scatteryoffsets is None:
            self._scatteryoffsets = np.array([3. / 8., 4. / 8., 2.5 / 8.])
        else:
            self._scatteryoffsets = np.asarray(scatteryoffsets)
        reps = self.scatterpoints // len(self._scatteryoffsets) + 1
        self._scatteryoffsets = np.tile(self._scatteryoffsets,
                                        reps)[:self.scatterpoints]

        # _legend_box is a VPacker instance that contains all
        # legend items and will be initialized from _init_legend_box()
        # method.
        self._legend_box = None

        if isinstance(parent, Axes):
            self.isaxes = True
            self.axes = parent
            self.set_figure(parent.figure)
        elif isinstance(parent, FigureBase):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError(
                "Legend needs either Axes or FigureBase as parent"
            )
        self.parent = parent

        self._loc_used_default = loc is None
        # ... other code
```
### 62 - lib/matplotlib/figure.py:

Start line: 2640, End line: 2652

```python
@_docstring.interpd
class Figure(FigureBase):

    def get_layout_engine(self):
        return self._layout_engine

    # TODO: I'd like to dynamically add the _repr_html_ method
    # to the figure in the right context, but then IPython doesn't
    # use it, for some reason.

    def _repr_html_(self):
        # We can't use "isinstance" here, because then we'd end up importing
        # webagg unconditionally.
        if 'WebAgg' in type(self.canvas).__name__:
            from matplotlib.backends import backend_webagg
            return backend_webagg.ipython_inline_display(self)
```
### 63 - lib/matplotlib/figure.py:

Start line: 388, End line: 397

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.02, y0=0.5, name='supylabel', ha='left',
                             va='center', rc='label')
    @_docstring.copy(_suplabels)
    def supylabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supylabel', 'x0': 0.02, 'y0': 0.5,
                'ha': 'left', 'va': 'center', 'rotation': 'vertical',
                'rotation_mode': 'anchor', 'size': 'figure.labelsize',
                'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)
```
### 64 - lib/matplotlib/legend.py:

Start line: 663, End line: 692

```python
class Legend(Artist):

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return

        renderer.open_group('legend', gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # if mode == fill, set the width of the legend_box to the
        # width of the parent (minus pads)
        if self._mode in ["expand"]:
            pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
            self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

        # update the location and size of the legend. This needs to
        # be done in any case to clip the figure right.
        bbox = self._legend_box.get_window_extent(renderer)
        self.legendPatch.set_bounds(bbox.bounds)
        self.legendPatch.set_mutation_scale(fontsize)

        if self.shadow:
            Shadow(self.legendPatch, 2, -2).draw(renderer)

        self.legendPatch.draw(renderer)
        self._legend_box.draw(renderer)

        renderer.close_group('legend')
        self.stale = False
```
### 68 - lib/matplotlib/_constrained_layout.py:

Start line: 197, End line: 240

```python
def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """

    if gs in layoutgrids or gs.figure is None:
        return layoutgrids
    # in order to do constrained_layout there has to be at least *one*
    # gridspec in the tree:
    layoutgrids['hasgrids'] = True
    if not hasattr(gs, '_subplot_spec'):
        # normal gridspec
        parent = layoutgrids[gs.figure]
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=parent,
                parent_inner=True,
                name='gridspec',
                ncols=gs._ncols, nrows=gs._nrows,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    else:
        # this is a gridspecfromsubplotspec:
        subplot_spec = gs._subplot_spec
        parentgs = subplot_spec.get_gridspec()
        # if a nested gridspec it is possible the parent is not in there yet:
        if parentgs not in layoutgrids:
            layoutgrids = make_layoutgrids_gs(layoutgrids, parentgs)
        subspeclb = layoutgrids[parentgs]
        # gridspecfromsubplotspec need an outer container:
        # get a unique representation:
        rep = (gs, 'top')
        if rep not in layoutgrids:
            layoutgrids[rep] = mlayoutgrid.LayoutGrid(
                parent=subspeclb,
                name='top',
                nrows=1, ncols=1,
                parent_pos=(subplot_spec.rowspan, subplot_spec.colspan))
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=layoutgrids[rep],
                name='gridspec',
                nrows=gs._nrows, ncols=gs._ncols,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    return layoutgrids
```
### 69 - lib/matplotlib/figure.py:

Start line: 378, End line: 386

```python
class FigureBase(Artist):

    @_docstring.Substitution(x0=0.5, y0=0.01, name='supxlabel', ha='center',
                             va='bottom', rc='label')
    @_docstring.copy(_suplabels)
    def supxlabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
                'ha': 'center', 'va': 'bottom', 'rotation': 0,
                'size': 'figure.labelsize', 'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)
```
### 70 - lib/matplotlib/legend.py:

Start line: 53, End line: 1157

```python
class DraggableLegend(DraggableOffsetBox):
    def __init__(self, legend, use_blit=False, update="loc"):
        """
        Wrapper around a `.Legend` to support mouse dragging.

        Parameters
        ----------
        legend : `.Legend`
            The `.Legend` instance to wrap.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            If "loc", update the *loc* parameter of the legend upon finalizing.
            If "bbox", update the *bbox_to_anchor* parameter.
        """
        self.legend = legend

        _api.check_in_list(["loc", "bbox"], update=update)
        self._update = update

        super().__init__(legend, legend._legend_box, use_blit=use_blit)

    def finalize_offset(self):
        if self._update == "loc":
            self._update_loc(self.get_loc_in_canvas())
        elif self._update == "bbox":
            self._bbox_to_anchor(self.get_loc_in_canvas())

    def _update_loc(self, loc_in_canvas):
        bbox = self.legend.get_bbox_to_anchor()
        # if bbox has zero width or height, the transformation is
        # ill-defined. Fall back to the default bbox_to_anchor.
        if bbox.width == 0 or bbox.height == 0:
            self.legend.set_bbox_to_anchor(None)
            bbox = self.legend.get_bbox_to_anchor()
        _bbox_transform = BboxTransformFrom(bbox)
        self.legend._loc = tuple(_bbox_transform.transform(loc_in_canvas))

    def _update_bbox_to_anchor(self, loc_in_canvas):
        loc_in_bbox = self.legend.axes.transAxes.transform(loc_in_canvas)
        self.legend.set_bbox_to_anchor(loc_in_bbox)


class Legend(Artist):
```
### 72 - lib/matplotlib/_constrained_layout.py:

Start line: 606, End line: 645

```python
def reposition_axes(layoutgrids, fig, renderer, *,
                    w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    Reposition all the axes based on the new inner bounding box.
    """
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure
    for sfig in fig.subfigs:
        bbox = layoutgrids[sfig].get_outer_bbox()
        sfig._redo_transform_rel_fig(
            bbox=bbox.transformed(trans_fig_to_subfig))
        reposition_axes(layoutgrids, sfig, renderer,
                        w_pad=w_pad, h_pad=h_pad,
                        wspace=wspace, hspace=hspace)

    for ax in fig._localaxes:
        if ax.get_subplotspec() is None or not ax.get_in_layout():
            continue

        # grid bbox is in Figure coordinates, but we specify in panel
        # coordinates...
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        if gs not in layoutgrids:
            return

        bbox = layoutgrids[gs].get_inner_bbox(rows=ss.rowspan,
                                              cols=ss.colspan)

        # transform from figure to panel for set_position:
        newbbox = trans_fig_to_subfig.transform_bbox(bbox)
        ax._set_position(newbbox)

        # move the colorbars:
        # we need to keep track of oldw and oldh if there is more than
        # one colorbar:
        offset = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        for nn, cbax in enumerate(ax._colorbars[::-1]):
            if ax == cbax._colorbar_info['parents'][0]:
                reposition_colorbar(layoutgrids, cbax, renderer,
                                    offset=offset)
```
### 73 - lib/matplotlib/figure.py:

Start line: 3464, End line: 3502

```python
@_docstring.interpd
class Figure(FigureBase):

    def tight_layout(self, *, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust the padding between and around subplots.

        To exclude an artist on the Axes from the bounding box calculation
        that determines the subplot parameters (i.e. legend, or annotation),
        set ``a.set_in_layout(False)`` for that artist.

        Parameters
        ----------
        pad : float, default: 1.08
            Padding between the figure edge and the edges of subplots,
            as a fraction of the font size.
        h_pad, w_pad : float, default: *pad*
            Padding (height/width) between edges of adjacent subplots,
            as a fraction of the font size.
        rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1)
            A rectangle in normalized figure coordinates into which the whole
            subplots area (including labels) will fit.

        See Also
        --------
        .Figure.set_layout_engine
        .pyplot.tight_layout
        """
        # note that here we do not permanently set the figures engine to
        # tight_layout but rather just perform the layout in place and remove
        # any previous engines.
        engine = TightLayoutEngine(pad=pad, h_pad=h_pad, w_pad=w_pad,
                                   rect=rect)
        try:
            previous_engine = self.get_layout_engine()
            self.set_layout_engine(engine)
            engine.execute(self)
            if not isinstance(previous_engine, TightLayoutEngine) \
                    and previous_engine is not None:
                _api.warn_external('The figure layout has changed to tight')
        finally:
            self.set_layout_engine(None)
```
### 76 - lib/matplotlib/_constrained_layout.py:

Start line: 243, End line: 260

```python
def check_no_collapsed_axes(layoutgrids, fig):
    """
    Check that no axes have collapsed to zero size.
    """
    for sfig in fig.subfigs:
        ok = check_no_collapsed_axes(layoutgrids, sfig)
        if not ok:
            return False
    for ax in fig.axes:
        gs = ax.get_gridspec()
        if gs in layoutgrids:  # also implies gs is not None.
            lg = layoutgrids[gs]
            for i in range(gs.nrows):
                for j in range(gs.ncols):
                    bb = lg.get_inner_bbox(i, j)
                    if bb.width <= 0 or bb.height <= 0:
                        return False
    return True
```
### 77 - lib/matplotlib/legend.py:

Start line: 1, End line: 50

```python
"""
The legend module defines the Legend class, which is responsible for
drawing legends associated with axes and/or figures.

.. important::

    It is unlikely that you would ever create a Legend instance manually.
    Most users would normally create a legend via the `~.Axes.legend`
    function. For more details on legends there is also a :doc:`legend guide
    </tutorials/intermediate/legend_guide>`.

The `Legend` class is a container of legend handles and legend texts.

The legend handler map specifies how to create legend handles from artists
(lines, patches, etc.) in the axes or figures. Default legend handlers are
defined in the :mod:`~matplotlib.legend_handler` module. While not all artist
types are covered by the default legend handlers, custom legend handlers can be
defined to support arbitrary objects.

See the :doc:`legend guide </tutorials/intermediate/legend_guide>` for more
information.
"""

import itertools
import logging
import time

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
                                StepPatch)
from matplotlib.collections import (
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection)
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
    AnchoredOffsetbox, DraggableOffsetBox,
    HPacker, VPacker,
    DrawingArea, TextArea,
)
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
```
### 80 - lib/matplotlib/figure.py:

Start line: 2047, End line: 2086

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
### 86 - lib/matplotlib/legend.py:

Start line: 1050, End line: 1067

```python
class Legend(Artist):

    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x, y) coordinate of the bbox.

        Parameters
        ----------
        loc : int
            A location code in range(1, 11). This corresponds to the possible
            values for ``self._loc``, excluding "best".
        bbox : `~matplotlib.transforms.Bbox`
            bbox to be placed, in display coordinates.
        parentbbox : `~matplotlib.transforms.Bbox`
            A parent box which will contain the bbox, in display coordinates.
        """
        return offsetbox._get_anchored_bbox(
            loc, bbox, parentbbox,
            self.borderaxespad * renderer.points_to_pixels(self._fontsize))
```
### 89 - lib/matplotlib/figure.py:

Start line: 596, End line: 613

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def add_axes(self, *args, **kwargs):
        # ... other code

        if isinstance(args[0], Axes):
            a = args[0]
            key = a._projection_init
            if a.get_figure() is not self:
                raise ValueError(
                    "The Axes must have been created in the present figure")
        else:
            rect = args[0]
            if not np.isfinite(rect).all():
                raise ValueError('all entries in rect must be finite '
                                 'not {}'.format(rect))
            projection_class, pkw = self._process_projection_requirements(
                *args, **kwargs)

            # create the new axes using the axes class given
            a = projection_class(self, rect, **pkw)
            key = (projection_class, pkw)
        return self._add_axes_internal(a, key)
```
### 92 - lib/matplotlib/figure.py:

Start line: 368, End line: 376

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
### 94 - lib/matplotlib/legend.py:

Start line: 763, End line: 844

```python
class Legend(Artist):

    def _init_legend_box(self, handles, labels, markerfirst=True):
        """
        Initialize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self._fontsize

        # legend_box is a HPacker, horizontally packed with columns.
        # Each column is a VPacker, vertically packed with legend items.
        # Each legend item is a HPacker packed with:
        # - handlebox: a DrawingArea which contains the legend handle.
        # - labelbox: a TextArea which contains the legend text.

        text_list = []  # the list of text instances
        handle_list = []  # the list of handle instances
        handles_and_labels = []

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.
        height = fontsize * self.handleheight - descent
        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their coordinates should
        # be given in the display coordinates.

        # The transformation of each handle will be automatically set
        # to self.get_transform(). If the artist does not use its
        # default transform (e.g., Collections), you need to
        # manually set their transform to the self.get_transform().
        legend_handler_map = self.get_legend_handler_map()

        for orig_handle, label in zip(handles, labels):
            handler = self.get_legend_handler(legend_handler_map, orig_handle)
            if handler is None:
                _api.warn_external(
                             "Legend does not support handles for {0} "
                             "instances.\nA proxy artist may be used "
                             "instead.\nSee: https://matplotlib.org/"
                             "stable/tutorials/intermediate/legend_guide.html"
                             "#controlling-the-legend-entries".format(
                                 type(orig_handle).__name__))
                # No handle for this artist, so we just defer to None.
                handle_list.append(None)
            else:
                textbox = TextArea(label, multilinebaseline=True,
                                   textprops=dict(
                                       verticalalignment='baseline',
                                       horizontalalignment='left',
                                       fontproperties=self.prop))
                handlebox = DrawingArea(width=self.handlelength * fontsize,
                                        height=height,
                                        xdescent=0., ydescent=descent)

                text_list.append(textbox._text)
                # Create the artist for the legend which represents the
                # original artist/handle.
                handle_list.append(handler.legend_artist(self, orig_handle,
                                                         fontsize, handlebox))
                handles_and_labels.append((handlebox, textbox))

        columnbox = []
        # array_split splits n handles_and_labels into ncols columns, with the
        # first n%ncols columns having an extra entry.  filter(len, ...)
        # handles the case where n < ncols: the last ncols-n columns are empty
        # and get filtered out.
        for handles_and_labels_column in filter(
                len, np.array_split(handles_and_labels, self._ncols)):
            # pack handlebox and labelbox into itembox
            itemboxes = [HPacker(pad=0,
                                 sep=self.handletextpad * fontsize,
                                 children=[h, t] if markerfirst else [t, h],
                                 align="baseline")
                         for h, t in handles_and_labels_column]
            # pack columnbox
            alignment = "baseline" if markerfirst else "right"
            columnbox.append(VPacker(pad=0,
                                     sep=self.labelspacing * fontsize,
                                     align=alignment,
                                     children=itemboxes))
        # ... other code
```
### 95 - lib/matplotlib/figure.py:

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
### 97 - lib/matplotlib/figure.py:

Start line: 1251, End line: 1282

```python
class FigureBase(Artist):

    @_docstring.dedent_interpd
    def colorbar(
            self, mappable, cax=None, ax=None, use_gridspec=True, **kwargs):
        # ... other code
        if cax is None:
            if ax is None:
                _api.warn_deprecated("3.6", message=(
                    'Unable to determine Axes to steal space for Colorbar. '
                    'Using gca(), but will raise in the future. '
                    'Either provide the *cax* argument to use as the Axes for '
                    'the Colorbar, provide the *ax* argument to steal space '
                    'from it, or add *mappable* to an Axes.'))
                ax = self.gca()
            current_ax = self.gca()
            userax = False
            if (use_gridspec
                    and isinstance(ax, mpl.axes._base._AxesBase)
                    and ax.get_subplotspec()):
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
### 98 - lib/matplotlib/figure.py:

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
### 101 - lib/matplotlib/figure.py:

Start line: 1284, End line: 1324

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
### 105 - lib/matplotlib/_constrained_layout.py:

Start line: 558, End line: 575

```python
def get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to:
    """
    rowstart = np.inf
    rowstop = -np.inf
    colstart = np.inf
    colstop = -np.inf
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
        rowstart = min(ss.rowspan.start, rowstart)
        rowstop = max(ss.rowspan.stop, rowstop)
        colstart = min(ss.colspan.start, colstart)
        colstop = max(ss.colspan.stop, colstop)

    rowspan = range(rowstart, rowstop)
    colspan = range(colstart, colstop)
    return rowspan, colspan
```
### 112 - lib/matplotlib/figure.py:

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
### 113 - lib/matplotlib/figure.py:

Start line: 288, End line: 366

```python
class FigureBase(Artist):

    def _suplabels(self, t, info, **kwargs):
        """
        Add a centered %(name)s to the figure.

        Parameters
        ----------
        t : str
            The %(name)s text.
        x : float, default: %(x0)s
            The x location of the text in figure coordinates.
        y : float, default: %(y0)s
            The y location of the text in figure coordinates.
        horizontalalignment, ha : {'center', 'left', 'right'}, default: %(ha)s
            The horizontal alignment of the text relative to (*x*, *y*).
        verticalalignment, va : {'top', 'center', 'bottom', 'baseline'}, \
         %(va)s
            The vertical alignment of the text relative to (*x*, *y*).
        fontsize, size : default: :rc:`figure.%(rc)ssize`
            The font size of the text. See `.Text.set_size` for possible
            values.
        fontweight, weight : default: :rc:`figure.%(rc)sweight`
            The font weight of the text. See `.Text.set_weight` for possible
            values.

        Returns
        -------
        text
            The `.Text` instance of the %(name)s.

        Other Parameters
        ----------------
        fontproperties : None or dict, optional
            A dict of font properties. If *fontproperties* is given the
            default values for font size and weight are taken from the
            `.FontProperties` defaults. :rc:`figure.%(rc)ssize` and
            :rc:`figure.%(rc)sweight` are ignored in this case.

        **kwargs
            Additional kwargs are `matplotlib.text.Text` properties.
        """

        suplab = getattr(self, info['name'])

        x = kwargs.pop('x', None)
        y = kwargs.pop('y', None)
        if info['name'] in ['_supxlabel', '_suptitle']:
            autopos = y is None
        elif info['name'] == '_supylabel':
            autopos = x is None
        if x is None:
            x = info['x0']
        if y is None:
            y = info['y0']

        if 'horizontalalignment' not in kwargs and 'ha' not in kwargs:
            kwargs['horizontalalignment'] = info['ha']
        if 'verticalalignment' not in kwargs and 'va' not in kwargs:
            kwargs['verticalalignment'] = info['va']
        if 'rotation' not in kwargs:
            kwargs['rotation'] = info['rotation']

        if 'fontproperties' not in kwargs:
            if 'fontsize' not in kwargs and 'size' not in kwargs:
                kwargs['size'] = mpl.rcParams[info['size']]
            if 'fontweight' not in kwargs and 'weight' not in kwargs:
                kwargs['weight'] = mpl.rcParams[info['weight']]

        sup = self.text(x, y, t, **kwargs)
        if suplab is not None:
            suplab.set_text(t)
            suplab.set_position((x, y))
            suplab.update_from(sup)
            sup.remove()
        else:
            suplab = sup
        suplab._autopos = autopos
        setattr(self, info['name'], suplab)
        self.stale = True
        return suplab
```
### 114 - lib/matplotlib/figure.py:

Start line: 2114, End line: 2210

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
    """
    callbacks = _api.deprecated(
            "3.6", alternative=("the 'resize_event' signal in "
                                "Figure.canvas.callbacks")
            )(property(lambda self: self._fig_callbacks))

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

        facecolor : default: :rc:`figure.facecolor`
            The figure patch face color.

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
            facecolor = mpl.rcParams['figure.facecolor']
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        self._subplotspec = subplotspec
        self._parent = parent
        self.figure = parent.figure
        self._fig_callbacks = parent._fig_callbacks

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
```
### 116 - tutorials/intermediate/legend_guide.py:

Start line: 286, End line: 305

```python
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                    edgecolor="red", linewidth=3)

fig, ax = plt.subplots()

ax.add_patch(c)
ax.legend([c], ["An ellipse, not a rectangle"],
          handler_map={mpatches.Circle: HandlerEllipse()})
```
### 119 - lib/matplotlib/figure.py:

Start line: 2236, End line: 2262

```python
@_docstring.interpd
class SubFigure(FigureBase):

    def _redo_transform_rel_fig(self, bbox=None):
        """
        Make the transSubfigure bbox relative to Figure transform.

        Parameters
        ----------
        bbox : bbox or None
            If not None, then the bbox is used for relative bounding box.
            Otherwise, it is calculated from the subplotspec.
        """
        if bbox is not None:
            self.bbox_relative.p0 = bbox.p0
            self.bbox_relative.p1 = bbox.p1
            return
        # need to figure out *where* this subplotspec is.
        gs = self._subplotspec.get_gridspec()
        wr = np.asarray(gs.get_width_ratios())
        hr = np.asarray(gs.get_height_ratios())
        dx = wr[self._subplotspec.colspan].sum() / wr.sum()
        dy = hr[self._subplotspec.rowspan].sum() / hr.sum()
        x0 = wr[:self._subplotspec.colspan.start].sum() / wr.sum()
        y0 = 1 - hr[:self._subplotspec.rowspan.stop].sum() / hr.sum()
        if self.bbox_relative is None:
            self.bbox_relative = Bbox.from_bounds(x0, y0, dx, dy)
        else:
            self.bbox_relative.p0 = (x0, y0)
            self.bbox_relative.p1 = (x0 + dx, y0 + dy)
```
### 124 - lib/matplotlib/figure.py:

Start line: 2306, End line: 2327

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
### 125 - lib/matplotlib/legend.py:

Start line: 646, End line: 661

```python
class Legend(Artist):

    def _findoffset(self, width, height, xdescent, ydescent, renderer):
        """Helper function to locate the legend."""

        if self._loc == 0:  # "best".
            x, y = self._find_best_position(width, height, renderer)
        elif self._loc in Legend.codes.values():  # Fixed location.
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
        else:  # Axes or figure coordinates.
            fx, fy = self._loc
            bbox = self.get_bbox_to_anchor()
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

        return x + xdescent, y + ydescent
```
### 127 - lib/matplotlib/legend.py:

Start line: 1160, End line: 1205

```python
# Helper functions to parse legend arguments for both `figure.legend` and
# `axes.legend`:
def _get_legend_handles(axs, legend_handler_map=None):
    """Yield artists that can be used as handles in a legend."""
    handles_original = []
    for ax in axs:
        handles_original += [
            *(a for a in ax._children
              if isinstance(a, (Line2D, Patch, Collection, Text))),
            *ax.containers]
        # support parasite axes:
        if hasattr(ax, 'parasites'):
            for axx in ax.parasites:
                handles_original += [
                    *(a for a in axx._children
                      if isinstance(a, (Line2D, Patch, Collection, Text))),
                    *axx.containers]

    handler_map = {**Legend.get_default_handler_map(),
                   **(legend_handler_map or {})}
    has_handler = Legend.get_legend_handler
    for handle in handles_original:
        label = handle.get_label()
        if label != '_nolegend_' and has_handler(handler_map, handle):
            yield handle
        elif (label and not label.startswith('_') and
                not has_handler(handler_map, handle)):
            _api.warn_external(
                             "Legend does not support handles for {0} "
                             "instances.\nSee: https://matplotlib.org/stable/"
                             "tutorials/intermediate/legend_guide.html"
                             "#implementing-a-custom-legend-handler".format(
                                 type(handle).__name__))
            continue


def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    handles = []
    labels = []
    for handle in _get_legend_handles(axs, legend_handler_map):
        label = handle.get_label()
        if label and not label.startswith('_'):
            handles.append(handle)
            labels.append(label)
    return handles, labels
```
### 129 - lib/matplotlib/figure.py:

Start line: 2288, End line: 2304

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
### 136 - lib/matplotlib/legend.py:

Start line: 863, End line: 896

```python
class Legend(Artist):

    def _auto_legend_data(self):
        """
        Return display coordinates for hit testing for "best" positioning.

        Returns
        -------
        bboxes
            List of bounding boxes of all patches.
        lines
            List of `.Path` corresponding to each line.
        offsets
            List of (x, y) offsets of all collection.
        """
        assert self.isaxes  # always holds, as this is only called internally
        bboxes = []
        lines = []
        offsets = []
        for artist in self.parent._children:
            if isinstance(artist, Line2D):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Rectangle):
                bboxes.append(
                    artist.get_bbox().transformed(artist.get_data_transform()))
            elif isinstance(artist, Patch):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Collection):
                transform, transOffset, hoffsets, _ = artist._prepare_points()
                if len(hoffsets):
                    for offset in transOffset.transform(hoffsets):
                        offsets.append(offset)

        return bboxes, lines, offsets
```
### 140 - lib/matplotlib/_constrained_layout.py:

Start line: 300, End line: 335

```python
def get_margin_from_padding(obj, *, w_pad=0, h_pad=0,
                            hspace=0, wspace=0):

    ss = obj._subplotspec
    gs = ss.get_gridspec()

    if hasattr(gs, 'hspace'):
        _hspace = (gs.hspace if gs.hspace is not None else hspace)
        _wspace = (gs.wspace if gs.wspace is not None else wspace)
    else:
        _hspace = (gs._hspace if gs._hspace is not None else hspace)
        _wspace = (gs._wspace if gs._wspace is not None else wspace)

    _wspace = _wspace / 2
    _hspace = _hspace / 2

    nrows, ncols = gs.get_geometry()
    # there are two margins for each direction.  The "cb"
    # margins are for pads and colorbars, the non-"cb" are
    # for the axes decorations (labels etc).
    margin = {'leftcb': w_pad, 'rightcb': w_pad,
              'bottomcb': h_pad, 'topcb': h_pad,
              'left': 0, 'right': 0,
              'top': 0, 'bottom': 0}
    if _wspace / ncols > w_pad:
        if ss.colspan.start > 0:
            margin['leftcb'] = _wspace / ncols
        if ss.colspan.stop < ncols:
            margin['rightcb'] = _wspace / ncols
    if _hspace / nrows > h_pad:
        if ss.rowspan.stop < nrows:
            margin['bottomcb'] = _hspace / nrows
        if ss.rowspan.start > 0:
            margin['topcb'] = _hspace / nrows

    return margin
```
### 143 - lib/matplotlib/figure.py:

Start line: 3108, End line: 3138

```python
@_docstring.interpd
class Figure(FigureBase):

    @_finalize_rasterization
    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = self._get_draw_artists(renderer)
        try:
            renderer.open_group('figure', gid=self.get_gid())
            if self.axes and self.get_layout_engine() is not None:
                try:
                    self.get_layout_engine().execute(self)
                except ValueError:
                    pass
                    # ValueError can occur when resizing a window.

            self.patch.draw(renderer)
            mimage._draw_list_compositing_images(
                renderer, self, artists, self.suppressComposite)

            for sfig in self.subfigs:
                sfig.draw(renderer)

            renderer.close_group('figure')
        finally:
            self.stale = False

        DrawEvent("draw_event", self.canvas, renderer)._process()
```
### 144 - lib/matplotlib/figure.py:

Start line: 3429, End line: 3462

```python
@_docstring.interpd
class Figure(FigureBase):

    def waitforbuttonpress(self, timeout=-1):
        """
        Blocking call to interact with the figure.

        Wait for user input and return True if a key was pressed, False if a
        mouse button was pressed and None if no input was given within
        *timeout* seconds.  Negative values deactivate *timeout*.
        """
        event = None

        def handler(ev):
            nonlocal event
            event = ev
            self.canvas.stop_event_loop()

        _blocking_input.blocking_input_loop(
            self, ["button_press_event", "key_press_event"], timeout, handler)

        return None if event is None else event.name == "key_press_event"

    @_api.deprecated("3.6", alternative="figure.get_layout_engine().execute()")
    def execute_constrained_layout(self, renderer=None):
        """
        Use ``layoutgrid`` to determine pos positions within Axes.

        See also `.set_constrained_layout_pads`.

        Returns
        -------
        layoutgrid : private debugging object
        """
        if not isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
            return None
        return self.get_layout_engine().execute(self)
```
