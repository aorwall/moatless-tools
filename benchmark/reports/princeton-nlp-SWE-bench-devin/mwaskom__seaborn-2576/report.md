# mwaskom__seaborn-2576

| **mwaskom/seaborn** | `430c1bf1fcc690f0431e6fc87b481b7b43776594` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 79222 |
| **Any found context length** | 1376 |
| **Avg pos** | 181.0 |
| **Min pos** | 1 |
| **Max pos** | 168 |
| **Top file pos** | 1 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/seaborn/regression.py b/seaborn/regression.py
--- a/seaborn/regression.py
+++ b/seaborn/regression.py
@@ -419,7 +419,8 @@ def lineplot(self, ax, kws):
 
         # Draw the regression line and confidence interval
         line, = ax.plot(grid, yhat, **kws)
-        line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
+        if not self.truncate:
+            line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
         if err_bands is not None:
             ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=.15)
 
@@ -562,13 +563,13 @@ def lmplot(
     data=None,
     hue=None, col=None, row=None,  # TODO move before data once * is enforced
     palette=None, col_wrap=None, height=5, aspect=1, markers="o",
-    sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None,
-    legend=True, legend_out=True, x_estimator=None, x_bins=None,
+    sharex=None, sharey=None, hue_order=None, col_order=None, row_order=None,
+    legend=True, legend_out=None, x_estimator=None, x_bins=None,
     x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
     units=None, seed=None, order=1, logistic=False, lowess=False,
     robust=False, logx=False, x_partial=None, y_partial=None,
     truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None,
-    line_kws=None, size=None
+    line_kws=None, facet_kws=None, size=None,
 ):
 
     # Handle deprecations
@@ -578,6 +579,22 @@ def lmplot(
                "please update your code.")
         warnings.warn(msg, UserWarning)
 
+    if facet_kws is None:
+        facet_kws = {}
+
+    def facet_kw_deprecation(key, val):
+        msg = (
+            f"{key} is deprecated from the `lmplot` function signature. "
+            "Please update your code to pass it using `facet_kws`."
+        )
+        if val is not None:
+            warnings.warn(msg, UserWarning)
+            facet_kws[key] = val
+
+    facet_kw_deprecation("sharex", sharex)
+    facet_kw_deprecation("sharey", sharey)
+    facet_kw_deprecation("legend_out", legend_out)
+
     if data is None:
         raise TypeError("Missing required keyword argument `data`.")
 
@@ -592,7 +609,7 @@ def lmplot(
         palette=palette,
         row_order=row_order, col_order=col_order, hue_order=hue_order,
         height=height, aspect=aspect, col_wrap=col_wrap,
-        sharex=sharex, sharey=sharey, legend_out=legend_out
+        **facet_kws,
     )
 
     # Add the markers here as FacetGrid has figured out how many levels of the
@@ -608,12 +625,12 @@ def lmplot(
                           "for each level of the hue variable"))
     facets.hue_kws = {"marker": markers}
 
-    # Hack to set the x limits properly, which needs to happen here
-    # because the extent of the regression estimate is determined
-    # by the limits of the plot
-    if sharex:
-        for ax in facets.axes.flat:
-            ax.scatter(data[x], np.ones(len(data)) * data[y].mean()).remove()
+    def update_datalim(data, x, y, ax, **kws):
+        xys = data[[x, y]].to_numpy().astype(float)
+        ax.update_datalim(xys, updatey=False)
+        ax.autoscale_view(scaley=False)
+
+    facets.map_dataframe(update_datalim, x=x, y=y)
 
     # Draw the regression plot on each facet
     regplot_kws = dict(
@@ -625,8 +642,6 @@ def lmplot(
         scatter_kws=scatter_kws, line_kws=line_kws,
     )
     facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)
-
-    # TODO this will need to change when we relax string requirement
     facets.set_axis_labels(x, y)
 
     # Add a legend
@@ -671,6 +686,10 @@ def lmplot(
         Markers for the scatterplot. If a list, each marker in the list will be
         used for each level of the ``hue`` variable.
     {share_xy}
+
+        .. deprecated:: 0.12.0
+            Pass using the `facet_kws` dictionary.
+
     {{hue,col,row}}_order : lists, optional
         Order for the levels of the faceting variables. By default, this will
         be the order that the levels appear in ``data`` or, if the variables
@@ -678,6 +697,10 @@ def lmplot(
     legend : bool, optional
         If ``True`` and there is a ``hue`` variable, add a legend.
     {legend_out}
+
+        .. deprecated:: 0.12.0
+            Pass using the `facet_kws` dictionary.
+
     {x_estimator}
     {x_bins}
     {x_ci}
@@ -696,6 +719,8 @@ def lmplot(
     {truncate}
     {xy_jitter}
     {scatter_line_kws}
+    facet_kws : dict
+        Dictionary of keyword arguments for :class:`FacetGrid`.
 
     See Also
     --------

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| seaborn/regression.py | 422 | 422 | 168 | 1 | 79222
| seaborn/regression.py | 565 | 571 | 2 | 1 | 2201
| seaborn/regression.py | 581 | 581 | 2 | 1 | 2201
| seaborn/regression.py | 595 | 595 | 2 | 1 | 2201
| seaborn/regression.py | 611 | 616 | 2 | 1 | 2201
| seaborn/regression.py | 628 | 629 | 2 | 1 | 2201
| seaborn/regression.py | 674 | 674 | 1 | 1 | 1376
| seaborn/regression.py | 681 | 681 | 1 | 1 | 1376
| seaborn/regression.py | 699 | 699 | 1 | 1 | 1376


## Problem Statement

```
 lmplot(sharey=False) not working
The following code behaves as if `sharey=True`.
(edit: actually, it does not behave the same, but it is still not rescaling the plots individually the way it should)

\`\`\`
df=pd.DataFrame({'x':[1,2,3,1,2,3], 'y':[4,5,2,400,500,200], 't':[1,1,1,2,2,2]}) 
sns.lmplot(data=df, x='x', y='y', col='t', sharey=False);
\`\`\`

If you do this, it suddenly works:
\`\`\`
sns.lmplot(data=df, x='x', y='y', col='t', sharex=False, sharey=False);
\`\`\`


Versions of seaborn and matplotlib:
\`\`\`
sns.__version__ 
'0.11.1'

matplotlib.__version__
'3.3.1'
\`\`\`

![image](https://user-images.githubusercontent.com/35338267/111419598-2525a900-86c0-11eb-9f22-8f0afb2f5007.png)


 lmplot(sharey=False) not working
The following code behaves as if `sharey=True`.
(edit: actually, it does not behave the same, but it is still not rescaling the plots individually the way it should)

\`\`\`
df=pd.DataFrame({'x':[1,2,3,1,2,3], 'y':[4,5,2,400,500,200], 't':[1,1,1,2,2,2]}) 
sns.lmplot(data=df, x='x', y='y', col='t', sharey=False);
\`\`\`

If you do this, it suddenly works:
\`\`\`
sns.lmplot(data=df, x='x', y='y', col='t', sharex=False, sharey=False);
\`\`\`


Versions of seaborn and matplotlib:
\`\`\`
sns.__version__ 
'0.11.1'

matplotlib.__version__
'3.3.1'
\`\`\`

![image](https://user-images.githubusercontent.com/35338267/111419598-2525a900-86c0-11eb-9f22-8f0afb2f5007.png)


Allow xlim as parameter for lmplot
Seaborn versions: latest dev version and 0.11.1

`lmplot` doesn't seem to accept the `xlim=` parameter, although FacetGrid does.

Use case: when `truncate=False`, the regression lines are extrapolated until they touch the current xlims.  If one afterwards want to extend these xlims, the regression lines are floating again.  A workaround is either to call FacetGrid and regplot separately, or to set very wide xmargins via the rcParams.

Example code.
\`\`\`
import seaborn as sns
import matplotlib as mpl

tips = sns.load_dataset('tips')
# mpl.rcParams['axes.xmargin'] = 0.5  # set very wide margins: 50% of the actual range
g = sns.lmplot(x="total_bill", y="tip", col="smoker", data=tips, truncate=False, xlim=(0, 80))
# mpl.rcParams['axes.xmargin'] = 0.05 # set the margins back to the default
g.set(xlim=(0, 80))
\`\`\`




```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 seaborn/regression.py** | 638 | 808| 1376 | 1376 | 9387 | 
| **-> 2 <-** | **1 seaborn/regression.py** | 555 | 635| 825 | 2201 | 9387 | 
| 3 | **1 seaborn/regression.py** | 841 | 1007| 1276 | 3477 | 9387 | 
| 4 | 2 examples/marginal_ticks.py | 1 | 16| 144 | 3621 | 9531 | 
| 5 | **2 seaborn/regression.py** | 1 | 23| 133 | 3754 | 9531 | 
| 6 | 3 seaborn/relational.py | 959 | 994| 409 | 4163 | 17921 | 
| 7 | 4 examples/many_facets.py | 1 | 40| 320 | 4483 | 18241 | 
| 8 | 5 examples/faceted_lineplot.py | 1 | 24| 140 | 4623 | 18381 | 
| 9 | **5 seaborn/regression.py** | 811 | 838| 290 | 4913 | 18381 | 
| 10 | 6 seaborn/distributions.py | 2140 | 2214| 583 | 5496 | 38577 | 
| 11 | 6 seaborn/relational.py | 850 | 957| 829 | 6325 | 38577 | 
| 12 | 7 seaborn/axisgrid.py | 2054 | 2218| 1469 | 7794 | 57249 | 
| 13 | 7 seaborn/distributions.py | 2216 | 2314| 862 | 8656 | 57249 | 
| 14 | 7 seaborn/relational.py | 601 | 649| 425 | 9081 | 57249 | 
| 15 | 8 examples/scatter_bubbles.py | 1 | 18| 111 | 9192 | 57360 | 
| 16 | 8 seaborn/axisgrid.py | 473 | 563| 907 | 10099 | 57360 | 
| 17 | 9 seaborn/categorical.py | 3224 | 3404| 1320 | 11419 | 90212 | 
| 18 | 9 seaborn/axisgrid.py | 2031 | 2051| 266 | 11685 | 90212 | 
| 19 | 9 seaborn/relational.py | 732 | 774| 343 | 12028 | 90212 | 
| 20 | 9 seaborn/axisgrid.py | 1961 | 2029| 742 | 12770 | 90212 | 
| 21 | 10 examples/multiple_ecdf.py | 1 | 18| 122 | 12892 | 90334 | 
| 22 | 11 examples/wide_data_lineplot.py | 1 | 20| 137 | 13029 | 90471 | 
| 23 | 11 seaborn/axisgrid.py | 287 | 471| 1516 | 14545 | 90471 | 
| 24 | 11 seaborn/axisgrid.py | 228 | 284| 503 | 15048 | 90471 | 
| 25 | 12 examples/scatterplot_sizes.py | 1 | 25| 164 | 15212 | 90635 | 
| 26 | 12 seaborn/categorical.py | 3768 | 3829| 723 | 15935 | 90635 | 
| 27 | 13 seaborn/__init__.py | 1 | 22| 236 | 16171 | 90871 | 
| 28 | 13 seaborn/relational.py | 180 | 288| 937 | 17108 | 90871 | 
| 29 | 13 seaborn/relational.py | 425 | 513| 735 | 17843 | 90871 | 
| 30 | 13 seaborn/categorical.py | 2605 | 2747| 1182 | 19025 | 90871 | 
| 31 | 14 examples/pairgrid_dotplot.py | 1 | 39| 268 | 19293 | 91139 | 
| 32 | 14 seaborn/categorical.py | 3832 | 3987| 1392 | 20685 | 91139 | 
| 33 | 14 seaborn/distributions.py | 527 | 717| 1493 | 22178 | 91139 | 
| 34 | **14 seaborn/regression.py** | 427 | 554| 1382 | 23560 | 91139 | 
| 35 | 14 seaborn/distributions.py | 876 | 1009| 937 | 24497 | 91139 | 
| 36 | 15 examples/grouped_boxplot.py | 1 | 19| 103 | 24600 | 91242 | 
| 37 | 15 seaborn/distributions.py | 2092 | 2137| 261 | 24861 | 91242 | 
| 38 | 15 seaborn/relational.py | 1 | 53| 395 | 25256 | 91242 | 
| 39 | 16 examples/logistic_regression.py | 1 | 20| 151 | 25407 | 91393 | 
| 40 | 16 seaborn/categorical.py | 3567 | 3767| 1622 | 27029 | 91393 | 
| 41 | 16 seaborn/distributions.py | 1574 | 1746| 1314 | 28343 | 91393 | 
| 42 | 17 seaborn/rcmod.py | 1 | 80| 392 | 28735 | 95153 | 
| 43 | 17 seaborn/distributions.py | 719 | 874| 1274 | 30009 | 95153 | 
| 44 | 18 examples/radial_facets.py | 1 | 28| 209 | 30218 | 95362 | 
| 45 | 18 seaborn/categorical.py | 3199 | 3221| 185 | 30403 | 95362 | 
| 46 | 18 seaborn/relational.py | 516 | 540| 194 | 30597 | 95362 | 
| 47 | 19 examples/anscombes_quartet.py | 1 | 17| 121 | 30718 | 95483 | 
| 48 | 20 examples/multiple_regression.py | 1 | 22| 123 | 30841 | 95606 | 
| 49 | 21 seaborn/_core.py | 1187 | 1268| 812 | 31653 | 108818 | 
| 50 | 21 seaborn/axisgrid.py | 1 | 27| 162 | 31815 | 108818 | 
| 51 | 21 seaborn/relational.py | 383 | 424| 472 | 32287 | 108818 | 
| 52 | 21 seaborn/categorical.py | 2750 | 2811| 422 | 32709 | 108818 | 
| 53 | **21 seaborn/regression.py** | 71 | 152| 719 | 33428 | 108818 | 
| 54 | 21 seaborn/distributions.py | 1111 | 1197| 775 | 34203 | 108818 | 
| 55 | 21 seaborn/categorical.py | 1854 | 1961| 933 | 35136 | 108818 | 
| 56 | 21 seaborn/relational.py | 55 | 177| 1169 | 36305 | 108818 | 
| 57 | 21 seaborn/categorical.py | 2875 | 2939| 435 | 36740 | 108818 | 
| 58 | 21 seaborn/categorical.py | 3407 | 3448| 270 | 37010 | 108818 | 
| 59 | 21 seaborn/distributions.py | 1 | 91| 595 | 37605 | 108818 | 
| 60 | 21 seaborn/relational.py | 350 | 381| 250 | 37855 | 108818 | 
| 61 | 21 seaborn/axisgrid.py | 1573 | 1669| 916 | 38771 | 108818 | 
| 62 | 21 seaborn/distributions.py | 2015 | 2089| 584 | 39355 | 108818 | 
| 63 | 22 examples/kde_ridgeplot.py | 1 | 49| 384 | 39739 | 109202 | 
| 64 | 22 seaborn/categorical.py | 3031 | 3196| 1198 | 40937 | 109202 | 
| 65 | 22 seaborn/categorical.py | 2347 | 2368| 178 | 41115 | 109202 | 
| 66 | 22 seaborn/relational.py | 997 | 1079| 601 | 41716 | 109202 | 
| 67 | 23 examples/regression_marginals.py | 1 | 15| 0 | 41716 | 109293 | 
| 68 | 23 seaborn/relational.py | 652 | 729| 526 | 42242 | 109293 | 
| 69 | 24 seaborn/utils.py | 323 | 390| 645 | 42887 | 114722 | 
| 70 | 24 seaborn/categorical.py | 2371 | 2602| 190 | 43077 | 114722 | 
| 71 | 24 seaborn/distributions.py | 2556 | 2647| 826 | 43903 | 114722 | 
| 72 | 25 examples/paired_pointplots.py | 1 | 21| 153 | 44056 | 114875 | 
| 73 | 26 examples/residplot.py | 1 | 17| 111 | 44167 | 114986 | 
| 74 | 26 seaborn/relational.py | 542 | 598| 461 | 44628 | 114986 | 
| 75 | 26 seaborn/categorical.py | 2214 | 2344| 963 | 45591 | 114986 | 
| 76 | 26 seaborn/categorical.py | 1753 | 1798| 400 | 45991 | 114986 | 
| 77 | 27 examples/large_distributions.py | 1 | 15| 107 | 46098 | 115093 | 
| 78 | 28 examples/grouped_violinplots.py | 1 | 18| 121 | 46219 | 115214 | 
| 79 | 28 seaborn/distributions.py | 1268 | 1300| 301 | 46520 | 115214 | 
| 80 | 29 examples/jitter_stripplot.py | 1 | 37| 287 | 46807 | 115501 | 
| 81 | 29 seaborn/distributions.py | 1347 | 1442| 608 | 47415 | 115501 | 
| 82 | 29 seaborn/categorical.py | 2187 | 2211| 166 | 47581 | 115501 | 
| 83 | 29 seaborn/distributions.py | 359 | 526| 1243 | 48824 | 115501 | 
| 84 | 30 examples/layered_bivariate_plot.py | 1 | 24| 186 | 49010 | 115687 | 
| 85 | 30 seaborn/_core.py | 403 | 480| 652 | 49662 | 115687 | 
| 86 | **30 seaborn/regression.py** | 1010 | 1096| 745 | 50407 | 115687 | 
| 87 | 30 seaborn/distributions.py | 2317 | 2396| 602 | 51009 | 115687 | 
| 88 | 30 seaborn/categorical.py | 754 | 814| 453 | 51462 | 115687 | 
| 89 | 30 seaborn/categorical.py | 3006 | 3028| 175 | 51637 | 115687 | 
| 90 | 30 seaborn/categorical.py | 281 | 374| 739 | 52376 | 115687 | 
| 91 | 31 seaborn/cm.py | 1 | 1561| 70 | 52446 | 150706 | 
| 92 | 31 seaborn/categorical.py | 156 | 188| 313 | 52759 | 150706 | 
| 93 | 31 seaborn/categorical.py | 1963 | 2027| 450 | 53209 | 150706 | 
| 94 | 31 seaborn/distributions.py | 1302 | 1344| 351 | 53560 | 150706 | 
| 95 | 32 examples/heat_scatter.py | 1 | 42| 330 | 53890 | 151036 | 
| 96 | **32 seaborn/regression.py** | 377 | 407| 349 | 54239 | 151036 | 
| 97 | 32 seaborn/categorical.py | 852 | 883| 273 | 54512 | 151036 | 
| 98 | 32 seaborn/axisgrid.py | 754 | 776| 183 | 54695 | 151036 | 
| 99 | 33 examples/joint_histogram.py | 1 | 27| 192 | 54887 | 151228 | 
| 100 | **33 seaborn/regression.py** | 266 | 289| 211 | 55098 | 151228 | 
| 101 | 33 seaborn/axisgrid.py | 605 | 685| 672 | 55770 | 151228 | 
| 102 | 34 seaborn/widgets.py | 1 | 58| 426 | 56196 | 154732 | 
| 103 | 34 seaborn/relational.py | 777 | 847| 424 | 56620 | 154732 | 
| 104 | 34 seaborn/_core.py | 342 | 401| 505 | 57125 | 154732 | 
| 105 | **34 seaborn/regression.py** | 291 | 316| 253 | 57378 | 154732 | 
| 106 | 35 examples/part_whole_bars.py | 1 | 31| 216 | 57594 | 154948 | 
| 107 | 35 seaborn/axisgrid.py | 1323 | 1406| 652 | 58246 | 154948 | 
| 108 | 35 seaborn/_core.py | 1303 | 1326| 198 | 58444 | 154948 | 
| 109 | 35 seaborn/distributions.py | 2649 | 2675| 286 | 58730 | 154948 | 
| 110 | 35 seaborn/axisgrid.py | 836 | 862| 257 | 58987 | 154948 | 
| 111 | 35 seaborn/categorical.py | 1658 | 1750| 745 | 59732 | 154948 | 
| 112 | **35 seaborn/regression.py** | 189 | 228| 361 | 60093 | 154948 | 
| 113 | 36 seaborn/matrix.py | 543 | 558| 235 | 60328 | 166704 | 
| 114 | 36 seaborn/matrix.py | 1221 | 1247| 274 | 60602 | 166704 | 
| 115 | 37 examples/horizontal_boxplot.py | 1 | 31| 192 | 60794 | 166896 | 
| 116 | 37 seaborn/matrix.py | 1416 | 1430| 281 | 61075 | 166896 | 
| 117 | 37 seaborn/axisgrid.py | 1740 | 1799| 491 | 61566 | 166896 | 
| 118 | 37 seaborn/categorical.py | 1386 | 1504| 772 | 62338 | 166896 | 
| 119 | 38 examples/grouped_barplot.py | 1 | 21| 121 | 62459 | 167017 | 
| 120 | 39 examples/different_scatter_variables.py | 1 | 26| 207 | 62666 | 167224 | 
| 121 | 39 seaborn/categorical.py | 1 | 40| 258 | 62924 | 167224 | 
| 122 | 39 seaborn/matrix.py | 1164 | 1219| 560 | 63484 | 167224 | 
| 123 | 40 examples/multiple_bivariate_kde.py | 1 | 25| 126 | 63610 | 167350 | 
| 124 | 41 seaborn/conftest.py | 45 | 145| 543 | 64153 | 168750 | 
| 125 | 41 seaborn/axisgrid.py | 2221 | 2280| 413 | 64566 | 168750 | 
| 126 | 41 seaborn/_core.py | 738 | 835| 787 | 65353 | 168750 | 
| 127 | 41 seaborn/categorical.py | 1506 | 1532| 322 | 65675 | 168750 | 
| 128 | 41 seaborn/categorical.py | 737 | 752| 127 | 65802 | 168750 | 
| 129 | 42 doc/tools/generate_logos.py | 159 | 225| 617 | 66419 | 170675 | 
| 130 | 42 seaborn/axisgrid.py | 778 | 834| 468 | 66887 | 170675 | 
| 131 | **42 seaborn/regression.py** | 341 | 375| 263 | 67150 | 170675 | 
| 132 | 42 seaborn/axisgrid.py | 1408 | 1447| 286 | 67436 | 170675 | 
| 133 | 42 seaborn/distributions.py | 1199 | 1266| 577 | 68013 | 170675 | 
| 134 | 42 seaborn/axisgrid.py | 1834 | 1875| 273 | 68286 | 170675 | 
| 135 | 43 seaborn/_docstrings.py | 137 | 199| 393 | 68679 | 172113 | 
| 136 | 43 seaborn/distributions.py | 1903 | 1956| 368 | 69047 | 172113 | 
| 137 | 43 seaborn/categorical.py | 1847 | 1852| 116 | 69163 | 172113 | 
| 138 | 44 seaborn/miscplot.py | 1 | 30| 231 | 69394 | 172491 | 
| 139 | **44 seaborn/regression.py** | 230 | 247| 170 | 69564 | 172491 | 
| 140 | 44 seaborn/cm.py | 1564 | 1585| 166 | 69730 | 172491 | 
| 141 | 44 seaborn/distributions.py | 1011 | 1110| 711 | 70441 | 172491 | 
| 142 | **44 seaborn/regression.py** | 57 | 68| 119 | 70560 | 172491 | 
| 143 | 44 seaborn/_core.py | 1 | 26| 117 | 70677 | 172491 | 
| 144 | 44 seaborn/axisgrid.py | 1299 | 1321| 185 | 70862 | 172491 | 
| 145 | 45 examples/simple_violinplots.py | 1 | 19| 121 | 70983 | 172612 | 
| 146 | 45 seaborn/matrix.py | 1069 | 1097| 306 | 71289 | 172612 | 
| 147 | 46 examples/histogram_stacked.py | 1 | 30| 152 | 71441 | 172764 | 
| 148 | 46 seaborn/axisgrid.py | 1252 | 1267| 121 | 71562 | 172764 | 
| 149 | **46 seaborn/regression.py** | 26 | 55| 235 | 71797 | 172764 | 
| 150 | **46 seaborn/regression.py** | 154 | 187| 241 | 72038 | 172764 | 
| 151 | 46 seaborn/categorical.py | 218 | 279| 503 | 72541 | 172764 | 
| 152 | 47 examples/wide_form_violinplot.py | 1 | 35| 291 | 72832 | 173055 | 
| 153 | 47 seaborn/categorical.py | 2030 | 2185| 1472 | 74304 | 173055 | 
| 154 | 47 seaborn/distributions.py | 172 | 193| 228 | 74532 | 173055 | 
| 155 | 48 examples/timeseries_facets.py | 1 | 40| 273 | 74805 | 173328 | 
| 156 | 48 seaborn/axisgrid.py | 1510 | 1553| 334 | 75139 | 173328 | 
| 157 | 48 seaborn/relational.py | 290 | 347| 420 | 75559 | 173328 | 
| 158 | 48 seaborn/categorical.py | 3545 | 3564| 172 | 75731 | 173328 | 
| 159 | 49 examples/smooth_bivariate_kde.py | 1 | 17| 131 | 75862 | 173459 | 
| 160 | 49 seaborn/_core.py | 1033 | 1070| 280 | 76142 | 173459 | 
| 161 | 49 seaborn/distributions.py | 2419 | 2554| 1120 | 77262 | 173459 | 
| 162 | 49 doc/tools/generate_logos.py | 61 | 156| 938 | 78200 | 173459 | 
| 163 | 49 seaborn/axisgrid.py | 956 | 990| 266 | 78466 | 173459 | 
| 164 | 50 examples/pointplot_anova.py | 1 | 18| 124 | 78590 | 173583 | 
| 165 | 51 examples/scatterplot_categorical.py | 1 | 17| 0 | 78590 | 173676 | 
| 166 | 51 seaborn/categorical.py | 2942 | 3003| 453 | 79043 | 173676 | 
| 167 | 52 examples/errorband_lineplots.py | 1 | 18| 0 | 79043 | 173765 | 
| **-> 168 <-** | **52 seaborn/regression.py** | 409 | 424| 179 | 79222 | 173765 | 
| 169 | 52 seaborn/rcmod.py | 179 | 303| 796 | 80018 | 173765 | 
| 170 | 52 seaborn/distributions.py | 94 | 135| 316 | 80334 | 173765 | 
| 171 | 52 seaborn/distributions.py | 1445 | 1571| 1106 | 81440 | 173765 | 
| 172 | 53 examples/palette_choices.py | 1 | 38| 314 | 81754 | 174079 | 
| 173 | 54 examples/scatterplot_matrix.py | 1 | 12| 0 | 81754 | 174127 | 
| 174 | 55 examples/palette_generation.py | 1 | 36| 282 | 82036 | 174409 | 
| 175 | 55 seaborn/categorical.py | 3451 | 3542| 608 | 82644 | 174409 | 
| 176 | 55 seaborn/axisgrid.py | 1878 | 1960| 813 | 83457 | 174409 | 
| 177 | **55 seaborn/regression.py** | 249 | 264| 139 | 83596 | 174409 | 
| 178 | **55 seaborn/regression.py** | 318 | 339| 216 | 83812 | 174409 | 
| 179 | 55 seaborn/categorical.py | 2814 | 2872| 436 | 84248 | 174409 | 
| 180 | 55 seaborn/categorical.py | 3990 | 4060| 668 | 84916 | 174409 | 
| 181 | 55 seaborn/matrix.py | 355 | 558| 106 | 85022 | 174409 | 
| 182 | 56 setup.py | 1 | 92| 699 | 85721 | 175124 | 
| 183 | 56 seaborn/matrix.py | 1 | 57| 344 | 86065 | 175124 | 
| 184 | 56 seaborn/categorical.py | 1604 | 1656| 458 | 86523 | 175124 | 
| 185 | 56 seaborn/axisgrid.py | 1449 | 1469| 241 | 86764 | 175124 | 
| 186 | 56 seaborn/rcmod.py | 338 | 436| 805 | 87569 | 175124 | 
| 187 | 57 examples/three_variable_histogram.py | 1 | 16| 0 | 87569 | 175212 | 
| 188 | 57 seaborn/axisgrid.py | 687 | 752| 534 | 88103 | 175212 | 
| 189 | 57 seaborn/axisgrid.py | 1010 | 1032| 180 | 88283 | 175212 | 
| 190 | 57 seaborn/rcmod.py | 83 | 124| 361 | 88644 | 175212 | 
| 191 | 57 seaborn/axisgrid.py | 1471 | 1508| 290 | 88934 | 175212 | 
| 192 | 58 examples/faceted_histogram.py | 1 | 15| 0 | 88934 | 175297 | 
| 193 | 59 examples/spreadsheet_heatmap.py | 1 | 17| 110 | 89044 | 175407 | 
| 194 | 59 seaborn/rcmod.py | 439 | 473| 289 | 89333 | 175407 | 
| 195 | 59 seaborn/categorical.py | 990 | 1008| 172 | 89505 | 175407 | 
| 196 | 59 seaborn/matrix.py | 1099 | 1162| 543 | 90048 | 175407 | 
| 197 | 60 seaborn/external/husl.py | 94 | 114| 301 | 90349 | 178077 | 
| 198 | 60 seaborn/categorical.py | 377 | 586| 1388 | 91737 | 178077 | 
| 199 | 60 seaborn/categorical.py | 1096 | 1272| 1222 | 92959 | 178077 | 
| 200 | 60 seaborn/categorical.py | 1322 | 1334| 188 | 93147 | 178077 | 
| 201 | 60 seaborn/distributions.py | 290 | 357| 413 | 93560 | 178077 | 
| 202 | 60 seaborn/axisgrid.py | 1269 | 1282| 117 | 93677 | 178077 | 
| 203 | 60 seaborn/axisgrid.py | 1083 | 1250| 1478 | 95155 | 178077 | 
| 204 | 60 seaborn/axisgrid.py | 169 | 187| 165 | 95320 | 178077 | 
| 205 | 60 seaborn/matrix.py | 683 | 737| 513 | 95833 | 178077 | 
| 206 | 60 seaborn/categorical.py | 4158 | 4185| 245 | 96078 | 178077 | 
| 207 | 60 seaborn/_core.py | 1072 | 1107| 296 | 96374 | 178077 | 
| 208 | 60 seaborn/categorical.py | 816 | 849| 281 | 96655 | 178077 | 
| 209 | 60 seaborn/categorical.py | 694 | 734| 338 | 96993 | 178077 | 
| 210 | 61 examples/hexbin_marginals.py | 1 | 16| 0 | 96993 | 178169 | 
| 211 | 61 seaborn/axisgrid.py | 30 | 63| 275 | 97268 | 178169 | 
| 212 | 61 seaborn/distributions.py | 195 | 227| 406 | 97674 | 178169 | 
| 213 | 62 seaborn/_testing.py | 50 | 72| 186 | 97860 | 178767 | 
| 214 | 63 examples/many_pairwise_correlations.py | 1 | 35| 230 | 98090 | 178997 | 
| 215 | 63 seaborn/matrix.py | 999 | 1015| 128 | 98218 | 178997 | 
| 216 | 63 seaborn/axisgrid.py | 1704 | 1738| 275 | 98493 | 178997 | 
| 217 | 63 seaborn/axisgrid.py | 864 | 954| 802 | 99295 | 178997 | 
| 218 | 63 seaborn/categorical.py | 672 | 692| 157 | 99452 | 178997 | 
| 219 | 63 seaborn/axisgrid.py | 1069 | 1082| 132 | 99584 | 178997 | 
| 220 | 63 seaborn/categorical.py | 1336 | 1354| 174 | 99758 | 178997 | 
| 221 | 63 seaborn/categorical.py | 94 | 141| 557 | 100315 | 178997 | 
| 222 | 63 seaborn/categorical.py | 143 | 154| 126 | 100441 | 178997 | 
| 223 | 63 seaborn/categorical.py | 190 | 216| 252 | 100693 | 178997 | 
| 224 | 63 seaborn/categorical.py | 1554 | 1601| 357 | 101050 | 178997 | 
| 225 | 63 seaborn/axisgrid.py | 1284 | 1297| 117 | 101167 | 178997 | 
| 226 | 63 seaborn/widgets.py | 359 | 383| 236 | 101403 | 178997 | 
| 227 | 63 seaborn/distributions.py | 229 | 288| 530 | 101933 | 178997 | 
| 228 | 63 seaborn/_core.py | 259 | 340| 490 | 102423 | 178997 | 
| 229 | 63 seaborn/utils.py | 694 | 706| 147 | 102570 | 178997 | 
| 230 | 64 seaborn/palettes.py | 47 | 57| 152 | 102722 | 187353 | 
| 231 | 64 seaborn/categorical.py | 4062 | 4092| 268 | 102990 | 187353 | 
| 232 | 64 seaborn/categorical.py | 43 | 92| 519 | 103509 | 187353 | 
| 233 | 64 seaborn/rcmod.py | 476 | 502| 192 | 103701 | 187353 | 
| 234 | 64 seaborn/categorical.py | 1288 | 1320| 330 | 104031 | 187353 | 
| 235 | 64 seaborn/_core.py | 1290 | 1301| 187 | 104218 | 187353 | 
| 236 | 64 seaborn/axisgrid.py | 1051 | 1066| 157 | 104375 | 187353 | 
| 237 | 64 seaborn/axisgrid.py | 992 | 1008| 161 | 104536 | 187353 | 
| 238 | 64 seaborn/matrix.py | 265 | 276| 123 | 104659 | 187353 | 
| 239 | 64 seaborn/external/husl.py | 238 | 314| 555 | 105214 | 187353 | 
| 240 | 64 seaborn/palettes.py | 980 | 1039| 486 | 105700 | 187353 | 
| 241 | 65 seaborn/_statistics.py | 106 | 122| 164 | 105864 | 191605 | 
| 242 | 65 seaborn/external/husl.py | 117 | 153| 266 | 106130 | 191605 | 
| 243 | 65 seaborn/distributions.py | 1959 | 2012| 340 | 106470 | 191605 | 
| 244 | 65 seaborn/matrix.py | 561 | 630| 480 | 106950 | 191605 | 
| 245 | 65 seaborn/categorical.py | 613 | 670| 493 | 107443 | 191605 | 
| 246 | 65 seaborn/matrix.py | 1250 | 1415| 1467 | 108910 | 191605 | 
| 247 | 65 seaborn/distributions.py | 1749 | 1900| 1275 | 110185 | 191605 | 
| 248 | 65 seaborn/matrix.py | 366 | 542| 1588 | 111773 | 191605 | 
| 249 | 65 seaborn/axisgrid.py | 1555 | 1570| 142 | 111915 | 191605 | 
| 250 | 65 seaborn/axisgrid.py | 1801 | 1831| 210 | 112125 | 191605 | 


### Hint

```
Worth noting: the y axes are not shared in the "wrong" plot, however the y axis autoscaling is off.

My suspicion is that this line is the culprit: https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py#L611-L616
"the y axes are not shared in the "wrong" plot"

You are right, the scales aren't actually identical. I didn't notice that.
It's fortunate as it makes the workaround (setting the ylim explicitly) a lot easier to accomplish than "unsharing" the axes, which is pretty difficult in matplotlib IIRC.
Worth noting: the y axes are not shared in the "wrong" plot, however the y axis autoscaling is off.

My suspicion is that this line is the culprit: https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py#L611-L616
"the y axes are not shared in the "wrong" plot"

You are right, the scales aren't actually identical. I didn't notice that.
It's fortunate as it makes the workaround (setting the ylim explicitly) a lot easier to accomplish than "unsharing" the axes, which is pretty difficult in matplotlib IIRC.
What should really happen is that `lmplot` should accept a `facet_kws` dictionary that it passes to `FacetGrid` to set it up. Also then some of the parameters of lmplot that are passed directly should be deprecated with the instruction that they should be packaged in `facet_kws` (not all of them, but less-often-used ones).

Unfortunately I have not been especially consistent across the figure-level functions with which `FacetGrid` parameters do or do not end up i the figure-level function signature. This would probably be good to standardize, but that might involve a lot of annoying deprecation.
```

## Patch

```diff
diff --git a/seaborn/regression.py b/seaborn/regression.py
--- a/seaborn/regression.py
+++ b/seaborn/regression.py
@@ -419,7 +419,8 @@ def lineplot(self, ax, kws):
 
         # Draw the regression line and confidence interval
         line, = ax.plot(grid, yhat, **kws)
-        line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
+        if not self.truncate:
+            line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
         if err_bands is not None:
             ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=.15)
 
@@ -562,13 +563,13 @@ def lmplot(
     data=None,
     hue=None, col=None, row=None,  # TODO move before data once * is enforced
     palette=None, col_wrap=None, height=5, aspect=1, markers="o",
-    sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None,
-    legend=True, legend_out=True, x_estimator=None, x_bins=None,
+    sharex=None, sharey=None, hue_order=None, col_order=None, row_order=None,
+    legend=True, legend_out=None, x_estimator=None, x_bins=None,
     x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
     units=None, seed=None, order=1, logistic=False, lowess=False,
     robust=False, logx=False, x_partial=None, y_partial=None,
     truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None,
-    line_kws=None, size=None
+    line_kws=None, facet_kws=None, size=None,
 ):
 
     # Handle deprecations
@@ -578,6 +579,22 @@ def lmplot(
                "please update your code.")
         warnings.warn(msg, UserWarning)
 
+    if facet_kws is None:
+        facet_kws = {}
+
+    def facet_kw_deprecation(key, val):
+        msg = (
+            f"{key} is deprecated from the `lmplot` function signature. "
+            "Please update your code to pass it using `facet_kws`."
+        )
+        if val is not None:
+            warnings.warn(msg, UserWarning)
+            facet_kws[key] = val
+
+    facet_kw_deprecation("sharex", sharex)
+    facet_kw_deprecation("sharey", sharey)
+    facet_kw_deprecation("legend_out", legend_out)
+
     if data is None:
         raise TypeError("Missing required keyword argument `data`.")
 
@@ -592,7 +609,7 @@ def lmplot(
         palette=palette,
         row_order=row_order, col_order=col_order, hue_order=hue_order,
         height=height, aspect=aspect, col_wrap=col_wrap,
-        sharex=sharex, sharey=sharey, legend_out=legend_out
+        **facet_kws,
     )
 
     # Add the markers here as FacetGrid has figured out how many levels of the
@@ -608,12 +625,12 @@ def lmplot(
                           "for each level of the hue variable"))
     facets.hue_kws = {"marker": markers}
 
-    # Hack to set the x limits properly, which needs to happen here
-    # because the extent of the regression estimate is determined
-    # by the limits of the plot
-    if sharex:
-        for ax in facets.axes.flat:
-            ax.scatter(data[x], np.ones(len(data)) * data[y].mean()).remove()
+    def update_datalim(data, x, y, ax, **kws):
+        xys = data[[x, y]].to_numpy().astype(float)
+        ax.update_datalim(xys, updatey=False)
+        ax.autoscale_view(scaley=False)
+
+    facets.map_dataframe(update_datalim, x=x, y=y)
 
     # Draw the regression plot on each facet
     regplot_kws = dict(
@@ -625,8 +642,6 @@ def lmplot(
         scatter_kws=scatter_kws, line_kws=line_kws,
     )
     facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)
-
-    # TODO this will need to change when we relax string requirement
     facets.set_axis_labels(x, y)
 
     # Add a legend
@@ -671,6 +686,10 @@ def lmplot(
         Markers for the scatterplot. If a list, each marker in the list will be
         used for each level of the ``hue`` variable.
     {share_xy}
+
+        .. deprecated:: 0.12.0
+            Pass using the `facet_kws` dictionary.
+
     {{hue,col,row}}_order : lists, optional
         Order for the levels of the faceting variables. By default, this will
         be the order that the levels appear in ``data`` or, if the variables
@@ -678,6 +697,10 @@ def lmplot(
     legend : bool, optional
         If ``True`` and there is a ``hue`` variable, add a legend.
     {legend_out}
+
+        .. deprecated:: 0.12.0
+            Pass using the `facet_kws` dictionary.
+
     {x_estimator}
     {x_bins}
     {x_ci}
@@ -696,6 +719,8 @@ def lmplot(
     {truncate}
     {xy_jitter}
     {scatter_line_kws}
+    facet_kws : dict
+        Dictionary of keyword arguments for :class:`FacetGrid`.
 
     See Also
     --------

```

## Test Patch

```diff
diff --git a/seaborn/tests/test_regression.py b/seaborn/tests/test_regression.py
--- a/seaborn/tests/test_regression.py
+++ b/seaborn/tests/test_regression.py
@@ -1,3 +1,4 @@
+from distutils.version import LooseVersion
 import numpy as np
 import matplotlib as mpl
 import matplotlib.pyplot as plt
@@ -596,6 +597,44 @@ def test_lmplot_scatter_kws(self):
         npt.assert_array_equal(red, red_scatter.get_facecolors()[0, :3])
         npt.assert_array_equal(blue, blue_scatter.get_facecolors()[0, :3])
 
+    @pytest.mark.skipif(LooseVersion(mpl.__version__) < "3.4",
+                        reason="MPL bug #15967")
+    @pytest.mark.parametrize("sharex", [True, False])
+    def test_lmplot_facet_truncate(self, sharex):
+
+        g = lm.lmplot(
+            data=self.df, x="x", y="y", hue="g", col="h",
+            truncate=False, facet_kws=dict(sharex=sharex),
+        )
+
+        for ax in g.axes.flat:
+            for line in ax.lines:
+                xdata = line.get_xdata()
+                assert ax.get_xlim() == tuple(xdata[[0, -1]])
+
+    def test_lmplot_sharey(self):
+
+        df = pd.DataFrame(dict(
+            x=[0, 1, 2, 0, 1, 2],
+            y=[1, -1, 0, -100, 200, 0],
+            z=["a", "a", "a", "b", "b", "b"],
+        ))
+
+        with pytest.warns(UserWarning):
+            g = lm.lmplot(data=df, x="x", y="y", col="z", sharey=False)
+        ax1, ax2 = g.axes.flat
+        assert ax1.get_ylim()[0] > ax2.get_ylim()[0]
+        assert ax1.get_ylim()[1] < ax2.get_ylim()[1]
+
+    def test_lmplot_facet_kws(self):
+
+        xlim = -4, 20
+        g = lm.lmplot(
+            data=self.df, x="x", y="y", col="h", facet_kws={"xlim": xlim}
+        )
+        for ax in g.axes.flat:
+            assert ax.get_xlim() == xlim
+
     def test_residplot(self):
 
         x, y = self.df.x, self.df.y

```


## Code snippets

### 1 - seaborn/regression.py:

Start line: 638, End line: 808

```python
lmplot.__doc__ = dedent("""\
    Plot data and regression model fits across a FacetGrid.

    This function combines :func:`regplot` and :class:`FacetGrid`. It is
    intended as a convenient interface to fit regression models across
    conditional subsets of a dataset.

    When thinking about how to assign variables to different facets, a general
    rule is that it makes sense to use ``hue`` for the most important
    comparison, followed by ``col`` and ``row``. However, always think about
    your particular dataset and the goals of the visualization you are
    creating.

    {model_api}

    The parameters to this function span most of the options in
    :class:`FacetGrid`, although there may be occasional cases where you will
    want to use that class and :func:`regplot` directly.

    Parameters
    ----------
    x, y : strings, optional
        Input variables; these should be column names in ``data``.
    {data}
    hue, col, row : strings
        Variables that define subsets of the data, which will be drawn on
        separate facets in the grid. See the ``*_order`` parameters to control
        the order of levels of this variable.
    {palette}
    {col_wrap}
    {height}
    {aspect}
    markers : matplotlib marker code or list of marker codes, optional
        Markers for the scatterplot. If a list, each marker in the list will be
        used for each level of the ``hue`` variable.
    {share_xy}
    {{hue,col,row}}_order : lists, optional
        Order for the levels of the faceting variables. By default, this will
        be the order that the levels appear in ``data`` or, if the variables
        are pandas categoricals, the category order.
    legend : bool, optional
        If ``True`` and there is a ``hue`` variable, add a legend.
    {legend_out}
    {x_estimator}
    {x_bins}
    {x_ci}
    {scatter}
    {fit_reg}
    {ci}
    {n_boot}
    {units}
    {seed}
    {order}
    {logistic}
    {lowess}
    {robust}
    {logx}
    {xy_partial}
    {truncate}
    {xy_jitter}
    {scatter_line_kws}

    See Also
    --------
    regplot : Plot data and a conditional model fit.
    FacetGrid : Subplot grid for plotting conditional relationships.
    pairplot : Combine :func:`regplot` and :class:`PairGrid` (when used with
               ``kind="reg"``).

    Notes
    -----

    {regplot_vs_lmplot}

    Examples
    --------

    These examples focus on basic regression model plots to exhibit the
    various faceting options; see the :func:`regplot` docs for demonstrations
    of the other options for plotting the data and models. There are also
    other examples for how to manipulate plot using the returned object on
    the :class:`FacetGrid` docs.

    Plot a simple linear relationship between two variables:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set_theme(color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.lmplot(x="total_bill", y="tip", data=tips)

    Condition on a third variable and plot the levels in different colors:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

    Use different markers as well as colors so the plot will reproduce to
    black-and-white more easily:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
        ...                markers=["o", "x"])

    Use a different color palette:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
        ...                palette="Set1")

    Map ``hue`` levels to colors with a dictionary:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
        ...                palette=dict(Yes="g", No="m"))

    Plot the levels of the third variable across different columns:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", col="smoker", data=tips)

    Change the height and aspect ratio of the facets:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="size", y="total_bill", hue="day", col="day",
        ...                data=tips, height=6, aspect=.4, x_jitter=.1)

    Wrap the levels of the column variable into multiple rows:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", col="day", hue="day",
        ...                data=tips, col_wrap=2, height=3)

    Condition on two variables to make a full grid:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", row="sex", col="time",
        ...                data=tips, height=3)

    Use methods on the returned :class:`FacetGrid` instance to further tweak
    the plot:

    .. plot::
        :context: close-figs

        >>> g = sns.lmplot(x="total_bill", y="tip", row="sex", col="time",
        ...                data=tips, height=3)
        >>> g = (g.set_axis_labels("Total bill (US Dollars)", "Tip")
        ...       .set(xlim=(0, 60), ylim=(0, 12),
        ...            xticks=[10, 30, 50], yticks=[2, 6, 10])
        ...       .fig.subplots_adjust(wspace=.02))



    """).format(**_regression_docs)
```
### 2 - seaborn/regression.py:

Start line: 555, End line: 635

```python
_regression_docs.update(_facet_docs)


@_deprecate_positional_args
def lmplot(
    *,
    x=None, y=None,
    data=None,
    hue=None, col=None, row=None,  # TODO move before data once * is enforced
    palette=None, col_wrap=None, height=5, aspect=1, markers="o",
    sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None,
    legend=True, legend_out=True, x_estimator=None, x_bins=None,
    x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
    units=None, seed=None, order=1, logistic=False, lowess=False,
    robust=False, logx=False, x_partial=None, y_partial=None,
    truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None,
    line_kws=None, size=None
):

    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    if data is None:
        raise TypeError("Missing required keyword argument `data`.")

    # Reduce the dataframe to only needed columns
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = np.unique([a for a in need_cols if a is not None]).tolist()
    data = data[cols]

    # Initialize the grid
    facets = FacetGrid(
        data, row=row, col=col, hue=hue,
        palette=palette,
        row_order=row_order, col_order=col_order, hue_order=hue_order,
        height=height, aspect=aspect, col_wrap=col_wrap,
        sharex=sharex, sharey=sharey, legend_out=legend_out
    )

    # Add the markers here as FacetGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if facets.hue_names is None:
        n_markers = 1
    else:
        n_markers = len(facets.hue_names)
    if not isinstance(markers, list):
        markers = [markers] * n_markers
    if len(markers) != n_markers:
        raise ValueError(("markers must be a singeton or a list of markers "
                          "for each level of the hue variable"))
    facets.hue_kws = {"marker": markers}

    # Hack to set the x limits properly, which needs to happen here
    # because the extent of the regression estimate is determined
    # by the limits of the plot
    if sharex:
        for ax in facets.axes.flat:
            ax.scatter(data[x], np.ones(len(data)) * data[y].mean()).remove()

    # Draw the regression plot on each facet
    regplot_kws = dict(
        x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
        scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
        seed=seed, order=order, logistic=logistic, lowess=lowess,
        robust=robust, logx=logx, x_partial=x_partial, y_partial=y_partial,
        truncate=truncate, x_jitter=x_jitter, y_jitter=y_jitter,
        scatter_kws=scatter_kws, line_kws=line_kws,
    )
    facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)

    # TODO this will need to change when we relax string requirement
    facets.set_axis_labels(x, y)

    # Add a legend
    if legend and (hue is not None) and (hue not in [col, row]):
        facets.add_legend()
    return facets
```
### 3 - seaborn/regression.py:

Start line: 841, End line: 1007

```python
regplot.__doc__ = dedent("""\
    Plot data and a linear regression model fit.

    {model_api}

    Parameters
    ----------
    x, y: string, series, or vector array
        Input variables. If strings, these should correspond with column names
        in ``data``. When pandas objects are used, axes will be labeled with
        the series name.
    {data}
    {x_estimator}
    {x_bins}
    {x_ci}
    {scatter}
    {fit_reg}
    {ci}
    {n_boot}
    {units}
    {seed}
    {order}
    {logistic}
    {lowess}
    {robust}
    {logx}
    {xy_partial}
    {truncate}
    {xy_jitter}
    label : string
        Label to apply to either the scatterplot or regression line (if
        ``scatter`` is ``False``) for use in a legend.
    color : matplotlib color
        Color to apply to all plot elements; will be superseded by colors
        passed in ``scatter_kws`` or ``line_kws``.
    marker : matplotlib marker code
        Marker to use for the scatterplot glyphs.
    {scatter_line_kws}
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax : matplotlib Axes
        The Axes object containing the plot.

    See Also
    --------
    lmplot : Combine :func:`regplot` and :class:`FacetGrid` to plot multiple
             linear relationships in a dataset.
    jointplot : Combine :func:`regplot` and :class:`JointGrid` (when used with
                ``kind="reg"``).
    pairplot : Combine :func:`regplot` and :class:`PairGrid` (when used with
               ``kind="reg"``).
    residplot : Plot the residuals of a linear regression model.

    Notes
    -----

    {regplot_vs_lmplot}


    It's also easy to combine combine :func:`regplot` and :class:`JointGrid` or
    :class:`PairGrid` through the :func:`jointplot` and :func:`pairplot`
    functions, although these do not directly accept all of :func:`regplot`'s
    parameters.

    Examples
    --------

    Plot the relationship between two variables in a DataFrame:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set_theme(color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.regplot(x="total_bill", y="tip", data=tips)

    Plot with two variables defined as numpy arrays; use a different color:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(8)
        >>> mean, cov = [4, 6], [(1.5, .7), (.7, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, 80).T
        >>> ax = sns.regplot(x=x, y=y, color="g")

    Plot with two variables defined as pandas Series; use a different marker:

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> x, y = pd.Series(x, name="x_var"), pd.Series(y, name="y_var")
        >>> ax = sns.regplot(x=x, y=y, marker="+")

    Use a 68% confidence interval, which corresponds with the standard error
    of the estimate, and extend the regression line to the axis limits:

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x=x, y=y, ci=68, truncate=False)

    Plot with a discrete ``x`` variable and add some jitter:

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x="size", y="total_bill", data=tips, x_jitter=.1)

    Plot with a discrete ``x`` variable showing means and confidence intervals
    for unique values:

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x="size", y="total_bill", data=tips,
        ...                  x_estimator=np.mean)

    Plot with a continuous variable divided into discrete bins:

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x=x, y=y, x_bins=4)

    Fit a higher-order polynomial regression:

    .. plot::
        :context: close-figs

        >>> ans = sns.load_dataset("anscombe")
        >>> ax = sns.regplot(x="x", y="y", data=ans.loc[ans.dataset == "II"],
        ...                  scatter_kws={{"s": 80}},
        ...                  order=2, ci=None)

    Fit a robust regression and don't plot a confidence interval:

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x="x", y="y", data=ans.loc[ans.dataset == "III"],
        ...                  scatter_kws={{"s": 80}},
        ...                  robust=True, ci=None)

    Fit a logistic regression; jitter the y variable and use fewer bootstrap
    iterations:

    .. plot::
        :context: close-figs

        >>> tips["big_tip"] = (tips.tip / tips.total_bill) > .175
        >>> ax = sns.regplot(x="total_bill", y="big_tip", data=tips,
        ...                  logistic=True, n_boot=500, y_jitter=.03)

    Fit the regression model using log(x):

    .. plot::
        :context: close-figs

        >>> ax = sns.regplot(x="size", y="total_bill", data=tips,
        ...                  x_estimator=np.mean, logx=True)

    """).format(**_regression_docs)
```
### 4 - examples/marginal_ticks.py:

Start line: 1, End line: 16

```python
"""
Scatterplot with marginal ticks
===============================

_thumb: .66, .34
"""
import seaborn as sns
sns.set_theme(style="white", color_codes=True)
mpg = sns.load_dataset("mpg")

# Use JointGrid directly to draw a custom plot
g = sns.JointGrid(data=mpg, x="mpg", y="acceleration", space=0, ratio=17)
g.plot_joint(sns.scatterplot, size=mpg["horsepower"], sizes=(30, 120),
             color="g", alpha=.6, legend=False)
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)
```
### 5 - seaborn/regression.py:

Start line: 1, End line: 23

```python
"""Plotting functions for linear models (broadly construed)."""
import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import statsmodels
    assert statsmodels
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
from ._decorators import _deprecate_positional_args


__all__ = ["lmplot", "regplot", "residplot"]
```
### 6 - seaborn/relational.py:

Start line: 959, End line: 994

```python
@_deprecate_positional_args
def relplot(
    *,
    x=None, y=None,
    hue=None, size=None, style=None, data=None,
    row=None, col=None,
    col_wrap=None, row_order=None, col_order=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=None, dashes=None, style_order=None,
    legend="auto", kind="scatter",
    height=5, aspect=1, facet_kws=None,
    units=None,
    **kwargs
):

    # Pass the row/col variables to FacetGrid with their original
    # names so that the axes titles render correctly
    grid_kws = {v: p.variables.get(v, None) for v in grid_semantics}
    full_data = p.plot_data.rename(columns=grid_kws)

    # Set up the FacetGrid object
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    facet_kws.update(grid_kws)
    g = FacetGrid(
        data=full_data,
        col_wrap=col_wrap, row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet_kws
    )

    # Draw the plot
    g.map_dataframe(func, **plot_kws)

    # Label the axes
    g.set_axis_labels(
        variables.get("x", None), variables.get("y", None)
    )

    # Show the legend
    if legend:
        # Replace the original plot data so the legend uses
        # numeric data with the correct type
        p.plot_data = plot_data
        p.add_legend_data(g.axes.flat[0])
        if p.legend_data:
            g.add_legend(legend_data=p.legend_data,
                         label_order=p.legend_order,
                         title=p.legend_title,
                         adjust_subtitles=True)

    return g
```
### 7 - examples/many_facets.py:

Start line: 1, End line: 40

```python
"""
Plotting on a large number of facets
====================================

_thumb: .4, .3

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

# Create a dataset with many short random walks
rs = np.random.RandomState(4)
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)
df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
```
### 8 - examples/faceted_lineplot.py:

Start line: 1, End line: 24

```python
"""
Line plots on multiple facets
=============================

_thumb: .48, .42

"""
import seaborn as sns
sns.set_theme(style="ticks")

dots = sns.load_dataset("dots")

# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.relplot(
    data=dots,
    x="time", y="firing_rate",
    hue="coherence", size="choice", col="align",
    kind="line", size_order=["T1", "T2"], palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
```
### 9 - seaborn/regression.py:

Start line: 811, End line: 838

```python
@_deprecate_positional_args
def regplot(
    *,
    x=None, y=None,
    data=None,
    x_estimator=None, x_bins=None, x_ci="ci",
    scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
    seed=None, order=1, logistic=False, lowess=False, robust=False,
    logx=False, x_partial=None, y_partial=None,
    truncate=True, dropna=True, x_jitter=None, y_jitter=None,
    label=None, color=None, marker="o",
    scatter_kws=None, line_kws=None, ax=None
):

    plotter = _RegressionPlotter(x, y, data, x_estimator, x_bins, x_ci,
                                 scatter, fit_reg, ci, n_boot, units, seed,
                                 order, logistic, lowess, robust, logx,
                                 x_partial, y_partial, truncate, dropna,
                                 x_jitter, y_jitter, color, label)

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax
```
### 10 - seaborn/distributions.py:

Start line: 2140, End line: 2214

```python
def displot(
    data=None, *,
    # Vector variables
    x=None, y=None, hue=None, row=None, col=None, weights=None,
    # Other plot parameters
    kind="hist", rug=False, rug_kws=None, log_scale=None, legend=True,
    # Hue-mapping parameters
    palette=None, hue_order=None, hue_norm=None, color=None,
    # Faceting parameters
    col_wrap=None, row_order=None, col_order=None,
    height=5, aspect=1, facet_kws=None,
    **kwargs,
):

    p = _DistributionFacetPlotter(
        data=data,
        variables=_DistributionFacetPlotter.get_semantics(locals())
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    _check_argument("kind", ["hist", "kde", "ecdf"], kind)

    # --- Initialize the FacetGrid object

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = (
            "`displot` is a figure-level function and does not accept "
            "the ax= parameter. You may wish to try {}plot.".format(kind)
        )
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    for var in ["row", "col"]:
        # Handle faceting variables that lack name information
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f"_{var}_"

    # Adapt the plot_data dataframe for use with FacetGrid
    data = p.plot_data.rename(columns=p.variables)
    data = data.loc[:, ~data.columns.duplicated()]

    col_name = p.variables.get("col", None)
    row_name = p.variables.get("row", None)

    if facet_kws is None:
        facet_kws = {}

    g = FacetGrid(
        data=data, row=row_name, col=col_name,
        col_wrap=col_wrap, row_order=row_order,
        col_order=col_order, height=height,
        aspect=aspect,
        **facet_kws,
    )

    # Now attach the axes object to the plotter object
    if kind == "kde":
        allowed_types = ["numeric", "datetime"]
    else:
        allowed_types = None
    p._attach(g, allowed_types=allowed_types, log_scale=log_scale)

    # Check for a specification that lacks x/y data and return early
    if not p.has_xy_data:
        return g

    if color is None and hue is None:
        color = "C0"
    # XXX else warn if hue is not None?

    kwargs["legend"] = legend

    # --- Draw the plots
    # ... other code
```
### 34 - seaborn/regression.py:

Start line: 427, End line: 554

```python
_regression_docs = dict(

    model_api=dedent("""\
    There are a number of mutually exclusive options for estimating the
    regression model. See the :ref:`tutorial <regression_tutorial>` for more
    information.\
    """),
    regplot_vs_lmplot=dedent("""\
    The :func:`regplot` and :func:`lmplot` functions are closely related, but
    the former is an axes-level function while the latter is a figure-level
    function that combines :func:`regplot` and :class:`FacetGrid`.\
    """),
    x_estimator=dedent("""\
    x_estimator : callable that maps vector -> scalar, optional
        Apply this function to each unique value of ``x`` and plot the
        resulting estimate. This is useful when ``x`` is a discrete variable.
        If ``x_ci`` is given, this estimate will be bootstrapped and a
        confidence interval will be drawn.\
    """),
    x_bins=dedent("""\
    x_bins : int or vector, optional
        Bin the ``x`` variable into discrete bins and then estimate the central
        tendency and a confidence interval. This binning only influences how
        the scatterplot is drawn; the regression is still fit to the original
        data.  This parameter is interpreted either as the number of
        evenly-sized (not necessary spaced) bins or the positions of the bin
        centers. When this parameter is used, it implies that the default of
        ``x_estimator`` is ``numpy.mean``.\
    """),
    x_ci=dedent("""\
    x_ci : "ci", "sd", int in [0, 100] or None, optional
        Size of the confidence interval used when plotting a central tendency
        for discrete values of ``x``. If ``"ci"``, defer to the value of the
        ``ci`` parameter. If ``"sd"``, skip bootstrapping and show the
        standard deviation of the observations in each bin.\
    """),
    scatter=dedent("""\
    scatter : bool, optional
        If ``True``, draw a scatterplot with the underlying observations (or
        the ``x_estimator`` values).\
    """),
    fit_reg=dedent("""\
    fit_reg : bool, optional
        If ``True``, estimate and plot a regression model relating the ``x``
        and ``y`` variables.\
    """),
    ci=dedent("""\
    ci : int in [0, 100] or None, optional
        Size of the confidence interval for the regression estimate. This will
        be drawn using translucent bands around the regression line. The
        confidence interval is estimated using a bootstrap; for large
        datasets, it may be advisable to avoid that computation by setting
        this parameter to None.\
    """),
    n_boot=dedent("""\
    n_boot : int, optional
        Number of bootstrap resamples used to estimate the ``ci``. The default
        value attempts to balance time and stability; you may want to increase
        this value for "final" versions of plots.\
    """),
    units=dedent("""\
    units : variable name in ``data``, optional
        If the ``x`` and ``y`` observations are nested within sampling units,
        those can be specified here. This will be taken into account when
        computing the confidence intervals by performing a multilevel bootstrap
        that resamples both units and observations (within unit). This does not
        otherwise influence how the regression is estimated or drawn.\
    """),
    seed=dedent("""\
    seed : int, numpy.random.Generator, or numpy.random.RandomState, optional
        Seed or random number generator for reproducible bootstrapping.\
    """),
    order=dedent("""\
    order : int, optional
        If ``order`` is greater than 1, use ``numpy.polyfit`` to estimate a
        polynomial regression.\
    """),
    logistic=dedent("""\
    logistic : bool, optional
        If ``True``, assume that ``y`` is a binary variable and use
        ``statsmodels`` to estimate a logistic regression model. Note that this
        is substantially more computationally intensive than linear regression,
        so you may wish to decrease the number of bootstrap resamples
        (``n_boot``) or set ``ci`` to None.\
    """),
    lowess=dedent("""\
    lowess : bool, optional
        If ``True``, use ``statsmodels`` to estimate a nonparametric lowess
        model (locally weighted linear regression). Note that confidence
        intervals cannot currently be drawn for this kind of model.\
    """),
    robust=dedent("""\
    robust : bool, optional
        If ``True``, use ``statsmodels`` to estimate a robust regression. This
        will de-weight outliers. Note that this is substantially more
        computationally intensive than standard linear regression, so you may
        wish to decrease the number of bootstrap resamples (``n_boot``) or set
        ``ci`` to None.\
    """),
    logx=dedent("""\
    logx : bool, optional
        If ``True``, estimate a linear regression of the form y ~ log(x), but
        plot the scatterplot and regression model in the input space. Note that
        ``x`` must be positive for this to work.\
    """),
    xy_partial=dedent("""\
    {x,y}_partial : strings in ``data`` or matrices
        Confounding variables to regress out of the ``x`` or ``y`` variables
        before plotting.\
    """),
    truncate=dedent("""\
    truncate : bool, optional
        If ``True``, the regression line is bounded by the data limits. If
        ``False``, it extends to the ``x`` axis limits.
    """),
    xy_jitter=dedent("""\
    {x,y}_jitter : floats, optional
        Add uniform random noise of this size to either the ``x`` or ``y``
        variables. The noise is added to a copy of the data after fitting the
        regression, and only influences the look of the scatterplot. This can
        be helpful when plotting variables that take discrete values.\
    """),
    scatter_line_kws=dedent("""\
    {scatter,line}_kws : dictionaries
        Additional keyword arguments to pass to ``plt.scatter`` and
        ``plt.plot``.\
    """),
)
```
### 53 - seaborn/regression.py:

Start line: 71, End line: 152

```python
class _RegressionPlotter(_LinearPlotter):
    """Plotter for numeric independent variables with regression model.

    This does the computations and drawing for the `regplot` function, and
    is thus also used indirectly by `lmplot`.
    """
    def __init__(self, x, y, data=None, x_estimator=None, x_bins=None,
                 x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
                 units=None, seed=None, order=1, logistic=False, lowess=False,
                 robust=False, logx=False, x_partial=None, y_partial=None,
                 truncate=False, dropna=True, x_jitter=None, y_jitter=None,
                 color=None, label=None):

        # Set member attributes
        self.x_estimator = x_estimator
        self.ci = ci
        self.x_ci = ci if x_ci == "ci" else x_ci
        self.n_boot = n_boot
        self.seed = seed
        self.scatter = scatter
        self.fit_reg = fit_reg
        self.order = order
        self.logistic = logistic
        self.lowess = lowess
        self.robust = robust
        self.logx = logx
        self.truncate = truncate
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        self.color = color
        self.label = label

        # Validate the regression options:
        if sum((order > 1, logistic, robust, lowess, logx)) > 1:
            raise ValueError("Mutually exclusive regression options.")

        # Extract the data vals from the arguments or passed dataframe
        self.establish_variables(data, x=x, y=y, units=units,
                                 x_partial=x_partial, y_partial=y_partial)

        # Drop null observations
        if dropna:
            self.dropna("x", "y", "units", "x_partial", "y_partial")

        # Regress nuisance variables out of the data
        if self.x_partial is not None:
            self.x = self.regress_out(self.x, self.x_partial)
        if self.y_partial is not None:
            self.y = self.regress_out(self.y, self.y_partial)

        # Possibly bin the predictor variable, which implies a point estimate
        if x_bins is not None:
            self.x_estimator = np.mean if x_estimator is None else x_estimator
            x_discrete, x_bins = self.bin_predictor(x_bins)
            self.x_discrete = x_discrete
        else:
            self.x_discrete = self.x

        # Disable regression in case of singleton inputs
        if len(self.x) <= 1:
            self.fit_reg = False

        # Save the range of the x variable for the grid later
        if self.fit_reg:
            self.x_range = self.x.min(), self.x.max()

    @property
    def scatter_data(self):
        """Data where each observation is a point."""
        x_j = self.x_jitter
        if x_j is None:
            x = self.x
        else:
            x = self.x + np.random.uniform(-x_j, x_j, len(self.x))

        y_j = self.y_jitter
        if y_j is None:
            y = self.y
        else:
            y = self.y + np.random.uniform(-y_j, y_j, len(self.y))

        return x, y
```
### 86 - seaborn/regression.py:

Start line: 1010, End line: 1096

```python
@_deprecate_positional_args
def residplot(
    *,
    x=None, y=None,
    data=None,
    lowess=False, x_partial=None, y_partial=None,
    order=1, robust=False, dropna=True, label=None, color=None,
    scatter_kws=None, line_kws=None, ax=None
):
    """Plot the residuals of a linear regression.

    This function will regress y on x (possibly as a robust or polynomial
    regression) and then draw a scatterplot of the residuals. You can
    optionally fit a lowess smoother to the residual plot, which can
    help in determining if there is structure to the residuals.

    Parameters
    ----------
    x : vector or string
        Data or column name in `data` for the predictor variable.
    y : vector or string
        Data or column name in `data` for the response variable.
    data : DataFrame, optional
        DataFrame to use if `x` and `y` are column names.
    lowess : boolean, optional
        Fit a lowess smoother to the residual scatterplot.
    {x, y}_partial : matrix or string(s) , optional
        Matrix with same first dimension as `x`, or column name(s) in `data`.
        These variables are treated as confounding and are removed from
        the `x` or `y` variables before plotting.
    order : int, optional
        Order of the polynomial to fit when calculating the residuals.
    robust : boolean, optional
        Fit a robust linear regression when calculating the residuals.
    dropna : boolean, optional
        If True, ignore observations with missing data when fitting and
        plotting.
    label : string, optional
        Label that will be used in any plot legends.
    color : matplotlib color, optional
        Color to use for all elements of the plot.
    {scatter, line}_kws : dictionaries, optional
        Additional keyword arguments passed to scatter() and plot() for drawing
        the components of the plot.
    ax : matplotlib axis, optional
        Plot into this axis, otherwise grab the current axis or make a new
        one if not existing.

    Returns
    -------
    ax: matplotlib axes
        Axes with the regression plot.

    See Also
    --------
    regplot : Plot a simple linear regression model.
    jointplot : Draw a :func:`residplot` with univariate marginal distributions
                (when used with ``kind="resid"``).

    """
    plotter = _RegressionPlotter(x, y, data, ci=None,
                                 order=order, robust=robust,
                                 x_partial=x_partial, y_partial=y_partial,
                                 dropna=dropna, color=color, label=label)

    if ax is None:
        ax = plt.gca()

    # Calculate the residual from a linear regression
    _, yhat, _ = plotter.fit_regression(grid=plotter.x)
    plotter.y = plotter.y - yhat

    # Set the regression option on the plotter
    if lowess:
        plotter.lowess = True
    else:
        plotter.fit_reg = False

    # Plot a horizontal line at 0
    ax.axhline(0, ls=":", c=".2")

    # Draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()
    plotter.plot(ax, scatter_kws, line_kws)
    return ax
```
### 96 - seaborn/regression.py:

Start line: 377, End line: 407

```python
class _RegressionPlotter(_LinearPlotter):

    def scatterplot(self, ax, kws):
        """Draw the data."""
        # Treat the line-based markers specially, explicitly setting larger
        # linewidth than is provided by the seaborn style defaults.
        # This would ideally be handled better in matplotlib (i.e., distinguish
        # between edgewidth for solid glyphs and linewidth for line glyphs
        # but this should do for now.
        line_markers = ["1", "2", "3", "4", "+", "x", "|", "_"]
        if self.x_estimator is None:
            if "marker" in kws and kws["marker"] in line_markers:
                lw = mpl.rcParams["lines.linewidth"]
            else:
                lw = mpl.rcParams["lines.markeredgewidth"]
            kws.setdefault("linewidths", lw)

            if not hasattr(kws['color'], 'shape') or kws['color'].shape[1] < 4:
                kws.setdefault("alpha", .8)

            x, y = self.scatter_data
            ax.scatter(x, y, **kws)
        else:
            # TODO abstraction
            ci_kws = {"color": kws["color"]}
            ci_kws["linewidth"] = mpl.rcParams["lines.linewidth"] * 1.75
            kws.setdefault("s", 50)

            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)
```
### 100 - seaborn/regression.py:

Start line: 266, End line: 289

```python
class _RegressionPlotter(_LinearPlotter):

    def fit_statsmodels(self, grid, model, **kwargs):
        """More general regression function using statsmodels objects."""
        import statsmodels.genmod.generalized_linear_model as glm
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), grid]

        def reg_func(_x, _y):
            try:
                yhat = model(_y, _x, **kwargs).fit().predict(grid)
            except glm.PerfectSeparationError:
                yhat = np.empty(len(grid))
                yhat.fill(np.nan)
            return yhat

        yhat = reg_func(X, y)
        if self.ci is None:
            return yhat, None

        yhat_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed)
        return yhat, yhat_boots
```
### 105 - seaborn/regression.py:

Start line: 291, End line: 316

```python
class _RegressionPlotter(_LinearPlotter):

    def fit_lowess(self):
        """Fit a locally-weighted regression, which returns its own grid."""
        from statsmodels.nonparametric.smoothers_lowess import lowess
        grid, yhat = lowess(self.y, self.x).T
        return grid, yhat

    def fit_logx(self, grid):
        """Fit the model in log-space."""
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), np.log(grid)]

        def reg_func(_x, _y):
            _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
            return np.linalg.pinv(_x).dot(_y)

        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return yhat, None

        beta_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return yhat, yhat_boots
```
### 112 - seaborn/regression.py:

Start line: 189, End line: 228

```python
class _RegressionPlotter(_LinearPlotter):

    def fit_regression(self, ax=None, x_range=None, grid=None):
        """Fit the regression model."""
        # Create the grid for the regression
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range
            else:
                if ax is None:
                    x_min, x_max = x_range
                else:
                    x_min, x_max = ax.get_xlim()
            grid = np.linspace(x_min, x_max, 100)
        ci = self.ci

        # Fit the regression
        if self.order > 1:
            yhat, yhat_boots = self.fit_poly(grid, self.order)
        elif self.logistic:
            from statsmodels.genmod.generalized_linear_model import GLM
            from statsmodels.genmod.families import Binomial
            yhat, yhat_boots = self.fit_statsmodels(grid, GLM,
                                                    family=Binomial())
        elif self.lowess:
            ci = None
            grid, yhat = self.fit_lowess()
        elif self.robust:
            from statsmodels.robust.robust_linear_model import RLM
            yhat, yhat_boots = self.fit_statsmodels(grid, RLM)
        elif self.logx:
            yhat, yhat_boots = self.fit_logx(grid)
        else:
            yhat, yhat_boots = self.fit_fast(grid)

        # Compute the confidence interval at each grid point
        if ci is None:
            err_bands = None
        else:
            err_bands = utils.ci(yhat_boots, ci, axis=0)

        return grid, yhat, err_bands
```
### 131 - seaborn/regression.py:

Start line: 341, End line: 375

```python
class _RegressionPlotter(_LinearPlotter):

    def plot(self, ax, scatter_kws, line_kws):
        """Draw the full plot."""
        # Insert the plot label into the correct set of keyword arguments
        if self.scatter:
            scatter_kws["label"] = self.label
        else:
            line_kws["label"] = self.label

        # Use the current color cycle state as a default
        if self.color is None:
            lines, = ax.plot([], [])
            color = lines.get_color()
            lines.remove()
        else:
            color = self.color

        # Ensure that color is hex to avoid matplotlib weirdness
        color = mpl.colors.rgb2hex(mpl.colors.colorConverter.to_rgb(color))

        # Let color in keyword arguments override overall plot color
        scatter_kws.setdefault("color", color)
        line_kws.setdefault("color", color)

        # Draw the constituent plots
        if self.scatter:
            self.scatterplot(ax, scatter_kws)

        if self.fit_reg:
            self.lineplot(ax, line_kws)

        # Label the axes
        if hasattr(self.x, "name"):
            ax.set_xlabel(self.x.name)
        if hasattr(self.y, "name"):
            ax.set_ylabel(self.y.name)
```
### 139 - seaborn/regression.py:

Start line: 230, End line: 247

```python
class _RegressionPlotter(_LinearPlotter):

    def fit_fast(self, grid):
        """Low-level regression and prediction using linear algebra."""
        def reg_func(_x, _y):
            return np.linalg.pinv(_x).dot(_y)

        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), grid]
        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return yhat, None

        beta_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return yhat, yhat_boots
```
### 142 - seaborn/regression.py:

Start line: 57, End line: 68

```python
class _LinearPlotter(object):

    def dropna(self, *vars):
        """Remove observations with missing data."""
        vals = [getattr(self, var) for var in vars]
        vals = [v for v in vals if v is not None]
        not_na = np.all(np.column_stack([pd.notnull(v) for v in vals]), axis=1)
        for var in vars:
            val = getattr(self, var)
            if val is not None:
                setattr(self, var, val[not_na])

    def plot(self, ax):
        raise NotImplementedError
```
### 149 - seaborn/regression.py:

Start line: 26, End line: 55

```python
class _LinearPlotter(object):
    """Base class for plotting relational data in tidy format.

    To get anything useful done you'll have to inherit from this, but setup
    code that can be abstracted out should be put here.

    """
    def establish_variables(self, data, **kws):
        """Extract variables from data or use directly."""
        self.data = data

        # Validate the inputs
        any_strings = any([isinstance(v, str) for v in kws.values()])
        if any_strings and data is None:
            raise ValueError("Must pass `data` if using named variables.")

        # Set the variables
        for var, val in kws.items():
            if isinstance(val, str):
                vector = data[val]
            elif isinstance(val, list):
                vector = np.asarray(val)
            else:
                vector = val
            if vector is not None and vector.shape != (1,):
                vector = np.squeeze(vector)
            if np.ndim(vector) > 1:
                err = "regplot inputs must be 1d"
                raise ValueError(err)
            setattr(self, var, vector)
```
### 150 - seaborn/regression.py:

Start line: 154, End line: 187

```python
class _RegressionPlotter(_LinearPlotter):

    @property
    def estimate_data(self):
        """Data with a point estimate and CI for each discrete x value."""
        x, y = self.x_discrete, self.y
        vals = sorted(np.unique(x))
        points, cis = [], []

        for val in vals:

            # Get the point estimate of the y variable
            _y = y[x == val]
            est = self.x_estimator(_y)
            points.append(est)

            # Compute the confidence interval for this estimate
            if self.x_ci is None:
                cis.append(None)
            else:
                units = None
                if self.x_ci == "sd":
                    sd = np.std(_y)
                    _ci = est - sd, est + sd
                else:
                    if self.units is not None:
                        units = self.units[x == val]
                    boots = algo.bootstrap(_y,
                                           func=self.x_estimator,
                                           n_boot=self.n_boot,
                                           units=units,
                                           seed=self.seed)
                    _ci = utils.ci(boots, self.x_ci)
                cis.append(_ci)

        return vals, points, cis
```
### 168 - seaborn/regression.py:

Start line: 409, End line: 424

```python
class _RegressionPlotter(_LinearPlotter):

    def lineplot(self, ax, kws):
        """Draw the model."""
        # Fit the regression model
        grid, yhat, err_bands = self.fit_regression(ax)
        edges = grid[0], grid[-1]

        # Get set default aesthetics
        fill_color = kws["color"]
        lw = kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.5)
        kws.setdefault("linewidth", lw)

        # Draw the regression line and confidence interval
        line, = ax.plot(grid, yhat, **kws)
        line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
        if err_bands is not None:
            ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=.15)
```
### 177 - seaborn/regression.py:

Start line: 249, End line: 264

```python
class _RegressionPlotter(_LinearPlotter):

    def fit_poly(self, grid, order):
        """Regression using numpy polyfit for higher-order trends."""
        def reg_func(_x, _y):
            return np.polyval(np.polyfit(_x, _y, order), grid)

        x, y = self.x, self.y
        yhat = reg_func(x, y)
        if self.ci is None:
            return yhat, None

        yhat_boots = algo.bootstrap(x, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed)
        return yhat, yhat_boots
```
### 178 - seaborn/regression.py:

Start line: 318, End line: 339

```python
class _RegressionPlotter(_LinearPlotter):

    def bin_predictor(self, bins):
        """Discretize a predictor by assigning value to closest bin."""
        x = np.asarray(self.x)
        if np.isscalar(bins):
            percentiles = np.linspace(0, 100, bins + 2)[1:-1]
            bins = np.percentile(x, percentiles)
        else:
            bins = np.ravel(bins)

        dist = np.abs(np.subtract.outer(x, bins))
        x_binned = bins[np.argmin(dist, axis=1)].ravel()

        return x_binned, bins

    def regress_out(self, a, b):
        """Regress b from a keeping a's original mean."""
        a_mean = a.mean()
        a = a - a_mean
        b = b - b.mean()
        b = np.c_[b]
        a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
        return np.asarray(a_prime + a_mean).reshape(a.shape)
```
