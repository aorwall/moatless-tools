# mwaskom__seaborn-2979

| **mwaskom/seaborn** | `ebc4bfe9f8bf5c4ff10b14da8a49c8baa1ba76d0` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 4467 |
| **Any found context length** | 1871 |
| **Avg pos** | 157.0 |
| **Min pos** | 3 |
| **Max pos** | 197 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/seaborn/_core/plot.py b/seaborn/_core/plot.py
--- a/seaborn/_core/plot.py
+++ b/seaborn/_core/plot.py
@@ -943,8 +943,11 @@ def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:
                 visible_side = {"x": "bottom", "y": "left"}.get(axis)
                 show_axis_label = (
                     sub[visible_side]
-                    or axis in p._pair_spec and bool(p._pair_spec.get("wrap"))
                     or not p._pair_spec.get("cross", True)
+                    or (
+                        axis in p._pair_spec.get("structure", {})
+                        and bool(p._pair_spec.get("wrap"))
+                    )
                 )
                 axis_obj.get_label().set_visible(show_axis_label)
                 show_tick_labels = (
@@ -1149,7 +1152,7 @@ def _setup_scales(
             # behavior, so we will raise rather than hack together a workaround.
             if axis is not None and Version(mpl.__version__) < Version("3.4.0"):
                 from seaborn._core.scales import Nominal
-                paired_axis = axis in p._pair_spec
+                paired_axis = axis in p._pair_spec.get("structure", {})
                 cat_scale = isinstance(scale, Nominal)
                 ok_dim = {"x": "col", "y": "row"}[axis]
                 shared_axes = share_state not in [False, "none", ok_dim]
diff --git a/seaborn/_core/subplots.py b/seaborn/_core/subplots.py
--- a/seaborn/_core/subplots.py
+++ b/seaborn/_core/subplots.py
@@ -30,9 +30,8 @@ class Subplots:
 
     """
     def __init__(
-        # TODO defined TypedDict types for these specs
         self,
-        subplot_spec: dict,
+        subplot_spec: dict,  # TODO define as TypedDict
         facet_spec: FacetSpec,
         pair_spec: PairSpec,
     ):
@@ -130,7 +129,7 @@ def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
             if key not in self.subplot_spec:
                 if axis in pair_spec.get("structure", {}):
                     # Paired axes are shared along one dimension by default
-                    if self.wrap in [None, 1] and pair_spec.get("cross", True):
+                    if self.wrap is None and pair_spec.get("cross", True):
                         val = axis_to_dim[axis]
                     else:
                         val = False

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| seaborn/_core/plot.py | 946 | 946 | 3 | 2 | 1871
| seaborn/_core/plot.py | 1152 | 1152 | 9 | 2 | 4467
| seaborn/_core/subplots.py | 33 | 35 | 197 | 9 | 90991
| seaborn/_core/subplots.py | 133 | 133 | 105 | 9 | 58032


## Problem Statement

```
Visibility of internal axis labels is wrong with wrapped pair plot
\`\`\`python
(
    so.Plot(mpg, y="mpg")
    .pair(["displacement", "weight", "horsepower", "cylinders"], wrap=2)
)
\`\`\`
![image](https://user-images.githubusercontent.com/315810/186793170-dedae71a-2cb9-4f0e-9339-07fc1d13ac59.png)

The top two subplots should have distinct x labels.
Visibility of internal axis labels is wrong with wrapped pair plot
\`\`\`python
(
    so.Plot(mpg, y="mpg")
    .pair(["displacement", "weight", "horsepower", "cylinders"], wrap=2)
)
\`\`\`
![image](https://user-images.githubusercontent.com/315810/186793170-dedae71a-2cb9-4f0e-9339-07fc1d13ac59.png)

The top two subplots should have distinct x labels.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 seaborn/axisgrid.py | 2089 | 2156| 726 | 726 | 19631 | 
| 2 | 1 seaborn/axisgrid.py | 2158 | 2178| 257 | 983 | 19631 | 
| **-> 3 <-** | **2 seaborn/_core/plot.py** | 894 | 987| 888 | 1871 | 32502 | 
| 4 | 2 seaborn/axisgrid.py | 1 | 28| 167 | 2038 | 32502 | 
| 5 | **2 seaborn/_core/plot.py** | 450 | 507| 487 | 2525 | 32502 | 
| 6 | 3 examples/pairgrid_dotplot.py | 1 | 39| 268 | 2793 | 32770 | 
| 7 | 4 examples/paired_pointplots.py | 1 | 21| 153 | 2946 | 32923 | 
| 8 | **4 seaborn/_core/plot.py** | 1324 | 1364| 331 | 3277 | 32923 | 
| **-> 9 <-** | **4 seaborn/_core/plot.py** | 1080 | 1198| 1190 | 4467 | 32923 | 
| 10 | 5 examples/marginal_ticks.py | 1 | 16| 144 | 4611 | 33067 | 
| 11 | 6 seaborn/_oldcore.py | 1218 | 1299| 812 | 5423 | 46581 | 
| 12 | 7 seaborn/relational.py | 1 | 50| 384 | 5807 | 55036 | 
| 13 | 8 seaborn/categorical.py | 321 | 416| 720 | 6527 | 83463 | 
| 14 | **9 seaborn/_core/subplots.py** | 47 | 77| 376 | 6903 | 85814 | 
| 15 | 9 seaborn/_oldcore.py | 1321 | 1332| 187 | 7090 | 85814 | 
| 16 | 9 seaborn/axisgrid.py | 2181 | 2341| 1464 | 8554 | 85814 | 
| 17 | 9 seaborn/axisgrid.py | 1171 | 1184| 131 | 8685 | 85814 | 
| 18 | 10 seaborn/_marks/bar.py | 182 | 230| 428 | 9113 | 87853 | 
| 19 | 10 seaborn/axisgrid.py | 1659 | 1672| 123 | 9236 | 87853 | 
| 20 | 11 seaborn/distributions.py | 742 | 897| 1277 | 10513 | 107451 | 
| 21 | **11 seaborn/_core/subplots.py** | 103 | 120| 204 | 10717 | 107451 | 
| 22 | 11 seaborn/relational.py | 423 | 515| 769 | 11486 | 107451 | 
| 23 | **11 seaborn/_core/plot.py** | 1292 | 1322| 304 | 11790 | 107451 | 
| 24 | **11 seaborn/_core/plot.py** | 1200 | 1268| 670 | 12460 | 107451 | 
| 25 | 11 seaborn/axisgrid.py | 2007 | 2088| 804 | 13264 | 107451 | 
| 26 | 12 examples/part_whole_bars.py | 1 | 31| 216 | 13480 | 107667 | 
| 27 | 12 seaborn/distributions.py | 550 | 740| 1491 | 14971 | 107667 | 
| 28 | 12 seaborn/categorical.py | 2999 | 3183| 1564 | 16535 | 107667 | 
| 29 | 12 seaborn/categorical.py | 255 | 319| 501 | 17036 | 107667 | 
| 30 | 12 seaborn/_oldcore.py | 760 | 857| 787 | 17823 | 107667 | 
| 31 | 12 seaborn/relational.py | 819 | 923| 840 | 18663 | 107667 | 
| 32 | 12 seaborn/axisgrid.py | 1554 | 1574| 241 | 18904 | 107667 | 
| 33 | 12 seaborn/distributions.py | 1140 | 1226| 775 | 19679 | 107667 | 
| 34 | 12 seaborn/categorical.py | 1841 | 1975| 1205 | 20884 | 107667 | 
| 35 | **12 seaborn/_core/plot.py** | 1 | 98| 547 | 21431 | 107667 | 
| 36 | 12 seaborn/categorical.py | 42 | 119| 762 | 22193 | 107667 | 
| 37 | 12 seaborn/distributions.py | 1603 | 1752| 1320 | 23513 | 107667 | 
| 38 | 12 seaborn/relational.py | 176 | 284| 937 | 24450 | 107667 | 
| 39 | 12 seaborn/relational.py | 925 | 984| 622 | 25072 | 107667 | 
| 40 | 12 seaborn/_marks/bar.py | 122 | 162| 463 | 25535 | 107667 | 
| 41 | 12 seaborn/axisgrid.py | 1515 | 1552| 279 | 25814 | 107667 | 
| 42 | 12 seaborn/categorical.py | 3184 | 3258| 804 | 26618 | 107667 | 
| 43 | **12 seaborn/_core/plot.py** | 1388 | 1461| 612 | 27230 | 107667 | 
| 44 | 12 seaborn/categorical.py | 2849 | 2912| 476 | 27706 | 107667 | 
| 45 | **12 seaborn/_core/plot.py** | 989 | 1041| 356 | 28062 | 107667 | 
| 46 | 12 seaborn/axisgrid.py | 362 | 542| 1452 | 29514 | 107667 | 
| 47 | 12 seaborn/categorical.py | 2494 | 2550| 421 | 29935 | 107667 | 
| 48 | 12 seaborn/categorical.py | 2418 | 2491| 670 | 30605 | 107667 | 
| 49 | 12 seaborn/relational.py | 597 | 640| 416 | 31021 | 107667 | 
| 50 | 12 seaborn/categorical.py | 1740 | 1785| 400 | 31421 | 107667 | 
| 51 | 12 seaborn/categorical.py | 2915 | 2952| 267 | 31688 | 107667 | 
| 52 | 12 seaborn/categorical.py | 894 | 925| 273 | 31961 | 107667 | 
| 53 | 12 seaborn/categorical.py | 2296 | 2313| 166 | 32127 | 107667 | 
| 54 | 13 seaborn/matrix.py | 1091 | 1154| 543 | 32670 | 119343 | 
| 55 | 13 seaborn/distributions.py | 899 | 1032| 937 | 33607 | 119343 | 
| 56 | 13 seaborn/distributions.py | 2284 | 2306| 296 | 33903 | 119343 | 
| 57 | 13 seaborn/categorical.py | 736 | 776| 337 | 34240 | 119343 | 
| 58 | 13 seaborn/axisgrid.py | 1406 | 1428| 185 | 34425 | 119343 | 
| 59 | 13 seaborn/relational.py | 378 | 422| 524 | 34949 | 119343 | 
| 60 | **13 seaborn/_core/plot.py** | 1270 | 1290| 198 | 35147 | 119343 | 
| 61 | 13 seaborn/categorical.py | 2398 | 2415| 191 | 35338 | 119343 | 
| 62 | 13 seaborn/axisgrid.py | 1576 | 1613| 290 | 35628 | 119343 | 
| 63 | 13 seaborn/matrix.py | 1156 | 1211| 562 | 36190 | 119343 | 
| 64 | 14 seaborn/utils.py | 325 | 392| 645 | 36835 | 125904 | 
| 65 | 15 examples/scatter_bubbles.py | 1 | 18| 111 | 36946 | 126015 | 
| 66 | 15 seaborn/axisgrid.py | 2344 | 2403| 413 | 37359 | 126015 | 
| 67 | 15 seaborn/axisgrid.py | 303 | 359| 503 | 37862 | 126015 | 
| 68 | 15 seaborn/categorical.py | 1645 | 1737| 745 | 38607 | 126015 | 
| 69 | 15 seaborn/axisgrid.py | 1185 | 1357| 1502 | 40109 | 126015 | 
| 70 | 15 seaborn/categorical.py | 796 | 856| 453 | 40562 | 126015 | 
| 71 | 16 seaborn/_core/data.py | 183 | 263| 721 | 41283 | 127951 | 
| 72 | 16 seaborn/categorical.py | 1494 | 1520| 322 | 41605 | 127951 | 
| 73 | 17 seaborn/regression.py | 557 | 641| 852 | 42457 | 137403 | 
| 74 | 17 seaborn/categorical.py | 2242 | 2293| 347 | 42804 | 137403 | 
| 75 | 17 seaborn/axisgrid.py | 1430 | 1513| 652 | 43456 | 137403 | 
| 76 | 17 seaborn/relational.py | 536 | 594| 503 | 43959 | 137403 | 
| 77 | 17 seaborn/regression.py | 827 | 851| 279 | 44238 | 137403 | 
| 78 | 17 seaborn/_oldcore.py | 1359 | 1450| 929 | 45167 | 137403 | 
| 79 | **17 seaborn/_core/plot.py** | 294 | 324| 256 | 45423 | 137403 | 
| 80 | 17 seaborn/axisgrid.py | 1615 | 1657| 327 | 45750 | 137403 | 
| 81 | **17 seaborn/_core/plot.py** | 863 | 892| 222 | 45972 | 137403 | 
| 82 | 17 seaborn/categorical.py | 1 | 39| 252 | 46224 | 137403 | 
| 83 | 17 seaborn/categorical.py | 2223 | 2239| 146 | 46370 | 137403 | 
| 84 | 17 seaborn/distributions.py | 1 | 91| 588 | 46958 | 137403 | 
| 85 | 17 seaborn/regression.py | 1 | 22| 121 | 47079 | 137403 | 
| 86 | 17 seaborn/regression.py | 854 | 1020| 1275 | 48354 | 137403 | 
| 87 | 17 seaborn/categorical.py | 2615 | 2673| 398 | 48752 | 137403 | 
| 88 | 17 seaborn/axisgrid.py | 1675 | 1763| 855 | 49607 | 137403 | 
| 89 | 18 examples/multiple_ecdf.py | 1 | 18| 122 | 49729 | 137525 | 
| 90 | 19 seaborn/_marks/base.py | 190 | 229| 301 | 50030 | 139644 | 
| 91 | 19 seaborn/_oldcore.py | 1334 | 1357| 198 | 50228 | 139644 | 
| 92 | **19 seaborn/_core/plot.py** | 592 | 605| 116 | 50344 | 139644 | 
| 93 | 19 seaborn/distributions.py | 382 | 549| 1246 | 51590 | 139644 | 
| 94 | 19 seaborn/categorical.py | 2553 | 2612| 444 | 52034 | 139644 | 
| 95 | 19 seaborn/distributions.py | 172 | 193| 228 | 52262 | 139644 | 
| 96 | 19 seaborn/categorical.py | 779 | 794| 127 | 52389 | 139644 | 
| 97 | 19 seaborn/regression.py | 644 | 824| 1440 | 53829 | 139644 | 
| 98 | 20 examples/kde_ridgeplot.py | 1 | 51| 399 | 54228 | 140043 | 
| 99 | 20 seaborn/categorical.py | 2765 | 2824| 387 | 54615 | 140043 | 
| 100 | 20 seaborn/distributions.py | 2117 | 2191| 585 | 55200 | 140043 | 
| 101 | 20 seaborn/relational.py | 726 | 757| 243 | 55443 | 140043 | 
| 102 | 20 seaborn/categorical.py | 419 | 628| 1386 | 56829 | 140043 | 
| 103 | 20 seaborn/distributions.py | 2193 | 2283| 810 | 57639 | 140043 | 
| 104 | 21 examples/layered_bivariate_plot.py | 1 | 24| 186 | 57825 | 140229 | 
| **-> 105 <-** | **21 seaborn/_core/subplots.py** | 122 | 141| 207 | 58032 | 140229 | 
| 106 | 22 seaborn/_marks/line.py | 71 | 105| 342 | 58374 | 142088 | 
| 107 | **22 seaborn/_core/subplots.py** | 143 | 221| 614 | 58988 | 142088 | 
| 108 | 23 seaborn/_marks/dot.py | 62 | 85| 196 | 59184 | 143513 | 
| 109 | 23 seaborn/categorical.py | 1977 | 2220| 539 | 59723 | 143513 | 
| 110 | 23 seaborn/matrix.py | 1213 | 1239| 274 | 59997 | 143513 | 
| 111 | 23 seaborn/_marks/line.py | 37 | 69| 303 | 60300 | 143513 | 
| 112 | 23 seaborn/distributions.py | 1034 | 1139| 752 | 61052 | 143513 | 
| 113 | 23 seaborn/categorical.py | 1138 | 1311| 1218 | 62270 | 143513 | 
| 114 | 24 seaborn/widgets.py | 1 | 58| 426 | 62696 | 147017 | 
| 115 | 24 seaborn/categorical.py | 2737 | 2762| 232 | 62928 | 147017 | 
| 116 | **24 seaborn/_core/plot.py** | 567 | 590| 281 | 63209 | 147017 | 
| 117 | 24 seaborn/axisgrid.py | 1834 | 1893| 493 | 63702 | 147017 | 
| 118 | 24 seaborn/_oldcore.py | 1055 | 1092| 278 | 63980 | 147017 | 
| 119 | 25 seaborn/_testing.py | 50 | 72| 186 | 64166 | 147615 | 
| 120 | 25 seaborn/axisgrid.py | 1359 | 1374| 121 | 64287 | 147615 | 
| 121 | 25 seaborn/matrix.py | 541 | 556| 226 | 64513 | 147615 | 
| 122 | 25 seaborn/distributions.py | 1297 | 1329| 301 | 64814 | 147615 | 
| 123 | 25 seaborn/distributions.py | 1376 | 1471| 608 | 65422 | 147615 | 
| 124 | 25 seaborn/categorical.py | 2827 | 2846| 193 | 65615 | 147615 | 
| 125 | **25 seaborn/_core/plot.py** | 743 | 777| 304 | 65919 | 147615 | 
| 126 | **25 seaborn/_core/plot.py** | 1043 | 1078| 284 | 66203 | 147615 | 
| 127 | 25 seaborn/categorical.py | 1542 | 1589| 357 | 66560 | 147615 | 
| 128 | 25 seaborn/relational.py | 286 | 343| 420 | 66980 | 147615 | 
| 129 | 25 seaborn/categorical.py | 858 | 891| 281 | 67261 | 147615 | 
| 130 | 26 seaborn/_docstrings.py | 137 | 199| 393 | 67654 | 149053 | 
| 131 | 27 seaborn/_core/scales.py | 749 | 833| 603 | 68257 | 155630 | 
| 132 | **27 seaborn/_core/plot.py** | 132 | 218| 671 | 68928 | 155630 | 
| 133 | 27 seaborn/categorical.py | 1425 | 1492| 490 | 69418 | 155630 | 
| 134 | 28 examples/scatterplot_sizes.py | 1 | 25| 164 | 69582 | 155794 | 
| 135 | 28 seaborn/distributions.py | 1995 | 2072| 699 | 70281 | 155794 | 
| 136 | 28 seaborn/relational.py | 52 | 173| 1142 | 71423 | 155794 | 
| 137 | 28 seaborn/categorical.py | 226 | 253| 264 | 71687 | 155794 | 
| 138 | 29 examples/horizontal_boxplot.py | 1 | 31| 192 | 71879 | 155986 | 
| 139 | 29 seaborn/distributions.py | 94 | 135| 316 | 72195 | 155986 | 
| 140 | 29 seaborn/distributions.py | 1331 | 1373| 351 | 72546 | 155986 | 
| 141 | 30 examples/faceted_lineplot.py | 1 | 24| 140 | 72686 | 156126 | 
| 142 | 31 examples/jitter_stripplot.py | 1 | 37| 273 | 72959 | 156399 | 
| 143 | 31 seaborn/axisgrid.py | 852 | 907| 464 | 73423 | 156399 | 
| 144 | 32 seaborn/_core/properties.py | 1 | 39| 253 | 73676 | 162409 | 
| 145 | **32 seaborn/_core/plot.py** | 1556 | 1581| 242 | 73918 | 162409 | 
| 146 | 32 seaborn/categorical.py | 182 | 193| 128 | 74046 | 162409 | 
| 147 | 33 examples/many_facets.py | 1 | 40| 312 | 74358 | 162721 | 
| 148 | **33 seaborn/_core/plot.py** | 824 | 861| 401 | 74759 | 162721 | 
| 149 | 33 seaborn/matrix.py | 1 | 56| 332 | 75091 | 162721 | 
| 150 | 33 seaborn/categorical.py | 195 | 224| 370 | 75461 | 162721 | 
| 151 | 33 seaborn/_oldcore.py | 1155 | 1217| 531 | 75992 | 162721 | 
| 152 | 33 seaborn/relational.py | 518 | 534| 147 | 76139 | 162721 | 
| 153 | 33 seaborn/categorical.py | 2955 | 2996| 236 | 76375 | 162721 | 
| 154 | 33 seaborn/axisgrid.py | 1391 | 1404| 117 | 76492 | 162721 | 
| 155 | **33 seaborn/_core/plot.py** | 509 | 565| 424 | 76916 | 162721 | 
| 156 | 34 examples/grouped_barplot.py | 1 | 21| 121 | 77037 | 162842 | 
| 157 | 34 seaborn/axisgrid.py | 544 | 634| 906 | 77943 | 162842 | 
| 158 | 34 seaborn/matrix.py | 1066 | 1089| 264 | 78207 | 162842 | 
| 159 | 34 seaborn/relational.py | 346 | 376| 251 | 78458 | 162842 | 
| 160 | 35 examples/grouped_boxplot.py | 1 | 19| 103 | 78561 | 162945 | 
| 161 | **35 seaborn/_core/plot.py** | 1366 | 1386| 176 | 78737 | 162945 | 
| 162 | 36 examples/joint_histogram.py | 1 | 27| 192 | 78929 | 163137 | 
| 163 | 36 seaborn/categorical.py | 2676 | 2734| 424 | 79353 | 163137 | 
| 164 | 36 seaborn/_core/scales.py | 1 | 48| 258 | 79611 | 163137 | 
| 165 | 37 examples/multiple_bivariate_kde.py | 1 | 25| 126 | 79737 | 163263 | 
| 166 | 37 seaborn/matrix.py | 681 | 735| 512 | 80249 | 163263 | 
| 167 | 38 examples/many_pairwise_correlations.py | 1 | 35| 230 | 80479 | 163493 | 
| 168 | 39 examples/palette_generation.py | 1 | 36| 282 | 80761 | 163775 | 
| 169 | **39 seaborn/_core/subplots.py** | 222 | 271| 534 | 81295 | 163775 | 
| 170 | 40 examples/pair_grid_with_kde.py | 1 | 16| 0 | 81295 | 163864 | 
| 171 | 40 seaborn/axisgrid.py | 1376 | 1389| 117 | 81412 | 163864 | 
| 172 | 40 seaborn/axisgrid.py | 1939 | 1960| 150 | 81562 | 163864 | 
| 173 | 40 seaborn/axisgrid.py | 909 | 935| 257 | 81819 | 163864 | 
| 174 | 40 seaborn/_docstrings.py | 62 | 134| 611 | 82430 | 163864 | 
| 175 | 40 seaborn/_marks/bar.py | 27 | 72| 408 | 82838 | 163864 | 
| 176 | 40 seaborn/matrix.py | 293 | 556| 667 | 83505 | 163864 | 
| 177 | 40 seaborn/axisgrid.py | 1963 | 2004| 273 | 83778 | 163864 | 
| 178 | 40 seaborn/regression.py | 376 | 408| 371 | 84149 | 163864 | 
| 179 | 40 seaborn/categorical.py | 714 | 734| 156 | 84305 | 163864 | 
| 180 | 40 seaborn/categorical.py | 1592 | 1643| 446 | 84751 | 163864 | 
| 181 | 40 seaborn/relational.py | 643 | 723| 560 | 85311 | 163864 | 
| 182 | 41 seaborn/_marks/area.py | 1 | 50| 308 | 85619 | 164979 | 
| 183 | 42 examples/wide_data_lineplot.py | 1 | 20| 137 | 85756 | 165116 | 
| 184 | 42 seaborn/categorical.py | 2316 | 2395| 737 | 86493 | 165116 | 
| 185 | 43 seaborn/_core/moves.py | 63 | 120| 476 | 86969 | 166620 | 
| 186 | 44 doc/tools/generate_logos.py | 61 | 156| 938 | 87907 | 168545 | 
| 187 | 44 seaborn/_marks/bar.py | 1 | 24| 113 | 88020 | 168545 | 
| 188 | 44 seaborn/matrix.py | 1407 | 1421| 272 | 88292 | 168545 | 
| 189 | 44 seaborn/axisgrid.py | 829 | 850| 176 | 88468 | 168545 | 
| 190 | 44 seaborn/_oldcore.py | 1301 | 1319| 136 | 88604 | 168545 | 
| 191 | 44 seaborn/distributions.py | 2500 | 2553| 537 | 89141 | 168545 | 
| 192 | 44 seaborn/_oldcore.py | 1094 | 1138| 372 | 89513 | 168545 | 
| 193 | 44 seaborn/_core/properties.py | 118 | 151| 254 | 89767 | 168545 | 
| 194 | 44 seaborn/relational.py | 987 | 1065| 557 | 90324 | 168545 | 
| 195 | 44 seaborn/categorical.py | 630 | 653| 168 | 90492 | 168545 | 
| 196 | **44 seaborn/_core/subplots.py** | 79 | 101| 222 | 90714 | 168545 | 
| **-> 197 <-** | **44 seaborn/_core/subplots.py** | 1 | 45| 277 | 90991 | 168545 | 
| 198 | 44 seaborn/_oldcore.py | 375 | 431| 486 | 91477 | 168545 | 
| 199 | 44 seaborn/distributions.py | 1228 | 1295| 577 | 92054 | 168545 | 
| 200 | 44 seaborn/utils.py | 635 | 673| 222 | 92276 | 168545 | 
| 201 | 45 examples/pointplot_anova.py | 1 | 18| 124 | 92400 | 168669 | 
| 202 | 45 seaborn/categorical.py | 3348 | 3418| 668 | 93068 | 168669 | 
| 203 | 45 seaborn/axisgrid.py | 97 | 118| 195 | 93263 | 168669 | 
| 204 | 45 seaborn/utils.py | 800 | 817| 169 | 93432 | 168669 | 
| 205 | 46 examples/heat_scatter.py | 1 | 42| 330 | 93762 | 168999 | 
| 206 | 47 examples/anscombes_quartet.py | 1 | 17| 121 | 93883 | 169120 | 
| 207 | 47 seaborn/matrix.py | 559 | 628| 479 | 94362 | 169120 | 
| 208 | 47 seaborn/distributions.py | 2075 | 2114| 222 | 94584 | 169120 | 
| 209 | 47 seaborn/categorical.py | 3261 | 3345| 712 | 95296 | 169120 | 
| 210 | 47 seaborn/_core/scales.py | 303 | 371| 465 | 95761 | 169120 | 
| 211 | 48 seaborn/rcmod.py | 1 | 80| 392 | 96153 | 172812 | 
| 212 | 48 seaborn/distributions.py | 2411 | 2498| 724 | 96877 | 172812 | 
| 213 | 48 seaborn/categorical.py | 1327 | 1359| 326 | 97203 | 172812 | 
| 214 | 48 seaborn/_testing.py | 1 | 47| 305 | 97508 | 172812 | 
| 215 | 49 examples/different_scatter_variables.py | 1 | 26| 207 | 97715 | 173019 | 
| 216 | 49 seaborn/_marks/line.py | 136 | 163| 213 | 97928 | 173019 | 
| 217 | 50 seaborn/miscplot.py | 1 | 30| 231 | 98159 | 173397 | 
| 218 | **50 seaborn/_core/plot.py** | 1463 | 1505| 342 | 98501 | 173397 | 
| 219 | **50 seaborn/_core/plot.py** | 1507 | 1554| 432 | 98933 | 173397 | 
| 220 | 50 seaborn/regression.py | 429 | 556| 1382 | 100315 | 173397 | 
| 221 | 50 seaborn/categorical.py | 170 | 180| 156 | 100471 | 173397 | 
| 222 | 50 seaborn/relational.py | 760 | 816| 327 | 100798 | 173397 | 
| 223 | 50 seaborn/categorical.py | 121 | 168| 557 | 101355 | 173397 | 
| 224 | **50 seaborn/_core/plot.py** | 624 | 645| 202 | 101557 | 173397 | 
| 225 | **50 seaborn/_core/plot.py** | 220 | 252| 299 | 101856 | 173397 | 
| 226 | 50 seaborn/_oldcore.py | 1556 | 1635| 632 | 102488 | 173397 | 
| 227 | **50 seaborn/_core/plot.py** | 780 | 822| 326 | 102814 | 173397 | 
| 228 | 50 seaborn/regression.py | 340 | 374| 263 | 103077 | 173397 | 
| 229 | 50 seaborn/categorical.py | 3420 | 3450| 268 | 103345 | 173397 | 
| 230 | **50 seaborn/_core/plot.py** | 365 | 448| 799 | 104144 | 173397 | 
| 231 | 50 seaborn/categorical.py | 1375 | 1393| 174 | 104318 | 173397 | 


### Hint

```


```

## Patch

```diff
diff --git a/seaborn/_core/plot.py b/seaborn/_core/plot.py
--- a/seaborn/_core/plot.py
+++ b/seaborn/_core/plot.py
@@ -943,8 +943,11 @@ def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:
                 visible_side = {"x": "bottom", "y": "left"}.get(axis)
                 show_axis_label = (
                     sub[visible_side]
-                    or axis in p._pair_spec and bool(p._pair_spec.get("wrap"))
                     or not p._pair_spec.get("cross", True)
+                    or (
+                        axis in p._pair_spec.get("structure", {})
+                        and bool(p._pair_spec.get("wrap"))
+                    )
                 )
                 axis_obj.get_label().set_visible(show_axis_label)
                 show_tick_labels = (
@@ -1149,7 +1152,7 @@ def _setup_scales(
             # behavior, so we will raise rather than hack together a workaround.
             if axis is not None and Version(mpl.__version__) < Version("3.4.0"):
                 from seaborn._core.scales import Nominal
-                paired_axis = axis in p._pair_spec
+                paired_axis = axis in p._pair_spec.get("structure", {})
                 cat_scale = isinstance(scale, Nominal)
                 ok_dim = {"x": "col", "y": "row"}[axis]
                 shared_axes = share_state not in [False, "none", ok_dim]
diff --git a/seaborn/_core/subplots.py b/seaborn/_core/subplots.py
--- a/seaborn/_core/subplots.py
+++ b/seaborn/_core/subplots.py
@@ -30,9 +30,8 @@ class Subplots:
 
     """
     def __init__(
-        # TODO defined TypedDict types for these specs
         self,
-        subplot_spec: dict,
+        subplot_spec: dict,  # TODO define as TypedDict
         facet_spec: FacetSpec,
         pair_spec: PairSpec,
     ):
@@ -130,7 +129,7 @@ def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
             if key not in self.subplot_spec:
                 if axis in pair_spec.get("structure", {}):
                     # Paired axes are shared along one dimension by default
-                    if self.wrap in [None, 1] and pair_spec.get("cross", True):
+                    if self.wrap is None and pair_spec.get("cross", True):
                         val = axis_to_dim[axis]
                     else:
                         val = False

```

## Test Patch

```diff
diff --git a/tests/_core/test_plot.py b/tests/_core/test_plot.py
--- a/tests/_core/test_plot.py
+++ b/tests/_core/test_plot.py
@@ -1538,8 +1538,10 @@ def test_x_wrapping(self, long_df):
 
         assert_gridspec_shape(p._figure.axes[0], len(x_vars) // wrap + 1, wrap)
         assert len(p._figure.axes) == len(x_vars)
-
-        # TODO test axis labels and visibility
+        for ax, var in zip(p._figure.axes, x_vars):
+            label = ax.xaxis.get_label()
+            assert label.get_visible()
+            assert label.get_text() == var
 
     def test_y_wrapping(self, long_df):
 
@@ -1547,10 +1549,17 @@ def test_y_wrapping(self, long_df):
         wrap = 3
         p = Plot(long_df, x="x").pair(y=y_vars, wrap=wrap).plot()
 
-        assert_gridspec_shape(p._figure.axes[0], wrap, len(y_vars) // wrap + 1)
+        n_row, n_col = wrap, len(y_vars) // wrap + 1
+        assert_gridspec_shape(p._figure.axes[0], n_row, n_col)
         assert len(p._figure.axes) == len(y_vars)
-
-        # TODO test axis labels and visibility
+        label_array = np.empty(n_row * n_col, object)
+        label_array[:len(y_vars)] = y_vars
+        label_array = label_array.reshape((n_row, n_col), order="F")
+        label_array = [y for y in label_array.flat if y is not None]
+        for i, ax in enumerate(p._figure.axes):
+            label = ax.yaxis.get_label()
+            assert label.get_visible()
+            assert label.get_text() == label_array[i]
 
     def test_non_cross_wrapping(self, long_df):
 
diff --git a/tests/_core/test_subplots.py b/tests/_core/test_subplots.py
--- a/tests/_core/test_subplots.py
+++ b/tests/_core/test_subplots.py
@@ -191,6 +191,18 @@ def test_y_paired_and_wrapped(self):
         assert s.subplot_spec["sharex"] is True
         assert s.subplot_spec["sharey"] is False
 
+    def test_y_paired_and_wrapped_single_row(self):
+
+        y = ["x", "y", "z"]
+        wrap = 1
+        s = Subplots({}, {}, {"structure": {"y": y}, "wrap": wrap})
+
+        assert s.n_subplots == len(y)
+        assert s.subplot_spec["ncols"] == len(y)
+        assert s.subplot_spec["nrows"] == 1
+        assert s.subplot_spec["sharex"] is True
+        assert s.subplot_spec["sharey"] is False
+
     def test_col_faceted_y_paired(self):
 
         y = ["x", "y", "z"]

```


## Code snippets

### 1 - seaborn/axisgrid.py:

Start line: 2089, End line: 2156

```python
def pairplot(
    data, *,
    hue=None, hue_order=None, palette=None,
    vars=None, x_vars=None, y_vars=None,
    kind="scatter", diag_kind="auto", markers=None,
    height=2.5, aspect=1, corner=False, dropna=False,
    plot_kws=None, diag_kws=None, grid_kws=None, size=None,
):
    # Avoid circular import
    from .distributions import histplot, kdeplot

    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"'data' must be pandas DataFrame object, not: {type(data)}")

    plot_kws = {} if plot_kws is None else plot_kws.copy()
    diag_kws = {} if diag_kws is None else diag_kws.copy()
    grid_kws = {} if grid_kws is None else grid_kws.copy()

    # Resolve "auto" diag kind
    if diag_kind == "auto":
        if hue is None:
            diag_kind = "kde" if kind == "kde" else "hist"
        else:
            diag_kind = "hist" if kind == "hist" else "kde"

    # Set up the PairGrid
    grid_kws.setdefault("diag_sharey", diag_kind == "hist")
    grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
                    hue_order=hue_order, palette=palette, corner=corner,
                    height=height, aspect=aspect, dropna=dropna, **grid_kws)

    # Add the markers here as PairGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if markers is not None:
        if kind == "reg":
            # Needed until regplot supports style
            if grid.hue_names is None:
                n_markers = 1
            else:
                n_markers = len(grid.hue_names)
            if not isinstance(markers, list):
                markers = [markers] * n_markers
            if len(markers) != n_markers:
                raise ValueError("markers must be a singleton or a list of "
                                 "markers for each level of the hue variable")
            grid.hue_kws = {"marker": markers}
        elif kind == "scatter":
            if isinstance(markers, str):
                plot_kws["marker"] = markers
            elif hue is not None:
                plot_kws["style"] = data[hue]
                plot_kws["markers"] = markers

    # Draw the marginal plots on the diagonal
    diag_kws = diag_kws.copy()
    diag_kws.setdefault("legend", False)
    if diag_kind == "hist":
        grid.map_diag(histplot, **diag_kws)
    elif diag_kind == "kde":
        diag_kws.setdefault("fill", True)
        diag_kws.setdefault("warn_singular", False)
        grid.map_diag(kdeplot, **diag_kws)

    # Maybe plot on the off-diagonals
    if diag_kind is not None:
        plotter = grid.map_offdiag
    else:
        plotter = grid.map
    # ... other code
```
### 2 - seaborn/axisgrid.py:

Start line: 2158, End line: 2178

```python
def pairplot(
    data, *,
    hue=None, hue_order=None, palette=None,
    vars=None, x_vars=None, y_vars=None,
    kind="scatter", diag_kind="auto", markers=None,
    height=2.5, aspect=1, corner=False, dropna=False,
    plot_kws=None, diag_kws=None, grid_kws=None, size=None,
):
    # ... other code

    if kind == "scatter":
        from .relational import scatterplot  # Avoid circular import
        plotter(scatterplot, **plot_kws)
    elif kind == "reg":
        from .regression import regplot  # Avoid circular import
        plotter(regplot, **plot_kws)
    elif kind == "kde":
        from .distributions import kdeplot  # Avoid circular import
        plot_kws.setdefault("warn_singular", False)
        plotter(kdeplot, **plot_kws)
    elif kind == "hist":
        from .distributions import histplot  # Avoid circular import
        plotter(histplot, **plot_kws)

    # Add a legend
    if hue is not None:
        grid.add_legend()

    grid.tight_layout()

    return grid
```
### 3 - seaborn/_core/plot.py:

Start line: 894, End line: 987

```python
class Plotter:

    def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        subplot_spec = p._subplot_spec.copy()
        facet_spec = p._facet_spec.copy()
        pair_spec = p._pair_spec.copy()

        for axis in "xy":
            if axis in p._shares:
                subplot_spec[f"share{axis}"] = p._shares[axis]

        for dim in ["col", "row"]:
            if dim in common.frame and dim not in facet_spec["structure"]:
                order = categorical_order(common.frame[dim])
                facet_spec["structure"][dim] = order

        self._subplots = subplots = Subplots(subplot_spec, facet_spec, pair_spec)

        # --- Figure initialization
        self._figure = subplots.init_figure(
            pair_spec, self._pyplot, p._figure_spec, p._target,
        )

        # --- Figure annotation
        for sub in subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]

                # ~~ Axis labels

                # TODO Should we make it possible to use only one x/y label for
                # all rows/columns in a faceted plot? Maybe using sub{axis}label,
                # although the alignments of the labels from that method leaves
                # something to be desired (in terms of how it defines 'centered').
                names = [
                    common.names.get(axis_key),
                    *(layer["data"].names.get(axis_key) for layer in layers)
                ]
                auto_label = next((name for name in names if name is not None), None)
                label = self._resolve_label(p, axis_key, auto_label)
                ax.set(**{f"{axis}label": label})

                # ~~ Decoration visibility

                # TODO there should be some override (in Plot.layout?) so that
                # tick labels can be shown on interior shared axes
                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or axis in p._pair_spec and bool(p._pair_spec.get("wrap"))
                    or not p._pair_spec.get("cross", True)
                )
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = (
                    show_axis_label
                    or subplot_spec.get(f"share{axis}") not in (
                        True, "all", {"x": "col", "y": "row"}[axis]
                    )
                )
                for group in ("major", "minor"):
                    for t in getattr(axis_obj, f"get_{group}ticklabels")():
                        t.set_visible(show_tick_labels)

            # TODO we want right-side titles for row facets in most cases?
            # Let's have what we currently call "margin titles" but properly using the
            # ax.set_title interface (see my gist)
            title_parts = []
            for dim in ["col", "row"]:
                if sub[dim] is not None:
                    val = self._resolve_label(p, "title", f"{sub[dim]}")
                    if dim in p._labels:
                        key = self._resolve_label(p, dim, common.names.get(dim))
                        val = f"{key} {val}"
                    title_parts.append(val)

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and p._facet_spec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)
            elif not (has_col or has_row):
                title = self._resolve_label(p, "title", None)
                title_text = ax.set_title(title)
```
### 4 - seaborn/axisgrid.py:

Start line: 1, End line: 28

```python
from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from ._oldcore import VectorPlotter, variable_type, categorical_order
from . import utils
from .utils import (
    adjust_legend_subtitles, _check_argument, _draw_figure, _disable_autolayout
)
from .palettes import color_palette, blend_palette
from ._docstrings import (
    DocstringComponents,
    _core_docs,
)

__all__ = ["FacetGrid", "PairGrid", "JointGrid", "pairplot", "jointplot"]


_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
)
```
### 5 - seaborn/_core/plot.py:

Start line: 450, End line: 507

```python
@build_plot_signature
class Plot:

    def pair(
        self,
        x: VariableSpecList = None,
        y: VariableSpecList = None,
        wrap: int | None = None,
        cross: bool = True,
    ) -> Plot:
        """
        Produce subplots with distinct `x` and/or `y` variables.

        Parameters
        ----------
        x, y : sequence(s) of data identifiers
            Variables that will define the grid of subplots.
        wrap : int
            Maximum height/width of the grid, with additional subplots "wrapped"
            on the other dimension. Requires that only one of `x` or `y` are set here.
        cross : bool
            When True, define a two-dimensional grid using the Cartesian product of `x`
            and `y`.  Otherwise, define a one-dimensional grid by pairing `x` and `y`
            entries in by position.

        """
        # TODO Add transpose= arg, which would then draw pair(y=[...]) across rows
        # This may also be possible by setting `wrap=1`, but is that too unobvious?
        # TODO PairGrid features not currently implemented: diagonals, corner

        pair_spec: PairSpec = {}

        axes = {"x": [] if x is None else x, "y": [] if y is None else y}
        for axis, arg in axes.items():
            if isinstance(arg, (str, int)):
                err = f"You must pass a sequence of variable keys to `{axis}`"
                raise TypeError(err)

        pair_spec["variables"] = {}
        pair_spec["structure"] = {}

        for axis in "xy":
            keys = []
            for i, col in enumerate(axes[axis]):
                key = f"{axis}{i}"
                keys.append(key)
                pair_spec["variables"][key] = col

            if keys:
                pair_spec["structure"][axis] = keys

        if not cross and len(axes["x"]) != len(axes["y"]):
            err = "Lengths of the `x` and `y` lists must match with cross=False"
            raise ValueError(err)

        pair_spec["cross"] = cross
        pair_spec["wrap"] = wrap

        new = self._clone()
        new._pair_spec.update(pair_spec)
        return new
```
### 6 - examples/pairgrid_dotplot.py:

Start line: 1, End line: 39

```python
"""
Dot plot with several variables
===============================

_thumb: .3, .3
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the dataset
crashes = sns.load_dataset("car_crashes")

# Make the PairGrid
g = sns.PairGrid(crashes.sort_values("total", ascending=False),
                 x_vars=crashes.columns[:-3], y_vars=["abbrev"],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Total crashes", "Speeding crashes", "Alcohol crashes",
          "Not distracted crashes", "No previous crashes"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
```
### 7 - examples/paired_pointplots.py:

Start line: 1, End line: 21

```python
"""
Paired categorical plots
========================

"""
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Set up a grid to plot survival probability against several variables
g = sns.PairGrid(titanic, y_vars="survived",
                 x_vars=["class", "sex", "who", "alone"],
                 height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
```
### 8 - seaborn/_core/plot.py:

Start line: 1324, End line: 1364

```python
class Plotter:

    def _generate_pairings(
        self, data: PlotData, pair_variables: dict,
    ) -> Generator[
        tuple[list[dict], DataFrame, dict[str, Scale]], None, None
    ]:
        # TODO retype return with subplot_spec or similar

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [axis]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for view in self._subplots:
                if (view["x"] == x) and (view["y"] == y):
                    subplots.append(view)

            if data.frame.empty and data.frames:
                out_df = data.frames[(x, y)].copy()
            elif not pair_variables:
                out_df = data.frame.copy()
            else:
                if data.frame.empty and data.frames:
                    out_df = data.frames[(x, y)].copy()
                else:
                    out_df = data.frame.copy()

            scales = self._scales.copy()
            if x in out_df:
                scales["x"] = self._scales[x]
            if y in out_df:
                scales["y"] = self._scales[y]

            for axis, var in zip("xy", (x, y)):
                if axis != var:
                    out_df = out_df.rename(columns={var: axis})
                    cols = [col for col in out_df if re.match(rf"{axis}\d+", col)]
                    out_df = out_df.drop(cols, axis=1)

            yield subplots, out_df, scales
```
### 9 - seaborn/_core/plot.py:

Start line: 1080, End line: 1198

```python
class Plotter:

    def _setup_scales(
        self, p: Plot,
        common: PlotData,
        layers: list[Layer],
        variables: list[str] | None = None,
    ) -> None:

        if variables is None:
            # Add variables that have data but not a scale, which happens
            # because this method can be called multiple time, to handle
            # variables added during the Stat transform.
            variables = []
            for layer in layers:
                variables.extend(layer["data"].frame.columns)
                for df in layer["data"].frames.values():
                    variables.extend(v for v in df if v not in variables)
            variables = [v for v in variables if v not in self._scales]

        for var in variables:

            # Determine whether this is a coordinate variable
            # (i.e., x/y, paired x/y, or derivative such as xmax)
            m = re.match(r"^(?P<coord>(?P<axis>x|y)\d*).*", var)
            if m is None:
                coord = axis = None
            else:
                coord = m["coord"]
                axis = m["axis"]

            # Get keys that handle things like x0, xmax, properly where relevant
            prop_key = var if axis is None else axis
            scale_key = var if coord is None else coord

            if prop_key not in PROPERTIES:
                continue

            # Concatenate layers, using only the relevant coordinate and faceting vars,
            # This is unnecessarily wasteful, as layer data will often be redundant.
            # But figuring out the minimal amount we need is more complicated.
            cols = [var, "col", "row"]
            parts = [common.frame.filter(cols)]
            for layer in layers:
                parts.append(layer["data"].frame.filter(cols))
                for df in layer["data"].frames.values():
                    parts.append(df.filter(cols))
            var_df = pd.concat(parts, ignore_index=True)

            prop = PROPERTIES[prop_key]
            scale = self._get_scale(p, scale_key, prop, var_df[var])

            if scale_key not in p._variables:
                # TODO this implies that the variable was added by the stat
                # It allows downstream orientation inference to work properly.
                # But it feels rather hacky, so ideally revisit.
                scale._priority = 0  # type: ignore

            if axis is None:
                # We could think about having a broader concept of (un)shared properties
                # In general, not something you want to do (different scales in facets)
                # But could make sense e.g. with paired plots. Build later.
                share_state = None
                subplots = []
            else:
                share_state = self._subplots.subplot_spec[f"share{axis}"]
                subplots = [view for view in self._subplots if view[axis] == coord]

            # Shared categorical axes are broken on matplotlib<3.4.0.
            # https://github.com/matplotlib/matplotlib/pull/18308
            # This only affects us when sharing *paired* axes. This is a novel/niche
            # behavior, so we will raise rather than hack together a workaround.
            if axis is not None and Version(mpl.__version__) < Version("3.4.0"):
                from seaborn._core.scales import Nominal
                paired_axis = axis in p._pair_spec
                cat_scale = isinstance(scale, Nominal)
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            if scale is None:
                self._scales[var] = Scale._identity()
            else:
                self._scales[var] = scale._setup(var_df[var], prop)

            # Everything below here applies only to coordinate variables
            # We additionally skip it when we're working with a value
            # that is derived from a coordinate we've already processed.
            # e.g., the Stat consumed y and added ymin/ymax. In that case,
            # we've already setup the y scale and ymin/max are in scale space.
            if axis is None or (var != coord and coord in p._variables):
                continue

            # Set up an empty series to receive the transformed values.
            # We need this to handle piecemeal transforms of categories -> floats.
            transformed_data = []
            for layer in layers:
                index = layer["data"].frame.index
                empty_series = pd.Series(dtype=float, index=index, name=var)
                transformed_data.append(empty_series)

            for view in subplots:

                axis_obj = getattr(view["ax"], f"{axis}axis")
                seed_values = self._get_subplot_data(var_df, var, view, share_state)
                view_scale = scale._setup(seed_values, prop, axis=axis_obj)
                set_scale_obj(view["ax"], axis, view_scale._matplotlib_scale)

                for layer, new_series in zip(layers, transformed_data):
                    layer_df = layer["data"].frame
                    if var in layer_df:
                        idx = self._get_subplot_index(layer_df, view)
                        new_series.loc[idx] = view_scale(layer_df.loc[idx, var])

            # Now the transformed data series are complete, set update the layer data
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer["data"].frame
                if var in layer_df:
                    layer_df[var] = new_series
```
### 10 - examples/marginal_ticks.py:

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
### 14 - seaborn/_core/subplots.py:

Start line: 47, End line: 77

```python
class Subplots:

    def _check_dimension_uniqueness(
        self, facet_spec: FacetSpec, pair_spec: PairSpec
    ) -> None:
        """Reject specs that pair and facet on (or wrap to) same figure dimension."""
        err = None

        facet_vars = facet_spec.get("variables", {})

        if facet_spec.get("wrap") and {"col", "row"} <= set(facet_vars):
            err = "Cannot wrap facets when specifying both `col` and `row`."
        elif (
            pair_spec.get("wrap")
            and pair_spec.get("cross", True)
            and len(pair_spec.get("structure", {}).get("x", [])) > 1
            and len(pair_spec.get("structure", {}).get("y", [])) > 1
        ):
            err = "Cannot wrap subplots when pairing on both `x` and `y`."

        collisions = {"x": ["columns", "rows"], "y": ["rows", "columns"]}
        for pair_axis, (multi_dim, wrap_dim) in collisions.items():
            if pair_axis not in pair_spec.get("structure", {}):
                continue
            elif multi_dim[:3] in facet_vars:
                err = f"Cannot facet the {multi_dim} while pairing on `{pair_axis}``."
            elif wrap_dim[:3] in facet_vars and facet_spec.get("wrap"):
                err = f"Cannot wrap the {wrap_dim} while pairing on `{pair_axis}``."
            elif wrap_dim[:3] in facet_vars and pair_spec.get("wrap"):
                err = f"Cannot wrap the {multi_dim} while faceting the {wrap_dim}."

        if err is not None:
            raise RuntimeError(err)  # TODO what err class? Define PlotSpecError?
```
### 21 - seaborn/_core/subplots.py:

Start line: 103, End line: 120

```python
class Subplots:

    def _handle_wrapping(
        self, facet_spec: FacetSpec, pair_spec: PairSpec
    ) -> None:
        """Update figure structure parameters based on facet/pair wrapping."""
        self.wrap = wrap = facet_spec.get("wrap") or pair_spec.get("wrap")
        if not wrap:
            return

        wrap_dim = "row" if self.subplot_spec["nrows"] > 1 else "col"
        flow_dim = {"row": "col", "col": "row"}[wrap_dim]
        n_subplots = self.subplot_spec[f"n{wrap_dim}s"]
        flow = int(np.ceil(n_subplots / wrap))

        if wrap < self.subplot_spec[f"n{wrap_dim}s"]:
            self.subplot_spec[f"n{wrap_dim}s"] = wrap
        self.subplot_spec[f"n{flow_dim}s"] = flow
        self.n_subplots = n_subplots
        self.wrap_dim = wrap_dim
```
### 23 - seaborn/_core/plot.py:

Start line: 1292, End line: 1322

```python
class Plotter:

    def _unscale_coords(
        self, subplots: list[dict], df: DataFrame, orient: str,
    ) -> DataFrame:
        # TODO do we still have numbers in the variable name at this point?
        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        drop_cols = [*coord_cols, "width"] if "width" in df else coord_cols
        out_df = (
            df
            .drop(drop_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
            .copy(deep=False)
        )

        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            for var, values in axes_df.items():

                axis = getattr(view["ax"], f"{var[0]}axis")
                # TODO see https://github.com/matplotlib/matplotlib/issues/22713
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, var] = inverted

                if var == orient and "width" in view_df:
                    width = view_df["width"]
                    out_df.loc[values.index, "width"] = (
                        transform(values + width / 2) - transform(values - width / 2)
                    )

        return out_df
```
### 24 - seaborn/_core/plot.py:

Start line: 1200, End line: 1268

```python
class Plotter:

    def _plot_layer(self, p: Plot, layer: Layer) -> None:

        data = layer["data"]
        mark = layer["mark"]
        move = layer["move"]

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        grouping_properties = [v for v in PROPERTIES if v[0] not in "xy"]

        pair_variables = p._pair_spec.get("structure", {})

        for subplots, df, scales in self._generate_pairings(data, pair_variables):

            orient = layer["orient"] or mark._infer_orient(scales)

            def get_order(var):
                # Ignore order for x/y: they have been scaled to numeric indices,
                # so any original order is no longer valid. Default ordering rules
                # sorted unique numbers will correctly reconstruct intended order
                # TODO This is tricky, make sure we add some tests for this
                if var not in "xy" and var in scales:
                    return getattr(scales[var], "order", None)

            if "width" in mark._mappable_props:
                width = mark._resolve(df, "width", None)
            else:
                width = df.get("width", 0.8)  # TODO what default
            if orient in df:
                df["width"] = width * scales[orient]._spacing(df[orient])

            if "baseline" in mark._mappable_props:
                # TODO what marks should have this?
                # If we can set baseline with, e.g., Bar(), then the
                # "other" (e.g. y for x oriented bars) parameterization
                # is somewhat ambiguous.
                baseline = mark._resolve(df, "baseline", None)
            else:
                # TODO unlike width, we might not want to add baseline to data
                # if the mark doesn't use it. Practically, there is a concern about
                # Mark abstraction like Area / Ribbon
                baseline = df.get("baseline", 0)
            df["baseline"] = baseline

            if move is not None:
                moves = move if isinstance(move, list) else [move]
                for move_step in moves:
                    move_by = getattr(move_step, "by", None)
                    if move_by is None:
                        move_by = grouping_properties
                    move_groupers = [*move_by, *default_grouping_vars]
                    if move_step.group_by_orient:
                        move_groupers.insert(0, orient)
                    order = {var: get_order(var) for var in move_groupers}
                    groupby = GroupBy(order)
                    df = move_step(df, groupby, orient, scales)

            df = self._unscale_coords(subplots, df, orient)

            grouping_vars = mark._grouping_props + default_grouping_vars
            split_generator = self._setup_split_generator(grouping_vars, df, subplots)

            mark._plot(split_generator, scales, orient)

        # TODO is this the right place for this?
        for view in self._subplots:
            view["ax"].autoscale_view()

        if layer["legend"]:
            self._update_legend_contents(p, mark, data, scales)
```
### 35 - seaborn/_core/plot.py:

Start line: 1, End line: 98

```python
"""The classes for specifying and compiling a declarative visualization."""
from __future__ import annotations

import io
import os
import re
import sys
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Optional, cast

from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import DataSource, VariableSpec, VariableSpecList, OrderSpec
from seaborn._core.rules import categorical_order
from seaborn._compat import set_scale_obj
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from seaborn.external.version import Version

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import SubFigure


if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# ---- Definitions for internal specs --------------------------------- #


class Layer(TypedDict, total=False):

    mark: Mark  # TODO allow list?
    stat: Stat | None  # TODO allow list?
    move: Move | list[Move] | None
    data: PlotData
    source: DataSource
    vars: dict[str, VariableSpec]
    orient: str
    legend: bool


class FacetSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    wrap: int | None


class PairSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    cross: bool
    wrap: int | None


# --- Local helpers ----------------------------------------------------------------

class Default:
    def __repr__(self):
        return "<default>"


default = Default()


@contextmanager
def theme_context(params: dict[str, Any]) -> Generator:
    """Temporarily modify specifc matplotlib rcParams."""
    orig = {k: mpl.rcParams[k] for k in params}
    try:
        mpl.rcParams.update(params)
        yield
    finally:
        mpl.rcParams.update(orig)
```
### 43 - seaborn/_core/plot.py:

Start line: 1388, End line: 1461

```python
class Plotter:

    def _setup_split_generator(
        self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO will need to recreate previous categorical plots

        grouping_keys = []
        grouping_vars = [
            v for v in grouping_vars if v in df and v not in ["col", "row"]
        ]
        for var in grouping_vars:
            order = getattr(self._scales[var], "order", None)
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator(keep_na=False) -> Generator:

            for view in subplots:

                axes_df = self._filter_subplot_data(df, view)

                with pd.option_context("mode.use_inf_as_null", True):
                    if keep_na:
                        # The simpler thing to do would be x.dropna().reindex(x.index).
                        # But that doesn't work with the way that the subset iteration
                        # is written below, which assumes data for grouping vars.
                        # Matplotlib (usually?) masks nan data, so this should "work".
                        # Downstream code can also drop these rows, at some speed cost.
                        present = axes_df.notna().all(axis=1)
                        nulled = {}
                        for axis in "xy":
                            if axis in axes_df:
                                nulled[axis] = axes_df[axis].where(present)
                        axes_df = axes_df.assign(**nulled)
                    else:
                        axes_df = axes_df.dropna()

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if view[dim] is not None:
                        subplot_keys[dim] = view[dim]

                if not grouping_vars or not any(grouping_keys):
                    yield subplot_keys, axes_df.copy(), view["ax"]
                    continue

                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False)

                for key in itertools.product(*grouping_keys):

                    # Pandas fails with singleton tuple inputs
                    pd_key = key[0] if len(key) == 1 else key

                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        # TODO (from initial work on categorical plots refactor)
                        # We are adding this to allow backwards compatability
                        # with the empty artists that old categorical plots would
                        # add (before 0.12), which we may decide to break, in which
                        # case this option could be removed
                        df_subset = axes_df.loc[[]]

                    if df_subset.empty and not allow_empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    # TODO need copy(deep=...) policy (here, above, anywhere else?)
                    yield sub_vars, df_subset.copy(), view["ax"]

        return split_generator
```
### 45 - seaborn/_core/plot.py:

Start line: 989, End line: 1041

```python
class Plotter:

    def _compute_stats(self, spec: Plot, layers: list[Layer]) -> None:

        grouping_vars = [v for v in PROPERTIES if v not in "xy"]
        grouping_vars += ["col", "row", "group"]

        pair_vars = spec._pair_spec.get("structure", {})

        for layer in layers:

            data = layer["data"]
            mark = layer["mark"]
            stat = layer["stat"]

            if stat is None:
                continue

            iter_axes = itertools.product(*[
                pair_vars.get(axis, [axis]) for axis in "xy"
            ])

            old = data.frame

            if pair_vars:
                data.frames = {}
                data.frame = data.frame.iloc[:0]  # TODO to simplify typing

            for coord_vars in iter_axes:

                pairings = "xy", coord_vars

                df = old.copy()
                scales = self._scales.copy()

                for axis, var in zip(*pairings):
                    if axis != var:
                        df = df.rename(columns={var: axis})
                        drop_cols = [x for x in df if re.match(rf"{axis}\d+", x)]
                        df = df.drop(drop_cols, axis=1)
                        scales[axis] = scales[var]

                orient = layer["orient"] or mark._infer_orient(scales)

                if stat.group_by_orient:
                    grouper = [orient, *grouping_vars]
                else:
                    grouper = grouping_vars
                groupby = GroupBy(grouper)
                res = stat(df, groupby, orient, scales)

                if pair_vars:
                    data.frames[coord_vars] = res
                else:
                    data.frame = res
```
### 60 - seaborn/_core/plot.py:

Start line: 1270, End line: 1290

```python
class Plotter:

    def _scale_coords(self, subplots: list[dict], df: DataFrame) -> DataFrame:
        # TODO stricter type on subplots

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .copy(deep=False)
            .drop(coord_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            with pd.option_context("mode.use_inf_as_null", True):
                axes_df = axes_df.dropna()
            for var, values in axes_df.items():
                scale = view[f"{var[0]}scale"]
                out_df.loc[values.index, var] = scale(values)

        return out_df
```
### 79 - seaborn/_core/plot.py:

Start line: 294, End line: 324

```python
@build_plot_signature
class Plot:

    def _theme_with_defaults(self) -> dict[str, Any]:

        style_groups = [
            "axes", "figure", "font", "grid", "hatch", "legend", "lines",
            "mathtext", "markers", "patch", "savefig", "scatter",
            "xaxis", "xtick", "yaxis", "ytick",
        ]
        base = {
            k: mpl.rcParamsDefault[k] for k in mpl.rcParams
            if any(k.startswith(p) for p in style_groups)
        }
        theme = {
            **base,
            **axes_style("darkgrid"),
            **plotting_context("notebook"),
            "axes.prop_cycle": cycler("color", color_palette("deep")),
        }
        theme.update(self._theme)
        return theme

    @property
    def _variables(self) -> list[str]:

        variables = (
            list(self._data.frame)
            + list(self._pair_spec.get("variables", []))
            + list(self._facet_spec.get("variables", []))
        )
        for layer in self._layers:
            variables.extend(c for c in layer["vars"] if c not in variables)
        return variables
```
### 81 - seaborn/_core/plot.py:

Start line: 863, End line: 892

```python
class Plotter:

    def _extract_data(self, p: Plot) -> tuple[PlotData, list[Layer]]:

        common_data = (
            p._data
            .join(None, p._facet_spec.get("variables"))
            .join(None, p._pair_spec.get("variables"))
        )

        layers: list[Layer] = []
        for layer in p._layers:
            spec = layer.copy()
            spec["data"] = common_data.join(layer.get("source"), layer.get("vars"))
            layers.append(spec)

        return common_data, layers

    def _resolve_label(self, p: Plot, var: str, auto_label: str | None) -> str:

        label: str
        if var in p._labels:
            manual_label = p._labels[var]
            if callable(manual_label) and auto_label is not None:
                label = manual_label(auto_label)
            else:
                label = cast(str, manual_label)
        elif auto_label is None:
            label = ""
        else:
            label = auto_label
        return label
```
### 92 - seaborn/_core/plot.py:

Start line: 592, End line: 605

```python
@build_plot_signature
class Plot:

    def share(self, **shares: bool | str) -> Plot:
        """
        Control sharing of axis limits and ticks across subplots.

        Keywords correspond to variables defined in the plot, and values can be
        boolean (to share across all subplots), or one of "row" or "col" (to share
        more selectively across one dimension of a grid).

        Behavior for non-coordinate variables is currently undefined.

        """
        new = self._clone()
        new._shares.update(shares)
        return new
```
### 105 - seaborn/_core/subplots.py:

Start line: 122, End line: 141

```python
class Subplots:

    def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
        """Update subplot spec with default or specified axis sharing parameters."""
        axis_to_dim = {"x": "col", "y": "row"}
        key: str
        val: str | bool
        for axis in "xy":
            key = f"share{axis}"
            # Always use user-specified value, if present
            if key not in self.subplot_spec:
                if axis in pair_spec.get("structure", {}):
                    # Paired axes are shared along one dimension by default
                    if self.wrap in [None, 1] and pair_spec.get("cross", True):
                        val = axis_to_dim[axis]
                    else:
                        val = False
                else:
                    # This will pick up faceted plots, as well as single subplot
                    # figures, where the value doesn't really matter
                    val = True
                self.subplot_spec[key] = val
```
### 107 - seaborn/_core/subplots.py:

Start line: 143, End line: 221

```python
class Subplots:

    def init_figure(
        self,
        pair_spec: PairSpec,
        pyplot: bool = False,
        figure_kws: dict | None = None,
        target: Axes | Figure | SubFigure = None,
    ) -> Figure:
        """Initialize matplotlib objects and add seaborn-relevant metadata."""
        # TODO reduce need to pass pair_spec here?

        if figure_kws is None:
            figure_kws = {}

        if isinstance(target, mpl.axes.Axes):

            if max(self.subplot_spec["nrows"], self.subplot_spec["ncols"]) > 1:
                err = " ".join([
                    "Cannot create multiple subplots after calling `Plot.on` with",
                    f"a {mpl.axes.Axes} object.",
                ])
                try:
                    err += f" You may want to use a {mpl.figure.SubFigure} instead."
                except AttributeError:  # SubFigure added in mpl 3.4
                    pass
                raise RuntimeError(err)

            self._subplot_list = [{
                "ax": target,
                "left": True,
                "right": True,
                "top": True,
                "bottom": True,
                "col": None,
                "row": None,
                "x": "x",
                "y": "y",
            }]
            self._figure = target.figure
            return self._figure

        elif (
            hasattr(mpl.figure, "SubFigure")  # Added in mpl 3.4
            and isinstance(target, mpl.figure.SubFigure)
        ):
            figure = target.figure
        elif isinstance(target, mpl.figure.Figure):
            figure = target
        else:
            if pyplot:
                figure = plt.figure(**figure_kws)
            else:
                figure = mpl.figure.Figure(**figure_kws)
            target = figure
        self._figure = figure

        axs = target.subplots(**self.subplot_spec, squeeze=False)

        if self.wrap:
            # Remove unused Axes and flatten the rest into a (2D) vector
            axs_flat = axs.ravel({"col": "C", "row": "F"}[self.wrap_dim])
            axs, extra = np.split(axs_flat, [self.n_subplots])
            for ax in extra:
                ax.remove()
            if self.wrap_dim == "col":
                axs = axs[np.newaxis, :]
            else:
                axs = axs[:, np.newaxis]

        # Get i, j coordinates for each Axes object
        # Note that i, j are with respect to faceting/pairing,
        # not the subplot grid itself, (which only matters in the case of wrapping).
        iter_axs: np.ndenumerate | zip
        if not pair_spec.get("cross", True):
            indices = np.arange(self.n_subplots)
            iter_axs = zip(zip(indices, indices), axs.flat)
        else:
            iter_axs = np.ndenumerate(axs)

        self._subplot_list = []
        # ... other code
```
### 116 - seaborn/_core/plot.py:

Start line: 567, End line: 590

```python
@build_plot_signature
class Plot:

    # TODO def twin()?

    def scale(self, **scales: Scale) -> Plot:
        """
        Control mappings from data units to visual properties.

        Keywords correspond to variables defined in the plot, including coordinate
        variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

        A number of "magic" arguments are accepted, including:
            - The name of a transform (e.g., `"log"`, `"sqrt"`)
            - The name of a palette (e.g., `"viridis"`, `"muted"`)
            - A tuple of values, defining the output range (e.g. `(1, 5)`)
            - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
            - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

        For more explicit control, pass a scale spec object such as :class:`Continuous`
        or :class:`Nominal`. Or use `None` to use an "identity" scale, which treats data
        values as literally encoding visual properties.

        """
        new = self._clone()
        new._scales.update(scales)
        return new
```
### 125 - seaborn/_core/plot.py:

Start line: 743, End line: 777

```python
@build_plot_signature
class Plot:

    def _plot(self, pyplot: bool = False) -> Plotter:

        # TODO if we have _target object, pyplot should be determined by whether it
        # is hooked into the pyplot state machine (how do we check?)

        plotter = Plotter(pyplot=pyplot, theme=self._theme_with_defaults())

        # Process the variable assignments and initialize the figure
        common, layers = plotter._extract_data(self)
        plotter._setup_figure(self, common, layers)

        # Process the scale spec for coordinate variables and transform their data
        coord_vars = [v for v in self._variables if re.match(r"^x|y", v)]
        plotter._setup_scales(self, common, layers, coord_vars)

        # Apply statistical transform(s)
        plotter._compute_stats(self, layers)

        # Process scale spec for semantic variables and coordinates computed by stat
        plotter._setup_scales(self, common, layers)

        # TODO Remove these after updating other methods
        # ---- Maybe have debug= param that attaches these when True?
        plotter._data = common
        plotter._layers = layers

        # Process the data for each layer and add matplotlib artists
        for layer in layers:
            plotter._plot_layer(self, layer)

        # Add various figure decorations
        plotter._make_legend(self)
        plotter._finalize_figure(self)

        return plotter
```
### 126 - seaborn/_core/plot.py:

Start line: 1043, End line: 1078

```python
class Plotter:

    def _get_scale(
        self, spec: Plot, var: str, prop: Property, values: Series
    ) -> Scale:

        if var in spec._scales:
            arg = spec._scales[var]
            if arg is None or isinstance(arg, Scale):
                scale = arg
            else:
                scale = prop.infer_scale(arg, values)
        else:
            scale = prop.default_scale(values)

        return scale

    def _get_subplot_data(self, df, var, view, share_state):

        if share_state in [True, "all"]:
            # The all-shared case is easiest, every subplot sees all the data
            seed_values = df[var]
        else:
            # Otherwise, we need to setup separate scales for different subplots
            if share_state in [False, "none"]:
                # Fully independent axes are also easy: use each subplot's data
                idx = self._get_subplot_index(df, view)
            elif share_state in df:
                # Sharing within row/col is more complicated
                use_rows = df[share_state] == view[share_state]
                idx = df.index[use_rows]
            else:
                # This configuration doesn't make much sense, but it's fine
                idx = df.index

            seed_values = df.loc[idx, var]

        return seed_values
```
### 132 - seaborn/_core/plot.py:

Start line: 132, End line: 218

```python
# ---- The main interface for declarative plotting -------------------- #


@build_plot_signature
class Plot:
    """
    An interface for declaratively specifying statistical graphics.

    Plots are constructed by initializing this class and adding one or more
    layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
    faceting variables or variable pairings may be defined to divide the space
    into multiple subplots. The mappings from data values to visual properties
    can be parametrized using scales, although the plot will try to infer good
    defaults when scales are not explicitly defined.

    The constructor accepts a data source (a :class:`pandas.DataFrame` or
    dictionary with columnar values) and variable assignments. Variables can be
    passed as keys to the data source or directly as data vectors.  If multiple
    data-containing objects are provided, they will be index-aligned.

    The data source and variables defined in the constructor will be used for
    all layers in the plot, unless overridden or disabled when adding a layer.

    The following variables can be defined in the constructor:
        {known_properties}

    The `data`, `x`, and `y` variables can be passed as positional arguments or
    using keywords. Whether the first positional argument is interpreted as a
    data source or `x` variable depends on its type.

    The methods of this class return a copy of the instance; use chaining to
    build up a plot through multiple calls. Methods can be called in any order.

    Most methods only add information to the plot spec; no actual processing
    happens until the plot is shown or saved. It is also possible to compile
    the plot without rendering it to access the lower-level representation.

    """
    _data: PlotData
    _layers: list[Layer]

    _scales: dict[str, Scale]
    _shares: dict[str, bool | str]
    _limits: dict[str, tuple[Any, Any]]
    _labels: dict[str, str | Callable[[str], str]]
    _theme: dict[str, Any]

    _facet_spec: FacetSpec
    _pair_spec: PairSpec

    _figure_spec: dict[str, Any]
    _subplot_spec: dict[str, Any]
    _layout_spec: dict[str, Any]

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        if args:
            data, variables = self._resolve_positionals(args, data, variables)

        unknown = [x for x in variables if x not in PROPERTIES]
        if unknown:
            err = f"Plot() got unexpected keyword argument(s): {', '.join(unknown)}"
            raise TypeError(err)

        self._data = PlotData(data, variables)

        self._layers = []

        self._scales = {}
        self._shares = {}
        self._limits = {}
        self._labels = {}
        self._theme = {}

        self._facet_spec = {}
        self._pair_spec = {}

        self._figure_spec = {}
        self._subplot_spec = {}
        self._layout_spec = {}

        self._target = None
```
### 145 - seaborn/_core/plot.py:

Start line: 1556, End line: 1581

```python
class Plotter:

    def _finalize_figure(self, p: Plot) -> None:

        for sub in self._subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]

                # Axis limits
                if axis_key in p._limits:
                    convert_units = getattr(ax, f"{axis}axis").convert_units
                    a, b = p._limits[axis_key]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f"{axis}lim": (lo, hi)})

        algo_default = None if p._target is not None else "tight"
        layout_algo = p._layout_spec.get("algo", algo_default)
        if layout_algo == "tight":
            self._figure.set_tight_layout(True)
        elif layout_algo == "constrained":
            self._figure.set_constrained_layout(True)
```
### 148 - seaborn/_core/plot.py:

Start line: 824, End line: 861

```python
class Plotter:

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        # TODO better to do this through a Jupyter hook? e.g.
        # ipy = IPython.core.formatters.get_ipython()
        # fmt = ipy.display_formatter.formatters["text/html"]
        # fmt.for_type(Plot, ...)
        # Would like to have a svg option too, not sure how to make that flexible

        # TODO use matplotlib backend directly instead of going through savefig?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.

        # TODO need to decide what the right default behavior here is:
        # - Use dpi=72 to match default InlineBackend figure size?
        # - Accept a generic "scaling" somewhere and scale DPI from that,
        #   either with 1x -> 72 or 1x -> 96 and the default scaling be .75?
        # - Listen to rcParams? InlineBackend behavior makes that so complicated :(
        # - Do we ever want to *not* use retina mode at this point?

        from PIL import Image

        dpi = 96
        buffer = io.BytesIO()

        with theme_context(self._theme):
            self._figure.savefig(buffer, dpi=dpi * 2, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        scaling = .85 / 2
        w, h = Image.open(buffer).size
        metadata = {"width": w * scaling, "height": h * scaling}
        return data, metadata
```
### 155 - seaborn/_core/plot.py:

Start line: 509, End line: 565

```python
@build_plot_signature
class Plot:

    def facet(
        self,
        # TODO require kwargs?
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[str, OrderSpec] = None,
        wrap: int | None = None,
    ) -> Plot:
        """
        Produce subplots with conditional subsets of the data.

        Parameters
        ----------
        col, row : data vectors or identifiers
            Variables used to define subsets along the columns and/or rows of the grid.
            Can be references to the global data source passed in the constructor.
        order : list of strings, or dict with dimensional keys
            Define the order of the faceting variables.
        wrap : int
            Maximum height/width of the grid, with additional subplots "wrapped"
            on the other dimension. Requires that only one of `x` or `y` are set here.

        """
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        structure = {}
        if isinstance(order, dict):
            for dim in ["col", "row"]:
                dim_order = order.get(dim)
                if dim_order is not None:
                    structure[dim] = list(dim_order)
        elif order is not None:
            if col is not None and row is not None:
                err = " ".join([
                    "When faceting on both col= and row=, passing `order` as a list"
                    "is ambiguous. Use a dict with 'col' and/or 'row' keys instead."
                ])
                raise RuntimeError(err)
            elif col is not None:
                structure["col"] = list(order)
            elif row is not None:
                structure["row"] = list(order)

        spec: FacetSpec = {
            "variables": variables,
            "structure": structure,
            "wrap": wrap,
        }

        new = self._clone()
        new._facet_spec.update(spec)

        return new
```
### 161 - seaborn/_core/plot.py:

Start line: 1366, End line: 1386

```python
class Plotter:

    def _get_subplot_index(self, df: DataFrame, subplot: dict) -> DataFrame:

        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df.index

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df.index[keep_rows]

    def _filter_subplot_data(self, df: DataFrame, subplot: dict) -> DataFrame:
        # TODO note redundancies with preceding function ... needs refactoring
        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]
```
### 169 - seaborn/_core/subplots.py:

Start line: 222, End line: 271

```python
class Subplots:

    def init_figure(
        self,
        pair_spec: PairSpec,
        pyplot: bool = False,
        figure_kws: dict | None = None,
        target: Axes | Figure | SubFigure = None,
    ) -> Figure:
        # ... other code
        for (i, j), ax in iter_axs:

            info = {"ax": ax}

            nrows, ncols = self.subplot_spec["nrows"], self.subplot_spec["ncols"]
            if not self.wrap:
                info["left"] = j % ncols == 0
                info["right"] = (j + 1) % ncols == 0
                info["top"] = i == 0
                info["bottom"] = i == nrows - 1
            elif self.wrap_dim == "col":
                info["left"] = j % ncols == 0
                info["right"] = ((j + 1) % ncols == 0) or ((j + 1) == self.n_subplots)
                info["top"] = j < ncols
                info["bottom"] = j >= (self.n_subplots - ncols)
            elif self.wrap_dim == "row":
                info["left"] = i < nrows
                info["right"] = i >= self.n_subplots - nrows
                info["top"] = i % nrows == 0
                info["bottom"] = ((i + 1) % nrows == 0) or ((i + 1) == self.n_subplots)

            if not pair_spec.get("cross", True):
                info["top"] = j < ncols
                info["bottom"] = j >= self.n_subplots - ncols

            for dim in ["row", "col"]:
                idx = {"row": i, "col": j}[dim]
                info[dim] = self.grid_dimensions[dim][idx]

            for axis in "xy":

                idx = {"x": j, "y": i}[axis]
                if axis in pair_spec.get("structure", {}):
                    key = f"{axis}{idx}"
                else:
                    key = axis
                info[axis] = key

            self._subplot_list.append(info)

        return figure

    def __iter__(self) -> Generator[dict, None, None]:  # TODO TypedDict?
        """Yield each subplot dictionary with Axes object and metadata."""
        yield from self._subplot_list

    def __len__(self) -> int:
        """Return the number of subplots in this figure."""
        return len(self._subplot_list)
```
### 196 - seaborn/_core/subplots.py:

Start line: 79, End line: 101

```python
class Subplots:

    def _determine_grid_dimensions(
        self, facet_spec: FacetSpec, pair_spec: PairSpec
    ) -> None:
        """Parse faceting and pairing information to define figure structure."""
        self.grid_dimensions: dict[str, list] = {}
        for dim, axis in zip(["col", "row"], ["x", "y"]):

            facet_vars = facet_spec.get("variables", {})
            if dim in facet_vars:
                self.grid_dimensions[dim] = facet_spec["structure"][dim]
            elif axis in pair_spec.get("structure", {}):
                self.grid_dimensions[dim] = [
                    None for _ in pair_spec.get("structure", {})[axis]
                ]
            else:
                self.grid_dimensions[dim] = [None]

            self.subplot_spec[f"n{dim}s"] = len(self.grid_dimensions[dim])

        if not pair_spec.get("cross", True):
            self.subplot_spec["nrows"] = 1

        self.n_subplots = self.subplot_spec["ncols"] * self.subplot_spec["nrows"]
```
### 197 - seaborn/_core/subplots.py:

Start line: 1, End line: 45

```python
from __future__ import annotations
from collections.abc import Generator

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # TODO move to seaborn._core.typing?
    from seaborn._core.plot import FacetSpec, PairSpec
    from matplotlib.figure import SubFigure


class Subplots:
    """
    Interface for creating and using matplotlib subplots based on seaborn parameters.

    Parameters
    ----------
    subplot_spec : dict
        Keyword args for :meth:`matplotlib.figure.Figure.subplots`.
    facet_spec : dict
        Parameters that control subplot faceting.
    pair_spec : dict
        Parameters that control subplot pairing.
    data : PlotData
        Data used to define figure setup.

    """
    def __init__(
        # TODO defined TypedDict types for these specs
        self,
        subplot_spec: dict,
        facet_spec: FacetSpec,
        pair_spec: PairSpec,
    ):

        self.subplot_spec = subplot_spec

        self._check_dimension_uniqueness(facet_spec, pair_spec)
        self._determine_grid_dimensions(facet_spec, pair_spec)
        self._handle_wrapping(facet_spec, pair_spec)
        self._determine_axis_sharing(pair_spec)
```
### 218 - seaborn/_core/plot.py:

Start line: 1463, End line: 1505

```python
class Plotter:

    def _update_legend_contents(
        self,
        p: Plot,
        mark: Mark,
        data: PlotData,
        scales: dict[str, Scale],
    ) -> None:
        """Add legend artists / labels for one layer in the plot."""
        if data.frame.empty and data.frames:
            legend_vars = set()
            for frame in data.frames.values():
                legend_vars.update(frame.columns.intersection(scales))
        else:
            legend_vars = data.frame.columns.intersection(scales)

        # First pass: Identify the values that will be shown for each variable
        schema: list[tuple[
            tuple[str, str | int], list[str], tuple[list, list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            var_legend = scales[var]._legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    if data.ids[var] == part_id:
                        # Allow multiple plot semantics to represent same data variable
                        part_vars.append(var)
                        break
                else:
                    title = self._resolve_label(p, var, data.names[var])
                    entry = (title, data.ids[var]), [var], (values, labels)
                    schema.append(entry)

        # Second pass, generate an artist corresponding to each value
        contents = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                artists.append(mark._legend_artist(variables, val, scales))
            contents.append((key, artists, labels))

        self._legend_contents.extend(contents)
```
### 219 - seaborn/_core/plot.py:

Start line: 1507, End line: 1554

```python
class Plotter:

    def _make_legend(self, p: Plot) -> None:
        """Create the legend artist(s) and add onto the figure."""
        # Combine artists representing same information across layers
        # Input list has an entry for each distinct variable in each layer
        # Output dict has an entry for each distinct variable
        merged_contents: dict[
            tuple[str, str | int], tuple[list[Artist], list[str]],
        ] = {}
        for key, artists, labels in self._legend_contents:
            # Key is (name, id); we need the id to resolve variable uniqueness,
            # but will need the name in the next step to title the legend
            if key in merged_contents:
                # Copy so inplace updates don't propagate back to legend_contents
                existing_artists = merged_contents[key][0]
                for i, artist in enumerate(existing_artists):
                    # Matplotlib accepts a tuple of artists and will overlay them
                    if isinstance(artist, tuple):
                        artist += artist[i],
                    else:
                        existing_artists[i] = artist, artists[i]
            else:
                merged_contents[key] = artists.copy(), labels

        # TODO explain
        loc = "center right" if self._pyplot else "center left"

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title=name,
                loc=loc,
                bbox_to_anchor=(.98, .55),
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)
```
### 224 - seaborn/_core/plot.py:

Start line: 624, End line: 645

```python
@build_plot_signature
class Plot:

    def label(self, *, title=None, **variables: str | Callable[[str], str]) -> Plot:
        """
        Add or modify labels for axes, legends, and subplots.

        Additional keywords correspond to variables defined in the plot.
        Values can be one of the following types:

        - string (used literally; pass "" to clear the default label)
        - function (called on the default label)

        For coordinate variables, the value sets the axis label.
        For semantic variables, the value sets the legend title.
        For faceting variables, `title=` modifies the subplot-specific label,
        while `col=` and/or `row=` add a label for the faceting variable.
        When using a single subplot, `title=` sets its title.

        """
        new = self._clone()
        if title is not None:
            new._labels["title"] = title
        new._labels.update(variables)
        return new
```
### 225 - seaborn/_core/plot.py:

Start line: 220, End line: 252

```python
@build_plot_signature
class Plot:

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataSource, dict[str, VariableSpec]]:
        """Handle positional arguments, which may contain data / x / y."""
        if len(args) > 3:
            err = "Plot() accepts no more than 3 positional arguments (data, x, y)."
            raise TypeError(err)

        # TODO need some clearer way to differentiate data / vector here
        # (There might be an abstract DataFrame class to use here?)
        if isinstance(args[0], (abc.Mapping, pd.DataFrame)):
            if data is not None:
                raise TypeError("`data` given by both name and position.")
            data, args = args[0], args[1:]

        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = *args, None
        else:
            x = y = None

        for name, var in zip("yx", (y, x)):
            if var is not None:
                if name in variables:
                    raise TypeError(f"`{name}` given by both name and position.")
                # Keep coordinates at the front of the variables dict
                variables = {name: var, **variables}

        return data, variables
```
### 227 - seaborn/_core/plot.py:

Start line: 780, End line: 822

```python
# ---- The plot compilation engine ---------------------------------------------- #


class Plotter:
    """
    Engine for compiling a :class:`Plot` spec into a Matplotlib figure.

    This class is not intended to be instantiated directly by users.

    """
    # TODO decide if we ever want these (Plot.plot(debug=True))?
    _data: PlotData
    _layers: list[Layer]
    _figure: Figure

    def __init__(self, pyplot: bool, theme: dict[str, Any]):

        self._pyplot = pyplot
        self._theme = theme
        self._legend_contents: list[tuple[
            tuple[str, str | int], list[Artist], list[str],
        ]] = []
        self._scales: dict[str, Scale] = {}

    def save(self, loc, **kwargs) -> Plotter:  # TODO type args
        kwargs.setdefault("dpi", 96)
        try:
            loc = os.path.expanduser(loc)
        except TypeError:
            # loc may be a buffer in which case that would not work
            pass
        self._figure.savefig(loc, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        import matplotlib.pyplot as plt
        with theme_context(self._theme):
            plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?
```
### 230 - seaborn/_core/plot.py:

Start line: 365, End line: 448

```python
@build_plot_signature
class Plot:

    def add(
        self,
        mark: Mark,
        *transforms: Stat | Mark,
        orient: str | None = None,
        legend: bool = True,
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:
        """
        Define a layer of the visualization in terms of mark and data transform(s).

        This is the main method for specifying how the data should be visualized.
        It can be called multiple times with different arguments to define
        a plot with multiple layers.

        Parameters
        ----------
        mark : :class:`seaborn.objects.Mark`
            The visual representation of the data to use in this layer.
        transforms : :class:`seaborn.objects.Stat` or :class:`seaborn.objects.Move`
            Objects representing transforms to be applied before plotting the data.
            Current, at most one :class:`seaborn.objects.Stat` can be used, and it
            must be passed first. This constraint will be relaxed in the future.
        orient : "x", "y", "v", or "h"
            The orientation of the mark, which affects how the stat is computed.
            Typically corresponds to the axis that defines groups for aggregation.
            The "v" (vertical) and "h" (horizontal) options are synonyms for "x" / "y",
            but may be more intuitive with some marks. When not provided, an
            orientation will be inferred from characteristics of the data and scales.
        legend : bool
            Option to suppress the mark/mappings for this layer from the legend.
        data : DataFrame or dict
            Data source to override the global source provided in the constructor.
        variables : data vectors or identifiers
            Additional layer-specific variables, including variables that will be
            passed directly to the transforms without scaling.

        """
        if not isinstance(mark, Mark):
            msg = f"mark must be a Mark instance, not {type(mark)!r}."
            raise TypeError(msg)

        # TODO This API for transforms was a late decision, and previously Plot.add
        # accepted 0 or 1 Stat instances and 0, 1, or a list of Move instances.
        # It will take some work to refactor the internals so that Stat and Move are
        # treated identically, and until then well need to "unpack" the transforms
        # here and enforce limitations on the order / types.

        stat: Optional[Stat]
        move: Optional[List[Move]]
        error = False
        if not transforms:
            stat, move = None, None
        elif isinstance(transforms[0], Stat):
            stat = transforms[0]
            move = [m for m in transforms[1:] if isinstance(m, Move)]
            error = len(move) != len(transforms) - 1
        else:
            stat = None
            move = [m for m in transforms if isinstance(m, Move)]
            error = len(move) != len(transforms)

        if error:
            msg = " ".join([
                "Transforms must have at most one Stat type (in the first position),",
                "and all others must be a Move type. Given transform type(s):",
                ", ".join(str(type(t).__name__) for t in transforms) + "."
            ])
            raise TypeError(msg)

        new = self._clone()
        new._layers.append({
            "mark": mark,
            "stat": stat,
            "move": move,
            # TODO it doesn't work to supply scalars to variables, but it should
            "vars": variables,
            "source": data,
            "legend": legend,
            "orient": {"v": "x", "h": "y"}.get(orient, orient),  # type: ignore
        })

        return new
```
