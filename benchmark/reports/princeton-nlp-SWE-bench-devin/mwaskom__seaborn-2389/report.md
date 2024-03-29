# mwaskom__seaborn-2389

| **mwaskom/seaborn** | `bcdac5411a1b71ff8d4a2fd12a937c129513e79e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 95808 |
| **Any found context length** | 316 |
| **Avg pos** | 411.0 |
| **Min pos** | 1 |
| **Max pos** | 203 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/seaborn/matrix.py b/seaborn/matrix.py
--- a/seaborn/matrix.py
+++ b/seaborn/matrix.py
@@ -38,22 +38,15 @@ def _index_to_ticklabels(index):
 
 def _convert_colors(colors):
     """Convert either a list of colors or nested lists of colors to RGB."""
-    to_rgb = mpl.colors.colorConverter.to_rgb
-
-    if isinstance(colors, pd.DataFrame):
-        # Convert dataframe
-        return pd.DataFrame({col: colors[col].map(to_rgb)
-                            for col in colors})
-    elif isinstance(colors, pd.Series):
-        return colors.map(to_rgb)
-    else:
-        try:
-            to_rgb(colors[0])
-            # If this works, there is only one level of colors
-            return list(map(to_rgb, colors))
-        except ValueError:
-            # If we get here, we have nested lists
-            return [list(map(to_rgb, l)) for l in colors]
+    to_rgb = mpl.colors.to_rgb
+
+    try:
+        to_rgb(colors[0])
+        # If this works, there is only one level of colors
+        return list(map(to_rgb, colors))
+    except ValueError:
+        # If we get here, we have nested lists
+        return [list(map(to_rgb, l)) for l in colors]
 
 
 def _matrix_mask(data, mask):
@@ -93,7 +86,7 @@ def _matrix_mask(data, mask):
     return mask
 
 
-class _HeatMapper(object):
+class _HeatMapper:
     """Draw a heatmap plot of a matrix with nice labels and colormaps."""
 
     def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
@@ -132,9 +125,6 @@ def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
         elif yticklabels is False:
             yticklabels = []
 
-        # Get the positions and used label for the ticks
-        nx, ny = data.T.shape
-
         if not len(xticklabels):
             self.xticks = []
             self.xticklabels = []
@@ -889,9 +879,9 @@ def _preprocess_colors(self, data, colors, axis):
                 else:
                     colors = colors.reindex(data.columns)
 
-                # Replace na's with background color
+                # Replace na's with white color
                 # TODO We should set these to transparent instead
-                colors = colors.fillna('white')
+                colors = colors.astype(object).fillna('white')
 
                 # Extract color values and labels from frame/series
                 if isinstance(colors, pd.DataFrame):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| seaborn/matrix.py | 41 | 56 | 4 | 1 | 906
| seaborn/matrix.py | 96 | 96 | 203 | 1 | 95808
| seaborn/matrix.py | 135 | 137 | 203 | 1 | 95808
| seaborn/matrix.py | 892 | 894 | 1 | 1 | 316


## Problem Statement

```
ValueError: fill value must be in categories
In the  _preprocess_colors function, there is the code to replace na's with background color as the comment said, using `colors = colors.fillna('white')`, however, if the original colors do not contain the 'white' category, this line would raise the Pandas ValueError:fill value must be in categories in `Pandas 0.25.3`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 seaborn/matrix.py** | 869 | 909| 316 | 316 | 11741 | 
| 2 | 2 seaborn/categorical.py | 1 | 28| 196 | 512 | 43088 | 
| 3 | 3 seaborn/_core.py | 176 | 211| 246 | 758 | 53991 | 
| **-> 4 <-** | **3 seaborn/matrix.py** | 39 | 56| 148 | 906 | 53991 | 
| 5 | 3 seaborn/categorical.py | 515 | 546| 273 | 1179 | 53991 | 
| 6 | 4 seaborn/palettes.py | 980 | 1039| 486 | 1665 | 62347 | 
| 7 | 5 seaborn/colors/__init__.py | 1 | 3| 0 | 1665 | 62380 | 
| 8 | 6 examples/palette_choices.py | 1 | 38| 314 | 1979 | 62694 | 
| 9 | 6 seaborn/palettes.py | 1 | 44| 750 | 2729 | 62694 | 
| 10 | 6 seaborn/categorical.py | 267 | 324| 495 | 3224 | 62694 | 
| 11 | 7 seaborn/colors/xkcd_rgb.py | 1 | 950| 9 | 3233 | 73213 | 
| 12 | 7 seaborn/palettes.py | 145 | 226| 675 | 3908 | 73213 | 
| 13 | 7 seaborn/palettes.py | 60 | 77| 138 | 4046 | 73213 | 
| 14 | **7 seaborn/matrix.py** | 1016 | 1065| 367 | 4413 | 73213 | 
| 15 | 7 seaborn/categorical.py | 1892 | 1999| 933 | 5346 | 73213 | 
| 16 | 7 seaborn/_core.py | 213 | 254| 339 | 5685 | 73213 | 
| 17 | 7 seaborn/categorical.py | 2788 | 2820| 243 | 5928 | 73213 | 
| 18 | 7 seaborn/categorical.py | 1547 | 1573| 322 | 6250 | 73213 | 
| 19 | 8 seaborn/utils.py | 1 | 29| 193 | 6443 | 78305 | 
| 20 | 8 seaborn/categorical.py | 2385 | 2406| 178 | 6621 | 78305 | 
| 21 | 8 seaborn/categorical.py | 3727 | 3862| 1225 | 7846 | 78305 | 
| 22 | 8 seaborn/categorical.py | 1049 | 1091| 289 | 8135 | 78305 | 
| 23 | 9 seaborn/widgets.py | 100 | 154| 577 | 8712 | 81809 | 
| 24 | 9 seaborn/palettes.py | 47 | 57| 152 | 8864 | 81809 | 
| 25 | 9 seaborn/_core.py | 140 | 174| 280 | 9144 | 81809 | 
| 26 | **9 seaborn/matrix.py** | 59 | 93| 253 | 9397 | 81809 | 
| 27 | 9 seaborn/palettes.py | 921 | 942| 254 | 9651 | 81809 | 
| 28 | 9 seaborn/categorical.py | 3567 | 3608| 270 | 9921 | 81809 | 
| 29 | 9 seaborn/_core.py | 68 | 138| 469 | 10390 | 81809 | 
| 30 | 10 examples/scatterplot_categorical.py | 1 | 17| 0 | 10390 | 81902 | 
| 31 | 11 seaborn/colors/crayons.py | 1 | 121| 1373 | 11763 | 83275 | 
| 32 | 11 seaborn/widgets.py | 418 | 441| 258 | 12021 | 83275 | 
| 33 | 12 examples/jitter_stripplot.py | 1 | 34| 245 | 12266 | 83520 | 
| 34 | 12 seaborn/categorical.py | 653 | 671| 172 | 12438 | 83520 | 
| 35 | 12 seaborn/widgets.py | 359 | 383| 236 | 12674 | 83520 | 
| 36 | 12 seaborn/palettes.py | 905 | 919| 195 | 12869 | 83520 | 
| 37 | 12 seaborn/categorical.py | 3166 | 3188| 175 | 13044 | 83520 | 
| 38 | 12 seaborn/widgets.py | 1 | 58| 426 | 13470 | 83520 | 
| 39 | 12 seaborn/categorical.py | 2068 | 2223| 1475 | 14945 | 83520 | 
| 40 | 13 seaborn/regression.py | 58 | 69| 119 | 15064 | 92919 | 
| 41 | 14 seaborn/miscplot.py | 1 | 30| 231 | 15295 | 93297 | 
| 42 | 14 seaborn/categorical.py | 3359 | 3381| 185 | 15480 | 93297 | 
| 43 | 15 seaborn/__init__.py | 1 | 22| 236 | 15716 | 93533 | 
| 44 | 15 seaborn/categorical.py | 400 | 415| 127 | 15843 | 93533 | 
| 45 | 16 seaborn/axisgrid.py | 1 | 28| 166 | 16009 | 111856 | 
| 46 | 16 seaborn/_core.py | 340 | 399| 505 | 16514 | 111856 | 
| 47 | 16 seaborn/palettes.py | 738 | 762| 179 | 16693 | 111856 | 
| 48 | 16 seaborn/categorical.py | 1794 | 1839| 400 | 17093 | 111856 | 
| 49 | **16 seaborn/matrix.py** | 544 | 559| 235 | 17328 | 111856 | 
| 50 | 16 seaborn/palettes.py | 765 | 790| 179 | 17507 | 111856 | 
| 51 | 16 seaborn/categorical.py | 2225 | 2249| 166 | 17673 | 111856 | 
| 52 | 16 seaborn/_core.py | 401 | 478| 652 | 18325 | 111856 | 
| 53 | **16 seaborn/matrix.py** | 194 | 249| 520 | 18845 | 111856 | 
| 54 | 16 seaborn/categorical.py | 348 | 397| 446 | 19291 | 111856 | 
| 55 | 16 seaborn/palettes.py | 374 | 467| 665 | 19956 | 111856 | 
| 56 | 17 seaborn/cm.py | 1 | 1561| 70 | 20026 | 146875 | 
| 57 | 17 seaborn/axisgrid.py | 198 | 234| 264 | 20290 | 146875 | 
| 58 | 17 seaborn/_core.py | 726 | 821| 775 | 21065 | 146875 | 
| 59 | 17 seaborn/palettes.py | 79 | 90| 131 | 21196 | 146875 | 
| 60 | 17 seaborn/palettes.py | 712 | 735| 213 | 21409 | 146875 | 
| 61 | **17 seaborn/matrix.py** | 356 | 559| 106 | 21515 | 146875 | 
| 62 | 17 seaborn/categorical.py | 31 | 240| 1388 | 22903 | 146875 | 
| 63 | 17 seaborn/categorical.py | 2001 | 2065| 450 | 23353 | 146875 | 
| 64 | 18 seaborn/relational.py | 56 | 176| 1175 | 24528 | 155753 | 
| 65 | 18 seaborn/cm.py | 1564 | 1585| 166 | 24694 | 155753 | 
| 66 | 18 seaborn/categorical.py | 2409 | 2640| 190 | 24884 | 155753 | 
| 67 | 18 seaborn/relational.py | 1 | 54| 399 | 25283 | 155753 | 
| 68 | 18 seaborn/utils.py | 144 | 168| 187 | 25470 | 155753 | 
| 69 | 18 seaborn/categorical.py | 2985 | 3017| 239 | 25709 | 155753 | 
| 70 | 18 seaborn/categorical.py | 2252 | 2382| 954 | 26663 | 155753 | 
| 71 | **18 seaborn/matrix.py** | 1097 | 1160| 543 | 27206 | 155753 | 
| 72 | 19 seaborn/conftest.py | 33 | 42| 139 | 27345 | 157334 | 
| 73 | 19 seaborn/utils.py | 519 | 530| 156 | 27501 | 157334 | 
| 74 | 19 seaborn/relational.py | 1007 | 1042| 409 | 27910 | 157334 | 
| 75 | 19 seaborn/conftest.py | 1 | 30| 231 | 28141 | 157334 | 
| 76 | **19 seaborn/matrix.py** | 1 | 36| 220 | 28361 | 157334 | 
| 77 | 20 seaborn/distributions.py | 188 | 210| 301 | 28662 | 177518 | 
| 78 | 20 seaborn/axisgrid.py | 757 | 778| 174 | 28836 | 177518 | 
| 79 | 21 examples/palette_generation.py | 1 | 36| 282 | 29118 | 177800 | 
| 80 | 21 seaborn/distributions.py | 1591 | 1757| 1288 | 30406 | 177800 | 
| 81 | 21 seaborn/palettes.py | 93 | 144| 472 | 30878 | 177800 | 
| 82 | 21 seaborn/distributions.py | 335 | 530| 1528 | 32406 | 177800 | 
| 83 | 21 seaborn/categorical.py | 548 | 652| 867 | 33273 | 177800 | 
| 84 | 21 seaborn/categorical.py | 759 | 935| 1222 | 34495 | 177800 | 
| 85 | 21 seaborn/distributions.py | 162 | 186| 271 | 34766 | 177800 | 
| 86 | 21 seaborn/categorical.py | 1876 | 1883| 121 | 34887 | 177800 | 
| 87 | 22 seaborn/external/husl.py | 156 | 175| 171 | 35058 | 180470 | 
| 88 | 22 seaborn/utils.py | 651 | 676| 205 | 35263 | 180470 | 
| 89 | 22 seaborn/categorical.py | 1094 | 1112| 195 | 35458 | 180470 | 
| 90 | 22 seaborn/categorical.py | 1885 | 1890| 116 | 35574 | 180470 | 
| 91 | 23 seaborn/rcmod.py | 516 | 557| 340 | 35914 | 184347 | 
| 92 | 23 seaborn/axisgrid.py | 237 | 293| 503 | 36417 | 184347 | 
| 93 | 23 seaborn/categorical.py | 1427 | 1545| 772 | 37189 | 184347 | 
| 94 | 23 seaborn/palettes.py | 551 | 629| 643 | 37832 | 184347 | 
| 95 | **23 seaborn/matrix.py** | 1162 | 1217| 560 | 38392 | 184347 | 
| 96 | 24 seaborn/_docstrings.py | 51 | 123| 611 | 39003 | 185673 | 
| 97 | 24 seaborn/utils.py | 171 | 215| 262 | 39265 | 185673 | 
| 98 | 24 seaborn/categorical.py | 2823 | 2982| 1195 | 40460 | 185673 | 
| 99 | 24 seaborn/axisgrid.py | 476 | 566| 905 | 41365 | 185673 | 
| 100 | 25 seaborn/_decorators.py | 30 | 47| 194 | 41559 | 186144 | 
| 101 | 25 seaborn/axisgrid.py | 1995 | 2014| 255 | 41814 | 186144 | 
| 102 | 25 seaborn/categorical.py | 3611 | 3702| 608 | 42422 | 186144 | 
| 103 | 25 seaborn/palettes.py | 470 | 548| 639 | 43061 | 186144 | 
| 104 | 25 seaborn/categorical.py | 2643 | 2785| 1182 | 44243 | 186144 | 
| 105 | 25 seaborn/categorical.py | 3191 | 3356| 1198 | 45441 | 186144 | 
| 106 | 25 seaborn/categorical.py | 326 | 346| 157 | 45598 | 186144 | 
| 107 | 25 seaborn/categorical.py | 3384 | 3564| 1320 | 46918 | 186144 | 
| 108 | 25 seaborn/relational.py | 786 | 822| 292 | 47210 | 186144 | 
| 109 | 25 seaborn/palettes.py | 793 | 904| 865 | 48075 | 186144 | 
| 110 | 25 seaborn/relational.py | 670 | 704| 286 | 48361 | 186144 | 
| 111 | 25 seaborn/palettes.py | 300 | 371| 483 | 48844 | 186144 | 
| 112 | 26 examples/errorband_lineplots.py | 1 | 18| 0 | 48844 | 186233 | 
| 113 | 26 seaborn/widgets.py | 242 | 324| 655 | 49499 | 186233 | 
| 114 | 26 seaborn/conftest.py | 123 | 165| 311 | 49810 | 186233 | 
| 115 | 26 seaborn/_core.py | 481 | 554| 534 | 50344 | 186233 | 
| 116 | 26 seaborn/categorical.py | 1699 | 1791| 745 | 51089 | 186233 | 
| 117 | 26 seaborn/categorical.py | 417 | 477| 453 | 51542 | 186233 | 
| 118 | 27 examples/part_whole_bars.py | 1 | 31| 216 | 51758 | 186449 | 
| 119 | 27 seaborn/palettes.py | 229 | 297| 457 | 52215 | 186449 | 
| 120 | 27 seaborn/conftest.py | 168 | 191| 239 | 52454 | 186449 | 
| 121 | 27 seaborn/_core.py | 257 | 338| 490 | 52944 | 186449 | 
| 122 | 27 seaborn/distributions.py | 1131 | 1217| 771 | 53715 | 186449 | 
| 123 | 27 seaborn/categorical.py | 999 | 1017| 174 | 53889 | 186449 | 
| 124 | 27 seaborn/categorical.py | 951 | 983| 328 | 54217 | 186449 | 
| 125 | 28 seaborn/_statistics.py | 1 | 33| 309 | 54526 | 189744 | 
| 126 | **28 seaborn/matrix.py** | 367 | 543| 1588 | 56114 | 189744 | 
| 127 | 28 seaborn/widgets.py | 157 | 239| 655 | 56769 | 189744 | 
| 128 | **28 seaborn/matrix.py** | 931 | 959| 190 | 56959 | 189744 | 
| 129 | 28 seaborn/categorical.py | 242 | 265| 169 | 57128 | 189744 | 
| 130 | 28 seaborn/_core.py | 683 | 724| 382 | 57510 | 189744 | 
| 131 | 28 seaborn/palettes.py | 632 | 709| 679 | 58189 | 189744 | 
| 132 | 29 examples/wide_data_lineplot.py | 1 | 20| 137 | 58326 | 189881 | 
| 133 | 29 seaborn/conftest.py | 45 | 100| 296 | 58622 | 189881 | 
| 134 | 29 seaborn/axisgrid.py | 1926 | 1993| 731 | 59353 | 189881 | 
| 135 | 29 seaborn/categorical.py | 479 | 512| 281 | 59634 | 189881 | 
| 136 | 29 seaborn/categorical.py | 1595 | 1642| 357 | 59991 | 189881 | 
| 137 | 30 seaborn/_testing.py | 1 | 38| 150 | 60141 | 190455 | 
| 138 | 30 seaborn/categorical.py | 694 | 724| 225 | 60366 | 190455 | 
| 139 | 30 seaborn/categorical.py | 1019 | 1046| 295 | 60661 | 190455 | 
| 140 | 30 seaborn/utils.py | 600 | 619| 159 | 60820 | 190455 | 
| 141 | 31 examples/scatterplot_sizes.py | 1 | 25| 164 | 60984 | 190619 | 
| 142 | 31 seaborn/relational.py | 898 | 1005| 829 | 61813 | 190619 | 
| 143 | 31 seaborn/categorical.py | 1334 | 1424| 775 | 62588 | 190619 | 
| 144 | **31 seaborn/matrix.py** | 1412 | 1423| 262 | 62850 | 190619 | 
| 145 | 31 seaborn/distributions.py | 877 | 1033| 1150 | 64000 | 190619 | 
| 146 | 31 seaborn/regression.py | 556 | 636| 825 | 64825 | 190619 | 
| 147 | 31 seaborn/distributions.py | 1035 | 1130| 693 | 65518 | 190619 | 
| 148 | 31 seaborn/distributions.py | 531 | 715| 1434 | 66952 | 190619 | 
| 149 | 31 seaborn/categorical.py | 3865 | 4022| 1397 | 68349 | 190619 | 
| 150 | 32 setup.py | 1 | 86| 679 | 69028 | 191314 | 
| 151 | 32 seaborn/categorical.py | 3705 | 3724| 172 | 69200 | 191314 | 
| 152 | 32 seaborn/categorical.py | 3020 | 3163| 1094 | 70294 | 191314 | 
| 153 | 32 seaborn/categorical.py | 726 | 757| 241 | 70535 | 191314 | 
| 154 | 32 seaborn/external/husl.py | 117 | 153| 266 | 70801 | 191314 | 
| 155 | **32 seaborn/matrix.py** | 998 | 1014| 128 | 70929 | 191314 | 
| 156 | 32 seaborn/categorical.py | 985 | 997| 188 | 71117 | 191314 | 
| 157 | 32 seaborn/distributions.py | 717 | 875| 1287 | 72404 | 191314 | 
| 158 | 32 seaborn/regression.py | 639 | 809| 1376 | 73780 | 191314 | 
| 159 | 32 seaborn/widgets.py | 327 | 357| 227 | 74007 | 191314 | 
| 160 | 32 seaborn/axisgrid.py | 2017 | 2179| 1447 | 75454 | 191314 | 
| 161 | 32 seaborn/axisgrid.py | 1538 | 1553| 142 | 75596 | 191314 | 
| 162 | 32 seaborn/relational.py | 471 | 563| 811 | 76407 | 191314 | 
| 163 | 32 seaborn/axisgrid.py | 296 | 474| 1478 | 77885 | 191314 | 
| 164 | 32 seaborn/_core.py | 1018 | 1058| 346 | 78231 | 191314 | 
| 165 | **32 seaborn/matrix.py** | 961 | 996| 226 | 78457 | 191314 | 
| 166 | 33 examples/regression_marginals.py | 1 | 15| 0 | 78457 | 191405 | 
| 167 | 33 seaborn/categorical.py | 1278 | 1298| 198 | 78655 | 191405 | 
| 168 | 33 seaborn/widgets.py | 386 | 416| 230 | 78885 | 191405 | 
| 169 | **33 seaborn/matrix.py** | 251 | 265| 197 | 79082 | 191405 | 
| 170 | 33 seaborn/distributions.py | 2018 | 2085| 539 | 79621 | 191405 | 
| 171 | 33 seaborn/relational.py | 179 | 286| 919 | 80540 | 191405 | 
| 172 | 33 seaborn/axisgrid.py | 780 | 835| 459 | 80999 | 191405 | 
| 173 | 33 seaborn/categorical.py | 1114 | 1164| 463 | 81462 | 191405 | 
| 174 | 33 seaborn/_core.py | 1 | 65| 459 | 81921 | 191405 | 
| 175 | 33 seaborn/utils.py | 92 | 141| 274 | 82195 | 191405 | 
| 176 | 33 seaborn/rcmod.py | 1 | 83| 418 | 82613 | 191405 | 
| 177 | 33 seaborn/_core.py | 823 | 935| 876 | 83489 | 191405 | 
| 178 | 34 doc/tools/generate_logos.py | 61 | 156| 938 | 84427 | 193330 | 
| 179 | 34 seaborn/distributions.py | 134 | 160| 229 | 84656 | 193330 | 
| 180 | 34 seaborn/conftest.py | 194 | 236| 214 | 84870 | 193330 | 
| 181 | 35 examples/spreadsheet_heatmap.py | 1 | 17| 110 | 84980 | 193440 | 
| 182 | 35 seaborn/_core.py | 1276 | 1348| 586 | 85566 | 193440 | 
| 183 | 36 examples/scatterplot_matrix.py | 1 | 12| 0 | 85566 | 193488 | 
| 184 | 36 seaborn/widgets.py | 61 | 98| 304 | 85870 | 193488 | 
| 185 | 36 seaborn/categorical.py | 1645 | 1697| 458 | 86328 | 193488 | 
| 186 | 37 examples/scatter_bubbles.py | 1 | 18| 111 | 86439 | 193599 | 
| 187 | 37 seaborn/distributions.py | 1 | 88| 555 | 86994 | 193599 | 
| 188 | 37 seaborn/distributions.py | 2543 | 2634| 826 | 87820 | 193599 | 
| 189 | 38 examples/faceted_histogram.py | 1 | 15| 0 | 87820 | 193684 | 
| 190 | **38 seaborn/matrix.py** | 1219 | 1245| 274 | 88094 | 193684 | 
| 191 | 38 seaborn/axisgrid.py | 1084 | 1248| 1451 | 89545 | 193684 | 
| 192 | 38 seaborn/utils.py | 371 | 394| 167 | 89712 | 193684 | 
| 193 | 38 seaborn/relational.py | 592 | 667| 747 | 90459 | 193684 | 
| 194 | 39 examples/pair_grid_with_kde.py | 1 | 16| 0 | 90459 | 193773 | 
| 195 | 39 seaborn/regression.py | 428 | 555| 1382 | 91841 | 193773 | 
| 196 | 39 seaborn/axisgrid.py | 608 | 689| 653 | 92494 | 193773 | 
| 197 | 39 seaborn/regression.py | 812 | 839| 290 | 92784 | 193773 | 
| 198 | 39 seaborn/categorical.py | 1576 | 1593| 163 | 92947 | 193773 | 
| 199 | 39 seaborn/utils.py | 250 | 317| 645 | 93592 | 193773 | 
| 200 | 39 doc/tools/generate_logos.py | 159 | 225| 617 | 94209 | 193773 | 
| 201 | 39 seaborn/rcmod.py | 174 | 298| 796 | 95005 | 193773 | 
| 202 | 40 examples/three_variable_histogram.py | 1 | 16| 0 | 95005 | 193861 | 
| **-> 203 <-** | **40 seaborn/matrix.py** | 96 | 192| 803 | 95808 | 193861 | 
| 204 | 40 seaborn/categorical.py | 1167 | 1192| 210 | 96018 | 193861 | 
| 205 | 40 seaborn/utils.py | 679 | 695| 179 | 96197 | 193861 | 
| 206 | 40 seaborn/rcmod.py | 487 | 513| 192 | 96389 | 193861 | 
| 207 | 40 seaborn/_decorators.py | 1 | 28| 183 | 96572 | 193861 | 
| 208 | 41 examples/structured_heatmap.py | 1 | 37| 308 | 96880 | 194169 | 
| 209 | 41 seaborn/regression.py | 1 | 24| 139 | 97019 | 194169 | 
| 210 | 41 seaborn/conftest.py | 103 | 120| 148 | 97167 | 194169 | 
| 211 | 41 seaborn/distributions.py | 212 | 268| 482 | 97649 | 194169 | 
| 212 | 42 examples/heat_scatter.py | 1 | 42| 330 | 97979 | 194499 | 
| 213 | 42 seaborn/external/husl.py | 71 | 91| 287 | 98266 | 194499 | 
| 214 | 42 seaborn/relational.py | 288 | 345| 420 | 98686 | 194499 | 
| 215 | 42 seaborn/distributions.py | 1283 | 1324| 374 | 99060 | 194499 | 
| 216 | 43 seaborn/algorithms.py | 1 | 86| 646 | 99706 | 195525 | 
| 217 | 43 seaborn/categorical.py | 1841 | 1874| 435 | 100141 | 195525 | 
| 218 | 43 seaborn/utils.py | 397 | 424| 190 | 100331 | 195525 | 
| 219 | 43 seaborn/distributions.py | 270 | 333| 394 | 100725 | 195525 | 
| 220 | 43 seaborn/axisgrid.py | 1052 | 1067| 157 | 100882 | 195525 | 
| 221 | 44 examples/simple_violinplots.py | 1 | 19| 121 | 101003 | 195646 | 
| 222 | 44 seaborn/_core.py | 556 | 577| 189 | 101192 | 195646 | 
| 223 | 44 seaborn/relational.py | 348 | 377| 238 | 101430 | 195646 | 
| 224 | 44 seaborn/relational.py | 566 | 590| 194 | 101624 | 195646 | 
| 225 | 45 examples/hexbin_marginals.py | 1 | 16| 0 | 101624 | 195738 | 
| 226 | 45 seaborn/axisgrid.py | 1011 | 1033| 180 | 101804 | 195738 | 
| 227 | 46 examples/paired_pointplots.py | 1 | 21| 153 | 101957 | 195891 | 
| 228 | 46 seaborn/axisgrid.py | 1321 | 1401| 627 | 102584 | 195891 | 
| 229 | 46 seaborn/categorical.py | 937 | 949| 133 | 102717 | 195891 | 
| 230 | 46 seaborn/regression.py | 842 | 1008| 1276 | 103993 | 195891 | 
| 231 | 46 seaborn/axisgrid.py | 691 | 755| 515 | 104508 | 195891 | 
| 232 | 46 seaborn/categorical.py | 1300 | 1332| 341 | 104849 | 195891 | 
| 233 | 46 seaborn/utils.py | 32 | 59| 191 | 105040 | 195891 | 
| 234 | 47 ci/check_gallery.py | 1 | 15| 0 | 105040 | 195970 | 
| 235 | 47 seaborn/utils.py | 698 | 711| 147 | 105187 | 195970 | 
| 236 | 48 examples/wide_form_violinplot.py | 1 | 35| 291 | 105478 | 196261 | 
| 237 | 48 seaborn/categorical.py | 1211 | 1243| 256 | 105734 | 196261 | 
| 238 | 48 seaborn/categorical.py | 1194 | 1209| 181 | 105915 | 196261 | 
| 239 | **48 seaborn/matrix.py** | 1248 | 1411| 1458 | 107373 | 196261 | 
| 240 | 48 seaborn/axisgrid.py | 993 | 1009| 161 | 107534 | 196261 | 
| 241 | 48 seaborn/categorical.py | 673 | 692| 212 | 107746 | 196261 | 
| 242 | 48 seaborn/_core.py | 1180 | 1191| 187 | 107933 | 196261 | 
| 243 | 48 seaborn/axisgrid.py | 837 | 863| 257 | 108190 | 196261 | 
| 244 | 48 seaborn/axisgrid.py | 178 | 196| 165 | 108355 | 196261 | 
| 245 | 48 seaborn/relational.py | 424 | 470| 472 | 108827 | 196261 | 
| 246 | 49 examples/pairgrid_dotplot.py | 1 | 39| 268 | 109095 | 196529 | 
| 247 | 49 seaborn/distributions.py | 1219 | 1281| 533 | 109628 | 196529 | 
| 248 | 49 seaborn/axisgrid.py | 1403 | 1438| 252 | 109880 | 196529 | 
| 249 | 49 seaborn/_testing.py | 41 | 61| 189 | 110069 | 196529 | 
| 250 | 49 seaborn/_docstrings.py | 126 | 188| 393 | 110462 | 196529 | 


### Hint

```
Can you please share a reproducible example that demonstrates the issue? I can't really figure out what you're talking about from this description.
Here's a self-contained example, using ``clustermap()``. This has to do with colors input for row/col colors that are pandas ``category`` dtype:
\`\`\`python
import seaborn as sns; sns.set(color_codes=True)
iris = sns.load_dataset("iris")
species = iris.pop("species")
row_colors=species.map(dict(zip(species.unique(), "rbg")))
row_colors=row_colors.astype('category')
g = sns.clustermap(iris, row_colors=row_colors)
\`\`\`

This raises the following error:
\`\`\`
ValueError: fill value must be in categories
\`\`\`
Thanks @MaozGelbart. It would still be helpful to understand the real-world case where the color annotations need to be categorical.
Same issue here
```

## Patch

```diff
diff --git a/seaborn/matrix.py b/seaborn/matrix.py
--- a/seaborn/matrix.py
+++ b/seaborn/matrix.py
@@ -38,22 +38,15 @@ def _index_to_ticklabels(index):
 
 def _convert_colors(colors):
     """Convert either a list of colors or nested lists of colors to RGB."""
-    to_rgb = mpl.colors.colorConverter.to_rgb
-
-    if isinstance(colors, pd.DataFrame):
-        # Convert dataframe
-        return pd.DataFrame({col: colors[col].map(to_rgb)
-                            for col in colors})
-    elif isinstance(colors, pd.Series):
-        return colors.map(to_rgb)
-    else:
-        try:
-            to_rgb(colors[0])
-            # If this works, there is only one level of colors
-            return list(map(to_rgb, colors))
-        except ValueError:
-            # If we get here, we have nested lists
-            return [list(map(to_rgb, l)) for l in colors]
+    to_rgb = mpl.colors.to_rgb
+
+    try:
+        to_rgb(colors[0])
+        # If this works, there is only one level of colors
+        return list(map(to_rgb, colors))
+    except ValueError:
+        # If we get here, we have nested lists
+        return [list(map(to_rgb, l)) for l in colors]
 
 
 def _matrix_mask(data, mask):
@@ -93,7 +86,7 @@ def _matrix_mask(data, mask):
     return mask
 
 
-class _HeatMapper(object):
+class _HeatMapper:
     """Draw a heatmap plot of a matrix with nice labels and colormaps."""
 
     def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
@@ -132,9 +125,6 @@ def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
         elif yticklabels is False:
             yticklabels = []
 
-        # Get the positions and used label for the ticks
-        nx, ny = data.T.shape
-
         if not len(xticklabels):
             self.xticks = []
             self.xticklabels = []
@@ -889,9 +879,9 @@ def _preprocess_colors(self, data, colors, axis):
                 else:
                     colors = colors.reindex(data.columns)
 
-                # Replace na's with background color
+                # Replace na's with white color
                 # TODO We should set these to transparent instead
-                colors = colors.fillna('white')
+                colors = colors.astype(object).fillna('white')
 
                 # Extract color values and labels from frame/series
                 if isinstance(colors, pd.DataFrame):

```

## Test Patch

```diff
diff --git a/seaborn/tests/test_matrix.py b/seaborn/tests/test_matrix.py
--- a/seaborn/tests/test_matrix.py
+++ b/seaborn/tests/test_matrix.py
@@ -780,6 +780,26 @@ def test_colors_input(self):
 
         assert len(cg.fig.axes) == 6
 
+    def test_categorical_colors_input(self):
+        kws = self.default_kws.copy()
+
+        row_colors = pd.Series(self.row_colors, dtype="category")
+        col_colors = pd.Series(
+            self.col_colors, dtype="category", index=self.df_norm.columns
+        )
+
+        kws['row_colors'] = row_colors
+        kws['col_colors'] = col_colors
+
+        exp_row_colors = list(map(mpl.colors.to_rgb, row_colors))
+        exp_col_colors = list(map(mpl.colors.to_rgb, col_colors))
+
+        cg = mat.ClusterGrid(self.df_norm, **kws)
+        npt.assert_array_equal(cg.row_colors, exp_row_colors)
+        npt.assert_array_equal(cg.col_colors, exp_col_colors)
+
+        assert len(cg.fig.axes) == 6
+
     def test_nested_colors_input(self):
         kws = self.default_kws.copy()
 

```


## Code snippets

### 1 - seaborn/matrix.py:

Start line: 869, End line: 909

```python
class ClusterGrid(Grid):

    def _preprocess_colors(self, data, colors, axis):
        """Preprocess {row/col}_colors to extract labels and convert colors."""
        labels = None

        if colors is not None:
            if isinstance(colors, (pd.DataFrame, pd.Series)):

                # If data is unindexed, raise
                if (not hasattr(data, "index") and axis == 0) or (
                    not hasattr(data, "columns") and axis == 1
                ):
                    axis_name = "col" if axis else "row"
                    msg = (f"{axis_name}_colors indices can't be matched with data "
                           f"indices. Provide {axis_name}_colors as a non-indexed "
                           "datatype, e.g. by using `.to_numpy()``")
                    raise TypeError(msg)

                # Ensure colors match data indices
                if axis == 0:
                    colors = colors.reindex(data.index)
                else:
                    colors = colors.reindex(data.columns)

                # Replace na's with background color
                # TODO We should set these to transparent instead
                colors = colors.fillna('white')

                # Extract color values and labels from frame/series
                if isinstance(colors, pd.DataFrame):
                    labels = list(colors.columns)
                    colors = colors.T.values
                else:
                    if colors.name is None:
                        labels = [""]
                    else:
                        labels = [colors.name]
                    colors = colors.values

            colors = _convert_colors(colors)

        return colors, labels
```
### 2 - seaborn/categorical.py:

Start line: 1, End line: 28

```python
from textwrap import dedent
from numbers import Number
import colorsys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings
from distutils.version import LooseVersion

from ._core import variable_type, infer_orient, categorical_order
from . import utils
from .utils import remove_na
from .algorithms import bootstrap
from .palettes import color_palette, husl_palette, light_palette, dark_palette
from .axisgrid import FacetGrid, _facet_docs
from ._decorators import _deprecate_positional_args


__all__ = [
    "catplot", "factorplot",
    "stripplot", "swarmplot",
    "boxplot", "violinplot", "boxenplot",
    "pointplot", "barplot", "countplot",
]
```
### 3 - seaborn/_core.py:

Start line: 176, End line: 211

```python
@share_init_params_with_map
class HueMapping(SemanticMapping):

    def categorical_mapping(self, data, palette, order):
        """Determine colors when the hue mapping is categorical."""
        # -- Identify the order and name of the levels

        levels = categorical_order(data, order)
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            lookup_table = palette

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table
```
### 4 - seaborn/matrix.py:

Start line: 39, End line: 56

```python
def _convert_colors(colors):
    """Convert either a list of colors or nested lists of colors to RGB."""
    to_rgb = mpl.colors.colorConverter.to_rgb

    if isinstance(colors, pd.DataFrame):
        # Convert dataframe
        return pd.DataFrame({col: colors[col].map(to_rgb)
                            for col in colors})
    elif isinstance(colors, pd.Series):
        return colors.map(to_rgb)
    else:
        try:
            to_rgb(colors[0])
            # If this works, there is only one level of colors
            return list(map(to_rgb, colors))
        except ValueError:
            # If we get here, we have nested lists
            return [list(map(to_rgb, l)) for l in colors]
```
### 5 - seaborn/categorical.py:

Start line: 515, End line: 546

```python
class _ViolinPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 bw, cut, scale, scale_hue, gridsize,
                 width, inner, split, dodge, orient, linewidth,
                 color, palette, saturation):

        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)
        self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge

        if inner is not None:
            if not any([inner.startswith("quart"),
                        inner.startswith("box"),
                        inner.startswith("stick"),
                        inner.startswith("point")]):
                err = "Inner style '{}' not recognized".format(inner)
                raise ValueError(err)
        self.inner = inner

        if split and self.hue_names is not None and len(self.hue_names) != 2:
            msg = "There must be exactly two hue levels to use `split`.'"
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth
```
### 6 - seaborn/palettes.py:

Start line: 980, End line: 1039

```python
def set_color_codes(palette="deep"):
    """Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : {deep, muted, pastel, dark, bright, colorblind}
        Named seaborn palette to use as the source of colors.

    See Also
    --------
    set : Color codes can be set through the high-level seaborn style
          manager.
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.

    Examples
    --------

    Map matplotlib color codes to the default seaborn palette.

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns; sns.set_theme()
        >>> sns.set_color_codes()
        >>> _ = plt.plot([0, 1], color="r")

    Use a different seaborn palette.

    .. plot::
        :context: close-figs

        >>> sns.set_color_codes("dark")
        >>> _ = plt.plot([0, 1], color="g")
        >>> _ = plt.plot([0, 2], color="m")

    """
    if palette == "reset":
        colors = [(0., 0., 1.), (0., .5, 0.), (1., 0., 0.), (.75, 0., .75),
                  (.75, .75, 0.), (0., .75, .75), (0., 0., 0.)]
    elif not isinstance(palette, str):
        err = "set_color_codes requires a named seaborn palette"
        raise TypeError(err)
    elif palette in SEABORN_PALETTES:
        if not palette.endswith("6"):
            palette = palette + "6"
        colors = SEABORN_PALETTES[palette] + [(.1, .1, .1)]
    else:
        err = "Cannot set colors with palette '{}'".format(palette)
        raise ValueError(err)

    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb
```
### 7 - seaborn/colors/__init__.py:

Start line: 1, End line: 3

```python

```
### 8 - examples/palette_choices.py:

Start line: 1, End line: 38

```python
"""
Color palette choices
=====================

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", context="talk")
rs = np.random.RandomState(8)

# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

# Generate some sequential data
x = np.array(list("ABCDEFGHIJ"))
y1 = np.arange(1, 11)
sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Sequential")

# Center the data to make it diverging
y2 = y1 - 5.5
sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("Diverging")

# Randomly reorder the data to make it qualitative
y3 = rs.choice(y1, len(y1), replace=False)
sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("Qualitative")

# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=2)
```
### 9 - seaborn/palettes.py:

Start line: 1, End line: 44

```python
import colorsys
from itertools import cycle

import numpy as np
import matplotlib as mpl

from .external import husl

from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons


__all__ = ["color_palette", "hls_palette", "husl_palette", "mpl_palette",
           "dark_palette", "light_palette", "diverging_palette",
           "blend_palette", "xkcd_palette", "crayon_palette",
           "cubehelix_palette", "set_color_codes"]


SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    deep6=["#4C72B0", "#55A868", "#C44E52",
           "#8172B3", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
           "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    muted6=["#4878D0", "#6ACC64", "#D65F5F",
            "#956CB4", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
            "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    pastel6=["#A1C9F4", "#8DE5A1", "#FF9F9B",
             "#D0BBFF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    bright6=["#023EFF", "#1AC938", "#E8000B",
             "#8B2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
          "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    dark6=["#001C7F", "#12711C", "#8C0800",
           "#591E71", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
    colorblind6=["#0173B2", "#029E73", "#D55E00",
                 "#CC78BC", "#ECE133", "#56B4E9"]
)
```
### 10 - seaborn/categorical.py:

Start line: 267, End line: 324

```python
class _CategoricalPlotter(object):

    def establish_colors(self, color, palette, saturation):
        """Get a list of colors for the main component of the plots."""
        if self.hue_names is None:
            n_colors = len(self.plot_data)
        else:
            n_colors = len(self.hue_names)

        # Determine the main colors
        if color is None and palette is None:
            # Determine whether the current palette will have enough values
            # If not, we'll default to the husl palette so each is distinct
            current_palette = utils.get_color_cycle()
            if n_colors <= len(current_palette):
                colors = color_palette(n_colors=n_colors)
            else:
                colors = husl_palette(n_colors, l=.7)  # noqa

        elif palette is None:
            # When passing a specific color, the interpretation depends
            # on whether there is a hue variable or not.
            # If so, we will make a blend palette so that the different
            # levels have some amount of variation.
            if self.hue_names is None:
                colors = [color] * n_colors
            else:
                if self.default_palette == "light":
                    colors = light_palette(color, n_colors)
                elif self.default_palette == "dark":
                    colors = dark_palette(color, n_colors)
                else:
                    raise RuntimeError("No default palette specified")
        else:

            # Let `palette` be a dict mapping level to color
            if isinstance(palette, dict):
                if self.hue_names is None:
                    levels = self.group_names
                else:
                    levels = self.hue_names
                palette = [palette[l] for l in levels]

            colors = color_palette(palette, n_colors)

        # Desaturate a bit because these are patches
        if saturation < 1:
            colors = color_palette(colors, desat=saturation)

        # Convert the colors to a common representations
        rgb_colors = color_palette(colors)

        # Determine the gray color to use for the lines framing the plot
        light_vals = [colorsys.rgb_to_hls(*c)[1] for c in rgb_colors]
        lum = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((lum, lum, lum))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray
```
### 14 - seaborn/matrix.py:

Start line: 1016, End line: 1065

```python
class ClusterGrid(Grid):

    @staticmethod
    def color_list_to_matrix_and_cmap(colors, ind, axis=0):
        """Turns a list of colors into a numpy matrix and matplotlib colormap

        These arguments can now be plotted using heatmap(matrix, cmap)
        and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        axis : int
            Which axis this is labeling

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each corresponds to a color
            from the originally provided list of colors
        cmap : matplotlib.colors.ListedColormap

        """
        # check for nested lists/color palettes.
        # Will fail if matplotlib color is list not tuple
        if any(issubclass(type(x), list) for x in colors):
            all_colors = set(itertools.chain(*colors))
            n = len(colors)
            m = len(colors[0])
        else:
            all_colors = set(colors)
            n = 1
            m = len(colors)
            colors = [colors]
        color_to_value = dict((col, i) for i, col in enumerate(all_colors))

        matrix = np.array([color_to_value[c]
                           for color in colors for c in color])

        shape = (n, m)
        matrix = matrix.reshape(shape)
        matrix = matrix[:, ind]
        if axis == 0:
            # row-side:
            matrix = matrix.T

        cmap = mpl.colors.ListedColormap(all_colors)
        return matrix, cmap
```
### 26 - seaborn/matrix.py:

Start line: 59, End line: 93

```python
def _matrix_mask(data, mask):
    """Ensure that data and mask are compatible and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask
```
### 49 - seaborn/matrix.py:

Start line: 544, End line: 559

```python
@_deprecate_positional_args
def heatmap(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    **kwargs
):
    # Initialize the plotter object
    plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws, xticklabels,
                          yticklabels, mask)

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
```
### 53 - seaborn/matrix.py:

Start line: 194, End line: 249

```python
class _HeatMapper(object):

    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""

        # plot_data is a np.ma.array instance
        calc_data = plot_data.astype(float).filled(np.nan)
        if vmin is None:
            if robust:
                vmin = np.nanpercentile(calc_data, 2)
            else:
                vmin = np.nanmin(calc_data)
        if vmax is None:
            if robust:
                vmax = np.nanpercentile(calc_data, 98)
            else:
                vmax = np.nanmax(calc_data)
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            self.cmap = mpl.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            self.cmap = mpl.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        # Recenter a divergent colormap
        if center is not None:

            # Copy bad values
            # in mpl<3.2 only masked values are honored with "bad" color spec
            # (see https://github.com/matplotlib/matplotlib/pull/14257)
            bad = self.cmap(np.ma.masked_invalid([np.nan]))[0]

            # under/over values are set for sure when cmap extremes
            # do not map to the same color as +-inf
            under = self.cmap(-np.inf)
            over = self.cmap(np.inf)
            under_set = under != self.cmap(0)
            over_set = over != self.cmap(self.cmap.N - 1)

            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = mpl.colors.ListedColormap(self.cmap(cc))
            self.cmap.set_bad(bad)
            if under_set:
                self.cmap.set_under(under)
            if over_set:
                self.cmap.set_over(over)
```
### 61 - seaborn/matrix.py:

Start line: 356, End line: 559

```python
@_deprecate_positional_args
def heatmap(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    **kwargs
):
    # ... other code
```
### 71 - seaborn/matrix.py:

Start line: 1097, End line: 1160

```python
class ClusterGrid(Grid):

    def plot_colors(self, xind, yind, **kws):
        """Plots color labels between the dendrogram and the heatmap

        Parameters
        ----------
        heatmap_kws : dict
            Keyword arguments heatmap

        """
        # Remove any custom colormap and centering
        # TODO this code has consistently caused problems when we
        # have missed kwargs that need to be excluded that it might
        # be better to rewrite *in*clusively.
        kws = kws.copy()
        kws.pop('cmap', None)
        kws.pop('norm', None)
        kws.pop('center', None)
        kws.pop('annot', None)
        kws.pop('vmin', None)
        kws.pop('vmax', None)
        kws.pop('robust', None)
        kws.pop('xticklabels', None)
        kws.pop('yticklabels', None)

        # Plot the row colors
        if self.row_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.row_colors, yind, axis=0)

            # Get row_color labels
            if self.row_color_labels is not None:
                row_color_labels = self.row_color_labels
            else:
                row_color_labels = False

            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_row_colors,
                    xticklabels=row_color_labels, yticklabels=False, **kws)

            # Adjust rotation of labels
            if row_color_labels is not False:
                plt.setp(self.ax_row_colors.get_xticklabels(), rotation=90)
        else:
            despine(self.ax_row_colors, left=True, bottom=True)

        # Plot the column colors
        if self.col_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.col_colors, xind, axis=1)

            # Get col_color labels
            if self.col_color_labels is not None:
                col_color_labels = self.col_color_labels
            else:
                col_color_labels = False

            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_col_colors,
                    xticklabels=False, yticklabels=col_color_labels, **kws)

            # Adjust rotation of labels, place on right side
            if col_color_labels is not False:
                self.ax_col_colors.yaxis.tick_right()
                plt.setp(self.ax_col_colors.get_yticklabels(), rotation=0)
        else:
            despine(self.ax_col_colors, left=True, bottom=True)
```
### 76 - seaborn/matrix.py:

Start line: 1, End line: 36

```python
"""Functions to visualize matrices of data."""
import itertools
import warnings

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from . import cm
from .axisgrid import Grid
from .utils import (despine, axis_ticklabels_overlap, relative_luminance,
                    to_utf8)
from ._decorators import _deprecate_positional_args


__all__ = ["heatmap", "clustermap"]


def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name


def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values
```
### 95 - seaborn/matrix.py:

Start line: 1162, End line: 1217

```python
class ClusterGrid(Grid):

    def plot_matrix(self, colorbar_kws, xind, yind, **kws):
        self.data2d = self.data2d.iloc[yind, xind]
        self.mask = self.mask.iloc[yind, xind]

        # Try to reorganize specified tick labels, if provided
        xtl = kws.pop("xticklabels", "auto")
        try:
            xtl = np.asarray(xtl)[xind]
        except (TypeError, IndexError):
            pass
        ytl = kws.pop("yticklabels", "auto")
        try:
            ytl = np.asarray(ytl)[yind]
        except (TypeError, IndexError):
            pass

        # Reorganize the annotations to match the heatmap
        annot = kws.pop("annot", None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
                annot_data = annot_data[yind][:, xind]
            annot = annot_data

        # Setting ax_cbar=None in clustermap call implies no colorbar
        kws.setdefault("cbar", self.ax_cbar is not None)
        heatmap(self.data2d, ax=self.ax_heatmap, cbar_ax=self.ax_cbar,
                cbar_kws=colorbar_kws, mask=self.mask,
                xticklabels=xtl, yticklabels=ytl, annot=annot, **kws)

        ytl = self.ax_heatmap.get_yticklabels()
        ytl_rot = None if not ytl else ytl[0].get_rotation()
        self.ax_heatmap.yaxis.set_ticks_position('right')
        self.ax_heatmap.yaxis.set_label_position('right')
        if ytl_rot is not None:
            ytl = self.ax_heatmap.get_yticklabels()
            plt.setp(ytl, rotation=ytl_rot)

        tight_params = dict(h_pad=.02, w_pad=.02)
        if self.ax_cbar is None:
            self.fig.tight_layout(**tight_params)
        else:
            # Turn the colorbar axes off for tight layout so that its
            # ticks don't interfere with the rest of the plot layout.
            # Then move it.
            self.ax_cbar.set_axis_off()
            self.fig.tight_layout(**tight_params)
            self.ax_cbar.set_axis_on()
            self.ax_cbar.set_position(self.cbar_pos)
```
### 126 - seaborn/matrix.py:

Start line: 367, End line: 543

```python
@_deprecate_positional_args
def heatmap(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    **kwargs
):
    """Plot rectangular data as a color-encoded matrix.

    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergant data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the data. Note that DataFrames will match on position, not index.
    fmt : str, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
        is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : bool, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    See Also
    --------
    clustermap : Plot a matrix using hierachical clustering to arrange the
                 rows and columns.

    Examples
    --------

    Plot a heatmap for a numpy array:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(0)
        >>> import seaborn as sns; sns.set_theme()
        >>> uniform_data = np.random.rand(10, 12)
        >>> ax = sns.heatmap(uniform_data)

    Change the limits of the colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(uniform_data, vmin=0, vmax=1)

    Plot a heatmap for data centered on 0 with a diverging colormap:

    .. plot::
        :context: close-figs

        >>> normal_data = np.random.randn(10, 12)
        >>> ax = sns.heatmap(normal_data, center=0)

    Plot a dataframe with meaningful row and column labels:

    .. plot::
        :context: close-figs

        >>> flights = sns.load_dataset("flights")
        >>> flights = flights.pivot("month", "year", "passengers")
        >>> ax = sns.heatmap(flights)

    Annotate each cell with the numeric value using integer formatting:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, annot=True, fmt="d")

    Add lines between each cell:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, linewidths=.5)

    Use a different colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cmap="YlGnBu")

    Center the colormap at a specific value:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, center=flights.loc["Jan", 1955])

    Plot every other column label and don't plot row labels:

    .. plot::
        :context: close-figs

        >>> data = np.random.randn(50, 20)
        >>> ax = sns.heatmap(data, xticklabels=2, yticklabels=False)

    Don't draw a colorbar:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cbar=False)

    Use different axes for the colorbar:

    .. plot::
        :context: close-figs

        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        >>> ax = sns.heatmap(flights, ax=ax,
        ...                  cbar_ax=cbar_ax,
        ...                  cbar_kws={"orientation": "horizontal"})

    Use a mask to plot only part of a matrix

    .. plot::
        :context: close-figs

        >>> corr = np.corrcoef(np.random.randn(10, 200))
        >>> mask = np.zeros_like(corr)
        >>> mask[np.triu_indices_from(mask)] = True
        >>> with sns.axes_style("white"):
        ...     f, ax = plt.subplots(figsize=(7, 5))
        ...     ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    """
    # ... other code
```
### 128 - seaborn/matrix.py:

Start line: 931, End line: 959

```python
class ClusterGrid(Grid):

    @staticmethod
    def z_score(data2d, axis=1):
        """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()

        if axis == 1:
            return z_scored
        else:
            return z_scored.T
```
### 144 - seaborn/matrix.py:

Start line: 1412, End line: 1423

```python
@_deprecate_positional_args
def clustermap(
    data, *,
    pivot_kws=None, method='average', metric='euclidean',
    z_score=None, standard_scale=None, figsize=(10, 10),
    cbar_kws=None, row_cluster=True, col_cluster=True,
    row_linkage=None, col_linkage=None,
    row_colors=None, col_colors=None, mask=None,
    dendrogram_ratio=.2, colors_ratio=0.03,
    cbar_pos=(.02, .8, .05, .18), tree_kws=None,
    **kwargs
):
    plotter = ClusterGrid(data, pivot_kws=pivot_kws, figsize=figsize,
                          row_colors=row_colors, col_colors=col_colors,
                          z_score=z_score, standard_scale=standard_scale,
                          mask=mask, dendrogram_ratio=dendrogram_ratio,
                          colors_ratio=colors_ratio, cbar_pos=cbar_pos)

    return plotter.plot(metric=metric, method=method,
                        colorbar_kws=cbar_kws,
                        row_cluster=row_cluster, col_cluster=col_cluster,
                        row_linkage=row_linkage, col_linkage=col_linkage,
                        tree_kws=tree_kws, **kwargs)
```
### 155 - seaborn/matrix.py:

Start line: 998, End line: 1014

```python
class ClusterGrid(Grid):

    def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
        """Get the proportions of the figure taken up by each axes."""
        ratios = [dendrogram_ratio]

        if colors is not None:
            # Colors are encoded as rgb, so ther is an extra dimention
            if np.ndim(colors) > 2:
                n_colors = len(colors)
            else:
                n_colors = 1

            ratios += [n_colors * colors_ratio]

        # Add the ratio for the heatmap itself
        ratios.append(1 - sum(ratios))

        return ratios
```
### 165 - seaborn/matrix.py:

Start line: 961, End line: 996

```python
class ClusterGrid(Grid):

    @staticmethod
    def standard_scale(data2d, axis=1):
        """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.
        vmin : int
            If 0, then subtract the minimum of the data before dividing by
            the range.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        # Normalize these values to range from 0 to 1
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
            standardized.max() - standardized.min())

        if axis == 1:
            return standardized
        else:
            return standardized.T
```
### 169 - seaborn/matrix.py:

Start line: 251, End line: 265

```python
class _HeatMapper(object):

    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        mesh.update_scalarmappable()
        height, width = self.annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array(), mesh.get_facecolors(),
                                       self.annot_data.flat):
            if m is not np.ma.masked:
                lum = relative_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                annotation = ("{:" + self.fmt + "}").format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kws)
                ax.text(x, y, annotation, **text_kwargs)
```
### 190 - seaborn/matrix.py:

Start line: 1219, End line: 1245

```python
class ClusterGrid(Grid):

    def plot(self, metric, method, colorbar_kws, row_cluster, col_cluster,
             row_linkage, col_linkage, tree_kws, **kws):

        # heatmap square=True sets the aspect ratio on the axes, but that is
        # not compatible with the multi-axes layout of clustergrid
        if kws.get("square", False):
            msg = "``square=True`` ignored in clustermap"
            warnings.warn(msg)
            kws.pop("square")

        colorbar_kws = {} if colorbar_kws is None else colorbar_kws

        self.plot_dendrograms(row_cluster, col_cluster, metric, method,
                              row_linkage=row_linkage, col_linkage=col_linkage,
                              tree_kws=tree_kws)
        try:
            xind = self.dendrogram_col.reordered_ind
        except AttributeError:
            xind = np.arange(self.data2d.shape[1])
        try:
            yind = self.dendrogram_row.reordered_ind
        except AttributeError:
            yind = np.arange(self.data2d.shape[0])

        self.plot_colors(xind, yind, **kws)
        self.plot_matrix(colorbar_kws, xind, yind, **kws)
        return self
```
### 203 - seaborn/matrix.py:

Start line: 96, End line: 192

```python
class _HeatMapper(object):
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # Validate the mask and convet to DataFrame
        mask = _matrix_mask(data, mask)

        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)

        # Get good names for the rows and columns
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []

        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []

        # Get the positions and used label for the ticks
        nx, ny = data.T.shape

        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == "auto":
            self.xticks = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels,
                                                             xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels,
                                                             ytickevery)

        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)

        # Sort out the annotations
        if annot is None or annot is False:
            annot = False
            annot_data = None
        else:
            if isinstance(annot, bool):
                annot_data = plot_data
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != plot_data.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
            annot = True

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws.copy()
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
```
### 239 - seaborn/matrix.py:

Start line: 1248, End line: 1411

```python
@_deprecate_positional_args
def clustermap(
    data, *,
    pivot_kws=None, method='average', metric='euclidean',
    z_score=None, standard_scale=None, figsize=(10, 10),
    cbar_kws=None, row_cluster=True, col_cluster=True,
    row_linkage=None, col_linkage=None,
    row_colors=None, col_colors=None, mask=None,
    dendrogram_ratio=.2, colors_ratio=0.03,
    cbar_pos=(.02, .8, .05, .18), tree_kws=None,
    **kwargs
):
    """
    Plot a matrix dataset as a hierarchically-clustered heatmap.

    Parameters
    ----------
    data : 2D array-like
        Rectangular data for clustering. Cannot contain NAs.
    pivot_kws : dict, optional
        If `data` is a tidy dataframe, can provide keyword arguments for
        pivot to create a rectangular dataframe.
    method : str, optional
        Linkage method to use for calculating clusters. See
        :func:`scipy.cluster.hierarchy.linkage` documentation for more
        information.
    metric : str, optional
        Distance metric to use for the data. See
        :func:`scipy.spatial.distance.pdist` documentation for more options.
        To use different metrics (or methods) for rows and columns, you may
        construct each linkage matrix yourself and provide them as
        `{row,col}_linkage`.
    z_score : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
        for the rows or the columns. Z scores are: z = (x - mean)/std, so
        values in each row (column) will get the mean of the row (column)
        subtracted, then divided by the standard deviation of the row (column).
        This ensures that each row (column) has mean of 0 and variance of 1.
    standard_scale : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to standardize that
        dimension, meaning for each row or column, subtract the minimum and
        divide each by its maximum.
    figsize : tuple of (width, height), optional
        Overall size of the figure.
    cbar_kws : dict, optional
        Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
        add a label to the colorbar.
    {row,col}_cluster : bool, optional
        If ``True``, cluster the {rows, columns}.
    {row,col}_linkage : :class:`numpy.ndarray`, optional
        Precomputed linkage matrix for the rows or columns. See
        :func:`scipy.cluster.hierarchy.linkage` for specific formats.
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        List of colors to label for either the rows or columns. Useful to evaluate
        whether samples within a group are clustered together. Can use nested lists or
        DataFrame for multiple color levels of labeling. If given as a
        :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
        extracted from the DataFrames column names or from the name of the Series.
        DataFrame/Series colors are also matched to the data by their index, ensuring
        colors are drawn in the correct order.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where `mask` is True.
        Cells with missing values are automatically masked. Only used for
        visualizing, not for calculating.
    {dendrogram,colors}_ratio : float, or pair of floats, optional
        Proportion of the figure size devoted to the two marginal elements. If
        a pair is given, they correspond to (row, col) ratios.
    cbar_pos : tuple of (left, bottom, width, height), optional
        Position of the colorbar axes in the figure. Setting to ``None`` will
        disable the colorbar.
    tree_kws : dict, optional
        Parameters for the :class:`matplotlib.collections.LineCollection`
        that is used to plot the lines of the dendrogram tree.
    kwargs : other keyword arguments
        All other keyword arguments are passed to :func:`heatmap`.

    Returns
    -------
    :class:`ClusterGrid`
        A :class:`ClusterGrid` instance.

    See Also
    --------
    heatmap : Plot rectangular data as a color-encoded matrix.

    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.

    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``

    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``

    Examples
    --------

    Plot a clustered heatmap:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set_theme(color_codes=True)
        >>> iris = sns.load_dataset("iris")
        >>> species = iris.pop("species")
        >>> g = sns.clustermap(iris)

    Change the size and layout of the figure:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris,
        ...                    figsize=(7, 5),
        ...                    row_cluster=False,
        ...                    dendrogram_ratio=(.1, .2),
        ...                    cbar_pos=(0, .2, .03, .4))

    Add colored labels to identify observations:

    .. plot::
        :context: close-figs

        >>> lut = dict(zip(species.unique(), "rbg"))
        >>> row_colors = species.map(lut)
        >>> g = sns.clustermap(iris, row_colors=row_colors)

    Use a different colormap and adjust the limits of the color range:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris, cmap="mako", vmin=0, vmax=10)

    Use a different similarity metric:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris, metric="correlation")

    Use a different clustering method:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris, method="single")

    Standardize the data within the columns:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris, standard_scale=1)

    Normalize the data within the rows:

    .. plot::
        :context: close-figs

        >>> g = sns.clustermap(iris, z_score=0, cmap="vlag")
    """
    # ... other code
```
