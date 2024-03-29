# mwaskom__seaborn-3217

| **mwaskom/seaborn** | `623b0b723c671e99f04e8ababf19adc563f30168` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 9867 |
| **Any found context length** | 9867 |
| **Avg pos** | 95.0 |
| **Min pos** | 16 |
| **Max pos** | 87 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/seaborn/_core/plot.py b/seaborn/_core/plot.py
--- a/seaborn/_core/plot.py
+++ b/seaborn/_core/plot.py
@@ -1377,10 +1377,9 @@ def _unscale_coords(
     ) -> DataFrame:
         # TODO do we still have numbers in the variable name at this point?
         coord_cols = [c for c in df if re.match(r"^[xy]\D*$", str(c))]
-        drop_cols = [*coord_cols, "width"] if "width" in df else coord_cols
         out_df = (
             df
-            .drop(drop_cols, axis=1)
+            .drop(coord_cols, axis=1)
             .reindex(df.columns, axis=1)  # So unscaled columns retain their place
             .copy(deep=False)
         )
@@ -1396,12 +1395,6 @@ def _unscale_coords(
                 inverted = transform(values)
                 out_df.loc[values.index, str(var)] = inverted
 
-                if var == orient and "width" in view_df:
-                    width = view_df["width"]
-                    out_df.loc[values.index, "width"] = (
-                        transform(values + width / 2) - transform(values - width / 2)
-                    )
-
         return out_df
 
     def _generate_pairings(
diff --git a/seaborn/_marks/bar.py b/seaborn/_marks/bar.py
--- a/seaborn/_marks/bar.py
+++ b/seaborn/_marks/bar.py
@@ -29,17 +29,23 @@ class BarBase(Mark):
 
     def _make_patches(self, data, scales, orient):
 
+        transform = scales[orient]._matplotlib_scale.get_transform()
+        forward = transform.transform
+        reverse = transform.inverted().transform
+
+        other = {"x": "y", "y": "x"}[orient]
+
+        pos = reverse(forward(data[orient]) - data["width"] / 2)
+        width = reverse(forward(data[orient]) + data["width"] / 2) - pos
+
+        val = (data[other] - data["baseline"]).to_numpy()
+        base = data["baseline"].to_numpy()
+
         kws = self._resolve_properties(data, scales)
         if orient == "x":
-            kws["x"] = (data["x"] - data["width"] / 2).to_numpy()
-            kws["y"] = data["baseline"].to_numpy()
-            kws["w"] = data["width"].to_numpy()
-            kws["h"] = (data["y"] - data["baseline"]).to_numpy()
+            kws.update(x=pos, y=base, w=width, h=val)
         else:
-            kws["x"] = data["baseline"].to_numpy()
-            kws["y"] = (data["y"] - data["width"] / 2).to_numpy()
-            kws["w"] = (data["x"] - data["baseline"]).to_numpy()
-            kws["h"] = data["width"].to_numpy()
+            kws.update(x=base, y=pos, w=val, h=width)
 
         kws.pop("width", None)
         kws.pop("baseline", None)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| seaborn/_core/plot.py | 1380 | 1383 | 87 | 18 | 37875
| seaborn/_core/plot.py | 1399 | 1404 | 87 | 18 | 37875
| seaborn/_marks/bar.py | 32 | 40 | 16 | 1 | 9867


## Problem Statement

```
Width computation after histogram slightly wrong with log scale
Note the slight overlap here:

\`\`\`python
(
    so.Plot(tips, "total_bill")
    .add(so.Bars(alpha=.3, edgewidth=0), so.Hist(bins=4))
    .scale(x="log")
)
\`\`\`
![image](https://user-images.githubusercontent.com/315810/178975852-d8fd830e-ae69-487d-be22-36531fca3f8f.png)

It becomes nearly imperceptible with more bins:

\`\`\`
(
    so.Plot(tips, "total_bill")
    .add(so.Bars(alpha=.3, edgewidth=0), so.Hist(bins=8))
    .scale(x="log")
)
\`\`\`
![image](https://user-images.githubusercontent.com/315810/178976113-7026b3ae-0b87-48df-adc0-00e90d5aea94.png)

This is not about `Bars`; `Bar` has it too:

\`\`\`python
(
    so.Plot(tips, "total_bill")
    .add(so.Bar(alpha=.3, edgewidth=0, width=1), so.Hist(bins=4))
    .scale(x="log")
)
\`\`\`
![image](https://user-images.githubusercontent.com/315810/178975910-484df65f-4ce6-482e-9992-5d02faf6b9ea.png)


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 seaborn/_marks/bar.py** | 203 | 251| 432 | 432 | 2123 | 
| 2 | 2 seaborn/distributions.py | 552 | 742| 1491 | 1923 | 21750 | 
| 3 | 2 seaborn/distributions.py | 744 | 899| 1277 | 3200 | 21750 | 
| 4 | 2 seaborn/distributions.py | 1468 | 1594| 1103 | 4303 | 21750 | 
| 5 | 3 examples/histogram_stacked.py | 1 | 30| 152 | 4455 | 21902 | 
| 6 | **3 seaborn/_marks/bar.py** | 133 | 173| 464 | 4919 | 21902 | 
| 7 | **3 seaborn/_marks/bar.py** | 176 | 201| 238 | 5157 | 21902 | 
| 8 | 3 seaborn/distributions.py | 379 | 551| 1323 | 6480 | 21902 | 
| 9 | 4 seaborn/categorical.py | 1425 | 1492| 490 | 6970 | 50418 | 
| 10 | 5 examples/part_whole_bars.py | 1 | 31| 216 | 7186 | 50634 | 
| 11 | 6 examples/joint_histogram.py | 1 | 27| 192 | 7378 | 50826 | 
| 12 | 6 seaborn/distributions.py | 1370 | 1465| 608 | 7986 | 50826 | 
| 13 | 6 seaborn/distributions.py | 2494 | 2547| 537 | 8523 | 50826 | 
| 14 | 6 seaborn/distributions.py | 1134 | 1220| 773 | 9296 | 50826 | 
| 15 | 7 seaborn/_stats/counting.py | 165 | 181| 163 | 9459 | 52763 | 
| **-> 16 <-** | **7 seaborn/_marks/bar.py** | 28 | 73| 408 | 9867 | 52763 | 
| 17 | 7 seaborn/categorical.py | 1542 | 1589| 357 | 10224 | 52763 | 
| 18 | 7 seaborn/categorical.py | 2739 | 2764| 232 | 10456 | 52763 | 
| 19 | 8 doc/tools/generate_logos.py | 61 | 156| 938 | 11394 | 54688 | 
| 20 | 8 seaborn/categorical.py | 2767 | 2826| 387 | 11781 | 54688 | 
| 21 | 9 examples/three_variable_histogram.py | 1 | 16| 0 | 11781 | 54776 | 
| 22 | **9 seaborn/_marks/bar.py** | 106 | 131| 239 | 12020 | 54776 | 
| 23 | 9 seaborn/distributions.py | 173 | 194| 228 | 12248 | 54776 | 
| 24 | 9 seaborn/distributions.py | 1 | 92| 618 | 12866 | 54776 | 
| 25 | 9 seaborn/_stats/counting.py | 201 | 233| 316 | 13182 | 54776 | 
| 26 | 10 seaborn/_statistics.py | 367 | 398| 282 | 13464 | 59177 | 
| 27 | 10 seaborn/distributions.py | 2187 | 2277| 810 | 14274 | 59177 | 
| 28 | 10 seaborn/_statistics.py | 335 | 365| 262 | 14536 | 59177 | 
| 29 | 10 seaborn/_statistics.py | 254 | 273| 191 | 14727 | 59177 | 
| 30 | 10 seaborn/distributions.py | 901 | 1034| 937 | 15664 | 59177 | 
| 31 | 11 examples/faceted_histogram.py | 1 | 15| 0 | 15664 | 59262 | 
| 32 | 11 seaborn/categorical.py | 2959 | 3000| 236 | 15900 | 59262 | 
| 33 | 12 examples/smooth_bivariate_kde.py | 1 | 17| 131 | 16031 | 59393 | 
| 34 | 13 examples/grouped_boxplot.py | 1 | 19| 103 | 16134 | 59496 | 
| 35 | 13 seaborn/categorical.py | 2419 | 2492| 670 | 16804 | 59496 | 
| 36 | 13 seaborn/_statistics.py | 275 | 333| 402 | 17206 | 59496 | 
| 37 | 13 seaborn/_stats/counting.py | 183 | 199| 145 | 17351 | 59496 | 
| 38 | 13 seaborn/_statistics.py | 519 | 555| 257 | 17608 | 59496 | 
| 39 | 14 examples/horizontal_boxplot.py | 1 | 31| 192 | 17800 | 59688 | 
| 40 | 14 seaborn/_stats/counting.py | 143 | 163| 201 | 18001 | 59688 | 
| 41 | 14 seaborn/distributions.py | 2385 | 2402| 168 | 18169 | 59688 | 
| 42 | 15 examples/grouped_barplot.py | 1 | 21| 122 | 18291 | 59810 | 
| 43 | 15 seaborn/distributions.py | 1036 | 1133| 692 | 18983 | 59810 | 
| 44 | 15 seaborn/categorical.py | 1741 | 1786| 400 | 19383 | 59810 | 
| 45 | 15 seaborn/_stats/counting.py | 49 | 120| 732 | 20115 | 59810 | 
| 46 | 15 seaborn/distributions.py | 2069 | 2108| 222 | 20337 | 59810 | 
| 47 | 16 examples/marginal_ticks.py | 1 | 16| 144 | 20481 | 59954 | 
| 48 | 16 seaborn/distributions.py | 2278 | 2300| 296 | 20777 | 59954 | 
| 49 | 17 examples/wide_data_lineplot.py | 1 | 20| 137 | 20914 | 60091 | 
| 50 | 17 seaborn/_stats/counting.py | 122 | 141| 163 | 21077 | 60091 | 
| 51 | **18 seaborn/_core/plot.py** | 1171 | 1293| 1179 | 22256 | 73686 | 
| 52 | 19 examples/kde_ridgeplot.py | 1 | 51| 399 | 22655 | 74085 | 
| 53 | 19 seaborn/distributions.py | 1989 | 2066| 699 | 23354 | 74085 | 
| 54 | 20 examples/hexbin_marginals.py | 1 | 16| 0 | 23354 | 74177 | 
| 55 | 20 seaborn/distributions.py | 1597 | 1746| 1320 | 24674 | 74177 | 
| 56 | 20 seaborn/categorical.py | 1842 | 1976| 1205 | 25879 | 74177 | 
| 57 | 21 seaborn/relational.py | 423 | 521| 818 | 26697 | 82698 | 
| 58 | 21 seaborn/categorical.py | 927 | 1031| 867 | 27564 | 82698 | 
| 59 | 22 seaborn/_core/scales.py | 1024 | 1046| 186 | 27750 | 90504 | 
| 60 | 23 examples/large_distributions.py | 1 | 15| 107 | 27857 | 90611 | 
| 61 | 23 seaborn/categorical.py | 1032 | 1050| 172 | 28029 | 90611 | 
| 62 | 23 seaborn/categorical.py | 1523 | 1540| 162 | 28191 | 90611 | 
| 63 | 23 seaborn/_core/scales.py | 202 | 218| 188 | 28379 | 90611 | 
| 64 | 24 examples/wide_form_violinplot.py | 1 | 35| 291 | 28670 | 90902 | 
| 65 | 24 seaborn/_core/scales.py | 1049 | 1066| 187 | 28857 | 90902 | 
| 66 | 24 seaborn/_core/scales.py | 667 | 721| 441 | 29298 | 90902 | 
| 67 | 25 doc/sphinxext/tutorial_builder.py | 204 | 214| 122 | 29420 | 94205 | 
| 68 | 25 seaborn/categorical.py | 1105 | 1136| 241 | 29661 | 94205 | 
| 69 | 25 seaborn/distributions.py | 297 | 377| 516 | 30177 | 94205 | 
| 70 | 26 seaborn/axisgrid.py | 2179 | 2339| 1464 | 31641 | 113807 | 
| 71 | **26 seaborn/_core/plot.py** | 1080 | 1132| 357 | 31998 | 113807 | 
| 72 | 27 examples/errorband_lineplots.py | 1 | 18| 0 | 31998 | 113896 | 
| 73 | 27 seaborn/categorical.py | 1327 | 1359| 326 | 32324 | 113896 | 
| 74 | 28 examples/many_facets.py | 1 | 40| 312 | 32636 | 114208 | 
| 75 | 28 seaborn/distributions.py | 230 | 295| 638 | 33274 | 114208 | 
| 76 | 29 examples/grouped_violinplots.py | 1 | 18| 121 | 33395 | 114329 | 
| 77 | 29 seaborn/categorical.py | 2317 | 2396| 737 | 34132 | 114329 | 
| 78 | 29 seaborn/categorical.py | 1494 | 1520| 322 | 34454 | 114329 | 
| 79 | **29 seaborn/_marks/bar.py** | 91 | 103| 113 | 34567 | 114329 | 
| 80 | 29 seaborn/distributions.py | 1291 | 1323| 301 | 34868 | 114329 | 
| 81 | **29 seaborn/_marks/bar.py** | 75 | 89| 146 | 35014 | 114329 | 
| 82 | 29 seaborn/distributions.py | 2405 | 2492| 724 | 35738 | 114329 | 
| 83 | 29 seaborn/distributions.py | 1222 | 1289| 577 | 36315 | 114329 | 
| 84 | 29 seaborn/_core/scales.py | 415 | 491| 577 | 36892 | 114329 | 
| 85 | 30 seaborn/_core/moves.py | 153 | 190| 304 | 37196 | 116189 | 
| 86 | 31 seaborn/_oldcore.py | 1093 | 1137| 372 | 37568 | 129661 | 
| **-> 87 <-** | **31 seaborn/_core/plot.py** | 1375 | 1405| 307 | 37875 | 129661 | 
| 88 | 31 seaborn/_core/scales.py | 977 | 994| 175 | 38050 | 129661 | 
| 89 | 31 seaborn/categorical.py | 1138 | 1311| 1218 | 39268 | 129661 | 
| 90 | 31 seaborn/_oldcore.py | 1217 | 1295| 777 | 40045 | 129661 | 
| 91 | 31 seaborn/categorical.py | 2851 | 2916| 501 | 40546 | 129661 | 
| 92 | 31 seaborn/distributions.py | 1877 | 1930| 368 | 40914 | 129661 | 
| 93 | 31 seaborn/categorical.py | 2919 | 2956| 267 | 41181 | 129661 | 
| 94 | 31 seaborn/categorical.py | 1361 | 1373| 188 | 41369 | 129661 | 
| 95 | 31 seaborn/relational.py | 603 | 646| 416 | 41785 | 129661 | 
| 96 | 32 seaborn/miscplot.py | 1 | 30| 231 | 42016 | 130039 | 
| 97 | 33 examples/regression_marginals.py | 1 | 15| 0 | 42016 | 130130 | 
| 98 | 33 seaborn/axisgrid.py | 2087 | 2154| 726 | 42742 | 130130 | 
| 99 | 34 examples/palette_generation.py | 1 | 36| 282 | 43024 | 130412 | 
| 100 | 35 seaborn/matrix.py | 1060 | 1115| 562 | 43586 | 141016 | 
| 101 | 35 seaborn/categorical.py | 1395 | 1422| 295 | 43881 | 141016 | 
| 102 | 35 seaborn/_core/scales.py | 653 | 665| 138 | 44019 | 141016 | 
| 103 | 35 seaborn/categorical.py | 1375 | 1393| 174 | 44193 | 141016 | 
| 104 | 35 seaborn/categorical.py | 321 | 416| 718 | 44911 | 141016 | 
| 105 | 35 doc/tools/generate_logos.py | 159 | 225| 617 | 45528 | 141016 | 
| 106 | 35 seaborn/categorical.py | 1788 | 1824| 479 | 46007 | 141016 | 
| 107 | 35 seaborn/categorical.py | 1646 | 1738| 749 | 46756 | 141016 | 
| 108 | 35 seaborn/categorical.py | 2243 | 2294| 347 | 47103 | 141016 | 
| 109 | 35 seaborn/categorical.py | 1835 | 1840| 116 | 47219 | 141016 | 
| 110 | 35 seaborn/relational.py | 176 | 284| 937 | 48156 | 141016 | 
| 111 | 35 seaborn/distributions.py | 2111 | 2185| 585 | 48741 | 141016 | 
| 112 | 36 examples/scatter_bubbles.py | 1 | 18| 111 | 48852 | 141127 | 
| 113 | 36 seaborn/distributions.py | 1933 | 1986| 340 | 49192 | 141127 | 
| 114 | 37 seaborn/regression.py | 1 | 22| 121 | 49313 | 149080 | 
| 115 | 37 seaborn/categorical.py | 1073 | 1103| 225 | 49538 | 149080 | 
| 116 | 37 seaborn/_oldcore.py | 759 | 856| 787 | 50325 | 149080 | 
| 117 | 37 seaborn/categorical.py | 2399 | 2416| 191 | 50516 | 149080 | 
| 118 | **37 seaborn/_core/plot.py** | 980 | 1078| 902 | 51418 | 149080 | 
| 119 | 37 seaborn/categorical.py | 2224 | 2240| 146 | 51564 | 149080 | 
| 120 | 37 seaborn/categorical.py | 226 | 253| 273 | 51837 | 149080 | 
| 121 | 38 seaborn/_core/properties.py | 275 | 331| 420 | 52257 | 155631 | 
| 122 | 38 seaborn/distributions.py | 1325 | 1367| 351 | 52608 | 155631 | 
| 123 | 39 examples/jitter_stripplot.py | 1 | 39| 278 | 52886 | 155909 | 
| 124 | **39 seaborn/_core/plot.py** | 603 | 630| 301 | 53187 | 155909 | 
| 125 | **39 seaborn/_core/plot.py** | 1295 | 1373| 768 | 53955 | 155909 | 
| 126 | 39 seaborn/categorical.py | 796 | 856| 453 | 54408 | 155909 | 
| 127 | 39 seaborn/categorical.py | 3003 | 3187| 1564 | 55972 | 155909 | 
| 128 | 40 seaborn/external/husl.py | 94 | 114| 301 | 56273 | 158579 | 
| 129 | 41 seaborn/utils.py | 318 | 385| 645 | 56918 | 165175 | 
| 130 | 41 seaborn/categorical.py | 2829 | 2848| 198 | 57116 | 165175 | 
| 131 | 41 seaborn/categorical.py | 3352 | 3422| 668 | 57784 | 165175 | 
| 132 | 42 seaborn/_marks/dot.py | 62 | 85| 192 | 57976 | 166690 | 
| 133 | 42 seaborn/axisgrid.py | 1 | 29| 174 | 58150 | 166690 | 
| 134 | 42 doc/sphinxext/tutorial_builder.py | 130 | 179| 620 | 58770 | 166690 | 
| 135 | 43 examples/pairgrid_dotplot.py | 1 | 39| 268 | 59038 | 166958 | 
| 136 | 43 seaborn/utils.py | 25 | 52| 191 | 59229 | 166958 | 
| 137 | 44 seaborn/_docstrings.py | 137 | 199| 393 | 59622 | 168396 | 
| 138 | 44 seaborn/distributions.py | 1749 | 1874| 1114 | 60736 | 168396 | 
| 139 | 44 seaborn/matrix.py | 995 | 1058| 543 | 61279 | 168396 | 
| 140 | 45 examples/scatterplot_categorical.py | 1 | 17| 0 | 61279 | 168489 | 
| 141 | 45 seaborn/_oldcore.py | 1330 | 1353| 198 | 61477 | 168489 | 
| 142 | 46 examples/scatterplot_sizes.py | 1 | 25| 164 | 61641 | 168653 | 
| 143 | **46 seaborn/_core/plot.py** | 910 | 947| 401 | 62042 | 168653 | 
| 144 | 46 seaborn/categorical.py | 2554 | 2614| 471 | 62513 | 168653 | 
| 145 | 46 seaborn/matrix.py | 265 | 276| 123 | 62636 | 168653 | 
| 146 | 46 seaborn/categorical.py | 2297 | 2314| 166 | 62802 | 168653 | 
| 147 | 46 seaborn/regression.py | 736 | 760| 279 | 63081 | 168653 | 
| 148 | 46 seaborn/matrix.py | 192 | 247| 517 | 63598 | 168653 | 
| 149 | 47 examples/heat_scatter.py | 1 | 42| 330 | 63928 | 168983 | 
| 150 | 48 seaborn/_stats/density.py | 172 | 215| 378 | 64306 | 170939 | 
| 151 | 48 seaborn/distributions.py | 2303 | 2382| 602 | 64908 | 170939 | 
| 152 | 48 seaborn/regression.py | 557 | 641| 852 | 65760 | 170939 | 
| 153 | **48 seaborn/_core/plot.py** | 1134 | 1169| 284 | 66044 | 170939 | 
| 154 | 48 seaborn/axisgrid.py | 2342 | 2401| 413 | 66457 | 170939 | 
| 155 | 48 seaborn/categorical.py | 894 | 925| 273 | 66730 | 170939 | 
| 156 | 48 seaborn/matrix.py | 278 | 292| 172 | 66902 | 170939 | 
| 157 | 48 seaborn/matrix.py | 900 | 916| 127 | 67029 | 170939 | 
| 158 | 48 seaborn/categorical.py | 3188 | 3262| 804 | 67833 | 170939 | 
| 159 | 49 seaborn/_stats/order.py | 2 | 34| 192 | 68025 | 171523 | 
| 160 | 49 seaborn/categorical.py | 255 | 319| 508 | 68533 | 171523 | 
| 161 | 49 seaborn/axisgrid.py | 1513 | 1550| 279 | 68812 | 171523 | 
| 162 | 49 seaborn/external/husl.py | 117 | 153| 266 | 69078 | 171523 | 
| 163 | 49 seaborn/categorical.py | 858 | 891| 281 | 69359 | 171523 | 
| 164 | 50 examples/faceted_lineplot.py | 1 | 24| 140 | 69499 | 171663 | 
| 165 | 50 seaborn/categorical.py | 1052 | 1071| 210 | 69709 | 171663 | 
| 166 | 50 seaborn/distributions.py | 196 | 228| 406 | 70115 | 171663 | 
| 167 | 50 seaborn/_oldcore.py | 1297 | 1315| 136 | 70251 | 171663 | 
| 168 | 50 seaborn/_stats/density.py | 1 | 92| 856 | 71107 | 171663 | 
| 169 | **50 seaborn/_core/plot.py** | 632 | 649| 136 | 71243 | 171663 | 
| 170 | **50 seaborn/_core/plot.py** | 1471 | 1543| 599 | 71842 | 171663 | 
| 171 | 50 seaborn/_core/scales.py | 997 | 1021| 139 | 71981 | 171663 | 
| 172 | 50 seaborn/_core/scales.py | 98 | 113| 134 | 72115 | 171663 | 
| 173 | 51 examples/layered_bivariate_plot.py | 1 | 24| 186 | 72301 | 171849 | 
| 174 | 52 examples/anscombes_quartet.py | 1 | 19| 124 | 72425 | 171973 | 
| 175 | 52 seaborn/axisgrid.py | 1673 | 1761| 855 | 73280 | 171973 | 
| 176 | 52 seaborn/_core/scales.py | 493 | 524| 244 | 73524 | 171973 | 
| 177 | 52 seaborn/axisgrid.py | 2156 | 2176| 257 | 73781 | 171973 | 
| 178 | 52 seaborn/categorical.py | 2495 | 2551| 421 | 74202 | 171973 | 
| 179 | 52 seaborn/matrix.py | 445 | 460| 226 | 74428 | 171973 | 
| 180 | 53 seaborn/palettes.py | 48 | 58| 152 | 74580 | 179203 | 
| 181 | 53 seaborn/categorical.py | 2678 | 2736| 424 | 75004 | 179203 | 
| 182 | 53 seaborn/matrix.py | 1117 | 1143| 274 | 75278 | 179203 | 
| 183 | 53 seaborn/_core/scales.py | 723 | 760| 236 | 75514 | 179203 | 
| 184 | 53 seaborn/_core/scales.py | 84 | 96| 127 | 75641 | 179203 | 
| 185 | 53 seaborn/matrix.py | 585 | 639| 512 | 76153 | 179203 | 
| 186 | 53 seaborn/categorical.py | 3520 | 3547| 245 | 76398 | 179203 | 
| 187 | 53 seaborn/categorical.py | 2617 | 2675| 398 | 76796 | 179203 | 
| 188 | **53 seaborn/_core/plot.py** | 329 | 342| 136 | 76932 | 179203 | 
| 189 | 53 seaborn/axisgrid.py | 304 | 360| 503 | 77435 | 179203 | 
| 190 | 54 examples/paired_pointplots.py | 1 | 21| 153 | 77588 | 179356 | 
| 191 | 54 seaborn/categorical.py | 1313 | 1325| 133 | 77721 | 179356 | 
| 192 | 54 seaborn/external/husl.py | 71 | 91| 287 | 78008 | 179356 | 
| 193 | 54 seaborn/_core/scales.py | 1 | 52| 308 | 78316 | 179356 | 
| 194 | 54 seaborn/relational.py | 1 | 50| 384 | 78700 | 179356 | 
| 195 | 54 seaborn/_core/scales.py | 220 | 238| 144 | 78844 | 179356 | 
| 196 | 54 doc/sphinxext/tutorial_builder.py | 217 | 243| 282 | 79126 | 179356 | 
| 197 | 54 seaborn/_core/scales.py | 873 | 957| 603 | 79729 | 179356 | 
| 198 | 54 seaborn/categorical.py | 1 | 39| 252 | 79981 | 179356 | 
| 199 | **54 seaborn/_core/plot.py** | 823 | 857| 304 | 80285 | 179356 | 
| 200 | 54 seaborn/_core/moves.py | 221 | 275| 329 | 80614 | 179356 | 
| 201 | 54 seaborn/categorical.py | 779 | 794| 127 | 80741 | 179356 | 
| 202 | 54 seaborn/matrix.py | 970 | 993| 264 | 81005 | 179356 | 
| 203 | 54 seaborn/categorical.py | 1978 | 2221| 539 | 81544 | 179356 | 
| 204 | 54 seaborn/_core/scales.py | 307 | 324| 213 | 81757 | 179356 | 
| 205 | 54 seaborn/axisgrid.py | 1431 | 1511| 616 | 82373 | 179356 | 
| 206 | 54 seaborn/_core/scales.py | 188 | 200| 160 | 82533 | 179356 | 
| 207 | 54 seaborn/categorical.py | 419 | 628| 1386 | 83919 | 179356 | 
| 208 | 54 seaborn/regression.py | 317 | 338| 216 | 84135 | 179356 | 
| 209 | 55 seaborn/_core/data.py | 181 | 261| 726 | 84861 | 181281 | 
| 210 | 56 seaborn/widgets.py | 345 | 369| 236 | 85097 | 184700 | 
| 211 | 57 examples/palette_choices.py | 1 | 38| 314 | 85411 | 185014 | 
| 212 | 58 examples/simple_violinplots.py | 1 | 19| 121 | 85532 | 185135 | 
| 213 | 59 seaborn/_stats/aggregation.py | 52 | 91| 345 | 85877 | 185957 | 
| 214 | 60 examples/multiple_bivariate_kde.py | 1 | 25| 126 | 86003 | 186083 | 
| 215 | 61 examples/multiple_ecdf.py | 1 | 18| 122 | 86125 | 186205 | 
| 216 | 61 seaborn/distributions.py | 95 | 136| 316 | 86441 | 186205 | 
| 217 | 61 seaborn/relational.py | 931 | 991| 639 | 87080 | 186205 | 
| 218 | **61 seaborn/_marks/bar.py** | 1 | 25| 119 | 87199 | 186205 | 
| 219 | 61 doc/sphinxext/tutorial_builder.py | 246 | 271| 343 | 87542 | 186205 | 
| 220 | 61 seaborn/relational.py | 825 | 929| 840 | 88382 | 186205 | 
| 221 | 61 seaborn/widgets.py | 1 | 44| 341 | 88723 | 186205 | 
| 222 | 61 seaborn/relational.py | 378 | 422| 524 | 89247 | 186205 | 
| 223 | 61 seaborn/_stats/density.py | 122 | 131| 126 | 89373 | 186205 | 
| 224 | 61 seaborn/_core/scales.py | 959 | 975| 136 | 89509 | 186205 | 
| 225 | 61 seaborn/_stats/order.py | 62 | 79| 207 | 89716 | 186205 | 
| 226 | **61 seaborn/_core/plot.py** | 651 | 670| 146 | 89862 | 186205 | 
| 227 | 61 doc/sphinxext/tutorial_builder.py | 287 | 310| 195 | 90057 | 186205 | 
| 228 | 61 seaborn/_core/scales.py | 164 | 186| 197 | 90254 | 186205 | 
| 229 | 61 seaborn/axisgrid.py | 1613 | 1655| 327 | 90581 | 186205 | 
| 230 | 61 seaborn/categorical.py | 195 | 224| 370 | 90951 | 186205 | 


## Patch

```diff
diff --git a/seaborn/_core/plot.py b/seaborn/_core/plot.py
--- a/seaborn/_core/plot.py
+++ b/seaborn/_core/plot.py
@@ -1377,10 +1377,9 @@ def _unscale_coords(
     ) -> DataFrame:
         # TODO do we still have numbers in the variable name at this point?
         coord_cols = [c for c in df if re.match(r"^[xy]\D*$", str(c))]
-        drop_cols = [*coord_cols, "width"] if "width" in df else coord_cols
         out_df = (
             df
-            .drop(drop_cols, axis=1)
+            .drop(coord_cols, axis=1)
             .reindex(df.columns, axis=1)  # So unscaled columns retain their place
             .copy(deep=False)
         )
@@ -1396,12 +1395,6 @@ def _unscale_coords(
                 inverted = transform(values)
                 out_df.loc[values.index, str(var)] = inverted
 
-                if var == orient and "width" in view_df:
-                    width = view_df["width"]
-                    out_df.loc[values.index, "width"] = (
-                        transform(values + width / 2) - transform(values - width / 2)
-                    )
-
         return out_df
 
     def _generate_pairings(
diff --git a/seaborn/_marks/bar.py b/seaborn/_marks/bar.py
--- a/seaborn/_marks/bar.py
+++ b/seaborn/_marks/bar.py
@@ -29,17 +29,23 @@ class BarBase(Mark):
 
     def _make_patches(self, data, scales, orient):
 
+        transform = scales[orient]._matplotlib_scale.get_transform()
+        forward = transform.transform
+        reverse = transform.inverted().transform
+
+        other = {"x": "y", "y": "x"}[orient]
+
+        pos = reverse(forward(data[orient]) - data["width"] / 2)
+        width = reverse(forward(data[orient]) + data["width"] / 2) - pos
+
+        val = (data[other] - data["baseline"]).to_numpy()
+        base = data["baseline"].to_numpy()
+
         kws = self._resolve_properties(data, scales)
         if orient == "x":
-            kws["x"] = (data["x"] - data["width"] / 2).to_numpy()
-            kws["y"] = data["baseline"].to_numpy()
-            kws["w"] = data["width"].to_numpy()
-            kws["h"] = (data["y"] - data["baseline"]).to_numpy()
+            kws.update(x=pos, y=base, w=width, h=val)
         else:
-            kws["x"] = data["baseline"].to_numpy()
-            kws["y"] = (data["y"] - data["width"] / 2).to_numpy()
-            kws["w"] = (data["x"] - data["baseline"]).to_numpy()
-            kws["h"] = data["width"].to_numpy()
+            kws.update(x=base, y=pos, w=val, h=width)
 
         kws.pop("width", None)
         kws.pop("baseline", None)

```

## Test Patch

```diff
diff --git a/tests/_marks/test_bar.py b/tests/_marks/test_bar.py
--- a/tests/_marks/test_bar.py
+++ b/tests/_marks/test_bar.py
@@ -200,3 +200,13 @@ def test_unfilled(self, x, y):
         colors = p._theme["axes.prop_cycle"].by_key()["color"]
         assert_array_equal(fcs, to_rgba_array([colors[0]] * len(x), 0))
         assert_array_equal(ecs, to_rgba_array([colors[4]] * len(x), 1))
+
+    def test_log_scale(self):
+
+        x = y = [1, 10, 100, 1000]
+        p = Plot(x, y).add(Bars()).scale(x="log").plot()
+        ax = p._figure.axes[0]
+
+        paths = ax.collections[0].get_paths()
+        for a, b in zip(paths, paths[1:]):
+            assert a.vertices[1, 0] == pytest.approx(b.vertices[0, 0])

```


## Code snippets

### 1 - seaborn/_marks/bar.py:

Start line: 203, End line: 251

```python
@document_properties
@dataclass
class Bars(BarBase):

    def _plot(self, split_gen, scales, orient):

        ori_idx = ["x", "y"].index(orient)
        val_idx = ["y", "x"].index(orient)

        patches = defaultdict(list)
        for _, data, ax in split_gen():
            bars, _ = self._make_patches(data, scales, orient)
            patches[ax].extend(bars)

        collections = {}
        for ax, ax_patches in patches.items():

            col = mpl.collections.PatchCollection(ax_patches, match_original=True)
            col.sticky_edges[val_idx][:] = (0, np.inf)
            ax.add_collection(col, autolim=False)
            collections[ax] = col

            # Workaround for matplotlib autoscaling bug
            # https://github.com/matplotlib/matplotlib/issues/11898
            # https://github.com/matplotlib/matplotlib/issues/23129
            xys = np.vstack([path.vertices for path in col.get_paths()])
            ax.update_datalim(xys)

        if "edgewidth" not in scales and isinstance(self.edgewidth, Mappable):

            for ax in collections:
                ax.autoscale_view()

            def get_dimensions(collection):
                edges, widths = [], []
                for verts in (path.vertices for path in collection.get_paths()):
                    edges.append(min(verts[:, ori_idx]))
                    widths.append(np.ptp(verts[:, ori_idx]))
                return np.array(edges), np.array(widths)

            min_width = np.inf
            for ax, col in collections.items():
                edges, widths = get_dimensions(col)
                points = 72 / ax.figure.dpi * abs(
                    ax.transData.transform([edges + widths] * 2)
                    - ax.transData.transform([edges] * 2)
                )
                min_width = min(min_width, min(points[:, ori_idx]))

            linewidth = min(.1 * min_width, mpl.rcParams["patch.linewidth"])
            for _, col in collections.items():
                col.set_linewidth(linewidth)
```
### 2 - seaborn/distributions.py:

Start line: 552, End line: 742

```python
class _DistributionPlotter(VectorPlotter):

    def plot_univariate_histogram(
        self,
        multiple,
        element,
        fill,
        common_norm,
        common_bins,
        shrink,
        kde,
        kde_kws,
        color,
        legend,
        line_kws,
        estimate_kws,
        **plot_kws,
    ):
        kde_kws =
        # ... other code
        for sub_vars, _ in self.iter_data("hue", reverse=True):

            key = tuple(sub_vars.items())
            hist = histograms[key].rename("heights").reset_index()
            bottom = np.asarray(baselines[key])

            ax = self._get_axes(sub_vars)

            # Define the matplotlib attributes that depend on semantic mapping
            if "hue" in self.variables:
                sub_color = self._hue_map(sub_vars["hue"])
            else:
                sub_color = color

            artist_kws = self._artist_kws(
                plot_kws, fill, element, multiple, sub_color, alpha
            )

            if element == "bars":

                # Use matplotlib bar plotting

                plot_func = ax.bar if self.data_variable == "x" else ax.barh
                artists = plot_func(
                    hist["edges"],
                    hist["heights"] - bottom,
                    hist["widths"],
                    bottom,
                    align="edge",
                    **artist_kws,
                )

                for bar in artists:
                    if self.data_variable == "x":
                        bar.sticky_edges.x[:] = sticky_data
                        bar.sticky_edges.y[:] = sticky_stat
                    else:
                        bar.sticky_edges.x[:] = sticky_stat
                        bar.sticky_edges.y[:] = sticky_data

                hist_artists.extend(artists)

            else:

                # Use either fill_between or plot to draw hull of histogram
                if element == "step":

                    final = hist.iloc[-1]
                    x = np.append(hist["edges"], final["edges"] + final["widths"])
                    y = np.append(hist["heights"], final["heights"])
                    b = np.append(bottom, bottom[-1])

                    if self.data_variable == "x":
                        step = "post"
                        drawstyle = "steps-post"
                    else:
                        step = "post"  # fillbetweenx handles mapping internally
                        drawstyle = "steps-pre"

                elif element == "poly":

                    x = hist["edges"] + hist["widths"] / 2
                    y = hist["heights"]
                    b = bottom

                    step = None
                    drawstyle = None

                if self.data_variable == "x":
                    if fill:
                        artist = ax.fill_between(x, b, y, step=step, **artist_kws)
                    else:
                        artist, = ax.plot(x, y, drawstyle=drawstyle, **artist_kws)
                    artist.sticky_edges.x[:] = sticky_data
                    artist.sticky_edges.y[:] = sticky_stat
                else:
                    if fill:
                        artist = ax.fill_betweenx(x, b, y, step=step, **artist_kws)
                    else:
                        artist, = ax.plot(y, x, drawstyle=drawstyle, **artist_kws)
                    artist.sticky_edges.x[:] = sticky_stat
                    artist.sticky_edges.y[:] = sticky_data

                hist_artists.append(artist)

            if kde:

                # Add in the density curves

                try:
                    density = densities[key]
                except KeyError:
                    continue
                support = density.index

                if "x" in self.variables:
                    line_args = support, density
                    sticky_x, sticky_y = None, (0, np.inf)
                else:
                    line_args = density, support
                    sticky_x, sticky_y = (0, np.inf), None

                line_kws["color"] = to_rgba(sub_color, 1)
                line, = ax.plot(
                    *line_args, **line_kws,
                )

                if sticky_x is not None:
                    line.sticky_edges.x[:] = sticky_x
                if sticky_y is not None:
                    line.sticky_edges.y[:] = sticky_y

        if element == "bars" and "linewidth" not in plot_kws:

            # Now we handle linewidth, which depends on the scaling of the plot

            # We will base everything on the minimum bin width
            hist_metadata = pd.concat([
                # Use .items for generality over dict or df
                h.index.to_frame() for _, h in histograms.items()
            ]).reset_index(drop=True)
            thin_bar_idx = hist_metadata["widths"].idxmin()
            binwidth = hist_metadata.loc[thin_bar_idx, "widths"]
            left_edge = hist_metadata.loc[thin_bar_idx, "edges"]

            # Set initial value
            default_linewidth = math.inf

            # Loop through subsets based only on facet variables
            for sub_vars, _ in self.iter_data():

                ax = self._get_axes(sub_vars)

                # Needed in some cases to get valid transforms.
                # Innocuous in other cases?
                ax.autoscale_view()

                # Convert binwidth from data coordinates to pixels
                pts_x, pts_y = 72 / ax.figure.dpi * abs(
                    ax.transData.transform([left_edge + binwidth] * 2)
                    - ax.transData.transform([left_edge] * 2)
                )
                if self.data_variable == "x":
                    binwidth_points = pts_x
                else:
                    binwidth_points = pts_y

                # The relative size of the lines depends on the appearance
                # This is a provisional value and may need more tweaking
                default_linewidth = min(.1 * binwidth_points, default_linewidth)

            # Set the attributes
            for bar in hist_artists:

                # Don't let the lines get too thick
                max_linewidth = bar.get_linewidth()
                if not fill:
                    max_linewidth *= 1.5

                linewidth = min(default_linewidth, max_linewidth)

                # If not filling, don't let lines disappear
                if not fill:
                    min_linewidth = .5
                    linewidth = max(linewidth, min_linewidth)

                bar.set_linewidth(linewidth)

        # --- Finalize the plot ----

        # Axis labels
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        default_x = default_y = ""
        if self.data_variable == "x":
            default_y = estimator.stat.capitalize()
        if self.data_variable == "y":
            default_x = estimator.stat.capitalize()
        self._add_axis_labels(ax, default_x, default_y)

        # Legend for semantic variables
        if "hue" in self.variables and legend:

            if fill or element == "bars":
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])

            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, fill, element, multiple, alpha, plot_kws, {},
            )
```
### 3 - seaborn/distributions.py:

Start line: 744, End line: 899

```python
class _DistributionPlotter(VectorPlotter):

    def plot_bivariate_histogram(
        self,
        common_bins, common_norm,
        thresh, pthresh, pmax,
        color, legend,
        cbar, cbar_ax, cbar_kws,
        estimate_kws,
        **plot_kws,
    ):

        # Default keyword dicts
        cbar_kws = {} if cbar_kws is None else cbar_kws.copy()

        # Now initialize the Histogram estimator
        estimator = Histogram(**estimate_kws)

        # Do pre-compute housekeeping related to multiple groups
        if set(self.variables) - {"x", "y"}:
            all_data = self.comp_data.dropna()
            if common_bins:
                estimator.define_bin_params(
                    all_data["x"],
                    all_data["y"],
                    all_data.get("weights", None),
                )
        else:
            common_norm = False

        # -- Determine colormap threshold and norm based on the full data

        full_heights = []
        for _, sub_data in self.iter_data(from_comp_data=True):
            sub_heights, _ = estimator(
                sub_data["x"], sub_data["y"], sub_data.get("weights", None)
            )
            full_heights.append(sub_heights)

        common_color_norm = not set(self.variables) - {"x", "y"} or common_norm

        if pthresh is not None and common_color_norm:
            thresh = self._quantile_to_level(full_heights, pthresh)

        plot_kws.setdefault("vmin", 0)
        if common_color_norm:
            if pmax is not None:
                vmax = self._quantile_to_level(full_heights, pmax)
            else:
                vmax = plot_kws.pop("vmax", max(map(np.max, full_heights)))
        else:
            vmax = None

        # Get a default color
        # (We won't follow the color cycle here, as multiple plots are unlikely)
        if color is None:
            color = "C0"

        # --- Loop over data (subsets) and draw the histograms
        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            if sub_data.empty:
                continue

            # Do the histogram computation
            heights, (x_edges, y_edges) = estimator(
                sub_data["x"],
                sub_data["y"],
                weights=sub_data.get("weights", None),
            )

            # Check for log scaling on the data axis
            if self._log_scaled("x"):
                x_edges = np.power(10, x_edges)
            if self._log_scaled("y"):
                y_edges = np.power(10, y_edges)

            # Apply scaling to normalize across groups
            if estimator.stat != "count" and common_norm:
                heights *= len(sub_data) / len(all_data)

            # Define the specific kwargs for this artist
            artist_kws = plot_kws.copy()
            if "hue" in self.variables:
                color = self._hue_map(sub_vars["hue"])
                cmap = self._cmap_from_color(color)
                artist_kws["cmap"] = cmap
            else:
                cmap = artist_kws.pop("cmap", None)
                if isinstance(cmap, str):
                    cmap = color_palette(cmap, as_cmap=True)
                elif cmap is None:
                    cmap = self._cmap_from_color(color)
                artist_kws["cmap"] = cmap

            # Set the upper norm on the colormap
            if not common_color_norm and pmax is not None:
                vmax = self._quantile_to_level(heights, pmax)
            if vmax is not None:
                artist_kws["vmax"] = vmax

            # Make cells at or below the threshold transparent
            if not common_color_norm and pthresh:
                thresh = self._quantile_to_level(heights, pthresh)
            if thresh is not None:
                heights = np.ma.masked_less_equal(heights, thresh)

            # Get the axes for this plot
            ax = self._get_axes(sub_vars)

            # pcolormesh is going to turn the grid off, but we want to keep it
            # I'm not sure if there's a better way to get the grid state
            x_grid = any([l.get_visible() for l in ax.xaxis.get_gridlines()])
            y_grid = any([l.get_visible() for l in ax.yaxis.get_gridlines()])

            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                heights.T,
                **artist_kws,
            )

            # pcolormesh sets sticky edges, but we only want them if not thresholding
            if thresh is not None:
                mesh.sticky_edges.x[:] = []
                mesh.sticky_edges.y[:] = []

            # Add an optional colorbar
            # Note, we want to improve this. When hue is used, it will stack
            # multiple colorbars with redundant ticks in an ugly way.
            # But it's going to take some work to have multiple colorbars that
            # share ticks nicely.
            if cbar:
                ax.figure.colorbar(mesh, cbar_ax, ax, **cbar_kws)

            # Reset the grid state
            if x_grid:
                ax.grid(True, axis="x")
            if y_grid:
                ax.grid(True, axis="y")

        # --- Finalize the plot

        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        self._add_axis_labels(ax)

        if "hue" in self.variables and legend:

            # TODO if possible, I would like to move the contour
            # intensity information into the legend too and label the
            # iso proportions rather than the raw density values

            artist_kws = {}
            artist = partial(mpl.patches.Patch)
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, True, False, "layer", 1, artist_kws, {},
            )
```
### 4 - seaborn/distributions.py:

Start line: 1468, End line: 1594

```python
histplot.__doc__ = """\
Plot univariate or bivariate histograms to show distributions of datasets.

A histogram is a classic visualization tool that represents the distribution
of one or more variables by counting the number of observations that fall within
discrete bins.

This function can normalize the statistic computed within each bin to estimate
frequency, density or probability mass, and it can add a smooth curve obtained
using a kernel density estimate, similar to :func:`kdeplot`.

More information is provided in the :ref:`user guide <tutorial_hist>`.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
weights : vector or key in ``data``
    If provided, weight the contribution of the corresponding data points
    towards the count in each bin by these factors.
{params.hist.stat}
{params.hist.bins}
{params.hist.binwidth}
{params.hist.binrange}
discrete : bool
    If True, default to ``binwidth=1`` and draw the bars so that they are
    centered on their corresponding data points. This avoids "gaps" that may
    otherwise appear when using discrete (integer) data.
cumulative : bool
    If True, plot the cumulative counts as bins increase.
common_bins : bool
    If True, use the same bins when semantic variables produce multiple
    plots. If using a reference rule to determine the bins, it will be computed
    with the full dataset.
common_norm : bool
    If True and using a normalized statistic, the normalization will apply over
    the full dataset. Otherwise, normalize each histogram independently.
multiple : {{"layer", "dodge", "stack", "fill"}}
    Approach to resolving multiple elements when semantic mapping creates subsets.
    Only relevant with univariate data.
element : {{"bars", "step", "poly"}}
    Visual representation of the histogram statistic.
    Only relevant with univariate data.
fill : bool
    If True, fill in the space under the histogram.
    Only relevant with univariate data.
shrink : number
    Scale the width of each bar relative to the binwidth by this factor.
    Only relevant with univariate data.
kde : bool
    If True, compute a kernel density estimate to smooth the distribution
    and show on the plot as (one or more) line(s).
    Only relevant with univariate data.
kde_kws : dict
    Parameters that control the KDE computation, as in :func:`kdeplot`.
line_kws : dict
    Parameters that control the KDE visualization, passed to
    :meth:`matplotlib.axes.Axes.plot`.
thresh : number or None
    Cells with a statistic less than or equal to this value will be transparent.
    Only relevant with bivariate data.
pthresh : number or None
    Like ``thresh``, but a value in [0, 1] such that cells with aggregate counts
    (or other statistics, when used) up to this proportion of the total will be
    transparent.
pmax : number or None
    A value in [0, 1] that sets that saturation point for the colormap at a value
    such that cells below constitute this proportion of the total count (or
    other statistic, when used).
{params.dist.cbar}
{params.dist.cbar_ax}
{params.dist.cbar_kws}
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.core.color}
{params.dist.log_scale}
{params.dist.legend}
{params.core.ax}
kwargs
    Other keyword arguments are passed to one of the following matplotlib
    functions:

    - :meth:`matplotlib.axes.Axes.bar` (univariate, element="bars")
    - :meth:`matplotlib.axes.Axes.fill_between` (univariate, other element, fill=True)
    - :meth:`matplotlib.axes.Axes.plot` (univariate, other element, fill=False)
    - :meth:`matplotlib.axes.Axes.pcolormesh` (bivariate)

Returns
-------
{returns.ax}

See Also
--------
{seealso.displot}
{seealso.kdeplot}
{seealso.rugplot}
{seealso.ecdfplot}
{seealso.jointplot}

Notes
-----

The choice of bins for computing and plotting a histogram can exert
substantial influence on the insights that one is able to draw from the
visualization. If the bins are too large, they may erase important features.
On the other hand, bins that are too small may be dominated by random
variability, obscuring the shape of the true underlying distribution. The
default bin size is determined using a reference rule that depends on the
sample size and variance. This works well in many cases, (i.e., with
"well-behaved" data) but it fails in others. It is always a good to try
different bin sizes to be sure that you are not missing something important.
This function allows you to specify bins in several different ways, such as
by setting the total number of bins to use, the width of each bin, or the
specific locations where the bins should break.

Examples
--------

.. include:: ../docstrings/histplot.rst

""".format(
    params=_param_docs,
    returns=_core_docs["returns"],
    seealso=_core_docs["seealso"],
)
```
### 5 - examples/histogram_stacked.py:

Start line: 1, End line: 30

```python
"""
Stacked histogram on a log scale
================================

_thumb: .5, .45

"""
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

diamonds = sns.load_dataset("diamonds")

f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)

sns.histplot(
    diamonds,
    x="price", hue="cut",
    multiple="stack",
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
)
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([500, 1000, 2000, 5000, 10000])
```
### 6 - seaborn/_marks/bar.py:

Start line: 133, End line: 173

```python
@document_properties
@dataclass
class Bar(BarBase):

    def _plot(self, split_gen, scales, orient):

        val_idx = ["y", "x"].index(orient)

        for _, data, ax in split_gen():

            bars, vals = self._make_patches(data, scales, orient)

            for bar in bars:

                # Because we are clipping the artist (see below), the edges end up
                # looking half as wide as they actually are. I don't love this clumsy
                # workaround, which is going to cause surprises if you work with the
                # artists directly. We may need to revisit after feedback.
                bar.set_linewidth(bar.get_linewidth() * 2)
                linestyle = bar.get_linestyle()
                if linestyle[1]:
                    linestyle = (linestyle[0], tuple(x / 2 for x in linestyle[1]))
                bar.set_linestyle(linestyle)

                # This is a bit of a hack to handle the fact that the edge lines are
                # centered on the actual extents of the bar, and overlap when bars are
                # stacked or dodged. We may discover that this causes problems and needs
                # to be revisited at some point. Also it should be faster to clip with
                # a bbox than a path, but I cant't work out how to get the intersection
                # with the axes bbox.
                bar.set_clip_path(bar.get_path(), bar.get_transform() + ax.transData)
                if self.artist_kws.get("clip_on", True):
                    # It seems the above hack undoes the default axes clipping
                    bar.set_clip_box(ax.bbox)
                bar.sticky_edges[val_idx][:] = (0, np.inf)
                ax.add_patch(bar)

            # Add a container which is useful for, e.g. Axes.bar_label
            if _version_predates(mpl, "3.4"):
                container_kws = {}
            else:
                orientation = {"x": "vertical", "y": "horizontal"}[orient]
                container_kws = dict(datavalues=vals, orientation=orientation)
            container = mpl.container.BarContainer(bars, **container_kws)
            ax.add_container(container)
```
### 7 - seaborn/_marks/bar.py:

Start line: 176, End line: 201

```python
@document_properties
@dataclass
class Bars(BarBase):
    """
    A faster bar mark with defaults more suitable histograms.

    See also
    --------
    Bar : A bar mark drawn between baseline and data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Bars.rst

    """
    color: MappableColor = Mappable("C0", grouping=False)
    alpha: MappableFloat = Mappable(.7, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(rc="patch.edgecolor", grouping=False)
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    edgewidth: MappableFloat = Mappable(auto=True, grouping=False)
    edgestyle: MappableStyle = Mappable("-", grouping=False)
    # pattern: MappableString = Mappable(None)  # TODO no Property yet

    width: MappableFloat = Mappable(1, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)  # TODO *is* this mappable?
```
### 8 - seaborn/distributions.py:

Start line: 379, End line: 551

```python
class _DistributionPlotter(VectorPlotter):

    def plot_univariate_histogram(
        self,
        multiple,
        element,
        fill,
        common_norm,
        common_bins,
        shrink,
        kde,
        kde_kws,
        color,
        legend,
        line_kws,
        estimate_kws,
        **plot_kws,
    ):

        # -- Default keyword dicts
        kde_kws = {} if kde_kws is None else kde_kws.copy()
        line_kws = {} if line_kws is None else line_kws.copy()
        estimate_kws = {} if estimate_kws is None else estimate_kws.copy()

        # --  Input checking
        _check_argument("multiple", ["layer", "stack", "fill", "dodge"], multiple)
        _check_argument("element", ["bars", "step", "poly"], element)

        auto_bins_with_weights = (
            "weights" in self.variables
            and estimate_kws["bins"] == "auto"
            and estimate_kws["binwidth"] is None
            and not estimate_kws["discrete"]
        )
        if auto_bins_with_weights:
            msg = (
                "`bins` cannot be 'auto' when using weights. "
                "Setting `bins=10`, but you will likely want to adjust."
            )
            warnings.warn(msg, UserWarning)
            estimate_kws["bins"] = 10

        # Simplify downstream code if we are not normalizing
        if estimate_kws["stat"] == "count":
            common_norm = False

        orient = self.data_variable

        # Now initialize the Histogram estimator
        estimator = Hist(**estimate_kws)
        histograms = {}

        # Do pre-compute housekeeping related to multiple groups
        all_data = self.comp_data.dropna()
        all_weights = all_data.get("weights", None)

        multiple_histograms = set(self.variables) - {"x", "y"}
        if multiple_histograms:
            if common_bins:
                bin_kws = estimator._define_bin_params(all_data, orient, None)
        else:
            common_norm = False

        if common_norm and all_weights is not None:
            whole_weight = all_weights.sum()
        else:
            whole_weight = len(all_data)

        # Estimate the smoothed kernel densities, for use later
        if kde:
            # TODO alternatively, clip at min/max bins?
            kde_kws.setdefault("cut", 0)
            kde_kws["cumulative"] = estimate_kws["cumulative"]
            log_scale = self._log_scaled(self.data_variable)
            densities = self._compute_univariate_density(
                self.data_variable,
                common_norm,
                common_bins,
                kde_kws,
                log_scale,
                warn_singular=False,
            )

        # First pass through the data to compute the histograms
        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            # Prepare the relevant data
            key = tuple(sub_vars.items())
            orient = self.data_variable

            if "weights" in self.variables:
                sub_data["weight"] = sub_data.pop("weights")
                part_weight = sub_data["weight"].sum()
            else:
                part_weight = len(sub_data)

            # Do the histogram computation
            if not (multiple_histograms and common_bins):
                bin_kws = estimator._define_bin_params(sub_data, orient, None)
            res = estimator._normalize(estimator._eval(sub_data, orient, bin_kws))
            heights = res[estimator.stat].to_numpy()
            widths = res["space"].to_numpy()
            edges = res[orient].to_numpy() - widths / 2

            # Rescale the smoothed curve to match the histogram
            if kde and key in densities:
                density = densities[key]
                if estimator.cumulative:
                    hist_norm = heights.max()
                else:
                    hist_norm = (heights * widths).sum()
                densities[key] *= hist_norm

            # Convert edges back to original units for plotting
            if self._log_scaled(self.data_variable):
                widths = np.power(10, edges + widths) - np.power(10, edges)
                edges = np.power(10, edges)

            # Pack the histogram data and metadata together
            edges = edges + (1 - shrink) / 2 * widths
            widths *= shrink
            index = pd.MultiIndex.from_arrays([
                pd.Index(edges, name="edges"),
                pd.Index(widths, name="widths"),
            ])
            hist = pd.Series(heights, index=index, name="heights")

            # Apply scaling to normalize across groups
            if common_norm:
                hist *= part_weight / whole_weight

            # Store the finalized histogram data for future plotting
            histograms[key] = hist

        # Modify the histogram and density data to resolve multiple groups
        histograms, baselines = self._resolve_multiple(histograms, multiple)
        if kde:
            densities, _ = self._resolve_multiple(
                densities, None if multiple == "dodge" else multiple
            )

        # Set autoscaling-related meta
        sticky_stat = (0, 1) if multiple == "fill" else (0, np.inf)
        if multiple == "fill":
            # Filled plots should not have any margins
            bin_vals = histograms.index.to_frame()
            edges = bin_vals["edges"]
            widths = bin_vals["widths"]
            sticky_data = (
                edges.min(),
                edges.max() + widths.loc[edges.idxmax()]
            )
        else:
            sticky_data = []

        # --- Handle default visual attributes

        # Note: default linewidth is determined after plotting

        # Default alpha should depend on other parameters
        if fill:
            # Note: will need to account for other grouping semantics if added
            if "hue" in self.variables and multiple == "layer":
                default_alpha = .5 if element == "bars" else .25
            elif kde:
                default_alpha = .5
            else:
                default_alpha = .75
        else:
            default_alpha = 1
        alpha = plot_kws.pop("alpha", default_alpha)  # TODO make parameter?

        hist_artists = []

        # Go back through the dataset and draw the plots
        # ... other code
```
### 9 - seaborn/categorical.py:

Start line: 1425, End line: 1492

```python
class _CategoricalStatPlotter(_CategoricalPlotter):

    require_numeric = True

    @property
    def nested_width(self):
        """A float with the width of plot elements when hue nesting is used."""
        if self.dodge:
            width = self.width / len(self.hue_names)
        else:
            width = self.width
        return width

    def estimate_statistic(self, estimator, errorbar, n_boot, seed):

        if self.hue_names is None:
            statistic = []
            confint = []
        else:
            statistic = [[] for _ in self.plot_data]
            confint = [[] for _ in self.plot_data]

        var = {"v": "y", "h": "x"}[self.orient]

        agg = EstimateAggregator(estimator, errorbar, n_boot=n_boot, seed=seed)

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single layer of grouping
            # --------------------------------------------
            if self.plot_hues is None:

                df = pd.DataFrame({var: group_data})
                if self.plot_units is not None:
                    df["units"] = self.plot_units[i]

                res = agg(df, var)

                statistic.append(res[var])
                if errorbar is not None:
                    confint.append((res[f"{var}min"], res[f"{var}max"]))

            # Option 2: we are grouping by a hue layer
            # ----------------------------------------

            else:
                for hue_level in self.hue_names:

                    if not self.plot_hues[i].size:
                        statistic[i].append(np.nan)
                        if errorbar is not None:
                            confint[i].append((np.nan, np.nan))
                        continue

                    hue_mask = self.plot_hues[i] == hue_level
                    df = pd.DataFrame({var: group_data[hue_mask]})
                    if self.plot_units is not None:
                        df["units"] = self.plot_units[i][hue_mask]

                    res = agg(df, var)

                    statistic[i].append(res[var])
                    if errorbar is not None:
                        confint[i].append((res[f"{var}min"], res[f"{var}max"]))

        # Save the resulting values for plotting
        self.statistic = np.array(statistic)
        self.confint = np.array(confint)
```
### 10 - examples/part_whole_bars.py:

Start line: 1, End line: 31

```python
"""
Horizontal bar plots
====================

"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)
```
### 16 - seaborn/_marks/bar.py:

Start line: 28, End line: 73

```python
class BarBase(Mark):

    def _make_patches(self, data, scales, orient):

        kws = self._resolve_properties(data, scales)
        if orient == "x":
            kws["x"] = (data["x"] - data["width"] / 2).to_numpy()
            kws["y"] = data["baseline"].to_numpy()
            kws["w"] = data["width"].to_numpy()
            kws["h"] = (data["y"] - data["baseline"]).to_numpy()
        else:
            kws["x"] = data["baseline"].to_numpy()
            kws["y"] = (data["y"] - data["width"] / 2).to_numpy()
            kws["w"] = (data["x"] - data["baseline"]).to_numpy()
            kws["h"] = data["width"].to_numpy()

        kws.pop("width", None)
        kws.pop("baseline", None)

        val_dim = {"x": "h", "y": "w"}[orient]
        bars, vals = [], []

        for i in range(len(data)):

            row = {k: v[i] for k, v in kws.items()}

            # Skip bars with no value. It's possible we'll want to make this
            # an option (i.e so you have an artist for animating or annotating),
            # but let's keep things simple for now.
            if not np.nan_to_num(row[val_dim]):
                continue

            bar = mpl.patches.Rectangle(
                xy=(row["x"], row["y"]),
                width=row["w"],
                height=row["h"],
                facecolor=row["facecolor"],
                edgecolor=row["edgecolor"],
                linestyle=row["edgestyle"],
                linewidth=row["edgewidth"],
                **self.artist_kws,
            )
            bars.append(bar)
            vals.append(row[val_dim])

        return bars, vals
```
### 22 - seaborn/_marks/bar.py:

Start line: 106, End line: 131

```python
@document_properties
@dataclass
class Bar(BarBase):
    """
    A bar mark drawn between baseline and data values.

    See also
    --------
    Bars : A faster bar mark with defaults more suitable for histograms.

    Examples
    --------
    .. include:: ../docstrings/objects.Bar.rst

    """
    color: MappableColor = Mappable("C0", grouping=False)
    alpha: MappableFloat = Mappable(.7, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(depend="color", grouping=False)
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth", grouping=False)
    edgestyle: MappableStyle = Mappable("-", grouping=False)
    # pattern: MappableString = Mappable(None)  # TODO no Property yet

    width: MappableFloat = Mappable(.8, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)  # TODO *is* this mappable?
```
### 51 - seaborn/_core/plot.py:

Start line: 1171, End line: 1293

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
                    variables.extend(str(v) for v in df if v not in variables)
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
            if axis is not None and _version_predates(mpl, "3.4"):
                paired_axis = axis in p._pair_spec.get("structure", {})
                cat_scale = isinstance(scale, Nominal)
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            if scale is None:
                self._scales[var] = Scale._identity()
            else:
                try:
                    self._scales[var] = scale._setup(var_df[var], prop)
                except Exception as err:
                    raise PlotSpecError._during("Scale setup", var) from err

            if axis is None or (var != coord and coord in p._variables):
                # Everything below here applies only to coordinate variables
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
                    if var not in layer_df:
                        continue

                    idx = self._get_subplot_index(layer_df, view)
                    try:
                        new_series.loc[idx] = view_scale(layer_df.loc[idx, var])
                    except Exception as err:
                        spec_error = PlotSpecError._during("Scaling operation", var)
                        raise spec_error from err

            # Now the transformed data series are complete, set update the layer data
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer["data"].frame
                if var in layer_df:
                    layer_df[var] = new_series
```
### 71 - seaborn/_core/plot.py:

Start line: 1080, End line: 1132

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
                        drop_cols = [x for x in df if re.match(rf"{axis}\d+", str(x))]
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
### 79 - seaborn/_marks/bar.py:

Start line: 91, End line: 103

```python
class BarBase(Mark):

    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    ) -> Artist:
        # TODO return some sensible default?
        key = {v: value for v in variables}
        key = self._resolve_properties(key, scales)
        artist = mpl.patches.Patch(
            facecolor=key["facecolor"],
            edgecolor=key["edgecolor"],
            linewidth=key["edgewidth"],
            linestyle=key["edgestyle"],
        )
        return artist
```
### 81 - seaborn/_marks/bar.py:

Start line: 75, End line: 89

```python
class BarBase(Mark):

    def _resolve_properties(self, data, scales):

        resolved = resolve_properties(self, data, scales)

        resolved["facecolor"] = resolve_color(self, data, "", scales)
        resolved["edgecolor"] = resolve_color(self, data, "edge", scales)

        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved
```
### 87 - seaborn/_core/plot.py:

Start line: 1375, End line: 1405

```python
class Plotter:

    def _unscale_coords(
        self, subplots: list[dict], df: DataFrame, orient: str,
    ) -> DataFrame:
        # TODO do we still have numbers in the variable name at this point?
        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", str(c))]
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

                axis = getattr(view["ax"], f"{str(var)[0]}axis")
                # TODO see https://github.com/matplotlib/matplotlib/issues/22713
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, str(var)] = inverted

                if var == orient and "width" in view_df:
                    width = view_df["width"]
                    out_df.loc[values.index, "width"] = (
                        transform(values + width / 2) - transform(values - width / 2)
                    )

        return out_df
```
### 118 - seaborn/_core/plot.py:

Start line: 980, End line: 1078

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
                # axis / tick labels can be shown on interior shared axes if desired

                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or not p._pair_spec.get("cross", True)
                    or (
                        axis in p._pair_spec.get("structure", {})
                        and bool(p._pair_spec.get("wrap"))
                    )
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
### 124 - seaborn/_core/plot.py:

Start line: 603, End line: 630

```python
@build_plot_signature
class Plot:

    # TODO def twin()?

    def scale(self, **scales: Scale) -> Plot:
        """
        Specify mappings from data units to visual properties.

        Keywords correspond to variables defined in the plot, including coordinate
        variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

        A number of "magic" arguments are accepted, including:
            - The name of a transform (e.g., `"log"`, `"sqrt"`)
            - The name of a palette (e.g., `"viridis"`, `"muted"`)
            - A tuple of values, defining the output range (e.g. `(1, 5)`)
            - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
            - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

        For more explicit control, pass a scale spec object such as :class:`Continuous`
        or :class:`Nominal`. Or pass `None` to use an "identity" scale, which treats
        data values as literally encoding visual properties.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.scale.rst

        """
        new = self._clone()
        new._scales.update(scales)
        return new
```
### 125 - seaborn/_core/plot.py:

Start line: 1295, End line: 1373

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

            if orient in df:
                width = pd.Series(index=df.index, dtype=float)
                for view in subplots:
                    view_idx = self._get_subplot_data(
                        df, orient, view, p._shares.get(orient)
                    ).index
                    view_df = df.loc[view_idx]
                    if "width" in mark._mappable_props:
                        view_width = mark._resolve(view_df, "width", None)
                    elif "width" in df:
                        view_width = view_df["width"]
                    else:
                        view_width = 0.8  # TODO what default?
                    spacing = scales[orient]._spacing(view_df.loc[view_idx, orient])
                    width.loc[view_idx] = view_width * spacing
                df["width"] = width

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
                baseline = 0 if "baseline" not in df else df["baseline"]
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
### 143 - seaborn/_core/plot.py:

Start line: 910, End line: 947

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
### 153 - seaborn/_core/plot.py:

Start line: 1134, End line: 1169

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
### 169 - seaborn/_core/plot.py:

Start line: 632, End line: 649

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

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.share.rst

        """
        new = self._clone()
        new._shares.update(shares)
        return new
```
### 170 - seaborn/_core/plot.py:

Start line: 1471, End line: 1543

```python
class Plotter:

    def _setup_split_generator(
        self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

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

                with pd.option_context("mode.use_inf_as_na", True):
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
                    if not axes_df.empty:
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

                    if df_subset.empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    # TODO need copy(deep=...) policy (here, above, anywhere else?)
                    yield sub_vars, df_subset.copy(), view["ax"]

        return split_generator
```
### 188 - seaborn/_core/plot.py:

Start line: 329, End line: 342

```python
@build_plot_signature
class Plot:

    @property
    def _variables(self) -> list[str]:

        variables = (
            list(self._data.frame)
            + list(self._pair_spec.get("variables", []))
            + list(self._facet_spec.get("variables", []))
        )
        for layer in self._layers:
            variables.extend(v for v in layer["vars"] if v not in variables)

        # Coerce to str in return to appease mypy; we know these will only
        # ever be strings but I don't think we can type a DataFrame that way yet
        return [str(v) for v in variables]
```
### 199 - seaborn/_core/plot.py:

Start line: 823, End line: 857

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
### 218 - seaborn/_marks/bar.py:

Start line: 1, End line: 25

```python
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableColor,
    MappableFloat,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties
)
from seaborn.utils import _version_predates

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from matplotlib.artist import Artist
    from seaborn._core.scales import Scale
```
### 226 - seaborn/_core/plot.py:

Start line: 651, End line: 670

```python
@build_plot_signature
class Plot:

    def limit(self, **limits: tuple[Any, Any]) -> Plot:
        """
        Control the range of visible data.

        Keywords correspond to variables defined in the plot, and values are a
        `(min, max)` tuple (where either can be `None` to leave unset).

        Limits apply only to the axis; data outside the visible range are
        still used for any stat transforms and added to the plot.

        Behavior for non-coordinate variables is currently undefined.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.limit.rst

        """
        new = self._clone()
        new._limits.update(limits)
        return new
```
