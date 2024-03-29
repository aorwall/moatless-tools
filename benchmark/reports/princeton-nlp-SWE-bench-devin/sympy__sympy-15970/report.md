# sympy__sympy-15970

| **sympy/sympy** | `c267d554e16f0392af2b22a2922cbe0db7e8c798` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 41335 |
| **Any found context length** | 22010 |
| **Avg pos** | 256.0 |
| **Min pos** | 20 |
| **Max pos** | 71 |
| **Top file pos** | 3 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1673,7 +1673,7 @@ def _print_TensorIndex(self, expr):
 
     def _print_tuple(self, expr):
         return r"\left( %s\right)" % \
-            r", \quad ".join([ self._print(i) for i in expr ])
+            r", \  ".join([ self._print(i) for i in expr ])
 
     def _print_TensorProduct(self, expr):
         elements = [self._print(a) for a in expr.args]
@@ -1688,7 +1688,7 @@ def _print_Tuple(self, expr):
 
     def _print_list(self, expr):
         return r"\left[ %s\right]" % \
-            r", \quad ".join([ self._print(i) for i in expr ])
+            r", \  ".join([ self._print(i) for i in expr ])
 
     def _print_dict(self, d):
         keys = sorted(d.keys(), key=default_sort_key)
@@ -1698,7 +1698,7 @@ def _print_dict(self, d):
             val = d[key]
             items.append("%s : %s" % (self._print(key), self._print(val)))
 
-        return r"\left\{ %s\right\}" % r", \quad ".join(items)
+        return r"\left\{ %s\right\}" % r", \  ".join(items)
 
     def _print_Dict(self, expr):
         return self._print_dict(expr)
@@ -2450,7 +2450,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
     dictionary.
 
     >>> print(latex([2/x, y], mode='inline'))
-    $\left[ 2 / x, \quad y\right]$
+    $\left[ 2 / x, \  y\right]$
 
     """
     if symbol_names is None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 1676 | 1676 | 71 | 3 | 41335
| sympy/printing/latex.py | 1691 | 1691 | 71 | 3 | 41335
| sympy/printing/latex.py | 1701 | 1701 | 71 | 3 | 41335
| sympy/printing/latex.py | 2453 | 2453 | 23 | 3 | 24680


## Problem Statement

```
Use '\ ' instead of '\quad' for latex of lists, tuples, and dicts
See [this](https://twitter.com/asmeurer/status/487982939536248833) Twitter
discussion.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/parsing/latex/_antlr/latexparser.py | 54 | 85| 1527 | 1527 | 30275 | 
| 2 | 2 sympy/parsing/latex/_antlr/latexlexer.py | 2 | 53| 1563 | 3090 | 41523 | 
| 3 | 2 sympy/parsing/latex/_antlr/latexlexer.py | 54 | 84| 1509 | 4599 | 41523 | 
| 4 | **3 sympy/printing/latex.py** | 82 | 117| 491 | 5090 | 66016 | 
| 5 | 3 sympy/parsing/latex/_antlr/latexlexer.py | 117 | 148| 1517 | 6607 | 66016 | 
| 6 | 3 sympy/parsing/latex/_antlr/latexparser.py | 86 | 119| 1530 | 8137 | 66016 | 
| 7 | 3 sympy/parsing/latex/_antlr/latexparser.py | 154 | 190| 1548 | 9685 | 66016 | 
| 8 | 3 sympy/parsing/latex/_antlr/latexlexer.py | 85 | 116| 1515 | 11200 | 66016 | 
| 9 | 3 sympy/parsing/latex/_antlr/latexlexer.py | 183 | 216| 1538 | 12738 | 66016 | 
| 10 | 3 sympy/parsing/latex/_antlr/latexparser.py | 120 | 153| 1519 | 14257 | 66016 | 
| 11 | 3 sympy/parsing/latex/_antlr/latexlexer.py | 149 | 182| 1522 | 15779 | 66016 | 
| 12 | 4 sympy/printing/pretty/pretty_symbology.py | 189 | 229| 743 | 16522 | 71608 | 
| 13 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2 | 53| 1560 | 18082 | 71608 | 
| 14 | 4 sympy/parsing/latex/_antlr/latexparser.py | 191 | 219| 1164 | 19246 | 71608 | 
| 15 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2885 | 2905| 176 | 19422 | 71608 | 
| 16 | 4 sympy/parsing/latex/_antlr/latexlexer.py | 233 | 310| 787 | 20209 | 71608 | 
| 17 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2838 | 2858| 177 | 20386 | 71608 | 
| 18 | 4 sympy/parsing/latex/_antlr/latexparser.py | 222 | 268| 778 | 21164 | 71608 | 
| 19 | **4 sympy/printing/latex.py** | 1807 | 1869| 542 | 21706 | 71608 | 
| **-> 20 <-** | **4 sympy/printing/latex.py** | 2255 | 2477| 304 | 22010 | 71608 | 
| 21 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2860 | 2880| 156 | 22166 | 71608 | 
| 22 | 4 sympy/printing/pretty/pretty_symbology.py | 279 | 308| 230 | 22396 | 71608 | 
| **-> 23 <-** | **4 sympy/printing/latex.py** | 2290 | 2455| 2284 | 24680 | 71608 | 
| 24 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2813 | 2833| 157 | 24837 | 71608 | 
| 25 | 5 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 59| 471 | 25308 | 76000 | 
| 26 | 5 sympy/parsing/latex/_antlr/latexparser.py | 996 | 1021| 225 | 25533 | 76000 | 
| 27 | 5 sympy/parsing/latex/_antlr/latexparser.py | 1998 | 2024| 228 | 25761 | 76000 | 
| 28 | 5 sympy/parsing/latex/_antlr/latexparser.py | 269 | 382| 947 | 26708 | 76000 | 
| 29 | 5 sympy/parsing/latex/_parse_latex_antlr.py | 522 | 555| 222 | 26930 | 76000 | 
| 30 | 5 sympy/printing/pretty/pretty_symbology.py | 230 | 277| 719 | 27649 | 76000 | 
| 31 | 5 sympy/parsing/latex/_parse_latex_antlr.py | 283 | 326| 372 | 28021 | 76000 | 
| 32 | 5 sympy/parsing/latex/_antlr/latexparser.py | 1767 | 1785| 154 | 28175 | 76000 | 
| 33 | **5 sympy/printing/latex.py** | 1871 | 1882| 133 | 28308 | 76000 | 
| 34 | **5 sympy/printing/latex.py** | 120 | 202| 676 | 28984 | 76000 | 
| 35 | **5 sympy/printing/latex.py** | 922 | 977| 577 | 29561 | 76000 | 
| 36 | 5 sympy/parsing/latex/_antlr/latexlexer.py | 312 | 348| 775 | 30336 | 76000 | 
| 37 | 6 sympy/physics/vector/dyadic.py | 376 | 399| 243 | 30579 | 80643 | 
| 38 | 6 sympy/parsing/latex/_antlr/latexparser.py | 2098 | 2118| 422 | 31001 | 80643 | 
| 39 | **6 sympy/printing/latex.py** | 1392 | 1410| 135 | 31136 | 80643 | 
| 40 | 6 sympy/parsing/latex/_antlr/latexparser.py | 1937 | 1960| 205 | 31341 | 80643 | 
| 41 | 6 sympy/parsing/latex/_antlr/latexparser.py | 1046 | 1071| 237 | 31578 | 80643 | 
| 42 | **6 sympy/printing/latex.py** | 2144 | 2208| 760 | 32338 | 80643 | 
| 43 | 6 sympy/parsing/latex/_antlr/latexparser.py | 843 | 890| 616 | 32954 | 80643 | 
| 44 | 6 sympy/printing/pretty/pretty_symbology.py | 107 | 187| 758 | 33712 | 80643 | 
| 45 | 6 sympy/parsing/latex/_antlr/latexparser.py | 2227 | 2439| 257 | 33969 | 80643 | 
| 46 | 6 sympy/parsing/latex/_antlr/latexparser.py | 510 | 528| 147 | 34116 | 80643 | 
| 47 | 6 sympy/parsing/latex/_parse_latex_antlr.py | 62 | 88| 188 | 34304 | 80643 | 
| 48 | 7 sympy/parsing/latex/_antlr/__init__.py | 2 | 13| 39 | 34343 | 80755 | 
| 49 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1626 | 1663| 328 | 34671 | 80755 | 
| 50 | **7 sympy/printing/latex.py** | 672 | 702| 323 | 34994 | 80755 | 
| 51 | **7 sympy/printing/latex.py** | 1 | 81| 720 | 35714 | 80755 | 
| 52 | **7 sympy/printing/latex.py** | 1364 | 1390| 210 | 35924 | 80755 | 
| 53 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1699 | 1743| 351 | 36275 | 80755 | 
| 54 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1819 | 1870| 424 | 36699 | 80755 | 
| 55 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1551 | 1596| 296 | 36995 | 80755 | 
| 56 | **7 sympy/printing/latex.py** | 907 | 920| 139 | 37134 | 80755 | 
| 57 | 8 sympy/core/basic.py | 417 | 430| 135 | 37269 | 96308 | 
| 58 | 8 sympy/parsing/latex/_antlr/latexparser.py | 2551 | 2600| 428 | 37697 | 96308 | 
| 59 | 8 sympy/parsing/latex/_antlr/latexparser.py | 2502 | 2546| 326 | 38023 | 96308 | 
| 60 | **8 sympy/printing/latex.py** | 1412 | 1423| 171 | 38194 | 96308 | 
| 61 | **8 sympy/printing/latex.py** | 663 | 670| 127 | 38321 | 96308 | 
| 62 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1872 | 1892| 160 | 38481 | 96308 | 
| 63 | **8 sympy/printing/latex.py** | 1425 | 1450| 271 | 38752 | 96308 | 
| 64 | **8 sympy/printing/latex.py** | 385 | 407| 268 | 39020 | 96308 | 
| 65 | **8 sympy/printing/latex.py** | 1706 | 1714| 135 | 39155 | 96308 | 
| 66 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1181 | 1205| 179 | 39334 | 96308 | 
| 67 | **8 sympy/printing/latex.py** | 1884 | 1931| 457 | 39791 | 96308 | 
| 68 | **8 sympy/printing/latex.py** | 509 | 559| 595 | 40386 | 96308 | 
| 69 | **8 sympy/printing/latex.py** | 1451 | 1474| 218 | 40604 | 96308 | 
| 70 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1519 | 1546| 170 | 40774 | 96308 | 
| **-> 71 <-** | **8 sympy/printing/latex.py** | 1639 | 1704| 561 | 41335 | 96308 | 
| 72 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1897 | 1932| 288 | 41623 | 96308 | 
| 73 | 8 sympy/parsing/latex/_antlr/latexparser.py | 638 | 674| 259 | 41882 | 96308 | 
| 74 | 8 sympy/parsing/latex/_parse_latex_antlr.py | 329 | 370| 451 | 42333 | 96308 | 
| 75 | 8 sympy/printing/pretty/pretty_symbology.py | 75 | 104| 167 | 42500 | 96308 | 
| 76 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1665 | 1694| 218 | 42718 | 96308 | 
| 77 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1745 | 1762| 131 | 42849 | 96308 | 
| 78 | **8 sympy/printing/latex.py** | 896 | 905| 136 | 42985 | 96308 | 
| 79 | **8 sympy/printing/latex.py** | 718 | 733| 161 | 43146 | 96308 | 
| 80 | 8 sympy/parsing/latex/_antlr/latexparser.py | 1307 | 1343| 234 | 43380 | 96308 | 
| 81 | 9 sympy/parsing/latex/errors.py | 1 | 3| 0 | 43380 | 96319 | 
| 82 | 9 sympy/parsing/latex/_antlr/latexparser.py | 1244 | 1268| 179 | 43559 | 96319 | 
| 83 | **9 sympy/printing/latex.py** | 1348 | 1362| 147 | 43706 | 96319 | 
| 84 | 9 sympy/parsing/latex/_antlr/latexparser.py | 678 | 724| 542 | 44248 | 96319 | 
| 85 | **9 sympy/printing/latex.py** | 1558 | 1605| 489 | 44737 | 96319 | 
| 86 | 9 sympy/parsing/latex/_antlr/latexparser.py | 1210 | 1242| 246 | 44983 | 96319 | 
| 87 | **9 sympy/printing/latex.py** | 821 | 894| 680 | 45663 | 96319 | 
| 88 | 9 sympy/parsing/latex/_antlr/latexparser.py | 1962 | 1993| 246 | 45909 | 96319 | 
| 89 | **9 sympy/printing/latex.py** | 704 | 716| 159 | 46068 | 96319 | 
| 90 | 9 sympy/parsing/latex/_antlr/latexparser.py | 387 | 436| 345 | 46413 | 96319 | 
| 91 | 9 sympy/parsing/latex/_antlr/latexparser.py | 925 | 971| 642 | 47055 | 96319 | 
| 92 | **9 sympy/printing/latex.py** | 2126 | 2142| 135 | 47190 | 96319 | 
| 93 | 9 sympy/parsing/latex/_antlr/latexparser.py | 726 | 762| 283 | 47473 | 96319 | 
| 94 | 10 sympy/parsing/mathematica.py | 370 | 390| 181 | 47654 | 99217 | 
| 95 | 10 sympy/parsing/latex/_antlr/latexparser.py | 1073 | 1119| 324 | 47978 | 99217 | 
| 96 | 10 sympy/parsing/latex/_antlr/latexparser.py | 1273 | 1305| 246 | 48224 | 99217 | 
| 97 | 10 sympy/parsing/latex/_antlr/latexlexer.py | 217 | 230| 597 | 48821 | 99217 | 
| 98 | 10 sympy/parsing/latex/_antlr/latexparser.py | 973 | 991| 139 | 48960 | 99217 | 
| 99 | 10 sympy/parsing/latex/_antlr/latexparser.py | 2779 | 2811| 277 | 49237 | 99217 | 
| 100 | 11 sympy/physics/quantum/qasm.py | 104 | 136| 300 | 49537 | 100973 | 
| 101 | **11 sympy/printing/latex.py** | 1792 | 1805| 143 | 49680 | 100973 | 
| 102 | 11 sympy/parsing/latex/_antlr/latexparser.py | 488 | 505| 132 | 49812 | 100973 | 
| 103 | **11 sympy/printing/latex.py** | 1139 | 1204| 672 | 50484 | 100973 | 
| 104 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2120 | 2222| 682 | 51166 | 100973 | 
| 105 | **11 sympy/printing/latex.py** | 1739 | 1775| 380 | 51546 | 100973 | 
| 106 | **11 sympy/printing/latex.py** | 1777 | 1790| 134 | 51680 | 100973 | 
| 107 | 11 sympy/parsing/latex/_antlr/latexparser.py | 1144 | 1179| 246 | 51926 | 100973 | 
| 108 | **11 sympy/printing/latex.py** | 1286 | 1346| 753 | 52679 | 100973 | 
| 109 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2449 | 2500| 323 | 53002 | 100973 | 
| 110 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2026 | 2093| 549 | 53551 | 100973 | 
| 111 | 11 sympy/parsing/latex/_antlr/latexparser.py | 1787 | 1814| 193 | 53744 | 100973 | 
| 112 | **11 sympy/printing/latex.py** | 1230 | 1284| 730 | 54474 | 100973 | 
| 113 | 11 sympy/parsing/latex/_antlr/latexparser.py | 440 | 486| 529 | 55003 | 100973 | 
| 114 | 11 sympy/parsing/latex/_parse_latex_antlr.py | 91 | 106| 127 | 55130 | 100973 | 
| 115 | **11 sympy/printing/latex.py** | 2034 | 2077| 771 | 55901 | 100973 | 
| 116 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2716 | 2748| 277 | 56178 | 100973 | 
| 117 | **11 sympy/printing/latex.py** | 1206 | 1215| 138 | 56316 | 100973 | 
| 118 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2687 | 2711| 175 | 56491 | 100973 | 
| 119 | **11 sympy/printing/latex.py** | 1217 | 1228| 187 | 56678 | 100973 | 
| 120 | 11 sympy/parsing/latex/_antlr/latexparser.py | 814 | 838| 172 | 56850 | 100973 | 
| 121 | **11 sympy/printing/latex.py** | 735 | 802| 658 | 57508 | 100973 | 
| 122 | **11 sympy/printing/latex.py** | 561 | 581| 221 | 57729 | 100973 | 
| 123 | 12 sympy/printing/mathml.py | 635 | 669| 302 | 58031 | 108329 | 
| 124 | **12 sympy/printing/latex.py** | 583 | 600| 197 | 58228 | 108329 | 
| 125 | 12 sympy/parsing/latex/_antlr/latexparser.py | 1598 | 1621| 159 | 58387 | 108329 | 
| 126 | 12 sympy/parsing/latex/_antlr/latexparser.py | 2655 | 2685| 238 | 58625 | 108329 | 
| 127 | 12 sympy/parsing/latex/_antlr/latexparser.py | 2750 | 2774| 175 | 58800 | 108329 | 
| 128 | **12 sympy/printing/latex.py** | 344 | 360| 165 | 58965 | 108329 | 
| 129 | **12 sympy/printing/latex.py** | 1519 | 1524| 116 | 59081 | 108329 | 
| 130 | 13 sympy/utilities/misc.py | 31 | 106| 544 | 59625 | 111632 | 
| 131 | **13 sympy/printing/latex.py** | 979 | 988| 126 | 59751 | 111632 | 
| 132 | 13 sympy/parsing/latex/_antlr/latexparser.py | 2602 | 2653| 332 | 60083 | 111632 | 
| 133 | **13 sympy/printing/latex.py** | 2232 | 2240| 129 | 60212 | 111632 | 
| 134 | **13 sympy/printing/latex.py** | 460 | 507| 529 | 60741 | 111632 | 
| 135 | 14 sympy/printing/pretty/pretty.py | 2013 | 2063| 366 | 61107 | 133497 | 
| 136 | 14 sympy/parsing/latex/_antlr/latexparser.py | 1347 | 1411| 607 | 61714 | 133497 | 
| 137 | **14 sympy/printing/latex.py** | 1607 | 1637| 279 | 61993 | 133497 | 
| 138 | 14 sympy/parsing/latex/_parse_latex_antlr.py | 373 | 457| 736 | 62729 | 133497 | 
| 139 | 14 sympy/parsing/latex/_antlr/latexparser.py | 530 | 560| 200 | 62929 | 133497 | 
| 140 | 14 sympy/parsing/latex/_parse_latex_antlr.py | 126 | 146| 200 | 63129 | 133497 | 
| 141 | 14 sympy/parsing/latex/_antlr/latexparser.py | 766 | 812| 563 | 63692 | 133497 | 
| 142 | 15 sympy/parsing/latex/_build_latex_antlr.py | 1 | 37| 269 | 63961 | 134121 | 
| 143 | 16 sympy/abc.py | 1 | 67| 801 | 64762 | 135346 | 
| 144 | **16 sympy/printing/latex.py** | 632 | 661| 272 | 65034 | 135346 | 
| 145 | 16 sympy/parsing/latex/_antlr/latexparser.py | 1121 | 1139| 139 | 65173 | 135346 | 
| 146 | **16 sympy/printing/latex.py** | 1003 | 1087| 809 | 65982 | 135346 | 
| 147 | **16 sympy/printing/latex.py** | 2079 | 2124| 407 | 66389 | 135346 | 
| 148 | 16 sympy/printing/pretty/pretty.py | 1 | 29| 275 | 66664 | 135346 | 
| 149 | **16 sympy/printing/latex.py** | 1728 | 1737| 131 | 66795 | 135346 | 
| 150 | 16 sympy/physics/quantum/qasm.py | 1 | 102| 721 | 67516 | 135346 | 
| 151 | 17 doc/src/conf.py | 102 | 207| 777 | 68293 | 137065 | 
| 152 | 17 sympy/parsing/latex/_antlr/latexparser.py | 1413 | 1449| 255 | 68548 | 137065 | 
| 153 | **17 sympy/printing/latex.py** | 2456 | 2484| 306 | 68854 | 137065 | 
| 154 | 17 sympy/parsing/latex/_antlr/latexparser.py | 2909 | 2922| 182 | 69036 | 137065 | 
| 155 | **17 sympy/printing/latex.py** | 1716 | 1726| 141 | 69177 | 137065 | 
| 156 | 17 sympy/parsing/latex/_antlr/latexparser.py | 2237 | 2447| 1899 | 71076 | 137065 | 
| 157 | **17 sympy/printing/latex.py** | 2242 | 2252| 159 | 71235 | 137065 | 
| 158 | 17 sympy/printing/pretty/pretty.py | 1820 | 1873| 388 | 71623 | 137065 | 
| 159 | 17 sympy/printing/mathml.py | 257 | 297| 345 | 71968 | 137065 | 
| 160 | 18 sympy/physics/vector/printing.py | 339 | 376| 339 | 72307 | 140425 | 
| 161 | **18 sympy/printing/latex.py** | 322 | 342| 148 | 72455 | 140425 | 
| 162 | 18 sympy/parsing/latex/_parse_latex_antlr.py | 169 | 200| 334 | 72789 | 140425 | 
| 163 | 18 sympy/parsing/latex/_antlr/latexparser.py | 892 | 920| 217 | 73006 | 140425 | 
| 164 | 19 sympy/printing/rust.py | 164 | 215| 213 | 73219 | 145862 | 
| 165 | **19 sympy/printing/latex.py** | 409 | 458| 383 | 73602 | 145862 | 
| 166 | 19 sympy/parsing/latex/_parse_latex_antlr.py | 203 | 215| 119 | 73721 | 145862 | 
| 167 | **19 sympy/printing/latex.py** | 2025 | 2032| 118 | 73839 | 145862 | 
| 168 | **19 sympy/printing/latex.py** | 1476 | 1497| 203 | 74042 | 145862 | 
| 169 | 19 sympy/parsing/latex/_antlr/latexparser.py | 1023 | 1041| 154 | 74196 | 145862 | 
| 170 | 20 sympy/parsing/latex/__init__.py | 1 | 35| 279 | 74475 | 146141 | 
| 171 | 20 sympy/printing/pretty/pretty.py | 1890 | 1915| 223 | 74698 | 146141 | 
| 172 | **20 sympy/printing/latex.py** | 1933 | 1988| 414 | 75112 | 146141 | 
| 173 | 20 sympy/parsing/latex/_parse_latex_antlr.py | 149 | 166| 143 | 75255 | 146141 | 
| 174 | **20 sympy/printing/latex.py** | 2222 | 2230| 123 | 75378 | 146141 | 
| 175 | 20 sympy/parsing/latex/_antlr/latexparser.py | 590 | 636| 479 | 75857 | 146141 | 
| 176 | **20 sympy/printing/latex.py** | 990 | 1001| 157 | 76014 | 146141 | 
| 177 | 20 sympy/physics/vector/dyadic.py | 155 | 190| 401 | 76415 | 146141 | 
| 178 | **20 sympy/printing/latex.py** | 2001 | 2023| 234 | 76649 | 146141 | 
| 179 | **20 sympy/printing/latex.py** | 804 | 819| 162 | 76811 | 146141 | 
| 180 | 20 sympy/parsing/latex/_antlr/latexparser.py | 2924 | 2957| 222 | 77033 | 146141 | 
| 181 | 20 sympy/parsing/latex/_antlr/latexparser.py | 1453 | 1517| 625 | 77658 | 146141 | 
| 182 | 20 sympy/printing/pretty/pretty.py | 1256 | 1338| 773 | 78431 | 146141 | 
| 183 | 21 sympy/utilities/runtests.py | 95 | 129| 258 | 78689 | 166415 | 
| 184 | 21 sympy/printing/pretty/pretty.py | 1208 | 1254| 403 | 79092 | 166415 | 
| 185 | **21 sympy/printing/latex.py** | 362 | 383| 196 | 79288 | 166415 | 
| 186 | 22 sympy/integrals/rubi/parsetools/parse.py | 1 | 103| 732 | 80020 | 173280 | 
| 187 | 22 sympy/printing/pretty/pretty_symbology.py | 466 | 501| 395 | 80415 | 173280 | 
| 188 | 22 sympy/printing/pretty/pretty_symbology.py | 1 | 46| 258 | 80673 | 173280 | 
| 189 | 23 sympy/utilities/lambdify.py | 568 | 583| 153 | 80826 | 181427 | 
| 190 | 24 sympy/interactive/printing.py | 84 | 106| 282 | 81108 | 185418 | 
| 191 | 24 sympy/printing/mathml.py | 67 | 116| 503 | 81611 | 185418 | 
| 192 | 24 sympy/printing/mathml.py | 410 | 469| 458 | 82069 | 185418 | 
| 193 | **24 sympy/printing/latex.py** | 2210 | 2220| 161 | 82230 | 185418 | 
| 194 | 24 sympy/parsing/latex/_build_latex_antlr.py | 40 | 89| 354 | 82584 | 185418 | 
| 195 | 24 sympy/printing/pretty/pretty.py | 1917 | 1951| 311 | 82895 | 185418 | 
| 196 | 25 sympy/physics/quantum/state.py | 171 | 190| 238 | 83133 | 192205 | 
| 197 | 25 sympy/parsing/latex/_antlr/latexparser.py | 562 | 586| 173 | 83306 | 192205 | 
| 198 | **25 sympy/printing/latex.py** | 1526 | 1556| 303 | 83609 | 192205 | 
| 199 | 25 sympy/printing/pretty/pretty.py | 103 | 119| 199 | 83808 | 192205 | 
| 200 | 26 sympy/printing/preview.py | 113 | 185| 733 | 84541 | 195148 | 
| 201 | 26 sympy/utilities/lambdify.py | 601 | 649| 391 | 84932 | 195148 | 
| 202 | **26 sympy/printing/latex.py** | 1499 | 1517| 166 | 85098 | 195148 | 
| 203 | 26 sympy/printing/pretty/pretty_symbology.py | 49 | 72| 201 | 85299 | 195148 | 
| 204 | 26 sympy/parsing/mathematica.py | 26 | 125| 776 | 86075 | 195148 | 
| 205 | 26 sympy/physics/quantum/state.py | 129 | 169| 467 | 86542 | 195148 | 
| 206 | **26 sympy/printing/latex.py** | 204 | 225| 219 | 86761 | 195148 | 
| 207 | 27 bin/mailmap_update.py | 1 | 104| 792 | 87553 | 196605 | 
| 208 | 28 sympy/printing/conventions.py | 1 | 71| 502 | 88055 | 197216 | 
| 209 | 28 sympy/utilities/misc.py | 1 | 28| 192 | 88247 | 197216 | 
| 210 | 28 sympy/printing/preview.py | 1 | 21| 125 | 88372 | 197216 | 
| 211 | 28 sympy/parsing/latex/_parse_latex_antlr.py | 248 | 280| 215 | 88587 | 197216 | 
| 212 | 28 sympy/printing/pretty/pretty.py | 627 | 653| 261 | 88848 | 197216 | 
| 213 | 28 sympy/printing/pretty/pretty.py | 1875 | 1888| 185 | 89033 | 197216 | 
| 214 | 28 sympy/parsing/mathematica.py | 392 | 420| 201 | 89234 | 197216 | 


## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1673,7 +1673,7 @@ def _print_TensorIndex(self, expr):
 
     def _print_tuple(self, expr):
         return r"\left( %s\right)" % \
-            r", \quad ".join([ self._print(i) for i in expr ])
+            r", \  ".join([ self._print(i) for i in expr ])
 
     def _print_TensorProduct(self, expr):
         elements = [self._print(a) for a in expr.args]
@@ -1688,7 +1688,7 @@ def _print_Tuple(self, expr):
 
     def _print_list(self, expr):
         return r"\left[ %s\right]" % \
-            r", \quad ".join([ self._print(i) for i in expr ])
+            r", \  ".join([ self._print(i) for i in expr ])
 
     def _print_dict(self, d):
         keys = sorted(d.keys(), key=default_sort_key)
@@ -1698,7 +1698,7 @@ def _print_dict(self, d):
             val = d[key]
             items.append("%s : %s" % (self._print(key), self._print(val)))
 
-        return r"\left\{ %s\right\}" % r", \quad ".join(items)
+        return r"\left\{ %s\right\}" % r", \  ".join(items)
 
     def _print_Dict(self, expr):
         return self._print_dict(expr)
@@ -2450,7 +2450,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
     dictionary.
 
     >>> print(latex([2/x, y], mode='inline'))
-    $\left[ 2 / x, \quad y\right]$
+    $\left[ 2 / x, \  y\right]$
 
     """
     if symbol_names is None:

```

## Test Patch

```diff
diff --git a/sympy/interactive/tests/test_ipythonprinting.py b/sympy/interactive/tests/test_ipythonprinting.py
--- a/sympy/interactive/tests/test_ipythonprinting.py
+++ b/sympy/interactive/tests/test_ipythonprinting.py
@@ -88,7 +88,7 @@ def test_print_builtin_option():
                     u'{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                     "{n_i: 3, pi: 3.14}",
                     u'{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')
-    assert latex == r'$\displaystyle \left\{ n_{i} : 3, \quad \pi : 3.14\right\}$'
+    assert latex == r'$\displaystyle \left\{ n_{i} : 3, \  \pi : 3.14\right\}$'
 
     app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
     app.run_cell("init_printing(use_latex=True, print_builtin=False)")
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -342,9 +342,9 @@ def test_latex_functions():
     assert latex(Order(x, (x, 0))) == r"O\left(x\right)"
     assert latex(Order(x, (x, oo))) == r"O\left(x; x\rightarrow \infty\right)"
     assert latex(Order(x - y, (x, y))) == r"O\left(x - y; x\rightarrow y\right)"
-    assert latex(Order(x, x, y)) == r"O\left(x; \left( x, \quad y\right)\rightarrow \left( 0, \quad 0\right)\right)"
-    assert latex(Order(x, x, y)) == r"O\left(x; \left( x, \quad y\right)\rightarrow \left( 0, \quad 0\right)\right)"
-    assert latex(Order(x, (x, oo), (y, oo))) == r"O\left(x; \left( x, \quad y\right)\rightarrow \left( \infty, \quad \infty\right)\right)"
+    assert latex(Order(x, x, y)) == r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
+    assert latex(Order(x, x, y)) == r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
+    assert latex(Order(x, (x, oo), (y, oo))) == r"O\left(x; \left( x, \  y\right)\rightarrow \left( \infty, \  \infty\right)\right)"
     assert latex(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
     assert latex(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'
 
@@ -867,19 +867,19 @@ def test_latex():
         "\\begin{equation*}8 \\sqrt{2} \\mu^{\\frac{7}{2}}\\end{equation*}"
     assert latex((2*mu)**Rational(7, 2), mode='equation', itex=True) == \
         "$$8 \\sqrt{2} \\mu^{\\frac{7}{2}}$$"
-    assert latex([2/x, y]) == r"\left[ \frac{2}{x}, \quad y\right]"
+    assert latex([2/x, y]) == r"\left[ \frac{2}{x}, \  y\right]"
 
 
 def test_latex_dict():
     d = {Rational(1): 1, x**2: 2, x: 3, x**3: 4}
-    assert latex(d) == r'\left\{ 1 : 1, \quad x : 3, \quad x^{2} : 2, \quad x^{3} : 4\right\}'
+    assert latex(d) == r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'
     D = Dict(d)
-    assert latex(D) == r'\left\{ 1 : 1, \quad x : 3, \quad x^{2} : 2, \quad x^{3} : 4\right\}'
+    assert latex(D) == r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'
 
 
 def test_latex_list():
     l = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
-    assert latex(l) == r'\left[ \omega_{1}, \quad a, \quad \alpha\right]'
+    assert latex(l) == r'\left[ \omega_{1}, \  a, \  \alpha\right]'
 
 
 def test_latex_rational():
@@ -1104,7 +1104,7 @@ def test_latex_Lambda():
     assert latex(Lambda(x, x + 1)) == \
         r"\left( x \mapsto x + 1 \right)"
     assert latex(Lambda((x, y), x + 1)) == \
-        r"\left( \left( x, \quad y\right) \mapsto x + 1 \right)"
+        r"\left( \left( x, \  y\right) \mapsto x + 1 \right)"
 
 
 def test_latex_PolyElement():
@@ -1325,19 +1325,19 @@ def test_categories():
 
     d = Diagram({f1: "unique", f2: S.EmptySet})
     assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
-        r"\rightarrow A_{3} : \emptyset, \quad id:A_{1}\rightarrow " \
-        r"A_{1} : \emptyset, \quad id:A_{2}\rightarrow A_{2} : " \
-        r"\emptyset, \quad id:A_{3}\rightarrow A_{3} : \emptyset, " \
-        r"\quad f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
-        r"\quad f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}"
+        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
+        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
+        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
+        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
+        r"\  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}"
 
     d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
     assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
-        r"\rightarrow A_{3} : \emptyset, \quad id:A_{1}\rightarrow " \
-        r"A_{1} : \emptyset, \quad id:A_{2}\rightarrow A_{2} : " \
-        r"\emptyset, \quad id:A_{3}\rightarrow A_{3} : \emptyset, " \
-        r"\quad f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
-        r" \quad f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}" \
+        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
+        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
+        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
+        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
+        r" \  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}" \
         r"\Longrightarrow \left\{ f_{2}\circ f_{1}:A_{1}" \
         r"\rightarrow A_{3} : \left\{unique\right\}\right\}"
 

```


## Code snippets

### 1 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 54, End line: 85

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\34\u0145\n\34\3\34\5\34\u0148\n\34\3\34\3\34\3\34\5")
        buf.write(u"\34\u014d\n\34\3\34\3\34\3\34\3\34\3\34\5\34\u0154\n")
        buf.write(u"\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34")
        buf.write(u"\3\34\5\34\u0161\n\34\3\34\3\34\3\34\3\34\3\34\3\34\5")
        buf.write(u"\34\u0169\n\34\3\35\3\35\3\35\3\35\3\35\5\35\u0170\n")
        buf.write(u"\35\3\36\3\36\3\36\3\36\3\36\3\36\3\36\3\36\3\36\5\36")
        buf.write(u"\u017b\n\36\3\36\3\36\3\37\3\37\3\37\3\37\3\37\5\37\u0184")
        buf.write(u"\n\37\3 \3 \3!\3!\3!\3!\3!\3!\5!\u018e\n!\3\"\3\"\3\"")
        buf.write(u"\3\"\3\"\3\"\5\"\u0196\n\"\3#\3#\3#\3#\3#\3$\3$\3$\3")
        buf.write(u"$\3$\3$\2\b\4\n\f\16 \"%\2\4\6\b\n\f\16\20\22\24\26\30")
        buf.write(u"\32\34\36 \"$&(*,.\60\62\64\668:<>@BDF\2\b\3\2\659\3")
        buf.write(u"\2\5\6\5\2\7\b*,\61\61\4\2\63\63;;\3\2\25(\3\2\23\24")
        buf.write(u"\2\u01b9\2H\3\2\2\2\4J\3\2\2\2\6U\3\2\2\2\bY\3\2\2\2")
        buf.write(u"\n[\3\2\2\2\ff\3\2\2\2\16q\3\2\2\2\20\u0083\3\2\2\2\22")
        buf.write(u"\u008e\3\2\2\2\24\u0090\3\2\2\2\26\u0097\3\2\2\2\30\u00a0")
        buf.write(u"\3\2\2\2\32\u00a2\3\2\2\2\34\u00aa\3\2\2\2\36\u00b2\3")
        buf.write(u"\2\2\2 \u00ba\3\2\2\2\"\u00ce\3\2\2\2$\u00e7\3\2\2\2")
        buf.write(u"&\u00ed\3\2\2\2(\u00fb\3\2\2\2*\u00fd\3\2\2\2,\u0108")
        buf.write(u"\3\2\2\2.\u010a\3\2\2\2\60\u0112\3\2\2\2\62\u0115\3\2")
        buf.write(u"\2\2\64\u011d\3\2\2\2\66\u0168\3\2\2\28\u016f\3\2\2\2")
        buf.write(u":\u0171\3\2\2\2<\u0183\3\2\2\2>\u0185\3\2\2\2@\u0187")
        buf.write(u"\3\2\2\2B\u018f\3\2\2\2D\u0197\3\2\2\2F\u019c\3\2\2\2")
        buf.write(u"HI\5\4\3\2I\3\3\2\2\2JK\b\3\1\2KL\5\b\5\2LR\3\2\2\2M")
        buf.write(u"N\f\4\2\2NO\t\2\2\2OQ\5\4\3\5PM\3\2\2\2QT\3\2\2\2RP\3")
        buf.write(u"\2\2\2RS\3\2\2\2S\5\3\2\2\2TR\3\2\2\2UV\5\b\5\2VW\7\65")
        buf.write(u"\2\2WX\5\b\5\2X\7\3\2\2\2YZ\5\n\6\2Z\t\3\2\2\2[\\\b\6")
        buf.write(u"\1\2\\]\5\f\7\2]c\3\2\2\2^_\f\4\2\2_`\t\3\2\2`b\5\n\6")
        buf.write(u"\5a^\3\2\2\2be\3\2\2\2ca\3\2\2\2cd\3\2\2\2d\13\3\2\2")
        buf.write(u"\2ec\3\2\2\2fg\b\7\1\2gh\5\20\t\2hn\3\2\2\2ij\f\4\2\2")
        buf.write(u"jk\t\4\2\2km\5\f\7\5li\3\2\2\2mp\3\2\2\2nl\3\2\2\2no")
        buf.write(u"\3\2\2\2o\r\3\2\2\2pn\3\2\2\2qr\b\b\1\2rs\5\22\n\2sy")
        buf.write(u"\3\2\2\2tu\f\4\2\2uv\t\4\2\2vx\5\16\b\5wt\3\2\2\2x{\3")
        # ... other code
```
### 2 - sympy/parsing/latex/_antlr/latexlexer.py:

Start line: 2, End line: 53

```python
# encoding: utf-8

# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated from ../LaTeX.g4, derived from latex2sympy
from __future__ import print_function
from antlr4 import *
from io import StringIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write(u"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2")
        buf.write(u";\u01e8\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4")
        buf.write(u"\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r")
        buf.write(u"\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22")
        buf.write(u"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4")
        buf.write(u"\30\t\30\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35")
        buf.write(u"\t\35\4\36\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4")
        buf.write(u"$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4,\t")
        buf.write(u",\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63")
        buf.write(u"\t\63\4\64\t\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\4")
        buf.write(u"9\t9\4:\t:\4;\t;\4<\t<\3\2\3\2\3\3\6\3}\n\3\r\3\16\3")
        buf.write(u"~\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t")
        buf.write(u"\3\t\3\n\3\n\3\13\3\13\3\f\3\f\3\r\3\r\3\16\3\16\3\17")
        buf.write(u"\3\17\3\17\3\17\3\17\3\20\3\20\3\20\3\20\3\20\3\20\3")
        buf.write(u"\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20")
        buf.write(u"\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3")
        buf.write(u"\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20")
        buf.write(u"\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3\20\3")
        buf.write(u"\20\3\20\3\20\3\20\3\20\3\20\3\20\5\20\u00d5\n\20\3\21")
        buf.write(u"\3\21\3\21\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3\23\3")
        buf.write(u"\23\3\23\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3\24\3\25")
        buf.write(u"\3\25\3\25\3\25\3\26\3\26\3\26\3\26\3\26\3\27\3\27\3")
        buf.write(u"\27\3\27\3\27\3\30\3\30\3\30\3\30\3\30\3\31\3\31\3\31")
        buf.write(u"\3\31\3\31\3\32\3\32\3\32\3\32\3\32\3\33\3\33\3\33\3")
        buf.write(u"\33\3\33\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\35")
        buf.write(u"\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\36\3\36\3\36\3")
        buf.write(u"\36\3\36\3\36\3\36\3\36\3\37\3\37\3\37\3\37\3\37\3\37")
        buf.write(u"\3\37\3\37\3 \3 \3 \3 \3 \3 \3 \3 \3!\3!\3!\3!\3!\3!")
        buf.write(u"\3!\3!\3\"\3\"\3\"\3\"\3\"\3\"\3#\3#\3#\3#\3#\3#\3$\3")
        buf.write(u"$\3$\3$\3$\3$\3%\3%\3%\3%\3%\3%\3%\3%\3&\3&\3&\3&\3&")
        buf.write(u"\3&\3&\3&\3\'\3\'\3\'\3\'\3\'\3\'\3\'\3\'\3(\3(\3(\3")
        buf.write(u"(\3(\3(\3)\3)\3)\3)\3)\3)\3)\3*\3*\3*\3*\3*\3*\3+\3+")
        buf.write(u"\3+\3+\3+\3,\3,\3,\3,\3,\3,\3-\3-\3-\3-\3-\3-\3-\3-\3")
        # ... other code
```
### 3 - sympy/parsing/latex/_antlr/latexlexer.py:

Start line: 54, End line: 84

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u".\3.\3/\3/\3\60\3\60\3\61\3\61\3\62\3\62\7\62\u0198\n")
        buf.write(u"\62\f\62\16\62\u019b\13\62\3\62\3\62\3\62\6\62\u01a0")
        buf.write(u"\n\62\r\62\16\62\u01a1\5\62\u01a4\n\62\3\63\3\63\3\64")
        buf.write(u"\3\64\3\65\6\65\u01ab\n\65\r\65\16\65\u01ac\3\65\3\65")
        buf.write(u"\3\65\3\65\3\65\7\65\u01b4\n\65\f\65\16\65\u01b7\13\65")
        buf.write(u"\3\65\7\65\u01ba\n\65\f\65\16\65\u01bd\13\65\3\65\3\65")
        buf.write(u"\3\65\3\65\3\65\7\65\u01c4\n\65\f\65\16\65\u01c7\13\65")
        buf.write(u"\3\65\3\65\6\65\u01cb\n\65\r\65\16\65\u01cc\5\65\u01cf")
        buf.write(u"\n\65\3\66\3\66\3\67\3\67\38\38\38\38\38\39\39\3:\3:")
        buf.write(u"\3:\3:\3:\3;\3;\3<\3<\6<\u01e5\n<\r<\16<\u01e6\3\u0199")
        buf.write(u"\2=\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27")
        buf.write(u"\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-")
        buf.write(u"\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%")
        buf.write(u"I&K\'M(O)Q*S+U,W-Y.[/]\60_\61a\2c\62e\63g\2i\64k\65m")
        buf.write(u"\66o\67q8s9u:w;\3\2\5\5\2\13\f\17\17\"\"\4\2C\\c|\3\2")
        buf.write(u"\62;\2\u01f4\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t")
        buf.write(u"\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3")
        buf.write(u"\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3")
        buf.write(u"\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2")
        buf.write(u"\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+")
        buf.write(u"\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2")
        buf.write(u"\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2")
        buf.write(u"\2=\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2")
        buf.write(u"\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2\2\2M\3\2\2\2\2O\3\2")
        buf.write(u"\2\2\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2W\3\2\2\2\2Y\3")
        buf.write(u"\2\2\2\2[\3\2\2\2\2]\3\2\2\2\2_\3\2\2\2\2c\3\2\2\2\2")
        buf.write(u"e\3\2\2\2\2i\3\2\2\2\2k\3\2\2\2\2m\3\2\2\2\2o\3\2\2\2")
        buf.write(u"\2q\3\2\2\2\2s\3\2\2\2\2u\3\2\2\2\2w\3\2\2\2\3y\3\2\2")
        buf.write(u"\2\5|\3\2\2\2\7\u0082\3\2\2\2\t\u0084\3\2\2\2\13\u0086")
        buf.write(u"\3\2\2\2\r\u0088\3\2\2\2\17\u008a\3\2\2\2\21\u008c\3")
        buf.write(u"\2\2\2\23\u008e\3\2\2\2\25\u0090\3\2\2\2\27\u0092\3\2")
        # ... other code
```
### 4 - sympy/printing/latex.py:

Start line: 82, End line: 117

```python
modifier_dict = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'[{ ]*[-+0-9]'),  # match
)
```
### 5 - sympy/parsing/latex/_antlr/latexlexer.py:

Start line: 117, End line: 148

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\u00b1\7v\2\2\u00b1\u00b2\7c\2\2\u00b2\u00b3\7t\2\2\u00b3")
        buf.write(u"\u00b4\7t\2\2\u00b4\u00b5\7q\2\2\u00b5\u00d5\7y\2\2\u00b6")
        buf.write(u"\u00b7\7^\2\2\u00b7\u00b8\7n\2\2\u00b8\u00b9\7q\2\2\u00b9")
        buf.write(u"\u00ba\7p\2\2\u00ba\u00bb\7i\2\2\u00bb\u00bc\7t\2\2\u00bc")
        buf.write(u"\u00bd\7k\2\2\u00bd\u00be\7i\2\2\u00be\u00bf\7j\2\2\u00bf")
        buf.write(u"\u00c0\7v\2\2\u00c0\u00c1\7c\2\2\u00c1\u00c2\7t\2\2\u00c2")
        buf.write(u"\u00c3\7t\2\2\u00c3\u00c4\7q\2\2\u00c4\u00d5\7y\2\2\u00c5")
        buf.write(u"\u00c6\7^\2\2\u00c6\u00c7\7N\2\2\u00c7\u00c8\7q\2\2\u00c8")
        buf.write(u"\u00c9\7p\2\2\u00c9\u00ca\7i\2\2\u00ca\u00cb\7t\2\2\u00cb")
        buf.write(u"\u00cc\7k\2\2\u00cc\u00cd\7i\2\2\u00cd\u00ce\7j\2\2\u00ce")
        buf.write(u"\u00cf\7v\2\2\u00cf\u00d0\7c\2\2\u00d0\u00d1\7t\2\2\u00d1")
        buf.write(u"\u00d2\7t\2\2\u00d2\u00d3\7q\2\2\u00d3\u00d5\7y\2\2\u00d4")
        buf.write(u"\u009d\3\2\2\2\u00d4\u00a0\3\2\2\2\u00d4\u00ab\3\2\2")
        buf.write(u"\2\u00d4\u00b6\3\2\2\2\u00d4\u00c5\3\2\2\2\u00d5 \3\2")
        buf.write(u"\2\2\u00d6\u00d7\7^\2\2\u00d7\u00d8\7k\2\2\u00d8\u00d9")
        buf.write(u"\7p\2\2\u00d9\u00da\7v\2\2\u00da\"\3\2\2\2\u00db\u00dc")
        buf.write(u"\7^\2\2\u00dc\u00dd\7u\2\2\u00dd\u00de\7w\2\2\u00de\u00df")
        buf.write(u"\7o\2\2\u00df$\3\2\2\2\u00e0\u00e1\7^\2\2\u00e1\u00e2")
        buf.write(u"\7r\2\2\u00e2\u00e3\7t\2\2\u00e3\u00e4\7q\2\2\u00e4\u00e5")
        buf.write(u"\7f\2\2\u00e5&\3\2\2\2\u00e6\u00e7\7^\2\2\u00e7\u00e8")
        buf.write(u"\7n\2\2\u00e8\u00e9\7q\2\2\u00e9\u00ea\7i\2\2\u00ea(")
        buf.write(u"\3\2\2\2\u00eb\u00ec\7^\2\2\u00ec\u00ed\7n\2\2\u00ed")
        buf.write(u"\u00ee\7p\2\2\u00ee*\3\2\2\2\u00ef\u00f0\7^\2\2\u00f0")
        buf.write(u"\u00f1\7u\2\2\u00f1\u00f2\7k\2\2\u00f2\u00f3\7p\2\2\u00f3")
        buf.write(u",\3\2\2\2\u00f4\u00f5\7^\2\2\u00f5\u00f6\7e\2\2\u00f6")
        buf.write(u"\u00f7\7q\2\2\u00f7\u00f8\7u\2\2\u00f8.\3\2\2\2\u00f9")
        buf.write(u"\u00fa\7^\2\2\u00fa\u00fb\7v\2\2\u00fb\u00fc\7c\2\2\u00fc")
        buf.write(u"\u00fd\7p\2\2\u00fd\60\3\2\2\2\u00fe\u00ff\7^\2\2\u00ff")
        buf.write(u"\u0100\7e\2\2\u0100\u0101\7u\2\2\u0101\u0102\7e\2\2\u0102")
        buf.write(u"\62\3\2\2\2\u0103\u0104\7^\2\2\u0104\u0105\7u\2\2\u0105")
        buf.write(u"\u0106\7g\2\2\u0106\u0107\7e\2\2\u0107\64\3\2\2\2\u0108")
        buf.write(u"\u0109\7^\2\2\u0109\u010a\7e\2\2\u010a\u010b\7q\2\2\u010b")
        # ... other code
```
### 6 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 86, End line: 119

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\2\2\2yw\3\2\2\2yz\3\2\2\2z\17\3\2\2\2{y\3\2\2\2|}\t")
        buf.write(u"\3\2\2}\u0084\5\20\t\2~\u0080\5\24\13\2\177~\3\2\2\2")
        buf.write(u"\u0080\u0081\3\2\2\2\u0081\177\3\2\2\2\u0081\u0082\3")
        buf.write(u"\2\2\2\u0082\u0084\3\2\2\2\u0083|\3\2\2\2\u0083\177\3")
        buf.write(u"\2\2\2\u0084\21\3\2\2\2\u0085\u0086\t\3\2\2\u0086\u008f")
        buf.write(u"\5\22\n\2\u0087\u008b\5\24\13\2\u0088\u008a\5\26\f\2")
        buf.write(u"\u0089\u0088\3\2\2\2\u008a\u008d\3\2\2\2\u008b\u0089")
        buf.write(u"\3\2\2\2\u008b\u008c\3\2\2\2\u008c\u008f\3\2\2\2\u008d")
        buf.write(u"\u008b\3\2\2\2\u008e\u0085\3\2\2\2\u008e\u0087\3\2\2")
        buf.write(u"\2\u008f\23\3\2\2\2\u0090\u0094\5 \21\2\u0091\u0093\5")
        buf.write(u"\30\r\2\u0092\u0091\3\2\2\2\u0093\u0096\3\2\2\2\u0094")
        buf.write(u"\u0092\3\2\2\2\u0094\u0095\3\2\2\2\u0095\25\3\2\2\2\u0096")
        buf.write(u"\u0094\3\2\2\2\u0097\u009b\5\"\22\2\u0098\u009a\5\30")
        buf.write(u"\r\2\u0099\u0098\3\2\2\2\u009a\u009d\3\2\2\2\u009b\u0099")
        buf.write(u"\3\2\2\2\u009b\u009c\3\2\2\2\u009c\27\3\2\2\2\u009d\u009b")
        buf.write(u"\3\2\2\2\u009e\u00a1\7:\2\2\u009f\u00a1\5\32\16\2\u00a0")
        buf.write(u"\u009e\3\2\2\2\u00a0\u009f\3\2\2\2\u00a1\31\3\2\2\2\u00a2")
        buf.write(u"\u00a8\7\17\2\2\u00a3\u00a9\5\36\20\2\u00a4\u00a9\5\34")
        buf.write(u"\17\2\u00a5\u00a6\5\36\20\2\u00a6\u00a7\5\34\17\2\u00a7")
        buf.write(u"\u00a9\3\2\2\2\u00a8\u00a3\3\2\2\2\u00a8\u00a4\3\2\2")
        buf.write(u"\2\u00a8\u00a5\3\2\2\2\u00a9\33\3\2\2\2\u00aa\u00ab\7")
        buf.write(u"/\2\2\u00ab\u00ae\7\13\2\2\u00ac\u00af\5\b\5\2\u00ad")
        buf.write(u"\u00af\5\6\4\2\u00ae\u00ac\3\2\2\2\u00ae\u00ad\3\2\2")
        buf.write(u"\2\u00af\u00b0\3\2\2\2\u00b0\u00b1\7\f\2\2\u00b1\35\3")
        buf.write(u"\2\2\2\u00b2\u00b3\7\60\2\2\u00b3\u00b6\7\13\2\2\u00b4")
        buf.write(u"\u00b7\5\b\5\2\u00b5\u00b7\5\6\4\2\u00b6\u00b4\3\2\2")
        buf.write(u"\2\u00b6\u00b5\3\2\2\2\u00b7\u00b8\3\2\2\2\u00b8\u00b9")
        buf.write(u"\7\f\2\2\u00b9\37\3\2\2\2\u00ba\u00bb\b\21\1\2\u00bb")
        buf.write(u"\u00bc\5$\23\2\u00bc\u00cb\3\2\2\2\u00bd\u00be\f\4\2")
        buf.write(u"\2\u00be\u00c4\7\60\2\2\u00bf\u00c5\5,\27\2\u00c0\u00c1")
        buf.write(u"\7\13\2\2\u00c1\u00c2\5\b\5\2\u00c2\u00c3\7\f\2\2\u00c3")
        buf.write(u"\u00c5\3\2\2\2\u00c4\u00bf\3\2\2\2\u00c4\u00c0\3\2\2")
        buf.write(u"\2\u00c5\u00c7\3\2\2\2\u00c6\u00c8\5@!\2\u00c7\u00c6")
        buf.write(u"\3\2\2\2\u00c7\u00c8\3\2\2\2\u00c8\u00ca\3\2\2\2\u00c9")
        # ... other code
```
### 7 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 154, End line: 190

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\3\2\2\2\u0115\u0116\7-\2\2\u0116\u0117\7\13\2\2\u0117")
        buf.write(u"\u0118\5\b\5\2\u0118\u0119\7\f\2\2\u0119\u011a\7\13\2")
        buf.write(u"\2\u011a\u011b\5\b\5\2\u011b\u011c\7\f\2\2\u011c\63\3")
        buf.write(u"\2\2\2\u011d\u011e\t\6\2\2\u011e\65\3\2\2\2\u011f\u012c")
        buf.write(u"\5\64\33\2\u0120\u0122\5@!\2\u0121\u0120\3\2\2\2\u0121")
        buf.write(u"\u0122\3\2\2\2\u0122\u0124\3\2\2\2\u0123\u0125\5B\"\2")
        buf.write(u"\u0124\u0123\3\2\2\2\u0124\u0125\3\2\2\2\u0125\u012d")
        buf.write(u"\3\2\2\2\u0126\u0128\5B\"\2\u0127\u0126\3\2\2\2\u0127")
        buf.write(u"\u0128\3\2\2\2\u0128\u012a\3\2\2\2\u0129\u012b\5@!\2")
        buf.write(u"\u012a\u0129\3\2\2\2\u012a\u012b\3\2\2\2\u012b\u012d")
        buf.write(u"\3\2\2\2\u012c\u0121\3\2\2\2\u012c\u0127\3\2\2\2\u012d")
        buf.write(u"\u0133\3\2\2\2\u012e\u012f\7\t\2\2\u012f\u0130\5<\37")
        buf.write(u"\2\u0130\u0131\7\n\2\2\u0131\u0134\3\2\2\2\u0132\u0134")
        buf.write(u"\5> \2\u0133\u012e\3\2\2\2\u0133\u0132\3\2\2\2\u0134")
        buf.write(u"\u0169\3\2\2\2\u0135\u0137\t\5\2\2\u0136\u0138\5@!\2")
        buf.write(u"\u0137\u0136\3\2\2\2\u0137\u0138\3\2\2\2\u0138\u0139")
        buf.write(u"\3\2\2\2\u0139\u013a\7\t\2\2\u013a\u013b\58\35\2\u013b")
        buf.write(u"\u013c\7\n\2\2\u013c\u0169\3\2\2\2\u013d\u0144\7\22\2")
        buf.write(u"\2\u013e\u013f\5@!\2\u013f\u0140\5B\"\2\u0140\u0145\3")
        buf.write(u"\2\2\2\u0141\u0142\5B\"\2\u0142\u0143\5@!\2\u0143\u0145")
        buf.write(u"\3\2\2\2\u0144\u013e\3\2\2\2\u0144\u0141\3\2\2\2\u0144")
        buf.write(u"\u0145\3\2\2\2\u0145\u014c\3\2\2\2\u0146\u0148\5\n\6")
        buf.write(u"\2\u0147\u0146\3\2\2\2\u0147\u0148\3\2\2\2\u0148\u0149")
        buf.write(u"\3\2\2\2\u0149\u014d\7\62\2\2\u014a\u014d\5\62\32\2\u014b")
        buf.write(u"\u014d\5\n\6\2\u014c\u0147\3\2\2\2\u014c\u014a\3\2\2")
        buf.write(u"\2\u014c\u014b\3\2\2\2\u014d\u0169\3\2\2\2\u014e\u0153")
        buf.write(u"\7)\2\2\u014f\u0150\7\r\2\2\u0150\u0151\5\b\5\2\u0151")
        buf.write(u"\u0152\7\16\2\2\u0152\u0154\3\2\2\2\u0153\u014f\3\2\2")
        buf.write(u"\2\u0153\u0154\3\2\2\2\u0154\u0155\3\2\2\2\u0155\u0156")
        buf.write(u"\7\13\2\2\u0156\u0157\5\b\5\2\u0157\u0158\7\f\2\2\u0158")
        buf.write(u"\u0169\3\2\2\2\u0159\u0160\t\7\2\2\u015a\u015b\5D#\2")
        buf.write(u"\u015b\u015c\5B\"\2\u015c\u0161\3\2\2\2\u015d\u015e\5")
        buf.write(u"B\"\2\u015e\u015f\5D#\2\u015f\u0161\3\2\2\2\u0160\u015a")
        buf.write(u"\3\2\2\2\u0160\u015d\3\2\2\2\u0161\u0162\3\2\2\2\u0162")
        buf.write(u"\u0163\5\f\7\2\u0163\u0169\3\2\2\2\u0164\u0165\7\20\2")
        buf.write(u"\2\u0165\u0166\5:\36\2\u0166\u0167\5\f\7\2\u0167\u0169")
        buf.write(u"\3\2\2\2\u0168\u011f\3\2\2\2\u0168\u0135\3\2\2\2\u0168")
        # ... other code
```
### 8 - sympy/parsing/latex/_antlr/latexlexer.py:

Start line: 85, End line: 116

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\2\2\31\u0094\3\2\2\2\33\u0096\3\2\2\2\35\u0098\3\2\2")
        buf.write(u"\2\37\u00d4\3\2\2\2!\u00d6\3\2\2\2#\u00db\3\2\2\2%\u00e0")
        buf.write(u"\3\2\2\2\'\u00e6\3\2\2\2)\u00eb\3\2\2\2+\u00ef\3\2\2")
        buf.write(u"\2-\u00f4\3\2\2\2/\u00f9\3\2\2\2\61\u00fe\3\2\2\2\63")
        buf.write(u"\u0103\3\2\2\2\65\u0108\3\2\2\2\67\u010d\3\2\2\29\u0115")
        buf.write(u"\3\2\2\2;\u011d\3\2\2\2=\u0125\3\2\2\2?\u012d\3\2\2\2")
        buf.write(u"A\u0135\3\2\2\2C\u013d\3\2\2\2E\u0143\3\2\2\2G\u0149")
        buf.write(u"\3\2\2\2I\u014f\3\2\2\2K\u0157\3\2\2\2M\u015f\3\2\2\2")
        buf.write(u"O\u0167\3\2\2\2Q\u016d\3\2\2\2S\u0174\3\2\2\2U\u017a")
        buf.write(u"\3\2\2\2W\u017f\3\2\2\2Y\u0185\3\2\2\2[\u018d\3\2\2\2")
        buf.write(u"]\u018f\3\2\2\2_\u0191\3\2\2\2a\u0193\3\2\2\2c\u0195")
        buf.write(u"\3\2\2\2e\u01a5\3\2\2\2g\u01a7\3\2\2\2i\u01ce\3\2\2\2")
        buf.write(u"k\u01d0\3\2\2\2m\u01d2\3\2\2\2o\u01d4\3\2\2\2q\u01d9")
        buf.write(u"\3\2\2\2s\u01db\3\2\2\2u\u01e0\3\2\2\2w\u01e2\3\2\2\2")
        buf.write(u"yz\7.\2\2z\4\3\2\2\2{}\t\2\2\2|{\3\2\2\2}~\3\2\2\2~|")
        buf.write(u"\3\2\2\2~\177\3\2\2\2\177\u0080\3\2\2\2\u0080\u0081\b")
        buf.write(u"\3\2\2\u0081\6\3\2\2\2\u0082\u0083\7-\2\2\u0083\b\3\2")
        buf.write(u"\2\2\u0084\u0085\7/\2\2\u0085\n\3\2\2\2\u0086\u0087\7")
        buf.write(u",\2\2\u0087\f\3\2\2\2\u0088\u0089\7\61\2\2\u0089\16\3")
        buf.write(u"\2\2\2\u008a\u008b\7*\2\2\u008b\20\3\2\2\2\u008c\u008d")
        buf.write(u"\7+\2\2\u008d\22\3\2\2\2\u008e\u008f\7}\2\2\u008f\24")
        buf.write(u"\3\2\2\2\u0090\u0091\7\177\2\2\u0091\26\3\2\2\2\u0092")
        buf.write(u"\u0093\7]\2\2\u0093\30\3\2\2\2\u0094\u0095\7_\2\2\u0095")
        buf.write(u"\32\3\2\2\2\u0096\u0097\7~\2\2\u0097\34\3\2\2\2\u0098")
        buf.write(u"\u0099\7^\2\2\u0099\u009a\7n\2\2\u009a\u009b\7k\2\2\u009b")
        buf.write(u"\u009c\7o\2\2\u009c\36\3\2\2\2\u009d\u009e\7^\2\2\u009e")
        buf.write(u"\u009f\7v\2\2\u009f\u00d5\7q\2\2\u00a0\u00a1\7^\2\2\u00a1")
        buf.write(u"\u00a2\7t\2\2\u00a2\u00a3\7k\2\2\u00a3\u00a4\7i\2\2\u00a4")
        buf.write(u"\u00a5\7j\2\2\u00a5\u00a6\7v\2\2\u00a6\u00a7\7c\2\2\u00a7")
        buf.write(u"\u00a8\7t\2\2\u00a8\u00a9\7t\2\2\u00a9\u00aa\7q\2\2\u00aa")
        buf.write(u"\u00d5\7y\2\2\u00ab\u00ac\7^\2\2\u00ac\u00ad\7T\2\2\u00ad")
        buf.write(u"\u00ae\7k\2\2\u00ae\u00af\7i\2\2\u00af\u00b0\7j\2\2\u00b0")
        # ... other code
```
### 9 - sympy/parsing/latex/_antlr/latexlexer.py:

Start line: 183, End line: 216

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\u0172\7g\2\2\u0172\u0173\7u\2\2\u0173R\3\2\2\2\u0174")
        buf.write(u"\u0175\7^\2\2\u0175\u0176\7e\2\2\u0176\u0177\7f\2\2\u0177")
        buf.write(u"\u0178\7q\2\2\u0178\u0179\7v\2\2\u0179T\3\2\2\2\u017a")
        buf.write(u"\u017b\7^\2\2\u017b\u017c\7f\2\2\u017c\u017d\7k\2\2\u017d")
        buf.write(u"\u017e\7x\2\2\u017eV\3\2\2\2\u017f\u0180\7^\2\2\u0180")
        buf.write(u"\u0181\7h\2\2\u0181\u0182\7t\2\2\u0182\u0183\7c\2\2\u0183")
        buf.write(u"\u0184\7e\2\2\u0184X\3\2\2\2\u0185\u0186\7^\2\2\u0186")
        buf.write(u"\u0187\7o\2\2\u0187\u0188\7c\2\2\u0188\u0189\7v\2\2\u0189")
        buf.write(u"\u018a\7j\2\2\u018a\u018b\7k\2\2\u018b\u018c\7v\2\2\u018c")
        buf.write(u"Z\3\2\2\2\u018d\u018e\7a\2\2\u018e\\\3\2\2\2\u018f\u0190")
        buf.write(u"\7`\2\2\u0190^\3\2\2\2\u0191\u0192\7<\2\2\u0192`\3\2")
        buf.write(u"\2\2\u0193\u0194\t\2\2\2\u0194b\3\2\2\2\u0195\u0199\7")
        buf.write(u"f\2\2\u0196\u0198\5a\61\2\u0197\u0196\3\2\2\2\u0198\u019b")
        buf.write(u"\3\2\2\2\u0199\u019a\3\2\2\2\u0199\u0197\3\2\2\2\u019a")
        buf.write(u"\u01a3\3\2\2\2\u019b\u0199\3\2\2\2\u019c\u01a4\t\3\2")
        buf.write(u"\2\u019d\u019f\7^\2\2\u019e\u01a0\t\3\2\2\u019f\u019e")
        buf.write(u"\3\2\2\2\u01a0\u01a1\3\2\2\2\u01a1\u019f\3\2\2\2\u01a1")
        buf.write(u"\u01a2\3\2\2\2\u01a2\u01a4\3\2\2\2\u01a3\u019c\3\2\2")
        buf.write(u"\2\u01a3\u019d\3\2\2\2\u01a4d\3\2\2\2\u01a5\u01a6\t\3")
        buf.write(u"\2\2\u01a6f\3\2\2\2\u01a7\u01a8\t\4\2\2\u01a8h\3\2\2")
        buf.write(u"\2\u01a9\u01ab\5g\64\2\u01aa\u01a9\3\2\2\2\u01ab\u01ac")
        buf.write(u"\3\2\2\2\u01ac\u01aa\3\2\2\2\u01ac\u01ad\3\2\2\2\u01ad")
        buf.write(u"\u01b5\3\2\2\2\u01ae\u01af\7.\2\2\u01af\u01b0\5g\64\2")
        buf.write(u"\u01b0\u01b1\5g\64\2\u01b1\u01b2\5g\64\2\u01b2\u01b4")
        buf.write(u"\3\2\2\2\u01b3\u01ae\3\2\2\2\u01b4\u01b7\3\2\2\2\u01b5")
        buf.write(u"\u01b3\3\2\2\2\u01b5\u01b6\3\2\2\2\u01b6\u01cf\3\2\2")
        buf.write(u"\2\u01b7\u01b5\3\2\2\2\u01b8\u01ba\5g\64\2\u01b9\u01b8")
        buf.write(u"\3\2\2\2\u01ba\u01bd\3\2\2\2\u01bb\u01b9\3\2\2\2\u01bb")
        buf.write(u"\u01bc\3\2\2\2\u01bc\u01c5\3\2\2\2\u01bd\u01bb\3\2\2")
        buf.write(u"\2\u01be\u01bf\7.\2\2\u01bf\u01c0\5g\64\2\u01c0\u01c1")
        buf.write(u"\5g\64\2\u01c1\u01c2\5g\64\2\u01c2\u01c4\3\2\2\2\u01c3")
        buf.write(u"\u01be\3\2\2\2\u01c4\u01c7\3\2\2\2\u01c5\u01c3\3\2\2")
        buf.write(u"\2\u01c5\u01c6\3\2\2\2\u01c6\u01c8\3\2\2\2\u01c7\u01c5")
        buf.write(u"\3\2\2\2\u01c8\u01ca\7\60\2\2\u01c9\u01cb\5g\64\2\u01ca")
        # ... other code
```
### 10 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 120, End line: 153

```python
def serializedATN():
    with StringIO() as buf:
        # ... other code
        buf.write(u"\u00bd\3\2\2\2\u00ca\u00cd\3\2\2\2\u00cb\u00c9\3\2\2")
        buf.write(u"\2\u00cb\u00cc\3\2\2\2\u00cc!\3\2\2\2\u00cd\u00cb\3\2")
        buf.write(u"\2\2\u00ce\u00cf\b\22\1\2\u00cf\u00d0\5&\24\2\u00d0\u00df")
        buf.write(u"\3\2\2\2\u00d1\u00d2\f\4\2\2\u00d2\u00d8\7\60\2\2\u00d3")
        buf.write(u"\u00d9\5,\27\2\u00d4\u00d5\7\13\2\2\u00d5\u00d6\5\b\5")
        buf.write(u"\2\u00d6\u00d7\7\f\2\2\u00d7\u00d9\3\2\2\2\u00d8\u00d3")
        buf.write(u"\3\2\2\2\u00d8\u00d4\3\2\2\2\u00d9\u00db\3\2\2\2\u00da")
        buf.write(u"\u00dc\5@!\2\u00db\u00da\3\2\2\2\u00db\u00dc\3\2\2\2")
        buf.write(u"\u00dc\u00de\3\2\2\2\u00dd\u00d1\3\2\2\2\u00de\u00e1")
        buf.write(u"\3\2\2\2\u00df\u00dd\3\2\2\2\u00df\u00e0\3\2\2\2\u00e0")
        buf.write(u"#\3\2\2\2\u00e1\u00df\3\2\2\2\u00e2\u00e8\5(\25\2\u00e3")
        buf.write(u"\u00e8\5*\26\2\u00e4\u00e8\5\66\34\2\u00e5\u00e8\5,\27")
        buf.write(u"\2\u00e6\u00e8\5\62\32\2\u00e7\u00e2\3\2\2\2\u00e7\u00e3")
        buf.write(u"\3\2\2\2\u00e7\u00e4\3\2\2\2\u00e7\u00e5\3\2\2\2\u00e7")
        buf.write(u"\u00e6\3\2\2\2\u00e8%\3\2\2\2\u00e9\u00ee\5(\25\2\u00ea")
        buf.write(u"\u00ee\5*\26\2\u00eb\u00ee\5,\27\2\u00ec\u00ee\5\62\32")
        buf.write(u"\2\u00ed\u00e9\3\2\2\2\u00ed\u00ea\3\2\2\2\u00ed\u00eb")
        buf.write(u"\3\2\2\2\u00ed\u00ec\3\2\2\2\u00ee\'\3\2\2\2\u00ef\u00f0")
        buf.write(u"\7\t\2\2\u00f0\u00f1\5\b\5\2\u00f1\u00f2\7\n\2\2\u00f2")
        buf.write(u"\u00fc\3\2\2\2\u00f3\u00f4\7\r\2\2\u00f4\u00f5\5\b\5")
        buf.write(u"\2\u00f5\u00f6\7\16\2\2\u00f6\u00fc\3\2\2\2\u00f7\u00f8")
        buf.write(u"\7\13\2\2\u00f8\u00f9\5\b\5\2\u00f9\u00fa\7\f\2\2\u00fa")
        buf.write(u"\u00fc\3\2\2\2\u00fb\u00ef\3\2\2\2\u00fb\u00f3\3\2\2")
        buf.write(u"\2\u00fb\u00f7\3\2\2\2\u00fc)\3\2\2\2\u00fd\u00fe\7\17")
        buf.write(u"\2\2\u00fe\u00ff\5\b\5\2\u00ff\u0100\7\17\2\2\u0100+")
        buf.write(u"\3\2\2\2\u0101\u0103\t\5\2\2\u0102\u0104\5@!\2\u0103")
        buf.write(u"\u0102\3\2\2\2\u0103\u0104\3\2\2\2\u0104\u0109\3\2\2")
        buf.write(u"\2\u0105\u0109\7\64\2\2\u0106\u0109\7\62\2\2\u0107\u0109")
        buf.write(u"\5.\30\2\u0108\u0101\3\2\2\2\u0108\u0105\3\2\2\2\u0108")
        buf.write(u"\u0106\3\2\2\2\u0108\u0107\3\2\2\2\u0109-\3\2\2\2\u010a")
        buf.write(u"\u010b\7.\2\2\u010b\u010c\7\13\2\2\u010c\u010d\5\60\31")
        buf.write(u"\2\u010d\u010e\7\f\2\2\u010e/\3\2\2\2\u010f\u0111\7\63")
        buf.write(u"\2\2\u0110\u010f\3\2\2\2\u0111\u0114\3\2\2\2\u0112\u0110")
        buf.write(u"\3\2\2\2\u0112\u0113\3\2\2\2\u0113\61\3\2\2\2\u0114\u0112")
        # ... other code
```
### 19 - sympy/printing/latex.py:

Start line: 1807, End line: 1869

```python
class LatexPrinter(Printer):

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        return r" \cup ".join([self._print(i) for i in u.args])

    def _print_Complement(self, u):
        return r" \setminus ".join([self._print(i) for i in u.args])

    def _print_Intersection(self, u):
        return r" \cap ".join([self._print(i) for i in u.args])

    def _print_SymmetricDifference(self, u):
        return r" \triangle ".join([self._print(i) for i in u.args])

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        sets = s.args[1:]
        varsets = [r"%s \in %s" % (self._print(var), self._print(setv))
            for var, setv in zip(s.lamda.variables, sets)]
        return r"\left\{%s\; |\; %s\right\}" % (
            self._print(s.lamda.expr),
            ', '.join(varsets))
```
### 20 - sympy/printing/latex.py:

Start line: 2255, End line: 2477

```python
def translate(s):
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True):
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s


def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None, root_notation=True, imaginary_unit="i"):
    # ... other code
```
### 23 - sympy/printing/latex.py:

Start line: 2290, End line: 2455

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None, root_notation=True, imaginary_unit="i"):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``abbreviated``, ``full``, or ``power``. Defaults to ``abbreviated``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``[``, ``(``, or
        the empty string. Defaults to ``[``.
    mat_str : string, optional
        Which matrix environment string to emit. ``smallmatrix``, ``matrix``,
        ``array``, etc. Defaults to ``smallmatrix`` for inline mode, ``matrix``
        for matrices of no more than 10 columns, and ``array`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``plain``, ``inline``, ``equation`` or ``equation*``.  If ``mode``
        is set to ``plain``, then the resulting code will not be delimited at
        all (this is the default). If ``mode`` is set to ``inline`` then inline
        LaTeX ``$...$`` will be used. If ``mode`` is set to ``equation`` or
        ``equation*``, the resulting code will be enclosed in the ``equation``
        or ``equation*`` environment (remember to import ``amsmath`` for
        ``equation*``), unless the ``itex`` option is set. In the latter case,
        the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``, ``ldot``,
        ``dot``, or ``times``.
    order: string, optional
        Any of the supported monomial orderings (currently ``lex``, ``grlex``,
        or ``grevlex``), ``old``, and ``none``. This parameter does nothing for
        Mul objects. Setting order to ``old`` uses the compatibility ordering
        for Add defined in Printer. For very large expressions, set the
        ``order`` keyword to ``none`` if speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal form.
        Default is ``True``, to print exponent in root form.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are "i" (default)
        and "j". Adding "b" or "t" in front gives ``\mathrm`` or ``\text``, so
        "bi" leads to ``\mathrm{i}`` which gives `\mathrm{i}`.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left(\frac{7}{2} \right)}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left(10 \right)}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left(10 \right)}

    ``latex()`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \quad y\right]$

    """
    # ... other code
```
### 33 - sympy/printing/latex.py:

Start line: 1871, End line: 1882

```python
class LatexPrinter(Printer):

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return r"\left\{%s \mid %s \right\}" % (
            vars_print,
            self._print(s.condition.as_expr()))

        return r"\left\{%s \mid %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition.as_expr()))
```
### 34 - sympy/printing/latex.py:

Start line: 120, End line: 202

```python
class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings = {
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "order": None,
        "symbol_names": {},
        "root_notation": True,
        "imaginary_unit": "i",
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation',
                           'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                    "'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        try:
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        self._delim_dict = {'(': ')', '[': ']'}

        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        try:
            self._settings['imaginary_unit_latex'] = \
                imaginary_unit_table[self._settings['imaginary_unit']]
        except KeyError:
            self._settings['imaginary_unit_latex'] = \
                self._settings['imaginary_unit']

    def parenthesize(self, item, level, strict=False):
        prec_val = precedence_traditional(item)
        if (prec_val < level) or ((not strict) and prec_val <= level):
            return r"\left(%s\right)" % self._print(item)
        else:
            return self._print(item)
```
### 35 - sympy/printing/latex.py:

Start line: 922, End line: 977

```python
class LatexPrinter(Printer):

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex
```
### 39 - sympy/printing/latex.py:

Start line: 1392, End line: 1410

```python
class LatexPrinter(Printer):

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        return "%s %s %s" % (self._print(expr.lhs),
            charmap[expr.rel_op], self._print(expr.rhs))
```
### 42 - sympy/printing/latex.py:

Start line: 2144, End line: 2208

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{%s}^{%s}' % (self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ %s \right]" % ",".join(
            '{' + self._print(x) + '}' for x in m)

    def _print_SubModule(self, m):
        return r"\left\langle %s \right\rangle" % ",".join(
            '{' + self._print(x) + '}' for x in m.gens)

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left\langle %s \right\rangle" % ",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens)

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(R.ring), self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{%s} + {%s}" % (self._print(x.data), self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{%s} + {%s}" % (self._print(m.data),
                                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(M.base),
                                   self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{%s} : {%s} \to {%s}" % (self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\boldsymbol{\mathrm{%s}}' % self._print(Symbol(string))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\partial_{%s}' % self._print(Symbol(string))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return r'\mathrm{d}%s' % self._print(Symbol(string))
        else:
            return 'd(%s)' % self._print(field)
            string = self._print(field)
            return r'\mathrm{d}\left(%s\right)' % string

    def _print_Tr(self, p):
        #Todo: Handle indices
        contents = self._print(p.args[0])
        return r'\mbox{Tr}\left(%s\right)' % (contents)
```
### 50 - sympy/printing/latex.py:

Start line: 672, End line: 702

```python
class LatexPrinter(Printer):

    def _print_Integral(self, expr):
        tex, symbols = "", []

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\, d%s" % self._print(symbol[0])
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] != 'inline' \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                symbols.insert(0, r"\, d%s" % self._print(symbol))

        return r"%s %s%s" % (tex,
            self.parenthesize(expr.function, PRECEDENCE["Mul"], strict=True), "".join(symbols))
```
### 51 - sympy/printing/latex.py:

Start line: 1, End line: 81

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true

## sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key, range, string_types
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which sympy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec', 'csc',
                    'cot', 'coth', 're', 'im', 'frac', 'root', 'arg',
                    ]

tex_greek_dictionary = {
    'Alpha': 'A',
    'Beta': 'B',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': 'E',
    'Zeta': 'Z',
    'Eta': 'H',
    'Theta': r'\Theta',
    'Iota': 'I',
    'Kappa': 'K',
    'Lambda': r'\Lambda',
    'Mu': 'M',
    'Nu': 'N',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': 'O',
    'Pi': r'\Pi',
    'Rho': 'P',
    'Sigma': r'\Sigma',
    'Tau': 'T',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': 'X',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'X',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = set(['aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp', ])

# Variable name modifiers
```
### 52 - sympy/printing/latex.py:

Start line: 1364, End line: 1390

```python
class LatexPrinter(Printer):

    def _print_Symbol(self, expr):
        if expr in self._settings['symbol_names']:
            return self._settings['symbol_names'][expr]

        return self._deal_with_super_sub(expr.name) if \
            '\\' not in expr.name else expr.name

    _print_RandomSymbol = _print_Symbol
    _print_MatrixSymbol = _print_Symbol

    def _deal_with_super_sub(self, string):
        if '{' in string:
            return string

        name, supers, subs = split_super_sub(string)

        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        # glue all items together:
        if len(supers) > 0:
            name += "^{%s}" % " ".join(supers)
        if len(subs) > 0:
            name += "_{%s}" % " ".join(subs)

        return name
```
### 56 - sympy/printing/latex.py:

Start line: 907, End line: 920

```python
class LatexPrinter(Printer):

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        for arg in args[1:]:
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex
```
### 60 - sympy/printing/latex.py:

Start line: 1412, End line: 1423

```python
class LatexPrinter(Printer):

    def _print_Piecewise(self, expr):
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)
```
### 61 - sympy/printing/latex.py:

Start line: 663, End line: 670

```python
class LatexPrinter(Printer):

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr, latex_subs)
```
### 63 - sympy/printing/latex.py:

Start line: 1425, End line: 1450

```python
class LatexPrinter(Printer):

    def _print_MatrixBase(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([ self._print(i) for i in expr[line, :] ]))

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str % r"\\".join(lines)
```
### 64 - sympy/printing/latex.py:

Start line: 385, End line: 407

```python
class LatexPrinter(Printer):

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(func, PRECEDENCE['Mul'])
```
### 65 - sympy/printing/latex.py:

Start line: 1706, End line: 1714

```python
class LatexPrinter(Printer):

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 67 - sympy/printing/latex.py:

Start line: 1884, End line: 1931

```python
class LatexPrinter(Printer):

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; |\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        return self._print_Add(s.truncate()) + self._print(r' + \ldots')

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)
```
### 68 - sympy/printing/latex.py:

Start line: 509, End line: 559

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1 and self._settings['root_notation']:
            base = self._print(expr.base)
            expq = expr.exp.q

            if expq == 2:
                tex = r"\sqrt{%s}" % base
            elif self._settings['itex']:
                tex = r"\root{%d}{%s}" % (expq, base)
            else:
                tex = r"\sqrt[%d]{%s}" % (expq, base)

            if expr.exp.is_negative:
                return r"\frac{1}{%s}" % tex
            else:
                return tex
        elif self._settings['fold_frac_powers'] \
            and expr.exp.is_Rational \
                and expr.exp.q != 1:
            base, p, q = self.parenthesize(expr.base, PRECEDENCE['Pow']), expr.exp.p, expr.exp.q
            # issue #12886: add parentheses for superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
            # special case for 1^(-x), issue 9216
            if expr.base == 1:
                return r"%s^{%s}" % (expr.base, expr.exp)
            # things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, exp=self._print(expr.exp))
            else:
                tex = r"%s^{%s}"
                exp = self._print(expr.exp)
                # issue #12886: add parentheses around superscripts raised to powers
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if '^' in base and expr.base.is_Symbol:
                    base = r"\left(%s\right)" % base
                elif isinstance(expr.base, Derivative
                        ) and base.startswith(r'\left('
                        ) and re.match(r'\\left\(\\d?d?dot', base
                        ) and base.endswith(r'\right)'):
                    # don't use parentheses around dotted derivative
                    base = base[6: -7]  # remove outermost added parens

                return tex % (base, exp)
```
### 69 - sympy/printing/latex.py:

Start line: 1451, End line: 1474

```python
class LatexPrinter(Printer):
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix \
                           = _print_Matrix \
                           = _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '_{%s, %s}' % (
            self._print(expr.i),
            self._print(expr.j)
        )

    def _print_MatrixSlice(self, expr):
        def latexslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(self._print, x))
        return (self._print(expr.parent) + r'\left[' +
                latexslice(expr.rowslice) + ', ' +
                latexslice(expr.colslice) + r'\right]')
```
### 71 - sympy/printing/latex.py:

Start line: 1639, End line: 1704

```python
class LatexPrinter(Printer):

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )
        return self._print(expr.args[0])

    def _print_tuple(self, expr):
        return r"\left( %s\right)" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        return r"\left[ %s\right]" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \quad ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)
```
### 78 - sympy/printing/latex.py:

Start line: 896, End line: 905

```python
class LatexPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg (%s)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
```
### 79 - sympy/printing/latex.py:

Start line: 718, End line: 733

```python
class LatexPrinter(Printer):

    def _hprint_Function(self, func):
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\'):
            name = func
        else:
            name = r"\operatorname{%s}" % func
        return name
```
### 83 - sympy/printing/latex.py:

Start line: 1348, End line: 1362

```python
class LatexPrinter(Printer):

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif len(expr.variables):
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"O\left(%s\right)" % s
```
### 85 - sympy/printing/latex.py:

Start line: 1558, End line: 1605

```python
class LatexPrinter(Printer):

    def _print_NDimArray(self, expr):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                      r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str
```
### 87 - sympy/printing/latex.py:

Start line: 821, End line: 894

```python
class LatexPrinter(Printer):

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _hprint_variadic_function(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (self._print((str(expr.func)).lower()), ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\left\lfloor{%s}\right\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\left\lceil{%s}\right\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left(%s \right)}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
    _print_Determinant = _print_Abs

    def _print_re(self, expr, exp=None):
        tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Func'])

        return self._do_exponent(tex, exp)
```
### 89 - sympy/printing/latex.py:

Start line: 704, End line: 716

```python
class LatexPrinter(Printer):

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))
```
### 92 - sympy/printing/latex.py:

Start line: 2126, End line: 2142

```python
class LatexPrinter(Printer):

    def _print_DiagramGrid(self, grid):
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        latex_result += "\\end{array}\n"
        return latex_result
```
### 101 - sympy/printing/latex.py:

Start line: 1792, End line: 1805

```python
class LatexPrinter(Printer):

    def _print_SeqFormula(self, s):
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (r'\ldots', s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(r'\ldots')
        else:
            printset = tuple(s)

        return (r"\left["
              + r", ".join(self._print(el) for el in printset)
              + r"\right]")
```
### 103 - sympy/printing/latex.py:

Start line: 1139, End line: 1204

```python
class LatexPrinter(Printer):

    def _hprint_vec(self, vec):
        if len(vec) == 0:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')
```
### 105 - sympy/printing/latex.py:

Start line: 1739, End line: 1775

```python
class LatexPrinter(Printer):

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_ProductSet(self, p):
        if len(p.sets) > 1 and not has_variety(p.sets):
            return self._print(p.sets[0]) + "^{%d}" % len(p.sets)
        else:
            return r" \times ".join(self._print(set) for set in p.sets)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return 'Domain on ' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        items = ", ".join(map(self._print, items))
        return r"\left\{%s\right\}" % items

    _print_frozenset = _print_set
```
### 106 - sympy/printing/latex.py:

Start line: 1777, End line: 1790

```python
class LatexPrinter(Printer):

    def _print_Range(self, s):
        dots = r'\ldots'

        if s.start.is_infinite:
            printset = s.start, dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite or len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)

        return (r"\left\{"
              + r", ".join(self._print(el) for el in printset)
              + r"\right\}")
```
### 108 - sympy/printing/latex.py:

Start line: 1286, End line: 1346

```python
class LatexPrinter(Printer):

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            return self._print(expr.p)
```
### 112 - sympy/printing/latex.py:

Start line: 1230, End line: 1284

```python
class LatexPrinter(Printer):

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (self._print(exp), tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (self._print(exp), tex)
        return r"\zeta%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (self._print(exp), tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, self._print(exp), tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex
```
### 115 - sympy/printing/latex.py:

Start line: 2034, End line: 2077

```python
class LatexPrinter(Printer):

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_MellinTransform(self, expr):
        return r"\mathcal{M}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseMellinTransform(self, expr):
        return r"\mathcal{M}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_LaplaceTransform(self, expr):
        return r"\mathcal{L}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseLaplaceTransform(self, expr):
        return r"\mathcal{L}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_FourierTransform(self, expr):
        return r"\mathcal{F}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseFourierTransform(self, expr):
        return r"\mathcal{F}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_SineTransform(self, expr):
        return r"\mathcal{SIN}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseSineTransform(self, expr):
        return r"\mathcal{SIN}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_CosineTransform(self, expr):
        return r"\mathcal{COS}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseCosineTransform(self, expr):
        return r"\mathcal{COS}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))
```
### 117 - sympy/printing/latex.py:

Start line: 1206, End line: 1215

```python
class LatexPrinter(Printer):

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 119 - sympy/printing/latex.py:

Start line: 1217, End line: 1228

```python
class LatexPrinter(Printer):

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 121 - sympy/printing/latex.py:

Start line: 735, End line: 802

```python
class LatexPrinter(Printer):

    def _print_Function(self, expr, exp=None):
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and \
            not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [ str(self._print(arg)) for arg in expr.args ]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    func = func
                elif inv_trig_style == "full":
                    func = "arc" + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                name = r'%s^{%s}' % (self._hprint_Function(func), exp)
            else:
                name = self._hprint_Function(func)

            if can_fold_brackets:
                if func in accepted_latex_functions:
                    # Wrap argument safely to avoid parse-time conflicts
                    # with the function name itself
                    name += r" {%s}"
                else:
                    name += r"%s"
            else:
                name += r"{\left(%s \right)}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)
```
### 122 - sympy/printing/latex.py:

Start line: 561, End line: 581

```python
class LatexPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 124 - sympy/printing/latex.py:

Start line: 583, End line: 600

```python
class LatexPrinter(Printer):

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 128 - sympy/printing/latex.py:

Start line: 344, End line: 360

```python
class LatexPrinter(Printer):

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    _print_Permutation = _print_Cycle
```
### 129 - sympy/printing/latex.py:

Start line: 1519, End line: 1524

```python
class LatexPrinter(Printer):

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s\bmod{%s}\right)^{%s}' % (self.parenthesize(expr.args[0],
                    PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]), self._print(exp))
        return r'%s\bmod{%s}' % (self.parenthesize(expr.args[0],
                PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]))
```
### 131 - sympy/printing/latex.py:

Start line: 979, End line: 988

```python
class LatexPrinter(Printer):

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex
```
### 133 - sympy/printing/latex.py:

Start line: 2232, End line: 2240

```python
class LatexPrinter(Printer):

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (self._print(exp), tex)
        return r"\sigma^*%s" % tex
```
### 134 - sympy/printing/latex.py:

Start line: 460, End line: 507

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        # ... other code

        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] \
                   and ldenom <= 2 and not "^" in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex
```
### 137 - sympy/printing/latex.py:

Start line: 1607, End line: 1637

```python
class LatexPrinter(Printer):

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_MutableSparseNDimArray = _print_NDimArray

    def _printer_tensor_indices(self, name, indices, index_map={}):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if ((index in index_map) or prev_map) and last_valence == new_valence:
                out_str += ","
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += "}"
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += "}"
        return out_str
```
### 144 - sympy/printing/latex.py:

Start line: 632, End line: 661

```python
class LatexPrinter(Printer):

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Derivative(self, expr):
        if requires_partial(expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol, self._print(x), num)

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, dim, tex)

        return r"%s %s" % (tex, self.parenthesize(expr.expr, PRECEDENCE["Mul"], strict=True))
```
### 146 - sympy/printing/latex.py:

Start line: 1003, End line: 1087

```python
class LatexPrinter(Printer):

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 147 - sympy/printing/latex.py:

Start line: 2079, End line: 2124

```python
class LatexPrinter(Printer):

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return "\\mathbf{%s}" % self._print(Symbol(morphism.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result
```
### 149 - sympy/printing/latex.py:

Start line: 1728, End line: 1737

```python
class LatexPrinter(Printer):

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex
```
### 153 - sympy/printing/latex.py:

Start line: 2456, End line: 2484

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None, root_notation=True, imaginary_unit="i"):
    if symbol_names is None:
        symbol_names = {}

    settings = {
        'fold_frac_powers' : fold_frac_powers,
        'fold_func_brackets' : fold_func_brackets,
        'fold_short_frac' : fold_short_frac,
        'inv_trig_style' : inv_trig_style,
        'itex' : itex,
        'ln_notation' : ln_notation,
        'long_frac_ratio' : long_frac_ratio,
        'mat_delim' : mat_delim,
        'mat_str' : mat_str,
        'mode' : mode,
        'mul_symbol' : mul_symbol,
        'order' : order,
        'symbol_names' : symbol_names,
        'root_notation' : root_notation,
        'imaginary_unit' : imaginary_unit,
    }

    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""
    print(latex(expr, **settings))
```
### 155 - sympy/printing/latex.py:

Start line: 1716, End line: 1726

```python
class LatexPrinter(Printer):

    def _print_SingularityFunction(self, expr):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\left\langle %s \right\rangle}^{%s}" % (shift, power)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        tex = r"\theta\left(%s\right)" % self._print(expr.args[0])
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 157 - sympy/printing/latex.py:

Start line: 2242, End line: 2252

```python
class LatexPrinter(Printer):

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])
```
### 161 - sympy/printing/latex.py:

Start line: 322, End line: 342

```python
class LatexPrinter(Printer):

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif _coeff_isneg(term):
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex

        return tex
```
### 165 - sympy/printing/latex.py:

Start line: 409, End line: 458

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        from sympy.core.power import Pow
        from sympy.physics.units import Quantity
        include_parens = False
        if _coeff_isneg(expr):
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        from sympy.simplify import fraction
        numer, denom = fraction(expr, exact=True)
        separator = self._settings['mul_symbol_latex']
        numbersep = self._settings['mul_symbol_latex_numbers']

        def convert(expr):
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                _tex = last_term_tex = ""

                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If quantities are present append them at the back
                args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                             (isinstance(x, Pow) and isinstance(x.base, Quantity)))

                for i, term in enumerate(args):
                    term_tex = self._print(term)

                    if self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1)):
                        term_tex = r"\left(%s\right)" % term_tex

                    if _between_two_numbers_p[0].search(last_term_tex) and \
                            _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator

                    _tex += term_tex
                    last_term_tex = term_tex
                return _tex
        # ... other code
```
### 167 - sympy/printing/latex.py:

Start line: 2025, End line: 2032

```python
class LatexPrinter(Printer):

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex
```
### 168 - sympy/printing/latex.py:

Start line: 1476, End line: 1497

```python
class LatexPrinter(Printer):

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^T" % self._print(mat)
        else:
            return "%s^T" % self._print(mat)

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\mathrm{tr}\left(%s \right)" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)
```
### 172 - sympy/printing/latex.py:

Start line: 1933, End line: 1988

```python
class LatexPrinter(Printer):

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ['-', '+']:
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left(%s \right)}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex
```
### 174 - sympy/printing/latex.py:

Start line: 2222, End line: 2230

```python
class LatexPrinter(Printer):

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (self._print(exp), tex)
        return r"\sigma%s" % tex
```
### 176 - sympy/printing/latex.py:

Start line: 990, End line: 1001

```python
class LatexPrinter(Printer):

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]), \
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex
```
### 178 - sympy/printing/latex.py:

Start line: 2001, End line: 2023

```python
class LatexPrinter(Printer):

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls, ", ".join(args))

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)
```
### 179 - sympy/printing/latex.py:

Start line: 804, End line: 819

```python
class LatexPrinter(Printer):

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}
```
### 185 - sympy/printing/latex.py:

Start line: 362, End line: 383

```python
class LatexPrinter(Printer):

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            return str_real
```
### 193 - sympy/printing/latex.py:

Start line: 2210, End line: 2220

```python
class LatexPrinter(Printer):

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])
```
### 198 - sympy/printing/latex.py:

Start line: 1526, End line: 1556

```python
class LatexPrinter(Printer):

    def _print_HadamardProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \circ '.join(map(parens, expr.args))

    def _print_KroneckerProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \otimes '.join(map(parens, expr.args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol):
            return r"\left(%s\right)^{%s}" % (self._print(base), self._print(exp))
        else:
            return "%s^{%s}" % (self._print(base), self._print(exp))

    def _print_ZeroMatrix(self, Z):
        return r"\mathbb{0}"

    def _print_Identity(self, I):
        return r"\mathbb{I}"
```
### 202 - sympy/printing/latex.py:

Start line: 1499, End line: 1517

```python
class LatexPrinter(Printer):

    def _print_MatMul(self, expr):
        from sympy import MatMul, Mul

        parens = lambda x: self.parenthesize(x, precedence_traditional(expr), False)

        args = expr.args
        if isinstance(args[0], Mul):
            args = args[0].as_ordered_factors() + list(args[1:])
        else:
            args = list(args)

        if isinstance(expr, MatMul) and _coeff_isneg(expr):
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            return '- ' + ' '.join(map(parens, args))
        else:
            return ' '.join(map(parens, args))
```
### 206 - sympy/printing/latex.py:

Start line: 204, End line: 225

```python
class LatexPrinter(Printer):

    def doprint(self, expr):
        tex = Printer.doprint(self, expr)

        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex
        elif self._settings['itex']:
            return r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)

    def _needs_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))
```
