# sympy__sympy-14774

| **sympy/sympy** | `8fc63c2d71752389a44367b8ef4aba8a91af6a45` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 650 |
| **Any found context length** | 650 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -740,7 +740,7 @@ def _print_Function(self, expr, exp=None):
                 len(args) == 1 and \
                 not self._needs_function_brackets(expr.args[0])
 
-            inv_trig_table = ["asin", "acos", "atan", "acot"]
+            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]
 
             # If the function is an inverse trig function, handle the style
             if func in inv_trig_table:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 743 | 743 | 1 | 1 | 650


## Problem Statement

```
Latex printer does not support full inverse trig function names for acsc and asec
For example
`latex(asin(x), inv_trig_style="full")` works as expected returning `'\\arcsin{\\left (x \\right )}'`
But `latex(acsc(x), inv_trig_style="full")` gives `'\\operatorname{acsc}{\\left (x \\right )}'` instead of `'\\operatorname{arccsc}{\\left (x \\right )}'`

A fix seems to be to change line 743 of sympy/printing/latex.py from
`inv_trig_table = ["asin", "acos", "atan", "acot"]` to
`inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/printing/latex.py** | 715 | 782| 650 | 650 | 23024 | 
| 2 | **1 sympy/printing/latex.py** | 912 | 967| 577 | 1227 | 23024 | 
| 3 | **1 sympy/printing/latex.py** | 1962 | 2005| 771 | 1998 | 23024 | 
| 4 | 2 sympy/physics/vector/printing.py | 45 | 120| 664 | 2662 | 26450 | 
| 5 | **2 sympy/printing/latex.py** | 83 | 118| 491 | 3153 | 26450 | 
| 6 | **2 sympy/printing/latex.py** | 652 | 682| 329 | 3482 | 26450 | 
| 7 | **2 sympy/printing/latex.py** | 1110 | 1125| 138 | 3620 | 26450 | 
| 8 | **2 sympy/printing/latex.py** | 1634 | 1642| 135 | 3755 | 26450 | 
| 9 | **2 sympy/printing/latex.py** | 1 | 82| 715 | 4470 | 26450 | 
| 10 | **2 sympy/printing/latex.py** | 801 | 884| 744 | 5214 | 26450 | 
| 11 | **2 sympy/printing/latex.py** | 886 | 895| 136 | 5350 | 26450 | 
| 12 | **2 sympy/printing/latex.py** | 365 | 387| 268 | 5618 | 26450 | 
| 13 | **2 sympy/printing/latex.py** | 613 | 641| 261 | 5879 | 26450 | 
| 14 | **2 sympy/printing/latex.py** | 1644 | 1654| 137 | 6016 | 26450 | 
| 15 | **2 sympy/printing/latex.py** | 489 | 540| 588 | 6604 | 26450 | 
| 16 | **2 sympy/printing/latex.py** | 1127 | 1192| 672 | 7276 | 26450 | 
| 17 | **2 sympy/printing/latex.py** | 1352 | 1378| 210 | 7486 | 26450 | 
| 18 | 3 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 59| 471 | 7957 | 30842 | 
| 19 | **3 sympy/printing/latex.py** | 1656 | 1665| 131 | 8088 | 30842 | 
| 20 | **3 sympy/printing/latex.py** | 1953 | 1960| 118 | 8206 | 30842 | 
| 21 | **3 sympy/printing/latex.py** | 1495 | 1507| 159 | 8365 | 30842 | 
| 22 | **3 sympy/printing/latex.py** | 1735 | 1797| 538 | 8903 | 30842 | 
| 23 | **3 sympy/printing/latex.py** | 698 | 713| 161 | 9064 | 30842 | 
| 24 | **3 sympy/printing/latex.py** | 1812 | 1859| 457 | 9521 | 30842 | 
| 25 | **3 sympy/printing/latex.py** | 324 | 340| 165 | 9686 | 30842 | 
| 26 | **3 sympy/printing/latex.py** | 1720 | 1733| 143 | 9829 | 30842 | 
| 27 | **3 sympy/printing/latex.py** | 969 | 978| 126 | 9955 | 30842 | 
| 28 | **3 sympy/printing/latex.py** | 389 | 438| 383 | 10338 | 30842 | 
| 29 | **3 sympy/printing/latex.py** | 1274 | 1334| 753 | 11091 | 30842 | 
| 30 | **3 sympy/printing/latex.py** | 980 | 991| 157 | 11248 | 30842 | 
| 31 | **3 sympy/printing/latex.py** | 993 | 1075| 781 | 12029 | 30842 | 
| 32 | **3 sympy/printing/latex.py** | 1516 | 1546| 303 | 12332 | 30842 | 
| 33 | **3 sympy/printing/latex.py** | 2160 | 2168| 129 | 12461 | 30842 | 
| 34 | **3 sympy/printing/latex.py** | 897 | 910| 139 | 12600 | 30842 | 
| 35 | **3 sympy/printing/latex.py** | 1509 | 1514| 116 | 12716 | 30842 | 
| 36 | **3 sympy/printing/latex.py** | 1918 | 1927| 125 | 12841 | 30842 | 
| 37 | **3 sympy/printing/latex.py** | 1667 | 1703| 380 | 13221 | 30842 | 
| 38 | **3 sympy/printing/latex.py** | 1461 | 1493| 263 | 13484 | 30842 | 
| 39 | **3 sympy/printing/latex.py** | 643 | 650| 127 | 13611 | 30842 | 
| 40 | **3 sympy/printing/latex.py** | 684 | 696| 159 | 13770 | 30842 | 
| 41 | **3 sympy/printing/latex.py** | 583 | 611| 239 | 14009 | 30842 | 
| 42 | **3 sympy/printing/latex.py** | 784 | 799| 162 | 14171 | 30842 | 
| 43 | **3 sympy/printing/latex.py** | 1194 | 1203| 138 | 14309 | 30842 | 
| 44 | **3 sympy/printing/latex.py** | 2150 | 2158| 123 | 14432 | 30842 | 
| 45 | **3 sympy/printing/latex.py** | 1705 | 1718| 134 | 14566 | 30842 | 
| 46 | **3 sympy/printing/latex.py** | 1205 | 1216| 187 | 14753 | 30842 | 
| 47 | **3 sympy/printing/latex.py** | 1077 | 1108| 308 | 15061 | 30842 | 
| 48 | **3 sympy/printing/latex.py** | 2072 | 2136| 754 | 15815 | 30842 | 
| 49 | **3 sympy/printing/latex.py** | 121 | 185| 536 | 16351 | 30842 | 
| 50 | **3 sympy/printing/latex.py** | 2183 | 2345| 230 | 16581 | 30842 | 
| 51 | **3 sympy/printing/latex.py** | 1218 | 1272| 730 | 17311 | 30842 | 
| 52 | **3 sympy/printing/latex.py** | 564 | 581| 197 | 17508 | 30842 | 
| 53 | **3 sympy/printing/latex.py** | 1929 | 1951| 234 | 17742 | 30842 | 
| 54 | **3 sympy/printing/latex.py** | 2138 | 2148| 161 | 17903 | 30842 | 
| 55 | **3 sympy/printing/latex.py** | 342 | 363| 196 | 18099 | 30842 | 
| 56 | **3 sympy/printing/latex.py** | 1380 | 1398| 135 | 18234 | 30842 | 
| 57 | **3 sympy/printing/latex.py** | 1413 | 1438| 271 | 18505 | 30842 | 
| 58 | **3 sympy/printing/latex.py** | 302 | 322| 149 | 18654 | 30842 | 
| 59 | **3 sympy/printing/latex.py** | 1439 | 1459| 208 | 18862 | 30842 | 
| 60 | 4 sympy/printing/glsl.py | 1 | 28| 202 | 19064 | 35640 | 
| 61 | **4 sympy/printing/latex.py** | 440 | 487| 529 | 19593 | 35640 | 
| 62 | **4 sympy/printing/latex.py** | 1400 | 1411| 171 | 19764 | 35640 | 
| 63 | **4 sympy/printing/latex.py** | 2170 | 2180| 159 | 19923 | 35640 | 
| 64 | **4 sympy/printing/latex.py** | 2007 | 2052| 407 | 20330 | 35640 | 
| 65 | **4 sympy/printing/latex.py** | 1336 | 1350| 147 | 20477 | 35640 | 
| 66 | 5 sympy/printing/theanocode.py | 1 | 64| 563 | 21040 | 37723 | 
| 67 | **5 sympy/printing/latex.py** | 1548 | 1595| 489 | 21529 | 37723 | 
| 68 | 5 sympy/physics/vector/printing.py | 122 | 160| 400 | 21929 | 37723 | 
| 69 | **5 sympy/printing/latex.py** | 1597 | 1632| 344 | 22273 | 37723 | 
| 70 | **5 sympy/printing/latex.py** | 1861 | 1916| 414 | 22687 | 37723 | 
| 71 | 6 sympy/functions/elementary/trigonometric.py | 2529 | 2535| 136 | 22823 | 62110 | 
| 72 | 6 sympy/functions/elementary/trigonometric.py | 2010 | 2045| 299 | 23122 | 62110 | 
| 73 | **6 sympy/printing/latex.py** | 1799 | 1810| 133 | 23255 | 62110 | 
| 74 | 7 sympy/parsing/latex/__init__.py | 1 | 32| 273 | 23528 | 62383 | 
| 75 | 7 sympy/parsing/latex/_parse_latex_antlr.py | 62 | 88| 188 | 23716 | 62383 | 
| 76 | **7 sympy/printing/latex.py** | 542 | 562| 221 | 23937 | 62383 | 
| 77 | 7 sympy/physics/vector/printing.py | 344 | 381| 339 | 24276 | 62383 | 
| 78 | 7 sympy/functions/elementary/trigonometric.py | 2168 | 2218| 457 | 24733 | 62383 | 
| 79 | 7 sympy/functions/elementary/trigonometric.py | 2506 | 2527| 182 | 24915 | 62383 | 
| 80 | 7 sympy/parsing/latex/_parse_latex_antlr.py | 373 | 457| 736 | 25651 | 62383 | 
| 81 | 7 sympy/parsing/latex/_parse_latex_antlr.py | 248 | 280| 215 | 25866 | 62383 | 
| 82 | **7 sympy/printing/latex.py** | 2213 | 2351| 1676 | 27542 | 62383 | 
| 83 | 7 sympy/functions/elementary/trigonometric.py | 2102 | 2148| 388 | 27930 | 62383 | 
| 84 | 7 sympy/functions/elementary/trigonometric.py | 2376 | 2395| 198 | 28128 | 62383 | 
| 85 | 7 sympy/functions/elementary/trigonometric.py | 2726 | 2756| 251 | 28379 | 62383 | 
| 86 | 7 sympy/functions/elementary/trigonometric.py | 2368 | 2374| 132 | 28511 | 62383 | 
| 87 | **7 sympy/printing/latex.py** | 187 | 208| 219 | 28730 | 62383 | 
| 88 | 8 sympy/interactive/printing.py | 85 | 97| 189 | 28919 | 66165 | 
| 89 | 8 sympy/functions/elementary/trigonometric.py | 2758 | 2794| 338 | 29257 | 66165 | 
| 90 | 8 sympy/functions/elementary/trigonometric.py | 2150 | 2166| 170 | 29427 | 66165 | 
| 91 | 8 sympy/functions/elementary/trigonometric.py | 2343 | 2366| 214 | 29641 | 66165 | 
| 92 | 9 sympy/parsing/latex/errors.py | 1 | 3| 0 | 29641 | 66176 | 
| 93 | 9 sympy/parsing/latex/_parse_latex_antlr.py | 329 | 370| 451 | 30092 | 66176 | 
| 94 | 9 sympy/functions/elementary/trigonometric.py | 2537 | 2561| 276 | 30368 | 66176 | 
| 95 | 10 sympy/printing/mathematica.py | 1 | 35| 412 | 30780 | 67396 | 
| 96 | 10 sympy/functions/elementary/trigonometric.py | 2927 | 2963| 359 | 31139 | 67396 | 
| 97 | 10 sympy/functions/elementary/trigonometric.py | 2646 | 2688| 400 | 31539 | 67396 | 
| 98 | 11 sympy/physics/vector/vector.py | 216 | 250| 388 | 31927 | 73332 | 
| 99 | 11 sympy/parsing/latex/_parse_latex_antlr.py | 522 | 555| 222 | 32149 | 73332 | 
| 100 | 11 sympy/functions/elementary/trigonometric.py | 2278 | 2341| 551 | 32700 | 73332 | 
| 101 | 12 sympy/printing/codeprinter.py | 460 | 495| 395 | 33095 | 77488 | 
| 102 | 12 sympy/functions/elementary/trigonometric.py | 2618 | 2644| 218 | 33313 | 77488 | 
| 103 | 12 sympy/parsing/latex/_parse_latex_antlr.py | 218 | 245| 220 | 33533 | 77488 | 
| 104 | 13 sympy/printing/str.py | 205 | 248| 480 | 34013 | 84635 | 
| 105 | 13 sympy/functions/elementary/trigonometric.py | 1994 | 2008| 154 | 34167 | 84635 | 
| 106 | 13 sympy/physics/vector/printing.py | 1 | 13| 127 | 34294 | 84635 | 
| 107 | 13 sympy/interactive/printing.py | 149 | 230| 698 | 34992 | 84635 | 
| 108 | 13 sympy/parsing/latex/_parse_latex_antlr.py | 460 | 504| 334 | 35326 | 84635 | 
| 109 | 13 sympy/functions/elementary/trigonometric.py | 2691 | 2724| 214 | 35540 | 84635 | 
| 110 | 14 sympy/integrals/manualintegrate.py | 317 | 334| 163 | 35703 | 98122 | 
| 111 | 15 sympy/printing/pycode.py | 273 | 292| 200 | 35903 | 102493 | 
| 112 | 15 sympy/functions/elementary/trigonometric.py | 2797 | 2890| 873 | 36776 | 102493 | 
| 113 | 16 sympy/simplify/trigsimp.py | 751 | 791| 748 | 37524 | 114606 | 
| 114 | 16 sympy/simplify/trigsimp.py | 792 | 832| 729 | 38253 | 114606 | 
| 115 | 17 sympy/physics/vector/dyadic.py | 155 | 190| 401 | 38654 | 119060 | 
| 116 | 17 sympy/functions/elementary/trigonometric.py | 2441 | 2504| 546 | 39200 | 119060 | 
| 117 | **17 sympy/printing/latex.py** | 2054 | 2070| 135 | 39335 | 119060 | 
| 118 | 17 sympy/parsing/latex/_parse_latex_antlr.py | 283 | 326| 372 | 39707 | 119060 | 
| 119 | 18 sympy/printing/octave.py | 375 | 457| 719 | 40426 | 125240 | 
| 120 | 19 sympy/functions/elementary/hyperbolic.py | 930 | 962| 273 | 40699 | 136471 | 
| 121 | 19 sympy/integrals/manualintegrate.py | 725 | 750| 445 | 41144 | 136471 | 
| 122 | 19 sympy/parsing/latex/_parse_latex_antlr.py | 126 | 146| 200 | 41344 | 136471 | 
| 123 | 19 sympy/parsing/latex/_parse_latex_antlr.py | 91 | 106| 127 | 41471 | 136471 | 
| 124 | 19 sympy/printing/octave.py | 1 | 54| 579 | 42050 | 136471 | 
| 125 | 20 sympy/printing/printer.py | 1 | 173| 1420 | 43470 | 138800 | 
| 126 | 20 sympy/printing/str.py | 323 | 376| 416 | 43886 | 138800 | 
| 127 | 20 sympy/functions/elementary/trigonometric.py | 2564 | 2616| 430 | 44316 | 138800 | 
| 128 | 20 sympy/interactive/printing.py | 133 | 147| 171 | 44487 | 138800 | 
| 129 | 20 sympy/functions/elementary/trigonometric.py | 1923 | 1992| 617 | 45104 | 138800 | 
| 130 | 20 sympy/functions/elementary/trigonometric.py | 1055 | 1073| 171 | 45275 | 138800 | 
| 131 | 21 sympy/plotting/intervalmath/lib_interval.py | 352 | 387| 283 | 45558 | 142451 | 
| 132 | 21 sympy/functions/elementary/trigonometric.py | 367 | 427| 547 | 46105 | 142451 | 
| 133 | 21 sympy/functions/elementary/trigonometric.py | 2892 | 2925| 327 | 46432 | 142451 | 
| 134 | 21 sympy/integrals/manualintegrate.py | 825 | 884| 702 | 47134 | 142451 | 
| 135 | 21 sympy/printing/glsl.py | 246 | 277| 297 | 47431 | 142451 | 
| 136 | 21 sympy/functions/elementary/trigonometric.py | 650 | 668| 164 | 47595 | 142451 | 
| 137 | 21 sympy/integrals/manualintegrate.py | 769 | 788| 175 | 47770 | 142451 | 
| 138 | 22 sympy/printing/pretty/pretty_symbology.py | 279 | 308| 230 | 48000 | 147725 | 
| 139 | 23 sympy/functions/special/error_functions.py | 1583 | 1595| 199 | 48199 | 167380 | 
| 140 | 24 sympy/parsing/latex/_antlr/__init__.py | 2 | 11| 0 | 48199 | 167485 | 
| 141 | 24 sympy/printing/theanocode.py | 98 | 154| 523 | 48722 | 167485 | 
| 142 | 24 sympy/functions/elementary/trigonometric.py | 670 | 711| 385 | 49107 | 167485 | 
| 143 | 24 sympy/functions/elementary/trigonometric.py | 1 | 20| 230 | 49337 | 167485 | 
| 144 | 25 sympy/printing/fcode.py | 1 | 56| 449 | 49786 | 173236 | 
| 145 | 26 sympy/integrals/transforms.py | 1157 | 1202| 412 | 50198 | 189891 | 
| 146 | 27 sympy/printing/lambdarepr.py | 1 | 17| 129 | 50327 | 191849 | 
| 147 | 28 sympy/core/backend.py | 1 | 24| 357 | 50684 | 192207 | 
| 148 | 28 sympy/printing/pretty/pretty_symbology.py | 237 | 277| 603 | 51287 | 192207 | 
| 149 | 29 sympy/printing/ccode.py | 616 | 624| 119 | 51406 | 199288 | 


## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -740,7 +740,7 @@ def _print_Function(self, expr, exp=None):
                 len(args) == 1 and \
                 not self._needs_function_brackets(expr.args[0])
 
-            inv_trig_table = ["asin", "acos", "atan", "acot"]
+            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]
 
             # If the function is an inverse trig function, handle the style
             if func in inv_trig_table:

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -6,7 +6,7 @@
     Lambda, LaplaceTransform, Limit, Matrix, Max, MellinTransform, Min, Mul,
     Order, Piecewise, Poly, ring, field, ZZ, Pow, Product, Range, Rational,
     RisingFactorial, rootof, RootSum, S, Shi, Si, SineTransform, Subs,
-    Sum, Symbol, ImageSet, Tuple, Union, Ynm, Znm, arg, asin, Mod,
+    Sum, Symbol, ImageSet, Tuple, Union, Ynm, Znm, arg, asin, acsc, Mod,
     assoc_laguerre, assoc_legendre, beta, binomial, catalan, ceiling, Complement,
     chebyshevt, chebyshevu, conjugate, cot, coth, diff, dirichlet_eta, euler,
     exp, expint, factorial, factorial2, floor, gamma, gegenbauer, hermite,
@@ -305,6 +305,8 @@ def test_latex_functions():
     assert latex(asin(x**2), inv_trig_style="power",
                  fold_func_brackets=True) == \
         r"\sin^{-1} {x^{2}}"
+    assert latex(acsc(x), inv_trig_style="full") == \
+        r"\operatorname{arccsc}{\left (x \right )}"
 
     assert latex(factorial(k)) == r"k!"
     assert latex(factorial(-k)) == r"\left(- k\right)!"

```


## Code snippets

### 1 - sympy/printing/latex.py:

Start line: 715, End line: 782

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
            not isinstance(expr.func, UndefinedFunction):
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

            inv_trig_table = ["asin", "acos", "atan", "acot"]

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
                name += r"{\left (%s \right )}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)
```
### 2 - sympy/printing/latex.py:

Start line: 912, End line: 967

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
        arg = r"{\left (%s \right )}" % self._print(expr.args[0])

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
### 3 - sympy/printing/latex.py:

Start line: 1962, End line: 2005

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
### 4 - sympy/physics/vector/printing.py:

Start line: 45, End line: 120

```python
class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t

        if hasattr(self, '_print_' + func) and \
            not isinstance(type(expr), UndefinedFunction):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and (expr.args == (t,)):

            name, supers, subs = split_super_sub(func)
            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]

            if len(supers) != 0:
                supers = r"^{%s}" % "".join(supers)
            else:
                supers = r""

            if len(subs) != 0:
                subs = r"_{%s}" % "".join(subs)
            else:
                subs = r""

            if exp:
                supers += r"^{%s}" % self._print(exp)

            return r"%s" % (name + supers + subs)
        else:
            args = [str(self._print(arg)) for arg in expr.args]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = ["asin", "acos", "atan", "acot"]

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
                name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                name = r"\operatorname{%s}^{%s}" % (func, exp)
            else:
                name = r"\operatorname{%s}" % func

            if can_fold_brackets:
                name += r"%s"
            else:
                name += r"\left(%s\right)"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)
```
### 5 - sympy/printing/latex.py:

Start line: 83, End line: 118

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
### 6 - sympy/printing/latex.py:

Start line: 652, End line: 682

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
                    if self._settings['mode'] in ['equation', 'equation*'] \
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
### 7 - sympy/printing/latex.py:

Start line: 1110, End line: 1125

```python
class LatexPrinter(Printer):

    def _hprint_BesselBase(self, expr, exp, sym):
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, self._print(exp))
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex
```
### 8 - sympy/printing/latex.py:

Start line: 1634, End line: 1642

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
### 9 - sympy/printing/latex.py:

Start line: 1, End line: 82

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.function import _coeff_isneg
from sympy.core.sympify import SympifyError
from sympy.core.alphabets import greeks
from sympy.core.operations import AssocOp
from sympy.core.containers import Tuple
from sympy.logic.boolalg import true
from sympy.core.function import UndefinedFunction, AppliedUndef

## sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from .printer import Printer
from .conventions import split_super_sub, requires_partial
from .precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key, range
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at http://www.mathjax.org/docs/1.1/tex.html#supported-latex-commands
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
### 10 - sympy/printing/latex.py:

Start line: 801, End line: 884

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

        args = (symbols, self._print(expr))
        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _print_Min(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\min\left(%s\right)" % ", ".join(texargs)

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Max(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\max\left(%s\right)" % ", ".join(texargs)

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_floor(self, expr, exp=None):
        tex = r"\lfloor{%s}\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\lceil{%s}\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left (%s \right )}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left (%s \right )}" % self._print(expr.args[0])

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
### 11 - sympy/printing/latex.py:

Start line: 886, End line: 895

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
### 12 - sympy/printing/latex.py:

Start line: 365, End line: 387

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
### 13 - sympy/printing/latex.py:

Start line: 613, End line: 641

```python
class LatexPrinter(Printer):

    def _print_Indexed(self, expr):
        tex = self._print(expr.base)+'_{%s}' % ','.join(
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
### 14 - sympy/printing/latex.py:

Start line: 1644, End line: 1654

```python
class LatexPrinter(Printer):

    def _print_SingularityFunction(self, expr):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\langle %s \rangle}^{%s}" % (shift, power)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        tex = r"\theta\left(%s\right)" % self._print(expr.args[0])
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 15 - sympy/printing/latex.py:

Start line: 489, End line: 540

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1:
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
            #fixes issue #12886, adds parentheses before superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, "%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
            # Things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, self._print(expr.exp))
            else:
                if expr.is_commutative and expr.exp == -1:
                    #solves issue 4129
                    #As Mul always simplify 1/x to x**-1
                    #The objective is achieved with this hack
                    #first we get the latex for -1 * expr,
                    #which is a Mul expression
                    tex = self._print(S.NegativeOne * expr).strip()
                    #the result comes with a minus and a space, so we remove
                    if tex[:1] == "-":
                        return tex[1:].strip()
                tex = r"%s^{%s}"
                #fixes issue #12886, adds parentheses before superscripts raised to powers
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if '^' in base and expr.base.is_Symbol:
                    base = r"\left(%s\right)" % base
                exp = self._print(expr.exp)

                return tex % (base, exp)
```
### 16 - sympy/printing/latex.py:

Start line: 1127, End line: 1192

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
### 17 - sympy/printing/latex.py:

Start line: 1352, End line: 1378

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
### 19 - sympy/printing/latex.py:

Start line: 1656, End line: 1665

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
### 20 - sympy/printing/latex.py:

Start line: 1953, End line: 1960

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
### 21 - sympy/printing/latex.py:

Start line: 1495, End line: 1507

```python
class LatexPrinter(Printer):

    def _print_MatMul(self, expr):
        from sympy import Add, MatAdd, HadamardProduct, MatMul, Mul

        def parens(x):
            if isinstance(x, (Add, MatAdd, HadamardProduct)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)

        if isinstance(expr, MatMul) and expr.args[0].is_Number and expr.args[0]<0:
            expr = Mul(-1*expr.args[0], MatMul(*expr.args[1:]))
            return '-' + ' '.join(map(parens, expr.args))
        else:
            return ' '.join(map(parens, expr.args))
```
### 22 - sympy/printing/latex.py:

Start line: 1735, End line: 1797

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
        return r"\langle %s, %s\rangle" % \
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
### 23 - sympy/printing/latex.py:

Start line: 698, End line: 713

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
### 24 - sympy/printing/latex.py:

Start line: 1812, End line: 1859

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
### 25 - sympy/printing/latex.py:

Start line: 324, End line: 340

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
### 26 - sympy/printing/latex.py:

Start line: 1720, End line: 1733

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
### 27 - sympy/printing/latex.py:

Start line: 969, End line: 978

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
### 28 - sympy/printing/latex.py:

Start line: 389, End line: 438

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
### 29 - sympy/printing/latex.py:

Start line: 1274, End line: 1334

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
### 30 - sympy/printing/latex.py:

Start line: 980, End line: 991

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
### 31 - sympy/printing/latex.py:

Start line: 993, End line: 1075

```python
class LatexPrinter(Printer):

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_gamma(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

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
### 32 - sympy/printing/latex.py:

Start line: 1516, End line: 1546

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
### 33 - sympy/printing/latex.py:

Start line: 2160, End line: 2168

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
### 34 - sympy/printing/latex.py:

Start line: 897, End line: 910

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
### 35 - sympy/printing/latex.py:

Start line: 1509, End line: 1514

```python
class LatexPrinter(Printer):

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s\bmod{%s}\right)^{%s}' % (self.parenthesize(expr.args[0],
                    PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]), self._print(exp))
        return r'%s\bmod{%s}' % (self.parenthesize(expr.args[0],
                PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]))
```
### 36 - sympy/printing/latex.py:

Start line: 1918, End line: 1927

```python
class LatexPrinter(Printer):

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr, index)
```
### 37 - sympy/printing/latex.py:

Start line: 1667, End line: 1703

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
            return self._print(p.sets[0]) + "^%d" % len(p.sets)
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
### 38 - sympy/printing/latex.py:

Start line: 1461, End line: 1493

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

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)

    def _print_MatAdd(self, expr):
        terms = [self._print(t) for t in expr.args]
        l = []
        for t in terms:
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 39 - sympy/printing/latex.py:

Start line: 643, End line: 650

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
### 40 - sympy/printing/latex.py:

Start line: 684, End line: 696

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
### 41 - sympy/printing/latex.py:

Start line: 583, End line: 611

```python
class LatexPrinter(Printer):

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        o1 = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = '(' + LatexPrinter().doprint(v) + ')'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr
```
### 42 - sympy/printing/latex.py:

Start line: 784, End line: 799

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
### 43 - sympy/printing/latex.py:

Start line: 1194, End line: 1203

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
### 44 - sympy/printing/latex.py:

Start line: 2150, End line: 2158

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
### 45 - sympy/printing/latex.py:

Start line: 1705, End line: 1718

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
### 46 - sympy/printing/latex.py:

Start line: 1205, End line: 1216

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
### 47 - sympy/printing/latex.py:

Start line: 1077, End line: 1108

```python
class LatexPrinter(Printer):

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)
```
### 48 - sympy/printing/latex.py:

Start line: 2072, End line: 2136

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{%s}^{%s}' % (self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ %s \right]" % ",".join(
            '{' + self._print(x) + '}' for x in m)

    def _print_SubModule(self, m):
        return r"\left< %s \right>" % ",".join(
            '{' + self._print(x) + '}' for x in m.gens)

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left< %s \right>" % ",".join(
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
### 49 - sympy/printing/latex.py:

Start line: 121, End line: 185

```python
class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings = {
        "order": None,
        "mode": "plain",
        "itex": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "long_frac_ratio": None,
        "mul_symbol": None,
        "inv_trig_style": "abbreviated",
        "mat_str": None,
        "mat_delim": "[",
        "symbol_names": {},
        "ln_notation": False,
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

    def parenthesize(self, item, level, strict=False):
        prec_val = precedence_traditional(item)
        if (prec_val < level) or ((not strict) and prec_val <= level):
            return r"\left(%s\right)" % self._print(item)
        else:
            return self._print(item)
```
### 50 - sympy/printing/latex.py:

Start line: 2183, End line: 2345

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

def latex(expr, **settings):
    # ... other code
```
### 51 - sympy/printing/latex.py:

Start line: 1218, End line: 1272

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
### 52 - sympy/printing/latex.py:

Start line: 564, End line: 581

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
### 53 - sympy/printing/latex.py:

Start line: 1929, End line: 1951

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
### 54 - sympy/printing/latex.py:

Start line: 2138, End line: 2148

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
### 55 - sympy/printing/latex.py:

Start line: 342, End line: 363

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
### 56 - sympy/printing/latex.py:

Start line: 1380, End line: 1398

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
### 57 - sympy/printing/latex.py:

Start line: 1413, End line: 1438

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
### 58 - sympy/printing/latex.py:

Start line: 302, End line: 322

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
### 59 - sympy/printing/latex.py:

Start line: 1439, End line: 1459

```python
class LatexPrinter(Printer):
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix \
                           = _print_Matrix \
                           = _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '_{%s, %s}' % (expr.i, expr.j)

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
### 61 - sympy/printing/latex.py:

Start line: 440, End line: 487

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
### 62 - sympy/printing/latex.py:

Start line: 1400, End line: 1411

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
### 63 - sympy/printing/latex.py:

Start line: 2170, End line: 2180

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
### 64 - sympy/printing/latex.py:

Start line: 2007, End line: 2052

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
### 65 - sympy/printing/latex.py:

Start line: 1336, End line: 1350

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
### 67 - sympy/printing/latex.py:

Start line: 1548, End line: 1595

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
### 69 - sympy/printing/latex.py:

Start line: 1597, End line: 1632

```python
class LatexPrinter(Printer):

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_MutableSparseNDimArray = _print_NDimArray

    def _print_tuple(self, expr):
        return r"\left ( %s\right )" % \
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
        return r"\left [ %s\right ]" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left \{ %s\right \}" % r", \quad ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)
```
### 70 - sympy/printing/latex.py:

Start line: 1861, End line: 1916

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
            tex = r"\%s {\left (%s \right )}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex
```
### 73 - sympy/printing/latex.py:

Start line: 1799, End line: 1810

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
### 76 - sympy/printing/latex.py:

Start line: 542, End line: 562

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
### 82 - sympy/printing/latex.py:

Start line: 2213, End line: 2351

```python
def latex(expr, **settings):
    r"""
    Convert the given expression to LaTeX representation.

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'

    order: Any of the supported monomial orderings (currently "lex", "grlex", or
    "grevlex"), "old", and "none". This parameter does nothing for Mul objects.
    Setting order to "old" uses the compatibility ordering for Add defined in
    Printer. For very large expressions, set the 'order' keyword to 'none' if
    speed is a concern.

    mode: Specifies how the generated code will be delimited. 'mode' can be one
    of 'plain', 'inline', 'equation' or 'equation*'.  If 'mode' is set to
    'plain', then the resulting code will not be delimited at all (this is the
    default). If 'mode' is set to 'inline' then inline LaTeX $ $ will be used.
    If 'mode' is set to 'equation' or 'equation*', the resulting code will be
    enclosed in the 'equation' or 'equation*' environment (remember to import
    'amsmath' for 'equation*'), unless the 'itex' option is set. In the latter
    case, the ``$$ $$`` syntax is used.

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}

    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$

    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}

    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}

    itex: Specifies if itex-specific syntax is used, including emitting ``$$ $$``.

    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    fold_frac_powers: Emit "^{p/q}" instead of "^{\frac{p}{q}}" for fractional
    powers.

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}

    fold_func_brackets: Fold function brackets where applicable.

    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left (\frac{7}{2} \right )}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets = True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}

    fold_short_frac: Emit "p / q" instead of "\frac{p}{q}" when the
    denominator is simple enough (at most two terms and no powers).
    The default value is `True` for inline mode, False otherwise.

    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y

    long_frac_ratio: The allowed ratio of the width of the numerator to the
    width of the denominator before we start breaking off long fractions.
    If None (the default value), long fractions are not broken up.

    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    mul_symbol: The symbol to use for multiplication. Can be one of None,
    "ldot", "dot", or "times".

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left (\frac{7}{2} \right )}}

    inv_trig_style: How inverse trig functions should be displayed. Can be one
    of "abbreviated", "full", or "power". Defaults to "abbreviated".

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left (\frac{7}{2} \right )}

    mat_str: Which matrix environment string to emit. "smallmatrix", "matrix",
    "array", etc. Defaults to "smallmatrix" for inline mode, "matrix" for
    matrices of no more than 10 columns, and "array" otherwise.

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]

    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]

    mat_delim: The delimiter to wrap around matrices. Can be one of "[", "(",
    or the empty string. Defaults to "[".

    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    symbol_names: Dictionary of symbols and the custom strings they should be
    emitted as.

    >>> print(latex(x**2, symbol_names={x:'x_i'}))
    x_i^{2}

    ``latex`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left [ 2 / x, \quad y\right ]$

    ln_notation: If set to ``True`` "\ln" is used instead of default "\log"

    >>> print(latex(log(10)))
    \log{\left (10 \right )}

    >>> print(latex(log(10), ln_notation=True))
    \ln{\left (10 \right )}

    """

    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression."""
    print(latex(expr, **settings))
```
### 87 - sympy/printing/latex.py:

Start line: 187, End line: 208

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
### 117 - sympy/printing/latex.py:

Start line: 2054, End line: 2070

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
