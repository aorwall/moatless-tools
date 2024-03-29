# sympy__sympy-18477

| **sympy/sympy** | `93d836fcdb38c6b3235f785adc45b34eb2a64a9e` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 45698 |
| **Any found context length** | 411 |
| **Avg pos** | 66.6 |
| **Min pos** | 2 |
| **Max pos** | 149 |
| **Top file pos** | 1 |
| **Missing snippets** | 13 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/printing/glsl.py b/sympy/printing/glsl.py
--- a/sympy/printing/glsl.py
+++ b/sympy/printing/glsl.py
@@ -53,7 +53,7 @@ class GLSLPrinter(CodePrinter):
         'allow_unknown_functions': False,
         'contract': True,
         'error_on_reserved': False,
-        'reserved_word_suffix': '_'
+        'reserved_word_suffix': '_',
     }
 
     def __init__(self, settings={}):
diff --git a/sympy/printing/jscode.py b/sympy/printing/jscode.py
--- a/sympy/printing/jscode.py
+++ b/sympy/printing/jscode.py
@@ -57,7 +57,7 @@ class JavascriptCodePrinter(CodePrinter):
         'user_functions': {},
         'human': True,
         'allow_unknown_functions': False,
-        'contract': True
+        'contract': True,
     }  # type: Dict[str, Any]
 
     def __init__(self, settings={}):
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -125,6 +125,7 @@ class LatexPrinter(Printer):
     printmethod = "_latex"
 
     _default_settings = {
+        "full_prec": False,
         "fold_frac_powers": False,
         "fold_func_brackets": False,
         "fold_short_frac": None,
@@ -144,6 +145,8 @@ class LatexPrinter(Printer):
         "gothic_re_im": False,
         "decimal_separator": "period",
         "perm_cyclic": True,
+        "min": None,
+        "max": None,
     }  # type: Dict[str, Any]
 
     def __init__(self, settings=None):
@@ -414,7 +417,10 @@ def _print_AppliedPermutation(self, expr):
     def _print_Float(self, expr):
         # Based off of that in StrPrinter
         dps = prec_to_dps(expr._prec)
-        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)
+        strip = False if self._settings['full_prec'] else True
+        low = self._settings["min"] if "min" in self._settings else None
+        high = self._settings["max"] if "max" in self._settings else None
+        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
 
         # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
         # thus we use the number separator
@@ -2550,8 +2556,8 @@ def translate(s):
         return s
 
 
-def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
-          fold_short_frac=None, inv_trig_style="abbreviated",
+def latex(expr, full_prec=False, min=None, max=None, fold_frac_powers=False,
+          fold_func_brackets=False, fold_short_frac=None, inv_trig_style="abbreviated",
           itex=False, ln_notation=False, long_frac_ratio=None,
           mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
           order=None, symbol_names=None, root_notation=True,
@@ -2561,6 +2567,8 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
 
     Parameters
     ==========
+    full_prec: boolean, optional
+        If set to True, a floating point number is printed with full precision.
     fold_frac_powers : boolean, optional
         Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
     fold_func_brackets : boolean, optional
@@ -2628,6 +2636,12 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
         separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
         ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
+    min: Integer or None, optional
+        Sets the lower bound for the exponent to print floating point numbers in
+        fixed-point format.
+    max: Integer or None, optional
+        Sets the upper bound for the exponent to print floating point numbers in
+        fixed-point format.
 
     Notes
     =====
@@ -2739,6 +2753,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         symbol_names = {}
 
     settings = {
+        'full_prec': full_prec,
         'fold_frac_powers': fold_frac_powers,
         'fold_func_brackets': fold_func_brackets,
         'fold_short_frac': fold_short_frac,
@@ -2758,6 +2773,8 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         'gothic_re_im': gothic_re_im,
         'decimal_separator': decimal_separator,
         'perm_cyclic' : perm_cyclic,
+        'min': min,
+        'max': max,
     }
 
     return LatexPrinter(settings).doprint(expr)
diff --git a/sympy/printing/pycode.py b/sympy/printing/pycode.py
--- a/sympy/printing/pycode.py
+++ b/sympy/printing/pycode.py
@@ -92,7 +92,7 @@ class AbstractPythonCodePrinter(CodePrinter):
         inline=True,
         fully_qualified_modules=True,
         contract=False,
-        standard='python3'
+        standard='python3',
     )
 
     def __init__(self, settings=None):
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -24,6 +24,8 @@ class StrPrinter(Printer):
         "sympy_integers": False,
         "abbrev": False,
         "perm_cyclic": True,
+        "min": None,
+        "max": None,
     }  # type: Dict[str, Any]
 
     _relationals = dict()  # type: Dict[str, str]
@@ -691,7 +693,9 @@ def _print_Float(self, expr):
             strip = True
         elif self._settings["full_prec"] == "auto":
             strip = self._print_level > 1
-        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip)
+        low = self._settings["min"] if "min" in self._settings else None
+        high = self._settings["max"] if "max" in self._settings else None
+        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
         if rv.startswith('-.0'):
             rv = '-0.' + rv[3:]
         elif rv.startswith('.0'):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/glsl.py | 56 | 56 | 149 | 21 | 54306
| sympy/printing/jscode.py | 60 | 60 | - | - | -
| sympy/printing/latex.py | 128 | 128 | 32 | 4 | 14208
| sympy/printing/latex.py | 147 | 147 | 32 | 4 | 14208
| sympy/printing/latex.py | 417 | 417 | 5 | 4 | 1291
| sympy/printing/latex.py | 2553 | 2554 | - | 4 | -
| sympy/printing/latex.py | 2564 | 2564 | - | 4 | -
| sympy/printing/latex.py | 2631 | 2631 | - | 4 | -
| sympy/printing/latex.py | 2742 | 2742 | - | 4 | -
| sympy/printing/latex.py | 2761 | 2761 | - | 4 | -
| sympy/printing/pycode.py | 95 | 95 | - | 1 | -
| sympy/printing/str.py | 27 | 27 | 113 | 2 | 45698
| sympy/printing/str.py | 694 | 694 | 2 | 2 | 411


## Problem Statement

```
Allow to set min_fixed and max_fixed for Float in the printers
The mpmath printer has `min_fixed` and `max_fixed` settings, which should be exposed to the printers. Right now, only the `strip_zeros` option is exposed. 

We should also unify the Float printer for the various printers. For example, the LaTeX printer doesn't have the same behavior as the string printer. 


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/pycode.py** | 513 | 534| 222 | 222 | 8784 | 
| **-> 2 <-** | **2 sympy/printing/str.py** | 682 | 702| 189 | 411 | 16284 | 
| 3 | **2 sympy/printing/pycode.py** | 572 | 599| 256 | 667 | 16284 | 
| 4 | 3 sympy/printing/mathml.py | 1169 | 1206| 333 | 1000 | 33118 | 
| **-> 5 <-** | **4 sympy/printing/latex.py** | 410 | 439| 291 | 1291 | 61232 | 
| 6 | 5 sympy/polys/numberfields.py | 1071 | 1084| 147 | 1438 | 70396 | 
| 7 | **5 sympy/printing/str.py** | 201 | 260| 554 | 1992 | 70396 | 
| 8 | 6 sympy/codegen/ast.py | 1153 | 1238| 682 | 2674 | 83911 | 
| 9 | 7 sympy/printing/mathematica.py | 310 | 335| 295 | 2969 | 87604 | 
| 10 | **7 sympy/printing/latex.py** | 1431 | 1460| 350 | 3319 | 87604 | 
| 11 | **7 sympy/printing/latex.py** | 1938 | 1953| 147 | 3466 | 87604 | 
| 12 | **7 sympy/printing/latex.py** | 1829 | 1858| 308 | 3774 | 87604 | 
| 13 | **7 sympy/printing/latex.py** | 2270 | 2280| 124 | 3898 | 87604 | 
| 14 | **7 sympy/printing/latex.py** | 2046 | 2118| 630 | 4528 | 87604 | 
| 15 | **7 sympy/printing/latex.py** | 1009 | 1064| 577 | 5105 | 87604 | 
| 16 | 8 sympy/core/numbers.py | 1143 | 1192| 522 | 5627 | 117291 | 
| 17 | **8 sympy/printing/latex.py** | 901 | 981| 767 | 6394 | 117291 | 
| 18 | **8 sympy/printing/latex.py** | 2291 | 2364| 735 | 7129 | 117291 | 
| 19 | **8 sympy/printing/str.py** | 607 | 680| 543 | 7672 | 117291 | 
| 20 | **8 sympy/printing/latex.py** | 521 | 568| 529 | 8201 | 117291 | 
| 21 | **8 sympy/printing/latex.py** | 570 | 611| 458 | 8659 | 117291 | 
| 22 | **8 sympy/printing/latex.py** | 1090 | 1174| 813 | 9472 | 117291 | 
| 23 | 8 sympy/core/numbers.py | 1211 | 1308| 777 | 10249 | 117291 | 
| 24 | **8 sympy/printing/latex.py** | 469 | 519| 385 | 10634 | 117291 | 
| 25 | **8 sympy/printing/pycode.py** | 493 | 510| 176 | 10810 | 117291 | 
| 26 | **8 sympy/printing/latex.py** | 1 | 85| 725 | 11535 | 117291 | 
| 27 | **8 sympy/printing/latex.py** | 778 | 790| 159 | 11694 | 117291 | 
| 28 | **8 sympy/printing/latex.py** | 2410 | 2476| 734 | 12428 | 117291 | 
| 29 | **8 sympy/printing/str.py** | 335 | 372| 284 | 12712 | 117291 | 
| 30 | **8 sympy/printing/pycode.py** | 537 | 569| 333 | 13045 | 117291 | 
| 31 | 9 sympy/printing/codeprinter.py | 504 | 539| 400 | 13445 | 121837 | 
| **-> 32 <-** | **9 sympy/printing/latex.py** | 124 | 216| 763 | 14208 | 121837 | 
| 33 | **9 sympy/printing/latex.py** | 2127 | 2137| 129 | 14337 | 121837 | 
| 34 | **9 sympy/printing/latex.py** | 441 | 467| 306 | 14643 | 121837 | 
| 35 | 9 sympy/printing/mathematica.py | 118 | 229| 788 | 15431 | 121837 | 
| 36 | **9 sympy/printing/str.py** | 751 | 830| 650 | 16081 | 121837 | 
| 37 | 10 sympy/printing/printer.py | 1 | 173| 1419 | 17500 | 124300 | 
| 38 | **10 sympy/printing/str.py** | 182 | 199| 153 | 17653 | 124300 | 
| 39 | 10 sympy/printing/mathematica.py | 298 | 308| 165 | 17818 | 124300 | 
| 40 | **10 sympy/printing/pycode.py** | 871 | 905| 315 | 18133 | 124300 | 
| 41 | **10 sympy/printing/str.py** | 279 | 333| 501 | 18634 | 124300 | 
| 42 | **10 sympy/printing/latex.py** | 1614 | 1633| 167 | 18801 | 124300 | 
| 43 | **10 sympy/printing/latex.py** | 744 | 776| 325 | 19126 | 124300 | 
| 44 | **10 sympy/printing/str.py** | 470 | 543| 529 | 19655 | 124300 | 
| 45 | **10 sympy/printing/str.py** | 167 | 180| 140 | 19795 | 124300 | 
| 46 | 10 sympy/printing/printer.py | 175 | 251| 490 | 20285 | 124300 | 
| 47 | 10 sympy/core/numbers.py | 1097 | 1141| 423 | 20708 | 124300 | 
| 48 | 11 sympy/printing/ccode.py | 511 | 530| 193 | 20901 | 132345 | 
| 49 | 11 sympy/core/numbers.py | 858 | 1020| 1445 | 22346 | 132345 | 
| 50 | 12 sympy/printing/llvmjitcode.py | 82 | 110| 267 | 22613 | 136380 | 
| 51 | 12 sympy/printing/ccode.py | 428 | 439| 151 | 22764 | 136380 | 
| 52 | 13 sympy/printing/rust.py | 57 | 162| 1085 | 23849 | 141857 | 
| 53 | **13 sympy/printing/latex.py** | 809 | 876| 656 | 24505 | 141857 | 
| 54 | **13 sympy/printing/latex.py** | 792 | 807| 161 | 24666 | 141857 | 
| 55 | **13 sympy/printing/latex.py** | 1226 | 1291| 668 | 25334 | 141857 | 
| 56 | 13 sympy/printing/mathml.py | 882 | 918| 381 | 25715 | 141857 | 
| 57 | **13 sympy/printing/latex.py** | 2478 | 2488| 163 | 25878 | 141857 | 
| 58 | 13 sympy/core/numbers.py | 1382 | 1420| 333 | 26211 | 141857 | 
| 59 | **13 sympy/printing/latex.py** | 1635 | 1643| 120 | 26331 | 141857 | 
| 60 | **13 sympy/printing/str.py** | 848 | 879| 287 | 26618 | 141857 | 
| 61 | **13 sympy/printing/latex.py** | 2500 | 2508| 129 | 26747 | 141857 | 
| 62 | 13 sympy/core/numbers.py | 1021 | 1095| 653 | 27400 | 141857 | 
| 63 | **13 sympy/printing/latex.py** | 2490 | 2498| 123 | 27523 | 141857 | 
| 64 | **13 sympy/printing/latex.py** | 1375 | 1429| 729 | 28252 | 141857 | 
| 65 | **13 sympy/printing/pycode.py** | 740 | 801| 769 | 29021 | 141857 | 
| 66 | 13 sympy/printing/mathml.py | 367 | 394| 259 | 29280 | 141857 | 
| 67 | 13 sympy/printing/mathml.py | 22 | 75| 407 | 29687 | 141857 | 
| 68 | **13 sympy/printing/latex.py** | 86 | 121| 491 | 30178 | 141857 | 
| 69 | 13 sympy/printing/mathml.py | 1486 | 1531| 397 | 30575 | 141857 | 
| 70 | **13 sympy/printing/latex.py** | 1955 | 1979| 223 | 30798 | 141857 | 
| 71 | 13 sympy/printing/ccode.py | 415 | 426| 151 | 30949 | 141857 | 
| 72 | 13 sympy/printing/mathml.py | 752 | 777| 211 | 31160 | 141857 | 
| 73 | **13 sympy/printing/latex.py** | 878 | 899| 209 | 31369 | 141857 | 
| 74 | 13 sympy/printing/mathml.py | 313 | 365| 422 | 31791 | 141857 | 
| 75 | 13 sympy/core/numbers.py | 1194 | 1209| 165 | 31956 | 141857 | 
| 76 | 14 sympy/printing/cxxcode.py | 79 | 108| 285 | 32241 | 143480 | 
| 77 | **14 sympy/printing/latex.py** | 1317 | 1373| 751 | 32992 | 143480 | 
| 78 | **14 sympy/printing/latex.py** | 2510 | 2520| 161 | 33153 | 143480 | 
| 79 | 14 sympy/printing/codeprinter.py | 400 | 451| 507 | 33660 | 143480 | 
| 80 | 14 sympy/printing/mathml.py | 1 | 19| 145 | 33805 | 143480 | 
| 81 | **14 sympy/printing/latex.py** | 1995 | 2009| 187 | 33992 | 143480 | 
| 82 | **14 sympy/printing/latex.py** | 699 | 732| 288 | 34280 | 143480 | 
| 83 | 15 sympy/printing/repr.py | 207 | 222| 223 | 34503 | 146259 | 
| 84 | **15 sympy/printing/latex.py** | 2139 | 2186| 457 | 34960 | 146259 | 
| 85 | 16 sympy/printing/pretty/pretty.py | 2238 | 2346| 799 | 35759 | 169709 | 
| 86 | **16 sympy/printing/latex.py** | 1697 | 1747| 495 | 36254 | 169709 | 
| 87 | 16 sympy/printing/mathml.py | 1454 | 1484| 262 | 36516 | 169709 | 
| 88 | **16 sympy/printing/str.py** | 412 | 468| 541 | 37057 | 169709 | 
| 89 | 17 sympy/printing/fcode.py | 656 | 667| 124 | 37181 | 177702 | 
| 90 | **17 sympy/printing/latex.py** | 2188 | 2243| 414 | 37595 | 177702 | 
| 91 | 17 sympy/printing/mathematica.py | 1 | 115| 80 | 37675 | 177702 | 
| 92 | 17 sympy/printing/codeprinter.py | 453 | 502| 449 | 38124 | 177702 | 
| 93 | 17 sympy/printing/mathml.py | 1551 | 1644| 838 | 38962 | 177702 | 
| 94 | 17 sympy/printing/mathml.py | 1265 | 1345| 698 | 39660 | 177702 | 
| 95 | 17 sympy/printing/pretty/pretty.py | 136 | 203| 639 | 40299 | 177702 | 
| 96 | 17 sympy/printing/mathml.py | 1040 | 1092| 472 | 40771 | 177702 | 
| 97 | 17 sympy/printing/ccode.py | 122 | 138| 169 | 40940 | 177702 | 
| 98 | 17 sympy/core/numbers.py | 1422 | 1450| 304 | 41244 | 177702 | 
| 99 | **17 sympy/printing/latex.py** | 1860 | 1882| 213 | 41457 | 177702 | 
| 100 | 17 sympy/printing/fcode.py | 300 | 321| 253 | 41710 | 177702 | 
| 101 | **17 sympy/printing/latex.py** | 1077 | 1088| 156 | 41866 | 177702 | 
| 102 | **17 sympy/printing/latex.py** | 2282 | 2289| 118 | 41984 | 177702 | 
| 103 | 17 sympy/printing/repr.py | 272 | 279| 121 | 42105 | 177702 | 
| 104 | 17 sympy/printing/mathml.py | 801 | 879| 630 | 42735 | 177702 | 
| 105 | **17 sympy/printing/str.py** | 72 | 165| 823 | 43558 | 177702 | 
| 106 | 17 sympy/printing/ccode.py | 717 | 735| 178 | 43736 | 177702 | 
| 107 | 17 sympy/printing/mathml.py | 779 | 799| 172 | 43908 | 177702 | 
| 108 | 17 sympy/core/numbers.py | 1452 | 1502| 363 | 44271 | 177702 | 
| 109 | **17 sympy/printing/latex.py** | 380 | 407| 267 | 44538 | 177702 | 
| 110 | 17 sympy/printing/mathml.py | 1241 | 1263| 182 | 44720 | 177702 | 
| 111 | 17 sympy/printing/mathml.py | 77 | 126| 503 | 45223 | 177702 | 
| 112 | 17 sympy/printing/mathml.py | 1533 | 1549| 155 | 45378 | 177702 | 
| **-> 113 <-** | **17 sympy/printing/str.py** | 1 | 46| 320 | 45698 | 177702 | 
| 114 | 17 sympy/printing/pretty/pretty.py | 1944 | 1971| 251 | 45949 | 177702 | 
| 115 | 17 sympy/core/numbers.py | 1326 | 1348| 282 | 46231 | 177702 | 
| 116 | **17 sympy/printing/latex.py** | 1884 | 1892| 135 | 46366 | 177702 | 
| 117 | 17 sympy/printing/pretty/pretty.py | 1416 | 1437| 231 | 46597 | 177702 | 
| 118 | 17 sympy/printing/mathml.py | 1094 | 1153| 431 | 47028 | 177702 | 
| 119 | 18 sympy/printing/tensorflow.py | 1 | 107| 878 | 47906 | 180154 | 
| 120 | 18 sympy/printing/pretty/pretty.py | 664 | 690| 261 | 48167 | 180154 | 
| 121 | **18 sympy/printing/latex.py** | 2012 | 2022| 127 | 48294 | 180154 | 
| 122 | 18 sympy/printing/cxxcode.py | 140 | 156| 130 | 48424 | 180154 | 
| 123 | **18 sympy/printing/latex.py** | 1209 | 1224| 138 | 48562 | 180154 | 
| 124 | **18 sympy/printing/latex.py** | 613 | 629| 199 | 48761 | 180154 | 
| 125 | 19 sympy/printing/julia.py | 221 | 263| 290 | 49051 | 186056 | 
| 126 | 19 sympy/printing/mathml.py | 920 | 949| 239 | 49290 | 186056 | 
| 127 | **19 sympy/printing/latex.py** | 1304 | 1315| 187 | 49477 | 186056 | 
| 128 | **19 sympy/printing/latex.py** | 1894 | 1904| 141 | 49618 | 186056 | 
| 129 | **19 sympy/printing/pycode.py** | 907 | 964| 549 | 50167 | 186056 | 
| 130 | 19 sympy/printing/cxxcode.py | 111 | 138| 285 | 50452 | 186056 | 
| 131 | 19 sympy/printing/ccode.py | 622 | 653| 314 | 50766 | 186056 | 
| 132 | **19 sympy/printing/latex.py** | 2245 | 2255| 126 | 50892 | 186056 | 
| 133 | 19 sympy/printing/mathml.py | 714 | 732| 160 | 51052 | 186056 | 
| 134 | **19 sympy/printing/latex.py** | 669 | 697| 239 | 51291 | 186056 | 
| 135 | 19 sympy/printing/mathematica.py | 231 | 250| 173 | 51464 | 186056 | 
| 136 | 20 sympy/printing/lambdarepr.py | 1 | 17| 108 | 51572 | 187530 | 
| 137 | 20 sympy/printing/pretty/pretty.py | 1687 | 1704| 173 | 51745 | 187530 | 
| 138 | 20 sympy/printing/fcode.py | 323 | 337| 131 | 51876 | 187530 | 
| 139 | **20 sympy/printing/latex.py** | 1917 | 1936| 228 | 52104 | 187530 | 
| 140 | **20 sympy/printing/pycode.py** | 224 | 238| 127 | 52231 | 187530 | 
| 141 | **20 sympy/printing/latex.py** | 2257 | 2268| 118 | 52349 | 187530 | 
| 142 | 20 sympy/printing/mathml.py | 1155 | 1167| 125 | 52474 | 187530 | 
| 143 | **20 sympy/printing/latex.py** | 1293 | 1302| 138 | 52612 | 187530 | 
| 144 | 20 sympy/printing/pretty/pretty.py | 267 | 333| 583 | 53195 | 187530 | 
| 145 | **20 sympy/printing/latex.py** | 631 | 648| 196 | 53391 | 187530 | 
| 146 | 20 sympy/printing/mathml.py | 202 | 235| 260 | 53651 | 187530 | 
| 147 | 20 sympy/core/numbers.py | 1310 | 1324| 179 | 53830 | 187530 | 
| 148 | **20 sympy/printing/latex.py** | 983 | 992| 140 | 53970 | 187530 | 
| **-> 149 <-** | **21 sympy/printing/glsl.py** | 30 | 78| 336 | 54306 | 192396 | 
| 150 | **21 sympy/printing/latex.py** | 1816 | 1827| 142 | 54448 | 192396 | 
| 151 | 21 sympy/printing/codeprinter.py | 373 | 398| 291 | 54739 | 192396 | 
| 152 | 21 sympy/printing/mathml.py | 667 | 712| 383 | 55122 | 192396 | 
| 153 | **21 sympy/printing/latex.py** | 1782 | 1814| 281 | 55403 | 192396 | 
| 154 | 21 sympy/printing/mathml.py | 1646 | 1670| 223 | 55626 | 192396 | 
| 155 | **21 sympy/printing/latex.py** | 734 | 742| 128 | 55754 | 192396 | 
| 156 | 21 sympy/printing/mathml.py | 237 | 263| 226 | 55980 | 192396 | 
| 157 | **21 sympy/printing/latex.py** | 1176 | 1207| 308 | 56288 | 192396 | 
| 158 | **21 sympy/printing/latex.py** | 994 | 1007| 139 | 56427 | 192396 | 
| 159 | **21 sympy/printing/pycode.py** | 602 | 639| 375 | 56802 | 192396 | 
| 160 | 22 sympy/printing/__init__.py | 1 | 120| 655 | 57457 | 193051 | 
| 161 | 22 sympy/printing/mathml.py | 1409 | 1452| 383 | 57840 | 193051 | 
| 162 | **22 sympy/printing/latex.py** | 1510 | 1528| 135 | 57975 | 193051 | 
| 163 | **22 sympy/printing/latex.py** | 1543 | 1568| 271 | 58246 | 193051 | 
| 164 | 22 sympy/printing/mathml.py | 265 | 284| 201 | 58447 | 193051 | 
| 165 | **22 sympy/printing/latex.py** | 2024 | 2044| 220 | 58667 | 193051 | 
| 166 | 22 sympy/printing/pretty/pretty.py | 31 | 92| 542 | 59209 | 193051 | 
| 167 | **22 sympy/printing/latex.py** | 1749 | 1780| 281 | 59490 | 193051 | 
| 168 | 22 sympy/printing/mathml.py | 734 | 750| 155 | 59645 | 193051 | 
| 169 | 23 sympy/printing/python.py | 1 | 42| 325 | 59970 | 193735 | 
| 170 | 23 sympy/printing/pretty/pretty.py | 612 | 662| 422 | 60392 | 193735 | 
| 171 | 23 sympy/printing/mathml.py | 1777 | 1804| 225 | 60617 | 193735 | 
| 172 | 24 sympy/printing/maple.py | 226 | 240| 197 | 60814 | 196464 | 
| 173 | 24 sympy/printing/julia.py | 121 | 193| 697 | 61511 | 196464 | 
| 174 | 24 sympy/printing/mathml.py | 507 | 572| 510 | 62021 | 196464 | 
| 175 | 24 sympy/printing/mathml.py | 1691 | 1775| 658 | 62679 | 196464 | 
| 176 | 24 sympy/printing/fcode.py | 1 | 64| 494 | 63173 | 196464 | 
| 177 | **24 sympy/printing/latex.py** | 1645 | 1695| 482 | 63655 | 196464 | 
| 178 | 24 sympy/printing/mathml.py | 396 | 419| 205 | 63860 | 196464 | 


## Missing Patch Files

 * 1: sympy/printing/glsl.py
 * 2: sympy/printing/jscode.py
 * 3: sympy/printing/latex.py
 * 4: sympy/printing/pycode.py
 * 5: sympy/printing/str.py

### Hint

```
Related: http://stackoverflow.com/q/25222681/161801

To fix issue #7847 I have done the following changes to the code in sympy/sympy/printing/str.py
Will this work?

\`\`\` python
class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "min": None,
        "max": None,
    }
\`\`\`

\`\`\` python
def _print_Float(self, expr):
        prec = expr._prec
        low = self._settings["min"]
        high = self._settings["max"]
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec) 
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        if low is None:
            low = min(-(dps//3), -5)
        if high is None:
            high = dps
        rv = mlib.to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        return rv
\`\`\`

@MridulS You should send a pull request for your changes. 

@hargup I just wanted to confirm whether this will work or not.

> I just wanted to confirm whether this will work or not

We can see that at the Pull Request, there we can see the results of the travis build. Also the reviewers will be able to directly pull your changes on the local into their local system. If you have made the changes sending the pull request should not be much work.

@MridulS  Hello the bug has been fixed? 
If not, would like to know more information to resolve this bug. You need help to implement anything else?

@lohmanndouglas You could add @MridulS's github fork of sympy as a remote and pull down his branch with his fix in https://github.com/sympy/sympy/issues/7847. Then play with it and see if it works.

@moorepants unfortunately i have deleted that branch. @lohmanndouglas you can still see the changes I tried at https://github.com/sympy/sympy/commit/61838749a78082453be4e779cb68e88605d49244

Is this issue fixed? I would like to take this on.
@Yathartha22 I think this is still open, if you're still interested --- though it's been a while...

The [SO question that (partially) prompted this has over 1000 views](https://stackoverflow.com/questions/25222681/scientific-exponential-notation-with-sympy-in-an-ipython-notebook) now.
Sure I will take this up.
Is this issue fixed? If not I'd like to work on it.
I don't think so. I don't see any cross-referenced pull requests listed here. 
```

## Patch

```diff
diff --git a/sympy/printing/glsl.py b/sympy/printing/glsl.py
--- a/sympy/printing/glsl.py
+++ b/sympy/printing/glsl.py
@@ -53,7 +53,7 @@ class GLSLPrinter(CodePrinter):
         'allow_unknown_functions': False,
         'contract': True,
         'error_on_reserved': False,
-        'reserved_word_suffix': '_'
+        'reserved_word_suffix': '_',
     }
 
     def __init__(self, settings={}):
diff --git a/sympy/printing/jscode.py b/sympy/printing/jscode.py
--- a/sympy/printing/jscode.py
+++ b/sympy/printing/jscode.py
@@ -57,7 +57,7 @@ class JavascriptCodePrinter(CodePrinter):
         'user_functions': {},
         'human': True,
         'allow_unknown_functions': False,
-        'contract': True
+        'contract': True,
     }  # type: Dict[str, Any]
 
     def __init__(self, settings={}):
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -125,6 +125,7 @@ class LatexPrinter(Printer):
     printmethod = "_latex"
 
     _default_settings = {
+        "full_prec": False,
         "fold_frac_powers": False,
         "fold_func_brackets": False,
         "fold_short_frac": None,
@@ -144,6 +145,8 @@ class LatexPrinter(Printer):
         "gothic_re_im": False,
         "decimal_separator": "period",
         "perm_cyclic": True,
+        "min": None,
+        "max": None,
     }  # type: Dict[str, Any]
 
     def __init__(self, settings=None):
@@ -414,7 +417,10 @@ def _print_AppliedPermutation(self, expr):
     def _print_Float(self, expr):
         # Based off of that in StrPrinter
         dps = prec_to_dps(expr._prec)
-        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)
+        strip = False if self._settings['full_prec'] else True
+        low = self._settings["min"] if "min" in self._settings else None
+        high = self._settings["max"] if "max" in self._settings else None
+        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
 
         # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
         # thus we use the number separator
@@ -2550,8 +2556,8 @@ def translate(s):
         return s
 
 
-def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
-          fold_short_frac=None, inv_trig_style="abbreviated",
+def latex(expr, full_prec=False, min=None, max=None, fold_frac_powers=False,
+          fold_func_brackets=False, fold_short_frac=None, inv_trig_style="abbreviated",
           itex=False, ln_notation=False, long_frac_ratio=None,
           mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
           order=None, symbol_names=None, root_notation=True,
@@ -2561,6 +2567,8 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
 
     Parameters
     ==========
+    full_prec: boolean, optional
+        If set to True, a floating point number is printed with full precision.
     fold_frac_powers : boolean, optional
         Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
     fold_func_brackets : boolean, optional
@@ -2628,6 +2636,12 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
         separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
         ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
+    min: Integer or None, optional
+        Sets the lower bound for the exponent to print floating point numbers in
+        fixed-point format.
+    max: Integer or None, optional
+        Sets the upper bound for the exponent to print floating point numbers in
+        fixed-point format.
 
     Notes
     =====
@@ -2739,6 +2753,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         symbol_names = {}
 
     settings = {
+        'full_prec': full_prec,
         'fold_frac_powers': fold_frac_powers,
         'fold_func_brackets': fold_func_brackets,
         'fold_short_frac': fold_short_frac,
@@ -2758,6 +2773,8 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         'gothic_re_im': gothic_re_im,
         'decimal_separator': decimal_separator,
         'perm_cyclic' : perm_cyclic,
+        'min': min,
+        'max': max,
     }
 
     return LatexPrinter(settings).doprint(expr)
diff --git a/sympy/printing/pycode.py b/sympy/printing/pycode.py
--- a/sympy/printing/pycode.py
+++ b/sympy/printing/pycode.py
@@ -92,7 +92,7 @@ class AbstractPythonCodePrinter(CodePrinter):
         inline=True,
         fully_qualified_modules=True,
         contract=False,
-        standard='python3'
+        standard='python3',
     )
 
     def __init__(self, settings=None):
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -24,6 +24,8 @@ class StrPrinter(Printer):
         "sympy_integers": False,
         "abbrev": False,
         "perm_cyclic": True,
+        "min": None,
+        "max": None,
     }  # type: Dict[str, Any]
 
     _relationals = dict()  # type: Dict[str, str]
@@ -691,7 +693,9 @@ def _print_Float(self, expr):
             strip = True
         elif self._settings["full_prec"] == "auto":
             strip = self._print_level > 1
-        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip)
+        low = self._settings["min"] if "min" in self._settings else None
+        high = self._settings["max"] if "max" in self._settings else None
+        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
         if rv.startswith('-.0'):
             rv = '-0.' + rv[3:]
         elif rv.startswith('.0'):

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -214,6 +214,14 @@ def test_latex_Float():
     assert latex(Float(1.0e-100)) == r"1.0 \cdot 10^{-100}"
     assert latex(Float(1.0e-100), mul_symbol="times") == \
         r"1.0 \times 10^{-100}"
+    assert latex(Float('10000.0'), full_prec=False, min=-2, max=2) == \
+        r"1.0 \cdot 10^{4}"
+    assert latex(Float('10000.0'), full_prec=False, min=-2, max=4) == \
+        r"1.0 \cdot 10^{4}"
+    assert latex(Float('10000.0'), full_prec=False, min=-2, max=5) == \
+        r"10000.0"
+    assert latex(Float('0.099999'), full_prec=True,  min=-2, max=5) == \
+        r"9.99990000000000 \cdot 10^{-2}"
 
 
 def test_latex_vector_expressions():
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -513,6 +513,10 @@ def test_Float():
                                      '5028841971693993751058209749445923')
     assert str(pi.round(-1)) == '0.0'
     assert str((pi**400 - (pi**400).round(1)).n(2)) == '-0.e+88'
+    assert sstr(Float("100"), full_prec=False, min=-2, max=2) == '1.0e+2'
+    assert sstr(Float("100"), full_prec=False, min=-2, max=3) == '100.0'
+    assert sstr(Float("0.1"), full_prec=False, min=-2, max=3) == '0.1'
+    assert sstr(Float("0.099"), min=-2, max=3) == '9.90000000000000e-2'
 
 
 def test_Relational():

```


## Code snippets

### 1 - sympy/printing/pycode.py:

Start line: 513, End line: 534

```python
class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = "_mpmathcode"

    language = "Python with mpmath"

    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]
    ))
    _kc = {k: 'mpmath.'+v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        # XXX: This does not handle setting mpmath.mp.dps. It is assumed that
        # the caller of the lambdified function will have set it to sufficient
        # precision to match the Floats in the expression.

        # Remove 'mpz' if gmpy is installed.
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)
```
### 2 - sympy/printing/str.py:

Start line: 682, End line: 702

```python
class StrPrinter(Printer):

    def _print_Float(self, expr):
        prec = expr._prec
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        if rv.startswith('+'):
            # e.g., +inf -> inf
            rv = rv[1:]
        return rv
```
### 3 - sympy/printing/pycode.py:

Start line: 572, End line: 599

```python
for k in MpmathPrinter._kf:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_mpmath:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)


_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
})
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'PINF',
    'NegativeInfinity': 'NINF'
}
```
### 4 - sympy/printing/mathml.py:

Start line: 1169, End line: 1206

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_mathml_numbers']
        mrow = self.dom.createElement('mrow')
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(mant))
            mrow.appendChild(mn)
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(separator))
            mrow.appendChild(mo)
            msup = self.dom.createElement('msup')
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode("10"))
            msup.appendChild(mn)
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(exp))
            msup.appendChild(mn)
            mrow.appendChild(msup)
            return mrow
        elif str_real == "+inf":
            return self._print_Infinity(None)
        elif str_real == "-inf":
            return self._print_NegativeInfinity(None)
        else:
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(str_real))
            return mn
```
### 5 - sympy/printing/latex.py:

Start line: 410, End line: 439

```python
class LatexPrinter(Printer):


    def _print_AppliedPermutation(self, expr):
        perm, var = expr.args
        return r"\sigma_{%s}(%s)" % (self._print(perm), self._print(var))

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
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.','{,}')

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.','{,}')
            return str_real
```
### 6 - sympy/polys/numberfields.py:

Start line: 1071, End line: 1084

```python
class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """

    def _print_Integer(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)
```
### 7 - sympy/printing/str.py:

Start line: 201, End line: 260

```python
class StrPrinter(Printer):

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

    def _print_LatticeOp(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        if str(dir) == "+":
            return "Limit(%s, %s, %s)" % tuple(map(self._print, (e, z, z0)))
        else:
            return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print,
                                                            (e, z, z0, dir)))

    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)

    def _print_MutableSparseMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_SparseMatrix(self, expr):
        from sympy.matrices import Matrix
        return self._print(Matrix(expr))

    def _print_ImmutableSparseMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_Matrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_DenseMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_MutableDenseMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_ImmutableMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_ImmutableDenseMatrix(self, expr):
        return self._print_MatrixBase(expr)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))
```
### 8 - sympy/codegen/ast.py:

Start line: 1153, End line: 1238

```python
class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S, Float
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """

    __slots__ = ('name', 'nbits', 'nmant', 'nexp',)

    _construct_nbits = _construct_nmant = _construct_nexp = Integer


    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        # cf. C++'s ``std::numeric_limits::max_exponent``
        return two**(self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        # cf. C++'s ``std::numeric_limits::min_exponent``
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        return (1 - two**-(self.nmant+1))*two**self.max_exponent

    @property
    def tiny(self):
        """ The minimum positive normalized value. """
        # See C macros: FLT_MIN, DBL_MIN, LDBL_MIN
        # or C++'s ``std::numeric_limits::min``
        # or numpy.finfo(dtype).tiny
        return two**(self.min_exponent - 1)


    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        return two**(-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        from sympy.functions import floor, log
        return floor(self.nmant * log(2)/log(10))
```
### 9 - sympy/printing/mathematica.py:

Start line: 310, End line: 335

```python
class MCodePrinter(CodePrinter):

    _print_MinMaxBase = _print_Function

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return "ProductLog[{}]".format(self._print(expr.args[0]))
        return "ProductLog[{}, {}]".format(
            self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_Integral(self, expr):
        if len(expr.variables) == 1 and not expr.limits[0][1:]:
            args = [expr.args[0], expr.variables[0]]
        else:
            args = expr.args
        return "Hold[Integrate[" + ', '.join(self.doprint(a) for a in args) + "]]"

    def _print_Sum(self, expr):
        return "Hold[Sum[" + ', '.join(self.doprint(a) for a in expr.args) + "]]"

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return "Hold[D[" + ', '.join(self.doprint(a) for a in [dexpr] + dvars) + "]]"


    def _get_comment(self, text):
        return "(* {} *)".format(text)
```
### 10 - sympy/printing/latex.py:

Start line: 1431, End line: 1460

```python
class LatexPrinter(Printer):

    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        a, q, z = map(self._print, args)
        sup = r"^{\prime}" if prime else ""
        exp = "" if not exp else "^{%s}" % self._print(exp)
        return r"%s%s\left(%s, %s, %s\right)%s" % (character, sup, a, q, z, exp)

    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, exp=exp)

    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, exp=exp)

    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, prime=True, exp=exp)

    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, prime=True, exp=exp)

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
### 11 - sympy/printing/latex.py:

Start line: 1938, End line: 1953

```python
class LatexPrinter(Printer):

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return r"\left\{%s\right\}" % items


    _print_frozenset = _print_set
```
### 12 - sympy/printing/latex.py:

Start line: 1829, End line: 1858

```python
class LatexPrinter(Printer):

    def _print_UniversalSet(self, expr):
        return r"\mathbb{U}"

    def _print_frac(self, expr, exp=None):
        if exp is None:
            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
        else:
            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
                    self._print(expr.args[0]), self._print(exp))

    def _print_tuple(self, expr):
        if self._settings['decimal_separator'] =='comma':
            return r"\left( %s\right)" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] =='period':
            return r"\left( %s\right)" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)
```
### 13 - sympy/printing/latex.py:

Start line: 2270, End line: 2280

```python
class LatexPrinter(Printer):

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
### 14 - sympy/printing/latex.py:

Start line: 2046, End line: 2118

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
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Rationals(self, i):
        return r"\mathbb{Q}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"
```
### 15 - sympy/printing/latex.py:

Start line: 1009, End line: 1064

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
### 17 - sympy/printing/latex.py:

Start line: 901, End line: 981

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
        tex = r"\%s\left(%s\right)" % (self._print((str(expr.func)).lower()),
                                       ", ".join(texargs))
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
        if self._settings['gothic_re_im']:
            tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{re}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{im}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)
```
### 18 - sympy/printing/latex.py:

Start line: 2291, End line: 2364

```python
class LatexPrinter(Printer):

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return r"W\left(%s\right)" % self._print(expr.args[0])
        return r"W_{%s}\left(%s\right)" % \
            (self._print(expr.args[1]), self._print(expr.args[0]))

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
```
### 19 - sympy/printing/str.py:

Start line: 607, End line: 680

```python
class StrPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False),
                         self.parenthesize(expr.exp, PREC, strict=False))

    def _print_ImmutableDenseNDimArray(self, expr):
        return str(expr)

    def _print_ImmutableSparseNDimArray(self, expr):
        return str(expr)

    def _print_Integer(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(%s)" % (expr)
        return str(expr.p)

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            if self._settings.get("sympy_integers", False):
                return "S(%s)/%s" % (expr.p, expr.q)
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)
```
### 20 - sympy/printing/latex.py:

Start line: 521, End line: 568

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
            if self._settings['fold_short_frac'] and ldenom <= 2 and \
                    "^" not in sdenom:
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
### 21 - sympy/printing/latex.py:

Start line: 570, End line: 611

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1 \
                and self._settings['root_notation']:
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
            base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
            p, q = expr.exp.p, expr.exp.q
            # issue #12886: add parentheses for superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and \
                expr.base.is_commutative:
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
                return self._helper_print_standard_power(expr, tex)
```
### 22 - sympy/printing/latex.py:

Start line: 1090, End line: 1174

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
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 24 - sympy/printing/latex.py:

Start line: 469, End line: 519

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
                              (isinstance(x, Pow) and
                               isinstance(x.base, Quantity)))

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
### 25 - sympy/printing/pycode.py:

Start line: 493, End line: 510

```python
_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
})
_known_constants_mpmath = {
    'Exp1': 'e',
    'Pi': 'pi',
    'GoldenRatio': 'phi',
    'EulerGamma': 'euler',
    'Catalan': 'catalan',
    'NaN': 'nan',
    'Infinity': 'inf',
    'NegativeInfinity': 'ninf'
}
```
### 26 - sympy/printing/latex.py:

Start line: 1, End line: 85

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

from typing import Any, Dict

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true

# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which sympy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg',
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
### 27 - sympy/printing/latex.py:

Start line: 778, End line: 790

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
### 28 - sympy/printing/latex.py:

Start line: 2410, End line: 2476

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    def _print_SubModule(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for x in m.gens))

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens))

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{{{}}} + {{{}}}".format(self._print(x.data),
                 self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{{{}}} + {{{}}}".format(self._print(m.data),
                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    def _print_Tr(self, p):
        # TODO: Handle indices
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)
```
### 29 - sympy/printing/str.py:

Start line: 335, End line: 372

```python
class StrPrinter(Printer):

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()
        if c.is_number and c < 0:
            expr = _keep_coeff(-c, m)
            sign = "-"
        else:
            sign = ""

        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )

    def _print_ElementwiseApplyFunction(self, expr):
        return "{0}.({1})".format(
            expr.function,
            self._print(expr.expr),
        )

    def _print_NaN(self, expr):
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        return '-oo'

    def _print_Order(self, expr):
        if not expr.variables or all(p is S.Zero for p in expr.point):
            if len(expr.variables) <= 1:
                return 'O(%s)' % self._print(expr.expr)
            else:
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)

    def _print_Ordinal(self, expr):
        return expr.__str__()

    def _print_Cycle(self, expr):
        return expr.__str__()
```
### 30 - sympy/printing/pycode.py:

Start line: 537, End line: 569

```python
class MpmathPrinter(PythonCodePrinter):


    def _print_Rational(self, e):
        return "{func}({p})/{func}({q})".format(
            func=self._module_format('mpmath.mpf'),
            q=self._print(e.q),
            p=self._print(e.p)
        )

    def _print_Half(self, e):
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        return "{0}({1}, {2}, {3})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]),
            self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        return "{0}({1}, 0, {2})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]))

    def _print_log2(self, e):
        return '{0}({1})/{0}(2)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        return '{0}({1}+1)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')
```
### 32 - sympy/printing/latex.py:

Start line: 124, End line: 216

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
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "gothic_re_im": False,
        "decimal_separator": "period",
        "perm_cyclic": True,
    }  # type: Dict[str, Any]

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
            return r"\left({}\right)".format(self._print(item))
        else:
            return self._print(item)

    def parenthesize_super(self, s):
        """ Parenthesize s if there is a superscript in s"""
        if "^" in s:
            return r"\left({}\right)".format(s)
        return s
```
### 33 - sympy/printing/latex.py:

Start line: 2127, End line: 2137

```python
class LatexPrinter(Printer):

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return r"\left\{%s \mid %s \right\}" % \
                (vars_print, self._print(s.condition))

        return r"\left\{%s \mid %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition))
```
### 34 - sympy/printing/latex.py:

Start line: 441, End line: 467

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
        return r"\nabla %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        func = expr._expr
        return r"\triangle %s" % self.parenthesize(func, PRECEDENCE['Mul'])
```
### 36 - sympy/printing/str.py:

Start line: 751, End line: 830

```python
class StrPrinter(Printer):

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if not args:
            return "set()"
        return '{%s}' % args

    def _print_frozenset(self, s):
        if not s:
            return "frozenset()"
        return "frozenset(%s)" % self._print_set(s)

    def _print_Sum(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        return "I"

    def _print_ZeroMatrix(self, expr):
        return "0"

    def _print_OneMatrix(self, expr):
        return "1"

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_str(self, expr):
        return str(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_Transpose(self, T):
        return "%s.T" % self.parenthesize(T.arg, PRECEDENCE["Pow"])

    def _print_Uniform(self, expr):
        return "Uniform(%s, %s)" % (self._print(expr.a), self._print(expr.b))

    def _print_Quantity(self, expr):
        if self._settings.get("abbrev", False):
            return "%s" % expr.abbrev
        return "%s" % expr.name

    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+"*"+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_Dimension(self, expr):
        return str(expr)

    def _print_Wild(self, expr):
        return expr.name + '_'

    def _print_WildFunction(self, expr):
        return expr.name + '_'

    def _print_Zero(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(0)"
        return "0"
```
### 38 - sympy/printing/str.py:

Start line: 182, End line: 199

```python
class StrPrinter(Printer):

    def _print_Interval(self, i):
        fin =  'Interval{m}({a}, {b})'
        a, b, l, r = i.args
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not r:
            m = ''
        elif b.is_infinite and not l:
            m = ''
        elif not l and not r:
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        return fin.format(**{'a': a, 'b': b, 'm': m})
```
### 40 - sympy/printing/pycode.py:

Start line: 871, End line: 905

```python
for k in NumPyPrinter._kf:
    setattr(NumPyPrinter, '_print_%s' % k, _print_known_func)

for k in NumPyPrinter._kc:
    setattr(NumPyPrinter, '_print_%s' % k, _print_known_const)


_known_functions_scipy_special = {
    'erf': 'erf',
    'erfc': 'erfc',
    'besselj': 'jv',
    'bessely': 'yv',
    'besseli': 'iv',
    'besselk': 'kv',
    'factorial': 'factorial',
    'gamma': 'gamma',
    'loggamma': 'gammaln',
    'digamma': 'psi',
    'RisingFactorial': 'poch',
    'jacobi': 'eval_jacobi',
    'gegenbauer': 'eval_gegenbauer',
    'chebyshevt': 'eval_chebyt',
    'chebyshevu': 'eval_chebyu',
    'legendre': 'eval_legendre',
    'hermite': 'eval_hermite',
    'laguerre': 'eval_laguerre',
    'assoc_laguerre': 'eval_genlaguerre',
    'beta': 'beta',
    'LambertW' : 'lambertw',
}

_known_constants_scipy_constants = {
    'GoldenRatio': 'golden_ratio',
    'Pi': 'pi',
}
```
### 41 - sympy/printing/str.py:

Start line: 279, End line: 333

```python
class StrPrinter(Printer):

    def _print_Mul(self, expr):

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)
```
### 42 - sympy/printing/latex.py:

Start line: 1614, End line: 1633

```python
class LatexPrinter(Printer):

    def _print_MatMul(self, expr):
        from sympy import MatMul, Mul

        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                             False)

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
### 43 - sympy/printing/latex.py:

Start line: 744, End line: 776

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

        return r"%s %s%s" % (tex, self.parenthesize(expr.function,
                                                    PRECEDENCE["Mul"],
                                                    strict=True),
                             "".join(symbols))
```
### 44 - sympy/printing/str.py:

Start line: 470, End line: 543

```python
class StrPrinter(Printer):

    def _print_Poly(self, expr):
        ATOM_PREC = PRECEDENCE["Atom"] - 1
        terms, gens = [], [ self.parenthesize(s, ATOM_PREC) for s in expr.gens ]

        for monom, coeff in expr.terms():
            s_monom = []

            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + "**%d" % exp)

            s_monom = "*".join(s_monom)

            if coeff.is_Add:
                if s_monom:
                    s_coeff = "(" + self._print(coeff) + ")"
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
                s_term = s_coeff + "*" + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ['-', '+']:
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        format = expr.__class__.__name__ + "(%s, %s"

        from sympy.polys.polyerrors import PolynomialError

        try:
            format += ", modulus=%s" % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()

        format += ")"

        for index, item in enumerate(gens):
            if len(item) > 2 and (item[:1] == "(" and item[len(item) - 1:] == ")"):
                gens[index] = item[1:len(item) - 1]

        return format % (' '.join(terms), ', '.join(gens))

    def _print_UniversalSet(self, p):
        return 'UniversalSet'

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())
```
### 45 - sympy/printing/str.py:

Start line: 167, End line: 180

```python
class StrPrinter(Printer):

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Infinity(self, expr):
        return 'oo'

    def _print_Integral(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Integral(%s, %s)' % (self._print(expr.function), L)
```
### 53 - sympy/printing/latex.py:

Start line: 809, End line: 876

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

            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    pass
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
### 54 - sympy/printing/latex.py:

Start line: 792, End line: 807

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
### 55 - sympy/printing/latex.py:

Start line: 1226, End line: 1291

```python
class LatexPrinter(Printer):

    def _hprint_vec(self, vec):
        if not vec:
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
### 57 - sympy/printing/latex.py:

Start line: 2478, End line: 2488

```python
class LatexPrinter(Printer):

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), self._print(exp))
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), self._print(exp))
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])
```
### 59 - sympy/printing/latex.py:

Start line: 1635, End line: 1643

```python
class LatexPrinter(Printer):

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s\bmod{%s}\right)^{%s}' % \
                (self.parenthesize(expr.args[0], PRECEDENCE['Mul'],
                                   strict=True), self._print(expr.args[1]),
                 self._print(exp))
        return r'%s\bmod{%s}' % (self.parenthesize(expr.args[0],
                                 PRECEDENCE['Mul'], strict=True),
                                 self._print(expr.args[1]))
```
### 60 - sympy/printing/str.py:

Start line: 848, End line: 879

```python
class StrPrinter(Printer):

    def _print_DMF(self, expr):
        return self._print_DMP(expr)

    def _print_Object(self, obj):
        return 'Object("%s")' % obj.name

    def _print_IdentityMorphism(self, morphism):
        return 'IdentityMorphism(%s)' % morphism.domain

    def _print_NamedMorphism(self, morphism):
        return 'NamedMorphism(%s, %s, "%s")' % \
               (morphism.domain, morphism.codomain, morphism.name)

    def _print_Category(self, category):
        return 'Category("%s")' % category.name

    def _print_BaseScalarField(self, field):
        return field._coord_sys._names[field._index]

    def _print_BaseVectorField(self, field):
        return 'e_%s' % field._coord_sys._names[field._index]

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys._names[field._index]
        else:
            return 'd(%s)' % self._print(field)

    def _print_Tr(self, expr):
        #TODO : Handle indices
        return "%s(%s)" % ("Tr", self._print(expr.args[0]))
```
### 61 - sympy/printing/latex.py:

Start line: 2500, End line: 2508

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
### 63 - sympy/printing/latex.py:

Start line: 2490, End line: 2498

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
### 64 - sympy/printing/latex.py:

Start line: 1375, End line: 1429

```python
class LatexPrinter(Printer):

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

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
```
### 65 - sympy/printing/pycode.py:

Start line: 740, End line: 801

```python
class NumPyPrinter(PythonCodePrinter):

    def _print_And(self, expr):
        "Logical And printer"
        # We have to override LambdaPrinter because it uses Python 'and' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
        return '{0}.reduce(({1}))'.format(self._module_format('numpy.logical_and'), ','.join(self._print(i) for i in expr.args))

    def _print_Or(self, expr):
        "Logical Or printer"
        # We have to override LambdaPrinter because it uses Python 'or' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
        return '{0}.reduce(({1}))'.format(self._module_format('numpy.logical_or'), ','.join(self._print(i) for i in expr.args))

    def _print_Not(self, expr):
        "Logical Not printer"
        # We have to override LambdaPrinter because it uses Python 'not' keyword.
        # If LambdaPrinter didn't define it, we would still have to define our
        #     own because StrPrinter doesn't define it.
        return '{0}({1})'.format(self._module_format('numpy.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX Workaround for negative integer power error
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = expr.base ** expr.exp.evalf()
        return self._hprint_Pow(expr, rational=rational, sqrt='numpy.sqrt')

    def _print_Min(self, expr):
        return '{0}(({1}))'.format(self._module_format('numpy.amin'), ','.join(self._print(i) for i in expr.args))

    def _print_Max(self, expr):
        return '{0}(({1}))'.format(self._module_format('numpy.amax'), ','.join(self._print(i) for i in expr.args))


    def _print_arg(self, expr):
        return "%s(%s)" % (self._module_format('numpy.angle'), self._print(expr.args[0]))

    def _print_im(self, expr):
        return "%s(%s)" % (self._module_format('numpy.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        return "%s(%s)" % (self._module_format('numpy.mod'), ', '.join(
            map(lambda arg: self._print(arg), expr.args)))

    def _print_re(self, expr):
        return "%s(%s)" % (self._module_format('numpy.real'), self._print(expr.args[0]))

    def _print_sinc(self, expr):
        return "%s(%s)" % (self._module_format('numpy.sinc'), self._print(expr.args[0]/S.Pi))

    def _print_MatrixBase(self, expr):
        func = self.known_functions.get(expr.__class__.__name__, None)
        if func is None:
            func = self._module_format('numpy.array')
        return "%s(%s)" % (func, self._print(expr.tolist()))

    def _print_Identity(self, expr):
        shape = expr.shape
        if all([dim.is_Integer for dim in shape]):
            return "%s(%s)" % (self._module_format('numpy.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")
```
### 68 - sympy/printing/latex.py:

Start line: 86, End line: 121

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
### 70 - sympy/printing/latex.py:

Start line: 1955, End line: 1979

```python
class LatexPrinter(Printer):

    def _print_Range(self, s):
        dots = r'\ldots'

        if s.has(Symbol):
            return self._print_Basic(s)

        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)

        return (r"\left\{" +
                r", ".join(self._print(el) for el in printset) +
                r"\right\}")
```
### 73 - sympy/printing/latex.py:

Start line: 878, End line: 899

```python
class LatexPrinter(Printer):

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

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
### 77 - sympy/printing/latex.py:

Start line: 1317, End line: 1373

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

    def _print_stieltjes(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_{%s}\left(%s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"_{%s}" % self._print(expr.args[0])
        if exp is not None:
            return r"\gamma%s^{%s}" % (tex, self._print(exp))
        return r"\gamma%s" % tex

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
```
### 78 - sympy/printing/latex.py:

Start line: 2510, End line: 2520

```python
class LatexPrinter(Printer):

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), self._print(exp))
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), self._print(exp))
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])
```
### 81 - sympy/printing/latex.py:

Start line: 1995, End line: 2009

```python
class LatexPrinter(Printer):

    def _print_bernoulli(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_bell(self, expr, exp=None):
        if len(expr.args) == 3:
            tex1 = r"B_{%s, %s}" % (self._print(expr.args[0]),
                                self._print(expr.args[1]))
            tex2 = r"\left(%s\right)" % r", ".join(self._print(el) for
                                               el in expr.args[2])
            if exp is not None:
                tex = r"%s^{%s}%s" % (tex1, self._print(exp), tex2)
            else:
                tex = tex1 + tex2
            return tex
        return self.__print_number_polynomial(expr, "B", exp)
```
### 82 - sympy/printing/latex.py:

Start line: 699, End line: 732

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
        if requires_partial(expr.expr):
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
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, self._print(dim), tex)

        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  strict=True))
```
### 84 - sympy/printing/latex.py:

Start line: 2139, End line: 2186

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
### 86 - sympy/printing/latex.py:

Start line: 1697, End line: 1747

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
                    level_str[back_outer_i].append(
                        r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(
                        block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + \
                            level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str
```
### 88 - sympy/printing/str.py:

Start line: 412, End line: 468

```python
class StrPrinter(Printer):

    def _print_Subs(self, obj):
        expr, old, new = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return "Subs(%s, %s, %s)" % (
            self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        return expr._print()

    def _print_TensorHead(self, expr):
        return expr._print()

    def _print_Tensor(self, expr):
        return expr._print()

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "*".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        return expr._print()

    def _print_PermutationGroup(self, expr):
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)

    def _print_Pi(self, expr):
        return 'pi'

    def _print_PolyRing(self, ring):
        return "Polynomial ring in %s over %s with %s order" % \
            (", ".join(map(lambda rs: self._print(rs), ring.symbols)),
            self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "Rational function field in %s over %s with %s order" % \
            (", ".join(map(lambda fs: self._print(fs), field.symbols)),
            self._print(field.domain), self._print(field.order))

    def _print_FreeGroupElement(self, elm):
        return elm.__str__()

    def _print_PolyElement(self, poly):
        return poly.str(self, PRECEDENCE, "%s**%s", "*")

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self.parenthesize(frac.numer, PRECEDENCE["Mul"], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"], strict=True)
            return numer + "/" + denom
```
### 90 - sympy/printing/latex.py:

Start line: 2188, End line: 2243

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
### 99 - sympy/printing/latex.py:

Start line: 1860, End line: 1882

```python
class LatexPrinter(Printer):

    def _print_list(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            return r"\left[ %s\right]" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return r"\left[ %s\right]" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')


    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \  ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)
```
### 101 - sympy/printing/latex.py:

Start line: 1077, End line: 1088

```python
class LatexPrinter(Printer):

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex
```
### 102 - sympy/printing/latex.py:

Start line: 2282, End line: 2289

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
### 105 - sympy/printing/str.py:

Start line: 72, End line: 165

```python
class StrPrinter(Printer):

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_Not(self, expr):
        return '~%s' %(self.parenthesize(expr.args[0],PRECEDENCE["Not"]))

    def _print_And(self, expr):
        return self.stringify(expr.args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_Or(self, expr):
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_Xor(self, expr):
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    def _print_AppliedPredicate(self, expr):
        return '%s(%s)' % (self._print(expr.func), self._print(expr.arg))

    def _print_Basic(self, expr):
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        return 'zoo'

    def _print_ConditionSet(self, s):
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Derivative(%s)' % ", ".join(map(lambda arg: self._print(arg), [dexpr] + dvars))

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        return 'E'

    def _print_ExprCondPair(self, expr):
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is special -- it's base is tuple
        return str(expr)

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'
```
### 109 - sympy/printing/latex.py:

Start line: 380, End line: 407

```python
class LatexPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation
        from sympy.utilities.exceptions import SymPyDeprecationWarning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            SymPyDeprecationWarning(
                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
                useinstead="init_printing(perm_cyclic={})"
                .format(perm_cyclic),
                issue=15201,
                deprecated_since_version="1.6").warn()
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(expr)

        if expr.size == 0:
            return r"\left( \right)"

        lower = [self._print(arg) for arg in expr.array_form]
        upper = [self._print(arg) for arg in range(len(lower))]

        row1 = " & ".join(upper)
        row2 = " & ".join(lower)
        mat = r" \\ ".join((row1, row2))
        return r"\begin{pmatrix} %s \end{pmatrix}" % mat
```
### 113 - sympy/printing/str.py:

Start line: 1, End line: 46

```python
"""
A Printer for generating readable representation of most sympy classes.
"""

from __future__ import print_function, division

from typing import Any, Dict

from sympy.core import S, Rational, Pow, Basic, Mul
from sympy.core.mul import _keep_coeff
from .printer import Printer
from sympy.printing.precedence import precedence, PRECEDENCE

from mpmath.libmp import prec_to_dps, to_str as mlib_to_str

from sympy.utilities import default_sort_key


class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
    }  # type: Dict[str, Any]

    _relationals = dict()  # type: Dict[str, str]

    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)
```
### 116 - sympy/printing/latex.py:

Start line: 1884, End line: 1892

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
### 121 - sympy/printing/latex.py:

Start line: 2012, End line: 2022

```python
class LatexPrinter(Printer):


    def _print_fibonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "F", exp)

    def _print_lucas(self, expr, exp=None):
        tex = r"L_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_tribonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "T", exp)
```
### 123 - sympy/printing/latex.py:

Start line: 1209, End line: 1224

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
### 124 - sympy/printing/latex.py:

Start line: 613, End line: 629

```python
class LatexPrinter(Printer):

    def _helper_print_standard_power(self, expr, template):
        exp = self._print(expr.exp)
        # issue #12886: add parentheses around superscripts raised
        # to powers
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        if '^' in base and expr.base.is_Symbol:
            base = r"\left(%s\right)" % base
        elif (isinstance(expr.base, Derivative)
            and base.startswith(r'\left(')
            and re.match(r'\\left\(\\d?d?dot', base)
            and base.endswith(r'\right)')):
            # don't use parentheses around dotted derivative
            base = base[6: -7]  # remove outermost added parens
        return template % (base, exp)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])
```
### 127 - sympy/printing/latex.py:

Start line: 1304, End line: 1315

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
### 128 - sympy/printing/latex.py:

Start line: 1894, End line: 1904

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
### 129 - sympy/printing/pycode.py:

Start line: 907, End line: 964

```python
class SciPyPrinter(NumPyPrinter):

    language = "Python with SciPy"

    _kf = dict(chain(
        NumPyPrinter._kf.items(),
        [(k, 'scipy.special.' + v) for k, v in _known_functions_scipy_special.items()]
    ))
    _kc =dict(chain(
        NumPyPrinter._kc.items(),
        [(k, 'scipy.constants.' + v) for k, v in _known_constants_scipy_constants.items()]
    ))

    def _print_SparseMatrix(self, expr):
        i, j, data = [], [], []
        for (r, c), v in expr._smat.items():
            i.append(r)
            j.append(c)
            data.append(v)

        return "{name}({data}, ({i}, {j}), shape={shape})".format(
            name=self._module_format('scipy.sparse.coo_matrix'),
            data=data, i=i, j=j, shape=expr.shape
        )

    _print_ImmutableSparseMatrix = _print_SparseMatrix

    # SciPy's lpmv has a different order of arguments from assoc_legendre
    def _print_assoc_legendre(self, expr):
        return "{0}({2}, {1}, {3})".format(
            self._module_format('scipy.special.lpmv'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]))

    def _print_lowergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_uppergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammaincc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_fresnels(self, expr):
        return "{0}({1})[0]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_fresnelc(self, expr):
        return "{0}({1})[1]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))
```
### 132 - sympy/printing/latex.py:

Start line: 2245, End line: 2255

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
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr,
                                                                 index)
```
### 134 - sympy/printing/latex.py:

Start line: 669, End line: 697

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
            inneritems.sort(key=lambda x: x[0].__str__())
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
### 139 - sympy/printing/latex.py:

Start line: 1917, End line: 1936

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

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('\\text{Domain: }' + self._print(d.symbols) + '\\text{ in }' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)
```
### 140 - sympy/printing/pycode.py:

Start line: 224, End line: 238

```python
class AbstractPythonCodePrinter(CodePrinter):

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return ('{0} % {1}'.format(*map(lambda x: self.parenthesize(x, PREC), expr.args)))
```
### 141 - sympy/printing/latex.py:

Start line: 2257, End line: 2268

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
            return r"\operatorname{%s} {\left(%s\right)}" % (cls,
                                                             ", ".join(args))
```
### 143 - sympy/printing/latex.py:

Start line: 1293, End line: 1302

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
### 145 - sympy/printing/latex.py:

Start line: 631, End line: 648

```python
class LatexPrinter(Printer):

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 148 - sympy/printing/latex.py:

Start line: 983, End line: 992

```python
class LatexPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
```
### 149 - sympy/printing/glsl.py:

Start line: 30, End line: 78

```python
class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported = set()  # type: Set[Basic]
    printmethod = "_glsl"
    language = "GLSL"

    _default_settings = {
        'use_operators': True,
        'mat_nested': False,
        'mat_separator': ',\n',
        'mat_transpose': False,
        'glsl_types': True,

        'order': None,
        'full_prec': 'auto',
        'precision': 9,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': False,
        'contract': True,
        'error_on_reserved': False,
        'reserved_word_suffix': '_'
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {0}".format(text)

    def _declare_number_const(self, name, value):
        return "float {0} = {1};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)
```
### 150 - sympy/printing/latex.py:

Start line: 1816, End line: 1827

```python
class LatexPrinter(Printer):

    def _print_PartialDerivative(self, expr):
        if len(expr.variables) == 1:
            return r"\frac{\partial}{\partial {%s}}{%s}" % (
                self._print(expr.variables[0]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
        else:
            return r"\frac{\partial^{%s}}{%s}{%s}" % (
                len(expr.variables),
                " ".join([r"\partial {%s}" % self._print(i) for i in expr.variables]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
```
### 153 - sympy/printing/latex.py:

Start line: 1782, End line: 1814

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
```
### 155 - sympy/printing/latex.py:

Start line: 734, End line: 742

```python
class LatexPrinter(Printer):

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr,
                                                         latex_subs)
```
### 157 - sympy/printing/latex.py:

Start line: 1176, End line: 1207

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
### 158 - sympy/printing/latex.py:

Start line: 994, End line: 1007

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
### 159 - sympy/printing/pycode.py:

Start line: 602, End line: 639

```python
class NumPyPrinter(PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    printmethod = "_numpycode"
    language = "Python with NumPy"

    _kf = dict(chain(
        PythonCodePrinter._kf.items(),
        [(k, 'numpy.' + v) for k, v in _known_functions_numpy.items()]
    ))
    _kc = {k: 'numpy.'+v for k, v in _known_constants_numpy.items()}


    def _print_seq(self, seq):
        "General sequence printer: converts to tuple"
        # Print tuples here instead of lists because numba supports
        #     tuples in nopython mode.
        delimiter=', '
        return '({},)'.format(delimiter.join(self._print(item) for item in seq))

    def _print_MatMul(self, expr):
        "Matrix multiplication printer"
        if expr.as_coeff_matrices()[0] is not S.One:
            expr_list = expr.as_coeff_matrices()[1]+[(expr.as_coeff_matrices()[0])]
            return '({0})'.format(').dot('.join(self._print(i) for i in expr_list))
        return '({0})'.format(').dot('.join(self._print(i) for i in expr.args))

    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{0}({1}, {2})'.format(self._module_format('numpy.linalg.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Inverse(self, expr):
        "Matrix inverse printer"
        return '{0}({1})'.format(self._module_format('numpy.linalg.inv'),
            self._print(expr.args[0]))
```
### 162 - sympy/printing/latex.py:

Start line: 1510, End line: 1528

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
### 163 - sympy/printing/latex.py:

Start line: 1543, End line: 1568

```python
class LatexPrinter(Printer):

    def _print_MatrixBase(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

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
### 165 - sympy/printing/latex.py:

Start line: 2024, End line: 2044

```python
class LatexPrinter(Printer):

    def _print_SeqFormula(self, s):
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            return r"\left\{%s\right\}_{%s=%s}^{%s}" % (
                self._print(s.formula),
                self._print(s.variables[0]),
                self._print(s.start),
                self._print(s.stop)
            )
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (r'\ldots', s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(r'\ldots')
        else:
            printset = tuple(s)

        return (r"\left[" +
                r", ".join(self._print(el) for el in printset) +
                r"\right]")
```
### 167 - sympy/printing/latex.py:

Start line: 1749, End line: 1780

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
            if ((index in index_map) or prev_map) and \
                    last_valence == new_valence:
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
### 177 - sympy/printing/latex.py:

Start line: 1645, End line: 1695

```python
class LatexPrinter(Printer):

    def _print_HadamardProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \circ '.join(
            map(lambda arg: parens(arg, prec, strict=True), args))

    def _print_HadamardPower(self, expr):
        if precedence_traditional(expr.exp) < PRECEDENCE["Mul"]:
            template = r"%s^{\circ \left({%s}\right)}"
        else:
            template = r"%s^{\circ {%s}}"
        return self._helper_print_standard_power(expr, template)

    def _print_KroneckerProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \otimes '.join(
            map(lambda arg: parens(arg, prec, strict=True), args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol):
            return "\\left(%s\\right)^{%s}" % (self._print(base),
                                              self._print(exp))
        else:
            return "%s^{%s}" % (self._print(base), self._print(exp))

    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings[
            'mat_symbol_style'])

    def _print_ZeroMatrix(self, Z):
        return r"\mathbb{0}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{0}"

    def _print_OneMatrix(self, O):
        return r"\mathbb{1}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{1}"

    def _print_Identity(self, I):
        return r"\mathbb{I}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{I}"

    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        return "P_{%s}" % perm_str
```
