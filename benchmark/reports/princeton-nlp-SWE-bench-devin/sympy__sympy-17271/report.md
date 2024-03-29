# sympy__sympy-17271

| **sympy/sympy** | `52641f02c78331a274ec79b6b2ccf78c38a3c6ce` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 334 |
| **Avg pos** | 41.0 |
| **Min pos** | 1 |
| **Max pos** | 80 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/functions/elementary/integers.py b/sympy/functions/elementary/integers.py
--- a/sympy/functions/elementary/integers.py
+++ b/sympy/functions/elementary/integers.py
@@ -3,9 +3,11 @@
 from sympy.core import Add, S
 from sympy.core.evalf import get_integer_part, PrecisionExhausted
 from sympy.core.function import Function
+from sympy.core.logic import fuzzy_or
 from sympy.core.numbers import Integer
-from sympy.core.relational import Gt, Lt, Ge, Le
+from sympy.core.relational import Gt, Lt, Ge, Le, Relational
 from sympy.core.symbol import Symbol
+from sympy.core.sympify import _sympify
 
 
 ###############################################################################
@@ -155,6 +157,8 @@ def _eval_Eq(self, other):
     def __le__(self, other):
         if self.args[0] == other and other.is_real:
             return S.true
+        if other is S.Infinity and self.is_finite:
+            return S.true
         return Le(self, other, evaluate=False)
 
     def __gt__(self, other):
@@ -244,6 +248,8 @@ def __lt__(self, other):
     def __ge__(self, other):
         if self.args[0] == other and other.is_real:
             return S.true
+        if other is S.NegativeInfinity and self.is_real:
+            return S.true
         return Ge(self, other, evaluate=False)
 
 
@@ -309,7 +315,7 @@ def _eval(arg):
                 if arg is S.NaN:
                     return S.NaN
                 elif arg is S.ComplexInfinity:
-                    return None
+                    return S.NaN
                 else:
                     return arg - floor(arg)
             return cls(arg, evaluate=False)
@@ -343,3 +349,85 @@ def _eval_Eq(self, other):
             if (self.rewrite(floor) == other) or \
                     (self.rewrite(ceiling) == other):
                 return S.true
+            # Check if other < 0
+            if other.is_extended_negative:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return S.false
+
+    def _eval_is_finite(self):
+        return True
+
+    def _eval_is_real(self):
+        return self.args[0].is_extended_real
+
+    def _eval_is_imaginary(self):
+        return self.args[0].is_imaginary
+
+    def _eval_is_integer(self):
+        return self.args[0].is_integer
+
+    def _eval_is_zero(self):
+        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])
+
+    def _eval_is_negative(self):
+        return False
+
+    def __ge__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other <= 0
+            if other.is_extended_nonpositive:
+                return S.true
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return not(res)
+        return Ge(self, other, evaluate=False)
+
+    def __gt__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other < 0
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return not(res)
+            # Check if other >= 1
+            if other.is_extended_negative:
+                return S.true
+        return Gt(self, other, evaluate=False)
+
+    def __le__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other < 0
+            if other.is_extended_negative:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return res
+        return Le(self, other, evaluate=False)
+
+    def __lt__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other <= 0
+            if other.is_extended_nonpositive:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return res
+        return Lt(self, other, evaluate=False)
+
+    def _value_one_or_more(self, other):
+        if other.is_extended_real:
+            if other.is_number:
+                res = other >= 1
+                if res and not isinstance(res, Relational):
+                    return S.true
+            if other.is_integer and other.is_positive:
+                return S.true
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1733,6 +1733,13 @@ def _print_TensorIndex(self, expr):
     def _print_UniversalSet(self, expr):
         return r"\mathbb{U}"
 
+    def _print_frac(self, expr, exp=None):
+        if exp is None:
+            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
+        else:
+            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
+                    self._print(expr.args[0]), self._print(exp))
+
     def _print_tuple(self, expr):
         if self._settings['decimal_separator'] =='comma':
             return r"\left( %s\right)" % \

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/functions/elementary/integers.py | 6 | 6 | - | 1 | -
| sympy/functions/elementary/integers.py | 158 | 158 | 80 | 1 | 30087
| sympy/functions/elementary/integers.py | 247 | 247 | - | 1 | -
| sympy/functions/elementary/integers.py | 312 | 312 | 1 | 1 | 334
| sympy/functions/elementary/integers.py | 346 | 346 | 1 | 1 | 334
| sympy/printing/latex.py | 1736 | 1736 | - | - | -


## Problem Statement

```
frac(zoo) gives TypeError
\`\`\`

In [1]: from sympy import frac, zoo

In [2]: frac(zoo)
Traceback (most recent call last):

  File "<ipython-input-2-eb6875922196>", line 1, in <module>
    frac(zoo)

  File "C:\Users\Oscar\sympy\sympy\core\function.py", line 458, in __new__
    result = super(Function, cls).__new__(cls, *args, **options)

  File "C:\Users\Oscar\sympy\sympy\core\function.py", line 277, in __new__
    evaluated = cls.eval(*args)

  File "C:\Users\Oscar\sympy\sympy\functions\elementary\integers.py", line 333, in eval
    return real + S.ImaginaryUnit*imag

TypeError: unsupported operand type(s) for +: 'NoneType' and 'Zero'
\`\`\`

Not sure what should happen, but not this. 

I am trying to cover these lines in a test:
https://github.com/sympy/sympy/blob/51630a792b1ff403151e70bdd692a0d290eb09ca/sympy/functions/elementary/integers.py#L311-L312

Clearly, they are covered by calling `frac(zoo)` since the `NoneType` comes from that line, but I do not really want an exception...

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/functions/elementary/integers.py** | 299 | 346| 334 | 334 | 2319 | 
| 2 | 2 sympy/functions/special/error_functions.py | 759 | 779| 175 | 509 | 21823 | 
| 3 | **2 sympy/functions/elementary/integers.py** | 250 | 298| 279 | 788 | 21823 | 
| 4 | 2 sympy/functions/special/error_functions.py | 480 | 505| 208 | 996 | 21823 | 
| 5 | 3 sympy/functions/special/zeta_functions.py | 442 | 478| 314 | 1310 | 27357 | 
| 6 | 3 sympy/functions/special/error_functions.py | 520 | 559| 460 | 1770 | 27357 | 
| 7 | 3 sympy/functions/special/error_functions.py | 353 | 401| 533 | 2303 | 27357 | 
| 8 | 3 sympy/functions/special/error_functions.py | 2262 | 2270| 200 | 2503 | 27357 | 
| 9 | 4 sympy/core/numbers.py | 3918 | 3978| 364 | 2867 | 57465 | 
| 10 | 4 sympy/functions/special/error_functions.py | 2125 | 2133| 212 | 3079 | 57465 | 
| 11 | 4 sympy/functions/special/error_functions.py | 2027 | 2110| 620 | 3699 | 57465 | 
| 12 | 4 sympy/functions/special/error_functions.py | 2302 | 2349| 493 | 4192 | 57465 | 
| 13 | 4 sympy/functions/special/error_functions.py | 2135 | 2162| 454 | 4646 | 57465 | 
| 14 | 4 sympy/functions/special/error_functions.py | 174 | 225| 537 | 5183 | 57465 | 
| 15 | 4 sympy/functions/special/error_functions.py | 2165 | 2247| 632 | 5815 | 57465 | 
| 16 | 4 sympy/functions/special/error_functions.py | 2272 | 2299| 453 | 6268 | 57465 | 
| 17 | 4 sympy/functions/special/error_functions.py | 1184 | 1205| 265 | 6533 | 57465 | 
| 18 | 4 sympy/functions/special/error_functions.py | 2371 | 2387| 171 | 6704 | 57465 | 
| 19 | 4 sympy/functions/special/error_functions.py | 1372 | 1425| 531 | 7235 | 57465 | 
| 20 | 4 sympy/core/numbers.py | 3312 | 3388| 383 | 7618 | 57465 | 
| 21 | 4 sympy/functions/special/zeta_functions.py | 118 | 177| 658 | 8276 | 57465 | 
| 22 | 4 sympy/core/numbers.py | 181 | 205| 184 | 8460 | 57465 | 
| 23 | 4 sympy/functions/special/error_functions.py | 1035 | 1080| 474 | 8934 | 57465 | 
| 24 | 4 sympy/functions/special/zeta_functions.py | 481 | 503| 211 | 9145 | 57465 | 
| 25 | 5 sympy/core/power.py | 169 | 251| 1060 | 10205 | 72483 | 
| 26 | 5 sympy/functions/special/zeta_functions.py | 179 | 199| 210 | 10415 | 72483 | 
| 27 | 5 sympy/functions/special/error_functions.py | 1207 | 1219| 150 | 10565 | 72483 | 
| 28 | 6 sympy/polys/fields.py | 538 | 564| 233 | 10798 | 77405 | 
| 29 | 6 sympy/functions/special/error_functions.py | 1261 | 1279| 115 | 10913 | 77405 | 
| 30 | 7 sympy/functions/special/hyper.py | 209 | 217| 148 | 11061 | 87734 | 
| 31 | 7 sympy/functions/special/zeta_functions.py | 307 | 327| 194 | 11255 | 87734 | 
| 32 | 7 sympy/core/numbers.py | 112 | 149| 385 | 11640 | 87734 | 
| 33 | 7 sympy/functions/special/error_functions.py | 1549 | 1562| 139 | 11779 | 87734 | 
| 34 | 7 sympy/functions/special/error_functions.py | 1970 | 2008| 223 | 12002 | 87734 | 
| 35 | 7 sympy/core/numbers.py | 3391 | 3449| 326 | 12328 | 87734 | 
| 36 | 8 sympy/integrals/meijerint.py | 1011 | 1052| 628 | 12956 | 112028 | 
| 37 | 8 sympy/polys/fields.py | 583 | 612| 306 | 13262 | 112028 | 
| 38 | 8 sympy/functions/special/error_functions.py | 1221 | 1232| 172 | 13434 | 112028 | 
| 39 | 8 sympy/functions/special/error_functions.py | 1083 | 1181| 889 | 14323 | 112028 | 
| 40 | 8 sympy/functions/special/hyper.py | 613 | 645| 386 | 14709 | 112028 | 
| 41 | 9 sympy/functions/elementary/complexes.py | 829 | 868| 320 | 15029 | 121269 | 
| 42 | 9 sympy/functions/special/error_functions.py | 404 | 478| 517 | 15546 | 121269 | 
| 43 | 9 sympy/polys/fields.py | 243 | 258| 132 | 15678 | 121269 | 
| 44 | 9 sympy/functions/special/zeta_functions.py | 272 | 305| 419 | 16097 | 121269 | 
| 45 | 9 sympy/functions/special/error_functions.py | 1428 | 1510| 550 | 16647 | 121269 | 
| 46 | 9 sympy/core/numbers.py | 1210 | 1307| 781 | 17428 | 121269 | 
| 47 | 9 sympy/functions/special/error_functions.py | 2010 | 2024| 122 | 17550 | 121269 | 
| 48 | 10 sympy/core/evalf.py | 936 | 1013| 783 | 18333 | 134987 | 
| 49 | 10 sympy/functions/special/error_functions.py | 130 | 159| 217 | 18550 | 134987 | 
| 50 | 10 sympy/functions/elementary/complexes.py | 1 | 15| 168 | 18718 | 134987 | 
| 51 | 10 sympy/core/evalf.py | 1031 | 1054| 214 | 18932 | 134987 | 
| 52 | 10 sympy/functions/special/hyper.py | 181 | 207| 315 | 19247 | 134987 | 
| 53 | 10 sympy/functions/special/error_functions.py | 313 | 336| 169 | 19416 | 134987 | 
| 54 | 11 sympy/polys/domains/old_fractionfield.py | 70 | 88| 236 | 19652 | 136620 | 
| 55 | 12 sympy/utilities/randtest.py | 57 | 79| 222 | 19874 | 138066 | 
| 56 | 12 sympy/functions/special/hyper.py | 896 | 917| 240 | 20114 | 138066 | 
| 57 | 13 sympy/integrals/transforms.py | 561 | 663| 991 | 21105 | 154832 | 
| 58 | 13 sympy/core/numbers.py | 1753 | 1767| 156 | 21261 | 154832 | 
| 59 | 13 sympy/functions/special/error_functions.py | 1234 | 1258| 278 | 21539 | 154832 | 
| 60 | 14 sympy/polys/domains/fractionfield.py | 1 | 117| 895 | 22434 | 155965 | 
| 61 | 14 sympy/core/numbers.py | 2929 | 2996| 476 | 22910 | 155965 | 
| 62 | 14 sympy/core/numbers.py | 2613 | 2640| 228 | 23138 | 155965 | 
| 63 | 14 sympy/functions/special/error_functions.py | 1579 | 1677| 674 | 23812 | 155965 | 
| 64 | 14 sympy/functions/special/error_functions.py | 782 | 850| 447 | 24259 | 155965 | 
| 65 | 14 sympy/core/numbers.py | 654 | 735| 610 | 24869 | 155965 | 
| 66 | 15 sympy/parsing/latex/_parse_latex_antlr.py | 327 | 368| 451 | 25320 | 160350 | 
| 67 | 15 sympy/core/numbers.py | 1801 | 1837| 404 | 25724 | 160350 | 
| 68 | 16 sympy/functions/special/elliptic_integrals.py | 208 | 233| 195 | 25919 | 164193 | 
| 69 | 16 sympy/functions/special/error_functions.py | 913 | 930| 144 | 26063 | 164193 | 
| 70 | 16 sympy/core/numbers.py | 3119 | 3187| 503 | 26566 | 164193 | 
| 71 | 16 sympy/core/numbers.py | 1839 | 1866| 188 | 26754 | 164193 | 
| 72 | 17 sympy/benchmarks/bench_symbench.py | 1 | 63| 547 | 27301 | 165219 | 
| 73 | 17 sympy/core/evalf.py | 1015 | 1028| 138 | 27439 | 165219 | 
| 74 | 17 sympy/functions/special/error_functions.py | 933 | 1033| 765 | 28204 | 165219 | 
| 75 | 17 sympy/functions/special/error_functions.py | 658 | 694| 430 | 28634 | 165219 | 
| 76 | 18 sympy/series/limits.py | 1 | 15| 151 | 28785 | 167368 | 
| 77 | 18 sympy/functions/special/hyper.py | 962 | 977| 237 | 29022 | 167368 | 
| 78 | 18 sympy/functions/special/error_functions.py | 2352 | 2368| 195 | 29217 | 167368 | 
| 79 | 19 sympy/core/function.py | 1579 | 1607| 301 | 29518 | 194137 | 
| **-> 80 <-** | **19 sympy/functions/elementary/integers.py** | 82 | 163| 569 | 30087 | 194137 | 
| 81 | 19 sympy/core/numbers.py | 1 | 39| 340 | 30427 | 194137 | 


## Missing Patch Files

 * 1: sympy/functions/elementary/integers.py
 * 2: sympy/printing/latex.py

### Hint

```
I think it should return nan instead of None so that `frac(zoo) -> nan`.
oo gives `AccumBounds(0, 1)` so an option may be `AccumBounds(0, 1) + I*AccumBounds(0, 1)` or something. Not sure when one would like to call it though. Even for oo.
I think that `nan` would be the best choice (for `oo` as well unless a "real nan" is implemented).
```

## Patch

```diff
diff --git a/sympy/functions/elementary/integers.py b/sympy/functions/elementary/integers.py
--- a/sympy/functions/elementary/integers.py
+++ b/sympy/functions/elementary/integers.py
@@ -3,9 +3,11 @@
 from sympy.core import Add, S
 from sympy.core.evalf import get_integer_part, PrecisionExhausted
 from sympy.core.function import Function
+from sympy.core.logic import fuzzy_or
 from sympy.core.numbers import Integer
-from sympy.core.relational import Gt, Lt, Ge, Le
+from sympy.core.relational import Gt, Lt, Ge, Le, Relational
 from sympy.core.symbol import Symbol
+from sympy.core.sympify import _sympify
 
 
 ###############################################################################
@@ -155,6 +157,8 @@ def _eval_Eq(self, other):
     def __le__(self, other):
         if self.args[0] == other and other.is_real:
             return S.true
+        if other is S.Infinity and self.is_finite:
+            return S.true
         return Le(self, other, evaluate=False)
 
     def __gt__(self, other):
@@ -244,6 +248,8 @@ def __lt__(self, other):
     def __ge__(self, other):
         if self.args[0] == other and other.is_real:
             return S.true
+        if other is S.NegativeInfinity and self.is_real:
+            return S.true
         return Ge(self, other, evaluate=False)
 
 
@@ -309,7 +315,7 @@ def _eval(arg):
                 if arg is S.NaN:
                     return S.NaN
                 elif arg is S.ComplexInfinity:
-                    return None
+                    return S.NaN
                 else:
                     return arg - floor(arg)
             return cls(arg, evaluate=False)
@@ -343,3 +349,85 @@ def _eval_Eq(self, other):
             if (self.rewrite(floor) == other) or \
                     (self.rewrite(ceiling) == other):
                 return S.true
+            # Check if other < 0
+            if other.is_extended_negative:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return S.false
+
+    def _eval_is_finite(self):
+        return True
+
+    def _eval_is_real(self):
+        return self.args[0].is_extended_real
+
+    def _eval_is_imaginary(self):
+        return self.args[0].is_imaginary
+
+    def _eval_is_integer(self):
+        return self.args[0].is_integer
+
+    def _eval_is_zero(self):
+        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])
+
+    def _eval_is_negative(self):
+        return False
+
+    def __ge__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other <= 0
+            if other.is_extended_nonpositive:
+                return S.true
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return not(res)
+        return Ge(self, other, evaluate=False)
+
+    def __gt__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other < 0
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return not(res)
+            # Check if other >= 1
+            if other.is_extended_negative:
+                return S.true
+        return Gt(self, other, evaluate=False)
+
+    def __le__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other < 0
+            if other.is_extended_negative:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return res
+        return Le(self, other, evaluate=False)
+
+    def __lt__(self, other):
+        if self.is_extended_real:
+            other = _sympify(other)
+            # Check if other <= 0
+            if other.is_extended_nonpositive:
+                return S.false
+            # Check if other >= 1
+            res = self._value_one_or_more(other)
+            if res is not None:
+                return res
+        return Lt(self, other, evaluate=False)
+
+    def _value_one_or_more(self, other):
+        if other.is_extended_real:
+            if other.is_number:
+                res = other >= 1
+                if res and not isinstance(res, Relational):
+                    return S.true
+            if other.is_integer and other.is_positive:
+                return S.true
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1733,6 +1733,13 @@ def _print_TensorIndex(self, expr):
     def _print_UniversalSet(self, expr):
         return r"\mathbb{U}"
 
+    def _print_frac(self, expr, exp=None):
+        if exp is None:
+            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
+        else:
+            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
+                    self._print(expr.args[0]), self._print(exp))
+
     def _print_tuple(self, expr):
         if self._settings['decimal_separator'] =='comma':
             return r"\left( %s\right)" % \

```

## Test Patch

```diff
diff --git a/sympy/functions/elementary/tests/test_integers.py b/sympy/functions/elementary/tests/test_integers.py
--- a/sympy/functions/elementary/tests/test_integers.py
+++ b/sympy/functions/elementary/tests/test_integers.py
@@ -1,5 +1,6 @@
 from sympy import AccumBounds, Symbol, floor, nan, oo, zoo, E, symbols, \
-        ceiling, pi, Rational, Float, I, sin, exp, log, factorial, frac, Eq
+        ceiling, pi, Rational, Float, I, sin, exp, log, factorial, frac, Eq, \
+        Le, Ge, Gt, Lt, Ne, sqrt
 
 from sympy.core.expr import unchanged
 from sympy.utilities.pytest import XFAIL
@@ -113,6 +114,7 @@ def test_floor():
     assert (floor(x) > x).is_Relational
     assert (floor(x) <= y).is_Relational  # arg is not same as rhs
     assert (floor(x) > y).is_Relational
+    assert (floor(y) <= oo) == True
 
     assert floor(y).rewrite(frac) == y - frac(y)
     assert floor(y).rewrite(ceiling) == -ceiling(-y)
@@ -228,6 +230,7 @@ def test_ceiling():
     assert (ceiling(x) < x).is_Relational
     assert (ceiling(x) >= y).is_Relational  # arg is not same as rhs
     assert (ceiling(x) < y).is_Relational
+    assert (ceiling(y) >= -oo) == True
 
     assert ceiling(y).rewrite(floor) == -floor(-y)
     assert ceiling(y).rewrite(frac) == y + frac(-y)
@@ -244,6 +247,7 @@ def test_frac():
     assert isinstance(frac(x), frac)
     assert frac(oo) == AccumBounds(0, 1)
     assert frac(-oo) == AccumBounds(0, 1)
+    assert frac(zoo) is nan
 
     assert frac(n) == 0
     assert frac(nan) == nan
@@ -269,6 +273,121 @@ def test_frac():
     assert Eq(frac(y), y - floor(y))
     assert Eq(frac(y), y + ceiling(-y))
 
+    r = Symbol('r', real=True)
+    p_i = Symbol('p_i', integer=True, positive=True)
+    n_i = Symbol('p_i', integer=True, negative=True)
+    np_i = Symbol('np_i', integer=True, nonpositive=True)
+    nn_i = Symbol('nn_i', integer=True, nonnegative=True)
+    p_r = Symbol('p_r', real=True, positive=True)
+    n_r = Symbol('n_r', real=True, negative=True)
+    np_r = Symbol('np_r', real=True, nonpositive=True)
+    nn_r = Symbol('nn_r', real=True, nonnegative=True)
+
+    # Real frac argument, integer rhs
+    assert frac(r) <= p_i
+    assert not frac(r) <= n_i
+    assert (frac(r) <= np_i).has(Le)
+    assert (frac(r) <= nn_i).has(Le)
+    assert frac(r) < p_i
+    assert not frac(r) < n_i
+    assert not frac(r) < np_i
+    assert (frac(r) < nn_i).has(Lt)
+    assert not frac(r) >= p_i
+    assert frac(r) >= n_i
+    assert frac(r) >= np_i
+    assert (frac(r) >= nn_i).has(Ge)
+    assert not frac(r) > p_i
+    assert frac(r) > n_i
+    assert (frac(r) > np_i).has(Gt)
+    assert (frac(r) > nn_i).has(Gt)
+
+    assert not Eq(frac(r), p_i)
+    assert not Eq(frac(r), n_i)
+    assert Eq(frac(r), np_i).has(Eq)
+    assert Eq(frac(r), nn_i).has(Eq)
+
+    assert Ne(frac(r), p_i)
+    assert Ne(frac(r), n_i)
+    assert Ne(frac(r), np_i).has(Ne)
+    assert Ne(frac(r), nn_i).has(Ne)
+
+
+    # Real frac argument, real rhs
+    assert (frac(r) <= p_r).has(Le)
+    assert not frac(r) <= n_r
+    assert (frac(r) <= np_r).has(Le)
+    assert (frac(r) <= nn_r).has(Le)
+    assert (frac(r) < p_r).has(Lt)
+    assert not frac(r) < n_r
+    assert not frac(r) < np_r
+    assert (frac(r) < nn_r).has(Lt)
+    assert (frac(r) >= p_r).has(Ge)
+    assert frac(r) >= n_r
+    assert frac(r) >= np_r
+    assert (frac(r) >= nn_r).has(Ge)
+    assert (frac(r) > p_r).has(Gt)
+    assert frac(r) > n_r
+    assert (frac(r) > np_r).has(Gt)
+    assert (frac(r) > nn_r).has(Gt)
+
+    assert not Eq(frac(r), n_r)
+    assert Eq(frac(r), p_r).has(Eq)
+    assert Eq(frac(r), np_r).has(Eq)
+    assert Eq(frac(r), nn_r).has(Eq)
+
+    assert Ne(frac(r), p_r).has(Ne)
+    assert Ne(frac(r), n_r)
+    assert Ne(frac(r), np_r).has(Ne)
+    assert Ne(frac(r), nn_r).has(Ne)
+
+    # Real frac argument, +/- oo rhs
+    assert frac(r) < oo
+    assert frac(r) <= oo
+    assert not frac(r) > oo
+    assert not frac(r) >= oo
+
+    assert not frac(r) < -oo
+    assert not frac(r) <= -oo
+    assert frac(r) > -oo
+    assert frac(r) >= -oo
+
+    assert frac(r) < 1
+    assert frac(r) <= 1
+    assert not frac(r) > 1
+    assert not frac(r) >= 1
+
+    assert not frac(r) < 0
+    assert (frac(r) <= 0).has(Le)
+    assert (frac(r) > 0).has(Gt)
+    assert frac(r) >= 0
+
+    # Some test for numbers
+    assert frac(r) <= sqrt(2)
+    assert (frac(r) <= sqrt(3) - sqrt(2)).has(Le)
+    assert not frac(r) <= sqrt(2) - sqrt(3)
+    assert not frac(r) >= sqrt(2)
+    assert (frac(r) >= sqrt(3) - sqrt(2)).has(Ge)
+    assert frac(r) >= sqrt(2) - sqrt(3)
+
+    assert not Eq(frac(r), sqrt(2))
+    assert Eq(frac(r), sqrt(3) - sqrt(2)).has(Eq)
+    assert not Eq(frac(r), sqrt(2) - sqrt(3))
+    assert Ne(frac(r), sqrt(2))
+    assert Ne(frac(r), sqrt(3) - sqrt(2)).has(Ne)
+    assert Ne(frac(r), sqrt(2) - sqrt(3))
+
+    assert frac(p_i, evaluate=False).is_zero
+    assert frac(p_i, evaluate=False).is_finite
+    assert frac(p_i, evaluate=False).is_integer
+    assert frac(p_i, evaluate=False).is_real
+    assert frac(r).is_finite
+    assert frac(r).is_real
+    assert frac(r).is_zero is None
+    assert frac(r).is_integer is None
+
+    assert frac(oo).is_finite
+    assert frac(oo).is_real
+
 
 def test_series():
     x, y = symbols('x,y')
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -10,7 +10,7 @@
     assoc_laguerre, assoc_legendre, beta, binomial, catalan, ceiling, Complement,
     chebyshevt, chebyshevu, conjugate, cot, coth, diff, dirichlet_eta, euler,
     exp, expint, factorial, factorial2, floor, gamma, gegenbauer, hermite,
-    hyper, im, jacobi, laguerre, legendre, lerchphi, log,
+    hyper, im, jacobi, laguerre, legendre, lerchphi, log, frac,
     meijerg, oo, polar_lift, polylog, re, root, sin, sqrt, symbols,
     uppergamma, zeta, subfactorial, totient, elliptic_k, elliptic_f,
     elliptic_e, elliptic_pi, cos, tan, Wild, true, false, Equivalent, Not,
@@ -370,8 +370,11 @@ def test_latex_functions():
 
     assert latex(floor(x)) == r"\left\lfloor{x}\right\rfloor"
     assert latex(ceiling(x)) == r"\left\lceil{x}\right\rceil"
+    assert latex(frac(x)) == r"\operatorname{frac}{\left(x\right)}"
     assert latex(floor(x)**2) == r"\left\lfloor{x}\right\rfloor^{2}"
     assert latex(ceiling(x)**2) == r"\left\lceil{x}\right\rceil^{2}"
+    assert latex(frac(x)**2) == r"\operatorname{frac}{\left(x\right)}^{2}"
+
     assert latex(Min(x, 2, x**3)) == r"\min\left(2, x, x^{3}\right)"
     assert latex(Min(x, y)**2) == r"\min\left(x, y\right)^{2}"
     assert latex(Max(x, 2, x**3)) == r"\max\left(2, x, x^{3}\right)"
@@ -2286,7 +2289,7 @@ def test_DiffGeomMethods():
         r'\operatorname{d}\left(g{\left(\mathbf{x},\mathbf{y} \right)}\right)'
 
 
-def test_unit_ptinting():
+def test_unit_printing():
     assert latex(5*meter) == r'5 \text{m}'
     assert latex(3*gibibyte) == r'3 \text{gibibyte}'
     assert latex(4*microgram/second) == r'\frac{4 \mu\text{g}}{\text{s}}'

```


## Code snippets

### 1 - sympy/functions/elementary/integers.py:

Start line: 299, End line: 346

```python
class frac(Function):
    @classmethod
    def eval(cls, arg):
        from sympy import AccumBounds, im

        def _eval(arg):
            if arg is S.Infinity or arg is S.NegativeInfinity:
                return AccumBounds(0, 1)
            if arg.is_integer:
                return S.Zero
            if arg.is_number:
                if arg is S.NaN:
                    return S.NaN
                elif arg is S.ComplexInfinity:
                    return None
                else:
                    return arg - floor(arg)
            return cls(arg, evaluate=False)

        terms = Add.make_args(arg)
        real, imag = S.Zero, S.Zero
        for t in terms:
            # Two checks are needed for complex arguments
            # see issue-7649 for details
            if t.is_imaginary or (S.ImaginaryUnit*t).is_real:
                i = im(t)
                if not i.has(S.ImaginaryUnit):
                    imag += i
                else:
                    real += t
            else:
                real += t

        real = _eval(real)
        imag = _eval(imag)
        return real + S.ImaginaryUnit*imag

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_Eq(self, other):
        if isinstance(self, frac):
            if (self.rewrite(floor) == other) or \
                    (self.rewrite(ceiling) == other):
                return S.true
```
### 2 - sympy/functions/special/error_functions.py:

Start line: 759, End line: 779

```python
class erfinv(Function):

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z is S.NegativeOne:
            return S.NegativeInfinity
        elif z is S.Zero:
            return S.Zero
        elif z is S.One:
            return S.Infinity

        if isinstance(z, erf) and z.args[0].is_extended_real:
            return z.args[0]

        # Try to pull out factors of -1
        nz = z.extract_multiplicatively(-1)
        if nz is not None and (isinstance(nz, erf) and (nz.args[0]).is_extended_real):
            return -nz.args[0]

    def _eval_rewrite_as_erfcinv(self, z, **kwargs):
       return erfcinv(1-z)
```
### 3 - sympy/functions/elementary/integers.py:

Start line: 250, End line: 298

```python
class frac(Function):
    r"""Represents the fractional part of x

    For real numbers it is defined [1]_ as

    .. math::
        x - \left\lfloor{x}\right\rfloor

    Examples
    ========

    >>> from sympy import Symbol, frac, Rational, floor, ceiling, I
    >>> frac(Rational(4, 3))
    1/3
    >>> frac(-Rational(4, 3))
    2/3

    returns zero for integer arguments

    >>> n = Symbol('n', integer=True)
    >>> frac(n)
    0

    rewrite as floor

    >>> x = Symbol('x')
    >>> frac(x).rewrite(floor)
    x - floor(x)

    for complex arguments

    >>> r = Symbol('r', real=True)
    >>> t = Symbol('t', real=True)
    >>> frac(t + I*r)
    I*frac(r) + frac(t)

    See Also
    ========

    sympy.functions.elementary.integers.floor
    sympy.functions.elementary.integers.ceiling

    References
    ===========

    .. [1] https://en.wikipedia.org/wiki/Fractional_part
    .. [2] http://mathworld.wolfram.com/FractionalPart.html

    """
```
### 4 - sympy/functions/special/error_functions.py:

Start line: 480, End line: 505

```python
class erfi(Function):

    @classmethod
    def eval(cls, z):
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z is S.Zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity

        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(-z)

        # Try to pull out factors of I
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I*nz.args[0]
            if isinstance(nz, erfcinv):
                return I*(S.One - nz.args[0])
            # Only happens with unevaluated erf2inv
            if isinstance(nz, erf2inv) and nz.args[0] is S.Zero:
                return I*nz.args[1]
```
### 5 - sympy/functions/special/zeta_functions.py:

Start line: 442, End line: 478

```python
class zeta(Function):

    @classmethod
    def eval(cls, z, a_=None):
        if a_ is None:
            z, a = list(map(sympify, (z, 1)))
        else:
            z, a = list(map(sympify, (z, a_)))

        if a.is_Number:
            if a is S.NaN:
                return S.NaN
            elif a is S.One and a_ is not None:
                return cls(z)
            # TODO Should a == 0 return S.NaN as well?

        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z is S.Infinity:
                return S.One
            elif z is S.Zero:
                return S.Half - a
            elif z is S.One:
                return S.ComplexInfinity
        if z.is_integer:
            if a.is_Integer:
                if z.is_negative:
                    zeta = (-1)**z * bernoulli(-z + 1)/(-z + 1)
                elif z.is_even and z.is_positive:
                    B, F = bernoulli(z), factorial(z)
                    zeta = ((-1)**(z/2+1) * 2**(z - 1) * B * pi**z) / F
                else:
                    return

                if a.is_negative:
                    return zeta + harmonic(abs(a), z)
                else:
                    return zeta - harmonic(a - 1, z)
```
### 6 - sympy/functions/special/error_functions.py:

Start line: 520, End line: 559

```python
class erfi(Function):

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_rewrite_as_tractable(self, z, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I*erf(I*z)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I*erfc(I*z) - I

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + S.ImaginaryUnit)*z/sqrt(pi)
        return (S.One - S.ImaginaryUnit)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + S.ImaginaryUnit)*z/sqrt(pi)
        return (S.One - S.ImaginaryUnit)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [-S.Half], -z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy import uppergamma
        return sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(S.Pi) - S.One)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(S.Pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    as_real_imag = real_to_real_as_real_imag
```
### 7 - sympy/functions/special/error_functions.py:

Start line: 353, End line: 401

```python
class erfc(Function):

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_rewrite_as_tractable(self, z, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return S.One + I*erfi(I*z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - S.ImaginaryUnit)*z/sqrt(pi)
        return S.One - (S.One + S.ImaginaryUnit)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One-S.ImaginaryUnit)*z/sqrt(pi)
        return S.One - (S.One + S.ImaginaryUnit)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return S.One - z/sqrt(pi)*meijerg([S.Half], [], [0], [-S.Half], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return S.One - 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy import uppergamma
        return S.One - sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(S.Pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(S.Pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x):
        from sympy import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return S.One
        else:
            return self.func(arg)

    as_real_imag = real_to_real_as_real_imag
```
### 8 - sympy/functions/special/error_functions.py:

Start line: 2262, End line: 2270

```python
class fresnelc(FresnelIntegral):

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([S.One/4], [S.One/2, S(5)/4], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**(S(3)/4) / (sqrt(2)*root(z**2, 4)*root(-z, 4))
                * meijerg([], [1], [S(1)/4], [S(3)/4, 0], -pi**2*z**4/16))
```
### 9 - sympy/core/numbers.py:

Start line: 3918, End line: 3978

```python
I = S.ImaginaryUnit


def sympify_fractions(f):
    return Rational(f.numerator, f.denominator, 1)

converter[fractions.Fraction] = sympify_fractions

try:
    if HAS_GMPY == 2:
        import gmpy2 as gmpy
    elif HAS_GMPY == 1:
        import gmpy
    else:
        raise ImportError

    def sympify_mpz(x):
        return Integer(long(x))

    def sympify_mpq(x):
        return Rational(long(x.numerator), long(x.denominator))

    converter[type(gmpy.mpz(1))] = sympify_mpz
    converter[type(gmpy.mpq(1, 2))] = sympify_mpq
except ImportError:
    pass


def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

converter[mpnumeric] = sympify_mpmath


def sympify_mpq(x):
    p, q = x._mpq_
    return Rational(p, q, 1)

converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpq


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

converter[complex] = sympify_complex

from .power import Pow, integer_nthroot
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()

def _register_classes():
    numbers.Number.register(Number)
    numbers.Real.register(Float)
    numbers.Rational.register(Rational)
    numbers.Rational.register(Integer)

_register_classes()
```
### 10 - sympy/functions/special/error_functions.py:

Start line: 2125, End line: 2133

```python
class fresnels(FresnelIntegral):

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One + I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return pi*z**3/6 * hyper([S(3)/4], [S(3)/2, S(7)/4], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**(S(9)/4) / (sqrt(2)*(z**2)**(S(3)/4)*(-z)**(S(3)/4))
                * meijerg([], [1], [S(3)/4], [S(1)/4, 0], -pi**2*z**4/16))
```
### 80 - sympy/functions/elementary/integers.py:

Start line: 82, End line: 163

```python
class floor(RoundFunction):
    """
    Floor is a univariate function which returns the largest integer
    value not greater than its argument. This implementation
    generalizes floor to complex numbers by taking the floor of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import floor, E, I, S, Float, Rational
    >>> floor(17)
    17
    >>> floor(Rational(23, 10))
    2
    >>> floor(2*E)
    5
    >>> floor(-Float(0.567))
    -1
    >>> floor(-I/2)
    -I
    >>> floor(S(5)/2 + 5*I/2)
    2 + 2*I

    See Also
    ========

    sympy.functions.elementary.integers.ceiling

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] http://mathworld.wolfram.com/FloorFunction.html

    """
    _dir = -1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.floor()
        elif any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[0]

    def _eval_nseries(self, x, n, logx):
        r = self.subs(x, 0)
        args = self.args[0]
        args0 = args.subs(x, 0)
        if args0 == r:
            direction = (args - args0).leadterm(x)[0]
            if direction.is_positive:
                return r
            else:
                return r - 1
        else:
            return r

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return -ceiling(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg - frac(arg)

    def _eval_Eq(self, other):
        if isinstance(self, floor):
            if (self.rewrite(ceiling) == other) or \
                    (self.rewrite(frac) == other):
                return S.true

    def __le__(self, other):
        if self.args[0] == other and other.is_real:
            return S.true
        return Le(self, other, evaluate=False)

    def __gt__(self, other):
        if self.args[0] == other and other.is_real:
            return S.false
        return Gt(self, other, evaluate=False)
```
