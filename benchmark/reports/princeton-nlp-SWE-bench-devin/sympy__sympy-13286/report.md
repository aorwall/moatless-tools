# sympy__sympy-13286

| **sympy/sympy** | `42136729bb7252803b0b52a8326a33d6e9b1e06a` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 1182 |
| **Any found context length** | 666 |
| **Avg pos** | 2.5 |
| **Min pos** | 2 |
| **Max pos** | 3 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -328,22 +328,66 @@ def periodicity(f, symbol, check=False):
 
     """
     from sympy import simplify, lcm_list
-    from sympy.functions.elementary.trigonometric import TrigonometricFunction
+    from sympy.functions.elementary.complexes import Abs
+    from sympy.functions.elementary.trigonometric import (
+        TrigonometricFunction, sin, cos, csc, sec)
     from sympy.solvers.decompogen import decompogen
+    from sympy.core.relational import Relational
+
+    def _check(orig_f, period):
+        '''Return the checked period or raise an error.'''
+        new_f = orig_f.subs(symbol, symbol + period)
+        if new_f.equals(orig_f):
+            return period
+        else:
+            raise NotImplementedError(filldedent('''
+                The period of the given function cannot be verified.
+                When `%s` was replaced with `%s + %s` in `%s`, the result
+                was `%s` which was not recognized as being the same as
+                the original function.
+                So either the period was wrong or the two forms were
+                not recognized as being equal.
+                Set check=False to obtain the value.''' %
+                (symbol, symbol, period, orig_f, new_f)))
 
     orig_f = f
     f = simplify(orig_f)
     period = None
 
-    if not f.has(symbol):
+    if symbol not in f.free_symbols:
         return S.Zero
 
+    if isinstance(f, Relational):
+        f = f.lhs - f.rhs
+
     if isinstance(f, TrigonometricFunction):
         try:
             period = f.period(symbol)
         except NotImplementedError:
             pass
 
+    if isinstance(f, Abs):
+        arg = f.args[0]
+        if isinstance(arg, (sec, csc, cos)):
+            # all but tan and cot might have a
+            # a period that is half as large
+            # so recast as sin
+            arg = sin(arg.args[0])
+        period = periodicity(arg, symbol)
+        if period is not None and isinstance(arg, sin):
+            # the argument of Abs was a trigonometric other than
+            # cot or tan; test to see if the half-period
+            # is valid. Abs(arg) has behaviour equivalent to
+            # orig_f, so use that for test:
+            orig_f = Abs(arg)
+            try:
+                return _check(orig_f, period/2)
+            except NotImplementedError as err:
+                if check:
+                    raise NotImplementedError(err)
+            # else let new orig_f and period be
+            # checked below
+
     if f.is_Pow:
         base, expo = f.args
         base_has_sym = base.has(symbol)
@@ -388,14 +432,7 @@ def periodicity(f, symbol, check=False):
 
     if period is not None:
         if check:
-            if orig_f.subs(symbol, symbol + period) == orig_f:
-                return period
-
-            else:
-                raise NotImplementedError(filldedent('''
-                    The period of the given function cannot be verified.
-                    Set check=False to obtain the value.'''))
-
+            return _check(orig_f, period)
         return period
 
     return None
diff --git a/sympy/solvers/decompogen.py b/sympy/solvers/decompogen.py
--- a/sympy/solvers/decompogen.py
+++ b/sympy/solvers/decompogen.py
@@ -1,5 +1,7 @@
-from sympy.core import Function, Pow, sympify
+from sympy.core import (Function, Pow, sympify, Expr)
+from sympy.core.relational import Relational
 from sympy.polys import Poly, decompose
+from sympy.utilities.misc import func_name
 
 
 def decompogen(f, symbol):
@@ -31,6 +33,11 @@ def decompogen(f, symbol):
 
     """
     f = sympify(f)
+    if not isinstance(f, Expr) or isinstance(f, Relational):
+        raise TypeError('expecting Expr but got: `%s`' % func_name(f))
+    if symbol not in f.free_symbols:
+        return [f]
+
     result = []
 
     # ===== Simple Functions ===== #

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/calculus/util.py | 331 | 338 | 3 | 2 | 1182
| sympy/calculus/util.py | 391 | 398 | 2 | 2 | 666
| sympy/solvers/decompogen.py | 1 | 1 | - | - | -
| sympy/solvers/decompogen.py | 34 | 34 | - | - | -


## Problem Statement

```
periodicity(Abs(sin(x)),x) return 2*pi
periodicity(Abs(sin(x)),x) returns 2*pi instead of pi
\`\`\`
>>> from sympy import *
>>> x=Symbol('x')
>>> periodicity(Abs(sin(x)),x,check=True)
2*pi
>>> periodicity(Abs(sin(x)),x)
2*pi
\`\`\`

#13205 periodicity(x > 2, x) give recursion error and #13207
It fixes issue #13205 it will stop any relational Expression from entering into infinite recursion and return None
It improves the periodicity of absolute trigonometric function issue #13207

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/plotting/intervalmath/lib_interval.py | 92 | 118| 263 | 263 | 3651 | 
| **-> 2 <-** | **2 sympy/calculus/util.py** | 347 | 401| 403 | 666 | 12994 | 
| **-> 3 <-** | **2 sympy/calculus/util.py** | 270 | 345| 516 | 1182 | 12994 | 
| 4 | 2 sympy/plotting/intervalmath/lib_interval.py | 121 | 149| 293 | 1475 | 12994 | 
| 5 | 3 sympy/functions/elementary/trigonometric.py | 66 | 90| 176 | 1651 | 36265 | 
| 6 | 3 sympy/functions/elementary/trigonometric.py | 198 | 252| 376 | 2027 | 36265 | 
| 7 | 3 sympy/functions/elementary/trigonometric.py | 466 | 516| 321 | 2348 | 36265 | 
| 8 | **3 sympy/calculus/util.py** | 404 | 421| 115 | 2463 | 36265 | 
| 9 | 4 sympy/functions/elementary/complexes.py | 884 | 911| 304 | 2767 | 45162 | 
| 10 | 4 sympy/functions/elementary/complexes.py | 834 | 859| 228 | 2995 | 45162 | 
| 11 | 5 sympy/functions/special/spherical_harmonics.py | 1 | 14| 125 | 3120 | 48334 | 
| 12 | 5 sympy/plotting/intervalmath/lib_interval.py | 223 | 251| 241 | 3361 | 48334 | 
| 13 | 6 sympy/series/sequences.py | 458 | 504| 291 | 3652 | 55651 | 
| 14 | 7 sympy/core/numbers.py | 3489 | 3557| 429 | 4081 | 84755 | 
| 15 | 7 sympy/functions/elementary/trigonometric.py | 1181 | 1231| 316 | 4397 | 84755 | 
| 16 | 7 sympy/plotting/intervalmath/lib_interval.py | 352 | 387| 283 | 4680 | 84755 | 
| 17 | 7 sympy/functions/elementary/trigonometric.py | 1514 | 1580| 692 | 5372 | 84755 | 
| 18 | 7 sympy/functions/elementary/trigonometric.py | 359 | 416| 528 | 5900 | 84755 | 
| 19 | 7 sympy/functions/elementary/trigonometric.py | 2391 | 2412| 182 | 6082 | 84755 | 
| 20 | 8 sympy/integrals/trigonometry.py | 232 | 274| 402 | 6484 | 87829 | 
| 21 | 8 sympy/plotting/intervalmath/lib_interval.py | 288 | 307| 200 | 6684 | 87829 | 
| 22 | 8 sympy/functions/elementary/trigonometric.py | 886 | 936| 312 | 6996 | 87829 | 
| 23 | 8 sympy/functions/elementary/trigonometric.py | 2414 | 2420| 136 | 7132 | 87829 | 
| 24 | 8 sympy/functions/elementary/trigonometric.py | 418 | 463| 438 | 7570 | 87829 | 
| 25 | 8 sympy/functions/elementary/trigonometric.py | 126 | 195| 515 | 8085 | 87829 | 
| 26 | 8 sympy/plotting/intervalmath/lib_interval.py | 152 | 177| 188 | 8273 | 87829 | 
| 27 | 8 sympy/functions/elementary/trigonometric.py | 1663 | 1727| 453 | 8726 | 87829 | 
| 28 | 8 sympy/integrals/trigonometry.py | 277 | 317| 381 | 9107 | 87829 | 
| 29 | 8 sympy/integrals/trigonometry.py | 1 | 30| 261 | 9368 | 87829 | 
| 30 | 9 examples/advanced/gibbs_phenomenon.py | 129 | 153| 226 | 9594 | 88943 | 
| 31 | 9 sympy/functions/elementary/trigonometric.py | 2088 | 2104| 170 | 9764 | 88943 | 
| 32 | 10 sympy/series/fourier.py | 411 | 487| 575 | 10339 | 92623 | 
| 33 | 10 sympy/functions/elementary/trigonometric.py | 1942 | 1956| 154 | 10493 | 92623 | 
| 34 | 10 sympy/series/fourier.py | 375 | 408| 236 | 10729 | 92623 | 
| 35 | 10 sympy/functions/elementary/trigonometric.py | 1355 | 1429| 707 | 11436 | 92623 | 
| 36 | 10 examples/advanced/gibbs_phenomenon.py | 1 | 19| 122 | 11558 | 92623 | 
| 37 | 11 sympy/polys/ring_series.py | 1221 | 1264| 394 | 11952 | 110815 | 
| 38 | 11 sympy/functions/elementary/trigonometric.py | 652 | 693| 385 | 12337 | 110815 | 
| 39 | 12 sympy/physics/optics/waves.py | 108 | 142| 203 | 12540 | 112986 | 
| 40 | 12 sympy/plotting/intervalmath/lib_interval.py | 1 | 32| 211 | 12751 | 112986 | 
| 41 | 12 sympy/polys/ring_series.py | 1383 | 1451| 699 | 13450 | 112986 | 
| 42 | 12 sympy/plotting/intervalmath/lib_interval.py | 254 | 285| 291 | 13741 | 112986 | 
| 43 | 12 sympy/series/fourier.py | 93 | 154| 311 | 14052 | 112986 | 
| 44 | 13 sympy/functions/special/bessel.py | 1295 | 1308| 213 | 14265 | 128178 | 
| 45 | 13 sympy/functions/elementary/complexes.py | 861 | 882| 188 | 14453 | 128178 | 
| 46 | 13 sympy/series/fourier.py | 39 | 90| 466 | 14919 | 128178 | 
| 47 | 13 sympy/functions/elementary/complexes.py | 913 | 927| 143 | 15062 | 128178 | 
| 48 | 13 sympy/functions/elementary/trigonometric.py | 1649 | 1660| 135 | 15197 | 128178 | 
| 49 | 13 sympy/functions/elementary/trigonometric.py | 2278 | 2284| 132 | 15329 | 128178 | 
| 50 | 14 sympy/plotting/intervalmath/interval_arithmetic.py | 356 | 375| 147 | 15476 | 131599 | 
| 51 | 14 sympy/functions/elementary/trigonometric.py | 1583 | 1647| 454 | 15930 | 131599 | 
| 52 | 14 sympy/functions/special/bessel.py | 1127 | 1140| 229 | 16159 | 131599 | 
| 53 | 14 sympy/functions/elementary/complexes.py | 959 | 1012| 494 | 16653 | 131599 | 
| 54 | 14 sympy/functions/elementary/trigonometric.py | 846 | 883| 325 | 16978 | 131599 | 
| 55 | 14 sympy/functions/elementary/trigonometric.py | 2253 | 2276| 214 | 17192 | 131599 | 
| 56 | 14 sympy/series/fourier.py | 1 | 16| 132 | 17324 | 131599 | 
| 57 | 14 sympy/functions/elementary/trigonometric.py | 254 | 357| 896 | 18220 | 131599 | 
| 58 | 15 sympy/holonomic/holonomic.py | 1237 | 1298| 767 | 18987 | 156405 | 
| 59 | 15 sympy/series/sequences.py | 506 | 544| 333 | 19320 | 156405 | 
| 60 | 15 sympy/plotting/intervalmath/lib_interval.py | 57 | 89| 265 | 19585 | 156405 | 
| 61 | 15 sympy/functions/elementary/trigonometric.py | 782 | 831| 596 | 20181 | 156405 | 
| 62 | 15 sympy/plotting/intervalmath/interval_arithmetic.py | 377 | 403| 210 | 20391 | 156405 | 
| 63 | 15 sympy/functions/elementary/trigonometric.py | 1069 | 1090| 255 | 20646 | 156405 | 
| 64 | 15 sympy/plotting/intervalmath/interval_arithmetic.py | 315 | 354| 341 | 20987 | 156405 | 
| 65 | **15 sympy/calculus/util.py** | 461 | 613| 1499 | 22486 | 156405 | 
| 66 | 15 sympy/functions/elementary/trigonometric.py | 744 | 780| 767 | 23253 | 156405 | 
| 67 | 15 sympy/functions/elementary/trigonometric.py | 1339 | 1353| 134 | 23387 | 156405 | 
| 68 | 15 sympy/polys/ring_series.py | 1609 | 1634| 235 | 23622 | 156405 | 
| 69 | 15 sympy/integrals/trigonometry.py | 122 | 229| 1173 | 24795 | 156405 | 
| 70 | 16 sympy/ntheory/continued_fraction.py | 166 | 208| 242 | 25037 | 158330 | 
| 71 | 16 sympy/polys/ring_series.py | 1453 | 1524| 701 | 25738 | 158330 | 
| 72 | 16 sympy/functions/elementary/trigonometric.py | 1121 | 1178| 481 | 26219 | 158330 | 
| 73 | 16 sympy/functions/elementary/trigonometric.py | 1729 | 1741| 149 | 26368 | 158330 | 
| 74 | 16 sympy/functions/elementary/trigonometric.py | 2648 | 2741| 873 | 27241 | 158330 | 
| 75 | 16 sympy/series/fourier.py | 290 | 316| 240 | 27481 | 158330 | 
| 76 | 17 sympy/core/function.py | 661 | 703| 469 | 27950 | 181419 | 
| 77 | 17 sympy/functions/elementary/trigonometric.py | 1 | 18| 206 | 28156 | 181419 | 
| 78 | 17 sympy/series/fourier.py | 19 | 36| 225 | 28381 | 181419 | 
| 79 | 18 sympy/functions/elementary/exponential.py | 120 | 157| 308 | 28689 | 188124 | 
| 80 | 19 sympy/functions/special/hyper.py | 569 | 610| 384 | 29073 | 197930 | 
| 81 | 19 sympy/functions/elementary/trigonometric.py | 1958 | 1993| 299 | 29372 | 197930 | 
| 82 | 19 sympy/functions/elementary/trigonometric.py | 1431 | 1471| 369 | 29741 | 197930 | 
| 83 | 19 sympy/functions/elementary/trigonometric.py | 1838 | 1892| 385 | 30126 | 197930 | 
| 84 | 19 sympy/series/fourier.py | 263 | 288| 217 | 30343 | 197930 | 


## Missing Patch Files

 * 1: sympy/calculus/util.py
 * 2: sympy/solvers/decompogen.py

### Hint

```
Can I have this issue ?
Note by the docstring we are not guaranteed to get the fundamental period. But of course it would be good to improve the answer if possible. 
@souravghosh97 Can you add few tests that could justify your changes. 
@smichr I don't know why this test fails.But the test runs locally
[https://travis-ci.org/sympy/sympy/builds/272842924](https://travis-ci.org/sympy/sympy/builds/272842924)
Some minor edits including a couple of new tests. Let's see if tests pass this time. As I recall, the error was just a time out.
```

## Patch

```diff
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -328,22 +328,66 @@ def periodicity(f, symbol, check=False):
 
     """
     from sympy import simplify, lcm_list
-    from sympy.functions.elementary.trigonometric import TrigonometricFunction
+    from sympy.functions.elementary.complexes import Abs
+    from sympy.functions.elementary.trigonometric import (
+        TrigonometricFunction, sin, cos, csc, sec)
     from sympy.solvers.decompogen import decompogen
+    from sympy.core.relational import Relational
+
+    def _check(orig_f, period):
+        '''Return the checked period or raise an error.'''
+        new_f = orig_f.subs(symbol, symbol + period)
+        if new_f.equals(orig_f):
+            return period
+        else:
+            raise NotImplementedError(filldedent('''
+                The period of the given function cannot be verified.
+                When `%s` was replaced with `%s + %s` in `%s`, the result
+                was `%s` which was not recognized as being the same as
+                the original function.
+                So either the period was wrong or the two forms were
+                not recognized as being equal.
+                Set check=False to obtain the value.''' %
+                (symbol, symbol, period, orig_f, new_f)))
 
     orig_f = f
     f = simplify(orig_f)
     period = None
 
-    if not f.has(symbol):
+    if symbol not in f.free_symbols:
         return S.Zero
 
+    if isinstance(f, Relational):
+        f = f.lhs - f.rhs
+
     if isinstance(f, TrigonometricFunction):
         try:
             period = f.period(symbol)
         except NotImplementedError:
             pass
 
+    if isinstance(f, Abs):
+        arg = f.args[0]
+        if isinstance(arg, (sec, csc, cos)):
+            # all but tan and cot might have a
+            # a period that is half as large
+            # so recast as sin
+            arg = sin(arg.args[0])
+        period = periodicity(arg, symbol)
+        if period is not None and isinstance(arg, sin):
+            # the argument of Abs was a trigonometric other than
+            # cot or tan; test to see if the half-period
+            # is valid. Abs(arg) has behaviour equivalent to
+            # orig_f, so use that for test:
+            orig_f = Abs(arg)
+            try:
+                return _check(orig_f, period/2)
+            except NotImplementedError as err:
+                if check:
+                    raise NotImplementedError(err)
+            # else let new orig_f and period be
+            # checked below
+
     if f.is_Pow:
         base, expo = f.args
         base_has_sym = base.has(symbol)
@@ -388,14 +432,7 @@ def periodicity(f, symbol, check=False):
 
     if period is not None:
         if check:
-            if orig_f.subs(symbol, symbol + period) == orig_f:
-                return period
-
-            else:
-                raise NotImplementedError(filldedent('''
-                    The period of the given function cannot be verified.
-                    Set check=False to obtain the value.'''))
-
+            return _check(orig_f, period)
         return period
 
     return None
diff --git a/sympy/solvers/decompogen.py b/sympy/solvers/decompogen.py
--- a/sympy/solvers/decompogen.py
+++ b/sympy/solvers/decompogen.py
@@ -1,5 +1,7 @@
-from sympy.core import Function, Pow, sympify
+from sympy.core import (Function, Pow, sympify, Expr)
+from sympy.core.relational import Relational
 from sympy.polys import Poly, decompose
+from sympy.utilities.misc import func_name
 
 
 def decompogen(f, symbol):
@@ -31,6 +33,11 @@ def decompogen(f, symbol):
 
     """
     f = sympify(f)
+    if not isinstance(f, Expr) or isinstance(f, Relational):
+        raise TypeError('expecting Expr but got: `%s`' % func_name(f))
+    if symbol not in f.free_symbols:
+        return [f]
+
     result = []
 
     # ===== Simple Functions ===== #

```

## Test Patch

```diff
diff --git a/sympy/calculus/tests/test_util.py b/sympy/calculus/tests/test_util.py
--- a/sympy/calculus/tests/test_util.py
+++ b/sympy/calculus/tests/test_util.py
@@ -94,6 +94,14 @@ def test_periodicity():
     assert periodicity(exp(x)**sin(x), x) is None
     assert periodicity(sin(x)**y, y) is None
 
+    assert periodicity(Abs(sin(Abs(sin(x)))),x) == pi
+    assert all(periodicity(Abs(f(x)),x) == pi for f in (
+        cos, sin, sec, csc, tan, cot))
+    assert periodicity(Abs(sin(tan(x))), x) == pi
+    assert periodicity(Abs(sin(sin(x) + tan(x))), x) == 2*pi
+    assert periodicity(sin(x) > S.Half, x) is 2*pi
+
+    assert periodicity(x > 2, x) is None
     assert periodicity(x**3 - x**2 + 1, x) is None
     assert periodicity(Abs(x), x) is None
     assert periodicity(Abs(x**2 - 1), x) is None
@@ -105,8 +113,9 @@ def test_periodicity_check():
 
     assert periodicity(tan(x), x, check=True) == pi
     assert periodicity(sin(x) + cos(x), x, check=True) == 2*pi
-    raises(NotImplementedError, lambda: periodicity(sec(x), x, check=True))
-    raises(NotImplementedError, lambda: periodicity(sin(x*y), x, check=True))
+    assert periodicity(sec(x), x) == 2*pi
+    assert periodicity(sin(x*y), x) == 2*pi/abs(y)
+    assert periodicity(Abs(sec(sec(x))), x) == pi
 
 
 def test_lcim():
diff --git a/sympy/solvers/tests/test_decompogen.py b/sympy/solvers/tests/test_decompogen.py
--- a/sympy/solvers/tests/test_decompogen.py
+++ b/sympy/solvers/tests/test_decompogen.py
@@ -1,7 +1,7 @@
 from sympy.solvers.decompogen import decompogen, compogen
 from sympy import sin, cos, sqrt, Abs
 from sympy import symbols
-from sympy.utilities.pytest import XFAIL
+from sympy.utilities.pytest import XFAIL, raises
 
 x, y = symbols('x y')
 
@@ -14,6 +14,9 @@ def test_decompogen():
     assert decompogen(Abs(cos(x)**2 + 3*cos(x) - 4), x) == [Abs(x), x**2 + 3*x - 4, cos(x)]
     assert decompogen(sin(x)**2 + sin(x) - sqrt(3)/2, x) == [x**2 + x - sqrt(3)/2, sin(x)]
     assert decompogen(Abs(cos(y)**2 + 3*cos(x) - 4), x) == [Abs(x), 3*x + cos(y)**2 - 4, cos(x)]
+    assert decompogen(x, y) == [x]
+    assert decompogen(1, x) == [1]
+    raises(TypeError, lambda: decompogen(x < 5, x))
 
 
 def test_decompogen_poly():

```


## Code snippets

### 1 - sympy/plotting/intervalmath/lib_interval.py:

Start line: 92, End line: 118

```python
#periodic
def sin(x):
    """evaluates the sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.sin(x.start), np.sin(x.end))
        end = max(np.sin(x.start), np.sin(x.end))
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na - 1) // 4 != (nb - 1) // 4:
                #sin has max
                end = 1
            if (na - 3) // 4 != (nb - 3) // 4:
                #sin has min
                start = -1
            return interval(start, end)
    else:
        raise NotImplementedError
```
### 2 - sympy/calculus/util.py:

Start line: 347, End line: 401

```python
def periodicity(f, symbol, check=False):
    # ... other code

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if base_has_sym and not expo_has_sym:
            period = periodicity(base, symbol)

        elif expo_has_sym and not base_has_sym:
            period = periodicity(expo, symbol)

        else:
            period = _periodicity(f.args, symbol)

    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or coeff is not S.One:
            period = periodicity(g, symbol)

        else:
            period = _periodicity(g.args, symbol)

    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)

        period = _periodicity(g.args, symbol)

    elif period is None:
        from sympy.solvers.decompogen import compogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g != orig_f and g != f: # Fix for issue 12620
                    period = periodicity(g, symbol)
                    if period is not None:
                        break

    if period is not None:
        if check:
            if orig_f.subs(symbol, symbol + period) == orig_f:
                return period

            else:
                raise NotImplementedError(filldedent('''
                    The period of the given function cannot be verified.
                    Set check=False to obtain the value.'''))

        return period

    return None
```
### 3 - sympy/calculus/util.py:

Start line: 270, End line: 345

```python
def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : Expr.
        The concerned function.
    symbol : Symbol
        The variable for which the period is to be determined.
    check : Boolean
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        `None` is returned when the function is aperiodic or has a complex period.
        The value of `0` is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as `exp`, `sinh` is evaluated to `None`.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the `check` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to `False` by default.

    Examples
    ========
    >>> from sympy import Symbol, sin, cos, tan, exp
    >>> from sympy.calculus.util import periodicity
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)

    """
    from sympy import simplify, lcm_list
    from sympy.functions.elementary.trigonometric import TrigonometricFunction
    from sympy.solvers.decompogen import decompogen

    orig_f = f
    f = simplify(orig_f)
    period = None

    if not f.has(symbol):
        return S.Zero

    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass
    # ... other code
```
### 4 - sympy/plotting/intervalmath/lib_interval.py:

Start line: 121, End line: 149

```python
#periodic
def cos(x):
    """Evaluates the cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not (np.isfinite(x.start) and np.isfinite(x.end)):
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.cos(x.start), np.cos(x.end))
        end = max(np.cos(x.start), np.cos(x.end))
        if nb - na > 4:
            #differ more than 2*pi
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            #in the same quadarant
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na) // 4 != (nb) // 4:
                #cos has max
                end = 1
            if (na - 2) // 4 != (nb - 2) // 4:
                #cos has min
                start = -1
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError
```
### 5 - sympy/functions/elementary/trigonometric.py:

Start line: 66, End line: 90

```python
class TrigonometricFunction(Function):

    def _period(self, general_period, symbol=None):
        f = self.args[0]
        if symbol is None:
            symbol = tuple(f.free_symbols)[0]

        if not f.has(symbol):
            return S.Zero

        if f == symbol:
            return general_period

        if symbol in f.free_symbols:
            p, q = Wild('p'), Wild('q')
            if f.is_Mul:
                g, h = f.as_independent(symbol)
                if h == symbol:
                    return general_period/abs(g)

            if f.is_Add:
                a, h = f.as_independent(symbol)
                g, h = h.as_independent(symbol, as_Add=False)
                if h == symbol:
                    return general_period/abs(g)

        raise NotImplementedError("Use the periodicity function instead.")
```
### 6 - sympy/functions/elementary/trigonometric.py:

Start line: 198, End line: 252

```python
class sin(TrigonometricFunction):
    """
    The sine function.

    Returns the sine of x (measured in radians).

    Notes
    =====

    This function will evaluate automatically in the
    case x/pi is some rational number [4]_.  For example,
    if x is a multiple of pi, pi/2, pi/3, pi/4 and pi/6.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] http://dlmf.nist.gov/4.14
    .. [3] http://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] http://mathworld.wolfram.com/TrigonometryAngles.html
    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)
```
### 7 - sympy/functions/elementary/trigonometric.py:

Start line: 466, End line: 516

```python
class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Notes
    =====

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sin, csc, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] http://dlmf.nist.gov/4.14
    .. [3] http://functions.wolfram.com/ElementaryFunctions/Cos
    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)
```
### 8 - sympy/calculus/util.py:

Start line: 404, End line: 421

```python
def _periodicity(args, symbol):
    """Helper for periodicity to find the period of a list of simpler
    functions. It uses the `lcim` method to find the least common period of
    all the functions.
    """
    periods = []
    for f in args:
        period = periodicity(f, symbol)
        if period is None:
            return None

        if period is not S.Zero:
            periods.append(period)

    if len(periods) > 1:
        return lcim(periods)

    return periods[0]
```
### 9 - sympy/functions/elementary/complexes.py:

Start line: 884, End line: 911

```python
class periodic_argument(Function):

    @classmethod
    def eval(cls, ar, period):
        # Our strategy is to evaluate the argument on the Riemann surface of the
        # logarithm, and then reduce.
        # NOTE evidently this means it is a rather bad idea to use this with
        # period != 2*pi and non-polar numbers.
        from sympy import ceiling, oo, atan2, atan, polar_lift, pi, Mul
        if not period.is_positive:
            return None
        if period == oo and isinstance(ar, principal_branch):
            return periodic_argument(*ar.args)
        if ar.func is polar_lift and period >= 2*pi:
            return periodic_argument(ar.args[0], period)
        if ar.is_Mul:
            newargs = [x for x in ar.args if not x.is_positive]
            if len(newargs) != len(ar.args):
                return periodic_argument(Mul(*newargs), period)
        unbranched = cls._getunbranched(ar)
        if unbranched is None:
            return None
        if unbranched.has(periodic_argument, atan2, arg, atan):
            return None
        if period == oo:
            return unbranched
        if period != oo:
            n = ceiling(unbranched/period - S(1)/2)*period
            if not n.has(ceiling):
                return unbranched - n
```
### 10 - sympy/functions/elementary/complexes.py:

Start line: 834, End line: 859

```python
class periodic_argument(Function):
    """
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period P, always return a value in
    (-P/2, P/2], by using exp(P*I) == 1.

    >>> from sympy import exp, exp_polar, periodic_argument, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp(5*I*pi))
    pi
    >>> unbranched_argument(exp_polar(5*I*pi))
    5*pi
    >>> periodic_argument(exp_polar(5*I*pi), 2*pi)
    pi
    >>> periodic_argument(exp_polar(5*I*pi), 3*pi)
    -pi
    >>> periodic_argument(exp_polar(5*I*pi), pi)
    0

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    principal_branch
    """
```
### 65 - sympy/calculus/util.py:

Start line: 461, End line: 613

```python
class AccumulationBounds(AtomicExpr):
    r"""
    # Note AccumulationBounds has an alias: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that a <= b.

    `\langle a, b\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\langle -\infty, b\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\langle a, \infty \rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\langle -\infty, \infty \rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    `oo` and `-oo` are added to the second and third definition respectively,
    since if either `-oo` or `oo` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, `1/AccumBounds(0, 1) = AccumBounds(1, oo)`, and the limit at
    `0` is not one-sided. As x tends to `0-`, then `1/x -> -oo`, so `-oo`
    should be interpreted as belonging to `AccumBounds(1, oo)` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of cos(x) + 1 are non-negative.
    (AccumBounds(-1, 1) + 1 = AccumBounds(0, 2))

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X * Y = \{ x*y \mid x \in X \cap y \in Y\}`

    There is, however, no consensus on Interval division.

    `X / Y = \{ z \mid \exists x \in X, y \in Y \mid y \neq 0, z = x/y\}`

    Note: According to this definition the quotient of two AccumulationBounds
    may not be a AccumulationBounds object but rather a union of
    AccumulationBounds.

    Note
    ====

    The main focus in the interval arithmetic is on the simplest way to
    calculate upper and lower endpoints for the range of values of a
    function in one or more variables. These barriers are not necessarily
    the supremum or infimum, since the precise calculation of those values
    can be difficult or impossible.

    Examples
    ========

    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x

    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    <1, 3>

    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    <-2, 1>

    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    <-3, 3>

    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    <3, 10>

    The exponentiation of AccumulationBounds is defined
    as follows:

    If 0 does not belong to `X` or `n > 0` then

    `X^n = \{ x^n \mid x \in X\}`

    otherwise

    `X^n = \{ x^n \mid x \neq 0, x \in X\} \cup \{-\infty, \infty\}`

    Here for fractional `n`, the part of `X` resulting in a complex
    AccumulationBounds object is neglected.

    >>> AccumBounds(-1, 4)**(S(1)/2)
    <0, 2>

    >>> AccumBounds(1, 2)**2
    <1, 4>

    >>> AccumBounds(-1, oo)**(-1)
    <-oo, oo>

    Note: `<a, b>^2` is not same as `<a, b>*<a, b>`

    >>> AccumBounds(-1, 1)**2
    <0, 1>

    >>> AccumBounds(1, 3) < 4
    True

    >>> AccumBounds(1, 3) < -1
    False

    Some elementary functions can also take AccumulationBounds as input.
    A function `f` evaluated for some real AccumulationBounds `<a, b>`
    is defined as `f(\langle a, b\rangle) = \{ f(x) \mid a \le x \le b \}`

    >>> sin(AccumBounds(pi/6, pi/3))
    <1/2, sqrt(3)/2>

    >>> exp(AccumBounds(0, 1))
    <1, E>

    >>> log(AccumBounds(1, E))
    <0, 1>

    Some symbol in an expression can be substituted for a AccumulationBounds
    object. But it doesn't necessarily evaluate the AccumulationBounds for
    that expression.

    Same expression can be evaluated to different values depending upon
    the form it is used for substituion. For example:

    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    <-1, 4>

    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    <0, 4>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] http://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """
```
