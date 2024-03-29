# sympy__sympy-13429

| **sympy/sympy** | `ac03325b44485e603992a0bb783536a9f8a9152f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 175 |
| **Avg pos** | 374.0 |
| **Min pos** | 1 |
| **Max pos** | 76 |
| **Top file pos** | 1 |
| **Missing snippets** | 16 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1259,13 +1259,13 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             if other.is_irrational:
                 return False
             return other.__eq__(self)
-        if isinstance(other, Float):
+        if other.is_Float:
             return bool(mlib.mpf_eq(self._mpf_, other._mpf_))
-        if isinstance(other, Number):
+        if other.is_Number:
             # numbers should compare at the same precision;
             # all _as_mpf_val routines should be sure to abide
             # by the request to change the prec if necessary; if
@@ -1283,11 +1283,14 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__lt__(self)
-        if other.is_comparable:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_gt(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__gt__(self, other)
@@ -1297,11 +1300,14 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__le__(self)
-        if other.is_comparable:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_ge(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__ge__(self, other)
@@ -1311,11 +1317,14 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__gt__(self)
-        if other.is_real and other.is_number:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_lt(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__lt__(self, other)
@@ -1325,11 +1334,14 @@ def __le__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__ge__(self)
-        if other.is_real and other.is_number:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_le(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__le__(self, other)
@@ -1720,16 +1732,16 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             if other.is_irrational:
                 return False
             return other.__eq__(self)
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 # a Rational is always in reduced form so will never be 2/4
                 # so we can just check equivalence of args
                 return self.p == other.p and self.q == other.q
-            if isinstance(other, Float):
+            if other.is_Float:
                 return mlib.mpf_eq(self._as_mpf_val(other._prec), other._mpf_)
         return False
 
@@ -1741,13 +1753,13 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__lt__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q > self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_gt(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1759,13 +1771,13 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__le__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                  return _sympify(bool(self.p*other.q >= self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_ge(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1777,13 +1789,13 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__gt__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q < self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_lt(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1796,12 +1808,12 @@ def __le__(self, other):
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
         expr = self
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__ge__(self)
-        elif isinstance(other, Number):
-            if isinstance(other, Rational):
+        elif other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q <= self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_le(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -2119,7 +2131,7 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p > other.p)
         return Rational.__gt__(self, other)
 
@@ -2128,7 +2140,7 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p < other.p)
         return Rational.__lt__(self, other)
 
@@ -2137,7 +2149,7 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p >= other.p)
         return Rational.__ge__(self, other)
 
@@ -2146,7 +2158,7 @@ def __le__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p <= other.p)
         return Rational.__le__(self, other)
 
@@ -3344,7 +3356,7 @@ def __eq__(self, other):
             return NotImplemented
         if self is other:
             return True
-        if isinstance(other, Number) and self.is_irrational:
+        if other.is_Number and self.is_irrational:
             return False
 
         return False    # NumberSymbol != non-(Number|self)
@@ -3352,61 +3364,15 @@ def __eq__(self, other):
     def __ne__(self, other):
         return not self == other
 
-    def __lt__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if self is other:
-            return S.false
-        if isinstance(other, Number):
-            approx = self.approximation_interval(other.__class__)
-            if approx is not None:
-                l, u = approx
-                if other < l:
-                    return S.false
-                if other > u:
-                    return S.true
-            return _sympify(self.evalf() < other)
-        if other.is_real and other.is_number:
-            other = other.evalf()
-            return _sympify(self.evalf() < other)
-        return Expr.__lt__(self, other)
-
     def __le__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s <= %s" % (self, other))
         if self is other:
             return S.true
-        if other.is_real and other.is_number:
-            other = other.evalf()
-        if isinstance(other, Number):
-            return _sympify(self.evalf() <= other)
         return Expr.__le__(self, other)
 
-    def __gt__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s > %s" % (self, other))
-        r = _sympify((-self) < (-other))
-        if r in (S.true, S.false):
-            return r
-        else:
-            return Expr.__gt__(self, other)
-
     def __ge__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        r = _sympify((-self) <= (-other))
-        if r in (S.true, S.false):
-            return r
-        else:
-            return Expr.__ge__(self, other)
+        if self is other:
+            return S.true
+        return Expr.__ge__(self, other)
 
     def __int__(self):
         # subclass with appropriate return value

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/numbers.py | 1262 | 1268 | 76 | 1 | 23900
| sympy/core/numbers.py | 1286 | 1290 | 53 | 1 | 16105
| sympy/core/numbers.py | 1300 | 1304 | 56 | 1 | 16746
| sympy/core/numbers.py | 1314 | 1318 | 63 | 1 | 19378
| sympy/core/numbers.py | 1328 | 1332 | 67 | 1 | 20870
| sympy/core/numbers.py | 1723 | 1732 | 5 | 1 | 869
| sympy/core/numbers.py | 1744 | 1750 | 2 | 1 | 350
| sympy/core/numbers.py | 1762 | 1768 | 1 | 1 | 175
| sympy/core/numbers.py | 1780 | 1786 | 3 | 1 | 525
| sympy/core/numbers.py | 1799 | 1804 | 4 | 1 | 700
| sympy/core/numbers.py | 2122 | 2122 | 11 | 1 | 2256
| sympy/core/numbers.py | 2131 | 2131 | 11 | 1 | 2256
| sympy/core/numbers.py | 2140 | 2140 | 11 | 1 | 2256
| sympy/core/numbers.py | 2149 | 2149 | 11 | 1 | 2256
| sympy/core/numbers.py | 3347 | 3347 | - | 1 | -
| sympy/core/numbers.py | 3355 | 3409 | - | 1 | -


## Problem Statement

```
Some comparisons between rational and irrational numbers are incorrect
If you choose just the right rational number, you can end up in a situation where it is neither less than pi, nor equal to it, nor is pi less than it. This is with sympy 1.1.1, using Python 3.6.0 from Anaconda on Ubuntu 16.04.
\`\`\`
>>> import sympy
>>> sympy.__version__
'1.1.1'
>>> r = sympy.Rational('905502432259640373/288230376151711744')
>>> r < sympy.pi
False
>>> r == sympy.pi
False
>>> sympy.pi < r
False
\`\`\`
Of course, that same number is greater than pi, even though pi is not less than it.
\`\`\`
>>> r > sympy.pi
True
\`\`\`
I believe this is a result of using evalf() to do comparisons between rationals and reals... As we can see, this particular fraction happens to be exactly equal to pi if we use the default evalf precision of 15, but not if we use more.
\`\`\`
>>> r == sympy.pi.evalf(15)
True
>>> r == sympy.pi.evalf(16)
False
\`\`\`
Hopefully this isn't a duplicate issue; I did a bit of searching for related ones, and found the likes of #12583 and #12534. I think this is different than #12583 because I'm only concerned about comparisons where one of the numbers is rational. That should always be decidable - or am I misunderstanding something about math?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/core/numbers.py** | 1762 | 1778| 175 | 175 | 29204 | 
| **-> 2 <-** | **1 sympy/core/numbers.py** | 1744 | 1760| 175 | 350 | 29204 | 
| **-> 3 <-** | **1 sympy/core/numbers.py** | 1780 | 1796| 175 | 525 | 29204 | 
| **-> 4 <-** | **1 sympy/core/numbers.py** | 1798 | 1814| 175 | 700 | 29204 | 
| **-> 5 <-** | **1 sympy/core/numbers.py** | 1723 | 1742| 169 | 869 | 29204 | 
| 6 | **1 sympy/core/numbers.py** | 1625 | 1637| 144 | 1013 | 29204 | 
| 7 | **1 sympy/core/numbers.py** | 1658 | 1698| 448 | 1461 | 29204 | 
| 8 | **1 sympy/core/numbers.py** | 1610 | 1624| 156 | 1617 | 29204 | 
| 9 | **1 sympy/core/numbers.py** | 1700 | 1721| 163 | 1780 | 29204 | 
| 10 | **1 sympy/core/numbers.py** | 1584 | 1595| 125 | 1905 | 29204 | 
| **-> 11 <-** | **1 sympy/core/numbers.py** | 2122 | 2167| 351 | 2256 | 29204 | 
| 12 | **1 sympy/core/numbers.py** | 1639 | 1656| 188 | 2444 | 29204 | 
| 13 | **1 sympy/core/numbers.py** | 1570 | 1583| 131 | 2575 | 29204 | 
| 14 | **1 sympy/core/numbers.py** | 1557 | 1569| 132 | 2707 | 29204 | 
| 15 | **1 sympy/core/numbers.py** | 1596 | 1608| 141 | 2848 | 29204 | 
| 16 | **1 sympy/core/numbers.py** | 1363 | 1455| 597 | 3445 | 29204 | 
| 17 | **1 sympy/core/numbers.py** | 1457 | 1527| 495 | 3940 | 29204 | 
| 18 | **1 sympy/core/numbers.py** | 1831 | 1854| 182 | 4122 | 29204 | 
| 19 | **1 sympy/core/numbers.py** | 2032 | 2120| 750 | 4872 | 29204 | 
| 20 | **1 sympy/core/numbers.py** | 1944 | 1985| 321 | 5193 | 29204 | 
| 21 | **1 sympy/core/numbers.py** | 134 | 155| 209 | 5402 | 29204 | 
| 22 | 2 sympy/polys/domains/pythonrational.py | 1 | 103| 640 | 6042 | 31247 | 
| 23 | **2 sympy/core/numbers.py** | 2169 | 2247| 747 | 6789 | 31247 | 
| 24 | 3 sympy/concrete/guess.py | 109 | 167| 552 | 7341 | 36204 | 
| 25 | **3 sympy/core/numbers.py** | 1529 | 1555| 205 | 7546 | 36204 | 
| 26 | **3 sympy/core/numbers.py** | 3498 | 3566| 429 | 7975 | 36204 | 
| 27 | 4 sympy/core/power.py | 1104 | 1120| 165 | 8140 | 50040 | 
| 28 | 4 sympy/polys/domains/pythonrational.py | 208 | 288| 500 | 8640 | 50040 | 
| 29 | **4 sympy/core/numbers.py** | 2248 | 2300| 471 | 9111 | 50040 | 
| 30 | **4 sympy/core/numbers.py** | 1856 | 1886| 197 | 9308 | 50040 | 
| 31 | **4 sympy/core/numbers.py** | 1987 | 2016| 172 | 9480 | 50040 | 
| 32 | 5 sympy/assumptions/handlers/sets.py | 97 | 160| 419 | 9899 | 54750 | 
| 33 | **5 sympy/core/numbers.py** | 2018 | 2030| 126 | 10025 | 54750 | 
| 34 | 6 sympy/core/benchmarks/bench_numbers.py | 1 | 94| 367 | 10392 | 55118 | 
| 35 | 7 sympy/core/basic.py | 572 | 616| 355 | 10747 | 69640 | 
| 36 | 8 sympy/core/expr.py | 2381 | 2449| 517 | 11264 | 98093 | 
| 37 | **8 sympy/core/numbers.py** | 1816 | 1829| 143 | 11407 | 98093 | 
| 38 | 8 sympy/core/expr.py | 2451 | 2511| 448 | 11855 | 98093 | 
| 39 | **8 sympy/core/numbers.py** | 2430 | 2485| 258 | 12113 | 98093 | 
| 40 | 8 sympy/core/power.py | 1143 | 1161| 133 | 12246 | 98093 | 
| 41 | 8 sympy/polys/domains/pythonrational.py | 156 | 179| 234 | 12480 | 98093 | 
| 42 | 9 sympy/assumptions/ask.py | 272 | 297| 133 | 12613 | 108877 | 
| 43 | **9 sympy/core/numbers.py** | 1212 | 1242| 337 | 12950 | 108877 | 
| 44 | 9 sympy/assumptions/ask.py | 299 | 326| 165 | 13115 | 108877 | 
| 45 | 9 sympy/polys/domains/pythonrational.py | 181 | 206| 263 | 13378 | 108877 | 
| 46 | 10 sympy/simplify/simplify.py | 1313 | 1373| 554 | 13932 | 121051 | 
| 47 | **10 sympy/core/numbers.py** | 507 | 526| 155 | 14087 | 121051 | 
| 48 | **10 sympy/core/numbers.py** | 1179 | 1201| 283 | 14370 | 121051 | 
| 49 | 10 sympy/polys/domains/pythonrational.py | 126 | 154| 277 | 14647 | 121051 | 
| 50 | 10 sympy/polys/domains/pythonrational.py | 105 | 124| 223 | 14870 | 121051 | 
| 51 | **10 sympy/core/numbers.py** | 3827 | 3877| 307 | 15177 | 121051 | 
| 52 | 10 sympy/simplify/simplify.py | 1178 | 1259| 779 | 15956 | 121051 | 
| **-> 53 <-** | **10 sympy/core/numbers.py** | 1283 | 1298| 149 | 16105 | 121051 | 
| 54 | **10 sympy/core/numbers.py** | 3381 | 3424| 336 | 16441 | 121051 | 
| 55 | **10 sympy/core/numbers.py** | 3360 | 3379| 172 | 16613 | 121051 | 
| **-> 56 <-** | **10 sympy/core/numbers.py** | 1300 | 1312| 133 | 16746 | 121051 | 
| 57 | **10 sympy/core/numbers.py** | 2625 | 2651| 134 | 16880 | 121051 | 
| 58 | **10 sympy/core/numbers.py** | 3049 | 3120| 498 | 17378 | 121051 | 
| 59 | **10 sympy/core/numbers.py** | 574 | 652| 604 | 17982 | 121051 | 
| 60 | 11 sympy/solvers/solveset.py | 495 | 527| 288 | 18270 | 141176 | 
| 61 | 11 sympy/assumptions/handlers/sets.py | 232 | 285| 560 | 18830 | 141176 | 
| 62 | 12 sympy/solvers/inequalities.py | 131 | 195| 412 | 19242 | 148332 | 
| **-> 63 <-** | **12 sympy/core/numbers.py** | 1314 | 1326| 136 | 19378 | 148332 | 
| 64 | 13 sympy/functions/elementary/exponential.py | 159 | 186| 241 | 19619 | 155039 | 
| 65 | 13 sympy/solvers/solveset.py | 388 | 403| 154 | 19773 | 155039 | 
| 66 | 14 sympy/core/mul.py | 1030 | 1132| 842 | 20615 | 169328 | 
| **-> 67 <-** | **14 sympy/core/numbers.py** | 1328 | 1360| 255 | 20870 | 169328 | 
| 68 | **14 sympy/core/numbers.py** | 1 | 35| 302 | 21172 | 169328 | 
| 69 | 15 sympy/polys/numberfields.py | 1078 | 1113| 221 | 21393 | 178259 | 
| 70 | **15 sympy/core/numbers.py** | 107 | 131| 184 | 21577 | 178259 | 
| 71 | **15 sympy/core/numbers.py** | 2828 | 2899| 494 | 22071 | 178259 | 
| 72 | **15 sympy/core/numbers.py** | 3791 | 3825| 303 | 22374 | 178259 | 
| 73 | **15 sympy/core/numbers.py** | 1089 | 1177| 752 | 23126 | 178259 | 
| 74 | 16 sympy/parsing/sympy_parser.py | 691 | 707| 131 | 23257 | 185579 | 
| 75 | 16 sympy/functions/elementary/exponential.py | 674 | 711| 301 | 23558 | 185579 | 
| **-> 76 <-** | **16 sympy/core/numbers.py** | 1244 | 1281| 342 | 23900 | 185579 | 


### Hint

```
Some experimentation shows that the behavior can be improved by computing the difference and then looking at the sign with .is_positive and .is_negative. However, that can break as well, you just need a bigger fraction:
\`\`\`
>>> r = sympy.Rational('472202503979844695356573871761845338575143343779448489867569357017941709222155070092152068445390137810467671349/150306725297525326584926758194517569752043683130132471725266622178061377607334940381676735896625196994043838464')
>>> r < sympy.pi
True
>>> r > sympy.pi
True
>>> x = r - sympy.pi
>>> repr(x.is_positive)
'None'
>>> repr(x.is_negative)
'None'
\`\`\`
This is the same as that issue. See the discussion there and at https://github.com/sympy/sympy/pull/12537. 
Ah, awesome.

Is there a workaround to force evalf() to use enough precision that the digits it gives me are correct? I don't mind if it uses a lot of memory, or loops forever, or tells me it can't determine the answer. I just want to avoid the case where I get an answer that's wrong.

Something like the behavior of Hans Boehm's [constructive reals](http://www.hboehm.info/new_crcalc/CRCalc.html) would be ideal.
evalf already does this. It's just the way it's being used in the comparisons is wrong. But if you do `(a - b).evalf()` the sign of the result should tell if you if a < b or a > b (or you'll get something like `-0.e-124` if it can't determine, which I believe you can check with `is_comparable). 
To clarify, evalf internally increases the precision to try to give you as many digits as you asked for correctly (the default is 15 digits). For comparisions, you can just use `evalf(2)`, which is what the [core does](https://github.com/sympy/sympy/blob/d1320814eda6549996190618a21eaf212cfd4d1e/sympy/core/expr.py#L3371). 

An example of where it can fail

\`\`\`
>>> (sin(1)**2 + cos(1)**2 - 1).evalf()
-0.e-124
>>> (sin(1)**2 + cos(1)**2 - 1).evalf().is_comparable
False
\`\`\`

Because `sin(1)**2 + cos(1)**2` and `1` are actually equal, it will never be able to compute enough digits of each to get something that it knows is greater than or less than 0 (you can also construct more diabolic examples where expressions aren't equal, but evalf won't try enough digits to determine that). 
Perfect.

So it seems like I can do the following to compare (rational or irrational reals) a and b:
1) check if a == b; if so, we're done
2) subtract, and keep calling evalf(2) with increasing arguments to maxn, until the result .is_comparable
3) if we find a comparable result, then return the appropriate ordering
4) otherwise give up at some point and claim the results are incomparable

Hmm, is there any benefit in actually checking a == b? Or will the subtraction always give exactly zero in that case?

Alternatively, is there any danger that a == b will be True even though the a and b aren't exactly equal?
For floats in the present code, yes (see https://github.com/sympy/sympy/issues/11707).  Once that is fixed no. 
```

## Patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1259,13 +1259,13 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             if other.is_irrational:
                 return False
             return other.__eq__(self)
-        if isinstance(other, Float):
+        if other.is_Float:
             return bool(mlib.mpf_eq(self._mpf_, other._mpf_))
-        if isinstance(other, Number):
+        if other.is_Number:
             # numbers should compare at the same precision;
             # all _as_mpf_val routines should be sure to abide
             # by the request to change the prec if necessary; if
@@ -1283,11 +1283,14 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__lt__(self)
-        if other.is_comparable:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_gt(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__gt__(self, other)
@@ -1297,11 +1300,14 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__le__(self)
-        if other.is_comparable:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_ge(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__ge__(self, other)
@@ -1311,11 +1317,14 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__gt__(self)
-        if other.is_real and other.is_number:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_lt(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__lt__(self, other)
@@ -1325,11 +1334,14 @@ def __le__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__ge__(self)
-        if other.is_real and other.is_number:
+        if other.is_Rational and not other.is_Integer:
+            self *= other.q
+            other = _sympify(other.p)
+        elif other.is_comparable:
             other = other.evalf()
-        if isinstance(other, Number) and other is not S.NaN:
+        if other.is_Number and other is not S.NaN:
             return _sympify(bool(
                 mlib.mpf_le(self._mpf_, other._as_mpf_val(self._prec))))
         return Expr.__le__(self, other)
@@ -1720,16 +1732,16 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             if other.is_irrational:
                 return False
             return other.__eq__(self)
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 # a Rational is always in reduced form so will never be 2/4
                 # so we can just check equivalence of args
                 return self.p == other.p and self.q == other.q
-            if isinstance(other, Float):
+            if other.is_Float:
                 return mlib.mpf_eq(self._as_mpf_val(other._prec), other._mpf_)
         return False
 
@@ -1741,13 +1753,13 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__lt__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q > self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_gt(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1759,13 +1771,13 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__le__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                  return _sympify(bool(self.p*other.q >= self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_ge(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1777,13 +1789,13 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__gt__(self)
         expr = self
-        if isinstance(other, Number):
-            if isinstance(other, Rational):
+        if other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q < self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_lt(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -1796,12 +1808,12 @@ def __le__(self, other):
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
         expr = self
-        if isinstance(other, NumberSymbol):
+        if other.is_NumberSymbol:
             return other.__ge__(self)
-        elif isinstance(other, Number):
-            if isinstance(other, Rational):
+        elif other.is_Number:
+            if other.is_Rational:
                 return _sympify(bool(self.p*other.q <= self.q*other.p))
-            if isinstance(other, Float):
+            if other.is_Float:
                 return _sympify(bool(mlib.mpf_le(
                     self._as_mpf_val(other._prec), other._mpf_)))
         elif other.is_number and other.is_real:
@@ -2119,7 +2131,7 @@ def __gt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s > %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p > other.p)
         return Rational.__gt__(self, other)
 
@@ -2128,7 +2140,7 @@ def __lt__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p < other.p)
         return Rational.__lt__(self, other)
 
@@ -2137,7 +2149,7 @@ def __ge__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p >= other.p)
         return Rational.__ge__(self, other)
 
@@ -2146,7 +2158,7 @@ def __le__(self, other):
             other = _sympify(other)
         except SympifyError:
             raise TypeError("Invalid comparison %s <= %s" % (self, other))
-        if isinstance(other, Integer):
+        if other.is_Integer:
             return _sympify(self.p <= other.p)
         return Rational.__le__(self, other)
 
@@ -3344,7 +3356,7 @@ def __eq__(self, other):
             return NotImplemented
         if self is other:
             return True
-        if isinstance(other, Number) and self.is_irrational:
+        if other.is_Number and self.is_irrational:
             return False
 
         return False    # NumberSymbol != non-(Number|self)
@@ -3352,61 +3364,15 @@ def __eq__(self, other):
     def __ne__(self, other):
         return not self == other
 
-    def __lt__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s < %s" % (self, other))
-        if self is other:
-            return S.false
-        if isinstance(other, Number):
-            approx = self.approximation_interval(other.__class__)
-            if approx is not None:
-                l, u = approx
-                if other < l:
-                    return S.false
-                if other > u:
-                    return S.true
-            return _sympify(self.evalf() < other)
-        if other.is_real and other.is_number:
-            other = other.evalf()
-            return _sympify(self.evalf() < other)
-        return Expr.__lt__(self, other)
-
     def __le__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s <= %s" % (self, other))
         if self is other:
             return S.true
-        if other.is_real and other.is_number:
-            other = other.evalf()
-        if isinstance(other, Number):
-            return _sympify(self.evalf() <= other)
         return Expr.__le__(self, other)
 
-    def __gt__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s > %s" % (self, other))
-        r = _sympify((-self) < (-other))
-        if r in (S.true, S.false):
-            return r
-        else:
-            return Expr.__gt__(self, other)
-
     def __ge__(self, other):
-        try:
-            other = _sympify(other)
-        except SympifyError:
-            raise TypeError("Invalid comparison %s >= %s" % (self, other))
-        r = _sympify((-self) <= (-other))
-        if r in (S.true, S.false):
-            return r
-        else:
-            return Expr.__ge__(self, other)
+        if self is other:
+            return S.true
+        return Expr.__ge__(self, other)
 
     def __int__(self):
         # subclass with appropriate return value

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_relational.py b/sympy/core/tests/test_relational.py
--- a/sympy/core/tests/test_relational.py
+++ b/sympy/core/tests/test_relational.py
@@ -718,6 +718,55 @@ def test_issue_10927():
     assert str(Eq(x, -oo)) == 'Eq(x, -oo)'
 
 
+def test_issues_13081_12583_12534():
+    # 13081
+    r = Rational('905502432259640373/288230376151711744')
+    assert (r < pi) is S.false
+    assert (r > pi) is S.true
+    # 12583
+    v = sqrt(2)
+    u = sqrt(v) + 2/sqrt(10 - 8/sqrt(2 - v) + 4*v*(1/sqrt(2 - v) - 1))
+    assert (u >= 0) is S.true
+    # 12534; Rational vs NumberSymbol
+    # here are some precisions for which Rational forms
+    # at a lower and higher precision bracket the value of pi
+    # e.g. for p = 20:
+    # Rational(pi.n(p + 1)).n(25) = 3.14159265358979323846 2834
+    #                    pi.n(25) = 3.14159265358979323846 2643
+    # Rational(pi.n(p    )).n(25) = 3.14159265358979323846 1987
+    assert [p for p in range(20, 50) if
+            (Rational(pi.n(p)) < pi) and
+            (pi < Rational(pi.n(p + 1)))
+        ] == [20, 24, 27, 33, 37, 43, 48]
+    # pick one such precision and affirm that the reversed operation
+    # gives the opposite result, i.e. if x < y is true then x > y
+    # must be false
+    p = 20
+    # Rational vs NumberSymbol
+    G = [Rational(pi.n(i)) > pi for i in (p, p + 1)]
+    L = [Rational(pi.n(i)) < pi for i in (p, p + 1)]
+    assert G == [False, True]
+    assert all(i is not j for i, j in zip(L, G))
+    # Float vs NumberSymbol
+    G = [pi.n(i) > pi for i in (p, p + 1)]
+    L = [pi.n(i) < pi for i in (p, p + 1)]
+    assert G == [False, True]
+    assert all(i is not j for i, j in zip(L, G))
+    # Float vs Float
+    G = [pi.n(p) > pi.n(p + 1)]
+    L = [pi.n(p) < pi.n(p + 1)]
+    assert G == [True]
+    assert all(i is not j for i, j in zip(L, G))
+    # Float vs Rational
+    # the rational form is less than the floating representation
+    # at the same precision
+    assert [i for i in range(15, 50) if Rational(pi.n(i)) > pi.n(i)
+        ] == []
+    # this should be the same if we reverse the relational
+    assert [i for i in range(15, 50) if pi.n(i) < Rational(pi.n(i))
+        ] == []
+
+
 def test_binary_symbols():
     ans = set([x])
     for f in Eq, Ne:

```


## Code snippets

### 1 - sympy/core/numbers.py:

Start line: 1762, End line: 1778

```python
class Rational(Number):

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        expr = self
        if isinstance(other, Number):
            if isinstance(other, Rational):
                 return _sympify(bool(self.p*other.q >= self.q*other.p))
            if isinstance(other, Float):
                return _sympify(bool(mlib.mpf_ge(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__ge__(expr, other)
```
### 2 - sympy/core/numbers.py:

Start line: 1744, End line: 1760

```python
class Rational(Number):

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        expr = self
        if isinstance(other, Number):
            if isinstance(other, Rational):
                return _sympify(bool(self.p*other.q > self.q*other.p))
            if isinstance(other, Float):
                return _sympify(bool(mlib.mpf_gt(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__gt__(expr, other)
```
### 3 - sympy/core/numbers.py:

Start line: 1780, End line: 1796

```python
class Rational(Number):

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        expr = self
        if isinstance(other, Number):
            if isinstance(other, Rational):
                return _sympify(bool(self.p*other.q < self.q*other.p))
            if isinstance(other, Float):
                return _sympify(bool(mlib.mpf_lt(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__lt__(expr, other)
```
### 4 - sympy/core/numbers.py:

Start line: 1798, End line: 1814

```python
class Rational(Number):

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        expr = self
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        elif isinstance(other, Number):
            if isinstance(other, Rational):
                return _sympify(bool(self.p*other.q <= self.q*other.p))
            if isinstance(other, Float):
                return _sympify(bool(mlib.mpf_le(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__le__(expr, other)
```
### 5 - sympy/core/numbers.py:

Start line: 1723, End line: 1742

```python
class Rational(Number):

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if isinstance(other, NumberSymbol):
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if isinstance(other, Number):
            if isinstance(other, Rational):
                # a Rational is always in reduced form so will never be 2/4
                # so we can just check equivalence of args
                return self.p == other.p and self.q == other.q
            if isinstance(other, Float):
                return mlib.mpf_eq(self._as_mpf_val(other._prec), other._mpf_)
        return False

    def __ne__(self, other):
        return not self == other
```
### 6 - sympy/core/numbers.py:

Start line: 1625, End line: 1637

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rdiv__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return other*(1/self)
            else:
                return Number.__rdiv__(self, other)
        return Number.__rdiv__(self, other)
    __truediv__ = __div__
```
### 7 - sympy/core/numbers.py:

Start line: 1658, End line: 1698

```python
class Rational(Number):

    def _eval_power(self, expt):
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return self._eval_evalf(expt._prec)**expt
            if expt.is_negative:
                # (3/4)**-2 -> (4/3)**2
                ne = -expt
                if (ne is S.One):
                    return Rational(self.q, self.p)
                if self.is_negative:
                    if expt.q != 1:
                        return -(S.NegativeOne)**((expt.p % expt.q) /
                               S(expt.q))*Rational(self.q, -self.p)**ne
                    else:
                        return S.NegativeOne**ne*Rational(self.q, -self.p)**ne
                else:
                    return Rational(self.q, self.p)**ne
            if expt is S.Infinity:  # -oo already caught by test for negative
                if self.p > self.q:
                    # (3/2)**oo -> oo
                    return S.Infinity
                if self.p < -self.q:
                    # (-3/2)**oo -> oo + I*oo
                    return S.Infinity + S.Infinity*S.ImaginaryUnit
                return S.Zero
            if isinstance(expt, Integer):
                # (4/3)**2 -> 4**2 / 3**2
                return Rational(self.p**expt.p, self.q**expt.p, 1)
            if isinstance(expt, Rational):
                if self.p != 1:
                    # (4/3)**(5/6) -> 4**(5/6)*3**(-5/6)
                    return Integer(self.p)**expt*Integer(self.q)**(-expt)
                # as the above caught negative self.p, now self is positive
                return Integer(self.q)**Rational(
                expt.p*(expt.q - 1), expt.q) / \
                    Integer(self.q)**Integer(expt.p)

        if self.is_negative and expt.is_even:
            return (-self)**expt

        return
```
### 8 - sympy/core/numbers.py:

Start line: 1610, End line: 1624

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                if self.p and other.p == S.Zero:
                    return S.ComplexInfinity
                else:
                    return Rational(self.p, self.q*other.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational(self.p*other.q, self.q*other.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return self*(1/other)
            else:
                return Number.__div__(self, other)
        return Number.__div__(self, other)
```
### 9 - sympy/core/numbers.py:

Start line: 1700, End line: 1721

```python
class Rational(Number):

    def _as_mpf_val(self, prec):
        return mlib.from_rational(self.p, self.q, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(mlib.from_rational(self.p, self.q, prec, rnd))

    def __abs__(self):
        return Rational(abs(self.p), self.q)

    def __int__(self):
        p, q = self.p, self.q
        if p < 0:
            return -int(-p//q)
        return int(p//q)

    __long__ = __int__

    def floor(self):
        return Integer(self.p // self.q)

    def ceiling(self):
        return -Integer(-self.p // self.q)
```
### 10 - sympy/core/numbers.py:

Start line: 1584, End line: 1595

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.q*other.p - self.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.q*other.p - self.p*other.q, self.q*other.q)
            elif isinstance(other, Float):
                return -self + other
            else:
                return Number.__rsub__(self, other)
        return Number.__rsub__(self, other)
```
### 11 - sympy/core/numbers.py:

Start line: 2122, End line: 2167

```python
class Integer(Rational):

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if isinstance(other, Integer):
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if isinstance(other, Integer):
            return _sympify(self.p < other.p)
        return Rational.__lt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if isinstance(other, Integer):
            return _sympify(self.p >= other.p)
        return Rational.__ge__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if isinstance(other, Integer):
            return _sympify(self.p <= other.p)
        return Rational.__le__(self, other)

    def __hash__(self):
        return hash(self.p)

    def __index__(self):
        return self.p

    ########################################

    def _eval_is_odd(self):
        return bool(self.p % 2)
```
### 12 - sympy/core/numbers.py:

Start line: 1639, End line: 1656

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Rational):
                n = (self.p*other.q) // (other.p*self.q)
                return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
            if isinstance(other, Float):
                # calculate mod with Rationals, *then* round the answer
                return Float(self.__mod__(Rational(other)),
                             precision=other._prec)
            return Number.__mod__(self, other)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Rational):
            return Rational.__mod__(other, self)
        return Number.__rmod__(self, other)
```
### 13 - sympy/core/numbers.py:

Start line: 1570, End line: 1583

```python
class Rational(Number):
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p - self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return -other + self
            else:
                return Number.__sub__(self, other)
        return Number.__sub__(self, other)
```
### 14 - sympy/core/numbers.py:

Start line: 1557, End line: 1569

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p + self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                #TODO: this can probably be optimized more
                return Rational(self.p*other.q + self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return other + self
            else:
                return Number.__add__(self, other)
        return Number.__add__(self, other)
```
### 15 - sympy/core/numbers.py:

Start line: 1596, End line: 1608

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p*other.p, self.q, igcd(other.p, self.q))
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, self.q*other.q, igcd(self.p, other.q)*igcd(self.q, other.p))
            elif isinstance(other, Float):
                return other*self
            else:
                return Number.__mul__(self, other)
        return Number.__mul__(self, other)
    __rmul__ = __mul__
```
### 16 - sympy/core/numbers.py:

Start line: 1363, End line: 1455

```python
class Rational(Number):
    """Represents integers and rational numbers (p/q) of any size.

    Examples
    ========

    >>> from sympy import Rational, nsimplify, S, pi
    >>> Rational(3)
    3
    >>> Rational(1, 2)
    1/2

    Rational is unprejudiced in accepting input. If a float is passed, the
    underlying value of the binary representation will be returned:

    >>> Rational(.5)
    1/2
    >>> Rational(.2)
    3602879701896397/18014398509481984

    If the simpler representation of the float is desired then consider
    limiting the denominator to the desired value or convert the float to
    a string (which is roughly equivalent to limiting the denominator to
    10**12):

    >>> Rational(str(.2))
    1/5
    >>> Rational(.2).limit_denominator(10**12)
    1/5

    An arbitrarily precise Rational is obtained when a string literal is
    passed:

    >>> Rational("1.23")
    123/100
    >>> Rational('1e-2')
    1/100
    >>> Rational(".1")
    1/10
    >>> Rational('1e-2/3.2')
    1/320

    The conversion of other types of strings can be handled by
    the sympify() function, and conversion of floats to expressions
    or simple fractions can be handled with nsimplify:

    >>> S('.[3]')  # repeating digits in brackets
    1/3
    >>> S('3**2/10')  # general expressions
    9/10
    >>> nsimplify(.3)  # numbers that have a simple form
    3/10

    But if the input does not reduce to a literal Rational, an error will
    be raised:

    >>> Rational(pi)
    Traceback (most recent call last):
    ...
    TypeError: invalid input: pi


    Low-level
    ---------

    Access numerator and denominator as .p and .q:

    >>> r = Rational(3, 4)
    >>> r
    3/4
    >>> r.p
    3
    >>> r.q
    4

    Note that p and q return integers (not SymPy Integers) so some care
    is needed when using them in expressions:

    >>> r.p/r.q
    0.75

    See Also
    ========
    sympify, sympy.simplify.simplify.nsimplify
    """
    is_real = True
    is_integer = False
    is_rational = True
    is_number = True

    __slots__ = ['p', 'q']

    is_Rational = True
```
### 17 - sympy/core/numbers.py:

Start line: 1457, End line: 1527

```python
class Rational(Number):

    @cacheit
    def __new__(cls, p, q=None, gcd=None):
        if q is None:
            if isinstance(p, Rational):
                return p

            if isinstance(p, string_types):
                if p.count('/') > 1:
                    raise TypeError('invalid input: %s' % p)
                pq = p.rsplit('/', 1)
                if len(pq) == 2:
                    p, q = pq
                    fp = fractions.Fraction(p)
                    fq = fractions.Fraction(q)
                    f = fp/fq
                    return Rational(f.numerator, f.denominator, 1)
                p = p.replace(' ', '')
                try:
                    p = fractions.Fraction(p)
                except ValueError:
                    pass  # error will raise below

            if not isinstance(p, string_types):
                try:
                    if isinstance(p, fractions.Fraction):
                        return Rational(p.numerator, p.denominator, 1)
                except NameError:
                    pass  # error will raise below

                if isinstance(p, (float, Float)):
                    return Rational(*_as_integer_ratio(p))

            if not isinstance(p, SYMPY_INTS + (Rational,)):
                raise TypeError('invalid input: %s' % p)
            q = q or S.One
            gcd = 1
        else:
            p = Rational(p)
            q = Rational(q)

        if isinstance(q, Rational):
            p *= q.q
            q = q.p
        if isinstance(p, Rational):
            q *= p.q
            p = p.p

        # p and q are now integers
        if q == 0:
            if p == 0:
                if _errdict["divide"]:
                    raise ValueError("Indeterminate 0/0")
                else:
                    return S.NaN
            return S.ComplexInfinity
        if q < 0:
            q = -q
            p = -p
        if not gcd:
            gcd = igcd(abs(p), q)
        if gcd > 1:
            p //= gcd
            q //= gcd
        if q == 1:
            return Integer(p)
        if p == 1 and q == 2:
            return S.Half
        obj = Expr.__new__(cls)
        obj.p = p
        obj.q = q
        return obj
```
### 18 - sympy/core/numbers.py:

Start line: 1831, End line: 1854

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            if other is S.Zero:
                return other
            return Rational(
                Integer(igcd(self.p, other.p)),
                Integer(ilcm(self.q, other.q)))
        return Number.gcd(self, other)

    @_sympifyit('other', NotImplemented)
    def lcm(self, other):
        if isinstance(other, Rational):
            return Rational(
                self.p*other.p//igcd(self.p, other.p),
                igcd(self.q, other.q))
        return Number.lcm(self, other)

    def as_numer_denom(self):
        return Integer(self.p), Integer(self.q)

    def _sage_(self):
        import sage.all as sage
        return sage.Integer(self.p)/sage.Integer(self.q)
```
### 19 - sympy/core/numbers.py:

Start line: 2032, End line: 2120

```python
class Integer(Rational):

    # TODO make it decorator + bytecodehacks?
    def __add__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q + other.p, other.q, 1)
            return Rational.__add__(self, other)
        else:
            return Add(self, other)

    def __radd__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational(other.p + self.p*other.q, other.q, 1)
            return Rational.__radd__(self, other)
        return Rational.__radd__(self, other)

    def __sub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - other.p, other.q, 1)
            return Rational.__sub__(self, other)
        return Rational.__sub__(self, other)

    def __rsub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational(other.p - self.p*other.q, other.q, 1)
            return Rational.__rsub__(self, other)
        return Rational.__rsub__(self, other)

    def __mul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)

    def __rmul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other*self.p)
            elif isinstance(other, Rational):
                return Rational(other.p*self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)

    def __mod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p % other)
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            return Rational.__mod__(self, other)
        return Rational.__mod__(self, other)

    def __rmod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other % self.p)
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            return Rational.__rmod__(self, other)
        return Rational.__rmod__(self, other)

    def __eq__(self, other):
        if isinstance(other, integer_types):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)

    def __ne__(self, other):
        return not self == other
```
### 20 - sympy/core/numbers.py:

Start line: 1944, End line: 1985

```python
class Integer(Rational):

    q = 1
    is_integer = True
    is_number = True

    is_Integer = True

    __slots__ = ['p']

    def _as_mpf_val(self, prec):
        return mlib.from_int(self.p, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(self._as_mpf_val(prec))

    # TODO caching with decorator, but not to degrade performance
    @int_trace
    def __new__(cls, i):
        if isinstance(i, string_types):
            i = i.replace(' ', '')
        # whereas we cannot, in general, make a Rational from an
        # arbitrary expression, we can make an Integer unambiguously
        # (except when a non-integer expression happens to round to
        # an integer). So we proceed by taking int() of the input and
        # let the int routines determine whether the expression can
        # be made into an int or whether an error should be raised.
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                'Integer can only work with integer expressions.')
        try:
            return _intcache[ival]
        except KeyError:
            # We only work with well-behaved integer types. This converts, for
            # example, numpy.int32 instances.
            obj = Expr.__new__(cls)
            obj.p = ival

            _intcache[ival] = obj
            return obj
```
### 21 - sympy/core/numbers.py:

Start line: 134, End line: 155

```python
def _decimal_to_Rational_prec(dec):
    """Convert an ordinary decimal instance to a Rational."""
    if not dec.is_finite():
        raise TypeError("dec must be finite, got %s." % dec)
    s, d, e = dec.as_tuple()
    prec = len(d)
    if e >= 0:  # it's an integer
        rv = Integer(int(dec))
    else:
        s = (-1)**s
        d = sum([di*10**i for i, di in enumerate(reversed(d))])
        rv = Rational(s*d, 10**-e)
    return rv, prec


def _literal_float(f):
    """Return True if n can be interpreted as a floating point number."""
    pat = r"[-+]?((\d*\.\d+)|(\d+\.?))(eE[-+]?\d+)?"
    return bool(regex.match(pat, f))

# (a,b) -> gcd(a,b)
_gcdcache = {}
```
### 23 - sympy/core/numbers.py:

Start line: 2169, End line: 2247

```python
class Integer(Rational):

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational(1, self)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super(Integer, self)._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                if expt.q != 1:
                    return -(S.NegativeOne)**((expt.p % expt.q) /
                            S(expt.q))*Rational(1, -self)**ne
                else:
                    return (S.NegativeOne)**ne*Rational(1, -self)**ne
            else:
                return Rational(1, self.p)**ne
        # see if base is a perfect root, sqrt(4) --> 2
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            # if it's a perfect root we've finished
            result = Integer(x**abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne**expt
            return result

        # The following is an algorithm where we collect perfect roots
        # from the factors of base.

        # if it's not an nth root, it still might be a perfect power
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            dict = {p[0]: p[1]}
        else:
            dict = Integer(self).factors(limit=2**15)

        # now process the dict of factors
        if self.is_negative:
            dict[-1] = 1
        out_int = 1  # integer part
        out_rad = 1  # extracted radicals
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        # ... other code
```
### 25 - sympy/core/numbers.py:

Start line: 1529, End line: 1555

```python
class Rational(Number):

    def limit_denominator(self, max_denominator=1000000):
        """Closest Rational to self with denominator at most max_denominator.

        >>> from sympy import Rational
        >>> Rational('3.141592653589793').limit_denominator(10)
        22/7
        >>> Rational('3.141592653589793').limit_denominator(100)
        311/99

        """
        f = fractions.Fraction(self.p, self.q)
        return Rational(f.limit_denominator(fractions.Fraction(int(max_denominator))))

    def __getnewargs__(self):
        return (self.p, self.q)

    def _hashable_content(self):
        return (self.p, self.q)

    def _eval_is_positive(self):
        return self.p > 0

    def _eval_is_zero(self):
        return self.p == 0

    def __neg__(self):
        return Rational(-self.p, self.q)
```
### 26 - sympy/core/numbers.py:

Start line: 3498, End line: 3566

```python
E = S.Exp1


class Pi(with_metaclass(Singleton, NumberSymbol)):
    r"""The `\pi` constant.

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Pi
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = []

    def _latex(self, printer):
        return r"\pi"

    @staticmethod
    def __abs__():
        return S.Pi

    def __int__(self):
        return 3

    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71), Rational(22, 7))

    def _sage_(self):
        import sage.all as sage
        return sage.pi
pi = S.Pi
```
### 29 - sympy/core/numbers.py:

Start line: 2248, End line: 2300

```python
class Integer(Rational):

    def _eval_power(self, expt):
        # ... other code
        for prime, exponent in dict.items():
            exponent *= expt.p
            # remove multiples of expt.q: (2**12)**(1/10) -> 2*(2**2)**(1/10)
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime**div_e
            if div_m > 0:
                # see if the reduced exponent shares a gcd with e.q
                # (2**2)**(1/10) -> 2**(1/5)
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational(div_m//g, expt.q//g))
                else:
                    sqr_dict[prime] = div_m
        # identify gcd of remaining powers
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k**(v//sqr_gcd)
        if sqr_int == self and out_int == 1 and out_rad == 1:
            result = None
        else:
            result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
        return result

    def _eval_is_prime(self):
        from sympy.ntheory import isprime

        return isprime(self)

    def _eval_is_composite(self):
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            return False

    def as_numer_denom(self):
        return self, S.One

    def __floordiv__(self, other):
        return Integer(self.p // Integer(other).p)

    def __rfloordiv__(self, other):
        return Integer(Integer(other).p // self.p)

# Add sympify converters
for i_type in integer_types:
    converter[i_type] = Integer
```
### 30 - sympy/core/numbers.py:

Start line: 1856, End line: 1886

```python
class Rational(Number):

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import S
        >>> (S(-3)/2).as_content_primitive()
        (3/2, -1)

        See docstring of Expr.as_content_primitive for more examples.
        """

        if self:
            if self.is_positive:
                return self, S.One
            return -self, S.NegativeOne
        return S.One, self

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product. """
        return self, S.One

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation. """
        return self, S.Zero


# int -> Integer
_intcache = {}
```
### 31 - sympy/core/numbers.py:

Start line: 1987, End line: 2016

```python
class Integer(Rational):

    def __getnewargs__(self):
        return (self.p,)

    # Arithmetic operations are here for efficiency
    def __int__(self):
        return self.p

    __long__ = __int__

    def floor(self):
        return Integer(self.p)

    def ceiling(self):
        return Integer(self.p)

    def __neg__(self):
        return Integer(-self.p)

    def __abs__(self):
        if self.p >= 0:
            return self
        else:
            return Integer(-self.p)

    def __divmod__(self, other):
        from .containers import Tuple
        if isinstance(other, Integer) and global_evaluate[0]:
            return Tuple(*(divmod(self.p, other.p)))
        else:
            return Number.__divmod__(self, other)
```
### 33 - sympy/core/numbers.py:

Start line: 2018, End line: 2030

```python
class Integer(Rational):

    def __rdivmod__(self, other):
        from .containers import Tuple
        if isinstance(other, integer_types) and global_evaluate[0]:
            return Tuple(*(divmod(other, self.p)))
        else:
            try:
                other = Number(other)
            except TypeError:
                msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
                oname = type(other).__name__
                sname = type(self).__name__
                raise TypeError(msg % (oname, sname))
            return Number.__divmod__(other, self)
```
### 37 - sympy/core/numbers.py:

Start line: 1816, End line: 1829

```python
class Rational(Number):

    def __hash__(self):
        return super(Rational, self).__hash__()

    def factors(self, limit=None, use_trial=True, use_rho=False,
                use_pm1=False, verbose=False, visual=False):
        """A wrapper to factorint which return factors of self that are
        smaller than limit (or cheap to compute). Special methods of
        factoring are disabled by default so that only trial division is used.
        """
        from sympy.ntheory import factorrat

        return factorrat(self, limit=limit, use_trial=use_trial,
                      use_rho=use_rho, use_pm1=use_pm1,
                      verbose=verbose).copy()
```
### 39 - sympy/core/numbers.py:

Start line: 2430, End line: 2485

```python
class RationalConstant(Rational):
    """
    Abstract base class for rationals with specific behaviors

    Derived classes must define class attributes p and q and should probably all
    be singletons.
    """
    __slots__ = []

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class IntegerConstant(Integer):
    __slots__ = []

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class Zero(with_metaclass(Singleton, IntegerConstant)):
    """The number zero.

    Zero is a singleton, and can be accessed by ``S.Zero``

    Examples
    ========

    >>> from sympy import S, Integer, zoo
    >>> Integer(0) is S.Zero
    True
    >>> 1/S.Zero
    zoo

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Zero
    """

    p = 0
    q = 1
    is_positive = False
    is_negative = False
    is_zero = True
    is_number = True

    __slots__ = []

    @staticmethod
    def __abs__():
        return S.Zero

    @staticmethod
    def __neg__():
        return S.Zero
```
### 43 - sympy/core/numbers.py:

Start line: 1212, End line: 1242

```python
class Float(Number):

    def _eval_power(self, expt):
        """
        expt is symbolic object but not equal to 0, 1

        (-p)**r -> exp(r*log(-p)) -> exp(r*(log(p) + I*Pi)) ->
                  -> p**r*(sin(Pi*r) + cos(Pi*r)*I)
        """
        if self == 0:
            if expt.is_positive:
                return S.Zero
            if expt.is_negative:
                return Float('inf')
        if isinstance(expt, Number):
            if isinstance(expt, Integer):
                prec = self._prec
                return Float._new(
                    mlib.mpf_pow_int(self._mpf_, expt.p, prec, rnd), prec)
            elif isinstance(expt, Rational) and \
                    expt.p == 1 and expt.q % 2 and self.is_negative:
                return Pow(S.NegativeOne, expt, evaluate=False)*(
                    -self)._eval_power(expt)
            expt, prec = expt._as_mpf_op(self._prec)
            mpfself = self._mpf_
            try:
                y = mpf_pow(mpfself, expt, prec, rnd)
                return Float._new(y, prec)
            except mlib.ComplexResult:
                re, im = mlib.mpc_pow(
                    (mpfself, _mpf_zero), (expt, _mpf_zero), prec, rnd)
                return Float._new(re, prec) + \
                    Float._new(im, prec)*S.ImaginaryUnit
```
### 47 - sympy/core/numbers.py:

Start line: 507, End line: 526

```python
class Number(AtomicExpr):
    """
    Represents any kind of number in sympy.

    Floating point numbers are represented by the Float class.
    Integer numbers (of any size), together with rational numbers (again,
    there is no limit on their size) are represented by the Rational class.

    If you want to represent, for example, ``1+sqrt(2)``, then you need to do::

      Rational(1) + sqrt(Rational(2))
    """
    is_commutative = True
    is_number = True
    is_Number = True

    __slots__ = []

    # Used to make max(x._prec, y._prec) return x._prec when only x is a float
    _prec = -1
```
### 48 - sympy/core/numbers.py:

Start line: 1179, End line: 1201

```python
class Float(Number):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number) and other != 0 and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__div__(self, other)

    __truediv__ = __div__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if isinstance(other, Rational) and other.q != 1 and global_evaluate[0]:
            # calculate mod with Rationals, *then* round the result
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        if isinstance(other, Float) and global_evaluate[0]:
            r = self/other
            if r == int(r):
                return Float(0, precision=max(self._prec, other._prec))
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mod__(self, other)
```
### 51 - sympy/core/numbers.py:

Start line: 3827, End line: 3877

```python
I = S.ImaginaryUnit


def sympify_fractions(f):
    return Rational(f.numerator, f.denominator)

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


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

converter[complex] = sympify_complex

_intcache[0] = S.Zero
_intcache[1] = S.One
_intcache[-1] = S.NegativeOne

from .power import Pow, integer_nthroot
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()
```
### 53 - sympy/core/numbers.py:

Start line: 1283, End line: 1298

```python
class Float(Number):

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        if other.is_comparable:
            other = other.evalf()
        if isinstance(other, Number) and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_gt(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__gt__(self, other)
```
### 54 - sympy/core/numbers.py:

Start line: 3381, End line: 3424

```python
class NumberSymbol(AtomicExpr):

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if self is other:
            return S.true
        if other.is_real and other.is_number:
            other = other.evalf()
        if isinstance(other, Number):
            return _sympify(self.evalf() <= other)
        return Expr.__le__(self, other)

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        r = _sympify((-self) < (-other))
        if r in (S.true, S.false):
            return r
        else:
            return Expr.__gt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        r = _sympify((-self) <= (-other))
        if r in (S.true, S.false):
            return r
        else:
            return Expr.__ge__(self, other)

    def __int__(self):
        # subclass with appropriate return value
        raise NotImplementedError

    def __long__(self):
        return self.__int__()

    def __hash__(self):
        return super(NumberSymbol, self).__hash__()
```
### 55 - sympy/core/numbers.py:

Start line: 3360, End line: 3379

```python
class NumberSymbol(AtomicExpr):

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if self is other:
            return S.false
        if isinstance(other, Number):
            approx = self.approximation_interval(other.__class__)
            if approx is not None:
                l, u = approx
                if other < l:
                    return S.false
                if other > u:
                    return S.true
            return _sympify(self.evalf() < other)
        if other.is_real and other.is_number:
            other = other.evalf()
            return _sympify(self.evalf() < other)
        return Expr.__lt__(self, other)
```
### 56 - sympy/core/numbers.py:

Start line: 1300, End line: 1312

```python
class Float(Number):

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        if other.is_comparable:
            other = other.evalf()
        if isinstance(other, Number) and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_ge(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__ge__(self, other)
```
### 57 - sympy/core/numbers.py:

Start line: 2625, End line: 2651

```python
class Half(with_metaclass(Singleton, RationalConstant)):
    """The rational number 1/2.

    Half is a singleton, and can be accessed by ``S.Half``.

    Examples
    ========

    >>> from sympy import S, Rational
    >>> Rational(1, 2) is S.Half
    True

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/One_half
    """
    is_number = True

    p = 1
    q = 2

    __slots__ = []

    @staticmethod
    def __abs__():
        return S.Half
```
### 58 - sympy/core/numbers.py:

Start line: 3049, End line: 3120

```python
class NegativeInfinity(with_metaclass(Singleton, Number)):

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def _sage_(self):
        import sage.all as sage
        return -(sage.oo)

    def __hash__(self):
        return super(NegativeInfinity, self).__hash__()

    def __eq__(self, other):
        return other is S.NegativeInfinity

    def __ne__(self, other):
        return other is not S.NegativeInfinity

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.Infinity:
                return S.true
            elif other.is_nonnegative:
                return S.true
            elif other.is_infinite and other.is_negative:
                return S.false
        return Expr.__lt__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_real:
            return S.true
        return Expr.__le__(self, other)

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_real:
            return S.false
        return Expr.__gt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.Infinity:
                return S.false
            elif other.is_nonnegative:
                return S.false
            elif other.is_infinite and other.is_negative:
                return S.true
        return Expr.__ge__(self, other)

    def __mod__(self, other):
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self
```
### 59 - sympy/core/numbers.py:

Start line: 574, End line: 652

```python
class Number(AtomicExpr):

    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
            raise TypeError(msg % (type(other).__name__, type(self).__name__))
        return divmod(other, self)

    def __round__(self, *args):
        return round(float(self), *args)

    def _as_mpf_val(self, prec):
        """Evaluation of mpf tuple accurate to at least prec bits."""
        raise NotImplementedError('%s needs ._as_mpf_val() method' %
            (self.__class__.__name__))

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def _as_mpf_op(self, prec):
        prec = max(prec, self._prec)
        return self._as_mpf_val(prec), prec

    def __float__(self):
        return mlib.to_float(self._as_mpf_val(53))

    def floor(self):
        raise NotImplementedError('%s needs .floor() method' %
            (self.__class__.__name__))

    def ceiling(self):
        raise NotImplementedError('%s needs .ceiling() method' %
            (self.__class__.__name__))

    def _eval_conjugate(self):
        return self

    def _eval_order(self, *symbols):
        from sympy import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # there is no other possibility

    def _eval_is_finite(self):
        return True

    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.Infinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        return AtomicExpr.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        return AtomicExpr.__sub__(self, other)
```
### 63 - sympy/core/numbers.py:

Start line: 1314, End line: 1326

```python
class Float(Number):

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        if other.is_real and other.is_number:
            other = other.evalf()
        if isinstance(other, Number) and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_lt(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__lt__(self, other)
```
### 67 - sympy/core/numbers.py:

Start line: 1328, End line: 1360

```python
class Float(Number):

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        if other.is_real and other.is_number:
            other = other.evalf()
        if isinstance(other, Number) and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_le(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__le__(self, other)

    def __hash__(self):
        return super(Float, self).__hash__()

    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    def _sage_(self):
        import sage.all as sage
        return sage.RealNumber(str(self))

    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)


# Add sympify converters
converter[float] = converter[decimal.Decimal] = Float

# this is here to work nicely in Sage
RealNumber = Float
```
### 68 - sympy/core/numbers.py:

Start line: 1, End line: 35

```python
from __future__ import print_function, division

import decimal
import fractions
import math
import warnings
import re as regex
from collections import defaultdict

from .containers import Tuple
from .sympify import converter, sympify, _sympify, SympifyError, _convert_numpy_types
from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .decorators import _sympifyit
from .cache import cacheit, clear_cache
from .logic import fuzzy_not
from sympy.core.compatibility import (
    as_int, integer_types, long, string_types, with_metaclass, HAS_GMPY,
    SYMPY_INTS, int_info)
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero as _mpf_zero, _normalize as mpf_normalize,
    prec_to_dps)
from sympy.utilities.misc import debug, filldedent
from .evaluate import global_evaluate

from sympy.utilities.exceptions import SymPyDeprecationWarning

rnd = mlib.round_nearest

_LOG2 = math.log(2)
```
### 70 - sympy/core/numbers.py:

Start line: 107, End line: 131

```python
# TODO: we should use the warnings module
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should sympy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, bc = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)
```
### 71 - sympy/core/numbers.py:

Start line: 2828, End line: 2899

```python
class Infinity(with_metaclass(Singleton, Number)):

    def _as_mpf_val(self, prec):
        return mlib.finf

    def _sage_(self):
        import sage.all as sage
        return sage.oo

    def __hash__(self):
        return super(Infinity, self).__hash__()

    def __eq__(self, other):
        return other is S.Infinity

    def __ne__(self, other):
        return other is not S.Infinity

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_real:
            return S.false
        return Expr.__lt__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.NegativeInfinity:
                return S.false
            elif other.is_nonpositive:
                return S.false
            elif other.is_infinite and other.is_positive:
                return S.true
        return Expr.__le__(self, other)

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.NegativeInfinity:
                return S.true
            elif other.is_nonpositive:
                return S.true
            elif other.is_infinite and other.is_positive:
                return S.false
        return Expr.__gt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_real:
            return S.true
        return Expr.__ge__(self, other)

    def __mod__(self, other):
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self
```
### 72 - sympy/core/numbers.py:

Start line: 3791, End line: 3825

```python
class ImaginaryUnit(with_metaclass(Singleton, AtomicExpr)):

    def _eval_power(self, expt):
        """
        b is I = sqrt(-1)
        e is symbolic object but not equal to 0, 1

        I**r -> (-1)**(r/2) -> exp(r/2*Pi*I) -> sin(Pi*r/2) + cos(Pi*r/2)*I, r is decimal
        I**0 mod 4 -> 1
        I**1 mod 4 -> I
        I**2 mod 4 -> -1
        I**3 mod 4 -> -I
        """

        if isinstance(expt, Number):
            if isinstance(expt, Integer):
                expt = expt.p % 4
                if expt == 0:
                    return S.One
                if expt == 1:
                    return S.ImaginaryUnit
                if expt == 2:
                    return -S.One
                return -S.ImaginaryUnit
            return (S.NegativeOne)**(expt*S.Half)
        return

    def as_base_exp(self):
        return S.NegativeOne, S.Half

    def _sage_(self):
        import sage.all as sage
        return sage.I

    @property
    def _mpc_(self):
        return (Float(0)._mpf_, Float(1)._mpf_)
```
### 73 - sympy/core/numbers.py:

Start line: 1089, End line: 1177

```python
class Float(Number):

    # mpz can't be pickled
    def __getnewargs__(self):
        return (mlib.to_pickable(self._mpf_),)

    def __getstate__(self):
        return {'_prec': self._prec}

    def _hashable_content(self):
        return (self._mpf_, self._prec)

    def floor(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    @property
    def num(self):
        return mpmath.mpf(self._mpf_)

    def _as_mpf_val(self, prec):
        rv = mpf_norm(self._mpf_, prec)
        if rv != self._mpf_ and self._prec == prec:
            debug(self._mpf_, rv)
        return rv

    def _as_mpf_op(self, prec):
        return self._mpf_, max(prec, self._prec)

    def _eval_is_finite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return False
        return True

    def _eval_is_infinite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return True
        return False

    def _eval_is_integer(self):
        return self._mpf_ == _mpf_zero

    def _eval_is_negative(self):
        if self._mpf_ == _mpf_ninf:
            return True
        if self._mpf_ == _mpf_inf:
            return False
        return self.num < 0

    def _eval_is_positive(self):
        if self._mpf_ == _mpf_inf:
            return True
        if self._mpf_ == _mpf_ninf:
            return False
        return self.num > 0

    def _eval_is_zero(self):
        return self._mpf_ == _mpf_zero

    def __nonzero__(self):
        return self._mpf_ != _mpf_zero

    __bool__ = __nonzero__

    def __neg__(self):
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        return Number.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mul__(self, other)
```
### 76 - sympy/core/numbers.py:

Start line: 1244, End line: 1281

```python
class Float(Number):

    def __abs__(self):
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        if self._mpf_ == _mpf_zero:
            return 0
        return int(mlib.to_int(self._mpf_))  # uses round_fast = round_down

    __long__ = __int__

    def __eq__(self, other):
        if isinstance(other, float):
            # coerce to Float at same precision
            o = Float(other)
            try:
                ompf = o._as_mpf_val(self._prec)
            except ValueError:
                return False
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if isinstance(other, NumberSymbol):
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if isinstance(other, Float):
            return bool(mlib.mpf_eq(self._mpf_, other._mpf_))
        if isinstance(other, Number):
            # numbers should compare at the same precision;
            # all _as_mpf_val routines should be sure to abide
            # by the request to change the prec if necessary; if
            # they don't, the equality test will fail since it compares
            # the mpf tuples
            ompf = other._as_mpf_val(self._prec)
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        return False    # Float != non-Number
```
