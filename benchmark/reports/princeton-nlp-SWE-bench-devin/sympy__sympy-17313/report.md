# sympy__sympy-17313

| **sympy/sympy** | `a4297a11fd8f3e8af17efda85e3047e32e470a70` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2668 |
| **Any found context length** | 582 |
| **Avg pos** | 14.0 |
| **Min pos** | 1 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/functions/elementary/integers.py b/sympy/functions/elementary/integers.py
--- a/sympy/functions/elementary/integers.py
+++ b/sympy/functions/elementary/integers.py
@@ -142,6 +142,12 @@ def _eval_nseries(self, x, n, logx):
         else:
             return r
 
+    def _eval_is_negative(self):
+        return self.args[0].is_negative
+
+    def _eval_is_nonnegative(self):
+        return self.args[0].is_nonnegative
+
     def _eval_rewrite_as_ceiling(self, arg, **kwargs):
         return -ceiling(-arg)
 
@@ -155,17 +161,60 @@ def _eval_Eq(self, other):
                 return S.true
 
     def __le__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] < other + 1
+            if other.is_number and other.is_real:
+                return self.args[0] < ceiling(other)
         if self.args[0] == other and other.is_real:
             return S.true
         if other is S.Infinity and self.is_finite:
             return S.true
+
         return Le(self, other, evaluate=False)
 
+    def __ge__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] >= other
+            if other.is_number and other.is_real:
+                return self.args[0] >= ceiling(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
+        return Ge(self, other, evaluate=False)
+
     def __gt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] >= other + 1
+            if other.is_number and other.is_real:
+                return self.args[0] >= ceiling(other)
         if self.args[0] == other and other.is_real:
             return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
         return Gt(self, other, evaluate=False)
 
+    def __lt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] < other
+            if other.is_number and other.is_real:
+                return self.args[0] < ceiling(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
+        return Lt(self, other, evaluate=False)
 
 class ceiling(RoundFunction):
     """
@@ -234,6 +283,12 @@ def _eval_rewrite_as_floor(self, arg, **kwargs):
     def _eval_rewrite_as_frac(self, arg, **kwargs):
         return arg + frac(-arg)
 
+    def _eval_is_positive(self):
+        return self.args[0].is_positive
+
+    def _eval_is_nonpositive(self):
+        return self.args[0].is_nonpositive
+
     def _eval_Eq(self, other):
         if isinstance(self, ceiling):
             if (self.rewrite(floor) == other) or \
@@ -241,17 +296,60 @@ def _eval_Eq(self, other):
                 return S.true
 
     def __lt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] <= other - 1
+            if other.is_number and other.is_real:
+                return self.args[0] <= floor(other)
         if self.args[0] == other and other.is_real:
             return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
         return Lt(self, other, evaluate=False)
 
+    def __gt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] > other
+            if other.is_number and other.is_real:
+                return self.args[0] > floor(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
+        return Gt(self, other, evaluate=False)
+
     def __ge__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] > other - 1
+            if other.is_number and other.is_real:
+                return self.args[0] > floor(other)
         if self.args[0] == other and other.is_real:
             return S.true
-        if other is S.NegativeInfinity and self.is_real:
+        if other is S.NegativeInfinity and self.is_finite:
             return S.true
+
         return Ge(self, other, evaluate=False)
 
+    def __le__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] <= other
+            if other.is_number and other.is_real:
+                return self.args[0] <= floor(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
+        return Le(self, other, evaluate=False)
 
 class frac(Function):
     r"""Represents the fractional part of x

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/functions/elementary/integers.py | 145 | 145 | 6 | 1 | 2668
| sympy/functions/elementary/integers.py | 158 | 158 | 6 | 1 | 2668
| sympy/functions/elementary/integers.py | 237 | 237 | 1 | 1 | 582
| sympy/functions/elementary/integers.py | 244 | 244 | 1 | 1 | 582


## Problem Statement

```
ceiling(pos) > 0 should be true
Also, shouldn't `floor(neg) < 0`, `floor(pos) >= 0` and `ceiling(neg) <=0` evaluate to True, too?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/functions/elementary/integers.py** | 170 | 253| 582 | 582 | 2961 | 
| 2 | 2 sympy/core/evalf.py | 305 | 340| 342 | 924 | 16679 | 
| 3 | 3 sympy/assumptions/handlers/calculus.py | 55 | 122| 669 | 1593 | 18519 | 
| 4 | 4 sympy/assumptions/ask.py | 396 | 432| 331 | 1924 | 29642 | 
| 5 | 5 sympy/core/mul.py | 1376 | 1389| 157 | 2081 | 45378 | 
| **-> 6 <-** | **5 sympy/functions/elementary/integers.py** | 84 | 167| 587 | 2668 | 45378 | 
| 7 | 6 sympy/ntheory/factor_.py | 2214 | 2237| 137 | 2805 | 63320 | 
| 8 | 7 sympy/assumptions/sathandlers.py | 295 | 381| 1219 | 4024 | 66767 | 
| 9 | 8 sympy/assumptions/handlers/order.py | 217 | 245| 217 | 4241 | 69112 | 
| 10 | 9 sympy/functions/elementary/piecewise.py | 883 | 944| 634 | 4875 | 80249 | 
| 11 | 9 sympy/ntheory/factor_.py | 2108 | 2139| 217 | 5092 | 80249 | 
| 12 | 10 sympy/logic/boolalg.py | 1063 | 1089| 157 | 5249 | 101316 | 
| 13 | 10 sympy/logic/boolalg.py | 2187 | 2263| 831 | 6080 | 101316 | 
| 14 | **10 sympy/functions/elementary/integers.py** | 341 | 434| 674 | 6754 | 101316 | 
| 15 | 10 sympy/assumptions/ask.py | 572 | 603| 275 | 7029 | 101316 | 
| 16 | 10 sympy/assumptions/handlers/calculus.py | 141 | 195| 411 | 7440 | 101316 | 
| 17 | 11 sympy/integrals/meijerint.py | 1175 | 1212| 760 | 8200 | 125610 | 
| 18 | 11 sympy/core/evalf.py | 341 | 414| 699 | 8899 | 125610 | 
| 19 | 11 sympy/assumptions/ask.py | 434 | 470| 343 | 9242 | 125610 | 
| 20 | 11 sympy/ntheory/factor_.py | 2188 | 2211| 150 | 9392 | 125610 | 
| 21 | 12 sympy/plotting/intervalmath/lib_interval.py | 310 | 328| 142 | 9534 | 129261 | 
| 22 | 13 sympy/assumptions/ask_generated.py | 17 | 90| 1512 | 11046 | 132762 | 
| 23 | 13 sympy/integrals/meijerint.py | 1011 | 1052| 628 | 11674 | 132762 | 
| 24 | 13 sympy/assumptions/ask.py | 538 | 570| 265 | 11939 | 132762 | 
| 25 | 13 sympy/assumptions/ask.py | 1511 | 1569| 690 | 12629 | 132762 | 
| 26 | 13 sympy/assumptions/ask.py | 498 | 536| 294 | 12923 | 132762 | 
| 27 | 13 sympy/assumptions/ask_generated.py | 92 | 168| 849 | 13772 | 132762 | 
| 28 | 14 sympy/core/expr.py | 853 | 906| 418 | 14190 | 165203 | 
| 29 | 14 sympy/integrals/meijerint.py | 1096 | 1133| 808 | 14998 | 165203 | 
| 30 | 15 sympy/polys/polyclasses.py | 1669 | 1779| 681 | 15679 | 179700 | 
| 31 | 15 sympy/integrals/meijerint.py | 1213 | 1239| 453 | 16132 | 179700 | 
| 32 | 15 sympy/assumptions/handlers/order.py | 247 | 366| 799 | 16931 | 179700 | 
| 33 | 16 sympy/calculus/util.py | 1425 | 1460| 299 | 17230 | 191723 | 
| 34 | 16 sympy/logic/boolalg.py | 903 | 937| 288 | 17518 | 191723 | 
| 35 | 16 sympy/calculus/util.py | 1462 | 1497| 300 | 17818 | 191723 | 


## Patch

```diff
diff --git a/sympy/functions/elementary/integers.py b/sympy/functions/elementary/integers.py
--- a/sympy/functions/elementary/integers.py
+++ b/sympy/functions/elementary/integers.py
@@ -142,6 +142,12 @@ def _eval_nseries(self, x, n, logx):
         else:
             return r
 
+    def _eval_is_negative(self):
+        return self.args[0].is_negative
+
+    def _eval_is_nonnegative(self):
+        return self.args[0].is_nonnegative
+
     def _eval_rewrite_as_ceiling(self, arg, **kwargs):
         return -ceiling(-arg)
 
@@ -155,17 +161,60 @@ def _eval_Eq(self, other):
                 return S.true
 
     def __le__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] < other + 1
+            if other.is_number and other.is_real:
+                return self.args[0] < ceiling(other)
         if self.args[0] == other and other.is_real:
             return S.true
         if other is S.Infinity and self.is_finite:
             return S.true
+
         return Le(self, other, evaluate=False)
 
+    def __ge__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] >= other
+            if other.is_number and other.is_real:
+                return self.args[0] >= ceiling(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
+        return Ge(self, other, evaluate=False)
+
     def __gt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] >= other + 1
+            if other.is_number and other.is_real:
+                return self.args[0] >= ceiling(other)
         if self.args[0] == other and other.is_real:
             return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
         return Gt(self, other, evaluate=False)
 
+    def __lt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] < other
+            if other.is_number and other.is_real:
+                return self.args[0] < ceiling(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
+        return Lt(self, other, evaluate=False)
 
 class ceiling(RoundFunction):
     """
@@ -234,6 +283,12 @@ def _eval_rewrite_as_floor(self, arg, **kwargs):
     def _eval_rewrite_as_frac(self, arg, **kwargs):
         return arg + frac(-arg)
 
+    def _eval_is_positive(self):
+        return self.args[0].is_positive
+
+    def _eval_is_nonpositive(self):
+        return self.args[0].is_nonpositive
+
     def _eval_Eq(self, other):
         if isinstance(self, ceiling):
             if (self.rewrite(floor) == other) or \
@@ -241,17 +296,60 @@ def _eval_Eq(self, other):
                 return S.true
 
     def __lt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] <= other - 1
+            if other.is_number and other.is_real:
+                return self.args[0] <= floor(other)
         if self.args[0] == other and other.is_real:
             return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
         return Lt(self, other, evaluate=False)
 
+    def __gt__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] > other
+            if other.is_number and other.is_real:
+                return self.args[0] > floor(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.NegativeInfinity and self.is_finite:
+            return S.true
+
+        return Gt(self, other, evaluate=False)
+
     def __ge__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] > other - 1
+            if other.is_number and other.is_real:
+                return self.args[0] > floor(other)
         if self.args[0] == other and other.is_real:
             return S.true
-        if other is S.NegativeInfinity and self.is_real:
+        if other is S.NegativeInfinity and self.is_finite:
             return S.true
+
         return Ge(self, other, evaluate=False)
 
+    def __le__(self, other):
+        other = S(other)
+        if self.args[0].is_real:
+            if other.is_integer:
+                return self.args[0] <= other
+            if other.is_number and other.is_real:
+                return self.args[0] <= floor(other)
+        if self.args[0] == other and other.is_real:
+            return S.false
+        if other is S.Infinity and self.is_finite:
+            return S.true
+
+        return Le(self, other, evaluate=False)
 
 class frac(Function):
     r"""Represents the fractional part of x

```

## Test Patch

```diff
diff --git a/sympy/functions/elementary/tests/test_integers.py b/sympy/functions/elementary/tests/test_integers.py
--- a/sympy/functions/elementary/tests/test_integers.py
+++ b/sympy/functions/elementary/tests/test_integers.py
@@ -108,13 +108,18 @@ def test_floor():
     assert floor(factorial(50)/exp(1)) == \
         11188719610782480504630258070757734324011354208865721592720336800
 
+    assert (floor(y) < y) == False
     assert (floor(y) <= y) == True
     assert (floor(y) > y) == False
+    assert (floor(y) >= y) == False
     assert (floor(x) <= x).is_Relational  # x could be non-real
     assert (floor(x) > x).is_Relational
     assert (floor(x) <= y).is_Relational  # arg is not same as rhs
     assert (floor(x) > y).is_Relational
     assert (floor(y) <= oo) == True
+    assert (floor(y) < oo) == True
+    assert (floor(y) >= -oo) == True
+    assert (floor(y) > -oo) == True
 
     assert floor(y).rewrite(frac) == y - frac(y)
     assert floor(y).rewrite(ceiling) == -ceiling(-y)
@@ -126,6 +131,70 @@ def test_floor():
     assert Eq(floor(y), y - frac(y))
     assert Eq(floor(y), -ceiling(-y))
 
+    neg = Symbol('neg', negative=True)
+    nn = Symbol('nn', nonnegative=True)
+    pos = Symbol('pos', positive=True)
+    np = Symbol('np', nonpositive=True)
+
+    assert (floor(neg) < 0) == True
+    assert (floor(neg) <= 0) == True
+    assert (floor(neg) > 0) == False
+    assert (floor(neg) >= 0) == False
+    assert (floor(neg) <= -1) == True
+    assert (floor(neg) >= -3) == (neg >= -3)
+    assert (floor(neg) < 5) == (neg < 5)
+
+    assert (floor(nn) < 0) == False
+    assert (floor(nn) >= 0) == True
+
+    assert (floor(pos) < 0) == False
+    assert (floor(pos) <= 0) == (pos < 1)
+    assert (floor(pos) > 0) == (pos >= 1)
+    assert (floor(pos) >= 0) == True
+    assert (floor(pos) >= 3) == (pos >= 3)
+
+    assert (floor(np) <= 0) == True
+    assert (floor(np) > 0) == False
+
+    assert floor(neg).is_negative == True
+    assert floor(neg).is_nonnegative == False
+    assert floor(nn).is_negative == False
+    assert floor(nn).is_nonnegative == True
+    assert floor(pos).is_negative == False
+    assert floor(pos).is_nonnegative == True
+    assert floor(np).is_negative is None
+    assert floor(np).is_nonnegative is None
+
+    assert (floor(7, evaluate=False) >= 7) == True
+    assert (floor(7, evaluate=False) > 7) == False
+    assert (floor(7, evaluate=False) <= 7) == True
+    assert (floor(7, evaluate=False) < 7) == False
+
+    assert (floor(7, evaluate=False) >= 6) == True
+    assert (floor(7, evaluate=False) > 6) == True
+    assert (floor(7, evaluate=False) <= 6) == False
+    assert (floor(7, evaluate=False) < 6) == False
+
+    assert (floor(7, evaluate=False) >= 8) == False
+    assert (floor(7, evaluate=False) > 8) == False
+    assert (floor(7, evaluate=False) <= 8) == True
+    assert (floor(7, evaluate=False) < 8) == True
+
+    assert (floor(x) <= 5.5) == Le(floor(x), 5.5, evaluate=False)
+    assert (floor(x) >= -3.2) == Ge(floor(x), -3.2, evaluate=False)
+    assert (floor(x) < 2.9) == Lt(floor(x), 2.9, evaluate=False)
+    assert (floor(x) > -1.7) == Gt(floor(x), -1.7, evaluate=False)
+
+    assert (floor(y) <= 5.5) == (y < 6)
+    assert (floor(y) >= -3.2) == (y >= -3)
+    assert (floor(y) < 2.9) == (y < 3)
+    assert (floor(y) > -1.7) == (y >= -1)
+
+    assert (floor(y) <= n) == (y < n + 1)
+    assert (floor(y) >= n) == (y >= n)
+    assert (floor(y) < n) == (y < n)
+    assert (floor(y) > n) == (y >= n + 1)
+
 
 def test_ceiling():
 
@@ -225,12 +294,17 @@ def test_ceiling():
         11188719610782480504630258070757734324011354208865721592720336801
 
     assert (ceiling(y) >= y) == True
+    assert (ceiling(y) > y) == False
     assert (ceiling(y) < y) == False
+    assert (ceiling(y) <= y) == False
     assert (ceiling(x) >= x).is_Relational  # x could be non-real
     assert (ceiling(x) < x).is_Relational
     assert (ceiling(x) >= y).is_Relational  # arg is not same as rhs
     assert (ceiling(x) < y).is_Relational
     assert (ceiling(y) >= -oo) == True
+    assert (ceiling(y) > -oo) == True
+    assert (ceiling(y) <= oo) == True
+    assert (ceiling(y) < oo) == True
 
     assert ceiling(y).rewrite(floor) == -floor(-y)
     assert ceiling(y).rewrite(frac) == y + frac(-y)
@@ -242,6 +316,70 @@ def test_ceiling():
     assert Eq(ceiling(y), y + frac(-y))
     assert Eq(ceiling(y), -floor(-y))
 
+    neg = Symbol('neg', negative=True)
+    nn = Symbol('nn', nonnegative=True)
+    pos = Symbol('pos', positive=True)
+    np = Symbol('np', nonpositive=True)
+
+    assert (ceiling(neg) <= 0) == True
+    assert (ceiling(neg) < 0) == (neg <= -1)
+    assert (ceiling(neg) > 0) == False
+    assert (ceiling(neg) >= 0) == (neg > -1)
+    assert (ceiling(neg) > -3) == (neg > -3)
+    assert (ceiling(neg) <= 10) == (neg <= 10)
+
+    assert (ceiling(nn) < 0) == False
+    assert (ceiling(nn) >= 0) == True
+
+    assert (ceiling(pos) < 0) == False
+    assert (ceiling(pos) <= 0) == False
+    assert (ceiling(pos) > 0) == True
+    assert (ceiling(pos) >= 0) == True
+    assert (ceiling(pos) >= 1) == True
+    assert (ceiling(pos) > 5) == (pos > 5)
+
+    assert (ceiling(np) <= 0) == True
+    assert (ceiling(np) > 0) == False
+
+    assert ceiling(neg).is_positive == False
+    assert ceiling(neg).is_nonpositive == True
+    assert ceiling(nn).is_positive is None
+    assert ceiling(nn).is_nonpositive is None
+    assert ceiling(pos).is_positive == True
+    assert ceiling(pos).is_nonpositive == False
+    assert ceiling(np).is_positive == False
+    assert ceiling(np).is_nonpositive == True
+
+    assert (ceiling(7, evaluate=False) >= 7) == True
+    assert (ceiling(7, evaluate=False) > 7) == False
+    assert (ceiling(7, evaluate=False) <= 7) == True
+    assert (ceiling(7, evaluate=False) < 7) == False
+
+    assert (ceiling(7, evaluate=False) >= 6) == True
+    assert (ceiling(7, evaluate=False) > 6) == True
+    assert (ceiling(7, evaluate=False) <= 6) == False
+    assert (ceiling(7, evaluate=False) < 6) == False
+
+    assert (ceiling(7, evaluate=False) >= 8) == False
+    assert (ceiling(7, evaluate=False) > 8) == False
+    assert (ceiling(7, evaluate=False) <= 8) == True
+    assert (ceiling(7, evaluate=False) < 8) == True
+
+    assert (ceiling(x) <= 5.5) == Le(ceiling(x), 5.5, evaluate=False)
+    assert (ceiling(x) >= -3.2) == Ge(ceiling(x), -3.2, evaluate=False)
+    assert (ceiling(x) < 2.9) == Lt(ceiling(x), 2.9, evaluate=False)
+    assert (ceiling(x) > -1.7) == Gt(ceiling(x), -1.7, evaluate=False)
+
+    assert (ceiling(y) <= 5.5) == (y <= 5)
+    assert (ceiling(y) >= -3.2) == (y > -4)
+    assert (ceiling(y) < 2.9) == (y <= 2)
+    assert (ceiling(y) > -1.7) == (y > -2)
+
+    assert (ceiling(y) <= n) == (y <= n)
+    assert (ceiling(y) >= n) == (y > n - 1)
+    assert (ceiling(y) < n) == (y <= n - 1)
+    assert (ceiling(y) > n) == (y > n)
+
 
 def test_frac():
     assert isinstance(frac(x), frac)

```


## Code snippets

### 1 - sympy/functions/elementary/integers.py:

Start line: 170, End line: 253

```python
class ceiling(RoundFunction):
    """
    Ceiling is a univariate function which returns the smallest integer
    value not less than its argument. This implementation
    generalizes ceiling to complex numbers by taking the ceiling of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import ceiling, E, I, S, Float, Rational
    >>> ceiling(17)
    17
    >>> ceiling(Rational(23, 10))
    3
    >>> ceiling(2*E)
    6
    >>> ceiling(-Float(0.567))
    0
    >>> ceiling(I/2)
    I
    >>> ceiling(S(5)/2 + 5*I/2)
    3 + 3*I

    See Also
    ========

    sympy.functions.elementary.integers.floor

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] http://mathworld.wolfram.com/CeilingFunction.html

    """
    _dir = 1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.ceiling()
        elif any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[1]

    def _eval_nseries(self, x, n, logx):
        r = self.subs(x, 0)
        args = self.args[0]
        args0 = args.subs(x, 0)
        if args0 == r:
            direction = (args - args0).leadterm(x)[0]
            if direction.is_positive:
                return r + 1
            else:
                return r
        else:
            return r

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return -floor(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg + frac(-arg)

    def _eval_Eq(self, other):
        if isinstance(self, ceiling):
            if (self.rewrite(floor) == other) or \
                    (self.rewrite(frac) == other):
                return S.true

    def __lt__(self, other):
        if self.args[0] == other and other.is_real:
            return S.false
        return Lt(self, other, evaluate=False)

    def __ge__(self, other):
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.NegativeInfinity and self.is_real:
            return S.true
        return Ge(self, other, evaluate=False)
```
### 2 - sympy/core/evalf.py:

Start line: 305, End line: 340

```python
def get_integer_part(expr, no, options, return_ints=False):
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # The expression is likely less than 2^30 or so
    assumed_size = 30
    ire, iim, ire_acc, iim_acc = evalf(expr, assumed_size, options)

    # We now know the size, so we can calculate how much extra precision
    # (if any) is needed to get within the nearest integer
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # ... or maybe the expression was exactly zero
        return None, None, None, None

    margin = 10

    if gap >= -margin:
        prec = margin + assumed_size + gap
        ire, iim, ire_acc, iim_acc = evalf(
            expr, prec, options)
    else:
        prec = assumed_size

    # We can now easily find the nearest integer, but to find floor/ceil, we
    # must also calculate whether the difference to the nearest integer is
    # positive or negative (which may fail if very close).
    # ... other code
```
### 3 - sympy/assumptions/handlers/calculus.py:

Start line: 55, End line: 122

```python
class AskFiniteHandler(CommonHandler):

    @staticmethod
    def Add(expr, assumptions):
        """
        Return True if expr is bounded, False if not and None if unknown.

        Truth Table:

        +-------+-----+-----------+-----------+
        |       |     |           |           |
        |       |  B  |     U     |     ?     |
        |       |     |           |           |
        +-------+-----+---+---+---+---+---+---+
        |       |     |   |   |   |   |   |   |
        |       |     |'+'|'-'|'x'|'+'|'-'|'x'|
        |       |     |   |   |   |   |   |   |
        +-------+-----+---+---+---+---+---+---+
        |       |     |           |           |
        |   B   |  B  |     U     |     ?     |
        |       |     |           |           |
        +---+---+-----+---+---+---+---+---+---+
        |   |   |     |   |   |   |   |   |   |
        |   |'+'|     | U | ? | ? | U | ? | ? |
        |   |   |     |   |   |   |   |   |   |
        |   +---+-----+---+---+---+---+---+---+
        |   |   |     |   |   |   |   |   |   |
        | U |'-'|     | ? | U | ? | ? | U | ? |
        |   |   |     |   |   |   |   |   |   |
        |   +---+-----+---+---+---+---+---+---+
        |   |   |     |           |           |
        |   |'x'|     |     ?     |     ?     |
        |   |   |     |           |           |
        +---+---+-----+---+---+---+---+---+---+
        |       |     |           |           |
        |   ?   |     |           |     ?     |
        |       |     |           |           |
        +-------+-----+-----------+---+---+---+

            * 'B' = Bounded

            * 'U' = Unbounded

            * '?' = unknown boundedness

            * '+' = positive sign

            * '-' = negative sign

            * 'x' = sign unknown



            * All Bounded -> True

            * 1 Unbounded and the rest Bounded -> False

            * >1 Unbounded, all with same known sign -> False

            * Any Unknown and unknown sign -> None

            * Else -> None

        When the signs are not the same you can have an undefined
        result as in oo - oo, hence 'bounded' is also undefined.

        """

        sign = -1  # sign of unknown or infinite
        result = True
        # ... other code
```
### 4 - sympy/assumptions/ask.py:

Start line: 396, End line: 432

```python
class AssumptionKeys(object):

    @predicate_memo
    def positive(self):
        r"""
        Positive real number predicate.

        ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
        is in the interval `(0, \infty)`.  In particular, infinity is not
        positive.

        A few important facts about positive numbers:

        - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
          thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
          whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
          positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
          `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
          true, whereas ``Q.nonpositive(I)`` is false.

        - See the documentation of ``Q.real`` for more information about
          related facts.

        Examples
        ========

        >>> from sympy import Q, ask, symbols, I
        >>> x = symbols('x')
        >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
        True
        >>> ask(Q.positive(1))
        True
        >>> ask(Q.nonpositive(I))
        False
        >>> ask(~Q.positive(I))
        True

        """
        return Predicate('positive')
```
### 5 - sympy/core/mul.py:

Start line: 1376, End line: 1389

```python
class Mul(Expr, AssocOp):

    def _eval_is_extended_positive(self):
        """Return True if self is positive, False if not, and None if it
        cannot be determined.

        This algorithm is non-recursive and works by keeping track of the
        sign which changes when a negative or nonpositive is encountered.
        Whether a nonpositive or nonnegative is seen is also tracked since
        the presence of these makes it impossible to return True, but
        possible to return False if the end result is nonpositive. e.g.

            pos * neg * nonpositive -> pos or zero -> None is returned
            pos * neg * nonnegative -> neg or zero -> False is returned
        """
        return self._eval_pos_neg(1)
```
### 6 - sympy/functions/elementary/integers.py:

Start line: 84, End line: 167

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
        if other is S.Infinity and self.is_finite:
            return S.true
        return Le(self, other, evaluate=False)

    def __gt__(self, other):
        if self.args[0] == other and other.is_real:
            return S.false
        return Gt(self, other, evaluate=False)
```
### 7 - sympy/ntheory/factor_.py:

Start line: 2214, End line: 2237

```python
def is_deficient(n):
    """Returns True if ``n`` is a deficient number, else False.

    A deficient number is greater than the sum of its positive proper divisors.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_deficient
    >>> is_deficient(20)
    False
    >>> is_deficient(15)
    True

    References
    ==========

    .. [1] http://mathworld.wolfram.com/DeficientNumber.html

    """
    n = as_int(n)
    if is_perfect(n):
        return False
    return bool(abundance(n) < 0)
```
### 8 - sympy/assumptions/sathandlers.py:

Start line: 295, End line: 381

```python
for klass, fact in [
    (Mul, Equivalent(Q.zero, AnyArgs(Q.zero))),
    (MatMul, Implies(AllArgs(Q.square), Equivalent(Q.invertible, AllArgs(Q.invertible)))),
    (Add, Implies(AllArgs(Q.positive), Q.positive)),
    (Add, Implies(AllArgs(Q.negative), Q.negative)),
    (Mul, Implies(AllArgs(Q.positive), Q.positive)),
    (Mul, Implies(AllArgs(Q.commutative), Q.commutative)),
    (Mul, Implies(AllArgs(Q.real), Q.commutative)),

    (Pow, CustomLambda(lambda power: Implies(Q.real(power.base) &
    Q.even(power.exp) & Q.nonnegative(power.exp), Q.nonnegative(power)))),
    (Pow, CustomLambda(lambda power: Implies(Q.nonnegative(power.base) & Q.odd(power.exp) & Q.nonnegative(power.exp), Q.nonnegative(power)))),
    (Pow, CustomLambda(lambda power: Implies(Q.nonpositive(power.base) & Q.odd(power.exp) & Q.nonnegative(power.exp), Q.nonpositive(power)))),

    # This one can still be made easier to read. I think we need basic pattern
    # matching, so that we can just write Equivalent(Q.zero(x**y), Q.zero(x) & Q.positive(y))
    (Pow, CustomLambda(lambda power: Equivalent(Q.zero(power), Q.zero(power.base) & Q.positive(power.exp)))),
    (Integer, CheckIsPrime(Q.prime)),
    (Integer, CheckOldAssump(Q.composite)),
    # Implicitly assumes Mul has more than one arg
    # Would be AllArgs(Q.prime | Q.composite) except 1 is composite
    (Mul, Implies(AllArgs(Q.prime), ~Q.prime)),
    # More advanced prime assumptions will require inequalities, as 1 provides
    # a corner case.
    (Mul, Implies(AllArgs(Q.imaginary | Q.real), Implies(ExactlyOneArg(Q.imaginary), Q.imaginary))),
    (Mul, Implies(AllArgs(Q.real), Q.real)),
    (Add, Implies(AllArgs(Q.real), Q.real)),
    # General Case: Odd number of imaginary args implies mul is imaginary(To be implemented)
    (Mul, Implies(AllArgs(Q.real), Implies(ExactlyOneArg(Q.irrational),
        Q.irrational))),
    (Add, Implies(AllArgs(Q.real), Implies(ExactlyOneArg(Q.irrational),
        Q.irrational))),
    (Mul, Implies(AllArgs(Q.rational), Q.rational)),
    (Add, Implies(AllArgs(Q.rational), Q.rational)),

    (Abs, Q.nonnegative),
    (Abs, Equivalent(AllArgs(~Q.zero), ~Q.zero)),

    # Including the integer qualification means we don't need to add any facts
    # for odd, since the assumptions already know that every integer is
    # exactly one of even or odd.
    (Mul, Implies(AllArgs(Q.integer), Equivalent(AnyArgs(Q.even), Q.even))),

    (Abs, Implies(AllArgs(Q.even), Q.even)),
    (Abs, Implies(AllArgs(Q.odd), Q.odd)),

    (Add, Implies(AllArgs(Q.integer), Q.integer)),
    (Add, Implies(ExactlyOneArg(~Q.integer), ~Q.integer)),
    (Mul, Implies(AllArgs(Q.integer), Q.integer)),
    (Mul, Implies(ExactlyOneArg(~Q.rational), ~Q.integer)),
    (Abs, Implies(AllArgs(Q.integer), Q.integer)),

    (Number, CheckOldAssump(Q.negative)),
    (Number, CheckOldAssump(Q.zero)),
    (Number, CheckOldAssump(Q.positive)),
    (Number, CheckOldAssump(Q.nonnegative)),
    (Number, CheckOldAssump(Q.nonzero)),
    (Number, CheckOldAssump(Q.nonpositive)),
    (Number, CheckOldAssump(Q.rational)),
    (Number, CheckOldAssump(Q.irrational)),
    (Number, CheckOldAssump(Q.even)),
    (Number, CheckOldAssump(Q.odd)),
    (Number, CheckOldAssump(Q.integer)),
    (Number, CheckOldAssump(Q.imaginary)),
    # For some reason NumberSymbol does not subclass Number
    (NumberSymbol, CheckOldAssump(Q.negative)),
    (NumberSymbol, CheckOldAssump(Q.zero)),
    (NumberSymbol, CheckOldAssump(Q.positive)),
    (NumberSymbol, CheckOldAssump(Q.nonnegative)),
    (NumberSymbol, CheckOldAssump(Q.nonzero)),
    (NumberSymbol, CheckOldAssump(Q.nonpositive)),
    (NumberSymbol, CheckOldAssump(Q.rational)),
    (NumberSymbol, CheckOldAssump(Q.irrational)),
    (NumberSymbol, CheckOldAssump(Q.imaginary)),
    (ImaginaryUnit, CheckOldAssump(Q.negative)),
    (ImaginaryUnit, CheckOldAssump(Q.zero)),
    (ImaginaryUnit, CheckOldAssump(Q.positive)),
    (ImaginaryUnit, CheckOldAssump(Q.nonnegative)),
    (ImaginaryUnit, CheckOldAssump(Q.nonzero)),
    (ImaginaryUnit, CheckOldAssump(Q.nonpositive)),
    (ImaginaryUnit, CheckOldAssump(Q.rational)),
    (ImaginaryUnit, CheckOldAssump(Q.irrational)),
    (ImaginaryUnit, CheckOldAssump(Q.imaginary))
    ]:

    register_fact(klass, fact)
```
### 9 - sympy/assumptions/handlers/order.py:

Start line: 217, End line: 245

```python
class AskPositiveHandler(CommonHandler):
    """
    Handler for key 'positive'
    Test that an expression is greater (strict) than zero
    """

    @staticmethod
    def Expr(expr, assumptions):
        return expr.is_positive

    @staticmethod
    def _number(expr, assumptions):
        r, i = expr.as_real_imag()
        # If the imaginary part can symbolically be shown to be zero then
        # we just evaluate the real part; otherwise we evaluate the imaginary
        # part to see if it actually evaluates to zero and if it does then
        # we make the comparison between the real part and zero.
        if not i:
            r = r.evalf(2)
            if r._prec != 1:
                return r > 0
        else:
            i = i.evalf(2)
            if i._prec != 1:
                if i != 0:
                    return False
                r = r.evalf(2)
                if r._prec != 1:
                    return r > 0
```
### 10 - sympy/functions/elementary/piecewise.py:

Start line: 883, End line: 944

```python
class Piecewise(Function):

    def _eval_transpose(self):
        return self.func(*[(e.transpose(), c) for e, c in self.args])

    def _eval_template_is_attr(self, is_attr):
        b = None
        for expr, _ in self.args:
            a = getattr(expr, is_attr)
            if a is None:
                return
            if b is None:
                b = a
            elif b is not a:
                return
        return b

    _eval_is_finite = lambda self: self._eval_template_is_attr(
        'is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr(
        'is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr(
        'is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr(
        'is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr(
        'is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr(
        'is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr(
            'is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr(
            'is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr(
            'is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr(
            'is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr(
            'is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr(
            'is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr(
        'is_zero')

    @classmethod
    def __eval_cond(cls, cond):
        """Return the truth value of the condition."""
        if cond == True:
            return True
        if isinstance(cond, Equality):
            try:
                diff = cond.lhs - cond.rhs
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass
```
### 14 - sympy/functions/elementary/integers.py:

Start line: 341, End line: 434

```python
class frac(Function):

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_Eq(self, other):
        if isinstance(self, frac):
            if (self.rewrite(floor) == other) or \
                    (self.rewrite(ceiling) == other):
                return S.true
            # Check if other < 0
            if other.is_extended_negative:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return S.false

    def _eval_is_finite(self):
        return True

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    def _eval_is_negative(self):
        return False

    def __ge__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.true
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
            # Check if other >= 1
            if other.is_extended_negative:
                return S.true
        return Gt(self, other, evaluate=False)

    def __le__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            if other.is_extended_negative:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Le(self, other, evaluate=False)

    def __lt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Lt(self, other, evaluate=False)

    def _value_one_or_more(self, other):
        if other.is_extended_real:
            if other.is_number:
                res = other >= 1
                if res and not isinstance(res, Relational):
                    return S.true
            if other.is_integer and other.is_positive:
                return S.true
```
