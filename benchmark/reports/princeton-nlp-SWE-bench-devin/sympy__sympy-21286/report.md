# sympy__sympy-21286

| **sympy/sympy** | `546e10799fe55b3e59dea8fa6b3a6d6e71843d33` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 674 |
| **Avg pos** | 51.0 |
| **Min pos** | 1 |
| **Max pos** | 11 |
| **Top file pos** | 1 |
| **Missing snippets** | 13 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -6,11 +6,11 @@
 from sympy.core.function import Lambda
 from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
 from sympy.core.numbers import oo
-from sympy.core.relational import Eq
+from sympy.core.relational import Eq, is_eq
 from sympy.core.singleton import Singleton, S
 from sympy.core.symbol import Dummy, symbols, Symbol
 from sympy.core.sympify import _sympify, sympify, converter
-from sympy.logic.boolalg import And
+from sympy.logic.boolalg import And, Or
 from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
     ProductSet)
 from sympy.utilities.misc import filldedent
@@ -571,7 +571,7 @@ class Range(Set):
         >>> r.inf
         n
         >>> pprint(r)
-        {n, n + 3, ..., n + 17}
+        {n, n + 3, ..., n + 18}
     """
 
     is_iterable = True
@@ -598,6 +598,8 @@ def __new__(cls, *args):
                         w.has(Symbol) and w.is_integer != False):
                     ok.append(w)
                 elif not w.is_Integer:
+                    if w.is_infinite:
+                        raise ValueError('infinite symbols not allowed')
                     raise ValueError
                 else:
                     ok.append(w)
@@ -610,10 +612,25 @@ def __new__(cls, *args):
 
         null = False
         if any(i.has(Symbol) for i in (start, stop, step)):
-            if start == stop:
+            dif = stop - start
+            n = dif/step
+            if n.is_Rational:
+                from sympy import floor
+                if dif == 0:
+                    null = True
+                else:  # (x, x + 5, 2) or (x, 3*x, x)
+                    n = floor(n)
+                    end = start + n*step
+                    if dif.is_Rational:  # (x, x + 5, 2)
+                        if (end - stop).is_negative:
+                            end += step
+                    else:  # (x, 3*x, x)
+                        if (end/stop - 1).is_negative:
+                            end += step
+            elif n.is_extended_negative:
                 null = True
             else:
-                end = stop
+                end = stop  # other methods like sup and reversed must fail
         elif start.is_infinite:
             span = step*(stop - start)
             if span is S.NaN or span <= 0:
@@ -631,8 +648,8 @@ def __new__(cls, *args):
             if n <= 0:
                 null = True
             elif oostep:
-                end = start + 1
-                step = S.One  # make it a canonical single step
+                step = S.One  # make it canonical
+                end = start + step
             else:
                 end = start + n*step
         if null:
@@ -656,34 +673,42 @@ def reversed(self):
         Range(9, -1, -1)
         """
         if self.has(Symbol):
-            _ = self.size  # validate
-        if not self:
+            n = (self.stop - self.start)/self.step
+            if not n.is_extended_positive or not all(
+                    i.is_integer or i.is_infinite for i in self.args):
+                raise ValueError('invalid method for symbolic range')
+        if self.start == self.stop:
             return self
         return self.func(
             self.stop - self.step, self.start - self.step, -self.step)
 
     def _contains(self, other):
-        if not self:
+        if self.start == self.stop:
             return S.false
         if other.is_infinite:
             return S.false
         if not other.is_integer:
             return other.is_integer
         if self.has(Symbol):
-            try:
-                _ = self.size  # validate
-            except ValueError:
+            n = (self.stop - self.start)/self.step
+            if not n.is_extended_positive or not all(
+                    i.is_integer or i.is_infinite for i in self.args):
                 return
+        else:
+            n = self.size
         if self.start.is_finite:
             ref = self.start
         elif self.stop.is_finite:
             ref = self.stop
         else:  # both infinite; step is +/- 1 (enforced by __new__)
             return S.true
-        if self.size == 1:
+        if n == 1:
             return Eq(other, self[0])
         res = (ref - other) % self.step
         if res == S.Zero:
+            if self.has(Symbol):
+                d = Dummy('i')
+                return self.as_relational(d).subs(d, other)
             return And(other >= self.inf, other <= self.sup)
         elif res.is_Integer:  # off sequence
             return S.false
@@ -691,20 +716,19 @@ def _contains(self, other):
             return None
 
     def __iter__(self):
-        if self.has(Symbol):
-            _ = self.size  # validate
+        n = self.size  # validate
         if self.start in [S.NegativeInfinity, S.Infinity]:
             raise TypeError("Cannot iterate over Range with infinite start")
-        elif self:
+        elif self.start != self.stop:
             i = self.start
-            step = self.step
-
-            while True:
-                if (step > 0 and not (self.start <= i < self.stop)) or \
-                   (step < 0 and not (self.stop < i <= self.start)):
-                    break
-                yield i
-                i += step
+            if n.is_infinite:
+                while True:
+                    yield i
+                    i += self.step
+            else:
+                for j in range(n):
+                    yield i
+                    i += self.step
 
     def __len__(self):
         rv = self.size
@@ -714,15 +738,15 @@ def __len__(self):
 
     @property
     def size(self):
-        if not self:
+        if self.start == self.stop:
             return S.Zero
         dif = self.stop - self.start
-        n = abs(dif // self.step)
-        if not n.is_Integer:
-            if n.is_infinite:
-                return S.Infinity
+        n = dif/self.step
+        if n.is_infinite:
+            return S.Infinity
+        if not n.is_Integer or not all(i.is_integer for i in self.args):
             raise ValueError('invalid method for symbolic range')
-        return n
+        return abs(n)
 
     @property
     def is_finite_set(self):
@@ -731,7 +755,13 @@ def is_finite_set(self):
         return self.size.is_finite
 
     def __bool__(self):
-        return self.start != self.stop
+        # this only distinguishes between definite null range
+        # and non-null/unknown null; getting True doesn't mean
+        # that it actually is not null
+        b = is_eq(self.start, self.stop)
+        if b is None:
+            raise ValueError('cannot tell if Range is null or not')
+        return not bool(b)
 
     def __getitem__(self, i):
         from sympy.functions.elementary.integers import ceiling
@@ -745,6 +775,8 @@ def __getitem__(self, i):
             "with an infinite value"
         if isinstance(i, slice):
             if self.size.is_finite:  # validates, too
+                if self.start == self.stop:
+                    return Range(0)
                 start, stop, step = i.indices(self.size)
                 n = ceiling((stop - start)/step)
                 if n <= 0:
@@ -845,44 +877,40 @@ def __getitem__(self, i):
                 elif start > 0:
                     raise ValueError(ooslice)
         else:
-            if not self:
+            if self.start == self.stop:
                 raise IndexError('Range index out of range')
+            if not (all(i.is_integer or i.is_infinite
+                    for i in self.args) and ((self.stop - self.start)/
+                    self.step).is_extended_positive):
+                raise ValueError('invalid method for symbolic range')
             if i == 0:
                 if self.start.is_infinite:
                     raise ValueError(ooslice)
-                if self.has(Symbol):
-                    if (self.stop > self.start) == self.step.is_positive and self.step.is_positive is not None:
-                        pass
-                    else:
-                        _ = self.size  # validate
                 return self.start
             if i == -1:
                 if self.stop.is_infinite:
                     raise ValueError(ooslice)
-                n = self.stop - self.step
-                if n.is_Integer or (
-                        n.is_integer and (
-                            (n - self.start).is_nonnegative ==
-                            self.step.is_positive)):
-                    return n
-            _ = self.size  # validate
+                return self.stop - self.step
+            n = self.size  # must be known for any other index
             rv = (self.stop if i < 0 else self.start) + i*self.step
             if rv.is_infinite:
                 raise ValueError(ooslice)
-            if rv < self.inf or rv > self.sup:
-                raise IndexError("Range index out of range")
-            return rv
+            if 0 <= (rv - self.start)/self.step <= n:
+                return rv
+            raise IndexError("Range index out of range")
 
     @property
     def _inf(self):
         if not self:
-            raise NotImplementedError
+            return S.EmptySet.inf
         if self.has(Symbol):
-            if self.step.is_positive:
-                return self[0]
-            elif self.step.is_negative:
-                return self[-1]
-            _ = self.size  # validate
+            if all(i.is_integer or i.is_infinite for i in self.args):
+                dif = self.stop - self.start
+                if self.step.is_positive and dif.is_positive:
+                    return self.start
+                elif self.step.is_negative and dif.is_negative:
+                    return self.stop - self.step
+            raise ValueError('invalid method for symbolic range')
         if self.step > 0:
             return self.start
         else:
@@ -891,13 +919,15 @@ def _inf(self):
     @property
     def _sup(self):
         if not self:
-            raise NotImplementedError
+            return S.EmptySet.sup
         if self.has(Symbol):
-            if self.step.is_positive:
-                return self[-1]
-            elif self.step.is_negative:
-                return self[0]
-            _ = self.size  # validate
+            if all(i.is_integer or i.is_infinite for i in self.args):
+                dif = self.stop - self.start
+                if self.step.is_positive and dif.is_positive:
+                    return self.stop - self.step
+                elif self.step.is_negative and dif.is_negative:
+                    return self.start
+            raise ValueError('invalid method for symbolic range')
         if self.step > 0:
             return self.stop - self.step
         else:
@@ -909,27 +939,37 @@ def _boundary(self):
 
     def as_relational(self, x):
         """Rewrite a Range in terms of equalities and logic operators. """
-        if self.size == 1:
-            return Eq(x, self[0])
-        elif self.size == 0:
-            return S.false
+        from sympy.core.mod import Mod
+        if self.start.is_infinite:
+            assert not self.stop.is_infinite  # by instantiation
+            a = self.reversed.start
         else:
-            from sympy.core.mod import Mod
-            cond = None
-            if self.start.is_infinite:
-                if self.stop.is_infinite:
-                    cond = S.true
-                else:
-                    a = self.reversed.start
-            elif self.start == self.stop:
-                cond = S.false  # null range
-            else:
-                a = self.start
-            step = abs(self.step)
-            cond = Eq(Mod(x, step), a % step) if cond is None else cond
-            return And(cond,
-                       x >= self.inf if self.inf in self else x > self.inf,
-                       x <= self.sup if self.sup in self else x < self.sup)
+            a = self.start
+        step = self.step
+        in_seq = Eq(Mod(x - a, step), 0)
+        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
+        n = (self.stop - self.start)/self.step
+        if n == 0:
+            return S.EmptySet.as_relational(x)
+        if n == 1:
+            return And(Eq(x, a), ints)
+        try:
+            a, b = self.inf, self.sup
+        except ValueError:
+            a = None
+        if a is not None:
+            range_cond = And(
+                x > a if a.is_infinite else x >= a,
+                x < b if b.is_infinite else x <= b)
+        else:
+            a, b = self.start, self.stop - self.step
+            range_cond = Or(
+                And(self.step >= 1, x > a if a.is_infinite else x >= a,
+                x < b if b.is_infinite else x <= b),
+                And(self.step <= -1, x < a if a.is_infinite else x <= a,
+                x > b if b.is_infinite else x >= b))
+        return And(in_seq, ints, range_cond)
+
 
 converter[range] = lambda r: Range(r.start, r.stop, r.step)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/fancysets.py | 9 | 13 | - | 1 | -
| sympy/sets/fancysets.py | 574 | 574 | 1 | 1 | 674
| sympy/sets/fancysets.py | 601 | 601 | 5 | 1 | 2666
| sympy/sets/fancysets.py | 613 | 616 | 5 | 1 | 2666
| sympy/sets/fancysets.py | 634 | 635 | 5 | 1 | 2666
| sympy/sets/fancysets.py | 659 | 683 | - | 1 | -
| sympy/sets/fancysets.py | 694 | 707 | 6 | 1 | 2952
| sympy/sets/fancysets.py | 717 | 725 | 6 | 1 | 2952
| sympy/sets/fancysets.py | 734 | 734 | 6 | 1 | 2952
| sympy/sets/fancysets.py | 748 | 748 | 2 | 1 | 1772
| sympy/sets/fancysets.py | 848 | 885 | - | 1 | -
| sympy/sets/fancysets.py | 894 | 900 | 11 | 1 | 4452
| sympy/sets/fancysets.py | 912 | 932 | 4 | 1 | 2133


## Problem Statement

```
make symbolic Range more canonical
Whereas a Range with numerical args is canonical, the Range containing symbols is not:
\`\`\`python
>>> [Range(3,j,2) for j in range(4,10)]
[Range(3, 5, 2), Range(3, 5, 2), Range(3, 7, 2), Range(3, 7, 2), Range(3, 9, 2), Range(3, 9, 2)]

vs

>>> [Range(i,i+j,5) for j in range(1,6)]
[Range(i, i + 1, 5), Range(i, i + 2, 5), Range(i, i + 3, 5), Range(i, i + 4, 5), Range(i, i + 5, 5)]

which could be

[Range(i, i + 1, 5), Range(i, i + 1, 5), Range(i, i + 1, 5), Range(i, i + 1, 5), Range(i, i + 1, 5)]

\`\`\`
The incorrect results are based on the assumption that the instantiated Range is canonical:
\`\`\`python
>> r=Range(2, 2 + 3, 5)
>>> r.inf,r.reversed.sup
(2, 2)
>>> r = Range(k, k + 3, 5)
>>> r.inf,r.reversed.sup
(k, k - 2)
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/sets/fancysets.py** | 494 | 577| 674 | 674 | 10890 | 
| **-> 2 <-** | **1 sympy/sets/fancysets.py** | 746 | 874| 1098 | 1772 | 10890 | 
| 3 | **1 sympy/sets/fancysets.py** | 643 | 663| 143 | 1915 | 10890 | 
| **-> 4 <-** | **1 sympy/sets/fancysets.py** | 910 | 934| 218 | 2133 | 10890 | 
| **-> 5 <-** | **1 sympy/sets/fancysets.py** | 579 | 641| 533 | 2666 | 10890 | 
| **-> 6 <-** | **1 sympy/sets/fancysets.py** | 693 | 734| 286 | 2952 | 10890 | 
| 7 | **1 sympy/sets/fancysets.py** | 736 | 745| 151 | 3103 | 10890 | 
| 8 | 2 sympy/calculus/util.py | 173 | 222| 347 | 3450 | 22773 | 
| 9 | **2 sympy/sets/fancysets.py** | 665 | 691| 206 | 3656 | 22773 | 
| 10 | 2 sympy/calculus/util.py | 89 | 171| 611 | 4267 | 22773 | 
| **-> 11 <-** | **2 sympy/sets/fancysets.py** | 876 | 908| 185 | 4452 | 22773 | 
| 12 | 3 sympy/tensor/indexed.py | 306 | 335| 291 | 4743 | 28806 | 
| 13 | 4 sympy/plotting/plot.py | 2381 | 2431| 430 | 5173 | 49797 | 
| 14 | 5 sympy/tensor/array/array_comprehension.py | 163 | 185| 198 | 5371 | 52884 | 
| 15 | 6 sympy/sets/handlers/intersection.py | 105 | 220| 985 | 6356 | 57067 | 
| 16 | 6 sympy/tensor/array/array_comprehension.py | 187 | 200| 112 | 6468 | 57067 | 
| 17 | 6 sympy/plotting/plot.py | 2433 | 2485| 526 | 6994 | 57067 | 
| 18 | 6 sympy/tensor/array/array_comprehension.py | 1 | 35| 354 | 7348 | 57067 | 
| 19 | 7 sympy/combinatorics/tensor_can.py | 1 | 71| 746 | 8094 | 70139 | 
| 20 | 8 sympy/stats/rv.py | 309 | 339| 220 | 8314 | 82784 | 
| 21 | 9 sympy/core/symbol.py | 555 | 670| 1075 | 9389 | 89672 | 
| 22 | 10 sympy/concrete/expr_with_limits.py | 84 | 158| 752 | 10141 | 94191 | 
| 23 | 11 sympy/abc.py | 72 | 111| 399 | 10540 | 95339 | 
| 24 | 11 sympy/tensor/array/array_comprehension.py | 137 | 161| 195 | 10735 | 95339 | 
| 25 | 12 sympy/matrices/matrices.py | 1786 | 1815| 280 | 11015 | 114567 | 
| 26 | 12 sympy/tensor/array/array_comprehension.py | 359 | 374| 172 | 11187 | 114567 | 
| 27 | 12 sympy/tensor/array/array_comprehension.py | 65 | 80| 136 | 11323 | 114567 | 
| 28 | 12 sympy/combinatorics/tensor_can.py | 639 | 763| 1423 | 12746 | 114567 | 
| 29 | 12 sympy/tensor/array/array_comprehension.py | 82 | 107| 242 | 12988 | 114567 | 
| 30 | 12 sympy/core/symbol.py | 429 | 520| 720 | 13708 | 114567 | 
| 31 | 12 sympy/tensor/array/array_comprehension.py | 330 | 358| 231 | 13939 | 114567 | 
| 32 | 12 sympy/core/symbol.py | 671 | 764| 721 | 14660 | 114567 | 
| 33 | 12 sympy/tensor/array/array_comprehension.py | 49 | 63| 118 | 14778 | 114567 | 
| 34 | 12 sympy/tensor/array/array_comprehension.py | 274 | 295| 185 | 14963 | 114567 | 
| 35 | 12 sympy/core/symbol.py | 830 | 894| 542 | 15505 | 114567 | 
| 36 | 13 sympy/simplify/cse_main.py | 30 | 61| 293 | 15798 | 120520 | 
| 37 | 13 sympy/combinatorics/tensor_can.py | 538 | 595| 759 | 16557 | 120520 | 
| 38 | 13 sympy/core/symbol.py | 298 | 358| 460 | 17017 | 120520 | 
| 39 | 14 sympy/combinatorics/testutil.py | 202 | 277| 700 | 17717 | 123647 | 
| 40 | 14 sympy/tensor/indexed.py | 1 | 106| 872 | 18589 | 123647 | 
| 41 | 15 sympy/series/order.py | 1 | 123| 1157 | 19746 | 127724 | 
| 42 | 16 sympy/holonomic/recurrence.py | 1 | 27| 204 | 19950 | 130331 | 
| 43 | 16 sympy/core/symbol.py | 181 | 227| 231 | 20181 | 130331 | 
| 44 | 17 sympy/utilities/iterables.py | 643 | 703| 359 | 20540 | 152158 | 
| 45 | 17 sympy/core/symbol.py | 522 | 553| 287 | 20827 | 152158 | 
| 46 | 17 sympy/tensor/array/array_comprehension.py | 36 | 47| 158 | 20985 | 152158 | 
| 47 | 18 sympy/solvers/solvers.py | 2960 | 3024| 508 | 21493 | 183849 | 


## Patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -6,11 +6,11 @@
 from sympy.core.function import Lambda
 from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
 from sympy.core.numbers import oo
-from sympy.core.relational import Eq
+from sympy.core.relational import Eq, is_eq
 from sympy.core.singleton import Singleton, S
 from sympy.core.symbol import Dummy, symbols, Symbol
 from sympy.core.sympify import _sympify, sympify, converter
-from sympy.logic.boolalg import And
+from sympy.logic.boolalg import And, Or
 from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
     ProductSet)
 from sympy.utilities.misc import filldedent
@@ -571,7 +571,7 @@ class Range(Set):
         >>> r.inf
         n
         >>> pprint(r)
-        {n, n + 3, ..., n + 17}
+        {n, n + 3, ..., n + 18}
     """
 
     is_iterable = True
@@ -598,6 +598,8 @@ def __new__(cls, *args):
                         w.has(Symbol) and w.is_integer != False):
                     ok.append(w)
                 elif not w.is_Integer:
+                    if w.is_infinite:
+                        raise ValueError('infinite symbols not allowed')
                     raise ValueError
                 else:
                     ok.append(w)
@@ -610,10 +612,25 @@ def __new__(cls, *args):
 
         null = False
         if any(i.has(Symbol) for i in (start, stop, step)):
-            if start == stop:
+            dif = stop - start
+            n = dif/step
+            if n.is_Rational:
+                from sympy import floor
+                if dif == 0:
+                    null = True
+                else:  # (x, x + 5, 2) or (x, 3*x, x)
+                    n = floor(n)
+                    end = start + n*step
+                    if dif.is_Rational:  # (x, x + 5, 2)
+                        if (end - stop).is_negative:
+                            end += step
+                    else:  # (x, 3*x, x)
+                        if (end/stop - 1).is_negative:
+                            end += step
+            elif n.is_extended_negative:
                 null = True
             else:
-                end = stop
+                end = stop  # other methods like sup and reversed must fail
         elif start.is_infinite:
             span = step*(stop - start)
             if span is S.NaN or span <= 0:
@@ -631,8 +648,8 @@ def __new__(cls, *args):
             if n <= 0:
                 null = True
             elif oostep:
-                end = start + 1
-                step = S.One  # make it a canonical single step
+                step = S.One  # make it canonical
+                end = start + step
             else:
                 end = start + n*step
         if null:
@@ -656,34 +673,42 @@ def reversed(self):
         Range(9, -1, -1)
         """
         if self.has(Symbol):
-            _ = self.size  # validate
-        if not self:
+            n = (self.stop - self.start)/self.step
+            if not n.is_extended_positive or not all(
+                    i.is_integer or i.is_infinite for i in self.args):
+                raise ValueError('invalid method for symbolic range')
+        if self.start == self.stop:
             return self
         return self.func(
             self.stop - self.step, self.start - self.step, -self.step)
 
     def _contains(self, other):
-        if not self:
+        if self.start == self.stop:
             return S.false
         if other.is_infinite:
             return S.false
         if not other.is_integer:
             return other.is_integer
         if self.has(Symbol):
-            try:
-                _ = self.size  # validate
-            except ValueError:
+            n = (self.stop - self.start)/self.step
+            if not n.is_extended_positive or not all(
+                    i.is_integer or i.is_infinite for i in self.args):
                 return
+        else:
+            n = self.size
         if self.start.is_finite:
             ref = self.start
         elif self.stop.is_finite:
             ref = self.stop
         else:  # both infinite; step is +/- 1 (enforced by __new__)
             return S.true
-        if self.size == 1:
+        if n == 1:
             return Eq(other, self[0])
         res = (ref - other) % self.step
         if res == S.Zero:
+            if self.has(Symbol):
+                d = Dummy('i')
+                return self.as_relational(d).subs(d, other)
             return And(other >= self.inf, other <= self.sup)
         elif res.is_Integer:  # off sequence
             return S.false
@@ -691,20 +716,19 @@ def _contains(self, other):
             return None
 
     def __iter__(self):
-        if self.has(Symbol):
-            _ = self.size  # validate
+        n = self.size  # validate
         if self.start in [S.NegativeInfinity, S.Infinity]:
             raise TypeError("Cannot iterate over Range with infinite start")
-        elif self:
+        elif self.start != self.stop:
             i = self.start
-            step = self.step
-
-            while True:
-                if (step > 0 and not (self.start <= i < self.stop)) or \
-                   (step < 0 and not (self.stop < i <= self.start)):
-                    break
-                yield i
-                i += step
+            if n.is_infinite:
+                while True:
+                    yield i
+                    i += self.step
+            else:
+                for j in range(n):
+                    yield i
+                    i += self.step
 
     def __len__(self):
         rv = self.size
@@ -714,15 +738,15 @@ def __len__(self):
 
     @property
     def size(self):
-        if not self:
+        if self.start == self.stop:
             return S.Zero
         dif = self.stop - self.start
-        n = abs(dif // self.step)
-        if not n.is_Integer:
-            if n.is_infinite:
-                return S.Infinity
+        n = dif/self.step
+        if n.is_infinite:
+            return S.Infinity
+        if not n.is_Integer or not all(i.is_integer for i in self.args):
             raise ValueError('invalid method for symbolic range')
-        return n
+        return abs(n)
 
     @property
     def is_finite_set(self):
@@ -731,7 +755,13 @@ def is_finite_set(self):
         return self.size.is_finite
 
     def __bool__(self):
-        return self.start != self.stop
+        # this only distinguishes between definite null range
+        # and non-null/unknown null; getting True doesn't mean
+        # that it actually is not null
+        b = is_eq(self.start, self.stop)
+        if b is None:
+            raise ValueError('cannot tell if Range is null or not')
+        return not bool(b)
 
     def __getitem__(self, i):
         from sympy.functions.elementary.integers import ceiling
@@ -745,6 +775,8 @@ def __getitem__(self, i):
             "with an infinite value"
         if isinstance(i, slice):
             if self.size.is_finite:  # validates, too
+                if self.start == self.stop:
+                    return Range(0)
                 start, stop, step = i.indices(self.size)
                 n = ceiling((stop - start)/step)
                 if n <= 0:
@@ -845,44 +877,40 @@ def __getitem__(self, i):
                 elif start > 0:
                     raise ValueError(ooslice)
         else:
-            if not self:
+            if self.start == self.stop:
                 raise IndexError('Range index out of range')
+            if not (all(i.is_integer or i.is_infinite
+                    for i in self.args) and ((self.stop - self.start)/
+                    self.step).is_extended_positive):
+                raise ValueError('invalid method for symbolic range')
             if i == 0:
                 if self.start.is_infinite:
                     raise ValueError(ooslice)
-                if self.has(Symbol):
-                    if (self.stop > self.start) == self.step.is_positive and self.step.is_positive is not None:
-                        pass
-                    else:
-                        _ = self.size  # validate
                 return self.start
             if i == -1:
                 if self.stop.is_infinite:
                     raise ValueError(ooslice)
-                n = self.stop - self.step
-                if n.is_Integer or (
-                        n.is_integer and (
-                            (n - self.start).is_nonnegative ==
-                            self.step.is_positive)):
-                    return n
-            _ = self.size  # validate
+                return self.stop - self.step
+            n = self.size  # must be known for any other index
             rv = (self.stop if i < 0 else self.start) + i*self.step
             if rv.is_infinite:
                 raise ValueError(ooslice)
-            if rv < self.inf or rv > self.sup:
-                raise IndexError("Range index out of range")
-            return rv
+            if 0 <= (rv - self.start)/self.step <= n:
+                return rv
+            raise IndexError("Range index out of range")
 
     @property
     def _inf(self):
         if not self:
-            raise NotImplementedError
+            return S.EmptySet.inf
         if self.has(Symbol):
-            if self.step.is_positive:
-                return self[0]
-            elif self.step.is_negative:
-                return self[-1]
-            _ = self.size  # validate
+            if all(i.is_integer or i.is_infinite for i in self.args):
+                dif = self.stop - self.start
+                if self.step.is_positive and dif.is_positive:
+                    return self.start
+                elif self.step.is_negative and dif.is_negative:
+                    return self.stop - self.step
+            raise ValueError('invalid method for symbolic range')
         if self.step > 0:
             return self.start
         else:
@@ -891,13 +919,15 @@ def _inf(self):
     @property
     def _sup(self):
         if not self:
-            raise NotImplementedError
+            return S.EmptySet.sup
         if self.has(Symbol):
-            if self.step.is_positive:
-                return self[-1]
-            elif self.step.is_negative:
-                return self[0]
-            _ = self.size  # validate
+            if all(i.is_integer or i.is_infinite for i in self.args):
+                dif = self.stop - self.start
+                if self.step.is_positive and dif.is_positive:
+                    return self.stop - self.step
+                elif self.step.is_negative and dif.is_negative:
+                    return self.start
+            raise ValueError('invalid method for symbolic range')
         if self.step > 0:
             return self.stop - self.step
         else:
@@ -909,27 +939,37 @@ def _boundary(self):
 
     def as_relational(self, x):
         """Rewrite a Range in terms of equalities and logic operators. """
-        if self.size == 1:
-            return Eq(x, self[0])
-        elif self.size == 0:
-            return S.false
+        from sympy.core.mod import Mod
+        if self.start.is_infinite:
+            assert not self.stop.is_infinite  # by instantiation
+            a = self.reversed.start
         else:
-            from sympy.core.mod import Mod
-            cond = None
-            if self.start.is_infinite:
-                if self.stop.is_infinite:
-                    cond = S.true
-                else:
-                    a = self.reversed.start
-            elif self.start == self.stop:
-                cond = S.false  # null range
-            else:
-                a = self.start
-            step = abs(self.step)
-            cond = Eq(Mod(x, step), a % step) if cond is None else cond
-            return And(cond,
-                       x >= self.inf if self.inf in self else x > self.inf,
-                       x <= self.sup if self.sup in self else x < self.sup)
+            a = self.start
+        step = self.step
+        in_seq = Eq(Mod(x - a, step), 0)
+        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
+        n = (self.stop - self.start)/self.step
+        if n == 0:
+            return S.EmptySet.as_relational(x)
+        if n == 1:
+            return And(Eq(x, a), ints)
+        try:
+            a, b = self.inf, self.sup
+        except ValueError:
+            a = None
+        if a is not None:
+            range_cond = And(
+                x > a if a.is_infinite else x >= a,
+                x < b if b.is_infinite else x <= b)
+        else:
+            a, b = self.start, self.stop - self.step
+            range_cond = Or(
+                And(self.step >= 1, x > a if a.is_infinite else x >= a,
+                x < b if b.is_infinite else x <= b),
+                And(self.step <= -1, x < a if a.is_infinite else x <= a,
+                x > b if b.is_infinite else x >= b))
+        return And(in_seq, ints, range_cond)
+
 
 converter[range] = lambda r: Range(r.start, r.stop, r.step)
 

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_fancysets.py b/sympy/sets/tests/test_fancysets.py
--- a/sympy/sets/tests/test_fancysets.py
+++ b/sympy/sets/tests/test_fancysets.py
@@ -9,7 +9,7 @@
                    Dummy, floor, And, Eq)
 from sympy.utilities.iterables import cartes
 from sympy.testing.pytest import XFAIL, raises
-from sympy.abc import x, y, t
+from sympy.abc import x, y, t, z
 from sympy.core.mod import Mod
 
 import itertools
@@ -174,8 +174,6 @@ def test_inf_Range_len():
     assert Range(0, -oo, -2).size is S.Infinity
     assert Range(oo, 0, -2).size is S.Infinity
     assert Range(-oo, 0, 2).size is S.Infinity
-    i = Symbol('i', integer=True)
-    assert Range(0, 4 * i, i).size == 4
 
 
 def test_Range_set():
@@ -209,6 +207,9 @@ def test_Range_set():
     assert Range(1, oo, -1) == empty
     assert Range(1, -oo, 1) == empty
     assert Range(1, -4, oo) == empty
+    ip = symbols('ip', positive=True)
+    assert Range(0, ip, -1) == empty
+    assert Range(0, -ip, 1) == empty
     assert Range(1, -4, -oo) == Range(1, 2)
     assert Range(1, 4, oo) == Range(1, 2)
     assert Range(-oo, oo).size == oo
@@ -231,13 +232,8 @@ def test_Range_set():
     assert Range(-oo, 1, 1)[-1] is S.Zero
     assert Range(oo, 1, -1)[-1] == 2
     assert inf not in Range(oo)
-    inf = symbols('inf', infinite=True)
-    assert inf not in Range(oo)
-    assert Range(-oo, 1, 1)[-1] is S.Zero
-    assert Range(oo, 1, -1)[-1] == 2
     assert Range(1, 10, 1)[-1] == 9
     assert all(i.is_Integer for i in Range(0, -1, 1))
-
     it = iter(Range(-oo, 0, 2))
     raises(TypeError, lambda: next(it))
 
@@ -278,6 +274,7 @@ def test_Range_set():
     raises(ValueError, lambda: Range(-oo, 4, 2)[2::-1])
     assert Range(-oo, 4, 2)[-2::2] == Range(0, 4, 4)
     assert Range(oo, 0, -2)[-10:0:2] == empty
+    raises(ValueError, lambda: Range(oo, 0, -2)[0])
     raises(ValueError, lambda: Range(oo, 0, -2)[-10:10:2])
     raises(ValueError, lambda: Range(oo, 0, -2)[0::-2])
     assert Range(oo, 0, -2)[0:-4:-2] == empty
@@ -297,6 +294,7 @@ def test_Range_set():
     assert empty[:0] == empty
     raises(NotImplementedError, lambda: empty.inf)
     raises(NotImplementedError, lambda: empty.sup)
+    assert empty.as_relational(x) is S.false
 
     AB = [None] + list(range(12))
     for R in [
@@ -330,45 +328,91 @@ def test_Range_set():
 
     # test Range.as_relational
     assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3)  & Eq(Mod(x, 1), 0)
-    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo)  & Eq(Mod(x, 2), 1)
+    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo)  & Eq(Mod(x + 1, -2), 0)
 
 
 def test_Range_symbolic():
     # symbolic Range
+    xr = Range(x, x + 4, 5)
     sr = Range(x, y, t)
     i = Symbol('i', integer=True)
     ip = Symbol('i', integer=True, positive=True)
-    ir = Range(i, i + 20, 2)
+    ipr = Range(ip)
+    inr = Range(0, -ip, -1)
+    ir = Range(i, i + 19, 2)
+    ir2 = Range(i, i*8, 3*i)
+    i = Symbol('i', integer=True)
     inf = symbols('inf', infinite=True)
+    raises(ValueError, lambda: Range(inf))
+    raises(ValueError, lambda: Range(inf, 0, -1))
+    raises(ValueError, lambda: Range(inf, inf, 1))
+    raises(ValueError, lambda: Range(1, 1, inf))
     # args
+    assert xr.args == (x, x + 5, 5)
     assert sr.args == (x, y, t)
     assert ir.args == (i, i + 20, 2)
+    assert ir2.args == (i, 10*i, 3*i)
     # reversed
+    raises(ValueError, lambda: xr.reversed)
     raises(ValueError, lambda: sr.reversed)
-    assert ir.reversed == Range(i + 18, i - 2, -2)
+    assert ipr.reversed.args == (ip - 1, -1, -1)
+    assert inr.reversed.args == (-ip + 1, 1, 1)
+    assert ir.reversed.args == (i + 18, i - 2, -2)
+    assert ir2.reversed.args == (7*i, -2*i, -3*i)
     # contains
     assert inf not in sr
     assert inf not in ir
+    assert 0 in ipr
+    assert 0 in inr
+    raises(TypeError, lambda: 1 in ipr)
+    raises(TypeError, lambda: -1 in inr)
     assert .1 not in sr
     assert .1 not in ir
     assert i + 1 not in ir
     assert i + 2 in ir
+    raises(TypeError, lambda: x in xr)  # XXX is this what contains is supposed to do?
     raises(TypeError, lambda: 1 in sr)  # XXX is this what contains is supposed to do?
     # iter
+    raises(ValueError, lambda: next(iter(xr)))
     raises(ValueError, lambda: next(iter(sr)))
     assert next(iter(ir)) == i
+    assert next(iter(ir2)) == i
     assert sr.intersect(S.Integers) == sr
     assert sr.intersect(FiniteSet(x)) == Intersection({x}, sr)
     raises(ValueError, lambda: sr[:2])
+    raises(ValueError, lambda: xr[0])
     raises(ValueError, lambda: sr[0])
-    raises(ValueError, lambda: sr.as_relational(x))
     # len
     assert len(ir) == ir.size == 10
+    assert len(ir2) == ir2.size == 3
+    raises(ValueError, lambda: len(xr))
+    raises(ValueError, lambda: xr.size)
     raises(ValueError, lambda: len(sr))
     raises(ValueError, lambda: sr.size)
     # bool
-    assert bool(ir) == bool(sr) == True
+    assert bool(Range(0)) == False
+    assert bool(xr)
+    assert bool(ir)
+    assert bool(ipr)
+    assert bool(inr)
+    raises(ValueError, lambda: bool(sr))
+    raises(ValueError, lambda: bool(ir2))
+    # inf
+    raises(ValueError, lambda: xr.inf)
+    raises(ValueError, lambda: sr.inf)
+    assert ipr.inf == 0
+    assert inr.inf == -ip + 1
+    assert ir.inf == i
+    raises(ValueError, lambda: ir2.inf)
+    # sup
+    raises(ValueError, lambda: xr.sup)
+    raises(ValueError, lambda: sr.sup)
+    assert ipr.sup == ip - 1
+    assert inr.sup == 0
+    assert ir.inf == i
+    raises(ValueError, lambda: ir2.sup)
     # getitem
+    raises(ValueError, lambda: xr[0])
     raises(ValueError, lambda: sr[0])
     raises(ValueError, lambda: sr[-1])
     raises(ValueError, lambda: sr[:2])
@@ -376,17 +420,33 @@ def test_Range_symbolic():
     assert ir[0] == i
     assert ir[-2] == i + 16
     assert ir[-1] == i + 18
+    assert ir2[:2] == Range(i, 7*i, 3*i)
+    assert ir2[0] == i
+    assert ir2[-2] == 4*i
+    assert ir2[-1] == 7*i
     raises(ValueError, lambda: Range(i)[-1])
-    assert Range(ip)[-1] == ip - 1
+    assert ipr[0] == ipr.inf == 0
+    assert ipr[-1] == ipr.sup == ip - 1
+    assert inr[0] == inr.sup == 0
+    assert inr[-1] == inr.inf == -ip + 1
+    raises(ValueError, lambda: ipr[-2])
     assert ir.inf == i
     assert ir.sup == i + 18
-    assert Range(ip).inf == 0
-    assert Range(ip).sup == ip - 1
     raises(ValueError, lambda: Range(i).inf)
     # as_relational
-    raises(ValueError, lambda: sr.as_relational(x))
-    assert ir.as_relational(x) == (x >= i) & (x <= i + 18) & Eq(Mod(x, 2), Mod(i, 2))
+    assert ir.as_relational(x) == ((x >= i) & (x <= i + 18) &
+        Eq(Mod(-i + x, 2), 0))
+    assert ir2.as_relational(x) == Eq(
+        Mod(-i + x, 3*i), 0) & (((x >= i) & (x <= 7*i) & (3*i >= 1)) |
+        ((x <= i) & (x >= 7*i) & (3*i <= -1)))
     assert Range(i, i + 1).as_relational(x) == Eq(x, i)
+    assert sr.as_relational(z) == Eq(
+        Mod(t, 1), 0) & Eq(Mod(x, 1), 0) & Eq(Mod(-x + z, t), 0
+        ) & (((z >= x) & (z <= -t + y) & (t >= 1)) |
+        ((z <= x) & (z >= -t + y) & (t <= -1)))
+    assert xr.as_relational(z) == Eq(z, x) & Eq(Mod(x, 1), 0)
+    # symbols can clash if user wants (but it must be integer)
+    assert xr.as_relational(x) == Eq(Mod(x, 1), 0)
     # contains() for symbolic values (issue #18146)
     e = Symbol('e', integer=True, even=True)
     o = Symbol('o', integer=True, odd=True)

```


## Code snippets

### 1 - sympy/sets/fancysets.py:

Start line: 494, End line: 577

```python
class Range(Set):
    """
    Represents a range of integers. Can be called as Range(stop),
    Range(start, stop), or Range(start, stop, step); when step is
    not given it defaults to 1.

    `Range(stop)` is the same as `Range(0, stop, 1)` and the stop value
    (juse as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although Range is a set (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where `range` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 17}
    """

    is_iterable = True
```
### 2 - sympy/sets/fancysets.py:

Start line: 746, End line: 874

```python
class Range(Set):

    def __getitem__(self, i):
        # ... other code
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite Range
                #   i.e. Range(-oo, oo) or Range(oo, -oo, -1)
                # --------------------
                if self.start.is_infinite and self.stop.is_infinite:
                    raise ValueError(infinite)
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if not self:
                raise IndexError('Range index out of range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                if self.has(Symbol):
                    if (self.stop > self.start) == self.step.is_positive and self.step.is_positive is not None:
                        pass
                    else:
                        _ = self.size  # validate
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                n = self.stop - self.step
                if n.is_Integer or (
                        n.is_integer and (
                            (n - self.start).is_nonnegative ==
                            self.step.is_positive)):
                    return n
            _ = self.size  # validate
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            if rv < self.inf or rv > self.sup:
                raise IndexError("Range index out of range")
            return rv
```
### 3 - sympy/sets/fancysets.py:

Start line: 643, End line: 663

```python
class Range(Set):

    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        if self.has(Symbol):
            _ = self.size  # validate
        if not self:
            return self
        return self.func(
            self.stop - self.step, self.start - self.step, -self.step)
```
### 4 - sympy/sets/fancysets.py:

Start line: 910, End line: 934

```python
class Range(Set):

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        if self.size == 1:
            return Eq(x, self[0])
        elif self.size == 0:
            return S.false
        else:
            from sympy.core.mod import Mod
            cond = None
            if self.start.is_infinite:
                if self.stop.is_infinite:
                    cond = S.true
                else:
                    a = self.reversed.start
            elif self.start == self.stop:
                cond = S.false  # null range
            else:
                a = self.start
            step = abs(self.step)
            cond = Eq(Mod(x, step), a % step) if cond is None else cond
            return And(cond,
                       x >= self.inf if self.inf in self else x > self.inf,
                       x <= self.sup if self.sup in self else x < self.sup)

converter[range] = lambda r: Range(r.start, r.stop, r.step)
```
### 5 - sympy/sets/fancysets.py:

Start line: 579, End line: 641

```python
class Range(Set):

    def __new__(cls, *args):
        from sympy.functions.elementary.integers import ceiling
        if len(args) == 1:
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        try:
            ok = []
            for w in (start, stop, step):
                w = sympify(w)
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                elif not w.is_Integer:
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            raise ValueError(filldedent('''
            rguments to Range must be integers; `imageset` can define
            ses, e.g. use `imageset(i, i/10, Range(3))` to give
            , 1/5].'''))
        start, stop, step = ok

        null = False
        if any(i.has(Symbol) for i in (start, stop, step)):
            if start == stop:
                null = True
            else:
                end = stop
        elif start.is_infinite:
            span = step*(stop - start)
            if span is S.NaN or span <= 0:
                null = True
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            n = ceiling((stop - start)/step)
            if n <= 0:
                null = True
            elif oostep:
                end = start + 1
                step = S.One  # make it a canonical single step
            else:
                end = start + n*step
        if null:
            start = end = S.Zero
            step = S.One
        return Basic.__new__(cls, start, end, step)
```
### 6 - sympy/sets/fancysets.py:

Start line: 693, End line: 734

```python
class Range(Set):

    def __iter__(self):
        if self.has(Symbol):
            _ = self.size  # validate
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise TypeError("Cannot iterate over Range with infinite start")
        elif self:
            i = self.start
            step = self.step

            while True:
                if (step > 0 and not (self.start <= i < self.stop)) or \
                   (step < 0 and not (self.stop < i <= self.start)):
                    break
                yield i
                i += step

    def __len__(self):
        rv = self.size
        if rv is S.Infinity:
            raise ValueError('Use .size to get the length of an infinite Range')
        return int(rv)

    @property
    def size(self):
        if not self:
            return S.Zero
        dif = self.stop - self.start
        n = abs(dif // self.step)
        if not n.is_Integer:
            if n.is_infinite:
                return S.Infinity
            raise ValueError('invalid method for symbolic range')
        return n

    @property
    def is_finite_set(self):
        if self.start.is_integer and self.stop.is_integer:
            return True
        return self.size.is_finite

    def __bool__(self):
        return self.start != self.stop
```
### 7 - sympy/sets/fancysets.py:

Start line: 736, End line: 745

```python
class Range(Set):

    def __getitem__(self, i):
        from sympy.functions.elementary.integers import ceiling
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        infinite = "slicing not possible on range with infinite start"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        # ... other code
```
### 8 - sympy/calculus/util.py:

Start line: 173, End line: 222

```python
def function_range(f, symbol, domain):
    # ... other code

    for interval in interval_iter:
        if isinstance(interval, FiniteSet):
            for singleton in interval:
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        elif isinstance(interval, Interval):
            vals = S.EmptySet
            critical_points = S.EmptySet
            critical_values = S.EmptySet
            bounds = ((interval.left_open, interval.inf, '+'),
                   (interval.right_open, interval.sup, '-'))

            for is_open, limit_point, direction in bounds:
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point, direction))
                    vals += critical_values

                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))

            solution = solveset(f.diff(symbol), symbol, interval)

            if not iterable(solution):
                raise NotImplementedError(
                        'Unable to find critical points for {}'.format(f))
            if isinstance(solution, ImageSet):
                raise NotImplementedError(
                        'Infinite number of critical points for {}'.format(f))

            critical_points += solution

            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))

            left_open, right_open = False, False

            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True

                if critical_values.sup == vals.sup:
                    right_open = True

            range_int += Interval(vals.inf, vals.sup, left_open, right_open)
        else:
            raise NotImplementedError(filldedent('''
                Unable to find range for the given domain.
                '''))

    return range_int
```
### 9 - sympy/sets/fancysets.py:

Start line: 665, End line: 691

```python
class Range(Set):

    def _contains(self, other):
        if not self:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return other.is_integer
        if self.has(Symbol):
            try:
                _ = self.size  # validate
            except ValueError:
                return
        if self.start.is_finite:
            ref = self.start
        elif self.stop.is_finite:
            ref = self.stop
        else:  # both infinite; step is +/- 1 (enforced by __new__)
            return S.true
        if self.size == 1:
            return Eq(other, self[0])
        res = (ref - other) % self.step
        if res == S.Zero:
            return And(other >= self.inf, other <= self.sup)
        elif res.is_Integer:  # off sequence
            return S.false
        else:  # symbolic/unsimplified residue modulo step
            return None
```
### 10 - sympy/calculus/util.py:

Start line: 89, End line: 171

```python
def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Parameters
    ==========

    f : Expr
        The concerned function.
    symbol : Symbol
        The variable for which the range of function is to be determined.
    domain : Interval
        The domain under which the range of the function has to be found.

    Examples
    ========

    >>> from sympy import Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.sets import Interval
    >>> from sympy.calculus.util import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    Interval(-1, 1)
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    Interval(-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> function_range(exp(x), x, S.Reals)
    Interval.open(0, oo)
    >>> function_range(log(x), x, S.Reals)
    Interval(-oo, oo)
    >>> function_range(sqrt(x), x , Interval(-5, 9))
    Interval(0, 3)

    Returns
    =======

    Interval
        Union of all ranges for all intervals under domain where function is
        continuous.

    Raises
    ======

    NotImplementedError
        If any of the intervals, in the given domain, for which function
        is continuous are not finite or real,
        OR if the critical points of the function on the domain can't be found.
    """
    from sympy.solvers.solveset import solveset

    if isinstance(domain, EmptySet):
        return S.EmptySet

    period = periodicity(f, symbol)
    if period == S.Zero:
        # the expression is constant wrt symbol
        return FiniteSet(f.expand())

    if period is not None:
        if isinstance(domain, Interval):
            if (domain.inf - domain.sup).is_infinite:
                domain = Interval(0, period)
        elif isinstance(domain, Union):
            for sub_dom in domain.args:
                if isinstance(sub_dom, Interval) and \
                ((sub_dom.inf - sub_dom.sup).is_infinite):
                    domain = Interval(0, period)

    intervals = continuous_domain(f, symbol, domain)
    range_int = S.EmptySet
    if isinstance(intervals,(Interval, FiniteSet)):
        interval_iter = (intervals,)

    elif isinstance(intervals, Union):
        interval_iter = intervals.args

    else:
            raise NotImplementedError(filldedent('''
                Unable to find range for the given domain.
                '''))
    # ... other code
```
### 11 - sympy/sets/fancysets.py:

Start line: 876, End line: 908

```python
class Range(Set):

    @property
    def _inf(self):
        if not self:
            raise NotImplementedError
        if self.has(Symbol):
            if self.step.is_positive:
                return self[0]
            elif self.step.is_negative:
                return self[-1]
            _ = self.size  # validate
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step

    @property
    def _sup(self):
        if not self:
            raise NotImplementedError
        if self.has(Symbol):
            if self.step.is_positive:
                return self[-1]
            elif self.step.is_negative:
                return self[0]
            _ = self.size  # validate
        if self.step > 0:
            return self.stop - self.step
        else:
            return self.start

    @property
    def _boundary(self):
        return self
```
