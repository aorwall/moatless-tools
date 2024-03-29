# sympy__sympy-18137

| **sympy/sympy** | `0bffa281e62b4d29fbe3cd22faa4d612a4b1ca76` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 468 |
| **Any found context length** | 146 |
| **Avg pos** | 44.0 |
| **Min pos** | 1 |
| **Max pos** | 41 |
| **Top file pos** | 1 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -683,11 +683,17 @@ def _contains(self, other):
             ref = self.start
         elif self.stop.is_finite:
             ref = self.stop
-        else:
-            return other.is_Integer
-        if (ref - other) % self.step:  # off sequence
+        else:  # both infinite; step is +/- 1 (enforced by __new__)
+            return S.true
+        if self.size == 1:
+            return Eq(other, self[0])
+        res = (ref - other) % self.step
+        if res == S.Zero:
+            return And(other >= self.inf, other <= self.sup)
+        elif res.is_Integer:  # off sequence
             return S.false
-        return _sympify(other >= self.inf and other <= self.sup)
+        else:  # symbolic/unsimplified residue modulo step
+            return None
 
     def __iter__(self):
         if self.has(Symbol):
@@ -899,10 +905,13 @@ def _boundary(self):
     def as_relational(self, x):
         """Rewrite a Range in terms of equalities and logic operators. """
         from sympy.functions.elementary.integers import floor
-        return And(
-            Eq(x, floor(x)),
-            x >= self.inf if self.inf in self else x > self.inf,
-            x <= self.sup if self.sup in self else x < self.sup)
+        if self.size == 1:
+            return Eq(x, self[0])
+        else:
+            return And(
+                Eq(x, floor(x)),
+                x >= self.inf if self.inf in self else x > self.inf,
+                x <= self.sup if self.sup in self else x < self.sup)
 
 
 # Using range from compatibility above (xrange on Py2)
diff --git a/sympy/sets/handlers/issubset.py b/sympy/sets/handlers/issubset.py
--- a/sympy/sets/handlers/issubset.py
+++ b/sympy/sets/handlers/issubset.py
@@ -1,4 +1,4 @@
-from sympy import S
+from sympy import S, Symbol
 from sympy.core.logic import fuzzy_and, fuzzy_bool, fuzzy_not, fuzzy_or
 from sympy.core.relational import Eq
 from sympy.sets.sets import FiniteSet, Interval, Set, Union
@@ -66,6 +66,40 @@ def is_subset_sets(a_range, b_interval): # noqa:F811
             cond_right = a_range.sup <= b_interval.right
         return fuzzy_and([cond_left, cond_right])
 
+@dispatch(Range, FiniteSet)
+def is_subset_sets(a_range, b_finiteset): # noqa:F811
+    try:
+        a_size = a_range.size
+    except ValueError:
+        # symbolic Range of unknown size
+        return None
+    if a_size > len(b_finiteset):
+        return False
+    elif any(arg.has(Symbol) for arg in a_range.args):
+        return fuzzy_and(b_finiteset.contains(x) for x in a_range)
+    else:
+        # Checking A \ B == EmptySet is more efficient than repeated naive
+        # membership checks on an arbitrary FiniteSet.
+        a_set = set(a_range)
+        b_remaining = len(b_finiteset)
+        # Symbolic expressions and numbers of unknown type (integer or not) are
+        # all counted as "candidates", i.e. *potentially* matching some a in
+        # a_range.
+        cnt_candidate = 0
+        for b in b_finiteset:
+            if b.is_Integer:
+                a_set.discard(b)
+            elif fuzzy_not(b.is_integer):
+                pass
+            else:
+                cnt_candidate += 1
+            b_remaining -= 1
+            if len(a_set) > b_remaining + cnt_candidate:
+                return False
+            if len(a_set) == 0:
+                return True
+        return None
+
 @dispatch(Interval, Range)
 def is_subset_sets(a_interval, b_range): # noqa:F811
     if a_interval.measure.is_extended_nonzero:
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -1761,8 +1761,10 @@ def __new__(cls, *args, **kwargs):
         else:
             args = list(map(sympify, args))
 
-        args = list(ordered(set(args), Set._infimum_key))
+        _args_set = set(args)
+        args = list(ordered(_args_set, Set._infimum_key))
         obj = Basic.__new__(cls, *args)
+        obj._args_set = _args_set
         return obj
 
     def _eval_Eq(self, other):
@@ -1830,8 +1832,9 @@ def _contains(self, other):
         """
         Tests whether an element, other, is in the set.
 
-        Relies on Python's set class. This tests for object equality
-        All inputs are sympified
+        The actual test is for mathematical equality (as opposed to
+        syntactical equality). In the worst case all elements of the
+        set must be checked.
 
         Examples
         ========
@@ -1843,9 +1846,13 @@ def _contains(self, other):
         False
 
         """
-        # evaluate=True is needed to override evaluate=False context;
-        # we need Eq to do the evaluation
-        return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True)) for e in self.args)
+        if other in self._args_set:
+            return True
+        else:
+            # evaluate=True is needed to override evaluate=False context;
+            # we need Eq to do the evaluation
+            return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True))
+                for e in self.args)
 
     def _eval_is_subset(self, other):
         return fuzzy_and(other._contains(e) for e in self.args)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/fancysets.py | 686 | 690 | 1 | 1 | 146
| sympy/sets/fancysets.py | 902 | 905 | 2 | 1 | 468
| sympy/sets/handlers/issubset.py | 1 | 1 | 21 | 6 | 12980
| sympy/sets/handlers/issubset.py | 69 | 69 | 26 | 6 | 14978
| sympy/sets/sets.py | 1764 | 1764 | - | 3 | -
| sympy/sets/sets.py | 1833 | 1834 | 41 | 3 | 19409
| sympy/sets/sets.py | 1846 | 1848 | 41 | 3 | 19409


## Problem Statement

```
Range(1).intersect(FiniteSet(n)) raises TypeError: cannot determine truth value of Relational
\`\`\`
n = Symbol('n', integer=True)
Range(1).intersect(FiniteSet(n))
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-66-74dcb9ca2d9f> in <module>
----> 1 Range(1).intersect(FiniteSet(n))

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in intersect(self, other)
    138 
    139         """
--> 140         return Intersection(self, other)
    141 
    142     def intersection(self, other):

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in __new__(cls, *args, **kwargs)
   1310         if evaluate:
   1311             args = list(cls._new_args_filter(args))
-> 1312             return simplify_intersection(args)
   1313 
   1314         args = list(ordered(args, Set._infimum_key))

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in simplify_intersection(args)
   2176 
   2177     # Handle Finite sets
-> 2178     rv = Intersection._handle_finite_sets(args)
   2179 
   2180     if rv is not None:

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in _handle_finite_sets(args)
   1395         definite = set()
   1396         for e in all_elements:
-> 1397             inall = fuzzy_and(s.contains(e) for s in args)
   1398             if inall is True:
   1399                 definite.add(e)

/opt/tljh/user/lib/python3.6/site-packages/sympy/core/logic.py in fuzzy_and(args)
    137 
    138     rv = True
--> 139     for ai in args:
    140         ai = fuzzy_bool(ai)
    141         if ai is False:

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in <genexpr>(.0)
   1395         definite = set()
   1396         for e in all_elements:
-> 1397             inall = fuzzy_and(s.contains(e) for s in args)
   1398             if inall is True:
   1399                 definite.add(e)

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/sets.py in contains(self, other)
    332         """
    333         other = sympify(other, strict=True)
--> 334         c = self._contains(other)
    335         if c is None:
    336             return Contains(other, self, evaluate=False)

/opt/tljh/user/lib/python3.6/site-packages/sympy/sets/fancysets.py in _contains(self, other)
    668         if (ref - other) % self.step:  # off sequence
    669             return S.false
--> 670         return _sympify(other >= self.inf and other <= self.sup)
    671 
    672     def __iter__(self):

/opt/tljh/user/lib/python3.6/site-packages/sympy/core/relational.py in __nonzero__(self)
    374 
    375     def __nonzero__(self):
--> 376         raise TypeError("cannot determine truth value of Relational")
    377 
    378     __bool__ = __nonzero__

TypeError: cannot determine truth value of Relational
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/sets/fancysets.py** | 670 | 690| 146 | 146 | 10652 | 
| **-> 2 <-** | **1 sympy/sets/fancysets.py** | 865 | 912| 322 | 468 | 10652 | 
| 3 | **1 sympy/sets/fancysets.py** | 692 | 730| 283 | 751 | 10652 | 
| 4 | 2 sympy/core/relational.py | 370 | 394| 195 | 946 | 19406 | 
| 5 | 2 sympy/core/relational.py | 479 | 577| 950 | 1896 | 19406 | 
| 6 | **3 sympy/sets/sets.py** | 1412 | 1514| 968 | 2864 | 35721 | 
| 7 | **3 sympy/sets/fancysets.py** | 741 | 863| 1041 | 3905 | 35721 | 
| 8 | 3 sympy/core/relational.py | 288 | 368| 756 | 4661 | 35721 | 
| 9 | **3 sympy/sets/fancysets.py** | 499 | 582| 674 | 5335 | 35721 | 
| 10 | **3 sympy/sets/sets.py** | 1049 | 1062| 122 | 5457 | 35721 | 
| 11 | **3 sympy/sets/fancysets.py** | 584 | 646| 533 | 5990 | 35721 | 
| 12 | 3 sympy/core/relational.py | 794 | 1025| 2080 | 8070 | 35721 | 
| 13 | 4 sympy/sets/handlers/intersection.py | 223 | 348| 1074 | 9144 | 39428 | 
| 14 | 4 sympy/core/relational.py | 62 | 93| 305 | 9449 | 39428 | 
| 15 | 4 sympy/sets/handlers/intersection.py | 405 | 457| 397 | 9846 | 39428 | 
| 16 | 5 sympy/solvers/inequalities.py | 481 | 679| 1588 | 11434 | 47232 | 
| 17 | 5 sympy/core/relational.py | 615 | 648| 248 | 11682 | 47232 | 
| 18 | 5 sympy/core/relational.py | 1028 | 1095| 395 | 12077 | 47232 | 
| 19 | **5 sympy/sets/sets.py** | 1790 | 1827| 337 | 12414 | 47232 | 
| 20 | 5 sympy/core/relational.py | 1 | 30| 250 | 12664 | 47232 | 
| **-> 21 <-** | **6 sympy/sets/handlers/issubset.py** | 1 | 32| 316 | 12980 | 48123 | 
| 22 | 6 sympy/core/relational.py | 456 | 477| 205 | 13185 | 48123 | 
| 23 | 7 sympy/calculus/util.py | 1466 | 1501| 300 | 13485 | 60173 | 
| 24 | 7 sympy/calculus/util.py | 1392 | 1427| 303 | 13788 | 60173 | 
| 25 | 7 sympy/sets/handlers/intersection.py | 105 | 220| 962 | 14750 | 60173 | 
| **-> 26 <-** | **7 sympy/sets/handlers/issubset.py** | 69 | 102| 228 | 14978 | 60173 | 
| 27 | 7 sympy/calculus/util.py | 1347 | 1390| 351 | 15329 | 60173 | 
| 28 | **7 sympy/sets/sets.py** | 1768 | 1788| 181 | 15510 | 60173 | 
| 29 | **7 sympy/sets/sets.py** | 285 | 327| 329 | 15839 | 60173 | 
| 30 | 7 sympy/sets/handlers/intersection.py | 30 | 75| 407 | 16246 | 60173 | 
| 31 | 7 sympy/calculus/util.py | 1429 | 1464| 299 | 16545 | 60173 | 
| 32 | **7 sympy/sets/sets.py** | 1374 | 1410| 233 | 16778 | 60173 | 
| 33 | 8 sympy/sets/conditionset.py | 1 | 18| 154 | 16932 | 62424 | 
| 34 | 8 sympy/sets/handlers/intersection.py | 77 | 103| 220 | 17152 | 62424 | 
| 35 | **8 sympy/sets/sets.py** | 1936 | 2001| 388 | 17540 | 62424 | 
| 36 | 9 sympy/logic/boolalg.py | 702 | 769| 608 | 18148 | 83711 | 
| 37 | **9 sympy/sets/sets.py** | 1064 | 1119| 423 | 18571 | 83711 | 
| 38 | **9 sympy/sets/sets.py** | 1911 | 1933| 206 | 18777 | 83711 | 
| 39 | **9 sympy/sets/fancysets.py** | 732 | 740| 137 | 18914 | 83711 | 
| 40 | 9 sympy/calculus/util.py | 192 | 241| 347 | 19261 | 83711 | 
| **-> 41 <-** | **9 sympy/sets/sets.py** | 1829 | 1848| 148 | 19409 | 83711 | 
| 42 | 10 sympy/polys/polytools.py | 3629 | 4161| 3474 | 22883 | 134386 | 
| 43 | 10 sympy/sets/handlers/intersection.py | 1 | 28| 248 | 23131 | 134386 | 
| 44 | **10 sympy/sets/sets.py** | 388 | 420| 227 | 23358 | 134386 | 
| 45 | 11 sympy/functions/elementary/piecewise.py | 694 | 719| 277 | 23635 | 145656 | 
| 46 | 12 sympy/sets/contains.py | 1 | 52| 333 | 23968 | 145989 | 
| 47 | **12 sympy/sets/sets.py** | 178 | 214| 363 | 24331 | 145989 | 
| 48 | 12 sympy/logic/boolalg.py | 1298 | 1347| 425 | 24756 | 145989 | 
| 49 | **12 sympy/sets/handlers/issubset.py** | 50 | 67| 177 | 24933 | 145989 | 
| 50 | **12 sympy/sets/sets.py** | 1850 | 1890| 271 | 25204 | 145989 | 
| 51 | 12 sympy/calculus/util.py | 1503 | 1535| 216 | 25420 | 145989 | 
| 52 | 13 sympy/sets/__init__.py | 1 | 34| 261 | 25681 | 146250 | 
| 53 | 13 sympy/core/relational.py | 152 | 187| 325 | 26006 | 146250 | 
| 54 | 14 sympy/stats/frv.py | 167 | 183| 150 | 26156 | 150020 | 
| 55 | **14 sympy/sets/sets.py** | 129 | 153| 153 | 26309 | 150020 | 
| 56 | **14 sympy/sets/fancysets.py** | 23 | 69| 331 | 26640 | 150020 | 
| 57 | 14 sympy/core/relational.py | 95 | 125| 220 | 26860 | 150020 | 
| 58 | **14 sympy/sets/fancysets.py** | 1237 | 1269| 264 | 27124 | 150020 | 
| 59 | 14 sympy/core/relational.py | 651 | 717| 469 | 27593 | 150020 | 
| 60 | **14 sympy/sets/fancysets.py** | 648 | 668| 143 | 27736 | 150020 | 
| 61 | 15 sympy/polys/agca/modules.py | 1232 | 1250| 261 | 27997 | 161855 | 
| 62 | **15 sympy/sets/sets.py** | 1349 | 1372| 134 | 28131 | 161855 | 
| 63 | 15 sympy/core/relational.py | 189 | 238| 436 | 28567 | 161855 | 
| 64 | **15 sympy/sets/sets.py** | 1517 | 1603| 510 | 29077 | 161855 | 
| 65 | **15 sympy/sets/sets.py** | 889 | 928| 372 | 29449 | 161855 | 
| 66 | 15 sympy/calculus/util.py | 1537 | 1580| 322 | 29771 | 161855 | 
| 67 | **15 sympy/sets/sets.py** | 778 | 790| 147 | 29918 | 161855 | 
| 68 | 16 sympy/core/numbers.py | 1937 | 1970| 245 | 30163 | 191690 | 
| 69 | **16 sympy/sets/fancysets.py** | 72 | 133| 406 | 30569 | 191690 | 
| 70 | **16 sympy/sets/sets.py** | 1 | 36| 323 | 30892 | 191690 | 
| 71 | 16 sympy/sets/handlers/intersection.py | 351 | 403| 397 | 31289 | 191690 | 
| 72 | 17 sympy/plotting/intervalmath/interval_membership.py | 1 | 79| 561 | 31850 | 192252 | 
| 73 | 18 sympy/concrete/expr_with_limits.py | 1 | 19| 168 | 32018 | 196641 | 
| 74 | **18 sympy/sets/fancysets.py** | 136 | 160| 167 | 32185 | 196641 | 
| 75 | 18 sympy/sets/conditionset.py | 190 | 206| 137 | 32322 | 196641 | 
| 76 | **18 sympy/sets/sets.py** | 155 | 176| 152 | 32474 | 196641 | 


## Patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -683,11 +683,17 @@ def _contains(self, other):
             ref = self.start
         elif self.stop.is_finite:
             ref = self.stop
-        else:
-            return other.is_Integer
-        if (ref - other) % self.step:  # off sequence
+        else:  # both infinite; step is +/- 1 (enforced by __new__)
+            return S.true
+        if self.size == 1:
+            return Eq(other, self[0])
+        res = (ref - other) % self.step
+        if res == S.Zero:
+            return And(other >= self.inf, other <= self.sup)
+        elif res.is_Integer:  # off sequence
             return S.false
-        return _sympify(other >= self.inf and other <= self.sup)
+        else:  # symbolic/unsimplified residue modulo step
+            return None
 
     def __iter__(self):
         if self.has(Symbol):
@@ -899,10 +905,13 @@ def _boundary(self):
     def as_relational(self, x):
         """Rewrite a Range in terms of equalities and logic operators. """
         from sympy.functions.elementary.integers import floor
-        return And(
-            Eq(x, floor(x)),
-            x >= self.inf if self.inf in self else x > self.inf,
-            x <= self.sup if self.sup in self else x < self.sup)
+        if self.size == 1:
+            return Eq(x, self[0])
+        else:
+            return And(
+                Eq(x, floor(x)),
+                x >= self.inf if self.inf in self else x > self.inf,
+                x <= self.sup if self.sup in self else x < self.sup)
 
 
 # Using range from compatibility above (xrange on Py2)
diff --git a/sympy/sets/handlers/issubset.py b/sympy/sets/handlers/issubset.py
--- a/sympy/sets/handlers/issubset.py
+++ b/sympy/sets/handlers/issubset.py
@@ -1,4 +1,4 @@
-from sympy import S
+from sympy import S, Symbol
 from sympy.core.logic import fuzzy_and, fuzzy_bool, fuzzy_not, fuzzy_or
 from sympy.core.relational import Eq
 from sympy.sets.sets import FiniteSet, Interval, Set, Union
@@ -66,6 +66,40 @@ def is_subset_sets(a_range, b_interval): # noqa:F811
             cond_right = a_range.sup <= b_interval.right
         return fuzzy_and([cond_left, cond_right])
 
+@dispatch(Range, FiniteSet)
+def is_subset_sets(a_range, b_finiteset): # noqa:F811
+    try:
+        a_size = a_range.size
+    except ValueError:
+        # symbolic Range of unknown size
+        return None
+    if a_size > len(b_finiteset):
+        return False
+    elif any(arg.has(Symbol) for arg in a_range.args):
+        return fuzzy_and(b_finiteset.contains(x) for x in a_range)
+    else:
+        # Checking A \ B == EmptySet is more efficient than repeated naive
+        # membership checks on an arbitrary FiniteSet.
+        a_set = set(a_range)
+        b_remaining = len(b_finiteset)
+        # Symbolic expressions and numbers of unknown type (integer or not) are
+        # all counted as "candidates", i.e. *potentially* matching some a in
+        # a_range.
+        cnt_candidate = 0
+        for b in b_finiteset:
+            if b.is_Integer:
+                a_set.discard(b)
+            elif fuzzy_not(b.is_integer):
+                pass
+            else:
+                cnt_candidate += 1
+            b_remaining -= 1
+            if len(a_set) > b_remaining + cnt_candidate:
+                return False
+            if len(a_set) == 0:
+                return True
+        return None
+
 @dispatch(Interval, Range)
 def is_subset_sets(a_interval, b_range): # noqa:F811
     if a_interval.measure.is_extended_nonzero:
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -1761,8 +1761,10 @@ def __new__(cls, *args, **kwargs):
         else:
             args = list(map(sympify, args))
 
-        args = list(ordered(set(args), Set._infimum_key))
+        _args_set = set(args)
+        args = list(ordered(_args_set, Set._infimum_key))
         obj = Basic.__new__(cls, *args)
+        obj._args_set = _args_set
         return obj
 
     def _eval_Eq(self, other):
@@ -1830,8 +1832,9 @@ def _contains(self, other):
         """
         Tests whether an element, other, is in the set.
 
-        Relies on Python's set class. This tests for object equality
-        All inputs are sympified
+        The actual test is for mathematical equality (as opposed to
+        syntactical equality). In the worst case all elements of the
+        set must be checked.
 
         Examples
         ========
@@ -1843,9 +1846,13 @@ def _contains(self, other):
         False
 
         """
-        # evaluate=True is needed to override evaluate=False context;
-        # we need Eq to do the evaluation
-        return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True)) for e in self.args)
+        if other in self._args_set:
+            return True
+        else:
+            # evaluate=True is needed to override evaluate=False context;
+            # we need Eq to do the evaluation
+            return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True))
+                for e in self.args)
 
     def _eval_is_subset(self, other):
         return fuzzy_and(other._contains(e) for e in self.args)

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_fancysets.py b/sympy/sets/tests/test_fancysets.py
--- a/sympy/sets/tests/test_fancysets.py
+++ b/sympy/sets/tests/test_fancysets.py
@@ -333,11 +333,14 @@ def test_Range_set():
     assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3) & Eq(x, floor(x))
     assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo) & Eq(x, floor(x))
 
+
+def test_Range_symbolic():
     # symbolic Range
     sr = Range(x, y, t)
     i = Symbol('i', integer=True)
     ip = Symbol('i', integer=True, positive=True)
     ir = Range(i, i + 20, 2)
+    inf = symbols('inf', infinite=True)
     # args
     assert sr.args == (x, y, t)
     assert ir.args == (i, i + 20, 2)
@@ -381,9 +384,27 @@ def test_Range_set():
     assert Range(ip).inf == 0
     assert Range(ip).sup == ip - 1
     raises(ValueError, lambda: Range(i).inf)
+    # as_relational
     raises(ValueError, lambda: sr.as_relational(x))
     assert ir.as_relational(x) == (
         x >= i) & Eq(x, floor(x)) & (x <= i + 18)
+    assert Range(i, i + 1).as_relational(x) == Eq(x, i)
+    # contains() for symbolic values (issue #18146)
+    e = Symbol('e', integer=True, even=True)
+    o = Symbol('o', integer=True, odd=True)
+    assert Range(5).contains(i) == And(i >= 0, i <= 4)
+    assert Range(1).contains(i) == Eq(i, 0)
+    assert Range(-oo, 5, 1).contains(i) == (i <= 4)
+    assert Range(-oo, oo).contains(i) == True
+    assert Range(0, 8, 2).contains(i) == Contains(i, Range(0, 8, 2))
+    assert Range(0, 8, 2).contains(e) == And(e >= 0, e <= 6)
+    assert Range(0, 8, 2).contains(2*i) == And(2*i >= 0, 2*i <= 6)
+    assert Range(0, 8, 2).contains(o) == False
+    assert Range(1, 9, 2).contains(e) == False
+    assert Range(1, 9, 2).contains(o) == And(o >= 1, o <= 7)
+    assert Range(8, 0, -2).contains(o) == False
+    assert Range(9, 1, -2).contains(o) == And(o >= 3, o <= 9)
+    assert Range(-oo, 8, 2).contains(i) == Contains(i, Range(-oo, 8, 2))
 
 
 def test_range_range_intersection():
diff --git a/sympy/sets/tests/test_sets.py b/sympy/sets/tests/test_sets.py
--- a/sympy/sets/tests/test_sets.py
+++ b/sympy/sets/tests/test_sets.py
@@ -613,6 +613,7 @@ def test_measure():
 def test_is_subset():
     assert Interval(0, 1).is_subset(Interval(0, 2)) is True
     assert Interval(0, 3).is_subset(Interval(0, 2)) is False
+    assert Interval(0, 1).is_subset(FiniteSet(0, 1)) is False
 
     assert FiniteSet(1, 2).is_subset(FiniteSet(1, 2, 3, 4))
     assert FiniteSet(4, 5).is_subset(FiniteSet(1, 2, 3, 4)) is False
@@ -646,6 +647,16 @@ def test_is_subset():
     assert Interval(0, 1).is_subset(Interval(0, 1, left_open=True)) is False
     assert Interval(-2, 3).is_subset(Union(Interval(-oo, -2), Interval(3, oo))) is False
 
+    n = Symbol('n', integer=True)
+    assert Range(-3, 4, 1).is_subset(FiniteSet(-10, 10)) is False
+    assert Range(S(10)**100).is_subset(FiniteSet(0, 1, 2)) is False
+    assert Range(6, 0, -2).is_subset(FiniteSet(2, 4, 6)) is True
+    assert Range(1, oo).is_subset(FiniteSet(1, 2)) is False
+    assert Range(-oo, 1).is_subset(FiniteSet(1)) is False
+    assert Range(3).is_subset(FiniteSet(0, 1, n)) is None
+    assert Range(n, n + 2).is_subset(FiniteSet(n, n + 1)) is True
+    assert Range(5).is_subset(Interval(0, 4, right_open=True)) is False
+
 
 def test_is_proper_subset():
     assert Interval(0, 1).is_proper_subset(Interval(0, 2)) is True

```


## Code snippets

### 1 - sympy/sets/fancysets.py:

Start line: 670, End line: 690

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
        else:
            return other.is_Integer
        if (ref - other) % self.step:  # off sequence
            return S.false
        return _sympify(other >= self.inf and other <= self.sup)
```
### 2 - sympy/sets/fancysets.py:

Start line: 865, End line: 912

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

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        from sympy.functions.elementary.integers import floor
        return And(
            Eq(x, floor(x)),
            x >= self.inf if self.inf in self else x > self.inf,
            x <= self.sup if self.sup in self else x < self.sup)


# Using range from compatibility above (xrange on Py2)
if PY3:
    converter[range] = lambda r: Range(r.start, r.stop, r.step)
else:
    converter[range] = lambda r: Range(*r.__reduce__()[1])
```
### 3 - sympy/sets/fancysets.py:

Start line: 692, End line: 730

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
        if self.has(Symbol):
            if dif.has(Symbol) or self.step.has(Symbol) or (
                    not self.start.is_integer and not self.stop.is_integer):
                raise ValueError('invalid method for symbolic range')
        if dif.is_infinite:
            return S.Infinity
        return Integer(abs(dif//self.step))

    def __nonzero__(self):
        return self.start != self.stop

    __bool__ = __nonzero__
```
### 4 - sympy/core/relational.py:

Start line: 370, End line: 394

```python
class Relational(Boolean, Expr, EvalfMixin):

    def _eval_trigsimp(self, **opts):
        from sympy.simplify import trigsimp
        return self.func(trigsimp(self.lhs, **opts), trigsimp(self.rhs, **opts))


    def __nonzero__(self):
        raise TypeError("cannot determine truth value of Relational")

    __bool__ = __nonzero__

    def _eval_as_set(self):
        # self is univariate and periodicity(self, x) in (0, None)
        from sympy.solvers.inequalities import solve_univariate_inequality
        syms = self.free_symbols
        assert len(syms) == 1
        x = syms.pop()
        return solve_univariate_inequality(self, x, relational=False)

    @property
    def binary_symbols(self):
        # override where necessary
        return set()


Rel = Relational
```
### 5 - sympy/core/relational.py:

Start line: 479, End line: 577

```python
class Equality(Relational):

    def __new__(cls, lhs, rhs=None, **options):
        # ... other code

        if evaluate:
            # If one expression has an _eval_Eq, return its results.
            if hasattr(lhs, '_eval_Eq'):
                r = lhs._eval_Eq(rhs)
                if r is not None:
                    return r
            if hasattr(rhs, '_eval_Eq'):
                r = rhs._eval_Eq(lhs)
                if r is not None:
                    return r
            # If expressions have the same structure, they must be equal.
            if lhs == rhs:
                return S.true  # e.g. True == True
            elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
                return S.false  # True != False
            elif not (lhs.is_Symbol or rhs.is_Symbol) and (
                    isinstance(lhs, Boolean) !=
                    isinstance(rhs, Boolean)):
                return S.false  # only Booleans can equal Booleans

            if lhs.is_infinite or rhs.is_infinite:
                if fuzzy_xor([lhs.is_infinite, rhs.is_infinite]):
                    return S.false
                if fuzzy_xor([lhs.is_extended_real, rhs.is_extended_real]):
                    return S.false
                if fuzzy_and([lhs.is_extended_real, rhs.is_extended_real]):
                    r = fuzzy_xor([lhs.is_extended_positive, fuzzy_not(rhs.is_extended_positive)])
                    return S(r)

                # Try to split real/imaginary parts and equate them
                I = S.ImaginaryUnit

                def split_real_imag(expr):
                    real_imag = lambda t: (
                            'real' if t.is_extended_real else
                            'imag' if (I*t).is_extended_real else None)
                    return sift(Add.make_args(expr), real_imag)

                lhs_ri = split_real_imag(lhs)
                if not lhs_ri[None]:
                    rhs_ri = split_real_imag(rhs)
                    if not rhs_ri[None]:
                        eq_real = Eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']))
                        eq_imag = Eq(I*Add(*lhs_ri['imag']), I*Add(*rhs_ri['imag']))
                        res = fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))
                        if res is not None:
                            return S(res)

                # Compare e.g. zoo with 1+I*oo by comparing args
                arglhs = arg(lhs)
                argrhs = arg(rhs)
                # Guard against Eq(nan, nan) -> False
                if not (arglhs == S.NaN and argrhs == S.NaN):
                    res = fuzzy_bool(Eq(arglhs, argrhs))
                    if res is not None:
                        return S(res)

                return Relational.__new__(cls, lhs, rhs, **options)

            if all(isinstance(i, Expr) for i in (lhs, rhs)):
                # see if the difference evaluates
                dif = lhs - rhs
                z = dif.is_zero
                if z is not None:
                    if z is False and dif.is_commutative:  # issue 10728
                        return S.false
                    if z:
                        return S.true
                # evaluate numerically if possible
                n2 = _n2(lhs, rhs)
                if n2 is not None:
                    return _sympify(n2 == 0)
                # see if the ratio evaluates
                n, d = dif.as_numer_denom()
                rv = None
                if n.is_zero:
                    rv = d.is_nonzero
                elif n.is_finite:
                    if d.is_infinite:
                        rv = S.true
                    elif n.is_zero is False:
                        rv = d.is_infinite
                        if rv is None:
                            # if the condition that makes the denominator
                            # infinite does not make the original expression
                            # True then False can be returned
                            l, r = clear_coefficients(d, S.Infinity)
                            args = [_.subs(l, r) for _ in (lhs, rhs)]
                            if args != [lhs, rhs]:
                                rv = fuzzy_bool(Eq(*args))
                                if rv is True:
                                    rv = None
                elif any(a.is_infinite for a in Add.make_args(n)):
                    # (inf or nan)/x != 0
                    rv = S.false
                if rv is not None:
                    return _sympify(rv)

        return Relational.__new__(cls, lhs, rhs, **options)
```
### 6 - sympy/sets/sets.py:

Start line: 1412, End line: 1514

```python
class Intersection(Set, LatticeOp):

    @staticmethod
    def _handle_finite_sets(args):
        '''Simplify intersection of one or more FiniteSets and other sets'''

        # First separate the FiniteSets from the others
        fs_args, others = sift(args, lambda x: x.is_FiniteSet, binary=True)

        # Let the caller handle intersection of non-FiniteSets
        if not fs_args:
            return

        # Convert to Python sets and build the set of all elements
        fs_sets = [set(fs) for fs in fs_args]
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Extract elements that are definitely in or definitely not in the
        # intersection. Here we check contains for all of args.
        definite = set()
        for e in all_elements:
            inall = fuzzy_and(s.contains(e) for s in args)
            if inall is True:
                definite.add(e)
            if inall is not None:
                for s in fs_sets:
                    s.discard(e)

        # At this point all elements in all of fs_sets are possibly in the
        # intersection. In some cases this is because they are definitely in
        # the intersection of the finite sets but it's not clear if they are
        # members of others. We might have {m, n}, {m}, and Reals where we
        # don't know if m or n is real. We want to remove n here but it is
        # possibly in because it might be equal to m. So what we do now is
        # extract the elements that are definitely in the remaining finite
        # sets iteratively until we end up with {n}, {}. At that point if we
        # get any empty set all remaining elements are discarded.

        fs_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Need fuzzy containment testing
        fs_symsets = [FiniteSet(*s) for s in fs_sets]

        while fs_elements:
            for e in fs_elements:
                infs = fuzzy_and(s.contains(e) for s in fs_symsets)
                if infs is True:
                    definite.add(e)
                if infs is not None:
                    for n, s in enumerate(fs_sets):
                        # Update Python set and FiniteSet
                        if e in s:
                            s.remove(e)
                            fs_symsets[n] = FiniteSet(*s)
                    fs_elements.remove(e)
                    break
            # If we completed the for loop without removing anything we are
            # done so quit the outer while loop
            else:
                break

        # If any of the sets of remainder elements is empty then we discard
        # all of them for the intersection.
        if not all(fs_sets):
            fs_sets = [set()]

        # Here we fold back the definitely included elements into each fs.
        # Since they are definitely included they must have been members of
        # each FiniteSet to begin with. We could instead fold these in with a
        # Union at the end to get e.g. {3}|({x}&{y}) rather than {3,x}&{3,y}.
        if definite:
            fs_sets = [fs | definite for fs in fs_sets]

        if fs_sets == [set()]:
            return S.EmptySet

        sets = [FiniteSet(*s) for s in fs_sets]

        # Any set in others is redundant if it contains all the elements that
        # are in the finite sets so we don't need it in the Intersection
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())
        is_redundant = lambda o: all(fuzzy_bool(o.contains(e)) for e in all_elements)
        others = [o for o in others if not is_redundant(o)]

        if others:
            rest = Intersection(*others)
            # XXX: Maybe this shortcut should be at the beginning. For large
            # FiniteSets it could much more efficient to process the other
            # sets first...
            if rest is S.EmptySet:
                return S.EmptySet
            # Flatten the Intersection
            if rest.is_Intersection:
                sets.extend(rest.args)
            else:
                sets.append(rest)

        if len(sets) == 1:
            return sets[0]
        else:
            return Intersection(*sets, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])
```
### 7 - sympy/sets/fancysets.py:

Start line: 741, End line: 863

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
### 8 - sympy/core/relational.py:

Start line: 288, End line: 368

```python
class Relational(Boolean, Expr, EvalfMixin):

    def _eval_simplify(self, **kwargs):
        r = self
        r = r.func(*[i.simplify(**kwargs) for i in r.args])
        if r.is_Relational:
            dif = r.lhs - r.rhs
            # replace dif with a valid Number that will
            # allow a definitive comparison with 0
            v = None
            if dif.is_comparable:
                v = dif.n(2)
            elif dif.equals(0):  # XXX this is expensive
                v = S.Zero
            if v is not None:
                r = r.func._eval_relation(v, S.Zero)
            r = r.canonical
            # If there is only one symbol in the expression,
            # try to write it on a simplified form
            free = list(filter(lambda x: x.is_real is not False, r.free_symbols))
            if len(free) == 1:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    x = free.pop()
                    dif = r.lhs - r.rhs
                    m, b = linear_coeffs(dif, x)
                    if m.is_zero is False:
                        if m.is_negative:
                            # Dividing with a negative number, so change order of arguments
                            # canonical will put the symbol back on the lhs later
                            r = r.func(-b/m, x)
                        else:
                            r = r.func(x, -b/m)
                    else:
                        r = r.func(b, S.zero)
                except ValueError:
                    # maybe not a linear function, try polynomial
                    from sympy.polys import Poly, poly, PolynomialError, gcd
                    try:
                        p = poly(dif, x)
                        c = p.all_coeffs()
                        constant = c[-1]
                        c[-1] = 0
                        scale = gcd(c)
                        c = [ctmp/scale for ctmp in c]
                        r = r.func(Poly.from_list(c, x).as_expr(), -constant/scale)
                    except PolynomialError:
                        pass
            elif len(free) >= 2:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    from sympy.polys import gcd
                    free = list(ordered(free))
                    dif = r.lhs - r.rhs
                    m = linear_coeffs(dif, *free)
                    constant = m[-1]
                    del m[-1]
                    scale = gcd(m)
                    m = [mtmp/scale for mtmp in m]
                    nzm = list(filter(lambda f: f[0] != 0, list(zip(m, free))))
                    if scale.is_zero is False:
                        if constant != 0:
                            # lhs: expression, rhs: constant
                            newexpr = Add(*[i*j for i, j in nzm])
                            r = r.func(newexpr, -constant/scale)
                        else:
                            # keep first term on lhs
                            lhsterm = nzm[0][0]*nzm[0][1]
                            del nzm[0]
                            newexpr = Add(*[i*j for i, j in nzm])
                            r = r.func(lhsterm, -newexpr)

                    else:
                        r = r.func(constant, S.zero)
                except ValueError:
                    pass
        # Did we get a simplified result?
        r = r.canonical
        measure = kwargs['measure']
        if measure(r) < kwargs['ratio']*measure(self):
            return r
        else:
            return self
```
### 9 - sympy/sets/fancysets.py:

Start line: 499, End line: 582

```python
class Range(Set):
    """
    Represents a range of integers. Can be called as Range(stop),
    Range(start, stop), or Range(start, stop, step); when stop is
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
### 10 - sympy/sets/sets.py:

Start line: 1049, End line: 1062

```python
class Interval(Set, EvalfMixin):

    def _contains(self, other):
        if not isinstance(other, Expr) or (
                other is S.Infinity or
                other is S.NegativeInfinity or
                other is S.NaN or
                other is S.ComplexInfinity) or other.is_extended_real is False:
            return false

        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            if not other.is_extended_real is None:
                return other.is_extended_real

        d = Dummy()
        return self.as_relational(d).subs(d, other)
```
### 11 - sympy/sets/fancysets.py:

Start line: 584, End line: 646

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
### 19 - sympy/sets/sets.py:

Start line: 1790, End line: 1827

```python
class FiniteSet(Set, EvalfMixin):

    def _complement(self, other):
        if isinstance(other, Interval):
            nums = sorted(m for m in self.args if m.is_number)
            if other == S.Reals and nums != []:
                syms = [m for m in self.args if m.is_Symbol]
                # Reals cannot contain elements other than numbers and symbols.

                intervals = []  # Build up a list of intervals between the elements
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))  # both open
                intervals.append(Interval(nums[-1], S.Infinity, True, True))

                if syms != []:
                    return Complement(Union(*intervals, evaluate=False),
                            FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:
                return None

        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)

        return Set._complement(self, other)
```
### 21 - sympy/sets/handlers/issubset.py:

Start line: 1, End line: 32

```python
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_bool, fuzzy_not, fuzzy_or
from sympy.core.relational import Eq
from sympy.sets.sets import FiniteSet, Interval, Set, Union
from sympy.sets.fancysets import Complexes, Reals, Range, Rationals
from sympy.multipledispatch import dispatch


_inf_sets = [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals, S.Complexes]

@dispatch(Set, Set)
def is_subset_sets(a, b): # noqa:F811
    return None

@dispatch(Interval, Interval)
def is_subset_sets(a, b): # noqa:F811
    # This is correct but can be made more comprehensive...
    if fuzzy_bool(a.start < b.start):
        return False
    if fuzzy_bool(a.end > b.end):
        return False
    if (b.left_open and not a.left_open and fuzzy_bool(Eq(a.start, b.start))):
        return False
    if (b.right_open and not a.right_open and fuzzy_bool(Eq(a.end, b.end))):
        return False

@dispatch(Interval, FiniteSet)
def is_subset_sets(a_interval, b_fs): # noqa:F811
    # An Interval can only be a subset of a finite set if it is finite
    # which can only happen if it has zero measure.
    if fuzzy_not(a_interval.measure.is_zero):
        return False
```
### 26 - sympy/sets/handlers/issubset.py:

Start line: 69, End line: 102

```python
@dispatch(Interval, Range)
def is_subset_sets(a_interval, b_range): # noqa:F811
    if a_interval.measure.is_extended_nonzero:
        return False

@dispatch(Interval, Rationals)
def is_subset_sets(a_interval, b_rationals): # noqa:F811
    if a_interval.measure.is_extended_nonzero:
        return False

@dispatch(Range, Complexes)
def is_subset_sets(a, b): # noqa:F811
    return True

@dispatch(Complexes, Interval)
def is_subset_sets(a, b): # noqa:F811
    return False

@dispatch(Complexes, Range)
def is_subset_sets(a, b): # noqa:F811
    return False

@dispatch(Complexes, Rationals)
def is_subset_sets(a, b): # noqa:F811
    return False

@dispatch(Rationals, Reals)
def is_subset_sets(a, b): # noqa:F811
    return True

@dispatch(Rationals, Range)
def is_subset_sets(a, b): # noqa:F811
    return False
```
### 28 - sympy/sets/sets.py:

Start line: 1768, End line: 1788

```python
class FiniteSet(Set, EvalfMixin):

    def _eval_Eq(self, other):
        if not isinstance(other, FiniteSet):
            # XXX: If Interval(x, x, evaluate=False) worked then the line
            # below would mean that
            #     FiniteSet(x) & Interval(x, x, evaluate=False) -> false
            if isinstance(other, Interval):
                return false
            elif isinstance(other, Set):
                return None
            return false

        def all_in_both():
            s_set = set(self.args)
            o_set = set(other.args)
            yield fuzzy_and(self._contains(e) for e in o_set - s_set)
            yield fuzzy_and(other._contains(e) for e in s_set - o_set)

        return tfn[fuzzy_and(all_in_both())]

    def __iter__(self):
        return iter(self.args)
```
### 29 - sympy/sets/sets.py:

Start line: 285, End line: 327

```python
class Set(Basic):

    def contains(self, other):
        """
        Returns a SymPy value indicating whether ``other`` is contained
        in ``self``: ``true`` if it is, ``false`` if it isn't, else
        an unevaluated ``Contains`` expression (or, as in the case of
        ConditionSet and a union of FiniteSet/Intervals, an expression
        indicating the conditions for containment).

        Examples
        ========

        >>> from sympy import Interval, S
        >>> from sympy.abc import x

        >>> Interval(0, 1).contains(0.5)
        True

        As a shortcut it is possible to use the 'in' operator, but that
        will raise an error unless an affirmative true or false is not
        obtained.

        >>> Interval(0, 1).contains(x)
        (0 <= x) & (x <= 1)
        >>> x in Interval(0, 1)
        Traceback (most recent call last):
        ...
        TypeError: did not evaluate to a bool: None

        The result of 'in' is a bool, not a SymPy value

        >>> 1 in Interval(0, 2)
        True
        >>> _ is S.true
        False
        """
        other = sympify(other, strict=True)
        c = self._contains(other)
        if c is None:
            return Contains(other, self, evaluate=False)
        b = tfn[c]
        if b is None:
            return c
        return b
```
### 32 - sympy/sets/sets.py:

Start line: 1374, End line: 1410

```python
class Intersection(Set, LatticeOp):

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - set((s,))
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")
```
### 35 - sympy/sets/sets.py:

Start line: 1936, End line: 2001

```python
class SymmetricDifference(Set):
    """Represents the set of elements which are in either of the
    sets and not in their intersection.

    Examples
    ========

    >>> from sympy import SymmetricDifference, FiniteSet
    >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
    FiniteSet(1, 2, 4, 5)

    See Also
    ========

    Complement, Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_difference
    """

    is_SymmetricDifference = True

    def __new__(cls, a, b, evaluate=True):
        if evaluate:
            return SymmetricDifference.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        result = B._symmetric_difference(A)
        if result is not None:
            return result
        else:
            return SymmetricDifference(A, B, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite a symmetric_difference in terms of equalities and
        logic operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = B.as_relational(symbol)

        return Xor(A_rel, B_rel)

    @property
    def is_iterable(self):
        if all(arg.is_iterable for arg in self.args):
            return True

    def __iter__(self):

        args = self.args
        union = roundrobin(*(iter(arg) for arg in args))

        for item in union:
            count = 0
            for s in args:
                if item in s:
                    count += 1

            if count % 2 == 1:
                yield item
```
### 37 - sympy/sets/sets.py:

Start line: 1064, End line: 1119

```python
class Interval(Set, EvalfMixin):

    def as_relational(self, x):
        """Rewrite an interval in terms of inequalities and logic operators."""
        x = sympify(x)
        if self.right_open:
            right = x < self.end
        else:
            right = x <= self.end
        if self.left_open:
            left = self.start < x
        else:
            left = self.start <= x
        return And(left, right)

    @property
    def _measure(self):
        return self.end - self.start

    def to_mpi(self, prec=53):
        return mpi(mpf(self.start._eval_evalf(prec)),
            mpf(self.end._eval_evalf(prec)))

    def _eval_evalf(self, prec):
        return Interval(self.left._eval_evalf(prec),
            self.right._eval_evalf(prec),
                        left_open=self.left_open, right_open=self.right_open)

    def _is_comparable(self, other):
        is_comparable = self.start.is_comparable
        is_comparable &= self.end.is_comparable
        is_comparable &= other.start.is_comparable
        is_comparable &= other.end.is_comparable

        return is_comparable

    @property
    def is_left_unbounded(self):
        """Return ``True`` if the left endpoint is negative infinity. """
        return self.left is S.NegativeInfinity or self.left == Float("-inf")

    @property
    def is_right_unbounded(self):
        """Return ``True`` if the right endpoint is positive infinity. """
        return self.right is S.Infinity or self.right == Float("+inf")

    def _eval_Eq(self, other):
        if not isinstance(other, Interval):
            if isinstance(other, FiniteSet):
                return false
            elif isinstance(other, Set):
                return None
            return false

        return And(Eq(self.left, other.left),
                   Eq(self.right, other.right),
                   self.left_open == other.left_open,
                   self.right_open == other.right_open)
```
### 38 - sympy/sets/sets.py:

Start line: 1911, End line: 1933

```python
class FiniteSet(Set, EvalfMixin):

    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return other.is_subset(self)

    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_superset(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_subset(other)

    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_subset(other)


converter[set] = lambda x: FiniteSet(*x)
converter[frozenset] = lambda x: FiniteSet(*x)
```
### 39 - sympy/sets/fancysets.py:

Start line: 732, End line: 740

```python
class Range(Set):

    def __getitem__(self, i):
        from sympy.functions.elementary.integers import ceiling
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        # ... other code
```
### 41 - sympy/sets/sets.py:

Start line: 1829, End line: 1848

```python
class FiniteSet(Set, EvalfMixin):

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Relies on Python's set class. This tests for object equality
        All inputs are sympified

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        # evaluate=True is needed to override evaluate=False context;
        # we need Eq to do the evaluation
        return fuzzy_or(fuzzy_bool(Eq(e, other, evaluate=True)) for e in self.args)
```
### 44 - sympy/sets/sets.py:

Start line: 388, End line: 420

```python
class Set(Basic):

    def _eval_is_subset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    def _eval_is_superset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # This should be deprecated:
    def issubset(self, other):
        """
        Alias for :meth:`is_subset()`
        """
        return self.is_subset(other)

    def is_proper_subset(self, other):
        """
        Returns True if 'self' is a proper subset of 'other'.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_proper_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_proper_subset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_subset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)
```
### 47 - sympy/sets/sets.py:

Start line: 178, End line: 214

```python
class Set(Basic):

    def _complement(self, other):
        # this behaves as other - self
        if isinstance(self, ProductSet) and isinstance(other, ProductSet):
            # If self and other are disjoint then other - self == self
            if len(self.sets) != len(other.sets):
                return other

            # There can be other ways to represent this but this gives:
            # (A x B) - (C x D) = ((A - C) x B) U (A x (B - D))
            overlaps = []
            pairs = list(zip(self.sets, other.sets))
            for n in range(len(pairs)):
                sets = (o if i != n else o-s for i, (s, o) in enumerate(pairs))
                overlaps.append(ProductSet(*sets))
            return Union(*overlaps)

        elif isinstance(other, Interval):
            if isinstance(self, Interval) or isinstance(self, FiniteSet):
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            return Union(*(o - self for o in other.args))

        elif isinstance(other, Complement):
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif isinstance(other, EmptySet):
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            from sympy.utilities.iterables import sift

            sifted = sift(other, lambda x: fuzzy_bool(self.contains(x)))
            # ignore those that are contained in self
            return Union(FiniteSet(*(sifted[False])),
                Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                if sifted[None] else S.EmptySet)
```
### 49 - sympy/sets/handlers/issubset.py:

Start line: 50, End line: 67

```python
@dispatch(Range, Range)
def is_subset_sets(a, b): # noqa:F811
    if a.step == b.step == 1:
        return fuzzy_and([fuzzy_bool(a.start >= b.start),
                          fuzzy_bool(a.stop <= b.stop)])

@dispatch(Range, Interval)
def is_subset_sets(a_range, b_interval): # noqa:F811
    if a_range.step.is_positive:
        if b_interval.left_open and a_range.inf.is_finite:
            cond_left = a_range.inf > b_interval.left
        else:
            cond_left = a_range.inf >= b_interval.left
        if b_interval.right_open and a_range.sup.is_finite:
            cond_right = a_range.sup < b_interval.right
        else:
            cond_right = a_range.sup <= b_interval.right
        return fuzzy_and([cond_left, cond_right])
```
### 50 - sympy/sets/sets.py:

Start line: 1850, End line: 1890

```python
class FiniteSet(Set, EvalfMixin):

    def _eval_is_subset(self, other):
        return fuzzy_and(other._contains(e) for e in self.args)

    @property
    def _boundary(self):
        return self

    @property
    def _inf(self):
        from sympy.functions.elementary.miscellaneous import Min
        return Min(*self)

    @property
    def _sup(self):
        from sympy.functions.elementary.miscellaneous import Max
        return Max(*self)

    @property
    def measure(self):
        return 0

    def __len__(self):
        return len(self.args)

    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        from sympy.core.relational import Eq
        return Or(*[Eq(symbol, elem) for elem in self])

    def compare(self, other):
        return (hash(self) - hash(other))

    def _eval_evalf(self, prec):
        return FiniteSet(*[elem._eval_evalf(prec) for elem in self])

    @property
    def _sorted_args(self):
        return self.args

    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])
```
### 55 - sympy/sets/sets.py:

Start line: 129, End line: 153

```python
class Set(Basic):

    def intersection(self, other):
        """
        Alias for :meth:`intersect()`
        """
        return self.intersect(other)

    def is_disjoint(self, other):
        """
        Returns True if 'self' and 'other' are disjoint

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 2).is_disjoint(Interval(1, 2))
        False
        >>> Interval(0, 2).is_disjoint(Interval(3, 4))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Disjoint_sets
        """
        return self.intersect(other) == S.EmptySet
```
### 56 - sympy/sets/fancysets.py:

Start line: 23, End line: 69

```python
class Rationals(with_metaclass(Singleton, Set)):
    """
    Represents the rational numbers. This set is also available as
    the Singleton, S.Rationals.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        if other.is_Number:
            return other.is_Rational
        return other.is_rational

    def __iter__(self):
        from sympy.core.numbers import igcd, Rational
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return self
```
### 58 - sympy/sets/fancysets.py:

Start line: 1237, End line: 1269

```python
class ComplexRegion(Set):

    def _contains(self, other):
        from sympy.functions import arg, Abs
        from sympy.core.containers import Tuple
        other = sympify(other)
        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

        # If the other is not an Expression, and neither a Tuple
        if not isinstance(other, Expr) and not isinstance(other, Tuple):
            return S.false
        # self in rectangular form
        if not self.polar:
            re, im = other if isTuple else other.as_real_imag()
            for element in self.psets:
                if And(element.args[0]._contains(re),
                        element.args[1]._contains(im)):
                    return True
            return False

        # self in polar form
        elif self.polar:
            if isTuple:
                r, theta = other
            elif other.is_zero:
                r, theta = S.Zero, S.Zero
            else:
                r, theta = Abs(other), arg(other)
            for element in self.psets:
                if And(element.args[0]._contains(r),
                        element.args[1]._contains(theta)):
                    return True
            return False
```
### 60 - sympy/sets/fancysets.py:

Start line: 648, End line: 668

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
### 62 - sympy/sets/sets.py:

Start line: 1349, End line: 1372

```python
class Intersection(Set, LatticeOp):

    @property
    @cacheit
    def args(self):
        return self._args

    @property
    def is_iterable(self):
        return any(arg.is_iterable for arg in self.args)

    @property
    def is_finite_set(self):
        if fuzzy_or(arg.is_finite_set for arg in self.args):
            return True

    @property
    def _inf(self):
        raise NotImplementedError()

    @property
    def _sup(self):
        raise NotImplementedError()

    def _contains(self, other):
        return And(*[set.contains(other) for set in self.args])
```
### 64 - sympy/sets/sets.py:

Start line: 1517, End line: 1603

```python
class Complement(Set, EvalfMixin):
    r"""Represents the set difference or relative complement of a set with
    another set.

    `A - B = \{x \in A \mid x \notin B\}`


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    FiniteSet(0, 2)

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] http://mathworld.wolfram.com/ComplementSet.html
    """

    is_Complement = True

    def __new__(cls, a, b, evaluate=True):
        if evaluate:
            return Complement.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):
            return S.EmptySet

        if isinstance(B, Union):
            return Intersection(*(s.complement(A) for s in B.args))

        result = B._complement(A)
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]
        B = self.args[1]
        return And(A.contains(other), Not(B.contains(other)))

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = Not(B.as_relational(symbol))

        return And(A_rel, B_rel)

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:
            return True

    @property
    def is_finite_set(self):
        A, B = self.args
        a_finite = A.is_finite_set
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:
            return False

    def __iter__(self):
        A, B = self.args
        for a in A:
            if a not in B:
                    yield a
            else:
                continue
```
### 65 - sympy/sets/sets.py:

Start line: 889, End line: 928

```python
class Interval(Set, EvalfMixin):

    def __new__(cls, start, end, left_open=False, right_open=False):

        start = _sympify(start)
        end = _sympify(end)
        left_open = _sympify(left_open)
        right_open = _sympify(right_open)

        if not all(isinstance(a, (type(true), type(false)))
            for a in [left_open, right_open]):
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        inftys = [S.Infinity, S.NegativeInfinity]
        # Only allow real intervals (use symbols with 'is_extended_real=True').
        if not all(i.is_extended_real is not False or i in inftys for i in (start, end)):
            raise ValueError("Non-real intervals are not supported")

        # evaluate if possible
        if (end < start) == True:
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start is S.Infinity or start is S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # Make sure infinite interval end points are open.
        if start is S.NegativeInfinity:
            left_open = true
        if end is S.Infinity:
            right_open = true
        if start == S.Infinity or end == S.NegativeInfinity:
            return S.EmptySet

        return Basic.__new__(cls, start, end, left_open, right_open)
```
### 67 - sympy/sets/sets.py:

Start line: 778, End line: 790

```python
class ProductSet(Set):

    def as_relational(self, *symbols):
        symbols = [_sympify(s) for s in symbols]
        if len(symbols) != len(self.sets) or not all(
                i.is_Symbol for i in symbols):
            raise ValueError(
                'number of symbols must match the number of sets')
        return And(*[s.as_relational(i) for s, i in zip(self.sets, symbols)])

    @property
    def _boundary(self):
        return Union(*(ProductSet(*(b + b.boundary if i != j else b.boundary
                                for j, b in enumerate(self.sets)))
                                for i, a in enumerate(self.sets)))
```
### 69 - sympy/sets/fancysets.py:

Start line: 72, End line: 133

```python
class Naturals(with_metaclass(Singleton, Set)):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the Singleton, S.Naturals.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========

    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf = S.One
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        elif other.is_positive and other.is_integer:
            return True
        elif other.is_integer is False or other.is_positive is False:
            return False

    def _eval_is_subset(self, other):
        return Range(1, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(1, oo).is_superset(other)

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        from sympy.functions.elementary.integers import floor
        return And(Eq(floor(x), x), x >= self.inf, x < oo)
```
### 70 - sympy/sets/sets.py:

Start line: 1, End line: 36

```python
from __future__ import print_function, division

from collections import defaultdict
import inspect

from sympy.core.basic import Basic
from sympy.core.compatibility import (iterable, with_metaclass,
    ordered, range, PY3, reduce)
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import deprecated
from sympy.core.evalf import EvalfMixin
from sympy.core.evaluate import global_evaluate
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_bool, fuzzy_or, fuzzy_and, fuzzy_not
from sympy.core.numbers import Float
from sympy.core.operations import LatticeOp
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Symbol, Dummy, _uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.sets.contains import Contains
from sympy.utilities import subsets
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.iterables import iproduct, sift, roundrobin
from sympy.utilities.misc import func_name, filldedent

from mpmath import mpi, mpf


tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})
```
### 74 - sympy/sets/fancysets.py:

Start line: 136, End line: 160

```python
class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(oo).is_superset(other)
```
### 76 - sympy/sets/sets.py:

Start line: 155, End line: 176

```python
class Set(Basic):

    def isdisjoint(self, other):
        """
        Alias for :meth:`is_disjoint()`
        """
        return self.is_disjoint(other)

    def complement(self, universe):
        r"""
        The complement of 'self' w.r.t the given universe.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(0, 1).complement(S.Reals)
        Union(Interval.open(-oo, 0), Interval.open(1, oo))

        >>> Interval(0, 1).complement(S.UniversalSet)
        Complement(UniversalSet, Interval(0, 1))

        """
        return Complement(universe, self)
```
