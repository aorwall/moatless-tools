# sympy__sympy-15241

| **sympy/sympy** | `5997e30a33f92e6b4b4d351e835feb7379a0e31d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 13505 |
| **Any found context length** | 13505 |
| **Avg pos** | 28.0 |
| **Min pos** | 28 |
| **Max pos** | 28 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -1298,72 +1298,101 @@ def _remove_derived_once(cls, v):
         return [i[0] if i[1] == 1 else i for i in v]
 
     @classmethod
-    def _sort_variable_count(cls, varcounts):
+    def _sort_variable_count(cls, vc):
         """
-        Sort (variable, count) pairs by variable, but disallow sorting of non-symbols.
+        Sort (variable, count) pairs into canonical order while
+        retaining order of variables that do not commute during
+        differentiation:
 
-        The count is not sorted. It is kept in the same order as the input
-        after sorting by variable.
-
-        When taking derivatives, the following rules usually hold:
-
-        * Derivative wrt different symbols commute.
-        * Derivative wrt different non-symbols commute.
-        * Derivatives wrt symbols and non-symbols don't commute.
+        * symbols and functions commute with each other
+        * derivatives commute with each other
+        * a derivative doesn't commute with anything it contains
+        * any other object is not allowed to commute if it has
+          free symbols in common with another object
 
         Examples
         ========
 
-        >>> from sympy import Derivative, Function, symbols
+        >>> from sympy import Derivative, Function, symbols, cos
         >>> vsort = Derivative._sort_variable_count
         >>> x, y, z = symbols('x y z')
         >>> f, g, h = symbols('f g h', cls=Function)
 
-        >>> vsort([(x, 3), (y, 2), (z, 1)])
-        [(x, 3), (y, 2), (z, 1)]
-
-        >>> vsort([(h(x), 1), (g(x), 1), (f(x), 1)])
-        [(f(x), 1), (g(x), 1), (h(x), 1)]
+        Contiguous items are collapsed into one pair:
 
-        >>> vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1), (f(x), 1)])
-        [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]
+        >>> vsort([(x, 1), (x, 1)])
+        [(x, 2)]
+        >>> vsort([(y, 1), (f(x), 1), (y, 1), (f(x), 1)])
+        [(y, 2), (f(x), 2)]
 
-        >>> vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)])
-        [(x, 1), (f(x), 1), (y, 1), (f(y), 1)]
+        Ordering is canonical.
 
-        >>> vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1), (h(x), 1), (y, 2), (x, 1)])
-        [(x, 2), (y, 1), (f(x), 1), (g(x), 1), (z, 1), (h(x), 1), (x, 1), (y, 2)]
+        >>> def vsort0(*v):
+        ...     # docstring helper to
+        ...     # change vi -> (vi, 0), sort, and return vi vals
+        ...     return [i[0] for i in vsort([(i, 0) for i in v])]
 
-        >>> vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)])
-        [(y, 1), (z, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]
+        >>> vsort0(y, x)
+        [x, y]
+        >>> vsort0(g(y), g(x), f(y))
+        [f(y), g(x), g(y)]
 
-        >>> vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)])
-        [(y, 2), (z, 1), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (x, 1), (y, 1), (z, 2), (z, 1)]
+        Symbols are sorted as far to the left as possible but never
+        move to the left of a derivative having the same symbol in
+        its variables; the same applies to AppliedUndef which are
+        always sorted after Symbols:
 
+        >>> dfx = f(x).diff(x)
+        >>> assert vsort0(dfx, y) == [y, dfx]
+        >>> assert vsort0(dfx, x) == [dfx, x]
         """
-        sorted_vars = []
-        symbol_part = []
-        non_symbol_part = []
-        for (v, c) in varcounts:
-            if not v.is_symbol:
-                if len(symbol_part) > 0:
-                    sorted_vars.extend(sorted(symbol_part,
-                                              key=lambda i: default_sort_key(i[0])))
-                    symbol_part = []
-                non_symbol_part.append((v, c))
+        from sympy.utilities.iterables import uniq, topological_sort
+        if not vc:
+            return []
+        vc = list(vc)
+        if len(vc) == 1:
+            return [Tuple(*vc[0])]
+        V = list(range(len(vc)))
+        E = []
+        v = lambda i: vc[i][0]
+        D = Dummy()
+        def _block(d, v, wrt=False):
+            # return True if v should not come before d else False
+            if d == v:
+                return wrt
+            if d.is_Symbol:
+                return False
+            if isinstance(d, Derivative):
+                # a derivative blocks if any of it's variables contain
+                # v; the wrt flag will return True for an exact match
+                # and will cause an AppliedUndef to block if v is in
+                # the arguments
+                if any(_block(k, v, wrt=True)
+                        for k, _ in d.variable_count):
+                    return True
+                return False
+            if not wrt and isinstance(d, AppliedUndef):
+                return False
+            if v.is_Symbol:
+                return v in d.free_symbols
+            if isinstance(v, AppliedUndef):
+                return _block(d.xreplace({v: D}), D)
+            return d.free_symbols & v.free_symbols
+        for i in range(len(vc)):
+            for j in range(i):
+                if _block(v(j), v(i)):
+                    E.append((j,i))
+        # this is the default ordering to use in case of ties
+        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
+        ix = topological_sort((V, E), key=lambda i: O[v(i)])
+        # merge counts of contiguously identical items
+        merged = []
+        for v, c in [vc[i] for i in ix]:
+            if merged and merged[-1][0] == v:
+                merged[-1][1] += c
             else:
-                if len(non_symbol_part) > 0:
-                    sorted_vars.extend(sorted(non_symbol_part,
-                                              key=lambda i: default_sort_key(i[0])))
-                    non_symbol_part = []
-                symbol_part.append((v, c))
-        if len(non_symbol_part) > 0:
-            sorted_vars.extend(sorted(non_symbol_part,
-                                      key=lambda i: default_sort_key(i[0])))
-        if len(symbol_part) > 0:
-            sorted_vars.extend(sorted(symbol_part,
-                                      key=lambda i: default_sort_key(i[0])))
-        return [Tuple(*i) for i in sorted_vars]
+                merged.append([v, c])
+        return [Tuple(*i) for i in merged]
 
     def _eval_is_commutative(self):
         return self.expr.is_commutative

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/function.py | 1301 | 1366 | 28 | 1 | 13505


## Problem Statement

```
better canonicalization of variables of Derivative
Better canonicalization of `Derivative._sort_variable_count` will be had if any symbols, appearing after functions, that are not in the free symbols of the function, appear before the functions: `Derivative(f(x, y), x, f(y), x)` should equal `Derivative(f(x, y), x, x, f(y))`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/core/function.py** | 1347 | 1366| 215 | 215 | 24673 | 
| 2 | **1 sympy/core/function.py** | 1300 | 1346| 752 | 967 | 24673 | 
| 3 | **1 sympy/core/function.py** | 1279 | 1298| 222 | 1189 | 24673 | 
| 4 | **1 sympy/core/function.py** | 1460 | 1498| 418 | 1607 | 24673 | 
| 5 | **1 sympy/core/function.py** | 1103 | 1196| 847 | 2454 | 24673 | 
| 6 | **1 sympy/core/function.py** | 1438 | 1458| 143 | 2597 | 24673 | 
| 7 | **1 sympy/core/function.py** | 913 | 1081| 1802 | 4399 | 24673 | 
| 8 | **1 sympy/core/function.py** | 1198 | 1277| 817 | 5216 | 24673 | 
| 9 | 2 sympy/core/basic.py | 513 | 538| 216 | 5432 | 39648 | 
| 10 | **2 sympy/core/function.py** | 1368 | 1382| 145 | 5577 | 39648 | 
| 11 | **2 sympy/core/function.py** | 1083 | 1101| 120 | 5697 | 39648 | 
| 12 | **2 sympy/core/function.py** | 1932 | 2328| 570 | 6267 | 39648 | 
| 13 | **2 sympy/core/function.py** | 1500 | 1526| 233 | 6500 | 39648 | 
| 14 | **2 sympy/core/function.py** | 1384 | 1407| 305 | 6805 | 39648 | 
| 15 | 3 sympy/printing/conventions.py | 71 | 84| 107 | 6912 | 40252 | 
| 16 | 4 sympy/combinatorics/tensor_can.py | 835 | 855| 327 | 7239 | 53399 | 
| 17 | 5 sympy/physics/quantum/operator.py | 530 | 553| 176 | 7415 | 57782 | 
| 18 | **5 sympy/core/function.py** | 1528 | 1603| 833 | 8248 | 57782 | 
| 19 | **5 sympy/core/function.py** | 1818 | 1868| 347 | 8595 | 57782 | 
| 20 | 6 sympy/unify/core.py | 133 | 168| 258 | 8853 | 59777 | 
| 21 | 7 sympy/calculus/finite_diff.py | 366 | 414| 439 | 9292 | 64960 | 
| 22 | 8 sympy/integrals/heurisch.py | 440 | 492| 538 | 9830 | 71050 | 
| 23 | 8 sympy/core/basic.py | 1810 | 1853| 310 | 10140 | 71050 | 
| 24 | **8 sympy/core/function.py** | 1890 | 1903| 216 | 10356 | 71050 | 
| 25 | 9 sympy/solvers/deutils.py | 1 | 85| 745 | 11101 | 73569 | 
| 26 | 10 sympy/parsing/autolev/_listener_autolev_antlr.py | 721 | 862| 1637 | 12738 | 96680 | 
| 27 | **10 sympy/core/function.py** | 1637 | 1662| 208 | 12946 | 96680 | 
| **-> 28 <-** | **10 sympy/core/function.py** | 838 | 1603| 559 | 13505 | 96680 | 
| 29 | 11 sympy/printing/pretty/pretty.py | 320 | 355| 287 | 13792 | 117834 | 
| 30 | 12 sympy/simplify/radsimp.py | 186 | 211| 200 | 13992 | 127394 | 
| 31 | 12 sympy/combinatorics/tensor_can.py | 633 | 755| 1423 | 15415 | 127394 | 
| 32 | 13 sympy/polys/modulargcd.py | 396 | 459| 476 | 15891 | 145889 | 
| 33 | 13 sympy/parsing/autolev/_listener_autolev_antlr.py | 267 | 358| 1119 | 17010 | 145889 | 
| 34 | **13 sympy/core/function.py** | 1754 | 1816| 564 | 17574 | 145889 | 
| 35 | 14 sympy/core/symbol.py | 131 | 166| 186 | 17760 | 152276 | 
| 36 | 15 sympy/integrals/integrals.py | 144 | 246| 983 | 18743 | 166034 | 
| 37 | 16 sympy/integrals/prde.py | 986 | 1040| 675 | 19418 | 181965 | 
| 38 | 16 sympy/combinatorics/tensor_can.py | 756 | 834| 847 | 20265 | 181965 | 


## Patch

```diff
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -1298,72 +1298,101 @@ def _remove_derived_once(cls, v):
         return [i[0] if i[1] == 1 else i for i in v]
 
     @classmethod
-    def _sort_variable_count(cls, varcounts):
+    def _sort_variable_count(cls, vc):
         """
-        Sort (variable, count) pairs by variable, but disallow sorting of non-symbols.
+        Sort (variable, count) pairs into canonical order while
+        retaining order of variables that do not commute during
+        differentiation:
 
-        The count is not sorted. It is kept in the same order as the input
-        after sorting by variable.
-
-        When taking derivatives, the following rules usually hold:
-
-        * Derivative wrt different symbols commute.
-        * Derivative wrt different non-symbols commute.
-        * Derivatives wrt symbols and non-symbols don't commute.
+        * symbols and functions commute with each other
+        * derivatives commute with each other
+        * a derivative doesn't commute with anything it contains
+        * any other object is not allowed to commute if it has
+          free symbols in common with another object
 
         Examples
         ========
 
-        >>> from sympy import Derivative, Function, symbols
+        >>> from sympy import Derivative, Function, symbols, cos
         >>> vsort = Derivative._sort_variable_count
         >>> x, y, z = symbols('x y z')
         >>> f, g, h = symbols('f g h', cls=Function)
 
-        >>> vsort([(x, 3), (y, 2), (z, 1)])
-        [(x, 3), (y, 2), (z, 1)]
-
-        >>> vsort([(h(x), 1), (g(x), 1), (f(x), 1)])
-        [(f(x), 1), (g(x), 1), (h(x), 1)]
+        Contiguous items are collapsed into one pair:
 
-        >>> vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1), (f(x), 1)])
-        [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]
+        >>> vsort([(x, 1), (x, 1)])
+        [(x, 2)]
+        >>> vsort([(y, 1), (f(x), 1), (y, 1), (f(x), 1)])
+        [(y, 2), (f(x), 2)]
 
-        >>> vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)])
-        [(x, 1), (f(x), 1), (y, 1), (f(y), 1)]
+        Ordering is canonical.
 
-        >>> vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1), (h(x), 1), (y, 2), (x, 1)])
-        [(x, 2), (y, 1), (f(x), 1), (g(x), 1), (z, 1), (h(x), 1), (x, 1), (y, 2)]
+        >>> def vsort0(*v):
+        ...     # docstring helper to
+        ...     # change vi -> (vi, 0), sort, and return vi vals
+        ...     return [i[0] for i in vsort([(i, 0) for i in v])]
 
-        >>> vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)])
-        [(y, 1), (z, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]
+        >>> vsort0(y, x)
+        [x, y]
+        >>> vsort0(g(y), g(x), f(y))
+        [f(y), g(x), g(y)]
 
-        >>> vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)])
-        [(y, 2), (z, 1), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (x, 1), (y, 1), (z, 2), (z, 1)]
+        Symbols are sorted as far to the left as possible but never
+        move to the left of a derivative having the same symbol in
+        its variables; the same applies to AppliedUndef which are
+        always sorted after Symbols:
 
+        >>> dfx = f(x).diff(x)
+        >>> assert vsort0(dfx, y) == [y, dfx]
+        >>> assert vsort0(dfx, x) == [dfx, x]
         """
-        sorted_vars = []
-        symbol_part = []
-        non_symbol_part = []
-        for (v, c) in varcounts:
-            if not v.is_symbol:
-                if len(symbol_part) > 0:
-                    sorted_vars.extend(sorted(symbol_part,
-                                              key=lambda i: default_sort_key(i[0])))
-                    symbol_part = []
-                non_symbol_part.append((v, c))
+        from sympy.utilities.iterables import uniq, topological_sort
+        if not vc:
+            return []
+        vc = list(vc)
+        if len(vc) == 1:
+            return [Tuple(*vc[0])]
+        V = list(range(len(vc)))
+        E = []
+        v = lambda i: vc[i][0]
+        D = Dummy()
+        def _block(d, v, wrt=False):
+            # return True if v should not come before d else False
+            if d == v:
+                return wrt
+            if d.is_Symbol:
+                return False
+            if isinstance(d, Derivative):
+                # a derivative blocks if any of it's variables contain
+                # v; the wrt flag will return True for an exact match
+                # and will cause an AppliedUndef to block if v is in
+                # the arguments
+                if any(_block(k, v, wrt=True)
+                        for k, _ in d.variable_count):
+                    return True
+                return False
+            if not wrt and isinstance(d, AppliedUndef):
+                return False
+            if v.is_Symbol:
+                return v in d.free_symbols
+            if isinstance(v, AppliedUndef):
+                return _block(d.xreplace({v: D}), D)
+            return d.free_symbols & v.free_symbols
+        for i in range(len(vc)):
+            for j in range(i):
+                if _block(v(j), v(i)):
+                    E.append((j,i))
+        # this is the default ordering to use in case of ties
+        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
+        ix = topological_sort((V, E), key=lambda i: O[v(i)])
+        # merge counts of contiguously identical items
+        merged = []
+        for v, c in [vc[i] for i in ix]:
+            if merged and merged[-1][0] == v:
+                merged[-1][1] += c
             else:
-                if len(non_symbol_part) > 0:
-                    sorted_vars.extend(sorted(non_symbol_part,
-                                              key=lambda i: default_sort_key(i[0])))
-                    non_symbol_part = []
-                symbol_part.append((v, c))
-        if len(non_symbol_part) > 0:
-            sorted_vars.extend(sorted(non_symbol_part,
-                                      key=lambda i: default_sort_key(i[0])))
-        if len(symbol_part) > 0:
-            sorted_vars.extend(sorted(symbol_part,
-                                      key=lambda i: default_sort_key(i[0])))
-        return [Tuple(*i) for i in sorted_vars]
+                merged.append([v, c])
+        return [Tuple(*i) for i in merged]
 
     def _eval_is_commutative(self):
         return self.expr.is_commutative

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_function.py b/sympy/core/tests/test_function.py
--- a/sympy/core/tests/test_function.py
+++ b/sympy/core/tests/test_function.py
@@ -1,9 +1,11 @@
 from sympy import (Lambda, Symbol, Function, Derivative, Subs, sqrt,
         log, exp, Rational, Float, sin, cos, acos, diff, I, re, im,
         E, expand, pi, O, Sum, S, polygamma, loggamma, expint,
-        Tuple, Dummy, Eq, Expr, symbols, nfloat, Piecewise, Indexed)
+        Tuple, Dummy, Eq, Expr, symbols, nfloat, Piecewise, Indexed,
+        Matrix, Basic)
 from sympy.utilities.pytest import XFAIL, raises
 from sympy.abc import t, w, x, y, z
+from sympy.core.basic import _aresame
 from sympy.core.function import PoleError, _mexpand
 from sympy.core.sympify import sympify
 from sympy.sets.sets import FiniteSet
@@ -643,30 +645,82 @@ def test_straight_line():
 
 def test_sort_variable():
     vsort = Derivative._sort_variable_count
-
+    def vsort0(*v, **kw):
+        reverse = kw.get('reverse', False)
+        return [i[0] for i in vsort([(i, 0) for i in (
+            reversed(v) if reverse else v)])]
+
+    for R in range(2):
+        assert vsort0(y, x, reverse=R) == [x, y]
+        assert vsort0(f(x), x, reverse=R) == [x, f(x)]
+        assert vsort0(f(y), f(x), reverse=R) == [f(x), f(y)]
+        assert vsort0(g(x), f(y), reverse=R) == [f(y), g(x)]
+        assert vsort0(f(x, y), f(x), reverse=R) == [f(x), f(x, y)]
+        fx = f(x).diff(x)
+        assert vsort0(fx, y, reverse=R) == [y, fx]
+        fy = f(y).diff(y)
+        assert vsort0(fy, fx, reverse=R) == [fx, fy]
+        fxx = fx.diff(x)
+        assert vsort0(fxx, fx, reverse=R) == [fx, fxx]
+        assert vsort0(Basic(x), f(x), reverse=R) == [f(x), Basic(x)]
+        assert vsort0(Basic(y), Basic(x), reverse=R) == [Basic(x), Basic(y)]
+        assert vsort0(Basic(y, z), Basic(x), reverse=R) == [
+            Basic(x), Basic(y, z)]
+        assert vsort0(fx, x, reverse=R) == [
+            x, fx] if R else [fx, x]
+        assert vsort0(Basic(x), x, reverse=R) == [
+            x, Basic(x)] if R else [Basic(x), x]
+        assert vsort0(Basic(f(x)), f(x), reverse=R) == [
+            f(x), Basic(f(x))] if R else [Basic(f(x)), f(x)]
+        assert vsort0(Basic(x, z), Basic(x), reverse=R) == [
+            Basic(x), Basic(x, z)] if R else [Basic(x, z), Basic(x)]
+    assert vsort([]) == []
+    assert _aresame(vsort([(x, 1)]), [Tuple(x, 1)])
+    assert vsort([(x, y), (x, z)]) == [(x, y + z)]
+    assert vsort([(y, 1), (x, 1 + y)]) == [(x, 1 + y), (y, 1)]
+    # coverage complete; legacy tests below
     assert vsort([(x, 3), (y, 2), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
-
-    assert vsort([(h(x), 1), (g(x), 1), (f(x), 1)]) == [(f(x), 1), (g(x), 1), (h(x), 1)]
-
-    assert vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1), (f(x), 1)]) == [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]
-
-    assert vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)]) == [(x, 1), (f(x), 1), (y, 1), (f(y), 1)]
-
-    assert vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1), (h(x), 1), (y, 2), (x, 1)]) == [(x, 2), (y, 1), (f(x), 1), (g(x), 1), (z, 1), (h(x), 1), (x, 1), (y, 2)]
-
-    assert vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]) == [(y, 1), (z, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]
-
-    assert vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1),
-    (z, 2), (z, 1), (y, 1), (x, 1)]) == [(y, 2), (z, 1), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (x, 1), (y, 1), (z, 2), (z, 1)]
-
-
-    assert vsort(((y, 2), (x, 1), (y, 1), (x, 1))) == [(x, 1), (x, 1), (y, 2), (y, 1)]
-
+    assert vsort([(h(x), 1), (g(x), 1), (f(x), 1)]) == [
+        (f(x), 1), (g(x), 1), (h(x), 1)]
+    assert vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1),
+        (f(x), 1)]) == [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1),
+        (h(x), 1)]
+    assert vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)]) == [(x, 1),
+        (y, 1), (f(x), 1), (f(y), 1)]
+    assert vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1),
+        (h(x), 1), (y, 2), (x, 1)]) == [(x, 3), (y, 3), (z, 1),
+        (f(x), 1), (g(x), 1), (h(x), 1)]
+    assert vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1),
+        (g(x), 1)]) == [(x, 1), (y, 1), (z, 1), (f(x), 2), (g(x), 1)]
+    assert vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2),
+        (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)]) == [(x, 3), (y, 3),
+        (z, 4), (f(x), 3), (g(x), 1)]
+    assert vsort(((y, 2), (x, 1), (y, 1), (x, 1))) == [(x, 2), (y, 3)]
     assert isinstance(vsort([(x, 3), (y, 2), (z, 1)])[0], Tuple)
+    assert vsort([(x, 1), (f(x), 1), (x, 1)]) == [(x, 2), (f(x), 1)]
+    assert vsort([(y, 2), (x, 3), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
+    assert vsort([(h(y), 1), (g(x), 1), (f(x), 1)]) == [
+        (f(x), 1), (g(x), 1), (h(y), 1)]
+    assert vsort([(x, 1), (y, 1), (x, 1)]) == [(x, 2), (y, 1)]
+    assert vsort([(f(x), 1), (f(y), 1), (f(x), 1)]) == [
+        (f(x), 2), (f(y), 1)]
+    dfx = f(x).diff(x)
+    self = [(dfx, 1), (x, 1)]
+    assert vsort(self) == self
+    assert vsort([
+        (dfx, 1), (y, 1), (f(x), 1), (x, 1), (f(y), 1), (x, 1)]) == [
+        (y, 1), (f(x), 1), (f(y), 1), (dfx, 1), (x, 2)]
+    dfy = f(y).diff(y)
+    assert vsort([(dfy, 1), (dfx, 1)]) == [(dfx, 1), (dfy, 1)]
+    d2fx = dfx.diff(x)
+    assert vsort([(d2fx, 1), (dfx, 1)]) == [(dfx, 1), (d2fx, 1)]
+
 
 def test_multiple_derivative():
     # Issue #15007
-    assert f(x,y).diff(y,y,x,y,x) == Derivative(f(x, y), (x, 2), (y, 3))
+    assert f(x, y).diff(y, y, x, y, x
+        ) == Derivative(f(x, y), (x, 2), (y, 3))
+
 
 def test_unhandled():
     class MyExpr(Expr):
@@ -677,8 +731,8 @@ def _eval_derivative(self, s):
                 return None
 
     expr = MyExpr(x, y, z)
-    assert diff(expr, x, y, f(x), z) == Derivative(expr, f(x), z)
-    assert diff(expr, f(x), x) == Derivative(expr, f(x), x)
+    assert diff(expr, x, y, f(x), z) == Derivative(expr, z, f(x))
+    assert diff(expr, f(x), x) == Derivative(expr, x, f(x))
 
 
 def test_nfloat():
@@ -998,3 +1052,19 @@ def test_undefined_function_eval():
     assert sympify(expr) == expr
     assert type(sympify(expr)).fdiff.__name__ == "<lambda>"
     assert expr.diff(t) == cos(t)
+
+
+def test_issue_15241():
+    F = f(x)
+    Fx = F.diff(x)
+    assert (F + x*Fx).diff(x, Fx) == 2
+    assert (F + x*Fx).diff(Fx, x) == 1
+    assert (x*F + x*Fx*F).diff(F, x) == x*Fx.diff(x) + Fx + 1
+    assert (x*F + x*Fx*F).diff(x, F) == x*Fx.diff(x) + Fx + 1
+    y = f(x)
+    G = f(y)
+    Gy = G.diff(y)
+    assert (G + y*Gy).diff(y, Gy) == 2
+    assert (G + y*Gy).diff(Gy, y) == 1
+    assert (y*G + y*Gy*G).diff(G, y) == y*Gy.diff(y) + Gy + 1
+    assert (y*G + y*Gy*G).diff(y, G) == y*Gy.diff(y) + Gy + 1

```


## Code snippets

### 1 - sympy/core/function.py:

Start line: 1347, End line: 1366

```python
class Derivative(Expr):

    @classmethod
    def _sort_variable_count(cls, varcounts):
        # ... other code
        for (v, c) in varcounts:
            if not v.is_symbol:
                if len(symbol_part) > 0:
                    sorted_vars.extend(sorted(symbol_part,
                                              key=lambda i: default_sort_key(i[0])))
                    symbol_part = []
                non_symbol_part.append((v, c))
            else:
                if len(non_symbol_part) > 0:
                    sorted_vars.extend(sorted(non_symbol_part,
                                              key=lambda i: default_sort_key(i[0])))
                    non_symbol_part = []
                symbol_part.append((v, c))
        if len(non_symbol_part) > 0:
            sorted_vars.extend(sorted(non_symbol_part,
                                      key=lambda i: default_sort_key(i[0])))
        if len(symbol_part) > 0:
            sorted_vars.extend(sorted(symbol_part,
                                      key=lambda i: default_sort_key(i[0])))
        return [Tuple(*i) for i in sorted_vars]
```
### 2 - sympy/core/function.py:

Start line: 1300, End line: 1346

```python
class Derivative(Expr):

    @classmethod
    def _sort_variable_count(cls, varcounts):
        """
        Sort (variable, count) pairs by variable, but disallow sorting of non-symbols.

        The count is not sorted. It is kept in the same order as the input
        after sorting by variable.

        When taking derivatives, the following rules usually hold:

        * Derivative wrt different symbols commute.
        * Derivative wrt different non-symbols commute.
        * Derivatives wrt symbols and non-symbols don't commute.

        Examples
        ========

        >>> from sympy import Derivative, Function, symbols
        >>> vsort = Derivative._sort_variable_count
        >>> x, y, z = symbols('x y z')
        >>> f, g, h = symbols('f g h', cls=Function)

        >>> vsort([(x, 3), (y, 2), (z, 1)])
        [(x, 3), (y, 2), (z, 1)]

        >>> vsort([(h(x), 1), (g(x), 1), (f(x), 1)])
        [(f(x), 1), (g(x), 1), (h(x), 1)]

        >>> vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1), (f(x), 1)])
        [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]

        >>> vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)])
        [(x, 1), (f(x), 1), (y, 1), (f(y), 1)]

        >>> vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1), (h(x), 1), (y, 2), (x, 1)])
        [(x, 2), (y, 1), (f(x), 1), (g(x), 1), (z, 1), (h(x), 1), (x, 1), (y, 2)]

        >>> vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)])
        [(y, 1), (z, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]

        >>> vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)])
        [(y, 2), (z, 1), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (x, 1), (y, 1), (z, 2), (z, 1)]

        """
        sorted_vars = []
        symbol_part = []
        non_symbol_part = []
        # ... other code
```
### 3 - sympy/core/function.py:

Start line: 1279, End line: 1298

```python
class Derivative(Expr):

    def __new__(cls, expr, *variables, **kwargs):
        # ... other code

        if unhandled_variable_count:
            unhandled_variable_count = cls._sort_variable_count(unhandled_variable_count)
            expr = Expr.__new__(cls, expr, *unhandled_variable_count)
        else:
            # We got a Derivative at the end of it all, and we rebuild it by
            # sorting its variables.
            if isinstance(expr, Derivative):
                expr = cls(
                    expr.args[0], *cls._sort_variable_count(expr.args[1:])
                )

        if (nderivs > 1) == True and kwargs.get('simplify', True):
            from sympy.core.exprtools import factor_terms
            from sympy.simplify.simplify import signsimp
            expr = factor_terms(signsimp(expr))
        return expr

    @classmethod
    def _remove_derived_once(cls, v):
        return [i[0] if i[1] == 1 else i for i in v]
```
### 4 - sympy/core/function.py:

Start line: 1460, End line: 1498

```python
class Derivative(Expr):

    def _eval_subs(self, old, new):
        if old in self.variables and not new._diff_wrt:
            # issue 4719
            return Subs(self, old, new)
        # If both are Derivatives with the same expr, check if old is
        # equivalent to self or if old is a subderivative of self.
        if old.is_Derivative and old.expr == self.expr:
            # Check if canonical order of variables is equal.
            old_vars = Counter(dict(reversed(old.variable_count)))
            self_vars = Counter(dict(reversed(self.variable_count)))
            if old_vars == self_vars:
                return new

            # collections.Counter doesn't have __le__
            def _subset(a, b):
                return all((a[i] <= b[i]) == True for i in a)

            if _subset(old_vars, self_vars):
                return Derivative(new, *(self_vars - old_vars).items())

        # Check whether the substitution (old, new) cannot be done inside
        # Derivative(expr, vars). Disallowed:
        # (1) changing expr by introducing a variable among vars
        # (2) changing vars by introducing a variable contained in expr
        old_symbols = (old.free_symbols if isinstance(old.free_symbols, set)
            else set())
        new_symbols = (new.free_symbols if isinstance(new.free_symbols, set)
            else set())
        introduced_symbols = new_symbols - old_symbols
        args_subbed = tuple(x._subs(old, new) for x in self.args)
        if ((self.args[0] != args_subbed[0] and
            len(set(self.variables) & introduced_symbols) > 0
            ) or
            (self.args[1:] != args_subbed[1:] and
            len(self.free_symbols & introduced_symbols) > 0
            )):
            return Subs(self, old, new)
        else:
            return Derivative(*args_subbed)
```
### 5 - sympy/core/function.py:

Start line: 1103, End line: 1196

```python
class Derivative(Expr):

    def __new__(cls, expr, *variables, **kwargs):

        from sympy.matrices.common import MatrixCommon
        from sympy import Integer
        from sympy.tensor.array import Array, NDimArray, derive_by_array
        from sympy.utilities.misc import filldedent

        expr = sympify(expr)
        try:
            has_symbol_set = isinstance(expr.free_symbols, set)
        except AttributeError:
            has_symbol_set = False
        if not has_symbol_set:
            raise ValueError(filldedent('''
                Since there are no variables in the expression %s,
                it cannot be differentiated.''' % expr))

        # There are no variables, we differentiate wrt all of the free symbols
        # in expr.
        if not variables:
            variables = expr.free_symbols
            if len(variables) != 1:
                if expr.is_number:
                    return S.Zero
                if len(variables) == 0:
                    raise ValueError(filldedent('''
                        Since there are no variables in the expression,
                        the variable(s) of differentiation must be supplied
                        to differentiate %s''' % expr))
                else:
                    raise ValueError(filldedent('''
                        Since there is more than one variable in the
                        expression, the variable(s) of differentiation
                        must be supplied to differentiate %s''' % expr))

        # Standardize the variables by sympifying them:
        variables = list(sympify(variables))

        # Split the list of variables into a list of the variables we are diff
        # wrt, where each element of the list has the form (s, count) where
        # s is the entity to diff wrt and count is the order of the
        # derivative.
        variable_count = []
        j = 0
        array_likes = (tuple, list, Tuple)

        for i, v in enumerate(variables):
            if isinstance(v, Integer):
                count = v
                if i == 0:
                    raise ValueError("First variable cannot be a number: %i" % v)
                prev, prevcount = variable_count[j-1]
                if prevcount != 1:
                    raise TypeError("tuple {0} followed by number {1}".format((prev, prevcount), v))
                if count == 0:
                    j -= 1
                    variable_count.pop()
                else:
                    variable_count[j-1] = Tuple(prev, count)
            else:
                if isinstance(v, array_likes):
                    if len(v) == 0:
                        # Ignore empty tuples: Derivative(expr, ... , (), ... )
                        continue
                    if isinstance(v[0], array_likes):
                        # Derive by array: Derivative(expr, ... , [[x, y, z]], ... )
                        if len(v) == 1:
                            v = Array(v[0])
                            count = 1
                        else:
                            v, count = v
                            v = Array(v)
                    else:
                        v, count = v
                else:
                    count = S(1)
                if count == 0:
                    continue
                if not v._diff_wrt:
                    last_digit = int(str(count)[-1])
                    ordinal = 'st' if last_digit == 1 else 'nd' if last_digit == 2 else 'rd' if last_digit == 3 else 'th'
                    raise ValueError(filldedent('''
                    Can\'t calculate %s%s derivative wrt %s.''' % (count, ordinal, v)))
                if j != 0 and v == variable_count[-1][0]:
                    prev, prevcount = variable_count[j-1]
                    variable_count[-1] = Tuple(prev, prevcount + count)
                else:
                    variable_count.append(Tuple(v, count))
                    j += 1

        # We make a special case for 0th derivative, because there is no
        # good way to unambiguously print this.
        if len(variable_count) == 0:
            return expr
        # ... other code
```
### 6 - sympy/core/function.py:

Start line: 1438, End line: 1458

```python
class Derivative(Expr):

    @property
    def expr(self):
        return self._args[0]

    @property
    def variables(self):
        # TODO: deprecate?
        # TODO: support for `d^n`?
        return tuple(v for v, count in self.variable_count if count.is_Integer for i in (range(count) if count.is_Integer else [1]))

    @property
    def variable_count(self):
        return self._args[1:]

    @property
    def derivative_count(self):
        return sum([count for var, count in self.variable_count], 0)

    @property
    def free_symbols(self):
        return self.expr.free_symbols
```
### 7 - sympy/core/function.py:

Start line: 913, End line: 1081

```python
class Derivative(Expr):
    """
    Carries out differentiation of the given expression with respect to symbols.

    expr must define ._eval_derivative(symbol) method that returns
    the differentiation result. This function only needs to consider the
    non-trivial case where expr contains symbol and it should call the diff()
    method internally (not _eval_derivative); Derivative should be the only
    one to call _eval_derivative.

    Simplification of high-order derivatives:

    Because there can be a significant amount of simplification that can be
    done when multiple differentiations are performed, results will be
    automatically simplified in a fairly conservative fashion unless the
    keyword ``simplify`` is set to False.

        >>> from sympy import sqrt, diff
        >>> from sympy.abc import x
        >>> e = sqrt((x + 1)**2 + x)
        >>> diff(e, (x, 5), simplify=False).count_ops()
        136
        >>> diff(e, (x, 5)).count_ops()
        30

    Ordering of variables:

    If evaluate is set to True and the expression can not be evaluated, the
    list of differentiation symbols will be sorted, that is, the expression is
    assumed to have continuous derivatives up to the order asked. This sorting
    assumes that derivatives wrt Symbols commute, derivatives wrt non-Symbols
    commute, but Symbol and non-Symbol derivatives don't commute with each
    other.

    Derivative wrt non-Symbols:

    This class also allows derivatives wrt non-Symbols that have _diff_wrt
    set to True, such as Function and Derivative. When a derivative wrt a non-
    Symbol is attempted, the non-Symbol is temporarily converted to a Symbol
    while the differentiation is performed.

    Note that this may seem strange, that Derivative allows things like
    f(g(x)).diff(g(x)), or even f(cos(x)).diff(cos(x)).  The motivation for
    allowing this syntax is to make it easier to work with variational calculus
    (i.e., the Euler-Lagrange method).  The best way to understand this is that
    the action of derivative with respect to a non-Symbol is defined by the
    above description:  the object is substituted for a Symbol and the
    derivative is taken with respect to that.  This action is only allowed for
    objects for which this can be done unambiguously, for example Function and
    Derivative objects.  Note that this leads to what may appear to be
    mathematically inconsistent results.  For example::

        >>> from sympy import cos, sin, sqrt
        >>> from sympy.abc import x
        >>> (2*cos(x)).diff(cos(x))
        2
        >>> (2*sqrt(1 - sin(x)**2)).diff(cos(x))
        0

    This appears wrong because in fact 2*cos(x) and 2*sqrt(1 - sin(x)**2) are
    identically equal.  However this is the wrong way to think of this.  Think
    of it instead as if we have something like this::

        >>> from sympy.abc import c, s, u, x
        >>> def F(u):
        ...     return 2*u
        ...
        >>> def G(u):
        ...     return 2*sqrt(1 - u**2)
        ...
        >>> F(cos(x))
        2*cos(x)
        >>> G(sin(x))
        2*sqrt(-sin(x)**2 + 1)
        >>> F(c).diff(c)
        2
        >>> F(cos(x)).diff(cos(x))
        2
        >>> G(s).diff(c)
        0
        >>> G(sin(x)).diff(cos(x))
        0

    Here, the Symbols c and s act just like the functions cos(x) and sin(x),
    respectively. Think of 2*cos(x) as f(c).subs(c, cos(x)) (or f(c) *at*
    c = cos(x)) and 2*sqrt(1 - sin(x)**2) as g(s).subs(s, sin(x)) (or g(s) *at*
    s = sin(x)), where f(u) == 2*u and g(u) == 2*sqrt(1 - u**2).  Here, we
    define the function first and evaluate it at the function, but we can
    actually unambiguously do this in reverse in SymPy, because
    expr.subs(Function, Symbol) is well-defined:  just structurally replace the
    function everywhere it appears in the expression.

    This is the same notational convenience used in the Euler-Lagrange method
    when one says F(t, f(t), f'(t)).diff(f(t)).  What is actually meant is
    that the expression in question is represented by some F(t, u, v) at u =
    f(t) and v = f'(t), and F(t, f(t), f'(t)).diff(f(t)) simply means F(t, u,
    v).diff(u) at u = f(t).

    We do not allow derivatives to be taken with respect to expressions where this
    is not so well defined.  For example, we do not allow expr.diff(x*y)
    because there are multiple ways of structurally defining where x*y appears
    in an expression, some of which may surprise the reader (for example, a
    very strict definition would have that (x*y*z).diff(x*y) == 0).

        >>> from sympy.abc import x, y, z
        >>> (x*y*z).diff(x*y)
        Traceback (most recent call last):
        ...
        ValueError: Can't differentiate wrt the variable: x*y, 1

    Note that this definition also fits in nicely with the definition of the
    chain rule.  Note how the chain rule in SymPy is defined using unevaluated
    Subs objects::

        >>> from sympy import symbols, Function
        >>> f, g = symbols('f g', cls=Function)
        >>> f(2*g(x)).diff(x)
        2*Derivative(g(x), x)*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, 2*g(x))
        >>> f(g(x)).diff(x)
        Derivative(g(x), x)*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, g(x))

    Finally, note that, to be consistent with variational calculus, and to
    ensure that the definition of substituting a Function for a Symbol in an
    expression is well-defined, derivatives of functions are assumed to not be
    related to the function.  In other words, we have::

        >>> from sympy import diff
        >>> diff(f(x), x).diff(f(x))
        0

    The same is true for derivatives of different orders::

        >>> diff(f(x), x, 2).diff(diff(f(x), x, 1))
        0
        >>> diff(f(x), x, 1).diff(diff(f(x), x, 2))
        0

    Note, any class can allow derivatives to be taken with respect to itself.
    See the docstring of Expr._diff_wrt.

    Examples
    ========

    Some basic examples:

        >>> from sympy import Derivative, Symbol, Function
        >>> f = Function('f')
        >>> g = Function('g')
        >>> x = Symbol('x')
        >>> y = Symbol('y')

        >>> Derivative(x**2, x, evaluate=True)
        2*x
        >>> Derivative(Derivative(f(x,y), x), y)
        Derivative(f(x, y), x, y)
        >>> Derivative(f(x), x, 3)
        Derivative(f(x), (x, 3))
        >>> Derivative(f(x, y), y, x, evaluate=True)
        Derivative(f(x, y), x, y)

    Now some derivatives wrt functions:

        >>> Derivative(f(x)**2, f(x), evaluate=True)
        2*f(x)
        >>> Derivative(f(g(x)), x, evaluate=True)
        Derivative(g(x), x)*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, g(x))

    """

    is_Derivative = True
```
### 8 - sympy/core/function.py:

Start line: 1198, End line: 1277

```python
class Derivative(Expr):

    def __new__(cls, expr, *variables, **kwargs):
        # ... other code

        evaluate = kwargs.get('evaluate', False)

        # Look for a quick exit if there are symbols that don't appear in
        # expression at all. Note, this cannot check non-symbols like
        # functions and Derivatives as those can be created by intermediate
        # derivatives.
        if evaluate and all(isinstance(sc[0], Symbol) for sc in variable_count):
            symbol_set = set(sc[0] for sc in variable_count if sc[1].is_positive)
            if symbol_set.difference(expr.free_symbols):
                if isinstance(expr, (MatrixCommon, NDimArray)):
                    return expr.zeros(*expr.shape)
                else:
                    return S.Zero

        # If we can't compute the derivative of expr (but we wanted to) and
        # expr is itself not a Derivative, finish building an unevaluated
        # derivative class by calling Expr.__new__.
        if (not (hasattr(expr, '_eval_derivative') and evaluate) and
           (not isinstance(expr, Derivative))):
            # If we wanted to evaluate, we sort the variables into standard
            # order for later comparisons. This is too aggressive if evaluate
            # is False, so we don't do it in that case.
            if evaluate:
                #TODO: check if assumption of discontinuous derivatives exist
                variable_count = cls._sort_variable_count(variable_count)
            obj = Expr.__new__(cls, expr, *variable_count)
            return obj

        # Compute the derivative now by repeatedly calling the
        # _eval_derivative method of expr for each variable. When this method
        # returns None, the derivative couldn't be computed wrt that variable
        # and we save the variable for later.
        unhandled_variable_count = []

        # Once we encouter a non_symbol that is unhandled, we stop taking
        # derivatives entirely. This is because derivatives wrt functions
        # don't commute with derivatives wrt symbols and we can't safely
        # continue.
        unhandled_non_symbol = False
        nderivs = 0  # how many derivatives were performed
        for v, count in variable_count:
            is_symbol = v.is_symbol

            if unhandled_non_symbol:
                obj = None
            elif (count < 0) == True:
                obj = None
            else:
                if isinstance(v, (Iterable, Tuple, MatrixCommon, NDimArray)):
                    # Treat derivatives by arrays/matrices as much as symbols.
                    is_symbol = True
                if not is_symbol:
                    new_v = Dummy('xi_%i' % i, dummy_index=hash(v))
                    expr = expr.xreplace({v: new_v})
                    old_v = v
                    v = new_v
                # Evaluate the derivative `n` times.  If
                # `_eval_derivative_n_times` is not overridden by the current
                # object, the default in `Basic` will call a loop over
                # `_eval_derivative`:
                obj = expr._eval_derivative_n_times(v, count)
                nderivs += count
                if not is_symbol:
                    if obj is not None:
                        if not old_v.is_symbol and obj.is_Derivative:
                            # Derivative evaluated at a point that is not a
                            # symbol, let subs check if this is okay to replace
                            obj = obj.subs(v, old_v)
                        else:
                            obj = obj.xreplace({v: old_v})
                    v = old_v

            if obj is None:
                unhandled_variable_count.append(Tuple(v, count))
                if not is_symbol:
                    unhandled_non_symbol = True
            elif obj is S.Zero:
                return S.Zero
            else:
                expr = obj
        # ... other code
```
### 9 - sympy/core/basic.py:

Start line: 513, End line: 538

```python
class Basic(with_metaclass(ManagedProperties)):

    @property
    def canonical_variables(self):
        """Return a dictionary mapping any variable defined in
        ``self.variables`` as underscore-suffixed numbers
        corresponding to their position in ``self.variables``. Enough
        underscores are added to ensure that there will be no clash with
        existing free symbols.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: 0_}
        """
        from sympy import Symbol
        if not hasattr(self, 'variables'):
            return {}
        u = "_"
        while any(str(s).endswith(u) for s in self.free_symbols):
            u += "_"
        name = '%%i%s' % u
        V = self.variables
        return dict(list(zip(V, [Symbol(name % i, **v.assumptions0)
            for i, v in enumerate(V)])))
```
### 10 - sympy/core/function.py:

Start line: 1368, End line: 1382

```python
class Derivative(Expr):

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def _eval_derivative_n_times(self, s, n):
        from sympy import Integer
        if isinstance(n, (int, Integer)):
            # TODO: it would be desirable to squash `_eval_derivative` into
            # this code.
            return super(Derivative, self)._eval_derivative_n_times(s, n)
        dict_var_count = dict(self.variable_count)
        if s in dict_var_count:
            dict_var_count[s] += n
        else:
            dict_var_count[s] = n
        return Derivative(self.expr, *dict_var_count.items())
```
### 11 - sympy/core/function.py:

Start line: 1083, End line: 1101

```python
class Derivative(Expr):

    @property
    def _diff_wrt(self):
        """Allow derivatives wrt Derivatives if it contains a function.

        Examples
        ========

            >>> from sympy import Function, Symbol, Derivative
            >>> f = Function('f')
            >>> x = Symbol('x')
            >>> Derivative(f(x),x)._diff_wrt
            True
            >>> Derivative(x**2,x)._diff_wrt
            False
        """
        if self.expr.is_Function:
            return True
        else:
            return False
```
### 12 - sympy/core/function.py:

Start line: 1932, End line: 2328

```python
def diff(f, *symbols, **kwargs):
    """
    Differentiate f with respect to symbols.

    This is just a wrapper to unify .diff() and the Derivative class; its
    interface is similar to that of integrate().  You can use the same
    shortcuts for multiple variables as with Derivative.  For example,
    diff(f(x), x, x, x) and diff(f(x), x, 3) both return the third derivative
    of f(x).

    You can pass evaluate=False to get an unevaluated Derivative class.  Note
    that if there are 0 symbols (such as diff(f(x), x, 0), then the result will
    be the function (the zeroth derivative), even if evaluate=False.

    Examples
    ========

    >>> from sympy import sin, cos, Function, diff
    >>> from sympy.abc import x, y
    >>> f = Function('f')

    >>> diff(sin(x), x)
    cos(x)
    >>> diff(f(x), x, x, x)
    Derivative(f(x), (x, 3))
    >>> diff(f(x), x, 3)
    Derivative(f(x), (x, 3))
    >>> diff(sin(x)*cos(y), x, 2, y, 2)
    sin(x)*cos(y)

    >>> type(diff(sin(x), x))
    cos
    >>> type(diff(sin(x), x, evaluate=False))
    <class 'sympy.core.function.Derivative'>
    >>> type(diff(sin(x), x, 0))
    sin
    >>> type(diff(sin(x), x, 0, evaluate=False))
    sin

    >>> diff(sin(x))
    cos(x)
    >>> diff(sin(x*y))
    Traceback (most recent call last):
    ...
    ValueError: specify differentiation variables to differentiate sin(x*y)

    Note that ``diff(sin(x))`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    References
    ==========

    http://reference.wolfram.com/legacy/v5_2/Built-inFunctions/AlgebraicComputation/Calculus/D.html

    See Also
    ========

    Derivative
    sympy.geometry.util.idiff: computes the derivative implicitly

    """
    kwargs.setdefault('evaluate', True)
    try:
        return f._eval_diff(*symbols, **kwargs)
    except AttributeError:
        pass
    return Derivative(f, *symbols, **kwargs)


def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    # ... other code
```
### 13 - sympy/core/function.py:

Start line: 1500, End line: 1526

```python
class Derivative(Expr):

    def _eval_lseries(self, x, logx):
        dx = self.variables
        for term in self.expr.lseries(x, logx=logx):
            yield self.func(term, *dx)

    def _eval_nseries(self, x, n, logx):
        arg = self.expr.nseries(x, n=n, logx=logx)
        o = arg.getO()
        dx = self.variables
        rv = [self.func(a, *dx) for a in Add.make_args(arg.removeO())]
        if o:
            rv.append(o/x)
        return Add(*rv)

    def _eval_as_leading_term(self, x):
        series_gen = self.expr.lseries(x)
        d = S.Zero
        for leading_term in series_gen:
            d = diff(leading_term, *self.variables)
            if d != 0:
                break
        return d

    def _sage_(self):
        import sage.all as sage
        args = [arg._sage_() for arg in self.args]
        return sage.derivative(*args)
```
### 14 - sympy/core/function.py:

Start line: 1384, End line: 1407

```python
class Derivative(Expr):

    def _eval_derivative(self, v):
        # If the variable s we are diff wrt is not in self.variables, we
        # assume that we might be able to take the derivative.
        if v not in self.variables:
            obj = self.expr.diff(v)
            if obj is S.Zero:
                return S.Zero
            if isinstance(obj, Derivative):
                return obj.func(obj.expr, *(self.variable_count + obj.variable_count))
            # The derivative wrt s could have simplified things such that the
            # derivative wrt things in self.variables can now be done. Thus,
            # we set evaluate=True to see if there are any other derivatives
            # that can be done. The most common case is when obj is a simple
            # number so that the derivative wrt anything else will vanish.
            return self.func(obj, *self.variables, evaluate=True)
        # In this case s was in self.variables so the derivatve wrt s has
        # already been attempted and was not computed, either because it
        # couldn't be or evaluate=False originally.
        variable_count = list(self.variable_count)
        if variable_count[-1][0] == v:
            variable_count[-1] = Tuple(v, variable_count[-1][1] + 1)
        else:
            variable_count.append(Tuple(v, S(1)))
        return self.func(self.expr, *variable_count, evaluate=False)
```
### 18 - sympy/core/function.py:

Start line: 1528, End line: 1603

```python
class Derivative(Expr):

    def as_finite_difference(self, points=1, x0=None, wrt=None):
        """ Expresses a Derivative instance as a finite difference.

        Parameters
        ==========
        points : sequence or coefficient, optional
            If sequence: discrete values (length >= order+1) of the
            independent variable used for generating the finite
            difference weights.
            If it is a coefficient, it will be used as the step-size
            for generating an equidistant sequence of length order+1
            centered around ``x0``. Default: 1 (step-size 1)
        x0 : number or Symbol, optional
            the value of the independent variable (``wrt``) at which the
            derivative is to be approximated. Default: same as ``wrt``.
        wrt : Symbol, optional
            "with respect to" the variable for which the (partial)
            derivative is to be approximated for. If not provided it
            is required that the derivative is ordinary. Default: ``None``.


        Examples
        ========
        >>> from sympy import symbols, Function, exp, sqrt, Symbol
        >>> x, h = symbols('x h')
        >>> f = Function('f')
        >>> f(x).diff(x).as_finite_difference()
        -f(x - 1/2) + f(x + 1/2)

        The default step size and number of points are 1 and
        ``order + 1`` respectively. We can change the step size by
        passing a symbol as a parameter:

        >>> f(x).diff(x).as_finite_difference(h)
        -f(-h/2 + x)/h + f(h/2 + x)/h

        We can also specify the discretized values to be used in a
        sequence:

        >>> f(x).diff(x).as_finite_difference([x, x+h, x+2*h])
        -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

        The algorithm is not restricted to use equidistant spacing, nor
        do we need to make the approximation around ``x0``, but we can get
        an expression estimating the derivative at an offset:

        >>> e, sq2 = exp(1), sqrt(2)
        >>> xl = [x-h, x+h, x+e*h]
        >>> f(x).diff(x, 1).as_finite_difference(xl, x+h*sq2)  # doctest: +ELLIPSIS
        2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/...

        Partial derivatives are also supported:

        >>> y = Symbol('y')
        >>> d2fdxdy=f(x,y).diff(x,y)
        >>> d2fdxdy.as_finite_difference(wrt=x)
        -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

        We can apply ``as_finite_difference`` to ``Derivative`` instances in
        compound expressions using ``replace``:

        >>> (1 + 42**f(x).diff(x)).replace(lambda arg: arg.is_Derivative,
        ...     lambda arg: arg.as_finite_difference())
        42**(-f(x - 1/2) + f(x + 1/2)) + 1


        See also
        ========

        sympy.calculus.finite_diff.apply_finite_diff
        sympy.calculus.finite_diff.differentiate_finite
        sympy.calculus.finite_diff.finite_diff_weights

        """
        from ..calculus.finite_diff import _as_finite_diff
        return _as_finite_diff(self, points, x0, wrt)
```
### 19 - sympy/core/function.py:

Start line: 1818, End line: 1868

```python
class Subs(Expr):

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self):
        return self.expr.doit().subs(list(zip(self.variables, self.point)))

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)

    n = evalf

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))

    @property
    def expr_free_symbols(self):
        return (self.expr.expr_free_symbols - set(self.variables) |
            set(self.point.expr_free_symbols))

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return super(Subs, self).__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))
```
### 24 - sympy/core/function.py:

Start line: 1890, End line: 1903

```python
class Subs(Expr):

    def _eval_derivative(self, s):
        # Apply the chain rule of the derivative on the substitution variables:
        val = Add.fromiter(p.diff(s) * Subs(self.expr.diff(v), self.variables, self.point).doit() for v, p in zip(self.variables, self.point))

        # Check if there are free symbols in `self.expr`:
        # First get the `expr_free_symbols`, which returns the free symbols
        # that are directly contained in an expression node (i.e. stop
        # searching if the node isn't an expression). At this point turn the
        # expressions into `free_symbols` and check if there are common free
        # symbols in `self.expr` and the deriving factor.
        fs1 = {j for i in self.expr_free_symbols for j in i.free_symbols}
        if len(fs1 & s.free_symbols) > 0:
            val += Subs(self.expr.diff(s), self.variables, self.point).doit()
        return val
```
### 27 - sympy/core/function.py:

Start line: 1637, End line: 1662

```python
class Lambda(Expr):

    def __new__(cls, variables, expr):
        from sympy.sets.sets import FiniteSet
        v = list(variables) if iterable(variables) else [variables]
        for i in v:
            if not getattr(i, 'is_symbol', False):
                raise TypeError('variable is not a symbol: %s' % i)
        if len(v) == 1 and v[0] == expr:
            return S.IdentityFunction

        obj = Expr.__new__(cls, Tuple(*v), sympify(expr))
        obj.nargs = FiniteSet(len(v))
        return obj

    @property
    def variables(self):
        """The variables used in the internal representation of the function"""
        return self._args[0]

    @property
    def expr(self):
        """The return value of the function"""
        return self._args[1]

    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables)
```
### 28 - sympy/core/function.py:

Start line: 838, End line: 1603

```python
class WildFunction(Function, AtomicExpr):
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """

    include = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            # Canonicalize nargs here.  See also FunctionClass.
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict={}, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None

        repl_dict = repl_dict.copy()
        repl_dict[self] = expr
        return repl_dict


class Derivative(Expr):
```
### 34 - sympy/core/function.py:

Start line: 1754, End line: 1816

```python
class Subs(Expr):
    def __new__(cls, expr, variables, point, **assumptions):
        from sympy import Symbol

        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)

        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        if not point:
            return sympify(expr)

        # denest
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)

        # use symbols with names equal to the point value (with preppended _)
        # to give a variable-independent expression
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p])
                for v, p in zip(variables, point)]
            # if any underscore-preppended symbol is already a free symbol
            # and is a variable with a different point value, then there
            # is a clash, e.g. _0 clashes in Subs(_0 + _1, (_0, _1), (1, 0))
            # because the new symbol that would be created is _1 but _1
            # is already mapped to 0 so __0 and __1 are used for the new
            # symbols
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.subs(reps)
        return obj
```
