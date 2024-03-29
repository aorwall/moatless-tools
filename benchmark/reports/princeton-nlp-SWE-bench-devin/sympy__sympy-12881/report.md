# sympy__sympy-12881

| **sympy/sympy** | `d2c3800fd3aaa226c0d37da84086530dd3e5abaf` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 6648 |
| **Any found context length** | 6648 |
| **Avg pos** | 72.5 |
| **Min pos** | 2 |
| **Max pos** | 25 |
| **Top file pos** | 2 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/polys/polytools.py b/sympy/polys/polytools.py
--- a/sympy/polys/polytools.py
+++ b/sympy/polys/polytools.py
@@ -255,7 +255,7 @@ def free_symbols(self):
         ========
 
         >>> from sympy import Poly
-        >>> from sympy.abc import x, y
+        >>> from sympy.abc import x, y, z
 
         >>> Poly(x**2 + 1).free_symbols
         {x}
@@ -263,12 +263,17 @@ def free_symbols(self):
         {x, y}
         >>> Poly(x**2 + y, x).free_symbols
         {x, y}
+        >>> Poly(x**2 + y, x, z).free_symbols
+        {x, y}
 
         """
-        symbols = set([])
-
-        for gen in self.gens:
-            symbols |= gen.free_symbols
+        symbols = set()
+        gens = self.gens
+        for i in range(len(gens)):
+            for monom in self.monoms():
+                if monom[i]:
+                    symbols |= gens[i].free_symbols
+                    break
 
         return symbols | self.free_symbols_in_domain
 
@@ -609,7 +614,10 @@ def reorder(f, *gens, **args):
 
     def ltrim(f, gen):
         """
-        Remove dummy generators from the "left" of ``f``.
+        Remove dummy generators from ``f`` that are to the left of
+        specified ``gen`` in the generators as ordered. When ``gen``
+        is an integer, it refers to the generator located at that
+        position within the tuple of generators of ``f``.
 
         Examples
         ========
@@ -619,20 +627,23 @@ def ltrim(f, gen):
 
         >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)
         Poly(y**2 + y*z**2, y, z, domain='ZZ')
+        >>> Poly(z, x, y, z).ltrim(-1)
+        Poly(z, z, domain='ZZ')
 
         """
         rep = f.as_dict(native=True)
         j = f._gen_to_level(gen)
+
         terms = {}
 
         for monom, coeff in rep.items():
-            monom = monom[j:]
 
-            if monom not in terms:
-                terms[monom] = coeff
-            else:
+            if any(i for i in monom[:j]):
+                # some generator is used in the portion to be trimmed
                 raise PolynomialError("can't left trim %s" % f)
 
+            terms[monom[j:]] = coeff
+
         gens = f.gens[j:]
 
         return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)
@@ -653,7 +664,7 @@ def has_only_gens(f, *gens):
         False
 
         """
-        indices = set([])
+        indices = set()
 
         for gen in gens:
             try:
diff --git a/sympy/solvers/polysys.py b/sympy/solvers/polysys.py
--- a/sympy/solvers/polysys.py
+++ b/sympy/solvers/polysys.py
@@ -6,7 +6,7 @@
 from sympy.polys import Poly, groebner, roots
 from sympy.polys.polytools import parallel_poly_from_expr
 from sympy.polys.polyerrors import (ComputationFailed,
-    PolificationFailed, CoercionFailed)
+    PolificationFailed, CoercionFailed, PolynomialError)
 from sympy.simplify import rcollect
 from sympy.utilities import default_sort_key, postfixes
 
@@ -37,10 +37,7 @@ def solve_poly_system(seq, *gens, **args):
     if len(polys) == len(opt.gens) == 2:
         f, g = polys
 
-        a, b = f.degree_list()
-        c, d = g.degree_list()
-
-        if a <= 2 and b <= 2 and c <= 2 and d <= 2:
+        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
             try:
                 return solve_biquadratic(f, g, opt)
             except SolveFailed:
@@ -79,13 +76,16 @@ def solve_biquadratic(f, g, opt):
     if len(G) != 2:
         raise SolveFailed
 
-    p, q = G
     x, y = opt.gens
+    p, q = G
+    if not p.gcd(q).is_ground:
+        # not 0-dimensional
+        raise SolveFailed
 
     p = Poly(p, x, expand=False)
-    q = q.ltrim(-1)
-
     p_roots = [ rcollect(expr, y) for expr in roots(p).keys() ]
+
+    q = q.ltrim(-1)
     q_roots = list(roots(q).keys())
 
     solutions = []
@@ -161,7 +161,7 @@ def solve_generic(polys, opt):
     def _is_univariate(f):
         """Returns True if 'f' is univariate in its last variable. """
         for monom in f.monoms():
-            if any(m > 0 for m in monom[:-1]):
+            if any(m for m in monom[:-1]):
                 return False
 
         return True

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/polys/polytools.py | 258 | 258 | 25 | 2 | 22270
| sympy/polys/polytools.py | 266 | 269 | 25 | 2 | 22270
| sympy/polys/polytools.py | 612 | 612 | 25 | 2 | 22270
| sympy/polys/polytools.py | 622 | 626 | 25 | 2 | 22270
| sympy/polys/polytools.py | 656 | 656 | 25 | 2 | 22270
| sympy/solvers/polysys.py | 9 | 9 | - | 5 | -
| sympy/solvers/polysys.py | 40 | 43 | - | 5 | -
| sympy/solvers/polysys.py | 82 | 87 | - | 5 | -
| sympy/solvers/polysys.py | 164 | 164 | 10 | 5 | 11771


## Problem Statement

```
Poly(x,x,y).free_symbols -> {x, y} instead of just {x}
No free symbols of generators that don't appear in the expression of the polynomial should appear in the set of free symbols.

\`\`\`
def free_symbols(poly):
 free = set()
 for i in range(len(poly.gens)):
    for m in poly.monoms():
        if i in m:
            free |= poly.gens[i].free_symbols
            break
 return free | poly.free_symbols_in_domain  # not sure about the domain part....
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/solvers/bivariate.py | 18 | 40| 195 | 195 | 3965 | 
| **-> 2 <-** | **2 sympy/polys/polytools.py** | 64 | 1047| 6453 | 6648 | 53076 | 
| 3 | **2 sympy/polys/polytools.py** | 4121 | 4151| 171 | 6819 | 53076 | 
| 4 | 3 sympy/combinatorics/free_groups.py | 42 | 64| 201 | 7020 | 62423 | 
| 5 | 4 sympy/polys/polyroots.py | 720 | 787| 394 | 7414 | 71952 | 
| 6 | 4 sympy/combinatorics/free_groups.py | 92 | 113| 150 | 7564 | 71952 | 
| 7 | **4 sympy/polys/polytools.py** | 3663 | 4118| 2827 | 10391 | 71952 | 
| 8 | 4 sympy/combinatorics/free_groups.py | 18 | 40| 190 | 10581 | 71952 | 
| 9 | **4 sympy/polys/polytools.py** | 6879 | 6958| 453 | 11034 | 71952 | 
| **-> 10 <-** | **5 sympy/solvers/polysys.py** | 101 | 167| 737 | 11771 | 74515 | 
| 11 | 5 sympy/combinatorics/free_groups.py | 66 | 89| 200 | 11971 | 74515 | 
| 12 | 6 sympy/polys/rings.py | 597 | 639| 364 | 12335 | 92074 | 
| 13 | **6 sympy/polys/polytools.py** | 4180 | 4212| 277 | 12612 | 92074 | 
| 14 | 7 sympy/solvers/solveset.py | 1917 | 2161| 236 | 12848 | 111556 | 
| 15 | 8 sympy/polys/polyutils.py | 228 | 299| 437 | 13285 | 114925 | 
| 16 | 8 sympy/polys/polyutils.py | 302 | 317| 150 | 13435 | 114925 | 
| 17 | 8 sympy/polys/polyutils.py | 178 | 225| 300 | 13735 | 114925 | 
| 18 | 8 sympy/combinatorics/free_groups.py | 164 | 180| 115 | 13850 | 114925 | 
| 19 | 8 sympy/polys/polyroots.py | 863 | 930| 470 | 14320 | 114925 | 
| 20 | 9 sympy/polys/polyfuncs.py | 98 | 153| 417 | 14737 | 117786 | 
| 21 | 10 sympy/polys/agca/modules.py | 1019 | 1051| 288 | 15025 | 129369 | 
| 22 | **10 sympy/polys/polytools.py** | 4153 | 4178| 194 | 15219 | 129369 | 
| 23 | 10 sympy/polys/rings.py | 422 | 460| 269 | 15488 | 129369 | 
| 24 | **10 sympy/polys/polytools.py** | 1049 | 1971| 6359 | 21847 | 129369 | 
| **-> 25 <-** | **10 sympy/polys/polytools.py** | 1 | 4118| 423 | 22270 | 129369 | 
| 26 | 10 sympy/polys/rings.py | 825 | 904| 515 | 22785 | 129369 | 
| 27 | 11 sympy/polys/domains/old_polynomialring.py | 352 | 432| 779 | 23564 | 133211 | 
| 28 | 11 sympy/solvers/solveset.py | 1876 | 1901| 202 | 23766 | 133211 | 
| 29 | 11 sympy/polys/polyfuncs.py | 23 | 96| 634 | 24400 | 133211 | 
| 30 | 11 sympy/polys/rings.py | 62 | 89| 239 | 24639 | 133211 | 
| 31 | 11 sympy/polys/rings.py | 33 | 60| 233 | 24872 | 133211 | 
| 32 | 12 sympy/polys/sqfreetools.py | 358 | 415| 454 | 25326 | 137005 | 
| 33 | 12 sympy/polys/rings.py | 256 | 319| 436 | 25762 | 137005 | 
| 34 | 13 sympy/polys/subresultants_qq_zz.py | 283 | 336| 420 | 26182 | 160628 | 
| 35 | 13 sympy/combinatorics/free_groups.py | 1 | 15| 131 | 26313 | 160628 | 
| 36 | 13 sympy/polys/agca/modules.py | 430 | 455| 221 | 26534 | 160628 | 
| 37 | 14 sympy/combinatorics/fp_groups.py | 713 | 773| 701 | 27235 | 168867 | 
| 38 | 14 sympy/combinatorics/free_groups.py | 139 | 162| 207 | 27442 | 168867 | 
| 39 | 14 sympy/polys/rings.py | 2362 | 2392| 250 | 27692 | 168867 | 
| 40 | 14 sympy/polys/rings.py | 516 | 541| 228 | 27920 | 168867 | 
| 41 | 14 sympy/polys/sqfreetools.py | 418 | 446| 265 | 28185 | 168867 | 
| 42 | 14 sympy/polys/rings.py | 91 | 118| 239 | 28424 | 168867 | 
| 43 | 14 sympy/solvers/solveset.py | 422 | 483| 594 | 29018 | 168867 | 
| 44 | 14 sympy/combinatorics/free_groups.py | 115 | 137| 139 | 29157 | 168867 | 
| 45 | 15 sympy/polys/polyoptions.py | 485 | 493| 131 | 29288 | 173565 | 
| 46 | 15 sympy/polys/rings.py | 2394 | 2456| 448 | 29736 | 173565 | 
| 47 | 16 sympy/polys/specialpolys.py | 1 | 33| 223 | 29959 | 177730 | 
| 48 | 16 sympy/polys/polyroots.py | 970 | 1071| 730 | 30689 | 177730 | 
| 49 | 16 sympy/polys/agca/modules.py | 1119 | 1128| 168 | 30857 | 177730 | 
| 50 | 17 sympy/geometry/curve.py | 145 | 165| 159 | 31016 | 179905 | 
| 51 | **17 sympy/polys/polytools.py** | 4215 | 4266| 342 | 31358 | 179905 | 
| 52 | 18 sympy/polys/polyclasses.py | 297 | 352| 505 | 31863 | 194226 | 
| 53 | 18 sympy/polys/sqfreetools.py | 273 | 327| 403 | 32266 | 194226 | 


### Hint

```
\`\`\`diff
diff --git a/sympy/polys/polytools.py b/sympy/polys/polytools.py
index 9c12741..92e7ca6 100644
--- a/sympy/polys/polytools.py
+++ b/sympy/polys/polytools.py
@@ -255,7 +255,7 @@ def free_symbols(self):
         ========
 
         >>> from sympy import Poly
-        >>> from sympy.abc import x, y
+        >>> from sympy.abc import x, y, z
 
         >>> Poly(x**2 + 1).free_symbols
         {x}
@@ -263,12 +263,17 @@ def free_symbols(self):
         {x, y}
         >>> Poly(x**2 + y, x).free_symbols
         {x, y}
+        >>> Poly(x**2 + y, x, z).free_symbols
+        {x, y}
 
         """
-        symbols = set([])
-
-        for gen in self.gens:
-            symbols |= gen.free_symbols
+        symbols = set()
+        gens = self.gens
+        for i in range(len(gens)):
+            for monom in self.monoms():
+                if monom[i]:
+                    symbols |= gens[i].free_symbols
+                    break
 
         return symbols | self.free_symbols_in_domain
 
diff --git a/sympy/polys/tests/test_polytools.py b/sympy/polys/tests/test_polytools.py
index a1e5179..8e30ef1 100644
--- a/sympy/polys/tests/test_polytools.py
+++ b/sympy/polys/tests/test_polytools.py
@@ -468,6 +468,8 @@ def test_Poly_free_symbols():
     assert Poly(x**2 + sin(y*z)).free_symbols == {x, y, z}
     assert Poly(x**2 + sin(y*z), x).free_symbols == {x, y, z}
     assert Poly(x**2 + sin(y*z), x, domain=EX).free_symbols == {x, y, z}
+    assert Poly(1 + x + x**2, x, y, z).free_symbols == {x}
+    assert Poly(x + sin(y), z).free_symbols == {x, y}
 
 
 def test_PurePoly_free_symbols():

\`\`\`
```

## Patch

```diff
diff --git a/sympy/polys/polytools.py b/sympy/polys/polytools.py
--- a/sympy/polys/polytools.py
+++ b/sympy/polys/polytools.py
@@ -255,7 +255,7 @@ def free_symbols(self):
         ========
 
         >>> from sympy import Poly
-        >>> from sympy.abc import x, y
+        >>> from sympy.abc import x, y, z
 
         >>> Poly(x**2 + 1).free_symbols
         {x}
@@ -263,12 +263,17 @@ def free_symbols(self):
         {x, y}
         >>> Poly(x**2 + y, x).free_symbols
         {x, y}
+        >>> Poly(x**2 + y, x, z).free_symbols
+        {x, y}
 
         """
-        symbols = set([])
-
-        for gen in self.gens:
-            symbols |= gen.free_symbols
+        symbols = set()
+        gens = self.gens
+        for i in range(len(gens)):
+            for monom in self.monoms():
+                if monom[i]:
+                    symbols |= gens[i].free_symbols
+                    break
 
         return symbols | self.free_symbols_in_domain
 
@@ -609,7 +614,10 @@ def reorder(f, *gens, **args):
 
     def ltrim(f, gen):
         """
-        Remove dummy generators from the "left" of ``f``.
+        Remove dummy generators from ``f`` that are to the left of
+        specified ``gen`` in the generators as ordered. When ``gen``
+        is an integer, it refers to the generator located at that
+        position within the tuple of generators of ``f``.
 
         Examples
         ========
@@ -619,20 +627,23 @@ def ltrim(f, gen):
 
         >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)
         Poly(y**2 + y*z**2, y, z, domain='ZZ')
+        >>> Poly(z, x, y, z).ltrim(-1)
+        Poly(z, z, domain='ZZ')
 
         """
         rep = f.as_dict(native=True)
         j = f._gen_to_level(gen)
+
         terms = {}
 
         for monom, coeff in rep.items():
-            monom = monom[j:]
 
-            if monom not in terms:
-                terms[monom] = coeff
-            else:
+            if any(i for i in monom[:j]):
+                # some generator is used in the portion to be trimmed
                 raise PolynomialError("can't left trim %s" % f)
 
+            terms[monom[j:]] = coeff
+
         gens = f.gens[j:]
 
         return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)
@@ -653,7 +664,7 @@ def has_only_gens(f, *gens):
         False
 
         """
-        indices = set([])
+        indices = set()
 
         for gen in gens:
             try:
diff --git a/sympy/solvers/polysys.py b/sympy/solvers/polysys.py
--- a/sympy/solvers/polysys.py
+++ b/sympy/solvers/polysys.py
@@ -6,7 +6,7 @@
 from sympy.polys import Poly, groebner, roots
 from sympy.polys.polytools import parallel_poly_from_expr
 from sympy.polys.polyerrors import (ComputationFailed,
-    PolificationFailed, CoercionFailed)
+    PolificationFailed, CoercionFailed, PolynomialError)
 from sympy.simplify import rcollect
 from sympy.utilities import default_sort_key, postfixes
 
@@ -37,10 +37,7 @@ def solve_poly_system(seq, *gens, **args):
     if len(polys) == len(opt.gens) == 2:
         f, g = polys
 
-        a, b = f.degree_list()
-        c, d = g.degree_list()
-
-        if a <= 2 and b <= 2 and c <= 2 and d <= 2:
+        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
             try:
                 return solve_biquadratic(f, g, opt)
             except SolveFailed:
@@ -79,13 +76,16 @@ def solve_biquadratic(f, g, opt):
     if len(G) != 2:
         raise SolveFailed
 
-    p, q = G
     x, y = opt.gens
+    p, q = G
+    if not p.gcd(q).is_ground:
+        # not 0-dimensional
+        raise SolveFailed
 
     p = Poly(p, x, expand=False)
-    q = q.ltrim(-1)
-
     p_roots = [ rcollect(expr, y) for expr in roots(p).keys() ]
+
+    q = q.ltrim(-1)
     q_roots = list(roots(q).keys())
 
     solutions = []
@@ -161,7 +161,7 @@ def solve_generic(polys, opt):
     def _is_univariate(f):
         """Returns True if 'f' is univariate in its last variable. """
         for monom in f.monoms():
-            if any(m > 0 for m in monom[:-1]):
+            if any(m for m in monom[:-1]):
                 return False
 
         return True

```

## Test Patch

```diff
diff --git a/sympy/polys/tests/test_polytools.py b/sympy/polys/tests/test_polytools.py
--- a/sympy/polys/tests/test_polytools.py
+++ b/sympy/polys/tests/test_polytools.py
@@ -468,6 +468,8 @@ def test_Poly_free_symbols():
     assert Poly(x**2 + sin(y*z)).free_symbols == {x, y, z}
     assert Poly(x**2 + sin(y*z), x).free_symbols == {x, y, z}
     assert Poly(x**2 + sin(y*z), x, domain=EX).free_symbols == {x, y, z}
+    assert Poly(1 + x + x**2, x, y, z).free_symbols == {x}
+    assert Poly(x + sin(y), z).free_symbols == {x, y}
 
 
 def test_PurePoly_free_symbols():
@@ -851,9 +853,10 @@ def test_Poly_reorder():
 def test_Poly_ltrim():
     f = Poly(y**2 + y*z**2, x, y, z).ltrim(y)
     assert f.as_expr() == y**2 + y*z**2 and f.gens == (y, z)
+    assert Poly(x*y - x, z, x, y).ltrim(1) == Poly(x*y - x, x, y)
 
     raises(PolynomialError, lambda: Poly(x*y**2 + y**2, x, y).ltrim(y))
-
+    raises(PolynomialError, lambda: Poly(x*y - x, x, y).ltrim(-1))
 
 def test_Poly_has_only_gens():
     assert Poly(x*y + 1, x, y, z).has_only_gens(x, y) is True
diff --git a/sympy/solvers/tests/test_polysys.py b/sympy/solvers/tests/test_polysys.py
--- a/sympy/solvers/tests/test_polysys.py
+++ b/sympy/solvers/tests/test_polysys.py
@@ -1,9 +1,12 @@
 """Tests for solvers of systems of polynomial equations. """
 
-from sympy import flatten, I, Integer, Poly, QQ, Rational, S, sqrt, symbols
+from sympy import (flatten, I, Integer, Poly, QQ, Rational, S, sqrt,
+    solve, symbols)
 from sympy.abc import x, y, z
 from sympy.polys import PolynomialError
-from sympy.solvers.polysys import solve_poly_system, solve_triangulated
+from sympy.solvers.polysys import (solve_poly_system,
+    solve_triangulated, solve_biquadratic, SolveFailed)
+from sympy.polys.polytools import parallel_poly_from_expr
 from sympy.utilities.pytest import raises
 
 
@@ -50,10 +53,10 @@ def test_solve_biquadratic():
 
     f_1 = (x - 1)**2 + (y - 1)**2 - r**2
     f_2 = (x - 2)**2 + (y - 2)**2 - r**2
-
-    assert solve_poly_system([f_1, f_2], x, y) == \
-        [(S(3)/2 - sqrt(-1 + 2*r**2)/2, S(3)/2 + sqrt(-1 + 2*r**2)/2),
-         (S(3)/2 + sqrt(-1 + 2*r**2)/2, S(3)/2 - sqrt(-1 + 2*r**2)/2)]
+    s = sqrt(2*r**2 - 1)
+    a = (3 - s)/2
+    b = (3 + s)/2
+    assert solve_poly_system([f_1, f_2], x, y) == [(a, b), (b, a)]
 
     f_1 = (x - 1)**2 + (y - 2)**2 - r**2
     f_2 = (x - 1)**2 + (y - 1)**2 - r**2
@@ -80,8 +83,28 @@ def test_solve_biquadratic():
     assert len(result) == 2 and all(len(r) == 2 for r in result)
     assert all(len(r.find(query)) == 1 for r in flatten(result))
 
-
-def test_solve_triangualted():
+    s1 = (x*y - y, x**2 - x)
+    assert solve(s1) == [{x: 1}, {x: 0, y: 0}]
+    s2 = (x*y - x, y**2 - y)
+    assert solve(s2) == [{y: 1}, {x: 0, y: 0}]
+    gens = (x, y)
+    for seq in (s1, s2):
+        (f, g), opt = parallel_poly_from_expr(seq, *gens)
+        raises(SolveFailed, lambda: solve_biquadratic(f, g, opt))
+    seq = (x**2 + y**2 - 2, y**2 - 1)
+    (f, g), opt = parallel_poly_from_expr(seq, *gens)
+    assert solve_biquadratic(f, g, opt) == [
+        (-1, -1), (-1, 1), (1, -1), (1, 1)]
+    ans = [(0, -1), (0, 1)]
+    seq = (x**2 + y**2 - 1, y**2 - 1)
+    (f, g), opt = parallel_poly_from_expr(seq, *gens)
+    assert solve_biquadratic(f, g, opt) == ans
+    seq = (x**2 + y**2 - 1, x**2 - x + y**2 - 1)
+    (f, g), opt = parallel_poly_from_expr(seq, *gens)
+    assert solve_biquadratic(f, g, opt) == ans
+
+
+def test_solve_triangulated():
     f_1 = x**2 + y + z - 1
     f_2 = x + y**2 + z - 1
     f_3 = x + y + z**2 - 1

```


## Code snippets

### 1 - sympy/solvers/bivariate.py:

Start line: 18, End line: 40

```python
def _filtered_gens(poly, symbol):
    """process the generators of ``poly``, returning the set of generators that
    have ``symbol``.  If there are two generators that are inverses of each other,
    prefer the one that has no denominator.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _filtered_gens
    >>> from sympy import Poly, exp
    >>> from sympy.abc import x
    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)
    {x, exp(x)}

    """
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        ag = 1/g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens
```
### 2 - sympy/polys/polytools.py:

Start line: 64, End line: 1047

```python
@public
class Poly(Expr):
    """
    Generic class for representing and operating on polynomial expressions.
    Subclasses Expr class.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x, y

    Create a univariate polynomial:

    >>> Poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    Create a univariate polynomial with specific domain:

    >>> from sympy import sqrt
    >>> Poly(x**2 + 2*x + sqrt(3), domain='R')
    Poly(1.0*x**2 + 2.0*x + 1.73205080756888, x, domain='RR')

    Create a multivariate polynomial:

    >>> Poly(y*x**2 + x*y + 1)
    Poly(x**2*y + x*y + 1, x, y, domain='ZZ')

    Create a univariate polynomial, where y is a constant:

    >>> Poly(y*x**2 + x*y + 1,x)
    Poly(y*x**2 + y*x + 1, x, domain='ZZ[y]')

    You can evaluate the above polynomial as a function of y:

    >>> Poly(y*x**2 + x*y + 1,x).eval(2)
    6*y + 1

    See Also
    ========
    sympy.core.expr.Expr

    """

    __slots__ = ['rep', 'gens']

    is_commutative = True
    is_Poly = True

    def __new__(cls, rep, *gens, **args):
        """Create a new polynomial instance out of something useful. """
        opt = options.build_options(gens, args)

        if 'order' in opt:
            raise NotImplementedError("'order' keyword is not implemented yet")

        if iterable(rep, exclude=str):
            if isinstance(rep, dict):
                return cls._from_dict(rep, opt)
            else:
                return cls._from_list(list(rep), opt)
        else:
            rep = sympify(rep)

            if rep.is_Poly:
                return cls._from_poly(rep, opt)
            else:
                return cls._from_expr(rep, opt)

    @classmethod
    def new(cls, rep, *gens):
        """Construct :class:`Poly` instance from raw representation. """
        if not isinstance(rep, DMP):
            raise PolynomialError(
                "invalid polynomial representation: %s" % rep)
        elif rep.lev != len(gens) - 1:
            raise PolynomialError("invalid arguments: %s, %s" % (rep, gens))

        obj = Basic.__new__(cls)

        obj.rep = rep
        obj.gens = gens

        return obj

    @classmethod
    def from_dict(cls, rep, *gens, **args):
        """Construct a polynomial from a ``dict``. """
        opt = options.build_options(gens, args)
        return cls._from_dict(rep, opt)

    @classmethod
    def from_list(cls, rep, *gens, **args):
        """Construct a polynomial from a ``list``. """
        opt = options.build_options(gens, args)
        return cls._from_list(rep, opt)

    @classmethod
    def from_poly(cls, rep, *gens, **args):
        """Construct a polynomial from a polynomial. """
        opt = options.build_options(gens, args)
        return cls._from_poly(rep, opt)

    @classmethod
    def from_expr(cls, rep, *gens, **args):
        """Construct a polynomial from an expression. """
        opt = options.build_options(gens, args)
        return cls._from_expr(rep, opt)

    @classmethod
    def _from_dict(cls, rep, opt):
        """Construct a polynomial from a ``dict``. """
        gens = opt.gens

        if not gens:
            raise GeneratorsNeeded(
                "can't initialize from 'dict' without generators")

        level = len(gens) - 1
        domain = opt.domain

        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            for monom, coeff in rep.items():
                rep[monom] = domain.convert(coeff)

        return cls.new(DMP.from_dict(rep, level, domain), *gens)

    @classmethod
    def _from_list(cls, rep, opt):
        """Construct a polynomial from a ``list``. """
        gens = opt.gens

        if not gens:
            raise GeneratorsNeeded(
                "can't initialize from 'list' without generators")
        elif len(gens) != 1:
            raise MultivariatePolynomialError(
                "'list' representation not supported")

        level = len(gens) - 1
        domain = opt.domain

        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            rep = list(map(domain.convert, rep))

        return cls.new(DMP.from_list(rep, level, domain), *gens)

    @classmethod
    def _from_poly(cls, rep, opt):
        """Construct a polynomial from a polynomial. """
        if cls != rep.__class__:
            rep = cls.new(rep.rep, *rep.gens)

        gens = opt.gens
        field = opt.field
        domain = opt.domain

        if gens and rep.gens != gens:
            if set(rep.gens) != set(gens):
                return cls._from_expr(rep.as_expr(), opt)
            else:
                rep = rep.reorder(*gens)

        if 'domain' in opt and domain:
            rep = rep.set_domain(domain)
        elif field is True:
            rep = rep.to_field()

        return rep

    @classmethod
    def _from_expr(cls, rep, opt):
        """Construct a polynomial from an expression. """
        rep, opt = _dict_from_expr(rep, opt)
        return cls._from_dict(rep, opt)

    def _hashable_content(self):
        """Allow SymPy to hash Poly instances. """
        return (self.rep, self.gens)

    def __hash__(self):
        return super(Poly, self).__hash__()

    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial expression.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1).free_symbols
        {x}
        >>> Poly(x**2 + y).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x).free_symbols
        {x, y}

        """
        symbols = set([])

        for gen in self.gens:
            symbols |= gen.free_symbols

        return symbols | self.free_symbols_in_domain

    @property
    def free_symbols_in_domain(self):
        """
        Free symbols of the domain of ``self``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y, x).free_symbols_in_domain
        {y}

        """
        domain, symbols = self.rep.dom, set()

        if domain.is_Composite:
            for gen in domain.symbols:
                symbols |= gen.free_symbols
        elif domain.is_EX:
            for coeff in self.coeffs():
                symbols |= coeff.free_symbols

        return symbols

    @property
    def args(self):
        """
        Don't mess up with the core.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).args
        (x**2 + 1,)

        """
        return (self.as_expr(),)

    @property
    def gen(self):
        """
        Return the principal generator.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).gen
        x

        """
        return self.gens[0]

    @property
    def domain(self):
        """Get the ground domain of ``self``. """
        return self.get_domain()

    @property
    def zero(self):
        """Return zero polynomial with ``self``'s properties. """
        return self.new(self.rep.zero(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def one(self):
        """Return one polynomial with ``self``'s properties. """
        return self.new(self.rep.one(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def unit(self):
        """Return unit polynomial with ``self``'s properties. """
        return self.new(self.rep.unit(self.rep.lev, self.rep.dom), *self.gens)

    def unify(f, g):
        """
        Make ``f`` and ``g`` belong to the same domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f, g = Poly(x/2 + 1), Poly(2*x + 1)

        >>> f
        Poly(1/2*x + 1, x, domain='QQ')
        >>> g
        Poly(2*x + 1, x, domain='ZZ')

        >>> F, G = f.unify(g)

        >>> F
        Poly(1/2*x + 1, x, domain='QQ')
        >>> G
        Poly(2*x + 1, x, domain='QQ')

        """
        _, per, F, G = f._unify(g)
        return per(F), per(G)

    def _unify(f, g):
        g = sympify(g)

        if not g.is_Poly:
            try:
                return f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g))
            except CoercionFailed:
                raise UnificationFailed("can't unify %s with %s" % (f, g))

        if isinstance(f.rep, DMP) and isinstance(g.rep, DMP):
            gens = _unify_gens(f.gens, g.gens)

            dom, lev = f.rep.dom.unify(g.rep.dom, gens), len(gens) - 1

            if f.gens != gens:
                f_monoms, f_coeffs = _dict_reorder(
                    f.rep.to_dict(), f.gens, gens)

                if f.rep.dom != dom:
                    f_coeffs = [dom.convert(c, f.rep.dom) for c in f_coeffs]

                F = DMP(dict(list(zip(f_monoms, f_coeffs))), dom, lev)
            else:
                F = f.rep.convert(dom)

            if g.gens != gens:
                g_monoms, g_coeffs = _dict_reorder(
                    g.rep.to_dict(), g.gens, gens)

                if g.rep.dom != dom:
                    g_coeffs = [dom.convert(c, g.rep.dom) for c in g_coeffs]

                G = DMP(dict(list(zip(g_monoms, g_coeffs))), dom, lev)
            else:
                G = g.rep.convert(dom)
        else:
            raise UnificationFailed("can't unify %s with %s" % (f, g))

        cls = f.__class__

        def per(rep, dom=dom, gens=gens, remove=None):
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]

                if not gens:
                    return dom.to_sympy(rep)

            return cls.new(rep, *gens)

        return dom, per, F, G

    def per(f, rep, gens=None, remove=None):
        """
        Create a Poly out of the given representation.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x, y

        >>> from sympy.polys.polyclasses import DMP

        >>> a = Poly(x**2 + 1)

        >>> a.per(DMP([ZZ(1), ZZ(1)], ZZ), gens=[y])
        Poly(y + 1, y, domain='ZZ')

        """
        if gens is None:
            gens = f.gens

        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]

            if not gens:
                return f.rep.dom.to_sympy(rep)

        return f.__class__.new(rep, *gens)

    def set_domain(f, domain):
        """Set the ground domain of ``f``. """
        opt = options.build_options(f.gens, {'domain': domain})
        return f.per(f.rep.convert(opt.domain))

    def get_domain(f):
        """Get the ground domain of ``f``. """
        return f.rep.dom

    def set_modulus(f, modulus):
        """
        Set the modulus of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(5*x**2 + 2*x - 1, x).set_modulus(2)
        Poly(x**2 + 1, x, modulus=2)

        """
        modulus = options.Modulus.preprocess(modulus)
        return f.set_domain(FF(modulus))

    def get_modulus(f):
        """
        Get the modulus of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, modulus=2).get_modulus()
        2

        """
        domain = f.get_domain()

        if domain.is_FiniteField:
            return Integer(domain.characteristic())
        else:
            raise PolynomialError("not a polynomial over a Galois field")

    def _eval_subs(f, old, new):
        """Internal implementation of :func:`subs`. """
        if old in f.gens:
            if new.is_number:
                return f.eval(old, new)
            else:
                try:
                    return f.replace(old, new)
                except PolynomialError:
                    pass

        return f.as_expr().subs(old, new)

    def exclude(f):
        """
        Remove unnecessary generators from ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import a, b, c, d, x

        >>> Poly(a + x, a, b, c, d, x).exclude()
        Poly(a + x, a, x, domain='ZZ')

        """
        J, new = f.rep.exclude()
        gens = []

        for j in range(len(f.gens)):
            if j not in J:
                gens.append(f.gens[j])

        return f.per(new, gens=gens)

    def replace(f, x, y=None):
        """
        Replace ``x`` with ``y`` in generators list.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1, x).replace(x, y)
        Poly(y**2 + 1, y, domain='ZZ')

        """
        if y is None:
            if f.is_univariate:
                x, y = f.gen, x
            else:
                raise PolynomialError(
                    "syntax supported only in univariate case")

        if x == y:
            return f

        if x in f.gens and y not in f.gens:
            dom = f.get_domain()

            if not dom.is_Composite or y not in dom.symbols:
                gens = list(f.gens)
                gens[gens.index(x)] = y
                return f.per(f.rep, gens=gens)

        raise PolynomialError("can't replace %s with %s in %s" % (x, y, f))

    def reorder(f, *gens, **args):
        """
        Efficiently apply new order of generators.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y**2, x, y).reorder(y, x)
        Poly(y**2*x + x**2, y, x, domain='ZZ')

        """
        opt = options.Options((), args)

        if not gens:
            gens = _sort_gens(f.gens, opt=opt)
        elif set(f.gens) != set(gens):
            raise PolynomialError(
                "generators list can differ only up to order of elements")

        rep = dict(list(zip(*_dict_reorder(f.rep.to_dict(), f.gens, gens))))

        return f.per(DMP(rep, f.rep.dom, len(gens) - 1), gens=gens)

    def ltrim(f, gen):
        """
        Remove dummy generators from the "left" of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)
        Poly(y**2 + y*z**2, y, z, domain='ZZ')

        """
        rep = f.as_dict(native=True)
        j = f._gen_to_level(gen)
        terms = {}

        for monom, coeff in rep.items():
            monom = monom[j:]

            if monom not in terms:
                terms[monom] = coeff
            else:
                raise PolynomialError("can't left trim %s" % f)

        gens = f.gens[j:]

        return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)

    def has_only_gens(f, *gens):
        """
        Return ``True`` if ``Poly(f, *gens)`` retains ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x*y + 1, x, y, z).has_only_gens(x, y)
        True
        >>> Poly(x*y + z, x, y, z).has_only_gens(x, y)
        False

        """
        indices = set([])

        for gen in gens:
            try:
                index = f.gens.index(gen)
            except ValueError:
                raise GeneratorsError(
                    "%s doesn't have %s as generator" % (f, gen))
            else:
                indices.add(index)

        for monom in f.monoms():
            for i, elt in enumerate(monom):
                if i not in indices and elt:
                    return False

        return True

    def to_ring(f):
        """
        Make the ground domain a ring.

        Examples
        ========

        >>> from sympy import Poly, QQ
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, domain=QQ).to_ring()
        Poly(x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'to_ring'):
            result = f.rep.to_ring()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'to_ring')

        return f.per(result)

    def to_field(f):
        """
        Make the ground domain a field.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x, domain=ZZ).to_field()
        Poly(x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'to_field'):
            result = f.rep.to_field()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'to_field')

        return f.per(result)

    def to_exact(f):
        """
        Make the ground domain exact.

        Examples
        ========

        >>> from sympy import Poly, RR
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1.0, x, domain=RR).to_exact()
        Poly(x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'to_exact'):
            result = f.rep.to_exact()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'to_exact')

        return f.per(result)

    def retract(f, field=None):
        """
        Recalculate the ground domain of a polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**2 + 1, x, domain='QQ[y]')
        >>> f
        Poly(x**2 + 1, x, domain='QQ[y]')

        >>> f.retract()
        Poly(x**2 + 1, x, domain='ZZ')
        >>> f.retract(field=True)
        Poly(x**2 + 1, x, domain='QQ')

        """
        dom, rep = construct_domain(f.as_dict(zero=True),
            field=field, composite=f.domain.is_Composite or None)
        return f.from_dict(rep, f.gens, domain=dom)

    def slice(f, x, m, n=None):
        """Take a continuous subsequence of terms of ``f``. """
        if n is None:
            j, m, n = 0, x, m
        else:
            j = f._gen_to_level(x)

        m, n = int(m), int(n)

        if hasattr(f.rep, 'slice'):
            result = f.rep.slice(m, n, j)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'slice')

        return f.per(result)

    def coeffs(f, order=None):
        """
        Returns all non-zero coefficients from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x + 3, x).coeffs()
        [1, 2, 3]

        See Also
        ========
        all_coeffs
        coeff_monomial
        nth

        """
        return [f.rep.dom.to_sympy(c) for c in f.rep.coeffs(order=order)]

    def monoms(f, order=None):
        """
        Returns all non-zero monomials from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).monoms()
        [(2, 0), (1, 2), (1, 1), (0, 1)]

        See Also
        ========
        all_monoms

        """
        return f.rep.monoms(order=order)

    def terms(f, order=None):
        """
        Returns all non-zero terms from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).terms()
        [((2, 0), 1), ((1, 2), 2), ((1, 1), 1), ((0, 1), 3)]

        See Also
        ========
        all_terms

        """
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.terms(order=order)]

    def all_coeffs(f):
        """
        Returns all coefficients from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_coeffs()
        [1, 0, 2, -1]

        """
        return [f.rep.dom.to_sympy(c) for c in f.rep.all_coeffs()]

    def all_monoms(f):
        """
        Returns all monomials from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_monoms()
        [(3,), (2,), (1,), (0,)]

        See Also
        ========
        all_terms

        """
        return f.rep.all_monoms()

    def all_terms(f):
        """
        Returns all terms from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_terms()
        [((3,), 1), ((2,), 0), ((1,), 2), ((0,), -1)]

        """
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.all_terms()]

    def termwise(f, func, *gens, **args):
        """
        Apply a function to all terms of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> def func(k, coeff):
        ...     k = k[0]
        ...     return coeff//10**(2-k)

        >>> Poly(x**2 + 20*x + 400).termwise(func)
        Poly(x**2 + 2*x + 4, x, domain='ZZ')

        """
        terms = {}

        for monom, coeff in f.terms():
            result = func(monom, coeff)

            if isinstance(result, tuple):
                monom, coeff = result
            else:
                coeff = result

            if coeff:
                if monom not in terms:
                    terms[monom] = coeff
                else:
                    raise PolynomialError(
                        "%s monomial was generated twice" % monom)

        return f.from_dict(terms, *(gens or f.gens), **args)

    def length(f):
        """
        Returns the number of non-zero terms in ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 2*x - 1).length()
        3

        """
        return len(f.as_dict())

    def as_dict(f, native=False, zero=False):
        """
        Switch to a ``dict`` representation.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 - y, x, y).as_dict()
        {(0, 1): -1, (1, 2): 2, (2, 0): 1}

        """
        if native:
            return f.rep.to_dict(zero=zero)
        else:
            return f.rep.to_sympy_dict(zero=zero)

    def as_list(f, native=False):
        """Switch to a ``list`` representation. """
        if native:
            return f.rep.to_list()
        else:
            return f.rep.to_sympy_list()

    def as_expr(f, *gens):
        """
        Convert a Poly instance to an Expr instance.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2 + 2*x*y**2 - y, x, y)

        >>> f.as_expr()
        x**2 + 2*x*y**2 - y
        >>> f.as_expr({x: 5})
        10*y**2 - y + 25
        >>> f.as_expr(5, 6)
        379

        """
        if not gens:
            gens = f.gens
        elif len(gens) == 1 and isinstance(gens[0], dict):
            mapping = gens[0]
            gens = list(f.gens)

            for gen, value in mapping.items():
                try:
                    index = gens.index(gen)
                except ValueError:
                    raise GeneratorsError(
                        "%s doesn't have %s as generator" % (f, gen))
                else:
                    gens[index] = value

        return basic_from_dict(f.rep.to_sympy_dict(), *gens)

    def lift(f):
        """
        Convert algebraic coefficients to rationals.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**2 + I*x + 1, x, extension=I).lift()
        Poly(x**4 + 3*x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'lift'):
            result = f.rep.lift()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'lift')

        return f.per(result)

    def deflate(f):
        """
        Reduce degree of ``f`` by mapping ``x_i**m`` to ``y_i``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3 + 1, x, y).deflate()
        ((3, 2), Poly(x**2*y + x + 1, x, y, domain='ZZ'))

        """
        if hasattr(f.rep, 'deflate'):
            J, result = f.rep.deflate()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'deflate')

        return J, f.per(result)
```
### 3 - sympy/polys/polytools.py:

Start line: 4121, End line: 4151

```python
@public
class PurePoly(Poly):
    """Class for representing pure polynomials. """

    def _hashable_content(self):
        """Allow SymPy to hash Poly instances. """
        return (self.rep,)

    def __hash__(self):
        return super(PurePoly, self).__hash__()

    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial.

        Examples
        ========

        >>> from sympy import PurePoly
        >>> from sympy.abc import x, y

        >>> PurePoly(x**2 + 1).free_symbols
        set()
        >>> PurePoly(x**2 + y).free_symbols
        set()
        >>> PurePoly(x**2 + y, x).free_symbols
        {y}

        """
        return self.free_symbols_in_domain
```
### 4 - sympy/combinatorics/free_groups.py:

Start line: 42, End line: 64

```python
@public
def xfree_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1)))``.

    Parameters
    ----------
    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import xfree_group
    >>> F, (x, y, z) = xfree_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> y**2*x**-2*z**-1
    y**2*x**-2*z**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group, _free_group.generators)
```
### 5 - sympy/polys/polyroots.py:

Start line: 720, End line: 787

```python
def preprocess_roots(poly):
    """Try to get rid of symbolic coefficients from ``poly``. """
    coeff = S.One

    try:
        _, poly = poly.clear_denoms(convert=True)
    except DomainError:
        return coeff, poly

    poly = poly.primitive()[1]
    poly = poly.retract()

    # TODO: This is fragile. Figure out how to make this independent of construct_domain().
    if poly.get_domain().is_Poly and all(c.is_term for c in poly.rep.coeffs()):
        poly = poly.inject()

        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])

        base, strips = strips[0], strips[1:]

        for gen, strip in zip(list(gens), strips):
            reverse = False

            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True

            ratio = None

            for a, b in zip(base, strip):
                if not a and not b:
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a

                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                if reverse:
                    ratio = -ratio

                poly = poly.eval(gen, 1)
                coeff *= gen**(-ratio)
                gens.remove(gen)

        if gens:
            poly = poly.eject(*gens)

    if poly.is_univariate and poly.get_domain().is_ZZ:
        basis = _integer_basis(poly)

        if basis is not None:
            n = poly.degree()

            def func(k, coeff):
                return coeff//basis**(n - k[0])

            poly = poly.termwise(func)
            coeff *= basis

    return coeff, poly
```
### 6 - sympy/combinatorics/free_groups.py:

Start line: 92, End line: 113

```python
def _parse_symbols(symbols):
    if not symbols:
        return tuple()
    if isinstance(symbols, string_types):
        return _symbols(symbols, seq=True)
    elif isinstance(symbols, Expr or FreeGroupElement):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, string_types) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols
    raise ValueError("The type of `symbols` must be one of the following: "
                     "a str, Symbol/Expr or a sequence of "
                     "one of these types")


##############################################################################
#                          FREE GROUP                                        #
##############################################################################

_free_group_cache = {}
```
### 7 - sympy/polys/polytools.py:

Start line: 3663, End line: 4118

```python
@public
class Poly(Expr):

    @property
    def is_zero(f):
        """
        Returns ``True`` if ``f`` is a zero polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_zero
        True
        >>> Poly(1, x).is_zero
        False

        """
        return f.rep.is_zero

    @property
    def is_one(f):
        """
        Returns ``True`` if ``f`` is a unit polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_one
        False
        >>> Poly(1, x).is_one
        True

        """
        return f.rep.is_one

    @property
    def is_sqf(f):
        """
        Returns ``True`` if ``f`` is a square-free polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).is_sqf
        False
        >>> Poly(x**2 - 1, x).is_sqf
        True

        """
        return f.rep.is_sqf

    @property
    def is_monic(f):
        """
        Returns ``True`` if the leading coefficient of ``f`` is one.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 2, x).is_monic
        True
        >>> Poly(2*x + 2, x).is_monic
        False

        """
        return f.rep.is_monic

    @property
    def is_primitive(f):
        """
        Returns ``True`` if GCD of the coefficients of ``f`` is one.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**2 + 6*x + 12, x).is_primitive
        False
        >>> Poly(x**2 + 3*x + 6, x).is_primitive
        True

        """
        return f.rep.is_primitive

    @property
    def is_ground(f):
        """
        Returns ``True`` if ``f`` is an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x, x).is_ground
        False
        >>> Poly(2, x).is_ground
        True
        >>> Poly(y, x).is_ground
        True

        """
        return f.rep.is_ground

    @property
    def is_linear(f):
        """
        Returns ``True`` if ``f`` is linear in all its variables.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x + y + 2, x, y).is_linear
        True
        >>> Poly(x*y + 2, x, y).is_linear
        False

        """
        return f.rep.is_linear

    @property
    def is_quadratic(f):
        """
        Returns ``True`` if ``f`` is quadratic in all its variables.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x*y + 2, x, y).is_quadratic
        True
        >>> Poly(x*y**2 + 2, x, y).is_quadratic
        False

        """
        return f.rep.is_quadratic

    @property
    def is_monomial(f):
        """
        Returns ``True`` if ``f`` is zero or has only one term.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(3*x**2, x).is_monomial
        True
        >>> Poly(3*x**2 + 1, x).is_monomial
        False

        """
        return f.rep.is_monomial

    @property
    def is_homogeneous(f):
        """
        Returns ``True`` if ``f`` is a homogeneous polynomial.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you want not
        only to check if a polynomial is homogeneous but also compute its
        homogeneous order, then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y, x, y).is_homogeneous
        True
        >>> Poly(x**3 + x*y, x, y).is_homogeneous
        False

        """
        return f.rep.is_homogeneous

    @property
    def is_irreducible(f):
        """
        Returns ``True`` if ``f`` has no factors over its domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + x + 1, x, modulus=2).is_irreducible
        True
        >>> Poly(x**2 + 1, x, modulus=2).is_irreducible
        False

        """
        return f.rep.is_irreducible

    @property
    def is_univariate(f):
        """
        Returns ``True`` if ``f`` is a univariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_univariate
        True
        >>> Poly(x*y**2 + x*y + 1, x, y).is_univariate
        False
        >>> Poly(x*y**2 + x*y + 1, x).is_univariate
        True
        >>> Poly(x**2 + x + 1, x, y).is_univariate
        False

        """
        return len(f.gens) == 1

    @property
    def is_multivariate(f):
        """
        Returns ``True`` if ``f`` is a multivariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_multivariate
        False
        >>> Poly(x*y**2 + x*y + 1, x, y).is_multivariate
        True
        >>> Poly(x*y**2 + x*y + 1, x).is_multivariate
        False
        >>> Poly(x**2 + x + 1, x, y).is_multivariate
        True

        """
        return len(f.gens) != 1

    @property
    def is_cyclotomic(f):
        """
        Returns ``True`` if ``f`` is a cyclotomic polnomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1

        >>> Poly(f).is_cyclotomic
        False

        >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1

        >>> Poly(g).is_cyclotomic
        True

        """
        return f.rep.is_cyclotomic

    def __abs__(f):
        return f.abs()

    def __neg__(f):
        return f.neg()

    @_sympifyit('g', NotImplemented)
    def __add__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return f.as_expr() + g

        return f.add(g)

    @_sympifyit('g', NotImplemented)
    def __radd__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return g + f.as_expr()

        return g.add(f)

    @_sympifyit('g', NotImplemented)
    def __sub__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return f.as_expr() - g

        return f.sub(g)

    @_sympifyit('g', NotImplemented)
    def __rsub__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return g - f.as_expr()

        return g.sub(f)

    @_sympifyit('g', NotImplemented)
    def __mul__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return f.as_expr()*g

        return f.mul(g)

    @_sympifyit('g', NotImplemented)
    def __rmul__(f, g):
        if not g.is_Poly:
            try:
                g = f.__class__(g, *f.gens)
            except PolynomialError:
                return g*f.as_expr()

        return g.mul(f)

    @_sympifyit('n', NotImplemented)
    def __pow__(f, n):
        if n.is_Integer and n >= 0:
            return f.pow(n)
        else:
            return f.as_expr()**n

    @_sympifyit('g', NotImplemented)
    def __divmod__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return f.div(g)

    @_sympifyit('g', NotImplemented)
    def __rdivmod__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return g.div(f)

    @_sympifyit('g', NotImplemented)
    def __mod__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return f.rem(g)

    @_sympifyit('g', NotImplemented)
    def __rmod__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return g.rem(f)

    @_sympifyit('g', NotImplemented)
    def __floordiv__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return f.quo(g)

    @_sympifyit('g', NotImplemented)
    def __rfloordiv__(f, g):
        if not g.is_Poly:
            g = f.__class__(g, *f.gens)

        return g.quo(f)

    @_sympifyit('g', NotImplemented)
    def __div__(f, g):
        return f.as_expr()/g.as_expr()

    @_sympifyit('g', NotImplemented)
    def __rdiv__(f, g):
        return g.as_expr()/f.as_expr()

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        f, g = self, other

        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False

        if f.gens != g.gens:
            return False

        if f.rep.dom != g.rep.dom:
            try:
                dom = f.rep.dom.unify(g.rep.dom, f.gens)
            except UnificationFailed:
                return False

            f = f.set_domain(dom)
            g = g.set_domain(dom)

        return f.rep == g.rep

    @_sympifyit('g', NotImplemented)
    def __ne__(f, g):
        return not f.__eq__(g)

    def __nonzero__(f):
        return not f.is_zero

    __bool__ = __nonzero__

    def eq(f, g, strict=False):
        if not strict:
            return f.__eq__(g)
        else:
            return f._strict_eq(sympify(g))

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def _strict_eq(f, g):
        return isinstance(g, f.__class__) and f.gens == g.gens and f.rep.eq(g.rep, strict=True)
```
### 8 - sympy/combinatorics/free_groups.py:

Start line: 18, End line: 40

```python
@public
def free_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1))``.

    Parameters
    ----------
    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> F, x, y, z = free_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> x**2*y**-1
    x**2*y**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group,) + tuple(_free_group.generators)
```
### 9 - sympy/polys/polytools.py:

Start line: 6879, End line: 6958

```python
@public
def poly(expr, *gens, **args):
    """
    Efficiently transform an expression into a polynomial.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.abc import x

    >>> poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    """
    options.allowed_flags(args, [])

    def _poly(expr, opt):
        terms, poly_terms = [], []

        for term in Add.make_args(expr):
            factors, poly_factors = [], []

            for factor in Mul.make_args(term):
                if factor.is_Add:
                    poly_factors.append(_poly(factor, opt))
                elif factor.is_Pow and factor.base.is_Add and factor.exp.is_Integer:
                    poly_factors.append(
                        _poly(factor.base, opt).pow(factor.exp))
                else:
                    factors.append(factor)

            if not poly_factors:
                terms.append(term)
            else:
                product = poly_factors[0]

                for factor in poly_factors[1:]:
                    product = product.mul(factor)

                if factors:
                    factor = Mul(*factors)

                    if factor.is_Number:
                        product = product.mul(factor)
                    else:
                        product = product.mul(Poly._from_expr(factor, opt))

                poly_terms.append(product)

        if not poly_terms:
            result = Poly._from_expr(expr, opt)
        else:
            result = poly_terms[0]

            for term in poly_terms[1:]:
                result = result.add(term)

            if terms:
                term = Add(*terms)

                if term.is_Number:
                    result = result.add(term)
                else:
                    result = result.add(Poly._from_expr(term, opt))

        return result.reorder(*opt.get('gens', ()), **args)

    expr = sympify(expr)

    if expr.is_Poly:
        return Poly(expr, *gens, **args)

    if 'expand' not in args:
        args['expand'] = False

    opt = options.build_options(gens, args)

    return _poly(expr, opt)
```
### 10 - sympy/solvers/polysys.py:

Start line: 101, End line: 167

```python
def solve_generic(polys, opt):
    """
    Solve a generic system of polynomial equations.

    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a
    set F = { f_1, f_2, ..., f_n } of polynomial equations,  using
    Groebner basis approach. For now only zero-dimensional systems
    are supported, which means F can have at most a finite number
    of solutions.

    The algorithm works by the fact that, supposing G is the basis
    of F with respect to an elimination order  (here lexicographic
    order is used), G and F generate the same ideal, they have the
    same set of solutions. By the elimination property,  if G is a
    reduced, zero-dimensional Groebner basis, then there exists an
    univariate polynomial in G (in its last variable). This can be
    solved by computing its roots. Substituting all computed roots
    for the last (eliminated) variable in other elements of G, new
    polynomial system is generated. Applying the above procedure
    recursively, a finite number of solutions can be found.

    The ability of finding all solutions by this procedure depends
    on the root finding algorithms. If no solutions were found, it
    means only that roots() failed, but the system is solvable. To
    overcome this difficulty use numerical algorithms instead.

    References
    ==========

    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short
    Introduction for Systems Theorists, In: R. Moreno-Diaz,
    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,
    February, 2001

    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties
    and Algorithms, Springer, Second Edition, 1997, pp. 112

    Examples
    ========

    >>> from sympy.polys import Poly, Options
    >>> from sympy.solvers.polysys import solve_generic
    >>> from sympy.abc import x, y
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(x - y + 5, x, y, domain='ZZ')
    >>> b = Poly(x + y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(-1, 4)]

    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')
    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(11/3, 13/3)]

    >>> a = Poly(x**2 + y, x, y, domain='ZZ')
    >>> b = Poly(x + y*4, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(0, 0), (1/4, -1/16)]
    """
    def _is_univariate(f):
        """Returns True if 'f' is univariate in its last variable. """
        for monom in f.monoms():
            if any(m > 0 for m in monom[:-1]):
                return False

        return True
    # ... other code
```
### 13 - sympy/polys/polytools.py:

Start line: 4180, End line: 4212

```python
@public
class PurePoly(Poly):

    def _unify(f, g):
        g = sympify(g)

        if not g.is_Poly:
            try:
                return f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g))
            except CoercionFailed:
                raise UnificationFailed("can't unify %s with %s" % (f, g))

        if len(f.gens) != len(g.gens):
            raise UnificationFailed("can't unify %s with %s" % (f, g))

        if not (isinstance(f.rep, DMP) and isinstance(g.rep, DMP)):
            raise UnificationFailed("can't unify %s with %s" % (f, g))

        cls = f.__class__
        gens = f.gens

        dom = f.rep.dom.unify(g.rep.dom, gens)

        F = f.rep.convert(dom)
        G = g.rep.convert(dom)

        def per(rep, dom=dom, gens=gens, remove=None):
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]

                if not gens:
                    return dom.to_sympy(rep)

            return cls.new(rep, *gens)

        return dom, per, F, G
```
### 22 - sympy/polys/polytools.py:

Start line: 4153, End line: 4178

```python
@public
class PurePoly(Poly):

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        f, g = self, other

        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False

        if len(f.gens) != len(g.gens):
            return False

        if f.rep.dom != g.rep.dom:
            try:
                dom = f.rep.dom.unify(g.rep.dom, f.gens)
            except UnificationFailed:
                return False

            f = f.set_domain(dom)
            g = g.set_domain(dom)

        return f.rep == g.rep

    def _strict_eq(f, g):
        return isinstance(g, f.__class__) and f.rep.eq(g.rep, strict=True)
```
### 24 - sympy/polys/polytools.py:

Start line: 1049, End line: 1971

```python
@public
class Poly(Expr):

    def inject(f, front=False):
        """
        Inject ground domain generators into ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x)

        >>> f.inject()
        Poly(x**2*y + x*y**3 + x*y + 1, x, y, domain='ZZ')
        >>> f.inject(front=True)
        Poly(y**3*x + y*x**2 + y*x + 1, y, x, domain='ZZ')

        """
        dom = f.rep.dom

        if dom.is_Numerical:
            return f
        elif not dom.is_Poly:
            raise DomainError("can't inject generators over %s" % dom)

        if hasattr(f.rep, 'inject'):
            result = f.rep.inject(front=front)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'inject')

        if front:
            gens = dom.symbols + f.gens
        else:
            gens = f.gens + dom.symbols

        return f.new(result, *gens)

    def eject(f, *gens):
        """
        Eject selected generators into the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

        >>> f.eject(x)
        Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
        >>> f.eject(y)
        Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

        """
        dom = f.rep.dom

        if not dom.is_Numerical:
            raise DomainError("can't eject generators over %s" % dom)

        n, k = len(f.gens), len(gens)

        if f.gens[:k] == gens:
            _gens, front = f.gens[k:], True
        elif f.gens[-k:] == gens:
            _gens, front = f.gens[:-k], False
        else:
            raise NotImplementedError(
                "can only eject front or back generators")

        dom = dom.inject(*gens)

        if hasattr(f.rep, 'eject'):
            result = f.rep.eject(dom, front=front)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'eject')

        return f.new(result, *_gens)

    def terms_gcd(f):
        """
        Remove GCD of terms from the polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3*y, x, y).terms_gcd()
        ((3, 1), Poly(x**3*y + 1, x, y, domain='ZZ'))

        """
        if hasattr(f.rep, 'terms_gcd'):
            J, result = f.rep.terms_gcd()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'terms_gcd')

        return J, f.per(result)

    def add_ground(f, coeff):
        """
        Add an element of the ground domain to ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).add_ground(2)
        Poly(x + 3, x, domain='ZZ')

        """
        if hasattr(f.rep, 'add_ground'):
            result = f.rep.add_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'add_ground')

        return f.per(result)

    def sub_ground(f, coeff):
        """
        Subtract an element of the ground domain from ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).sub_ground(2)
        Poly(x - 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sub_ground'):
            result = f.rep.sub_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sub_ground')

        return f.per(result)

    def mul_ground(f, coeff):
        """
        Multiply ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).mul_ground(2)
        Poly(2*x + 2, x, domain='ZZ')

        """
        if hasattr(f.rep, 'mul_ground'):
            result = f.rep.mul_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'mul_ground')

        return f.per(result)

    def quo_ground(f, coeff):
        """
        Quotient of ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).quo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).quo_ground(2)
        Poly(x + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'quo_ground'):
            result = f.rep.quo_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'quo_ground')

        return f.per(result)

    def exquo_ground(f, coeff):
        """
        Exact quotient of ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).exquo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).exquo_ground(2)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2 does not divide 3 in ZZ

        """
        if hasattr(f.rep, 'exquo_ground'):
            result = f.rep.exquo_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'exquo_ground')

        return f.per(result)

    def abs(f):
        """
        Make all coefficients in ``f`` positive.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).abs()
        Poly(x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'abs'):
            result = f.rep.abs()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'abs')

        return f.per(result)

    def neg(f):
        """
        Negate all coefficients in ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).neg()
        Poly(-x**2 + 1, x, domain='ZZ')

        >>> -Poly(x**2 - 1, x)
        Poly(-x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'neg'):
            result = f.rep.neg()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'neg')

        return f.per(result)

    def add(f, g):
        """
        Add two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).add(Poly(x - 2, x))
        Poly(x**2 + x - 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) + Poly(x - 2, x)
        Poly(x**2 + x - 1, x, domain='ZZ')

        """
        g = sympify(g)

        if not g.is_Poly:
            return f.add_ground(g)

        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'add'):
            result = F.add(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'add')

        return per(result)

    def sub(f, g):
        """
        Subtract two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).sub(Poly(x - 2, x))
        Poly(x**2 - x + 3, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) - Poly(x - 2, x)
        Poly(x**2 - x + 3, x, domain='ZZ')

        """
        g = sympify(g)

        if not g.is_Poly:
            return f.sub_ground(g)

        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'sub'):
            result = F.sub(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sub')

        return per(result)

    def mul(f, g):
        """
        Multiply two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).mul(Poly(x - 2, x))
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x)*Poly(x - 2, x)
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        """
        g = sympify(g)

        if not g.is_Poly:
            return f.mul_ground(g)

        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'mul'):
            result = F.mul(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'mul')

        return per(result)

    def sqr(f):
        """
        Square a polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x - 2, x).sqr()
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        >>> Poly(x - 2, x)**2
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sqr'):
            result = f.rep.sqr()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sqr')

        return f.per(result)

    def pow(f, n):
        """
        Raise ``f`` to a non-negative power ``n``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x - 2, x).pow(3)
        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')

        >>> Poly(x - 2, x)**3
        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')

        """
        n = int(n)

        if hasattr(f.rep, 'pow'):
            result = f.rep.pow(n)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'pow')

        return f.per(result)

    def pdiv(f, g):
        """
        Polynomial pseudo-division of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).pdiv(Poly(2*x - 4, x))
        (Poly(2*x + 4, x, domain='ZZ'), Poly(20, x, domain='ZZ'))

        """
        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'pdiv'):
            q, r = F.pdiv(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'pdiv')

        return per(q), per(r)

    def prem(f, g):
        """
        Polynomial pseudo-remainder of ``f`` by ``g``.

        Caveat: The function prem(f, g, x) can be safely used to compute
          in Z[x] _only_ subresultant polynomial remainder sequences (prs's).

          To safely compute Euclidean and Sturmian prs's in Z[x]
          employ anyone of the corresponding functions found in
          the module sympy.polys.subresultants_qq_zz. The functions
          in the module with suffix _pg compute prs's in Z[x] employing
          rem(f, g, x), whereas the functions with suffix _amv
          compute prs's in Z[x] employing rem_z(f, g, x).

          The function rem_z(f, g, x) differs from prem(f, g, x) in that
          to compute the remainder polynomials in Z[x] it premultiplies
          the divident times the absolute value of the leading coefficient
          of the divisor raised to the power degree(f, x) - degree(g, x) + 1.


        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).prem(Poly(2*x - 4, x))
        Poly(20, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'prem'):
            result = F.prem(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'prem')

        return per(result)

    def pquo(f, g):
        """
        Polynomial pseudo-quotient of ``f`` by ``g``.

        See the Caveat note in the function prem(f, g).

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).pquo(Poly(2*x - 4, x))
        Poly(2*x + 4, x, domain='ZZ')

        >>> Poly(x**2 - 1, x).pquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'pquo'):
            result = F.pquo(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'pquo')

        return per(result)

    def pexquo(f, g):
        """
        Polynomial exact pseudo-quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).pexquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).pexquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        _, per, F, G = f._unify(g)

        if hasattr(f.rep, 'pexquo'):
            try:
                result = F.pexquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'pexquo')

        return per(result)

    def div(f, g, auto=True):
        """
        Polynomial division with remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x))
        (Poly(1/2*x + 1, x, domain='QQ'), Poly(5, x, domain='QQ'))

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x), auto=False)
        (Poly(0, x, domain='ZZ'), Poly(x**2 + 1, x, domain='ZZ'))

        """
        dom, per, F, G = f._unify(g)
        retract = False

        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        if hasattr(f.rep, 'div'):
            q, r = F.div(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'div')

        if retract:
            try:
                Q, R = q.to_ring(), r.to_ring()
            except CoercionFailed:
                pass
            else:
                q, r = Q, R

        return per(q), per(r)

    def rem(f, g, auto=True):
        """
        Computes the polynomial remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x))
        Poly(5, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x), auto=False)
        Poly(x**2 + 1, x, domain='ZZ')

        """
        dom, per, F, G = f._unify(g)
        retract = False

        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        if hasattr(f.rep, 'rem'):
            r = F.rem(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'rem')

        if retract:
            try:
                r = r.to_ring()
            except CoercionFailed:
                pass

        return per(r)

    def quo(f, g, auto=True):
        """
        Computes polynomial quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).quo(Poly(2*x - 4, x))
        Poly(1/2*x + 1, x, domain='QQ')

        >>> Poly(x**2 - 1, x).quo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        """
        dom, per, F, G = f._unify(g)
        retract = False

        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        if hasattr(f.rep, 'quo'):
            q = F.quo(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'quo')

        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass

        return per(q)

    def exquo(f, g, auto=True):
        """
        Computes polynomial exact quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).exquo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).exquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        dom, per, F, G = f._unify(g)
        retract = False

        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        if hasattr(f.rep, 'exquo'):
            try:
                q = F.exquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'exquo')

        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass

        return per(q)

    def _gen_to_level(f, gen):
        """Returns level associated with the given generator. """
        if isinstance(gen, int):
            length = len(f.gens)

            if -length <= gen < length:
                if gen < 0:
                    return length + gen
                else:
                    return gen
            else:
                raise PolynomialError("-%s <= gen < %s expected, got %s" %
                                      (length, length, gen))
        else:
            try:
                return f.gens.index(sympify(gen))
            except ValueError:
                raise PolynomialError(
                    "a valid generator expected, got %s" % gen)

    def degree(f, gen=0):
        """
        Returns degree of ``f`` in ``x_j``.

        The degree of 0 is negative infinity.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree()
        2
        >>> Poly(x**2 + y*x + y, x, y).degree(y)
        1
        >>> Poly(0, x).degree()
        -oo

        """
        j = f._gen_to_level(gen)

        if hasattr(f.rep, 'degree'):
            return f.rep.degree(j)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'degree')

    def degree_list(f):
        """
        Returns a list of degrees of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree_list()
        (2, 1)

        """
        if hasattr(f.rep, 'degree_list'):
            return f.rep.degree_list()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'degree_list')

    def total_degree(f):
        """
        Returns the total degree of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).total_degree()
        2
        >>> Poly(x + y**5, x, y).total_degree()
        5

        """
        if hasattr(f.rep, 'total_degree'):
            return f.rep.total_degree()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'total_degree')

    def homogenize(f, s):
        """
        Returns the homogeneous polynomial of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you only
        want to check if a polynomial is homogeneous, then use
        :func:`Poly.is_homogeneous`. If you want not only to check if a
        polynomial is homogeneous but also compute its homogeneous order,
        then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> f = Poly(x**5 + 2*x**2*y**2 + 9*x*y**3)
        >>> f.homogenize(z)
        Poly(x**5 + 2*x**2*y**2*z + 9*x*y**3*z, x, y, z, domain='ZZ')

        """
        if not isinstance(s, Symbol):
            raise TypeError("``Symbol`` expected, got %s" % type(s))
        if s in f.gens:
            i = f.gens.index(s)
            gens = f.gens
        else:
            i = len(f.gens)
            gens = f.gens + (s,)
        if hasattr(f.rep, 'homogenize'):
            return f.per(f.rep.homogenize(i), gens=gens)
        raise OperationNotSupported(f, 'homogeneous_order')

    def homogeneous_order(f):
        """
        Returns the homogeneous order of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. This degree is
        the homogeneous order of ``f``. If you only want to check if a
        polynomial is homogeneous, then use :func:`Poly.is_homogeneous`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**5 + 2*x**3*y**2 + 9*x*y**4)
        >>> f.homogeneous_order()
        5

        """
        if hasattr(f.rep, 'homogeneous_order'):
            return f.rep.homogeneous_order()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'homogeneous_order')

    def LC(f, order=None):
        """
        Returns the leading coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(4*x**3 + 2*x**2 + 3*x, x).LC()
        4

        """
        if order is not None:
            return f.coeffs(order)[0]

        if hasattr(f.rep, 'LC'):
            result = f.rep.LC()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'LC')

        return f.rep.dom.to_sympy(result)

    def TC(f):
        """
        Returns the trailing coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).TC()
        0

        """
        if hasattr(f.rep, 'TC'):
            result = f.rep.TC()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'TC')

        return f.rep.dom.to_sympy(result)

    def EC(f, order=None):
        """
        Returns the last non-zero coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).EC()
        3

        """
        if hasattr(f.rep, 'coeffs'):
            return f.coeffs(order)[-1]
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'EC')

    def coeff_monomial(f, monom):
        """
        Returns the coefficient of ``monom`` in ``f`` if there, else None.

        Examples
        ========

        >>> from sympy import Poly, exp
        >>> from sympy.abc import x, y

        >>> p = Poly(24*x*y*exp(8) + 23*x, x, y)

        >>> p.coeff_monomial(x)
        23
        >>> p.coeff_monomial(y)
        0
        >>> p.coeff_monomial(x*y)
        24*exp(8)

        Note that ``Expr.coeff()`` behaves differently, collecting terms
        if possible; the Poly must be converted to an Expr to use that
        method, however:

        >>> p.as_expr().coeff(x)
        24*y*exp(8) + 23
        >>> p.as_expr().coeff(y)
        24*x*exp(8)
        >>> p.as_expr().coeff(x*y)
        24*exp(8)

        See Also
        ========
        nth: more efficient query using exponents of the monomial's generators

        """
        return f.nth(*Monomial(monom, f.gens).exponents)
```
### 25 - sympy/polys/polytools.py:

Start line: 1, End line: 4118

```python
"""User-friendly public interface to polynomial functions. """

from __future__ import print_function, division

from sympy.core import (
    S, Basic, Expr, I, Integer, Add, Mul, Dummy, Tuple
)

from sympy.core.mul import _keep_coeff
from sympy.core.symbol import Symbol
from sympy.core.basic import preorder_traversal
from sympy.core.relational import Relational
from sympy.core.sympify import sympify
from sympy.core.decorators import _sympifyit
from sympy.core.function import Derivative

from sympy.logic.boolalg import BooleanAtom

from sympy.polys.polyclasses import DMP

from sympy.polys.polyutils import (
    basic_from_dict,
    _sort_gens,
    _unify_gens,
    _dict_reorder,
    _dict_from_expr,
    _parallel_dict_from_expr,
)

from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key

from sympy.polys.polyerrors import (
    OperationNotSupported, DomainError,
    CoercionFailed, UnificationFailed,
    GeneratorsNeeded, PolynomialError,
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    GeneratorsError,
)

from sympy.utilities import group, sift, public

import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence

from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.constructor import construct_domain

from sympy.polys import polyoptions as options

from sympy.core.compatibility import iterable, range


@public
class Poly(Expr):
```
### 51 - sympy/polys/polytools.py:

Start line: 4215, End line: 4266

```python
@public
def poly_from_expr(expr, *gens, **args):
    """Construct a polynomial from an expression. """
    opt = options.build_options(gens, args)
    return _poly_from_expr(expr, opt)


def _poly_from_expr(expr, opt):
    """Construct a polynomial from an expression. """
    orig, expr = expr, sympify(expr)

    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)

        opt.gens = poly.gens
        opt.domain = poly.domain

        if opt.polys is None:
            opt.polys = True

        return poly, opt
    elif opt.expand:
        expr = expr.expand()

    rep, opt = _dict_from_expr(expr, opt)
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)

    monoms, coeffs = list(zip(*list(rep.items())))
    domain = opt.domain

    if domain is None:
        opt.domain, coeffs = construct_domain(coeffs, opt=opt)
    else:
        coeffs = list(map(domain.from_sympy, coeffs))

    rep = dict(list(zip(monoms, coeffs)))
    poly = Poly._from_dict(rep, opt)

    if opt.polys is None:
        opt.polys = False

    return poly, opt


@public
def parallel_poly_from_expr(exprs, *gens, **args):
    """Construct polynomials from expressions. """
    opt = options.build_options(gens, args)
    return _parallel_poly_from_expr(exprs, opt)
```
