# sympy__sympy-21931

| **sympy/sympy** | `8cb334cf8b0d8f9be490fecf578aca408069b671` |
| ---- | ---- |
| **No of patches** | 17 |
| **All found context length** | - |
| **Any found context length** | 5121 |
| **Avg pos** | 3.176470588235294 |
| **Min pos** | 4 |
| **Max pos** | 26 |
| **Top file pos** | 2 |
| **Missing snippets** | 63 |
| **Missing patch files** | 15 |


## Expected patch

```diff
diff --git a/sympy/calculus/singularities.py b/sympy/calculus/singularities.py
--- a/sympy/calculus/singularities.py
+++ b/sympy/calculus/singularities.py
@@ -73,13 +73,13 @@ def singularities(expression, symbol, domain=None):
     >>> singularities(x**2 + x + 1, x)
     EmptySet
     >>> singularities(1/(x + 1), x)
-    FiniteSet(-1)
+    {-1}
     >>> singularities(1/(y**2 + 1), y)
-    FiniteSet(I, -I)
+    {-I, I}
     >>> singularities(1/(y**3 + 1), y)
-    FiniteSet(-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2)
+    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
     >>> singularities(log(x), x)
-    FiniteSet(0)
+    {0}
 
     """
     from sympy.functions.elementary.exponential import log
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -735,7 +735,7 @@ def stationary_points(f, symbol, domain=S.Reals):
               2                                2
 
     >>> stationary_points(sin(x),x, Interval(0, 4*pi))
-    FiniteSet(pi/2, 3*pi/2, 5*pi/2, 7*pi/2)
+    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}
 
     """
     from sympy import solveset, diff
@@ -1492,7 +1492,7 @@ def intersection(self, other):
         EmptySet
 
         >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
-        FiniteSet(1, 2)
+        {1, 2}
 
         """
         if not isinstance(other, (AccumBounds, FiniteSet)):
diff --git a/sympy/categories/baseclasses.py b/sympy/categories/baseclasses.py
--- a/sympy/categories/baseclasses.py
+++ b/sympy/categories/baseclasses.py
@@ -522,7 +522,7 @@ def objects(self):
         >>> B = Object("B")
         >>> K = Category("K", FiniteSet(A, B))
         >>> K.objects
-        Class(FiniteSet(Object("A"), Object("B")))
+        Class({Object("A"), Object("B")})
 
         """
         return self.args[1]
@@ -727,7 +727,7 @@ def __new__(cls, *args):
         True
         >>> d = Diagram([f, g], {g * f: "unique"})
         >>> d.conclusions[g * f]
-        FiniteSet(unique)
+        {unique}
 
         """
         premises = {}
@@ -859,7 +859,7 @@ def objects(self):
         >>> g = NamedMorphism(B, C, "g")
         >>> d = Diagram([f, g])
         >>> d.objects
-        FiniteSet(Object("A"), Object("B"), Object("C"))
+        {Object("A"), Object("B"), Object("C")}
 
         """
         return self.args[2]
diff --git a/sympy/combinatorics/partitions.py b/sympy/combinatorics/partitions.py
--- a/sympy/combinatorics/partitions.py
+++ b/sympy/combinatorics/partitions.py
@@ -40,7 +40,7 @@ def __new__(cls, *partition):
         >>> from sympy.combinatorics.partitions import Partition
         >>> a = Partition([1, 2], [3])
         >>> a
-        Partition(FiniteSet(1, 2), FiniteSet(3))
+        Partition({3}, {1, 2})
         >>> a.partition
         [[1, 2], [3]]
         >>> len(a)
@@ -51,7 +51,7 @@ def __new__(cls, *partition):
         Creating Partition from Python sets:
 
         >>> Partition({1, 2, 3}, {4, 5})
-        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
+        Partition({4, 5}, {1, 2, 3})
 
         Creating Partition from SymPy finite sets:
 
@@ -59,7 +59,7 @@ def __new__(cls, *partition):
         >>> a = FiniteSet(1, 2, 3)
         >>> b = FiniteSet(4, 5)
         >>> Partition(a, b)
-        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
+        Partition({4, 5}, {1, 2, 3})
         """
         args = []
         dups = False
@@ -105,7 +105,7 @@ def sort_key(self, order=None):
         >>> d = Partition(list(range(4)))
         >>> l = [d, b, a + 1, a, c]
         >>> l.sort(key=default_sort_key); l
-        [Partition(FiniteSet(1, 2)), Partition(FiniteSet(1), FiniteSet(2)), Partition(FiniteSet(1, x)), Partition(FiniteSet(3, 4)), Partition(FiniteSet(0, 1, 2, 3))]
+        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]
         """
         if order is None:
             members = self.members
@@ -251,7 +251,7 @@ def RGS(self):
         >>> a.RGS
         (0, 0, 1, 2, 2)
         >>> a + 1
-        Partition(FiniteSet(1, 2), FiniteSet(3), FiniteSet(4), FiniteSet(5))
+        Partition({3}, {4}, {5}, {1, 2})
         >>> _.RGS
         (0, 0, 1, 2, 3)
         """
@@ -282,12 +282,12 @@ def from_rgs(self, rgs, elements):
 
         >>> from sympy.combinatorics.partitions import Partition
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
-        Partition(FiniteSet(c), FiniteSet(a, d), FiniteSet(b, e))
+        Partition({c}, {a, d}, {b, e})
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
-        Partition(FiniteSet(e), FiniteSet(a, c), FiniteSet(b, d))
+        Partition({e}, {a, c}, {b, d})
         >>> a = Partition([1, 4], [2], [3, 5])
         >>> Partition.from_rgs(a.RGS, a.members)
-        Partition(FiniteSet(1, 4), FiniteSet(2), FiniteSet(3, 5))
+        Partition({2}, {1, 4}, {3, 5})
         """
         if len(rgs) != len(elements):
             raise ValueError('mismatch in rgs and element lengths')
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -52,9 +52,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
             >>> from sympy.combinatorics.polyhedron import Polyhedron
             >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
-            FiniteSet((0, 1, 2))
+            {(0, 1, 2)}
             >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
-            FiniteSet((0, 1, 2))
+            {(0, 1, 2)}
 
         The allowed transformations are entered as allowable permutations
         of the vertices for the polyhedron. Instance of Permutations
@@ -98,7 +98,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         >>> tetra.size
         4
         >>> tetra.edges
-        FiniteSet((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
+        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
         >>> tetra.corners
         (w, x, y, z)
 
@@ -371,7 +371,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
         >>> from sympy.combinatorics.polyhedron import cube
         >>> cube.edges
-        FiniteSet((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))
+        {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)}
 
         If you want to use letters or other names for the corners you
         can still use the pre-calculated faces:
@@ -498,7 +498,7 @@ def edges(self):
         >>> corners = (a, b, c)
         >>> faces = [(0, 1, 2)]
         >>> Polyhedron(corners, faces).edges
-        FiniteSet((0, 1), (0, 2), (1, 2))
+        {(0, 1), (0, 2), (1, 2)}
 
         """
         if self._edges is None:
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -231,9 +231,9 @@ def nargs(self):
         corresponding set will be returned:
 
         >>> Function('f', nargs=1).nargs
-        FiniteSet(1)
+        {1}
         >>> Function('f', nargs=(2, 1)).nargs
-        FiniteSet(1, 2)
+        {1, 2}
 
         The undefined function, after application, also has the nargs
         attribute; the actual number of arguments is always available by
@@ -1003,7 +1003,7 @@ class WildFunction(Function, AtomicExpr):  # type: ignore
 
     >>> F = WildFunction('F', nargs=2)
     >>> F.nargs
-    FiniteSet(2)
+    {2}
     >>> f(x).match(F)
     >>> f(x, y).match(F)
     {F_: f(x, y)}
@@ -1014,7 +1014,7 @@ class WildFunction(Function, AtomicExpr):  # type: ignore
 
     >>> F = WildFunction('F', nargs=(1, 2))
     >>> F.nargs
-    FiniteSet(1, 2)
+    {1, 2}
     >>> f(x).match(F)
     {F_: f(x)}
     >>> f(x, y).match(F)
diff --git a/sympy/logic/boolalg.py b/sympy/logic/boolalg.py
--- a/sympy/logic/boolalg.py
+++ b/sympy/logic/boolalg.py
@@ -141,7 +141,7 @@ def as_set(self):
         >>> from sympy import Symbol, Eq, Or, And
         >>> x = Symbol('x', real=True)
         >>> Eq(x, 0).as_set()
-        FiniteSet(0)
+        {0}
         >>> (x > 0).as_set()
         Interval.open(0, oo)
         >>> And(-2 < x, x < 2).as_set()
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -7,6 +7,7 @@
 from sympy.core import S, Rational, Pow, Basic, Mul, Number
 from sympy.core.mul import _keep_coeff
 from sympy.core.function import _coeff_isneg
+from sympy.sets.sets import FiniteSet
 from .printer import Printer, print_function
 from sympy.printing.precedence import precedence, PRECEDENCE
 
@@ -796,6 +797,20 @@ def _print_set(self, s):
             return "set()"
         return '{%s}' % args
 
+    def _print_FiniteSet(self, s):
+        items = sorted(s, key=default_sort_key)
+
+        args = ', '.join(self._print(item) for item in items)
+        if any(item.has(FiniteSet) for item in items):
+            return 'FiniteSet({})'.format(args)
+        return '{{{}}}'.format(args)
+
+    def _print_Partition(self, s):
+        items = sorted(s, key=default_sort_key)
+
+        args = ', '.join(self._print(arg) for arg in items)
+        return 'Partition({})'.format(args)
+
     def _print_frozenset(self, s):
         if not s:
             return "frozenset()"
diff --git a/sympy/sets/conditionset.py b/sympy/sets/conditionset.py
--- a/sympy/sets/conditionset.py
+++ b/sympy/sets/conditionset.py
@@ -60,7 +60,7 @@ class ConditionSet(Set):
 
     >>> c = ConditionSet(x, x < 1, {x, z})
     >>> c.subs(x, y)
-    ConditionSet(x, x < 1, FiniteSet(y, z))
+    ConditionSet(x, x < 1, {y, z})
 
     To check if ``pi`` is in ``c`` use:
 
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -306,7 +306,7 @@ class ImageSet(Set):
     False
 
     >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
-    FiniteSet(1, 4, 9)
+    {1, 4, 9}
 
     >>> square_iterable = iter(squares)
     >>> for i in range(4):
@@ -328,7 +328,7 @@ class ImageSet(Set):
     >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
     >>> dom = Interval(-1, 1)
     >>> dom.intersect(solutions)
-    FiniteSet(0)
+    {0}
 
     See Also
     ========
@@ -1021,7 +1021,7 @@ def normalize_theta_set(theta):
     >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
     Interval(pi/2, 3*pi/2)
     >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
-    FiniteSet(0, pi)
+    {0, pi}
 
     """
     from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
@@ -1300,7 +1300,7 @@ def from_real(cls, sets):
         >>> from sympy import Interval, ComplexRegion
         >>> unit = Interval(0,1)
         >>> ComplexRegion.from_real(unit)
-        CartesianComplexRegion(ProductSet(Interval(0, 1), FiniteSet(0)))
+        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))
 
         """
         if not sets.is_subset(S.Reals):
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -41,7 +41,7 @@ class PowerSet(Set):
     A power set of a finite set:
 
     >>> PowerSet(FiniteSet(1, 2, 3))
-    PowerSet(FiniteSet(1, 2, 3))
+    PowerSet({1, 2, 3})
 
     A power set of an empty set:
 
@@ -58,9 +58,7 @@ class PowerSet(Set):
     Evaluating the power set of a finite set to its explicit form:
 
     >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
-    FiniteSet(FiniteSet(1), FiniteSet(1, 2), FiniteSet(1, 3),
-            FiniteSet(1, 2, 3), FiniteSet(2), FiniteSet(2, 3),
-            FiniteSet(3), EmptySet)
+    FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})
 
     References
     ==========
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -101,7 +101,7 @@ def union(self, other):
         >>> Interval(0, 1) + Interval(2, 3)
         Union(Interval(0, 1), Interval(2, 3))
         >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
-        Union(FiniteSet(3), Interval.Lopen(1, 2))
+        Union({3}, Interval.Lopen(1, 2))
 
         Similarly it is possible to use the '-' operator for set differences:
 
@@ -492,7 +492,7 @@ def powerset(self):
 
         >>> A = EmptySet
         >>> A.powerset()
-        FiniteSet(EmptySet)
+        {EmptySet}
 
         A power set of a finite set:
 
@@ -558,9 +558,9 @@ def boundary(self):
 
         >>> from sympy import Interval
         >>> Interval(0, 1).boundary
-        FiniteSet(0, 1)
+        {0, 1}
         >>> Interval(0, 1, True, False).boundary
-        FiniteSet(0, 1)
+        {0, 1}
         """
         return self._boundary
 
@@ -711,7 +711,7 @@ class ProductSet(Set):
     >>> from sympy import Interval, FiniteSet, ProductSet
     >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
     >>> ProductSet(I, S)
-    ProductSet(Interval(0, 5), FiniteSet(1, 2, 3))
+    ProductSet(Interval(0, 5), {1, 2, 3})
 
     >>> (2, 2) in ProductSet(I, S)
     True
@@ -1546,7 +1546,7 @@ class Complement(Set):
 
     >>> from sympy import Complement, FiniteSet
     >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
-    FiniteSet(0, 2)
+    {0, 2}
 
     See Also
     =========
@@ -1748,18 +1748,18 @@ class FiniteSet(Set):
 
     >>> from sympy import FiniteSet
     >>> FiniteSet(1, 2, 3, 4)
-    FiniteSet(1, 2, 3, 4)
+    {1, 2, 3, 4}
     >>> 3 in FiniteSet(1, 2, 3, 4)
     True
 
     >>> members = [1, 2, 3, 4]
     >>> f = FiniteSet(*members)
     >>> f
-    FiniteSet(1, 2, 3, 4)
+    {1, 2, 3, 4}
     >>> f - FiniteSet(2)
-    FiniteSet(1, 3, 4)
+    {1, 3, 4}
     >>> f + FiniteSet(2, 5)
-    FiniteSet(1, 2, 3, 4, 5)
+    {1, 2, 3, 4, 5}
 
     References
     ==========
@@ -1979,7 +1979,7 @@ class SymmetricDifference(Set):
 
     >>> from sympy import SymmetricDifference, FiniteSet
     >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
-    FiniteSet(1, 2, 4, 5)
+    {1, 2, 4, 5}
 
     See Also
     ========
@@ -2050,14 +2050,14 @@ class DisjointUnion(Set):
     >>> A = FiniteSet(1, 2, 3)
     >>> B = Interval(0, 5)
     >>> DisjointUnion(A, B)
-    DisjointUnion(FiniteSet(1, 2, 3), Interval(0, 5))
+    DisjointUnion({1, 2, 3}, Interval(0, 5))
     >>> DisjointUnion(A, B).rewrite(Union)
-    Union(ProductSet(FiniteSet(1, 2, 3), FiniteSet(0)), ProductSet(Interval(0, 5), FiniteSet(1)))
+    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
     >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
     >>> DisjointUnion(C, C)
-    DisjointUnion(FiniteSet(x, y, z), FiniteSet(x, y, z))
+    DisjointUnion({x, y, z}, {x, y, z})
     >>> DisjointUnion(C, C).rewrite(Union)
-    ProductSet(FiniteSet(x, y, z), FiniteSet(0, 1))
+    ProductSet({x, y, z}, {0, 1})
 
     References
     ==========
diff --git a/sympy/solvers/inequalities.py b/sympy/solvers/inequalities.py
--- a/sympy/solvers/inequalities.py
+++ b/sympy/solvers/inequalities.py
@@ -28,13 +28,13 @@ def solve_poly_inequality(poly, rel):
     >>> from sympy.solvers.inequalities import solve_poly_inequality
 
     >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
-    [FiniteSet(0)]
+    [{0}]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
     [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
-    [FiniteSet(-1), FiniteSet(1)]
+    [{-1}, {1}]
 
     See Also
     ========
@@ -140,7 +140,7 @@ def solve_rational_inequalities(eqs):
     >>> solve_rational_inequalities([[
     ... ((Poly(-x + 1), Poly(1, x)), '>='),
     ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
-    FiniteSet(1)
+    {1}
 
     >>> solve_rational_inequalities([[
     ... ((Poly(x), Poly(1, x)), '!='),
diff --git a/sympy/solvers/solveset.py b/sympy/solvers/solveset.py
--- a/sympy/solvers/solveset.py
+++ b/sympy/solvers/solveset.py
@@ -144,14 +144,14 @@ def _invert(f_x, y, x, domain=S.Complexes):
     >>> invert_complex(exp(x), y, x)
     (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
     >>> invert_real(exp(x), y, x)
-    (x, Intersection(FiniteSet(log(y)), Reals))
+    (x, Intersection({log(y)}, Reals))
 
     When does exp(x) == 1?
 
     >>> invert_complex(exp(x), 1, x)
     (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
     >>> invert_real(exp(x), 1, x)
-    (x, FiniteSet(0))
+    (x, {0})
 
     See Also
     ========
@@ -914,7 +914,7 @@ def solve_decomposition(f, symbol, domain):
     >>> x = Symbol('x')
     >>> f1 = exp(2*x) - 3*exp(x) + 2
     >>> sd(f1, x, S.Reals)
-    FiniteSet(0, log(2))
+    {0, log(2)}
     >>> f2 = sin(x)**2 + 2*sin(x) + 1
     >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
               3*pi
@@ -1492,11 +1492,11 @@ def _solve_exponential(lhs, rhs, symbol, domain):
     >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
     ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
     >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
-    ConditionSet(x, (a > 0) & (b > 0), FiniteSet(0))
+    ConditionSet(x, (a > 0) & (b > 0), {0})
     >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
-    FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
+    {-3*log(2)/(-2*log(3) + log(2))}
     >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
-    FiniteSet(0)
+    {0}
 
     * Proof of correctness of the method
 
@@ -1654,7 +1654,7 @@ def _solve_logarithm(lhs, rhs, symbol, domain):
     >>> x = symbols('x')
     >>> f = log(x - 3) + log(x + 3)
     >>> solve_log(f, 0, x, S.Reals)
-    FiniteSet(sqrt(10), -sqrt(10))
+    {-sqrt(10), sqrt(10)}
 
     * Proof of correctness
 
@@ -1900,7 +1900,7 @@ def _transolve(f, symbol, domain):
     >>> from sympy import symbols, S, pprint
     >>> x = symbols('x', real=True) # assumption added
     >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
-    FiniteSet(-(log(3) + 3*log(5))/(-log(5) + 2*log(3)))
+    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}
 
     How ``_transolve`` works
     ========================
@@ -2142,9 +2142,9 @@ def solveset(f, symbol=None, domain=S.Complexes):
     >>> R = S.Reals
     >>> x = Symbol('x')
     >>> solveset(exp(x) - 1, x, R)
-    FiniteSet(0)
+    {0}
     >>> solveset_real(exp(x) - 1, x)
-    FiniteSet(0)
+    {0}
 
     The solution is unaffected by assumptions on the symbol:
 
@@ -2673,7 +2673,7 @@ def linsolve(system, *symbols):
     [6],
     [9]])
     >>> linsolve((A, b), [x, y, z])
-    FiniteSet((-1, 2, 0))
+    {(-1, 2, 0)}
 
     * Parametric Solution: In case the system is underdetermined, the
       function will return a parametric solution in terms of the given
@@ -2684,20 +2684,20 @@ def linsolve(system, *symbols):
     >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     >>> b = Matrix([3, 6, 9])
     >>> linsolve((A, b), x, y, z)
-    FiniteSet((z - 1, 2 - 2*z, z))
+    {(z - 1, 2 - 2*z, z)}
 
     If no symbols are given, internally generated symbols will be used.
     The `tau0` in the 3rd position indicates (as before) that the 3rd
     variable -- whatever it's named -- can take on any value:
 
     >>> linsolve((A, b))
-    FiniteSet((tau0 - 1, 2 - 2*tau0, tau0))
+    {(tau0 - 1, 2 - 2*tau0, tau0)}
 
     * List of Equations as input
 
     >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
     >>> linsolve(Eqns, x, y, z)
-    FiniteSet((1, -2, -2))
+    {(1, -2, -2)}
 
     * Augmented Matrix as input
 
@@ -2708,21 +2708,21 @@ def linsolve(system, *symbols):
     [2, 6,  8, 3],
     [6, 8, 18, 5]])
     >>> linsolve(aug, x, y, z)
-    FiniteSet((3/10, 2/5, 0))
+    {(3/10, 2/5, 0)}
 
     * Solve for symbolic coefficients
 
     >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
     >>> eqns = [a*x + b*y - c, d*x + e*y - f]
     >>> linsolve(eqns, x, y)
-    FiniteSet(((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d)))
+    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}
 
     * A degenerate system returns solution as set of given
       symbols.
 
     >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
     >>> linsolve(system, x, y)
-    FiniteSet((x, y))
+    {(x, y)}
 
     * For an empty system linsolve returns empty set
 
@@ -2733,7 +2733,7 @@ def linsolve(system, *symbols):
       is detected:
 
     >>> linsolve([x*(1/x - 1), (y - 1)**2 - y**2 + 1], x, y)
-    FiniteSet((1, 1))
+    {(1, 1)}
     >>> linsolve([x**2 - 1], x)
     Traceback (most recent call last):
     ...
@@ -2906,33 +2906,33 @@ def substitution(system, symbols, result=[{}], known_symbols=[],
     >>> x, y = symbols('x, y', real=True)
     >>> from sympy.solvers.solveset import substitution
     >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
-    FiniteSet((-1, 1))
+    {(-1, 1)}
 
     * when you want soln should not satisfy eq `x + 1 = 0`
 
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
     EmptySet
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
-    FiniteSet((1, -1))
+    {(1, -1)}
     >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
-    FiniteSet((-3, 4), (2, -1))
+    {(-3, 4), (2, -1)}
 
     * Returns both real and complex solution
 
     >>> x, y, z = symbols('x, y, z')
     >>> from sympy import exp, sin
     >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
-    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
-            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
+    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
+     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
 
     >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
     >>> substitution(eqs, [y, z])
-    FiniteSet((-log(3), sqrt(-exp(2*x) - sin(log(3)))),
-    (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
-    (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-       ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
-    (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-       ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)))
+    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
+     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
+      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
+     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
+      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}
 
     """
 
@@ -3527,7 +3527,7 @@ def nonlinsolve(system, *symbols):
     >>> from sympy.solvers.solveset import nonlinsolve
     >>> x, y, z = symbols('x, y, z', real=True)
     >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
-    FiniteSet((-1, -1), (-1/2, -2), (1/2, 2), (1, 1))
+    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}
 
     1. Positive dimensional system and complements:
 
@@ -3546,7 +3546,7 @@ def nonlinsolve(system, *symbols):
     {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
        d       d               d       d
     >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
-    FiniteSet((2 - y, y))
+    {(2 - y, y)}
 
     2. If some of the equations are non-polynomial then `nonlinsolve`
     will call the `substitution` function and return real and complex solutions,
@@ -3554,9 +3554,8 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import exp, sin
     >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
-    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
-            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
-
+    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
+     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
 
     3. If system is non-linear polynomial and zero-dimensional then it
     returns both solution (real and complex solutions, if present) using
@@ -3564,7 +3563,7 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import sqrt
     >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
-    FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
+    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
 
     4. `nonlinsolve` can solve some linear (zero or positive dimensional)
     system (because it uses the `groebner` function to get the
@@ -3573,7 +3572,7 @@ def nonlinsolve(system, *symbols):
     `nonlinsolve`, because `linsolve` is better for general linear systems.
 
     >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9 , y + z - 4], [x, y, z])
-    FiniteSet((3*z - 5, 4 - z, z))
+    {(3*z - 5, 4 - z, z)}
 
     5. System having polynomial equations and only real solution is
     solved using `solve_poly_system`:
@@ -3581,11 +3580,11 @@ def nonlinsolve(system, *symbols):
     >>> e1 = sqrt(x**2 + y**2) - 10
     >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
     >>> nonlinsolve((e1, e2), (x, y))
-    FiniteSet((191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20))
+    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
-    FiniteSet((1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5)))
+    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
-    FiniteSet((2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5)))
+    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}
 
     6. It is better to use symbols instead of Trigonometric Function or
     Function (e.g. replace `sin(x)` with symbol, replace `f(x)` with symbol
diff --git a/sympy/stats/rv_interface.py b/sympy/stats/rv_interface.py
--- a/sympy/stats/rv_interface.py
+++ b/sympy/stats/rv_interface.py
@@ -411,10 +411,10 @@ def median(X, evaluate=True, **kwargs):
     >>> from sympy.stats import Normal, Die, median
     >>> N = Normal('N', 3, 1)
     >>> median(N)
-    FiniteSet(3)
+    {3}
     >>> D = Die('D')
     >>> median(D)
-    FiniteSet(3, 4)
+    {3, 4}
 
     References
     ==========
diff --git a/sympy/stats/stochastic_process_types.py b/sympy/stats/stochastic_process_types.py
--- a/sympy/stats/stochastic_process_types.py
+++ b/sympy/stats/stochastic_process_types.py
@@ -816,7 +816,7 @@ class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
     >>> YS = DiscreteMarkovChain("Y")
 
     >>> Y.state_space
-    FiniteSet(0, 1, 2)
+    {0, 1, 2}
     >>> Y.transition_probabilities
     Matrix([
     [0.5, 0.2, 0.3],
@@ -1489,7 +1489,7 @@ class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
     >>> C.limiting_distribution()
     Matrix([[1/2, 1/2]])
     >>> C.state_space
-    FiniteSet(0, 1)
+    {0, 1}
     >>> C.generator_matrix
     Matrix([
     [-1,  1],
@@ -1613,7 +1613,7 @@ class BernoulliProcess(DiscreteTimeStochasticProcess):
     >>> from sympy import Eq, Gt
     >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
     >>> B.state_space
-    FiniteSet(0, 1)
+    {0, 1}
     >>> (B.p).round(2)
     0.70
     >>> B.success
diff --git a/sympy/vector/implicitregion.py b/sympy/vector/implicitregion.py
--- a/sympy/vector/implicitregion.py
+++ b/sympy/vector/implicitregion.py
@@ -36,7 +36,7 @@ class ImplicitRegion(Basic):
     >>> r.variables
     (x, y, z)
     >>> r.singular_points()
-    FiniteSet((0, 0, 0))
+    {(0, 0, 0)}
     >>> r.regular_point()
     (-10, -10, 200)
 
@@ -288,7 +288,7 @@ def singular_points(self):
         >>> from sympy.vector import ImplicitRegion
         >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
         >>> I.singular_points()
-        FiniteSet((1, 1))
+        {(1, 1)}
 
         """
         eq_list = [self.equation]
@@ -311,7 +311,7 @@ def multiplicity(self, point):
         >>> from sympy.vector import ImplicitRegion
         >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
         >>> I.singular_points()
-        FiniteSet((0, 0, 0))
+        {(0, 0, 0)}
         >>> I.multiplicity((0, 0, 0))
         2
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/calculus/singularities.py | 76 | 82 | - | - | -
| sympy/calculus/util.py | 738 | 738 | - | - | -
| sympy/calculus/util.py | 1495 | 1495 | - | - | -
| sympy/categories/baseclasses.py | 525 | 525 | - | - | -
| sympy/categories/baseclasses.py | 730 | 730 | - | - | -
| sympy/categories/baseclasses.py | 862 | 862 | - | - | -
| sympy/combinatorics/partitions.py | 43 | 43 | 4 | 2 | 5121
| sympy/combinatorics/partitions.py | 54 | 54 | 4 | 2 | 5121
| sympy/combinatorics/partitions.py | 62 | 62 | 4 | 2 | 5121
| sympy/combinatorics/partitions.py | 108 | 108 | - | 2 | -
| sympy/combinatorics/partitions.py | 254 | 254 | 26 | 2 | 13672
| sympy/combinatorics/partitions.py | 285 | 290 | 16 | 2 | 9391
| sympy/combinatorics/polyhedron.py | 55 | 57 | - | - | -
| sympy/combinatorics/polyhedron.py | 101 | 101 | - | - | -
| sympy/combinatorics/polyhedron.py | 374 | 374 | - | - | -
| sympy/combinatorics/polyhedron.py | 501 | 501 | - | - | -
| sympy/core/function.py | 234 | 236 | - | - | -
| sympy/core/function.py | 1006 | 1006 | - | - | -
| sympy/core/function.py | 1017 | 1017 | - | - | -
| sympy/logic/boolalg.py | 144 | 144 | - | - | -
| sympy/printing/str.py | 10 | 10 | - | 9 | -
| sympy/printing/str.py | 799 | 799 | - | 9 | -
| sympy/sets/conditionset.py | 63 | 63 | - | - | -
| sympy/sets/fancysets.py | 309 | 309 | - | - | -
| sympy/sets/fancysets.py | 331 | 331 | - | - | -
| sympy/sets/fancysets.py | 1024 | 1024 | - | - | -
| sympy/sets/fancysets.py | 1303 | 1303 | - | - | -
| sympy/sets/powerset.py | 44 | 44 | - | - | -
| sympy/sets/powerset.py | 61 | 63 | - | - | -
| sympy/sets/sets.py | 104 | 104 | - | - | -
| sympy/sets/sets.py | 495 | 495 | - | - | -
| sympy/sets/sets.py | 561 | 563 | - | - | -
| sympy/sets/sets.py | 714 | 714 | - | - | -
| sympy/sets/sets.py | 1549 | 1549 | - | - | -
| sympy/sets/sets.py | 1751 | 1762 | - | - | -
| sympy/sets/sets.py | 1982 | 1982 | - | - | -
| sympy/sets/sets.py | 2053 | 2060 | - | - | -
| sympy/solvers/inequalities.py | 31 | 37 | - | - | -
| sympy/solvers/inequalities.py | 143 | 143 | - | - | -
| sympy/solvers/solveset.py | 147 | 154 | - | - | -
| sympy/solvers/solveset.py | 917 | 917 | - | - | -
| sympy/solvers/solveset.py | 1495 | 1499 | - | - | -
| sympy/solvers/solveset.py | 1657 | 1657 | - | - | -
| sympy/solvers/solveset.py | 1903 | 1903 | - | - | -
| sympy/solvers/solveset.py | 2145 | 2147 | - | - | -
| sympy/solvers/solveset.py | 2676 | 2676 | - | - | -
| sympy/solvers/solveset.py | 2687 | 2700 | - | - | -
| sympy/solvers/solveset.py | 2711 | 2725 | - | - | -
| sympy/solvers/solveset.py | 2736 | 2736 | - | - | -
| sympy/solvers/solveset.py | 2909 | 2935 | - | - | -
| sympy/solvers/solveset.py | 3530 | 3530 | - | - | -
| sympy/solvers/solveset.py | 3549 | 3549 | - | - | -
| sympy/solvers/solveset.py | 3557 | 3559 | - | - | -
| sympy/solvers/solveset.py | 3567 | 3567 | - | - | -
| sympy/solvers/solveset.py | 3576 | 3576 | - | - | -
| sympy/solvers/solveset.py | 3584 | 3588 | - | - | -
| sympy/stats/rv_interface.py | 414 | 417 | - | - | -
| sympy/stats/stochastic_process_types.py | 819 | 819 | - | - | -
| sympy/stats/stochastic_process_types.py | 1492 | 1492 | - | - | -
| sympy/stats/stochastic_process_types.py | 1616 | 1616 | - | - | -
| sympy/vector/implicitregion.py | 39 | 39 | - | - | -
| sympy/vector/implicitregion.py | 291 | 291 | - | - | -
| sympy/vector/implicitregion.py | 314 | 314 | - | - | -


## Problem Statement

```
nicer printing of Permutation (and others)
Perhaps Partition's args print with FiniteSet because the args were made to be SymPy types. But the printing need not be so verbose. 

\`\`\`python
>>> Partition([1,2])
Partition(FiniteSet(1, 2))
>>> Partition({1,2})
Partition(FiniteSet(1, 2))
\`\`\`
Printing of its (and other combinatoric funcs as pertinent) args can be done with lists, tuples or sets as community preferences dictate, e.g. `Partition([1,2])` or `Partition({1,2})`, the latter more suggestive that the parts of the Partition are subsets of the set from which they were taken.
nicer printing of Permutation (and others)
Perhaps Partition's args print with FiniteSet because the args were made to be SymPy types. But the printing need not be so verbose. 

\`\`\`python
>>> Partition([1,2])
Partition(FiniteSet(1, 2))
>>> Partition({1,2})
Partition(FiniteSet(1, 2))
\`\`\`
Printing of its (and other combinatoric funcs as pertinent) args can be done with lists, tuples or sets as community preferences dictate, e.g. `Partition([1,2])` or `Partition({1,2})`, the latter more suggestive that the parts of the Partition are subsets of the set from which they were taken.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/combinatorics/permutations.py | 472 | 881| 4021 | 4021 | 24277 | 
| 2 | **2 sympy/combinatorics/partitions.py** | 117 | 155| 266 | 4287 | 30019 | 
| 3 | 3 sympy/functions/combinatorial/numbers.py | 1372 | 1416| 218 | 4505 | 49001 | 
| **-> 4 <-** | **3 sympy/combinatorics/partitions.py** | 1 | 88| 616 | 5121 | 49001 | 
| 5 | 4 sympy/combinatorics/__init__.py | 1 | 41| 407 | 5528 | 49408 | 
| 6 | 4 sympy/combinatorics/permutations.py | 961 | 1007| 449 | 5977 | 49408 | 
| 7 | 4 sympy/functions/combinatorial/numbers.py | 1348 | 1370| 212 | 6189 | 49408 | 
| 8 | 5 sympy/printing/pretty/pretty.py | 2098 | 2125| 250 | 6439 | 74704 | 
| 9 | 6 sympy/printing/repr.py | 61 | 96| 369 | 6808 | 77610 | 
| 10 | 6 sympy/printing/pretty/pretty.py | 391 | 422| 263 | 7071 | 77610 | 
| 11 | 6 sympy/functions/combinatorial/numbers.py | 1307 | 1346| 295 | 7366 | 77610 | 
| 12 | 6 sympy/combinatorics/permutations.py | 1100 | 1141| 306 | 7672 | 77610 | 
| 13 | 6 sympy/combinatorics/permutations.py | 1060 | 1098| 335 | 8007 | 77610 | 
| 14 | **6 sympy/combinatorics/partitions.py** | 546 | 576| 196 | 8203 | 77610 | 
| 15 | 6 sympy/combinatorics/permutations.py | 883 | 960| 807 | 9010 | 77610 | 
| **-> 16 <-** | **6 sympy/combinatorics/partitions.py** | 266 | 302| 381 | 9391 | 77610 | 
| 17 | 7 sympy/combinatorics/perm_groups.py | 1 | 4988| 250 | 9641 | 123284 | 
| 18 | 7 sympy/combinatorics/permutations.py | 1606 | 1645| 380 | 10021 | 123284 | 
| 19 | 8 sympy/utilities/iterables.py | 1639 | 1716| 694 | 10715 | 145288 | 
| 20 | **8 sympy/combinatorics/partitions.py** | 157 | 174| 124 | 10839 | 145288 | 
| 21 | 8 sympy/combinatorics/permutations.py | 1276 | 1297| 177 | 11016 | 145288 | 
| 22 | **9 sympy/printing/str.py** | 409 | 445| 379 | 11395 | 153492 | 
| 23 | 9 sympy/combinatorics/permutations.py | 1035 | 1058| 244 | 11639 | 153492 | 
| 24 | 10 sympy/utilities/enumerative.py | 1 | 129| 1509 | 13148 | 163716 | 
| 25 | 10 sympy/combinatorics/permutations.py | 385 | 406| 179 | 13327 | 163716 | 
| **-> 26 <-** | **10 sympy/combinatorics/partitions.py** | 231 | 264| 345 | 13672 | 163716 | 
| 27 | 10 sympy/printing/pretty/pretty.py | 136 | 203| 639 | 14311 | 163716 | 
| 28 | 10 sympy/printing/pretty/pretty.py | 2408 | 2516| 793 | 15104 | 163716 | 
| 29 | **10 sympy/combinatorics/partitions.py** | 305 | 334| 254 | 15358 | 163716 | 
| 30 | 10 sympy/printing/pretty/pretty.py | 2346 | 2396| 366 | 15724 | 163716 | 
| 31 | 10 sympy/combinatorics/permutations.py | 408 | 429| 176 | 15900 | 163716 | 
| 32 | 10 sympy/combinatorics/permutations.py | 1647 | 1660| 138 | 16038 | 163716 | 
| 33 | **10 sympy/combinatorics/partitions.py** | 485 | 501| 151 | 16189 | 163716 | 
| 34 | 10 sympy/printing/pretty/pretty.py | 2127 | 2180| 388 | 16577 | 163716 | 
| 35 | 11 sympy/combinatorics/fp_groups.py | 400 | 417| 180 | 16757 | 175833 | 
| 36 | 11 sympy/combinatorics/fp_groups.py | 362 | 398| 316 | 17073 | 175833 | 
| 37 | 11 sympy/printing/pretty/pretty.py | 372 | 389| 152 | 17225 | 175833 | 
| 38 | 11 sympy/combinatorics/permutations.py | 2295 | 2320| 183 | 17408 | 175833 | 
| 39 | 12 sympy/physics/secondquant.py | 3016 | 3107| 775 | 18183 | 198595 | 
| 40 | 12 sympy/printing/pretty/pretty.py | 2398 | 2406| 116 | 18299 | 198595 | 
| 41 | 12 sympy/printing/pretty/pretty.py | 267 | 333| 576 | 18875 | 198595 | 
| 42 | 13 sympy/printing/__init__.py | 1 | 112| 633 | 19508 | 199228 | 
| 43 | 13 sympy/combinatorics/permutations.py | 1662 | 1709| 403 | 19911 | 199228 | 
| 44 | 13 sympy/combinatorics/fp_groups.py | 419 | 548| 733 | 20644 | 199228 | 


## Missing Patch Files

 * 1: sympy/calculus/singularities.py
 * 2: sympy/calculus/util.py
 * 3: sympy/categories/baseclasses.py
 * 4: sympy/combinatorics/partitions.py
 * 5: sympy/combinatorics/polyhedron.py
 * 6: sympy/core/function.py
 * 7: sympy/logic/boolalg.py
 * 8: sympy/printing/str.py
 * 9: sympy/sets/conditionset.py
 * 10: sympy/sets/fancysets.py
 * 11: sympy/sets/powerset.py
 * 12: sympy/sets/sets.py
 * 13: sympy/solvers/inequalities.py
 * 14: sympy/solvers/solveset.py
 * 15: sympy/stats/rv_interface.py
 * 16: sympy/stats/stochastic_process_types.py
 * 17: sympy/vector/implicitregion.py

### Hint

```
Is it really necessary for FiniteSet to ever print as "FiniteSet" instead of using "{...}" with the str printer? The latter will create a Python set, which will be converted to a SymPy object when mixed with other SymPy operations. It's no different from printing numbers as `1` instead of `Integer(1)`.
Is it really necessary for FiniteSet to ever print as "FiniteSet" instead of using "{...}" with the str printer? The latter will create a Python set, which will be converted to a SymPy object when mixed with other SymPy operations. It's no different from printing numbers as `1` instead of `Integer(1)`.
```

## Patch

```diff
diff --git a/sympy/calculus/singularities.py b/sympy/calculus/singularities.py
--- a/sympy/calculus/singularities.py
+++ b/sympy/calculus/singularities.py
@@ -73,13 +73,13 @@ def singularities(expression, symbol, domain=None):
     >>> singularities(x**2 + x + 1, x)
     EmptySet
     >>> singularities(1/(x + 1), x)
-    FiniteSet(-1)
+    {-1}
     >>> singularities(1/(y**2 + 1), y)
-    FiniteSet(I, -I)
+    {-I, I}
     >>> singularities(1/(y**3 + 1), y)
-    FiniteSet(-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2)
+    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
     >>> singularities(log(x), x)
-    FiniteSet(0)
+    {0}
 
     """
     from sympy.functions.elementary.exponential import log
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -735,7 +735,7 @@ def stationary_points(f, symbol, domain=S.Reals):
               2                                2
 
     >>> stationary_points(sin(x),x, Interval(0, 4*pi))
-    FiniteSet(pi/2, 3*pi/2, 5*pi/2, 7*pi/2)
+    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}
 
     """
     from sympy import solveset, diff
@@ -1492,7 +1492,7 @@ def intersection(self, other):
         EmptySet
 
         >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
-        FiniteSet(1, 2)
+        {1, 2}
 
         """
         if not isinstance(other, (AccumBounds, FiniteSet)):
diff --git a/sympy/categories/baseclasses.py b/sympy/categories/baseclasses.py
--- a/sympy/categories/baseclasses.py
+++ b/sympy/categories/baseclasses.py
@@ -522,7 +522,7 @@ def objects(self):
         >>> B = Object("B")
         >>> K = Category("K", FiniteSet(A, B))
         >>> K.objects
-        Class(FiniteSet(Object("A"), Object("B")))
+        Class({Object("A"), Object("B")})
 
         """
         return self.args[1]
@@ -727,7 +727,7 @@ def __new__(cls, *args):
         True
         >>> d = Diagram([f, g], {g * f: "unique"})
         >>> d.conclusions[g * f]
-        FiniteSet(unique)
+        {unique}
 
         """
         premises = {}
@@ -859,7 +859,7 @@ def objects(self):
         >>> g = NamedMorphism(B, C, "g")
         >>> d = Diagram([f, g])
         >>> d.objects
-        FiniteSet(Object("A"), Object("B"), Object("C"))
+        {Object("A"), Object("B"), Object("C")}
 
         """
         return self.args[2]
diff --git a/sympy/combinatorics/partitions.py b/sympy/combinatorics/partitions.py
--- a/sympy/combinatorics/partitions.py
+++ b/sympy/combinatorics/partitions.py
@@ -40,7 +40,7 @@ def __new__(cls, *partition):
         >>> from sympy.combinatorics.partitions import Partition
         >>> a = Partition([1, 2], [3])
         >>> a
-        Partition(FiniteSet(1, 2), FiniteSet(3))
+        Partition({3}, {1, 2})
         >>> a.partition
         [[1, 2], [3]]
         >>> len(a)
@@ -51,7 +51,7 @@ def __new__(cls, *partition):
         Creating Partition from Python sets:
 
         >>> Partition({1, 2, 3}, {4, 5})
-        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
+        Partition({4, 5}, {1, 2, 3})
 
         Creating Partition from SymPy finite sets:
 
@@ -59,7 +59,7 @@ def __new__(cls, *partition):
         >>> a = FiniteSet(1, 2, 3)
         >>> b = FiniteSet(4, 5)
         >>> Partition(a, b)
-        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
+        Partition({4, 5}, {1, 2, 3})
         """
         args = []
         dups = False
@@ -105,7 +105,7 @@ def sort_key(self, order=None):
         >>> d = Partition(list(range(4)))
         >>> l = [d, b, a + 1, a, c]
         >>> l.sort(key=default_sort_key); l
-        [Partition(FiniteSet(1, 2)), Partition(FiniteSet(1), FiniteSet(2)), Partition(FiniteSet(1, x)), Partition(FiniteSet(3, 4)), Partition(FiniteSet(0, 1, 2, 3))]
+        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]
         """
         if order is None:
             members = self.members
@@ -251,7 +251,7 @@ def RGS(self):
         >>> a.RGS
         (0, 0, 1, 2, 2)
         >>> a + 1
-        Partition(FiniteSet(1, 2), FiniteSet(3), FiniteSet(4), FiniteSet(5))
+        Partition({3}, {4}, {5}, {1, 2})
         >>> _.RGS
         (0, 0, 1, 2, 3)
         """
@@ -282,12 +282,12 @@ def from_rgs(self, rgs, elements):
 
         >>> from sympy.combinatorics.partitions import Partition
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
-        Partition(FiniteSet(c), FiniteSet(a, d), FiniteSet(b, e))
+        Partition({c}, {a, d}, {b, e})
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
-        Partition(FiniteSet(e), FiniteSet(a, c), FiniteSet(b, d))
+        Partition({e}, {a, c}, {b, d})
         >>> a = Partition([1, 4], [2], [3, 5])
         >>> Partition.from_rgs(a.RGS, a.members)
-        Partition(FiniteSet(1, 4), FiniteSet(2), FiniteSet(3, 5))
+        Partition({2}, {1, 4}, {3, 5})
         """
         if len(rgs) != len(elements):
             raise ValueError('mismatch in rgs and element lengths')
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -52,9 +52,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
             >>> from sympy.combinatorics.polyhedron import Polyhedron
             >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
-            FiniteSet((0, 1, 2))
+            {(0, 1, 2)}
             >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
-            FiniteSet((0, 1, 2))
+            {(0, 1, 2)}
 
         The allowed transformations are entered as allowable permutations
         of the vertices for the polyhedron. Instance of Permutations
@@ -98,7 +98,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         >>> tetra.size
         4
         >>> tetra.edges
-        FiniteSet((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
+        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
         >>> tetra.corners
         (w, x, y, z)
 
@@ -371,7 +371,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
         >>> from sympy.combinatorics.polyhedron import cube
         >>> cube.edges
-        FiniteSet((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))
+        {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)}
 
         If you want to use letters or other names for the corners you
         can still use the pre-calculated faces:
@@ -498,7 +498,7 @@ def edges(self):
         >>> corners = (a, b, c)
         >>> faces = [(0, 1, 2)]
         >>> Polyhedron(corners, faces).edges
-        FiniteSet((0, 1), (0, 2), (1, 2))
+        {(0, 1), (0, 2), (1, 2)}
 
         """
         if self._edges is None:
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -231,9 +231,9 @@ def nargs(self):
         corresponding set will be returned:
 
         >>> Function('f', nargs=1).nargs
-        FiniteSet(1)
+        {1}
         >>> Function('f', nargs=(2, 1)).nargs
-        FiniteSet(1, 2)
+        {1, 2}
 
         The undefined function, after application, also has the nargs
         attribute; the actual number of arguments is always available by
@@ -1003,7 +1003,7 @@ class WildFunction(Function, AtomicExpr):  # type: ignore
 
     >>> F = WildFunction('F', nargs=2)
     >>> F.nargs
-    FiniteSet(2)
+    {2}
     >>> f(x).match(F)
     >>> f(x, y).match(F)
     {F_: f(x, y)}
@@ -1014,7 +1014,7 @@ class WildFunction(Function, AtomicExpr):  # type: ignore
 
     >>> F = WildFunction('F', nargs=(1, 2))
     >>> F.nargs
-    FiniteSet(1, 2)
+    {1, 2}
     >>> f(x).match(F)
     {F_: f(x)}
     >>> f(x, y).match(F)
diff --git a/sympy/logic/boolalg.py b/sympy/logic/boolalg.py
--- a/sympy/logic/boolalg.py
+++ b/sympy/logic/boolalg.py
@@ -141,7 +141,7 @@ def as_set(self):
         >>> from sympy import Symbol, Eq, Or, And
         >>> x = Symbol('x', real=True)
         >>> Eq(x, 0).as_set()
-        FiniteSet(0)
+        {0}
         >>> (x > 0).as_set()
         Interval.open(0, oo)
         >>> And(-2 < x, x < 2).as_set()
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -7,6 +7,7 @@
 from sympy.core import S, Rational, Pow, Basic, Mul, Number
 from sympy.core.mul import _keep_coeff
 from sympy.core.function import _coeff_isneg
+from sympy.sets.sets import FiniteSet
 from .printer import Printer, print_function
 from sympy.printing.precedence import precedence, PRECEDENCE
 
@@ -796,6 +797,20 @@ def _print_set(self, s):
             return "set()"
         return '{%s}' % args
 
+    def _print_FiniteSet(self, s):
+        items = sorted(s, key=default_sort_key)
+
+        args = ', '.join(self._print(item) for item in items)
+        if any(item.has(FiniteSet) for item in items):
+            return 'FiniteSet({})'.format(args)
+        return '{{{}}}'.format(args)
+
+    def _print_Partition(self, s):
+        items = sorted(s, key=default_sort_key)
+
+        args = ', '.join(self._print(arg) for arg in items)
+        return 'Partition({})'.format(args)
+
     def _print_frozenset(self, s):
         if not s:
             return "frozenset()"
diff --git a/sympy/sets/conditionset.py b/sympy/sets/conditionset.py
--- a/sympy/sets/conditionset.py
+++ b/sympy/sets/conditionset.py
@@ -60,7 +60,7 @@ class ConditionSet(Set):
 
     >>> c = ConditionSet(x, x < 1, {x, z})
     >>> c.subs(x, y)
-    ConditionSet(x, x < 1, FiniteSet(y, z))
+    ConditionSet(x, x < 1, {y, z})
 
     To check if ``pi`` is in ``c`` use:
 
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -306,7 +306,7 @@ class ImageSet(Set):
     False
 
     >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
-    FiniteSet(1, 4, 9)
+    {1, 4, 9}
 
     >>> square_iterable = iter(squares)
     >>> for i in range(4):
@@ -328,7 +328,7 @@ class ImageSet(Set):
     >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
     >>> dom = Interval(-1, 1)
     >>> dom.intersect(solutions)
-    FiniteSet(0)
+    {0}
 
     See Also
     ========
@@ -1021,7 +1021,7 @@ def normalize_theta_set(theta):
     >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
     Interval(pi/2, 3*pi/2)
     >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
-    FiniteSet(0, pi)
+    {0, pi}
 
     """
     from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
@@ -1300,7 +1300,7 @@ def from_real(cls, sets):
         >>> from sympy import Interval, ComplexRegion
         >>> unit = Interval(0,1)
         >>> ComplexRegion.from_real(unit)
-        CartesianComplexRegion(ProductSet(Interval(0, 1), FiniteSet(0)))
+        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))
 
         """
         if not sets.is_subset(S.Reals):
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -41,7 +41,7 @@ class PowerSet(Set):
     A power set of a finite set:
 
     >>> PowerSet(FiniteSet(1, 2, 3))
-    PowerSet(FiniteSet(1, 2, 3))
+    PowerSet({1, 2, 3})
 
     A power set of an empty set:
 
@@ -58,9 +58,7 @@ class PowerSet(Set):
     Evaluating the power set of a finite set to its explicit form:
 
     >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
-    FiniteSet(FiniteSet(1), FiniteSet(1, 2), FiniteSet(1, 3),
-            FiniteSet(1, 2, 3), FiniteSet(2), FiniteSet(2, 3),
-            FiniteSet(3), EmptySet)
+    FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})
 
     References
     ==========
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -101,7 +101,7 @@ def union(self, other):
         >>> Interval(0, 1) + Interval(2, 3)
         Union(Interval(0, 1), Interval(2, 3))
         >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
-        Union(FiniteSet(3), Interval.Lopen(1, 2))
+        Union({3}, Interval.Lopen(1, 2))
 
         Similarly it is possible to use the '-' operator for set differences:
 
@@ -492,7 +492,7 @@ def powerset(self):
 
         >>> A = EmptySet
         >>> A.powerset()
-        FiniteSet(EmptySet)
+        {EmptySet}
 
         A power set of a finite set:
 
@@ -558,9 +558,9 @@ def boundary(self):
 
         >>> from sympy import Interval
         >>> Interval(0, 1).boundary
-        FiniteSet(0, 1)
+        {0, 1}
         >>> Interval(0, 1, True, False).boundary
-        FiniteSet(0, 1)
+        {0, 1}
         """
         return self._boundary
 
@@ -711,7 +711,7 @@ class ProductSet(Set):
     >>> from sympy import Interval, FiniteSet, ProductSet
     >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
     >>> ProductSet(I, S)
-    ProductSet(Interval(0, 5), FiniteSet(1, 2, 3))
+    ProductSet(Interval(0, 5), {1, 2, 3})
 
     >>> (2, 2) in ProductSet(I, S)
     True
@@ -1546,7 +1546,7 @@ class Complement(Set):
 
     >>> from sympy import Complement, FiniteSet
     >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
-    FiniteSet(0, 2)
+    {0, 2}
 
     See Also
     =========
@@ -1748,18 +1748,18 @@ class FiniteSet(Set):
 
     >>> from sympy import FiniteSet
     >>> FiniteSet(1, 2, 3, 4)
-    FiniteSet(1, 2, 3, 4)
+    {1, 2, 3, 4}
     >>> 3 in FiniteSet(1, 2, 3, 4)
     True
 
     >>> members = [1, 2, 3, 4]
     >>> f = FiniteSet(*members)
     >>> f
-    FiniteSet(1, 2, 3, 4)
+    {1, 2, 3, 4}
     >>> f - FiniteSet(2)
-    FiniteSet(1, 3, 4)
+    {1, 3, 4}
     >>> f + FiniteSet(2, 5)
-    FiniteSet(1, 2, 3, 4, 5)
+    {1, 2, 3, 4, 5}
 
     References
     ==========
@@ -1979,7 +1979,7 @@ class SymmetricDifference(Set):
 
     >>> from sympy import SymmetricDifference, FiniteSet
     >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
-    FiniteSet(1, 2, 4, 5)
+    {1, 2, 4, 5}
 
     See Also
     ========
@@ -2050,14 +2050,14 @@ class DisjointUnion(Set):
     >>> A = FiniteSet(1, 2, 3)
     >>> B = Interval(0, 5)
     >>> DisjointUnion(A, B)
-    DisjointUnion(FiniteSet(1, 2, 3), Interval(0, 5))
+    DisjointUnion({1, 2, 3}, Interval(0, 5))
     >>> DisjointUnion(A, B).rewrite(Union)
-    Union(ProductSet(FiniteSet(1, 2, 3), FiniteSet(0)), ProductSet(Interval(0, 5), FiniteSet(1)))
+    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
     >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
     >>> DisjointUnion(C, C)
-    DisjointUnion(FiniteSet(x, y, z), FiniteSet(x, y, z))
+    DisjointUnion({x, y, z}, {x, y, z})
     >>> DisjointUnion(C, C).rewrite(Union)
-    ProductSet(FiniteSet(x, y, z), FiniteSet(0, 1))
+    ProductSet({x, y, z}, {0, 1})
 
     References
     ==========
diff --git a/sympy/solvers/inequalities.py b/sympy/solvers/inequalities.py
--- a/sympy/solvers/inequalities.py
+++ b/sympy/solvers/inequalities.py
@@ -28,13 +28,13 @@ def solve_poly_inequality(poly, rel):
     >>> from sympy.solvers.inequalities import solve_poly_inequality
 
     >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
-    [FiniteSet(0)]
+    [{0}]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
     [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
-    [FiniteSet(-1), FiniteSet(1)]
+    [{-1}, {1}]
 
     See Also
     ========
@@ -140,7 +140,7 @@ def solve_rational_inequalities(eqs):
     >>> solve_rational_inequalities([[
     ... ((Poly(-x + 1), Poly(1, x)), '>='),
     ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
-    FiniteSet(1)
+    {1}
 
     >>> solve_rational_inequalities([[
     ... ((Poly(x), Poly(1, x)), '!='),
diff --git a/sympy/solvers/solveset.py b/sympy/solvers/solveset.py
--- a/sympy/solvers/solveset.py
+++ b/sympy/solvers/solveset.py
@@ -144,14 +144,14 @@ def _invert(f_x, y, x, domain=S.Complexes):
     >>> invert_complex(exp(x), y, x)
     (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
     >>> invert_real(exp(x), y, x)
-    (x, Intersection(FiniteSet(log(y)), Reals))
+    (x, Intersection({log(y)}, Reals))
 
     When does exp(x) == 1?
 
     >>> invert_complex(exp(x), 1, x)
     (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
     >>> invert_real(exp(x), 1, x)
-    (x, FiniteSet(0))
+    (x, {0})
 
     See Also
     ========
@@ -914,7 +914,7 @@ def solve_decomposition(f, symbol, domain):
     >>> x = Symbol('x')
     >>> f1 = exp(2*x) - 3*exp(x) + 2
     >>> sd(f1, x, S.Reals)
-    FiniteSet(0, log(2))
+    {0, log(2)}
     >>> f2 = sin(x)**2 + 2*sin(x) + 1
     >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
               3*pi
@@ -1492,11 +1492,11 @@ def _solve_exponential(lhs, rhs, symbol, domain):
     >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
     ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
     >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
-    ConditionSet(x, (a > 0) & (b > 0), FiniteSet(0))
+    ConditionSet(x, (a > 0) & (b > 0), {0})
     >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
-    FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
+    {-3*log(2)/(-2*log(3) + log(2))}
     >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
-    FiniteSet(0)
+    {0}
 
     * Proof of correctness of the method
 
@@ -1654,7 +1654,7 @@ def _solve_logarithm(lhs, rhs, symbol, domain):
     >>> x = symbols('x')
     >>> f = log(x - 3) + log(x + 3)
     >>> solve_log(f, 0, x, S.Reals)
-    FiniteSet(sqrt(10), -sqrt(10))
+    {-sqrt(10), sqrt(10)}
 
     * Proof of correctness
 
@@ -1900,7 +1900,7 @@ def _transolve(f, symbol, domain):
     >>> from sympy import symbols, S, pprint
     >>> x = symbols('x', real=True) # assumption added
     >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
-    FiniteSet(-(log(3) + 3*log(5))/(-log(5) + 2*log(3)))
+    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}
 
     How ``_transolve`` works
     ========================
@@ -2142,9 +2142,9 @@ def solveset(f, symbol=None, domain=S.Complexes):
     >>> R = S.Reals
     >>> x = Symbol('x')
     >>> solveset(exp(x) - 1, x, R)
-    FiniteSet(0)
+    {0}
     >>> solveset_real(exp(x) - 1, x)
-    FiniteSet(0)
+    {0}
 
     The solution is unaffected by assumptions on the symbol:
 
@@ -2673,7 +2673,7 @@ def linsolve(system, *symbols):
     [6],
     [9]])
     >>> linsolve((A, b), [x, y, z])
-    FiniteSet((-1, 2, 0))
+    {(-1, 2, 0)}
 
     * Parametric Solution: In case the system is underdetermined, the
       function will return a parametric solution in terms of the given
@@ -2684,20 +2684,20 @@ def linsolve(system, *symbols):
     >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     >>> b = Matrix([3, 6, 9])
     >>> linsolve((A, b), x, y, z)
-    FiniteSet((z - 1, 2 - 2*z, z))
+    {(z - 1, 2 - 2*z, z)}
 
     If no symbols are given, internally generated symbols will be used.
     The `tau0` in the 3rd position indicates (as before) that the 3rd
     variable -- whatever it's named -- can take on any value:
 
     >>> linsolve((A, b))
-    FiniteSet((tau0 - 1, 2 - 2*tau0, tau0))
+    {(tau0 - 1, 2 - 2*tau0, tau0)}
 
     * List of Equations as input
 
     >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
     >>> linsolve(Eqns, x, y, z)
-    FiniteSet((1, -2, -2))
+    {(1, -2, -2)}
 
     * Augmented Matrix as input
 
@@ -2708,21 +2708,21 @@ def linsolve(system, *symbols):
     [2, 6,  8, 3],
     [6, 8, 18, 5]])
     >>> linsolve(aug, x, y, z)
-    FiniteSet((3/10, 2/5, 0))
+    {(3/10, 2/5, 0)}
 
     * Solve for symbolic coefficients
 
     >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
     >>> eqns = [a*x + b*y - c, d*x + e*y - f]
     >>> linsolve(eqns, x, y)
-    FiniteSet(((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d)))
+    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}
 
     * A degenerate system returns solution as set of given
       symbols.
 
     >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
     >>> linsolve(system, x, y)
-    FiniteSet((x, y))
+    {(x, y)}
 
     * For an empty system linsolve returns empty set
 
@@ -2733,7 +2733,7 @@ def linsolve(system, *symbols):
       is detected:
 
     >>> linsolve([x*(1/x - 1), (y - 1)**2 - y**2 + 1], x, y)
-    FiniteSet((1, 1))
+    {(1, 1)}
     >>> linsolve([x**2 - 1], x)
     Traceback (most recent call last):
     ...
@@ -2906,33 +2906,33 @@ def substitution(system, symbols, result=[{}], known_symbols=[],
     >>> x, y = symbols('x, y', real=True)
     >>> from sympy.solvers.solveset import substitution
     >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
-    FiniteSet((-1, 1))
+    {(-1, 1)}
 
     * when you want soln should not satisfy eq `x + 1 = 0`
 
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
     EmptySet
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
-    FiniteSet((1, -1))
+    {(1, -1)}
     >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
-    FiniteSet((-3, 4), (2, -1))
+    {(-3, 4), (2, -1)}
 
     * Returns both real and complex solution
 
     >>> x, y, z = symbols('x, y, z')
     >>> from sympy import exp, sin
     >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
-    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
-            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
+    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
+     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
 
     >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
     >>> substitution(eqs, [y, z])
-    FiniteSet((-log(3), sqrt(-exp(2*x) - sin(log(3)))),
-    (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
-    (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-       ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
-    (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-       ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)))
+    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
+     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
+      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
+     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
+      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}
 
     """
 
@@ -3527,7 +3527,7 @@ def nonlinsolve(system, *symbols):
     >>> from sympy.solvers.solveset import nonlinsolve
     >>> x, y, z = symbols('x, y, z', real=True)
     >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
-    FiniteSet((-1, -1), (-1/2, -2), (1/2, 2), (1, 1))
+    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}
 
     1. Positive dimensional system and complements:
 
@@ -3546,7 +3546,7 @@ def nonlinsolve(system, *symbols):
     {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
        d       d               d       d
     >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
-    FiniteSet((2 - y, y))
+    {(2 - y, y)}
 
     2. If some of the equations are non-polynomial then `nonlinsolve`
     will call the `substitution` function and return real and complex solutions,
@@ -3554,9 +3554,8 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import exp, sin
     >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
-    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
-            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
-
+    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
+     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
 
     3. If system is non-linear polynomial and zero-dimensional then it
     returns both solution (real and complex solutions, if present) using
@@ -3564,7 +3563,7 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import sqrt
     >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
-    FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
+    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
 
     4. `nonlinsolve` can solve some linear (zero or positive dimensional)
     system (because it uses the `groebner` function to get the
@@ -3573,7 +3572,7 @@ def nonlinsolve(system, *symbols):
     `nonlinsolve`, because `linsolve` is better for general linear systems.
 
     >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9 , y + z - 4], [x, y, z])
-    FiniteSet((3*z - 5, 4 - z, z))
+    {(3*z - 5, 4 - z, z)}
 
     5. System having polynomial equations and only real solution is
     solved using `solve_poly_system`:
@@ -3581,11 +3580,11 @@ def nonlinsolve(system, *symbols):
     >>> e1 = sqrt(x**2 + y**2) - 10
     >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
     >>> nonlinsolve((e1, e2), (x, y))
-    FiniteSet((191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20))
+    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
-    FiniteSet((1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5)))
+    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
-    FiniteSet((2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5)))
+    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}
 
     6. It is better to use symbols instead of Trigonometric Function or
     Function (e.g. replace `sin(x)` with symbol, replace `f(x)` with symbol
diff --git a/sympy/stats/rv_interface.py b/sympy/stats/rv_interface.py
--- a/sympy/stats/rv_interface.py
+++ b/sympy/stats/rv_interface.py
@@ -411,10 +411,10 @@ def median(X, evaluate=True, **kwargs):
     >>> from sympy.stats import Normal, Die, median
     >>> N = Normal('N', 3, 1)
     >>> median(N)
-    FiniteSet(3)
+    {3}
     >>> D = Die('D')
     >>> median(D)
-    FiniteSet(3, 4)
+    {3, 4}
 
     References
     ==========
diff --git a/sympy/stats/stochastic_process_types.py b/sympy/stats/stochastic_process_types.py
--- a/sympy/stats/stochastic_process_types.py
+++ b/sympy/stats/stochastic_process_types.py
@@ -816,7 +816,7 @@ class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
     >>> YS = DiscreteMarkovChain("Y")
 
     >>> Y.state_space
-    FiniteSet(0, 1, 2)
+    {0, 1, 2}
     >>> Y.transition_probabilities
     Matrix([
     [0.5, 0.2, 0.3],
@@ -1489,7 +1489,7 @@ class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
     >>> C.limiting_distribution()
     Matrix([[1/2, 1/2]])
     >>> C.state_space
-    FiniteSet(0, 1)
+    {0, 1}
     >>> C.generator_matrix
     Matrix([
     [-1,  1],
@@ -1613,7 +1613,7 @@ class BernoulliProcess(DiscreteTimeStochasticProcess):
     >>> from sympy import Eq, Gt
     >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
     >>> B.state_space
-    FiniteSet(0, 1)
+    {0, 1}
     >>> (B.p).round(2)
     0.70
     >>> B.success
diff --git a/sympy/vector/implicitregion.py b/sympy/vector/implicitregion.py
--- a/sympy/vector/implicitregion.py
+++ b/sympy/vector/implicitregion.py
@@ -36,7 +36,7 @@ class ImplicitRegion(Basic):
     >>> r.variables
     (x, y, z)
     >>> r.singular_points()
-    FiniteSet((0, 0, 0))
+    {(0, 0, 0)}
     >>> r.regular_point()
     (-10, -10, 200)
 
@@ -288,7 +288,7 @@ def singular_points(self):
         >>> from sympy.vector import ImplicitRegion
         >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
         >>> I.singular_points()
-        FiniteSet((1, 1))
+        {(1, 1)}
 
         """
         eq_list = [self.equation]
@@ -311,7 +311,7 @@ def multiplicity(self, point):
         >>> from sympy.vector import ImplicitRegion
         >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
         >>> I.singular_points()
-        FiniteSet((0, 0, 0))
+        {(0, 0, 0)}
         >>> I.multiplicity((0, 0, 0))
         2
 

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -5,7 +5,8 @@
     symbols, Wild, WildFunction, zeta, zoo, Dummy, Dict, Tuple, FiniteSet, factor,
     subfactorial, true, false, Equivalent, Xor, Complement, SymmetricDifference,
     AccumBounds, UnevaluatedExpr, Eq, Ne, Quaternion, Subs, MatrixSymbol, MatrixSlice,
-    Q)
+    Q,)
+from sympy.combinatorics.partitions import Partition
 from sympy.core import Expr, Mul
 from sympy.core.parameters import _exp_is_pow
 from sympy.external import import_module
@@ -892,13 +893,19 @@ def test_RandomDomain():
 
 def test_FiniteSet():
     assert str(FiniteSet(*range(1, 51))) == (
-        'FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
+        '{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
         ' 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,'
-        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50)'
+        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}'
     )
-    assert str(FiniteSet(*range(1, 6))) == 'FiniteSet(1, 2, 3, 4, 5)'
+    assert str(FiniteSet(*range(1, 6))) == '{1, 2, 3, 4, 5}'
+    assert str(FiniteSet(*[x*y, x**2])) == '{x**2, x*y}'
+    assert str(FiniteSet(FiniteSet(FiniteSet(x, y), 5), FiniteSet(x,y), 5)
+               ) == 'FiniteSet(5, FiniteSet(5, {x, y}), {x, y})'
 
 
+def test_Partition():
+    assert str(Partition(FiniteSet(x, y), {z})) == 'Partition({z}, {x, y})'
+
 def test_UniversalSet():
     assert str(S.UniversalSet) == 'UniversalSet'
 
@@ -1066,6 +1073,11 @@ def test_issue_14567():
     assert factorial(Sum(-1, (x, 0, 0))) + y  # doesn't raise an error
 
 
+def test_issue_21823():
+    assert str(Partition([1, 2])) == 'Partition({1, 2})'
+    assert str(Partition({1, 2})) == 'Partition({1, 2})'
+
+
 def test_issue_21119_21460():
     ss = lambda x: str(S(x, evaluate=False))
     assert ss('4/2') == '4/2'

```


## Code snippets

### 1 - sympy/combinatorics/permutations.py:

Start line: 472, End line: 881

```python
class Permutation(Atom):
    """
    A permutation, alternatively known as an 'arrangement number' or 'ordering'
    is an arrangement of the elements of an ordered list into a one-to-one
    mapping with itself. The permutation of a given arrangement is given by
    indicating the positions of the elements after re-arrangement [2]_. For
    example, if one started with elements [x, y, a, b] (in that order) and
    they were reordered as [x, y, b, a] then the permutation would be
    [0, 1, 3, 2]. Notice that (in SymPy) the first element is always referred
    to as 0 and the permutation uses the indices of the elements in the
    original ordering, not the elements (a, b, etc...) themselves.

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.interactive import init_printing
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    Permutations Notation
    =====================

    Permutations are commonly represented in disjoint cycle or array forms.

    Array Notation and 2-line Form
    ------------------------------------

    In the 2-line form, the elements and their final positions are shown
    as a matrix with 2 rows:

    [0    1    2     ... n-1]
    [p(0) p(1) p(2)  ... p(n-1)]

    Since the first line is always range(n), where n is the size of p,
    it is sufficient to represent the permutation by the second line,
    referred to as the "array form" of the permutation. This is entered
    in brackets as the argument to the Permutation class:

    >>> p = Permutation([0, 2, 1]); p
    Permutation([0, 2, 1])

    Given i in range(p.size), the permutation maps i to i^p

    >>> [i^p for i in range(p.size)]
    [0, 2, 1]

    The composite of two permutations p*q means first apply p, then q, so
    i^(p*q) = (i^p)^q which is i^p^q according to Python precedence rules:

    >>> q = Permutation([2, 1, 0])
    >>> [i^p^q for i in range(3)]
    [2, 0, 1]
    >>> [i^(p*q) for i in range(3)]
    [2, 0, 1]

    One can use also the notation p(i) = i^p, but then the composition
    rule is (p*q)(i) = q(p(i)), not p(q(i)):

    >>> [(p*q)(i) for i in range(p.size)]
    [2, 0, 1]
    >>> [q(p(i)) for i in range(p.size)]
    [2, 0, 1]
    >>> [p(q(i)) for i in range(p.size)]
    [1, 2, 0]

    Disjoint Cycle Notation
    -----------------------

    In disjoint cycle notation, only the elements that have shifted are
    indicated. In the above case, the 2 and 1 switched places. This can
    be entered in two ways:

    >>> Permutation(1, 2) == Permutation([[1, 2]]) == p
    True

    Only the relative ordering of elements in a cycle matter:

    >>> Permutation(1,2,3) == Permutation(2,3,1) == Permutation(3,1,2)
    True

    The disjoint cycle notation is convenient when representing
    permutations that have several cycles in them:

    >>> Permutation(1, 2)(3, 5) == Permutation([[1, 2], [3, 5]])
    True

    It also provides some economy in entry when computing products of
    permutations that are written in disjoint cycle notation:

    >>> Permutation(1, 2)(1, 3)(2, 3)
    Permutation([0, 3, 2, 1])
    >>> _ == Permutation([[1, 2]])*Permutation([[1, 3]])*Permutation([[2, 3]])
    True

        Caution: when the cycles have common elements
        between them then the order in which the
        permutations are applied matters. The
        convention is that the permutations are
        applied from *right to left*. In the following, the
        transposition of elements 2 and 3 is followed
        by the transposition of elements 1 and 2:

        >>> Permutation(1, 2)(2, 3) == Permutation([(1, 2), (2, 3)])
        True
        >>> Permutation(1, 2)(2, 3).list()
        [0, 3, 1, 2]

        If the first and second elements had been
        swapped first, followed by the swapping of the second
        and third, the result would have been [0, 2, 3, 1].
        If, for some reason, you want to apply the cycles
        in the order they are entered, you can simply reverse
        the order of cycles:

        >>> Permutation([(1, 2), (2, 3)][::-1]).list()
        [0, 2, 3, 1]

    Entering a singleton in a permutation is a way to indicate the size of the
    permutation. The ``size`` keyword can also be used.

    Array-form entry:

    >>> Permutation([[1, 2], [9]])
    Permutation([0, 2, 1], size=10)
    >>> Permutation([[1, 2]], size=10)
    Permutation([0, 2, 1], size=10)

    Cyclic-form entry:

    >>> Permutation(1, 2, size=10)
    Permutation([0, 2, 1], size=10)
    >>> Permutation(9)(1, 2)
    Permutation([0, 2, 1], size=10)

    Caution: no singleton containing an element larger than the largest
    in any previous cycle can be entered. This is an important difference
    in how Permutation and Cycle handle the __call__ syntax. A singleton
    argument at the start of a Permutation performs instantiation of the
    Permutation and is permitted:

    >>> Permutation(5)
    Permutation([], size=6)

    A singleton entered after instantiation is a call to the permutation
    -- a function call -- and if the argument is out of range it will
    trigger an error. For this reason, it is better to start the cycle
    with the singleton:

    The following fails because there is no element 3:

    >>> Permutation(1, 2)(3)
    Traceback (most recent call last):
    ...
    IndexError: list index out of range

    This is ok: only the call to an out of range singleton is prohibited;
    otherwise the permutation autosizes:

    >>> Permutation(3)(1, 2)
    Permutation([0, 2, 1, 3])
    >>> Permutation(1, 2)(3, 4) == Permutation(3, 4)(1, 2)
    True


    Equality testing
    ----------------

    The array forms must be the same in order for permutations to be equal:

    >>> Permutation([1, 0, 2, 3]) == Permutation([1, 0])
    False


    Identity Permutation
    --------------------

    The identity permutation is a permutation in which no element is out of
    place. It can be entered in a variety of ways. All the following create
    an identity permutation of size 4:

    >>> I = Permutation([0, 1, 2, 3])
    >>> all(p == I for p in [
    ... Permutation(3),
    ... Permutation(range(4)),
    ... Permutation([], size=4),
    ... Permutation(size=4)])
    True

    Watch out for entering the range *inside* a set of brackets (which is
    cycle notation):

    >>> I == Permutation([range(4)])
    False


    Permutation Printing
    ====================

    There are a few things to note about how Permutations are printed.

    1) If you prefer one form (array or cycle) over another, you can set
    ``init_printing`` with the ``perm_cyclic`` flag.

    >>> from sympy import init_printing
    >>> p = Permutation(1, 2)(4, 5)(3, 4)
    >>> p
    Permutation([0, 2, 1, 4, 5, 3])

    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (1 2)(3 4 5)

    2) Regardless of the setting, a list of elements in the array for cyclic
    form can be obtained and either of those can be copied and supplied as
    the argument to Permutation:

    >>> p.array_form
    [0, 2, 1, 4, 5, 3]
    >>> p.cyclic_form
    [[1, 2], [3, 4, 5]]
    >>> Permutation(_) == p
    True

    3) Printing is economical in that as little as possible is printed while
    retaining all information about the size of the permutation:

    >>> init_printing(perm_cyclic=False, pretty_print=False)
    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    >>> Permutation([1, 0, 2, 3], size=20)
    Permutation([1, 0], size=20)
    >>> Permutation([1, 0, 2, 4, 3, 5, 6], size=20)
    Permutation([1, 0, 2, 4, 3], size=20)

    >>> p = Permutation([1, 0, 2, 3])
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (3)(0 1)
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    The 2 was not printed but it is still there as can be seen with the
    array_form and size methods:

    >>> p.array_form
    [1, 0, 2, 3]
    >>> p.size
    4

    Short introduction to other methods
    ===================================

    The permutation can act as a bijective function, telling what element is
    located at a given position

    >>> q = Permutation([5, 2, 3, 4, 1, 0])
    >>> q.array_form[1] # the hard way
    2
    >>> q(1) # the easy way
    2
    >>> {i: q(i) for i in range(q.size)} # showing the bijection
    {0: 5, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0}

    The full cyclic form (including singletons) can be obtained:

    >>> p.full_cyclic_form
    [[0, 1], [2], [3]]

    Any permutation can be factored into transpositions of pairs of elements:

    >>> Permutation([[1, 2], [3, 4, 5]]).transpositions()
    [(1, 2), (3, 5), (3, 4)]
    >>> Permutation.rmul(*[Permutation([ti], size=6) for ti in _]).cyclic_form
    [[1, 2], [3, 4, 5]]

    The number of permutations on a set of n elements is given by n! and is
    called the cardinality.

    >>> p.size
    4
    >>> p.cardinality
    24

    A given permutation has a rank among all the possible permutations of the
    same elements, but what that rank is depends on how the permutations are
    enumerated. (There are a number of different methods of doing so.) The
    lexicographic rank is given by the rank method and this rank is used to
    increment a permutation with addition/subtraction:

    >>> p.rank()
    6
    >>> p + 1
    Permutation([1, 0, 3, 2])
    >>> p.next_lex()
    Permutation([1, 0, 3, 2])
    >>> _.rank()
    7
    >>> p.unrank_lex(p.size, rank=7)
    Permutation([1, 0, 3, 2])

    The product of two permutations p and q is defined as their composition as
    functions, (p*q)(i) = q(p(i)) [6]_.

    >>> p = Permutation([1, 0, 2, 3])
    >>> q = Permutation([2, 3, 1, 0])
    >>> list(q*p)
    [2, 3, 0, 1]
    >>> list(p*q)
    [3, 2, 1, 0]
    >>> [q(p(i)) for i in range(p.size)]
    [3, 2, 1, 0]

    The permutation can be 'applied' to any list-like object, not only
    Permutations:

    >>> p(['zero', 'one', 'four', 'two'])
    ['one', 'zero', 'four', 'two']
    >>> p('zo42')
    ['o', 'z', '4', '2']

    If you have a list of arbitrary elements, the corresponding permutation
    can be found with the from_sequence method:

    >>> Permutation.from_sequence('SymPy')
    Permutation([1, 3, 2, 0, 4])

    Checking if a Permutation is contained in a Group
    =================================================

    Generally if you have a group of permutations G on n symbols, and
    you're checking if a permutation on less than n symbols is part
    of that group, the check will fail.

    Here is an example for n=5 and we check if the cycle
    (1,2,3) is in G:

    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> from sympy.combinatorics import Cycle, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> G = PermutationGroup(Cycle(2, 3)(4, 5), Cycle(1, 2, 3, 4, 5))
    >>> p1 = Permutation(Cycle(2, 5, 3))
    >>> p2 = Permutation(Cycle(1, 2, 3))
    >>> a1 = Permutation(Cycle(1, 2, 3).list(6))
    >>> a2 = Permutation(Cycle(1, 2, 3)(5))
    >>> a3 = Permutation(Cycle(1, 2, 3),size=6)
    >>> for p in [p1,p2,a1,a2,a3]: p, G.contains(p)
    ((2 5 3), True)
    ((1 2 3), False)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)

    The check for p2 above will fail.

    Checking if p1 is in G works because SymPy knows
    G is a group on 5 symbols, and p1 is also on 5 symbols
    (its largest element is 5).

    For ``a1``, the ``.list(6)`` call will extend the permutation to 5
    symbols, so the test will work as well. In the case of ``a2`` the
    permutation is being extended to 5 symbols by using a singleton,
    and in the case of ``a3`` it's extended through the constructor
    argument ``size=6``.

    There is another way to do this, which is to tell the ``contains``
    method that the number of symbols the group is on doesn't need to
    match perfectly the number of symbols for the permutation:

    >>> G.contains(p2,strict=False)
    True

    This can be via the ``strict`` argument to the ``contains`` method,
    and SymPy will try to extend the permutation on its own and then
    perform the containment check.

    See Also
    ========

    Cycle

    References
    ==========

    .. [1] Skiena, S. 'Permutations.' 1.1 in Implementing Discrete Mathematics
           Combinatorics and Graph Theory with Mathematica.  Reading, MA:
           Addison-Wesley, pp. 3-16, 1990.

    .. [2] Knuth, D. E. The Art of Computer Programming, Vol. 4: Combinatorial
           Algorithms, 1st ed. Reading, MA: Addison-Wesley, 2011.

    .. [3] Wendy Myrvold and Frank Ruskey. 2001. Ranking and unranking
           permutations in linear time. Inf. Process. Lett. 79, 6 (September 2001),
           281-284. DOI=10.1016/S0020-0190(01)00141-7

    .. [4] D. L. Kreher, D. R. Stinson 'Combinatorial Algorithms'
           CRC Press, 1999

    .. [5] Graham, R. L.; Knuth, D. E.; and Patashnik, O.
           Concrete Mathematics: A Foundation for Computer Science, 2nd ed.
           Reading, MA: Addison-Wesley, 1994.

    .. [6] https://en.wikipedia.org/wiki/Permutation#Product_and_inverse

    .. [7] https://en.wikipedia.org/wiki/Lehmer_code

    """

    is_Permutation = True

    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None
```
### 2 - sympy/combinatorics/partitions.py:

Start line: 117, End line: 155

```python
class Partition(FiniteSet):

    @property
    def partition(self):
        """Return partition as a sorted list of lists.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import Partition
        >>> Partition([1], [2, 3]).partition
        [[1], [2, 3]]
        """
        if self._partition is None:
            self._partition = sorted([sorted(p, key=default_sort_key)
                                      for p in self.args])
        return self._partition

    def __add__(self, other):
        """
        Return permutation whose rank is ``other`` greater than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics.partitions import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a + 1).rank
        2
        >>> (a + 100).rank
        1
        """
        other = as_int(other)
        offset = self.rank + other
        result = RGS_unrank((offset) %
                            RGS_enum(self.size),
                            self.size)
        return Partition.from_rgs(result, self.members)
```
### 3 - sympy/functions/combinatorial/numbers.py:

Start line: 1372, End line: 1416

```python
class partition(Function):

    @classmethod
    def eval(cls, n):
        is_int = n.is_integer
        if is_int == False:
            raise ValueError("Partition numbers are defined only for "
                             "integers")
        elif is_int:
            if n.is_negative:
                return S.Zero

            if n.is_zero or (n - 1).is_zero:
                return S.One

            if n.is_Integer:
                return Integer(cls._partition(n))


    def _eval_is_integer(self):
        if self.args[0].is_integer:
            return True

    def _eval_is_negative(self):
        if self.args[0].is_integer:
            return False

    def _eval_is_positive(self):
        n = self.args[0]
        if n.is_nonnegative and n.is_integer:
            return True


#######################################################################
###
### Functions for enumerating partitions, permutations and combinations
###
#######################################################################


class _MultisetHistogram(tuple):
    pass


_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)
```
### 4 - sympy/combinatorics/partitions.py:

Start line: 1, End line: 88

```python
from sympy.core import Basic, Dict, sympify
from sympy.core.compatibility import as_int, default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group

from collections import defaultdict


class Partition(FiniteSet):
    """
    This class represents an abstract partition.

    A partition is a set of disjoint sets whose union equals a given set.

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions
    """

    _rank = None
    _partition = None

    def __new__(cls, *partition):
        """
        Generates a new partition object.

        This method also verifies if the arguments passed are
        valid and raises a ValueError if they are not.

        Examples
        ========

        Creating Partition from Python lists:

        >>> from sympy.combinatorics.partitions import Partition
        >>> a = Partition([1, 2], [3])
        >>> a
        Partition(FiniteSet(1, 2), FiniteSet(3))
        >>> a.partition
        [[1, 2], [3]]
        >>> len(a)
        2
        >>> a.members
        (1, 2, 3)

        Creating Partition from Python sets:

        >>> Partition({1, 2, 3}, {4, 5})
        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))

        Creating Partition from SymPy finite sets:

        >>> from sympy.sets.sets import FiniteSet
        >>> a = FiniteSet(1, 2, 3)
        >>> b = FiniteSet(4, 5)
        >>> Partition(a, b)
        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
        """
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)
                if len(as_set) < len(arg):
                    dups = True
                    break  # error below
                arg = as_set
            args.append(_sympify(arg))

        if not all(isinstance(part, FiniteSet) for part in args):
            raise ValueError(
                "Each argument to Partition should be " \
                "a list, set, or a FiniteSet")

        # sort so we have a canonical reference for RGS
        U = Union(*args)
        if dups or len(U) < sum(len(arg) for arg in args):
            raise ValueError("Partition contained duplicate elements.")

        obj = FiniteSet.__new__(cls, *args)
        obj.members = tuple(U)
        obj.size = len(U)
        return obj
```
### 5 - sympy/combinatorics/__init__.py:

Start line: 1, End line: 41

```python
from sympy.combinatorics.permutations import Permutation, Cycle
from sympy.combinatorics.prufer import Prufer
from sympy.combinatorics.generators import cyclic, alternating, symmetric, dihedral
from sympy.combinatorics.subsets import Subset
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
    RGS_rank, RGS_unrank, RGS_enum)
from sympy.combinatorics.polyhedron import (Polyhedron, tetrahedron, cube,
    octahedron, dodecahedron, icosahedron)
from sympy.combinatorics.perm_groups import PermutationGroup, Coset, SymmetricPermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.named_groups import (SymmetricGroup, DihedralGroup,
    CyclicGroup, AlternatingGroup, AbelianGroup, RubikGroup)
from sympy.combinatorics.pc_groups import PolycyclicGroup, Collector

__all__ = [
    'Permutation', 'Cycle',

    'Prufer',

    'cyclic', 'alternating', 'symmetric', 'dihedral',

    'Subset',

    'Partition', 'IntegerPartition', 'RGS_rank', 'RGS_unrank', 'RGS_enum',

    'Polyhedron', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron',
    'icosahedron',

    'PermutationGroup', 'Coset', 'SymmetricPermutationGroup',

    'DirectProduct',

    'GrayCode',

    'SymmetricGroup', 'DihedralGroup', 'CyclicGroup', 'AlternatingGroup',
    'AbelianGroup', 'RubikGroup',

    'PolycyclicGroup', 'Collector',
]
```
### 6 - sympy/combinatorics/permutations.py:

Start line: 961, End line: 1007

```python
class Permutation(Atom):

    def __new__(cls, *args, size=None, **kwargs):
        # ... other code
        if not ok:
            raise ValueError("Permutation argument must be a list of ints, "
                             "a list of lists, Permutation or Cycle.")

        # safe to assume args are valid; this also makes a copy
        # of the args
        args = list(args[0])

        is_cycle = args and is_sequence(args[0])
        if is_cycle:  # e
            args = [[int(i) for i in c] for c in args]
        else:  # d
            args = [int(i) for i in args]

        # if there are n elements present, 0, 1, ..., n-1 should be present
        # unless a cycle notation has been provided. A 0 will be added
        # for convenience in case one wants to enter permutations where
        # counting starts from 1.

        temp = flatten(args)
        if has_dups(temp) and not is_cycle:
            raise ValueError('there were repeated elements.')
        temp = set(temp)

        if not is_cycle:
            if any(i not in temp for i in range(len(temp))):
                raise ValueError('Integers 0 through %s must be present.' %
                max(temp))
            if size is not None and temp and max(temp) + 1 > size:
                raise ValueError('max element should not exceed %s' % (size - 1))

        if is_cycle:
            # it's not necessarily canonical so we won't store
            # it -- use the array form instead
            c = Cycle()
            for ci in args:
                c = c(*ci)
            aform = c.list()
        else:
            aform = list(args)
        if size and size > len(aform):
            # don't allow for truncation of permutation which
            # might split a cycle and lead to an invalid aform
            # but do allow the permutation size to be increased
            aform.extend(list(range(len(aform), size)))

        return cls._af_new(aform)
```
### 7 - sympy/functions/combinatorial/numbers.py:

Start line: 1348, End line: 1370

```python
class partition(Function):

    @staticmethod
    def _partition(n):
        L = len(_npartition)
        if n < L:
            return _npartition[n]
        # lengthen cache
        for _n in range(L, n + 1):
            v, p, i = 0, 0, 0
            while 1:
                s = 0
                p += 3*i + 1  # p = pentagonal number: 1, 5, 12, ...
                if _n >= p:
                    s += _npartition[_n - p]
                i += 1
                gp = p + i  # gp = generalized pentagonal: 2, 7, 15, ...
                if _n >= gp:
                    s += _npartition[_n - gp]
                if s == 0:
                    break
                else:
                    v += s if i%2 == 1 else -s
            _npartition.append(v)
        return v
```
### 8 - sympy/printing/pretty/pretty.py:

Start line: 2098, End line: 2125

```python
class PrettyPrinter(Printer):

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_seq(items, '{', '}', ', ' )

    def _print_Range(self, s):

        if self._use_unicode:
            dots = "\N{HORIZONTAL ELLIPSIS}"
        else:
            dots = '...'

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

        return self._print_seq(printset, '{', '}', ', ' )
```
### 9 - sympy/printing/repr.py:

Start line: 61, End line: 96

```python
class ReprPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
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
            if not expr.size:
                return 'Permutation()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' %s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % str(expr.array_form)
                return 'Permutation([], size=%s)' % expr.size
            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
            use = full = str(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use
```
### 10 - sympy/printing/pretty/pretty.py:

Start line: 391, End line: 422

```python
class PrettyPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle

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
            return self._print_Cycle(Cycle(expr))

        lower = expr.array_form
        upper = list(range(len(lower)))

        result = stringPict('')
        first = True
        for u, l in zip(upper, lower):
            s1 = self._print(u)
            s2 = self._print(l)
            col = prettyForm(*s1.below(s2))
            if first:
                first = False
            else:
                col = prettyForm(*col.left(" "))
            result = prettyForm(*result.right(col))
        return prettyForm(*result.parens())
```
### 14 - sympy/combinatorics/partitions.py:

Start line: 546, End line: 576

```python
class IntegerPartition(Basic):

    def __le__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([4])
        >>> a <= a
        True
        """
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())
        #####
        #
        #
        """
        return "\n".join([char*i for i in self.partition])

    def __str__(self):
        return str(list(self.partition))
```
### 16 - sympy/combinatorics/partitions.py:

Start line: 266, End line: 302

```python
class Partition(FiniteSet):

    @classmethod
    def from_rgs(self, rgs, elements):
        """
        Creates a set partition from a restricted growth string.

        Explanation
        ===========

        The indices given in rgs are assumed to be the index
        of the element as given in elements *as provided* (the
        elements are not sorted by this routine). Block numbering
        starts from 0. If any block was not referenced in ``rgs``
        an error will be raised.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import Partition
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
        Partition(FiniteSet(c), FiniteSet(a, d), FiniteSet(b, e))
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
        Partition(FiniteSet(e), FiniteSet(a, c), FiniteSet(b, d))
        >>> a = Partition([1, 4], [2], [3, 5])
        >>> Partition.from_rgs(a.RGS, a.members)
        Partition(FiniteSet(1, 4), FiniteSet(2), FiniteSet(3, 5))
        """
        if len(rgs) != len(elements):
            raise ValueError('mismatch in rgs and element lengths')
        max_elem = max(rgs) + 1
        partition = [[] for i in range(max_elem)]
        j = 0
        for i in rgs:
            partition[i].append(elements[j])
            j += 1
        if not all(p for p in partition):
            raise ValueError('some blocks of the partition were empty.')
        return Partition(*partition)
```
### 20 - sympy/combinatorics/partitions.py:

Start line: 157, End line: 174

```python
class Partition(FiniteSet):

    def __sub__(self, other):
        """
        Return permutation whose rank is ``other`` less than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics.partitions import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a - 1).rank
        0
        >>> (a - 100).rank
        1
        """
        return self.__add__(-other)
```
### 22 - sympy/printing/str.py:

Start line: 409, End line: 445

```python
class StrPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
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
            if not expr.size:
                return '()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                return 'Permutation([], size=%s)' % self._print(expr.size)
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            use = full = self._print(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use
```
### 26 - sympy/combinatorics/partitions.py:

Start line: 231, End line: 264

```python
class Partition(FiniteSet):

    @property
    def RGS(self):
        """
        Returns the "restricted growth string" of the partition.

        Explanation
        ===========

        The RGS is returned as a list of indices, L, where L[i] indicates
        the block in which element i appears. For example, in a partition
        of 3 elements (a, b, c) into 2 blocks ([c], [a, b]) the RGS is
        [1, 1, 0]: "a" is in block 1, "b" is in block 1 and "c" is in block 0.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.members
        (1, 2, 3, 4, 5)
        >>> a.RGS
        (0, 0, 1, 2, 2)
        >>> a + 1
        Partition(FiniteSet(1, 2), FiniteSet(3), FiniteSet(4), FiniteSet(5))
        >>> _.RGS
        (0, 0, 1, 2, 3)
        """
        rgs = {}
        partition = self.partition
        for i, part in enumerate(partition):
            for j in part:
                rgs[j] = i
        return tuple([rgs[i] for i in sorted(
            [i for p in partition for i in p], key=default_sort_key)])
```
### 29 - sympy/combinatorics/partitions.py:

Start line: 305, End line: 334

```python
class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """

    _dict = None
    _keys = None
```
### 33 - sympy/combinatorics/partitions.py:

Start line: 485, End line: 501

```python
class IntegerPartition(Basic):

    def as_dict(self):
        """Return the partition as a dictionary whose keys are the
        partition integers and the values are the multiplicity of that
        integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()
        {1: 3, 2: 1, 3: 4}
        """
        if self._dict is None:
            groups = group(self.partition, multiple=False)
            self._keys = [g[0] for g in groups]
            self._dict = dict(groups)
        return self._dict
```
