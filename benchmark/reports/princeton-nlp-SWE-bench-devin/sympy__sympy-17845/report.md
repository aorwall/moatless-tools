# sympy__sympy-17845

| **sympy/sympy** | `dd53633d0f28ed8656480e25a49615258121cb5d` |
| ---- | ---- |
| **No of patches** | 15 |
| **All found context length** | - |
| **Any found context length** | 23591 |
| **Avg pos** | 4.733333333333333 |
| **Min pos** | 71 |
| **Max pos** | 71 |
| **Top file pos** | 1 |
| **Missing snippets** | 55 |
| **Missing patch files** | 12 |


## Expected patch

```diff
diff --git a/sympy/calculus/singularities.py b/sympy/calculus/singularities.py
--- a/sympy/calculus/singularities.py
+++ b/sympy/calculus/singularities.py
@@ -73,11 +73,11 @@ def singularities(expression, symbol):
     >>> singularities(x**2 + x + 1, x)
     EmptySet
     >>> singularities(1/(x + 1), x)
-    {-1}
+    FiniteSet(-1)
     >>> singularities(1/(y**2 + 1), y)
-    {-I, I}
+    FiniteSet(I, -I)
     >>> singularities(1/(y**3 + 1), y)
-    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
+    FiniteSet(-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2)
 
     """
     if not expression.is_rational_function(symbol):
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -744,7 +744,7 @@ def stationary_points(f, symbol, domain=S.Reals):
               2                                2
 
     >>> stationary_points(sin(x),x, Interval(0, 4*pi))
-    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}
+    FiniteSet(pi/2, 3*pi/2, 5*pi/2, 7*pi/2)
 
     """
     from sympy import solveset, diff
@@ -1550,7 +1550,7 @@ def intersection(self, other):
         EmptySet
 
         >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
-        {1, 2}
+        FiniteSet(1, 2)
 
         """
         if not isinstance(other, (AccumBounds, FiniteSet)):
diff --git a/sympy/categories/baseclasses.py b/sympy/categories/baseclasses.py
--- a/sympy/categories/baseclasses.py
+++ b/sympy/categories/baseclasses.py
@@ -482,7 +482,7 @@ def objects(self):
         >>> B = Object("B")
         >>> K = Category("K", FiniteSet(A, B))
         >>> K.objects
-        Class({Object("A"), Object("B")})
+        Class(FiniteSet(Object("A"), Object("B")))
 
         """
         return self.args[1]
@@ -677,7 +677,7 @@ def __new__(cls, *args):
         True
         >>> d = Diagram([f, g], {g * f: "unique"})
         >>> d.conclusions[g * f]
-        {unique}
+        FiniteSet(unique)
 
         """
         premises = {}
@@ -809,7 +809,7 @@ def objects(self):
         >>> g = NamedMorphism(B, C, "g")
         >>> d = Diagram([f, g])
         >>> d.objects
-        {Object("A"), Object("B"), Object("C")}
+        FiniteSet(Object("A"), Object("B"), Object("C"))
 
         """
         return self.args[2]
diff --git a/sympy/combinatorics/partitions.py b/sympy/combinatorics/partitions.py
--- a/sympy/combinatorics/partitions.py
+++ b/sympy/combinatorics/partitions.py
@@ -42,7 +42,7 @@ def __new__(cls, *partition):
         >>> from sympy.combinatorics.partitions import Partition
         >>> a = Partition([1, 2], [3])
         >>> a
-        {{3}, {1, 2}}
+        Partition(FiniteSet(1, 2), FiniteSet(3))
         >>> a.partition
         [[1, 2], [3]]
         >>> len(a)
@@ -53,7 +53,7 @@ def __new__(cls, *partition):
         Creating Partition from Python sets:
 
         >>> Partition({1, 2, 3}, {4, 5})
-        {{4, 5}, {1, 2, 3}}
+        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
 
         Creating Partition from SymPy finite sets:
 
@@ -61,7 +61,7 @@ def __new__(cls, *partition):
         >>> a = FiniteSet(1, 2, 3)
         >>> b = FiniteSet(4, 5)
         >>> Partition(a, b)
-        {{4, 5}, {1, 2, 3}}
+        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
         """
         args = []
         dups = False
@@ -107,7 +107,7 @@ def sort_key(self, order=None):
         >>> d = Partition(list(range(4)))
         >>> l = [d, b, a + 1, a, c]
         >>> l.sort(key=default_sort_key); l
-        [{{1, 2}}, {{1}, {2}}, {{1, x}}, {{3, 4}}, {{0, 1, 2, 3}}]
+        [Partition(FiniteSet(1, 2)), Partition(FiniteSet(1), FiniteSet(2)), Partition(FiniteSet(1, x)), Partition(FiniteSet(3, 4)), Partition(FiniteSet(0, 1, 2, 3))]
         """
         if order is None:
             members = self.members
@@ -250,7 +250,7 @@ def RGS(self):
         >>> a.RGS
         (0, 0, 1, 2, 2)
         >>> a + 1
-        {{3}, {4}, {5}, {1, 2}}
+        Partition(FiniteSet(1, 2), FiniteSet(3), FiniteSet(4), FiniteSet(5))
         >>> _.RGS
         (0, 0, 1, 2, 3)
         """
@@ -278,12 +278,12 @@ def from_rgs(self, rgs, elements):
 
         >>> from sympy.combinatorics.partitions import Partition
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
-        {{c}, {a, d}, {b, e}}
+        Partition(FiniteSet(c), FiniteSet(a, d), FiniteSet(b, e))
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
-        {{e}, {a, c}, {b, d}}
+        Partition(FiniteSet(e), FiniteSet(a, c), FiniteSet(b, d))
         >>> a = Partition([1, 4], [2], [3, 5])
         >>> Partition.from_rgs(a.RGS, a.members)
-        {{2}, {1, 4}, {3, 5}}
+        Partition(FiniteSet(1, 4), FiniteSet(2), FiniteSet(3, 5))
         """
         if len(rgs) != len(elements):
             raise ValueError('mismatch in rgs and element lengths')
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -47,9 +47,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
             >>> from sympy.combinatorics.polyhedron import Polyhedron
             >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
-            {(0, 1, 2)}
+            FiniteSet((0, 1, 2))
             >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
-            {(0, 1, 2)}
+            FiniteSet((0, 1, 2))
 
         The allowed transformations are entered as allowable permutations
         of the vertices for the polyhedron. Instance of Permutations
@@ -92,7 +92,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         >>> tetra.size
         4
         >>> tetra.edges
-        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
+        FiniteSet((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
         >>> tetra.corners
         (w, x, y, z)
 
@@ -365,7 +365,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
         >>> from sympy.combinatorics.polyhedron import cube
         >>> cube.edges
-        {(0, 1), (0, 3), (0, 4), '...', (4, 7), (5, 6), (6, 7)}
+        FiniteSet((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))
 
         If you want to use letters or other names for the corners you
         can still use the pre-calculated faces:
@@ -493,7 +493,7 @@ def edges(self):
         >>> corners = (a, b, c)
         >>> faces = [(0, 1, 2)]
         >>> Polyhedron(corners, faces).edges
-        {(0, 1), (0, 2), (1, 2)}
+        FiniteSet((0, 1), (0, 2), (1, 2))
 
         """
         if self._edges is None:
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -237,9 +237,9 @@ def nargs(self):
         corresponding set will be returned:
 
         >>> Function('f', nargs=1).nargs
-        {1}
+        FiniteSet(1)
         >>> Function('f', nargs=(2, 1)).nargs
-        {1, 2}
+        FiniteSet(1, 2)
 
         The undefined function, after application, also has the nargs
         attribute; the actual number of arguments is always available by
@@ -972,7 +972,7 @@ class WildFunction(Function, AtomicExpr):
 
     >>> F = WildFunction('F', nargs=2)
     >>> F.nargs
-    {2}
+    FiniteSet(2)
     >>> f(x).match(F)
     >>> f(x, y).match(F)
     {F_: f(x, y)}
@@ -983,7 +983,7 @@ class WildFunction(Function, AtomicExpr):
 
     >>> F = WildFunction('F', nargs=(1, 2))
     >>> F.nargs
-    {1, 2}
+    FiniteSet(1, 2)
     >>> f(x).match(F)
     {F_: f(x)}
     >>> f(x, y).match(F)
diff --git a/sympy/logic/boolalg.py b/sympy/logic/boolalg.py
--- a/sympy/logic/boolalg.py
+++ b/sympy/logic/boolalg.py
@@ -131,7 +131,7 @@ def as_set(self):
         >>> from sympy import Symbol, Eq, Or, And
         >>> x = Symbol('x', real=True)
         >>> Eq(x, 0).as_set()
-        {0}
+        FiniteSet(0)
         >>> (x > 0).as_set()
         Interval.open(0, oo)
         >>> And(-2 < x, x < 2).as_set()
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -149,14 +149,6 @@ def _print_Exp1(self, expr):
     def _print_ExprCondPair(self, expr):
         return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))
 
-    def _print_FiniteSet(self, s):
-        s = sorted(s, key=default_sort_key)
-        if len(s) > 10:
-            printset = s[:3] + ['...'] + s[-3:]
-        else:
-            printset = s
-        return '{' + ', '.join(self._print(el) for el in printset) + '}'
-
     def _print_Function(self, expr):
         return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")
 
diff --git a/sympy/sets/conditionset.py b/sympy/sets/conditionset.py
--- a/sympy/sets/conditionset.py
+++ b/sympy/sets/conditionset.py
@@ -65,20 +65,20 @@ class ConditionSet(Set):
 
     >>> c = ConditionSet(x, x < 1, {x, z})
     >>> c.subs(x, y)
-    ConditionSet(x, x < 1, {y, z})
+    ConditionSet(x, x < 1, FiniteSet(y, z))
 
     A second substitution is needed to change the dummy symbol, too:
 
     >>> _.subs(x, y)
-    ConditionSet(y, y < 1, {y, z})
+    ConditionSet(y, y < 1, FiniteSet(y, z))
 
     And trying to replace the dummy symbol with anything but a symbol
     is ignored: the only change possible will be in the base set:
 
     >>> ConditionSet(y, y < 1, {y, z}).subs(y, 1)
-    ConditionSet(y, y < 1, {z})
+    ConditionSet(y, y < 1, FiniteSet(z))
     >>> _.subs(y, 1)
-    ConditionSet(y, y < 1, {z})
+    ConditionSet(y, y < 1, FiniteSet(z))
 
     Notes
     =====
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -278,7 +278,7 @@ class ImageSet(Set):
     False
 
     >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
-    {1, 4, 9}
+    FiniteSet(1, 4, 9)
 
     >>> square_iterable = iter(squares)
     >>> for i in range(4):
@@ -300,7 +300,7 @@ class ImageSet(Set):
     >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
     >>> dom = Interval(-1, 1)
     >>> dom.intersect(solutions)
-    {0}
+    FiniteSet(0)
 
     See Also
     ========
@@ -921,7 +921,7 @@ def normalize_theta_set(theta):
     >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
     Interval(pi/2, 3*pi/2)
     >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
-    {0, pi}
+    FiniteSet(0, pi)
 
     """
     from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
@@ -1200,7 +1200,7 @@ def from_real(cls, sets):
         >>> from sympy import Interval, ComplexRegion
         >>> unit = Interval(0,1)
         >>> ComplexRegion.from_real(unit)
-        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))
+        CartesianComplexRegion(ProductSet(Interval(0, 1), FiniteSet(0)))
 
         """
         if not sets.is_subset(S.Reals):
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -43,7 +43,7 @@ class PowerSet(Set):
     A power set of a finite set:
 
     >>> PowerSet(FiniteSet(1, 2, 3))
-    PowerSet({1, 2, 3})
+    PowerSet(FiniteSet(1, 2, 3))
 
     A power set of an empty set:
 
@@ -60,7 +60,9 @@ class PowerSet(Set):
     Evaluating the power set of a finite set to its explicit form:
 
     >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
-    {EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}
+    FiniteSet(FiniteSet(1), FiniteSet(1, 2), FiniteSet(1, 3),
+            FiniteSet(1, 2, 3), FiniteSet(2), FiniteSet(2, 3),
+            FiniteSet(3), EmptySet)
 
     References
     ==========
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -109,7 +109,7 @@ def union(self, other):
         >>> Interval(0, 1) + Interval(2, 3)
         Union(Interval(0, 1), Interval(2, 3))
         >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
-        Union({3}, Interval.Lopen(1, 2))
+        Union(FiniteSet(3), Interval.Lopen(1, 2))
 
         Similarly it is possible to use the '-' operator for set differences:
 
@@ -469,7 +469,7 @@ def powerset(self):
         >>> from sympy import FiniteSet, EmptySet
         >>> A = EmptySet
         >>> A.powerset()
-        {EmptySet}
+        FiniteSet(EmptySet)
 
         A power set of a finite set:
 
@@ -532,9 +532,9 @@ def boundary(self):
 
         >>> from sympy import Interval
         >>> Interval(0, 1).boundary
-        {0, 1}
+        FiniteSet(0, 1)
         >>> Interval(0, 1, True, False).boundary
-        {0, 1}
+        FiniteSet(0, 1)
         """
         return self._boundary
 
@@ -659,7 +659,7 @@ class ProductSet(Set):
     >>> from sympy import Interval, FiniteSet, ProductSet
     >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
     >>> ProductSet(I, S)
-    ProductSet(Interval(0, 5), {1, 2, 3})
+    ProductSet(Interval(0, 5), FiniteSet(1, 2, 3))
 
     >>> (2, 2) in ProductSet(I, S)
     True
@@ -1492,7 +1492,7 @@ class Complement(Set, EvalfMixin):
 
     >>> from sympy import Complement, FiniteSet
     >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
-    {0, 2}
+    FiniteSet(0, 2)
 
     See Also
     =========
@@ -1683,18 +1683,18 @@ class FiniteSet(Set, EvalfMixin):
 
     >>> from sympy import FiniteSet
     >>> FiniteSet(1, 2, 3, 4)
-    {1, 2, 3, 4}
+    FiniteSet(1, 2, 3, 4)
     >>> 3 in FiniteSet(1, 2, 3, 4)
     True
 
     >>> members = [1, 2, 3, 4]
     >>> f = FiniteSet(*members)
     >>> f
-    {1, 2, 3, 4}
+    FiniteSet(1, 2, 3, 4)
     >>> f - FiniteSet(2)
-    {1, 3, 4}
+    FiniteSet(1, 3, 4)
     >>> f + FiniteSet(2, 5)
-    {1, 2, 3, 4, 5}
+    FiniteSet(1, 2, 3, 4, 5)
 
     References
     ==========
@@ -1893,7 +1893,7 @@ class SymmetricDifference(Set):
 
     >>> from sympy import SymmetricDifference, FiniteSet
     >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
-    {1, 2, 4, 5}
+    FiniteSet(1, 2, 4, 5)
 
     See Also
     ========
diff --git a/sympy/solvers/inequalities.py b/sympy/solvers/inequalities.py
--- a/sympy/solvers/inequalities.py
+++ b/sympy/solvers/inequalities.py
@@ -29,13 +29,13 @@ def solve_poly_inequality(poly, rel):
     >>> from sympy.solvers.inequalities import solve_poly_inequality
 
     >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
-    [{0}]
+    [FiniteSet(0)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
     [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
-    [{-1}, {1}]
+    [FiniteSet(-1), FiniteSet(1)]
 
     See Also
     ========
@@ -141,7 +141,7 @@ def solve_rational_inequalities(eqs):
     >>> solve_rational_inequalities([[
     ... ((Poly(-x + 1), Poly(1, x)), '>='),
     ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
-    {1}
+    FiniteSet(1)
 
     >>> solve_rational_inequalities([[
     ... ((Poly(x), Poly(1, x)), '!='),
diff --git a/sympy/solvers/solveset.py b/sympy/solvers/solveset.py
--- a/sympy/solvers/solveset.py
+++ b/sympy/solvers/solveset.py
@@ -136,14 +136,14 @@ def _invert(f_x, y, x, domain=S.Complexes):
     >>> invert_complex(exp(x), y, x)
     (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
     >>> invert_real(exp(x), y, x)
-    (x, Intersection({log(y)}, Reals))
+    (x, Intersection(FiniteSet(log(y)), Reals))
 
     When does exp(x) == 1?
 
     >>> invert_complex(exp(x), 1, x)
     (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
     >>> invert_real(exp(x), 1, x)
-    (x, {0})
+    (x, FiniteSet(0))
 
     See Also
     ========
@@ -805,7 +805,7 @@ def solve_decomposition(f, symbol, domain):
     >>> x = Symbol('x')
     >>> f1 = exp(2*x) - 3*exp(x) + 2
     >>> sd(f1, x, S.Reals)
-    {0, log(2)}
+    FiniteSet(0, log(2))
     >>> f2 = sin(x)**2 + 2*sin(x) + 1
     >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
               3*pi
@@ -1365,11 +1365,11 @@ def _solve_exponential(lhs, rhs, symbol, domain):
     >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
     ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
     >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
-    ConditionSet(x, (a > 0) & (b > 0), {0})
+    ConditionSet(x, (a > 0) & (b > 0), FiniteSet(0))
     >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
-    {-3*log(2)/(-2*log(3) + log(2))}
+    FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
     >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
-    {0}
+    FiniteSet(0)
 
     * Proof of correctness of the method
 
@@ -1525,7 +1525,7 @@ def _solve_logarithm(lhs, rhs, symbol, domain):
     >>> x = symbols('x')
     >>> f = log(x - 3) + log(x + 3)
     >>> solve_log(f, 0, x, S.Reals)
-    {-sqrt(10), sqrt(10)}
+    FiniteSet(sqrt(10), -sqrt(10))
 
     * Proof of correctness
 
@@ -1679,7 +1679,7 @@ def _transolve(f, symbol, domain):
     >>> from sympy import symbols, S, pprint
     >>> x = symbols('x', real=True) # assumption added
     >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
-    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}
+    FiniteSet(-(log(3) + 3*log(5))/(-log(5) + 2*log(3)))
 
     How ``_transolve`` works
     ========================
@@ -1921,9 +1921,9 @@ def solveset(f, symbol=None, domain=S.Complexes):
     >>> R = S.Reals
     >>> x = Symbol('x')
     >>> solveset(exp(x) - 1, x, R)
-    {0}
+    FiniteSet(0)
     >>> solveset_real(exp(x) - 1, x)
-    {0}
+    FiniteSet(0)
 
     The solution is mostly unaffected by assumptions on the symbol,
     but there may be some slight difference:
@@ -2423,7 +2423,7 @@ def linsolve(system, *symbols):
     [6],
     [9]])
     >>> linsolve((A, b), [x, y, z])
-    {(-1, 2, 0)}
+    FiniteSet((-1, 2, 0))
 
     * Parametric Solution: In case the system is underdetermined, the
       function will return a parametric solution in terms of the given
@@ -2434,20 +2434,20 @@ def linsolve(system, *symbols):
     >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     >>> b = Matrix([3, 6, 9])
     >>> linsolve((A, b), x, y, z)
-    {(z - 1, 2 - 2*z, z)}
+    FiniteSet((z - 1, 2 - 2*z, z))
 
     If no symbols are given, internally generated symbols will be used.
     The `tau0` in the 3rd position indicates (as before) that the 3rd
     variable -- whatever it's named -- can take on any value:
 
     >>> linsolve((A, b))
-    {(tau0 - 1, 2 - 2*tau0, tau0)}
+    FiniteSet((tau0 - 1, 2 - 2*tau0, tau0))
 
     * List of Equations as input
 
     >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
     >>> linsolve(Eqns, x, y, z)
-    {(1, -2, -2)}
+    FiniteSet((1, -2, -2))
 
     * Augmented Matrix as input
 
@@ -2458,21 +2458,21 @@ def linsolve(system, *symbols):
     [2, 6,  8, 3],
     [6, 8, 18, 5]])
     >>> linsolve(aug, x, y, z)
-    {(3/10, 2/5, 0)}
+    FiniteSet((3/10, 2/5, 0))
 
     * Solve for symbolic coefficients
 
     >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
     >>> eqns = [a*x + b*y - c, d*x + e*y - f]
     >>> linsolve(eqns, x, y)
-    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}
+    FiniteSet(((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d)))
 
     * A degenerate system returns solution as set of given
       symbols.
 
     >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
     >>> linsolve(system, x, y)
-    {(x, y)}
+    FiniteSet((x, y))
 
     * For an empty system linsolve returns empty set
 
@@ -2483,7 +2483,7 @@ def linsolve(system, *symbols):
       is detected:
 
     >>> linsolve([x*(1/x - 1), (y - 1)**2 - y**2 + 1], x, y)
-    {(1, 1)}
+    FiniteSet((1, 1))
     >>> linsolve([x**2 - 1], x)
     Traceback (most recent call last):
     ...
@@ -2647,33 +2647,33 @@ def substitution(system, symbols, result=[{}], known_symbols=[],
     >>> x, y = symbols('x, y', real=True)
     >>> from sympy.solvers.solveset import substitution
     >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
-    {(-1, 1)}
+    FiniteSet((-1, 1))
 
     * when you want soln should not satisfy eq `x + 1 = 0`
 
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
     EmptySet
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
-    {(1, -1)}
+    FiniteSet((1, -1))
     >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
-    {(-3, 4), (2, -1)}
+    FiniteSet((-3, 4), (2, -1))
 
     * Returns both real and complex solution
 
     >>> x, y, z = symbols('x, y, z')
     >>> from sympy import exp, sin
     >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
-    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
-    (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
+    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
+            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
 
     >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
     >>> substitution(eqs, [y, z])
-    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
-    (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+    FiniteSet((-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+    (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-    ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
+       ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-    ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}
+       ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)))
 
     """
 
@@ -3254,7 +3254,7 @@ def nonlinsolve(system, *symbols):
     >>> from sympy.solvers.solveset import nonlinsolve
     >>> x, y, z = symbols('x, y, z', real=True)
     >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
-    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}
+    FiniteSet((-1, -1), (-1/2, -2), (1/2, 2), (1, 1))
 
     1. Positive dimensional system and complements:
 
@@ -3273,7 +3273,7 @@ def nonlinsolve(system, *symbols):
     {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
        d       d               d       d
     >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
-    {(2 - y, y)}
+    FiniteSet((2 - y, y))
 
     2. If some of the equations are non-polynomial then `nonlinsolve`
     will call the `substitution` function and return real and complex solutions,
@@ -3281,8 +3281,9 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import exp, sin
     >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
-    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
-    (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
+    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
+            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
+
 
     3. If system is non-linear polynomial and zero-dimensional then it
     returns both solution (real and complex solutions, if present) using
@@ -3290,7 +3291,7 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import sqrt
     >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
-    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
+    FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
 
     4. `nonlinsolve` can solve some linear (zero or positive dimensional)
     system (because it uses the `groebner` function to get the
@@ -3299,7 +3300,7 @@ def nonlinsolve(system, *symbols):
     `nonlinsolve`, because `linsolve` is better for general linear systems.
 
     >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9 , y + z - 4], [x, y, z])
-    {(3*z - 5, 4 - z, z)}
+    FiniteSet((3*z - 5, 4 - z, z))
 
     5. System having polynomial equations and only real solution is
     solved using `solve_poly_system`:
@@ -3307,11 +3308,11 @@ def nonlinsolve(system, *symbols):
     >>> e1 = sqrt(x**2 + y**2) - 10
     >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
     >>> nonlinsolve((e1, e2), (x, y))
-    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
+    FiniteSet((191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20))
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
-    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
+    FiniteSet((1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5)))
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
-    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}
+    FiniteSet((2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5)))
 
     6. It is better to use symbols instead of Trigonometric Function or
     Function (e.g. replace `sin(x)` with symbol, replace `f(x)` with symbol
diff --git a/sympy/stats/stochastic_process_types.py b/sympy/stats/stochastic_process_types.py
--- a/sympy/stats/stochastic_process_types.py
+++ b/sympy/stats/stochastic_process_types.py
@@ -553,7 +553,7 @@ class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
     >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
     >>> YS = DiscreteMarkovChain("Y")
     >>> Y.state_space
-    {0, 1, 2}
+    FiniteSet(0, 1, 2)
     >>> Y.transition_probabilities
     Matrix([
     [0.5, 0.2, 0.3],

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/calculus/singularities.py | 76 | 80 | - | - | -
| sympy/calculus/util.py | 747 | 747 | - | - | -
| sympy/calculus/util.py | 1553 | 1553 | - | - | -
| sympy/categories/baseclasses.py | 485 | 485 | - | - | -
| sympy/categories/baseclasses.py | 680 | 680 | - | - | -
| sympy/categories/baseclasses.py | 812 | 812 | - | - | -
| sympy/combinatorics/partitions.py | 45 | 45 | - | - | -
| sympy/combinatorics/partitions.py | 56 | 56 | - | - | -
| sympy/combinatorics/partitions.py | 64 | 64 | - | - | -
| sympy/combinatorics/partitions.py | 110 | 110 | - | - | -
| sympy/combinatorics/partitions.py | 253 | 253 | - | - | -
| sympy/combinatorics/partitions.py | 281 | 286 | - | - | -
| sympy/combinatorics/polyhedron.py | 50 | 52 | - | - | -
| sympy/combinatorics/polyhedron.py | 95 | 95 | - | - | -
| sympy/combinatorics/polyhedron.py | 368 | 368 | - | - | -
| sympy/combinatorics/polyhedron.py | 496 | 496 | - | - | -
| sympy/core/function.py | 240 | 242 | - | - | -
| sympy/core/function.py | 975 | 975 | - | - | -
| sympy/core/function.py | 986 | 986 | - | - | -
| sympy/logic/boolalg.py | 134 | 134 | - | - | -
| sympy/printing/str.py | 152 | 159 | - | 1 | -
| sympy/sets/conditionset.py | 68 | 81 | - | - | -
| sympy/sets/fancysets.py | 281 | 281 | - | 19 | -
| sympy/sets/fancysets.py | 303 | 303 | - | 19 | -
| sympy/sets/fancysets.py | 924 | 924 | - | 19 | -
| sympy/sets/fancysets.py | 1203 | 1203 | - | 19 | -
| sympy/sets/powerset.py | 46 | 46 | - | - | -
| sympy/sets/powerset.py | 63 | 63 | - | - | -
| sympy/sets/sets.py | 112 | 112 | - | 2 | -
| sympy/sets/sets.py | 472 | 472 | - | 2 | -
| sympy/sets/sets.py | 535 | 537 | - | 2 | -
| sympy/sets/sets.py | 662 | 662 | - | 2 | -
| sympy/sets/sets.py | 1495 | 1495 | - | 2 | -
| sympy/sets/sets.py | 1686 | 1697 | 71 | 2 | 23591
| sympy/sets/sets.py | 1896 | 1896 | - | 2 | -
| sympy/solvers/inequalities.py | 32 | 38 | - | - | -
| sympy/solvers/inequalities.py | 144 | 144 | - | - | -
| sympy/solvers/solveset.py | 139 | 146 | - | - | -
| sympy/solvers/solveset.py | 808 | 808 | - | - | -
| sympy/solvers/solveset.py | 1368 | 1372 | - | - | -
| sympy/solvers/solveset.py | 1528 | 1528 | - | - | -
| sympy/solvers/solveset.py | 1682 | 1682 | - | - | -
| sympy/solvers/solveset.py | 1924 | 1926 | - | - | -
| sympy/solvers/solveset.py | 2426 | 2426 | - | - | -
| sympy/solvers/solveset.py | 2437 | 2450 | - | - | -
| sympy/solvers/solveset.py | 2461 | 2475 | - | - | -
| sympy/solvers/solveset.py | 2486 | 2486 | - | - | -
| sympy/solvers/solveset.py | 2650 | 2676 | - | - | -
| sympy/solvers/solveset.py | 3257 | 3257 | - | - | -
| sympy/solvers/solveset.py | 3276 | 3276 | - | - | -
| sympy/solvers/solveset.py | 3284 | 3285 | - | - | -
| sympy/solvers/solveset.py | 3293 | 3293 | - | - | -
| sympy/solvers/solveset.py | 3302 | 3302 | - | - | -
| sympy/solvers/solveset.py | 3310 | 3314 | - | - | -
| sympy/stats/stochastic_process_types.py | 556 | 556 | - | - | -


## Problem Statement

```
Interval and FiniteSet printing
Currently 
str(Interval(0,1)) produces "[0, 1]" 
and 
str(FiniteSet(1,2,3)) produces "{1, 2, 3}"

This violates the str(object) is valid code to create object principle. 

If we change this then code for Interval looks quite ugly. We will end up printing things like "Interval(0, 1, True, False)" to the screen.

Original issue for #6265: http://code.google.com/p/sympy/issues/detail?id=3166
Original author: https://code.google.com/u/109882876523836932473/


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/str.py** | 188 | 205| 153 | 153 | 7367 | 
| 2 | **2 sympy/sets/sets.py** | 871 | 910| 372 | 525 | 23264 | 
| 3 | 3 sympy/plotting/intervalmath/interval_arithmetic.py | 281 | 320| 341 | 866 | 26460 | 
| 4 | **3 sympy/printing/str.py** | 70 | 158| 805 | 1671 | 26460 | 
| 5 | 3 sympy/plotting/intervalmath/interval_arithmetic.py | 146 | 163| 153 | 1824 | 26460 | 
| 6 | 3 sympy/plotting/intervalmath/interval_arithmetic.py | 94 | 124| 226 | 2050 | 26460 | 
| 7 | 4 sympy/printing/pretty/pretty.py | 1906 | 1928| 193 | 2243 | 49657 | 
| 8 | **4 sympy/printing/str.py** | 727 | 810| 678 | 2921 | 49657 | 
| 9 | 4 sympy/plotting/intervalmath/interval_arithmetic.py | 165 | 182| 137 | 3058 | 49657 | 
| 10 | **4 sympy/sets/sets.py** | 1027 | 1040| 122 | 3180 | 49657 | 
| 11 | 4 sympy/plotting/intervalmath/interval_arithmetic.py | 126 | 144| 156 | 3336 | 49657 | 
| 12 | **4 sympy/sets/sets.py** | 1042 | 1097| 423 | 3759 | 49657 | 
| 13 | **4 sympy/printing/str.py** | 207 | 249| 448 | 4207 | 49657 | 
| 14 | **4 sympy/printing/str.py** | 583 | 656| 543 | 4750 | 49657 | 
| 15 | 4 sympy/plotting/intervalmath/interval_arithmetic.py | 184 | 210| 212 | 4962 | 49657 | 
| 16 | 4 sympy/printing/pretty/pretty.py | 1930 | 1983| 388 | 5350 | 49657 | 
| 17 | 4 sympy/plotting/intervalmath/interval_arithmetic.py | 244 | 279| 292 | 5642 | 49657 | 
| 18 | 4 sympy/plotting/intervalmath/interval_arithmetic.py | 212 | 242| 234 | 5876 | 49657 | 
| 19 | **4 sympy/sets/sets.py** | 827 | 869| 309 | 6185 | 49657 | 
| 20 | **4 sympy/printing/str.py** | 446 | 519| 529 | 6714 | 49657 | 
| 21 | 5 sympy/polys/numberfields.py | 1071 | 1084| 147 | 6861 | 58823 | 
| 22 | **5 sympy/printing/str.py** | 363 | 386| 266 | 7127 | 58823 | 
| 23 | 5 sympy/plotting/intervalmath/interval_arithmetic.py | 343 | 369| 210 | 7337 | 58823 | 
| 24 | **5 sympy/printing/str.py** | 388 | 444| 541 | 7878 | 58823 | 
| 25 | **5 sympy/printing/str.py** | 160 | 186| 237 | 8115 | 58823 | 
| 26 | 5 sympy/plotting/intervalmath/interval_arithmetic.py | 322 | 341| 147 | 8262 | 58823 | 
| 27 | 5 sympy/plotting/intervalmath/interval_arithmetic.py | 42 | 92| 424 | 8686 | 58823 | 
| 28 | **5 sympy/printing/str.py** | 658 | 678| 189 | 8875 | 58823 | 
| 29 | **5 sympy/printing/str.py** | 828 | 859| 287 | 9162 | 58823 | 
| 30 | 6 sympy/plotting/pygletplot/plot_interval.py | 1 | 111| 838 | 10000 | 60150 | 
| 31 | **6 sympy/printing/str.py** | 18 | 44| 189 | 10189 | 60150 | 
| 32 | 6 sympy/plotting/pygletplot/plot_interval.py | 113 | 165| 355 | 10544 | 60150 | 
| 33 | **6 sympy/sets/sets.py** | 1744 | 1781| 337 | 10881 | 60150 | 
| 34 | **6 sympy/printing/str.py** | 324 | 361| 285 | 11166 | 60150 | 
| 35 | **6 sympy/sets/sets.py** | 1722 | 1742| 181 | 11347 | 60150 | 
| 36 | 6 sympy/printing/pretty/pretty.py | 2001 | 2027| 241 | 11588 | 60150 | 
| 37 | 7 sympy/printing/codeprinter.py | 502 | 537| 400 | 11988 | 64667 | 
| 38 | 7 sympy/printing/codeprinter.py | 131 | 208| 718 | 12706 | 64667 | 
| 39 | 8 sympy/printing/julia.py | 292 | 329| 215 | 12921 | 70566 | 
| 40 | 9 sympy/plotting/intervalmath/__init__.py | 1 | 5| 0 | 12921 | 70639 | 
| 41 | 9 sympy/printing/pretty/pretty.py | 2194 | 2302| 799 | 13720 | 70639 | 
| 42 | **9 sympy/printing/str.py** | 812 | 826| 132 | 13852 | 70639 | 
| 43 | 10 sympy/printing/mathml.py | 1215 | 1237| 182 | 14034 | 87240 | 
| 44 | 11 sympy/plotting/intervalmath/interval_membership.py | 1 | 79| 561 | 14595 | 87802 | 
| 45 | 12 sympy/printing/octave.py | 294 | 325| 175 | 14770 | 94544 | 
| 46 | 13 sympy/plotting/pygletplot/plot_mode.py | 325 | 344| 207 | 14977 | 97673 | 
| 47 | 13 sympy/plotting/intervalmath/interval_arithmetic.py | 1 | 39| 335 | 15312 | 97673 | 
| 48 | 14 sympy/printing/latex.py | 1886 | 1901| 147 | 15459 | 125193 | 
| 49 | 14 sympy/printing/latex.py | 1986 | 2055| 610 | 16069 | 125193 | 
| 50 | **14 sympy/sets/sets.py** | 1862 | 1884| 206 | 16275 | 125193 | 
| 51 | **14 sympy/printing/str.py** | 680 | 699| 185 | 16460 | 125193 | 
| 52 | 14 sympy/printing/codeprinter.py | 398 | 449| 507 | 16967 | 125193 | 
| 53 | 15 sympy/printing/pycode.py | 226 | 240| 127 | 17094 | 133426 | 
| 54 | **15 sympy/printing/str.py** | 46 | 68| 160 | 17254 | 133426 | 
| 55 | 16 sympy/printing/fcode.py | 322 | 336| 131 | 17385 | 141416 | 
| 56 | 16 sympy/printing/fcode.py | 486 | 506| 176 | 17561 | 141416 | 
| 57 | 17 sympy/printing/repr.py | 47 | 107| 447 | 18008 | 143747 | 
| 58 | **17 sympy/printing/str.py** | 251 | 266| 146 | 18154 | 143747 | 
| 59 | 17 sympy/printing/fcode.py | 417 | 433| 187 | 18341 | 143747 | 
| 60 | **17 sympy/printing/str.py** | 268 | 322| 501 | 18842 | 143747 | 
| 61 | 18 sympy/printing/pretty/pretty_symbology.py | 260 | 300| 603 | 19445 | 149676 | 
| 62 | **19 sympy/sets/fancysets.py** | 716 | 838| 1041 | 20486 | 160119 | 
| 63 | 19 sympy/printing/pretty/pretty.py | 1 | 28| 267 | 20753 | 160119 | 
| 64 | 20 sympy/plotting/intervalmath/lib_interval.py | 414 | 433| 186 | 20939 | 163770 | 
| 65 | 21 sympy/printing/defaults.py | 1 | 21| 157 | 21096 | 163927 | 
| 66 | **21 sympy/sets/fancysets.py** | 479 | 562| 674 | 21770 | 163927 | 
| 67 | 21 sympy/printing/fcode.py | 655 | 666| 124 | 21894 | 163927 | 
| 68 | **21 sympy/sets/sets.py** | 912 | 1025| 654 | 22548 | 163927 | 
| 69 | 21 sympy/printing/julia.py | 358 | 374| 181 | 22729 | 163927 | 
| 70 | 21 sympy/printing/pycode.py | 281 | 340| 539 | 23268 | 163927 | 
| **-> 71 <-** | **21 sympy/sets/sets.py** | 1677 | 1720| 323 | 23591 | 163927 | 
| 72 | 22 sympy/printing/ccode.py | 303 | 338| 288 | 23879 | 172113 | 
| 73 | 23 sympy/sets/handlers/functions.py | 25 | 112| 754 | 24633 | 174095 | 
| 74 | 23 sympy/printing/pretty/pretty.py | 2029 | 2063| 311 | 24944 | 174095 | 
| 75 | **23 sympy/sets/fancysets.py** | 1 | 20| 186 | 25130 | 174095 | 
| 76 | 23 sympy/printing/julia.py | 398 | 433| 323 | 25453 | 174095 | 
| 77 | 23 sympy/printing/pretty/pretty_symbology.py | 302 | 331| 230 | 25683 | 174095 | 
| 78 | 23 sympy/printing/octave.py | 392 | 479| 780 | 26463 | 174095 | 
| 79 | 23 sympy/printing/pretty/pretty.py | 1985 | 1999| 174 | 26637 | 174095 | 
| 80 | 23 sympy/printing/pretty/pretty.py | 390 | 464| 607 | 27244 | 174095 | 
| 81 | 24 sympy/printing/rust.py | 337 | 405| 527 | 27771 | 179570 | 
| 82 | 24 sympy/printing/fcode.py | 757 | 773| 212 | 27983 | 179570 | 


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
 * 15: sympy/stats/stochastic_process_types.py

### Hint

```
But may be I mistaken about of this printing policy. It is possible that this policy (as I described above) is outdated.

But I note, that only the `repr` must return valid code.
For `str` ( which prints for the user reading) it is not obligatory.

At least it is written in the docstrings of modules, as I understand.

**Labels:** Printing  

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c1
Original author: https://code.google.com/u/109448925098397033296/

Another idea, for the classes which can take Intervals as arguments, it is possible to use the short construction string.
\`\`\`
In [3]: x = Symbol('x', real=True)

In [4]: Intersection(Interval(1, 3), Interval(x, 6))
Out[4]: [1, 3] ∩ [x, 6]

In [5]: str(Intersection(Interval(1, 3), Interval(x, 6)))
Out[5]: Intersection([1, 3], [x, 6])

The Out[5] can be valid not only as:
Out[5]: Intersection(Interval(1, 3), Interval(x, 6))
\`\`\`
but and as
\`\`\`
Out[5]: Intersection((1, 3), (x, 6))
\`\`\`
if Intersection constructor can accept tuple and understand that it is Interval and parse correctly.

This case is only of the ends are not open. (Or for open? As it will be confused and strange that (1, 3) --> [1, 3] for `pprint`)

Unfortunately it is not possiblely to use `Intersection([1, 3], [x, 6])`, because 
arguments must be immutable.

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c2
Original author: https://code.google.com/u/109448925098397033296/

I think it is better not to connect Intersection and Interval too strongly.

Intersection will be used for many different kinds of classes, not just Interval. (1,2) could equally well refer to Interval(1,2) or FiniteSet(1,2). 

I think that creating the special syntax will create problems in the future. 

If, as you say in your first comment, it is not important for str(object) to be valid code to produce object then I think that this issue is not important.

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c3
Original author: https://code.google.com/u/109882876523836932473/

\`\`\`
**Status:** Valid  

\`\`\`

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c4
Original author: https://code.google.com/u/asmeurer@gmail.com/

\`\`\`
We have moved issues to GitHub https://github.com/sympy/sympy/issues .

**Labels:** Restrict-AddIssueComment-Commit  

\`\`\`

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c5
Original author: https://code.google.com/u/asmeurer@gmail.com/

\`\`\`
We have moved issues to GitHub https://github.com/sympy/sympy/issues .
\`\`\`

Original comment: http://code.google.com/p/sympy/issues/detail?id=3166#c6
Original author: https://code.google.com/u/asmeurer@gmail.com/

Is this issue still worth fixing? The printing of both FiniteSet and Interval has been unchanged for some time now. It would seem gratuitous to fix at at this point...
\`\`\`python
>>> S(str(Interval(0,1)) )
Interval(0, 1)
>>> type(S(str(FiniteSet(0,1)) ))
<class 'set'>
\`\`\`

Perhaps `set`, like `dict`, is just not handled yet by `sympify`.
Sympify will convert a `set` to a `FiniteSet` but doesn't recognise a set literal and passes that on to eval.
For the original issue this has already been changed for Interval but not FiniteSet:
\`\`\`julia
In [1]: str(Interval(0, 1))                                                                                                                                   
Out[1]: 'Interval(0, 1)'

In [2]: str(FiniteSet(1))                                                                                                                                     
Out[2]: '{1}'
\`\`\`
Changing this for FiniteSet does not seem problematic. PR underway...
```

## Patch

```diff
diff --git a/sympy/calculus/singularities.py b/sympy/calculus/singularities.py
--- a/sympy/calculus/singularities.py
+++ b/sympy/calculus/singularities.py
@@ -73,11 +73,11 @@ def singularities(expression, symbol):
     >>> singularities(x**2 + x + 1, x)
     EmptySet
     >>> singularities(1/(x + 1), x)
-    {-1}
+    FiniteSet(-1)
     >>> singularities(1/(y**2 + 1), y)
-    {-I, I}
+    FiniteSet(I, -I)
     >>> singularities(1/(y**3 + 1), y)
-    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
+    FiniteSet(-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2)
 
     """
     if not expression.is_rational_function(symbol):
diff --git a/sympy/calculus/util.py b/sympy/calculus/util.py
--- a/sympy/calculus/util.py
+++ b/sympy/calculus/util.py
@@ -744,7 +744,7 @@ def stationary_points(f, symbol, domain=S.Reals):
               2                                2
 
     >>> stationary_points(sin(x),x, Interval(0, 4*pi))
-    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}
+    FiniteSet(pi/2, 3*pi/2, 5*pi/2, 7*pi/2)
 
     """
     from sympy import solveset, diff
@@ -1550,7 +1550,7 @@ def intersection(self, other):
         EmptySet
 
         >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
-        {1, 2}
+        FiniteSet(1, 2)
 
         """
         if not isinstance(other, (AccumBounds, FiniteSet)):
diff --git a/sympy/categories/baseclasses.py b/sympy/categories/baseclasses.py
--- a/sympy/categories/baseclasses.py
+++ b/sympy/categories/baseclasses.py
@@ -482,7 +482,7 @@ def objects(self):
         >>> B = Object("B")
         >>> K = Category("K", FiniteSet(A, B))
         >>> K.objects
-        Class({Object("A"), Object("B")})
+        Class(FiniteSet(Object("A"), Object("B")))
 
         """
         return self.args[1]
@@ -677,7 +677,7 @@ def __new__(cls, *args):
         True
         >>> d = Diagram([f, g], {g * f: "unique"})
         >>> d.conclusions[g * f]
-        {unique}
+        FiniteSet(unique)
 
         """
         premises = {}
@@ -809,7 +809,7 @@ def objects(self):
         >>> g = NamedMorphism(B, C, "g")
         >>> d = Diagram([f, g])
         >>> d.objects
-        {Object("A"), Object("B"), Object("C")}
+        FiniteSet(Object("A"), Object("B"), Object("C"))
 
         """
         return self.args[2]
diff --git a/sympy/combinatorics/partitions.py b/sympy/combinatorics/partitions.py
--- a/sympy/combinatorics/partitions.py
+++ b/sympy/combinatorics/partitions.py
@@ -42,7 +42,7 @@ def __new__(cls, *partition):
         >>> from sympy.combinatorics.partitions import Partition
         >>> a = Partition([1, 2], [3])
         >>> a
-        {{3}, {1, 2}}
+        Partition(FiniteSet(1, 2), FiniteSet(3))
         >>> a.partition
         [[1, 2], [3]]
         >>> len(a)
@@ -53,7 +53,7 @@ def __new__(cls, *partition):
         Creating Partition from Python sets:
 
         >>> Partition({1, 2, 3}, {4, 5})
-        {{4, 5}, {1, 2, 3}}
+        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
 
         Creating Partition from SymPy finite sets:
 
@@ -61,7 +61,7 @@ def __new__(cls, *partition):
         >>> a = FiniteSet(1, 2, 3)
         >>> b = FiniteSet(4, 5)
         >>> Partition(a, b)
-        {{4, 5}, {1, 2, 3}}
+        Partition(FiniteSet(1, 2, 3), FiniteSet(4, 5))
         """
         args = []
         dups = False
@@ -107,7 +107,7 @@ def sort_key(self, order=None):
         >>> d = Partition(list(range(4)))
         >>> l = [d, b, a + 1, a, c]
         >>> l.sort(key=default_sort_key); l
-        [{{1, 2}}, {{1}, {2}}, {{1, x}}, {{3, 4}}, {{0, 1, 2, 3}}]
+        [Partition(FiniteSet(1, 2)), Partition(FiniteSet(1), FiniteSet(2)), Partition(FiniteSet(1, x)), Partition(FiniteSet(3, 4)), Partition(FiniteSet(0, 1, 2, 3))]
         """
         if order is None:
             members = self.members
@@ -250,7 +250,7 @@ def RGS(self):
         >>> a.RGS
         (0, 0, 1, 2, 2)
         >>> a + 1
-        {{3}, {4}, {5}, {1, 2}}
+        Partition(FiniteSet(1, 2), FiniteSet(3), FiniteSet(4), FiniteSet(5))
         >>> _.RGS
         (0, 0, 1, 2, 3)
         """
@@ -278,12 +278,12 @@ def from_rgs(self, rgs, elements):
 
         >>> from sympy.combinatorics.partitions import Partition
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
-        {{c}, {a, d}, {b, e}}
+        Partition(FiniteSet(c), FiniteSet(a, d), FiniteSet(b, e))
         >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
-        {{e}, {a, c}, {b, d}}
+        Partition(FiniteSet(e), FiniteSet(a, c), FiniteSet(b, d))
         >>> a = Partition([1, 4], [2], [3, 5])
         >>> Partition.from_rgs(a.RGS, a.members)
-        {{2}, {1, 4}, {3, 5}}
+        Partition(FiniteSet(1, 4), FiniteSet(2), FiniteSet(3, 5))
         """
         if len(rgs) != len(elements):
             raise ValueError('mismatch in rgs and element lengths')
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -47,9 +47,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
             >>> from sympy.combinatorics.polyhedron import Polyhedron
             >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
-            {(0, 1, 2)}
+            FiniteSet((0, 1, 2))
             >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
-            {(0, 1, 2)}
+            FiniteSet((0, 1, 2))
 
         The allowed transformations are entered as allowable permutations
         of the vertices for the polyhedron. Instance of Permutations
@@ -92,7 +92,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         >>> tetra.size
         4
         >>> tetra.edges
-        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
+        FiniteSet((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
         >>> tetra.corners
         (w, x, y, z)
 
@@ -365,7 +365,7 @@ def __new__(cls, corners, faces=[], pgroup=[]):
 
         >>> from sympy.combinatorics.polyhedron import cube
         >>> cube.edges
-        {(0, 1), (0, 3), (0, 4), '...', (4, 7), (5, 6), (6, 7)}
+        FiniteSet((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))
 
         If you want to use letters or other names for the corners you
         can still use the pre-calculated faces:
@@ -493,7 +493,7 @@ def edges(self):
         >>> corners = (a, b, c)
         >>> faces = [(0, 1, 2)]
         >>> Polyhedron(corners, faces).edges
-        {(0, 1), (0, 2), (1, 2)}
+        FiniteSet((0, 1), (0, 2), (1, 2))
 
         """
         if self._edges is None:
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -237,9 +237,9 @@ def nargs(self):
         corresponding set will be returned:
 
         >>> Function('f', nargs=1).nargs
-        {1}
+        FiniteSet(1)
         >>> Function('f', nargs=(2, 1)).nargs
-        {1, 2}
+        FiniteSet(1, 2)
 
         The undefined function, after application, also has the nargs
         attribute; the actual number of arguments is always available by
@@ -972,7 +972,7 @@ class WildFunction(Function, AtomicExpr):
 
     >>> F = WildFunction('F', nargs=2)
     >>> F.nargs
-    {2}
+    FiniteSet(2)
     >>> f(x).match(F)
     >>> f(x, y).match(F)
     {F_: f(x, y)}
@@ -983,7 +983,7 @@ class WildFunction(Function, AtomicExpr):
 
     >>> F = WildFunction('F', nargs=(1, 2))
     >>> F.nargs
-    {1, 2}
+    FiniteSet(1, 2)
     >>> f(x).match(F)
     {F_: f(x)}
     >>> f(x, y).match(F)
diff --git a/sympy/logic/boolalg.py b/sympy/logic/boolalg.py
--- a/sympy/logic/boolalg.py
+++ b/sympy/logic/boolalg.py
@@ -131,7 +131,7 @@ def as_set(self):
         >>> from sympy import Symbol, Eq, Or, And
         >>> x = Symbol('x', real=True)
         >>> Eq(x, 0).as_set()
-        {0}
+        FiniteSet(0)
         >>> (x > 0).as_set()
         Interval.open(0, oo)
         >>> And(-2 < x, x < 2).as_set()
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -149,14 +149,6 @@ def _print_Exp1(self, expr):
     def _print_ExprCondPair(self, expr):
         return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))
 
-    def _print_FiniteSet(self, s):
-        s = sorted(s, key=default_sort_key)
-        if len(s) > 10:
-            printset = s[:3] + ['...'] + s[-3:]
-        else:
-            printset = s
-        return '{' + ', '.join(self._print(el) for el in printset) + '}'
-
     def _print_Function(self, expr):
         return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")
 
diff --git a/sympy/sets/conditionset.py b/sympy/sets/conditionset.py
--- a/sympy/sets/conditionset.py
+++ b/sympy/sets/conditionset.py
@@ -65,20 +65,20 @@ class ConditionSet(Set):
 
     >>> c = ConditionSet(x, x < 1, {x, z})
     >>> c.subs(x, y)
-    ConditionSet(x, x < 1, {y, z})
+    ConditionSet(x, x < 1, FiniteSet(y, z))
 
     A second substitution is needed to change the dummy symbol, too:
 
     >>> _.subs(x, y)
-    ConditionSet(y, y < 1, {y, z})
+    ConditionSet(y, y < 1, FiniteSet(y, z))
 
     And trying to replace the dummy symbol with anything but a symbol
     is ignored: the only change possible will be in the base set:
 
     >>> ConditionSet(y, y < 1, {y, z}).subs(y, 1)
-    ConditionSet(y, y < 1, {z})
+    ConditionSet(y, y < 1, FiniteSet(z))
     >>> _.subs(y, 1)
-    ConditionSet(y, y < 1, {z})
+    ConditionSet(y, y < 1, FiniteSet(z))
 
     Notes
     =====
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -278,7 +278,7 @@ class ImageSet(Set):
     False
 
     >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
-    {1, 4, 9}
+    FiniteSet(1, 4, 9)
 
     >>> square_iterable = iter(squares)
     >>> for i in range(4):
@@ -300,7 +300,7 @@ class ImageSet(Set):
     >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
     >>> dom = Interval(-1, 1)
     >>> dom.intersect(solutions)
-    {0}
+    FiniteSet(0)
 
     See Also
     ========
@@ -921,7 +921,7 @@ def normalize_theta_set(theta):
     >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
     Interval(pi/2, 3*pi/2)
     >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
-    {0, pi}
+    FiniteSet(0, pi)
 
     """
     from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
@@ -1200,7 +1200,7 @@ def from_real(cls, sets):
         >>> from sympy import Interval, ComplexRegion
         >>> unit = Interval(0,1)
         >>> ComplexRegion.from_real(unit)
-        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))
+        CartesianComplexRegion(ProductSet(Interval(0, 1), FiniteSet(0)))
 
         """
         if not sets.is_subset(S.Reals):
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -43,7 +43,7 @@ class PowerSet(Set):
     A power set of a finite set:
 
     >>> PowerSet(FiniteSet(1, 2, 3))
-    PowerSet({1, 2, 3})
+    PowerSet(FiniteSet(1, 2, 3))
 
     A power set of an empty set:
 
@@ -60,7 +60,9 @@ class PowerSet(Set):
     Evaluating the power set of a finite set to its explicit form:
 
     >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
-    {EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}
+    FiniteSet(FiniteSet(1), FiniteSet(1, 2), FiniteSet(1, 3),
+            FiniteSet(1, 2, 3), FiniteSet(2), FiniteSet(2, 3),
+            FiniteSet(3), EmptySet)
 
     References
     ==========
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -109,7 +109,7 @@ def union(self, other):
         >>> Interval(0, 1) + Interval(2, 3)
         Union(Interval(0, 1), Interval(2, 3))
         >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
-        Union({3}, Interval.Lopen(1, 2))
+        Union(FiniteSet(3), Interval.Lopen(1, 2))
 
         Similarly it is possible to use the '-' operator for set differences:
 
@@ -469,7 +469,7 @@ def powerset(self):
         >>> from sympy import FiniteSet, EmptySet
         >>> A = EmptySet
         >>> A.powerset()
-        {EmptySet}
+        FiniteSet(EmptySet)
 
         A power set of a finite set:
 
@@ -532,9 +532,9 @@ def boundary(self):
 
         >>> from sympy import Interval
         >>> Interval(0, 1).boundary
-        {0, 1}
+        FiniteSet(0, 1)
         >>> Interval(0, 1, True, False).boundary
-        {0, 1}
+        FiniteSet(0, 1)
         """
         return self._boundary
 
@@ -659,7 +659,7 @@ class ProductSet(Set):
     >>> from sympy import Interval, FiniteSet, ProductSet
     >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
     >>> ProductSet(I, S)
-    ProductSet(Interval(0, 5), {1, 2, 3})
+    ProductSet(Interval(0, 5), FiniteSet(1, 2, 3))
 
     >>> (2, 2) in ProductSet(I, S)
     True
@@ -1492,7 +1492,7 @@ class Complement(Set, EvalfMixin):
 
     >>> from sympy import Complement, FiniteSet
     >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
-    {0, 2}
+    FiniteSet(0, 2)
 
     See Also
     =========
@@ -1683,18 +1683,18 @@ class FiniteSet(Set, EvalfMixin):
 
     >>> from sympy import FiniteSet
     >>> FiniteSet(1, 2, 3, 4)
-    {1, 2, 3, 4}
+    FiniteSet(1, 2, 3, 4)
     >>> 3 in FiniteSet(1, 2, 3, 4)
     True
 
     >>> members = [1, 2, 3, 4]
     >>> f = FiniteSet(*members)
     >>> f
-    {1, 2, 3, 4}
+    FiniteSet(1, 2, 3, 4)
     >>> f - FiniteSet(2)
-    {1, 3, 4}
+    FiniteSet(1, 3, 4)
     >>> f + FiniteSet(2, 5)
-    {1, 2, 3, 4, 5}
+    FiniteSet(1, 2, 3, 4, 5)
 
     References
     ==========
@@ -1893,7 +1893,7 @@ class SymmetricDifference(Set):
 
     >>> from sympy import SymmetricDifference, FiniteSet
     >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
-    {1, 2, 4, 5}
+    FiniteSet(1, 2, 4, 5)
 
     See Also
     ========
diff --git a/sympy/solvers/inequalities.py b/sympy/solvers/inequalities.py
--- a/sympy/solvers/inequalities.py
+++ b/sympy/solvers/inequalities.py
@@ -29,13 +29,13 @@ def solve_poly_inequality(poly, rel):
     >>> from sympy.solvers.inequalities import solve_poly_inequality
 
     >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
-    [{0}]
+    [FiniteSet(0)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
     [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]
 
     >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
-    [{-1}, {1}]
+    [FiniteSet(-1), FiniteSet(1)]
 
     See Also
     ========
@@ -141,7 +141,7 @@ def solve_rational_inequalities(eqs):
     >>> solve_rational_inequalities([[
     ... ((Poly(-x + 1), Poly(1, x)), '>='),
     ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
-    {1}
+    FiniteSet(1)
 
     >>> solve_rational_inequalities([[
     ... ((Poly(x), Poly(1, x)), '!='),
diff --git a/sympy/solvers/solveset.py b/sympy/solvers/solveset.py
--- a/sympy/solvers/solveset.py
+++ b/sympy/solvers/solveset.py
@@ -136,14 +136,14 @@ def _invert(f_x, y, x, domain=S.Complexes):
     >>> invert_complex(exp(x), y, x)
     (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
     >>> invert_real(exp(x), y, x)
-    (x, Intersection({log(y)}, Reals))
+    (x, Intersection(FiniteSet(log(y)), Reals))
 
     When does exp(x) == 1?
 
     >>> invert_complex(exp(x), 1, x)
     (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
     >>> invert_real(exp(x), 1, x)
-    (x, {0})
+    (x, FiniteSet(0))
 
     See Also
     ========
@@ -805,7 +805,7 @@ def solve_decomposition(f, symbol, domain):
     >>> x = Symbol('x')
     >>> f1 = exp(2*x) - 3*exp(x) + 2
     >>> sd(f1, x, S.Reals)
-    {0, log(2)}
+    FiniteSet(0, log(2))
     >>> f2 = sin(x)**2 + 2*sin(x) + 1
     >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
               3*pi
@@ -1365,11 +1365,11 @@ def _solve_exponential(lhs, rhs, symbol, domain):
     >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
     ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
     >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
-    ConditionSet(x, (a > 0) & (b > 0), {0})
+    ConditionSet(x, (a > 0) & (b > 0), FiniteSet(0))
     >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
-    {-3*log(2)/(-2*log(3) + log(2))}
+    FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
     >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
-    {0}
+    FiniteSet(0)
 
     * Proof of correctness of the method
 
@@ -1525,7 +1525,7 @@ def _solve_logarithm(lhs, rhs, symbol, domain):
     >>> x = symbols('x')
     >>> f = log(x - 3) + log(x + 3)
     >>> solve_log(f, 0, x, S.Reals)
-    {-sqrt(10), sqrt(10)}
+    FiniteSet(sqrt(10), -sqrt(10))
 
     * Proof of correctness
 
@@ -1679,7 +1679,7 @@ def _transolve(f, symbol, domain):
     >>> from sympy import symbols, S, pprint
     >>> x = symbols('x', real=True) # assumption added
     >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
-    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}
+    FiniteSet(-(log(3) + 3*log(5))/(-log(5) + 2*log(3)))
 
     How ``_transolve`` works
     ========================
@@ -1921,9 +1921,9 @@ def solveset(f, symbol=None, domain=S.Complexes):
     >>> R = S.Reals
     >>> x = Symbol('x')
     >>> solveset(exp(x) - 1, x, R)
-    {0}
+    FiniteSet(0)
     >>> solveset_real(exp(x) - 1, x)
-    {0}
+    FiniteSet(0)
 
     The solution is mostly unaffected by assumptions on the symbol,
     but there may be some slight difference:
@@ -2423,7 +2423,7 @@ def linsolve(system, *symbols):
     [6],
     [9]])
     >>> linsolve((A, b), [x, y, z])
-    {(-1, 2, 0)}
+    FiniteSet((-1, 2, 0))
 
     * Parametric Solution: In case the system is underdetermined, the
       function will return a parametric solution in terms of the given
@@ -2434,20 +2434,20 @@ def linsolve(system, *symbols):
     >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     >>> b = Matrix([3, 6, 9])
     >>> linsolve((A, b), x, y, z)
-    {(z - 1, 2 - 2*z, z)}
+    FiniteSet((z - 1, 2 - 2*z, z))
 
     If no symbols are given, internally generated symbols will be used.
     The `tau0` in the 3rd position indicates (as before) that the 3rd
     variable -- whatever it's named -- can take on any value:
 
     >>> linsolve((A, b))
-    {(tau0 - 1, 2 - 2*tau0, tau0)}
+    FiniteSet((tau0 - 1, 2 - 2*tau0, tau0))
 
     * List of Equations as input
 
     >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
     >>> linsolve(Eqns, x, y, z)
-    {(1, -2, -2)}
+    FiniteSet((1, -2, -2))
 
     * Augmented Matrix as input
 
@@ -2458,21 +2458,21 @@ def linsolve(system, *symbols):
     [2, 6,  8, 3],
     [6, 8, 18, 5]])
     >>> linsolve(aug, x, y, z)
-    {(3/10, 2/5, 0)}
+    FiniteSet((3/10, 2/5, 0))
 
     * Solve for symbolic coefficients
 
     >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
     >>> eqns = [a*x + b*y - c, d*x + e*y - f]
     >>> linsolve(eqns, x, y)
-    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}
+    FiniteSet(((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d)))
 
     * A degenerate system returns solution as set of given
       symbols.
 
     >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
     >>> linsolve(system, x, y)
-    {(x, y)}
+    FiniteSet((x, y))
 
     * For an empty system linsolve returns empty set
 
@@ -2483,7 +2483,7 @@ def linsolve(system, *symbols):
       is detected:
 
     >>> linsolve([x*(1/x - 1), (y - 1)**2 - y**2 + 1], x, y)
-    {(1, 1)}
+    FiniteSet((1, 1))
     >>> linsolve([x**2 - 1], x)
     Traceback (most recent call last):
     ...
@@ -2647,33 +2647,33 @@ def substitution(system, symbols, result=[{}], known_symbols=[],
     >>> x, y = symbols('x, y', real=True)
     >>> from sympy.solvers.solveset import substitution
     >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
-    {(-1, 1)}
+    FiniteSet((-1, 1))
 
     * when you want soln should not satisfy eq `x + 1 = 0`
 
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
     EmptySet
     >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
-    {(1, -1)}
+    FiniteSet((1, -1))
     >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
-    {(-3, 4), (2, -1)}
+    FiniteSet((-3, 4), (2, -1))
 
     * Returns both real and complex solution
 
     >>> x, y, z = symbols('x, y, z')
     >>> from sympy import exp, sin
     >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
-    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
-    (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
+    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
+            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
 
     >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
     >>> substitution(eqs, [y, z])
-    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
-    (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+    FiniteSet((-log(3), sqrt(-exp(2*x) - sin(log(3)))),
+    (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-    ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
+       ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
-    ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}
+       ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)))
 
     """
 
@@ -3254,7 +3254,7 @@ def nonlinsolve(system, *symbols):
     >>> from sympy.solvers.solveset import nonlinsolve
     >>> x, y, z = symbols('x, y, z', real=True)
     >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
-    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}
+    FiniteSet((-1, -1), (-1/2, -2), (1/2, 2), (1, 1))
 
     1. Positive dimensional system and complements:
 
@@ -3273,7 +3273,7 @@ def nonlinsolve(system, *symbols):
     {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
        d       d               d       d
     >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
-    {(2 - y, y)}
+    FiniteSet((2 - y, y))
 
     2. If some of the equations are non-polynomial then `nonlinsolve`
     will call the `substitution` function and return real and complex solutions,
@@ -3281,8 +3281,9 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import exp, sin
     >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
-    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
-    (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}
+    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
+            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))
+
 
     3. If system is non-linear polynomial and zero-dimensional then it
     returns both solution (real and complex solutions, if present) using
@@ -3290,7 +3291,7 @@ def nonlinsolve(system, *symbols):
 
     >>> from sympy import sqrt
     >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
-    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}
+    FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
 
     4. `nonlinsolve` can solve some linear (zero or positive dimensional)
     system (because it uses the `groebner` function to get the
@@ -3299,7 +3300,7 @@ def nonlinsolve(system, *symbols):
     `nonlinsolve`, because `linsolve` is better for general linear systems.
 
     >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9 , y + z - 4], [x, y, z])
-    {(3*z - 5, 4 - z, z)}
+    FiniteSet((3*z - 5, 4 - z, z))
 
     5. System having polynomial equations and only real solution is
     solved using `solve_poly_system`:
@@ -3307,11 +3308,11 @@ def nonlinsolve(system, *symbols):
     >>> e1 = sqrt(x**2 + y**2) - 10
     >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
     >>> nonlinsolve((e1, e2), (x, y))
-    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
+    FiniteSet((191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20))
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
-    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
+    FiniteSet((1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5)))
     >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
-    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}
+    FiniteSet((2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5)))
 
     6. It is better to use symbols instead of Trigonometric Function or
     Function (e.g. replace `sin(x)` with symbol, replace `f(x)` with symbol
diff --git a/sympy/stats/stochastic_process_types.py b/sympy/stats/stochastic_process_types.py
--- a/sympy/stats/stochastic_process_types.py
+++ b/sympy/stats/stochastic_process_types.py
@@ -553,7 +553,7 @@ class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
     >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
     >>> YS = DiscreteMarkovChain("Y")
     >>> Y.state_space
-    {0, 1, 2}
+    FiniteSet(0, 1, 2)
     >>> Y.transition_probabilities
     Matrix([
     [0.5, 0.2, 0.3],

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -707,8 +707,12 @@ def test_RandomDomain():
 
 
 def test_FiniteSet():
-    assert str(FiniteSet(*range(1, 51))) == '{1, 2, 3, ..., 48, 49, 50}'
-    assert str(FiniteSet(*range(1, 6))) == '{1, 2, 3, 4, 5}'
+    assert str(FiniteSet(*range(1, 51))) == (
+        'FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,'
+        ' 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,'
+        ' 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50)'
+    )
+    assert str(FiniteSet(*range(1, 6))) == 'FiniteSet(1, 2, 3, 4, 5)'
 
 
 def test_UniversalSet():

```


## Code snippets

### 1 - sympy/printing/str.py:

Start line: 188, End line: 205

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
### 2 - sympy/sets/sets.py:

Start line: 871, End line: 910

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
### 3 - sympy/plotting/intervalmath/interval_arithmetic.py:

Start line: 281, End line: 320

```python
class interval(object):

    def __div__(self, other):
        # Both None and False are handled
        if not self.is_valid:
            # Don't divide as the value is not valid
            return interval(-float('inf'), float('inf'), is_valid=self.is_valid)
        if isinstance(other, (int, float)):
            if other == 0:
                # Divide by zero encountered. valid nowhere
                return interval(-float('inf'), float('inf'), is_valid=False)
            else:
                return interval(self.start / other, self.end / other)

        elif isinstance(other, interval):
            if other.is_valid is False or self.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif other.is_valid is None or self.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
               # denominator contains both signs, i.e. being divided by zero
               # return the whole real line with is_valid = None
                if 0 in other:
                    return interval(-float('inf'), float('inf'), is_valid=None)

                # denominator negative
                this = self
                if other.end < 0:
                    this = -this
                    other = -other

                # denominator positive
                inters = []
                inters.append(this.start / other.start)
                inters.append(this.end / other.start)
                inters.append(this.start / other.end)
                inters.append(this.end / other.end)
                start = max(inters)
                end = min(inters)
                return interval(start, end)
        else:
            return NotImplemented
```
### 4 - sympy/printing/str.py:

Start line: 70, End line: 158

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

    def _print_FiniteSet(self, s):
        s = sorted(s, key=default_sort_key)
        if len(s) > 10:
            printset = s[:3] + ['...'] + s[-3:]
        else:
            printset = s
        return '{' + ', '.join(self._print(el) for el in printset) + '}'
```
### 5 - sympy/plotting/intervalmath/interval_arithmetic.py:

Start line: 146, End line: 163

```python
class interval(object):

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(False, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(True, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(False, valid)
            if not self.__lt__(other)[0] is None:
                return intervalMembership(True, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented
```
### 6 - sympy/plotting/intervalmath/interval_arithmetic.py:

Start line: 94, End line: 124

```python
class interval(object):

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            if self.end < other:
                return intervalMembership(True, self.is_valid)
            elif self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        elif isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end < other. start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            if self.start > other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__lt__(self)
        else:
            return NotImplemented
```
### 7 - sympy/printing/pretty/pretty.py:

Start line: 1906, End line: 1928

```python
class PrettyPrinter(Printer):

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_seq(items, '{', '}', ', ' )

    def _print_Range(self, s):

        if self._use_unicode:
            dots = u"\N{HORIZONTAL ELLIPSIS}"
        else:
            dots = '...'

        if s.start.is_infinite:
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
### 8 - sympy/printing/str.py:

Start line: 727, End line: 810

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

    def _print_SparseMatrix(self, expr):
        from sympy.matrices import Matrix
        return self._print(Matrix(expr))

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
### 9 - sympy/plotting/intervalmath/interval_arithmetic.py:

Start line: 165, End line: 182

```python
class interval(object):

    def __le__(self, other):
        if isinstance(other, (int, float)):
            if self.end <= other:
                return intervalMembership(True, self.is_valid)
            if self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end <= other.start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented
```
### 10 - sympy/sets/sets.py:

Start line: 1027, End line: 1040

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
### 12 - sympy/sets/sets.py:

Start line: 1042, End line: 1097

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
### 13 - sympy/printing/str.py:

Start line: 207, End line: 249

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
    _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))
```
### 14 - sympy/printing/str.py:

Start line: 583, End line: 656

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
### 19 - sympy/sets/sets.py:

Start line: 827, End line: 869

```python
class Interval(Set, EvalfMixin):
    """
    Represents a real interval as a Set.

    Usage:
        Returns an interval with end points "start" and "end".

        For left_open=True (default left_open is False) the interval
        will be open on the left. Similarly, for right_open=True the interval
        will be open on the right.

    Examples
    ========

    >>> from sympy import Symbol, Interval
    >>> Interval(0, 1)
    Interval(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Lopen(0, 1)
    Interval.Lopen(0, 1)
    >>> Interval.open(0, 1)
    Interval.open(0, 1)

    >>> a = Symbol('a', real=True)
    >>> Interval(0, a)
    Interval(0, a)

    Notes
    =====
    - Only real end points are supported
    - Interval(a, b) with a > b will return the empty set
    - Use the evalf() method to turn an Interval into an mpmath
      'mpi' interval instance

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_%28mathematics%29
    """
    is_Interval = True
```
### 20 - sympy/printing/str.py:

Start line: 446, End line: 519

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
### 22 - sympy/printing/str.py:

Start line: 363, End line: 386

```python
class StrPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
        if Permutation.print_cyclic:
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
### 24 - sympy/printing/str.py:

Start line: 388, End line: 444

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
### 25 - sympy/printing/str.py:

Start line: 160, End line: 186

```python
class StrPrinter(Printer):

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is special -- it's base is tuple
        return str(expr)

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'

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
### 28 - sympy/printing/str.py:

Start line: 658, End line: 678

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
### 29 - sympy/printing/str.py:

Start line: 828, End line: 859

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
### 31 - sympy/printing/str.py:

Start line: 18, End line: 44

```python
class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
    }

    _relationals = dict()

    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, string_types):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)
```
### 33 - sympy/sets/sets.py:

Start line: 1744, End line: 1781

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
### 34 - sympy/printing/str.py:

Start line: 324, End line: 361

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
        return "{0}({1}...)".format(
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
### 35 - sympy/sets/sets.py:

Start line: 1722, End line: 1742

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
### 42 - sympy/printing/str.py:

Start line: 812, End line: 826

```python
class StrPrinter(Printer):

    def _print_DMP(self, p):
        from sympy.core.sympify import SympifyError
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass

        cls = p.__class__.__name__
        rep = self._print(p.rep)
        dom = self._print(p.dom)
        ring = self._print(p.ring)

        return "%s(%s, %s, %s)" % (cls, rep, dom, ring)
```
### 50 - sympy/sets/sets.py:

Start line: 1862, End line: 1884

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
### 51 - sympy/printing/str.py:

Start line: 680, End line: 699

```python
class StrPrinter(Printer):

    def _print_Relational(self, expr):

        charmap = {
            "==": "Eq",
            "!=": "Ne",
            ":=": "Assignment",
            '+=': "AddAugmentedAssignment",
            "-=": "SubAugmentedAssignment",
            "*=": "MulAugmentedAssignment",
            "/=": "DivAugmentedAssignment",
            "%=": "ModAugmentedAssignment",
        }

        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs),
                                   self._print(expr.rhs))

        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))
```
### 54 - sympy/printing/str.py:

Start line: 46, End line: 68

```python
class StrPrinter(Printer):

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < PREC:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 58 - sympy/printing/str.py:

Start line: 251, End line: 266

```python
class StrPrinter(Printer):

    def _print_MatrixSlice(self, expr):
        def strslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(lambda arg: self._print(arg), x))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice) + ', ' +
                strslice(expr.colslice) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name
```
### 60 - sympy/printing/str.py:

Start line: 268, End line: 322

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
### 62 - sympy/sets/fancysets.py:

Start line: 716, End line: 838

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
### 66 - sympy/sets/fancysets.py:

Start line: 479, End line: 562

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
### 68 - sympy/sets/sets.py:

Start line: 912, End line: 1025

```python
class Interval(Set, EvalfMixin):

    @property
    def start(self):
        """
        The left end point of 'self'.

        This property takes the same value as the 'inf' property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]

    _inf = left = start

    @classmethod
    def open(cls, a, b):
        """Return an interval including neither boundary."""
        return cls(a, b, True, True)

    @classmethod
    def Lopen(cls, a, b):
        """Return an interval not including the left boundary."""
        return cls(a, b, True, False)

    @classmethod
    def Ropen(cls, a, b):
        """Return an interval not including the right boundary."""
        return cls(a, b, False, True)

    @property
    def end(self):
        """
        The right end point of 'self'.

        This property takes the same value as the 'sup' property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).end
        1

        """
        return self._args[1]

    _sup = right = end

    @property
    def left_open(self):
        """
        True if 'self' is left-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, left_open=True).left_open
        True
        >>> Interval(0, 1, left_open=False).left_open
        False

        """
        return self._args[2]

    @property
    def right_open(self):
        """
        True if 'self' is right-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, right_open=True).right_open
        True
        >>> Interval(0, 1, right_open=False).right_open
        False

        """
        return self._args[3]

    @property
    def is_empty(self):
        if self.left_open or self.right_open:
            cond = self.start >= self.end  # One/both bounds open
        else:
            cond = self.start > self.end  # Both bounds closed
        return fuzzy_bool(cond)

    def _complement(self, other):
        if other == S.Reals:
            a = Interval(S.NegativeInfinity, self.start,
                         True, not self.left_open)
            b = Interval(self.end, S.Infinity, not self.right_open, True)
            return Union(a, b)

        if isinstance(other, FiniteSet):
            nums = [m for m in other.args if m.is_number]
            if nums == []:
                return None

        return Set._complement(self, other)

    @property
    def _boundary(self):
        finite_points = [p for p in (self.start, self.end)
                         if abs(p) != S.Infinity]
        return FiniteSet(*finite_points)
```
### 71 - sympy/sets/sets.py:

Start line: 1677, End line: 1720

```python
class FiniteSet(Set, EvalfMixin):
    """
    Represents a finite set of discrete numbers

    Examples
    ========

    >>> from sympy import FiniteSet
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True

    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    is_FiniteSet = True
    is_iterable = True
    is_empty = False

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_evaluate[0])
        if evaluate:
            args = list(map(sympify, args))

            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))

        args = list(ordered(set(args), Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        return obj
```
### 75 - sympy/sets/fancysets.py:

Start line: 1, End line: 20

```python
from __future__ import print_function, division

from functools import reduce

from sympy.core.basic import Basic
from sympy.core.compatibility import with_metaclass, range, PY3
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import oo, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And
from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
    ProductSet)
from sympy.utilities.misc import filldedent
from sympy.utilities.iterables import cartes
```
