# sympy__sympy-18033

| **sympy/sympy** | `cab3c1cbfa415ced4ea4e46542ae7eb7044df6d6` |
| ---- | ---- |
| **No of patches** | 14 |
| **All found context length** | 13641 |
| **Any found context length** | 266 |
| **Avg pos** | 95.92857142857143 |
| **Min pos** | 1 |
| **Max pos** | 181 |
| **Top file pos** | 1 |
| **Missing snippets** | 88 |
| **Missing patch files** | 8 |


## Expected patch

```diff
diff --git a/sympy/codegen/array_utils.py b/sympy/codegen/array_utils.py
--- a/sympy/codegen/array_utils.py
+++ b/sympy/codegen/array_utils.py
@@ -571,7 +571,6 @@ def nest_permutation(self):
         >>> from sympy.codegen.array_utils import (CodegenArrayPermuteDims, CodegenArrayTensorProduct, nest_permutation)
         >>> from sympy import MatrixSymbol
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
 
         >>> M = MatrixSymbol("M", 3, 3)
         >>> N = MatrixSymbol("N", 3, 3)
@@ -1055,7 +1054,6 @@ def parse_indexed_expression(expr, first_indices=None):
     >>> from sympy.codegen.array_utils import parse_indexed_expression
     >>> from sympy import MatrixSymbol, Sum, symbols
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
 
     >>> i, j, k, d = symbols("i j k d")
     >>> M = MatrixSymbol("M", d, d)
diff --git a/sympy/combinatorics/generators.py b/sympy/combinatorics/generators.py
--- a/sympy/combinatorics/generators.py
+++ b/sympy/combinatorics/generators.py
@@ -15,7 +15,6 @@ def symmetric(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import symmetric
     >>> list(symmetric(3))
     [(2), (1 2), (2)(0 1), (0 1 2), (0 2 1), (0 2)]
@@ -32,7 +31,6 @@ def cyclic(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import cyclic
     >>> list(cyclic(5))
     [(4), (0 1 2 3 4), (0 2 4 1 3),
@@ -57,7 +55,6 @@ def alternating(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import alternating
     >>> list(alternating(3))
     [(2), (0 1 2), (0 2 1)]
@@ -80,7 +77,6 @@ def dihedral(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import dihedral
     >>> list(dihedral(3))
     [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]
diff --git a/sympy/combinatorics/homomorphisms.py b/sympy/combinatorics/homomorphisms.py
--- a/sympy/combinatorics/homomorphisms.py
+++ b/sympy/combinatorics/homomorphisms.py
@@ -445,7 +445,6 @@ def group_isomorphism(G, H, isomorphism=True):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
     >>> from sympy.combinatorics.free_groups import free_group
     >>> from sympy.combinatorics.fp_groups import FpGroup
diff --git a/sympy/combinatorics/named_groups.py b/sympy/combinatorics/named_groups.py
--- a/sympy/combinatorics/named_groups.py
+++ b/sympy/combinatorics/named_groups.py
@@ -20,7 +20,6 @@ def AbelianGroup(*cyclic_orders):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import AbelianGroup
     >>> AbelianGroup(3, 4)
     PermutationGroup([
diff --git a/sympy/combinatorics/perm_groups.py b/sympy/combinatorics/perm_groups.py
--- a/sympy/combinatorics/perm_groups.py
+++ b/sympy/combinatorics/perm_groups.py
@@ -36,7 +36,6 @@ class PermutationGroup(Basic):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.permutations import Cycle
     >>> from sympy.combinatorics.polyhedron import Polyhedron
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
@@ -1114,7 +1113,6 @@ def coset_factor(self, g, factor_index=False):
         ========
 
         >>> from sympy.combinatorics import Permutation, PermutationGroup
-        >>> Permutation.print_cyclic = True
         >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
         >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
         >>> G = PermutationGroup([a, b])
@@ -1239,7 +1237,6 @@ def coset_rank(self, g):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
         >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
@@ -1307,7 +1304,6 @@ def degree(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 0, 2])
         >>> G = PermutationGroup([a])
@@ -1423,7 +1419,6 @@ def derived_subgroup(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 0, 2, 4, 3])
         >>> b = Permutation([0, 1, 3, 2, 4])
@@ -1471,7 +1466,6 @@ def generate(self, method="coset", af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics import PermutationGroup
         >>> from sympy.combinatorics.polyhedron import tetrahedron
 
@@ -1518,7 +1512,6 @@ def generate_dimino(self, af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1, 3])
         >>> b = Permutation([0, 2, 3, 1])
@@ -1579,7 +1572,6 @@ def generate_schreier_sims(self, af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1, 3])
         >>> b = Permutation([0, 2, 3, 1])
@@ -1649,7 +1641,6 @@ def generators(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1])
         >>> b = Permutation([1, 0, 2])
@@ -1675,7 +1666,6 @@ def contains(self, g, strict=True):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
 
         >>> a = Permutation(1, 2)
@@ -1750,7 +1740,6 @@ def is_abelian(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1])
         >>> b = Permutation([1, 0, 2])
@@ -2055,7 +2044,6 @@ def is_normal(self, gr, strict=True):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 2, 0])
         >>> b = Permutation([1, 0, 2])
@@ -2725,7 +2713,6 @@ def orbit_rep(self, alpha, beta, schreier_vector=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import AlternatingGroup
         >>> G = AlternatingGroup(5)
@@ -2768,7 +2755,6 @@ def orbit_transversal(self, alpha, pairs=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import DihedralGroup
         >>> G = DihedralGroup(6)
@@ -3161,7 +3147,6 @@ def make_perm(self, n, seed=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a, b = [Permutation([1, 0, 3, 2]), Permutation([1, 3, 0, 2])]
         >>> G = PermutationGroup([a, b])
@@ -3696,7 +3681,6 @@ def stabilizer(self, alpha):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import DihedralGroup
         >>> G = DihedralGroup(6)
@@ -4919,7 +4903,6 @@ def _orbit_transversal(degree, generators, alpha, pairs, af=False, slp=False):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.perm_groups import _orbit_transversal
     >>> G = DihedralGroup(6)
@@ -4972,7 +4955,6 @@ def _stabilizer(degree, generators, alpha):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import _stabilizer
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> G = DihedralGroup(6)
diff --git a/sympy/combinatorics/permutations.py b/sympy/combinatorics/permutations.py
--- a/sympy/combinatorics/permutations.py
+++ b/sympy/combinatorics/permutations.py
@@ -23,7 +23,6 @@ def _af_rmul(a, b):
     ========
 
     >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-    >>> Permutation.print_cyclic = False
 
     >>> a, b = [1, 0, 2], [0, 2, 1]
     >>> _af_rmul(a, b)
@@ -57,7 +56,6 @@ def _af_rmuln(*abc):
     ========
 
     >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-    >>> Permutation.print_cyclic = False
 
     >>> a, b = [1, 0, 2], [0, 2, 1]
     >>> _af_rmul(a, b)
@@ -179,7 +177,6 @@ def _af_pow(a, n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation, _af_pow
-    >>> Permutation.print_cyclic = False
     >>> p = Permutation([2, 0, 3, 1])
     >>> p.order()
     4
@@ -358,7 +355,6 @@ def list(self, size=None):
 
         >>> from sympy.combinatorics.permutations import Cycle
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Cycle(2, 3)(4, 5)
         >>> p.list()
         [0, 1, 3, 2, 5, 4]
@@ -479,7 +475,8 @@ class Permutation(Atom):
     original ordering, not the elements (a, b, etc...) themselves.
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = False
+    >>> from sympy.interactive import init_printing
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
 
     Permutations Notation
     =====================
@@ -662,17 +659,17 @@ class Permutation(Atom):
 
     There are a few things to note about how Permutations are printed.
 
-    1) If you prefer one form (array or cycle) over another, you can set that
-    with the print_cyclic flag.
+    1) If you prefer one form (array or cycle) over another, you can set
+    ``init_printing`` with the ``perm_cyclic`` flag.
 
-    >>> Permutation(1, 2)(4, 5)(3, 4)
+    >>> from sympy import init_printing
+    >>> p = Permutation(1, 2)(4, 5)(3, 4)
+    >>> p
     Permutation([0, 2, 1, 4, 5, 3])
-    >>> p = _
 
-    >>> Permutation.print_cyclic = True
+    >>> init_printing(perm_cyclic=True, pretty_print=False)
     >>> p
     (1 2)(3 4 5)
-    >>> Permutation.print_cyclic = False
 
     2) Regardless of the setting, a list of elements in the array for cyclic
     form can be obtained and either of those can be copied and supplied as
@@ -688,6 +685,7 @@ class Permutation(Atom):
     3) Printing is economical in that as little as possible is printed while
     retaining all information about the size of the permutation:
 
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
     >>> Permutation([1, 0, 2, 3])
     Permutation([1, 0, 2, 3])
     >>> Permutation([1, 0, 2, 3], size=20)
@@ -696,10 +694,10 @@ class Permutation(Atom):
     Permutation([1, 0, 2, 4, 3], size=20)
 
     >>> p = Permutation([1, 0, 2, 3])
-    >>> Permutation.print_cyclic = True
+    >>> init_printing(perm_cyclic=True, pretty_print=False)
     >>> p
     (3)(0 1)
-    >>> Permutation.print_cyclic = False
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
 
     The 2 was not printed but it is still there as can be seen with the
     array_form and size methods:
@@ -776,7 +774,7 @@ class Permutation(Atom):
     Permutations:
 
     >>> p(['zero', 'one', 'four', 'two'])
-     ['one', 'zero', 'four', 'two']
+    ['one', 'zero', 'four', 'two']
     >>> p('zo42')
     ['o', 'z', '4', '2']
 
@@ -836,7 +834,8 @@ def __new__(cls, *args, **kwargs):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
 
         Permutations entered in array-form are left unaltered:
 
@@ -971,8 +970,9 @@ def _af_new(cls, perm):
         ========
 
         >>> from sympy.combinatorics.permutations import Perm
-        >>> Perm.print_cyclic = False
-        >>> a = [2,1,3,0]
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> a = [2, 1, 3, 0]
         >>> p = Perm._af_new(a)
         >>> p
         Permutation([2, 1, 3, 0])
@@ -996,7 +996,6 @@ def array_form(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([[2, 0], [3, 1]])
         >>> p.array_form
         [2, 3, 0, 1]
@@ -1009,29 +1008,6 @@ def array_form(self):
         """
         return self._array_form[:]
 
-    def __repr__(self):
-        if Permutation.print_cyclic:
-            if not self.size:
-                return 'Permutation()'
-            # before taking Cycle notation, see if the last element is
-            # a singleton and move it to the head of the string
-            s = Cycle(self)(self.size - 1).__repr__()[len('Cycle'):]
-            last = s.rfind('(')
-            if not last == 0 and ',' not in s[last:]:
-                s = s[last:] + s[:last]
-            return 'Permutation%s' %s
-        else:
-            s = self.support()
-            if not s:
-                if self.size < 5:
-                    return 'Permutation(%s)' % str(self.array_form)
-                return 'Permutation([], size=%s)' % self.size
-            trim = str(self.array_form[:s[-1] + 1]) + ', size=%s' % self.size
-            use = full = str(self.array_form)
-            if len(trim) < len(full):
-                use = trim
-            return 'Permutation(%s)' % use
-
     def list(self, size=None):
         """Return the permutation as an explicit list, possibly
         trimming unmoved elements if size is less than the maximum
@@ -1042,7 +1018,6 @@ def list(self, size=None):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation(2, 3)(4, 5)
         >>> p.list()
         [0, 1, 3, 2, 5, 4]
@@ -1083,7 +1058,6 @@ def cyclic_form(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([0, 3, 1, 2])
         >>> p.cyclic_form
         [[1, 3, 2]]
@@ -1179,7 +1153,6 @@ def __add__(self, other):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> I = Permutation([0, 1, 2, 3])
         >>> a = Permutation([2, 1, 3, 0])
         >>> I + a.rank() == a
@@ -1218,7 +1191,6 @@ def rmul(*args):
         ========
 
         >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-        >>> Permutation.print_cyclic = False
 
         >>> a, b = [1, 0, 2], [0, 2, 1]
         >>> a = Permutation(a); b = Permutation(b)
@@ -1283,7 +1255,6 @@ def __mul__(self, other):
         ========
 
         >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-        >>> Permutation.print_cyclic = False
 
         >>> a, b = [1, 0, 2], [0, 2, 1]
         >>> a = Permutation(a); b = Permutation(b)
@@ -1303,6 +1274,8 @@ def __mul__(self, other):
         It is acceptable for the arrays to have different lengths; the shorter
         one will be padded to match the longer one:
 
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> b*Permutation([1, 0])
         Permutation([1, 2, 0])
         >>> Permutation([1, 0])*b
@@ -1364,8 +1337,9 @@ def __pow__(self, n):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
-        >>> p = Permutation([2,0,3,1])
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> p = Permutation([2, 0, 3, 1])
         >>> p.order()
         4
         >>> p**4
@@ -1404,7 +1378,6 @@ def __xor__(self, h):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = True
         >>> p = Permutation(1, 2, 9)
         >>> q = Permutation(6, 9, 8)
         >>> p*q != q*p
@@ -1519,7 +1492,6 @@ def from_sequence(self, i, key=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
 
         >>> Permutation.from_sequence('SymPy')
         (4)(0 1 3)
@@ -1545,8 +1517,9 @@ def __invert__(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
-        >>> p = Permutation([[2,0], [3,1]])
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> p = Permutation([[2, 0], [3, 1]])
         >>> ~p
         Permutation([2, 3, 0, 1])
         >>> _ == p**-1
@@ -1569,6 +1542,10 @@ def __iter__(self):
         for i in self.array_form:
             yield i
 
+    def __repr__(self):
+        from sympy.printing.repr import srepr
+        return srepr(self)
+
     def __call__(self, *i):
         """
         Allows applying a permutation instance as a bijective function.
@@ -1676,7 +1653,8 @@ def unrank_nonlex(self, n, r):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.unrank_nonlex(4, 5)
         Permutation([2, 0, 3, 1])
         >>> Permutation.unrank_nonlex(4, -1)
@@ -1743,7 +1721,8 @@ def next_nonlex(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([2, 0, 3, 1]); p.rank_nonlex()
         5
         >>> p = p.next_nonlex(); p
@@ -2129,7 +2108,8 @@ def commutator(self, x):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([0, 2, 3, 1])
         >>> x = Permutation([2, 0, 3, 1])
         >>> c = p.commutator(x); c
@@ -2209,7 +2189,8 @@ def order(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([3, 1, 5, 2, 4, 0])
         >>> p.order()
         4
@@ -2254,7 +2235,6 @@ def cycle_structure(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> Permutation(3).cycle_structure
         {1: 4}
         >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure
@@ -2349,7 +2329,6 @@ def inversion_vector(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])
         >>> p.inversion_vector()
         [4, 7, 0, 5, 0, 2, 1, 1]
@@ -2364,13 +2343,12 @@ def inversion_vector(self):
         >>> while p:
         ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))
         ...     p = p.next_lex()
-        ...
-        Permutation([0, 1, 2]) [0, 0] 0
-        Permutation([0, 2, 1]) [0, 1] 1
-        Permutation([1, 0, 2]) [1, 0] 2
-        Permutation([1, 2, 0]) [1, 1] 3
-        Permutation([2, 0, 1]) [2, 0] 4
-        Permutation([2, 1, 0]) [2, 1] 5
+        (2) [0, 0] 0
+        (1 2) [0, 1] 1
+        (2)(0 1) [1, 0] 2
+        (0 1 2) [1, 1] 3
+        (0 2 1) [2, 0] 4
+        (0 2) [2, 1] 5
 
         See Also
         ========
@@ -2440,6 +2418,8 @@ def unrank_trotterjohnson(cls, size, rank):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.unrank_trotterjohnson(5, 10)
         Permutation([0, 3, 1, 2, 4])
 
@@ -2479,7 +2459,8 @@ def next_trotterjohnson(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([3, 0, 2, 1])
         >>> p.rank_trotterjohnson()
         4
@@ -2530,7 +2511,8 @@ def get_precedence_matrix(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation.josephus(3, 6, 1)
         >>> p
         Permutation([2, 5, 3, 1, 4, 0])
@@ -2761,7 +2743,8 @@ def from_inversion_vector(cls, inversion):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])
         Permutation([3, 2, 1, 0, 4, 5])
 
@@ -2807,7 +2790,8 @@ def unrank_lex(cls, size, rank):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> a = Permutation.unrank_lex(5, 10)
         >>> a.rank()
         10
@@ -2832,10 +2816,8 @@ def unrank_lex(cls, size, rank):
             psize = new_psize
         return cls._af_new(perm_array)
 
-    # global flag to control how permutations are printed
-    # when True, Permutation([0, 2, 1, 3]) -> Cycle(1, 2)
-    # when False, Permutation([0, 2, 1, 3]) -> Permutation([0, 2, 1])
-    print_cyclic = True
+    # XXX Deprecated flag
+    print_cyclic = None
 
 
 def _merge(arr, temp, left, mid, right):
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -60,8 +60,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
         >>> from sympy.abc import w, x, y, z
+        >>> init_printing(pretty_print=False, perm_cyclic=False)
 
         Here we construct the Polyhedron object for a tetrahedron.
 
diff --git a/sympy/combinatorics/tensor_can.py b/sympy/combinatorics/tensor_can.py
--- a/sympy/combinatorics/tensor_can.py
+++ b/sympy/combinatorics/tensor_can.py
@@ -918,7 +918,6 @@ def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
-    >>> Permutation.print_cyclic = True
     >>> base1, gens1 = get_symmetric_group_sgs(1)
     >>> base2, gens2 = get_symmetric_group_sgs(2)
     >>> bsgs_direct_product(base1, gens1, base2, gens2)
@@ -953,7 +952,6 @@ def get_symmetric_group_sgs(n, antisym=False):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
-    >>> Permutation.print_cyclic = True
     >>> get_symmetric_group_sgs(3)
     ([0, 1], [(4)(0 1), (4)(1 2)])
     """
@@ -1028,7 +1026,6 @@ def get_minimal_bsgs(base, gens):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
-    >>> Permutation.print_cyclic = True
     >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
     >>> get_minimal_bsgs(*riemann_bsgs1)
     ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
@@ -1059,7 +1056,6 @@ def tensor_gens(base, gens, list_free_indices, sym=0):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs
-    >>> Permutation.print_cyclic = True
 
     two symmetric tensors with 3 indices without free indices
 
@@ -1176,7 +1172,6 @@ def gens_products(*v):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
-    >>> Permutation.print_cyclic = True
     >>> base, gens = get_symmetric_group_sgs(2)
     >>> gens_products((base, gens, [[], []], 0))
     (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
diff --git a/sympy/combinatorics/util.py b/sympy/combinatorics/util.py
--- a/sympy/combinatorics/util.py
+++ b/sympy/combinatorics/util.py
@@ -143,7 +143,6 @@ def _distribute_gens_by_base(base, gens):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.util import _distribute_gens_by_base
     >>> D = DihedralGroup(3)
@@ -211,7 +210,6 @@ def _handle_precomputed_bsgs(base, strong_gens, transversals=None,
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.util import _handle_precomputed_bsgs
     >>> D = DihedralGroup(3)
@@ -271,7 +269,6 @@ def _orbits_transversals_from_bsgs(base, strong_gens_distr,
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.util import _orbits_transversals_from_bsgs
     >>> from sympy.combinatorics.util import (_orbits_transversals_from_bsgs,
@@ -415,7 +412,6 @@ def _strip(g, base, orbits, transversals):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.permutations import Permutation
     >>> from sympy.combinatorics.util import _strip
@@ -509,7 +505,6 @@ def _strong_gens_from_distr(strong_gens_distr):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
     ... _distribute_gens_by_base)
diff --git a/sympy/interactive/printing.py b/sympy/interactive/printing.py
--- a/sympy/interactive/printing.py
+++ b/sympy/interactive/printing.py
@@ -550,13 +550,16 @@ def init_printing(pretty_print=True, order=None, use_unicode=None,
         _stringify_func = stringify_func
 
         if pretty_print:
-            stringify_func = lambda expr: \
+            stringify_func = lambda expr, **settings: \
                              _stringify_func(expr, order=order,
                                              use_unicode=use_unicode,
                                              wrap_line=wrap_line,
-                                             num_columns=num_columns)
+                                             num_columns=num_columns,
+                                             **settings)
         else:
-            stringify_func = lambda expr: _stringify_func(expr, order=order)
+            stringify_func = \
+                lambda expr, **settings: _stringify_func(
+                    expr, order=order, **settings)
 
     if in_ipython:
         mode_in_settings = settings.pop("mode", None)
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -141,6 +141,7 @@ class LatexPrinter(Printer):
         "imaginary_unit": "i",
         "gothic_re_im": False,
         "decimal_separator": "period",
+        "perm_cyclic": True,
     }
 
     def __init__(self, settings=None):
@@ -374,7 +375,35 @@ def _print_Cycle(self, expr):
         term_tex = term_tex.replace(']', r"\right)")
         return term_tex
 
-    _print_Permutation = _print_Cycle
+    def _print_Permutation(self, expr):
+        from sympy.combinatorics.permutations import Permutation
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            return self._print_Cycle(expr)
+
+        if expr.size == 0:
+            return r"\left( \right)"
+
+        lower = [self._print(arg) for arg in expr.array_form]
+        upper = [self._print(arg) for arg in range(len(lower))]
+
+        row1 = " & ".join(upper)
+        row2 = " & ".join(lower)
+        mat = r" \\ ".join((row1, row2))
+        return r"\begin{pmatrix} %s \end{pmatrix}" % mat
+
 
     def _print_Float(self, expr):
         # Based off of that in StrPrinter
@@ -2501,7 +2530,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
           mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
           order=None, symbol_names=None, root_notation=True,
           mat_symbol_style="plain", imaginary_unit="i", gothic_re_im=False,
-          decimal_separator="period" ):
+          decimal_separator="period", perm_cyclic=True):
     r"""Convert the given expression to LaTeX string representation.
 
     Parameters
@@ -2702,6 +2731,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         'imaginary_unit': imaginary_unit,
         'gothic_re_im': gothic_re_im,
         'decimal_separator': decimal_separator,
+        'perm_cyclic' : perm_cyclic,
     }
 
     return LatexPrinter(settings).doprint(expr)
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -17,6 +17,7 @@
 from sympy.printing.str import sstr
 from sympy.utilities import default_sort_key
 from sympy.utilities.iterables import has_variety
+from sympy.utilities.exceptions import SymPyDeprecationWarning
 
 from sympy.printing.pretty.stringpict import prettyForm, stringPict
 from sympy.printing.pretty.pretty_symbology import xstr, hobj, vobj, xobj, \
@@ -42,6 +43,7 @@ class PrettyPrinter(Printer):
         "root_notation": True,
         "mat_symbol_style": "plain",
         "imaginary_unit": "i",
+        "perm_cyclic": True
     }
 
     def __init__(self, settings=None):
@@ -387,6 +389,41 @@ def _print_Cycle(self, dc):
             cyc = prettyForm(*cyc.right(l))
         return cyc
 
+    def _print_Permutation(self, expr):
+        from ..str import sstr
+        from sympy.combinatorics.permutations import Permutation, Cycle
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            return self._print_Cycle(Cycle(expr))
+
+        lower = expr.array_form
+        upper = list(range(len(lower)))
+
+        result = stringPict('')
+        first = True
+        for u, l in zip(upper, lower):
+            s1 = self._print(u)
+            s2 = self._print(l)
+            col = prettyForm(*s1.below(s2))
+            if first:
+                first = False
+            else:
+                col = prettyForm(*col.left(" "))
+            result = prettyForm(*result.right(col))
+        return prettyForm(*result.parens())
+
+
     def _print_Integral(self, integral):
         f = integral.function
 
@@ -2613,9 +2650,7 @@ def pretty(expr, **settings):
         pretty_use_unicode(uflag)
 
 
-def pretty_print(expr, wrap_line=True, num_columns=None, use_unicode=None,
-                 full_prec="auto", order=None, use_unicode_sqrt_char=True,
-                 root_notation = True, mat_symbol_style="plain", imaginary_unit="i"):
+def pretty_print(expr, **kwargs):
     """Prints expr in pretty form.
 
     pprint is just a shortcut for this function.
@@ -2658,11 +2693,7 @@ def pretty_print(expr, wrap_line=True, num_columns=None, use_unicode=None,
         Letter to use for imaginary unit when use_unicode is True.
         Can be "i" (default) or "j".
     """
-    print(pretty(expr, wrap_line=wrap_line, num_columns=num_columns,
-                 use_unicode=use_unicode, full_prec=full_prec, order=order,
-                 use_unicode_sqrt_char=use_unicode_sqrt_char,
-                 root_notation=root_notation, mat_symbol_style=mat_symbol_style,
-                 imaginary_unit=imaginary_unit))
+    print(pretty(expr, **kwargs))
 
 pprint = pretty_print
 
diff --git a/sympy/printing/repr.py b/sympy/printing/repr.py
--- a/sympy/printing/repr.py
+++ b/sympy/printing/repr.py
@@ -8,16 +8,19 @@
 from __future__ import print_function, division
 
 from sympy.core.function import AppliedUndef
-from .printer import Printer
 from mpmath.libmp import repr_dps, to_str as mlib_to_str
 from sympy.core.compatibility import range, string_types
 
+from .printer import Printer
+from .str import sstr
+
 
 class ReprPrinter(Printer):
     printmethod = "_sympyrepr"
 
     _default_settings = {
-        "order": None
+        "order": None,
+        "perm_cyclic" : True,
     }
 
     def reprify(self, args, sep):
@@ -57,7 +60,41 @@ def _print_Cycle(self, expr):
         return expr.__repr__()
 
     def _print_Permutation(self, expr):
-        return expr.__repr__()
+        from sympy.combinatorics.permutations import Permutation, Cycle
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            if not expr.size:
+                return 'Permutation()'
+            # before taking Cycle notation, see if the last element is
+            # a singleton and move it to the head of the string
+            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
+            last = s.rfind('(')
+            if not last == 0 and ',' not in s[last:]:
+                s = s[last:] + s[:last]
+            return 'Permutation%s' %s
+        else:
+            s = expr.support()
+            if not s:
+                if expr.size < 5:
+                    return 'Permutation(%s)' % str(expr.array_form)
+                return 'Permutation([], size=%s)' % expr.size
+            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
+            use = full = str(expr.array_form)
+            if len(trim) < len(full):
+                use = trim
+            return 'Permutation(%s)' % use
 
     def _print_Function(self, expr):
         r = self._print(expr.func)
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -22,6 +22,7 @@ class StrPrinter(Printer):
         "full_prec": "auto",
         "sympy_integers": False,
         "abbrev": False,
+        "perm_cyclic": True,
     }
 
     _relationals = dict()
@@ -354,7 +355,20 @@ def _print_Cycle(self, expr):
 
     def _print_Permutation(self, expr):
         from sympy.combinatorics.permutations import Permutation, Cycle
-        if Permutation.print_cyclic:
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
             if not expr.size:
                 return '()'
             # before taking Cycle notation, see if the last element is

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/codegen/array_utils.py | 574 | 574 | - | - | -
| sympy/codegen/array_utils.py | 1058 | 1058 | - | - | -
| sympy/combinatorics/generators.py | 18 | 18 | - | - | -
| sympy/combinatorics/generators.py | 35 | 35 | - | - | -
| sympy/combinatorics/generators.py | 60 | 60 | - | - | -
| sympy/combinatorics/generators.py | 83 | 83 | - | - | -
| sympy/combinatorics/homomorphisms.py | 448 | 448 | - | - | -
| sympy/combinatorics/named_groups.py | 23 | 23 | - | - | -
| sympy/combinatorics/perm_groups.py | 39 | 39 | - | - | -
| sympy/combinatorics/perm_groups.py | 1117 | 1117 | - | - | -
| sympy/combinatorics/perm_groups.py | 1242 | 1242 | - | - | -
| sympy/combinatorics/perm_groups.py | 1310 | 1310 | - | - | -
| sympy/combinatorics/perm_groups.py | 1426 | 1426 | - | - | -
| sympy/combinatorics/perm_groups.py | 1474 | 1474 | - | - | -
| sympy/combinatorics/perm_groups.py | 1521 | 1521 | - | - | -
| sympy/combinatorics/perm_groups.py | 1582 | 1582 | - | - | -
| sympy/combinatorics/perm_groups.py | 1652 | 1652 | - | - | -
| sympy/combinatorics/perm_groups.py | 1678 | 1678 | - | - | -
| sympy/combinatorics/perm_groups.py | 1753 | 1753 | - | - | -
| sympy/combinatorics/perm_groups.py | 2058 | 2058 | - | - | -
| sympy/combinatorics/perm_groups.py | 2728 | 2728 | - | - | -
| sympy/combinatorics/perm_groups.py | 2771 | 2771 | - | - | -
| sympy/combinatorics/perm_groups.py | 3164 | 3164 | - | - | -
| sympy/combinatorics/perm_groups.py | 3699 | 3699 | - | - | -
| sympy/combinatorics/perm_groups.py | 4922 | 4922 | - | - | -
| sympy/combinatorics/perm_groups.py | 4975 | 4975 | - | - | -
| sympy/combinatorics/permutations.py | 26 | 26 | - | 3 | -
| sympy/combinatorics/permutations.py | 60 | 60 | - | 3 | -
| sympy/combinatorics/permutations.py | 182 | 182 | - | 3 | -
| sympy/combinatorics/permutations.py | 361 | 361 | - | 3 | -
| sympy/combinatorics/permutations.py | 482 | 482 | 5 | 3 | 4245
| sympy/combinatorics/permutations.py | 665 | 675 | 5 | 3 | 4245
| sympy/combinatorics/permutations.py | 691 | 691 | 5 | 3 | 4245
| sympy/combinatorics/permutations.py | 699 | 702 | 5 | 3 | 4245
| sympy/combinatorics/permutations.py | 779 | 779 | 5 | 3 | 4245
| sympy/combinatorics/permutations.py | 839 | 839 | - | 3 | -
| sympy/combinatorics/permutations.py | 974 | 975 | - | 3 | -
| sympy/combinatorics/permutations.py | 999 | 999 | - | 3 | -
| sympy/combinatorics/permutations.py | 1012 | 1034 | - | 3 | -
| sympy/combinatorics/permutations.py | 1045 | 1045 | 116 | 3 | 40238
| sympy/combinatorics/permutations.py | 1086 | 1086 | 17 | 3 | 8452
| sympy/combinatorics/permutations.py | 1182 | 1182 | - | 3 | -
| sympy/combinatorics/permutations.py | 1221 | 1221 | - | 3 | -
| sympy/combinatorics/permutations.py | 1286 | 1286 | - | 3 | -
| sympy/combinatorics/permutations.py | 1306 | 1306 | - | 3 | -
| sympy/combinatorics/permutations.py | 1367 | 1368 | - | 3 | -
| sympy/combinatorics/permutations.py | 1407 | 1407 | - | 3 | -
| sympy/combinatorics/permutations.py | 1522 | 1522 | - | 3 | -
| sympy/combinatorics/permutations.py | 1548 | 1549 | - | 3 | -
| sympy/combinatorics/permutations.py | 1572 | 1572 | - | 3 | -
| sympy/combinatorics/permutations.py | 1679 | 1679 | - | 3 | -
| sympy/combinatorics/permutations.py | 1746 | 1746 | - | 3 | -
| sympy/combinatorics/permutations.py | 2132 | 2132 | - | 3 | -
| sympy/combinatorics/permutations.py | 2212 | 2212 | - | 3 | -
| sympy/combinatorics/permutations.py | 2257 | 2257 | 154 | 3 | 53957
| sympy/combinatorics/permutations.py | 2352 | 2352 | - | 3 | -
| sympy/combinatorics/permutations.py | 2367 | 2373 | - | 3 | -
| sympy/combinatorics/permutations.py | 2443 | 2443 | - | 3 | -
| sympy/combinatorics/permutations.py | 2482 | 2482 | - | 3 | -
| sympy/combinatorics/permutations.py | 2533 | 2533 | - | 3 | -
| sympy/combinatorics/permutations.py | 2764 | 2764 | - | 3 | -
| sympy/combinatorics/permutations.py | 2810 | 2810 | 181 | 3 | 61844
| sympy/combinatorics/permutations.py | 2835 | 2838 | 181 | 3 | 61844
| sympy/combinatorics/polyhedron.py | 63 | 63 | - | - | -
| sympy/combinatorics/tensor_can.py | 921 | 921 | - | - | -
| sympy/combinatorics/tensor_can.py | 956 | 956 | - | - | -
| sympy/combinatorics/tensor_can.py | 1031 | 1031 | - | - | -
| sympy/combinatorics/tensor_can.py | 1062 | 1062 | - | - | -
| sympy/combinatorics/tensor_can.py | 1179 | 1179 | - | - | -
| sympy/combinatorics/util.py | 146 | 146 | - | - | -
| sympy/combinatorics/util.py | 214 | 214 | - | - | -
| sympy/combinatorics/util.py | 274 | 274 | - | - | -
| sympy/combinatorics/util.py | 418 | 418 | - | - | -
| sympy/combinatorics/util.py | 512 | 512 | - | - | -
| sympy/interactive/printing.py | 553 | 559 | 135 | 20 | 47476
| sympy/printing/latex.py | 144 | 144 | - | 11 | -
| sympy/printing/latex.py | 377 | 377 | 52 | 11 | 20138
| sympy/printing/latex.py | 2504 | 2504 | - | 11 | -
| sympy/printing/latex.py | 2705 | 2705 | - | 11 | -
| sympy/printing/pretty/pretty.py | 20 | 20 | 3 | 2 | 685
| sympy/printing/pretty/pretty.py | 45 | 45 | 129 | 2 | 44093
| sympy/printing/pretty/pretty.py | 390 | 390 | 153 | 2 | 53742
| sympy/printing/pretty/pretty.py | 2616 | 2618 | - | 2 | -
| sympy/printing/pretty/pretty.py | 2661 | 2665 | - | 2 | -
| sympy/printing/repr.py | 11 | 20 | - | 10 | -
| sympy/printing/repr.py | 60 | 60 | 45 | 10 | 17998
| sympy/printing/str.py | 25 | 25 | 30 | 1 | 13641
| sympy/printing/str.py | 357 | 357 | 1 | 1 | 266


## Problem Statement

```
Remove Permutation.print_cyclic flag
See the discussion at https://github.com/sympy/sympy/pull/15198. The Permutation printing should be handled in the SymPy printers, not on the object itself. The flag should be a flag to the printer. Any doctest that wants to change the printing should set the flag in `init_printing`. However, whichever is set as the default should be used everywhere. 

Since it is publicly documented, it will need to be deprecated https://github.com/sympy/sympy/wiki/Deprecating-policy.

Additionally, it would be best if the `str` printer printed a Python valid representation and the pretty printers only (pprint/latex) printed things like (1 2 3).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/printing/str.py** | 355 | 378| 266 | 266 | 7287 | 
| 2 | **2 sympy/printing/pretty/pretty.py** | 371 | 388| 152 | 418 | 30542 | 
| **-> 3 <-** | **2 sympy/printing/pretty/pretty.py** | 1 | 28| 267 | 685 | 30542 | 
| 4 | **3 sympy/combinatorics/permutations.py** | 383 | 404| 179 | 864 | 53159 | 
| **-> 5 <-** | **3 sympy/combinatorics/permutations.py** | 470 | 827| 3381 | 4245 | 53159 | 
| 6 | **3 sympy/printing/pretty/pretty.py** | 2431 | 2463| 272 | 4517 | 53159 | 
| 7 | **3 sympy/printing/str.py** | 804 | 818| 132 | 4649 | 53159 | 
| 8 | **3 sympy/printing/str.py** | 719 | 802| 678 | 5327 | 53159 | 
| 9 | **3 sympy/combinatorics/permutations.py** | 1012 | 1033| 235 | 5562 | 53159 | 
| 10 | **3 sympy/printing/pretty/pretty.py** | 266 | 332| 583 | 6145 | 53159 | 
| 11 | 4 sympy/printing/codeprinter.py | 502 | 537| 400 | 6545 | 57676 | 
| 12 | **4 sympy/printing/pretty/pretty.py** | 1401 | 1433| 258 | 6803 | 57676 | 
| 13 | **4 sympy/printing/pretty/pretty.py** | 2189 | 2197| 116 | 6919 | 57676 | 
| 14 | **4 sympy/printing/pretty/pretty.py** | 228 | 243| 165 | 7084 | 57676 | 
| 15 | **4 sympy/printing/str.py** | 380 | 436| 541 | 7625 | 57676 | 
| 16 | 4 sympy/printing/codeprinter.py | 398 | 449| 507 | 8132 | 57676 | 
| **-> 17 <-** | **4 sympy/combinatorics/permutations.py** | 1076 | 1118| 320 | 8452 | 57676 | 
| 18 | **4 sympy/printing/pretty/pretty.py** | 1588 | 1607| 194 | 8646 | 57676 | 
| 19 | 5 sympy/printing/pycode.py | 1 | 75| 650 | 9296 | 65923 | 
| 20 | **5 sympy/combinatorics/permutations.py** | 406 | 427| 176 | 9472 | 65923 | 
| 21 | 6 sympy/printing/printer.py | 1 | 173| 1419 | 10891 | 68356 | 
| 22 | **6 sympy/printing/pretty/pretty.py** | 135 | 202| 639 | 11530 | 68356 | 
| 23 | **6 sympy/printing/pretty/pretty.py** | 1935 | 1988| 388 | 11918 | 68356 | 
| 24 | **6 sympy/printing/pretty/pretty.py** | 1906 | 1933| 251 | 12169 | 68356 | 
| 25 | **6 sympy/printing/str.py** | 316 | 353| 284 | 12453 | 68356 | 
| 26 | 6 sympy/printing/codeprinter.py | 451 | 500| 449 | 12902 | 68356 | 
| 27 | **6 sympy/printing/pretty/pretty.py** | 245 | 264| 165 | 13067 | 68356 | 
| 28 | **6 sympy/printing/pretty/pretty.py** | 2413 | 2429| 184 | 13251 | 68356 | 
| 29 | 7 sympy/printing/pretty/pretty_symbology.py | 50 | 73| 201 | 13452 | 74382 | 
| **-> 30 <-** | **7 sympy/printing/str.py** | 18 | 44| 189 | 13641 | 74382 | 
| 31 | **7 sympy/printing/pretty/pretty.py** | 629 | 655| 261 | 13902 | 74382 | 
| 32 | **7 sympy/printing/str.py** | 1 | 15| 108 | 14010 | 74382 | 
| 33 | 7 sympy/printing/pycode.py | 805 | 818| 146 | 14156 | 74382 | 
| 34 | **7 sympy/printing/pretty/pretty.py** | 2374 | 2398| 212 | 14368 | 74382 | 
| 35 | 8 sympy/printing/defaults.py | 1 | 21| 157 | 14525 | 74539 | 
| 36 | **8 sympy/printing/str.py** | 260 | 314| 501 | 15026 | 74539 | 
| 37 | 9 sympy/printing/fcode.py | 757 | 773| 212 | 15238 | 82529 | 
| 38 | **9 sympy/printing/str.py** | 438 | 511| 529 | 15767 | 82529 | 
| 39 | **9 sympy/printing/pretty/pretty.py** | 2137 | 2187| 366 | 16133 | 82529 | 
| 40 | 9 sympy/printing/pycode.py | 917 | 940| 209 | 16342 | 82529 | 
| 41 | **9 sympy/printing/str.py** | 199 | 241| 448 | 16790 | 82529 | 
| 42 | 9 sympy/printing/pycode.py | 821 | 855| 315 | 17105 | 82529 | 
| 43 | **9 sympy/printing/pretty/pretty.py** | 105 | 121| 199 | 17304 | 82529 | 
| 44 | 9 sympy/printing/pycode.py | 342 | 364| 247 | 17551 | 82529 | 
| **-> 45 <-** | **10 sympy/printing/repr.py** | 47 | 107| 447 | 17998 | 84860 | 
| 46 | **10 sympy/printing/pretty/pretty.py** | 2095 | 2135| 316 | 18314 | 84860 | 
| 47 | **10 sympy/printing/pretty/pretty.py** | 2006 | 2032| 241 | 18555 | 84860 | 
| 48 | **10 sympy/printing/pretty/pretty.py** | 2337 | 2352| 179 | 18734 | 84860 | 
| 49 | **10 sympy/printing/pretty/pretty.py** | 2465 | 2478| 129 | 18863 | 84860 | 
| 50 | **10 sympy/printing/str.py** | 820 | 851| 287 | 19150 | 84860 | 
| 51 | **10 sympy/printing/str.py** | 70 | 163| 823 | 19973 | 84860 | 
| **-> 52 <-** | **11 sympy/printing/latex.py** | 361 | 377| 165 | 20138 | 112458 | 
| 53 | 12 sympy/printing/ccode.py | 642 | 653| 107 | 20245 | 120644 | 
| 54 | 13 sympy/physics/vector/printing.py | 1 | 12| 134 | 20379 | 124059 | 
| 55 | **13 sympy/printing/pretty/pretty.py** | 2199 | 2307| 799 | 21178 | 124059 | 
| 56 | **13 sympy/printing/pretty/pretty.py** | 1378 | 1399| 231 | 21409 | 124059 | 
| 57 | 14 sympy/printing/cxxcode.py | 110 | 137| 285 | 21694 | 125682 | 
| 58 | **14 sympy/printing/pretty/pretty.py** | 93 | 103| 135 | 21829 | 125682 | 
| 59 | **14 sympy/printing/pretty/pretty.py** | 2480 | 2544| 612 | 22441 | 125682 | 
| 60 | **14 sympy/printing/pretty/pretty.py** | 1880 | 1904| 228 | 22669 | 125682 | 
| 61 | **14 sympy/printing/str.py** | 575 | 648| 543 | 23212 | 125682 | 
| 62 | 14 sympy/printing/ccode.py | 626 | 639| 157 | 23369 | 125682 | 
| 63 | **14 sympy/printing/pretty/pretty.py** | 2354 | 2372| 220 | 23589 | 125682 | 
| 64 | **14 sympy/printing/pretty/pretty.py** | 466 | 518| 416 | 24005 | 125682 | 
| 65 | 14 sympy/printing/pycode.py | 440 | 462| 222 | 24227 | 125682 | 
| 66 | **14 sympy/printing/pretty/pretty.py** | 1246 | 1292| 403 | 24630 | 125682 | 
| 67 | **14 sympy/printing/pretty/pretty.py** | 1553 | 1567| 177 | 24807 | 125682 | 
| 68 | 14 sympy/printing/pycode.py | 78 | 141| 523 | 25330 | 125682 | 
| 69 | 14 sympy/printing/printer.py | 175 | 249| 460 | 25790 | 125682 | 
| 70 | **14 sympy/printing/pretty/pretty.py** | 896 | 910| 140 | 25930 | 125682 | 
| 71 | 14 sympy/printing/ccode.py | 1 | 86| 774 | 26704 | 125682 | 
| 72 | **14 sympy/printing/pretty/pretty.py** | 204 | 226| 195 | 26899 | 125682 | 
| 73 | 15 sympy/printing/julia.py | 358 | 374| 181 | 27080 | 131581 | 
| 74 | **15 sympy/printing/pretty/pretty.py** | 1166 | 1220| 420 | 27500 | 131581 | 
| 75 | **15 sympy/printing/pretty/pretty.py** | 1696 | 1737| 325 | 27825 | 131581 | 
| 76 | 16 sympy/printing/__init__.py | 1 | 73| 518 | 28343 | 132099 | 
| 77 | 16 sympy/printing/pycode.py | 191 | 224| 247 | 28590 | 132099 | 
| 78 | **16 sympy/printing/pretty/pretty.py** | 1435 | 1455| 241 | 28831 | 132099 | 
| 79 | 16 sympy/printing/ccode.py | 656 | 685| 312 | 29143 | 132099 | 
| 80 | 16 sympy/printing/codeprinter.py | 1 | 35| 233 | 29376 | 132099 | 
| 81 | **16 sympy/printing/pretty/pretty.py** | 2070 | 2093| 227 | 29603 | 132099 | 
| 82 | **16 sympy/printing/pretty/pretty.py** | 1005 | 1024| 217 | 29820 | 132099 | 
| 83 | **16 sympy/printing/pretty/pretty.py** | 1538 | 1551| 142 | 29962 | 132099 | 
| 84 | **16 sympy/printing/pretty/pretty.py** | 334 | 369| 288 | 30250 | 132099 | 
| 85 | 17 sympy/combinatorics/__init__.py | 1 | 41| 387 | 30637 | 132486 | 
| 86 | **17 sympy/printing/pretty/pretty.py** | 1517 | 1536| 236 | 30873 | 132486 | 
| 87 | 17 sympy/printing/pycode.py | 367 | 380| 152 | 31025 | 132486 | 
| 88 | **17 sympy/printing/pretty/pretty.py** | 2400 | 2411| 147 | 31172 | 132486 | 
| 89 | **17 sympy/printing/pretty/pretty.py** | 123 | 133| 134 | 31306 | 132486 | 
| 90 | 17 sympy/printing/fcode.py | 400 | 415| 124 | 31430 | 132486 | 
| 91 | **17 sympy/printing/pretty/pretty.py** | 1990 | 2004| 174 | 31604 | 132486 | 
| 92 | 17 sympy/printing/fcode.py | 149 | 164| 131 | 31735 | 132486 | 
| 93 | **17 sympy/printing/repr.py** | 186 | 219| 382 | 32117 | 132486 | 
| 94 | **17 sympy/printing/pretty/pretty.py** | 1842 | 1857| 184 | 32301 | 132486 | 
| 95 | 17 sympy/printing/codeprinter.py | 371 | 396| 292 | 32593 | 132486 | 
| 96 | 17 sympy/printing/pycode.py | 574 | 601| 256 | 32849 | 132486 | 
| 97 | **17 sympy/printing/pretty/pretty.py** | 1609 | 1632| 258 | 33107 | 132486 | 
| 98 | 17 sympy/printing/julia.py | 1 | 43| 496 | 33603 | 132486 | 
| 99 | **17 sympy/printing/pretty/pretty.py** | 1739 | 1794| 564 | 34167 | 132486 | 
| 100 | 17 sympy/physics/vector/printing.py | 382 | 422| 336 | 34503 | 132486 | 
| 101 | **17 sympy/printing/pretty/pretty.py** | 1569 | 1586| 184 | 34687 | 132486 | 
| 102 | 17 sympy/printing/fcode.py | 166 | 189| 207 | 34894 | 132486 | 
| 103 | 17 sympy/printing/pycode.py | 690 | 751| 769 | 35663 | 132486 | 
| 104 | **17 sympy/printing/pretty/pretty.py** | 2546 | 2554| 119 | 35782 | 132486 | 
| 105 | 17 sympy/printing/fcode.py | 668 | 688| 192 | 35974 | 132486 | 
| 106 | **17 sympy/printing/pretty/pretty.py** | 1294 | 1376| 773 | 36747 | 132486 | 
| 107 | **17 sympy/printing/str.py** | 46 | 68| 160 | 36907 | 132486 | 
| 108 | **17 sympy/combinatorics/permutations.py** | 909 | 949| 371 | 37278 | 132486 | 
| 109 | 17 sympy/printing/pycode.py | 281 | 340| 539 | 37817 | 132486 | 
| 110 | 17 sympy/printing/codeprinter.py | 131 | 208| 718 | 38535 | 132486 | 
| 111 | **17 sympy/printing/pretty/pretty.py** | 854 | 894| 352 | 38887 | 132486 | 
| 112 | **17 sympy/printing/pretty/pretty.py** | 1137 | 1164| 212 | 39099 | 132486 | 
| 113 | 17 sympy/printing/fcode.py | 299 | 320| 253 | 39352 | 132486 | 
| 114 | **17 sympy/printing/pretty/pretty.py** | 800 | 823| 216 | 39568 | 132486 | 
| 115 | 18 sympy/printing/python.py | 1 | 42| 325 | 39893 | 133170 | 
| **-> 116 <-** | **18 sympy/combinatorics/permutations.py** | 1035 | 1074| 345 | 40238 | 133170 | 
| 117 | **18 sympy/printing/latex.py** | 974 | 1029| 577 | 40815 | 133170 | 
| 118 | 19 sympy/printing/pretty/__init__.py | 1 | 8| 0 | 40815 | 133224 | 
| 119 | **19 sympy/printing/pretty/pretty.py** | 842 | 852| 124 | 40939 | 133224 | 
| 120 | 19 sympy/printing/ccode.py | 486 | 511| 250 | 41189 | 133224 | 
| **-> 121 <-** | **20 sympy/interactive/printing.py** | 337 | 571| 114 | 41303 | 138289 | 
| 122 | 21 sympy/printing/dot.py | 1 | 19| 137 | 41440 | 140566 | 
| 123 | 21 sympy/printing/julia.py | 119 | 191| 697 | 42137 | 140566 | 
| 124 | **21 sympy/printing/repr.py** | 173 | 184| 141 | 42278 | 140566 | 
| 125 | 21 sympy/printing/codeprinter.py | 38 | 71| 199 | 42477 | 140566 | 
| 126 | 22 sympy/printing/pretty/stringpict.py | 1 | 18| 173 | 42650 | 144935 | 
| 127 | **22 sympy/printing/pretty/pretty.py** | 1649 | 1666| 173 | 42823 | 144935 | 
| 128 | **22 sympy/printing/latex.py** | 2236 | 2309| 735 | 43558 | 144935 | 
| **-> 129 <-** | **22 sympy/printing/pretty/pretty.py** | 31 | 91| 535 | 44093 | 144935 | 
| 130 | **22 sympy/printing/pretty/pretty.py** | 1064 | 1097| 331 | 44424 | 144935 | 
| 131 | **22 sympy/printing/repr.py** | 123 | 154| 256 | 44680 | 144935 | 
| 132 | **22 sympy/interactive/printing.py** | 347 | 494| 1662 | 46342 | 144935 | 
| 133 | 22 sympy/printing/fcode.py | 435 | 451| 149 | 46491 | 144935 | 
| 134 | 22 sympy/printing/cxxcode.py | 78 | 107| 285 | 46776 | 144935 | 
| **-> 135 <-** | **22 sympy/interactive/printing.py** | 495 | 572| 700 | 47476 | 144935 | 
| 136 | **22 sympy/printing/pretty/pretty.py** | 577 | 627| 422 | 47898 | 144935 | 
| 137 | 22 sympy/printing/cxxcode.py | 1 | 60| 649 | 48547 | 144935 | 
| 138 | 22 sympy/printing/fcode.py | 1 | 63| 499 | 49046 | 144935 | 
| 139 | 22 sympy/printing/ccode.py | 146 | 249| 777 | 49823 | 144935 | 
| 140 | 22 sympy/printing/pycode.py | 857 | 914| 549 | 50372 | 144935 | 
| 141 | **22 sympy/printing/str.py** | 650 | 670| 189 | 50561 | 144935 | 
| 142 | 22 sympy/printing/ccode.py | 403 | 415| 164 | 50725 | 144935 | 
| 143 | **22 sympy/printing/pretty/pretty.py** | 1099 | 1135| 307 | 51032 | 144935 | 
| 144 | 22 sympy/printing/fcode.py | 629 | 653| 204 | 51236 | 144935 | 
| 145 | **22 sympy/printing/pretty/pretty.py** | 1222 | 1244| 180 | 51416 | 144935 | 
| 146 | **22 sympy/printing/latex.py** | 2215 | 2225| 124 | 51540 | 144935 | 
| 147 | 22 sympy/printing/ccode.py | 513 | 532| 193 | 51733 | 144935 | 
| 148 | **22 sympy/printing/pretty/pretty.py** | 1634 | 1647| 180 | 51913 | 144935 | 
| 149 | 22 sympy/printing/codeprinter.py | 210 | 245| 257 | 52170 | 144935 | 
| 150 | 22 sympy/printing/printer.py | 251 | 305| 557 | 52727 | 144935 | 
| 151 | 22 sympy/printing/fcode.py | 655 | 666| 124 | 52851 | 144935 | 
| 152 | **22 sympy/printing/pretty/pretty.py** | 726 | 751| 284 | 53135 | 144935 | 
| **-> 153 <-** | **22 sympy/printing/pretty/pretty.py** | 390 | 464| 607 | 53742 | 144935 | 
| **-> 154 <-** | **22 sympy/combinatorics/permutations.py** | 2248 | 2274| 215 | 53957 | 144935 | 
| 155 | 22 sympy/printing/cxxcode.py | 139 | 155| 130 | 54087 | 144935 | 
| 156 | 23 sympy/printing/rcode.py | 1 | 74| 485 | 54572 | 148652 | 
| 157 | 23 sympy/printing/pycode.py | 539 | 571| 333 | 54905 | 148652 | 
| 158 | **23 sympy/printing/pretty/pretty.py** | 912 | 924| 134 | 55039 | 148652 | 
| 159 | **23 sympy/printing/latex.py** | 1991 | 2063| 630 | 55669 | 148652 | 
| 160 | **23 sympy/printing/pretty/pretty.py** | 1859 | 1878| 199 | 55868 | 148652 | 
| 161 | **23 sympy/printing/pretty/pretty.py** | 1026 | 1062| 378 | 56246 | 148652 | 
| 162 | 23 sympy/printing/julia.py | 219 | 261| 290 | 56536 | 148652 | 
| 163 | **23 sympy/printing/str.py** | 165 | 178| 140 | 56676 | 148652 | 
| 164 | 23 sympy/printing/codeprinter.py | 338 | 369| 257 | 56933 | 148652 | 
| 165 | **23 sympy/printing/pretty/pretty.py** | 1796 | 1840| 501 | 57434 | 148652 | 
| 166 | 24 sympy/printing/rust.py | 1 | 55| 566 | 58000 | 154127 | 
| 167 | **24 sympy/printing/pretty/pretty.py** | 825 | 840| 136 | 58136 | 154127 | 
| 168 | **24 sympy/printing/latex.py** | 1886 | 1901| 147 | 58283 | 154127 | 
| 169 | 24 sympy/printing/pycode.py | 226 | 240| 127 | 58410 | 154127 | 
| 170 | 24 sympy/printing/ccode.py | 470 | 484| 163 | 58573 | 154127 | 
| 171 | **24 sympy/printing/pretty/pretty.py** | 532 | 575| 479 | 59052 | 154127 | 
| 172 | 24 sympy/printing/fcode.py | 709 | 736| 196 | 59248 | 154127 | 
| 173 | 24 sympy/printing/fcode.py | 417 | 433| 187 | 59435 | 154127 | 
| 174 | 24 sympy/printing/rust.py | 434 | 470| 342 | 59777 | 154127 | 
| 175 | **24 sympy/printing/pretty/pretty.py** | 2034 | 2068| 311 | 60088 | 154127 | 
| 176 | 24 sympy/printing/ccode.py | 534 | 549| 182 | 60270 | 154127 | 
| 177 | **24 sympy/printing/pretty/pretty.py** | 1457 | 1515| 564 | 60834 | 154127 | 
| 178 | 24 sympy/printing/pycode.py | 604 | 641| 375 | 61209 | 154127 | 
| 179 | 25 sympy/printing/mathematica.py | 226 | 245| 173 | 61382 | 157769 | 
| 180 | **25 sympy/printing/latex.py** | 948 | 957| 140 | 61522 | 157769 | 
| **-> 181 <-** | **25 sympy/combinatorics/permutations.py** | 2801 | 2838| 322 | 61844 | 157769 | 
| 182 | 25 sympy/printing/julia.py | 292 | 329| 215 | 62059 | 157769 | 
| 183 | 25 sympy/printing/pretty/pretty_symbology.py | 302 | 331| 230 | 62289 | 157769 | 
| 184 | 26 sympy/printing/octave.py | 138 | 210| 700 | 62989 | 164511 | 
| 185 | 26 sympy/printing/ccode.py | 291 | 301| 130 | 63119 | 164511 | 
| 186 | 26 sympy/printing/fcode.py | 202 | 211| 142 | 63261 | 164511 | 
| 187 | **26 sympy/printing/pretty/pretty.py** | 2556 | 2613| 449 | 63710 | 164511 | 
| 188 | 26 sympy/printing/fcode.py | 338 | 355| 171 | 63881 | 164511 | 
| 189 | 26 sympy/printing/fcode.py | 453 | 483| 344 | 64225 | 164511 | 
| 190 | 26 sympy/printing/fcode.py | 357 | 398| 461 | 64686 | 164511 | 
| 191 | 26 sympy/printing/codeprinter.py | 247 | 300| 429 | 65115 | 164511 | 
| 192 | **26 sympy/printing/latex.py** | 2355 | 2421| 734 | 65849 | 164511 | 


## Missing Patch Files

 * 1: sympy/codegen/array_utils.py
 * 2: sympy/combinatorics/generators.py
 * 3: sympy/combinatorics/homomorphisms.py
 * 4: sympy/combinatorics/named_groups.py
 * 5: sympy/combinatorics/perm_groups.py
 * 6: sympy/combinatorics/permutations.py
 * 7: sympy/combinatorics/polyhedron.py
 * 8: sympy/combinatorics/tensor_can.py
 * 9: sympy/combinatorics/util.py
 * 10: sympy/interactive/printing.py
 * 11: sympy/printing/latex.py
 * 12: sympy/printing/pretty/pretty.py
 * 13: sympy/printing/repr.py
 * 14: sympy/printing/str.py

### Hint

```
Hi I am looking to fix this error. Could you guide me on this one a bit? 
From what I understood `permutations.py` has some functions which use `print_cyclic` flag. But since this is a global flag, it should not be used. Instead it should use something from the `printing` module? Do I need to go through the `printing` module thoroughly? How do I get started on this?
@sudz123 Users should set printing preferences via `init_printing`, not by changing class attributes. So `print_cyclic` from the `Permutation` class should be deprecated, and `init_printing` should get a new keyword argument so we could do `init_printing(cyclic=True)` (or something like that) if we wanted permutations to be printed in cyclic notation.`init_printing` is [here](https://docs.sympy.org/latest/_modules/sympy/interactive/printing.html#init_printing).

> Additionally, it would be best if the str printer printed a Python valid representation and the pretty printers only (pprint/latex) printed things like (1 2 3).

Right now, permutation printing is all specified in its `__repr__` - this should always return `Permutation(<list>)` (i.e. what's returned now when `Permutation.print_cyclic=False`). The new "cyclic" flag should only be relevant when `pprint` and `latex` are called. The printing module is huge, but you only need to work out how to make `pprint` and `latex` work properly with the new keyword argument. For example, for `pprint`, you'll need to add `'cyclic'` (or whatever you decide to call it) to `PrettyPrinter._default_settings` and write a new method `_print_Permutation`.
Is my following interpretation of Printing correct?
`init_printing()` only sets what happens in an interactive session. i.e. like jupyter notebook.
For printing in a regular python file. we have to only use `pprint()` or `print_latex()`

After I change the codes for `pprint()` and `print_latex()`. How do I test it for interactive ipython. In my jupyter notebbok, I am only able to get output for Latex Printing and not pprint using `init_printing()`.

@valglad @asmeurer  Please help.
```

## Patch

```diff
diff --git a/sympy/codegen/array_utils.py b/sympy/codegen/array_utils.py
--- a/sympy/codegen/array_utils.py
+++ b/sympy/codegen/array_utils.py
@@ -571,7 +571,6 @@ def nest_permutation(self):
         >>> from sympy.codegen.array_utils import (CodegenArrayPermuteDims, CodegenArrayTensorProduct, nest_permutation)
         >>> from sympy import MatrixSymbol
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
 
         >>> M = MatrixSymbol("M", 3, 3)
         >>> N = MatrixSymbol("N", 3, 3)
@@ -1055,7 +1054,6 @@ def parse_indexed_expression(expr, first_indices=None):
     >>> from sympy.codegen.array_utils import parse_indexed_expression
     >>> from sympy import MatrixSymbol, Sum, symbols
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
 
     >>> i, j, k, d = symbols("i j k d")
     >>> M = MatrixSymbol("M", d, d)
diff --git a/sympy/combinatorics/generators.py b/sympy/combinatorics/generators.py
--- a/sympy/combinatorics/generators.py
+++ b/sympy/combinatorics/generators.py
@@ -15,7 +15,6 @@ def symmetric(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import symmetric
     >>> list(symmetric(3))
     [(2), (1 2), (2)(0 1), (0 1 2), (0 2 1), (0 2)]
@@ -32,7 +31,6 @@ def cyclic(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import cyclic
     >>> list(cyclic(5))
     [(4), (0 1 2 3 4), (0 2 4 1 3),
@@ -57,7 +55,6 @@ def alternating(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import alternating
     >>> list(alternating(3))
     [(2), (0 1 2), (0 2 1)]
@@ -80,7 +77,6 @@ def dihedral(n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.generators import dihedral
     >>> list(dihedral(3))
     [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]
diff --git a/sympy/combinatorics/homomorphisms.py b/sympy/combinatorics/homomorphisms.py
--- a/sympy/combinatorics/homomorphisms.py
+++ b/sympy/combinatorics/homomorphisms.py
@@ -445,7 +445,6 @@ def group_isomorphism(G, H, isomorphism=True):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
     >>> from sympy.combinatorics.free_groups import free_group
     >>> from sympy.combinatorics.fp_groups import FpGroup
diff --git a/sympy/combinatorics/named_groups.py b/sympy/combinatorics/named_groups.py
--- a/sympy/combinatorics/named_groups.py
+++ b/sympy/combinatorics/named_groups.py
@@ -20,7 +20,6 @@ def AbelianGroup(*cyclic_orders):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import AbelianGroup
     >>> AbelianGroup(3, 4)
     PermutationGroup([
diff --git a/sympy/combinatorics/perm_groups.py b/sympy/combinatorics/perm_groups.py
--- a/sympy/combinatorics/perm_groups.py
+++ b/sympy/combinatorics/perm_groups.py
@@ -36,7 +36,6 @@ class PermutationGroup(Basic):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.permutations import Cycle
     >>> from sympy.combinatorics.polyhedron import Polyhedron
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
@@ -1114,7 +1113,6 @@ def coset_factor(self, g, factor_index=False):
         ========
 
         >>> from sympy.combinatorics import Permutation, PermutationGroup
-        >>> Permutation.print_cyclic = True
         >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
         >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
         >>> G = PermutationGroup([a, b])
@@ -1239,7 +1237,6 @@ def coset_rank(self, g):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
         >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
@@ -1307,7 +1304,6 @@ def degree(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 0, 2])
         >>> G = PermutationGroup([a])
@@ -1423,7 +1419,6 @@ def derived_subgroup(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 0, 2, 4, 3])
         >>> b = Permutation([0, 1, 3, 2, 4])
@@ -1471,7 +1466,6 @@ def generate(self, method="coset", af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics import PermutationGroup
         >>> from sympy.combinatorics.polyhedron import tetrahedron
 
@@ -1518,7 +1512,6 @@ def generate_dimino(self, af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1, 3])
         >>> b = Permutation([0, 2, 3, 1])
@@ -1579,7 +1572,6 @@ def generate_schreier_sims(self, af=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1, 3])
         >>> b = Permutation([0, 2, 3, 1])
@@ -1649,7 +1641,6 @@ def generators(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1])
         >>> b = Permutation([1, 0, 2])
@@ -1675,7 +1666,6 @@ def contains(self, g, strict=True):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
 
         >>> a = Permutation(1, 2)
@@ -1750,7 +1740,6 @@ def is_abelian(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([0, 2, 1])
         >>> b = Permutation([1, 0, 2])
@@ -2055,7 +2044,6 @@ def is_normal(self, gr, strict=True):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a = Permutation([1, 2, 0])
         >>> b = Permutation([1, 0, 2])
@@ -2725,7 +2713,6 @@ def orbit_rep(self, alpha, beta, schreier_vector=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import AlternatingGroup
         >>> G = AlternatingGroup(5)
@@ -2768,7 +2755,6 @@ def orbit_transversal(self, alpha, pairs=False):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import DihedralGroup
         >>> G = DihedralGroup(6)
@@ -3161,7 +3147,6 @@ def make_perm(self, n, seed=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> a, b = [Permutation([1, 0, 3, 2]), Permutation([1, 3, 0, 2])]
         >>> G = PermutationGroup([a, b])
@@ -3696,7 +3681,6 @@ def stabilizer(self, alpha):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> from sympy.combinatorics.perm_groups import PermutationGroup
         >>> from sympy.combinatorics.named_groups import DihedralGroup
         >>> G = DihedralGroup(6)
@@ -4919,7 +4903,6 @@ def _orbit_transversal(degree, generators, alpha, pairs, af=False, slp=False):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.perm_groups import _orbit_transversal
     >>> G = DihedralGroup(6)
@@ -4972,7 +4955,6 @@ def _stabilizer(degree, generators, alpha):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import _stabilizer
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> G = DihedralGroup(6)
diff --git a/sympy/combinatorics/permutations.py b/sympy/combinatorics/permutations.py
--- a/sympy/combinatorics/permutations.py
+++ b/sympy/combinatorics/permutations.py
@@ -23,7 +23,6 @@ def _af_rmul(a, b):
     ========
 
     >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-    >>> Permutation.print_cyclic = False
 
     >>> a, b = [1, 0, 2], [0, 2, 1]
     >>> _af_rmul(a, b)
@@ -57,7 +56,6 @@ def _af_rmuln(*abc):
     ========
 
     >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-    >>> Permutation.print_cyclic = False
 
     >>> a, b = [1, 0, 2], [0, 2, 1]
     >>> _af_rmul(a, b)
@@ -179,7 +177,6 @@ def _af_pow(a, n):
     ========
 
     >>> from sympy.combinatorics.permutations import Permutation, _af_pow
-    >>> Permutation.print_cyclic = False
     >>> p = Permutation([2, 0, 3, 1])
     >>> p.order()
     4
@@ -358,7 +355,6 @@ def list(self, size=None):
 
         >>> from sympy.combinatorics.permutations import Cycle
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Cycle(2, 3)(4, 5)
         >>> p.list()
         [0, 1, 3, 2, 5, 4]
@@ -479,7 +475,8 @@ class Permutation(Atom):
     original ordering, not the elements (a, b, etc...) themselves.
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = False
+    >>> from sympy.interactive import init_printing
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
 
     Permutations Notation
     =====================
@@ -662,17 +659,17 @@ class Permutation(Atom):
 
     There are a few things to note about how Permutations are printed.
 
-    1) If you prefer one form (array or cycle) over another, you can set that
-    with the print_cyclic flag.
+    1) If you prefer one form (array or cycle) over another, you can set
+    ``init_printing`` with the ``perm_cyclic`` flag.
 
-    >>> Permutation(1, 2)(4, 5)(3, 4)
+    >>> from sympy import init_printing
+    >>> p = Permutation(1, 2)(4, 5)(3, 4)
+    >>> p
     Permutation([0, 2, 1, 4, 5, 3])
-    >>> p = _
 
-    >>> Permutation.print_cyclic = True
+    >>> init_printing(perm_cyclic=True, pretty_print=False)
     >>> p
     (1 2)(3 4 5)
-    >>> Permutation.print_cyclic = False
 
     2) Regardless of the setting, a list of elements in the array for cyclic
     form can be obtained and either of those can be copied and supplied as
@@ -688,6 +685,7 @@ class Permutation(Atom):
     3) Printing is economical in that as little as possible is printed while
     retaining all information about the size of the permutation:
 
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
     >>> Permutation([1, 0, 2, 3])
     Permutation([1, 0, 2, 3])
     >>> Permutation([1, 0, 2, 3], size=20)
@@ -696,10 +694,10 @@ class Permutation(Atom):
     Permutation([1, 0, 2, 4, 3], size=20)
 
     >>> p = Permutation([1, 0, 2, 3])
-    >>> Permutation.print_cyclic = True
+    >>> init_printing(perm_cyclic=True, pretty_print=False)
     >>> p
     (3)(0 1)
-    >>> Permutation.print_cyclic = False
+    >>> init_printing(perm_cyclic=False, pretty_print=False)
 
     The 2 was not printed but it is still there as can be seen with the
     array_form and size methods:
@@ -776,7 +774,7 @@ class Permutation(Atom):
     Permutations:
 
     >>> p(['zero', 'one', 'four', 'two'])
-     ['one', 'zero', 'four', 'two']
+    ['one', 'zero', 'four', 'two']
     >>> p('zo42')
     ['o', 'z', '4', '2']
 
@@ -836,7 +834,8 @@ def __new__(cls, *args, **kwargs):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
 
         Permutations entered in array-form are left unaltered:
 
@@ -971,8 +970,9 @@ def _af_new(cls, perm):
         ========
 
         >>> from sympy.combinatorics.permutations import Perm
-        >>> Perm.print_cyclic = False
-        >>> a = [2,1,3,0]
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> a = [2, 1, 3, 0]
         >>> p = Perm._af_new(a)
         >>> p
         Permutation([2, 1, 3, 0])
@@ -996,7 +996,6 @@ def array_form(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([[2, 0], [3, 1]])
         >>> p.array_form
         [2, 3, 0, 1]
@@ -1009,29 +1008,6 @@ def array_form(self):
         """
         return self._array_form[:]
 
-    def __repr__(self):
-        if Permutation.print_cyclic:
-            if not self.size:
-                return 'Permutation()'
-            # before taking Cycle notation, see if the last element is
-            # a singleton and move it to the head of the string
-            s = Cycle(self)(self.size - 1).__repr__()[len('Cycle'):]
-            last = s.rfind('(')
-            if not last == 0 and ',' not in s[last:]:
-                s = s[last:] + s[:last]
-            return 'Permutation%s' %s
-        else:
-            s = self.support()
-            if not s:
-                if self.size < 5:
-                    return 'Permutation(%s)' % str(self.array_form)
-                return 'Permutation([], size=%s)' % self.size
-            trim = str(self.array_form[:s[-1] + 1]) + ', size=%s' % self.size
-            use = full = str(self.array_form)
-            if len(trim) < len(full):
-                use = trim
-            return 'Permutation(%s)' % use
-
     def list(self, size=None):
         """Return the permutation as an explicit list, possibly
         trimming unmoved elements if size is less than the maximum
@@ -1042,7 +1018,6 @@ def list(self, size=None):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation(2, 3)(4, 5)
         >>> p.list()
         [0, 1, 3, 2, 5, 4]
@@ -1083,7 +1058,6 @@ def cyclic_form(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([0, 3, 1, 2])
         >>> p.cyclic_form
         [[1, 3, 2]]
@@ -1179,7 +1153,6 @@ def __add__(self, other):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> I = Permutation([0, 1, 2, 3])
         >>> a = Permutation([2, 1, 3, 0])
         >>> I + a.rank() == a
@@ -1218,7 +1191,6 @@ def rmul(*args):
         ========
 
         >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-        >>> Permutation.print_cyclic = False
 
         >>> a, b = [1, 0, 2], [0, 2, 1]
         >>> a = Permutation(a); b = Permutation(b)
@@ -1283,7 +1255,6 @@ def __mul__(self, other):
         ========
 
         >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
-        >>> Permutation.print_cyclic = False
 
         >>> a, b = [1, 0, 2], [0, 2, 1]
         >>> a = Permutation(a); b = Permutation(b)
@@ -1303,6 +1274,8 @@ def __mul__(self, other):
         It is acceptable for the arrays to have different lengths; the shorter
         one will be padded to match the longer one:
 
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> b*Permutation([1, 0])
         Permutation([1, 2, 0])
         >>> Permutation([1, 0])*b
@@ -1364,8 +1337,9 @@ def __pow__(self, n):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
-        >>> p = Permutation([2,0,3,1])
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> p = Permutation([2, 0, 3, 1])
         >>> p.order()
         4
         >>> p**4
@@ -1404,7 +1378,6 @@ def __xor__(self, h):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = True
         >>> p = Permutation(1, 2, 9)
         >>> q = Permutation(6, 9, 8)
         >>> p*q != q*p
@@ -1519,7 +1492,6 @@ def from_sequence(self, i, key=None):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
 
         >>> Permutation.from_sequence('SymPy')
         (4)(0 1 3)
@@ -1545,8 +1517,9 @@ def __invert__(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
-        >>> p = Permutation([[2,0], [3,1]])
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
+        >>> p = Permutation([[2, 0], [3, 1]])
         >>> ~p
         Permutation([2, 3, 0, 1])
         >>> _ == p**-1
@@ -1569,6 +1542,10 @@ def __iter__(self):
         for i in self.array_form:
             yield i
 
+    def __repr__(self):
+        from sympy.printing.repr import srepr
+        return srepr(self)
+
     def __call__(self, *i):
         """
         Allows applying a permutation instance as a bijective function.
@@ -1676,7 +1653,8 @@ def unrank_nonlex(self, n, r):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.unrank_nonlex(4, 5)
         Permutation([2, 0, 3, 1])
         >>> Permutation.unrank_nonlex(4, -1)
@@ -1743,7 +1721,8 @@ def next_nonlex(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([2, 0, 3, 1]); p.rank_nonlex()
         5
         >>> p = p.next_nonlex(); p
@@ -2129,7 +2108,8 @@ def commutator(self, x):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([0, 2, 3, 1])
         >>> x = Permutation([2, 0, 3, 1])
         >>> c = p.commutator(x); c
@@ -2209,7 +2189,8 @@ def order(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([3, 1, 5, 2, 4, 0])
         >>> p.order()
         4
@@ -2254,7 +2235,6 @@ def cycle_structure(self):
         ========
 
         >>> from sympy.combinatorics import Permutation
-        >>> Permutation.print_cyclic = True
         >>> Permutation(3).cycle_structure
         {1: 4}
         >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure
@@ -2349,7 +2329,6 @@ def inversion_vector(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
         >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])
         >>> p.inversion_vector()
         [4, 7, 0, 5, 0, 2, 1, 1]
@@ -2364,13 +2343,12 @@ def inversion_vector(self):
         >>> while p:
         ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))
         ...     p = p.next_lex()
-        ...
-        Permutation([0, 1, 2]) [0, 0] 0
-        Permutation([0, 2, 1]) [0, 1] 1
-        Permutation([1, 0, 2]) [1, 0] 2
-        Permutation([1, 2, 0]) [1, 1] 3
-        Permutation([2, 0, 1]) [2, 0] 4
-        Permutation([2, 1, 0]) [2, 1] 5
+        (2) [0, 0] 0
+        (1 2) [0, 1] 1
+        (2)(0 1) [1, 0] 2
+        (0 1 2) [1, 1] 3
+        (0 2 1) [2, 0] 4
+        (0 2) [2, 1] 5
 
         See Also
         ========
@@ -2440,6 +2418,8 @@ def unrank_trotterjohnson(cls, size, rank):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.unrank_trotterjohnson(5, 10)
         Permutation([0, 3, 1, 2, 4])
 
@@ -2479,7 +2459,8 @@ def next_trotterjohnson(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation([3, 0, 2, 1])
         >>> p.rank_trotterjohnson()
         4
@@ -2530,7 +2511,8 @@ def get_precedence_matrix(self):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> p = Permutation.josephus(3, 6, 1)
         >>> p
         Permutation([2, 5, 3, 1, 4, 0])
@@ -2761,7 +2743,8 @@ def from_inversion_vector(cls, inversion):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])
         Permutation([3, 2, 1, 0, 4, 5])
 
@@ -2807,7 +2790,8 @@ def unrank_lex(cls, size, rank):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
+        >>> init_printing(perm_cyclic=False, pretty_print=False)
         >>> a = Permutation.unrank_lex(5, 10)
         >>> a.rank()
         10
@@ -2832,10 +2816,8 @@ def unrank_lex(cls, size, rank):
             psize = new_psize
         return cls._af_new(perm_array)
 
-    # global flag to control how permutations are printed
-    # when True, Permutation([0, 2, 1, 3]) -> Cycle(1, 2)
-    # when False, Permutation([0, 2, 1, 3]) -> Permutation([0, 2, 1])
-    print_cyclic = True
+    # XXX Deprecated flag
+    print_cyclic = None
 
 
 def _merge(arr, temp, left, mid, right):
diff --git a/sympy/combinatorics/polyhedron.py b/sympy/combinatorics/polyhedron.py
--- a/sympy/combinatorics/polyhedron.py
+++ b/sympy/combinatorics/polyhedron.py
@@ -60,8 +60,9 @@ def __new__(cls, corners, faces=[], pgroup=[]):
         ========
 
         >>> from sympy.combinatorics.permutations import Permutation
-        >>> Permutation.print_cyclic = False
+        >>> from sympy.interactive import init_printing
         >>> from sympy.abc import w, x, y, z
+        >>> init_printing(pretty_print=False, perm_cyclic=False)
 
         Here we construct the Polyhedron object for a tetrahedron.
 
diff --git a/sympy/combinatorics/tensor_can.py b/sympy/combinatorics/tensor_can.py
--- a/sympy/combinatorics/tensor_can.py
+++ b/sympy/combinatorics/tensor_can.py
@@ -918,7 +918,6 @@ def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
-    >>> Permutation.print_cyclic = True
     >>> base1, gens1 = get_symmetric_group_sgs(1)
     >>> base2, gens2 = get_symmetric_group_sgs(2)
     >>> bsgs_direct_product(base1, gens1, base2, gens2)
@@ -953,7 +952,6 @@ def get_symmetric_group_sgs(n, antisym=False):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
-    >>> Permutation.print_cyclic = True
     >>> get_symmetric_group_sgs(3)
     ([0, 1], [(4)(0 1), (4)(1 2)])
     """
@@ -1028,7 +1026,6 @@ def get_minimal_bsgs(base, gens):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
-    >>> Permutation.print_cyclic = True
     >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
     >>> get_minimal_bsgs(*riemann_bsgs1)
     ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
@@ -1059,7 +1056,6 @@ def tensor_gens(base, gens, list_free_indices, sym=0):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs
-    >>> Permutation.print_cyclic = True
 
     two symmetric tensors with 3 indices without free indices
 
@@ -1176,7 +1172,6 @@ def gens_products(*v):
 
     >>> from sympy.combinatorics import Permutation
     >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
-    >>> Permutation.print_cyclic = True
     >>> base, gens = get_symmetric_group_sgs(2)
     >>> gens_products((base, gens, [[], []], 0))
     (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
diff --git a/sympy/combinatorics/util.py b/sympy/combinatorics/util.py
--- a/sympy/combinatorics/util.py
+++ b/sympy/combinatorics/util.py
@@ -143,7 +143,6 @@ def _distribute_gens_by_base(base, gens):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.util import _distribute_gens_by_base
     >>> D = DihedralGroup(3)
@@ -211,7 +210,6 @@ def _handle_precomputed_bsgs(base, strong_gens, transversals=None,
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import DihedralGroup
     >>> from sympy.combinatorics.util import _handle_precomputed_bsgs
     >>> D = DihedralGroup(3)
@@ -271,7 +269,6 @@ def _orbits_transversals_from_bsgs(base, strong_gens_distr,
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.util import _orbits_transversals_from_bsgs
     >>> from sympy.combinatorics.util import (_orbits_transversals_from_bsgs,
@@ -415,7 +412,6 @@ def _strip(g, base, orbits, transversals):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.permutations import Permutation
     >>> from sympy.combinatorics.util import _strip
@@ -509,7 +505,6 @@ def _strong_gens_from_distr(strong_gens_distr):
     ========
 
     >>> from sympy.combinatorics import Permutation
-    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.named_groups import SymmetricGroup
     >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
     ... _distribute_gens_by_base)
diff --git a/sympy/interactive/printing.py b/sympy/interactive/printing.py
--- a/sympy/interactive/printing.py
+++ b/sympy/interactive/printing.py
@@ -550,13 +550,16 @@ def init_printing(pretty_print=True, order=None, use_unicode=None,
         _stringify_func = stringify_func
 
         if pretty_print:
-            stringify_func = lambda expr: \
+            stringify_func = lambda expr, **settings: \
                              _stringify_func(expr, order=order,
                                              use_unicode=use_unicode,
                                              wrap_line=wrap_line,
-                                             num_columns=num_columns)
+                                             num_columns=num_columns,
+                                             **settings)
         else:
-            stringify_func = lambda expr: _stringify_func(expr, order=order)
+            stringify_func = \
+                lambda expr, **settings: _stringify_func(
+                    expr, order=order, **settings)
 
     if in_ipython:
         mode_in_settings = settings.pop("mode", None)
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -141,6 +141,7 @@ class LatexPrinter(Printer):
         "imaginary_unit": "i",
         "gothic_re_im": False,
         "decimal_separator": "period",
+        "perm_cyclic": True,
     }
 
     def __init__(self, settings=None):
@@ -374,7 +375,35 @@ def _print_Cycle(self, expr):
         term_tex = term_tex.replace(']', r"\right)")
         return term_tex
 
-    _print_Permutation = _print_Cycle
+    def _print_Permutation(self, expr):
+        from sympy.combinatorics.permutations import Permutation
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            return self._print_Cycle(expr)
+
+        if expr.size == 0:
+            return r"\left( \right)"
+
+        lower = [self._print(arg) for arg in expr.array_form]
+        upper = [self._print(arg) for arg in range(len(lower))]
+
+        row1 = " & ".join(upper)
+        row2 = " & ".join(lower)
+        mat = r" \\ ".join((row1, row2))
+        return r"\begin{pmatrix} %s \end{pmatrix}" % mat
+
 
     def _print_Float(self, expr):
         # Based off of that in StrPrinter
@@ -2501,7 +2530,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
           mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
           order=None, symbol_names=None, root_notation=True,
           mat_symbol_style="plain", imaginary_unit="i", gothic_re_im=False,
-          decimal_separator="period" ):
+          decimal_separator="period", perm_cyclic=True):
     r"""Convert the given expression to LaTeX string representation.
 
     Parameters
@@ -2702,6 +2731,7 @@ def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
         'imaginary_unit': imaginary_unit,
         'gothic_re_im': gothic_re_im,
         'decimal_separator': decimal_separator,
+        'perm_cyclic' : perm_cyclic,
     }
 
     return LatexPrinter(settings).doprint(expr)
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -17,6 +17,7 @@
 from sympy.printing.str import sstr
 from sympy.utilities import default_sort_key
 from sympy.utilities.iterables import has_variety
+from sympy.utilities.exceptions import SymPyDeprecationWarning
 
 from sympy.printing.pretty.stringpict import prettyForm, stringPict
 from sympy.printing.pretty.pretty_symbology import xstr, hobj, vobj, xobj, \
@@ -42,6 +43,7 @@ class PrettyPrinter(Printer):
         "root_notation": True,
         "mat_symbol_style": "plain",
         "imaginary_unit": "i",
+        "perm_cyclic": True
     }
 
     def __init__(self, settings=None):
@@ -387,6 +389,41 @@ def _print_Cycle(self, dc):
             cyc = prettyForm(*cyc.right(l))
         return cyc
 
+    def _print_Permutation(self, expr):
+        from ..str import sstr
+        from sympy.combinatorics.permutations import Permutation, Cycle
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            return self._print_Cycle(Cycle(expr))
+
+        lower = expr.array_form
+        upper = list(range(len(lower)))
+
+        result = stringPict('')
+        first = True
+        for u, l in zip(upper, lower):
+            s1 = self._print(u)
+            s2 = self._print(l)
+            col = prettyForm(*s1.below(s2))
+            if first:
+                first = False
+            else:
+                col = prettyForm(*col.left(" "))
+            result = prettyForm(*result.right(col))
+        return prettyForm(*result.parens())
+
+
     def _print_Integral(self, integral):
         f = integral.function
 
@@ -2613,9 +2650,7 @@ def pretty(expr, **settings):
         pretty_use_unicode(uflag)
 
 
-def pretty_print(expr, wrap_line=True, num_columns=None, use_unicode=None,
-                 full_prec="auto", order=None, use_unicode_sqrt_char=True,
-                 root_notation = True, mat_symbol_style="plain", imaginary_unit="i"):
+def pretty_print(expr, **kwargs):
     """Prints expr in pretty form.
 
     pprint is just a shortcut for this function.
@@ -2658,11 +2693,7 @@ def pretty_print(expr, wrap_line=True, num_columns=None, use_unicode=None,
         Letter to use for imaginary unit when use_unicode is True.
         Can be "i" (default) or "j".
     """
-    print(pretty(expr, wrap_line=wrap_line, num_columns=num_columns,
-                 use_unicode=use_unicode, full_prec=full_prec, order=order,
-                 use_unicode_sqrt_char=use_unicode_sqrt_char,
-                 root_notation=root_notation, mat_symbol_style=mat_symbol_style,
-                 imaginary_unit=imaginary_unit))
+    print(pretty(expr, **kwargs))
 
 pprint = pretty_print
 
diff --git a/sympy/printing/repr.py b/sympy/printing/repr.py
--- a/sympy/printing/repr.py
+++ b/sympy/printing/repr.py
@@ -8,16 +8,19 @@
 from __future__ import print_function, division
 
 from sympy.core.function import AppliedUndef
-from .printer import Printer
 from mpmath.libmp import repr_dps, to_str as mlib_to_str
 from sympy.core.compatibility import range, string_types
 
+from .printer import Printer
+from .str import sstr
+
 
 class ReprPrinter(Printer):
     printmethod = "_sympyrepr"
 
     _default_settings = {
-        "order": None
+        "order": None,
+        "perm_cyclic" : True,
     }
 
     def reprify(self, args, sep):
@@ -57,7 +60,41 @@ def _print_Cycle(self, expr):
         return expr.__repr__()
 
     def _print_Permutation(self, expr):
-        return expr.__repr__()
+        from sympy.combinatorics.permutations import Permutation, Cycle
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
+            if not expr.size:
+                return 'Permutation()'
+            # before taking Cycle notation, see if the last element is
+            # a singleton and move it to the head of the string
+            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
+            last = s.rfind('(')
+            if not last == 0 and ',' not in s[last:]:
+                s = s[last:] + s[:last]
+            return 'Permutation%s' %s
+        else:
+            s = expr.support()
+            if not s:
+                if expr.size < 5:
+                    return 'Permutation(%s)' % str(expr.array_form)
+                return 'Permutation([], size=%s)' % expr.size
+            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
+            use = full = str(expr.array_form)
+            if len(trim) < len(full):
+                use = trim
+            return 'Permutation(%s)' % use
 
     def _print_Function(self, expr):
         r = self._print(expr.func)
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -22,6 +22,7 @@ class StrPrinter(Printer):
         "full_prec": "auto",
         "sympy_integers": False,
         "abbrev": False,
+        "perm_cyclic": True,
     }
 
     _relationals = dict()
@@ -354,7 +355,20 @@ def _print_Cycle(self, expr):
 
     def _print_Permutation(self, expr):
         from sympy.combinatorics.permutations import Permutation, Cycle
-        if Permutation.print_cyclic:
+        from sympy.utilities.exceptions import SymPyDeprecationWarning
+
+        perm_cyclic = Permutation.print_cyclic
+        if perm_cyclic is not None:
+            SymPyDeprecationWarning(
+                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
+                useinstead="init_printing(perm_cyclic={})"
+                .format(perm_cyclic),
+                issue=15201,
+                deprecated_since_version="1.6").warn()
+        else:
+            perm_cyclic = self._settings.get("perm_cyclic", True)
+
+        if perm_cyclic:
             if not expr.size:
                 return '()'
             # before taking Cycle notation, see if the last element is

```

## Test Patch

```diff
diff --git a/sympy/combinatorics/tests/test_permutations.py b/sympy/combinatorics/tests/test_permutations.py
--- a/sympy/combinatorics/tests/test_permutations.py
+++ b/sympy/combinatorics/tests/test_permutations.py
@@ -7,7 +7,10 @@
 from sympy.core.singleton import S
 from sympy.combinatorics.permutations import (Permutation, _af_parity,
     _af_rmul, _af_rmuln, Cycle)
-from sympy.utilities.pytest import raises
+from sympy.printing import sstr, srepr, pretty, latex
+from sympy.utilities.pytest import raises, SymPyDeprecationWarning, \
+    warns_deprecated_sympy
+
 
 rmul = Permutation.rmul
 a = Symbol('a', integer=True)
@@ -443,7 +446,6 @@ def test_from_sequence():
 
 
 def test_printing_cyclic():
-    Permutation.print_cyclic = True
     p1 = Permutation([0, 2, 1])
     assert repr(p1) == 'Permutation(1, 2)'
     assert str(p1) == '(1 2)'
@@ -455,19 +457,46 @@ def test_printing_cyclic():
 
 
 def test_printing_non_cyclic():
-    Permutation.print_cyclic = False
+    from sympy.printing import sstr, srepr
     p1 = Permutation([0, 1, 2, 3, 4, 5])
-    assert repr(p1) == 'Permutation([], size=6)'
-    assert str(p1) == 'Permutation([], size=6)'
+    assert srepr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
+    assert sstr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
     p2 = Permutation([0, 1, 2])
-    assert repr(p2) == 'Permutation([0, 1, 2])'
-    assert str(p2) == 'Permutation([0, 1, 2])'
+    assert srepr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
+    assert sstr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
 
     p3 = Permutation([0, 2, 1])
-    assert repr(p3) == 'Permutation([0, 2, 1])'
-    assert str(p3) == 'Permutation([0, 2, 1])'
+    assert srepr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
+    assert sstr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
     p4 = Permutation([0, 1, 3, 2, 4, 5, 6, 7])
-    assert repr(p4) == 'Permutation([0, 1, 3, 2], size=8)'
+    assert srepr(p4, perm_cyclic=False) == 'Permutation([0, 1, 3, 2], size=8)'
+
+
+def test_deprecated_print_cyclic():
+    p = Permutation(0, 1, 2)
+    try:
+        Permutation.print_cyclic = True
+        with warns_deprecated_sympy():
+            assert sstr(p) == '(0 1 2)'
+        with warns_deprecated_sympy():
+            assert srepr(p) == 'Permutation(0, 1, 2)'
+        with warns_deprecated_sympy():
+            assert pretty(p) == '(0 1 2)'
+        with warns_deprecated_sympy():
+            assert latex(p) == r'\left( 0\; 1\; 2\right)'
+
+        Permutation.print_cyclic = False
+        with warns_deprecated_sympy():
+            assert sstr(p) == 'Permutation([1, 2, 0])'
+        with warns_deprecated_sympy():
+            assert srepr(p) == 'Permutation([1, 2, 0])'
+        with warns_deprecated_sympy():
+            assert pretty(p, use_unicode=False) == '/0 1 2\\\n\\1 2 0/'
+        with warns_deprecated_sympy():
+            assert latex(p) == \
+                r'\begin{pmatrix} 0 & 1 & 2 \\ 1 & 2 & 0 \end{pmatrix}'
+    finally:
+        Permutation.print_cyclic = None
 
 
 def test_permutation_equality():
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -365,6 +365,18 @@ def test_pretty_Cycle():
     assert pretty(Cycle()) == '()'
 
 
+def test_pretty_Permutation():
+    from sympy.combinatorics.permutations import Permutation
+    p1 = Permutation(1, 2)(3, 4)
+    assert xpretty(p1, perm_cyclic=True, use_unicode=True) == "(1 2)(3 4)"
+    assert xpretty(p1, perm_cyclic=True, use_unicode=False) == "(1 2)(3 4)"
+    assert xpretty(p1, perm_cyclic=False, use_unicode=True) == \
+    u'⎛0 1 2 3 4⎞\n'\
+    u'⎝0 2 1 4 3⎠'
+    assert xpretty(p1, perm_cyclic=False, use_unicode=False) == \
+    "/0 1 2 3 4\\\n"\
+    "\\0 2 1 4 3/"
+
 def test_pretty_basic():
     assert pretty( -Rational(1)/2 ) == '-1/2'
     assert pretty( -Rational(13)/22 ) == \
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -198,6 +198,13 @@ def test_latex_permutation():
         r"\left( 2\; 4\right)\left( 5\right)"
     assert latex(Permutation(5)) == r"\left( 5\right)"
 
+    assert latex(Permutation(0, 1), perm_cyclic=False) == \
+        r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
+    assert latex(Permutation(0, 1)(2, 3), perm_cyclic=False) == \
+        r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
+    assert latex(Permutation(), perm_cyclic=False) == \
+        r"\left( \right)"
+
 
 def test_latex_Float():
     assert latex(Float(1.0e100)) == r"1.0 \cdot 10^{100}"
diff --git a/sympy/printing/tests/test_repr.py b/sympy/printing/tests/test_repr.py
--- a/sympy/printing/tests/test_repr.py
+++ b/sympy/printing/tests/test_repr.py
@@ -302,9 +302,4 @@ def test_Cycle():
 
 def test_Permutation():
     import_stmt = "from sympy.combinatorics import Permutation"
-    print_cyclic = Permutation.print_cyclic
-    try:
-        Permutation.print_cyclic = True
-        sT(Permutation(1, 2), "Permutation(1, 2)", import_stmt)
-    finally:
-        Permutation.print_cyclic = print_cyclic
+    sT(Permutation(1, 2), "Permutation(1, 2)", import_stmt)
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -274,9 +274,8 @@ def test_Permutation_Cycle():
         (Cycle(3, 4)(1, 2)(3, 4),
         '(1 2)(4)'),
     ]:
-        assert str(p) == s
+        assert sstr(p) == s
 
-    Permutation.print_cyclic = False
     for p, s in [
         (Permutation([]),
         'Permutation([])'),
@@ -293,9 +292,8 @@ def test_Permutation_Cycle():
         (Permutation([1, 0, 2, 3, 4, 5], size=10),
         'Permutation([1, 0], size=10)'),
     ]:
-        assert str(p) == s
+        assert sstr(p, perm_cyclic=False) == s
 
-    Permutation.print_cyclic = True
     for p, s in [
         (Permutation([]),
         '()'),
@@ -314,7 +312,7 @@ def test_Permutation_Cycle():
         (Permutation([0, 1, 3, 2, 4, 5], size=10),
         '(9)(2 3)'),
     ]:
-        assert str(p) == s
+        assert sstr(p) == s
 
 
 def test_Pi():

```


## Code snippets

### 1 - sympy/printing/str.py:

Start line: 355, End line: 378

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
### 2 - sympy/printing/pretty/pretty.py:

Start line: 371, End line: 388

```python
class PrettyPrinter(Printer):

    def _print_Cycle(self, dc):
        from sympy.combinatorics.permutations import Permutation, Cycle
        # for Empty Cycle
        if dc == Cycle():
            cyc = stringPict('')
            return prettyForm(*cyc.parens())

        dc_list = Permutation(dc.list()).cyclic_form
        # for Identity Cycle
        if dc_list == []:
            cyc = self._print(dc.size - 1)
            return prettyForm(*cyc.parens())

        cyc = stringPict('')
        for i in dc_list:
            l = self._print(str(tuple(i)).replace(',', ''))
            cyc = prettyForm(*cyc.right(l))
        return cyc
```
### 3 - sympy/printing/pretty/pretty.py:

Start line: 1, End line: 28

```python
from __future__ import print_function, division

import itertools

from sympy.core import S
from sympy.core.compatibility import range, string_types
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer
from sympy.printing.str import sstr
from sympy.utilities import default_sort_key
from sympy.utilities.iterables import has_variety

from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import xstr, hobj, vobj, xobj, \
    xsym, pretty_symbol, pretty_atom, pretty_use_unicode, greek_unicode, U, \
    pretty_try_use_unicode,  annotated

# rename for usage from outside
pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode
```
### 4 - sympy/combinatorics/permutations.py:

Start line: 383, End line: 404

```python
class Cycle(dict):

    def __repr__(self):
        """We want it to print as a Cycle, not as a dict.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> print(_)
        (1 2)
        >>> list(Cycle(1, 2).items())
        [(1, 2), (2, 1)]
        """
        if not self:
            return 'Cycle()'
        cycles = Permutation(self).cyclic_form
        s = ''.join(str(tuple(c)) for c in cycles)
        big = self.size - 1
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        return 'Cycle%s' % s
```
### 5 - sympy/combinatorics/permutations.py:

Start line: 470, End line: 827

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
    >>> Permutation.print_cyclic = False

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

    The following fails because there is is no element 3:

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

    1) If you prefer one form (array or cycle) over another, you can set that
    with the print_cyclic flag.

    >>> Permutation(1, 2)(4, 5)(3, 4)
    Permutation([0, 2, 1, 4, 5, 3])
    >>> p = _

    >>> Permutation.print_cyclic = True
    >>> p
    (1 2)(3 4 5)
    >>> Permutation.print_cyclic = False

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

    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    >>> Permutation([1, 0, 2, 3], size=20)
    Permutation([1, 0], size=20)
    >>> Permutation([1, 0, 2, 4, 3, 5, 6], size=20)
    Permutation([1, 0, 2, 4, 3], size=20)

    >>> p = Permutation([1, 0, 2, 3])
    >>> Permutation.print_cyclic = True
    >>> p
    (3)(0 1)
    >>> Permutation.print_cyclic = False

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
### 6 - sympy/printing/pretty/pretty.py:

Start line: 2431, End line: 2463

```python
class PrettyPrinter(Printer):

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(pretty_symbol(object.name))

    def _print_Morphism(self, morphism):
        arrow = xsym("-->")

        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        tail = domain.right(arrow, codomain)[0]

        return prettyForm(tail)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(pretty_symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(":", pretty_morphism)[0])

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(
            NamedMorphism(morphism.domain, morphism.codomain, "id"))
```
### 7 - sympy/printing/str.py:

Start line: 804, End line: 818

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
### 8 - sympy/printing/str.py:

Start line: 719, End line: 802

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
### 9 - sympy/combinatorics/permutations.py:

Start line: 1012, End line: 1033

```python
class Permutation(Atom):

    def __repr__(self):
        if Permutation.print_cyclic:
            if not self.size:
                return 'Permutation()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(self)(self.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' %s
        else:
            s = self.support()
            if not s:
                if self.size < 5:
                    return 'Permutation(%s)' % str(self.array_form)
                return 'Permutation([], size=%s)' % self.size
            trim = str(self.array_form[:s[-1] + 1]) + ', size=%s' % self.size
            use = full = str(self.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use
```
### 10 - sympy/printing/pretty/pretty.py:

Start line: 266, End line: 332

```python
class PrettyPrinter(Printer):

    def _print_And(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, u"\N{LOGICAL AND}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Or(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, u"\N{LOGICAL OR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Xor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, u"\N{XOR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Nand(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, u"\N{NAND}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Nor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, u"\N{NOR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Implies(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or u"\N{RIGHTWARDS ARROW}", sort=False)
        else:
            return self._print_Function(e)

    def _print_Equivalent(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or u"\N{LEFT RIGHT DOUBLE ARROW}")
        else:
            return self._print_Function(e, sort=True)

    def _print_conjugate(self, e):
        pform = self._print(e.args[0])
        return prettyForm( *pform.above( hobj('_', pform.width())) )

    def _print_Abs(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens('|', '|'))
        return pform
    _print_Determinant = _print_Abs

    def _print_floor(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lfloor', 'rfloor'))
            return pform
        else:
            return self._print_Function(e)

    def _print_ceiling(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lceil', 'rceil'))
            return pform
        else:
            return self._print_Function(e)
```
### 12 - sympy/printing/pretty/pretty.py:

Start line: 1401, End line: 1433

```python
class PrettyPrinter(Printer):

    def _helper_print_function(self, func, args, sort=False, func_name=None, delimiter=', ', elementwise=False):
        if sort:
            args = sorted(args, key=default_sort_key)

        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        if func_name:
            prettyFunc = self._print(Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens())

        if elementwise:
            if self._use_unicode:
                circ = pretty_atom('Modifier Letter Low Ring')
            else:
                circ = '.'
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE,
                *stringPict.next(prettyFunc, circ)
            )

        prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens())

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform
```
### 13 - sympy/printing/pretty/pretty.py:

Start line: 2189, End line: 2197

```python
class PrettyPrinter(Printer):

    def _print_frozenset(self, s):
        if not s:
            return prettyForm('frozenset()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        pretty = prettyForm(*pretty.parens('(', ')', ifascii_nougly=True))
        pretty = prettyForm(*stringPict.next(type(s).__name__, pretty))
        return pretty
```
### 14 - sympy/printing/pretty/pretty.py:

Start line: 228, End line: 243

```python
class PrettyPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if self._use_unicode:
            arg = e.args[0]
            pform = self._print(arg)
            if isinstance(arg, Equivalent):
                return self._print_Equivalent(arg, altchar=u"\N{LEFT RIGHT DOUBLE ARROW WITH STROKE}")
            if isinstance(arg, Implies):
                return self._print_Implies(arg, altchar=u"\N{RIGHTWARDS ARROW WITH STROKE}")

            if arg.is_Boolean and not arg.is_Not:
                pform = prettyForm(*pform.parens())

            return prettyForm(*pform.left(u"\N{NOT SIGN}"))
        else:
            return self._print_Function(e)
```
### 15 - sympy/printing/str.py:

Start line: 380, End line: 436

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
### 17 - sympy/combinatorics/permutations.py:

Start line: 1076, End line: 1118

```python
class Permutation(Atom):

    @property
    def cyclic_form(self):
        """
        This is used to convert to the cyclic notation
        from the canonical notation. Singletons are omitted.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = False
        >>> p = Permutation([0, 3, 1, 2])
        >>> p.cyclic_form
        [[1, 3, 2]]
        >>> Permutation([1, 0, 2, 4, 3, 5]).cyclic_form
        [[0, 1], [3, 4]]

        See Also
        ========

        array_form, full_cyclic_form
        """
        if self._cyclic_form is not None:
            return list(self._cyclic_form)
        array_form = self.array_form
        unchecked = [True] * len(array_form)
        cyclic_form = []
        for i in range(len(array_form)):
            if unchecked[i]:
                cycle = []
                cycle.append(i)
                unchecked[i] = False
                j = i
                while unchecked[array_form[j]]:
                    j = array_form[j]
                    cycle.append(j)
                    unchecked[j] = False
                if len(cycle) > 1:
                    cyclic_form.append(cycle)
                    assert cycle == list(minlex(cycle, is_set=True))
        cyclic_form.sort()
        self._cyclic_form = cyclic_form[:]
        return cyclic_form
```
### 18 - sympy/printing/pretty/pretty.py:

Start line: 1588, End line: 1607

```python
class PrettyPrinter(Printer):

    def _print_expint(self, e):
        from sympy import Function
        if e.args[0].is_Integer and self._use_unicode:
            return self._print_Function(Function('E_%s' % e.args[0])(e.args[1]))
        return self._print_Function(e)

    def _print_Chi(self, e):
        # This needs a special case since otherwise it comes out as greek
        # letter chi...
        prettyFunc = prettyForm("Chi")
        prettyArgs = prettyForm(*self._print_seq(e.args).parens())

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform
```
### 20 - sympy/combinatorics/permutations.py:

Start line: 406, End line: 427

```python
class Cycle(dict):

    def __str__(self):
        """We want it to be printed in a Cycle notation with no
        comma in-between.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> Cycle(1, 2, 4)(5, 6)
        (1 2 4)(5 6)
        """
        if not self:
            return '()'
        cycles = Permutation(self).cyclic_form
        s = ''.join(str(tuple(c)) for c in cycles)
        big = self.size - 1
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        s = s.replace(',', '')
        return s
```
### 22 - sympy/printing/pretty/pretty.py:

Start line: 135, End line: 202

```python
class PrettyPrinter(Printer):

    def _print_Gradient(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Laplacian(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('INCREMENT'))))
        return pform

    def _print_Atom(self, e):
        try:
            # print atoms like Exp1 or Pi
            return prettyForm(pretty_atom(e.__class__.__name__, printer=self))
        except KeyError:
            return self.emptyPrinter(e)

    # Infinity inherits from Number, so we have to override _print_XXX order
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom

    _print_EmptySequence = _print_Atom

    def _print_Reals(self, e):
        if self._use_unicode:
            return self._print_Atom(e)
        else:
            inf_list = ['-oo', 'oo']
            return self._print_seq(inf_list, '(', ')')

    def _print_subfactorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('!'))
        return pform

    def _print_factorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!'))
        return pform

    def _print_factorial2(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!!'))
        return pform
```
### 23 - sympy/printing/pretty/pretty.py:

Start line: 1935, End line: 1988

```python
class PrettyPrinter(Printer):

    def _print_Interval(self, i):
        if i.start == i.end:
            return self._print_seq(i.args[:1], '{', '}')

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return self._print_seq(i.args[:2], left, right)

    def _print_AccumulationBounds(self, i):
        left = '<'
        right = '>'

        return self._print_seq(i.args[:2], left, right)

    def _print_Intersection(self, u):

        delimiter = ' %s ' % pretty_atom('Intersection', 'n')

        return self._print_seq(u.args, None, None, delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Union or set.is_Complement)

    def _print_Union(self, u):

        union_delimiter = ' %s ' % pretty_atom('Union', 'U')

        return self._print_seq(u.args, None, None, union_delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Intersection or set.is_Complement)

    def _print_SymmetricDifference(self, u):
        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of SymmetricDifference is not implemented")

        sym_delimeter = ' %s ' % pretty_atom('SymmetricDifference')

        return self._print_seq(u.args, None, None, sym_delimeter)

    def _print_Complement(self, u):

        delimiter = r' \ '

        return self._print_seq(u.args, None, None, delimiter,
             parenthesize=lambda set: set.is_ProductSet or set.is_Intersection
                               or set.is_Union)
```
### 24 - sympy/printing/pretty/pretty.py:

Start line: 1906, End line: 1933

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
### 25 - sympy/printing/str.py:

Start line: 316, End line: 353

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
        return "{0}.({1})".format(
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
### 27 - sympy/printing/pretty/pretty.py:

Start line: 245, End line: 264

```python
class PrettyPrinter(Printer):

    def __print_Boolean(self, e, char, sort=True):
        args = e.args
        if sort:
            args = sorted(e.args, key=default_sort_key)
        arg = args[0]
        pform = self._print(arg)

        if arg.is_Boolean and not arg.is_Not:
            pform = prettyForm(*pform.parens())

        for arg in args[1:]:
            pform_arg = self._print(arg)

            if arg.is_Boolean and not arg.is_Not:
                pform_arg = prettyForm(*pform_arg.parens())

            pform = prettyForm(*pform.right(u' %s ' % char))
            pform = prettyForm(*pform.right(pform_arg))

        return pform
```
### 28 - sympy/printing/pretty/pretty.py:

Start line: 2413, End line: 2429

```python
class PrettyPrinter(Printer):

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.as_boolean())))
            return pform
        elif hasattr(d, 'set'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            pform = prettyForm(*pform.right(self._print(' in ')))
            pform = prettyForm(*pform.right(self._print(d.set)))
            return pform
        elif hasattr(d, 'symbols'):
            pform = self._print('Domain on ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            return pform
        else:
            return self._print(None)
```
### 30 - sympy/printing/str.py:

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
### 31 - sympy/printing/pretty/pretty.py:

Start line: 629, End line: 655

```python
class PrettyPrinter(Printer):

    def _print_Limit(self, l):
        e, z, z0, dir = l.args

        E = self._print(e)
        if precedence(e) <= PRECEDENCE["Mul"]:
            E = prettyForm(*E.parens('(', ')'))
        Lim = prettyForm('lim')

        LimArg = self._print(z)
        if self._use_unicode:
            LimArg = prettyForm(*LimArg.right(u'\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{RIGHTWARDS ARROW}'))
        else:
            LimArg = prettyForm(*LimArg.right('->'))
        LimArg = prettyForm(*LimArg.right(self._print(z0)))

        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            dir = ""
        else:
            if self._use_unicode:
                dir = u'\N{SUPERSCRIPT PLUS SIGN}' if str(dir) == "+" else u'\N{SUPERSCRIPT MINUS}'

        LimArg = prettyForm(*LimArg.right(self._print(dir)))

        Lim = prettyForm(*Lim.below(LimArg))
        Lim = prettyForm(*Lim.right(E), binding=prettyForm.MUL)

        return Lim
```
### 32 - sympy/printing/str.py:

Start line: 1, End line: 15

```python
"""
A Printer for generating readable representation of most sympy classes.
"""

from __future__ import print_function, division

from sympy.core import S, Rational, Pow, Basic, Mul
from sympy.core.mul import _keep_coeff
from sympy.core.compatibility import string_types
from .printer import Printer
from sympy.printing.precedence import precedence, PRECEDENCE

from mpmath.libmp import prec_to_dps, to_str as mlib_to_str

from sympy.utilities import default_sort_key
```
### 34 - sympy/printing/pretty/pretty.py:

Start line: 2374, End line: 2398

```python
class PrettyPrinter(Printer):

    def _print_euler(self, e):
        return self._print_number_function(e, "E")

    def _print_catalan(self, e):
        return self._print_number_function(e, "C")

    def _print_bernoulli(self, e):
        return self._print_number_function(e, "B")

    _print_bell = _print_bernoulli

    def _print_lucas(self, e):
        return self._print_number_function(e, "L")

    def _print_fibonacci(self, e):
        return self._print_number_function(e, "F")

    def _print_tribonacci(self, e):
        return self._print_number_function(e, "T")

    def _print_stieltjes(self, e):
        if self._use_unicode:
            return self._print_number_function(e, u'\N{GREEK SMALL LETTER GAMMA}')
        else:
            return self._print_number_function(e, "stieltjes")
```
### 36 - sympy/printing/str.py:

Start line: 260, End line: 314

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
### 38 - sympy/printing/str.py:

Start line: 438, End line: 511

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
### 39 - sympy/printing/pretty/pretty.py:

Start line: 2137, End line: 2187

```python
class PrettyPrinter(Printer):

    def join(self, delimiter, args):
        pform = None

        for arg in args:
            if pform is None:
                pform = arg
            else:
                pform = prettyForm(*pform.right(delimiter))
                pform = prettyForm(*pform.right(arg))

        if pform is None:
            return prettyForm("")
        else:
            return pform

    def _print_list(self, l):
        return self._print_seq(l, '[', ']')

    def _print_tuple(self, t):
        if len(t) == 1:
            ptuple = prettyForm(*stringPict.next(self._print(t[0]), ','))
            return prettyForm(*ptuple.parens('(', ')', ifascii_nougly=True))
        else:
            return self._print_seq(t, '(', ')')

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for k in keys:
            K = self._print(k)
            V = self._print(d[k])
            s = prettyForm(*stringPict.next(K, ': ', V))

            items.append(s)

        return self._print_seq(items, '{', '}')

    def _print_Dict(self, d):
        return self._print_dict(d)

    def _print_set(self, s):
        if not s:
            return prettyForm('set()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        return pretty
```
### 41 - sympy/printing/str.py:

Start line: 199, End line: 241

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
### 43 - sympy/printing/pretty/pretty.py:

Start line: 105, End line: 121

```python
class PrettyPrinter(Printer):

    def _print_Curl(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Divergence(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform
```
### 45 - sympy/printing/repr.py:

Start line: 47, End line: 107

```python
class ReprPrinter(Printer):

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        nargs = len(args)
        args = map(self._print, args)
        clsname = type(expr).__name__
        if nargs > 255:  # Issue #10259, Python < 3.7
            return clsname + "(*[%s])" % ", ".join(args)
        return clsname + "(%s)" % ", ".join(args)

    def _print_Cycle(self, expr):
        return expr.__repr__()

    def _print_Permutation(self, expr):
        return expr.__repr__()

    def _print_Function(self, expr):
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.args])
        return r

    def _print_FunctionClass(self, expr):
        if issubclass(expr, AppliedUndef):
            return 'Function(%r)' % (expr.__name__)
        else:
            return expr.__name__

    def _print_Half(self, expr):
        return 'Rational(1, 2)'

    def _print_RationalConstant(self, expr):
        return str(expr)

    def _print_AtomicExpr(self, expr):
        return str(expr)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Integer(self, expr):
        return 'Integer(%i)' % expr.p

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_list(self, expr):
        return "[%s]" % self.reprify(expr, ", ")
```
### 46 - sympy/printing/pretty/pretty.py:

Start line: 2095, End line: 2135

```python
class PrettyPrinter(Printer):

    def _print_seq(self, seq, left=None, right=None, delimiter=', ',
            parenthesize=lambda x: False):
        s = None
        try:
            for item in seq:
                pform = self._print(item)

                if parenthesize(item):
                    pform = prettyForm(*pform.parens())
                if s is None:
                    # first element
                    s = pform
                else:
                    # XXX: Under the tests from #15686 this raises:
                    # AttributeError: 'Fake' object has no attribute 'baseline'
                    # This is caught below but that is not the right way to
                    # fix it.
                    s = prettyForm(*stringPict.next(s, delimiter))
                    s = prettyForm(*stringPict.next(s, pform))

            if s is None:
                s = stringPict('')

        except AttributeError:
            s = None
            for item in seq:
                pform = self.doprint(item)
                if parenthesize(item):
                    pform = prettyForm(*pform.parens())
                if s is None:
                    # first element
                    s = pform
                else :
                    s = prettyForm(*stringPict.next(s, delimiter))
                    s = prettyForm(*stringPict.next(s, pform))

            if s is None:
                s = stringPict('')

        s = prettyForm(*s.parens(left, right, ifascii_nougly=True))
        return s
```
### 47 - sympy/printing/pretty/pretty.py:

Start line: 2006, End line: 2032

```python
class PrettyPrinter(Printer):

    def _print_ConditionSet(self, ts):
        if self._use_unicode:
            inn = u"\N{SMALL ELEMENT OF}"
            # using _and because and is a keyword and it is bad practice to
            # overwrite them
            _and = u"\N{LOGICAL AND}"
        else:
            inn = 'in'
            _and = 'and'

        variables = self._print_seq(Tuple(ts.sym))
        as_expr = getattr(ts.condition, 'as_expr', None)
        if as_expr is not None:
            cond = self._print(ts.condition.as_expr())
        else:
            cond = self._print(ts.condition)
            if self._use_unicode:
                cond = self._print_seq(cond, "(", ")")

        bar = self._print("|")

        if ts.base_set is S.UniversalSet:
            return self._print_seq((variables, bar, cond), "{", "}", ' ')

        base = self._print(ts.base_set)
        return self._print_seq((variables, bar, variables, inn,
                                base, _and, cond), "{", "}", ' ')
```
### 48 - sympy/printing/pretty/pretty.py:

Start line: 2337, End line: 2352

```python
class PrettyPrinter(Printer):

    def _print_Subs(self, e):
        pform = self._print(e.expr)
        pform = prettyForm(*pform.parens())

        h = pform.height() if pform.height() > 1 else 2
        rvert = stringPict(vobj('|', h), baseline=pform.baseline)
        pform = prettyForm(*pform.right(rvert))

        b = pform.baseline
        pform.baseline = pform.height() - 1
        pform = prettyForm(*pform.right(self._print_seq([
            self._print_seq((self._print(v[0]), xsym('=='), self._print(v[1])),
                delimiter='') for v in zip(e.variables, e.point) ])))

        pform.baseline = b
        return pform
```
### 49 - sympy/printing/pretty/pretty.py:

Start line: 2465, End line: 2478

```python
class PrettyPrinter(Printer):

    def _print_CompositeMorphism(self, morphism):

        circle = xsym(".")

        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [pretty_symbol(component.name) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = circle.join(component_names_list) + ":"

        pretty_name = self._print(component_names)
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(pretty_morphism)[0])
```
### 50 - sympy/printing/str.py:

Start line: 820, End line: 851

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
### 51 - sympy/printing/str.py:

Start line: 70, End line: 163

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

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is special -- it's base is tuple
        return str(expr)

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'
```
### 52 - sympy/printing/latex.py:

Start line: 361, End line: 377

```python
class LatexPrinter(Printer):

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    _print_Permutation = _print_Cycle
```
### 55 - sympy/printing/pretty/pretty.py:

Start line: 2199, End line: 2307

```python
class PrettyPrinter(Printer):

    def _print_UniversalSet(self, s):
        if self._use_unicode:
            return prettyForm(u"\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL U}")
        else:
            return prettyForm('UniversalSet')

    def _print_PolyRing(self, ring):
        return prettyForm(sstr(ring))

    def _print_FracField(self, field):
        return prettyForm(sstr(field))

    def _print_FreeGroupElement(self, elm):
        return prettyForm(str(elm))

    def _print_PolyElement(self, poly):
        return prettyForm(sstr(poly))

    def _print_FracElement(self, frac):
        return prettyForm(sstr(frac))

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_ComplexRootOf(self, expr):
        args = [self._print_Add(expr.expr, order='lex'), expr.index]
        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('CRootOf'))
        return pform

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('RootSum'))

        return pform

    def _print_FiniteField(self, expr):
        if self._use_unicode:
            form = u'\N{DOUBLE-STRUCK CAPITAL Z}_%d'
        else:
            form = 'GF(%d)'

        return prettyForm(pretty_symbol(form % expr.mod))

    def _print_IntegerRing(self, expr):
        if self._use_unicode:
            return prettyForm(u'\N{DOUBLE-STRUCK CAPITAL Z}')
        else:
            return prettyForm('ZZ')

    def _print_RationalField(self, expr):
        if self._use_unicode:
            return prettyForm(u'\N{DOUBLE-STRUCK CAPITAL Q}')
        else:
            return prettyForm('QQ')

    def _print_RealField(self, domain):
        if self._use_unicode:
            prefix = u'\N{DOUBLE-STRUCK CAPITAL R}'
        else:
            prefix = 'RR'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_ComplexField(self, domain):
        if self._use_unicode:
            prefix = u'\N{DOUBLE-STRUCK CAPITAL C}'
        else:
            prefix = 'CC'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_PolynomialRing(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_FractionField(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '(', ')')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform
```
### 56 - sympy/printing/pretty/pretty.py:

Start line: 1378, End line: 1399

```python
class PrettyPrinter(Printer):

    def _print_ExpBase(self, e):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        base = prettyForm(pretty_atom('Exp1', 'e'))
        return base ** self._print(e.args[0])

    def _print_Function(self, e, sort=False, func_name=None):
        # optional argument func_name for supplying custom names
        # XXX works only for applied functions
        return self._helper_print_function(e.func, e.args, sort=sort, func_name=func_name)

    def _print_mathieuc(self, e):
        return self._print_Function(e, func_name='C')

    def _print_mathieus(self, e):
        return self._print_Function(e, func_name='S')

    def _print_mathieucprime(self, e):
        return self._print_Function(e, func_name="C'")

    def _print_mathieusprime(self, e):
        return self._print_Function(e, func_name="S'")
```
### 58 - sympy/printing/pretty/pretty.py:

Start line: 93, End line: 103

```python
class PrettyPrinter(Printer):

    def _print_Cross(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform
```
### 59 - sympy/printing/pretty/pretty.py:

Start line: 2480, End line: 2544

```python
class PrettyPrinter(Printer):

    def _print_Category(self, category):
        return self._print(pretty_symbol(category.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        pretty_result = self._print(diagram.premises)
        if diagram.conclusions:
            results_arrow = " %s " % xsym("==>")

            pretty_conclusions = self._print(diagram.conclusions)[0]
            pretty_result = pretty_result.right(
                results_arrow, pretty_conclusions)

        return prettyForm(pretty_result[0])

    def _print_DiagramGrid(self, grid):
        from sympy.matrices import Matrix
        from sympy import Symbol
        matrix = Matrix([[grid[i, j] if grid[i, j] else Symbol(" ")
                          for j in range(grid.width)]
                         for i in range(grid.height)])
        return self._print_matrix_contents(matrix)

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return self._print_seq(m, '[', ']')

    def _print_SubModule(self, M):
        return self._print_seq(M.gens, '<', '>')

    def _print_FreeModule(self, M):
        return self._print(M.ring)**self._print(M.rank)

    def _print_ModuleImplementedIdeal(self, M):
        return self._print_seq([x for [x] in M._module.gens], '<', '>')

    def _print_QuotientRing(self, R):
        return self._print(R.ring) / self._print(R.base_ideal)

    def _print_QuotientRingElement(self, R):
        return self._print(R.data) + self._print(R.ring.base_ideal)

    def _print_QuotientModuleElement(self, m):
        return self._print(m.data) + self._print(m.module.killed_module)

    def _print_QuotientModule(self, M):
        return self._print(M.base) / self._print(M.killed_module)

    def _print_MatrixHomomorphism(self, h):
        matrix = self._print(h._sympy_matrix())
        matrix.baseline = matrix.height() // 2
        pform = prettyForm(*matrix.right(' : ', self._print(h.domain),
            ' %s> ' % hobj('-', 2), self._print(h.codomain)))
        return pform

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return self._print(pretty_symbol(string))

    def _print_BaseVectorField(self, field):
        s = U('PARTIAL DIFFERENTIAL') + '_' + field._coord_sys._names[field._index]
        return self._print(pretty_symbol(s))
```
### 60 - sympy/printing/pretty/pretty.py:

Start line: 1880, End line: 1904

```python
class PrettyPrinter(Printer):

    def _print_Rational(self, expr):
        result = self.__print_numer_denom(expr.p, expr.q)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_Fraction(self, expr):
        result = self.__print_numer_denom(expr.numerator, expr.denominator)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_ProductSet(self, p):
        if len(p.sets) >= 1 and not has_variety(p.sets):
            from sympy import Pow
            return self._print(Pow(p.sets[0], len(p.sets), evaluate=False))
        else:
            prod_char = u"\N{MULTIPLICATION SIGN}" if self._use_unicode else 'x'
            return self._print_seq(p.sets, None, None, ' %s ' % prod_char,
                                   parenthesize=lambda set: set.is_Union or
                                   set.is_Intersection or set.is_ProductSet)
```
### 61 - sympy/printing/str.py:

Start line: 575, End line: 648

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
### 63 - sympy/printing/pretty/pretty.py:

Start line: 2354, End line: 2372

```python
class PrettyPrinter(Printer):

    def _print_number_function(self, e, name):
        # Print name_arg[0] for one argument or name_arg[0](arg[1])
        # for more than one argument
        pform = prettyForm(name)
        arg = self._print(e.args[0])
        pform_arg = prettyForm(" "*arg.width())
        pform_arg = prettyForm(*pform_arg.below(arg))
        pform = prettyForm(*pform.right(pform_arg))
        if len(e.args) == 1:
            return pform
        m, x = e.args
        # TODO: copy-pasted from _print_Function: can we do better?
        prettyFunc = pform
        prettyArgs = prettyForm(*self._print_seq([x]).parens())
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        return pform
```
### 64 - sympy/printing/pretty/pretty.py:

Start line: 466, End line: 518

```python
class PrettyPrinter(Printer):

    def _print_Product(self, expr):
        func = expr.term
        pretty_func = self._print(func)

        horizontal_chr = xobj('_', 1)
        corner_chr = xobj('_', 1)
        vertical_chr = xobj('|', 1)

        if self._use_unicode:
            # use unicode corners
            horizontal_chr = xobj('-', 1)
            corner_chr = u'\N{BOX DRAWINGS LIGHT DOWN AND HORIZONTAL}'

        func_height = pretty_func.height()

        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            pretty_lower, pretty_upper = self.__print_SumProduct_Limits(lim)

            width = (func_height + 2) * 5 // 3 - 2
            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width-2)) + corner_chr + horizontal_chr]
            for _ in range(func_height + 1):
                sign_lines.append(' ' + vertical_chr + (' ' * (width-2)) + vertical_chr + ' ')

            pretty_sign = stringPict('')
            pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))


            max_upper = max(max_upper, pretty_upper.height())

            if first:
                sign_height = pretty_sign.height()

            pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
            pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))

            if first:
                pretty_func.baseline = 0
                first = False

            height = pretty_sign.height()
            padding = stringPict('')
            padding = prettyForm(*padding.stack(*[' ']*(height - 1)))
            pretty_sign = prettyForm(*pretty_sign.right(padding))

            pretty_func = prettyForm(*pretty_sign.right(pretty_func))

        pretty_func.baseline = max_upper + sign_height//2
        pretty_func.binding = prettyForm.MUL
        return pretty_func
```
### 66 - sympy/printing/pretty/pretty.py:

Start line: 1246, End line: 1292

```python
class PrettyPrinter(Printer):

    def _print_hyper(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        ap = [self._print(a) for a in e.ap]
        bq = [self._print(b) for b in e.bq]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        # Drawing result - first create the ap, bq vectors
        D = None
        for v in [ap, bq]:
            D_row = self._hprint_vec(v)
            if D is None:
                D = D_row       # first row in a picture
            else:
                D = prettyForm(*D.below(' '))
                D = prettyForm(*D.below(D_row))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the F symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('F')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)
        add = (sz + 1)//2

        F = prettyForm(*F.left(self._print(len(e.ap))))
        F = prettyForm(*F.right(self._print(len(e.bq))))
        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D
```
### 67 - sympy/printing/pretty/pretty.py:

Start line: 1553, End line: 1567

```python
class PrettyPrinter(Printer):

    def _print_beta(self, e):
        func_name = greek_unicode['Beta'] if self._use_unicode else 'B'
        return self._print_Function(e, func_name=func_name)

    def _print_gamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_uppergamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_lowergamma(self, e):
        func_name = greek_unicode['gamma'] if self._use_unicode else 'lowergamma'
        return self._print_Function(e, func_name=func_name)
```
### 70 - sympy/printing/pretty/pretty.py:

Start line: 896, End line: 910

```python
class PrettyPrinter(Printer):

    def _print_HadamardPower(self, expr):
        # from sympy import MatAdd, MatMul
        if self._use_unicode:
            circ = pretty_atom('Ring')
        else:
            circ = self._print('.')
        pretty_base = self._print(expr.base)
        pretty_exp = self._print(expr.exp)
        if precedence(expr.exp) < PRECEDENCE["Mul"]:
            pretty_exp = prettyForm(*pretty_exp.parens())
        pretty_circ_exp = prettyForm(
            binding=prettyForm.LINE,
            *stringPict.next(circ, pretty_exp)
        )
        return pretty_base**pretty_circ_exp
```
### 72 - sympy/printing/pretty/pretty.py:

Start line: 204, End line: 226

```python
class PrettyPrinter(Printer):

    def _print_binomial(self, e):
        n, k = e.args

        n_pform = self._print(n)
        k_pform = self._print(k)

        bar = ' '*max(n_pform.width(), k_pform.width())

        pform = prettyForm(*k_pform.above(bar))
        pform = prettyForm(*pform.above(n_pform))
        pform = prettyForm(*pform.parens('(', ')'))

        pform.baseline = (pform.baseline + 1)//2

        return pform

    def _print_Relational(self, e):
        op = prettyForm(' ' + xsym(e.rel_op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r))
        return pform
```
### 74 - sympy/printing/pretty/pretty.py:

Start line: 1166, End line: 1220

```python
class PrettyPrinter(Printer):

    def _print_Piecewise(self, pexpr):

        P = {}
        for n, ec in enumerate(pexpr.args):
            P[n, 0] = self._print(ec.expr)
            if ec.cond == True:
                P[n, 1] = prettyForm('otherwise')
            else:
                P[n, 1] = prettyForm(
                    *prettyForm('for ').right(self._print(ec.cond)))
        hsep = 2
        vsep = 1
        len_args = len(pexpr.args)

        # max widths
        maxw = [max([P[i, j].width() for i in range(len_args)])
                for j in range(2)]

        # FIXME: Refactor this code and matrix into some tabular environment.
        # drawing result
        D = None

        for i in range(len_args):
            D_row = None
            for j in range(2):
                p = P[i, j]
                assert p.width() <= maxw[j]

                wdelta = maxw[j] - p.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                p = prettyForm(*p.right(' '*wright))
                p = prettyForm(*p.left(' '*wleft))

                if D_row is None:
                    D_row = p
                    continue

                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                D_row = prettyForm(*D_row.right(p))
            if D is None:
                D = D_row       # first row in a picture
                continue

            # v-spacer
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            D = prettyForm(*D.below(D_row))

        D = prettyForm(*D.parens('{', ''))
        D.baseline = D.height()//2
        D.binding = prettyForm.OPEN
        return D
```
### 75 - sympy/printing/pretty/pretty.py:

Start line: 1696, End line: 1737

```python
class PrettyPrinter(Printer):

    def _print_Add(self, expr, order=None):
        # ... other code

        for i, term in enumerate(terms):
            if term.is_Mul and _coeff_isneg(term):
                coeff, other = term.as_coeff_mul(rational=False)
                pform = self._print(Mul(-coeff, *other, evaluate=False))
                pforms.append(pretty_negative(pform, i))
            elif term.is_Rational and term.q > 1:
                pforms.append(None)
                indices.append(i)
            elif term.is_Number and term < 0:
                pform = self._print(-term)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Relational:
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                pforms.append(self._print(term))

        if indices:
            large = True

            for pform in pforms:
                if pform is not None and pform.height() > 1:
                    break
            else:
                large = False

            for i in indices:
                term, negative = terms[i], False

                if term < 0:
                    term, negative = -term, True

                if large:
                    pform = prettyForm(str(term.p))/prettyForm(str(term.q))
                else:
                    pform = self._print(term)

                if negative:
                    pform = pretty_negative(pform, i)

                pforms[i] = pform

        return prettyForm.__add__(*pforms)
```
### 78 - sympy/printing/pretty/pretty.py:

Start line: 1435, End line: 1455

```python
class PrettyPrinter(Printer):

    def _print_ElementwiseApplyFunction(self, e):
        func = e.function
        arg = e.expr
        args = [arg]
        return self._helper_print_function(func, args, delimiter="", elementwise=True)

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.zeta_functions import lerchphi
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: [greek_unicode['delta'], 'delta'],
                gamma: [greek_unicode['Gamma'], 'Gamma'],
                lerchphi: [greek_unicode['Phi'], 'lerchphi'],
                lowergamma: [greek_unicode['gamma'], 'gamma'],
                beta: [greek_unicode['Beta'], 'B'],
                DiracDelta: [greek_unicode['delta'], 'delta'],
                Chi: ['Chi', 'Chi']}
```
### 81 - sympy/printing/pretty/pretty.py:

Start line: 2070, End line: 2093

```python
class PrettyPrinter(Printer):

    def _print_SeqFormula(self, s):
        if self._use_unicode:
            dots = u"\N{HORIZONTAL ELLIPSIS}"
        else:
            dots = '...'

        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            raise NotImplementedError("Pretty printing of sequences with symbolic bound not implemented")

        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
            printset = tuple(printset)
        else:
            printset = tuple(s)
        return self._print_list(printset)

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula
```
### 82 - sympy/printing/pretty/pretty.py:

Start line: 1005, End line: 1024

```python
class PrettyPrinter(Printer):

    def _print_BasisDependent(self, expr):
        # ... other code

        for i, parts in enumerate(o1):
            lengths.append(len(parts[flag[i]]))
            for j in range(n_newlines):
                if j+1 <= len(parts):
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    if j == flag[i]:
                        strs[flag[i]] += parts[flag[i]] + ' + '
                    else:
                        strs[j] += parts[j] + ' '*(lengths[-1] -
                                                   len(parts[j])+
                                                   3)
                else:
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    strs[j] += ' '*(lengths[-1]+3)

        return prettyForm(u'\n'.join([s[:-3] for s in strs]))
```
### 83 - sympy/printing/pretty/pretty.py:

Start line: 1538, End line: 1551

```python
class PrettyPrinter(Printer):

    def _print_SingularityFunction(self, e):
        if self._use_unicode:
            shift = self._print(e.args[0]-e.args[1])
            n = self._print(e.args[2])
            base = prettyForm("<")
            base = prettyForm(*base.right(shift))
            base = prettyForm(*base.right(">"))
            pform = base**n
            return pform
        else:
            n = self._print(e.args[2])
            shift = self._print(e.args[0]-e.args[1])
            base = self._print_seq(shift, "<", ">", ' ')
            return base**n
```
### 84 - sympy/printing/pretty/pretty.py:

Start line: 334, End line: 369

```python
class PrettyPrinter(Printer):

    def _print_Derivative(self, deriv):
        if requires_partial(deriv.expr) and self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None
        count_total_deriv = 0

        for sym, num in reversed(deriv.variable_count):
            s = self._print(sym)
            ds = prettyForm(*s.left(deriv_symbol))
            count_total_deriv += num

            if (not num.is_Integer) or (num > 1):
                ds = ds**prettyForm(str(num))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        if (count_total_deriv > 1) != False:
            pform = pform**prettyForm(str(count_total_deriv))

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform
```
### 86 - sympy/printing/pretty/pretty.py:

Start line: 1517, End line: 1536

```python
class PrettyPrinter(Printer):

    def _print_Order(self, expr):
        pform = self._print(expr.expr)
        if (expr.point and any(p != S.Zero for p in expr.point)) or \
           len(expr.variables) > 1:
            pform = prettyForm(*pform.right("; "))
            if len(expr.variables) > 1:
                pform = prettyForm(*pform.right(self._print(expr.variables)))
            elif len(expr.variables):
                pform = prettyForm(*pform.right(self._print(expr.variables[0])))
            if self._use_unicode:
                pform = prettyForm(*pform.right(u" \N{RIGHTWARDS ARROW} "))
            else:
                pform = prettyForm(*pform.right(" -> "))
            if len(expr.point) > 1:
                pform = prettyForm(*pform.right(self._print(expr.point)))
            else:
                pform = prettyForm(*pform.right(self._print(expr.point[0])))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left("O"))
        return pform
```
### 88 - sympy/printing/pretty/pretty.py:

Start line: 2400, End line: 2411

```python
class PrettyPrinter(Printer):

    def _print_KroneckerDelta(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.right((prettyForm(','))))
        pform = prettyForm(*pform.right((self._print(e.args[1]))))
        if self._use_unicode:
            a = stringPict(pretty_symbol('delta'))
        else:
            a = stringPict('d')
        b = pform
        top = stringPict(*b.left(' '*a.width()))
        bot = stringPict(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))
```
### 89 - sympy/printing/pretty/pretty.py:

Start line: 123, End line: 133

```python
class PrettyPrinter(Printer):

    def _print_Dot(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform
```
### 91 - sympy/printing/pretty/pretty.py:

Start line: 1990, End line: 2004

```python
class PrettyPrinter(Printer):

    def _print_ImageSet(self, ts):
        if self._use_unicode:
            inn = u"\N{SMALL ELEMENT OF}"
        else:
            inn = 'in'
        fun = ts.lamda
        sets = ts.base_sets
        signature = fun.signature
        expr = self._print(fun.expr)
        bar = self._print("|")
        if len(signature) == 1:
            return self._print_seq((expr, bar, signature[0], inn, sets[0]), "{", "}", ' ')
        else:
            pargs = tuple(j for var, setv in zip(signature, sets) for j in (var, inn, setv, ","))
            return self._print_seq((expr, bar) + pargs[:-1], "{", "}", ' ')
```
### 93 - sympy/printing/repr.py:

Start line: 186, End line: 219

```python
class ReprPrinter(Printer):

    def _print_Predicate(self, expr):
        return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))

    def _print_AppliedPredicate(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__, expr.func, expr.arg)

    def _print_str(self, expr):
        return repr(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.reprify(expr, ", ")

    def _print_WildFunction(self, expr):
        return "%s('%s')" % (expr.__class__.__name__, expr.name)

    def _print_AlgebraicNumber(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__,
            self._print(expr.root), self._print(expr.coeffs()))

    def _print_PolyRing(self, ring):
        return "%s(%s, %s, %s)" % (ring.__class__.__name__,
            self._print(ring.symbols), self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "%s(%s, %s, %s)" % (field.__class__.__name__,
            self._print(field.symbols), self._print(field.domain), self._print(field.order))

    def _print_PolyElement(self, poly):
        terms = list(poly.terms())
        terms.sort(key=poly.ring.order, reverse=True)
        return "%s(%s, %s)" % (poly.__class__.__name__, self._print(poly.ring), self._print(terms))
```
### 94 - sympy/printing/pretty/pretty.py:

Start line: 1842, End line: 1857

```python
class PrettyPrinter(Printer):

    def _print_Pow(self, power):
        from sympy.simplify.simplify import fraction
        b, e = power.as_base_exp()
        if power.is_commutative:
            if e is S.NegativeOne:
                return prettyForm("1")/self._print(b)
            n, d = fraction(e)
            if n is S.One and d.is_Atom and not e.is_Integer and self._settings['root_notation']:
                return self._print_nth_root(b, e)
            if e.is_Rational and e < 0:
                return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))

        if b.is_Relational:
            return prettyForm(*self._print(b).parens()).__pow__(self._print(e))

        return self._print(b)**self._print(e)
```
### 97 - sympy/printing/pretty/pretty.py:

Start line: 1609, End line: 1632

```python
class PrettyPrinter(Printer):

    def _print_elliptic_e(self, e):
        pforma0 = self._print(e.args[0])
        if len(e.args) == 1:
            pform = pforma0
        else:
            pforma1 = self._print(e.args[1])
            pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('E'))
        return pform

    def _print_elliptic_k(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('K'))
        return pform

    def _print_elliptic_f(self, e):
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('F'))
        return pform
```
### 99 - sympy/printing/pretty/pretty.py:

Start line: 1739, End line: 1794

```python
class PrettyPrinter(Printer):

    def _print_Mul(self, product):
        from sympy.physics.units import Quantity
        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = product.as_ordered_factors()
        else:
            args = list(product.args)

        # If quantities are present append them at the back
        args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                     (isinstance(x, Pow) and isinstance(x.base, Quantity)))

        # Gather terms for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append( Rational(item.p) )
                if item.q != 1:
                    b.append( Rational(item.q) )
            else:
                a.append(item)

        from sympy import Integral, Piecewise, Product, Sum

        # Convert to pretty forms. Add parens to Add instances if there
        # is more than one term in the numer/denom
        for i in range(0, len(a)):
            if (a[i].is_Add and len(a) > 1) or (i != len(a) - 1 and
                    isinstance(a[i], (Integral, Piecewise, Product, Sum))):
                a[i] = prettyForm(*self._print(a[i]).parens())
            elif a[i].is_Relational:
                a[i] = prettyForm(*self._print(a[i]).parens())
            else:
                a[i] = self._print(a[i])

        for i in range(0, len(b)):
            if (b[i].is_Add and len(b) > 1) or (i != len(b) - 1 and
                    isinstance(b[i], (Integral, Piecewise, Product, Sum))):
                b[i] = prettyForm(*self._print(b[i]).parens())
            else:
                b[i] = self._print(b[i])

        # Construct a pretty form
        if len(b) == 0:
            return prettyForm.__mul__(*a)
        else:
            if len(a) == 0:
                a.append( self._print(S.One) )
            return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)
```
### 101 - sympy/printing/pretty/pretty.py:

Start line: 1569, End line: 1586

```python
class PrettyPrinter(Printer):

    def _print_DiracDelta(self, e):
        if self._use_unicode:
            if len(e.args) == 2:
                a = prettyForm(greek_unicode['delta'])
                b = self._print(e.args[1])
                b = prettyForm(*b.parens())
                c = self._print(e.args[0])
                c = prettyForm(*c.parens())
                pform = a**b
                pform = prettyForm(*pform.right(' '))
                pform = prettyForm(*pform.right(c))
                return pform
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens())
            pform = prettyForm(*pform.left(greek_unicode['delta']))
            return pform
        else:
            return self._print_Function(e)
```
### 104 - sympy/printing/pretty/pretty.py:

Start line: 2546, End line: 2554

```python
class PrettyPrinter(Printer):

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return self._print(u'\N{DOUBLE-STRUCK ITALIC SMALL D} ' + pretty_symbol(string))
        else:
            pform = self._print(field)
            pform = prettyForm(*pform.parens())
            return prettyForm(*pform.left(u"\N{DOUBLE-STRUCK ITALIC SMALL D}"))
```
### 106 - sympy/printing/pretty/pretty.py:

Start line: 1294, End line: 1376

```python
class PrettyPrinter(Printer):

    def _print_meijerg(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment

        v = {}
        v[(0, 0)] = [self._print(a) for a in e.an]
        v[(0, 1)] = [self._print(a) for a in e.aother]
        v[(1, 0)] = [self._print(b) for b in e.bm]
        v[(1, 1)] = [self._print(b) for b in e.bother]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        vp = {}
        for idx in v:
            vp[idx] = self._hprint_vec(v[idx])

        for i in range(2):
            maxw = max(vp[(0, i)].width(), vp[(1, i)].width())
            for j in range(2):
                s = vp[(j, i)]
                left = (maxw - s.width()) // 2
                right = maxw - left - s.width()
                s = prettyForm(*s.left(' ' * left))
                s = prettyForm(*s.right(' ' * right))
                vp[(j, i)] = s

        D1 = prettyForm(*vp[(0, 0)].right('  ', vp[(0, 1)]))
        D1 = prettyForm(*D1.below(' '))
        D2 = prettyForm(*vp[(1, 0)].right('  ', vp[(1, 1)]))
        D = prettyForm(*D1.below(D2))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the G symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('G')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)

        pp = self._print(len(e.ap))
        pq = self._print(len(e.bq))
        pm = self._print(len(e.bm))
        pn = self._print(len(e.an))

        def adjust(p1, p2):
            diff = p1.width() - p2.width()
            if diff == 0:
                return p1, p2
            elif diff > 0:
                return p1, prettyForm(*p2.left(' '*diff))
            else:
                return prettyForm(*p1.left(' '*-diff)), p2
        pp, pm = adjust(pp, pm)
        pq, pn = adjust(pq, pn)
        pu = prettyForm(*pm.right(', ', pn))
        pl = prettyForm(*pp.right(', ', pq))

        ht = F.baseline - above - 2
        if ht > 0:
            pu = prettyForm(*pu.below('\n'*ht))
        p = prettyForm(*pu.below(pl))

        F.baseline = above
        F = prettyForm(*F.right(p))

        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D
```
### 107 - sympy/printing/str.py:

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
### 108 - sympy/combinatorics/permutations.py:

Start line: 909, End line: 949

```python
class Permutation(Atom):

    def __new__(cls, *args, **kwargs):

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

        if not is_cycle and \
                any(i not in temp for i in range(len(temp))):
            raise ValueError("Integers 0 through %s must be present." %
                             max(temp))

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
### 111 - sympy/printing/pretty/pretty.py:

Start line: 854, End line: 894

```python
class PrettyPrinter(Printer):

    def _print_Identity(self, expr):
        if self._use_unicode:
            return prettyForm(u'\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL I}')
        else:
            return prettyForm('I')

    def _print_ZeroMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(u'\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO}')
        else:
            return prettyForm('0')

    def _print_OneMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(u'\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ONE}')
        else:
            return prettyForm('1')

    def _print_DotProduct(self, expr):
        args = list(expr.args)

        for i, a in enumerate(args):
            args[i] = self._print(a)
        return prettyForm.__mul__(*args)

    def _print_MatPow(self, expr):
        pform = self._print(expr.base)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.base, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**(self._print(expr.exp))
        return pform

    def _print_HadamardProduct(self, expr):
        from sympy import MatAdd, MatMul, HadamardProduct
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            delim = '.*'
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul, HadamardProduct)))
```
### 112 - sympy/printing/pretty/pretty.py:

Start line: 1137, End line: 1164

```python
class PrettyPrinter(Printer):

    def _print_PartialDerivative(self, deriv):
        if self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None

        for variable in reversed(deriv.variables):
            s = self._print(variable)
            ds = prettyForm(*s.left(deriv_symbol))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform
```
### 114 - sympy/printing/pretty/pretty.py:

Start line: 800, End line: 823

```python
class PrettyPrinter(Printer):

    def _print_Transpose(self, expr):
        pform = self._print(expr.arg)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.arg, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**(prettyForm('T'))
        return pform

    def _print_Adjoint(self, expr):
        pform = self._print(expr.arg)
        if self._use_unicode:
            dag = prettyForm(u'\N{DAGGER}')
        else:
            dag = prettyForm('+')
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.arg, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**dag
        return pform

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            return self._print(B.blocks[0, 0])
        return self._print(B.blocks)
```
### 116 - sympy/combinatorics/permutations.py:

Start line: 1035, End line: 1074

```python
class Permutation(Atom):

    def list(self, size=None):
        """Return the permutation as an explicit list, possibly
        trimming unmoved elements if size is less than the maximum
        element in the permutation; if this is desired, setting
        ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = False
        >>> p = Permutation(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Permutation(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        >>> Permutation(3).list(-1)
        []
        """
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        rv = self.array_form
        if size is not None:
            if size > self.size:
                rv.extend(list(range(self.size, size)))
            else:
                # find first value from rhs where rv[i] != i
                i = self.size - 1
                while rv:
                    if rv[-1] != i:
                        break
                    rv.pop()
                    i -= 1
        return rv
```
### 117 - sympy/printing/latex.py:

Start line: 974, End line: 1029

```python
class LatexPrinter(Printer):

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex
```
### 119 - sympy/printing/pretty/pretty.py:

Start line: 842, End line: 852

```python
class PrettyPrinter(Printer):

    def _print_MatMul(self, expr):
        args = list(expr.args)
        from sympy import Add, MatAdd, HadamardProduct, KroneckerProduct
        for i, a in enumerate(args):
            if (isinstance(a, (Add, MatAdd, HadamardProduct, KroneckerProduct))
                    and len(expr.args) > 1):
                args[i] = prettyForm(*self._print(a).parens())
            else:
                args[i] = self._print(a)

        return prettyForm.__mul__(*args)
```
### 121 - sympy/interactive/printing.py:

Start line: 337, End line: 571

```python
# Used by the doctester to override the default for no_global
NO_GLOBAL = False

def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    # ... other code
```
### 124 - sympy/printing/repr.py:

Start line: 173, End line: 184

```python
class ReprPrinter(Printer):

    def _print_Symbol(self, expr):
        d = expr._assumptions.generator
        # print the dummy_index like it was an assumption
        if expr.is_Dummy:
            d['dummy_index'] = expr.dummy_index

        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))
```
### 127 - sympy/printing/pretty/pretty.py:

Start line: 1649, End line: 1666

```python
class PrettyPrinter(Printer):

    def _print_GoldenRatio(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('phi'))
        return self._print(Symbol("GoldenRatio"))

    def _print_EulerGamma(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('gamma'))
        return self._print(Symbol("EulerGamma"))

    def _print_Mod(self, expr):
        pform = self._print(expr.args[0])
        if pform.binding > prettyForm.MUL:
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right(' mod '))
        pform = prettyForm(*pform.right(self._print(expr.args[1])))
        pform.binding = prettyForm.OPEN
        return pform
```
### 128 - sympy/printing/latex.py:

Start line: 2236, End line: 2309

```python
class LatexPrinter(Printer):

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return r"W\left(%s\right)" % self._print(expr.args[0])
        return r"W_{%s}\left(%s\right)" % \
            (self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))
```
### 129 - sympy/printing/pretty/pretty.py:

Start line: 31, End line: 91

```python
class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""
    printmethod = "_pretty"

    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "use_unicode": None,
        "wrap_line": True,
        "num_columns": None,
        "use_unicode_sqrt_char": True,
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if not isinstance(self._settings['imaginary_unit'], string_types):
            raise TypeError("'imaginary_unit' must a string, not {}".format(self._settings['imaginary_unit']))
        elif self._settings['imaginary_unit'] not in ["i", "j"]:
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '{}'".format(self._settings['imaginary_unit']))
        self.emptyPrinter = lambda x: prettyForm(xstr(x))

    @property
    def _use_unicode(self):
        if self._settings['use_unicode']:
            return True
        else:
            return pretty_use_unicode()

    def doprint(self, expr):
        return self._print(expr).render(**self._settings)

    # empty op so _print(stringPict) returns the same
    def _print_stringPict(self, e):
        return e

    def _print_basestring(self, e):
        return prettyForm(e)

    def _print_atan2(self, e):
        pform = prettyForm(*self._print_seq(e.args).parens())
        pform = prettyForm(*pform.left('atan2'))
        return pform

    def _print_Symbol(self, e, bold_name=False):
        symb = pretty_symbol(e.name, bold_name)
        return prettyForm(symb)
    _print_RandomSymbol = _print_Symbol
    def _print_MatrixSymbol(self, e):
        return self._print_Symbol(e, self._settings['mat_symbol_style'] == "bold")

    def _print_Float(self, e):
        # we will use StrPrinter's Float printer, but we need to handle the
        # full_prec ourselves, according to the self._print_level
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))
```
### 130 - sympy/printing/pretty/pretty.py:

Start line: 1064, End line: 1097

```python
class PrettyPrinter(Printer):

    def _printer_tensor_indices(self, name, indices, index_map={}):
        center = stringPict(name)
        top = stringPict(" "*center.width())
        bot = stringPict(" "*center.width())

        last_valence = None
        prev_map = None

        for i, index in enumerate(indices):
            indpic = self._print(index.args[0])
            if ((index in index_map) or prev_map) and last_valence == index.is_up:
                if index.is_up:
                    top = prettyForm(*stringPict.next(top, ","))
                else:
                    bot = prettyForm(*stringPict.next(bot, ","))
            if index in index_map:
                indpic = prettyForm(*stringPict.next(indpic, "="))
                indpic = prettyForm(*stringPict.next(indpic, self._print(index_map[index])))
                prev_map = True
            else:
                prev_map = False
            if index.is_up:
                top = stringPict(*top.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                bot = stringPict(*bot.right(" "*indpic.width()))
            else:
                bot = stringPict(*bot.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                top = stringPict(*top.right(" "*indpic.width()))
            last_valence = index.is_up

        pict = prettyForm(*center.above(top))
        pict = prettyForm(*pict.below(bot))
        return pict
```
### 131 - sympy/printing/repr.py:

Start line: 123, End line: 154

```python
class ReprPrinter(Printer):

    _print_SparseMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_NaN(self, expr):
        return "nan"

    def _print_Mul(self, expr, order=None):
        terms = expr.args
        if self.order != 'old':
            args = expr._new_rawargs(*terms).as_ordered_factors()
        else:
            args = terms

        nargs = len(args)
        args = map(self._print, args)
        clsname = type(expr).__name__
        if nargs > 255:  # Issue #10259, Python < 3.7
            return clsname + "(*[%s])" % ", ".join(args)
        return clsname + "(%s)" % ", ".join(args)
```
### 132 - sympy/interactive/printing.py:

Start line: 347, End line: 494

```python
def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    r"""
    Initializes pretty-printer depending on the environment.

    Parameters
    ==========

    pretty_print : boolean, default=True
        If True, use pretty_print to stringify or the provided pretty
        printer; if False, use sstrrepr to stringify or the provided string
        printer.
    order : string or None, default='lex'
        There are a few different settings for this parameter:
        lex (default), which is lexographic order;
        grlex, which is graded lexographic order;
        grevlex, which is reversed graded lexographic order;
        old, which is used for compatibility reasons and for long expressions;
        None, which sets it to lex.
    use_unicode : boolean or None, default=None
        If True, use unicode characters;
        if False, do not use unicode characters;
        if None, make a guess based on the environment.
    use_latex : string, boolean, or None, default=None
        If True, use default LaTeX rendering in GUI interfaces (png and
        mathjax);
        if False, do not use LaTeX rendering;
        if None, make a guess based on the environment;
        if 'png', enable latex rendering with an external latex compiler,
        falling back to matplotlib if external compilation fails;
        if 'matplotlib', enable LaTeX rendering with matplotlib;
        if 'mathjax', enable LaTeX text generation, for example MathJax
        rendering in IPython notebook or text rendering in LaTeX documents;
        if 'svg', enable LaTeX rendering with an external latex compiler,
        no fallback
    wrap_line : boolean
        If True, lines will wrap at the end; if False, they will not wrap
        but continue as one line. This is only relevant if ``pretty_print`` is
        True.
    num_columns : int or None, default=None
        If int, number of columns before wrapping is set to num_columns; if
        None, number of columns before wrapping is set to terminal width.
        This is only relevant if ``pretty_print`` is True.
    no_global : boolean, default=False
        If True, the settings become system wide;
        if False, use just for this console/session.
    ip : An interactive console
        This can either be an instance of IPython,
        or a class that derives from code.InteractiveConsole.
    euler : boolean, optional, default=False
        Loads the euler package in the LaTeX preamble for handwritten style
        fonts (http://www.ctan.org/pkg/euler).
    forecolor : string or None, optional, default=None
        DVI setting for foreground color. None means that either 'Black',
        'White', or 'Gray' will be selected based on a guess of the IPython
        terminal color setting. See notes.
    backcolor : string, optional, default='Transparent'
        DVI setting for background color. See notes.
    fontsize : string, optional, default='10pt'
        A font size to pass to the LaTeX documentclass function in the
        preamble. Note that the options are limited by the documentclass.
        Consider using scale instead.
    latex_mode : string, optional, default='plain'
        The mode used in the LaTeX printer. Can be one of:
        {'inline'|'plain'|'equation'|'equation*'}.
    print_builtin : boolean, optional, default=True
        If ``True`` then floats and integers will be printed. If ``False`` the
        printer will only print SymPy types.
    str_printer : function, optional, default=None
        A custom string printer function. This should mimic
        sympy.printing.sstrrepr().
    pretty_printer : function, optional, default=None
        A custom pretty printer. This should mimic sympy.printing.pretty().
    latex_printer : function, optional, default=None
        A custom LaTeX printer. This should mimic sympy.printing.latex().
    scale : float, optional, default=1.0
        Scale the LaTeX output when using the ``png`` or ``svg`` backends.
        Useful for high dpi screens.
    settings :
        Any additional settings for the ``latex`` and ``pretty`` commands can
        be used to fine-tune the output.

    Examples
    ========

    >>> from sympy.interactive import init_printing
    >>> from sympy import Symbol, sqrt
    >>> from sympy.abc import x, y
    >>> sqrt(5)
    sqrt(5)
    >>> init_printing(pretty_print=True) # doctest: +SKIP
    >>> sqrt(5) # doctest: +SKIP
      ___
    \/ 5
    >>> theta = Symbol('theta') # doctest: +SKIP
    >>> init_printing(use_unicode=True) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    >>> init_printing(use_unicode=False) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    theta
    >>> init_printing(order='lex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grlex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grevlex') # doctest: +SKIP
    >>> str(y * x**2 + x * y**2) # doctest: +SKIP
    x**2*y + x*y**2
    >>> init_printing(order='old') # doctest: +SKIP
    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(num_columns=10) # doctest: +SKIP
    >>> x**2 + x + y**2 + y # doctest: +SKIP
    x + y +
    x**2 + y**2

    Notes
    =====

    The foreground and background colors can be selected when using 'png' or
    'svg' LaTeX rendering. Note that before the ``init_printing`` command is
    executed, the LaTeX rendering is handled by the IPython console and not SymPy.

    The colors can be selected among the 68 standard colors known to ``dvips``,
    for a list see [1]_. In addition, the background color can be
    set to  'Transparent' (which is the default value).

    When using the 'Auto' foreground color, the guess is based on the
    ``colors`` variable in the IPython console, see [2]_. Hence, if
    that variable is set correctly in your IPython console, there is a high
    chance that the output will be readable, although manual settings may be
    needed.


    References
    ==========

    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips

    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors

    See Also
    ========

    sympy.printing.latex
    sympy.printing.pretty

    """
    # ... other code
```
### 135 - sympy/interactive/printing.py:

Start line: 495, End line: 572

```python
def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    import sys
    from sympy.printing.printer import Printer

    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    else:
        if str_printer is not None:
            stringify_func = str_printer
        else:
            from sympy.printing import sstrrepr as stringify_func

    # Even if ip is not passed, double check that not in IPython shell
    in_ipython = False
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass
        else:
            in_ipython = (ip is not None)

    if ip and not in_ipython:
        in_ipython = _is_ipython(ip)

    if in_ipython and pretty_print:
        try:
            import IPython
            # IPython 1.0 deprecates the frontend module, so we import directly
            # from the terminal module to prevent a deprecation message from being
            # shown.
            if V(IPython.__version__) >= '1.0':
                from IPython.terminal.interactiveshell import TerminalInteractiveShell
            else:
                from IPython.frontend.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # This will be True if we are in the qtconsole or notebook
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    if not NO_GLOBAL and not no_global:
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        if pretty_print:
            stringify_func = lambda expr: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns)
        else:
            stringify_func = lambda expr: _stringify_func(expr, order=order)

    if in_ipython:
        mode_in_settings = settings.pop("mode", None)
        if mode_in_settings:
            debug("init_printing: Mode is not able to be set due to internals"
                  "of IPython printing")
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer, scale,
                               **settings)
    else:
        _init_python_printing(stringify_func, **settings)
```
### 136 - sympy/printing/pretty/pretty.py:

Start line: 577, End line: 627

```python
class PrettyPrinter(Printer):

    def _print_Sum(self, expr):
        # ... other code

        f = expr.function

        prettyF = self._print(f)

        if f.is_Add:  # add parens
            prettyF = prettyForm(*prettyF.parens())

        H = prettyF.height() + 2

        # \sum \sum \sum ...
        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            prettyLower, prettyUpper = self.__print_SumProduct_Limits(lim)

            max_upper = max(max_upper, prettyUpper.height())

            # Create sum sign based on the height of the argument
            d, h, slines, adjustment = asum(
                H, prettyLower.width(), prettyUpper.width(), ascii_mode)
            prettySign = stringPict('')
            prettySign = prettyForm(*prettySign.stack(*slines))

            if first:
                sign_height = prettySign.height()

            prettySign = prettyForm(*prettySign.above(prettyUpper))
            prettySign = prettyForm(*prettySign.below(prettyLower))

            if first:
                # change F baseline so it centers on the sign
                prettyF.baseline -= d - (prettyF.height()//2 -
                                         prettyF.baseline)
                first = False

            # put padding to the right
            pad = stringPict('')
            pad = prettyForm(*pad.stack(*[' ']*h))
            prettySign = prettyForm(*prettySign.right(pad))
            # put the present prettyF to the right
            prettyF = prettyForm(*prettySign.right(prettyF))

        # adjust baseline of ascii mode sigma with an odd height so that it is
        # exactly through the center
        ascii_adjustment = ascii_mode if not adjustment else 0
        prettyF.baseline = max_upper + sign_height//2 + ascii_adjustment

        prettyF.binding = prettyForm.MUL
        return prettyF
```
### 141 - sympy/printing/str.py:

Start line: 650, End line: 670

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
### 143 - sympy/printing/pretty/pretty.py:

Start line: 1099, End line: 1135

```python
class PrettyPrinter(Printer):

    def _print_Tensor(self, expr):
        name = expr.args[0].name
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].name
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        sign, args = expr._get_args_for_traditional_printer()
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in args
        ]
        pform = prettyForm.__mul__(*args)
        if sign:
            return prettyForm(*pform.left(sign))
        else:
            return pform

    def _print_TensAdd(self, expr):
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in expr.args
        ]
        return prettyForm.__add__(*args)

    def _print_TensorIndex(self, expr):
        sym = expr.args[0]
        if not expr.is_up:
            sym = -sym
        return self._print(sym)
```
### 145 - sympy/printing/pretty/pretty.py:

Start line: 1222, End line: 1244

```python
class PrettyPrinter(Printer):

    def _print_ITE(self, ite):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(ite.rewrite(Piecewise))

    def _hprint_vec(self, v):
        D = None

        for a in v:
            p = a
            if D is None:
                D = p
            else:
                D = prettyForm(*D.right(', '))
                D = prettyForm(*D.right(p))
        if D is None:
            D = stringPict(' ')

        return D

    def _hprint_vseparator(self, p1, p2):
        tmp = prettyForm(*p1.right(p2))
        sep = stringPict(vobj('|', tmp.height()), baseline=tmp.baseline)
        return prettyForm(*p1.right(sep, p2))
```
### 146 - sympy/printing/latex.py:

Start line: 2215, End line: 2225

```python
class LatexPrinter(Printer):

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)
```
### 148 - sympy/printing/pretty/pretty.py:

Start line: 1634, End line: 1647

```python
class PrettyPrinter(Printer):

    def _print_elliptic_pi(self, e):
        name = greek_unicode['Pi'] if self._use_unicode else 'Pi'
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        if len(e.args) == 2:
            pform = self._hprint_vseparator(pforma0, pforma1)
        else:
            pforma2 = self._print(e.args[2])
            pforma = self._hprint_vseparator(pforma1, pforma2)
            pforma = prettyForm(*pforma.left('; '))
            pform = prettyForm(*pforma.left(pforma0))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(name))
        return pform
```
### 152 - sympy/printing/pretty/pretty.py:

Start line: 726, End line: 751

```python
class PrettyPrinter(Printer):

    def _print_MatrixBase(self, e):
        D = self._print_matrix_contents(e)
        D.baseline = D.height()//2
        D = prettyForm(*D.parens('[', ']'))
        return D
    _print_ImmutableMatrix = _print_MatrixBase
    _print_Matrix = _print_MatrixBase

    def _print_TensorProduct(self, expr):
        # This should somehow share the code with _print_WedgeProduct:
        circled_times = "\u2297"
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_WedgeProduct(self, expr):
        # This should somehow share the code with _print_TensorProduct:
        wedge_symbol = u"\u2227"
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_Trace(self, e):
        D = self._print(e.arg)
        D = prettyForm(*D.parens('(',')'))
        D.baseline = D.height()//2
        D = prettyForm(*D.left('\n'*(0) + 'tr'))
        return D
```
### 153 - sympy/printing/pretty/pretty.py:

Start line: 390, End line: 464

```python
class PrettyPrinter(Printer):

    def _print_Integral(self, integral):
        f = integral.function

        # Add parentheses if arg involves addition of terms and
        # create a pretty form for the argument
        prettyF = self._print(f)
        # XXX generalize parens
        if f.is_Add:
            prettyF = prettyForm(*prettyF.parens())

        # dx dy dz ...
        arg = prettyF
        for x in integral.limits:
            prettyArg = self._print(x[0])
            # XXX qparens (parens if needs-parens)
            if prettyArg.width() > 1:
                prettyArg = prettyForm(*prettyArg.parens())

            arg = prettyForm(*arg.right(' d', prettyArg))

        # \int \int \int ...
        firstterm = True
        s = None
        for lim in integral.limits:
            x = lim[0]
            # Create bar based on the height of the argument
            h = arg.height()
            H = h + 2

            # XXX hack!
            ascii_mode = not self._use_unicode
            if ascii_mode:
                H += 2

            vint = vobj('int', H)

            # Construct the pretty form with the integral sign and the argument
            pform = prettyForm(vint)
            pform.baseline = arg.baseline + (
                H - h)//2    # covering the whole argument

            if len(lim) > 1:
                # Create pretty forms for endpoints, if definite integral.
                # Do not print empty endpoints.
                if len(lim) == 2:
                    prettyA = prettyForm("")
                    prettyB = self._print(lim[1])
                if len(lim) == 3:
                    prettyA = self._print(lim[1])
                    prettyB = self._print(lim[2])

                if ascii_mode:  # XXX hack
                    # Add spacing so that endpoint can more easily be
                    # identified with the correct integral sign
                    spc = max(1, 3 - prettyB.width())
                    prettyB = prettyForm(*prettyB.left(' ' * spc))

                    spc = max(1, 4 - prettyA.width())
                    prettyA = prettyForm(*prettyA.right(' ' * spc))

                pform = prettyForm(*pform.above(prettyB))
                pform = prettyForm(*pform.below(prettyA))

            if not ascii_mode:  # XXX hack
                pform = prettyForm(*pform.right(' '))

            if firstterm:
                s = pform   # first term
                firstterm = False
            else:
                s = prettyForm(*s.left(pform))

        pform = prettyForm(*arg.left(s))
        pform.binding = prettyForm.MUL
        return pform
```
### 154 - sympy/combinatorics/permutations.py:

Start line: 2248, End line: 2274

```python
class Permutation(Atom):

    @property
    def cycle_structure(self):
        """Return the cycle structure of the permutation as a dictionary
        indicating the multiplicity of each cycle length.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.print_cyclic = True
        >>> Permutation(3).cycle_structure
        {1: 4}
        >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure
        {2: 2, 3: 1}
        """
        if self._cycle_structure:
            rv = self._cycle_structure
        else:
            rv = defaultdict(int)
            singletons = self.size
            for c in self.cyclic_form:
                rv[len(c)] += 1
                singletons -= len(c)
            if singletons:
                rv[1] = singletons
            self._cycle_structure = rv
        return dict(rv)  # make a copy
```
### 158 - sympy/printing/pretty/pretty.py:

Start line: 912, End line: 924

```python
class PrettyPrinter(Printer):

    def _print_KroneckerProduct(self, expr):
        from sympy import MatAdd, MatMul
        if self._use_unicode:
            delim = u' \N{N-ARY CIRCLED TIMES OPERATOR} '
        else:
            delim = ' x '
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_FunctionMatrix(self, X):
        D = self._print(X.lamda.expr)
        D = prettyForm(*D.parens('[', ']'))
        return D
```
### 159 - sympy/printing/latex.py:

Start line: 1991, End line: 2063

```python
class LatexPrinter(Printer):

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Rationals(self, i):
        return r"\mathbb{Q}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"
```
### 160 - sympy/printing/pretty/pretty.py:

Start line: 1859, End line: 1878

```python
class PrettyPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def __print_numer_denom(self, p, q):
        if q == 1:
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)
            else:
                return prettyForm(str(p))
        elif abs(p) >= 10 and abs(q) >= 10:
            # If more than one digit in numer and denom, print larger fraction
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)/prettyForm(str(q))
                # Old printing method:
                #pform = prettyForm(str(-p))/prettyForm(str(q))
                #return prettyForm(binding=prettyForm.NEG, *pform.left('- '))
            else:
                return prettyForm(str(p))/prettyForm(str(q))
        else:
            return None
```
### 161 - sympy/printing/pretty/pretty.py:

Start line: 1026, End line: 1062

```python
class PrettyPrinter(Printer):

    def _print_NDimArray(self, expr):
        from sympy import ImmutableMatrix

        if expr.rank() == 0:
            return self._print(expr[()])

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        # leave eventual matrix elements unflattened
        mat = lambda x: ImmutableMatrix(x, evaluate=False)
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(expr[outer_i])
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(level_str[back_outer_i+1])
                else:
                    level_str[back_outer_i].append(mat(
                        level_str[back_outer_i+1]))
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = mat(
                            [[level_str[back_outer_i][-1]]])
                even = not even
                level_str[back_outer_i+1] = []

        out_expr = level_str[0][0]
        if expr.rank() % 2 == 1:
            out_expr = mat([out_expr])

        return self._print(out_expr)

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_MutableSparseNDimArray = _print_NDimArray
```
### 163 - sympy/printing/str.py:

Start line: 165, End line: 178

```python
class StrPrinter(Printer):

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
### 165 - sympy/printing/pretty/pretty.py:

Start line: 1796, End line: 1840

```python
class PrettyPrinter(Printer):

    # A helper function for _print_Pow to print x**(1/n)
    def _print_nth_root(self, base, expt):
        bpretty = self._print(base)

        # In very simple cases, use a single-char root sign
        if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
            and expt is S.Half and bpretty.height() == 1
            and (bpretty.width() == 1
                 or (base.is_Integer and base.is_nonnegative))):
            return prettyForm(*bpretty.left(u'\N{SQUARE ROOT}'))

        # Construct root sign, start with the \/ shape
        _zZ = xobj('/', 1)
        rootsign = xobj('\\', 1) + _zZ
        # Make exponent number to put above it
        if isinstance(expt, Rational):
            exp = str(expt.q)
            if exp == '2':
                exp = ''
        else:
            exp = str(expt.args[0])
        exp = exp.ljust(2)
        if len(exp) > 2:
            rootsign = ' '*(len(exp) - 2) + rootsign
        # Stack the exponent
        rootsign = stringPict(exp + '\n' + rootsign)
        rootsign.baseline = 0
        # Diagonal: length is one less than height of base
        linelength = bpretty.height() - 1
        diagonal = stringPict('\n'.join(
            ' '*(linelength - i - 1) + _zZ + ' '*i
            for i in range(linelength)
        ))
        # Put baseline just below lowest line: next to exp
        diagonal.baseline = linelength - 1
        # Make the root symbol
        rootsign = prettyForm(*rootsign.right(diagonal))
        # Det the baseline to match contents to fix the height
        # but if the height of bpretty is one, the rootsign must be one higher
        rootsign.baseline = max(1, bpretty.baseline)
        #build result
        s = prettyForm(hobj('_', 2 + bpretty.width()))
        s = prettyForm(*bpretty.above(s))
        s = prettyForm(*s.left(rootsign))
        return s
```
### 167 - sympy/printing/pretty/pretty.py:

Start line: 825, End line: 840

```python
class PrettyPrinter(Printer):

    def _print_MatAdd(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                coeff = item.as_coeff_mmul()[0]
                if _coeff_isneg(S(coeff)):
                    s = prettyForm(*stringPict.next(s, ' '))
                    pform = self._print(item)
                else:
                    s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))

        return s
```
### 168 - sympy/printing/latex.py:

Start line: 1886, End line: 1901

```python
class LatexPrinter(Printer):

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return r"\left\{%s\right\}" % items


    _print_frozenset = _print_set
```
### 171 - sympy/printing/pretty/pretty.py:

Start line: 532, End line: 575

```python
class PrettyPrinter(Printer):

    def _print_Sum(self, expr):
        ascii_mode = not self._use_unicode

        def asum(hrequired, lower, upper, use_ascii):
            def adjust(s, wid=None, how='<^>'):
                if not wid or len(s) > wid:
                    return s
                need = wid - len(s)
                if how == '<^>' or how == "<" or how not in list('<^>'):
                    return s + ' '*need
                half = need//2
                lead = ' '*half
                if how == ">":
                    return " "*need + s
                return lead + s + ' '*(need - len(lead))

            h = max(hrequired, 2)
            d = h//2
            w = d + 1
            more = hrequired % 2

            lines = []
            if use_ascii:
                lines.append("_"*(w) + ' ')
                lines.append(r"\%s`" % (' '*(w - 1)))
                for i in range(1, d):
                    lines.append('%s\\%s' % (' '*i, ' '*(w - i)))
                if more:
                    lines.append('%s)%s' % (' '*(d), ' '*(w - d)))
                for i in reversed(range(1, d)):
                    lines.append('%s/%s' % (' '*i, ' '*(w - i)))
                lines.append("/" + "_"*(w - 1) + ',')
                return d, h + more, lines, more
            else:
                w = w + more
                d = d + more
                vsum = vobj('sum', 4)
                lines.append("_"*(w))
                for i in range(0, d):
                    lines.append('%s%s%s' % (' '*i, vsum[2], ' '*(w - i - 1)))
                for i in reversed(range(0, d)):
                    lines.append('%s%s%s' % (' '*i, vsum[4], ' '*(w - i - 1)))
                lines.append(vsum[8]*(w))
                return d, h + 2*more, lines, more
        # ... other code
```
### 175 - sympy/printing/pretty/pretty.py:

Start line: 2034, End line: 2068

```python
class PrettyPrinter(Printer):

    def _print_ComplexRegion(self, ts):
        if self._use_unicode:
            inn = u"\N{SMALL ELEMENT OF}"
        else:
            inn = 'in'
        variables = self._print_seq(ts.variables)
        expr = self._print(ts.expr)
        bar = self._print("|")
        prodsets = self._print(ts.sets)

        return self._print_seq((expr, bar, variables, inn, prodsets), "{", "}", ' ')

    def _print_Contains(self, e):
        var, set = e.args
        if self._use_unicode:
            el = u" \N{ELEMENT OF} "
            return prettyForm(*stringPict.next(self._print(var),
                                               el, self._print(set)), binding=8)
        else:
            return prettyForm(sstr(e))

    def _print_FourierSeries(self, s):
        if self._use_unicode:
            dots = u"\N{HORIZONTAL ELLIPSIS}"
        else:
            dots = '...'
        return self._print_Add(s.truncate()) + self._print(dots)

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_SetExpr(self, se):
        pretty_set = prettyForm(*self._print(se.set).parens())
        pretty_name = self._print(Symbol("SetExpr"))
        return prettyForm(*pretty_name.right(pretty_set))
```
### 177 - sympy/printing/pretty/pretty.py:

Start line: 1457, End line: 1515

```python
class PrettyPrinter(Printer):

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                if self._use_unicode:
                    return prettyForm(self._special_function_classes[cls][0])
                else:
                    return prettyForm(self._special_function_classes[cls][1])
        func_name = expr.__name__
        return prettyForm(pretty_symbol(func_name))

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is based on Tuple but should not print like a Tuple
        return self.emptyPrinter(expr)

    def _print_lerchphi(self, e):
        func_name = greek_unicode['Phi'] if self._use_unicode else 'lerchphi'
        return self._print_Function(e, func_name=func_name)

    def _print_dirichlet_eta(self, e):
        func_name = greek_unicode['eta'] if self._use_unicode else 'dirichlet_eta'
        return self._print_Function(e, func_name=func_name)

    def _print_Heaviside(self, e):
        func_name = greek_unicode['theta'] if self._use_unicode else 'Heaviside'
        return self._print_Function(e, func_name=func_name)

    def _print_fresnels(self, e):
        return self._print_Function(e, func_name="S")

    def _print_fresnelc(self, e):
        return self._print_Function(e, func_name="C")

    def _print_airyai(self, e):
        return self._print_Function(e, func_name="Ai")

    def _print_airybi(self, e):
        return self._print_Function(e, func_name="Bi")

    def _print_airyaiprime(self, e):
        return self._print_Function(e, func_name="Ai'")

    def _print_airybiprime(self, e):
        return self._print_Function(e, func_name="Bi'")

    def _print_LambertW(self, e):
        return self._print_Function(e, func_name="W")

    def _print_Lambda(self, e):
        expr = e.expr
        sig = e.signature
        if self._use_unicode:
            arrow = u" \N{RIGHTWARDS ARROW FROM BAR} "
        else:
            arrow = " -> "
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        var_form = self._print(sig)

        return prettyForm(*stringPict.next(var_form, arrow, self._print(expr)), binding=8)
```
### 180 - sympy/printing/latex.py:

Start line: 948, End line: 957

```python
class LatexPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
```
### 181 - sympy/combinatorics/permutations.py:

Start line: 2801, End line: 2838

```python
class Permutation(Atom):

    @classmethod
    def unrank_lex(cls, size, rank):
        """
        Lexicographic permutation unranking.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = False
        >>> a = Permutation.unrank_lex(5, 10)
        >>> a.rank()
        10
        >>> a
        Permutation([0, 2, 4, 1, 3])

        See Also
        ========

        rank, next_lex
        """
        perm_array = [0] * size
        psize = 1
        for i in range(size):
            new_psize = psize*(i + 1)
            d = (rank % new_psize) // psize
            rank -= d*psize
            perm_array[size - i - 1] = d
            for j in range(size - i, size):
                if perm_array[j] > d - 1:
                    perm_array[j] += 1
            psize = new_psize
        return cls._af_new(perm_array)

    # global flag to control how permutations are printed
    # when True, Permutation([0, 2, 1, 3]) -> Cycle(1, 2)
    # when False, Permutation([0, 2, 1, 3]) -> Permutation([0, 2, 1])
    print_cyclic = True
```
### 187 - sympy/printing/pretty/pretty.py:

Start line: 2556, End line: 2613

```python
class PrettyPrinter(Printer):

    def _print_Tr(self, p):
        #TODO: Handle indices
        pform = self._print(p.args[0])
        pform = prettyForm(*pform.left('%s(' % (p.__class__.__name__)))
        pform = prettyForm(*pform.right(')'))
        return pform

    def _print_primenu(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['nu']))
        else:
            pform = prettyForm(*pform.left('nu'))
        return pform

    def _print_primeomega(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['Omega']))
        else:
            pform = prettyForm(*pform.left('Omega'))
        return pform

    def _print_Quantity(self, e):
        if e.name.name == 'degree':
            pform = self._print(u"\N{DEGREE SIGN}")
            return pform
        else:
            return self.emptyPrinter(e)

    def _print_AssignmentBase(self, e):

        op = prettyForm(' ' + xsym(e.op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r))
        return pform


def pretty(expr, **settings):
    """Returns a string containing the prettified form of expr.

    For information on keyword arguments see pretty_print function.

    """
    pp = PrettyPrinter(settings)

    # XXX: this is an ugly hack, but at least it works
    use_unicode = pp._settings['use_unicode']
    uflag = pretty_use_unicode(use_unicode)

    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)
```
### 192 - sympy/printing/latex.py:

Start line: 2355, End line: 2421

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    def _print_SubModule(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for x in m.gens))

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens))

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{{{}}} + {{{}}}".format(self._print(x.data),
                 self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{{{}}} + {{{}}}".format(self._print(m.data),
                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    def _print_Tr(self, p):
        # TODO: Handle indices
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)
```
