# sympy__sympy-21930

| **sympy/sympy** | `de446c6d85f633271dfec1452f6f28ea783e293f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 10028 |
| **Avg pos** | 22.0 |
| **Min pos** | 22 |
| **Max pos** | 22 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/physics/secondquant.py b/sympy/physics/secondquant.py
--- a/sympy/physics/secondquant.py
+++ b/sympy/physics/secondquant.py
@@ -218,7 +218,7 @@ def _sortkey(cls, index):
             return (12, label, h)
 
     def _latex(self, printer):
-        return "%s^{%s}_{%s}" % (
+        return "{%s^{%s}_{%s}}" % (
             self.symbol,
             "".join([ i.name for i in self.args[1]]),
             "".join([ i.name for i in self.args[2]])
@@ -478,7 +478,7 @@ def __repr__(self):
         return "CreateBoson(%s)" % self.state
 
     def _latex(self, printer):
-        return "b^\\dagger_{%s}" % self.state.name
+        return "{b^\\dagger_{%s}}" % self.state.name
 
 B = AnnihilateBoson
 Bd = CreateBoson
@@ -939,7 +939,7 @@ def __repr__(self):
         return "CreateFermion(%s)" % self.state
 
     def _latex(self, printer):
-        return "a^\\dagger_{%s}" % self.state.name
+        return "{a^\\dagger_{%s}}" % self.state.name
 
 Fd = CreateFermion
 F = AnnihilateFermion

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/physics/secondquant.py | 221 | 221 | 22 | 2 | 10028
| sympy/physics/secondquant.py | 481 | 481 | - | 2 | -
| sympy/physics/secondquant.py | 942 | 942 | - | 2 | -


## Problem Statement

```
Issues with Latex printing output in second quantization module
There are Latex rendering problems within the "secondquant" module, as it does not correctly interpret double superscripts containing the "dagger" command within Jupyter Notebook.

Let's see a minimal example

\`\`\`
In [1]: import sympy as sp
        from sympy.physics.secondquant import B, Bd, Commutator
        sp.init_printing()

In [2]: a = sp.Symbol('0')

In [3]: Commutator(Bd(a)**2, B(a))
Out[3]: \displaystyle - \left[b_{0},b^\dagger_{0}^{2}\right]
\`\`\`
So, it doesn't render correctly, and that's because the double superscript `"b^\dagger_{0}^{2}"`. It should be correct by adding curly brackets `"{b^\dagger_{0}}^{2}"`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/physics/vector/printing.py | 289 | 326| 343 | 343 | 2936 | 
| 2 | **2 sympy/physics/secondquant.py** | 1 | 78| 465 | 808 | 25697 | 
| 3 | 2 sympy/physics/vector/printing.py | 69 | 103| 349 | 1157 | 25697 | 
| 4 | 3 sympy/physics/quantum/cg.py | 314 | 323| 152 | 1309 | 32433 | 
| 5 | 4 sympy/printing/latex.py | 84 | 119| 491 | 1800 | 62653 | 
| 6 | 4 sympy/physics/quantum/cg.py | 418 | 427| 185 | 1985 | 62653 | 
| 7 | 4 sympy/printing/latex.py | 1 | 83| 722 | 2707 | 62653 | 
| 8 | **4 sympy/physics/secondquant.py** | 1764 | 1802| 315 | 3022 | 62653 | 
| 9 | 4 sympy/physics/quantum/cg.py | 139 | 148| 152 | 3174 | 62653 | 
| 10 | 4 sympy/printing/latex.py | 867 | 964| 908 | 4082 | 62653 | 
| 11 | 5 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 58| 476 | 4558 | 67362 | 
| 12 | 6 sympy/physics/quantum/commutator.py | 213 | 236| 253 | 4811 | 69416 | 
| 13 | 6 sympy/printing/latex.py | 2782 | 2984| 2702 | 7513 | 69416 | 
| 14 | 7 sympy/physics/quantum/qexpr.py | 1 | 23| 122 | 7635 | 72451 | 
| 15 | 8 sympy/physics/quantum/spin.py | 1 | 58| 473 | 8108 | 92843 | 
| 16 | 9 sympy/physics/vector/dyadic.py | 159 | 193| 382 | 8490 | 97658 | 
| 17 | 9 sympy/physics/vector/printing.py | 44 | 67| 229 | 8719 | 97658 | 
| 18 | 10 sympy/physics/quantum/sho1d.py | 144 | 160| 165 | 8884 | 103049 | 
| 19 | 10 sympy/physics/quantum/cg.py | 380 | 416| 349 | 9233 | 103049 | 
| 20 | 10 sympy/physics/quantum/cg.py | 279 | 312| 322 | 9555 | 103049 | 
| 21 | 11 sympy/physics/quantum/anticommutator.py | 124 | 147| 257 | 9812 | 104218 | 
| **-> 22 <-** | **11 sympy/physics/secondquant.py** | 220 | 245| 216 | 10028 | 104218 | 
| 23 | 11 sympy/physics/quantum/cg.py | 213 | 234| 255 | 10283 | 104218 | 
| 24 | 11 sympy/physics/vector/printing.py | 1 | 12| 133 | 10416 | 104218 | 
| 25 | 12 sympy/printing/printer.py | 1 | 212| 1913 | 12329 | 107510 | 
| 26 | 13 sympy/printing/pretty/pretty_symbology.py | 89 | 165| 756 | 13085 | 113366 | 
| 27 | 13 sympy/physics/quantum/cg.py | 104 | 137| 317 | 13402 | 113366 | 
| 28 | 13 sympy/printing/pretty/pretty_symbology.py | 166 | 197| 314 | 13716 | 113366 | 
| 29 | 13 sympy/physics/quantum/spin.py | 783 | 814| 232 | 13948 | 113366 | 
| 30 | 14 sympy/physics/quantum/tensorproduct.py | 204 | 231| 285 | 14233 | 116814 | 
| 31 | 14 sympy/physics/quantum/commutator.py | 121 | 135| 148 | 14381 | 116814 | 
| 32 | **14 sympy/physics/secondquant.py** | 1657 | 1702| 356 | 14737 | 116814 | 
| 33 | 15 sympy/printing/mathml.py | 1695 | 1779| 658 | 15395 | 133704 | 
| 34 | 16 sympy/physics/vector/vector.py | 221 | 253| 373 | 15768 | 140223 | 
| 35 | **16 sympy/physics/secondquant.py** | 174 | 193| 162 | 15930 | 140223 | 
| 36 | 16 sympy/printing/mathml.py | 419 | 453| 282 | 16212 | 140223 | 
| 37 | 16 sympy/parsing/latex/_parse_latex_antlr.py | 562 | 595| 222 | 16434 | 140223 | 
| 38 | 17 examples/advanced/qft.py | 85 | 138| 607 | 17041 | 141526 | 
| 39 | 17 sympy/physics/quantum/spin.py | 816 | 850| 325 | 17366 | 141526 | 
| 40 | 17 sympy/printing/latex.py | 122 | 2746| 133 | 17499 | 141526 | 
| 41 | 18 sympy/physics/quantum/innerproduct.py | 119 | 137| 155 | 17654 | 142603 | 
| 42 | 18 sympy/printing/mathml.py | 1371 | 1410| 341 | 17995 | 142603 | 
| 43 | 19 sympy/printing/theanocode.py | 1 | 67| 613 | 18608 | 146969 | 
| 44 | 19 sympy/physics/vector/dyadic.py | 195 | 247| 456 | 19064 | 146969 | 
| 45 | **19 sympy/physics/secondquant.py** | 1704 | 1762| 499 | 19563 | 146969 | 
| 46 | 19 examples/advanced/qft.py | 1 | 36| 236 | 19799 | 146969 | 
| 47 | 19 sympy/printing/mathml.py | 1009 | 1042| 286 | 20085 | 146969 | 
| 48 | 20 sympy/parsing/latex/__init__.py | 1 | 36| 273 | 20358 | 147242 | 
| 49 | 21 sympy/physics/quantum/fermion.py | 81 | 109| 264 | 20622 | 148428 | 
| 50 | 22 sympy/physics/quantum/__init__.py | 1 | 60| 465 | 21087 | 148894 | 
| 51 | 22 sympy/printing/mathml.py | 1098 | 1157| 429 | 21516 | 148894 | 
| 52 | 23 sympy/printing/pretty/pretty.py | 835 | 858| 215 | 21731 | 174190 | 
| 53 | 24 sympy/physics/quantum/state.py | 171 | 190| 238 | 21969 | 181283 | 
| 54 | **24 sympy/physics/secondquant.py** | 2457 | 2566| 783 | 22752 | 181283 | 
| 55 | 24 sympy/printing/latex.py | 2275 | 2746| 4506 | 27258 | 181283 | 
| 56 | 24 sympy/printing/mathml.py | 1810 | 1867| 479 | 27737 | 181283 | 
| 57 | 24 sympy/physics/quantum/qexpr.py | 304 | 328| 178 | 27915 | 181283 | 
| 58 | **24 sympy/physics/secondquant.py** | 81 | 105| 145 | 28060 | 181283 | 
| 59 | 24 sympy/physics/quantum/commutator.py | 100 | 119| 187 | 28247 | 181283 | 
| 60 | 25 sympy/printing/numpy.py | 400 | 436| 368 | 28615 | 186234 | 
| 61 | 25 sympy/printing/mathml.py | 1173 | 1210| 333 | 28948 | 186234 | 
| 62 | 26 sympy/printing/octave.py | 378 | 465| 780 | 29728 | 192824 | 
| 63 | 26 sympy/printing/latex.py | 2749 | 2984| 239 | 29967 | 192824 | 
| 64 | 26 sympy/physics/quantum/spin.py | 348 | 418| 625 | 30592 | 192824 | 
| 65 | 27 sympy/core/_print_helpers.py | 54 | 66| 116 | 30708 | 193360 | 


## Patch

```diff
diff --git a/sympy/physics/secondquant.py b/sympy/physics/secondquant.py
--- a/sympy/physics/secondquant.py
+++ b/sympy/physics/secondquant.py
@@ -218,7 +218,7 @@ def _sortkey(cls, index):
             return (12, label, h)
 
     def _latex(self, printer):
-        return "%s^{%s}_{%s}" % (
+        return "{%s^{%s}_{%s}}" % (
             self.symbol,
             "".join([ i.name for i in self.args[1]]),
             "".join([ i.name for i in self.args[2]])
@@ -478,7 +478,7 @@ def __repr__(self):
         return "CreateBoson(%s)" % self.state
 
     def _latex(self, printer):
-        return "b^\\dagger_{%s}" % self.state.name
+        return "{b^\\dagger_{%s}}" % self.state.name
 
 B = AnnihilateBoson
 Bd = CreateBoson
@@ -939,7 +939,7 @@ def __repr__(self):
         return "CreateFermion(%s)" % self.state
 
     def _latex(self, printer):
-        return "a^\\dagger_{%s}" % self.state.name
+        return "{a^\\dagger_{%s}}" % self.state.name
 
 Fd = CreateFermion
 F = AnnihilateFermion

```

## Test Patch

```diff
diff --git a/sympy/physics/tests/test_secondquant.py b/sympy/physics/tests/test_secondquant.py
--- a/sympy/physics/tests/test_secondquant.py
+++ b/sympy/physics/tests/test_secondquant.py
@@ -94,7 +94,7 @@ def test_operator():
 def test_create():
     i, j, n, m = symbols('i,j,n,m')
     o = Bd(i)
-    assert latex(o) == "b^\\dagger_{i}"
+    assert latex(o) == "{b^\\dagger_{i}}"
     assert isinstance(o, CreateBoson)
     o = o.subs(i, j)
     assert o.atoms(Symbol) == {j}
@@ -258,7 +258,7 @@ def test_commutation():
     c1 = Commutator(F(a), Fd(a))
     assert Commutator.eval(c1, c1) == 0
     c = Commutator(Fd(a)*F(i),Fd(b)*F(j))
-    assert latex(c) == r'\left[a^\dagger_{a} a_{i},a^\dagger_{b} a_{j}\right]'
+    assert latex(c) == r'\left[{a^\dagger_{a}} a_{i},{a^\dagger_{b}} a_{j}\right]'
     assert repr(c) == 'Commutator(CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j))'
     assert str(c) == '[CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j)]'
 
@@ -288,7 +288,7 @@ def test_create_f():
     assert Dagger(B(p)).apply_operator(q) == q*CreateBoson(p)
     assert repr(Fd(p)) == 'CreateFermion(p)'
     assert srepr(Fd(p)) == "CreateFermion(Symbol('p'))"
-    assert latex(Fd(p)) == r'a^\dagger_{p}'
+    assert latex(Fd(p)) == r'{a^\dagger_{p}}'
 
 
 def test_annihilate_f():
@@ -426,7 +426,7 @@ def test_NO():
     assert no.has_q_annihilators == -1
     assert str(no) == ':CreateFermion(a)*CreateFermion(i):'
     assert repr(no) == 'NO(CreateFermion(a)*CreateFermion(i))'
-    assert latex(no) == r'\left\{a^\dagger_{a} a^\dagger_{i}\right\}'
+    assert latex(no) == r'\left\{{a^\dagger_{a}} {a^\dagger_{i}}\right\}'
     raises(NotImplementedError, lambda:  NO(Bd(p)*F(q)))
 
 
@@ -531,7 +531,7 @@ def test_Tensors():
     assert tabij.subs(b, c) == AT('t', (a, c), (i, j))
     assert (2*tabij).subs(i, c) == 2*AT('t', (a, b), (c, j))
     assert tabij.symbol == Symbol('t')
-    assert latex(tabij) == 't^{ab}_{ij}'
+    assert latex(tabij) == '{t^{ab}_{ij}}'
     assert str(tabij) == 't((_a, _b),(_i, _j))'
 
     assert AT('t', (a, a), (i, j)).subs(a, b) == AT('t', (b, b), (i, j))
@@ -1255,6 +1255,12 @@ def test_internal_external_pqrs_AT():
         assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
 
 
+def test_issue_19661():
+    a = Symbol('0')
+    assert latex(Commutator(Bd(a)**2, B(a))
+                 ) == '- \\left[b_{0},{b^\\dagger_{0}}^{2}\\right]'
+
+
 def test_canonical_ordering_AntiSymmetricTensor():
     v = symbols("v")
 

```


## Code snippets

### 1 - sympy/physics/vector/printing.py:

Start line: 289, End line: 326

```python
def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\mathbf{\\hat{n}_x} + \\mathbf{\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'
    >>> vlatex(q1d)
    '\\dot{q}_{1}'
    >>> vlatex(q1 * q2d)
    'q_{1} \\dot{q}_{2}'
    >>> vlatex(q1dd * q1 / q1d)
    '\\frac{q_{1} \\ddot{q}_{1}}{\\dot{q}_{1}}'

    """
    latex_printer = VectorLatexPrinter(settings)

    return latex_printer.doprint(expr)
```
### 2 - sympy/physics/secondquant.py:

Start line: 1, End line: 78

```python
"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory
of Many-Particle Systems."
"""
from collections import defaultdict

from sympy import (Add, Basic, cacheit, Dummy, Expr, Function, I,
                   KroneckerDelta, Mul, Pow, S, sqrt, Symbol, sympify, Tuple,
                   zeros)
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
from sympy.utilities import default_sort_key

__all__ = [
    'Dagger',
    'KroneckerDelta',
    'BosonicOperator',
    'AnnihilateBoson',
    'CreateBoson',
    'AnnihilateFermion',
    'CreateFermion',
    'FockState',
    'FockStateBra',
    'FockStateKet',
    'FockStateBosonKet',
    'FockStateBosonBra',
    'FockStateFermionKet',
    'FockStateFermionBra',
    'BBra',
    'BKet',
    'FBra',
    'FKet',
    'F',
    'Fd',
    'B',
    'Bd',
    'apply_operators',
    'InnerProduct',
    'BosonicBasis',
    'VarBosonicBasis',
    'FixedBosonicBasis',
    'Commutator',
    'matrix_rep',
    'contraction',
    'wicks',
    'NO',
    'evaluate_deltas',
    'AntiSymmetricTensor',
    'substitute_dummies',
    'PermutationOperator',
    'simplify_index_permutations',
]


class SecondQuantizationError(Exception):
    pass


class AppliesOnlyToSymbolicIndex(SecondQuantizationError):
    pass


class ContractionAppliesOnlyToFermions(SecondQuantizationError):
    pass


class ViolationOfPauliPrinciple(SecondQuantizationError):
    pass


class SubstitutionOfAmbigousOperatorFailed(SecondQuantizationError):
    pass


class WicksTheoremDoesNotApply(SecondQuantizationError):
    pass
```
### 3 - sympy/physics/vector/printing.py:

Start line: 69, End line: 103

```python
class VectorLatexPrinter(LatexPrinter):

    def _print_Derivative(self, der_expr):
        from sympy.physics.vector.functions import dynamicsymbols
        # make sure it is in the right form
        der_expr = der_expr.doit()
        if not isinstance(der_expr, Derivative):
            return r"\left(%s\right)" % self.doprint(der_expr)

        # check if expr is a dynamicsymbol
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all([True for i in red if i.free_symbols == {t}])
        test2 = not all([(t == i) for i in syms])
        if test1 or test2:
            return super()._print_Derivative(der_expr)

        # done checking
        dots = len(syms)
        base = self._print_Function(expr)
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else: # Fallback to standard printing
            return super()._print_Derivative(der_expr)
        if len(base_split) != 1:
            base += '_' + base_split[1]
        return base
```
### 4 - sympy/physics/quantum/cg.py:

Start line: 314, End line: 323

```python
class Wigner6j(Expr):

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12,
                    self.j3, self.j, self.j23))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)
```
### 5 - sympy/printing/latex.py:

Start line: 84, End line: 119

```python
modifier_dict = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'[{ ]*[-+0-9]'),  # match
)
```
### 6 - sympy/physics/quantum/cg.py:

Start line: 418, End line: 427

```python
class Wigner9j(Expr):

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3,
                self.j4, self.j34, self.j13, self.j24, self.j))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)
```
### 7 - sympy/printing/latex.py:

Start line: 1, End line: 83

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from typing import Any, Dict

import itertools

from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true

# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which sympy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg',
                            ]

tex_greek_dictionary = {
    'Alpha': 'A',
    'Beta': 'B',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': 'E',
    'Zeta': 'Z',
    'Eta': 'H',
    'Theta': r'\Theta',
    'Iota': 'I',
    'Kappa': 'K',
    'Lambda': r'\Lambda',
    'Mu': 'M',
    'Nu': 'N',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': 'O',
    'Pi': r'\Pi',
    'Rho': 'P',
    'Sigma': r'\Sigma',
    'Tau': 'T',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': 'X',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'X',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = {'aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp'}

# Variable name modifiers
```
### 8 - sympy/physics/secondquant.py:

Start line: 1764, End line: 1802

```python
class Commutator(Function):

    def doit(self, **hints):
        """
        Enables the computation of complex expressions.

        Examples
        ========

        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy import symbols
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> c = Commutator(Fd(a)*F(i),Fd(b)*F(j))
        >>> c.doit(wicks=True)
        0
        """
        a = self.args[0]
        b = self.args[1]

        if hints.get("wicks"):
            a = a.doit(**hints)
            b = b.doit(**hints)
            try:
                return wicks(a*b) - wicks(b*a)
            except ContractionAppliesOnlyToFermions:
                pass
            except WicksTheoremDoesNotApply:
                pass

        return (a*b - b*a).doit(**hints)

    def __repr__(self):
        return "Commutator(%s,%s)" % (self.args[0], self.args[1])

    def __str__(self):
        return "[%s,%s]" % (self.args[0], self.args[1])

    def _latex(self, printer):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg) for arg in self.args])
```
### 9 - sympy/physics/quantum/cg.py:

Start line: 139, End line: 148

```python
class Wigner3j(Expr):

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j3,
                    self.m1, self.m2, self.m3))
        return r'\left(\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right)' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)
```
### 10 - sympy/printing/latex.py:

Start line: 867, End line: 964

```python
class LatexPrinter(Printer):

    def _print_Function(self, expr, exp=None):
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and \
                not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [str(self._print(arg)) for arg in expr.args]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = [
                "asin", "acos", "atan",
                "acsc", "asec", "acot",
                "asinh", "acosh", "atanh",
                "acsch", "asech", "acoth",
            ]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    pass
                elif inv_trig_style == "full":
                    func = "arc" + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                func_tex = self._hprint_Function(func)
                func_tex = self.parenthesize_super(func_tex)
                name = r'%s^{%s}' % (func_tex, exp)
            else:
                name = self._hprint_Function(func)

            if can_fold_brackets:
                if func in accepted_latex_functions:
                    # Wrap argument safely to avoid parse-time conflicts
                    # with the function name itself
                    name += r" {%s}"
                else:
                    name += r"%s"
            else:
                name += r"{\left(%s \right)}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}
```
### 22 - sympy/physics/secondquant.py:

Start line: 220, End line: 245

```python
class AntiSymmetricTensor(TensorSymbol):

    def _latex(self, printer):
        return "%s^{%s}_{%s}" % (
            self.symbol,
            "".join([ i.name for i in self.args[1]]),
            "".join([ i.name for i in self.args[2]])
        )

    @property
    def symbol(self):
        """
        Returns the symbol of the tensor.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).symbol
        v

        """
        return self.args[0]
```
### 32 - sympy/physics/secondquant.py:

Start line: 1657, End line: 1702

```python
class Commutator(Function):
    """
    The Commutator:  [A, B] = A*B - B*A

    The arguments are ordered according to .__cmp__()

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import Commutator
    >>> A, B = symbols('A,B', commutative=False)
    >>> Commutator(B, A)
    -Commutator(A, B)

    Evaluate the commutator with .doit()

    >>> comm = Commutator(A,B); comm
    Commutator(A, B)
    >>> comm.doit()
    A*B - B*A


    For two second quantization operators the commutator is evaluated
    immediately:

    >>> from sympy.physics.secondquant import Fd, F
    >>> a = symbols('a', above_fermi=True)
    >>> i = symbols('i', below_fermi=True)
    >>> p,q = symbols('p,q')

    >>> Commutator(Fd(a),Fd(i))
    2*NO(CreateFermion(a)*CreateFermion(i))

    But for more complicated expressions, the evaluation is triggered by
    a call to .doit()

    >>> comm = Commutator(Fd(p)*Fd(q),F(i)); comm
    Commutator(CreateFermion(p)*CreateFermion(q), AnnihilateFermion(i))
    >>> comm.doit(wicks=True)
    -KroneckerDelta(i, p)*CreateFermion(q) +
     KroneckerDelta(i, q)*CreateFermion(p)

    """

    is_commutative = False
```
### 35 - sympy/physics/secondquant.py:

Start line: 174, End line: 193

```python
class AntiSymmetricTensor(TensorSymbol):

    def __new__(cls, symbol, upper, lower):

        try:
            upper, signu = _sort_anticommuting_fermions(
                upper, key=cls._sortkey)
            lower, signl = _sort_anticommuting_fermions(
                lower, key=cls._sortkey)

        except ViolationOfPauliPrinciple:
            return S.Zero

        symbol = sympify(symbol)
        upper = Tuple(*upper)
        lower = Tuple(*lower)

        if (signu + signl) % 2:
            return -TensorSymbol.__new__(cls, symbol, upper, lower)
        else:

            return TensorSymbol.__new__(cls, symbol, upper, lower)
```
### 45 - sympy/physics/secondquant.py:

Start line: 1704, End line: 1762

```python
class Commutator(Function):

    @classmethod
    def eval(cls, a, b):
        """
        The Commutator [A,B] is on canonical form if A < B.

        Examples
        ========

        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy.abc import x
        >>> c1 = Commutator(F(x), Fd(x))
        >>> c2 = Commutator(Fd(x), F(x))
        >>> Commutator.eval(c1, c2)
        0
        """
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero

        #
        # [A+B,C]  ->  [A,C] + [B,C]
        #
        a = a.expand()
        if isinstance(a, Add):
            return Add(*[cls(term, b) for term in a.args])
        b = b.expand()
        if isinstance(b, Add):
            return Add(*[cls(a, term) for term in b.args])

        #
        # [xA,yB]  ->  xy*[A,B]
        #
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = list(ca) + list(cb)
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        #
        # single second quantization operators
        #
        if isinstance(a, BosonicOperator) and isinstance(b, BosonicOperator):
            if isinstance(b, CreateBoson) and isinstance(a, AnnihilateBoson):
                return KroneckerDelta(a.state, b.state)
            if isinstance(a, CreateBoson) and isinstance(b, AnnihilateBoson):
                return S.NegativeOne*KroneckerDelta(a.state, b.state)
            else:
                return S.Zero
        if isinstance(a, FermionicOperator) and isinstance(b, FermionicOperator):
            return wicks(a*b) - wicks(b*a)

        #
        # Canonical ordering of arguments
        #
        if a.sort_key() > b.sort_key():
            return S.NegativeOne*cls(b, a)
```
### 54 - sympy/physics/secondquant.py:

Start line: 2457, End line: 2566

```python
def substitute_dummies(expr, new_indices=False, pretty_indices={}):
    if new_indices:
        letters_above = pretty_indices.get('above', "")
        letters_below = pretty_indices.get('below', "")
        letters_general = pretty_indices.get('general', "")
        len_above = len(letters_above)
        len_below = len(letters_below)
        len_general = len(letters_general)

        def _i(number):
            try:
                return letters_below[number]
            except IndexError:
                return 'i_' + str(number - len_below)

        def _a(number):
            try:
                return letters_above[number]
            except IndexError:
                return 'a_' + str(number - len_above)

        def _p(number):
            try:
                return letters_general[number]
            except IndexError:
                return 'p_' + str(number - len_general)

    aboves = []
    belows = []
    generals = []

    dummies = expr.atoms(Dummy)
    if not new_indices:
        dummies = sorted(dummies, key=default_sort_key)

    # generate lists with the dummies we will insert
    a = i = p = 0
    for d in dummies:
        assum = d.assumptions0

        if assum.get("above_fermi"):
            if new_indices:
                sym = _a(a)
                a += 1
            l1 = aboves
        elif assum.get("below_fermi"):
            if new_indices:
                sym = _i(i)
                i += 1
            l1 = belows
        else:
            if new_indices:
                sym = _p(p)
                p += 1
            l1 = generals

        if new_indices:
            l1.append(Dummy(sym, **assum))
        else:
            l1.append(d)

    expr = expr.expand()
    terms = Add.make_args(expr)
    new_terms = []
    for term in terms:
        i = iter(belows)
        a = iter(aboves)
        p = iter(generals)
        ordered = _get_ordered_dummies(term)
        subsdict = {}
        for d in ordered:
            if d.assumptions0.get('below_fermi'):
                subsdict[d] = next(i)
            elif d.assumptions0.get('above_fermi'):
                subsdict[d] = next(a)
            else:
                subsdict[d] = next(p)
        subslist = []
        final_subs = []
        for k, v in subsdict.items():
            if k == v:
                continue
            if v in subsdict:
                # We check if the sequence of substitutions end quickly.  In
                # that case, we can avoid temporary symbols if we ensure the
                # correct substitution order.
                if subsdict[v] in subsdict:
                    # (x, y) -> (y, x),  we need a temporary variable
                    x = Dummy('x')
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    # (x, y) -> (y, a),  x->y must be done last
                    # but before temporary variables are resolved
                    final_subs.insert(0, (k, v))
            else:
                subslist.append((k, v))
        subslist.extend(final_subs)
        new_terms.append(term.subs(subslist))
    return Add(*new_terms)


class KeyPrinter(StrPrinter):
    """Printer for which only equal objects are equal in print"""
    def _print_Dummy(self, expr):
        return "(%s_%i)" % (expr.name, expr.dummy_index)


def __kprint(expr):
    p = KeyPrinter()
    return p.doprint(expr)
```
### 58 - sympy/physics/secondquant.py:

Start line: 81, End line: 105

```python
class Dagger(Expr):
    """
    Hermitian conjugate of creation/annihilation operators.

    Examples
    ========

    >>> from sympy import I
    >>> from sympy.physics.secondquant import Dagger, B, Bd
    >>> Dagger(2*I)
    -2*I
    >>> Dagger(B(0))
    CreateBoson(0)
    >>> Dagger(Bd(0))
    AnnihilateBoson(0)

    """

    def __new__(cls, arg):
        arg = sympify(arg)
        r = cls.eval(arg)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, arg)
        return obj
```
