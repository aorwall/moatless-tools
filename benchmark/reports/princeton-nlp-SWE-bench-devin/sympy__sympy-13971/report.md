# sympy__sympy-13971

| **sympy/sympy** | `84c125972ad535b2dfb245f8d311d347b45e5b8a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1657,9 +1657,9 @@ def _print_SeqFormula(self, s):
         else:
             printset = tuple(s)
 
-        return (r"\left\["
+        return (r"\left["
               + r", ".join(self._print(el) for el in printset)
-              + r"\right\]")
+              + r"\right]")
 
     _print_SeqPer = _print_SeqFormula
     _print_SeqAdd = _print_SeqFormula

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 1660 | 1662 | - | 4 | -


## Problem Statement

```
Display of SeqFormula()
\`\`\`
import sympy as sp
k, m, n = sp.symbols('k m n', integer=True)
sp.init_printing()

sp.SeqFormula(n**2, (n,0,sp.oo))
\`\`\`

The Jupyter rendering of this command backslash-escapes the brackets producing:

`\left\[0, 1, 4, 9, \ldots\right\]`

Copying this output to a markdown cell this does not render properly.  Whereas:

`[0, 1, 4, 9, \ldots ]`

does render just fine.  

So - sequence output should not backslash-escape square brackets, or, `\]` should instead render?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/series/sequences.py | 602 | 640| 219 | 219 | 7317 | 
| 2 | 1 sympy/series/sequences.py | 642 | 656| 143 | 362 | 7317 | 
| 3 | 1 sympy/series/sequences.py | 658 | 679| 234 | 596 | 7317 | 
| 4 | 1 sympy/series/sequences.py | 681 | 711| 297 | 893 | 7317 | 
| 5 | 2 sympy/physics/quantum/qexpr.py | 28 | 52| 206 | 1099 | 10517 | 
| 6 | 3 sympy/printing/pretty/pretty.py | 1778 | 1798| 191 | 1290 | 30214 | 
| 7 | 3 sympy/physics/quantum/qexpr.py | 55 | 77| 155 | 1445 | 30214 | 
| 8 | **4 sympy/printing/latex.py** | 1647 | 1660| 144 | 1589 | 52097 | 
| 9 | 4 sympy/series/sequences.py | 181 | 199| 124 | 1713 | 52097 | 
| 10 | 4 sympy/series/sequences.py | 239 | 254| 113 | 1826 | 52097 | 
| 11 | 4 sympy/series/sequences.py | 274 | 294| 163 | 1989 | 52097 | 
| 12 | 5 sympy/printing/pycode.py | 391 | 411| 195 | 2184 | 56152 | 
| 13 | 5 sympy/series/sequences.py | 408 | 455| 253 | 2437 | 56152 | 
| 14 | 5 sympy/series/sequences.py | 944 | 976| 226 | 2663 | 56152 | 
| 15 | 5 sympy/printing/pretty/pretty.py | 1800 | 1820| 158 | 2821 | 56152 | 
| 16 | 5 sympy/printing/pretty/pretty.py | 750 | 769| 207 | 3028 | 56152 | 
| 17 | 5 sympy/physics/quantum/qexpr.py | 155 | 262| 806 | 3834 | 56152 | 
| 18 | 5 sympy/series/sequences.py | 912 | 942| 314 | 4148 | 56152 | 
| 19 | 5 sympy/printing/pycode.py | 260 | 282| 208 | 4356 | 56152 | 
| 20 | 5 sympy/series/sequences.py | 84 | 99| 124 | 4480 | 56152 | 
| 21 | 6 sympy/simplify/hyperexpand.py | 300 | 345| 683 | 5163 | 80875 | 
| 22 | 7 sympy/printing/mathml.py | 442 | 472| 182 | 5345 | 84508 | 
| 23 | 7 sympy/printing/pretty/pretty.py | 1255 | 1268| 142 | 5487 | 84508 | 
| 24 | 7 sympy/simplify/hyperexpand.py | 154 | 197| 771 | 6258 | 84508 | 
| 25 | 7 sympy/printing/pretty/pretty.py | 796 | 865| 597 | 6855 | 84508 | 
| 26 | 8 sympy/printing/str.py | 241 | 256| 142 | 6997 | 91209 | 
| 27 | **8 sympy/printing/latex.py** | 82 | 117| 491 | 7488 | 91209 | 
| 28 | 8 sympy/printing/mathml.py | 271 | 306| 310 | 7798 | 91209 | 
| 29 | 9 sympy/series/formal.py | 363 | 389| 235 | 8033 | 101025 | 
| 30 | **9 sympy/printing/latex.py** | 1662 | 1752| 805 | 8838 | 101025 | 
| 31 | 9 sympy/simplify/hyperexpand.py | 198 | 233| 686 | 9524 | 101025 | 
| 32 | 10 sympy/printing/fcode.py | 222 | 263| 352 | 9876 | 106555 | 
| 33 | **10 sympy/printing/latex.py** | 1949 | 2013| 754 | 10630 | 106555 | 
| 34 | 10 sympy/printing/mathml.py | 329 | 388| 454 | 11084 | 106555 | 
| 35 | 11 sympy/simplify/hyperexpand_doc.py | 1 | 19| 127 | 11211 | 106683 | 
| 36 | 11 sympy/printing/pretty/pretty.py | 867 | 942| 687 | 11898 | 106683 | 
| 37 | 11 sympy/series/sequences.py | 1 | 19| 175 | 12073 | 106683 | 
| 38 | 11 sympy/printing/mathml.py | 148 | 173| 208 | 12281 | 106683 | 
| 39 | 12 sympy/printing/pretty/pretty_symbology.py | 237 | 277| 603 | 12884 | 111957 | 
| 40 | 13 sympy/printing/julia.py | 481 | 623| 1597 | 14481 | 117593 | 
| 41 | 13 sympy/printing/julia.py | 118 | 181| 586 | 15067 | 117593 | 
| 42 | 14 sympy/printing/mathematica.py | 118 | 130| 106 | 15173 | 118795 | 
| 43 | 14 sympy/printing/pretty/pretty_symbology.py | 189 | 236| 772 | 15945 | 118795 | 
| 44 | 14 sympy/printing/mathml.py | 85 | 118| 257 | 16202 | 118795 | 
| 45 | 14 sympy/printing/pretty/pretty.py | 772 | 794| 224 | 16426 | 118795 | 
| 46 | 14 sympy/printing/julia.py | 362 | 380| 187 | 16613 | 118795 | 
| 47 | **14 sympy/printing/latex.py** | 1412 | 1443| 287 | 16900 | 118795 | 
| 48 | 14 sympy/series/sequences.py | 578 | 599| 202 | 17102 | 118795 | 
| 49 | 14 sympy/simplify/hyperexpand.py | 99 | 153| 758 | 17860 | 118795 | 
| 50 | 14 sympy/printing/pretty/pretty.py | 1747 | 1776| 261 | 18121 | 118795 | 
| 51 | 15 sympy/printing/octave.py | 127 | 190| 588 | 18709 | 124838 | 
| 52 | 16 sympy/physics/vector/printing.py | 343 | 380| 339 | 19048 | 128249 | 
| 53 | 16 sympy/printing/octave.py | 363 | 445| 719 | 19767 | 128249 | 
| 54 | 17 sympy/core/expr.py | 2624 | 2647| 307 | 20074 | 157010 | 
| 55 | 18 sympy/printing/jscode.py | 206 | 320| 1145 | 21219 | 159817 | 
| 56 | **18 sympy/printing/latex.py** | 1475 | 1522| 489 | 21708 | 159817 | 
| 57 | 18 sympy/series/sequences.py | 714 | 741| 214 | 21922 | 159817 | 
| 58 | 18 sympy/printing/mathml.py | 120 | 146| 229 | 22151 | 159817 | 
| 59 | 18 sympy/printing/pretty/pretty.py | 944 | 976| 354 | 22505 | 159817 | 
| 60 | 18 sympy/printing/julia.py | 343 | 359| 181 | 22686 | 159817 | 
| 61 | 18 sympy/printing/octave.py | 518 | 662| 1626 | 24312 | 159817 | 
| 62 | **18 sympy/printing/latex.py** | 1364 | 1389| 271 | 24583 | 159817 | 
| 63 | **18 sympy/printing/latex.py** | 481 | 532| 588 | 25171 | 159817 | 
| 64 | 18 sympy/printing/str.py | 514 | 575| 478 | 25649 | 159817 | 
| 65 | 19 sympy/core/mul.py | 176 | 234| 536 | 26185 | 174580 | 
| 66 | 19 sympy/printing/pretty/pretty.py | 722 | 747| 284 | 26469 | 174580 | 
| 67 | **19 sympy/printing/latex.py** | 1045 | 1059| 166 | 26635 | 174580 | 
| 68 | **19 sympy/printing/latex.py** | 1390 | 1410| 208 | 26843 | 174580 | 
| 69 | 19 sympy/core/mul.py | 905 | 941| 421 | 27264 | 174580 | 
| 70 | 20 sympy/printing/codeprinter.py | 65 | 121| 477 | 27741 | 178621 | 
| 71 | 20 sympy/core/mul.py | 568 | 626| 543 | 28284 | 178621 | 
| 72 | 20 sympy/printing/pycode.py | 414 | 432| 165 | 28449 | 178621 | 
| 73 | **20 sympy/printing/latex.py** | 708 | 775| 639 | 29088 | 178621 | 
| 74 | 20 sympy/series/sequences.py | 506 | 544| 333 | 29421 | 178621 | 
| 75 | 20 sympy/physics/quantum/qexpr.py | 1 | 25| 135 | 29556 | 178621 | 
| 76 | 21 sympy/core/sympify.py | 78 | 259| 1755 | 31311 | 182636 | 
| 77 | 21 sympy/core/mul.py | 235 | 364| 964 | 32275 | 182636 | 
| 78 | 22 sympy/printing/lambdarepr.py | 148 | 241| 742 | 33017 | 184594 | 
| 79 | **22 sympy/printing/latex.py** | 433 | 479| 522 | 33539 | 184594 | 
| 80 | 22 sympy/printing/pretty/pretty_symbology.py | 279 | 308| 230 | 33769 | 184594 | 
| 81 | 23 sympy/printing/glsl.py | 312 | 490| 1932 | 35701 | 189392 | 
| 82 | 23 sympy/printing/octave.py | 323 | 339| 183 | 35884 | 189392 | 
| 83 | 23 sympy/printing/str.py | 785 | 803| 111 | 35995 | 189392 | 
| 84 | 23 sympy/core/mul.py | 366 | 378| 164 | 36159 | 189392 | 
| 85 | 23 sympy/printing/pretty/pretty_symbology.py | 501 | 554| 426 | 36585 | 189392 | 
| 86 | **23 sympy/printing/latex.py** | 575 | 603| 239 | 36824 | 189392 | 
| 87 | 23 sympy/printing/julia.py | 204 | 246| 290 | 37114 | 189392 | 
| 88 | 23 sympy/printing/mathml.py | 1 | 42| 274 | 37388 | 189392 | 
| 89 | 23 sympy/simplify/hyperexpand.py | 378 | 387| 190 | 37578 | 189392 | 
| 90 | 23 sympy/simplify/hyperexpand.py | 266 | 299| 672 | 38250 | 189392 | 
| 91 | 23 sympy/printing/julia.py | 383 | 408| 238 | 38488 | 189392 | 
| 92 | 24 sympy/matrices/expressions/matexpr.py | 1 | 30| 229 | 38717 | 195460 | 
| 93 | 25 sympy/interactive/printing.py | 149 | 230| 698 | 39415 | 199242 | 


## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1657,9 +1657,9 @@ def _print_SeqFormula(self, s):
         else:
             printset = tuple(s)
 
-        return (r"\left\["
+        return (r"\left["
               + r", ".join(self._print(el) for el in printset)
-              + r"\right\]")
+              + r"\right]")
 
     _print_SeqPer = _print_SeqFormula
     _print_SeqAdd = _print_SeqFormula

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -614,46 +614,46 @@ def test_latex_sequences():
     s1 = SeqFormula(a**2, (0, oo))
     s2 = SeqPer((1, 2))
 
-    latex_str = r'\left\[0, 1, 4, 9, \ldots\right\]'
+    latex_str = r'\left[0, 1, 4, 9, \ldots\right]'
     assert latex(s1) == latex_str
 
-    latex_str = r'\left\[1, 2, 1, 2, \ldots\right\]'
+    latex_str = r'\left[1, 2, 1, 2, \ldots\right]'
     assert latex(s2) == latex_str
 
     s3 = SeqFormula(a**2, (0, 2))
     s4 = SeqPer((1, 2), (0, 2))
 
-    latex_str = r'\left\[0, 1, 4\right\]'
+    latex_str = r'\left[0, 1, 4\right]'
     assert latex(s3) == latex_str
 
-    latex_str = r'\left\[1, 2, 1\right\]'
+    latex_str = r'\left[1, 2, 1\right]'
     assert latex(s4) == latex_str
 
     s5 = SeqFormula(a**2, (-oo, 0))
     s6 = SeqPer((1, 2), (-oo, 0))
 
-    latex_str = r'\left\[\ldots, 9, 4, 1, 0\right\]'
+    latex_str = r'\left[\ldots, 9, 4, 1, 0\right]'
     assert latex(s5) == latex_str
 
-    latex_str = r'\left\[\ldots, 2, 1, 2, 1\right\]'
+    latex_str = r'\left[\ldots, 2, 1, 2, 1\right]'
     assert latex(s6) == latex_str
 
-    latex_str = r'\left\[1, 3, 5, 11, \ldots\right\]'
+    latex_str = r'\left[1, 3, 5, 11, \ldots\right]'
     assert latex(SeqAdd(s1, s2)) == latex_str
 
-    latex_str = r'\left\[1, 3, 5\right\]'
+    latex_str = r'\left[1, 3, 5\right]'
     assert latex(SeqAdd(s3, s4)) == latex_str
 
-    latex_str = r'\left\[\ldots, 11, 5, 3, 1\right\]'
+    latex_str = r'\left[\ldots, 11, 5, 3, 1\right]'
     assert latex(SeqAdd(s5, s6)) == latex_str
 
-    latex_str = r'\left\[0, 2, 4, 18, \ldots\right\]'
+    latex_str = r'\left[0, 2, 4, 18, \ldots\right]'
     assert latex(SeqMul(s1, s2)) == latex_str
 
-    latex_str = r'\left\[0, 2, 4\right\]'
+    latex_str = r'\left[0, 2, 4\right]'
     assert latex(SeqMul(s3, s4)) == latex_str
 
-    latex_str = r'\left\[\ldots, 18, 4, 2, 0\right\]'
+    latex_str = r'\left[\ldots, 18, 4, 2, 0\right]'
     assert latex(SeqMul(s5, s6)) == latex_str
 
 

```


## Code snippets

### 1 - sympy/series/sequences.py:

Start line: 602, End line: 640

```python
class SeqFormula(SeqExpr):
    """Represents sequence based on a formula.

    Elements are generated using a formula.

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    For value at a particular point

    >>> s.coeff(3)
    9

    supports slicing

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    iterable

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    sequence starts from negative infinity

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """
```
### 2 - sympy/series/sequences.py:

Start line: 642, End line: 656

```python
class SeqFormula(SeqExpr):

    def __new__(cls, formula, limits=None):
        formula = sympify(formula)

        def _find_x(formula):
            free = formula.free_symbols
            if len(formula.free_symbols) == 1:
                return free.pop()
            elif len(formula.free_symbols) == 0:
                return Dummy('k')
            else:
                raise ValueError(
                    " specify dummy variables for %s. If the formula contains"
                    " more than one free symbol, a dummy variable should be"
                    " supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
                    % formula)
        # ... other code
```
### 3 - sympy/series/sequences.py:

Start line: 658, End line: 679

```python
class SeqFormula(SeqExpr):

    def __new__(cls, formula, limits=None):
        # ... other code

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(formula), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value"
                                 "cannot be unbounded")
        limits = sympify((x, start, stop))

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, formula, limits)
```
### 4 - sympy/series/sequences.py:

Start line: 681, End line: 711

```python
class SeqFormula(SeqExpr):

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 * form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        formula = self.formula * coeff
        return SeqFormula(formula, self.args[1])
```
### 5 - sympy/physics/quantum/qexpr.py:

Start line: 28, End line: 52

```python
def _qsympify_sequence(seq):
    """Convert elements of a sequence to standard form.

    This is like sympify, but it performs special logic for arguments passed
    to QExpr. The following conversions are done:

    * (list, tuple, Tuple) => _qsympify_sequence each element and convert
      sequence to a Tuple.
    * basestring => Symbol
    * Matrix => Matrix
    * other => sympify

    Strings are passed to Symbol, not sympify to make sure that variables like
    'pi' are kept as Symbols, not the SymPy built-in number subclasses.

    Examples
    ========

    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence
    >>> _qsympify_sequence((1,2,[3,4,[1,]]))
    (1, 2, (3, 4, (1,)))

    """

    return tuple(__qsympify_sequence_helper(seq))
```
### 6 - sympy/printing/pretty/pretty.py:

Start line: 1778, End line: 1798

```python
class PrettyPrinter(Printer):

    def _print_SeqFormula(self, s):
        if self._use_unicode:
            dots = u"\N{HORIZONTAL ELLIPSIS}"
        else:
            dots = '...'

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
### 7 - sympy/physics/quantum/qexpr.py:

Start line: 55, End line: 77

```python
def __qsympify_sequence_helper(seq):
    """
       Helper function for _qsympify_sequence
       This function does the actual work.
    """
    #base case. If not a list, do Sympification
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq
        elif isinstance(seq, string_types):
            return Symbol(seq)
        else:
            return sympify(seq)

    # base condition, when seq is QExpr and also
    # is iterable.
    if isinstance(seq, QExpr):
        return seq

    #if list, recurse on each item in the list
    result = [__qsympify_sequence_helper(item) for item in seq]

    return Tuple(*result)
```
### 8 - sympy/printing/latex.py:

Start line: 1647, End line: 1660

```python
class LatexPrinter(Printer):

    def _print_SeqFormula(self, s):
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (r'\ldots', s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(r'\ldots')
        else:
            printset = tuple(s)

        return (r"\left\["
              + r", ".join(self._print(el) for el in printset)
              + r"\right\]")
```
### 9 - sympy/series/sequences.py:

Start line: 181, End line: 199

```python
class SeqBase(Basic):

    def coeff_mul(self, other):
        """
        Should be used when ``other`` is not a sequence. Should be
        defined to define custom behaviour.

        Examples
        ========

        >>> from sympy import S, oo, SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' defines multiplication of sequences with sequences only.
        """
        return Mul(self, other)
```
### 10 - sympy/series/sequences.py:

Start line: 239, End line: 254

```python
class SeqBase(Basic):

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """Negates the sequence.

        Examples
        ========

        >>> from sympy import S, oo, SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)
```
### 27 - sympy/printing/latex.py:

Start line: 82, End line: 117

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
### 30 - sympy/printing/latex.py:

Start line: 1662, End line: 1752

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
        return r"\langle %s, %s\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        return r" \cup ".join([self._print(i) for i in u.args])

    def _print_Complement(self, u):
        return r" \setminus ".join([self._print(i) for i in u.args])

    def _print_Intersection(self, u):
        return r" \cap ".join([self._print(i) for i in u.args])

    def _print_SymmetricDifference(self, u):
        return r" \triangle ".join([self._print(i) for i in u.args])

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        return r"\left\{%s\; |\; %s \in %s\right\}" % (
            self._print(s.lamda.expr),
            ', '.join([self._print(var) for var in s.lamda.variables]),
            self._print(s.base_set))

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        return r"\left\{%s\; |\; %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition.as_expr()))

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; |\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        return self._print_Add(s.truncate()) + self._print(r' + \ldots')

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"
```
### 33 - sympy/printing/latex.py:

Start line: 1949, End line: 2013

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{%s}^{%s}' % (self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ %s \right]" % ",".join(
            '{' + self._print(x) + '}' for x in m)

    def _print_SubModule(self, m):
        return r"\left< %s \right>" % ",".join(
            '{' + self._print(x) + '}' for x in m.gens)

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left< %s \right>" % ",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens)

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(R.ring), self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{%s} + {%s}" % (self._print(x.data), self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{%s} + {%s}" % (self._print(m.data),
                                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(M.base),
                                   self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{%s} : {%s} \to {%s}" % (self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\boldsymbol{\mathrm{%s}}' % self._print(Symbol(string))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\partial_{%s}' % self._print(Symbol(string))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return r'\mathrm{d}%s' % self._print(Symbol(string))
        else:
            return 'd(%s)' % self._print(field)
            string = self._print(field)
            return r'\mathrm{d}\left(%s\right)' % string

    def _print_Tr(self, p):
        #Todo: Handle indices
        contents = self._print(p.args[0])
        return r'\mbox{Tr}\left(%s\right)' % (contents)
```
### 47 - sympy/printing/latex.py:

Start line: 1412, End line: 1443

```python
class LatexPrinter(Printer):

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^T" % self._print(mat)
        else:
            return "%s^T" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)

    def _print_MatAdd(self, expr):
        terms = list(expr.args)
        tex = " + ".join(map(self._print, terms))
        return tex

    def _print_MatMul(self, expr):
        from sympy import Add, MatAdd, HadamardProduct

        def parens(x):
            if isinstance(x, (Add, MatAdd, HadamardProduct)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return ' '.join(map(parens, expr.args))
```
### 56 - sympy/printing/latex.py:

Start line: 1475, End line: 1522

```python
class LatexPrinter(Printer):

    def _print_NDimArray(self, expr):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                      r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str
```
### 62 - sympy/printing/latex.py:

Start line: 1364, End line: 1389

```python
class LatexPrinter(Printer):

    def _print_MatrixBase(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([ self._print(i) for i in expr[line, :] ]))

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str % r"\\".join(lines)
```
### 63 - sympy/printing/latex.py:

Start line: 481, End line: 532

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1:
            base = self._print(expr.base)
            expq = expr.exp.q

            if expq == 2:
                tex = r"\sqrt{%s}" % base
            elif self._settings['itex']:
                tex = r"\root{%d}{%s}" % (expq, base)
            else:
                tex = r"\sqrt[%d]{%s}" % (expq, base)

            if expr.exp.is_negative:
                return r"\frac{1}{%s}" % tex
            else:
                return tex
        elif self._settings['fold_frac_powers'] \
            and expr.exp.is_Rational \
                and expr.exp.q != 1:
            base, p, q = self.parenthesize(expr.base, PRECEDENCE['Pow']), expr.exp.p, expr.exp.q
            #fixes issue #12886, adds parentheses before superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, "%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
            # Things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, self._print(expr.exp))
            else:
                if expr.is_commutative and expr.exp == -1:
                    #solves issue 4129
                    #As Mul always simplify 1/x to x**-1
                    #The objective is achieved with this hack
                    #first we get the latex for -1 * expr,
                    #which is a Mul expression
                    tex = self._print(S.NegativeOne * expr).strip()
                    #the result comes with a minus and a space, so we remove
                    if tex[:1] == "-":
                        return tex[1:].strip()
                tex = r"%s^{%s}"
                #fixes issue #12886, adds parentheses before superscripts raised to powers
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if '^' in base and expr.base.is_Symbol:
                    base = r"\left(%s\right)" % base
                exp = self._print(expr.exp)

                return tex % (base, exp)
```
### 67 - sympy/printing/latex.py:

Start line: 1045, End line: 1059

```python
class LatexPrinter(Printer):

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)
```
### 68 - sympy/printing/latex.py:

Start line: 1390, End line: 1410

```python
class LatexPrinter(Printer):
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix \
                           = _print_Matrix \
                           = _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '_{%s, %s}' % (expr.i, expr.j)

    def _print_MatrixSlice(self, expr):
        def latexslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(self._print, x))
        return (self._print(expr.parent) + r'\left[' +
                latexslice(expr.rowslice) + ', ' +
                latexslice(expr.colslice) + r'\right]')
```
### 73 - sympy/printing/latex.py:

Start line: 708, End line: 775

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

        if hasattr(self, '_print_' + func):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [ str(self._print(arg)) for arg in expr.args ]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = ["asin", "acos", "atan", "acot"]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    func = func
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
                name = r'%s^{%s}' % (self._hprint_Function(func), exp)
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
                name += r"{\left (%s \right )}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)
```
### 79 - sympy/printing/latex.py:

Start line: 433, End line: 479

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        # ... other code

        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] \
                    and ldenom <= 2 and not "^" in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex
```
### 86 - sympy/printing/latex.py:

Start line: 575, End line: 603

```python
class LatexPrinter(Printer):

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        o1 = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = '(' + LatexPrinter().doprint(v) + ')'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr
```
