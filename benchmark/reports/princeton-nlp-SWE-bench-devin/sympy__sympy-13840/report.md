# sympy__sympy-13840

| **sympy/sympy** | `8be967b5b2b81365c12030c41da68230e39cdf33` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/printing/rcode.py b/sympy/printing/rcode.py
--- a/sympy/printing/rcode.py
+++ b/sympy/printing/rcode.py
@@ -22,7 +22,6 @@
 known_functions = {
     #"Abs": [(lambda x: not x.is_integer, "fabs")],
     "Abs": "abs",
-    "gamma": "gamma",
     "sin": "sin",
     "cos": "cos",
     "tan": "tan",
@@ -42,6 +41,13 @@
     "floor": "floor",
     "ceiling": "ceiling",
     "sign": "sign",
+    "Max": "max",
+    "Min": "min",
+    "factorial": "factorial",
+    "gamma": "gamma",
+    "digamma": "digamma",
+    "trigamma": "trigamma",
+    "beta": "beta",
 }
 
 # These are the core reserved words in the R language. Taken from:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/rcode.py | 25 | 25 | - | - | -
| sympy/printing/rcode.py | 45 | 45 | - | - | -


## Problem Statement

```
Max & Min converting using SymPy
Why many languages likes js and R cannot be converted from Max & Min?
![image](https://user-images.githubusercontent.com/26391392/34533015-54ffb7d4-f086-11e7-945a-5708f6739d5d.png)


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/functions/elementary/miscellaneous.py | 332 | 368| 255 | 255 | 6446 | 
| 2 | 1 sympy/functions/elementary/miscellaneous.py | 642 | 728| 677 | 932 | 6446 | 
| 3 | 1 sympy/functions/elementary/miscellaneous.py | 742 | 757| 160 | 1092 | 6446 | 
| 4 | 2 sympy/parsing/maxima.py | 1 | 27| 175 | 1267 | 6968 | 
| 5 | 2 sympy/functions/elementary/miscellaneous.py | 475 | 503| 265 | 1532 | 6968 | 
| 6 | 2 sympy/functions/elementary/miscellaneous.py | 730 | 740| 132 | 1664 | 6968 | 
| 7 | 2 sympy/parsing/maxima.py | 29 | 47| 212 | 1876 | 6968 | 
| 8 | 2 sympy/functions/elementary/miscellaneous.py | 805 | 821| 159 | 2035 | 6968 | 
| 9 | 2 sympy/parsing/maxima.py | 50 | 71| 133 | 2168 | 6968 | 
| 10 | 2 sympy/functions/elementary/miscellaneous.py | 590 | 640| 825 | 2993 | 6968 | 
| 11 | 2 sympy/functions/elementary/miscellaneous.py | 793 | 803| 136 | 3129 | 6968 | 
| 12 | 2 sympy/functions/elementary/miscellaneous.py | 760 | 791| 240 | 3369 | 6968 | 
| 13 | 2 sympy/functions/elementary/miscellaneous.py | 1 | 31| 286 | 3655 | 6968 | 
| 14 | 3 sympy/core/sympify.py | 76 | 257| 1755 | 5410 | 10907 | 
| 15 | 4 sympy/__init__.py | 1 | 94| 691 | 6101 | 11598 | 
| 16 | 5 sympy/plotting/intervalmath/lib_interval.py | 201 | 220| 195 | 6296 | 15249 | 
| 17 | 5 sympy/functions/elementary/miscellaneous.py | 370 | 474| 843 | 7139 | 15249 | 
| 18 | 6 sympy/core/numbers.py | 1 | 36| 313 | 7452 | 44150 | 
| 19 | 6 sympy/functions/elementary/miscellaneous.py | 532 | 557| 197 | 7649 | 44150 | 
| 20 | 6 sympy/functions/elementary/miscellaneous.py | 559 | 588| 223 | 7872 | 44150 | 
| 21 | 6 sympy/core/numbers.py | 3794 | 3844| 307 | 8179 | 44150 | 
| 22 | 6 sympy/plotting/intervalmath/lib_interval.py | 180 | 198| 195 | 8374 | 44150 | 
| 23 | 6 sympy/core/numbers.py | 1338 | 1373| 277 | 8651 | 44150 | 
| 24 | 7 sympy/printing/latex.py | 82 | 117| 491 | 9142 | 66033 | 
| 25 | 8 sympy/parsing/sympy_parser.py | 630 | 688| 508 | 9650 | 73353 | 
| 26 | 9 sympy/calculus/util.py | 727 | 798| 356 | 10006 | 83547 | 
| 27 | 10 sympy/core/backend.py | 1 | 24| 357 | 10363 | 83905 | 
| 28 | 11 sympy/series/gruntz.py | 1 | 119| 1345 | 11708 | 90160 | 
| 29 | 12 sympy/printing/rust.py | 57 | 162| 1065 | 12773 | 95597 | 
| 30 | 13 sympy/external/importtools.py | 109 | 178| 698 | 13471 | 97264 | 
| 31 | 13 sympy/functions/elementary/miscellaneous.py | 505 | 530| 185 | 13656 | 97264 | 
| 32 | 14 sympy/polys/numberfields.py | 512 | 579| 709 | 14365 | 106191 | 
| 33 | 15 sympy/core/evalf.py | 1 | 50| 487 | 14852 | 119410 | 
| 34 | 16 sympy/concrete/expr_with_limits.py | 69 | 120| 532 | 15384 | 123020 | 
| 35 | 17 sympy/core/expr.py | 92 | 150| 483 | 15867 | 151704 | 
| 36 | 17 sympy/calculus/util.py | 705 | 725| 177 | 16044 | 151704 | 
| 37 | 17 sympy/core/sympify.py | 1 | 23| 173 | 16217 | 151704 | 
| 38 | 17 sympy/polys/numberfields.py | 702 | 749| 407 | 16624 | 151704 | 
| 39 | 18 sympy/polys/monomials.py | 212 | 232| 182 | 16806 | 156069 | 
| 40 | 19 sympy/benchmarks/bench_meijerint.py | 1 | 56| 685 | 17491 | 160571 | 
| 41 | 20 sympy/parsing/mathematica.py | 26 | 125| 776 | 18267 | 163472 | 
| 42 | 20 sympy/series/gruntz.py | 312 | 337| 275 | 18542 | 163472 | 
| 43 | 20 sympy/core/numbers.py | 1284 | 1302| 174 | 18716 | 163472 | 
| 44 | 20 sympy/core/numbers.py | 1090 | 1178| 752 | 19468 | 163472 | 
| 45 | 21 sympy/physics/units/__init__.py | 85 | 208| 973 | 20441 | 165387 | 
| 46 | 22 sympy/integrals/transforms.py | 299 | 312| 131 | 20572 | 181970 | 
| 47 | 23 sympy/physics/units/definitions.py | 1 | 53| 732 | 21304 | 185518 | 
| 48 | 23 sympy/core/numbers.py | 1321 | 1336| 158 | 21462 | 185518 | 
| 49 | 24 sympy/functions/elementary/piecewise.py | 1 | 16| 171 | 21633 | 194916 | 
| 50 | 24 sympy/core/numbers.py | 1304 | 1319| 158 | 21791 | 194916 | 


## Missing Patch Files

 * 1: sympy/printing/rcode.py

### Hint

```
I suppose these should be added, considering  JavaScript does have `Math.max` and `Math.min`. 

Meanwhile, there is a workaround: Max(x, y) is equivalent to `(x+y+Abs(x-y))/2`, and Abs is supported. 
\`\`\`
>>> jscode((1+y+Abs(1-y)) / 2)
'(1/2)*y + (1/2)*Math.abs(y - 1) + 1/2'
\`\`\`
Similarly, Min(x, y) is equivalent to (x+y-Abs(x-y))/2.
  
```

## Patch

```diff
diff --git a/sympy/printing/rcode.py b/sympy/printing/rcode.py
--- a/sympy/printing/rcode.py
+++ b/sympy/printing/rcode.py
@@ -22,7 +22,6 @@
 known_functions = {
     #"Abs": [(lambda x: not x.is_integer, "fabs")],
     "Abs": "abs",
-    "gamma": "gamma",
     "sin": "sin",
     "cos": "cos",
     "tan": "tan",
@@ -42,6 +41,13 @@
     "floor": "floor",
     "ceiling": "ceiling",
     "sign": "sign",
+    "Max": "max",
+    "Min": "min",
+    "factorial": "factorial",
+    "gamma": "gamma",
+    "digamma": "digamma",
+    "trigamma": "trigamma",
+    "beta": "beta",
 }
 
 # These are the core reserved words in the R language. Taken from:

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_rcode.py b/sympy/printing/tests/test_rcode.py
--- a/sympy/printing/tests/test_rcode.py
+++ b/sympy/printing/tests/test_rcode.py
@@ -1,7 +1,7 @@
 from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
                         GoldenRatio, EulerGamma, Catalan, Lambda, Dummy, Eq)
 from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
-                             gamma, sign, Max)
+                             gamma, sign, Max, Min, factorial, beta)
 from sympy.sets import Range
 from sympy.logic import ITE
 from sympy.codegen import For, aug_assign, Assignment
@@ -82,6 +82,8 @@ def test_rcode_Integer():
 
 def test_rcode_functions():
     assert rcode(sin(x) ** cos(x)) == "sin(x)^cos(x)"
+    assert rcode(factorial(x) + gamma(y)) == "factorial(x) + gamma(y)"
+    assert rcode(beta(Min(x, y), Max(x, y))) == "beta(min(x, y), max(x, y))"
 
 
 def test_rcode_inline_function():

```


## Code snippets

### 1 - sympy/functions/elementary/miscellaneous.py:

Start line: 332, End line: 368

```python
###############################################################################
############################# MINIMUM and MAXIMUM #############################
###############################################################################


class MinMaxBase(Expr, LatticeOp):
    def __new__(cls, *args, **assumptions):
        if not args:
            raise ValueError("The Max/Min functions must have arguments.")

        args = (sympify(arg) for arg in args)

        # first standard filter, for cls.zero and cls.identity
        # also reshape Max(a, Max(b, c)) to Max(a, b, c)
        try:
            args = frozenset(cls._new_args_filter(args))
        except ShortCircuit:
            return cls.zero

        if assumptions.pop('evaluate', True):
            # remove redundant args that are easily identified
            args = cls._collapse_arguments(args, **assumptions)

        # find local zeros
        args = cls._find_localzeros(args, **assumptions)

        if not args:
            return cls.identity

        if len(args) == 1:
            return list(args).pop()

        # base creation
        _args = frozenset(args)
        obj = Expr.__new__(cls, _args, **assumptions)
        obj._argset = _args
        return obj
```
### 2 - sympy/functions/elementary/miscellaneous.py:

Start line: 642, End line: 728

```python
class Max(MinMaxBase, Application):
    """
    Return, if possible, the maximum value of the list.

    When number of arguments is equal one, then
    return this argument.

    When number of arguments is equal two, then
    return, if possible, the value from (a, b) that is >= the other.

    In common case, when the length of list greater than 2, the task
    is more complicated. Return only the arguments, which are greater
    than others, if it is possible to determine directional relation.

    If is not possible to determine such a relation, return a partially
    evaluated result.

    Assumptions are used to make the decision too.

    Also, only comparable arguments are permitted.

    It is named ``Max`` and not ``max`` to avoid conflicts
    with the built-in function ``max``.


    Examples
    ========

    >>> from sympy import Max, Symbol, oo
    >>> from sympy.abc import x, y
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Max(x, -2)                  #doctest: +SKIP
    Max(x, -2)
    >>> Max(x, -2).subs(x, 3)
    3
    >>> Max(p, -2)
    p
    >>> Max(x, y)
    Max(x, y)
    >>> Max(x, y) == Max(y, x)
    True
    >>> Max(x, Max(y, z))           #doctest: +SKIP
    Max(x, y, z)
    >>> Max(n, 8, p, 7, -oo)        #doctest: +SKIP
    Max(8, p)
    >>> Max (1, x, oo)
    oo

    * Algorithm

    The task can be considered as searching of supremums in the
    directed complete partial orders [1]_.

    The source values are sequentially allocated by the isolated subsets
    in which supremums are searched and result as Max arguments.

    If the resulted supremum is single, then it is returned.

    The isolated subsets are the sets of values which are only the comparable
    with each other in the current set. E.g. natural numbers are comparable with
    each other, but not comparable with the `x` symbol. Another example: the
    symbol `x` with negative assumption is comparable with a natural number.

    Also there are "least" elements, which are comparable with all others,
    and have a zero property (maximum or minimum for all elements). E.g. `oo`.
    In case of it the allocation operation is terminated and only this value is
    returned.

    Assumption:
       - if A > B > C then A > C
       - if A == B then B can be removed

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Directed_complete_partial_order
    .. [2] http://en.wikipedia.org/wiki/Lattice_%28order%29

    See Also
    ========

    Min : find minimum values
    """
    zero = S.Infinity
    identity = S.NegativeInfinity
```
### 3 - sympy/functions/elementary/miscellaneous.py:

Start line: 742, End line: 757

```python
class Max(MinMaxBase, Application):

    def _eval_rewrite_as_Heaviside(self, *args):
        from sympy import Heaviside
        return Add(*[j*Mul(*[Heaviside(j - i) for i in args if i!=j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args):
        return _minmax_as_Piecewise('>=', *args)

    def _eval_is_positive(self):
        return fuzzy_or(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        return fuzzy_or(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        return fuzzy_and(a.is_negative for a in self.args)
```
### 4 - sympy/parsing/maxima.py:

Start line: 1, End line: 27

```python
from __future__ import print_function, division

import re
from sympy import sympify, Sum, product, sin, cos


class MaximaHelpers:
    def maxima_expand(expr):
        return expr.expand()

    def maxima_float(expr):
        return expr.evalf()

    def maxima_trigexpand(expr):
        return expr.expand(trig=True)

    def maxima_sum(a1, a2, a3, a4):
        return Sum(a1, (a2, a3, a4)).doit()

    def maxima_product(a1, a2, a3, a4):
        return product(a1, (a2, a3, a4))

    def maxima_csc(expr):
        return 1/sin(expr)

    def maxima_sec(expr):
        return 1/cos(expr)
```
### 5 - sympy/functions/elementary/miscellaneous.py:

Start line: 475, End line: 503

```python
class MinMaxBase(Expr, LatticeOp):

    @classmethod
    def _collapse_arguments(cls, args, **assumptions):
        # Min(Max(x, y), Max(x, z)) -> Max(x, Min(y, z))
        # and vice versa when swapping Min/Max -- do this only for the
        # easy case where all functions contain something in common;
        # trying to find some optimal subset of args to modify takes
        # too long
        if len(args) > 1:
            common = None
            remove = []
            sets = []
            for i in range(len(args)):
                a = args[i]
                if not isinstance(a, other):
                    continue
                s = set(a.args)
                common = s if common is None else (common & s)
                if not common:
                    break
                sets.append(s)
                remove.append(i)
            if common:
                sets = filter(None, [s - common for s in sets])
                sets = [other(*s, evaluate=False) for s in sets]
                for i in reversed(remove):
                    args.pop(i)
                oargs = [cls(*sets)] if sets else []
                oargs.extend(common)
                args.append(other(*oargs, evaluate=False))

        return args
```
### 6 - sympy/functions/elementary/miscellaneous.py:

Start line: 730, End line: 740

```python
class Max(MinMaxBase, Application):

    def fdiff( self, argindex ):
        from sympy import Heaviside
        n = len(self.args)
        if 0 < argindex and argindex <= n:
            argindex -= 1
            if n == 2:
                return Heaviside(self.args[argindex] - self.args[1 - argindex])
            newargs = tuple([self.args[i] for i in range(n) if i != argindex])
            return Heaviside(self.args[argindex] - Max(*newargs))
        else:
            raise ArgumentIndexError(self, argindex)
```
### 7 - sympy/parsing/maxima.py:

Start line: 29, End line: 47

```python
sub_dict = {
    'pi': re.compile(r'%pi'),
    'E': re.compile(r'%e'),
    'I': re.compile(r'%i'),
    '**': re.compile(r'\^'),
    'oo': re.compile(r'\binf\b'),
    '-oo': re.compile(r'\bminf\b'),
    "'-'": re.compile(r'\bminus\b'),
    'maxima_expand': re.compile(r'\bexpand\b'),
    'maxima_float': re.compile(r'\bfloat\b'),
    'maxima_trigexpand': re.compile(r'\btrigexpand'),
    'maxima_sum': re.compile(r'\bsum\b'),
    'maxima_product': re.compile(r'\bproduct\b'),
    'cancel': re.compile(r'\bratsimp\b'),
    'maxima_csc': re.compile(r'\bcsc\b'),
    'maxima_sec': re.compile(r'\bsec\b')
}

var_name = re.compile(r'^\s*(\w+)\s*:')
```
### 8 - sympy/functions/elementary/miscellaneous.py:

Start line: 805, End line: 821

```python
class Min(MinMaxBase, Application):

    def _eval_rewrite_as_Heaviside(self, *args):
        from sympy import Heaviside
        return Add(*[j*Mul(*[Heaviside(i-j) for i in args if i!=j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args):
        return _minmax_as_Piecewise('<=', *args)

    def _eval_is_positive(self):
        return fuzzy_and(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        return fuzzy_and(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        return fuzzy_or(a.is_negative for a in self.args)
```
### 9 - sympy/parsing/maxima.py:

Start line: 50, End line: 71

```python
def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()
    str = str.rstrip('; ')

    for k, v in sub_dict.items():
        str = v.sub(k, str)

    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()

    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)

    if assign_var and globals:
        globals[assign_var] = obj

    return obj
```
### 10 - sympy/functions/elementary/miscellaneous.py:

Start line: 590, End line: 640

```python
class MinMaxBase(Expr, LatticeOp):

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da is S.Zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = Function.fdiff(self, i)
            l.append(df * da)
        return Add(*l)

    def _eval_rewrite_as_Abs(self, *args):
        from sympy.functions.elementary.complexes import Abs
        s = (args[0] + self.func(*args[1:]))/2
        d = abs(args[0] - self.func(*args[1:]))/2
        return (s + d if isinstance(self, Max) else s - d).rewrite(Abs)

    def evalf(self, prec=None, **options):
        return self.func(*[a.evalf(prec, **options) for a in self.args])
    n = evalf

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)
    _eval_is_antihermitian = lambda s: _torf(i.is_antihermitian for i in s.args)
    _eval_is_commutative = lambda s: _torf(i.is_commutative for i in s.args)
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)
    _eval_is_nonnegative = lambda s: _torf(i.is_nonnegative for i in s.args)
    _eval_is_nonpositive = lambda s: _torf(i.is_nonpositive for i in s.args)
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)
    _eval_is_transcendental = lambda s: _torf(i.is_transcendental for i in s.args)
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)
```
