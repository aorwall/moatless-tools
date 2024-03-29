# sympy__sympy-15345

| **sympy/sympy** | `9ef28fba5b4d6d0168237c9c005a550e6dc27d81` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11252 |
| **Any found context length** | 1370 |
| **Avg pos** | 26.0 |
| **Min pos** | 4 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/mathematica.py b/sympy/printing/mathematica.py
--- a/sympy/printing/mathematica.py
+++ b/sympy/printing/mathematica.py
@@ -31,7 +31,8 @@
     "asech": [(lambda x: True, "ArcSech")],
     "acsch": [(lambda x: True, "ArcCsch")],
     "conjugate": [(lambda x: True, "Conjugate")],
-
+    "Max": [(lambda *x: True, "Max")],
+    "Min": [(lambda *x: True, "Min")],
 }
 
 
@@ -101,6 +102,8 @@ def _print_Function(self, expr):
                     return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
         return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")
 
+    _print_MinMaxBase = _print_Function
+
     def _print_Integral(self, expr):
         if len(expr.variables) == 1 and not expr.limits[0][1:]:
             args = [expr.args[0], expr.variables[0]]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/mathematica.py | 34 | 34 | 4 | 1 | 1370
| sympy/printing/mathematica.py | 104 | 104 | 22 | 1 | 11252


## Problem Statement

```
mathematica_code gives wrong output with Max
If I run the code

\`\`\`
x = symbols('x')
mathematica_code(Max(x,2))
\`\`\`

then I would expect the output `'Max[x,2]'` which is valid Mathematica code but instead I get `'Max(2, x)'` which is not valid Mathematica code.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/mathematica.py** | 120 | 132| 106 | 106 | 1228 | 
| 2 | 2 sympy/functions/elementary/miscellaneous.py | 645 | 731| 677 | 783 | 7858 | 
| 3 | 3 sympy/parsing/maxima.py | 1 | 27| 175 | 958 | 8380 | 
| **-> 4 <-** | **3 sympy/printing/mathematica.py** | 1 | 35| 412 | 1370 | 8380 | 
| 5 | 3 sympy/functions/elementary/miscellaneous.py | 745 | 762| 184 | 1554 | 8380 | 
| 6 | 3 sympy/functions/elementary/miscellaneous.py | 733 | 743| 132 | 1686 | 8380 | 
| 7 | 3 sympy/parsing/maxima.py | 50 | 71| 133 | 1819 | 8380 | 
| 8 | 4 sympy/printing/ccode.py | 740 | 876| 1518 | 3337 | 16377 | 
| 9 | 4 sympy/parsing/maxima.py | 29 | 47| 212 | 3549 | 16377 | 
| 10 | 5 sympy/parsing/mathematica.py | 26 | 125| 776 | 4325 | 19275 | 
| 11 | 5 sympy/functions/elementary/miscellaneous.py | 478 | 506| 265 | 4590 | 19275 | 
| 12 | 5 sympy/functions/elementary/miscellaneous.py | 335 | 371| 255 | 4845 | 19275 | 
| 13 | 6 sympy/series/gruntz.py | 340 | 349| 143 | 4988 | 25530 | 
| 14 | 6 sympy/series/gruntz.py | 312 | 337| 275 | 5263 | 25530 | 
| 15 | 7 sympy/printing/octave.py | 566 | 710| 1626 | 6889 | 32043 | 
| 16 | 8 sympy/codegen/cfunctions.py | 1 | 20| 142 | 7031 | 35134 | 
| 17 | 9 sympy/printing/jscode.py | 207 | 321| 1145 | 8176 | 37949 | 
| 18 | 10 sympy/polys/numberfields.py | 710 | 757| 407 | 8583 | 47048 | 
| 19 | 10 sympy/parsing/mathematica.py | 266 | 330| 428 | 9011 | 47048 | 
| 20 | 11 sympy/printing/rcode.py | 307 | 420| 1239 | 10250 | 50772 | 
| 21 | 11 sympy/printing/ccode.py | 683 | 716| 292 | 10542 | 50772 | 
| **-> 22 <-** | **11 sympy/printing/mathematica.py** | 38 | 117| 710 | 11252 | 50772 | 
| 23 | 12 sympy/polys/monomials.py | 236 | 256| 182 | 11434 | 55412 | 
| 24 | 13 sympy/printing/fcode.py | 775 | 901| 1361 | 12795 | 63333 | 
| 25 | 14 sympy/integrals/meijerint.py | 1011 | 1052| 628 | 13423 | 87615 | 
| 26 | 15 sympy/codegen/rewriting.py | 126 | 162| 398 | 13821 | 89469 | 
| 27 | 16 sympy/printing/julia.py | 491 | 633| 1597 | 15418 | 95228 | 
| 28 | 16 sympy/printing/ccode.py | 418 | 437| 251 | 15669 | 95228 | 
| 29 | 17 sympy/core/expr.py | 3406 | 3434| 242 | 15911 | 124251 | 
| 30 | 18 sympy/simplify/hyperexpand.py | 84 | 96| 160 | 16071 | 148974 | 
| 31 | 19 sympy/core/power.py | 157 | 239| 1060 | 17131 | 163557 | 
| 32 | 19 sympy/printing/fcode.py | 254 | 295| 352 | 17483 | 163557 | 
| 33 | 19 sympy/integrals/meijerint.py | 1692 | 1726| 342 | 17825 | 163557 | 
| 34 | 19 sympy/series/gruntz.py | 409 | 441| 329 | 18154 | 163557 | 
| 35 | 19 sympy/integrals/meijerint.py | 987 | 1010| 442 | 18596 | 163557 | 
| 36 | 20 sympy/benchmarks/bench_meijerint.py | 1 | 56| 685 | 19281 | 168059 | 
| 37 | 20 sympy/parsing/mathematica.py | 370 | 390| 181 | 19462 | 168059 | 
| 38 | 20 sympy/functions/elementary/miscellaneous.py | 593 | 643| 828 | 20290 | 168059 | 
| 39 | 20 sympy/core/expr.py | 2891 | 2914| 186 | 20476 | 168059 | 
| 40 | 20 sympy/printing/octave.py | 478 | 522| 496 | 20972 | 168059 | 
| 41 | 20 sympy/printing/octave.py | 1 | 59| 662 | 21634 | 168059 | 
| 42 | 21 sympy/polys/densearith.py | 1702 | 1741| 226 | 21860 | 180059 | 
| 43 | 21 sympy/integrals/meijerint.py | 1290 | 1306| 231 | 22091 | 180059 | 
| 44 | 21 sympy/parsing/mathematica.py | 392 | 420| 201 | 22292 | 180059 | 
| 45 | 22 sympy/printing/mathml.py | 246 | 286| 345 | 22637 | 187319 | 
| 46 | 22 sympy/functions/elementary/miscellaneous.py | 810 | 828| 183 | 22820 | 187319 | 
| 47 | 23 sympy/plotting/intervalmath/lib_interval.py | 201 | 220| 195 | 23015 | 190970 | 
| 48 | 23 sympy/printing/jscode.py | 1 | 43| 314 | 23329 | 190970 | 
| 49 | 24 sympy/printing/cxxcode.py | 79 | 108| 285 | 23614 | 192600 | 
| 50 | 25 sympy/printing/pycode.py | 306 | 334| 270 | 23884 | 197720 | 


### Hint

```
Hi, I'm new (to the project and development in general, but I'm a long time Mathematica user) and have been looking into this problem.

The `mathematica.py` file goes thru a table of known functions (of which neither Mathematica `Max` or `Min` functions are in) that are specified with lowercase capitalization, so it might be that doing `mathematica_code(Max(x,2))` is just yielding the unevaluated expression of `mathematica_code`. But there is a problem when I do `mathematica_code(max(x,2))` I get an error occurring in the Relational class in `core/relational.py`

Still checking it out, though.
`max` (lowercase `m`) is the Python builtin which tries to compare the items directly and give a result. Since `x` and `2` cannot be compared, you get an error. `Max` is the SymPy version that can return unevaluated results. 
```

## Patch

```diff
diff --git a/sympy/printing/mathematica.py b/sympy/printing/mathematica.py
--- a/sympy/printing/mathematica.py
+++ b/sympy/printing/mathematica.py
@@ -31,7 +31,8 @@
     "asech": [(lambda x: True, "ArcSech")],
     "acsch": [(lambda x: True, "ArcCsch")],
     "conjugate": [(lambda x: True, "Conjugate")],
-
+    "Max": [(lambda *x: True, "Max")],
+    "Min": [(lambda *x: True, "Min")],
 }
 
 
@@ -101,6 +102,8 @@ def _print_Function(self, expr):
                     return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
         return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")
 
+    _print_MinMaxBase = _print_Function
+
     def _print_Integral(self, expr):
         if len(expr.variables) == 1 and not expr.limits[0][1:]:
             args = [expr.args[0], expr.variables[0]]

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_mathematica.py b/sympy/printing/tests/test_mathematica.py
--- a/sympy/printing/tests/test_mathematica.py
+++ b/sympy/printing/tests/test_mathematica.py
@@ -2,7 +2,7 @@
                         Rational, Integer, Tuple, Derivative)
 from sympy.integrals import Integral
 from sympy.concrete import Sum
-from sympy.functions import exp, sin, cos, conjugate
+from sympy.functions import exp, sin, cos, conjugate, Max, Min
 
 from sympy import mathematica_code as mcode
 
@@ -28,6 +28,7 @@ def test_Function():
     assert mcode(f(x, y, z)) == "f[x, y, z]"
     assert mcode(sin(x) ** cos(x)) == "Sin[x]^Cos[x]"
     assert mcode(conjugate(x)) == "Conjugate[x]"
+    assert mcode(Max(x,y,z)*Min(y,z)) == "Max[x, y, z]*Min[y, z]"
 
 
 def test_Pow():

```


## Code snippets

### 1 - sympy/printing/mathematica.py:

Start line: 120, End line: 132

```python
def mathematica_code(expr, **settings):
    r"""Converts an expr to a string of the Wolfram Mathematica code

    Examples
    ========

    >>> from sympy import mathematica_code as mcode, symbols, sin
    >>> x = symbols('x')
    >>> mcode(sin(x).series(x).removeO())
    '(1/120)*x^5 - 1/6*x^3 + x'
    """
    return MCodePrinter(settings).doprint(expr)
```
### 2 - sympy/functions/elementary/miscellaneous.py:

Start line: 645, End line: 731

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
### 3 - sympy/parsing/maxima.py:

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
### 4 - sympy/printing/mathematica.py:

Start line: 1, End line: 35

```python
"""
Mathematica code printer
"""

from __future__ import print_function, division
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence

# Used in MCodePrinter._print_Function(self)
known_functions = {
    "exp": [(lambda x: True, "Exp")],
    "log": [(lambda x: True, "Log")],
    "sin": [(lambda x: True, "Sin")],
    "cos": [(lambda x: True, "Cos")],
    "tan": [(lambda x: True, "Tan")],
    "cot": [(lambda x: True, "Cot")],
    "asin": [(lambda x: True, "ArcSin")],
    "acos": [(lambda x: True, "ArcCos")],
    "atan": [(lambda x: True, "ArcTan")],
    "sinh": [(lambda x: True, "Sinh")],
    "cosh": [(lambda x: True, "Cosh")],
    "tanh": [(lambda x: True, "Tanh")],
    "coth": [(lambda x: True, "Coth")],
    "sech": [(lambda x: True, "Sech")],
    "csch": [(lambda x: True, "Csch")],
    "asinh": [(lambda x: True, "ArcSinh")],
    "acosh": [(lambda x: True, "ArcCosh")],
    "atanh": [(lambda x: True, "ArcTanh")],
    "acoth": [(lambda x: True, "ArcCoth")],
    "asech": [(lambda x: True, "ArcSech")],
    "acsch": [(lambda x: True, "ArcCsch")],
    "conjugate": [(lambda x: True, "Conjugate")],

}
```
### 5 - sympy/functions/elementary/miscellaneous.py:

Start line: 745, End line: 762

```python
class Max(MinMaxBase, Application):

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        from sympy import Heaviside
        return Add(*[j*Mul(*[Heaviside(j - i) for i in args if i!=j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        is_real = all(i.is_real for i in args)
        if is_real:
            return _minmax_as_Piecewise('>=', *args)

    def _eval_is_positive(self):
        return fuzzy_or(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        return fuzzy_or(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        return fuzzy_and(a.is_negative for a in self.args)
```
### 6 - sympy/functions/elementary/miscellaneous.py:

Start line: 733, End line: 743

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
### 8 - sympy/printing/ccode.py:

Start line: 740, End line: 876

```python
def ccode(expr, assign_to=None, standard='c99', **settings):
    """Converts an expr to a string of c code

    Parameters
    ==========

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    standard : str, optional
        String specifying the standard. If your compiler supports a more modern
        standard you may set this to 'c99' to allow the printer to use more math
        functions. [default='c89'].
    precision : integer, optional
        The precision for numbers such as pi [default=17].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)] or [(argument_test, cfunction_formater)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    '8*M_SQRT2*pow(tau, 7.0/2.0)'
    >>> ccode(expr, math_macros={})
    '8*sqrt(2)*pow(tau, 7.0/2.0)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    '8*M_SQRT2l*powl(tau, 7.0L/2.0L)'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> ccode(func(Abs(x) + ceiling(x)), standard='C89', user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the C-function takes a subset of the original arguments:

    >>> ccode(2**x + 3**x, standard='C99', user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard='C89'))
    if (x > 0) {
    tau = x + 1;
    }
    else {
    tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard='C89')
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(ccode(mat, A, standard='C89'))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    """
    return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)


def print_ccode(expr, **settings):
    """Prints C representation of the given expression."""
    print(ccode(expr, **settings))
```
### 9 - sympy/parsing/maxima.py:

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
### 10 - sympy/parsing/mathematica.py:

Start line: 26, End line: 125

```python
@_deco
class MathematicaParser(object):
    '''An instance of this class converts a string of a basic Mathematica
    expression to SymPy style. Output is string type.'''

    # left: Mathematica, right: SymPy
    CORRESPONDENCES = {
        'Sqrt[x]': 'sqrt(x)',
        'Exp[x]': 'exp(x)',
        'Log[x]': 'log(x)',
        'Log[x,y]': 'log(y,x)',
        'Log2[x]': 'log(x,2)',
        'Log10[x]': 'log(x,10)',
        'Mod[x,y]': 'Mod(x,y)',
        'Max[*x]': 'Max(*x)',
        'Min[*x]': 'Min(*x)',
    }

    # trigonometric, e.t.c.
    for arc, tri, h in product(('', 'Arc'), (
            'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc'), ('', 'h')):
        fm = arc + tri + h + '[x]'
        if arc:  # arc func
            fs = 'a' + tri.lower() + h + '(x)'
        else:    # non-arc func
            fs = tri.lower() + h + '(x)'
        CORRESPONDENCES.update({fm: fs})

    REPLACEMENTS = {
        ' ': '',
        '^': '**',
        '{': '[',
        '}': ']',
    }

    RULES = {
        # a single whitespace to '*'
        'whitespace': (
            re.compile(r'''
                (?<=[a-zA-Z\d])     # a letter or a number
                \                   # a whitespace
                (?=[a-zA-Z\d])      # a letter or a number
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character
        'add*_1': (
            re.compile(r'''
                (?<=[])\d])         # ], ) or a number
                                    # ''
                (?=[(a-zA-Z])       # ( or a single letter
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character (variable letter preceding)
        'add*_2': (
            re.compile(r'''
                (?<=[a-zA-Z])       # a letter
                \(                  # ( as a character
                (?=.)               # any characters
                ''', re.VERBOSE),
            '*('),

        # convert 'Pi' to 'pi'
        'Pi': (
            re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                Pi                  # 'Pi' is 3.14159... in Mathematica
                (?=[^a-zA-Z])
                ''', re.VERBOSE),
            'pi'),
    }

    # Mathematica function name pattern
    FM_PATTERN = re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])   # at the top or a non-letter
                )
                [A-Z][a-zA-Z\d]*    # Function
                (?=\[)              # [ as a character
                ''', re.VERBOSE)

    # list or matrix pattern (for future usage)
    ARG_MTRX_PATTERN = re.compile(r'''
                \{.*\}
                ''', re.VERBOSE)

    # regex string for function argument pattern
    ARGS_PATTERN_TEMPLATE = r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                {arguments}         # model argument like x, y,...
                (?=[^a-zA-Z])
                '''

    # will contain transformed CORRESPONDENCES dictionary
    TRANSLATIONS = {}
```
### 22 - sympy/printing/mathematica.py:

Start line: 38, End line: 117

```python
class MCodePrinter(CodePrinter):
    """A printer to convert python expressions to
    strings of the Wolfram's Mathematica code
    """
    printmethod = "_mcode"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': False,
    }

    _number_symbols = set()
    _not_supported = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
                self.known_functions.update(userfuncs)

    doprint = StrPrinter.doprint

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        PREC = precedence(expr)
        c, nc = expr.args_cnc()
        res = super(MCodePrinter, self)._print_Mul(expr.func(*c))
        if nc:
            res += '*'
            res += '**'.join(self.parenthesize(a, PREC) for a in nc)
        return res

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_Infinity(self, expr):
        return 'Infinity'

    def _print_NegativeInfinity(self, expr):
        return '-Infinity'

    def _print_list(self, expr):
        return '{' + ', '.join(self.doprint(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_mfunc = self.known_functions[expr.func.__name__]
            for cond, mfunc in cond_mfunc:
                if cond(*expr.args):
                    return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
        return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")

    def _print_Integral(self, expr):
        if len(expr.variables) == 1 and not expr.limits[0][1:]:
            args = [expr.args[0], expr.variables[0]]
        else:
            args = expr.args
        return "Hold[Integrate[" + ', '.join(self.doprint(a) for a in args) + "]]"

    def _print_Sum(self, expr):
        return "Hold[Sum[" + ', '.join(self.doprint(a) for a in expr.args) + "]]"

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return "Hold[D[" + ', '.join(self.doprint(a) for a in [dexpr] + dvars) + "]]"
```
