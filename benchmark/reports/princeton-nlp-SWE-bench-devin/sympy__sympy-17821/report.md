# sympy__sympy-17821

| **sympy/sympy** | `647a123703e0f5de659087bef860adc3cdf4f9b6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4973 |
| **Any found context length** | 4973 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -3829,6 +3829,13 @@ def approximation_interval(self, number_cls):
         elif issubclass(number_cls, Rational):
             return (Rational(9, 10), S.One)
 
+    def _eval_rewrite_as_Sum(self, k_sym=None, symbols=None):
+        from sympy import Sum, Dummy
+        if (k_sym is not None) or (symbols is not None):
+            return self
+        k = Dummy('k', integer=True, nonnegative=True)
+        return Sum((-1)**k / (2*k+1)**2, (k, 0, S.Infinity))
+
     def _sage_(self):
         import sage.all as sage
         return sage.catalan

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/numbers.py | 3832 | 3832 | 5 | 4 | 4973


## Problem Statement

```
Catalan rewrite and doctests for latex equations
First, implement `S.Catalan.rewrite(Sum)`.

Also, something I've been thinking about for while: we have lots of LaTeX in our docs.  In many cases we could generate those equations ourselves instead of typing them manually (I found errors while doing #11014 for example).

This PR should demonstrate the idea.  @asmeurer what do you think?  Will this work?  Its certainly nice for maintainance, although it is probably slightly less readable...

(If we want to do this widely, the latex printer could probably be optimized for things like `^{2}` and when it uses `\left(` instead of `(`.)

#### Release notes

<!-- BEGIN RELEASE NOTES -->
* core
  * Catalan can be rewritten as a sum
<!-- END RELEASE NOTES -->

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/plotting/experimental_lambdify.py | 1 | 76| 868 | 868 | 5903 | 
| 2 | 2 sympy/printing/latex.py | 2497 | 2674| 2535 | 3403 | 33423 | 
| 3 | 2 sympy/printing/latex.py | 2228 | 2301| 735 | 4138 | 33423 | 
| 4 | 3 sympy/parsing/latex/_antlr/latexparser.py | 590 | 636| 479 | 4617 | 63698 | 
| **-> 5 <-** | **4 sympy/core/numbers.py** | 3784 | 3834| 356 | 4973 | 93797 | 
| 6 | 4 sympy/printing/latex.py | 596 | 613| 196 | 5169 | 93797 | 
| 7 | 4 sympy/printing/latex.py | 974 | 1029| 577 | 5746 | 93797 | 
| 8 | 4 sympy/printing/latex.py | 1 | 83| 720 | 6466 | 93797 | 
| 9 | 4 sympy/printing/latex.py | 1055 | 1139| 813 | 7279 | 93797 | 
| 10 | 4 sympy/parsing/latex/_antlr/latexparser.py | 222 | 268| 778 | 8057 | 93797 | 
| 11 | 4 sympy/printing/latex.py | 1396 | 1425| 350 | 8407 | 93797 | 
| 12 | 4 sympy/printing/latex.py | 486 | 533| 529 | 8936 | 93797 | 
| 13 | 4 sympy/parsing/latex/_antlr/latexparser.py | 269 | 382| 947 | 9883 | 93797 | 
| 14 | 4 sympy/parsing/latex/_antlr/latexparser.py | 2237 | 2447| 1899 | 11782 | 93797 | 
| 15 | 4 sympy/printing/latex.py | 2347 | 2413| 734 | 12516 | 93797 | 
| 16 | 4 sympy/printing/latex.py | 866 | 946| 767 | 13283 | 93797 | 
| 17 | 4 sympy/printing/latex.py | 406 | 432| 306 | 13589 | 93797 | 
| 18 | 4 sympy/printing/latex.py | 1986 | 2055| 610 | 14199 | 93797 | 
| 19 | 4 sympy/printing/latex.py | 2460 | 2487| 218 | 14417 | 93797 | 
| 20 | 4 sympy/parsing/latex/_antlr/latexparser.py | 1347 | 1411| 607 | 15024 | 93797 | 
| 21 | 4 sympy/printing/latex.py | 2076 | 2123| 457 | 15481 | 93797 | 
| 22 | 4 sympy/printing/latex.py | 434 | 484| 385 | 15866 | 93797 | 
| 23 | 4 sympy/printing/latex.py | 1964 | 1984| 220 | 16086 | 93797 | 
| 24 | 4 sympy/printing/latex.py | 84 | 119| 491 | 16577 | 93797 | 
| 25 | 4 sympy/printing/latex.py | 2709 | 2776| 664 | 17241 | 93797 | 
| 26 | 5 sympy/parsing/latex/_parse_latex_antlr.py | 107 | 121| 111 | 17352 | 98182 | 
| 27 | 6 sympy/concrete/summations.py | 706 | 750| 457 | 17809 | 110053 | 
| 28 | 6 sympy/printing/latex.py | 535 | 576| 458 | 18267 | 110053 | 
| 29 | 6 sympy/printing/latex.py | 2490 | 2699| 106 | 18373 | 110053 | 
| 30 | 7 doc/src/conf.py | 107 | 202| 714 | 19087 | 112343 | 
| 31 | 7 sympy/parsing/latex/_antlr/latexparser.py | 2838 | 2858| 177 | 19264 | 112343 | 
| 32 | 7 sympy/parsing/latex/_antlr/latexparser.py | 387 | 436| 345 | 19609 | 112343 | 
| 33 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1897 | 1932| 288 | 19897 | 112343 | 
| 34 | 7 sympy/parsing/latex/_antlr/latexparser.py | 1937 | 1960| 205 | 20102 | 112343 | 
| 35 | 7 sympy/parsing/latex/_antlr/latexparser.py | 678 | 724| 542 | 20644 | 112343 | 
| 36 | 7 sympy/printing/latex.py | 1340 | 1394| 729 | 21373 | 112343 | 
| 37 | 8 sympy/functions/combinatorial/numbers.py | 1164 | 1206| 430 | 21803 | 130317 | 
| 38 | 8 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 57| 464 | 22267 | 130317 | 
| 39 | 8 sympy/parsing/latex/_antlr/latexparser.py | 2779 | 2811| 277 | 22544 | 130317 | 
| 40 | 8 sympy/parsing/latex/_antlr/latexparser.py | 2924 | 2957| 222 | 22766 | 130317 | 
| 41 | 8 sympy/parsing/latex/_antlr/latexparser.py | 440 | 486| 529 | 23295 | 130317 | 
| 42 | 8 sympy/parsing/latex/_antlr/latexparser.py | 2227 | 2439| 257 | 23552 | 130317 | 
| 43 | 8 sympy/parsing/latex/_parse_latex_antlr.py | 505 | 517| 139 | 23691 | 130317 | 
| 44 | 8 sympy/parsing/latex/_parse_latex_antlr.py | 201 | 213| 119 | 23810 | 130317 | 
| 45 | 9 sympy/parsing/latex/_build_latex_antlr.py | 40 | 89| 354 | 24164 | 130941 | 
| 46 | 9 sympy/printing/latex.py | 578 | 594| 199 | 24363 | 130941 | 
| 47 | 9 sympy/printing/latex.py | 664 | 697| 288 | 24651 | 130941 | 
| 48 | 9 sympy/printing/latex.py | 1282 | 1338| 751 | 25402 | 130941 | 
| 49 | 9 sympy/parsing/latex/_antlr/latexparser.py | 562 | 586| 173 | 25575 | 130941 | 
| 50 | 9 sympy/printing/latex.py | 2064 | 2074| 131 | 25706 | 130941 | 
| 51 | 9 sympy/parsing/latex/_parse_latex_antlr.py | 89 | 104| 127 | 25833 | 130941 | 
| 52 | 9 sympy/printing/latex.py | 774 | 841| 656 | 26489 | 130941 | 
| 53 | 9 sympy/printing/latex.py | 1865 | 1884| 228 | 26717 | 130941 | 
| 54 | 9 sympy/parsing/latex/_antlr/latexparser.py | 2716 | 2748| 277 | 26994 | 130941 | 


## Patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -3829,6 +3829,13 @@ def approximation_interval(self, number_cls):
         elif issubclass(number_cls, Rational):
             return (Rational(9, 10), S.One)
 
+    def _eval_rewrite_as_Sum(self, k_sym=None, symbols=None):
+        from sympy import Sum, Dummy
+        if (k_sym is not None) or (symbols is not None):
+            return self
+        k = Dummy('k', integer=True, nonnegative=True)
+        return Sum((-1)**k / (2*k+1)**2, (k, 0, S.Infinity))
+
     def _sage_(self):
         import sage.all as sage
         return sage.catalan

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -6,7 +6,7 @@
                    TribonacciConstant, cos, exp,
                    Number, zoo, log, Mul, Pow, Tuple, latex, Gt, Lt, Ge, Le,
                    AlgebraicNumber, simplify, sin, fibonacci, RealField,
-                   sympify, srepr)
+                   sympify, srepr, Dummy, Sum)
 from sympy.core.compatibility import long
 from sympy.core.expr import unchanged
 from sympy.core.logic import fuzzy_not
@@ -1674,6 +1674,11 @@ def test_Catalan_EulerGamma_prec():
     assert f._prec == 20
     assert n._as_mpf_val(20) == f._mpf_
 
+def test_Catalan_rewrite():
+    k = Dummy('k', integer=True, nonnegative=True)
+    assert Catalan.rewrite(Sum).dummy_eq(
+            Sum((-1)**k/(2*k + 1)**2, (k, 0, oo)))
+    assert Catalan.rewrite() == Catalan
 
 def test_bool_eq():
     assert 0 == False

```


## Code snippets

### 1 - sympy/plotting/experimental_lambdify.py:

Start line: 1, End line: 76

```python
""" rewrite of lambdify - This stuff is not stable at all.

It is for internal use in the new plotting module.
It may (will! see the Q'n'A in the source) be rewritten.

It's completely self contained. Especially it does not use lambdarepr.

It does not aim to replace the current lambdify. Most importantly it will never
ever support anything else than sympy expressions (no Matrices, dictionaries
and so on).
"""

from __future__ import print_function, division

import re
from sympy import Symbol, NumberSymbol, I, zoo, oo
from sympy.core.compatibility import exec_, string_types
from sympy.utilities.iterables import numbered_symbols

#  We parse the expression string into a tree that identifies functions. Then
# we translate the names of the functions and we translate also some strings
# that are not names of functions (all this according to translation
# dictionaries).
#  If the translation goes to another module (like numpy) the
# module is imported and 'func' is translated to 'module.func'.
#  If a function can not be translated, the inner nodes of that part of the
# tree are not translated. So if we have Integral(sqrt(x)), sqrt is not
# translated to np.sqrt and the Integral does not crash.
#  A namespace for all this is generated by crawling the (func, args) tree of
# the expression. The creation of this namespace involves many ugly
# workarounds.
#  The namespace consists of all the names needed for the sympy expression and
# all the name of modules used for translation. Those modules are imported only
# as a name (import numpy as np) in order to keep the namespace small and
# manageable.

#  Please, if there is a bug, do not try to fix it here! Rewrite this by using
# the method proposed in the last Q'n'A below. That way the new function will
# work just as well, be just as simple, but it wont need any new workarounds.
#  If you insist on fixing it here, look at the workarounds in the function
# sympy_expression_namespace and in lambdify.

# Q: Why are you not using python abstract syntax tree?
# A: Because it is more complicated and not much more powerful in this case.

# Q: What if I have Symbol('sin') or g=Function('f')?
# A: You will break the algorithm. We should use srepr to defend against this?
#  The problem with Symbol('sin') is that it will be printed as 'sin'. The
# parser will distinguish it from the function 'sin' because functions are
# detected thanks to the opening parenthesis, but the lambda expression won't
# understand the difference if we have also the sin function.
# The solution (complicated) is to use srepr and maybe ast.
#  The problem with the g=Function('f') is that it will be printed as 'f' but in
# the global namespace we have only 'g'. But as the same printer is used in the
# constructor of the namespace there will be no problem.

# Q: What if some of the printers are not printing as expected?
# A: The algorithm wont work. You must use srepr for those cases. But even
# srepr may not print well. All problems with printers should be considered
# bugs.

# Q: What about _imp_ functions?
# A: Those are taken care for by evalf. A special case treatment will work
# faster but it's not worth the code complexity.

# Q: Will ast fix all possible problems?
# A: No. You will always have to use some printer. Even srepr may not work in
# some cases. But if the printer does not work, that should be considered a
# bug.

# Q: Is there same way to fix all possible problems?
# A: Probably by constructing our strings ourself by traversing the (func,
# args) tree and creating the namespace at the same time. That actually sounds
# good.

from sympy.external import import_module
```
### 2 - sympy/printing/latex.py:

Start line: 2497, End line: 2674

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
          fold_short_frac=None, inv_trig_style="abbreviated",
          itex=False, ln_notation=False, long_frac_ratio=None,
          mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
          order=None, symbol_names=None, root_notation=True,
          mat_symbol_style="plain", imaginary_unit="i", gothic_re_im=False,
          decimal_separator="period" ):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``abbreviated``, ``full``, or ``power``. Defaults to ``abbreviated``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``[``, ``(``, or
        the empty string. Defaults to ``[``.
    mat_str : string, optional
        Which matrix environment string to emit. ``smallmatrix``, ``matrix``,
        ``array``, etc. Defaults to ``smallmatrix`` for inline mode, ``matrix``
        for matrices of no more than 10 columns, and ``array`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``plain``, ``inline``, ``equation`` or ``equation*``.  If ``mode``
        is set to ``plain``, then the resulting code will not be delimited at
        all (this is the default). If ``mode`` is set to ``inline`` then inline
        LaTeX ``$...$`` will be used. If ``mode`` is set to ``equation`` or
        ``equation*``, the resulting code will be enclosed in the ``equation``
        or ``equation*`` environment (remember to import ``amsmath`` for
        ``equation*``), unless the ``itex`` option is set. In the latter case,
        the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``, ``ldot``,
        ``dot``, or ``times``.
    order: string, optional
        Any of the supported monomial orderings (currently ``lex``, ``grlex``,
        or ``grevlex``), ``old``, and ``none``. This parameter does nothing for
        Mul objects. Setting order to ``old`` uses the compatibility ordering
        for Add defined in Printer. For very large expressions, set the
        ``order`` keyword to ``none`` if speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``plain`` (default) or ``bold``. If set to ``bold``,
        a MatrixSymbol A will be printed as ``\mathbf{A}``, otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are "i" (default)
        and "j". Adding "r" or "t" in front gives ``\mathrm`` or ``\text``, so
        "ri" leads to ``\mathrm{i}`` which gives `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
    decimal_separator : string, optional
        Specifies what separator to use to separate the whole and fractional parts of a
        floating point number as in `2.5` for the default, ``period`` or `2{,}5`
        when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
        separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
        ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left(\frac{7}{2} \right)}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left(10 \right)}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left(10 \right)}

    ``latex()`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \  y\right]$

    """
    # ... other code
```
### 3 - sympy/printing/latex.py:

Start line: 2228, End line: 2301

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
### 4 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 590, End line: 636

```python
class LaTeXParser ( Parser ):



    def additive(self, _p=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.AdditiveContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 8
        self.enterRecursionRule(localctx, 8, self.RULE_additive, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 90
            self.mp(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 97
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.AdditiveContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_additive)
                    self.state = 92
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 93
                    _la = self._input.LA(1)
                    if not(_la==LaTeXParser.ADD or _la==LaTeXParser.SUB):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 94
                    self.additive(3)
                self.state = 99
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx
```
### 5 - sympy/core/numbers.py:

Start line: 3784, End line: 3834

```python
class Catalan(with_metaclass(Singleton, NumberSymbol)):
    r"""Catalan's constant.

    `K = 0.91596559\ldots` is given by the infinite series

    .. math:: K = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2}

    Catalan is a singleton, and can be accessed by ``S.Catalan``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Catalan.is_irrational
    >>> S.Catalan > 0
    True
    >>> S.Catalan > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan%27s_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None
    is_number = True

    __slots__ = []

    def __int__(self):
        return 0

    def _as_mpf_val(self, prec):
        # XXX track down why this has to be increased
        v = mlib.catalan_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (Rational(9, 10), S.One)

    def _sage_(self):
        import sage.all as sage
        return sage.catalan
```
### 6 - sympy/printing/latex.py:

Start line: 596, End line: 613

```python
class LatexPrinter(Printer):

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 7 - sympy/printing/latex.py:

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
### 8 - sympy/printing/latex.py:

Start line: 1, End line: 83

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true

# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key, range
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

other_symbols = set(['aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp', ])

# Variable name modifiers
```
### 9 - sympy/printing/latex.py:

Start line: 1055, End line: 1139

```python
class LatexPrinter(Printer):

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 10 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 222, End line: 268

```python
class LaTeXParser ( Parser ):

    grammarFileName = "LaTeX.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ u"<INVALID>", u"','", u"<INVALID>", u"'+'", u"'-'",
                     u"'*'", u"'/'", u"'('", u"')'", u"'{'", u"'}'", u"'['",
                     u"']'", u"'|'", u"'\\lim'", u"<INVALID>", u"'\\int'",
                     u"'\\sum'", u"'\\prod'", u"'\\log'", u"'\\ln'", u"'\\sin'",
                     u"'\\cos'", u"'\\tan'", u"'\\csc'", u"'\\sec'", u"'\\cot'",
                     u"'\\arcsin'", u"'\\arccos'", u"'\\arctan'", u"'\\arccsc'",
                     u"'\\arcsec'", u"'\\arccot'", u"'\\sinh'", u"'\\cosh'",
                     u"'\\tanh'", u"'\\arsinh'", u"'\\arcosh'", u"'\\artanh'",
                     u"'\\sqrt'", u"'\\times'", u"'\\cdot'", u"'\\div'",
                     u"'\\frac'", u"'\\mathit'", u"'_'", u"'^'", u"':'",
                     u"<INVALID>", u"<INVALID>", u"<INVALID>", u"'='", u"'<'",
                     u"'\\leq'", u"'>'", u"'\\geq'", u"'!'" ]

    symbolicNames = [ u"<INVALID>", u"<INVALID>", u"WS", u"ADD", u"SUB",
                      u"MUL", u"DIV", u"L_PAREN", u"R_PAREN", u"L_BRACE",
                      u"R_BRACE", u"L_BRACKET", u"R_BRACKET", u"BAR", u"FUNC_LIM",
                      u"LIM_APPROACH_SYM", u"FUNC_INT", u"FUNC_SUM", u"FUNC_PROD",
                      u"FUNC_LOG", u"FUNC_LN", u"FUNC_SIN", u"FUNC_COS",
                      u"FUNC_TAN", u"FUNC_CSC", u"FUNC_SEC", u"FUNC_COT",
                      u"FUNC_ARCSIN", u"FUNC_ARCCOS", u"FUNC_ARCTAN", u"FUNC_ARCCSC",
                      u"FUNC_ARCSEC", u"FUNC_ARCCOT", u"FUNC_SINH", u"FUNC_COSH",
                      u"FUNC_TANH", u"FUNC_ARSINH", u"FUNC_ARCOSH", u"FUNC_ARTANH",
                      u"FUNC_SQRT", u"CMD_TIMES", u"CMD_CDOT", u"CMD_DIV",
                      u"CMD_FRAC", u"CMD_MATHIT", u"UNDERSCORE", u"CARET",
                      u"COLON", u"DIFFERENTIAL", u"LETTER", u"NUMBER", u"EQUAL",
                      u"LT", u"LTE", u"GT", u"GTE", u"BANG", u"SYMBOL" ]

    RULE_math = 0
    RULE_relation = 1
    RULE_equality = 2
    RULE_expr = 3
    RULE_additive = 4
    RULE_mp = 5
    RULE_mp_nofunc = 6
    RULE_unary = 7
    RULE_unary_nofunc = 8
    RULE_postfix = 9
```
