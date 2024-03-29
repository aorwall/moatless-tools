# sympy__sympy-14166

| **sympy/sympy** | `2c0a3a103baa547de12e332382d44ee3733d485f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 31015 |
| **Any found context length** | 31015 |
| **Avg pos** | 60.0 |
| **Min pos** | 60 |
| **Max pos** | 60 |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1329,7 +1329,7 @@ def _print_Order(self, expr):
                 s += self._print(expr.point)
             else:
                 s += self._print(expr.point[0])
-        return r"\mathcal{O}\left(%s\right)" % s
+        return r"O\left(%s\right)" % s
 
     def _print_Symbol(self, expr):
         if expr in self._settings['symbol_names']:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 1332 | 1332 | 60 | 5 | 31015


## Problem Statement

```
Typesetting of big-O symbol
Currently typesetting of big-O symbol uses the ordinary 'O', we can use the typesetting as defined here https://en.wikipedia.org/wiki/Big_O_notation#Typesetting .

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/expr.py | 1063 | 1117| 400 | 400 | 28937 | 
| 2 | 2 sympy/series/order.py | 1 | 122| 1162 | 1562 | 33044 | 
| 3 | 3 sympy/printing/pretty/pretty_symbology.py | 237 | 277| 603 | 2165 | 38318 | 
| 4 | 3 sympy/printing/pretty/pretty_symbology.py | 189 | 236| 772 | 2937 | 38318 | 
| 5 | 3 sympy/printing/pretty/pretty_symbology.py | 107 | 187| 758 | 3695 | 38318 | 
| 6 | 4 sympy/printing/mathml.py | 625 | 659| 302 | 3997 | 45578 | 
| 7 | 4 sympy/printing/pretty/pretty_symbology.py | 279 | 308| 230 | 4227 | 45578 | 
| 8 | **5 sympy/printing/latex.py** | 1 | 82| 715 | 4942 | 67836 | 
| 9 | **5 sympy/printing/latex.py** | 83 | 118| 491 | 5433 | 67836 | 
| 10 | 5 sympy/printing/pretty/pretty_symbology.py | 501 | 554| 426 | 5859 | 67836 | 
| 11 | 6 sympy/parsing/sympy_parser.py | 585 | 621| 270 | 6129 | 75569 | 
| 12 | 7 sympy/printing/dot.py | 1 | 25| 218 | 6347 | 77410 | 
| 13 | 7 sympy/printing/pretty/pretty_symbology.py | 557 | 583| 360 | 6707 | 77410 | 
| 14 | 8 sympy/abc.py | 1 | 67| 801 | 7508 | 78635 | 
| 15 | 9 sympy/sets/ordinals.py | 1 | 53| 337 | 7845 | 80483 | 
| 16 | 9 sympy/printing/pretty/pretty_symbology.py | 75 | 104| 167 | 8012 | 80483 | 
| 17 | 10 sympy/printing/pretty/pretty.py | 1392 | 1409| 173 | 8185 | 100459 | 
| 18 | 10 sympy/printing/pretty/pretty_symbology.py | 379 | 464| 744 | 8929 | 100459 | 
| 19 | 10 sympy/printing/pretty/pretty_symbology.py | 1 | 46| 258 | 9187 | 100459 | 
| 20 | 11 sympy/printing/rust.py | 57 | 162| 1065 | 10252 | 105896 | 
| 21 | 11 sympy/printing/pretty/pretty_symbology.py | 466 | 498| 346 | 10598 | 105896 | 
| 22 | 11 sympy/printing/mathml.py | 246 | 286| 345 | 10943 | 105896 | 
| 23 | 11 sympy/parsing/sympy_parser.py | 623 | 711| 703 | 11646 | 105896 | 
| 24 | 11 sympy/printing/pretty/pretty.py | 1534 | 1578| 501 | 12147 | 105896 | 
| 25 | 12 sympy/printing/codeprinter.py | 358 | 407| 486 | 12633 | 109937 | 
| 26 | 13 sympy/series/gruntz.py | 554 | 625| 707 | 13340 | 116192 | 
| 27 | 14 sympy/printing/octave.py | 521 | 665| 1626 | 14966 | 122257 | 
| 28 | 15 sympy/printing/pretty/stringpict.py | 217 | 249| 330 | 15296 | 126529 | 
| 29 | 16 sympy/printing/julia.py | 204 | 246| 290 | 15586 | 132165 | 
| 30 | 16 sympy/printing/octave.py | 366 | 448| 719 | 16305 | 132165 | 
| 31 | 17 sympy/printing/conventions.py | 1 | 68| 489 | 16794 | 132765 | 
| 32 | **17 sympy/printing/latex.py** | 894 | 949| 577 | 17371 | 132765 | 
| 33 | 17 sympy/printing/octave.py | 326 | 342| 183 | 17554 | 132765 | 
| 34 | 17 sympy/printing/octave.py | 196 | 213| 199 | 17753 | 132765 | 
| 35 | 18 sympy/physics/units/prefixes.py | 148 | 212| 600 | 18353 | 134398 | 
| 36 | **18 sympy/printing/latex.py** | 2093 | 2247| 230 | 18583 | 134398 | 
| 37 | **18 sympy/printing/latex.py** | 1256 | 1316| 753 | 19336 | 134398 | 
| 38 | 19 sympy/codegen/ast.py | 1 | 73| 567 | 19903 | 142383 | 
| 39 | 19 sympy/printing/mathml.py | 750 | 785| 301 | 20204 | 142383 | 
| 40 | 19 sympy/printing/pretty/pretty_symbology.py | 49 | 72| 201 | 20405 | 142383 | 
| 41 | **19 sympy/printing/latex.py** | 2080 | 2090| 159 | 20564 | 142383 | 
| 42 | **19 sympy/printing/latex.py** | 1109 | 1174| 672 | 21236 | 142383 | 
| 43 | 20 sympy/printing/pycode.py | 250 | 265| 236 | 21472 | 146715 | 
| 44 | **20 sympy/printing/latex.py** | 975 | 1057| 781 | 22253 | 146715 | 
| 45 | 20 sympy/printing/pretty/pretty.py | 1296 | 1310| 177 | 22430 | 146715 | 
| 46 | 20 sympy/printing/pretty/pretty.py | 1281 | 1294| 142 | 22572 | 146715 | 
| 47 | 20 sympy/printing/octave.py | 216 | 237| 172 | 22744 | 146715 | 
| 48 | 21 sympy/printing/ccode.py | 282 | 321| 343 | 23087 | 153796 | 
| 49 | **21 sympy/printing/latex.py** | 482 | 533| 588 | 23675 | 153796 | 
| 50 | 22 sympy/printing/mathematica.py | 1 | 34| 394 | 24069 | 154998 | 
| 51 | 23 sympy/printing/theanocode.py | 98 | 154| 523 | 24592 | 157081 | 
| 52 | **23 sympy/printing/latex.py** | 2123 | 2253| 1602 | 26194 | 157081 | 
| 53 | 24 sympy/printing/cxxcode.py | 1 | 61| 652 | 26846 | 158611 | 
| 54 | **24 sympy/printing/latex.py** | 1334 | 1360| 210 | 27056 | 158611 | 
| 55 | 24 sympy/printing/pretty/pretty.py | 131 | 188| 556 | 27612 | 158611 | 
| 56 | 25 sympy/core/power.py | 157 | 239| 1060 | 28672 | 172894 | 
| 57 | 25 sympy/printing/octave.py | 130 | 193| 589 | 29261 | 172894 | 
| 58 | 25 sympy/printing/pretty/pretty.py | 1106 | 1188| 773 | 30034 | 172894 | 
| 59 | 25 sympy/printing/pretty/pretty.py | 1901 | 2012| 831 | 30865 | 172894 | 
| **-> 60 <-** | **25 sympy/printing/latex.py** | 1318 | 1332| 150 | 31015 | 172894 | 
| 61 | 25 sympy/printing/pretty/pretty.py | 101 | 117| 199 | 31214 | 172894 | 
| 62 | 25 sympy/printing/pretty/pretty.py | 1 | 34| 287 | 31501 | 172894 | 
| 63 | 25 sympy/printing/octave.py | 268 | 299| 175 | 31676 | 172894 | 
| 64 | 25 sympy/printing/mathml.py | 730 | 748| 153 | 31829 | 172894 | 
| 65 | **25 sympy/printing/latex.py** | 708 | 775| 650 | 32479 | 172894 | 
| 66 | 25 sympy/printing/octave.py | 1 | 54| 579 | 33058 | 172894 | 
| 67 | 26 sympy/printing/fcode.py | 1 | 56| 449 | 33507 | 178645 | 
| 68 | 26 sympy/printing/rust.py | 1 | 56| 581 | 34088 | 178645 | 
| 69 | **26 sympy/printing/latex.py** | 1784 | 1812| 269 | 34357 | 178645 | 
| 70 | 27 sympy/printing/str.py | 306 | 340| 297 | 34654 | 185389 | 
| 71 | 27 sympy/printing/dot.py | 58 | 76| 177 | 34831 | 185389 | 
| 72 | 27 sympy/printing/fcode.py | 134 | 149| 131 | 34962 | 185389 | 
| 73 | 27 sympy/printing/ccode.py | 1 | 89| 736 | 35698 | 185389 | 
| 74 | 28 sympy/physics/quantum/sho1d.py | 148 | 164| 166 | 35864 | 190835 | 
| 75 | 28 sympy/printing/pretty/pretty.py | 796 | 865| 597 | 36461 | 190835 | 
| 76 | **28 sympy/printing/latex.py** | 1200 | 1254| 730 | 37191 | 190835 | 
| 77 | 29 sympy/interactive/printing.py | 99 | 131| 373 | 37564 | 194617 | 
| 78 | 29 sympy/printing/pretty/pretty.py | 252 | 318| 583 | 38147 | 194617 | 
| 79 | 30 sympy/codegen/ffunctions.py | 50 | 72| 193 | 38340 | 195101 | 
| 80 | 30 sympy/printing/fcode.py | 241 | 282| 352 | 38692 | 195101 | 
| 81 | 30 sympy/interactive/printing.py | 149 | 230| 698 | 39390 | 195101 | 
| 82 | 30 sympy/series/gruntz.py | 517 | 551| 261 | 39651 | 195101 | 
| 83 | 31 sympy/printing/rcode.py | 158 | 179| 177 | 39828 | 198826 | 
| 84 | 31 sympy/printing/pycode.py | 351 | 403| 688 | 40516 | 198826 | 
| 85 | **31 sympy/printing/latex.py** | 1982 | 2046| 754 | 41270 | 198826 | 


### Hint

```
What I was saying was that we should replace the calligraphy O with a normal O, not the other way around. 
Oh sorry, @asmeurer I didn't see your comment. Should I close #14164?
I also read the issue wrong. But I just checked the master branch:
\`\`\`
>>> pprint(series(cos(x), x, pi, 3))
            2
     (x - π)     ⎛       3       ⎞
-1 + ──────── + O⎝(x - π) ; x → π⎠
        2
\`\`\`
The `O` here is ordinary O only, [see](https://github.com/sympy/sympy/blob/2c0a3a103baa547de12e332382d44ee3733d485f/sympy/printing/pretty/pretty.py#L1278).
@jashan498 The issue is about LaTeX printer, which uses calligraphic O
```

## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1329,7 +1329,7 @@ def _print_Order(self, expr):
                 s += self._print(expr.point)
             else:
                 s += self._print(expr.point[0])
-        return r"\mathcal{O}\left(%s\right)" % s
+        return r"O\left(%s\right)" % s
 
     def _print_Symbol(self, expr):
         if expr in self._settings['symbol_names']:

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -330,14 +330,14 @@ def test_latex_functions():
     assert latex(gamma(x)) == r"\Gamma\left(x\right)"
     w = Wild('w')
     assert latex(gamma(w)) == r"\Gamma\left(w\right)"
-    assert latex(Order(x)) == r"\mathcal{O}\left(x\right)"
-    assert latex(Order(x, x)) == r"\mathcal{O}\left(x\right)"
-    assert latex(Order(x, (x, 0))) == r"\mathcal{O}\left(x\right)"
-    assert latex(Order(x, (x, oo))) == r"\mathcal{O}\left(x; x\rightarrow \infty\right)"
-    assert latex(Order(x - y, (x, y))) == r"\mathcal{O}\left(x - y; x\rightarrow y\right)"
-    assert latex(Order(x, x, y)) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow \left ( 0, \quad 0\right )\right)"
-    assert latex(Order(x, x, y)) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow \left ( 0, \quad 0\right )\right)"
-    assert latex(Order(x, (x, oo), (y, oo))) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow \left ( \infty, \quad \infty\right )\right)"
+    assert latex(Order(x)) == r"O\left(x\right)"
+    assert latex(Order(x, x)) == r"O\left(x\right)"
+    assert latex(Order(x, (x, 0))) == r"O\left(x\right)"
+    assert latex(Order(x, (x, oo))) == r"O\left(x; x\rightarrow \infty\right)"
+    assert latex(Order(x - y, (x, y))) == r"O\left(x - y; x\rightarrow y\right)"
+    assert latex(Order(x, x, y)) == r"O\left(x; \left ( x, \quad y\right )\rightarrow \left ( 0, \quad 0\right )\right)"
+    assert latex(Order(x, x, y)) == r"O\left(x; \left ( x, \quad y\right )\rightarrow \left ( 0, \quad 0\right )\right)"
+    assert latex(Order(x, (x, oo), (y, oo))) == r"O\left(x; \left ( x, \quad y\right )\rightarrow \left ( \infty, \quad \infty\right )\right)"
     assert latex(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
     assert latex(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'
 

```


## Code snippets

### 1 - sympy/core/expr.py:

Start line: 1063, End line: 1117

```python
class Expr(Basic, EvalfMixin):

    def removeO(self):
        """Removes the additive O(..) symbol if there is one"""
        return self

    def getO(self):
        """Returns the additive O(..) symbol if there is one, else None."""
        return None

    def getn(self):
        """
        Returns the order of the expression.

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
        from sympy import Dummy, Symbol
        o = self.getO()
        if o is None:
            return None
        elif o.is_Order:
            o = o.expr
            if o is S.One:
                return S.Zero
            if o.is_Symbol:
                return S.One
            if o.is_Pow:
                return o.args[1]
            if o.is_Mul:  # x**n*log(x)**n or x**n/log(x)**n
                for oi in o.args:
                    if oi.is_Symbol:
                        return S.One
                    if oi.is_Pow:
                        syms = oi.atoms(Symbol)
                        if len(syms) == 1:
                            x = syms.pop()
                            oi = oi.subs(x, Dummy('x', positive=True))
                            if oi.base.is_Symbol and oi.exp.is_Rational:
                                return abs(oi.exp)

        raise NotImplementedError('not sure of order of %s' % o)

    def count_ops(self, visual=None):
        """wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)
```
### 2 - sympy/series/order.py:

Start line: 1, End line: 122

```python
from __future__ import print_function, division

from sympy.core import S, sympify, Expr, Rational, Symbol, Dummy
from sympy.core import Add, Mul, expand_power_base, expand_log
from sympy.core.cache import cacheit
from sympy.core.compatibility import default_sort_key, is_sequence
from sympy.core.containers import Tuple
from sympy.utilities.iterables import uniq
from sympy.sets.sets import Complement


class Order(Expr):
    r""" Represents the limiting behavior of some function

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if for any
    `\delta > 0` there exists a `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\lim_{x \rightarrow a}
    \sup |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, for any `\delta > 0` there is an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <http://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.

    """
```
### 3 - sympy/printing/pretty/pretty_symbology.py:

Start line: 237, End line: 277

```python
_xobj_unicode = {

    # vertical symbols
    #       (( ext, top, bot, mid ), c1)
    '(':    (( EXT('('), HUP('('), HLO('(') ), '('),
    ')':    (( EXT(')'), HUP(')'), HLO(')') ), ')'),
    '[':    (( EXT('['), CUP('['), CLO('[') ), '['),
    ']':    (( EXT(']'), CUP(']'), CLO(']') ), ']'),
    '{':    (( EXT('{}'), HUP('{'), HLO('{'), MID('{') ), '{'),
    '}':    (( EXT('{}'), HUP('}'), HLO('}'), MID('}') ), '}'),
    '|':    U('BOX DRAWINGS LIGHT VERTICAL'),

    '<':    ((U('BOX DRAWINGS LIGHT VERTICAL'),
              U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
              U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')), '<'),

    '>':    ((U('BOX DRAWINGS LIGHT VERTICAL'),
              U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
              U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), '>'),

    'lfloor': (( EXT('['), EXT('['), CLO('[') ), U('LEFT FLOOR')),
    'rfloor': (( EXT(']'), EXT(']'), CLO(']') ), U('RIGHT FLOOR')),
    'lceil':  (( EXT('['), CUP('['), EXT('[') ), U('LEFT CEILING')),
    'rceil':  (( EXT(']'), CUP(']'), EXT(']') ), U('RIGHT CEILING')),

    'int':  (( EXT('int'), U('TOP HALF INTEGRAL'), U('BOTTOM HALF INTEGRAL') ), U('INTEGRAL')),
    'sum':  (( U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'), '_', U('OVERLINE'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), U('N-ARY SUMMATION')),

    # horizontal objects
    #'-':   '-',
    '-':    U('BOX DRAWINGS LIGHT HORIZONTAL'),
    '_':    U('LOW LINE'),
    # We used to use this, but LOW LINE looks better for roots, as it's a
    # little lower (i.e., it lines up with the / perfectly.  But perhaps this
    # one would still be wanted for some cases?
    # '_':    U('HORIZONTAL SCAN LINE-9'),

    # diagonal objects '\' & '/' ?
    '/':    U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
    '\\':   U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
}
```
### 4 - sympy/printing/pretty/pretty_symbology.py:

Start line: 189, End line: 236

```python
for s in '+-=()':
    sub[s] = SSUB(s)
    sup[s] = SSUP(s)

# Variable modifiers
# TODO: Is it worth trying to handle faces with, e.g., 'MATHEMATICAL BOLD CAPITAL A'?
# TODO: Make brackets adjust to height of contents
modifier_dict = {
    # Accents
    'mathring': lambda s: s+u'\N{COMBINING RING ABOVE}',
    'ddddot': lambda s: s+u'\N{COMBINING DIAERESIS}\N{COMBINING DIAERESIS}',
    'dddot': lambda s: s+u'\N{COMBINING DIAERESIS}\N{COMBINING DOT ABOVE}',
    'ddot': lambda s: s+u'\N{COMBINING DIAERESIS}',
    'dot': lambda s: s+u'\N{COMBINING DOT ABOVE}',
    'check': lambda s: s+u'\N{COMBINING CARON}',
    'breve': lambda s: s+u'\N{COMBINING BREVE}',
    'acute': lambda s: s+u'\N{COMBINING ACUTE ACCENT}',
    'grave': lambda s: s+u'\N{COMBINING GRAVE ACCENT}',
    'tilde': lambda s: s+u'\N{COMBINING TILDE}',
    'hat': lambda s: s+u'\N{COMBINING CIRCUMFLEX ACCENT}',
    'bar': lambda s: s+u'\N{COMBINING OVERLINE}',
    'vec': lambda s: s+u'\N{COMBINING RIGHT ARROW ABOVE}',
    'prime': lambda s: s+u'\N{PRIME}',
    'prm': lambda s: s+u'\N{PRIME}',
    # # Faces -- these are here for some compatibility with latex printing
    # 'bold': lambda s: s,
    # 'bm': lambda s: s,
    # 'cal': lambda s: s,
    # 'scr': lambda s: s,
    # 'frak': lambda s: s,
    # Brackets
    'norm': lambda s: u'\N{DOUBLE VERTICAL LINE}'+s+u'\N{DOUBLE VERTICAL LINE}',
    'avg': lambda s: u'\N{MATHEMATICAL LEFT ANGLE BRACKET}'+s+u'\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
    'abs': lambda s: u'\N{VERTICAL LINE}'+s+u'\N{VERTICAL LINE}',
    'mag': lambda s: u'\N{VERTICAL LINE}'+s+u'\N{VERTICAL LINE}',
}

# VERTICAL OBJECTS
HUP = lambda symb: U('%s UPPER HOOK' % symb_2txt[symb])
CUP = lambda symb: U('%s UPPER CORNER' % symb_2txt[symb])
MID = lambda symb: U('%s MIDDLE PIECE' % symb_2txt[symb])
EXT = lambda symb: U('%s EXTENSION' % symb_2txt[symb])
HLO = lambda symb: U('%s LOWER HOOK' % symb_2txt[symb])
CLO = lambda symb: U('%s LOWER CORNER' % symb_2txt[symb])
TOP = lambda symb: U('%s TOP' % symb_2txt[symb])
BOT = lambda symb: U('%s BOTTOM' % symb_2txt[symb])

# {} '('  ->  (extension, start, end, middle) 1-character
```
### 5 - sympy/printing/pretty/pretty_symbology.py:

Start line: 107, End line: 187

```python
def xstr(*args):
    """call str or unicode depending on current mode"""
    if _use_unicode:
        return unicode(*args)
    else:
        return str(*args)

# GREEK
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())

greek_letters = list(greeks) # make a copy
# deal with Unicode's funny spelling of lambda
greek_letters[greek_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_unicode = {l: (g(l), G(l)) for l in greek_letters}
greek_unicode = dict((L, g(L)) for L in greek_letters)
greek_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_letters)

# aliases
greek_unicode['lambda'] = greek_unicode['lamda']
greek_unicode['Lambda'] = greek_unicode['Lamda']
greek_unicode['varsigma'] = u'\N{GREEK SMALL LETTER FINAL SIGMA}'

digit_2txt = {
    '0':    'ZERO',
    '1':    'ONE',
    '2':    'TWO',
    '3':    'THREE',
    '4':    'FOUR',
    '5':    'FIVE',
    '6':    'SIX',
    '7':    'SEVEN',
    '8':    'EIGHT',
    '9':    'NINE',
}

symb_2txt = {
    '+':    'PLUS SIGN',
    '-':    'MINUS',
    '=':    'EQUALS SIGN',
    '(':    'LEFT PARENTHESIS',
    ')':    'RIGHT PARENTHESIS',
    '[':    'LEFT SQUARE BRACKET',
    ']':    'RIGHT SQUARE BRACKET',
    '{':    'LEFT CURLY BRACKET',
    '}':    'RIGHT CURLY BRACKET',

    # non-std
    '{}':   'CURLY BRACKET',
    'sum':  'SUMMATION',
    'int':  'INTEGRAL',
}

# SUBSCRIPT & SUPERSCRIPT
LSUB = lambda letter: U('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
GSUB = lambda letter: U('GREEK SUBSCRIPT SMALL LETTER %s' % letter.upper())
DSUB = lambda digit:  U('SUBSCRIPT %s' % digit_2txt[digit])
SSUB = lambda symb:   U('SUBSCRIPT %s' % symb_2txt[symb])

LSUP = lambda letter: U('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
DSUP = lambda digit:  U('SUPERSCRIPT %s' % digit_2txt[digit])
SSUP = lambda symb:   U('SUPERSCRIPT %s' % symb_2txt[symb])

sub = {}    # symb -> subscript symbol
sup = {}    # symb -> superscript symbol

# latin subscripts
for l in 'aeioruvxhklmnpst':
    sub[l] = LSUB(l)

for l in 'in':
    sup[l] = LSUP(l)

for gl in ['beta', 'gamma', 'rho', 'phi', 'chi']:
    sub[gl] = GSUB(gl)

for d in [str(i) for i in range(10)]:
    sub[d] = DSUB(d)
    sup[d] = DSUP(d)
```
### 6 - sympy/printing/mathml.py:

Start line: 625, End line: 659

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_ImaginaryUnit(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ImaginaryI;'))
        return x

    def _print_GoldenRatio(self, e):
        """We use unicode #x3c6 for Greek letter phi as defined here
        http://www.w3.org/2003/entities/2007doc/isogrk1.html"""
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode(u"\N{GREEK SMALL LETTER PHI}"))
        return x

    def _print_Exp1(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ExponentialE;'))
        return x

    def _print_Pi(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&pi;'))
        return x

    def _print_Infinity(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x221E;'))
        return x

    def _print_Negative_Infinity(self, e):
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mo')
        y.appendChild(self.dom.createTextNode('-'))
        x = self._print_Infinity(-e)
        mrow.appendChild(y)
        mrow.appendChild(x)
        return mrow
```
### 7 - sympy/printing/pretty/pretty_symbology.py:

Start line: 279, End line: 308

```python
_xobj_ascii = {
    # vertical symbols
    #       (( ext, top, bot, mid ), c1)
    '(':    (( '|', '/', '\\' ), '('),
    ')':    (( '|', '\\', '/' ), ')'),

# XXX this looks ugly
#   '[':    (( '|', '-', '-' ), '['),
#   ']':    (( '|', '-', '-' ), ']'),
# XXX not so ugly :(
    '[':    (( '[', '[', '[' ), '['),
    ']':    (( ']', ']', ']' ), ']'),

    '{':    (( '|', '/', '\\', '<' ), '{'),
    '}':    (( '|', '\\', '/', '>' ), '}'),
    '|':    '|',

    '<':    (( '|', '/', '\\' ), '<'),
    '>':    (( '|', '\\', '/' ), '>'),

    'int':  ( ' | ', '  /', '/  ' ),

    # horizontal objects
    '-':    '-',
    '_':    '_',

    # diagonal objects '\' & '/' ?
    '/':    '/',
    '\\':   '\\',
}
```
### 8 - sympy/printing/latex.py:

Start line: 1, End line: 82

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.function import _coeff_isneg
from sympy.core.sympify import SympifyError
from sympy.core.alphabets import greeks
from sympy.core.operations import AssocOp
from sympy.core.containers import Tuple
from sympy.logic.boolalg import true
from sympy.core.function import UndefinedFunction, AppliedUndef

## sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from .printer import Printer
from .conventions import split_super_sub, requires_partial
from .precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key, range
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at http://www.mathjax.org/docs/1.1/tex.html#supported-latex-commands
# This variable only contains those functions which sympy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec', 'csc',
                    'cot', 'coth', 're', 'im', 'frac', 'root', 'arg',
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

Start line: 83, End line: 118

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
### 10 - sympy/printing/pretty/pretty_symbology.py:

Start line: 501, End line: 554

```python
def pretty_symbol(symb_name):
    """return pretty representation of a symbol"""
    # let's split symb_name into symbol + index
    # UC: beta1
    # UC: f_beta

    if not _use_unicode:
        return symb_name

    name, sups, subs = split_super_sub(symb_name)

    def translate(s) :
        gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True) :
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s

    name = translate(name)

    # Let's prettify sups/subs. If it fails at one of them, pretty sups/subs are
    # not used at all.
    def pretty_list(l, mapping):
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:  # match by separate characters
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result

    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None

    # glue the results into one string
    if pretty_subs is None:  # nice formatting of sups/subs did not work
        if subs:
            name += '_'+'_'.join([translate(s) for s in subs])
        if sups:
            name += '__'+'__'.join([translate(s) for s in sups])
        return name
    else:
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)

    return ''.join([name, sups_result, subs_result])
```
### 32 - sympy/printing/latex.py:

Start line: 894, End line: 949

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
        arg = r"{\left (%s \right )}" % self._print(expr.args[0])

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
### 36 - sympy/printing/latex.py:

Start line: 2093, End line: 2247

```python
def translate(s):
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True):
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s

def latex(expr, **settings):
    # ... other code
```
### 37 - sympy/printing/latex.py:

Start line: 1256, End line: 1316

```python
class LatexPrinter(Printer):

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            return self._print(expr.p)
```
### 41 - sympy/printing/latex.py:

Start line: 2080, End line: 2090

```python
class LatexPrinter(Printer):

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])
```
### 42 - sympy/printing/latex.py:

Start line: 1109, End line: 1174

```python
class LatexPrinter(Printer):

    def _hprint_vec(self, vec):
        if len(vec) == 0:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')
```
### 44 - sympy/printing/latex.py:

Start line: 975, End line: 1057

```python
class LatexPrinter(Printer):

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_gamma(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

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
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 49 - sympy/printing/latex.py:

Start line: 482, End line: 533

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
### 52 - sympy/printing/latex.py:

Start line: 2123, End line: 2253

```python
def latex(expr, **settings):
    r"""
    Convert the given expression to LaTeX representation.

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational
    >>> from sympy.abc import x, y, mu, r, tau

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'

    order: Any of the supported monomial orderings (currently "lex", "grlex", or
    "grevlex"), "old", and "none". This parameter does nothing for Mul objects.
    Setting order to "old" uses the compatibility ordering for Add defined in
    Printer. For very large expressions, set the 'order' keyword to 'none' if
    speed is a concern.

    mode: Specifies how the generated code will be delimited. 'mode' can be one
    of 'plain', 'inline', 'equation' or 'equation*'.  If 'mode' is set to
    'plain', then the resulting code will not be delimited at all (this is the
    default). If 'mode' is set to 'inline' then inline LaTeX $ $ will be used.
    If 'mode' is set to 'equation' or 'equation*', the resulting code will be
    enclosed in the 'equation' or 'equation*' environment (remember to import
    'amsmath' for 'equation*'), unless the 'itex' option is set. In the latter
    case, the ``$$ $$`` syntax is used.

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}

    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$

    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}

    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}

    itex: Specifies if itex-specific syntax is used, including emitting ``$$ $$``.

    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    fold_frac_powers: Emit "^{p/q}" instead of "^{\frac{p}{q}}" for fractional
    powers.

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}

    fold_func_brackets: Fold function brackets where applicable.

    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left (\frac{7}{2} \right )}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets = True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}

    fold_short_frac: Emit "p / q" instead of "\frac{p}{q}" when the
    denominator is simple enough (at most two terms and no powers).
    The default value is `True` for inline mode, False otherwise.

    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y

    long_frac_ratio: The allowed ratio of the width of the numerator to the
    width of the denominator before we start breaking off long fractions.
    The default value is 2.

    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    mul_symbol: The symbol to use for multiplication. Can be one of None,
    "ldot", "dot", or "times".

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left (\frac{7}{2} \right )}}

    inv_trig_style: How inverse trig functions should be displayed. Can be one
    of "abbreviated", "full", or "power". Defaults to "abbreviated".

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left (\frac{7}{2} \right )}

    mat_str: Which matrix environment string to emit. "smallmatrix", "matrix",
    "array", etc. Defaults to "smallmatrix" for inline mode, "matrix" for
    matrices of no more than 10 columns, and "array" otherwise.

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]

    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]

    mat_delim: The delimiter to wrap around matrices. Can be one of "[", "(",
    or the empty string. Defaults to "[".

    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    symbol_names: Dictionary of symbols and the custom strings they should be
    emitted as.

    >>> print(latex(x**2, symbol_names={x:'x_i'}))
    x_i^{2}

    ``latex`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left [ 2 / x, \quad y\right ]$

    """

    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression."""
    print(latex(expr, **settings))
```
### 54 - sympy/printing/latex.py:

Start line: 1334, End line: 1360

```python
class LatexPrinter(Printer):

    def _print_Symbol(self, expr):
        if expr in self._settings['symbol_names']:
            return self._settings['symbol_names'][expr]

        return self._deal_with_super_sub(expr.name) if \
            '\\' not in expr.name else expr.name

    _print_RandomSymbol = _print_Symbol
    _print_MatrixSymbol = _print_Symbol

    def _deal_with_super_sub(self, string):
        if '{' in string:
            return string

        name, supers, subs = split_super_sub(string)

        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        # glue all items together:
        if len(supers) > 0:
            name += "^{%s}" % " ".join(supers)
        if len(subs) > 0:
            name += "_{%s}" % " ".join(subs)

        return name
```
### 60 - sympy/printing/latex.py:

Start line: 1318, End line: 1332

```python
class LatexPrinter(Printer):

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif len(expr.variables):
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"\mathcal{O}\left(%s\right)" % s
```
### 65 - sympy/printing/latex.py:

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
        if hasattr(self, '_print_' + func) and \
            not isinstance(expr.func, UndefinedFunction):
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
### 69 - sympy/printing/latex.py:

Start line: 1784, End line: 1812

```python
class LatexPrinter(Printer):

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)
```
### 76 - sympy/printing/latex.py:

Start line: 1200, End line: 1254

```python
class LatexPrinter(Printer):

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (self._print(exp), tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (self._print(exp), tex)
        return r"\zeta%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (self._print(exp), tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, self._print(exp), tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex
```
### 85 - sympy/printing/latex.py:

Start line: 1982, End line: 2046

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
