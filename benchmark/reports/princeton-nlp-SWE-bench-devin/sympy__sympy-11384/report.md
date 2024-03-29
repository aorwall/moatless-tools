# sympy__sympy-11384

| **sympy/sympy** | `496e776108957d8c049cbef49522cef4c1955e2f` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 22486 |
| **Any found context length** | 22486 |
| **Avg pos** | 22.0 |
| **Min pos** | 44 |
| **Max pos** | 44 |
| **Top file pos** | 6 |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1605,7 +1605,7 @@ def _print_FourierSeries(self, s):
         return self._print_Add(s.truncate()) + self._print(' + \ldots')
 
     def _print_FormalPowerSeries(self, s):
-        return self._print_Add(s.truncate())
+        return self._print_Add(s.infinite)
 
     def _print_FiniteField(self, expr):
         return r"\mathbb{F}_{%s}" % expr.mod
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -1629,7 +1629,7 @@ def _print_FourierSeries(self, s):
         return self._print_Add(s.truncate()) + self._print(dots)
 
     def _print_FormalPowerSeries(self, s):
-        return self._print_Add(s.truncate())
+        return self._print_Add(s.infinite)
 
     def _print_SeqFormula(self, s):
         if self._use_unicode:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 1608 | 1608 | 44 | 6 | 22486
| sympy/printing/pretty/pretty.py | 1632 | 1632 | - | - | -


## Problem Statement

```
fps should print as a formal power series
When I first used `fps`, I didn't realize it really was a formal power series as it claims to be, because it prints like a normal series (same as `series`)

\`\`\`
In [21]: fps(sin(x))
Out[21]:
     3     5
    x     x     ⎛ 6⎞
x - ── + ─── + O⎝x ⎠
    6    120
\`\`\`

But if you look at the string form, you see

\`\`\`
In [22]: print(fps(sin(x)))
FormalPowerSeries(sin(x), x, 0, 1, (SeqFormula(Piecewise(((-1/4)**(_k/2 - 1/2)/(RisingFactorial(3/2, _k/2 - 1/2)*factorial(_k/2 - 1/2)), Eq(Mod(_k, 2), 1)), (0, True)), (_k, 2, oo)), SeqFormula(x**_k, (_k, 0, oo)), x))
\`\`\`

That is, it really does represent it as the formula `Sum((-1)**n/factorial(2*n + 1)*x**n, (n, 0, oo))` (albiet, not simplified). It out to print it like this, so you can see that that's what it's working with.

Side question: if you enter something it can't compute, it just returns the function

\`\`\`
In [25]: fps(tan(x))
Out[25]: tan(x)
\`\`\`

Is that intentional? It seems like it ought to raise an exception in that case. 

@leosartaj 


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/series/formal.py | 1182 | 1256| 615 | 615 | 9813 | 
| 2 | 1 sympy/series/formal.py | 752 | 829| 691 | 1306 | 9813 | 
| 3 | 1 sympy/series/formal.py | 832 | 900| 491 | 1797 | 9813 | 
| 4 | 1 sympy/series/formal.py | 903 | 979| 426 | 2223 | 9813 | 
| 5 | 1 sympy/series/formal.py | 1071 | 1113| 391 | 2614 | 9813 | 
| 6 | 2 sympy/series/fourier.py | 321 | 397| 575 | 3189 | 12740 | 
| 7 | 2 sympy/series/formal.py | 1048 | 1069| 229 | 3418 | 12740 | 
| 8 | 2 sympy/series/formal.py | 1017 | 1046| 212 | 3630 | 12740 | 
| 9 | 2 sympy/series/formal.py | 1153 | 1179| 199 | 3829 | 12740 | 
| 10 | 3 sympy/simplify/fu.py | 1 | 189| 2051 | 5880 | 31028 | 
| 11 | 4 sympy/core/function.py | 650 | 692| 469 | 6349 | 52553 | 
| 12 | 5 sympy/printing/fcode.py | 418 | 538| 1286 | 7635 | 57258 | 
| 13 | 5 sympy/series/formal.py | 1115 | 1151| 310 | 7945 | 57258 | 
| 14 | 5 sympy/printing/fcode.py | 241 | 255| 160 | 8105 | 57258 | 
| 15 | 5 sympy/series/fourier.py | 1 | 16| 130 | 8235 | 57258 | 
| 16 | 5 sympy/series/formal.py | 1 | 26| 228 | 8463 | 57258 | 
| 17 | 5 sympy/printing/fcode.py | 215 | 239| 219 | 8682 | 57258 | 
| 18 | 5 sympy/series/fourier.py | 285 | 318| 236 | 8918 | 57258 | 
| 19 | **6 sympy/printing/latex.py** | 987 | 1001| 166 | 9084 | 77342 | 
| 20 | 6 sympy/series/fourier.py | 19 | 36| 225 | 9309 | 77342 | 
| 21 | 6 sympy/series/formal.py | 363 | 389| 235 | 9544 | 77342 | 
| 22 | 7 sympy/functions/special/error_functions.py | 2204 | 2226| 343 | 9887 | 97375 | 
| 23 | 7 sympy/series/formal.py | 981 | 997| 126 | 10013 | 97375 | 
| 24 | 8 sympy/core/expr.py | 2843 | 2891| 410 | 10423 | 124820 | 
| 25 | 8 sympy/series/fourier.py | 93 | 171| 401 | 10824 | 124820 | 
| 26 | 8 sympy/printing/fcode.py | 1 | 49| 361 | 11185 | 124820 | 
| 27 | 9 sympy/core/evalf.py | 631 | 713| 848 | 12033 | 137935 | 
| 28 | 9 sympy/series/fourier.py | 228 | 255| 236 | 12269 | 137935 | 
| 29 | 9 sympy/core/function.py | 589 | 649| 663 | 12932 | 137935 | 
| 30 | **9 sympy/printing/latex.py** | 903 | 985| 755 | 13687 | 137935 | 
| 31 | 9 sympy/printing/fcode.py | 173 | 213| 338 | 14025 | 137935 | 
| 32 | 9 sympy/series/fourier.py | 257 | 283| 235 | 14260 | 137935 | 
| 33 | 10 sympy/integrals/transforms.py | 554 | 656| 991 | 15251 | 154432 | 
| 34 | **10 sympy/printing/latex.py** | 1928 | 2058| 1604 | 16855 | 154432 | 
| 35 | **10 sympy/printing/latex.py** | 650 | 717| 638 | 17493 | 154432 | 
| 36 | 10 sympy/functions/special/error_functions.py | 2336 | 2358| 345 | 17838 | 154432 | 
| 37 | 10 sympy/series/formal.py | 999 | 1015| 128 | 17966 | 154432 | 
| 38 | 11 sympy/functions/elementary/trigonometric.py | 322 | 376| 496 | 18462 | 176706 | 
| 39 | **11 sympy/printing/latex.py** | 418 | 462| 489 | 18951 | 176706 | 
| 40 | 12 sympy/core/power.py | 102 | 182| 1032 | 19983 | 189887 | 
| 41 | 13 sympy/series/gruntz.py | 1 | 119| 1345 | 21328 | 196078 | 
| 42 | 14 examples/advanced/gibbs_phenomenon.py | 129 | 153| 226 | 21554 | 197192 | 
| 43 | 15 examples/beginner/series.py | 1 | 29| 128 | 21682 | 197320 | 
| **-> 44 <-** | **15 sympy/printing/latex.py** | 1524 | 1614| 804 | 22486 | 197320 | 


## Missing Patch Files

 * 1: sympy/printing/latex.py
 * 2: sympy/printing/pretty/pretty.py

### Hint

```
> That is, it really does represent it as the formula Sum((-1)**n/factorial(2_n + 1)_x**n, (n, 0, oo)) (albiet, not simplified). It out to print it like this, so you can see that that's what it's working with.

I got to admit that not much discussion was done on the printing aspect of Formal Power Series.  When I first wrote the code, I tried to keep it as similar as possible to what series has to offer. Since, Formal Power Series is all about a formula computed for the series expansion, I guess this is a good idea. +1

> Side question: if you enter something it can't compute, it just returns the function
> 
> In [25]: fps(tan(x))
> Out[25]: tan(x)
> Is that intentional? It seems like it ought to raise an exception in that case.

This is again similar to what series does. Return it in the original form, if it's unable to compute the expansion. 

\`\`\`
>>> series(log(x))
log(x)
\`\`\`

If we want to raise an exception, the inability to compute can be from various reasons:
1. This is simply not covered by the algorithm.
2. SymPy does not have the required capabilities(eg. we need to construct a differential equation as part of the algorithm).
3. There is some bug in the code (that can be fixed ofcourse).

I am not sure here. Should we raise an exception or keep it just like `series`?

I guess it depends on what the use-cases for fps are.  FWIW, I think series returning expressions unchanged is not so great either (but that's somewhat part of a bigger problem, where the type of series produced by `series` is not very well-defined). 

```

## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -1605,7 +1605,7 @@ def _print_FourierSeries(self, s):
         return self._print_Add(s.truncate()) + self._print(' + \ldots')
 
     def _print_FormalPowerSeries(self, s):
-        return self._print_Add(s.truncate())
+        return self._print_Add(s.infinite)
 
     def _print_FiniteField(self, expr):
         return r"\mathbb{F}_{%s}" % expr.mod
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -1629,7 +1629,7 @@ def _print_FourierSeries(self, s):
         return self._print_Add(s.truncate()) + self._print(dots)
 
     def _print_FormalPowerSeries(self, s):
-        return self._print_Add(s.truncate())
+        return self._print_Add(s.infinite)
 
     def _print_SeqFormula(self, s):
         if self._use_unicode:

```

## Test Patch

```diff
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -3430,20 +3430,32 @@ def test_pretty_FourierSeries():
 def test_pretty_FormalPowerSeries():
     f = fps(log(1 + x))
 
+
     ascii_str = \
 """\
-     2    3    4    5        \n\
-    x    x    x    x     / 6\\\n\
-x - -- + -- - -- + -- + O\\x /\n\
-    2    3    4    5         \
+  oo             \n\
+____             \n\
+\   `            \n\
+ \         -k  k \n\
+  \   -(-1)  *x  \n\
+  /   -----------\n\
+ /         k     \n\
+/___,            \n\
+k = 1            \
 """
 
     ucode_str = \
 u("""\
-     2    3    4    5        \n\
-    x    x    x    x     ⎛ 6⎞\n\
-x - ── + ── - ── + ── + O⎝x ⎠\n\
-    2    3    4    5         \
+  ∞              \n\
+ ____            \n\
+ ╲               \n\
+  ╲        -k  k \n\
+   ╲  -(-1)  ⋅x  \n\
+   ╱  ───────────\n\
+  ╱        k     \n\
+ ╱               \n\
+ ‾‾‾‾            \n\
+k = 1            \
 """)
 
     assert pretty(f) == ascii_str
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -600,7 +600,7 @@ def test_latex_FourierSeries():
 
 
 def test_latex_FormalPowerSeries():
-    latex_str = r'x - \frac{x^{2}}{2} + \frac{x^{3}}{3} - \frac{x^{4}}{4} + \frac{x^{5}}{5} + \mathcal{O}\left(x^{6}\right)'
+    latex_str = r'\sum_{k=1}^{\infty} - \frac{\left(-1\right)^{- k}}{k} x^{k}'
     assert latex(fps(log(1 + x))) == latex_str
 
 

```


## Code snippets

### 1 - sympy/series/formal.py:

Start line: 1182, End line: 1256

```python
def fps(f, x=None, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    """Generates Formal Power Series of f.

    Returns the formal series expansion of ``f`` around ``x = x0``
    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.

    Formal Power Series is represented using an explicit formula
    computed using different algorithms.

    See :func:`compute_fps` for the more details regarding the computation
    of formula.

    Parameters
    ==========

    x : Symbol, optional
        If x is None and ``f`` is univariate, the univariate symbols will be
        supplied, otherwise an error will be raised.
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Examples
    ========

    >>> from sympy import fps, O, ln, atan
    >>> from sympy.abc import x

    Rational Functions

    >>> fps(ln(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.compute_fps
    """
    f = sympify(f)

    if x is None:
        free = f.free_symbols
        if len(free) == 1:
            x = free.pop()
        elif not free:
            return f
        else:
            raise NotImplementedError("multivariate formal power series")

    result = compute_fps(f, x, x0, dir, hyper, order, rational, full)

    if result is None:
        return f

    return FormalPowerSeries(f, x, x0, dir, result)
```
### 2 - sympy/series/formal.py:

Start line: 752, End line: 829

```python
def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
    if x0 in [S.Infinity, -S.Infinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1/x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1/x), result[2].subs(x, 1/x))
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b),
                result[2].subs(x, rep2 + rep2b))

    if f.is_polynomial(x):
        return None

    #  Break instances of Add
    #  this allows application of different
    #  algorithms on different terms increasing the
    #  range of admissible functions.
    if isinstance(f, Add):
        result = False
        ak = sequence(S.Zero, (0, oo))
        ind, xk = S.Zero, None
        for t in Add.make_args(f):
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            if res:
                if not result:
                    result = True
                    xk = res[1]
                if res[0].start > ak.start:
                    seq = ak
                    s, f = ak.start, res[0].start
                else:
                    seq = res[0]
                    s, f = res[0].start, ak.start
                save = Add(*[z[0]*z[1] for z in zip(seq[0:(f - s)], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                ind += t
        if result:
            return ak, xk, ind
        return None

    result = None

    # from here on it's x0=0 and dir=1 handling
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)

    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)

    if result is None:
        return None

    ak = sequence(result[0], (k, result[2], oo))
    xk = sequence(x**k, (k, 0, oo))
    ind = result[1]

    return ak, xk, ind
```
### 3 - sympy/series/formal.py:

Start line: 832, End line: 900

```python
def compute_fps(f, x, x0=0, dir=1, hyper=True, order=4, rational=True,
                full=False):
    """Computes the formula for Formal Power Series of a function.

    Tries to compute the formula by applying the following techniques
    (in order):

    * rational_algorithm
    * Hypergeomitric algorithm

    Parameters
    ==========

    x : Symbol
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Returns
    =======

    ak : sequence
        Sequence of coefficients.
    xk : sequence
        Sequence of powers of x.
    ind : Expr
        Independent terms.
    mul : Pow
        Common terms.

    See Also
    ========

    sympy.series.formal.rational_algorithm
    sympy.series.formal.hyper_algorithm
    """
    f = sympify(f)
    x = sympify(x)

    if not f.has(x):
        return None

    x0 = sympify(x0)

    if dir == '+':
        dir = S.One
    elif dir == '-':
        dir = -S.One
    elif dir not in [S.One, -S.One]:
        raise ValueError("Dir must be '+' or '-'")
    else:
        dir = sympify(dir)

    return _compute_fps(f, x, x0, dir, hyper, order, rational, full)
```
### 4 - sympy/series/formal.py:

Start line: 903, End line: 979

```python
class FormalPowerSeries(SeriesBase):
    """Represents Formal Power Series of a function.

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1]

    @property
    def x0(self):
        return self.args[2]

    @property
    def dir(self):
        return self.args[3]

    @property
    def ak(self):
        return self.args[4][0]

    @property
    def xk(self):
        return self.args[4][1]

    @property
    def ind(self):
        return self.args[4][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
        from sympy.concrete import Sum
        ak, xk = self.ak, self.xk
        k = ak.variables[0]
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))

        return self.ind + inf_sum

    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
        xterm, pow_x = term.as_independent(self.x)[1].as_base_exp()
        if not xterm.has(self.x):
            return S.Zero
        return pow_x
```
### 5 - sympy/series/formal.py:

Start line: 1071, End line: 1113

```python
class FormalPowerSeries(SeriesBase):

    def integrate(self, x=None):
        """Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> f.integrate((x, 0, 1))
        -cos(1) + 1
        """
        from sympy.integrals import integrate

        if x is None:
            x = self.x
        elif iterable(x):
            return integrate(self.function, x)

        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        ind += (f - ind).limit(x, 0)  # constant of integration

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1),
                          (k, ak.start + 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))
```
### 6 - sympy/series/fourier.py:

Start line: 321, End line: 397

```python
def fourier_series(f, limits=None):
    """Computes Fourier sine/cosine series expansion.

    Returns a :class:`FourierSeries` object.

    Examples
    ========

    >>> from sympy import fourier_series, pi, cos
    >>> from sympy.abc import x

    >>> s = fourier_series(x**2, (x, -pi, pi))
    >>> s.truncate(n=3)
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Notes
    =====

    Computing Fourier series can be slow
    due to the integration required in computing
    an, bn.

    It is faster to compute Fourier series of a function
    by using shifting and scaling on an already
    computed Fourier series rather than computing
    again.

    e.g. If the Fourier series of ``x**2`` is known
    the Fourier series of ``x**2 - 1`` can be found by shifting by ``-1``.

    See Also
    ========

    sympy.series.fourier.FourierSeries

    References
    ==========

    .. [1] mathworld.wolfram.com/FourierSeries.html
    """
    f = sympify(f)

    limits = _process_limits(f, limits)
    x = limits[0]

    if x not in f.free_symbols:
        return f

    n = Dummy('n')
    neg_f = f.subs(x, -x)
    if f == neg_f:
        a0, an = fourier_cos_seq(f, limits, n)
        bn = SeqFormula(0, (1, oo))
    elif f == -neg_f:
        a0 = S.Zero
        an = SeqFormula(0, (1, oo))
        bn = fourier_sin_seq(f, limits, n)
    else:
        a0, an = fourier_cos_seq(f, limits, n)
        bn = fourier_sin_seq(f, limits, n)

    return FourierSeries(f, limits, (a0, an, bn))
```
### 7 - sympy/series/formal.py:

Start line: 1048, End line: 1069

```python
class FormalPowerSeries(SeriesBase):

    def _eval_derivative(self, x):
        f = self.function.diff(x)
        ind = self.ind.diff(x)

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t * (pow_xk + pow_x)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1),
                          (k, ak.start - 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))
```
### 8 - sympy/series/formal.py:

Start line: 1017, End line: 1046

```python
class FormalPowerSeries(SeriesBase):

    def _eval_term(self, pt):
        try:
            pt_xk = self.xk.coeff(pt)
            pt_ak = self.ak.coeff(pt).simplify()  # Simplify the coefficients
        except IndexError:
            term = S.Zero
        else:
            term = (pt_ak * pt_xk)

        if self.ind:
            ind = S.Zero
            for t in Add.make_args(self.ind):
                pow_x = self._get_pow_x(t)
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            term += ind

        return term.collect(self.x)

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def _eval_as_leading_term(self, x):
        for t in self:
            if t is not S.Zero:
                return t
```
### 9 - sympy/series/formal.py:

Start line: 1153, End line: 1179

```python
class FormalPowerSeries(SeriesBase):

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.func(-self.function, self.x, self.x0, self.dir,
                         (-self.ak, self.xk, -self.ind))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other = sympify(other)

        if other.has(self.x):
            return Mul(self, other)

        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def __rmul__(self, other):
        return self.__mul__(other)
```
### 10 - sympy/simplify/fu.py:

Start line: 1, End line: 189

```python
"""
Implementation of the trigsimp algorithm by Fu et al.

The idea behind the ``fu`` algorithm is to use a sequence of rules, applied
in what is heuristically known to be a smart order, to select a simpler
expression that is equivalent to the input.

There are transform rules in which a single rule is applied to the
expression tree. The following are just mnemonic in nature; see the
docstrings for examples.

    TR0 - simplify expression
    TR1 - sec-csc to cos-sin
    TR2 - tan-cot to sin-cos ratio
    TR2i - sin-cos ratio to tan
    TR3 - angle canonicalization
    TR4 - functions at special angles
    TR5 - powers of sin to powers of cos
    TR6 - powers of cos to powers of sin
    TR7 - reduce cos power (increase angle)
    TR8 - expand products of sin-cos to sums
    TR9 - contract sums of sin-cos to products
    TR10 - separate sin-cos arguments
    TR10i - collect sin-cos arguments
    TR11 - reduce double angles
    TR12 - separate tan arguments
    TR12i - collect tan arguments
    TR13 - expand product of tan-cot
    TRmorrie - prod(cos(x*2**i), (i, 0, k - 1)) -> sin(2**k*x)/(2**k*sin(x))
    TR14 - factored powers of sin or cos to cos or sin power
    TR15 - negative powers of sin to cot power
    TR16 - negative powers of cos to tan power
    TR22 - tan-cot powers to negative powers of sec-csc functions
    TR111 - negative sin-cos-tan powers to csc-sec-cot

There are 4 combination transforms (CTR1 - CTR4) in which a sequence of
transformations are applied and the simplest expression is selected from
a few options.

Finally, there are the 2 rule lists (RL1 and RL2), which apply a
sequence of transformations and combined transformations, and the ``fu``
algorithm itself, which applies rules and rule lists and selects the
best expressions. There is also a function ``L`` which counts the number
of trigonometric functions that appear in the expression.

Other than TR0, re-writing of expressions is not done by the transformations.
e.g. TR10i finds pairs of terms in a sum that are in the form like
``cos(x)*cos(y) + sin(x)*sin(y)``. Such expression are targeted in a bottom-up
traversal of the expression, but no manipulation to make them appear is
attempted. For example,

    Set-up for examples below:

    >>> from sympy.simplify.fu import fu, L, TR9, TR10i, TR11
    >>> from sympy import factor, sin, cos, powsimp
    >>> from sympy.abc import x, y, z, a
    >>> from time import time

>>> eq = cos(x + y)/cos(x)
>>> TR10i(eq.expand(trig=True))
-sin(x)*sin(y)/cos(x) + cos(y)

If the expression is put in "normal" form (with a common denominator) then
the transformation is successful:

>>> TR10i(_.normal())
cos(x + y)/cos(x)

TR11's behavior is similar. It rewrites double angles as smaller angles but
doesn't do any simplification of the result.

>>> TR11(sin(2)**a*cos(1)**(-a), 1)
(2*sin(1)*cos(1))**a*cos(1)**(-a)
>>> powsimp(_)
(2*sin(1))**a

The temptation is to try make these TR rules "smarter" but that should really
be done at a higher level; the TR rules should try maintain the "do one thing
well" principle.  There is one exception, however. In TR10i and TR9 terms are
recognized even when they are each multiplied by a common factor:

>>> fu(a*cos(x)*cos(y) + a*sin(x)*sin(y))
a*cos(x - y)

Factoring with ``factor_terms`` is used but it it "JIT"-like, being delayed
until it is deemed necessary. Furthermore, if the factoring does not
help with the simplification, it is not retained, so
``a*cos(x)*cos(y) + a*sin(x)*sin(z)`` does not become the factored
(but unsimplified in the trigonometric sense) expression:

>>> fu(a*cos(x)*cos(y) + a*sin(x)*sin(z))
a*sin(x)*sin(z) + a*cos(x)*cos(y)

In some cases factoring might be a good idea, but the user is left
to make that decision. For example:

>>> expr=((15*sin(2*x) + 19*sin(x + y) + 17*sin(x + z) + 19*cos(x - z) +
... 25)*(20*sin(2*x) + 15*sin(x + y) + sin(y + z) + 14*cos(x - z) +
... 14*cos(y - z))*(9*sin(2*y) + 12*sin(y + z) + 10*cos(x - y) + 2*cos(y -
... z) + 18)).expand(trig=True).expand()

In the expanded state, there are nearly 1000 trig functions:

>>> L(expr)
932

If the expression where factored first, this would take time but the
resulting expression would be transformed very quickly:

>>> def clock(f, n=2):
...    t=time(); f(); return round(time()-t, n)
...
>>> clock(lambda: factor(expr))  # doctest: +SKIP
0.86
>>> clock(lambda: TR10i(expr), 3)  # doctest: +SKIP
0.016

If the unexpanded expression is used, the transformation takes longer but
not as long as it took to factor it and then transform it:

>>> clock(lambda: TR10i(expr), 2)  # doctest: +SKIP
0.28

So neither expansion nor factoring is used in ``TR10i``: if the
expression is already factored (or partially factored) then expansion
with ``trig=True`` would destroy what is already known and take
longer; if the expression is expanded, factoring may take longer than
simply applying the transformation itself.

Although the algorithms should be canonical, always giving the same
result, they may not yield the best result. This, in general, is
the nature of simplification where searching all possible transformation
paths is very expensive. Here is a simple example. There are 6 terms
in the following sum:

>>> expr = (sin(x)**2*cos(y)*cos(z) + sin(x)*sin(y)*cos(x)*cos(z) +
... sin(x)*sin(z)*cos(x)*cos(y) + sin(y)*sin(z)*cos(x)**2 + sin(y)*sin(z) +
... cos(y)*cos(z))
>>> args = expr.args

Serendipitously, fu gives the best result:

>>> fu(expr)
3*cos(y - z)/2 - cos(2*x + y + z)/2

But if different terms were combined, a less-optimal result might be
obtained, requiring some additional work to get better simplification,
but still less than optimal. The following shows an alternative form
of ``expr`` that resists optimal simplification once a given step
is taken since it leads to a dead end:

>>> TR9(-cos(x)**2*cos(y + z) + 3*cos(y - z)/2 +
...     cos(y + z)/2 + cos(-2*x + y + z)/4 - cos(2*x + y + z)/4)
sin(2*x)*sin(y + z)/2 - cos(x)**2*cos(y + z) + 3*cos(y - z)/2 + cos(y + z)/2

Here is a smaller expression that exhibits the same behavior:

>>> a = sin(x)*sin(z)*cos(x)*cos(y) + sin(x)*sin(y)*cos(x)*cos(z)
>>> TR10i(a)
sin(x)*sin(y + z)*cos(x)
>>> newa = _
>>> TR10i(expr - a)  # this combines two more of the remaining terms
sin(x)**2*cos(y)*cos(z) + sin(y)*sin(z)*cos(x)**2 + cos(y - z)
>>> TR10i(_ + newa) == _ + newa  # but now there is no more simplification
True

Without getting lucky or trying all possible pairings of arguments, the
final result may be less than optimal and impossible to find without
better heuristics or brute force trial of all possibilities.

Notes
=====

This work was started by Dimitar Vlahovski at the Technological School
"Electronic systems" (30.11.2011).

References
==========

Fu, Hongguang, Xiuqin Zhong, and Zhenbing Zeng. "Automated and readable
simplification of trigonometric expressions." Mathematical and computer
modelling 44.11 (2006): 1169-1177.
http://rfdz.ph-noe.ac.at/fileadmin/Mathematik_Uploads/ACDCA/DESTIME2006/DES_contribs/Fu/simplification.pdf

http://www.sosmath.com/trig/Trig5/trig5/pdf/pdf.html gives a formula sheet.

"""

from __future__ import print_function, division
```
### 19 - sympy/printing/latex.py:

Start line: 987, End line: 1001

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
### 30 - sympy/printing/latex.py:

Start line: 903, End line: 985

```python
class LatexPrinter(Printer):

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

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 34 - sympy/printing/latex.py:

Start line: 1928, End line: 2058

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
    $8 \sqrt{2} \tau^{\frac{7}{2}}$

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
### 35 - sympy/printing/latex.py:

Start line: 650, End line: 717

```python
class LatexPrinter(Printer):

    def _print_Function(self, expr, exp=None):
        '''
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
### 39 - sympy/printing/latex.py:

Start line: 418, End line: 462

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

                return tex % (self.parenthesize(expr.base, PRECEDENCE['Pow']),
                              self._print(expr.exp))
```
### 44 - sympy/printing/latex.py:

Start line: 1524, End line: 1614

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
        return self._print_Add(s.truncate()) + self._print(' + \ldots')

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.truncate())

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"
```
