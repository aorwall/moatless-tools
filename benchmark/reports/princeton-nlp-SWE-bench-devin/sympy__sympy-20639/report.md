# sympy__sympy-20639

| **sympy/sympy** | `eb926a1d0c1158bf43f01eaf673dc84416b5ebb1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 30497 |
| **Avg pos** | 77.0 |
| **Min pos** | 77 |
| **Max pos** | 77 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -1902,12 +1902,12 @@ def _print_Mul(self, product):
             return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)
 
     # A helper function for _print_Pow to print x**(1/n)
-    def _print_nth_root(self, base, expt):
+    def _print_nth_root(self, base, root):
         bpretty = self._print(base)
 
         # In very simple cases, use a single-char root sign
         if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
-            and expt is S.Half and bpretty.height() == 1
+            and root == 2 and bpretty.height() == 1
             and (bpretty.width() == 1
                  or (base.is_Integer and base.is_nonnegative))):
             return prettyForm(*bpretty.left('\N{SQUARE ROOT}'))
@@ -1915,14 +1915,13 @@ def _print_nth_root(self, base, expt):
         # Construct root sign, start with the \/ shape
         _zZ = xobj('/', 1)
         rootsign = xobj('\\', 1) + _zZ
-        # Make exponent number to put above it
-        if isinstance(expt, Rational):
-            exp = str(expt.q)
-            if exp == '2':
-                exp = ''
-        else:
-            exp = str(expt.args[0])
-        exp = exp.ljust(2)
+        # Constructing the number to put on root
+        rpretty = self._print(root)
+        # roots look bad if they are not a single line
+        if rpretty.height() != 1:
+            return self._print(base)**self._print(1/root)
+        # If power is half, no number should appear on top of root sign
+        exp = '' if root == 2 else str(rpretty).ljust(2)
         if len(exp) > 2:
             rootsign = ' '*(len(exp) - 2) + rootsign
         # Stack the exponent
@@ -1954,8 +1953,9 @@ def _print_Pow(self, power):
             if e is S.NegativeOne:
                 return prettyForm("1")/self._print(b)
             n, d = fraction(e)
-            if n is S.One and d.is_Atom and not e.is_Integer and self._settings['root_notation']:
-                return self._print_nth_root(b, e)
+            if n is S.One and d.is_Atom and not e.is_Integer and (e.is_Rational or d.is_Symbol) \
+                    and self._settings['root_notation']:
+                return self._print_nth_root(b, d)
             if e.is_Rational and e < 0:
                 return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/pretty/pretty.py | 1905 | 1910 | - | 1 | -
| sympy/printing/pretty/pretty.py | 1918 | 1925 | - | 1 | -
| sympy/printing/pretty/pretty.py | 1957 | 1958 | 77 | 1 | 30497


## Problem Statement

```
inaccurate rendering of pi**(1/E)
This claims to be version 1.5.dev; I just merged from the project master, so I hope this is current.  I didn't notice this bug among others in printing.pretty.

\`\`\`
In [52]: pi**(1/E)                                                               
Out[52]: 
-1___
╲╱ π 

\`\`\`
LaTeX and str not fooled:
\`\`\`
In [53]: print(latex(pi**(1/E)))                                                 
\pi^{e^{-1}}

In [54]: str(pi**(1/E))                                                          
Out[54]: 'pi**exp(-1)'
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/pretty/pretty.py** | 1722 | 1735| 180 | 180 | 24148 | 
| 2 | 2 sympy/core/numbers.py | 3410 | 3483| 474 | 654 | 53801 | 
| 3 | 3 sympy/printing/latex.py | 1138 | 1149| 156 | 810 | 83154 | 
| 4 | 4 sympy/functions/special/error_functions.py | 1009 | 1115| 772 | 1582 | 103343 | 
| 5 | 5 sympy/printing/mathml.py | 1938 | 1951| 130 | 1712 | 120233 | 
| 6 | 5 sympy/core/numbers.py | 3484 | 3555| 435 | 2147 | 120233 | 
| 7 | 5 sympy/printing/latex.py | 1127 | 1136| 126 | 2273 | 120233 | 
| 8 | **5 sympy/printing/pretty/pretty.py** | 1697 | 1720| 258 | 2531 | 120233 | 
| 9 | 5 sympy/printing/latex.py | 613 | 654| 449 | 2980 | 120233 | 
| 10 | 5 sympy/printing/latex.py | 2713 | 2915| 2702 | 5682 | 120233 | 
| 11 | 5 sympy/functions/special/error_functions.py | 1349 | 1381| 170 | 5852 | 120233 | 
| 12 | **5 sympy/printing/pretty/pretty.py** | 136 | 203| 639 | 6491 | 120233 | 
| 13 | **5 sympy/printing/pretty/pretty.py** | 106 | 122| 199 | 6690 | 120233 | 
| 14 | 6 sympy/functions/special/elliptic_integrals.py | 361 | 409| 466 | 7156 | 124583 | 
| 15 | 6 sympy/printing/latex.py | 1070 | 1125| 577 | 7733 | 124583 | 
| 16 | 6 sympy/printing/latex.py | 1504 | 1521| 188 | 7921 | 124583 | 
| 17 | 7 sympy/functions/elementary/exponential.py | 170 | 197| 242 | 8163 | 134435 | 
| 18 | 7 sympy/printing/latex.py | 1151 | 1235| 813 | 8976 | 134435 | 
| 19 | 7 sympy/functions/special/error_functions.py | 1168 | 1269| 890 | 9866 | 134435 | 
| 20 | 7 sympy/functions/special/elliptic_integrals.py | 421 | 452| 438 | 10304 | 134435 | 
| 21 | 8 sympy/printing/pycode.py | 567 | 608| 405 | 10709 | 143819 | 
| 22 | 8 sympy/printing/latex.py | 440 | 468| 310 | 11019 | 143819 | 
| 23 | **8 sympy/printing/pretty/pretty.py** | 1626 | 1639| 142 | 11161 | 143819 | 
| 24 | **8 sympy/printing/pretty/pretty.py** | 2482 | 2506| 211 | 11372 | 143819 | 
| 25 | **8 sympy/printing/pretty/pretty.py** | 1466 | 1487| 231 | 11603 | 143819 | 
| 26 | **8 sympy/printing/pretty/pretty.py** | 1657 | 1674| 184 | 11787 | 143819 | 
| 27 | 9 examples/advanced/pidigits.py | 1 | 35| 290 | 12077 | 144602 | 
| 28 | 9 sympy/printing/mathml.py | 756 | 781| 211 | 12288 | 144602 | 
| 29 | **9 sympy/printing/pretty/pretty.py** | 1676 | 1695| 194 | 12482 | 144602 | 
| 30 | 9 sympy/functions/special/error_functions.py | 1117 | 1165| 500 | 12982 | 144602 | 
| 31 | 9 sympy/printing/latex.py | 2636 | 2644| 126 | 13108 | 144602 | 
| 32 | 9 sympy/printing/latex.py | 1044 | 1053| 140 | 13248 | 144602 | 
| 33 | **9 sympy/printing/pretty/pretty.py** | 1382 | 1464| 773 | 14021 | 144602 | 
| 34 | 9 sympy/printing/latex.py | 2626 | 2634| 120 | 14141 | 144602 | 
| 35 | 9 sympy/printing/latex.py | 794 | 827| 342 | 14483 | 144602 | 
| 36 | 9 sympy/printing/mathml.py | 805 | 883| 630 | 15113 | 144602 | 
| 37 | 9 sympy/printing/latex.py | 1378 | 1441| 810 | 15923 | 144602 | 
| 38 | 9 sympy/printing/mathml.py | 1044 | 1096| 472 | 16395 | 144602 | 
| 39 | 9 sympy/printing/latex.py | 1287 | 1352| 668 | 17063 | 144602 | 
| 40 | 9 sympy/printing/latex.py | 1237 | 1268| 308 | 17371 | 144602 | 
| 41 | 9 sympy/printing/latex.py | 1270 | 1285| 135 | 17506 | 144602 | 
| 42 | **9 sympy/printing/pretty/pretty.py** | 94 | 104| 135 | 17641 | 144602 | 
| 43 | 9 sympy/core/numbers.py | 3866 | 3899| 287 | 17928 | 144602 | 
| 44 | 9 sympy/printing/latex.py | 542 | 611| 701 | 18629 | 144602 | 
| 45 | 9 sympy/printing/latex.py | 2037 | 2049| 167 | 18796 | 144602 | 
| 46 | 9 sympy/functions/special/elliptic_integrals.py | 411 | 419| 117 | 18913 | 144602 | 
| 47 | 9 sympy/functions/special/elliptic_integrals.py | 194 | 240| 500 | 19413 | 144602 | 
| 48 | 9 sympy/printing/latex.py | 1950 | 1960| 141 | 19554 | 144602 | 
| 49 | 9 sympy/core/numbers.py | 1340 | 1370| 333 | 19887 | 144602 | 
| 50 | 9 sympy/printing/latex.py | 656 | 672| 190 | 20077 | 144602 | 
| 51 | **9 sympy/printing/pretty/pretty.py** | 2508 | 2519| 145 | 20222 | 144602 | 
| 52 | **9 sympy/printing/pretty/pretty.py** | 1334 | 1380| 403 | 20625 | 144602 | 
| 53 | 9 sympy/printing/latex.py | 1973 | 1992| 228 | 20853 | 144602 | 
| 54 | 9 sympy/printing/latex.py | 2051 | 2065| 184 | 21037 | 144602 | 
| 55 | 9 sympy/core/numbers.py | 3695 | 3755| 430 | 21467 | 144602 | 
| 56 | 9 sympy/functions/special/error_functions.py | 1309 | 1331| 280 | 21747 | 144602 | 
| 57 | **9 sympy/printing/pretty/pretty.py** | 761 | 784| 261 | 22008 | 144602 | 
| 58 | 9 sympy/printing/latex.py | 2068 | 2078| 124 | 22132 | 144602 | 
| 59 | **9 sympy/printing/pretty/pretty.py** | 267 | 333| 576 | 22708 | 144602 | 
| 60 | 9 sympy/printing/mathml.py | 1695 | 1779| 658 | 23366 | 144602 | 
| 61 | 9 sympy/printing/latex.py | 860 | 934| 706 | 24072 | 144602 | 
| 62 | 9 sympy/printing/pycode.py | 1028 | 1044| 195 | 24267 | 144602 | 
| 63 | 9 sympy/printing/latex.py | 751 | 782| 273 | 24540 | 144602 | 
| 64 | 9 sympy/printing/mathml.py | 1953 | 2050| 808 | 25348 | 144602 | 
| 65 | 9 sympy/functions/elementary/exponential.py | 125 | 168| 305 | 25653 | 144602 | 
| 66 | **9 sympy/printing/pretty/pretty.py** | 124 | 134| 134 | 25787 | 144602 | 
| 67 | 10 sympy/core/power.py | 362 | 443| 793 | 26580 | 160201 | 
| 68 | 11 sympy/functions/elementary/trigonometric.py | 134 | 206| 524 | 27104 | 190521 | 
| 69 | 11 sympy/functions/special/error_functions.py | 2539 | 2555| 170 | 27274 | 190521 | 
| 70 | 12 sympy/codegen/cfunctions.py | 21 | 83| 384 | 27658 | 193705 | 
| 71 | **12 sympy/printing/pretty/pretty.py** | 1545 | 1603| 563 | 28221 | 193705 | 
| 72 | 12 sympy/printing/latex.py | 1443 | 1502| 782 | 29003 | 193705 | 
| 73 | 12 sympy/printing/mathml.py | 1098 | 1157| 429 | 29432 | 193705 | 
| 74 | 12 sympy/printing/latex.py | 829 | 841| 159 | 29591 | 193705 | 
| 75 | 12 sympy/functions/special/elliptic_integrals.py | 282 | 313| 389 | 29980 | 193705 | 
| 76 | 12 sympy/printing/mathml.py | 1173 | 1210| 333 | 30313 | 193705 | 
| **-> 77 <-** | **12 sympy/printing/pretty/pretty.py** | 1950 | 1965| 184 | 30497 | 193705 | 


### Hint

```
I can confirm this bug on master. Looks like it's been there a while
https://github.com/sympy/sympy/blob/2d700c4b3c0871a26741456787b0555eed9d5546/sympy/printing/pretty/pretty.py#L1814

`1/E` is `exp(-1)` which has totally different arg structure than something like `1/pi`:

\`\`\`
>>> (1/E).args
(-1,)
>>> (1/pi).args
(pi, -1)
\`\`\`
@ethankward nice!  Also, the use of `str` there isn't correct:
\`\`\`
>>> pprint(7**(1/(pi)))                                                                                                                                                          
pi___
╲╱ 7 

>>> pprint(pi**(1/(pi)))                                                                                                                                                        
pi___
╲╱ π 

>>> pprint(pi**(1/(EulerGamma)))                                                                                                                                                
EulerGamma___
        ╲╱ π 
\`\`\`
(`pi` and `EulerGamma` were not pretty printed)
I guess str is used because it's hard to put 2-D stuff in the corner of the radical like that. But I think it would be better to recursively call the pretty printer, and if it is multiline, or maybe even if it is a more complicated expression than just a single number or symbol name, then print it without the radical like

\`\`\`
  1
  ─
  e
π
\`\`\`

or

\`\`\`
 ⎛ -1⎞
 ⎝e  ⎠
π
```

## Patch

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -1902,12 +1902,12 @@ def _print_Mul(self, product):
             return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)
 
     # A helper function for _print_Pow to print x**(1/n)
-    def _print_nth_root(self, base, expt):
+    def _print_nth_root(self, base, root):
         bpretty = self._print(base)
 
         # In very simple cases, use a single-char root sign
         if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
-            and expt is S.Half and bpretty.height() == 1
+            and root == 2 and bpretty.height() == 1
             and (bpretty.width() == 1
                  or (base.is_Integer and base.is_nonnegative))):
             return prettyForm(*bpretty.left('\N{SQUARE ROOT}'))
@@ -1915,14 +1915,13 @@ def _print_nth_root(self, base, expt):
         # Construct root sign, start with the \/ shape
         _zZ = xobj('/', 1)
         rootsign = xobj('\\', 1) + _zZ
-        # Make exponent number to put above it
-        if isinstance(expt, Rational):
-            exp = str(expt.q)
-            if exp == '2':
-                exp = ''
-        else:
-            exp = str(expt.args[0])
-        exp = exp.ljust(2)
+        # Constructing the number to put on root
+        rpretty = self._print(root)
+        # roots look bad if they are not a single line
+        if rpretty.height() != 1:
+            return self._print(base)**self._print(1/root)
+        # If power is half, no number should appear on top of root sign
+        exp = '' if root == 2 else str(rpretty).ljust(2)
         if len(exp) > 2:
             rootsign = ' '*(len(exp) - 2) + rootsign
         # Stack the exponent
@@ -1954,8 +1953,9 @@ def _print_Pow(self, power):
             if e is S.NegativeOne:
                 return prettyForm("1")/self._print(b)
             n, d = fraction(e)
-            if n is S.One and d.is_Atom and not e.is_Integer and self._settings['root_notation']:
-                return self._print_nth_root(b, e)
+            if n is S.One and d.is_Atom and not e.is_Integer and (e.is_Rational or d.is_Symbol) \
+                    and self._settings['root_notation']:
+                return self._print_nth_root(b, d)
             if e.is_Rational and e < 0:
                 return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))
 

```

## Test Patch

```diff
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -5942,7 +5942,11 @@ def test_PrettyPoly():
 
 def test_issue_6285():
     assert pretty(Pow(2, -5, evaluate=False)) == '1 \n--\n 5\n2 '
-    assert pretty(Pow(x, (1/pi))) == 'pi___\n\\/ x '
+    assert pretty(Pow(x, (1/pi))) == \
+    ' 1 \n'\
+    ' --\n'\
+    ' pi\n'\
+    'x  '
 
 
 def test_issue_6359():
@@ -7205,6 +7209,51 @@ def test_is_combining():
         [False, True, False, False]
 
 
+def test_issue_17616():
+    assert pretty(pi**(1/exp(1))) == \
+   '  / -1\\\n'\
+   '  \e  /\n'\
+   'pi     '
+
+    assert upretty(pi**(1/exp(1))) == \
+   ' ⎛ -1⎞\n'\
+   ' ⎝ℯ  ⎠\n'\
+   'π     '
+
+    assert pretty(pi**(1/pi)) == \
+    '  1 \n'\
+    '  --\n'\
+    '  pi\n'\
+    'pi  '
+
+    assert upretty(pi**(1/pi)) == \
+    ' 1\n'\
+    ' ─\n'\
+    ' π\n'\
+    'π '
+
+    assert pretty(pi**(1/EulerGamma)) == \
+    '      1     \n'\
+    '  ----------\n'\
+    '  EulerGamma\n'\
+    'pi          '
+
+    assert upretty(pi**(1/EulerGamma)) == \
+    ' 1\n'\
+    ' ─\n'\
+    ' γ\n'\
+    'π '
+
+    z = Symbol("x_17")
+    assert upretty(7**(1/z)) == \
+    'x₁₇___\n'\
+    ' ╲╱ 7 '
+
+    assert pretty(7**(1/z)) == \
+    'x_17___\n'\
+    '  \\/ 7 '
+
+
 def test_issue_17857():
     assert pretty(Range(-oo, oo)) == '{..., -1, 0, 1, ...}'
     assert pretty(Range(oo, -oo, -1)) == '{..., 1, 0, -1, ...}'

```


## Code snippets

### 1 - sympy/printing/pretty/pretty.py:

Start line: 1722, End line: 1735

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
### 2 - sympy/core/numbers.py:

Start line: 3410, End line: 3483

```python
class Exp1(NumberSymbol, metaclass=Singleton):
    r"""The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """

    is_real = True
    is_positive = True
    is_negative = False  # XXX Forces is_negative/is_nonnegative
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"e"

    @staticmethod
    def __abs__():
        return S.Exp1

    def __int__(self):
        return 2

    def _as_mpf_val(self, prec):
        return mpf_e(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(2), Integer(3))
        elif issubclass(number_cls, Rational):
            pass

    def _eval_power(self, expt):
        from sympy import exp
        return exp(expt)

    def _eval_rewrite_as_sin(self, **kwargs):
        from sympy import sin
        I = S.ImaginaryUnit
        return sin(I + S.Pi/2) - I*sin(I)

    def _eval_rewrite_as_cos(self, **kwargs):
        from sympy import cos
        I = S.ImaginaryUnit
        return cos(I) + I*cos(I + S.Pi/2)

    def _sage_(self):
        import sage.all as sage
        return sage.e
```
### 3 - sympy/printing/latex.py:

Start line: 1138, End line: 1149

```python
class LatexPrinter(Printer):

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex
```
### 4 - sympy/functions/special/error_functions.py:

Start line: 1009, End line: 1115

```python
###############################################################################
#################### EXPONENTIAL INTEGRALS ####################################
###############################################################################

class Ei(Function):
    r"""
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Ei}(x) = \sum_{n=1}^\infty \frac{x^n}{n\, n!}
                                     + \log(x) + \gamma,

    where $\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\mathbb{C} \setminus (-\infty, 0]$.

    **Background**

    The name exponential integral comes from the following statement:

    .. math:: \operatorname{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t} \mathrm{d}t

    If the integral is interpreted as a Cauchy principal value, this statement
    holds for $x > 0$ and $\operatorname{Ei}(x)$ as defined above.

    Examples
    ========

    >>> from sympy import Ei, polar_lift, exp_polar, I, pi
    >>> from sympy.abc import x

    >>> Ei(-1)
    Ei(-1)

    This yields a real value:

    >>> Ei(-1).n(chop=True)
    -0.219383934395520

    On the other hand the analytic continuation is not real:

    >>> Ei(polar_lift(-1)).n(chop=True)
    -0.21938393439552 + 3.14159265358979*I

    The exponential integral has a logarithmic branch point at the origin:

    >>> Ei(x*exp_polar(2*I*pi))
    Ei(x) + 2*I*pi

    Differentiation is supported:

    >>> Ei(x).diff(x)
    exp(x)/x

    The exponential integral is related to many other special functions.
    For example:

    >>> from sympy import expint, Shi
    >>> Ei(x).rewrite(expint)
    -expint(1, x*exp_polar(I*pi)) - I*pi
    >>> Ei(x).rewrite(Shi)
    Chi(x) + Shi(x)

    See Also
    ========

    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma: Upper incomplete gamma function.

    References
    ==========

    .. [1] http://dlmf.nist.gov/6.6
    .. [2] https://en.wikipedia.org/wiki/Exponential_integral
    .. [3] Abramowitz & Stegun, section 5: http://people.math.sfu.ca/~cbm/aands/page_228.htm

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        elif z is S.NegativeInfinity:
            return S.Zero

        if z.is_zero:
            return S.NegativeInfinity

        nz, n = z.extract_branch_factor()
        if n:
            return Ei(nz) + 2*I*pi*n
```
### 5 - sympy/printing/mathml.py:

Start line: 1938, End line: 1951

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_elliptic_pi(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d6f1;'))
        x.appendChild(mi)
        y = self.dom.createElement('mfenced')
        if len(e.args) == 2:
            y.setAttribute("separators", "|")
        else:
            y.setAttribute("separators", ";|")
        for i in e.args:
            y.appendChild(self._print(i))
        x.appendChild(y)
        return x
```
### 6 - sympy/core/numbers.py:

Start line: 3484, End line: 3555

```python
E = S.Exp1


class Pi(NumberSymbol, metaclass=Singleton):
    r"""The `\pi` constant.

    Explanation
    ===========

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pi
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"\pi"

    @staticmethod
    def __abs__():
        return S.Pi

    def __int__(self):
        return 3

    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71), Rational(22, 7))

    def _sage_(self):
        import sage.all as sage
        return sage.pi
pi = S.Pi
```
### 7 - sympy/printing/latex.py:

Start line: 1127, End line: 1136

```python
class LatexPrinter(Printer):

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex
```
### 8 - sympy/printing/pretty/pretty.py:

Start line: 1697, End line: 1720

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
### 9 - sympy/printing/latex.py:

Start line: 613, End line: 654

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1 \
                and self._settings['root_notation']:
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
            base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
            p, q = expr.exp.p, expr.exp.q
            # issue #12886: add parentheses for superscripts raised to powers
            if expr.base.is_Symbol:
                base = self.parenthesize_super(base)
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and \
                expr.base.is_commutative:
            # special case for 1^(-x), issue 9216
            if expr.base == 1:
                return r"%s^{%s}" % (expr.base, expr.exp)
            # things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, exp=self._print(expr.exp))
            else:
                tex = r"%s^{%s}"
                return self._helper_print_standard_power(expr, tex)
```
### 10 - sympy/printing/latex.py:

Start line: 2713, End line: 2915

```python
@print_function(LatexPrinter)
def latex(expr, **settings):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    full_prec: boolean, optional
        If set to True, a floating point number is printed with full precision.
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
    parenthesize_super : boolean, optional
        If set to ``False``, superscripted expressions will not be parenthesized when
        powered. Default is ``True``, which parenthesizes the expression when powered.
    min: Integer or None, optional
        Sets the lower bound for the exponent to print floating point numbers in
        fixed-point format.
    max: Integer or None, optional
        Sets the upper bound for the exponent to print floating point numbers in
        fixed-point format.

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

    ``latex()`` also supports the builtin container types :class:`list`,
    :class:`tuple`, and :class:`dict`:

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \  y\right]$

    Unsupported types are rendered as monospaced plaintext:

    >>> print(latex(int))
    \mathtt{\text{<class 'int'>}}
    >>> print(latex("plain % text"))
    \mathtt{\text{plain \% text}}

    See :ref:`printer_method_example` for an example of how to override
    this behavior for your own types by implementing ``_latex``.

    .. versionchanged:: 1.7.0
        Unsupported types no longer have their ``str`` representation treated as valid latex.

    """
    return LatexPrinter(settings).doprint(expr)
```
### 12 - sympy/printing/pretty/pretty.py:

Start line: 136, End line: 203

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
### 13 - sympy/printing/pretty/pretty.py:

Start line: 106, End line: 122

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
### 23 - sympy/printing/pretty/pretty.py:

Start line: 1626, End line: 1639

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
### 24 - sympy/printing/pretty/pretty.py:

Start line: 2482, End line: 2506

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
            return self._print_number_function(e, '\N{GREEK SMALL LETTER GAMMA}')
        else:
            return self._print_number_function(e, "stieltjes")
```
### 25 - sympy/printing/pretty/pretty.py:

Start line: 1466, End line: 1487

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
### 26 - sympy/printing/pretty/pretty.py:

Start line: 1657, End line: 1674

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
### 29 - sympy/printing/pretty/pretty.py:

Start line: 1676, End line: 1695

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
### 33 - sympy/printing/pretty/pretty.py:

Start line: 1382, End line: 1464

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
### 42 - sympy/printing/pretty/pretty.py:

Start line: 94, End line: 104

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
### 51 - sympy/printing/pretty/pretty.py:

Start line: 2508, End line: 2519

```python
class PrettyPrinter(Printer):

    def _print_KroneckerDelta(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(self._print(e.args[1])))
        if self._use_unicode:
            a = stringPict(pretty_symbol('delta'))
        else:
            a = stringPict('d')
        b = pform
        top = stringPict(*b.left(' '*a.width()))
        bot = stringPict(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))
```
### 52 - sympy/printing/pretty/pretty.py:

Start line: 1334, End line: 1380

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
### 57 - sympy/printing/pretty/pretty.py:

Start line: 761, End line: 784

```python
class PrettyPrinter(Printer):

    def _print_MatrixBase(self, e):
        D = self._print_matrix_contents(e)
        D.baseline = D.height()//2
        D = prettyForm(*D.parens('[', ']'))
        return D

    def _print_TensorProduct(self, expr):
        # This should somehow share the code with _print_WedgeProduct:
        circled_times = "\u2297"
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_WedgeProduct(self, expr):
        # This should somehow share the code with _print_TensorProduct:
        wedge_symbol = "\u2227"
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_Trace(self, e):
        D = self._print(e.arg)
        D = prettyForm(*D.parens('(',')'))
        D.baseline = D.height()//2
        D = prettyForm(*D.left('\n'*(0) + 'tr'))
        return D
```
### 59 - sympy/printing/pretty/pretty.py:

Start line: 267, End line: 333

```python
class PrettyPrinter(Printer):

    def _print_And(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, "\N{LOGICAL AND}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Or(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, "\N{LOGICAL OR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Xor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, "\N{XOR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Nand(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, "\N{NAND}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Nor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, "\N{NOR}")
        else:
            return self._print_Function(e, sort=True)

    def _print_Implies(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or "\N{RIGHTWARDS ARROW}", sort=False)
        else:
            return self._print_Function(e)

    def _print_Equivalent(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or "\N{LEFT RIGHT DOUBLE ARROW}")
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
### 66 - sympy/printing/pretty/pretty.py:

Start line: 124, End line: 134

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
### 71 - sympy/printing/pretty/pretty.py:

Start line: 1545, End line: 1603

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
            arrow = " \N{RIGHTWARDS ARROW FROM BAR} "
        else:
            arrow = " -> "
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        var_form = self._print(sig)

        return prettyForm(*stringPict.next(var_form, arrow, self._print(expr)), binding=8)
```
### 77 - sympy/printing/pretty/pretty.py:

Start line: 1950, End line: 1965

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
