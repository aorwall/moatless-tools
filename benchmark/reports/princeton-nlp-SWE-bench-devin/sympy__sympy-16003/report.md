# sympy__sympy-16003

| **sympy/sympy** | `701441853569d370506514083b995d11f9a130bd` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3347 |
| **Any found context length** | 305 |
| **Avg pos** | 8.0 |
| **Min pos** | 1 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/mathml.py b/sympy/printing/mathml.py
--- a/sympy/printing/mathml.py
+++ b/sympy/printing/mathml.py
@@ -423,10 +423,14 @@ def _print_Derivative(self, e):
         if requires_partial(e):
             diff_symbol = 'partialdiff'
         x.appendChild(self.dom.createElement(diff_symbol))
-
         x_1 = self.dom.createElement('bvar')
-        for sym in e.variables:
+
+        for sym, times in reversed(e.variable_count):
             x_1.appendChild(self._print(sym))
+            if times > 1:
+                degree = self.dom.createElement('degree')
+                degree.appendChild(self._print(sympify(times)))
+                x_1.appendChild(degree)
 
         x.appendChild(x_1)
         x.appendChild(self._print(e.expr))
@@ -839,39 +843,52 @@ def _print_Number(self, e):
         return x
 
     def _print_Derivative(self, e):
-        mrow = self.dom.createElement('mrow')
-        x = self.dom.createElement('mo')
+
         if requires_partial(e):
-            x.appendChild(self.dom.createTextNode('&#x2202;'))
-            y = self.dom.createElement('mo')
-            y.appendChild(self.dom.createTextNode('&#x2202;'))
+            d = '&#x2202;'
         else:
-            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-            y = self.dom.createElement('mo')
-            y.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-
-        brac = self.dom.createElement('mfenced')
-        brac.appendChild(self._print(e.expr))
-        mrow = self.dom.createElement('mrow')
-        mrow.appendChild(x)
-        mrow.appendChild(brac)
-
-        for sym in e.variables:
-            frac = self.dom.createElement('mfrac')
-            m = self.dom.createElement('mrow')
-            x = self.dom.createElement('mo')
-            if requires_partial(e):
-                x.appendChild(self.dom.createTextNode('&#x2202;'))
+            d = self.mathml_tag(e)
+
+        # Determine denominator
+        m = self.dom.createElement('mrow')
+        dim = 0 # Total diff dimension, for numerator
+        for sym, num in reversed(e.variable_count):
+            dim += num
+            if num >= 2:
+                x = self.dom.createElement('msup')
+                xx = self.dom.createElement('mo')
+                xx.appendChild(self.dom.createTextNode(d))
+                x.appendChild(xx)
+                x.appendChild(self._print(num))
             else:
-                x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-            y = self._print(sym)
+                x = self.dom.createElement('mo')
+                x.appendChild(self.dom.createTextNode(d))
             m.appendChild(x)
+            y = self._print(sym)
             m.appendChild(y)
-            frac.appendChild(mrow)
-            frac.appendChild(m)
-            mrow = frac
 
-        return frac
+        mnum = self.dom.createElement('mrow')
+        if dim >= 2:
+            x = self.dom.createElement('msup')
+            xx = self.dom.createElement('mo')
+            xx.appendChild(self.dom.createTextNode(d))
+            x.appendChild(xx)
+            x.appendChild(self._print(dim))
+        else:
+            x = self.dom.createElement('mo')
+            x.appendChild(self.dom.createTextNode(d))
+
+        mnum.appendChild(x)
+        mrow = self.dom.createElement('mrow')
+        frac = self.dom.createElement('mfrac')
+        frac.appendChild(mnum)
+        frac.appendChild(m)
+        mrow.appendChild(frac)
+
+        # Print function
+        mrow.appendChild(self._print(e.expr))
+
+        return mrow
 
     def _print_Function(self, e):
         mrow = self.dom.createElement('mrow')

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/mathml.py | 426 | 428 | 7 | 1 | 3347
| sympy/printing/mathml.py | 842 | 874 | 1 | 1 | 305


## Problem Statement

```
MathML presentation printing of multiple derivatives messed up
Currently, the MathML presentation printed version of the expression `Derivative(f(x, y, z), x, z, x, z, z, y)`
looks like:
![image](https://user-images.githubusercontent.com/8114497/52842849-a3d64380-3100-11e9-845f-8abacba54635.png)

while a proper rending would be more along the lines of the LaTeX equivalent:
![image](https://user-images.githubusercontent.com/8114497/52843456-78545880-3102-11e9-9d73-1d2d515a888c.png)

Hence, the `_print_Derivative` method should be improved, first and foremost to print all the derivative variables on a single line and to get the correct power in the numerator.

It is also preferred if the actual function ends up on a separate line (not sure if there is some logic to tell when this should or should not happen).

If possible, the logic to group adjacent identical terms can be applied, see the discussion and code in #15975 which gives an idea of how to implement it.

[To be closed] Added _print_derivative2 methods from #3926
<!-- Your title above should be a short description of what
was changed. Do not include the issue number in the title. -->

#### References to other Issues or PRs
<!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
format, e.g. "Fixes #1234". See
https://github.com/blog/1506-closing-issues-via-pull-requests . Please also
write a comment on that issue linking back to this pull request once it is
open. -->
Closes #3926 

#### Brief description of what is fixed or changed
As the attached diff in #3926 was pretty large due to line endings, I extracted the interesting parts, the methods `_print_derivative2` for LaTex, pretty and MathML printers.

#### Other comments
Not sure what to do with it. It looked quite promising in the original PR. Maybe one should have a switch to select between these two methods of printing?

I have not checked the code more than modifying it to work with current Python and sympy version, at least from a "no-static-warnings-in-Spyder"-perspective.

#### Release Notes

<!-- Write the release notes for this release below. See
https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more information
on how to write release notes. The bot will check your release notes
automatically to see if they are formatted correctly. -->

<!-- BEGIN RELEASE NOTES -->
NO ENTRY
<!-- END RELEASE NOTES -->

MathML presentation printing of multiple derivatives messed up
Currently, the MathML presentation printed version of the expression `Derivative(f(x, y, z), x, z, x, z, z, y)`
looks like:
![image](https://user-images.githubusercontent.com/8114497/52842849-a3d64380-3100-11e9-845f-8abacba54635.png)

while a proper rending would be more along the lines of the LaTeX equivalent:
![image](https://user-images.githubusercontent.com/8114497/52843456-78545880-3102-11e9-9d73-1d2d515a888c.png)

Hence, the `_print_Derivative` method should be improved, first and foremost to print all the derivative variables on a single line and to get the correct power in the numerator.

It is also preferred if the actual function ends up on a separate line (not sure if there is some logic to tell when this should or should not happen).

If possible, the logic to group adjacent identical terms can be applied, see the discussion and code in #15975 which gives an idea of how to implement it.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/printing/mathml.py** | 836 | 874| 305 | 305 | 7456 | 
| 2 | 2 sympy/printing/pretty/pretty.py | 325 | 360| 287 | 592 | 29488 | 
| 3 | 2 sympy/printing/pretty/pretty.py | 1100 | 1127| 212 | 804 | 29488 | 
| 4 | 3 sympy/printing/printer.py | 1 | 173| 1419 | 2223 | 31871 | 
| 5 | **3 sympy/printing/mathml.py** | 523 | 568| 336 | 2559 | 31871 | 
| 6 | **3 sympy/printing/mathml.py** | 676 | 712| 330 | 2889 | 31871 | 
| **-> 7 <-** | **3 sympy/printing/mathml.py** | 415 | 474| 458 | 3347 | 31871 | 
| 8 | 4 sympy/physics/vector/printing.py | 123 | 157| 352 | 3699 | 35219 | 
| 9 | **4 sympy/printing/mathml.py** | 570 | 588| 160 | 3859 | 35219 | 
| 10 | **4 sympy/printing/mathml.py** | 876 | 919| 361 | 4220 | 35219 | 
| 11 | 5 sympy/printing/mathematica.py | 121 | 147| 330 | 4550 | 36737 | 
| 12 | **5 sympy/printing/mathml.py** | 67 | 116| 503 | 5053 | 36737 | 
| 13 | 6 sympy/printing/latex.py | 2042 | 2085| 771 | 5824 | 61354 | 
| 14 | **6 sympy/printing/mathml.py** | 204 | 230| 232 | 6056 | 61354 | 
| 15 | 6 sympy/printing/latex.py | 2152 | 2216| 760 | 6816 | 61354 | 
| 16 | 6 sympy/printing/latex.py | 510 | 560| 595 | 7411 | 61354 | 
| 17 | 7 sympy/core/function.py | 1165 | 1262| 802 | 8213 | 87530 | 
| 18 | 7 sympy/printing/latex.py | 386 | 408| 268 | 8481 | 87530 | 
| 19 | **7 sympy/printing/mathml.py** | 169 | 202| 260 | 8741 | 87530 | 
| 20 | 7 sympy/printing/latex.py | 923 | 978| 577 | 9318 | 87530 | 
| 21 | 7 sympy/printing/latex.py | 633 | 662| 272 | 9590 | 87530 | 
| 22 | 7 sympy/core/function.py | 1263 | 1341| 684 | 10274 | 87530 | 
| 23 | 7 sympy/printing/latex.py | 1287 | 1347| 753 | 11027 | 87530 | 
| 24 | 7 sympy/printing/latex.py | 1004 | 1088| 809 | 11836 | 87530 | 
| 25 | **7 sympy/printing/mathml.py** | 714 | 743| 239 | 12075 | 87530 | 
| 26 | **7 sympy/printing/mathml.py** | 640 | 674| 302 | 12377 | 87530 | 
| 27 | 7 sympy/printing/latex.py | 2299 | 2467| 2345 | 14722 | 87530 | 
| 28 | **7 sympy/printing/mathml.py** | 768 | 802| 291 | 15013 | 87530 | 
| 29 | **7 sympy/printing/mathml.py** | 259 | 299| 345 | 15358 | 87530 | 
| 30 | **7 sympy/printing/mathml.py** | 590 | 616| 218 | 15576 | 87530 | 
| 31 | 7 sympy/core/function.py | 1342 | 1416| 615 | 16191 | 87530 | 
| 32 | **7 sympy/printing/mathml.py** | 618 | 638| 172 | 16363 | 87530 | 
| 33 | **7 sympy/printing/mathml.py** | 301 | 328| 259 | 16622 | 87530 | 
| 34 | **7 sympy/printing/mathml.py** | 1 | 65| 467 | 17089 | 87530 | 
| 35 | 7 sympy/printing/latex.py | 822 | 895| 680 | 17769 | 87530 | 
| 36 | 7 sympy/printing/latex.py | 1 | 81| 717 | 18486 | 87530 | 
| 37 | 7 sympy/printing/latex.py | 410 | 459| 383 | 18869 | 87530 | 
| 38 | **7 sympy/printing/mathml.py** | 355 | 389| 298 | 19167 | 87530 | 
| 39 | 7 sympy/printing/pretty/pretty.py | 1257 | 1339| 773 | 19940 | 87530 | 
| 40 | 7 sympy/printing/latex.py | 461 | 508| 529 | 20469 | 87530 | 
| 41 | 7 sympy/printing/latex.py | 2087 | 2132| 407 | 20876 | 87530 | 
| 42 | 7 sympy/printing/pretty/pretty.py | 136 | 193| 559 | 21435 | 87530 | 
| 43 | 7 sympy/core/function.py | 1604 | 1696| 1003 | 22438 | 87530 | 
| 44 | 7 sympy/core/function.py | 1538 | 1565| 289 | 22727 | 87530 | 
| 45 | 7 sympy/printing/latex.py | 1231 | 1285| 730 | 23457 | 87530 | 
| 46 | 7 sympy/printing/latex.py | 1140 | 1205| 672 | 24129 | 87530 | 
| 47 | **7 sympy/printing/mathml.py** | 745 | 766| 174 | 24303 | 87530 | 
| 48 | 7 sympy/physics/vector/printing.py | 379 | 419| 338 | 24641 | 87530 | 
| 49 | 7 sympy/printing/mathematica.py | 1 | 36| 442 | 25083 | 87530 | 
| 50 | 8 sympy/plotting/experimental_lambdify.py | 1 | 76| 865 | 25948 | 93396 | 
| 51 | 8 sympy/printing/latex.py | 736 | 803| 658 | 26606 | 93396 | 
| 52 | **8 sympy/printing/mathml.py** | 477 | 521| 353 | 26959 | 93396 | 
| 53 | 8 sympy/printing/latex.py | 1647 | 1712| 561 | 27520 | 93396 | 
| 54 | 8 sympy/printing/printer.py | 251 | 301| 507 | 28027 | 93396 | 
| 55 | **8 sympy/printing/mathml.py** | 330 | 353| 205 | 28232 | 93396 | 
| 56 | **8 sympy/printing/mathml.py** | 804 | 834| 284 | 28516 | 93396 | 
| 57 | 9 sympy/printing/theanocode.py | 139 | 215| 745 | 29261 | 97563 | 
| 58 | 9 sympy/printing/pretty/pretty.py | 106 | 122| 199 | 29460 | 97563 | 
| 59 | 10 sympy/printing/tensorflow.py | 88 | 100| 125 | 29585 | 99662 | 
| 60 | 10 sympy/printing/latex.py | 1892 | 1939| 457 | 30042 | 99662 | 
| 61 | 11 sympy/printing/pycode.py | 268 | 347| 719 | 30761 | 106228 | 
| 62 | 11 sympy/physics/vector/printing.py | 1 | 14| 142 | 30903 | 106228 | 
| 63 | **11 sympy/printing/mathml.py** | 232 | 257| 211 | 31114 | 106228 | 
| 64 | 11 sympy/printing/latex.py | 82 | 117| 491 | 31605 | 106228 | 
| 65 | 11 sympy/printing/latex.py | 1815 | 1877| 542 | 32147 | 106228 | 
| 66 | **11 sympy/printing/mathml.py** | 391 | 413| 222 | 32369 | 106228 | 
| 67 | 11 sympy/printing/pretty/pretty.py | 1209 | 1255| 403 | 32772 | 106228 | 
| 68 | 11 sympy/printing/pretty/pretty.py | 398 | 472| 607 | 33379 | 106228 | 
| 69 | 12 sympy/integrals/prde.py | 986 | 1040| 675 | 34054 | 122159 | 
| 70 | 12 sympy/printing/pretty/pretty.py | 727 | 752| 284 | 34338 | 122159 | 


### Hint

```



```

## Patch

```diff
diff --git a/sympy/printing/mathml.py b/sympy/printing/mathml.py
--- a/sympy/printing/mathml.py
+++ b/sympy/printing/mathml.py
@@ -423,10 +423,14 @@ def _print_Derivative(self, e):
         if requires_partial(e):
             diff_symbol = 'partialdiff'
         x.appendChild(self.dom.createElement(diff_symbol))
-
         x_1 = self.dom.createElement('bvar')
-        for sym in e.variables:
+
+        for sym, times in reversed(e.variable_count):
             x_1.appendChild(self._print(sym))
+            if times > 1:
+                degree = self.dom.createElement('degree')
+                degree.appendChild(self._print(sympify(times)))
+                x_1.appendChild(degree)
 
         x.appendChild(x_1)
         x.appendChild(self._print(e.expr))
@@ -839,39 +843,52 @@ def _print_Number(self, e):
         return x
 
     def _print_Derivative(self, e):
-        mrow = self.dom.createElement('mrow')
-        x = self.dom.createElement('mo')
+
         if requires_partial(e):
-            x.appendChild(self.dom.createTextNode('&#x2202;'))
-            y = self.dom.createElement('mo')
-            y.appendChild(self.dom.createTextNode('&#x2202;'))
+            d = '&#x2202;'
         else:
-            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-            y = self.dom.createElement('mo')
-            y.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-
-        brac = self.dom.createElement('mfenced')
-        brac.appendChild(self._print(e.expr))
-        mrow = self.dom.createElement('mrow')
-        mrow.appendChild(x)
-        mrow.appendChild(brac)
-
-        for sym in e.variables:
-            frac = self.dom.createElement('mfrac')
-            m = self.dom.createElement('mrow')
-            x = self.dom.createElement('mo')
-            if requires_partial(e):
-                x.appendChild(self.dom.createTextNode('&#x2202;'))
+            d = self.mathml_tag(e)
+
+        # Determine denominator
+        m = self.dom.createElement('mrow')
+        dim = 0 # Total diff dimension, for numerator
+        for sym, num in reversed(e.variable_count):
+            dim += num
+            if num >= 2:
+                x = self.dom.createElement('msup')
+                xx = self.dom.createElement('mo')
+                xx.appendChild(self.dom.createTextNode(d))
+                x.appendChild(xx)
+                x.appendChild(self._print(num))
             else:
-                x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
-            y = self._print(sym)
+                x = self.dom.createElement('mo')
+                x.appendChild(self.dom.createTextNode(d))
             m.appendChild(x)
+            y = self._print(sym)
             m.appendChild(y)
-            frac.appendChild(mrow)
-            frac.appendChild(m)
-            mrow = frac
 
-        return frac
+        mnum = self.dom.createElement('mrow')
+        if dim >= 2:
+            x = self.dom.createElement('msup')
+            xx = self.dom.createElement('mo')
+            xx.appendChild(self.dom.createTextNode(d))
+            x.appendChild(xx)
+            x.appendChild(self._print(dim))
+        else:
+            x = self.dom.createElement('mo')
+            x.appendChild(self.dom.createTextNode(d))
+
+        mnum.appendChild(x)
+        mrow = self.dom.createElement('mrow')
+        frac = self.dom.createElement('mfrac')
+        frac.appendChild(mnum)
+        frac.appendChild(m)
+        mrow.appendChild(frac)
+
+        # Print function
+        mrow.appendChild(self._print(e.expr))
+
+        return mrow
 
     def _print_Function(self, e):
         mrow = self.dom.createElement('mrow')

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_mathml.py b/sympy/printing/tests/test_mathml.py
--- a/sympy/printing/tests/test_mathml.py
+++ b/sympy/printing/tests/test_mathml.py
@@ -1,7 +1,7 @@
 from sympy import diff, Integral, Limit, sin, Symbol, Integer, Rational, cos, \
     tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, E, I, oo, \
     pi, GoldenRatio, EulerGamma, Sum, Eq, Ne, Ge, Lt, Float, Matrix, Basic, S, \
-    MatrixSymbol
+    MatrixSymbol, Function, Derivative
 from sympy.stats.rv import RandomSymbol
 from sympy.printing.mathml import mathml, MathMLContentPrinter, MathMLPresentationPrinter, \
     MathMLPrinter
@@ -508,22 +508,28 @@ def test_presentation_mathml_functions():
         ].childNodes[0].nodeValue == 'x'
 
     mml_2 = mpp._print(diff(sin(x), x, evaluate=False))
-    assert mml_2.nodeName == 'mfrac'
+    assert mml_2.nodeName == 'mrow'
     assert mml_2.childNodes[0].childNodes[0
-        ].childNodes[0].nodeValue == '&dd;'
-    assert mml_2.childNodes[0].childNodes[1
+        ].childNodes[0].childNodes[0].nodeValue == '&dd;'
+    assert mml_2.childNodes[1].childNodes[1
         ].nodeName == 'mfenced'
-    assert mml_2.childNodes[1].childNodes[
-        0].childNodes[0].nodeValue == '&dd;'
+    assert mml_2.childNodes[0].childNodes[1
+        ].childNodes[0].childNodes[0].nodeValue == '&dd;'
 
     mml_3 = mpp._print(diff(cos(x*y), x, evaluate=False))
-    assert mml_3.nodeName == 'mfrac'
+    assert mml_3.childNodes[0].nodeName == 'mfrac'
     assert mml_3.childNodes[0].childNodes[0
-        ].childNodes[0].nodeValue == '&#x2202;'
-    assert mml_2.childNodes[0].childNodes[1
-        ].nodeName == 'mfenced'
-    assert mml_3.childNodes[1].childNodes[
-        0].childNodes[0].nodeValue == '&#x2202;'
+        ].childNodes[0].childNodes[0].nodeValue == '&#x2202;'
+    assert mml_3.childNodes[1].childNodes[0
+        ].childNodes[0].nodeValue == 'cos'
+
+
+def test_print_derivative():
+    f = Function('f')
+    z = Symbol('z')
+    d = Derivative(f(x, y, z), x, z, x, z, z, y)
+    assert mathml(d) == r'<apply><partialdiff/><bvar><ci>y</ci><ci>z</ci><degree><cn>2</cn></degree><ci>x</ci><ci>z</ci><ci>x</ci></bvar><apply><f/><ci>x</ci><ci>y</ci><ci>z</ci></apply></apply>'
+    assert mathml(d, printer='presentation') == r'<mrow><mfrac><mrow><msup><mo>&#x2202;</mo><mn>6</mn></msup></mrow><mrow><mo>&#x2202;</mo><mi>y</mi><msup><mo>&#x2202;</mo><mn>2</mn></msup><mi>z</mi><mo>&#x2202;</mo><mi>x</mi><mo>&#x2202;</mo><mi>z</mi><mo>&#x2202;</mo><mi>x</mi></mrow></mfrac><mrow><mi>f</mi><mfenced><mi>x</mi><mi>y</mi><mi>z</mi></mfenced></mrow></mrow>'
 
 
 def test_presentation_mathml_limits():

```


## Code snippets

### 1 - sympy/printing/mathml.py:

Start line: 836, End line: 874

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_Derivative(self, e):
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mo')
        if requires_partial(e):
            x.appendChild(self.dom.createTextNode('&#x2202;'))
            y = self.dom.createElement('mo')
            y.appendChild(self.dom.createTextNode('&#x2202;'))
        else:
            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
            y = self.dom.createElement('mo')
            y.appendChild(self.dom.createTextNode(self.mathml_tag(e)))

        brac = self.dom.createElement('mfenced')
        brac.appendChild(self._print(e.expr))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(x)
        mrow.appendChild(brac)

        for sym in e.variables:
            frac = self.dom.createElement('mfrac')
            m = self.dom.createElement('mrow')
            x = self.dom.createElement('mo')
            if requires_partial(e):
                x.appendChild(self.dom.createTextNode('&#x2202;'))
            else:
                x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
            y = self._print(sym)
            m.appendChild(x)
            m.appendChild(y)
            frac.appendChild(mrow)
            frac.appendChild(m)
            mrow = frac

        return frac
```
### 2 - sympy/printing/pretty/pretty.py:

Start line: 325, End line: 360

```python
class PrettyPrinter(Printer):

    def _print_Derivative(self, deriv):
        if requires_partial(deriv) and self._use_unicode:
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
### 3 - sympy/printing/pretty/pretty.py:

Start line: 1100, End line: 1127

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
### 4 - sympy/printing/printer.py:

Start line: 1, End line: 173

```python
"""Printing subsystem driver

SymPy's printing system works the following way: Any expression can be
passed to a designated Printer who then is responsible to return an
adequate representation of that expression.

**The basic concept is the following:**
  1. Let the object print itself if it knows how.
  2. Take the best fitting method defined in the printer.
  3. As fall-back use the emptyPrinter method for the printer.

Which Method is Responsible for Printing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole printing process is started by calling ``.doprint(expr)`` on the printer
which you want to use. This method looks for an appropriate method which can
print the given expression in the given style that the printer defines.
While looking for the method, it follows these steps:

1. **Let the object print itself if it knows how.**

    The printer looks for a specific method in every object. The name of that method
    depends on the specific printer and is defined under ``Printer.printmethod``.
    For example, StrPrinter calls ``_sympystr`` and LatexPrinter calls ``_latex``.
    Look at the documentation of the printer that you want to use.
    The name of the method is specified there.

    This was the original way of doing printing in sympy. Every class had
    its own latex, mathml, str and repr methods, but it turned out that it
    is hard to produce a high quality printer, if all the methods are spread
    out that far. Therefore all printing code was combined into the different
    printers, which works great for built-in sympy objects, but not that
    good for user defined classes where it is inconvenient to patch the
    printers.

2. **Take the best fitting method defined in the printer.**

    The printer loops through expr classes (class + its bases), and tries
    to dispatch the work to ``_print_<EXPR_CLASS>``

    e.g., suppose we have the following class hierarchy::

            Basic
            |
            Atom
            |
            Number
            |
        Rational

    then, for ``expr=Rational(...)``, the Printer will try
    to call printer methods in the order as shown in the figure below::

        p._print(expr)
        |
        |-- p._print_Rational(expr)
        |
        |-- p._print_Number(expr)
        |
        |-- p._print_Atom(expr)
        |
        `-- p._print_Basic(expr)

    if ``._print_Rational`` method exists in the printer, then it is called,
    and the result is returned back. Otherwise, the printer tries to call
    ``._print_Number`` and so on.

3. **As a fall-back use the emptyPrinter method for the printer.**

    As fall-back ``self.emptyPrinter`` will be called with the expression. If
    not defined in the Printer subclass this will be the same as ``str(expr)``.

Example of Custom Printer
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _printer_example:

In the example below, we have a printer which prints the derivative of a function
in a shorter form.

.. code-block:: python

    from sympy import Symbol
    from sympy.printing.latex import LatexPrinter, print_latex
    from sympy.core.function import UndefinedFunction, Function


    class MyLatexPrinter(LatexPrinter):
        \"\"\"Print derivative of a function of symbols in a shorter form.
        \"\"\"
        def _print_Derivative(self, expr):
            function, *vars = expr.args
            if not isinstance(type(function), UndefinedFunction) or \\
               not all(isinstance(i, Symbol) for i in vars):
                return super()._print_Derivative(expr)

            # If you want the printer to work correctly for nested
            # expressions then use self._print() instead of str() or latex().
            # See the example of nested modulo below in the custom printing
            # method section.
            return "{}_{{{}}}".format(
                self._print(Symbol(function.func.__name__)),
                            ''.join(self._print(i) for i in vars))


    def print_my_latex(expr):
        \"\"\" Most of the printers define their own wrappers for print().
        These wrappers usually take printer settings. Our printer does not have
        any settings.
        \"\"\"
        print(MyLatexPrinter().doprint(expr))


    y = Symbol("y")
    x = Symbol("x")
    f = Function("f")
    expr = f(x, y).diff(x, y)

    # Print the expression using the normal latex printer and our custom
    # printer.
    print_latex(expr)
    print_my_latex(expr)

The output of the code above is::

    \\frac{\\partial^{2}}{\\partial x\\partial y}  f{\\left(x,y \\right)}
    f_{xy}

Example of Custom Printing Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, the latex printing of the modulo operator is modified.
This is done by overriding the method ``_latex`` of ``Mod``.

.. code-block:: python

    from sympy import Symbol, Mod, Integer
    from sympy.printing.latex import print_latex


    class ModOp(Mod):
        def _latex(self, printer=None):
            # Always use printer.doprint() otherwise nested expressions won't
            # work. See the example of ModOpWrong.
            a, b = [printer.doprint(i) for i in self.args]
            return r"\\operatorname{Mod}{\\left( %s,%s \\right)}" % (a,b)


    class ModOpWrong(Mod):
        def _latex(self, printer=None):
            a, b = [str(i) for i in self.args]
            return r"\\operatorname{Mod}{\\left( %s,%s \\right)}" % (a,b)


    x = Symbol('x')
    m = Symbol('m')

    print_latex(ModOp(x, m))
    print_latex(Mod(x, m))

    # Nested modulo.
    print_latex(ModOp(ModOp(x, m), Integer(7)))
    print_latex(ModOpWrong(ModOpWrong(x, m), Integer(7)))

The output of the code above is::

    \\operatorname{Mod}{\\left( x,m \\right)}
    x\\bmod{m}
    \\operatorname{Mod}{\\left( \\operatorname{Mod}{\\left( x,m \\right)},7 \\right)}
    \\operatorname{Mod}{\\left( ModOpWrong(x, m),7 \\right)}
"""

from __future__ import print_function, division
```
### 5 - sympy/printing/mathml.py:

Start line: 523, End line: 568

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Mul(self, expr):

        def multiply(expr, mrow):
            from sympy.simplify import fraction
            numer, denom = fraction(expr)

            if denom is not S.One:
                frac = self.dom.createElement('mfrac')
                xnum = self._print(numer)
                xden = self._print(denom)
                frac.appendChild(xnum)
                frac.appendChild(xden)
                return frac

            coeff, terms = expr.as_coeff_mul()
            if coeff is S.One and len(terms) == 1:
                return self._print(terms[0])

            if self.order != 'old':
                terms = Mul._from_args(terms).as_ordered_factors()

            if(coeff != 1):
                x = self._print(coeff)
                y = self.dom.createElement('mo')
                y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                mrow.appendChild(x)
                mrow.appendChild(y)
            for term in terms:
                x = self._print(term)
                mrow.appendChild(x)
                if not term == terms[-1]:
                    y = self.dom.createElement('mo')
                    y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                    mrow.appendChild(y)
            return mrow

        mrow = self.dom.createElement('mrow')
        if _coeff_isneg(expr):
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(x)
            mrow = multiply(-expr, mrow)
        else:
            mrow = multiply(expr, mrow)

        return mrow
```
### 6 - sympy/printing/mathml.py:

Start line: 676, End line: 712

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Integral(self, e):
        limits = list(e.limits)
        if len(limits[0]) == 3:
            subsup = self.dom.createElement('msubsup')
            low_elem = self._print(limits[0][1])
            up_elem = self._print(limits[0][2])
            integral = self.dom.createElement('mo')
            integral.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
            subsup.appendChild(integral)
            subsup.appendChild(low_elem)
            subsup.appendChild(up_elem)
        if len(limits[0]) == 1:
            subsup = self.dom.createElement('mrow')
            integral = self.dom.createElement('mo')
            integral.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
            subsup.appendChild(integral)

        mrow = self.dom.createElement('mrow')
        diff = self.dom.createElement('mo')
        diff.appendChild(self.dom.createTextNode('&dd;'))
        if len(str(limits[0][0])) > 1:
            var = self.dom.createElement('mfenced')
            var.appendChild(self._print(limits[0][0]))
        else:
            var = self._print(limits[0][0])

        mrow.appendChild(subsup)
        if len(str(e.function)) == 1:
            mrow.appendChild(self._print(e.function))
        else:
            fence = self.dom.createElement('mfenced')
            fence.appendChild(self._print(e.function))
            mrow.appendChild(fence)

        mrow.appendChild(diff)
        mrow.appendChild(var)
        return mrow
```
### 7 - sympy/printing/mathml.py:

Start line: 415, End line: 474

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_Derivative(self, e):
        x = self.dom.createElement('apply')
        diff_symbol = self.mathml_tag(e)
        if requires_partial(e):
            diff_symbol = 'partialdiff'
        x.appendChild(self.dom.createElement(diff_symbol))

        x_1 = self.dom.createElement('bvar')
        for sym in e.variables:
            x_1.appendChild(self._print(sym))

        x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    def _print_Function(self, e):
        x = self.dom.createElement("apply")
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Basic(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_AssocOp(self, e):
        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Relational(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        x.appendChild(self._print(e.lhs))
        x.appendChild(self._print(e.rhs))
        return x

    def _print_list(self, seq):
        """MathML reference for the <list> element:
        http://www.w3.org/TR/MathML2/chapter4.html#contm.list"""
        dom_element = self.dom.createElement('list')
        for item in seq:
            dom_element.appendChild(self._print(item))
        return dom_element

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element
```
### 8 - sympy/physics/vector/printing.py:

Start line: 123, End line: 157

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
            return LatexPrinter().doprint(der_expr)

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
            return LatexPrinter().doprint(der_expr)
        if len(base_split) is not 1:
            base += '_' + base_split[1]
        return base
```
### 9 - sympy/printing/mathml.py:

Start line: 570, End line: 588

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Add(self, expr, order=None):
        mrow = self.dom.createElement('mrow')
        args = self._as_ordered_terms(expr, order=order)
        mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            if _coeff_isneg(arg):
                # use minus
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('-'))
                y = self._print(-arg)
                # invert expression since this is now minused
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('+'))
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)

        return mrow
```
### 10 - sympy/printing/mathml.py:

Start line: 876, End line: 919

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Function(self, e):
        mrow = self.dom.createElement('mrow')
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        y = self.dom.createElement('mfenced')
        for arg in e.args:
            y.appendChild(self._print(arg))
        mrow.appendChild(x)
        mrow.appendChild(y)
        return mrow

    def _print_Basic(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        brac = self.dom.createElement('mfenced')
        for arg in e.args:
            brac.appendChild(self._print(arg))
        mrow.appendChild(brac)
        return mrow

    def _print_AssocOp(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.append(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        for arg in e.args:
            mrow.appendChild(self._print(arg))
        return mrow

    def _print_Relational(self, e):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.lhs))
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(x)
        mrow.appendChild(self._print(e.rhs))
        return mrow

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element
```
### 12 - sympy/printing/mathml.py:

Start line: 67, End line: 116

```python
class MathMLPrinterBase(Printer):

    def apply_patch(self):
        # Applying the patch of xml.dom.minidom bug
        # Date: 2011-11-18
        # Description: http://ronrothman.com/public/leftbraned/xml-dom-minidom-\
        #                   toprettyxml-and-silly-whitespace/#best-solution
        # Issue: http://bugs.python.org/issue4147
        # Patch: http://hg.python.org/cpython/rev/7262f8f276ff/

        from xml.dom.minidom import Element, Text, Node, _write_data

        def writexml(self, writer, indent="", addindent="", newl=""):
            # indent = current indentation
            # addindent = indentation to add to higher levels
            # newl = newline string
            writer.write(indent + "<" + self.tagName)

            attrs = self._get_attributes()
            a_names = list(attrs.keys())
            a_names.sort()

            for a_name in a_names:
                writer.write(" %s=\"" % a_name)
                _write_data(writer, attrs[a_name].value)
                writer.write("\"")
            if self.childNodes:
                writer.write(">")
                if (len(self.childNodes) == 1 and
                        self.childNodes[0].nodeType == Node.TEXT_NODE):
                    self.childNodes[0].writexml(writer, '', '', '')
                else:
                    writer.write(newl)
                    for node in self.childNodes:
                        node.writexml(
                            writer, indent + addindent, addindent, newl)
                    writer.write(indent)
                writer.write("</%s>%s" % (self.tagName, newl))
            else:
                writer.write("/>%s" % (newl))
        self._Element_writexml_old = Element.writexml
        Element.writexml = writexml

        def writexml(self, writer, indent="", addindent="", newl=""):
            _write_data(writer, "%s%s%s" % (indent, self.data, newl))
        self._Text_writexml_old = Text.writexml
        Text.writexml = writexml

    def restore_patch(self):
        from xml.dom.minidom import Element, Text
        Element.writexml = self._Element_writexml_old
        Text.writexml = self._Text_writexml_old
```
### 14 - sympy/printing/mathml.py:

Start line: 204, End line: 230

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        lastProcessed = self._print(args[0])
        plusNodes = []
        for arg in args[1:]:
            if _coeff_isneg(arg):
                # use minus
                x = self.dom.createElement('apply')
                x.appendChild(self.dom.createElement('minus'))
                x.appendChild(lastProcessed)
                x.appendChild(self._print(-arg))
                # invert expression since this is now minused
                lastProcessed = x
                if(arg == args[-1]):
                    plusNodes.append(lastProcessed)
            else:
                plusNodes.append(lastProcessed)
                lastProcessed = self._print(arg)
                if(arg == args[-1]):
                    plusNodes.append(self._print(arg))
        if len(plusNodes) == 1:
            return lastProcessed
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('plus'))
        while len(plusNodes) > 0:
            x.appendChild(plusNodes.pop(0))
        return x
```
### 19 - sympy/printing/mathml.py:

Start line: 169, End line: 202

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Mul(self, expr):

        if _coeff_isneg(expr):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('minus'))
            x.appendChild(self._print_Mul(-expr))
            return x

        from sympy.simplify import fraction
        numer, denom = fraction(expr)

        if denom is not S.One:
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('divide'))
            x.appendChild(self._print(numer))
            x.appendChild(self._print(denom))
            return x

        coeff, terms = expr.as_coeff_mul()
        if coeff is S.One and len(terms) == 1:
            # XXX since the negative coefficient has been handled, I don't
            # think a coeff of 1 can remain
            return self._print(terms[0])

        if self.order != 'old':
            terms = Mul._from_args(terms).as_ordered_factors()

        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('times'))
        if(coeff != 1):
            x.appendChild(self._print(coeff))
        for term in terms:
            x.appendChild(self._print(term))
        return x
```
### 25 - sympy/printing/mathml.py:

Start line: 714, End line: 743

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Sum(self, e):
        limits = list(e.limits)
        subsup = self.dom.createElement('munderover')
        low_elem = self._print(limits[0][1])
        up_elem = self._print(limits[0][2])
        summand = self.dom.createElement('mo')
        summand.appendChild(self.dom.createTextNode(self.mathml_tag(e)))

        low = self.dom.createElement('mrow')
        var = self._print(limits[0][0])
        equal = self.dom.createElement('mo')
        equal.appendChild(self.dom.createTextNode('='))
        low.appendChild(var)
        low.appendChild(equal)
        low.appendChild(low_elem)

        subsup.appendChild(summand)
        subsup.appendChild(low)
        subsup.appendChild(up_elem)

        mrow = self.dom.createElement('mrow')
        mrow.appendChild(subsup)
        if len(str(e.function)) == 1:
            mrow.appendChild(self._print(e.function))
        else:
            fence = self.dom.createElement('mfenced')
            fence.appendChild(self._print(e.function))
            mrow.appendChild(fence)

        return mrow
```
### 26 - sympy/printing/mathml.py:

Start line: 640, End line: 674

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
### 28 - sympy/printing/mathml.py:

Start line: 768, End line: 802

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Symbol(self, sym, style='plain'):

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mi')
        mname.appendChild(self.dom.createTextNode(name))
        if len(supers) == 0:
            if len(subs) == 0:
                x.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                x.appendChild(msub)
        else:
            if len(subs) == 0:
                msup = self.dom.createElement('msup')
                msup.appendChild(mname)
                msup.appendChild(join(supers))
                x.appendChild(msup)
            else:
                msubsup = self.dom.createElement('msubsup')
                msubsup.appendChild(mname)
                msubsup.appendChild(join(subs))
                msubsup.appendChild(join(supers))
                x.appendChild(msubsup)
        return x
```
### 29 - sympy/printing/mathml.py:

Start line: 259, End line: 299

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Limit(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))

        x_1 = self.dom.createElement('bvar')
        x_2 = self.dom.createElement('lowlimit')
        x_1.appendChild(self._print(e.args[1]))
        x_2.appendChild(self._print(e.args[2]))

        x.appendChild(x_1)
        x.appendChild(x_2)
        x.appendChild(self._print(e.args[0]))
        return x

    def _print_ImaginaryUnit(self, e):
        return self.dom.createElement('imaginaryi')

    def _print_EulerGamma(self, e):
        return self.dom.createElement('eulergamma')

    def _print_GoldenRatio(self, e):
        """We use unicode #x3c6 for Greek letter phi as defined here
        http://www.w3.org/2003/entities/2007doc/isogrk1.html"""
        x = self.dom.createElement('cn')
        x.appendChild(self.dom.createTextNode(u"\N{GREEK SMALL LETTER PHI}"))
        return x

    def _print_Exp1(self, e):
        return self.dom.createElement('exponentiale')

    def _print_Pi(self, e):
        return self.dom.createElement('pi')

    def _print_Infinity(self, e):
        return self.dom.createElement('infinity')

    def _print_Negative_Infinity(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('minus'))
        x.appendChild(self.dom.createElement('infinity'))
        return x
```
### 30 - sympy/printing/mathml.py:

Start line: 590, End line: 616

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_MatrixBase(self, m):
        brac = self.dom.createElement('mfenced')
        table = self.dom.createElement('mtable')
        for i in range(m.rows):
            x = self.dom.createElement('mtr')
            for j in range(m.cols):
                y = self.dom.createElement('mtd')
                y.appendChild(self._print(m[i, j]))
                x.appendChild(y)
            table.appendChild(x)
        brac.appendChild(table)
        return brac

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            x = self.dom.createElement('mn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        x = self.dom.createElement('mfrac')
        num = self.dom.createElement('mn')
        num.appendChild(self.dom.createTextNode(str(e.p)))
        x.appendChild(num)
        den = self.dom.createElement('mn')
        den.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(den)
        return x
```
### 32 - sympy/printing/mathml.py:

Start line: 618, End line: 638

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Limit(self, e):
        mrow = self.dom.createElement('mrow')
        munder = self.dom.createElement('munder')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('lim'))

        x = self.dom.createElement('mrow')
        x_1 = self._print(e.args[1])
        arrow = self.dom.createElement('mo')
        arrow.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        x_2 = self._print(e.args[2])
        x.appendChild(x_1)
        x.appendChild(arrow)
        x.appendChild(x_2)

        munder.appendChild(mi)
        munder.appendChild(x)
        mrow.appendChild(munder)
        mrow.appendChild(self._print(e.args[0]))

        return mrow
```
### 33 - sympy/printing/mathml.py:

Start line: 301, End line: 328

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Integral(self, e):
        def lime_recur(limits):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement(self.mathml_tag(e)))
            bvar_elem = self.dom.createElement('bvar')
            bvar_elem.appendChild(self._print(limits[0][0]))
            x.appendChild(bvar_elem)

            if len(limits[0]) == 3:
                low_elem = self.dom.createElement('lowlimit')
                low_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(low_elem)
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][2]))
                x.appendChild(up_elem)
            if len(limits[0]) == 2:
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(up_elem)
            if len(limits) == 1:
                x.appendChild(self._print(e.function))
            else:
                x.appendChild(lime_recur(limits[1:]))
            return x

        limits = list(e.limits)
        limits.reverse()
        return lime_recur(limits)
```
### 34 - sympy/printing/mathml.py:

Start line: 1, End line: 65

```python
"""
A MathML printer.
"""

from __future__ import print_function, division

from sympy import sympify, S, Mul
from sympy.core.function import _coeff_isneg
from sympy.core.compatibility import range
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer


class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """

    _default_settings = {
        "order": None,
        "encoding": "utf-8",
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_symbol_style": "plain",
        "mul_symbol": None,
        "root_notation": True,
        "symbol_names": {},
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        from xml.dom.minidom import Document,Text

        self.dom = Document()

        # Workaround to allow strings to remain unescaped
        # Based on https://stackoverflow.com/questions/38015864/python-xml-dom-minidom-please-dont-escape-my-strings/38041194
        class RawText(Text):
            def writexml(self, writer, indent='', addindent='', newl=''):
                if self.data:
                    writer.write(u'{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r

        self.dom.createTextNode = createRawTextNode

    def doprint(self, expr):
        """
        Prints the expression as MathML.
        """
        mathML = Printer._print(self, expr)
        unistr = mathML.toxml()
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')
        res = xmlbstr.decode()
        return res
```
### 38 - sympy/printing/mathml.py:

Start line: 355, End line: 389

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Symbol(self, sym):

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mml:mi')
        mname.appendChild(self.dom.createTextNode(name))
        if len(supers) == 0:
            if len(subs) == 0:
                ci.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('mml:msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                ci.appendChild(msub)
        else:
            if len(subs) == 0:
                msup = self.dom.createElement('mml:msup')
                msup.appendChild(mname)
                msup.appendChild(join(supers))
                ci.appendChild(msup)
            else:
                msubsup = self.dom.createElement('mml:msubsup')
                msubsup.appendChild(mname)
                msubsup.appendChild(join(subs))
                msubsup.appendChild(join(supers))
                ci.appendChild(msubsup)
        return ci
```
### 47 - sympy/printing/mathml.py:

Start line: 745, End line: 766

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_Symbol(self, sym, style='plain'):
        x = self.dom.createElement('mi')

        if style == 'bold':
            x.setAttribute('mathvariant', 'bold')

        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi
        # ... other code
```
### 52 - sympy/printing/mathml.py:

Start line: 477, End line: 521

```python
class MathMLPresentationPrinter(MathMLPrinterBase):
    """Prints an expression to the Presentation MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter3.html
    """
    printmethod = "_mathml_presentation"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        translate = {
            'Mul': '&InvisibleTimes;',
            'Number': 'mn',
            'Limit' : '&#x2192;',
            'Derivative': '&dd;',
            'int': 'mn',
            'Symbol': 'mi',
            'Integral': '&int;',
            'Sum': '&#x2211;',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'acot': 'arccot',
            'atan2': 'arctan',
            'Equality': '=',
            'Unequality': '&#x2260;',
            'GreaterThan': '&#x2265;',
            'LessThan': '&#x2264;',
            'StrictGreaterThan': '>',
            'StrictLessThan': '<',
        }

        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        # Not found in the MRO set
        n = e.__class__.__name__
        return n.lower()
```
### 55 - sympy/printing/mathml.py:

Start line: 330, End line: 353

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_Sum(self, e):
        # Printer can be shared because Sum and Integral have the
        # same internal representation.
        return self._print_Integral(e)

    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))

        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mml:mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mml:mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mml:mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mml:mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi
        # ... other code
```
### 56 - sympy/printing/mathml.py:

Start line: 804, End line: 834

```python
class MathMLPresentationPrinter(MathMLPrinterBase):

    def _print_MatrixSymbol(self, sym):
        return self._print_Symbol(sym, style=self._settings['mat_symbol_style'])

    _print_RandomSymbol = _print_Symbol

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the reciprocal of an integer
        if e.exp.is_negative or len(str(e.base)) > 1:
            mrow = self.dom.createElement('mrow')
            x = self.dom.createElement('mfenced')
            x.appendChild(self._print(e.base))
            mrow.appendChild(x)
            x = self.dom.createElement('msup')
            x.appendChild(mrow)
            x.appendChild(self._print(e.exp))
            return x

        if e.exp.is_Rational and e.exp.p == 1 and self._settings['root_notation']:
            if e.exp.q == 2:
                x = self.dom.createElement('msqrt')
                x.appendChild(self._print(e.base))
            if e.exp.q != 2:
                x = self.dom.createElement('mroot')
                x.appendChild(self._print(e.base))
                x.appendChild(self._print(e.exp.q))
            return x

        x = self.dom.createElement('msup')
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x
```
### 63 - sympy/printing/mathml.py:

Start line: 232, End line: 257

```python
class MathMLContentPrinter(MathMLPrinterBase):

    def _print_MatrixBase(self, m):
        x = self.dom.createElement('matrix')
        for i in range(m.rows):
            x_r = self.dom.createElement('matrixrow')
            for j in range(m.cols):
                x_r.appendChild(self._print(m[i, j]))
            x.appendChild(x_r)
        return x

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            x = self.dom.createElement('cn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('divide'))
        # numerator
        xnum = self.dom.createElement('cn')
        xnum.appendChild(self.dom.createTextNode(str(e.p)))
        # denominator
        xdenom = self.dom.createElement('cn')
        xdenom.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(xnum)
        x.appendChild(xdenom)
        return x
```
### 66 - sympy/printing/mathml.py:

Start line: 391, End line: 413

```python
class MathMLContentPrinter(MathMLPrinterBase):

    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the reciprocal of an integer
        if self._settings['root_notation'] and e.exp.is_Rational and e.exp.p == 1:
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('root'))
            if e.exp.q != 2:
                xmldeg = self.dom.createElement('degree')
                xmlci = self.dom.createElement('ci')
                xmlci.appendChild(self.dom.createTextNode(str(e.exp.q)))
                xmldeg.appendChild(xmlci)
                x.appendChild(xmldeg)
            x.appendChild(self._print(e.base))
            return x

        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x
```
