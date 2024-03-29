# sympy__sympy-11400

| **sympy/sympy** | `8dcb12a6cf500e8738d6729ab954a261758f49ca` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3148 |
| **Any found context length** | 3148 |
| **Avg pos** | 8.0 |
| **Min pos** | 8 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -231,6 +231,20 @@ def _print_Symbol(self, expr):
         else:
             return name
 
+    def _print_Relational(self, expr):
+        lhs_code = self._print(expr.lhs)
+        rhs_code = self._print(expr.rhs)
+        op = expr.rel_op
+        return ("{0} {1} {2}").format(lhs_code, op, rhs_code)
+
+    def _print_sinc(self, expr):
+        from sympy.functions.elementary.trigonometric import sin
+        from sympy.core.relational import Ne
+        from sympy.functions import Piecewise
+        _piecewise = Piecewise(
+            (sin(expr.args[0]) / expr.args[0], Ne(expr.args[0], 0)), (1, True))
+        return self._print(_piecewise)
+
     def _print_AugmentedAssignment(self, expr):
         lhs_code = self._print(expr.lhs)
         op = expr.rel_op

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/ccode.py | 234 | 234 | 8 | 1 | 3148


## Problem Statement

```
ccode(sinc(x)) doesn't work
\`\`\`
In [30]: ccode(sinc(x))
Out[30]: '// Not supported in C:\n// sinc\nsinc(x)'
\`\`\`

I don't think `math.h` has `sinc`, but it could print

\`\`\`
In [38]: ccode(Piecewise((sin(theta)/theta, Ne(theta, 0)), (1, True)))
Out[38]: '((Ne(theta, 0)) ? (\n   sin(theta)/theta\n)\n: (\n   1\n))'
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/ccode.py** | 1 | 83| 512 | 512 | 3438 | 
| 2 | **1 sympy/printing/ccode.py** | 283 | 402| 1237 | 1749 | 3438 | 
| 3 | **1 sympy/printing/ccode.py** | 155 | 182| 218 | 1967 | 3438 | 
| 4 | **1 sympy/printing/ccode.py** | 184 | 214| 358 | 2325 | 3438 | 
| 5 | 2 sympy/functions/elementary/trigonometric.py | 1551 | 1603| 362 | 2687 | 25712 | 
| 6 | 3 sympy/functions/elementary/hyperbolic.py | 484 | 501| 126 | 2813 | 35911 | 
| 7 | **3 sympy/printing/ccode.py** | 143 | 153| 124 | 2937 | 35911 | 
| **-> 8 <-** | **3 sympy/printing/ccode.py** | 216 | 238| 211 | 3148 | 35911 | 
| 9 | 4 sympy/printing/mathematica.py | 1 | 34| 394 | 3542 | 37038 | 
| 10 | 4 sympy/functions/elementary/trigonometric.py | 731 | 780| 600 | 4142 | 37038 | 
| 11 | **4 sympy/printing/ccode.py** | 240 | 252| 163 | 4305 | 37038 | 
| 12 | 4 sympy/functions/elementary/hyperbolic.py | 259 | 274| 130 | 4435 | 37038 | 
| 13 | 4 sympy/functions/elementary/hyperbolic.py | 200 | 257| 362 | 4797 | 37038 | 
| 14 | 4 sympy/functions/elementary/hyperbolic.py | 566 | 579| 157 | 4954 | 37038 | 
| 15 | 4 sympy/functions/elementary/trigonometric.py | 1155 | 1251| 804 | 5758 | 37038 | 
| 16 | 4 sympy/functions/elementary/hyperbolic.py | 295 | 310| 173 | 5931 | 37038 | 
| 17 | 4 sympy/functions/elementary/trigonometric.py | 1269 | 1333| 590 | 6521 | 37038 | 
| 18 | 4 sympy/functions/elementary/hyperbolic.py | 312 | 344| 293 | 6814 | 37038 | 
| 19 | 4 sympy/functions/elementary/trigonometric.py | 1106 | 1153| 299 | 7113 | 37038 | 
| 20 | 4 sympy/functions/elementary/trigonometric.py | 2438 | 2521| 628 | 7741 | 37038 | 
| 21 | 4 sympy/functions/elementary/hyperbolic.py | 276 | 293| 181 | 7922 | 37038 | 
| 22 | 4 sympy/functions/elementary/hyperbolic.py | 503 | 545| 295 | 8217 | 37038 | 
| 23 | 4 sympy/functions/elementary/hyperbolic.py | 148 | 163| 171 | 8388 | 37038 | 
| 24 | 4 sympy/functions/elementary/trigonometric.py | 1335 | 1375| 369 | 8757 | 37038 | 
| 25 | 4 sympy/printing/mathematica.py | 113 | 125| 106 | 8863 | 37038 | 
| 26 | 4 sympy/functions/elementary/trigonometric.py | 1620 | 1665| 253 | 9116 | 37038 | 
| 27 | 4 sympy/functions/elementary/trigonometric.py | 1253 | 1267| 134 | 9250 | 37038 | 
| 28 | 5 sympy/functions/special/error_functions.py | 1995 | 2011| 166 | 9416 | 57071 | 
| 29 | 6 sympy/printing/jscode.py | 91 | 128| 313 | 9729 | 59749 | 
| 30 | 6 sympy/functions/special/error_functions.py | 1901 | 1993| 685 | 10414 | 59749 | 
| 31 | **6 sympy/printing/ccode.py** | 130 | 141| 134 | 10548 | 59749 | 
| 32 | 7 sympy/printing/fcode.py | 241 | 255| 160 | 10708 | 64454 | 
| 33 | 8 sympy/integrals/manualintegrate.py | 577 | 645| 731 | 11439 | 76088 | 
| 34 | 8 sympy/functions/elementary/hyperbolic.py | 165 | 197| 289 | 11728 | 76088 | 
| 35 | 8 sympy/printing/fcode.py | 418 | 538| 1286 | 13014 | 76088 | 
| 36 | 8 sympy/functions/elementary/hyperbolic.py | 581 | 605| 255 | 13269 | 76088 | 
| 37 | 8 sympy/printing/fcode.py | 173 | 213| 338 | 13607 | 76088 | 
| 38 | 8 sympy/functions/special/error_functions.py | 1709 | 1809| 770 | 14377 | 76088 | 
| 39 | 8 sympy/functions/elementary/hyperbolic.py | 63 | 105| 284 | 14661 | 76088 | 
| 40 | 8 sympy/functions/elementary/hyperbolic.py | 862 | 928| 634 | 15295 | 76088 | 
| 41 | 8 sympy/functions/elementary/hyperbolic.py | 547 | 564| 146 | 15441 | 76088 | 
| 42 | 9 sympy/printing/octave.py | 1 | 49| 536 | 15977 | 82089 | 
| 43 | 9 sympy/printing/fcode.py | 131 | 171| 422 | 16399 | 82089 | 
| 44 | 9 sympy/printing/jscode.py | 130 | 160| 358 | 16757 | 82089 | 
| 45 | 9 sympy/functions/elementary/hyperbolic.py | 824 | 838| 155 | 16912 | 82089 | 
| 46 | 9 sympy/integrals/manualintegrate.py | 646 | 671| 445 | 17357 | 82089 | 
| 47 | 9 sympy/functions/elementary/hyperbolic.py | 347 | 369| 141 | 17498 | 82089 | 
| 48 | 9 sympy/functions/elementary/trigonometric.py | 322 | 376| 496 | 17994 | 82089 | 
| 49 | 9 sympy/functions/elementary/trigonometric.py | 1 | 18| 202 | 18196 | 82089 | 
| 50 | 9 sympy/printing/fcode.py | 1 | 49| 361 | 18557 | 82089 | 
| 51 | 10 sympy/integrals/trigonometry.py | 277 | 317| 381 | 18938 | 85163 | 
| 52 | 10 sympy/functions/elementary/trigonometric.py | 1605 | 1617| 149 | 19087 | 85163 | 
| 53 | 10 sympy/functions/elementary/trigonometric.py | 601 | 642| 381 | 19468 | 85163 | 
| 54 | 10 sympy/functions/elementary/trigonometric.py | 1004 | 1025| 255 | 19723 | 85163 | 
| 55 | 10 sympy/functions/elementary/hyperbolic.py | 730 | 766| 295 | 20018 | 85163 | 
| 56 | 10 sympy/functions/elementary/trigonometric.py | 2290 | 2296| 136 | 20154 | 85163 | 
| 57 | 10 sympy/functions/elementary/trigonometric.py | 782 | 829| 414 | 20568 | 85163 | 
| 58 | 11 sympy/plotting/intervalmath/lib_interval.py | 223 | 251| 241 | 20809 | 88814 | 
| 59 | 11 sympy/printing/octave.py | 372 | 438| 585 | 21394 | 88814 | 
| 60 | 11 sympy/integrals/trigonometry.py | 1 | 30| 261 | 21655 | 88814 | 
| 61 | **11 sympy/printing/ccode.py** | 86 | 128| 319 | 21974 | 88814 | 
| 62 | 11 sympy/functions/elementary/hyperbolic.py | 930 | 961| 276 | 22250 | 88814 | 
| 63 | 11 sympy/plotting/intervalmath/lib_interval.py | 352 | 387| 283 | 22533 | 88814 | 
| 64 | 11 sympy/functions/elementary/hyperbolic.py | 434 | 447| 156 | 22689 | 88814 | 
| 65 | 11 sympy/printing/octave.py | 270 | 301| 175 | 22864 | 88814 | 
| 66 | 12 sympy/simplify/trigsimp.py | 736 | 776| 748 | 23612 | 100876 | 
| 67 | 12 sympy/functions/elementary/trigonometric.py | 502 | 599| 926 | 24538 | 100876 | 
| 68 | 13 sympy/functions/special/spherical_harmonics.py | 1 | 14| 125 | 24663 | 104048 | 
| 69 | 13 sympy/functions/elementary/trigonometric.py | 2267 | 2288| 182 | 24845 | 104048 | 
| 70 | 13 sympy/functions/special/error_functions.py | 2336 | 2358| 345 | 25190 | 104048 | 
| 71 | 13 sympy/printing/jscode.py | 1 | 35| 234 | 25424 | 104048 | 
| 72 | 13 sympy/functions/elementary/hyperbolic.py | 1039 | 1082| 290 | 25714 | 104048 | 
| 73 | 13 sympy/functions/elementary/trigonometric.py | 2184 | 2219| 251 | 25965 | 104048 | 
| 74 | 13 sympy/printing/fcode.py | 215 | 239| 219 | 26184 | 104048 | 
| 75 | 13 sympy/integrals/trigonometry.py | 232 | 274| 402 | 26586 | 104048 | 
| 76 | 13 sympy/functions/elementary/trigonometric.py | 1667 | 1711| 292 | 26878 | 104048 | 
| 77 | 13 sympy/functions/elementary/trigonometric.py | 1964 | 1980| 170 | 27048 | 104048 | 
| 78 | 13 sympy/functions/elementary/trigonometric.py | 1818 | 1832| 154 | 27202 | 104048 | 
| 79 | 13 sympy/functions/elementary/trigonometric.py | 225 | 320| 844 | 28046 | 104048 | 
| 80 | 13 sympy/functions/elementary/trigonometric.py | 881 | 986| 906 | 28952 | 104048 | 
| 81 | 13 sympy/functions/elementary/hyperbolic.py | 679 | 701| 162 | 29114 | 104048 | 
| 82 | 13 sympy/printing/fcode.py | 257 | 273| 179 | 29293 | 104048 | 
| 83 | 13 sympy/functions/elementary/hyperbolic.py | 449 | 481| 294 | 29587 | 104048 | 
| 84 | 13 sympy/functions/elementary/trigonometric.py | 644 | 660| 226 | 29813 | 104048 | 
| 85 | 14 sympy/functions/special/bessel.py | 1 | 16| 170 | 29983 | 119223 | 
| 86 | 14 sympy/printing/octave.py | 441 | 475| 412 | 30395 | 119223 | 
| 87 | 14 sympy/printing/octave.py | 511 | 655| 1628 | 32023 | 119223 | 
| 88 | 14 sympy/functions/elementary/hyperbolic.py | 1238 | 1276| 305 | 32328 | 119223 | 
| 89 | 14 sympy/functions/elementary/hyperbolic.py | 415 | 432| 140 | 32468 | 119223 | 
| 90 | 15 sympy/parsing/mathematica.py | 1 | 68| 613 | 33081 | 119837 | 
| 91 | 15 sympy/functions/elementary/trigonometric.py | 2154 | 2160| 132 | 33213 | 119837 | 
| 92 | 15 sympy/functions/elementary/hyperbolic.py | 1084 | 1108| 185 | 33398 | 119837 | 
| 93 | 16 sympy/printing/theanocode.py | 1 | 64| 563 | 33961 | 121912 | 
| 94 | 16 sympy/functions/elementary/trigonometric.py | 662 | 691| 259 | 34220 | 121912 | 
| 95 | 16 sympy/functions/elementary/trigonometric.py | 2298 | 2322| 272 | 34492 | 121912 | 
| 96 | 16 sympy/functions/special/error_functions.py | 1609 | 1707| 668 | 35160 | 121912 | 
| 97 | 16 sympy/functions/elementary/trigonometric.py | 2129 | 2152| 214 | 35374 | 121912 | 
| 98 | 16 sympy/functions/elementary/trigonometric.py | 1930 | 1962| 278 | 35652 | 121912 | 
| 99 | 16 sympy/integrals/trigonometry.py | 122 | 229| 1173 | 36825 | 121912 | 
| 100 | 16 sympy/printing/jscode.py | 162 | 192| 245 | 37070 | 121912 | 
| 101 | 16 sympy/functions/elementary/trigonometric.py | 1056 | 1103| 364 | 37434 | 121912 | 
| 102 | 16 sympy/functions/elementary/trigonometric.py | 172 | 223| 357 | 37791 | 121912 | 
| 103 | 16 sympy/functions/elementary/trigonometric.py | 378 | 423| 438 | 38229 | 121912 | 
| 104 | 16 sympy/plotting/intervalmath/lib_interval.py | 254 | 285| 291 | 38520 | 121912 | 
| 105 | 17 sympy/printing/codeprinter.py | 419 | 456| 411 | 38931 | 125715 | 
| 106 | 17 sympy/functions/special/spherical_harmonics.py | 241 | 260| 164 | 39095 | 125715 | 
| 107 | 18 sympy/functions/special/hyper.py | 992 | 1011| 196 | 39291 | 135402 | 
| 108 | 19 examples/advanced/curvilinear_coordinates.py | 77 | 117| 327 | 39618 | 136348 | 
| 109 | 19 sympy/functions/special/error_functions.py | 2229 | 2311| 632 | 40250 | 136348 | 
| 110 | 19 sympy/functions/elementary/trigonometric.py | 475 | 501| 241 | 40491 | 136348 | 
| 111 | **19 sympy/printing/ccode.py** | 254 | 280| 209 | 40700 | 136348 | 
| 112 | 19 sympy/functions/elementary/hyperbolic.py | 1156 | 1208| 648 | 41348 | 136348 | 
| 113 | 19 sympy/functions/special/error_functions.py | 1812 | 1898| 566 | 41914 | 136348 | 
| 114 | 19 sympy/functions/elementary/trigonometric.py | 1537 | 1548| 135 | 42049 | 136348 | 
| 115 | 19 sympy/functions/elementary/hyperbolic.py | 371 | 413| 294 | 42343 | 136348 | 
| 116 | 19 sympy/functions/elementary/trigonometric.py | 2221 | 2265| 395 | 42738 | 136348 | 
| 117 | 19 sympy/printing/octave.py | 190 | 207| 199 | 42937 | 136348 | 
| 118 | 19 sympy/functions/elementary/hyperbolic.py | 107 | 125| 144 | 43081 | 136348 | 
| 119 | 20 sympy/ntheory/partitions_.py | 123 | 138| 202 | 43283 | 138460 | 
| 120 | 20 sympy/simplify/trigsimp.py | 777 | 817| 729 | 44012 | 138460 | 
| 121 | 20 sympy/functions/elementary/trigonometric.py | 2524 | 2617| 873 | 44885 | 138460 | 
| 122 | 20 sympy/functions/elementary/trigonometric.py | 988 | 1002| 131 | 45016 | 138460 | 
| 123 | 20 sympy/functions/elementary/hyperbolic.py | 840 | 859| 130 | 45146 | 138460 | 
| 124 | 20 sympy/functions/elementary/trigonometric.py | 1770 | 1816| 449 | 45595 | 138460 | 
| 125 | 20 sympy/functions/elementary/hyperbolic.py | 36 | 61| 149 | 45744 | 138460 | 
| 126 | 20 sympy/functions/elementary/trigonometric.py | 693 | 729| 767 | 46511 | 138460 | 
| 127 | 20 sympy/functions/special/spherical_harmonics.py | 187 | 203| 203 | 46714 | 138460 | 
| 128 | 21 sympy/simplify/fu.py | 229 | 250| 152 | 46866 | 156748 | 
| 129 | 21 sympy/functions/elementary/trigonometric.py | 1483 | 1535| 361 | 47227 | 156748 | 
| 130 | 22 sympy/printing/julia.py | 204 | 254| 354 | 47581 | 162431 | 
| 131 | 22 sympy/printing/octave.py | 210 | 239| 237 | 47818 | 162431 | 
| 132 | 22 sympy/functions/special/error_functions.py | 2313 | 2324| 147 | 47965 | 162431 | 
| 133 | 23 sympy/functions/special/mathieu_functions.py | 83 | 137| 425 | 48390 | 164344 | 
| 134 | 23 sympy/printing/julia.py | 1 | 43| 492 | 48882 | 164344 | 
| 135 | 24 sympy/holonomic/holonomic.py | 1817 | 1839| 229 | 49111 | 183094 | 
| 136 | 24 sympy/functions/elementary/hyperbolic.py | 127 | 146| 194 | 49305 | 183094 | 
| 137 | 24 sympy/plotting/intervalmath/lib_interval.py | 92 | 118| 263 | 49568 | 183094 | 
| 138 | 24 sympy/functions/special/error_functions.py | 1277 | 1290| 170 | 49738 | 183094 | 
| 139 | 25 sympy/functions/__init__.py | 1 | 50| 801 | 50539 | 183896 | 
| 140 | 25 sympy/functions/special/error_functions.py | 1594 | 1606| 199 | 50738 | 183896 | 
| 141 | 25 sympy/functions/elementary/hyperbolic.py | 1278 | 1330| 526 | 51264 | 183896 | 
| 142 | 25 sympy/printing/mathematica.py | 37 | 110| 627 | 51891 | 183896 | 
| 143 | 25 sympy/functions/elementary/trigonometric.py | 2086 | 2127| 376 | 52267 | 183896 | 
| 144 | 26 examples/beginner/series.py | 1 | 29| 128 | 52395 | 184024 | 
| 145 | 26 sympy/functions/elementary/hyperbolic.py | 770 | 822| 325 | 52720 | 184024 | 
| 146 | 26 sympy/functions/elementary/hyperbolic.py | 1 | 33| 187 | 52907 | 184024 | 
| 147 | 26 sympy/functions/elementary/trigonometric.py | 832 | 879| 295 | 53202 | 184024 | 
| 148 | 26 sympy/printing/octave.py | 124 | 187| 588 | 53790 | 184024 | 


### Hint

```
@asmeurer I would like to fix this issue. Should I work upon  the codegen.py file ? If there's something else tell me how to start ?

The relevant file is sympy/printing/ccode.py

@asmeurer I am new here. I would like to work on this issue. Please tell me how to start?

Since there are two people asking, maybe one person can try #11286 which is very similar, maybe even easier.

```

## Patch

```diff
diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -231,6 +231,20 @@ def _print_Symbol(self, expr):
         else:
             return name
 
+    def _print_Relational(self, expr):
+        lhs_code = self._print(expr.lhs)
+        rhs_code = self._print(expr.rhs)
+        op = expr.rel_op
+        return ("{0} {1} {2}").format(lhs_code, op, rhs_code)
+
+    def _print_sinc(self, expr):
+        from sympy.functions.elementary.trigonometric import sin
+        from sympy.core.relational import Ne
+        from sympy.functions import Piecewise
+        _piecewise = Piecewise(
+            (sin(expr.args[0]) / expr.args[0], Ne(expr.args[0], 0)), (1, True))
+        return self._print(_piecewise)
+
     def _print_AugmentedAssignment(self, expr):
         lhs_code = self._print(expr.lhs)
         op = expr.rel_op

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_ccode.py b/sympy/printing/tests/test_ccode.py
--- a/sympy/printing/tests/test_ccode.py
+++ b/sympy/printing/tests/test_ccode.py
@@ -120,6 +120,16 @@ def test_ccode_boolean():
     assert ccode((x | y) & z) == "z && (x || y)"
 
 
+def test_ccode_Relational():
+    from sympy import Eq, Ne, Le, Lt, Gt, Ge
+    assert ccode(Eq(x, y)) == "x == y"
+    assert ccode(Ne(x, y)) == "x != y"
+    assert ccode(Le(x, y)) == "x <= y"
+    assert ccode(Lt(x, y)) == "x < y"
+    assert ccode(Gt(x, y)) == "x > y"
+    assert ccode(Ge(x, y)) == "x >= y"
+
+
 def test_ccode_Piecewise():
     expr = Piecewise((x, x < 1), (x**2, True))
     assert ccode(expr) == (
@@ -162,6 +172,18 @@ def test_ccode_Piecewise():
     raises(ValueError, lambda: ccode(expr))
 
 
+def test_ccode_sinc():
+    from sympy import sinc
+    expr = sinc(x)
+    assert ccode(expr) == (
+            "((x != 0) ? (\n"
+            "   sin(x)/x\n"
+            ")\n"
+            ": (\n"
+            "   1\n"
+            "))")
+
+
 def test_ccode_Piecewise_deep():
     p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
     assert p == (

```


## Code snippets

### 1 - sympy/printing/ccode.py:

Start line: 1, End line: 83

```python
"""
C code printer

The CCodePrinter converts single sympy expressions into single C expressions,
using the functions defined in math.h where possible.

A complete code generator, which uses ccode extensively, can be found in
sympy.utilities.codegen. The codegen module can be used to generate complete
source code files that are compilable without further modifications.


"""

from __future__ import print_function, division

from sympy.core import S
from sympy.core.compatibility import string_types, range
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from sympy.sets.fancysets import Range

# dictionary mapping sympy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_Function(self)
known_functions = {
    "Abs": [(lambda x: not x.is_integer, "fabs")],
    "gamma": "tgamma",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "erf": "erf",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceil",
}

# These are the core reserved words in the C language. Taken from:
# http://crasseux.com/books/ctutorial/Reserved-words-in-C.html

reserved_words = ['auto',
                  'if',
                  'break',
                  'int',
                  'case',
                  'long',
                  'char',
                  'register',
                  'continue',
                  'return',
                  'default',
                  'short',
                  'do',
                  'sizeof',
                  'double',
                  'static',
                  'else',
                  'struct',
                  'entry',
                  'switch',
                  'extern',
                  'typedef',
                  'float',
                  'union',
                  'for',
                  'unsigned',
                  'goto',
                  'while',
                  'enum',
                  'void',
                  'const',
                  'signed',
                  'volatile']
```
### 2 - sympy/printing/ccode.py:

Start line: 283, End line: 402

```python
def ccode(expr, assign_to=None, **settings):
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
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
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
    >>> ccode((2*tau)**Rational(7, 2))
    '8*sqrt(2)*pow(tau, 7.0L/2.0L)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'

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
    >>> ccode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau))
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
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(ccode(mat, A))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    """

    return CCodePrinter(settings).doprint(expr, assign_to)


def print_ccode(expr, **settings):
    """Prints C representation of the given expression."""
    print(ccode(expr, **settings))
```
### 3 - sympy/printing/ccode.py:

Start line: 155, End line: 182

```python
class CCodePrinter(CodePrinter):

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0L/%d.0L' % (p, q)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return "M_E"

    def _print_Pi(self, expr):
        return 'M_PI'

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'
```
### 4 - sympy/printing/ccode.py:

Start line: 184, End line: 214

```python
class CCodePrinter(CodePrinter):

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])
```
### 5 - sympy/functions/elementary/trigonometric.py:

Start line: 1551, End line: 1603

```python
class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Notes
    =====

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] http://dlmf.nist.gov/4.14
    .. [3] http://functions.wolfram.com/ElementaryFunctions/Csc
    """

    _reciprocal_of = sin
    _is_odd = True

    def _eval_rewrite_as_sin(self, arg):
        return (1/sin(arg))

    def _eval_rewrite_as_sincos(self, arg):
        return cos(arg)/(sin(arg)*cos(arg))

    def _eval_rewrite_as_cot(self, arg):
        cot_half = cot(arg/2)
        return (1 + cot_half**2)/(2*cot_half)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -cot(self.args[0])*csc(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)
```
### 6 - sympy/functions/elementary/hyperbolic.py:

Start line: 484, End line: 501

```python
class coth(HyperbolicFunction):
    r"""
    The hyperbolic cotangent function, `\frac{\cosh(x)}{\sinh(x)}`.

    * coth(x) -> Returns the hyperbolic cotangent of x
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sinh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acoth
```
### 7 - sympy/printing/ccode.py:

Start line: 143, End line: 153

```python
class CCodePrinter(CodePrinter):

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if expr.exp == -1:
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return 'pow(%s, %s)' % (self._print(expr.base),
                                 self._print(expr.exp))
```
### 8 - sympy/printing/ccode.py:

Start line: 216, End line: 238

```python
class CCodePrinter(CodePrinter):

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        _piecewise = Piecewise((expr.args[1], expr.args[0]), (expr.args[2], True))
        return self._print(_piecewise)

    def _print_MatrixElement(self, expr):
        return "{0}[{1}]".format(expr.parent, expr.j +
                expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):

        name = super(CCodePrinter, self)._print_Symbol(expr)

        if expr in self._dereference:
            return '(*{0})'.format(name)
        else:
            return name

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.rel_op
        rhs_code = self._print(expr.rhs)
        return "{0} {1} {2};".format(lhs_code, op, rhs_code)
```
### 9 - sympy/printing/mathematica.py:

Start line: 1, End line: 34

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

}
```
### 10 - sympy/functions/elementary/trigonometric.py:

Start line: 731, End line: 780

```python
class cos(TrigonometricFunction):

    def _eval_rewrite_as_sqrt(self, arg):
        # ... other code

        cst_table_some = {
            3: S.Half,
            5: (sqrt(5) + 1)/4,
            17: sqrt((15 + sqrt(17))/32 + sqrt(2)*(sqrt(17 - sqrt(17)) +
                sqrt(sqrt(2)*(-8*sqrt(17 + sqrt(17)) - (1 - sqrt(17))
                *sqrt(17 - sqrt(17))) + 6*sqrt(17) + 34))/32),
            257: _cospi257()
            # 65537 is the only other known Fermat prime and the very
            # large expression is intentionally omitted from SymPy; see
            # http://www.susqu.edu/brakke/constructions/65537-gon.m.txt
        }

        def _fermatCoords(n):
            # if n can be factored in terms of Fermat primes with
            # multiplicity of each being 1, return those primes, else
            # False
            from sympy import chebyshevt
            primes = []
            for p_i in cst_table_some:
                n, r = divmod(n, p_i)
                if not r:
                    primes.append(p_i)
                    if n == 1:
                        return tuple(primes)
            return False

        if pi_coeff.q in cst_table_some:
            rv = chebyshevt(pi_coeff.p, cst_table_some[pi_coeff.q])
            if pi_coeff.q < 257:
                rv = rv.expand()
            return rv

        if not pi_coeff.q % 2:  # recursively remove factors of 2
            pico2 = pi_coeff*2
            nval = cos(pico2*S.Pi).rewrite(sqrt)
            x = (pico2 + 1)/2
            sign_cos = -1 if int(x) % 2 else 1
            return sign_cos*sqrt( (1 + nval)/2 )

        FC = _fermatCoords(pi_coeff.q)
        if FC:
            decomp = ipartfrac(pi_coeff, FC)
            X = [(x[1], x[0]*S.Pi) for x in zip(decomp, numbered_symbols('z'))]
            pcls = cos(sum([x[0] for x in X]))._eval_expand_trig().subs(X)
            return pcls.rewrite(sqrt)
        else:
            decomp = ipartfrac(pi_coeff)
            X = [(x[1], x[0]*S.Pi) for x in zip(decomp, numbered_symbols('z'))]
            pcls = cos(sum([x[0] for x in X]))._eval_expand_trig().subs(X)
            return pcls
```
### 11 - sympy/printing/ccode.py:

Start line: 240, End line: 252

```python
class CCodePrinter(CodePrinter):

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step}) {{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)

    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))
```
### 31 - sympy/printing/ccode.py:

Start line: 130, End line: 141

```python
class CCodePrinter(CodePrinter):

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"
        for i in indices:
            # C arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines
```
### 61 - sympy/printing/ccode.py:

Start line: 86, End line: 128

```python
class CCodePrinter(CodePrinter):
    """A printer to convert python expressions to strings of c code"""
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'contract': True,
        'dereference': set(),
        'error_on_reserved': False,
        'reserved_word_suffix': '_',
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {0}".format(text)

    def _declare_number_const(self, name, value):
        return "double const {0} = {1};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))
```
### 111 - sympy/printing/ccode.py:

Start line: 254, End line: 280

```python
class CCodePrinter(CodePrinter):

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty
```
