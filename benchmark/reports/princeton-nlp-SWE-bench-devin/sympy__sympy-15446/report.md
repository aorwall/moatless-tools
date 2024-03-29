# sympy__sympy-15446

| **sympy/sympy** | `6399a809e2683f89d74a6540fb51293f38e9923d` |
| ---- | ---- |
| **No of patches** | 6 |
| **All found context length** | 16174 |
| **Any found context length** | 16174 |
| **Avg pos** | 23.833333333333332 |
| **Min pos** | 39 |
| **Max pos** | 104 |
| **Top file pos** | 1 |
| **Missing snippets** | 8 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -77,8 +77,18 @@ def _coeff_isneg(a):
     >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
     False
 
+    For matrix expressions:
+
+    >>> from sympy import MatrixSymbol, sqrt
+    >>> A = MatrixSymbol("A", 3, 3)
+    >>> _coeff_isneg(-sqrt(2)*A)
+    True
+    >>> _coeff_isneg(sqrt(2)*A)
+    False
     """
 
+    if a.is_MatMul:
+        a = a.args[0]
     if a.is_Mul:
         a = a.args[0]
     return a.is_Number and a.is_negative
diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py
--- a/sympy/matrices/expressions/blockmatrix.py
+++ b/sympy/matrices/expressions/blockmatrix.py
@@ -42,7 +42,7 @@ class BlockMatrix(MatrixExpr):
     Matrix([[I, Z]])
 
     >>> print(block_collapse(C*B))
-    Matrix([[X, Z*Y + Z]])
+    Matrix([[X, Z + Z*Y]])
 
     """
     def __new__(cls, *args):
@@ -283,7 +283,7 @@ def block_collapse(expr):
     Matrix([[I, Z]])
 
     >>> print(block_collapse(C*B))
-    Matrix([[X, Z*Y + Z]])
+    Matrix([[X, Z + Z*Y]])
     """
     hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)
     rule = exhaust(
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -661,7 +661,7 @@ class MatrixSymbol(MatrixExpr):
     >>> A.shape
     (3, 4)
     >>> 2*A*B + Identity(3)
-    2*A*B + I
+    I + 2*A*B
     """
     is_commutative = False
     is_symbol = True
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -302,7 +302,6 @@ def _print_bool(self, e):
     def _print_NoneType(self, e):
         return r"\mathrm{%s}" % e
 
-
     def _print_Add(self, expr, order=None):
         if self.order == 'none':
             terms = list(expr.args)
@@ -1478,34 +1477,25 @@ def _print_Adjoint(self, expr):
         else:
             return r"%s^\dagger" % self._print(mat)
 
-    def _print_MatAdd(self, expr):
-        terms = [self._print(t) for t in expr.args]
-        l = []
-        for t in terms:
-            if t.startswith('-'):
-                sign = "-"
-                t = t[1:]
-            else:
-                sign = "+"
-            l.extend([sign, t])
-        sign = l.pop(0)
-        if sign == '+':
-            sign = ""
-        return sign + ' '.join(l)
-
     def _print_MatMul(self, expr):
         from sympy import Add, MatAdd, HadamardProduct, MatMul, Mul
 
-        def parens(x):
-            if isinstance(x, (Add, MatAdd, HadamardProduct)):
-                return r"\left(%s\right)" % self._print(x)
-            return self._print(x)
+        parens = lambda x: self.parenthesize(x, precedence_traditional(expr), False)
 
-        if isinstance(expr, MatMul) and expr.args[0].is_Number and expr.args[0]<0:
-            expr = Mul(-1*expr.args[0], MatMul(*expr.args[1:]))
-            return '-' + ' '.join(map(parens, expr.args))
+        args = expr.args
+        if isinstance(args[0], Mul):
+            args = args[0].as_ordered_factors() + list(args[1:])
+        else:
+            args = list(args)
+
+        if isinstance(expr, MatMul) and _coeff_isneg(expr):
+            if args[0] == -1:
+                args = args[1:]
+            else:
+                args[0] = -args[0]
+            return '- ' + ' '.join(map(parens, args))
         else:
-            return ' '.join(map(parens, expr.args))
+            return ' '.join(map(parens, args))
 
     def _print_Mod(self, expr, exp=None):
         if exp is not None:
diff --git a/sympy/printing/precedence.py b/sympy/printing/precedence.py
--- a/sympy/printing/precedence.py
+++ b/sympy/printing/precedence.py
@@ -38,9 +38,9 @@
     "Function" : PRECEDENCE["Func"],
     "NegativeInfinity": PRECEDENCE["Add"],
     "MatAdd": PRECEDENCE["Add"],
-    "MatMul": PRECEDENCE["Mul"],
     "MatPow": PRECEDENCE["Pow"],
     "TensAdd": PRECEDENCE["Add"],
+    # As soon as `TensMul` is a subclass of `Mul`, remove this:
     "TensMul": PRECEDENCE["Mul"],
     "HadamardProduct": PRECEDENCE["Mul"],
     "KroneckerProduct": PRECEDENCE["Mul"],
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -339,22 +339,6 @@ def _print_HadamardProduct(self, expr):
         return '.*'.join([self.parenthesize(arg, precedence(expr))
             for arg in expr.args])
 
-    def _print_MatAdd(self, expr):
-        terms = [self.parenthesize(arg, precedence(expr))
-             for arg in expr.args]
-        l = []
-        for t in terms:
-            if t.startswith('-'):
-                sign = "-"
-                t = t[1:]
-            else:
-                sign = "+"
-            l.extend([sign, t])
-        sign = l.pop(0)
-        if sign == '+':
-            sign = ""
-        return sign + ' '.join(l)
-
     def _print_NaN(self, expr):
         return 'nan'
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/function.py | 80 | 80 | - | - | -
| sympy/matrices/expressions/blockmatrix.py | 45 | 45 | - | 12 | -
| sympy/matrices/expressions/blockmatrix.py | 286 | 286 | - | 12 | -
| sympy/matrices/expressions/matexpr.py | 664 | 664 | 39 | 11 | 16174
| sympy/printing/latex.py | 305 | 305 | - | 1 | -
| sympy/printing/latex.py | 1481 | 1508 | - | 1 | -
| sympy/printing/precedence.py | 41 | 41 | - | - | -
| sympy/printing/str.py | 342 | 357 | 104 | 5 | 36511


## Problem Statement

```
LaTeX printer omits necessary parentheses in matrix products such as x(-y)
The product of x and -y, where x, y are MatrixSymbols, is printed as `x -y` by the LaTeX printer:
\`\`\`
from sympy import *
x = MatrixSymbol('x', 2, 2)
y = MatrixSymbol('y', 2, 2)
expr = (x*y).subs(y, -y)
print(latex(expr))   
\`\`\`

Source: [Subsitute a matrix M by (-M) in SymPy and display it unambiguously](https://stackoverflow.com/q/53044835) on Stack Overflow.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/latex.py** | 1458 | 1494| 299 | 299 | 24270 | 
| 2 | **1 sympy/printing/latex.py** | 1436 | 1456| 208 | 507 | 24270 | 
| 3 | **1 sympy/printing/latex.py** | 1517 | 1547| 303 | 810 | 24270 | 
| 4 | **1 sympy/printing/latex.py** | 1496 | 1508| 159 | 969 | 24270 | 
| 5 | **1 sympy/printing/latex.py** | 1410 | 1435| 271 | 1240 | 24270 | 
| 6 | **1 sympy/printing/latex.py** | 2281 | 2439| 2158 | 3398 | 24270 | 
| 7 | **1 sympy/printing/latex.py** | 1 | 82| 713 | 4111 | 24270 | 
| 8 | 2 sympy/physics/vector/printing.py | 337 | 374| 339 | 4450 | 27617 | 
| 9 | **2 sympy/printing/latex.py** | 567 | 584| 197 | 4647 | 27617 | 
| 10 | **2 sympy/printing/latex.py** | 393 | 442| 383 | 5030 | 27617 | 
| 11 | **2 sympy/printing/latex.py** | 369 | 391| 268 | 5298 | 27617 | 
| 12 | **2 sympy/printing/latex.py** | 2025 | 2068| 771 | 6069 | 27617 | 
| 13 | 3 sympy/printing/pretty/pretty.py | 722 | 747| 284 | 6353 | 49176 | 
| 14 | 3 sympy/printing/pretty/pretty.py | 838 | 888| 449 | 6802 | 49176 | 
| 15 | **3 sympy/printing/latex.py** | 2135 | 2199| 754 | 7556 | 49176 | 
| 16 | **3 sympy/printing/latex.py** | 493 | 543| 587 | 8143 | 49176 | 
| 17 | 3 sympy/printing/pretty/pretty.py | 750 | 769| 207 | 8350 | 49176 | 
| 18 | 4 sympy/matrices/expressions/hadamard.py | 34 | 84| 396 | 8746 | 49817 | 
| 19 | **5 sympy/printing/str.py** | 253 | 268| 146 | 8892 | 57138 | 
| 20 | **5 sympy/printing/latex.py** | 1549 | 1596| 489 | 9381 | 57138 | 
| 21 | 6 sympy/printing/repr.py | 88 | 100| 139 | 9520 | 59320 | 
| 22 | **6 sympy/printing/latex.py** | 1349 | 1375| 210 | 9730 | 59320 | 
| 23 | **6 sympy/printing/latex.py** | 1271 | 1331| 753 | 10483 | 59320 | 
| 24 | 7 sympy/matrices/expressions/dotproduct.py | 46 | 59| 131 | 10614 | 59863 | 
| 25 | 8 sympy/matrices/expressions/matmul.py | 16 | 46| 226 | 10840 | 62624 | 
| 26 | 9 sympy/physics/quantum/tensorproduct.py | 210 | 237| 285 | 11125 | 66140 | 
| 27 | 10 sympy/printing/julia.py | 372 | 390| 187 | 11312 | 71899 | 
| 28 | **10 sympy/printing/latex.py** | 1730 | 1766| 380 | 11692 | 71899 | 
| 29 | **10 sympy/printing/latex.py** | 444 | 491| 529 | 12221 | 71899 | 
| 30 | **11 sympy/matrices/expressions/matexpr.py** | 140 | 200| 443 | 12664 | 78127 | 
| 31 | **11 sympy/printing/latex.py** | 328 | 344| 165 | 12829 | 78127 | 
| 32 | **11 sympy/printing/latex.py** | 586 | 614| 239 | 13068 | 78127 | 
| 33 | **11 sympy/printing/latex.py** | 1630 | 1695| 562 | 13630 | 78127 | 
| 34 | 11 sympy/matrices/expressions/dotproduct.py | 29 | 44| 173 | 13803 | 78127 | 
| 35 | **11 sympy/printing/latex.py** | 719 | 786| 658 | 14461 | 78127 | 
| 36 | **11 sympy/printing/latex.py** | 616 | 645| 272 | 14733 | 78127 | 
| 37 | **11 sympy/printing/latex.py** | 656 | 686| 329 | 15062 | 78127 | 
| 38 | **11 sympy/printing/latex.py** | 805 | 879| 684 | 15746 | 78127 | 
| **-> 39 <-** | **11 sympy/matrices/expressions/matexpr.py** | 652 | 709| 428 | 16174 | 78127 | 
| 40 | **11 sympy/printing/latex.py** | 907 | 962| 577 | 16751 | 78127 | 
| 41 | **11 sympy/printing/latex.py** | 2246 | 2459| 294 | 17045 | 78127 | 
| 42 | **11 sympy/printing/latex.py** | 1707 | 1717| 137 | 17182 | 78127 | 
| 43 | 11 sympy/physics/vector/printing.py | 45 | 120| 664 | 17846 | 78127 | 
| 44 | **11 sympy/printing/latex.py** | 2440 | 2466| 275 | 18121 | 78127 | 
| 45 | **12 sympy/matrices/expressions/blockmatrix.py** | 1 | 20| 219 | 18340 | 81874 | 
| 46 | 12 sympy/matrices/expressions/matmul.py | 78 | 136| 503 | 18843 | 81874 | 
| 47 | **12 sympy/printing/latex.py** | 1202 | 1213| 187 | 19030 | 81874 | 
| 48 | **12 sympy/printing/latex.py** | 1124 | 1189| 672 | 19702 | 81874 | 
| 49 | **12 sympy/printing/latex.py** | 2016 | 2023| 118 | 19820 | 81874 | 
| 50 | 12 sympy/matrices/expressions/matmul.py | 305 | 340| 229 | 20049 | 81874 | 
| 51 | 13 sympy/matrices/expressions/matadd.py | 1 | 14| 131 | 20180 | 82813 | 
| 52 | **13 sympy/printing/latex.py** | 545 | 565| 221 | 20401 | 82813 | 
| 53 | **13 sympy/matrices/expressions/matexpr.py** | 202 | 246| 417 | 20818 | 82813 | 
| 54 | **13 sympy/printing/latex.py** | 306 | 326| 149 | 20967 | 82813 | 
| 55 | 14 sympy/physics/vector/dyadic.py | 155 | 190| 401 | 21368 | 87267 | 
| 56 | **14 sympy/printing/latex.py** | 881 | 890| 136 | 21504 | 87267 | 
| 57 | **14 sympy/printing/latex.py** | 1510 | 1515| 116 | 21620 | 87267 | 
| 58 | 15 sympy/printing/fcode.py | 254 | 295| 352 | 21972 | 95188 | 
| 59 | **15 sympy/printing/latex.py** | 1697 | 1705| 135 | 22107 | 95188 | 
| 60 | **15 sympy/printing/latex.py** | 346 | 367| 196 | 22303 | 95188 | 
| 61 | 16 sympy/matrices/expressions/funcmatrix.py | 1 | 53| 390 | 22693 | 95578 | 
| 62 | 16 sympy/printing/pretty/pretty.py | 796 | 819| 216 | 22909 | 95578 | 
| 63 | **16 sympy/printing/latex.py** | 1875 | 1922| 457 | 23366 | 95578 | 
| 64 | **16 sympy/printing/latex.py** | 2223 | 2231| 129 | 23495 | 95578 | 
| 65 | 16 sympy/matrices/expressions/hadamard.py | 1 | 31| 245 | 23740 | 95578 | 
| 66 | **16 sympy/printing/latex.py** | 83 | 118| 491 | 24231 | 95578 | 
| 67 | 17 sympy/matrices/expressions/kronecker.py | 212 | 233| 208 | 24439 | 98887 | 
| 68 | **17 sympy/printing/latex.py** | 1798 | 1860| 538 | 24977 | 98887 | 
| 69 | 17 sympy/matrices/expressions/matmul.py | 1 | 13| 128 | 25105 | 98887 | 
| 70 | **17 sympy/matrices/expressions/matexpr.py** | 1 | 30| 229 | 25334 | 98887 | 
| 71 | **17 sympy/printing/latex.py** | 1719 | 1728| 131 | 25465 | 98887 | 
| 72 | **17 sympy/printing/latex.py** | 964 | 973| 126 | 25591 | 98887 | 
| 73 | **17 sympy/printing/latex.py** | 1377 | 1395| 135 | 25726 | 98887 | 
| 74 | 17 sympy/matrices/expressions/kronecker.py | 86 | 163| 648 | 26374 | 98887 | 
| 75 | **17 sympy/printing/latex.py** | 788 | 803| 162 | 26536 | 98887 | 
| 76 | **17 sympy/printing/latex.py** | 1191 | 1200| 138 | 26674 | 98887 | 
| 77 | **17 sympy/matrices/expressions/matexpr.py** | 625 | 649| 239 | 26913 | 98887 | 
| 78 | 18 sympy/parsing/latex/__init__.py | 1 | 35| 279 | 27192 | 99166 | 
| 79 | 18 sympy/matrices/expressions/kronecker.py | 1 | 18| 158 | 27350 | 99166 | 
| 80 | 18 sympy/printing/julia.py | 342 | 350| 139 | 27489 | 99166 | 
| 81 | **18 sympy/printing/latex.py** | 1397 | 1408| 171 | 27660 | 99166 | 
| 82 | **18 sympy/printing/latex.py** | 1992 | 2014| 234 | 27894 | 99166 | 
| 83 | 18 sympy/matrices/expressions/matmul.py | 203 | 224| 238 | 28132 | 99166 | 
| 84 | 18 sympy/printing/pretty/pretty.py | 772 | 794| 224 | 28356 | 99166 | 
| 85 | 19 examples/advanced/qft.py | 85 | 138| 607 | 28963 | 100481 | 
| 86 | **19 sympy/matrices/expressions/matexpr.py** | 248 | 286| 281 | 29244 | 100481 | 
| 87 | **19 sympy/printing/latex.py** | 1107 | 1122| 138 | 29382 | 100481 | 
| 88 | 20 sympy/printing/glsl.py | 318 | 496| 1932 | 31314 | 105306 | 
| 89 | 21 sympy/printing/pycode.py | 640 | 660| 195 | 31509 | 111273 | 
| 90 | **21 sympy/printing/latex.py** | 688 | 700| 159 | 31668 | 111273 | 
| 91 | 22 sympy/printing/mathml.py | 219 | 244| 211 | 31879 | 118533 | 
| 92 | 22 sympy/printing/pycode.py | 460 | 471| 141 | 32020 | 118533 | 
| 93 | **22 sympy/printing/latex.py** | 187 | 208| 219 | 32239 | 118533 | 
| 94 | **22 sympy/printing/latex.py** | 2213 | 2221| 123 | 32362 | 118533 | 
| 95 | **22 sympy/printing/latex.py** | 988 | 1072| 809 | 33171 | 118533 | 
| 96 | **22 sympy/printing/latex.py** | 1333 | 1347| 147 | 33318 | 118533 | 
| 97 | 22 sympy/printing/repr.py | 102 | 132| 243 | 33561 | 118533 | 
| 98 | **22 sympy/printing/latex.py** | 2201 | 2211| 161 | 33722 | 118533 | 
| 99 | 22 sympy/printing/julia.py | 491 | 633| 1597 | 35319 | 118533 | 
| 100 | **22 sympy/printing/latex.py** | 647 | 654| 127 | 35446 | 118533 | 
| 101 | 22 sympy/matrices/expressions/matmul.py | 48 | 76| 318 | 35764 | 118533 | 
| 102 | **22 sympy/printing/latex.py** | 1783 | 1796| 143 | 35907 | 118533 | 
| 103 | 22 sympy/physics/quantum/tensorproduct.py | 1 | 31| 185 | 36092 | 118533 | 
| **-> 104 <-** | **22 sympy/printing/str.py** | 326 | 380| 419 | 36511 | 118533 | 
| 105 | **22 sympy/matrices/expressions/blockmatrix.py** | 323 | 344| 217 | 36728 | 118533 | 
| 106 | 22 sympy/matrices/expressions/dotproduct.py | 1 | 27| 251 | 36979 | 118533 | 
| 107 | 23 sympy/printing/octave.py | 360 | 378| 189 | 37168 | 125046 | 
| 108 | 23 sympy/printing/mathml.py | 555 | 573| 160 | 37328 | 125046 | 
| 109 | 23 sympy/printing/mathml.py | 508 | 553| 336 | 37664 | 125046 | 
| 110 | 24 sympy/printing/theanocode.py | 139 | 215| 745 | 38409 | 129212 | 
| 111 | **24 sympy/printing/latex.py** | 1215 | 1269| 730 | 39139 | 129212 | 
| 112 | **24 sympy/matrices/expressions/matexpr.py** | 33 | 138| 794 | 39933 | 129212 | 
| 113 | 25 sympy/matrices/common.py | 2041 | 2074| 299 | 40232 | 146571 | 
| 114 | 26 sympy/printing/mathematica.py | 123 | 135| 106 | 40338 | 147840 | 
| 115 | 26 sympy/matrices/expressions/kronecker.py | 21 | 83| 706 | 41044 | 147840 | 
| 116 | 27 sympy/matrices/sparse.py | 1 | 17| 120 | 41164 | 158284 | 
| 117 | 27 sympy/printing/julia.py | 327 | 339| 181 | 41345 | 158284 | 
| 118 | 28 sympy/solvers/solveset.py | 1874 | 1967| 750 | 42095 | 185520 | 
| 119 | 28 sympy/physics/quantum/tensorproduct.py | 133 | 164| 275 | 42370 | 185520 | 
| 120 | 28 sympy/printing/mathml.py | 156 | 189| 260 | 42630 | 185520 | 
| 121 | 29 sympy/printing/ccode.py | 737 | 873| 1518 | 44148 | 193504 | 
| 122 | 29 sympy/printing/glsl.py | 135 | 168| 339 | 44487 | 193504 | 
| 123 | **29 sympy/printing/latex.py** | 975 | 986| 157 | 44644 | 193504 | 
| 124 | **29 sympy/printing/latex.py** | 1924 | 1979| 414 | 45058 | 193504 | 
| 125 | **29 sympy/matrices/expressions/matexpr.py** | 288 | 321| 355 | 45413 | 193504 | 
| 126 | 29 sympy/physics/quantum/tensorproduct.py | 166 | 208| 449 | 45862 | 193504 | 
| 127 | 30 sympy/physics/quantum/innerproduct.py | 121 | 139| 155 | 46017 | 194590 | 
| 128 | **30 sympy/printing/latex.py** | 2070 | 2115| 407 | 46424 | 194590 | 
| 129 | 30 sympy/printing/mathml.py | 575 | 601| 218 | 46642 | 194590 | 
| 130 | **30 sympy/printing/latex.py** | 2233 | 2243| 159 | 46801 | 194590 | 
| 131 | **30 sympy/matrices/expressions/matexpr.py** | 763 | 824| 382 | 47183 | 194590 | 
| 132 | 30 sympy/matrices/common.py | 2113 | 2156| 328 | 47511 | 194590 | 
| 133 | 30 sympy/printing/julia.py | 353 | 369| 181 | 47692 | 194590 | 
| 134 | 30 sympy/matrices/common.py | 1922 | 1951| 299 | 47991 | 194590 | 
| 135 | 30 sympy/physics/vector/printing.py | 122 | 153| 320 | 48311 | 194590 | 
| 136 | **30 sympy/printing/latex.py** | 1074 | 1105| 308 | 48619 | 194590 | 
| 137 | 30 sympy/printing/pycode.py | 500 | 558| 774 | 49393 | 194590 | 
| 138 | **30 sympy/matrices/expressions/matexpr.py** | 447 | 471| 240 | 49633 | 194590 | 
| 139 | **30 sympy/matrices/expressions/blockmatrix.py** | 388 | 397| 127 | 49760 | 194590 | 
| 140 | 31 sympy/matrices/expressions/__init__.py | 1 | 20| 187 | 49947 | 194778 | 
| 141 | **31 sympy/printing/str.py** | 207 | 251| 481 | 50428 | 194778 | 


## Missing Patch Files

 * 1: sympy/core/function.py
 * 2: sympy/matrices/expressions/blockmatrix.py
 * 3: sympy/matrices/expressions/matexpr.py
 * 4: sympy/printing/latex.py
 * 5: sympy/printing/precedence.py
 * 6: sympy/printing/str.py

## Patch

```diff
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -77,8 +77,18 @@ def _coeff_isneg(a):
     >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
     False
 
+    For matrix expressions:
+
+    >>> from sympy import MatrixSymbol, sqrt
+    >>> A = MatrixSymbol("A", 3, 3)
+    >>> _coeff_isneg(-sqrt(2)*A)
+    True
+    >>> _coeff_isneg(sqrt(2)*A)
+    False
     """
 
+    if a.is_MatMul:
+        a = a.args[0]
     if a.is_Mul:
         a = a.args[0]
     return a.is_Number and a.is_negative
diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py
--- a/sympy/matrices/expressions/blockmatrix.py
+++ b/sympy/matrices/expressions/blockmatrix.py
@@ -42,7 +42,7 @@ class BlockMatrix(MatrixExpr):
     Matrix([[I, Z]])
 
     >>> print(block_collapse(C*B))
-    Matrix([[X, Z*Y + Z]])
+    Matrix([[X, Z + Z*Y]])
 
     """
     def __new__(cls, *args):
@@ -283,7 +283,7 @@ def block_collapse(expr):
     Matrix([[I, Z]])
 
     >>> print(block_collapse(C*B))
-    Matrix([[X, Z*Y + Z]])
+    Matrix([[X, Z + Z*Y]])
     """
     hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)
     rule = exhaust(
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -661,7 +661,7 @@ class MatrixSymbol(MatrixExpr):
     >>> A.shape
     (3, 4)
     >>> 2*A*B + Identity(3)
-    2*A*B + I
+    I + 2*A*B
     """
     is_commutative = False
     is_symbol = True
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -302,7 +302,6 @@ def _print_bool(self, e):
     def _print_NoneType(self, e):
         return r"\mathrm{%s}" % e
 
-
     def _print_Add(self, expr, order=None):
         if self.order == 'none':
             terms = list(expr.args)
@@ -1478,34 +1477,25 @@ def _print_Adjoint(self, expr):
         else:
             return r"%s^\dagger" % self._print(mat)
 
-    def _print_MatAdd(self, expr):
-        terms = [self._print(t) for t in expr.args]
-        l = []
-        for t in terms:
-            if t.startswith('-'):
-                sign = "-"
-                t = t[1:]
-            else:
-                sign = "+"
-            l.extend([sign, t])
-        sign = l.pop(0)
-        if sign == '+':
-            sign = ""
-        return sign + ' '.join(l)
-
     def _print_MatMul(self, expr):
         from sympy import Add, MatAdd, HadamardProduct, MatMul, Mul
 
-        def parens(x):
-            if isinstance(x, (Add, MatAdd, HadamardProduct)):
-                return r"\left(%s\right)" % self._print(x)
-            return self._print(x)
+        parens = lambda x: self.parenthesize(x, precedence_traditional(expr), False)
 
-        if isinstance(expr, MatMul) and expr.args[0].is_Number and expr.args[0]<0:
-            expr = Mul(-1*expr.args[0], MatMul(*expr.args[1:]))
-            return '-' + ' '.join(map(parens, expr.args))
+        args = expr.args
+        if isinstance(args[0], Mul):
+            args = args[0].as_ordered_factors() + list(args[1:])
+        else:
+            args = list(args)
+
+        if isinstance(expr, MatMul) and _coeff_isneg(expr):
+            if args[0] == -1:
+                args = args[1:]
+            else:
+                args[0] = -args[0]
+            return '- ' + ' '.join(map(parens, args))
         else:
-            return ' '.join(map(parens, expr.args))
+            return ' '.join(map(parens, args))
 
     def _print_Mod(self, expr, exp=None):
         if exp is not None:
diff --git a/sympy/printing/precedence.py b/sympy/printing/precedence.py
--- a/sympy/printing/precedence.py
+++ b/sympy/printing/precedence.py
@@ -38,9 +38,9 @@
     "Function" : PRECEDENCE["Func"],
     "NegativeInfinity": PRECEDENCE["Add"],
     "MatAdd": PRECEDENCE["Add"],
-    "MatMul": PRECEDENCE["Mul"],
     "MatPow": PRECEDENCE["Pow"],
     "TensAdd": PRECEDENCE["Add"],
+    # As soon as `TensMul` is a subclass of `Mul`, remove this:
     "TensMul": PRECEDENCE["Mul"],
     "HadamardProduct": PRECEDENCE["Mul"],
     "KroneckerProduct": PRECEDENCE["Mul"],
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -339,22 +339,6 @@ def _print_HadamardProduct(self, expr):
         return '.*'.join([self.parenthesize(arg, precedence(expr))
             for arg in expr.args])
 
-    def _print_MatAdd(self, expr):
-        terms = [self.parenthesize(arg, precedence(expr))
-             for arg in expr.args]
-        l = []
-        for t in terms:
-            if t.startswith('-'):
-                sign = "-"
-                t = t[1:]
-            else:
-                sign = "+"
-            l.extend([sign, t])
-        sign = l.pop(0)
-        if sign == '+':
-            sign = ""
-        return sign + ' '.join(l)
-
     def _print_NaN(self, expr):
         return 'nan'
 

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_ccode.py b/sympy/printing/tests/test_ccode.py
--- a/sympy/printing/tests/test_ccode.py
+++ b/sympy/printing/tests/test_ccode.py
@@ -778,7 +778,7 @@ def test_MatrixElement_printing():
     assert(ccode(3 * A[0, 0]) == "3*A[0]")
 
     F = C[0, 0].subs(C, A - B)
-    assert(ccode(F) == "(-B + A)[0]")
+    assert(ccode(F) == "(A - B)[0]")
 
 
 def test_subclass_CCodePrinter():
diff --git a/sympy/printing/tests/test_fcode.py b/sympy/printing/tests/test_fcode.py
--- a/sympy/printing/tests/test_fcode.py
+++ b/sympy/printing/tests/test_fcode.py
@@ -765,7 +765,7 @@ def test_MatrixElement_printing():
     assert(fcode(3 * A[0, 0]) == "      3*A(1, 1)")
 
     F = C[0, 0].subs(C, A - B)
-    assert(fcode(F) == "      (-B + A)(1, 1)")
+    assert(fcode(F) == "      (A - B)(1, 1)")
 
 
 def test_aug_assign():
diff --git a/sympy/printing/tests/test_jscode.py b/sympy/printing/tests/test_jscode.py
--- a/sympy/printing/tests/test_jscode.py
+++ b/sympy/printing/tests/test_jscode.py
@@ -382,4 +382,4 @@ def test_MatrixElement_printing():
     assert(jscode(3 * A[0, 0]) == "3*A[0]")
 
     F = C[0, 0].subs(C, A - B)
-    assert(jscode(F) == "(-B + A)[0]")
+    assert(jscode(F) == "(A - B)[0]")
diff --git a/sympy/printing/tests/test_julia.py b/sympy/printing/tests/test_julia.py
--- a/sympy/printing/tests/test_julia.py
+++ b/sympy/printing/tests/test_julia.py
@@ -377,4 +377,4 @@ def test_MatrixElement_printing():
     assert(julia_code(3 * A[0, 0]) == "3*A[1,1]")
 
     F = C[0, 0].subs(C, A - B)
-    assert(julia_code(F) == "(-B + A)[1,1]")
+    assert(julia_code(F) == "(A - B)[1,1]")
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -1212,10 +1212,10 @@ def test_matAdd():
     C = MatrixSymbol('C', 5, 5)
     B = MatrixSymbol('B', 5, 5)
     l = LatexPrinter()
-    assert l._print_MatAdd(C - 2*B) in ['-2 B + C', 'C -2 B']
-    assert l._print_MatAdd(C + 2*B) in ['2 B + C', 'C + 2 B']
-    assert l._print_MatAdd(B - 2*C) in ['B -2 C', '-2 C + B']
-    assert l._print_MatAdd(B + 2*C) in ['B + 2 C', '2 C + B']
+    assert l._print(C - 2*B) in ['- 2 B + C', 'C -2 B']
+    assert l._print(C + 2*B) in ['2 B + C', 'C + 2 B']
+    assert l._print(B - 2*C) in ['B - 2 C', '- 2 C + B']
+    assert l._print(B + 2*C) in ['B + 2 C', '2 C + B']
 
 
 def test_matMul():
@@ -1227,13 +1227,13 @@ def test_matMul():
     l = LatexPrinter()
     assert l._print_MatMul(2*A) == '2 A'
     assert l._print_MatMul(2*x*A) == '2 x A'
-    assert l._print_MatMul(-2*A) == '-2 A'
+    assert l._print_MatMul(-2*A) == '- 2 A'
     assert l._print_MatMul(1.5*A) == '1.5 A'
     assert l._print_MatMul(sqrt(2)*A) == r'\sqrt{2} A'
     assert l._print_MatMul(-sqrt(2)*A) == r'- \sqrt{2} A'
     assert l._print_MatMul(2*sqrt(2)*x*A) == r'2 \sqrt{2} x A'
-    assert l._print_MatMul(-2*A*(A + 2*B)) in [r'-2 A \left(A + 2 B\right)',
-        r'-2 A \left(2 B + A\right)']
+    assert l._print_MatMul(-2*A*(A + 2*B)) in [r'- 2 A \left(A + 2 B\right)',
+        r'- 2 A \left(2 B + A\right)']
 
 
 def test_latex_MatrixSlice():
@@ -1682,6 +1682,14 @@ def test_issue_7117():
     assert latex(q) == r"\left(x + 1 = 2 x\right)^{2}"
 
 
+def test_issue_15439():
+    x = MatrixSymbol('x', 2, 2)
+    y = MatrixSymbol('y', 2, 2)
+    assert latex((x * y).subs(y, -y)) == r"x \left(- y\right)"
+    assert latex((x * y).subs(y, -2*y)) == r"x \left(- 2 y\right)"
+    assert latex((x * y).subs(x, -x)) == r"- x y"
+
+
 def test_issue_2934():
     assert latex(Symbol(r'\frac{a_1}{b_1}')) == '\\frac{a_1}{b_1}'
 
@@ -1728,7 +1736,7 @@ def test_MatrixElement_printing():
     assert latex(3 * A[0, 0]) == r"3 A_{0, 0}"
 
     F = C[0, 0].subs(C, A - B)
-    assert latex(F) == r"\left(-B + A\right)_{0, 0}"
+    assert latex(F) == r"\left(A - B\right)_{0, 0}"
 
 
 def test_MatrixSymbol_printing():
@@ -1737,9 +1745,9 @@ def test_MatrixSymbol_printing():
     B = MatrixSymbol("B", 3, 3)
     C = MatrixSymbol("C", 3, 3)
 
-    assert latex(-A) == r"-A"
-    assert latex(A - A*B - B) == r"-B - A B + A"
-    assert latex(-A*B - A*B*C - B) == r"-B - A B - A B C"
+    assert latex(-A) == r"- A"
+    assert latex(A - A*B - B) == r"A - A B - B"
+    assert latex(-A*B - A*B*C - B) == r"- A B - A B C - B"
 
 
 def test_Quaternion_latex_printing():
diff --git a/sympy/printing/tests/test_octave.py b/sympy/printing/tests/test_octave.py
--- a/sympy/printing/tests/test_octave.py
+++ b/sympy/printing/tests/test_octave.py
@@ -481,7 +481,7 @@ def test_MatrixElement_printing():
     assert mcode(3 * A[0, 0]) == "3*A(1, 1)"
 
     F = C[0, 0].subs(C, A - B)
-    assert mcode(F) == "(-B + A)(1, 1)"
+    assert mcode(F) == "(A - B)(1, 1)"
 
 
 def test_zeta_printing_issue_14820():
diff --git a/sympy/printing/tests/test_rcode.py b/sympy/printing/tests/test_rcode.py
--- a/sympy/printing/tests/test_rcode.py
+++ b/sympy/printing/tests/test_rcode.py
@@ -488,4 +488,4 @@ def test_MatrixElement_printing():
     assert(rcode(3 * A[0, 0]) == "3*A[0]")
 
     F = C[0, 0].subs(C, A - B)
-    assert(rcode(F) == "(-B + A)[0]")
+    assert(rcode(F) == "(A - B)[0]")
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -795,14 +795,14 @@ def test_MatrixElement_printing():
     assert(str(3 * A[0, 0]) == "3*A[0, 0]")
 
     F = C[0, 0].subs(C, A - B)
-    assert str(F) == "(-B + A)[0, 0]"
+    assert str(F) == "(A - B)[0, 0]"
 
 
 def test_MatrixSymbol_printing():
     A = MatrixSymbol("A", 3, 3)
     B = MatrixSymbol("B", 3, 3)
 
-    assert str(A - A*B - B) == "-B - A*B + A"
+    assert str(A - A*B - B) == "A - A*B - B"
     assert str(A*B - (A+B)) == "-(A + B) + A*B"
 
 

```


## Code snippets

### 1 - sympy/printing/latex.py:

Start line: 1458, End line: 1494

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

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\mathrm{tr}\left (%s \right )" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)

    def _print_MatAdd(self, expr):
        terms = [self._print(t) for t in expr.args]
        l = []
        for t in terms:
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 2 - sympy/printing/latex.py:

Start line: 1436, End line: 1456

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
### 3 - sympy/printing/latex.py:

Start line: 1517, End line: 1547

```python
class LatexPrinter(Printer):

    def _print_HadamardProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \circ '.join(map(parens, expr.args))

    def _print_KroneckerProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \otimes '.join(map(parens, expr.args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol):
            return r"\left(%s\right)^{%s}" % (self._print(base), self._print(exp))
        else:
            return "%s^{%s}" % (self._print(base), self._print(exp))

    def _print_ZeroMatrix(self, Z):
        return r"\mathbb{0}"

    def _print_Identity(self, I):
        return r"\mathbb{I}"
```
### 4 - sympy/printing/latex.py:

Start line: 1496, End line: 1508

```python
class LatexPrinter(Printer):

    def _print_MatMul(self, expr):
        from sympy import Add, MatAdd, HadamardProduct, MatMul, Mul

        def parens(x):
            if isinstance(x, (Add, MatAdd, HadamardProduct)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)

        if isinstance(expr, MatMul) and expr.args[0].is_Number and expr.args[0]<0:
            expr = Mul(-1*expr.args[0], MatMul(*expr.args[1:]))
            return '-' + ' '.join(map(parens, expr.args))
        else:
            return ' '.join(map(parens, expr.args))
```
### 5 - sympy/printing/latex.py:

Start line: 1410, End line: 1435

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
### 6 - sympy/printing/latex.py:

Start line: 2281, End line: 2439

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
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
    \left(2 \tau\right)^{\sin{\left (\frac{7}{2} \right )}}
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
    \left(2 \times \tau\right)^{\sin{\left (\frac{7}{2} \right )}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left (\frac{7}{2} \right )}

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
    \log{\left (10 \right )}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left (10 \right )}

    ``latex()`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left [ 2 / x, \quad y\right ]$

    """
    # ... other code
```
### 7 - sympy/printing/latex.py:

Start line: 1, End line: 82

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.sympify import SympifyError
from sympy.core.alphabets import greeks
from sympy.core.operations import AssocOp
from sympy.core.containers import Tuple
from sympy.logic.boolalg import true
from sympy.core.function import (_coeff_isneg,
    UndefinedFunction, AppliedUndef, Derivative)

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
### 8 - sympy/physics/vector/printing.py:

Start line: 337, End line: 374

```python
def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's latex(); see that function for more information;

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
### 9 - sympy/printing/latex.py:

Start line: 567, End line: 584

```python
class LatexPrinter(Printer):

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 10 - sympy/printing/latex.py:

Start line: 393, End line: 442

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        from sympy.core.power import Pow
        from sympy.physics.units import Quantity
        include_parens = False
        if _coeff_isneg(expr):
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        from sympy.simplify import fraction
        numer, denom = fraction(expr, exact=True)
        separator = self._settings['mul_symbol_latex']
        numbersep = self._settings['mul_symbol_latex_numbers']

        def convert(expr):
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                _tex = last_term_tex = ""

                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If quantities are present append them at the back
                args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                             (isinstance(x, Pow) and isinstance(x.base, Quantity)))

                for i, term in enumerate(args):
                    term_tex = self._print(term)

                    if self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1)):
                        term_tex = r"\left(%s\right)" % term_tex

                    if _between_two_numbers_p[0].search(last_term_tex) and \
                            _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator

                    _tex += term_tex
                    last_term_tex = term_tex
                return _tex
        # ... other code
```
### 11 - sympy/printing/latex.py:

Start line: 369, End line: 391

```python
class LatexPrinter(Printer):

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(func, PRECEDENCE['Mul'])
```
### 12 - sympy/printing/latex.py:

Start line: 2025, End line: 2068

```python
class LatexPrinter(Printer):

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_MellinTransform(self, expr):
        return r"\mathcal{M}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseMellinTransform(self, expr):
        return r"\mathcal{M}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_LaplaceTransform(self, expr):
        return r"\mathcal{L}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseLaplaceTransform(self, expr):
        return r"\mathcal{L}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_FourierTransform(self, expr):
        return r"\mathcal{F}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseFourierTransform(self, expr):
        return r"\mathcal{F}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_SineTransform(self, expr):
        return r"\mathcal{SIN}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseSineTransform(self, expr):
        return r"\mathcal{SIN}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_CosineTransform(self, expr):
        return r"\mathcal{COS}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseCosineTransform(self, expr):
        return r"\mathcal{COS}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))
```
### 15 - sympy/printing/latex.py:

Start line: 2135, End line: 2199

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
### 16 - sympy/printing/latex.py:

Start line: 493, End line: 543

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
            # issue #12886: add parentheses for superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
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
                exp = self._print(expr.exp)
                # issue #12886: add parentheses around superscripts raised to powers
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if '^' in base and expr.base.is_Symbol:
                    base = r"\left(%s\right)" % base
                elif isinstance(expr.base, Derivative
                        ) and base.startswith(r'\left('
                        ) and re.match(r'\\left\(\\d?d?dot', base
                        ) and base.endswith(r'\right)'):
                    # don't use parentheses around dotted derivative
                    base = base[6: -7]  # remove outermost added parens

                return tex % (base, exp)
```
### 19 - sympy/printing/str.py:

Start line: 253, End line: 268

```python
class StrPrinter(Printer):

    def _print_MatrixSlice(self, expr):
        def strslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(lambda arg: self._print(arg), x))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice) + ', ' +
                strslice(expr.colslice) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name
```
### 20 - sympy/printing/latex.py:

Start line: 1549, End line: 1596

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
### 22 - sympy/printing/latex.py:

Start line: 1349, End line: 1375

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
### 23 - sympy/printing/latex.py:

Start line: 1271, End line: 1331

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
### 28 - sympy/printing/latex.py:

Start line: 1730, End line: 1766

```python
class LatexPrinter(Printer):

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_ProductSet(self, p):
        if len(p.sets) > 1 and not has_variety(p.sets):
            return self._print(p.sets[0]) + "^{%d}" % len(p.sets)
        else:
            return r" \times ".join(self._print(set) for set in p.sets)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return 'Domain on ' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        items = ", ".join(map(self._print, items))
        return r"\left\{%s\right\}" % items

    _print_frozenset = _print_set
```
### 29 - sympy/printing/latex.py:

Start line: 444, End line: 491

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
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
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
### 30 - sympy/matrices/expressions/matexpr.py:

Start line: 140, End line: 200

```python
class MatrixExpr(Expr):

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        raise NotImplementedError()
        #return MatMul(other, Pow(self, S.NegativeOne))

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def is_square(self):
        return self.rows == self.cols

    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        from sympy.matrices.expressions.transpose import Transpose
        return Adjoint(Transpose(self))

    def as_real_imag(self):
        from sympy import I
        real = (S(1)/2) * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate())/(2*I)
        return (real, im)

    def _eval_inverse(self):
        from sympy.matrices.expressions.inverse import Inverse
        return Inverse(self)

    def _eval_transpose(self):
        return Transpose(self)

    def _eval_power(self, exp):
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            return self.__class__(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)
```
### 31 - sympy/printing/latex.py:

Start line: 328, End line: 344

```python
class LatexPrinter(Printer):

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    _print_Permutation = _print_Cycle
```
### 32 - sympy/printing/latex.py:

Start line: 586, End line: 614

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
### 33 - sympy/printing/latex.py:

Start line: 1630, End line: 1695

```python
class LatexPrinter(Printer):

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )
        return self._print(expr.args[0])

    def _print_tuple(self, expr):
        return r"\left ( %s\right )" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        return r"\left [ %s\right ]" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left \{ %s\right \}" % r", \quad ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)
```
### 35 - sympy/printing/latex.py:

Start line: 719, End line: 786

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

            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]

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
### 36 - sympy/printing/latex.py:

Start line: 616, End line: 645

```python
class LatexPrinter(Printer):

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Derivative(self, expr):
        if requires_partial(expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol, self._print(x), num)

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, dim, tex)

        return r"%s %s" % (tex, self.parenthesize(expr.expr, PRECEDENCE["Mul"], strict=True))
```
### 37 - sympy/printing/latex.py:

Start line: 656, End line: 686

```python
class LatexPrinter(Printer):

    def _print_Integral(self, expr):
        tex, symbols = "", []

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\, d%s" % self._print(symbol[0])
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] in ['equation', 'equation*'] \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                symbols.insert(0, r"\, d%s" % self._print(symbol))

        return r"%s %s%s" % (tex,
            self.parenthesize(expr.function, PRECEDENCE["Mul"], strict=True), "".join(symbols))
```
### 38 - sympy/printing/latex.py:

Start line: 805, End line: 879

```python
class LatexPrinter(Printer):

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        args = (symbols, self._print(expr))
        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _hprint_variadic_function(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (self._print((str(expr.func)).lower()), ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\lfloor{%s}\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\lceil{%s}\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left (%s \right )}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left (%s \right )}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
    _print_Determinant = _print_Abs

    def _print_re(self, expr, exp=None):
        tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Func'])

        return self._do_exponent(tex, exp)
```
### 39 - sympy/matrices/expressions/matexpr.py:

Start line: 652, End line: 709

```python
class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    2*A*B + I
    """
    is_commutative = False
    is_symbol = True
    _diff_wrt = True

    def __new__(cls, name, n, m):
        n, m = sympify(n), sympify(m)
        obj = Basic.__new__(cls, name, n, m)
        return obj

    def _hashable_content(self):
        return(self.name, self.shape)

    @property
    def shape(self):
        return self.args[1:3]

    @property
    def name(self):
        return self.args[0]

    def _eval_subs(self, old, new):
        # only do substitutions in shape
        shape = Tuple(*self.shape)._subs(old, new)
        return MatrixSymbol(self.name, *shape)

    def __call__(self, *args):
        raise TypeError( "%s object is not callable" % self.__class__ )

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return set((self,))

    def doit(self, **hints):
        if hints.get('deep', True):
            return type(self)(self.name, self.args[1].doit(**hints),
                    self.args[2].doit(**hints))
        else:
            return self

    def _eval_simplify(self, **kwargs):
        return self
```
### 40 - sympy/printing/latex.py:

Start line: 907, End line: 962

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
### 41 - sympy/printing/latex.py:

Start line: 2246, End line: 2459

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


def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
    # ... other code
```
### 42 - sympy/printing/latex.py:

Start line: 1707, End line: 1717

```python
class LatexPrinter(Printer):

    def _print_SingularityFunction(self, expr):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\langle %s \rangle}^{%s}" % (shift, power)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        tex = r"\theta\left(%s\right)" % self._print(expr.args[0])
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 44 - sympy/printing/latex.py:

Start line: 2440, End line: 2466

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
    if symbol_names is None:
        symbol_names = {}

    settings = {
        'fold_frac_powers' : fold_frac_powers,
        'fold_func_brackets' : fold_func_brackets,
        'fold_short_frac' : fold_short_frac,
        'inv_trig_style' : inv_trig_style,
        'itex' : itex,
        'ln_notation' : ln_notation,
        'long_frac_ratio' : long_frac_ratio,
        'mat_delim' : mat_delim,
        'mat_str' : mat_str,
        'mode' : mode,
        'mul_symbol' : mul_symbol,
        'order' : order,
        'symbol_names' : symbol_names,
    }

    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""
    print(latex(expr, **settings))
```
### 45 - sympy/matrices/expressions/blockmatrix.py:

Start line: 1, End line: 20

```python
from __future__ import print_function, division

from sympy import ask, Q
from sympy.core import Basic, Add, sympify
from sympy.core.compatibility import range
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift

from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.transpose import Transpose, transpose
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices import Matrix, ShapeError
from sympy.functions.elementary.complexes import re, im
```
### 47 - sympy/printing/latex.py:

Start line: 1202, End line: 1213

```python
class LatexPrinter(Printer):

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 48 - sympy/printing/latex.py:

Start line: 1124, End line: 1189

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
### 49 - sympy/printing/latex.py:

Start line: 2016, End line: 2023

```python
class LatexPrinter(Printer):

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex
```
### 52 - sympy/printing/latex.py:

Start line: 545, End line: 565

```python
class LatexPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 53 - sympy/matrices/expressions/matexpr.py:

Start line: 202, End line: 246

```python
class MatrixExpr(Expr):

    def _eval_derivative(self, v):
        if not isinstance(v, MatrixExpr):
            return None

        # Convert to the index-summation notation, perform the derivative, then
        # reconvert it back to matrix expression.
        from sympy import symbols, Dummy, Lambda, Trace
        i, j, m, n = symbols("i j m n", cls=Dummy)
        M = self._entry(i, j, expand=False)

        # Replace traces with summations:
        def getsum(x):
            di = Dummy("d_i")
            return Sum(x.args[0], (di, 0, x.args[0].shape[0]-1))
        M = M.replace(lambda x: isinstance(x, Trace), getsum)

        repl = {}
        if self.shape[0] == 1:
            repl[i] = 0
        if self.shape[1] == 1:
            repl[j] = 0
        if v.shape[0] == 1:
            repl[m] = 0
        if v.shape[1] == 1:
            repl[n] = 0
        res = M.diff(v[m, n])
        res = res.xreplace(repl)
        if res == 0:
            return res
        if len(repl) < 2:
            parsed = res
        else:
            if m not in repl:
                parsed = MatrixExpr.from_index_summation(res, m)
            elif i not in repl:
                parsed = MatrixExpr.from_index_summation(res, i)
            else:
                parsed = MatrixExpr.from_index_summation(res)

        if (parsed.has(m)) or (parsed.has(n)) or (parsed.has(i)) or (parsed.has(j)):
            # In this case, there are still some KroneckerDelta.
            # It's because the result is not a matrix, but a higher dimensional array.
            return None
        else:
            return parsed
```
### 54 - sympy/printing/latex.py:

Start line: 306, End line: 326

```python
class LatexPrinter(Printer):


    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif _coeff_isneg(term):
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex

        return tex
```
### 56 - sympy/printing/latex.py:

Start line: 881, End line: 890

```python
class LatexPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg (%s)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
```
### 57 - sympy/printing/latex.py:

Start line: 1510, End line: 1515

```python
class LatexPrinter(Printer):

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s\bmod{%s}\right)^{%s}' % (self.parenthesize(expr.args[0],
                    PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]), self._print(exp))
        return r'%s\bmod{%s}' % (self.parenthesize(expr.args[0],
                PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]))
```
### 59 - sympy/printing/latex.py:

Start line: 1697, End line: 1705

```python
class LatexPrinter(Printer):

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 60 - sympy/printing/latex.py:

Start line: 346, End line: 367

```python
class LatexPrinter(Printer):

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            return str_real
```
### 63 - sympy/printing/latex.py:

Start line: 1875, End line: 1922

```python
class LatexPrinter(Printer):

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
### 64 - sympy/printing/latex.py:

Start line: 2223, End line: 2231

```python
class LatexPrinter(Printer):

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (self._print(exp), tex)
        return r"\sigma^*%s" % tex
```
### 66 - sympy/printing/latex.py:

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
### 68 - sympy/printing/latex.py:

Start line: 1798, End line: 1860

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
        sets = s.args[1:]
        varsets = [r"%s \in %s" % (self._print(var), self._print(setv))
            for var, setv in zip(s.lamda.variables, sets)]
        return r"\left\{%s\; |\; %s\right\}" % (
            self._print(s.lamda.expr),
            ', '.join(varsets))
```
### 70 - sympy/matrices/expressions/matexpr.py:

Start line: 1, End line: 30

```python
from __future__ import print_function, division

from functools import wraps, reduce
import collections

from sympy.core import S, Symbol, Tuple, Integer, Basic, Expr, Eq
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import range, SYMPY_INTS, default_sort_key
from sympy.core.sympify import SympifyError, sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices import ShapeError
from sympy.simplify import simplify
from sympy.utilities.misc import filldedent


def _sympifyit(arg, retval=None):
    # This version of _sympifyit sympifies MutableMatrix objects
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval

        return __sympifyit_wrapper

    return deco
```
### 71 - sympy/printing/latex.py:

Start line: 1719, End line: 1728

```python
class LatexPrinter(Printer):

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex
```
### 72 - sympy/printing/latex.py:

Start line: 964, End line: 973

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
### 73 - sympy/printing/latex.py:

Start line: 1377, End line: 1395

```python
class LatexPrinter(Printer):

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        return "%s %s %s" % (self._print(expr.lhs),
            charmap[expr.rel_op], self._print(expr.rhs))
```
### 75 - sympy/printing/latex.py:

Start line: 788, End line: 803

```python
class LatexPrinter(Printer):

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

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
### 76 - sympy/printing/latex.py:

Start line: 1191, End line: 1200

```python
class LatexPrinter(Printer):

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 77 - sympy/matrices/expressions/matexpr.py:

Start line: 625, End line: 649

```python
class MatrixElement(Expr):

    def _eval_derivative(self, v):
        from sympy import Sum, symbols, Dummy

        if not isinstance(v, MatrixElement):
            from sympy import MatrixBase
            if isinstance(self.parent, MatrixBase):
                return self.parent.diff(v)[self.i, self.j]
            return S.Zero

        M = self.args[0]

        if M == v.args[0]:
            return KroneckerDelta(self.args[1], v.args[1])*KroneckerDelta(self.args[2], v.args[2])

        if isinstance(M, Inverse):
            i, j = self.args[1:]
            i1, i2 = symbols("z1, z2", cls=Dummy)
            Y = M.args[0]
            r1, r2 = Y.shape
            return -Sum(M[i, i1]*Y[i1, i2].diff(v)*M[i2, j], (i1, 0, r1-1), (i2, 0, r2-1))

        if self.has(v.args[0]):
            return None

        return S.Zero
```
### 81 - sympy/printing/latex.py:

Start line: 1397, End line: 1408

```python
class LatexPrinter(Printer):

    def _print_Piecewise(self, expr):
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)
```
### 82 - sympy/printing/latex.py:

Start line: 1992, End line: 2014

```python
class LatexPrinter(Printer):

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls, ", ".join(args))

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)
```
### 86 - sympy/matrices/expressions/matexpr.py:

Start line: 248, End line: 286

```python
class MatrixExpr(Expr):

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _entry(self, i, j, **kwargs):
        raise NotImplementedError(
            "Indexing not implemented for %s" % self.__class__.__name__)

    def adjoint(self):
        return adjoint(self)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product. """
        return S.One, self

    def conjugate(self):
        return conjugate(self)

    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    T = property(transpose, None, None, 'Matrix transposition.')

    def inverse(self):
        return self._eval_inverse()

    inv = inverse

    @property
    def I(self):
        return self.inverse()

    def valid_index(self, i, j):
        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return (is_valid(i) and is_valid(j) and
                (self.rows is None or
                (0 <= i) != False and (i < self.rows) != False) and
                (0 <= j) != False and (j < self.cols) != False)
```
### 87 - sympy/printing/latex.py:

Start line: 1107, End line: 1122

```python
class LatexPrinter(Printer):

    def _hprint_BesselBase(self, expr, exp, sym):
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, self._print(exp))
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex
```
### 90 - sympy/printing/latex.py:

Start line: 688, End line: 700

```python
class LatexPrinter(Printer):

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))
```
### 93 - sympy/printing/latex.py:

Start line: 187, End line: 208

```python
class LatexPrinter(Printer):

    def doprint(self, expr):
        tex = Printer.doprint(self, expr)

        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex
        elif self._settings['itex']:
            return r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)

    def _needs_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))
```
### 94 - sympy/printing/latex.py:

Start line: 2213, End line: 2221

```python
class LatexPrinter(Printer):

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (self._print(exp), tex)
        return r"\sigma%s" % tex
```
### 95 - sympy/printing/latex.py:

Start line: 988, End line: 1072

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
### 96 - sympy/printing/latex.py:

Start line: 1333, End line: 1347

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
        return r"O\left(%s\right)" % s
```
### 98 - sympy/printing/latex.py:

Start line: 2201, End line: 2211

```python
class LatexPrinter(Printer):

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])
```
### 100 - sympy/printing/latex.py:

Start line: 647, End line: 654

```python
class LatexPrinter(Printer):

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr, latex_subs)
```
### 102 - sympy/printing/latex.py:

Start line: 1783, End line: 1796

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

        return (r"\left["
              + r", ".join(self._print(el) for el in printset)
              + r"\right]")
```
### 104 - sympy/printing/str.py:

Start line: 326, End line: 380

```python
class StrPrinter(Printer):

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()
        if c.is_number and c < 0:
            expr = _keep_coeff(-c, m)
            sign = "-"
        else:
            sign = ""

        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )

    def _print_HadamardProduct(self, expr):
        return '.*'.join([self.parenthesize(arg, precedence(expr))
            for arg in expr.args])

    def _print_MatAdd(self, expr):
        terms = [self.parenthesize(arg, precedence(expr))
             for arg in expr.args]
        l = []
        for t in terms:
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_NaN(self, expr):
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        return '-oo'

    def _print_Normal(self, expr):
        return "Normal(%s, %s)" % (self._print(expr.mu), self._print(expr.sigma))

    def _print_Order(self, expr):
        if all(p is S.Zero for p in expr.point) or not len(expr.variables):
            if len(expr.variables) <= 1:
                return 'O(%s)' % self._print(expr.expr)
            else:
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)

    def _print_Ordinal(self, expr):
        return expr.__str__()

    def _print_Cycle(self, expr):
        return expr.__str__()
```
### 105 - sympy/matrices/expressions/blockmatrix.py:

Start line: 323, End line: 344

```python
def bc_block_plus_ident(expr):
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr

    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if (blocks and all(b.structurally_equal(blocks[0]) for b in blocks)
               and blocks[0].is_structurally_symmetric):
        block_id = BlockDiagMatrix(*[Identity(k)
                                        for k in blocks[0].rowblocksizes])
        return MatAdd(block_id * len(idents), *blocks).doit()

    return expr

def bc_dist(expr):
    """ Turn  a*[X, Y] into [a*X, a*Y] """
    factor, mat = expr.as_coeff_mmul()
    if factor != 1 and isinstance(unpack(mat), BlockMatrix):
        B = unpack(mat).blocks
        return BlockMatrix([[factor * B[i, j] for j in range(B.cols)]
                                              for i in range(B.rows)])
    return expr
```
### 111 - sympy/printing/latex.py:

Start line: 1215, End line: 1269

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
### 112 - sympy/matrices/expressions/matexpr.py:

Start line: 33, End line: 138

```python
class MatrixExpr(Expr):
    """ Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========
        MatrixSymbol
        MatAdd
        MatMul
        Transpose
        Inverse
    """

    # Should not be considered iterable by the
    # sympy.core.compatibility.iterable function. Subclass that actually are
    # iterable (i.e., explicit matrices) should set this to True.
    _iterable = False

    _op_priority = 11.0

    is_Matrix = True
    is_MatrixExpr = True
    is_Identity = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False

    is_commutative = False
    is_number = False
    is_symbol = False

    def __new__(cls, *args, **kwargs):
        args = map(sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    # The following is adapted from the core Expr object
    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.is_square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif self.is_Identity:
            return self
        elif other is S.Zero:
            return Identity(self.rows)
        elif other is S.One:
            return self
        return MatPow(self, other).doit()
```
### 123 - sympy/printing/latex.py:

Start line: 975, End line: 986

```python
class LatexPrinter(Printer):

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]), \
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex
```
### 124 - sympy/printing/latex.py:

Start line: 1924, End line: 1979

```python
class LatexPrinter(Printer):

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ['-', '+']:
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left (%s \right )}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex
```
### 125 - sympy/matrices/expressions/matexpr.py:

Start line: 288, End line: 321

```python
class MatrixExpr(Expr):

    def __getitem__(self, key):
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            i, j = sympify(i), sympify(j)
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid indices (%s, %s)" % (i, j))
        elif isinstance(key, (SYMPY_INTS, Integer)):
            # row-wise decomposition of matrix
            rows, cols = self.shape
            # allow single indexing if number of columns is known
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('''
                    Single indexing is only supported when the number
                    of columns is known.'''))
            key = sympify(key)
            i = key // cols
            j = key % cols
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid index %s" % key)
        elif isinstance(key, (Symbol, Expr)):
                raise IndexError(filldedent('''
                    Only integers may be used when addressing the matrix
                    with a single index.'''))
        raise IndexError("Invalid index, wanted %s[i,j]" % self)
```
### 128 - sympy/printing/latex.py:

Start line: 2070, End line: 2115

```python
class LatexPrinter(Printer):

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

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

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return "\\mathbf{%s}" % self._print(Symbol(morphism.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result
```
### 130 - sympy/printing/latex.py:

Start line: 2233, End line: 2243

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
### 131 - sympy/matrices/expressions/matexpr.py:

Start line: 763, End line: 824

```python
class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A+Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True

    def __new__(cls, m, n):
        return super(ZeroMatrix, cls).__new__(cls, m, n)

    @property
    def shape(self):
        return (self.args[0], self.args[1])


    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if other != 1 and not self.is_square:
            raise ShapeError("Power of non-square matrix %s" % self)
        if other == 0:
            return Identity(self.rows)
        if other < 1:
            raise ValueError("Matrix det == 0; not invertible.")
        return self

    def _eval_transpose(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.Zero

    def _eval_determinant(self):
        return S.Zero

    def conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.Zero

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__


def matrix_symbols(expr):
    return [sym for sym in expr.free_symbols if sym.is_Matrix]

from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
```
### 136 - sympy/printing/latex.py:

Start line: 1074, End line: 1105

```python
class LatexPrinter(Printer):

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
### 138 - sympy/matrices/expressions/matexpr.py:

Start line: 447, End line: 471

```python
class MatrixExpr(Expr):

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        # ... other code

        def remove_matelement(expr, i1, i2):

            def repl_match(pos):
                def func(x):
                    if not isinstance(x, MatrixElement):
                        return False
                    if x.args[pos] != i1:
                        return False
                    if x.args[3-pos] == 0:
                        if x.args[0].shape[2-pos] == 1:
                            return True
                        else:
                            return False
                    return True
                return func

            expr = expr.replace(repl_match(1),
                lambda x: x.args[0])
            expr = expr.replace(repl_match(2),
                lambda x: transpose(x.args[0]))

            # Make sure that all Mul are transformed to MatMul and that they
            # are flattened:
            rule = bottom_up(lambda x: reduce(lambda a, b: a*b, x.args) if isinstance(x, (Mul, MatMul)) else x)
            return rule(expr)
        # ... other code
```
### 139 - sympy/matrices/expressions/blockmatrix.py:

Start line: 388, End line: 397

```python
def blockinverse_2x2(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        # Cite: The Matrix Cookbook Section 9.1.3
        [[A, B],
         [C, D]] = expr.arg.blocks.tolist()

        return BlockMatrix([[ (A - B*D.I*C).I,  (-A).I*B*(D - C*A.I*B).I],
                            [-(D - C*A.I*B).I*C*A.I,     (D - C*A.I*B).I]])
    else:
        return expr
```
### 141 - sympy/printing/str.py:

Start line: 207, End line: 251

```python
class StrPrinter(Printer):

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s^-1" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        args, expr = obj.args
        if len(args) == 1:
            return "Lambda(%s, %s)" % (self._print(args.args[0]), self._print(expr))
        else:
            arg_string = ", ".join(self._print(arg) for arg in args)
            return "Lambda((%s), %s)" % (arg_string, self._print(expr))

    def _print_LatticeOp(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        if str(dir) == "+":
            return "Limit(%s, %s, %s)" % tuple(map(self._print, (e, z, z0)))
        else:
            return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print,
                                                            (e, z, z0, dir)))

    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)
    _print_SparseMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))
```
