# sympy__sympy-22706

| **sympy/sympy** | `d5f5ed31adf36c8f98459acb87ba97d62ee135b6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 700 |
| **Any found context length** | 700 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -287,13 +287,15 @@ def _print_Mul(self, expr):
                     e = Mul._from_args(dargs)
                 d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base
 
+            pre = []
             # don't parenthesize first factor if negative
-            if n[0].could_extract_minus_sign():
+            if n and n[0].could_extract_minus_sign():
                 pre = [str(n.pop(0))]
-            else:
-                pre = []
+
             nfactors = pre + [self.parenthesize(a, prec, strict=False)
                 for a in n]
+            if not nfactors:
+                nfactors = ['1']
 
             # don't parenthesize first of denominator unless singleton
             if len(d) > 1 and d[0].could_extract_minus_sign():

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/str.py | 290 | 293 | 1 | 1 | 700


## Problem Statement

```
IndexError in StrPrinter for UnevaluatedMul
`print(Mul(Pow(x,-2, evaluate=False), Pow(3,-1,evaluate=False), evaluate=False))` gives 
`    if _coeff_isneg(n[0]):
IndexError: list index out of range`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/printing/str.py** | 265 | 343| 700 | 700 | 8481 | 
| 2 | **1 sympy/printing/str.py** | 381 | 396| 126 | 826 | 8481 | 
| 3 | **1 sympy/printing/str.py** | 344 | 379| 357 | 1183 | 8481 | 
| 4 | 2 sympy/printing/codeprinter.py | 502 | 558| 551 | 1734 | 16638 | 
| 5 | 3 sympy/printing/julia.py | 119 | 191| 697 | 2431 | 22354 | 
| 6 | **3 sympy/printing/str.py** | 668 | 735| 504 | 2935 | 22354 | 
| 7 | 4 sympy/core/mul.py | 713 | 740| 283 | 3218 | 40408 | 
| 8 | 4 sympy/core/mul.py | 1350 | 1424| 660 | 3878 | 40408 | 
| 9 | 5 sympy/printing/octave.py | 137 | 209| 700 | 4578 | 47000 | 
| 10 | 6 sympy/printing/repr.py | 194 | 206| 145 | 4723 | 50035 | 
| 11 | **6 sympy/printing/str.py** | 606 | 666| 603 | 5326 | 50035 | 
| 12 | 6 sympy/core/mul.py | 1931 | 2013| 719 | 6045 | 50035 | 
| 13 | **6 sympy/printing/str.py** | 398 | 423| 200 | 6245 | 50035 | 
| 14 | **6 sympy/printing/str.py** | 737 | 759| 239 | 6484 | 50035 | 
| 15 | 7 sympy/printing/maple.py | 226 | 237| 168 | 6652 | 52780 | 
| 16 | **7 sympy/printing/str.py** | 167 | 188| 220 | 6872 | 52780 | 
| 17 | 7 sympy/core/mul.py | 1667 | 1690| 254 | 7126 | 52780 | 
| 18 | 7 sympy/core/mul.py | 476 | 559| 863 | 7989 | 52780 | 
| 19 | 8 sympy/printing/lambdarepr.py | 73 | 184| 963 | 8952 | 54431 | 
| 20 | 8 sympy/core/mul.py | 560 | 650| 830 | 9782 | 54431 | 
| 21 | **8 sympy/printing/str.py** | 531 | 604| 529 | 10311 | 54431 | 
| 22 | **8 sympy/printing/str.py** | 209 | 246| 399 | 10710 | 54431 | 
| 23 | 8 sympy/printing/julia.py | 219 | 261| 260 | 10970 | 54431 | 
| 24 | 8 sympy/core/mul.py | 991 | 1021| 396 | 11366 | 54431 | 
| 25 | 9 sympy/printing/llvmjitcode.py | 59 | 79| 242 | 11608 | 58446 | 
| 26 | 9 sympy/core/mul.py | 937 | 977| 310 | 11918 | 58446 | 
| 27 | **9 sympy/printing/str.py** | 782 | 806| 247 | 12165 | 58446 | 
| 28 | 9 sympy/core/mul.py | 1598 | 1629| 255 | 12420 | 58446 | 
| 29 | 10 sympy/printing/numpy.py | 179 | 233| 743 | 13163 | 63398 | 
| 30 | 11 sympy/printing/mathml.py | 671 | 716| 384 | 13547 | 80297 | 
| 31 | 11 sympy/printing/codeprinter.py | 560 | 591| 328 | 13875 | 80297 | 
| 32 | 12 sympy/matrices/expressions/matmul.py | 74 | 114| 380 | 14255 | 83954 | 
| 33 | **12 sympy/printing/str.py** | 425 | 461| 379 | 14634 | 83954 | 
| 34 | 12 sympy/core/mul.py | 1426 | 1474| 452 | 15086 | 83954 | 
| 35 | 12 sympy/printing/mathml.py | 200 | 233| 261 | 15347 | 83954 | 
| 36 | 12 sympy/printing/mathml.py | 1781 | 1808| 231 | 15578 | 83954 | 
| 37 | 12 sympy/core/mul.py | 1631 | 1648| 158 | 15736 | 83954 | 
| 38 | 12 sympy/printing/llvmjitcode.py | 81 | 109| 267 | 16003 | 83954 | 
| 39 | **12 sympy/printing/str.py** | 463 | 529| 638 | 16641 | 83954 | 
| 40 | 12 sympy/printing/codeprinter.py | 252 | 287| 257 | 16898 | 83954 | 
| 41 | 12 sympy/core/mul.py | 1023 | 1037| 182 | 17080 | 83954 | 
| 42 | 13 sympy/printing/pretty/pretty.py | 1939 | 1997| 591 | 17671 | 109313 | 
| 43 | 14 sympy/printing/pycode.py | 370 | 426| 468 | 18139 | 114572 | 
| 44 | 14 sympy/core/mul.py | 1560 | 1596| 270 | 18409 | 114572 | 
| 45 | 14 sympy/matrices/expressions/matmul.py | 152 | 192| 354 | 18763 | 114572 | 
| 46 | 14 sympy/core/mul.py | 742 | 760| 146 | 18909 | 114572 | 
| 47 | 15 sympy/printing/glsl.py | 335 | 549| 139 | 19048 | 119801 | 
| 48 | 15 sympy/core/mul.py | 2015 | 2025| 135 | 19183 | 119801 | 
| 49 | 16 sympy/printing/fortran.py | 301 | 322| 253 | 19436 | 126453 | 
| 50 | 16 sympy/printing/codeprinter.py | 443 | 500| 559 | 19995 | 126453 | 
| 51 | **16 sympy/printing/str.py** | 52 | 71| 142 | 20137 | 126453 | 
| 52 | 17 sympy/printing/tensorflow.py | 181 | 193| 138 | 20275 | 128873 | 
| 53 | **17 sympy/printing/str.py** | 248 | 263| 161 | 20436 | 128873 | 
| 54 | 17 sympy/printing/julia.py | 1 | 43| 481 | 20917 | 128873 | 
| 55 | 17 sympy/core/mul.py | 329 | 447| 894 | 21811 | 128873 | 
| 56 | 17 sympy/printing/codeprinter.py | 173 | 250| 718 | 22529 | 128873 | 
| 57 | 18 sympy/printing/mathematica.py | 235 | 253| 162 | 22691 | 132664 | 
| 58 | 18 sympy/core/mul.py | 835 | 846| 142 | 22833 | 132664 | 
| 59 | 18 sympy/printing/fortran.py | 324 | 338| 131 | 22964 | 132664 | 
| 60 | 18 sympy/printing/fortran.py | 340 | 357| 171 | 23135 | 132664 | 
| 61 | 18 sympy/printing/numpy.py | 294 | 307| 143 | 23278 | 132664 | 
| 62 | 18 sympy/core/mul.py | 1692 | 1711| 205 | 23483 | 132664 | 
| 63 | 18 sympy/core/mul.py | 1247 | 1349| 867 | 24350 | 132664 | 
| 64 | 18 sympy/core/mul.py | 1713 | 1801| 814 | 25164 | 132664 | 
| 65 | 18 sympy/core/mul.py | 1803 | 1929| 955 | 26119 | 132664 | 
| 66 | 18 sympy/core/mul.py | 864 | 916| 460 | 26579 | 132664 | 
| 67 | 18 sympy/printing/pretty/pretty.py | 883 | 895| 152 | 26731 | 132664 | 
| 68 | 18 sympy/printing/julia.py | 385 | 420| 323 | 27054 | 132664 | 
| 69 | **18 sympy/printing/str.py** | 761 | 780| 185 | 27239 | 132664 | 
| 70 | 18 sympy/core/mul.py | 1476 | 1519| 280 | 27519 | 132664 | 
| 71 | **18 sympy/printing/str.py** | 73 | 165| 823 | 28342 | 132664 | 
| 72 | 18 sympy/core/mul.py | 979 | 989| 121 | 28463 | 132664 | 
| 73 | 18 sympy/core/mul.py | 38 | 89| 396 | 28859 | 132664 | 
| 74 | 18 sympy/core/mul.py | 92 | 189| 709 | 29568 | 132664 | 
| 75 | 19 sympy/core/evalf.py | 636 | 717| 694 | 30262 | 148597 | 
| 76 | 19 sympy/core/mul.py | 270 | 328| 524 | 30786 | 148597 | 
| 77 | 19 sympy/core/evalf.py | 718 | 758| 458 | 31244 | 148597 | 
| 78 | 19 sympy/printing/pretty/pretty.py | 1892 | 1937| 354 | 31598 | 148597 | 
| 79 | 19 sympy/core/mul.py | 1039 | 1069| 301 | 31899 | 148597 | 
| 80 | 19 sympy/core/mul.py | 1521 | 1540| 149 | 32048 | 148597 | 
| 81 | 19 sympy/printing/pycode.py | 223 | 237| 123 | 32171 | 148597 | 
| 82 | 19 sympy/printing/mathematica.py | 301 | 312| 177 | 32348 | 148597 | 
| 83 | 19 sympy/printing/mathematica.py | 122 | 233| 780 | 33128 | 148597 | 
| 84 | 19 sympy/printing/pycode.py | 588 | 629| 405 | 33533 | 148597 | 
| 85 | 19 sympy/core/mul.py | 918 | 935| 144 | 33677 | 148597 | 
| 86 | **19 sympy/printing/str.py** | 808 | 910| 817 | 34494 | 148597 | 
| 87 | 19 sympy/printing/pycode.py | 429 | 457| 275 | 34769 | 148597 | 
| 88 | 20 sympy/printing/printer.py | 1 | 212| 1912 | 36681 | 151985 | 
| 89 | 20 sympy/printing/pretty/pretty.py | 2062 | 2081| 199 | 36880 | 151985 | 
| 90 | **20 sympy/printing/str.py** | 190 | 207| 153 | 37033 | 151985 | 
| 91 | 20 sympy/printing/codeprinter.py | 415 | 441| 303 | 37336 | 151985 | 
| 92 | 20 sympy/printing/numpy.py | 93 | 104| 129 | 37465 | 151985 | 
| 93 | 20 sympy/matrices/expressions/matmul.py | 1 | 18| 179 | 37644 | 151985 | 
| 94 | 20 sympy/matrices/expressions/matmul.py | 312 | 354| 354 | 37998 | 151985 | 
| 95 | **20 sympy/printing/str.py** | 21 | 50| 228 | 38226 | 151985 | 
| 96 | 20 sympy/matrices/expressions/matmul.py | 194 | 217| 195 | 38421 | 151985 | 
| 97 | **20 sympy/printing/str.py** | 912 | 976| 523 | 38944 | 151985 | 
| 98 | 20 sympy/printing/codeprinter.py | 139 | 171| 343 | 39287 | 151985 | 
| 99 | 21 sympy/tensor/index_methods.py | 65 | 101| 290 | 39577 | 155861 | 
| 100 | 21 sympy/printing/octave.py | 1 | 60| 660 | 40237 | 155861 | 
| 101 | 21 sympy/printing/numpy.py | 106 | 114| 131 | 40368 | 155861 | 
| 102 | 21 sympy/printing/mathematica.py | 314 | 339| 295 | 40663 | 155861 | 
| 103 | 21 sympy/matrices/expressions/matmul.py | 40 | 72| 236 | 40899 | 155861 | 
| 104 | 21 sympy/printing/mathml.py | 1173 | 1210| 333 | 41232 | 155861 | 
| 105 | 22 sympy/core/power.py | 895 | 952| 695 | 41927 | 173551 | 
| 106 | 22 sympy/printing/pretty/pretty.py | 957 | 984| 278 | 42205 | 173551 | 
| 107 | 22 sympy/printing/numpy.py | 116 | 148| 390 | 42595 | 173551 | 
| 108 | 22 sympy/printing/pycode.py | 632 | 650| 175 | 42770 | 173551 | 
| 109 | 23 sympy/matrices/expressions/matpow.py | 97 | 143| 421 | 43191 | 174656 | 
| 110 | 23 sympy/core/mul.py | 651 | 711| 578 | 43769 | 174656 | 
| 111 | 23 sympy/printing/numpy.py | 235 | 249| 203 | 43972 | 174656 | 
| 112 | 23 sympy/printing/glsl.py | 266 | 308| 328 | 44300 | 174656 | 
| 113 | 23 sympy/printing/maple.py | 239 | 256| 214 | 44514 | 174656 | 
| 114 | 23 sympy/printing/julia.py | 193 | 216| 249 | 44763 | 174656 | 
| 115 | 24 sympy/printing/rcode.py | 143 | 153| 130 | 44893 | 178256 | 
| 116 | 24 sympy/printing/numpy.py | 400 | 436| 368 | 45261 | 178256 | 
| 117 | 24 sympy/core/mul.py | 1071 | 1131| 438 | 45699 | 178256 | 
| 118 | 25 examples/beginner/precision.py | 1 | 23| 128 | 45827 | 178384 | 
| 119 | 25 sympy/printing/pretty/pretty.py | 941 | 955| 140 | 45967 | 178384 | 
| 120 | 26 sympy/printing/python.py | 1 | 40| 310 | 46277 | 179106 | 
| 121 | 26 sympy/printing/pretty/pretty.py | 1192 | 1223| 317 | 46594 | 179106 | 
| 122 | 26 sympy/matrices/expressions/matpow.py | 45 | 58| 158 | 46752 | 179106 | 
| 123 | 27 sympy/matrices/expressions/kronecker.py | 376 | 396| 172 | 46924 | 182750 | 
| 124 | 27 sympy/printing/julia.py | 293 | 330| 215 | 47139 | 182750 | 
| 125 | 27 sympy/printing/numpy.py | 251 | 275| 223 | 47362 | 182750 | 
| 126 | 27 sympy/printing/fortran.py | 204 | 213| 148 | 47510 | 182750 | 
| 127 | 28 sympy/assumptions/sathandlers.py | 233 | 244| 151 | 47661 | 185302 | 
| 128 | 28 sympy/printing/octave.py | 468 | 493| 231 | 47892 | 185302 | 
| 129 | 28 sympy/printing/fortran.py | 258 | 299| 334 | 48226 | 185302 | 
| 130 | 28 sympy/printing/pretty/pretty.py | 137 | 204| 639 | 48865 | 185302 | 
| 131 | 28 sympy/printing/octave.py | 237 | 262| 221 | 49086 | 185302 | 
| 132 | **28 sympy/printing/str.py** | 1 | 17| 144 | 49230 | 185302 | 
| 133 | 29 sympy/printing/jscode.py | 98 | 112| 168 | 49398 | 188306 | 
| 134 | 30 sympy/printing/c.py | 277 | 292| 215 | 49613 | 194990 | 
| 135 | 30 sympy/printing/codeprinter.py | 122 | 137| 200 | 49813 | 194990 | 


## Patch

```diff
diff --git a/sympy/printing/str.py b/sympy/printing/str.py
--- a/sympy/printing/str.py
+++ b/sympy/printing/str.py
@@ -287,13 +287,15 @@ def _print_Mul(self, expr):
                     e = Mul._from_args(dargs)
                 d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base
 
+            pre = []
             # don't parenthesize first factor if negative
-            if n[0].could_extract_minus_sign():
+            if n and n[0].could_extract_minus_sign():
                 pre = [str(n.pop(0))]
-            else:
-                pre = []
+
             nfactors = pre + [self.parenthesize(a, prec, strict=False)
                 for a in n]
+            if not nfactors:
+                nfactors = ['1']
 
             # don't parenthesize first of denominator unless singleton
             if len(d) > 1 and d[0].could_extract_minus_sign():

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_str.py b/sympy/printing/tests/test_str.py
--- a/sympy/printing/tests/test_str.py
+++ b/sympy/printing/tests/test_str.py
@@ -1103,6 +1103,10 @@ def test_issue_21823():
     assert str(Partition({1, 2})) == 'Partition({1, 2})'
 
 
+def test_issue_22689():
+    assert str(Mul(Pow(x,-2, evaluate=False), Pow(3,-1,evaluate=False), evaluate=False)) == "1/(x**2*3)"
+
+
 def test_issue_21119_21460():
     ss = lambda x: str(S(x, evaluate=False))
     assert ss('4/2') == '4/2'

```


## Code snippets

### 1 - sympy/printing/str.py:

Start line: 265, End line: 343

```python
class StrPrinter(Printer):

    def _print_Mul(self, expr):

        prec = precedence(expr)

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = expr.args
        if args[0] is S.One or any(
                isinstance(a, Number) or
                a.is_Pow and all(ai.is_Integer for ai in a.args)
                for a in args[1:]):
            d, n = sift(args, lambda x:
                isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0),
                binary=True)
            for i, di in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base

            # don't parenthesize first factor if negative
            if n[0].could_extract_minus_sign():
                pre = [str(n.pop(0))]
            else:
                pre = []
            nfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in n]

            # don't parenthesize first of denominator unless singleton
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [str(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in d]

            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        def apow(i):
            b, e = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        # ... other code
```
### 2 - sympy/printing/str.py:

Start line: 381, End line: 396

```python
class StrPrinter(Printer):

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()

        sign = ""
        if c.is_number:
            re, im = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"

        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )
```
### 3 - sympy/printing/str.py:

Start line: 344, End line: 379

```python
class StrPrinter(Printer):

    def _print_Mul(self, expr):
        # ... other code
        for item in args:
            if (item.is_commutative and
                    isinstance(item, Pow) and
                    bool(item.exp.as_coeff_Mul()[0] < 0)):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if (len(item.args[0].args) != 1 and
                            isinstance(item.base, (Mul, Pow))):
                        # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)
```
### 4 - sympy/printing/codeprinter.py:

Start line: 502, End line: 558

```python
class CodePrinter(StrPrinter):

    def _print_Mul(self, expr):

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            else:
                a.append(item)

        a = a or [S.One]

        if len(a) == 1 and sign == "-":
            # Unary minus does not have a SymPy class, and hence there's no
            # precedence weight associated with it, Python's unary minus has
            # an operator precedence between multiplication and exponentiation,
            # so we use this to compute a weight.
            a_str = [self.parenthesize(a[0], 0.5*(PRECEDENCE["Pow"]+PRECEDENCE["Mul"]))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)
```
### 5 - sympy/printing/julia.py:

Start line: 119, End line: 191

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Mul(self, expr):
        # print complex numbers nicely in Julia
        if (expr.is_number and expr.is_imaginary and
                expr.as_coeff_Mul()[0].is_integer):
            return "%sim" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))
```
### 6 - sympy/printing/str.py:

Start line: 668, End line: 735

```python
class StrPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False),
                         self.parenthesize(expr.exp, PREC, strict=False))

    def _print_Integer(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(%s)" % (expr)
        return str(expr.p)

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            if self._settings.get("sympy_integers", False):
                return "S(%s)/%s" % (expr.p, expr.q)
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)
```
### 7 - sympy/core/mul.py:

Start line: 713, End line: 740

```python
class Mul(Expr, AssocOp):

    def _eval_power(self, e):

        # don't break up NC terms: (A*B)**3 != A**3*B**3, it is A*B*A*B*A*B
        cargs, nc = self.args_cnc(split_1=False)

        if e.is_Integer:
            return Mul(*[Pow(b, e, evaluate=False) for b in cargs]) * \
                Pow(Mul._from_args(nc), e, evaluate=False)
        if e.is_Rational and e.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    from .power import integer_nthroot
                    n, d = abs(a/2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    if t:
                        d, t = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n)/d
                            return _unevaluated_Mul(r**e.p, (1 + sign(a)*S.ImaginaryUnit)**e.p)

        p = Pow(self, e, evaluate=False)

        if e.is_Rational or e.is_Float:
            return p._eval_expand_power_base()

        return p
```
### 8 - sympy/core/mul.py:

Start line: 1350, End line: 1424

```python
class Mul(Expr, AssocOp):
    def _eval_is_integer(self):
        from sympy.ntheory.factor_ import trailing
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        unknown = False
        for a in self.args:
            hit = False
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer:
                    hit = unknown = True
                if e.is_negative:
                    denominators.append(2 if a is S.Half else
                        Pow(a, S.NegativeOne))
                elif not hit:
                    # int b and pos int e: a = b**e is integer
                    assert not e.is_positive
                    # for rational self and e equal to zero: a = b**e is 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer unknown
            else:
                return

        if not denominators and not unknown:
            return True

        allodd = lambda x: all(i.is_odd for i in x)
        alleven = lambda x: all(i.is_even for i in x)
        anyeven = lambda x: any(i.is_even for i in x)

        from .relational import is_gt
        if not numerators and denominators and all(is_gt(_, S.One)
                for _ in denominators):
            return False
        elif unknown:
            return
        elif allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators
                ) and (Mul(*denominators, evaluate=False) - 1
                ).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                # if minimal power of 2 in num vs den is not
                # negative then we have an integer
                if (Add(*[i.as_base_exp()[1] for i in
                        numerators if i.is_even]) - trailing(d.p)
                        ).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                # if minimal power of 2 in den vs num is positive
                # then we have have a non-integer
                if (Add(*[i.as_base_exp()[1] for i in
                        denominators if i.is_even]) - trailing(n.p)
                        ).is_positive:
                    return False
```
### 9 - sympy/printing/octave.py:

Start line: 137, End line: 209

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            return "%si" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))
```
### 10 - sympy/printing/repr.py:

Start line: 194, End line: 206

```python
class ReprPrinter(Printer):

    def _print_Mul(self, expr, order=None):
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        nargs = len(args)
        args = map(self._print, args)
        clsname = type(expr).__name__
        if nargs > 255:  # Issue #10259, Python < 3.7
            return clsname + "(*[%s])" % ", ".join(args)
        return clsname + "(%s)" % ", ".join(args)
```
### 11 - sympy/printing/str.py:

Start line: 606, End line: 666

```python
class StrPrinter(Printer):

    def _print_Pow(self, expr, rational=False):
        """Printing helper function for ``Pow``

        Parameters
        ==========

        rational : bool, optional
            If ``True``, it will not attempt printing ``sqrt(x)`` or
            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``
            instead.

            See examples for additional details

        Examples
        ========

        >>> from sympy.functions import sqrt
        >>> from sympy.printing.str import StrPrinter
        >>> from sympy.abc import x

        How ``rational`` keyword works with ``sqrt``:

        >>> printer = StrPrinter()
        >>> printer._print_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._print_Pow(sqrt(x), rational=False)
        'sqrt(x)'
        >>> printer._print_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._print_Pow(1/sqrt(x), rational=False)
        '1/sqrt(x)'

        Notes
        =====

        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,
        so there is no need of defining a separate printer for ``sqrt``.
        Instead, it should be handled here as well.
        """
        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "%s/sqrt(%s)" % tuple(map(lambda arg: self._print(arg), (S.One, expr.base)))
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '%s/%s' % (self._print(S.One),
                                  self.parenthesize(expr.base, PREC, strict=False))

        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)
```
### 13 - sympy/printing/str.py:

Start line: 398, End line: 423

```python
class StrPrinter(Printer):

    def _print_ElementwiseApplyFunction(self, expr):
        return "{}.({})".format(
            expr.function,
            self._print(expr.expr),
        )

    def _print_NaN(self, expr):
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        return '-oo'

    def _print_Order(self, expr):
        if not expr.variables or all(p is S.Zero for p in expr.point):
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
### 14 - sympy/printing/str.py:

Start line: 737, End line: 759

```python
class StrPrinter(Printer):

    def _print_Float(self, expr):
        prec = expr._prec
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        if rv.startswith('+'):
            # e.g., +inf -> inf
            rv = rv[1:]
        return rv
```
### 16 - sympy/printing/str.py:

Start line: 167, End line: 188

```python
class StrPrinter(Printer):

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default 1/2 for
        # 2nd args
        return expr.func.__name__ + "(%s)" % self.stringify(expr.pargs, ", ")

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Infinity(self, expr):
        return 'oo'

    def _print_Integral(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Integral(%s, %s)' % (self._print(expr.function), L)
```
### 21 - sympy/printing/str.py:

Start line: 531, End line: 604

```python
class StrPrinter(Printer):

    def _print_Poly(self, expr):
        ATOM_PREC = PRECEDENCE["Atom"] - 1
        terms, gens = [], [ self.parenthesize(s, ATOM_PREC) for s in expr.gens ]

        for monom, coeff in expr.terms():
            s_monom = []

            for i, e in enumerate(monom):
                if e > 0:
                    if e == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + "**%d" % e)

            s_monom = "*".join(s_monom)

            if coeff.is_Add:
                if s_monom:
                    s_coeff = "(" + self._print(coeff) + ")"
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
                s_term = s_coeff + "*" + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        format = expr.__class__.__name__ + "(%s, %s"

        from sympy.polys.polyerrors import PolynomialError

        try:
            format += ", modulus=%s" % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()

        format += ")"

        for index, item in enumerate(gens):
            if len(item) > 2 and (item[:1] == "(" and item[len(item) - 1:] == ")"):
                gens[index] = item[1:len(item) - 1]

        return format % (' '.join(terms), ', '.join(gens))

    def _print_UniversalSet(self, p):
        return 'UniversalSet'

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())
```
### 22 - sympy/printing/str.py:

Start line: 209, End line: 246

```python
class StrPrinter(Printer):

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

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

    def _print_List(self, expr):
        return self._print_list(expr)

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))
```
### 27 - sympy/printing/str.py:

Start line: 782, End line: 806

```python
class StrPrinter(Printer):

    def _print_ComplexRootOf(self, expr):
        return "CRootOf(%s, %d)" % (self._print_Add(expr.expr,  order='lex'),
                                    expr.index)

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        return "RootSum(%s)" % ", ".join(args)

    def _print_GroebnerBasis(self, basis):
        cls = basis.__class__.__name__

        exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
        exprs = "[%s]" % ", ".join(exprs)

        gens = [ self._print(gen) for gen in basis.gens ]
        domain = "domain='%s'" % self._print(basis.domain)
        order = "order='%s'" % self._print(basis.order)

        args = [exprs] + gens + [domain, order]

        return "%s(%s)" % (cls, ", ".join(args))
```
### 33 - sympy/printing/str.py:

Start line: 425, End line: 461

```python
class StrPrinter(Printer):

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import SymPyDeprecationWarning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            SymPyDeprecationWarning(
                feature="Permutation.print_cyclic = {}".format(perm_cyclic),
                useinstead="init_printing(perm_cyclic={})"
                .format(perm_cyclic),
                issue=15201,
                deprecated_since_version="1.6").warn()
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return '()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                return 'Permutation([], size=%s)' % self._print(expr.size)
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            use = full = self._print(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use
```
### 39 - sympy/printing/str.py:

Start line: 463, End line: 529

```python
class StrPrinter(Printer):

    def _print_Subs(self, obj):
        expr, old, new = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return "Subs(%s, %s, %s)" % (
            self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        return expr._print()

    def _print_TensorHead(self, expr):
        return expr._print()

    def _print_Tensor(self, expr):
        return expr._print()

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "*".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        return expr._print()

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "%s[%s]" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True), ", ".join([self._print(i) for i in expr.indices]))

    def _print_PermutationGroup(self, expr):
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)

    def _print_Pi(self, expr):
        return 'pi'

    def _print_PolyRing(self, ring):
        return "Polynomial ring in %s over %s with %s order" % \
            (", ".join(map(lambda rs: self._print(rs), ring.symbols)),
            self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "Rational function field in %s over %s with %s order" % \
            (", ".join(map(lambda fs: self._print(fs), field.symbols)),
            self._print(field.domain), self._print(field.order))

    def _print_FreeGroupElement(self, elm):
        return elm.__str__()

    def _print_GaussianElement(self, poly):
        return "(%s + %s*I)" % (poly.x, poly.y)

    def _print_PolyElement(self, poly):
        return poly.str(self, PRECEDENCE, "%s**%s", "*")

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self.parenthesize(frac.numer, PRECEDENCE["Mul"], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"], strict=True)
            return numer + "/" + denom
```
### 51 - sympy/printing/str.py:

Start line: 52, End line: 71

```python
class StrPrinter(Printer):

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < PREC:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 53 - sympy/printing/str.py:

Start line: 248, End line: 263

```python
class StrPrinter(Printer):

    def _print_MatrixSlice(self, expr):
        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join(map(lambda arg: self._print(arg), x))
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + '[' +
                strslice(expr.rowslice, expr.parent.rows) + ', ' +
                strslice(expr.colslice, expr.parent.cols) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name
```
### 69 - sympy/printing/str.py:

Start line: 761, End line: 780

```python
class StrPrinter(Printer):

    def _print_Relational(self, expr):

        charmap = {
            "==": "Eq",
            "!=": "Ne",
            ":=": "Assignment",
            '+=': "AddAugmentedAssignment",
            "-=": "SubAugmentedAssignment",
            "*=": "MulAugmentedAssignment",
            "/=": "DivAugmentedAssignment",
            "%=": "ModAugmentedAssignment",
        }

        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs),
                                   self._print(expr.rhs))

        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))
```
### 71 - sympy/printing/str.py:

Start line: 73, End line: 165

```python
class StrPrinter(Printer):

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_Not(self, expr):
        return '~%s' %(self.parenthesize(expr.args[0],PRECEDENCE["Not"]))

    def _print_And(self, expr):
        args = list(expr.args)
        for j, i in enumerate(args):
            if isinstance(i, Relational) and (
                    i.canonical.rhs is S.NegativeInfinity):
                args.insert(0, args.pop(j))
        return self.stringify(args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_Or(self, expr):
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_Xor(self, expr):
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    def _print_AppliedPredicate(self, expr):
        return '%s(%s)' % (
            self._print(expr.function), self.stringify(expr.arguments, ", "))

    def _print_Basic(self, expr):
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        return 'zoo'

    def _print_ConditionSet(self, s):
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Derivative(%s)' % ", ".join(map(lambda arg: self._print(arg), [dexpr] + dvars))

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        return 'E'

    def _print_ExprCondPair(self, expr):
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'
```
### 86 - sympy/printing/str.py:

Start line: 808, End line: 910

```python
class StrPrinter(Printer):

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if not args:
            return "set()"
        return '{%s}' % args

    def _print_FiniteSet(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if any(item.has(FiniteSet) for item in items):
            return 'FiniteSet({})'.format(args)
        return '{{{}}}'.format(args)

    def _print_Partition(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(arg) for arg in items)
        return 'Partition({})'.format(args)

    def _print_frozenset(self, s):
        if not s:
            return "frozenset()"
        return "frozenset(%s)" % self._print_set(s)

    def _print_Sum(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        return "I"

    def _print_ZeroMatrix(self, expr):
        return "0"

    def _print_OneMatrix(self, expr):
        return "1"

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_str(self, expr):
        return str(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_Transpose(self, T):
        return "%s.T" % self.parenthesize(T.arg, PRECEDENCE["Pow"])

    def _print_Uniform(self, expr):
        return "Uniform(%s, %s)" % (self._print(expr.a), self._print(expr.b))

    def _print_Quantity(self, expr):
        if self._settings.get("abbrev", False):
            return "%s" % expr.abbrev
        return "%s" % expr.name

    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+"*"+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_Dimension(self, expr):
        return str(expr)

    def _print_Wild(self, expr):
        return expr.name + '_'

    def _print_WildFunction(self, expr):
        return expr.name + '_'

    def _print_WildDot(self, expr):
        return expr.name

    def _print_WildPlus(self, expr):
        return expr.name

    def _print_WildStar(self, expr):
        return expr.name

    def _print_Zero(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(0)"
        return "0"
```
### 90 - sympy/printing/str.py:

Start line: 190, End line: 207

```python
class StrPrinter(Printer):

    def _print_Interval(self, i):
        fin =  'Interval{m}({a}, {b})'
        a, b, l, r = i.args
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not r:
            m = ''
        elif b.is_infinite and not l:
            m = ''
        elif not l and not r:
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        return fin.format(**{'a': a, 'b': b, 'm': m})
```
### 95 - sympy/printing/str.py:

Start line: 21, End line: 50

```python
class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
        "min": None,
        "max": None,
    }  # type: tDict[str, Any]

    _relationals = dict()  # type: tDict[str, str]

    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)
```
### 97 - sympy/printing/str.py:

Start line: 912, End line: 976

```python
class StrPrinter(Printer):

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass

        cls = p.__class__.__name__
        rep = self._print(p.rep)
        dom = self._print(p.dom)
        ring = self._print(p.ring)

        return "%s(%s, %s, %s)" % (cls, rep, dom, ring)

    def _print_DMF(self, expr):
        return self._print_DMP(expr)

    def _print_Object(self, obj):
        return 'Object("%s")' % obj.name

    def _print_IdentityMorphism(self, morphism):
        return 'IdentityMorphism(%s)' % morphism.domain

    def _print_NamedMorphism(self, morphism):
        return 'NamedMorphism(%s, %s, "%s")' % \
               (morphism.domain, morphism.codomain, morphism.name)

    def _print_Category(self, category):
        return 'Category("%s")' % category.name

    def _print_Manifold(self, manifold):
        return manifold.name.name

    def _print_Patch(self, patch):
        return patch.name.name

    def _print_CoordSystem(self, coords):
        return coords.name.name

    def _print_BaseScalarField(self, field):
        return field._coord_sys.symbols[field._index].name

    def _print_BaseVectorField(self, field):
        return 'e_%s' % field._coord_sys.symbols[field._index].name

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys.symbols[field._index].name
        else:
            return 'd(%s)' % self._print(field)

    def _print_Tr(self, expr):
        #TODO : Handle indices
        return "%s(%s)" % ("Tr", self._print(expr.args[0]))

    def _print_Str(self, s):
        return self._print(s.name)

    def _print_AppliedBinaryRelation(self, expr):
        rel = expr.function
        return '%s(%s, %s)' % (self._print(rel),
                               self._print(expr.lhs),
                               self._print(expr.rhs))
```
### 132 - sympy/printing/str.py:

Start line: 1, End line: 17

```python
"""
A Printer for generating readable representation of most SymPy classes.
"""

from typing import Any, Dict as tDict

from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.core.relational import Relational
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.sets.sets import FiniteSet
from sympy.utilities.iterables import sift
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function

from mpmath.libmp import prec_to_dps, to_str as mlib_to_str
```
