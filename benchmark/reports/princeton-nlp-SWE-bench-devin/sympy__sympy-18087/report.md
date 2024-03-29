# sympy__sympy-18087

| **sympy/sympy** | `9da013ad0ddc3cd96fe505f2e47c63e372040916` |
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
diff --git a/sympy/core/exprtools.py b/sympy/core/exprtools.py
--- a/sympy/core/exprtools.py
+++ b/sympy/core/exprtools.py
@@ -358,8 +358,8 @@ def __init__(self, factors=None):  # Factors
             for f in list(factors.keys()):
                 if isinstance(f, Rational) and not isinstance(f, Integer):
                     p, q = Integer(f.p), Integer(f.q)
-                    factors[p] = (factors[p] if p in factors else 0) + factors[f]
-                    factors[q] = (factors[q] if q in factors else 0) - factors[f]
+                    factors[p] = (factors[p] if p in factors else S.Zero) + factors[f]
+                    factors[q] = (factors[q] if q in factors else S.Zero) - factors[f]
                     factors.pop(f)
             if i:
                 factors[I] = S.One*i
@@ -448,14 +448,12 @@ def as_expr(self):  # Factors
         args = []
         for factor, exp in self.factors.items():
             if exp != 1:
-                b, e = factor.as_base_exp()
-                if isinstance(exp, int):
-                    e = _keep_coeff(Integer(exp), e)
-                elif isinstance(exp, Rational):
+                if isinstance(exp, Integer):
+                    b, e = factor.as_base_exp()
                     e = _keep_coeff(exp, e)
+                    args.append(b**e)
                 else:
-                    e *= exp
-                args.append(b**e)
+                    args.append(factor**exp)
             else:
                 args.append(factor)
         return Mul(*args)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/exprtools.py | 361 | 362 | - | - | -
| sympy/core/exprtools.py | 451 | 458 | - | - | -


## Problem Statement

```
Simplify of simple trig expression fails
trigsimp in various versions, including 1.5, incorrectly simplifies cos(x)+sqrt(sin(x)**2) as though it were cos(x)+sin(x) for general complex x. (Oddly it gets this right if x is real.)

Embarrassingly I found this by accident while writing sympy-based teaching material...


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/simplify/trigsimp.py | 605 | 666| 585 | 585 | 12228 | 
| 2 | 1 sympy/simplify/trigsimp.py | 511 | 537| 217 | 802 | 12228 | 
| 3 | 1 sympy/simplify/trigsimp.py | 591 | 603| 134 | 936 | 12228 | 
| 4 | 2 sympy/functions/elementary/trigonometric.py | 821 | 870| 599 | 1535 | 38718 | 
| 5 | 2 sympy/simplify/trigsimp.py | 985 | 1065| 724 | 2259 | 38718 | 
| 6 | 2 sympy/simplify/trigsimp.py | 25 | 114| 876 | 3135 | 38718 | 
| 7 | 2 sympy/simplify/trigsimp.py | 795 | 835| 729 | 3864 | 38718 | 
| 8 | 2 sympy/simplify/trigsimp.py | 421 | 484| 496 | 4360 | 38718 | 
| 9 | 2 sympy/simplify/trigsimp.py | 539 | 590| 492 | 4852 | 38718 | 
| 10 | 2 sympy/simplify/trigsimp.py | 173 | 199| 455 | 5307 | 38718 | 
| 11 | 2 sympy/simplify/trigsimp.py | 717 | 741| 216 | 5523 | 38718 | 
| 12 | 2 sympy/functions/elementary/trigonometric.py | 752 | 781| 262 | 5785 | 38718 | 
| 13 | 2 sympy/simplify/trigsimp.py | 281 | 346| 800 | 6585 | 38718 | 
| 14 | 2 sympy/simplify/trigsimp.py | 115 | 172| 827 | 7412 | 38718 | 
| 15 | 2 sympy/functions/elementary/trigonometric.py | 734 | 750| 229 | 7641 | 38718 | 
| 16 | 2 sympy/functions/elementary/trigonometric.py | 906 | 928| 156 | 7797 | 38718 | 
| 17 | 2 sympy/simplify/trigsimp.py | 754 | 794| 748 | 8545 | 38718 | 
| 18 | 2 sympy/simplify/trigsimp.py | 667 | 703| 297 | 8842 | 38718 | 
| 19 | 2 sympy/functions/elementary/trigonometric.py | 581 | 665| 829 | 9671 | 38718 | 
| 20 | 3 sympy/simplify/fu.py | 1232 | 1289| 527 | 10198 | 57783 | 
| 21 | 3 sympy/simplify/fu.py | 529 | 547| 172 | 10370 | 57783 | 
| 22 | 3 sympy/simplify/fu.py | 550 | 568| 173 | 10543 | 57783 | 
| 23 | 3 sympy/simplify/trigsimp.py | 930 | 983| 443 | 10986 | 57783 | 
| 24 | 3 sympy/functions/elementary/trigonometric.py | 885 | 904| 217 | 11203 | 57783 | 
| 25 | 3 sympy/functions/elementary/trigonometric.py | 691 | 732| 406 | 11609 | 57783 | 
| 26 | 3 sympy/simplify/trigsimp.py | 499 | 508| 124 | 11733 | 57783 | 
| 27 | 3 sympy/functions/elementary/trigonometric.py | 466 | 492| 181 | 11914 | 57783 | 
| 28 | 3 sympy/functions/elementary/trigonometric.py | 374 | 434| 580 | 12494 | 57783 | 
| 29 | 3 sympy/functions/elementary/trigonometric.py | 260 | 352| 842 | 13336 | 57783 | 
| 30 | 3 sympy/simplify/fu.py | 284 | 310| 231 | 13567 | 57783 | 
| 31 | 3 sympy/functions/elementary/trigonometric.py | 783 | 819| 770 | 14337 | 57783 | 
| 32 | 3 sympy/functions/elementary/trigonometric.py | 872 | 883| 127 | 14464 | 57783 | 
| 33 | 3 sympy/functions/elementary/trigonometric.py | 436 | 464| 331 | 14795 | 57783 | 
| 34 | 3 sympy/simplify/trigsimp.py | 1 | 21| 251 | 15046 | 57783 | 
| 35 | 3 sympy/simplify/fu.py | 400 | 426| 248 | 15294 | 57783 | 
| 36 | 3 sympy/functions/elementary/trigonometric.py | 667 | 689| 192 | 15486 | 57783 | 
| 37 | 3 sympy/simplify/trigsimp.py | 486 | 497| 132 | 15618 | 57783 | 
| 38 | 3 sympy/simplify/trigsimp.py | 705 | 715| 127 | 15745 | 57783 | 
| 39 | 3 sympy/simplify/fu.py | 254 | 281| 180 | 15925 | 57783 | 
| 40 | 4 sympy/core/evalf.py | 766 | 829| 572 | 16497 | 71517 | 
| 41 | 4 sympy/simplify/fu.py | 1473 | 1499| 197 | 16694 | 71517 | 
| 42 | 4 sympy/simplify/trigsimp.py | 886 | 927| 475 | 17169 | 71517 | 
| 43 | 4 sympy/simplify/fu.py | 1352 | 1372| 226 | 17395 | 71517 | 
| 44 | 4 sympy/simplify/fu.py | 230 | 251| 152 | 17547 | 71517 | 
| 45 | 5 sympy/simplify/radsimp.py | 946 | 968| 278 | 17825 | 81563 | 
| 46 | 5 sympy/simplify/fu.py | 1 | 189| 2051 | 19876 | 81563 | 
| 47 | 5 sympy/functions/elementary/trigonometric.py | 548 | 580| 282 | 20158 | 81563 | 
| 48 | 6 sympy/simplify/simplify.py | 393 | 529| 1421 | 21579 | 98912 | 
| 49 | 6 sympy/simplify/fu.py | 571 | 592| 160 | 21739 | 98912 | 
| 50 | 6 sympy/simplify/fu.py | 2198 | 2218| 161 | 21900 | 98912 | 
| 51 | 6 sympy/functions/elementary/trigonometric.py | 354 | 372| 167 | 22067 | 98912 | 
| 52 | 6 sympy/simplify/trigsimp.py | 348 | 372| 305 | 22372 | 98912 | 
| 53 | 6 sympy/simplify/fu.py | 1502 | 1528| 197 | 22569 | 98912 | 
| 54 | 6 sympy/functions/elementary/trigonometric.py | 1187 | 1264| 641 | 23210 | 98912 | 
| 55 | 6 sympy/simplify/radsimp.py | 723 | 795| 753 | 23963 | 98912 | 
| 56 | 6 sympy/simplify/radsimp.py | 796 | 841| 630 | 24593 | 98912 | 
| 57 | 6 sympy/simplify/fu.py | 2155 | 2195| 329 | 24922 | 98912 | 
| 58 | 6 sympy/functions/elementary/trigonometric.py | 1135 | 1156| 258 | 25180 | 98912 | 
| 59 | 6 sympy/simplify/simplify.py | 1469 | 1502| 347 | 25527 | 98912 | 
| 60 | 6 sympy/functions/elementary/trigonometric.py | 1018 | 1117| 873 | 26400 | 98912 | 
| 61 | 6 sympy/simplify/fu.py | 670 | 691| 223 | 26623 | 98912 | 
| 62 | 6 sympy/simplify/trigsimp.py | 1113 | 1170| 702 | 27325 | 98912 | 
| 63 | 6 sympy/functions/elementary/trigonometric.py | 1446 | 1520| 727 | 28052 | 98912 | 
| 64 | 6 sympy/simplify/radsimp.py | 843 | 944| 995 | 29047 | 98912 | 
| 65 | 6 sympy/simplify/simplify.py | 1295 | 1322| 247 | 29294 | 98912 | 
| 66 | 7 sympy/functions/elementary/hyperbolic.py | 372 | 387| 173 | 29467 | 111703 | 
| 67 | 7 sympy/simplify/fu.py | 1968 | 2016| 509 | 29976 | 111703 | 
| 68 | 7 sympy/functions/elementary/trigonometric.py | 1158 | 1185| 265 | 30241 | 111703 | 
| 69 | 7 sympy/functions/elementary/hyperbolic.py | 353 | 370| 182 | 30423 | 111703 | 
| 70 | 7 sympy/functions/elementary/trigonometric.py | 1946 | 1955| 118 | 30541 | 111703 | 
| 71 | 7 sympy/functions/elementary/hyperbolic.py | 194 | 209| 171 | 30712 | 111703 | 
| 72 | 7 sympy/functions/elementary/trigonometric.py | 1320 | 1428| 889 | 31601 | 111703 | 
| 73 | 7 sympy/functions/elementary/trigonometric.py | 2152 | 2166| 154 | 31755 | 111703 | 
| 74 | 7 sympy/functions/elementary/hyperbolic.py | 389 | 413| 263 | 32018 | 111703 | 
| 75 | 7 sympy/functions/elementary/trigonometric.py | 1922 | 1944| 155 | 32173 | 111703 | 
| 76 | 7 sympy/functions/elementary/trigonometric.py | 2308 | 2324| 170 | 32343 | 111703 | 
| 77 | 8 sympy/integrals/trigonometry.py | 1 | 28| 253 | 32596 | 114889 | 
| 78 | 8 sympy/functions/elementary/trigonometric.py | 984 | 1017| 292 | 32888 | 114889 | 
| 79 | 8 sympy/functions/elementary/trigonometric.py | 3119 | 3158| 416 | 33304 | 114889 | 
| 80 | 8 sympy/functions/elementary/trigonometric.py | 2264 | 2306| 336 | 33640 | 114889 | 
| 81 | 8 sympy/functions/elementary/hyperbolic.py | 336 | 351| 130 | 33770 | 114889 | 
| 82 | 8 sympy/functions/elementary/trigonometric.py | 495 | 546| 321 | 34091 | 114889 | 
| 83 | 8 sympy/functions/elementary/trigonometric.py | 1522 | 1548| 271 | 34362 | 114889 | 
| 84 | 8 sympy/functions/elementary/trigonometric.py | 2326 | 2376| 480 | 34842 | 114889 | 
| 85 | 8 sympy/simplify/simplify.py | 618 | 717| 843 | 35685 | 114889 | 
| 86 | 8 sympy/simplify/trigsimp.py | 1066 | 1110| 336 | 36021 | 114889 | 
| 87 | 8 sympy/functions/elementary/trigonometric.py | 1866 | 1920| 335 | 36356 | 114889 | 
| 88 | 8 sympy/functions/elementary/trigonometric.py | 2667 | 2685| 165 | 36521 | 114889 | 
| 89 | 9 sympy/functions/special/error_functions.py | 1795 | 1899| 784 | 37305 | 134921 | 
| 90 | 9 sympy/functions/elementary/trigonometric.py | 55 | 66| 125 | 37430 | 134921 | 
| 91 | 9 sympy/simplify/radsimp.py | 667 | 720| 405 | 37835 | 134921 | 
| 92 | 9 sympy/functions/elementary/trigonometric.py | 1779 | 1849| 500 | 38335 | 134921 | 
| 93 | 9 sympy/functions/elementary/trigonometric.py | 2687 | 2693| 141 | 38476 | 134921 | 
| 94 | 9 sympy/functions/elementary/trigonometric.py | 2168 | 2203| 319 | 38795 | 134921 | 
| 95 | 9 sympy/simplify/trigsimp.py | 374 | 418| 618 | 39413 | 134921 | 
| 96 | 9 sympy/simplify/fu.py | 612 | 667| 532 | 39945 | 134921 | 
| 97 | 9 sympy/simplify/simplify.py | 1503 | 1518| 153 | 40098 | 134921 | 
| 98 | 9 sympy/functions/elementary/trigonometric.py | 1623 | 1689| 717 | 40815 | 134921 | 
| 99 | 9 sympy/simplify/fu.py | 834 | 892| 489 | 41304 | 134921 | 
| 100 | 9 sympy/simplify/simplify.py | 1213 | 1277| 688 | 41992 | 134921 | 
| 101 | 9 sympy/functions/elementary/trigonometric.py | 2895 | 2935| 312 | 42304 | 134921 | 
| 102 | 9 sympy/integrals/trigonometry.py | 292 | 332| 381 | 42685 | 134921 | 
| 103 | 9 sympy/functions/elementary/trigonometric.py | 1 | 20| 228 | 42913 | 134921 | 
| 104 | 9 sympy/functions/elementary/trigonometric.py | 2778 | 2811| 275 | 43188 | 134921 | 
| 105 | 9 sympy/functions/elementary/hyperbolic.py | 211 | 266| 460 | 43648 | 134921 | 
| 106 | 9 sympy/functions/elementary/trigonometric.py | 2496 | 2516| 195 | 43843 | 134921 | 
| 107 | 10 sympy/polys/numberfields.py | 420 | 449| 316 | 44159 | 144087 | 
| 108 | 10 sympy/simplify/simplify.py | 1939 | 1963| 238 | 44397 | 144087 | 
| 109 | 10 sympy/functions/elementary/trigonometric.py | 2098 | 2150| 388 | 44785 | 144087 | 
| 110 | 10 sympy/integrals/trigonometry.py | 247 | 289| 402 | 45187 | 144087 | 
| 111 | 10 sympy/simplify/fu.py | 428 | 439| 154 | 45341 | 144087 | 
| 112 | 10 sympy/simplify/fu.py | 764 | 780| 169 | 45510 | 144087 | 
| 113 | 10 sympy/functions/elementary/trigonometric.py | 1550 | 1572| 186 | 45696 | 144087 | 
| 114 | 10 sympy/simplify/simplify.py | 1279 | 1293| 181 | 45877 | 144087 | 
| 115 | 10 sympy/integrals/trigonometry.py | 137 | 244| 1168 | 47045 | 144087 | 
| 116 | 10 sympy/simplify/radsimp.py | 550 | 583| 288 | 47333 | 144087 | 
| 117 | 11 sympy/integrals/transforms.py | 877 | 918| 359 | 47692 | 160907 | 
| 118 | 11 sympy/functions/elementary/hyperbolic.py | 269 | 334| 413 | 48105 | 160907 | 
| 119 | 11 sympy/simplify/fu.py | 986 | 1021| 272 | 48377 | 160907 | 
| 120 | 11 sympy/simplify/fu.py | 1563 | 1589| 236 | 48613 | 160907 | 
| 121 | 11 sympy/functions/elementary/trigonometric.py | 203 | 258| 376 | 48989 | 160907 | 
| 122 | 11 sympy/polys/numberfields.py | 382 | 417| 404 | 49393 | 160907 | 
| 123 | 11 sympy/simplify/fu.py | 782 | 809| 209 | 49602 | 160907 | 
| 124 | 11 sympy/simplify/radsimp.py | 1086 | 1110| 119 | 49721 | 160907 | 
| 125 | 12 sympy/simplify/gammasimp.py | 84 | 111| 246 | 49967 | 165115 | 
| 126 | 13 sympy/simplify/hyperexpand.py | 300 | 345| 706 | 50673 | 189942 | 
| 127 | 13 sympy/functions/special/error_functions.py | 1671 | 1683| 199 | 50872 | 189942 | 
| 128 | 13 sympy/simplify/simplify.py | 1803 | 1937| 1346 | 52218 | 189942 | 
| 129 | 13 sympy/functions/elementary/trigonometric.py | 1119 | 1133| 131 | 52349 | 189942 | 
| 130 | 13 sympy/simplify/radsimp.py | 162 | 186| 176 | 52525 | 189942 | 
| 131 | 13 sympy/functions/elementary/trigonometric.py | 2518 | 2524| 132 | 52657 | 189942 | 
| 132 | 13 sympy/simplify/gammasimp.py | 383 | 471| 790 | 53447 | 189942 | 
| 133 | 13 sympy/functions/elementary/trigonometric.py | 1851 | 1863| 149 | 53596 | 189942 | 
| 134 | 13 sympy/functions/elementary/hyperbolic.py | 415 | 426| 128 | 53724 | 189942 | 
| 135 | 13 sympy/functions/elementary/hyperbolic.py | 101 | 151| 335 | 54059 | 189942 | 


## Missing Patch Files

 * 1: sympy/core/exprtools.py

### Hint

```
I guess you mean this:
\`\`\`julia
In [16]: cos(x) + sqrt(sin(x)**2)                                                                                                 
Out[16]: 
   _________         
  ╱    2             
╲╱  sin (x)  + cos(x)

In [17]: simplify(cos(x) + sqrt(sin(x)**2))                                                                                       
Out[17]: 
      ⎛    π⎞
√2⋅sin⎜x + ─⎟
      ⎝    4⎠
\`\`\`
Which is incorrect if `sin(x)` is negative:
\`\`\`julia
In [27]: (cos(x) + sqrt(sin(x)**2)).evalf(subs={x:-1})                                                                            
Out[27]: 1.38177329067604

In [28]: simplify(cos(x) + sqrt(sin(x)**2)).evalf(subs={x:-1})                                                                    
Out[28]: -0.301168678939757
\`\`\`
For real x this works because the sqrt auto simplifies to abs before simplify is called:
\`\`\`julia
In [18]: x = Symbol('x', real=True)                                                                                               

In [19]: simplify(cos(x) + sqrt(sin(x)**2))                                                                                       
Out[19]: cos(x) + │sin(x)│

In [20]: cos(x) + sqrt(sin(x)**2)                                                                                                 
Out[20]: cos(x) + │sin(x)│
\`\`\`
Yes, that's the issue I mean.
`fu` and `trigsimp` return the same erroneous simplification. All three simplification functions end up in Fu's `TR10i()` and this is what it returns:
\`\`\`
In [5]: from sympy.simplify.fu import *

In [6]: e = cos(x) + sqrt(sin(x)**2)

In [7]: TR10i(sqrt(sin(x)**2))
Out[7]: 
   _________
  ╱    2    
╲╱  sin (x) 

In [8]: TR10i(e)
Out[8]: 
      ⎛    π⎞
√2⋅sin⎜x + ─⎟
      ⎝    4⎠
\`\`\`
The other `TR*` functions keep the `sqrt` around, it's only `TR10i` that mishandles it. (Or it's called with an expression outside its scope of application...)
I tracked down where the invalid simplification of `sqrt(x**2)` takes place or at least I think so:
`TR10i` calls `trig_split` (also in fu.py) where the line
https://github.com/sympy/sympy/blob/0d99c52566820e9a5bb72eaec575fce7c0df4782/sympy/simplify/fu.py#L1901
in essence applies `._as_expr()` to `Factors({sin(x)**2: S.Half})` which then returns `sin(x)`.

If I understand `Factors` (sympy.core.exprtools) correctly, its intent is to have an efficient internal representation of products and `.as_expr()` is supposed to reconstruct a standard expression from such a representation. But here's what it does to a general complex variable `x`:
\`\`\`
In [21]: Factors(sqrt(x**2))
Out[21]: Factors({x**2: 1/2})
In [22]: _.as_expr()
Out[22]: x
\`\`\`
It seems line 455 below
https://github.com/sympy/sympy/blob/0d99c52566820e9a5bb72eaec575fce7c0df4782/sympy/core/exprtools.py#L449-L458
unconditionally multiplies exponents if a power of a power is encountered. However this is not generally valid for non-integer exponents...

And line 457 does the same for other non-integer exponents:
\`\`\`
In [23]: Factors((x**y)**z)
Out[23]: Factors({x**y: z})

In [24]: _.as_expr()
Out[24]:
 y⋅z
x
\`\`\`
```

## Patch

```diff
diff --git a/sympy/core/exprtools.py b/sympy/core/exprtools.py
--- a/sympy/core/exprtools.py
+++ b/sympy/core/exprtools.py
@@ -358,8 +358,8 @@ def __init__(self, factors=None):  # Factors
             for f in list(factors.keys()):
                 if isinstance(f, Rational) and not isinstance(f, Integer):
                     p, q = Integer(f.p), Integer(f.q)
-                    factors[p] = (factors[p] if p in factors else 0) + factors[f]
-                    factors[q] = (factors[q] if q in factors else 0) - factors[f]
+                    factors[p] = (factors[p] if p in factors else S.Zero) + factors[f]
+                    factors[q] = (factors[q] if q in factors else S.Zero) - factors[f]
                     factors.pop(f)
             if i:
                 factors[I] = S.One*i
@@ -448,14 +448,12 @@ def as_expr(self):  # Factors
         args = []
         for factor, exp in self.factors.items():
             if exp != 1:
-                b, e = factor.as_base_exp()
-                if isinstance(exp, int):
-                    e = _keep_coeff(Integer(exp), e)
-                elif isinstance(exp, Rational):
+                if isinstance(exp, Integer):
+                    b, e = factor.as_base_exp()
                     e = _keep_coeff(exp, e)
+                    args.append(b**e)
                 else:
-                    e *= exp
-                args.append(b**e)
+                    args.append(factor**exp)
             else:
                 args.append(factor)
         return Mul(*args)

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_exprtools.py b/sympy/core/tests/test_exprtools.py
--- a/sympy/core/tests/test_exprtools.py
+++ b/sympy/core/tests/test_exprtools.py
@@ -27,6 +27,8 @@ def test_Factors():
     assert Factors({x: 2, y: 3, sin(x): 4}).as_expr() == x**2*y**3*sin(x)**4
     assert Factors(S.Infinity) == Factors({oo: 1})
     assert Factors(S.NegativeInfinity) == Factors({oo: 1, -1: 1})
+    # issue #18059:
+    assert Factors((x**2)**S.Half).as_expr() == (x**2)**S.Half
 
     a = Factors({x: 5, y: 3, z: 7})
     b = Factors({      y: 4, z: 3, t: 10})
diff --git a/sympy/simplify/tests/test_fu.py b/sympy/simplify/tests/test_fu.py
--- a/sympy/simplify/tests/test_fu.py
+++ b/sympy/simplify/tests/test_fu.py
@@ -276,6 +276,9 @@ def test_fu():
     expr = Mul(*[cos(2**i) for i in range(10)])
     assert fu(expr) == sin(1024)/(1024*sin(1))
 
+    # issue #18059:
+    assert fu(cos(x) + sqrt(sin(x)**2)) == cos(x) + sqrt(sin(x)**2)
+
 
 def test_objective():
     assert fu(sin(x)/cos(x), measure=lambda x: x.count_ops()) == \

```


## Code snippets

### 1 - sympy/simplify/trigsimp.py:

Start line: 605, End line: 666

```python
#-------------------- the old trigsimp routines ---------------------

def trigsimp_old(expr, **opts):
    """
    reduces expression by using known trig identities

    Notes
    =====

    deep:
    - Apply trigsimp inside all objects with arguments

    recursive:
    - Use common subexpression elimination (cse()) and apply
    trigsimp recursively (this is quite expensive if the
    expression is large)

    method:
    - Determine the method to use. Valid choices are 'matching' (default),
    'groebner', 'combined', 'fu' and 'futrig'. If 'matching', simplify the
    expression recursively by pattern matching. If 'groebner', apply an
    experimental groebner basis algorithm. In this case further options
    are forwarded to ``trigsimp_groebner``, please refer to its docstring.
    If 'combined', first run the groebner basis algorithm with small
    default parameters, then run the 'matching' algorithm. 'fu' runs the
    collection of trigonometric transformations described by Fu, et al.
    (see the `fu` docstring) while `futrig` runs a subset of Fu-transforms
    that mimic the behavior of `trigsimp`.

    compare:
    - show input and output from `trigsimp` and `futrig` when different,
    but returns the `trigsimp` value.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log, cosh, sinh, tan, cot
    >>> from sympy.abc import x, y
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e, old=True)
    2
    >>> trigsimp(log(e), old=True)
    log(2*sin(x)**2 + 2*cos(x)**2)
    >>> trigsimp(log(e), deep=True, old=True)
    log(2)

    Using `method="groebner"` (or `"combined"`) can sometimes lead to a lot
    more simplification:

    >>> e = (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)
    >>> trigsimp(e, old=True)
    (1 - sin(x))/cos(x) + cos(x)/(1 - sin(x))
    >>> trigsimp(e, method="groebner", old=True)
    2/cos(x)

    >>> trigsimp(1/cot(x)**2, compare=True, old=True)
          futrig: tan(x)**2
    cot(x)**(-2)

    """
    old = expr
    first = opts.pop('first', True)
    # ... other code
```
### 2 - sympy/simplify/trigsimp.py:

Start line: 511, End line: 537

```python
def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i
    from sympy.simplify.simplify import bottom_up

    def exp_trig(e):
        # select the better of e, and e rewritten in terms of exp or trig
        # functions
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)
    # ... other code
```
### 3 - sympy/simplify/trigsimp.py:

Start line: 591, End line: 603

```python
def exptrigsimp(expr):
    # ... other code
    newexpr = bottom_up(newexpr, f)

    # sin/cos and sinh/cosh ratios to tan and tanh, respectively
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # can we ever generate an I where there was none previously?
    if not (newexpr.has(I) and not expr.has(I)):
        expr = newexpr
    return expr
```
### 4 - sympy/functions/elementary/trigonometric.py:

Start line: 821, End line: 870

```python
class cos(TrigonometricFunction):

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
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
            primes = []
            for p_i in cst_table_some:
                quotient, remainder = divmod(n, p_i)
                if remainder == 0:
                    n = quotient
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
### 5 - sympy/simplify/trigsimp.py:

Start line: 985, End line: 1065

```python
@cacheit
def __trigsimp(expr, deep=False):
    # ... other code

    if expr.is_Add:
        args = []
        for term in expr.args:
            if not term.is_commutative:
                com, nc = term.args_cnc()
                nc = Mul._from_args(nc)
                term = Mul._from_args(com)
            else:
                nc = S.One
            term = _trigsimp(term, deep)
            for pattern, result in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            args.append(term*nc)
        if args != expr.args:
            expr = Add(*args)
            expr = min(expr, expand(expr), key=count_ops)
        if expr.is_Add:
            for pattern, result in matchers_add:
                if not _dotrig(expr, pattern):
                    continue
                expr = TR10i(expr)
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    # if "d" contains any trig or hyperbolic funcs with
                    # argument "a" or "b" then skip the simplification;
                    # this isn't perfect -- see tests
                    if res is None or not (a in res and b in res) or any(
                        w.args[0] in (res[a], res[b]) for w in res[d].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    expr = result.subs(res)
                    break

        # Reduce any lingering artifacts, such as sin(x)**2 changing
        # to 1 - cos(x)**2 when sin(x)**2 was "simpler"
        for pattern, result, ex in artifacts:
            if not _dotrig(expr, pattern):
                continue
            # Substitute a new wild that excludes some function(s)
            # to help influence a better match. This is because
            # sometimes, for example, 'a' would match sec(x)**2
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)

            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                if m[a_t] == 0 or \
                        -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t]*m[d] + m[c] == 0:
                    break
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)

    elif expr.is_Mul or expr.is_Pow or deep and expr.args:
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])

    try:
        if not expr.has(*_trigs):
            raise TypeError
        e = expr.atoms(exp)
        new = expr.rewrite(exp, deep=deep)
        if new == e:
            raise TypeError
        fnew = factor(new)
        if fnew != new:
            new = sorted([new, factor(new)], key=count_ops)[0]
        # if all exp that were introduced disappeared then accept it
        if not (new.atoms(exp) - e):
            expr = new
    except TypeError:
        pass

    return expr
```
### 6 - sympy/simplify/trigsimp.py:

Start line: 25, End line: 114

```python
def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    """
    Simplify trigonometric expressions using a groebner basis algorithm.

    This routine takes a fraction involving trigonometric or hyperbolic
    expressions, and tries to simplify it. The primary metric is the
    total degree. Some attempts are made to choose the simplest possible
    expression of the minimal degree, but this is non-rigorous, and also
    very slow (see the ``quick=True`` option).

    If ``polynomial`` is set to True, instead of simplifying numerator and
    denominator together, this function just brings numerator and denominator
    into a canonical form. This is much faster, but has potentially worse
    results. However, if the input is a polynomial, then the result is
    guaranteed to be an equivalent polynomial of minimal degree.

    The most important option is hints. Its entries can be any of the
    following:

    - a natural number
    - a function
    - an iterable of the form (func, var1, var2, ...)
    - anything else, interpreted as a generator

    A number is used to indicate that the search space should be increased.
    A function is used to indicate that said function is likely to occur in a
    simplified expression.
    An iterable is used indicate that func(var1 + var2 + ...) is likely to
    occur in a simplified .
    An additional generator also indicates that it is likely to occur.
    (See examples below).

    This routine carries out various computationally intensive algorithms.
    The option ``quick=True`` can be used to suppress one particularly slow
    step (at the expense of potentially more complicated results, but never at
    the expense of increased total degree).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import sin, tan, cos, sinh, cosh, tanh
    >>> from sympy.simplify.trigsimp import trigsimp_groebner

    Suppose you want to simplify ``sin(x)*cos(x)``. Naively, nothing happens:

    >>> ex = sin(x)*cos(x)
    >>> trigsimp_groebner(ex)
    sin(x)*cos(x)

    This is because ``trigsimp_groebner`` only looks for a simplification
    involving just ``sin(x)`` and ``cos(x)``. You can tell it to also try
    ``2*x`` by passing ``hints=[2]``:

    >>> trigsimp_groebner(ex, hints=[2])
    sin(2*x)/2
    >>> trigsimp_groebner(sin(x)**2 - cos(x)**2, hints=[2])
    -cos(2*x)

    Increasing the search space this way can quickly become expensive. A much
    faster way is to give a specific expression that is likely to occur:

    >>> trigsimp_groebner(ex, hints=[sin(2*x)])
    sin(2*x)/2

    Hyperbolic expressions are similarly supported:

    >>> trigsimp_groebner(sinh(2*x)/sinh(x))
    2*cosh(x)

    Note how no hints had to be passed, since the expression already involved
    ``2*x``.

    The tangent function is also supported. You can either pass ``tan`` in the
    hints, to indicate that tan should be tried whenever cosine or sine are,
    or you can pass a specific generator:

    >>> trigsimp_groebner(sin(x)/cos(x), hints=[tan])
    tan(x)
    >>> trigsimp_groebner(sinh(x)/cosh(x), hints=[tanh(x)])
    tanh(x)

    Finally, you can use the iterable form to suggest that angle sum formulae
    should be tried:

    >>> ex = (tan(x) + tan(y))/(1 - tan(x)*tan(y))
    >>> trigsimp_groebner(ex, hints=[(tan, x, y)])
    tan(x + y)
    """
    # ... other code
```
### 7 - sympy/simplify/trigsimp.py:

Start line: 795, End line: 835

```python
def _trigpats():
    # ... other code
    matchers_identity = (
        (a*sin(b)**2, a - a*cos(b)**2),
        (a*tan(b)**2, a*(1/cos(b))**2 - a),
        (a*cot(b)**2, a*(1/sin(b))**2 - a),
        (a*sin(b + c), a*(sin(b)*cos(c) + sin(c)*cos(b))),
        (a*cos(b + c), a*(cos(b)*cos(c) - sin(b)*sin(c))),
        (a*tan(b + c), a*((tan(b) + tan(c))/(1 - tan(b)*tan(c)))),

        (a*sinh(b)**2, a*cosh(b)**2 - a),
        (a*tanh(b)**2, a - a*(1/cosh(b))**2),
        (a*coth(b)**2, a + a*(1/sinh(b))**2),
        (a*sinh(b + c), a*(sinh(b)*cosh(c) + sinh(c)*cosh(b))),
        (a*cosh(b + c), a*(cosh(b)*cosh(c) + sinh(b)*sinh(c))),
        (a*tanh(b + c), a*((tanh(b) + tanh(c))/(1 + tanh(b)*tanh(c)))),

    )

    # Reduce any lingering artifacts, such as sin(x)**2 changing
    # to 1-cos(x)**2 when sin(x)**2 was "simpler"
    artifacts = (
        (a - a*cos(b)**2 + c, a*sin(b)**2 + c, cos),
        (a - a*(1/cos(b))**2 + c, -a*tan(b)**2 + c, cos),
        (a - a*(1/sin(b))**2 + c, -a*cot(b)**2 + c, sin),

        (a - a*cosh(b)**2 + c, -a*sinh(b)**2 + c, cosh),
        (a - a*(1/cosh(b))**2 + c, a*tanh(b)**2 + c, cosh),
        (a + a*(1/sinh(b))**2 + c, a*coth(b)**2 + c, sinh),

        # same as above but with noncommutative prefactor
        (a*d - a*d*cos(b)**2 + c, a*d*sin(b)**2 + c, cos),
        (a*d - a*d*(1/cos(b))**2 + c, -a*d*tan(b)**2 + c, cos),
        (a*d - a*d*(1/sin(b))**2 + c, -a*d*cot(b)**2 + c, sin),

        (a*d - a*d*cosh(b)**2 + c, -a*d*sinh(b)**2 + c, cosh),
        (a*d - a*d*(1/cosh(b))**2 + c, a*d*tanh(b)**2 + c, cosh),
        (a*d + a*d*(1/sinh(b))**2 + c, a*d*coth(b)**2 + c, sinh),
    )

    _trigpat = (a, b, c, d, matchers_division, matchers_add,
        matchers_identity, artifacts)
    return _trigpat
```
### 8 - sympy/simplify/trigsimp.py:

Start line: 421, End line: 484

```python
_trigs = (TrigonometricFunction, HyperbolicFunction)


def trigsimp(expr, **opts):
    """
    reduces expression by using known trig identities

    Notes
    =====

    method:
    - Determine the method to use. Valid choices are 'matching' (default),
    'groebner', 'combined', and 'fu'. If 'matching', simplify the
    expression recursively by targeting common patterns. If 'groebner', apply
    an experimental groebner basis algorithm. In this case further options
    are forwarded to ``trigsimp_groebner``, please refer to its docstring.
    If 'combined', first run the groebner basis algorithm with small
    default parameters, then run the 'matching' algorithm. 'fu' runs the
    collection of trigonometric transformations described by Fu, et al.
    (see the `fu` docstring).


    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log
    >>> from sympy.abc import x, y
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e)
    2

    Simplification occurs wherever trigonometric functions are located.

    >>> trigsimp(log(e))
    log(2)

    Using `method="groebner"` (or `"combined"`) might lead to greater
    simplification.

    The old trigsimp routine can be accessed as with method 'old'.

    >>> from sympy import coth, tanh
    >>> t = 3*tanh(x)**7 - 2/coth(x)**7
    >>> trigsimp(t, method='old') == t
    True
    >>> trigsimp(t)
    tanh(x)**7

    """
    from sympy.simplify.fu import fu

    expr = sympify(expr)

    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        return _eval_trigsimp(**opts)

    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'
    # ... other code
```
### 9 - sympy/simplify/trigsimp.py:

Start line: 539, End line: 590

```python
def exptrigsimp(expr):
    # ... other code

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        # Since as_powers_dict loses order information,
        # if there is more than one noncommutative factor,
        # it should only be used to simplify the commutative part.
        if (len(noncommutative_part) > 1):
            return f(Mul(*commutative_part))*Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=1):
            if expr is S.Exp1:
                return sign, 1
            elif isinstance(expr, exp):
                return sign, expr.args[0]
            elif sign == 1:
                return signlog(-expr, sign=-1)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1]/c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x*m/2:
                    # sinh and cosh
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2*c*cosh(x/2)] += m
                    else:
                        newd[-2*c*sinh(x/2)] += m
                elif newd[1 - sign*S.Exp1**x] == -m:
                    # tanh
                    del newd[1 - sign*S.Exp1**x]
                    if sign == 1:
                        newd[-c/tanh(x/2)] += m
                    else:
                        newd[-c*tanh(x/2)] += m
                else:
                    newd[1 + sign*S.Exp1**x] += m
                    newd[c] += m

        return Mul(*[k**newd[k] for k in newd])
    # ... other code
```
### 10 - sympy/simplify/trigsimp.py:

Start line: 173, End line: 199

```python
def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    # or tan(n*x), with n an integer. Suppose first there are no tan terms.
    # The ideal [sin(x)**2 + cos(x)**2 - 1] is geometrically prime, since
    # X**2 + Y**2 - 1 is irreducible over CC.
    # Now, if we have a generator sin(n*x), than we can, using trig identities,
    # express sin(n*x) as a polynomial in sin(x) and cos(x). We can add this
    # relation to the ideal, preserving geometric primality, since the quotient
    # ring is unchanged.
    # Thus we have treated all sin and cos terms.
    # For tan(n*x), we add a relation tan(n*x)*cos(n*x) - sin(n*x) = 0.
    # (This requires of course that we already have relations for cos(n*x) and
    # sin(n*x).) It is not obvious, but it seems that this preserves geometric
    # primality.
    # XXX A real proof would be nice. HELP!
    #     Sketch that <S**2 + C**2 - 1, C*T - S> is a prime ideal of
    #     CC[S, C, T]:
    #     - it suffices to show that the projective closure in CP**3 is
    #       irreducible
    #     - using the half-angle substitutions, we can express sin(x), tan(x),
    #       cos(x) as rational functions in tan(x/2)
    #     - from this, we get a rational map from CP**1 to our curve
    #     - this is a morphism, hence the curve is prime
    #
    # Step (2) is trivial.
    #
    # Step (3) works by adding selected relations of the form
    # sin(x + y) - sin(x)*cos(y) - sin(y)*cos(x), etc. Geometric primality is
    # preserved by the same argument as before.
    # ... other code
```
