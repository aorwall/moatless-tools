# sympy__sympy-18200

| **sympy/sympy** | `c559a8421ac4865ebfe66024be6cd43a6103a62b` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 1074 |
| **Any found context length** | 1074 |
| **Avg pos** | 0.5 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/sets/handlers/intersection.py b/sympy/sets/handlers/intersection.py
--- a/sympy/sets/handlers/intersection.py
+++ b/sympy/sets/handlers/intersection.py
@@ -235,26 +235,46 @@ def intersection_sets(self, other): # noqa:F811
     # diophantine equations f(n)=g(m).
     # If the solutions for n are {h(t) : t in Integers} then we return
     # {f(h(t)) : t in integers}.
+    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
+    # {f(n_i) : 1 <= i <= k}.
     if base_set is S.Integers:
         gm = None
         if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
             gm = other.lamda.expr
-            m = other.lamda.variables[0]
+            var = other.lamda.variables[0]
+            # Symbol of second ImageSet lambda must be distinct from first
+            m = Dummy('m')
+            gm = gm.subs(var, m)
         elif other is S.Integers:
-            m = gm = Dummy('x')
+            m = gm = Dummy('m')
         if gm is not None:
             fn = self.lamda.expr
             n = self.lamda.variables[0]
-            solns = list(diophantine(fn - gm, syms=(n, m)))
+            try:
+                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
+            except (TypeError, NotImplementedError):
+                # TypeError if equation not polynomial with rational coeff.
+                # NotImplementedError if correct format but no solver.
+                return
+            # 3 cases are possible for solns:
+            # - empty set,
+            # - one or more parametric (infinite) solutions,
+            # - a finite number of (non-parametric) solution couples.
+            # Among those, there is one type of solution set that is
+            # not helpful here: multiple parametric solutions.
             if len(solns) == 0:
                 return EmptySet
-            elif len(solns) != 1:
-                return
+            elif any(not isinstance(s, int) and s.free_symbols
+                     for tupl in solns for s in tupl):
+                if len(solns) == 1:
+                    soln, solm = solns[0]
+                    (t,) = soln.free_symbols
+                    expr = fn.subs(n, soln.subs(t, n)).expand()
+                    return imageset(Lambda(n, expr), S.Integers)
+                else:
+                    return
             else:
-                soln, solm = solns[0]
-                (t,) = soln.free_symbols
-                expr = fn.subs(n, soln.subs(t, n))
-                return imageset(Lambda(n, expr), S.Integers)
+                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))
 
     if other == S.Reals:
         from sympy.solvers.solveset import solveset_real
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -294,7 +294,10 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
         else:
             raise TypeError
     except (TypeError, NotImplementedError):
-        terms = factor_list(eq)[1]
+        fl = factor_list(eq)
+        if fl[0].is_Rational and fl[0] != 1:
+            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
+        terms = fl[1]
 
     sols = set([])
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/handlers/intersection.py | 238 | 253 | 1 | 1 | 1074
| sympy/solvers/diophantine.py | 297 | 297 | - | 15 | -


## Problem Statement

```
ImageSet(Lambda(n, n**2), S.Integers).intersect(S.Integers) raises AttributeError
\`\`\`
In [3]: ImageSet(Lambda(n, n**2), S.Integers).intersect(S.Integers)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-3-90c3407ef4ee> in <module>()
----> 1 ImageSet(Lambda(n, n**2), S.Integers).intersect(S.Integers)
  
/root/sympy/sympy/sets/sets.py in intersect(self, other)
    125
    126         """
--> 127         return Intersection(self, other)
    128
    129     def intersection(self, other):

/root/sympy/sympy/sets/sets.py in __new__(cls, *args, **kwargs)
   1339         if evaluate:
   1340             args = list(cls._new_args_filter(args))
-> 1341             return simplify_intersection(args)
   1342
   1343         args = list(ordered(args, Set._infimum_key))

/root/sympy/sympy/sets/sets.py in simplify_intersection(args)
   2260             new_args = False
   2261             for t in args - set((s,)):
-> 2262                 new_set = intersection_sets(s, t)
   2263                 # This returns None if s does not know how to intersect
   2264                 # with t. Returns the newly intersected set otherwise

/root/sympy/sympy/multipledispatch/dispatcher.py in __call__(self, *args, **kwargs)
    196             self._cache[types] = func
    197         try:
--> 198             return func(*args, **kwargs)
    199
    200         except MDNotImplementedError:

/root/sympy/sympy/sets/handlers/intersection.py in intersection_sets(self, other)
    256             else:
    257                 soln, solm = solns[0]
--> 258                 (t,) = soln.free_symbols
    259                 expr = fn.subs(n, soln.subs(t, n))
    260                 return imageset(Lambda(n, expr), S.Integers)

AttributeError: 'int' object has no attribute 'free_symbols'
\`\`\`

This is in the `diophantine` related intersection code. See also: #17568, #18081
and https://github.com/sympy/sympy/issues/9616#issuecomment-568465831

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/sets/handlers/intersection.py** | 223 | 348| 1074 | 1074 | 3707 | 
| 2 | **1 sympy/sets/handlers/intersection.py** | 30 | 75| 407 | 1481 | 3707 | 
| 3 | **1 sympy/sets/handlers/intersection.py** | 1 | 28| 248 | 1729 | 3707 | 
| 4 | 2 sympy/sets/sets.py | 1389 | 1425| 233 | 1962 | 20244 | 
| 5 | 3 sympy/sets/fancysets.py | 330 | 357| 242 | 2204 | 31050 | 
| 6 | 3 sympy/sets/fancysets.py | 431 | 481| 444 | 2648 | 31050 | 
| 7 | 3 sympy/sets/sets.py | 1427 | 1529| 968 | 3616 | 31050 | 
| 8 | 3 sympy/sets/fancysets.py | 273 | 329| 459 | 4075 | 31050 | 
| 9 | 3 sympy/sets/sets.py | 1347 | 1362| 128 | 4203 | 31050 | 
| 10 | 3 sympy/sets/sets.py | 2119 | 2150| 255 | 4458 | 31050 | 
| 11 | **3 sympy/sets/handlers/intersection.py** | 405 | 457| 397 | 4855 | 31050 | 
| 12 | 3 sympy/sets/fancysets.py | 395 | 429| 279 | 5134 | 31050 | 
| 13 | 3 sympy/sets/sets.py | 114 | 130| 146 | 5280 | 31050 | 
| 14 | 3 sympy/sets/sets.py | 1364 | 1387| 134 | 5414 | 31050 | 
| 15 | 3 sympy/sets/sets.py | 132 | 156| 160 | 5574 | 31050 | 
| 16 | 3 sympy/sets/fancysets.py | 359 | 393| 320 | 5894 | 31050 | 
| 17 | 3 sympy/sets/sets.py | 2026 | 2117| 787 | 6681 | 31050 | 
| 18 | 3 sympy/sets/sets.py | 1311 | 1345| 176 | 6857 | 31050 | 
| 19 | **3 sympy/sets/handlers/intersection.py** | 77 | 103| 220 | 7077 | 31050 | 
| 20 | 4 sympy/sets/__init__.py | 1 | 34| 261 | 7338 | 31311 | 
| 21 | 4 sympy/sets/sets.py | 181 | 217| 370 | 7708 | 31311 | 
| 22 | **4 sympy/sets/handlers/intersection.py** | 351 | 403| 397 | 8105 | 31311 | 
| 23 | 4 sympy/sets/fancysets.py | 483 | 496| 146 | 8251 | 31311 | 
| 24 | **4 sympy/sets/handlers/intersection.py** | 105 | 220| 962 | 9213 | 31311 | 
| 25 | 4 sympy/sets/sets.py | 2225 | 2298| 510 | 9723 | 31311 | 
| 26 | 4 sympy/sets/sets.py | 158 | 179| 159 | 9882 | 31311 | 
| 27 | 5 sympy/sets/handlers/functions.py | 25 | 112| 759 | 10641 | 33424 | 
| 28 | 6 sympy/solvers/solveset.py | 2770 | 2817| 469 | 11110 | 64098 | 
| 29 | 6 sympy/sets/sets.py | 1222 | 1264| 414 | 11524 | 64098 | 
| 30 | 6 sympy/sets/handlers/functions.py | 114 | 141| 283 | 11807 | 64098 | 
| 31 | 6 sympy/sets/sets.py | 1807 | 1844| 337 | 12144 | 64098 | 
| 32 | 6 sympy/solvers/solveset.py | 2023 | 2078| 493 | 12637 | 64098 | 
| 33 | 7 sympy/sets/handlers/union.py | 1 | 39| 282 | 12919 | 65290 | 
| 34 | 7 sympy/sets/sets.py | 628 | 672| 356 | 13275 | 65290 | 
| 35 | 7 sympy/sets/sets.py | 243 | 286| 246 | 13521 | 65290 | 
| 36 | 7 sympy/sets/handlers/functions.py | 1 | 23| 198 | 13719 | 65290 | 
| 37 | 7 sympy/solvers/solveset.py | 2743 | 2768| 273 | 13992 | 65290 | 
| 38 | 7 sympy/sets/sets.py | 2312 | 2331| 204 | 14196 | 65290 | 
| 39 | 7 sympy/sets/sets.py | 1191 | 1220| 225 | 14421 | 65290 | 
| 40 | 7 sympy/sets/sets.py | 904 | 943| 372 | 14793 | 65290 | 
| 41 | 7 sympy/sets/handlers/functions.py | 168 | 215| 428 | 15221 | 65290 | 
| 42 | 7 sympy/sets/sets.py | 1064 | 1077| 122 | 15343 | 65290 | 
| 43 | 7 sympy/sets/sets.py | 1174 | 1189| 128 | 15471 | 65290 | 
| 44 | 7 sympy/solvers/solveset.py | 1287 | 1301| 166 | 15637 | 65290 | 
| 45 | 7 sympy/solvers/solveset.py | 907 | 989| 861 | 16498 | 65290 | 
| 46 | 8 sympy/geometry/entity.py | 565 | 589| 203 | 16701 | 70472 | 
| 47 | 9 sympy/sets/setexpr.py | 1 | 97| 832 | 17533 | 71304 | 
| 48 | 9 sympy/sets/sets.py | 1785 | 1805| 181 | 17714 | 71304 | 
| 49 | 9 sympy/sets/sets.py | 1958 | 2023| 388 | 18102 | 71304 | 
| 50 | 9 sympy/solvers/solveset.py | 3362 | 3419| 494 | 18596 | 71304 | 
| 51 | 9 sympy/sets/fancysets.py | 1 | 20| 186 | 18782 | 71304 | 
| 52 | 9 sympy/sets/sets.py | 1294 | 1308| 126 | 18908 | 71304 | 
| 53 | 9 sympy/solvers/solveset.py | 2843 | 2866| 221 | 19129 | 71304 | 
| 54 | 9 sympy/sets/sets.py | 1266 | 1281| 142 | 19271 | 71304 | 
| 55 | 9 sympy/solvers/solveset.py | 1954 | 2020| 606 | 19877 | 71304 | 
| 56 | 10 sympy/solvers/solvers.py | 973 | 1064| 840 | 20717 | 103861 | 
| 57 | 10 sympy/solvers/solveset.py | 2502 | 2598| 849 | 21566 | 103861 | 
| 58 | 10 sympy/sets/sets.py | 1933 | 1955| 206 | 21772 | 103861 | 
| 59 | 10 sympy/solvers/solveset.py | 269 | 289| 241 | 22013 | 103861 | 
| 60 | 10 sympy/sets/fancysets.py | 584 | 646| 533 | 22546 | 103861 | 
| 61 | 10 sympy/sets/sets.py | 1137 | 1172| 201 | 22747 | 103861 | 
| 62 | 11 sympy/sets/handlers/power.py | 1 | 25| 198 | 22945 | 104697 | 
| 63 | 11 sympy/sets/sets.py | 391 | 423| 234 | 23179 | 104697 | 
| 64 | 11 sympy/sets/fancysets.py | 163 | 231| 397 | 23576 | 104697 | 
| 65 | 11 sympy/sets/fancysets.py | 698 | 736| 283 | 23859 | 104697 | 
| 66 | 11 sympy/solvers/solveset.py | 3207 | 3361| 1926 | 25785 | 104697 | 
| 67 | 12 sympy/abc.py | 72 | 112| 432 | 26217 | 105850 | 
| 68 | 12 sympy/sets/sets.py | 1 | 37| 334 | 26551 | 105850 | 
| 69 | 13 sympy/calculus/util.py | 1537 | 1580| 322 | 26873 | 117900 | 
| 70 | 13 sympy/solvers/solveset.py | 2818 | 2842| 290 | 27163 | 117900 | 
| 71 | 14 sympy/integrals/integrals.py | 112 | 145| 347 | 27510 | 132226 | 
| 72 | 14 sympy/solvers/solveset.py | 1145 | 1209| 659 | 28169 | 132226 | 
| 73 | 14 sympy/sets/handlers/union.py | 86 | 108| 214 | 28383 | 132226 | 
| 74 | 14 sympy/solvers/solveset.py | 3132 | 3157| 202 | 28585 | 132226 | 
| 75 | 14 sympy/sets/handlers/functions.py | 143 | 166| 198 | 28783 | 132226 | 
| 76 | **15 sympy/solvers/diophantine.py** | 1 | 98| 743 | 29526 | 163768 | 
| 77 | 15 sympy/sets/fancysets.py | 670 | 696| 206 | 29732 | 163768 | 
| 78 | 15 sympy/solvers/solveset.py | 991 | 1016| 233 | 29965 | 163768 | 
| 79 | 15 sympy/integrals/integrals.py | 1243 | 1277| 418 | 30383 | 163768 | 
| 80 | 15 sympy/solvers/solveset.py | 3071 | 3131| 585 | 30968 | 163768 | 
| 81 | 15 sympy/sets/sets.py | 40 | 85| 329 | 31297 | 163768 | 
| 82 | 16 sympy/sets/handlers/issubset.py | 1 | 32| 318 | 31615 | 164945 | 
| 83 | 17 sympy/integrals/intpoly.py | 1092 | 1148| 689 | 32304 | 176769 | 
| 84 | 17 sympy/sets/sets.py | 793 | 805| 147 | 32451 | 176769 | 
| 85 | 17 sympy/solvers/solveset.py | 2349 | 2501| 1516 | 33967 | 176769 | 
| 86 | 17 sympy/sets/sets.py | 611 | 626| 119 | 34086 | 176769 | 
| 87 | 17 sympy/solvers/solveset.py | 1856 | 1953| 767 | 34853 | 176769 | 
| 88 | 18 sympy/geometry/ellipse.py | 668 | 691| 244 | 35097 | 189871 | 
| 89 | 19 sympy/sets/handlers/mul.py | 1 | 19| 117 | 35214 | 190407 | 
| 90 | 19 sympy/solvers/solveset.py | 3174 | 3418| 236 | 35450 | 190407 | 
| 91 | 19 sympy/sets/sets.py | 721 | 743| 206 | 35656 | 190407 | 
| 92 | 19 sympy/sets/sets.py | 332 | 340| 120 | 35776 | 190407 | 
| 93 | 20 sympy/sets/conditionset.py | 1 | 18| 154 | 35930 | 192658 | 
| 94 | 20 sympy/sets/sets.py | 219 | 241| 177 | 36107 | 192658 | 
| 95 | 21 sympy/sets/handlers/add.py | 1 | 72| 521 | 36628 | 193179 | 
| 96 | **21 sympy/solvers/diophantine.py** | 168 | 211| 420 | 37048 | 193179 | 
| 97 | 21 sympy/sets/sets.py | 2301 | 2310| 131 | 37179 | 193179 | 


### Hint

```
The call to `diophantine` is `diophantine(n**2 - _x, syms=(n, _x))` which returns `[(0, 0)]` where the `0`s are plain `int`s.

So:
1. The code in lines 258-260 seems to assume that a single solution tuple will only ever arise in parametrized solutions. This isn't the case here.
2. `(0, 0)` isn't the only integer solution to the equation, so either I don't understand what `diophantine` is supposed to return (partial results?) or it's broken for quadratic equations. (See also #18114.)

Hmm, for point 2 above I just noticed that `diophantine` depends on the ~~assumptions on the variables. So for a plain `isympy` session with `m` and `n` integers and `x` and `y` generic symbols we have~~ alphabetic order of the variables.
\`\`\`
In [1]: diophantine(m**2 - n)
Out[1]:
⎧⎛    2⎞⎫
⎨⎝t, t ⎠⎬
⎩       ⎭

In [2]: diophantine(x**2 - n)
Out[2]: {(0, 0)}

In [3]: diophantine(m**2 - y)
Out[3]:
⎧⎛    2⎞⎫
⎨⎝t, t ⎠⎬
⎩       ⎭

In [4]: diophantine(x**2 - y)
Out[4]:
⎧⎛    2⎞⎫
⎨⎝t, t ⎠⎬
⎩       ⎭
\`\`\`
I filed #18122 for the `diophantine` issue. This one's then only about the uncaught exception in the sets code.
```

## Patch

```diff
diff --git a/sympy/sets/handlers/intersection.py b/sympy/sets/handlers/intersection.py
--- a/sympy/sets/handlers/intersection.py
+++ b/sympy/sets/handlers/intersection.py
@@ -235,26 +235,46 @@ def intersection_sets(self, other): # noqa:F811
     # diophantine equations f(n)=g(m).
     # If the solutions for n are {h(t) : t in Integers} then we return
     # {f(h(t)) : t in integers}.
+    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
+    # {f(n_i) : 1 <= i <= k}.
     if base_set is S.Integers:
         gm = None
         if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
             gm = other.lamda.expr
-            m = other.lamda.variables[0]
+            var = other.lamda.variables[0]
+            # Symbol of second ImageSet lambda must be distinct from first
+            m = Dummy('m')
+            gm = gm.subs(var, m)
         elif other is S.Integers:
-            m = gm = Dummy('x')
+            m = gm = Dummy('m')
         if gm is not None:
             fn = self.lamda.expr
             n = self.lamda.variables[0]
-            solns = list(diophantine(fn - gm, syms=(n, m)))
+            try:
+                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
+            except (TypeError, NotImplementedError):
+                # TypeError if equation not polynomial with rational coeff.
+                # NotImplementedError if correct format but no solver.
+                return
+            # 3 cases are possible for solns:
+            # - empty set,
+            # - one or more parametric (infinite) solutions,
+            # - a finite number of (non-parametric) solution couples.
+            # Among those, there is one type of solution set that is
+            # not helpful here: multiple parametric solutions.
             if len(solns) == 0:
                 return EmptySet
-            elif len(solns) != 1:
-                return
+            elif any(not isinstance(s, int) and s.free_symbols
+                     for tupl in solns for s in tupl):
+                if len(solns) == 1:
+                    soln, solm = solns[0]
+                    (t,) = soln.free_symbols
+                    expr = fn.subs(n, soln.subs(t, n)).expand()
+                    return imageset(Lambda(n, expr), S.Integers)
+                else:
+                    return
             else:
-                soln, solm = solns[0]
-                (t,) = soln.free_symbols
-                expr = fn.subs(n, soln.subs(t, n))
-                return imageset(Lambda(n, expr), S.Integers)
+                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))
 
     if other == S.Reals:
         from sympy.solvers.solveset import solveset_real
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -294,7 +294,10 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
         else:
             raise TypeError
     except (TypeError, NotImplementedError):
-        terms = factor_list(eq)[1]
+        fl = factor_list(eq)
+        if fl[0].is_Rational and fl[0] != 1:
+            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
+        terms = fl[1]
 
     sols = set([])
 

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_fancysets.py b/sympy/sets/tests/test_fancysets.py
--- a/sympy/sets/tests/test_fancysets.py
+++ b/sympy/sets/tests/test_fancysets.py
@@ -618,6 +618,49 @@ def test_imageset_intersect_interval():
     assert f9.intersect(Interval(1, 2)) == Intersection(f9, Interval(1, 2))
 
 
+def test_imageset_intersect_diophantine():
+    from sympy.abc import m, n
+    # Check that same lambda variable for both ImageSets is handled correctly
+    img1 = ImageSet(Lambda(n, 2*n + 1), S.Integers)
+    img2 = ImageSet(Lambda(n, 4*n + 1), S.Integers)
+    assert img1.intersect(img2) == img2
+    # Empty solution set returned by diophantine:
+    assert ImageSet(Lambda(n, 2*n), S.Integers).intersect(
+            ImageSet(Lambda(n, 2*n + 1), S.Integers)) == S.EmptySet
+    # Check intersection with S.Integers:
+    assert ImageSet(Lambda(n, 9/n + 20*n/3), S.Integers).intersect(
+            S.Integers) == FiniteSet(-61, -23, 23, 61)
+    # Single solution (2, 3) for diophantine solution:
+    assert ImageSet(Lambda(n, (n - 2)**2), S.Integers).intersect(
+            ImageSet(Lambda(n, -(n - 3)**2), S.Integers)) == FiniteSet(0)
+    # Single parametric solution for diophantine solution:
+    assert ImageSet(Lambda(n, n**2 + 5), S.Integers).intersect(
+            ImageSet(Lambda(m, 2*m), S.Integers)) == ImageSet(
+            Lambda(n, 4*n**2 + 4*n + 6), S.Integers)
+    # 4 non-parametric solution couples for dioph. equation:
+    assert ImageSet(Lambda(n, n**2 - 9), S.Integers).intersect(
+            ImageSet(Lambda(m, -m**2), S.Integers)) == FiniteSet(-9, 0)
+    # Double parametric solution for diophantine solution:
+    assert ImageSet(Lambda(m, m**2 + 40), S.Integers).intersect(
+            ImageSet(Lambda(n, 41*n), S.Integers)) == Intersection(
+            ImageSet(Lambda(m, m**2 + 40), S.Integers),
+            ImageSet(Lambda(n, 41*n), S.Integers))
+    # Check that diophantine returns *all* (8) solutions (permute=True)
+    assert ImageSet(Lambda(n, n**4 - 2**4), S.Integers).intersect(
+            ImageSet(Lambda(m, -m**4 + 3**4), S.Integers)) == FiniteSet(0, 65)
+    assert ImageSet(Lambda(n, pi/12 + n*5*pi/12), S.Integers).intersect(
+            ImageSet(Lambda(n, 7*pi/12 + n*11*pi/12), S.Integers)) == ImageSet(
+            Lambda(n, 55*pi*n/12 + 17*pi/4), S.Integers)
+    # TypeError raised by diophantine (#18081)
+    assert ImageSet(Lambda(n, n*log(2)), S.Integers).intersection(S.Integers) \
+            == Intersection(ImageSet(Lambda(n, n*log(2)), S.Integers), S.Integers)
+    # NotImplementedError raised by diophantine (no solver for cubic_thue)
+    assert ImageSet(Lambda(n, n**3 + 1), S.Integers).intersect(
+            ImageSet(Lambda(n, n**3), S.Integers)) == Intersection(
+            ImageSet(Lambda(n, n**3 + 1), S.Integers),
+            ImageSet(Lambda(n, n**3), S.Integers))
+
+
 def test_infinitely_indexed_set_3():
     from sympy.abc import n, m, t
     assert imageset(Lambda(m, 2*pi*m), S.Integers).intersect(
@@ -656,7 +699,6 @@ def test_ImageSet_contains():
 
 
 def test_ComplexRegion_contains():
-
     # contains in ComplexRegion
     a = Interval(2, 3)
     b = Interval(4, 6)
@@ -687,7 +729,6 @@ def test_ComplexRegion_contains():
 
 
 def test_ComplexRegion_intersect():
-
     # Polar form
     X_axis = ComplexRegion(Interval(0, oo)*FiniteSet(0, S.Pi), polar=True)
 
@@ -735,7 +776,6 @@ def test_ComplexRegion_intersect():
 
 
 def test_ComplexRegion_union():
-
     # Polar form
     c1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
     c2 = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
@@ -782,7 +822,6 @@ def test_ComplexRegion_measure():
 
 
 def test_normalize_theta_set():
-
     # Interval
     assert normalize_theta_set(Interval(pi, 2*pi)) == \
         Union(FiniteSet(0), Interval.Ropen(pi, 2*pi))
diff --git a/sympy/solvers/tests/test_diophantine.py b/sympy/solvers/tests/test_diophantine.py
--- a/sympy/solvers/tests/test_diophantine.py
+++ b/sympy/solvers/tests/test_diophantine.py
@@ -487,12 +487,15 @@ def test_diophantine():
     assert check_solutions((x**2 - 3*y**2 - 1)*(y - 7*z))
     assert check_solutions((x**2 + y**2 - z**2)*(x - 7*y - 3*z + 4*w))
     # Following test case caused problems in parametric representation
-    # But this can be solved by factroing out y.
+    # But this can be solved by factoring out y.
     # No need to use methods for ternary quadratic equations.
     assert check_solutions(y**2 - 7*x*y + 4*y*z)
     assert check_solutions(x**2 - 2*x + 1)
 
     assert diophantine(x - y) == diophantine(Eq(x, y))
+    # 18196
+    eq = x**4 + y**4 - 97
+    assert diophantine(eq, permute=True) == diophantine(-eq, permute=True)
     assert diophantine(3*x*pi - 2*y*pi) == set([(2*t_0, 3*t_0)])
     eq = x**2 + y**2 + z**2 - 14
     base_sol = set([(1, 2, 3)])

```


## Code snippets

### 1 - sympy/sets/handlers/intersection.py:

Start line: 223, End line: 348

```python
@dispatch(ImageSet, Set)
def intersection_sets(self, other): # noqa:F811
    from sympy.solvers.diophantine import diophantine

    # Only handle the straight-forward univariate case
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # Intersection between ImageSets with Integers as base set
    # For {f(n) : n in Integers} & {g(m) : m in Integers} we solve the
    # diophantine equations f(n)=g(m).
    # If the solutions for n are {h(t) : t in Integers} then we return
    # {f(h(t)) : t in integers}.
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            m = other.lamda.variables[0]
        elif other is S.Integers:
            m = gm = Dummy('x')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            solns = list(diophantine(fn - gm, syms=(n, m)))
            if len(solns) == 0:
                return EmptySet
            elif len(solns) != 1:
                return
            else:
                soln, solm = solns[0]
                (t,) = soln.free_symbols
                expr = fn.subs(n, soln.subs(t, n))
                return imageset(Lambda(n, expr), S.Integers)

    if other == S.Reals:
        from sympy.solvers.solveset import solveset_real
        from sympy.core.function import expand_complex

        f = self.lamda.expr
        n = self.lamda.variables[0]

        n_ = Dummy(n.name, real=True)
        f_ = f.subs(n, n_)

        re, im = f_.as_real_imag()
        im = expand_complex(im)

        re = re.subs(n_, n)
        im = im.subs(n_, n)
        ifree = im.free_symbols
        lam = Lambda(n, re)
        if not im:
            # allow re-evaluation
            # of self in this case to make
            # the result canonical
            pass
        elif im.is_zero is False:
            return S.EmptySet
        elif ifree != {n}:
            return None
        else:
            # univarite imaginary part in same variable
            base_set = base_set.intersect(solveset_real(im, n))
        return imageset(lam, base_set)

    elif isinstance(other, Interval):
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None
        new_lopen, new_ropen = other.left_open, other.right_open

        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]
            # TODO: Design a technique to handle multiple-inverse
            # functions

            # Any of the new boundary values cannot be determined
            if any(i is None for i in (new_sup, new_inf)):
                return


            range_set = S.EmptySet

            if all(i.is_real for i in (new_sup, new_inf)):
                # this assumes continuity of underlying function
                # however fixes the case when it is decreasing
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                range_set = base_set.intersect(new_interval)
            else:
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            if range_set is S.EmptySet:
                return S.EmptySet
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return
```
### 2 - sympy/sets/handlers/intersection.py:

Start line: 30, End line: 75

```python
@dispatch(ComplexRegion, Set)
def intersection_sets(self, other): # noqa:F811
    if other.is_ComplexRegion:
        # self in rectangular form
        if (not self.polar) and (not other.polar):
            return ComplexRegion(Intersection(self.sets, other.sets))

        # self in polar form
        elif self.polar and other.polar:
            r1, theta1 = self.a_interval, self.b_interval
            r2, theta2 = other.a_interval, other.b_interval
            new_r_interval = Intersection(r1, r2)
            new_theta_interval = Intersection(theta1, theta2)

            # 0 and 2*Pi means the same
            if ((2*S.Pi in theta1 and S.Zero in theta2) or
               (2*S.Pi in theta2 and S.Zero in theta1)):
                new_theta_interval = Union(new_theta_interval,
                                           FiniteSet(0))
            return ComplexRegion(new_r_interval*new_theta_interval,
                                polar=True)


    if other.is_subset(S.Reals):
        new_interval = []
        x = symbols("x", cls=Dummy, real=True)

        # self in rectangular form
        if not self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

        # self in polar form
        elif self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
                if S.Pi in element.args[1]:
                    new_interval.append(ImageSet(Lambda(x, -x), element.args[0]))
                if S.Zero in element.args[0]:
                    new_interval.append(FiniteSet(0))
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)
```
### 3 - sympy/sets/handlers/intersection.py:

Start line: 1, End line: 28

```python
from sympy import (S, Dummy, Lambda, symbols, Interval, Intersection, Set,
                   EmptySet, FiniteSet, Union, ComplexRegion)
from sympy.multipledispatch import dispatch
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import (Integers, Naturals, Reals, Range,
    ImageSet, Rationals)
from sympy.sets.sets import UniversalSet, imageset, ProductSet


@dispatch(ConditionSet, ConditionSet)
def intersection_sets(a, b): # noqa:F811
    return None

@dispatch(ConditionSet, Set)
def intersection_sets(a, b): # noqa:F811
    return ConditionSet(a.sym, a.condition, Intersection(a.base_set, b))

@dispatch(Naturals, Integers)
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Naturals, Naturals)
def intersection_sets(a, b): # noqa:F811
    return a if a is S.Naturals else b

@dispatch(Interval, Naturals)
def intersection_sets(a, b): # noqa:F811
    return intersection_sets(b, a)
```
### 4 - sympy/sets/sets.py:

Start line: 1389, End line: 1425

```python
class Intersection(Set, LatticeOp):

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - set((s,))
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")
```
### 5 - sympy/sets/fancysets.py:

Start line: 330, End line: 357

```python
class ImageSet(Set):
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        signature = flambda.signature

        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        sets = [_sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)
```
### 6 - sympy/sets/fancysets.py:

Start line: 431, End line: 481

```python
class ImageSet(Set):

    def _contains(self, other):
        # ... other code

        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # Get the basic objects together:
        other = _sympify(other)
        expr = self.lamda.expr
        sig = self.lamda.signature
        variables = self.lamda.variables
        base_sets = self.base_sets

        # Use dummy symbols for ImageSet parameters so they don't match
        # anything in other
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # Map the parts of other to those in the Lambda expr
        equations = []
        for eq in get_equations(expr, other):
            # Unsatisfiable equation?
            if eq is False:
                return False
            equations.append(eq)

        # Map the symbols in the signature to the corresponding domains
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # Can't factor the base sets to a ProductSet
            return None

        # Which of the variables in the Lambda signature need to be solved for?
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # Use internal multivariate solveset
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return fuzzy_not(solnset.is_empty)
```
### 7 - sympy/sets/sets.py:

Start line: 1427, End line: 1529

```python
class Intersection(Set, LatticeOp):

    @staticmethod
    def _handle_finite_sets(args):
        '''Simplify intersection of one or more FiniteSets and other sets'''

        # First separate the FiniteSets from the others
        fs_args, others = sift(args, lambda x: x.is_FiniteSet, binary=True)

        # Let the caller handle intersection of non-FiniteSets
        if not fs_args:
            return

        # Convert to Python sets and build the set of all elements
        fs_sets = [set(fs) for fs in fs_args]
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Extract elements that are definitely in or definitely not in the
        # intersection. Here we check contains for all of args.
        definite = set()
        for e in all_elements:
            inall = fuzzy_and(s.contains(e) for s in args)
            if inall is True:
                definite.add(e)
            if inall is not None:
                for s in fs_sets:
                    s.discard(e)

        # At this point all elements in all of fs_sets are possibly in the
        # intersection. In some cases this is because they are definitely in
        # the intersection of the finite sets but it's not clear if they are
        # members of others. We might have {m, n}, {m}, and Reals where we
        # don't know if m or n is real. We want to remove n here but it is
        # possibly in because it might be equal to m. So what we do now is
        # extract the elements that are definitely in the remaining finite
        # sets iteratively until we end up with {n}, {}. At that point if we
        # get any empty set all remaining elements are discarded.

        fs_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Need fuzzy containment testing
        fs_symsets = [FiniteSet(*s) for s in fs_sets]

        while fs_elements:
            for e in fs_elements:
                infs = fuzzy_and(s.contains(e) for s in fs_symsets)
                if infs is True:
                    definite.add(e)
                if infs is not None:
                    for n, s in enumerate(fs_sets):
                        # Update Python set and FiniteSet
                        if e in s:
                            s.remove(e)
                            fs_symsets[n] = FiniteSet(*s)
                    fs_elements.remove(e)
                    break
            # If we completed the for loop without removing anything we are
            # done so quit the outer while loop
            else:
                break

        # If any of the sets of remainder elements is empty then we discard
        # all of them for the intersection.
        if not all(fs_sets):
            fs_sets = [set()]

        # Here we fold back the definitely included elements into each fs.
        # Since they are definitely included they must have been members of
        # each FiniteSet to begin with. We could instead fold these in with a
        # Union at the end to get e.g. {3}|({x}&{y}) rather than {3,x}&{3,y}.
        if definite:
            fs_sets = [fs | definite for fs in fs_sets]

        if fs_sets == [set()]:
            return S.EmptySet

        sets = [FiniteSet(*s) for s in fs_sets]

        # Any set in others is redundant if it contains all the elements that
        # are in the finite sets so we don't need it in the Intersection
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())
        is_redundant = lambda o: all(fuzzy_bool(o.contains(e)) for e in all_elements)
        others = [o for o in others if not is_redundant(o)]

        if others:
            rest = Intersection(*others)
            # XXX: Maybe this shortcut should be at the beginning. For large
            # FiniteSets it could much more efficient to process the other
            # sets first...
            if rest is S.EmptySet:
                return S.EmptySet
            # Flatten the Intersection
            if rest.is_Intersection:
                sets.extend(rest.args)
            else:
                sets.append(rest)

        if len(sets) == 1:
            return sets[0]
        else:
            return Intersection(*sets, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])
```
### 8 - sympy/sets/fancysets.py:

Start line: 273, End line: 329

```python
class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from `imageset`.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy.sets.sets import FiniteSet, Interval
    >>> from sympy.sets.fancysets import ImageSet

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    FiniteSet(1, 4, 9)

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in `base_set` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    FiniteSet(0)

    See Also
    ========

    sympy.sets.sets.imageset
    """
```
### 9 - sympy/sets/sets.py:

Start line: 1347, End line: 1362

```python
class Intersection(Set, LatticeOp):

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_evaluate[0])

        # flatten inputs to merge intersections and iterables
        args = list(ordered(set(_sympify(args))))

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj
```
### 10 - sympy/sets/sets.py:

Start line: 2119, End line: 2150

```python
def imageset(*args):
    # ... other code

    if len(set_list) == 1:
        set = set_list[0]
        try:
            # TypeError if arg count != set dimensions
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args

        if f.variables[0] == f.expr:
            return set

        if isinstance(set, ImageSet):
            # XXX: Maybe this should just be:
            # f2 = set.lambda
            # fun = Lambda(f2.signature, f(*f2.expr))
            # return imageset(fun, *set.base_sets)
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        if r is not None:
            return r

    return ImageSet(f, *set_list)
```
### 11 - sympy/sets/handlers/intersection.py:

Start line: 405, End line: 457

```python
@dispatch(type(EmptySet), Set)
def intersection_sets(a, b): # noqa:F811
    return S.EmptySet

@dispatch(UniversalSet, Set)
def intersection_sets(a, b): # noqa:F811
    return b

@dispatch(FiniteSet, FiniteSet)
def intersection_sets(a, b): # noqa:F811
    return FiniteSet(*(a._elements & b._elements))

@dispatch(FiniteSet, Set)
def intersection_sets(a, b): # noqa:F811
    try:
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # could not evaluate `el in b` due to symbolic ranges.

@dispatch(Set, Set)
def intersection_sets(a, b): # noqa:F811
    return None

@dispatch(Integers, Rationals)
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Naturals, Rationals)
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Rationals, Reals)
def intersection_sets(a, b): # noqa:F811
    return a

def _intlike_interval(a, b):
    try:
        from sympy.functions.elementary.integers import floor, ceiling
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        return intersection_sets(s, b)  # take out endpoints if open interval
    except ValueError:
        return None

@dispatch(Integers, Interval)
def intersection_sets(a, b): # noqa:F811
    return _intlike_interval(a, b)

@dispatch(Naturals, Interval)
def intersection_sets(a, b): # noqa:F811
    return _intlike_interval(a, b)
```
### 19 - sympy/sets/handlers/intersection.py:

Start line: 77, End line: 103

```python
@dispatch(Integers, Reals)
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Range, Interval)
def intersection_sets(a, b): # noqa:F811
    from sympy.functions.elementary.integers import floor, ceiling
    if not all(i.is_number for i in b.args[:2]):
        return

    # In case of null Range, return an EmptySet.
    if a.size == 0:
        return S.EmptySet

    # trim down to self's size, and represent
    # as a Range with step 1.
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    return intersection_sets(a, Range(start, end + 1))

@dispatch(Range, Naturals)
def intersection_sets(a, b): # noqa:F811
    return intersection_sets(a, Interval(b.inf, S.Infinity))
```
### 22 - sympy/sets/handlers/intersection.py:

Start line: 351, End line: 403

```python
@dispatch(ProductSet, ProductSet)
def intersection_sets(a, b): # noqa:F811
    if len(b.args) != len(a.args):
        return S.EmptySet
    return ProductSet(*(i.intersect(j) for i, j in zip(a.sets, b.sets)))


@dispatch(Interval, Interval)
def intersection_sets(a, b): # noqa:F811
    # handle (-oo, oo)
    infty = S.NegativeInfinity, S.Infinity
    if a == Interval(*infty):
        l, r = a.left, a.right
        if l.is_real or l in infty or r.is_real or r in infty:
            return b

    # We can't intersect [0,3] with [x,6] -- we don't know if x>0 or x<0
    if not a._is_comparable(b):
        return None

    empty = False

    if a.start <= b.end and b.start <= a.end:
        # Get topology right.
        if a.start < b.start:
            start = b.start
            left_open = b.left_open
        elif a.start > b.start:
            start = a.start
            left_open = a.left_open
        else:
            start = a.start
            left_open = a.left_open or b.left_open

        if a.end < b.end:
            end = a.end
            right_open = a.right_open
        elif a.end > b.end:
            end = b.end
            right_open = b.right_open
        else:
            end = a.end
            right_open = a.right_open or b.right_open

        if end - start == 0 and (left_open or right_open):
            empty = True
    else:
        empty = True

    if empty:
        return S.EmptySet

    return Interval(start, end, left_open, right_open)
```
### 24 - sympy/sets/handlers/intersection.py:

Start line: 105, End line: 220

```python
@dispatch(Range, Range)
def intersection_sets(a, b): # noqa:F811
    from sympy.solvers.diophantine import diop_linear
    from sympy.core.numbers import ilcm
    from sympy import sign

    # non-overlap quick exits
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet

    # work with finite end at the start
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed

    # If both ends are infinite then it means that one Range is just the set
    # of all integers (the step must be 1).
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a

    # this equation represents the values of the Range;
    # it's a linear equation
    eq = lambda r, i: r.start + i*r.step

    # we want to know when the two equations might
    # have integer solutions so we use the diophantine
    # solver
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))

    # check for no solution
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet

    # there is a solution
    # -------------------

    # find the coincident point, c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)

    # find the first point, if possible, in each range
    # since c may not be that point
    def _first_finite_point(r1, c):
        if c == r1.start:
            return c
        # st is the signed step we need to take to
        # get from c to r1.start
        st = sign(r1.start - c)*step
        # use Range to calculate the first point:
        # we want to get as close as possible to
        # r1.start; the Range will not be null since
        # it will at least contain c
        s1 = Range(c, r1.start + st, st)[-1]
        if s1 == r1.start:
            pass
        else:
            # if we didn't hit r1.start then, if the
            # sign of st didn't match the sign of r1.step
            # we are off by one and s1 is not in r1
            if sign(r1.step) != sign(st):
                s1 -= st
        if s1 not in r1:
            return
        return s1

    # calculate the step size of the new Range
    step = abs(ilcm(r1.step, r2.step))
    s1 = _first_finite_point(r1, c)
    if s1 is None:
        return S.EmptySet
    s2 = _first_finite_point(r2, c)
    if s2 is None:
        return S.EmptySet

    # replace the corresponding start or stop in
    # the original Ranges with these points; the
    # result must have at least one point since
    # we know that s1 and s2 are in the Ranges
    def _updated_range(r, first):
        st = sign(r.step)*step
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # work with them both in the increasing direction
    if sign(r1.step) < 0:
        r1 = r1.reversed
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # return clipped Range with positive step; it
    # can't be empty at this point
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)


@dispatch(Range, Integers)
def intersection_sets(a, b): # noqa:F811
    return a
```
### 76 - sympy/solvers/diophantine.py:

Start line: 1, End line: 98

```python
from __future__ import print_function, division

from sympy.core.add import Add
from sympy.core.compatibility import as_int, is_sequence, range
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
    divisors, factorint, multiplicity, perfect_power)
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solvers import check_assumptions
from sympy.solvers.solveset import solveset_real
from sympy.utilities import default_sort_key, numbered_symbols
from sympy.utilities.misc import filldedent



# these are imported with 'from sympy.solvers.diophantine import *
__all__ = ['diophantine', 'classify_diop']


# these types are known (but not necessarily handled)
diop_known = {
    "binary_quadratic",
    "cubic_thue",
    "general_pythagorean",
    "general_sum_of_even_powers",
    "general_sum_of_squares",
    "homogeneous_general_quadratic",
    "homogeneous_ternary_quadratic",
    "homogeneous_ternary_quadratic_normal",
    "inhomogeneous_general_quadratic",
    "inhomogeneous_ternary_quadratic",
    "linear",
    "univariate"}


def _is_int(i):
    try:
        as_int(i)
        return True
    except ValueError:
        pass


def _sorted_tuple(*i):
    return tuple(sorted(i))


def _remove_gcd(*x):
    try:
        g = igcd(*x)
    except ValueError:
        fx = list(filter(None, x))
        if len(fx) < 2:
            return x
        g = igcd(*[i.as_content_primitive()[0] for i in fx])
    except TypeError:
        raise TypeError('_remove_gcd(a,b,c) or _remove_gcd(*container)')
    if g == 1:
        return x
    return tuple([i//g for i in x])


def _rational_pq(a, b):
    # return `(numer, denom)` for a/b; sign in numer and gcd removed
    return _remove_gcd(sign(b)*a, abs(b))


def _nint_or_floor(p, q):
    # return nearest int to p/q; in case of tie return floor(p/q)
    w, r = divmod(p, q)
    if abs(r) <= abs(q)//2:
        return w
    return w + 1


def _odd(i):
    return i % 2 != 0


def _even(i):
    return i % 2 == 0
```
### 96 - sympy/solvers/diophantine.py:

Start line: 168, End line: 211

```python
def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):

    from sympy.utilities.iterables import (
        subsets, permute_signs, signed_permutations)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    try:
        var = list(eq.expand(force=True).free_symbols)
        var.sort(key=default_sort_key)
        if syms:
            if not is_sequence(syms):
                raise TypeError(
                    'syms should be given as a sequence, e.g. a list')
            syms = [i for i in syms if i in var]
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                return {tuple([t[dict_sym_index[i]] for i in var])
                            for t in diophantine(eq, param, permute=permute)}
        n, d = eq.as_numer_denom()
        if n.is_number:
            return set()
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        else:
            eq = n
        eq = factor_terms(eq)
        assert not eq.is_number
        eq = eq.as_independent(*var, as_Add=False)[1]
        p = Poly(eq)
        assert not any(g.is_number for g in p.gens)
        eq = p.as_expr()
        assert eq.is_polynomial()
    except (GeneratorsNeeded, AssertionError, AttributeError):
        raise TypeError(filldedent('''
        tion should be a polynomial with Rational coefficients.'''))

    # permute only sign
    do_permute_signs = False
    # permute sign and values
    do_permute_signs_var = False
    # permute few signs
    permute_few_signs = False
    # ... other code
```
