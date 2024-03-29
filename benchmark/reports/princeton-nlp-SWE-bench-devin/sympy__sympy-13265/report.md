# sympy__sympy-13265

| **sympy/sympy** | `1599a0d7cdf529c2d0db3a68e74a9aabb8334aa5` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 699 |
| **Any found context length** | 699 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -594,7 +594,7 @@ def shorter(*choices):
     short = shorter(short, cancel(short))
     short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
     if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase):
-        short = exptrigsimp(short, simplify=False)
+        short = exptrigsimp(short)
 
     # get rid of hollow 2-arg Mul factorization
     hollow_mul = Transform(
@@ -1093,7 +1093,7 @@ def tofunc(nu, z):
     def expander(fro):
         def repl(nu, z):
             if (nu % 1) == S(1)/2:
-                return exptrigsimp(trigsimp(unpolarify(
+                return simplify(trigsimp(unpolarify(
                         fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                             func=True)).subs(z0, z)))
             elif nu.is_Integer and nu > 1:
diff --git a/sympy/simplify/trigsimp.py b/sympy/simplify/trigsimp.py
--- a/sympy/simplify/trigsimp.py
+++ b/sympy/simplify/trigsimp.py
@@ -513,12 +513,9 @@ def traverse(e):
     return trigsimpfunc(expr)
 
 
-def exptrigsimp(expr, simplify=True):
+def exptrigsimp(expr):
     """
     Simplifies exponential / trigonometric / hyperbolic functions.
-    When ``simplify`` is True (default) the expression obtained after the
-    simplification step will be then be passed through simplify to
-    precondition it so the final transformations will be applied.
 
     Examples
     ========
@@ -544,35 +541,53 @@ def exp_trig(e):
         return min(*choices, key=count_ops)
     newexpr = bottom_up(expr, exp_trig)
 
-    if simplify:
-        newexpr = newexpr.simplify()
-
-    # conversion from exp to hyperbolic
-    ex = newexpr.atoms(exp, S.Exp1)
-    ex = [ei for ei in ex if 1/ei not in ex]
-    ## sinh and cosh
-    for ei in ex:
-        e2 = ei**-2
-        if e2 in ex:
-            a = e2.args[0]/2 if not e2 is S.Exp1 else S.Half
-            newexpr = newexpr.subs((e2 + 1)*ei, 2*cosh(a))
-            newexpr = newexpr.subs((e2 - 1)*ei, 2*sinh(a))
-    ## exp ratios to tan and tanh
-    for ei in ex:
-        n, d = ei - 1, ei + 1
-        et = n/d
-        etinv = d/n  # not 1/et or else recursion errors arise
-        a = ei.args[0] if ei.func is exp else S.One
-        if a.is_Mul or a is S.ImaginaryUnit:
-            c = a.as_coefficient(I)
-            if c:
-                t = S.ImaginaryUnit*tan(c/2)
-                newexpr = newexpr.subs(etinv, 1/t)
-                newexpr = newexpr.subs(et, t)
-                continue
-        t = tanh(a/2)
-        newexpr = newexpr.subs(etinv, 1/t)
-        newexpr = newexpr.subs(et, t)
+    def f(rv):
+        if not rv.is_Mul:
+            return rv
+        rvd = rv.as_powers_dict()
+        newd = rvd.copy()
+
+        def signlog(expr, sign=1):
+            if expr is S.Exp1:
+                return sign, 1
+            elif isinstance(expr, exp):
+                return sign, expr.args[0]
+            elif sign == 1:
+                return signlog(-expr, sign=-1)
+            else:
+                return None, None
+
+        ee = rvd[S.Exp1]
+        for k in rvd:
+            if k.is_Add and len(k.args) == 2:
+                # k == c*(1 + sign*E**x)
+                c = k.args[0]
+                sign, x = signlog(k.args[1]/c)
+                if not x:
+                    continue
+                m = rvd[k]
+                newd[k] -= m
+                if ee == -x*m/2:
+                    # sinh and cosh
+                    newd[S.Exp1] -= ee
+                    ee = 0
+                    if sign == 1:
+                        newd[2*c*cosh(x/2)] += m
+                    else:
+                        newd[-2*c*sinh(x/2)] += m
+                elif newd[1 - sign*S.Exp1**x] == -m:
+                    # tanh
+                    del newd[1 - sign*S.Exp1**x]
+                    if sign == 1:
+                        newd[-c/tanh(x/2)] += m
+                    else:
+                        newd[-c*tanh(x/2)] += m
+                else:
+                    newd[1 + sign*S.Exp1**x] += m
+                    newd[c] += m
+
+        return Mul(*[k**newd[k] for k in newd])
+    newexpr = bottom_up(newexpr, f)
 
     # sin/cos and sinh/cosh ratios to tan and tanh, respectively
     if newexpr.has(HyperbolicFunction):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/simplify/simplify.py | 597 | 597 | - | - | -
| sympy/simplify/simplify.py | 1096 | 1096 | - | - | -
| sympy/simplify/trigsimp.py | 516 | 521 | 1 | 1 | 699
| sympy/simplify/trigsimp.py | 547 | 575 | 1 | 1 | 699


## Problem Statement

```
Simplification fails to recognize sin expressed as exponentials
\`\`\`

In [2]: exp(Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
Out[2]: 
⎡    -ⅈ    ⅈ          -ⅈ      ⅈ   ⎤
⎢   ℯ     ℯ        ⅈ⋅ℯ     ⅈ⋅ℯ    ⎥
⎢   ─── + ──     - ───── + ────  0⎥
⎢    2    2          2      2     ⎥
⎢                                 ⎥
⎢     ⅈ      -ⅈ      -ⅈ    ⅈ      ⎥
⎢  ⅈ⋅ℯ    ⅈ⋅ℯ       ℯ     ℯ       ⎥
⎢- ──── + ─────     ─── + ──     0⎥
⎢   2       2        2    2       ⎥
⎢                                 ⎥
⎣      0               0         1⎦

In [3]: simplify(_)
Out[3]: 
⎡     cos(1)       -sin(1)  0⎤
⎢                            ⎥
⎢  ⎛     2⋅ⅈ⎞  -ⅈ            ⎥
⎢ⅈ⋅⎝1 - ℯ   ⎠⋅ℯ              ⎥
⎢────────────────  cos(1)   0⎥
⎢       2                    ⎥
⎢                            ⎥
⎣       0             0     1⎦

In [4]: m = _

In [5]: fu(_)
Out[5]: 
⎡     cos(1)       -sin(1)  0⎤
⎢                            ⎥
⎢  ⎛     2⋅ⅈ⎞  -ⅈ            ⎥
⎢ⅈ⋅⎝1 - ℯ   ⎠⋅ℯ              ⎥
⎢────────────────  cos(1)   0⎥
⎢       2                    ⎥
⎢                            ⎥
⎣       0             0     1⎦

In [6]: sqrt
sqrt           sqrt_mod       sqrt_mod_iter  sqrtdenest     

In [6]: sqrtdenest(_)
Out[6]: 
⎡    cos(1)      -sin(1)  0⎤
⎢                          ⎥
⎢     ⅈ      -ⅈ            ⎥
⎢  ⅈ⋅ℯ    ⅈ⋅ℯ              ⎥
⎢- ──── + ─────  cos(1)   0⎥
⎢   2       2              ⎥
⎢                          ⎥
⎣      0            0     1⎦

In [7]: trig
trigamma      trigonometry  trigsimp      

In [7]: trigsimp(_)
Out[7]: 
⎡    cos(1)      -sin(1)  0⎤
⎢                          ⎥
⎢     ⅈ      -ⅈ            ⎥
⎢  ⅈ⋅ℯ    ⅈ⋅ℯ              ⎥
⎢- ──── + ─────  cos(1)   0⎥
⎢   2       2              ⎥
⎢                          ⎥
⎣      0            0     1⎦

\`\`\`

The expression for `sin(1)` has not been recognized, while expressions for `cos` and `-sin(1)` have.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/simplify/trigsimp.py** | 516 | 587| 699 | 699 | 12062 | 
| 2 | 2 sympy/simplify/fu.py | 1 | 189| 2051 | 2750 | 30350 | 
| 3 | 3 sympy/functions/elementary/trigonometric.py | 782 | 831| 596 | 3346 | 53621 | 
| 4 | 3 sympy/simplify/fu.py | 1227 | 1284| 527 | 3873 | 53621 | 
| 5 | **3 sympy/simplify/trigsimp.py** | 777 | 817| 729 | 4602 | 53621 | 
| 6 | 3 sympy/functions/elementary/trigonometric.py | 254 | 357| 896 | 5498 | 53621 | 
| 7 | **3 sympy/simplify/trigsimp.py** | 425 | 489| 480 | 5978 | 53621 | 
| 8 | **3 sympy/simplify/trigsimp.py** | 589 | 650| 590 | 6568 | 53621 | 
| 9 | 3 sympy/functions/elementary/trigonometric.py | 359 | 416| 528 | 7096 | 53621 | 
| 10 | 3 sympy/simplify/fu.py | 2096 | 2137| 329 | 7425 | 53621 | 
| 11 | **3 sympy/simplify/trigsimp.py** | 285 | 350| 801 | 8226 | 53621 | 
| 12 | **3 sympy/simplify/trigsimp.py** | 28 | 117| 876 | 9102 | 53621 | 
| 13 | 4 sympy/integrals/rubi/utility_function.py | 4743 | 4838| 961 | 10063 | 135875 | 
| 14 | 4 sympy/simplify/fu.py | 665 | 686| 223 | 10286 | 135875 | 
| 15 | 4 sympy/functions/elementary/trigonometric.py | 418 | 463| 438 | 10724 | 135875 | 
| 16 | **4 sympy/simplify/trigsimp.py** | 967 | 1047| 724 | 11448 | 135875 | 
| 17 | 4 sympy/simplify/fu.py | 1344 | 1364| 226 | 11674 | 135875 | 
| 18 | **4 sympy/simplify/trigsimp.py** | 736 | 776| 748 | 12422 | 135875 | 
| 19 | 4 sympy/functions/elementary/trigonometric.py | 744 | 780| 767 | 13189 | 135875 | 
| 20 | 4 sympy/functions/elementary/trigonometric.py | 652 | 693| 385 | 13574 | 135875 | 
| 21 | 4 sympy/simplify/fu.py | 399 | 425| 248 | 13822 | 135875 | 
| 22 | 4 sympy/functions/elementary/trigonometric.py | 545 | 630| 843 | 14665 | 135875 | 
| 23 | **4 sympy/simplify/trigsimp.py** | 118 | 175| 827 | 15492 | 135875 | 
| 24 | 4 sympy/functions/elementary/trigonometric.py | 713 | 742| 259 | 15751 | 135875 | 
| 25 | 4 sympy/simplify/fu.py | 524 | 542| 174 | 15925 | 135875 | 
| 26 | 4 sympy/functions/elementary/trigonometric.py | 938 | 1031| 833 | 16758 | 135875 | 
| 27 | 4 sympy/functions/elementary/trigonometric.py | 695 | 711| 226 | 16984 | 135875 | 
| 28 | 4 sympy/functions/elementary/trigonometric.py | 846 | 883| 325 | 17309 | 135875 | 
| 29 | 4 sympy/functions/elementary/trigonometric.py | 1121 | 1178| 481 | 17790 | 135875 | 
| 30 | 4 sympy/integrals/rubi/utility_function.py | 4145 | 4741| 6555 | 24345 | 135875 | 
| 31 | **4 sympy/simplify/trigsimp.py** | 176 | 202| 455 | 24800 | 135875 | 
| 32 | 4 sympy/simplify/fu.py | 545 | 563| 175 | 24975 | 135875 | 
| 33 | 4 sympy/simplify/fu.py | 283 | 309| 231 | 25206 | 135875 | 
| 34 | 5 sympy/core/evalf.py | 742 | 805| 572 | 25778 | 149087 | 
| 35 | 5 sympy/simplify/fu.py | 253 | 280| 180 | 25958 | 149087 | 
| 36 | 5 sympy/integrals/rubi/utility_function.py | 6673 | 6719| 798 | 26756 | 149087 | 
| 37 | **5 sympy/simplify/trigsimp.py** | 912 | 965| 443 | 27199 | 149087 | 
| 38 | 6 sympy/simplify/combsimp.py | 224 | 301| 782 | 27981 | 153416 | 
| 39 | 6 sympy/simplify/fu.py | 191 | 226| 315 | 28296 | 153416 | 
| 40 | 7 sympy/parsing/sympy_parser.py | 310 | 357| 405 | 28701 | 160736 | 
| 41 | 7 sympy/functions/elementary/trigonometric.py | 1355 | 1429| 707 | 29408 | 160736 | 
| 42 | 7 sympy/functions/elementary/trigonometric.py | 833 | 844| 124 | 29532 | 160736 | 


## Missing Patch Files

 * 1: sympy/simplify/simplify.py
 * 2: sympy/simplify/trigsimp.py

## Patch

```diff
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -594,7 +594,7 @@ def shorter(*choices):
     short = shorter(short, cancel(short))
     short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
     if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase):
-        short = exptrigsimp(short, simplify=False)
+        short = exptrigsimp(short)
 
     # get rid of hollow 2-arg Mul factorization
     hollow_mul = Transform(
@@ -1093,7 +1093,7 @@ def tofunc(nu, z):
     def expander(fro):
         def repl(nu, z):
             if (nu % 1) == S(1)/2:
-                return exptrigsimp(trigsimp(unpolarify(
+                return simplify(trigsimp(unpolarify(
                         fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                             func=True)).subs(z0, z)))
             elif nu.is_Integer and nu > 1:
diff --git a/sympy/simplify/trigsimp.py b/sympy/simplify/trigsimp.py
--- a/sympy/simplify/trigsimp.py
+++ b/sympy/simplify/trigsimp.py
@@ -513,12 +513,9 @@ def traverse(e):
     return trigsimpfunc(expr)
 
 
-def exptrigsimp(expr, simplify=True):
+def exptrigsimp(expr):
     """
     Simplifies exponential / trigonometric / hyperbolic functions.
-    When ``simplify`` is True (default) the expression obtained after the
-    simplification step will be then be passed through simplify to
-    precondition it so the final transformations will be applied.
 
     Examples
     ========
@@ -544,35 +541,53 @@ def exp_trig(e):
         return min(*choices, key=count_ops)
     newexpr = bottom_up(expr, exp_trig)
 
-    if simplify:
-        newexpr = newexpr.simplify()
-
-    # conversion from exp to hyperbolic
-    ex = newexpr.atoms(exp, S.Exp1)
-    ex = [ei for ei in ex if 1/ei not in ex]
-    ## sinh and cosh
-    for ei in ex:
-        e2 = ei**-2
-        if e2 in ex:
-            a = e2.args[0]/2 if not e2 is S.Exp1 else S.Half
-            newexpr = newexpr.subs((e2 + 1)*ei, 2*cosh(a))
-            newexpr = newexpr.subs((e2 - 1)*ei, 2*sinh(a))
-    ## exp ratios to tan and tanh
-    for ei in ex:
-        n, d = ei - 1, ei + 1
-        et = n/d
-        etinv = d/n  # not 1/et or else recursion errors arise
-        a = ei.args[0] if ei.func is exp else S.One
-        if a.is_Mul or a is S.ImaginaryUnit:
-            c = a.as_coefficient(I)
-            if c:
-                t = S.ImaginaryUnit*tan(c/2)
-                newexpr = newexpr.subs(etinv, 1/t)
-                newexpr = newexpr.subs(et, t)
-                continue
-        t = tanh(a/2)
-        newexpr = newexpr.subs(etinv, 1/t)
-        newexpr = newexpr.subs(et, t)
+    def f(rv):
+        if not rv.is_Mul:
+            return rv
+        rvd = rv.as_powers_dict()
+        newd = rvd.copy()
+
+        def signlog(expr, sign=1):
+            if expr is S.Exp1:
+                return sign, 1
+            elif isinstance(expr, exp):
+                return sign, expr.args[0]
+            elif sign == 1:
+                return signlog(-expr, sign=-1)
+            else:
+                return None, None
+
+        ee = rvd[S.Exp1]
+        for k in rvd:
+            if k.is_Add and len(k.args) == 2:
+                # k == c*(1 + sign*E**x)
+                c = k.args[0]
+                sign, x = signlog(k.args[1]/c)
+                if not x:
+                    continue
+                m = rvd[k]
+                newd[k] -= m
+                if ee == -x*m/2:
+                    # sinh and cosh
+                    newd[S.Exp1] -= ee
+                    ee = 0
+                    if sign == 1:
+                        newd[2*c*cosh(x/2)] += m
+                    else:
+                        newd[-2*c*sinh(x/2)] += m
+                elif newd[1 - sign*S.Exp1**x] == -m:
+                    # tanh
+                    del newd[1 - sign*S.Exp1**x]
+                    if sign == 1:
+                        newd[-c/tanh(x/2)] += m
+                    else:
+                        newd[-c*tanh(x/2)] += m
+                else:
+                    newd[1 + sign*S.Exp1**x] += m
+                    newd[c] += m
+
+        return Mul(*[k**newd[k] for k in newd])
+    newexpr = bottom_up(newexpr, f)
 
     # sin/cos and sinh/cosh ratios to tan and tanh, respectively
     if newexpr.has(HyperbolicFunction):

```

## Test Patch

```diff
diff --git a/sympy/simplify/tests/test_simplify.py b/sympy/simplify/tests/test_simplify.py
--- a/sympy/simplify/tests/test_simplify.py
+++ b/sympy/simplify/tests/test_simplify.py
@@ -156,8 +156,11 @@ def test_simplify_other():
 def test_simplify_complex():
     cosAsExp = cos(x)._eval_rewrite_as_exp(x)
     tanAsExp = tan(x)._eval_rewrite_as_exp(x)
-    assert simplify(cosAsExp*tanAsExp).expand() == (
-        sin(x))._eval_rewrite_as_exp(x).expand()  # issue 4341
+    assert simplify(cosAsExp*tanAsExp) == sin(x) # issue 4341
+
+    # issue 10124
+    assert simplify(exp(Matrix([[0, -1], [1, 0]]))) == Matrix([[cos(1),
+        -sin(1)], [sin(1), cos(1)]])
 
 
 def test_simplify_ratio():
diff --git a/sympy/simplify/tests/test_trigsimp.py b/sympy/simplify/tests/test_trigsimp.py
--- a/sympy/simplify/tests/test_trigsimp.py
+++ b/sympy/simplify/tests/test_trigsimp.py
@@ -361,6 +361,8 @@ def valid(a, b):
 
     assert exptrigsimp(exp(x) + exp(-x)) == 2*cosh(x)
     assert exptrigsimp(exp(x) - exp(-x)) == 2*sinh(x)
+    assert exptrigsimp((2*exp(x)-2*exp(-x))/(exp(x)+exp(-x))) == 2*tanh(x)
+    assert exptrigsimp((2*exp(2*x)-2)/(exp(2*x)+1)) == 2*tanh(x)
     e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
          cosh(x) - sinh(x), cosh(x) + sinh(x)]
     ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
@@ -378,12 +380,8 @@ def valid(a, b):
     for a in (1, I, x, I*x, 1 + I):
         w = exp(a)
         eq = y*(w - 1/w)/(w + 1/w)
-        s = simplify(eq)
-        assert s == exptrigsimp(eq)
-        res.append(s)
-        sinv = simplify(1/eq)
-        assert sinv == exptrigsimp(1/eq)
-        res.append(sinv)
+        res.append(simplify(eq))
+        res.append(simplify(1/eq))
     assert all(valid(i, j) for i, j in zip(res, ok))
 
     for a in range(1, 3):

```


## Code snippets

### 1 - sympy/simplify/trigsimp.py:

Start line: 516, End line: 587

```python
def exptrigsimp(expr, simplify=True):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.
    When ``simplify`` is True (default) the expression obtained after the
    simplification step will be then be passed through simplify to
    precondition it so the final transformations will be applied.

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

    if simplify:
        newexpr = newexpr.simplify()

    # conversion from exp to hyperbolic
    ex = newexpr.atoms(exp, S.Exp1)
    ex = [ei for ei in ex if 1/ei not in ex]
    ## sinh and cosh
    for ei in ex:
        e2 = ei**-2
        if e2 in ex:
            a = e2.args[0]/2 if not e2 is S.Exp1 else S.Half
            newexpr = newexpr.subs((e2 + 1)*ei, 2*cosh(a))
            newexpr = newexpr.subs((e2 - 1)*ei, 2*sinh(a))
    ## exp ratios to tan and tanh
    for ei in ex:
        n, d = ei - 1, ei + 1
        et = n/d
        etinv = d/n  # not 1/et or else recursion errors arise
        a = ei.args[0] if ei.func is exp else S.One
        if a.is_Mul or a is S.ImaginaryUnit:
            c = a.as_coefficient(I)
            if c:
                t = S.ImaginaryUnit*tan(c/2)
                newexpr = newexpr.subs(etinv, 1/t)
                newexpr = newexpr.subs(et, t)
                continue
        t = tanh(a/2)
        newexpr = newexpr.subs(etinv, 1/t)
        newexpr = newexpr.subs(et, t)

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
### 2 - sympy/simplify/fu.py:

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
### 3 - sympy/functions/elementary/trigonometric.py:

Start line: 782, End line: 831

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
### 4 - sympy/simplify/fu.py:

Start line: 1227, End line: 1284

```python
def TRmorrie(rv):
    """Returns cos(x)*cos(2*x)*...*cos(2**(k-1)*x) -> sin(2**k*x)/(2**k*sin(x))

    Examples
    ========

    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3
    >>> from sympy.abc import x
    >>> from sympy import Mul, cos, pi
    >>> TRmorrie(cos(x)*cos(2*x))
    sin(4*x)/(4*sin(x))
    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))
    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))

    Sometimes autosimplification will cause a power to be
    not recognized. e.g. in the following, cos(4*pi/7) automatically
    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are
    recognized:

    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))
    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))

    A touch by TR8 resolves the expression to a Rational

    >>> TR8(_)
    -1/8

    In this case, if eq is unsimplified, the answer is obtained
    directly:

    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)
    >>> TRmorrie(eq)
    1/16

    But if angles are made canonical with TR3 then the answer
    is not simplified without further work:

    >>> TR3(eq)
    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2
    >>> TRmorrie(_)
    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))
    >>> TR8(_)
    cos(7*pi/18)/(16*sin(pi/9))
    >>> TR3(_)
    1/16

    The original expression would have resolve to 1/16 directly with TR8,
    however:

    >>> TR8(eq)
    1/16

    References
    ==========

    http://en.wikipedia.org/wiki/Morrie%27s_law

    """
    # ... other code
```
### 5 - sympy/simplify/trigsimp.py:

Start line: 777, End line: 817

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
### 6 - sympy/functions/elementary/trigonometric.py:

Start line: 254, End line: 357

```python
class sin(TrigonometricFunction):

    @classmethod
    def eval(cls, arg):
        from sympy.calculus import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.Infinity or arg is S.NegativeInfinity:
                return AccumBounds(-1, 1)

        if isinstance(arg, AccumBounds):
            min, max = arg.min, arg.max
            d = floor(min/(2*S.Pi))
            if min is not S.NegativeInfinity:
                min = min - d*2*S.Pi
            if max is not S.Infinity:
                max = max - d*2*S.Pi
            if AccumBounds(min, max).intersection(FiniteSet(S.Pi/2, 5*S.Pi/2)) \
                    is not S.EmptySet and \
                    AccumBounds(min, max).intersection(FiniteSet(3*S.Pi/2,
                        7*S.Pi/2)) is not S.EmptySet:
                return AccumBounds(-1, 1)
            elif AccumBounds(min, max).intersection(FiniteSet(S.Pi/2, 5*S.Pi/2)) \
                    is not S.EmptySet:
                return AccumBounds(Min(sin(min), sin(max)), 1)
            elif AccumBounds(min, max).intersection(FiniteSet(3*S.Pi/2, 8*S.Pi/2)) \
                        is not S.EmptySet:
                return AccumBounds(-1, Max(sin(min), sin(max)))
            else:
                return AccumBounds(Min(sin(min), sin(max)),
                                Max(sin(min), sin(max)))

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return S.ImaginaryUnit * sinh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if (2*pi_coeff).is_integer:
                if pi_coeff.is_even:
                    return S.Zero
                elif pi_coeff.is_even is False:
                    return S.NegativeOne**(pi_coeff - S.Half)

            if not pi_coeff.is_Rational:
                narg = pi_coeff*S.Pi
                if narg != arg:
                    return cls(narg)
                return None

            # https://github.com/sympy/sympy/issues/6048
            # transform a sine to a cosine, to avoid redundant code
            if pi_coeff.is_Rational:
                x = pi_coeff % 2
                if x > 1:
                    return -cls((x % 1)*S.Pi)
                if 2*x > 1:
                    return cls((1 - x)*S.Pi)
                narg = ((pi_coeff + Rational(3, 2)) % 2)*S.Pi
                result = cos(narg)
                if not isinstance(result, cos):
                    return result
                if pi_coeff*S.Pi != arg:
                    return cls(pi_coeff*S.Pi)
                return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                return sin(m)*cos(x) + cos(m)*sin(x)

        if arg.func is asin:
            return arg.args[0]

        if arg.func is atan:
            x = arg.args[0]
            return x / sqrt(1 + x**2)

        if arg.func is atan2:
            y, x = arg.args
            return y / sqrt(x**2 + y**2)

        if arg.func is acos:
            x = arg.args[0]
            return sqrt(1 - x**2)

        if arg.func is acot:
            x = arg.args[0]
            return 1 / (sqrt(1 + 1 / x**2) * x)

        if arg.func is acsc:
            x = arg.args[0]
            return 1 / x

        if arg.func is asec:
            x = arg.args[0]
            return sqrt(1 - 1 / x**2)
```
### 7 - sympy/simplify/trigsimp.py:

Start line: 425, End line: 489

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

    try:
        return expr._eval_trigsimp(**opts)
    except AttributeError:
        pass

    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        recursive = opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'
    # ... other code
```
### 8 - sympy/simplify/trigsimp.py:

Start line: 589, End line: 650

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
    (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)
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
### 9 - sympy/functions/elementary/trigonometric.py:

Start line: 359, End line: 416

```python
class sin(TrigonometricFunction):

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p * x**2 / (n*(n - 1))
            else:
                return (-1)**(n//2) * x**(n)/factorial(n)

    def _eval_rewrite_as_exp(self, arg):
        I = S.ImaginaryUnit
        if isinstance(arg, TrigonometricFunction) or isinstance(arg, HyperbolicFunction):
            arg = arg.func(arg.args[0]).rewrite(exp)
        return (exp(arg*I) - exp(-arg*I)) / (2*I)

    def _eval_rewrite_as_Pow(self, arg):
        if arg.func is log:
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*x**-I / 2 - I*x**I /2

    def _eval_rewrite_as_cos(self, arg):
        return cos(arg - S.Pi / 2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    def _eval_rewrite_as_sincos(self, arg):
        return sin(arg)*cos(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg):
        cot_half = cot(S.Half*arg)
        return 2*cot_half/(1 + cot_half**2)

    def _eval_rewrite_as_pow(self, arg):
        return self.rewrite(cos).rewrite(pow)

    def _eval_rewrite_as_sqrt(self, arg):
        return self.rewrite(cos).rewrite(sqrt)

    def _eval_rewrite_as_csc(self, arg):
        return 1/csc(arg)

    def _eval_rewrite_as_sec(self, arg):
        return 1 / sec(arg - S.Pi / 2, evaluate=False)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        return (sin(re)*cosh(im), cos(re)*sinh(im))
```
### 10 - sympy/simplify/fu.py:

Start line: 2096, End line: 2137

```python
def hyper_as_trig(rv):
    """Return an expression containing hyperbolic functions in terms
    of trigonometric functions. Any trigonometric functions initially
    present are replaced with Dummy symbols and the function to undo
    the masking and the conversion back to hyperbolics is also returned. It
    should always be true that::

        t, f = hyper_as_trig(expr)
        expr == f(t)

    Examples
    ========

    >>> from sympy.simplify.fu import hyper_as_trig, fu
    >>> from sympy.abc import x
    >>> from sympy import cosh, sinh
    >>> eq = sinh(x)**2 + cosh(x)**2
    >>> t, f = hyper_as_trig(eq)
    >>> f(fu(t))
    cosh(2*x)

    References
    ==========

    http://en.wikipedia.org/wiki/Hyperbolic_function
    """
    from sympy.simplify.simplify import signsimp
    from sympy.simplify.radsimp import collect

    # mask off trig functions
    trigs = rv.atoms(TrigonometricFunction)
    reps = [(t, Dummy()) for t in trigs]
    masked = rv.xreplace(dict(reps))

    # get inversion substitutions in place
    reps = [(v, k) for k, v in reps]

    d = Dummy()

    return _osborne(masked, d), lambda x: collect(signsimp(
        _osbornei(x, d).xreplace(dict(reps))), S.ImaginaryUnit)
```
### 11 - sympy/simplify/trigsimp.py:

Start line: 285, End line: 350

```python
def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):

    def analyse_gens(gens, hints):
        # ... other code

        for key, val in trigdict.items():
            # We have now assembeled a dictionary. Its keys are common
            # arguments in trigonometric expressions, and values are lists of
            # pairs (fn, coeff). x0, (fn, coeff) in trigdict means that we
            # need to deal with fn(coeff*x0). We take the rational gcd of the
            # coeffs, call it ``gcd``. We then use x = x0/gcd as "base symbol",
            # all other arguments are integral multiples thereof.
            # We will build an ideal which works with sin(x), cos(x).
            # If hint tan is provided, also work with tan(x). Moreover, if
            # n > 1, also work with sin(k*x) for k <= n, and similarly for cos
            # (and tan if the hint is provided). Finally, any generators which
            # the ideal does not work with but we need to accomodate (either
            # because it was in expr or because it was provided as a hint)
            # we also build into the ideal.
            # This selection process is expressed in the list ``terms``.
            # build_ideal then generates the actual relations in our ideal,
            # from this list.
            fns = [x[1] for x in val]
            val = [x[0] for x in val]
            gcd = reduce(igcd, val)
            terms = [(fn, v/gcd) for (fn, v) in zip(fns, val)]
            fs = set(funcs + fns)
            for c, s, t in ([cos, sin, tan], [cosh, sinh, tanh]):
                if any(x in fs for x in (c, s, t)):
                    fs.add(c)
                    fs.add(s)
            for fn in fs:
                for k in range(1, n + 1):
                    terms.append((fn, k))
            extra = []
            for fn, v in terms:
                if fn == tan:
                    extra.append((sin, v))
                    extra.append((cos, v))
                if fn in [sin, cos] and tan in fs:
                    extra.append((tan, v))
                if fn == tanh:
                    extra.append((sinh, v))
                    extra.append((cosh, v))
                if fn in [sinh, cosh] and tanh in fs:
                    extra.append((tanh, v))
            terms.extend(extra)
            x = gcd*Mul(*key)
            r = build_ideal(x, terms)
            res.extend(r)
            newgens.extend(set(fn(v*x) for fn, v in terms))

        # Add generators for compound expressions from iterables
        for fn, args in iterables:
            if fn == tan:
                # Tan expressions are recovered from sin and cos.
                iterables.extend([(sin, args), (cos, args)])
            elif fn == tanh:
                # Tanh expressions are recovered from sihn and cosh.
                iterables.extend([(sinh, args), (cosh, args)])
            else:
                dummys = symbols('d:%i' % len(args), cls=Dummy)
                expr = fn( Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
                res.append(fn(Add(*args)) - expr)

        if myI in gens:
            res.append(myI**2 + 1)
            freegens.remove(myI)
            newgens.append(myI)

        return res, freegens, newgens
    # ... other code
```
### 12 - sympy/simplify/trigsimp.py:

Start line: 28, End line: 117

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
    hints, to indicate that than should be tried whenever cosine or sine are,
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
### 16 - sympy/simplify/trigsimp.py:

Start line: 967, End line: 1047

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
### 18 - sympy/simplify/trigsimp.py:

Start line: 736, End line: 776

```python
def _trigpats():
    global _trigpat
    a, b, c = symbols('a b c', cls=Wild)
    d = Wild('d', commutative=False)

    # for the simplifications like sinh/cosh -> tanh:
    # DO NOT REORDER THE FIRST 14 since these are assumed to be in this
    # order in _match_div_rewrite.
    matchers_division = (
        (a*sin(b)**c/cos(b)**c, a*tan(b)**c, sin(b), cos(b)),
        (a*tan(b)**c*cos(b)**c, a*sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*sin(b)**c, a*cos(b)**c, sin(b), cos(b)),
        (a*tan(b)**c/sin(b)**c, a/cos(b)**c, sin(b), cos(b)),
        (a*cot(b)**c/cos(b)**c, a/sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*tan(b)**c, a, sin(b), cos(b)),
        (a*(cos(b) + 1)**c*(cos(b) - 1)**c,
            a*(-sin(b)**2)**c, cos(b) + 1, cos(b) - 1),
        (a*(sin(b) + 1)**c*(sin(b) - 1)**c,
            a*(-cos(b)**2)**c, sin(b) + 1, sin(b) - 1),

        (a*sinh(b)**c/cosh(b)**c, a*tanh(b)**c, S.One, S.One),
        (a*tanh(b)**c*cosh(b)**c, a*sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*sinh(b)**c, a*cosh(b)**c, S.One, S.One),
        (a*tanh(b)**c/sinh(b)**c, a/cosh(b)**c, S.One, S.One),
        (a*coth(b)**c/cosh(b)**c, a/sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*tanh(b)**c, a, S.One, S.One),

        (c*(tanh(a) + tanh(b))/(1 + tanh(a)*tanh(b)),
            tanh(a + b)*c, S.One, S.One),
    )

    matchers_add = (
        (c*sin(a)*cos(b) + c*cos(a)*sin(b) + d, sin(a + b)*c + d),
        (c*cos(a)*cos(b) - c*sin(a)*sin(b) + d, cos(a + b)*c + d),
        (c*sin(a)*cos(b) - c*cos(a)*sin(b) + d, sin(a - b)*c + d),
        (c*cos(a)*cos(b) + c*sin(a)*sin(b) + d, cos(a - b)*c + d),
        (c*sinh(a)*cosh(b) + c*sinh(b)*cosh(a) + d, sinh(a + b)*c + d),
        (c*cosh(a)*cosh(b) + c*sinh(a)*sinh(b) + d, cosh(a + b)*c + d),
    )

    # for cos(x)**2 + sin(x)**2 -> 1
    # ... other code
```
### 23 - sympy/simplify/trigsimp.py:

Start line: 118, End line: 175

```python
def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    # TODO
    #  - preprocess by replacing everything by funcs we can handle
    # - optionally use cot instead of tan
    # - more intelligent hinting.
    #     For example, if the ideal is small, and we have sin(x), sin(y),
    #     add sin(x + y) automatically... ?
    # - algebraic numbers ...
    # - expressions of lowest degree are not distinguished properly
    #   e.g. 1 - sin(x)**2
    # - we could try to order the generators intelligently, so as to influence
    #   which monomials appear in the quotient basis

    # THEORY
    # ------
    # Ratsimpmodprime above can be used to "simplify" a rational function
    # modulo a prime ideal. "Simplify" mainly means finding an equivalent
    # expression of lower total degree.
    #
    # We intend to use this to simplify trigonometric functions. To do that,
    # we need to decide (a) which ring to use, and (b) modulo which ideal to
    # simplify. In practice, (a) means settling on a list of "generators"
    # a, b, c, ..., such that the fraction we want to simplify is a rational
    # function in a, b, c, ..., with coefficients in ZZ (integers).
    # (2) means that we have to decide what relations to impose on the
    # generators. There are two practical problems:
    #   (1) The ideal has to be *prime* (a technical term).
    #   (2) The relations have to be polynomials in the generators.
    #
    # We typically have two kinds of generators:
    # - trigonometric expressions, like sin(x), cos(5*x), etc
    # - "everything else", like gamma(x), pi, etc.
    #
    # Since this function is trigsimp, we will concentrate on what to do with
    # trigonometric expressions. We can also simplify hyperbolic expressions,
    # but the extensions should be clear.
    #
    # One crucial point is that all *other* generators really should behave
    # like indeterminates. In particular if (say) "I" is one of them, then
    # in fact I**2 + 1 = 0 and we may and will compute non-sensical
    # expressions. However, we can work with a dummy and add the relation
    # I**2 + 1 = 0 to our ideal, then substitute back in the end.
    #
    # Now regarding trigonometric generators. We split them into groups,
    # according to the argument of the trigonometric functions. We want to
    # organise this in such a way that most trigonometric identities apply in
    # the same group. For example, given sin(x), cos(2*x) and cos(y), we would
    # group as [sin(x), cos(2*x)] and [cos(y)].
    #
    # Our prime ideal will be built in three steps:
    # (1) For each group, compute a "geometrically prime" ideal of relations.
    #     Geometrically prime means that it generates a prime ideal in
    #     CC[gens], not just ZZ[gens].
    # (2) Take the union of all the generators of the ideals for all groups.
    #     By the geometric primality condition, this is still prime.
    # (3) Add further inter-group relations which preserve primality.
    #
    # Step (1) works as follows. We will isolate common factors in the
    # argument, so that all our generators are of the form sin(n*x), cos(n*x)
    # ... other code
```
### 31 - sympy/simplify/trigsimp.py:

Start line: 176, End line: 202

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
### 37 - sympy/simplify/trigsimp.py:

Start line: 912, End line: 965

```python
def _trigsimp(expr, deep=False):
    # protect the cache from non-trig patterns; we only allow
    # trig patterns to enter the cache
    if expr.has(*_trigs):
        return __trigsimp(expr, deep)
    return expr


@cacheit
def __trigsimp(expr, deep=False):
    """recursive helper for trigsimp"""
    from sympy.simplify.fu import TR10i

    if _trigpat is None:
        _trigpats()
    a, b, c, d, matchers_division, matchers_add, \
    matchers_identity, artifacts = _trigpat

    if expr.is_Mul:
        # do some simplifications like sin/cos -> tan:
        if not expr.is_commutative:
            com, nc = expr.args_cnc()
            expr = _trigsimp(Mul._from_args(com), deep)*Mul._from_args(nc)
        else:
            for i, (pattern, simp, ok1, ok2) in enumerate(matchers_division):
                if not _dotrig(expr, pattern):
                    continue

                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue

                # use SymPy matching instead
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue
                    # if "a" contains any of trig or hyperbolic funcs with
                    # argument "b" then skip the simplification
                    if any(w.args[0] == res[b] for w in res[a].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    # simplify and finish:
                    expr = simp.subs(res)
                    break  # process below
    # ... other code
```
