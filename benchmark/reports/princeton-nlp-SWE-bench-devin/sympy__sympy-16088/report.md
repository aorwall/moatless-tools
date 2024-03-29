# sympy__sympy-16088

| **sympy/sympy** | `b750e609ab48eed4fccc18617d57c8e8bfda662a` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | - |
| **Any found context length** | 9616 |
| **Avg pos** | 3.2 |
| **Min pos** | 6 |
| **Max pos** | 10 |
| **Top file pos** | 3 |
| **Missing snippets** | 15 |
| **Missing patch files** | 3 |


## Expected patch

```diff
diff --git a/sympy/core/exprtools.py b/sympy/core/exprtools.py
--- a/sympy/core/exprtools.py
+++ b/sympy/core/exprtools.py
@@ -1098,6 +1098,54 @@ def handle(a):
     return terms.func(*[handle(i) for i in terms.args])
 
 
+def _factor_sum_int(expr, **kwargs):
+    """Return Sum or Integral object with factors that are not
+    in the wrt variables removed. In cases where there are additive
+    terms in the function of the object that are independent, the
+    object will be separated into two objects.
+
+    Examples
+    ========
+
+    >>> from sympy import Sum, factor_terms
+    >>> from sympy.abc import x, y
+    >>> factor_terms(Sum(x + y, (x, 1, 3)))
+    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))
+    >>> factor_terms(Sum(x*y, (x, 1, 3)))
+    y*Sum(x, (x, 1, 3))
+
+    Notes
+    =====
+
+    If a function in the summand or integrand is replaced
+    with a symbol, then this simplification should not be
+    done or else an incorrect result will be obtained when
+    the symbol is replaced with an expression that depends
+    on the variables of summation/integration:
+
+    >>> eq = Sum(y, (x, 1, 3))
+    >>> factor_terms(eq).subs(y, x).doit()
+    3*x
+    >>> eq.subs(y, x).doit()
+    6
+    """
+    result = expr.function
+    if result == 0:
+        return S.Zero
+    limits = expr.limits
+
+    # get the wrt variables
+    wrt = set([i.args[0] for i in limits])
+
+    # factor out any common terms that are independent of wrt
+    f = factor_terms(result, **kwargs)
+    i, d = f.as_independent(*wrt)
+    if isinstance(f, Add):
+        return i * expr.func(1, *limits) + expr.func(d, *limits)
+    else:
+        return i * expr.func(d, *limits)
+
+
 def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
     """Remove common factors from terms in all arguments without
     changing the underlying structure of the expr. No expansion or
@@ -1153,7 +1201,7 @@ def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
     """
     def do(expr):
         from sympy.concrete.summations import Sum
-        from sympy.simplify.simplify import factor_sum
+        from sympy.integrals.integrals import Integral
         is_iterable = iterable(expr)
 
         if not isinstance(expr, Basic) or expr.is_Atom:
@@ -1169,8 +1217,10 @@ def do(expr):
                 return expr
             return expr.func(*newargs)
 
-        if isinstance(expr, Sum):
-            return factor_sum(expr, radical=radical, clear=clear, fraction=fraction, sign=sign)
+        if isinstance(expr, (Sum, Integral)):
+            return _factor_sum_int(expr,
+                radical=radical, clear=clear,
+                fraction=fraction, sign=sign)
 
         cont, p = expr.as_content_primitive(radical=radical, clear=clear)
         if p.is_Add:
diff --git a/sympy/integrals/integrals.py b/sympy/integrals/integrals.py
--- a/sympy/integrals/integrals.py
+++ b/sympy/integrals/integrals.py
@@ -1100,6 +1100,16 @@ def _eval_as_leading_term(self, x):
                 break
         return integrate(leading_term, *self.args[1:])
 
+    def _eval_simplify(self, ratio=1.7, measure=None, rational=False, inverse=False):
+        from sympy.core.exprtools import factor_terms
+        from sympy.simplify.simplify import simplify
+
+        expr = factor_terms(self)
+        kwargs = dict(ratio=ratio, measure=measure, rational=rational, inverse=inverse)
+        if isinstance(expr, Integral):
+            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
+        return expr.simplify(**kwargs)
+
     def as_sum(self, n=None, method="midpoint", evaluate=True):
         """
         Approximates a definite integral by a sum.
diff --git a/sympy/physics/continuum_mechanics/beam.py b/sympy/physics/continuum_mechanics/beam.py
--- a/sympy/physics/continuum_mechanics/beam.py
+++ b/sympy/physics/continuum_mechanics/beam.py
@@ -1530,7 +1530,7 @@ class Beam3D(Beam):
     is restricted.
 
     >>> from sympy.physics.continuum_mechanics.beam import Beam3D
-    >>> from sympy import symbols, simplify
+    >>> from sympy import symbols, simplify, collect
     >>> l, E, G, I, A = symbols('l, E, G, I, A')
     >>> b = Beam3D(l, E, G, I, A)
     >>> x, q, m = symbols('x, q, m')
@@ -1545,20 +1545,17 @@ class Beam3D(Beam):
     >>> b.solve_slope_deflection()
     >>> b.slope()
     [0, 0, l*x*(-l*q + 3*l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I)) + 3*m)/(6*E*I)
-    + q*x**3/(6*E*I) + x**2*(-l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I))
-    - m)/(2*E*I)]
+        + x**2*(-3*l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I)) - 3*m + q*x)/(6*E*I)]
     >>> dx, dy, dz = b.deflection()
-    >>> dx
-    0
-    >>> dz
-    0
-    >>> expectedy = (
-    ... -l**2*q*x**2/(12*E*I) + l**2*x**2*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(8*E*I*(A*G*l**2 + 12*E*I))
-    ... + l*m*x**2/(4*E*I) - l*x**3*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(12*E*I*(A*G*l**2 + 12*E*I)) - m*x**3/(6*E*I)
-    ... + q*x**4/(24*E*I) + l*x*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*A*G*(A*G*l**2 + 12*E*I)) - q*x**2/(2*A*G)
-    ... )
-    >>> simplify(dy - expectedy)
-    0
+    >>> dy = collect(simplify(dy), x)
+    >>> dx == dz == 0
+    True
+    >>> dy == (x*(12*A*E*G*I*l**3*q - 24*A*E*G*I*l**2*m + 144*E**2*I**2*l*q +
+    ...           x**3*(A**2*G**2*l**2*q + 12*A*E*G*I*q) +
+    ...           x**2*(-2*A**2*G**2*l**3*q - 24*A*E*G*I*l*q - 48*A*E*G*I*m) +
+    ...           x*(A**2*G**2*l**4*q + 72*A*E*G*I*l*m - 144*E**2*I**2*q)
+    ...           )/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
+    True
 
     References
     ==========
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -26,8 +26,7 @@
 from sympy.simplify.radsimp import radsimp, fraction
 from sympy.simplify.sqrtdenest import sqrtdenest
 from sympy.simplify.trigsimp import trigsimp, exptrigsimp
-from sympy.utilities.iterables import has_variety
-
+from sympy.utilities.iterables import has_variety, sift
 
 
 import mpmath
@@ -511,7 +510,10 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
     x belongs to the set where this relation is true. The default is
     False.
     """
+
     expr = sympify(expr)
+    kwargs = dict(ratio=ratio, measure=measure,
+        rational=rational, inverse=inverse)
 
     _eval_simplify = getattr(expr, '_eval_simplify', None)
     if _eval_simplify is not None:
@@ -521,7 +523,7 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
 
     from sympy.simplify.hyperexpand import hyperexpand
     from sympy.functions.special.bessel import BesselBase
-    from sympy import Sum, Product
+    from sympy import Sum, Product, Integral
 
     if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
         return expr
@@ -532,8 +534,7 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
             return expr
 
     if not isinstance(expr, (Add, Mul, Pow, ExpBase)):
-        return expr.func(*[simplify(x, ratio=ratio, measure=measure, rational=rational, inverse=inverse)
-                         for x in expr.args])
+        return expr.func(*[simplify(x, **kwargs) for x in expr.args])
 
     if not expr.is_commutative:
         expr = nc_simplify(expr)
@@ -590,7 +591,11 @@ def shorter(*choices):
         expr = combsimp(expr)
 
     if expr.has(Sum):
-        expr = sum_simplify(expr)
+        expr = sum_simplify(expr, **kwargs)
+
+    if expr.has(Integral):
+        expr = expr.xreplace(dict([
+            (i, factor_terms(i)) for i in expr.atoms(Integral)]))
 
     if expr.has(Product):
         expr = product_simplify(expr)
@@ -639,49 +644,36 @@ def shorter(*choices):
     return expr
 
 
-def sum_simplify(s):
+def sum_simplify(s, **kwargs):
     """Main function for Sum simplification"""
     from sympy.concrete.summations import Sum
     from sympy.core.function import expand
 
-    terms = Add.make_args(expand(s))
+    if not isinstance(s, Add):
+        s = s.xreplace(dict([(a, sum_simplify(a, **kwargs))
+            for a in s.atoms(Add) if a.has(Sum)]))
+    s = expand(s)
+    if not isinstance(s, Add):
+        return s
+
+    terms = s.args
     s_t = [] # Sum Terms
     o_t = [] # Other Terms
 
     for term in terms:
-        if isinstance(term, Mul):
-            other = 1
-            sum_terms = []
-
-            if not term.has(Sum):
-                o_t.append(term)
-                continue
-
-            mul_terms = Mul.make_args(term)
-            for mul_term in mul_terms:
-                if isinstance(mul_term, Sum):
-                    r = mul_term._eval_simplify()
-                    sum_terms.extend(Add.make_args(r))
-                else:
-                    other = other * mul_term
-            if len(sum_terms):
-                #some simplification may have happened
-                #use if so
-                s_t.append(Mul(*sum_terms) * other)
-            else:
-                o_t.append(other)
-        elif isinstance(term, Sum):
-            #as above, we need to turn this into an add list
-            r = term._eval_simplify()
-            s_t.extend(Add.make_args(r))
-        else:
+        sum_terms, other = sift(Mul.make_args(term),
+            lambda i: isinstance(i, Sum), binary=True)
+        if not sum_terms:
             o_t.append(term)
-
+            continue
+        other = [Mul(*other)]
+        s_t.append(Mul(*(other + [s._eval_simplify(**kwargs) for s in sum_terms])))
 
     result = Add(sum_combine(s_t), *o_t)
 
     return result
 
+
 def sum_combine(s_t):
     """Helper function for Sum simplification
 
@@ -690,7 +682,6 @@ def sum_combine(s_t):
     """
     from sympy.concrete.summations import Sum
 
-
     used = [False] * len(s_t)
 
     for method in range(2):
@@ -711,37 +702,32 @@ def sum_combine(s_t):
 
     return result
 
+
 def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
-    """Helper function for Sum simplification
+    """Return Sum with constant factors extracted.
 
-       if limits is specified, "self" is the inner part of a sum
+    If ``limits`` is specified then ``self`` is the summand; the other
+    keywords are passed to ``factor_terms``.
 
-       Returns the sum with constant factors brought outside
+    Examples
+    ========
+
+    >>> from sympy import Sum, Integral
+    >>> from sympy.abc import x, y
+    >>> from sympy.simplify.simplify import factor_sum
+    >>> s = Sum(x*y, (x, 1, 3))
+    >>> factor_sum(s)
+    y*Sum(x, (x, 1, 3))
+    >>> factor_sum(s.function, s.limits)
+    y*Sum(x, (x, 1, 3))
     """
-    from sympy.core.exprtools import factor_terms
+    # XXX deprecate in favor of direct call to factor_terms
     from sympy.concrete.summations import Sum
+    kwargs = dict(radical=radical, clear=clear,
+        fraction=fraction, sign=sign)
+    expr = Sum(self, *limits) if limits else self
+    return factor_terms(expr, **kwargs)
 
-    result = self.function if limits is None else self
-    limits = self.limits if limits is None else limits
-    #avoid any confusion w/ as_independent
-    if result == 0:
-        return S.Zero
-
-    #get the summation variables
-    sum_vars = set([limit.args[0] for limit in limits])
-
-    #finally we try to factor out any common terms
-    #and remove the from the sum if independent
-    retv = factor_terms(result, radical=radical, clear=clear, fraction=fraction, sign=sign)
-    #avoid doing anything bad
-    if not result.is_commutative:
-        return Sum(result, *limits)
-
-    i, d = retv.as_independent(*sum_vars)
-    if isinstance(retv, Add):
-        return i * Sum(1, *limits) + Sum(d, *limits)
-    else:
-        return i * Sum(d, *limits)
 
 def sum_add(self, other, method=0):
     """Helper function for Sum simplification"""
diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py
--- a/sympy/solvers/ode.py
+++ b/sympy/solvers/ode.py
@@ -4132,11 +4132,14 @@ def unreplace(eq, var):
     var = func.args[0]
     subs_eqn = replace(eq, var)
     try:
-        solns = solve(subs_eqn, func)
+        # turn off simplification to protect Integrals that have
+        # _t instead of fx in them and would otherwise factor
+        # as t_*Integral(1, x)
+        solns = solve(subs_eqn, func, simplify=False)
     except NotImplementedError:
         solns = []
 
-    solns = [unreplace(soln, var) for soln in solns]
+    solns = [simplify(unreplace(soln, var)) for soln in solns]
     solns = [Equality(func, soln) for soln in solns]
     return {'var':var, 'solutions':solns}
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/exprtools.py | 1101 | 1101 | - | - | -
| sympy/core/exprtools.py | 1156 | 1156 | - | - | -
| sympy/core/exprtools.py | 1172 | 1173 | - | - | -
| sympy/integrals/integrals.py | 1103 | 1103 | - | 3 | -
| sympy/physics/continuum_mechanics/beam.py | 1533 | 1533 | - | - | -
| sympy/physics/continuum_mechanics/beam.py | 1548 | 1561 | - | - | -
| sympy/simplify/simplify.py | 29 | 30 | - | 5 | -
| sympy/simplify/simplify.py | 514 | 514 | - | 5 | -
| sympy/simplify/simplify.py | 524 | 524 | - | 5 | -
| sympy/simplify/simplify.py | 535 | 536 | - | 5 | -
| sympy/simplify/simplify.py | 593 | 593 | - | 5 | -
| sympy/simplify/simplify.py | 642 | 679 | 6 | 5 | 9616
| sympy/simplify/simplify.py | 693 | 693 | - | 5 | -
| sympy/simplify/simplify.py | 714 | 743 | 10 | 5 | 17364
| sympy/solvers/ode.py | 4135 | 4139 | - | - | -


## Problem Statement

```
Using Simplify in Integral will pull out the constant term
<!-- Your title above should be a short description of what
was changed. Do not include the issue number in the title. -->

#### References to other Issues or PRs
<!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
format, e.g. "Fixes #1234". See
https://github.com/blog/1506-closing-issues-via-pull-requests . Please also
write a comment on that issue linking back to this pull request once it is
open. -->
Fixes##15965

#### Brief description of what is fixed or changed
Using simplify in `Sum `pulls out the constant term(independent term) outside the summation but this property is not present in `Integral` 
Example-
\`\`\`
>>> Sum(x*y, (x, 1, n)).simplify()
    n    
   __    
   \ `   
y*  )   x
   /_,   
  x = 1  
>>> Integral(x*y, (x, 1, n)).simplify()
  n       
  /       
 |        
 |  x*y dx
 |        
/         
1
\`\`\`
Now it is working -
\`\`\`
In [4]: (Integral(x*y-z,x)).simplify()                                              
Out[4]: 
  ⌠          ⌠     
y⋅⎮ x dx - z⋅⎮ 1 dx
  ⌡          ⌡     

In [5]:  Integral(x*y, (x, 1, n)).simplify()                                        
Out[5]: 
  n     
  ⌠     
y⋅⎮ x dx
  ⌡     
  1   

\`\`\`
#### Other comments
previous issue about this -#7971
and they talked about `doit`  by using simplify .
I don't have any idea about adding `doit`method in simplify.
#### Release Notes

<!-- Write the release notes for this release below. See
https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more `inIntegeralformation`
on how to write release notes. The bot will check your release notes
automatically to see if they are formatted correctly. -->

<!-- BEGIN RELEASE NOTES -->
- simplify
  -  simplify now pulls independent factors out of integrals
<!-- END RELEASE NOTES -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/expr.py | 3228 | 3312| 767 | 767 | 30054 | 
| 2 | 2 sympy/integrals/rubi/utility_function.py | 7247 | 7343| 1411 | 2178 | 115213 | 
| 3 | 2 sympy/integrals/rubi/utility_function.py | 5992 | 6306| 6118 | 8296 | 115213 | 
| 4 | **3 sympy/integrals/integrals.py** | 572 | 646| 640 | 8936 | 129123 | 
| 5 | 4 sympy/concrete/summations.py | 263 | 311| 409 | 9345 | 139848 | 
| **-> 6 <-** | **5 sympy/simplify/simplify.py** | 642 | 683| 271 | 9616 | 155987 | 
| 7 | 5 sympy/integrals/rubi/utility_function.py | 6520 | 7028| 6205 | 15821 | 155987 | 
| 8 | **5 sympy/integrals/integrals.py** | 824 | 904| 836 | 16657 | 155987 | 
| 9 | **5 sympy/simplify/simplify.py** | 746 | 790| 429 | 17086 | 155987 | 
| **-> 10 <-** | **5 sympy/simplify/simplify.py** | 714 | 744| 278 | 17364 | 155987 | 
| 11 | **5 sympy/integrals/integrals.py** | 144 | 246| 981 | 18345 | 155987 | 
| 12 | **5 sympy/simplify/simplify.py** | 381 | 513| 1374 | 19719 | 155987 | 
| 13 | **5 sympy/integrals/integrals.py** | 905 | 1072| 1365 | 21084 | 155987 | 
| 14 | **5 sympy/integrals/integrals.py** | 440 | 570| 1419 | 22503 | 155987 | 
| 15 | **5 sympy/integrals/integrals.py** | 1074 | 1101| 255 | 22758 | 155987 | 
| 16 | 6 sympy/integrals/transforms.py | 279 | 287| 151 | 22909 | 172751 | 
| 17 | **6 sympy/integrals/integrals.py** | 1198 | 1232| 418 | 23327 | 172751 | 
| 18 | **6 sympy/simplify/simplify.py** | 1104 | 1168| 688 | 24015 | 172751 | 
| 19 | 6 sympy/integrals/transforms.py | 938 | 961| 211 | 24226 | 172751 | 
| 20 | 6 sympy/concrete/summations.py | 29 | 156| 1344 | 25570 | 172751 | 
| 21 | 6 sympy/integrals/transforms.py | 241 | 277| 355 | 25925 | 172751 | 
| 22 | 7 sympy/integrals/manualintegrate.py | 502 | 590| 821 | 26746 | 188831 | 


## Missing Patch Files

 * 1: sympy/core/exprtools.py
 * 2: sympy/integrals/integrals.py
 * 3: sympy/physics/continuum_mechanics/beam.py
 * 4: sympy/simplify/simplify.py
 * 5: sympy/solvers/ode.py

## Patch

```diff
diff --git a/sympy/core/exprtools.py b/sympy/core/exprtools.py
--- a/sympy/core/exprtools.py
+++ b/sympy/core/exprtools.py
@@ -1098,6 +1098,54 @@ def handle(a):
     return terms.func(*[handle(i) for i in terms.args])
 
 
+def _factor_sum_int(expr, **kwargs):
+    """Return Sum or Integral object with factors that are not
+    in the wrt variables removed. In cases where there are additive
+    terms in the function of the object that are independent, the
+    object will be separated into two objects.
+
+    Examples
+    ========
+
+    >>> from sympy import Sum, factor_terms
+    >>> from sympy.abc import x, y
+    >>> factor_terms(Sum(x + y, (x, 1, 3)))
+    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))
+    >>> factor_terms(Sum(x*y, (x, 1, 3)))
+    y*Sum(x, (x, 1, 3))
+
+    Notes
+    =====
+
+    If a function in the summand or integrand is replaced
+    with a symbol, then this simplification should not be
+    done or else an incorrect result will be obtained when
+    the symbol is replaced with an expression that depends
+    on the variables of summation/integration:
+
+    >>> eq = Sum(y, (x, 1, 3))
+    >>> factor_terms(eq).subs(y, x).doit()
+    3*x
+    >>> eq.subs(y, x).doit()
+    6
+    """
+    result = expr.function
+    if result == 0:
+        return S.Zero
+    limits = expr.limits
+
+    # get the wrt variables
+    wrt = set([i.args[0] for i in limits])
+
+    # factor out any common terms that are independent of wrt
+    f = factor_terms(result, **kwargs)
+    i, d = f.as_independent(*wrt)
+    if isinstance(f, Add):
+        return i * expr.func(1, *limits) + expr.func(d, *limits)
+    else:
+        return i * expr.func(d, *limits)
+
+
 def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
     """Remove common factors from terms in all arguments without
     changing the underlying structure of the expr. No expansion or
@@ -1153,7 +1201,7 @@ def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
     """
     def do(expr):
         from sympy.concrete.summations import Sum
-        from sympy.simplify.simplify import factor_sum
+        from sympy.integrals.integrals import Integral
         is_iterable = iterable(expr)
 
         if not isinstance(expr, Basic) or expr.is_Atom:
@@ -1169,8 +1217,10 @@ def do(expr):
                 return expr
             return expr.func(*newargs)
 
-        if isinstance(expr, Sum):
-            return factor_sum(expr, radical=radical, clear=clear, fraction=fraction, sign=sign)
+        if isinstance(expr, (Sum, Integral)):
+            return _factor_sum_int(expr,
+                radical=radical, clear=clear,
+                fraction=fraction, sign=sign)
 
         cont, p = expr.as_content_primitive(radical=radical, clear=clear)
         if p.is_Add:
diff --git a/sympy/integrals/integrals.py b/sympy/integrals/integrals.py
--- a/sympy/integrals/integrals.py
+++ b/sympy/integrals/integrals.py
@@ -1100,6 +1100,16 @@ def _eval_as_leading_term(self, x):
                 break
         return integrate(leading_term, *self.args[1:])
 
+    def _eval_simplify(self, ratio=1.7, measure=None, rational=False, inverse=False):
+        from sympy.core.exprtools import factor_terms
+        from sympy.simplify.simplify import simplify
+
+        expr = factor_terms(self)
+        kwargs = dict(ratio=ratio, measure=measure, rational=rational, inverse=inverse)
+        if isinstance(expr, Integral):
+            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
+        return expr.simplify(**kwargs)
+
     def as_sum(self, n=None, method="midpoint", evaluate=True):
         """
         Approximates a definite integral by a sum.
diff --git a/sympy/physics/continuum_mechanics/beam.py b/sympy/physics/continuum_mechanics/beam.py
--- a/sympy/physics/continuum_mechanics/beam.py
+++ b/sympy/physics/continuum_mechanics/beam.py
@@ -1530,7 +1530,7 @@ class Beam3D(Beam):
     is restricted.
 
     >>> from sympy.physics.continuum_mechanics.beam import Beam3D
-    >>> from sympy import symbols, simplify
+    >>> from sympy import symbols, simplify, collect
     >>> l, E, G, I, A = symbols('l, E, G, I, A')
     >>> b = Beam3D(l, E, G, I, A)
     >>> x, q, m = symbols('x, q, m')
@@ -1545,20 +1545,17 @@ class Beam3D(Beam):
     >>> b.solve_slope_deflection()
     >>> b.slope()
     [0, 0, l*x*(-l*q + 3*l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I)) + 3*m)/(6*E*I)
-    + q*x**3/(6*E*I) + x**2*(-l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I))
-    - m)/(2*E*I)]
+        + x**2*(-3*l*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*(A*G*l**2 + 12*E*I)) - 3*m + q*x)/(6*E*I)]
     >>> dx, dy, dz = b.deflection()
-    >>> dx
-    0
-    >>> dz
-    0
-    >>> expectedy = (
-    ... -l**2*q*x**2/(12*E*I) + l**2*x**2*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(8*E*I*(A*G*l**2 + 12*E*I))
-    ... + l*m*x**2/(4*E*I) - l*x**3*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(12*E*I*(A*G*l**2 + 12*E*I)) - m*x**3/(6*E*I)
-    ... + q*x**4/(24*E*I) + l*x*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*A*G*(A*G*l**2 + 12*E*I)) - q*x**2/(2*A*G)
-    ... )
-    >>> simplify(dy - expectedy)
-    0
+    >>> dy = collect(simplify(dy), x)
+    >>> dx == dz == 0
+    True
+    >>> dy == (x*(12*A*E*G*I*l**3*q - 24*A*E*G*I*l**2*m + 144*E**2*I**2*l*q +
+    ...           x**3*(A**2*G**2*l**2*q + 12*A*E*G*I*q) +
+    ...           x**2*(-2*A**2*G**2*l**3*q - 24*A*E*G*I*l*q - 48*A*E*G*I*m) +
+    ...           x*(A**2*G**2*l**4*q + 72*A*E*G*I*l*m - 144*E**2*I**2*q)
+    ...           )/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
+    True
 
     References
     ==========
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -26,8 +26,7 @@
 from sympy.simplify.radsimp import radsimp, fraction
 from sympy.simplify.sqrtdenest import sqrtdenest
 from sympy.simplify.trigsimp import trigsimp, exptrigsimp
-from sympy.utilities.iterables import has_variety
-
+from sympy.utilities.iterables import has_variety, sift
 
 
 import mpmath
@@ -511,7 +510,10 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
     x belongs to the set where this relation is true. The default is
     False.
     """
+
     expr = sympify(expr)
+    kwargs = dict(ratio=ratio, measure=measure,
+        rational=rational, inverse=inverse)
 
     _eval_simplify = getattr(expr, '_eval_simplify', None)
     if _eval_simplify is not None:
@@ -521,7 +523,7 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
 
     from sympy.simplify.hyperexpand import hyperexpand
     from sympy.functions.special.bessel import BesselBase
-    from sympy import Sum, Product
+    from sympy import Sum, Product, Integral
 
     if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
         return expr
@@ -532,8 +534,7 @@ def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
             return expr
 
     if not isinstance(expr, (Add, Mul, Pow, ExpBase)):
-        return expr.func(*[simplify(x, ratio=ratio, measure=measure, rational=rational, inverse=inverse)
-                         for x in expr.args])
+        return expr.func(*[simplify(x, **kwargs) for x in expr.args])
 
     if not expr.is_commutative:
         expr = nc_simplify(expr)
@@ -590,7 +591,11 @@ def shorter(*choices):
         expr = combsimp(expr)
 
     if expr.has(Sum):
-        expr = sum_simplify(expr)
+        expr = sum_simplify(expr, **kwargs)
+
+    if expr.has(Integral):
+        expr = expr.xreplace(dict([
+            (i, factor_terms(i)) for i in expr.atoms(Integral)]))
 
     if expr.has(Product):
         expr = product_simplify(expr)
@@ -639,49 +644,36 @@ def shorter(*choices):
     return expr
 
 
-def sum_simplify(s):
+def sum_simplify(s, **kwargs):
     """Main function for Sum simplification"""
     from sympy.concrete.summations import Sum
     from sympy.core.function import expand
 
-    terms = Add.make_args(expand(s))
+    if not isinstance(s, Add):
+        s = s.xreplace(dict([(a, sum_simplify(a, **kwargs))
+            for a in s.atoms(Add) if a.has(Sum)]))
+    s = expand(s)
+    if not isinstance(s, Add):
+        return s
+
+    terms = s.args
     s_t = [] # Sum Terms
     o_t = [] # Other Terms
 
     for term in terms:
-        if isinstance(term, Mul):
-            other = 1
-            sum_terms = []
-
-            if not term.has(Sum):
-                o_t.append(term)
-                continue
-
-            mul_terms = Mul.make_args(term)
-            for mul_term in mul_terms:
-                if isinstance(mul_term, Sum):
-                    r = mul_term._eval_simplify()
-                    sum_terms.extend(Add.make_args(r))
-                else:
-                    other = other * mul_term
-            if len(sum_terms):
-                #some simplification may have happened
-                #use if so
-                s_t.append(Mul(*sum_terms) * other)
-            else:
-                o_t.append(other)
-        elif isinstance(term, Sum):
-            #as above, we need to turn this into an add list
-            r = term._eval_simplify()
-            s_t.extend(Add.make_args(r))
-        else:
+        sum_terms, other = sift(Mul.make_args(term),
+            lambda i: isinstance(i, Sum), binary=True)
+        if not sum_terms:
             o_t.append(term)
-
+            continue
+        other = [Mul(*other)]
+        s_t.append(Mul(*(other + [s._eval_simplify(**kwargs) for s in sum_terms])))
 
     result = Add(sum_combine(s_t), *o_t)
 
     return result
 
+
 def sum_combine(s_t):
     """Helper function for Sum simplification
 
@@ -690,7 +682,6 @@ def sum_combine(s_t):
     """
     from sympy.concrete.summations import Sum
 
-
     used = [False] * len(s_t)
 
     for method in range(2):
@@ -711,37 +702,32 @@ def sum_combine(s_t):
 
     return result
 
+
 def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
-    """Helper function for Sum simplification
+    """Return Sum with constant factors extracted.
 
-       if limits is specified, "self" is the inner part of a sum
+    If ``limits`` is specified then ``self`` is the summand; the other
+    keywords are passed to ``factor_terms``.
 
-       Returns the sum with constant factors brought outside
+    Examples
+    ========
+
+    >>> from sympy import Sum, Integral
+    >>> from sympy.abc import x, y
+    >>> from sympy.simplify.simplify import factor_sum
+    >>> s = Sum(x*y, (x, 1, 3))
+    >>> factor_sum(s)
+    y*Sum(x, (x, 1, 3))
+    >>> factor_sum(s.function, s.limits)
+    y*Sum(x, (x, 1, 3))
     """
-    from sympy.core.exprtools import factor_terms
+    # XXX deprecate in favor of direct call to factor_terms
     from sympy.concrete.summations import Sum
+    kwargs = dict(radical=radical, clear=clear,
+        fraction=fraction, sign=sign)
+    expr = Sum(self, *limits) if limits else self
+    return factor_terms(expr, **kwargs)
 
-    result = self.function if limits is None else self
-    limits = self.limits if limits is None else limits
-    #avoid any confusion w/ as_independent
-    if result == 0:
-        return S.Zero
-
-    #get the summation variables
-    sum_vars = set([limit.args[0] for limit in limits])
-
-    #finally we try to factor out any common terms
-    #and remove the from the sum if independent
-    retv = factor_terms(result, radical=radical, clear=clear, fraction=fraction, sign=sign)
-    #avoid doing anything bad
-    if not result.is_commutative:
-        return Sum(result, *limits)
-
-    i, d = retv.as_independent(*sum_vars)
-    if isinstance(retv, Add):
-        return i * Sum(1, *limits) + Sum(d, *limits)
-    else:
-        return i * Sum(d, *limits)
 
 def sum_add(self, other, method=0):
     """Helper function for Sum simplification"""
diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py
--- a/sympy/solvers/ode.py
+++ b/sympy/solvers/ode.py
@@ -4132,11 +4132,14 @@ def unreplace(eq, var):
     var = func.args[0]
     subs_eqn = replace(eq, var)
     try:
-        solns = solve(subs_eqn, func)
+        # turn off simplification to protect Integrals that have
+        # _t instead of fx in them and would otherwise factor
+        # as t_*Integral(1, x)
+        solns = solve(subs_eqn, func, simplify=False)
     except NotImplementedError:
         solns = []
 
-    solns = [unreplace(soln, var) for soln in solns]
+    solns = [simplify(unreplace(soln, var)) for soln in solns]
     solns = [Equality(func, soln) for soln in solns]
     return {'var':var, 'solutions':solns}
 

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_exprtools.py b/sympy/core/tests/test_exprtools.py
--- a/sympy/core/tests/test_exprtools.py
+++ b/sympy/core/tests/test_exprtools.py
@@ -288,10 +288,11 @@ def test_factor_terms():
     assert factor_terms(e, sign=False) == e
     assert factor_terms(exp(-4*x - 2) - x) == -x + exp(Mul(-2, 2*x + 1, evaluate=False))
 
-    # sum tests
-    assert factor_terms(Sum(x, (y, 1, 10))) == x * Sum(1, (y, 1, 10))
-    assert factor_terms(Sum(x, (y, 1, 10)) + x) == x * (1 + Sum(1, (y, 1, 10)))
-    assert factor_terms(Sum(x*y + x*y**2, (y, 1, 10))) == x*Sum(y*(y + 1), (y, 1, 10))
+    # sum/integral tests
+    for F in (Sum, Integral):
+        assert factor_terms(F(x, (y, 1, 10))) == x * F(1, (y, 1, 10))
+        assert factor_terms(F(x, (y, 1, 10)) + x) == x * (1 + F(1, (y, 1, 10)))
+        assert factor_terms(F(x*y + x*y**2, (y, 1, 10))) == x*F(y*(y + 1), (y, 1, 10))
 
 
 def test_xreplace():
diff --git a/sympy/physics/continuum_mechanics/tests/test_beam.py b/sympy/physics/continuum_mechanics/tests/test_beam.py
--- a/sympy/physics/continuum_mechanics/tests/test_beam.py
+++ b/sympy/physics/continuum_mechanics/tests/test_beam.py
@@ -503,17 +503,14 @@ def test_Beam3D():
 
     assert b.shear_force() == [0, -q*x, 0]
     assert b.bending_moment() == [0, 0, -m*x + q*x**2/2]
-    expected_deflection = (-l**2*q*x**2/(12*E*I) + l**2*x**2*(A*G*l*(l*q - 2*m)
-            + 12*E*I*q)/(8*E*I*(A*G*l**2 + 12*E*I)) + l*m*x**2/(4*E*I)
-            - l*x**3*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(12*E*I*(A*G*l**2 + 12*E*I))
-            - m*x**3/(6*E*I) + q*x**4/(24*E*I)
-            + l*x*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(2*A*G*(A*G*l**2 + 12*E*I))
-            - q*x**2/(2*A*G)
-            )
+    expected_deflection = (x*(A*G*q*x**3/4 + A*G*x**2*(-l*(A*G*l*(l*q - 2*m) +
+        12*E*I*q)/(A*G*l**2 + 12*E*I)/2 - m) + 3*E*I*l*(A*G*l*(l*q - 2*m) +
+        12*E*I*q)/(A*G*l**2 + 12*E*I) + x*(-A*G*l**2*q/2 +
+        3*A*G*l**2*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(A*G*l**2 + 12*E*I)/4 +
+        3*A*G*l*m/2 - 3*E*I*q))/(6*A*E*G*I))
     dx, dy, dz = b.deflection()
     assert dx == dz == 0
-    assert simplify(dy - expected_deflection) == 0  # == doesn't work
-
+    assert dy == expected_deflection
 
     b2 = Beam3D(30, E, G, I, A, x)
     b2.apply_load(50, start=0, order=0, dir="y")
@@ -524,12 +521,12 @@ def test_Beam3D():
     assert b2.reaction_loads == {R1: -750, R2: -750}
 
     b2.solve_slope_deflection()
-    assert b2.slope() == [0, 0, 25*x**3/(3*E*I) - 375*x**2/(E*I) + 3750*x/(E*I)]
-    expected_deflection = (25*x**4/(12*E*I) - 125*x**3/(E*I) + 1875*x**2/(E*I)
-                        - 25*x**2/(A*G) + 750*x/(A*G))
+    assert b2.slope() == [0, 0, x**2*(50*x - 2250)/(6*E*I) + 3750*x/(E*I)]
+    expected_deflection = (x*(25*A*G*x**3/2 - 750*A*G*x**2 + 4500*E*I +
+        15*x*(750*A*G - 10*E*I))/(6*A*E*G*I))
     dx, dy, dz = b2.deflection()
     assert dx == dz == 0
-    assert simplify(dy - expected_deflection) == 0  # == doesn't work
+    assert dy == expected_deflection
 
     # Test for solve_for_reaction_loads
     b3 = Beam3D(30, E, G, I, A, x)
diff --git a/sympy/simplify/tests/test_simplify.py b/sympy/simplify/tests/test_simplify.py
--- a/sympy/simplify/tests/test_simplify.py
+++ b/sympy/simplify/tests/test_simplify.py
@@ -793,3 +793,12 @@ def _check(expr, simplified, deep=True, matrix=True):
     assert nc_simplify(expr) == (1-c)**-1
     # commutative expressions should be returned without an error
     assert nc_simplify(2*x**2) == 2*x**2
+
+def test_issue_15965():
+    A = Sum(z*x**y, (x, 1, a))
+    anew = z*Sum(x**y, (x, 1, a))
+    B = Integral(x*y, x)
+    bnew = y*Integral(x, x)
+    assert simplify(A + B) == anew + bnew
+    assert simplify(A) == anew
+    assert simplify(B) == bnew

```


## Code snippets

### 1 - sympy/core/expr.py:

Start line: 3228, End line: 3312

```python
class Expr(Basic, EvalfMixin):

    ###########################################################################
    ################### GLOBAL ACTION VERB WRAPPER METHODS ####################
    ###########################################################################

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals import integrate
        return integrate(self, *args, **kwargs)

    def simplify(self, ratio=1.7, measure=None, rational=False, inverse=False):
        """See the simplify function in sympy.simplify"""
        from sympy.simplify import simplify
        from sympy.core.function import count_ops
        measure = measure or count_ops
        return simplify(self, ratio, measure)

    def nsimplify(self, constants=[], tolerance=None, full=False):
        """See the nsimplify function in sympy.simplify"""
        from sympy.simplify import nsimplify
        return nsimplify(self, constants, tolerance, full)

    def separate(self, deep=False, force=False):
        """See the separate function in sympy.simplify"""
        from sympy.core.function import expand_power_base
        return expand_power_base(self, deep=deep, force=force)

    def collect(self, syms, func=None, evaluate=True, exact=False, distribute_order_term=True):
        """See the collect function in sympy.simplify"""
        from sympy.simplify import collect
        return collect(self, syms, func, evaluate, exact, distribute_order_term)

    def together(self, *args, **kwargs):
        """See the together function in sympy.polys"""
        from sympy.polys import together
        return together(self, *args, **kwargs)

    def apart(self, x=None, **args):
        """See the apart function in sympy.polys"""
        from sympy.polys import apart
        return apart(self, x, **args)

    def ratsimp(self):
        """See the ratsimp function in sympy.simplify"""
        from sympy.simplify import ratsimp
        return ratsimp(self)

    def trigsimp(self, **args):
        """See the trigsimp function in sympy.simplify"""
        from sympy.simplify import trigsimp
        return trigsimp(self, **args)

    def radsimp(self, **kwargs):
        """See the radsimp function in sympy.simplify"""
        from sympy.simplify import radsimp
        return radsimp(self, **kwargs)

    def powsimp(self, *args, **kwargs):
        """See the powsimp function in sympy.simplify"""
        from sympy.simplify import powsimp
        return powsimp(self, *args, **kwargs)

    def combsimp(self):
        """See the combsimp function in sympy.simplify"""
        from sympy.simplify import combsimp
        return combsimp(self)

    def gammasimp(self):
        """See the gammasimp function in sympy.simplify"""
        from sympy.simplify import gammasimp
        return gammasimp(self)

    def factor(self, *gens, **args):
        """See the factor() function in sympy.polys.polytools"""
        from sympy.polys import factor
        return factor(self, *gens, **args)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions import refine
        return refine(self, assumption)

    def cancel(self, *gens, **args):
        """See the cancel function in sympy.polys"""
        from sympy.polys import cancel
        return cancel(self, *gens, **args)
```
### 2 - sympy/integrals/rubi/utility_function.py:

Start line: 7247, End line: 7343

```python
def _ExpandIntegrand():
    # ... other code
    pattern30 = Pattern(UtilityOperator(((u_**WC('n', S(1))*WC('g', S(1)) + WC('f', S(0)))*WC('e', S(1)) + WC('d', S(0)))/(u_**WC('j', S(1))*WC('c', S(1)) + u_**WC('n', S(1))*WC('b', S(1)) + WC('a', S(0))), x_), cons3, cons4, cons5, cons6, cons7, cons8, cons9, cons14, cons23, cons43)
    rule30 = ReplacementRule(pattern30, With30)
    def With31(v, x, u):
        lst = CoefficientList(u, x)
        i = Symbol('i')
        return x**Exponent(u, x)*lst[-1]/v + Sum_doit(x**(i - 1)*Part(lst, i), List(i, 1, Exponent(u, x)))/v
    pattern31 = Pattern(UtilityOperator(u_/v_, x_), cons19, cons47, cons48, cons49)
    rule31 = ReplacementRule(pattern31, With31)
    pattern32 = Pattern(UtilityOperator(u_/v_, x_), cons19, cons47, cons50)
    def replacement32(v, x, u):
        return PolynomialDivide(u, v, x)
    rule32 = ReplacementRule(pattern32, replacement32)
    pattern33 = Pattern(UtilityOperator(u_*(x_*WC('a', S(1)))**p_, x_), cons51, cons19)
    def replacement33(x, a, u, p):
        return ExpandToSum((a*x)**p, u, x)
    rule33 = ReplacementRule(pattern33, replacement33)
    pattern34 = Pattern(UtilityOperator(v_**p_*WC('u', S(1)), x_), cons51)
    def replacement34(v, x, u, p):
        return ExpandIntegrand(NormalizeIntegrand(v**p, x), u, x)
    rule34 = ReplacementRule(pattern34, replacement34)
    pattern35 = Pattern(UtilityOperator(u_, x_))
    def replacement35(x, u):
        return ExpandExpression(u, x)
    rule35 = ReplacementRule(pattern35, replacement35)
    return [ rule2,rule3, rule4, rule5, rule6, rule7, rule8, rule10, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35]

def _RemoveContentAux():
    def cons_f1(b, a):
        return IntegersQ(a, b)

    cons1 = CustomConstraint(cons_f1)

    def cons_f2(b, a):
        return Equal(a + b, S(0))

    cons2 = CustomConstraint(cons_f2)

    def cons_f3(m):
        return RationalQ(m)

    cons3 = CustomConstraint(cons_f3)

    def cons_f4(m, n):
        return RationalQ(m, n)

    cons4 = CustomConstraint(cons_f4)

    def cons_f5(m, n):
        return GreaterEqual(-m + n, S(0))

    cons5 = CustomConstraint(cons_f5)

    def cons_f6(a, x):
        return FreeQ(a, x)

    cons6 = CustomConstraint(cons_f6)

    def cons_f7(m, n, p):
        return RationalQ(m, n, p)

    cons7 = CustomConstraint(cons_f7)

    def cons_f8(m, p):
        return GreaterEqual(-m + p, S(0))

    cons8 = CustomConstraint(cons_f8)

    pattern1 = Pattern(UtilityOperator(a_**m_*WC('u', S(1)) + b_*WC('v', S(1)), x_), cons1, cons2, cons3)
    def replacement1(v, x, a, u, m, b):
        return If(Greater(m, S(1)), RemoveContentAux(a**(m + S(-1))*u - v, x), RemoveContentAux(-a**(-m + S(1))*v + u, x))
    rule1 = ReplacementRule(pattern1, replacement1)
    pattern2 = Pattern(UtilityOperator(a_**WC('m', S(1))*WC('u', S(1)) + a_**WC('n', S(1))*WC('v', S(1)), x_), cons6, cons4, cons5)
    def replacement2(n, v, x, u, m, a):
        return RemoveContentAux(a**(-m + n)*v + u, x)
    rule2 = ReplacementRule(pattern2, replacement2)
    pattern3 = Pattern(UtilityOperator(a_**WC('m', S(1))*WC('u', S(1)) + a_**WC('n', S(1))*WC('v', S(1)) + a_**WC('p', S(1))*WC('w', S(1)), x_), cons6, cons7, cons5, cons8)
    def replacement3(n, v, x, p, u, w, m, a):
        return RemoveContentAux(a**(-m + n)*v + a**(-m + p)*w + u, x)
    rule3 = ReplacementRule(pattern3, replacement3)
    pattern4 = Pattern(UtilityOperator(u_, x_))
    def replacement4(u, x):
        return If(And(SumQ(u), NegQ(First(u))), -u, u)
    rule4 = ReplacementRule(pattern4, replacement4)
    return [rule1, rule2, rule3, rule4, ]

IntHide = Int
Log = rubi_log
Null = None
if matchpy:
    RemoveContentAux_replacer = ManyToOneReplacer(* _RemoveContentAux())
    ExpandIntegrand_rules = _ExpandIntegrand()
    TrigSimplifyAux_replacer = _TrigSimplifyAux()
    SimplifyAntiderivative_replacer = _SimplifyAntiderivative()
    SimplifyAntiderivativeSum_replacer = _SimplifyAntiderivativeSum()
    FixSimplify_rules = _FixSimplify()
    SimpFixFactor_replacer = _SimpFixFactor()
```
### 3 - sympy/integrals/rubi/utility_function.py:

Start line: 5992, End line: 6306

```python
@doctest_depends_on(modules=('matchpy',))
def _FixSimplify():
    Plus = Add
    def cons_f1(n):
        return OddQ(n)
    cons1 = CustomConstraint(cons_f1)

    def cons_f2(m):
        return RationalQ(m)
    cons2 = CustomConstraint(cons_f2)

    def cons_f3(n):
        return FractionQ(n)
    cons3 = CustomConstraint(cons_f3)

    def cons_f4(u):
        return SqrtNumberSumQ(u)
    cons4 = CustomConstraint(cons_f4)

    def cons_f5(v):
        return SqrtNumberSumQ(v)
    cons5 = CustomConstraint(cons_f5)

    def cons_f6(u):
        return PositiveQ(u)
    cons6 = CustomConstraint(cons_f6)

    def cons_f7(v):
        return PositiveQ(v)
    cons7 = CustomConstraint(cons_f7)

    def cons_f8(v):
        return SqrtNumberSumQ(S(1)/v)
    cons8 = CustomConstraint(cons_f8)

    def cons_f9(m):
        return IntegerQ(m)
    cons9 = CustomConstraint(cons_f9)

    def cons_f10(u):
        return NegativeQ(u)
    cons10 = CustomConstraint(cons_f10)

    def cons_f11(n, m, a, b):
        return RationalQ(a, b, m, n)
    cons11 = CustomConstraint(cons_f11)

    def cons_f12(a):
        return Greater(a, S(0))
    cons12 = CustomConstraint(cons_f12)

    def cons_f13(b):
        return Greater(b, S(0))
    cons13 = CustomConstraint(cons_f13)

    def cons_f14(p):
        return PositiveIntegerQ(p)
    cons14 = CustomConstraint(cons_f14)

    def cons_f15(p):
        return IntegerQ(p)
    cons15 = CustomConstraint(cons_f15)

    def cons_f16(p, n):
        return Greater(-n + p, S(0))
    cons16 = CustomConstraint(cons_f16)

    def cons_f17(a, b):
        return SameQ(a + b, S(0))
    cons17 = CustomConstraint(cons_f17)

    def cons_f18(n):
        return Not(IntegerQ(n))
    cons18 = CustomConstraint(cons_f18)

    def cons_f19(c, a, b, d):
        return ZeroQ(-a*d + b*c)
    cons19 = CustomConstraint(cons_f19)

    def cons_f20(a):
        return Not(RationalQ(a))
    cons20 = CustomConstraint(cons_f20)

    def cons_f21(t):
        return IntegerQ(t)
    cons21 = CustomConstraint(cons_f21)

    def cons_f22(n, m):
        return RationalQ(m, n)
    cons22 = CustomConstraint(cons_f22)

    def cons_f23(n, m):
        return Inequality(S(0), Less, m, LessEqual, n)
    cons23 = CustomConstraint(cons_f23)

    def cons_f24(p, n, m):
        return RationalQ(m, n, p)
    cons24 = CustomConstraint(cons_f24)

    def cons_f25(p, n, m):
        return Inequality(S(0), Less, m, LessEqual, n, LessEqual, p)
    cons25 = CustomConstraint(cons_f25)

    def cons_f26(p, n, m, q):
        return Inequality(S(0), Less, m, LessEqual, n, LessEqual, p, LessEqual, q)
    cons26 = CustomConstraint(cons_f26)

    def cons_f27(w):
        return Not(RationalQ(w))
    cons27 = CustomConstraint(cons_f27)

    def cons_f28(n):
        return Less(n, S(0))
    cons28 = CustomConstraint(cons_f28)

    def cons_f29(n, w, v):
        return ZeroQ(v + w**(-n))
    cons29 = CustomConstraint(cons_f29)

    def cons_f30(n):
        return IntegerQ(n)
    cons30 = CustomConstraint(cons_f30)

    def cons_f31(w, v):
        return ZeroQ(v + w)
    cons31 = CustomConstraint(cons_f31)

    def cons_f32(p, n):
        return IntegerQ(n/p)
    cons32 = CustomConstraint(cons_f32)

    def cons_f33(w, v):
        return ZeroQ(v - w)
    cons33 = CustomConstraint(cons_f33)

    def cons_f34(p, n):
        return IntegersQ(n, n/p)
    cons34 = CustomConstraint(cons_f34)

    def cons_f35(a):
        return AtomQ(a)
    cons35 = CustomConstraint(cons_f35)

    def cons_f36(b):
        return AtomQ(b)
    cons36 = CustomConstraint(cons_f36)

    pattern1 = Pattern(UtilityOperator((w_ + Complex(S(0), b_)*WC('v', S(1)))**WC('n', S(1))*Complex(S(0), a_)*WC('u', S(1))), cons1)
    def replacement1(n, u, w, v, a, b):
        return (S(-1))**(n/S(2) + S(1)/2)*a*u*FixSimplify((b*v - w*Complex(S(0), S(1)))**n)
    rule1 = ReplacementRule(pattern1, replacement1)
    def With2(m, n, u, w, v):
        z = u**(m/GCD(m, n))*v**(n/GCD(m, n))
        if Or(AbsurdNumberQ(z), SqrtNumberSumQ(z)):
            return True
        return False
    pattern2 = Pattern(UtilityOperator(u_**WC('m', S(1))*v_**n_*WC('w', S(1))), cons2, cons3, cons4, cons5, cons6, cons7, CustomConstraint(With2))
    def replacement2(m, n, u, w, v):
        z = u**(m/GCD(m, n))*v**(n/GCD(m, n))
        return FixSimplify(w*z**GCD(m, n))
    rule2 = ReplacementRule(pattern2, replacement2)
    def With3(m, n, u, w, v):
        z = u**(m/GCD(m, -n))*v**(n/GCD(m, -n))
        if Or(AbsurdNumberQ(z), SqrtNumberSumQ(z)):
            return True
        return False
    pattern3 = Pattern(UtilityOperator(u_**WC('m', S(1))*v_**n_*WC('w', S(1))), cons2, cons3, cons4, cons8, cons6, cons7, CustomConstraint(With3))
    def replacement3(m, n, u, w, v):
        z = u**(m/GCD(m, -n))*v**(n/GCD(m, -n))
        return FixSimplify(w*z**GCD(m, -n))
    rule3 = ReplacementRule(pattern3, replacement3)
    def With4(m, n, u, w, v):
        z = v**(n/GCD(m, n))*(-u)**(m/GCD(m, n))
        if Or(AbsurdNumberQ(z), SqrtNumberSumQ(z)):
            return True
        return False
    pattern4 = Pattern(UtilityOperator(u_**WC('m', S(1))*v_**n_*WC('w', S(1))), cons9, cons3, cons4, cons5, cons10, cons7, CustomConstraint(With4))
    def replacement4(m, n, u, w, v):
        z = v**(n/GCD(m, n))*(-u)**(m/GCD(m, n))
        return FixSimplify(-w*z**GCD(m, n))
    rule4 = ReplacementRule(pattern4, replacement4)
    def With5(m, n, u, w, v):
        z = v**(n/GCD(m, -n))*(-u)**(m/GCD(m, -n))
        if Or(AbsurdNumberQ(z), SqrtNumberSumQ(z)):
            return True
        return False
    pattern5 = Pattern(UtilityOperator(u_**WC('m', S(1))*v_**n_*WC('w', S(1))), cons9, cons3, cons4, cons8, cons10, cons7, CustomConstraint(With5))
    def replacement5(m, n, u, w, v):
        z = v**(n/GCD(m, -n))*(-u)**(m/GCD(m, -n))
        return FixSimplify(-w*z**GCD(m, -n))
    rule5 = ReplacementRule(pattern5, replacement5)
    def With6(p, m, n, u, w, v, a, b):
        c = a**(m/p)*b**n
        if RationalQ(c):
            return True
        return False
    pattern6 = Pattern(UtilityOperator(a_**m_*(b_**n_*WC('v', S(1)) + u_)**WC('p', S(1))*WC('w', S(1))), cons11, cons12, cons13, cons14, CustomConstraint(With6))
    def replacement6(p, m, n, u, w, v, a, b):
        c = a**(m/p)*b**n
        return FixSimplify(w*(a**(m/p)*u + c*v)**p)
    rule6 = ReplacementRule(pattern6, replacement6)
    pattern7 = Pattern(UtilityOperator(a_**WC('m', S(1))*(a_**n_*WC('u', S(1)) + b_**WC('p', S(1))*WC('v', S(1)))*WC('w', S(1))), cons2, cons3, cons15, cons16, cons17)
    def replacement7(p, m, n, u, w, v, a, b):
        return FixSimplify(a**(m + n)*w*((S(-1))**p*a**(-n + p)*v + u))
    rule7 = ReplacementRule(pattern7, replacement7)
    def With8(m, d, n, w, c, a, b):
        q = b/d
        if FreeQ(q, Plus):
            return True
        return False
    pattern8 = Pattern(UtilityOperator((a_ + b_)**WC('m', S(1))*(c_ + d_)**n_*WC('w', S(1))), cons9, cons18, cons19, CustomConstraint(With8))
    def replacement8(m, d, n, w, c, a, b):
        q = b/d
        return FixSimplify(q**m*w*(c + d)**(m + n))
    rule8 = ReplacementRule(pattern8, replacement8)
    pattern9 = Pattern(UtilityOperator((a_**WC('m', S(1))*WC('u', S(1)) + a_**WC('n', S(1))*WC('v', S(1)))**WC('t', S(1))*WC('w', S(1))), cons20, cons21, cons22, cons23)
    def replacement9(m, n, u, w, v, a, t):
        return FixSimplify(a**(m*t)*w*(a**(-m + n)*v + u)**t)
    rule9 = ReplacementRule(pattern9, replacement9)
    pattern10 = Pattern(UtilityOperator((a_**WC('m', S(1))*WC('u', S(1)) + a_**WC('n', S(1))*WC('v', S(1)) + a_**WC('p', S(1))*WC('z', S(1)))**WC('t', S(1))*WC('w', S(1))), cons20, cons21, cons24, cons25)
    def replacement10(p, m, n, u, w, v, a, z, t):
        return FixSimplify(a**(m*t)*w*(a**(-m + n)*v + a**(-m + p)*z + u)**t)
    rule10 = ReplacementRule(pattern10, replacement10)
    pattern11 = Pattern(UtilityOperator((a_**WC('m', S(1))*WC('u', S(1)) + a_**WC('n', S(1))*WC('v', S(1)) + a_**WC('p', S(1))*WC('z', S(1)) + a_**WC('q', S(1))*WC('y', S(1)))**WC('t', S(1))*WC('w', S(1))), cons20, cons21, cons24, cons26)
    def replacement11(p, m, n, u, q, w, v, a, z, y, t):
        return FixSimplify(a**(m*t)*w*(a**(-m + n)*v + a**(-m + p)*z + a**(-m + q)*y + u)**t)
    rule11 = ReplacementRule(pattern11, replacement11)
    pattern12 = Pattern(UtilityOperator((sqrt(v_)*WC('b', S(1)) + sqrt(v_)*WC('c', S(1)) + sqrt(v_)*WC('d', S(1)) + sqrt(v_)*WC('a', S(1)) + WC('u', S(0)))*WC('w', S(1))))
    def replacement12(d, u, w, v, c, a, b):
        return FixSimplify(w*(u + sqrt(v)*FixSimplify(a + b + c + d)))
    rule12 = ReplacementRule(pattern12, replacement12)
    pattern13 = Pattern(UtilityOperator((sqrt(v_)*WC('b', S(1)) + sqrt(v_)*WC('c', S(1)) + sqrt(v_)*WC('a', S(1)) + WC('u', S(0)))*WC('w', S(1))))
    def replacement13(u, w, v, c, a, b):
        return FixSimplify(w*(u + sqrt(v)*FixSimplify(a + b + c)))
    rule13 = ReplacementRule(pattern13, replacement13)
    pattern14 = Pattern(UtilityOperator((sqrt(v_)*WC('b', S(1)) + sqrt(v_)*WC('a', S(1)) + WC('u', S(0)))*WC('w', S(1))))
    def replacement14(u, w, v, a, b):
        return FixSimplify(w*(u + sqrt(v)*FixSimplify(a + b)))
    rule14 = ReplacementRule(pattern14, replacement14)
    pattern15 = Pattern(UtilityOperator(v_**m_*w_**n_*WC('u', S(1))), cons2, cons27, cons3, cons28, cons29)
    def replacement15(m, n, u, w, v):
        return -FixSimplify(u*v**(m + S(-1)))
    rule15 = ReplacementRule(pattern15, replacement15)
    pattern16 = Pattern(UtilityOperator(v_**m_*w_**WC('n', S(1))*WC('u', S(1))), cons2, cons27, cons30, cons31)
    def replacement16(m, n, u, w, v):
        return (S(-1))**n*FixSimplify(u*v**(m + n))
    rule16 = ReplacementRule(pattern16, replacement16)
    pattern17 = Pattern(UtilityOperator(w_**WC('n', S(1))*(-v_**WC('p', S(1)))**m_*WC('u', S(1))), cons2, cons27, cons32, cons33)
    def replacement17(p, m, n, u, w, v):
        return (S(-1))**(n/p)*FixSimplify(u*(-v**p)**(m + n/p))
    rule17 = ReplacementRule(pattern17, replacement17)
    pattern18 = Pattern(UtilityOperator(w_**WC('n', S(1))*(-v_**WC('p', S(1)))**m_*WC('u', S(1))), cons2, cons27, cons34, cons31)
    def replacement18(p, m, n, u, w, v):
        return (S(-1))**(n + n/p)*FixSimplify(u*(-v**p)**(m + n/p))
    rule18 = ReplacementRule(pattern18, replacement18)
    pattern19 = Pattern(UtilityOperator((a_ - b_)**WC('m', S(1))*(a_ + b_)**WC('m', S(1))*WC('u', S(1))), cons9, cons35, cons36)
    def replacement19(m, u, a, b):
        return u*(a**S(2) - b**S(2))**m
    rule19 = ReplacementRule(pattern19, replacement19)
    pattern20 = Pattern(UtilityOperator((S(729)*c - e*(-S(20)*e + S(540)))**WC('m', S(1))*WC('u', S(1))), cons2)
    def replacement20(m, u):
        return u*(a*e**S(2) - b*d*e + c*d**S(2))**m
    rule20 = ReplacementRule(pattern20, replacement20)
    pattern21 = Pattern(UtilityOperator((S(729)*c + e*(S(20)*e + S(-540)))**WC('m', S(1))*WC('u', S(1))), cons2)
    def replacement21(m, u):
        return u*(a*e**S(2) - b*d*e + c*d**S(2))**m
    rule21 = ReplacementRule(pattern21, replacement21)
    pattern22 = Pattern(UtilityOperator(u_))
    def replacement22(u):
        return u
    rule22 = ReplacementRule(pattern22, replacement22)
    return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, ]

@doctest_depends_on(modules=('matchpy',))
def FixSimplify(expr):
    if isinstance(expr, (list, tuple, TupleArg)):
        return [replace_all(UtilityOperator(i), FixSimplify_rules) for i in expr]
    return replace_all(UtilityOperator(expr), FixSimplify_rules)

@doctest_depends_on(modules=('matchpy',))
def _SimplifyAntiderivativeSum():
    replacer = ManyToOneReplacer()

    pattern1 = Pattern(UtilityOperator(Add(Mul(Log(Add(a_, Mul(WC('b', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('A', S(1))), Mul(Log(Cos(u_)), WC('B', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A, n: ZeroQ(Add(Mul(n, A), Mul(S(1), B)))))
    rule1 = ReplacementRule(pattern1, lambda n, x, v, b, B, A, u, a : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(a, Pow(Cos(u), n)), Mul(b, Pow(Sin(u), n))), x)))))
    replacer.add(rule1)

    pattern2 = Pattern(UtilityOperator(Add(Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('b', S(1))), a_)), WC('A', S(1))), Mul(Log(Sin(u_)), WC('B', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A, n: ZeroQ(Add(Mul(n, A), Mul(S(1), B)))))
    rule2 = ReplacementRule(pattern2, lambda n, x, v, b, B, A, a, u : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(a, Pow(Sin(u), n)), Mul(b, Pow(Cos(u), n))), x)))))
    replacer.add(rule2)

    pattern3 = Pattern(UtilityOperator(Add(Mul(Log(Add(a_, Mul(WC('b', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('A', S(1))), Mul(Log(Add(c_, Mul(WC('d', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('B', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A: ZeroQ(Add(A, B))))
    rule3 = ReplacementRule(pattern3, lambda n, x, v, b, A, B, u, c, d, a : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(a, Pow(Cos(u), n)), Mul(b, Pow(Sin(u), n))), x))), Mul(B, Log(RemoveContent(Add(Mul(c, Pow(Cos(u), n)), Mul(d, Pow(Sin(u), n))), x)))))
    replacer.add(rule3)

    pattern4 = Pattern(UtilityOperator(Add(Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('b', S(1))), a_)), WC('A', S(1))), Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('d', S(1))), c_)), WC('B', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A: ZeroQ(Add(A, B))))
    rule4 = ReplacementRule(pattern4, lambda n, x, v, b, A, B, c, a, d, u : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(b, Pow(Cos(u), n)), Mul(a, Pow(Sin(u), n))), x))), Mul(B, Log(RemoveContent(Add(Mul(d, Pow(Cos(u), n)), Mul(c, Pow(Sin(u), n))), x)))))
    replacer.add(rule4)

    pattern5 = Pattern(UtilityOperator(Add(Mul(Log(Add(a_, Mul(WC('b', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('A', S(1))), Mul(Log(Add(c_, Mul(WC('d', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('B', S(1))), Mul(Log(Add(e_, Mul(WC('f', S(1)), Pow(Tan(u_), WC('n', S(1)))))), WC('C', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda C, x: FreeQ(C, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A, C: ZeroQ(Add(A, B, C))))
    rule5 = ReplacementRule(pattern5, lambda n, e, x, v, b, A, B, u, c, f, d, a, C : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(a, Pow(Cos(u), n)), Mul(b, Pow(Sin(u), n))), x))), Mul(B, Log(RemoveContent(Add(Mul(c, Pow(Cos(u), n)), Mul(d, Pow(Sin(u), n))), x))), Mul(C, Log(RemoveContent(Add(Mul(e, Pow(Cos(u), n)), Mul(f, Pow(Sin(u), n))), x)))))
    replacer.add(rule5)

    pattern6 = Pattern(UtilityOperator(Add(Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('b', S(1))), a_)), WC('A', S(1))), Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('d', S(1))), c_)), WC('B', S(1))), Mul(Log(Add(Mul(Pow(Cot(u_), WC('n', S(1))), WC('f', S(1))), e_)), WC('C', S(1))), WC('v', S(0))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda A, x: FreeQ(A, x)), CustomConstraint(lambda B, x: FreeQ(B, x)), CustomConstraint(lambda C, x: FreeQ(C, x)), CustomConstraint(lambda n: IntegerQ(n)), CustomConstraint(lambda B, A, C: ZeroQ(Add(A, B, C))))
    rule6 = ReplacementRule(pattern6, lambda n, e, x, v, b, A, B, c, a, f, d, u, C : Add(SimplifyAntiderivativeSum(v, x), Mul(A, Log(RemoveContent(Add(Mul(b, Pow(Cos(u), n)), Mul(a, Pow(Sin(u), n))), x))), Mul(B, Log(RemoveContent(Add(Mul(d, Pow(Cos(u), n)), Mul(c, Pow(Sin(u), n))), x))), Mul(C, Log(RemoveContent(Add(Mul(f, Pow(Cos(u), n)), Mul(e, Pow(Sin(u), n))), x)))))
    replacer.add(rule6)

    return replacer

@doctest_depends_on(modules=('matchpy',))
def SimplifyAntiderivativeSum(expr, x):
    r = SimplifyAntiderivativeSum_replacer.replace(UtilityOperator(expr, x))
    if isinstance(r, UtilityOperator):
        return expr
    return r
```
### 4 - sympy/integrals/integrals.py:

Start line: 572, End line: 646

```python
class Integral(AddWithLimits):

    def doit(self, **hints):
        for xab in self.limits:
            if len(xab) == 1:
            # ... other code

            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*([function] + [xab])).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            else:
                if len(xab) == 1:
                    function = antideriv
                else:
                    if len(xab) == 3:
                        x, a, b = xab
                    elif len(xab) == 2:
                        x, b = xab
                        a = None
                    else:
                        raise NotImplementedError

                    if deep:
                        if isinstance(a, Basic):
                            a = a.doit(**hints)
                        if isinstance(b, Basic):
                            b = b.doit(**hints)

                    if antideriv.is_Poly:
                        gens = list(antideriv.gens)
                        gens.remove(x)

                        antideriv = antideriv.as_expr()

                        function = antideriv._eval_interval(x, a, b)
                        function = Poly(function, *gens)
                    else:
                        def is_indef_int(g, x):
                            return (isinstance(g, Integral) and
                                    any(i == (x,) for i in g.limits))

                        def eval_factored(f, x, a, b):
                            # _eval_interval for integrals with
                            # (constant) factors
                            # a single indefinite integral is assumed
                            args = []
                            for g in Mul.make_args(f):
                                if is_indef_int(g, x):
                                    args.append(g._eval_interval(x, a, b))
                                else:
                                    args.append(g)
                            return Mul(*args)

                        integrals, others, piecewises = [], [], []
                        for f in Add.make_args(antideriv):
                            if any(is_indef_int(g, x)
                                   for g in Mul.make_args(f)):
                                integrals.append(f)
                            elif any(isinstance(g, Piecewise)
                                     for g in Mul.make_args(f)):
                                piecewises.append(piecewise_fold(f))
                            else:
                                others.append(f)
                        uneval = Add(*[eval_factored(f, x, a, b)
                                       for f in integrals])
                        try:
                            evalued = Add(*others)._eval_interval(x, a, b)
                            evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                            function = uneval + evalued + evalued_pw
                        except NotImplementedError:
                            # This can happen if _eval_interval depends in a
                            # complicated way on limits that cannot be computed
                            undone_limits.append(xab)
                            function = self.func(*([function] + [xab]))
                            factored_function = function.factor()
                            if not isinstance(factored_function, Integral):
                                function = factored_function
        return function
```
### 5 - sympy/concrete/summations.py:

Start line: 263, End line: 311

```python
class Sum(AddWithLimits, ExprWithIntLimits):

    def _eval_difference_delta(self, n, step):
        k, _, upper = self.args[-1]
        new_upper = upper.subs(n, n + step)

        if len(self.args) == 2:
            f = self.args[0]
        else:
            f = self.func(*self.args[:-1])

        return Sum(f, (k, upper + 1, new_upper)).doit()

    def _eval_simplify(self, ratio=1.7, measure=None, rational=False, inverse=False):
        from sympy.simplify.simplify import factor_sum, sum_combine
        from sympy.core.function import expand
        from sympy.core.mul import Mul

        # split the function into adds
        terms = Add.make_args(expand(self.function))
        s_t = [] # Sum Terms
        o_t = [] # Other Terms

        for term in terms:
            if term.has(Sum):
                # if there is an embedded sum here
                # it is of the form x * (Sum(whatever))
                # hence we make a Mul out of it, and simplify all interior sum terms
                subterms = Mul.make_args(expand(term))
                out_terms = []
                for subterm in subterms:
                    # go through each term
                    if isinstance(subterm, Sum):
                        # if it's a sum, simplify it
                        out_terms.append(subterm._eval_simplify())
                    else:
                        # otherwise, add it as is
                        out_terms.append(subterm)

                # turn it back into a Mul
                s_t.append(Mul(*out_terms))
            else:
                o_t.append(term)

        # next try to combine any interior sums for further simplification
        result = Add(sum_combine(s_t), *o_t)

        return factor_sum(result, limits=self.limits)

    def _eval_summation(self, f, x):
        return None
```
### 6 - sympy/simplify/simplify.py:

Start line: 642, End line: 683

```python
def sum_simplify(s):
    """Main function for Sum simplification"""
    from sympy.concrete.summations import Sum
    from sympy.core.function import expand

    terms = Add.make_args(expand(s))
    s_t = [] # Sum Terms
    o_t = [] # Other Terms

    for term in terms:
        if isinstance(term, Mul):
            other = 1
            sum_terms = []

            if not term.has(Sum):
                o_t.append(term)
                continue

            mul_terms = Mul.make_args(term)
            for mul_term in mul_terms:
                if isinstance(mul_term, Sum):
                    r = mul_term._eval_simplify()
                    sum_terms.extend(Add.make_args(r))
                else:
                    other = other * mul_term
            if len(sum_terms):
                #some simplification may have happened
                #use if so
                s_t.append(Mul(*sum_terms) * other)
            else:
                o_t.append(other)
        elif isinstance(term, Sum):
            #as above, we need to turn this into an add list
            r = term._eval_simplify()
            s_t.extend(Add.make_args(r))
        else:
            o_t.append(term)


    result = Add(sum_combine(s_t), *o_t)

    return result
```
### 7 - sympy/integrals/rubi/utility_function.py:

Start line: 6520, End line: 7028

```python
@doctest_depends_on(modules=('matchpy',))
def _TrigSimplifyAux():
    # ... other code
    rule13 = ReplacementRule(pattern13, lambda u, v, b, a : Mul(u, Add(Mul(S(1), Pow(a, S(-1))), Mul(S(-1), Mul(Sin(v), Pow(b, S(-1)))))))
    replacer.add(rule13)

    pattern14 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(tan(v_), WC('n', S(1))), Pow(Add(a_, Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule14 = ReplacementRule(pattern14, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Cot(v), n))), S(-1))))
    replacer.add(rule14)

    pattern15 = Pattern(UtilityOperator(Mul(Pow(cot(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule15 = ReplacementRule(pattern15, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Tan(v), n))), S(-1))))
    replacer.add(rule15)

    pattern16 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(sec(v_), WC('n', S(1))), Pow(Add(a_, Mul(WC('b', S(1)), Pow(sec(v_), WC('n', S(1))))), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule16 = ReplacementRule(pattern16, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Cos(v), n))), S(-1))))
    replacer.add(rule16)

    pattern17 = Pattern(UtilityOperator(Mul(Pow(csc(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule17 = ReplacementRule(pattern17, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Sin(v), n))), S(-1))))
    replacer.add(rule17)

    pattern18 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(a_, Mul(WC('b', S(1)), Pow(sec(v_), WC('n', S(1))))), S(-1)), Pow(tan(v_), WC('n', S(1))))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule18 = ReplacementRule(pattern18, lambda n, a, u, v, b : Mul(u, Mul(Pow(Sin(v), n), Pow(Add(b, Mul(a, Pow(Cos(v), n))), S(-1)))))
    replacer.add(rule18)

    pattern19 = Pattern(UtilityOperator(Mul(Pow(cot(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule19 = ReplacementRule(pattern19, lambda n, a, u, v, b : Mul(u, Mul(Pow(Cos(v), n), Pow(Add(b, Mul(a, Pow(Sin(v), n))), S(-1)))))
    replacer.add(rule19)

    pattern20 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(WC('a', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule20 = ReplacementRule(pattern20, lambda n, a, p, u, v, b : Mul(u, Pow(Sec(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Sin(v), n))), p)))
    replacer.add(rule20)

    pattern21 = Pattern(UtilityOperator(Mul(Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('a', S(1))), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule21 = ReplacementRule(pattern21, lambda n, a, p, u, v, b : Mul(u, Pow(Csc(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Cos(v), n))), p)))
    replacer.add(rule21)

    pattern22 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(WC('b', S(1)), Pow(sin(v_), WC('n', S(1)))), Mul(WC('a', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule22 = ReplacementRule(pattern22, lambda n, a, p, u, v, b : Mul(u, Pow(Tan(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Cos(v), n))), p)))
    replacer.add(rule22)

    pattern23 = Pattern(UtilityOperator(Mul(Pow(Add(Mul(Pow(cot(v_), WC('n', S(1))), WC('a', S(1))), Mul(Pow(cos(v_), WC('n', S(1))), WC('b', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule23 = ReplacementRule(pattern23, lambda n, a, p, u, v, b : Mul(u, Pow(Cot(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Sin(v), n))), p)))
    replacer.add(rule23)

    pattern24 = Pattern(UtilityOperator(Mul(Pow(cos(v_), WC('m', S(1))), WC('u', S(1)), Pow(Add(WC('a', S(0)), Mul(WC('c', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule24 = ReplacementRule(pattern24, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Cos(v), Add(m, Mul(S(-1), Mul(n, p)))), Pow(Add(c, Mul(b, Pow(Sin(v), n)), Mul(a, Pow(Cos(v), n))), p)))
    replacer.add(rule24)

    pattern25 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(sec(v_), WC('m', S(1))), Pow(Add(WC('a', S(0)), Mul(WC('c', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule25 = ReplacementRule(pattern25, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Sec(v), Add(m, Mul(n, p))), Pow(Add(c, Mul(b, Pow(Sin(v), n)), Mul(a, Pow(Cos(v), n))), p)))
    replacer.add(rule25)

    pattern26 = Pattern(UtilityOperator(Mul(Pow(Add(WC('a', S(0)), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), Mul(Pow(csc(v_), WC('n', S(1))), WC('c', S(1)))), WC('p', S(1))), WC('u', S(1)), Pow(sin(v_), WC('m', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule26 = ReplacementRule(pattern26, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Sin(v), Add(m, Mul(S(-1), Mul(n, p)))), Pow(Add(c, Mul(b, Pow(Cos(v), n)), Mul(a, Pow(Sin(v), n))), p)))
    replacer.add(rule26)

    pattern27 = Pattern(UtilityOperator(Mul(Pow(csc(v_), WC('m', S(1))), Pow(Add(WC('a', S(0)), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), Mul(Pow(csc(v_), WC('n', S(1))), WC('c', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule27 = ReplacementRule(pattern27, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Csc(v), Add(m, Mul(n, p))), Pow(Add(c, Mul(b, Pow(Cos(v), n)), Mul(a, Pow(Sin(v), n))), p)))
    replacer.add(rule27)

    pattern28 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('m', S(1))), WC('a', S(1))), Mul(WC('b', S(1)), Pow(sin(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule28 = ReplacementRule(pattern28, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Cos(v), S('2')), Pow(Pow(Sin(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Sin(v), Add(m, n)))), Pow(Pow(Sin(v), m), S(-1))), p))))
    replacer.add(rule28)

    pattern29 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(cos(v_), WC('n', S(1))), WC('b', S(1))), Mul(WC('a', S(1)), Pow(sec(v_), WC('m', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule29 = ReplacementRule(pattern29, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Sin(v), S('2')), Pow(Pow(Cos(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Cos(v), Add(m, n)))), Pow(Pow(Cos(v), m), S(-1))), p))))
    replacer.add(rule29)

    pattern30 = Pattern(UtilityOperator(u_))
    rule30 = ReplacementRule(pattern30, lambda u : u)
    replacer.add(rule30)

    return replacer

@doctest_depends_on(modules=('matchpy',))
def TrigSimplifyAux(expr):
    return TrigSimplifyAux_replacer.replace(UtilityOperator(expr))

def Cancel(expr):
    return cancel(expr)

class Util_Part(Function):
    def doit(self):
        i = Simplify(self.args[0])
        if len(self.args) > 2 :
            lst = list(self.args[1:])
        else:
            lst = self.args[1]
        if isinstance(i, (int, Integer)):
            if isinstance(lst, list):
                return lst[i - 1]
            elif AtomQ(lst):
                return lst
            return lst.args[i - 1]
        else:
            return self

def Part(lst, i): #see i = -1
    if isinstance(lst, list):
        return Util_Part(i, *lst).doit()
    return Util_Part(i, lst).doit()

def PolyLog(n, p, z=None):
    return polylog(n, p)

def D(f, x):
    try:
        return f.diff(x)
    except ValueError:
        return Function('D')(f, x)

def IntegralFreeQ(u):
    return FreeQ(u, Integral)

def Dist(u, v, x):
    #Dist(u,v) returns the sum of u times each term of v, provided v is free of Int
    u = replace_pow_exp(u) # to replace back to sympy's exp
    v = replace_pow_exp(v)
    w = Simp(u*x**2, x)/x**2
    if u == 1:
        return v
    elif u == 0:
        return 0
    elif NumericFactor(u) < 0 and NumericFactor(-u) > 0:
        return -Dist(-u, v, x)
    elif SumQ(v):
        return Add(*[Dist(u, i, x) for i in v.args])
    elif IntegralFreeQ(v):
        return Simp(u*v, x)
    elif w != u and FreeQ(w, x) and w == Simp(w, x) and w == Simp(w*x**2, x)/x**2:
        return Dist(w, v, x)
    else:
        return Simp(u*v, x)

def PureFunctionOfCothQ(u, v, x):
    # If u is a pure function of Coth[v], PureFunctionOfCothQ[u,v,x] returns True;
    if AtomQ(u):
        return u != x
    elif CalculusQ(u):
        return False
    elif HyperbolicQ(u) and ZeroQ(u.args[0] - v):
        return CothQ(u)
    return all(PureFunctionOfCothQ(i, v, x) for i in u.args)

def LogIntegral(z):
    return li(z)

def ExpIntegralEi(z):
    return Ei(z)

def ExpIntegralE(a, b):
    return expint(a, b).evalf()

def SinIntegral(z):
    return Si(z)

def CosIntegral(z):
    return Ci(z)

def SinhIntegral(z):
    return Shi(z)

def CoshIntegral(z):
    return Chi(z)

class PolyGamma(Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            return polygamma(args[0], args[1])
        return digamma(args[0])

def LogGamma(z):
    return loggamma(z)

class ProductLog(Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            return LambertW(args[1], args[0]).evalf()
        return LambertW(args[0]).evalf()

def Factorial(a):
    return factorial(a)

def Zeta(*args):
    return zeta(*args)

def HypergeometricPFQ(a, b, c):
    return hyper(a, b, c)

def Sum_doit(exp, args):
    '''
    This function perform summation using sympy's `Sum`.

    Examples
    ========

    >>> from sympy.integrals.rubi.utility_function import Sum_doit
    >>> from sympy.abc import x
    >>> Sum_doit(2*x + 2, [x, 0, 1.7])
    6

    '''
    exp = replace_pow_exp(exp)
    if not isinstance(args[2], (int, Integer)):
        new_args = [args[0], args[1], Floor(args[2])]
        return Sum(exp, new_args).doit()

    return Sum(exp, args).doit()

def PolynomialQuotient(p, q, x):
    try:
        p = poly(p, x)
        q = poly(q, x)

    except:
        p = poly(p)
        q = poly(q)
    try:
        return quo(p, q).as_expr()
    except (PolynomialDivisionFailed, UnificationFailed):
        return p/q

def PolynomialRemainder(p, q, x):
    try:
        p = poly(p, x)
        q = poly(q, x)

    except:
        p = poly(p)
        q = poly(q)
    try:
        return rem(p, q).as_expr()
    except (PolynomialDivisionFailed, UnificationFailed):
        return S(0)

def Floor(x, a = None):
    if a is None:
        return floor(x)
    return a*floor(x/a)

def Factor(var):
    return factor(var)

def Rule(a, b):
    return {a: b}

def Distribute(expr, *args):
    if len(args) == 1:
        if isinstance(expr, args[0]):
            return expr
        else:
            return expr.expand()
    if len(args) == 2:
        if isinstance(expr, args[1]):
            return expr.expand()
        else:
            return expr
    return expr.expand()

def CoprimeQ(*args):
    args = S(args)
    g = gcd(*args)
    if g == 1:
        return True
    return False

def Discriminant(a, b):
    try:
        return discriminant(a, b)
    except PolynomialError:
        return Function('Discriminant')(a, b)

def Negative(x):
    return x < S(0)

def Quotient(m, n):
    return Floor(m/n)

def process_trig(expr):
    '''
    This function processes trigonometric expressions such that all `cot` is
    rewritten in terms of `tan`, `sec` in terms of `cos`, `csc` in terms of `sin` and
    similarly for `coth`, `sech` and `csch`.

    Examples
    ========

    >>> from sympy.integrals.rubi.utility_function import process_trig
    >>> from sympy.abc import x
    >>> from sympy import coth, cot, csc
    >>> process_trig(x*cot(x))
    x/tan(x)
    >>> process_trig(coth(x)*csc(x))
    1/(sin(x)*tanh(x))

    '''
    expr = expr.replace(lambda x: isinstance(x, cot), lambda x: 1/tan(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, sec), lambda x: 1/cos(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, csc), lambda x: 1/sin(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, coth), lambda x: 1/tanh(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, sech), lambda x: 1/cosh(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, csch), lambda x: 1/sinh(x.args[0]))
    return expr

def _ExpandIntegrand():
    Plus = Add
    Times = Mul
    def cons_f1(m):
        return PositiveIntegerQ(m)

    cons1 = CustomConstraint(cons_f1)
    def cons_f2(d, c, b, a):
        return ZeroQ(-a*d + b*c)

    cons2 = CustomConstraint(cons_f2)
    def cons_f3(a, x):
        return FreeQ(a, x)

    cons3 = CustomConstraint(cons_f3)
    def cons_f4(b, x):
        return FreeQ(b, x)

    cons4 = CustomConstraint(cons_f4)
    def cons_f5(c, x):
        return FreeQ(c, x)

    cons5 = CustomConstraint(cons_f5)
    def cons_f6(d, x):
        return FreeQ(d, x)

    cons6 = CustomConstraint(cons_f6)
    def cons_f7(e, x):
        return FreeQ(e, x)

    cons7 = CustomConstraint(cons_f7)
    def cons_f8(f, x):
        return FreeQ(f, x)

    cons8 = CustomConstraint(cons_f8)
    def cons_f9(g, x):
        return FreeQ(g, x)

    cons9 = CustomConstraint(cons_f9)
    def cons_f10(h, x):
        return FreeQ(h, x)

    cons10 = CustomConstraint(cons_f10)
    def cons_f11(e, b, c, f, n, p, F, x, d, m):
        if not isinstance(x, Symbol):
            return False
        return FreeQ(List(F, b, c, d, e, f, m, n, p), x)

    cons11 = CustomConstraint(cons_f11)
    def cons_f12(F, x):
        return FreeQ(F, x)

    cons12 = CustomConstraint(cons_f12)
    def cons_f13(m, x):
        return FreeQ(m, x)

    cons13 = CustomConstraint(cons_f13)
    def cons_f14(n, x):
        return FreeQ(n, x)

    cons14 = CustomConstraint(cons_f14)
    def cons_f15(p, x):
        return FreeQ(p, x)

    cons15 = CustomConstraint(cons_f15)
    def cons_f16(e, b, c, f, n, a, p, F, x, d, m):
        if not isinstance(x, Symbol):
            return False
        return FreeQ(List(F, a, b, c, d, e, f, m, n, p), x)

    cons16 = CustomConstraint(cons_f16)
    def cons_f17(n, m):
        return IntegersQ(m, n)

    cons17 = CustomConstraint(cons_f17)
    def cons_f18(n):
        return Less(n, S(0))

    cons18 = CustomConstraint(cons_f18)
    def cons_f19(x, u):
        if not isinstance(x, Symbol):
            return False
        return PolynomialQ(u, x)

    cons19 = CustomConstraint(cons_f19)
    def cons_f20(G, F, u):
        return SameQ(F(u)*G(u), S(1))

    cons20 = CustomConstraint(cons_f20)
    def cons_f21(q, x):
        return FreeQ(q, x)

    cons21 = CustomConstraint(cons_f21)
    def cons_f22(F):
        return MemberQ(List(ArcSin, ArcCos, ArcSinh, ArcCosh), F)

    cons22 = CustomConstraint(cons_f22)
    def cons_f23(j, n):
        return ZeroQ(j - S(2)*n)

    cons23 = CustomConstraint(cons_f23)
    def cons_f24(A, x):
        return FreeQ(A, x)

    cons24 = CustomConstraint(cons_f24)
    def cons_f25(B, x):
        return FreeQ(B, x)

    cons25 = CustomConstraint(cons_f25)
    def cons_f26(m, u, x):
        if not isinstance(x, Symbol):
            return False
        def _cons_f_u(d, w, c, p, x):
            return And(FreeQ(List(c, d), x), IntegerQ(p), Greater(p, m))
        cons_u = CustomConstraint(_cons_f_u)
        pat = Pattern(UtilityOperator((c_ + x_*WC('d', S(1)))**p_*WC('w', S(1)), x_), cons_u)
        result_matchq = is_match(UtilityOperator(u, x), pat)
        return Not(And(PositiveIntegerQ(m), result_matchq))

    cons26 = CustomConstraint(cons_f26)
    def cons_f27(b, v, n, a, x, u, m):
        if not isinstance(x, Symbol):
            return False
        return And(FreeQ(List(a, b, m), x), NegativeIntegerQ(n), Not(IntegerQ(m)), PolynomialQ(u, x), PolynomialQ(v, x),\
            RationalQ(m), Less(m, -1), GreaterEqual(Exponent(u, x), (-n - IntegerPart(m))*Exponent(v, x)))
    cons27 = CustomConstraint(cons_f27)
    def cons_f28(v, n, x, u, m):
        if not isinstance(x, Symbol):
            return False
        return And(FreeQ(List(a, b, m), x), NegativeIntegerQ(n), Not(IntegerQ(m)), PolynomialQ(u, x),\
            PolynomialQ(v, x), GreaterEqual(Exponent(u, x), -n*Exponent(v, x)))
    cons28 = CustomConstraint(cons_f28)
    def cons_f29(n):
        return PositiveIntegerQ(n/S(4))

    cons29 = CustomConstraint(cons_f29)
    def cons_f30(n):
        return IntegerQ(n)

    cons30 = CustomConstraint(cons_f30)
    def cons_f31(n):
        return Greater(n, S(1))

    cons31 = CustomConstraint(cons_f31)
    def cons_f32(n, m):
        return Less(S(0), m, n)

    cons32 = CustomConstraint(cons_f32)
    def cons_f33(n, m):
        return OddQ(n/GCD(m, n))

    cons33 = CustomConstraint(cons_f33)
    def cons_f34(a, b):
        return PosQ(a/b)

    cons34 = CustomConstraint(cons_f34)
    def cons_f35(n, m, p):
        return IntegersQ(m, n, p)

    cons35 = CustomConstraint(cons_f35)
    def cons_f36(n, m, p):
        return Less(S(0), m, p, n)

    cons36 = CustomConstraint(cons_f36)
    def cons_f37(q, n, m, p):
        return IntegersQ(m, n, p, q)

    cons37 = CustomConstraint(cons_f37)
    def cons_f38(n, q, m, p):
        return Less(S(0), m, p, q, n)

    cons38 = CustomConstraint(cons_f38)
    def cons_f39(n):
        return IntegerQ(n/S(2))

    cons39 = CustomConstraint(cons_f39)
    def cons_f40(p):
        return NegativeIntegerQ(p)

    cons40 = CustomConstraint(cons_f40)
    def cons_f41(n, m):
        return IntegersQ(m, n/S(2))

    cons41 = CustomConstraint(cons_f41)
    def cons_f42(n, m):
        return Unequal(m, n/S(2))

    cons42 = CustomConstraint(cons_f42)
    def cons_f43(c, b, a):
        return NonzeroQ(-S(4)*a*c + b**S(2))

    cons43 = CustomConstraint(cons_f43)
    def cons_f44(j, n, m):
        return IntegersQ(m, n, j)

    cons44 = CustomConstraint(cons_f44)
    def cons_f45(n, m):
        return Less(S(0), m, S(2)*n)

    cons45 = CustomConstraint(cons_f45)
    def cons_f46(n, m, p):
        return Not(And(Equal(m, n), Equal(p, S(-1))))

    cons46 = CustomConstraint(cons_f46)
    # ... other code
```
### 8 - sympy/integrals/integrals.py:

Start line: 824, End line: 904

```python
class Integral(AddWithLimits):

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       heurisch=None, conds='piecewise'):
        from sympy.integrals.deltafunctions import deltaintegrate
        from sympy.integrals.singularityfunctions import singularityintegrate
        from sympy.integrals.heurisch import heurisch as heurisch_, heurisch_wrapper
        from sympy.integrals.rationaltools import ratint
        from sympy.integrals.risch import risch_integrate

        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None

        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass

        eval_kwargs = dict(meijerg=meijerg, risch=risch, manual=manual,
            heurisch=heurisch, conds=conds)

        # if it is a poly(x) then let the polynomial integrate itself (fast)
        #
        # It is important to make this check first, otherwise the other code
        # will return a sympy expression instead of a Polynomial.
        #
        # see Polynomial for details.
        if isinstance(f, Poly) and not (manual or meijerg or risch):
            return f.integrate(x)

        # Piecewise antiderivatives need to call special integrate.
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)

        # let's cut it short if `f` does not depend on `x`; if
        # x is only a dummy, that will be handled below
        if not f.has(x):
            return f*x

        # try to convert to poly(x) and then integrate if successful (fast)
        poly = f.as_poly(x)
        if poly is not None and not (manual or meijerg or risch):
            return poly.integrate().as_expr()

        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True,
                    conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    # There was a nonelementary integral. Try integrating it.

                    # if no part of the NonElementaryIntegral is integrated by
                    # the Risch algorithm, then use the original function to
                    # integrate, instead of re-written one
                    if result == 0:
                        from sympy.integrals.risch import NonElementaryIntegral
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result

        # since Integral(f=g1+g2+...) == Integral(g1) + Integral(g2) + ...
        # we are going to handle Add terms separately,
        # if `f` is not Add -- we only have one term

        # Note that in general, this is a bad idea, because Integral(g1) +
        # Integral(g2) might not be computable, even if Integral(g1 + g2) is.
        # For example, Integral(x**x + x**x*log(x)).  But many heuristics only
        # work term-wise.  So we compute this step last, after trying
        # risch_integrate.  We also try risch_integrate again in this loop,
        # because maybe the integral is a sum of an elementary part and a
        # nonelementary part (like erf(x) + exp(x)).  risch_integrate() is
        # quite fast, so this is acceptable.
        parts = []
        args = Add.make_args(f)
        # ... other code
```
### 9 - sympy/simplify/simplify.py:

Start line: 746, End line: 790

```python
def sum_add(self, other, method=0):
    """Helper function for Sum simplification"""
    from sympy.concrete.summations import Sum
    from sympy import Mul

    #we know this is something in terms of a constant * a sum
    #so we temporarily put the constants inside for simplification
    #then simplify the result
    def __refactor(val):
        args = Mul.make_args(val)
        sumv = next(x for x in args if isinstance(x, Sum))
        constant = Mul(*[x for x in args if x != sumv])
        return Sum(constant * sumv.function, *sumv.limits)

    if isinstance(self, Mul):
        rself = __refactor(self)
    else:
        rself = self

    if isinstance(other, Mul):
        rother = __refactor(other)
    else:
        rother = other

    if type(rself) == type(rother):
        if method == 0:
            if rself.limits == rother.limits:
                return factor_sum(Sum(rself.function + rother.function, *rself.limits))
        elif method == 1:
            if simplify(rself.function - rother.function) == 0:
                if len(rself.limits) == len(rother.limits) == 1:
                    i = rself.limits[0][0]
                    x1 = rself.limits[0][1]
                    y1 = rself.limits[0][2]
                    j = rother.limits[0][0]
                    x2 = rother.limits[0][1]
                    y2 = rother.limits[0][2]

                    if i == j:
                        if x2 == y1 + 1:
                            return factor_sum(Sum(rself.function, (i, x1, y2)))
                        elif x1 == y2 + 1:
                            return factor_sum(Sum(rself.function, (i, x2, y1)))

    return Add(self, other)
```
### 10 - sympy/simplify/simplify.py:

Start line: 714, End line: 744

```python
def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
    """Helper function for Sum simplification

       if limits is specified, "self" is the inner part of a sum

       Returns the sum with constant factors brought outside
    """
    from sympy.core.exprtools import factor_terms
    from sympy.concrete.summations import Sum

    result = self.function if limits is None else self
    limits = self.limits if limits is None else limits
    #avoid any confusion w/ as_independent
    if result == 0:
        return S.Zero

    #get the summation variables
    sum_vars = set([limit.args[0] for limit in limits])

    #finally we try to factor out any common terms
    #and remove the from the sum if independent
    retv = factor_terms(result, radical=radical, clear=clear, fraction=fraction, sign=sign)
    #avoid doing anything bad
    if not result.is_commutative:
        return Sum(result, *limits)

    i, d = retv.as_independent(*sum_vars)
    if isinstance(retv, Add):
        return i * Sum(1, *limits) + Sum(d, *limits)
    else:
        return i * Sum(d, *limits)
```
### 11 - sympy/integrals/integrals.py:

Start line: 144, End line: 246

```python
class Integral(AddWithLimits):

    def transform(self, x, u):
        r"""
        Performs a change of variables from `x` to `u` using the relationship
        given by `x` and `u` which will define the transformations `f` and `F`
        (which are inverses of each other) as follows:

        1) If `x` is a Symbol (which is a variable of integration) then `u`
           will be interpreted as some function, f(u), with inverse F(u).
           This, in effect, just makes the substitution of x with f(x).

        2) If `u` is a Symbol then `x` will be interpreted as some function,
           F(x), with inverse f(u). This is commonly referred to as
           u-substitution.

        Once f and F have been identified, the transformation is made as
        follows:

        .. math:: \int_a^b x \mathrm{d}x \rightarrow \int_{F(a)}^{F(b)} f(x)
                  \frac{\mathrm{d}}{\mathrm{d}x}

        where `F(x)` is the inverse of `f(x)` and the limits and integrand have
        been corrected so as to retain the same value after integration.

        Notes
        =====

        The mappings, F(x) or f(u), must lead to a unique integral. Linear
        or rational linear expression, `2*x`, `1/x` and `sqrt(x)`, will
        always work; quadratic expressions like `x**2 - 1` are acceptable
        as long as the resulting integrand does not depend on the sign of
        the solutions (see examples).

        The integral will be returned unchanged if `x` is not a variable of
        integration.

        `x` must be (or contain) only one of of the integration variables. If
        `u` has more than one free symbol then it should be sent as a tuple
        (`u`, `uvar`) where `uvar` identifies which variable is replacing
        the integration variable.
        XXX can it contain another integration variable?

        Examples
        ========

        >>> from sympy.abc import a, b, c, d, x, u, y
        >>> from sympy import Integral, S, cos, sqrt

        >>> i = Integral(x*cos(x**2 - 1), (x, 0, 1))

        transform can change the variable of integration

        >>> i.transform(x, u)
        Integral(u*cos(u**2 - 1), (u, 0, 1))

        transform can perform u-substitution as long as a unique
        integrand is obtained:

        >>> i.transform(x**2 - 1, u)
        Integral(cos(u)/2, (u, -1, 0))

        This attempt fails because x = +/-sqrt(u + 1) and the
        sign does not cancel out of the integrand:

        >>> Integral(cos(x**2 - 1), (x, 0, 1)).transform(x**2 - 1, u)
        Traceback (most recent call last):
        ...
        ValueError:
        The mapping between F(x) and f(u) did not give a unique integrand.

        transform can do a substitution. Here, the previous
        result is transformed back into the original expression
        using "u-substitution":

        >>> ui = _
        >>> _.transform(sqrt(u + 1), x) == i
        True

        We can accomplish the same with a regular substitution:

        >>> ui.transform(u, x**2 - 1) == i
        True

        If the `x` does not contain a symbol of integration then
        the integral will be returned unchanged. Integral `i` does
        not have an integration variable `a` so no change is made:

        >>> i.transform(a, x) == i
        True

        When `u` has more than one free symbol the symbol that is
        replacing `x` must be identified by passing `u` as a tuple:

        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, u))
        Integral(a + u, (u, -a, 1 - a))
        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, a))
        Integral(a + u, (a, -u, 1 - u))

        See Also
        ========

        variables : Lists the integration variables
        as_dummy : Replace integration variables with dummy ones
        """
        # ... other code
```
### 12 - sympy/simplify/simplify.py:

Start line: 381, End line: 513

```python
def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False):
    """Simplifies the given expression.

    Simplification is not a well defined term and the exact strategies
    this function tries can change in the future versions of SymPy. If
    your algorithm relies on "simplification" (whatever it is), try to
    determine what you need exactly  -  is it powsimp()?, radsimp()?,
    together()?, logcombine()?, or something else? And use this particular
    function directly, because those are well defined and thus your algorithm
    will be robust.

    Nonetheless, especially for interactive use, or when you don't know
    anything about the structure of the expression, simplify() tries to apply
    intelligent heuristics to make the input expression "simpler".  For
    example:

    >>> from sympy import simplify, cos, sin
    >>> from sympy.abc import x, y
    >>> a = (x + x**2)/(x*sin(y)**2 + x*cos(y)**2)
    >>> a
    (x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)
    >>> simplify(a)
    x + 1

    Note that we could have obtained the same result by using specific
    simplification functions:

    >>> from sympy import trigsimp, cancel
    >>> trigsimp(a)
    (x**2 + x)/x
    >>> cancel(_)
    x + 1

    In some cases, applying :func:`simplify` may actually result in some more
    complicated expression. The default ``ratio=1.7`` prevents more extreme
    cases: if (result length)/(input length) > ratio, then input is returned
    unmodified.  The ``measure`` parameter lets you specify the function used
    to determine how complex an expression is.  The function should take a
    single argument as an expression and return a number such that if
    expression ``a`` is more complex than expression ``b``, then
    ``measure(a) > measure(b)``.  The default measure function is
    :func:`count_ops`, which returns the total number of operations in the
    expression.

    For example, if ``ratio=1``, ``simplify`` output can't be longer
    than input.

    ::

        >>> from sympy import sqrt, simplify, count_ops, oo
        >>> root = 1/(sqrt(2)+3)

    Since ``simplify(root)`` would result in a slightly longer expression,
    root is returned unchanged instead::

       >>> simplify(root, ratio=1) == root
       True

    If ``ratio=oo``, simplify will be applied anyway::

        >>> count_ops(simplify(root, ratio=oo)) > count_ops(root)
        True

    Note that the shortest expression is not necessary the simplest, so
    setting ``ratio`` to 1 may not be a good idea.
    Heuristically, the default value ``ratio=1.7`` seems like a reasonable
    choice.

    You can easily define your own measure function based on what you feel
    should represent the "size" or "complexity" of the input expression.  Note
    that some choices, such as ``lambda expr: len(str(expr))`` may appear to be
    good metrics, but have other problems (in this case, the measure function
    may slow down simplify too much for very large expressions).  If you don't
    know what a good metric would be, the default, ``count_ops``, is a good
    one.

    For example:

    >>> from sympy import symbols, log
    >>> a, b = symbols('a b', positive=True)
    >>> g = log(a) + log(b) + log(a)*log(1/b)
    >>> h = simplify(g)
    >>> h
    log(a*b**(1 - log(a)))
    >>> count_ops(g)
    8
    >>> count_ops(h)
    5

    So you can see that ``h`` is simpler than ``g`` using the count_ops metric.
    However, we may not like how ``simplify`` (in this case, using
    ``logcombine``) has created the ``b**(log(1/a) + 1)`` term.  A simple way
    to reduce this would be to give more weight to powers as operations in
    ``count_ops``.  We can do this by using the ``visual=True`` option:

    >>> print(count_ops(g, visual=True))
    2*ADD + DIV + 4*LOG + MUL
    >>> print(count_ops(h, visual=True))
    2*LOG + MUL + POW + SUB

    >>> from sympy import Symbol, S
    >>> def my_measure(expr):
    ...     POW = Symbol('POW')
    ...     # Discourage powers by giving POW a weight of 10
    ...     count = count_ops(expr, visual=True).subs(POW, 10)
    ...     # Every other operation gets a weight of 1 (the default)
    ...     count = count.replace(Symbol, type(S.One))
    ...     return count
    >>> my_measure(g)
    8
    >>> my_measure(h)
    14
    >>> 15./8 > 1.7 # 1.7 is the default ratio
    True
    >>> simplify(g, measure=my_measure)
    -log(a)*log(b) + log(a) + log(b)

    Note that because ``simplify()`` internally tries many different
    simplification strategies and then compares them using the measure
    function, we get a completely different result that is still different
    from the input expression by doing this.

    If rational=True, Floats will be recast as Rationals before simplification.
    If rational=None, Floats will be recast as Rationals but the result will
    be recast as Floats. If rational=False(default) then nothing will be done
    to the Floats.

    If inverse=True, it will be assumed that a composition of inverse
    functions, such as sin and asin, can be cancelled in any order.
    For example, ``asin(sin(x))`` will yield ``x`` without checking whether
    x belongs to the set where this relation is true. The default is
    False.
    """
    # ... other code
```
### 13 - sympy/integrals/integrals.py:

Start line: 905, End line: 1072

```python
class Integral(AddWithLimits):

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       heurisch=None, conds='piecewise'):
        # ... other code
        for g in args:
            coeff, g = g.as_independent(x)

            # g(x) = const
            if g is S.One and not meijerg:
                parts.append(coeff*x)
                continue

            # g(x) = expr + O(x**n)
            order_term = g.getO()

            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)

                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)

                    if h_order_expr is not None:
                        h_order_term = order_term.func(
                            h_order_expr, *order_term.variables)
                        parts.append(coeff*(h + h_order_term))
                        continue

                # NOTE: if there is O(x**n) and we fail to integrate then
                # there is no point in trying other methods because they
                # will fail, too.
                return None

            #               c
            # g(x) = (a*x+b)
            if g.is_Pow and not g.exp.has(x) and not meijerg:
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])

                M = g.base.match(a*x + b)

                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base**(g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base**(g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))

                    parts.append(coeff * h / M[a])
                    continue

            #        poly(x)
            # g(x) = -------
            #        poly(x)
            if g.is_rational_function(x) and not (manual or meijerg or risch):
                parts.append(coeff * ratint(g, x))
                continue

            if not (manual or meijerg or risch):
                # g(x) = Mul(trig)
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a DiracDelta term
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a Singularity Function term
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # Try risch again.
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x,
                            separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)

                        parts.append(coeff*h)
                        continue

                # fall back to heurisch
                if heurisch is not False:
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        # XXX: this exception means there is a bug in the
                        # implementation of heuristic Risch integration
                        # algorithm.
                        h = None
            else:
                h = None

            if meijerg is not False and h is None:
                # rewrite using G functions
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    from sympy.integrals.meijerint import _debug
                    _debug('NotImplementedError from meijerint_definite')
                    res = None
                if h is not None:
                    parts.append(coeff * h)
                    continue

            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and not isinstance(result, Integral):
                        if result.has(Integral) and not manual:
                            # Try to have other algorithms do the integrals
                            # manualintegrate can't handle,
                            # unless we were asked to use manual only.
                            # Keep the rest of eval_kwargs in case another
                            # method was set to False already
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs["manual"] = False
                            result = result.func(*[
                                arg.doit(**new_eval_kwargs) if
                                arg.has(Integral) else arg
                                for arg in result.args
                            ]).expand(multinomial=False,
                                      log=False,
                                      power_exp=False,
                                      power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    # can't handle some SymPy expressions
                    pass

            # if we failed maybe it was because we had
            # a product that could have been expanded,
            # so let's try an expansion of the whole
            # thing before giving up; we don't try this
            # at the outset because there are things
            # that cannot be solved unless they are
            # NOT expanded e.g., x**x*(1+log(x)). There
            # should probably be a checker somewhere in this
            # routine to look for such cases and try to do
            # collection on the expressions if they are already
            # in an expanded form
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    # Note: risch will be identical on the expanded
                    # expression, but maybe it will be able to pick out parts,
                    # like x*(exp(x) + erf(x)).
                    return self._eval_integral(f, x, **eval_kwargs)

            if h is not None:
                parts.append(coeff * h)
            else:
                return None

        return Add(*parts)
```
### 14 - sympy/integrals/integrals.py:

Start line: 440, End line: 570

```python
class Integral(AddWithLimits):

    def doit(self, **hints):
        # ... other code
        for xab in self.limits:
            # compute uli, the free symbols in the
            # Upper and Lower limits of limit I
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            # this integral can be done as long as there is no blocking
            # limit that has been undone. An undone limit is blocking if
            # it contains an integration variable that is in this limit's
            # upper or lower free symbols or vice versa
            if xab[0] in ulj or any(v[0] in uli for v in undone_limits):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*([function] + [xab]))
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue

            if function.has(Abs, sign) and (
                (len(xab) < 3 and all(x.is_real for x in xab)) or
                (len(xab) == 3 and all(x.is_real and not x.is_infinite for
                 x in xab[1:]))):
                    # some improper integrals are better off with Abs
                    xr = Dummy("xr", real=True)
                    function = (function.xreplace({xab[0]: xr})
                        .rewrite(Piecewise).xreplace({xr: xab[0]}))
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if (function.has(Piecewise) and
                not isinstance(function, Piecewise)):
                    function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0],
                        **eval_kwargs)
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
            else:
                # There are a number of tradeoffs in using the
                # Meijer G method. It can sometimes be a lot faster
                # than other methods, and sometimes slower. And
                # there are certain types of integrals for which it
                # is more likely to work than others. These
                # heuristics are incorporated in deciding what
                # integration methods to try, in what order. See the
                # integrate() docstring for details.
                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            from sympy.integrals.meijerint import _debug
                            _debug('NotImplementedError '
                                'from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                ret = Piecewise(
                                    (f, cond),
                                    (self.func(
                                    function, (x, a, b)), True))
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('''
                                        conds=separate not supported in
                                        multiple integrals'''))
                                ret = f, cond
                            else:
                                ret = f
                    return ret

                meijerg1 = meijerg
                if (meijerg is not False and
                        len(xab) == 3 and xab[1].is_real and xab[2].is_real
                        and not function.is_Poly and
                        (xab[1].has(oo, -oo) or xab[2].has(oo, -oo))):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                # If the special meijerg code did not succeed in
                # finding a definite integral, then the code using
                # meijerint_indefinite will not either (it might
                # find an antiderivative, but the answer is likely
                # to be nonsensical). Thus if we are requested to
                # only use Meijer G-function methods, we give up at
                # this stage. Otherwise we just disable G-function
                # methods.
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue

            if not isinstance(antideriv, Integral) and antideriv is not None:
                sym = xab[0]
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        # The coefficient of `tan` should be constant
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a-pi/2)/pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        # The coefficient of `cot` should be constant
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a)/pi)))
            # ... other code
        # ... other code
```
### 15 - sympy/integrals/integrals.py:

Start line: 1074, End line: 1101

```python
class Integral(AddWithLimits):

    def _eval_lseries(self, x, logx):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        terms, order = expr.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *expr.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x):
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])
```
### 17 - sympy/integrals/integrals.py:

Start line: 1198, End line: 1232

```python
class Integral(AddWithLimits):

    def as_sum(self, n=None, method="midpoint", evaluate=True):

        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError(
                "Multidimensional midpoint rule not implemented yet")
        else:
            limit = limits[0]
            if (len(limit) != 3 or limit[1].is_finite is False or
                limit[2].is_finite is False):
                raise ValueError("Expecting a definite integral over "
                                  "a finite interval.")
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if (n.is_positive is False or n.is_integer is False or
            n.is_finite is False):
            raise ValueError("n must be a positive integer, got %s" % n)
        x, a, b = limit
        dx = (b - a)/n
        k = Dummy('k', integer=True, positive=True)
        f = self.function

        if method == "left":
            result = dx*Sum(f.subs(x, a + (k-1)*dx), (k, 1, n))
        elif method == "right":
            result = dx*Sum(f.subs(x, a + k*dx), (k, 1, n))
        elif method == "midpoint":
            result = dx*Sum(f.subs(x, a + k*dx - dx/2), (k, 1, n))
        elif method == "trapezoid":
            result = dx*((f.subs(x, a) + f.subs(x, b))/2 +
                Sum(f.subs(x, a + k*dx), (k, 1, n - 1)))
        else:
            raise ValueError("Unknown method %s" % method)
        return result.doit() if evaluate else result
```
### 18 - sympy/simplify/simplify.py:

Start line: 1104, End line: 1168

```python
def besselsimp(expr):
    """
    Simplify bessel-type functions.

    This routine tries to simplify bessel-type functions. Currently it only
    works on the Bessel J and I functions, however. It works by looking at all
    such functions in turn, and eliminating factors of "I" and "-1" (actually
    their polar equivalents) in front of the argument. Then, functions of
    half-integer order are rewritten using strigonometric functions and
    functions of integer order (> 1) are rewritten using functions
    of low order.  Finally, if the expression was changed, compute
    factorization of the result with factor().

    >>> from sympy import besselj, besseli, besselsimp, polar_lift, I, S
    >>> from sympy.abc import z, nu
    >>> besselsimp(besselj(nu, z*polar_lift(-1)))
    exp(I*pi*nu)*besselj(nu, z)
    >>> besselsimp(besseli(nu, z*polar_lift(-I)))
    exp(-I*pi*nu/2)*besselj(nu, z)
    >>> besselsimp(besseli(S(-1)/2, z))
    sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    >>> besselsimp(z*besseli(0, z) + z*(besseli(2, z))/2 + besseli(1, z))
    3*z*besseli(0, z)/2
    """
    # TODO
    # - better algorithm?
    # - simplify (cos(pi*b)*besselj(b,z) - besselj(-b,z))/sin(pi*b) ...
    # - use contiguity relations?

    def replacer(fro, to, factors):
        factors = set(factors)

        def repl(nu, z):
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        def tofunc(nu, z):
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        def tofunc(nu, z):
            return exp(I*pi*nu)*fro(nu, exp_polar(-I*pi)*z)
        return tofunc

    orig_expr = expr

    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    expr = expr.replace(
        besselj, replacer(besselj,
        torewrite(besselj, besseli), ifactors))
    expr = expr.replace(
        besseli, replacer(besseli,
        torewrite(besseli, besselj), ifactors))

    minusfactors = [-1, exp_polar(I*pi)]
    expr = expr.replace(
        besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(
        besseli, replacer(besseli, tominus(besseli), minusfactors))

    z0 = Dummy('z')
    # ... other code
```
