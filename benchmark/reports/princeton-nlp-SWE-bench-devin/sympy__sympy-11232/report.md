# sympy__sympy-11232

| **sympy/sympy** | `4c8a8590be682e74ec91ab217c646baa4686a255` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 15921 |
| **Any found context length** | 840 |
| **Avg pos** | 15.8 |
| **Min pos** | 3 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 11 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/sympy/core/expr.py b/sympy/core/expr.py
--- a/sympy/core/expr.py
+++ b/sympy/core/expr.py
@@ -1044,7 +1044,7 @@ def args_cnc(self, cset=False, warn=True, split_1=True):
         Note: -1 is always separated from a Number unless split_1 is False.
 
         >>> from sympy import symbols, oo
-        >>> A, B = symbols('A B', commutative=0)
+        >>> A, B = symbols('A B', commutative=False)
         >>> x, y = symbols('x y')
         >>> (-2*x*y).args_cnc()
         [[-1, 2, x, y], []]
diff --git a/sympy/matrices/expressions/matmul.py b/sympy/matrices/expressions/matmul.py
--- a/sympy/matrices/expressions/matmul.py
+++ b/sympy/matrices/expressions/matmul.py
@@ -120,9 +120,11 @@ def doit(self, **kwargs):
     # Needed for partial compatibility with Mul
     def args_cnc(self, **kwargs):
         coeff, matrices = self.as_coeff_matrices()
-        # I don't know how coeff could have noncommutative factors, but this
-        # handles it.
         coeff_c, coeff_nc = coeff.args_cnc(**kwargs)
+        if coeff_c == [1]:
+            coeff_c = []
+        elif coeff_c == set([1]):
+            coeff_c = set()
 
         return coeff_c, coeff_nc + matrices
 
diff --git a/sympy/printing/llvmjitcode.py b/sympy/printing/llvmjitcode.py
--- a/sympy/printing/llvmjitcode.py
+++ b/sympy/printing/llvmjitcode.py
@@ -428,9 +428,9 @@ def llvm_callable(args, expr, callback_type=None):
     >>> from sympy.abc import x,y
     >>> e1 = x*x + y*y
     >>> e2 = 4*(x*x + y*y) + 8.0
-    >>> after_cse = cse([e1,e2])
+    >>> after_cse = cse([e1, e2])
     >>> after_cse
-    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
+    ([(x0, x**2 + y**2)], [x0, 4*x0 + 8.0])
     >>> j1 = jit.llvm_callable([x,y], after_cse)
     >>> j1(1.0, 2.0)
     (5.0, 28.0)
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -2,13 +2,14 @@
 """
 from __future__ import print_function, division
 
-from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple
+from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple, igcd
+from sympy.core.numbers import Integer
 from sympy.core.singleton import S
 from sympy.core.function import _coeff_isneg
 from sympy.core.exprtools import factor_terms
-from sympy.core.compatibility import iterable, range
+from sympy.core.compatibility import iterable, range, as_int
 from sympy.utilities.iterables import filter_symbols, \
-    numbered_symbols, sift, topological_sort, ordered
+    numbered_symbols, sift, topological_sort, ordered, subsets
 
 from . import cse_opts
 
@@ -136,7 +137,46 @@ def postprocess_for_cse(expr, optimizations):
     return expr
 
 
-def opt_cse(exprs, order='canonical'):
+def pairwise_most_common(sets):
+    """Return a list of `(s, L)` tuples where `s` is the largest subset
+    of elements that appear in pairs of sets given by `sets` and `L`
+    is a list of tuples giving the indices of the pairs of sets in
+    which those elements appeared. All `s` will be of the same length.
+
+    Examples
+    ========
+
+    >>> from sympy.simplify.cse_main import pairwise_most_common
+    >>> pairwise_most_common((
+    ...     set([1,2,3]),
+    ...     set([1,3,5]),
+    ...     set([1,2,3,4,5]),
+    ...     set([1,2,3,6])))
+    [(set([1, 3, 5]), [(1, 2)]), (set([1, 2, 3]), [(0, 2), (0, 3), (2, 3)])]
+    >>>
+    """
+    from sympy.utilities.iterables import subsets
+    from collections import defaultdict
+    most = -1
+    for i, j in subsets(list(range(len(sets))), 2):
+        com = sets[i] & sets[j]
+        if com and len(com) > most:
+            best = defaultdict(list)
+            best_keys = []
+            most = len(com)
+        if len(com) == most:
+            if com not in best_keys:
+                best_keys.append(com)
+            best[best_keys.index(com)].append((i,j))
+    if most == -1:
+        return []
+    for k in range(len(best)):
+        best_keys[k] = (best_keys[k], best[k])
+    best_keys.sort(key=lambda x: len(x[1]))
+    return best_keys
+
+
+def opt_cse(exprs, order='canonical', verbose=False):
     """Find optimization opportunities in Adds, Muls, Pows and negative
     coefficient Muls
 
@@ -147,6 +187,8 @@ def opt_cse(exprs, order='canonical'):
     order : string, 'none' or 'canonical'
         The order by which Mul and Add arguments are processed. For large
         expressions where speed is a concern, use the setting order='none'.
+    verbose : bool
+        Print debug information (default=False)
 
     Returns
     -------
@@ -218,51 +260,149 @@ def _match_common_args(Func, funcs):
         else:
             funcs = sorted(funcs, key=lambda x: len(x.args))
 
-        func_args = [set(e.args) for e in funcs]
-        for i in range(len(func_args)):
-            for j in range(i + 1, len(func_args)):
-                com_args = func_args[i].intersection(func_args[j])
-                if len(com_args) > 1:
-                    com_func = Func(*com_args)
-
-                    # for all sets, replace the common symbols by the function
-                    # over them, to allow recursive matches
-
-                    diff_i = func_args[i].difference(com_args)
-                    func_args[i] = diff_i | {com_func}
-                    if diff_i:
-                        opt_subs[funcs[i]] = Func(Func(*diff_i), com_func,
-                                                  evaluate=False)
-
-                    diff_j = func_args[j].difference(com_args)
-                    func_args[j] = diff_j | {com_func}
-                    opt_subs[funcs[j]] = Func(Func(*diff_j), com_func,
-                                              evaluate=False)
-
-                    for k in range(j + 1, len(func_args)):
-                        if not com_args.difference(func_args[k]):
-                            diff_k = func_args[k].difference(com_args)
-                            func_args[k] = diff_k | {com_func}
-                            opt_subs[funcs[k]] = Func(Func(*diff_k), com_func,
-                                                      evaluate=False)
+        if Func is Mul:
+            F = Pow
+            meth = 'as_powers_dict'
+            from sympy.core.add import _addsort as inplace_sorter
+        elif Func is Add:
+            F = Mul
+            meth = 'as_coefficients_dict'
+            from sympy.core.mul import _mulsort as inplace_sorter
+        else:
+            assert None  # expected Mul or Add
+
+        # ----------------- helpers ---------------------------
+        def ufunc(*args):
+            # return a well formed unevaluated function from the args
+            # SHARES Func, inplace_sorter
+            args = list(args)
+            inplace_sorter(args)
+            return Func(*args, evaluate=False)
+
+        def as_dict(e):
+            # creates a dictionary of the expression using either
+            # as_coefficients_dict or as_powers_dict, depending on Func
+            # SHARES meth
+            d = getattr(e, meth, lambda: {a: S.One for a in e.args})()
+            for k in list(d.keys()):
+                try:
+                    as_int(d[k])
+                except ValueError:
+                    d[F(k, d.pop(k))] = S.One
+            return d
+
+        def from_dict(d):
+            # build expression from dict from
+            # as_coefficients_dict or as_powers_dict
+            # SHARES F
+            return ufunc(*[F(k, v) for k, v in d.items()])
+
+        def update(k):
+            # updates all of the info associated with k using
+            # the com_dict: func_dicts, func_args, opt_subs
+            # returns True if all values were updated, else None
+            # SHARES com_dict, com_func, func_dicts, func_args,
+            #        opt_subs, funcs, verbose
+            for di in com_dict:
+                # don't allow a sign to change
+                if com_dict[di] > func_dicts[k][di]:
+                    return
+            # remove it
+            if Func is Add:
+                take = min(func_dicts[k][i] for i in com_dict)
+                com_func_take = Mul(take, from_dict(com_dict), evaluate=False)
+            else:
+                take = igcd(*[func_dicts[k][i] for i in com_dict])
+                com_func_take = Pow(from_dict(com_dict), take, evaluate=False)
+            for di in com_dict:
+                func_dicts[k][di] -= take*com_dict[di]
+            # compute the remaining expression
+            rem = from_dict(func_dicts[k])
+            # reject hollow change, e.g extracting x + 1 from x + 3
+            if Func is Add and rem and rem.is_Integer and 1 in com_dict:
+                return
+            if verbose:
+                print('\nfunc %s (%s) \ncontains %s \nas %s \nleaving %s' %
+                    (funcs[k], func_dicts[k], com_func, com_func_take, rem))
+            # recompute the dict since some keys may now
+            # have corresponding values of 0; one could
+            # keep track of which ones went to zero but
+            # this seems cleaner
+            func_dicts[k] = as_dict(rem)
+            # update associated info
+            func_dicts[k][com_func] = take
+            func_args[k] = set(func_dicts[k])
+            # keep the constant separate from the remaining
+            # part of the expression, e.g. 2*(a*b) rather than 2*a*b
+            opt_subs[funcs[k]] = ufunc(rem, com_func_take)
+            # everything was updated
+            return True
+
+        def get_copy(i):
+            return [func_dicts[i].copy(), func_args[i].copy(), funcs[i], i]
+
+        def restore(dafi):
+            i = dafi.pop()
+            func_dicts[i], func_args[i], funcs[i] = dafi
+
+        # ----------------- end helpers -----------------------
+
+        func_dicts = [as_dict(f) for f in funcs]
+        func_args = [set(d) for d in func_dicts]
+        while True:
+            hit = pairwise_most_common(func_args)
+            if not hit or len(hit[0][0]) <= 1:
+                break
+            changed = False
+            for com_args, ij in hit:
+                take = len(com_args)
+                ALL = list(ordered(com_args))
+                while take >= 2:
+                    for com_args in subsets(ALL, take):
+                        com_func = Func(*com_args)
+                        com_dict = as_dict(com_func)
+                        for i, j in ij:
+                            dafi = None
+                            if com_func != funcs[i]:
+                                dafi = get_copy(i)
+                                ch = update(i)
+                                if not ch:
+                                    restore(dafi)
+                                    continue
+                            if com_func != funcs[j]:
+                                dafj = get_copy(j)
+                                ch = update(j)
+                                if not ch:
+                                    if dafi is not None:
+                                        restore(dafi)
+                                    restore(dafj)
+                                    continue
+                            changed = True
+                        if changed:
+                            break
+                    else:
+                        take -= 1
+                        continue
+                    break
+                else:
+                    continue
+                break
+            if not changed:
+                break
 
     # split muls into commutative
-    comutative_muls = set()
+    commutative_muls = set()
     for m in muls:
         c, nc = m.args_cnc(cset=True)
         if c:
             c_mul = m.func(*c)
             if nc:
-                if c_mul == 1:
-                    new_obj = m.func(*nc)
-                else:
-                    new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
-                opt_subs[m] = new_obj
+                opt_subs[m] = m.func(c_mul, m.func(*nc), evaluate=False)
             if len(c) > 1:
-                comutative_muls.add(c_mul)
+                commutative_muls.add(c_mul)
 
     _match_common_args(Add, adds)
-    _match_common_args(Mul, comutative_muls)
+    _match_common_args(Mul, commutative_muls)
 
     return opt_subs
 
@@ -394,6 +534,30 @@ def _rebuild(expr):
             reduced_e = e
         reduced_exprs.append(reduced_e)
 
+    # don't allow hollow nesting
+    # e.g if p = [b + 2*d + e + f, b + 2*d + f + g, a + c + d + f + g]
+    # and R, C = cse(p) then
+    #     R = [(x0, d + f), (x1, b + d)]
+    #     C = [e + x0 + x1, g + x0 + x1, a + c + d + f + g]
+    # but the args of C[-1] should not be `(a + c, d + f + g)`
+    nested = [[i for i in f.args if isinstance(i, f.func)] for f in exprs]
+    for i in range(len(exprs)):
+        F = reduced_exprs[i].func
+        if not (F is Mul or F is Add):
+            continue
+        nested = [a for a in exprs[i].args if isinstance(a, F)]
+        args = []
+        for a in reduced_exprs[i].args:
+            if isinstance(a, F):
+                for ai in a.args:
+                    if isinstance(ai, F) and ai not in nested:
+                        args.extend(ai.args)
+                    else:
+                        args.append(ai)
+            else:
+                args.append(a)
+        reduced_exprs[i] = F(*args)
+
     return replacements, reduced_exprs
 
 
@@ -444,7 +608,7 @@ def cse(exprs, symbols=None, optimizations=None, postprocess=None,
     >>> from sympy import cse, SparseMatrix
     >>> from sympy.abc import x, y, z, w
     >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
-    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])
+    ([(x0, w + y + z)], [x0*(x + x0)/(w + x)**3])
 
     Note that currently, y + z will not get substituted if -y - z is used.
 
diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py
--- a/sympy/solvers/ode.py
+++ b/sympy/solvers/ode.py
@@ -4338,8 +4338,6 @@ def ode_linear_coefficients(eq, func, order, match):
     >>> f = Function('f')
     >>> df = f(x).diff(x)
     >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
-    >>> dsolve(eq, hint='linear_coefficients')
-    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
     >>> pprint(dsolve(eq, hint='linear_coefficients'))
                       ___________                     ___________
                    /         2                     /         2
@@ -4403,8 +4401,6 @@ def ode_separable_reduced(eq, func, order, match):
     >>> f = Function('f')
     >>> d = f(x).diff(x)
     >>> eq = (x - x**2*f(x))*d - f(x)
-    >>> dsolve(eq, hint='separable_reduced')
-    [Eq(f(x), (-sqrt(C1*x**2 + 1) + 1)/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
     >>> pprint(dsolve(eq, hint='separable_reduced'))
                  ___________                ___________
                 /     2                    /     2

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/expr.py | 1047 | 1047 | - | - | -
| sympy/matrices/expressions/matmul.py | 123 | 124 | - | - | -
| sympy/printing/llvmjitcode.py | 431 | 433 | 22 | 9 | 15921
| sympy/simplify/cse_main.py | 5 | 11 | 6 | 1 | 2041
| sympy/simplify/cse_main.py | 139 | 139 | 20 | 1 | 14772
| sympy/simplify/cse_main.py | 150 | 150 | 20 | 1 | 14772
| sympy/simplify/cse_main.py | 221 | 265 | - | 1 | -
| sympy/simplify/cse_main.py | 397 | 397 | 3 | 1 | 840
| sympy/simplify/cse_main.py | 447 | 447 | 8 | 1 | 4209
| sympy/solvers/ode.py | 4341 | 4342 | - | 10 | -
| sympy/solvers/ode.py | 4406 | 4407 | - | 10 | -


## Problem Statement

```
cse leaves behind unevaluated subexpressions
\`\`\` python
>>> cse((j*l**2*y, j*l*o*r*y, k*o*r*s))
([(x0, j*y)], [l**2*x0, l*o*r*x0, (k*s)*(o*r)])
>>> u = _[1][-1]
>>> u.args
(k*s, o*r)

This can lead to problems when trying to work with the result:

>>> u.subs(s*o, 2)
(k*s)*(o*r)
>>> Mul(*flatten([i.args for i in u.args]))
k*o*r*s
>>> _.subs(s*o,2)
2*k*r
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/simplify/cse_main.py** | 173 | 213| 268 | 268 | 4150 | 
| 2 | **1 sympy/simplify/cse_main.py** | 249 | 267| 161 | 429 | 4150 | 
| **-> 3 <-** | **1 sympy/simplify/cse_main.py** | 332 | 397| 411 | 840 | 4150 | 
| 4 | **1 sympy/simplify/cse_main.py** | 467 | 539| 594 | 1434 | 4150 | 
| 5 | **1 sympy/simplify/cse_main.py** | 215 | 247| 327 | 1761 | 4150 | 
| **-> 6 <-** | **1 sympy/simplify/cse_main.py** | 1 | 29| 280 | 2041 | 4150 | 
| 7 | 2 sympy/solvers/solvers.py | 3179 | 3277| 1432 | 3473 | 33733 | 
| **-> 8 <-** | **2 sympy/simplify/cse_main.py** | 400 | 466| 736 | 4209 | 33733 | 
| 9 | 3 sympy/simplify/cse_opts.py | 1 | 44| 378 | 4587 | 34112 | 
| 10 | **3 sympy/simplify/cse_main.py** | 65 | 88| 335 | 4922 | 34112 | 
| 11 | 4 sympy/core/basic.py | 716 | 828| 1037 | 5959 | 48260 | 
| 12 | 5 sympy/core/mul.py | 1449 | 1575| 955 | 6914 | 62183 | 
| 13 | 5 sympy/core/mul.py | 236 | 361| 944 | 7858 | 62183 | 
| 14 | 6 sympy/core/function.py | 1507 | 1549| 393 | 8251 | 84060 | 
| 15 | 6 sympy/solvers/solvers.py | 358 | 800| 4409 | 12660 | 84060 | 
| 16 | 6 sympy/solvers/solvers.py | 3149 | 3177| 310 | 12970 | 84060 | 
| 17 | 6 sympy/core/function.py | 1602 | 1665| 456 | 13426 | 84060 | 
| 18 | **6 sympy/simplify/cse_main.py** | 270 | 330| 348 | 13774 | 84060 | 
| 19 | 7 sympy/core/sympify.py | 358 | 450| 761 | 14535 | 87706 | 
| **-> 20 <-** | **7 sympy/simplify/cse_main.py** | 139 | 171| 237 | 14772 | 87706 | 
| 21 | 8 sympy/simplify/powsimp.py | 227 | 251| 322 | 15094 | 94489 | 
| **-> 22 <-** | **9 sympy/printing/llvmjitcode.py** | 359 | 437| 827 | 15921 | 98515 | 
| 23 | 9 sympy/solvers/solvers.py | 3112 | 3147| 325 | 16246 | 98515 | 
| 24 | 9 sympy/core/function.py | 1667 | 1691| 282 | 16528 | 98515 | 


## Missing Patch Files

 * 1: sympy/core/expr.py
 * 2: sympy/matrices/expressions/matmul.py
 * 3: sympy/printing/llvmjitcode.py
 * 4: sympy/simplify/cse_main.py
 * 5: sympy/solvers/ode.py

## Patch

```diff
diff --git a/sympy/core/expr.py b/sympy/core/expr.py
--- a/sympy/core/expr.py
+++ b/sympy/core/expr.py
@@ -1044,7 +1044,7 @@ def args_cnc(self, cset=False, warn=True, split_1=True):
         Note: -1 is always separated from a Number unless split_1 is False.
 
         >>> from sympy import symbols, oo
-        >>> A, B = symbols('A B', commutative=0)
+        >>> A, B = symbols('A B', commutative=False)
         >>> x, y = symbols('x y')
         >>> (-2*x*y).args_cnc()
         [[-1, 2, x, y], []]
diff --git a/sympy/matrices/expressions/matmul.py b/sympy/matrices/expressions/matmul.py
--- a/sympy/matrices/expressions/matmul.py
+++ b/sympy/matrices/expressions/matmul.py
@@ -120,9 +120,11 @@ def doit(self, **kwargs):
     # Needed for partial compatibility with Mul
     def args_cnc(self, **kwargs):
         coeff, matrices = self.as_coeff_matrices()
-        # I don't know how coeff could have noncommutative factors, but this
-        # handles it.
         coeff_c, coeff_nc = coeff.args_cnc(**kwargs)
+        if coeff_c == [1]:
+            coeff_c = []
+        elif coeff_c == set([1]):
+            coeff_c = set()
 
         return coeff_c, coeff_nc + matrices
 
diff --git a/sympy/printing/llvmjitcode.py b/sympy/printing/llvmjitcode.py
--- a/sympy/printing/llvmjitcode.py
+++ b/sympy/printing/llvmjitcode.py
@@ -428,9 +428,9 @@ def llvm_callable(args, expr, callback_type=None):
     >>> from sympy.abc import x,y
     >>> e1 = x*x + y*y
     >>> e2 = 4*(x*x + y*y) + 8.0
-    >>> after_cse = cse([e1,e2])
+    >>> after_cse = cse([e1, e2])
     >>> after_cse
-    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
+    ([(x0, x**2 + y**2)], [x0, 4*x0 + 8.0])
     >>> j1 = jit.llvm_callable([x,y], after_cse)
     >>> j1(1.0, 2.0)
     (5.0, 28.0)
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -2,13 +2,14 @@
 """
 from __future__ import print_function, division
 
-from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple
+from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple, igcd
+from sympy.core.numbers import Integer
 from sympy.core.singleton import S
 from sympy.core.function import _coeff_isneg
 from sympy.core.exprtools import factor_terms
-from sympy.core.compatibility import iterable, range
+from sympy.core.compatibility import iterable, range, as_int
 from sympy.utilities.iterables import filter_symbols, \
-    numbered_symbols, sift, topological_sort, ordered
+    numbered_symbols, sift, topological_sort, ordered, subsets
 
 from . import cse_opts
 
@@ -136,7 +137,46 @@ def postprocess_for_cse(expr, optimizations):
     return expr
 
 
-def opt_cse(exprs, order='canonical'):
+def pairwise_most_common(sets):
+    """Return a list of `(s, L)` tuples where `s` is the largest subset
+    of elements that appear in pairs of sets given by `sets` and `L`
+    is a list of tuples giving the indices of the pairs of sets in
+    which those elements appeared. All `s` will be of the same length.
+
+    Examples
+    ========
+
+    >>> from sympy.simplify.cse_main import pairwise_most_common
+    >>> pairwise_most_common((
+    ...     set([1,2,3]),
+    ...     set([1,3,5]),
+    ...     set([1,2,3,4,5]),
+    ...     set([1,2,3,6])))
+    [(set([1, 3, 5]), [(1, 2)]), (set([1, 2, 3]), [(0, 2), (0, 3), (2, 3)])]
+    >>>
+    """
+    from sympy.utilities.iterables import subsets
+    from collections import defaultdict
+    most = -1
+    for i, j in subsets(list(range(len(sets))), 2):
+        com = sets[i] & sets[j]
+        if com and len(com) > most:
+            best = defaultdict(list)
+            best_keys = []
+            most = len(com)
+        if len(com) == most:
+            if com not in best_keys:
+                best_keys.append(com)
+            best[best_keys.index(com)].append((i,j))
+    if most == -1:
+        return []
+    for k in range(len(best)):
+        best_keys[k] = (best_keys[k], best[k])
+    best_keys.sort(key=lambda x: len(x[1]))
+    return best_keys
+
+
+def opt_cse(exprs, order='canonical', verbose=False):
     """Find optimization opportunities in Adds, Muls, Pows and negative
     coefficient Muls
 
@@ -147,6 +187,8 @@ def opt_cse(exprs, order='canonical'):
     order : string, 'none' or 'canonical'
         The order by which Mul and Add arguments are processed. For large
         expressions where speed is a concern, use the setting order='none'.
+    verbose : bool
+        Print debug information (default=False)
 
     Returns
     -------
@@ -218,51 +260,149 @@ def _match_common_args(Func, funcs):
         else:
             funcs = sorted(funcs, key=lambda x: len(x.args))
 
-        func_args = [set(e.args) for e in funcs]
-        for i in range(len(func_args)):
-            for j in range(i + 1, len(func_args)):
-                com_args = func_args[i].intersection(func_args[j])
-                if len(com_args) > 1:
-                    com_func = Func(*com_args)
-
-                    # for all sets, replace the common symbols by the function
-                    # over them, to allow recursive matches
-
-                    diff_i = func_args[i].difference(com_args)
-                    func_args[i] = diff_i | {com_func}
-                    if diff_i:
-                        opt_subs[funcs[i]] = Func(Func(*diff_i), com_func,
-                                                  evaluate=False)
-
-                    diff_j = func_args[j].difference(com_args)
-                    func_args[j] = diff_j | {com_func}
-                    opt_subs[funcs[j]] = Func(Func(*diff_j), com_func,
-                                              evaluate=False)
-
-                    for k in range(j + 1, len(func_args)):
-                        if not com_args.difference(func_args[k]):
-                            diff_k = func_args[k].difference(com_args)
-                            func_args[k] = diff_k | {com_func}
-                            opt_subs[funcs[k]] = Func(Func(*diff_k), com_func,
-                                                      evaluate=False)
+        if Func is Mul:
+            F = Pow
+            meth = 'as_powers_dict'
+            from sympy.core.add import _addsort as inplace_sorter
+        elif Func is Add:
+            F = Mul
+            meth = 'as_coefficients_dict'
+            from sympy.core.mul import _mulsort as inplace_sorter
+        else:
+            assert None  # expected Mul or Add
+
+        # ----------------- helpers ---------------------------
+        def ufunc(*args):
+            # return a well formed unevaluated function from the args
+            # SHARES Func, inplace_sorter
+            args = list(args)
+            inplace_sorter(args)
+            return Func(*args, evaluate=False)
+
+        def as_dict(e):
+            # creates a dictionary of the expression using either
+            # as_coefficients_dict or as_powers_dict, depending on Func
+            # SHARES meth
+            d = getattr(e, meth, lambda: {a: S.One for a in e.args})()
+            for k in list(d.keys()):
+                try:
+                    as_int(d[k])
+                except ValueError:
+                    d[F(k, d.pop(k))] = S.One
+            return d
+
+        def from_dict(d):
+            # build expression from dict from
+            # as_coefficients_dict or as_powers_dict
+            # SHARES F
+            return ufunc(*[F(k, v) for k, v in d.items()])
+
+        def update(k):
+            # updates all of the info associated with k using
+            # the com_dict: func_dicts, func_args, opt_subs
+            # returns True if all values were updated, else None
+            # SHARES com_dict, com_func, func_dicts, func_args,
+            #        opt_subs, funcs, verbose
+            for di in com_dict:
+                # don't allow a sign to change
+                if com_dict[di] > func_dicts[k][di]:
+                    return
+            # remove it
+            if Func is Add:
+                take = min(func_dicts[k][i] for i in com_dict)
+                com_func_take = Mul(take, from_dict(com_dict), evaluate=False)
+            else:
+                take = igcd(*[func_dicts[k][i] for i in com_dict])
+                com_func_take = Pow(from_dict(com_dict), take, evaluate=False)
+            for di in com_dict:
+                func_dicts[k][di] -= take*com_dict[di]
+            # compute the remaining expression
+            rem = from_dict(func_dicts[k])
+            # reject hollow change, e.g extracting x + 1 from x + 3
+            if Func is Add and rem and rem.is_Integer and 1 in com_dict:
+                return
+            if verbose:
+                print('\nfunc %s (%s) \ncontains %s \nas %s \nleaving %s' %
+                    (funcs[k], func_dicts[k], com_func, com_func_take, rem))
+            # recompute the dict since some keys may now
+            # have corresponding values of 0; one could
+            # keep track of which ones went to zero but
+            # this seems cleaner
+            func_dicts[k] = as_dict(rem)
+            # update associated info
+            func_dicts[k][com_func] = take
+            func_args[k] = set(func_dicts[k])
+            # keep the constant separate from the remaining
+            # part of the expression, e.g. 2*(a*b) rather than 2*a*b
+            opt_subs[funcs[k]] = ufunc(rem, com_func_take)
+            # everything was updated
+            return True
+
+        def get_copy(i):
+            return [func_dicts[i].copy(), func_args[i].copy(), funcs[i], i]
+
+        def restore(dafi):
+            i = dafi.pop()
+            func_dicts[i], func_args[i], funcs[i] = dafi
+
+        # ----------------- end helpers -----------------------
+
+        func_dicts = [as_dict(f) for f in funcs]
+        func_args = [set(d) for d in func_dicts]
+        while True:
+            hit = pairwise_most_common(func_args)
+            if not hit or len(hit[0][0]) <= 1:
+                break
+            changed = False
+            for com_args, ij in hit:
+                take = len(com_args)
+                ALL = list(ordered(com_args))
+                while take >= 2:
+                    for com_args in subsets(ALL, take):
+                        com_func = Func(*com_args)
+                        com_dict = as_dict(com_func)
+                        for i, j in ij:
+                            dafi = None
+                            if com_func != funcs[i]:
+                                dafi = get_copy(i)
+                                ch = update(i)
+                                if not ch:
+                                    restore(dafi)
+                                    continue
+                            if com_func != funcs[j]:
+                                dafj = get_copy(j)
+                                ch = update(j)
+                                if not ch:
+                                    if dafi is not None:
+                                        restore(dafi)
+                                    restore(dafj)
+                                    continue
+                            changed = True
+                        if changed:
+                            break
+                    else:
+                        take -= 1
+                        continue
+                    break
+                else:
+                    continue
+                break
+            if not changed:
+                break
 
     # split muls into commutative
-    comutative_muls = set()
+    commutative_muls = set()
     for m in muls:
         c, nc = m.args_cnc(cset=True)
         if c:
             c_mul = m.func(*c)
             if nc:
-                if c_mul == 1:
-                    new_obj = m.func(*nc)
-                else:
-                    new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
-                opt_subs[m] = new_obj
+                opt_subs[m] = m.func(c_mul, m.func(*nc), evaluate=False)
             if len(c) > 1:
-                comutative_muls.add(c_mul)
+                commutative_muls.add(c_mul)
 
     _match_common_args(Add, adds)
-    _match_common_args(Mul, comutative_muls)
+    _match_common_args(Mul, commutative_muls)
 
     return opt_subs
 
@@ -394,6 +534,30 @@ def _rebuild(expr):
             reduced_e = e
         reduced_exprs.append(reduced_e)
 
+    # don't allow hollow nesting
+    # e.g if p = [b + 2*d + e + f, b + 2*d + f + g, a + c + d + f + g]
+    # and R, C = cse(p) then
+    #     R = [(x0, d + f), (x1, b + d)]
+    #     C = [e + x0 + x1, g + x0 + x1, a + c + d + f + g]
+    # but the args of C[-1] should not be `(a + c, d + f + g)`
+    nested = [[i for i in f.args if isinstance(i, f.func)] for f in exprs]
+    for i in range(len(exprs)):
+        F = reduced_exprs[i].func
+        if not (F is Mul or F is Add):
+            continue
+        nested = [a for a in exprs[i].args if isinstance(a, F)]
+        args = []
+        for a in reduced_exprs[i].args:
+            if isinstance(a, F):
+                for ai in a.args:
+                    if isinstance(ai, F) and ai not in nested:
+                        args.extend(ai.args)
+                    else:
+                        args.append(ai)
+            else:
+                args.append(a)
+        reduced_exprs[i] = F(*args)
+
     return replacements, reduced_exprs
 
 
@@ -444,7 +608,7 @@ def cse(exprs, symbols=None, optimizations=None, postprocess=None,
     >>> from sympy import cse, SparseMatrix
     >>> from sympy.abc import x, y, z, w
     >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
-    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])
+    ([(x0, w + y + z)], [x0*(x + x0)/(w + x)**3])
 
     Note that currently, y + z will not get substituted if -y - z is used.
 
diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py
--- a/sympy/solvers/ode.py
+++ b/sympy/solvers/ode.py
@@ -4338,8 +4338,6 @@ def ode_linear_coefficients(eq, func, order, match):
     >>> f = Function('f')
     >>> df = f(x).diff(x)
     >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
-    >>> dsolve(eq, hint='linear_coefficients')
-    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
     >>> pprint(dsolve(eq, hint='linear_coefficients'))
                       ___________                     ___________
                    /         2                     /         2
@@ -4403,8 +4401,6 @@ def ode_separable_reduced(eq, func, order, match):
     >>> f = Function('f')
     >>> d = f(x).diff(x)
     >>> eq = (x - x**2*f(x))*d - f(x)
-    >>> dsolve(eq, hint='separable_reduced')
-    [Eq(f(x), (-sqrt(C1*x**2 + 1) + 1)/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
     >>> pprint(dsolve(eq, hint='separable_reduced'))
                  ___________                ___________
                 /     2                    /     2

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_subs.py b/sympy/core/tests/test_subs.py
--- a/sympy/core/tests/test_subs.py
+++ b/sympy/core/tests/test_subs.py
@@ -595,9 +595,9 @@ def test_issue_6559():
     # though this involves cse it generated a failure in Mul._eval_subs
     x0, x1 = symbols('x0 x1')
     e = -log(-12*sqrt(2) + 17)/24 - log(-2*sqrt(2) + 3)/12 + sqrt(2)/3
-    # XXX modify cse so x1 is eliminated and x0 = -sqrt(2)?
     assert cse(e) == (
-        [(x0, sqrt(2))], [x0/3 - log(-12*x0 + 17)/24 - log(-2*x0 + 3)/12])
+        [(x0, sqrt(2))],
+        [x0/3 - log(-12*x0 + 17)/24 - log(-2*x0 + 3)/12])
 
 
 def test_issue_5261():
diff --git a/sympy/matrices/expressions/tests/test_matmul.py b/sympy/matrices/expressions/tests/test_matmul.py
--- a/sympy/matrices/expressions/tests/test_matmul.py
+++ b/sympy/matrices/expressions/tests/test_matmul.py
@@ -130,4 +130,4 @@ def test_matmul_no_matrices():
 def test_matmul_args_cnc():
     a, b = symbols('a b', commutative=False)
     assert MatMul(n, a, b, A, A.T).args_cnc() == ([n], [a, b, A, A.T])
-    assert MatMul(A, A.T).args_cnc() == ([1], [A, A.T])
+    assert MatMul(A, A.T).args_cnc() == ([], [A, A.T])
diff --git a/sympy/simplify/tests/test_cse.py b/sympy/simplify/tests/test_cse.py
--- a/sympy/simplify/tests/test_cse.py
+++ b/sympy/simplify/tests/test_cse.py
@@ -2,10 +2,11 @@
 
 from sympy import (Add, Pow, Symbol, exp, sqrt, symbols, sympify, cse,
                    Matrix, S, cos, sin, Eq, Function, Tuple, CRootOf,
-                   IndexedBase, Idx, Piecewise, O)
+                   IndexedBase, Idx, Piecewise, O, Mul)
 from sympy.simplify.cse_opts import sub_pre, sub_post
 from sympy.functions.special.hyper import meijerg
 from sympy.simplify import cse_main, cse_opts
+from sympy.utilities.iterables import subsets
 from sympy.utilities.pytest import XFAIL, raises
 from sympy.matrices import (eye, SparseMatrix, MutableDenseMatrix,
     MutableSparseMatrix, ImmutableDenseMatrix, ImmutableSparseMatrix)
@@ -15,7 +16,7 @@
 
 
 w, x, y, z = symbols('w,x,y,z')
-x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = symbols('x:13')
+x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18 = symbols('x:19')
 
 
 def test_numbered_symbols():
@@ -174,9 +175,17 @@ def test_non_commutative_order():
     assert cse(l) == ([(x0, B+C)], [x0, A*x0])
 
 
-@XFAIL
-def test_powers():
-    assert cse(x*y**2 + x*y) == ([(x0, x*y)], [x0*y + x0])
+def test_issue_10228():
+    assert cse([x*y**2 + x*y]) == ([(x0, x*y)], [x0*y + x0])
+    assert cse([x + y, 2*x + y]) == ([(x0, x + y)], [x0, x + x0])
+    assert cse((w + 2*x + y + z, w + x + 1)) == (
+        [(x0, w + x)], [x0 + x + y + z, x0 + 1])
+    assert cse(((w + x + y + z)*(w - x))/(w + x)) == (
+        [(x0, w + x)], [(x0 + y + z)*(w - x)/x0])
+    a, b, c, d, f, g, j, m = symbols('a, b, c, d, f, g, j, m')
+    exprs = (d*g**2*j*m, 4*a*f*g*m, a*b*c*f**2)
+    assert cse(exprs) == (
+        [(x0, g*m), (x1, a*f)], [d*g*j*x0, 4*x0*x1, b*c*f*x1])
 
 
 def test_issue_4498():
@@ -264,12 +273,21 @@ def test_issue_4499():
         sqrt(z))*G(b)*G(2*a - b + 1), 1, 0, S(1)/2, z/2, -b + 1, -2*a + b,
         -2*a))
     c = cse(t)
+    # check rebuild
+    r = c[0]
+    tt = list(c[1][0])
+    for i in range(len(tt)):
+        for re in reversed(r):
+            tt[i] = tt[i].subs(*re)
+        assert tt[i] == t[i]
+    # check answer
     ans = (
-        [(x0, 2*a), (x1, -b), (x2, x1 + 1), (x3, x0 + x2), (x4, sqrt(z)), (x5,
-        B(x0 + x1, x4)), (x6, G(b)), (x7, G(x3)), (x8, -x0), (x9,
-        (x4/2)**(x8 + 1)), (x10, x6*x7*x9*B(b - 1, x4)), (x11, x6*x7*x9*B(b,
-        x4)), (x12, B(x3, x4))], [(a, a + S(1)/2, x0, b, x3, x10*x5,
-        x11*x4*x5, x10*x12*x4, x11*x12, 1, 0, S(1)/2, z/2, x2, b + x8, x8)])
+        [(x0, 2*a), (x1, -b), (x2, x0 + x1 + 1), (x3, sqrt(z)), (x4,
+        B(x0 + x1, x3)), (x5, G(b)), (x6, G(x2)), (x7, -x0), (x8,
+        (x3/2)**(x7 + 1)), (x9, x5*x6*x8*B(b - 1, x3)), (x10,
+        x5*x6*x8*B(b, x3)), (x11, B(x2, x3))], [(a, a + 1/2, x0, b,
+        x2, x4*x9, x10*x3*x4, x11*x3*x9, x10*x11, 1, 0, 1/2, z/2, x1 +
+        1, b + x7, x7)])
     assert ans == c
 
 
@@ -303,12 +321,13 @@ def test_cse_MatrixSymbol():
     B = MatrixSymbol("B", n, n)
     assert cse(B) == ([], [B])
 
+
 def test_cse_MatrixExpr():
     from sympy import MatrixSymbol
     A = MatrixSymbol('A', 3, 3)
     y = MatrixSymbol('y', 3, 1)
 
-    expr1 = (A.T*A).I * A * y
+    expr1 = 2*(A.T*A).I * A * y
     expr2 = (A.T*A) * A * y
     replacements, reduced_exprs = cse([expr1, expr2])
     assert len(replacements) > 0
@@ -319,6 +338,7 @@ def test_cse_MatrixExpr():
     replacements, reduced_exprs = cse([A**2, A + A**2])
     assert replacements
 
+
 def test_Piecewise():
     f = Piecewise((-z + x*y, Eq(y, 0)), (-z - x*y, True))
     ans = cse(f)
@@ -399,3 +419,61 @@ def test_issue_8891():
         ans = ([(x0, x + y)], [x0, cls([[x0, 0], [0, 0]])])
         assert res == ans
         assert isinstance(res[1][-1], cls)
+
+
+def test_issue_11230():
+    from random import choice
+    from sympy.core.function import expand_mul
+    s = symbols('a:m')
+    # 35 Mul tests, none of which should ever fail
+    ex = [Mul(*[choice(s) for i in range(5)]) for i in range(7)]
+    for p in subsets(ex, 3):
+        p = list(p)
+        R, C = cse(p)
+        assert not any(i.is_Mul for a in C for i in a.args)
+        for ri in reversed(R):
+            for i in range(len(C)):
+                C[i] = C[i].subs(*ri)
+        assert p == C
+    # 35 Add tests, none of which should ever fail
+    ex = [Add(*[choice(s[:7]) for i in range(5)]) for i in range(7)]
+    for p in subsets(ex, 3):
+        p = list(p)
+        was = R, C = cse(p)
+        assert not any(i.is_Add for a in C for i in a.args)
+        for ri in reversed(R):
+            for i in range(len(C)):
+                C[i] = C[i].subs(*ri)
+        # use expand_mul to handle cases like this:
+        # p = [a + 2*b + 2*e, 2*b + c + 2*e, b + 2*c + 2*g]
+        # x0 = 2*(b + e) is identified giving a rebuilt p that
+        # is now `[a + 2*(b + e), c + 2*(b + e), b + 2*c + 2*g]`
+        assert p == [expand_mul(i) for i in C]
+
+
+@XFAIL
+def test_issue_11577():
+    def check(eq):
+        from sympy.core.function import count_ops
+        r, c = cse(eq)
+        assert eq.count_ops() >= \
+            len(r) + sum([i[1].count_ops() for i in r]) + \
+            count_ops(c)
+
+    eq = x**5*y**2 + x**5*y + x**5
+    assert cse(eq) == (
+        [(x0, x**4), (x1, x*y)], [x**5 + x0*x1*y + x0*x1])
+        # ([(x0, x**5*y)], [x0*y + x0 + x**5]) or
+        # ([(x0, x**5)], [x0*y**2 + x0*y + x0])
+    check(eq)
+
+    eq = x**2/(y + 1)**2 + x/(y + 1)
+    assert cse(eq) == (
+        [(x0, y + 1)], [x**2/x0**2 + x/x0])
+        # ([(x0, x/(y + 1))], [x0**2 + x0])
+    check(eq)
+
+
+def test_hollow_rejection():
+    eq = [x + 3, x + 4]
+    assert cse(eq) == ([], eq)

```


## Code snippets

### 1 - sympy/simplify/cse_main.py:

Start line: 173, End line: 213

```python
def opt_cse(exprs, order='canonical'):
    # ... other code

    def _find_opts(expr):

        if not isinstance(expr, Basic):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            list(map(_find_opts, expr))
            return

        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)

        list(map(_find_opts, expr.args))

        if _coeff_isneg(expr):
            neg_expr = -expr
            if not neg_expr.is_Atom:
                opt_subs[expr] = Mul(S.NegativeOne, neg_expr, evaluate=False)
                seen_subexp.add(neg_expr)
                expr = neg_expr

        if isinstance(expr, (Mul, MatMul)):
            muls.add(expr)

        elif isinstance(expr, (Add, MatAdd)):
            adds.add(expr)

        elif isinstance(expr, (Pow, MatPow)):
            if _coeff_isneg(expr.exp):
                opt_subs[expr] = Pow(Pow(expr.base, -expr.exp), S.NegativeOne,
                                     evaluate=False)

    for e in exprs:
        if isinstance(e, Basic):
            _find_opts(e)

    ## Process Adds and commutative Muls
    # ... other code
```
### 2 - sympy/simplify/cse_main.py:

Start line: 249, End line: 267

```python
def opt_cse(exprs, order='canonical'):

    # split muls into commutative
    comutative_muls = set()
    for m in muls:
        c, nc = m.args_cnc(cset=True)
        if c:
            c_mul = m.func(*c)
            if nc:
                if c_mul == 1:
                    new_obj = m.func(*nc)
                else:
                    new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                opt_subs[m] = new_obj
            if len(c) > 1:
                comutative_muls.add(c_mul)

    _match_common_args(Add, adds)
    _match_common_args(Mul, comutative_muls)

    return opt_subs
```
### 3 - sympy/simplify/cse_main.py:

Start line: 332, End line: 397

```python
def tree_cse(exprs, symbols, opt_subs=None, order='canonical'):
    # ... other code

    def _rebuild(expr):
        if not isinstance(expr, Basic):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)

    return replacements, reduced_exprs
```
### 4 - sympy/simplify/cse_main.py:

Start line: 467, End line: 539

```python
def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
    from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                                SparseMatrix, ImmutableSparseMatrix)

    # Handle the case if just one expression was passed.
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]

    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e._mat))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e._smat.items()))
        else:
            temp.append(e)
    exprs = temp
    del temp

    if optimizations is None:
        optimizations = list()
    elif optimizations == 'basic':
        optimizations = basic_optimizations

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    excluded_symbols = set().union(*[expr.atoms(Symbol)
                                   for expr in reduced_exprs])

    if symbols is None:
        symbols = numbered_symbols()
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    symbols = filter_symbols(symbols, excluded_symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs, order)

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order)

    # Postprocess the expressions to return the expressions to canonical form.
    exprs = copy
    for i, (sym, subtree) in enumerate(replacements):
        subtree = postprocess_for_cse(subtree, optimizations)
        replacements[i] = (sym, subtree)
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]

    # Get the matrices back
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)
```
### 5 - sympy/simplify/cse_main.py:

Start line: 215, End line: 247

```python
def opt_cse(exprs, order='canonical'):
    # ... other code

    def _match_common_args(Func, funcs):
        if order != 'none':
            funcs = list(ordered(funcs))
        else:
            funcs = sorted(funcs, key=lambda x: len(x.args))

        func_args = [set(e.args) for e in funcs]
        for i in range(len(func_args)):
            for j in range(i + 1, len(func_args)):
                com_args = func_args[i].intersection(func_args[j])
                if len(com_args) > 1:
                    com_func = Func(*com_args)

                    # for all sets, replace the common symbols by the function
                    # over them, to allow recursive matches

                    diff_i = func_args[i].difference(com_args)
                    func_args[i] = diff_i | {com_func}
                    if diff_i:
                        opt_subs[funcs[i]] = Func(Func(*diff_i), com_func,
                                                  evaluate=False)

                    diff_j = func_args[j].difference(com_args)
                    func_args[j] = diff_j | {com_func}
                    opt_subs[funcs[j]] = Func(Func(*diff_j), com_func,
                                              evaluate=False)

                    for k in range(j + 1, len(func_args)):
                        if not com_args.difference(func_args[k]):
                            diff_k = func_args[k].difference(com_args)
                            func_args[k] = diff_k | {com_func}
                            opt_subs[funcs[k]] = Func(Func(*diff_k), com_func,
                                                      evaluate=False)
    # ... other code
```
### 6 - sympy/simplify/cse_main.py:

Start line: 1, End line: 29

```python
""" Tools for doing common subexpression elimination.
"""
from __future__ import print_function, division

from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple
from sympy.core.singleton import S
from sympy.core.function import _coeff_isneg
from sympy.core.exprtools import factor_terms
from sympy.core.compatibility import iterable, range
from sympy.utilities.iterables import filter_symbols, \
    numbered_symbols, sift, topological_sort, ordered

from . import cse_opts

# (preprocessor, postprocessor) pairs which are commonly useful. They should
# each take a sympy expression and return a possibly transformed expression.
# When used in the function ``cse()``, the target expressions will be transformed
# by each of the preprocessor functions in order. After the common
# subexpressions are eliminated, each resulting expression will have the
# postprocessor functions transform them in *reverse* order in order to undo the
# transformation if necessary. This allows the algorithm to operate on
# a representation of the expressions that allows for more optimization
# opportunities.
# ``None`` can be used to specify no transformation for either the preprocessor or
# postprocessor.


basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post),
                       (factor_terms, None)]
```
### 7 - sympy/solvers/solvers.py:

Start line: 3179, End line: 3277

```python
def unrad(eq, *syms, **flags):

    if len(rterms) == 1 and not (rterms[0].is_Add and lcm > 2):
    else:
        # ... other code

        if len(rterms) == 2:
            if not others:
                eq = rterms[0]**lcm - (-rterms[1])**lcm
                ok = True
            elif not log(lcm, 2).is_Integer:
                # the lcm-is-power-of-two case is handled below
                r0, r1 = rterms
                if flags.get('_reverse', False):
                    r1, r0 = r0, r1
                i0 = _rads0, _bases0, lcm0 = _rads_bases_lcm(r0.as_poly())
                i1 = _rads1, _bases1, lcm1 = _rads_bases_lcm(r1.as_poly())
                for reverse in range(2):
                    if reverse:
                        i0, i1 = i1, i0
                        r0, r1 = r1, r0
                    _rads1, _, lcm1 = i1
                    _rads1 = Mul(*_rads1)
                    t1 = _rads1**lcm1
                    c = covsym**lcm1 - t1
                    for x in syms:
                        try:
                            sol = _solve(c, x, **uflags)
                            if not sol:
                                raise NotImplementedError
                            neweq = r0.subs(x, sol[0]) + covsym*r1/_rads1 + \
                                others
                            tmp = unrad(neweq, covsym)
                            if tmp:
                                eq, newcov = tmp
                                if newcov:
                                    newp, newc = newcov
                                    _cov(newp, c.subs(covsym,
                                        _solve(newc, covsym, **uflags)[0]))
                                else:
                                    _cov(covsym, c)
                            else:
                                eq = neweq
                                _cov(covsym, c)
                            ok = True
                            break
                        except NotImplementedError:
                            if reverse:
                                raise NotImplementedError(
                                    'no successful change of variable found')
                            else:
                                pass
                    if ok:
                        break
        elif len(rterms) == 3:
            # two cube roots and another with order less than 5
            # (so an analytical solution can be found) or a base
            # that matches one of the cube root bases
            info = [_rads_bases_lcm(i.as_poly()) for i in rterms]
            RAD = 0
            BASES = 1
            LCM = 2
            if info[0][LCM] != 3:
                info.append(info.pop(0))
                rterms.append(rterms.pop(0))
            elif info[1][LCM] != 3:
                info.append(info.pop(1))
                rterms.append(rterms.pop(1))
            if info[0][LCM] == info[1][LCM] == 3:
                if info[1][BASES] != info[2][BASES]:
                    info[0], info[1] = info[1], info[0]
                    rterms[0], rterms[1] = rterms[1], rterms[0]
                if info[1][BASES] == info[2][BASES]:
                    eq = rterms[0]**3 + (rterms[1] + rterms[2] + others)**3
                    ok = True
                elif info[2][LCM] < 5:
                    # a*root(A, 3) + b*root(B, 3) + others = c
                    a, b, c, d, A, B = [Dummy(i) for i in 'abcdAB']
                    # zz represents the unraded expression into which the
                    # specifics for this case are substituted
                    zz = (c - d)*(A**3*a**9 + 3*A**2*B*a**6*b**3 -
                        3*A**2*a**6*c**3 + 9*A**2*a**6*c**2*d - 9*A**2*a**6*c*d**2 +
                        3*A**2*a**6*d**3 + 3*A*B**2*a**3*b**6 + 21*A*B*a**3*b**3*c**3 -
                        63*A*B*a**3*b**3*c**2*d + 63*A*B*a**3*b**3*c*d**2 -
                        21*A*B*a**3*b**3*d**3 + 3*A*a**3*c**6 - 18*A*a**3*c**5*d +
                        45*A*a**3*c**4*d**2 - 60*A*a**3*c**3*d**3 + 45*A*a**3*c**2*d**4 -
                        18*A*a**3*c*d**5 + 3*A*a**3*d**6 + B**3*b**9 - 3*B**2*b**6*c**3 +
                        9*B**2*b**6*c**2*d - 9*B**2*b**6*c*d**2 + 3*B**2*b**6*d**3 +
                        3*B*b**3*c**6 - 18*B*b**3*c**5*d + 45*B*b**3*c**4*d**2 -
                        60*B*b**3*c**3*d**3 + 45*B*b**3*c**2*d**4 - 18*B*b**3*c*d**5 +
                        3*B*b**3*d**6 - c**9 + 9*c**8*d - 36*c**7*d**2 + 84*c**6*d**3 -
                        126*c**5*d**4 + 126*c**4*d**5 - 84*c**3*d**6 + 36*c**2*d**7 -
                        9*c*d**8 + d**9)
                    def _t(i):
                        b = Mul(*info[i][RAD])
                        return cancel(rterms[i]/b), Mul(*info[i][BASES])
                    aa, AA = _t(0)
                    bb, BB = _t(1)
                    cc = -rterms[2]
                    dd = others
                    eq = zz.xreplace(dict(zip(
                        (a, A, b, B, c, d),
                        (aa, AA, bb, BB, cc, dd))))
                    ok = True
        # handle power-of-2 cases
        # ... other code
    # ... other code
```
### 8 - sympy/simplify/cse_main.py:

Start line: 400, End line: 466

```python
def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of sympy expressions, or a single sympy expression
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out. The ``numbered_symbols`` generator is useful. The default is a
        stream of symbols of the form "x0", "x1", etc. This must be an
        infinite iterator.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs of external optimization
        functions. Optionally 'basic' can be passed for a set of predefined
        basic optimizations. Such 'basic' optimizations were used by default
        in old implementation, however they can be really slow on larger
        expressions. Now, no pre or post optimizations are made by default.
    postprocess : a function which accepts the two return values of cse and
        returns the desired form of output from cse, e.g. if you want the
        replacements reversed the function might be the following lambda:
        lambda r, e: return reversed(r), e
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. If set to
        'canonical', arguments will be canonically ordered. If set to 'none',
        ordering will be faster but dependent on expressions hashes, thus
        machine dependent and variable. For large expressions where speed is a
        concern, use the setting order='none'.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of sympy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])

    Note that currently, y + z will not get substituted if -y - z is used.

     >>> cse(((w + x + y + z)*(w - y - z))/(w + x)**3)
     ([(x0, w + x)], [(w - y - z)*(x0 + y + z)/x0**3])

    List of expressions with recursive substitutions:

    >>> m = SparseMatrix([x + y, x + y + z])
    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])
    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([
    [x0],
    [x1]])])

    Note: the type and mutability of input matrices is retained.

    >>> isinstance(_[1][-1], SparseMatrix)
    True
    """
    # ... other code
```
### 9 - sympy/simplify/cse_opts.py:

Start line: 1, End line: 44

```python
""" Optimizations of the expression tree representation for better CSE
opportunities.
"""
from __future__ import print_function, division

from sympy.core import Add, Basic, Mul
from sympy.core.basic import preorder_traversal
from sympy.core.singleton import S
from sympy.utilities.iterables import default_sort_key


def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
    reps = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]

    # make it canonical
    reps.sort(key=default_sort_key)

    e = e.xreplace(dict((a, Mul._from_args([S.NegativeOne, -a])) for a in reps))
    # repeat again for persisting Adds but mark these with a leading 1, -1
    # e.g. y - x -> 1*-1*(x - y)
    if isinstance(e, Basic):
        negs = {}
        for a in sorted(e.atoms(Add), key=default_sort_key):
            if a in reps or a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        e = e.xreplace(negs)
    return e


def sub_post(e):
    """ Replace 1*-1*x with -x.
    """
    replacements = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and \
            node.args[0] is S.One and node.args[1] is S.NegativeOne:
            replacements.append((node, -Mul._from_args(node.args[2:])))
    for node, replacement in replacements:
        e = e.xreplace({node: replacement})

    return e
```
### 10 - sympy/simplify/cse_main.py:

Start line: 65, End line: 88

```python
def cse_separate(r, e):
    """Move expressions that are in the form (symbol, expr) out of the
    expressions and sort them into the replacements using the reps_toposort.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse_separate
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, exp, cse, Eq, symbols
    >>> x0, x1 = symbols('x:2')
    >>> eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    >>> cse([eq, Eq(x, z + 1), z - 2], postprocess=cse_separate) in [
    ... [[(x0, y + 1), (x, z + 1), (x1, x + 1)],
    ...  [x1 + exp(x1/x0) + cos(x0), z - 2]],
    ... [[(x1, y + 1), (x, z + 1), (x0, x + 1)],
    ...  [x0 + exp(x0/x1) + cos(x1), z - 2]]]
    ...
    True
    """
    d = sift(e, lambda w: w.is_Equality and w.lhs.is_Symbol)
    r = r + [w.args for w in d[True]]
    e = d[False]
    return [reps_toposort(r), e]
```
### 18 - sympy/simplify/cse_main.py:

Start line: 270, End line: 330

```python
def tree_cse(exprs, symbols, opt_subs=None, order='canonical'):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of sympy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    """
    from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd

    if opt_subs is None:
        opt_subs = dict()

    ## Find repeated sub-expressions

    to_eliminate = set()

    seen_subexp = set()

    def _find_repeated(expr):
        if not isinstance(expr, Basic):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            args = expr

        else:
            if expr in seen_subexp:
                to_eliminate.add(expr)
                return

            seen_subexp.add(expr)

            if expr in opt_subs:
                expr = opt_subs[expr]

            args = expr.args

        list(map(_find_repeated, args))

    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)

    ## Rebuild tree

    replacements = []

    subs = dict()
    # ... other code
```
### 20 - sympy/simplify/cse_main.py:

Start line: 139, End line: 171

```python
def opt_cse(exprs, order='canonical'):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls

    Parameters
    ----------
    exprs : list of sympy expressions
        The expressions to optimize.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.

    Returns
    -------
    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.

    Examples
    ========

    >>> from sympy.simplify.cse_main import opt_cse
    >>> from sympy.abc import x
    >>> opt_subs = opt_cse([x**-2])
    >>> print(opt_subs)
    {x**(-2): 1/(x**2)}
    """
    from sympy.matrices.expressions import MatAdd, MatMul, MatPow
    opt_subs = dict()

    adds = set()
    muls = set()

    seen_subexp = set()
    # ... other code
```
### 22 - sympy/printing/llvmjitcode.py:

Start line: 359, End line: 437

```python
@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    '''Compile function from a Sympy expression

    Expressions are evaluated using double precision arithmetic.
    Some single argument math functions (exp, sin, cos, etc.) are supported
    in expressions.

    Parameters
    ==========
    args : List of Symbol
        Arguments to the generated function.  Usually the free symbols in
        the expression.  Currently each one is assumed to convert to
        a double precision scalar.
    expr : Expr, or (Replacements, Expr) as returned from 'cse'
        Expression to compile.
    callback_type : string
        Create function with signature appropriate to use as a callback.
        Currently supported:
           'scipy.integrate'
           'scipy.integrate.test'
           'cubature'

    Returns
    =======
    Compiled function that can evaluate the expression.

    Examples
    ========
    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> e = a*a + a + 1
    >>> e1 = jit.llvm_callable([a], e)
    >>> e.subs(a, 1.1)   # Evaluate via substitution
    3.31000000000000
    >>> e1(1.1)  # Evaluate using JIT-compiled code
    3.3100000000000005


    Callbacks for integration functions can be JIT compiled.
    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> from sympy import integrate
    >>> from scipy.integrate import quad
    >>> e = a*a
    >>> e1 = jit.llvm_callable([a], e, callback_type='scipy.integrate')
    >>> integrate(e, (a, 0.0, 2.0))
    2.66666666666667
    >>> quad(e1, 0.0, 2.0)[0]
    2.66666666666667

    The 'cubature' callback is for the Python wrapper around the
    cubature package ( https://github.com/saullocastro/cubature )
    and ( http://ab-initio.mit.edu/wiki/index.php/Cubature )

    There are two signatures for the SciPy integration callbacks.
    The first ('scipy.integrate') is the function to be passed to the
    integration routine, and will pass the signature checks.
    The second ('scipy.integrate.test') is only useful for directly calling
    the function using ctypes variables. It will not pass the signature checks
    for scipy.integrate.

    The return value from the cse module can also be compiled.  This
    can improve the performance of the compiled function.  If multiple
    expressions are given to cse, the compiled function returns a tuple.
    The 'cubature' callback handles multiple expressions (set `fdim`
    to match in the integration call.)
    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy import cse, exp
    >>> from sympy.abc import x,y
    >>> e1 = x*x + y*y
    >>> e2 = 4*(x*x + y*y) + 8.0
    >>> after_cse = cse([e1,e2])
    >>> after_cse
    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
    >>> j1 = jit.llvm_callable([x,y], after_cse)
    >>> j1(1.0, 2.0)
    (5.0, 28.0)
    '''
    # ... other code
```
### 25 - sympy/solvers/ode.py:

Start line: 1988, End line: 2571

```python
@vectorize(0)
def odesimp(eq, func, order, constants, hint):
    r"""
    Simplifies ODEs, including trying to solve for ``func`` and running
    :py:meth:`~sympy.solvers.ode.constantsimp`.

    It may use knowledge of the type of solution that the hint returns to
    apply additional simplifications.

    It also attempts to integrate any :py:class:`~sympy.integrals.Integral`\s
    in the expression, if the hint is not an ``_Integral`` hint.

    This function should have no effect on expressions returned by
    :py:meth:`~sympy.solvers.ode.dsolve`, as
    :py:meth:`~sympy.solvers.ode.dsolve` already calls
    :py:meth:`~sympy.solvers.ode.odesimp`, but the individual hint functions
    do not call :py:meth:`~sympy.solvers.ode.odesimp` (because the
    :py:meth:`~sympy.solvers.ode.dsolve` wrapper does).  Therefore, this
    function is designed for mainly internal use.

    Examples
    ========

    >>> from sympy import sin, symbols, dsolve, pprint, Function
    >>> from sympy.solvers.ode import odesimp
    >>> x , u2, C1= symbols('x,u2,C1')
    >>> f = Function('f')

    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    ... simplify=False)
    >>> pprint(eq, wrap_line=False)
                            x
                           ----
                           f(x)
                             /
                            |
                            |   /        1   \
                            |  -|u2 + -------|
                            |   |        /1 \|
                            |   |     sin|--||
                            |   \        \u2//
    log(f(x)) = log(C1) +   |  ---------------- d(u2)
                            |          2
                            |        u2
                            |
                           /

    >>> pprint(odesimp(eq, f(x), 1, set([C1]),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'
    ... )) #doctest: +SKIP
        x
    --------- = C1
       /f(x)\
    tan|----|
       \2*x /

    """
    x = func.args[0]
    f = func.func
    C1 = get_numbered_constants(eq, num=1)

    # First, integrate if the hint allows it.
    eq = _handle_Integral(eq, func, order, hint)
    if hint.startswith("nth_linear_euler_eq_nonhomogeneous"):
        eq = simplify(eq)
    if not isinstance(eq, Equality):
        raise TypeError("eq should be an instance of Equality")

    # Second, clean up the arbitrary constants.
    # Right now, nth linear hints can put as many as 2*order constants in an
    # expression.  If that number grows with another hint, the third argument
    # here should be raised accordingly, or constantsimp() rewritten to handle
    # an arbitrary number of constants.
    eq = constantsimp(eq, constants)

    # Lastly, now that we have cleaned up the expression, try solving for func.
    # When CRootOf is implemented in solve(), we will want to return a CRootOf
    # everytime instead of an Equality.

    # Get the f(x) on the left if possible.
    if eq.rhs == func and not eq.lhs.has(func):
        eq = [Eq(eq.rhs, eq.lhs)]

    # make sure we are working with lists of solutions in simplified form.
    if eq.lhs == func and not eq.rhs.has(func):
        # The solution is already solved
        eq = [eq]

        # special simplification of the rhs
        if hint.startswith("nth_linear_constant_coeff"):
            # Collect terms to make the solution look nice.
            # This is also necessary for constantsimp to remove unnecessary
            # terms from the particular solution from variation of parameters
            #
            # Collect is not behaving reliably here.  The results for
            # some linear constant-coefficient equations with repeated
            # roots do not properly simplify all constants sometimes.
            # 'collectterms' gives different orders sometimes, and results
            # differ in collect based on that order.  The
            # sort-reverse trick fixes things, but may fail in the
            # future. In addition, collect is splitting exponentials with
            # rational powers for no reason.  We have to do a match
            # to fix this using Wilds.
            global collectterms
            try:
                collectterms.sort(key=default_sort_key)
                collectterms.reverse()
            except Exception:
                pass
            assert len(eq) == 1 and eq[0].lhs == f(x)
            sol = eq[0].rhs
            sol = expand_mul(sol)
            for i, reroot, imroot in collectterms:
                sol = collect(sol, x**i*exp(reroot*x)*sin(abs(imroot)*x))
                sol = collect(sol, x**i*exp(reroot*x)*cos(imroot*x))
            for i, reroot, imroot in collectterms:
                sol = collect(sol, x**i*exp(reroot*x))
            del collectterms

            # Collect is splitting exponentials with rational powers for
            # no reason.  We call powsimp to fix.
            sol = powsimp(sol)

            eq[0] = Eq(f(x), sol)

    else:
        # The solution is not solved, so try to solve it
        try:
            floats = any(i.is_Float for i in eq.atoms(Number))
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            eq = [eq]
        else:
            def _expand(expr):
                numer, denom = expr.as_numer_denom()

                if denom.is_Add:
                    return expr
                else:
                    return powsimp(expr.expand(), combine='exp', deep=True)

            # XXX: the rest of odesimp() expects each ``t`` to be in a
            # specific normal form: rational expression with numerator
            # expanded, but with combined exponential functions (at
            # least in this setup all tests pass).
            eq = [Eq(f(x), _expand(t)) for t in eqsol]

        # special simplification of the lhs.
        if hint.startswith("1st_homogeneous_coeff"):
            for j, eqi in enumerate(eq):
                newi = logcombine(eqi, force=True)
                if newi.lhs.func is log and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0]/C1, C1)
                eq[j] = newi

    # We cleaned up the constants before solving to help the solve engine with
    # a simpler expression, but the solved expression could have introduced
    # things like -C1, so rerun constantsimp() one last time before returning.
    for i, eqi in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        eq[i] = constant_renumber(eq[i], 'C', 1, 2*order)

    # If there is only 1 solution, return it;
    # otherwise return the list of solutions.
    if len(eq) == 1:
        eq = eq[0]
    return eq

def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    r"""
    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.

    This only works when ``func`` is one function, like `f(x)`.  ``sol`` can
    be a single solution or a list of solutions.  Each solution may be an
    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,
    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an
    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it
    will not be necessary to explicitly identify the function, but if the
    function cannot be inferred from the original equation it can be supplied
    through the ``func`` argument.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    It tries the following methods, in order, until it finds zero equivalence:

    1. Substitute the solution for `f` in the original equation.  This only
       works if ``ode`` is solved for `f`.  It will attempt to solve it first
       unless ``solve_for_func == False``.
    2. Take `n` derivatives of the solution, where `n` is the order of
       ``ode``, and check to see if that is equal to the solution.  This only
       works on exact ODEs.
    3. Take the 1st, 2nd, ..., `n`\th derivatives of the solution, each time
       solving for the derivative of `f` of that order (this will always be
       possible because `f` is a linear operator). Then back substitute each
       derivative into ``ode`` in reverse order.

    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results in ``0``, and ``False`` otherwise. The second
    item in the tuple is what the substitution results in.  It should always
    be ``0`` if the first item is ``True``. Note that sometimes this function
    will ``False``, but with an expression that is identically equal to ``0``,
    instead of returning ``True``.  This is because
    :py:meth:`~sympy.simplify.simplify.simplify` cannot reduce the expression
    to ``0``.  If an expression returned by this function vanishes
    identically, then ``sol`` really is a solution to ``ode``.

    If this function seems to hang, it is probably because of a hard
    simplification.

    To use this function to test, test the first item of the tuple.

    Examples
    ========

    >>> from sympy import Eq, Function, checkodesol, symbols
    >>> x, C1 = symbols('x,C1')
    >>> f = Function('f')
    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))
    (True, 0)
    >>> assert checkodesol(f(x).diff(x), C1)[0]
    >>> assert not checkodesol(f(x).diff(x), x)[0]
    >>> checkodesol(f(x).diff(x, 2), x**2)
    (False, 2)

    """
    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)
    if func is None:
        try:
            _, func = _preprocess(ode.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkodesol for this case.')
            func = funcs.pop()
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError(
            "func must be a function of one variable, not %s" % func)
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])

    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    if order == 'auto':
        order = ode_order(ode, func)
    solved = sol.lhs == func and not sol.rhs.has(func)
    if solve_for_func and not solved:
        rhs = solve(sol, func)
        if rhs:
            eqs = [Eq(func, t) for t in rhs]
            if len(rhs) == 1:
                eqs = eqs[0]
            return checkodesol(ode, eqs, order=order,
                solve_for_func=False)

    s = True
    testnum = 0
    x = func.args[0]
    while s:
        if testnum == 0:
            # First pass, try substituting a solved solution directly into the
            # ODE. This has the highest chance of succeeding.
            ode_diff = ode.lhs - ode.rhs

            if sol.lhs == func:
                s = sub_func_doit(ode_diff, func, sol.rhs)
            else:
                testnum += 1
                continue
            ss = simplify(s)
            if ss:
                # with the new numer_denom in power.py, if we do a simple
                # expansion then testnum == 0 verifies all solutions.
                s = ss.expand(force=True)
            else:
                s = 0
            testnum += 1
        elif testnum == 1:
            # Second pass. If we cannot substitute f, try seeing if the nth
            # derivative is equal, this will only work for odes that are exact,
            # by definition.
            s = simplify(
                trigsimp(diff(sol.lhs, x, order) - diff(sol.rhs, x, order)) -
                trigsimp(ode.lhs) + trigsimp(ode.rhs))
            # s2 = simplify(
            #     diff(sol.lhs, x, order) - diff(sol.rhs, x, order) - \
            #     ode.lhs + ode.rhs)
            testnum += 1
        elif testnum == 2:
            # Third pass. Try solving for df/dx and substituting that into the
            # ODE. Thanks to Chris Smith for suggesting this method.  Many of
            # the comments below are his, too.
            # The method:
            # - Take each of 1..n derivatives of the solution.
            # - Solve each nth derivative for d^(n)f/dx^(n)
            #   (the differential of that order)
            # - Back substitute into the ODE in decreasing order
            #   (i.e., n, n-1, ...)
            # - Check the result for zero equivalence
            if sol.lhs == func and not sol.rhs.has(func):
                diffsols = {0: sol.rhs}
            elif sol.rhs == func and not sol.lhs.has(func):
                diffsols = {0: sol.lhs}
            else:
                diffsols = {}
            sol = sol.lhs - sol.rhs
            for i in range(1, order + 1):
                # Differentiation is a linear operator, so there should always
                # be 1 solution. Nonetheless, we test just to make sure.
                # We only need to solve once.  After that, we automatically
                # have the solution to the differential in the order we want.
                if i == 1:
                    ds = sol.diff(x)
                    try:
                        sdf = solve(ds, func.diff(x, i))
                        if not sdf:
                            raise NotImplementedError
                    except NotImplementedError:
                        testnum += 1
                        break
                    else:
                        diffsols[i] = sdf[0]
                else:
                    # This is what the solution says df/dx should be.
                    diffsols[i] = diffsols[i - 1].diff(x)

            # Make sure the above didn't fail.
            if testnum > 2:
                continue
            else:
                # Substitute it into ODE to check for self consistency.
                lhs, rhs = ode.lhs, ode.rhs
                for i in range(order, -1, -1):
                    if i == 0 and 0 not in diffsols:
                        # We can only substitute f(x) if the solution was
                        # solved for f(x).
                        break
                    lhs = sub_func_doit(lhs, func.diff(x, i), diffsols[i])
                    rhs = sub_func_doit(rhs, func.diff(x, i), diffsols[i])
                    ode_or_bool = Eq(lhs, rhs)
                    ode_or_bool = simplify(ode_or_bool)

                    if isinstance(ode_or_bool, (bool, BooleanAtom)):
                        if ode_or_bool:
                            lhs = rhs = S.Zero
                    else:
                        lhs = ode_or_bool.lhs
                        rhs = ode_or_bool.rhs
                # No sense in overworking simplify -- just prove that the
                # numerator goes to zero
                num = trigsimp((lhs - rhs).as_numer_denom()[0])
                # since solutions are obtained using force=True we test
                # using the same level of assumptions
                ## replace function with dummy so assumptions will work
                _func = Dummy('func')
                num = num.subs(func, _func)
                ## posify the expression
                num, reps = posify(num)
                s = simplify(num).xreplace(reps).xreplace({_func: func})
                testnum += 1
        else:
            break

    if not s:
        return (True, s)
    elif s is True:  # The code above never was able to change s
        raise NotImplementedError("Unable to test if " + str(sol) +
            " is a solution to " + str(ode) + ".")
    else:
        return (False, s)


def ode_sol_simplicity(sol, func, trysolving=True):
    r"""
    Returns an extended integer representing how simple a solution to an ODE
    is.

    The following things are considered, in order from most simple to least:

    - ``sol`` is solved for ``func``.
    - ``sol`` is not solved for ``func``, but can be if passed to solve (e.g.,
      a solution returned by ``dsolve(ode, func, simplify=False``).
    - If ``sol`` is not solved for ``func``, then base the result on the
      length of ``sol``, as computed by ``len(str(sol))``.
    - If ``sol`` has any unevaluated :py:class:`~sympy.integrals.Integral`\s,
      this will automatically be considered less simple than any of the above.

    This function returns an integer such that if solution A is simpler than
    solution B by above metric, then ``ode_sol_simplicity(sola, func) <
    ode_sol_simplicity(solb, func)``.

    Currently, the following are the numbers returned, but if the heuristic is
    ever improved, this may change.  Only the ordering is guaranteed.

    +----------------------------------------------+-------------------+
    | Simplicity                                   | Return            |
    +==============================================+===================+
    | ``sol`` solved for ``func``                  | ``-2``            |
    +----------------------------------------------+-------------------+
    | ``sol`` not solved for ``func`` but can be   | ``-1``            |
    +----------------------------------------------+-------------------+
    | ``sol`` is not solved nor solvable for       | ``len(str(sol))`` |
    | ``func``                                     |                   |
    +----------------------------------------------+-------------------+
    | ``sol`` contains an                          | ``oo``            |
    | :py:class:`~sympy.integrals.Integral`        |                   |
    +----------------------------------------------+-------------------+

    ``oo`` here means the SymPy infinity, which should compare greater than
    any integer.

    If you already know :py:meth:`~sympy.solvers.solvers.solve` cannot solve
    ``sol``, you can use ``trysolving=False`` to skip that step, which is the
    only potentially slow step.  For example,
    :py:meth:`~sympy.solvers.ode.dsolve` with the ``simplify=False`` flag
    should do this.

    If ``sol`` is a list of solutions, if the worst solution in the list
    returns ``oo`` it returns that, otherwise it returns ``len(str(sol))``,
    that is, the length of the string representation of the whole list.

    Examples
    ========

    This function is designed to be passed to ``min`` as the key argument,
    such as ``min(listofsolutions, key=lambda i: ode_sol_simplicity(i,
    f(x)))``.

    >>> from sympy import symbols, Function, Eq, tan, cos, sqrt, Integral
    >>> from sympy.solvers.ode import ode_sol_simplicity
    >>> x, C1, C2 = symbols('x, C1, C2')
    >>> f = Function('f')

    >>> ode_sol_simplicity(Eq(f(x), C1*x**2), f(x))
    -2
    >>> ode_sol_simplicity(Eq(x**2 + f(x), C1), f(x))
    -1
    >>> ode_sol_simplicity(Eq(f(x), C1*Integral(2*x, x)), f(x))
    oo
    >>> eq1 = Eq(f(x)/tan(f(x)/(2*x)), C1)
    >>> eq2 = Eq(f(x)/tan(f(x)/(2*x) + f(x)), C2)
    >>> [ode_sol_simplicity(eq, f(x)) for eq in [eq1, eq2]]
    [28, 35]
    >>> min([eq1, eq2], key=lambda i: ode_sol_simplicity(i, f(x)))
    Eq(f(x)/tan(f(x)/(2*x)), C1)

    """
    # TODO: if two solutions are solved for f(x), we still want to be
    # able to get the simpler of the two

    # See the docstring for the coercion rules.  We check easier (faster)
    # things here first, to save time.

    if iterable(sol):
        # See if there are Integrals
        for i in sol:
            if ode_sol_simplicity(i, func, trysolving=trysolving) == oo:
                return oo

        return len(str(sol))

    if sol.has(Integral):
        return oo

    # Next, try to solve for func.  This code will change slightly when CRootOf
    # is implemented in solve().  Probably a CRootOf solution should fall
    # somewhere between a normal solution and an unsolvable expression.

    # First, see if they are already solved
    if sol.lhs == func and not sol.rhs.has(func) or \
            sol.rhs == func and not sol.lhs.has(func):
        return -2
    # We are not so lucky, try solving manually
    if trysolving:
        try:
            sols = solve(sol, func)
            if not sols:
                raise NotImplementedError
        except NotImplementedError:
            pass
        else:
            return -1

    # Finally, a naive computation based on the length of the string version
    # of the expression.  This may favor combined fractions because they
    # will not have duplicate denominators, and may slightly favor expressions
    # with fewer additions and subtractions, as those are separated by spaces
    # by the printer.

    # Additional ideas for simplicity heuristics are welcome, like maybe
    # checking if a equation has a larger domain, or if constantsimp has
    # introduced arbitrary constants numbered higher than the order of a
    # given ODE that sol is a solution of.
    return len(str(sol))


def _get_constant_subexpressions(expr, Cs):
    Cs = set(Cs)
    Ces = []
    def _recursive_walk(expr):
        expr_syms = expr.free_symbols
        if len(expr_syms) > 0 and expr_syms.issubset(Cs):
            Ces.append(expr)
        else:
            if expr.func == exp:
                expr = expr.expand(mul=True)
            if expr.func in (Add, Mul):
                d = sift(expr.args, lambda i : i.free_symbols.issubset(Cs))
                if len(d[True]) > 1:
                    x = expr.func(*d[True])
                    if not x.is_number:
                        Ces.append(x)
            elif isinstance(expr, Integral):
                if expr.free_symbols.issubset(Cs) and \
                            all(len(x) == 3 for x in expr.limits):
                    Ces.append(expr)
            for i in expr.args:
                _recursive_walk(i)
        return
    _recursive_walk(expr)
    return Ces

def __remove_linear_redundancies(expr, Cs):
    cnts = {i: expr.count(i) for i in Cs}
    Cs = [i for i in Cs if cnts[i] > 0]

    def _linear(expr):
        if expr.func is Add:
            xs = [i for i in Cs if expr.count(i)==cnts[i] \
                and 0 == expr.diff(i, 2)]
            d = {}
            for x in xs:
                y = expr.diff(x)
                if y not in d:
                    d[y]=[]
                d[y].append(x)
            for y in d:
                if len(d[y]) > 1:
                    d[y].sort(key=str)
                    for x in d[y][1:]:
                        expr = expr.subs(x, 0)
        return expr

    def _recursive_walk(expr):
        if len(expr.args) != 0:
            expr = expr.func(*[_recursive_walk(i) for i in expr.args])
        expr = _linear(expr)
        return expr

    if expr.func is Equality:
        lhs, rhs = [_recursive_walk(i) for i in expr.args]
        f = lambda i: isinstance(i, Number) or i in Cs
        if lhs.func is Symbol and lhs in Cs:
            rhs, lhs = lhs, rhs
        if lhs.func in (Add, Symbol) and rhs.func in (Add, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            drhs = sift([rhs] if isinstance(rhs, AtomicExpr) else rhs.args, f)
            for i in [True, False]:
                for hs in [dlhs, drhs]:
                    if i not in hs:
                        hs[i] = [0]
            # this calculation can be simplified
            lhs = Add(*dlhs[False]) - Add(*drhs[False])
            rhs = Add(*drhs[True]) - Add(*dlhs[True])
        elif lhs.func in (Mul, Symbol) and rhs.func in (Mul, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            if True in dlhs:
                if False not in dlhs:
                    dlhs[False] = [1]
                lhs = Mul(*dlhs[False])
                rhs = rhs/Mul(*dlhs[True])
        return Eq(lhs, rhs)
    else:
        return _recursive_walk(expr)
```
