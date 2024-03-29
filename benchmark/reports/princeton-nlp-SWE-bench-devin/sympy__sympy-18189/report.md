# sympy__sympy-18189

| **sympy/sympy** | `1923822ddf8265199dbd9ef9ce09641d3fd042b9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 414 |
| **Any found context length** | 414 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -182,7 +182,7 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
             if syms != var:
                 dict_sym_index = dict(zip(syms, range(len(syms))))
                 return {tuple([t[dict_sym_index[i]] for i in var])
-                            for t in diophantine(eq, param)}
+                            for t in diophantine(eq, param, permute=permute)}
         n, d = eq.as_numer_denom()
         if n.is_number:
             return set()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/solvers/diophantine.py | 185 | 185 | 1 | 1 | 414


## Problem Statement

```
diophantine: incomplete results depending on syms order with permute=True
\`\`\`
In [10]: diophantine(n**4 + m**4 - 2**4 - 3**4, syms=(m,n), permute=True)
Out[10]: {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

In [11]: diophantine(n**4 + m**4 - 2**4 - 3**4, syms=(n,m), permute=True)
Out[11]: {(3, 2)}
\`\`\`

diophantine: incomplete results depending on syms order with permute=True
\`\`\`
In [10]: diophantine(n**4 + m**4 - 2**4 - 3**4, syms=(m,n), permute=True)
Out[10]: {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

In [11]: diophantine(n**4 + m**4 - 2**4 - 3**4, syms=(n,m), permute=True)
Out[11]: {(3, 2)}
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/solvers/diophantine.py** | 168 | 211| 414 | 414 | 31536 | 
| 2 | **1 sympy/solvers/diophantine.py** | 212 | 297| 897 | 1311 | 31536 | 
| 3 | **1 sympy/solvers/diophantine.py** | 299 | 351| 462 | 1773 | 31536 | 
| 4 | **1 sympy/solvers/diophantine.py** | 1172 | 1314| 922 | 2695 | 31536 | 
| 5 | 2 sympy/utilities/iterables.py | 1807 | 1884| 697 | 3392 | 53410 | 
| 6 | 2 sympy/utilities/iterables.py | 1885 | 1940| 483 | 3875 | 53410 | 
| 7 | **2 sympy/solvers/diophantine.py** | 3081 | 3128| 370 | 4245 | 53410 | 
| 8 | 3 sympy/functions/combinatorial/numbers.py | 1569 | 1619| 559 | 4804 | 71386 | 
| 9 | **3 sympy/solvers/diophantine.py** | 3002 | 3078| 787 | 5591 | 71386 | 
| 10 | 3 sympy/utilities/iterables.py | 1726 | 1804| 531 | 6122 | 71386 | 
| 11 | **3 sympy/solvers/diophantine.py** | 1 | 98| 743 | 6865 | 71386 | 
| 12 | 4 sympy/ntheory/multinomial.py | 131 | 192| 616 | 7481 | 73119 | 
| 13 | 5 sympy/functions/special/polynomials.py | 919 | 934| 265 | 7746 | 86108 | 
| 14 | 6 sympy/combinatorics/permutations.py | 470 | 829| 3435 | 11181 | 109712 | 
| 15 | 6 sympy/functions/combinatorial/numbers.py | 851 | 868| 219 | 11400 | 109712 | 
| 16 | 7 sympy/combinatorics/generators.py | 240 | 310| 432 | 11832 | 112195 | 
| 17 | 7 sympy/functions/special/polynomials.py | 856 | 917| 603 | 12435 | 112195 | 
| 18 | 8 sympy/codegen/array_utils.py | 576 | 631| 684 | 13119 | 125620 | 
| 19 | 8 sympy/utilities/iterables.py | 1331 | 1377| 416 | 13535 | 125620 | 
| 20 | 8 sympy/utilities/iterables.py | 1418 | 1494| 802 | 14337 | 125620 | 
| 21 | 9 sympy/solvers/solveset.py | 1145 | 1209| 659 | 14996 | 156294 | 
| 22 | 10 sympy/logic/boolalg.py | 1962 | 2032| 473 | 15469 | 177581 | 
| 23 | 10 sympy/ntheory/multinomial.py | 57 | 128| 347 | 15816 | 177581 | 
| 24 | 10 sympy/utilities/iterables.py | 2086 | 2150| 756 | 16572 | 177581 | 
| 25 | 10 sympy/functions/combinatorial/numbers.py | 1514 | 1566| 437 | 17009 | 177581 | 
| 26 | 10 sympy/codegen/array_utils.py | 89 | 124| 328 | 17337 | 177581 | 
| 27 | **10 sympy/solvers/diophantine.py** | 947 | 1043| 1140 | 18477 | 177581 | 
| 28 | **10 sympy/solvers/diophantine.py** | 3212 | 3239| 338 | 18815 | 177581 | 
| 29 | 11 sympy/combinatorics/named_groups.py | 233 | 308| 815 | 19630 | 180290 | 
| 30 | 12 sympy/combinatorics/util.py | 1 | 70| 532 | 20162 | 184965 | 
| 31 | 12 sympy/combinatorics/named_groups.py | 169 | 230| 664 | 20826 | 184965 | 
| 32 | 13 sympy/ntheory/residue_ntheory.py | 874 | 959| 702 | 21528 | 194959 | 
| 33 | 13 sympy/functions/combinatorial/numbers.py | 1349 | 1371| 212 | 21740 | 194959 | 
| 34 | 13 sympy/functions/combinatorial/numbers.py | 1373 | 1417| 218 | 21958 | 194959 | 
| 35 | **13 sympy/solvers/diophantine.py** | 101 | 166| 737 | 22695 | 194959 | 
| 36 | 13 sympy/combinatorics/permutations.py | 2743 | 2790| 396 | 23091 | 194959 | 
| 37 | **13 sympy/solvers/diophantine.py** | 2710 | 2738| 461 | 23552 | 194959 | 
| 38 | **13 sympy/solvers/diophantine.py** | 1412 | 1471| 548 | 24100 | 194959 | 
| 39 | 14 sympy/matrices/expressions/permutation.py | 102 | 164| 423 | 24523 | 196941 | 
| 40 | 14 sympy/utilities/iterables.py | 1275 | 1328| 442 | 24965 | 196941 | 


### Hint

```
\`\`\`diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
index 6092e35..b43f5c1 100644
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -182,7 +182,7 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
             if syms != var:
                 dict_sym_index = dict(zip(syms, range(len(syms))))
                 return {tuple([t[dict_sym_index[i]] for i in var])
-                            for t in diophantine(eq, param)}
+                            for t in diophantine(eq, param, permute=permute)}
         n, d = eq.as_numer_denom()
         if n.is_number:
             return set()
\`\`\`
Based on a cursory glance at the code it seems that `permute=True` is lost when `diophantine` calls itself:
https://github.com/sympy/sympy/blob/d98abf000b189d4807c6f67307ebda47abb997f8/sympy/solvers/diophantine.py#L182-L185.
That should be easy to solve; I'll include a fix in my next PR (which is related).
Ah, ninja'd by @smichr :-)
\`\`\`diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
index 6092e35..b43f5c1 100644
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -182,7 +182,7 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
             if syms != var:
                 dict_sym_index = dict(zip(syms, range(len(syms))))
                 return {tuple([t[dict_sym_index[i]] for i in var])
-                            for t in diophantine(eq, param)}
+                            for t in diophantine(eq, param, permute=permute)}
         n, d = eq.as_numer_denom()
         if n.is_number:
             return set()
\`\`\`
Based on a cursory glance at the code it seems that `permute=True` is lost when `diophantine` calls itself:
https://github.com/sympy/sympy/blob/d98abf000b189d4807c6f67307ebda47abb997f8/sympy/solvers/diophantine.py#L182-L185.
That should be easy to solve; I'll include a fix in my next PR (which is related).
Ah, ninja'd by @smichr :-)
```

## Patch

```diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -182,7 +182,7 @@ def diophantine(eq, param=symbols("t", integer=True), syms=None,
             if syms != var:
                 dict_sym_index = dict(zip(syms, range(len(syms))))
                 return {tuple([t[dict_sym_index[i]] for i in var])
-                            for t in diophantine(eq, param)}
+                            for t in diophantine(eq, param, permute=permute)}
         n, d = eq.as_numer_denom()
         if n.is_number:
             return set()

```

## Test Patch

```diff
diff --git a/sympy/solvers/tests/test_diophantine.py b/sympy/solvers/tests/test_diophantine.py
--- a/sympy/solvers/tests/test_diophantine.py
+++ b/sympy/solvers/tests/test_diophantine.py
@@ -547,6 +547,13 @@ def test_diophantine():
     assert diophantine(x**2 + y**2 +3*x- 5, permute=True) == \
         set([(-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)])
 
+
+    #test issue 18186
+    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(x, y), permute=True) == \
+        set([(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)])
+    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(y, x), permute=True) == \
+        set([(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)])
+
     # issue 18122
     assert check_solutions(x**2-y)
     assert check_solutions(y**2-x)
@@ -554,6 +561,7 @@ def test_diophantine():
     assert diophantine((y**2-x), t) == set([(t**2, -t)])
 
 
+
 def test_general_pythagorean():
     from sympy.abc import a, b, c, d, e
 

```


## Code snippets

### 1 - sympy/solvers/diophantine.py:

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
                            for t in diophantine(eq, param)}
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
### 2 - sympy/solvers/diophantine.py:

Start line: 212, End line: 297

```python
def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    # ... other code
    try:
        # if we know that factoring should not be attempted, skip
        # the factoring step
        v, c, t = classify_diop(eq)

        # check for permute sign
        if permute:
            len_var = len(v)
            permute_signs_for = [
                'general_sum_of_squares',
                'general_sum_of_even_powers']
            permute_signs_check = [
                'homogeneous_ternary_quadratic',
                'homogeneous_ternary_quadratic_normal',
                'binary_quadratic']
            if t in permute_signs_for:
                do_permute_signs_var = True
            elif t in permute_signs_check:
                # if all the variables in eq have even powers
                # then do_permute_sign = True
                if len_var == 3:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y), (x, z), (y, z)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = map(lambda a: a[0]*a[1], var_mul)
                    # if coeff(y*z), coeff(y*x), coeff(x*z) is not 0 then
                    # `xy_coeff` => True and do_permute_sign => False.
                    # Means no permuted solution.
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any([xy_coeff, x_coeff]):
                        # means only x**2, y**2, z**2, const is present
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
                elif len_var == 2:
                    var_mul = list(subsets(v, 2))
                    # here var_mul is like [(x, y)]
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = map(lambda x: x[0]*x[1], var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    # here var_mul is like [(x,), (y, )]
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any([xy_coeff, x_coeff]):
                        # means only x**2, y**2 and const is present
                        # so we can get more soln by permuting this soln.
                        do_permute_signs = True
                    elif not x_coeff:
                        # when coeff(x), coeff(y) is not present then signs of
                        #  x, y can be permuted such that their sign are same
                        # as sign of x*y.
                        # e.g 1. (x_val,y_val)=> (x_val,y_val), (-x_val,-y_val)
                        # 2. (-x_vall, y_val)=> (-x_val,y_val), (x_val,-y_val)
                        permute_few_signs = True
        if t == 'general_sum_of_squares':
            # trying to factor such expressions will sometimes hang
            terms = [(eq, 1)]
        else:
            raise TypeError
    except (TypeError, NotImplementedError):
        terms = factor_list(eq)[1]
    # ... other code
```
### 3 - sympy/solvers/diophantine.py:

Start line: 299, End line: 351

```python
def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    # ... other code

    sols = set([])

    for term in terms:

        base, _ = term
        var_t, _, eq_type = classify_diop(base, _dict=False)
        _, base = signsimp(base, evaluate=False).as_coeff_Mul()
        solution = diop_solve(base, param)

        if eq_type in [
                "linear",
                "homogeneous_ternary_quadratic",
                "homogeneous_ternary_quadratic_normal",
                "general_pythagorean"]:
            sols.add(merge_solution(var, var_t, solution))

        elif eq_type in [
                "binary_quadratic",
                "general_sum_of_squares",
                "general_sum_of_even_powers",
                "univariate"]:
            for sol in solution:
                sols.add(merge_solution(var, var_t, sol))

        else:
            raise NotImplementedError('unhandled type: %s' % eq_type)

    # remove null merge results
    if () in sols:
        sols.remove(())
    null = tuple([0]*len(var))
    # if there is no solution, return trivial solution
    if not sols and eq.subs(zip(var, null)).is_zero:
        sols.add(null)
    final_soln = set([])
    for sol in sols:
        if all(_is_int(s) for s in sol):
            if do_permute_signs:
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            elif permute_few_signs:
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0]*x[1] == sol[1]*sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            elif do_permute_signs_var:
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                final_soln.add(sol)
        else:
                final_soln.add(sol)
    return final_soln
```
### 4 - sympy/solvers/diophantine.py:

Start line: 1172, End line: 1314

```python
def diop_DN(D, N, t=symbols("t", integer=True)):
    if D < 0:
        if N == 0:
            return [(0, 0)]
        elif N < 0:
            return []
        elif N > 0:
            sol = []
            for d in divisors(square_factor(N)):
                sols = cornacchia(1, -D, N // d**2)
                if sols:
                    for x, y in sols:
                        sol.append((d*x, d*y))
                        if D == -1:
                            sol.append((d*y, d*x))
            return sol

    elif D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]
        sN, _exact = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]
        else:
            return []

    else:  # D > 0
        sD, _exact = integer_nthroot(D, 2)
        if _exact:
            if N == 0:
                return [(sD*t, t)]
            else:
                sol = []

                for y in range(floor(sign(N)*(N - 1)/(2*sD)) + 1):
                    try:
                        sq, _exact = integer_nthroot(D*y**2 + N, 2)
                    except ValueError:
                        _exact = False
                    if _exact:
                        sol.append((sq, y))

                return sol

        elif 1 < N**2 < D:
            # It is much faster to call `_special_diop_DN`.
            return _special_diop_DN(D, N)

        else:
            if N == 0:
                return [(0, 0)]

            elif abs(N) == 1:

                pqa = PQa(0, 1, D)
                j = 0
                G = []
                B = []

                for i in pqa:

                    a = i[2]
                    G.append(i[5])
                    B.append(i[4])

                    if j != 0 and a == 2*sD:
                        break
                    j = j + 1

                if _odd(j):

                    if N == -1:
                        x = G[j - 1]
                        y = B[j - 1]
                    else:
                        count = j
                        while count < 2*j - 1:
                            i = next(pqa)
                            G.append(i[5])
                            B.append(i[4])
                            count += 1

                        x = G[count]
                        y = B[count]
                else:
                    if N == 1:
                        x = G[j - 1]
                        y = B[j - 1]
                    else:
                        return []

                return [(x, y)]

            else:

                fs = []
                sol = []
                div = divisors(N)

                for d in div:
                    if divisible(N, d**2):
                        fs.append(d)

                for f in fs:
                    m = N // f**2

                    zs = sqrt_mod(D, abs(m), all_roots=True)
                    zs = [i for i in zs if i <= abs(m) // 2 ]

                    if abs(m) != 2:
                        zs = zs + [-i for i in zs if i]  # omit dupl 0

                    for z in zs:

                        pqa = PQa(z, abs(m), D)
                        j = 0
                        G = []
                        B = []

                        for i in pqa:

                            G.append(i[5])
                            B.append(i[4])

                            if j != 0 and abs(i[1]) == 1:
                                r = G[j-1]
                                s = B[j-1]

                                if r**2 - D*s**2 == m:
                                    sol.append((f*r, f*s))

                                elif diop_DN(D, -1) != []:
                                    a = diop_DN(D, -1)
                                    sol.append((f*(r*a[0][0] + a[0][1]*s*D), f*(r*a[0][1] + s*a[0][0])))

                                break

                            j = j + 1
                            if j == length(z, abs(m), D):
                                break

                return sol
```
### 5 - sympy/utilities/iterables.py:

Start line: 1807, End line: 1884

```python
def ordered_partitions(n, m=None, sort=True):
    """Generates ordered partitions of integer ``n``.

    Parameters
    ==========

    m : integer (default None)
        The default value gives partitions of all sizes else only
        those with size m. In addition, if ``m`` is not None then
        partitions are generated *in place* (see examples).
    sort : bool (default True)
        Controls whether partitions are
        returned in sorted order when ``m`` is not None; when False,
        the partitions are returned as fast as possible with elements
        sorted, but when m|n the partitions will not be in
        ascending lexicographical order.

    Examples
    ========

    >>> from sympy.utilities.iterables import ordered_partitions

    All partitions of 5 in ascending lexicographical:

    >>> for p in ordered_partitions(5):
    ...     print(p)
    [1, 1, 1, 1, 1]
    [1, 1, 1, 2]
    [1, 1, 3]
    [1, 2, 2]
    [1, 4]
    [2, 3]
    [5]

    Only partitions of 5 with two parts:

    >>> for p in ordered_partitions(5, 2):
    ...     print(p)
    [1, 4]
    [2, 3]

    When ``m`` is given, a given list objects will be used more than
    once for speed reasons so you will not see the correct partitions
    unless you make a copy of each as it is generated:

    >>> [p for p in ordered_partitions(7, 3)]
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]]
    >>> [list(p) for p in ordered_partitions(7, 3)]
    [[1, 1, 5], [1, 2, 4], [1, 3, 3], [2, 2, 3]]

    When ``n`` is a multiple of ``m``, the elements are still sorted
    but the partitions themselves will be *unordered* if sort is False;
    the default is to return them in ascending lexicographical order.

    >>> for p in ordered_partitions(6, 2):
    ...     print(p)
    [1, 5]
    [2, 4]
    [3, 3]

    But if speed is more important than ordering, sort can be set to
    False:

    >>> for p in ordered_partitions(6, 2, sort=False):
    ...     print(p)
    [1, 5]
    [3, 3]
    [2, 4]

    References
    ==========

    .. [1] Generating Integer Partitions, [online],
        Available: https://jeromekelleher.net/generating-integer-partitions.html
    .. [2] Jerome Kelleher and Barry O'Sullivan, "Generating All
        Partitions: A Comparison Of Two Encodings", [online],
        Available: https://arxiv.org/pdf/0909.2331v2.pdf
    """
    # ... other code
```
### 6 - sympy/utilities/iterables.py:

Start line: 1885, End line: 1940

```python
def ordered_partitions(n, m=None, sort=True):
    if n < 1 or m is not None and m < 1:
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        yield []
        return

    if m is None:
        # The list `a`'s leading elements contain the partition in which
        # y is the biggest element and x is either the same as y or the
        # 2nd largest element; v and w are adjacent element indices
        # to which x and y are being assigned, respectively.
        a = [1]*n
        y = -1
        v = n
        while v > 0:
            v -= 1
            x = a[v] + 1
            while y >= 2 * x:
                a[v] = x
                y -= x
                v += 1
            w = v + 1
            while x <= y:
                a[v] = x
                a[w] = y
                yield a[:w + 1]
                x += 1
                y -= 1
            a[v] = x + y
            y = a[v] - 1
            yield a[:w]
    elif m == 1:
        yield [n]
    elif n == m:
        yield [1]*n
    else:
        # recursively generate partitions of size m
        for b in range(1, n//m + 1):
            a = [b]*m
            x = n - b*m
            if not x:
                if sort:
                    yield a
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]
                    yield a
                    a[-mi:] = [b]*mi
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]
                        yield a
                        a[-mi:] = [b]*mi
```
### 7 - sympy/solvers/diophantine.py:

Start line: 3081, End line: 3128

```python
def sum_of_four_squares(n):
    r"""
    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.

    Here `a, b, c, d \geq 0`.

    Usage
    =====

    ``sum_of_four_squares(n)``: Here ``n`` is a non-negative integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine import sum_of_four_squares
    >>> sum_of_four_squares(3456)
    (8, 8, 32, 48)
    >>> sum_of_four_squares(1294585930293)
    (0, 1234, 2161, 1137796)

    References
    ==========

    .. [1] Representing a number as a sum of four squares, [online],
        Available: http://schorn.ch/lagrange.html

    See Also
    ========
    sum_of_squares()
    """
    if n == 0:
        return (0, 0, 0, 0)

    v = multiplicity(4, n)
    n //= 4**v

    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 == 6 or n % 8 == 2:
        d = 1
        n = n - 1
    else:
        d = 0

    x, y, z = sum_of_three_squares(n)

    return _sorted_tuple(2**v*d, 2**v*x, 2**v*y, 2**v*z)
```
### 8 - sympy/functions/combinatorial/numbers.py:

Start line: 1569, End line: 1619

```python
@cacheit
def _AOP_product(n):
    """for n = (m1, m2, .., mk) return the coefficients of the polynomial,
    prod(sum(x**i for i in range(nj + 1)) for nj in n); i.e. the coefficients
    of the product of AOPs (all-one polynomials) or order given in n.  The
    resulting coefficient corresponding to x**r is the number of r-length
    combinations of sum(n) elements with multiplicities given in n.
    The coefficients are given as a default dictionary (so if a query is made
    for a key that is not present, 0 will be returned).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import _AOP_product
    >>> from sympy.abc import x
    >>> n = (2, 2, 3)  # e.g. aabbccc
    >>> prod = ((x**2 + x + 1)*(x**2 + x + 1)*(x**3 + x**2 + x + 1)).expand()
    >>> c = _AOP_product(n); dict(c)
    {0: 1, 1: 3, 2: 6, 3: 8, 4: 8, 5: 6, 6: 3, 7: 1}
    >>> [c[i] for i in range(8)] == [prod.coeff(x, i) for i in range(8)]
    True

    The generating poly used here is the same as that listed in
    http://tinyurl.com/cep849r, but in a refactored form.

    """
    from collections import defaultdict

    n = list(n)
    ord = sum(n)
    need = (ord + 2)//2
    rv = [1]*(n.pop() + 1)
    rv.extend([0]*(need - len(rv)))
    rv = rv[:need]
    while n:
        ni = n.pop()
        N = ni + 1
        was = rv[:]
        for i in range(1, min(N, len(rv))):
            rv[i] += rv[i - 1]
        for i in range(N, need):
            rv[i] += rv[i - 1] - was[i - N]
    rev = list(reversed(rv))
    if ord % 2:
        rv = rv + rev
    else:
        rv[-1:] = rev
    d = defaultdict(int)
    for i in range(len(rv)):
        d[i] = rv[i]
    return d
```
### 9 - sympy/solvers/diophantine.py:

Start line: 3002, End line: 3078

```python
def sum_of_three_squares(n):
    r"""
    Returns a 3-tuple `(a, b, c)` such that `a^2 + b^2 + c^2 = n` and
    `a, b, c \geq 0`.

    Returns None if `n = 4^a(8m + 7)` for some `a, m \in Z`. See
    [1]_ for more details.

    Usage
    =====

    ``sum_of_three_squares(n)``: Here ``n`` is a non-negative integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: http://schorn.ch/lagrange.html

    See Also
    ========
    sum_of_squares()
    """
    special = {1:(1, 0, 0), 2:(1, 1, 0), 3:(1, 1, 1), 10: (1, 3, 0), 34: (3, 3, 4), 58:(3, 7, 0),
        85:(6, 7, 0), 130:(3, 11, 0), 214:(3, 6, 13), 226:(8, 9, 9), 370:(8, 9, 15),
        526:(6, 7, 21), 706:(15, 15, 16), 730:(1, 27, 0), 1414:(6, 17, 33), 1906:(13, 21, 36),
        2986: (21, 32, 39), 9634: (56, 57, 57)}

    v = 0

    if n == 0:
        return (0, 0, 0)

    v = multiplicity(4, n)
    n //= 4**v

    if n % 8 == 7:
        return

    if n in special.keys():
        x, y, z = special[n]
        return _sorted_tuple(2**v*x, 2**v*y, 2**v*z)

    s, _exact = integer_nthroot(n, 2)

    if _exact:
        return (2**v*s, 0, 0)

    x = None

    if n % 8 == 3:
        s = s if _odd(s) else s - 1

        for x in range(s, -1, -2):
            N = (n - x**2) // 2
            if isprime(N):
                y, z = prime_as_sum_of_two_squares(N)
                return _sorted_tuple(2**v*x, 2**v*(y + z), 2**v*abs(y - z))
        return

    if n % 8 == 2 or n % 8 == 6:
        s = s if _odd(s) else s - 1
    else:
        s = s - 1 if _odd(s) else s

    for x in range(s, -1, -2):
        N = n - x**2
        if isprime(N):
            y, z = prime_as_sum_of_two_squares(N)
            return _sorted_tuple(2**v*x, 2**v*y, 2**v*z)
```
### 10 - sympy/utilities/iterables.py:

Start line: 1726, End line: 1804

```python
def partitions(n, m=None, k=None, size=False):
    if (n <= 0 or
        m is not None and m < 1 or
        k is not None and k < 1 or
        m and k and m*k < n):
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        if size:
            yield 0, {}
        else:
            yield {}
        return

    if m is None:
        m = n
    else:
        m = min(m, n)

    if n == 0:
        if size:
            yield 1, {0: 1}
        else:
            yield {0: 1}
        return

    k = min(k or n, n)

    n, m, k = as_int(n), as_int(m), as_int(k)
    q, r = divmod(n, k)
    ms = {k: q}
    keys = [k]  # ms.keys(), from largest to smallest
    if r:
        ms[r] = 1
        keys.append(r)
    room = m - q - bool(r)
    if size:
        yield sum(ms.values()), ms
    else:
        yield ms

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
            room += reuse
        else:
            reuse = 0

        while 1:
            # Let i be the smallest key larger than 1.  Reuse one
            # instance of i.
            i = keys[-1]
            newcount = ms[i] = ms[i] - 1
            reuse += i
            if newcount == 0:
                del keys[-1], ms[i]
            room += 1

            # Break the remainder into pieces of size i-1.
            i -= 1
            q, r = divmod(reuse, i)
            need = q + bool(r)
            if need > room:
                if not keys:
                    return
                continue

            ms[i] = q
            keys.append(i)
            if r:
                ms[r] = 1
                keys.append(r)
            break
        room -= need
        if size:
            yield sum(ms.values()), ms
        else:
            yield ms
```
### 11 - sympy/solvers/diophantine.py:

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
### 27 - sympy/solvers/diophantine.py:

Start line: 947, End line: 1043

```python
def _diop_quadratic(var, coeff, t):
    # ... other code
    if A == 0 and C == 0 and B != 0:

        if D*E - B*F == 0:
            q, r = divmod(E, B)
            if not r:
                sol.add((-q, t))
            q, r = divmod(D, B)
            if not r:
                sol.add((t, -q))
        else:
            div = divisors(D*E - B*F)
            div = div + [-term for term in div]
            for d in div:
                x0, r = divmod(d - E, B)
                if not r:
                    q, r = divmod(D*E - B*F, d)
                    if not r:
                        y0, r = divmod(q - D, B)
                        if not r:
                            sol.add((x0, y0))

    # (2) Parabolic case: B**2 - 4*A*C = 0
    # There are two subcases to be considered in this case.
    # sqrt(c)D - sqrt(a)E = 0 and sqrt(c)D - sqrt(a)E != 0
    # More Details, http://www.alpertron.com.ar/METHODS.HTM#Parabol

    elif discr == 0:

        if A == 0:
            s = _diop_quadratic([y, x], coeff, t)
            for soln in s:
                sol.add((soln[1], soln[0]))

        else:
            g = sign(A)*igcd(A, C)
            a = A // g
            c = C // g
            e = sign(B/A)

            sqa = isqrt(a)
            sqc = isqrt(c)
            _c = e*sqc*D - sqa*E
            if not _c:
                z = symbols("z", real=True)
                eq = sqa*g*z**2 + D*z + sqa*F
                roots = solveset_real(eq, z).intersect(S.Integers)
                for root in roots:
                    ans = diop_solve(sqa*x + e*sqc*y - root)
                    sol.add((ans[0], ans[1]))

            elif _is_int(c):
                solve_x = lambda u: -e*sqc*g*_c*t**2 - (E + 2*e*sqc*g*u)*t\
                    - (e*sqc*g*u**2 + E*u + e*sqc*F) // _c

                solve_y = lambda u: sqa*g*_c*t**2 + (D + 2*sqa*g*u)*t \
                    + (sqa*g*u**2 + D*u + sqa*F) // _c

                for z0 in range(0, abs(_c)):
                    # Check if the coefficients of y and x obtained are integers or not
                    if (divisible(sqa*g*z0**2 + D*z0 + sqa*F, _c) and
                            divisible(e*sqc*g*z0**2 + E*z0 + e*sqc*F, _c)):
                        sol.add((solve_x(z0), solve_y(z0)))

    # (3) Method used when B**2 - 4*A*C is a square, is described in p. 6 of the below paper
    # by John P. Robertson.
    # http://www.jpr2718.org/ax2p.pdf

    elif is_square(discr):
        if A != 0:
            r = sqrt(discr)
            u, v = symbols("u, v", integer=True)
            eq = _mexpand(
                4*A*r*u*v + 4*A*D*(B*v + r*u + r*v - B*u) +
                2*A*4*A*E*(u - v) + 4*A*r*4*A*F)

            solution = diop_solve(eq, t)

            for s0, t0 in solution:

                num = B*t0 + r*s0 + r*t0 - B*s0
                x_0 = S(num)/(4*A*r)
                y_0 = S(s0 - t0)/(2*r)
                if isinstance(s0, Symbol) or isinstance(t0, Symbol):
                    if check_param(x_0, y_0, 4*A*r, t) != (None, None):
                        ans = check_param(x_0, y_0, 4*A*r, t)
                        sol.add((ans[0], ans[1]))
                elif x_0.is_Integer and y_0.is_Integer:
                    if is_solution_quad(var, coeff, x_0, y_0):
                        sol.add((x_0, y_0))

        else:
            s = _diop_quadratic(var[::-1], coeff, t)  # Interchange x and y
            while s:                                  #         |
                sol.add(s.pop()[::-1])  # and solution <--------+


    # (4) B**2 - 4*A*C > 0 and B**2 - 4*A*C not a square or B**2 - 4*A*C < 0
    # ... other code
    # ... other code
```
### 28 - sympy/solvers/diophantine.py:

Start line: 3212, End line: 3239

```python
def power_representation(n, p, k, zeros=False):
    # ... other code

    if p == 2:
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros and n > 33 and k >= 5 and k <= n and n - k in (
                13, 10, 7, 5, 4, 2, 1):
            '''Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].
                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'''
            return
        if feasible is not True:  # it's prime and k == 2
            yield prime_as_sum_of_two_squares(n)
            return

    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return  # Fermat: a**n + b**n = c**n has no solution for n > 2

    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))

    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,) * (k - i)))
```
### 35 - sympy/solvers/diophantine.py:

Start line: 101, End line: 166

```python
def diophantine(eq, param=symbols("t", integer=True), syms=None,
                permute=False):
    """
    Simplify the solution procedure of diophantine equation ``eq`` by
    converting it into a product of terms which should equal zero.

    For example, when solving, `x^2 - y^2 = 0` this is treated as
    `(x + y)(x - y) = 0` and `x + y = 0` and `x - y = 0` are solved
    independently and combined. Each term is solved by calling
    ``diop_solve()``. (Although it is possible to call ``diop_solve()``
    directly, one must be careful to pass an equation in the correct
    form and to interpret the output correctly; ``diophantine()`` is
    the public-facing function to use in general.)

    Output of ``diophantine()`` is a set of tuples. The elements of the
    tuple are the solutions for each variable in the equation and
    are arranged according to the alphabetic ordering of the variables.
    e.g. For an equation with two variables, `a` and `b`, the first
    element of the tuple is the solution for `a` and the second for `b`.

    Usage
    =====

    ``diophantine(eq, t, syms)``: Solve the diophantine
    equation ``eq``.
    ``t`` is the optional parameter to be used by ``diop_solve()``.
    ``syms`` is an optional list of symbols which determines the
    order of the elements in the returned tuple.

    By default, only the base solution is returned. If ``permute`` is set to
    True then permutations of the base solution and/or permutations of the
    signs of the values will be returned when applicable.

    >>> from sympy.solvers.diophantine import diophantine
    >>> from sympy.abc import a, b
    >>> eq = a**4 + b**4 - (2**4 + 3**4)
    >>> diophantine(eq)
    {(2, 3)}
    >>> diophantine(eq, permute=True)
    {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> diophantine(x**2 - y**2)
    {(t_0, -t_0), (t_0, t_0)}

    >>> diophantine(x*(2*x + 3*y - z))
    {(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}
    >>> diophantine(x**2 + 3*x*y + 4*x)
    {(0, n1), (3*t_0 - 4, -t_0)}

    See Also
    ========

    diop_solve()
    sympy.utilities.iterables.permute_signs
    sympy.utilities.iterables.signed_permutations
    """
    # ... other code
```
### 37 - sympy/solvers/diophantine.py:

Start line: 2710, End line: 2738

```python
def diop_general_pythagorean(eq, param=symbols("m", integer=True)):
    """
    Solves the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Returns a tuple which contains a parametrized solution to the equation,
    sorted in the same order as the input variables.

    Usage
    =====

    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general
    pythagorean equation which is assumed to be zero and ``param`` is the base
    parameter used to construct other parameters by subscripting.

    Examples
    ========

    >>> from sympy.solvers.diophantine import diop_general_pythagorean
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)
    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)
    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)
    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)
    """
    var, coeff, diop_type  = classify_diop(eq, _dict=False)

    if diop_type == "general_pythagorean":
        return _diop_general_pythagorean(var, coeff, param)
```
### 38 - sympy/solvers/diophantine.py:

Start line: 1412, End line: 1471

```python
def cornacchia(a, b, m):
    r"""
    Solves `ax^2 + by^2 = m` where `\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.

    Uses the algorithm due to Cornacchia. The method only finds primitive
    solutions, i.e. ones with `\gcd(x, y) = 1`. So this method can't be used to
    find the solutions of `x^2 + y^2 = 20` since the only solution to former is
    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the
    solutions with `x \leq y` are found. For more details, see the References.

    Examples
    ========

    >>> from sympy.solvers.diophantine import cornacchia
    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35
    {(2, 3), (4, 1)}
    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25
    {(4, 3)}

    References
    ===========

    .. [1] A. Nitaj, "L'algorithme de Cornacchia"
    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia's
        method, [online], Available:
        http://www.numbertheory.org/php/cornacchia.html

    See Also
    ========
    sympy.utilities.iterables.signed_permutations
    """
    sols = set()

    a1 = igcdex(a, m)[0]
    v = sqrt_mod(-b*a1, m, all_roots=True)
    if not v:
        return None

    for t in v:
        if t < m // 2:
            continue

        u, r = t, m

        while True:
            u, r = r, u % r
            if a*r**2 < m:
                break

        m1 = m - a*r**2

        if m1 % b == 0:
            m1 = m1 // b
            s, _exact = integer_nthroot(m1, 2)
            if _exact:
                if a == b and r < s:
                    r, s = s, r
                sols.add((int(r), int(s)))

    return sols
```
