# sympy__sympy-18810

| **sympy/sympy** | `a1fbd0066219a7a1d14d4d9024d8aeeb5cb8d51a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 326 |
| **Any found context length** | 326 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/utilities/iterables.py b/sympy/utilities/iterables.py
--- a/sympy/utilities/iterables.py
+++ b/sympy/utilities/iterables.py
@@ -2251,12 +2251,9 @@ def generate_derangements(perm):
     ========
     sympy.functions.combinatorial.factorials.subfactorial
     """
-    p = multiset_permutations(perm)
-    indices = range(len(perm))
-    p0 = next(p)
-    for pi in p:
-        if all(pi[i] != p0[i] for i in indices):
-            yield pi
+    for p in multiset_permutations(perm):
+        if not any(i == j for i, j in zip(perm, p)):
+            yield p
 
 
 def necklaces(n, k, free=False):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/utilities/iterables.py | 2254 | 2259 | 1 | 1 | 326


## Problem Statement

```
generate_derangements mishandles unsorted perm
The following is incorrect:
\`\`\`python
>>> list('TRUMP') in generate_derangements('TRUMP')
True
\`\`\`
The routine is assuming that the `perm` is sorted (though this is not a requirement):
\`\`\`python
>>> list('MPRTU') in generate_derangements('MPRTU')
False
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/utilities/iterables.py** | 2229 | 2259| 326 | 326 | 21839 | 
| 2 | 2 sympy/polys/polyutils.py | 62 | 107| 247 | 573 | 25284 | 
| 3 | 3 sympy/combinatorics/permutations.py | 1705 | 1736| 275 | 848 | 48958 | 
| 4 | 3 sympy/combinatorics/permutations.py | 2471 | 2507| 334 | 1182 | 48958 | 
| 5 | 3 sympy/combinatorics/permutations.py | 2843 | 2876| 264 | 1446 | 48958 | 
| 6 | 3 sympy/combinatorics/permutations.py | 470 | 829| 3434 | 4880 | 48958 | 
| 7 | 3 sympy/combinatorics/permutations.py | 2509 | 2562| 468 | 5348 | 48958 | 
| 8 | 3 sympy/combinatorics/permutations.py | 1802 | 1837| 251 | 5599 | 48958 | 
| 9 | 3 sympy/combinatorics/permutations.py | 1738 | 1772| 261 | 5860 | 48958 | 
| 10 | 4 sympy/combinatorics/perm_groups.py | 1 | 4862| 245 | 6105 | 93156 | 
| 11 | 4 sympy/combinatorics/permutations.py | 1662 | 1703| 297 | 6402 | 93156 | 
| 12 | 5 sympy/combinatorics/generators.py | 239 | 309| 432 | 6834 | 95630 | 
| 13 | 5 sympy/combinatorics/permutations.py | 2240 | 2265| 183 | 7017 | 95630 | 
| 14 | 6 sympy/combinatorics/testutil.py | 1 | 30| 260 | 7277 | 98704 | 
| 15 | 6 sympy/combinatorics/permutations.py | 2033 | 2053| 155 | 7432 | 98704 | 
| 16 | 6 sympy/combinatorics/permutations.py | 1774 | 1800| 214 | 7646 | 98704 | 
| 17 | 7 sympy/codegen/array_utils.py | 552 | 574| 174 | 7820 | 112129 | 
| 18 | 8 sympy/combinatorics/prufer.py | 293 | 315| 153 | 7973 | 115502 | 
| 19 | 9 sympy/combinatorics/__init__.py | 1 | 41| 387 | 8360 | 115889 | 
| 20 | 9 sympy/combinatorics/permutations.py | 894 | 956| 602 | 8962 | 115889 | 
| 21 | 9 sympy/codegen/array_utils.py | 576 | 631| 684 | 9646 | 115889 | 
| 22 | 9 sympy/combinatorics/permutations.py | 2103 | 2158| 405 | 10051 | 115889 | 
| 23 | 9 sympy/combinatorics/prufer.py | 317 | 334| 150 | 10201 | 115889 | 
| 24 | 10 sympy/combinatorics/partitions.py | 677 | 709| 268 | 10469 | 121610 | 
| 25 | 11 sympy/combinatorics/pc_groups.py | 444 | 481| 305 | 10774 | 126878 | 
| 26 | 11 sympy/combinatorics/permutations.py | 2429 | 2469| 314 | 11088 | 126878 | 
| 27 | 11 sympy/combinatorics/perm_groups.py | 1409 | 1503| 824 | 11912 | 126878 | 
| 28 | 11 sympy/combinatorics/partitions.py | 712 | 733| 177 | 12089 | 126878 | 
| 29 | **11 sympy/utilities/iterables.py** | 1330 | 1376| 416 | 12505 | 126878 | 
| 30 | 11 sympy/combinatorics/prufer.py | 97 | 121| 164 | 12669 | 126878 | 
| 31 | 11 sympy/combinatorics/permutations.py | 2011 | 2031| 155 | 12824 | 126878 | 
| 32 | 11 sympy/combinatorics/prufer.py | 411 | 434| 162 | 12986 | 126878 | 
| 33 | 11 sympy/combinatorics/permutations.py | 1456 | 1492| 332 | 13318 | 126878 | 
| 34 | 11 sympy/combinatorics/permutations.py | 2747 | 2794| 396 | 13714 | 126878 | 
| 35 | 11 sympy/combinatorics/perm_groups.py | 4938 | 4965| 225 | 13939 | 126878 | 
| 36 | 11 sympy/combinatorics/generators.py | 104 | 122| 451 | 14390 | 126878 | 
| 37 | 11 sympy/codegen/array_utils.py | 524 | 551| 225 | 14615 | 126878 | 
| 38 | 12 sympy/matrices/expressions/permutation.py | 218 | 250| 264 | 14879 | 128841 | 
| 39 | 13 sympy/combinatorics/polyhedron.py | 509 | 575| 621 | 15500 | 141744 | 
| 40 | 13 sympy/combinatorics/permutations.py | 2357 | 2375| 168 | 15668 | 141744 | 
| 41 | 13 sympy/combinatorics/permutations.py | 2209 | 2238| 166 | 15834 | 141744 | 
| 42 | 13 sympy/combinatorics/permutations.py | 1019 | 1057| 335 | 16169 | 141744 | 
| 43 | 13 sympy/combinatorics/prufer.py | 389 | 409| 162 | 16331 | 141744 | 
| 44 | 13 sympy/matrices/expressions/permutation.py | 252 | 274| 158 | 16489 | 141744 | 
| 45 | **13 sympy/utilities/iterables.py** | 2085 | 2149| 756 | 17245 | 141744 | 
| 46 | 14 sympy/combinatorics/util.py | 381 | 455| 674 | 17919 | 146410 | 
| 47 | 14 sympy/combinatorics/permutations.py | 1154 | 1188| 222 | 18141 | 146410 | 
| 48 | 14 sympy/combinatorics/testutil.py | 78 | 113| 302 | 18443 | 146410 | 
| 49 | 14 sympy/combinatorics/permutations.py | 958 | 992| 298 | 18741 | 146410 | 
| 50 | 14 sympy/combinatorics/permutations.py | 2055 | 2077| 140 | 18881 | 146410 | 
| 51 | 14 sympy/combinatorics/permutations.py | 2377 | 2427| 470 | 19351 | 146410 | 
| 52 | 15 sympy/utilities/enumerative.py | 425 | 440| 141 | 19492 | 156645 | 
| 53 | 15 sympy/combinatorics/partitions.py | 119 | 157| 266 | 19758 | 156645 | 
| 54 | 15 sympy/combinatorics/util.py | 1 | 69| 523 | 20281 | 156645 | 
| 55 | 15 sympy/matrices/expressions/permutation.py | 101 | 163| 423 | 20704 | 156645 | 
| 56 | 15 sympy/combinatorics/permutations.py | 1320 | 1338| 168 | 20872 | 156645 | 
| 57 | 15 sympy/combinatorics/permutations.py | 2079 | 2101| 140 | 21012 | 156645 | 
| 58 | 15 sympy/combinatorics/permutations.py | 2824 | 2841| 118 | 21130 | 156645 | 
| 59 | 16 sympy/physics/secondquant.py | 2620 | 2674| 557 | 21687 | 179180 | 
| 60 | 16 sympy/combinatorics/permutations.py | 1911 | 1932| 127 | 21814 | 179180 | 
| 61 | 17 sympy/core/trace.py | 22 | 55| 273 | 22087 | 180685 | 
| 62 | 17 sympy/combinatorics/permutations.py | 2564 | 2597| 333 | 22420 | 180685 | 
| 63 | 17 sympy/matrices/expressions/permutation.py | 276 | 302| 182 | 22602 | 180685 | 


## Patch

```diff
diff --git a/sympy/utilities/iterables.py b/sympy/utilities/iterables.py
--- a/sympy/utilities/iterables.py
+++ b/sympy/utilities/iterables.py
@@ -2251,12 +2251,9 @@ def generate_derangements(perm):
     ========
     sympy.functions.combinatorial.factorials.subfactorial
     """
-    p = multiset_permutations(perm)
-    indices = range(len(perm))
-    p0 = next(p)
-    for pi in p:
-        if all(pi[i] != p0[i] for i in indices):
-            yield pi
+    for p in multiset_permutations(perm):
+        if not any(i == j for i, j in zip(perm, p)):
+            yield p
 
 
 def necklaces(n, k, free=False):

```

## Test Patch

```diff
diff --git a/sympy/utilities/tests/test_iterables.py b/sympy/utilities/tests/test_iterables.py
--- a/sympy/utilities/tests/test_iterables.py
+++ b/sympy/utilities/tests/test_iterables.py
@@ -543,6 +543,7 @@ def test_derangements():
         [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], [3, 2, 1, 0]]
     assert list(generate_derangements([0, 1, 2, 2])) == [
         [2, 2, 0, 1], [2, 2, 1, 0]]
+    assert list(generate_derangements('ba')) == [list('ab')]
 
 
 def test_necklaces():

```


## Code snippets

### 1 - sympy/utilities/iterables.py:

Start line: 2229, End line: 2259

```python
def generate_derangements(perm):
    """
    Routine to generate unique derangements.

    TODO: This will be rewritten to use the
    ECO operator approach once the permutations
    branch is in master.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 3]))
    [[1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 3, 1], \
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], \
    [3, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========
    sympy.functions.combinatorial.factorials.subfactorial
    """
    p = multiset_permutations(perm)
    indices = range(len(perm))
    p0 = next(p)
    for pi in p:
        if all(pi[i] != p0[i] for i in indices):
            yield pi
```
### 2 - sympy/polys/polyutils.py:

Start line: 62, End line: 107

```python
def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
    opt = build_options(args)

    gens_order, wrt = {}, None

    if opt is not None:
        gens_order, wrt = {}, opt.wrt

        for i, gen in enumerate(opt.sort):
            gens_order[gen] = i + 1

    def order_key(gen):
        gen = str(gen)

        if wrt is not None:
            try:
                return (-len(wrt) + wrt.index(gen), gen, 0)
            except ValueError:
                pass

        name, index = _re_gen.match(gen).groups()

        if index:
            index = int(index)
        else:
            index = 0

        try:
            return ( gens_order[name], name, index)
        except KeyError:
            pass

        try:
            return (_gens_order[name], name, index)
        except KeyError:
            pass

        return (_max_order, name, index)

    try:
        gens = sorted(gens, key=order_key)
    except TypeError:  # pragma: no cover
        pass

    return tuple(gens)
```
### 3 - sympy/combinatorics/permutations.py:

Start line: 1705, End line: 1736

```python
class Permutation(Atom):

    @classmethod
    def unrank_nonlex(self, n, r):
        """
        This is a linear time unranking algorithm that does not
        respect lexicographic order [3].

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_nonlex(4, 5)
        Permutation([2, 0, 3, 1])
        >>> Permutation.unrank_nonlex(4, -1)
        Permutation([0, 1, 2, 3])

        See Also
        ========

        next_nonlex, rank_nonlex
        """
        def _unrank1(n, r, a):
            if n > 0:
                a[n - 1], a[r % n] = a[r % n], a[n - 1]
                _unrank1(n - 1, r//n, a)

        id_perm = list(range(n))
        n = int(n)
        r = r % ifac(n)
        _unrank1(n, r, id_perm)
        return self._af_new(id_perm)
```
### 4 - sympy/combinatorics/permutations.py:

Start line: 2471, End line: 2507

```python
class Permutation(Atom):

    @classmethod
    def unrank_trotterjohnson(cls, size, rank):
        """
        Trotter Johnson permutation unranking. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_trotterjohnson(5, 10)
        Permutation([0, 3, 1, 2, 4])

        See Also
        ========

        rank_trotterjohnson, next_trotterjohnson
        """
        perm = [0]*size
        r2 = 0
        n = ifac(size)
        pj = 1
        for j in range(2, size + 1):
            pj *= j
            r1 = (rank * pj) // n
            k = r1 - j*r2
            if r2 % 2 == 0:
                for i in range(j - 1, j - k - 1, -1):
                    perm[i] = perm[i - 1]
                perm[j - k - 1] = j - 1
            else:
                for i in range(j - 1, k, -1):
                    perm[i] = perm[i - 1]
                perm[k] = j - 1
            r2 = r1
        return cls._af_new(perm)
```
### 5 - sympy/combinatorics/permutations.py:

Start line: 2843, End line: 2876

```python
class Permutation(Atom):

    @classmethod
    def unrank_lex(cls, size, rank):
        """
        Lexicographic permutation unranking.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = Permutation.unrank_lex(5, 10)
        >>> a.rank()
        10
        >>> a
        Permutation([0, 2, 4, 1, 3])

        See Also
        ========

        rank, next_lex
        """
        perm_array = [0] * size
        psize = 1
        for i in range(size):
            new_psize = psize*(i + 1)
            d = (rank % new_psize) // psize
            rank -= d*psize
            perm_array[size - i - 1] = d
            for j in range(size - i, size):
                if perm_array[j] > d - 1:
                    perm_array[j] += 1
            psize = new_psize
        return cls._af_new(perm_array)
```
### 6 - sympy/combinatorics/permutations.py:

Start line: 470, End line: 829

```python
class Permutation(Atom):
    """
    A permutation, alternatively known as an 'arrangement number' or 'ordering'
    is an arrangement of the elements of an ordered list into a one-to-one
    mapping with itself. The permutation of a given arrangement is given by
    indicating the positions of the elements after re-arrangement [2]_. For
    example, if one started with elements [x, y, a, b] (in that order) and
    they were reordered as [x, y, b, a] then the permutation would be
    [0, 1, 3, 2]. Notice that (in SymPy) the first element is always referred
    to as 0 and the permutation uses the indices of the elements in the
    original ordering, not the elements (a, b, etc...) themselves.

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.interactive import init_printing
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    Permutations Notation
    =====================

    Permutations are commonly represented in disjoint cycle or array forms.

    Array Notation and 2-line Form
    ------------------------------------

    In the 2-line form, the elements and their final positions are shown
    as a matrix with 2 rows:

    [0    1    2     ... n-1]
    [p(0) p(1) p(2)  ... p(n-1)]

    Since the first line is always range(n), where n is the size of p,
    it is sufficient to represent the permutation by the second line,
    referred to as the "array form" of the permutation. This is entered
    in brackets as the argument to the Permutation class:

    >>> p = Permutation([0, 2, 1]); p
    Permutation([0, 2, 1])

    Given i in range(p.size), the permutation maps i to i^p

    >>> [i^p for i in range(p.size)]
    [0, 2, 1]

    The composite of two permutations p*q means first apply p, then q, so
    i^(p*q) = (i^p)^q which is i^p^q according to Python precedence rules:

    >>> q = Permutation([2, 1, 0])
    >>> [i^p^q for i in range(3)]
    [2, 0, 1]
    >>> [i^(p*q) for i in range(3)]
    [2, 0, 1]

    One can use also the notation p(i) = i^p, but then the composition
    rule is (p*q)(i) = q(p(i)), not p(q(i)):

    >>> [(p*q)(i) for i in range(p.size)]
    [2, 0, 1]
    >>> [q(p(i)) for i in range(p.size)]
    [2, 0, 1]
    >>> [p(q(i)) for i in range(p.size)]
    [1, 2, 0]

    Disjoint Cycle Notation
    -----------------------

    In disjoint cycle notation, only the elements that have shifted are
    indicated. In the above case, the 2 and 1 switched places. This can
    be entered in two ways:

    >>> Permutation(1, 2) == Permutation([[1, 2]]) == p
    True

    Only the relative ordering of elements in a cycle matter:

    >>> Permutation(1,2,3) == Permutation(2,3,1) == Permutation(3,1,2)
    True

    The disjoint cycle notation is convenient when representing
    permutations that have several cycles in them:

    >>> Permutation(1, 2)(3, 5) == Permutation([[1, 2], [3, 5]])
    True

    It also provides some economy in entry when computing products of
    permutations that are written in disjoint cycle notation:

    >>> Permutation(1, 2)(1, 3)(2, 3)
    Permutation([0, 3, 2, 1])
    >>> _ == Permutation([[1, 2]])*Permutation([[1, 3]])*Permutation([[2, 3]])
    True

        Caution: when the cycles have common elements
        between them then the order in which the
        permutations are applied matters. The
        convention is that the permutations are
        applied from *right to left*. In the following, the
        transposition of elements 2 and 3 is followed
        by the transposition of elements 1 and 2:

        >>> Permutation(1, 2)(2, 3) == Permutation([(1, 2), (2, 3)])
        True
        >>> Permutation(1, 2)(2, 3).list()
        [0, 3, 1, 2]

        If the first and second elements had been
        swapped first, followed by the swapping of the second
        and third, the result would have been [0, 2, 3, 1].
        If, for some reason, you want to apply the cycles
        in the order they are entered, you can simply reverse
        the order of cycles:

        >>> Permutation([(1, 2), (2, 3)][::-1]).list()
        [0, 2, 3, 1]

    Entering a singleton in a permutation is a way to indicate the size of the
    permutation. The ``size`` keyword can also be used.

    Array-form entry:

    >>> Permutation([[1, 2], [9]])
    Permutation([0, 2, 1], size=10)
    >>> Permutation([[1, 2]], size=10)
    Permutation([0, 2, 1], size=10)

    Cyclic-form entry:

    >>> Permutation(1, 2, size=10)
    Permutation([0, 2, 1], size=10)
    >>> Permutation(9)(1, 2)
    Permutation([0, 2, 1], size=10)

    Caution: no singleton containing an element larger than the largest
    in any previous cycle can be entered. This is an important difference
    in how Permutation and Cycle handle the __call__ syntax. A singleton
    argument at the start of a Permutation performs instantiation of the
    Permutation and is permitted:

    >>> Permutation(5)
    Permutation([], size=6)

    A singleton entered after instantiation is a call to the permutation
    -- a function call -- and if the argument is out of range it will
    trigger an error. For this reason, it is better to start the cycle
    with the singleton:

    The following fails because there is no element 3:

    >>> Permutation(1, 2)(3)
    Traceback (most recent call last):
    ...
    IndexError: list index out of range

    This is ok: only the call to an out of range singleton is prohibited;
    otherwise the permutation autosizes:

    >>> Permutation(3)(1, 2)
    Permutation([0, 2, 1, 3])
    >>> Permutation(1, 2)(3, 4) == Permutation(3, 4)(1, 2)
    True


    Equality testing
    ----------------

    The array forms must be the same in order for permutations to be equal:

    >>> Permutation([1, 0, 2, 3]) == Permutation([1, 0])
    False


    Identity Permutation
    --------------------

    The identity permutation is a permutation in which no element is out of
    place. It can be entered in a variety of ways. All the following create
    an identity permutation of size 4:

    >>> I = Permutation([0, 1, 2, 3])
    >>> all(p == I for p in [
    ... Permutation(3),
    ... Permutation(range(4)),
    ... Permutation([], size=4),
    ... Permutation(size=4)])
    True

    Watch out for entering the range *inside* a set of brackets (which is
    cycle notation):

    >>> I == Permutation([range(4)])
    False


    Permutation Printing
    ====================

    There are a few things to note about how Permutations are printed.

    1) If you prefer one form (array or cycle) over another, you can set
    ``init_printing`` with the ``perm_cyclic`` flag.

    >>> from sympy import init_printing
    >>> p = Permutation(1, 2)(4, 5)(3, 4)
    >>> p
    Permutation([0, 2, 1, 4, 5, 3])

    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (1 2)(3 4 5)

    2) Regardless of the setting, a list of elements in the array for cyclic
    form can be obtained and either of those can be copied and supplied as
    the argument to Permutation:

    >>> p.array_form
    [0, 2, 1, 4, 5, 3]
    >>> p.cyclic_form
    [[1, 2], [3, 4, 5]]
    >>> Permutation(_) == p
    True

    3) Printing is economical in that as little as possible is printed while
    retaining all information about the size of the permutation:

    >>> init_printing(perm_cyclic=False, pretty_print=False)
    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    >>> Permutation([1, 0, 2, 3], size=20)
    Permutation([1, 0], size=20)
    >>> Permutation([1, 0, 2, 4, 3, 5, 6], size=20)
    Permutation([1, 0, 2, 4, 3], size=20)

    >>> p = Permutation([1, 0, 2, 3])
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (3)(0 1)
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    The 2 was not printed but it is still there as can be seen with the
    array_form and size methods:

    >>> p.array_form
    [1, 0, 2, 3]
    >>> p.size
    4

    Short introduction to other methods
    ===================================

    The permutation can act as a bijective function, telling what element is
    located at a given position

    >>> q = Permutation([5, 2, 3, 4, 1, 0])
    >>> q.array_form[1] # the hard way
    2
    >>> q(1) # the easy way
    2
    >>> {i: q(i) for i in range(q.size)} # showing the bijection
    {0: 5, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0}

    The full cyclic form (including singletons) can be obtained:

    >>> p.full_cyclic_form
    [[0, 1], [2], [3]]

    Any permutation can be factored into transpositions of pairs of elements:

    >>> Permutation([[1, 2], [3, 4, 5]]).transpositions()
    [(1, 2), (3, 5), (3, 4)]
    >>> Permutation.rmul(*[Permutation([ti], size=6) for ti in _]).cyclic_form
    [[1, 2], [3, 4, 5]]

    The number of permutations on a set of n elements is given by n! and is
    called the cardinality.

    >>> p.size
    4
    >>> p.cardinality
    24

    A given permutation has a rank among all the possible permutations of the
    same elements, but what that rank is depends on how the permutations are
    enumerated. (There are a number of different methods of doing so.) The
    lexicographic rank is given by the rank method and this rank is used to
    increment a permutation with addition/subtraction:

    >>> p.rank()
    6
    >>> p + 1
    Permutation([1, 0, 3, 2])
    >>> p.next_lex()
    Permutation([1, 0, 3, 2])
    >>> _.rank()
    7
    >>> p.unrank_lex(p.size, rank=7)
    Permutation([1, 0, 3, 2])

    The product of two permutations p and q is defined as their composition as
    functions, (p*q)(i) = q(p(i)) [6]_.

    >>> p = Permutation([1, 0, 2, 3])
    >>> q = Permutation([2, 3, 1, 0])
    >>> list(q*p)
    [2, 3, 0, 1]
    >>> list(p*q)
    [3, 2, 1, 0]
    >>> [q(p(i)) for i in range(p.size)]
    [3, 2, 1, 0]

    The permutation can be 'applied' to any list-like object, not only
    Permutations:

    >>> p(['zero', 'one', 'four', 'two'])
    ['one', 'zero', 'four', 'two']
    >>> p('zo42')
    ['o', 'z', '4', '2']

    If you have a list of arbitrary elements, the corresponding permutation
    can be found with the from_sequence method:

    >>> Permutation.from_sequence('SymPy')
    Permutation([1, 3, 2, 0, 4])

    See Also
    ========

    Cycle

    References
    ==========

    .. [1] Skiena, S. 'Permutations.' 1.1 in Implementing Discrete Mathematics
           Combinatorics and Graph Theory with Mathematica.  Reading, MA:
           Addison-Wesley, pp. 3-16, 1990.

    .. [2] Knuth, D. E. The Art of Computer Programming, Vol. 4: Combinatorial
           Algorithms, 1st ed. Reading, MA: Addison-Wesley, 2011.

    .. [3] Wendy Myrvold and Frank Ruskey. 2001. Ranking and unranking
           permutations in linear time. Inf. Process. Lett. 79, 6 (September 2001),
           281-284. DOI=10.1016/S0020-0190(01)00141-7

    .. [4] D. L. Kreher, D. R. Stinson 'Combinatorial Algorithms'
           CRC Press, 1999

    .. [5] Graham, R. L.; Knuth, D. E.; and Patashnik, O.
           Concrete Mathematics: A Foundation for Computer Science, 2nd ed.
           Reading, MA: Addison-Wesley, 1994.

    .. [6] https://en.wikipedia.org/wiki/Permutation#Product_and_inverse

    .. [7] https://en.wikipedia.org/wiki/Lehmer_code

    """

    is_Permutation = True

    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None
```
### 7 - sympy/combinatorics/permutations.py:

Start line: 2509, End line: 2562

```python
class Permutation(Atom):

    def next_trotterjohnson(self):
        """
        Returns the next permutation in Trotter-Johnson order.
        If self is the last permutation it returns None.
        See [4] section 2.4. If it is desired to generate all such
        permutations, they can be generated in order more quickly
        with the ``generate_bell`` function.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 0, 2, 1])
        >>> p.rank_trotterjohnson()
        4
        >>> p = p.next_trotterjohnson(); p
        Permutation([0, 3, 2, 1])
        >>> p.rank_trotterjohnson()
        5

        See Also
        ========

        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell
        """
        pi = self.array_form[:]
        n = len(pi)
        st = 0
        rho = pi[:]
        done = False
        m = n-1
        while m > 0 and not done:
            d = rho.index(m)
            for i in range(d, m):
                rho[i] = rho[i + 1]
            par = _af_parity(rho[:m])
            if par == 1:
                if d == m:
                    m -= 1
                else:
                    pi[st + d], pi[st + d + 1] = pi[st + d + 1], pi[st + d]
                    done = True
            else:
                if d == 0:
                    m -= 1
                    st += 1
                else:
                    pi[st + d], pi[st + d - 1] = pi[st + d - 1], pi[st + d]
                    done = True
        if m == 0:
            return None
        return self._af_new(pi)
```
### 8 - sympy/combinatorics/permutations.py:

Start line: 1802, End line: 1837

```python
class Permutation(Atom):

    def rank(self):
        """
        Returns the lexicographic rank of the permutation.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank()
        0
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.rank()
        23

        See Also
        ========

        next_lex, unrank_lex, cardinality, length, order, size
        """
        if not self._rank is None:
            return self._rank
        rank = 0
        rho = self.array_form[:]
        n = self.size - 1
        size = n + 1
        psize = int(ifac(n))
        for j in range(size - 1):
            rank += rho[j]*psize
            for i in range(j + 1, size):
                if rho[i] > rho[j]:
                    rho[i] -= 1
            psize //= n
            n -= 1
        self._rank = rank
        return rank
```
### 9 - sympy/combinatorics/permutations.py:

Start line: 1738, End line: 1772

```python
class Permutation(Atom):

    def rank_nonlex(self, inv_perm=None):
        """
        This is a linear time ranking algorithm that does not
        enforce lexicographic order [3].


        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_nonlex()
        23

        See Also
        ========

        next_nonlex, unrank_nonlex
        """
        def _rank1(n, perm, inv_perm):
            if n == 1:
                return 0
            s = perm[n - 1]
            t = inv_perm[n - 1]
            perm[n - 1], perm[t] = perm[t], s
            inv_perm[n - 1], inv_perm[s] = inv_perm[s], t
            return s + n*_rank1(n - 1, perm, inv_perm)

        if inv_perm is None:
            inv_perm = (~self).array_form
        if not inv_perm:
            return 0
        perm = self.array_form[:]
        r = _rank1(len(perm), perm, inv_perm)
        return r
```
### 10 - sympy/combinatorics/perm_groups.py:

Start line: 1, End line: 4862

```python
from __future__ import print_function, division

from random import randrange, choice
from math import log
from sympy.ntheory import primefactors
from sympy import multiplicity, factorint

from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
    _af_rmul, _af_rmuln, _af_pow, Cycle)
from sympy.combinatorics.util import (_check_cycles_alt_sym,
    _distribute_gens_by_base, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs, _base_ordering, _strong_gens_from_distr,
    _strip, _strip_af)
from sympy.core import Basic
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import sieve
from sympy.utilities.iterables import has_variety, is_sequence, uniq
from sympy.testing.randtest import _randrange
from itertools import islice

rmul = Permutation.rmul_with_af
_af_new = Permutation._af_new


class PermutationGroup(Basic):
```
### 29 - sympy/utilities/iterables.py:

Start line: 1330, End line: 1376

```python
def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    if g is None:
        if type(m) is dict:
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum([gi[1] for gi in do])
    if not do or size is not None and (size > SUM or size < 1):
        if size < 1:
            yield []
        return
    elif size == 1:
        for k, v in do:
            yield [k]
    elif len(do) == 1:
        k, v = do[0]
        v = v if size is None else (size if size <= v else 0)
        yield [k for i in range(v)]
    elif all(v == 1 for k, v in do):
        for p in permutations([k for k, v in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1
```
### 45 - sympy/utilities/iterables.py:

Start line: 2085, End line: 2149

```python
def generate_bell(n):
    """Return permutations of [0, 1, ..., n - 1] such that each permutation
    differs from the last by the exchange of a single pair of neighbors.
    The ``n!`` permutations are returned as an iterator. In order to obtain
    the next permutation from a random starting permutation, use the
    ``next_trotterjohnson`` method of the Permutation class (which generates
    the same sequence in a different manner).

    Examples
    ========

    >>> from itertools import permutations
    >>> from sympy.utilities.iterables import generate_bell
    >>> from sympy import zeros, Matrix

    This is the sort of permutation used in the ringing of physical bells,
    and does not produce permutations in lexicographical order. Rather, the
    permutations differ from each other by exactly one inversion, and the
    position at which the swapping occurs varies periodically in a simple
    fashion. Consider the first few permutations of 4 elements generated
    by ``permutations`` and ``generate_bell``:

    >>> list(permutations(range(4)))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    >>> list(generate_bell(4))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]

    Notice how the 2nd and 3rd lexicographical permutations have 3 elements
    out of place whereas each "bell" permutation always has only two
    elements out of place relative to the previous permutation (and so the
    signature (+/-1) of a permutation is opposite of the signature of the
    previous permutation).

    How the position of inversion varies across the elements can be seen
    by tracing out where the largest number appears in the permutations:

    >>> m = zeros(4, 24)
    >>> for i, p in enumerate(generate_bell(4)):
    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero
    >>> m.print_nonzero('X')
    [XXX  XXXXXX  XXXXXX  XXX]
    [XX XX XXXX XX XXXX XX XX]
    [X XXXX XX XXXX XX XXXX X]
    [ XXXXXX  XXXXXX  XXXXXX ]

    See Also
    ========
    sympy.combinatorics.permutations.Permutation.next_trotterjohnson

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Method_ringing

    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018

    .. [3] http://programminggeeks.com/bell-algorithm-for-permutation/

    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

    .. [5] Generating involutions, derangements, and relatives by ECO
           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010

    """
    n = as_int(n)
    # ... other code
```
