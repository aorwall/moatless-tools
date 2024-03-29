# sympy__sympy-21864

| **sympy/sympy** | `ec0fe8c5f3e59840e8aa5d3d6a7c976e40f76b64` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 416 |
| **Any found context length** | 416 |
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
@@ -1419,7 +1419,7 @@ def multiset_permutations(m, size=None, g=None):
     do = [gi for gi in g if gi[1] > 0]
     SUM = sum([gi[1] for gi in do])
     if not do or size is not None and (size > SUM or size < 1):
-        if size < 1:
+        if not do and size is None or size == 0:
             yield []
         return
     elif size == 1:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/utilities/iterables.py | 1422 | 1422 | 1 | 1 | 416


## Problem Statement

```
multiset_permutations needs to handle []
\`\`\`diff
diff --git a/sympy/utilities/iterables.py b/sympy/utilities/iterables.py
index 83fc2f48d2..0a91615dde 100644
--- a/sympy/utilities/iterables.py
+++ b/sympy/utilities/iterables.py
@@ -1419,7 +1419,7 @@ def multiset_permutations(m, size=None, g=None):
     do = [gi for gi in g if gi[1] > 0]
     SUM = sum([gi[1] for gi in do])
     if not do or size is not None and (size > SUM or size < 1):
-        if size < 1:
+        if not do and size is None or size < 1:
             yield []
         return
     elif size == 1:
diff --git a/sympy/utilities/tests/test_iterables.py b/sympy/utilities/tests/test_iterables.py
index 221b03f618..b405ac37f5 100644
--- a/sympy/utilities/tests/test_iterables.py
+++ b/sympy/utilities/tests/test_iterables.py
@@ -423,6 +423,9 @@ def test_multiset_permutations():
         [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
     assert len(list(multiset_permutations('a', 0))) == 1
     assert len(list(multiset_permutations('a', 3))) == 0
+    for nul in ([], {}, ''):
+        assert list(multiset_permutations(nul)) == [[]], list(multiset_permutations(nul))
+        assert list(multiset_permutations(nul, 1)) == []
 
     def test():
         for i in range(1, 7):
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/utilities/iterables.py** | 1396 | 1442| 416 | 416 | 21997 | 
| 2 | **1 sympy/utilities/iterables.py** | 1639 | 1716| 694 | 1110 | 21997 | 
| 3 | **1 sympy/utilities/iterables.py** | 1340 | 1393| 442 | 1552 | 21997 | 
| 4 | **1 sympy/utilities/iterables.py** | 1562 | 1638| 755 | 2307 | 21997 | 
| 5 | 2 sympy/combinatorics/permutations.py | 472 | 881| 4021 | 6328 | 46274 | 
| 6 | 3 sympy/utilities/enumerative.py | 370 | 421| 436 | 6764 | 56498 | 
| 7 | 4 sympy/combinatorics/perm_groups.py | 1 | 4988| 250 | 7014 | 102172 | 
| 8 | **4 sympy/utilities/iterables.py** | 1 | 18| 120 | 7134 | 102172 | 
| 9 | **4 sympy/utilities/iterables.py** | 2288 | 2317| 309 | 7443 | 102172 | 
| 10 | **4 sympy/utilities/iterables.py** | 2595 | 2623| 247 | 7690 | 102172 | 
| 11 | **4 sympy/utilities/iterables.py** | 303 | 323| 121 | 7811 | 102172 | 
| 12 | 4 sympy/utilities/enumerative.py | 241 | 296| 475 | 8286 | 102172 | 
| 13 | 4 sympy/utilities/enumerative.py | 783 | 851| 520 | 8806 | 102172 | 
| 14 | 5 sympy/combinatorics/testutil.py | 1 | 33| 267 | 9073 | 105299 | 
| 15 | 5 sympy/combinatorics/permutations.py | 961 | 1007| 449 | 9522 | 105299 | 
| 16 | 5 sympy/combinatorics/permutations.py | 1 | 18| 148 | 9670 | 105299 | 
| 17 | 5 sympy/combinatorics/permutations.py | 2008 | 2031| 119 | 9789 | 105299 | 
| 18 | 6 sympy/functions/combinatorial/numbers.py | 1372 | 1416| 218 | 10007 | 124281 | 
| 19 | 6 sympy/utilities/enumerative.py | 132 | 241| 964 | 10971 | 124281 | 
| 20 | 7 sympy/ntheory/multinomial.py | 129 | 189| 607 | 11578 | 125993 | 
| 21 | 7 sympy/utilities/enumerative.py | 976 | 1005| 216 | 11794 | 125993 | 
| 22 | 7 sympy/functions/combinatorial/numbers.py | 1419 | 1444| 265 | 12059 | 125993 | 
| 23 | 8 sympy/combinatorics/__init__.py | 1 | 41| 407 | 12466 | 126400 | 
| 24 | 8 sympy/combinatorics/permutations.py | 1606 | 1645| 380 | 12846 | 126400 | 
| 25 | 8 sympy/utilities/enumerative.py | 853 | 912| 443 | 13289 | 126400 | 
| 26 | 8 sympy/utilities/enumerative.py | 914 | 974| 550 | 13839 | 126400 | 
| 27 | **8 sympy/utilities/iterables.py** | 1809 | 1847| 266 | 14105 | 126400 | 
| 28 | **8 sympy/utilities/iterables.py** | 1928 | 1983| 483 | 14588 | 126400 | 
| 29 | 8 sympy/utilities/enumerative.py | 1007 | 1082| 761 | 15349 | 126400 | 
| 30 | 8 sympy/utilities/enumerative.py | 553 | 578| 333 | 15682 | 126400 | 
| 31 | 9 sympy/matrices/expressions/matmul.py | 350 | 370| 149 | 15831 | 129996 | 
| 32 | 9 sympy/combinatorics/perm_groups.py | 5321 | 5357| 330 | 16161 | 129996 | 
| 33 | 10 sympy/combinatorics/generators.py | 233 | 303| 432 | 16593 | 132398 | 
| 34 | 10 sympy/combinatorics/permutations.py | 1276 | 1297| 177 | 16770 | 132398 | 
| 35 | 10 sympy/combinatorics/permutations.py | 1986 | 2006| 118 | 16888 | 132398 | 
| 36 | 10 sympy/combinatorics/permutations.py | 2811 | 2858| 396 | 17284 | 132398 | 
| 37 | 10 sympy/utilities/enumerative.py | 1083 | 1129| 449 | 17733 | 132398 | 
| 38 | 10 sympy/combinatorics/permutations.py | 883 | 960| 807 | 18540 | 132398 | 
| 39 | 10 sympy/combinatorics/permutations.py | 2526 | 2562| 334 | 18874 | 132398 | 
| 40 | 10 sympy/combinatorics/permutations.py | 1100 | 1141| 306 | 19180 | 132398 | 
| 41 | 10 sympy/combinatorics/permutations.py | 2155 | 2213| 412 | 19592 | 132398 | 
| 42 | **10 sympy/utilities/iterables.py** | 2144 | 2208| 750 | 20342 | 132398 | 
| 43 | 11 sympy/matrices/expressions/permutation.py | 220 | 252| 259 | 20601 | 134369 | 
| 44 | 12 sympy/tensor/array/expressions/array_expressions.py | 290 | 352| 551 | 21152 | 146502 | 
| 45 | **12 sympy/utilities/iterables.py** | 593 | 640| 524 | 21676 | 146502 | 
| 46 | 12 sympy/combinatorics/permutations.py | 1060 | 1098| 335 | 22011 | 146502 | 
| 47 | 12 sympy/tensor/array/expressions/array_expressions.py | 536 | 555| 248 | 22259 | 146502 | 
| 48 | **12 sympy/utilities/iterables.py** | 2642 | 2658| 315 | 22574 | 146502 | 
| 49 | 12 sympy/combinatorics/permutations.py | 1143 | 1177| 214 | 22788 | 146502 | 
| 50 | 12 sympy/tensor/array/expressions/array_expressions.py | 467 | 534| 685 | 23473 | 146502 | 
| 51 | 13 sympy/combinatorics/partitions.py | 650 | 684| 222 | 23695 | 152244 | 
| 52 | 13 sympy/combinatorics/permutations.py | 1662 | 1709| 403 | 24098 | 152244 | 
| 53 | 13 sympy/combinatorics/permutations.py | 1231 | 1274| 346 | 24444 | 152244 | 
| 54 | 13 sympy/combinatorics/permutations.py | 1009 | 1033| 232 | 24676 | 152244 | 
| 55 | 13 sympy/combinatorics/permutations.py | 2942 | 3010| 415 | 25091 | 152244 | 
| 56 | 13 sympy/matrices/expressions/permutation.py | 254 | 276| 158 | 25249 | 152244 | 
| 57 | 13 sympy/utilities/enumerative.py | 1 | 129| 1509 | 26758 | 152244 | 
| 58 | 13 sympy/utilities/enumerative.py | 423 | 438| 141 | 26899 | 152244 | 
| 59 | 13 sympy/matrices/expressions/permutation.py | 103 | 165| 423 | 27322 | 152244 | 
| 60 | 13 sympy/combinatorics/permutations.py | 1888 | 1938| 307 | 27629 | 152244 | 
| 61 | 13 sympy/matrices/expressions/permutation.py | 278 | 304| 182 | 27811 | 152244 | 
| 62 | 13 sympy/functions/combinatorial/numbers.py | 2007 | 2045| 346 | 28157 | 152244 | 
| 63 | 13 sympy/matrices/expressions/permutation.py | 1 | 101| 613 | 28770 | 152244 | 
| 64 | 13 sympy/combinatorics/perm_groups.py | 1460 | 1559| 827 | 29597 | 152244 | 
| 65 | 14 sympy/diffgeom/diffgeom.py | 1 | 27| 207 | 29804 | 170036 | 
| 66 | 15 sympy/sets/sets.py | 644 | 693| 421 | 30225 | 187699 | 
| 67 | 15 sympy/combinatorics/permutations.py | 1503 | 1542| 339 | 30564 | 187699 | 
| 68 | 15 sympy/combinatorics/permutations.py | 2564 | 2617| 468 | 31032 | 187699 | 
| 69 | 15 sympy/functions/combinatorial/numbers.py | 1447 | 1510| 536 | 31568 | 187699 | 
| 70 | **15 sympy/utilities/iterables.py** | 548 | 590| 409 | 31977 | 187699 | 
| 71 | **15 sympy/utilities/iterables.py** | 2251 | 2285| 231 | 32208 | 187699 | 
| 72 | 15 sympy/combinatorics/permutations.py | 1647 | 1660| 138 | 32346 | 187699 | 
| 73 | **15 sympy/utilities/iterables.py** | 1483 | 1559| 802 | 33148 | 187699 | 
| 74 | 15 sympy/combinatorics/permutations.py | 2370 | 2391| 152 | 33300 | 187699 | 
| 75 | 15 sympy/ntheory/multinomial.py | 55 | 126| 347 | 33647 | 187699 | 
| 76 | 15 sympy/tensor/array/expressions/array_expressions.py | 354 | 391| 329 | 33976 | 187699 | 
| 77 | 15 sympy/sets/sets.py | 186 | 222| 374 | 34350 | 187699 | 
| 78 | 15 sympy/tensor/array/expressions/array_expressions.py | 557 | 582| 256 | 34606 | 187699 | 
| 79 | 15 sympy/functions/combinatorial/numbers.py | 1699 | 1714| 168 | 34774 | 187699 | 
| 80 | 15 sympy/combinatorics/perm_groups.py | 5210 | 5242| 218 | 34992 | 187699 | 
| 81 | 15 sympy/combinatorics/permutations.py | 2907 | 2940| 264 | 35256 | 187699 | 
| 82 | 16 sympy/sets/fancysets.py | 403 | 437| 279 | 35535 | 199171 | 
| 83 | 16 sympy/combinatorics/permutations.py | 2322 | 2341| 121 | 35656 | 199171 | 


## Patch

```diff
diff --git a/sympy/utilities/iterables.py b/sympy/utilities/iterables.py
--- a/sympy/utilities/iterables.py
+++ b/sympy/utilities/iterables.py
@@ -1419,7 +1419,7 @@ def multiset_permutations(m, size=None, g=None):
     do = [gi for gi in g if gi[1] > 0]
     SUM = sum([gi[1] for gi in do])
     if not do or size is not None and (size > SUM or size < 1):
-        if size < 1:
+        if not do and size is None or size == 0:
             yield []
         return
     elif size == 1:

```

## Test Patch

```diff
diff --git a/sympy/utilities/tests/test_iterables.py b/sympy/utilities/tests/test_iterables.py
--- a/sympy/utilities/tests/test_iterables.py
+++ b/sympy/utilities/tests/test_iterables.py
@@ -423,6 +423,12 @@ def test_multiset_permutations():
         [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
     assert len(list(multiset_permutations('a', 0))) == 1
     assert len(list(multiset_permutations('a', 3))) == 0
+    for nul in ([], {}, ''):
+        assert list(multiset_permutations(nul)) == [[]]
+    assert list(multiset_permutations(nul, 0)) == [[]]
+    # impossible requests give no result
+    assert list(multiset_permutations(nul, 1)) == []
+    assert list(multiset_permutations(nul, -1)) == []
 
     def test():
         for i in range(1, 7):

```


## Code snippets

### 1 - sympy/utilities/iterables.py:

Start line: 1396, End line: 1442

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
### 2 - sympy/utilities/iterables.py:

Start line: 1639, End line: 1716

```python
def multiset_partitions(multiset, m=None):
    if type(multiset) is int:
        n = multiset
        if m and m > n:
            return
        multiset = list(range(n))
        if m == 1:
            yield [multiset[:]]
            return

        # If m is not None, it can sometimes be faster to use
        # MultisetPartitionTraverser.enum_range() even for inputs
        # which are sets.  Since the _set_partitions code is quite
        # fast, this is only advantageous when the overall set
        # partitions outnumber those with the desired number of parts
        # by a large factor.  (At least 60.)  Such a switch is not
        # currently implemented.
        for nc, q in _set_partitions(n):
            if m is None or nc == m:
                rv = [[] for i in range(nc)]
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                yield rv
        return

    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]

    if not has_variety(multiset):
        # Only one component, repeated n times.  The resulting
        # partitions correspond to partitions of integer n.
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        x = multiset[:1]
        for size, p in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                for k in sorted(p):
                    rv.extend([x*k]*p[k])
                yield rv
    else:
        multiset = list(ordered(multiset))
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return

        # Split the information of the multiset into two lists -
        # one of the elements themselves, and one (of the same length)
        # giving the number of repeats for the corresponding element.
        elements, multiplicities = zip(*group(multiset, False))

        if len(elements) < len(multiset):
            # General case - multiset with more than one distinct element
            # and at least one element repeated more than once.
            if m:
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m-1, m):
                    yield list_visitor(state, elements)
            else:
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            # Set partitions case - no repeated elements. Pretty much
            # same as int argument case above, with same possible, but
            # currently unimplemented optimization for some cases when
            # m is not None
            for nc, q in _set_partitions(n):
                if m is None or nc == m:
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]
```
### 3 - sympy/utilities/iterables.py:

Start line: 1340, End line: 1393

```python
def multiset_combinations(m, n, g=None):
    """
    Return the unique combinations of size ``n`` from multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_combinations
    >>> from itertools import combinations
    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]
    ['abb', 'aby', 'bby']

    >>> def count(f, s): return len(list(f(s, 3)))

    The number of combinations depends on the number of letters; the
    number of unique combinations depends on how the letters are
    repeated.

    >>> s1 = 'abracadabra'
    >>> s2 = 'banana tree'
    >>> count(combinations, s1), count(multiset_combinations, s1)
    (165, 23)
    >>> count(combinations, s2), count(multiset_combinations, s2)
    (165, 54)

    """
    if g is None:
        if type(m) is dict:
            if n > sum(m.values()):
                return
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(m)
            if n > len(m):
                return
            try:
                m = multiset(m)
                g = [(k, m[k]) for k in ordered(m)]
            except TypeError:
                m = list(ordered(m))
                g = [list(i) for i in group(m, multiple=False)]
        del m
    if sum(v for k, v in g) < n or not n:
        yield []
    else:
        for i, (k, v) in enumerate(g):
            if v >= n:
                yield [k]*n
                v = n - 1
            for v in range(min(n, v), 0, -1):
                for j in multiset_combinations(None, n - v, g[i + 1:]):
                    rv = [k]*v + j
                    if len(rv) == n:
                        yield rv
```
### 4 - sympy/utilities/iterables.py:

Start line: 1562, End line: 1638

```python
def multiset_partitions(multiset, m=None):
    """
    Return unique partitions of the given multiset (in list form).
    If ``m`` is None, all multisets will be returned, otherwise only
    partitions with ``m`` parts will be returned.

    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]
    will be supplied.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_partitions
    >>> list(multiset_partitions([1, 2, 3, 4], 2))
    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
    [[1], [2, 3, 4]]]
    >>> list(multiset_partitions([1, 2, 3, 4], 1))
    [[[1, 2, 3, 4]]]

    Only unique partitions are returned and these will be returned in a
    canonical order regardless of the order of the input:

    >>> a = [1, 2, 2, 1]
    >>> ans = list(multiset_partitions(a, 2))
    >>> a.sort()
    >>> list(multiset_partitions(a, 2)) == ans
    True
    >>> a = range(3, 1, -1)
    >>> (list(multiset_partitions(a)) ==
    ...  list(multiset_partitions(sorted(a))))
    True

    If m is omitted then all partitions will be returned:

    >>> list(multiset_partitions([1, 1, 2]))
    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]
    >>> list(multiset_partitions([1]*3))
    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    Counting
    ========

    The number of partitions of a set is given by the bell number:

    >>> from sympy import bell
    >>> len(list(multiset_partitions(5))) == bell(5) == 52
    True

    The number of partitions of length k from a set of size n is given by the
    Stirling Number of the 2nd kind:

    >>> from sympy.functions.combinatorial.numbers import stirling
    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15
    True

    These comments on counting apply to *sets*, not multisets.

    Notes
    =====

    When all the elements are the same in the multiset, the order
    of the returned partitions is determined by the ``partitions``
    routine. If one is counting partitions then it is better to use
    the ``nT`` function.

    See Also
    ========

    partitions
    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition
    sympy.functions.combinatorial.numbers.nT

    """
    # This function looks at the supplied input and dispatches to
    # several special-case routines as they apply.
    # ... other code
```
### 5 - sympy/combinatorics/permutations.py:

Start line: 472, End line: 881

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

    Checking if a Permutation is contained in a Group
    =================================================

    Generally if you have a group of permutations G on n symbols, and
    you're checking if a permutation on less than n symbols is part
    of that group, the check will fail.

    Here is an example for n=5 and we check if the cycle
    (1,2,3) is in G:

    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> from sympy.combinatorics import Cycle, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> G = PermutationGroup(Cycle(2, 3)(4, 5), Cycle(1, 2, 3, 4, 5))
    >>> p1 = Permutation(Cycle(2, 5, 3))
    >>> p2 = Permutation(Cycle(1, 2, 3))
    >>> a1 = Permutation(Cycle(1, 2, 3).list(6))
    >>> a2 = Permutation(Cycle(1, 2, 3)(5))
    >>> a3 = Permutation(Cycle(1, 2, 3),size=6)
    >>> for p in [p1,p2,a1,a2,a3]: p, G.contains(p)
    ((2 5 3), True)
    ((1 2 3), False)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)

    The check for p2 above will fail.

    Checking if p1 is in G works because SymPy knows
    G is a group on 5 symbols, and p1 is also on 5 symbols
    (its largest element is 5).

    For ``a1``, the ``.list(6)`` call will extend the permutation to 5
    symbols, so the test will work as well. In the case of ``a2`` the
    permutation is being extended to 5 symbols by using a singleton,
    and in the case of ``a3`` it's extended through the constructor
    argument ``size=6``.

    There is another way to do this, which is to tell the ``contains``
    method that the number of symbols the group is on doesn't need to
    match perfectly the number of symbols for the permutation:

    >>> G.contains(p2,strict=False)
    True

    This can be via the ``strict`` argument to the ``contains`` method,
    and SymPy will try to extend the permutation on its own and then
    perform the containment check.

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
### 6 - sympy/utilities/enumerative.py:

Start line: 370, End line: 421

```python
class MultisetPartitionTraverser():
    """
    Has methods to ``enumerate`` and ``count`` the partitions of a multiset.

    This implements a refactored and extended version of Knuth's algorithm
    7.1.2.5M [AOCP]_."

    The enumeration methods of this class are generators and return
    data structures which can be interpreted by the same visitor
    functions used for the output of ``multiset_partitions_taocp``.

    Examples
    ========

    >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
    >>> m = MultisetPartitionTraverser()
    >>> m.count_partitions([4,4,4,2])
    127750
    >>> m.count_partitions([3,3,3])
    686

    See Also
    ========

    multiset_partitions_taocp
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [AOCP] Algorithm 7.1.2.5M in Volume 4A, Combinatoral Algorithms,
           Part 1, of The Art of Computer Programming, by Donald Knuth.

    .. [Factorisatio] On a Problem of Oppenheim concerning
           "Factorisatio Numerorum" E. R. Canfield, Paul Erdos, Carl
           Pomerance, JOURNAL OF NUMBER THEORY, Vol. 17, No. 1. August
           1983.  See section 7 for a description of an algorithm
           similar to Knuth's.

    .. [Yorgey] Generating Multiset Partitions, Brent Yorgey, The
           Monad.Reader, Issue 8, September 2007.

    """

    def __init__(self):
        self.debug = False
        # TRACING variables.  These are useful for gathering
        # statistics on the algorithm itself, but have no particular
        # benefit to a user of the code.
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
```
### 7 - sympy/combinatorics/perm_groups.py:

Start line: 1, End line: 4988

```python
from random import randrange, choice
from math import log
from sympy.ntheory import primefactors
from sympy import multiplicity, factorint, Symbol

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
from sympy.core.sympify import _sympify
rmul = Permutation.rmul_with_af
_af_new = Permutation._af_new


class PermutationGroup(Basic):
```
### 8 - sympy/utilities/iterables.py:

Start line: 1, End line: 18

```python
from collections import defaultdict, OrderedDict
from itertools import (
    combinations, combinations_with_replacement, permutations,
    product, product as cartes
)
import random
from operator import gt

from sympy.core import Basic

# this is the logical location of these functions
from sympy.core.compatibility import (as_int, is_sequence, iterable, ordered)
from sympy.core.compatibility import default_sort_key  # noqa: F401

import sympy

from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)
```
### 9 - sympy/utilities/iterables.py:

Start line: 2288, End line: 2317

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
    for p in multiset_permutations(perm):
        if not any(i == j for i, j in zip(perm, p)):
            yield p
```
### 10 - sympy/utilities/iterables.py:

Start line: 2595, End line: 2623

```python
def kbins(l, k, ordered=None):
    # ... other code

    if ordered is None:
        yield from partition(l, k)
    elif ordered == 11:
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from partition(pl, k)
    elif ordered == 00:
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        for kgot, p in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                for size, multiplicity in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i: j])
                        i = j
                yield rv
    else:
        raise ValueError(
            'ordered must be one of 00, 01, 10 or 11, not %s' % ordered)
```
### 11 - sympy/utilities/iterables.py:

Start line: 303, End line: 323

```python
def multiset(seq):
    """Return the hashable sequence in multiset form with values being the
    multiplicity of the item in the sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset
    >>> multiset('mississippi')
    {'i': 4, 'm': 1, 'p': 2, 's': 4}

    See Also
    ========

    group

    """
    rv = defaultdict(int)
    for s in seq:
        rv[s] += 1
    return dict(rv)
```
### 27 - sympy/utilities/iterables.py:

Start line: 1809, End line: 1847

```python
def partitions(n, m=None, k=None, size=False):
    # ... other code

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
            yield sum(ms.values()), ms.copy()
        else:
            yield ms.copy()
```
### 28 - sympy/utilities/iterables.py:

Start line: 1928, End line: 1983

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
### 42 - sympy/utilities/iterables.py:

Start line: 2144, End line: 2208

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
    # ... other code
```
### 45 - sympy/utilities/iterables.py:

Start line: 593, End line: 640

```python
def subsets(seq, k=None, repetition=False):
    r"""Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.

    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The
    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,
    whereas there are `2^n` subsets all together. If `k` is ``None`` then all
    `2^n` subsets will be returned from shortest to longest.

    Examples
    ========

    >>> from sympy.utilities.iterables import subsets

    ``subsets(seq, k)`` will return the `\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":

        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    ``subsets(seq, k, repetition=True)`` will return the `\frac{(n - 1 + k)!}{k!(n - 1)!}`
    combinations *with* repetition:

        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    """
    if k is None:
        for k in range(len(seq) + 1):
            yield from subsets(seq, k, repetition)
    else:
        if not repetition:
            yield from combinations(seq, k)
        else:
            yield from combinations_with_replacement(seq, k)
```
### 48 - sympy/utilities/iterables.py:

Start line: 2642, End line: 2658

```python
def signed_permutations(t):
    """Return iterator in which the signs of non-zero elements
    of t and the order of the elements are permuted.

    Examples
    ========

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),
    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),
    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),
    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),
    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]
    """
    return (type(t)(i) for j in permutations(t)
        for i in permute_signs(j))
```
### 70 - sympy/utilities/iterables.py:

Start line: 548, End line: 590

```python
def variations(seq, n, repetition=False):
    r"""Returns a generator of the n-sized variations of ``seq`` (size N).
    ``repetition`` controls whether items in ``seq`` can appear more than once;

    Examples
    ========

    ``variations(seq, n)`` will return `\frac{N!}{(N - n)!}` permutations without
    repetition of ``seq``'s elements:

        >>> from sympy.utilities.iterables import variations
        >>> list(variations([1, 2], 2))
        [(1, 2), (2, 1)]

    ``variations(seq, n, True)`` will return the `N^n` permutations obtained
    by allowing repetition of elements:

        >>> list(variations([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 1), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(variations([0, 1], 3, repetition=False))
        []
        >>> list(variations([0, 1], 3, repetition=True))[:4]
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    .. seealso::

       `itertools.permutations <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,
       `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    """
    if not repetition:
        seq = tuple(seq)
        if len(seq) < n:
            return
        yield from permutations(seq, n)
    else:
        if n == 0:
            yield ()
        else:
            yield from product(seq, repeat=n)
```
### 71 - sympy/utilities/iterables.py:

Start line: 2251, End line: 2285

```python
def generate_involutions(n):
    """
    Generates involutions.

    An involution is a permutation that when multiplied
    by itself equals the identity permutation. In this
    implementation the involutions are generated using
    Fixed Points.

    Alternatively, an involution can be considered as
    a permutation that does not contain any cycles with
    a length that is greater than two.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_involutions
    >>> list(generate_involutions(3))
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    >>> len(list(generate_involutions(4)))
    10

    References
    ==========

    .. [1] http://mathworld.wolfram.com/PermutationInvolution.html

    """
    idx = list(range(n))
    for p in permutations(idx):
        for i in idx:
            if p[p[i]] != i:
                break
        else:
            yield p
```
### 73 - sympy/utilities/iterables.py:

Start line: 1483, End line: 1559

```python
def _set_partitions(n):
    """Cycle through all partions of n elements, yielding the
    current number of partitions, ``m``, and a mutable list, ``q``
    such that element[i] is in part q[i] of the partition.

    NOTE: ``q`` is modified in place and generally should not be changed
    between function calls.

    Examples
    ========

    >>> from sympy.utilities.iterables import _set_partitions, _partition
    >>> for m, q in _set_partitions(3):
    ...     print('%s %s %s' % (m, q, _partition('abc', q, m)))
    1 [0, 0, 0] [['a', 'b', 'c']]
    2 [0, 0, 1] [['a', 'b'], ['c']]
    2 [0, 1, 0] [['a', 'c'], ['b']]
    2 [0, 1, 1] [['a'], ['b', 'c']]
    3 [0, 1, 2] [['a'], ['b'], ['c']]

    Notes
    =====

    This algorithm is similar to, and solves the same problem as,
    Algorithm 7.2.1.5H, from volume 4A of Knuth's The Art of Computer
    Programming.  Knuth uses the term "restricted growth string" where
    this code refers to a "partition vector". In each case, the meaning is
    the same: the value in the ith element of the vector specifies to
    which part the ith set element is to be assigned.

    At the lowest level, this code implements an n-digit big-endian
    counter (stored in the array q) which is incremented (with carries) to
    get the next partition in the sequence.  A special twist is that a
    digit is constrained to be at most one greater than the maximum of all
    the digits to the left of it.  The array p maintains this maximum, so
    that the code can efficiently decide when a digit can be incremented
    in place or whether it needs to be reset to 0 and trigger a carry to
    the next digit.  The enumeration starts with all the digits 0 (which
    corresponds to all the set elements being assigned to the same 0th
    part), and ends with 0123...n, which corresponds to each set element
    being assigned to a different, singleton, part.

    This routine was rewritten to use 0-based lists while trying to
    preserve the beauty and efficiency of the original algorithm.

    References
    ==========

    .. [1] Nijenhuis, Albert and Wilf, Herbert. (1978) Combinatorial Algorithms,
        2nd Ed, p 91, algorithm "nexequ". Available online from
        https://www.math.upenn.edu/~wilf/website/CombAlgDownld.html (viewed
        November 17, 2012).

    """
    p = [0]*n
    q = [0]*n
    nc = 1
    yield nc, q
    while nc != n:
        m = n
        while 1:
            m -= 1
            i = q[m]
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        nc += m - n
        p[0] += n - m
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield nc, q
```
