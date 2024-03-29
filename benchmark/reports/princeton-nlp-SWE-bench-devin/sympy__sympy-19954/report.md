# sympy__sympy-19954

| **sympy/sympy** | `6f54459aa0248bf1467ad12ee6333d8bc924a642` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7237 |
| **Any found context length** | 7237 |
| **Avg pos** | 13.0 |
| **Min pos** | 4 |
| **Max pos** | 9 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/combinatorics/perm_groups.py b/sympy/combinatorics/perm_groups.py
--- a/sympy/combinatorics/perm_groups.py
+++ b/sympy/combinatorics/perm_groups.py
@@ -2194,18 +2194,19 @@ def _number_blocks(blocks):
                 # check if the system is minimal with
                 # respect to the already discovere ones
                 minimal = True
-                to_remove = []
+                blocks_remove_mask = [False] * len(blocks)
                 for i, r in enumerate(rep_blocks):
                     if len(r) > len(rep) and rep.issubset(r):
                         # i-th block system is not minimal
-                        del num_blocks[i], blocks[i]
-                        to_remove.append(rep_blocks[i])
+                        blocks_remove_mask[i] = True
                     elif len(r) < len(rep) and r.issubset(rep):
                         # the system being checked is not minimal
                         minimal = False
                         break
                 # remove non-minimal representative blocks
-                rep_blocks = [r for r in rep_blocks if r not in to_remove]
+                blocks = [b for i, b in enumerate(blocks) if not blocks_remove_mask[i]]
+                num_blocks = [n for i, n in enumerate(num_blocks) if not blocks_remove_mask[i]]
+                rep_blocks = [r for i, r in enumerate(rep_blocks) if not blocks_remove_mask[i]]
 
                 if minimal and num_block not in num_blocks:
                     blocks.append(block)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/combinatorics/perm_groups.py | 2197 | 2208 | 9 | 1 | 14986


## Problem Statement

```
sylow_subgroup() IndexError 
I use sympy 1.6.1, with numpy 1.18.5, scipy 1.4.1, under Python '3.8.5 (default, Aug  5 2020, 09:44:06) [MSC v.1916 64 bit (AMD64)]'. 

The code that I run as the following gives IndexError for sylow_subgroup():

from sympy.combinatorics import DihedralGroup, PermutationGroup, Permutation

G = DihedralGroup(18)

S2 = G.sylow_subgroup(p=2)
 
Traceback (most recent call last):
  File "<input>", line 7, in <module>
  File "D:\anaconda38\envs\default\lib\site-packages\sympy\combinatorics\perm_groups.py", line 4370, in sylow_subgroup
    blocks = self.minimal_blocks()
  File "D:\anaconda38\envs\default\lib\site-packages\sympy\combinatorics\perm_groups.py", line 2207, in minimal_blocks
    del num_blocks[i], blocks[i]
IndexError: list assignment index out of range

The same error shows up as well when I set: 
G = DihedralGroup(2*25)

S2 = G.sylow_subgroup(p=2)



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/combinatorics/perm_groups.py** | 3702 | 4354| 6250 | 6250 | 45282 | 
| 2 | 2 sympy/combinatorics/__init__.py | 1 | 41| 407 | 6657 | 45689 | 
| 3 | **2 sympy/combinatorics/perm_groups.py** | 5173 | 5209| 330 | 6987 | 45689 | 
| **-> 4 <-** | **2 sympy/combinatorics/perm_groups.py** | 1 | 4849| 250 | 7237 | 45689 | 
| 5 | 3 sympy/combinatorics/fp_groups.py | 690 | 716| 251 | 7488 | 57842 | 
| 6 | **3 sympy/combinatorics/perm_groups.py** | 1406 | 1499| 811 | 8299 | 57842 | 
| 7 | **3 sympy/combinatorics/perm_groups.py** | 5062 | 5094| 218 | 8517 | 57842 | 
| 8 | 3 sympy/combinatorics/fp_groups.py | 279 | 325| 424 | 8941 | 57842 | 
| **-> 9 <-** | **3 sympy/combinatorics/perm_groups.py** | 1501 | 2252| 6045 | 14986 | 57842 | 
| 10 | 4 sympy/combinatorics/pc_groups.py | 614 | 656| 346 | 15332 | 63129 | 
| 11 | 5 sympy/combinatorics/named_groups.py | 165 | 226| 664 | 15996 | 65806 | 
| 12 | 5 sympy/combinatorics/fp_groups.py | 719 | 777| 629 | 16625 | 65806 | 
| 13 | 5 sympy/combinatorics/fp_groups.py | 553 | 567| 135 | 16760 | 65806 | 
| 14 | 5 sympy/combinatorics/fp_groups.py | 569 | 688| 1067 | 17827 | 65806 | 
| 15 | 6 sympy/combinatorics/polyhedron.py | 759 | 805| 694 | 18521 | 78684 | 
| 16 | 6 sympy/combinatorics/fp_groups.py | 327 | 362| 311 | 18832 | 78684 | 
| 17 | 6 sympy/combinatorics/named_groups.py | 229 | 304| 815 | 19647 | 78684 | 
| 18 | **6 sympy/combinatorics/perm_groups.py** | 694 | 1404| 6059 | 25706 | 78684 | 
| 19 | 7 sympy/combinatorics/homomorphisms.py | 499 | 552| 437 | 26143 | 83226 | 
| 20 | 8 sympy/combinatorics/free_groups.py | 182 | 216| 237 | 26380 | 93330 | 
| 21 | 8 sympy/combinatorics/free_groups.py | 1000 | 1013| 156 | 26536 | 93330 | 
| 22 | 8 sympy/combinatorics/polyhedron.py | 1 | 393| 230 | 26766 | 93330 | 
| 23 | 8 sympy/combinatorics/free_groups.py | 139 | 162| 207 | 26973 | 93330 | 
| 24 | 8 sympy/combinatorics/homomorphisms.py | 388 | 422| 320 | 27293 | 93330 | 
| 25 | 8 sympy/combinatorics/fp_groups.py | 780 | 805| 309 | 27602 | 93330 | 
| 26 | 8 sympy/combinatorics/free_groups.py | 278 | 311| 226 | 27828 | 93330 | 
| 27 | **8 sympy/combinatorics/perm_groups.py** | 4355 | 4849| 4299 | 32127 | 93330 | 
| 28 | 8 sympy/combinatorics/fp_groups.py | 421 | 550| 733 | 32860 | 93330 | 
| 29 | 9 sympy/benchmarks/bench_discrete_log.py | 1 | 48| 568 | 33428 | 94184 | 
| 30 | **9 sympy/combinatorics/perm_groups.py** | 5211 | 5228| 135 | 33563 | 94184 | 
| 31 | 9 sympy/combinatorics/fp_groups.py | 364 | 400| 316 | 33879 | 94184 | 
| 32 | 9 sympy/combinatorics/fp_groups.py | 808 | 828| 210 | 34089 | 94184 | 
| 33 | 9 sympy/combinatorics/polyhedron.py | 725 | 757| 613 | 34702 | 94184 | 
| 34 | **9 sympy/combinatorics/perm_groups.py** | 5096 | 5144| 266 | 34968 | 94184 | 
| 35 | 9 sympy/combinatorics/fp_groups.py | 1126 | 1180| 550 | 35518 | 94184 | 
| 36 | 9 sympy/combinatorics/free_groups.py | 536 | 555| 150 | 35668 | 94184 | 
| 37 | 9 sympy/combinatorics/polyhedron.py | 696 | 724| 397 | 36065 | 94184 | 
| 38 | 9 sympy/combinatorics/polyhedron.py | 823 | 869| 616 | 36681 | 94184 | 
| 39 | 9 sympy/combinatorics/polyhedron.py | 807 | 821| 316 | 36997 | 94184 | 
| 40 | 9 sympy/combinatorics/named_groups.py | 124 | 162| 356 | 37353 | 94184 | 
| 41 | **9 sympy/combinatorics/perm_groups.py** | 2254 | 2940| 6016 | 43369 | 94184 | 
| 42 | 10 sympy/combinatorics/coset_table.py | 966 | 1118| 1810 | 45179 | 105684 | 
| 43 | 10 sympy/combinatorics/fp_groups.py | 253 | 276| 215 | 45394 | 105684 | 
| 44 | 10 sympy/combinatorics/fp_groups.py | 1063 | 1123| 701 | 46095 | 105684 | 
| 45 | 10 sympy/combinatorics/free_groups.py | 218 | 235| 140 | 46235 | 105684 | 
| 46 | 10 sympy/combinatorics/homomorphisms.py | 424 | 497| 738 | 46973 | 105684 | 
| 47 | **10 sympy/combinatorics/perm_groups.py** | 5230 | 5263| 222 | 47195 | 105684 | 
| 48 | 11 sympy/combinatorics/tensor_can.py | 936 | 960| 246 | 47441 | 118715 | 
| 49 | 12 sympy/combinatorics/util.py | 373 | 446| 661 | 48102 | 123222 | 
| 50 | 12 sympy/combinatorics/fp_groups.py | 120 | 156| 411 | 48513 | 123222 | 
| 51 | 12 sympy/combinatorics/pc_groups.py | 317 | 365| 467 | 48980 | 123222 | 
| 52 | 12 sympy/combinatorics/pc_groups.py | 446 | 483| 305 | 49285 | 123222 | 
| 53 | 12 sympy/combinatorics/pc_groups.py | 82 | 134| 468 | 49753 | 123222 | 
| 54 | 12 sympy/combinatorics/tensor_can.py | 424 | 535| 1356 | 51109 | 123222 | 
| 55 | 13 sympy/matrices/matrices.py | 1 | 61| 660 | 51769 | 142170 | 
| 56 | 14 sympy/codegen/array_utils.py | 1 | 16| 159 | 51928 | 155444 | 
| 57 | 14 sympy/combinatorics/homomorphisms.py | 105 | 136| 225 | 52153 | 155444 | 
| 58 | 14 sympy/combinatorics/pc_groups.py | 1 | 39| 262 | 52415 | 155444 | 
| 59 | **14 sympy/combinatorics/perm_groups.py** | 4925 | 4952| 221 | 52636 | 155444 | 
| 60 | 14 sympy/combinatorics/fp_groups.py | 402 | 419| 180 | 52816 | 155444 | 
| 61 | 14 sympy/combinatorics/fp_groups.py | 897 | 944| 434 | 53250 | 155444 | 
| 62 | 14 sympy/combinatorics/named_groups.py | 53 | 121| 535 | 53785 | 155444 | 
| 63 | 14 sympy/combinatorics/fp_groups.py | 217 | 251| 322 | 54107 | 155444 | 
| 64 | 15 sympy/polys/groebnertools.py | 636 | 697| 499 | 54606 | 162145 | 
| 65 | 15 sympy/combinatorics/polyhedron.py | 871 | 943| 361 | 54967 | 162145 | 
| 66 | 16 sympy/combinatorics/permutations.py | 464 | 823| 3434 | 58401 | 185771 | 
| 67 | 16 sympy/combinatorics/pc_groups.py | 658 | 675| 139 | 58540 | 185771 | 
| 68 | 17 sympy/combinatorics/generators.py | 62 | 95| 300 | 58840 | 188173 | 
| 69 | 17 sympy/combinatorics/homomorphisms.py | 1 | 26| 200 | 59040 | 188173 | 
| 70 | 17 sympy/combinatorics/fp_groups.py | 1251 | 1290| 516 | 59556 | 188173 | 
| 71 | 17 sympy/combinatorics/free_groups.py | 939 | 968| 211 | 59767 | 188173 | 
| 72 | 17 sympy/combinatorics/generators.py | 233 | 303| 432 | 60199 | 188173 | 
| 73 | 17 sympy/combinatorics/free_groups.py | 468 | 506| 289 | 60488 | 188173 | 
| 74 | 17 sympy/combinatorics/free_groups.py | 313 | 337| 162 | 60650 | 188173 | 
| 75 | 17 sympy/combinatorics/free_groups.py | 112 | 137| 164 | 60814 | 188173 | 
| 76 | 17 sympy/combinatorics/pc_groups.py | 368 | 444| 656 | 61470 | 188173 | 
| 77 | 17 sympy/combinatorics/free_groups.py | 970 | 998| 183 | 61653 | 188173 | 
| 78 | 17 sympy/combinatorics/tensor_can.py | 165 | 401| 3163 | 64816 | 188173 | 


## Patch

```diff
diff --git a/sympy/combinatorics/perm_groups.py b/sympy/combinatorics/perm_groups.py
--- a/sympy/combinatorics/perm_groups.py
+++ b/sympy/combinatorics/perm_groups.py
@@ -2194,18 +2194,19 @@ def _number_blocks(blocks):
                 # check if the system is minimal with
                 # respect to the already discovere ones
                 minimal = True
-                to_remove = []
+                blocks_remove_mask = [False] * len(blocks)
                 for i, r in enumerate(rep_blocks):
                     if len(r) > len(rep) and rep.issubset(r):
                         # i-th block system is not minimal
-                        del num_blocks[i], blocks[i]
-                        to_remove.append(rep_blocks[i])
+                        blocks_remove_mask[i] = True
                     elif len(r) < len(rep) and r.issubset(rep):
                         # the system being checked is not minimal
                         minimal = False
                         break
                 # remove non-minimal representative blocks
-                rep_blocks = [r for r in rep_blocks if r not in to_remove]
+                blocks = [b for i, b in enumerate(blocks) if not blocks_remove_mask[i]]
+                num_blocks = [n for i, n in enumerate(num_blocks) if not blocks_remove_mask[i]]
+                rep_blocks = [r for i, r in enumerate(rep_blocks) if not blocks_remove_mask[i]]
 
                 if minimal and num_block not in num_blocks:
                     blocks.append(block)

```

## Test Patch

```diff
diff --git a/sympy/combinatorics/tests/test_perm_groups.py b/sympy/combinatorics/tests/test_perm_groups.py
--- a/sympy/combinatorics/tests/test_perm_groups.py
+++ b/sympy/combinatorics/tests/test_perm_groups.py
@@ -905,6 +905,14 @@ def test_sylow_subgroup():
     assert G.order() % S.order() == 0
     assert G.order()/S.order() % 2 > 0
 
+    G = DihedralGroup(18)
+    S = G.sylow_subgroup(p=2)
+    assert S.order() == 4
+
+    G = DihedralGroup(50)
+    S = G.sylow_subgroup(p=2)
+    assert S.order() == 4
+
 
 @slow
 def test_presentation():

```


## Code snippets

### 1 - sympy/combinatorics/perm_groups.py:

Start line: 3702, End line: 4354

```python
class PermutationGroup(Basic):

    def schreier_vector(self, alpha):
        """Computes the schreier vector for ``alpha``.

        The Schreier vector efficiently stores information
        about the orbit of ``alpha``. It can later be used to quickly obtain
        elements of the group that send ``alpha`` to a particular element
        in the orbit. Notice that the Schreier vector depends on the order
        in which the group generators are listed. For a definition, see [3].
        Since list indices start from zero, we adopt the convention to use
        "None" instead of 0 to signify that an element doesn't belong
        to the orbit.
        For the algorithm and its correctness, see [2], pp.78-80.

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> a = Permutation([2, 4, 6, 3, 1, 5, 0])
        >>> b = Permutation([0, 1, 3, 5, 4, 6, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_vector(0)
        [-1, None, 0, 1, None, 1, 0]

        See Also
        ========

        orbit

        """
        n = self.degree
        v = [None]*n
        v[alpha] = -1
        orb = [alpha]
        used = [False]*n
        used[alpha] = True
        gens = self.generators
        r = len(gens)
        for b in orb:
            for i in range(r):
                temp = gens[i]._array_form[b]
                if used[temp] is False:
                    orb.append(temp)
                    used[temp] = True
                    v[temp] = i
        return v

    def stabilizer(self, alpha):
        r"""Return the stabilizer subgroup of ``alpha``.

        The stabilizer of `\alpha` is the group `G_\alpha =
        \{g \in G | g(\alpha) = \alpha\}`.
        For a proof of correctness, see [1], p.79.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.stabilizer(5)
        PermutationGroup([
            (5)(0 4)(1 3)])

        See Also
        ========

        orbit

        """
        return PermGroup(_stabilizer(self._degree, self._generators, alpha))

    @property
    def strong_gens(self):
        r"""Return a strong generating set from the Schreier-Sims algorithm.

        A generating set `S = \{g_1, g_2, ..., g_t\}` for a permutation group
        `G` is a strong generating set relative to the sequence of points
        (referred to as a "base") `(b_1, b_2, ..., b_k)` if, for
        `1 \leq i \leq k` we have that the intersection of the pointwise
        stabilizer `G^{(i+1)} := G_{b_1, b_2, ..., b_i}` with `S` generates
        the pointwise stabilizer `G^{(i+1)}`. The concepts of a base and
        strong generating set and their applications are discussed in depth
        in [1], pp. 87-89 and [2], pp. 55-57.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> D.strong_gens
        [(0 1 2 3), (0 3)(1 2), (1 3)]
        >>> D.base
        [0, 1]

        See Also
        ========

        base, basic_transversals, basic_orbits, basic_stabilizers

        """
        if self._strong_gens == []:
            self.schreier_sims()
        return self._strong_gens

    def subgroup(self, gens):
        """
           Return the subgroup generated by `gens` which is a list of
           elements of the group
        """

        if not all([g in self for g in gens]):
            raise ValueError("The group doesn't contain the supplied generators")

        G = PermutationGroup(gens)
        return G

    def subgroup_search(self, prop, base=None, strong_gens=None, tests=None,
                        init_subgroup=None):
        """Find the subgroup of all elements satisfying the property ``prop``.

        This is done by a depth-first search with respect to base images that
        uses several tests to prune the search tree.

        Parameters
        ==========

        prop
            The property to be used. Has to be callable on group elements
            and always return ``True`` or ``False``. It is assumed that
            all group elements satisfying ``prop`` indeed form a subgroup.
        base
            A base for the supergroup.
        strong_gens
            A strong generating set for the supergroup.
        tests
            A list of callables of length equal to the length of ``base``.
            These are used to rule out group elements by partial base images,
            so that ``tests[l](g)`` returns False if the element ``g`` is known
            not to satisfy prop base on where g sends the first ``l + 1`` base
            points.
        init_subgroup
            if a subgroup of the sought group is
            known in advance, it can be passed to the function as this
            parameter.

        Returns
        =======

        res
            The subgroup of all elements satisfying ``prop``. The generating
            set for this group is guaranteed to be a strong generating set
            relative to the base ``base``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> S = SymmetricGroup(7)
        >>> prop_even = lambda x: x.is_even
        >>> base, strong_gens = S.schreier_sims_incremental()
        >>> G = S.subgroup_search(prop_even, base=base, strong_gens=strong_gens)
        >>> G.is_subgroup(AlternatingGroup(7))
        True
        >>> _verify_bsgs(G, base, G.generators)
        True

        Notes
        =====

        This function is extremely lengthy and complicated and will require
        some careful attention. The implementation is described in
        [1], pp. 114-117, and the comments for the code here follow the lines
        of the pseudocode in the book for clarity.

        The complexity is exponential in general, since the search process by
        itself visits all members of the supergroup. However, there are a lot
        of tests which are used to prune the search tree, and users can define
        their own tests via the ``tests`` parameter, so in practice, and for
        some computations, it's not terrible.

        A crucial part in the procedure is the frequent base change performed
        (this is line 11 in the pseudocode) in order to obtain a new basic
        stabilizer. The book mentiones that this can be done by using
        ``.baseswap(...)``, however the current implementation uses a more
        straightforward way to find the next basic stabilizer - calling the
        function ``.stabilizer(...)`` on the previous basic stabilizer.

        """
        # initialize BSGS and basic group properties
        def get_reps(orbits):
            # get the minimal element in the base ordering
            return [min(orbit, key = lambda x: base_ordering[x]) \
              for orbit in orbits]

        def update_nu(l):
            temp_index = len(basic_orbits[l]) + 1 -\
                         len(res_basic_orbits_init_base[l])
            # this corresponds to the element larger than all points
            if temp_index >= len(sorted_orbits[l]):
                nu[l] = base_ordering[degree]
            else:
                nu[l] = sorted_orbits[l][temp_index]

        if base is None:
            base, strong_gens = self.schreier_sims_incremental()
        base_len = len(base)
        degree = self.degree
        identity = _af_new(list(range(degree)))
        base_ordering = _base_ordering(base, degree)
        # add an element larger than all points
        base_ordering.append(degree)
        # add an element smaller than all points
        base_ordering.append(-1)
        # compute BSGS-related structures
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_orbits, transversals = _orbits_transversals_from_bsgs(base,
                                     strong_gens_distr)
        # handle subgroup initialization and tests
        if init_subgroup is None:
            init_subgroup = PermutationGroup([identity])
        if tests is None:
            trivial_test = lambda x: True
            tests = []
            for i in range(base_len):
                tests.append(trivial_test)
        # line 1: more initializations.
        res = init_subgroup
        f = base_len - 1
        l = base_len - 1
        # line 2: set the base for K to the base for G
        res_base = base[:]
        # line 3: compute BSGS and related structures for K
        res_base, res_strong_gens = res.schreier_sims_incremental(
            base=res_base)
        res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                res_strong_gens)
        res_generators = res.generators
        res_basic_orbits_init_base = \
        [_orbit(degree, res_strong_gens_distr[i], res_base[i])\
         for i in range(base_len)]
        # initialize orbit representatives
        orbit_reps = [None]*base_len
        # line 4: orbit representatives for f-th basic stabilizer of K
        orbits = _orbits(degree, res_strong_gens_distr[f])
        orbit_reps[f] = get_reps(orbits)
        # line 5: remove the base point from the representatives to avoid
        # getting the identity element as a generator for K
        orbit_reps[f].remove(base[f])
        # line 6: more initializations
        c = [0]*base_len
        u = [identity]*base_len
        sorted_orbits = [None]*base_len
        for i in range(base_len):
            sorted_orbits[i] = basic_orbits[i][:]
            sorted_orbits[i].sort(key=lambda point: base_ordering[point])
        # line 7: initializations
        mu = [None]*base_len
        nu = [None]*base_len
        # this corresponds to the element smaller than all points
        mu[l] = degree + 1
        update_nu(l)
        # initialize computed words
        computed_words = [identity]*base_len
        # line 8: main loop
        while True:
            # apply all the tests
            while l < base_len - 1 and \
                computed_words[l](base[l]) in orbit_reps[l] and \
                base_ordering[mu[l]] < \
                base_ordering[computed_words[l](base[l])] < \
                base_ordering[nu[l]] and \
                    tests[l](computed_words):
                # line 11: change the (partial) base of K
                new_point = computed_words[l](base[l])
                res_base[l] = new_point
                new_stab_gens = _stabilizer(degree, res_strong_gens_distr[l],
                        new_point)
                res_strong_gens_distr[l + 1] = new_stab_gens
                # line 12: calculate minimal orbit representatives for the
                # l+1-th basic stabilizer
                orbits = _orbits(degree, new_stab_gens)
                orbit_reps[l + 1] = get_reps(orbits)
                # line 13: amend sorted orbits
                l += 1
                temp_orbit = [computed_words[l - 1](point) for point
                             in basic_orbits[l]]
                temp_orbit.sort(key=lambda point: base_ordering[point])
                sorted_orbits[l] = temp_orbit
                # lines 14 and 15: update variables used minimality tests
                new_mu = degree + 1
                for i in range(l):
                    if base[l] in res_basic_orbits_init_base[i]:
                        candidate = computed_words[i](base[i])
                        if base_ordering[candidate] > base_ordering[new_mu]:
                            new_mu = candidate
                mu[l] = new_mu
                update_nu(l)
                # line 16: determine the new transversal element
                c[l] = 0
                temp_point = sorted_orbits[l][c[l]]
                gamma = computed_words[l - 1]._array_form.index(temp_point)
                u[l] = transversals[l][gamma]
                # update computed words
                computed_words[l] = rmul(computed_words[l - 1], u[l])
            # lines 17 & 18: apply the tests to the group element found
            g = computed_words[l]
            temp_point = g(base[l])
            if l == base_len - 1 and \
                base_ordering[mu[l]] < \
                base_ordering[temp_point] < base_ordering[nu[l]] and \
                temp_point in orbit_reps[l] and \
                tests[l](computed_words) and \
                    prop(g):
                # line 19: reset the base of K
                res_generators.append(g)
                res_base = base[:]
                # line 20: recalculate basic orbits (and transversals)
                res_strong_gens.append(g)
                res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                                          res_strong_gens)
                res_basic_orbits_init_base = \
                [_orbit(degree, res_strong_gens_distr[i], res_base[i]) \
                 for i in range(base_len)]
                # line 21: recalculate orbit representatives
                # line 22: reset the search depth
                orbit_reps[f] = get_reps(orbits)
                l = f
            # line 23: go up the tree until in the first branch not fully
            # searched
            while l >= 0 and c[l] == len(basic_orbits[l]) - 1:
                l = l - 1
            # line 24: if the entire tree is traversed, return K
            if l == -1:
                return PermutationGroup(res_generators)
            # lines 25-27: update orbit representatives
            if l < f:
                # line 26
                f = l
                c[l] = 0
                # line 27
                temp_orbits = _orbits(degree, res_strong_gens_distr[f])
                orbit_reps[f] = get_reps(temp_orbits)
                # line 28: update variables used for minimality testing
                mu[l] = degree + 1
                temp_index = len(basic_orbits[l]) + 1 - \
                    len(res_basic_orbits_init_base[l])
                if temp_index >= len(sorted_orbits[l]):
                    nu[l] = base_ordering[degree]
                else:
                    nu[l] = sorted_orbits[l][temp_index]
            # line 29: set the next element from the current branch and update
            # accordingly
            c[l] += 1
            if l == 0:
                gamma  = sorted_orbits[l][c[l]]
            else:
                gamma = computed_words[l - 1]._array_form.index(sorted_orbits[l][c[l]])

            u[l] = transversals[l][gamma]
            if l == 0:
                computed_words[l] = u[l]
            else:
                computed_words[l] = rmul(computed_words[l - 1], u[l])

    @property
    def transitivity_degree(self):
        r"""Compute the degree of transitivity of the group.

        A permutation group `G` acting on `\Omega = \{0, 1, ..., n-1\}` is
        ``k``-fold transitive, if, for any k points
        `(a_1, a_2, ..., a_k)\in\Omega` and any k points
        `(b_1, b_2, ..., b_k)\in\Omega` there exists `g\in G` such that
        `g(a_1)=b_1, g(a_2)=b_2, ..., g(a_k)=b_k`
        The degree of transitivity of `G` is the maximum ``k`` such that
        `G` is ``k``-fold transitive. ([8])

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.transitivity_degree
        3

        See Also
        ========

        is_transitive, orbit

        """
        if self._transitivity_degree is None:
            n = self.degree
            G = self
            # if G is k-transitive, a tuple (a_0,..,a_k)
            # can be brought to (b_0,...,b_(k-1), b_k)
            # where b_0,...,b_(k-1) are fixed points;
            # consider the group G_k which stabilizes b_0,...,b_(k-1)
            # if G_k is transitive on the subset excluding b_0,...,b_(k-1)
            # then G is (k+1)-transitive
            for i in range(n):
                orb = G.orbit(i)
                if len(orb) != n - i:
                    self._transitivity_degree = i
                    return i
                G = G.stabilizer(i)
            self._transitivity_degree = n
            return n
        else:
            return self._transitivity_degree

    def _p_elements_group(G, p):
        '''
        For an abelian p-group G return the subgroup consisting of
        all elements of order p (and the identity)

        '''
        gens = G.generators[:]
        gens = sorted(gens, key=lambda x: x.order(), reverse=True)
        gens_p = [g**(g.order()/p) for g in gens]
        gens_r = []
        for i in range(len(gens)):
            x = gens[i]
            x_order = x.order()
            # x_p has order p
            x_p = x**(x_order/p)
            if i > 0:
                P = PermutationGroup(gens_p[:i])
            else:
                P = PermutationGroup(G.identity)
            if x**(x_order/p) not in P:
                gens_r.append(x**(x_order/p))
            else:
                # replace x by an element of order (x.order()/p)
                # so that gens still generates G
                g = P.generator_product(x_p, original=True)
                for s in g:
                    x = x*s**-1
                x_order = x_order/p
                # insert x to gens so that the sorting is preserved
                del gens[i]
                del gens_p[i]
                j = i - 1
                while j < len(gens) and gens[j].order() >= x_order:
                    j += 1
                gens = gens[:j] + [x] + gens[j:]
                gens_p = gens_p[:j] + [x] + gens_p[j:]
        return PermutationGroup(gens_r)

    def _sylow_alt_sym(self, p):
        '''
        Return a p-Sylow subgroup of a symmetric or an
        alternating group.

        The algorithm for this is hinted at in [1], Chapter 4,
        Exercise 4.

        For Sym(n) with n = p^i, the idea is as follows. Partition
        the interval [0..n-1] into p equal parts, each of length p^(i-1):
        [0..p^(i-1)-1], [p^(i-1)..2*p^(i-1)-1]...[(p-1)*p^(i-1)..p^i-1].
        Find a p-Sylow subgroup of Sym(p^(i-1)) (treated as a subgroup
        of ``self``) acting on each of the parts. Call the subgroups
        P_1, P_2...P_p. The generators for the subgroups P_2...P_p
        can be obtained from those of P_1 by applying a "shifting"
        permutation to them, that is, a permutation mapping [0..p^(i-1)-1]
        to the second part (the other parts are obtained by using the shift
        multiple times). The union of this permutation and the generators
        of P_1 is a p-Sylow subgroup of ``self``.

        For n not equal to a power of p, partition
        [0..n-1] in accordance with how n would be written in base p.
        E.g. for p=2 and n=11, 11 = 2^3 + 2^2 + 1 so the partition
        is [[0..7], [8..9], {10}]. To generate a p-Sylow subgroup,
        take the union of the generators for each of the parts.
        For the above example, {(0 1), (0 2)(1 3), (0 4), (1 5)(2 7)}
        from the first part, {(8 9)} from the second part and
        nothing from the third. This gives 4 generators in total, and
        the subgroup they generate is p-Sylow.

        Alternating groups are treated the same except when p=2. In this
        case, (0 1)(s s+1) should be added for an appropriate s (the start
        of a part) for each part in the partitions.

        See Also
        ========

        sylow_subgroup, is_alt_sym

        '''
        n = self.degree
        gens = []
        identity = Permutation(n-1)
        # the case of 2-sylow subgroups of alternating groups
        # needs special treatment
        alt = p == 2 and all(g.is_even for g in self.generators)

        # find the presentation of n in base p
        coeffs = []
        m = n
        while m > 0:
            coeffs.append(m % p)
            m = m // p

        power = len(coeffs)-1
        # for a symmetric group, gens[:i] is the generating
        # set for a p-Sylow subgroup on [0..p**(i-1)-1]. For
        # alternating groups, the same is given by gens[:2*(i-1)]
        for i in range(1, power+1):
            if i == 1 and alt:
                # (0 1) shouldn't be added for alternating groups
                continue
            gen = Permutation([(j + p**(i-1)) % p**i for j in range(p**i)])
            gens.append(identity*gen)
            if alt:
                gen = Permutation(0, 1)*gen*Permutation(0, 1)*gen
                gens.append(gen)

        # the first point in the current part (see the algorithm
        # description in the docstring)
        start = 0

        while power > 0:
            a = coeffs[power]

            # make the permutation shifting the start of the first
            # part ([0..p^i-1] for some i) to the current one
            for s in range(a):
                shift = Permutation()
                if start > 0:
                    for i in range(p**power):
                        shift = shift(i, start + i)

                    if alt:
                        gen = Permutation(0, 1)*shift*Permutation(0, 1)*shift
                        gens.append(gen)
                        j = 2*(power - 1)
                    else:
                        j = power

                    for i, gen in enumerate(gens[:j]):
                        if alt and i % 2 == 1:
                            continue
                        # shift the generator to the start of the
                        # partition part
                        gen = shift*gen*shift
                        gens.append(gen)

                start += p**power
            power = power-1

        return gens

    def sylow_subgroup(self, p):
        '''
        Return a p-Sylow subgroup of the group.

        The algorithm is described in [1], Chapter 4, Section 7

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import AlternatingGroup

        >>> D = DihedralGroup(6)
        >>> S = D.sylow_subgroup(2)
        >>> S.order()
        4
        >>> G = SymmetricGroup(6)
        >>> S = G.sylow_subgroup(5)
        >>> S.order()
        5

        >>> G1 = AlternatingGroup(3)
        >>> G2 = AlternatingGroup(5)
        >>> G3 = AlternatingGroup(9)

        >>> S1 = G1.sylow_subgroup(3)
        >>> S2 = G2.sylow_subgroup(3)
        >>> S3 = G3.sylow_subgroup(3)

        >>> len1 = len(S1.lower_central_series())
        >>> len2 = len(S2.lower_central_series())
        >>> len3 = len(S3.lower_central_series())

        >>> len1 == len2
        True
        >>> len1 < len3
        True

        '''
        from sympy.combinatorics.homomorphisms import (
                orbit_homomorphism, block_homomorphism)
        from sympy.ntheory.primetest import isprime

        if not isprime(p):
            raise ValueError("p must be a prime")

        def is_p_group(G):
            # check if the order of G is a power of p
            # and return the power
            m = G.order()
            n = 0
            while m % p == 0:
                m = m/p
                n += 1
                if m == 1:
                    return True, n
            return False, n

        def _sylow_reduce(mu, nu):
            # reduction based on two homomorphisms
            # mu and nu with trivially intersecting
            # kernels
            Q = mu.image().sylow_subgroup(p)
            Q = mu.invert_subgroup(Q)
            nu = nu.restrict_to(Q)
            R = nu.image().sylow_subgroup(p)
            return nu.invert_subgroup(R)

        order = self.order()
        if order % p != 0:
            return PermutationGroup([self.identity])
        p_group, n = is_p_group(self)
        if p_group:
            return self

        if self.is_alt_sym():
            return PermutationGroup(self._sylow_alt_sym(p))

        # if there is a non-trivial orbit with size not divisible
        # by p, the sylow subgroup is contained in its stabilizer
        # (by orbit-stabilizer theorem)
        orbits = self.orbits()
        non_p_orbits = [o for o in orbits if len(o) % p != 0 and len(o) != 1]
        if non_p_orbits:
            G = self.stabilizer(list(non_p_orbits[0]).pop())
            return G.sylow_subgroup(p)

        if not self.is_transitive():
            # apply _sylow_reduce to orbit actions
            orbits = sorted(orbits, key = lambda x: len(x))
            omega1 = orbits.pop()
            omega2 = orbits[0].union(*orbits)
            mu = orbit_homomorphism(self, omega1)
            nu = orbit_homomorphism(self, omega2)
            return _sylow_reduce(mu, nu)

        blocks = self.minimal_blocks()
        # ... other code
```
### 2 - sympy/combinatorics/__init__.py:

Start line: 1, End line: 41

```python
from sympy.combinatorics.permutations import Permutation, Cycle
from sympy.combinatorics.prufer import Prufer
from sympy.combinatorics.generators import cyclic, alternating, symmetric, dihedral
from sympy.combinatorics.subsets import Subset
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
    RGS_rank, RGS_unrank, RGS_enum)
from sympy.combinatorics.polyhedron import (Polyhedron, tetrahedron, cube,
    octahedron, dodecahedron, icosahedron)
from sympy.combinatorics.perm_groups import PermutationGroup, Coset, SymmetricPermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.named_groups import (SymmetricGroup, DihedralGroup,
    CyclicGroup, AlternatingGroup, AbelianGroup, RubikGroup)
from sympy.combinatorics.pc_groups import PolycyclicGroup, Collector

__all__ = [
    'Permutation', 'Cycle',

    'Prufer',

    'cyclic', 'alternating', 'symmetric', 'dihedral',

    'Subset',

    'Partition', 'IntegerPartition', 'RGS_rank', 'RGS_unrank', 'RGS_enum',

    'Polyhedron', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron',
    'icosahedron',

    'PermutationGroup', 'Coset', 'SymmetricPermutationGroup',

    'DirectProduct',

    'GrayCode',

    'SymmetricGroup', 'DihedralGroup', 'CyclicGroup', 'AlternatingGroup',
    'AbelianGroup', 'RubikGroup',

    'PolycyclicGroup', 'Collector',
]
```
### 3 - sympy/combinatorics/perm_groups.py:

Start line: 5173, End line: 5209

```python
class Coset(Basic):

    def __new__(cls, g, H, G=None, dir="+"):
        g = _sympify(g)
        if not isinstance(g, Permutation):
            raise NotImplementedError

        H = _sympify(H)
        if not isinstance(H, PermutationGroup):
            raise NotImplementedError

        if G is not None:
            G = _sympify(G)
            if not isinstance(G, PermutationGroup) and not isinstance(G, SymmetricPermutationGroup):
                raise NotImplementedError
            if not H.is_subgroup(G):
                raise ValueError("{} must be a subgroup of {}.".format(H, G))
            if g not in G:
                raise ValueError("{} must be an element of {}.".format(g, G))
        else:
            g_size = g.size
            h_degree = H.degree
            if g_size != h_degree:
                raise ValueError(
                    "The size of the permutation {} and the degree of "
                    "the permutation group {} should be matching "
                    .format(g, H))
            G = SymmetricPermutationGroup(g.size)

        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("dir must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-'):
            raise ValueError("dir must be one of '+' or '-' not %s" % dir)
        obj = Basic.__new__(cls, g, H, G, dir)
        obj._dir = dir
        return obj
```
### 4 - sympy/combinatorics/perm_groups.py:

Start line: 1, End line: 4849

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
### 5 - sympy/combinatorics/fp_groups.py:

Start line: 690, End line: 716

```python
class FpSubgroup(DefaultPrinting):

    def order(self):
        from sympy import S
        if not self.generators:
            return 1
        if isinstance(self.parent, FreeGroup):
            return S.Infinity
        if self.C is None:
            C = self.parent.coset_enumeration(self.generators)
            self.C = C
        # This is valid because `len(self.C.table)` (the index of the subgroup)
        # will always be finite - otherwise coset enumeration doesn't terminate
        return self.parent.order()/len(self.C.table)

    def to_FpGroup(self):
        if isinstance(self.parent, FreeGroup):
            gen_syms = [('x_%d'%i) for i in range(len(self.generators))]
            return free_group(', '.join(gen_syms))[0]
        return self.parent.subgroup(C=self.C)

    def __str__(self):
        if len(self.generators) > 30:
            str_form = "<fp subgroup with %s generators>" % len(self.generators)
        else:
            str_form = "<fp subgroup on the generators %s>" % str(self.generators)
        return str_form

    __repr__ = __str__
```
### 6 - sympy/combinatorics/perm_groups.py:

Start line: 1406, End line: 1499

```python
class PermutationGroup(Basic):

    def derived_subgroup(self):
        r"""Compute the derived subgroup.

        The derived subgroup, or commutator subgroup is the subgroup generated
        by all commutators `[g, h] = hgh^{-1}g^{-1}` for `g, h\in G` ; it is
        equal to the normal closure of the set of commutators of the generators
        ([1], p.28, [11]).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([1, 0, 2, 4, 3])
        >>> b = Permutation([0, 1, 3, 2, 4])
        >>> G = PermutationGroup([a, b])
        >>> C = G.derived_subgroup()
        >>> list(C.generate(af=True))
        [[0, 1, 2, 3, 4], [0, 1, 3, 4, 2], [0, 1, 4, 2, 3]]

        See Also
        ========

        derived_series

        """
        r = self._r
        gens = [p._array_form for p in self.generators]
        set_commutators = set()
        degree = self._degree
        rng = list(range(degree))
        for i in range(r):
            for j in range(r):
                p1 = gens[i]
                p2 = gens[j]
                c = list(range(degree))
                for k in rng:
                    c[p2[p1[k]]] = p1[p2[k]]
                ct = tuple(c)
                if not ct in set_commutators:
                    set_commutators.add(ct)
        cms = [_af_new(p) for p in set_commutators]
        G2 = self.normal_closure(cms)
        return G2

    def generate(self, method="coset", af=False):
        """Return iterator to generate the elements of the group

        Iteration is done with one of these methods::

          method='coset'  using the Schreier-Sims coset representation
          method='dimino' using the Dimino method

        If af = True it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import PermutationGroup
        >>> from sympy.combinatorics.polyhedron import tetrahedron

        The permutation group given in the tetrahedron object is also
        true groups:

        >>> G = tetrahedron.pgroup
        >>> G.is_group
        True

        Also the group generated by the permutations in the tetrahedron
        pgroup -- even the first two -- is a proper group:

        >>> H = PermutationGroup(G[0], G[1])
        >>> J = PermutationGroup(list(H.generate())); J
        PermutationGroup([
            (0 1)(2 3),
            (1 2 3),
            (1 3 2),
            (0 3 1),
            (0 2 3),
            (0 3)(1 2),
            (0 1 3),
            (3)(0 2 1),
            (0 3 2),
            (3)(0 1 2),
            (0 2)(1 3)])
        >>> _.is_group
        True
        """
        if method == "coset":
            return self.generate_schreier_sims(af)
        elif method == "dimino":
            return self.generate_dimino(af)
        else:
            raise NotImplementedError('No generation defined for %s' % method)
```
### 7 - sympy/combinatorics/perm_groups.py:

Start line: 5062, End line: 5094

```python
PermGroup = PermutationGroup

class SymmetricPermutationGroup(Basic):
    """
    The class defining the lazy form of SymmetricGroup.

    deg : int

    """

    def __new__(cls, deg):
        deg = _sympify(deg)
        obj = Basic.__new__(cls, deg)
        obj._deg = deg
        obj._order = None
        return obj

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> Permutation(1, 2, 3) in G
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError("A SymmetricPermutationGroup contains only Permutations as "
                            "elements, not elements of type %s" % type(i))
        return i.size == self.degree
```
### 8 - sympy/combinatorics/fp_groups.py:

Start line: 279, End line: 325

```python
class FpGroup(DefaultPrinting):


    def _finite_index_subgroup(self, s=[]):
        '''
        Find the elements of `self` that generate a finite index subgroup
        and, if found, return the list of elements and the coset table of `self` by
        the subgroup, otherwise return `(None, None)`

        '''
        gen = self.most_frequent_generator()
        rels = list(self.generators)
        rels.extend(self.relators)
        if not s:
            if len(self.generators) == 2:
                s = [gen] + [g for g in self.generators if g != gen]
            else:
                rand = self.free_group.identity
                i = 0
                while ((rand in rels or rand**-1 in rels or rand.is_identity)
                        and i<10):
                    rand = self.random()
                    i += 1
                s = [gen, rand] + [g for g in self.generators if g != gen]
        mid = (len(s)+1)//2
        half1 = s[:mid]
        half2 = s[mid:]
        draft1 = None
        draft2 = None
        m = 200
        C = None
        while not C and (m/2 < CosetTable.coset_table_max_limit):
            m = min(m, CosetTable.coset_table_max_limit)
            draft1 = self.coset_enumeration(half1, max_cosets=m,
                                 draft=draft1, incomplete=True)
            if draft1.is_complete():
                C = draft1
                half = half1
            else:
                draft2 = self.coset_enumeration(half2, max_cosets=m,
                                 draft=draft2, incomplete=True)
                if draft2.is_complete():
                    C = draft2
                    half = half2
            if not C:
                m *= 2
        if not C:
            return None, None
        C.compress()
        return half, C
```
### 9 - sympy/combinatorics/perm_groups.py:

Start line: 1501, End line: 2252

```python
class PermutationGroup(Basic):

    def generate_dimino(self, af=False):
        """Yield group elements using Dimino's algorithm

        If af == True it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_dimino(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1],
         [0, 1, 3, 2], [0, 3, 2, 1], [0, 3, 1, 2]]

        References
        ==========

        .. [1] The Implementation of Various Algorithms for Permutation Groups in
               the Computer Algebra System: AXIOM, N.J. Doye, M.Sc. Thesis

        """
        idn = list(range(self.degree))
        order = 0
        element_list = [idn]
        set_element_list = {tuple(idn)}
        if af:
            yield idn
        else:
            yield _af_new(idn)
        gens = [p._array_form for p in self.generators]

        for i in range(len(gens)):
            # D elements of the subgroup G_i generated by gens[:i]
            D = element_list[:]
            N = [idn]
            while N:
                A = N
                N = []
                for a in A:
                    for g in gens[:i + 1]:
                        ag = _af_rmul(a, g)
                        if tuple(ag) not in set_element_list:
                            # produce G_i*g
                            for d in D:
                                order += 1
                                ap = _af_rmul(d, ag)
                                if af:
                                    yield ap
                                else:
                                    p = _af_new(ap)
                                    yield p
                                element_list.append(ap)
                                set_element_list.add(tuple(ap))
                                N.append(ap)
        self._order = len(element_list)

    def generate_schreier_sims(self, af=False):
        """Yield group elements using the Schreier-Sims representation
        in coset_rank order

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_schreier_sims(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1],
         [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2]]
        """

        n = self._degree
        u = self.basic_transversals
        basic_orbits = self._basic_orbits
        if len(u) == 0:
            for x in self.generators:
                if af:
                    yield x._array_form
                else:
                    yield x
            return
        if len(u) == 1:
            for i in basic_orbits[0]:
                if af:
                    yield u[0][i]._array_form
                else:
                    yield u[0][i]
            return

        u = list(reversed(u))
        basic_orbits = basic_orbits[::-1]
        # stg stack of group elements
        stg = [list(range(n))]
        posmax = [len(x) for x in u]
        n1 = len(posmax) - 1
        pos = [0]*n1
        h = 0
        while 1:
            # backtrack when finished iterating over coset
            if pos[h] >= posmax[h]:
                if h == 0:
                    return
                pos[h] = 0
                h -= 1
                stg.pop()
                continue
            p = _af_rmul(u[h][basic_orbits[h][pos[h]]]._array_form, stg[-1])
            pos[h] += 1
            stg.append(p)
            h += 1
            if h == n1:
                if af:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        yield p
                else:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        p1 = _af_new(p)
                        yield p1
                stg.pop()
                h -= 1

    @property
    def generators(self):
        """Returns the generators of the group.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.generators
        [(1 2), (2)(0 1)]

        """
        return self._generators

    def contains(self, g, strict=True):
        """Test if permutation ``g`` belong to self, ``G``.

        If ``g`` is an element of ``G`` it can be written as a product
        of factors drawn from the cosets of ``G``'s stabilizers. To see
        if ``g`` is one of the actual generators defining the group use
        ``G.has(g)``.

        If ``strict`` is not ``True``, ``g`` will be resized, if necessary,
        to match the size of permutations in ``self``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup

        >>> a = Permutation(1, 2)
        >>> b = Permutation(2, 3, 1)
        >>> G = PermutationGroup(a, b, degree=5)
        >>> G.contains(G[0]) # trivial check
        True
        >>> elem = Permutation([[2, 3]], size=5)
        >>> G.contains(elem)
        True
        >>> G.contains(Permutation(4)(0, 1, 2, 3))
        False

        If strict is False, a permutation will be resized, if
        necessary:

        >>> H = PermutationGroup(Permutation(5))
        >>> H.contains(Permutation(3))
        False
        >>> H.contains(Permutation(3), strict=False)
        True

        To test if a given permutation is present in the group:

        >>> elem in G.generators
        False
        >>> G.has(elem)
        False

        See Also
        ========

        coset_factor, sympy.core.basic.Basic.has, __contains__

        """
        if not isinstance(g, Permutation):
            return False
        if g.size != self.degree:
            if strict:
                return False
            g = Permutation(g, size=self.degree)
        if g in self.generators:
            return True
        return bool(self.coset_factor(g.array_form, True))

    @property
    def is_perfect(self):
        """Return ``True`` if the group is perfect.
        A group is perfect if it equals to its derived subgroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation(1,2,3)(4,5)
        >>> b = Permutation(1,2,3,4,5)
        >>> G = PermutationGroup([a, b])
        >>> G.is_perfect
        False

        """
        if self._is_perfect is None:
            self._is_perfect = self == self.derived_subgroup()
        return self._is_perfect

    @property
    def is_abelian(self):
        """Test if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.is_abelian
        False
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_abelian
        True

        """
        if self._is_abelian is not None:
            return self._is_abelian

        self._is_abelian = True
        gens = [p._array_form for p in self.generators]
        for x in gens:
            for y in gens:
                if y <= x:
                    continue
                if not _af_commutes_with(x, y):
                    self._is_abelian = False
                    return False
        return True

    def abelian_invariants(self):
        """
        Returns the abelian invariants for the given group.
        Let ``G`` be a nontrivial finite abelian group. Then G is isomorphic to
        the direct product of finitely many nontrivial cyclic groups of
        prime-power order.

        The prime-powers that occur as the orders of the factors are uniquely
        determined by G. More precisely, the primes that occur in the orders of the
        factors in any such decomposition of ``G`` are exactly the primes that divide
        ``|G|`` and for any such prime ``p``, if the orders of the factors that are
        p-groups in one such decomposition of ``G`` are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``,
        then the orders of the factors that are p-groups in any such decomposition of ``G``
        are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``.

        The uniquely determined integers ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``, taken
        for all primes that divide ``|G|`` are called the invariants of the nontrivial
        group ``G`` as suggested in ([14], p. 542).

        Notes
        =====

        We adopt the convention that the invariants of a trivial group are [].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.abelian_invariants()
        [2]
        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> G = CyclicGroup(7)
        >>> G.abelian_invariants()
        [7]

        """
        if self.is_trivial:
            return []
        gns = self.generators
        inv = []
        G = self
        H = G.derived_subgroup()
        Hgens = H.generators
        for p in primefactors(G.order()):
            ranks = []
            while True:
                pows = []
                for g in gns:
                    elm = g**p
                    if not H.contains(elm):
                        pows.append(elm)
                K = PermutationGroup(Hgens + pows) if pows else H
                r = G.order()//K.order()
                G = K
                gns = pows
                if r == 1:
                    break;
                ranks.append(multiplicity(p, r))

            if ranks:
                pows = [1]*ranks[0]
                for i in ranks:
                    for j in range(0, i):
                        pows[j] = pows[j]*p
                inv.extend(pows)
        inv.sort()
        return inv

    def is_elementary(self, p):
        """Return ``True`` if the group is elementary abelian. An elementary
        abelian group is a finite abelian group, where every nontrivial
        element has order `p`, where `p` is a prime.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_elementary(2)
        True
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([3, 1, 2, 0])
        >>> G = PermutationGroup([a, b])
        >>> G.is_elementary(2)
        True
        >>> G.is_elementary(3)
        False

        """
        return self.is_abelian and all(g.order() == p for g in self.generators)

    def _eval_is_alt_sym_naive(self, only_sym=False, only_alt=False):
        """A naive test using the group order."""
        if only_sym and only_alt:
            raise ValueError(
                "Both {} and {} cannot be set to True"
                .format(only_sym, only_alt))

        n = self.degree
        sym_order = 1
        for i in range(2, n+1):
            sym_order *= i
        order = self.order()

        if order == sym_order:
            self._is_sym = True
            self._is_alt = False
            if only_alt:
                return False
            return True

        elif 2*order == sym_order:
            self._is_sym = False
            self._is_alt = True
            if only_sym:
                return False
            return True

        return False

    def _eval_is_alt_sym_monte_carlo(self, eps=0.05, perms=None):
        """A test using monte-carlo algorithm.

        Parameters
        ==========

        eps : float, optional
            The criterion for the incorrect ``False`` return.

        perms : list[Permutation], optional
            If explicitly given, it tests over the given candidats
            for testing.

            If ``None``, it randomly computes ``N_eps`` and chooses
            ``N_eps`` sample of the permutation from the group.

        See Also
        ========

        _check_cycles_alt_sym
        """
        if perms is None:
            n = self.degree
            if n < 17:
                c_n = 0.34
            else:
                c_n = 0.57
            d_n = (c_n*log(2))/log(n)
            N_eps = int(-log(eps)/d_n)

            perms = (self.random_pr() for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        for perm in perms:
            if _check_cycles_alt_sym(perm):
                return True
        return False

    def is_alt_sym(self, eps=0.05, _random_prec=None):
        r"""Monte Carlo test for the symmetric/alternating group for degrees
        >= 8.

        More specifically, it is one-sided Monte Carlo with the
        answer True (i.e., G is symmetric/alternating) guaranteed to be
        correct, and the answer False being incorrect with probability eps.

        For degree < 8, the order of the group is checked so the test
        is deterministic.

        Notes
        =====

        The algorithm itself uses some nontrivial results from group theory and
        number theory:
        1) If a transitive group ``G`` of degree ``n`` contains an element
        with a cycle of length ``n/2 < p < n-2`` for ``p`` a prime, ``G`` is the
        symmetric or alternating group ([1], pp. 81-82)
        2) The proportion of elements in the symmetric/alternating group having
        the property described in 1) is approximately `\log(2)/\log(n)`
        ([1], p.82; [2], pp. 226-227).
        The helper function ``_check_cycles_alt_sym`` is used to
        go over the cycles in a permutation and look for ones satisfying 1).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_alt_sym()
        False

        See Also
        ========

        _check_cycles_alt_sym

        """
        if _random_prec is not None:
            N_eps = _random_prec['N_eps']
            perms= (_random_prec[i] for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        if self._is_sym or self._is_alt:
            return True
        if self._is_sym is False and self._is_alt is False:
            return False

        n = self.degree
        if n < 8:
            return self._eval_is_alt_sym_naive()
        elif self.is_transitive():
            return self._eval_is_alt_sym_monte_carlo(eps=eps)

        self._is_sym, self._is_alt = False, False
        return False

    @property
    def is_nilpotent(self):
        """Test if the group is nilpotent.

        A group `G` is nilpotent if it has a central series of finite length.
        Alternatively, `G` is nilpotent if its lower central series terminates
        with the trivial group. Every nilpotent group is also solvable
        ([1], p.29, [12]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> C = CyclicGroup(6)
        >>> C.is_nilpotent
        True
        >>> S = SymmetricGroup(5)
        >>> S.is_nilpotent
        False

        See Also
        ========

        lower_central_series, is_solvable

        """
        if self._is_nilpotent is None:
            lcs = self.lower_central_series()
            terminator = lcs[len(lcs) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                self._is_nilpotent = True
                return True
            else:
                self._is_nilpotent = False
                return False
        else:
            return self._is_nilpotent

    def is_normal(self, gr, strict=True):
        """Test if ``G=self`` is a normal subgroup of ``gr``.

        G is normal in gr if
        for each g2 in G, g1 in gr, ``g = g1*g2*g1**-1`` belongs to G
        It is sufficient to check this for each g1 in gr.generators and
        g2 in G.generators.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G1 = PermutationGroup([a, Permutation([2, 0, 1])])
        >>> G1.is_normal(G)
        True

        """
        if not self.is_subgroup(gr, strict=strict):
            return False
        d_self = self.degree
        d_gr = gr.degree
        if self.is_trivial and (d_self == d_gr or not strict):
            return True
        if self._is_abelian:
            return True
        new_self = self.copy()
        if not strict and d_self != d_gr:
            if d_self < d_gr:
                new_self = PermGroup(new_self.generators + [Permutation(d_gr - 1)])
            else:
                gr = PermGroup(gr.generators + [Permutation(d_self - 1)])
        gens2 = [p._array_form for p in new_self.generators]
        gens1 = [p._array_form for p in gr.generators]
        for g1 in gens1:
            for g2 in gens2:
                p = _af_rmuln(g1, g2, _af_invert(g1))
                if not new_self.coset_factor(p, True):
                    return False
        return True

    def is_primitive(self, randomized=True):
        r"""Test if a group is primitive.

        A permutation group ``G`` acting on a set ``S`` is called primitive if
        ``S`` contains no nontrivial block under the action of ``G``
        (a block is nontrivial if its cardinality is more than ``1``).

        Notes
        =====

        The algorithm is described in [1], p.83, and uses the function
        minimal_block to search for blocks of the form `\{0, k\}` for ``k``
        ranging over representatives for the orbits of `G_0`, the stabilizer of
        ``0``. This algorithm has complexity `O(n^2)` where ``n`` is the degree
        of the group, and will perform badly if `G_0` is small.

        There are two implementations offered: one finds `G_0`
        deterministically using the function ``stabilizer``, and the other
        (default) produces random elements of `G_0` using ``random_stab``,
        hoping that they generate a subgroup of `G_0` with not too many more
        orbits than `G_0` (this is suggested in [1], p.83). Behavior is changed
        by the ``randomized`` flag.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_primitive()
        False

        See Also
        ========

        minimal_block, random_stab

        """
        if self._is_primitive is not None:
            return self._is_primitive

        if self.is_transitive() is False:
            return False

        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for i in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0 and any(e != 0 for e in self.minimal_block([0, x])):
                self._is_primitive = False
                return False
        self._is_primitive = True
        return True

    def minimal_blocks(self, randomized=True):
        '''
        For a transitive group, return the list of all minimal
        block systems. If a group is intransitive, return `False`.

        Examples
        ========
        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> DihedralGroup(6).minimal_blocks()
        [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]
        >>> G = PermutationGroup(Permutation(1,2,5))
        >>> G.minimal_blocks()
        False

        See Also
        ========

        minimal_block, is_transitive, is_primitive

        '''
        def _number_blocks(blocks):
            # number the blocks of a block system
            # in order and return the number of
            # blocks and the tuple with the
            # reordering
            n = len(blocks)
            appeared = {}
            m = 0
            b = [None]*n
            for i in range(n):
                if blocks[i] not in appeared:
                    appeared[blocks[i]] = m
                    b[i] = m
                    m += 1
                else:
                    b[i] = appeared[blocks[i]]
            return tuple(b), m

        if not self.is_transitive():
            return False
        blocks = []
        num_blocks = []
        rep_blocks = []
        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for i in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0:
                block = self.minimal_block([0, x])
                num_block, m = _number_blocks(block)
                # a representative block (containing 0)
                rep = {j for j in range(self.degree) if num_block[j] == 0}
                # check if the system is minimal with
                # respect to the already discovere ones
                minimal = True
                to_remove = []
                for i, r in enumerate(rep_blocks):
                    if len(r) > len(rep) and rep.issubset(r):
                        # i-th block system is not minimal
                        del num_blocks[i], blocks[i]
                        to_remove.append(rep_blocks[i])
                    elif len(r) < len(rep) and r.issubset(rep):
                        # the system being checked is not minimal
                        minimal = False
                        break
                # remove non-minimal representative blocks
                rep_blocks = [r for r in rep_blocks if r not in to_remove]

                if minimal and num_block not in num_blocks:
                    blocks.append(block)
                    num_blocks.append(num_block)
                    rep_blocks.append(rep)
        return blocks

    @property
    def is_solvable(self):
        """Test if the group is solvable.

        ``G`` is solvable if its derived series terminates with the trivial
        group ([1], p.29).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(3)
        >>> S.is_solvable
        True

        See Also
        ========

        is_nilpotent, derived_series

        """
        if self._is_solvable is None:
            if self.order() % 2 != 0:
                return True
            ds = self.derived_series()
            terminator = ds[len(ds) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                return True
            else:
                self._is_solvable = False
                return False
        else:
            return self._is_solvable
```
### 10 - sympy/combinatorics/pc_groups.py:

Start line: 614, End line: 656

```python
class Collector(DefaultPrinting):

    def induced_pcgs(self, gens):
        """

        Parameters
        ==========

        gens : list
            A list of generators on which polycyclic subgroup
            is to be defined.

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
        z = [1]*len(self.pcgs)
        G = gens
        while G:
            g = G.pop(0)
            h = self._sift(z, g)
            d = self.depth(h)
            if d < len(self.pcgs):
                for gen in z:
                    if gen != 1:
                        G.append(h**-1*gen**-1*h*gen)
                z[d-1] = h;
        z = [gen for gen in z if gen != 1]
        return z
```
### 18 - sympy/combinatorics/perm_groups.py:

Start line: 694, End line: 1404

```python
class PermutationGroup(Basic):

    def composition_series(self):
        r"""
        Return the composition series for a group as a list
        of permutation groups.

        The composition series for a group `G` is defined as a
        subnormal series `G = H_0 > H_1 > H_2 \ldots` A composition
        series is a subnormal series such that each factor group
        `H(i+1) / H(i)` is simple.
        A subnormal series is a composition series only if it is of
        maximum length.

        The algorithm works as follows:
        Starting with the derived series the idea is to fill
        the gap between `G = der[i]` and `H = der[i+1]` for each
        `i` independently. Since, all subgroups of the abelian group
        `G/H` are normal so, first step is to take the generators
        `g` of `G` and add them to generators of `H` one by one.

        The factor groups formed are not simple in general. Each
        group is obtained from the previous one by adding one
        generator `g`, if the previous group is denoted by `H`
        then the next group `K` is generated by `g` and `H`.
        The factor group `K/H` is cyclic and it's order is
        `K.order()//G.order()`. The series is then extended between
        `K` and `H` by groups generated by powers of `g` and `H`.
        The series formed is then prepended to the already existing
        series.

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> S = SymmetricGroup(12)
        >>> G = S.sylow_subgroup(2)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        >>> G = S.sylow_subgroup(3)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [243, 81, 27, 9, 3, 1]
        >>> G = CyclicGroup(12)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [12, 6, 3, 1]

        """
        der = self.derived_series()
        if not (all(g.is_identity for g in der[-1].generators)):
            raise NotImplementedError('Group should be solvable')
        series = []

        for i in range(len(der)-1):
            H = der[i+1]
            up_seg = []
            for g in der[i].generators:
                K = PermutationGroup([g] + H.generators)
                order = K.order() // H.order()
                down_seg = []
                for p, e in factorint(order).items():
                    for j in range(e):
                        down_seg.append(PermutationGroup([g] + H.generators))
                        g = g**p
                up_seg = down_seg + up_seg
                H = K
            up_seg[0] = der[i]
            series.extend(up_seg)
        series.append(der[-1])
        return series

    def coset_transversal(self, H):
        """Return a transversal of the right cosets of self by its subgroup H
        using the second method described in [1], Subsection 4.6.7

        """

        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")

        if H.order() == 1:
            return self._elements

        self._schreier_sims(base=H.base) # make G.base an extension of H.base

        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        identity = Permutation(self.degree - 1)

        transversals = self.basic_transversals[:]
        # transversals is a list of dictionaries. Get rid of the keys
        # so that it is a list of lists and sort each list in
        # the increasing order of base[l]^x
        for l, t in enumerate(transversals):
            transversals[l] = sorted(t.values(),
                                key = lambda x: base_ordering[base[l]^x])

        orbits = H.basic_orbits
        h_stabs = H.basic_stabilizers
        g_stabs = self.basic_stabilizers

        indices = [x.order()//y.order() for x, y in zip(g_stabs, h_stabs)]

        # T^(l) should be a right transversal of H^(l) in G^(l) for
        # 1<=l<=len(base). While H^(l) is the trivial group, T^(l)
        # contains all the elements of G^(l) so we might just as well
        # start with l = len(h_stabs)-1
        if len(g_stabs) > len(h_stabs):
            T = g_stabs[len(h_stabs)]._elements
        else:
            T = [identity]
        l = len(h_stabs)-1
        t_len = len(T)
        while l > -1:
            T_next = []
            for u in transversals[l]:
                if u == identity:
                    continue
                b = base_ordering[base[l]^u]
                for t in T:
                    p = t*u
                    if all([base_ordering[h^p] >= b for h in orbits[l]]):
                        T_next.append(p)
                    if t_len + len(T_next) == indices[l]:
                        break
                if t_len + len(T_next) == indices[l]:
                    break
            T += T_next
            t_len += len(T_next)
            l -= 1
        T.remove(identity)
        T = [identity] + T
        return T

    def _coset_representative(self, g, H):
        """Return the representative of Hg from the transversal that
        would be computed by ``self.coset_transversal(H)``.

        """
        if H.order() == 1:
            return g
        # The base of self must be an extension of H.base.
        if not(self.base[:len(H.base)] == H.base):
            self._schreier_sims(base=H.base)
        orbits = H.basic_orbits[:]
        h_transversals = [list(_.values()) for _ in H.basic_transversals]
        transversals = [list(_.values()) for _ in self.basic_transversals]
        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        def step(l, x):
            gamma = sorted(orbits[l], key = lambda y: base_ordering[y^x])[0]
            i = [base[l]^h for h in h_transversals[l]].index(gamma)
            x = h_transversals[l][i]*x
            if l < len(orbits)-1:
                for u in transversals[l]:
                    if base[l]^u == base[l]^x:
                        break
                x = step(l+1, x*u**-1)*u
            return x
        return step(0, g)

    def coset_table(self, H):
        """Return the standardised (right) coset table of self in H as
        a list of lists.
        """
        # Maybe this should be made to return an instance of CosetTable
        # from fp_groups.py but the class would need to be changed first
        # to be compatible with PermutationGroups

        from itertools import chain, product
        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")
        T = self.coset_transversal(H)
        n = len(T)

        A = list(chain.from_iterable((gen, gen**-1)
                    for gen in self.generators))

        table = []
        for i in range(n):
            row = [self._coset_representative(T[i]*x, H) for x in A]
            row = [T.index(r) for r in row]
            table.append(row)

        # standardize (this is the same as the algorithm used in coset_table)
        # If CosetTable is made compatible with PermutationGroups, this
        # should be replaced by table.standardize()
        A = range(len(A))
        gamma = 1
        for alpha, a in product(range(n), A):
            beta = table[alpha][a]
            if beta >= gamma:
                if beta > gamma:
                    for x in A:
                        z = table[gamma][x]
                        table[gamma][x] = table[beta][x]
                        table[beta][x] = z
                        for i in range(n):
                            if table[i][x] == beta:
                                table[i][x] = gamma
                            elif table[i][x] == gamma:
                                table[i][x] = beta
                gamma += 1
            if gamma >= n-1:
                return table

    def center(self):
        r"""
        Return the center of a permutation group.

        The center for a group `G` is defined as
        `Z(G) = \{z\in G | \forall g\in G, zg = gz \}`,
        the set of elements of `G` that commute with all elements of `G`.
        It is equal to the centralizer of `G` inside `G`, and is naturally a
        subgroup of `G` ([9]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> G = D.center()
        >>> G.order()
        2

        See Also
        ========

        centralizer

        Notes
        =====

        This is a naive implementation that is a straightforward application
        of ``.centralizer()``

        """
        return self.centralizer(self)

    def centralizer(self, other):
        r"""
        Return the centralizer of a group/set/element.

        The centralizer of a set of permutations ``S`` inside
        a group ``G`` is the set of elements of ``G`` that commute with all
        elements of ``S``::

            `C_G(S) = \{ g \in G | gs = sg \forall s \in S\}` ([10])

        Usually, ``S`` is a subset of ``G``, but if ``G`` is a proper subgroup of
        the full symmetric group, we allow for ``S`` to have elements outside
        ``G``.

        It is naturally a subgroup of ``G``; the centralizer of a permutation
        group is equal to the centralizer of any set of generators for that
        group, since any element commuting with the generators commutes with
        any product of the  generators.

        Parameters
        ==========

        other
            a permutation group/list of permutations/single permutation

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> S = SymmetricGroup(6)
        >>> C = CyclicGroup(6)
        >>> H = S.centralizer(C)
        >>> H.is_subgroup(C)
        True

        See Also
        ========

        subgroup_search

        Notes
        =====

        The implementation is an application of ``.subgroup_search()`` with
        tests using a specific base for the group ``G``.

        """
        if hasattr(other, 'generators'):
            if other.is_trivial or self.is_trivial:
                return self
            degree = self.degree
            identity = _af_new(list(range(degree)))
            orbits = other.orbits()
            num_orbits = len(orbits)
            orbits.sort(key=lambda x: -len(x))
            long_base = []
            orbit_reps = [None]*num_orbits
            orbit_reps_indices = [None]*num_orbits
            orbit_descr = [None]*degree
            for i in range(num_orbits):
                orbit = list(orbits[i])
                orbit_reps[i] = orbit[0]
                orbit_reps_indices[i] = len(long_base)
                for point in orbit:
                    orbit_descr[point] = i
                long_base = long_base + orbit
            base, strong_gens = self.schreier_sims_incremental(base=long_base)
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            i = 0
            for i in range(len(base)):
                if strong_gens_distr[i] == [identity]:
                    break
            base = base[:i]
            base_len = i
            for j in range(num_orbits):
                if base[base_len - 1] in orbits[j]:
                    break
            rel_orbits = orbits[: j + 1]
            num_rel_orbits = len(rel_orbits)
            transversals = [None]*num_rel_orbits
            for j in range(num_rel_orbits):
                rep = orbit_reps[j]
                transversals[j] = dict(
                    other.orbit_transversal(rep, pairs=True))
            trivial_test = lambda x: True
            tests = [None]*base_len
            for l in range(base_len):
                if base[l] in orbit_reps:
                    tests[l] = trivial_test
                else:
                    def test(computed_words, l=l):
                        g = computed_words[l]
                        rep_orb_index = orbit_descr[base[l]]
                        rep = orbit_reps[rep_orb_index]
                        im = g._array_form[base[l]]
                        im_rep = g._array_form[rep]
                        tr_el = transversals[rep_orb_index][base[l]]
                        # using the definition of transversal,
                        # base[l]^g = rep^(tr_el*g);
                        # if g belongs to the centralizer, then
                        # base[l]^g = (rep^g)^tr_el
                        return im == tr_el._array_form[im_rep]
                    tests[l] = test

            def prop(g):
                return [rmul(g, gen) for gen in other.generators] == \
                       [rmul(gen, g) for gen in other.generators]
            return self.subgroup_search(prop, base=base,
                                        strong_gens=strong_gens, tests=tests)
        elif hasattr(other, '__getitem__'):
            gens = list(other)
            return self.centralizer(PermutationGroup(gens))
        elif hasattr(other, 'array_form'):
            return self.centralizer(PermutationGroup([other]))

    def commutator(self, G, H):
        """
        Return the commutator of two subgroups.

        For a permutation group ``K`` and subgroups ``G``, ``H``, the
        commutator of ``G`` and ``H`` is defined as the group generated
        by all the commutators `[g, h] = hgh^{-1}g^{-1}` for ``g`` in ``G`` and
        ``h`` in ``H``. It is naturally a subgroup of ``K`` ([1], p.27).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> A = AlternatingGroup(5)
        >>> G = S.commutator(S, A)
        >>> G.is_subgroup(A)
        True

        See Also
        ========

        derived_subgroup

        Notes
        =====

        The commutator of two subgroups `H, G` is equal to the normal closure
        of the commutators of all the generators, i.e. `hgh^{-1}g^{-1}` for `h`
        a generator of `H` and `g` a generator of `G` ([1], p.28)

        """
        ggens = G.generators
        hgens = H.generators
        commutators = []
        for ggen in ggens:
            for hgen in hgens:
                commutator = rmul(hgen, ggen, ~hgen, ~ggen)
                if commutator not in commutators:
                    commutators.append(commutator)
        res = self.normal_closure(commutators)
        return res

    def coset_factor(self, g, factor_index=False):
        """Return ``G``'s (self's) coset factorization of ``g``

        If ``g`` is an element of ``G`` then it can be written as the product
        of permutations drawn from the Schreier-Sims coset decomposition,

        The permutations returned in ``f`` are those for which
        the product gives ``g``: ``g = f[n]*...f[1]*f[0]`` where ``n = len(B)``
        and ``B = G.base``. f[i] is one of the permutations in
        ``self._basic_orbits[i]``.

        If factor_index==True,
        returns a tuple ``[b[0],..,b[n]]``, where ``b[i]``
        belongs to ``self._basic_orbits[i]``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])

        Define g:

        >>> g = Permutation(7)(1, 2, 4)(3, 6, 5)

        Confirm that it is an element of G:

        >>> G.contains(g)
        True

        Thus, it can be written as a product of factors (up to
        3) drawn from u. See below that a factor from u1 and u2
        and the Identity permutation have been used:

        >>> f = G.coset_factor(g)
        >>> f[2]*f[1]*f[0] == g
        True
        >>> f1 = G.coset_factor(g, True); f1
        [0, 4, 4]
        >>> tr = G.basic_transversals
        >>> f[0] == tr[0][f1[0]]
        True

        If g is not an element of G then [] is returned:

        >>> c = Permutation(5, 6, 7)
        >>> G.coset_factor(c)
        []

        See Also
        ========

        sympy.combinatorics.util._strip

        """
        if isinstance(g, (Cycle, Permutation)):
            g = g.list()
        if len(g) != self._degree:
            # this could either adjust the size or return [] immediately
            # but we don't choose between the two and just signal a possible
            # error
            raise ValueError('g should be the same size as permutations of G')
        I = list(range(self._degree))
        basic_orbits = self.basic_orbits
        transversals = self._transversals
        factors = []
        base = self.base
        h = g
        for i in range(len(base)):
            beta = h[base[i]]
            if beta == base[i]:
                factors.append(beta)
                continue
            if beta not in basic_orbits[i]:
                return []
            u = transversals[i][beta]._array_form
            h = _af_rmul(_af_invert(u), h)
            factors.append(beta)
        if h != I:
            return []
        if factor_index:
            return factors
        tr = self.basic_transversals
        factors = [tr[i][factors[i]] for i in range(len(base))]
        return factors

    def generator_product(self, g, original=False):
        '''
        Return a list of strong generators `[s1, ..., sn]`
        s.t `g = sn*...*s1`. If `original=True`, make the list
        contain only the original group generators

        '''
        product = []
        if g.is_identity:
            return []
        if g in self.strong_gens:
            if not original or g in self.generators:
                return [g]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                return product
        elif g**-1 in self.strong_gens:
            g = g**-1
            if not original or g in self.generators:
                return [g**-1]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                l = len(product)
                product = [product[l-i-1]**-1 for i in range(l)]
                return product

        f = self.coset_factor(g, True)
        for i, j in enumerate(f):
            slp = self._transversal_slp[i][j]
            for s in slp:
                if not original:
                    product.append(self.strong_gens[s])
                else:
                    s = self.strong_gens[s]
                    product.extend(self.generator_product(s, original=True))
        return product

    def coset_rank(self, g):
        """rank using Schreier-Sims representation

        The coset rank of ``g`` is the ordering number in which
        it appears in the lexicographic listing according to the
        coset decomposition

        The ordering is the same as in G.generate(method='coset').
        If ``g`` does not belong to the group it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])
        >>> c = Permutation(7)(2, 4)(3, 5)
        >>> G.coset_rank(c)
        16
        >>> G.coset_unrank(16)
        (7)(2 4)(3 5)

        See Also
        ========

        coset_factor

        """
        factors = self.coset_factor(g, True)
        if not factors:
            return None
        rank = 0
        b = 1
        transversals = self._transversals
        base = self._base
        basic_orbits = self._basic_orbits
        for i in range(len(base)):
            k = factors[i]
            j = basic_orbits[i].index(k)
            rank += b*j
            b = b*len(transversals[i])
        return rank

    def coset_unrank(self, rank, af=False):
        """unrank using Schreier-Sims representation

        coset_unrank is the inverse operation of coset_rank
        if 0 <= rank < order; otherwise it returns None.

        """
        if rank < 0 or rank >= self.order():
            return None
        base = self.base
        transversals = self.basic_transversals
        basic_orbits = self.basic_orbits
        m = len(base)
        v = [0]*m
        for i in range(m):
            rank, c = divmod(rank, len(transversals[i]))
            v[i] = basic_orbits[i][c]
        a = [transversals[i][v[i]]._array_form for i in range(m)]
        h = _af_rmuln(*a)
        if af:
            return h
        else:
            return _af_new(h)

    @property
    def degree(self):
        """Returns the size of the permutations in the group.

        The number of permutations comprising the group is given by
        ``len(group)``; the number of permutations that can be generated
        by the group is given by ``group.order()``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        See Also
        ========

        order
        """
        return self._degree

    @property
    def identity(self):
        '''
        Return the identity element of the permutation group.

        '''
        return _af_new(list(range(self.degree)))

    @property
    def elements(self):
        """Returns all the elements of the permutation group as a set

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = PermutationGroup(Permutation(1, 3), Permutation(1, 2))
        >>> p.elements
        {(1 2 3), (1 3 2), (1 3), (2 3), (3), (3)(1 2)}

        """
        return set(self._elements)

    @property
    def _elements(self):
        """Returns all the elements of the permutation group as a list

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = PermutationGroup(Permutation(1, 3), Permutation(1, 2))
        >>> p._elements
        [(3), (3)(1 2), (1 3), (2 3), (1 2 3), (1 3 2)]

        """
        return list(islice(self.generate(), None))

    def derived_series(self):
        r"""Return the derived series for the group.

        The derived series for a group `G` is defined as
        `G = G_0 > G_1 > G_2 > \ldots` where `G_i = [G_{i-1}, G_{i-1}]`,
        i.e. `G_i` is the derived subgroup of `G_{i-1}`, for
        `i\in\mathbb{N}`. When we have `G_k = G_{k-1}` for some
        `k\in\mathbb{N}`, the series terminates.

        Returns
        =======

        A list of permutation groups containing the members of the derived
        series in the order `G = G_0, G_1, G_2, \ldots`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup, DihedralGroup)
        >>> A = AlternatingGroup(5)
        >>> len(A.derived_series())
        1
        >>> S = SymmetricGroup(4)
        >>> len(S.derived_series())
        4
        >>> S.derived_series()[1].is_subgroup(AlternatingGroup(4))
        True
        >>> S.derived_series()[2].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        derived_subgroup

        """
        res = [self]
        current = self
        next = self.derived_subgroup()
        while not current.is_subgroup(next):
            res.append(next)
            current = next
            next = next.derived_subgroup()
        return res
```
### 27 - sympy/combinatorics/perm_groups.py:

Start line: 4355, End line: 4849

```python
class PermutationGroup(Basic):

    def sylow_subgroup(self, p):
        # ... other code
        if len(blocks) > 1:
            # apply _sylow_reduce to block system actions
            mu = block_homomorphism(self, blocks[0])
            nu = block_homomorphism(self, blocks[1])
            return _sylow_reduce(mu, nu)
        elif len(blocks) == 1:
            block = list(blocks)[0]
            if any(e != 0 for e in block):
                # self is imprimitive
                mu = block_homomorphism(self, block)
                if not is_p_group(mu.image())[0]:
                    S = mu.image().sylow_subgroup(p)
                    return mu.invert_subgroup(S).sylow_subgroup(p)

        # find an element of order p
        g = self.random()
        g_order = g.order()
        while g_order % p != 0 or g_order == 0:
            g = self.random()
            g_order = g.order()
        g = g**(g_order // p)
        if order % p**2 != 0:
            return PermutationGroup(g)

        C = self.centralizer(g)
        while C.order() % p**n != 0:
            S = C.sylow_subgroup(p)
            s_order = S.order()
            Z = S.center()
            P = Z._p_elements_group(p)
            h = P.random()
            C_h = self.centralizer(h)
            while C_h.order() % p*s_order != 0:
                h = P.random()
                C_h = self.centralizer(h)
            C = C_h

        return C.sylow_subgroup(p)

    def _block_verify(H, L, alpha):
        delta = sorted(list(H.orbit(alpha)))
        H_gens = H.generators
        # p[i] will be the number of the block
        # delta[i] belongs to
        p = [-1]*len(delta)
        blocks = [-1]*len(delta)

        B = [[]] # future list of blocks
        u = [0]*len(delta) # u[i] in L s.t. alpha^u[i] = B[0][i]

        t = L.orbit_transversal(alpha, pairs=True)
        for a, beta in t:
            B[0].append(a)
            i_a = delta.index(a)
            p[i_a] = 0
            blocks[i_a] = alpha
            u[i_a] = beta

        rho = 0
        m = 0 # number of blocks - 1

        while rho <= m:
            beta = B[rho][0]
            for g in H_gens:
                d = beta^g
                i_d = delta.index(d)
                sigma = p[i_d]
                if sigma < 0:
                    # define a new block
                    m += 1
                    sigma = m
                    u[i_d] = u[delta.index(beta)]*g
                    p[i_d] = sigma
                    rep = d
                    blocks[i_d] = rep
                    newb = [rep]
                    for gamma in B[rho][1:]:
                        i_gamma = delta.index(gamma)
                        d = gamma^g
                        i_d = delta.index(d)
                        if p[i_d] < 0:
                            u[i_d] = u[i_gamma]*g
                            p[i_d] = sigma
                            blocks[i_d] = rep
                            newb.append(d)
                        else:
                            # B[rho] is not a block
                            s = u[i_gamma]*g*u[i_d]**(-1)
                            return False, s

                    B.append(newb)
                else:
                    for h in B[rho][1:]:
                        if not h^g in B[sigma]:
                            # B[rho] is not a block
                            s = u[delta.index(beta)]*g*u[i_d]**(-1)
                            return False, s
            rho += 1

        return True, blocks

    def _verify(H, K, phi, z, alpha):
        '''
        Return a list of relators ``rels`` in generators ``gens`_h` that
        are mapped to ``H.generators`` by ``phi`` so that given a finite
        presentation <gens_k | rels_k> of ``K`` on a subset of ``gens_h``
        <gens_h | rels_k + rels> is a finite presentation of ``H``.

        ``H`` should be generated by the union of ``K.generators`` and ``z``
        (a single generator), and ``H.stabilizer(alpha) == K``; ``phi`` is a
        canonical injection from a free group into a permutation group
        containing ``H``.

        The algorithm is described in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.homomorphisms import homomorphism
        >>> from sympy.combinatorics.free_groups import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup

        >>> H = PermutationGroup(Permutation(0, 2), Permutation (1, 5))
        >>> K = PermutationGroup(Permutation(5)(0, 2))
        >>> F = free_group("x_0 x_1")[0]
        >>> gens = F.generators
        >>> phi = homomorphism(F, H, F.generators, H.generators)
        >>> rels_k = [gens[0]**2] # relators for presentation of K
        >>> z= Permutation(1, 5)
        >>> check, rels_h = H._verify(K, phi, z, 1)
        >>> check
        True
        >>> rels = rels_k + rels_h
        >>> G = FpGroup(F, rels) # presentation of H
        >>> G.order() == H.order()
        True

        See also
        ========

        strong_presentation, presentation, stabilizer

        '''

        orbit = H.orbit(alpha)
        beta = alpha^(z**-1)

        K_beta = K.stabilizer(beta)

        # orbit representatives of K_beta
        gammas = [alpha, beta]
        orbits = list({tuple(K_beta.orbit(o)) for o in orbit})
        orbit_reps = [orb[0] for orb in orbits]
        for rep in orbit_reps:
            if rep not in gammas:
                gammas.append(rep)

        # orbit transversal of K
        betas = [alpha, beta]
        transversal = {alpha: phi.invert(H.identity), beta: phi.invert(z**-1)}

        for s, g in K.orbit_transversal(beta, pairs=True):
            if not s in transversal:
                transversal[s] = transversal[beta]*phi.invert(g)


        union = K.orbit(alpha).union(K.orbit(beta))
        while (len(union) < len(orbit)):
            for gamma in gammas:
                if gamma in union:
                    r = gamma^z
                    if r not in union:
                        betas.append(r)
                        transversal[r] = transversal[gamma]*phi.invert(z)
                        for s, g in K.orbit_transversal(r, pairs=True):
                            if not s in transversal:
                                transversal[s] = transversal[r]*phi.invert(g)
                        union = union.union(K.orbit(r))
                        break

        # compute relators
        rels = []

        for b in betas:
            k_gens = K.stabilizer(b).generators
            for y in k_gens:
                new_rel = transversal[b]
                gens = K.generator_product(y, original=True)
                for g in gens[::-1]:
                    new_rel = new_rel*phi.invert(g)
                new_rel = new_rel*transversal[b]**-1

                perm = phi(new_rel)
                try:
                    gens = K.generator_product(perm, original=True)
                except ValueError:
                    return False, perm
                for g in gens:
                    new_rel = new_rel*phi.invert(g)**-1
                if new_rel not in rels:
                    rels.append(new_rel)

        for gamma in gammas:
            new_rel = transversal[gamma]*phi.invert(z)*transversal[gamma^z]**-1
            perm = phi(new_rel)
            try:
                gens = K.generator_product(perm, original=True)
            except ValueError:
                return False, perm
            for g in gens:
               new_rel = new_rel*phi.invert(g)**-1
            if new_rel not in rels:
                rels.append(new_rel)

        return True, rels

    def strong_presentation(G):
        '''
        Return a strong finite presentation of `G`. The generators
        of the returned group are in the same order as the strong
        generators of `G`.

        The algorithm is based on Sims' Verify algorithm described
        in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> P = DihedralGroup(4)
        >>> G = P.strong_presentation()
        >>> P.order() == G.order()
        True

        See Also
        ========

        presentation, _verify

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import (block_homomorphism,
                                           homomorphism, GroupHomomorphism)

        strong_gens = G.strong_gens[:]
        stabs = G.basic_stabilizers[:]
        base = G.base[:]

        # injection from a free group on len(strong_gens)
        # generators into G
        gen_syms = [('x_%d'%i) for i in range(len(strong_gens))]
        F = free_group(', '.join(gen_syms))[0]
        phi = homomorphism(F, G, F.generators, strong_gens)

        H = PermutationGroup(G.identity)
        while stabs:
            alpha = base.pop()
            K = H
            H = stabs.pop()
            new_gens = [g for g in H.generators if g not in K]

            if K.order() == 1:
                z = new_gens.pop()
                rels = [F.generators[-1]**z.order()]
                intermediate_gens = [z]
                K = PermutationGroup(intermediate_gens)

            # add generators one at a time building up from K to H
            while new_gens:
                z = new_gens.pop()
                intermediate_gens = [z] + intermediate_gens
                K_s = PermutationGroup(intermediate_gens)
                orbit = K_s.orbit(alpha)
                orbit_k = K.orbit(alpha)

                # split into cases based on the orbit of K_s
                if orbit_k == orbit:
                    if z in K:
                        rel = phi.invert(z)
                        perm = z
                    else:
                        t = K.orbit_rep(alpha, alpha^z)
                        rel = phi.invert(z)*phi.invert(t)**-1
                        perm = z*t**-1
                    for g in K.generator_product(perm, original=True):
                        rel = rel*phi.invert(g)**-1
                    new_rels = [rel]
                elif len(orbit_k) == 1:
                    # `success` is always true because `strong_gens`
                    # and `base` are already a verified BSGS. Later
                    # this could be changed to start with a randomly
                    # generated (potential) BSGS, and then new elements
                    # would have to be appended to it when `success`
                    # is false.
                    success, new_rels = K_s._verify(K, phi, z, alpha)
                else:
                    # K.orbit(alpha) should be a block
                    # under the action of K_s on K_s.orbit(alpha)
                    check, block = K_s._block_verify(K, alpha)
                    if check:
                        # apply _verify to the action of K_s
                        # on the block system; for convenience,
                        # add the blocks as additional points
                        # that K_s should act on
                        t = block_homomorphism(K_s, block)
                        m = t.codomain.degree # number of blocks
                        d = K_s.degree

                        # conjugating with p will shift
                        # permutations in t.image() to
                        # higher numbers, e.g.
                        # p*(0 1)*p = (m m+1)
                        p = Permutation()
                        for i in range(m):
                            p *= Permutation(i, i+d)

                        t_img = t.images
                        # combine generators of K_s with their
                        # action on the block system
                        images = {g: g*p*t_img[g]*p for g in t_img}
                        for g in G.strong_gens[:-len(K_s.generators)]:
                            images[g] = g
                        K_s_act = PermutationGroup(list(images.values()))
                        f = GroupHomomorphism(G, K_s_act, images)

                        K_act = PermutationGroup([f(g) for g in K.generators])
                        success, new_rels = K_s_act._verify(K_act, f.compose(phi), f(z), d)

                for n in new_rels:
                    if not n in rels:
                        rels.append(n)
                K = K_s

        group = FpGroup(F, rels)
        return simplify_presentation(group)

    def presentation(G, eliminate_gens=True):
        '''
        Return an `FpGroup` presentation of the group.

        The algorithm is described in [1], Chapter 6.1.

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.coset_table import CosetTable
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import homomorphism
        from itertools import product

        if G._fp_presentation:
            return G._fp_presentation

        if G._fp_presentation:
            return G._fp_presentation

        def _factor_group_by_rels(G, rels):
            if isinstance(G, FpGroup):
                rels.extend(G.relators)
                return FpGroup(G.free_group, list(set(rels)))
            return FpGroup(G, rels)

        gens = G.generators
        len_g = len(gens)

        if len_g == 1:
            order = gens[0].order()
            # handle the trivial group
            if order == 1:
                return free_group([])[0]
            F, x = free_group('x')
            return FpGroup(F, [x**order])

        if G.order() > 20:
            half_gens = G.generators[0:(len_g+1)//2]
        else:
            half_gens = []
        H = PermutationGroup(half_gens)
        H_p = H.presentation()

        len_h = len(H_p.generators)

        C = G.coset_table(H)
        n = len(C) # subgroup index

        gen_syms = [('x_%d'%i) for i in range(len(gens))]
        F = free_group(', '.join(gen_syms))[0]

        # mapping generators of H_p to those of F
        images = [F.generators[i] for i in range(len_h)]
        R = homomorphism(H_p, F, H_p.generators, images, check=False)

        # rewrite relators
        rels = R(H_p.relators)
        G_p = FpGroup(F, rels)

        # injective homomorphism from G_p into G
        T = homomorphism(G_p, G, G_p.generators, gens)

        C_p = CosetTable(G_p, [])

        C_p.table = [[None]*(2*len_g) for i in range(n)]

        # initiate the coset transversal
        transversal = [None]*n
        transversal[0] = G_p.identity

        # fill in the coset table as much as possible
        for i in range(2*len_h):
            C_p.table[0][i] = 0

        gamma = 1
        for alpha, x in product(range(0, n), range(2*len_g)):
            beta = C[alpha][x]
            if beta == gamma:
                gen = G_p.generators[x//2]**((-1)**(x % 2))
                transversal[beta] = transversal[alpha]*gen
                C_p.table[alpha][x] = beta
                C_p.table[beta][x + (-1)**(x % 2)] = alpha
                gamma += 1
                if gamma == n:
                    break

        C_p.p = list(range(n))
        beta = x = 0

        while not C_p.is_complete():
            # find the first undefined entry
            while C_p.table[beta][x] == C[beta][x]:
                x = (x + 1) % (2*len_g)
                if x == 0:
                    beta = (beta + 1) % n

            # define a new relator
            gen = G_p.generators[x//2]**((-1)**(x % 2))
            new_rel = transversal[beta]*gen*transversal[C[beta][x]]**-1
            perm = T(new_rel)
            next = G_p.identity
            for s in H.generator_product(perm, original=True):
                next = next*T.invert(s)**-1
            new_rel = new_rel*next

            # continue coset enumeration
            G_p = _factor_group_by_rels(G_p, [new_rel])
            C_p.scan_and_fill(0, new_rel)
            C_p = G_p.coset_enumeration([], strategy="coset_table",
                                draft=C_p, max_cosets=n, incomplete=True)

        G._fp_presentation = simplify_presentation(G_p)
        return G._fp_presentation

    def polycyclic_group(self):
        """
        Return the PolycyclicGroup instance with below parameters:

        * ``pc_sequence`` : Polycyclic sequence is formed by collecting all
          the missing generators between the adjacent groups in the
          derived series of given permutation group.

        * ``pc_series`` : Polycyclic series is formed by adding all the missing
          generators of ``der[i+1]`` in ``der[i]``, where ``der`` represents
          the derived series.

        * ``relative_order`` : A list, computed by the ratio of adjacent groups in
          pc_series.

        """
        from sympy.combinatorics.pc_groups import PolycyclicGroup
        if not self.is_polycyclic:
            raise ValueError("The group must be solvable")

        der = self.derived_series()
        pc_series = []
        pc_sequence = []
        relative_order = []
        pc_series.append(der[-1])
        der.reverse()

        for i in range(len(der)-1):
            H = der[i]
            for g in der[i+1].generators:
                if g not in H:
                    H = PermutationGroup([g] + H.generators)
                    pc_series.insert(0, H)
                    pc_sequence.insert(0, g)

                    G1 = pc_series[0].order()
                    G2 = pc_series[1].order()
                    relative_order.insert(0, G1 // G2)

        return PolycyclicGroup(pc_sequence, pc_series, relative_order, collector=None)
```
### 30 - sympy/combinatorics/perm_groups.py:

Start line: 5211, End line: 5228

```python
class Coset(Basic):

    @property
    def is_left_coset(self):
        """
        Check if the coset is left coset that is ``gH``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="-")
        >>> cst.is_left_coset
        True

        """
        return str(self._dir) == '-'
```
### 34 - sympy/combinatorics/perm_groups.py:

Start line: 5096, End line: 5144

```python
class SymmetricPermutationGroup(Basic):

    def order(self):
        """
        Return the order of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.order()
        24
        """
        if self._order is not None:
            return self._order
        n = self._deg
        self._order = factorial(n)
        return self._order

    @property
    def degree(self):
        """
        Return the degree of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.degree
        4

        """
        return self._deg

    @property
    def identity(self):
        '''
        Return the identity element of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.identity()
        (3)

        '''
        return _af_new(list(range(self._deg)))
```
### 41 - sympy/combinatorics/perm_groups.py:

Start line: 2254, End line: 2940

```python
class PermutationGroup(Basic):

    def is_subgroup(self, G, strict=True):
        """Return ``True`` if all elements of ``self`` belong to ``G``.

        If ``strict`` is ``False`` then if ``self``'s degree is smaller
        than ``G``'s, the elements will be resized to have the same degree.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ...    CyclicGroup)

        Testing is strict by default: the degree of each group must be the
        same:

        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G1 = PermutationGroup([Permutation(0, 1, 2), Permutation(0, 1)])
        >>> G2 = PermutationGroup([Permutation(0, 2), Permutation(0, 1, 2)])
        >>> G3 = PermutationGroup([p, p**2])
        >>> assert G1.order() == G2.order() == G3.order() == 6
        >>> G1.is_subgroup(G2)
        True
        >>> G1.is_subgroup(G3)
        False
        >>> G3.is_subgroup(PermutationGroup(G3[1]))
        False
        >>> G3.is_subgroup(PermutationGroup(G3[0]))
        True

        To ignore the size, set ``strict`` to ``False``:

        >>> S3 = SymmetricGroup(3)
        >>> S5 = SymmetricGroup(5)
        >>> S3.is_subgroup(S5, strict=False)
        True
        >>> C7 = CyclicGroup(7)
        >>> G = S5*C7
        >>> S5.is_subgroup(G, False)
        True
        >>> C7.is_subgroup(G, 0)
        False

        """
        if isinstance(G, SymmetricPermutationGroup):
            if self.degree != G.degree:
                return False
            return True
        if not isinstance(G, PermutationGroup):
            return False
        if self == G or self.generators[0]==Permutation():
            return True
        if G.order() % self.order() != 0:
            return False
        if self.degree == G.degree or \
                (self.degree < G.degree and not strict):
            gens = self.generators
        else:
            return False
        return all(G.contains(g, strict=strict) for g in gens)

    @property
    def is_polycyclic(self):
        """Return ``True`` if a group is polycyclic. A group is polycyclic if
        it has a subnormal series with cyclic factors. For finite groups,
        this is the same as if the group is solvable.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G = PermutationGroup([a, b])
        >>> G.is_polycyclic
        True

        """
        return self.is_solvable

    def is_transitive(self, strict=True):
        """Test if the group is transitive.

        A group is transitive if it has a single orbit.

        If ``strict`` is ``False`` the group is transitive if it has
        a single orbit of length different from 1.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G1 = PermutationGroup([a, b])
        >>> G1.is_transitive()
        False
        >>> G1.is_transitive(strict=False)
        True
        >>> c = Permutation([2, 3, 0, 1])
        >>> G2 = PermutationGroup([a, c])
        >>> G2.is_transitive()
        True
        >>> d = Permutation([1, 0, 2, 3])
        >>> e = Permutation([0, 1, 3, 2])
        >>> G3 = PermutationGroup([d, e])
        >>> G3.is_transitive() or G3.is_transitive(strict=False)
        False

        """
        if self._is_transitive:  # strict or not, if True then True
            return self._is_transitive
        if strict:
            if self._is_transitive is not None:  # we only store strict=True
                return self._is_transitive

            ans = len(self.orbit(0)) == self.degree
            self._is_transitive = ans
            return ans

        got_orb = False
        for x in self.orbits():
            if len(x) > 1:
                if got_orb:
                    return False
                got_orb = True
        return got_orb

    @property
    def is_trivial(self):
        """Test if the group is the trivial group.

        This is true if the group contains only the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 1, 2])])
        >>> G.is_trivial
        True

        """
        if self._is_trivial is None:
            self._is_trivial = len(self) == 1 and self[0].is_Identity
        return self._is_trivial

    def lower_central_series(self):
        r"""Return the lower central series for the group.

        The lower central series for a group `G` is the series
        `G = G_0 > G_1 > G_2 > \ldots` where
        `G_k = [G, G_{k-1}]`, i.e. every term after the first is equal to the
        commutator of `G` and the previous term in `G1` ([1], p.29).

        Returns
        =======

        A list of permutation groups in the order `G = G_0, G_1, G_2, \ldots`

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (AlternatingGroup,
        ... DihedralGroup)
        >>> A = AlternatingGroup(4)
        >>> len(A.lower_central_series())
        2
        >>> A.lower_central_series()[1].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        commutator, derived_series

        """
        res = [self]
        current = self
        next = self.commutator(self, current)
        while not current.is_subgroup(next):
            res.append(next)
            current = next
            next = self.commutator(self, current)
        return res

    @property
    def max_div(self):
        """Maximum proper divisor of the degree of a permutation group.

        Notes
        =====

        Obviously, this is the degree divided by its minimal proper divisor
        (larger than ``1``, if one exists). As it is guaranteed to be prime,
        the ``sieve`` from ``sympy.ntheory`` is used.
        This function is also used as an optimization tool for the functions
        ``minimal_block`` and ``_union_find_merge``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 2, 1, 3])])
        >>> G.max_div
        2

        See Also
        ========

        minimal_block, _union_find_merge

        """
        if self._max_div is not None:
            return self._max_div
        n = self.degree
        if n == 1:
            return 1
        for x in sieve:
            if n % x == 0:
                d = n//x
                self._max_div = d
                return d

    def minimal_block(self, points):
        r"""For a transitive group, finds the block system generated by
        ``points``.

        If a group ``G`` acts on a set ``S``, a nonempty subset ``B`` of ``S``
        is called a block under the action of ``G`` if for all ``g`` in ``G``
        we have ``gB = B`` (``g`` fixes ``B``) or ``gB`` and ``B`` have no
        common points (``g`` moves ``B`` entirely). ([1], p.23; [6]).

        The distinct translates ``gB`` of a block ``B`` for ``g`` in ``G``
        partition the set ``S`` and this set of translates is known as a block
        system. Moreover, we obviously have that all blocks in the partition
        have the same size, hence the block size divides ``|S|`` ([1], p.23).
        A ``G``-congruence is an equivalence relation ``~`` on the set ``S``
        such that ``a ~ b`` implies ``g(a) ~ g(b)`` for all ``g`` in ``G``.
        For a transitive group, the equivalence classes of a ``G``-congruence
        and the blocks of a block system are the same thing ([1], p.23).

        The algorithm below checks the group for transitivity, and then finds
        the ``G``-congruence generated by the pairs ``(p_0, p_1), (p_0, p_2),
        ..., (p_0,p_{k-1})`` which is the same as finding the maximal block
        system (i.e., the one with minimum block size) such that
        ``p_0, ..., p_{k-1}`` are in the same block ([1], p.83).

        It is an implementation of Atkinson's algorithm, as suggested in [1],
        and manipulates an equivalence relation on the set ``S`` using a
        union-find data structure. The running time is just above
        `O(|points||S|)`. ([1], pp. 83-87; [7]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.minimal_block([0, 5])
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        >>> D.minimal_block([0, 1])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        See Also
        ========

        _union_find_rep, _union_find_merge, is_transitive, is_primitive

        """
        if not self.is_transitive():
            return False
        n = self.degree
        gens = self.generators
        # initialize the list of equivalence class representatives
        parents = list(range(n))
        ranks = [1]*n
        not_rep = []
        k = len(points)
        # the block size must divide the degree of the group
        if k > self.max_div:
            return [0]*n
        for i in range(k - 1):
            parents[points[i + 1]] = points[0]
            not_rep.append(points[i + 1])
        ranks[points[0]] = k
        i = 0
        len_not_rep = k - 1
        while i < len_not_rep:
            gamma = not_rep[i]
            i += 1
            for gen in gens:
                # find has side effects: performs path compression on the list
                # of representatives
                delta = self._union_find_rep(gamma, parents)
                # union has side effects: performs union by rank on the list
                # of representatives
                temp = self._union_find_merge(gen(gamma), gen(delta), ranks,
                                              parents, not_rep)
                if temp == -1:
                    return [0]*n
                len_not_rep += temp
        for i in range(n):
            # force path compression to get the final state of the equivalence
            # relation
            self._union_find_rep(i, parents)

        # rewrite result so that block representatives are minimal
        new_reps = {}
        return [new_reps.setdefault(r, i) for i, r in enumerate(parents)]

    def conjugacy_class(self, x):
        r"""Return the conjugacy class of an element in the group.

        The conjugacy class of an element ``g`` in a group ``G`` is the set of
        elements ``x`` in ``G`` that are conjugate with ``g``, i.e. for which

            ``g = xax^{-1}``

        for some ``a`` in ``G``.

        Note that conjugacy is an equivalence relation, and therefore that
        conjugacy classes are partitions of ``G``. For a list of all the
        conjugacy classes of the group, use the conjugacy_classes() method.

        In a permutation group, each conjugacy class corresponds to a particular
        `cycle structure': for example, in ``S_3``, the conjugacy classes are:

            * the identity class, ``{()}``
            * all transpositions, ``{(1 2), (1 3), (2 3)}``
            * all 3-cycles, ``{(1 2 3), (1 3 2)}``

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S3 = SymmetricGroup(3)
        >>> S3.conjugacy_class(Permutation(0, 1, 2))
        {(0 1 2), (0 2 1)}

        Notes
        =====

        This procedure computes the conjugacy class directly by finding the
        orbit of the element under conjugation in G. This algorithm is only
        feasible for permutation groups of relatively small order, but is like
        the orbit() function itself in that respect.
        """
        # Ref: "Computing the conjugacy classes of finite groups"; Butler, G.
        # Groups '93 Galway/St Andrews; edited by Campbell, C. M.
        new_class = {x}
        last_iteration = new_class

        while len(last_iteration) > 0:
            this_iteration = set()

            for y in last_iteration:
                for s in self.generators:
                    conjugated = s * y * (~s)
                    if conjugated not in new_class:
                        this_iteration.add(conjugated)

            new_class.update(last_iteration)
            last_iteration = this_iteration

        return new_class


    def conjugacy_classes(self):
        r"""Return the conjugacy classes of the group.

        As described in the documentation for the .conjugacy_class() function,
        conjugacy is an equivalence relation on a group G which partitions the
        set of elements. This method returns a list of all these conjugacy
        classes of G.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> SymmetricGroup(3).conjugacy_classes()
        [{(2)}, {(0 1 2), (0 2 1)}, {(0 2), (1 2), (2)(0 1)}]

        """
        identity = _af_new(list(range(self.degree)))
        known_elements = {identity}
        classes = [known_elements.copy()]

        for x in self.generate():
            if x not in known_elements:
                new_class = self.conjugacy_class(x)
                classes.append(new_class)
                known_elements.update(new_class)

        return classes

    def normal_closure(self, other, k=10):
        r"""Return the normal closure of a subgroup/set of permutations.

        If ``S`` is a subset of a group ``G``, the normal closure of ``A`` in ``G``
        is defined as the intersection of all normal subgroups of ``G`` that
        contain ``A`` ([1], p.14). Alternatively, it is the group generated by
        the conjugates ``x^{-1}yx`` for ``x`` a generator of ``G`` and ``y`` a
        generator of the subgroup ``\left\langle S\right\rangle`` generated by
        ``S`` (for some chosen generating set for ``\left\langle S\right\rangle``)
        ([1], p.73).

        Parameters
        ==========

        other
            a subgroup/list of permutations/single permutation
        k
            an implementation-specific parameter that determines the number
            of conjugates that are adjoined to ``other`` at once

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup, AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> C = CyclicGroup(5)
        >>> G = S.normal_closure(C)
        >>> G.order()
        60
        >>> G.is_subgroup(AlternatingGroup(5))
        True

        See Also
        ========

        commutator, derived_subgroup, random_pr

        Notes
        =====

        The algorithm is described in [1], pp. 73-74; it makes use of the
        generation of random elements for permutation groups by the product
        replacement algorithm.

        """
        if hasattr(other, 'generators'):
            degree = self.degree
            identity = _af_new(list(range(degree)))

            if all(g == identity for g in other.generators):
                return other
            Z = PermutationGroup(other.generators[:])
            base, strong_gens = Z.schreier_sims_incremental()
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            basic_orbits, basic_transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)

            self._random_pr_init(r=10, n=20)

            _loop = True
            while _loop:
                Z._random_pr_init(r=10, n=10)
                for i in range(k):
                    g = self.random_pr()
                    h = Z.random_pr()
                    conj = h^g
                    res = _strip(conj, base, basic_orbits, basic_transversals)
                    if res[0] != identity or res[1] != len(base) + 1:
                        gens = Z.generators
                        gens.append(conj)
                        Z = PermutationGroup(gens)
                        strong_gens.append(conj)
                        temp_base, temp_strong_gens = \
                            Z.schreier_sims_incremental(base, strong_gens)
                        base, strong_gens = temp_base, temp_strong_gens
                        strong_gens_distr = \
                            _distribute_gens_by_base(base, strong_gens)
                        basic_orbits, basic_transversals = \
                            _orbits_transversals_from_bsgs(base,
                                strong_gens_distr)
                _loop = False
                for g in self.generators:
                    for h in Z.generators:
                        conj = h^g
                        res = _strip(conj, base, basic_orbits,
                                     basic_transversals)
                        if res[0] != identity or res[1] != len(base) + 1:
                            _loop = True
                            break
                    if _loop:
                        break
            return Z
        elif hasattr(other, '__getitem__'):
            return self.normal_closure(PermutationGroup(other))
        elif hasattr(other, 'array_form'):
            return self.normal_closure(PermutationGroup([other]))

    def orbit(self, alpha, action='tuples'):
        r"""Compute the orbit of alpha `\{g(\alpha) | g \in G\}` as a set.

        The time complexity of the algorithm used here is `O(|Orb|*r)` where
        `|Orb|` is the size of the orbit and ``r`` is the number of generators of
        the group. For a more detailed analysis, see [1], p.78, [2], pp. 19-21.
        Here alpha can be a single point, or a list of points.

        If alpha is a single point, the ordinary orbit is computed.
        if alpha is a list of points, there are three available options:

        'union' - computes the union of the orbits of the points in the list
        'tuples' - computes the orbit of the list interpreted as an ordered
        tuple under the group action ( i.e., g((1,2,3)) = (g(1), g(2), g(3)) )
        'sets' - computes the orbit of the list interpreted as a sets

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
        >>> G = PermutationGroup([a])
        >>> G.orbit(0)
        {0, 1, 2}
        >>> G.orbit([0, 4], 'union')
        {0, 1, 2, 3, 4, 5, 6}

        See Also
        ========

        orbit_transversal

        """
        return _orbit(self.degree, self.generators, alpha, action)

    def orbit_rep(self, alpha, beta, schreier_vector=None):
        """Return a group element which sends ``alpha`` to ``beta``.

        If ``beta`` is not in the orbit of ``alpha``, the function returns
        ``False``. This implementation makes use of the schreier vector.
        For a proof of correctness, see [1], p.80

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> G = AlternatingGroup(5)
        >>> G.orbit_rep(0, 4)
        (0 4 1 2 3)

        See Also
        ========

        schreier_vector

        """
        if schreier_vector is None:
            schreier_vector = self.schreier_vector(alpha)
        if schreier_vector[beta] is None:
            return False
        k = schreier_vector[beta]
        gens = [x._array_form for x in self.generators]
        a = []
        while k != -1:
            a.append(gens[k])
            beta = gens[k].index(beta) # beta = (~gens[k])(beta)
            k = schreier_vector[beta]
        if a:
            return _af_new(_af_rmuln(*a))
        else:
            return _af_new(list(range(self._degree)))

    def orbit_transversal(self, alpha, pairs=False):
        r"""Computes a transversal for the orbit of ``alpha`` as a set.

        For a permutation group `G`, a transversal for the orbit
        `Orb = \{g(\alpha) | g \in G\}` is a set
        `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
        Note that there may be more than one possible transversal.
        If ``pairs`` is set to ``True``, it returns the list of pairs
        `(\beta, g_\beta)`. For a proof of correctness, see [1], p.79

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.orbit_transversal(0)
        [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]

        See Also
        ========

        orbit

        """
        return _orbit_transversal(self._degree, self.generators, alpha, pairs)

    def orbits(self, rep=False):
        """Return the orbits of ``self``, ordered according to lowest element
        in each orbit.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation(1, 5)(2, 3)(4, 0, 6)
        >>> b = Permutation(1, 5)(3, 4)(2, 6, 0)
        >>> G = PermutationGroup([a, b])
        >>> G.orbits()
        [{0, 2, 3, 4, 6}, {1, 5}]
        """
        return _orbits(self._degree, self._generators)

    def order(self):
        """Return the order of the group: the number of permutations that
        can be generated from elements of the group.

        The number of permutations comprising the group is given by
        ``len(group)``; the length of each permutation in the group is
        given by ``group.size``.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup

        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.order()
        6

        See Also
        ========

        degree

        """
        if self._order is not None:
            return self._order
        if self._is_sym:
            n = self._degree
            self._order = factorial(n)
            return self._order
        if self._is_alt:
            n = self._degree
            self._order = factorial(n)/2
            return self._order

        basic_transversals = self.basic_transversals
        m = 1
        for x in basic_transversals:
            m *= len(x)
        self._order = m
        return m

    def index(self, H):
        """
        Returns the index of a permutation group.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> a = Permutation(1,2,3)
        >>> b =Permutation(3)
        >>> G = PermutationGroup([a])
        >>> H = PermutationGroup([b])
        >>> G.index(H)
        3

        """
        if H.is_subgroup(self):
            return self.order()//H.order()
```
### 47 - sympy/combinatorics/perm_groups.py:

Start line: 5230, End line: 5263

```python
class Coset(Basic):

    @property
    def is_right_coset(self):
        """
        Check if the coset is right coset that is ``Hg``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="+")
        >>> cst.is_right_coset
        True

        """
        return str(self._dir) == '+'

    def as_list(self):
        """
        Return all the elements of coset in the form of list.
        """
        g = self.args[0]
        H = self.args[1]
        cst = []
        if str(self._dir) == '+':
            for h in H.elements:
                cst.append(h*g)
        else:
            for h in H.elements:
                cst.append(g*h)
        return cst
```
### 59 - sympy/combinatorics/perm_groups.py:

Start line: 4925, End line: 4952

```python
def _orbits(degree, generators):
    """Compute the orbits of G.

    If ``rep=False`` it returns a list of sets else it returns a list of
    representatives of the orbits

    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.perm_groups import _orbits
    >>> a = Permutation([0, 2, 1])
    >>> b = Permutation([1, 0, 2])
    >>> _orbits(a.size, [a, b])
    [{0, 1, 2}]
    """

    orbs = []
    sorted_I = list(range(degree))
    I = set(sorted_I)
    while I:
        i = sorted_I[0]
        orb = _orbit(degree, generators, i)
        orbs.append(orb)
        # remove all indices that are in this orbit
        I -= orb
        sorted_I = [i for i in sorted_I if i not in orb]
    return orbs
```
