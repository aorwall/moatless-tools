# astropy__astropy-14590

| **astropy/astropy** | `5f74eacbcc7fff707a44d8eb58adaa514cb7dcb5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6870 |
| **Any found context length** | 176 |
| **Avg pos** | 31.0 |
| **Min pos** | 1 |
| **Max pos** | 21 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/utils/masked/core.py b/astropy/utils/masked/core.py
--- a/astropy/utils/masked/core.py
+++ b/astropy/utils/masked/core.py
@@ -671,20 +671,35 @@ def __ne__(self, other):
         )
         return result.any(axis=-1)
 
-    def _combine_masks(self, masks, out=None):
+    def _combine_masks(self, masks, out=None, where=True, copy=True):
+        """Combine masks, possibly storing it in some output.
+
+        Parameters
+        ----------
+        masks : tuple of array of bool or None
+            Input masks.  Any that are `None` or `False` are ignored.
+            Should broadcast to each other.
+        out : output mask array, optional
+            Possible output array to hold the result.
+        where : array of bool, optional
+            Which elements of the output array to fill.
+        copy : bool optional
+            Whether to ensure a copy is made. Only relevant if a single
+            input mask is not `None`, and ``out`` is not given.
+        """
         masks = [m for m in masks if m is not None and m is not False]
         if not masks:
             return False
         if len(masks) == 1:
             if out is None:
-                return masks[0].copy()
+                return masks[0].copy() if copy else masks[0]
             else:
-                np.copyto(out, masks[0])
+                np.copyto(out, masks[0], where=where)
                 return out
 
-        out = np.logical_or(masks[0], masks[1], out=out)
+        out = np.logical_or(masks[0], masks[1], out=out, where=where)
         for mask in masks[2:]:
-            np.logical_or(out, mask, out=out)
+            np.logical_or(out, mask, out=out, where=where)
         return out
 
     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
@@ -701,6 +716,15 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                 elif out_mask is None:
                     out_mask = m
 
+        # TODO: where is only needed for __call__ and reduce;
+        # this is very fast, but still worth separating out?
+        where = kwargs.pop("where", True)
+        if where is True:
+            where_unmasked = True
+            where_mask = None
+        else:
+            where_unmasked, where_mask = self._get_data_and_mask(where)
+
         unmasked, masks = self._get_data_and_masks(*inputs)
 
         if ufunc.signature:
@@ -731,7 +755,7 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                         else np.logical_or.reduce(mask1)
                     )
 
-                mask = self._combine_masks(masks, out=out_mask)
+                mask = self._combine_masks(masks, out=out_mask, copy=False)
 
             else:
                 # Parse signature with private numpy function. Note it
@@ -769,7 +793,11 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
 
         elif method == "__call__":
             # Regular ufunc call.
-            mask = self._combine_masks(masks, out=out_mask)
+            # Combine the masks from the input, possibly selecting elements.
+            mask = self._combine_masks(masks, out=out_mask, where=where_unmasked)
+            # If relevant, also mask output elements for which where was masked.
+            if where_mask is not None:
+                mask |= where_mask
 
         elif method == "outer":
             # Must have two arguments; adjust masks as will be done for data.
@@ -779,51 +807,50 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
 
         elif method in {"reduce", "accumulate"}:
             # Reductions like np.add.reduce (sum).
-            if masks[0] is not None:
+            # Treat any masked where as if the input element was masked.
+            mask = self._combine_masks((masks[0], where_mask), copy=False)
+            if mask is not False:
                 # By default, we simply propagate masks, since for
                 # things like np.sum, it makes no sense to do otherwise.
                 # Individual methods need to override as needed.
-                # TODO: take care of 'out' too?
                 if method == "reduce":
                     axis = kwargs.get("axis", None)
                     keepdims = kwargs.get("keepdims", False)
-                    where = kwargs.get("where", True)
                     mask = np.logical_or.reduce(
-                        masks[0],
-                        where=where,
+                        mask,
+                        where=where_unmasked,
                         axis=axis,
                         keepdims=keepdims,
                         out=out_mask,
                     )
-                    if where is not True:
-                        # Mask also whole rows that were not selected by where,
-                        # so would have been left as unmasked above.
-                        mask |= np.logical_and.reduce(
-                            masks[0], where=where, axis=axis, keepdims=keepdims
+                    if where_unmasked is not True:
+                        # Mask also whole rows in which no elements were selected;
+                        # those will have been left as unmasked above.
+                        mask |= ~np.logical_or.reduce(
+                            where_unmasked, axis=axis, keepdims=keepdims
                         )
 
                 else:
                     # Accumulate
                     axis = kwargs.get("axis", 0)
-                    mask = np.logical_or.accumulate(masks[0], axis=axis, out=out_mask)
+                    mask = np.logical_or.accumulate(mask, axis=axis, out=out_mask)
 
-            elif out is not None:
-                mask = False
-
-            else:  # pragma: no cover
+            elif out is None:
                 # Can only get here if neither input nor output was masked, but
-                # perhaps axis or where was masked (in NUMPY_LT_1_21 this is
-                # possible).  We don't support this.
+                # perhaps where was masked (possible in "not NUMPY_LT_1_25" and
+                # in NUMPY_LT_1_21 (latter also allowed axis).
+                # We don't support this.
                 return NotImplemented
 
         elif method in {"reduceat", "at"}:  # pragma: no cover
-            # TODO: implement things like np.add.accumulate (used for cumsum).
             raise NotImplementedError(
                 "masked instances cannot yet deal with 'reduceat' or 'at'."
             )
 
         if out_unmasked is not None:
             kwargs["out"] = out_unmasked
+        if where_unmasked is not True:
+            kwargs["where"] = where_unmasked
         result = getattr(ufunc, method)(*unmasked, **kwargs)
 
         if result is None:  # pragma: no cover

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/utils/masked/core.py | 674 | 687 | 21 | 1 | 6870
| astropy/utils/masked/core.py | 704 | 704 | 1 | 1 | 176
| astropy/utils/masked/core.py | 734 | 734 | 3 | 1 | 2114
| astropy/utils/masked/core.py | 772 | 772 | 3 | 1 | 2114
| astropy/utils/masked/core.py | 782 | 820 | 3 | 1 | 2114


## Problem Statement

```
TST: np.fix check fails with numpy-dev (TypeError: cannot write to unmasked output)
Started popping up in numpy-dev jobs. @mhvk is investigating.

\`\`\`
____________________________ TestUfuncLike.test_fix ____________________________

self = <astropy.utils.masked.tests.test_function_helpers.TestUfuncLike object at 0x7fdd354916c0>

    def test_fix(self):
>       self.check(np.fix)

astropy/utils/masked/tests/test_function_helpers.py:672: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
astropy/utils/masked/tests/test_function_helpers.py:75: in check
    o = func(self.ma, *args, **kwargs)
astropy/utils/masked/core.py:842: in __array_function__
    return super().__array_function__(function, types, args, kwargs)
numpy/lib/ufunclike.py:62: in fix
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))
astropy/utils/masked/core.py:828: in __array_ufunc__
    result = getattr(ufunc, method)(*unmasked, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = MaskedNDArray([[  ———,  True,  True],
               [ True,   ———,  True]])
ufunc = <ufunc 'floor'>, method = '__call__'
inputs = (array([[0., 1., 2.],
       [3., 4., 5.]]),)
kwargs = {'where': MaskedNDArray([[  ———,  True,  True],
               [ True,   ———,  True]])}
out = (array([[0., 1., 2.],
       [3., 4., 5.]]),)
out_unmasked = (array([[0., 1., 2.],
       [3., 4., 5.]]),), out_mask = None
out_masks = (None,), d = array([[0., 1., 2.],
       [3., 4., 5.]]), m = None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)
        out_unmasked = None
        out_mask = None
        if out is not None:
            out_unmasked, out_masks = self._get_data_and_masks(*out)
            for d, m in zip(out_unmasked, out_masks):
                if m is None:
                    # TODO: allow writing to unmasked output if nothing is masked?
                    if d is not None:
>                       raise TypeError("cannot write to unmasked output")
E                       TypeError: cannot write to unmasked output

astropy/utils/masked/core.py:701: TypeError
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 astropy/utils/masked/core.py** | 690 | 704| 176 | 176 | 10404 | 
| 2 | 2 astropy/utils/masked/function_helpers.py | 84 | 161| 756 | 932 | 19311 | 
| **-> 3 <-** | **2 astropy/utils/masked/core.py** | 706 | 835| 1182 | 2114 | 19311 | 
| 4 | **2 astropy/utils/masked/core.py** | 837 | 883| 379 | 2493 | 19311 | 
| 5 | **2 astropy/utils/masked/core.py** | 941 | 999| 645 | 3138 | 19311 | 
| 6 | **2 astropy/utils/masked/core.py** | 885 | 897| 181 | 3319 | 19311 | 
| 7 | **2 astropy/utils/masked/core.py** | 18 | 40| 137 | 3456 | 19311 | 
| 8 | 2 astropy/utils/masked/function_helpers.py | 1023 | 1040| 162 | 3618 | 19311 | 
| 9 | **2 astropy/utils/masked/core.py** | 1130 | 1153| 259 | 3877 | 19311 | 
| 10 | 2 astropy/utils/masked/function_helpers.py | 12 | 83| 537 | 4414 | 19311 | 
| 11 | **2 astropy/utils/masked/core.py** | 643 | 672| 302 | 4716 | 19311 | 
| 12 | **2 astropy/utils/masked/core.py** | 899 | 923| 267 | 4983 | 19311 | 
| 13 | **2 astropy/utils/masked/core.py** | 1155 | 1192| 357 | 5340 | 19311 | 
| 14 | **2 astropy/utils/masked/core.py** | 227 | 244| 173 | 5513 | 19311 | 
| 15 | 3 astropy/cosmology/funcs/comparison.py | 62 | 70| 135 | 5648 | 22834 | 
| 16 | 3 astropy/utils/masked/function_helpers.py | 1042 | 1063| 187 | 5835 | 22834 | 
| 17 | **3 astropy/utils/masked/core.py** | 1098 | 1128| 317 | 6152 | 22834 | 
| 18 | 3 astropy/utils/masked/function_helpers.py | 1066 | 1082| 133 | 6285 | 22834 | 
| 19 | **3 astropy/utils/masked/core.py** | 619 | 641| 241 | 6526 | 22834 | 
| 20 | **3 astropy/utils/masked/core.py** | 280 | 300| 200 | 6726 | 22834 | 
| **-> 21 <-** | **3 astropy/utils/masked/core.py** | 674 | 688| 144 | 6870 | 22834 | 
| 22 | 3 astropy/utils/masked/function_helpers.py | 476 | 498| 303 | 7173 | 22834 | 
| 23 | 3 astropy/utils/masked/function_helpers.py | 850 | 876| 218 | 7391 | 22834 | 
| 24 | 3 astropy/utils/masked/function_helpers.py | 386 | 399| 136 | 7527 | 22834 | 
| 25 | **3 astropy/utils/masked/core.py** | 581 | 598| 193 | 7720 | 22834 | 
| 26 | 4 astropy/nddata/compat.py | 159 | 180| 197 | 7917 | 24977 | 
| 27 | **4 astropy/utils/masked/core.py** | 1195 | 1220| 237 | 8154 | 24977 | 
| 28 | 5 astropy/io/votable/tree.py | 5 | 132| 590 | 8744 | 54371 | 
| 29 | 6 astropy/time/core.py | 3215 | 3249| 310 | 9054 | 82633 | 
| 30 | 7 astropy/nddata/utils.py | 6 | 37| 157 | 9211 | 90169 | 
| 31 | 7 astropy/utils/masked/function_helpers.py | 903 | 932| 300 | 9511 | 90169 | 
| 32 | 7 astropy/utils/masked/function_helpers.py | 654 | 669| 156 | 9667 | 90169 | 
| 33 | 8 astropy/units/quantity_helper/function_helpers.py | 102 | 133| 366 | 10033 | 100263 | 
| 34 | 8 astropy/utils/masked/function_helpers.py | 164 | 231| 599 | 10632 | 100263 | 
| 35 | 9 astropy/utils/masked/__init__.py | 1 | 11| 0 | 10632 | 100347 | 
| 36 | 9 astropy/utils/masked/function_helpers.py | 577 | 596| 162 | 10794 | 100347 | 
| 37 | 10 astropy/utils/compat/numpycompat.py | 7 | 29| 246 | 11040 | 100634 | 
| 38 | 11 astropy/conftest.py | 7 | 61| 368 | 11408 | 101793 | 
| 39 | 11 astropy/units/quantity_helper/function_helpers.py | 186 | 272| 771 | 12179 | 101793 | 
| 40 | 12 astropy/time/time_helper/function_helpers.py | 1 | 36| 243 | 12422 | 102037 | 
| 41 | **12 astropy/utils/masked/core.py** | 246 | 278| 204 | 12626 | 102037 | 
| 42 | 13 astropy/table/operations.py | 1598 | 1635| 315 | 12941 | 116523 | 
| 43 | 13 astropy/utils/masked/function_helpers.py | 330 | 345| 142 | 13083 | 116523 | 
| 44 | 13 astropy/utils/masked/function_helpers.py | 630 | 651| 224 | 13307 | 116523 | 
| 45 | **13 astropy/utils/masked/core.py** | 341 | 375| 304 | 13611 | 116523 | 
| 46 | **13 astropy/utils/masked/core.py** | 511 | 554| 350 | 13961 | 116523 | 
| 47 | 13 astropy/utils/masked/function_helpers.py | 672 | 691| 139 | 14100 | 116523 | 
| 48 | 14 astropy/units/quantity_helper/helpers.py | 425 | 501| 742 | 14842 | 120378 | 
| 49 | 15 astropy/nddata/nduncertainty.py | 3 | 27| 141 | 14983 | 129980 | 
| 50 | 16 astropy/io/votable/converters.py | 80 | 91| 114 | 15097 | 139920 | 
| 51 | 17 astropy/nddata/ccddata.py | 4 | 31| 217 | 15314 | 146915 | 
| 52 | **17 astropy/utils/masked/core.py** | 394 | 410| 126 | 15440 | 146915 | 
| 53 | 17 astropy/utils/masked/function_helpers.py | 428 | 448| 135 | 15575 | 146915 | 
| 54 | 18 astropy/nddata/bitmask.py | 1 | 23| 117 | 15692 | 153646 | 
| 55 | 19 astropy/io/fits/hdu/base.py | 4 | 82| 457 | 16149 | 167106 | 
| 56 | 20 astropy/io/misc/asdf/tags/fits/fits.py | 2 | 35| 251 | 16400 | 167870 | 
| 57 | 20 astropy/nddata/ccddata.py | 368 | 444| 815 | 17215 | 167870 | 
| 58 | **20 astropy/utils/masked/core.py** | 70 | 80| 131 | 17346 | 167870 | 
| 59 | 21 astropy/io/misc/asdf/tags/transform/polynomial.py | 2 | 42| 282 | 17628 | 170717 | 
| 60 | 21 astropy/units/quantity_helper/helpers.py | 338 | 424| 758 | 18386 | 170717 | 
| 61 | 21 astropy/utils/masked/function_helpers.py | 451 | 473| 186 | 18572 | 170717 | 
| 62 | 22 astropy/io/fits/diff.py | 8 | 55| 266 | 18838 | 183806 | 
| 63 | 23 astropy/nddata/blocks.py | 6 | 33| 211 | 19049 | 185637 | 
| 64 | **23 astropy/utils/masked/core.py** | 110 | 125| 174 | 19223 | 185637 | 
| 65 | 23 astropy/utils/masked/function_helpers.py | 1013 | 1020| 108 | 19331 | 185637 | 
| 66 | 24 astropy/table/pprint.py | 3 | 35| 189 | 19520 | 192072 | 


### Hint

```
Ah, yes, that was https://github.com/numpy/numpy/pull/23240 and we actually checked in that discussion whether it would pose problems for astropy - https://github.com/numpy/numpy/pull/23240#discussion_r1112314891 - conclusion was that only `np.fix` was affected and that it would be a trivial fix. I'll make that now...
```

## Patch

```diff
diff --git a/astropy/utils/masked/core.py b/astropy/utils/masked/core.py
--- a/astropy/utils/masked/core.py
+++ b/astropy/utils/masked/core.py
@@ -671,20 +671,35 @@ def __ne__(self, other):
         )
         return result.any(axis=-1)
 
-    def _combine_masks(self, masks, out=None):
+    def _combine_masks(self, masks, out=None, where=True, copy=True):
+        """Combine masks, possibly storing it in some output.
+
+        Parameters
+        ----------
+        masks : tuple of array of bool or None
+            Input masks.  Any that are `None` or `False` are ignored.
+            Should broadcast to each other.
+        out : output mask array, optional
+            Possible output array to hold the result.
+        where : array of bool, optional
+            Which elements of the output array to fill.
+        copy : bool optional
+            Whether to ensure a copy is made. Only relevant if a single
+            input mask is not `None`, and ``out`` is not given.
+        """
         masks = [m for m in masks if m is not None and m is not False]
         if not masks:
             return False
         if len(masks) == 1:
             if out is None:
-                return masks[0].copy()
+                return masks[0].copy() if copy else masks[0]
             else:
-                np.copyto(out, masks[0])
+                np.copyto(out, masks[0], where=where)
                 return out
 
-        out = np.logical_or(masks[0], masks[1], out=out)
+        out = np.logical_or(masks[0], masks[1], out=out, where=where)
         for mask in masks[2:]:
-            np.logical_or(out, mask, out=out)
+            np.logical_or(out, mask, out=out, where=where)
         return out
 
     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
@@ -701,6 +716,15 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                 elif out_mask is None:
                     out_mask = m
 
+        # TODO: where is only needed for __call__ and reduce;
+        # this is very fast, but still worth separating out?
+        where = kwargs.pop("where", True)
+        if where is True:
+            where_unmasked = True
+            where_mask = None
+        else:
+            where_unmasked, where_mask = self._get_data_and_mask(where)
+
         unmasked, masks = self._get_data_and_masks(*inputs)
 
         if ufunc.signature:
@@ -731,7 +755,7 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                         else np.logical_or.reduce(mask1)
                     )
 
-                mask = self._combine_masks(masks, out=out_mask)
+                mask = self._combine_masks(masks, out=out_mask, copy=False)
 
             else:
                 # Parse signature with private numpy function. Note it
@@ -769,7 +793,11 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
 
         elif method == "__call__":
             # Regular ufunc call.
-            mask = self._combine_masks(masks, out=out_mask)
+            # Combine the masks from the input, possibly selecting elements.
+            mask = self._combine_masks(masks, out=out_mask, where=where_unmasked)
+            # If relevant, also mask output elements for which where was masked.
+            if where_mask is not None:
+                mask |= where_mask
 
         elif method == "outer":
             # Must have two arguments; adjust masks as will be done for data.
@@ -779,51 +807,50 @@ def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
 
         elif method in {"reduce", "accumulate"}:
             # Reductions like np.add.reduce (sum).
-            if masks[0] is not None:
+            # Treat any masked where as if the input element was masked.
+            mask = self._combine_masks((masks[0], where_mask), copy=False)
+            if mask is not False:
                 # By default, we simply propagate masks, since for
                 # things like np.sum, it makes no sense to do otherwise.
                 # Individual methods need to override as needed.
-                # TODO: take care of 'out' too?
                 if method == "reduce":
                     axis = kwargs.get("axis", None)
                     keepdims = kwargs.get("keepdims", False)
-                    where = kwargs.get("where", True)
                     mask = np.logical_or.reduce(
-                        masks[0],
-                        where=where,
+                        mask,
+                        where=where_unmasked,
                         axis=axis,
                         keepdims=keepdims,
                         out=out_mask,
                     )
-                    if where is not True:
-                        # Mask also whole rows that were not selected by where,
-                        # so would have been left as unmasked above.
-                        mask |= np.logical_and.reduce(
-                            masks[0], where=where, axis=axis, keepdims=keepdims
+                    if where_unmasked is not True:
+                        # Mask also whole rows in which no elements were selected;
+                        # those will have been left as unmasked above.
+                        mask |= ~np.logical_or.reduce(
+                            where_unmasked, axis=axis, keepdims=keepdims
                         )
 
                 else:
                     # Accumulate
                     axis = kwargs.get("axis", 0)
-                    mask = np.logical_or.accumulate(masks[0], axis=axis, out=out_mask)
+                    mask = np.logical_or.accumulate(mask, axis=axis, out=out_mask)
 
-            elif out is not None:
-                mask = False
-
-            else:  # pragma: no cover
+            elif out is None:
                 # Can only get here if neither input nor output was masked, but
-                # perhaps axis or where was masked (in NUMPY_LT_1_21 this is
-                # possible).  We don't support this.
+                # perhaps where was masked (possible in "not NUMPY_LT_1_25" and
+                # in NUMPY_LT_1_21 (latter also allowed axis).
+                # We don't support this.
                 return NotImplemented
 
         elif method in {"reduceat", "at"}:  # pragma: no cover
-            # TODO: implement things like np.add.accumulate (used for cumsum).
             raise NotImplementedError(
                 "masked instances cannot yet deal with 'reduceat' or 'at'."
             )
 
         if out_unmasked is not None:
             kwargs["out"] = out_unmasked
+        if where_unmasked is not True:
+            kwargs["where"] = where_unmasked
         result = getattr(ufunc, method)(*unmasked, **kwargs)
 
         if result is None:  # pragma: no cover

```

## Test Patch

```diff
diff --git a/astropy/utils/masked/tests/test_functions.py b/astropy/utils/masked/tests/test_functions.py
--- a/astropy/utils/masked/tests/test_functions.py
+++ b/astropy/utils/masked/tests/test_functions.py
@@ -11,6 +11,7 @@
 
 from astropy import units as u
 from astropy.units import Quantity
+from astropy.utils.compat.numpycompat import NUMPY_LT_1_25
 from astropy.utils.masked.core import Masked
 
 from .test_masked import (
@@ -44,6 +45,57 @@ def test_ufunc_inplace(self, ufunc):
         assert result is out
         assert_masked_equal(result, ma_mb)
 
+    @pytest.mark.parametrize("base_mask", [True, False])
+    def test_ufunc_inplace_where(self, base_mask):
+        # Construct base filled with -9 and base_mask (copying to get unit/class).
+        base = self.ma.copy()
+        base.unmasked.view(np.ndarray)[...] = -9.0
+        base._mask[...] = base_mask
+        out = base.copy()
+        where = np.array([[True, False, False], [False, True, False]])
+        result = np.add(self.ma, self.mb, out=out, where=where)
+        # Direct checks.
+        assert np.all(result.unmasked[~where] == base.unmasked[0, 0])
+        assert np.all(result.unmasked[where] == (self.a + self.b)[where])
+        # Full comparison.
+        expected = base.unmasked.copy()
+        np.add(self.a, self.b, out=expected, where=where)
+        expected_mask = base.mask.copy()
+        np.logical_or(self.mask_a, self.mask_b, out=expected_mask, where=where)
+        assert_array_equal(result.unmasked, expected)
+        assert_array_equal(result.mask, expected_mask)
+
+    @pytest.mark.parametrize("base_mask", [True, False])
+    def test_ufunc_inplace_masked_where(self, base_mask):
+        base = self.ma.copy()
+        base.unmasked.view(np.ndarray)[...] = -9.0
+        base._mask[...] = base_mask
+        out = base.copy()
+        where = Masked(
+            [[True, False, True], [False, False, True]],
+            mask=[[True, False, False], [True, False, True]],
+        )
+        result = np.add(self.ma, self.mb, out=out, where=where)
+        # Direct checks.
+        assert np.all(result.unmasked[~where.unmasked] == base.unmasked[0, 0])
+        assert np.all(
+            result.unmasked[where.unmasked] == (self.a + self.b)[where.unmasked]
+        )
+        assert np.all(result.mask[where.mask])
+        assert np.all(result.mask[~where.mask & ~where.unmasked] == base.mask[0, 0])
+        assert np.all(
+            result.mask[~where.mask & where.unmasked]
+            == (self.mask_a | self.mask_b)[~where.mask & where.unmasked]
+        )
+        # Full comparison.
+        expected = base.unmasked.copy()
+        np.add(self.a, self.b, out=expected, where=where.unmasked)
+        expected_mask = base.mask.copy()
+        np.logical_or(self.mask_a, self.mask_b, out=expected_mask, where=where.unmasked)
+        expected_mask |= where.mask
+        assert_array_equal(result.unmasked, expected)
+        assert_array_equal(result.mask, expected_mask)
+
     def test_ufunc_inplace_no_masked_input(self):
         a_b = np.add(self.a, self.b)
         out = Masked(np.zeros_like(a_b))
@@ -53,10 +105,19 @@ def test_ufunc_inplace_no_masked_input(self):
         assert_array_equal(result.mask, np.zeros(a_b.shape, bool))
 
     def test_ufunc_inplace_error(self):
+        # Output is not masked.
         out = np.zeros(self.ma.shape)
         with pytest.raises(TypeError):
             np.add(self.ma, self.mb, out=out)
 
+    @pytest.mark.xfail(NUMPY_LT_1_25, reason="masked where not supported in numpy<1.25")
+    def test_ufunc_inplace_error_masked_where(self):
+        # Input and output are not masked, but where is.
+        # Note: prior to numpy 1.25, we cannot control this.
+        out = self.a.copy()
+        with pytest.raises(TypeError):
+            np.add(self.a, self.b, out=out, where=Masked(True, mask=True))
+
     @pytest.mark.parametrize("ufunc", (np.add.outer, np.minimum.outer))
     def test_2op_ufunc_outer(self, ufunc):
         ma_mb = ufunc(self.ma, self.mb)

```


## Code snippets

### 1 - astropy/utils/masked/core.py:

Start line: 690, End line: 704

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)
        out_unmasked = None
        out_mask = None
        if out is not None:
            out_unmasked, out_masks = self._get_data_and_masks(*out)
            for d, m in zip(out_unmasked, out_masks):
                if m is None:
                    # TODO: allow writing to unmasked output if nothing is masked?
                    if d is not None:
                        raise TypeError("cannot write to unmasked output")
                elif out_mask is None:
                    out_mask = m

        unmasked, masks = self._get_data_and_masks(*inputs)
        # ... other code
```
### 2 - astropy/utils/masked/function_helpers.py:

Start line: 84, End line: 161

```python
MASKED_SAFE_FUNCTIONS |= {
    # built-in from multiarray
    np.may_share_memory, np.can_cast, np.min_scalar_type, np.result_type,
    np.shares_memory,
    # np.core.arrayprint
    np.array_repr,
    # np.core.function_base
    np.linspace, np.logspace, np.geomspace,
    # np.core.numeric
    np.isclose, np.allclose, np.flatnonzero, np.argwhere,
    # np.core.shape_base
    np.atleast_1d, np.atleast_2d, np.atleast_3d, np.stack, np.hstack, np.vstack,
    # np.lib.function_base
    np.average, np.diff, np.extract, np.meshgrid, np.trapz, np.gradient,
    # np.lib.index_tricks
    np.diag_indices_from, np.triu_indices_from, np.tril_indices_from,
    np.fill_diagonal,
    # np.lib.shape_base
    np.column_stack, np.row_stack, np.dstack,
    np.array_split, np.split, np.hsplit, np.vsplit, np.dsplit,
    np.expand_dims, np.apply_along_axis, np.kron, np.tile,
    np.take_along_axis, np.put_along_axis,
    # np.lib.type_check (all but asfarray, nan_to_num)
    np.iscomplexobj, np.isrealobj, np.imag, np.isreal, np.real,
    np.real_if_close, np.common_type,
    # np.lib.ufunclike
    np.fix, np.isneginf, np.isposinf,
    # np.lib.function_base
    np.angle, np.i0,
}  # fmt: skip
IGNORED_FUNCTIONS = {
    # I/O - useless for Masked, since no way to store the mask.
    np.save, np.savez, np.savetxt, np.savez_compressed,
    # Polynomials
    np.poly, np.polyadd, np.polyder, np.polydiv, np.polyfit, np.polyint,
    np.polymul, np.polysub, np.polyval, np.roots, np.vander,
}  # fmt: skip
IGNORED_FUNCTIONS |= {
    np.pad, np.searchsorted, np.digitize,
    np.is_busday, np.busday_count, np.busday_offset,
    # numpy.lib.function_base
    np.cov, np.corrcoef, np.trim_zeros,
    # numpy.core.numeric
    np.correlate, np.convolve,
    # numpy.lib.histograms
    np.histogram, np.histogram2d, np.histogramdd, np.histogram_bin_edges,
    # TODO!!
    np.dot, np.vdot, np.inner, np.tensordot, np.cross,
    np.einsum, np.einsum_path,
}  # fmt: skip

# Really should do these...
IGNORED_FUNCTIONS |= {
    getattr(np, setopsname) for setopsname in np.lib.arraysetops.__all__
}


if NUMPY_LT_1_23:
    IGNORED_FUNCTIONS |= {
        # Deprecated, removed in numpy 1.23
        np.asscalar,
        np.alen,
    }

# Explicitly unsupported functions
UNSUPPORTED_FUNCTIONS |= {
    np.unravel_index,
    np.ravel_multi_index,
    np.ix_,
}

# No support for the functions also not supported by Quantity
# (io, polynomial, etc.).
UNSUPPORTED_FUNCTIONS |= IGNORED_FUNCTIONS


apply_to_both = FunctionAssigner(APPLY_TO_BOTH_FUNCTIONS)
dispatched_function = FunctionAssigner(DISPATCHED_FUNCTIONS)
```
### 3 - astropy/utils/masked/core.py:

Start line: 706, End line: 835

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # ... other code

        if ufunc.signature:
            # We're dealing with a gufunc. For now, only deal with
            # np.matmul and gufuncs for which the mask of any output always
            # depends on all core dimension values of all inputs.
            # Also ignore axes keyword for now...
            # TODO: in principle, it should be possible to generate the mask
            # purely based on the signature.
            if "axes" in kwargs:
                raise NotImplementedError(
                    "Masked does not yet support gufunc calls with 'axes'."
                )
            if ufunc is np.matmul:
                # np.matmul is tricky and its signature cannot be parsed by
                # _parse_gufunc_signature.
                unmasked = np.atleast_1d(*unmasked)
                mask0, mask1 = masks
                masks = []
                is_mat1 = unmasked[1].ndim >= 2
                if mask0 is not None:
                    masks.append(np.logical_or.reduce(mask0, axis=-1, keepdims=is_mat1))

                if mask1 is not None:
                    masks.append(
                        np.logical_or.reduce(mask1, axis=-2, keepdims=True)
                        if is_mat1
                        else np.logical_or.reduce(mask1)
                    )

                mask = self._combine_masks(masks, out=out_mask)

            else:
                # Parse signature with private numpy function. Note it
                # cannot handle spaces in tuples, so remove those.
                in_sig, out_sig = np.lib.function_base._parse_gufunc_signature(
                    ufunc.signature.replace(" ", "")
                )
                axis = kwargs.get("axis", -1)
                keepdims = kwargs.get("keepdims", False)
                in_masks = []
                for sig, mask in zip(in_sig, masks):
                    if mask is not None:
                        if sig:
                            # Input has core dimensions.  Assume that if any
                            # value in those is masked, the output will be
                            # masked too (TODO: for multiple core dimensions
                            # this may be too strong).
                            mask = np.logical_or.reduce(
                                mask, axis=axis, keepdims=keepdims
                            )
                        in_masks.append(mask)

                mask = self._combine_masks(in_masks)
                result_masks = []
                for os in out_sig:
                    if os:
                        # Output has core dimensions.  Assume all those
                        # get the same mask.
                        result_mask = np.expand_dims(mask, axis)
                    else:
                        result_mask = mask
                    result_masks.append(result_mask)

                mask = result_masks if len(result_masks) > 1 else result_masks[0]

        elif method == "__call__":
            # Regular ufunc call.
            mask = self._combine_masks(masks, out=out_mask)

        elif method == "outer":
            # Must have two arguments; adjust masks as will be done for data.
            assert len(masks) == 2
            masks = [(m if m is not None else False) for m in masks]
            mask = np.logical_or.outer(masks[0], masks[1], out=out_mask)

        elif method in {"reduce", "accumulate"}:
            # Reductions like np.add.reduce (sum).
            if masks[0] is not None:
                # By default, we simply propagate masks, since for
                # things like np.sum, it makes no sense to do otherwise.
                # Individual methods need to override as needed.
                # TODO: take care of 'out' too?
                if method == "reduce":
                    axis = kwargs.get("axis", None)
                    keepdims = kwargs.get("keepdims", False)
                    where = kwargs.get("where", True)
                    mask = np.logical_or.reduce(
                        masks[0],
                        where=where,
                        axis=axis,
                        keepdims=keepdims,
                        out=out_mask,
                    )
                    if where is not True:
                        # Mask also whole rows that were not selected by where,
                        # so would have been left as unmasked above.
                        mask |= np.logical_and.reduce(
                            masks[0], where=where, axis=axis, keepdims=keepdims
                        )

                else:
                    # Accumulate
                    axis = kwargs.get("axis", 0)
                    mask = np.logical_or.accumulate(masks[0], axis=axis, out=out_mask)

            elif out is not None:
                mask = False

            else:  # pragma: no cover
                # Can only get here if neither input nor output was masked, but
                # perhaps axis or where was masked (in NUMPY_LT_1_21 this is
                # possible).  We don't support this.
                return NotImplemented

        elif method in {"reduceat", "at"}:  # pragma: no cover
            # TODO: implement things like np.add.accumulate (used for cumsum).
            raise NotImplementedError(
                "masked instances cannot yet deal with 'reduceat' or 'at'."
            )

        if out_unmasked is not None:
            kwargs["out"] = out_unmasked
        result = getattr(ufunc, method)(*unmasked, **kwargs)

        if result is None:  # pragma: no cover
            # This happens for the "at" method.
            return result

        if out is not None and len(out) == 1:
            out = out[0]
        return self._masked_result(result, mask, out)
```
### 4 - astropy/utils/masked/core.py:

Start line: 837, End line: 883

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def __array_function__(self, function, types, args, kwargs):
        # TODO: go through functions systematically to see which ones
        # work and/or can be supported.
        if function in MASKED_SAFE_FUNCTIONS:
            return super().__array_function__(function, types, args, kwargs)

        elif function in APPLY_TO_BOTH_FUNCTIONS:
            helper = APPLY_TO_BOTH_FUNCTIONS[function]
            try:
                helper_result = helper(*args, **kwargs)
            except NotImplementedError:
                return self._not_implemented_or_raise(function, types)

            data_args, mask_args, kwargs, out = helper_result
            if out is not None:
                if not isinstance(out, Masked):
                    return self._not_implemented_or_raise(function, types)
                function(*mask_args, out=out.mask, **kwargs)
                function(*data_args, out=out.unmasked, **kwargs)
                return out

            mask = function(*mask_args, **kwargs)
            result = function(*data_args, **kwargs)

        elif function in DISPATCHED_FUNCTIONS:
            dispatched_function = DISPATCHED_FUNCTIONS[function]
            try:
                dispatched_result = dispatched_function(*args, **kwargs)
            except NotImplementedError:
                return self._not_implemented_or_raise(function, types)

            if not isinstance(dispatched_result, tuple):
                return dispatched_result

            result, mask, out = dispatched_result

        elif function in UNSUPPORTED_FUNCTIONS:
            return NotImplemented

        else:  # pragma: no cover
            # By default, just pass it through for now.
            return super().__array_function__(function, types, args, kwargs)

        if mask is None:
            return result
        else:
            return self._masked_result(result, mask, out)
```
### 5 - astropy/utils/masked/core.py:

Start line: 941, End line: 999

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        # Unfortunately, cannot override the call to diagonal inside trace, so
        # duplicate implementation in numpy/core/src/multiarray/calculation.c.
        diagonal = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        return diagonal.sum(-1, dtype=dtype, out=out)

    def min(self, axis=None, out=None, **kwargs):
        return super().min(
            axis=axis, out=out, **self._reduce_defaults(kwargs, np.nanmax)
        )

    def max(self, axis=None, out=None, **kwargs):
        return super().max(
            axis=axis, out=out, **self._reduce_defaults(kwargs, np.nanmin)
        )

    def nonzero(self):
        unmasked_nonzero = self.unmasked.nonzero()
        if self.ndim >= 1:
            not_masked = ~self.mask[unmasked_nonzero]
            return tuple(u[not_masked] for u in unmasked_nonzero)
        else:
            return unmasked_nonzero if not self.mask else np.nonzero(0)

    def compress(self, condition, axis=None, out=None):
        if out is not None:
            raise NotImplementedError("cannot yet give output")
        return self._apply("compress", condition, axis=axis)

    def repeat(self, repeats, axis=None):
        return self._apply("repeat", repeats, axis=axis)

    def choose(self, choices, out=None, mode="raise"):
        # Let __array_function__ take care since choices can be masked too.
        return np.choose(self, choices, out=out, mode=mode)

    if NUMPY_LT_1_22:

        def argmin(self, axis=None, out=None):
            # Todo: should this return a masked integer array, with masks
            # if all elements were masked?
            at_min = self == self.min(axis=axis, keepdims=True)
            return at_min.filled(False).argmax(axis=axis, out=out)

        def argmax(self, axis=None, out=None):
            at_max = self == self.max(axis=axis, keepdims=True)
            return at_max.filled(False).argmax(axis=axis, out=out)

    else:

        def argmin(self, axis=None, out=None, *, keepdims=False):
            # Todo: should this return a masked integer array, with masks
            # if all elements were masked?
            at_min = self == self.min(axis=axis, keepdims=True)
            return at_min.filled(False).argmax(axis=axis, out=out, keepdims=keepdims)

        def argmax(self, axis=None, out=None, *, keepdims=False):
            at_max = self == self.max(axis=axis, keepdims=True)
            return at_max.filled(False).argmax(axis=axis, out=out, keepdims=keepdims)
```
### 6 - astropy/utils/masked/core.py:

Start line: 885, End line: 897

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def _not_implemented_or_raise(self, function, types):
        # Our function helper or dispatcher found that the function does not
        # work with Masked.  In principle, there may be another class that
        # knows what to do with us, for which we should return NotImplemented.
        # But if there is ndarray (or a non-Masked subclass of it) around,
        # it quite likely coerces, so we should just break.
        if any(issubclass(t, np.ndarray) and not issubclass(t, Masked) for t in types):
            raise TypeError(
                "the MaskedNDArray implementation cannot handle {} "
                "with the given arguments.".format(function)
            ) from None
        else:
            return NotImplemented
```
### 7 - astropy/utils/masked/core.py:

Start line: 18, End line: 40

```python
import builtins

import numpy as np

from astropy.utils.compat import NUMPY_LT_1_22
from astropy.utils.data_info import ParentDtypeInfo
from astropy.utils.shapes import NDArrayShapeMethods

from .function_helpers import (
    APPLY_TO_BOTH_FUNCTIONS,
    DISPATCHED_FUNCTIONS,
    MASKED_SAFE_FUNCTIONS,
    UNSUPPORTED_FUNCTIONS,
)

__all__ = ["Masked", "MaskedNDArray"]


get__doc__ = """Masked version of {0.__name__}.

Except for the ability to pass in a ``mask``, parameters are
as for `{0.__module__}.{0.__name__}`.
""".format
```
### 8 - astropy/utils/masked/function_helpers.py:

Start line: 1023, End line: 1040

```python
def masked_nanfunc(nanfuncname):
    np_func = getattr(np, nanfuncname[3:])
    fill_value = _nanfunc_fill_values.get(nanfuncname, None)

    def nanfunc(a, *args, **kwargs):
        from astropy.utils.masked import Masked

        a, mask = Masked._get_data_and_mask(a)
        if issubclass(a.dtype.type, np.inexact):
            nans = np.isnan(a)
            mask = nans if mask is None else (nans | mask)

        if mask is not None:
            a = Masked(a, mask)
            if fill_value is not None:
                a = a.filled(fill_value)

        return np_func(a, *args, **kwargs)
    # ... other code
```
### 9 - astropy/utils/masked/core.py:

Start line: 1130, End line: 1153

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def var(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        where_final = ~self.mask & where

        # Simplified implementation based on that in numpy/core/_methods.py
        n = np.add.reduce(where_final, axis=axis, keepdims=keepdims)[...]

        # Cast bool, unsigned int, and int to float64 by default.
        if dtype is None and issubclass(self.dtype.type, (np.integer, np.bool_)):
            dtype = np.dtype("f8")
        mean = self.mean(axis=axis, dtype=dtype, keepdims=True, where=where)

        x = self - mean
        x *= x.conjugate()  # Conjugate just returns x if not complex.

        result = x.sum(
            axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where_final
        )
        n -= ddof
        n = np.maximum(n, 0, out=n)
        result /= n
        result._mask |= n == 0
        return result
```
### 10 - astropy/utils/masked/function_helpers.py:

Start line: 12, End line: 83

```python
import numpy as np

from astropy.units.quantity_helper.function_helpers import FunctionAssigner
from astropy.utils.compat import NUMPY_LT_1_23, NUMPY_LT_1_24

# This module should not really be imported, but we define __all__
# such that sphinx can typeset the functions with docstrings.
# The latter are added to __all__ at the end.
__all__ = [
    "MASKED_SAFE_FUNCTIONS",
    "APPLY_TO_BOTH_FUNCTIONS",
    "DISPATCHED_FUNCTIONS",
    "UNSUPPORTED_FUNCTIONS",
]


MASKED_SAFE_FUNCTIONS = set()
"""Set of functions that work fine on Masked classes already.

Most of these internally use `numpy.ufunc` or other functions that
are already covered.
"""

APPLY_TO_BOTH_FUNCTIONS = {}
"""Dict of functions that should apply to both data and mask.

The `dict` is keyed by the numpy function and the values are functions
that take the input arguments of the numpy function and organize these
for passing the data and mask to the numpy function.

Returns
-------
data_args : tuple
    Arguments to pass on to the numpy function for the unmasked data.
mask_args : tuple
    Arguments to pass on to the numpy function for the masked data.
kwargs : dict
    Keyword arguments to pass on for both unmasked data and mask.
out : `~astropy.utils.masked.Masked` instance or None
    Optional instance in which to store the output.

Raises
------
NotImplementedError
   When an arguments is masked when it should not be or vice versa.
"""

DISPATCHED_FUNCTIONS = {}
"""Dict of functions that provide the numpy function's functionality.

These are for more complicated versions where the numpy function itself
cannot easily be used.  It should return either the result of the
function, or a tuple consisting of the unmasked result, the mask for the
result and a possible output instance.

It should raise `NotImplementedError` if one of the arguments is masked
when it should not be or vice versa.
"""

UNSUPPORTED_FUNCTIONS = set()
"""Set of numpy functions that are not supported for masked arrays.

For most, masked input simply makes no sense, but for others it may have
been lack of time.  Issues or PRs for support for functions are welcome.
"""

# Almost all from np.core.fromnumeric defer to methods so are OK.
MASKED_SAFE_FUNCTIONS |= {
    getattr(np, name)
    for name in np.core.fromnumeric.__all__
    if name not in {"choose", "put", "resize", "searchsorted", "where", "alen"}
}
```
### 11 - astropy/utils/masked/core.py:

Start line: 643, End line: 672

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    _eq_simple = _comparison_method("__eq__")
    _ne_simple = _comparison_method("__ne__")
    __lt__ = _comparison_method("__lt__")
    __le__ = _comparison_method("__le__")
    __gt__ = _comparison_method("__gt__")
    __ge__ = _comparison_method("__ge__")

    def __eq__(self, other):
        if not self.dtype.names:
            return self._eq_simple(other)

        # For structured arrays, we treat this as a reduction over the fields,
        # where masked fields are skipped and thus do not influence the result.
        other = np.asanyarray(other, dtype=self.dtype)
        result = np.stack(
            [self[field] == other[field] for field in self.dtype.names], axis=-1
        )
        return result.all(axis=-1)

    def __ne__(self, other):
        if not self.dtype.names:
            return self._ne_simple(other)

        # For structured arrays, we treat this as a reduction over the fields,
        # where masked fields are skipped and thus do not influence the result.
        other = np.asanyarray(other, dtype=self.dtype)
        result = np.stack(
            [self[field] != other[field] for field in self.dtype.names], axis=-1
        )
        return result.any(axis=-1)
```
### 12 - astropy/utils/masked/core.py:

Start line: 899, End line: 923

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def _masked_result(self, result, mask, out):
        if isinstance(result, tuple):
            if out is None:
                out = (None,) * len(result)
            if not isinstance(mask, (list, tuple)):
                mask = (mask,) * len(result)
            return tuple(
                self._masked_result(result_, mask_, out_)
                for (result_, mask_, out_) in zip(result, mask, out)
            )

        if out is None:
            # Note that we cannot count on result being the same class as
            # 'self' (e.g., comparison of quantity results in an ndarray, most
            # operations on Longitude and Latitude result in Angle or
            # Quantity), so use Masked to determine the appropriate class.
            return Masked(result, mask)

        # TODO: remove this sanity check once test cases are more complete.
        assert isinstance(out, Masked)
        # If we have an output, the result was written in-place, so we should
        # also write the mask in-place (if not done already in the code).
        if out._mask is not mask:
            out._mask[...] = mask
        return out
```
### 13 - astropy/utils/masked/core.py:

Start line: 1155, End line: 1192

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        result = self.var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
        )
        return np.sqrt(result, out=result)

    def __bool__(self):
        # First get result from array itself; this will error if not a scalar.
        result = super().__bool__()
        return result and not self.mask

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np.logical_or.reduce(
            self, axis=axis, out=out, keepdims=keepdims, where=~self.mask & where
        )

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np.logical_and.reduce(
            self, axis=axis, out=out, keepdims=keepdims, where=~self.mask & where
        )

    # Following overrides needed since somehow the ndarray implementation
    # does not actually call these.
    def __str__(self):
        return np.array_str(self)

    def __repr__(self):
        return np.array_repr(self)

    def __format__(self, format_spec):
        string = super().__format__(format_spec)
        if self.shape == () and self.mask:
            n = min(3, max(1, len(string)))
            return " " * (len(string) - n) + "\u2014" * n
        else:
            return string
```
### 14 - astropy/utils/masked/core.py:

Start line: 227, End line: 244

```python
class Masked(NDArrayShapeMethods):

    def _set_mask(self, mask, copy=False):
        self_dtype = getattr(self, "dtype", None)
        mask_dtype = (
            np.ma.make_mask_descr(self_dtype)
            if self_dtype and self_dtype.names
            else np.dtype("?")
        )
        ma = np.asanyarray(mask, dtype=mask_dtype)
        if ma.shape != self.shape:
            # This will fail (correctly) if not broadcastable.
            self._mask = np.empty(self.shape, dtype=mask_dtype)
            self._mask[...] = ma
        elif ma is mask:
            # Even if not copying use a view so that shape setting
            # does not propagate.
            self._mask = mask.copy() if copy else mask.view()
        else:
            self._mask = ma
```
### 17 - astropy/utils/masked/core.py:

Start line: 1098, End line: 1128

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        # Implementation based on that in numpy/core/_methods.py
        # Cast bool, unsigned int, and int to float64 by default,
        # and do float16 at higher precision.
        is_float16_result = False
        if dtype is None:
            if issubclass(self.dtype.type, (np.integer, np.bool_)):
                dtype = np.dtype("f8")
            elif issubclass(self.dtype.type, np.float16):
                dtype = np.dtype("f4")
                is_float16_result = out is None

        where = ~self.mask & where

        result = self.sum(
            axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        )
        n = np.add.reduce(where, axis=axis, keepdims=keepdims)

        # catch the case when an axis is fully masked to prevent div by zero:
        n = np.add.reduce(where, axis=axis, keepdims=keepdims)
        neq0 = n == 0
        n += neq0
        result /= n

        # correct fully-masked slice results to what is expected for 0/0 division
        result.unmasked[neq0] = np.nan

        if is_float16_result:
            result = result.astype(self.dtype)
        return result
```
### 19 - astropy/utils/masked/core.py:

Start line: 619, End line: 641

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    @shape.setter
    def shape(self, shape):
        old_shape = self.shape
        self._mask.shape = shape
        # Reshape array proper in try/except just in case some broadcasting
        # or so causes it to fail.
        try:
            super(MaskedNDArray, type(self)).shape.__set__(self, shape)
        except Exception as exc:
            self._mask.shape = old_shape
            # Given that the mask reshaping succeeded, the only logical
            # reason for an exception is something like a broadcast error in
            # in __array_finalize__, or a different memory ordering between
            # mask and data.  For those, give a more useful error message;
            # otherwise just raise the error.
            if "could not broadcast" in exc.args[0]:
                raise AttributeError(
                    "Incompatible shape for in-place modification. "
                    "Use `.reshape()` to make a copy with the desired "
                    "shape."
                ) from None
            else:  # pragma: no cover
                raise
```
### 20 - astropy/utils/masked/core.py:

Start line: 280, End line: 300

```python
class Masked(NDArrayShapeMethods):

    def _apply(self, method, *args, **kwargs):
        # Required method for NDArrayShapeMethods, to help provide __getitem__
        # and shape-changing methods.
        if callable(method):
            data = method(self.unmasked, *args, **kwargs)
            mask = method(self.mask, *args, **kwargs)
        else:
            data = getattr(self.unmasked, method)(*args, **kwargs)
            mask = getattr(self.mask, method)(*args, **kwargs)

        result = self.from_unmasked(data, mask, copy=False)
        if "info" in self.__dict__:
            result.info = self.info

        return result

    def __setitem__(self, item, value):
        value, mask = self._get_data_and_mask(value, allow_ma_masked=True)
        if value is not None:
            self.unmasked[item] = value
        self.mask[item] = mask
```
### 21 - astropy/utils/masked/core.py:

Start line: 674, End line: 688

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def _combine_masks(self, masks, out=None):
        masks = [m for m in masks if m is not None and m is not False]
        if not masks:
            return False
        if len(masks) == 1:
            if out is None:
                return masks[0].copy()
            else:
                np.copyto(out, masks[0])
                return out

        out = np.logical_or(masks[0], masks[1], out=out)
        for mask in masks[2:]:
            np.logical_or(out, mask, out=out)
        return out
```
### 25 - astropy/utils/masked/core.py:

Start line: 581, End line: 598

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    def __array_finalize__(self, obj):
        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        # Logically, this should come from ndarray and hence be None, but
        # just in case someone creates a new mixin, we check.
        super_array_finalize = super().__array_finalize__
        if super_array_finalize:  # pragma: no cover
            super_array_finalize(obj)

        if self._mask is None:
            # Got here after, e.g., a view of another masked class.
            # Get its mask, or initialize ours.
            self._set_mask(getattr(obj, "_mask", False))

        if "info" in obj.__dict__:
            self.info = obj.info
```
### 27 - astropy/utils/masked/core.py:

Start line: 1195, End line: 1220

```python
class MaskedRecarray(np.recarray, MaskedNDArray, data_cls=np.recarray):
    # Explicit definition since we need to override some methods.

    def __array_finalize__(self, obj):
        # recarray.__array_finalize__ does not do super, so we do it
        # explicitly.
        super().__array_finalize__(obj)
        super(np.recarray, self).__array_finalize__(obj)

    # __getattribute__, __setattr__, and field use these somewhat
    # obscrure ndarray methods.  TODO: override in MaskedNDArray?
    def getfield(self, dtype, offset=0):
        for field, info in self.dtype.fields.items():
            if offset == info[1] and dtype == info[0]:
                return self[field]

        raise NotImplementedError("can only get existing field from structured dtype.")

    def setfield(self, val, dtype, offset=0):
        for field, info in self.dtype.fields.items():
            if offset == info[1] and dtype == info[0]:
                self[field] = val
                return

        raise NotImplementedError("can only set existing field from structured dtype.")
```
### 41 - astropy/utils/masked/core.py:

Start line: 246, End line: 278

```python
class Masked(NDArrayShapeMethods):

    mask = property(_get_mask, _set_mask)

    # Note: subclass should generally override the unmasked property.
    # This one assumes the unmasked data is stored in a private attribute.
    @property
    def unmasked(self):
        """The unmasked values.

        See Also
        --------
        astropy.utils.masked.Masked.filled
        """
        return self._unmasked

    def filled(self, fill_value):
        """Get a copy of the underlying data, with masked values filled in.

        Parameters
        ----------
        fill_value : object
            Value to replace masked values with.

        See Also
        --------
        astropy.utils.masked.Masked.unmasked
        """
        unmasked = self.unmasked.copy()
        if self.mask.dtype.names:
            np.ma.core._recursive_filled(unmasked, self.mask, fill_value)
        else:
            unmasked[self.mask] = fill_value

        return unmasked
```
### 45 - astropy/utils/masked/core.py:

Start line: 341, End line: 375

```python
class MaskedNDArrayInfo(MaskedInfoBase, ParentDtypeInfo):

    def _represent_as_dict(self):
        out = super()._represent_as_dict()

        masked_array = self._parent

        # If the serialize method for this context (e.g. 'fits' or 'ecsv') is
        # 'data_mask', that means to serialize using an explicit mask column.
        method = self.serialize_method[self._serialize_context]

        if method == "data_mask":
            out["data"] = masked_array.unmasked

            if np.any(masked_array.mask):
                # Only if there are actually masked elements do we add the ``mask`` column
                out["mask"] = masked_array.mask

        elif method == "null_value":
            out["data"] = np.ma.MaskedArray(
                masked_array.unmasked, mask=masked_array.mask
            )

        else:
            raise ValueError(
                'serialize method must be either "data_mask" or "null_value"'
            )

        return out

    def _construct_from_dict(self, map):
        # Override usual handling, since MaskedNDArray takes shape and buffer
        # as input, which is less useful here.
        # The map can contain either a MaskedColumn or a Column and a mask.
        # Extract the mask for the former case.
        map.setdefault("mask", getattr(map["data"], "mask", False))
        return self._parent_cls.from_unmasked(**map)
```
### 46 - astropy/utils/masked/core.py:

Start line: 511, End line: 554

```python
class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):

    # The two pieces typically overridden.
    @classmethod
    def from_unmasked(cls, data, mask=None, copy=False):
        # Note: have to override since __new__ would use ndarray.__new__
        # which expects the shape as its first argument, not an array.
        data = np.array(data, subok=True, copy=copy)
        self = data.view(cls)
        self._set_mask(mask, copy=copy)
        return self

    @property
    def unmasked(self):
        return super().view(self._data_cls)

    @classmethod
    def _get_masked_cls(cls, data_cls):
        # Short-cuts
        if data_cls is np.ndarray:
            return MaskedNDArray
        elif data_cls is None:  # for .view()
            return cls

        return super()._get_masked_cls(data_cls)

    @property
    def flat(self):
        """A 1-D iterator over the Masked array.

        This returns a ``MaskedIterator`` instance, which behaves the same
        as the `~numpy.flatiter` instance returned by `~numpy.ndarray.flat`,
        and is similar to Python's built-in iterator, except that it also
        allows assignment.
        """
        return MaskedIterator(self)

    @property
    def _baseclass(self):
        """Work-around for MaskedArray initialization.

        Allows the base class to be inferred correctly when a masked instance
        is used to initialize (or viewed as) a `~numpy.ma.MaskedArray`.

        """
        return self._data_cls
```
### 52 - astropy/utils/masked/core.py:

Start line: 394, End line: 410

```python
def _comparison_method(op):
    """
    Create a comparison operator for MaskedNDArray.

    Needed since for string dtypes the base operators bypass __array_ufunc__
    and hence return unmasked results.
    """

    def _compare(self, other):
        other_data, other_mask = self._get_data_and_mask(other)
        result = getattr(self.unmasked, op)(other_data)
        if result is NotImplemented:
            return NotImplemented
        mask = self.mask | (other_mask if other_mask is not None else False)
        return self._masked_result(result, mask, None)

    return _compare
```
### 58 - astropy/utils/masked/core.py:

Start line: 70, End line: 80

```python
class Masked(NDArrayShapeMethods):

    def __new__(cls, *args, **kwargs):
        if cls is Masked:
            # Initializing with Masked itself means we're in "factory mode".
            if not kwargs and len(args) == 1 and isinstance(args[0], type):
                # Create a new masked class.
                return cls._get_masked_cls(args[0])
            else:
                return cls._get_masked_instance(*args, **kwargs)
        else:
            # Otherwise we're a subclass and should just pass information on.
            return super().__new__(cls, *args, **kwargs)
```
### 64 - astropy/utils/masked/core.py:

Start line: 110, End line: 125

```python
class Masked(NDArrayShapeMethods):

    # This base implementation just uses the class initializer.
    # Subclasses can override this in case the class does not work
    # with this signature, or to provide a faster implementation.
    @classmethod
    def from_unmasked(cls, data, mask=None, copy=False):
        """Create an instance from unmasked data and a mask."""
        return cls(data, mask=mask, copy=copy)

    @classmethod
    def _get_masked_instance(cls, data, mask=None, copy=False):
        data, data_mask = cls._get_data_and_mask(data)
        if mask is None:
            mask = False if data_mask is None else data_mask

        masked_cls = cls._get_masked_cls(data.__class__)
        return masked_cls.from_unmasked(data, mask, copy)
```
