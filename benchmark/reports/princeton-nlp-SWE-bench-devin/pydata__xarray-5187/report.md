# pydata__xarray-5187

| **pydata/xarray** | `b2351cbe3f3e92f0e242312dae5791fc83a4467a` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 8842 |
| **Any found context length** | 5132 |
| **Avg pos** | 44.2 |
| **Min pos** | 12 |
| **Max pos** | 44 |
| **Top file pos** | 10 |
| **Missing snippets** | 10 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -51,3 +51,24 @@ def least_squares(lhs, rhs, rcond=None, skipna=False):
         # See issue dask/dask#6516
         coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
     return coeffs, residuals
+
+
+def push(array, n, axis):
+    """
+    Dask-aware bottleneck.push
+    """
+    from bottleneck import push
+
+    if len(array.chunks[axis]) > 1 and n is not None and n < array.shape[axis]:
+        raise NotImplementedError(
+            "Cannot fill along a chunked axis when limit is not None."
+            "Either rechunk to a single chunk along this axis or call .compute() or .load() first."
+        )
+    if all(c == 1 for c in array.chunks[axis]):
+        array = array.rechunk({axis: 2})
+    pushed = array.map_blocks(push, axis=axis, n=n)
+    if len(array.chunks[axis]) > 1:
+        pushed = pushed.map_overlap(
+            push, axis=axis, n=n, depth={axis: (1, 0)}, boundary="none"
+        )
+    return pushed
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2515,7 +2515,8 @@ def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
             The maximum number of consecutive NaN values to forward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
@@ -2539,7 +2540,8 @@ def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
             The maximum number of consecutive NaN values to backward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4654,7 +4654,8 @@ def ffill(self, dim: Hashable, limit: int = None) -> "Dataset":
             The maximum number of consecutive NaN values to forward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
@@ -4679,7 +4680,8 @@ def bfill(self, dim: Hashable, limit: int = None) -> "Dataset":
             The maximum number of consecutive NaN values to backward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -631,3 +631,12 @@ def least_squares(lhs, rhs, rcond=None, skipna=False):
         return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
     else:
         return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
+
+
+def push(array, n, axis):
+    from bottleneck import push
+
+    if is_duck_dask_array(array):
+        return dask_array_ops.push(array, n, axis)
+    else:
+        return push(array, n, axis)
diff --git a/xarray/core/missing.py b/xarray/core/missing.py
--- a/xarray/core/missing.py
+++ b/xarray/core/missing.py
@@ -11,7 +11,7 @@
 from . import utils
 from .common import _contains_datetime_like_objects, ones_like
 from .computation import apply_ufunc
-from .duck_array_ops import datetime_to_numeric, timedelta_to_numeric
+from .duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
 from .options import _get_keep_attrs
 from .pycompat import is_duck_dask_array
 from .utils import OrderedSet, is_scalar
@@ -390,12 +390,10 @@ def func_interpolate_na(interpolator, y, x, **kwargs):
 
 def _bfill(arr, n=None, axis=-1):
     """inverse of ffill"""
-    import bottleneck as bn
-
     arr = np.flip(arr, axis=axis)
 
     # fill
-    arr = bn.push(arr, axis=axis, n=n)
+    arr = push(arr, axis=axis, n=n)
 
     # reverse back to original
     return np.flip(arr, axis=axis)
@@ -403,17 +401,15 @@ def _bfill(arr, n=None, axis=-1):
 
 def ffill(arr, dim=None, limit=None):
     """forward fill missing values"""
-    import bottleneck as bn
-
     axis = arr.get_axis_num(dim)
 
     # work around for bottleneck 178
     _limit = limit if limit is not None else arr.shape[axis]
 
     return apply_ufunc(
-        bn.push,
+        push,
         arr,
-        dask="parallelized",
+        dask="allowed",
         keep_attrs=True,
         output_dtypes=[arr.dtype],
         kwargs=dict(n=_limit, axis=axis),
@@ -430,7 +426,7 @@ def bfill(arr, dim=None, limit=None):
     return apply_ufunc(
         _bfill,
         arr,
-        dask="parallelized",
+        dask="allowed",
         keep_attrs=True,
         output_dtypes=[arr.dtype],
         kwargs=dict(n=_limit, axis=axis),

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/dask_array_ops.py | 54 | 54 | - | - | -
| xarray/core/dataarray.py | 2518 | 2518 | 25 | 11 | 8842
| xarray/core/dataarray.py | 2542 | 2542 | 12 | 11 | 5132
| xarray/core/dataset.py | 4657 | 4657 | - | 10 | -
| xarray/core/dataset.py | 4682 | 4682 | - | 10 | -
| xarray/core/duck_array_ops.py | 634 | 634 | 44 | 12 | 16997
| xarray/core/missing.py | 14 | 14 | 40 | 16 | 14048
| xarray/core/missing.py | 393 | 398 | 39 | 16 | 13915
| xarray/core/missing.py | 406 | 416 | 39 | 16 | 13915
| xarray/core/missing.py | 433 | 433 | 22 | 16 | 7876


## Problem Statement

```
bfill behavior dask arrays with small chunk size
\`\`\`python
data = np.random.rand(100)
data[25] = np.nan
da = xr.DataArray(data)

#unchunked 
print('output : orig',da[25].values, ' backfill : ',da.bfill('dim_0')[25].values )
output : orig nan  backfill :  0.024710724099643477

#small chunk
da1 = da.chunk({'dim_0':1})
print('output chunks==1 : orig',da1[25].values, ' backfill : ',da1.bfill('dim_0')[25].values )
output chunks==1 : orig nan  backfill :  nan

# medium chunk
da1 = da.chunk({'dim_0':10})
print('output chunks==10 : orig',da1[25].values, ' backfill : ',da1.bfill('dim_0')[25].values )
output chunks==10 : orig nan  backfill :  0.024710724099643477
\`\`\`




#### Problem description
bfill methods seems to miss nans when dask array chunk size is small. Resulting array still has nan present  (see 'small chunk' section of code)


#### Expected Output
absence of nans
#### Output of ``xr.show_versions()``
INSTALLED VERSIONS
------------------
commit: None
python: 3.6.8.final.0
python-bits: 64
OS: Linux
OS-release: 4.15.0-43-generic
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_CA.UTF-8
LOCALE: en_CA.UTF-8
xarray: 0.11.0
pandas: 0.23.4
numpy: 1.15.4
scipy: None
netCDF4: None
h5netcdf: None
h5py: None
Nio: None
zarr: None
cftime: None
PseudonetCDF: None
rasterio: None
iris: None
bottleneck: 1.2.1
cyordereddict: None
dask: 1.0.0
distributed: 1.25.2
matplotlib: None
cartopy: None
seaborn: None
setuptools: 40.6.3
pip: 18.1
conda: None
pytest: None
IPython: None
sphinx: None



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 531 | 531 | 
| 2 | 2 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 1264 | 2091 | 
| 3 | 3 asv_bench/benchmarks/rolling.py | 1 | 17| 117 | 1381 | 3084 | 
| 4 | 4 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 1565 | 3569 | 
| 5 | 5 xarray/core/dask_array_compat.py | 61 | 152| 596 | 2161 | 5388 | 
| 6 | 6 xarray/core/nanops.py | 78 | 118| 400 | 2561 | 7250 | 
| 7 | 7 asv_bench/benchmarks/dataset_io.py | 1 | 90| 703 | 3264 | 10831 | 
| 8 | 8 asv_bench/benchmarks/unstacking.py | 1 | 30| 194 | 3458 | 11025 | 
| 9 | 9 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 3871 | 11438 | 
| 10 | 9 xarray/core/nanops.py | 163 | 205| 432 | 4303 | 11438 | 
| 11 | **10 xarray/core/dataset.py** | 1 | 133| 655 | 4958 | 73094 | 
| **-> 12 <-** | **11 xarray/core/dataarray.py** | 2528 | 2550| 174 | 5132 | 111872 | 
| 13 | **12 xarray/core/duck_array_ops.py** | 353 | 378| 270 | 5402 | 117086 | 
| 14 | 12 asv_bench/benchmarks/rolling.py | 64 | 86| 215 | 5617 | 117086 | 
| 15 | 13 asv_bench/benchmarks/pandas.py | 1 | 25| 160 | 5777 | 117246 | 
| 16 | 13 asv_bench/benchmarks/dataset_io.py | 187 | 216| 265 | 6042 | 117246 | 
| 17 | **13 xarray/core/dataarray.py** | 1 | 87| 464 | 6506 | 117246 | 
| 18 | 13 xarray/core/nanops.py | 151 | 160| 128 | 6634 | 117246 | 
| 19 | 14 xarray/__init__.py | 1 | 93| 603 | 7237 | 117849 | 
| 20 | 14 asv_bench/benchmarks/dataset_io.py | 395 | 426| 264 | 7501 | 117849 | 
| 21 | 15 xarray/core/pycompat.py | 1 | 38| 208 | 7709 | 118057 | 
| **-> 22 <-** | **16 xarray/core/missing.py** | 423 | 447| 167 | 7876 | 124156 | 
| 23 | 16 asv_bench/benchmarks/dataset_io.py | 341 | 392| 442 | 8318 | 124156 | 
| 24 | 16 asv_bench/benchmarks/dataset_io.py | 147 | 184| 349 | 8667 | 124156 | 
| **-> 25 <-** | **16 xarray/core/dataarray.py** | 2504 | 2526| 175 | 8842 | 124156 | 
| 26 | 17 xarray/core/parallel.py | 1 | 47| 225 | 9067 | 129202 | 
| 27 | **17 xarray/core/dataarray.py** | 2388 | 2502| 1219 | 10286 | 129202 | 
| 28 | 18 xarray/core/common.py | 1 | 45| 240 | 10526 | 145043 | 
| 29 | 18 xarray/core/dask_array_compat.py | 155 | 225| 658 | 11184 | 145043 | 
| 30 | **18 xarray/core/dataset.py** | 382 | 412| 215 | 11399 | 145043 | 
| 31 | 18 asv_bench/benchmarks/dataset_io.py | 429 | 464| 186 | 11585 | 145043 | 
| 32 | 19 xarray/core/nputils.py | 205 | 216| 155 | 11740 | 146892 | 
| 33 | **19 xarray/core/dataset.py** | 354 | 379| 191 | 11931 | 146892 | 
| 34 | 19 asv_bench/benchmarks/indexing.py | 145 | 162| 145 | 12076 | 146892 | 
| 35 | 19 xarray/core/nanops.py | 137 | 148| 115 | 12191 | 146892 | 
| 36 | **19 xarray/core/dataarray.py** | 1145 | 1261| 1183 | 13374 | 146892 | 
| 37 | **19 xarray/core/dataset.py** | 415 | 440| 207 | 13581 | 146892 | 
| 38 | 19 xarray/core/nanops.py | 34 | 47| 152 | 13733 | 146892 | 
| **-> 39 <-** | **19 xarray/core/missing.py** | 391 | 420| 182 | 13915 | 146892 | 
| **-> 40 <-** | **19 xarray/core/missing.py** | 1 | 18| 133 | 14048 | 146892 | 
| 41 | 20 xarray/core/rolling_exp.py | 1 | 32| 266 | 14314 | 147996 | 
| 42 | 20 xarray/core/nanops.py | 1 | 31| 161 | 14475 | 147996 | 
| 43 | 21 xarray/core/alignment.py | 82 | 270| 1951 | 16426 | 154055 | 
| **-> 44 <-** | **21 xarray/core/duck_array_ops.py** | 567 | 634| 571 | 16997 | 154055 | 
| 45 | 22 xarray/core/rolling.py | 522 | 565| 350 | 17347 | 161587 | 
| 46 | 22 xarray/core/nputils.py | 1 | 33| 251 | 17598 | 161587 | 
| 47 | 23 xarray/core/indexing.py | 1 | 20| 117 | 17715 | 173548 | 
| 48 | 24 xarray/core/npcompat.py | 31 | 52| 193 | 17908 | 175537 | 
| 49 | **24 xarray/core/duck_array_ops.py** | 313 | 350| 321 | 18229 | 175537 | 
| 50 | 25 xarray/core/ops.py | 120 | 154| 305 | 18534 | 177919 | 
| 51 | **25 xarray/core/dataarray.py** | 3789 | 3939| 1801 | 20335 | 177919 | 
| 52 | **25 xarray/core/dataset.py** | 2097 | 2745| 6100 | 26435 | 177919 | 
| 53 | 25 xarray/core/npcompat.py | 80 | 129| 391 | 26826 | 177919 | 
| 54 | 26 xarray/backends/api.py | 1 | 109| 668 | 27494 | 189975 | 
| 55 | 26 xarray/core/indexing.py | 1120 | 1133| 116 | 27610 | 189975 | 
| 56 | 26 asv_bench/benchmarks/indexing.py | 125 | 142| 147 | 27757 | 189975 | 
| 57 | 26 asv_bench/benchmarks/indexing.py | 62 | 76| 151 | 27908 | 189975 | 
| 58 | **26 xarray/core/dataarray.py** | 1450 | 1541| 830 | 28738 | 189975 | 
| 59 | 27 xarray/conventions.py | 1 | 29| 168 | 28906 | 196376 | 
| 60 | 27 xarray/core/rolling.py | 477 | 520| 416 | 29322 | 196376 | 
| 61 | **27 xarray/core/dataarray.py** | 2361 | 2386| 204 | 29526 | 196376 | 
| 62 | 27 asv_bench/benchmarks/dataset_io.py | 325 | 338| 112 | 29638 | 196376 | 


## Missing Patch Files

 * 1: xarray/core/dask_array_ops.py
 * 2: xarray/core/dataarray.py
 * 3: xarray/core/dataset.py
 * 4: xarray/core/duck_array_ops.py
 * 5: xarray/core/missing.py

### Hint

```
Thanks for the clear report. Indeed, this looks like a bug.

`bfill()` and `ffill()` are implemented on dask arrays via `apply_ufunc`, but they're applied independently on each chunk -- there's no filling between chunks:
https://github.com/pydata/xarray/blob/ddacf405fb256714ce01e1c4c464f829e1cc5058/xarray/core/missing.py#L262-L289

Instead, I think we need a multi-step process for parallelizing `bottleneck.push`, e.g.,
1. Forward fill each chunk independently.
2. Slice out the *last element* of each chunk and forward fill these.
3. Prepend filled last elements to the start of each chunk, and forward fill them again.
I think this will work (though it needs more tests):
\`\`\`python
import bottleneck
import dask.array as da
import numpy as np

def _last_element(array, axis):
  slices = [slice(None)] * array.ndim
  slices[axis] = slice(-1, None)
  return array[tuple(slices)]

def _concat_push_slice(last_elements, array, axis):
  concatenated = np.concatenate([last_elements, array], axis=axis)
  pushed = bottleneck.push(concatenated, axis=axis)
  slices = [slice(None)] * array.ndim
  slices[axis] = slice(1, None)
  sliced = pushed[tuple(slices)]
  return sliced

def push(array, axis):
  if axis < 0:
    axis += array.ndim
  pushed = array.map_blocks(bottleneck.push, dtype=array.dtype, axis=axis)
  new_chunks = list(array.chunks)
  new_chunks[axis] = tuple(1 for _ in array.chunks[axis])
  last_elements = pushed.map_blocks(
      _last_element, dtype=array.dtype, chunks=tuple(new_chunks), axis=axis)
  pushed_last_elements = (
      last_elements.rechunk({axis: -1})
      .map_blocks(bottleneck.push, dtype=array.dtype, axis=axis)
      .rechunk({axis: 1})
  )
  nan_shape = tuple(1 if axis == a else s for a, s in enumerate(array.shape))
  nan_chunks = tuple((1,) if axis == a else c for a, c in enumerate(array.chunks))
  shifted_pushed_last_elements = da.concatenate(
      [da.full(np.nan, shape=nan_shape, chunks=nan_chunks),
       pushed_last_elements[(slice(None),) * axis + (slice(None, -1),)]],
      axis=axis)
  return da.map_blocks(
      _concat_push_slice,
      shifted_pushed_last_elements,
      pushed,
      dtype=array.dtype,
      chunks=array.chunks,
      axis=axis,
  )

# tests
array = np.array([np.nan, np.nan, np.nan, 1, 2, 3,
                  np.nan, np.nan, 4, 5, np.nan, 6])
expected = bottleneck.push(array, axis=0)
for c in range(1, 11):
  actual = push(da.from_array(array, chunks=c), axis=0).compute()
  np.testing.assert_equal(actual, expected)
\`\`\`
I also recently encountered this bug and without user warnings it took me a while to identify its origin. I'll use this temporary fix. Thanks
I encountered this bug a few days ago.
I understand it isn't trivial to fix, but would it be possible to check and throw an exception? Still better than having it go unnoticed. Thanks
```

## Patch

```diff
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -51,3 +51,24 @@ def least_squares(lhs, rhs, rcond=None, skipna=False):
         # See issue dask/dask#6516
         coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
     return coeffs, residuals
+
+
+def push(array, n, axis):
+    """
+    Dask-aware bottleneck.push
+    """
+    from bottleneck import push
+
+    if len(array.chunks[axis]) > 1 and n is not None and n < array.shape[axis]:
+        raise NotImplementedError(
+            "Cannot fill along a chunked axis when limit is not None."
+            "Either rechunk to a single chunk along this axis or call .compute() or .load() first."
+        )
+    if all(c == 1 for c in array.chunks[axis]):
+        array = array.rechunk({axis: 2})
+    pushed = array.map_blocks(push, axis=axis, n=n)
+    if len(array.chunks[axis]) > 1:
+        pushed = pushed.map_overlap(
+            push, axis=axis, n=n, depth={axis: (1, 0)}, boundary="none"
+        )
+    return pushed
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2515,7 +2515,8 @@ def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
             The maximum number of consecutive NaN values to forward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
@@ -2539,7 +2540,8 @@ def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
             The maximum number of consecutive NaN values to backward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4654,7 +4654,8 @@ def ffill(self, dim: Hashable, limit: int = None) -> "Dataset":
             The maximum number of consecutive NaN values to forward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
@@ -4679,7 +4680,8 @@ def bfill(self, dim: Hashable, limit: int = None) -> "Dataset":
             The maximum number of consecutive NaN values to backward fill. In
             other words, if there is a gap with more than this number of
             consecutive NaNs, it will only be partially filled. Must be greater
-            than 0 or None for no limit.
+            than 0 or None for no limit. Must be None or greater than or equal
+            to axis length if filling along chunked axes (dimensions).
 
         Returns
         -------
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -631,3 +631,12 @@ def least_squares(lhs, rhs, rcond=None, skipna=False):
         return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
     else:
         return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
+
+
+def push(array, n, axis):
+    from bottleneck import push
+
+    if is_duck_dask_array(array):
+        return dask_array_ops.push(array, n, axis)
+    else:
+        return push(array, n, axis)
diff --git a/xarray/core/missing.py b/xarray/core/missing.py
--- a/xarray/core/missing.py
+++ b/xarray/core/missing.py
@@ -11,7 +11,7 @@
 from . import utils
 from .common import _contains_datetime_like_objects, ones_like
 from .computation import apply_ufunc
-from .duck_array_ops import datetime_to_numeric, timedelta_to_numeric
+from .duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
 from .options import _get_keep_attrs
 from .pycompat import is_duck_dask_array
 from .utils import OrderedSet, is_scalar
@@ -390,12 +390,10 @@ def func_interpolate_na(interpolator, y, x, **kwargs):
 
 def _bfill(arr, n=None, axis=-1):
     """inverse of ffill"""
-    import bottleneck as bn
-
     arr = np.flip(arr, axis=axis)
 
     # fill
-    arr = bn.push(arr, axis=axis, n=n)
+    arr = push(arr, axis=axis, n=n)
 
     # reverse back to original
     return np.flip(arr, axis=axis)
@@ -403,17 +401,15 @@ def _bfill(arr, n=None, axis=-1):
 
 def ffill(arr, dim=None, limit=None):
     """forward fill missing values"""
-    import bottleneck as bn
-
     axis = arr.get_axis_num(dim)
 
     # work around for bottleneck 178
     _limit = limit if limit is not None else arr.shape[axis]
 
     return apply_ufunc(
-        bn.push,
+        push,
         arr,
-        dask="parallelized",
+        dask="allowed",
         keep_attrs=True,
         output_dtypes=[arr.dtype],
         kwargs=dict(n=_limit, axis=axis),
@@ -430,7 +426,7 @@ def bfill(arr, dim=None, limit=None):
     return apply_ufunc(
         _bfill,
         arr,
-        dask="parallelized",
+        dask="allowed",
         keep_attrs=True,
         output_dtypes=[arr.dtype],
         kwargs=dict(n=_limit, axis=axis),

```

## Test Patch

```diff
diff --git a/xarray/tests/test_duck_array_ops.py b/xarray/tests/test_duck_array_ops.py
--- a/xarray/tests/test_duck_array_ops.py
+++ b/xarray/tests/test_duck_array_ops.py
@@ -20,6 +20,7 @@
     mean,
     np_timedelta64_to_float,
     pd_timedelta_to_float,
+    push,
     py_timedelta_to_float,
     stack,
     timedelta_to_numeric,
@@ -34,6 +35,7 @@
     has_dask,
     has_scipy,
     raise_if_dask_computes,
+    requires_bottleneck,
     requires_cftime,
     requires_dask,
 )
@@ -858,3 +860,26 @@ def test_least_squares(use_dask, skipna):
 
     np.testing.assert_allclose(coeffs, [1.5, 1.25])
     np.testing.assert_allclose(residuals, [2.0])
+
+
+@requires_dask
+@requires_bottleneck
+def test_push_dask():
+    import bottleneck
+    import dask.array
+
+    array = np.array([np.nan, np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, np.nan, 6])
+    expected = bottleneck.push(array, axis=0)
+    for c in range(1, 11):
+        with raise_if_dask_computes():
+            actual = push(dask.array.from_array(array, chunks=c), axis=0, n=None)
+        np.testing.assert_equal(actual, expected)
+
+    # some chunks of size-1 with NaN
+    with raise_if_dask_computes():
+        actual = push(
+            dask.array.from_array(array, chunks=(1, 2, 3, 2, 2, 1, 1)),
+            axis=0,
+            n=None,
+        )
+    np.testing.assert_equal(actual, expected)
diff --git a/xarray/tests/test_missing.py b/xarray/tests/test_missing.py
--- a/xarray/tests/test_missing.py
+++ b/xarray/tests/test_missing.py
@@ -17,6 +17,7 @@
     assert_allclose,
     assert_array_equal,
     assert_equal,
+    raise_if_dask_computes,
     requires_bottleneck,
     requires_cftime,
     requires_dask,
@@ -393,37 +394,39 @@ def test_ffill():
 
 @requires_bottleneck
 @requires_dask
-def test_ffill_dask():
+@pytest.mark.parametrize("method", ["ffill", "bfill"])
+def test_ffill_bfill_dask(method):
     da, _ = make_interpolate_example_data((40, 40), 0.5)
     da = da.chunk({"x": 5})
-    actual = da.ffill("time")
-    expected = da.load().ffill("time")
-    assert isinstance(actual.data, dask_array_type)
-    assert_equal(actual, expected)
 
-    # with limit
-    da = da.chunk({"x": 5})
-    actual = da.ffill("time", limit=3)
-    expected = da.load().ffill("time", limit=3)
-    assert isinstance(actual.data, dask_array_type)
+    dask_method = getattr(da, method)
+    numpy_method = getattr(da.compute(), method)
+    # unchunked axis
+    with raise_if_dask_computes():
+        actual = dask_method("time")
+    expected = numpy_method("time")
     assert_equal(actual, expected)
 
-
-@requires_bottleneck
-@requires_dask
-def test_bfill_dask():
-    da, _ = make_interpolate_example_data((40, 40), 0.5)
-    da = da.chunk({"x": 5})
-    actual = da.bfill("time")
-    expected = da.load().bfill("time")
-    assert isinstance(actual.data, dask_array_type)
+    # chunked axis
+    with raise_if_dask_computes():
+        actual = dask_method("x")
+    expected = numpy_method("x")
     assert_equal(actual, expected)
 
     # with limit
-    da = da.chunk({"x": 5})
-    actual = da.bfill("time", limit=3)
-    expected = da.load().bfill("time", limit=3)
-    assert isinstance(actual.data, dask_array_type)
+    with raise_if_dask_computes():
+        actual = dask_method("time", limit=3)
+    expected = numpy_method("time", limit=3)
+    assert_equal(actual, expected)
+
+    # limit < axis size
+    with pytest.raises(NotImplementedError):
+        actual = dask_method("x", limit=2)
+
+    # limit > axis size
+    with raise_if_dask_computes():
+        actual = dask_method("x", limit=41)
+    expected = numpy_method("x", limit=41)
     assert_equal(actual, expected)
 
 

```


## Code snippets

### 1 - asv_bench/benchmarks/dataarray_missing.py:

Start line: 1, End line: 75

```python
import pandas as pd

import xarray as xr

from . import randn, requires_dask

try:
    import dask  # noqa: F401
except ImportError:
    pass


def make_bench_data(shape, frac_nan, chunks):
    vals = randn(shape, frac_nan)
    coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x", "y"), coords=coords)

    if chunks is not None:
        da = da.chunk(chunks)

    return da


def time_interpolate_na(shape, chunks, method, limit):
    if chunks is not None:
        requires_dask()
    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.interpolate_na(dim="time", method="linear", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_interpolate_na.param_names = ["shape", "chunks", "method", "limit"]
time_interpolate_na.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    ["linear", "spline", "quadratic", "cubic"],
    [None, 3],
)


def time_ffill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.ffill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_ffill.param_names = ["shape", "chunks", "limit"]
time_ffill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)


def time_bfill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.bfill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_bfill.param_names = ["shape", "chunks", "limit"]
time_bfill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)
```
### 2 - asv_bench/benchmarks/indexing.py:

Start line: 1, End line: 59

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import randint, randn, requires_dask

nx = 3000
ny = 2000
nt = 1000

basic_indexes = {
    "1slice": {"x": slice(0, 3)},
    "1slice-1scalar": {"x": 0, "y": slice(None, None, 3)},
    "2slicess-1scalar": {"x": slice(3, -3, 3), "y": 1, "t": slice(None, -3, 3)},
}

basic_assignment_values = {
    "1slice": xr.DataArray(randn((3, ny), frac_nan=0.1), dims=["x", "y"]),
    "1slice-1scalar": xr.DataArray(randn(int(ny / 3) + 1, frac_nan=0.1), dims=["y"]),
    "2slicess-1scalar": xr.DataArray(
        randn(int((nx - 6) / 3), frac_nan=0.1), dims=["x"]
    ),
}

outer_indexes = {
    "1d": {"x": randint(0, nx, 400)},
    "2d": {"x": randint(0, nx, 500), "y": randint(0, ny, 400)},
    "2d-1scalar": {"x": randint(0, nx, 100), "y": 1, "t": randint(0, nt, 400)},
}

outer_assignment_values = {
    "1d": xr.DataArray(randn((400, ny), frac_nan=0.1), dims=["x", "y"]),
    "2d": xr.DataArray(randn((500, 400), frac_nan=0.1), dims=["x", "y"]),
    "2d-1scalar": xr.DataArray(randn(100, frac_nan=0.1), dims=["x"]),
}

vectorized_indexes = {
    "1-1d": {"x": xr.DataArray(randint(0, nx, 400), dims="a")},
    "2-1d": {
        "x": xr.DataArray(randint(0, nx, 400), dims="a"),
        "y": xr.DataArray(randint(0, ny, 400), dims="a"),
    },
    "3-2d": {
        "x": xr.DataArray(randint(0, nx, 400).reshape(4, 100), dims=["a", "b"]),
        "y": xr.DataArray(randint(0, ny, 400).reshape(4, 100), dims=["a", "b"]),
        "t": xr.DataArray(randint(0, nt, 400).reshape(4, 100), dims=["a", "b"]),
    },
}

vectorized_assignment_values = {
    "1-1d": xr.DataArray(randn((400, 2000)), dims=["a", "y"], coords={"a": randn(400)}),
    "2-1d": xr.DataArray(randn(400), dims=["a"], coords={"a": randn(400)}),
    "3-2d": xr.DataArray(
        randn((4, 100)), dims=["a", "b"], coords={"a": randn(4), "b": randn(100)}
    ),
}
```
### 3 - asv_bench/benchmarks/rolling.py:

Start line: 1, End line: 17

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 3000
long_nx = 30000000
ny = 2000
nt = 1000
window = 20

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))
randn_long = randn((long_nx,), frac_nan=0.1)
```
### 4 - asv_bench/benchmarks/interp.py:

Start line: 1, End line: 22

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 3000
long_nx = 30000000
ny = 2000
nt = 1000
window = 20

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))
randn_long = randn((long_nx,), frac_nan=0.1)


new_x_short = np.linspace(0.3 * nx, 0.7 * nx, 100)
new_x_long = np.linspace(0.3 * nx, 0.7 * nx, 1000)
new_y_long = np.linspace(0.1, 0.9, 1000)
```
### 5 - xarray/core/dask_array_compat.py:

Start line: 61, End line: 152

```python
if LooseVersion(dask_version) > LooseVersion("2.9.0"):
    nanmedian = da.nanmedian
else:

    def nanmedian(a, axis=None, keepdims=False):
        """
        This works by automatically chunking the reduced axes to a single chunk
        and then calling ``numpy.nanmedian`` function across the remaining dimensions
        """

        if axis is None:
            raise NotImplementedError(
                "The da.nanmedian function only works along an axis.  "
                "The full algorithm is difficult to do in parallel"
            )

        if not isinstance(axis, Iterable):
            axis = (axis,)

        axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

        result = da.map_blocks(
            np.nanmedian,
            a,
            axis=axis,
            keepdims=keepdims,
            drop_axis=axis if not keepdims else None,
            chunks=[1 if ax in axis else c for ax, c in enumerate(a.chunks)]
            if keepdims
            else None,
        )

        return result


if LooseVersion(dask_version) > LooseVersion("2.30.0"):
    ensure_minimum_chunksize = da.overlap.ensure_minimum_chunksize
else:

    # copied from dask
    def ensure_minimum_chunksize(size, chunks):
        """Determine new chunks to ensure that every chunk >= size

        Parameters
        ----------
        size : int
            The maximum size of any chunk.
        chunks : tuple
            Chunks along one axis, e.g. ``(3, 3, 2)``

        Examples
        --------
        >>> ensure_minimum_chunksize(10, (20, 20, 1))
        (20, 11, 10)
        >>> ensure_minimum_chunksize(3, (1, 1, 3))
        (5,)

        See Also
        --------
        overlap
        """
        if size <= min(chunks):
            return chunks

        # add too-small chunks to chunks before them
        output = []
        new = 0
        for c in chunks:
            if c < size:
                if new > size + (size - c):
                    output.append(new - (size - c))
                    new = size
                else:
                    new += c
            if new >= size:
                output.append(new)
                new = 0
            if c >= size:
                new += c
        if new >= size:
            output.append(new)
        elif len(output) >= 1:
            output[-1] += new
        else:
            raise ValueError(
                f"The overlapping depth {size} is larger than your "
                f"array {sum(chunks)}."
            )

        return tuple(output)
```
### 6 - xarray/core/nanops.py:

Start line: 78, End line: 118

```python
def nanmin(a, axis=None, out=None):
    if a.dtype.kind == "O":
        return _nan_minmax_object("min", dtypes.get_pos_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmin(a, axis=axis)


def nanmax(a, axis=None, out=None):
    if a.dtype.kind == "O":
        return _nan_minmax_object("max", dtypes.get_neg_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmax(a, axis=axis)


def nanargmin(a, axis=None):
    if a.dtype.kind == "O":
        fill_value = dtypes.get_pos_infinity(a.dtype)
        return _nan_argminmax_object("argmin", fill_value, a, axis=axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanargmin(a, axis=axis)


def nanargmax(a, axis=None):
    if a.dtype.kind == "O":
        fill_value = dtypes.get_neg_infinity(a.dtype)
        return _nan_argminmax_object("argmax", fill_value, a, axis=axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanargmax(a, axis=axis)


def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 0)
    result = _dask_or_eager_func("sum")(a, axis=axis, dtype=dtype)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result
```
### 7 - asv_bench/benchmarks/dataset_io.py:

Start line: 1, End line: 90

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import randint, randn, requires_dask

try:
    import dask
    import dask.multiprocessing
except ImportError:
    pass


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class IOSingleNetCDF:
    """
    A few examples that benchmark reading/writing a single netCDF file with
    xarray
    """

    timeout = 300.0
    repeat = 1
    number = 5

    def make_ds(self):

        # single Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45

        self.block_chunks = {
            "time": self.nt / 4,
            "lon": self.nx / 3,
            "lat": self.ny / 3,
        }

        self.time_chunks = {"time": int(self.nt / 36)}

        times = pd.date_range("1970-01-01", periods=self.nt, freq="D")
        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.ds["foo"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="foo",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["bar"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="bar",
            attrs={"units": "bar units", "description": "a description"},
        )
        self.ds["baz"] = xr.DataArray(
            randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
            coords={"lon": lons, "lat": lats},
            dims=("lon", "lat"),
            name="baz",
            attrs={"units": "baz units", "description": "a description"},
        )

        self.ds.attrs = {"history": "created for xarray benchmarking"}

        self.oinds = {
            "time": randint(0, self.nt, 120),
            "lon": randint(0, self.nx, 20),
            "lat": randint(0, self.ny, 10),
        }
        self.vinds = {
            "time": xr.DataArray(randint(0, self.nt, 120), dims="x"),
            "lon": xr.DataArray(randint(0, self.nx, 120), dims="x"),
            "lat": slice(3, 20),
        }
```
### 8 - asv_bench/benchmarks/unstacking.py:

Start line: 1, End line: 30

```python
import numpy as np

import xarray as xr

from . import requires_dask


class Unstacking:
    def setup(self):
        data = np.random.RandomState(0).randn(500, 1000)
        self.da_full = xr.DataArray(data, dims=list("ab")).stack(flat_dim=[...])
        self.da_missing = self.da_full[:-1]
        self.df_missing = self.da_missing.to_pandas()

    def time_unstack_fast(self):
        self.da_full.unstack("flat_dim")

    def time_unstack_slow(self):
        self.da_missing.unstack("flat_dim")

    def time_unstack_pandas_slow(self):
        self.df_missing.unstack()


class UnstackingDask(Unstacking):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.da_full = self.da_full.chunk({"flat_dim": 50})
```
### 9 - asv_bench/benchmarks/reindexing.py:

Start line: 1, End line: 49

```python
import numpy as np

import xarray as xr

from . import requires_dask


class Reindex:
    def setup(self):
        data = np.random.RandomState(0).randn(1000, 100, 100)
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={"time": np.arange(1000), "x": np.arange(100), "y": np.arange(100)},
        )

    def time_1d_coarse(self):
        self.ds.reindex(time=np.arange(0, 1000, 5)).load()

    def time_1d_fine_all_found(self):
        self.ds.reindex(time=np.arange(0, 1000, 0.5), method="nearest").load()

    def time_1d_fine_some_missing(self):
        self.ds.reindex(
            time=np.arange(0, 1000, 0.5), method="nearest", tolerance=0.1
        ).load()

    def time_2d_coarse(self):
        self.ds.reindex(x=np.arange(0, 100, 2), y=np.arange(0, 100, 2)).load()

    def time_2d_fine_all_found(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5), y=np.arange(0, 100, 0.5), method="nearest"
        ).load()

    def time_2d_fine_some_missing(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5),
            y=np.arange(0, 100, 0.5),
            method="nearest",
            tolerance=0.1,
        ).load()


class ReindexDask(Reindex):
    def setup(self):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
```
### 10 - xarray/core/nanops.py:

Start line: 163, End line: 205

```python
def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
    value_mean = _nanmean_ddof_object(
        ddof=0, value=value, axis=axis, keepdims=True, **kwargs
    )
    squared = (value.astype(value_mean.dtype) - value_mean) ** 2
    return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    if a.dtype.kind == "O":
        return _nanvar_object(a, axis=axis, dtype=dtype, ddof=ddof)

    return _dask_or_eager_func("nanvar", eager_module=nputils)(
        a, axis=axis, dtype=dtype, ddof=ddof
    )


def nanstd(a, axis=None, dtype=None, out=None, ddof=0):
    return _dask_or_eager_func("nanstd", eager_module=nputils)(
        a, axis=axis, dtype=dtype, ddof=ddof
    )


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 1)
    result = _dask_or_eager_func("nanprod")(a, axis=axis, dtype=dtype, out=out)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def nancumsum(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func("nancumsum", eager_module=nputils)(
        a, axis=axis, dtype=dtype
    )


def nancumprod(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func("nancumprod", eager_module=nputils)(
        a, axis=axis, dtype=dtype
    )
```
### 11 - xarray/core/dataset.py:

Start line: 1, End line: 133

```python
import copy
import datetime
import inspect
import sys
import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from html import escape
from numbers import Number
from operator import methodcaller
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd

import xarray as xr

from ..coding.cftimeindex import _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import (
    alignment,
    dtypes,
    duck_array_ops,
    formatting,
    formatting_html,
    groupby,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects
from .coordinates import (
    DatasetCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    remove_unused_levels_categories,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    SortedKeysDict,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    hashable,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    assert_unique_multiindex_level_names,
    broadcast_variables,
)

if TYPE_CHECKING:
    from ..backends import AbstractDataStore, ZarrStore
    from .dataarray import DataArray
    from .merge import CoercibleMapping

    T_DSorDA = TypeVar("T_DSorDA", DataArray, "Dataset")

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "dayofyear",
    "weekofyear",
    "dayofweek",
    "quarter",
]
```
### 12 - xarray/core/dataarray.py:

Start line: 2528, End line: 2550

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import bfill

        return bfill(self, dim, limit=limit)
```
### 13 - xarray/core/duck_array_ops.py:

Start line: 353, End line: 378

```python
# Attributes `numeric_only`, `available_min_count` is used for docs.
# See ops.inject_reduce_methods
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True)
min = _create_nan_agg_method("min", coerce_strings=True)
sum = _create_nan_agg_method("sum")
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", dask_module=dask_array_compat)
median.numeric_only = True
prod = _create_nan_agg_method("prod")
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod")
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum")
cumsum_1d.numeric_only = True
unravel_index = _dask_or_eager_func("unravel_index")


_mean = _create_nan_agg_method("mean")
```
### 17 - xarray/core/dataarray.py:

Start line: 1, End line: 87

```python
import datetime
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .arithmetic import DataArrayArithmetic
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)
if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None
    try:
        from cdms2 import Variable as cdms2_Variable
    except ImportError:
        cdms2_Variable = None
    try:
        from iris.cube import Cube as iris_Cube
    except ImportError:
        iris_Cube = None
```
### 22 - xarray/core/missing.py:

Start line: 423, End line: 447

```python
def bfill(arr, dim=None, limit=None):
    """backfill missing values"""
    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(
        _bfill,
        arr,
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[arr.dtype],
        kwargs=dict(n=_limit, axis=axis),
    ).transpose(*arr.dims)


def _import_interpolant(interpolant, method):
    """Import interpolant from scipy.interpolate."""
    try:
        from scipy import interpolate

        return getattr(interpolate, interpolant)
    except ImportError as e:
        raise ImportError(f"Interpolation with method {method} requires scipy.") from e
```
### 25 - xarray/core/dataarray.py:

Start line: 2504, End line: 2526

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)
```
### 27 - xarray/core/dataarray.py:

Start line: 2388, End line: 2502

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def interpolate_na(
        self,
        dim: Hashable = None,
        method: str = "linear",
        limit: int = None,
        use_coordinate: Union[bool, str] = True,
        max_gap: Union[
            int, float, str, pd.Timedelta, np.timedelta64, datetime.timedelta
        ] = None,
        keep_attrs: bool = None,
        **kwargs: Any,
    ) -> "DataArray":
        """Fill in NaNs by interpolating according to different methods.

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to interpolate.
        method : str, optional
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation (Default). Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krog', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.
        use_coordinate : bool or str, default: True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            eqaully-spaced along ``dim``. If True, the IndexVariable `dim` is
            used. If ``use_coordinate`` is a string, it specifies the name of a
            coordinate variariable to use as the index.
        limit : int, default: None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit. This filling is done regardless of the size of
            the gap in the data. To only interpolate over gaps less than a given length,
            see ``max_gap``.
        max_gap : int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default: None
            Maximum size of gap, a continuous sequence of NaNs, that will be filled.
            Use None for no limit. When interpolating along a datetime64 dimension
            and ``use_coordinate=True``, ``max_gap`` can be one of the following:

            - a string that is valid input for pandas.to_timedelta
            - a :py:class:`numpy.timedelta64` object
            - a :py:class:`pandas.Timedelta` object
            - a :py:class:`datetime.timedelta` object

            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled
            dimensions has not been implemented yet. Gap length is defined as the difference
            between coordinate values at the first data point after a gap and the last value
            before a gap. For gaps at the beginning (end), gap length is defined as the difference
            between coordinate values at the first (last) valid data point and the first (last) NaN.
            For example, consider::

                <xarray.DataArray (x: 9)>
                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])
                Coordinates:
                  * x        (x) int64 0 1 2 3 4 5 6 7 8

            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively
        keep_attrs : bool, default: True
            If True, the dataarray's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.
        **kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: DataArray
            Filled in DataArray.

        See Also
        --------
        numpy.interp
        scipy.interpolate

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [np.nan, 2, 3, np.nan, 0], dims="x", coords={"x": [0, 1, 2, 3, 4]}
        ... )
        >>> da
        <xarray.DataArray (x: 5)>
        array([nan,  2.,  3., nan,  0.])
        Coordinates:
          * x        (x) int64 0 1 2 3 4

        >>> da.interpolate_na(dim="x", method="linear")
        <xarray.DataArray (x: 5)>
        array([nan, 2. , 3. , 1.5, 0. ])
        Coordinates:
          * x        (x) int64 0 1 2 3 4

        >>> da.interpolate_na(dim="x", method="linear", fill_value="extrapolate")
        <xarray.DataArray (x: 5)>
        array([1. , 2. , 3. , 1.5, 0. ])
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        """
        from .missing import interp_na

        return interp_na(
            self,
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            keep_attrs=keep_attrs,
            **kwargs,
        )
```
### 30 - xarray/core/dataset.py:

Start line: 382, End line: 412

```python
def _get_chunk(var, chunks):
    # chunks need to be explicity computed to take correctly into accout
    # backend preferred chunking
    import dask.array as da

    if isinstance(var, IndexVariable):
        return {}

    if isinstance(chunks, int) or (chunks == "auto"):
        chunks = dict.fromkeys(var.dims, chunks)

    preferred_chunks = var.encoding.get("preferred_chunks", {})
    preferred_chunks_list = [
        preferred_chunks.get(dim, shape) for dim, shape in zip(var.dims, var.shape)
    ]

    chunks_list = [
        chunks.get(dim, None) or preferred_chunks.get(dim, None) for dim in var.dims
    ]

    output_chunks_list = da.core.normalize_chunks(
        chunks_list,
        shape=var.shape,
        dtype=var.dtype,
        previous_chunks=preferred_chunks_list,
    )

    output_chunks = dict(zip(var.dims, output_chunks_list))
    _check_chunks_compatibility(var, output_chunks, preferred_chunks)

    return output_chunks
```
### 33 - xarray/core/dataset.py:

Start line: 354, End line: 379

```python
def _assert_empty(args: tuple, msg: str = "%s") -> None:
    if args:
        raise ValueError(msg % args)


def _check_chunks_compatibility(var, chunks, preferred_chunks):
    for dim in var.dims:
        if dim not in chunks or (dim not in preferred_chunks):
            continue

        preferred_chunks_dim = preferred_chunks.get(dim)
        chunks_dim = chunks.get(dim)

        if isinstance(chunks_dim, int):
            chunks_dim = (chunks_dim,)
        else:
            chunks_dim = chunks_dim[:-1]

        if any(s % preferred_chunks_dim for s in chunks_dim):
            warnings.warn(
                f"Specified Dask chunks {chunks[dim]} would separate "
                f"on disks chunk shape {preferred_chunks[dim]} for dimension {dim}. "
                "This could degrade performance. "
                "Consider rechunking after loading instead.",
                stacklevel=2,
            )
```
### 36 - xarray/core/dataarray.py:

Start line: 1145, End line: 1261

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance=None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by selecting index
        labels along the specified dimension(s).

        In contrast to `DataArray.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        .. warning::

          Do not try to assign values when using any of the indexing methods
          ``isel`` or ``sel``::

            da = xr.DataArray([0, 1, 2, 3], dims=['x'])
            # DO NOT do this
            da.isel(x=[0, 1, 2])[1] = -1

          Assigning values with the chained indexing using ``.sel`` or
          ``.isel`` fails silently.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            A new DataArray with the same contents as this DataArray, except the
            data and each dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this DataArray, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.arange(25).reshape(5, 5),
        ...     coords={"x": np.arange(5), "y": np.arange(5)},
        ...     dims=("x", "y"),
        ... )
        >>> da
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Coordinates:
          * x        (x) int64 0 1 2 3 4
          * y        (y) int64 0 1 2 3 4

        >>> tgt_x = xr.DataArray(np.linspace(0, 4, num=5), dims="points")
        >>> tgt_y = xr.DataArray(np.linspace(0, 4, num=5), dims="points")
        >>> da = da.sel(x=tgt_x, y=tgt_y, method="nearest")
        >>> da
        <xarray.DataArray (points: 5)>
        array([ 0,  6, 12, 18, 24])
        Coordinates:
            x        (points) int64 0 1 2 3 4
            y        (points) int64 0 1 2 3 4
        Dimensions without coordinates: points
        """
        ds = self._to_temp_dataset().sel(
            indexers=indexers,
            drop=drop,
            method=method,
            tolerance=tolerance,
            **indexers_kwargs,
        )
        return self._from_temp_dataset(ds)
```
### 37 - xarray/core/dataset.py:

Start line: 415, End line: 440

```python
def _maybe_chunk(
    name,
    var,
    chunks,
    token=None,
    lock=None,
    name_prefix="xarray-",
    overwrite_encoded_chunks=False,
):
    from dask.base import tokenize

    if chunks is not None:
        chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
    if var.ndim:
        # when rechunking by different amounts, make sure dask names change
        # by provinding chunks as an input to tokenize.
        # subtle bugs result otherwise. see GH3350
        token2 = tokenize(name, token if token else var._data, chunks)
        name2 = f"{name_prefix}{name}-{token2}"
        var = var.chunk(chunks, name=name2, lock=lock)

        if overwrite_encoded_chunks and var.chunks is not None:
            var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
        return var
    else:
        return var
```
### 39 - xarray/core/missing.py:

Start line: 391, End line: 420

```python
def _bfill(arr, n=None, axis=-1):
    """inverse of ffill"""
    import bottleneck as bn

    arr = np.flip(arr, axis=axis)

    # fill
    arr = bn.push(arr, axis=axis, n=n)

    # reverse back to original
    return np.flip(arr, axis=axis)


def ffill(arr, dim=None, limit=None):
    """forward fill missing values"""
    import bottleneck as bn

    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(
        bn.push,
        arr,
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[arr.dtype],
        kwargs=dict(n=_limit, axis=axis),
    ).transpose(*arr.dims)
```
### 40 - xarray/core/missing.py:

Start line: 1, End line: 18

```python
import datetime as dt
import warnings
from distutils.version import LooseVersion
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Hashable, Sequence, Union

import numpy as np
import pandas as pd

from . import utils
from .common import _contains_datetime_like_objects, ones_like
from .computation import apply_ufunc
from .duck_array_ops import datetime_to_numeric, timedelta_to_numeric
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import OrderedSet, is_scalar
from .variable import Variable, broadcast_variables
```
### 44 - xarray/core/duck_array_ops.py:

Start line: 567, End line: 634

```python
mean.numeric_only = True  # type: ignore[attr-defined]


def _nd_cum_func(cum_func, array, axis, **kwargs):
    array = asarray(array)
    if axis is None:
        axis = tuple(range(array.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    out = array
    for ax in axis:
        out = cum_func(out, axis=ax, **kwargs)
    return out


def cumprod(array, axis=None, **kwargs):
    """N-dimensional version of cumprod."""
    return _nd_cum_func(cumprod_1d, array, axis, **kwargs)


def cumsum(array, axis=None, **kwargs):
    """N-dimensional version of cumsum."""
    return _nd_cum_func(cumsum_1d, array, axis, **kwargs)


_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg="%r with skipna=True is not yet implemented on dask arrays",
)


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis"""
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis"""
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanlast(values, axis)
    return take(values, -1, axis=axis)


def sliding_window_view(array, window_shape, axis):
    """
    Make an ndarray with a rolling window of axis-th dimension.
    The rolling dimension will be placed at the last dimension.
    """
    if is_duck_dask_array(array):
        return dask_array_compat.sliding_window_view(array, window_shape, axis)
    else:
        return npcompat.sliding_window_view(array, window_shape, axis)


def least_squares(lhs, rhs, rcond=None, skipna=False):
    """Return the coefficients and residuals of a least-squares fit."""
    if is_duck_dask_array(rhs):
        return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
    else:
        return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
```
### 49 - xarray/core/duck_array_ops.py:

Start line: 313, End line: 350

```python
def _create_nan_agg_method(name, dask_module=dask_array, coerce_strings=False):
    from . import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop("out", None) is not None:
            raise TypeError(f"`out` is not valid for {name}")

        values = asarray(values)

        if coerce_strings and values.dtype.kind in "SU":
            values = values.astype(object)

        func = None
        if skipna or (skipna is None and values.dtype.kind in "cfO"):
            nanname = "nan" + name
            func = getattr(nanops, nanname)
        else:
            if name in ["sum", "prod"]:
                kwargs.pop("min_count", None)
            func = _dask_or_eager_func(name, dask_module=dask_module)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice encountered")
                return func(values, axis=axis, **kwargs)
        except AttributeError:
            if not is_duck_dask_array(values):
                raise
            try:  # dask/dask#3133 dask sometimes needs dtype argument
                # if func does not accept dtype, then raises TypeError
                return func(values, axis=axis, dtype=values.dtype, **kwargs)
            except (AttributeError, TypeError):
                raise NotImplementedError(
                    f"{name} is not yet implemented on dask arrays"
                )

    f.__name__ = name
    return f
```
### 51 - xarray/core/dataarray.py:

Start line: 3789, End line: 3939

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> "DataArray":
        """Pad this array along one or more dimensions.

        .. warning::
            This function is experimental and its behaviour is likely to change
            especially regarding padding of dimension coordinates (or IndexVariables).

        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),
        coordinates will be padded with the same mode, otherwise coordinates
        are padded using the "constant" mode with fill_value dtypes.NA.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : str, default: "constant"
            One of the following string values (taken from numpy docs)

            'constant' (default)
                Pads with a constant value.
            'edge'
                Pads with the edge values of array.
            'linear_ramp'
                Pads with the linear ramp between end_value and the
                array edge value.
            'maximum'
                Pads with the maximum value of all or part of the
                vector along each axis.
            'mean'
                Pads with the mean value of all or part of the
                vector along each axis.
            'median'
                Pads with the median value of all or part of the
                vector along each axis.
            'minimum'
                Pads with the minimum value of all or part of the
                vector along each axis.
            'reflect'
                Pads with the reflection of the vector mirrored on
                the first and last values of the vector along each
                axis.
            'symmetric'
                Pads with the reflection of the vector mirrored
                along the edge of the array.
            'wrap'
                Pads with the wrap of the vector along the axis.
                The first values are used to pad the end and the
                end values are used to pad the beginning.
        stat_length : int, tuple or mapping of hashable to tuple, default: None
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
            statistic lengths along each dimension.
            ((before, after),) yields same before and after statistic lengths
            for each dimension.
            (stat_length,) or int is a shortcut for before = after = statistic
            length for all axes.
            Default is ``None``, to use the entire axis.
        constant_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'constant'.  The values to set the padded values for each
            axis.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            pad constants along each dimension.
            ``((before, after),)`` yields same before and after constants for each
            dimension.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all dimensions.
            Default is 0.
        end_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            end values along each dimension.
            ``((before, after),)`` yields same before and after end values for each
            axis.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all axes.
            Default is 0.
        reflect_type : {"even", "odd"}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        **pad_width_kwargs
            The keyword arguments form of ``pad_width``.
            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

        Returns
        -------
        padded : DataArray
            DataArray with the padded coordinates and data.

        See Also
        --------
        DataArray.shift, DataArray.roll, DataArray.bfill, DataArray.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        For ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``.

        Examples
        --------
        >>> arr = xr.DataArray([5, 6, 7], coords=[("x", [0, 1, 2])])
        >>> arr.pad(x=(1, 2), constant_values=0)
        <xarray.DataArray (x: 6)>
        array([0, 5, 6, 7, 0, 0])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 2.0 nan nan

        >>> da = xr.DataArray(
        ...     [[0, 1, 2, 3], [10, 11, 12, 13]],
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 1], "y": [10, 20, 30, 40], "z": ("x", [100, 200])},
        ... )
        >>> da.pad(x=1)
        <xarray.DataArray (x: 4, y: 4)>
        array([[nan, nan, nan, nan],
               [ 0.,  1.,  2.,  3.],
               [10., 11., 12., 13.],
               [nan, nan, nan, nan]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan

        Careful, ``constant_values`` are coerced to the data type of the array which may
        lead to a loss of precision:

        >>> da.pad(x=1, constant_values=1.23456789)
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1,  1,  1,  1],
               [ 0,  1,  2,  3],
               [10, 11, 12, 13],
               [ 1,  1,  1,  1]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan
        """
        ds = self._to_temp_dataset().pad(
            pad_width=pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            **pad_width_kwargs,
        )
        return self._from_temp_dataset(ds)
```
### 52 - xarray/core/dataset.py:

Start line: 2097, End line: 2745

```python
class Dataset(DataWithCoords, DatasetArithmetic, Mapping):

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed along the specified
        dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            return self._isel_fancy(indexers, drop=drop, missing_dims=missing_dims)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's
        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        variables = {}
        dims: Dict[Hashable, Tuple[int, ...]] = {}
        coord_names = self._coord_names.copy()
        indexes = self._indexes.copy() if self._indexes is not None else None

        for var_name, var_value in self._variables.items():
            var_indexers = {k: v for k, v in indexers.items() if k in var_value.dims}
            if var_indexers:
                var_value = var_value.isel(var_indexers)
                if drop and var_value.ndim == 0 and var_name in coord_names:
                    coord_names.remove(var_name)
                    if indexes:
                        indexes.pop(var_name, None)
                    continue
                if indexes and var_name in indexes:
                    if var_value.ndim == 1:
                        indexes[var_name] = var_value.to_index()
                    else:
                        del indexes[var_name]
            variables[var_name] = var_value
            dims.update(zip(var_value.dims, var_value.shape))

        return self._construct_direct(
            variables=variables,
            coord_names=coord_names,
            dims=dims,
            attrs=self._attrs,
            indexes=indexes,
            encoding=self._encoding,
            close=self._close,
        )

    def _isel_fancy(
        self,
        indexers: Mapping[Hashable, Any],
        *,
        drop: bool,
        missing_dims: str = "raise",
    ) -> "Dataset":
        # Note: we need to preserve the original indexers variable in order to merge the
        # coords below
        indexers_list = list(self._validate_indexers(indexers, missing_dims))

        variables: Dict[Hashable, Variable] = {}
        indexes: Dict[Hashable, pd.Index] = {}

        for name, var in self.variables.items():
            var_indexers = {k: v for k, v in indexers_list if k in var.dims}
            if drop and name in var_indexers:
                continue  # drop this variable

            if name in self.indexes:
                new_var, new_index = isel_variable_and_index(
                    name, var, self.indexes[name], var_indexers
                )
                if new_index is not None:
                    indexes[name] = new_index
            elif var_indexers:
                new_var = var.isel(indexers=var_indexers)
            else:
                new_var = var.copy(deep=False)

            variables[name] = new_var

        coord_names = self._coord_names & variables.keys()
        selected = self._replace_with_new_dims(variables, coord_names, indexes)

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(indexers)
        variables.update(coord_vars)
        indexes.update(new_indexes)
        coord_names = self._coord_names & variables.keys() | coord_vars.keys()
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        pos_indexers, new_indexes = remap_label_indexers(
            self, indexers=indexers, method=method, tolerance=tolerance
        )
        result = self.isel(indexers=pos_indexers, drop=drop)
        return result._overwrite_indexes(new_indexes)

    def head(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with the first `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.tail
        Dataset.thin
        DataArray.head
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "head")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
        indexers_slices = {k: slice(val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def tail(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with the last `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.head
        Dataset.thin
        DataArray.tail
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "tail")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
        indexers_slices = {
            k: slice(-val, None) if val != 0 else slice(val)
            for k, val in indexers.items()
        }
        return self.isel(indexers_slices)

    def thin(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Returns a new dataset with each array indexed along every `n`-th
        value for the specified dimension(s)

        Parameters
        ----------
        indexers : dict or int
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.head
        Dataset.tail
        DataArray.thin
        """
        if (
            not indexers_kwargs
            and not isinstance(indexers, int)
            and not is_dict_like(indexers)
        ):
            raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "thin")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    "dimension %r, found %r" % (k, type(v))
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    "for dimension %r, found %s" % (k, v)
                )
            elif v == 0:
                raise ValueError("step cannot be zero")
        indexers_slices = {k: slice(None, None, val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def broadcast_like(
        self, other: Union["Dataset", "DataArray"], exclude: Iterable[Hashable] = None
    ) -> "Dataset":
        """Broadcast this DataArray against another Dataset or DataArray.
        This is equivalent to xr.broadcast(other, self)[1]

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

        """
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        args = align(other, self, join="outer", copy=False, exclude=exclude)

        dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)

        return _broadcast_helper(args[1], exclude, dims_map, common_coords)

    def reindex_like(
        self,
        other: Union["Dataset", "DataArray"],
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
    ) -> "Dataset":
        """Conform this object onto the indexes of another object, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to pandas.Index objects, which provides coordinates upon
            which to index the variables in this dataset. The indexes on this
            other object need not be the same as the indexes on this
            dataset. Any mis-matched index values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for filling index values from other not found in this
            dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like maps
            variable names to fill values.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but coordinates from the
            other object.

        See Also
        --------
        Dataset.reindex
        align
        """
        indexers = alignment.reindex_like_indexers(self, other)
        return self.reindex(
            indexers=indexers,
            method=method,
            copy=copy,
            fill_value=fill_value,
            tolerance=tolerance,
        )

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like,
            maps variable names (including coordinates) to fill values.
        sparse : bool, default: False
            use sparse-array.
        **indexers_kwargs : {dim: indexer, ...}, optional
            Keyword arguments in the same form as ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer

        Examples
        --------
        Create a dataset with some fictional data.

        >>> import xarray as xr
        >>> import pandas as pd
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature": ("station", 20 * np.random.rand(4)),
        ...         "pressure": ("station", 500 * np.random.rand(4)),
        ...     },
        ...     coords={"station": ["boston", "nyc", "seattle", "denver"]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'nyc' 'seattle' 'denver'
        Data variables:
            temperature  (station) float64 10.98 14.3 12.06 10.9
            pressure     (station) float64 211.8 322.9 218.8 445.9
        >>> x.indexes
        station: Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ["boston", "austin", "seattle", "lincoln"]
        >>> x.reindex({"station": new_index})
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 nan 12.06 nan
            pressure     (station) float64 211.8 nan 218.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({"station": new_index}, fill_value=0)
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 0.0 218.8 0.0

        We can also use different fill values for each variable.

        >>> x.reindex(
        ...     {"station": new_index}, fill_value={"temperature": 0, "pressure": 100}
        ... )
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 100.0 218.8 100.0

        Because the index is not monotonically increasing or decreasing, we cannot use arguments
        to the keyword method to fill the `NaN` values.

        >>> x.reindex({"station": new_index}, method="nearest")
        Traceback (most recent call last):
        ...
            raise ValueError('index must be monotonic increasing or decreasing')
        ValueError: index must be monotonic increasing or decreasing

        To further illustrate the filling functionality in reindex, we will create a
        dataset with a monotonically increasing index (for example, a sequence of dates).

        >>> x2 = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             "time",
        ...             [15.57, 12.77, np.nan, 0.3081, 16.59, 15.12],
        ...         ),
        ...         "pressure": ("time", 500 * np.random.rand(6)),
        ...     },
        ...     coords={"time": pd.date_range("01/01/2019", periods=6, freq="D")},
        ... )
        >>> x2
        <xarray.Dataset>
        Dimensions:      (time: 6)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-01 2019-01-02 ... 2019-01-06
        Data variables:
            temperature  (time) float64 15.57 12.77 nan 0.3081 16.59 15.12
            pressure     (time) float64 481.8 191.7 395.9 264.4 284.0 462.8

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range("12/29/2018", periods=10, freq="D")
        >>> x2.reindex({"time": time_index2})
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 nan nan nan 481.8 ... 264.4 284.0 462.8 nan

        The index entries that did not have a value in the original data frame (for example, `2018-12-29`)
        are by default filled with NaN. If desired, we can fill in the missing values using one of several options.

        For example, to back-propagate the last valid value to fill the `NaN` values,
        pass `bfill` as an argument to the `method` keyword.

        >>> x3 = x2.reindex({"time": time_index2}, method="bfill")
        >>> x3
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 15.57 15.57 15.57 15.57 ... 16.59 15.12 nan
            pressure     (time) float64 481.8 481.8 481.8 481.8 ... 284.0 462.8 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 1)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03
        Data variables:
            temperature  (time) float64 nan
            pressure     (time) float64 395.9
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 2)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan
            pressure     (time) float64 395.9 nan

        This is because filling while reindexing does not look at dataset values, but only compares
        the original and desired indexes. If you do want to fill in the `NaN` values present in the
        original dataset, use the :py:meth:`~Dataset.fillna()` method.

        """
        return self._reindex(
            indexers,
            method,
            tolerance,
            copy,
            fill_value,
            sparse=False,
            **indexers_kwargs,
        )
```
### 58 - xarray/core/dataarray.py:

Start line: 1450, End line: 1541

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance=None,
        copy: bool = True,
        fill_value=dtypes.NA,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Conform this object onto the indexes of another object, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found on
            this data array:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values. Use this
            data array's name to refer to the data array's values.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but replaced
            coordinates.

        Examples
        --------
        Reverse latitude:

        >>> da = xr.DataArray(
        ...     np.arange(4),
        ...     coords=[np.array([90, 89, 88, 87])],
        ...     dims="lat",
        ... )
        >>> da
        <xarray.DataArray (lat: 4)>
        array([0, 1, 2, 3])
        Coordinates:
          * lat      (lat) int64 90 89 88 87
        >>> da.reindex(lat=da.lat[::-1])
        <xarray.DataArray (lat: 4)>
        array([3, 2, 1, 0])
        Coordinates:
          * lat      (lat) int64 87 88 89 90

        See Also
        --------
        DataArray.reindex_like
        align
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")
        if isinstance(fill_value, dict):
            fill_value = fill_value.copy()
            sentinel = object()
            value = fill_value.pop(self.name, sentinel)
            if value is not sentinel:
                fill_value[_THIS_ARRAY] = value

        ds = self._to_temp_dataset().reindex(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )
        return self._from_temp_dataset(ds)
```
### 61 - xarray/core/dataarray.py:

Start line: 2361, End line: 2386

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    def fillna(self, value: Any) -> "DataArray":
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray or DataArray
            Used to fill all matching missing values in this array. If the
            argument is a DataArray, it is first aligned with (reindexed to)
            this array.

        Returns
        -------
        DataArray
        """
        if utils.is_dict_like(value):
            raise TypeError(
                "cannot provide fill value as a dictionary with "
                "fillna on a DataArray"
            )
        out = ops.fillna(self, value)
        return out
```
