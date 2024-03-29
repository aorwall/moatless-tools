# pydata__xarray-4184

| **pydata/xarray** | `65ca92a5c0a4143d00dd7a822bcb1d49738717f1` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 39181 |
| **Any found context length** | 39181 |
| **Avg pos** | 46.666666666666664 |
| **Min pos** | 28 |
| **Max pos** | 28 |
| **Top file pos** | 5 |
| **Missing snippets** | 8 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/asv_bench/benchmarks/pandas.py b/asv_bench/benchmarks/pandas.py
new file mode 100644
--- /dev/null
+++ b/asv_bench/benchmarks/pandas.py
@@ -0,0 +1,24 @@
+import numpy as np
+import pandas as pd
+
+from . import parameterized
+
+
+class MultiIndexSeries:
+    def setup(self, dtype, subset):
+        data = np.random.rand(100000).astype(dtype)
+        index = pd.MultiIndex.from_product(
+            [
+                list("abcdefhijk"),
+                list("abcdefhijk"),
+                pd.date_range(start="2000-01-01", periods=1000, freq="B"),
+            ]
+        )
+        series = pd.Series(data, index)
+        if subset:
+            series = series[::3]
+        self.series = series
+
+    @parameterized(["dtype", "subset"], ([int, float], [True, False]))
+    def time_to_xarray(self, dtype, subset):
+        self.series.to_xarray()
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4543,11 +4543,10 @@ def to_dataframe(self):
         return self._to_dataframe(self.dims)
 
     def _set_sparse_data_from_dataframe(
-        self, dataframe: pd.DataFrame, dims: tuple
+        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
     ) -> None:
         from sparse import COO
 
-        idx = dataframe.index
         if isinstance(idx, pd.MultiIndex):
             coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
             is_sorted = idx.is_lexsorted()
@@ -4557,11 +4556,7 @@ def _set_sparse_data_from_dataframe(
             is_sorted = True
             shape = (idx.size,)
 
-        for name, series in dataframe.items():
-            # Cast to a NumPy array first, in case the Series is a pandas
-            # Extension array (which doesn't have a valid NumPy dtype)
-            values = np.asarray(series)
-
+        for name, values in arrays:
             # In virtually all real use cases, the sparse array will now have
             # missing values and needs a fill_value. For consistency, don't
             # special case the rare exceptions (e.g., dtype=int without a
@@ -4580,18 +4575,36 @@ def _set_sparse_data_from_dataframe(
             self[name] = (dims, data)
 
     def _set_numpy_data_from_dataframe(
-        self, dataframe: pd.DataFrame, dims: tuple
+        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
     ) -> None:
-        idx = dataframe.index
-        if isinstance(idx, pd.MultiIndex):
-            # expand the DataFrame to include the product of all levels
-            full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
-            dataframe = dataframe.reindex(full_idx)
-            shape = tuple(lev.size for lev in idx.levels)
-        else:
-            shape = (idx.size,)
-        for name, series in dataframe.items():
-            data = np.asarray(series).reshape(shape)
+        if not isinstance(idx, pd.MultiIndex):
+            for name, values in arrays:
+                self[name] = (dims, values)
+            return
+
+        shape = tuple(lev.size for lev in idx.levels)
+        indexer = tuple(idx.codes)
+
+        # We already verified that the MultiIndex has all unique values, so
+        # there are missing values if and only if the size of output arrays is
+        # larger that the index.
+        missing_values = np.prod(shape) > idx.shape[0]
+
+        for name, values in arrays:
+            # NumPy indexing is much faster than using DataFrame.reindex() to
+            # fill in missing values:
+            # https://stackoverflow.com/a/35049899/809705
+            if missing_values:
+                dtype, fill_value = dtypes.maybe_promote(values.dtype)
+                data = np.full(shape, fill_value, dtype)
+            else:
+                # If there are no missing values, keep the existing dtype
+                # instead of promoting to support NA, e.g., keep integer
+                # columns as integers.
+                # TODO: consider removing this special case, which doesn't
+                # exist for sparse=True.
+                data = np.zeros(shape, values.dtype)
+            data[indexer] = values
             self[name] = (dims, data)
 
     @classmethod
@@ -4631,7 +4644,19 @@ def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Datas
         if not dataframe.columns.is_unique:
             raise ValueError("cannot convert DataFrame with non-unique columns")
 
-        idx, dataframe = remove_unused_levels_categories(dataframe.index, dataframe)
+        idx = remove_unused_levels_categories(dataframe.index)
+
+        if isinstance(idx, pd.MultiIndex) and not idx.is_unique:
+            raise ValueError(
+                "cannot convert a DataFrame with a non-unique MultiIndex into xarray"
+            )
+
+        # Cast to a NumPy array first, in case the Series is a pandas Extension
+        # array (which doesn't have a valid NumPy dtype)
+        # TODO: allow users to control how this casting happens, e.g., by
+        # forwarding arguments to pandas.Series.to_numpy?
+        arrays = [(k, np.asarray(v)) for k, v in dataframe.items()]
+
         obj = cls()
 
         if isinstance(idx, pd.MultiIndex):
@@ -4647,9 +4672,9 @@ def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Datas
             obj[index_name] = (dims, idx)
 
         if sparse:
-            obj._set_sparse_data_from_dataframe(dataframe, dims)
+            obj._set_sparse_data_from_dataframe(idx, arrays, dims)
         else:
-            obj._set_numpy_data_from_dataframe(dataframe, dims)
+            obj._set_numpy_data_from_dataframe(idx, arrays, dims)
         return obj
 
     def to_dask_dataframe(self, dim_order=None, set_index=False):
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -9,7 +9,7 @@
 from .variable import Variable
 
 
-def remove_unused_levels_categories(index, dataframe=None):
+def remove_unused_levels_categories(index: pd.Index) -> pd.Index:
     """
     Remove unused levels from MultiIndex and unused categories from CategoricalIndex
     """
@@ -25,14 +25,15 @@ def remove_unused_levels_categories(index, dataframe=None):
                 else:
                     level = level[index.codes[i]]
                 levels.append(level)
+            # TODO: calling from_array() reorders MultiIndex levels. It would
+            # be best to avoid this, if possible, e.g., by using
+            # MultiIndex.remove_unused_levels() (which does not reorder) on the
+            # part of the MultiIndex that is not categorical, or by fixing this
+            # upstream in pandas.
             index = pd.MultiIndex.from_arrays(levels, names=index.names)
     elif isinstance(index, pd.CategoricalIndex):
         index = index.remove_unused_categories()
-
-    if dataframe is None:
-        return index
-    dataframe = dataframe.set_index(index)
-    return dataframe.index, dataframe
+    return index
 
 
 class Indexes(collections.abc.Mapping):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| asv_bench/benchmarks/pandas.py | 0 | 0 | - | - | -
| xarray/core/dataset.py | 4546 | 4550 | 28 | 5 | 39181
| xarray/core/dataset.py | 4560 | 4564 | 28 | 5 | 39181
| xarray/core/dataset.py | 4583 | 4594 | 28 | 5 | 39181
| xarray/core/dataset.py | 4634 | 4634 | 28 | 5 | 39181
| xarray/core/dataset.py | 4650 | 4652 | 28 | 5 | 39181
| xarray/core/indexes.py | 12 | 12 | - | - | -
| xarray/core/indexes.py | 28 | 32 | - | - | -


## Problem Statement

```
Stack + to_array before to_xarray is much faster that a simple to_xarray
I was seeing some slow performance around `to_xarray()` on MultiIndexed series, and found that unstacking one of the dimensions before running `to_xarray()`, and then restacking with `to_array()` was ~30x faster. This time difference is consistent with larger data sizes.

To reproduce:

Create a series with a MultiIndex, ensuring the MultiIndex isn't a simple product:

\`\`\`python
s = pd.Series(
    np.random.rand(100000), 
    index=pd.MultiIndex.from_product([
        list('abcdefhijk'),
        list('abcdefhijk'),
        pd.DatetimeIndex(start='2000-01-01', periods=1000, freq='B'),
    ]))

cropped = s[::3]
cropped.index=pd.MultiIndex.from_tuples(cropped.index, names=list('xyz'))

cropped.head()

# x  y  z         
# a  a  2000-01-03    0.993989
#      2000-01-06    0.850518
#      2000-01-11    0.068944
#      2000-01-14    0.237197
#      2000-01-19    0.784254
# dtype: float64
\`\`\`

Two approaches for getting this into xarray;
1 - Simple `.to_xarray()`:

\`\`\`python
# current_method = cropped.to_xarray()

<xarray.DataArray (x: 10, y: 10, z: 1000)>
array([[[0.993989,      nan, ...,      nan, 0.721663],
        [     nan,      nan, ..., 0.58224 ,      nan],
        ...,
        [     nan, 0.369382, ...,      nan,      nan],
        [0.98558 ,      nan, ...,      nan, 0.403732]],

       [[     nan,      nan, ..., 0.493711,      nan],
        [     nan, 0.126761, ...,      nan,      nan],
        ...,
        [0.976758,      nan, ...,      nan, 0.816612],
        [     nan,      nan, ..., 0.982128,      nan]],

       ...,

       [[     nan, 0.971525, ...,      nan,      nan],
        [0.146774,      nan, ...,      nan, 0.419806],
        ...,
        [     nan,      nan, ..., 0.700764,      nan],
        [     nan, 0.502058, ...,      nan,      nan]],

       [[0.246768,      nan, ...,      nan, 0.079266],
        [     nan,      nan, ..., 0.802297,      nan],
        ...,
        [     nan, 0.636698, ...,      nan,      nan],
        [0.025195,      nan, ...,      nan, 0.629305]]])
Coordinates:
  * x        (x) object 'a' 'b' 'c' 'd' 'e' 'f' 'h' 'i' 'j' 'k'
  * y        (y) object 'a' 'b' 'c' 'd' 'e' 'f' 'h' 'i' 'j' 'k'
  * z        (z) datetime64[ns] 2000-01-03 2000-01-04 ... 2003-10-30 2003-10-31
\`\`\`

This takes *536 ms*

2 - unstack in pandas first, and then use `to_array` to do the equivalent of a restack:
\`\`\`
proposed_version = (
    cropped
    .unstack('y')
    .to_xarray()
    .to_array('y')
)
\`\`\`

This takes *17.3 ms*

To confirm these are identical:

\`\`\`
proposed_version_adj = (
    proposed_version
    .assign_coords(y=proposed_version['y'].astype(object))
    .transpose(*current_version.dims)
)

proposed_version_adj.equals(current_version)
# True
\`\`\`

#### Problem description

A default operation is much slower than a (potentially) equivalent operation that's not the default.

I need to look more at what's causing the issues. I think it's to do with the `.reindex(full_idx)`, but I'm unclear why it's so much faster in the alternative route, and whether there's a fix that we can make to make the default path fast.


#### Output of ``xr.show_versions()``

<details>
INSTALLED VERSIONS
------------------
commit: None
python: 2.7.14.final.0
python-bits: 64
OS: Linux
OS-release: 4.9.93-linuxkit-aufs
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: en_US.UTF-8
LANG: en_US.utf8
LOCALE: None.None

xarray: 0.10.9
pandas: 0.23.4
numpy: 1.15.2
scipy: 1.1.0
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
dask: None
distributed: None
matplotlib: 2.2.3
cartopy: 0.16.0
seaborn: 0.9.0
setuptools: 40.4.3
pip: 18.0
conda: None
pytest: 3.8.1
IPython: 5.8.0
sphinx: None
</details>

to_xarray() result is incorrect when one of multi-index levels is not sorted
<!-- Please include a self-contained copy-pastable example that generates the issue if possible.

Please be concise with code posted. See guidelines below on how to provide a good bug report:

- Craft Minimal Bug Reports: http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports
- Minimal Complete Verifiable Examples: https://stackoverflow.com/help/mcve

Bug reports that follow these guidelines are easier to diagnose, and so are often handled much more quickly.
-->

**What happened**:
to_xarray() sorts multi-index level **values** and returns result for the sorted values but it doesn't sort **levels** or expects levels to be sorted resulting in completely incorrect order of data for the displayed coordinates.
\`\`\`python
df:
            C1  C2
lev1 lev2        
b    foo    0   1
a    foo    2   3 

df.to_xarray():
 <xarray.Dataset>
Dimensions:  (lev1: 2, lev2: 1)
Coordinates:
  * lev1     (lev1) object 'b' 'a'
  * lev2     (lev2) object 'foo'
Data variables:
    C1       (lev1, lev2) int64 2 0
    C2       (lev1, lev2) int64 3 1 
\`\`\`

**What you expected to happen**:
Should account for the order of levels in the original index.

**Minimal Complete Verifiable Example**:

\`\`\`python
import pandas as pd
df = pd.concat(
    {
        'b': pd.DataFrame([[0, 1]], index=['foo'], columns=['C1', 'C2']),
        'a': pd.DataFrame([[2, 3]], index=['foo'], columns=['C1', 'C2']),
    }
).rename_axis(['lev1', 'lev2'])
print('df:\n', df, '\n')
print('df.to_xarray():\n', df.to_xarray(), '\n')
print('df.index.levels[0]:\n', df.index.levels[0])
\`\`\`

**Anything else we need to know?**:

**Environment**:

<details><summary>Output of <tt>xr.show_versions()</tt></summary>

<!-- Paste the output here xr.show_versions() here -->
INSTALLED VERSIONS
------------------
commit: None
python: 3.8.2 | packaged by conda-forge | (default, Apr 24 2020, 08:20:52) 
[GCC 7.3.0]
python-bits: 64
OS: Linux
OS-release: 5.4.7-100.fc30.x86_64
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_GB.UTF-8
LOCALE: en_GB.UTF-8
libhdf5: 1.10.4
libnetcdf: 4.6.3
xarray: 0.15.1
pandas: 1.0.5
numpy: 1.19.0
scipy: 1.5.0
netCDF4: 1.5.3
pydap: None
h5netcdf: None
h5py: None
Nio: None
zarr: None
cftime: 1.1.3
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: 1.3.2
dask: 2.19.0
distributed: 2.19.0
matplotlib: 3.2.2
cartopy: None
seaborn: 0.10.1
numbagg: installed
setuptools: 46.3.0.post20200513
pip: 20.1
conda: None
pytest: 5.4.3
IPython: 7.15.0
sphinx: None

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/alignment.py | 69 | 255| 1897 | 1897 | 5962 | 
| 2 | 2 xarray/__init__.py | 1 | 93| 603 | 2500 | 6565 | 
| 3 | 3 xarray/core/dataarray.py | 1 | 81| 433 | 2933 | 38498 | 
| 4 | 4 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 3663 | 39910 | 
| 5 | **5 xarray/core/dataset.py** | 1 | 136| 647 | 4310 | 92509 | 
| 6 | 6 xarray/core/merge.py | 635 | 840| 2707 | 7017 | 100332 | 
| 7 | 7 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 7176 | 100491 | 
| 8 | 8 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 7589 | 100904 | 
| 9 | 9 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 8120 | 101435 | 
| 10 | 10 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 8835 | 105040 | 
| 11 | 11 xarray/core/combine.py | 516 | 712| 2442 | 11277 | 111753 | 
| 12 | **11 xarray/core/dataset.py** | 5161 | 5721| 5042 | 16319 | 111753 | 
| 13 | 11 xarray/core/dataarray.py | 1846 | 1899| 458 | 16777 | 111753 | 
| 14 | **11 xarray/core/dataset.py** | 555 | 1390| 6285 | 23062 | 111753 | 
| 15 | 11 xarray/core/alignment.py | 534 | 603| 663 | 23725 | 111753 | 
| 16 | 12 doc/conf.py | 1 | 99| 647 | 24372 | 115482 | 
| 17 | 13 xarray/core/groupby.py | 740 | 751| 123 | 24495 | 123342 | 
| 18 | 14 xarray/core/utils.py | 1 | 84| 467 | 24962 | 128798 | 
| 19 | 14 xarray/core/groupby.py | 186 | 200| 174 | 25136 | 128798 | 
| 20 | 15 xarray/core/resample_cftime.py | 80 | 110| 261 | 25397 | 132112 | 
| 21 | **15 xarray/core/dataset.py** | 3030 | 3779| 6067 | 31464 | 132112 | 
| 22 | 15 xarray/core/dataarray.py | 1755 | 1790| 293 | 31757 | 132112 | 
| 23 | 16 ci/min_deps_check.py | 1 | 39| 227 | 31984 | 133646 | 
| 24 | 17 xarray/util/print_versions.py | 80 | 161| 707 | 32691 | 134858 | 
| 25 | 18 xarray/core/missing.py | 1 | 16| 118 | 32809 | 140136 | 
| 26 | 19 xarray/core/nputils.py | 1 | 33| 244 | 33053 | 142414 | 
| 27 | 20 xarray/core/common.py | 1 | 35| 181 | 33234 | 155816 | 
| **-> 28 <-** | **20 xarray/core/dataset.py** | 4401 | 5159| 5947 | 39181 | 155816 | 
| 29 | 21 xarray/convert.py | 1 | 60| 321 | 39502 | 158169 | 
| 30 | 22 xarray/core/coordinates.py | 1 | 30| 154 | 39656 | 161141 | 
| 31 | 23 xarray/plot/utils.py | 1 | 54| 283 | 39939 | 167618 | 
| 32 | 24 xarray/core/indexing.py | 1000 | 1067| 746 | 40685 | 179281 | 
| 33 | 25 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 40869 | 179766 | 
| 34 | 25 xarray/core/dataarray.py | 2970 | 3026| 578 | 41447 | 179766 | 
| 35 | 26 xarray/coding/cftimeindex.py | 262 | 326| 644 | 42091 | 185752 | 
| 36 | 27 xarray/plot/plot.py | 617 | 773| 1721 | 43812 | 193939 | 
| 37 | 28 xarray/coding/times.py | 1 | 64| 451 | 44263 | 197793 | 


## Missing Patch Files

 * 1: asv_bench/benchmarks/pandas.py
 * 2: xarray/core/dataset.py
 * 3: xarray/core/indexes.py

### Hint

```
Here are the top entries I see with `%prun cropped.to_xarray()`:
\`\`\`
         308597 function calls (308454 primitive calls) in 0.651 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   100000    0.255    0.000    0.275    0.000 datetimes.py:606(<lambda>)
        1    0.165    0.165    0.165    0.165 {built-in method pandas._libs.lib.is_datetime_with_singletz_array}
        1    0.071    0.071    0.634    0.634 {method 'get_indexer' of 'pandas._libs.index.BaseMultiIndexCodesEngine' objects}
        1    0.054    0.054    0.054    0.054 {pandas._libs.lib.fast_zip}
        1    0.029    0.029    0.304    0.304 {pandas._libs.lib.map_infer}
   100009    0.011    0.000    0.011    0.000 datetimelike.py:232(freq)
        9    0.010    0.001    0.010    0.001 {pandas._libs.lib.infer_dtype}
   100021    0.010    0.000    0.010    0.000 datetimes.py:684(tz)
        1    0.009    0.009    0.009    0.009 {built-in method pandas._libs.tslib.array_to_datetime}
        2    0.008    0.004    0.008    0.004 {method 'get_indexer' of 'pandas._libs.index.IndexEngine' objects}
        1    0.008    0.008    0.651    0.651 dataarray.py:1827(from_series)
    66/65    0.005    0.000    0.005    0.000 {built-in method numpy.core.multiarray.array}
    24/22    0.001    0.000    0.362    0.016 base.py:677(_values)
       17    0.001    0.000    0.001    0.000 {built-in method numpy.core.multiarray.empty}
    19/18    0.001    0.000    0.189    0.010 base.py:4914(_ensure_index)
        5    0.001    0.000    0.001    0.000 {method 'repeat' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 {method 'tolist' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 {pandas._libs.algos.take_1d_object_object}
        4    0.001    0.000    0.001    0.000 {pandas._libs.algos.take_1d_int64_int64}
     1846    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       16    0.001    0.000    0.001    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.001    0.001    0.001    0.001 {method 'get_indexer' of 'pandas._libs.index.DatetimeEngine' objects}
\`\`\`

There seems to be a suspiciously large amount of effort applying a function to individual datetime objects.
When I stepped through, it was by-and-large all taken up by https://github.com/pydata/xarray/blob/master/xarray/core/dataset.py#L3121. That's where the boxing & unboxing of the datetimes is from.

I haven't yet discovered how the alternative path avoids this work. If anyone has priors please lmk!
It's 3x faster to unstack & stack all-but-one level, vs reindexing over a filled-out index (and I think always produces the same result). 

Our current code takes the slow path.

I could make that change, but that strongly feels like I don't understand the root cause. I haven't spent much time with reshaping code - lmk if anyone has ideas.

\`\`\`python

idx = cropped.index
full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)

reindexed = cropped.reindex(full_idx)

%timeit reindexed = cropped.reindex(full_idx)
# 1 loop, best of 3: 278 ms per loop

%%timeit
stack_unstack = (
    cropped
    .unstack(list('yz'))
    .stack(list('yz'),dropna=False)
)
# 10 loops, best of 3: 80.8 ms per loop

stack_unstack.equals(reindexed)
# True
\`\`\`
My working hypothesis is that pandas has a set of fast routines in C, such that it can stack without reindexing to the full index. The routines only work in 1-2 dimensions.

So without some hackery (i.e. converting multi-dimensional arrays to pandas' size and back), the current implementation is reasonable*. Next step would be to write our own routines that can operate on multiple dimensions (numbagg!).

Is that consistent with others' views, particularly those who know this area well?

'* one small fix that would improve performance of `series.to_xarray()` only, is the [comment above](https://github.com/pydata/xarray/issues/2459#issuecomment-426483497). Lmk if you think worth making that change
The vast majority of the time in xarray's current implementation seems to be spent in `DataFrame.reindex()`, but I see no reason why this operations needs to be so slow. I expect we could probably optimize this significantly on the pandas side.

See these results from line-profiler:
\`\`\`
In [8]: %lprun -f xarray.Dataset.from_dataframe cropped.to_xarray()
Timer unit: 1e-06 s

Total time: 0.727191 s
File: /Users/shoyer/dev/xarray/xarray/core/dataset.py
Function: from_dataframe at line 3094

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
  3094                                               @classmethod
  3095                                               def from_dataframe(cls, dataframe):
  3096                                                   """Convert a pandas.DataFrame into an xarray.Dataset
  3097
  3098                                                   Each column will be converted into an independent variable in the
  3099                                                   Dataset. If the dataframe's index is a MultiIndex, it will be expanded
  3100                                                   into a tensor product of one-dimensional indices (filling in missing
  3101                                                   values with NaN). This method will produce a Dataset very similar to
  3102                                                   that on which the 'to_dataframe' method was called, except with
  3103                                                   possibly redundant dimensions (since all dataset variables will have
  3104                                                   the same dimensionality).
  3105                                                   """
  3106                                                   # TODO: Add an option to remove dimensions along which the variables
  3107                                                   # are constant, to enable consistent serialization to/from a dataframe,
  3108                                                   # even if some variables have different dimensionality.
  3109
  3110         1        352.0    352.0      0.0          if not dataframe.columns.is_unique:
  3111                                                       raise ValueError(
  3112                                                           'cannot convert DataFrame with non-unique columns')
  3113
  3114         1          3.0      3.0      0.0          idx = dataframe.index
  3115         1        356.0    356.0      0.0          obj = cls()
  3116
  3117         1          2.0      2.0      0.0          if isinstance(idx, pd.MultiIndex):
  3118                                                       # it's a multi-index
  3119                                                       # expand the DataFrame to include the product of all levels
  3120         1       4524.0   4524.0      0.6              full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
  3121         1     717008.0 717008.0     98.6              dataframe = dataframe.reindex(full_idx)
  3122         1          3.0      3.0      0.0              dims = [name if name is not None else 'level_%i' % n
  3123         1         20.0     20.0      0.0                      for n, name in enumerate(idx.names)]
  3124         4          9.0      2.2      0.0              for dim, lev in zip(dims, idx.levels):
  3125         3       2973.0    991.0      0.4                  obj[dim] = (dim, lev)
  3126         1         37.0     37.0      0.0              shape = [lev.size for lev in idx.levels]
  3127                                                   else:
  3128                                                       dims = (idx.name if idx.name is not None else 'index',)
  3129                                                       obj[dims[0]] = (dims, idx)
  3130                                                       shape = -1
  3131
  3132         2        350.0    175.0      0.0          for name, series in iteritems(dataframe):
  3133         1         33.0     33.0      0.0              data = np.asarray(series).reshape(shape)
  3134         1       1520.0   1520.0      0.2              obj[name] = (dims, data)
  3135         1          1.0      1.0      0.0          return obj
\`\`\`
@max-sixty nevermind, you seem to have already discovered that :)
I've run into this twice. This time I'm seeing a difference of very roughly 100x or more just using a transpose -- I can't test or time it properly right now, but this is what it looks like:

\`\`\`
ipdb> df
x              a       b  ...    c      d
y              0       0  ...    7      7
z                         ...            
0       0.000000     0.0  ...  0.0    0.0
1      -0.000416     0.0  ...  0.0    0.0

[2 rows x 2932 columns]
ipdb> df.to_xarray()

<I quit out because it takes at least 30s>

ipdb> df.T.to_xarray()

<Finishes instantly>

\`\`\`

@tqfjo unrelated. You're comparing the creation of a dataset with 2 variables with the creation of one with 3000. Unsurprisingly, the latter will take 1500x. If your dataset doesn't functionally contain 3000 variables but just a single two-dimensional variable, use ``xarray.DataArray(ds)``.
@crusaderky Thanks for the pointer to `xarray.DataArray(df)` -- that makes my life a ton easier. 

<br>

That said, if it helps anyone to know, I did just want a `DataArray`, but figured there was no alternative to first running the rather singular `to_xarray`. I also still find the runtime surprising, though I know nothing about `xarray`'s internals.




I know this is not a recent thread but I found no resolution, and we just ran in the same issue recently. In our case we had a pandas series of roughly 15 milliion entries, with a 3-level multi-index which had to be converted to an xarray.DataArray. The .to_xarray took almost 2 minutes. Unstack + to_array took it down to roughly 3 seconds, provided the last level of the multi index was unstacked. 

However a much faster solution was through numpy array. The below code is based on the [idea of Igor Raush](https://stackoverflow.com/a/35049899)

(In this case df is a dataframe with a single column, or a series)
\`\`\`
arr = np.full(df.index.levshape, np.nan)
arr[tuple(df.index.codes)] = df.values.flat
da = xr.DataArray(arr,dims=df.index.names,coords=dict(zip(df.index.names, df.index.levels)))
\`\`\`


Hi All. I stumble across the same issue trying to convert a 5000 column dataframe to xarray (it was never going to happen...).
I found a workaround and I am posting the test below. Hope it helps.

\`\`\`python
import xarray as xr
import pandas as pd
import numpy as np

xr.__version__

    '0.15.1'

pd.__version__

    '1.0.5'

df = pd.DataFrame(np.random.randn(200, 500))

%%time
one = df.to_xarray()

    CPU times: user 29.6 s, sys: 60.4 ms, total: 29.6 s
    Wall time: 29.7 s

%%time
dic={}
for name in df.columns:
    dic.update({name:(['index'],df[name].values)})

two = xr.Dataset(dic, coords={'index': ('index', df.index.values)})         

    CPU times: user 17.6 ms, sys: 158 µs, total: 17.8 ms
    Wall time: 17.8 ms

one.equals(two)

    True

\`\`\`

```

## Patch

```diff
diff --git a/asv_bench/benchmarks/pandas.py b/asv_bench/benchmarks/pandas.py
new file mode 100644
--- /dev/null
+++ b/asv_bench/benchmarks/pandas.py
@@ -0,0 +1,24 @@
+import numpy as np
+import pandas as pd
+
+from . import parameterized
+
+
+class MultiIndexSeries:
+    def setup(self, dtype, subset):
+        data = np.random.rand(100000).astype(dtype)
+        index = pd.MultiIndex.from_product(
+            [
+                list("abcdefhijk"),
+                list("abcdefhijk"),
+                pd.date_range(start="2000-01-01", periods=1000, freq="B"),
+            ]
+        )
+        series = pd.Series(data, index)
+        if subset:
+            series = series[::3]
+        self.series = series
+
+    @parameterized(["dtype", "subset"], ([int, float], [True, False]))
+    def time_to_xarray(self, dtype, subset):
+        self.series.to_xarray()
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -4543,11 +4543,10 @@ def to_dataframe(self):
         return self._to_dataframe(self.dims)
 
     def _set_sparse_data_from_dataframe(
-        self, dataframe: pd.DataFrame, dims: tuple
+        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
     ) -> None:
         from sparse import COO
 
-        idx = dataframe.index
         if isinstance(idx, pd.MultiIndex):
             coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
             is_sorted = idx.is_lexsorted()
@@ -4557,11 +4556,7 @@ def _set_sparse_data_from_dataframe(
             is_sorted = True
             shape = (idx.size,)
 
-        for name, series in dataframe.items():
-            # Cast to a NumPy array first, in case the Series is a pandas
-            # Extension array (which doesn't have a valid NumPy dtype)
-            values = np.asarray(series)
-
+        for name, values in arrays:
             # In virtually all real use cases, the sparse array will now have
             # missing values and needs a fill_value. For consistency, don't
             # special case the rare exceptions (e.g., dtype=int without a
@@ -4580,18 +4575,36 @@ def _set_sparse_data_from_dataframe(
             self[name] = (dims, data)
 
     def _set_numpy_data_from_dataframe(
-        self, dataframe: pd.DataFrame, dims: tuple
+        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
     ) -> None:
-        idx = dataframe.index
-        if isinstance(idx, pd.MultiIndex):
-            # expand the DataFrame to include the product of all levels
-            full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
-            dataframe = dataframe.reindex(full_idx)
-            shape = tuple(lev.size for lev in idx.levels)
-        else:
-            shape = (idx.size,)
-        for name, series in dataframe.items():
-            data = np.asarray(series).reshape(shape)
+        if not isinstance(idx, pd.MultiIndex):
+            for name, values in arrays:
+                self[name] = (dims, values)
+            return
+
+        shape = tuple(lev.size for lev in idx.levels)
+        indexer = tuple(idx.codes)
+
+        # We already verified that the MultiIndex has all unique values, so
+        # there are missing values if and only if the size of output arrays is
+        # larger that the index.
+        missing_values = np.prod(shape) > idx.shape[0]
+
+        for name, values in arrays:
+            # NumPy indexing is much faster than using DataFrame.reindex() to
+            # fill in missing values:
+            # https://stackoverflow.com/a/35049899/809705
+            if missing_values:
+                dtype, fill_value = dtypes.maybe_promote(values.dtype)
+                data = np.full(shape, fill_value, dtype)
+            else:
+                # If there are no missing values, keep the existing dtype
+                # instead of promoting to support NA, e.g., keep integer
+                # columns as integers.
+                # TODO: consider removing this special case, which doesn't
+                # exist for sparse=True.
+                data = np.zeros(shape, values.dtype)
+            data[indexer] = values
             self[name] = (dims, data)
 
     @classmethod
@@ -4631,7 +4644,19 @@ def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Datas
         if not dataframe.columns.is_unique:
             raise ValueError("cannot convert DataFrame with non-unique columns")
 
-        idx, dataframe = remove_unused_levels_categories(dataframe.index, dataframe)
+        idx = remove_unused_levels_categories(dataframe.index)
+
+        if isinstance(idx, pd.MultiIndex) and not idx.is_unique:
+            raise ValueError(
+                "cannot convert a DataFrame with a non-unique MultiIndex into xarray"
+            )
+
+        # Cast to a NumPy array first, in case the Series is a pandas Extension
+        # array (which doesn't have a valid NumPy dtype)
+        # TODO: allow users to control how this casting happens, e.g., by
+        # forwarding arguments to pandas.Series.to_numpy?
+        arrays = [(k, np.asarray(v)) for k, v in dataframe.items()]
+
         obj = cls()
 
         if isinstance(idx, pd.MultiIndex):
@@ -4647,9 +4672,9 @@ def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Datas
             obj[index_name] = (dims, idx)
 
         if sparse:
-            obj._set_sparse_data_from_dataframe(dataframe, dims)
+            obj._set_sparse_data_from_dataframe(idx, arrays, dims)
         else:
-            obj._set_numpy_data_from_dataframe(dataframe, dims)
+            obj._set_numpy_data_from_dataframe(idx, arrays, dims)
         return obj
 
     def to_dask_dataframe(self, dim_order=None, set_index=False):
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -9,7 +9,7 @@
 from .variable import Variable
 
 
-def remove_unused_levels_categories(index, dataframe=None):
+def remove_unused_levels_categories(index: pd.Index) -> pd.Index:
     """
     Remove unused levels from MultiIndex and unused categories from CategoricalIndex
     """
@@ -25,14 +25,15 @@ def remove_unused_levels_categories(index, dataframe=None):
                 else:
                     level = level[index.codes[i]]
                 levels.append(level)
+            # TODO: calling from_array() reorders MultiIndex levels. It would
+            # be best to avoid this, if possible, e.g., by using
+            # MultiIndex.remove_unused_levels() (which does not reorder) on the
+            # part of the MultiIndex that is not categorical, or by fixing this
+            # upstream in pandas.
             index = pd.MultiIndex.from_arrays(levels, names=index.names)
     elif isinstance(index, pd.CategoricalIndex):
         index = index.remove_unused_categories()
-
-    if dataframe is None:
-        return index
-    dataframe = dataframe.set_index(index)
-    return dataframe.index, dataframe
+    return index
 
 
 class Indexes(collections.abc.Mapping):

```

## Test Patch

```diff
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -4013,6 +4013,49 @@ def test_to_and_from_empty_dataframe(self):
         assert len(actual) == 0
         assert expected.equals(actual)
 
+    def test_from_dataframe_multiindex(self):
+        index = pd.MultiIndex.from_product([["a", "b"], [1, 2, 3]], names=["x", "y"])
+        df = pd.DataFrame({"z": np.arange(6)}, index=index)
+
+        expected = Dataset(
+            {"z": (("x", "y"), [[0, 1, 2], [3, 4, 5]])},
+            coords={"x": ["a", "b"], "y": [1, 2, 3]},
+        )
+        actual = Dataset.from_dataframe(df)
+        assert_identical(actual, expected)
+
+        df2 = df.iloc[[3, 2, 1, 0, 4, 5], :]
+        actual = Dataset.from_dataframe(df2)
+        assert_identical(actual, expected)
+
+        df3 = df.iloc[:4, :]
+        expected3 = Dataset(
+            {"z": (("x", "y"), [[0, 1, 2], [3, np.nan, np.nan]])},
+            coords={"x": ["a", "b"], "y": [1, 2, 3]},
+        )
+        actual = Dataset.from_dataframe(df3)
+        assert_identical(actual, expected3)
+
+        df_nonunique = df.iloc[[0, 0], :]
+        with raises_regex(ValueError, "non-unique MultiIndex"):
+            Dataset.from_dataframe(df_nonunique)
+
+    def test_from_dataframe_unsorted_levels(self):
+        # regression test for GH-4186
+        index = pd.MultiIndex(
+            levels=[["b", "a"], ["foo"]], codes=[[0, 1], [0, 0]], names=["lev1", "lev2"]
+        )
+        df = pd.DataFrame({"c1": [0, 2], "c2": [1, 3]}, index=index)
+        expected = Dataset(
+            {
+                "c1": (("lev1", "lev2"), [[0], [2]]),
+                "c2": (("lev1", "lev2"), [[1], [3]]),
+            },
+            coords={"lev1": ["b", "a"], "lev2": ["foo"]},
+        )
+        actual = Dataset.from_dataframe(df)
+        assert_identical(actual, expected)
+
     def test_from_dataframe_non_unique_columns(self):
         # regression test for GH449
         df = pd.DataFrame(np.zeros((2, 2)))

```


## Code snippets

### 1 - xarray/core/alignment.py:

Start line: 69, End line: 255

```python
def align(
    *objects,
    join="inner",
    copy=True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
):
    """
    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with ``fill_value``.
    The default fill value is NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {'outer', 'inner', 'left', 'right', 'exact', 'override'}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : sequence of str, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar, optional
        Value to use for newly missing values

    Returns
    -------
    aligned : same as `*objects`
        Tuple of objects with aligned coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.

    Examples
    --------

    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     [[25, 35], [10, 24]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ... )
    >>> y = xr.DataArray(
    ...     [[20, 5], [7, 13]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 120.0]},
    ... )

    >>> x
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
    * lat      (lat) float64 35.0 42.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y)
    >>> a
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[25, 35]])
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[20,  5]])
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer")
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[25., 35.],
           [10., 24.],
           [nan, nan]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[20.,  5.],
           [nan, nan],
           [ 7., 13.]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer", fill_value=-999)
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  25,   35],
           [  10,   24],
           [-999, -999]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  20,    5],
           [-999, -999],
           [   7,   13]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="left")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20.,  5.],
           [nan, nan]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="right")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25., 35.],
           [nan, nan]])
    Coordinates:
    * lat      (lat) float64 35.0 42.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
    * lat      (lat) float64 35.0 42.0
    * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="exact")
    Traceback (most recent call last):
    ...
        "indexes along dimension {!r} are not equal".format(dim)
    ValueError: indexes along dimension 'lat' are not equal

    >>> a, b = xr.align(x, y, join="override")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0

    """
    # ... other code
```
### 2 - xarray/__init__.py:

Start line: 1, End line: 93

```python
import pkg_resources

from . import testing, tutorial, ufuncs
from .backends.api import (
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    save_mfdataset,
)
from .backends.rasterio_ import open_rasterio
from .backends.zarr import open_zarr
from .coding.cftime_offsets import cftime_range
from .coding.cftimeindex import CFTimeIndex
from .coding.frequencies import infer_freq
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast
from .core.combine import combine_by_coords, combine_nested
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import apply_ufunc, corr, cov, dot, polyval, where
from .core.concat import concat
from .core.dataarray import DataArray
from .core.dataset import Dataset
from .core.extensions import register_dataarray_accessor, register_dataset_accessor
from .core.merge import MergeError, merge
from .core.options import set_options
from .core.parallel import map_blocks
from .core.variable import Coordinate, IndexVariable, Variable, as_variable
from .util.print_versions import show_versions

try:
    __version__ = pkg_resources.get_distribution("xarray").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# A hardcoded __all__ variable is necessary to appease
# `mypy --strict` running in projects that import xarray.
__all__ = (
    # Sub-packages
    "ufuncs",
    "testing",
    "tutorial",
    # Top-level functions
    "align",
    "apply_ufunc",
    "as_variable",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "decode_cf",
    "dot",
    "cov",
    "corr",
    "full_like",
    "infer_freq",
    "load_dataarray",
    "load_dataset",
    "map_blocks",
    "merge",
    "ones_like",
    "open_dataarray",
    "open_dataset",
    "open_mfdataset",
    "open_rasterio",
    "open_zarr",
    "polyval",
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "save_mfdataset",
    "set_options",
    "show_versions",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
    "Coordinate",
    "DataArray",
    "Dataset",
    "IndexVariable",
    "Variable",
    # Exceptions
    "MergeError",
    "SerializationWarning",
    # Constants
    "__version__",
    "ALL_DIMS",
)
```
### 3 - xarray/core/dataarray.py:

Start line: 1, End line: 81

```python
import datetime
import functools
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
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS
from .utils import Default, ReprObject, _check_inplace, _default, either_dict_or_kwargs
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

if TYPE_CHECKING:
    T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)

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
### 4 - asv_bench/benchmarks/indexing.py:

Start line: 1, End line: 57

```python
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
### 5 - xarray/core/dataset.py:

Start line: 1, End line: 136

```python
import copy
import datetime
import functools
import sys
import warnings
from collections import defaultdict
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
from .common import (
    DataWithCoords,
    ImplementsDatasetReduce,
    _contains_datetime_like_objects,
)
from .coordinates import (
    DatasetCoordinates,
    LevelCoordinatesSource,
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
from .pycompat import dask_array_type
from .utils import (
    Default,
    Frozen,
    SortedKeysDict,
    _check_inplace,
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
### 6 - xarray/core/merge.py:

Start line: 635, End line: 840

```python
def merge(
    objects: Iterable[Union["DataArray", "CoercibleMapping"]],
    compat: str = "no_conflicts",
    join: str = "outer",
    fill_value: object = dtypes.NA,
    combine_attrs: str = "drop",
) -> "Dataset":
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - 'override': skip comparing and pick variable from first dataset
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes in objects.

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    fill_value : scalar, optional
        Value to use for newly missing values
    combine_attrs : {'drop', 'identical', 'no_conflicts', 'override'},
                    default 'drop'
        String indicating how to combine attrs of the objects being merged:

        - 'drop': empty attrs on returned Dataset.
        - 'identical': all attrs must be the same on every object.
        - 'no_conflicts': attrs from all objects are combined, any that have
          the same name must also have the same value.
        - 'override': skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     [[1.0, 2.0], [3.0, 5.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ...     name="var1",
    ... )
    >>> y = xr.DataArray(
    ...     [[5.0, 6.0], [7.0, 8.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 150.0]},
    ...     name="var2",
    ... )
    >>> z = xr.DataArray(
    ...     [[0.0, 3.0], [4.0, 9.0]],
    ...     dims=("time", "lon"),
    ...     coords={"time": [30.0, 60.0], "lon": [100.0, 150.0]},
    ...     name="var3",
    ... )

    >>> x
    <xarray.DataArray 'var1' (lat: 2, lon: 2)>
    array([[1., 2.],
           [3., 5.]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray 'var2' (lat: 2, lon: 2)>
    array([[5., 6.],
           [7., 8.]])
    Coordinates:
    * lat      (lat) float64 35.0 42.0
    * lon      (lon) float64 100.0 150.0

    >>> z
    <xarray.DataArray 'var3' (time: 2, lon: 2)>
    array([[0., 3.],
           [4., 9.]])
    Coordinates:
    * time     (time) float64 30.0 60.0
    * lon      (lon) float64 100.0 150.0

    >>> xr.merge([x, y, z])
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="identical")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals", fill_value=-999.0)
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 -999.0 3.0 ... -999.0 -999.0 -999.0
        var2     (lat, lon) float64 5.0 -999.0 6.0 -999.0 ... -999.0 7.0 -999.0 8.0
        var3     (time, lon) float64 0.0 -999.0 3.0 4.0 -999.0 9.0

    >>> xr.merge([x, y, z], join="override")
    <xarray.Dataset>
    Dimensions:  (lat: 2, lon: 2, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 3.0 5.0
        var2     (lat, lon) float64 5.0 6.0 7.0 8.0
        var3     (time, lon) float64 0.0 3.0 4.0 9.0

    >>> xr.merge([x, y, z], join="inner")
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat="identical", join="inner")
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat="broadcast_equals", join="outer")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], join="exact")
    Traceback (most recent call last):
    ...
    ValueError: indexes along dimension 'lat' are not equal

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    """
    # ... other code
```
### 7 - asv_bench/benchmarks/unstacking.py:

Start line: 1, End line: 25

```python
import numpy as np

import xarray as xr

from . import requires_dask


class Unstacking:
    def setup(self):
        data = np.random.RandomState(0).randn(1, 1000, 500)
        self.ds = xr.DataArray(data).stack(flat_dim=["dim_1", "dim_2"])

    def time_unstack_fast(self):
        self.ds.unstack("flat_dim")

    def time_unstack_slow(self):
        self.ds[:, ::-1].unstack("flat_dim")


class UnstackingDask(Unstacking):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"flat_dim": 50})
```
### 8 - asv_bench/benchmarks/reindexing.py:

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
### 9 - asv_bench/benchmarks/dataarray_missing.py:

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
### 10 - asv_bench/benchmarks/dataset_io.py:

Start line: 1, End line: 93

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
            encoding=None,
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["bar"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="bar",
            encoding=None,
            attrs={"units": "bar units", "description": "a description"},
        )
        self.ds["baz"] = xr.DataArray(
            randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
            coords={"lon": lons, "lat": lats},
            dims=("lon", "lat"),
            name="baz",
            encoding=None,
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
### 12 - xarray/core/dataset.py:

Start line: 5161, End line: 5721

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def sortby(self, variables, ascending=True):
        """
        Sort object by labels or values (along an axis).

        Sorts the dataset, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables: str, DataArray, or list of either
            1D DataArray objects or name(s) of 1D variable(s) in
            coords/data_vars whose values are used to sort the dataset.
        ascending: boolean, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted: Dataset
            A new dataset where all the specified dims are sorted by dim
            labels.
        """
        from .dataarray import DataArray

        if not isinstance(variables, list):
            variables = [variables]
        else:
            variables = variables
        variables = [v if isinstance(v, DataArray) else self[v] for v in variables]
        aligned_vars = align(self, *variables, join="left")
        aligned_self = aligned_vars[0]
        aligned_other_vars = aligned_vars[1:]
        vars_by_dim = defaultdict(list)
        for data_array in aligned_other_vars:
            if data_array.ndim != 1:
                raise ValueError("Input DataArray is not 1-D.")
            (key,) = data_array.dims
            vars_by_dim[key].append(data_array)

        indices = {}
        for key, arrays in vars_by_dim.items():
            order = np.lexsort(tuple(reversed(arrays)))
            indices[key] = order if ascending else order[::-1]
        return aligned_self.isel(**indices)

    def quantile(
        self,
        q,
        dim=None,
        interpolation="linear",
        numeric_only=False,
        keep_attrs=None,
        skipna=True,
    ):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float in range of [0,1] or array-like of floats
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        skipna : bool, optional
            Whether to skip missing values when aggregating.

        Returns
        -------
        quantiles : Dataset
            If `q` is a single quantile, then the result is a scalar for each
            variable in data_vars. If multiple percentiles are given, first
            axis of the result corresponds to the quantile and a quantile
            dimension is added to the return Dataset. The other dimensions are
            the dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, DataArray.quantile

        Examples
        --------

        >>> ds = xr.Dataset(
        ...     {"a": (("x", "y"), [[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]])},
        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},
        ... )
        >>> ds.quantile(0)  # or ds.quantile(0, dim=...)
        <xarray.Dataset>
        Dimensions:   ()
        Coordinates:
            quantile  float64 0.0
        Data variables:
            a         float64 0.7
        >>> ds.quantile(0, dim="x")
        <xarray.Dataset>
        Dimensions:   (y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
            quantile  float64 0.0
        Data variables:
            a         (y) float64 0.7 4.2 2.6 1.5
        >>> ds.quantile([0, 0.5, 1])
        <xarray.Dataset>
        Dimensions:   (quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile) float64 0.7 3.4 9.4
        >>> ds.quantile([0, 0.5, 1], dim="x")
        <xarray.Dataset>
        Dimensions:   (quantile: 3, y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile, y) float64 0.7 4.2 2.6 1.5 3.6 ... 1.7 6.5 7.3 9.4 1.9
        """

        if isinstance(dim, str):
            dims = {dim}
        elif dim in [None, ...]:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty(
            [d for d in dims if d not in self.dims],
            "Dataset does not contain the dimensions: %s",
        )

        q = np.asarray(q, dtype=np.float64)

        variables = {}
        for name, var in self.variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (
                        not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                        or var.dtype == np.bool_
                    ):
                        if len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.quantile(
                            q,
                            dim=reduce_dims,
                            interpolation=interpolation,
                            keep_attrs=keep_attrs,
                            skipna=skipna,
                        )

            else:
                variables[name] = var

        # construct the new dataset
        coord_names = {k for k in self.coords if k in variables}
        indexes = {k: v for k, v in self.indexes.items() if k in variables}
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        new = self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )
        return new.assign_coords(quantile=q)

    def rank(self, dim, pct=False, keep_attrs=None):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within
        that set.
        Ranks begin at 1, not 0. If pct is True, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : Dataset
            Variables that do not depend on `dim` are dropped.
        """
        if dim not in self.dims:
            raise ValueError("Dataset does not contain the dimension: %s" % dim)

        variables = {}
        for name, var in self.variables.items():
            if name in self.data_vars:
                if dim in var.dims:
                    variables[name] = var.rank(dim, pct=pct)
            else:
                variables[name] = var

        coord_names = set(self.coords)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        return self._replace(variables, coord_names, attrs=attrs)

    def differentiate(self, coord, edge_order=1, datetime_unit=None):
        """ Differentiate with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: str
            The coordinate to be used to compute the gradient.
        edge_order: 1 or 2. Default 1
            N-th order accurate differences at the boundaries.
        datetime_unit: None or any of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
            'us', 'ns', 'ps', 'fs', 'as'}
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: Dataset

        See also
        --------
        numpy.gradient: corresponding numpy function
        """
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._to_numeric(datetime_unit=datetime_unit)

        variables = {}
        for k, v in self.variables.items():
            if k in self.data_vars and dim in v.dims and k not in self.coords:
                if _contains_datetime_like_objects(v):
                    v = v._to_numeric(datetime_unit=datetime_unit)
                grad = duck_array_ops.gradient(
                    v.data, coord_var, edge_order=edge_order, axis=v.get_axis_num(dim)
                )
                variables[k] = Variable(v.dims, grad)
            else:
                variables[k] = v
        return self._replace(variables)

    def integrate(self, coord, datetime_unit=None):
        """ integrate the array with the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: str, or a sequence of str
            Coordinate(s) used for the integration.
        datetime_unit
            Can be specify the unit if datetime coordinate is used. One of
            {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs',
            'as'}

        Returns
        -------
        integrated: Dataset

        See also
        --------
        DataArray.integrate
        numpy.trapz: corresponding numpy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) int64 5 5 6 6
            b        (x) int64 1 2 1 0
        >>> ds.integrate("x")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 16.5
            b        float64 3.5
        >>> ds.integrate("y")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 20.0
            b        float64 4.0
        """
        if not isinstance(coord, (list, tuple)):
            coord = (coord,)
        result = self
        for c in coord:
            result = result._integrate_one(c, datetime_unit=datetime_unit)
        return result

    def _integrate_one(self, coord, datetime_unit=None):
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._replace(
                data=datetime_to_numeric(coord_var.data, datetime_unit=datetime_unit)
            )

        variables = {}
        coord_names = set()
        for k, v in self.variables.items():
            if k in self.coords:
                if dim not in v.dims:
                    variables[k] = v
                    coord_names.add(k)
            else:
                if k in self.data_vars and dim in v.dims:
                    if _contains_datetime_like_objects(v):
                        v = datetime_to_numeric(v, datetime_unit=datetime_unit)
                    integ = duck_array_ops.trapz(
                        v.data, coord_var.data, axis=v.get_axis_num(dim)
                    )
                    v_dims = list(v.dims)
                    v_dims.remove(dim)
                    variables[k] = Variable(v_dims, integ)
                else:
                    variables[k] = v
        indexes = {k: v for k, v in self.indexes.items() if k in variables}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    @property
    def real(self):
        return self._unary_op(lambda x: x.real, keep_attrs=True)(self)

    @property
    def imag(self):
        return self._unary_op(lambda x: x.imag, keep_attrs=True)(self)

    plot = utils.UncachedAccessor(_Dataset_PlotMethods)

    def filter_by_attrs(self, **kwargs):
        """Returns a ``Dataset`` with variables that match specific conditions.

        Can pass in ``key=value`` or ``key=callable``.  A Dataset is returned
        containing only the variables for which all the filter tests pass.
        These tests are either ``key=value`` for which the attribute ``key``
        has the exact value ``value`` or the callable passed into
        ``key=callable`` returns True. The callable will be passed a single
        value, either the value of the attribute ``key`` or ``None`` if the
        DataArray does not have an attribute with the name ``key``.

        Parameters
        ----------
        **kwargs : key=value
            key : str
                Attribute name.
            value : callable or obj
                If value is a callable, it should return a boolean in the form
                of bool = func(attr) where attr is da.attrs[key].
                Otherwise, value will be compared to the each
                DataArray's attrs[key].

        Returns
        -------
        new : Dataset
            New dataset with variables filtered by attribute.

        Examples
        --------
        >>> # Create an example dataset:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ["x", "y", "time"]
        >>> temp_attr = dict(standard_name="air_potential_temperature")
        >>> precip_attr = dict(standard_name="convective_precipitation_flux")
        >>> ds = xr.Dataset(
        ...     {
        ...         "temperature": (dims, temp, temp_attr),
        ...         "precipitation": (dims, precip, precip_attr),
        ...     },
        ...     coords={
        ...         "lon": (["x", "y"], lon),
        ...         "lat": (["x", "y"], lat),
        ...         "time": pd.date_range("2014-09-06", periods=3),
        ...         "reference_time": pd.Timestamp("2014-09-05"),
        ...     },
        ... )
        >>> # Get variables matching a specific standard_name.
        >>> ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
          * x               (x) int64 0 1
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * y               (y) int64 0 1
            reference_time  datetime64[ns] 2014-09-05
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        Data variables:
            precipitation   (x, y, time) float64 4.178 2.307 6.041 6.046 0.06648 ...
        >>> # Get all variables that have a standard_name attribute.
        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * x               (x) int64 0 1
          * y               (y) int64 0 1
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Data variables:
            temperature     (x, y, time) float64 25.86 20.82 6.954 23.13 10.25 11.68 ...
            precipitation   (x, y, time) float64 5.702 0.9422 2.075 1.178 3.284 ...

        """
        selection = []
        for var_name, variable in self.variables.items():
            has_value_flag = False
            for attr_name, pattern in kwargs.items():
                attr_value = variable.attrs.get(attr_name)
                if (callable(pattern) and pattern(attr_value)) or attr_value == pattern:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                selection.append(var_name)
        return self[selection]

    def unify_chunks(self) -> "Dataset":
        """ Unify chunk size along all chunked dimensions of this Dataset.

        Returns
        -------

        Dataset with consistent chunk sizes for all dask-array variables

        See Also
        --------

        dask.array.core.unify_chunks
        """

        try:
            self.chunks
        except ValueError:  # "inconsistent chunks"
            pass
        else:
            # No variables with dask backend, or all chunks are already aligned
            return self.copy()

        # import dask is placed after the quick exit test above to allow
        # running this method if dask isn't installed and there are no chunks
        import dask.array

        ds = self.copy()

        dims_pos_map = {dim: index for index, dim in enumerate(ds.dims)}

        dask_array_names = []
        dask_unify_args = []
        for name, variable in ds.variables.items():
            if isinstance(variable.data, dask.array.Array):
                dims_tuple = [dims_pos_map[dim] for dim in variable.dims]
                dask_array_names.append(name)
                dask_unify_args.append(variable.data)
                dask_unify_args.append(dims_tuple)

        _, rechunked_arrays = dask.array.core.unify_chunks(*dask_unify_args)

        for name, new_array in zip(dask_array_names, rechunked_arrays):
            ds.variables[name]._data = new_array

        return ds
```
### 14 - xarray/core/dataset.py:

Start line: 555, End line: 1390

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    @classmethod
    def load_store(cls, store, decoder=None) -> "Dataset":
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables, attributes = store.load()
        if decoder:
            variables, attributes = decoder(variables, attributes)
        obj = cls(variables, attrs=attributes)
        obj._file_obj = store
        return obj

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)

    @property
    def attrs(self) -> Dict[Hashable, Any]:
        """Dictionary of global attributes on this dataset
        """
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self) -> Dict:
        """Dictionary of global encoding attributes on this dataset
        """
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See also
        --------
        DataArray.sizes
        """
        return self.dims

    def load(self, **kwargs) -> "Dataset":
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return this dataset.
        Unlike compute, the original dataset is modified and returned.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data
            for k, v in self.variables.items()
            if isinstance(v._data, dask_array_type)
        }
        if lazy_data:
            import dask.array as da

            # evaluate all the dask arrays simultaneously
            evaluated_data = da.compute(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        # load everything else sequentially
        for k, v in self.variables.items():
            if k not in lazy_data:
                v.load()

        return self

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token(
            (type(self), self._variables, self._coord_names, self._attrs)
        )

    def __dask_graph__(self):
        graphs = {k: v.__dask_graph__() for k, v in self.variables.items()}
        graphs = {k: v for k, v in graphs.items() if v is not None}
        if not graphs:
            return None
        else:
            try:
                from dask.highlevelgraph import HighLevelGraph

                return HighLevelGraph.merge(*graphs.values())
            except ImportError:
                from dask import sharedict

                return sharedict.merge(*graphs.values())

    def __dask_keys__(self):
        import dask

        return [
            v.__dask_keys__()
            for v in self.variables.values()
            if dask.is_dask_collection(v)
        ]

    def __dask_layers__(self):
        import dask

        return sum(
            [
                v.__dask_layers__()
                for v in self.variables.values()
                if dask.is_dask_collection(v)
            ],
            (),
        )

    @property
    def __dask_optimize__(self):
        import dask.array as da

        return da.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        import dask.array as da

        return da.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        import dask

        info = [
            (True, k, v.__dask_postcompute__())
            if dask.is_dask_collection(v)
            else (False, k, v)
            for k, v in self._variables.items()
        ]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postcompute, args

    def __dask_postpersist__(self):
        import dask

        info = [
            (True, k, v.__dask_postpersist__())
            if dask.is_dask_collection(v)
            else (False, k, v)
            for k, v in self._variables.items()
        ]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postpersist, args

    @staticmethod
    def _dask_postcompute(results, info, *args):
        variables = {}
        results2 = list(results[::-1])
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                r = results2.pop()
                result = func(r, *args2)
            else:
                result = v
            variables[k] = result

        final = Dataset._construct_direct(variables, *args)
        return final

    @staticmethod
    def _dask_postpersist(dsk, info, *args):
        variables = {}
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                result = func(dsk, *args2)
            else:
                result = v
            variables[k] = result

        return Dataset._construct_direct(variables, *args)

    def compute(self, **kwargs) -> "Dataset":
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return a new dataset.
        Unlike load, the original dataset is left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def _persist_inplace(self, **kwargs) -> "Dataset":
        """Persist all Dask arrays in memory
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data
            for k, v in self.variables.items()
            if isinstance(v._data, dask_array_type)
        }
        if lazy_data:
            import dask

            # evaluate all the dask arrays simultaneously
            evaluated_data = dask.persist(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        return self

    def persist(self, **kwargs) -> "Dataset":
        """ Trigger computation, keeping data as dask arrays

        This operation can be used to trigger computation on underlying dask
        arrays, similar to ``.compute()`` or ``.load()``.  However this
        operation keeps the data as dask arrays. This is particularly useful
        when using the dask.distributed scheduler and you want to load a large
        amount of data into distributed memory.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        new = self.copy(deep=False)
        return new._persist_inplace(**kwargs)

    @classmethod
    def _construct_direct(
        cls,
        variables,
        coord_names,
        dims=None,
        attrs=None,
        indexes=None,
        encoding=None,
        file_obj=None,
    ):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._file_obj = file_obj
        obj._encoding = encoding
        return obj

    def _replace(
        self,
        variables: Dict[Hashable, Variable] = None,
        coord_names: Set[Hashable] = None,
        dims: Dict[Any, int] = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Any, pd.Index], None, Default] = _default,
        encoding: Union[dict, None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Fastpath constructor for internal use.

        Returns an object with optionally with replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        dataset. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not _default:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is _default:
                indexes = copy.copy(self._indexes)
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj

    def _replace_with_new_dims(
        self,
        variables: Dict[Hashable, Variable],
        coord_names: set = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Hashable, pd.Index], None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )

    def _replace_vars_and_dims(
        self,
        variables: Dict[Hashable, Variable],
        coord_names: set = None,
        dims: Dict[Hashable, int] = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Deprecated version of _replace_with_new_dims().

        Unlike _replace_with_new_dims(), this method always recalculates
        indexes from variables.
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes=None, inplace=inplace
        )

    def _overwrite_indexes(self, indexes: Mapping[Any, pd.Index]) -> "Dataset":
        if not indexes:
            return self

        variables = self._variables.copy()
        new_indexes = dict(self.indexes)
        for name, idx in indexes.items():
            variables[name] = IndexVariable(name, idx)
            new_indexes[name] = idx
        obj = self._replace(variables, indexes=new_indexes)

        # switch from dimension to level names, if necessary
        dim_names: Dict[Hashable, str] = {}
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def copy(self, deep: bool = False, data: Mapping = None) -> "Dataset":
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------

        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])}, coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 -0.8079 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 0 1 2 3 4 5
            bar      (x) <U1 'a' 'b'

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            variables = {k: v.copy(deep=deep) for k, v in self._variables.items()}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")
        else:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    "Data must only contain variables in original "
                    "dataset. Extra variables: {}".format(keys_not_in_vars)
                )
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    "Data must contain all variables in original "
                    "dataset. Data is missing {}".format(keys_missing_from_data)
                )
            variables = {
                k: v.copy(deep=deep, data=data.get(k))
                for k, v in self._variables.items()
            }

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)

        return self._replace(variables, attrs=attrs)

    @property
    def _level_coords(self) -> Dict[str, Hashable]:
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords: Dict[str, Hashable] = {}
        for name, index in self.indexes.items():
            if isinstance(index, pd.MultiIndex):
                level_names = index.names
                (dim,) = self.variables[name].dims
                level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _copy_listed(self, names: Iterable[Hashable]) -> "Dataset":
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables: Dict[Hashable, Variable] = {}
        coord_names = set()
        indexes: Dict[Hashable, pd.Index] = {}

        for name in names:
            try:
                variables[name] = self._variables[name]
            except KeyError:
                ref_name, var_name, var = _get_virtual_variable(
                    self._variables, name, self._level_coords, self.dims
                )
                variables[var_name] = var
                if ref_name in self._coord_names or ref_name in self.dims:
                    coord_names.add(var_name)
                if (var_name,) == var.dims:
                    indexes[var_name] = var.to_index()

        needed_dims: Set[Hashable] = set()
        for v in variables.values():
            needed_dims.update(v.dims)

        dims = {k: self.dims[k] for k in needed_dims}

        for k in self._coord_names:
            if set(self.variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
                if k in self.indexes:
                    indexes[k] = self.indexes[k]

        return self._replace(variables, coord_names, dims, indexes=indexes)

    def _construct_dataarray(self, name: Hashable) -> "DataArray":
        """Construct a DataArray by indexing this dataset
        """
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims
            )

        needed_dims = set(variable.dims)

        coords: Dict[Hashable, Variable] = {}
        for k in self.coords:
            if set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = {k: v for k, v in self._indexes.items() if k in coords}

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)

    def __copy__(self) -> "Dataset":
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> "Dataset":
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    @property
    def _attr_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for attribute-style access
        """
        return self._item_sources + [self.attrs]

    @property
    def _item_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for key-completion
        """
        return [
            self.data_vars,
            self.coords,
            {d: self[d] for d in self.dims},
            LevelCoordinatesSource(self),
        ]

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._variables

    def __len__(self) -> int:
        return len(self.data_vars)

    def __bool__(self) -> bool:
        return bool(self.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)

    def __array__(self, dtype=None):
        raise TypeError(
            "cannot directly convert an xarray.Dataset into a "
            "numpy array. Instead, create an xarray.DataArray "
            "first, either with indexing on the Dataset or by "
            "invoking the `to_array()` method."
        )

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.variables.values())

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        return _LocIndexer(self)

    # FIXME https://github.com/python/mypy/issues/7328
    @overload
    def __getitem__(self, key: Mapping) -> "Dataset":  # type: ignore
        ...

    @overload
    def __getitem__(self, key: Hashable) -> "DataArray":  # type: ignore
        ...

    @overload
    def __getitem__(self, key: Any) -> "Dataset":
        ...

    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))

    def __setitem__(self, key: Hashable, value) -> None:
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError(
                "cannot yet use a dictionary as a key " "to set Dataset values"
            )

        self.update({key: value})

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)
        if key in self.indexes:
            assert self._indexes is not None
            del self._indexes[key]
        self._dims = calculate_dimensions(self._variables)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    def _all_compat(self, other: "Dataset", compat_str: str) -> bool:
        """Helper function for equals and identical
        """

        # some stores (e.g., scipy) do not seem to preserve order, so don't
        # require matching order for equality
        def compat(x: Variable, y: Variable) -> bool:
            return getattr(x, compat_str)(y)

        return self._coord_names == other._coord_names and utils.dict_equiv(
            self._variables, other._variables, compat=compat
        )

    def broadcast_equals(self, other: "Dataset") -> bool:
        """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        See Also
        --------
        Dataset.equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, "broadcast_equals")
        except (TypeError, AttributeError):
            return False

    def equals(self, other: "Dataset") -> bool:
        """Two Datasets are equal if they have matching variables and
        coordinates, all of which are equal.

        Datasets can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``Dataset``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, "equals")
        except (TypeError, AttributeError):
            return False

    def identical(self, other: "Dataset") -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return utils.dict_equiv(self.attrs, other.attrs) and self._all_compat(
                other, "identical"
            )
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing
        """
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables
        """
        return DataVariables(self)
```
### 21 - xarray/core/dataset.py:

Start line: 3030, End line: 3779

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def expand_dims(
        self,
        dim: Union[None, Hashable, Sequence[Hashable], Mapping[Hashable, Any]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        **dim_kwargs: Any,
    ) -> "Dataset":
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.  The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : hashable, sequence of hashable, mapping, or None
            Dimensions to include on the new variable. If provided as hashable
            or sequence of hashable, then dimensions are inserted with length
            1. If provided as a mapping, then the keys are the new dimensions
            and the values are either integers (giving the length of the new
            dimensions) or array-like (giving the coordinates of the new
            dimensions).
        axis : integer, sequence of integers, or None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a list (or tuple) of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence/ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None.

        Returns
        -------
        expanded : same type as caller
            This object, but with an additional dimension(s).
        """
        if dim is None:
            pass
        elif isinstance(dim, Mapping):
            # We're later going to modify dim in place; don't tamper with
            # the input
            dim = dict(dim)
        elif isinstance(dim, int):
            raise TypeError(
                "dim should be hashable or sequence of hashables or mapping"
            )
        elif isinstance(dim, str) or not isinstance(dim, Sequence):
            dim = {dim: 1}
        elif isinstance(dim, Sequence):
            if len(dim) != len(set(dim)):
                raise ValueError("dims should not contain duplicate values.")
            dim = {d: 1 for d in dim}

        dim = either_dict_or_kwargs(dim, dim_kwargs, "expand_dims")
        assert isinstance(dim, MutableMapping)

        if axis is None:
            axis = list(range(len(dim)))
        elif not isinstance(axis, Sequence):
            axis = [axis]

        if len(dim) != len(axis):
            raise ValueError("lengths of dim and axis should be identical.")
        for d in dim:
            if d in self.dims:
                raise ValueError(f"Dimension {d} already exists.")
            if d in self._variables and not utils.is_scalar(self._variables[d]):
                raise ValueError(
                    "{dim} already exists as coordinate or"
                    " variable name.".format(dim=d)
                )

        variables: Dict[Hashable, Variable] = {}
        coord_names = self._coord_names.copy()
        # If dim is a dict, then ensure that the values are either integers
        # or iterables.
        for k, v in dim.items():
            if hasattr(v, "__iter__"):
                # If the value for the new dimension is an iterable, then
                # save the coordinates to the variables dict, and set the
                # value within the dim dict to the length of the iterable
                # for later use.
                variables[k] = xr.IndexVariable((k,), v)
                coord_names.add(k)
                dim[k] = variables[k].size
            elif isinstance(v, int):
                pass  # Do nothing if the dimensions value is just an int
            else:
                raise TypeError(
                    "The value of new dimension {k} must be "
                    "an iterable or an int".format(k=k)
                )

        for k, v in self._variables.items():
            if k not in dim:
                if k in coord_names:  # Do not change coordinates
                    variables[k] = v
                else:
                    result_ndim = len(v.dims) + len(axis)
                    for a in axis:
                        if a < -result_ndim or result_ndim - 1 < a:
                            raise IndexError(
                                f"Axis {a} of variable {k} is out of bounds of the "
                                f"expanded dimension size {result_ndim}"
                            )

                    axis_pos = [a if a >= 0 else result_ndim + a for a in axis]
                    if len(axis_pos) != len(set(axis_pos)):
                        raise ValueError("axis should not contain duplicate values")
                    # We need to sort them to make sure `axis` equals to the
                    # axis positions of the result array.
                    zip_axis_dim = sorted(zip(axis_pos, dim.items()))

                    all_dims = list(zip(v.dims, v.shape))
                    for d, c in zip_axis_dim:
                        all_dims.insert(d, c)
                    variables[k] = v.set_dims(dict(all_dims))
            else:
                # If dims includes a label of a non-dimension coordinate,
                # it will be promoted to a 1D coordinate with a single value.
                variables[k] = v.set_dims(k).to_index_variable()

        new_dims = self._dims.copy()
        new_dims.update(dim)

        return self._replace_vars_and_dims(
            variables, dims=new_dims, coord_names=coord_names
        )

    def set_index(
        self,
        indexes: Mapping[Hashable, Union[Hashable, Sequence[Hashable]]] = None,
        append: bool = False,
        inplace: bool = None,
        **indexes_kwargs: Union[Hashable, Sequence[Hashable]],
    ) -> "Dataset":
        """Set Dataset (multi-)indexes using one or more existing coordinates
        or variables.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs: optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> ds = xr.Dataset({"v": arr})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0
        >>> ds.set_index(x="a")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0

        See Also
        --------
        Dataset.reset_index
        Dataset.swap_dims
        """
        _check_inplace(inplace)
        indexes = either_dict_or_kwargs(indexes, indexes_kwargs, "set_index")
        variables, coord_names = merge_indexes(
            indexes, self._variables, self._coord_names, append=append
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reset_index(
        self,
        dims_or_levels: Union[Hashable, Sequence[Hashable]],
        drop: bool = False,
        inplace: bool = None,
    ) -> "Dataset":
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : str or list
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        """
        _check_inplace(inplace)
        variables, coord_names = split_indexes(
            dims_or_levels,
            self._variables,
            self._coord_names,
            cast(Mapping[Hashable, Hashable], self._level_coords),
            drop=drop,
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reorder_levels(
        self,
        dim_order: Mapping[Hashable, Sequence[int]] = None,
        inplace: bool = None,
        **dim_order_kwargs: Sequence[int],
    ) -> "Dataset":
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs: optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        _check_inplace(inplace)
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs, "reorder_levels")
        variables = self._variables.copy()
        indexes = dict(self.indexes)
        for dim, order in dim_order.items():
            coord = self._variables[dim]
            index = self.indexes[dim]
            if not isinstance(index, pd.MultiIndex):
                raise ValueError(f"coordinate {dim} has no MultiIndex")
            new_index = index.reorder_levels(order)
            variables[dim] = IndexVariable(coord.dims, new_index)
            indexes[dim] = new_index

        return self._replace(variables, indexes=indexes)

    def _stack_once(self, dims, new_dim):
        if ... in dims:
            dims = list(infix_dims(dims, self.dims))
        variables = {}
        for name, var in self.variables.items():
            if name not in dims:
                if any(d in var.dims for d in dims):
                    add_dims = [d for d in dims if d not in var.dims]
                    vdims = list(var.dims) + add_dims
                    shape = [self.dims[d] for d in vdims]
                    exp_var = var.set_dims(vdims, shape)
                    stacked_var = exp_var.stack(**{new_dim: dims})
                    variables[name] = stacked_var
                else:
                    variables[name] = var.copy(deep=False)

        # consider dropping levels that are unused?
        levels = [self.get_index(dim) for dim in dims]
        idx = utils.multiindex_from_product_levels(levels, names=dims)
        variables[new_dim] = IndexVariable(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | {new_dim}

        indexes = {k: v for k, v in self.indexes.items() if k not in dims}
        indexes[new_dim] = idx

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def stack(
        self,
        dimensions: Mapping[Hashable, Sequence[Hashable]] = None,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> "Dataset":
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : Mapping of the form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace. An ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        **dimensions_kwargs:
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See also
        --------
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def to_stacked_array(
        self,
        new_dim: Hashable,
        sample_dims: Sequence[Hashable],
        variable_dim: str = "variable",
        name: Hashable = None,
    ) -> "DataArray":
        """Combine variables of differing dimensionality into a DataArray
        without broadcasting.

        This method is similar to Dataset.to_array but does not broadcast the
        variables.

        Parameters
        ----------
        new_dim : Hashable
            Name of the new stacked coordinate
        sample_dims : Sequence[Hashable]
            Dimensions that **will not** be stacked. Each array in the dataset
            must share these dimensions. For machine learning applications,
            these define the dimensions over which samples are drawn.
        variable_dim : str, optional
            Name of the level in the stacked coordinate which corresponds to
            the variables.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        stacked : DataArray
            DataArray with the specified dimensions and data variables
            stacked together. The stacked coordinate is named ``new_dim``
            and represented by a MultiIndex object with a level containing the
            data variable names. The name of this level is controlled using
            the ``variable_dim`` argument.

        See Also
        --------
        Dataset.to_array
        Dataset.stack
        DataArray.to_unstacked_dataset

        Examples
        --------
        >>> data = xr.Dataset(
        ...     data_vars={
        ...         "a": (("x", "y"), [[0, 1, 2], [3, 4, 5]]),
        ...         "b": ("x", [6, 7]),
        ...     },
        ...     coords={"y": ["u", "v", "w"]},
        ... )

        >>> data
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
        * y        (y) <U1 'u' 'v' 'w'
        Dimensions without coordinates: x
        Data variables:
            a        (x, y) int64 0 1 2 3 4 5
            b        (x) int64 6 7

        >>> data.to_stacked_array("z", sample_dims=["x"])
        <xarray.DataArray (x: 2, z: 4)>
        array([[0, 1, 2, 6],
            [3, 4, 5, 7]])
        Coordinates:
        * z         (z) MultiIndex
        - variable  (z) object 'a' 'a' 'a' 'b'
        - y         (z) object 'u' 'v' 'w' nan
        Dimensions without coordinates: x

        """
        stacking_dims = tuple(dim for dim in self.dims if dim not in sample_dims)

        for variable in self:
            dims = self[variable].dims
            dims_include_sample_dims = set(sample_dims) <= set(dims)
            if not dims_include_sample_dims:
                raise ValueError(
                    "All variables in the dataset must contain the "
                    "dimensions {}.".format(dims)
                )

        def ensure_stackable(val):
            assign_coords = {variable_dim: val.name}
            for dim in stacking_dims:
                if dim not in val.dims:
                    assign_coords[dim] = None

            expand_dims = set(stacking_dims).difference(set(val.dims))
            expand_dims.add(variable_dim)
            # must be list for .expand_dims
            expand_dims = list(expand_dims)

            return (
                val.assign_coords(**assign_coords)
                .expand_dims(expand_dims)
                .stack({new_dim: (variable_dim,) + stacking_dims})
            )

        # concatenate the arrays
        stackable_vars = [ensure_stackable(self[key]) for key in self.data_vars]
        data_array = xr.concat(stackable_vars, dim=new_dim)

        # coerce the levels of the MultiIndex to have the same type as the
        # input dimensions. This code is messy, so it might be better to just
        # input a dummy value for the singleton dimension.
        idx = data_array.indexes[new_dim]
        levels = [idx.levels[0]] + [
            level.astype(self[level.name].dtype) for level in idx.levels[1:]
        ]
        new_idx = idx.set_levels(levels)
        data_array[new_dim] = IndexVariable(new_dim, new_idx)

        if name is not None:
            data_array.name = name

        return data_array

    def _unstack_once(self, dim: Hashable, fill_value, sparse) -> "Dataset":
        index = self.get_index(dim)
        index = remove_unused_levels_categories(index)
        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)

        # take a shortcut in case the MultiIndex was not modified.
        if index.equals(full_idx):
            obj = self
        else:
            obj = self._reindex(
                {dim: full_idx}, copy=False, fill_value=fill_value, sparse=sparse
            )

        new_dim_names = index.names
        new_dim_sizes = [lev.size for lev in index.levels]

        variables: Dict[Hashable, Variable] = {}
        indexes = {k: v for k, v in self.indexes.items() if k != dim}

        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = dict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack({dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            variables[name] = IndexVariable(name, lev)
            indexes[name] = lev

        coord_names = set(self._coord_names) - {dim} | set(new_dim_names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def unstack(
        self,
        dim: Union[Hashable, Iterable[Hashable]] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> "Dataset":
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : Hashable or iterable of Hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value: value to be filled. By default, np.nan
        sparse: use sparse-array if True

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See also
        --------
        Dataset.stack
        """
        if dim is None:
            dims = [
                d for d in self.dims if isinstance(self.get_index(d), pd.MultiIndex)
            ]
        else:
            if isinstance(dim, str) or not isinstance(dim, Iterable):
                dims = [dim]
            else:
                dims = list(dim)

            missing_dims = [d for d in dims if d not in self.dims]
            if missing_dims:
                raise ValueError(
                    "Dataset does not contain the dimensions: %s" % missing_dims
                )

            non_multi_dims = [
                d for d in dims if not isinstance(self.get_index(d), pd.MultiIndex)
            ]
            if non_multi_dims:
                raise ValueError(
                    "cannot unstack dimensions that do not "
                    "have a MultiIndex: %s" % non_multi_dims
                )

        result = self.copy(deep=False)
        for dim in dims:
            result = result._unstack_once(dim, fill_value, sparse)
        return result

    def update(self, other: "CoercibleMapping", inplace: bool = None) -> "Dataset":
        """Update this dataset's variables with those from another dataset.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}


        Returns
        -------
        updated : Dataset
            Updated dataset.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.
        """
        _check_inplace(inplace)
        merge_result = dataset_update_method(self, other)
        return self._replace(inplace=True, **merge_result._asdict())

    def merge(
        self,
        other: Union["CoercibleMapping", "DataArray"],
        inplace: bool = None,
        overwrite_vars: Union[Hashable, Iterable[Hashable]] = frozenset(),
        compat: str = "no_conflicts",
        join: str = "outer",
        fill_value: Any = dtypes.NA,
    ) -> "Dataset":
        """Merge the arrays of two datasets into a single dataset.

        This method generally does not allow for overriding data, with the
        exception of attributes, which are ignored on the second dataset.
        Variables with the same name are checked for conflicts via the equals
        or identical methods.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Dataset or variables to merge with this dataset.
        overwrite_vars : Hashable or iterable of Hashable, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {'broadcast_equals', 'equals', 'identical',
                  'no_conflicts'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
            - 'no_conflicts': only values which are not null in both datasets
              must be equal. The returned dataset then contains the combination
              of all non-null values.

        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
        fill_value: scalar, optional
            Value to use for newly missing values

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).
        """
        _check_inplace(inplace)
        other = other.to_dataset() if isinstance(other, xr.DataArray) else other
        merge_result = dataset_merge_method(
            self,
            other,
            overwrite_vars=overwrite_vars,
            compat=compat,
            join=join,
            fill_value=fill_value,
        )
        return self._replace(**merge_result._asdict())

    def _assert_all_in_dataset(
        self, names: Iterable[Hashable], virtual_okay: bool = False
    ) -> None:
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError(
                "One or more of the specified variables "
                "cannot be found in this dataset"
            )

    def drop_vars(
        self, names: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
    ) -> "Dataset":
        """Drop variables from this dataset.

        Parameters
        ----------
        names : hashable or iterable of hashables
            Name(s) of variables to drop.
        errors: {'raise', 'ignore'}, optional
            If 'raise' (default), raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            dataset are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset

        """
        # the Iterable check is required for mypy
        if is_scalar(names) or not isinstance(names, Iterable):
            names = {names}
        else:
            names = set(names)
        if errors == "raise":
            self._assert_all_in_dataset(names)

        variables = {k: v for k, v in self._variables.items() if k not in names}
        coord_names = {k for k in self._coord_names if k in variables}
        indexes = {k: v for k, v in self.indexes.items() if k not in names}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def drop(self, labels=None, dim=None, *, errors="raise", **labels_kwargs):
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        Dataset.drop_vars
        Dataset.drop_sel
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if is_dict_like(labels) and not isinstance(labels, dict):
            warnings.warn(
                "dropping coordinates using `drop` is be deprecated; use drop_vars.",
                FutureWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)

        if labels_kwargs or isinstance(labels, dict):
            if dim is not None:
                raise ValueError("cannot specify dim and dict-like arguments.")
            labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        if dim is None and (is_scalar(labels) or isinstance(labels, Iterable)):
            warnings.warn(
                "dropping variables using `drop` will be deprecated; using drop_vars is encouraged.",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)
        if dim is not None:
            warnings.warn(
                "dropping labels using list-like labels is deprecated; using "
                "dict-like arguments with `drop_sel`, e.g. `ds.drop_sel(dim=[labels]).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.drop_sel({dim: labels}, errors=errors, **labels_kwargs)

        warnings.warn(
            "dropping labels using `drop` will be deprecated; using drop_sel is encouraged.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.drop_sel(labels, errors=errors)
```
### 28 - xarray/core/dataset.py:

Start line: 4401, End line: 5159

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def assign(
        self, variables: Mapping[Hashable, Any] = None, **variables_kwargs: Hashable
    ) -> "Dataset":
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        variables : mapping, value pairs
            Mapping from variables names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.
        **variables_kwargs:
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

        Returns
        -------
        ds : Dataset
            A new Dataset with the new variables in addition to all the
            existing variables.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        pandas.DataFrame.assign

        Examples
        --------
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lat            (lat) int64 10 20
        * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 18.04 12.51 17.64 9.313
            precipitation  (lat, lon) float64 0.4751 0.6827 0.3697 0.03524

        Where the value is a callable, evaluated on dataset:

        >>> x.assign(temperature_f=lambda x: x.temperature_c * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lat            (lat) int64 10 20
        * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 18.04 12.51 17.64 9.313
            precipitation  (lat, lon) float64 0.4751 0.6827 0.3697 0.03524
            temperature_f  (lat, lon) float64 64.47 54.51 63.75 48.76

        Alternatively, the same behavior can be achieved by directly referencing an existing dataarray:

        >>> x.assign(temperature_f=x["temperature_c"] * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
        * lat            (lat) int64 10 20
        * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 18.04 12.51 17.64 9.313
            precipitation  (lat, lon) float64 0.4751 0.6827 0.3697 0.03524
            temperature_f  (lat, lon) float64 64.47 54.51 63.75 48.76

        """
        variables = either_dict_or_kwargs(variables, variables_kwargs, "assign")
        data = self.copy()
        # do all calculations first...
        results = data._calc_assign_results(variables)
        # ... and then assign
        data.update(results)
        return data

    def to_array(self, dim="variable", name=None):
        """Convert this dataset into an xarray.DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : str, optional
            Name of the new dimension.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        array : xarray.DataArray
        """
        from .dataarray import DataArray

        data_vars = [self.variables[k] for k in self.data_vars]
        broadcast_vars = broadcast_variables(*data_vars)
        data = duck_array_ops.stack([b.data for b in broadcast_vars], axis=0)

        coords = dict(self.coords)
        coords[dim] = list(self.data_vars)
        indexes = propagate_indexes(self._indexes)

        dims = (dim,) + broadcast_vars[0].dims

        return DataArray(
            data, coords, dims, attrs=self.attrs, name=name, indexes=indexes
        )

    def _to_dataframe(self, ordered_dims):
        columns = [k for k in self.variables if k not in self.dims]
        data = [
            self._variables[k].set_dims(ordered_dims).values.reshape(-1)
            for k in columns
        ]
        index = self.coords.to_index(ordered_dims)
        return pd.DataFrame(dict(zip(columns, data)), index=index)

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        return self._to_dataframe(self.dims)

    def _set_sparse_data_from_dataframe(
        self, dataframe: pd.DataFrame, dims: tuple
    ) -> None:
        from sparse import COO

        idx = dataframe.index
        if isinstance(idx, pd.MultiIndex):
            coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
            is_sorted = idx.is_lexsorted()
            shape = tuple(lev.size for lev in idx.levels)
        else:
            coords = np.arange(idx.size).reshape(1, -1)
            is_sorted = True
            shape = (idx.size,)

        for name, series in dataframe.items():
            # Cast to a NumPy array first, in case the Series is a pandas
            # Extension array (which doesn't have a valid NumPy dtype)
            values = np.asarray(series)

            # In virtually all real use cases, the sparse array will now have
            # missing values and needs a fill_value. For consistency, don't
            # special case the rare exceptions (e.g., dtype=int without a
            # MultiIndex).
            dtype, fill_value = dtypes.maybe_promote(values.dtype)
            values = np.asarray(values, dtype=dtype)

            data = COO(
                coords,
                values,
                shape,
                has_duplicates=False,
                sorted=is_sorted,
                fill_value=fill_value,
            )
            self[name] = (dims, data)

    def _set_numpy_data_from_dataframe(
        self, dataframe: pd.DataFrame, dims: tuple
    ) -> None:
        idx = dataframe.index
        if isinstance(idx, pd.MultiIndex):
            # expand the DataFrame to include the product of all levels
            full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
            dataframe = dataframe.reindex(full_idx)
            shape = tuple(lev.size for lev in idx.levels)
        else:
            shape = (idx.size,)
        for name, series in dataframe.items():
            data = np.asarray(series).reshape(shape)
            self[name] = (dims, data)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Dataset":
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality)

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame from which to copy data and indices.
        sparse : bool
            If true, create a sparse arrays instead of dense numpy arrays. This
            can potentially save a large amount of memory if the DataFrame has
            a MultiIndex. Requires the sparse package (sparse.pydata.org).

        Returns
        -------
        New Dataset.

        See also
        --------
        xarray.DataArray.from_series
        pandas.DataFrame.to_xarray
        """
        # TODO: Add an option to remove dimensions along which the variables
        # are constant, to enable consistent serialization to/from a dataframe,
        # even if some variables have different dimensionality.

        if not dataframe.columns.is_unique:
            raise ValueError("cannot convert DataFrame with non-unique columns")

        idx, dataframe = remove_unused_levels_categories(dataframe.index, dataframe)
        obj = cls()

        if isinstance(idx, pd.MultiIndex):
            dims = tuple(
                name if name is not None else "level_%i" % n
                for n, name in enumerate(idx.names)
            )
            for dim, lev in zip(dims, idx.levels):
                obj[dim] = (dim, lev)
        else:
            index_name = idx.name if idx.name is not None else "index"
            dims = (index_name,)
            obj[index_name] = (dims, idx)

        if sparse:
            obj._set_sparse_data_from_dataframe(dataframe, dims)
        else:
            obj._set_numpy_data_from_dataframe(dataframe, dims)
        return obj

    def to_dask_dataframe(self, dim_order=None, set_index=False):
        """
        Convert this dataset into a dask.dataframe.DataFrame.

        The dimensions, coordinates and data variables in this dataset form
        the columns of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting dask
            dataframe.

            If provided, must include all dimensions on this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, optional
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames to not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """

        import dask.array as da
        import dask.dataframe as dd

        if dim_order is None:
            dim_order = list(self.dims)
        elif set(dim_order) != set(self.dims):
            raise ValueError(
                "dim_order {} does not match the set of dimensions on this "
                "Dataset: {}".format(dim_order, list(self.dims))
            )

        ordered_dims = {k: self.dims[k] for k in dim_order}

        columns = list(ordered_dims)
        columns.extend(k for k in self.coords if k not in self.dims)
        columns.extend(self.data_vars)

        series_list = []
        for name in columns:
            try:
                var = self.variables[name]
            except KeyError:
                # dimension without a matching coordinate
                size = self.dims[name]
                data = da.arange(size, chunks=size, dtype=np.int64)
                var = Variable((name,), data)

            # IndexVariable objects have a dummy .chunk() method
            if isinstance(var, IndexVariable):
                var = var.to_base_variable()

            dask_array = var.set_dims(ordered_dims).chunk(self.chunks).data
            series = dd.from_array(dask_array.reshape(-1), columns=[name])
            series_list.append(series)

        df = dd.concat(series_list, axis=1)

        if set_index:
            if len(dim_order) == 1:
                (dim,) = dim_order
                df = df.set_index(dim)
            else:
                # triggers an error about multi-indexes, even if only one
                # dimension is passed
                df = df.set_index(dim_order)

        return df

    def to_dict(self, data=True):
        """
        Convert this dataset to a dictionary following xarray naming
        conventions.

        Converts all variables and attributes to native Python objects
        Useful for converting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See also
        --------
        Dataset.from_dict
        """
        d = {
            "coords": {},
            "attrs": decode_numpy_dict_values(self.attrs),
            "dims": dict(self.dims),
            "data_vars": {},
        }
        for k in self.coords:
            d["coords"].update({k: self[k].variable.to_dict(data=data)})
        for k in self.data_vars:
            d["data_vars"].update({k: self[k].variable.to_dict(data=data)})
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary into an xarray.Dataset.

        Input dict can take several forms::

            d = {'t': {'dims': ('t'), 'data': t},
                 'a': {'dims': ('t'), 'data': x},
                 'b': {'dims': ('t'), 'data': y}}

            d = {'coords': {'t': {'dims': 't', 'data': t,
                                  'attrs': {'units':'s'}}},
                 'attrs': {'title': 'air temperature'},
                 'dims': 't',
                 'data_vars': {'a': {'dims': 't', 'data': x, },
                               'b': {'dims': 't', 'data': y}}}

        where 't' is the name of the dimesion, 'a' and 'b' are names of data
        variables and t, x, and y are lists, numpy.arrays or pandas objects.

        Parameters
        ----------
        d : dict, with a minimum structure of {'var_0': {'dims': [..], \
                                                         'data': [..]}, \
                                               ...}

        Returns
        -------
        obj : xarray.Dataset

        See also
        --------
        Dataset.to_dict
        DataArray.from_dict
        """

        if not {"coords", "data_vars"}.issubset(set(d)):
            variables = d.items()
        else:
            import itertools

            variables = itertools.chain(
                d.get("coords", {}).items(), d.get("data_vars", {}).items()
            )
        try:
            variable_dict = {
                k: (v["dims"], v["data"], v.get("attrs")) for k, v in variables
            }
        except KeyError as e:
            raise ValueError(
                "cannot convert dict without the key "
                "'{dims_data}'".format(dims_data=str(e.args[0]))
            )
        obj = cls(variable_dict)

        # what if coords aren't dims?
        coords = set(d.get("coords", {})) - set(d.get("dims", {}))
        obj = obj.set_coords(coords)

        obj.attrs.update(d.get("attrs", {}))

        return obj

    @staticmethod
    def _unary_op(f, keep_attrs=False):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            variables = {}
            for k, v in self._variables.items():
                if k in self._coord_names:
                    variables[k] = v
                else:
                    variables[k] = f(v, *args, **kwargs)
            attrs = self._attrs if keep_attrs else None
            return self._replace_with_new_dims(variables, attrs=attrs)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join=None):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                return NotImplemented
            align_type = OPTIONS["arithmetic_join"] if join is None else join
            if isinstance(other, (DataArray, Dataset)):
                self, other = align(self, other, join=align_type, copy=False)
            g = f if not reflexive else lambda x, y: f(y, x)
            ds = self._calculate_binary_op(g, other, join=align_type)
            return ds

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a Dataset and "
                    "a grouped object are not permitted"
                )
            # we don't actually modify arrays in-place with in-place Dataset
            # arithmetic -- this lets us automatically align things
            if isinstance(other, (DataArray, Dataset)):
                other = other.reindex_like(self, copy=False)
            g = ops.inplace_to_noninplace_op(f)
            ds = self._calculate_binary_op(g, other, inplace=True)
            self._replace_with_new_dims(
                ds._variables,
                ds._coord_names,
                attrs=ds._attrs,
                indexes=ds._indexes,
                inplace=True,
            )
            return self

        return func

    def _calculate_binary_op(self, f, other, join="inner", inplace=False):
        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if inplace and set(lhs_data_vars) != set(rhs_data_vars):
                raise ValueError(
                    "datasets must have the same data variables "
                    "for in-place arithmetic operations: %s, %s"
                    % (list(lhs_data_vars), list(rhs_data_vars))
                )

            dest_vars = {}

            for k in lhs_data_vars:
                if k in rhs_data_vars:
                    dest_vars[k] = f(lhs_vars[k], rhs_vars[k])
                elif join in ["left", "outer"]:
                    dest_vars[k] = f(lhs_vars[k], np.nan)
            for k in rhs_data_vars:
                if k not in dest_vars and join in ["right", "outer"]:
                    dest_vars[k] = f(rhs_vars[k], np.nan)
            return dest_vars

        if utils.is_dict_like(other) and not isinstance(other, Dataset):
            # can't use our shortcut of doing the binary operation with
            # Variable objects, so apply over our data vars instead.
            new_data_vars = apply_over_both(
                self.data_vars, other, self.data_vars, other
            )
            return Dataset(new_data_vars)

        other_coords = getattr(other, "coords", None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(
                self.data_vars, other.data_vars, self.variables, other.variables
            )
        else:
            other_variable = getattr(other, "variable", other)
            new_vars = {k: f(self.variables[k], other_variable) for k in self.data_vars}
        ds._variables.update(new_vars)
        ds._dims = calculate_dimensions(ds._variables)
        return ds

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs
        for v in other.variables:
            if v in self.variables:
                self.variables[v].attrs = other.variables[v].attrs

    def diff(self, dim, n=1, label="upper"):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : str, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        .. note::

            `n` matches numpy's behavior and is different from pandas' first
            argument named `periods`.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", [5, 5, 6, 6])})
        >>> ds.diff("x")
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 1 2 3
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff("x", 2)
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
        * x        (x) int64 2 3
        Data variables:
        foo      (x) int64 1 -1

        See Also
        --------
        Dataset.differentiate
        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError(f"order `n` must be non-negative but got {n}")

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == "upper":
            kwargs_new = kwargs_end
        elif label == "lower":
            kwargs_new = kwargs_start
        else:
            raise ValueError(
                "The 'label' argument has to be either " "'upper' or 'lower'"
            )

        variables = {}

        for name, var in self.variables.items():
            if dim in var.dims:
                if name in self.data_vars:
                    variables[name] = var.isel(**kwargs_end) - var.isel(**kwargs_start)
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var

        indexes = dict(self.indexes)
        if dim in indexes:
            indexes[dim] = indexes[dim][kwargs_new[dim]]

        difference = self._replace_with_new_dims(variables, indexes=indexes)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def shift(self, shifts=None, fill_value=dtypes.NA, **shifts_kwargs):
        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See also
        --------
        roll

        Examples
        --------

        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))})
        >>> ds.shift(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            foo      (x) object nan nan 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "shift")
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        variables = {}
        for name, var in self.variables.items():
            if name in self.data_vars:
                var_shifts = {k: v for k, v in shifts.items() if k in var.dims}
                variables[name] = var.shift(fill_value=fill_value, shifts=var_shifts)
            else:
                variables[name] = var

        return self._replace(variables)

    def roll(self, shifts=None, roll_coords=None, **shifts_kwargs):
        """Roll this dataset by an offset along one or more dimensions.

        Unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------

        shifts : dict, optional
            A dict with keys matching dimensions and values given
            by integers to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.
        roll_coords : bool
            Indicates whether to  roll the coordinates by the offset
            The current default of roll_coords (None, equivalent to True) is
            deprecated and will change to False in a future version.
            Explicitly pass roll_coords to silence the warning.
        **shifts_kwargs : {dim: offset, ...}, optional
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.
        Returns
        -------
        rolled : Dataset
            Dataset with the same coordinates and attributes but rolled
            variables.

        See also
        --------
        shift

        Examples
        --------

        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))})
        >>> ds.roll(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 3 4 0 1 2
        Data variables:
            foo      (x) object 'd' 'e' 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "roll")
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        if roll_coords is None:
            warnings.warn(
                "roll_coords will be set to False in the future."
                " Explicitly set roll_coords to silence warning.",
                FutureWarning,
                stacklevel=2,
            )
            roll_coords = True

        unrolled_vars = () if roll_coords else self.coords

        variables = {}
        for k, v in self.variables.items():
            if k not in unrolled_vars:
                variables[k] = v.roll(
                    **{k: s for k, s in shifts.items() if k in v.dims}
                )
            else:
                variables[k] = v

        if roll_coords:
            indexes = {}
            for k, v in self.indexes.items():
                (dim,) = self.variables[k].dims
                if dim in shifts:
                    indexes[k] = roll_index(v, shifts[dim])
                else:
                    indexes[k] = v
        else:
            indexes = dict(self.indexes)

        return self._replace(variables, indexes=indexes)
```
