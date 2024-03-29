# pydata__xarray-6400

| **pydata/xarray** | `728b648d5c7c3e22fe3704ba163012840408bf66` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 12473 |
| **Any found context length** | 11922 |
| **Avg pos** | 41.0 |
| **Min pos** | 40 |
| **Max pos** | 43 |
| **Top file pos** | 20 |
| **Missing snippets** | 8 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -520,7 +520,11 @@ def short_numpy_repr(array):
 
     # default to lower precision so a full (abbreviated) line can fit on
     # one line with the default display_width
-    options = {"precision": 6, "linewidth": OPTIONS["display_width"], "threshold": 200}
+    options = {
+        "precision": 6,
+        "linewidth": OPTIONS["display_width"],
+        "threshold": OPTIONS["display_values_threshold"],
+    }
     if array.ndim < 3:
         edgeitems = 3
     elif array.ndim == 3:
diff --git a/xarray/core/indexing.py b/xarray/core/indexing.py
--- a/xarray/core/indexing.py
+++ b/xarray/core/indexing.py
@@ -5,6 +5,7 @@
 from contextlib import suppress
 from dataclasses import dataclass, field
 from datetime import timedelta
+from html import escape
 from typing import (
     TYPE_CHECKING,
     Any,
@@ -25,6 +26,7 @@
 
 from . import duck_array_ops, nputils, utils
 from .npcompat import DTypeLike
+from .options import OPTIONS
 from .pycompat import dask_version, integer_types, is_duck_dask_array, sparse_array_type
 from .types import T_Xarray
 from .utils import either_dict_or_kwargs, get_valid_numpy_dtype
@@ -1507,23 +1509,31 @@ def __repr__(self) -> str:
             )
             return f"{type(self).__name__}{props}"
 
-    def _repr_inline_(self, max_width) -> str:
-        # special implementation to speed-up the repr for big multi-indexes
+    def _get_array_subset(self) -> np.ndarray:
+        # used to speed-up the repr for big multi-indexes
+        threshold = max(100, OPTIONS["display_values_threshold"] + 2)
+        if self.size > threshold:
+            pos = threshold // 2
+            indices = np.concatenate([np.arange(0, pos), np.arange(-pos, 0)])
+            subset = self[OuterIndexer((indices,))]
+        else:
+            subset = self
+
+        return np.asarray(subset)
+
+    def _repr_inline_(self, max_width: int) -> str:
+        from .formatting import format_array_flat
+
         if self.level is None:
             return "MultiIndex"
         else:
-            from .formatting import format_array_flat
+            return format_array_flat(self._get_array_subset(), max_width)
 
-            if self.size > 100 and max_width < self.size:
-                n_values = max_width
-                indices = np.concatenate(
-                    [np.arange(0, n_values), np.arange(-n_values, 0)]
-                )
-                subset = self[OuterIndexer((indices,))]
-            else:
-                subset = self
+    def _repr_html_(self) -> str:
+        from .formatting import short_numpy_repr
 
-            return format_array_flat(np.asarray(subset), max_width)
+        array_repr = short_numpy_repr(self._get_array_subset())
+        return f"<pre>{escape(array_repr)}</pre>"
 
     def copy(self, deep: bool = True) -> "PandasMultiIndexingAdapter":
         # see PandasIndexingAdapter.copy
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -15,6 +15,7 @@ class T_Options(TypedDict):
     cmap_divergent: Union[str, "Colormap"]
     cmap_sequential: Union[str, "Colormap"]
     display_max_rows: int
+    display_values_threshold: int
     display_style: Literal["text", "html"]
     display_width: int
     display_expand_attrs: Literal["default", True, False]
@@ -33,6 +34,7 @@ class T_Options(TypedDict):
     "cmap_divergent": "RdBu_r",
     "cmap_sequential": "viridis",
     "display_max_rows": 12,
+    "display_values_threshold": 200,
     "display_style": "html",
     "display_width": 80,
     "display_expand_attrs": "default",
@@ -57,6 +59,7 @@ def _positive_integer(value):
 _VALIDATORS = {
     "arithmetic_join": _JOIN_OPTIONS.__contains__,
     "display_max_rows": _positive_integer,
+    "display_values_threshold": _positive_integer,
     "display_style": _DISPLAY_OPTIONS.__contains__,
     "display_width": _positive_integer,
     "display_expand_attrs": lambda choice: choice in [True, False, "default"],
@@ -154,6 +157,9 @@ class set_options:
         * ``default`` : to expand unless over a pre-defined limit
     display_max_rows : int, default: 12
         Maximum display rows.
+    display_values_threshold : int, default: 200
+        Total number of array elements which trigger summarization rather
+        than full repr for variable data views (numpy arrays).
     display_style : {"text", "html"}, default: "html"
         Display style to use in jupyter for xarray objects.
     display_width : int, default: 80

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/formatting.py | 523 | 523 | - | - | -
| xarray/core/indexing.py | 8 | 8 | 40 | 20 | 11922
| xarray/core/indexing.py | 28 | 28 | 40 | 20 | 11922
| xarray/core/indexing.py | 1510 | 1526 | 43 | 20 | 12473
| xarray/core/options.py | 18 | 18 | - | - | -
| xarray/core/options.py | 36 | 36 | - | - | -
| xarray/core/options.py | 60 | 60 | - | - | -
| xarray/core/options.py | 157 | 157 | - | - | -


## Problem Statement

```
Very poor html repr performance on large multi-indexes
<!-- Please include a self-contained copy-pastable example that generates the issue if possible.

Please be concise with code posted. See guidelines below on how to provide a good bug report:

- Craft Minimal Bug Reports: http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports
- Minimal Complete Verifiable Examples: https://stackoverflow.com/help/mcve

Bug reports that follow these guidelines are easier to diagnose, and so are often handled much more quickly.
-->

**What happened**:

We have catestrophic performance on the  html repr of some long multi-indexed data arrays. Here's a case of it taking 12s.


**Minimal Complete Verifiable Example**:

\`\`\`python
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds["air"].stack(z=[...])

da.shape 

# (3869000,)

%timeit -n 1 -r 1 da._repr_html_()

# 12.4 s !!

\`\`\`

**Anything else we need to know?**:

I thought we'd fixed some issues here: https://github.com/pydata/xarray/pull/4846/files

**Environment**:

<details><summary>Output of <tt>xr.show_versions()</tt></summary>

INSTALLED VERSIONS
------------------
commit: None
python: 3.8.10 (default, May  9 2021, 13:21:55) 
[Clang 12.0.5 (clang-1205.0.22.9)]
python-bits: 64
OS: Darwin
OS-release: 20.4.0
machine: x86_64
processor: i386
byteorder: little
LC_ALL: None
LANG: None
LOCALE: ('en_US', 'UTF-8')
libhdf5: None
libnetcdf: None

xarray: 0.18.2
pandas: 1.2.4
numpy: 1.20.3
scipy: 1.6.3
netCDF4: None
pydap: None
h5netcdf: None
h5py: None
Nio: None
zarr: 2.8.3
cftime: 1.4.1
nc_time_axis: None
PseudoNetCDF: None
rasterio: 1.2.3
cfgrib: None
iris: None
bottleneck: 1.3.2
dask: 2021.06.1
distributed: 2021.06.1
matplotlib: 3.4.2
cartopy: None
seaborn: 0.11.1
numbagg: 0.2.1
pint: None
setuptools: 56.0.0
pip: 21.1.2
conda: None
pytest: 6.2.4
IPython: 7.24.0
sphinx: 4.0.1


</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 asv_bench/benchmarks/repr.py | 1 | 41| 277 | 277 | 276 | 
| 2 | 2 asv_bench/benchmarks/reindexing.py | 1 | 53| 415 | 692 | 691 | 
| 3 | 3 asv_bench/benchmarks/indexing.py | 133 | 150| 145 | 837 | 2192 | 
| 4 | 3 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 1575 | 2192 | 
| 5 | 4 asv_bench/benchmarks/dataset_io.py | 1 | 93| 737 | 2312 | 5897 | 
| 6 | 5 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 2479 | 6064 | 
| 7 | 5 asv_bench/benchmarks/indexing.py | 79 | 90| 118 | 2597 | 6064 | 
| 8 | 6 asv_bench/benchmarks/interp.py | 1 | 18| 150 | 2747 | 6515 | 
| 9 | 7 asv_bench/benchmarks/groupby.py | 1 | 20| 159 | 2906 | 7805 | 
| 10 | 8 xarray/core/dataset.py | 1 | 131| 688 | 3594 | 76929 | 
| 11 | 9 asv_bench/benchmarks/rolling.py | 1 | 17| 113 | 3707 | 78204 | 
| 12 | 10 asv_bench/benchmarks/dataarray_missing.py | 1 | 16| 114 | 3821 | 78705 | 
| 13 | 10 asv_bench/benchmarks/dataset_io.py | 435 | 479| 246 | 4067 | 78705 | 
| 14 | 10 asv_bench/benchmarks/indexing.py | 62 | 76| 151 | 4218 | 78705 | 
| 15 | 10 asv_bench/benchmarks/dataset_io.py | 222 | 296| 631 | 4849 | 78705 | 
| 16 | 10 asv_bench/benchmarks/indexing.py | 113 | 130| 147 | 4996 | 78705 | 
| 17 | 11 asv_bench/benchmarks/unstacking.py | 1 | 30| 201 | 5197 | 79199 | 
| 18 | 12 xarray/core/dataarray.py | 1 | 80| 479 | 5676 | 121566 | 
| 19 | 12 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 5941 | 121566 | 
| 20 | 12 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 6290 | 121566 | 
| 21 | 13 asv_bench/benchmarks/combine.py | 1 | 39| 334 | 6624 | 121900 | 
| 22 | 14 xarray/core/indexes.py | 1 | 31| 151 | 6775 | 132433 | 
| 23 | 14 asv_bench/benchmarks/dataarray_missing.py | 40 | 73| 238 | 7013 | 132433 | 
| 24 | 14 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 7455 | 132433 | 
| 25 | 14 asv_bench/benchmarks/groupby.py | 22 | 46| 266 | 7721 | 132433 | 
| 26 | 14 asv_bench/benchmarks/indexing.py | 93 | 110| 201 | 7922 | 132433 | 
| 27 | 14 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 8186 | 132433 | 
| 28 | 15 xarray/__init__.py | 1 | 112| 701 | 8887 | 133134 | 
| 29 | 16 doc/conf.py | 273 | 327| 703 | 9590 | 136556 | 
| 30 | 16 asv_bench/benchmarks/rolling.py | 80 | 102| 215 | 9805 | 136556 | 
| 31 | 17 xarray/core/common.py | 1 | 47| 243 | 10048 | 152410 | 
| 32 | 17 asv_bench/benchmarks/groupby.py | 98 | 108| 130 | 10178 | 152410 | 
| 33 | 17 asv_bench/benchmarks/rolling.py | 61 | 77| 207 | 10385 | 152410 | 
| 34 | 17 asv_bench/benchmarks/groupby.py | 110 | 143| 346 | 10731 | 152410 | 
| 35 | 17 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 10894 | 152410 | 
| 36 | 17 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 11163 | 152410 | 
| 37 | 18 xarray/core/missing.py | 1 | 18| 139 | 11302 | 158601 | 
| 38 | 18 asv_bench/benchmarks/rolling.py | 105 | 116| 177 | 11479 | 158601 | 
| 39 | 19 xarray/core/formatting_html.py | 246 | 285| 265 | 11744 | 160745 | 
| **-> 40 <-** | **20 xarray/core/indexing.py** | 1 | 34| 178 | 11922 | 172916 | 
| 41 | 20 asv_bench/benchmarks/unstacking.py | 51 | 65| 123 | 12045 | 172916 | 
| 42 | 20 asv_bench/benchmarks/rolling.py | 39 | 59| 219 | 12264 | 172916 | 
| **-> 43 <-** | **20 xarray/core/indexing.py** | 1510 | 1532| 209 | 12473 | 172916 | 
| 44 | **20 xarray/core/indexing.py** | 987 | 1054| 746 | 13219 | 172916 | 
| 45 | 20 xarray/core/formatting_html.py | 1 | 31| 211 | 13430 | 172916 | 
| 46 | 21 xarray/coding/times.py | 1 | 77| 478 | 13908 | 178807 | 
| 47 | 21 doc/conf.py | 1 | 101| 690 | 14598 | 178807 | 
| 48 | 21 asv_bench/benchmarks/rolling.py | 119 | 131| 172 | 14770 | 178807 | 
| 49 | 21 asv_bench/benchmarks/unstacking.py | 33 | 49| 178 | 14948 | 178807 | 
| 50 | 21 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 15060 | 178807 | 
| 51 | 21 asv_bench/benchmarks/rolling.py | 20 | 37| 178 | 15238 | 178807 | 
| 52 | 21 xarray/core/dataset.py | 1359 | 2112| 6311 | 21549 | 178807 | 
| 53 | 21 asv_bench/benchmarks/groupby.py | 49 | 58| 139 | 21688 | 178807 | 
| 54 | 22 xarray/core/groupby.py | 1 | 38| 269 | 21957 | 187216 | 
| 55 | 22 xarray/core/indexes.py | 423 | 477| 454 | 22411 | 187216 | 
| 56 | **22 xarray/core/indexing.py** | 37 | 80| 325 | 22736 | 187216 | 
| 57 | 23 xarray/core/parallel.py | 484 | 563| 790 | 23526 | 192319 | 
| 58 | 24 xarray/util/generate_reductions.py | 291 | 380| 663 | 24189 | 195276 | 


## Missing Patch Files

 * 1: xarray/core/formatting.py
 * 2: xarray/core/indexing.py
 * 3: xarray/core/options.py

### Hint

```
I think it's some lazy calculation that kicks in. Because I can reproduce using np.asarray.

\`\`\`python
import numpy as np
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds["air"].stack(z=[...])

coord = da.z.variable.to_index_variable()

# This is very slow:
a = np.asarray(coord)

da._repr_html_()
\`\`\`
![image](https://user-images.githubusercontent.com/14371165/123465543-8c6fc500-d5ee-11eb-90b3-e814b3411ad4.png)

Yes, I think it's materializing the multiindex as an array of tuples. Which we definitely shouldn't be doing for reprs.

@Illviljan nice profiling view! What is that?
One way of solving it could be to slice the arrays to a smaller size but still showing the same repr. Because `coords[0:12]` seems easy to print, not sure how tricky it is to slice it in this way though.

I'm using https://github.com/spyder-ide/spyder for the profiling and general hacking.
Yes very much so @Illviljan . But weirdly the linked PR is attempting to do that — so maybe this code path doesn't hit that change?

Spyder's profiler looks good! 
> But weirdly the linked PR is attempting to do that — so maybe this code path doesn't hit that change?

I think the linked PR only fixed the summary (inline) repr. The bottleneck here is when formatting the array detailed view for the multi-index coordinates, which triggers the conversion of the whole pandas MultiIndex (tuple elements) and each of its levels as a numpy arrays.
```

## Patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -520,7 +520,11 @@ def short_numpy_repr(array):
 
     # default to lower precision so a full (abbreviated) line can fit on
     # one line with the default display_width
-    options = {"precision": 6, "linewidth": OPTIONS["display_width"], "threshold": 200}
+    options = {
+        "precision": 6,
+        "linewidth": OPTIONS["display_width"],
+        "threshold": OPTIONS["display_values_threshold"],
+    }
     if array.ndim < 3:
         edgeitems = 3
     elif array.ndim == 3:
diff --git a/xarray/core/indexing.py b/xarray/core/indexing.py
--- a/xarray/core/indexing.py
+++ b/xarray/core/indexing.py
@@ -5,6 +5,7 @@
 from contextlib import suppress
 from dataclasses import dataclass, field
 from datetime import timedelta
+from html import escape
 from typing import (
     TYPE_CHECKING,
     Any,
@@ -25,6 +26,7 @@
 
 from . import duck_array_ops, nputils, utils
 from .npcompat import DTypeLike
+from .options import OPTIONS
 from .pycompat import dask_version, integer_types, is_duck_dask_array, sparse_array_type
 from .types import T_Xarray
 from .utils import either_dict_or_kwargs, get_valid_numpy_dtype
@@ -1507,23 +1509,31 @@ def __repr__(self) -> str:
             )
             return f"{type(self).__name__}{props}"
 
-    def _repr_inline_(self, max_width) -> str:
-        # special implementation to speed-up the repr for big multi-indexes
+    def _get_array_subset(self) -> np.ndarray:
+        # used to speed-up the repr for big multi-indexes
+        threshold = max(100, OPTIONS["display_values_threshold"] + 2)
+        if self.size > threshold:
+            pos = threshold // 2
+            indices = np.concatenate([np.arange(0, pos), np.arange(-pos, 0)])
+            subset = self[OuterIndexer((indices,))]
+        else:
+            subset = self
+
+        return np.asarray(subset)
+
+    def _repr_inline_(self, max_width: int) -> str:
+        from .formatting import format_array_flat
+
         if self.level is None:
             return "MultiIndex"
         else:
-            from .formatting import format_array_flat
+            return format_array_flat(self._get_array_subset(), max_width)
 
-            if self.size > 100 and max_width < self.size:
-                n_values = max_width
-                indices = np.concatenate(
-                    [np.arange(0, n_values), np.arange(-n_values, 0)]
-                )
-                subset = self[OuterIndexer((indices,))]
-            else:
-                subset = self
+    def _repr_html_(self) -> str:
+        from .formatting import short_numpy_repr
 
-            return format_array_flat(np.asarray(subset), max_width)
+        array_repr = short_numpy_repr(self._get_array_subset())
+        return f"<pre>{escape(array_repr)}</pre>"
 
     def copy(self, deep: bool = True) -> "PandasMultiIndexingAdapter":
         # see PandasIndexingAdapter.copy
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -15,6 +15,7 @@ class T_Options(TypedDict):
     cmap_divergent: Union[str, "Colormap"]
     cmap_sequential: Union[str, "Colormap"]
     display_max_rows: int
+    display_values_threshold: int
     display_style: Literal["text", "html"]
     display_width: int
     display_expand_attrs: Literal["default", True, False]
@@ -33,6 +34,7 @@ class T_Options(TypedDict):
     "cmap_divergent": "RdBu_r",
     "cmap_sequential": "viridis",
     "display_max_rows": 12,
+    "display_values_threshold": 200,
     "display_style": "html",
     "display_width": 80,
     "display_expand_attrs": "default",
@@ -57,6 +59,7 @@ def _positive_integer(value):
 _VALIDATORS = {
     "arithmetic_join": _JOIN_OPTIONS.__contains__,
     "display_max_rows": _positive_integer,
+    "display_values_threshold": _positive_integer,
     "display_style": _DISPLAY_OPTIONS.__contains__,
     "display_width": _positive_integer,
     "display_expand_attrs": lambda choice: choice in [True, False, "default"],
@@ -154,6 +157,9 @@ class set_options:
         * ``default`` : to expand unless over a pre-defined limit
     display_max_rows : int, default: 12
         Maximum display rows.
+    display_values_threshold : int, default: 200
+        Total number of array elements which trigger summarization rather
+        than full repr for variable data views (numpy arrays).
     display_style : {"text", "html"}, default: "html"
         Display style to use in jupyter for xarray objects.
     display_width : int, default: 80

```

## Test Patch

```diff
diff --git a/xarray/tests/test_formatting.py b/xarray/tests/test_formatting.py
--- a/xarray/tests/test_formatting.py
+++ b/xarray/tests/test_formatting.py
@@ -479,6 +479,12 @@ def test_short_numpy_repr() -> None:
         num_lines = formatting.short_numpy_repr(array).count("\n") + 1
         assert num_lines < 30
 
+    # threshold option (default: 200)
+    array = np.arange(100)
+    assert "..." not in formatting.short_numpy_repr(array)
+    with xr.set_options(display_values_threshold=10):
+        assert "..." in formatting.short_numpy_repr(array)
+
 
 def test_large_array_repr_length() -> None:
 

```


## Code snippets

### 1 - asv_bench/benchmarks/repr.py:

Start line: 1, End line: 41

```python
import numpy as np
import pandas as pd

import xarray as xr


class Repr:
    def setup(self):
        a = np.arange(0, 100)
        data_vars = dict()
        for i in a:
            data_vars[f"long_variable_name_{i}"] = xr.DataArray(
                name=f"long_variable_name_{i}",
                data=np.arange(0, 20),
                dims=[f"long_coord_name_{i}_x"],
                coords={f"long_coord_name_{i}_x": np.arange(0, 20) * 2},
            )
        self.ds = xr.Dataset(data_vars)
        self.ds.attrs = {f"attr_{k}": 2 for k in a}

    def time_repr(self):
        repr(self.ds)

    def time_repr_html(self):
        self.ds._repr_html_()


class ReprMultiIndex:
    def setup(self):
        index = pd.MultiIndex.from_product(
            [range(1000), range(1000)], names=("level_0", "level_1")
        )
        series = pd.Series(range(1000 * 1000), index=index)
        self.da = xr.DataArray(series)

    def time_repr(self):
        repr(self.da)

    def time_repr_html(self):
        self.da._repr_html_()
```
### 2 - asv_bench/benchmarks/reindexing.py:

Start line: 1, End line: 53

```python
import numpy as np

import xarray as xr

from . import requires_dask

ntime = 500
nx = 50
ny = 50


class Reindex:
    def setup(self):
        data = np.random.RandomState(0).randn(ntime, nx, ny)
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={"time": np.arange(ntime), "x": np.arange(nx), "y": np.arange(ny)},
        )

    def time_1d_coarse(self):
        self.ds.reindex(time=np.arange(0, ntime, 5)).load()

    def time_1d_fine_all_found(self):
        self.ds.reindex(time=np.arange(0, ntime, 0.5), method="nearest").load()

    def time_1d_fine_some_missing(self):
        self.ds.reindex(
            time=np.arange(0, ntime, 0.5), method="nearest", tolerance=0.1
        ).load()

    def time_2d_coarse(self):
        self.ds.reindex(x=np.arange(0, nx, 2), y=np.arange(0, ny, 2)).load()

    def time_2d_fine_all_found(self):
        self.ds.reindex(
            x=np.arange(0, nx, 0.5), y=np.arange(0, ny, 0.5), method="nearest"
        ).load()

    def time_2d_fine_some_missing(self):
        self.ds.reindex(
            x=np.arange(0, nx, 0.5),
            y=np.arange(0, ny, 0.5),
            method="nearest",
            tolerance=0.1,
        ).load()


class ReindexDask(Reindex):
    def setup(self):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
```
### 3 - asv_bench/benchmarks/indexing.py:

Start line: 133, End line: 150

```python
class HugeAxisSmallSliceIndexing:
    # https://github.com/pydata/xarray/pull/4560
    def setup(self):
        self.filepath = "test_indexing_huge_axis_small_slice.nc"
        if not os.path.isfile(self.filepath):
            xr.Dataset(
                {"a": ("x", np.arange(10_000_000))},
                coords={"x": np.arange(10_000_000)},
            ).to_netcdf(self.filepath, format="NETCDF4")

        self.ds = xr.open_dataset(self.filepath)

    def time_indexing(self):
        self.ds.isel(x=slice(100))

    def cleanup(self):
        self.ds.close()
```
### 4 - asv_bench/benchmarks/indexing.py:

Start line: 1, End line: 59

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randint, randn, requires_dask

nx = 2000
ny = 1000
nt = 500

basic_indexes = {
    "1slice": {"x": slice(0, 3)},
    "1slice-1scalar": {"x": 0, "y": slice(None, None, 3)},
    "2slicess-1scalar": {"x": slice(3, -3, 3), "y": 1, "t": slice(None, -3, 3)},
}

basic_assignment_values = {
    "1slice": xr.DataArray(randn((3, ny), frac_nan=0.1), dims=["x", "y"]),
    "1slice-1scalar": xr.DataArray(randn(int(ny / 3) + 1, frac_nan=0.1), dims=["y"]),
    "2slicess-1scalar": xr.DataArray(
        randn(np.empty(nx)[slice(3, -3, 3)].size, frac_nan=0.1), dims=["x"]
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
    "1-1d": xr.DataArray(randn((400, ny)), dims=["a", "y"], coords={"a": randn(400)}),
    "2-1d": xr.DataArray(randn(400), dims=["a"], coords={"a": randn(400)}),
    "3-2d": xr.DataArray(
        randn((4, 100)), dims=["a", "b"], coords={"a": randn(4), "b": randn(100)}
    ),
}
```
### 5 - asv_bench/benchmarks/dataset_io.py:

Start line: 1, End line: 93

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import _skip_slow, randint, randn, requires_dask

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
        # TODO: Lazily skipped in CI as it is very demanding and slow.
        # Improve times and remove errors.
        _skip_slow()

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
### 6 - asv_bench/benchmarks/pandas.py:

Start line: 1, End line: 27

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized


class MultiIndexSeries:
    def setup(self, dtype, subset):
        data = np.random.rand(100000).astype(dtype)
        index = pd.MultiIndex.from_product(
            [
                list("abcdefhijk"),
                list("abcdefhijk"),
                pd.date_range(start="2000-01-01", periods=1000, freq="B"),
            ]
        )
        series = pd.Series(data, index)
        if subset:
            series = series[::3]
        self.series = series

    @parameterized(["dtype", "subset"], ([int, float], [True, False]))
    def time_from_series(self, dtype, subset):
        xr.DataArray.from_series(self.series)
```
### 7 - asv_bench/benchmarks/indexing.py:

Start line: 79, End line: 90

```python
class Indexing(Base):
    @parameterized(["key"], [list(basic_indexes.keys())])
    def time_indexing_basic(self, key):
        self.ds.isel(**basic_indexes[key]).load()

    @parameterized(["key"], [list(outer_indexes.keys())])
    def time_indexing_outer(self, key):
        self.ds.isel(**outer_indexes[key]).load()

    @parameterized(["key"], [list(vectorized_indexes.keys())])
    def time_indexing_vectorized(self, key):
        self.ds.isel(**vectorized_indexes[key]).load()
```
### 8 - asv_bench/benchmarks/interp.py:

Start line: 1, End line: 18

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 1500
ny = 1000
nt = 500

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))

new_x_short = np.linspace(0.3 * nx, 0.7 * nx, 100)
new_x_long = np.linspace(0.3 * nx, 0.7 * nx, 500)
new_y_long = np.linspace(0.1, 0.9, 500)
```
### 9 - asv_bench/benchmarks/groupby.py:

Start line: 1, End line: 20

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import _skip_slow, parameterized, requires_dask


class GroupBy:
    def setup(self, *args, **kwargs):
        self.n = 100
        self.ds1d = xr.Dataset(
            {
                "a": xr.DataArray(np.r_[np.repeat(1, self.n), np.repeat(2, self.n)]),
                "b": xr.DataArray(np.arange(2 * self.n)),
            }
        )
        self.ds2d = self.ds1d.expand_dims(z=10)
        self.ds1d_mean = self.ds1d.groupby("b").mean()
        self.ds2d_mean = self.ds2d.groupby("b").mean()
```
### 10 - xarray/core/dataset.py:

Start line: 1, End line: 131

```python
from __future__ import annotations

import copy
import datetime
import inspect
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
    overload,
)

import numpy as np
import pandas as pd

import xarray as xr

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
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
from ._reductions import DatasetReductions
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from .computation import unify_chunks
from .coordinates import DatasetCoordinates, assert_coordinate_consistent
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
    assert_no_index_corrupted,
    create_default_index_implicit,
    filter_indexes_from_coords,
    isel_indexes,
    remove_unused_levels_categories,
    roll_indexes,
)
from .indexing import is_fancy_indexer, map_index_queries
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
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
    broadcast_variables,
    calculate_dimensions,
)

if TYPE_CHECKING:
    from ..backends import AbstractDataStore, ZarrStore
    from .dataarray import DataArray
    from .merge import CoercibleMapping
    from .types import T_Xarray

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
### 40 - xarray/core/indexing.py:

Start line: 1, End line: 34

```python
import enum
import functools
import operator
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
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
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from packaging.version import Version

from . import duck_array_ops, nputils, utils
from .npcompat import DTypeLike
from .pycompat import dask_version, integer_types, is_duck_dask_array, sparse_array_type
from .types import T_Xarray
from .utils import either_dict_or_kwargs, get_valid_numpy_dtype

if TYPE_CHECKING:
    from .indexes import Index
    from .variable import Variable
```
### 43 - xarray/core/indexing.py:

Start line: 1510, End line: 1532

```python
class PandasMultiIndexingAdapter(PandasIndexingAdapter):

    def _repr_inline_(self, max_width) -> str:
        # special implementation to speed-up the repr for big multi-indexes
        if self.level is None:
            return "MultiIndex"
        else:
            from .formatting import format_array_flat

            if self.size > 100 and max_width < self.size:
                n_values = max_width
                indices = np.concatenate(
                    [np.arange(0, n_values), np.arange(-n_values, 0)]
                )
                subset = self[OuterIndexer((indices,))]
            else:
                subset = self

            return format_array_flat(np.asarray(subset), max_width)

    def copy(self, deep: bool = True) -> "PandasMultiIndexingAdapter":
        # see PandasIndexingAdapter.copy
        array = self.array.copy(deep=True) if deep else self.array
        return type(self)(array, self._dtype, self.level)
```
### 44 - xarray/core/indexing.py:

Start line: 987, End line: 1054

```python
def _decompose_outer_indexer(
    indexer: Union[BasicIndexer, OuterIndexer],
    shape: Tuple[int, ...],
    indexing_support: IndexingSupport,
) -> Tuple[ExplicitIndexer, ExplicitIndexer]:
    # ... other code

    if indexing_support is IndexingSupport.OUTER_1VECTOR:
        # some backends such as h5py supports only 1 vector in indexers
        # We choose the most efficient axis
        gains = [
            (np.max(k) - np.min(k) + 1.0) / len(np.unique(k))
            if isinstance(k, np.ndarray)
            else 0
            for k in indexer_elems
        ]
        array_index = np.argmax(np.array(gains)) if len(gains) > 0 else None

        for i, (k, s) in enumerate(zip(indexer_elems, shape)):
            if isinstance(k, np.ndarray) and i != array_index:
                # np.ndarray key is converted to slice that covers the entire
                # entries of this key.
                backend_indexer.append(slice(np.min(k), np.max(k) + 1))
                np_indexer.append(k - np.min(k))
            elif isinstance(k, np.ndarray):
                # Remove duplicates and sort them in the increasing order
                pkey, ekey = np.unique(k, return_inverse=True)
                backend_indexer.append(pkey)
                np_indexer.append(ekey)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            else:  # slice:  convert positive step slice for backend
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)

        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))

    if indexing_support == IndexingSupport.OUTER:
        for k, s in zip(indexer_elems, shape):
            if isinstance(k, slice):
                # slice:  convert positive step slice for backend
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            elif isinstance(k, np.ndarray) and (np.diff(k) >= 0).all():
                backend_indexer.append(k)
                np_indexer.append(slice(None))
            else:
                # Remove duplicates and sort them in the increasing order
                oind, vind = np.unique(k, return_inverse=True)
                backend_indexer.append(oind)
                np_indexer.append(vind.reshape(*k.shape))

        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))

    # basic indexer
    assert indexing_support == IndexingSupport.BASIC

    for k, s in zip(indexer_elems, shape):
        if isinstance(k, np.ndarray):
            # np.ndarray key is converted to slice that covers the entire
            # entries of this key.
            backend_indexer.append(slice(np.min(k), np.max(k) + 1))
            np_indexer.append(k - np.min(k))
        elif isinstance(k, integer_types):
            backend_indexer.append(k)
        else:  # slice:  convert positive step slice for backend
            bk_slice, np_slice = _decompose_slice(k, s)
            backend_indexer.append(bk_slice)
            np_indexer.append(np_slice)

    return (BasicIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))
```
### 56 - xarray/core/indexing.py:

Start line: 37, End line: 80

```python
@dataclass
class IndexSelResult:
    """Index query results.

    Attributes
    ----------
    dim_indexers: dict
        A dictionary where keys are array dimensions and values are
        location-based indexers.
    indexes: dict, optional
        New indexes to replace in the resulting DataArray or Dataset.
    variables : dict, optional
        New variables to replace in the resulting DataArray or Dataset.
    drop_coords : list, optional
        Coordinate(s) to drop in the resulting DataArray or Dataset.
    drop_indexes : list, optional
        Index(es) to drop in the resulting DataArray or Dataset.
    rename_dims : dict, optional
        A dictionary in the form ``{old_dim: new_dim}`` for dimension(s) to
        rename in the resulting DataArray or Dataset.

    """

    dim_indexers: Dict[Any, Any]
    indexes: Dict[Any, "Index"] = field(default_factory=dict)
    variables: Dict[Any, "Variable"] = field(default_factory=dict)
    drop_coords: List[Hashable] = field(default_factory=list)
    drop_indexes: List[Hashable] = field(default_factory=list)
    rename_dims: Dict[Any, Hashable] = field(default_factory=dict)

    def as_tuple(self):
        """Unlike ``dataclasses.astuple``, return a shallow copy.

        See https://stackoverflow.com/a/51802661

        """
        return (
            self.dim_indexers,
            self.indexes,
            self.variables,
            self.drop_coords,
            self.drop_indexes,
            self.rename_dims,
        )
```
