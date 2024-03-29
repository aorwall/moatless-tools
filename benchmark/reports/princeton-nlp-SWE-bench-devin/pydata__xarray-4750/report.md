# pydata__xarray-4750

| **pydata/xarray** | `0f1eb96c924bad60ea87edd9139325adabfefa33` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 23 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -365,12 +365,23 @@ def _calculate_col_width(col_items):
     return col_width
 
 
-def _mapping_repr(mapping, title, summarizer, col_width=None):
+def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
     if col_width is None:
         col_width = _calculate_col_width(mapping)
+    if max_rows is None:
+        max_rows = OPTIONS["display_max_rows"]
     summary = [f"{title}:"]
     if mapping:
-        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
+        if len(mapping) > max_rows:
+            first_rows = max_rows // 2 + max_rows % 2
+            items = list(mapping.items())
+            summary += [summarizer(k, v, col_width) for k, v in items[:first_rows]]
+            if max_rows > 1:
+                last_rows = max_rows // 2
+                summary += [pretty_print("    ...", col_width) + " ..."]
+                summary += [summarizer(k, v, col_width) for k, v in items[-last_rows:]]
+        else:
+            summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
     else:
         summary += [EMPTY_REPR]
     return "\n".join(summary)
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -1,26 +1,28 @@
 import warnings
 
-DISPLAY_WIDTH = "display_width"
 ARITHMETIC_JOIN = "arithmetic_join"
+CMAP_DIVERGENT = "cmap_divergent"
+CMAP_SEQUENTIAL = "cmap_sequential"
+DISPLAY_MAX_ROWS = "display_max_rows"
+DISPLAY_STYLE = "display_style"
+DISPLAY_WIDTH = "display_width"
 ENABLE_CFTIMEINDEX = "enable_cftimeindex"
 FILE_CACHE_MAXSIZE = "file_cache_maxsize"
-WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
-CMAP_SEQUENTIAL = "cmap_sequential"
-CMAP_DIVERGENT = "cmap_divergent"
 KEEP_ATTRS = "keep_attrs"
-DISPLAY_STYLE = "display_style"
+WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
 
 
 OPTIONS = {
-    DISPLAY_WIDTH: 80,
     ARITHMETIC_JOIN: "inner",
+    CMAP_DIVERGENT: "RdBu_r",
+    CMAP_SEQUENTIAL: "viridis",
+    DISPLAY_MAX_ROWS: 12,
+    DISPLAY_STYLE: "html",
+    DISPLAY_WIDTH: 80,
     ENABLE_CFTIMEINDEX: True,
     FILE_CACHE_MAXSIZE: 128,
-    WARN_FOR_UNCLOSED_FILES: False,
-    CMAP_SEQUENTIAL: "viridis",
-    CMAP_DIVERGENT: "RdBu_r",
     KEEP_ATTRS: "default",
-    DISPLAY_STYLE: "html",
+    WARN_FOR_UNCLOSED_FILES: False,
 }
 
 _JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
@@ -32,13 +34,14 @@ def _positive_integer(value):
 
 
 _VALIDATORS = {
-    DISPLAY_WIDTH: _positive_integer,
     ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
+    DISPLAY_MAX_ROWS: _positive_integer,
+    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
+    DISPLAY_WIDTH: _positive_integer,
     ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
     FILE_CACHE_MAXSIZE: _positive_integer,
-    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
     KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
-    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
+    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
 }
 
 
@@ -57,8 +60,8 @@ def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
 
 
 _SETTERS = {
-    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
     ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
+    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
 }
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/formatting.py | 368 | 373 | - | 23 | -
| xarray/core/options.py | 3 | 23 | - | - | -
| xarray/core/options.py | 35 | 41 | - | - | -
| xarray/core/options.py | 60 | 60 | - | - | -


## Problem Statement

```
Limit number of data variables shown in repr
<!-- Please include a self-contained copy-pastable example that generates the issue if possible.

Please be concise with code posted. See guidelines below on how to provide a good bug report:

- Craft Minimal Bug Reports: http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports
- Minimal Complete Verifiable Examples: https://stackoverflow.com/help/mcve

Bug reports that follow these guidelines are easier to diagnose, and so are often handled much more quickly.
-->

**What happened**:
xarray feels very unresponsive when using datasets with >2000 data variables because it has to print all the 2000 variables everytime you print something to console.

**What you expected to happen**:
xarray should limit the number of variables printed to console. Maximum maybe 25?
Same idea probably apply to dimensions, coordinates and attributes as well,

pandas only shows 2 for reference, the first and last variables.

**Minimal Complete Verifiable Example**:

\`\`\`python
import numpy as np
import xarray as xr

a = np.arange(0, 2000)
b = np.core.defchararray.add("long_variable_name", a.astype(str))
data_vars = dict()
for v in b:
    data_vars[v] = xr.DataArray(
        name=v,
        data=[3, 4],
        dims=["time"],
        coords=dict(time=[0, 1])
    )
ds = xr.Dataset(data_vars)

# Everything above feels fast. Printing to console however takes about 13 seconds for me:
print(ds)
\`\`\`

**Anything else we need to know?**:
Out of scope brainstorming:
Though printing 2000 variables is probably madness for most people it is kind of nice to show all variables because you sometimes want to know what happened to a few other variables as well. Is there already an easy and fast way to create subgroup of the dataset, so we don' have to rely on the dataset printing everything to the console everytime?

**Environment**:

<details><summary>Output of <tt>xr.show_versions()</tt></summary>

xr.show_versions()

INSTALLED VERSIONS
------------------
commit: None
python: 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
python-bits: 64
OS: Windows
OS-release: 10

libhdf5: 1.10.4
libnetcdf: None

xarray: 0.16.2
pandas: 1.1.5
numpy: 1.17.5
scipy: 1.4.1
netCDF4: None
pydap: None
h5netcdf: None
h5py: 2.10.0
Nio: None
zarr: None
cftime: None
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: 1.3.2
dask: 2020.12.0
distributed: 2020.12.0
matplotlib: 3.3.2
cartopy: None
seaborn: 0.11.1
numbagg: None
pint: None
setuptools: 51.0.0.post20201207
pip: 20.3.3
conda: 4.9.2
pytest: 6.2.1
IPython: 7.19.0
sphinx: 3.4.0


</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/__init__.py | 1 | 93| 603 | 603 | 603 | 
| 2 | 2 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 1336 | 2163 | 
| 3 | 3 xarray/util/print_versions.py | 80 | 161| 707 | 2043 | 3375 | 
| 4 | 4 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 2758 | 6980 | 
| 5 | 5 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 2942 | 7465 | 
| 6 | 6 xarray/core/dataarray.py | 1 | 82| 436 | 3378 | 43830 | 
| 7 | 7 asv_bench/benchmarks/rolling.py | 1 | 17| 117 | 3495 | 44459 | 
| 8 | 8 xarray/core/dataset.py | 1 | 135| 643 | 4138 | 101893 | 
| 9 | 9 xarray/core/merge.py | 635 | 842| 2740 | 6878 | 109760 | 
| 10 | 10 doc/conf.py | 1 | 105| 751 | 7629 | 113309 | 
| 11 | 11 asv_bench/benchmarks/pandas.py | 1 | 25| 160 | 7789 | 113469 | 
| 12 | 11 asv_bench/benchmarks/dataset_io.py | 222 | 296| 613 | 8402 | 113469 | 
| 13 | 12 asv_bench/benchmarks/combine.py | 1 | 39| 332 | 8734 | 113801 | 
| 14 | 13 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 9147 | 114214 | 
| 15 | 14 xarray/core/missing.py | 1 | 18| 133 | 9280 | 120247 | 
| 16 | 15 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 9811 | 120778 | 
| 17 | 16 xarray/core/variable.py | 1546 | 1573| 262 | 10073 | 142737 | 
| 18 | 17 xarray/core/common.py | 1 | 35| 183 | 10256 | 158083 | 
| 19 | 18 xarray/plot/dataset_plot.py | 400 | 450| 390 | 10646 | 161493 | 
| 20 | 19 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 10805 | 161652 | 
| 21 | 19 xarray/core/variable.py | 1 | 81| 441 | 11246 | 161652 | 
| 22 | 20 xarray/core/computation.py | 509 | 534| 196 | 11442 | 175766 | 
| 23 | 20 xarray/core/dataset.py | 458 | 6851| 411 | 11853 | 175766 | 
| 24 | 20 xarray/core/variable.py | 1354 | 1373| 202 | 12055 | 175766 | 
| 25 | 20 xarray/core/variable.py | 1329 | 1352| 371 | 12426 | 175766 | 
| 26 | 21 xarray/plot/utils.py | 1 | 54| 283 | 12709 | 182394 | 
| 27 | 22 xarray/core/combine.py | 552 | 750| 2476 | 15185 | 189633 | 
| 28 | 22 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 15627 | 189633 | 
| 29 | 22 asv_bench/benchmarks/indexing.py | 145 | 162| 145 | 15772 | 189633 | 
| 30 | 22 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 16036 | 189633 | 
| 31 | **23 xarray/core/formatting.py** | 496 | 525| 206 | 16242 | 194877 | 


## Missing Patch Files

 * 1: xarray/core/formatting.py
 * 2: xarray/core/options.py

### Hint

```
👍🏽 on adding a configurable option to the list of options supported via `xr.set_options()`

\`\`\`python
import xarray as xr
xr.set_options(display_max_num_variables=25)
\`\`\`


Yes, this sounds like a welcome new feature! As a general rule, the output of repr() should fit on one screen.
```

## Patch

```diff
diff --git a/xarray/core/formatting.py b/xarray/core/formatting.py
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -365,12 +365,23 @@ def _calculate_col_width(col_items):
     return col_width
 
 
-def _mapping_repr(mapping, title, summarizer, col_width=None):
+def _mapping_repr(mapping, title, summarizer, col_width=None, max_rows=None):
     if col_width is None:
         col_width = _calculate_col_width(mapping)
+    if max_rows is None:
+        max_rows = OPTIONS["display_max_rows"]
     summary = [f"{title}:"]
     if mapping:
-        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
+        if len(mapping) > max_rows:
+            first_rows = max_rows // 2 + max_rows % 2
+            items = list(mapping.items())
+            summary += [summarizer(k, v, col_width) for k, v in items[:first_rows]]
+            if max_rows > 1:
+                last_rows = max_rows // 2
+                summary += [pretty_print("    ...", col_width) + " ..."]
+                summary += [summarizer(k, v, col_width) for k, v in items[-last_rows:]]
+        else:
+            summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
     else:
         summary += [EMPTY_REPR]
     return "\n".join(summary)
diff --git a/xarray/core/options.py b/xarray/core/options.py
--- a/xarray/core/options.py
+++ b/xarray/core/options.py
@@ -1,26 +1,28 @@
 import warnings
 
-DISPLAY_WIDTH = "display_width"
 ARITHMETIC_JOIN = "arithmetic_join"
+CMAP_DIVERGENT = "cmap_divergent"
+CMAP_SEQUENTIAL = "cmap_sequential"
+DISPLAY_MAX_ROWS = "display_max_rows"
+DISPLAY_STYLE = "display_style"
+DISPLAY_WIDTH = "display_width"
 ENABLE_CFTIMEINDEX = "enable_cftimeindex"
 FILE_CACHE_MAXSIZE = "file_cache_maxsize"
-WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
-CMAP_SEQUENTIAL = "cmap_sequential"
-CMAP_DIVERGENT = "cmap_divergent"
 KEEP_ATTRS = "keep_attrs"
-DISPLAY_STYLE = "display_style"
+WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
 
 
 OPTIONS = {
-    DISPLAY_WIDTH: 80,
     ARITHMETIC_JOIN: "inner",
+    CMAP_DIVERGENT: "RdBu_r",
+    CMAP_SEQUENTIAL: "viridis",
+    DISPLAY_MAX_ROWS: 12,
+    DISPLAY_STYLE: "html",
+    DISPLAY_WIDTH: 80,
     ENABLE_CFTIMEINDEX: True,
     FILE_CACHE_MAXSIZE: 128,
-    WARN_FOR_UNCLOSED_FILES: False,
-    CMAP_SEQUENTIAL: "viridis",
-    CMAP_DIVERGENT: "RdBu_r",
     KEEP_ATTRS: "default",
-    DISPLAY_STYLE: "html",
+    WARN_FOR_UNCLOSED_FILES: False,
 }
 
 _JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
@@ -32,13 +34,14 @@ def _positive_integer(value):
 
 
 _VALIDATORS = {
-    DISPLAY_WIDTH: _positive_integer,
     ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
+    DISPLAY_MAX_ROWS: _positive_integer,
+    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
+    DISPLAY_WIDTH: _positive_integer,
     ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
     FILE_CACHE_MAXSIZE: _positive_integer,
-    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
     KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
-    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
+    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
 }
 
 
@@ -57,8 +60,8 @@ def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
 
 
 _SETTERS = {
-    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
     ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
+    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
 }
 
 

```

## Test Patch

```diff
diff --git a/xarray/tests/test_formatting.py b/xarray/tests/test_formatting.py
--- a/xarray/tests/test_formatting.py
+++ b/xarray/tests/test_formatting.py
@@ -463,3 +463,36 @@ def test_large_array_repr_length():
 
     result = repr(da).splitlines()
     assert len(result) < 50
+
+
+@pytest.mark.parametrize(
+    "display_max_rows, n_vars, n_attr",
+    [(50, 40, 30), (35, 40, 30), (11, 40, 30), (1, 40, 30)],
+)
+def test__mapping_repr(display_max_rows, n_vars, n_attr):
+    long_name = "long_name"
+    a = np.core.defchararray.add(long_name, np.arange(0, n_vars).astype(str))
+    b = np.core.defchararray.add("attr_", np.arange(0, n_attr).astype(str))
+    attrs = {k: 2 for k in b}
+    coords = dict(time=np.array([0, 1]))
+    data_vars = dict()
+    for v in a:
+        data_vars[v] = xr.DataArray(
+            name=v,
+            data=np.array([3, 4]),
+            dims=["time"],
+            coords=coords,
+        )
+    ds = xr.Dataset(data_vars)
+    ds.attrs = attrs
+
+    with xr.set_options(display_max_rows=display_max_rows):
+
+        # Parse the data_vars print and show only data_vars rows:
+        summary = formatting.data_vars_repr(ds.data_vars).split("\n")
+        summary = [v for v in summary if long_name in v]
+
+        # The length should be less than or equal to display_max_rows:
+        len_summary = len(summary)
+        data_vars_print_size = min(display_max_rows, len_summary)
+        assert len_summary == data_vars_print_size

```


## Code snippets

### 1 - xarray/__init__.py:

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
### 3 - xarray/util/print_versions.py:

Start line: 80, End line: 161

```python
def show_versions(file=sys.stdout):
    """print the versions of xarray and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("xarray", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        # xarray optionals
        ("netCDF4", lambda mod: mod.__version__),
        ("pydap", lambda mod: mod.__version__),
        ("h5netcdf", lambda mod: mod.__version__),
        ("h5py", lambda mod: mod.__version__),
        ("Nio", lambda mod: mod.__version__),
        ("zarr", lambda mod: mod.__version__),
        ("cftime", lambda mod: mod.__version__),
        ("nc_time_axis", lambda mod: mod.__version__),
        ("PseudoNetCDF", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("cfgrib", lambda mod: mod.__version__),
        ("iris", lambda mod: mod.__version__),
        ("bottleneck", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("distributed", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        ("cartopy", lambda mod: mod.__version__),
        ("seaborn", lambda mod: mod.__version__),
        ("numbagg", lambda mod: mod.__version__),
        ("pint", lambda mod: mod.__version__),
        # xarray setup/test
        ("setuptools", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
        ("conda", lambda mod: mod.__version__),
        ("pytest", lambda mod: mod.__version__),
        # Misc.
        ("IPython", lambda mod: mod.__version__),
        ("sphinx", lambda mod: mod.__version__),
    ]

    deps_blob = []
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except Exception:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, "installed"))

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    print("", file=file)
    for k, stat in deps_blob:
        print(f"{k}: {stat}", file=file)


if __name__ == "__main__":
    show_versions()
```
### 4 - asv_bench/benchmarks/dataset_io.py:

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
### 5 - asv_bench/benchmarks/interp.py:

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
### 6 - xarray/core/dataarray.py:

Start line: 1, End line: 82

```python
import datetime
import functools
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
from .options import OPTIONS, _get_keep_attrs
from .utils import Default, ReprObject, _default, either_dict_or_kwargs
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
### 7 - asv_bench/benchmarks/rolling.py:

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
### 8 - xarray/core/dataset.py:

Start line: 1, End line: 135

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
from .pycompat import is_duck_dask_array
from .utils import (
    Default,
    Frozen,
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
### 9 - xarray/core/merge.py:

Start line: 635, End line: 842

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
    objects : iterable of Dataset or iterable of DataArray or iterable of dict-like
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects.

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    combine_attrs : {"drop", "identical", "no_conflicts", "override"}, \
                    default: "drop"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "override": skip comparing and copy attrs from the first dataset to
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
### 10 - doc/conf.py:

Start line: 1, End line: 105

```python
#
# xarray documentation build configuration file, created by
# sphinx-quickstart on Thu Feb  6 18:57:54 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


import datetime
import os
import pathlib
import subprocess
import sys
from contextlib import suppress

import sphinx_autosummary_accessors
from jinja2.defaults import DEFAULT_FILTERS

import xarray

allowed_failures = set()

print("python exec:", sys.executable)
print("sys.path:", sys.path)

if "conda" in sys.executable:
    print("conda environment:")
    subprocess.run(["conda", "list"])
else:
    print("pip environment:")
    subprocess.run(["pip", "list"])

print(f"xarray: {xarray.__version__}, {xarray.__file__}")

with suppress(ImportError):
    import matplotlib

    matplotlib.use("Agg")

try:
    import rasterio  # noqa: F401
except ImportError:
    allowed_failures.update(
        ["gallery/plot_rasterio_rgb.py", "gallery/plot_rasterio.py"]
    )

try:
    import cartopy  # noqa: F401
except ImportError:
    allowed_failures.update(
        [
            "gallery/plot_cartopy_facetgrid.py",
            "gallery/plot_rasterio_rgb.py",
            "gallery/plot_rasterio.py",
        ]
    )

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx_autosummary_accessors",
    "scanpydoc.rtd_github_links",
]

extlinks = {
    "issue": ("https://github.com/pydata/xarray/issues/%s", "GH"),
    "pull": ("https://github.com/pydata/xarray/pull/%s", "PR"),
}

nbsphinx_timeout = 600
nbsphinx_execute = "always"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

You can run this notebook in a `live session <https://mybinder.org/v2/gh/pydata/xarray/doc/examples/master?urlpath=lab/tree/doc/{{ docname }}>`_ |Binder| or view it `on Github <https://github.com/pydata/xarray/blob/master/doc/{{ docname }}>`_.

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pydata/xarray/master?urlpath=lab/tree/doc/{{ docname }}
"""

autosummary_generate = True

# for scanpydoc's jinja filter
project_dir = pathlib.Path(__file__).parent.parent
```
### 31 - xarray/core/formatting.py:

Start line: 496, End line: 525

```python
def dataset_repr(ds):
    summary = ["<xarray.{}>".format(type(ds).__name__)]

    col_width = _calculate_col_width(_get_col_items(ds.variables))

    dims_start = pretty_print("Dimensions:", col_width)
    summary.append("{}({})".format(dims_start, dim_summary(ds)))

    if ds.coords:
        summary.append(coords_repr(ds.coords, col_width=col_width))

    unindexed_dims_str = unindexed_dims_repr(ds.dims, ds.coords)
    if unindexed_dims_str:
        summary.append(unindexed_dims_str)

    summary.append(data_vars_repr(ds.data_vars, col_width=col_width))

    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return "\n".join(summary)


def diff_dim_summary(a, b):
    if a.dims != b.dims:
        return "Differing dimensions:\n    ({}) != ({})".format(
            dim_summary(a), dim_summary(b)
        )
    else:
        return ""
```
