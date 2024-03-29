# pydata__xarray-3635

| **pydata/xarray** | `f2b2f9f62ea0f1020262a7ff563bfe74258ffaa1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 23 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/variable.py b/xarray/core/variable.py
--- a/xarray/core/variable.py
+++ b/xarray/core/variable.py
@@ -1731,6 +1731,10 @@ def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
         scalar = utils.is_scalar(q)
         q = np.atleast_1d(np.asarray(q, dtype=np.float64))
 
+        # TODO: remove once numpy >= 1.15.0 is the minimum requirement
+        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
+            raise ValueError("Quantiles must be in the range [0, 1]")
+
         if dim is None:
             dim = self.dims
 
@@ -1739,6 +1743,8 @@ def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
 
         def _wrapper(npa, **kwargs):
             # move quantile axis to end. required for apply_ufunc
+
+            # TODO: use np.nanquantile once numpy >= 1.15.0 is the minimum requirement
             return np.moveaxis(np.nanpercentile(npa, **kwargs), 0, -1)
 
         axis = np.arange(-1, -1 * len(dim) - 1, -1)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/variable.py | 1734 | 1734 | - | 23 | -
| xarray/core/variable.py | 1742 | 1742 | - | 23 | -


## Problem Statement

```
"ValueError: Percentiles must be in the range [0, 100]"
#### MCVE Code Sample
<!-- In order for the maintainers to efficiently understand and prioritize issues, we ask you post a "Minimal, Complete and Verifiable Example" (MCVE): http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports -->

\`\`\`python
import xarray as xr
da = xr.DataArray([0, 1, 2])
da.quantile(q=50)

>>> ValueError: Percentiles must be in the range [0, 100]
\`\`\`



#### Expected Output
\`\`\`python
ValueError: Quantiles must be in the range [0, 1]
\`\`\`

#### Problem Description

By wrapping `np.nanpercentile` (xref: #3559) we also get the numpy error. However, the error message is wrong as xarray needs it to be in 0..1.

BTW: thanks for #3559, makes my life easier!

#### Output of ``xr.show_versions()``

---
Edit: uses `nanpercentile` internally.

<details>

INSTALLED VERSIONS
------------------
commit: None
python: 3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
[GCC 7.3.0]
python-bits: 64
OS: Linux
OS-release: 4.12.14-lp151.28.36-default
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_GB.UTF-8
LOCALE: en_US.UTF-8
libhdf5: 1.10.5
libnetcdf: 4.7.1

xarray: 0.14.1+28.gf2b2f9f6 (current master)
pandas: 0.25.2
numpy: 1.17.3
scipy: 1.3.1
netCDF4: 1.5.3
pydap: None
h5netcdf: 0.7.4
h5py: 2.10.0
Nio: None
zarr: None
cftime: 1.0.4.2
nc_time_axis: 1.2.0
PseudoNetCDF: None
rasterio: 1.1.1
cfgrib: None
iris: None
bottleneck: 1.2.1
dask: 2.6.0
distributed: 2.6.0
matplotlib: 3.1.2
cartopy: 0.17.0
seaborn: 0.9.0
numbagg: None
setuptools: 41.4.0
pip: 19.3.1
conda: None
pytest: 5.2.2
IPython: 7.9.0
sphinx: 2.2.1
</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 setup.py | 31 | 111| 745 | 745 | 1002 | 
| 2 | 2 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 1475 | 2414 | 
| 3 | 3 xarray/plot/utils.py | 1 | 74| 399 | 1874 | 8233 | 
| 4 | 4 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 2405 | 8764 | 
| 5 | 5 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 3120 | 12369 | 
| 6 | 6 xarray/core/dtypes.py | 1 | 42| 277 | 3397 | 13426 | 
| 7 | 7 xarray/core/dataarray.py | 1 | 80| 423 | 3820 | 39382 | 
| 8 | 8 xarray/core/nputils.py | 231 | 242| 155 | 3975 | 41382 | 
| 9 | 9 xarray/core/nanops.py | 133 | 189| 554 | 4529 | 43126 | 
| 10 | 10 xarray/core/common.py | 1 | 35| 178 | 4707 | 55570 | 
| 11 | 10 xarray/core/nputils.py | 1 | 41| 300 | 5007 | 55570 | 
| 12 | 11 xarray/core/dataset.py | 1 | 123| 597 | 5604 | 101792 | 
| 13 | 12 asv_bench/benchmarks/rolling.py | 1 | 17| 117 | 5721 | 102421 | 
| 14 | 12 xarray/core/dataarray.py | 2935 | 3019| 955 | 6676 | 102421 | 
| 15 | 13 doc/conf.py | 1 | 113| 794 | 7470 | 105240 | 
| 16 | 14 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 7654 | 105725 | 
| 17 | 15 xarray/core/duck_array_ops.py | 327 | 351| 244 | 7898 | 109904 | 
| 18 | 15 setup.py | 1 | 30| 257 | 8155 | 109904 | 
| 19 | 16 xarray/core/npcompat.py | 31 | 50| 171 | 8326 | 110672 | 
| 20 | 17 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 8739 | 111085 | 
| 21 | 18 doc/examples/_code/weather_data_setup.py | 1 | 23| 199 | 8938 | 111284 | 
| 22 | 18 xarray/core/nanops.py | 74 | 114| 400 | 9338 | 111284 | 
| 23 | 19 xarray/core/utils.py | 1 | 83| 464 | 9802 | 116137 | 
| 24 | 20 xarray/coding/times.py | 1 | 64| 451 | 10253 | 119976 | 
| 25 | 20 xarray/core/duck_array_ops.py | 57 | 86| 277 | 10530 | 119976 | 
| 26 | 21 xarray/backends/api.py | 1 | 58| 309 | 10839 | 130788 | 
| 27 | 21 xarray/plot/utils.py | 548 | 584| 458 | 11297 | 130788 | 
| 28 | 22 xarray/util/print_versions.py | 62 | 77| 128 | 11425 | 131945 | 
| 29 | 22 xarray/core/nanops.py | 22 | 43| 192 | 11617 | 131945 | 
| 30 | **23 xarray/core/variable.py** | 1 | 66| 396 | 12013 | 150358 | 
| 31 | 24 xarray/convert.py | 1 | 60| 321 | 12334 | 152714 | 
| 32 | 25 xarray/core/dask_array_compat.py | 1 | 92| 673 | 13007 | 153388 | 
| 33 | 26 xarray/plot/dataset_plot.py | 1 | 76| 518 | 13525 | 156809 | 
| 34 | 27 xarray/coding/cftime_offsets.py | 581 | 643| 832 | 14357 | 165127 | 
| 35 | 28 xarray/core/alignment.py | 69 | 249| 1865 | 16222 | 171056 | 
| 36 | 29 xarray/core/rolling.py | 124 | 140| 151 | 16373 | 175985 | 
| 37 | 29 xarray/core/npcompat.py | 78 | 97| 119 | 16492 | 175985 | 
| 38 | 30 xarray/core/groupby.py | 561 | 661| 1127 | 17619 | 183822 | 
| 39 | 30 xarray/util/print_versions.py | 80 | 153| 654 | 18273 | 183822 | 
| 40 | 30 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 18537 | 183822 | 
| 41 | 30 xarray/plot/utils.py | 530 | 545| 125 | 18662 | 183822 | 
| 42 | 30 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 19104 | 183822 | 
| 43 | 30 xarray/core/nanops.py | 61 | 71| 153 | 19257 | 183822 | 
| 44 | 31 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 19416 | 183981 | 
| 45 | 32 ci/min_deps_check.py | 175 | 202| 191 | 19607 | 185510 | 
| 46 | 33 xarray/core/rolling_exp.py | 1 | 23| 180 | 19787 | 186404 | 
| 47 | 33 asv_bench/benchmarks/indexing.py | 60 | 74| 151 | 19938 | 186404 | 
| 48 | 33 doc/conf.py | 351 | 362| 165 | 20103 | 186404 | 
| 49 | 33 xarray/core/nanops.py | 46 | 58| 124 | 20227 | 186404 | 
| 50 | 34 xarray/backends/netCDF4_.py | 1 | 26| 198 | 20425 | 190046 | 
| 51 | 34 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 20537 | 190046 | 
| 52 | 34 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 20806 | 190046 | 
| 53 | 35 xarray/backends/pynio_.py | 1 | 13| 127 | 20933 | 190651 | 
| 54 | 35 xarray/core/dataarray.py | 3021 | 3061| 338 | 21271 | 190651 | 
| 55 | 35 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 21620 | 190651 | 
| 56 | 36 xarray/backends/common.py | 1 | 40| 215 | 21835 | 193053 | 
| 57 | 36 asv_bench/benchmarks/dataset_io.py | 222 | 296| 613 | 22448 | 193053 | 
| 58 | 36 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 22713 | 193053 | 
| 59 | 37 xarray/backends/rasterio_.py | 1 | 20| 114 | 22827 | 196219 | 
| 60 | 37 xarray/backends/api.py | 81 | 106| 157 | 22984 | 196219 | 
| 61 | 37 xarray/core/common.py | 1010 | 1059| 513 | 23497 | 196219 | 
| 62 | 37 ci/min_deps_check.py | 73 | 115| 392 | 23889 | 196219 | 
| 63 | 37 xarray/core/duck_array_ops.py | 287 | 324| 320 | 24209 | 196219 | 
| 64 | 37 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 24372 | 196219 | 
| 65 | 37 xarray/core/utils.py | 589 | 623| 235 | 24607 | 196219 | 
| 66 | 37 ci/min_deps_check.py | 118 | 172| 454 | 25061 | 196219 | 
| 67 | 37 xarray/core/rolling.py | 309 | 350| 364 | 25425 | 196219 | 
| 68 | 38 xarray/core/arithmetic.py | 82 | 105| 151 | 25576 | 196982 | 
| 69 | 38 asv_bench/benchmarks/rolling.py | 39 | 70| 337 | 25913 | 196982 | 
| 70 | 38 xarray/core/common.py | 1444 | 1480| 273 | 26186 | 196982 | 
| 71 | 38 ci/min_deps_check.py | 1 | 38| 222 | 26408 | 196982 | 
| 72 | 38 xarray/core/utils.py | 290 | 313| 172 | 26580 | 196982 | 
| 73 | 39 doc/gallery/plot_rasterio.py | 1 | 55| 402 | 26982 | 197384 | 
| 74 | 39 xarray/plot/utils.py | 658 | 686| 384 | 27366 | 197384 | 
| 75 | 40 xarray/core/options.py | 1 | 75| 523 | 27889 | 198615 | 
| 76 | 41 xarray/backends/netcdf3.py | 1 | 33| 220 | 28109 | 199583 | 
| 77 | 41 xarray/core/duck_array_ops.py | 415 | 449| 324 | 28433 | 199583 | 


### Hint

```
Looks straightforward - could you open a PR?
```

## Patch

```diff
diff --git a/xarray/core/variable.py b/xarray/core/variable.py
--- a/xarray/core/variable.py
+++ b/xarray/core/variable.py
@@ -1731,6 +1731,10 @@ def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
         scalar = utils.is_scalar(q)
         q = np.atleast_1d(np.asarray(q, dtype=np.float64))
 
+        # TODO: remove once numpy >= 1.15.0 is the minimum requirement
+        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
+            raise ValueError("Quantiles must be in the range [0, 1]")
+
         if dim is None:
             dim = self.dims
 
@@ -1739,6 +1743,8 @@ def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
 
         def _wrapper(npa, **kwargs):
             # move quantile axis to end. required for apply_ufunc
+
+            # TODO: use np.nanquantile once numpy >= 1.15.0 is the minimum requirement
             return np.moveaxis(np.nanpercentile(npa, **kwargs), 0, -1)
 
         axis = np.arange(-1, -1 * len(dim) - 1, -1)

```

## Test Patch

```diff
diff --git a/xarray/tests/test_variable.py b/xarray/tests/test_variable.py
--- a/xarray/tests/test_variable.py
+++ b/xarray/tests/test_variable.py
@@ -1542,6 +1542,14 @@ def test_quantile_chunked_dim_error(self):
         with raises_regex(ValueError, "dimension 'x'"):
             v.quantile(0.5, dim="x")
 
+    @pytest.mark.parametrize("q", [-0.1, 1.1, [2], [0.25, 2]])
+    def test_quantile_out_of_bounds(self, q):
+        v = Variable(["x", "y"], self.d)
+
+        # escape special characters
+        with raises_regex(ValueError, r"Quantiles must be in the range \[0, 1\]"):
+            v.quantile(q, dim="x")
+
     @requires_dask
     @requires_bottleneck
     def test_rank_dask_raises(self):

```


## Code snippets

### 1 - setup.py:

Start line: 31, End line: 111

```python
LONG_DESCRIPTION = """
**xarray** (formerly **xray**) is an open source project and Python package
that makes working with labelled multi-dimensional arrays simple,
efficient, and fun!

Xarray introduces labels in the form of dimensions, coordinates and
attributes on top of raw NumPy_-like arrays, which allows for a more
intuitive, more concise, and less error-prone developer experience.
The package includes a large and growing library of domain-agnostic functions
for advanced analytics and visualization with these data structures.

Xarray was inspired by and borrows heavily from pandas_, the popular data
analysis package focused on labelled tabular data.
It is particularly tailored to working with netCDF_ files, which were the
source of xarray's data model, and integrates tightly with dask_ for parallel
computing.

.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _dask: https://dask.org
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf

Why xarray?
-----------

Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called
"tensors") are an essential part of computational science.
They are encountered in a wide range of fields, including physics, astronomy,
geoscience, bioinformatics, engineering, finance, and deep learning.
In Python, NumPy_ provides the fundamental data structure and API for
working with raw ND arrays.
However, real-world datasets are usually more than just raw numbers;
they have labels which encode information about how the array values map
to locations in space, time, etc.

Xarray doesn't just keep track of labels on arrays -- it uses them to provide a
powerful and concise interface. For example:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (array broadcasting) based on dimension names, not shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like alignment based on coordinate labels that smoothly
   handles missing values: ``x, y = xr.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

Learn more
----------

- Documentation: http://xarray.pydata.org
- Issue tracker: http://github.com/pydata/xarray/issues
- Source code: http://github.com/pydata/xarray
- SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
    package_data={
        "xarray": ["py.typed", "tests/data/*", "static/css/*", "static/html/*"]
    },
)
```
### 2 - asv_bench/benchmarks/indexing.py:

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
### 3 - xarray/plot/utils.py:

Start line: 1, End line: 74

```python
import itertools
import textwrap
import warnings
from datetime import datetime
from inspect import getfullargspec
from typing import Any, Iterable, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from ..core.options import OPTIONS
from ..core.utils import is_scalar

try:
    import nc_time_axis  # noqa: F401

    nc_time_axis_available = True
except ImportError:
    nc_time_axis_available = False

ROBUST_PERCENTILE = 2.0


def import_seaborn():
    """import seaborn and handle deprecation of apionly module"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import seaborn.apionly as sns

            if (
                w
                and issubclass(w[-1].category, UserWarning)
                and ("seaborn.apionly module" in str(w[-1].message))
            ):
                raise ImportError
        except ImportError:
            import seaborn as sns
        finally:
            warnings.resetwarnings()
    return sns


_registered = False


def register_pandas_datetime_converter_if_needed():
    # based on https://github.com/pandas-dev/pandas/pull/17710
    global _registered
    if not _registered:
        pd.plotting.register_matplotlib_converters()
        _registered = True


def import_matplotlib_pyplot():
    """Import pyplot as register appropriate converters."""
    register_pandas_datetime_converter_if_needed()
    import matplotlib.pyplot as plt

    return plt


def _determine_extend(calc_data, vmin, vmax):
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        extend = "both"
    elif extend_min:
        extend = "min"
    elif extend_max:
        extend = "max"
    else:
        extend = "neither"
    return extend
```
### 4 - asv_bench/benchmarks/dataarray_missing.py:

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
### 5 - asv_bench/benchmarks/dataset_io.py:

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
### 6 - xarray/core/dtypes.py:

Start line: 1, End line: 42

```python
import functools

import numpy as np

from . import utils

# Use as a sentinel value to indicate a dtype appropriate NA value.
NA = utils.ReprObject("<NA>")


@functools.total_ordering
class AlwaysGreaterThan:
    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


@functools.total_ordering
class AlwaysLessThan:
    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


# Equivalence to np.inf (-np.inf) for object-type
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()


# Pairs of types that, if both found, should be promoted to object dtype
# instead of following NumPy's own type-promotion rules. These type promotion
# rules match pandas instead. For reference, see the NumPy type hierarchy:
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
PROMOTE_TO_OBJECT = [
    {np.number, np.character},  # numpy promotes to character
    {np.bool_, np.character},  # numpy promotes to character
    {np.bytes_, np.unicode_},  # numpy promotes to unicode
]
```
### 7 - xarray/core/dataarray.py:

Start line: 1, End line: 80

```python
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
)
from .accessor_dt import DatetimeAccessor
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
from .merge import PANDAS_TYPES, _extract_indexes_from_coords
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
### 8 - xarray/core/nputils.py:

Start line: 231, End line: 242

```python
nanmin = _create_bottleneck_method("nanmin")
nanmax = _create_bottleneck_method("nanmax")
nanmean = _create_bottleneck_method("nanmean")
nanmedian = _create_bottleneck_method("nanmedian")
nanvar = _create_bottleneck_method("nanvar")
nanstd = _create_bottleneck_method("nanstd")
nanprod = _create_bottleneck_method("nanprod")
nancumsum = _create_bottleneck_method("nancumsum")
nancumprod = _create_bottleneck_method("nancumprod")
nanargmin = _create_bottleneck_method("nanargmin")
nanargmax = _create_bottleneck_method("nanargmax")
```
### 9 - xarray/core/nanops.py:

Start line: 133, End line: 189

```python
def nanmean(a, axis=None, dtype=None, out=None):
    if a.dtype.kind == "O":
        return _nanmean_ddof_object(0, a, axis=axis, dtype=dtype)

    if isinstance(a, dask_array_type):
        return dask_array.nanmean(a, axis=axis, dtype=dtype)

    return np.nanmean(a, axis=axis, dtype=dtype)


def nanmedian(a, axis=None, out=None):
    return _dask_or_eager_func("nanmedian", eager_module=nputils)(a, axis=axis)


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
### 10 - xarray/core/common.py:

Start line: 1, End line: 35

```python
import warnings
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, formatting, formatting_html, ops
from .arithmetic import SupportsArithmetic
from .npcompat import DTypeLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .rolling_exp import RollingExp
from .utils import Frozen, either_dict_or_kwargs

# Used as a sentinel value to indicate a all dimensions
ALL_DIMS = ...


C = TypeVar("C")
T = TypeVar("T")
```
### 30 - xarray/core/variable.py:

Start line: 1, End line: 66

```python
import copy
import functools
import itertools
import warnings
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any, Dict, Hashable, Mapping, TypeVar, Union

import numpy as np
import pandas as pd

import xarray as xr  # only for Dataset and DataArray

from . import arithmetic, common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import IS_NEP18_ACTIVE
from .options import _get_keep_attrs
from .pycompat import dask_array_type, integer_types
from .utils import (
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
)

try:
    import dask.array as da
except ImportError:
    pass


NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
) + dask_array_type
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore

VariableType = TypeVar("VariableType", bound="Variable")
"""Type annotation to be used when methods of Variable return self or a copy of self.
When called from an instance of a subclass, e.g. IndexVariable, mypy identifies the
output as an instance of the subclass.

Usage::

   class Variable:
       def f(self: VariableType, ...) -> VariableType:
           ...
"""


class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name.
    """

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?
```
