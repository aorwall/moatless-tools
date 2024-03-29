# pydata__xarray-6461

| **pydata/xarray** | `851dadeb0338403e5021c3fbe80cbc9127ee672d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 149 |
| **Any found context length** | 149 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -1825,11 +1825,10 @@ def where(cond, x, y, keep_attrs=None):
     """
     if keep_attrs is None:
         keep_attrs = _get_keep_attrs(default=False)
-
     if keep_attrs is True:
         # keep the attributes of x, the second parameter, by default to
         # be consistent with the `where` method of `DataArray` and `Dataset`
-        keep_attrs = lambda attrs, context: attrs[1]
+        keep_attrs = lambda attrs, context: getattr(x, "attrs", {})
 
     # alignment for three arguments is complicated, so don't support it yet
     return apply_ufunc(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/computation.py | 1828 | 1832 | 1 | 1 | 149


## Problem Statement

```
xr.where with scalar as second argument fails with keep_attrs=True
### What happened?

\`\`\` python
import xarray as xr

xr.where(xr.DataArray([1, 2, 3]) > 0, 1, 0)
\`\`\`

fails with

\`\`\`
   1809 if keep_attrs is True:
   1810     # keep the attributes of x, the second parameter, by default to
   1811     # be consistent with the `where` method of `DataArray` and `Dataset`
-> 1812     keep_attrs = lambda attrs, context: attrs[1]
   1814 # alignment for three arguments is complicated, so don't support it yet
   1815 return apply_ufunc(
   1816     duck_array_ops.where,
   1817     cond,
   (...)
   1823     keep_attrs=keep_attrs,
   1824 )

IndexError: list index out of range
\`\`\`

The workaround is to pass `keep_attrs=False`

### What did you expect to happen?

_No response_

### Minimal Complete Verifiable Example

_No response_

### Relevant log output

_No response_

### Anything else we need to know?

_No response_

### Environment

xarray 2022.3.0

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 xarray/core/computation.py** | 1826 | 1844| 149 | 149 | 17109 | 
| 2 | **1 xarray/core/computation.py** | 1743 | 1825| 904 | 1053 | 17109 | 
| 3 | 2 xarray/core/common.py | 1259 | 1291| 279 | 1332 | 32963 | 
| 4 | 2 xarray/core/common.py | 1184 | 1258| 832 | 2164 | 32963 | 
| 5 | 3 xarray/core/dataarray.py | 3113 | 3126| 161 | 2325 | 75330 | 
| 6 | 4 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 3063 | 76831 | 
| 7 | 4 xarray/core/common.py | 1121 | 1182| 636 | 3699 | 76831 | 
| 8 | 5 xarray/core/duck_array_ops.py | 351 | 375| 292 | 3991 | 82309 | 
| 9 | 6 xarray/__init__.py | 1 | 112| 701 | 4692 | 83010 | 
| 10 | 7 xarray/core/merge.py | 762 | 976| 2805 | 7497 | 91919 | 
| 11 | 7 xarray/core/common.py | 1351 | 1385| 240 | 7737 | 91919 | 
| 12 | 8 xarray/ufuncs.py | 1 | 36| 239 | 7976 | 93132 | 
| 13 | 9 asv_bench/benchmarks/dataarray_missing.py | 1 | 16| 114 | 8090 | 93633 | 
| 14 | 9 xarray/core/duck_array_ops.py | 1 | 95| 785 | 8875 | 93633 | 
| 15 | 9 xarray/core/duck_array_ops.py | 261 | 301| 288 | 9163 | 93633 | 
| 16 | 9 xarray/core/dataarray.py | 3152 | 3173| 254 | 9417 | 93633 | 
| 17 | 9 xarray/core/common.py | 1315 | 1349| 239 | 9656 | 93633 | 
| 18 | 10 xarray/core/combine.py | 671 | 919| 3129 | 12785 | 102605 | 
| 19 | 10 xarray/core/common.py | 537 | 667| 1158 | 13943 | 102605 | 
| 20 | 10 xarray/core/common.py | 1 | 47| 243 | 14186 | 102605 | 
| 21 | 11 xarray/core/ops.py | 157 | 185| 197 | 14383 | 104986 | 
| 22 | 12 xarray/core/dataset.py | 1 | 132| 691 | 15074 | 174226 | 
| 23 | 13 xarray/core/missing.py | 696 | 763| 566 | 15640 | 180417 | 
| 24 | 13 xarray/core/missing.py | 1 | 18| 139 | 15779 | 180417 | 
| 25 | 14 xarray/core/indexing.py | 1 | 36| 189 | 15968 | 192672 | 
| 26 | 15 xarray/core/types.py | 1 | 36| 317 | 16285 | 192989 | 


## Patch

```diff
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -1825,11 +1825,10 @@ def where(cond, x, y, keep_attrs=None):
     """
     if keep_attrs is None:
         keep_attrs = _get_keep_attrs(default=False)
-
     if keep_attrs is True:
         # keep the attributes of x, the second parameter, by default to
         # be consistent with the `where` method of `DataArray` and `Dataset`
-        keep_attrs = lambda attrs, context: attrs[1]
+        keep_attrs = lambda attrs, context: getattr(x, "attrs", {})
 
     # alignment for three arguments is complicated, so don't support it yet
     return apply_ufunc(

```

## Test Patch

```diff
diff --git a/xarray/tests/test_computation.py b/xarray/tests/test_computation.py
--- a/xarray/tests/test_computation.py
+++ b/xarray/tests/test_computation.py
@@ -1928,6 +1928,10 @@ def test_where_attrs() -> None:
     expected = xr.DataArray([1, 0], dims="x", attrs={"attr": "x"})
     assert_identical(expected, actual)
 
+    # ensure keep_attrs can handle scalar values
+    actual = xr.where(cond, 1, 0, keep_attrs=True)
+    assert actual.attrs == {}
+
 
 @pytest.mark.parametrize("use_dask", [True, False])
 @pytest.mark.parametrize("use_datetime", [True, False])

```


## Code snippets

### 1 - xarray/core/computation.py:

Start line: 1826, End line: 1844

```python
def where(cond, x, y, keep_attrs=None):
    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=False)

    if keep_attrs is True:
        # keep the attributes of x, the second parameter, by default to
        # be consistent with the `where` method of `DataArray` and `Dataset`
        keep_attrs = lambda attrs, context: attrs[1]

    # alignment for three arguments is complicated, so don't support it yet
    return apply_ufunc(
        duck_array_ops.where,
        cond,
        x,
        y,
        join="exact",
        dataset_join="exact",
        dask="allowed",
        keep_attrs=keep_attrs,
    )
```
### 2 - xarray/core/computation.py:

Start line: 1743, End line: 1825

```python
def where(cond, x, y, keep_attrs=None):
    """Return elements from `x` or `y` depending on `cond`.

    Performs xarray-like broadcasting across input arguments.

    All dimension coordinates on `x` and `y`  must be aligned with each
    other and with `cond`.

    Parameters
    ----------
    cond : scalar, array, Variable, DataArray or Dataset
        When True, return values from `x`, otherwise returns values from `y`.
    x : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is True
    y : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is False
    keep_attrs : bool or str or callable, optional
        How to treat attrs. If True, keep the attrs of `x`.

    Returns
    -------
    Dataset, DataArray, Variable or array
        In priority order: Dataset, DataArray, Variable or array, whichever
        type appears as an input argument.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     0.1 * np.arange(10),
    ...     dims=["lat"],
    ...     coords={"lat": np.arange(10)},
    ...     name="sst",
    ... )
    >>> x
    <xarray.DataArray 'sst' (lat: 10)>
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Coordinates:
      * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> xr.where(x < 0.5, x, x * 100)
    <xarray.DataArray 'sst' (lat: 10)>
    array([ 0. ,  0.1,  0.2,  0.3,  0.4, 50. , 60. , 70. , 80. , 90. ])
    Coordinates:
      * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> y = xr.DataArray(
    ...     0.1 * np.arange(9).reshape(3, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": np.arange(3), "lon": 10 + np.arange(3)},
    ...     name="sst",
    ... )
    >>> y
    <xarray.DataArray 'sst' (lat: 3, lon: 3)>
    array([[0. , 0.1, 0.2],
           [0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8]])
    Coordinates:
      * lat      (lat) int64 0 1 2
      * lon      (lon) int64 10 11 12

    >>> xr.where(y.lat < 1, y, -1)
    <xarray.DataArray (lat: 3, lon: 3)>
    array([[ 0. ,  0.1,  0.2],
           [-1. , -1. , -1. ],
           [-1. , -1. , -1. ]])
    Coordinates:
      * lat      (lat) int64 0 1 2
      * lon      (lon) int64 10 11 12

    >>> cond = xr.DataArray([True, False], dims=["x"])
    >>> x = xr.DataArray([1, 2], dims=["y"])
    >>> xr.where(cond, x, 0)
    <xarray.DataArray (x: 2, y: 2)>
    array([[1, 2],
           [0, 0]])
    Dimensions without coordinates: x, y

    See Also
    --------
    numpy.where : corresponding numpy function
    Dataset.where, DataArray.where :
        equivalent methods
    """
    # ... other code
```
### 3 - xarray/core/common.py:

Start line: 1259, End line: 1291

```python
class DataWithCoords(AttrAccessMixin):

    def where(self, cond, other=dtypes.NA, drop: bool = False):
        from .alignment import align
        from .dataarray import DataArray
        from .dataset import Dataset

        if callable(cond):
            cond = cond(self)

        if drop:
            if other is not dtypes.NA:
                raise ValueError("cannot set `other` if drop=True")

            if not isinstance(cond, (Dataset, DataArray)):
                raise TypeError(
                    f"cond argument is {cond!r} but must be a {Dataset!r} or {DataArray!r}"
                )

            # align so we can use integer indexing
            self, cond = align(self, cond)

            # get cond with the minimal size needed for the Dataset
            if isinstance(cond, Dataset):
                clipcond = cond.to_array().any("variable")
            else:
                clipcond = cond

            # clip the data corresponding to coordinate dims that are not used
            nonzeros = zip(clipcond.dims, np.nonzero(clipcond.values))
            indexers = {k: np.unique(v) for k, v in nonzeros}

            self = self.isel(**indexers)
            cond = cond.isel(**indexers)

        return ops.where_method(self, cond, other)
```
### 4 - xarray/core/common.py:

Start line: 1184, End line: 1258

```python
class DataWithCoords(AttrAccessMixin):

    def where(self, cond, other=dtypes.NA, drop: bool = False):
        """Filter elements from this object according to a condition.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic.

        Parameters
        ----------
        cond : DataArray, Dataset, or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, it must expect this object as its only parameter.
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, these locations filled with NA.
        drop : bool, optional
            If True, coordinate labels that only correspond to False values of
            the condition are dropped from the result. Mutually exclusive with
            ``other``.

        Returns
        -------
        DataArray or Dataset
            Same xarray type as caller, with dtype float64.

        Examples
        --------
        >>> a = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
        >>> a
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 4)
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0.,  1.,  2.,  3., nan],
               [ 5.,  6.,  7., nan, nan],
               [10., 11., nan, nan, nan],
               [15., nan, nan, nan, nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 5, -1)
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8, -1],
               [10, 11, 12, -1, -1],
               [15, 16, -1, -1, -1],
               [20, -1, -1, -1, -1]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 4, drop=True)
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 0.,  1.,  2.,  3.],
               [ 5.,  6.,  7., nan],
               [10., 11., nan, nan],
               [15., nan, nan, nan]])
        Dimensions without coordinates: x, y

        >>> a.where(lambda x: x.x + x.y < 4, drop=True)
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 0.,  1.,  2.,  3.],
               [ 5.,  6.,  7., nan],
               [10., 11., nan, nan],
               [15., nan, nan, nan]])
        Dimensions without coordinates: x, y

        See Also
        --------
        numpy.where : corresponding numpy function
        where : equivalent function
        """
        # ... other code
```
### 5 - xarray/core/dataarray.py:

Start line: 3113, End line: 3126

```python
class DataArray(
    AbstractArray, DataWithCoords, DataArrayArithmetic, DataArrayReductions
):

    def _unary_op(self, f: Callable, *args, **kwargs):
        keep_attrs = kwargs.pop("keep_attrs", None)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            warnings.filterwarnings(
                "ignore", r"Mean of empty slice", category=RuntimeWarning
            )
            with np.errstate(all="ignore"):
                da = self.__array_wrap__(f(self.variable.data, *args, **kwargs))
            if keep_attrs:
                da.attrs = self.attrs
            return da
```
### 6 - asv_bench/benchmarks/indexing.py:

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
### 7 - xarray/core/common.py:

Start line: 1121, End line: 1182

```python
class DataWithCoords(AttrAccessMixin):

    def resample(
        self,
        indexer: Mapping[Any, str] = None,
        skipna=None,
        closed: str = None,
        label: str = None,
        base: int = 0,
        keep_attrs: bool = None,
        loffset=None,
        restore_coord_dims: bool = None,
        **indexer_kwargs: str,
    ):
        # TODO support non-string indexer after removing the old API.

        from ..coding.cftimeindex import CFTimeIndex
        from .dataarray import DataArray
        from .resample import RESAMPLE_DIM

        if keep_attrs is not None:
            warnings.warn(
                "Passing ``keep_attrs`` to ``resample`` has no effect and will raise an"
                " error in xarray 0.20. Pass ``keep_attrs`` directly to the applied"
                " function, e.g. ``resample(...).mean(keep_attrs=True)``."
            )

        # note: the second argument (now 'skipna') use to be 'dim'
        if (
            (skipna is not None and not isinstance(skipna, bool))
            or ("how" in indexer_kwargs and "how" not in self.dims)
            or ("dim" in indexer_kwargs and "dim" not in self.dims)
        ):
            raise TypeError(
                "resample() no longer supports the `how` or "
                "`dim` arguments. Instead call methods on resample "
                "objects, e.g., data.resample(time='1D').mean()"
            )

        indexer = either_dict_or_kwargs(indexer, indexer_kwargs, "resample")
        if len(indexer) != 1:
            raise ValueError("Resampling only supported along single dimensions.")
        dim, freq = next(iter(indexer.items()))

        dim_name = dim
        dim_coord = self[dim]

        # TODO: remove once pandas=1.1 is the minimum required version
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r"'(base|loffset)' in .resample\(\) and in Grouper\(\) is deprecated.",
                category=FutureWarning,
            )

            if isinstance(self._indexes[dim_name].to_pandas_index(), CFTimeIndex):
                from .resample_cftime import CFTimeGrouper

                grouper = CFTimeGrouper(freq, closed, label, base, loffset)
            else:
                grouper = pd.Grouper(
                    freq=freq, closed=closed, label=label, base=base, loffset=loffset
                )
        group = DataArray(
            dim_coord, coords=dim_coord.coords, dims=dim_coord.dims, name=RESAMPLE_DIM
        )
        resampler = self._resample_cls(
            self,
            group=group,
            dim=dim_name,
            grouper=grouper,
            resample_dim=RESAMPLE_DIM,
            restore_coord_dims=restore_coord_dims,
        )

        return resampler
```
### 8 - xarray/core/duck_array_ops.py:

Start line: 351, End line: 375

```python
# Attributes `numeric_only`, `available_min_count` is used for docs.
# See ops.inject_reduce_methods
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True, invariant_0d=True)
min = _create_nan_agg_method("min", coerce_strings=True, invariant_0d=True)
sum = _create_nan_agg_method("sum", invariant_0d=True)
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", invariant_0d=True)
median.numeric_only = True
prod = _create_nan_agg_method("prod", invariant_0d=True)
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod", invariant_0d=True)
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum", invariant_0d=True)
cumsum_1d.numeric_only = True


_mean = _create_nan_agg_method("mean", invariant_0d=True)
```
### 9 - xarray/__init__.py:

Start line: 1, End line: 112

```python
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
from .coding.cftime_offsets import cftime_range, date_range, date_range_like
from .coding.cftimeindex import CFTimeIndex
from .coding.frequencies import infer_freq
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast
from .core.combine import combine_by_coords, combine_nested
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import (
    apply_ufunc,
    corr,
    cov,
    cross,
    dot,
    polyval,
    unify_chunks,
    where,
)
from .core.concat import concat
from .core.dataarray import DataArray
from .core.dataset import Dataset
from .core.extensions import register_dataarray_accessor, register_dataset_accessor
from .core.merge import Context, MergeError, merge
from .core.options import get_options, set_options
from .core.parallel import map_blocks
from .core.variable import Coordinate, IndexVariable, Variable, as_variable
from .util.print_versions import show_versions

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("xarray")
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
    "date_range",
    "date_range_like",
    "decode_cf",
    "dot",
    "cov",
    "corr",
    "cross",
    "full_like",
    "get_options",
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
    "unify_chunks",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
    "Context",
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
### 10 - xarray/core/merge.py:

Start line: 762, End line: 976

```python
def merge(
    objects: Iterable[DataArray | CoercibleMapping],
    compat: str = "no_conflicts",
    join: str = "outer",
    fill_value: object = dtypes.NA,
    combine_attrs: str = "override",
) -> Dataset:
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
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", \
                    "override"} or callable, default: "override"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
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
    ValueError: cannot align objects with join='exact' where ...

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    combine_nested
    combine_by_coords
    """
    # ... other code
```
