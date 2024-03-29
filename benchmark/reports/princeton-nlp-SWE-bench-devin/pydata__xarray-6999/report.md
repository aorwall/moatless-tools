# pydata__xarray-6999

| **pydata/xarray** | `1f4be33365573da19a684dd7f2fc97ace5d28710` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2032,11 +2032,11 @@ def rename(
         if utils.is_dict_like(new_name_or_name_dict) or new_name_or_name_dict is None:
             # change dims/coords
             name_dict = either_dict_or_kwargs(new_name_or_name_dict, names, "rename")
-            dataset = self._to_temp_dataset().rename(name_dict)
+            dataset = self._to_temp_dataset()._rename(name_dict)
             return self._from_temp_dataset(dataset)
         if utils.hashable(new_name_or_name_dict) and names:
             # change name + dims/coords
-            dataset = self._to_temp_dataset().rename(names)
+            dataset = self._to_temp_dataset()._rename(names)
             dataarray = self._from_temp_dataset(dataset)
             return dataarray._replace(name=new_name_or_name_dict)
         # only change name
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -3560,6 +3560,48 @@ def _rename_all(
 
         return variables, coord_names, dims, indexes
 
+    def _rename(
+        self: T_Dataset,
+        name_dict: Mapping[Any, Hashable] | None = None,
+        **names: Hashable,
+    ) -> T_Dataset:
+        """Also used internally by DataArray so that the warning (if any)
+        is raised at the right stack level.
+        """
+        name_dict = either_dict_or_kwargs(name_dict, names, "rename")
+        for k in name_dict.keys():
+            if k not in self and k not in self.dims:
+                raise ValueError(
+                    f"cannot rename {k!r} because it is not a "
+                    "variable or dimension in this dataset"
+                )
+
+            create_dim_coord = False
+            new_k = name_dict[k]
+
+            if k in self.dims and new_k in self._coord_names:
+                coord_dims = self._variables[name_dict[k]].dims
+                if coord_dims == (k,):
+                    create_dim_coord = True
+            elif k in self._coord_names and new_k in self.dims:
+                coord_dims = self._variables[k].dims
+                if coord_dims == (new_k,):
+                    create_dim_coord = True
+
+            if create_dim_coord:
+                warnings.warn(
+                    f"rename {k!r} to {name_dict[k]!r} does not create an index "
+                    "anymore. Try using swap_dims instead or use set_index "
+                    "after rename to create an indexed coordinate.",
+                    UserWarning,
+                    stacklevel=3,
+                )
+
+        variables, coord_names, dims, indexes = self._rename_all(
+            name_dict=name_dict, dims_dict=name_dict
+        )
+        return self._replace(variables, coord_names, dims=dims, indexes=indexes)
+
     def rename(
         self: T_Dataset,
         name_dict: Mapping[Any, Hashable] | None = None,
@@ -3588,18 +3630,7 @@ def rename(
         Dataset.rename_dims
         DataArray.rename
         """
-        name_dict = either_dict_or_kwargs(name_dict, names, "rename")
-        for k in name_dict.keys():
-            if k not in self and k not in self.dims:
-                raise ValueError(
-                    f"cannot rename {k!r} because it is not a "
-                    "variable or dimension in this dataset"
-                )
-
-        variables, coord_names, dims, indexes = self._rename_all(
-            name_dict=name_dict, dims_dict=name_dict
-        )
-        return self._replace(variables, coord_names, dims=dims, indexes=indexes)
+        return self._rename(name_dict=name_dict, **names)
 
     def rename_dims(
         self: T_Dataset,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/dataarray.py | 2035 | 2039 | - | 1 | -
| xarray/core/dataset.py | 3563 | 3563 | - | 6 | -
| xarray/core/dataset.py | 3591 | 3602 | - | 6 | -


## Problem Statement

```
[Bug]: rename_vars to dimension coordinate does not create an index
### What happened?

We used `Data{set,Array}.rename{_vars}({coord: dim_coord})` to make a coordinate a dimension coordinate (instead of `set_index`).
This results in the coordinate correctly being displayed as a dimension coordinate (with the *) but it does not create an index, such that further operations like `sel` fail with a strange `KeyError`.

### What did you expect to happen?

I expect one of two things to be true:

1. `rename{_vars}` does not allow setting dimension coordinates (raises Error and tells you to use set_index)
2. `rename{_vars}` checks for this occasion and sets the index correctly

### Minimal Complete Verifiable Example

\`\`\`python
import xarray as xr

data = xr.DataArray([5, 6, 7], coords={"c": ("x", [1, 2, 3])}, dims="x")
# <xarray.DataArray (x: 3)>
# array([5, 6, 7])
# Coordinates:
#     c        (x) int64 1 2 3
# Dimensions without coordinates: x

data_renamed = data.rename({"c": "x"})
# <xarray.DataArray (x: 3)>
# array([5, 6, 7])
# Coordinates:
#   * x        (x) int64 1 2 3

data_renamed.indexes
# Empty
data_renamed.sel(x=2)
# KeyError: 'no index found for coordinate x'

# if we use set_index it works
data_indexed = data.set_index({"x": "c"})
# looks the same as data_renamed!
# <xarray.DataArray (x: 3)>
# array([1, 2, 3])
# Coordinates:
#   * x        (x) int64 1 2 3

data_indexed.indexes
# x: Int64Index([1, 2, 3], dtype='int64', name='x')
\`\`\`


### Relevant log output

_No response_

### Anything else we need to know?

_No response_

### Environment

INSTALLED VERSIONS
------------------
commit: None
python: 3.9.1 (default, Jan 13 2021, 15:21:08) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-44)]
python-bits: 64
OS: Linux
OS-release: 3.10.0-1160.49.1.el7.x86_64
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_US.UTF-8
LOCALE: ('en_US', 'UTF-8')
libhdf5: 1.12.0
libnetcdf: 4.7.4

xarray: 0.20.2
pandas: 1.3.5
numpy: 1.21.5
scipy: 1.7.3
netCDF4: 1.5.8
pydap: None
h5netcdf: None
h5py: None
Nio: None
zarr: None
cftime: 1.5.1.1
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: None
dask: None
distributed: None
matplotlib: 3.5.1
cartopy: None
seaborn: None
numbagg: None
fsspec: None
cupy: None
pint: None
sparse: None
setuptools: 49.2.1
pip: 22.0.2
conda: None
pytest: 6.2.5
IPython: 8.0.0
sphinx: None

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 xarray/core/dataarray.py** | 1 | 96| 642 | 642 | 51221 | 
| 2 | 2 xarray/__init__.py | 1 | 111| 692 | 1334 | 51913 | 
| 3 | 3 xarray/core/types.py | 1 | 84| 777 | 2111 | 52930 | 
| 4 | 4 xarray/core/coordinates.py | 1 | 22| 157 | 2268 | 56408 | 
| 5 | 5 xarray/core/variable.py | 1 | 81| 417 | 2685 | 80904 | 
| 6 | **6 xarray/core/dataset.py** | 1 | 135| 802 | 3487 | 157303 | 
| 7 | 7 xarray/core/indexes.py | 1 | 31| 155 | 3642 | 167918 | 
| 8 | 7 xarray/core/coordinates.py | 25 | 80| 360 | 4002 | 167918 | 
| 9 | 8 xarray/core/missing.py | 1 | 25| 184 | 4186 | 174165 | 
| 10 | 9 xarray/core/groupby.py | 1 | 64| 387 | 4573 | 185131 | 
| 11 | 10 xarray/core/common.py | 1 | 52| 283 | 4856 | 200037 | 


### Hint

```
This has been discussed in #4825.

A third option for `rename{_vars}` would be to rename the coordinate and its index (if any), regardless of whether the old and new names correspond to existing dimensions. We plan to drop the concept of a "dimension coordinate" with an implicit index in favor of indexes explicitly part of Xarray's data model (see https://github.com/pydata/xarray/projects/1), so that it will be possible to set indexes for non-dimension coordinates and/or set dimension coordinates without indexes.

Re your example, in #5692 `data.rename({"c": "x"})` does not implicitly create anymore an indexed coordinate (no `*`):

\`\`\`python
data_renamed
# <xarray.DataArray (x: 3)>
# array([5, 6, 7])
# Coordinates:
#     x        (x) int64 1 2 3
\`\`\`

Instead, it should be possible to directly set an index for the `c` coordinate without the need to rename it, e.g.,

\`\`\`python
# API has still to be defined
data_indexed = data.set_index("c", index_cls=xr.PandasIndex)

data_indexed.sel(c=[1, 2])
# <xarray.DataArray (x: 2)>
# array([5, 6])
# Coordinates:
#   * c       (x) int64 1 2
\`\`\`


> `data.rename({"c": "x"})` does not implicitly create anymore an indexed coordinate

I have code that relied on automatic index creation through rename and some downstream code broke.

I think we need to address this through a warning or error so that users can be alerted that behaviour has changed.
```

## Patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -2032,11 +2032,11 @@ def rename(
         if utils.is_dict_like(new_name_or_name_dict) or new_name_or_name_dict is None:
             # change dims/coords
             name_dict = either_dict_or_kwargs(new_name_or_name_dict, names, "rename")
-            dataset = self._to_temp_dataset().rename(name_dict)
+            dataset = self._to_temp_dataset()._rename(name_dict)
             return self._from_temp_dataset(dataset)
         if utils.hashable(new_name_or_name_dict) and names:
             # change name + dims/coords
-            dataset = self._to_temp_dataset().rename(names)
+            dataset = self._to_temp_dataset()._rename(names)
             dataarray = self._from_temp_dataset(dataset)
             return dataarray._replace(name=new_name_or_name_dict)
         # only change name
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -3560,6 +3560,48 @@ def _rename_all(
 
         return variables, coord_names, dims, indexes
 
+    def _rename(
+        self: T_Dataset,
+        name_dict: Mapping[Any, Hashable] | None = None,
+        **names: Hashable,
+    ) -> T_Dataset:
+        """Also used internally by DataArray so that the warning (if any)
+        is raised at the right stack level.
+        """
+        name_dict = either_dict_or_kwargs(name_dict, names, "rename")
+        for k in name_dict.keys():
+            if k not in self and k not in self.dims:
+                raise ValueError(
+                    f"cannot rename {k!r} because it is not a "
+                    "variable or dimension in this dataset"
+                )
+
+            create_dim_coord = False
+            new_k = name_dict[k]
+
+            if k in self.dims and new_k in self._coord_names:
+                coord_dims = self._variables[name_dict[k]].dims
+                if coord_dims == (k,):
+                    create_dim_coord = True
+            elif k in self._coord_names and new_k in self.dims:
+                coord_dims = self._variables[k].dims
+                if coord_dims == (new_k,):
+                    create_dim_coord = True
+
+            if create_dim_coord:
+                warnings.warn(
+                    f"rename {k!r} to {name_dict[k]!r} does not create an index "
+                    "anymore. Try using swap_dims instead or use set_index "
+                    "after rename to create an indexed coordinate.",
+                    UserWarning,
+                    stacklevel=3,
+                )
+
+        variables, coord_names, dims, indexes = self._rename_all(
+            name_dict=name_dict, dims_dict=name_dict
+        )
+        return self._replace(variables, coord_names, dims=dims, indexes=indexes)
+
     def rename(
         self: T_Dataset,
         name_dict: Mapping[Any, Hashable] | None = None,
@@ -3588,18 +3630,7 @@ def rename(
         Dataset.rename_dims
         DataArray.rename
         """
-        name_dict = either_dict_or_kwargs(name_dict, names, "rename")
-        for k in name_dict.keys():
-            if k not in self and k not in self.dims:
-                raise ValueError(
-                    f"cannot rename {k!r} because it is not a "
-                    "variable or dimension in this dataset"
-                )
-
-        variables, coord_names, dims, indexes = self._rename_all(
-            name_dict=name_dict, dims_dict=name_dict
-        )
-        return self._replace(variables, coord_names, dims=dims, indexes=indexes)
+        return self._rename(name_dict=name_dict, **names)
 
     def rename_dims(
         self: T_Dataset,

```

## Test Patch

```diff
diff --git a/xarray/tests/test_dataarray.py b/xarray/tests/test_dataarray.py
--- a/xarray/tests/test_dataarray.py
+++ b/xarray/tests/test_dataarray.py
@@ -1742,6 +1742,23 @@ def test_rename(self) -> None:
         )
         assert_identical(renamed_all, expected_all)
 
+    def test_rename_dimension_coord_warnings(self) -> None:
+        # create a dimension coordinate by renaming a dimension or coordinate
+        # should raise a warning (no index created)
+        da = DataArray([0, 0], coords={"x": ("y", [0, 1])}, dims="y")
+
+        with pytest.warns(
+            UserWarning, match="rename 'x' to 'y' does not create an index.*"
+        ):
+            da.rename(x="y")
+
+        da = xr.DataArray([0, 0], coords={"y": ("x", [0, 1])}, dims="x")
+
+        with pytest.warns(
+            UserWarning, match="rename 'x' to 'y' does not create an index.*"
+        ):
+            da.rename(x="y")
+
     def test_init_value(self) -> None:
         expected = DataArray(
             np.full((3, 4), 3), dims=["x", "y"], coords=[range(3), range(4)]
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -2892,6 +2892,23 @@ def test_rename_dimension_coord(self) -> None:
         actual_2 = original.rename_dims({"x": "x_new"})
         assert "x" in actual_2.xindexes
 
+    def test_rename_dimension_coord_warnings(self) -> None:
+        # create a dimension coordinate by renaming a dimension or coordinate
+        # should raise a warning (no index created)
+        ds = Dataset(coords={"x": ("y", [0, 1])})
+
+        with pytest.warns(
+            UserWarning, match="rename 'x' to 'y' does not create an index.*"
+        ):
+            ds.rename(x="y")
+
+        ds = Dataset(coords={"y": ("x", [0, 1])})
+
+        with pytest.warns(
+            UserWarning, match="rename 'x' to 'y' does not create an index.*"
+        ):
+            ds.rename(x="y")
+
     def test_rename_multiindex(self) -> None:
         mindex = pd.MultiIndex.from_tuples([([1, 2]), ([3, 4])], names=["a", "b"])
         original = Dataset({}, {"x": mindex})

```


## Code snippets

### 1 - xarray/core/dataarray.py:

Start line: 1, End line: 96

```python
from __future__ import annotations

import datetime
import warnings
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    cast,
    overload,
)

import numpy as np
import pandas as pd

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex
from ..plot.plot import _PlotMethods
from ..plot.utils import _get_units_from_attrs
from . import alignment, computation, dtypes, indexing, ops, utils
from ._reductions import DataArrayReductions
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DataArrayArithmetic
from .common import AbstractArray, DataWithCoords, get_chunksizes
from .computation import unify_chunks
from .coordinates import DataArrayCoordinates, assert_coordinate_consistent
from .dataset import Dataset
from .formatting import format_item
from .indexes import (
    Index,
    Indexes,
    PandasMultiIndex,
    filter_indexes_from_coords,
    isel_indexes,
)
from .indexing import is_fancy_indexer, map_index_queries
from .merge import PANDAS_TYPES, MergeError, _create_indexes_from_coords
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import IndexVariable, Variable, as_compatible_data, as_variable

if TYPE_CHECKING:
    from typing import TypeVar, Union

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None  # type: ignore
    try:
        from cdms2 import Variable as cdms2_Variable
    except ImportError:
        cdms2_Variable = None
    try:
        from iris.cube import Cube as iris_Cube
    except ImportError:
        iris_Cube = None

    from ..backends.api import T_NetcdfEngine, T_NetcdfTypes
    from .groupby import DataArrayGroupBy
    from .resample import DataArrayResample
    from .rolling import DataArrayCoarsen, DataArrayRolling
    from .types import (
        CoarsenBoundaryOptions,
        DatetimeUnitOptions,
        Ellipsis,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        PadModeOptions,
        PadReflectOptions,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_DataArray,
        T_Xarray,
    )
    from .weighted import DataArrayWeighted

    T_XarrayOther = TypeVar("T_XarrayOther", bound=Union["DataArray", Dataset])
```
### 2 - xarray/__init__.py:

Start line: 1, End line: 111

```python
from . import testing, tutorial
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
### 3 - xarray/core/types.py:

Start line: 1, End line: 84

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar, Union

import numpy as np

if TYPE_CHECKING:

    from .common import AbstractArray, DataWithCoords
    from .dataarray import DataArray
    from .dataset import Dataset
    from .groupby import DataArrayGroupBy, GroupBy
    from .indexes import Index
    from .npcompat import ArrayLike
    from .variable import Variable

    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray  # type: ignore

    # TODO: Turn on when https://github.com/python/mypy/issues/11871 is fixed.
    # Can be uncommented if using pyright though.
    # import sys

    # try:
    #     if sys.version_info >= (3, 11):
    #         from typing import Self
    #     else:
    #         from typing_extensions import Self
    # except ImportError:
    #     Self: Any = None
    Self: Any = None

    Ellipsis = ellipsis

else:
    Self: Any = None
    Ellipsis: Any = None


T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")
T_Array = TypeVar("T_Array", bound="AbstractArray")
T_Index = TypeVar("T_Index", bound="Index")

T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", bound=Union["Dataset", "DataArray"])

# Maybe we rename this to T_Data or something less Fortran-y?
T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")

ScalarOrArray = Union["ArrayLike", np.generic, np.ndarray, "DaskArray"]
DsCompatible = Union["Dataset", "DataArray", "Variable", "GroupBy", "ScalarOrArray"]
DaCompatible = Union["DataArray", "Variable", "DataArrayGroupBy", "ScalarOrArray"]
VarCompatible = Union["Variable", "ScalarOrArray"]
GroupByIncompatible = Union["Variable", "GroupBy"]

ErrorOptions = Literal["raise", "ignore"]
ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]

CompatOptions = Literal[
    "identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"
]
ConcatOptions = Literal["all", "minimal", "different"]
CombineAttrsOptions = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    Callable[..., Any],
]
JoinOptions = Literal["outer", "inner", "left", "right", "exact", "override"]

Interp1dOptions = Literal[
    "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
]
InterpolantOptions = Literal["barycentric", "krog", "pchip", "spline", "akima"]
InterpOptions = Union[Interp1dOptions, InterpolantOptions]

DatetimeUnitOptions = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", None
]

QueryEngineOptions = Literal["python", "numexpr", None]
QueryParserOptions = Literal["pandas", "python"]
```
### 4 - xarray/core/coordinates.py:

Start line: 1, End line: 22

```python
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Hashable, Iterator, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from . import formatting
from .indexes import Index, Indexes, PandasMultiIndex, assert_no_index_corrupted
from .merge import merge_coordinates_without_align, merge_coords
from .utils import Frozen, ReprObject
from .variable import Variable, calculate_dimensions

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_THIS_ARRAY = ReprObject("<this-array>")
```
### 5 - xarray/core/variable.py:

Start line: 1, End line: 81

```python
from __future__ import annotations

import copy
import itertools
import math
import numbers
import warnings
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
)

import numpy as np
import pandas as pd
from packaging.version import Version

import xarray as xr  # only for Dataset and DataArray

from . import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .arithmetic import VariableArithmetic
from .common import AbstractArray
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexingAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import (
    DuckArrayModule,
    cupy_array_type,
    integer_types,
    is_duck_dask_array,
    sparse_array_type,
)
from .utils import (
    Frozen,
    NdimSizeLenMixin,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
)
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)

if TYPE_CHECKING:
    from .types import (
        Ellipsis,
        ErrorOptionsWithWarn,
        PadModeOptions,
        PadReflectOptions,
        T_Variable,
    )


class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name."""

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?
```
### 6 - xarray/core/dataset.py:

Start line: 1, End line: 135

```python
from __future__ import annotations

import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generic,
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

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import alignment
from . import dtypes as xrdtypes
from . import duck_array_ops, formatting, formatting_html, ops, utils
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
from .types import T_Dataset
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
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
    from ..backends.api import T_NetcdfEngine, T_NetcdfTypes
    from .coordinates import Coordinates
    from .dataarray import DataArray
    from .groupby import DatasetGroupBy
    from .merge import CoercibleMapping
    from .resample import DatasetResample
    from .rolling import DatasetCoarsen, DatasetRolling
    from .types import (
        CFCalendar,
        CoarsenBoundaryOptions,
        CombineAttrsOptions,
        CompatOptions,
        DatetimeUnitOptions,
        Ellipsis,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        JoinOptions,
        PadModeOptions,
        PadReflectOptions,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_Xarray,
    )
    from .weighted import DatasetWeighted

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None  # type: ignore
    try:
        from dask.dataframe import DataFrame as DaskDataFrame
    except ImportError:
        DaskDataFrame = None  # type: ignore


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
```
### 7 - xarray/core/indexes.py:

Start line: 1, End line: 31

```python
from __future__ import annotations

import collections.abc
import copy
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd

from . import formatting, nputils, utils
from .indexing import IndexSelResult, PandasIndexingAdapter, PandasMultiIndexingAdapter
from .utils import Frozen, get_valid_numpy_dtype, is_dict_like, is_scalar

if TYPE_CHECKING:
    from .types import ErrorOptions, T_Index
    from .variable import Variable

IndexVars = Dict[Any, "Variable"]
```
### 8 - xarray/core/coordinates.py:

Start line: 25, End line: 80

```python
class Coordinates(Mapping[Hashable, "DataArray"]):
    __slots__ = ()

    def __getitem__(self, key: Hashable) -> DataArray:
        raise NotImplementedError()

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.update({key: value})

    @property
    def _names(self) -> set[Hashable]:
        raise NotImplementedError()

    @property
    def dims(self) -> Mapping[Hashable, int] | tuple[Hashable, ...]:
        raise NotImplementedError()

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        raise NotImplementedError()

    @property
    def indexes(self) -> Indexes[pd.Index]:
        return self._data.indexes  # type: ignore[attr-defined]

    @property
    def xindexes(self) -> Indexes[Index]:
        return self._data.xindexes  # type: ignore[attr-defined]

    @property
    def variables(self):
        raise NotImplementedError()

    def _update_coords(self, coords, indexes):
        raise NotImplementedError()

    def _maybe_drop_multiindex_coords(self, coords):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Hashable]:
        # needs to be in the same order as the dataset variables
        for k in self.variables:
            if k in self._names:
                yield k

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._names

    def __repr__(self) -> str:
        return formatting.coords_repr(self)

    def to_dataset(self) -> Dataset:
        raise NotImplementedError()
```
### 9 - xarray/core/missing.py:

Start line: 1, End line: 25

```python
from __future__ import annotations

import datetime as dt
import warnings
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Hashable, Sequence, get_args

import numpy as np
import pandas as pd
from packaging.version import Version

from . import utils
from .common import _contains_datetime_like_objects, ones_like
from .computation import apply_ufunc
from .duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_version, is_duck_dask_array
from .types import Interp1dOptions, InterpOptions
from .utils import OrderedSet, is_scalar
from .variable import Variable, broadcast_variables

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset
```
### 10 - xarray/core/groupby.py:

Start line: 1, End line: 64

```python
from __future__ import annotations

import datetime
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, nputils, ops
from ._reductions import DataArrayGroupByReductions, DatasetGroupByReductions
from .alignment import align
from .arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
from .indexes import create_default_index_implicit, filter_indexes_from_coords
from .npcompat import QUANTILE_METHODS, ArrayLike
from .ops import IncludeCumMethods
from .options import _get_keep_attrs
from .pycompat import integer_types
from .types import T_Xarray
from .utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset
    from .utils import Frozen

    GroupKey = Any


def check_reduce_dims(reduce_dims, dimensions):

    if reduce_dims is not ...:
        if is_scalar(reduce_dims):
            reduce_dims = [reduce_dims]
        if any(dim not in dimensions for dim in reduce_dims):
            raise ValueError(
                f"cannot reduce over dimensions {reduce_dims!r}. expected either '...' "
                f"to reduce over all dimensions or one or more of {dimensions!r}."
            )
```
