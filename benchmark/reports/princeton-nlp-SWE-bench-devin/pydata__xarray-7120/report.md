# pydata__xarray-7120

| **pydata/xarray** | `58ab594aa4315e75281569902e29c8c69834151f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 8 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -5401,6 +5401,13 @@ def transpose(
         numpy.transpose
         DataArray.transpose
         """
+        # Raise error if list is passed as dims
+        if (len(dims) > 0) and (isinstance(dims[0], list)):
+            list_fix = [f"{repr(x)}" if isinstance(x, str) else f"{x}" for x in dims[0]]
+            raise TypeError(
+                f'transpose requires dims to be passed as multiple arguments. Expected `{", ".join(list_fix)}`. Received `{dims[0]}` instead'
+            )
+
         # Use infix_dims to check once for missing dimensions
         if len(dims) != 0:
             _ = list(infix_dims(dims, self.dims, missing_dims))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/dataset.py | 5404 | 5404 | - | 8 | -


## Problem Statement

```
Raise nicer error if passing a list of dimension names to transpose
### What happened?

Hello,

in xarray 0.20.1, I am getting the following error

`ds = xr.Dataset({"foo": (("x", "y", "z"), [[[42]]]), "bar": (("y", "z"), [[24]])})`

`ds.transpose("y", "z", "x")`


\`\`\`
868 """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    869 are not present in dims.
    870 
   (...)
    875 missing_dims : {"raise", "warn", "ignore"}
    876 """
    878 if missing_dims == "raise":
--> 879     supplied_dims_set = {val for val in supplied_dims if val is not ...}
    880     invalid = supplied_dims_set - set(dims)
    881     if invalid:

TypeError: unhashable type: 'list'
\`\`\`

### What did you expect to happen?

The expected result is 
\`\`\`
ds.transpose("y", "z", "x")

<xarray.Dataset>
Dimensions:  (x: 1, y: 1, z: 1)
Dimensions without coordinates: x, y, z
Data variables:
    foo      (y, z, x) int64 42
    bar      (y, z) int64 24
\`\`\`

### Minimal Complete Verifiable Example

_No response_

### Relevant log output

_No response_

### Anything else we need to know?

_No response_

### Environment

<details>

INSTALLED VERSIONS
------------------
commit: None
python: 3.9.12 (main, Apr  5 2022, 06:56:58) 
[GCC 7.5.0]
python-bits: 64
OS: Linux
OS-release: 3.10.0-1160.42.2.el7.x86_64
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_US
LOCALE: ('en_US', 'ISO8859-1')
libhdf5: 1.12.1
libnetcdf: 4.8.1

xarray: 0.20.1
pandas: 1.4.1
numpy: 1.21.5
scipy: 1.8.0
netCDF4: 1.5.7
pydap: None
h5netcdf: 999
h5py: 3.6.0
Nio: None
zarr: None
cftime: 1.5.1.1
nc_time_axis: 1.4.0
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: 1.3.4
dask: 2022.02.1
distributed: 2022.2.1
matplotlib: 3.5.1
cartopy: 0.18.0
seaborn: 0.11.2
numbagg: None
fsspec: 2022.02.0
cupy: None
pint: 0.18
sparse: 0.13.0
setuptools: 61.2.0
pip: 21.2.4
conda: None
pytest: None
IPython: 8.2.0
sphinx: None

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/types.py | 1 | 93| 790 | 790 | 1045 | 
| 2 | 2 xarray/core/variable.py | 1 | 81| 417 | 1207 | 25874 | 
| 3 | 3 xarray/__init__.py | 1 | 111| 692 | 1899 | 26566 | 
| 4 | 4 xarray/core/utils.py | 896 | 935| 274 | 2173 | 33285 | 
| 5 | 5 xarray/core/common.py | 1 | 52| 283 | 2456 | 48191 | 
| 6 | 6 xarray/core/dataarray.py | 1 | 96| 642 | 3098 | 106135 | 
| 7 | 6 xarray/core/dataarray.py | 1934 | 2572| 5932 | 9030 | 106135 | 
| 8 | 6 xarray/core/dataarray.py | 226 | 999| 6161 | 15191 | 106135 | 
| 9 | 7 xarray/core/missing.py | 1 | 25| 184 | 15375 | 112382 | 
| 10 | **8 xarray/core/dataset.py** | 1 | 135| 802 | 16177 | 190709 | 


### Hint

```
I can't reproduce on our dev branch. Can you try upgrading xarray please?

EDIT: can't reproduce on 2022.03.0 either.
Thanks. I upgraded to 2022.03.0 

I am still getting the error

\`\`\`
Python 3.9.12 (main, Apr  5 2022, 06:56:58) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import xarray as xr
>>> xr.__version__
'2022.3.0'
>>> ds = xr.Dataset({"foo": (("x", "y", "z"), [[[42]]]), "bar": (("y", "z"), [[24]])})
>>> ds.transpose(['y','z','y'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/nbhome/f1p/miniconda3/envs/f1p_gfdl/lib/python3.9/site-packages/xarray/core/dataset.py", line 4650, in transpose
    _ = list(infix_dims(dims, self.dims, missing_dims))
  File "/nbhome/f1p/miniconda3/envs/f1p_gfdl/lib/python3.9/site-packages/xarray/core/utils.py", line 786, in infix_dims
    existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
  File "/nbhome/f1p/miniconda3/envs/f1p_gfdl/lib/python3.9/site-packages/xarray/core/utils.py", line 874, in drop_missing_dims
    supplied_dims_set = {val for val in supplied_dims if val is not ...}
  File "/nbhome/f1p/miniconda3/envs/f1p_gfdl/lib/python3.9/site-packages/xarray/core/utils.py", line 874, in <setcomp>
    supplied_dims_set = {val for val in supplied_dims if val is not ...}
TypeError: unhashable type: 'list'
\`\`\`
\`\`\`
ds.transpose(['y','z','y'])
\`\`\`

Ah... Reemove the list here and try `ds.transpose("y", "z", x")` (no list) which is what you have in the first post. 
Oh... I am so sorry about this. This works as expected now. 
It's weird that using list seemed to have worked at some point. Thanks a lot for your help
I think we should raise a nicer error message. Transpose is an outlier in  our API. In nearly every other function, you are expected to pass a list of dimension names.
```

## Patch

```diff
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -5401,6 +5401,13 @@ def transpose(
         numpy.transpose
         DataArray.transpose
         """
+        # Raise error if list is passed as dims
+        if (len(dims) > 0) and (isinstance(dims[0], list)):
+            list_fix = [f"{repr(x)}" if isinstance(x, str) else f"{x}" for x in dims[0]]
+            raise TypeError(
+                f'transpose requires dims to be passed as multiple arguments. Expected `{", ".join(list_fix)}`. Received `{dims[0]}` instead'
+            )
+
         # Use infix_dims to check once for missing dimensions
         if len(dims) != 0:
             _ = list(infix_dims(dims, self.dims, missing_dims))

```

## Test Patch

```diff
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 
 import pickle
+import re
 import sys
 import warnings
 from copy import copy, deepcopy
@@ -6806,3 +6807,17 @@ def test_string_keys_typing() -> None:
     ds = xr.Dataset(dict(x=da))
     mapping = {"y": da}
     ds.assign(variables=mapping)
+
+
+def test_transpose_error() -> None:
+    # Transpose dataset with list as argument
+    # Should raise error
+    ds = xr.Dataset({"foo": (("x", "y"), [[21]]), "bar": (("x", "y"), [[12]])})
+
+    with pytest.raises(
+        TypeError,
+        match=re.escape(
+            "transpose requires dims to be passed as multiple arguments. Expected `'y', 'x'`. Received `['y', 'x']` instead"
+        ),
+    ):
+        ds.transpose(["y", "x"])  # type: ignore

```


## Code snippets

### 1 - xarray/core/types.py:

Start line: 1, End line: 93

```python
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    TypeVar,
    Union,
)

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

else:
    Self: Any = None


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

Dims = Union[str, Iterable[Hashable], None]

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
### 2 - xarray/core/variable.py:

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
        Dims,
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
### 3 - xarray/__init__.py:

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
### 4 - xarray/core/utils.py:

Start line: 896, End line: 935

```python
def drop_missing_dims(
    supplied_dims: Collection, dims: Collection, missing_dims: ErrorOptionsWithWarn
) -> Collection:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        supplied_dims_set = {val for val in supplied_dims if val is not ...}
        invalid = supplied_dims_set - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return supplied_dims

    elif missing_dims == "warn":

        invalid = set(supplied_dims) - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return [val for val in supplied_dims if val in dims or val is ...]

    elif missing_dims == "ignore":
        return [val for val in supplied_dims if val in dims or val is ...]

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
```
### 5 - xarray/core/common.py:

Start line: 1, End line: 52

```python
from __future__ import annotations

import warnings
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, formatting, formatting_html, ops
from .npcompat import DTypeLike, DTypeLikeSave
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import Frozen, either_dict_or_kwargs, is_scalar

try:
    import cftime
except ImportError:
    cftime = None

# Used as a sentinel value to indicate a all dimensions
ALL_DIMS = ...


if TYPE_CHECKING:
    import datetime

    from .dataarray import DataArray
    from .dataset import Dataset
    from .indexes import Index
    from .resample import Resample
    from .rolling_exp import RollingExp
    from .types import ScalarOrArray, SideOptions, T_DataWithCoords
    from .variable import Variable


T_Resample = TypeVar("T_Resample", bound="Resample")
C = TypeVar("C")
T = TypeVar("T")
```
### 6 - xarray/core/dataarray.py:

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
        Dims,
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
### 7 - xarray/core/dataarray.py:

Start line: 1934, End line: 2572

```python
class DataArray(
    AbstractArray, DataWithCoords, DataArrayArithmetic, DataArrayReductions
):

    def interp(
        self: T_DataArray,
        coords: Mapping[Any, Any] | None = None,
        method: InterpOptions = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] | None = None,
        **coords_kwargs: Any,
    ) -> T_DataArray:
        """Interpolate a DataArray onto new coordinates

        Performs univariate or multivariate interpolation of a DataArray onto
        new coordinates using scipy's interpolation routines. If interpolating
        along an existing dimension, :py:class:`scipy.interpolate.interp1d` is
        called. When interpolating along multiple existing dimensions, an
        attempt is made to decompose the interpolation into multiple
        1-dimensional interpolations. If this is possible,
        :py:class:`scipy.interpolate.interp1d` is called. Otherwise,
        :py:func:`scipy.interpolate.interpn` is called.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.
            If DataArrays are passed as new coordinates, their dimensions are
            used for the broadcasting. Missing values are skipped.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear"
            The method used to interpolate. The method should be supported by
            the scipy interpolator:

            - ``interp1d``: {"linear", "nearest", "zero", "slinear",
              "quadratic", "cubic", "polynomial"}
            - ``interpn``: {"linear", "nearest"}

            If ``"polynomial"`` is passed, the ``order`` keyword argument must
            also be provided.
        assume_sorted : bool, default: False
            If False, values of x can be in any order and they are sorted
            first. If True, x has to be an array of monotonically increasing
            values.
        kwargs : dict-like or None, default: None
            Additional keyword arguments passed to scipy's interpolator. Valid
            options and their behavior depend whether ``interp1d`` or
            ``interpn`` is used.
        **coords_kwargs : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated : DataArray
            New dataarray on the new coordinates.

        Notes
        -----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn

        Examples
        --------
        >>> da = xr.DataArray(
        ...     data=[[1, 4, 2, 9], [2, 7, 6, np.nan], [6, np.nan, 5, 8]],
        ...     dims=("x", "y"),
        ...     coords={"x": [0, 1, 2], "y": [10, 12, 14, 16]},
        ... )
        >>> da
        <xarray.DataArray (x: 3, y: 4)>
        array([[ 1.,  4.,  2.,  9.],
               [ 2.,  7.,  6., nan],
               [ 6., nan,  5.,  8.]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 10 12 14 16

        1D linear interpolation (the default):

        >>> da.interp(x=[0, 0.75, 1.25, 1.75])
        <xarray.DataArray (x: 4, y: 4)>
        array([[1.  , 4.  , 2.  ,  nan],
               [1.75, 6.25, 5.  ,  nan],
               [3.  ,  nan, 5.75,  nan],
               [5.  ,  nan, 5.25,  nan]])
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 0.0 0.75 1.25 1.75

        1D nearest interpolation:

        >>> da.interp(x=[0, 0.75, 1.25, 1.75], method="nearest")
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  4.,  2.,  9.],
               [ 2.,  7.,  6., nan],
               [ 2.,  7.,  6., nan],
               [ 6., nan,  5.,  8.]])
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 0.0 0.75 1.25 1.75

        1D linear extrapolation:

        >>> da.interp(
        ...     x=[1, 1.5, 2.5, 3.5],
        ...     method="linear",
        ...     kwargs={"fill_value": "extrapolate"},
        ... )
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 2. ,  7. ,  6. ,  nan],
               [ 4. ,  nan,  5.5,  nan],
               [ 8. ,  nan,  4.5,  nan],
               [12. ,  nan,  3.5,  nan]])
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 1.0 1.5 2.5 3.5

        2D linear interpolation:

        >>> da.interp(x=[0, 0.75, 1.25, 1.75], y=[11, 13, 15], method="linear")
        <xarray.DataArray (x: 4, y: 3)>
        array([[2.5  , 3.   ,   nan],
               [4.   , 5.625,   nan],
               [  nan,   nan,   nan],
               [  nan,   nan,   nan]])
        Coordinates:
          * x        (x) float64 0.0 0.75 1.25 1.75
          * y        (y) int64 11 13 15
        """
        if self.dtype.kind not in "uifc":
            raise TypeError(
                "interp only works for a numeric type array. "
                "Given {}.".format(self.dtype)
            )
        ds = self._to_temp_dataset().interp(
            coords,
            method=method,
            kwargs=kwargs,
            assume_sorted=assume_sorted,
            **coords_kwargs,
        )
        return self._from_temp_dataset(ds)

    def interp_like(
        self: T_DataArray,
        other: DataArray | Dataset,
        method: InterpOptions = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] | None = None,
    ) -> T_DataArray:
        """Interpolate this object onto the coordinates of another object,
        filling out of range values with NaN.

        If interpolating along a single existing dimension,
        :py:class:`scipy.interpolate.interp1d` is called. When interpolating
        along multiple existing dimensions, an attempt is made to decompose the
        interpolation into multiple 1-dimensional interpolations. If this is
        possible, :py:class:`scipy.interpolate.interp1d` is called. Otherwise,
        :py:func:`scipy.interpolate.interpn` is called.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset. Missing values are skipped.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear"
            The method used to interpolate. The method should be supported by
            the scipy interpolator:

            - {"linear", "nearest", "zero", "slinear", "quadratic", "cubic",
              "polynomial"} when ``interp1d`` is called.
            - {"linear", "nearest"} when ``interpn`` is called.

            If ``"polynomial"`` is passed, the ``order`` keyword argument must
            also be provided.
        assume_sorted : bool, default: False
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword passed to scipy's interpolator.

        Returns
        -------
        interpolated : DataArray
            Another dataarray by interpolating this dataarray's data along the
            coordinates of the other object.

        Notes
        -----
        scipy is required.
        If the dataarray has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        DataArray.interp
        DataArray.reindex_like
        """
        if self.dtype.kind not in "uifc":
            raise TypeError(
                "interp only works for a numeric type array. "
                "Given {}.".format(self.dtype)
            )
        ds = self._to_temp_dataset().interp_like(
            other, method=method, kwargs=kwargs, assume_sorted=assume_sorted
        )
        return self._from_temp_dataset(ds)

    # change type of self and return to T_DataArray once
    # https://github.com/python/mypy/issues/12846 is resolved
    def rename(
        self,
        new_name_or_name_dict: Hashable | Mapping[Any, Hashable] | None = None,
        **names: Hashable,
    ) -> DataArray:
        """Returns a new DataArray with renamed coordinates, dimensions or a new name.

        Parameters
        ----------
        new_name_or_name_dict : str or dict-like, optional
            If the argument is dict-like, it used as a mapping from old
            names to new names for coordinates or dimensions. Otherwise,
            use the argument as the new name for this array.
        **names : Hashable, optional
            The keyword arguments form of a mapping from old names to
            new names for coordinates or dimensions.
            One of new_name_or_name_dict or names must be provided.

        Returns
        -------
        renamed : DataArray
            Renamed array or array with renamed coordinates.

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        if new_name_or_name_dict is None and not names:
            # change name to None?
            return self._replace(name=None)
        if utils.is_dict_like(new_name_or_name_dict) or new_name_or_name_dict is None:
            # change dims/coords
            name_dict = either_dict_or_kwargs(new_name_or_name_dict, names, "rename")
            dataset = self._to_temp_dataset()._rename(name_dict)
            return self._from_temp_dataset(dataset)
        if utils.hashable(new_name_or_name_dict) and names:
            # change name + dims/coords
            dataset = self._to_temp_dataset()._rename(names)
            dataarray = self._from_temp_dataset(dataset)
            return dataarray._replace(name=new_name_or_name_dict)
        # only change name
        return self._replace(name=new_name_or_name_dict)

    def swap_dims(
        self: T_DataArray,
        dims_dict: Mapping[Any, Hashable] | None = None,
        **dims_kwargs,
    ) -> T_DataArray:
        """Returns a new DataArray with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.
        **dims_kwargs : {existing_dim: new_dim, ...}, optional
            The keyword arguments form of ``dims_dict``.
            One of dims_dict or dims_kwargs must be provided.

        Returns
        -------
        swapped : DataArray
            DataArray with swapped dimensions.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=[0, 1],
        ...     dims="x",
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> arr
        <xarray.DataArray (x: 2)>
        array([0, 1])
        Coordinates:
          * x        (x) <U1 'a' 'b'
            y        (x) int64 0 1

        >>> arr.swap_dims({"x": "y"})
        <xarray.DataArray (y: 2)>
        array([0, 1])
        Coordinates:
            x        (y) <U1 'a' 'b'
          * y        (y) int64 0 1

        >>> arr.swap_dims({"x": "z"})
        <xarray.DataArray (z: 2)>
        array([0, 1])
        Coordinates:
            x        (z) <U1 'a' 'b'
            y        (z) int64 0 1
        Dimensions without coordinates: z

        See Also
        --------
        DataArray.rename
        Dataset.swap_dims
        """
        dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, "swap_dims")
        ds = self._to_temp_dataset().swap_dims(dims_dict)
        return self._from_temp_dataset(ds)

    # change type of self and return to T_DataArray once
    # https://github.com/python/mypy/issues/12846 is resolved
    def expand_dims(
        self,
        dim: None | Hashable | Sequence[Hashable] | Mapping[Any, Any] = None,
        axis: None | int | Sequence[int] = None,
        **dim_kwargs: Any,
    ) -> DataArray:
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape. The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : Hashable, sequence of Hashable, dict, or None, optional
            Dimensions to include on the new variable.
            If provided as str or sequence of str, then dimensions are inserted
            with length 1. If provided as a dict, then the keys are the new
            dimensions and the values are either integers (giving the length of
            the new dimensions) or sequence/ndarray (giving the coordinates of
            the new dimensions).
        axis : int, sequence of int, or None, default: None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a sequence of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence or ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None.

        Returns
        -------
        expanded : DataArray
            This object, but with additional dimension(s).

        See Also
        --------
        Dataset.expand_dims

        Examples
        --------
        >>> da = xr.DataArray(np.arange(5), dims=("x"))
        >>> da
        <xarray.DataArray (x: 5)>
        array([0, 1, 2, 3, 4])
        Dimensions without coordinates: x

        Add new dimension of length 2:

        >>> da.expand_dims(dim={"y": 2})
        <xarray.DataArray (y: 2, x: 5)>
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]])
        Dimensions without coordinates: y, x

        >>> da.expand_dims(dim={"y": 2}, axis=1)
        <xarray.DataArray (x: 5, y: 2)>
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [4, 4]])
        Dimensions without coordinates: x, y

        Add a new dimension with coordinates from array:

        >>> da.expand_dims(dim={"y": np.arange(5)}, axis=0)
        <xarray.DataArray (y: 5, x: 5)>
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]])
        Coordinates:
          * y        (y) int64 0 1 2 3 4
        Dimensions without coordinates: x
        """
        if isinstance(dim, int):
            raise TypeError("dim should be Hashable or sequence/mapping of Hashables")
        elif isinstance(dim, Sequence) and not isinstance(dim, str):
            if len(dim) != len(set(dim)):
                raise ValueError("dims should not contain duplicate values.")
            dim = dict.fromkeys(dim, 1)
        elif dim is not None and not isinstance(dim, Mapping):
            dim = {cast(Hashable, dim): 1}

        dim = either_dict_or_kwargs(dim, dim_kwargs, "expand_dims")
        ds = self._to_temp_dataset().expand_dims(dim, axis)
        return self._from_temp_dataset(ds)

    # change type of self and return to T_DataArray once
    # https://github.com/python/mypy/issues/12846 is resolved
    def set_index(
        self,
        indexes: Mapping[Any, Hashable | Sequence[Hashable]] = None,
        append: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> DataArray:
        """Set DataArray (multi-)indexes using one or more existing
        coordinates.

        This legacy method is limited to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See
        :py:meth:`~DataArray.set_xindex` for setting a pandas or a custom
        Xarray-compatible index from one or more arbitrary coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, default: False
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another DataArray, with this data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        >>> arr.set_index(x="a")
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2

        See Also
        --------
        DataArray.reset_index
        DataArray.set_xindex
        """
        ds = self._to_temp_dataset().set_index(indexes, append=append, **indexes_kwargs)
        return self._from_temp_dataset(ds)

    # change type of self and return to T_DataArray once
    # https://github.com/python/mypy/issues/12846 is resolved
    def reset_index(
        self,
        dims_or_levels: Hashable | Sequence[Hashable],
        drop: bool = False,
    ) -> DataArray:
        """Reset the specified index(es) or multi-index level(s).

        This legacy method is specific to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See the more generic
        :py:meth:`~DataArray.drop_indexes` and :py:meth:`~DataArray.set_xindex`
        method to respectively drop and set pandas or custom indexes for
        arbitrary coordinates.

        Parameters
        ----------
        dims_or_levels : Hashable or sequence of Hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, default: False
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.

        See Also
        --------
        DataArray.set_index
        DataArray.set_xindex
        DataArray.drop_indexes
        """
        ds = self._to_temp_dataset().reset_index(dims_or_levels, drop=drop)
        return self._from_temp_dataset(ds)

    def set_xindex(
        self: T_DataArray,
        coord_names: str | Sequence[Hashable],
        index_cls: type[Index] | None = None,
        **options,
    ) -> T_DataArray:
        """Set a new, Xarray-compatible index from one or more existing
        coordinate(s).

        Parameters
        ----------
        coord_names : str or list
            Name(s) of the coordinate(s) used to build the index.
            If several names are given, their order matters.
        index_cls : subclass of :class:`~xarray.indexes.Index`
            The type of index to create. By default, try setting
            a pandas (multi-)index from the supplied coordinates.
        **options
            Options passed to the index constructor.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data and with a new index.

        """
        ds = self._to_temp_dataset().set_xindex(coord_names, index_cls, **options)
        return self._from_temp_dataset(ds)

    def reorder_levels(
        self: T_DataArray,
        dim_order: Mapping[Any, Sequence[int | Hashable]] | None = None,
        **dim_order_kwargs: Sequence[int | Hashable],
    ) -> T_DataArray:
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order dict-like of Hashable to int or Hashable: optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs : optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.
        """
        ds = self._to_temp_dataset().reorder_levels(dim_order, **dim_order_kwargs)
        return self._from_temp_dataset(ds)

    def stack(
        self: T_DataArray,
        dimensions: Mapping[Any, Sequence[Hashable]] | None = None,
        create_index: bool | None = True,
        index_cls: type[Index] = PandasMultiIndex,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> T_DataArray:
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : mapping of Hashable to sequence of Hashable
            Mapping of the form `new_name=(dim1, dim2, ...)`.
            Names of new dimensions, and the existing dimensions that they
            replace. An ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        create_index : bool or None, default: True
            If True, create a multi-index for each of the stacked dimensions.
            If False, don't create any index.
            If None, create a multi-index only if exactly one single (1-d) coordinate
            index is found for every dimension to stack.
        index_cls: class, optional
            Can be used to pass a custom multi-index type. Must be an Xarray index that
            implements `.stack()`. By default, a pandas multi-index wrapper is used.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : DataArray
            DataArray with stacked data.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> stacked = arr.stack(z=("x", "y"))
        >>> stacked.indexes["z"]
        MultiIndex([('a', 0),
                    ('a', 1),
                    ('a', 2),
                    ('b', 0),
                    ('b', 1),
                    ('b', 2)],
                   name='z')

        See Also
        --------
        DataArray.unstack
        """
        ds = self._to_temp_dataset().stack(
            dimensions,
            create_index=create_index,
            index_cls=index_cls,
            **dimensions_kwargs,
        )
        return self._from_temp_dataset(ds)

    # change type of self and return to T_DataArray once
    # https://github.com/python/mypy/issues/12846 is resolved
```
### 8 - xarray/core/dataarray.py:

Start line: 226, End line: 999

```python
class DataArray(
    AbstractArray, DataWithCoords, DataArrayArithmetic, DataArrayReductions
):
    """N-dimensional array with labeled coordinates and dimensions.

    DataArray provides a wrapper around numpy ndarrays that uses
    labeled dimensions and coordinates to support metadata aware
    operations. The API is similar to that for the pandas Series or
    DataFrame, but DataArray objects can have any number of dimensions,
    and their contents have fixed data types.

    Additional features over raw numpy arrays:

    - Apply operations over dimensions by name: ``x.sum('time')``.
    - Select or assign values by integer location (like numpy):
      ``x[:10]`` or by label (like pandas): ``x.loc['2014-01-01']`` or
      ``x.sel(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across
      multiple dimensions (known in numpy as "broadcasting") based on
      dimension names, regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python
      dictionary: ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a
    DataArray always returns another DataArray.

    Parameters
    ----------
    data : array_like
        Values for this array. Must be an ``numpy.ndarray``, ndarray
        like, or castable to an ``ndarray``. If a self-described xarray
        or pandas object, attempts are made to use this array's
        metadata to fill in other unspecified arguments. A view of the
        array's data is used instead of a copy if possible.
    coords : sequence or dict of array_like, optional
        Coordinates (tick labels) to use for indexing along each
        dimension. The following notations are accepted:

        - mapping {dimension name: array-like}
        - sequence of tuples that are valid arguments for
          ``xarray.Variable()``
          - (dims, data)
          - (dims, data, attrs)
          - (dims, data, attrs, encoding)

        Additionally, it is possible to define a coord whose name
        does not match the dimension name, or a coord based on multiple
        dimensions, with one of the following notations:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}

    dims : Hashable or sequence of Hashable, optional
        Name(s) of the data dimension(s). Must be either a Hashable
        (only for 1D data) or a sequence of Hashables with length equal
        to the number of dimensions. If this argument is omitted,
        dimension names are taken from ``coords`` (if possible) and
        otherwise default to ``['dim_0', ... 'dim_n']``.
    name : str or None, optional
        Name of this array.
    attrs : dict_like or None, optional
        Attributes to assign to the new instance. By default, an empty
        attribute dictionary is initialized.

    Examples
    --------
    Create data:

    >>> np.random.seed(0)
    >>> temperature = 15 + 8 * np.random.randn(2, 2, 3)
    >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
    >>> lat = [[42.25, 42.21], [42.63, 42.59]]
    >>> time = pd.date_range("2014-09-06", periods=3)
    >>> reference_time = pd.Timestamp("2014-09-05")

    Initialize a dataarray with multiple dimensions:

    >>> da = xr.DataArray(
    ...     data=temperature,
    ...     dims=["x", "y", "time"],
    ...     coords=dict(
    ...         lon=(["x", "y"], lon),
    ...         lat=(["x", "y"], lat),
    ...         time=time,
    ...         reference_time=reference_time,
    ...     ),
    ...     attrs=dict(
    ...         description="Ambient temperature.",
    ...         units="degC",
    ...     ),
    ... )
    >>> da
    <xarray.DataArray (x: 2, y: 2, time: 3)>
    array([[[29.11241877, 18.20125767, 22.82990387],
            [32.92714559, 29.94046392,  7.18177696]],
    <BLANKLINE>
           [[22.60070734, 13.78914233, 14.17424919],
            [18.28478802, 16.15234857, 26.63418806]]])
    Coordinates:
        lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        lat             (x, y) float64 42.25 42.21 42.63 42.59
      * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Dimensions without coordinates: x, y
    Attributes:
        description:  Ambient temperature.
        units:        degC

    Find out where the coldest temperature was:

    >>> da.isel(da.argmin(...))
    <xarray.DataArray ()>
    array(7.18177696)
    Coordinates:
        lon             float64 -99.32
        lat             float64 42.21
        time            datetime64[ns] 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Attributes:
        description:  Ambient temperature.
        units:        degC
    """

    _cache: dict[str, Any]
    _coords: dict[Any, Variable]
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index]
    _name: Hashable | None
    _variable: Variable

    __slots__ = (
        "_cache",
        "_coords",
        "_close",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )

    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor["DataArray"])

    def __init__(
        self,
        data: Any = dtypes.NA,
        coords: Sequence[Sequence[Any] | pd.Index | DataArray]
        | Mapping[Any, Any]
        | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        name: Hashable = None,
        attrs: Mapping = None,
        # internal parameters
        indexes: dict[Hashable, Index] = None,
        fastpath: bool = False,
    ) -> None:
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
            assert indexes is not None
        else:
            # TODO: (benbovy - explicit indexes) remove
            # once it becomes part of the public interface
            if indexes is not None:
                raise ValueError("Providing explicit indexes is not supported yet")

            # try to fill in arguments from data if they weren't supplied
            if coords is None:

                if isinstance(data, DataArray):
                    coords = data.coords
                elif isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]

            if dims is None:
                dims = getattr(data, "dims", getattr(coords, "dims", None))
            if name is None:
                name = getattr(data, "name", None)
            if attrs is None and not isinstance(data, PANDAS_TYPES):
                attrs = getattr(data, "attrs", None)

            data = _check_data_shape(data, coords, dims)
            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, fastpath=True)
            indexes, coords = _create_indexes_from_coords(coords)

        # These fully describe a DataArray
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name

        # TODO(shoyer): document this argument, once it becomes part of the
        # public interface.
        self._indexes = indexes  # type: ignore[assignment]

        self._close = None

    @classmethod
    def _construct_direct(
        cls: type[T_DataArray],
        variable: Variable,
        coords: dict[Any, Variable],
        name: Hashable,
        indexes: dict[Hashable, Index],
    ) -> T_DataArray:
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        obj = object.__new__(cls)
        obj._variable = variable
        obj._coords = coords
        obj._name = name
        obj._indexes = indexes
        obj._close = None
        return obj

    def _replace(
        self: T_DataArray,
        variable: Variable = None,
        coords=None,
        name: Hashable | None | Default = _default,
        indexes=None,
    ) -> T_DataArray:
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if indexes is None:
            indexes = self._indexes
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, indexes=indexes, fastpath=True)

    def _replace_maybe_drop_dims(
        self: T_DataArray,
        variable: Variable,
        name: Hashable | None | Default = _default,
    ) -> T_DataArray:
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
            indexes = self._indexes
        elif variable.dims == self.dims:
            # Shape has changed (e.g. from reduce(..., keepdims=True)
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = {
                k: v
                for k, v in self._coords.items()
                if v.shape == tuple(new_sizes[d] for d in v.dims)
            }
            indexes = filter_indexes_from_coords(self._indexes, set(coords))
        else:
            allowed_dims = set(variable.dims)
            coords = {
                k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            }
            indexes = filter_indexes_from_coords(self._indexes, set(coords))
        return self._replace(variable, coords, name, indexes=indexes)

    def _overwrite_indexes(
        self: T_DataArray,
        indexes: Mapping[Any, Index],
        coords: Mapping[Any, Variable] = None,
        drop_coords: list[Hashable] = None,
        rename_dims: Mapping[Any, Any] = None,
    ) -> T_DataArray:
        """Maybe replace indexes and their corresponding coordinates."""
        if not indexes:
            return self

        if coords is None:
            coords = {}
        if drop_coords is None:
            drop_coords = []

        new_variable = self.variable.copy()
        new_coords = self._coords.copy()
        new_indexes = dict(self._indexes)

        for name in indexes:
            new_coords[name] = coords[name]
            new_indexes[name] = indexes[name]

        for name in drop_coords:
            new_coords.pop(name)
            new_indexes.pop(name)

        if rename_dims:
            new_variable.dims = [rename_dims.get(d, d) for d in new_variable.dims]

        return self._replace(
            variable=new_variable, coords=new_coords, indexes=new_indexes
        )

    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)

    def _from_temp_dataset(
        self: T_DataArray, dataset: Dataset, name: Hashable | None | Default = _default
    ) -> T_DataArray:
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        indexes = dataset._indexes
        return self._replace(variable, coords, name, indexes=indexes)

    def _to_dataset_split(self, dim: Hashable) -> Dataset:
        """splits dataarray along dimension 'dim'"""

        def subset(dim, label):
            array = self.loc[{dim: label}]
            array.attrs = {}
            return as_variable(array)

        variables = {label: subset(dim, label) for label in self.get_index(dim)}
        variables.update({k: v for k, v in self._coords.items() if k != dim})
        coord_names = set(self._coords) - {dim}
        indexes = filter_indexes_from_coords(self._indexes, coord_names)
        dataset = Dataset._construct_direct(
            variables, coord_names, indexes=indexes, attrs=self.attrs
        )
        return dataset

    def _to_dataset_whole(
        self, name: Hashable = None, shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "unable to convert unnamed DataArray to a "
                "Dataset without providing an explicit name"
            )
        if name in self.coords:
            raise ValueError(
                "cannot create a Dataset from a DataArray with "
                "the same name as one of its coordinates"
            )
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        indexes = self._indexes

        coord_names = set(self._coords)
        return Dataset._construct_direct(variables, coord_names, indexes=indexes)

    def to_dataset(
        self,
        dim: Hashable = None,
        *,
        name: Hashable = None,
        promote_attrs: bool = False,
    ) -> Dataset:
        """Convert a DataArray to a Dataset.

        Parameters
        ----------
        dim : Hashable, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : Hashable, optional
            Name to substitute for this array's name. Only valid if ``dim`` is
            not provided.
        promote_attrs : bool, default: False
            Set to True to shallow copy attrs of DataArray to returned Dataset.

        Returns
        -------
        dataset : Dataset
        """
        if dim is not None and dim not in self.dims:
            raise TypeError(
                f"{dim} is not a dim. If supplying a ``name``, pass as a kwarg."
            )

        if dim is not None:
            if name is not None:
                raise TypeError("cannot supply both dim and name arguments")
            result = self._to_dataset_split(dim)
        else:
            result = self._to_dataset_whole(name)

        if promote_attrs:
            result.attrs = dict(self.attrs)

        return result

    @property
    def name(self) -> Hashable | None:
        """The name of this array."""
        return self._name

    @name.setter
    def name(self, value: Hashable | None) -> None:
        self._name = value

    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable

    @property
    def dtype(self) -> np.dtype:
        """
        Data-type of the array’s elements.

        See Also
        --------
        ndarray.dtype
        numpy.dtype
        """
        return self.variable.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions.

        See Also
        --------
        numpy.ndarray.shape
        """
        return self.variable.shape

    @property
    def size(self) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return self.variable.size

    @property
    def nbytes(self) -> int:
        """
        Total bytes consumed by the elements of this DataArray's data.

        If the underlying data array does not include ``nbytes``, estimates
        the bytes consumed based on the ``size`` and ``dtype``.
        """
        return self.variable.nbytes

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return self.variable.ndim

    def __len__(self) -> int:
        return len(self.variable)

    @property
    def data(self) -> Any:
        """
        The DataArray's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        See Also
        --------
        DataArray.to_numpy
        DataArray.as_numpy
        DataArray.values
        """
        return self.variable.data

    @data.setter
    def data(self, value: Any) -> None:
        self.variable.data = value

    @property
    def values(self) -> np.ndarray:
        """
        The array's data as a numpy.ndarray.

        If the array's data is not a numpy.ndarray this will attempt to convert
        it naively using np.array(), which will raise an error if the array
        type does not support coercion like this (e.g. cupy).
        """
        return self.variable.values

    @values.setter
    def values(self, value: Any) -> None:
        self.variable.values = value

    def to_numpy(self) -> np.ndarray:
        """
        Coerces wrapped data to numpy and returns a numpy.ndarray.

        See Also
        --------
        DataArray.as_numpy : Same but returns the surrounding DataArray instead.
        Dataset.as_numpy
        DataArray.values
        DataArray.data
        """
        return self.variable.to_numpy()

    def as_numpy(self: T_DataArray) -> T_DataArray:
        """
        Coerces wrapped data and coordinates into numpy arrays, returning a DataArray.

        See Also
        --------
        DataArray.to_numpy : Same but returns only the data as a numpy.ndarray object.
        Dataset.as_numpy : Converts all variables in a Dataset.
        DataArray.values
        DataArray.data
        """
        coords = {k: v.as_numpy() for k, v in self._coords.items()}
        return self._replace(self.variable.as_numpy(), coords, indexes=self._indexes)

    @property
    def _in_memory(self) -> bool:
        return self.variable._in_memory

    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index. Only possible for 1D
        arrays.
        """
        return self.variable.to_index()

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Tuple of dimension names associated with this array.

        Note that the type of this property is inconsistent with
        `Dataset.dims`.  See `Dataset.sizes` and `DataArray.sizes` for
        consistently named properties.

        See Also
        --------
        DataArray.sizes
        Dataset.dims
        """
        return self.variable.dims

    @dims.setter
    def dims(self, value: Any) -> NoReturn:
        raise AttributeError(
            "you cannot assign dims on a DataArray. Use "
            ".rename() or .swap_dims() instead."
        )

    def _item_key_to_dict(self, key: Any) -> Mapping[Hashable, Any]:
        if utils.is_dict_like(key):
            return key
        key = indexing.expanded_indexer(key, self.ndim)
        return dict(zip(self.dims, key))

    def _getitem_coord(self: T_DataArray, key: Any) -> T_DataArray:
        from .dataset import _get_virtual_variable

        try:
            var = self._coords[key]
        except KeyError:
            dim_sizes = dict(zip(self.dims, self.shape))
            _, key, var = _get_virtual_variable(self._coords, key, dim_sizes)

        return self._replace_maybe_drop_dims(var, name=key)

    def __getitem__(self: T_DataArray, key: Any) -> T_DataArray:
        if isinstance(key, str):
            return self._getitem_coord(key)
        else:
            # xarray-style array indexing
            return self.isel(indexers=self._item_key_to_dict(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.coords[key] = value
        else:
            # Coordinates in key, value and self[key] should be consistent.
            # TODO Coordinate consistency in key is checked here, but it
            # causes unnecessary indexing. It should be optimized.
            obj = self[key]
            if isinstance(value, DataArray):
                assert_coordinate_consistent(value, obj.coords.variables)
            # DataArray key -> Variable key
            key = {
                k: v.variable if isinstance(v, DataArray) else v
                for k, v in self._item_key_to_dict(key).items()
            }
            self.variable[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.coords[key]

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        yield from self._item_sources
        yield self.attrs

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-completion"""
        yield HybridMappingProxy(keys=self._coords, mapping=self.coords)

        # virtual coordinates
        # uses empty dict -- everything here can already be found in self.coords.
        yield HybridMappingProxy(keys=self.dims, mapping={})

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing like pandas."""
        return _LocIndexer(self)

    @property
    def attrs(self) -> dict[Any, Any]:
        """Dictionary storing arbitrary metadata with this array."""
        return self.variable.attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self.variable.attrs = dict(value)

    @property
    def encoding(self) -> dict[Any, Any]:
        """Dictionary of format-specific settings for how this array should be
        serialized."""
        return self.variable.encoding

    @encoding.setter
    def encoding(self, value: Mapping[Any, Any]) -> None:
        self.variable.encoding = dict(value)

    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this Dataset has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        DataArray.xindexes

        """
        return self.xindexes.to_pandas_indexes()

    @property
    def xindexes(self) -> Indexes:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._coords[k] for k in self._indexes})

    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays."""
        return DataArrayCoordinates(self)

    @overload
    def reset_coords(
        self: T_DataArray,
        names: Dims = None,
        drop: Literal[False] = False,
    ) -> Dataset:
        ...

    @overload
    def reset_coords(
        self: T_DataArray,
        names: Dims = None,
        *,
        drop: Literal[True],
    ) -> T_DataArray:
        ...

    def reset_coords(
        self: T_DataArray,
        names: Dims = None,
        drop: bool = False,
    ) -> T_DataArray | Dataset:
        """Given names of coordinates, reset them to become variables.

        Parameters
        ----------
        names : str, Iterable of Hashable or None, optional
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
        drop : bool, default: False
            If True, remove coordinates instead of converting them into
            variables.

        Returns
        -------
        Dataset, or DataArray if ``drop == True``

        Examples
        --------
        >>> temperature = np.arange(25).reshape(5, 5)
        >>> pressure = np.arange(50, 75).reshape(5, 5)
        >>> da = xr.DataArray(
        ...     data=temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 15)),
        ...         lat=("y", np.arange(20, 25)),
        ...         Pressure=(["x", "y"], pressure),
        ...     ),
        ...     name="Temperature",
        ... )
        >>> da
        <xarray.DataArray 'Temperature' (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Coordinates:
            lon       (x) int64 10 11 12 13 14
            lat       (y) int64 20 21 22 23 24
            Pressure  (x, y) int64 50 51 52 53 54 55 56 57 ... 67 68 69 70 71 72 73 74
        Dimensions without coordinates: x, y

        Return Dataset with target coordinate as a data variable rather than a coordinate variable:

        >>> da.reset_coords(names="Pressure")
        <xarray.Dataset>
        Dimensions:      (x: 5, y: 5)
        Coordinates:
            lon          (x) int64 10 11 12 13 14
            lat          (y) int64 20 21 22 23 24
        Dimensions without coordinates: x, y
        Data variables:
            Pressure     (x, y) int64 50 51 52 53 54 55 56 57 ... 68 69 70 71 72 73 74
            Temperature  (x, y) int64 0 1 2 3 4 5 6 7 8 9 ... 16 17 18 19 20 21 22 23 24

        Return DataArray without targeted coordinate:

        >>> da.reset_coords(names="Pressure", drop=True)
        <xarray.DataArray 'Temperature' (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Coordinates:
            lon      (x) int64 10 11 12 13 14
            lat      (y) int64 20 21 22 23 24
        Dimensions without coordinates: x, y
        """
        if names is None:
            names = set(self.coords) - set(self._indexes)
        dataset = self.coords.to_dataset().reset_coords(names, drop)
        if drop:
            return self._replace(coords=dataset._variables)
        if self.name is None:
            raise ValueError(
                "cannot reset_coords with drop=False on an unnamed DataArrray"
            )
        dataset[self.name] = self.variable
        return dataset
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
### 10 - xarray/core/dataset.py:

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
        Dims,
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
