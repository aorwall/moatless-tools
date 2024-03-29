# pydata__xarray-7150

| **pydata/xarray** | `f93b467db5e35ca94fefa518c32ee9bf93232475` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -234,7 +234,7 @@ def _get_mtime(filename_or_obj):
 
 def _protect_dataset_variables_inplace(dataset, cache):
     for name, variable in dataset.variables.items():
-        if name not in variable.dims:
+        if name not in dataset._indexes:
             # no need to protect IndexVariable objects
             data = indexing.CopyOnWriteArray(variable._data)
             if cache:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/backends/api.py | 237 | 237 | - | 4 | -


## Problem Statement

```
xarray.open_dataset has issues if the dataset returned by the backend contains a multiindex
### What happened?

As a follow up of this comment: https://github.com/pydata/xarray/issues/6752#issuecomment-1236756285 I'm currently trying to implement a custom `NetCDF4` backend that allows me to also handle multiindices when loading a NetCDF dataset using `xr.open_dataset`. 

I'm using the following two functions to convert the dataset to a NetCDF compatible version and back again:
https://github.com/pydata/xarray/issues/1077#issuecomment-1101505074.

Here is a small code example:

### Creating the dataset
\`\`\`python
import xarray as xr
import pandas

def create_multiindex(**kwargs):
    return pandas.MultiIndex.from_arrays(list(kwargs.values()), names=kwargs.keys())

dataset = xr.Dataset()
dataset.coords["observation"] = ["A", "B"]
dataset.coords["wavelength"] = [0.4, 0.5, 0.6, 0.7]
dataset.coords["stokes"] = ["I", "Q"]
dataset["measurement"] = create_multiindex(
    observation=["A", "A", "B", "B"],
    wavelength=[0.4, 0.5, 0.6, 0.7],
    stokes=["I", "Q", "I", "I"],
)
\`\`\`

### Saving as NetCDF
\`\`\`python
from cf_xarray import encode_multi_index_as_compress
patched = encode_multi_index_as_compress(dataset)
patched.to_netcdf("multiindex.nc")
\`\`\`

### And loading again
\`\`\`python
from cf_xarray import decode_compress_to_multi_index
loaded = xr.open_dataset("multiindex.nc")
loaded = decode_compress_to_multiindex(loaded)
assert loaded.equals(dataset)  # works
\`\`\`

### Custom Backend
While the manual patching for saving is currently still required, I tried to at least work around the added function call in `open_dataset` by creating a custom NetCDF Backend:

\`\`\`python
# registered as netcdf4-multiindex backend in setup.py
class MultiindexNetCDF4BackendEntrypoint(NetCDF4BackendEntrypoint):
    def open_dataset(self, *args, handle_multiindex=True, **kwargs):
        ds = super().open_dataset(*args, **kwargs)

        if handle_multiindex:  # here is where the restore operation happens:
            ds = decode_compress_to_multiindex(ds)

        return ds
\`\`\`

### The error
\`\`\`python
>>> loaded = xr.open_dataset("multiindex.nc", engine="netcdf4-multiindex", handle_multiindex=True)  # fails

File ~/.local/share/virtualenvs/test-oePfdNug/lib/python3.8/site-packages/xarray/core/variable.py:2795, in IndexVariable.data(self, data)
   2793 @Variable.data.setter  # type: ignore[attr-defined]
   2794 def data(self, data):
-> 2795     raise ValueError(
   2796         f"Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. "
   2797         f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
   2798     )

ValueError: Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable 'measurement'. Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate.
\`\`\`

but this works:
\`\`\`python
>>> loaded = xr.open_dataset("multiindex.nc", engine="netcdf4-multiindex", handle_multiindex=False)
>>> loaded = decode_compress_to_multiindex(loaded)
>>> assert loaded.equals(dataset)
\`\`\`

So I'm guessing `xarray` is performing some operation on the dataset returned by the backend, and one of those leads to a failure if there is a multiindex already contained.

### What did you expect to happen?

I expected that it doesn't matter wheter `decode_compress_to_multi_index` is called inside the backend or afterwards, and the same dataset will be returned each time.

### Minimal Complete Verifiable Example

\`\`\`Python
See above.
\`\`\`


### MVCE confirmation

- [X] Minimal example — the example is as focused as reasonably possible to demonstrate the underlying issue in xarray.
- [X] Complete example — the example is self-contained, including all data and the text of any traceback.
- [X] Verifiable example — the example copy & pastes into an IPython prompt or [Binder notebook](https://mybinder.org/v2/gh/pydata/xarray/main?urlpath=lab/tree/doc/examples/blank_template.ipynb), returning the result.
- [X] New issue — a search of GitHub Issues suggests this is not a duplicate.

### Relevant log output

_No response_

### Anything else we need to know?

I'm also open to other suggestions how I could simplify the usage of multiindices, maybe there is an approach that doesn't require a custom backend at all?



### Environment

<details>
INSTALLED VERSIONS
------------------
commit: None
python: 3.8.10 (default, Jan 28 2022, 09:41:12) 
[GCC 9.3.0]
python-bits: 64
OS: Linux
OS-release: 5.10.102.1-microsoft-standard-WSL2
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: C.UTF-8
LOCALE: ('en_US', 'UTF-8')
libhdf5: 1.10.5
libnetcdf: 4.6.3

xarray: 2022.9.0
pandas: 1.5.0
numpy: 1.23.3
scipy: 1.9.1
netCDF4: 1.5.4
pydap: None
h5netcdf: None
h5py: 3.7.0
Nio: None
zarr: None
cftime: 1.6.2
nc_time_axis: None
PseudoNetCDF: None
rasterio: 1.3.2
cfgrib: None
iris: None
bottleneck: None
dask: None
distributed: None
matplotlib: 3.6.0
cartopy: 0.19.0.post1
seaborn: None
numbagg: None
fsspec: None
cupy: None
pint: None
sparse: 0.13.0
flox: None
numpy_groupies: None
setuptools: 65.3.0
pip: 22.2.2
conda: None
pytest: 7.1.3
IPython: 8.5.0
sphinx: 4.5.0
</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/__init__.py | 1 | 111| 692 | 692 | 692 | 
| 2 | 2 xarray/core/types.py | 1 | 93| 790 | 1482 | 1737 | 
| 3 | 3 xarray/core/dataarray.py | 1 | 96| 642 | 2124 | 59770 | 
| 4 | **4 xarray/backends/api.py** | 1 | 83| 566 | 2690 | 74405 | 
| 5 | 5 xarray/core/dataset.py | 1 | 135| 802 | 3492 | 152919 | 
| 6 | 6 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 4230 | 154420 | 
| 7 | 7 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 4672 | 158125 | 
| 8 | 7 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 4936 | 158125 | 
| 9 | 8 xarray/core/common.py | 1 | 52| 283 | 5219 | 173031 | 
| 10 | 8 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 5568 | 173031 | 
| 11 | 8 asv_bench/benchmarks/dataset_io.py | 222 | 296| 631 | 6199 | 173031 | 
| 12 | 8 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 6311 | 173031 | 
| 13 | 8 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 6576 | 173031 | 
| 14 | 9 xarray/core/indexes.py | 1 | 31| 155 | 6731 | 184177 | 
| 15 | 9 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 6894 | 184177 | 
| 16 | 9 asv_bench/benchmarks/dataset_io.py | 315 | 328| 114 | 7008 | 184177 | 
| 17 | **9 xarray/backends/api.py** | 928 | 997| 801 | 7809 | 184177 | 
| 18 | 9 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 8078 | 184177 | 
| 19 | **9 xarray/backends/api.py** | 998 | 1050| 556 | 8634 | 184177 | 
| 20 | 10 xarray/backends/netCDF4_.py | 1 | 50| 328 | 8962 | 188370 | 
| 21 | 11 xarray/backends/pseudonetcdf_.py | 1 | 28| 179 | 9141 | 189385 | 
| 22 | 12 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 9308 | 189552 | 
| 23 | 13 xarray/core/indexing.py | 1 | 32| 195 | 9503 | 202168 | 


### Hint

```
Hi @lukasbindreiter, could you add the whole error traceback please?
I can see this type of decoding breaking some assumption in the file reading process. A full traceback would help identify where.

I think the real solution is actually #4490, so you could explicitly provide a coder.
Here is the full stacktrace:

\`\`\`python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In [12], line 7
----> 7 loaded = xr.open_dataset("multiindex.nc", engine="netcdf4-multiindex", handle_multiindex=True)
      8 print(loaded)

File ~/.local/share/virtualenvs/test-oePfdNug/lib/python3.8/site-packages/xarray/backends/api.py:537, in open_dataset(filename_or_obj, engine, chunks, cache, decode_cf, mask_and_scale, decode_times, decode_timedelta, use_cftime, concat_characters, decode_coords, drop_variables, inline_array, backend_kwargs, **kwargs)
    530 overwrite_encoded_chunks = kwargs.pop("overwrite_encoded_chunks", None)
    531 backend_ds = backend.open_dataset(
    532     filename_or_obj,
    533     drop_variables=drop_variables,
    534     **decoders,
    535     **kwargs,
    536 )
--> 537 ds = _dataset_from_backend_dataset(
    538     backend_ds,
    539     filename_or_obj,
    540     engine,
    541     chunks,
    542     cache,
    543     overwrite_encoded_chunks,
    544     inline_array,
    545     drop_variables=drop_variables,
    546     **decoders,
    547     **kwargs,
    548 )
    549 return ds

File ~/.local/share/virtualenvs/test-oePfdNug/lib/python3.8/site-packages/xarray/backends/api.py:345, in _dataset_from_backend_dataset(backend_ds, filename_or_obj, engine, chunks, cache, overwrite_encoded_chunks, inline_array, **extra_tokens)
    340 if not isinstance(chunks, (int, dict)) and chunks not in {None, "auto"}:
    341     raise ValueError(
    342         f"chunks must be an int, dict, 'auto', or None. Instead found {chunks}."
    343     )
--> 345 _protect_dataset_variables_inplace(backend_ds, cache)
    346 if chunks is None:
    347     ds = backend_ds

File ~/.local/share/virtualenvs/test-oePfdNug/lib/python3.8/site-packages/xarray/backends/api.py:239, in _protect_dataset_variables_inplace(dataset, cache)
    237 if cache:
    238     data = indexing.MemoryCachedArray(data)
--> 239 variable.data = data

File ~/.local/share/virtualenvs/test-oePfdNug/lib/python3.8/site-packages/xarray/core/variable.py:2795, in IndexVariable.data(self, data)
   2793 @Variable.data.setter  # type: ignore[attr-defined]
   2794 def data(self, data):
-> 2795     raise ValueError(
   2796         f"Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. "
   2797         f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
   2798     )

ValueError: Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable 'measurement'. Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate.
\`\`\`
Looks like the backend logic needs some updates to make it compatible with the new xarray data model with explicit indexes (i.e., possible indexed coordinates with name != dimension like for multi-index levels now), e.g., here:

https://github.com/pydata/xarray/blob/8eea8bb67bad0b5ac367c082125dd2b2519d4f52/xarray/backends/api.py#L234-L241


```

## Patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -234,7 +234,7 @@ def _get_mtime(filename_or_obj):
 
 def _protect_dataset_variables_inplace(dataset, cache):
     for name, variable in dataset.variables.items():
-        if name not in variable.dims:
+        if name not in dataset._indexes:
             # no need to protect IndexVariable objects
             data = indexing.CopyOnWriteArray(variable._data)
             if cache:

```

## Test Patch

```diff
diff --git a/xarray/tests/test_backends_api.py b/xarray/tests/test_backends_api.py
--- a/xarray/tests/test_backends_api.py
+++ b/xarray/tests/test_backends_api.py
@@ -48,6 +48,25 @@ def open_dataset(
     assert_identical(expected, actual)
 
 
+def test_multiindex() -> None:
+    # GH7139
+    # Check that we properly handle backends that change index variables
+    dataset = xr.Dataset(coords={"coord1": ["A", "B"], "coord2": [1, 2]})
+    dataset = dataset.stack(z=["coord1", "coord2"])
+
+    class MultiindexBackend(xr.backends.BackendEntrypoint):
+        def open_dataset(
+            self,
+            filename_or_obj,
+            drop_variables=None,
+            **kwargs,
+        ) -> xr.Dataset:
+            return dataset.copy(deep=True)
+
+    loaded = xr.open_dataset("fake_filename", engine=MultiindexBackend)
+    assert_identical(dataset, loaded)
+
+
 class PassThroughBackendEntrypoint(xr.backends.BackendEntrypoint):
     """Access an object passed to the `open_dataset` method."""
 

```


## Code snippets

### 1 - xarray/__init__.py:

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
### 2 - xarray/core/types.py:

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
### 3 - xarray/core/dataarray.py:

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
### 4 - xarray/backends/api.py:

Start line: 1, End line: 83

```python
from __future__ import annotations

import os
from functools import partial
from glob import glob
from io import BytesIO
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

import numpy as np

from .. import backends, conventions
from ..core import indexing
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from ..core.dataarray import DataArray
from ..core.dataset import Dataset, _get_chunk, _maybe_chunk
from ..core.indexes import Index
from ..core.utils import is_remote_uri
from . import plugins
from .common import AbstractDataStore, ArrayWriter, _normalize_path
from .locks import _get_scheduler

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None  # type: ignore
    from io import BufferedIOBase

    from ..core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
        NestedSequence,
    )
    from .common import BackendEntrypoint

    T_NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
    T_Engine = Union[
        T_NetcdfEngine,
        Literal["pydap", "pynio", "pseudonetcdf", "cfgrib", "zarr"],
        Type[BackendEntrypoint],
        str,  # no nice typing support for custom backends
        None,
    ]
    T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
    T_NetcdfTypes = Literal[
        "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
    ]


DATAARRAY_NAME = "__xarray_dataarray_name__"
DATAARRAY_VARIABLE = "__xarray_dataarray_variable__"

ENGINES = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "pydap": backends.PydapDataStore.open,
    "h5netcdf": backends.H5NetCDFStore.open,
    "pynio": backends.NioDataStore,
    "pseudonetcdf": backends.PseudoNetCDFDataStore.open,
    "cfgrib": backends.CfGribDataStore,
    "zarr": backends.ZarrStore.open_group,
}
```
### 5 - xarray/core/dataset.py:

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
### 7 - asv_bench/benchmarks/dataset_io.py:

Start line: 347, End line: 398

```python
class IOReadMultipleNetCDF4Dask(IOMultipleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF4"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_netcdf4_with_block_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.block_chunks
        ).load()

    def time_load_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.block_chunks
            ).load()

    def time_load_dataset_netcdf4_with_time_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.time_chunks
        ).load()

    def time_load_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.time_chunks
            ).load()

    def time_open_dataset_netcdf4_with_block_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.block_chunks
        )

    def time_open_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.block_chunks
            )

    def time_open_dataset_netcdf4_with_time_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.time_chunks
        )

    def time_open_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.time_chunks
            )
```
### 8 - asv_bench/benchmarks/dataset_io.py:

Start line: 401, End line: 432

```python
class IOReadMultipleNetCDF3Dask(IOReadMultipleNetCDF4Dask):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF3_64BIT"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.block_chunks
            ).load()

    def time_load_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.time_chunks
            ).load()

    def time_open_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.block_chunks
            )

    def time_open_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.time_chunks
            )
```
### 9 - xarray/core/common.py:

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
### 10 - asv_bench/benchmarks/dataset_io.py:

Start line: 150, End line: 187

```python
class IOReadSingleNetCDF4Dask(IOSingleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = "test_single_file.nc4.nc"
        self.format = "NETCDF4"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4_with_block_chunks(self):
        xr.open_dataset(
            self.filepath, engine="netcdf4", chunks=self.block_chunks
        ).load()

    def time_load_dataset_netcdf4_with_block_chunks_oindexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.block_chunks)
        ds = ds.isel(**self.oinds).load()

    def time_load_dataset_netcdf4_with_block_chunks_vindexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.block_chunks)
        ds = ds.isel(**self.vinds).load()

    def time_load_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="netcdf4", chunks=self.block_chunks
            ).load()

    def time_load_dataset_netcdf4_with_time_chunks(self):
        xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.time_chunks).load()

    def time_load_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="netcdf4", chunks=self.time_chunks
            ).load()
```
### 17 - xarray/backends/api.py:

Start line: 928, End line: 997

```python
def open_mfdataset(
    paths: str | NestedSequence[str | os.PathLike],
    chunks: T_Chunks = None,
    concat_dim: str
    | DataArray
    | Index
    | Sequence[str]
    | Sequence[DataArray]
    | Sequence[Index]
    | None = None,
    compat: CompatOptions = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    engine: T_Engine = None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: bool = False,
    join: JoinOptions = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: CombineAttrsOptions = "override",
    **kwargs,
) -> Dataset:
    if isinstance(paths, str):
        if is_remote_uri(paths) and engine == "zarr":
            try:
                from fsspec.core import get_fs_token_paths
            except ImportError as e:
                raise ImportError(
                    "The use of remote URLs for opening zarr requires the package fsspec"
                ) from e

            fs, _, _ = get_fs_token_paths(
                paths,
                mode="rb",
                storage_options=kwargs.get("backend_kwargs", {}).get(
                    "storage_options", {}
                ),
                expand=False,
            )
            tmp_paths = fs.glob(fs._strip_protocol(paths))  # finds directories
            paths = [fs.get_mapper(path) for path in tmp_paths]
        elif is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs "
                f"unless engine='zarr' is specified. Got paths: {paths}. "
                "Instead, supply paths as an explicit list of strings."
            )
        else:
            paths = sorted(glob(_normalize_path(paths)))
    elif isinstance(paths, os.PathLike):
        paths = [os.fspath(paths)]
    else:
        paths = [os.fspath(p) if isinstance(p, os.PathLike) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    if combine == "nested":
        if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
            concat_dim = [concat_dim]  # type: ignore[assignment]

        # This creates a flat list which is easier to iterate over, whilst
        # encoding the originally-supplied structure as "ids".
        # The "ids" are not used at all if combine='by_coords`.
        combined_ids_paths = _infer_concat_order_from_positions(paths)
        ids, paths = (
            list(combined_ids_paths.keys()),
            list(combined_ids_paths.values()),
        )
    elif combine == "by_coords" and concat_dim is not None:
        raise ValueError(
            "When combine='by_coords', passing a value for `concat_dim` has no "
            "effect. To manually combine along a specific dimension you should "
            "instead specify combine='nested' along with a value for `concat_dim`.",
        )

    open_kwargs = dict(engine=engine, chunks=chunks or {}, **kwargs)

    if parallel:
        import dask

        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(open_dataset)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    else:
        open_ = open_dataset
        getattr_ = getattr

    datasets = [open_(p, **open_kwargs) for p in paths]
    closers = [getattr_(ds, "_close") for ds in datasets]
    # ... other code
```
### 19 - xarray/backends/api.py:

Start line: 998, End line: 1050

```python
def open_mfdataset(
    paths: str | NestedSequence[str | os.PathLike],
    chunks: T_Chunks = None,
    concat_dim: str
    | DataArray
    | Index
    | Sequence[str]
    | Sequence[DataArray]
    | Sequence[Index]
    | None = None,
    compat: CompatOptions = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    engine: T_Engine = None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: bool = False,
    join: JoinOptions = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: CombineAttrsOptions = "override",
    **kwargs,
) -> Dataset:
    # ... other code
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, closers = dask.compute(datasets, closers)

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined = _nested_combine(
                datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
                combine_attrs=combine_attrs,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(
                datasets,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                combine_attrs=combine_attrs,
            )
        else:
            raise ValueError(
                "{} is an invalid option for the keyword argument"
                " ``combine``".format(combine)
            )
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    combined.set_close(partial(_multi_file_closer, closers))

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, os.PathLike):
            attrs_file = cast(str, os.fspath(attrs_file))
        combined.attrs = datasets[paths.index(attrs_file)].attrs

    return combined
```
