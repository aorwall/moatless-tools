# pydata__xarray-6889

| **pydata/xarray** | `790a444b11c244fd2d33e2d2484a590f8fc000ff` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 14 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -490,7 +490,7 @@ def _dataset_concat(
     )
 
     # determine which variables to merge, and then merge them according to compat
-    variables_to_merge = (coord_names | data_names) - concat_over - dim_names
+    variables_to_merge = (coord_names | data_names) - concat_over - unlabeled_dims
 
     result_vars = {}
     result_indexes = {}

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/concat.py | 493 | 493 | - | 14 | -


## Problem Statement

```
Alignment of dataset with MultiIndex fails after applying xr.concat  
### What happened?

After applying the `concat` function to a dataset with a Multiindex, a lot of functions related to indexing are broken. For example, it is not possible to apply `reindex_like` to itself anymore. 

The error is raised in the alignment module. It seems that the function `find_matching_indexes` does not find indexes that belong to the same dimension. 

### What did you expect to happen?

I expected the alignment to be functional and that these basic functions work.  

### Minimal Complete Verifiable Example

\`\`\`Python
import xarray as xr
import pandas as pd

index = pd.MultiIndex.from_product([[1,2], ['a', 'b']], names=('level1', 'level2'))
index.name = 'dim'

var = xr.DataArray(1, coords=[index])
ds = xr.Dataset({"var":var})

new = xr.concat([ds], dim='newdim')
xr.Dataset(new) # breaks
new.reindex_like(new) # breaks
\`\`\`


### MVCE confirmation

- [X] Minimal example — the example is as focused as reasonably possible to demonstrate the underlying issue in xarray.
- [X] Complete example — the example is self-contained, including all data and the text of any traceback.
- [X] Verifiable example — the example copy & pastes into an IPython prompt or [Binder notebook](https://mybinder.org/v2/gh/pydata/xarray/main?urlpath=lab/tree/doc/examples/blank_template.ipynb), returning the result.
- [X] New issue — a search of GitHub Issues suggests this is not a duplicate.

### Relevant log output

\`\`\`Python
Traceback (most recent call last):

  File "/tmp/ipykernel_407170/4030736219.py", line 11, in <cell line: 11>
    xr.Dataset(new) # breaks

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/dataset.py", line 599, in __init__
    variables, coord_names, dims, indexes, _ = merge_data_and_coords(

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/merge.py", line 575, in merge_data_and_coords
    return merge_core(

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/merge.py", line 752, in merge_core
    aligned = deep_align(

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/alignment.py", line 827, in deep_align
    aligned = align(

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/alignment.py", line 764, in align
    aligner.align()

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/alignment.py", line 550, in align
    self.assert_no_index_conflict()

  File "/home/fabian/.miniconda3/lib/python3.10/site-packages/xarray/core/alignment.py", line 319, in assert_no_index_conflict
    raise ValueError(

ValueError: cannot re-index or align objects with conflicting indexes found for the following dimensions: 'dim' (2 conflicting indexes)
Conflicting indexes may occur when
- they relate to different sets of coordinate and/or dimension names
- they don't have the same type
- they may be used to reindex data along common dimensions
\`\`\`


### Anything else we need to know?

_No response_

### Environment

<details>

INSTALLED VERSIONS
------------------
commit: None
python: 3.10.5 | packaged by conda-forge | (main, Jun 14 2022, 07:04:59) [GCC 10.3.0]
python-bits: 64
OS: Linux
OS-release: 5.15.0-41-generic
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_US.UTF-8
LOCALE: ('en_US', 'UTF-8')
libhdf5: 1.12.1
libnetcdf: 4.8.1

xarray: 2022.6.0
pandas: 1.4.2
numpy: 1.21.6
scipy: 1.8.1
netCDF4: 1.6.0
pydap: None
h5netcdf: None
h5py: 3.6.0
Nio: None
zarr: None
cftime: 1.5.1.1
nc_time_axis: None
PseudoNetCDF: None
rasterio: 1.2.10
cfgrib: None
iris: None
bottleneck: 1.3.4
dask: 2022.6.1
distributed: 2022.6.1
matplotlib: 3.5.1
cartopy: 0.20.2
seaborn: 0.11.2
numbagg: None
fsspec: 2022.3.0
cupy: None
pint: None
sparse: 0.13.0
flox: None
numpy_groupies: None
setuptools: 61.2.0
pip: 22.1.2
conda: 4.13.0
pytest: 7.1.2
IPython: 7.33.0
sphinx: 5.0.2
/home/fabian/.miniconda3/lib/python3.10/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/alignment.py | 568 | 765| 1982 | 1982 | 8510 | 
| 2 | 1 xarray/core/alignment.py | 243 | 285| 402 | 2384 | 8510 | 
| 3 | 1 xarray/core/alignment.py | 1 | 36| 206 | 2590 | 8510 | 
| 4 | 2 xarray/core/types.py | 1 | 79| 771 | 3361 | 9495 | 
| 5 | 3 xarray/core/merge.py | 792 | 1008| 2810 | 6171 | 18765 | 
| 6 | 4 xarray/core/indexes.py | 430 | 484| 454 | 6625 | 29380 | 
| 7 | 5 xarray/core/dataarray.py | 1 | 95| 638 | 7263 | 80593 | 
| 8 | 6 xarray/__init__.py | 1 | 111| 692 | 7955 | 81285 | 
| 9 | 6 xarray/core/alignment.py | 534 | 765| 233 | 8188 | 81285 | 
| 10 | 6 xarray/core/indexes.py | 918 | 934| 147 | 8335 | 81285 | 
| 11 | 7 xarray/core/dataset.py | 1 | 134| 798 | 9133 | 157606 | 
| 12 | 8 xarray/core/parallel.py | 1 | 48| 227 | 9360 | 162721 | 
| 13 | 9 xarray/core/combine.py | 674 | 926| 3146 | 12506 | 171804 | 
| 14 | 9 xarray/core/alignment.py | 440 | 455| 156 | 12662 | 171804 | 
| 15 | 10 asv_bench/benchmarks/reindexing.py | 1 | 53| 415 | 13077 | 172219 | 
| 16 | 10 xarray/core/alignment.py | 890 | 936| 319 | 13396 | 172219 | 
| 17 | 11 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 14134 | 173720 | 
| 18 | 11 xarray/core/indexes.py | 586 | 604| 156 | 14290 | 173720 | 
| 19 | 12 xarray/core/coordinates.py | 1 | 22| 157 | 14447 | 177198 | 
| 20 | 12 xarray/core/alignment.py | 851 | 887| 286 | 14733 | 177198 | 
| 21 | 12 xarray/core/indexes.py | 771 | 875| 898 | 15631 | 177198 | 
| 22 | 13 xarray/core/missing.py | 1 | 25| 184 | 15815 | 183445 | 
| 23 | 13 xarray/core/alignment.py | 514 | 532| 138 | 15953 | 183445 | 
| 24 | 13 xarray/core/indexes.py | 1 | 31| 155 | 16108 | 183445 | 
| 25 | 13 xarray/core/alignment.py | 287 | 326| 416 | 16524 | 183445 | 
| 26 | 13 xarray/core/indexes.py | 284 | 304| 178 | 16702 | 183445 | 
| 27 | 13 xarray/core/alignment.py | 100 | 179| 629 | 17331 | 183445 | 
| 28 | 13 xarray/core/alignment.py | 476 | 489| 133 | 17464 | 183445 | 
| 29 | 13 xarray/core/alignment.py | 491 | 512| 206 | 17670 | 183445 | 
| 30 | 13 xarray/core/indexes.py | 877 | 916| 314 | 17984 | 183445 | 
| 31 | **14 xarray/core/concat.py** | 1 | 22| 140 | 18124 | 189153 | 
| 32 | 14 xarray/core/dataarray.py | 1628 | 1928| 1551 | 19675 | 189153 | 
| 33 | 14 asv_bench/benchmarks/indexing.py | 133 | 150| 145 | 19820 | 189153 | 


## Patch

```diff
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -490,7 +490,7 @@ def _dataset_concat(
     )
 
     # determine which variables to merge, and then merge them according to compat
-    variables_to_merge = (coord_names | data_names) - concat_over - dim_names
+    variables_to_merge = (coord_names | data_names) - concat_over - unlabeled_dims
 
     result_vars = {}
     result_indexes = {}

```

## Test Patch

```diff
diff --git a/xarray/tests/test_concat.py b/xarray/tests/test_concat.py
--- a/xarray/tests/test_concat.py
+++ b/xarray/tests/test_concat.py
@@ -513,6 +513,16 @@ def test_concat_multiindex(self) -> None:
         assert expected.equals(actual)
         assert isinstance(actual.x.to_index(), pd.MultiIndex)
 
+    def test_concat_along_new_dim_multiindex(self) -> None:
+        # see https://github.com/pydata/xarray/issues/6881
+        level_names = ["x_level_0", "x_level_1"]
+        x = pd.MultiIndex.from_product([[1, 2, 3], ["a", "b"]], names=level_names)
+        ds = Dataset(coords={"x": x})
+        concatenated = concat([ds], "new")
+        actual = list(concatenated.xindexes.get_all_coords("x"))
+        expected = ["x"] + level_names
+        assert actual == expected
+
     @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"a": 2, "b": 1}])
     def test_concat_fill_value(self, fill_value) -> None:
         datasets = [

```


## Code snippets

### 1 - xarray/core/alignment.py:

Start line: 568, End line: 765

```python
def align(
    *objects: DataAlignable,
    join: JoinOptions = "inner",
    copy: bool = True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
) -> tuple[DataAlignable, ...]:
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
    join : {"outer", "inner", "left", "right", "exact", "override"}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    copy : bool, default: True
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : sequence of str, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    Returns
    -------
    aligned : tuple of DataArray or Dataset
        Tuple of objects with the same type as `*objects` with aligned
        coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.

    Examples
    --------
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
    ValueError: cannot align objects with join='exact' ...

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
    aligner = Aligner(
        objects,
        join=join,
        copy=copy,
        indexes=indexes,
        exclude_dims=exclude,
        fill_value=fill_value,
    )
    aligner.align()
    return aligner.results
```
### 2 - xarray/core/alignment.py:

Start line: 243, End line: 285

```python
class Aligner(Generic[DataAlignable]):

    def find_matching_indexes(self) -> None:
        all_indexes: dict[MatchingIndexKey, list[Index]]
        all_index_vars: dict[MatchingIndexKey, list[dict[Hashable, Variable]]]
        all_indexes_dim_sizes: dict[MatchingIndexKey, dict[Hashable, set]]
        objects_matching_indexes: list[dict[MatchingIndexKey, Index]]

        all_indexes = defaultdict(list)
        all_index_vars = defaultdict(list)
        all_indexes_dim_sizes = defaultdict(lambda: defaultdict(set))
        objects_matching_indexes = []

        for obj in self.objects:
            obj_indexes, obj_index_vars = self._normalize_indexes(obj.xindexes)
            objects_matching_indexes.append(obj_indexes)
            for key, idx in obj_indexes.items():
                all_indexes[key].append(idx)
            for key, index_vars in obj_index_vars.items():
                all_index_vars[key].append(index_vars)
                for dim, size in calculate_dimensions(index_vars).items():
                    all_indexes_dim_sizes[key][dim].add(size)

        self.objects_matching_indexes = tuple(objects_matching_indexes)
        self.all_indexes = all_indexes
        self.all_index_vars = all_index_vars

        if self.join == "override":
            for dim_sizes in all_indexes_dim_sizes.values():
                for dim, sizes in dim_sizes.items():
                    if len(sizes) > 1:
                        raise ValueError(
                            "cannot align objects with join='override' with matching indexes "
                            f"along dimension {dim!r} that don't have the same size"
                        )

    def find_matching_unindexed_dims(self) -> None:
        unindexed_dim_sizes = defaultdict(set)

        for obj in self.objects:
            for dim in obj.dims:
                if dim not in self.exclude_dims and dim not in obj.xindexes.dims:
                    unindexed_dim_sizes[dim].add(obj.sizes[dim])

        self.unindexed_dim_sizes = unindexed_dim_sizes
```
### 3 - xarray/core/alignment.py:

Start line: 1, End line: 36

```python
from __future__ import annotations

import functools
import operator
from collections import defaultdict
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd

from . import dtypes
from .common import DataWithCoords
from .indexes import Index, Indexes, PandasIndex, PandasMultiIndex, indexes_all_equal
from .utils import is_dict_like, is_full_slice, safe_cast_to_index
from .variable import Variable, as_compatible_data, calculate_dimensions

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset
    from .types import JoinOptions, T_DataArray, T_DataArrayOrSet, T_Dataset

DataAlignable = TypeVar("DataAlignable", bound=DataWithCoords)
```
### 4 - xarray/core/types.py:

Start line: 1, End line: 79

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from .common import DataWithCoords
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

ReindexMethodOptions = Literal["nearest", "pad", "ffill", "backfill", "bfill", None]
```
### 5 - xarray/core/merge.py:

Start line: 792, End line: 1008

```python
def merge(
    objects: Iterable[DataArray | CoercibleMapping],
    compat: CompatOptions = "no_conflicts",
    join: JoinOptions = "outer",
    fill_value: object = dtypes.NA,
    combine_attrs: CombineAttrsOptions = "override",
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
### 6 - xarray/core/indexes.py:

Start line: 430, End line: 484

```python
class PandasIndex(Index):

    def equals(self, other: Index):
        if not isinstance(other, PandasIndex):
            return False
        return self.index.equals(other.index) and self.dim == other.dim

    def join(self: PandasIndex, other: PandasIndex, how: str = "inner") -> PandasIndex:
        if how == "outer":
            index = self.index.union(other.index)
        else:
            # how = "inner"
            index = self.index.intersection(other.index)

        coord_dtype = np.result_type(self.coord_dtype, other.coord_dtype)
        return type(self)(index, self.dim, coord_dtype=coord_dtype)

    def reindex_like(
        self, other: PandasIndex, method=None, tolerance=None
    ) -> dict[Hashable, Any]:
        if not self.index.is_unique:
            raise ValueError(
                f"cannot reindex or align along dimension {self.dim!r} because the "
                "(pandas) index has duplicate values"
            )

        return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

    def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
        shift = shifts[self.dim] % self.index.shape[0]

        if shift != 0:
            new_pd_idx = self.index[-shift:].append(self.index[:-shift])
        else:
            new_pd_idx = self.index[:]

        return self._replace(new_pd_idx)

    def rename(self, name_dict, dims_dict):
        if self.index.name not in name_dict and self.dim not in dims_dict:
            return self

        new_name = name_dict.get(self.index.name, self.index.name)
        index = self.index.rename(new_name)
        new_dim = dims_dict.get(self.dim, self.dim)
        return self._replace(index, dim=new_dim)

    def copy(self, deep=True):
        if deep:
            index = self.index.copy(deep=True)
        else:
            # index will be copied in constructor
            index = self.index
        return self._replace(index)

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])
```
### 7 - xarray/core/dataarray.py:

Start line: 1, End line: 95

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
### 8 - xarray/__init__.py:

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
### 9 - xarray/core/alignment.py:

Start line: 534, End line: 765

```python
class Aligner(Generic[DataAlignable]):

    def reindex_all(self) -> None:
        self.results = tuple(
            self._reindex_one(obj, matching_indexes)
            for obj, matching_indexes in zip(
                self.objects, self.objects_matching_indexes
            )
        )

    def align(self) -> None:
        if not self.indexes and len(self.objects) == 1:
            # fast path for the trivial case
            (obj,) = self.objects
            self.results = (obj.copy(deep=self.copy),)

        self.find_matching_indexes()
        self.find_matching_unindexed_dims()
        self.assert_no_index_conflict()
        self.align_indexes()
        self.assert_unindexed_dim_sizes_equal()

        if self.join == "override":
            self.override_indexes()
        else:
            self.reindex_all()


def align(
    *objects: DataAlignable,
    join: JoinOptions = "inner",
    copy: bool = True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
) -> tuple[DataAlignable, ...]:
    # ... other code
```
### 10 - xarray/core/indexes.py:

Start line: 918, End line: 934

```python
class PandasMultiIndex(PandasIndex):

    def join(self, other, how: str = "inner"):
        if how == "outer":
            # bug in pandas? need to reset index.name
            other_index = other.index.copy()
            other_index.name = None
            index = self.index.union(other_index)
            index.name = self.dim
        else:
            # how = "inner"
            index = self.index.intersection(other.index)

        level_coords_dtype = {
            k: np.result_type(lvl_dtype, other.level_coords_dtype[k])
            for k, lvl_dtype in self.level_coords_dtype.items()
        }

        return type(self)(index, self.dim, level_coords_dtype=level_coords_dtype)
```
### 31 - xarray/core/concat.py:

Start line: 1, End line: 22

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Hashable, Iterable, cast, overload

import pandas as pd

from . import dtypes, utils
from .alignment import align
from .duck_array_ops import lazy_array_equiv
from .indexes import Index, PandasIndex
from .merge import (
    _VALID_COMPAT,
    collect_variables_and_indexes,
    merge_attrs,
    merge_collected,
)
from .types import T_DataArray, T_Dataset
from .variable import Variable
from .variable import concat as concat_vars

if TYPE_CHECKING:
    from .types import CombineAttrsOptions, CompatOptions, ConcatOptions, JoinOptions
```
