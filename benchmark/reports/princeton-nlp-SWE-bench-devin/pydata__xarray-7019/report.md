# pydata__xarray-7019

| **pydata/xarray** | `964d350a80fe21d4babf939c108986d5fd90a2cf` |
| ---- | ---- |
| **No of patches** | 23 |
| **All found context length** | 3876 |
| **Any found context length** | 1610 |
| **Avg pos** | 15.608695652173912 |
| **Min pos** | 2 |
| **Max pos** | 26 |
| **Top file pos** | 2 |
| **Missing snippets** | 132 |
| **Missing patch files** | 14 |


## Expected patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -6,7 +6,16 @@
 from glob import glob
 from io import BytesIO
 from numbers import Number
-from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Union, cast, overload
+from typing import (
+    TYPE_CHECKING,
+    Any,
+    Callable,
+    Final,
+    Literal,
+    Union,
+    cast,
+    overload,
+)
 
 import numpy as np
 
@@ -20,9 +29,11 @@
     _nested_combine,
     combine_by_coords,
 )
+from xarray.core.daskmanager import DaskManager
 from xarray.core.dataarray import DataArray
 from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
 from xarray.core.indexes import Index
+from xarray.core.parallelcompat import guess_chunkmanager
 from xarray.core.utils import is_remote_uri
 
 if TYPE_CHECKING:
@@ -38,6 +49,7 @@
         CompatOptions,
         JoinOptions,
         NestedSequence,
+        T_Chunks,
     )
 
     T_NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
@@ -48,7 +60,6 @@
         str,  # no nice typing support for custom backends
         None,
     ]
-    T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
     T_NetcdfTypes = Literal[
         "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
     ]
@@ -297,17 +308,27 @@ def _chunk_ds(
     chunks,
     overwrite_encoded_chunks,
     inline_array,
+    chunked_array_type,
+    from_array_kwargs,
     **extra_tokens,
 ):
-    from dask.base import tokenize
+    chunkmanager = guess_chunkmanager(chunked_array_type)
+
+    # TODO refactor to move this dask-specific logic inside the DaskManager class
+    if isinstance(chunkmanager, DaskManager):
+        from dask.base import tokenize
 
-    mtime = _get_mtime(filename_or_obj)
-    token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
-    name_prefix = f"open_dataset-{token}"
+        mtime = _get_mtime(filename_or_obj)
+        token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
+        name_prefix = "open_dataset-"
+    else:
+        # not used
+        token = (None,)
+        name_prefix = None
 
     variables = {}
     for name, var in backend_ds.variables.items():
-        var_chunks = _get_chunk(var, chunks)
+        var_chunks = _get_chunk(var, chunks, chunkmanager)
         variables[name] = _maybe_chunk(
             name,
             var,
@@ -316,6 +337,8 @@ def _chunk_ds(
             name_prefix=name_prefix,
             token=token,
             inline_array=inline_array,
+            chunked_array_type=chunkmanager,
+            from_array_kwargs=from_array_kwargs.copy(),
         )
     return backend_ds._replace(variables)
 
@@ -328,6 +351,8 @@ def _dataset_from_backend_dataset(
     cache,
     overwrite_encoded_chunks,
     inline_array,
+    chunked_array_type,
+    from_array_kwargs,
     **extra_tokens,
 ):
     if not isinstance(chunks, (int, dict)) and chunks not in {None, "auto"}:
@@ -346,6 +371,8 @@ def _dataset_from_backend_dataset(
             chunks,
             overwrite_encoded_chunks,
             inline_array,
+            chunked_array_type,
+            from_array_kwargs,
             **extra_tokens,
         )
 
@@ -373,6 +400,8 @@ def open_dataset(
     decode_coords: Literal["coordinates", "all"] | bool | None = None,
     drop_variables: str | Iterable[str] | None = None,
     inline_array: bool = False,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     backend_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ) -> Dataset:
@@ -465,6 +494,15 @@ def open_dataset(
         itself, and each chunk refers to that task by its key. With
         ``inline_array=True``, Dask will instead inline the array directly
         in the values of the task graph. See :py:func:`dask.array.from_array`.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce this datasets' arrays to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
     backend_kwargs: dict
         Additional keyword arguments passed on to the engine open function,
         equivalent to `**kwargs`.
@@ -508,6 +546,9 @@ def open_dataset(
     if engine is None:
         engine = plugins.guess_engine(filename_or_obj)
 
+    if from_array_kwargs is None:
+        from_array_kwargs = {}
+
     backend = plugins.get_backend(engine)
 
     decoders = _resolve_decoders_kwargs(
@@ -536,6 +577,8 @@ def open_dataset(
         cache,
         overwrite_encoded_chunks,
         inline_array,
+        chunked_array_type,
+        from_array_kwargs,
         drop_variables=drop_variables,
         **decoders,
         **kwargs,
@@ -546,8 +589,8 @@ def open_dataset(
 def open_dataarray(
     filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
     *,
-    engine: T_Engine = None,
-    chunks: T_Chunks = None,
+    engine: T_Engine | None = None,
+    chunks: T_Chunks | None = None,
     cache: bool | None = None,
     decode_cf: bool | None = None,
     mask_and_scale: bool | None = None,
@@ -558,6 +601,8 @@ def open_dataarray(
     decode_coords: Literal["coordinates", "all"] | bool | None = None,
     drop_variables: str | Iterable[str] | None = None,
     inline_array: bool = False,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     backend_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ) -> DataArray:
@@ -652,6 +697,15 @@ def open_dataarray(
         itself, and each chunk refers to that task by its key. With
         ``inline_array=True``, Dask will instead inline the array directly
         in the values of the task graph. See :py:func:`dask.array.from_array`.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
     backend_kwargs: dict
         Additional keyword arguments passed on to the engine open function,
         equivalent to `**kwargs`.
@@ -695,6 +749,8 @@ def open_dataarray(
         cache=cache,
         drop_variables=drop_variables,
         inline_array=inline_array,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
         backend_kwargs=backend_kwargs,
         use_cftime=use_cftime,
         decode_timedelta=decode_timedelta,
@@ -726,7 +782,7 @@ def open_dataarray(
 
 def open_mfdataset(
     paths: str | NestedSequence[str | os.PathLike],
-    chunks: T_Chunks = None,
+    chunks: T_Chunks | None = None,
     concat_dim: str
     | DataArray
     | Index
@@ -736,7 +792,7 @@ def open_mfdataset(
     | None = None,
     compat: CompatOptions = "no_conflicts",
     preprocess: Callable[[Dataset], Dataset] | None = None,
-    engine: T_Engine = None,
+    engine: T_Engine | None = None,
     data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
     coords="different",
     combine: Literal["by_coords", "nested"] = "by_coords",
@@ -1490,6 +1546,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> backends.ZarrStore:
     ...
 
@@ -1512,6 +1569,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> Delayed:
     ...
 
@@ -1531,6 +1589,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> backends.ZarrStore | Delayed:
     """This function creates an appropriate datastore for writing a dataset to
     a zarr ztore
@@ -1652,7 +1711,9 @@ def to_zarr(
     writer = ArrayWriter()
     # TODO: figure out how to properly handle unlimited_dims
     dump_to_store(dataset, zstore, writer, encoding=encoding)
-    writes = writer.sync(compute=compute)
+    writes = writer.sync(
+        compute=compute, chunkmanager_store_kwargs=chunkmanager_store_kwargs
+    )
 
     if compute:
         _finalize_store(writes, zstore)
diff --git a/xarray/backends/common.py b/xarray/backends/common.py
--- a/xarray/backends/common.py
+++ b/xarray/backends/common.py
@@ -11,7 +11,8 @@
 
 from xarray.conventions import cf_encoder
 from xarray.core import indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
 
 if TYPE_CHECKING:
@@ -153,7 +154,7 @@ def __init__(self, lock=None):
         self.lock = lock
 
     def add(self, source, target, region=None):
-        if is_duck_dask_array(source):
+        if is_chunked_array(source):
             self.sources.append(source)
             self.targets.append(target)
             self.regions.append(region)
@@ -163,21 +164,25 @@ def add(self, source, target, region=None):
             else:
                 target[...] = source
 
-    def sync(self, compute=True):
+    def sync(self, compute=True, chunkmanager_store_kwargs=None):
         if self.sources:
-            import dask.array as da
+            chunkmanager = get_chunked_array_type(*self.sources)
 
             # TODO: consider wrapping targets with dask.delayed, if this makes
             # for any discernible difference in perforance, e.g.,
             # targets = [dask.delayed(t) for t in self.targets]
 
-            delayed_store = da.store(
+            if chunkmanager_store_kwargs is None:
+                chunkmanager_store_kwargs = {}
+
+            delayed_store = chunkmanager.store(
                 self.sources,
                 self.targets,
                 lock=self.lock,
                 compute=compute,
                 flush=True,
                 regions=self.regions,
+                **chunkmanager_store_kwargs,
             )
             self.sources = []
             self.targets = []
diff --git a/xarray/backends/plugins.py b/xarray/backends/plugins.py
--- a/xarray/backends/plugins.py
+++ b/xarray/backends/plugins.py
@@ -146,7 +146,7 @@ def refresh_engines() -> None:
 
 def guess_engine(
     store_spec: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
-):
+) -> str | type[BackendEntrypoint]:
     engines = list_engines()
 
     for engine, backend in engines.items():
diff --git a/xarray/backends/zarr.py b/xarray/backends/zarr.py
--- a/xarray/backends/zarr.py
+++ b/xarray/backends/zarr.py
@@ -19,6 +19,7 @@
 )
 from xarray.backends.store import StoreBackendEntrypoint
 from xarray.core import indexing
+from xarray.core.parallelcompat import guess_chunkmanager
 from xarray.core.pycompat import integer_types
 from xarray.core.utils import (
     FrozenDict,
@@ -716,6 +717,8 @@ def open_zarr(
     decode_timedelta=None,
     use_cftime=None,
     zarr_version=None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ):
     """Load and decode a dataset from a Zarr store.
@@ -800,6 +803,15 @@ def open_zarr(
         The desired zarr spec version to target (currently 2 or 3). The default
         of None will attempt to determine the zarr version from ``store`` when
         possible, otherwise defaulting to 2.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce this datasets' arrays to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        Defaults to {'manager': 'dask'}, meaning additional kwargs will be passed eventually to
+        :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -817,12 +829,17 @@ def open_zarr(
     """
     from xarray.backends.api import open_dataset
 
+    if from_array_kwargs is None:
+        from_array_kwargs = {}
+
     if chunks == "auto":
         try:
-            import dask.array  # noqa
+            guess_chunkmanager(
+                chunked_array_type
+            )  # attempt to import that parallel backend
 
             chunks = {}
-        except ImportError:
+        except ValueError:
             chunks = None
 
     if kwargs:
@@ -851,6 +868,8 @@ def open_zarr(
         engine="zarr",
         chunks=chunks,
         drop_variables=drop_variables,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
         backend_kwargs=backend_kwargs,
         decode_timedelta=decode_timedelta,
         use_cftime=use_cftime,
diff --git a/xarray/coding/strings.py b/xarray/coding/strings.py
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -14,7 +14,7 @@
     unpack_for_encoding,
 )
 from xarray.core import indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.variable import Variable
 
 
@@ -134,10 +134,10 @@ def bytes_to_char(arr):
     if arr.dtype.kind != "S":
         raise ValueError("argument must have a fixed-width bytes dtype")
 
-    if is_duck_dask_array(arr):
-        import dask.array as da
+    if is_chunked_array(arr):
+        chunkmanager = get_chunked_array_type(arr)
 
-        return da.map_blocks(
+        return chunkmanager.map_blocks(
             _numpy_bytes_to_char,
             arr,
             dtype="S1",
@@ -169,8 +169,8 @@ def char_to_bytes(arr):
         # can't make an S0 dtype
         return np.zeros(arr.shape[:-1], dtype=np.string_)
 
-    if is_duck_dask_array(arr):
-        import dask.array as da
+    if is_chunked_array(arr):
+        chunkmanager = get_chunked_array_type(arr)
 
         if len(arr.chunks[-1]) > 1:
             raise ValueError(
@@ -179,7 +179,7 @@ def char_to_bytes(arr):
             )
 
         dtype = np.dtype("S" + str(arr.shape[-1]))
-        return da.map_blocks(
+        return chunkmanager.map_blocks(
             _numpy_char_to_bytes,
             arr,
             dtype=dtype,
diff --git a/xarray/coding/variables.py b/xarray/coding/variables.py
--- a/xarray/coding/variables.py
+++ b/xarray/coding/variables.py
@@ -10,7 +10,8 @@
 import pandas as pd
 
 from xarray.core import dtypes, duck_array_ops, indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.variable import Variable
 
 if TYPE_CHECKING:
@@ -57,7 +58,7 @@ class _ElementwiseFunctionArray(indexing.ExplicitlyIndexedNDArrayMixin):
     """
 
     def __init__(self, array, func: Callable, dtype: np.typing.DTypeLike):
-        assert not is_duck_dask_array(array)
+        assert not is_chunked_array(array)
         self.array = indexing.as_indexable(array)
         self.func = func
         self._dtype = dtype
@@ -158,10 +159,10 @@ def lazy_elemwise_func(array, func: Callable, dtype: np.typing.DTypeLike):
     -------
     Either a dask.array.Array or _ElementwiseFunctionArray.
     """
-    if is_duck_dask_array(array):
-        import dask.array as da
+    if is_chunked_array(array):
+        chunkmanager = get_chunked_array_type(array)
 
-        return da.map_blocks(func, array, dtype=dtype)
+        return chunkmanager.map_blocks(func, array, dtype=dtype)
     else:
         return _ElementwiseFunctionArray(array, func, dtype)
 
@@ -330,7 +331,7 @@ def encode(self, variable: Variable, name: T_Name = None) -> Variable:
 
         if "scale_factor" in encoding or "add_offset" in encoding:
             dtype = _choose_float_dtype(data.dtype, "add_offset" in encoding)
-            data = data.astype(dtype=dtype, copy=True)
+            data = duck_array_ops.astype(data, dtype=dtype, copy=True)
         if "add_offset" in encoding:
             data -= pop_to(encoding, attrs, "add_offset", name=name)
         if "scale_factor" in encoding:
@@ -377,7 +378,7 @@ def encode(self, variable: Variable, name: T_Name = None) -> Variable:
             if "_FillValue" in attrs:
                 new_fill = signed_dtype.type(attrs["_FillValue"])
                 attrs["_FillValue"] = new_fill
-            data = duck_array_ops.around(data).astype(signed_dtype)
+            data = duck_array_ops.astype(duck_array_ops.around(data), signed_dtype)
 
             return Variable(dims, data, attrs, encoding, fastpath=True)
         else:
diff --git a/xarray/core/common.py b/xarray/core/common.py
--- a/xarray/core/common.py
+++ b/xarray/core/common.py
@@ -13,8 +13,9 @@
 from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
 from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
 from xarray.core.options import OPTIONS, _get_keep_attrs
+from xarray.core.parallelcompat import get_chunked_array_type, guess_chunkmanager
 from xarray.core.pdcompat import _convert_base_to_offset
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.utils import (
     Frozen,
     either_dict_or_kwargs,
@@ -46,6 +47,7 @@
         DTypeLikeSave,
         ScalarOrArray,
         SideOptions,
+        T_Chunks,
         T_DataWithCoords,
         T_Variable,
     )
@@ -159,7 +161,7 @@ def __int__(self: Any) -> int:
     def __complex__(self: Any) -> complex:
         return complex(self.values)
 
-    def __array__(self: Any, dtype: DTypeLike = None) -> np.ndarray:
+    def __array__(self: Any, dtype: DTypeLike | None = None) -> np.ndarray:
         return np.asarray(self.values, dtype=dtype)
 
     def __repr__(self) -> str:
@@ -1396,28 +1398,52 @@ def __getitem__(self, value):
 
 @overload
 def full_like(
-    other: DataArray, fill_value: Any, dtype: DTypeLikeSave = None
+    other: DataArray,
+    fill_value: Any,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> DataArray:
     ...
 
 
 @overload
 def full_like(
-    other: Dataset, fill_value: Any, dtype: DTypeMaybeMapping = None
+    other: Dataset,
+    fill_value: Any,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset:
     ...
 
 
 @overload
 def full_like(
-    other: Variable, fill_value: Any, dtype: DTypeLikeSave = None
+    other: Variable,
+    fill_value: Any,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Variable:
     ...
 
 
 @overload
 def full_like(
-    other: Dataset | DataArray, fill_value: Any, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    fill_value: Any,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = {},
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
@@ -1426,7 +1452,11 @@ def full_like(
 def full_like(
     other: Dataset | DataArray | Variable,
     fill_value: Any,
-    dtype: DTypeMaybeMapping = None,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
@@ -1434,9 +1464,16 @@ def full_like(
 def full_like(
     other: Dataset | DataArray | Variable,
     fill_value: Any,
-    dtype: DTypeMaybeMapping = None,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
-    """Return a new object with the same shape and type as a given object.
+    """
+    Return a new object with the same shape and type as a given object.
+
+    Returned object will be chunked if if the given object is chunked, or if chunks or chunked_array_type are specified.
 
     Parameters
     ----------
@@ -1449,6 +1486,18 @@ def full_like(
     dtype : dtype or dict-like of dtype, optional
         dtype of the new array. If a dict-like, maps dtypes to
         variables. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1562,7 +1611,12 @@ def full_like(
 
         data_vars = {
             k: _full_like_variable(
-                v.variable, fill_value.get(k, dtypes.NA), dtype_.get(k, None)
+                v.variable,
+                fill_value.get(k, dtypes.NA),
+                dtype_.get(k, None),
+                chunks,
+                chunked_array_type,
+                from_array_kwargs,
             )
             for k, v in other.data_vars.items()
         }
@@ -1571,7 +1625,14 @@ def full_like(
         if isinstance(dtype, Mapping):
             raise ValueError("'dtype' cannot be dict-like when passing a DataArray")
         return DataArray(
-            _full_like_variable(other.variable, fill_value, dtype),
+            _full_like_variable(
+                other.variable,
+                fill_value,
+                dtype,
+                chunks,
+                chunked_array_type,
+                from_array_kwargs,
+            ),
             dims=other.dims,
             coords=other.coords,
             attrs=other.attrs,
@@ -1580,13 +1641,20 @@ def full_like(
     elif isinstance(other, Variable):
         if isinstance(dtype, Mapping):
             raise ValueError("'dtype' cannot be dict-like when passing a Variable")
-        return _full_like_variable(other, fill_value, dtype)
+        return _full_like_variable(
+            other, fill_value, dtype, chunks, chunked_array_type, from_array_kwargs
+        )
     else:
         raise TypeError("Expected DataArray, Dataset, or Variable")
 
 
 def _full_like_variable(
-    other: Variable, fill_value: Any, dtype: DTypeLike = None
+    other: Variable,
+    fill_value: Any,
+    dtype: DTypeLike | None = None,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Variable:
     """Inner function of full_like, where other must be a variable"""
     from xarray.core.variable import Variable
@@ -1594,13 +1662,28 @@ def _full_like_variable(
     if fill_value is dtypes.NA:
         fill_value = dtypes.get_fill_value(dtype if dtype is not None else other.dtype)
 
-    if is_duck_dask_array(other.data):
-        import dask.array
+    if (
+        is_chunked_array(other.data)
+        or chunked_array_type is not None
+        or chunks is not None
+    ):
+        if chunked_array_type is None:
+            chunkmanager = get_chunked_array_type(other.data)
+        else:
+            chunkmanager = guess_chunkmanager(chunked_array_type)
 
         if dtype is None:
             dtype = other.dtype
-        data = dask.array.full(
-            other.shape, fill_value, dtype=dtype, chunks=other.data.chunks
+
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
+        data = chunkmanager.array_api.full(
+            other.shape,
+            fill_value,
+            dtype=dtype,
+            chunks=chunks if chunks else other.data.chunks,
+            **from_array_kwargs,
         )
     else:
         data = np.full_like(other.data, fill_value, dtype=dtype)
@@ -1609,36 +1692,72 @@ def _full_like_variable(
 
 
 @overload
-def zeros_like(other: DataArray, dtype: DTypeLikeSave = None) -> DataArray:
+def zeros_like(
+    other: DataArray,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> DataArray:
     ...
 
 
 @overload
-def zeros_like(other: Dataset, dtype: DTypeMaybeMapping = None) -> Dataset:
+def zeros_like(
+    other: Dataset,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Dataset:
     ...
 
 
 @overload
-def zeros_like(other: Variable, dtype: DTypeLikeSave = None) -> Variable:
+def zeros_like(
+    other: Variable,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Variable:
     ...
 
 
 @overload
 def zeros_like(
-    other: Dataset | DataArray, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
 
 @overload
 def zeros_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
 
 def zeros_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     """Return a new object of zeros with the same shape and
     type as a given dataarray or dataset.
@@ -1649,6 +1768,18 @@ def zeros_like(
         The reference object. The output will have the same dimensions and coordinates as this object.
     dtype : dtype, optional
         dtype of the new array. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1692,40 +1823,83 @@ def zeros_like(
     full_like
 
     """
-    return full_like(other, 0, dtype)
+    return full_like(
+        other,
+        0,
+        dtype,
+        chunks=chunks,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
+    )
 
 
 @overload
-def ones_like(other: DataArray, dtype: DTypeLikeSave = None) -> DataArray:
+def ones_like(
+    other: DataArray,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> DataArray:
     ...
 
 
 @overload
-def ones_like(other: Dataset, dtype: DTypeMaybeMapping = None) -> Dataset:
+def ones_like(
+    other: Dataset,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Dataset:
     ...
 
 
 @overload
-def ones_like(other: Variable, dtype: DTypeLikeSave = None) -> Variable:
+def ones_like(
+    other: Variable,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Variable:
     ...
 
 
 @overload
 def ones_like(
-    other: Dataset | DataArray, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
 
 @overload
 def ones_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
 
 def ones_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     """Return a new object of ones with the same shape and
     type as a given dataarray or dataset.
@@ -1736,6 +1910,18 @@ def ones_like(
         The reference object. The output will have the same dimensions and coordinates as this object.
     dtype : dtype, optional
         dtype of the new array. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1771,7 +1957,14 @@ def ones_like(
     full_like
 
     """
-    return full_like(other, 1, dtype)
+    return full_like(
+        other,
+        1,
+        dtype,
+        chunks=chunks,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
+    )
 
 
 def get_chunksizes(
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -20,7 +20,8 @@
 from xarray.core.indexes import Index, filter_indexes_from_coords
 from xarray.core.merge import merge_attrs, merge_coordinates_without_align
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array, is_duck_dask_array
 from xarray.core.types import Dims, T_DataArray
 from xarray.core.utils import is_dict_like, is_scalar
 from xarray.core.variable import Variable
@@ -675,16 +676,18 @@ def apply_variable_ufunc(
         for arg, core_dims in zip(args, signature.input_core_dims)
     ]
 
-    if any(is_duck_dask_array(array) for array in input_data):
+    if any(is_chunked_array(array) for array in input_data):
         if dask == "forbidden":
             raise ValueError(
-                "apply_ufunc encountered a dask array on an "
-                "argument, but handling for dask arrays has not "
+                "apply_ufunc encountered a chunked array on an "
+                "argument, but handling for chunked arrays has not "
                 "been enabled. Either set the ``dask`` argument "
                 "or load your data into memory first with "
                 "``.load()`` or ``.compute()``"
             )
         elif dask == "parallelized":
+            chunkmanager = get_chunked_array_type(*input_data)
+
             numpy_func = func
 
             if dask_gufunc_kwargs is None:
@@ -697,7 +700,7 @@ def apply_variable_ufunc(
                 for n, (data, core_dims) in enumerate(
                     zip(input_data, signature.input_core_dims)
                 ):
-                    if is_duck_dask_array(data):
+                    if is_chunked_array(data):
                         # core dimensions cannot span multiple chunks
                         for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                             if len(data.chunks[axis]) != 1:
@@ -705,7 +708,7 @@ def apply_variable_ufunc(
                                     f"dimension {dim} on {n}th function argument to "
                                     "apply_ufunc with dask='parallelized' consists of "
                                     "multiple chunks, but is also a core dimension. To "
-                                    "fix, either rechunk into a single dask array chunk along "
+                                    "fix, either rechunk into a single array chunk along "
                                     f"this dimension, i.e., ``.chunk(dict({dim}=-1))``, or "
                                     "pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs`` "
                                     "but beware that this may significantly increase memory usage."
@@ -732,9 +735,7 @@ def apply_variable_ufunc(
                     )
 
             def func(*arrays):
-                import dask.array as da
-
-                res = da.apply_gufunc(
+                res = chunkmanager.apply_gufunc(
                     numpy_func,
                     signature.to_gufunc_string(exclude_dims),
                     *arrays,
@@ -749,8 +750,7 @@ def func(*arrays):
             pass
         else:
             raise ValueError(
-                "unknown setting for dask array handling in "
-                "apply_ufunc: {}".format(dask)
+                "unknown setting for chunked array handling in " f"apply_ufunc: {dask}"
             )
     else:
         if vectorize:
@@ -812,7 +812,7 @@ def func(*arrays):
 
 def apply_array_ufunc(func, *args, dask="forbidden"):
     """Apply a ndarray level function over ndarray objects."""
-    if any(is_duck_dask_array(arg) for arg in args):
+    if any(is_chunked_array(arg) for arg in args):
         if dask == "forbidden":
             raise ValueError(
                 "apply_ufunc encountered a dask array on an "
@@ -2013,7 +2013,7 @@ def to_floatable(x: DataArray) -> DataArray:
             )
         elif x.dtype.kind == "m":
             # timedeltas
-            return x.astype(float)
+            return duck_array_ops.astype(x, dtype=float)
         return x
 
     if isinstance(data, Dataset):
@@ -2061,12 +2061,11 @@ def _calc_idxminmax(
     # This will run argmin or argmax.
     indx = func(array, dim=dim, axis=None, keep_attrs=keep_attrs, skipna=skipna)
 
-    # Handle dask arrays.
-    if is_duck_dask_array(array.data):
-        import dask.array
-
+    # Handle chunked arrays (e.g. dask).
+    if is_chunked_array(array.data):
+        chunkmanager = get_chunked_array_type(array.data)
         chunks = dict(zip(array.dims, array.chunks))
-        dask_coord = dask.array.from_array(array[dim].data, chunks=chunks[dim])
+        dask_coord = chunkmanager.from_array(array[dim].data, chunks=chunks[dim])
         res = indx.copy(data=dask_coord[indx.data.ravel()].reshape(indx.shape))
         # we need to attach back the dim name
         res.name = dim
@@ -2153,16 +2152,14 @@ def unify_chunks(*objects: Dataset | DataArray) -> tuple[Dataset | DataArray, ..
     if not unify_chunks_args:
         return objects
 
-    # Run dask.array.core.unify_chunks
-    from dask.array.core import unify_chunks
-
-    _, dask_data = unify_chunks(*unify_chunks_args)
-    dask_data_iter = iter(dask_data)
+    chunkmanager = get_chunked_array_type(*[arg for arg in unify_chunks_args])
+    _, chunked_data = chunkmanager.unify_chunks(*unify_chunks_args)
+    chunked_data_iter = iter(chunked_data)
     out: list[Dataset | DataArray] = []
     for obj, ds in zip(objects, datasets):
         for k, v in ds._variables.items():
             if v.chunks is not None:
-                ds._variables[k] = v.copy(data=next(dask_data_iter))
+                ds._variables[k] = v.copy(data=next(chunked_data_iter))
         out.append(obj._from_temp_dataset(ds) if isinstance(obj, DataArray) else ds)
 
     return tuple(out)
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -1,9 +1,5 @@
 from __future__ import annotations
 
-from functools import partial
-
-from numpy.core.multiarray import normalize_axis_index  # type: ignore[attr-defined]
-
 from xarray.core import dtypes, nputils
 
 
@@ -96,36 +92,3 @@ def _fill_with_last_one(a, b):
         axis=axis,
         dtype=array.dtype,
     )
-
-
-def _first_last_wrapper(array, *, axis, op, keepdims):
-    return op(array, axis, keepdims=keepdims)
-
-
-def _first_or_last(darray, axis, op):
-    import dask.array
-
-    # This will raise the same error message seen for numpy
-    axis = normalize_axis_index(axis, darray.ndim)
-
-    wrapped_op = partial(_first_last_wrapper, op=op)
-    return dask.array.reduction(
-        darray,
-        chunk=wrapped_op,
-        aggregate=wrapped_op,
-        axis=axis,
-        dtype=darray.dtype,
-        keepdims=False,  # match numpy version
-    )
-
-
-def nanfirst(darray, axis):
-    from xarray.core.duck_array_ops import nanfirst
-
-    return _first_or_last(darray, axis, op=nanfirst)
-
-
-def nanlast(darray, axis):
-    from xarray.core.duck_array_ops import nanlast
-
-    return _first_or_last(darray, axis, op=nanlast)
diff --git a/xarray/core/daskmanager.py b/xarray/core/daskmanager.py
new file mode 100644
--- /dev/null
+++ b/xarray/core/daskmanager.py
@@ -0,0 +1,215 @@
+from __future__ import annotations
+
+from collections.abc import Iterable, Sequence
+from typing import TYPE_CHECKING, Any, Callable
+
+import numpy as np
+from packaging.version import Version
+
+from xarray.core.duck_array_ops import dask_available
+from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
+from xarray.core.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
+from xarray.core.pycompat import is_duck_dask_array
+
+if TYPE_CHECKING:
+    from xarray.core.types import DaskArray, T_Chunks, T_NormalizedChunks
+
+
+class DaskManager(ChunkManagerEntrypoint["DaskArray"]):
+    array_cls: type[DaskArray]
+    available: bool = dask_available
+
+    def __init__(self) -> None:
+        # TODO can we replace this with a class attribute instead?
+
+        from dask.array import Array
+
+        self.array_cls = Array
+
+    def is_chunked_array(self, data: Any) -> bool:
+        return is_duck_dask_array(data)
+
+    def chunks(self, data: DaskArray) -> T_NormalizedChunks:
+        return data.chunks
+
+    def normalize_chunks(
+        self,
+        chunks: T_Chunks | T_NormalizedChunks,
+        shape: tuple[int, ...] | None = None,
+        limit: int | None = None,
+        dtype: np.dtype | None = None,
+        previous_chunks: T_NormalizedChunks | None = None,
+    ) -> T_NormalizedChunks:
+        """Called by open_dataset"""
+        from dask.array.core import normalize_chunks
+
+        return normalize_chunks(
+            chunks,
+            shape=shape,
+            limit=limit,
+            dtype=dtype,
+            previous_chunks=previous_chunks,
+        )
+
+    def from_array(self, data: Any, chunks, **kwargs) -> DaskArray:
+        import dask.array as da
+
+        if isinstance(data, ImplicitToExplicitIndexingAdapter):
+            # lazily loaded backend array classes should use NumPy array operations.
+            kwargs["meta"] = np.ndarray
+
+        return da.from_array(
+            data,
+            chunks,
+            **kwargs,
+        )
+
+    def compute(self, *data: DaskArray, **kwargs) -> tuple[np.ndarray, ...]:
+        from dask.array import compute
+
+        return compute(*data, **kwargs)
+
+    @property
+    def array_api(self) -> Any:
+        from dask import array as da
+
+        return da
+
+    def reduction(
+        self,
+        arr: T_ChunkedArray,
+        func: Callable,
+        combine_func: Callable | None = None,
+        aggregate_func: Callable | None = None,
+        axis: int | Sequence[int] | None = None,
+        dtype: np.dtype | None = None,
+        keepdims: bool = False,
+    ) -> T_ChunkedArray:
+        from dask.array import reduction
+
+        return reduction(
+            arr,
+            chunk=func,
+            combine=combine_func,
+            aggregate=aggregate_func,
+            axis=axis,
+            dtype=dtype,
+            keepdims=keepdims,
+        )
+
+    def apply_gufunc(
+        self,
+        func: Callable,
+        signature: str,
+        *args: Any,
+        axes: Sequence[tuple[int, ...]] | None = None,
+        axis: int | None = None,
+        keepdims: bool = False,
+        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
+        output_sizes: dict[str, int] | None = None,
+        vectorize: bool | None = None,
+        allow_rechunk: bool = False,
+        meta: tuple[np.ndarray, ...] | None = None,
+        **kwargs,
+    ):
+        from dask.array.gufunc import apply_gufunc
+
+        return apply_gufunc(
+            func,
+            signature,
+            *args,
+            axes=axes,
+            axis=axis,
+            keepdims=keepdims,
+            output_dtypes=output_dtypes,
+            output_sizes=output_sizes,
+            vectorize=vectorize,
+            allow_rechunk=allow_rechunk,
+            meta=meta,
+            **kwargs,
+        )
+
+    def map_blocks(
+        self,
+        func: Callable,
+        *args: Any,
+        dtype: np.typing.DTypeLike | None = None,
+        chunks: tuple[int, ...] | None = None,
+        drop_axis: int | Sequence[int] | None = None,
+        new_axis: int | Sequence[int] | None = None,
+        **kwargs,
+    ):
+        import dask
+        from dask.array import map_blocks
+
+        if drop_axis is None and Version(dask.__version__) < Version("2022.9.1"):
+            # See https://github.com/pydata/xarray/pull/7019#discussion_r1196729489
+            # TODO remove once dask minimum version >= 2022.9.1
+            drop_axis = []
+
+        # pass through name, meta, token as kwargs
+        return map_blocks(
+            func,
+            *args,
+            dtype=dtype,
+            chunks=chunks,
+            drop_axis=drop_axis,
+            new_axis=new_axis,
+            **kwargs,
+        )
+
+    def blockwise(
+        self,
+        func: Callable,
+        out_ind: Iterable,
+        *args: Any,
+        # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
+        name: str | None = None,
+        token=None,
+        dtype: np.dtype | None = None,
+        adjust_chunks: dict[Any, Callable] | None = None,
+        new_axes: dict[Any, int] | None = None,
+        align_arrays: bool = True,
+        concatenate: bool | None = None,
+        meta=None,
+        **kwargs,
+    ):
+        from dask.array import blockwise
+
+        return blockwise(
+            func,
+            out_ind,
+            *args,
+            name=name,
+            token=token,
+            dtype=dtype,
+            adjust_chunks=adjust_chunks,
+            new_axes=new_axes,
+            align_arrays=align_arrays,
+            concatenate=concatenate,
+            meta=meta,
+            **kwargs,
+        )
+
+    def unify_chunks(
+        self,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
+        **kwargs,
+    ) -> tuple[dict[str, T_NormalizedChunks], list[DaskArray]]:
+        from dask.array.core import unify_chunks
+
+        return unify_chunks(*args, **kwargs)
+
+    def store(
+        self,
+        sources: DaskArray | Sequence[DaskArray],
+        targets: Any,
+        **kwargs,
+    ):
+        from dask.array import store
+
+        return store(
+            sources=sources,
+            targets=targets,
+            **kwargs,
+        )
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -77,6 +77,7 @@
     from xarray.backends import ZarrStore
     from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
     from xarray.core.groupby import DataArrayGroupBy
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.resample import DataArrayResample
     from xarray.core.rolling import DataArrayCoarsen, DataArrayRolling
     from xarray.core.types import (
@@ -1264,6 +1265,8 @@ def chunk(
         token: str | None = None,
         lock: bool = False,
         inline_array: bool = False,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: Any,
     ) -> T_DataArray:
         """Coerce this array's data into a dask arrays with the given chunks.
@@ -1285,12 +1288,21 @@ def chunk(
             Prefix for the name of the new dask array.
         token : str, optional
             Token uniquely identifying this array.
-        lock : optional
+        lock : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
-        inline_array: optional
+        inline_array: bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce the underlying data array to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided.
@@ -1328,6 +1340,8 @@ def chunk(
             token=token,
             lock=lock,
             inline_array=inline_array,
+            chunked_array_type=chunked_array_type,
+            from_array_kwargs=from_array_kwargs,
         )
         return self._from_temp_dataset(ds)
 
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -51,6 +51,7 @@
 )
 from xarray.core.computation import unify_chunks
 from xarray.core.coordinates import DatasetCoordinates, assert_coordinate_consistent
+from xarray.core.daskmanager import DaskManager
 from xarray.core.duck_array_ops import datetime_to_numeric
 from xarray.core.indexes import (
     Index,
@@ -73,7 +74,16 @@
 )
 from xarray.core.missing import get_clean_interp_index
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import array_type, is_duck_array, is_duck_dask_array
+from xarray.core.parallelcompat import (
+    get_chunked_array_type,
+    guess_chunkmanager,
+)
+from xarray.core.pycompat import (
+    array_type,
+    is_chunked_array,
+    is_duck_array,
+    is_duck_dask_array,
+)
 from xarray.core.types import QuantileMethods, T_Dataset
 from xarray.core.utils import (
     Default,
@@ -107,6 +117,7 @@
     from xarray.core.dataarray import DataArray
     from xarray.core.groupby import DatasetGroupBy
     from xarray.core.merge import CoercibleMapping
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.resample import DatasetResample
     from xarray.core.rolling import DatasetCoarsen, DatasetRolling
     from xarray.core.types import (
@@ -202,13 +213,11 @@ def _assert_empty(args: tuple, msg: str = "%s") -> None:
         raise ValueError(msg % args)
 
 
-def _get_chunk(var, chunks):
+def _get_chunk(var: Variable, chunks, chunkmanager: ChunkManagerEntrypoint):
     """
     Return map from each dim to chunk sizes, accounting for backend's preferred chunks.
     """
 
-    import dask.array as da
-
     if isinstance(var, IndexVariable):
         return {}
     dims = var.dims
@@ -225,7 +234,8 @@ def _get_chunk(var, chunks):
         chunks.get(dim, None) or preferred_chunk_sizes
         for dim, preferred_chunk_sizes in zip(dims, preferred_chunk_shape)
     )
-    chunk_shape = da.core.normalize_chunks(
+
+    chunk_shape = chunkmanager.normalize_chunks(
         chunk_shape, shape=shape, dtype=var.dtype, previous_chunks=preferred_chunk_shape
     )
 
@@ -242,7 +252,7 @@ def _get_chunk(var, chunks):
             # expresses the preferred chunks, the sequence sums to the size.
             preferred_stops = (
                 range(preferred_chunk_sizes, size, preferred_chunk_sizes)
-                if isinstance(preferred_chunk_sizes, Number)
+                if isinstance(preferred_chunk_sizes, int)
                 else itertools.accumulate(preferred_chunk_sizes[:-1])
             )
             # Gather any stop indices of the specified chunks that are not a stop index
@@ -253,7 +263,7 @@ def _get_chunk(var, chunks):
             )
             if breaks:
                 warnings.warn(
-                    "The specified Dask chunks separate the stored chunks along "
+                    "The specified chunks separate the stored chunks along "
                     f'dimension "{dim}" starting at index {min(breaks)}. This could '
                     "degrade performance. Instead, consider rechunking after loading."
                 )
@@ -270,18 +280,37 @@ def _maybe_chunk(
     name_prefix="xarray-",
     overwrite_encoded_chunks=False,
     inline_array=False,
+    chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+    from_array_kwargs=None,
 ):
-    from dask.base import tokenize
-
     if chunks is not None:
         chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
+
     if var.ndim:
-        # when rechunking by different amounts, make sure dask names change
-        # by provinding chunks as an input to tokenize.
-        # subtle bugs result otherwise. see GH3350
-        token2 = tokenize(name, token if token else var._data, chunks)
-        name2 = f"{name_prefix}{name}-{token2}"
-        var = var.chunk(chunks, name=name2, lock=lock, inline_array=inline_array)
+        chunked_array_type = guess_chunkmanager(
+            chunked_array_type
+        )  # coerce string to ChunkManagerEntrypoint type
+        if isinstance(chunked_array_type, DaskManager):
+            from dask.base import tokenize
+
+            # when rechunking by different amounts, make sure dask names change
+            # by providing chunks as an input to tokenize.
+            # subtle bugs result otherwise. see GH3350
+            token2 = tokenize(name, token if token else var._data, chunks)
+            name2 = f"{name_prefix}{name}-{token2}"
+
+            from_array_kwargs = utils.consolidate_dask_from_array_kwargs(
+                from_array_kwargs,
+                name=name2,
+                lock=lock,
+                inline_array=inline_array,
+            )
+
+        var = var.chunk(
+            chunks,
+            chunked_array_type=chunked_array_type,
+            from_array_kwargs=from_array_kwargs,
+        )
 
         if overwrite_encoded_chunks and var.chunks is not None:
             var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
@@ -743,13 +772,13 @@ def load(self: T_Dataset, **kwargs) -> T_Dataset:
         """
         # access .data to coerce everything to numpy or dask arrays
         lazy_data = {
-            k: v._data for k, v in self.variables.items() if is_duck_dask_array(v._data)
+            k: v._data for k, v in self.variables.items() if is_chunked_array(v._data)
         }
         if lazy_data:
-            import dask.array as da
+            chunkmanager = get_chunked_array_type(*lazy_data.values())
 
-            # evaluate all the dask arrays simultaneously
-            evaluated_data = da.compute(*lazy_data.values(), **kwargs)
+            # evaluate all the chunked arrays simultaneously
+            evaluated_data = chunkmanager.compute(*lazy_data.values(), **kwargs)
 
             for k, data in zip(lazy_data, evaluated_data):
                 self.variables[k].data = data
@@ -1575,7 +1604,7 @@ def _setitem_check(self, key, value):
                 val = np.array(val)
 
             # type conversion
-            new_value[name] = val.astype(var_k.dtype, copy=False)
+            new_value[name] = duck_array_ops.astype(val, dtype=var_k.dtype, copy=False)
 
         # check consistency of dimension sizes and dimension coordinates
         if isinstance(value, DataArray) or isinstance(value, Dataset):
@@ -1945,6 +1974,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> ZarrStore:
         ...
 
@@ -1966,6 +1996,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> Delayed:
         ...
 
@@ -1984,6 +2015,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> ZarrStore | Delayed:
         """Write dataset contents to a zarr group.
 
@@ -2072,6 +2104,10 @@ def to_zarr(
             The desired zarr spec version to target (currently 2 or 3). The
             default of None will attempt to determine the zarr version from
             ``store`` when possible, otherwise defaulting to 2.
+        chunkmanager_store_kwargs : dict, optional
+            Additional keyword arguments passed on to the `ChunkManager.store` method used to store
+            chunked arrays. For example for a dask array additional kwargs will be passed eventually to
+            :py:func:`dask.array.store()`. Experimental API that should not be relied upon.
 
         Returns
         -------
@@ -2117,6 +2153,7 @@ def to_zarr(
             region=region,
             safe_chunks=safe_chunks,
             zarr_version=zarr_version,
+            chunkmanager_store_kwargs=chunkmanager_store_kwargs,
         )
 
     def __repr__(self) -> str:
@@ -2205,6 +2242,8 @@ def chunk(
         token: str | None = None,
         lock: bool = False,
         inline_array: bool = False,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: None | int | str | tuple[int, ...],
     ) -> T_Dataset:
         """Coerce all arrays in this dataset into dask arrays with the given
@@ -2232,6 +2271,15 @@ def chunk(
         inline_array: bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce this datasets' arrays to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided
@@ -2266,8 +2314,22 @@ def chunk(
                 f"some chunks keys are not dimensions on this object: {bad_dims}"
             )
 
+        chunkmanager = guess_chunkmanager(chunked_array_type)
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
         variables = {
-            k: _maybe_chunk(k, v, chunks, token, lock, name_prefix)
+            k: _maybe_chunk(
+                k,
+                v,
+                chunks,
+                token,
+                lock,
+                name_prefix,
+                inline_array=inline_array,
+                chunked_array_type=chunkmanager,
+                from_array_kwargs=from_array_kwargs.copy(),
+            )
             for k, v in self.variables.items()
         }
         return self._replace(variables)
@@ -2305,7 +2367,7 @@ def _validate_indexers(
                 if v.dtype.kind in "US":
                     index = self._indexes[k].to_pandas_index()
                     if isinstance(index, pd.DatetimeIndex):
-                        v = v.astype("datetime64[ns]")
+                        v = duck_array_ops.astype(v, dtype="datetime64[ns]")
                     elif isinstance(index, CFTimeIndex):
                         v = _parse_array_of_cftime_strings(v, index.date_type)
 
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -9,6 +9,7 @@
 import datetime
 import inspect
 import warnings
+from functools import partial
 from importlib import import_module
 
 import numpy as np
@@ -29,10 +30,11 @@
     zeros_like,  # noqa
 )
 from numpy import concatenate as _concatenate
+from numpy.core.multiarray import normalize_axis_index  # type: ignore[attr-defined]
 from numpy.lib.stride_tricks import sliding_window_view  # noqa
 
 from xarray.core import dask_array_ops, dtypes, nputils
-from xarray.core.nputils import nanfirst, nanlast
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.pycompat import array_type, is_duck_dask_array
 from xarray.core.utils import is_duck_array, module_available
 
@@ -640,10 +642,10 @@ def first(values, axis, skipna=None):
     """Return the first non-NA elements in this array along the given axis"""
     if (skipna or skipna is None) and values.dtype.kind not in "iSU":
         # only bother for dtypes that can hold NaN
-        if is_duck_dask_array(values):
-            return dask_array_ops.nanfirst(values, axis)
+        if is_chunked_array(values):
+            return chunked_nanfirst(values, axis)
         else:
-            return nanfirst(values, axis)
+            return nputils.nanfirst(values, axis)
     return take(values, 0, axis=axis)
 
 
@@ -651,10 +653,10 @@ def last(values, axis, skipna=None):
     """Return the last non-NA elements in this array along the given axis"""
     if (skipna or skipna is None) and values.dtype.kind not in "iSU":
         # only bother for dtypes that can hold NaN
-        if is_duck_dask_array(values):
-            return dask_array_ops.nanlast(values, axis)
+        if is_chunked_array(values):
+            return chunked_nanlast(values, axis)
         else:
-            return nanlast(values, axis)
+            return nputils.nanlast(values, axis)
     return take(values, -1, axis=axis)
 
 
@@ -673,3 +675,32 @@ def push(array, n, axis):
         return dask_array_ops.push(array, n, axis)
     else:
         return push(array, n, axis)
+
+
+def _first_last_wrapper(array, *, axis, op, keepdims):
+    return op(array, axis, keepdims=keepdims)
+
+
+def _chunked_first_or_last(darray, axis, op):
+    chunkmanager = get_chunked_array_type(darray)
+
+    # This will raise the same error message seen for numpy
+    axis = normalize_axis_index(axis, darray.ndim)
+
+    wrapped_op = partial(_first_last_wrapper, op=op)
+    return chunkmanager.reduction(
+        darray,
+        func=wrapped_op,
+        aggregate_func=wrapped_op,
+        axis=axis,
+        dtype=darray.dtype,
+        keepdims=False,  # match numpy version
+    )
+
+
+def chunked_nanfirst(darray, axis):
+    return _chunked_first_or_last(darray, axis, op=nputils.nanfirst)
+
+
+def chunked_nanlast(darray, axis):
+    return _chunked_first_or_last(darray, axis, op=nputils.nanlast)
diff --git a/xarray/core/indexing.py b/xarray/core/indexing.py
--- a/xarray/core/indexing.py
+++ b/xarray/core/indexing.py
@@ -17,6 +17,7 @@
 from xarray.core import duck_array_ops
 from xarray.core.nputils import NumpyVIndexAdapter
 from xarray.core.options import OPTIONS
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.pycompat import (
     array_type,
     integer_types,
@@ -1142,16 +1143,15 @@ def _arrayize_vectorized_indexer(indexer, shape):
     return VectorizedIndexer(tuple(new_key))
 
 
-def _dask_array_with_chunks_hint(array, chunks):
-    """Create a dask array using the chunks hint for dimensions of size > 1."""
-    import dask.array as da
+def _chunked_array_with_chunks_hint(array, chunks, chunkmanager):
+    """Create a chunked array using the chunks hint for dimensions of size > 1."""
 
     if len(chunks) < array.ndim:
         raise ValueError("not enough chunks in hint")
     new_chunks = []
     for chunk, size in zip(chunks, array.shape):
         new_chunks.append(chunk if size > 1 else (1,))
-    return da.from_array(array, new_chunks)
+    return chunkmanager.from_array(array, new_chunks)
 
 
 def _logical_any(args):
@@ -1165,8 +1165,11 @@ def _masked_result_drop_slice(key, data=None):
     new_keys = []
     for k in key:
         if isinstance(k, np.ndarray):
-            if is_duck_dask_array(data):
-                new_keys.append(_dask_array_with_chunks_hint(k, chunks_hint))
+            if is_chunked_array(data):
+                chunkmanager = get_chunked_array_type(data)
+                new_keys.append(
+                    _chunked_array_with_chunks_hint(k, chunks_hint, chunkmanager)
+                )
             elif isinstance(data, array_type("sparse")):
                 import sparse
 
diff --git a/xarray/core/missing.py b/xarray/core/missing.py
--- a/xarray/core/missing.py
+++ b/xarray/core/missing.py
@@ -15,7 +15,7 @@
 from xarray.core.computation import apply_ufunc
 from xarray.core.duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.types import Interp1dOptions, InterpOptions
 from xarray.core.utils import OrderedSet, is_scalar
 from xarray.core.variable import Variable, broadcast_variables
@@ -693,8 +693,8 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
     else:
         func, kwargs = _get_interpolator_nd(method, **kwargs)
 
-    if is_duck_dask_array(var):
-        import dask.array as da
+    if is_chunked_array(var):
+        chunkmanager = get_chunked_array_type(var)
 
         ndim = var.ndim
         nconst = ndim - len(x)
@@ -716,7 +716,7 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
             *new_x_arginds,
         )
 
-        _, rechunked = da.unify_chunks(*args)
+        _, rechunked = chunkmanager.unify_chunks(*args)
 
         args = tuple(elem for pair in zip(rechunked, args[1::2]) for elem in pair)
 
@@ -741,8 +741,8 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
 
         meta = var._meta
 
-        return da.blockwise(
-            _dask_aware_interpnd,
+        return chunkmanager.blockwise(
+            _chunked_aware_interpnd,
             out_ind,
             *args,
             interp_func=func,
@@ -785,8 +785,8 @@ def _interpnd(var, x, new_x, func, kwargs):
     return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)
 
 
-def _dask_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
-    """Wrapper for `_interpnd` through `blockwise`
+def _chunked_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
+    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.
 
     The first half arrays in `coords` are original coordinates,
     the other half are destination coordinates
diff --git a/xarray/core/nanops.py b/xarray/core/nanops.py
--- a/xarray/core/nanops.py
+++ b/xarray/core/nanops.py
@@ -6,6 +6,7 @@
 
 from xarray.core import dtypes, nputils, utils
 from xarray.core.duck_array_ops import (
+    astype,
     count,
     fillna,
     isnull,
@@ -22,7 +23,7 @@ def _maybe_null_out(result, axis, mask, min_count=1):
     if axis is not None and getattr(result, "ndim", False):
         null_mask = (np.take(mask.shape, axis).prod() - mask.sum(axis) - min_count) < 0
         dtype, fill_value = dtypes.maybe_promote(result.dtype)
-        result = where(null_mask, fill_value, result.astype(dtype))
+        result = where(null_mask, fill_value, astype(result, dtype))
 
     elif getattr(result, "dtype", None) not in dtypes.NAT_TYPES:
         null_mask = mask.size - mask.sum()
@@ -140,7 +141,7 @@ def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
     value_mean = _nanmean_ddof_object(
         ddof=0, value=value, axis=axis, keepdims=True, **kwargs
     )
-    squared = (value.astype(value_mean.dtype) - value_mean) ** 2
+    squared = (astype(value, value_mean.dtype) - value_mean) ** 2
     return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)
 
 
diff --git a/xarray/core/parallelcompat.py b/xarray/core/parallelcompat.py
new file mode 100644
--- /dev/null
+++ b/xarray/core/parallelcompat.py
@@ -0,0 +1,280 @@
+"""
+The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
+It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
+but for now it is just a private experiment.
+"""
+from __future__ import annotations
+
+import functools
+import sys
+from abc import ABC, abstractmethod
+from collections.abc import Iterable, Sequence
+from importlib.metadata import EntryPoint, entry_points
+from typing import (
+    TYPE_CHECKING,
+    Any,
+    Callable,
+    Generic,
+    TypeVar,
+)
+
+import numpy as np
+
+from xarray.core.pycompat import is_chunked_array
+
+T_ChunkedArray = TypeVar("T_ChunkedArray")
+
+if TYPE_CHECKING:
+    from xarray.core.types import T_Chunks, T_NormalizedChunks
+
+
+@functools.lru_cache(maxsize=1)
+def list_chunkmanagers() -> dict[str, ChunkManagerEntrypoint]:
+    """
+    Return a dictionary of available chunk managers and their ChunkManagerEntrypoint objects.
+
+    Notes
+    -----
+    # New selection mechanism introduced with Python 3.10. See GH6514.
+    """
+    if sys.version_info >= (3, 10):
+        entrypoints = entry_points(group="xarray.chunkmanagers")
+    else:
+        entrypoints = entry_points().get("xarray.chunkmanagers", ())
+
+    return load_chunkmanagers(entrypoints)
+
+
+def load_chunkmanagers(
+    entrypoints: Sequence[EntryPoint],
+) -> dict[str, ChunkManagerEntrypoint]:
+    """Load entrypoints and instantiate chunkmanagers only once."""
+
+    loaded_entrypoints = {
+        entrypoint.name: entrypoint.load() for entrypoint in entrypoints
+    }
+
+    available_chunkmanagers = {
+        name: chunkmanager()
+        for name, chunkmanager in loaded_entrypoints.items()
+        if chunkmanager.available
+    }
+    return available_chunkmanagers
+
+
+def guess_chunkmanager(
+    manager: str | ChunkManagerEntrypoint | None,
+) -> ChunkManagerEntrypoint:
+    """
+    Get namespace of chunk-handling methods, guessing from what's available.
+
+    If the name of a specific ChunkManager is given (e.g. "dask"), then use that.
+    Else use whatever is installed, defaulting to dask if there are multiple options.
+    """
+
+    chunkmanagers = list_chunkmanagers()
+
+    if manager is None:
+        if len(chunkmanagers) == 1:
+            # use the only option available
+            manager = next(iter(chunkmanagers.keys()))
+        else:
+            # default to trying to use dask
+            manager = "dask"
+
+    if isinstance(manager, str):
+        if manager not in chunkmanagers:
+            raise ValueError(
+                f"unrecognized chunk manager {manager} - must be one of: {list(chunkmanagers)}"
+            )
+
+        return chunkmanagers[manager]
+    elif isinstance(manager, ChunkManagerEntrypoint):
+        # already a valid ChunkManager so just pass through
+        return manager
+    else:
+        raise TypeError(
+            f"manager must be a string or instance of ChunkManagerEntrypoint, but received type {type(manager)}"
+        )
+
+
+def get_chunked_array_type(*args) -> ChunkManagerEntrypoint:
+    """
+    Detects which parallel backend should be used for given set of arrays.
+
+    Also checks that all arrays are of same chunking type (i.e. not a mix of cubed and dask).
+    """
+
+    # TODO this list is probably redundant with something inside xarray.apply_ufunc
+    ALLOWED_NON_CHUNKED_TYPES = {int, float, np.ndarray}
+
+    chunked_arrays = [
+        a
+        for a in args
+        if is_chunked_array(a) and type(a) not in ALLOWED_NON_CHUNKED_TYPES
+    ]
+
+    # Asserts all arrays are the same type (or numpy etc.)
+    chunked_array_types = {type(a) for a in chunked_arrays}
+    if len(chunked_array_types) > 1:
+        raise TypeError(
+            f"Mixing chunked array types is not supported, but received multiple types: {chunked_array_types}"
+        )
+    elif len(chunked_array_types) == 0:
+        raise TypeError("Expected a chunked array but none were found")
+
+    # iterate over defined chunk managers, seeing if each recognises this array type
+    chunked_arr = chunked_arrays[0]
+    chunkmanagers = list_chunkmanagers()
+    selected = [
+        chunkmanager
+        for chunkmanager in chunkmanagers.values()
+        if chunkmanager.is_chunked_array(chunked_arr)
+    ]
+    if not selected:
+        raise TypeError(
+            f"Could not find a Chunk Manager which recognises type {type(chunked_arr)}"
+        )
+    elif len(selected) >= 2:
+        raise TypeError(f"Multiple ChunkManagers recognise type {type(chunked_arr)}")
+    else:
+        return selected[0]
+
+
+class ChunkManagerEntrypoint(ABC, Generic[T_ChunkedArray]):
+    """
+    Adapter between a particular parallel computing framework and xarray.
+
+    Attributes
+    ----------
+    array_cls
+        Type of the array class this parallel computing framework provides.
+
+        Parallel frameworks need to provide an array class that supports the array API standard.
+        Used for type checking.
+    """
+
+    array_cls: type[T_ChunkedArray]
+    available: bool = True
+
+    @abstractmethod
+    def __init__(self) -> None:
+        raise NotImplementedError()
+
+    def is_chunked_array(self, data: Any) -> bool:
+        return isinstance(data, self.array_cls)
+
+    @abstractmethod
+    def chunks(self, data: T_ChunkedArray) -> T_NormalizedChunks:
+        raise NotImplementedError()
+
+    @abstractmethod
+    def normalize_chunks(
+        self,
+        chunks: T_Chunks | T_NormalizedChunks,
+        shape: tuple[int, ...] | None = None,
+        limit: int | None = None,
+        dtype: np.dtype | None = None,
+        previous_chunks: T_NormalizedChunks | None = None,
+    ) -> T_NormalizedChunks:
+        """Called by open_dataset"""
+        raise NotImplementedError()
+
+    @abstractmethod
+    def from_array(
+        self, data: np.ndarray, chunks: T_Chunks, **kwargs
+    ) -> T_ChunkedArray:
+        """Called when .chunk is called on an xarray object that is not already chunked."""
+        raise NotImplementedError()
+
+    def rechunk(
+        self,
+        data: T_ChunkedArray,
+        chunks: T_NormalizedChunks | tuple[int, ...] | T_Chunks,
+        **kwargs,
+    ) -> T_ChunkedArray:
+        """Called when .chunk is called on an xarray object that is already chunked."""
+        return data.rechunk(chunks, **kwargs)  # type: ignore[attr-defined]
+
+    @abstractmethod
+    def compute(self, *data: T_ChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
+        """Used anytime something needs to computed, including multiple arrays at once."""
+        raise NotImplementedError()
+
+    @property
+    def array_api(self) -> Any:
+        """Return the array_api namespace following the python array API standard."""
+        raise NotImplementedError()
+
+    def reduction(
+        self,
+        arr: T_ChunkedArray,
+        func: Callable,
+        combine_func: Callable | None = None,
+        aggregate_func: Callable | None = None,
+        axis: int | Sequence[int] | None = None,
+        dtype: np.dtype | None = None,
+        keepdims: bool = False,
+    ) -> T_ChunkedArray:
+        """Used in some reductions like nanfirst, which is used by groupby.first"""
+        raise NotImplementedError()
+
+    @abstractmethod
+    def apply_gufunc(
+        self,
+        func: Callable,
+        signature: str,
+        *args: Any,
+        axes: Sequence[tuple[int, ...]] | None = None,
+        keepdims: bool = False,
+        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
+        vectorize: bool | None = None,
+        **kwargs,
+    ):
+        """
+        Called inside xarray.apply_ufunc, so must be supplied for vast majority of xarray computations to be supported.
+        """
+        raise NotImplementedError()
+
+    def map_blocks(
+        self,
+        func: Callable,
+        *args: Any,
+        dtype: np.typing.DTypeLike | None = None,
+        chunks: tuple[int, ...] | None = None,
+        drop_axis: int | Sequence[int] | None = None,
+        new_axis: int | Sequence[int] | None = None,
+        **kwargs,
+    ):
+        """Called in elementwise operations, but notably not called in xarray.map_blocks."""
+        raise NotImplementedError()
+
+    def blockwise(
+        self,
+        func: Callable,
+        out_ind: Iterable,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
+        adjust_chunks: dict[Any, Callable] | None = None,
+        new_axes: dict[Any, int] | None = None,
+        align_arrays: bool = True,
+        **kwargs,
+    ):
+        """Called by some niche functions in xarray."""
+        raise NotImplementedError()
+
+    def unify_chunks(
+        self,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
+        **kwargs,
+    ) -> tuple[dict[str, T_NormalizedChunks], list[T_ChunkedArray]]:
+        """Called by xr.unify_chunks."""
+        raise NotImplementedError()
+
+    def store(
+        self,
+        sources: T_ChunkedArray | Sequence[T_ChunkedArray],
+        targets: Any,
+        **kwargs: dict[str, Any],
+    ):
+        """Used when writing to any backend."""
+        raise NotImplementedError()
diff --git a/xarray/core/pycompat.py b/xarray/core/pycompat.py
--- a/xarray/core/pycompat.py
+++ b/xarray/core/pycompat.py
@@ -12,7 +12,7 @@
 integer_types = (int, np.integer)
 
 if TYPE_CHECKING:
-    ModType = Literal["dask", "pint", "cupy", "sparse"]
+    ModType = Literal["dask", "pint", "cupy", "sparse", "cubed"]
     DuckArrayTypes = tuple[type[Any], ...]  # TODO: improve this? maybe Generic
 
 
@@ -30,7 +30,7 @@ class DuckArrayModule:
     available: bool
 
     def __init__(self, mod: ModType) -> None:
-        duck_array_module: ModuleType | None = None
+        duck_array_module: ModuleType | None
         duck_array_version: Version
         duck_array_type: DuckArrayTypes
         try:
@@ -45,6 +45,8 @@ def __init__(self, mod: ModType) -> None:
                 duck_array_type = (duck_array_module.ndarray,)
             elif mod == "sparse":
                 duck_array_type = (duck_array_module.SparseArray,)
+            elif mod == "cubed":
+                duck_array_type = (duck_array_module.Array,)
             else:
                 raise NotImplementedError
 
@@ -81,5 +83,9 @@ def is_duck_dask_array(x):
     return is_duck_array(x) and is_dask_collection(x)
 
 
+def is_chunked_array(x) -> bool:
+    return is_duck_dask_array(x) or (is_duck_array(x) and hasattr(x, "chunks"))
+
+
 def is_0d_dask_array(x):
     return is_duck_dask_array(x) and is_scalar(x)
diff --git a/xarray/core/rolling.py b/xarray/core/rolling.py
--- a/xarray/core/rolling.py
+++ b/xarray/core/rolling.py
@@ -158,9 +158,9 @@ def method(self, keep_attrs=None, **kwargs):
         return method
 
     def _mean(self, keep_attrs, **kwargs):
-        result = self.sum(keep_attrs=False, **kwargs) / self.count(
-            keep_attrs=False
-        ).astype(self.obj.dtype, copy=False)
+        result = self.sum(keep_attrs=False, **kwargs) / duck_array_ops.astype(
+            self.count(keep_attrs=False), dtype=self.obj.dtype, copy=False
+        )
         if keep_attrs:
             result.attrs = self.obj.attrs
         return result
diff --git a/xarray/core/types.py b/xarray/core/types.py
--- a/xarray/core/types.py
+++ b/xarray/core/types.py
@@ -33,6 +33,16 @@
     except ImportError:
         DaskArray = np.ndarray  # type: ignore
 
+    try:
+        from cubed import Array as CubedArray
+    except ImportError:
+        CubedArray = np.ndarray
+
+    try:
+        from zarr.core import Array as ZarrArray
+    except ImportError:
+        ZarrArray = np.ndarray
+
     # TODO: Turn on when https://github.com/python/mypy/issues/11871 is fixed.
     # Can be uncommented if using pyright though.
     # import sys
@@ -105,6 +115,9 @@
 Dims = Union[str, Iterable[Hashable], "ellipsis", None]
 OrderedDims = Union[str, Sequence[Union[Hashable, "ellipsis"]], "ellipsis", None]
 
+T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
+T_NormalizedChunks = tuple[tuple[int, ...], ...]
+
 ErrorOptions = Literal["raise", "ignore"]
 ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]
 
diff --git a/xarray/core/utils.py b/xarray/core/utils.py
--- a/xarray/core/utils.py
+++ b/xarray/core/utils.py
@@ -1202,3 +1202,66 @@ def emit_user_level_warning(message, category=None):
     """Emit a warning at the user level by inspecting the stack trace."""
     stacklevel = find_stack_level()
     warnings.warn(message, category=category, stacklevel=stacklevel)
+
+
+def consolidate_dask_from_array_kwargs(
+    from_array_kwargs: dict,
+    name: str | None = None,
+    lock: bool | None = None,
+    inline_array: bool | None = None,
+) -> dict:
+    """
+    Merge dask-specific kwargs with arbitrary from_array_kwargs dict.
+
+    Temporary function, to be deleted once explicitly passing dask-specific kwargs to .chunk() is deprecated.
+    """
+
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="name",
+        passed_kwarg_value=name,
+        default=None,
+        err_msg_dict_name="from_array_kwargs",
+    )
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="lock",
+        passed_kwarg_value=lock,
+        default=False,
+        err_msg_dict_name="from_array_kwargs",
+    )
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="inline_array",
+        passed_kwarg_value=inline_array,
+        default=False,
+        err_msg_dict_name="from_array_kwargs",
+    )
+
+    return from_array_kwargs
+
+
+def _resolve_doubly_passed_kwarg(
+    kwargs_dict: dict,
+    kwarg_name: str,
+    passed_kwarg_value: str | bool | None,
+    default: bool | None,
+    err_msg_dict_name: str,
+) -> dict:
+    # if in kwargs_dict but not passed explicitly then just pass kwargs_dict through unaltered
+    if kwarg_name in kwargs_dict and passed_kwarg_value is None:
+        pass
+    # if passed explicitly but not in kwargs_dict then use that
+    elif kwarg_name not in kwargs_dict and passed_kwarg_value is not None:
+        kwargs_dict[kwarg_name] = passed_kwarg_value
+    # if in neither then use default
+    elif kwarg_name not in kwargs_dict and passed_kwarg_value is None:
+        kwargs_dict[kwarg_name] = default
+    # if in both then raise
+    else:
+        raise ValueError(
+            f"argument {kwarg_name} cannot be passed both as a keyword argument and within "
+            f"the {err_msg_dict_name} dictionary"
+        )
+
+    return kwargs_dict
diff --git a/xarray/core/variable.py b/xarray/core/variable.py
--- a/xarray/core/variable.py
+++ b/xarray/core/variable.py
@@ -26,10 +26,15 @@
     as_indexable,
 )
 from xarray.core.options import OPTIONS, _get_keep_attrs
+from xarray.core.parallelcompat import (
+    get_chunked_array_type,
+    guess_chunkmanager,
+)
 from xarray.core.pycompat import (
     array_type,
     integer_types,
     is_0d_dask_array,
+    is_chunked_array,
     is_duck_dask_array,
 )
 from xarray.core.utils import (
@@ -54,6 +59,7 @@
 BASIC_INDEXING_TYPES = integer_types + (slice,)
 
 if TYPE_CHECKING:
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.types import (
         Dims,
         ErrorOptionsWithWarn,
@@ -194,10 +200,10 @@ def _as_nanosecond_precision(data):
             nanosecond_precision_dtype = pd.DatetimeTZDtype("ns", dtype.tz)
         else:
             nanosecond_precision_dtype = "datetime64[ns]"
-        return data.astype(nanosecond_precision_dtype)
+        return duck_array_ops.astype(data, nanosecond_precision_dtype)
     elif dtype.kind == "m" and dtype != np.dtype("timedelta64[ns]"):
         utils.emit_user_level_warning(NON_NANOSECOND_WARNING.format(case="timedelta"))
-        return data.astype("timedelta64[ns]")
+        return duck_array_ops.astype(data, "timedelta64[ns]")
     else:
         return data
 
@@ -368,7 +374,7 @@ def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
             self.encoding = encoding
 
     @property
-    def dtype(self):
+    def dtype(self) -> np.dtype:
         """
         Data-type of the array’s elements.
 
@@ -380,7 +386,7 @@ def dtype(self):
         return self._data.dtype
 
     @property
-    def shape(self):
+    def shape(self) -> tuple[int, ...]:
         """
         Tuple of array dimensions.
 
@@ -533,8 +539,10 @@ def load(self, **kwargs):
         --------
         dask.array.compute
         """
-        if is_duck_dask_array(self._data):
-            self._data = as_compatible_data(self._data.compute(**kwargs))
+        if is_chunked_array(self._data):
+            chunkmanager = get_chunked_array_type(self._data)
+            loaded_data, *_ = chunkmanager.compute(self._data, **kwargs)
+            self._data = as_compatible_data(loaded_data)
         elif isinstance(self._data, indexing.ExplicitlyIndexed):
             self._data = self._data.get_duck_array()
         elif not is_duck_array(self._data):
@@ -1166,8 +1174,10 @@ def chunk(
             | Mapping[Any, None | int | tuple[int, ...]]
         ) = {},
         name: str | None = None,
-        lock: bool = False,
-        inline_array: bool = False,
+        lock: bool | None = None,
+        inline_array: bool | None = None,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: Any,
     ) -> Variable:
         """Coerce this array's data into a dask array with the given chunks.
@@ -1188,12 +1198,21 @@ def chunk(
         name : str, optional
             Used to generate the name for this array in the internal dask
             graph. Does not need not be unique.
-        lock : optional
+        lock : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
-        inline_array: optional
+        inline_array : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce this datasets' arrays to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntrypoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided.
@@ -1209,7 +1228,6 @@ def chunk(
         xarray.unify_chunks
         dask.array.from_array
         """
-        import dask.array as da
 
         if chunks is None:
             warnings.warn(
@@ -1220,6 +1238,8 @@ def chunk(
             chunks = {}
 
         if isinstance(chunks, (float, str, int, tuple, list)):
+            # TODO we shouldn't assume here that other chunkmanagers can handle these types
+            # TODO should we call normalize_chunks here?
             pass  # dask.array.from_array can handle these directly
         else:
             chunks = either_dict_or_kwargs(chunks, chunks_kwargs, "chunk")
@@ -1227,9 +1247,22 @@ def chunk(
         if utils.is_dict_like(chunks):
             chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}
 
+        chunkmanager = guess_chunkmanager(chunked_array_type)
+
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
+        # TODO deprecate passing these dask-specific arguments explicitly. In future just pass everything via from_array_kwargs
+        _from_array_kwargs = utils.consolidate_dask_from_array_kwargs(
+            from_array_kwargs,
+            name=name,
+            lock=lock,
+            inline_array=inline_array,
+        )
+
         data = self._data
-        if is_duck_dask_array(data):
-            data = data.rechunk(chunks)
+        if chunkmanager.is_chunked_array(data):
+            data = chunkmanager.rechunk(data, chunks)  # type: ignore[arg-type]
         else:
             if isinstance(data, indexing.ExplicitlyIndexed):
                 # Unambiguously handle array storage backends (like NetCDF4 and h5py)
@@ -1244,17 +1277,13 @@ def chunk(
                     data, indexing.OuterIndexer
                 )
 
-                # All of our lazily loaded backend array classes should use NumPy
-                # array operations.
-                kwargs = {"meta": np.ndarray}
-            else:
-                kwargs = {}
-
             if utils.is_dict_like(chunks):
-                chunks = tuple(chunks.get(n, s) for n, s in enumerate(self.shape))
+                chunks = tuple(chunks.get(n, s) for n, s in enumerate(data.shape))
 
-            data = da.from_array(
-                data, chunks, name=name, lock=lock, inline_array=inline_array, **kwargs
+            data = chunkmanager.from_array(
+                data,
+                chunks,  # type: ignore[arg-type]
+                **_from_array_kwargs,
             )
 
         return self._replace(data=data)
@@ -1266,7 +1295,8 @@ def to_numpy(self) -> np.ndarray:
 
         # TODO first attempt to call .to_numpy() once some libraries implement it
         if hasattr(data, "chunks"):
-            data = data.compute()
+            chunkmanager = get_chunked_array_type(data)
+            data, *_ = chunkmanager.compute(data)
         if isinstance(data, array_type("cupy")):
             data = data.get()
         # pint has to be imported dynamically as pint imports xarray
@@ -2903,7 +2933,15 @@ def values(self, values):
             f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
         )
 
-    def chunk(self, chunks={}, name=None, lock=False, inline_array=False):
+    def chunk(
+        self,
+        chunks={},
+        name=None,
+        lock=False,
+        inline_array=False,
+        chunked_array_type=None,
+        from_array_kwargs=None,
+    ):
         # Dummy - do not chunk. This method is invoked e.g. by Dataset.chunk()
         return self.copy(deep=False)
 
diff --git a/xarray/core/weighted.py b/xarray/core/weighted.py
--- a/xarray/core/weighted.py
+++ b/xarray/core/weighted.py
@@ -238,7 +238,10 @@ def _sum_of_weights(self, da: DataArray, dim: Dims = None) -> DataArray:
         # (and not 2); GH4074
         if self.weights.dtype == bool:
             sum_of_weights = self._reduce(
-                mask, self.weights.astype(int), dim=dim, skipna=False
+                mask,
+                duck_array_ops.astype(self.weights, dtype=int),
+                dim=dim,
+                skipna=False,
             )
         else:
             sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/backends/api.py | 9 | 9 | 24 | 9 | 13305
| xarray/backends/api.py | 23 | 23 | 24 | 9 | 13305
| xarray/backends/api.py | 41 | 41 | 24 | 9 | 13305
| xarray/backends/api.py | 51 | 51 | 24 | 9 | 13305
| xarray/backends/api.py | 300 | 308 | - | 9 | -
| xarray/backends/api.py | 319 | 319 | - | 9 | -
| xarray/backends/api.py | 331 | 331 | - | 9 | -
| xarray/backends/api.py | 349 | 349 | - | 9 | -
| xarray/backends/api.py | 376 | 376 | - | 9 | -
| xarray/backends/api.py | 468 | 468 | - | 9 | -
| xarray/backends/api.py | 511 | 511 | - | 9 | -
| xarray/backends/api.py | 539 | 539 | - | 9 | -
| xarray/backends/api.py | 549 | 550 | - | 9 | -
| xarray/backends/api.py | 561 | 561 | - | 9 | -
| xarray/backends/api.py | 655 | 655 | - | 9 | -
| xarray/backends/api.py | 698 | 698 | - | 9 | -
| xarray/backends/api.py | 729 | 729 | - | 9 | -
| xarray/backends/api.py | 739 | 739 | - | 9 | -
| xarray/backends/api.py | 1493 | 1493 | - | 9 | -
| xarray/backends/api.py | 1515 | 1515 | - | 9 | -
| xarray/backends/api.py | 1534 | 1534 | - | 9 | -
| xarray/backends/api.py | 1655 | 1655 | - | 9 | -
| xarray/backends/common.py | 14 | 14 | - | - | -
| xarray/backends/common.py | 156 | 156 | - | - | -
| xarray/backends/common.py | 166 | 174 | - | - | -
| xarray/backends/plugins.py | 149 | 149 | - | - | -
| xarray/backends/zarr.py | 22 | 22 | - | - | -
| xarray/backends/zarr.py | 719 | 719 | - | - | -
| xarray/backends/zarr.py | 803 | 803 | - | - | -
| xarray/backends/zarr.py | 820 | 823 | - | - | -
| xarray/backends/zarr.py | 854 | 854 | - | - | -
| xarray/coding/strings.py | 17 | 17 | - | - | -
| xarray/coding/strings.py | 137 | 140 | - | - | -
| xarray/coding/strings.py | 172 | 173 | - | - | -
| xarray/coding/strings.py | 182 | 182 | - | - | -
| xarray/coding/variables.py | 13 | 13 | - | - | -
| xarray/coding/variables.py | 60 | 60 | - | - | -
| xarray/coding/variables.py | 161 | 164 | - | - | -
| xarray/coding/variables.py | 333 | 333 | - | - | -
| xarray/coding/variables.py | 380 | 380 | - | - | -
| xarray/core/common.py | 16 | 16 | 6 | 4 | 4254
| xarray/core/common.py | 49 | 49 | 6 | 4 | 4254
| xarray/core/common.py | 162 | 162 | - | 4 | -
| xarray/core/common.py | 1399 | 1420 | 11 | 4 | 6113
| xarray/core/common.py | 1429 | 1429 | 11 | 4 | 6113
| xarray/core/common.py | 1437 | 1439 | - | 4 | -
| xarray/core/common.py | 1452 | 1452 | - | 4 | -
| xarray/core/common.py | 1565 | 1565 | - | 4 | -
| xarray/core/common.py | 1574 | 1574 | - | 4 | -
| xarray/core/common.py | 1583 | 1589 | - | 4 | -
| xarray/core/common.py | 1597 | 1603 | - | 4 | -
| xarray/core/common.py | 1612 | 1641 | - | 4 | -
| xarray/core/common.py | 1652 | 1652 | - | 4 | -
| xarray/core/common.py | 1695 | 1728 | - | 4 | -
| xarray/core/common.py | 1739 | 1739 | - | 4 | -
| xarray/core/common.py | 1774 | 1774 | - | 4 | -
| xarray/core/computation.py | 23 | 23 | - | 8 | -
| xarray/core/computation.py | 678 | 682 | 18 | 8 | 10372
| xarray/core/computation.py | 700 | 700 | 18 | 8 | 10372
| xarray/core/computation.py | 708 | 708 | 18 | 8 | 10372
| xarray/core/computation.py | 735 | 737 | 18 | 8 | 10372
| xarray/core/computation.py | 752 | 753 | 18 | 8 | 10372
| xarray/core/computation.py | 815 | 815 | - | 8 | -
| xarray/core/computation.py | 2016 | 2016 | - | 8 | -
| xarray/core/computation.py | 2064 | 2069 | - | 8 | -
| xarray/core/computation.py | 2156 | 2165 | - | 8 | -
| xarray/core/dask_array_ops.py | 3 | 6 | - | 7 | -
| xarray/core/dask_array_ops.py | 99 | 131 | - | 7 | -
| xarray/core/daskmanager.py | 0 | 0 | - | - | -
| xarray/core/dataarray.py | 80 | 80 | 4 | 3 | 3139
| xarray/core/dataarray.py | 1267 | 1267 | - | 3 | -
| xarray/core/dataarray.py | 1288 | 1291 | - | 3 | -
| xarray/core/dataarray.py | 1331 | 1331 | - | 3 | -
| xarray/core/dataset.py | 54 | 54 | 26 | 11 | 13995
| xarray/core/dataset.py | 76 | 76 | 26 | 11 | 13995
| xarray/core/dataset.py | 110 | 110 | - | 11 | -
| xarray/core/dataset.py | 205 | 211 | - | 11 | -
| xarray/core/dataset.py | 228 | 228 | - | 11 | -
| xarray/core/dataset.py | 245 | 245 | - | 11 | -
| xarray/core/dataset.py | 256 | 256 | - | 11 | -
| xarray/core/dataset.py | 273 | 283 | - | 11 | -
| xarray/core/dataset.py | 746 | 752 | - | 11 | -
| xarray/core/dataset.py | 1578 | 1578 | - | 11 | -
| xarray/core/dataset.py | 1948 | 1948 | - | 11 | -
| xarray/core/dataset.py | 1969 | 1969 | - | 11 | -
| xarray/core/dataset.py | 1987 | 1987 | - | 11 | -
| xarray/core/dataset.py | 2075 | 2075 | - | 11 | -
| xarray/core/dataset.py | 2120 | 2120 | - | 11 | -
| xarray/core/dataset.py | 2208 | 2208 | - | 11 | -
| xarray/core/dataset.py | 2235 | 2235 | - | 11 | -
| xarray/core/dataset.py | 2269 | 2269 | - | 11 | -
| xarray/core/dataset.py | 2308 | 2308 | - | 11 | -
| xarray/core/duck_array_ops.py | 12 | 12 | 12 | 6 | 6390
| xarray/core/duck_array_ops.py | 32 | 32 | 12 | 6 | 6390
| xarray/core/duck_array_ops.py | 643 | 646 | 9 | 6 | 5652
| xarray/core/duck_array_ops.py | 654 | 657 | 9 | 6 | 5652
| xarray/core/duck_array_ops.py | 676 | 676 | 9 | 6 | 5652
| xarray/core/indexing.py | 20 | 20 | - | 10 | -
| xarray/core/indexing.py | 1145 | 1154 | 25 | 10 | 13421
| xarray/core/indexing.py | 1168 | 1169 | - | 10 | -
| xarray/core/missing.py | 18 | 18 | - | - | -
| xarray/core/missing.py | 696 | 697 | - | - | -
| xarray/core/missing.py | 719 | 719 | - | - | -
| xarray/core/missing.py | 744 | 745 | - | - | -
| xarray/core/missing.py | 788 | 789 | - | - | -
| xarray/core/nanops.py | 9 | 9 | - | - | -
| xarray/core/nanops.py | 25 | 25 | - | - | -
| xarray/core/nanops.py | 143 | 143 | - | - | -
| xarray/core/parallelcompat.py | 0 | 0 | - | - | -
| xarray/core/pycompat.py | 15 | 15 | - | - | -
| xarray/core/pycompat.py | 33 | 33 | - | - | -
| xarray/core/pycompat.py | 48 | 48 | - | - | -
| xarray/core/pycompat.py | 84 | 84 | - | - | -
| xarray/core/rolling.py | 161 | 163 | - | - | -
| xarray/core/types.py | 36 | 36 | 2 | 2 | 1610
| xarray/core/types.py | 108 | 108 | 5 | 2 | 3876
| xarray/core/utils.py | 1205 | 1205 | - | - | -
| xarray/core/variable.py | 29 | 29 | - | - | -
| xarray/core/variable.py | 57 | 57 | - | - | -
| xarray/core/variable.py | 197 | 200 | - | - | -
| xarray/core/variable.py | 371 | 371 | - | - | -
| xarray/core/variable.py | 383 | 383 | - | - | -
| xarray/core/variable.py | 536 | 537 | - | - | -
| xarray/core/variable.py | 1169 | 1170 | - | - | -
| xarray/core/variable.py | 1191 | 1194 | - | - | -
| xarray/core/variable.py | 1212 | 1212 | - | - | -
| xarray/core/variable.py | 1223 | 1223 | - | - | -
| xarray/core/variable.py | 1230 | 1231 | - | - | -
| xarray/core/variable.py | 1247 | 1257 | - | - | -
| xarray/core/variable.py | 1269 | 1269 | - | - | -
| xarray/core/variable.py | 2906 | 2906 | - | - | -
| xarray/core/weighted.py | 241 | 241 | - | - | -


## Problem Statement

```
Alternative parallel execution frameworks in xarray
### Is your feature request related to a problem?

Since early on the project xarray has supported wrapping `dask.array` objects in a first-class manner. However recent work on flexible array wrapping has made it possible to wrap all sorts of array types (and with #6804 we should support wrapping any array that conforms to the [array API standard](https://data-apis.org/array-api/latest/index.html)).

Currently though the only way to parallelize array operations with xarray "automatically" is to use dask. (You could use [xarray-beam](https://github.com/google/xarray-beam) or other options too but they don't "automatically" generate the computation for you like dask does.)

When dask is the only type of parallel framework exposing an array-like API then there is no need for flexibility, but now we have nascent projects like [cubed](https://github.com/tomwhite/cubed) to consider too. @tomwhite 

### Describe the solution you'd like

Refactor the internals so that dask is one option among many, and that any newer options can plug in in an extensible way.

In particular cubed deliberately uses the same API as `dask.array`, exposing:
1) the methods needed to conform to the array API standard
2) a `.chunk` and `.compute` method, which we could dispatch to
3) dask-like functions to create computation graphs including [`blockwise`](https://github.com/tomwhite/cubed/blob/400dc9adcf21c8b468fce9f24e8d4b8cb9ef2f11/cubed/core/ops.py#L43), [`map_blocks`](https://github.com/tomwhite/cubed/blob/400dc9adcf21c8b468fce9f24e8d4b8cb9ef2f11/cubed/core/ops.py#L221), and [`rechunk`](https://github.com/tomwhite/cubed/blob/main/cubed/primitive/rechunk.py)

I would like to see xarray able to wrap any array-like object which offers this set of methods / functions, and call the corresponding version of that method for the correct library (i.e. dask vs cubed) automatically.

That way users could try different parallel execution frameworks simply via a switch like 
\`\`\`python
ds.chunk(**chunk_pattern, manager="dask")
\`\`\`
and see which one works best for their particular problem.

### Describe alternatives you've considered

If we leave it the way it is now then xarray will not be truly flexible in this respect.

Any library can wrap (or subclass if they are really brave) xarray objects to provide parallelism but that's not the same level of flexibility.

### Additional context

[cubed repo](https://github.com/tomwhite/cubed)

[PR](https://github.com/pydata/xarray/pull/6804) about making xarray able to wrap objects conforming to the new [array API standard](https://data-apis.org/array-api/latest/index.html)

cc @shoyer @rabernat @dcherian @keewis 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/parallel.py | 477 | 556| 792 | 792 | 5126 | 
| **-> 2 <-** | **2 xarray/core/types.py** | 1 | 97| 818 | 1610 | 6821 | 
| 3 | 2 xarray/core/parallel.py | 299 | 379| 778 | 2388 | 6821 | 
| **-> 4 <-** | **3 xarray/core/dataarray.py** | 1 | 102| 751 | 3139 | 70436 | 
| **-> 5 <-** | **3 xarray/core/types.py** | 99 | 179| 737 | 3876 | 70436 | 
| **-> 6 <-** | **4 xarray/core/common.py** | 1 | 59| 378 | 4254 | 85853 | 
| 7 | 5 xarray/__init__.py | 1 | 112| 687 | 4941 | 86540 | 
| 8 | 5 xarray/core/parallel.py | 1 | 30| 187 | 5128 | 86540 | 
| **-> 9 <-** | **6 xarray/core/duck_array_ops.py** | 613 | 676| 524 | 5652 | 92255 | 
| 10 | 6 xarray/core/parallel.py | 558 | 583| 272 | 5924 | 92255 | 
| **-> 11 <-** | **6 xarray/core/common.py** | 1397 | 1431| 189 | 6113 | 92255 | 
| **-> 12 <-** | **6 xarray/core/duck_array_ops.py** | 1 | 46| 277 | 6390 | 92255 | 
| 13 | **7 xarray/core/dask_array_ops.py** | 101 | 132| 207 | 6597 | 93272 | 
| 14 | 7 xarray/core/parallel.py | 381 | 421| 461 | 7058 | 93272 | 
| 15 | 7 xarray/core/parallel.py | 147 | 249| 1298 | 8356 | 93272 | 
| 16 | **8 xarray/core/computation.py** | 1088 | 1163| 789 | 9145 | 112032 | 
| 17 | 8 xarray/core/parallel.py | 251 | 297| 423 | 9568 | 112032 | 
| **-> 18 <-** | **8 xarray/core/computation.py** | 678 | 761| 804 | 10372 | 112032 | 
| 19 | **8 xarray/core/dataarray.py** | 1018 | 1119| 817 | 11189 | 112032 | 
| 20 | **8 xarray/core/duck_array_ops.py** | 49 | 68| 130 | 11319 | 112032 | 
| 21 | 8 xarray/core/parallel.py | 423 | 475| 510 | 11829 | 112032 | 
| 22 | **8 xarray/core/dask_array_ops.py** | 62 | 98| 311 | 12140 | 112032 | 
| 23 | **8 xarray/core/computation.py** | 1707 | 1770| 613 | 12753 | 112032 | 
| **-> 24 <-** | **9 xarray/backends/api.py** | 1 | 68| 552 | 13305 | 126772 | 
| **-> 25 <-** | **10 xarray/core/indexing.py** | 1145 | 1158| 116 | 13421 | 140006 | 


## Missing Patch Files

 * 1: xarray/backends/api.py
 * 2: xarray/backends/common.py
 * 3: xarray/backends/plugins.py
 * 4: xarray/backends/zarr.py
 * 5: xarray/coding/strings.py
 * 6: xarray/coding/variables.py
 * 7: xarray/core/common.py
 * 8: xarray/core/computation.py
 * 9: xarray/core/dask_array_ops.py
 * 10: xarray/core/daskmanager.py
 * 11: xarray/core/dataarray.py
 * 12: xarray/core/dataset.py
 * 13: xarray/core/duck_array_ops.py
 * 14: xarray/core/indexing.py
 * 15: xarray/core/missing.py
 * 16: xarray/core/nanops.py
 * 17: xarray/core/parallelcompat.py
 * 18: xarray/core/pycompat.py
 * 19: xarray/core/rolling.py
 * 20: xarray/core/types.py
 * 21: xarray/core/utils.py
 * 22: xarray/core/variable.py
 * 23: xarray/core/weighted.py

### Hint

```
This sounds great! We should finish up https://github.com/pydata/xarray/pull/4972 to make it easier to test.
Another parallel framework would be [Ramba](https://github.com/Python-for-HPC/ramba) 

cc @DrTodd13
Sounds good to me. The challenge will be defining a parallel computing API that works across all these projects, with their slightly different models.
at SciPy i learned of [fugue](https://github.com/fugue-project/fugue) which tries to provide a unified API for distributed DataFrames on top of Spark and Dask. it could be a great source of inspiration. 
Thanks for opening this @TomNicholas 

> The challenge will be defining a parallel computing API that works across all these projects, with their slightly different models.

Agreed. I feel like there's already an implicit set of "chunked array" methods that xarray expects from Dask that could be formalised a bit and exposed as an integration point.
```

## Patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -6,7 +6,16 @@
 from glob import glob
 from io import BytesIO
 from numbers import Number
-from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Union, cast, overload
+from typing import (
+    TYPE_CHECKING,
+    Any,
+    Callable,
+    Final,
+    Literal,
+    Union,
+    cast,
+    overload,
+)
 
 import numpy as np
 
@@ -20,9 +29,11 @@
     _nested_combine,
     combine_by_coords,
 )
+from xarray.core.daskmanager import DaskManager
 from xarray.core.dataarray import DataArray
 from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
 from xarray.core.indexes import Index
+from xarray.core.parallelcompat import guess_chunkmanager
 from xarray.core.utils import is_remote_uri
 
 if TYPE_CHECKING:
@@ -38,6 +49,7 @@
         CompatOptions,
         JoinOptions,
         NestedSequence,
+        T_Chunks,
     )
 
     T_NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
@@ -48,7 +60,6 @@
         str,  # no nice typing support for custom backends
         None,
     ]
-    T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
     T_NetcdfTypes = Literal[
         "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
     ]
@@ -297,17 +308,27 @@ def _chunk_ds(
     chunks,
     overwrite_encoded_chunks,
     inline_array,
+    chunked_array_type,
+    from_array_kwargs,
     **extra_tokens,
 ):
-    from dask.base import tokenize
+    chunkmanager = guess_chunkmanager(chunked_array_type)
+
+    # TODO refactor to move this dask-specific logic inside the DaskManager class
+    if isinstance(chunkmanager, DaskManager):
+        from dask.base import tokenize
 
-    mtime = _get_mtime(filename_or_obj)
-    token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
-    name_prefix = f"open_dataset-{token}"
+        mtime = _get_mtime(filename_or_obj)
+        token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
+        name_prefix = "open_dataset-"
+    else:
+        # not used
+        token = (None,)
+        name_prefix = None
 
     variables = {}
     for name, var in backend_ds.variables.items():
-        var_chunks = _get_chunk(var, chunks)
+        var_chunks = _get_chunk(var, chunks, chunkmanager)
         variables[name] = _maybe_chunk(
             name,
             var,
@@ -316,6 +337,8 @@ def _chunk_ds(
             name_prefix=name_prefix,
             token=token,
             inline_array=inline_array,
+            chunked_array_type=chunkmanager,
+            from_array_kwargs=from_array_kwargs.copy(),
         )
     return backend_ds._replace(variables)
 
@@ -328,6 +351,8 @@ def _dataset_from_backend_dataset(
     cache,
     overwrite_encoded_chunks,
     inline_array,
+    chunked_array_type,
+    from_array_kwargs,
     **extra_tokens,
 ):
     if not isinstance(chunks, (int, dict)) and chunks not in {None, "auto"}:
@@ -346,6 +371,8 @@ def _dataset_from_backend_dataset(
             chunks,
             overwrite_encoded_chunks,
             inline_array,
+            chunked_array_type,
+            from_array_kwargs,
             **extra_tokens,
         )
 
@@ -373,6 +400,8 @@ def open_dataset(
     decode_coords: Literal["coordinates", "all"] | bool | None = None,
     drop_variables: str | Iterable[str] | None = None,
     inline_array: bool = False,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     backend_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ) -> Dataset:
@@ -465,6 +494,15 @@ def open_dataset(
         itself, and each chunk refers to that task by its key. With
         ``inline_array=True``, Dask will instead inline the array directly
         in the values of the task graph. See :py:func:`dask.array.from_array`.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce this datasets' arrays to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
     backend_kwargs: dict
         Additional keyword arguments passed on to the engine open function,
         equivalent to `**kwargs`.
@@ -508,6 +546,9 @@ def open_dataset(
     if engine is None:
         engine = plugins.guess_engine(filename_or_obj)
 
+    if from_array_kwargs is None:
+        from_array_kwargs = {}
+
     backend = plugins.get_backend(engine)
 
     decoders = _resolve_decoders_kwargs(
@@ -536,6 +577,8 @@ def open_dataset(
         cache,
         overwrite_encoded_chunks,
         inline_array,
+        chunked_array_type,
+        from_array_kwargs,
         drop_variables=drop_variables,
         **decoders,
         **kwargs,
@@ -546,8 +589,8 @@ def open_dataset(
 def open_dataarray(
     filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
     *,
-    engine: T_Engine = None,
-    chunks: T_Chunks = None,
+    engine: T_Engine | None = None,
+    chunks: T_Chunks | None = None,
     cache: bool | None = None,
     decode_cf: bool | None = None,
     mask_and_scale: bool | None = None,
@@ -558,6 +601,8 @@ def open_dataarray(
     decode_coords: Literal["coordinates", "all"] | bool | None = None,
     drop_variables: str | Iterable[str] | None = None,
     inline_array: bool = False,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     backend_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ) -> DataArray:
@@ -652,6 +697,15 @@ def open_dataarray(
         itself, and each chunk refers to that task by its key. With
         ``inline_array=True``, Dask will instead inline the array directly
         in the values of the task graph. See :py:func:`dask.array.from_array`.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
     backend_kwargs: dict
         Additional keyword arguments passed on to the engine open function,
         equivalent to `**kwargs`.
@@ -695,6 +749,8 @@ def open_dataarray(
         cache=cache,
         drop_variables=drop_variables,
         inline_array=inline_array,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
         backend_kwargs=backend_kwargs,
         use_cftime=use_cftime,
         decode_timedelta=decode_timedelta,
@@ -726,7 +782,7 @@ def open_dataarray(
 
 def open_mfdataset(
     paths: str | NestedSequence[str | os.PathLike],
-    chunks: T_Chunks = None,
+    chunks: T_Chunks | None = None,
     concat_dim: str
     | DataArray
     | Index
@@ -736,7 +792,7 @@ def open_mfdataset(
     | None = None,
     compat: CompatOptions = "no_conflicts",
     preprocess: Callable[[Dataset], Dataset] | None = None,
-    engine: T_Engine = None,
+    engine: T_Engine | None = None,
     data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
     coords="different",
     combine: Literal["by_coords", "nested"] = "by_coords",
@@ -1490,6 +1546,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> backends.ZarrStore:
     ...
 
@@ -1512,6 +1569,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> Delayed:
     ...
 
@@ -1531,6 +1589,7 @@ def to_zarr(
     safe_chunks: bool = True,
     storage_options: dict[str, str] | None = None,
     zarr_version: int | None = None,
+    chunkmanager_store_kwargs: dict[str, Any] | None = None,
 ) -> backends.ZarrStore | Delayed:
     """This function creates an appropriate datastore for writing a dataset to
     a zarr ztore
@@ -1652,7 +1711,9 @@ def to_zarr(
     writer = ArrayWriter()
     # TODO: figure out how to properly handle unlimited_dims
     dump_to_store(dataset, zstore, writer, encoding=encoding)
-    writes = writer.sync(compute=compute)
+    writes = writer.sync(
+        compute=compute, chunkmanager_store_kwargs=chunkmanager_store_kwargs
+    )
 
     if compute:
         _finalize_store(writes, zstore)
diff --git a/xarray/backends/common.py b/xarray/backends/common.py
--- a/xarray/backends/common.py
+++ b/xarray/backends/common.py
@@ -11,7 +11,8 @@
 
 from xarray.conventions import cf_encoder
 from xarray.core import indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
 
 if TYPE_CHECKING:
@@ -153,7 +154,7 @@ def __init__(self, lock=None):
         self.lock = lock
 
     def add(self, source, target, region=None):
-        if is_duck_dask_array(source):
+        if is_chunked_array(source):
             self.sources.append(source)
             self.targets.append(target)
             self.regions.append(region)
@@ -163,21 +164,25 @@ def add(self, source, target, region=None):
             else:
                 target[...] = source
 
-    def sync(self, compute=True):
+    def sync(self, compute=True, chunkmanager_store_kwargs=None):
         if self.sources:
-            import dask.array as da
+            chunkmanager = get_chunked_array_type(*self.sources)
 
             # TODO: consider wrapping targets with dask.delayed, if this makes
             # for any discernible difference in perforance, e.g.,
             # targets = [dask.delayed(t) for t in self.targets]
 
-            delayed_store = da.store(
+            if chunkmanager_store_kwargs is None:
+                chunkmanager_store_kwargs = {}
+
+            delayed_store = chunkmanager.store(
                 self.sources,
                 self.targets,
                 lock=self.lock,
                 compute=compute,
                 flush=True,
                 regions=self.regions,
+                **chunkmanager_store_kwargs,
             )
             self.sources = []
             self.targets = []
diff --git a/xarray/backends/plugins.py b/xarray/backends/plugins.py
--- a/xarray/backends/plugins.py
+++ b/xarray/backends/plugins.py
@@ -146,7 +146,7 @@ def refresh_engines() -> None:
 
 def guess_engine(
     store_spec: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
-):
+) -> str | type[BackendEntrypoint]:
     engines = list_engines()
 
     for engine, backend in engines.items():
diff --git a/xarray/backends/zarr.py b/xarray/backends/zarr.py
--- a/xarray/backends/zarr.py
+++ b/xarray/backends/zarr.py
@@ -19,6 +19,7 @@
 )
 from xarray.backends.store import StoreBackendEntrypoint
 from xarray.core import indexing
+from xarray.core.parallelcompat import guess_chunkmanager
 from xarray.core.pycompat import integer_types
 from xarray.core.utils import (
     FrozenDict,
@@ -716,6 +717,8 @@ def open_zarr(
     decode_timedelta=None,
     use_cftime=None,
     zarr_version=None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
     **kwargs,
 ):
     """Load and decode a dataset from a Zarr store.
@@ -800,6 +803,15 @@ def open_zarr(
         The desired zarr spec version to target (currently 2 or 3). The default
         of None will attempt to determine the zarr version from ``store`` when
         possible, otherwise defaulting to 2.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce this datasets' arrays to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        Defaults to {'manager': 'dask'}, meaning additional kwargs will be passed eventually to
+        :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -817,12 +829,17 @@ def open_zarr(
     """
     from xarray.backends.api import open_dataset
 
+    if from_array_kwargs is None:
+        from_array_kwargs = {}
+
     if chunks == "auto":
         try:
-            import dask.array  # noqa
+            guess_chunkmanager(
+                chunked_array_type
+            )  # attempt to import that parallel backend
 
             chunks = {}
-        except ImportError:
+        except ValueError:
             chunks = None
 
     if kwargs:
@@ -851,6 +868,8 @@ def open_zarr(
         engine="zarr",
         chunks=chunks,
         drop_variables=drop_variables,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
         backend_kwargs=backend_kwargs,
         decode_timedelta=decode_timedelta,
         use_cftime=use_cftime,
diff --git a/xarray/coding/strings.py b/xarray/coding/strings.py
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -14,7 +14,7 @@
     unpack_for_encoding,
 )
 from xarray.core import indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.variable import Variable
 
 
@@ -134,10 +134,10 @@ def bytes_to_char(arr):
     if arr.dtype.kind != "S":
         raise ValueError("argument must have a fixed-width bytes dtype")
 
-    if is_duck_dask_array(arr):
-        import dask.array as da
+    if is_chunked_array(arr):
+        chunkmanager = get_chunked_array_type(arr)
 
-        return da.map_blocks(
+        return chunkmanager.map_blocks(
             _numpy_bytes_to_char,
             arr,
             dtype="S1",
@@ -169,8 +169,8 @@ def char_to_bytes(arr):
         # can't make an S0 dtype
         return np.zeros(arr.shape[:-1], dtype=np.string_)
 
-    if is_duck_dask_array(arr):
-        import dask.array as da
+    if is_chunked_array(arr):
+        chunkmanager = get_chunked_array_type(arr)
 
         if len(arr.chunks[-1]) > 1:
             raise ValueError(
@@ -179,7 +179,7 @@ def char_to_bytes(arr):
             )
 
         dtype = np.dtype("S" + str(arr.shape[-1]))
-        return da.map_blocks(
+        return chunkmanager.map_blocks(
             _numpy_char_to_bytes,
             arr,
             dtype=dtype,
diff --git a/xarray/coding/variables.py b/xarray/coding/variables.py
--- a/xarray/coding/variables.py
+++ b/xarray/coding/variables.py
@@ -10,7 +10,8 @@
 import pandas as pd
 
 from xarray.core import dtypes, duck_array_ops, indexing
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.variable import Variable
 
 if TYPE_CHECKING:
@@ -57,7 +58,7 @@ class _ElementwiseFunctionArray(indexing.ExplicitlyIndexedNDArrayMixin):
     """
 
     def __init__(self, array, func: Callable, dtype: np.typing.DTypeLike):
-        assert not is_duck_dask_array(array)
+        assert not is_chunked_array(array)
         self.array = indexing.as_indexable(array)
         self.func = func
         self._dtype = dtype
@@ -158,10 +159,10 @@ def lazy_elemwise_func(array, func: Callable, dtype: np.typing.DTypeLike):
     -------
     Either a dask.array.Array or _ElementwiseFunctionArray.
     """
-    if is_duck_dask_array(array):
-        import dask.array as da
+    if is_chunked_array(array):
+        chunkmanager = get_chunked_array_type(array)
 
-        return da.map_blocks(func, array, dtype=dtype)
+        return chunkmanager.map_blocks(func, array, dtype=dtype)
     else:
         return _ElementwiseFunctionArray(array, func, dtype)
 
@@ -330,7 +331,7 @@ def encode(self, variable: Variable, name: T_Name = None) -> Variable:
 
         if "scale_factor" in encoding or "add_offset" in encoding:
             dtype = _choose_float_dtype(data.dtype, "add_offset" in encoding)
-            data = data.astype(dtype=dtype, copy=True)
+            data = duck_array_ops.astype(data, dtype=dtype, copy=True)
         if "add_offset" in encoding:
             data -= pop_to(encoding, attrs, "add_offset", name=name)
         if "scale_factor" in encoding:
@@ -377,7 +378,7 @@ def encode(self, variable: Variable, name: T_Name = None) -> Variable:
             if "_FillValue" in attrs:
                 new_fill = signed_dtype.type(attrs["_FillValue"])
                 attrs["_FillValue"] = new_fill
-            data = duck_array_ops.around(data).astype(signed_dtype)
+            data = duck_array_ops.astype(duck_array_ops.around(data), signed_dtype)
 
             return Variable(dims, data, attrs, encoding, fastpath=True)
         else:
diff --git a/xarray/core/common.py b/xarray/core/common.py
--- a/xarray/core/common.py
+++ b/xarray/core/common.py
@@ -13,8 +13,9 @@
 from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
 from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
 from xarray.core.options import OPTIONS, _get_keep_attrs
+from xarray.core.parallelcompat import get_chunked_array_type, guess_chunkmanager
 from xarray.core.pdcompat import _convert_base_to_offset
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.pycompat import is_chunked_array
 from xarray.core.utils import (
     Frozen,
     either_dict_or_kwargs,
@@ -46,6 +47,7 @@
         DTypeLikeSave,
         ScalarOrArray,
         SideOptions,
+        T_Chunks,
         T_DataWithCoords,
         T_Variable,
     )
@@ -159,7 +161,7 @@ def __int__(self: Any) -> int:
     def __complex__(self: Any) -> complex:
         return complex(self.values)
 
-    def __array__(self: Any, dtype: DTypeLike = None) -> np.ndarray:
+    def __array__(self: Any, dtype: DTypeLike | None = None) -> np.ndarray:
         return np.asarray(self.values, dtype=dtype)
 
     def __repr__(self) -> str:
@@ -1396,28 +1398,52 @@ def __getitem__(self, value):
 
 @overload
 def full_like(
-    other: DataArray, fill_value: Any, dtype: DTypeLikeSave = None
+    other: DataArray,
+    fill_value: Any,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> DataArray:
     ...
 
 
 @overload
 def full_like(
-    other: Dataset, fill_value: Any, dtype: DTypeMaybeMapping = None
+    other: Dataset,
+    fill_value: Any,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset:
     ...
 
 
 @overload
 def full_like(
-    other: Variable, fill_value: Any, dtype: DTypeLikeSave = None
+    other: Variable,
+    fill_value: Any,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Variable:
     ...
 
 
 @overload
 def full_like(
-    other: Dataset | DataArray, fill_value: Any, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    fill_value: Any,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = {},
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
@@ -1426,7 +1452,11 @@ def full_like(
 def full_like(
     other: Dataset | DataArray | Variable,
     fill_value: Any,
-    dtype: DTypeMaybeMapping = None,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
@@ -1434,9 +1464,16 @@ def full_like(
 def full_like(
     other: Dataset | DataArray | Variable,
     fill_value: Any,
-    dtype: DTypeMaybeMapping = None,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
-    """Return a new object with the same shape and type as a given object.
+    """
+    Return a new object with the same shape and type as a given object.
+
+    Returned object will be chunked if if the given object is chunked, or if chunks or chunked_array_type are specified.
 
     Parameters
     ----------
@@ -1449,6 +1486,18 @@ def full_like(
     dtype : dtype or dict-like of dtype, optional
         dtype of the new array. If a dict-like, maps dtypes to
         variables. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1562,7 +1611,12 @@ def full_like(
 
         data_vars = {
             k: _full_like_variable(
-                v.variable, fill_value.get(k, dtypes.NA), dtype_.get(k, None)
+                v.variable,
+                fill_value.get(k, dtypes.NA),
+                dtype_.get(k, None),
+                chunks,
+                chunked_array_type,
+                from_array_kwargs,
             )
             for k, v in other.data_vars.items()
         }
@@ -1571,7 +1625,14 @@ def full_like(
         if isinstance(dtype, Mapping):
             raise ValueError("'dtype' cannot be dict-like when passing a DataArray")
         return DataArray(
-            _full_like_variable(other.variable, fill_value, dtype),
+            _full_like_variable(
+                other.variable,
+                fill_value,
+                dtype,
+                chunks,
+                chunked_array_type,
+                from_array_kwargs,
+            ),
             dims=other.dims,
             coords=other.coords,
             attrs=other.attrs,
@@ -1580,13 +1641,20 @@ def full_like(
     elif isinstance(other, Variable):
         if isinstance(dtype, Mapping):
             raise ValueError("'dtype' cannot be dict-like when passing a Variable")
-        return _full_like_variable(other, fill_value, dtype)
+        return _full_like_variable(
+            other, fill_value, dtype, chunks, chunked_array_type, from_array_kwargs
+        )
     else:
         raise TypeError("Expected DataArray, Dataset, or Variable")
 
 
 def _full_like_variable(
-    other: Variable, fill_value: Any, dtype: DTypeLike = None
+    other: Variable,
+    fill_value: Any,
+    dtype: DTypeLike | None = None,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Variable:
     """Inner function of full_like, where other must be a variable"""
     from xarray.core.variable import Variable
@@ -1594,13 +1662,28 @@ def _full_like_variable(
     if fill_value is dtypes.NA:
         fill_value = dtypes.get_fill_value(dtype if dtype is not None else other.dtype)
 
-    if is_duck_dask_array(other.data):
-        import dask.array
+    if (
+        is_chunked_array(other.data)
+        or chunked_array_type is not None
+        or chunks is not None
+    ):
+        if chunked_array_type is None:
+            chunkmanager = get_chunked_array_type(other.data)
+        else:
+            chunkmanager = guess_chunkmanager(chunked_array_type)
 
         if dtype is None:
             dtype = other.dtype
-        data = dask.array.full(
-            other.shape, fill_value, dtype=dtype, chunks=other.data.chunks
+
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
+        data = chunkmanager.array_api.full(
+            other.shape,
+            fill_value,
+            dtype=dtype,
+            chunks=chunks if chunks else other.data.chunks,
+            **from_array_kwargs,
         )
     else:
         data = np.full_like(other.data, fill_value, dtype=dtype)
@@ -1609,36 +1692,72 @@ def _full_like_variable(
 
 
 @overload
-def zeros_like(other: DataArray, dtype: DTypeLikeSave = None) -> DataArray:
+def zeros_like(
+    other: DataArray,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> DataArray:
     ...
 
 
 @overload
-def zeros_like(other: Dataset, dtype: DTypeMaybeMapping = None) -> Dataset:
+def zeros_like(
+    other: Dataset,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Dataset:
     ...
 
 
 @overload
-def zeros_like(other: Variable, dtype: DTypeLikeSave = None) -> Variable:
+def zeros_like(
+    other: Variable,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Variable:
     ...
 
 
 @overload
 def zeros_like(
-    other: Dataset | DataArray, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
 
 @overload
 def zeros_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
 
 def zeros_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     """Return a new object of zeros with the same shape and
     type as a given dataarray or dataset.
@@ -1649,6 +1768,18 @@ def zeros_like(
         The reference object. The output will have the same dimensions and coordinates as this object.
     dtype : dtype, optional
         dtype of the new array. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1692,40 +1823,83 @@ def zeros_like(
     full_like
 
     """
-    return full_like(other, 0, dtype)
+    return full_like(
+        other,
+        0,
+        dtype,
+        chunks=chunks,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
+    )
 
 
 @overload
-def ones_like(other: DataArray, dtype: DTypeLikeSave = None) -> DataArray:
+def ones_like(
+    other: DataArray,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> DataArray:
     ...
 
 
 @overload
-def ones_like(other: Dataset, dtype: DTypeMaybeMapping = None) -> Dataset:
+def ones_like(
+    other: Dataset,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Dataset:
     ...
 
 
 @overload
-def ones_like(other: Variable, dtype: DTypeLikeSave = None) -> Variable:
+def ones_like(
+    other: Variable,
+    dtype: DTypeLikeSave | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
+) -> Variable:
     ...
 
 
 @overload
 def ones_like(
-    other: Dataset | DataArray, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray:
     ...
 
 
 @overload
 def ones_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     ...
 
 
 def ones_like(
-    other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping = None
+    other: Dataset | DataArray | Variable,
+    dtype: DTypeMaybeMapping | None = None,
+    *,
+    chunks: T_Chunks = None,
+    chunked_array_type: str | None = None,
+    from_array_kwargs: dict[str, Any] | None = None,
 ) -> Dataset | DataArray | Variable:
     """Return a new object of ones with the same shape and
     type as a given dataarray or dataset.
@@ -1736,6 +1910,18 @@ def ones_like(
         The reference object. The output will have the same dimensions and coordinates as this object.
     dtype : dtype, optional
         dtype of the new array. If omitted, it defaults to other.dtype.
+    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
+        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
+        ``{"x": 5, "y": 5}``.
+    chunked_array_type: str, optional
+        Which chunked array type to coerce the underlying data array to.
+        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+        Experimental API that should not be relied upon.
+    from_array_kwargs: dict, optional
+        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+        For example, with dask as the default chunked array type, this method would pass additional kwargs
+        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
 
     Returns
     -------
@@ -1771,7 +1957,14 @@ def ones_like(
     full_like
 
     """
-    return full_like(other, 1, dtype)
+    return full_like(
+        other,
+        1,
+        dtype,
+        chunks=chunks,
+        chunked_array_type=chunked_array_type,
+        from_array_kwargs=from_array_kwargs,
+    )
 
 
 def get_chunksizes(
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -20,7 +20,8 @@
 from xarray.core.indexes import Index, filter_indexes_from_coords
 from xarray.core.merge import merge_attrs, merge_coordinates_without_align
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type
+from xarray.core.pycompat import is_chunked_array, is_duck_dask_array
 from xarray.core.types import Dims, T_DataArray
 from xarray.core.utils import is_dict_like, is_scalar
 from xarray.core.variable import Variable
@@ -675,16 +676,18 @@ def apply_variable_ufunc(
         for arg, core_dims in zip(args, signature.input_core_dims)
     ]
 
-    if any(is_duck_dask_array(array) for array in input_data):
+    if any(is_chunked_array(array) for array in input_data):
         if dask == "forbidden":
             raise ValueError(
-                "apply_ufunc encountered a dask array on an "
-                "argument, but handling for dask arrays has not "
+                "apply_ufunc encountered a chunked array on an "
+                "argument, but handling for chunked arrays has not "
                 "been enabled. Either set the ``dask`` argument "
                 "or load your data into memory first with "
                 "``.load()`` or ``.compute()``"
             )
         elif dask == "parallelized":
+            chunkmanager = get_chunked_array_type(*input_data)
+
             numpy_func = func
 
             if dask_gufunc_kwargs is None:
@@ -697,7 +700,7 @@ def apply_variable_ufunc(
                 for n, (data, core_dims) in enumerate(
                     zip(input_data, signature.input_core_dims)
                 ):
-                    if is_duck_dask_array(data):
+                    if is_chunked_array(data):
                         # core dimensions cannot span multiple chunks
                         for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                             if len(data.chunks[axis]) != 1:
@@ -705,7 +708,7 @@ def apply_variable_ufunc(
                                     f"dimension {dim} on {n}th function argument to "
                                     "apply_ufunc with dask='parallelized' consists of "
                                     "multiple chunks, but is also a core dimension. To "
-                                    "fix, either rechunk into a single dask array chunk along "
+                                    "fix, either rechunk into a single array chunk along "
                                     f"this dimension, i.e., ``.chunk(dict({dim}=-1))``, or "
                                     "pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs`` "
                                     "but beware that this may significantly increase memory usage."
@@ -732,9 +735,7 @@ def apply_variable_ufunc(
                     )
 
             def func(*arrays):
-                import dask.array as da
-
-                res = da.apply_gufunc(
+                res = chunkmanager.apply_gufunc(
                     numpy_func,
                     signature.to_gufunc_string(exclude_dims),
                     *arrays,
@@ -749,8 +750,7 @@ def func(*arrays):
             pass
         else:
             raise ValueError(
-                "unknown setting for dask array handling in "
-                "apply_ufunc: {}".format(dask)
+                "unknown setting for chunked array handling in " f"apply_ufunc: {dask}"
             )
     else:
         if vectorize:
@@ -812,7 +812,7 @@ def func(*arrays):
 
 def apply_array_ufunc(func, *args, dask="forbidden"):
     """Apply a ndarray level function over ndarray objects."""
-    if any(is_duck_dask_array(arg) for arg in args):
+    if any(is_chunked_array(arg) for arg in args):
         if dask == "forbidden":
             raise ValueError(
                 "apply_ufunc encountered a dask array on an "
@@ -2013,7 +2013,7 @@ def to_floatable(x: DataArray) -> DataArray:
             )
         elif x.dtype.kind == "m":
             # timedeltas
-            return x.astype(float)
+            return duck_array_ops.astype(x, dtype=float)
         return x
 
     if isinstance(data, Dataset):
@@ -2061,12 +2061,11 @@ def _calc_idxminmax(
     # This will run argmin or argmax.
     indx = func(array, dim=dim, axis=None, keep_attrs=keep_attrs, skipna=skipna)
 
-    # Handle dask arrays.
-    if is_duck_dask_array(array.data):
-        import dask.array
-
+    # Handle chunked arrays (e.g. dask).
+    if is_chunked_array(array.data):
+        chunkmanager = get_chunked_array_type(array.data)
         chunks = dict(zip(array.dims, array.chunks))
-        dask_coord = dask.array.from_array(array[dim].data, chunks=chunks[dim])
+        dask_coord = chunkmanager.from_array(array[dim].data, chunks=chunks[dim])
         res = indx.copy(data=dask_coord[indx.data.ravel()].reshape(indx.shape))
         # we need to attach back the dim name
         res.name = dim
@@ -2153,16 +2152,14 @@ def unify_chunks(*objects: Dataset | DataArray) -> tuple[Dataset | DataArray, ..
     if not unify_chunks_args:
         return objects
 
-    # Run dask.array.core.unify_chunks
-    from dask.array.core import unify_chunks
-
-    _, dask_data = unify_chunks(*unify_chunks_args)
-    dask_data_iter = iter(dask_data)
+    chunkmanager = get_chunked_array_type(*[arg for arg in unify_chunks_args])
+    _, chunked_data = chunkmanager.unify_chunks(*unify_chunks_args)
+    chunked_data_iter = iter(chunked_data)
     out: list[Dataset | DataArray] = []
     for obj, ds in zip(objects, datasets):
         for k, v in ds._variables.items():
             if v.chunks is not None:
-                ds._variables[k] = v.copy(data=next(dask_data_iter))
+                ds._variables[k] = v.copy(data=next(chunked_data_iter))
         out.append(obj._from_temp_dataset(ds) if isinstance(obj, DataArray) else ds)
 
     return tuple(out)
diff --git a/xarray/core/dask_array_ops.py b/xarray/core/dask_array_ops.py
--- a/xarray/core/dask_array_ops.py
+++ b/xarray/core/dask_array_ops.py
@@ -1,9 +1,5 @@
 from __future__ import annotations
 
-from functools import partial
-
-from numpy.core.multiarray import normalize_axis_index  # type: ignore[attr-defined]
-
 from xarray.core import dtypes, nputils
 
 
@@ -96,36 +92,3 @@ def _fill_with_last_one(a, b):
         axis=axis,
         dtype=array.dtype,
     )
-
-
-def _first_last_wrapper(array, *, axis, op, keepdims):
-    return op(array, axis, keepdims=keepdims)
-
-
-def _first_or_last(darray, axis, op):
-    import dask.array
-
-    # This will raise the same error message seen for numpy
-    axis = normalize_axis_index(axis, darray.ndim)
-
-    wrapped_op = partial(_first_last_wrapper, op=op)
-    return dask.array.reduction(
-        darray,
-        chunk=wrapped_op,
-        aggregate=wrapped_op,
-        axis=axis,
-        dtype=darray.dtype,
-        keepdims=False,  # match numpy version
-    )
-
-
-def nanfirst(darray, axis):
-    from xarray.core.duck_array_ops import nanfirst
-
-    return _first_or_last(darray, axis, op=nanfirst)
-
-
-def nanlast(darray, axis):
-    from xarray.core.duck_array_ops import nanlast
-
-    return _first_or_last(darray, axis, op=nanlast)
diff --git a/xarray/core/daskmanager.py b/xarray/core/daskmanager.py
new file mode 100644
--- /dev/null
+++ b/xarray/core/daskmanager.py
@@ -0,0 +1,215 @@
+from __future__ import annotations
+
+from collections.abc import Iterable, Sequence
+from typing import TYPE_CHECKING, Any, Callable
+
+import numpy as np
+from packaging.version import Version
+
+from xarray.core.duck_array_ops import dask_available
+from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
+from xarray.core.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
+from xarray.core.pycompat import is_duck_dask_array
+
+if TYPE_CHECKING:
+    from xarray.core.types import DaskArray, T_Chunks, T_NormalizedChunks
+
+
+class DaskManager(ChunkManagerEntrypoint["DaskArray"]):
+    array_cls: type[DaskArray]
+    available: bool = dask_available
+
+    def __init__(self) -> None:
+        # TODO can we replace this with a class attribute instead?
+
+        from dask.array import Array
+
+        self.array_cls = Array
+
+    def is_chunked_array(self, data: Any) -> bool:
+        return is_duck_dask_array(data)
+
+    def chunks(self, data: DaskArray) -> T_NormalizedChunks:
+        return data.chunks
+
+    def normalize_chunks(
+        self,
+        chunks: T_Chunks | T_NormalizedChunks,
+        shape: tuple[int, ...] | None = None,
+        limit: int | None = None,
+        dtype: np.dtype | None = None,
+        previous_chunks: T_NormalizedChunks | None = None,
+    ) -> T_NormalizedChunks:
+        """Called by open_dataset"""
+        from dask.array.core import normalize_chunks
+
+        return normalize_chunks(
+            chunks,
+            shape=shape,
+            limit=limit,
+            dtype=dtype,
+            previous_chunks=previous_chunks,
+        )
+
+    def from_array(self, data: Any, chunks, **kwargs) -> DaskArray:
+        import dask.array as da
+
+        if isinstance(data, ImplicitToExplicitIndexingAdapter):
+            # lazily loaded backend array classes should use NumPy array operations.
+            kwargs["meta"] = np.ndarray
+
+        return da.from_array(
+            data,
+            chunks,
+            **kwargs,
+        )
+
+    def compute(self, *data: DaskArray, **kwargs) -> tuple[np.ndarray, ...]:
+        from dask.array import compute
+
+        return compute(*data, **kwargs)
+
+    @property
+    def array_api(self) -> Any:
+        from dask import array as da
+
+        return da
+
+    def reduction(
+        self,
+        arr: T_ChunkedArray,
+        func: Callable,
+        combine_func: Callable | None = None,
+        aggregate_func: Callable | None = None,
+        axis: int | Sequence[int] | None = None,
+        dtype: np.dtype | None = None,
+        keepdims: bool = False,
+    ) -> T_ChunkedArray:
+        from dask.array import reduction
+
+        return reduction(
+            arr,
+            chunk=func,
+            combine=combine_func,
+            aggregate=aggregate_func,
+            axis=axis,
+            dtype=dtype,
+            keepdims=keepdims,
+        )
+
+    def apply_gufunc(
+        self,
+        func: Callable,
+        signature: str,
+        *args: Any,
+        axes: Sequence[tuple[int, ...]] | None = None,
+        axis: int | None = None,
+        keepdims: bool = False,
+        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
+        output_sizes: dict[str, int] | None = None,
+        vectorize: bool | None = None,
+        allow_rechunk: bool = False,
+        meta: tuple[np.ndarray, ...] | None = None,
+        **kwargs,
+    ):
+        from dask.array.gufunc import apply_gufunc
+
+        return apply_gufunc(
+            func,
+            signature,
+            *args,
+            axes=axes,
+            axis=axis,
+            keepdims=keepdims,
+            output_dtypes=output_dtypes,
+            output_sizes=output_sizes,
+            vectorize=vectorize,
+            allow_rechunk=allow_rechunk,
+            meta=meta,
+            **kwargs,
+        )
+
+    def map_blocks(
+        self,
+        func: Callable,
+        *args: Any,
+        dtype: np.typing.DTypeLike | None = None,
+        chunks: tuple[int, ...] | None = None,
+        drop_axis: int | Sequence[int] | None = None,
+        new_axis: int | Sequence[int] | None = None,
+        **kwargs,
+    ):
+        import dask
+        from dask.array import map_blocks
+
+        if drop_axis is None and Version(dask.__version__) < Version("2022.9.1"):
+            # See https://github.com/pydata/xarray/pull/7019#discussion_r1196729489
+            # TODO remove once dask minimum version >= 2022.9.1
+            drop_axis = []
+
+        # pass through name, meta, token as kwargs
+        return map_blocks(
+            func,
+            *args,
+            dtype=dtype,
+            chunks=chunks,
+            drop_axis=drop_axis,
+            new_axis=new_axis,
+            **kwargs,
+        )
+
+    def blockwise(
+        self,
+        func: Callable,
+        out_ind: Iterable,
+        *args: Any,
+        # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
+        name: str | None = None,
+        token=None,
+        dtype: np.dtype | None = None,
+        adjust_chunks: dict[Any, Callable] | None = None,
+        new_axes: dict[Any, int] | None = None,
+        align_arrays: bool = True,
+        concatenate: bool | None = None,
+        meta=None,
+        **kwargs,
+    ):
+        from dask.array import blockwise
+
+        return blockwise(
+            func,
+            out_ind,
+            *args,
+            name=name,
+            token=token,
+            dtype=dtype,
+            adjust_chunks=adjust_chunks,
+            new_axes=new_axes,
+            align_arrays=align_arrays,
+            concatenate=concatenate,
+            meta=meta,
+            **kwargs,
+        )
+
+    def unify_chunks(
+        self,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
+        **kwargs,
+    ) -> tuple[dict[str, T_NormalizedChunks], list[DaskArray]]:
+        from dask.array.core import unify_chunks
+
+        return unify_chunks(*args, **kwargs)
+
+    def store(
+        self,
+        sources: DaskArray | Sequence[DaskArray],
+        targets: Any,
+        **kwargs,
+    ):
+        from dask.array import store
+
+        return store(
+            sources=sources,
+            targets=targets,
+            **kwargs,
+        )
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -77,6 +77,7 @@
     from xarray.backends import ZarrStore
     from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
     from xarray.core.groupby import DataArrayGroupBy
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.resample import DataArrayResample
     from xarray.core.rolling import DataArrayCoarsen, DataArrayRolling
     from xarray.core.types import (
@@ -1264,6 +1265,8 @@ def chunk(
         token: str | None = None,
         lock: bool = False,
         inline_array: bool = False,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: Any,
     ) -> T_DataArray:
         """Coerce this array's data into a dask arrays with the given chunks.
@@ -1285,12 +1288,21 @@ def chunk(
             Prefix for the name of the new dask array.
         token : str, optional
             Token uniquely identifying this array.
-        lock : optional
+        lock : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
-        inline_array: optional
+        inline_array: bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce the underlying data array to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided.
@@ -1328,6 +1340,8 @@ def chunk(
             token=token,
             lock=lock,
             inline_array=inline_array,
+            chunked_array_type=chunked_array_type,
+            from_array_kwargs=from_array_kwargs,
         )
         return self._from_temp_dataset(ds)
 
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -51,6 +51,7 @@
 )
 from xarray.core.computation import unify_chunks
 from xarray.core.coordinates import DatasetCoordinates, assert_coordinate_consistent
+from xarray.core.daskmanager import DaskManager
 from xarray.core.duck_array_ops import datetime_to_numeric
 from xarray.core.indexes import (
     Index,
@@ -73,7 +74,16 @@
 )
 from xarray.core.missing import get_clean_interp_index
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import array_type, is_duck_array, is_duck_dask_array
+from xarray.core.parallelcompat import (
+    get_chunked_array_type,
+    guess_chunkmanager,
+)
+from xarray.core.pycompat import (
+    array_type,
+    is_chunked_array,
+    is_duck_array,
+    is_duck_dask_array,
+)
 from xarray.core.types import QuantileMethods, T_Dataset
 from xarray.core.utils import (
     Default,
@@ -107,6 +117,7 @@
     from xarray.core.dataarray import DataArray
     from xarray.core.groupby import DatasetGroupBy
     from xarray.core.merge import CoercibleMapping
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.resample import DatasetResample
     from xarray.core.rolling import DatasetCoarsen, DatasetRolling
     from xarray.core.types import (
@@ -202,13 +213,11 @@ def _assert_empty(args: tuple, msg: str = "%s") -> None:
         raise ValueError(msg % args)
 
 
-def _get_chunk(var, chunks):
+def _get_chunk(var: Variable, chunks, chunkmanager: ChunkManagerEntrypoint):
     """
     Return map from each dim to chunk sizes, accounting for backend's preferred chunks.
     """
 
-    import dask.array as da
-
     if isinstance(var, IndexVariable):
         return {}
     dims = var.dims
@@ -225,7 +234,8 @@ def _get_chunk(var, chunks):
         chunks.get(dim, None) or preferred_chunk_sizes
         for dim, preferred_chunk_sizes in zip(dims, preferred_chunk_shape)
     )
-    chunk_shape = da.core.normalize_chunks(
+
+    chunk_shape = chunkmanager.normalize_chunks(
         chunk_shape, shape=shape, dtype=var.dtype, previous_chunks=preferred_chunk_shape
     )
 
@@ -242,7 +252,7 @@ def _get_chunk(var, chunks):
             # expresses the preferred chunks, the sequence sums to the size.
             preferred_stops = (
                 range(preferred_chunk_sizes, size, preferred_chunk_sizes)
-                if isinstance(preferred_chunk_sizes, Number)
+                if isinstance(preferred_chunk_sizes, int)
                 else itertools.accumulate(preferred_chunk_sizes[:-1])
             )
             # Gather any stop indices of the specified chunks that are not a stop index
@@ -253,7 +263,7 @@ def _get_chunk(var, chunks):
             )
             if breaks:
                 warnings.warn(
-                    "The specified Dask chunks separate the stored chunks along "
+                    "The specified chunks separate the stored chunks along "
                     f'dimension "{dim}" starting at index {min(breaks)}. This could '
                     "degrade performance. Instead, consider rechunking after loading."
                 )
@@ -270,18 +280,37 @@ def _maybe_chunk(
     name_prefix="xarray-",
     overwrite_encoded_chunks=False,
     inline_array=False,
+    chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+    from_array_kwargs=None,
 ):
-    from dask.base import tokenize
-
     if chunks is not None:
         chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
+
     if var.ndim:
-        # when rechunking by different amounts, make sure dask names change
-        # by provinding chunks as an input to tokenize.
-        # subtle bugs result otherwise. see GH3350
-        token2 = tokenize(name, token if token else var._data, chunks)
-        name2 = f"{name_prefix}{name}-{token2}"
-        var = var.chunk(chunks, name=name2, lock=lock, inline_array=inline_array)
+        chunked_array_type = guess_chunkmanager(
+            chunked_array_type
+        )  # coerce string to ChunkManagerEntrypoint type
+        if isinstance(chunked_array_type, DaskManager):
+            from dask.base import tokenize
+
+            # when rechunking by different amounts, make sure dask names change
+            # by providing chunks as an input to tokenize.
+            # subtle bugs result otherwise. see GH3350
+            token2 = tokenize(name, token if token else var._data, chunks)
+            name2 = f"{name_prefix}{name}-{token2}"
+
+            from_array_kwargs = utils.consolidate_dask_from_array_kwargs(
+                from_array_kwargs,
+                name=name2,
+                lock=lock,
+                inline_array=inline_array,
+            )
+
+        var = var.chunk(
+            chunks,
+            chunked_array_type=chunked_array_type,
+            from_array_kwargs=from_array_kwargs,
+        )
 
         if overwrite_encoded_chunks and var.chunks is not None:
             var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
@@ -743,13 +772,13 @@ def load(self: T_Dataset, **kwargs) -> T_Dataset:
         """
         # access .data to coerce everything to numpy or dask arrays
         lazy_data = {
-            k: v._data for k, v in self.variables.items() if is_duck_dask_array(v._data)
+            k: v._data for k, v in self.variables.items() if is_chunked_array(v._data)
         }
         if lazy_data:
-            import dask.array as da
+            chunkmanager = get_chunked_array_type(*lazy_data.values())
 
-            # evaluate all the dask arrays simultaneously
-            evaluated_data = da.compute(*lazy_data.values(), **kwargs)
+            # evaluate all the chunked arrays simultaneously
+            evaluated_data = chunkmanager.compute(*lazy_data.values(), **kwargs)
 
             for k, data in zip(lazy_data, evaluated_data):
                 self.variables[k].data = data
@@ -1575,7 +1604,7 @@ def _setitem_check(self, key, value):
                 val = np.array(val)
 
             # type conversion
-            new_value[name] = val.astype(var_k.dtype, copy=False)
+            new_value[name] = duck_array_ops.astype(val, dtype=var_k.dtype, copy=False)
 
         # check consistency of dimension sizes and dimension coordinates
         if isinstance(value, DataArray) or isinstance(value, Dataset):
@@ -1945,6 +1974,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> ZarrStore:
         ...
 
@@ -1966,6 +1996,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> Delayed:
         ...
 
@@ -1984,6 +2015,7 @@ def to_zarr(
         safe_chunks: bool = True,
         storage_options: dict[str, str] | None = None,
         zarr_version: int | None = None,
+        chunkmanager_store_kwargs: dict[str, Any] | None = None,
     ) -> ZarrStore | Delayed:
         """Write dataset contents to a zarr group.
 
@@ -2072,6 +2104,10 @@ def to_zarr(
             The desired zarr spec version to target (currently 2 or 3). The
             default of None will attempt to determine the zarr version from
             ``store`` when possible, otherwise defaulting to 2.
+        chunkmanager_store_kwargs : dict, optional
+            Additional keyword arguments passed on to the `ChunkManager.store` method used to store
+            chunked arrays. For example for a dask array additional kwargs will be passed eventually to
+            :py:func:`dask.array.store()`. Experimental API that should not be relied upon.
 
         Returns
         -------
@@ -2117,6 +2153,7 @@ def to_zarr(
             region=region,
             safe_chunks=safe_chunks,
             zarr_version=zarr_version,
+            chunkmanager_store_kwargs=chunkmanager_store_kwargs,
         )
 
     def __repr__(self) -> str:
@@ -2205,6 +2242,8 @@ def chunk(
         token: str | None = None,
         lock: bool = False,
         inline_array: bool = False,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: None | int | str | tuple[int, ...],
     ) -> T_Dataset:
         """Coerce all arrays in this dataset into dask arrays with the given
@@ -2232,6 +2271,15 @@ def chunk(
         inline_array: bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce this datasets' arrays to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided
@@ -2266,8 +2314,22 @@ def chunk(
                 f"some chunks keys are not dimensions on this object: {bad_dims}"
             )
 
+        chunkmanager = guess_chunkmanager(chunked_array_type)
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
         variables = {
-            k: _maybe_chunk(k, v, chunks, token, lock, name_prefix)
+            k: _maybe_chunk(
+                k,
+                v,
+                chunks,
+                token,
+                lock,
+                name_prefix,
+                inline_array=inline_array,
+                chunked_array_type=chunkmanager,
+                from_array_kwargs=from_array_kwargs.copy(),
+            )
             for k, v in self.variables.items()
         }
         return self._replace(variables)
@@ -2305,7 +2367,7 @@ def _validate_indexers(
                 if v.dtype.kind in "US":
                     index = self._indexes[k].to_pandas_index()
                     if isinstance(index, pd.DatetimeIndex):
-                        v = v.astype("datetime64[ns]")
+                        v = duck_array_ops.astype(v, dtype="datetime64[ns]")
                     elif isinstance(index, CFTimeIndex):
                         v = _parse_array_of_cftime_strings(v, index.date_type)
 
diff --git a/xarray/core/duck_array_ops.py b/xarray/core/duck_array_ops.py
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -9,6 +9,7 @@
 import datetime
 import inspect
 import warnings
+from functools import partial
 from importlib import import_module
 
 import numpy as np
@@ -29,10 +30,11 @@
     zeros_like,  # noqa
 )
 from numpy import concatenate as _concatenate
+from numpy.core.multiarray import normalize_axis_index  # type: ignore[attr-defined]
 from numpy.lib.stride_tricks import sliding_window_view  # noqa
 
 from xarray.core import dask_array_ops, dtypes, nputils
-from xarray.core.nputils import nanfirst, nanlast
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.pycompat import array_type, is_duck_dask_array
 from xarray.core.utils import is_duck_array, module_available
 
@@ -640,10 +642,10 @@ def first(values, axis, skipna=None):
     """Return the first non-NA elements in this array along the given axis"""
     if (skipna or skipna is None) and values.dtype.kind not in "iSU":
         # only bother for dtypes that can hold NaN
-        if is_duck_dask_array(values):
-            return dask_array_ops.nanfirst(values, axis)
+        if is_chunked_array(values):
+            return chunked_nanfirst(values, axis)
         else:
-            return nanfirst(values, axis)
+            return nputils.nanfirst(values, axis)
     return take(values, 0, axis=axis)
 
 
@@ -651,10 +653,10 @@ def last(values, axis, skipna=None):
     """Return the last non-NA elements in this array along the given axis"""
     if (skipna or skipna is None) and values.dtype.kind not in "iSU":
         # only bother for dtypes that can hold NaN
-        if is_duck_dask_array(values):
-            return dask_array_ops.nanlast(values, axis)
+        if is_chunked_array(values):
+            return chunked_nanlast(values, axis)
         else:
-            return nanlast(values, axis)
+            return nputils.nanlast(values, axis)
     return take(values, -1, axis=axis)
 
 
@@ -673,3 +675,32 @@ def push(array, n, axis):
         return dask_array_ops.push(array, n, axis)
     else:
         return push(array, n, axis)
+
+
+def _first_last_wrapper(array, *, axis, op, keepdims):
+    return op(array, axis, keepdims=keepdims)
+
+
+def _chunked_first_or_last(darray, axis, op):
+    chunkmanager = get_chunked_array_type(darray)
+
+    # This will raise the same error message seen for numpy
+    axis = normalize_axis_index(axis, darray.ndim)
+
+    wrapped_op = partial(_first_last_wrapper, op=op)
+    return chunkmanager.reduction(
+        darray,
+        func=wrapped_op,
+        aggregate_func=wrapped_op,
+        axis=axis,
+        dtype=darray.dtype,
+        keepdims=False,  # match numpy version
+    )
+
+
+def chunked_nanfirst(darray, axis):
+    return _chunked_first_or_last(darray, axis, op=nputils.nanfirst)
+
+
+def chunked_nanlast(darray, axis):
+    return _chunked_first_or_last(darray, axis, op=nputils.nanlast)
diff --git a/xarray/core/indexing.py b/xarray/core/indexing.py
--- a/xarray/core/indexing.py
+++ b/xarray/core/indexing.py
@@ -17,6 +17,7 @@
 from xarray.core import duck_array_ops
 from xarray.core.nputils import NumpyVIndexAdapter
 from xarray.core.options import OPTIONS
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.pycompat import (
     array_type,
     integer_types,
@@ -1142,16 +1143,15 @@ def _arrayize_vectorized_indexer(indexer, shape):
     return VectorizedIndexer(tuple(new_key))
 
 
-def _dask_array_with_chunks_hint(array, chunks):
-    """Create a dask array using the chunks hint for dimensions of size > 1."""
-    import dask.array as da
+def _chunked_array_with_chunks_hint(array, chunks, chunkmanager):
+    """Create a chunked array using the chunks hint for dimensions of size > 1."""
 
     if len(chunks) < array.ndim:
         raise ValueError("not enough chunks in hint")
     new_chunks = []
     for chunk, size in zip(chunks, array.shape):
         new_chunks.append(chunk if size > 1 else (1,))
-    return da.from_array(array, new_chunks)
+    return chunkmanager.from_array(array, new_chunks)
 
 
 def _logical_any(args):
@@ -1165,8 +1165,11 @@ def _masked_result_drop_slice(key, data=None):
     new_keys = []
     for k in key:
         if isinstance(k, np.ndarray):
-            if is_duck_dask_array(data):
-                new_keys.append(_dask_array_with_chunks_hint(k, chunks_hint))
+            if is_chunked_array(data):
+                chunkmanager = get_chunked_array_type(data)
+                new_keys.append(
+                    _chunked_array_with_chunks_hint(k, chunks_hint, chunkmanager)
+                )
             elif isinstance(data, array_type("sparse")):
                 import sparse
 
diff --git a/xarray/core/missing.py b/xarray/core/missing.py
--- a/xarray/core/missing.py
+++ b/xarray/core/missing.py
@@ -15,7 +15,7 @@
 from xarray.core.computation import apply_ufunc
 from xarray.core.duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
 from xarray.core.options import OPTIONS, _get_keep_attrs
-from xarray.core.pycompat import is_duck_dask_array
+from xarray.core.parallelcompat import get_chunked_array_type, is_chunked_array
 from xarray.core.types import Interp1dOptions, InterpOptions
 from xarray.core.utils import OrderedSet, is_scalar
 from xarray.core.variable import Variable, broadcast_variables
@@ -693,8 +693,8 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
     else:
         func, kwargs = _get_interpolator_nd(method, **kwargs)
 
-    if is_duck_dask_array(var):
-        import dask.array as da
+    if is_chunked_array(var):
+        chunkmanager = get_chunked_array_type(var)
 
         ndim = var.ndim
         nconst = ndim - len(x)
@@ -716,7 +716,7 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
             *new_x_arginds,
         )
 
-        _, rechunked = da.unify_chunks(*args)
+        _, rechunked = chunkmanager.unify_chunks(*args)
 
         args = tuple(elem for pair in zip(rechunked, args[1::2]) for elem in pair)
 
@@ -741,8 +741,8 @@ def interp_func(var, x, new_x, method: InterpOptions, kwargs):
 
         meta = var._meta
 
-        return da.blockwise(
-            _dask_aware_interpnd,
+        return chunkmanager.blockwise(
+            _chunked_aware_interpnd,
             out_ind,
             *args,
             interp_func=func,
@@ -785,8 +785,8 @@ def _interpnd(var, x, new_x, func, kwargs):
     return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)
 
 
-def _dask_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
-    """Wrapper for `_interpnd` through `blockwise`
+def _chunked_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
+    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.
 
     The first half arrays in `coords` are original coordinates,
     the other half are destination coordinates
diff --git a/xarray/core/nanops.py b/xarray/core/nanops.py
--- a/xarray/core/nanops.py
+++ b/xarray/core/nanops.py
@@ -6,6 +6,7 @@
 
 from xarray.core import dtypes, nputils, utils
 from xarray.core.duck_array_ops import (
+    astype,
     count,
     fillna,
     isnull,
@@ -22,7 +23,7 @@ def _maybe_null_out(result, axis, mask, min_count=1):
     if axis is not None and getattr(result, "ndim", False):
         null_mask = (np.take(mask.shape, axis).prod() - mask.sum(axis) - min_count) < 0
         dtype, fill_value = dtypes.maybe_promote(result.dtype)
-        result = where(null_mask, fill_value, result.astype(dtype))
+        result = where(null_mask, fill_value, astype(result, dtype))
 
     elif getattr(result, "dtype", None) not in dtypes.NAT_TYPES:
         null_mask = mask.size - mask.sum()
@@ -140,7 +141,7 @@ def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
     value_mean = _nanmean_ddof_object(
         ddof=0, value=value, axis=axis, keepdims=True, **kwargs
     )
-    squared = (value.astype(value_mean.dtype) - value_mean) ** 2
+    squared = (astype(value, value_mean.dtype) - value_mean) ** 2
     return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)
 
 
diff --git a/xarray/core/parallelcompat.py b/xarray/core/parallelcompat.py
new file mode 100644
--- /dev/null
+++ b/xarray/core/parallelcompat.py
@@ -0,0 +1,280 @@
+"""
+The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
+It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
+but for now it is just a private experiment.
+"""
+from __future__ import annotations
+
+import functools
+import sys
+from abc import ABC, abstractmethod
+from collections.abc import Iterable, Sequence
+from importlib.metadata import EntryPoint, entry_points
+from typing import (
+    TYPE_CHECKING,
+    Any,
+    Callable,
+    Generic,
+    TypeVar,
+)
+
+import numpy as np
+
+from xarray.core.pycompat import is_chunked_array
+
+T_ChunkedArray = TypeVar("T_ChunkedArray")
+
+if TYPE_CHECKING:
+    from xarray.core.types import T_Chunks, T_NormalizedChunks
+
+
+@functools.lru_cache(maxsize=1)
+def list_chunkmanagers() -> dict[str, ChunkManagerEntrypoint]:
+    """
+    Return a dictionary of available chunk managers and their ChunkManagerEntrypoint objects.
+
+    Notes
+    -----
+    # New selection mechanism introduced with Python 3.10. See GH6514.
+    """
+    if sys.version_info >= (3, 10):
+        entrypoints = entry_points(group="xarray.chunkmanagers")
+    else:
+        entrypoints = entry_points().get("xarray.chunkmanagers", ())
+
+    return load_chunkmanagers(entrypoints)
+
+
+def load_chunkmanagers(
+    entrypoints: Sequence[EntryPoint],
+) -> dict[str, ChunkManagerEntrypoint]:
+    """Load entrypoints and instantiate chunkmanagers only once."""
+
+    loaded_entrypoints = {
+        entrypoint.name: entrypoint.load() for entrypoint in entrypoints
+    }
+
+    available_chunkmanagers = {
+        name: chunkmanager()
+        for name, chunkmanager in loaded_entrypoints.items()
+        if chunkmanager.available
+    }
+    return available_chunkmanagers
+
+
+def guess_chunkmanager(
+    manager: str | ChunkManagerEntrypoint | None,
+) -> ChunkManagerEntrypoint:
+    """
+    Get namespace of chunk-handling methods, guessing from what's available.
+
+    If the name of a specific ChunkManager is given (e.g. "dask"), then use that.
+    Else use whatever is installed, defaulting to dask if there are multiple options.
+    """
+
+    chunkmanagers = list_chunkmanagers()
+
+    if manager is None:
+        if len(chunkmanagers) == 1:
+            # use the only option available
+            manager = next(iter(chunkmanagers.keys()))
+        else:
+            # default to trying to use dask
+            manager = "dask"
+
+    if isinstance(manager, str):
+        if manager not in chunkmanagers:
+            raise ValueError(
+                f"unrecognized chunk manager {manager} - must be one of: {list(chunkmanagers)}"
+            )
+
+        return chunkmanagers[manager]
+    elif isinstance(manager, ChunkManagerEntrypoint):
+        # already a valid ChunkManager so just pass through
+        return manager
+    else:
+        raise TypeError(
+            f"manager must be a string or instance of ChunkManagerEntrypoint, but received type {type(manager)}"
+        )
+
+
+def get_chunked_array_type(*args) -> ChunkManagerEntrypoint:
+    """
+    Detects which parallel backend should be used for given set of arrays.
+
+    Also checks that all arrays are of same chunking type (i.e. not a mix of cubed and dask).
+    """
+
+    # TODO this list is probably redundant with something inside xarray.apply_ufunc
+    ALLOWED_NON_CHUNKED_TYPES = {int, float, np.ndarray}
+
+    chunked_arrays = [
+        a
+        for a in args
+        if is_chunked_array(a) and type(a) not in ALLOWED_NON_CHUNKED_TYPES
+    ]
+
+    # Asserts all arrays are the same type (or numpy etc.)
+    chunked_array_types = {type(a) for a in chunked_arrays}
+    if len(chunked_array_types) > 1:
+        raise TypeError(
+            f"Mixing chunked array types is not supported, but received multiple types: {chunked_array_types}"
+        )
+    elif len(chunked_array_types) == 0:
+        raise TypeError("Expected a chunked array but none were found")
+
+    # iterate over defined chunk managers, seeing if each recognises this array type
+    chunked_arr = chunked_arrays[0]
+    chunkmanagers = list_chunkmanagers()
+    selected = [
+        chunkmanager
+        for chunkmanager in chunkmanagers.values()
+        if chunkmanager.is_chunked_array(chunked_arr)
+    ]
+    if not selected:
+        raise TypeError(
+            f"Could not find a Chunk Manager which recognises type {type(chunked_arr)}"
+        )
+    elif len(selected) >= 2:
+        raise TypeError(f"Multiple ChunkManagers recognise type {type(chunked_arr)}")
+    else:
+        return selected[0]
+
+
+class ChunkManagerEntrypoint(ABC, Generic[T_ChunkedArray]):
+    """
+    Adapter between a particular parallel computing framework and xarray.
+
+    Attributes
+    ----------
+    array_cls
+        Type of the array class this parallel computing framework provides.
+
+        Parallel frameworks need to provide an array class that supports the array API standard.
+        Used for type checking.
+    """
+
+    array_cls: type[T_ChunkedArray]
+    available: bool = True
+
+    @abstractmethod
+    def __init__(self) -> None:
+        raise NotImplementedError()
+
+    def is_chunked_array(self, data: Any) -> bool:
+        return isinstance(data, self.array_cls)
+
+    @abstractmethod
+    def chunks(self, data: T_ChunkedArray) -> T_NormalizedChunks:
+        raise NotImplementedError()
+
+    @abstractmethod
+    def normalize_chunks(
+        self,
+        chunks: T_Chunks | T_NormalizedChunks,
+        shape: tuple[int, ...] | None = None,
+        limit: int | None = None,
+        dtype: np.dtype | None = None,
+        previous_chunks: T_NormalizedChunks | None = None,
+    ) -> T_NormalizedChunks:
+        """Called by open_dataset"""
+        raise NotImplementedError()
+
+    @abstractmethod
+    def from_array(
+        self, data: np.ndarray, chunks: T_Chunks, **kwargs
+    ) -> T_ChunkedArray:
+        """Called when .chunk is called on an xarray object that is not already chunked."""
+        raise NotImplementedError()
+
+    def rechunk(
+        self,
+        data: T_ChunkedArray,
+        chunks: T_NormalizedChunks | tuple[int, ...] | T_Chunks,
+        **kwargs,
+    ) -> T_ChunkedArray:
+        """Called when .chunk is called on an xarray object that is already chunked."""
+        return data.rechunk(chunks, **kwargs)  # type: ignore[attr-defined]
+
+    @abstractmethod
+    def compute(self, *data: T_ChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
+        """Used anytime something needs to computed, including multiple arrays at once."""
+        raise NotImplementedError()
+
+    @property
+    def array_api(self) -> Any:
+        """Return the array_api namespace following the python array API standard."""
+        raise NotImplementedError()
+
+    def reduction(
+        self,
+        arr: T_ChunkedArray,
+        func: Callable,
+        combine_func: Callable | None = None,
+        aggregate_func: Callable | None = None,
+        axis: int | Sequence[int] | None = None,
+        dtype: np.dtype | None = None,
+        keepdims: bool = False,
+    ) -> T_ChunkedArray:
+        """Used in some reductions like nanfirst, which is used by groupby.first"""
+        raise NotImplementedError()
+
+    @abstractmethod
+    def apply_gufunc(
+        self,
+        func: Callable,
+        signature: str,
+        *args: Any,
+        axes: Sequence[tuple[int, ...]] | None = None,
+        keepdims: bool = False,
+        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
+        vectorize: bool | None = None,
+        **kwargs,
+    ):
+        """
+        Called inside xarray.apply_ufunc, so must be supplied for vast majority of xarray computations to be supported.
+        """
+        raise NotImplementedError()
+
+    def map_blocks(
+        self,
+        func: Callable,
+        *args: Any,
+        dtype: np.typing.DTypeLike | None = None,
+        chunks: tuple[int, ...] | None = None,
+        drop_axis: int | Sequence[int] | None = None,
+        new_axis: int | Sequence[int] | None = None,
+        **kwargs,
+    ):
+        """Called in elementwise operations, but notably not called in xarray.map_blocks."""
+        raise NotImplementedError()
+
+    def blockwise(
+        self,
+        func: Callable,
+        out_ind: Iterable,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
+        adjust_chunks: dict[Any, Callable] | None = None,
+        new_axes: dict[Any, int] | None = None,
+        align_arrays: bool = True,
+        **kwargs,
+    ):
+        """Called by some niche functions in xarray."""
+        raise NotImplementedError()
+
+    def unify_chunks(
+        self,
+        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
+        **kwargs,
+    ) -> tuple[dict[str, T_NormalizedChunks], list[T_ChunkedArray]]:
+        """Called by xr.unify_chunks."""
+        raise NotImplementedError()
+
+    def store(
+        self,
+        sources: T_ChunkedArray | Sequence[T_ChunkedArray],
+        targets: Any,
+        **kwargs: dict[str, Any],
+    ):
+        """Used when writing to any backend."""
+        raise NotImplementedError()
diff --git a/xarray/core/pycompat.py b/xarray/core/pycompat.py
--- a/xarray/core/pycompat.py
+++ b/xarray/core/pycompat.py
@@ -12,7 +12,7 @@
 integer_types = (int, np.integer)
 
 if TYPE_CHECKING:
-    ModType = Literal["dask", "pint", "cupy", "sparse"]
+    ModType = Literal["dask", "pint", "cupy", "sparse", "cubed"]
     DuckArrayTypes = tuple[type[Any], ...]  # TODO: improve this? maybe Generic
 
 
@@ -30,7 +30,7 @@ class DuckArrayModule:
     available: bool
 
     def __init__(self, mod: ModType) -> None:
-        duck_array_module: ModuleType | None = None
+        duck_array_module: ModuleType | None
         duck_array_version: Version
         duck_array_type: DuckArrayTypes
         try:
@@ -45,6 +45,8 @@ def __init__(self, mod: ModType) -> None:
                 duck_array_type = (duck_array_module.ndarray,)
             elif mod == "sparse":
                 duck_array_type = (duck_array_module.SparseArray,)
+            elif mod == "cubed":
+                duck_array_type = (duck_array_module.Array,)
             else:
                 raise NotImplementedError
 
@@ -81,5 +83,9 @@ def is_duck_dask_array(x):
     return is_duck_array(x) and is_dask_collection(x)
 
 
+def is_chunked_array(x) -> bool:
+    return is_duck_dask_array(x) or (is_duck_array(x) and hasattr(x, "chunks"))
+
+
 def is_0d_dask_array(x):
     return is_duck_dask_array(x) and is_scalar(x)
diff --git a/xarray/core/rolling.py b/xarray/core/rolling.py
--- a/xarray/core/rolling.py
+++ b/xarray/core/rolling.py
@@ -158,9 +158,9 @@ def method(self, keep_attrs=None, **kwargs):
         return method
 
     def _mean(self, keep_attrs, **kwargs):
-        result = self.sum(keep_attrs=False, **kwargs) / self.count(
-            keep_attrs=False
-        ).astype(self.obj.dtype, copy=False)
+        result = self.sum(keep_attrs=False, **kwargs) / duck_array_ops.astype(
+            self.count(keep_attrs=False), dtype=self.obj.dtype, copy=False
+        )
         if keep_attrs:
             result.attrs = self.obj.attrs
         return result
diff --git a/xarray/core/types.py b/xarray/core/types.py
--- a/xarray/core/types.py
+++ b/xarray/core/types.py
@@ -33,6 +33,16 @@
     except ImportError:
         DaskArray = np.ndarray  # type: ignore
 
+    try:
+        from cubed import Array as CubedArray
+    except ImportError:
+        CubedArray = np.ndarray
+
+    try:
+        from zarr.core import Array as ZarrArray
+    except ImportError:
+        ZarrArray = np.ndarray
+
     # TODO: Turn on when https://github.com/python/mypy/issues/11871 is fixed.
     # Can be uncommented if using pyright though.
     # import sys
@@ -105,6 +115,9 @@
 Dims = Union[str, Iterable[Hashable], "ellipsis", None]
 OrderedDims = Union[str, Sequence[Union[Hashable, "ellipsis"]], "ellipsis", None]
 
+T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
+T_NormalizedChunks = tuple[tuple[int, ...], ...]
+
 ErrorOptions = Literal["raise", "ignore"]
 ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]
 
diff --git a/xarray/core/utils.py b/xarray/core/utils.py
--- a/xarray/core/utils.py
+++ b/xarray/core/utils.py
@@ -1202,3 +1202,66 @@ def emit_user_level_warning(message, category=None):
     """Emit a warning at the user level by inspecting the stack trace."""
     stacklevel = find_stack_level()
     warnings.warn(message, category=category, stacklevel=stacklevel)
+
+
+def consolidate_dask_from_array_kwargs(
+    from_array_kwargs: dict,
+    name: str | None = None,
+    lock: bool | None = None,
+    inline_array: bool | None = None,
+) -> dict:
+    """
+    Merge dask-specific kwargs with arbitrary from_array_kwargs dict.
+
+    Temporary function, to be deleted once explicitly passing dask-specific kwargs to .chunk() is deprecated.
+    """
+
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="name",
+        passed_kwarg_value=name,
+        default=None,
+        err_msg_dict_name="from_array_kwargs",
+    )
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="lock",
+        passed_kwarg_value=lock,
+        default=False,
+        err_msg_dict_name="from_array_kwargs",
+    )
+    from_array_kwargs = _resolve_doubly_passed_kwarg(
+        from_array_kwargs,
+        kwarg_name="inline_array",
+        passed_kwarg_value=inline_array,
+        default=False,
+        err_msg_dict_name="from_array_kwargs",
+    )
+
+    return from_array_kwargs
+
+
+def _resolve_doubly_passed_kwarg(
+    kwargs_dict: dict,
+    kwarg_name: str,
+    passed_kwarg_value: str | bool | None,
+    default: bool | None,
+    err_msg_dict_name: str,
+) -> dict:
+    # if in kwargs_dict but not passed explicitly then just pass kwargs_dict through unaltered
+    if kwarg_name in kwargs_dict and passed_kwarg_value is None:
+        pass
+    # if passed explicitly but not in kwargs_dict then use that
+    elif kwarg_name not in kwargs_dict and passed_kwarg_value is not None:
+        kwargs_dict[kwarg_name] = passed_kwarg_value
+    # if in neither then use default
+    elif kwarg_name not in kwargs_dict and passed_kwarg_value is None:
+        kwargs_dict[kwarg_name] = default
+    # if in both then raise
+    else:
+        raise ValueError(
+            f"argument {kwarg_name} cannot be passed both as a keyword argument and within "
+            f"the {err_msg_dict_name} dictionary"
+        )
+
+    return kwargs_dict
diff --git a/xarray/core/variable.py b/xarray/core/variable.py
--- a/xarray/core/variable.py
+++ b/xarray/core/variable.py
@@ -26,10 +26,15 @@
     as_indexable,
 )
 from xarray.core.options import OPTIONS, _get_keep_attrs
+from xarray.core.parallelcompat import (
+    get_chunked_array_type,
+    guess_chunkmanager,
+)
 from xarray.core.pycompat import (
     array_type,
     integer_types,
     is_0d_dask_array,
+    is_chunked_array,
     is_duck_dask_array,
 )
 from xarray.core.utils import (
@@ -54,6 +59,7 @@
 BASIC_INDEXING_TYPES = integer_types + (slice,)
 
 if TYPE_CHECKING:
+    from xarray.core.parallelcompat import ChunkManagerEntrypoint
     from xarray.core.types import (
         Dims,
         ErrorOptionsWithWarn,
@@ -194,10 +200,10 @@ def _as_nanosecond_precision(data):
             nanosecond_precision_dtype = pd.DatetimeTZDtype("ns", dtype.tz)
         else:
             nanosecond_precision_dtype = "datetime64[ns]"
-        return data.astype(nanosecond_precision_dtype)
+        return duck_array_ops.astype(data, nanosecond_precision_dtype)
     elif dtype.kind == "m" and dtype != np.dtype("timedelta64[ns]"):
         utils.emit_user_level_warning(NON_NANOSECOND_WARNING.format(case="timedelta"))
-        return data.astype("timedelta64[ns]")
+        return duck_array_ops.astype(data, "timedelta64[ns]")
     else:
         return data
 
@@ -368,7 +374,7 @@ def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
             self.encoding = encoding
 
     @property
-    def dtype(self):
+    def dtype(self) -> np.dtype:
         """
         Data-type of the array’s elements.
 
@@ -380,7 +386,7 @@ def dtype(self):
         return self._data.dtype
 
     @property
-    def shape(self):
+    def shape(self) -> tuple[int, ...]:
         """
         Tuple of array dimensions.
 
@@ -533,8 +539,10 @@ def load(self, **kwargs):
         --------
         dask.array.compute
         """
-        if is_duck_dask_array(self._data):
-            self._data = as_compatible_data(self._data.compute(**kwargs))
+        if is_chunked_array(self._data):
+            chunkmanager = get_chunked_array_type(self._data)
+            loaded_data, *_ = chunkmanager.compute(self._data, **kwargs)
+            self._data = as_compatible_data(loaded_data)
         elif isinstance(self._data, indexing.ExplicitlyIndexed):
             self._data = self._data.get_duck_array()
         elif not is_duck_array(self._data):
@@ -1166,8 +1174,10 @@ def chunk(
             | Mapping[Any, None | int | tuple[int, ...]]
         ) = {},
         name: str | None = None,
-        lock: bool = False,
-        inline_array: bool = False,
+        lock: bool | None = None,
+        inline_array: bool | None = None,
+        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
+        from_array_kwargs=None,
         **chunks_kwargs: Any,
     ) -> Variable:
         """Coerce this array's data into a dask array with the given chunks.
@@ -1188,12 +1198,21 @@ def chunk(
         name : str, optional
             Used to generate the name for this array in the internal dask
             graph. Does not need not be unique.
-        lock : optional
+        lock : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
-        inline_array: optional
+        inline_array : bool, default: False
             Passed on to :py:func:`dask.array.from_array`, if the array is not
             already as dask array.
+        chunked_array_type: str, optional
+            Which chunked array type to coerce this datasets' arrays to.
+            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntrypoint` system.
+            Experimental API that should not be relied upon.
+        from_array_kwargs: dict, optional
+            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
+            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
+            For example, with dask as the default chunked array type, this method would pass additional kwargs
+            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
         **chunks_kwargs : {dim: chunks, ...}, optional
             The keyword arguments form of ``chunks``.
             One of chunks or chunks_kwargs must be provided.
@@ -1209,7 +1228,6 @@ def chunk(
         xarray.unify_chunks
         dask.array.from_array
         """
-        import dask.array as da
 
         if chunks is None:
             warnings.warn(
@@ -1220,6 +1238,8 @@ def chunk(
             chunks = {}
 
         if isinstance(chunks, (float, str, int, tuple, list)):
+            # TODO we shouldn't assume here that other chunkmanagers can handle these types
+            # TODO should we call normalize_chunks here?
             pass  # dask.array.from_array can handle these directly
         else:
             chunks = either_dict_or_kwargs(chunks, chunks_kwargs, "chunk")
@@ -1227,9 +1247,22 @@ def chunk(
         if utils.is_dict_like(chunks):
             chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}
 
+        chunkmanager = guess_chunkmanager(chunked_array_type)
+
+        if from_array_kwargs is None:
+            from_array_kwargs = {}
+
+        # TODO deprecate passing these dask-specific arguments explicitly. In future just pass everything via from_array_kwargs
+        _from_array_kwargs = utils.consolidate_dask_from_array_kwargs(
+            from_array_kwargs,
+            name=name,
+            lock=lock,
+            inline_array=inline_array,
+        )
+
         data = self._data
-        if is_duck_dask_array(data):
-            data = data.rechunk(chunks)
+        if chunkmanager.is_chunked_array(data):
+            data = chunkmanager.rechunk(data, chunks)  # type: ignore[arg-type]
         else:
             if isinstance(data, indexing.ExplicitlyIndexed):
                 # Unambiguously handle array storage backends (like NetCDF4 and h5py)
@@ -1244,17 +1277,13 @@ def chunk(
                     data, indexing.OuterIndexer
                 )
 
-                # All of our lazily loaded backend array classes should use NumPy
-                # array operations.
-                kwargs = {"meta": np.ndarray}
-            else:
-                kwargs = {}
-
             if utils.is_dict_like(chunks):
-                chunks = tuple(chunks.get(n, s) for n, s in enumerate(self.shape))
+                chunks = tuple(chunks.get(n, s) for n, s in enumerate(data.shape))
 
-            data = da.from_array(
-                data, chunks, name=name, lock=lock, inline_array=inline_array, **kwargs
+            data = chunkmanager.from_array(
+                data,
+                chunks,  # type: ignore[arg-type]
+                **_from_array_kwargs,
             )
 
         return self._replace(data=data)
@@ -1266,7 +1295,8 @@ def to_numpy(self) -> np.ndarray:
 
         # TODO first attempt to call .to_numpy() once some libraries implement it
         if hasattr(data, "chunks"):
-            data = data.compute()
+            chunkmanager = get_chunked_array_type(data)
+            data, *_ = chunkmanager.compute(data)
         if isinstance(data, array_type("cupy")):
             data = data.get()
         # pint has to be imported dynamically as pint imports xarray
@@ -2903,7 +2933,15 @@ def values(self, values):
             f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
         )
 
-    def chunk(self, chunks={}, name=None, lock=False, inline_array=False):
+    def chunk(
+        self,
+        chunks={},
+        name=None,
+        lock=False,
+        inline_array=False,
+        chunked_array_type=None,
+        from_array_kwargs=None,
+    ):
         # Dummy - do not chunk. This method is invoked e.g. by Dataset.chunk()
         return self.copy(deep=False)
 
diff --git a/xarray/core/weighted.py b/xarray/core/weighted.py
--- a/xarray/core/weighted.py
+++ b/xarray/core/weighted.py
@@ -238,7 +238,10 @@ def _sum_of_weights(self, da: DataArray, dim: Dims = None) -> DataArray:
         # (and not 2); GH4074
         if self.weights.dtype == bool:
             sum_of_weights = self._reduce(
-                mask, self.weights.astype(int), dim=dim, skipna=False
+                mask,
+                duck_array_ops.astype(self.weights, dtype=int),
+                dim=dim,
+                skipna=False,
             )
         else:
             sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)

```

## Test Patch

```diff
diff --git a/xarray/tests/test_dask.py b/xarray/tests/test_dask.py
--- a/xarray/tests/test_dask.py
+++ b/xarray/tests/test_dask.py
@@ -904,13 +904,12 @@ def test_to_dask_dataframe_dim_order(self):
 
 @pytest.mark.parametrize("method", ["load", "compute"])
 def test_dask_kwargs_variable(method):
-    x = Variable("y", da.from_array(np.arange(3), chunks=(2,)))
-    # args should be passed on to da.Array.compute()
-    with mock.patch.object(
-        da.Array, "compute", return_value=np.arange(3)
-    ) as mock_compute:
+    chunked_array = da.from_array(np.arange(3), chunks=(2,))
+    x = Variable("y", chunked_array)
+    # args should be passed on to dask.compute() (via DaskManager.compute())
+    with mock.patch.object(da, "compute", return_value=(np.arange(3),)) as mock_compute:
         getattr(x, method)(foo="bar")
-    mock_compute.assert_called_with(foo="bar")
+    mock_compute.assert_called_with(chunked_array, foo="bar")
 
 
 @pytest.mark.parametrize("method", ["load", "compute", "persist"])
diff --git a/xarray/tests/test_parallelcompat.py b/xarray/tests/test_parallelcompat.py
new file mode 100644
--- /dev/null
+++ b/xarray/tests/test_parallelcompat.py
@@ -0,0 +1,219 @@
+from __future__ import annotations
+
+from typing import Any
+
+import numpy as np
+import pytest
+
+from xarray.core.daskmanager import DaskManager
+from xarray.core.parallelcompat import (
+    ChunkManagerEntrypoint,
+    get_chunked_array_type,
+    guess_chunkmanager,
+    list_chunkmanagers,
+)
+from xarray.core.types import T_Chunks, T_NormalizedChunks
+from xarray.tests import has_dask, requires_dask
+
+
+class DummyChunkedArray(np.ndarray):
+    """
+    Mock-up of a chunked array class.
+
+    Adds a (non-functional) .chunks attribute by following this example in the numpy docs
+    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
+    """
+
+    chunks: T_NormalizedChunks
+
+    def __new__(
+        cls,
+        shape,
+        dtype=float,
+        buffer=None,
+        offset=0,
+        strides=None,
+        order=None,
+        chunks=None,
+    ):
+        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
+        obj.chunks = chunks
+        return obj
+
+    def __array_finalize__(self, obj):
+        if obj is None:
+            return
+        self.chunks = getattr(obj, "chunks", None)
+
+    def rechunk(self, chunks, **kwargs):
+        copied = self.copy()
+        copied.chunks = chunks
+        return copied
+
+
+class DummyChunkManager(ChunkManagerEntrypoint):
+    """Mock-up of ChunkManager class for DummyChunkedArray"""
+
+    def __init__(self):
+        self.array_cls = DummyChunkedArray
+
+    def is_chunked_array(self, data: Any) -> bool:
+        return isinstance(data, DummyChunkedArray)
+
+    def chunks(self, data: DummyChunkedArray) -> T_NormalizedChunks:
+        return data.chunks
+
+    def normalize_chunks(
+        self,
+        chunks: T_Chunks | T_NormalizedChunks,
+        shape: tuple[int, ...] | None = None,
+        limit: int | None = None,
+        dtype: np.dtype | None = None,
+        previous_chunks: T_NormalizedChunks | None = None,
+    ) -> T_NormalizedChunks:
+        from dask.array.core import normalize_chunks
+
+        return normalize_chunks(chunks, shape, limit, dtype, previous_chunks)
+
+    def from_array(
+        self, data: np.ndarray, chunks: T_Chunks, **kwargs
+    ) -> DummyChunkedArray:
+        from dask import array as da
+
+        return da.from_array(data, chunks, **kwargs)
+
+    def rechunk(self, data: DummyChunkedArray, chunks, **kwargs) -> DummyChunkedArray:
+        return data.rechunk(chunks, **kwargs)
+
+    def compute(self, *data: DummyChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
+        from dask.array import compute
+
+        return compute(*data, **kwargs)
+
+    def apply_gufunc(
+        self,
+        func,
+        signature,
+        *args,
+        axes=None,
+        axis=None,
+        keepdims=False,
+        output_dtypes=None,
+        output_sizes=None,
+        vectorize=None,
+        allow_rechunk=False,
+        meta=None,
+        **kwargs,
+    ):
+        from dask.array.gufunc import apply_gufunc
+
+        return apply_gufunc(
+            func,
+            signature,
+            *args,
+            axes=axes,
+            axis=axis,
+            keepdims=keepdims,
+            output_dtypes=output_dtypes,
+            output_sizes=output_sizes,
+            vectorize=vectorize,
+            allow_rechunk=allow_rechunk,
+            meta=meta,
+            **kwargs,
+        )
+
+
+@pytest.fixture
+def register_dummy_chunkmanager(monkeypatch):
+    """
+    Mocks the registering of an additional ChunkManagerEntrypoint.
+
+    This preserves the presence of the existing DaskManager, so a test that relies on this and DaskManager both being
+    returned from list_chunkmanagers() at once would still work.
+
+    The monkeypatching changes the behavior of list_chunkmanagers when called inside xarray.core.parallelcompat,
+    but not when called from this tests file.
+    """
+    # Should include DaskManager iff dask is available to be imported
+    preregistered_chunkmanagers = list_chunkmanagers()
+
+    monkeypatch.setattr(
+        "xarray.core.parallelcompat.list_chunkmanagers",
+        lambda: {"dummy": DummyChunkManager()} | preregistered_chunkmanagers,
+    )
+    yield
+
+
+class TestGetChunkManager:
+    def test_get_chunkmanger(self, register_dummy_chunkmanager) -> None:
+        chunkmanager = guess_chunkmanager("dummy")
+        assert isinstance(chunkmanager, DummyChunkManager)
+
+    def test_fail_on_nonexistent_chunkmanager(self) -> None:
+        with pytest.raises(ValueError, match="unrecognized chunk manager foo"):
+            guess_chunkmanager("foo")
+
+    @requires_dask
+    def test_get_dask_if_installed(self) -> None:
+        chunkmanager = guess_chunkmanager(None)
+        assert isinstance(chunkmanager, DaskManager)
+
+    @pytest.mark.skipif(has_dask, reason="requires dask not to be installed")
+    def test_dont_get_dask_if_not_installed(self) -> None:
+        with pytest.raises(ValueError, match="unrecognized chunk manager dask"):
+            guess_chunkmanager("dask")
+
+    @requires_dask
+    def test_choose_dask_over_other_chunkmanagers(
+        self, register_dummy_chunkmanager
+    ) -> None:
+        chunk_manager = guess_chunkmanager(None)
+        assert isinstance(chunk_manager, DaskManager)
+
+
+class TestGetChunkedArrayType:
+    def test_detect_chunked_arrays(self, register_dummy_chunkmanager) -> None:
+        dummy_arr = DummyChunkedArray([1, 2, 3])
+
+        chunk_manager = get_chunked_array_type(dummy_arr)
+        assert isinstance(chunk_manager, DummyChunkManager)
+
+    def test_ignore_inmemory_arrays(self, register_dummy_chunkmanager) -> None:
+        dummy_arr = DummyChunkedArray([1, 2, 3])
+
+        chunk_manager = get_chunked_array_type(*[dummy_arr, 1.0, np.array([5, 6])])
+        assert isinstance(chunk_manager, DummyChunkManager)
+
+        with pytest.raises(TypeError, match="Expected a chunked array"):
+            get_chunked_array_type(5.0)
+
+    def test_raise_if_no_arrays_chunked(self, register_dummy_chunkmanager) -> None:
+        with pytest.raises(TypeError, match="Expected a chunked array "):
+            get_chunked_array_type(*[1.0, np.array([5, 6])])
+
+    def test_raise_if_no_matching_chunkmanagers(self) -> None:
+        dummy_arr = DummyChunkedArray([1, 2, 3])
+
+        with pytest.raises(
+            TypeError, match="Could not find a Chunk Manager which recognises"
+        ):
+            get_chunked_array_type(dummy_arr)
+
+    @requires_dask
+    def test_detect_dask_if_installed(self) -> None:
+        import dask.array as da
+
+        dask_arr = da.from_array([1, 2, 3], chunks=(1,))
+
+        chunk_manager = get_chunked_array_type(dask_arr)
+        assert isinstance(chunk_manager, DaskManager)
+
+    @requires_dask
+    def test_raise_on_mixed_array_types(self, register_dummy_chunkmanager) -> None:
+        import dask.array as da
+
+        dummy_arr = DummyChunkedArray([1, 2, 3])
+        dask_arr = da.from_array([1, 2, 3], chunks=(1,))
+
+        with pytest.raises(TypeError, match="received multiple types"):
+            get_chunked_array_type(*[dask_arr, dummy_arr])
diff --git a/xarray/tests/test_plugins.py b/xarray/tests/test_plugins.py
--- a/xarray/tests/test_plugins.py
+++ b/xarray/tests/test_plugins.py
@@ -236,6 +236,7 @@ def test_lazy_import() -> None:
         "sparse",
         "cupy",
         "pint",
+        "cubed",
     ]
     # ensure that none of the above modules has been imported before
     modules_backup = {}

```


## Code snippets

### 1 - xarray/core/parallel.py:

Start line: 477, End line: 556

```python
def map_blocks(
    func: Callable[..., T_Xarray],
    obj: DataArray | Dataset,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    template: DataArray | Dataset | None = None,
) -> T_Xarray:

    # iterate over all possible chunk combinations
    for chunk_tuple in itertools.product(*ichunk.values()):
        # mapping from dimension name to chunk index
        chunk_index = dict(zip(ichunk.keys(), chunk_tuple))

        blocked_args = [
            subset_dataset_to_block(graph, gname, arg, input_chunk_bounds, chunk_index)
            if isxr
            else arg
            for isxr, arg in zip(is_xarray, npargs)
        ]

        # expected["shapes", "coords", "data_vars", "indexes"] are used to
        # raise nice error messages in _wrapper
        expected = {}
        # input chunk 0 along a dimension maps to output chunk 0 along the same dimension
        # even if length of dimension is changed by the applied function
        expected["shapes"] = {
            k: output_chunks[k][v] for k, v in chunk_index.items() if k in output_chunks
        }
        expected["data_vars"] = set(template.data_vars.keys())  # type: ignore[assignment]
        expected["coords"] = set(template.coords.keys())  # type: ignore[assignment]
        expected["indexes"] = {
            dim: indexes[dim][_get_chunk_slicer(dim, chunk_index, output_chunk_bounds)]
            for dim in indexes
        }

        from_wrapper = (gname,) + chunk_tuple
        graph[from_wrapper] = (_wrapper, func, blocked_args, kwargs, is_array, expected)

        # mapping from variable name to dask graph key
        var_key_map: dict[Hashable, str] = {}
        for name, variable in template.variables.items():
            if name in indexes:
                continue
            gname_l = f"{name}-{gname}"
            var_key_map[name] = gname_l

            key: tuple[Any, ...] = (gname_l,)
            for dim in variable.dims:
                if dim in chunk_index:
                    key += (chunk_index[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    # output can have new dimensions with exactly one chunk
                    key += (0,)

            # We're adding multiple new layers to the graph:
            # The first new layer is the result of the computation on
            # the array.
            # Then we add one layer per variable, which extracts the
            # result for that variable, and depends on just the first new
            # layer.
            new_layers[gname_l][key] = (operator.getitem, from_wrapper, name)

    hlg = HighLevelGraph.from_collections(
        gname,
        graph,
        dependencies=[arg for arg in npargs if dask.is_dask_collection(arg)],
    )

    # This adds in the getitems for each variable in the dataset.
    hlg = HighLevelGraph(
        {**hlg.layers, **new_layers},
        dependencies={
            **hlg.dependencies,
            **{name: {gname} for name in new_layers.keys()},
        },
    )

    # TODO: benbovy - flexible indexes: make it work with custom indexes
    # this will need to pass both indexes and coords to the Dataset constructor
    result = Dataset(
        coords={k: idx.to_pandas_index() for k, idx in indexes.items()},
        attrs=template.attrs,
    )

    for index in result._indexes:
        result[index].attrs = template[index].attrs
        result[index].encoding = template[index].encoding
  # ... other code
```
### 2 - xarray/core/types.py:

Start line: 1, End line: 97

```python
from __future__ import annotations

import datetime
from collections.abc import Hashable, Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    SupportsIndex,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from packaging.version import Version

if TYPE_CHECKING:
    from numpy._typing import _SupportsDType
    from numpy.typing import ArrayLike

    from xarray.backends.common import BackendEntrypoint
    from xarray.core.common import AbstractArray, DataWithCoords
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.groupby import DataArrayGroupBy, GroupBy
    from xarray.core.indexes import Index
    from xarray.core.variable import Variable

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

    # Anything that can be coerced to a shape tuple
    _ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
    _DTypeLikeNested = Any  # TODO: wait for support for recursive types

    # Xarray requires a Mapping[Hashable, dtype] in many places which
    # conflics with numpys own DTypeLike (with dtypes for fields).
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.DTypeLike
    # This is a copy of this DTypeLike that allows only non-Mapping dtypes.
    DTypeLikeSave = Union[
        np.dtype[Any],
        # default data type (float64)
        None,
        # array-scalar types and generic types
        type[Any],
        # character codes, type strings or comma-separated fields, e.g., 'float64'
        str,
        # (flexible_dtype, itemsize)
        tuple[_DTypeLikeNested, int],
        # (fixed_dtype, shape)
        tuple[_DTypeLikeNested, _ShapeLike],
        # (base_dtype, new_dtype)
        tuple[_DTypeLikeNested, _DTypeLikeNested],
        # because numpy does the same?
        list[Any],
        # anything with a dtype attribute
        _SupportsDType[np.dtype[Any]],
    ]
    try:
        from cftime import datetime as CFTimeDatetime
    except ImportError:
        CFTimeDatetime = Any
    DatetimeLike = Union[pd.Timestamp, datetime.datetime, np.datetime64, CFTimeDatetime]
else:
    Self: Any = None
    DTypeLikeSave: Any = None


T_Backend = TypeVar("T_Backend", bound="BackendEntrypoint")
T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")
T_Array = TypeVar("T_Array", bound="AbstractArray")
T_Index = TypeVar("T_Index", bound="Index")

T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", bound=Union["Dataset", "DataArray"])

# Maybe we rename this to T_Data or something less Fortran-y?
T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")
```
### 3 - xarray/core/parallel.py:

Start line: 299, End line: 379

```python
def map_blocks(
    func: Callable[..., T_Xarray],
    obj: DataArray | Dataset,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    template: DataArray | Dataset | None = None,
) -> T_Xarray:
    # ... other code

    if template is not None and not isinstance(template, (DataArray, Dataset)):
        raise TypeError(
            f"template must be a DataArray or Dataset. Received {type(template).__name__} instead."
        )
    if not isinstance(args, Sequence):
        raise TypeError("args must be a sequence (for example, a list or tuple).")
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping (for example, a dict)")

    for value in kwargs.values():
        if is_dask_collection(value):
            raise TypeError(
                "Cannot pass dask collections in kwargs yet. Please compute or "
                "load values before passing to map_blocks."
            )

    if not is_dask_collection(obj):
        return func(obj, *args, **kwargs)

    try:
        import dask
        import dask.array
        from dask.highlevelgraph import HighLevelGraph

    except ImportError:
        pass

    all_args = [obj] + list(args)
    is_xarray = [isinstance(arg, (Dataset, DataArray)) for arg in all_args]
    is_array = [isinstance(arg, DataArray) for arg in all_args]

    # there should be a better way to group this. partition?
    xarray_indices, xarray_objs = unzip(
        (index, arg) for index, arg in enumerate(all_args) if is_xarray[index]
    )
    others = [
        (index, arg) for index, arg in enumerate(all_args) if not is_xarray[index]
    ]

    # all xarray objects must be aligned. This is consistent with apply_ufunc.
    aligned = align(*xarray_objs, join="exact")
    xarray_objs = tuple(
        dataarray_to_dataset(arg) if isinstance(arg, DataArray) else arg
        for arg in aligned
    )

    _, npargs = unzip(
        sorted(list(zip(xarray_indices, xarray_objs)) + others, key=lambda x: x[0])
    )

    # check that chunk sizes are compatible
    input_chunks = dict(npargs[0].chunks)
    input_indexes = dict(npargs[0]._indexes)
    for arg in xarray_objs[1:]:
        assert_chunks_compatible(npargs[0], arg)
        input_chunks.update(arg.chunks)
        input_indexes.update(arg._indexes)

    if template is None:
        # infer template by providing zero-shaped arrays
        template = infer_template(func, aligned[0], *args, **kwargs)
        template_indexes = set(template._indexes)
        preserved_indexes = template_indexes & set(input_indexes)
        new_indexes = template_indexes - set(input_indexes)
        indexes = {dim: input_indexes[dim] for dim in preserved_indexes}
        indexes.update({k: template._indexes[k] for k in new_indexes})
        output_chunks: Mapping[Hashable, tuple[int, ...]] = {
            dim: input_chunks[dim] for dim in template.dims if dim in input_chunks
        }

    else:
        # template xarray object has been provided with proper sizes and chunk shapes
        indexes = dict(template._indexes)
        output_chunks = template.chunksizes
        if not output_chunks:
            raise ValueError(
                "Provided template has no dask arrays. "
                " Please construct a template with appropriately chunked dask arrays."
            )
  # ... other code
```
### 4 - xarray/core/dataarray.py:

Start line: 1, End line: 102

```python
from __future__ import annotations

import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast, overload

import numpy as np
import pandas as pd

from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
)
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import DataArrayCoordinates, assert_coordinate_consistent
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import (
    Index,
    Indexes,
    PandasMultiIndex,
    filter_indexes_from_coords,
    isel_indexes,
)
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError, _create_indexes_from_coords
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from xarray.core.variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
)
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs

if TYPE_CHECKING:
    from typing import TypeVar, Union

    from numpy.typing import ArrayLike

    try:
        from dask.dataframe import DataFrame as DaskDataFrame
    except ImportError:
        DaskDataFrame = None  # type: ignore
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

    from xarray.backends import ZarrStore
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
    from xarray.core.groupby import DataArrayGroupBy
    from xarray.core.resample import DataArrayResample
    from xarray.core.rolling import DataArrayCoarsen, DataArrayRolling
    from xarray.core.types import (
        CoarsenBoundaryOptions,
        DatetimeLike,
        DatetimeUnitOptions,
        Dims,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        PadModeOptions,
        PadReflectOptions,
        QuantileMethods,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_DataArray,
        T_Xarray,
    )
    from xarray.core.weighted import DataArrayWeighted

    T_XarrayOther = TypeVar("T_XarrayOther", bound=Union["DataArray", Dataset])
```
### 5 - xarray/core/types.py:

Start line: 99, End line: 179

```python
ScalarOrArray = Union["ArrayLike", np.generic, np.ndarray, "DaskArray"]
DsCompatible = Union["Dataset", "DataArray", "Variable", "GroupBy", "ScalarOrArray"]
DaCompatible = Union["DataArray", "Variable", "DataArrayGroupBy", "ScalarOrArray"]
VarCompatible = Union["Variable", "ScalarOrArray"]
GroupByIncompatible = Union["Variable", "GroupBy"]

Dims = Union[str, Iterable[Hashable], "ellipsis", None]
OrderedDims = Union[str, Sequence[Union[Hashable, "ellipsis"]], "ellipsis", None]

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

PadModeOptions = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
]
PadReflectOptions = Literal["even", "odd", None]

CFCalendar = Literal[
    "standard",
    "gregorian",
    "proleptic_gregorian",
    "noleap",
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
]

CoarsenBoundaryOptions = Literal["exact", "trim", "pad"]
SideOptions = Literal["left", "right"]
InclusiveOptions = Literal["both", "neither", "left", "right"]

ScaleOptions = Literal["linear", "symlog", "log", "logit", None]
HueStyleOptions = Literal["continuous", "discrete", None]
AspectOptions = Union[Literal["auto", "equal"], float, None]
ExtendOptions = Literal["neither", "both", "min", "max", None]

# TODO: Wait until mypy supports recursive objects in combination with typevars
_T = TypeVar("_T")
NestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]
```
### 6 - xarray/core/common.py:

Start line: 1, End line: 59

```python
from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload

import numpy as np
import pandas as pd

from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.pycompat import is_duck_dask_array
from xarray.core.utils import (
    Frozen,
    either_dict_or_kwargs,
    emit_user_level_warning,
    is_scalar,
)

try:
    import cftime
except ImportError:
    cftime = None

# Used as a sentinel value to indicate a all dimensions
ALL_DIMS = ...


if TYPE_CHECKING:
    import datetime

    from numpy.typing import DTypeLike

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.indexes import Index
    from xarray.core.resample import Resample
    from xarray.core.rolling_exp import RollingExp
    from xarray.core.types import (
        DatetimeLike,
        DTypeLikeSave,
        ScalarOrArray,
        SideOptions,
        T_DataWithCoords,
        T_Variable,
    )
    from xarray.core.variable import Variable

    DTypeMaybeMapping = Union[DTypeLikeSave, Mapping[Any, DTypeLikeSave]]


T_Resample = TypeVar("T_Resample", bound="Resample")
C = TypeVar("C")
T = TypeVar("T")
```
### 7 - xarray/__init__.py:

Start line: 1, End line: 112

```python
from xarray import testing, tutorial
from xarray.backends.api import (
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    save_mfdataset,
)
from xarray.backends.zarr import open_zarr
from xarray.coding.cftime_offsets import cftime_range, date_range, date_range_like
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.frequencies import infer_freq
from xarray.conventions import SerializationWarning, decode_cf
from xarray.core.alignment import align, broadcast
from xarray.core.combine import combine_by_coords, combine_nested
from xarray.core.common import ALL_DIMS, full_like, ones_like, zeros_like
from xarray.core.computation import (
    apply_ufunc,
    corr,
    cov,
    cross,
    dot,
    polyval,
    unify_chunks,
    where,
)
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.extensions import (
    register_dataarray_accessor,
    register_dataset_accessor,
)
from xarray.core.merge import Context, MergeError, merge
from xarray.core.options import get_options, set_options
from xarray.core.parallel import map_blocks
from xarray.core.variable import Coordinate, IndexVariable, Variable, as_variable
from xarray.util.print_versions import show_versions

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version

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
### 8 - xarray/core/parallel.py:

Start line: 1, End line: 30

```python
from __future__ import annotations

import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from xarray.core.alignment import align
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.pycompat import is_dask_collection

if TYPE_CHECKING:
    from xarray.core.types import T_Xarray


def unzip(iterable):
    return zip(*iterable)


def assert_chunks_compatible(a: Dataset, b: Dataset):
    a = a.unify_chunks()
    b = b.unify_chunks()

    for dim in set(a.chunks).intersection(set(b.chunks)):
        if a.chunks[dim] != b.chunks[dim]:
            raise ValueError(f"Chunk sizes along dimension {dim!r} are not equal.")
```
### 9 - xarray/core/duck_array_ops.py:

Start line: 613, End line: 676

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


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis"""
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        if is_duck_dask_array(values):
            return dask_array_ops.nanfirst(values, axis)
        else:
            return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis"""
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        if is_duck_dask_array(values):
            return dask_array_ops.nanlast(values, axis)
        else:
            return nanlast(values, axis)
    return take(values, -1, axis=axis)


def least_squares(lhs, rhs, rcond=None, skipna=False):
    """Return the coefficients and residuals of a least-squares fit."""
    if is_duck_dask_array(rhs):
        return dask_array_ops.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)
    else:
        return nputils.least_squares(lhs, rhs, rcond=rcond, skipna=skipna)


def push(array, n, axis):
    from bottleneck import push

    if is_duck_dask_array(array):
        return dask_array_ops.push(array, n, axis)
    else:
        return push(array, n, axis)
```
### 10 - xarray/core/parallel.py:

Start line: 558, End line: 583

```python
def map_blocks(
    func: Callable[..., T_Xarray],
    obj: DataArray | Dataset,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    template: DataArray | Dataset | None = None,
) -> T_Xarray:
    # ... other code

    for name, gname_l in var_key_map.items():
        dims = template[name].dims
        var_chunks = []
        for dim in dims:
            if dim in output_chunks:
                var_chunks.append(output_chunks[dim])
            elif dim in result._indexes:
                var_chunks.append((result.sizes[dim],))
            elif dim in template.dims:
                # new unindexed dimension
                var_chunks.append((template.sizes[dim],))

        data = dask.array.Array(
            hlg, name=gname_l, chunks=var_chunks, dtype=template[name].dtype
        )
        result[name] = (dims, data, template[name].attrs)
        result[name].encoding = template[name].encoding

    result = result.set_coords(template._coord_names)

    if result_is_array:
        da = dataset_to_dataarray(result)
        da.name = template_name
        return da  # type: ignore[return-value]
    return result  # type: ignore[return-value]
```
### 11 - xarray/core/common.py:

Start line: 1397, End line: 1431

```python
@overload
def full_like(
    other: DataArray, fill_value: Any, dtype: DTypeLikeSave = None
) -> DataArray:
    ...


@overload
def full_like(
    other: Dataset, fill_value: Any, dtype: DTypeMaybeMapping = None
) -> Dataset:
    ...


@overload
def full_like(
    other: Variable, fill_value: Any, dtype: DTypeLikeSave = None
) -> Variable:
    ...


@overload
def full_like(
    other: Dataset | DataArray, fill_value: Any, dtype: DTypeMaybeMapping = None
) -> Dataset | DataArray:
    ...


@overload
def full_like(
    other: Dataset | DataArray | Variable,
    fill_value: Any,
    dtype: DTypeMaybeMapping = None,
) -> Dataset | DataArray | Variable:
    ...
```
### 12 - xarray/core/duck_array_ops.py:

Start line: 1, End line: 46

```python
"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import annotations

import contextlib
import datetime
import inspect
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import (  # noqa
    around,  # noqa
    einsum,
    gradient,
    isclose,
    isin,
    isnat,
    take,
    tensordot,
    transpose,
    unravel_index,
    zeros_like,  # noqa
)
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view  # noqa

from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.nputils import nanfirst, nanlast
from xarray.core.pycompat import array_type, is_duck_dask_array
from xarray.core.utils import is_duck_array, module_available

dask_available = module_available("dask")


def get_array_namespace(x):
    if hasattr(x, "__array_namespace__"):
        return x.__array_namespace__()
    else:
        return np
```
### 13 - xarray/core/dask_array_ops.py:

Start line: 101, End line: 132

```python
def _first_last_wrapper(array, *, axis, op, keepdims):
    return op(array, axis, keepdims=keepdims)


def _first_or_last(darray, axis, op):
    import dask.array

    # This will raise the same error message seen for numpy
    axis = normalize_axis_index(axis, darray.ndim)

    wrapped_op = partial(_first_last_wrapper, op=op)
    return dask.array.reduction(
        darray,
        chunk=wrapped_op,
        aggregate=wrapped_op,
        axis=axis,
        dtype=darray.dtype,
        keepdims=False,  # match numpy version
    )


def nanfirst(darray, axis):
    from xarray.core.duck_array_ops import nanfirst

    return _first_or_last(darray, axis, op=nanfirst)


def nanlast(darray, axis):
    from xarray.core.duck_array_ops import nanlast

    return _first_or_last(darray, axis, op=nanlast)
```
### 16 - xarray/core/computation.py:

Start line: 1088, End line: 1163

```python
def apply_ufunc(
    func: Callable,
    *args: Any,
    input_core_dims: Sequence[Sequence] | None = None,
    output_core_dims: Sequence[Sequence] | None = ((),),
    exclude_dims: Set = frozenset(),
    vectorize: bool = False,
    join: JoinOptions = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = _NO_FILL_VALUE,
    keep_attrs: bool | str | None = None,
    kwargs: Mapping | None = None,
    dask: str = "forbidden",
    output_dtypes: Sequence | None = None,
    output_sizes: Mapping[Any, int] | None = None,
    meta: Any = None,
    dask_gufunc_kwargs: dict[str, Any] | None = None,
) -> Any:
    from xarray.core.dataarray import DataArray
    from xarray.core.groupby import GroupBy
    from xarray.core.variable import Variable

    if input_core_dims is None:
        input_core_dims = ((),) * (len(args))
    elif len(input_core_dims) != len(args):
        raise ValueError(
            f"input_core_dims must be None or a tuple with the length same to "
            f"the number of arguments. "
            f"Given {len(input_core_dims)} input_core_dims: {input_core_dims}, "
            f" but number of args is {len(args)}."
        )

    if kwargs is None:
        kwargs = {}

    signature = _UFuncSignature(input_core_dims, output_core_dims)

    if exclude_dims:
        if not isinstance(exclude_dims, set):
            raise TypeError(
                f"Expected exclude_dims to be a 'set'. Received '{type(exclude_dims).__name__}' instead."
            )
        if not exclude_dims <= signature.all_core_dims:
            raise ValueError(
                f"each dimension in `exclude_dims` must also be a "
                f"core dimension in the function signature. "
                f"Please make {(exclude_dims - signature.all_core_dims)} a core dimension"
            )

    # handle dask_gufunc_kwargs
    if dask == "parallelized":
        if dask_gufunc_kwargs is None:
            dask_gufunc_kwargs = {}
        else:
            dask_gufunc_kwargs = dask_gufunc_kwargs.copy()
        # todo: remove warnings after deprecation cycle
        if meta is not None:
            warnings.warn(
                "``meta`` should be given in the ``dask_gufunc_kwargs`` parameter."
                " It will be removed as direct parameter in a future version.",
                FutureWarning,
                stacklevel=2,
            )
            dask_gufunc_kwargs.setdefault("meta", meta)
        if output_sizes is not None:
            warnings.warn(
                "``output_sizes`` should be given in the ``dask_gufunc_kwargs`` "
                "parameter. It will be removed as direct parameter in a future "
                "version.",
                FutureWarning,
                stacklevel=2,
            )
            dask_gufunc_kwargs.setdefault("output_sizes", output_sizes)

    if kwargs:
        func = functools.partial(func, **kwargs)

    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=False)

    if isinstance(keep_attrs, bool):
        keep_attrs = "override" if keep_attrs else "drop"

    variables_vfunc = functools.partial(
        apply_variable_ufunc,
        func,
        signature=signature,
        exclude_dims=exclude_dims,
        keep_attrs=keep_attrs,
        dask=dask,
        vectorize=vectorize,
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs=dask_gufunc_kwargs,
    )
    # ... other code
```
### 18 - xarray/core/computation.py:

Start line: 678, End line: 761

```python
def apply_variable_ufunc(
    func,
    *args,
    signature: _UFuncSignature,
    exclude_dims=frozenset(),
    dask="forbidden",
    output_dtypes=None,
    vectorize=False,
    keep_attrs="override",
    dask_gufunc_kwargs=None,
) -> Variable | tuple[Variable, ...]:
    # ... other code

    if any(is_duck_dask_array(array) for array in input_data):
        if dask == "forbidden":
            raise ValueError(
                "apply_ufunc encountered a dask array on an "
                "argument, but handling for dask arrays has not "
                "been enabled. Either set the ``dask`` argument "
                "or load your data into memory first with "
                "``.load()`` or ``.compute()``"
            )
        elif dask == "parallelized":
            numpy_func = func

            if dask_gufunc_kwargs is None:
                dask_gufunc_kwargs = {}
            else:
                dask_gufunc_kwargs = dask_gufunc_kwargs.copy()

            allow_rechunk = dask_gufunc_kwargs.get("allow_rechunk", None)
            if allow_rechunk is None:
                for n, (data, core_dims) in enumerate(
                    zip(input_data, signature.input_core_dims)
                ):
                    if is_duck_dask_array(data):
                        # core dimensions cannot span multiple chunks
                        for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                            if len(data.chunks[axis]) != 1:
                                raise ValueError(
                                    f"dimension {dim} on {n}th function argument to "
                                    "apply_ufunc with dask='parallelized' consists of "
                                    "multiple chunks, but is also a core dimension. To "
                                    "fix, either rechunk into a single dask array chunk along "
                                    f"this dimension, i.e., ``.chunk(dict({dim}=-1))``, or "
                                    "pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs`` "
                                    "but beware that this may significantly increase memory usage."
                                )
                dask_gufunc_kwargs["allow_rechunk"] = True

            output_sizes = dask_gufunc_kwargs.pop("output_sizes", {})
            if output_sizes:
                output_sizes_renamed = {}
                for key, value in output_sizes.items():
                    if key not in signature.all_output_core_dims:
                        raise ValueError(
                            f"dimension '{key}' in 'output_sizes' must correspond to output_core_dims"
                        )
                    output_sizes_renamed[signature.dims_map[key]] = value
                dask_gufunc_kwargs["output_sizes"] = output_sizes_renamed

            for key in signature.all_output_core_dims:
                if (
                    key not in signature.all_input_core_dims or key in exclude_dims
                ) and key not in output_sizes:
                    raise ValueError(
                        f"dimension '{key}' in 'output_core_dims' needs corresponding (dim, size) in 'output_sizes'"
                    )

            def func(*arrays):
                import dask.array as da

                res = da.apply_gufunc(
                    numpy_func,
                    signature.to_gufunc_string(exclude_dims),
                    *arrays,
                    vectorize=vectorize,
                    output_dtypes=output_dtypes,
                    **dask_gufunc_kwargs,
                )

                return res

        elif dask == "allowed":
            pass
        else:
            raise ValueError(
                "unknown setting for dask array handling in "
                "apply_ufunc: {}".format(dask)
            )
    else:
        if vectorize:
            func = _vectorize(
                func, signature, output_dtypes=output_dtypes, exclude_dims=exclude_dims
            )

    result_data = func(*input_data)
    # ... other code
```
### 19 - xarray/core/dataarray.py:

Start line: 1018, End line: 1119

```python
class DataArray(
    AbstractArray,
    DataWithCoords,
    DataArrayArithmetic,
    DataArrayAggregations,
):

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._variable, self._coords, self._name))

    def __dask_graph__(self):
        return self._to_temp_dataset().__dask_graph__()

    def __dask_keys__(self):
        return self._to_temp_dataset().__dask_keys__()

    def __dask_layers__(self):
        return self._to_temp_dataset().__dask_layers__()

    @property
    def __dask_optimize__(self):
        return self._to_temp_dataset().__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return self._to_temp_dataset().__dask_scheduler__

    def __dask_postcompute__(self):
        func, args = self._to_temp_dataset().__dask_postcompute__()
        return self._dask_finalize, (self.name, func) + args

    def __dask_postpersist__(self):
        func, args = self._to_temp_dataset().__dask_postpersist__()
        return self._dask_finalize, (self.name, func) + args

    @staticmethod
    def _dask_finalize(results, name, func, *args, **kwargs) -> DataArray:
        ds = func(results, *args, **kwargs)
        variable = ds._variables.pop(_THIS_ARRAY)
        coords = ds._variables
        indexes = ds._indexes
        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)

    def load(self: T_DataArray, **kwargs) -> T_DataArray:
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return this array.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        ds = self._to_temp_dataset().load(**kwargs)
        new = self._from_temp_dataset(ds)
        self._variable = new._variable
        self._coords = new._coords
        return self

    def compute(self: T_DataArray, **kwargs) -> T_DataArray:
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return a new array. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def persist(self: T_DataArray, **kwargs) -> T_DataArray:
        """Trigger computation in constituent dask arrays

        This keeps them as dask arrays but encourages them to keep data in
        memory.  This is particularly useful when on a distributed machine.
        When on a single machine consider using ``.compute()`` instead.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        ds = self._to_temp_dataset().persist(**kwargs)
        return self._from_temp_dataset(ds)
```
### 20 - xarray/core/duck_array_ops.py:

Start line: 49, End line: 68

```python
def _dask_or_eager_func(
    name,
    eager_module=np,
    dask_module="dask.array",
):
    """Create a function that dispatches to dask for dask array inputs."""

    def f(*args, **kwargs):
        if any(is_duck_dask_array(a) for a in args):
            mod = (
                import_module(dask_module)
                if isinstance(dask_module, str)
                else dask_module
            )
            wrapped = getattr(mod, name)
        else:
            wrapped = getattr(eager_module, name)
        return wrapped(*args, **kwargs)

    return f
```
### 22 - xarray/core/dask_array_ops.py:

Start line: 62, End line: 98

```python
def push(array, n, axis):
    """
    Dask-aware bottleneck.push
    """
    import bottleneck
    import dask.array as da
    import numpy as np

    def _fill_with_last_one(a, b):
        # cumreduction apply the push func over all the blocks first so, the only missing part is filling
        # the missing values using the last data of the previous chunk
        return np.where(~np.isnan(b), b, a)

    if n is not None and 0 < n < array.shape[axis] - 1:
        arange = da.broadcast_to(
            da.arange(
                array.shape[axis], chunks=array.chunks[axis], dtype=array.dtype
            ).reshape(
                tuple(size if i == axis else 1 for i, size in enumerate(array.shape))
            ),
            array.shape,
            array.chunks,
        )
        valid_arange = da.where(da.notnull(array), arange, np.nan)
        valid_limits = (arange - push(valid_arange, None, axis)) <= n
        # omit the forward fill that violate the limit
        return da.where(valid_limits, push(array, None, axis), np.nan)

    # The method parameter makes that the tests for python 3.7 fails.
    return da.reductions.cumreduction(
        func=bottleneck.push,
        binop=_fill_with_last_one,
        ident=np.nan,
        x=array,
        axis=axis,
        dtype=array.dtype,
    )
```
### 23 - xarray/core/computation.py:

Start line: 1707, End line: 1770

```python
def dot(
    *arrays,
    dims: Dims = None,
    **kwargs: Any,
):
    # ... other code

    if any(not isinstance(arr, (Variable, DataArray)) for arr in arrays):
        raise TypeError(
            "Only xr.DataArray and xr.Variable are supported."
            "Given {}.".format([type(arr) for arr in arrays])
        )

    if len(arrays) == 0:
        raise TypeError("At least one array should be given.")

    common_dims: set[Hashable] = set.intersection(*(set(arr.dims) for arr in arrays))
    all_dims = []
    for arr in arrays:
        all_dims += [d for d in arr.dims if d not in all_dims]

    einsum_axes = "abcdefghijklmnopqrstuvwxyz"
    dim_map = {d: einsum_axes[i] for i, d in enumerate(all_dims)}

    if dims is ...:
        dims = all_dims
    elif isinstance(dims, str):
        dims = (dims,)
    elif dims is None:
        # find dimensions that occur more than one times
        dim_counts: Counter = Counter()
        for arr in arrays:
            dim_counts.update(arr.dims)
        dims = tuple(d for d, c in dim_counts.items() if c > 1)

    dot_dims: set[Hashable] = set(dims)

    # dimensions to be parallelized
    broadcast_dims = common_dims - dot_dims
    input_core_dims = [
        [d for d in arr.dims if d not in broadcast_dims] for arr in arrays
    ]
    output_core_dims = [
        [d for d in all_dims if d not in dot_dims and d not in broadcast_dims]
    ]

    # construct einsum subscripts, such as '...abc,...ab->...c'
    # Note: input_core_dims are always moved to the last position
    subscripts_list = [
        "..." + "".join(dim_map[d] for d in ds) for ds in input_core_dims
    ]
    subscripts = ",".join(subscripts_list)
    subscripts += "->..." + "".join(dim_map[d] for d in output_core_dims[0])

    join = OPTIONS["arithmetic_join"]
    # using "inner" emulates `(a * b).sum()` for all joins (except "exact")
    if join != "exact":
        join = "inner"

    # subscripts should be passed to np.einsum as arg, not as kwargs. We need
    # to construct a partial function for apply_ufunc to work.
    func = functools.partial(duck_array_ops.einsum, subscripts, **kwargs)
    result = apply_ufunc(
        func,
        *arrays,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        join=join,
        dask="allowed",
    )
    return result.transpose(*all_dims, missing_dims="ignore")
```
### 24 - xarray/backends/api.py:

Start line: 1, End line: 68

```python
from __future__ import annotations

import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from glob import glob
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Union, cast, overload

import numpy as np

from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import AbstractDataStore, ArrayWriter, _normalize_path
from xarray.backends.locks import _get_scheduler
from xarray.core import indexing
from xarray.core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.utils import is_remote_uri

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None  # type: ignore
    from io import BufferedIOBase

    from xarray.backends.common import BackendEntrypoint
    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
        NestedSequence,
    )

    T_NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
    T_Engine = Union[
        T_NetcdfEngine,
        Literal["pydap", "pynio", "pseudonetcdf", "zarr"],
        type[BackendEntrypoint],
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
    "zarr": backends.ZarrStore.open_group,
}
```
### 25 - xarray/core/indexing.py:

Start line: 1145, End line: 1158

```python
def _dask_array_with_chunks_hint(array, chunks):
    """Create a dask array using the chunks hint for dimensions of size > 1."""
    import dask.array as da

    if len(chunks) < array.ndim:
        raise ValueError("not enough chunks in hint")
    new_chunks = []
    for chunk, size in zip(chunks, array.shape):
        new_chunks.append(chunk if size > 1 else (1,))
    return da.from_array(array, new_chunks)


def _logical_any(args):
    return functools.reduce(operator.or_, args)
```
### 26 - xarray/core/dataset.py:

Start line: 1, End line: 99

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
from collections.abc import (
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload

import numpy as np
import pandas as pd

from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
    alignment,
    duck_array_ops,
    formatting,
    formatting_html,
    ops,
    utils,
)
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
)
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
    DataWithCoords,
    _contains_datetime_like_objects,
    get_chunksizes,
)
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import DatasetCoordinates, assert_coordinate_consistent
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
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
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.pycompat import array_type, is_duck_array, is_duck_dask_array
from xarray.core.types import QuantileMethods, T_Dataset
from xarray.core.utils import (
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
from xarray.core.variable import (
    IndexVariable,
    Variable,
    as_variable,
    broadcast_variables,
    calculate_dimensions,
)
from xarray.plot.accessor import DatasetPlotAccessor
```
