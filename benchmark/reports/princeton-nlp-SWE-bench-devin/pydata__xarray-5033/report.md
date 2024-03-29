# pydata__xarray-5033

| **pydata/xarray** | `f94de6b4504482ab206f93ec800608f2e1f47b19` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 3043 |
| **Any found context length** | 624 |
| **Avg pos** | 15.0 |
| **Min pos** | 1 |
| **Max pos** | 21 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -375,10 +375,11 @@ def open_dataset(
         scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
         objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
     engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
-        "pseudonetcdf", "zarr"}, optional
+        "pseudonetcdf", "zarr"} or subclass of xarray.backends.BackendEntrypoint, optional
         Engine to use when reading files. If not provided, the default engine
         is chosen based on available dependencies, with a preference for
-        "netcdf4".
+        "netcdf4". A custom backend class (a subclass of ``BackendEntrypoint``)
+        can also be used.
     chunks : int or dict, optional
         If chunks is provided, it is used to load the new dataset into dask
         arrays. ``chunks=-1`` loads the dataset with dask using a single
diff --git a/xarray/backends/plugins.py b/xarray/backends/plugins.py
--- a/xarray/backends/plugins.py
+++ b/xarray/backends/plugins.py
@@ -5,7 +5,7 @@
 
 import pkg_resources
 
-from .common import BACKEND_ENTRYPOINTS
+from .common import BACKEND_ENTRYPOINTS, BackendEntrypoint
 
 STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]
 
@@ -113,10 +113,22 @@ def guess_engine(store_spec):
 
 
 def get_backend(engine):
-    """Select open_dataset method based on current engine"""
-    engines = list_engines()
-    if engine not in engines:
-        raise ValueError(
-            f"unrecognized engine {engine} must be one of: {list(engines)}"
+    """Select open_dataset method based on current engine."""
+    if isinstance(engine, str):
+        engines = list_engines()
+        if engine not in engines:
+            raise ValueError(
+                f"unrecognized engine {engine} must be one of: {list(engines)}"
+            )
+        backend = engines[engine]
+    elif isinstance(engine, type) and issubclass(engine, BackendEntrypoint):
+        backend = engine
+    else:
+        raise TypeError(
+            (
+                "engine must be a string or a subclass of "
+                f"xarray.backends.BackendEntrypoint: {engine}"
+            )
         )
-    return engines[engine]
+
+    return backend

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/backends/api.py | 378 | 381 | 21 | 2 | 9379
| xarray/backends/plugins.py | 8 | 8 | 8 | 1 | 3043
| xarray/backends/plugins.py | 116 | 122 | 1 | 1 | 624


## Problem Statement

```
Simplify adding custom backends
<!-- Please do a quick search of existing issues to make sure that this has not been asked before. -->

**Is your feature request related to a problem? Please describe.**
I've been working on opening custom hdf formats in xarray, reading up on the apiv2 it is currently only possible to declare a new external plugin in setup.py but that doesn't seem easy or intuitive to me.

**Describe the solution you'd like**
Why can't we simply be allowed to add functions to the engine parameter? Example:
\`\`\`python
from custom_backend import engine

ds = xr.load_dataset(filename, engine=engine)
\`\`\`
This seems like a small function change to me from my initial _quick_ look because there's mainly a bunch of string checks in the normal case until we get to the registered backend functions, if we send in a function instead in the engine-parameter we can just bypass those checks.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 xarray/backends/plugins.py** | 35 | 123| 624 | 624 | 872 | 
| 2 | **2 xarray/backends/api.py** | 1 | 119| 713 | 1337 | 12940 | 
| 3 | **2 xarray/backends/api.py** | 311 | 347| 218 | 1555 | 12940 | 
| 4 | **2 xarray/backends/api.py** | 282 | 308| 162 | 1717 | 12940 | 
| 5 | **2 xarray/backends/api.py** | 923 | 977| 420 | 2137 | 12940 | 
| 6 | **2 xarray/backends/api.py** | 480 | 527| 353 | 2490 | 12940 | 
| 7 | 3 xarray/backends/common.py | 346 | 386| 306 | 2796 | 15415 | 
| **-> 8 <-** | **3 xarray/backends/plugins.py** | 1 | 32| 247 | 3043 | 15415 | 
| 9 | **3 xarray/backends/api.py** | 855 | 922| 652 | 3695 | 15415 | 
| 10 | 4 xarray/backends/scipy_.py | 235 | 286| 308 | 4003 | 17391 | 
| 11 | **4 xarray/backends/api.py** | 661 | 970| 459 | 4462 | 17391 | 
| 12 | 5 xarray/backends/pydap_.py | 110 | 148| 231 | 4693 | 18308 | 
| 13 | 6 doc/conf.py | 1 | 101| 670 | 5363 | 21638 | 
| 14 | 7 xarray/backends/pynio_.py | 100 | 137| 216 | 5579 | 22511 | 
| 15 | 8 xarray/backends/cfgrib_.py | 96 | 151| 355 | 5934 | 23481 | 
| 16 | **8 xarray/backends/api.py** | 189 | 227| 259 | 6193 | 23481 | 
| 17 | 9 xarray/backends/zarr.py | 685 | 732| 277 | 6470 | 29069 | 
| 18 | 9 doc/conf.py | 102 | 172| 767 | 7237 | 29069 | 
| 19 | 10 xarray/backends/pseudonetcdf_.py | 103 | 155| 322 | 7559 | 30060 | 
| 20 | 11 xarray/backends/h5netcdf_.py | 336 | 396| 353 | 7912 | 32827 | 
| **-> 21 <-** | **11 xarray/backends/api.py** | 350 | 479| 1467 | 9379 | 32827 | 
| 22 | 11 doc/conf.py | 260 | 309| 586 | 9965 | 32827 | 
| 23 | 12 xarray/backends/file_manager.py | 1 | 21| 139 | 10104 | 35389 | 
| 24 | **12 xarray/backends/api.py** | 1203 | 1246| 320 | 10424 | 35389 | 
| 25 | 13 xarray/__init__.py | 1 | 93| 603 | 11027 | 35992 | 
| 26 | **13 xarray/backends/api.py** | 1273 | 1362| 789 | 11816 | 35992 | 
| 27 | 13 doc/conf.py | 173 | 259| 747 | 12563 | 35992 | 
| 28 | 14 xarray/tutorial.py | 1 | 38| 208 | 12771 | 36991 | 
| 29 | 15 xarray/backends/store.py | 1 | 46| 259 | 13030 | 37251 | 
| 30 | 15 xarray/backends/pynio_.py | 1 | 27| 186 | 13216 | 37251 | 
| 31 | 16 xarray/core/common.py | 511 | 643| 1180 | 14396 | 52953 | 
| 32 | 17 xarray/core/options.py | 1 | 78| 547 | 14943 | 54310 | 
| 33 | 18 xarray/core/variable.py | 1086 | 1114| 242 | 15185 | 77030 | 
| 34 | 19 xarray/plot/dataset_plot.py | 277 | 389| 828 | 16013 | 81716 | 
| 35 | 20 xarray/backends/netCDF4_.py | 515 | 574| 353 | 16366 | 85822 | 
| 36 | 21 xarray/plot/plot.py | 636 | 793| 1739 | 18105 | 94176 | 
| 37 | 22 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 18838 | 95736 | 
| 38 | 22 xarray/backends/pydap_.py | 1 | 21| 118 | 18956 | 95736 | 
| 39 | 22 xarray/core/common.py | 1506 | 1522| 107 | 19063 | 95736 | 
| 40 | 22 xarray/backends/zarr.py | 643 | 682| 311 | 19374 | 95736 | 
| 41 | 22 xarray/backends/scipy_.py | 1 | 37| 251 | 19625 | 95736 | 
| 42 | **22 xarray/backends/api.py** | 530 | 660| 1491 | 21116 | 95736 | 
| 43 | 22 xarray/backends/pydap_.py | 54 | 70| 132 | 21248 | 95736 | 
| 44 | 23 xarray/backends/__init__.py | 1 | 37| 297 | 21545 | 96033 | 
| 45 | **23 xarray/backends/api.py** | 723 | 854| 1610 | 23155 | 96033 | 
| 46 | 23 xarray/backends/common.py | 128 | 167| 245 | 23400 | 96033 | 
| 47 | 24 xarray/core/duck_array_ops.py | 33 | 63| 209 | 23609 | 101247 | 
| 48 | 24 xarray/backends/scipy_.py | 73 | 107| 300 | 23909 | 101247 | 
| 49 | 25 asv_bench/benchmarks/interp.py | 41 | 56| 162 | 24071 | 101732 | 
| 50 | 25 doc/conf.py | 310 | 322| 204 | 24275 | 101732 | 
| 51 | 25 xarray/plot/dataset_plot.py | 1 | 101| 728 | 25003 | 101732 | 
| 52 | 26 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 25534 | 102263 | 
| 53 | 26 xarray/backends/zarr.py | 1 | 30| 154 | 25688 | 102263 | 
| 54 | 27 xarray/core/indexing.py | 1030 | 1097| 746 | 26434 | 114224 | 
| 55 | 27 xarray/backends/file_manager.py | 299 | 334| 204 | 26638 | 114224 | 
| 56 | 27 xarray/backends/common.py | 334 | 343| 109 | 26747 | 114224 | 
| 57 | 27 xarray/backends/h5netcdf_.py | 1 | 33| 197 | 26944 | 114224 | 
| 58 | 28 xarray/core/parallel.py | 482 | 551| 730 | 27674 | 119270 | 
| 59 | 29 asv_bench/benchmarks/rolling.py | 64 | 86| 215 | 27889 | 120263 | 
| 60 | 29 xarray/tutorial.py | 41 | 114| 480 | 28369 | 120263 | 
| 61 | 29 xarray/core/options.py | 81 | 164| 810 | 29179 | 120263 | 
| 62 | 29 xarray/plot/plot.py | 432 | 472| 293 | 29472 | 120263 | 
| 63 | 30 xarray/plot/utils.py | 1 | 54| 283 | 29755 | 127002 | 
| 64 | 30 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 29939 | 127002 | 
| 65 | 31 asv_bench/benchmarks/pandas.py | 1 | 25| 160 | 30099 | 127162 | 
| 66 | 32 asv_bench/benchmarks/dataset_io.py | 93 | 123| 269 | 30368 | 130743 | 
| 67 | 33 conftest.py | 1 | 42| 278 | 30646 | 131021 | 
| 68 | 34 xarray/core/missing.py | 766 | 779| 182 | 30828 | 137120 | 
| 69 | 35 xarray/core/dataset.py | 503 | 514| 177 | 31005 | 198815 | 
| 70 | 35 asv_bench/benchmarks/indexing.py | 99 | 122| 231 | 31236 | 198815 | 
| 71 | **35 xarray/backends/api.py** | 980 | 1073| 766 | 32002 | 198815 | 
| 72 | 35 xarray/backends/netCDF4_.py | 1 | 40| 265 | 32267 | 198815 | 
| 73 | 35 asv_bench/benchmarks/dataset_io.py | 395 | 426| 264 | 32531 | 198815 | 
| 74 | 35 asv_bench/benchmarks/dataset_io.py | 429 | 464| 186 | 32717 | 198815 | 
| 75 | 35 xarray/core/variable.py | 2303 | 2321| 174 | 32891 | 198815 | 
| 76 | 35 xarray/backends/pydap_.py | 24 | 51| 215 | 33106 | 198815 | 
| 77 | 35 asv_bench/benchmarks/indexing.py | 79 | 96| 153 | 33259 | 198815 | 


## Patch

```diff
diff --git a/xarray/backends/api.py b/xarray/backends/api.py
--- a/xarray/backends/api.py
+++ b/xarray/backends/api.py
@@ -375,10 +375,11 @@ def open_dataset(
         scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
         objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
     engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
-        "pseudonetcdf", "zarr"}, optional
+        "pseudonetcdf", "zarr"} or subclass of xarray.backends.BackendEntrypoint, optional
         Engine to use when reading files. If not provided, the default engine
         is chosen based on available dependencies, with a preference for
-        "netcdf4".
+        "netcdf4". A custom backend class (a subclass of ``BackendEntrypoint``)
+        can also be used.
     chunks : int or dict, optional
         If chunks is provided, it is used to load the new dataset into dask
         arrays. ``chunks=-1`` loads the dataset with dask using a single
diff --git a/xarray/backends/plugins.py b/xarray/backends/plugins.py
--- a/xarray/backends/plugins.py
+++ b/xarray/backends/plugins.py
@@ -5,7 +5,7 @@
 
 import pkg_resources
 
-from .common import BACKEND_ENTRYPOINTS
+from .common import BACKEND_ENTRYPOINTS, BackendEntrypoint
 
 STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]
 
@@ -113,10 +113,22 @@ def guess_engine(store_spec):
 
 
 def get_backend(engine):
-    """Select open_dataset method based on current engine"""
-    engines = list_engines()
-    if engine not in engines:
-        raise ValueError(
-            f"unrecognized engine {engine} must be one of: {list(engines)}"
+    """Select open_dataset method based on current engine."""
+    if isinstance(engine, str):
+        engines = list_engines()
+        if engine not in engines:
+            raise ValueError(
+                f"unrecognized engine {engine} must be one of: {list(engines)}"
+            )
+        backend = engines[engine]
+    elif isinstance(engine, type) and issubclass(engine, BackendEntrypoint):
+        backend = engine
+    else:
+        raise TypeError(
+            (
+                "engine must be a string or a subclass of "
+                f"xarray.backends.BackendEntrypoint: {engine}"
+            )
         )
-    return engines[engine]
+
+    return backend

```

## Test Patch

```diff
diff --git a/xarray/tests/test_backends_api.py b/xarray/tests/test_backends_api.py
--- a/xarray/tests/test_backends_api.py
+++ b/xarray/tests/test_backends_api.py
@@ -1,6 +1,9 @@
+import numpy as np
+
+import xarray as xr
 from xarray.backends.api import _get_default_engine
 
-from . import requires_netCDF4, requires_scipy
+from . import assert_identical, requires_netCDF4, requires_scipy
 
 
 @requires_netCDF4
@@ -14,3 +17,20 @@ def test__get_default_engine():
 
     engine_default = _get_default_engine("/example")
     assert engine_default == "netcdf4"
+
+
+def test_custom_engine():
+    expected = xr.Dataset(
+        dict(a=2 * np.arange(5)), coords=dict(x=("x", np.arange(5), dict(units="s")))
+    )
+
+    class CustomBackend(xr.backends.BackendEntrypoint):
+        def open_dataset(
+            filename_or_obj,
+            drop_variables=None,
+            **kwargs,
+        ):
+            return expected.copy(deep=True)
+
+    actual = xr.open_dataset("fake_filename", engine=CustomBackend)
+    assert_identical(expected, actual)

```


## Code snippets

### 1 - xarray/backends/plugins.py:

Start line: 35, End line: 123

```python
def detect_parameters(open_dataset):
    signature = inspect.signature(open_dataset)
    parameters = signature.parameters
    parameters_list = []
    for name, param in parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise TypeError(
                f"All the parameters in {open_dataset!r} signature should be explicit. "
                "*args and **kwargs is not supported"
            )
        if name != "self":
            parameters_list.append(name)
    return tuple(parameters_list)


def backends_dict_from_pkg(pkg_entrypoints):
    backend_entrypoints = {}
    for pkg_ep in pkg_entrypoints:
        name = pkg_ep.name
        try:
            backend = pkg_ep.load()
            backend_entrypoints[name] = backend
        except Exception as ex:
            warnings.warn(f"Engine {name!r} loading failed:\n{ex}", RuntimeWarning)
    return backend_entrypoints


def set_missing_parameters(backend_entrypoints):
    for name, backend in backend_entrypoints.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


def sort_backends(backend_entrypoints):
    ordered_backends_entrypoints = {}
    for be_name in STANDARD_BACKENDS_ORDER:
        if backend_entrypoints.get(be_name, None) is not None:
            ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
    ordered_backends_entrypoints.update(
        {name: backend_entrypoints[name] for name in sorted(backend_entrypoints)}
    )
    return ordered_backends_entrypoints


def build_engines(pkg_entrypoints):
    backend_entrypoints = BACKEND_ENTRYPOINTS.copy()
    pkg_entrypoints = remove_duplicates(pkg_entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(pkg_entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    engines = {}
    for name, backend in backend_entrypoints.items():
        engines[name] = backend()
    return engines


@functools.lru_cache(maxsize=1)
def list_engines():
    pkg_entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    return build_engines(pkg_entrypoints)


def guess_engine(store_spec):
    engines = list_engines()

    for engine, backend in engines.items():
        try:
            if backend.guess_can_open and backend.guess_can_open(store_spec):
                return engine
        except Exception:
            warnings.warn(f"{engine!r} fails while guessing", RuntimeWarning)

    raise ValueError("cannot guess the engine, try passing one explicitly")


def get_backend(engine):
    """Select open_dataset method based on current engine"""
    engines = list_engines()
    if engine not in engines:
        raise ValueError(
            f"unrecognized engine {engine} must be one of: {list(engines)}"
        )
    return engines[engine]
```
### 2 - xarray/backends/api.py:

Start line: 1, End line: 119

```python
import os
from glob import glob
from io import BytesIO
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    Union,
)

import numpy as np

from .. import backends, coding, conventions
from ..core import indexing
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from ..core.dataarray import DataArray
from ..core.dataset import Dataset, _get_chunk, _maybe_chunk
from ..core.utils import is_remote_uri
from . import plugins
from .common import AbstractDataStore, ArrayWriter
from .locks import _get_scheduler

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


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


def _get_default_engine_remote_uri():
    try:
        import netCDF4  # noqa: F401

        engine = "netcdf4"
    except ImportError:  # pragma: no cover
        try:
            import pydap  # noqa: F401

            engine = "pydap"
        except ImportError:
            raise ValueError(
                "netCDF4 or pydap is required for accessing "
                "remote datasets via OPeNDAP"
            )
    return engine


def _get_default_engine_gz():
    try:
        import scipy  # noqa: F401

        engine = "scipy"
    except ImportError:  # pragma: no cover
        raise ValueError("scipy is required for accessing .gz files")
    return engine


def _get_default_engine_netcdf():
    try:
        import netCDF4  # noqa: F401

        engine = "netcdf4"
    except ImportError:  # pragma: no cover
        try:
            import scipy.io.netcdf  # noqa: F401

            engine = "scipy"
        except ImportError:
            raise ValueError(
                "cannot read or write netCDF files without "
                "netCDF4-python or scipy installed"
            )
    return engine


def _get_default_engine(path: str, allow_remote: bool = False):
    if allow_remote and is_remote_uri(path):
        engine = _get_default_engine_remote_uri()
    elif path.endswith(".gz"):
        engine = _get_default_engine_gz()
    else:
        engine = _get_default_engine_netcdf()
    return engine


def _normalize_path(path):
    if isinstance(path, Path):
        path = str(path)

    if isinstance(path, str) and not is_remote_uri(path):
        path = os.path.abspath(os.path.expanduser(path))

    return path
```
### 3 - xarray/backends/api.py:

Start line: 311, End line: 347

```python
def _dataset_from_backend_dataset(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    cache,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    if not (isinstance(chunks, (int, dict)) or chunks is None):
        if chunks != "auto":
            raise ValueError(
                "chunks must be an int, dict, 'auto', or None. "
                "Instead found %s. " % chunks
            )

    _protect_dataset_variables_inplace(backend_ds, cache)
    if chunks is None:
        ds = backend_ds
    else:
        ds = _chunk_ds(
            backend_ds,
            filename_or_obj,
            engine,
            chunks,
            overwrite_encoded_chunks,
            **extra_tokens,
        )

    ds.set_close(backend_ds._close)

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds
```
### 4 - xarray/backends/api.py:

Start line: 282, End line: 308

```python
def _chunk_ds(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    from dask.base import tokenize

    mtime = _get_mtime(filename_or_obj)
    token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
    name_prefix = "open_dataset-%s" % token

    variables = {}
    for name, var in backend_ds.variables.items():
        var_chunks = _get_chunk(var, chunks)
        variables[name] = _maybe_chunk(
            name,
            var,
            var_chunks,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
            name_prefix=name_prefix,
            token=token,
        )
    ds = backend_ds._replace(variables)
    return ds
```
### 5 - xarray/backends/api.py:

Start line: 923, End line: 977

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    data_vars="all",
    coords="different",
    combine="by_coords",
    parallel=False,
    join="outer",
    attrs_file=None,
    combine_attrs="override",
    **kwargs,
):
    # ... other code
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

    def multi_file_closer():
        for closer in closers:
            closer()

    combined.set_close(multi_file_closer)

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, Path):
            attrs_file = str(attrs_file)
        combined.attrs = datasets[paths.index(attrs_file)].attrs

    return combined


WRITEABLE_STORES: Dict[str, Callable] = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "h5netcdf": backends.H5NetCDFStore.open,
}
```
### 6 - xarray/backends/api.py:

Start line: 480, End line: 527

```python
def open_dataset(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    if len(args) > 0:
        raise TypeError(
            "open_dataset() takes only 1 positional argument starting from version 0.18.0, "
            "all other options must be passed as keyword arguments"
        )

    if cache is None:
        cache = chunks is None

    if backend_kwargs is not None:
        kwargs.update(backend_kwargs)

    if engine is None:
        engine = plugins.guess_engine(filename_or_obj)

    backend = plugins.get_backend(engine)

    decoders = _resolve_decoders_kwargs(
        decode_cf,
        open_backend_dataset_parameters=backend.open_dataset_parameters,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        decode_timedelta=decode_timedelta,
        concat_characters=concat_characters,
        use_cftime=use_cftime,
        decode_coords=decode_coords,
    )

    overwrite_encoded_chunks = kwargs.pop("overwrite_encoded_chunks", None)
    backend_ds = backend.open_dataset(
        filename_or_obj,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )
    ds = _dataset_from_backend_dataset(
        backend_ds,
        filename_or_obj,
        engine,
        chunks,
        cache,
        overwrite_encoded_chunks,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )

    return ds
```
### 7 - xarray/backends/common.py:

Start line: 346, End line: 386

```python
class BackendEntrypoint:
    """
    ``BackendEntrypoint`` is a class container and it is the main interface
    for the backend plugins, see :ref:`RST backend_entrypoint`.
    It shall implement:

    - ``open_dataset`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~xarray.Dataset`.
      It shall take in input at least ``filename_or_obj`` argument and
      ``drop_variables`` keyword argument.
      For more details see :ref:`RST open_dataset`.
    - ``guess_can_open`` method: it shall return ``True`` if the backend is able to open
      ``filename_or_obj``, ``False`` otherwise. The implementation of this
      method is not mandatory.
    """

    open_dataset_parameters: Union[Tuple, None] = None
    """list of ``open_dataset`` method parameters"""

    def open_dataset(
        self,
        filename_or_obj: str,
        drop_variables: Tuple[str] = None,
        **kwargs: Any,
    ):
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        raise NotImplementedError

    def guess_can_open(self, filename_or_obj):
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """

        return False


BACKEND_ENTRYPOINTS: Dict[str, Type[BackendEntrypoint]] = {}
```
### 8 - xarray/backends/plugins.py:

Start line: 1, End line: 32

```python
import functools
import inspect
import itertools
import warnings

import pkg_resources

from .common import BACKEND_ENTRYPOINTS

STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]


def remove_duplicates(pkg_entrypoints):

    # sort and group entrypoints by name
    pkg_entrypoints = sorted(pkg_entrypoints, key=lambda ep: ep.name)
    pkg_entrypoints_grouped = itertools.groupby(pkg_entrypoints, key=lambda ep: ep.name)
    # check if there are multiple entrypoints for the same name
    unique_pkg_entrypoints = []
    for name, matches in pkg_entrypoints_grouped:
        matches = list(matches)
        unique_pkg_entrypoints.append(matches[0])
        matches_len = len(matches)
        if matches_len > 1:
            selected_module_name = matches[0].module_name
            all_module_names = [e.module_name for e in matches]
            warnings.warn(
                f"Found {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
                RuntimeWarning,
            )
    return unique_pkg_entrypoints
```
### 9 - xarray/backends/api.py:

Start line: 855, End line: 922

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    data_vars="all",
    coords="different",
    combine="by_coords",
    parallel=False,
    join="outer",
    attrs_file=None,
    combine_attrs="override",
    **kwargs,
):
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
            paths = fs.glob(fs._strip_protocol(paths))  # finds directories
            paths = [fs.get_mapper(path) for path in paths]
        elif is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        else:
            paths = sorted(glob(_normalize_path(paths)))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    # If combine='by_coords' then this is unnecessary, but quick.
    # If combine='nested' then this creates a flat list which is easier to
    # iterate over, while saving the originally-supplied structure as "ids"
    if combine == "nested":
        if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
            concat_dim = [concat_dim]
    combined_ids_paths = _infer_concat_order_from_positions(paths)
    ids, paths = (list(combined_ids_paths.keys()), list(combined_ids_paths.values()))

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
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, closers = dask.compute(datasets, closers)

    # Combine all datasets, closing them in case of a ValueError
    # ... other code
```
### 10 - xarray/backends/scipy_.py:

Start line: 235, End line: 286

```python
class ScipyBackendEntrypoint(BackendEntrypoint):
    def guess_can_open(self, filename_or_obj):
        try:
            return read_magic_number(filename_or_obj).startswith(b"CDF")
        except TypeError:
            pass

        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".nc", ".nc4", ".cdf", ".gz"}

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        mode="r",
        format=None,
        group=None,
        mmap=None,
        lock=None,
    ):

        store = ScipyDataStore(
            filename_or_obj, mode=mode, format=format, group=group, mmap=mmap, lock=lock
        )

        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


if has_scipy:
    BACKEND_ENTRYPOINTS["scipy"] = ScipyBackendEntrypoint
```
### 11 - xarray/backends/api.py:

Start line: 661, End line: 970

```python
def open_dataarray(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    if len(args) > 0:
        raise TypeError(
            "open_dataarray() takes only 1 positional argument starting from version 0.18.0, "
            "all other options must be passed as keyword arguments"
        )

    dataset = open_dataset(
        filename_or_obj,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        **kwargs,
    )

    if len(dataset.data_vars) != 1:
        raise ValueError(
            "Given file dataset contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        (data_array,) = dataset.data_vars.values()

    data_array.set_close(dataset._close)

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in dataset.attrs:
        data_array.name = dataset.attrs[DATAARRAY_NAME]
        del dataset.attrs[DATAARRAY_NAME]

    if data_array.name == DATAARRAY_VARIABLE:
        data_array.name = None

    return data_array


def open_mfdataset(
    paths,
    chunks=None,
    concat_dim=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    data_vars="all",
    coords="different",
    combine="by_coords",
    parallel=False,
    join="outer",
    attrs_file=None,
    combine_attrs="override",
    **kwargs,
):
    # ... other code
```
### 16 - xarray/backends/api.py:

Start line: 189, End line: 227

```python
def _resolve_decoders_kwargs(decode_cf, open_backend_dataset_parameters, **decoders):
    for d in list(decoders):
        if decode_cf is False and d in open_backend_dataset_parameters:
            decoders[d] = False
        if decoders[d] is None:
            decoders.pop(d)
    return decoders


def _get_mtime(filename_or_obj):
    # if passed an actual file path, augment the token with
    # the file modification time
    mtime = None

    try:
        path = os.fspath(filename_or_obj)
    except TypeError:
        path = None

    if path and not is_remote_uri(path):
        mtime = os.path.getmtime(filename_or_obj)

    return mtime


def _protect_dataset_variables_inplace(dataset, cache):
    for name, variable in dataset.variables.items():
        if name not in variable.dims:
            # no need to protect IndexVariable objects
            data = indexing.CopyOnWriteArray(variable._data)
            if cache:
                data = indexing.MemoryCachedArray(data)
            variable.data = data


def _finalize_store(write, store):
    """ Finalize this store by explicitly syncing and closing"""
    del write  # ensure writing is done first
    store.close()
```
### 21 - xarray/backends/api.py:

Start line: 350, End line: 479

```python
def open_dataset(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr"}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks=-1`` loads the dataset with dask using a single
        chunk for all arrays. `chunks={}`` loads the dataset with dask using
        engine preferred chunks if exposed by the backend, otherwise with
        a single chunk for all arrays.
        ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
        engine preferred chunks. See dask chunking for more details.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "pynio", "pseudonetcdf", "cfgrib".

        See engine open function for kwargs accepted by each specific engine.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.

    See Also
    --------
    open_mfdataset
    """
    # ... other code
```
### 24 - xarray/backends/api.py:

Start line: 1203, End line: 1246

```python
def save_mfdataset(
    datasets, paths, mode="w", format=None, groups=None, engine=None, compute=True
):
    if mode == "w" and len(set(paths)) < len(paths):
        raise ValueError(
            "cannot use mode='w' when writing multiple datasets to the same path"
        )

    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError(
                "save_mfdataset only supports writing Dataset "
                "objects, received type %s" % type(obj)
            )

    if groups is None:
        groups = [None] * len(datasets)

    if len({len(datasets), len(paths), len(groups)}) > 1:
        raise ValueError(
            "must supply lists of the same length for the "
            "datasets, paths and groups arguments to "
            "save_mfdataset"
        )

    writers, stores = zip(
        *[
            to_netcdf(
                ds, path, mode, format, group, engine, compute=compute, multifile=True
            )
            for ds, path, group in zip(datasets, paths, groups)
        ]
    )

    try:
        writes = [w.sync(compute=compute) for w in writers]
    finally:
        if compute:
            for store in stores:
                store.close()

    if not compute:
        import dask

        return dask.delayed(
            [dask.delayed(_finalize_store)(w, s) for w, s in zip(writes, stores)]
        )
```
### 26 - xarray/backends/api.py:

Start line: 1273, End line: 1362

```python
def _validate_append_dim_and_encoding(
    ds_to_append, store, append_dim, region, encoding, **open_kwargs
):
    try:
        ds = backends.zarr.open_zarr(store, **open_kwargs)
    except ValueError:  # store empty
        return

    if append_dim:
        if append_dim not in ds.dims:
            raise ValueError(
                f"append_dim={append_dim!r} does not match any existing "
                f"dataset dimensions {ds.dims}"
            )
        if region is not None and append_dim in region:
            raise ValueError(
                f"cannot list the same dimension in both ``append_dim`` and "
                f"``region`` with to_zarr(), got {append_dim} in both"
            )

    if region is not None:
        if not isinstance(region, dict):
            raise TypeError(f"``region`` must be a dict, got {type(region)}")
        for k, v in region.items():
            if k not in ds_to_append.dims:
                raise ValueError(
                    f"all keys in ``region`` are not in Dataset dimensions, got "
                    f"{list(region)} and {list(ds_to_append.dims)}"
                )
            if not isinstance(v, slice):
                raise TypeError(
                    "all values in ``region`` must be slice objects, got "
                    f"region={region}"
                )
            if v.step not in {1, None}:
                raise ValueError(
                    "step on all slices in ``region`` must be 1 or None, got "
                    f"region={region}"
                )

        non_matching_vars = [
            k
            for k, v in ds_to_append.variables.items()
            if not set(region).intersection(v.dims)
        ]
        if non_matching_vars:
            raise ValueError(
                f"when setting `region` explicitly in to_zarr(), all "
                f"variables in the dataset to write must have at least "
                f"one dimension in common with the region's dimensions "
                f"{list(region.keys())}, but that is not "
                f"the case for some variables here. To drop these variables "
                f"from this dataset before exporting to zarr, write: "
                f".drop({non_matching_vars!r})"
            )

    for var_name, new_var in ds_to_append.variables.items():
        if var_name in ds.variables:
            existing_var = ds.variables[var_name]
            if new_var.dims != existing_var.dims:
                raise ValueError(
                    f"variable {var_name!r} already exists with different "
                    f"dimension names {existing_var.dims} != "
                    f"{new_var.dims}, but changing variable "
                    f"dimensions is not supported by to_zarr()."
                )

            existing_sizes = {}
            for dim, size in existing_var.sizes.items():
                if region is not None and dim in region:
                    start, stop, stride = region[dim].indices(size)
                    assert stride == 1  # region was already validated above
                    size = stop - start
                if dim != append_dim:
                    existing_sizes[dim] = size

            new_sizes = {
                dim: size for dim, size in new_var.sizes.items() if dim != append_dim
            }
            if existing_sizes != new_sizes:
                raise ValueError(
                    f"variable {var_name!r} already exists with different "
                    f"dimension sizes: {existing_sizes} != {new_sizes}. "
                    f"to_zarr() only supports changing dimension sizes when "
                    f"explicitly appending, but append_dim={append_dim!r}."
                )
            if var_name in encoding.keys():
                raise ValueError(
                    f"variable {var_name!r} already exists, but encoding was provided"
                )
```
### 42 - xarray/backends/api.py:

Start line: 530, End line: 660

```python
def open_dataarray(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    """Open an DataArray from a file or file-like object containing a single
    data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr"}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks=-1`` loads the dataset with dask using a single
        chunk for all arrays. `chunks={}`` loads the dataset with dask using
        engine preferred chunks if exposed by the backend, otherwise with
        a single chunk for all arrays.
        ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
        engine preferred chunks. See dask chunking for more details.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "pynio", "pseudonetcdf", "cfgrib".

        See engine open function for kwargs accepted by each specific engine.

    Notes
    -----
    This is designed to be fully compatible with `DataArray.to_netcdf`. Saving
    using `DataArray.to_netcdf` and then loading with this function will
    produce an identical result.

    All parameters are passed directly to `xarray.open_dataset`. See that
    documentation for further details.

    See also
    --------
    open_dataset
    """
    # ... other code
```
### 45 - xarray/backends/api.py:

Start line: 723, End line: 854

```python
def open_mfdataset(
    paths,
    chunks=None,
    concat_dim=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    data_vars="all",
    coords="different",
    combine="by_coords",
    parallel=False,
    join="outer",
    attrs_file=None,
    combine_attrs="override",
    **kwargs,
):
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default ``combine='by_coords'``
    will be used. Requires dask to be installed. See documentation for
    details on dask [1]_. Global attributes from the ``attrs_file`` are used
    for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_.
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='by_coords'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        ``xarray.merge``.
    combine : {"by_coords", "nested"}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. Default is to use ``xarray.combine_by_coords``.
    compat : {"identical", "equals", "broadcast_equals", \
              "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

         * "broadcast_equals": all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * "equals": all values and dimensions must be the same.
         * "identical": all values, dimensions and attributes must be the
           same.
         * "no_conflicts": only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
         * "override": skip comparing and pick variable from first dataset

    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", "zarr"}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    data_vars : {"minimal", "different", "all"} or list of str, optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    coords : {"minimal", "different", "all"} or list of str, optional
        These coordinate variables will be concatenated together:
         * "minimal": Only coordinates in which the dimension already appears
           are included.
         * "different": Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * "all": All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the "minimal" coordinates.
    parallel : bool, optional
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {"outer", "inner", "left", "right", "exact, "override"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    attrs_file : str or pathlib.Path, optional
        Path of the file used to read global attributes from.
        By default global attributes are read from the first file provided,
        with wildcard matches sorted by filename.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    ``open_mfdataset`` opens files with read-only access. When you modify values
    of a Dataset, even one linked to files on disk, only the in-memory copy you
    are manipulating in xarray is modified: the original file on disk is never
    touched.

    See Also
    --------
    combine_by_coords
    combine_nested
    open_dataset

    References
    ----------

    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    # ... other code
```
### 71 - xarray/backends/api.py:

Start line: 980, End line: 1073

```python
def to_netcdf(
    dataset: Dataset,
    path_or_file=None,
    mode: str = "w",
    format: str = None,
    group: str = None,
    engine: str = None,
    encoding: Mapping = None,
    unlimited_dims: Iterable[Hashable] = None,
    compute: bool = True,
    multifile: bool = False,
    invalid_netcdf: bool = False,
) -> Union[Tuple[ArrayWriter, AbstractDataStore], bytes, "Delayed", None]:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    if isinstance(path_or_file, Path):
        path_or_file = str(path_or_file)

    if encoding is None:
        encoding = {}

    if path_or_file is None:
        if engine is None:
            engine = "scipy"
        elif engine != "scipy":
            raise ValueError(
                "invalid engine for creating bytes with "
                "to_netcdf: %r. Only the default engine "
                "or engine='scipy' is supported" % engine
            )
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning bytes"
            )
    elif isinstance(path_or_file, str):
        if engine is None:
            engine = _get_default_engine(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    else:  # file-like object
        engine = "scipy"

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset, invalid_netcdf=invalid_netcdf and engine == "h5netcdf")

    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError("unrecognized engine for to_netcdf: %r" % engine)

    if format is not None:
        format = format.upper()

    # handle scheduler specific logic
    scheduler = _get_scheduler()
    have_chunks = any(v.chunks for v in dataset.variables.values())

    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]
    if autoclose and engine == "scipy":
        raise NotImplementedError(
            "Writing netCDF files with the %s backend "
            "is not currently supported with dask's %s "
            "scheduler" % (engine, scheduler)
        )

    target = path_or_file if path_or_file is not None else BytesIO()
    kwargs = dict(autoclose=True) if autoclose else {}
    if invalid_netcdf:
        if engine == "h5netcdf":
            kwargs["invalid_netcdf"] = invalid_netcdf
        else:
            raise ValueError(
                "unrecognized option 'invalid_netcdf' for engine %s" % engine
            )
    store = store_open(target, mode, format, group, **kwargs)

    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get("unlimited_dims", None)
    if unlimited_dims is not None:
        if isinstance(unlimited_dims, str) or not isinstance(unlimited_dims, Iterable):
            unlimited_dims = [unlimited_dims]
        else:
            unlimited_dims = list(unlimited_dims)

    writer = ArrayWriter()

    # TODO: figure out how to refactor this logic (here and in save_mfdataset)
    # to avoid this mess of conditionals
    # ... other code
```
