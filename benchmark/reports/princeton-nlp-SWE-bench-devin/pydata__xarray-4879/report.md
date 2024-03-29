# pydata__xarray-4879

| **pydata/xarray** | `15c68366b8ba8fd678d675df5688cf861d1c7235` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 8 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/xarray/backends/file_manager.py b/xarray/backends/file_manager.py
--- a/xarray/backends/file_manager.py
+++ b/xarray/backends/file_manager.py
@@ -3,8 +3,9 @@
 import contextlib
 import io
 import threading
+import uuid
 import warnings
-from typing import Any
+from typing import Any, Hashable
 
 from ..core import utils
 from ..core.options import OPTIONS
@@ -12,12 +13,11 @@
 from .lru_cache import LRUCache
 
 # Global cache for storing open files.
-FILE_CACHE: LRUCache[str, io.IOBase] = LRUCache(
+FILE_CACHE: LRUCache[Any, io.IOBase] = LRUCache(
     maxsize=OPTIONS["file_cache_maxsize"], on_evict=lambda k, v: v.close()
 )
 assert FILE_CACHE.maxsize, "file cache must be at least size one"
 
-
 REF_COUNTS: dict[Any, int] = {}
 
 _DEFAULT_MODE = utils.ReprObject("<unused>")
@@ -85,12 +85,13 @@ def __init__(
         kwargs=None,
         lock=None,
         cache=None,
+        manager_id: Hashable | None = None,
         ref_counts=None,
     ):
-        """Initialize a FileManager.
+        """Initialize a CachingFileManager.
 
-        The cache and ref_counts arguments exist solely to facilitate
-        dependency injection, and should only be set for tests.
+        The cache, manager_id and ref_counts arguments exist solely to
+        facilitate dependency injection, and should only be set for tests.
 
         Parameters
         ----------
@@ -120,6 +121,8 @@ def __init__(
             global variable and contains non-picklable file objects, an
             unpickled FileManager objects will be restored with the default
             cache.
+        manager_id : hashable, optional
+            Identifier for this CachingFileManager.
         ref_counts : dict, optional
             Optional dict to use for keeping track the number of references to
             the same file.
@@ -129,13 +132,17 @@ def __init__(
         self._mode = mode
         self._kwargs = {} if kwargs is None else dict(kwargs)
 
-        self._default_lock = lock is None or lock is False
-        self._lock = threading.Lock() if self._default_lock else lock
+        self._use_default_lock = lock is None or lock is False
+        self._lock = threading.Lock() if self._use_default_lock else lock
 
         # cache[self._key] stores the file associated with this object.
         if cache is None:
             cache = FILE_CACHE
         self._cache = cache
+        if manager_id is None:
+            # Each call to CachingFileManager should separately open files.
+            manager_id = str(uuid.uuid4())
+        self._manager_id = manager_id
         self._key = self._make_key()
 
         # ref_counts[self._key] stores the number of CachingFileManager objects
@@ -153,6 +160,7 @@ def _make_key(self):
             self._args,
             "a" if self._mode == "w" else self._mode,
             tuple(sorted(self._kwargs.items())),
+            self._manager_id,
         )
         return _HashedSequence(value)
 
@@ -223,20 +231,14 @@ def close(self, needs_lock=True):
             if file is not None:
                 file.close()
 
-    def __del__(self):
-        # If we're the only CachingFileManger referencing a unclosed file, we
-        # should remove it from the cache upon garbage collection.
+    def __del__(self) -> None:
+        # If we're the only CachingFileManger referencing a unclosed file,
+        # remove it from the cache upon garbage collection.
         #
-        # Keeping our own count of file references might seem like overkill,
-        # but it's actually pretty common to reopen files with the same
-        # variable name in a notebook or command line environment, e.g., to
-        # fix the parameters used when opening a file:
-        #    >>> ds = xarray.open_dataset('myfile.nc')
-        #    >>> ds = xarray.open_dataset('myfile.nc', decode_times=False)
-        # This second assignment to "ds" drops CPython's ref-count on the first
-        # "ds" argument to zero, which can trigger garbage collections. So if
-        # we didn't check whether another object is referencing 'myfile.nc',
-        # the newly opened file would actually be immediately closed!
+        # We keep track of our own reference count because we don't want to
+        # close files if another identical file manager needs it. This can
+        # happen if a CachingFileManager is pickled and unpickled without
+        # closing the original file.
         ref_count = self._ref_counter.decrement(self._key)
 
         if not ref_count and self._key in self._cache:
@@ -249,30 +251,40 @@ def __del__(self):
 
             if OPTIONS["warn_for_unclosed_files"]:
                 warnings.warn(
-                    "deallocating {}, but file is not already closed. "
-                    "This may indicate a bug.".format(self),
+                    f"deallocating {self}, but file is not already closed. "
+                    "This may indicate a bug.",
                     RuntimeWarning,
                     stacklevel=2,
                 )
 
     def __getstate__(self):
         """State for pickling."""
-        # cache and ref_counts are intentionally omitted: we don't want to try
-        # to serialize these global objects.
-        lock = None if self._default_lock else self._lock
-        return (self._opener, self._args, self._mode, self._kwargs, lock)
+        # cache is intentionally omitted: we don't want to try to serialize
+        # these global objects.
+        lock = None if self._use_default_lock else self._lock
+        return (
+            self._opener,
+            self._args,
+            self._mode,
+            self._kwargs,
+            lock,
+            self._manager_id,
+        )
 
-    def __setstate__(self, state):
+    def __setstate__(self, state) -> None:
         """Restore from a pickle."""
-        opener, args, mode, kwargs, lock = state
-        self.__init__(opener, *args, mode=mode, kwargs=kwargs, lock=lock)
+        opener, args, mode, kwargs, lock, manager_id = state
+        self.__init__(  # type: ignore
+            opener, *args, mode=mode, kwargs=kwargs, lock=lock, manager_id=manager_id
+        )
 
-    def __repr__(self):
+    def __repr__(self) -> str:
         args_string = ", ".join(map(repr, self._args))
         if self._mode is not _DEFAULT_MODE:
             args_string += f", mode={self._mode!r}"
-        return "{}({!r}, {}, kwargs={})".format(
-            type(self).__name__, self._opener, args_string, self._kwargs
+        return (
+            f"{type(self).__name__}({self._opener!r}, {args_string}, "
+            f"kwargs={self._kwargs}, manager_id={self._manager_id!r})"
         )
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/backends/file_manager.py | 6 | 6 | - | - | -
| xarray/backends/file_manager.py | 15 | 20 | - | - | -
| xarray/backends/file_manager.py | 88 | 91 | - | - | -
| xarray/backends/file_manager.py | 123 | 123 | - | - | -
| xarray/backends/file_manager.py | 132 | 133 | - | - | -
| xarray/backends/file_manager.py | 156 | 156 | - | - | -
| xarray/backends/file_manager.py | 226 | 239 | - | - | -
| xarray/backends/file_manager.py | 252 | 275 | - | - | -


## Problem Statement

```
jupyter repr caching deleted netcdf file
**What happened**:

Testing xarray data storage in a jupyter notebook with varying data sizes and storing to a netcdf, i noticed that open_dataset/array (both show this behaviour) continue to return data from the first testing run, ignoring the fact that each run deletes the previously created netcdf file.
This only happens once the `repr` was used to display the xarray object. 
But once in error mode, even the previously fine printed objects are then showing the wrong data.

This was hard to track down as it depends on the precise sequence in jupyter.

**What you expected to happen**:

when i use `open_dataset/array`, the resulting object should reflect reality on disk.

**Minimal Complete Verifiable Example**:

\`\`\`python
import xarray as xr
from pathlib import Path
import numpy as np

def test_repr(nx):
    ds = xr.DataArray(np.random.rand(nx))
    path = Path("saved_on_disk.nc")
    if path.exists():
        path.unlink()
    ds.to_netcdf(path)
    return path
\`\`\`

When executed in a cell with print for display, all is fine:
\`\`\`python
test_repr(4)
print(xr.open_dataset("saved_on_disk.nc"))
test_repr(5)
print(xr.open_dataset("saved_on_disk.nc"))
\`\`\`

but as soon as one cell used the jupyter repr:

\`\`\`python
xr.open_dataset("saved_on_disk.nc")
\`\`\`

all future file reads, even after executing the test function again and even using `print` and not `repr`, show the data from the last repr use.


**Anything else we need to know?**:

Here's a notebook showing the issue:
https://gist.github.com/05c2542ed33662cdcb6024815cc0c72c

**Environment**:

<details><summary>Output of <tt>xr.show_versions()</tt></summary>

INSTALLED VERSIONS
------------------
commit: None
python: 3.7.6 | packaged by conda-forge | (default, Jun  1 2020, 18:57:50) 
[GCC 7.5.0]
python-bits: 64
OS: Linux
OS-release: 5.4.0-40-generic
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: en_US.UTF-8
LOCALE: en_US.UTF-8
libhdf5: 1.10.6
libnetcdf: 4.7.4

xarray: 0.16.0
pandas: 1.0.5
numpy: 1.19.0
scipy: 1.5.1
netCDF4: 1.5.3
pydap: None
h5netcdf: None
h5py: 2.10.0
Nio: None
zarr: None
cftime: 1.2.1
nc_time_axis: None
PseudoNetCDF: None
rasterio: 1.1.5
cfgrib: None
iris: None
bottleneck: None
dask: 2.21.0
distributed: 2.21.0
matplotlib: 3.3.0
cartopy: 0.18.0
seaborn: 0.10.1
numbagg: None
pint: None
setuptools: 49.2.0.post20200712
pip: 20.1.1
conda: installed
pytest: 6.0.0rc1
IPython: 7.16.1
sphinx: 3.1.2

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 asv_bench/benchmarks/dataset_io.py | 1 | 93| 737 | 737 | 3705 | 
| 2 | 2 asv_bench/benchmarks/repr.py | 1 | 41| 277 | 1014 | 3981 | 
| 3 | 2 asv_bench/benchmarks/dataset_io.py | 222 | 296| 631 | 1645 | 3981 | 
| 4 | 3 asv_bench/benchmarks/reindexing.py | 1 | 53| 415 | 2060 | 4396 | 
| 5 | 3 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 2409 | 4396 | 
| 6 | 4 xarray/__init__.py | 1 | 111| 692 | 3101 | 5088 | 
| 7 | 4 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 3370 | 5088 | 
| 8 | 4 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 3635 | 5088 | 
| 9 | 4 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 4077 | 5088 | 
| 10 | 4 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 4341 | 5088 | 
| 11 | 4 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 4504 | 5088 | 
| 12 | 4 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 4616 | 5088 | 
| 13 | 5 asv_bench/benchmarks/interp.py | 1 | 18| 150 | 4766 | 5539 | 
| 14 | 6 asv_bench/benchmarks/rolling.py | 1 | 17| 113 | 4879 | 6814 | 
| 15 | 6 asv_bench/benchmarks/dataset_io.py | 315 | 328| 114 | 4993 | 6814 | 
| 16 | 7 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 5731 | 8315 | 
| 17 | 8 xarray/util/print_versions.py | 62 | 77| 128 | 5859 | 9577 | 
| 18 | 8 asv_bench/benchmarks/dataset_io.py | 299 | 312| 113 | 5972 | 9577 | 
| 19 | 8 asv_bench/benchmarks/dataset_io.py | 435 | 479| 246 | 6218 | 9577 | 
| 20 | 9 xarray/core/common.py | 1 | 55| 305 | 6523 | 24485 | 
| 21 | 10 doc/conf.py | 283 | 337| 702 | 7225 | 28370 | 
| 22 | 10 xarray/util/print_versions.py | 80 | 166| 773 | 7998 | 28370 | 
| 23 | 10 asv_bench/benchmarks/indexing.py | 133 | 150| 145 | 8143 | 28370 | 
| 24 | 11 xarray/core/missing.py | 1 | 25| 184 | 8327 | 34617 | 
| 25 | 12 xarray/plot/utils.py | 1 | 81| 413 | 8740 | 48413 | 
| 26 | 13 xarray/backends/pseudonetcdf_.py | 1 | 28| 179 | 8919 | 49428 | 
| 27 | 14 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 9086 | 49595 | 
| 28 | 15 xarray/backends/api.py | 1 | 83| 566 | 9652 | 64230 | 
| 29 | 16 xarray/backends/h5netcdf_.py | 1 | 44| 265 | 9917 | 67128 | 
| 30 | 16 doc/conf.py | 1 | 111| 733 | 10650 | 67128 | 
| 31 | 17 xarray/core/types.py | 1 | 101| 814 | 11464 | 68733 | 
| 32 | 18 xarray/core/dataarray.py | 1 | 98| 644 | 12108 | 126734 | 


## Missing Patch Files

 * 1: xarray/backends/file_manager.py

### Hint

```
Thanks for the clear example!

This happens dues to xarray's caching logic for files: 
https://github.com/pydata/xarray/blob/b1c7e315e8a18e86c5751a0aa9024d41a42ca5e8/xarray/backends/file_manager.py#L50-L76

This means that when you open the same filename, xarray doesn't actually reopen the file from disk -- instead it points to the same file object already cached in memory.

I can see why this could be confusing. We do need this caching logic for files opened from the same `backends.*DataStore` class, but this could include some sort of unique identifier (i.e., from `uuid`) to ensure each separate call to `xr.open_dataset` results in a separately cached/opened file object: 
https://github.com/pydata/xarray/blob/b1c7e315e8a18e86c5751a0aa9024d41a42ca5e8/xarray/backends/netCDF4_.py#L355-L357
is there a workaround for forcing the opening without restarting the notebook?
now i'm wondering why the caching logic is only activated by the `repr`? As you can see, when printed, it always updated to the status on disk?
Probably the easiest work around is to call `.close()` on the original dataset. Failing that, the file is cached in `xarray.backends.file_manager.FILE_CACHE`, which you could muck around with.

I believe it only gets activated by `repr()` because array values from netCDF file are loaded lazily. Not 100% without more testing, though.
Would it be an option to consider the time stamp of the file's last change as a caching criterion?
I've stumbled over this weird behaviour many times and was wondering why this happens. So AFAICT @shoyer hit the nail on the head but the root cause is that the Dataset is added to the notebook namespace somehow, if one just evaluates it in the cell.

This doesn't happen if you invoke the `__repr__` via

\`\`\`python
display(xr.open_dataset("saved_on_disk.nc"))
\`\`\`

I've forced myself to use either `print` or `display` for xarray data. As this also happens if the Dataset is attached to a variable you would need to specifically delete (or .close()) the variable in question before opening again. 

\`\`\`python
try: 
    del ds
except NameError:
    pass
ds = xr.open_dataset("saved_on_disk.nc")
\`\`\`

```

## Patch

```diff
diff --git a/xarray/backends/file_manager.py b/xarray/backends/file_manager.py
--- a/xarray/backends/file_manager.py
+++ b/xarray/backends/file_manager.py
@@ -3,8 +3,9 @@
 import contextlib
 import io
 import threading
+import uuid
 import warnings
-from typing import Any
+from typing import Any, Hashable
 
 from ..core import utils
 from ..core.options import OPTIONS
@@ -12,12 +13,11 @@
 from .lru_cache import LRUCache
 
 # Global cache for storing open files.
-FILE_CACHE: LRUCache[str, io.IOBase] = LRUCache(
+FILE_CACHE: LRUCache[Any, io.IOBase] = LRUCache(
     maxsize=OPTIONS["file_cache_maxsize"], on_evict=lambda k, v: v.close()
 )
 assert FILE_CACHE.maxsize, "file cache must be at least size one"
 
-
 REF_COUNTS: dict[Any, int] = {}
 
 _DEFAULT_MODE = utils.ReprObject("<unused>")
@@ -85,12 +85,13 @@ def __init__(
         kwargs=None,
         lock=None,
         cache=None,
+        manager_id: Hashable | None = None,
         ref_counts=None,
     ):
-        """Initialize a FileManager.
+        """Initialize a CachingFileManager.
 
-        The cache and ref_counts arguments exist solely to facilitate
-        dependency injection, and should only be set for tests.
+        The cache, manager_id and ref_counts arguments exist solely to
+        facilitate dependency injection, and should only be set for tests.
 
         Parameters
         ----------
@@ -120,6 +121,8 @@ def __init__(
             global variable and contains non-picklable file objects, an
             unpickled FileManager objects will be restored with the default
             cache.
+        manager_id : hashable, optional
+            Identifier for this CachingFileManager.
         ref_counts : dict, optional
             Optional dict to use for keeping track the number of references to
             the same file.
@@ -129,13 +132,17 @@ def __init__(
         self._mode = mode
         self._kwargs = {} if kwargs is None else dict(kwargs)
 
-        self._default_lock = lock is None or lock is False
-        self._lock = threading.Lock() if self._default_lock else lock
+        self._use_default_lock = lock is None or lock is False
+        self._lock = threading.Lock() if self._use_default_lock else lock
 
         # cache[self._key] stores the file associated with this object.
         if cache is None:
             cache = FILE_CACHE
         self._cache = cache
+        if manager_id is None:
+            # Each call to CachingFileManager should separately open files.
+            manager_id = str(uuid.uuid4())
+        self._manager_id = manager_id
         self._key = self._make_key()
 
         # ref_counts[self._key] stores the number of CachingFileManager objects
@@ -153,6 +160,7 @@ def _make_key(self):
             self._args,
             "a" if self._mode == "w" else self._mode,
             tuple(sorted(self._kwargs.items())),
+            self._manager_id,
         )
         return _HashedSequence(value)
 
@@ -223,20 +231,14 @@ def close(self, needs_lock=True):
             if file is not None:
                 file.close()
 
-    def __del__(self):
-        # If we're the only CachingFileManger referencing a unclosed file, we
-        # should remove it from the cache upon garbage collection.
+    def __del__(self) -> None:
+        # If we're the only CachingFileManger referencing a unclosed file,
+        # remove it from the cache upon garbage collection.
         #
-        # Keeping our own count of file references might seem like overkill,
-        # but it's actually pretty common to reopen files with the same
-        # variable name in a notebook or command line environment, e.g., to
-        # fix the parameters used when opening a file:
-        #    >>> ds = xarray.open_dataset('myfile.nc')
-        #    >>> ds = xarray.open_dataset('myfile.nc', decode_times=False)
-        # This second assignment to "ds" drops CPython's ref-count on the first
-        # "ds" argument to zero, which can trigger garbage collections. So if
-        # we didn't check whether another object is referencing 'myfile.nc',
-        # the newly opened file would actually be immediately closed!
+        # We keep track of our own reference count because we don't want to
+        # close files if another identical file manager needs it. This can
+        # happen if a CachingFileManager is pickled and unpickled without
+        # closing the original file.
         ref_count = self._ref_counter.decrement(self._key)
 
         if not ref_count and self._key in self._cache:
@@ -249,30 +251,40 @@ def __del__(self):
 
             if OPTIONS["warn_for_unclosed_files"]:
                 warnings.warn(
-                    "deallocating {}, but file is not already closed. "
-                    "This may indicate a bug.".format(self),
+                    f"deallocating {self}, but file is not already closed. "
+                    "This may indicate a bug.",
                     RuntimeWarning,
                     stacklevel=2,
                 )
 
     def __getstate__(self):
         """State for pickling."""
-        # cache and ref_counts are intentionally omitted: we don't want to try
-        # to serialize these global objects.
-        lock = None if self._default_lock else self._lock
-        return (self._opener, self._args, self._mode, self._kwargs, lock)
+        # cache is intentionally omitted: we don't want to try to serialize
+        # these global objects.
+        lock = None if self._use_default_lock else self._lock
+        return (
+            self._opener,
+            self._args,
+            self._mode,
+            self._kwargs,
+            lock,
+            self._manager_id,
+        )
 
-    def __setstate__(self, state):
+    def __setstate__(self, state) -> None:
         """Restore from a pickle."""
-        opener, args, mode, kwargs, lock = state
-        self.__init__(opener, *args, mode=mode, kwargs=kwargs, lock=lock)
+        opener, args, mode, kwargs, lock, manager_id = state
+        self.__init__(  # type: ignore
+            opener, *args, mode=mode, kwargs=kwargs, lock=lock, manager_id=manager_id
+        )
 
-    def __repr__(self):
+    def __repr__(self) -> str:
         args_string = ", ".join(map(repr, self._args))
         if self._mode is not _DEFAULT_MODE:
             args_string += f", mode={self._mode!r}"
-        return "{}({!r}, {}, kwargs={})".format(
-            type(self).__name__, self._opener, args_string, self._kwargs
+        return (
+            f"{type(self).__name__}({self._opener!r}, {args_string}, "
+            f"kwargs={self._kwargs}, manager_id={self._manager_id!r})"
         )
 
 

```

## Test Patch

```diff
diff --git a/xarray/tests/test_backends.py b/xarray/tests/test_backends.py
--- a/xarray/tests/test_backends.py
+++ b/xarray/tests/test_backends.py
@@ -1207,6 +1207,39 @@ def test_multiindex_not_implemented(self) -> None:
                 pass
 
 
+class NetCDFBase(CFEncodedBase):
+    """Tests for all netCDF3 and netCDF4 backends."""
+
+    @pytest.mark.skipif(
+        ON_WINDOWS, reason="Windows does not allow modifying open files"
+    )
+    def test_refresh_from_disk(self) -> None:
+        # regression test for https://github.com/pydata/xarray/issues/4862
+
+        with create_tmp_file() as example_1_path:
+            with create_tmp_file() as example_1_modified_path:
+
+                with open_example_dataset("example_1.nc") as example_1:
+                    self.save(example_1, example_1_path)
+
+                    example_1.rh.values += 100
+                    self.save(example_1, example_1_modified_path)
+
+                a = open_dataset(example_1_path, engine=self.engine).load()
+
+                # Simulate external process modifying example_1.nc while this script is running
+                shutil.copy(example_1_modified_path, example_1_path)
+
+                # Reopen example_1.nc (modified) as `b`; note that `a` has NOT been closed
+                b = open_dataset(example_1_path, engine=self.engine).load()
+
+                try:
+                    assert not np.array_equal(a.rh.values, b.rh.values)
+                finally:
+                    a.close()
+                    b.close()
+
+
 _counter = itertools.count()
 
 
@@ -1238,7 +1271,7 @@ def create_tmp_files(
         yield files
 
 
-class NetCDF4Base(CFEncodedBase):
+class NetCDF4Base(NetCDFBase):
     """Tests for both netCDF4-python and h5netcdf."""
 
     engine: T_NetcdfEngine = "netcdf4"
@@ -1595,6 +1628,10 @@ def test_setncattr_string(self) -> None:
                 assert_array_equal(one_element_list_of_strings, totest.attrs["bar"])
                 assert one_string == totest.attrs["baz"]
 
+    @pytest.mark.skip(reason="https://github.com/Unidata/netcdf4-python/issues/1195")
+    def test_refresh_from_disk(self) -> None:
+        super().test_refresh_from_disk()
+
 
 @requires_netCDF4
 class TestNetCDF4AlreadyOpen:
@@ -3182,20 +3219,20 @@ def test_open_mfdataset_list_attr() -> None:
 
     with create_tmp_files(2) as nfiles:
         for i in range(2):
-            f = Dataset(nfiles[i], "w")
-            f.createDimension("x", 3)
-            vlvar = f.createVariable("test_var", np.int32, ("x"))
-            # here create an attribute as a list
-            vlvar.test_attr = [f"string a {i}", f"string b {i}"]
-            vlvar[:] = np.arange(3)
-            f.close()
-        ds1 = open_dataset(nfiles[0])
-        ds2 = open_dataset(nfiles[1])
-        original = xr.concat([ds1, ds2], dim="x")
-        with xr.open_mfdataset(
-            [nfiles[0], nfiles[1]], combine="nested", concat_dim="x"
-        ) as actual:
-            assert_identical(actual, original)
+            with Dataset(nfiles[i], "w") as f:
+                f.createDimension("x", 3)
+                vlvar = f.createVariable("test_var", np.int32, ("x"))
+                # here create an attribute as a list
+                vlvar.test_attr = [f"string a {i}", f"string b {i}"]
+                vlvar[:] = np.arange(3)
+
+        with open_dataset(nfiles[0]) as ds1:
+            with open_dataset(nfiles[1]) as ds2:
+                original = xr.concat([ds1, ds2], dim="x")
+                with xr.open_mfdataset(
+                    [nfiles[0], nfiles[1]], combine="nested", concat_dim="x"
+                ) as actual:
+                    assert_identical(actual, original)
 
 
 @requires_scipy_or_netCDF4
diff --git a/xarray/tests/test_backends_file_manager.py b/xarray/tests/test_backends_file_manager.py
--- a/xarray/tests/test_backends_file_manager.py
+++ b/xarray/tests/test_backends_file_manager.py
@@ -7,6 +7,7 @@
 
 import pytest
 
+# from xarray.backends import file_manager
 from xarray.backends.file_manager import CachingFileManager
 from xarray.backends.lru_cache import LRUCache
 from xarray.core.options import set_options
@@ -89,7 +90,7 @@ def test_file_manager_repr() -> None:
     assert "my-file" in repr(manager)
 
 
-def test_file_manager_refcounts() -> None:
+def test_file_manager_cache_and_refcounts() -> None:
     mock_file = mock.Mock()
     opener = mock.Mock(spec=open, return_value=mock_file)
     cache: dict = {}
@@ -97,47 +98,72 @@ def test_file_manager_refcounts() -> None:
 
     manager = CachingFileManager(opener, "filename", cache=cache, ref_counts=ref_counts)
     assert ref_counts[manager._key] == 1
+
+    assert not cache
     manager.acquire()
-    assert cache
+    assert len(cache) == 1
 
-    manager2 = CachingFileManager(
-        opener, "filename", cache=cache, ref_counts=ref_counts
-    )
-    assert cache
-    assert manager._key == manager2._key
-    assert ref_counts[manager._key] == 2
+    with set_options(warn_for_unclosed_files=False):
+        del manager
+        gc.collect()
+
+    assert not ref_counts
+    assert not cache
+
+
+def test_file_manager_cache_repeated_open() -> None:
+    mock_file = mock.Mock()
+    opener = mock.Mock(spec=open, return_value=mock_file)
+    cache: dict = {}
+
+    manager = CachingFileManager(opener, "filename", cache=cache)
+    manager.acquire()
+    assert len(cache) == 1
+
+    manager2 = CachingFileManager(opener, "filename", cache=cache)
+    manager2.acquire()
+    assert len(cache) == 2
 
     with set_options(warn_for_unclosed_files=False):
         del manager
         gc.collect()
 
-    assert cache
-    assert ref_counts[manager2._key] == 1
-    mock_file.close.assert_not_called()
+    assert len(cache) == 1
 
     with set_options(warn_for_unclosed_files=False):
         del manager2
         gc.collect()
 
-    assert not ref_counts
     assert not cache
 
 
-def test_file_manager_replace_object() -> None:
-    opener = mock.Mock()
+def test_file_manager_cache_with_pickle(tmpdir) -> None:
+
+    path = str(tmpdir.join("testing.txt"))
+    with open(path, "w") as f:
+        f.write("data")
     cache: dict = {}
-    ref_counts: dict = {}
 
-    manager = CachingFileManager(opener, "filename", cache=cache, ref_counts=ref_counts)
-    manager.acquire()
-    assert ref_counts[manager._key] == 1
-    assert cache
+    with mock.patch("xarray.backends.file_manager.FILE_CACHE", cache):
+        assert not cache
 
-    manager = CachingFileManager(opener, "filename", cache=cache, ref_counts=ref_counts)
-    assert ref_counts[manager._key] == 1
-    assert cache
+        manager = CachingFileManager(open, path, mode="r")
+        manager.acquire()
+        assert len(cache) == 1
 
-    manager.close()
+        manager2 = pickle.loads(pickle.dumps(manager))
+        manager2.acquire()
+        assert len(cache) == 1
+
+        with set_options(warn_for_unclosed_files=False):
+            del manager
+            gc.collect()
+        # assert len(cache) == 1
+
+        with set_options(warn_for_unclosed_files=False):
+            del manager2
+            gc.collect()
+        assert not cache
 
 
 def test_file_manager_write_consecutive(tmpdir, file_cache) -> None:

```


## Code snippets

### 1 - asv_bench/benchmarks/dataset_io.py:

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
### 2 - asv_bench/benchmarks/repr.py:

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
### 3 - asv_bench/benchmarks/dataset_io.py:

Start line: 222, End line: 296

```python
class IOMultipleNetCDF:
    """
    A few examples that benchmark reading/writing multiple netCDF files with
    xarray
    """

    timeout = 300.0
    repeat = 1
    number = 5

    def make_ds(self, nfiles=10):
        # TODO: Lazily skipped in CI as it is very demanding and slow.
        # Improve times and remove errors.
        _skip_slow()

        # multiple Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45
        self.nfiles = nfiles

        self.block_chunks = {
            "time": self.nt / 4,
            "lon": self.nx / 3,
            "lat": self.ny / 3,
        }

        self.time_chunks = {"time": int(self.nt / 36)}

        self.time_vars = np.split(
            pd.date_range("1970-01-01", periods=self.nt, freq="D"), self.nfiles
        )

        self.ds_list = []
        self.filenames_list = []
        for i, times in enumerate(self.time_vars):
            ds = xr.Dataset()
            nt = len(times)
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
            ds["foo"] = xr.DataArray(
                randn((nt, self.nx, self.ny), frac_nan=0.2),
                coords={"lon": lons, "lat": lats, "time": times},
                dims=("time", "lon", "lat"),
                name="foo",
                attrs={"units": "foo units", "description": "a description"},
            )
            ds["bar"] = xr.DataArray(
                randn((nt, self.nx, self.ny), frac_nan=0.2),
                coords={"lon": lons, "lat": lats, "time": times},
                dims=("time", "lon", "lat"),
                name="bar",
                attrs={"units": "bar units", "description": "a description"},
            )
            ds["baz"] = xr.DataArray(
                randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
                coords={"lon": lons, "lat": lats},
                dims=("lon", "lat"),
                name="baz",
                attrs={"units": "baz units", "description": "a description"},
            )

            ds.attrs = {"history": "created for xarray benchmarking"}

            self.ds_list.append(ds)
            self.filenames_list.append("test_netcdf_%i.nc" % i)
```
### 4 - asv_bench/benchmarks/reindexing.py:

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
### 5 - asv_bench/benchmarks/dataset_io.py:

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
### 6 - xarray/__init__.py:

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
### 7 - asv_bench/benchmarks/dataset_io.py:

Start line: 96, End line: 126

```python
class IOWriteSingleNetCDF3(IOSingleNetCDF):
    def setup(self):
        self.format = "NETCDF3_64BIT"
        self.make_ds()

    def time_write_dataset_netcdf4(self):
        self.ds.to_netcdf("test_netcdf4_write.nc", engine="netcdf4", format=self.format)

    def time_write_dataset_scipy(self):
        self.ds.to_netcdf("test_scipy_write.nc", engine="scipy", format=self.format)


class IOReadSingleNetCDF4(IOSingleNetCDF):
    def setup(self):

        self.make_ds()

        self.filepath = "test_single_file.nc4.nc"
        self.format = "NETCDF4"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4(self):
        xr.open_dataset(self.filepath, engine="netcdf4").load()

    def time_orthogonal_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4")
        ds = ds.isel(**self.oinds).load()

    def time_vectorized_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4")
        ds = ds.isel(**self.vinds).load()
```
### 8 - asv_bench/benchmarks/dataset_io.py:

Start line: 190, End line: 219

```python
class IOReadSingleNetCDF3Dask(IOReadSingleNetCDF4Dask):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = "test_single_file.nc3.nc"
        self.format = "NETCDF3_64BIT"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="scipy", chunks=self.block_chunks
            ).load()

    def time_load_dataset_scipy_with_block_chunks_oindexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy", chunks=self.block_chunks)
        ds = ds.isel(**self.oinds).load()

    def time_load_dataset_scipy_with_block_chunks_vindexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy", chunks=self.block_chunks)
        ds = ds.isel(**self.vinds).load()

    def time_load_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="scipy", chunks=self.time_chunks
            ).load()
```
### 9 - asv_bench/benchmarks/dataset_io.py:

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
### 10 - asv_bench/benchmarks/dataset_io.py:

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
