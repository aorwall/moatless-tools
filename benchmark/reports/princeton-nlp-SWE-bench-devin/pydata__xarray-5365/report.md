# pydata__xarray-5365

| **pydata/xarray** | `3960ea3ba08f81d211899827612550f6ac2de804` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 2528 |
| **Any found context length** | 718 |
| **Avg pos** | 4.5 |
| **Min pos** | 1 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/__init__.py b/xarray/__init__.py
--- a/xarray/__init__.py
+++ b/xarray/__init__.py
@@ -16,7 +16,16 @@
 from .core.alignment import align, broadcast
 from .core.combine import combine_by_coords, combine_nested
 from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
-from .core.computation import apply_ufunc, corr, cov, dot, polyval, unify_chunks, where
+from .core.computation import (
+    apply_ufunc,
+    corr,
+    cov,
+    cross,
+    dot,
+    polyval,
+    unify_chunks,
+    where,
+)
 from .core.concat import concat
 from .core.dataarray import DataArray
 from .core.dataset import Dataset
@@ -60,6 +69,7 @@
     "dot",
     "cov",
     "corr",
+    "cross",
     "full_like",
     "get_options",
     "infer_freq",
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -36,6 +36,7 @@
 
 if TYPE_CHECKING:
     from .coordinates import Coordinates
+    from .dataarray import DataArray
     from .dataset import Dataset
     from .types import T_Xarray
 
@@ -1373,6 +1374,214 @@ def _cov_corr(da_a, da_b, dim=None, ddof=0, method=None):
         return corr
 
 
+def cross(
+    a: Union[DataArray, Variable], b: Union[DataArray, Variable], *, dim: Hashable
+) -> Union[DataArray, Variable]:
+    """
+    Compute the cross product of two (arrays of) vectors.
+
+    The cross product of `a` and `b` in :math:`R^3` is a vector
+    perpendicular to both `a` and `b`. The vectors in `a` and `b` are
+    defined by the values along the dimension `dim` and can have sizes
+    1, 2 or 3. Where the size of either `a` or `b` is
+    1 or 2, the remaining components of the input vector is assumed to
+    be zero and the cross product calculated accordingly. In cases where
+    both input vectors have dimension 2, the z-component of the cross
+    product is returned.
+
+    Parameters
+    ----------
+    a, b : DataArray or Variable
+        Components of the first and second vector(s).
+    dim : hashable
+        The dimension along which the cross product will be computed.
+        Must be available in both vectors.
+
+    Examples
+    --------
+    Vector cross-product with 3 dimensions:
+
+    >>> a = xr.DataArray([1, 2, 3])
+    >>> b = xr.DataArray([4, 5, 6])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray (dim_0: 3)>
+    array([-3,  6, -3])
+    Dimensions without coordinates: dim_0
+
+    Vector cross-product with 2 dimensions, returns in the perpendicular
+    direction:
+
+    >>> a = xr.DataArray([1, 2])
+    >>> b = xr.DataArray([4, 5])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray ()>
+    array(-3)
+
+    Vector cross-product with 3 dimensions but zeros at the last axis
+    yields the same results as with 2 dimensions:
+
+    >>> a = xr.DataArray([1, 2, 0])
+    >>> b = xr.DataArray([4, 5, 0])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray (dim_0: 3)>
+    array([ 0,  0, -3])
+    Dimensions without coordinates: dim_0
+
+    One vector with dimension 2:
+
+    >>> a = xr.DataArray(
+    ...     [1, 2],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y"])),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [4, 5, 6],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (cartesian: 3)>
+    array([12, -6, -3])
+    Coordinates:
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    One vector with dimension 2 but coords in other positions:
+
+    >>> a = xr.DataArray(
+    ...     [1, 2],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "z"])),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [4, 5, 6],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (cartesian: 3)>
+    array([-10,   2,   5])
+    Coordinates:
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    Multiple vector cross-products. Note that the direction of the
+    cross product vector is defined by the right-hand rule:
+
+    >>> a = xr.DataArray(
+    ...     [[1, 2, 3], [4, 5, 6]],
+    ...     dims=("time", "cartesian"),
+    ...     coords=dict(
+    ...         time=(["time"], [0, 1]),
+    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
+    ...     ),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [[4, 5, 6], [1, 2, 3]],
+    ...     dims=("time", "cartesian"),
+    ...     coords=dict(
+    ...         time=(["time"], [0, 1]),
+    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
+    ...     ),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (time: 2, cartesian: 3)>
+    array([[-3,  6, -3],
+           [ 3, -6,  3]])
+    Coordinates:
+      * time       (time) int64 0 1
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    Cross can be called on Datasets by converting to DataArrays and later
+    back to a Dataset:
+
+    >>> ds_a = xr.Dataset(dict(x=("dim_0", [1]), y=("dim_0", [2]), z=("dim_0", [3])))
+    >>> ds_b = xr.Dataset(dict(x=("dim_0", [4]), y=("dim_0", [5]), z=("dim_0", [6])))
+    >>> c = xr.cross(
+    ...     ds_a.to_array("cartesian"), ds_b.to_array("cartesian"), dim="cartesian"
+    ... )
+    >>> c.to_dataset(dim="cartesian")
+    <xarray.Dataset>
+    Dimensions:  (dim_0: 1)
+    Dimensions without coordinates: dim_0
+    Data variables:
+        x        (dim_0) int64 -3
+        y        (dim_0) int64 6
+        z        (dim_0) int64 -3
+
+    See Also
+    --------
+    numpy.cross : Corresponding numpy function
+    """
+
+    if dim not in a.dims:
+        raise ValueError(f"Dimension {dim!r} not on a")
+    elif dim not in b.dims:
+        raise ValueError(f"Dimension {dim!r} not on b")
+
+    if not 1 <= a.sizes[dim] <= 3:
+        raise ValueError(
+            f"The size of {dim!r} on a must be 1, 2, or 3 to be "
+            f"compatible with a cross product but is {a.sizes[dim]}"
+        )
+    elif not 1 <= b.sizes[dim] <= 3:
+        raise ValueError(
+            f"The size of {dim!r} on b must be 1, 2, or 3 to be "
+            f"compatible with a cross product but is {b.sizes[dim]}"
+        )
+
+    all_dims = list(dict.fromkeys(a.dims + b.dims))
+
+    if a.sizes[dim] != b.sizes[dim]:
+        # Arrays have different sizes. Append zeros where the smaller
+        # array is missing a value, zeros will not affect np.cross:
+
+        if (
+            not isinstance(a, Variable)  # Only used to make mypy happy.
+            and dim in getattr(a, "coords", {})
+            and not isinstance(b, Variable)  # Only used to make mypy happy.
+            and dim in getattr(b, "coords", {})
+        ):
+            # If the arrays have coords we know which indexes to fill
+            # with zeros:
+            a, b = align(
+                a,
+                b,
+                fill_value=0,
+                join="outer",
+                exclude=set(all_dims) - {dim},
+            )
+        elif min(a.sizes[dim], b.sizes[dim]) == 2:
+            # If the array doesn't have coords we can only infer
+            # that it has composite values if the size is at least 2.
+            # Once padded, rechunk the padded array because apply_ufunc
+            # requires core dimensions not to be chunked:
+            if a.sizes[dim] < b.sizes[dim]:
+                a = a.pad({dim: (0, 1)}, constant_values=0)
+                # TODO: Should pad or apply_ufunc handle correct chunking?
+                a = a.chunk({dim: -1}) if is_duck_dask_array(a.data) else a
+            else:
+                b = b.pad({dim: (0, 1)}, constant_values=0)
+                # TODO: Should pad or apply_ufunc handle correct chunking?
+                b = b.chunk({dim: -1}) if is_duck_dask_array(b.data) else b
+        else:
+            raise ValueError(
+                f"{dim!r} on {'a' if a.sizes[dim] == 1 else 'b'} is incompatible:"
+                " dimensions without coordinates must have have a length of 2 or 3"
+            )
+
+    c = apply_ufunc(
+        np.cross,
+        a,
+        b,
+        input_core_dims=[[dim], [dim]],
+        output_core_dims=[[dim] if a.sizes[dim] == 3 else []],
+        dask="parallelized",
+        output_dtypes=[np.result_type(a, b)],
+    )
+    c = c.transpose(*all_dims, missing_dims="ignore")
+
+    return c
+
+
 def dot(*arrays, dims=None, **kwargs):
     """Generalized dot product for xarray objects. Like np.einsum, but
     provides a simpler interface based on array dimensions.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/__init__.py | 19 | 19 | 4 | 4 | 2528
| xarray/__init__.py | 63 | 63 | 4 | 4 | 2528
| xarray/core/computation.py | 39 | 39 | - | 1 | -
| xarray/core/computation.py | 1376 | 1376 | 1 | 1 | 718


## Problem Statement

```
Feature request: vector cross product
xarray currently has the `xarray.dot()` function for calculating arbitrary dot products which is indeed very handy.
Sometimes, especially for physical applications I also need a vector cross product. I' wondering whether you would be interested in having ` xarray.cross` as a wrapper of [`numpy.cross`.](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cross.html) I currently use the following implementation:

\`\`\`python
def cross(a, b, spatial_dim, output_dtype=None):
    """xarray-compatible cross product
    
    Compatible with dask, parallelization uses a.dtype as output_dtype
    """
    # TODO find spatial dim default by looking for unique 3(or 2)-valued dim?
    for d in (a, b):
        if spatial_dim not in d.dims:
            raise ValueError('dimension {} not in {}'.format(spatial_dim, d))
        if d.sizes[spatial_dim] != 3:  #TODO handle 2-valued cases
            raise ValueError('dimension {} has not length 3 in {}'.format(d))
        
    if output_dtype is None: 
        output_dtype = a.dtype  # TODO some better way to determine default?
    c = xr.apply_ufunc(np.cross, a, b,
                       input_core_dims=[[spatial_dim], [spatial_dim]], 
                       output_core_dims=[[spatial_dim]], 
                       dask='parallelized', output_dtypes=[output_dtype]
                      )
    return c

\`\`\`

#### Example usage

\`\`\`python
import numpy as np
import xarray as xr
a = xr.DataArray(np.empty((10, 3)), dims=['line', 'cartesian'])
b = xr.full_like(a, 1)
c = cross(a, b, 'cartesian')
\`\`\`

#### Main question
Do you want such a function (and possibly associated `DataArray.cross` methods) in the `xarray` namespace, or should it be in some other package?  I didn't find a package which would be a good fit as this is close to core numpy functionality and isn't as domain specific as some geo packages. I'm not aware of some "xrphysics" package.

I could make a PR if you'd want to have it in `xarray` directly.

#### Output of ``xr.show_versions()``
<details>
# Paste the output here xr.show_versions() here
INSTALLED VERSIONS
------------------
commit: None
python: 3.7.3 (default, Mar 27 2019, 22:11:17) 
[GCC 7.3.0]
python-bits: 64
OS: Linux
OS-release: 4.9.0-9-amd64
machine: x86_64
processor: 
byteorder: little
LC_ALL: None
LANG: en_US.UTF-8
LOCALE: en_US.UTF-8
libhdf5: 1.10.4
libnetcdf: 4.6.1

xarray: 0.12.3
pandas: 0.24.2
numpy: 1.16.4
scipy: 1.3.0
netCDF4: 1.4.2
pydap: None
h5netcdf: 0.7.4
h5py: 2.9.0
Nio: None
zarr: None
cftime: 1.0.3.4
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: 1.2.1
dask: 2.1.0
distributed: 2.1.0
matplotlib: 3.1.0
cartopy: None
seaborn: 0.9.0
numbagg: None
setuptools: 41.0.1
pip: 19.1.1
conda: 4.7.11
pytest: 5.0.1
IPython: 7.6.1
sphinx: 2.1.2
</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 xarray/core/computation.py** | 1376 | 1454| 718 | 718 | 14806 | 
| 2 | 2 xarray/core/dataarray.py | 3301 | 3359| 416 | 1134 | 54923 | 
| 3 | 3 xarray/core/duck_array_ops.py | 1 | 93| 727 | 1861 | 60506 | 
| **-> 4 <-** | **4 xarray/__init__.py** | 1 | 100| 667 | 2528 | 61173 | 
| 5 | 4 xarray/core/dataarray.py | 1 | 89| 465 | 2993 | 61173 | 
| 6 | **4 xarray/core/computation.py** | 1456 | 1518| 601 | 3594 | 61173 | 
| 7 | 5 xarray/core/dataset.py | 1 | 135| 661 | 4255 | 126802 | 
| 8 | 6 xarray/ufuncs.py | 1 | 36| 239 | 4494 | 128019 | 
| 9 | 7 xarray/core/types.py | 1 | 34| 297 | 4791 | 128316 | 
| 10 | 8 xarray/core/dask_array_compat.py | 117 | 187| 652 | 5443 | 129853 | 
| 11 | 9 asv_bench/benchmarks/interp.py | 1 | 18| 150 | 5593 | 130304 | 
| 12 | 10 xarray/core/common.py | 1 | 52| 258 | 5851 | 146191 | 
| 13 | 11 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 6589 | 147692 | 
| 14 | 12 xarray/core/arithmetic.py | 41 | 90| 458 | 7047 | 148651 | 
| 15 | 13 xarray/core/npcompat.py | 31 | 79| 390 | 7437 | 150289 | 
| 16 | 14 xarray/core/pycompat.py | 46 | 68| 129 | 7566 | 150708 | 
| 17 | 15 xarray/core/parallel.py | 1 | 52| 239 | 7805 | 155836 | 
| 18 | 16 asv_bench/benchmarks/rolling.py | 1 | 17| 113 | 7918 | 157111 | 
| 19 | 16 xarray/core/duck_array_ops.py | 578 | 654| 620 | 8538 | 157111 | 
| 20 | 16 xarray/core/dataarray.py | 218 | 340| 1282 | 9820 | 157111 | 
| 21 | 17 xarray/plot/utils.py | 1 | 66| 321 | 10141 | 166354 | 
| 22 | 17 xarray/core/dataarray.py | 3071 | 3093| 188 | 10329 | 166354 | 
| 23 | 18 xarray/core/indexing.py | 1 | 22| 124 | 10453 | 177666 | 
| 24 | 18 xarray/core/dataarray.py | 1621 | 1749| 1358 | 11811 | 177666 | 
| 25 | 18 xarray/core/dataarray.py | 1223 | 1339| 1181 | 12992 | 177666 | 
| 26 | 19 xarray/testing.py | 119 | 173| 490 | 13482 | 180755 | 
| 27 | 20 asv_bench/benchmarks/reindexing.py | 1 | 53| 415 | 13897 | 181170 | 
| 28 | 21 xarray/core/missing.py | 1 | 18| 139 | 14036 | 187381 | 
| 29 | 21 xarray/core/pycompat.py | 1 | 43| 288 | 14324 | 187381 | 
| 30 | 21 xarray/core/arithmetic.py | 93 | 146| 258 | 14582 | 187381 | 
| 31 | 22 xarray/core/alignment.py | 85 | 272| 1943 | 16525 | 193650 | 
| 32 | 22 xarray/core/dataarray.py | 764 | 836| 574 | 17099 | 193650 | 
| 33 | 23 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 17266 | 193817 | 
| 34 | 23 xarray/core/missing.py | 699 | 766| 568 | 17834 | 193817 | 
| 35 | 23 xarray/core/dataarray.py | 3928 | 4078| 1787 | 19621 | 193817 | 


### Hint

```
Very useful :+1: 
I would add:
\`\`\`
    try:
        c.attrs["units"] = a.attrs["units"] + '*' + b.attrs["units"]
    except KeyError:
        pass
\`\`\`
to preserve units - but I am not sure that is in scope for xarray.
it is not, but we have been working on [unit aware arrays with `pint`](https://github.com/pydata/xarray/issues/3594). Once that is done, unit propagation should work automatically.
```

## Patch

```diff
diff --git a/xarray/__init__.py b/xarray/__init__.py
--- a/xarray/__init__.py
+++ b/xarray/__init__.py
@@ -16,7 +16,16 @@
 from .core.alignment import align, broadcast
 from .core.combine import combine_by_coords, combine_nested
 from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
-from .core.computation import apply_ufunc, corr, cov, dot, polyval, unify_chunks, where
+from .core.computation import (
+    apply_ufunc,
+    corr,
+    cov,
+    cross,
+    dot,
+    polyval,
+    unify_chunks,
+    where,
+)
 from .core.concat import concat
 from .core.dataarray import DataArray
 from .core.dataset import Dataset
@@ -60,6 +69,7 @@
     "dot",
     "cov",
     "corr",
+    "cross",
     "full_like",
     "get_options",
     "infer_freq",
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -36,6 +36,7 @@
 
 if TYPE_CHECKING:
     from .coordinates import Coordinates
+    from .dataarray import DataArray
     from .dataset import Dataset
     from .types import T_Xarray
 
@@ -1373,6 +1374,214 @@ def _cov_corr(da_a, da_b, dim=None, ddof=0, method=None):
         return corr
 
 
+def cross(
+    a: Union[DataArray, Variable], b: Union[DataArray, Variable], *, dim: Hashable
+) -> Union[DataArray, Variable]:
+    """
+    Compute the cross product of two (arrays of) vectors.
+
+    The cross product of `a` and `b` in :math:`R^3` is a vector
+    perpendicular to both `a` and `b`. The vectors in `a` and `b` are
+    defined by the values along the dimension `dim` and can have sizes
+    1, 2 or 3. Where the size of either `a` or `b` is
+    1 or 2, the remaining components of the input vector is assumed to
+    be zero and the cross product calculated accordingly. In cases where
+    both input vectors have dimension 2, the z-component of the cross
+    product is returned.
+
+    Parameters
+    ----------
+    a, b : DataArray or Variable
+        Components of the first and second vector(s).
+    dim : hashable
+        The dimension along which the cross product will be computed.
+        Must be available in both vectors.
+
+    Examples
+    --------
+    Vector cross-product with 3 dimensions:
+
+    >>> a = xr.DataArray([1, 2, 3])
+    >>> b = xr.DataArray([4, 5, 6])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray (dim_0: 3)>
+    array([-3,  6, -3])
+    Dimensions without coordinates: dim_0
+
+    Vector cross-product with 2 dimensions, returns in the perpendicular
+    direction:
+
+    >>> a = xr.DataArray([1, 2])
+    >>> b = xr.DataArray([4, 5])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray ()>
+    array(-3)
+
+    Vector cross-product with 3 dimensions but zeros at the last axis
+    yields the same results as with 2 dimensions:
+
+    >>> a = xr.DataArray([1, 2, 0])
+    >>> b = xr.DataArray([4, 5, 0])
+    >>> xr.cross(a, b, dim="dim_0")
+    <xarray.DataArray (dim_0: 3)>
+    array([ 0,  0, -3])
+    Dimensions without coordinates: dim_0
+
+    One vector with dimension 2:
+
+    >>> a = xr.DataArray(
+    ...     [1, 2],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y"])),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [4, 5, 6],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (cartesian: 3)>
+    array([12, -6, -3])
+    Coordinates:
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    One vector with dimension 2 but coords in other positions:
+
+    >>> a = xr.DataArray(
+    ...     [1, 2],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "z"])),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [4, 5, 6],
+    ...     dims=["cartesian"],
+    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (cartesian: 3)>
+    array([-10,   2,   5])
+    Coordinates:
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    Multiple vector cross-products. Note that the direction of the
+    cross product vector is defined by the right-hand rule:
+
+    >>> a = xr.DataArray(
+    ...     [[1, 2, 3], [4, 5, 6]],
+    ...     dims=("time", "cartesian"),
+    ...     coords=dict(
+    ...         time=(["time"], [0, 1]),
+    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
+    ...     ),
+    ... )
+    >>> b = xr.DataArray(
+    ...     [[4, 5, 6], [1, 2, 3]],
+    ...     dims=("time", "cartesian"),
+    ...     coords=dict(
+    ...         time=(["time"], [0, 1]),
+    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
+    ...     ),
+    ... )
+    >>> xr.cross(a, b, dim="cartesian")
+    <xarray.DataArray (time: 2, cartesian: 3)>
+    array([[-3,  6, -3],
+           [ 3, -6,  3]])
+    Coordinates:
+      * time       (time) int64 0 1
+      * cartesian  (cartesian) <U1 'x' 'y' 'z'
+
+    Cross can be called on Datasets by converting to DataArrays and later
+    back to a Dataset:
+
+    >>> ds_a = xr.Dataset(dict(x=("dim_0", [1]), y=("dim_0", [2]), z=("dim_0", [3])))
+    >>> ds_b = xr.Dataset(dict(x=("dim_0", [4]), y=("dim_0", [5]), z=("dim_0", [6])))
+    >>> c = xr.cross(
+    ...     ds_a.to_array("cartesian"), ds_b.to_array("cartesian"), dim="cartesian"
+    ... )
+    >>> c.to_dataset(dim="cartesian")
+    <xarray.Dataset>
+    Dimensions:  (dim_0: 1)
+    Dimensions without coordinates: dim_0
+    Data variables:
+        x        (dim_0) int64 -3
+        y        (dim_0) int64 6
+        z        (dim_0) int64 -3
+
+    See Also
+    --------
+    numpy.cross : Corresponding numpy function
+    """
+
+    if dim not in a.dims:
+        raise ValueError(f"Dimension {dim!r} not on a")
+    elif dim not in b.dims:
+        raise ValueError(f"Dimension {dim!r} not on b")
+
+    if not 1 <= a.sizes[dim] <= 3:
+        raise ValueError(
+            f"The size of {dim!r} on a must be 1, 2, or 3 to be "
+            f"compatible with a cross product but is {a.sizes[dim]}"
+        )
+    elif not 1 <= b.sizes[dim] <= 3:
+        raise ValueError(
+            f"The size of {dim!r} on b must be 1, 2, or 3 to be "
+            f"compatible with a cross product but is {b.sizes[dim]}"
+        )
+
+    all_dims = list(dict.fromkeys(a.dims + b.dims))
+
+    if a.sizes[dim] != b.sizes[dim]:
+        # Arrays have different sizes. Append zeros where the smaller
+        # array is missing a value, zeros will not affect np.cross:
+
+        if (
+            not isinstance(a, Variable)  # Only used to make mypy happy.
+            and dim in getattr(a, "coords", {})
+            and not isinstance(b, Variable)  # Only used to make mypy happy.
+            and dim in getattr(b, "coords", {})
+        ):
+            # If the arrays have coords we know which indexes to fill
+            # with zeros:
+            a, b = align(
+                a,
+                b,
+                fill_value=0,
+                join="outer",
+                exclude=set(all_dims) - {dim},
+            )
+        elif min(a.sizes[dim], b.sizes[dim]) == 2:
+            # If the array doesn't have coords we can only infer
+            # that it has composite values if the size is at least 2.
+            # Once padded, rechunk the padded array because apply_ufunc
+            # requires core dimensions not to be chunked:
+            if a.sizes[dim] < b.sizes[dim]:
+                a = a.pad({dim: (0, 1)}, constant_values=0)
+                # TODO: Should pad or apply_ufunc handle correct chunking?
+                a = a.chunk({dim: -1}) if is_duck_dask_array(a.data) else a
+            else:
+                b = b.pad({dim: (0, 1)}, constant_values=0)
+                # TODO: Should pad or apply_ufunc handle correct chunking?
+                b = b.chunk({dim: -1}) if is_duck_dask_array(b.data) else b
+        else:
+            raise ValueError(
+                f"{dim!r} on {'a' if a.sizes[dim] == 1 else 'b'} is incompatible:"
+                " dimensions without coordinates must have have a length of 2 or 3"
+            )
+
+    c = apply_ufunc(
+        np.cross,
+        a,
+        b,
+        input_core_dims=[[dim], [dim]],
+        output_core_dims=[[dim] if a.sizes[dim] == 3 else []],
+        dask="parallelized",
+        output_dtypes=[np.result_type(a, b)],
+    )
+    c = c.transpose(*all_dims, missing_dims="ignore")
+
+    return c
+
+
 def dot(*arrays, dims=None, **kwargs):
     """Generalized dot product for xarray objects. Like np.einsum, but
     provides a simpler interface based on array dimensions.

```

## Test Patch

```diff
diff --git a/xarray/tests/test_computation.py b/xarray/tests/test_computation.py
--- a/xarray/tests/test_computation.py
+++ b/xarray/tests/test_computation.py
@@ -1952,3 +1952,110 @@ def test_polyval(use_dask, use_datetime) -> None:
     da_pv = xr.polyval(da.x, coeffs)
 
     xr.testing.assert_allclose(da, da_pv.T)
+
+
+@pytest.mark.parametrize("use_dask", [False, True])
+@pytest.mark.parametrize(
+    "a, b, ae, be, dim, axis",
+    [
+        [
+            xr.DataArray([1, 2, 3]),
+            xr.DataArray([4, 5, 6]),
+            [1, 2, 3],
+            [4, 5, 6],
+            "dim_0",
+            -1,
+        ],
+        [
+            xr.DataArray([1, 2]),
+            xr.DataArray([4, 5, 6]),
+            [1, 2],
+            [4, 5, 6],
+            "dim_0",
+            -1,
+        ],
+        [
+            xr.Variable(dims=["dim_0"], data=[1, 2, 3]),
+            xr.Variable(dims=["dim_0"], data=[4, 5, 6]),
+            [1, 2, 3],
+            [4, 5, 6],
+            "dim_0",
+            -1,
+        ],
+        [
+            xr.Variable(dims=["dim_0"], data=[1, 2]),
+            xr.Variable(dims=["dim_0"], data=[4, 5, 6]),
+            [1, 2],
+            [4, 5, 6],
+            "dim_0",
+            -1,
+        ],
+        [  # Test dim in the middle:
+            xr.DataArray(
+                np.arange(0, 5 * 3 * 4).reshape((5, 3, 4)),
+                dims=["time", "cartesian", "var"],
+                coords=dict(
+                    time=(["time"], np.arange(0, 5)),
+                    cartesian=(["cartesian"], ["x", "y", "z"]),
+                    var=(["var"], [1, 1.5, 2, 2.5]),
+                ),
+            ),
+            xr.DataArray(
+                np.arange(0, 5 * 3 * 4).reshape((5, 3, 4)) + 1,
+                dims=["time", "cartesian", "var"],
+                coords=dict(
+                    time=(["time"], np.arange(0, 5)),
+                    cartesian=(["cartesian"], ["x", "y", "z"]),
+                    var=(["var"], [1, 1.5, 2, 2.5]),
+                ),
+            ),
+            np.arange(0, 5 * 3 * 4).reshape((5, 3, 4)),
+            np.arange(0, 5 * 3 * 4).reshape((5, 3, 4)) + 1,
+            "cartesian",
+            1,
+        ],
+        [  # Test 1 sized arrays with coords:
+            xr.DataArray(
+                np.array([1]),
+                dims=["cartesian"],
+                coords=dict(cartesian=(["cartesian"], ["z"])),
+            ),
+            xr.DataArray(
+                np.array([4, 5, 6]),
+                dims=["cartesian"],
+                coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+            ),
+            [0, 0, 1],
+            [4, 5, 6],
+            "cartesian",
+            -1,
+        ],
+        [  # Test filling inbetween with coords:
+            xr.DataArray(
+                [1, 2],
+                dims=["cartesian"],
+                coords=dict(cartesian=(["cartesian"], ["x", "z"])),
+            ),
+            xr.DataArray(
+                [4, 5, 6],
+                dims=["cartesian"],
+                coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
+            ),
+            [1, 0, 2],
+            [4, 5, 6],
+            "cartesian",
+            -1,
+        ],
+    ],
+)
+def test_cross(a, b, ae, be, dim: str, axis: int, use_dask: bool) -> None:
+    expected = np.cross(ae, be, axis=axis)
+
+    if use_dask:
+        if not has_dask:
+            pytest.skip("test for dask.")
+        a = a.chunk()
+        b = b.chunk()
+
+    actual = xr.cross(a, b, dim=dim)
+    xr.testing.assert_duckarray_allclose(expected, actual)

```


## Code snippets

### 1 - xarray/core/computation.py:

Start line: 1376, End line: 1454

```python
def dot(*arrays, dims=None, **kwargs):
    """Generalized dot product for xarray objects. Like np.einsum, but
    provides a simpler interface based on array dimensions.

    Parameters
    ----------
    *arrays : DataArray or Variable
        Arrays to compute.
    dims : ..., str or tuple of str, optional
        Which dimensions to sum over. Ellipsis ('...') sums over all dimensions.
        If not specified, then all the common dimensions are summed over.
    **kwargs : dict
        Additional keyword arguments passed to numpy.einsum or
        dask.array.einsum

    Returns
    -------
    DataArray

    Examples
    --------
    >>> da_a = xr.DataArray(np.arange(3 * 2).reshape(3, 2), dims=["a", "b"])
    >>> da_b = xr.DataArray(np.arange(3 * 2 * 2).reshape(3, 2, 2), dims=["a", "b", "c"])
    >>> da_c = xr.DataArray(np.arange(2 * 3).reshape(2, 3), dims=["c", "d"])

    >>> da_a
    <xarray.DataArray (a: 3, b: 2)>
    array([[0, 1],
           [2, 3],
           [4, 5]])
    Dimensions without coordinates: a, b

    >>> da_b
    <xarray.DataArray (a: 3, b: 2, c: 2)>
    array([[[ 0,  1],
            [ 2,  3]],
    <BLANKLINE>
           [[ 4,  5],
            [ 6,  7]],
    <BLANKLINE>
           [[ 8,  9],
            [10, 11]]])
    Dimensions without coordinates: a, b, c

    >>> da_c
    <xarray.DataArray (c: 2, d: 3)>
    array([[0, 1, 2],
           [3, 4, 5]])
    Dimensions without coordinates: c, d

    >>> xr.dot(da_a, da_b, dims=["a", "b"])
    <xarray.DataArray (c: 2)>
    array([110, 125])
    Dimensions without coordinates: c

    >>> xr.dot(da_a, da_b, dims=["a"])
    <xarray.DataArray (b: 2, c: 2)>
    array([[40, 46],
           [70, 79]])
    Dimensions without coordinates: b, c

    >>> xr.dot(da_a, da_b, da_c, dims=["b", "c"])
    <xarray.DataArray (a: 3, d: 3)>
    array([[  9,  14,  19],
           [ 93, 150, 207],
           [273, 446, 619]])
    Dimensions without coordinates: a, d

    >>> xr.dot(da_a, da_b)
    <xarray.DataArray (c: 2)>
    array([110, 125])
    Dimensions without coordinates: c

    >>> xr.dot(da_a, da_b, dims=...)
    <xarray.DataArray ()>
    array(235)
    """
    from .dataarray import DataArray
    from .variable import Variable
    # ... other code
```
### 2 - xarray/core/dataarray.py:

Start line: 3301, End line: 3359

```python
class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):

    @property
    def real(self) -> "DataArray":
        return self._replace(self.variable.real)

    @property
    def imag(self) -> "DataArray":
        return self._replace(self.variable.imag)

    def dot(
        self, other: "DataArray", dims: Union[Hashable, Sequence[Hashable], None] = None
    ) -> "DataArray":
        """Perform dot product of two DataArrays along their shared dims.

        Equivalent to taking taking tensordot over all shared dims.

        Parameters
        ----------
        other : DataArray
            The other array with which the dot product is performed.
        dims : ..., hashable or sequence of hashable, optional
            Which dimensions to sum over. Ellipsis (`...`) sums over all dimensions.
            If not specified, then all the common dimensions are summed over.

        Returns
        -------
        result : DataArray
            Array resulting from the dot product over all shared dimensions.

        See Also
        --------
        dot
        numpy.tensordot

        Examples
        --------
        >>> da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        >>> da = xr.DataArray(da_vals, dims=["x", "y", "z"])
        >>> dm_vals = np.arange(4)
        >>> dm = xr.DataArray(dm_vals, dims=["z"])

        >>> dm.dims
        ('z',)

        >>> da.dims
        ('x', 'y', 'z')

        >>> dot_result = da.dot(dm)
        >>> dot_result.dims
        ('x', 'y')

        """
        if isinstance(other, Dataset):
            raise NotImplementedError(
                "dot products are not yet supported with Dataset objects."
            )
        if not isinstance(other, DataArray):
            raise TypeError("dot only operates on DataArrays.")

        return computation.dot(self, other, dims=dims)
```
### 3 - xarray/core/duck_array_ops.py:

Start line: 1, End line: 93

```python
"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
import contextlib
import datetime
import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import zeros_like  # noqa
from numpy import around, broadcast_to  # noqa
from numpy import concatenate as _concatenate
from numpy import einsum, isclose, isin, isnan, isnat, pad  # noqa
from numpy import stack as _stack
from numpy import take, tensordot, transpose, unravel_index  # noqa
from numpy import where as _where
from packaging.version import Version

from . import dask_array_compat, dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import (
    cupy_array_type,
    dask_array_type,
    is_duck_dask_array,
    sparse_array_type,
    sparse_version,
)
from .utils import is_duck_array

try:
    import dask.array as dask_array
    from dask.base import tokenize
except ImportError:
    dask_array = None


def _dask_or_eager_func(
    name,
    eager_module=np,
    dask_module=dask_array,
):
    """Create a function that dispatches to dask for dask array inputs."""

    def f(*args, **kwargs):
        if any(is_duck_dask_array(a) for a in args):
            wrapped = getattr(dask_module, name)
        else:
            wrapped = getattr(eager_module, name)
        return wrapped(*args, **kwargs)

    return f


def fail_on_dask_array_input(values, msg=None, func_name=None):
    if is_duck_dask_array(values):
        if msg is None:
            msg = "%r is not yet a valid method on dask arrays"
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)


# Requires special-casing because pandas won't automatically dispatch to dask.isnull via NEP-18
pandas_isnull = _dask_or_eager_func("isnull", eager_module=pd, dask_module=dask_array)

# np.around has failing doctests, overwrite it so they pass:
# https://github.com/numpy/numpy/issues/19759
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.])",
    "array([0., 2.])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.])",
    "array([0., 2.])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.4,  1.6])",
    "array([0.4, 1.6])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.,  2.,  4.,  4.])",
    "array([0., 2., 2., 4., 4.])",
)
```
### 4 - xarray/__init__.py:

Start line: 1, End line: 100

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
from .coding.cftime_offsets import cftime_range
from .coding.cftimeindex import CFTimeIndex
from .coding.frequencies import infer_freq
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast
from .core.combine import combine_by_coords, combine_nested
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import apply_ufunc, corr, cov, dot, polyval, unify_chunks, where
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
    "decode_cf",
    "dot",
    "cov",
    "corr",
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
### 5 - xarray/core/dataarray.py:

Start line: 1, End line: 89

```python
from __future__ import annotations

import datetime
import warnings
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
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..plot.plot import _PlotMethods
from ..plot.utils import _get_units_from_attrs
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
from .arithmetic import DataArrayArithmetic
from .common import AbstractArray, DataWithCoords, get_chunksizes
from .computation import unify_chunks
from .coordinates import (
    DataArrayCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Index, Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

if TYPE_CHECKING:
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

    from .types import T_DataArray, T_Xarray
```
### 6 - xarray/core/computation.py:

Start line: 1456, End line: 1518

```python
def dot(*arrays, dims=None, **kwargs):
    # ... other code

    if any(not isinstance(arr, (Variable, DataArray)) for arr in arrays):
        raise TypeError(
            "Only xr.DataArray and xr.Variable are supported."
            "Given {}.".format([type(arr) for arr in arrays])
        )

    if len(arrays) == 0:
        raise TypeError("At least one array should be given.")

    if isinstance(dims, str):
        dims = (dims,)

    common_dims = set.intersection(*[set(arr.dims) for arr in arrays])
    all_dims = []
    for arr in arrays:
        all_dims += [d for d in arr.dims if d not in all_dims]

    einsum_axes = "abcdefghijklmnopqrstuvwxyz"
    dim_map = {d: einsum_axes[i] for i, d in enumerate(all_dims)}

    if dims is ...:
        dims = all_dims
    elif dims is None:
        # find dimensions that occur more than one times
        dim_counts = Counter()
        for arr in arrays:
            dim_counts.update(arr.dims)
        dims = tuple(d for d, c in dim_counts.items() if c > 1)

    dims = tuple(dims)  # make dims a tuple

    # dimensions to be parallelized
    broadcast_dims = tuple(d for d in all_dims if d in common_dims and d not in dims)
    input_core_dims = [
        [d for d in arr.dims if d not in broadcast_dims] for arr in arrays
    ]
    output_core_dims = [tuple(d for d in all_dims if d not in dims + broadcast_dims)]

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
### 7 - xarray/core/dataset.py:

Start line: 1, End line: 135

```python
import copy
import datetime
import inspect
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
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
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from .computation import unify_chunks
from .coordinates import (
    DatasetCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
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
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
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
    from .types import T_Xarray

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
### 8 - xarray/ufuncs.py:

Start line: 1, End line: 36

```python
"""xarray specific universal functions

Handles unary and binary operations for the following types, in ascending
priority order:
- scalars
- numpy.ndarray
- dask.array.Array
- xarray.Variable
- xarray.DataArray
- xarray.Dataset
- xarray.core.groupby.GroupBy

Once NumPy 1.10 comes out with support for overriding ufuncs, this module will
hopefully no longer be necessary.
"""
import textwrap
import warnings as _warnings

import numpy as _np

from .core.dataarray import DataArray as _DataArray
from .core.dataset import Dataset as _Dataset
from .core.groupby import GroupBy as _GroupBy
from .core.pycompat import dask_array_type as _dask_array_type
from .core.variable import Variable as _Variable

_xarray_types = (_Variable, _DataArray, _Dataset, _GroupBy)
_dispatch_order = (_np.ndarray, _dask_array_type) + _xarray_types
_UNDEFINED = object()


def _dispatch_priority(obj):
    for priority, cls in enumerate(_dispatch_order):
        if isinstance(obj, cls):
            return priority
    return -1
```
### 9 - xarray/core/types.py:

Start line: 1, End line: 34

```python
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from .common import DataWithCoords
    from .dataarray import DataArray
    from .dataset import Dataset
    from .groupby import DataArrayGroupBy, GroupBy
    from .npcompat import ArrayLike
    from .variable import Variable

    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray


T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")

# Maybe we rename this to T_Data or something less Fortran-y?
T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")

ScalarOrArray = Union["ArrayLike", np.generic, np.ndarray, "DaskArray"]
DsCompatible = Union["Dataset", "DataArray", "Variable", "GroupBy", "ScalarOrArray"]
DaCompatible = Union["DataArray", "Variable", "DataArrayGroupBy", "ScalarOrArray"]
VarCompatible = Union["Variable", "ScalarOrArray"]
GroupByIncompatible = Union["Variable", "GroupBy"]
```
### 10 - xarray/core/dask_array_compat.py:

Start line: 117, End line: 187

```python
if dask_version > Version("2021.03.0"):
    sliding_window_view = da.lib.stride_tricks.sliding_window_view
else:

    def sliding_window_view(x, window_shape, axis=None):
        from dask.array.overlap import map_overlap
        from numpy.core.numeric import normalize_axis_tuple

        from .npcompat import sliding_window_view as _np_sliding_window_view

        window_shape = (
            tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
        )

        window_shape_array = np.array(window_shape)
        if np.any(window_shape_array <= 0):
            raise ValueError("`window_shape` must contain positive values")

        if axis is None:
            axis = tuple(range(x.ndim))
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Since axis is `None`, must provide "
                    f"window_shape for all dimensions of `x`; "
                    f"got {len(window_shape)} window_shape elements "
                    f"and `x.ndim` is {x.ndim}."
                )
        else:
            axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Must provide matching length window_shape and "
                    f"axis; got {len(window_shape)} window_shape "
                    f"elements and {len(axis)} axes elements."
                )

        depths = [0] * x.ndim
        for ax, window in zip(axis, window_shape):
            depths[ax] += window - 1

        # Ensure that each chunk is big enough to leave at least a size-1 chunk
        # after windowing (this is only really necessary for the last chunk).
        safe_chunks = tuple(
            ensure_minimum_chunksize(d + 1, c) for d, c in zip(depths, x.chunks)
        )
        x = x.rechunk(safe_chunks)

        # result.shape = x_shape_trimmed + window_shape,
        # where x_shape_trimmed is x.shape with every entry
        # reduced by one less than the corresponding window size.
        # trim chunks to match x_shape_trimmed
        newchunks = tuple(
            c[:-1] + (c[-1] - d,) for d, c in zip(depths, x.chunks)
        ) + tuple((window,) for window in window_shape)

        kwargs = dict(
            depth=tuple((0, d) for d in depths),  # Overlap on +ve side only
            boundary="none",
            meta=x._meta,
            new_axis=range(x.ndim, x.ndim + len(axis)),
            chunks=newchunks,
            trim=False,
            window_shape=window_shape,
            axis=axis,
        )
        # map_overlap's signature changed in https://github.com/dask/dask/pull/6165
        if dask_version > Version("2.18.0"):
            return map_overlap(_np_sliding_window_view, x, align_arrays=False, **kwargs)
        else:
            return map_overlap(x, _np_sliding_window_view, **kwargs)
```
