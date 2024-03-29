# pydata__xarray-6548

| **pydata/xarray** | `126051f2bf2ddb7926a7da11b047b852d5ca6b87` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 2757 |
| **Avg pos** | 3.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 6 |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/asv_bench/benchmarks/polyfit.py b/asv_bench/benchmarks/polyfit.py
new file mode 100644
--- /dev/null
+++ b/asv_bench/benchmarks/polyfit.py
@@ -0,0 +1,38 @@
+import numpy as np
+
+import xarray as xr
+
+from . import parameterized, randn, requires_dask
+
+NDEGS = (2, 5, 20)
+NX = (10**2, 10**6)
+
+
+class Polyval:
+    def setup(self, *args, **kwargs):
+        self.xs = {nx: xr.DataArray(randn((nx,)), dims="x", name="x") for nx in NX}
+        self.coeffs = {
+            ndeg: xr.DataArray(
+                randn((ndeg,)), dims="degree", coords={"degree": np.arange(ndeg)}
+            )
+            for ndeg in NDEGS
+        }
+
+    @parameterized(["nx", "ndeg"], [NX, NDEGS])
+    def time_polyval(self, nx, ndeg):
+        x = self.xs[nx]
+        c = self.coeffs[ndeg]
+        xr.polyval(x, c).compute()
+
+    @parameterized(["nx", "ndeg"], [NX, NDEGS])
+    def peakmem_polyval(self, nx, ndeg):
+        x = self.xs[nx]
+        c = self.coeffs[ndeg]
+        xr.polyval(x, c).compute()
+
+
+class PolyvalDask(Polyval):
+    def setup(self, *args, **kwargs):
+        requires_dask()
+        super().setup(*args, **kwargs)
+        self.xs = {k: v.chunk({"x": 10000}) for k, v in self.xs.items()}
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -17,12 +17,15 @@
     Iterable,
     Mapping,
     Sequence,
+    overload,
 )
 
 import numpy as np
 
 from . import dtypes, duck_array_ops, utils
 from .alignment import align, deep_align
+from .common import zeros_like
+from .duck_array_ops import datetime_to_numeric
 from .indexes import Index, filter_indexes_from_coords
 from .merge import merge_attrs, merge_coordinates_without_align
 from .options import OPTIONS, _get_keep_attrs
@@ -1843,36 +1846,100 @@ def where(cond, x, y, keep_attrs=None):
     )
 
 
-def polyval(coord, coeffs, degree_dim="degree"):
+@overload
+def polyval(coord: DataArray, coeffs: DataArray, degree_dim: Hashable) -> DataArray:
+    ...
+
+
+@overload
+def polyval(coord: T_Xarray, coeffs: Dataset, degree_dim: Hashable) -> Dataset:
+    ...
+
+
+@overload
+def polyval(coord: Dataset, coeffs: T_Xarray, degree_dim: Hashable) -> Dataset:
+    ...
+
+
+def polyval(
+    coord: T_Xarray, coeffs: T_Xarray, degree_dim: Hashable = "degree"
+) -> T_Xarray:
     """Evaluate a polynomial at specific values
 
     Parameters
     ----------
-    coord : DataArray
-        The 1D coordinate along which to evaluate the polynomial.
-    coeffs : DataArray
-        Coefficients of the polynomials.
-    degree_dim : str, default: "degree"
+    coord : DataArray or Dataset
+        Values at which to evaluate the polynomial.
+    coeffs : DataArray or Dataset
+        Coefficients of the polynomial.
+    degree_dim : Hashable, default: "degree"
         Name of the polynomial degree dimension in `coeffs`.
 
+    Returns
+    -------
+    DataArray or Dataset
+        Evaluated polynomial.
+
     See Also
     --------
     xarray.DataArray.polyfit
-    numpy.polyval
+    numpy.polynomial.polynomial.polyval
     """
-    from .dataarray import DataArray
-    from .missing import get_clean_interp_index
 
-    x = get_clean_interp_index(coord, coord.name, strict=False)
+    if degree_dim not in coeffs._indexes:
+        raise ValueError(
+            f"Dimension `{degree_dim}` should be a coordinate variable with labels."
+        )
+    if not np.issubdtype(coeffs[degree_dim].dtype, int):
+        raise ValueError(
+            f"Dimension `{degree_dim}` should be of integer dtype. Received {coeffs[degree_dim].dtype} instead."
+        )
+    max_deg = coeffs[degree_dim].max().item()
+    coeffs = coeffs.reindex(
+        {degree_dim: np.arange(max_deg + 1)}, fill_value=0, copy=False
+    )
+    coord = _ensure_numeric(coord)
+
+    # using Horner's method
+    # https://en.wikipedia.org/wiki/Horner%27s_method
+    res = coeffs.isel({degree_dim: max_deg}, drop=True) + zeros_like(coord)
+    for deg in range(max_deg - 1, -1, -1):
+        res *= coord
+        res += coeffs.isel({degree_dim: deg}, drop=True)
 
-    deg_coord = coeffs[degree_dim]
+    return res
 
-    lhs = DataArray(
-        np.vander(x, int(deg_coord.max()) + 1),
-        dims=(coord.name, degree_dim),
-        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
-    )
-    return (lhs * coeffs).sum(degree_dim)
+
+def _ensure_numeric(data: T_Xarray) -> T_Xarray:
+    """Converts all datetime64 variables to float64
+
+    Parameters
+    ----------
+    data : DataArray or Dataset
+        Variables with possible datetime dtypes.
+
+    Returns
+    -------
+    DataArray or Dataset
+        Variables with datetime64 dtypes converted to float64.
+    """
+    from .dataset import Dataset
+
+    def to_floatable(x: DataArray) -> DataArray:
+        if x.dtype.kind in "mM":
+            return x.copy(
+                data=datetime_to_numeric(
+                    x.data,
+                    offset=np.datetime64("1970-01-01"),
+                    datetime_unit="ns",
+                ),
+            )
+        return x
+
+    if isinstance(data, Dataset):
+        return data.map(to_floatable)
+    else:
+        return to_floatable(data)
 
 
 def _calc_idxminmax(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| asv_bench/benchmarks/polyfit.py | 0 | 0 | - | - | -
| xarray/core/computation.py | 20 | 20 | - | 6 | -
| xarray/core/computation.py | 1846 | 1875 | 6 | 6 | 2757


## Problem Statement

```
xr.polyval first arg requires name attribute
### What happened?

I have some polynomial coefficients and want to evaluate them at some values using `xr.polyval`.

As described in the docstring/docu I created a 1D coordinate DataArray and pass it to `xr.polyval` but it raises a KeyError (see example).


### What did you expect to happen?

I expected that the polynomial would be evaluated at the given points.

### Minimal Complete Verifiable Example

\`\`\`Python
import xarray as xr

coeffs = xr.DataArray([1, 2, 3], dims="degree")

# With a "handmade" coordinate it fails:
coord = xr.DataArray([0, 1, 2], dims="x")

xr.polyval(coord, coeffs)
# raises:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "xarray/core/computation.py", line 1847, in polyval
#     x = get_clean_interp_index(coord, coord.name, strict=False)
#   File "xarray/core/missing.py", line 252, in get_clean_interp_index
#     index = arr.get_index(dim)
#   File "xarray/core/common.py", line 404, in get_index
#     raise KeyError(key)
# KeyError: None

# If one adds a name to the coord that is called like the dimension:
coord2 = xr.DataArray([0, 1, 2], dims="x", name="x")

xr.polyval(coord2, coeffs)
# works
\`\`\`


### Relevant log output

_No response_

### Anything else we need to know?

I assume that the "standard" workflow is to obtain the `coord` argument from an existing DataArrays coordinate, where the name would be correctly set already.
However, that is not clear from the description, and also prevents my "manual" workflow.

It could be that the problem will be solved by replacing the coord DataArray argument by an explicit Index in the future.

### Environment

<details>

INSTALLED VERSIONS
------------------
commit: None
python: 3.9.10 (main, Mar 15 2022, 15:56:56) 
[GCC 7.5.0]
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

xarray: 2022.3.0
pandas: 1.4.2
numpy: 1.22.3
scipy: None
netCDF4: 1.5.8
pydap: None
h5netcdf: None
h5py: None
Nio: None
zarr: None
cftime: 1.6.0
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: None
dask: None
distributed: None
matplotlib: 3.5.1
cartopy: 0.20.2
seaborn: None
numbagg: None
fsspec: None
cupy: None
pint: None
sparse: None
setuptools: 58.1.0
pip: 22.0.4
conda: None
pytest: None
IPython: 8.2.0
sphinx: None

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/__init__.py | 1 | 111| 692 | 692 | 692 | 
| 2 | 2 xarray/core/variable.py | 1 | 69| 396 | 1088 | 24809 | 
| 3 | 3 asv_bench/benchmarks/indexing.py | 1 | 59| 738 | 1826 | 26310 | 
| 4 | 4 asv_bench/benchmarks/interp.py | 1 | 18| 150 | 1976 | 26761 | 
| 5 | 5 xarray/core/missing.py | 696 | 763| 566 | 2542 | 32952 | 
| **-> 6 <-** | **6 xarray/core/computation.py** | 1846 | 1875| 215 | 2757 | 50065 | 
| 7 | 6 xarray/core/missing.py | 777 | 790| 182 | 2939 | 50065 | 
| 8 | 7 xarray/core/coordinates.py | 1 | 31| 163 | 3102 | 52877 | 
| 9 | 8 xarray/core/dataarray.py | 1 | 80| 479 | 3581 | 95399 | 
| 10 | 9 xarray/core/dataset.py | 1 | 132| 691 | 4272 | 164722 | 
| 11 | 9 xarray/core/missing.py | 766 | 774| 110 | 4382 | 164722 | 
| 12 | 9 xarray/core/dataarray.py | 1686 | 1814| 1365 | 5747 | 164722 | 
| 13 | 10 asv_bench/benchmarks/dataarray_missing.py | 1 | 16| 114 | 5861 | 165223 | 
| 14 | 11 xarray/core/indexing.py | 1 | 26| 173 | 6034 | 177452 | 
| 15 | 12 xarray/core/common.py | 1121 | 1182| 636 | 6670 | 193402 | 
| 16 | 12 xarray/core/common.py | 1 | 47| 243 | 6913 | 193402 | 
| 17 | 13 asv_bench/benchmarks/rolling.py | 1 | 17| 113 | 7026 | 194677 | 
| 18 | 13 xarray/core/missing.py | 1 | 18| 139 | 7165 | 194677 | 
| 19 | 13 asv_bench/benchmarks/indexing.py | 62 | 76| 151 | 7316 | 194677 | 
| 20 | 14 asv_bench/benchmarks/pandas.py | 1 | 27| 167 | 7483 | 194844 | 
| 21 | 15 xarray/core/nputils.py | 163 | 175| 150 | 7633 | 196706 | 


## Missing Patch Files

 * 1: asv_bench/benchmarks/polyfit.py
 * 2: xarray/core/computation.py

### Hint

```
Actually, I just realized that the second version also does not work since it uses the index of the `coord` argument and not its values. I guess that was meant by "The 1D coordinate along which to evaluate the polynomial".

Would you be open to a PR that allows any DataArray as `coord` argument and evaluates the polynomial at its values? Maybe that would break backwards compatibility though.
> Would you be open to a PR that allows any DataArray as coord argument and evaluates the polynomial at its values? 

I think yes. Note https://github.com/pydata/xarray/issues/4375 for the inverse problem.
```

## Patch

```diff
diff --git a/asv_bench/benchmarks/polyfit.py b/asv_bench/benchmarks/polyfit.py
new file mode 100644
--- /dev/null
+++ b/asv_bench/benchmarks/polyfit.py
@@ -0,0 +1,38 @@
+import numpy as np
+
+import xarray as xr
+
+from . import parameterized, randn, requires_dask
+
+NDEGS = (2, 5, 20)
+NX = (10**2, 10**6)
+
+
+class Polyval:
+    def setup(self, *args, **kwargs):
+        self.xs = {nx: xr.DataArray(randn((nx,)), dims="x", name="x") for nx in NX}
+        self.coeffs = {
+            ndeg: xr.DataArray(
+                randn((ndeg,)), dims="degree", coords={"degree": np.arange(ndeg)}
+            )
+            for ndeg in NDEGS
+        }
+
+    @parameterized(["nx", "ndeg"], [NX, NDEGS])
+    def time_polyval(self, nx, ndeg):
+        x = self.xs[nx]
+        c = self.coeffs[ndeg]
+        xr.polyval(x, c).compute()
+
+    @parameterized(["nx", "ndeg"], [NX, NDEGS])
+    def peakmem_polyval(self, nx, ndeg):
+        x = self.xs[nx]
+        c = self.coeffs[ndeg]
+        xr.polyval(x, c).compute()
+
+
+class PolyvalDask(Polyval):
+    def setup(self, *args, **kwargs):
+        requires_dask()
+        super().setup(*args, **kwargs)
+        self.xs = {k: v.chunk({"x": 10000}) for k, v in self.xs.items()}
diff --git a/xarray/core/computation.py b/xarray/core/computation.py
--- a/xarray/core/computation.py
+++ b/xarray/core/computation.py
@@ -17,12 +17,15 @@
     Iterable,
     Mapping,
     Sequence,
+    overload,
 )
 
 import numpy as np
 
 from . import dtypes, duck_array_ops, utils
 from .alignment import align, deep_align
+from .common import zeros_like
+from .duck_array_ops import datetime_to_numeric
 from .indexes import Index, filter_indexes_from_coords
 from .merge import merge_attrs, merge_coordinates_without_align
 from .options import OPTIONS, _get_keep_attrs
@@ -1843,36 +1846,100 @@ def where(cond, x, y, keep_attrs=None):
     )
 
 
-def polyval(coord, coeffs, degree_dim="degree"):
+@overload
+def polyval(coord: DataArray, coeffs: DataArray, degree_dim: Hashable) -> DataArray:
+    ...
+
+
+@overload
+def polyval(coord: T_Xarray, coeffs: Dataset, degree_dim: Hashable) -> Dataset:
+    ...
+
+
+@overload
+def polyval(coord: Dataset, coeffs: T_Xarray, degree_dim: Hashable) -> Dataset:
+    ...
+
+
+def polyval(
+    coord: T_Xarray, coeffs: T_Xarray, degree_dim: Hashable = "degree"
+) -> T_Xarray:
     """Evaluate a polynomial at specific values
 
     Parameters
     ----------
-    coord : DataArray
-        The 1D coordinate along which to evaluate the polynomial.
-    coeffs : DataArray
-        Coefficients of the polynomials.
-    degree_dim : str, default: "degree"
+    coord : DataArray or Dataset
+        Values at which to evaluate the polynomial.
+    coeffs : DataArray or Dataset
+        Coefficients of the polynomial.
+    degree_dim : Hashable, default: "degree"
         Name of the polynomial degree dimension in `coeffs`.
 
+    Returns
+    -------
+    DataArray or Dataset
+        Evaluated polynomial.
+
     See Also
     --------
     xarray.DataArray.polyfit
-    numpy.polyval
+    numpy.polynomial.polynomial.polyval
     """
-    from .dataarray import DataArray
-    from .missing import get_clean_interp_index
 
-    x = get_clean_interp_index(coord, coord.name, strict=False)
+    if degree_dim not in coeffs._indexes:
+        raise ValueError(
+            f"Dimension `{degree_dim}` should be a coordinate variable with labels."
+        )
+    if not np.issubdtype(coeffs[degree_dim].dtype, int):
+        raise ValueError(
+            f"Dimension `{degree_dim}` should be of integer dtype. Received {coeffs[degree_dim].dtype} instead."
+        )
+    max_deg = coeffs[degree_dim].max().item()
+    coeffs = coeffs.reindex(
+        {degree_dim: np.arange(max_deg + 1)}, fill_value=0, copy=False
+    )
+    coord = _ensure_numeric(coord)
+
+    # using Horner's method
+    # https://en.wikipedia.org/wiki/Horner%27s_method
+    res = coeffs.isel({degree_dim: max_deg}, drop=True) + zeros_like(coord)
+    for deg in range(max_deg - 1, -1, -1):
+        res *= coord
+        res += coeffs.isel({degree_dim: deg}, drop=True)
 
-    deg_coord = coeffs[degree_dim]
+    return res
 
-    lhs = DataArray(
-        np.vander(x, int(deg_coord.max()) + 1),
-        dims=(coord.name, degree_dim),
-        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
-    )
-    return (lhs * coeffs).sum(degree_dim)
+
+def _ensure_numeric(data: T_Xarray) -> T_Xarray:
+    """Converts all datetime64 variables to float64
+
+    Parameters
+    ----------
+    data : DataArray or Dataset
+        Variables with possible datetime dtypes.
+
+    Returns
+    -------
+    DataArray or Dataset
+        Variables with datetime64 dtypes converted to float64.
+    """
+    from .dataset import Dataset
+
+    def to_floatable(x: DataArray) -> DataArray:
+        if x.dtype.kind in "mM":
+            return x.copy(
+                data=datetime_to_numeric(
+                    x.data,
+                    offset=np.datetime64("1970-01-01"),
+                    datetime_unit="ns",
+                ),
+            )
+        return x
+
+    if isinstance(data, Dataset):
+        return data.map(to_floatable)
+    else:
+        return to_floatable(data)
 
 
 def _calc_idxminmax(

```

## Test Patch

```diff
diff --git a/xarray/tests/test_computation.py b/xarray/tests/test_computation.py
--- a/xarray/tests/test_computation.py
+++ b/xarray/tests/test_computation.py
@@ -1933,37 +1933,100 @@ def test_where_attrs() -> None:
     assert actual.attrs == {}
 
 
-@pytest.mark.parametrize("use_dask", [True, False])
-@pytest.mark.parametrize("use_datetime", [True, False])
-def test_polyval(use_dask, use_datetime) -> None:
-    if use_dask and not has_dask:
-        pytest.skip("requires dask")
-
-    if use_datetime:
-        xcoord = xr.DataArray(
-            pd.date_range("2000-01-01", freq="D", periods=10), dims=("x",), name="x"
-        )
-        x = xr.core.missing.get_clean_interp_index(xcoord, "x")
-    else:
-        x = np.arange(10)
-        xcoord = xr.DataArray(x, dims=("x",), name="x")
-
-    da = xr.DataArray(
-        np.stack((1.0 + x + 2.0 * x**2, 1.0 + 2.0 * x + 3.0 * x**2)),
-        dims=("d", "x"),
-        coords={"x": xcoord, "d": [0, 1]},
-    )
-    coeffs = xr.DataArray(
-        [[2, 1, 1], [3, 2, 1]],
-        dims=("d", "degree"),
-        coords={"d": [0, 1], "degree": [2, 1, 0]},
-    )
+@pytest.mark.parametrize("use_dask", [False, True])
+@pytest.mark.parametrize(
+    ["x", "coeffs", "expected"],
+    [
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.DataArray([2, 3, 4], dims="degree", coords={"degree": [0, 1, 2]}),
+            xr.DataArray([9, 2 + 6 + 16, 2 + 9 + 36], dims="x"),
+            id="simple",
+        ),
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.DataArray(
+                [[0, 1], [0, 1]], dims=("y", "degree"), coords={"degree": [0, 1]}
+            ),
+            xr.DataArray([[1, 2, 3], [1, 2, 3]], dims=("y", "x")),
+            id="broadcast-x",
+        ),
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.DataArray(
+                [[0, 1], [1, 0], [1, 1]],
+                dims=("x", "degree"),
+                coords={"degree": [0, 1]},
+            ),
+            xr.DataArray([1, 1, 1 + 3], dims="x"),
+            id="shared-dim",
+        ),
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.DataArray([1, 0, 0], dims="degree", coords={"degree": [2, 1, 0]}),
+            xr.DataArray([1, 2**2, 3**2], dims="x"),
+            id="reordered-index",
+        ),
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.DataArray([5], dims="degree", coords={"degree": [3]}),
+            xr.DataArray([5, 5 * 2**3, 5 * 3**3], dims="x"),
+            id="sparse-index",
+        ),
+        pytest.param(
+            xr.DataArray([1, 2, 3], dims="x"),
+            xr.Dataset(
+                {"a": ("degree", [0, 1]), "b": ("degree", [1, 0])},
+                coords={"degree": [0, 1]},
+            ),
+            xr.Dataset({"a": ("x", [1, 2, 3]), "b": ("x", [1, 1, 1])}),
+            id="array-dataset",
+        ),
+        pytest.param(
+            xr.Dataset({"a": ("x", [1, 2, 3]), "b": ("x", [2, 3, 4])}),
+            xr.DataArray([1, 1], dims="degree", coords={"degree": [0, 1]}),
+            xr.Dataset({"a": ("x", [2, 3, 4]), "b": ("x", [3, 4, 5])}),
+            id="dataset-array",
+        ),
+        pytest.param(
+            xr.Dataset({"a": ("x", [1, 2, 3]), "b": ("y", [2, 3, 4])}),
+            xr.Dataset(
+                {"a": ("degree", [0, 1]), "b": ("degree", [1, 1])},
+                coords={"degree": [0, 1]},
+            ),
+            xr.Dataset({"a": ("x", [1, 2, 3]), "b": ("y", [3, 4, 5])}),
+            id="dataset-dataset",
+        ),
+        pytest.param(
+            xr.DataArray(pd.date_range("1970-01-01", freq="s", periods=3), dims="x"),
+            xr.DataArray([0, 1], dims="degree", coords={"degree": [0, 1]}),
+            xr.DataArray(
+                [0, 1e9, 2e9],
+                dims="x",
+                coords={"x": pd.date_range("1970-01-01", freq="s", periods=3)},
+            ),
+            id="datetime",
+        ),
+    ],
+)
+def test_polyval(use_dask, x, coeffs, expected) -> None:
     if use_dask:
-        coeffs = coeffs.chunk({"d": 2})
+        if not has_dask:
+            pytest.skip("requires dask")
+        coeffs = coeffs.chunk({"degree": 2})
+        x = x.chunk({"x": 2})
+    with raise_if_dask_computes():
+        actual = xr.polyval(x, coeffs)
+    xr.testing.assert_allclose(actual, expected)
 
-    da_pv = xr.polyval(da.x, coeffs)
 
-    xr.testing.assert_allclose(da, da_pv.T)
+def test_polyval_degree_dim_checks():
+    x = (xr.DataArray([1, 2, 3], dims="x"),)
+    coeffs = xr.DataArray([2, 3, 4], dims="degree", coords={"degree": [0, 1, 2]})
+    with pytest.raises(ValueError):
+        xr.polyval(x, coeffs.drop_vars("degree"))
+    with pytest.raises(ValueError):
+        xr.polyval(x, coeffs.assign_coords(degree=coeffs.degree.astype(float)))
 
 
 @pytest.mark.parametrize("use_dask", [False, True])

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
### 2 - xarray/core/variable.py:

Start line: 1, End line: 69

```python
from __future__ import annotations

import copy
import itertools
import numbers
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Hashable, Literal, Mapping, Sequence

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
    dask_array_type,
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
    (
        indexing.ExplicitlyIndexed,
        pd.Index,
    )
    + dask_array_type
    + cupy_array_type
)
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)

if TYPE_CHECKING:
    from .types import T_Variable


class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name."""

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?
```
### 3 - asv_bench/benchmarks/indexing.py:

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
### 4 - asv_bench/benchmarks/interp.py:

Start line: 1, End line: 18

```python
import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 1500
ny = 1000
nt = 500

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))

new_x_short = np.linspace(0.3 * nx, 0.7 * nx, 100)
new_x_long = np.linspace(0.3 * nx, 0.7 * nx, 500)
new_y_long = np.linspace(0.1, 0.9, 500)
```
### 5 - xarray/core/missing.py:

Start line: 696, End line: 763

```python
def interp_func(var, x, new_x, method, kwargs):
    # ... other code

    if is_duck_dask_array(var):
        import dask.array as da

        ndim = var.ndim
        nconst = ndim - len(x)

        out_ind = list(range(nconst)) + list(range(ndim, ndim + new_x[0].ndim))

        # blockwise args format
        x_arginds = [[_x, (nconst + index,)] for index, _x in enumerate(x)]
        x_arginds = [item for pair in x_arginds for item in pair]
        new_x_arginds = [
            [_x, [ndim + index for index in range(_x.ndim)]] for _x in new_x
        ]
        new_x_arginds = [item for pair in new_x_arginds for item in pair]

        args = (
            var,
            range(ndim),
            *x_arginds,
            *new_x_arginds,
        )

        _, rechunked = da.unify_chunks(*args)

        args = tuple(elem for pair in zip(rechunked, args[1::2]) for elem in pair)

        new_x = rechunked[1 + (len(rechunked) - 1) // 2 :]

        new_axes = {
            ndim + i: new_x[0].chunks[i]
            if new_x[0].chunks is not None
            else new_x[0].shape[i]
            for i in range(new_x[0].ndim)
        }

        # if useful, re-use localize for each chunk of new_x
        localize = (method in ["linear", "nearest"]) and (new_x[0].chunks is not None)

        # scipy.interpolate.interp1d always forces to float.
        # Use the same check for blockwise as well:
        if not issubclass(var.dtype.type, np.inexact):
            dtype = np.float_
        else:
            dtype = var.dtype

        if dask_version < Version("2020.12"):
            # Using meta and dtype at the same time doesn't work.
            # Remove this whenever the minimum requirement for dask is 2020.12:
            meta = None
        else:
            meta = var._meta

        return da.blockwise(
            _dask_aware_interpnd,
            out_ind,
            *args,
            interp_func=func,
            interp_kwargs=kwargs,
            localize=localize,
            concatenate=True,
            dtype=dtype,
            new_axes=new_axes,
            meta=meta,
            align_arrays=False,
        )

    return _interpnd(var, x, new_x, func, kwargs)
```
### 6 - xarray/core/computation.py:

Start line: 1846, End line: 1875

```python
def polyval(coord, coeffs, degree_dim="degree"):
    """Evaluate a polynomial at specific values

    Parameters
    ----------
    coord : DataArray
        The 1D coordinate along which to evaluate the polynomial.
    coeffs : DataArray
        Coefficients of the polynomials.
    degree_dim : str, default: "degree"
        Name of the polynomial degree dimension in `coeffs`.

    See Also
    --------
    xarray.DataArray.polyfit
    numpy.polyval
    """
    from .dataarray import DataArray
    from .missing import get_clean_interp_index

    x = get_clean_interp_index(coord, coord.name, strict=False)

    deg_coord = coeffs[degree_dim]

    lhs = DataArray(
        np.vander(x, int(deg_coord.max()) + 1),
        dims=(coord.name, degree_dim),
        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
    )
    return (lhs * coeffs).sum(degree_dim)
```
### 7 - xarray/core/missing.py:

Start line: 777, End line: 790

```python
def _interpnd(var, x, new_x, func, kwargs):
    x, new_x = _floatize_x(x, new_x)

    if len(x) == 1:
        return _interp1d(var, x, new_x, func, kwargs)

    # move the interpolation axes to the start position
    var = var.transpose(range(-len(x), var.ndim - len(x)))
    # stack new_x to 1 vector, with reshape
    xi = np.stack([x1.values.ravel() for x1 in new_x], axis=-1)
    rslt = func(x, var, xi, **kwargs)
    # move back the interpolation axes to the last position
    rslt = rslt.transpose(range(-rslt.ndim + 1, 1))
    return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)
```
### 8 - xarray/core/coordinates.py:

Start line: 1, End line: 31

```python
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from . import formatting
from .indexes import Index, Indexes, assert_no_index_corrupted
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
### 9 - xarray/core/dataarray.py:

Start line: 1, End line: 80

```python
from __future__ import annotations

import datetime
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    cast,
)

import numpy as np
import pandas as pd

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex
from ..plot.plot import _PlotMethods
from ..plot.utils import _get_units_from_attrs
from . import (
    alignment,
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
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
### 10 - xarray/core/dataset.py:

Start line: 1, End line: 132

```python
from __future__ import annotations

import copy
import datetime
import inspect
import itertools
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

import xarray as xr

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
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
    broadcast_variables,
    calculate_dimensions,
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
